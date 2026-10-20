"""
Change Detection's "Change over time" section: the event's index over
time inside the area the comparison flagged, against nearby unchanged
land (see core.timeseries).
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Any, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from rasterio.enums import Resampling

from app.components.scene_loader import load_scene_cached
from app.components.scene_selector import _search_sentinel2_cached
from core.change import _scl_valid_mask, align_to_reference
from core.findings import findings_markdown
from core.indices import compute_indices
from core.timeseries import (
    RECOVERY_PRESETS,
    SamplePoint,
    add_months,
    pick_scenes,
    recovery_findings,
    recovery_summary,
    sample_point,
)
from core.utils import resample_bands, scale_bands

# Each scene is read small: the page needs area medians, not detail.
SCENE_MAX_DIM = 256
MAX_SCENES = 40
AFFECTED_COLOR = "#eb6834"
CONTROL_COLOR = "#2a78d6"


def _scene_point(item: Any, aoi_wkt: str, index_name: str, reference, affected, control) -> SamplePoint:
    """Load one scene, compute the index on the reference grid and summarise both areas."""
    bands, _, _ = load_scene_cached(item, aoi_wkt, item.id, SCENE_MAX_DIM)
    bands = scale_bands(resample_bands(bands))
    index = compute_indices(bands)[index_name]
    valid = _scl_valid_mask(bands["scl"].values)
    index = align_to_reference(np.where(valid, index, np.nan), bands["nir"], reference)
    valid = align_to_reference(valid.astype("float32"), bands["nir"], reference, Resampling.nearest) > 0.5
    return sample_point(index, valid, affected, control, item.datetime.date())


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _scene_point_cached(
    _item: Any, item_id: str, aoi_wkt: str, index_name: str, _reference, _affected, _control, grid_key: str
) -> SamplePoint:
    """_scene_point, cached on the scene id, index and grid (with the change classes)."""
    return _scene_point(_item, aoi_wkt, index_name, _reference, _affected, _control)


def _sample_points(
    scenes: List[Any], aoi_wkt: str, index_name: str, reference, affected, control, grid_key: str
) -> Tuple[List[SamplePoint], List[str]]:
    """Sample points for each scene, plus the scenes that failed to load."""
    points, failed = [], []
    progress = st.progress(0.0, text="Loading scenes...")
    for i, item in enumerate(scenes):
        progress.progress(i / len(scenes), text=f"Loading scene {i + 1} of {len(scenes)} ({item.datetime.date()})")
        try:
            points.append(_scene_point_cached(item, item.id, aoi_wkt, index_name, reference, affected, control, grid_key))
        except Exception as exc:  # one bad scene shouldn't sink the series
            failed.append(f"{item.datetime.date()} ({type(exc).__name__})")
    progress.empty()
    return points, failed


def _chart(points: List[SamplePoint], preset, index_name: str, before: date, after: date) -> alt.Chart:
    rows = []
    for p in points:
        if p.gap is None:
            continue
        rows.append({"Date": pd.Timestamp(p.when), "Area": preset.area_name.removeprefix("the ").capitalize(), "Median": p.affected})
        rows.append({"Date": pd.Timestamp(p.when), "Area": preset.control_name.capitalize(), "Median": p.control})
    df = pd.DataFrame(rows)
    names = list(dict.fromkeys(df["Area"]))
    color = alt.Color("Area:N", scale=alt.Scale(domain=names, range=[AFFECTED_COLOR, CONTROL_COLOR]),
                      legend=alt.Legend(orient="top", title=None))
    hover = alt.selection_point(fields=["Date"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(df).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("Median:Q", title=f"Median {index_name.upper()}", scale=alt.Scale(zero=False)),
        color=color,
    )
    lines = base.mark_line(strokeWidth=2)
    dots = base.mark_point(filled=True, size=50).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0.6)),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Area:N"), alt.Tooltip("Median:Q", format=".3f")],
    ).add_params(hover)
    event = alt.Chart(pd.DataFrame({"start": [pd.Timestamp(before)], "end": [pd.Timestamp(after)]})).mark_rect(
        opacity=0.15, color="gray"
    ).encode(x="start:T", x2="end:T")
    return (event + lines + dots).properties(height=360)


def recovery_display(
    preset_key: str, index_name: str, codes: np.ndarray, reference, before_item: Any, after_item: Any,
    aoi_wkt: str, key_prefix: str,
) -> None:
    """The change-over-time section for one comparison (wildfire, flood or logging presets)."""
    preset = RECOVERY_PRESETS[preset_key]
    before, after = before_item.datetime.date(), after_item.datetime.date()
    col1, col2, col3 = st.columns(3)
    months_before = col1.number_input("Months before", 1, 36, preset.months_before, key=f"{key_prefix}_ts_months_before")
    months_after = col2.number_input("Months after", 1, 60, preset.months_after, key=f"{key_prefix}_ts_months_after")
    step = col3.selectbox("One scene every", [1, 2, 3, 6], index=[1, 2, 3, 6].index(preset.step_months),
                          format_func=lambda m: f"{m} month{'s' if m > 1 else ''}", key=f"{key_prefix}_ts_step")
    start = add_months(before, -int(months_before))
    end = min(add_months(after, int(months_after)), date.today())

    try:
        with st.spinner("Searching Sentinel-2 scenes..."):
            items = _search_sentinel2_cached(aoi_wkt, start, end, 60)
    except Exception as exc:  # network/data problems shouldn't break the page
        st.caption(f"Scenes couldn't be searched on Planetary Computer ({type(exc).__name__}), so this is skipped.")
        return
    scenes = pick_scenes(items, start, step, keep=[before_item, after_item])
    if len(scenes) > MAX_SCENES:
        st.warning(f"That window needs {len(scenes)} scenes; showing the first {MAX_SCENES}. Sample less often to cover it all.")
        scenes = scenes[:MAX_SCENES]

    affected = np.isin(codes, preset.affected_codes)
    control = codes == preset.control_code
    if not affected.any():
        st.info(f"The comparison found no {preset.area_name.removeprefix('the ')}, so there's nothing to track.")
        return
    grid_key = f"{reference.rio.crs}|{tuple(reference.rio.transform())}|{reference.shape}|{hashlib.md5(codes.tobytes()).hexdigest()}"
    points, failed = _sample_points(scenes, aoi_wkt, index_name, reference, affected, control, grid_key)
    clear = [p for p in points if p.gap is not None]

    summary = recovery_summary(points, before, after)
    with st.container(border=True):
        st.markdown("**Over time**")
        if summary is None:
            st.markdown("Not enough clear scenes around the event dates to measure change and recovery.")
        else:
            lines = findings_markdown(recovery_findings(summary, preset, index_name))
            st.markdown("\n".join(part for part in (lines["summary"], lines["caveats"]) if part))

    if summary is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Event change ({index_name.upper()})", f"{summary.event_shift:+.2f}",
                  help=f"Median {index_name.upper()} of {preset.area_name}, after minus before.")
        m2.metric("Recovery so far", "n/a" if summary.recovery_pct is None else f"{max(summary.recovery_pct, 0):.0f}%",
                  help=(f"Share of the event's gap to {preset.control_name} that has closed by the latest clear "
                        "scene. Measuring against unchanged land cancels most of the seasonal swing."))
        m3.metric("Latest clear scene", str(summary.latest.when) if summary.latest else "none yet")

    if clear:
        st.altair_chart(_chart(clear, preset, index_name, before, after), width="stretch")
        st.caption(
            f"Median {index_name.upper()} of the clear pixels in each area, one scene every {step} "
            f"month{'s' if step > 1 else ''} (least cloudy), plus the Change Detection pair. "
            "The shaded band spans the before and after dates."
        )
    skipped = [p for p in points if p.gap is None]
    notes = []
    if skipped:
        notes.append(f"{len(skipped)} scene(s) skipped because less than 30% of {preset.area_name} was clear")
    if failed:
        notes.append(f"{len(failed)} failed to load: {', '.join(failed)}")
    if notes:
        st.caption("; ".join(notes) + ".")

    with st.expander("Data table"):
        st.dataframe(pd.DataFrame([{
            "Date": p.when,
            f"{preset.area_name.removeprefix('the ').capitalize()}": None if p.affected is None else round(p.affected, 3),
            preset.control_name.capitalize(): None if p.control is None else round(p.control, 3),
            "Clear share of affected area (%)": round(100 * p.clear_share),
        } for p in points]), hide_index=True, width="stretch")

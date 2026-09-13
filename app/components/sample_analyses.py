"""
Curated "sample analyses" for the Change Detection page: real, well-known
before/after events with AOI + date ranges pre-configured, so a visitor can
see a dramatic result in one click without first understanding AOI
selection, date ranges, or event presets.

Scene selection within each sample's date range is automatic (lowest cloud
cover available) -- asking a first-time visitor to also pick a scene
manually would defeat the point of a "one click" sample.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple
import math

import streamlit as st
from shapely.geometry import box, Polygon

# Reusing the existing cached STAC search rather than duplicating a second
# cache for the same underlying query.
from app.components.scene_selector import _search_sentinel2_cached


# Progressive relaxation steps tried in order until a search returns
# results. Tried first at the original date window, then -- if still
# nothing -- retried at a widened window. Event-driven searches (a flood's
# "after" date, in particular) are disproportionately likely to hit cloudy
# weather right when you most want a clear scene, so failing after a
# single strict search wastes what's often a recoverable situation.
CLOUD_COVER_STEPS = (40, 60, 80, 100)
DATE_WINDOW_EXPANSIONS_DAYS = (0, 30)


def _search_with_fallback(
    aoi_wkt: str, start: date, end: date
) -> Tuple[List[Any], Optional[int], Optional[int]]:
    """
    Search with progressively relaxed cloud-cover thresholds; if still
    empty, widen the date window symmetrically and retry the same
    relaxation sequence.

    Returns
    -------
    tuple(list, int or None, int or None)
        (items, cloud_cover_used, expansion_days_used). The latter two are
        None only when items is empty (nothing found at any relaxation
        level tried).
    """
    for expand_days in DATE_WINDOW_EXPANSIONS_DAYS:
        window_start = start - timedelta(days=expand_days)
        window_end = end + timedelta(days=expand_days)

        for cloud_pct in CLOUD_COVER_STEPS:
            items = _search_sentinel2_cached(aoi_wkt, window_start, window_end, cloud_pct)
            if items:
                return items, cloud_pct, expand_days

    return [], None, None


SAMPLE_ANALYSES: Dict[str, Dict[str, Any]] = {
    "camp_fire_2018": {
        "label": "Camp Fire — Paradise, CA (Nov 2018)",
        "preset": "wildfire",
        "lat": 39.7596, "lon": -121.6219, "width_km": 25, "height_km": 25,
        "before_range": (date(2018, 9, 1), date(2018, 10, 31)),
        "after_range": (date(2018, 12, 1), date(2019, 1, 31)),
        "description": (
            "The Camp Fire ignited on November 8, 2018 and destroyed the "
            "town of Paradise, California within hours -- at the time, the "
            "deadliest and most destructive wildfire in California history."
        ),
    },
    "kangaroo_island_2020": {
        "label": "Kangaroo Island Bushfires, Australia (Jan 2020)",
        "preset": "wildfire",
        "lat": -35.7751, "lon": 137.2131, "width_km": 60, "height_km": 60,
        "before_range": (date(2019, 11, 1), date(2019, 11, 30)),
        "after_range": (date(2020, 2, 1), date(2020, 2, 29)),
        "description": (
            "Bushfires during Australia's 2019-2020 \"Black Summer\" burned "
            "roughly half of Kangaroo Island, including large portions of "
            "Flinders Chase National Park."
        ),
    },
    "harvey_houston_2017": {
        "label": "Hurricane Harvey Flooding — Houston, TX (Aug 2017)",
        "preset": "flood",
        "lat": 29.7604, "lon": -95.3698, "width_km": 40, "height_km": 40,
        "before_range": (date(2017, 7, 1), date(2017, 7, 31)),
        "after_range": (date(2017, 8, 29), date(2017, 9, 15)),
        "description": (
            "Hurricane Harvey stalled over southeast Texas in late August "
            "2017, dropping historic rainfall and causing catastrophic "
            "flooding across the Houston metro area."
        ),
    },
    "rondonia_deforestation": {
        "label": "Amazon Deforestation — Rondônia, Brazil (2019-2023)",
        "preset": "logging",
        "lat": -10.83, "lon": -62.90, "width_km": 40, "height_km": 40,
        "before_range": (date(2019, 6, 1), date(2019, 7, 31)),
        "after_range": (date(2023, 6, 1), date(2023, 7, 31)),
        "description": (
            "Rondônia sits on Brazil's \"arc of deforestation\" -- one of "
            "the most actively cleared regions of the Amazon, driven "
            "largely by cattle ranching and agricultural expansion. "
            "Comparing dry-season imagery a few years apart makes gradual "
            "clearing visible in a way a single before/after pair usually "
            "can't."
        ),
    },
    "bangladesh_monsoon_flood": {
        "label": "Monsoon Flooding — Ganges-Brahmaputra Delta, Bangladesh",
        "preset": "flood",
        "lat": 23.6850, "lon": 90.3563, "width_km": 60, "height_km": 60,
        "before_range": (date(2022, 3, 1), date(2022, 4, 30)),
        "after_range": (date(2022, 7, 15), date(2022, 8, 31)),
        "description": (
            "Bangladesh's low-lying delta floods seasonally every year "
            "during the June-September monsoon. This shows a representative "
            "dry-season-to-monsoon transition rather than one specific "
            "flood event."
        ),
    },
}


def _build_aoi(lat: float, lon: float, width_km: float, height_km: float) -> Polygon:
    dlat = (height_km / 2) / 111.0
    dlon = (width_km / 2) / (111.0 * abs(math.cos(math.radians(lat))))
    return box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _run_sample_analysis(key: str) -> None:
    """
    Populate session_state for a sample analysis: AOI, event preset,
    before/after date ranges, and an automatically-selected (lowest cloud
    cover) scene for each date range.

    Caller is responsible for calling st.rerun() immediately afterward, so
    that the date_input / selectbox widgets on the page -- which read their
    initial value from session_state[key] if already present -- pick up
    these new values on the next run instead of their hardcoded defaults.
    """
    sample = SAMPLE_ANALYSES[key]

    aoi = _build_aoi(sample["lat"], sample["lon"], sample["width_km"], sample["height_km"])
    st.session_state["aoi"] = aoi
    st.session_state["aoi_label"] = sample["label"]
    st.session_state["change_preset"] = sample["preset"]

    before_start, before_end = sample["before_range"]
    after_start, after_end = sample["after_range"]
    st.session_state["before_start_date"] = before_start
    st.session_state["before_end_date"] = before_end
    st.session_state["after_start_date"] = after_start
    st.session_state["after_end_date"] = after_end

    with st.spinner("Finding before/after scenes for this sample..."):
        before_items, before_cloud, before_expand = _search_with_fallback(
            aoi.wkt, before_start, before_end
        )
        after_items, after_cloud, after_expand = _search_with_fallback(
            aoi.wkt, after_start, after_end
        )

    st.session_state["before_stac_items"] = before_items
    st.session_state["after_stac_items"] = after_items

    if not before_items or not after_items:
        missing = []
        if not before_items:
            missing.append("before")
        if not after_items:
            missing.append("after")
        st.session_state["sample_analysis_error"] = (
            f"Couldn't find any scenes for the {', '.join(missing)} window(s), "
            "even after relaxing the cloud-cover filter and widening the date "
            "range. Try picking scenes manually below with a looser cloud "
            "cover slider, or run the sample again later."
        )
        st.session_state.pop("before_stac_item", None)
        st.session_state.pop("after_stac_item", None)
        return

    st.session_state["before_stac_item"] = min(
        before_items, key=lambda it: it.properties.get("eo:cloud_cover", 100)
    )
    st.session_state["after_stac_item"] = min(
        after_items, key=lambda it: it.properties.get("eo:cloud_cover", 100)
    )
    st.session_state["sample_analysis_error"] = None

    # Let the user know if either side needed relaxed cloud cover or a
    # widened window -- worth knowing, since a heavily relaxed cloud filter
    # can mean a noticeably cloudier scene than the default 40% would give.
    notes = []
    if before_cloud > 40 or before_expand > 0:
        notes.append(
            f"Before: used cloud cover < {before_cloud}%"
            + (f", widened by {before_expand} days" if before_expand else "")
        )
    if after_cloud > 40 or after_expand > 0:
        notes.append(
            f"After: used cloud cover < {after_cloud}%"
            + (f", widened by {after_expand} days" if after_expand else "")
        )
    st.session_state["sample_analysis_relaxation_note"] = "; ".join(notes) if notes else None


def sample_analysis_picker() -> None:
    """
    Render the "Try a Sample Analysis" section: a picker plus a button that
    runs _run_sample_analysis and reruns the page so every downstream
    widget (event type, date ranges, scene pickers, comparison) reflects
    the choice immediately.
    """
    st.subheader("Try a Sample Analysis")
    st.caption(
        "Real, well-documented before/after events, pre-configured end to "
        "end -- AOI, dates, and event type all set automatically."
    )

    options = ["-- Select a sample --"] + list(SAMPLE_ANALYSES.keys())
    choice = st.selectbox(
        "Sample event",
        options=options,
        format_func=lambda k: SAMPLE_ANALYSES[k]["label"] if k in SAMPLE_ANALYSES else k,
        key="sample_analysis_choice",
    )

    if choice in SAMPLE_ANALYSES:
        st.caption(SAMPLE_ANALYSES[choice]["description"])
        if st.button("Run this analysis", key="run_sample_analysis"):
            _run_sample_analysis(choice)
            st.rerun()

    error = st.session_state.get("sample_analysis_error")
    if error:
        st.warning(error)

    note = st.session_state.get("sample_analysis_relaxation_note")
    if note:
        st.info(f"Note: the strict search was empty, so this used a fallback. {note}")

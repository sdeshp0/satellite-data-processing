"""
Scene selection widgets for Streamlit.

- scene_selector: triggers a STAC search and stores results in session state.
- scene_picker: renders search results as a table with thumbnail previews,
  date, and cloud cover, letting the user pick a row instead of parsing a
  concatenated string in a dropdown.
"""

from __future__ import annotations

from datetime import date
from typing import Any, List, Optional

import pandas as pd
import streamlit as st
from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon

from core.stac import search_sentinel2
from core.change import day_of_year_distance, sun_elevation_diff


@st.cache_data(show_spinner=False, ttl=3600)
def _search_sentinel2_cached(
    aoi_wkt: str, start_date: date, end_date: date, max_cloud_cover: int = 40
):
    """
    Cached wrapper around search_sentinel2, keyed on AOI WKT + dates +
    max_cloud_cover so unrelated widget interactions elsewhere on the page
    don't retrigger a fresh STAC search.
    """
    aoi = shapely_wkt.loads(aoi_wkt)
    return search_sentinel2(aoi, start_date, end_date, max_cloud_cover=max_cloud_cover)


def scene_selector(
    aoi: Polygon,
    start_date: date,
    end_date: date,
    key_prefix: str = "single",
) -> None:
    """
    Search STAC for Sentinel-2 scenes intersecting the AOI and store results.

    Parameters
    ----------
    aoi : shapely.geometry.Polygon
        Area of interest.
    start_date, end_date : datetime.date
        Date range.
    key_prefix : str
        Prefix used for the Streamlit widget key and the session_state key
        results are stored under (f"{key_prefix}_stac_items"). This lets the
        component be called more than once on the same page -- e.g. a
        before/after change-detection page calling it with key_prefix
        "before" and "after" -- without the two calls overwriting each
        other's results.

    Returns
    -------
    None
        Results are stored in st.session_state[f"{key_prefix}_stac_items"].
    """
    st.subheader("Search Sentinel‑2 Scenes")

    max_cloud_cover = st.slider(
        "Max cloud cover (%)",
        min_value=0, max_value=100, value=40,
        key=f"{key_prefix}_max_cloud_cover",
        help="Raise this if a search comes back empty -- some regions/seasons rarely have fully clear scenes.",
    )

    if st.button("Search", key=f"{key_prefix}_search_button"):
        if aoi is None:
            st.warning("Please select an AOI before searching.")
            return

        with st.spinner("Searching STAC…"):
            items = _search_sentinel2_cached(aoi.wkt, start_date, end_date, max_cloud_cover)

        st.session_state[f"{key_prefix}_stac_items"] = items

        if not items:
            st.warning(
                "No scenes found for the selected AOI, date range, and cloud "
                "cover filter. Try raising the cloud cover slider or "
                "widening the date range."
            )
        else:
            st.success(f"Found {len(items)} scene(s).")


def _thumbnail_url(item: Any) -> str:
    """
    Best-effort lookup of a quicklook preview for a STAC item. Planetary
    Computer's Sentinel-2 items include a `rendered_preview` asset -- a
    dynamically rendered PNG of the full scene (not cropped to the AOI, but
    useful for judging cloud cover / general scene content at a glance).
    """
    asset = item.assets.get("rendered_preview")
    return asset.href if asset is not None else ""


def scene_picker(
    items: List[Any],
    key_prefix: str = "single",
    title: str = "Select a Scene",
    compare_item: Optional[Any] = None,
    default_min_coverage: int = 50,
) -> Optional[Any]:
    """
    Render search results as a selectable table with thumbnail previews,
    acquisition date, cloud cover, and platform -- in place of a dropdown
    whose only visual cue is a concatenated ID string.

    Parameters
    ----------
    items : list of pystac.Item
        Search results, in the order returned by scene_selector.
    key_prefix : str
        Prefix for the widget key, so multiple pickers can coexist on one
        page (e.g. "before" / "after").
    title : str
        Subheader text above the table. Override this when more than one
        picker appears on the same page, so it's clear which is which
        (e.g. "Before — Select a Scene").
    compare_item : pystac.Item, optional
        The scene already selected on the "other side" of a before/after
        comparison, if any (e.g. pass the current after-scene when
        rendering the before-picker). When given, adds "Δ Day-of-Year" and
        "Δ Sun Elev (°)" columns computed against it, so a seasonal or
        illumination mismatch is visible *before* committing to a pair --
        see core.change.comparability_checks for the same checks applied
        after a pair is fully loaded.
    default_min_coverage : int
        Initial position (percent) of the "Minimum coverage" slider that
        filters out low-coverage scenes before they're shown. See that
        slider's help text for what "coverage" means here.

    Returns
    -------
    pystac.Item or None
        The selected item, or None if nothing is currently selected.
    """
    if not items:
        return None

    st.subheader(title)

    min_coverage = st.slider(
        "Minimum coverage (%)",
        min_value=0, max_value=100, value=default_min_coverage,
        key=f"{key_prefix}_min_coverage",
        help=(
            "Hides scenes below this granule-level coverage (100% minus "
            "Sentinel-2's own s2:nodata_pixel_percentage property). This "
            "is coverage of the WHOLE scene, not specifically your AOI -- "
            "a scene can clear this filter and still only partially cover "
            "the AOI, or vice versa, since a small AOI can sit entirely "
            "within a mostly-empty granule's one good corner. The "
            "AOI-specific check happens after you load a scene (see the "
            "coverage warnings and coverage-overlap map once a "
            "before/after comparison is run). Scenes with no coverage "
            "metadata reported are never hidden by this filter."
        ),
    )

    def _granule_coverage_pct(item: Any) -> Optional[float]:
        nodata_pct = item.properties.get("s2:nodata_pixel_percentage")
        if nodata_pct is None:
            return None
        return 100.0 - float(nodata_pct)

    visible_items = [
        item for item in items
        if (cov := _granule_coverage_pct(item)) is None or cov >= min_coverage
    ]
    hidden_count = len(items) - len(visible_items)
    if hidden_count:
        st.caption(
            f"Hiding {hidden_count} of {len(items)} scene(s) below "
            f"{min_coverage}% coverage. Lower the slider to see them."
        )

    if not visible_items:
        st.warning(
            f"No scenes meet the {min_coverage}% coverage threshold. "
            "Lower the slider to see the hidden scene(s)."
        )
        return None

    rows = []
    for item in visible_items:
        row = {
            "Preview": _thumbnail_url(item),
            "Date": item.datetime.strftime("%Y-%m-%d"),
            "Cloud cover (%)": item.properties.get("eo:cloud_cover"),
            "NoData (%)": item.properties.get("s2:nodata_pixel_percentage"),
            "MGRS Tile": item.properties.get("s2:mgrs_tile", "N/A"),
            "UTM Zone": item.properties.get("proj:epsg", "N/A"),
            "Platform": item.properties.get("platform", "N/A"),
        }
        if compare_item is not None:
            row["\u0394 Day-of-Year"] = day_of_year_distance(
                item.datetime.date(), compare_item.datetime.date()
            )
            elev_diff = sun_elevation_diff(item, compare_item)
            row["\u0394 Sun Elev (\u00b0)"] = round(elev_diff, 1) if elev_diff is not None else None
        rows.append(row)
    df = pd.DataFrame(rows)

    if compare_item is not None:
        st.caption(
            "\u0394 Day-of-Year and \u0394 Sun Elev are shown relative to "
            "the scene currently selected on the other side -- lower is a "
            "closer seasonal/illumination match."
        )

    column_config = {
        "Preview": st.column_config.ImageColumn("Preview", width="medium"),
        "Cloud cover (%)": st.column_config.NumberColumn(format="%.1f"),
        "NoData (%)": st.column_config.NumberColumn(
            format="%.1f",
            help="Percentage of this scene's granule with no real data (swath-edge gaps). High values mean the AOI may fall partly outside actual coverage.",
        ),
    }
    if compare_item is not None:
        column_config["\u0394 Day-of-Year"] = st.column_config.NumberColumn(
            help="Circular day-of-year distance from the other side's selected scene (ignores year) -- lower means a closer seasonal match.",
        )
        column_config["\u0394 Sun Elev (\u00b0)"] = st.column_config.NumberColumn(
            format="%.1f",
            help="Absolute difference in sun elevation from the other side's selected scene -- lower means more similar shadows/illumination.",
        )

    with st.expander("Available scenes", expanded=True):
        event = st.dataframe(
            df,
            column_config=column_config,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key=f"{key_prefix}_scene_table",
        )

        selected_rows = event.selection.get("rows", []) if event else []
        if not selected_rows:
            st.caption("Select a row above to load that scene.")
            return None

    return visible_items[selected_rows[0]]

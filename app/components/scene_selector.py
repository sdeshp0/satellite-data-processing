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

    Returns
    -------
    pystac.Item or None
        The selected item, or None if nothing is currently selected.
    """
    if not items:
        return None

    rows = [
        {
            "Preview": _thumbnail_url(item),
            "Date": item.datetime.strftime("%Y-%m-%d"),
            "Cloud cover (%)": item.properties.get("eo:cloud_cover"),
            "Platform": item.properties.get("platform", "N/A"),
        }
        for item in items
    ]
    df = pd.DataFrame(rows)

    st.subheader(title)

    with st.expander("Available scenes", expanded=True):
        event = st.dataframe(
            df,
            column_config={
                "Preview": st.column_config.ImageColumn("Preview", width="medium"),
                "Cloud cover (%)": st.column_config.NumberColumn(format="%.1f"),
            },
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

    return items[selected_rows[0]]

"""
Scene selection widget for Streamlit.
Triggers a STAC search and stores results in session state.
"""

from __future__ import annotations

from datetime import date
import streamlit as st
from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon

from core.stac import search_sentinel2


@st.cache_data(show_spinner=False, ttl=3600)
def _search_sentinel2_cached(aoi_wkt: str, start_date: date, end_date: date):
    """
    Cached wrapper around search_sentinel2.

    Streamlit reruns the whole script on every widget interaction, so
    without caching, re-clicking unrelated widgets on the page would trigger
    a fresh STAC search. Keying the cache on the AOI's WKT string (rather
    than the shapely Polygon object) and the plain dates keeps the cache key
    simple and hashable.
    """
    aoi = shapely_wkt.loads(aoi_wkt)
    return search_sentinel2(aoi, start_date, end_date)


def scene_selector(aoi: Polygon, start_date: date, end_date: date) -> None:
    """
    Search STAC for Sentinel-2 scenes intersecting the AOI and store results.

    Returns
    -------
    None
        Results are stored in st.session_state["stac_items"].
    """
    st.subheader("Search Sentinel‑2 Scenes")

    if st.button("Search"):
        if aoi is None:
            st.warning("Please select an AOI before searching.")
            return

        with st.spinner("Searching STAC…"):
            items = _search_sentinel2_cached(aoi.wkt, start_date, end_date)

        st.session_state["stac_items"] = items

        if not items:
            st.warning("No scenes found for the selected AOI and date range.")
        else:
            st.success(f"Found {len(items)} scene(s). Scroll down to select one.")

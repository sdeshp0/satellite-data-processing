"""
Scene selection widget for Streamlit.
Triggers a STAC search and stores results in session state.
"""

from __future__ import annotations

from typing import Optional, List, Any
import streamlit as st

from core.stac import search_sentinel2


def scene_selector(aoi, start_date, end_date) -> None:
    """
    Search STAC for Sentinel‑2 scenes intersecting the AOI and store results.

    Parameters
    ----------
    aoi : shapely.geometry.Polygon or GeoJSON-like dict
        Area of interest.
    start_date : datetime.date
        Start of date range.
    end_date : datetime.date
        End of date range.

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
            items = search_sentinel2(aoi, start_date, end_date)

        # Store results
        st.session_state["stac_items"] = items

        # User feedback
        if not items:
            st.warning("No scenes found for the selected AOI and date range.")
        else:
            st.success(f"Found {len(items)} scene(s). Scroll down to select one.")

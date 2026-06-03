import streamlit as st
from core.stac import search_sentinel2


def scene_selector(aoi, start_date, end_date):
    """
    Search STAC and let user pick a scene.
    """

    st.subheader("Search Sentinel-2 Scenes")

    stac_search = st.button("Search")

    if stac_search:

        with st.spinner("Searching STAC..."):
            items = search_sentinel2(aoi, start_date, end_date)
            st.session_state["stac_items"] = items

        if not items:
            st.warning("No scenes found.")

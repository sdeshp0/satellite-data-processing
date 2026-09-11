"""
AOI selection widget for Streamlit.
Allows users to search for a location, define AOI size, and preview on a map.
"""

from __future__ import annotations

from typing import Optional, Tuple
import math
import json

import streamlit as st
import leafmap.foliumap as leafmap
from geopy.geocoders import Nominatim
from shapely.geometry import box, Polygon, mapping


def aoi_selector() -> Tuple[Optional[Polygon], Optional[str]]:
    """
    Interactive AOI selector using a text-based location search.

    Returns
    -------
    Tuple[Optional[Polygon], Optional[str]]
        (AOI rectangle in EPSG:4326, human-readable label)
    """
    st.subheader("Area of Interest")

    # --- User input ---
    location_query = st.text_input(
        "Enter a location (city, address, landmark):",
        value=""
    )

    col1, col2 = st.columns(2)
    width_km = col1.number_input(
        "AOI width (km)", min_value=1.0, max_value=200.0, value=10.0
    )
    height_km = col2.number_input(
        "AOI height (km)", min_value=1.0, max_value=200.0, value=10.0
    )

    # --- Search trigger ---
    if st.button("Search location"):
        st.session_state["do_search"] = True

    if not location_query:
        st.info("Enter a location to generate an AOI.")
        return None, None

    if not st.session_state.get("do_search", False):
        return None, None

    # --- Geocoding ---
    # Nominatim's usage policy (https://operations.osmfoundation.org/policies/nominatim/)
    # asks for a genuinely identifying user agent, ideally with contact info.
    # egress IP with a generic user agent is more likely to get rate-limited.
    geolocator = Nominatim(
        user_agent="satellite-data-processing-app for learning how to process satellite image data "
                   "(contact: sidprojects01@gmail.com)",
        timeout=10,
    )

    try:
        location = geolocator.geocode(location_query)
    except Exception:
        st.error("Geocoding service unavailable. Please try again.")
        st.session_state["do_search"] = False
        return None, None

    if location is None:
        st.error("Location not found. Try a different search.")
        st.session_state["do_search"] = False
        return None, None

    lat, lon = location.latitude, location.longitude

    # --- Convert km → degrees ---
    dlat = (height_km / 2) / 111.0
    dlon = (width_km / 2) / (111.0 * abs(math.cos(math.radians(lat))))

    minx = lon - dlon
    maxx = lon + dlon
    miny = lat - dlat
    maxy = lat + dlat

    rect = box(minx, miny, maxx, maxy)

    # Save AOI globally
    st.session_state["aoi"] = rect
    st.session_state["aoi_label"] = location.address

    # Reset search flag
    st.session_state["do_search"] = False

    st.success(f"AOI centered on: {location.address}")
    st.write("Bounds:", rect.bounds)

    # --- Map preview ---
    m = leafmap.Map(center=(lat, lon), zoom=10)

    geojson = json.dumps({
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(rect),
                "properties": {},
            }
        ],
    })

    m.add_geojson(geojson, layer_name="AOI")
    m.to_streamlit(height=500, width=None, embed=True)

    # Prevent rerun loop
    st.stop()


def render_aoi_preview(aoi, height: int = 350):
    """
    Display an AOI preview map in Streamlit.

    Parameters
    ----------
    aoi : Polygon
        AOI geometry in EPSG:4326. If None, nothing is shown.
    height : int
        Height of the map in pixels. Default raised from the original 200px:
        combined with `width=None` (fills the container), a short height on
        a wide page produced a very wide, very thin map. Callers on wide
        pages should also constrain the width by placing this in a
        narrower column rather than relying on height alone -- see
        app/pages/01_Single_Scene.py for an example.
    """

    if aoi is None:
        return

    m = leafmap.Map(center=aoi.centroid.coords[0], zoom=10)
    m.add_geojson(json.dumps(mapping(aoi)), layer_name="AOI")
    m.to_streamlit(height=height, embed=True, width=None)

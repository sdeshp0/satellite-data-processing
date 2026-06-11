"""
AOI selection widget for Streamlit.
Allows users to search for a location, define AOI size, and preview on a map.
"""

from __future__ import annotations

from typing import Optional
import math
import json

import streamlit as st
import leafmap.foliumap as leafmap
from geopy.geocoders import Nominatim
from shapely.geometry import box, Polygon, mapping


def aoi_selector() -> Optional[Polygon]:
    """
    Interactive AOI selector using a text-based location search.

    Returns
    -------
    shapely.geometry.Polygon or None
        AOI rectangle in EPSG:4326 coordinates.
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

    if not location_query:
        st.info("Enter a location to generate an AOI.")
        return None

    # --- Geocoding ---
    geolocator = Nominatim(user_agent="aoi_selector")
    location = geolocator.geocode(location_query)

    if location is None:
        st.error("Location not found. Try a different search.")
        return None

    lat, lon = location.latitude, location.longitude

    # --- Convert km → degrees ---
    dlat = (height_km / 2) / 111.0
    dlon = (width_km / 2) / (111.0 * abs(math.cos(math.radians(lat))))

    minx = lon - dlon
    maxx = lon + dlon
    miny = lat - dlat
    maxy = lat + dlat

    rect = box(minx, miny, maxx, maxy)

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
    m.to_streamlit(height=500, use_container_width=True, embed=True)

    return rect

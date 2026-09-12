"""
AOI selection widget for Streamlit.
Allows users to pick a location -- either from a curated sample list, or by
text search via Nominatim -- define AOI size, and preview on a map.
"""

from __future__ import annotations

from typing import Optional, Tuple
import math
import json

import requests
import streamlit as st
import leafmap.foliumap as leafmap
from geopy.geocoders import Nominatim
from shapely.geometry import box, Polygon, mapping


# Curated fallback for environments where Nominatim search doesn't work --
# notably Streamlit Community Cloud, whose shared egress IP ranges are
# prone to being rate-limited or blocked by Nominatim's usage policy
# (https://operations.osmfoundation.org/policies/nominatim/). This mode has
# no external dependency, so it works everywhere unconditionally.
#
# Locations are picked to double as a starting point for "before/after"
# change-detection demos later -- one or two per event type, plus a couple
# of general-purpose cities.
SAMPLE_LOCATIONS = {
    "Los Angeles, USA (Wildfire)": {
        "lat": 34.0522, "lon": -118.2437, "width_km": 40, "height_km": 40,
    },
    "Sydney Basin, Australia (Wildfire)": {
        "lat": -33.8688, "lon": 151.2093, "width_km": 40, "height_km": 40,
    },
    "Houston, USA (Flood)": {
        "lat": 29.7604, "lon": -95.3698, "width_km": 40, "height_km": 40,
    },
    "Ganges-Brahmaputra Delta, Bangladesh (Flood)": {
        "lat": 23.6850, "lon": 90.3563, "width_km": 60, "height_km": 60,
    },
    "Rondônia, Brazil (Amazon Deforestation)": {
        "lat": -10.83, "lon": -62.90, "width_km": 60, "height_km": 60,
    },
    "Riau, Sumatra, Indonesia (Deforestation)": {
        "lat": 0.293, "lon": 101.706, "width_km": 60, "height_km": 60,
    },
    "San Francisco Bay Area, USA": {
        "lat": 37.7749, "lon": -122.4194, "width_km": 30, "height_km": 30,
    },
    "New York City, USA": {
        "lat": 40.7128, "lon": -74.0060, "width_km": 30, "height_km": 30,
    },
}


def _geocode_locationiq(query: str, api_key: str, timeout: int = 10):
    """
    Forward-geocode via LocationIQ -- a hosted, Nominatim-compatible service
    (same underlying OSM data, same result shape) built for production/cloud
    traffic, unlike OSM's own public Nominatim instance. Free tier is
    generous enough for a demo app; sign up at https://locationiq.com.

    Returns
    -------
    tuple(float, float, str) or None
        (lat, lon, display_name), or None if nothing matched.
    """
    response = requests.get(
        "https://us1.locationiq.com/v1/search",
        params={"key": api_key, "q": query, "format": "json", "limit": 1},
        timeout=timeout,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        return None
    top = results[0]
    return float(top["lat"]), float(top["lon"]), top.get("display_name", query)


def aoi_selector() -> Tuple[Optional[Polygon], Optional[str]]:
    """
    Interactive AOI selector, with two input modes:

    - "Sample locations": pick from a curated list, no external service
      required. Default, since it's the only mode guaranteed to work on
      Streamlit Community Cloud today.
    - "Search by name": free-text search via Nominatim geocoding.

    Returns
    -------
    Tuple[Optional[Polygon], Optional[str]]
        (AOI rectangle in EPSG:4326, human-readable label)
    """
    st.subheader("Area of Interest")

    mode = st.radio(
        "Choose a location by:",
        ["Sample locations", "Search by name"],
        horizontal=True,
        help=(
            "Sample locations always work. Search-by-name depends on "
            "Nominatim's free geocoding service, which can be unreliable "
            "or blocked from shared cloud hosts."
        ),
    )

    if mode == "Sample locations":
        return _sample_location_selector()
    return _search_location_selector()


def _sample_location_selector() -> Tuple[Optional[Polygon], Optional[str]]:
    """
    AOI selection from the curated SAMPLE_LOCATIONS list. No external
    service call, so nothing here can fail the way Nominatim search can.
    """
    location_name = st.selectbox("Sample location", options=list(SAMPLE_LOCATIONS.keys()))
    preset = SAMPLE_LOCATIONS[location_name]

    col1, col2 = st.columns(2)
    width_km = col1.number_input(
        "AOI width (km)", min_value=1.0, max_value=200.0,
        value=float(preset["width_km"]), key="sample_width_km",
    )
    height_km = col2.number_input(
        "AOI height (km)", min_value=1.0, max_value=200.0,
        value=float(preset["height_km"]), key="sample_height_km",
    )

    lat, lon = preset["lat"], preset["lon"]

    dlat = (height_km / 2) / 111.0
    dlon = (width_km / 2) / (111.0 * abs(math.cos(math.radians(lat))))

    rect = box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)

    st.session_state["aoi"] = rect
    st.session_state["aoi_label"] = location_name

    st.success(f"AOI centered on: {location_name}")
    st.caption(
        "Preview shown below. AOI is applied immediately -- no search "
        "button needed for this mode."
    )

    return rect, location_name


def _search_location_selector() -> Tuple[Optional[Polygon], Optional[str]]:
    """
    Free-text AOI selection. Uses LocationIQ if LOCATIONIQ_API_KEY is set in
    st.secrets, otherwise falls back to free Nominatim geocoding.

    The Nominatim fallback works fine in most environments, but not
    reliably from Streamlit Community Cloud: many unrelated apps share the
    same outbound IP ranges there, and Nominatim's usage policy
    (https://operations.osmfoundation.org/policies/nominatim/) blocks at
    the IP level once that shared traffic crosses its 1-request/second
    limit -- a properly identifying user agent doesn't clear an IP-level
    block. LocationIQ is a hosted, Nominatim-compatible service actually
    built for this kind of traffic; see _geocode_locationiq's docstring for
    setup. To enable it: add `LOCATIONIQ_API_KEY = "..."` to your Streamlit
    Cloud app's Secrets (or a local .streamlit/secrets.toml for dev).
    """
    try:
        locationiq_key = st.secrets.get("LOCATIONIQ_API_KEY")
    except Exception:
        locationiq_key = None

    if locationiq_key:
        st.caption("Using LocationIQ for geocoding.")
    else:
        st.caption(
            "Using free OSM Nominatim -- can be unreliable on shared cloud "
            "hosts. Add a LOCATIONIQ_API_KEY secret for a more reliable "
            "option, or use 'Sample locations' above."
        )

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
    if locationiq_key:
        try:
            result = _geocode_locationiq(location_query, locationiq_key)
        except Exception as e:
            st.error(f"LocationIQ geocoding failed: {type(e).__name__}: {e}")
            st.session_state["do_search"] = False
            return None, None

        if result is None:
            st.error("Location not found. Try a different search.")
            st.session_state["do_search"] = False
            return None, None

        lat, lon, display_name = result
    else:
        # Nominatim's usage policy requires a genuinely identifying user
        # agent, ideally with real contact info. Make sure the value below
        # has been swapped for a real email/URL -- but note this alone
        # won't help if the block is at the IP level (see docstring above).
        geolocator = Nominatim(
            user_agent="satellite-data-processing-app for learning how to work with Satellite data"
                       "(contact: sidprojects01@gmail.com)",
            timeout=10,
        )

        try:
            location = geolocator.geocode(location_query)
        except Exception as e:
            # Surfacing the real exception (rather than a generic message)
            # is what lets you tell a 403/block apart from a timeout or
            # something else entirely.
            st.error(
                f"Geocoding service unavailable ({type(e).__name__}: {e}). "
                "This is a known issue on shared cloud hosts (e.g. "
                "Streamlit Community Cloud) -- try 'Sample locations' "
                "above, or configure a LOCATIONIQ_API_KEY secret."
            )
            st.session_state["do_search"] = False
            return None, None

        if location is None:
            st.error("Location not found. Try a different search.")
            st.session_state["do_search"] = False
            return None, None

        lat, lon, display_name = location.latitude, location.longitude, location.address

    # --- Convert km → degrees ---
    dlat = (height_km / 2) / 111.0
    dlon = (width_km / 2) / (111.0 * abs(math.cos(math.radians(lat))))

    rect = box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)

    # Save AOI globally
    st.session_state["aoi"] = rect
    st.session_state["aoi_label"] = display_name

    # Reset search flag
    st.session_state["do_search"] = False

    st.success(f"AOI centered on: {display_name}")
    st.write("Bounds:", rect.bounds)

    return rect, display_name


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

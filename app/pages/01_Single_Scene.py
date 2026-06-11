"""
Single Scene Explorer page.

Workflow:
1. Use the main page to select AOI
2. Select date range
3. Search Sentinel‑2 scenes
4. Pick one scene
5. Display RGB + spectral indices
"""

from __future__ import annotations

from datetime import date
import streamlit as st

from app.components.scene_selector import scene_selector
from app.components.index_display import index_display

st.title("Single Scene Explorer")

# --- Require AOI from main page ---
if "aoi" not in st.session_state:
    st.warning("Please select an AOI on the home page before continuing.")
    st.stop()

aoi = st.session_state["aoi"]

# --- Date Range ---
st.subheader("Date Range")
col1, col2 = st.columns(2)

start_date = col1.date_input("Start date", value=date(2023, 1, 1))
end_date = col2.date_input("End date", value=date(2023, 12, 31))

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

# --- Scene Search ---
scene_selector(aoi, start_date, end_date)

# --- Scene Selection ---
if "stac_items" in st.session_state:
    items = st.session_state["stac_items"]

    if items:
        options = [
            f"{item.id} — {item.datetime.date()} — Cloud {item.properties.get('eo:cloud_cover', 'N/A')}%"
            for item in items
        ]

        selected_label = st.selectbox("Select a scene", options)
        selected_item = items[options.index(selected_label)]

        st.session_state["stac_item"] = selected_item

# --- Display Selected Scene ---
if "stac_item" in st.session_state and aoi is not None:
    index_display(st.session_state["stac_item"], aoi)

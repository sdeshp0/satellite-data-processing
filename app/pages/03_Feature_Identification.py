"""
Feature Identification page (experimental).

Highlights outlines of water bodies/rivers, urban/built-up areas,
vegetation, and bare soil directly on a scene's RGB image, by thresholding
a spectral index into a cleaned mask and tracing its boundary.

Deliberately a standalone page first -- if this holds up well across
different scenes/AOIs, the same overlay can be added to Single Scene and
Change Detection too (e.g. a "damage area" overlay derived from a change
classification, rather than a single-date index).
"""

from __future__ import annotations

import os
import sys
from datetime import date
import streamlit as st

# --- Ensure project root is on sys.path ---
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.components.aoi_selector import render_aoi_preview
from app.components.scene_selector import scene_selector, scene_picker
from app.components.feature_display import feature_display

KEY_PREFIX = "feature"

st.title("Feature Identification")
st.caption(
    "Experimental. Highlights water, urban, vegetation, and bare-soil "
    "extent directly on the RGB image using spectral thresholds."
)

# --- Require AOI from the home page ---
if "aoi" not in st.session_state:
    st.warning("Please select an AOI on the home page before continuing.")
    st.stop()

aoi = st.session_state["aoi"]
aoi_label = st.session_state.get("aoi_label", "Unknown location")

# --- Sidebar: controls ---
with st.sidebar:
    st.subheader("Scene Search")
    st.caption(f"AOI: {aoi_label}")

    start_date = st.date_input(
        "Start date", value=date(2023, 1, 1), key=f"{KEY_PREFIX}_start_date"
    )
    end_date = st.date_input(
        "End date", value=date(2023, 12, 31), key=f"{KEY_PREFIX}_end_date"
    )

    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    scene_selector(aoi, start_date, end_date, key_prefix=KEY_PREFIX)

# --- AOI reference ---

with st.expander(f"AOI preview — {aoi_label}"):
    _, preview_col, _ = st.columns([1, 2, 1])
    with preview_col:
        render_aoi_preview(aoi, height=420, zoom=12)

items_key = f"{KEY_PREFIX}_stac_items"
item_key = f"{KEY_PREFIX}_stac_item"

# --- Scene Selection ---
if items_key in st.session_state:
    items = st.session_state[items_key]

    if items:
        selected_item = scene_picker(items, key_prefix=KEY_PREFIX)
        if selected_item is not None:
            st.session_state[item_key] = selected_item

# --- Display ---
if item_key in st.session_state and aoi is not None:
    feature_display(st.session_state[item_key], aoi, key_prefix=KEY_PREFIX)

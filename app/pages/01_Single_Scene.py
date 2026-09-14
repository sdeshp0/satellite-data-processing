"""
Single Scene Explorer page.

Layout:
- Sidebar: date range + scene search (controls, not content)
- Main area: AOI reference (collapsed), scene picker table with thumbnails,
  then RGB + spectral indices for the selected scene.
"""

from __future__ import annotations

import os
import sys
from datetime import date
import streamlit as st

# --- Ensure project root is on sys.path ---
# Home.py does this too, but on a multi-page deploy a visitor can
# land directly on this page's URL (bookmark, shared link, fresh tab)
# without Home.py ever having run first in that process --
# without this, that path hits "ModuleNotFoundError: No module named 'app'".
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.components.aoi_selector import render_aoi_preview
from app.components.scene_selector import scene_selector, scene_picker
from app.components.index_display import index_display

# Namespaces this page's widget/session_state keys. Distinct from any
# key_prefix used on a future change-detection page (e.g. "before"/"after"),
# so the two pages never collide even though they reuse the same components.
KEY_PREFIX = "single"

st.title("Single Scene Explorer")

# --- Require AOI from main page ---
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

# --- AOI reference, tucked away so it doesn't dominate the page ---
# The map is constrained to a centered column (rather than the full-width
# container) so it renders at a reasonable aspect ratio instead of very
# wide and very thin.
with st.expander(f"AOI preview — {aoi_label}"):
    _, preview_col, _ = st.columns([1, 2, 1])
    with preview_col:
        render_aoi_preview(aoi, height=420)

items_key = f"{KEY_PREFIX}_stac_items"
item_key = f"{KEY_PREFIX}_stac_item"

# --- Scene Selection (thumbnail table instead of a text dropdown) ---
if items_key in st.session_state:
    items = st.session_state[items_key]

    if items:
        selected_item = scene_picker(items, key_prefix=KEY_PREFIX)
        if selected_item is not None:
            st.session_state[item_key] = selected_item

# --- Display Selected Scene ---
if item_key in st.session_state and aoi is not None:
    index_display(st.session_state[item_key], aoi, key_prefix=KEY_PREFIX)

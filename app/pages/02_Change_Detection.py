"""
Change Detection page.

Workflow:
1. AOI comes from the home page (session_state["aoi"]).
2. Sidebar: choose an event type (auto-configures index + methodology),
   then search for a "before" and "after" scene independently.
3. Main: pick one scene from each set, then view the before/after
   comparison -- RGB pair, delta map, classification map, and stats.
"""

from __future__ import annotations

import os
import sys
from datetime import date
import streamlit as st

# --- Ensure project root is on sys.path ---
# Home.py does this too, but a visitor can land directly on this
# page's URL (bookmark, shared link, fresh tab) without Home.py
# having run first in that process -- without this, that hits
# "ModuleNotFoundError: No module named 'app'".
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.components.aoi_selector import render_aoi_preview
from app.components.scene_selector import scene_selector, scene_picker
from app.components.change_display import change_display
from app.components.sample_analyses import sample_analysis_picker
from core.change import EVENT_PRESETS

st.title("Change Detection")

# --- Sample analyses: works even without a prior AOI selection, since it
# sets its own AOI/dates/preset and reruns the page once done. Placed
# before the "AOI required" check below so this is a true one-click entry
# point, not something gated behind first visiting the home page. ---
sample_analysis_picker()
st.divider()

# --- Require AOI (either just set by a sample analysis, or from the home page) ---
if "aoi" not in st.session_state:
    st.info("Select a sample analysis above, or choose an AOI on the home page.")
    st.stop()

aoi = st.session_state["aoi"]
aoi_label = st.session_state.get("aoi_label", "Unknown location")

# --- Sidebar: event type + before/after search controls ---
with st.sidebar:
    st.subheader("Event Type")
    preset_key = st.selectbox(
        "What kind of change are you looking for?",
        options=list(EVENT_PRESETS.keys()),
        format_func=lambda k: EVENT_PRESETS[k]["label"],
        key="change_preset",
    )
    st.caption(EVENT_PRESETS[preset_key]["methodology"])

    st.divider()
    st.subheader("Before")
    st.caption(f"AOI: {aoi_label}")
    before_start = st.date_input(
        "Start date", value=date(2023, 1, 1), key="before_start_date"
    )
    before_end = st.date_input(
        "End date", value=date(2023, 3, 31), key="before_end_date"
    )
    if before_start > before_end:
        st.error("Before: start date must be before end date.")
        st.stop()
    scene_selector(aoi, before_start, before_end, key_prefix="before")

    st.divider()
    st.subheader("After")
    after_start = st.date_input(
        "Start date", value=date(2023, 9, 1), key="after_start_date"
    )
    after_end = st.date_input(
        "End date", value=date(2023, 11, 30), key="after_end_date"
    )
    if after_start > after_end:
        st.error("After: start date must be before end date.")
        st.stop()
    scene_selector(aoi, after_start, after_end, key_prefix="after")

# --- AOI reference, tucked away so it doesn't dominate the page ---
with st.expander(f"AOI preview — {aoi_label}"):
    _, preview_col, _ = st.columns([1, 2, 1])
    with preview_col:
        render_aoi_preview(aoi, height=420)

# --- Scene selection: before/after side by side ---
before_col, after_col = st.columns(2)

with before_col:
    if "before_stac_items" in st.session_state:
        items = st.session_state["before_stac_items"]
        if items:
            selected = scene_picker(
                items, key_prefix="before", title="Before — Select a Scene"
            )
            if selected is not None:
                st.session_state["before_stac_item"] = selected

with after_col:
    if "after_stac_items" in st.session_state:
        items = st.session_state["after_stac_items"]
        if items:
            selected = scene_picker(
                items, key_prefix="after", title="After — Select a Scene"
            )
            if selected is not None:
                st.session_state["after_stac_item"] = selected

before_item = st.session_state.get("before_stac_item")
after_item = st.session_state.get("after_stac_item")

# --- Comparison ---
if before_item is not None and after_item is not None:
    change_display(before_item, after_item, aoi, preset_key, key_prefix="change")
else:
    st.info("Search for and select a scene on both sides to run the comparison.")

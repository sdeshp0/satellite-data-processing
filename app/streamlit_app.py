"""
Main entry point for the Satellite Data Explorer Streamlit app.
Configures the UI, sets the global AOI, and provides top-level navigation.
"""

from __future__ import annotations

import os
import sys
import streamlit as st

# --- Ensure project root is on sys.path ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.components.aoi_selector import aoi_selector, render_aoi_preview

# --- Page Configuration ---
st.set_page_config(
    page_title="Satellite Data Explorer",
    layout="wide",
)

# --- Sidebar ---
st.sidebar.title("Navigation")
st.sidebar.write("Use the pages on the left to explore satellite data.")

# AOI status indicator
if "aoi" in st.session_state and st.session_state["aoi"] is not None:
    st.sidebar.success(f"AOI selected: {st.session_state.get('aoi_label', '')}")

# --- Main Page ---
st.title("Satellite Data Processing App")
st.write("Select an Area of Interest (AOI) here, then use the pages to run analyses.")

# --- AOI Selection Section ---
st.subheader("Select Area of Interest")
aoi_selector()   # Handles its own rerun logic and st.stop()

# --- AOI Banner + Preview ---
if "aoi" in st.session_state and st.session_state["aoi"] is not None:
    st.success(f"AOI selected: {st.session_state.get('aoi_label', '')}")
    render_aoi_preview(st.session_state["aoi"], height=250)
else:
    st.info("Select an AOI to enable analysis pages.")

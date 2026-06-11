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

from app.components.aoi_selector import aoi_selector

# --- Page Configuration ---
st.set_page_config(
    page_title="Satellite Data Explorer",
    layout="wide",
)

# --- Sidebar ---
st.sidebar.title("Navigation")
st.sidebar.write("Use the pages on the left to explore satellite data.")

if "aoi" in st.session_state:
    st.sidebar.success("AOI selected")
else:
    st.sidebar.warning("No AOI selected")

# --- Main Page ---
st.title("Satellite Data Processing App")
st.write("Select an Area of Interest (AOI) here, then use the pages to run analyses.")

st.subheader("Select Area of Interest")
aoi = aoi_selector()

if aoi is not None:
    st.session_state["aoi"] = aoi
    st.success("AOI saved. You can now navigate to any analysis page.")
else:
    st.info("Select an AOI to enable analysis pages.")

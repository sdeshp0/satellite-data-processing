"""
Main entry point for the Satellite Data Explorer Streamlit app.
Configures the UI and provides top-level navigation.
"""

from __future__ import annotations

import os
import sys
import streamlit as st

# --- Ensure project root is on sys.path ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# --- Page Configuration ---
st.set_page_config(
    page_title="Satellite Data Explorer",
    layout="wide",
)

# --- Sidebar ---
st.sidebar.title("Navigation")
st.sidebar.write("Use the pages on the left to explore satellite data.")

# --- Main Page ---
st.title("Satellite Data Processing App")
st.write("Select 'Single Scene Explorer' from the sidebar to begin.")

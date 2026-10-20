"""
Change & Recovery page.

Takes the latest Change Detection comparison and follows its affected
area over time against nearby unchanged land: was the change the event or
the season, and how far has it recovered since? See core.timeseries.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# --- Ensure project root is on sys.path (see 02_Change_Detection.py) ---
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.components.recovery_display import recovery_display

st.title("Change & Recovery")

result = st.session_state.get("change_result")
if result is None:
    st.info(
        "Run a comparison on the Change Detection page first (a sample analysis there works too). "
        "This page then tracks the area it found over time."
    )
    st.stop()

recovery_display(result)

import streamlit as st
from app.components.aoi_selector import aoi_selector
from app.components.scene_selector import scene_selector
from app.components.index_display import index_display

st.title("Single Scene Explorer")

# AOI selection
aoi = aoi_selector()

# Date range
start_date, end_date = st.date_input("Date range", [])

# Search + scene selection
item = scene_selector(aoi, start_date, end_date)

# Display indices + RGB
if item:
    index_display(item, aoi)
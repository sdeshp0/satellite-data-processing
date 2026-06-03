import streamlit as st
from datetime import date

from app.components.aoi_selector import aoi_selector
from app.components.scene_selector import scene_selector
from app.components.index_display import index_display

st.title("Single Scene Explorer")

# --- AOI ---
aoi = aoi_selector()

# --- Date Range ---
st.subheader("Date Range")
col1, col2 = st.columns(2)
start_date = col1.date_input("Start date", value=date(2023, 1, 1))
end_date = col2.date_input("End date", value=date(2023, 12, 31))

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

# --- Scene Selection ---
scene_selector(aoi, start_date, end_date)

if "stac_items" in st.session_state:
    items = st.session_state["stac_items"]
    options = [
        f"{item.id} — {item.datetime.date()} — Cloud {item.properties.get('eo:cloud_cover', 'N/A')}%"
        for item in items
    ]

    selected_item = st.selectbox("Select a scene", options)
    item = items[options.index(selected_item)]
    st.session_state["stac_item"] = item

# --- Display ---
if "stac_item" in st.session_state:
    index_display(st.session_state["stac_item"], aoi)

import streamlit as st
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)


st.set_page_config(page_title="Satellite Data Explorer", layout="wide")

st.sidebar.title("Navigation")
st.sidebar.write("Use the pages on the left to explore satellite data.")

st.title("Satellite Data Processing App")
st.write("Select 'Single Scene Explorer' from the sidebar to begin.")
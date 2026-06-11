"""
Scene loading, processing, and visualization for a selected Sentinel‑2 item.
"""

from __future__ import annotations

from typing import Any, Dict
import streamlit as st
import matplotlib.pyplot as plt

from core.load import load_scene
from core.indices import compute_indices
from core.viz import to_rgb, viz_scl
from core.utils import resample_bands, scale_bands, apply_cloud_mask


def index_display(item: Any, aoi) -> None:
    """
    Load a Sentinel‑2 scene, compute spectral indices, and display results.

    Parameters
    ----------
    item : pystac.Item
        Selected STAC item.
    aoi : shapely.geometry.Polygon
        Area of interest used for clipping.

    Returns
    -------
    None
        Results are displayed directly in Streamlit.
    """
    st.subheader("Scene Visualization")

    # --- Load scene ---
    with st.spinner("Loading scene..."):
        bands = load_scene(item, aoi)

    # --- Process scene ---
    with st.spinner("Processing scene..."):
        bands = resample_bands(bands)
        bands = scale_bands(bands)
        # Optional cloud masking
        # bands = apply_cloud_mask(bands)

    # --- RGB ---
    rgb = to_rgb(bands)
    st.image(rgb, caption="RGB", use_column_width=True)

    # --- SCL  ---
    # scl_fig = viz_scl(bands)
    # st.pyplot(scl_fig)

    # --- Spectral Indices ---
    st.subheader("Spectral Indices")

    indices = compute_indices(bands)

    for name, arr in indices.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(arr, cmap="RdYlGn")
        ax.set_title(name)
        ax.axis("off")
        st.pyplot(fig)

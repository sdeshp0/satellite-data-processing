import streamlit as st
import matplotlib.pyplot as plt

from core.load import load_scene
from core.indices import compute_indices
from core.viz import to_rgb, viz_scl
from core.utils import resample_bands, scale_bands, apply_cloud_mask

def index_display(item, aoi):
    """
    Load scene, compute indices, and display results.
    """

    st.subheader("Scene Visualization")

    with st.spinner("Loading scene..."):
        bands = load_scene(item, aoi)

    with st.spinner("Processing scene..."):
        bands = resample_bands(bands)
        bands = scale_bands(bands)
        #bands = apply_cloud_mask(bands)

    # --- RGB ---
    rgb = to_rgb(bands)
    st.image(rgb, caption="RGB")

    # --- SCL ---
    #scl = viz_scl(bands)
    #st.pyplot(scl)

    # --- Indices ---
    indices = compute_indices(bands)

    st.subheader("Spectral Indices")

    for name, arr in indices.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(arr, cmap="RdYlGn")
        ax.set_title(name)
        ax.axis("off")
        st.pyplot(fig)

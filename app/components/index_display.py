"""
Scene loading, processing, and visualization for a selected Sentinel‑2 item.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st
import matplotlib.pyplot as plt
from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon

from core.load import load_scene
from core.indices import compute_indices
from core.viz import to_rgb, viz_scl
from core.utils import resample_bands, scale_bands, apply_cloud_mask


@st.cache_data(show_spinner=False, ttl=3600)
def _load_scene_cached(
    _item: Any,
    aoi_wkt: str,
    item_id: str,
    max_dim: Optional[int],
) -> Dict[str, Any]:
    """
    Cached wrapper around load_scene.

    `_item` is prefixed with an underscore so Streamlit skips hashing the
    pystac.Item object itself (which can be slow/unstable to hash). The
    actual cache key is `item_id` + `aoi_wkt` + `max_dim` instead, all of
    which are cheap, stable, hashable values.
    """
    aoi = shapely_wkt.loads(aoi_wkt)
    return load_scene(_item, aoi, max_dim=max_dim)


def index_display(item: Any, aoi: Polygon, preview_max_dim: int = 1024) -> None:
    """
    Load a Sentinel‑2 scene, compute spectral indices, and display results.

    Parameters
    ----------
    item : pystac.Item
        Selected STAC item.
    aoi : shapely.geometry.Polygon
        Area of interest used for clipping.
    preview_max_dim : int
        Caps the longer side of each band read to this many pixels. This is
        a decimated read done directly by GDAL, so it also reduces the
        amount of data pulled over the network -- not just faster plotting.
        Set to None if you need full 10m-resolution index values (e.g. for
        precise quantitative analysis rather than a quick-look map).

    Returns
    -------
    None
        Results are displayed directly in Streamlit.
    """
    st.subheader("Scene Visualization")

    with st.spinner("Loading scene..."):
        bands = _load_scene_cached(item, aoi.wkt, item.id, preview_max_dim)

    with st.spinner("Processing scene..."):
        bands = resample_bands(bands)
        bands = scale_bands(bands)
        # Cloud masking is now on by default -- previously commented out,
        # which meant cloudy/shadowed pixels were included in RGB + indices.
        bands = apply_cloud_mask(bands)

    rgb = to_rgb(bands)
    st.image(rgb, caption="RGB", width="stretch")

    # --- SCL ---
    # scl_fig = viz_scl(bands)
    # st.pyplot(scl_fig)

    st.subheader("Spectral Indices")

    indices = compute_indices(bands)

    for name, arr in indices.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(arr, cmap="RdYlGn")
        ax.set_title(name)
        ax.axis("off")
        st.pyplot(fig)

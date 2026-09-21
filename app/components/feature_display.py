"""
Feature identification display: threshold selected indices into cleaned
masks, extract contours, and overlay them on the scene's RGB image.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from rasterio.io import MemoryFile
from shapely.geometry import Polygon

from app.components.scene_loader import load_scene_cached
from core.utils import resample_bands, scale_bands, apply_cloud_mask
from core.indices import compute_indices
from core.viz import to_rgb
from core.features import (
    FEATURE_TYPES,
    threshold_mask,
    extract_contours,
    contours_to_geojson,
    region_stats,
)


def _mask_to_geotiff_bytes(mask: np.ndarray, reference) -> bytes:
    """Package a boolean mask as a single-band GeoTIFF (0/1, uint8)."""
    transform = reference.rio.transform()
    crs = reference.rio.crs
    height, width = mask.shape
    data = mask.astype("uint8")

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=height, width=width, count=1,
            dtype="uint8", crs=crs, transform=transform, nodata=255,
        ) as dst:
            dst.write(data, 1)
        return memfile.read()


def feature_display(
    item: Any,
    aoi: Polygon,
    key_prefix: str = "feature",
    preview_max_dim: int = 1024,
) -> None:
    """
    Threshold spectral indices into cleaned feature masks, overlay their
    outlines on the scene's RGB image, and offer stats + downloads.

    Parameters
    ----------
    item : pystac.Item
        Selected STAC item.
    aoi : shapely.geometry.Polygon
        Area of interest used for clipping.
    key_prefix : str
        Prefix for Streamlit widget/session_state keys.
    preview_max_dim : int
        Caps the longer side of each band read via a decimated read.
    """
    st.subheader("Feature Identification")
    st.caption(
        "Thresholds a spectral index into a cleaned mask and traces its "
        "boundary on the RGB image. Works well for water bodies, urban "
        "extent, and vegetation/bare-soil boundaries -- all several pixels "
        "wide at Sentinel-2's 10m resolution. Roads and individual "
        "buildings are sub-pixel and can't be reliably extracted this way "
        "(see the note at the bottom of this page)."
    )

    selected_features = st.multiselect(
        "Features to identify",
        options=list(FEATURE_TYPES.keys()),
        default=["water"],
        format_func=lambda k: FEATURE_TYPES[k]["label"],
        key=f"{key_prefix}_feature_select",
    )

    if not selected_features:
        st.info("Select at least one feature type above.")
        return

    # Rendered in the sidebar (alongside the search controls) rather than
    # here in the main body -- see index_display.py's identical treatment
    # of this same control for the reasoning (a processing choice made
    # once before viewing results, not tied to a specific visualization).
    with st.sidebar:
        mask_clouds = st.checkbox(
            "Mask clouds / shadows using SCL", value=True, key=f"{key_prefix}_mask_clouds"
        )

    with st.spinner("Loading scene..."):
        bands, coverage_fraction, _load_info = load_scene_cached(
            item, aoi.wkt, item.id, preview_max_dim
        )
        # _load_info carries multi-tile-mosaic cost/coverage diagnostics
        # (see core.load.load_scene) -- not surfaced on this page; see
        # app/components/change_display.py's "Load cost" expander for
        # where that's shown.

    if coverage_fraction < 0.9:
        st.warning(
            f"Only ~{coverage_fraction * 100:.0f}% of the AOI falls within "
            "this scene's actual data footprint -- masks may look sparse "
            "or cut off near the edge."
        )

    bands = resample_bands(bands)
    bands = scale_bands(bands)
    if mask_clouds:
        bands = apply_cloud_mask(bands)

    indices = compute_indices(bands)
    reference = bands["nir"]

    # --- Per-feature threshold controls ---
    threshold_cols = st.columns(len(selected_features))
    thresholds = {}
    for col, key in zip(threshold_cols, selected_features):
        info = FEATURE_TYPES[key]
        with col:
            thresholds[key] = st.slider(
                f"{info['label']} threshold",
                min_value=-1.0, max_value=1.0,
                value=float(info["default_threshold"]),
                step=0.05,
                key=f"{key_prefix}_threshold_{key}",
            )

    min_region_px = st.slider(
        "Minimum region size (pixels)", min_value=1, max_value=200, value=20,
        key=f"{key_prefix}_min_region",
        help="Connected regions smaller than this are dropped as noise.",
    )

    # --- Masks + contours per feature ---
    masks = {}
    contours_by_feature = {}
    stats_rows = []
    for key in selected_features:
        info = FEATURE_TYPES[key]
        arr = indices[info["index"]]
        mask = threshold_mask(arr, thresholds[key], info["direction"], min_region_px)
        masks[key] = mask
        contours_by_feature[key] = extract_contours(mask)

        stats = region_stats(mask, reference)
        stats_rows.append({
            "Feature": info["label"],
            "Regions": stats["num_regions"],
            "Area (km\u00b2)": round(stats["area_km2"], 2),
        })

    # --- RGB + contour overlay ---
    rgb = to_rgb(bands)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb)
    ax.axis("off")

    handles = []
    for key in selected_features:
        info = FEATURE_TYPES[key]
        for contour in contours_by_feature[key]:
            ax.plot(contour[:, 1], contour[:, 0], color=info["color"], linewidth=1.2)
        handles.append(Patch(color=info["color"], label=info["label"]))

    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
    st.pyplot(fig)

    st.dataframe(pd.DataFrame(stats_rows), hide_index=True, width="stretch")

    # --- Downloads ---
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        geojson = contours_to_geojson(contours_by_feature, reference)
        st.download_button(
            "Download outlines (GeoJSON)",
            data=json.dumps(geojson, indent=2),
            file_name=f"{item.id}_features.geojson",
            mime="application/geo+json",
            key=f"{key_prefix}_download_geojson",
        )

    with dl_col2:
        if len(selected_features) == 1:
            only_key = selected_features[0]
            st.download_button(
                f"Download {FEATURE_TYPES[only_key]['label']} mask (GeoTIFF)",
                data=_mask_to_geotiff_bytes(masks[only_key], reference),
                file_name=f"{item.id}_{only_key}_mask.tif",
                mime="image/tiff",
                key=f"{key_prefix}_download_mask",
            )
        else:
            st.caption("Select a single feature type to download its mask as GeoTIFF.")

    st.caption(
        "Want road locations? Deriving roads from spectral reflectance "
        "isn't reliable at 10m resolution. A better approach is overlaying "
        "real road vector data (e.g. OpenStreetMap) as a reference layer "
        "rather than trying to detect roads from pixels -- not built yet."
    )

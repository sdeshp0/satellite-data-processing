"""
Before/after change detection display: loads two scenes for the same AOI,
aligns them to a common grid, computes the event-appropriate index and
classification, and renders the comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.io import MemoryFile
from shapely.geometry import Polygon

from app.components.scene_loader import load_scene_cached
from core.utils import resample_bands, scale_bands
from core.indices import compute_indices
from core.viz import to_rgb
from core.change import (
    EVENT_PRESETS,
    align_bands,
    combined_valid_mask,
    compute_delta,
    class_breakdown,
)


ALL_INDEX_NAMES = ["ndvi", "evi", "savi", "nbr", "ndmi", "ndwi", "ndbi", "bsi"]


def _array_to_geotiff_bytes(arr: np.ndarray, reference) -> bytes:
    """
    Package a 2D array as a single-band, georeferenced GeoTIFF using a
    reference band's CRS/transform (matches the pattern in index_display.py).
    """
    transform = reference.rio.transform()
    crs = reference.rio.crs
    height, width = arr.shape
    data = arr.astype("float32")

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=height, width=width, count=1,
            dtype="float32", crs=crs, transform=transform, nodata=np.nan,
        ) as dst:
            dst.write(data, 1)
        return memfile.read()


def _plot_classification(codes: np.ndarray, labels, colors, title: str):
    """
    Render a discrete classification map with a matching legend. codes < 0
    (masked/invalid pixels) are shown as transparent.
    """
    ordered = sorted(labels.keys())
    cmap = ListedColormap([colors[c] for c in ordered])
    display = np.ma.masked_where(codes < 0, codes)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(display, cmap=cmap, vmin=min(ordered), vmax=max(ordered))
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    handles = [Patch(color=colors[c], label=labels[c]) for c in ordered]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=1, fontsize=8, frameon=False,
    )
    return fig


def change_display(
    before_item: Any,
    after_item: Any,
    aoi: Polygon,
    preset_key: str,
    key_prefix: str = "change",
    preview_max_dim: int = 1024,
) -> None:
    """
    Load, align, and compare two Sentinel-2 scenes over the same AOI.

    Parameters
    ----------
    before_item, after_item : pystac.Item
        The two scenes to compare.
    aoi : shapely.geometry.Polygon
        Area of interest.
    preset_key : str
        Key into core.change.EVENT_PRESETS selecting the index, delta
        direction, and classification methodology to use.
    key_prefix : str
        Prefix for widget/session_state keys on this page.
    preview_max_dim : int
        Passed through to the scene loader; caps read resolution.

    Returns
    -------
    None
        Results are displayed directly in Streamlit.
    """
    preset = EVENT_PRESETS[preset_key]

    st.subheader(preset["label"])
    st.caption(preset["methodology"])

    # --- Index selection (custom preset only) ---
    if preset["index"] is None:
        index_name = st.selectbox(
            "Index to compare",
            options=ALL_INDEX_NAMES,
            index=0,
            key=f"{key_prefix}_custom_index",
        )
    else:
        index_name = preset["index"]

    # --- Load both scenes ---
    with st.spinner("Loading before/after scenes..."):
        bands_before, before_coverage = load_scene_cached(
            before_item, aoi.wkt, before_item.id, preview_max_dim
        )
        bands_after, after_coverage = load_scene_cached(
            after_item, aoi.wkt, after_item.id, preview_max_dim
        )

        bands_before = scale_bands(resample_bands(bands_before))
        bands_after = scale_bands(resample_bands(bands_after))

        # Align "after" onto "before"'s grid HERE, before anything (RGB or
        # indices) is derived from it. Two independently loaded scenes over
        # the "same" AOI can come from different UTM zones and therefore
        # different pixel dimensions -- aligning only a derived index (as an
        # earlier version did) left the RGB preview unaligned, which is why
        # one image could look stretched relative to the other.
        bands_after = align_bands(bands_after, bands_before["nir"])

    # --- Data coverage check ---
    # Sentinel-2 granules aren't always fully covered by real data at swath
    # edges. If the AOI mostly misses one scene's actual footprint, that
    # scene renders as mostly black/empty AND -- before the boundless-read
    # fix in core/load.py -- could produce genuinely corrupted-looking
    # output. Coverage alone no longer causes corruption, but a low value
    # still means real data is missing for a meaningful chunk of the AOI,
    # so it's still worth flagging as a reason to pick a different scene.
    if before_coverage < 0.9 or after_coverage < 0.9:
        low = []
        if before_coverage < 0.9:
            low.append(f"before (~{before_coverage * 100:.0f}% coverage)")
        if after_coverage < 0.9:
            low.append(f"after (~{after_coverage * 100:.0f}% coverage)")
        st.warning(
            f"The AOI only partially falls within the real data footprint "
            f"of the {' and '.join(low)} scene. Consider picking a "
            "different scene for a cleaner comparison."
        )

    # --- Comparability checks: are these two scenes really apples-to-apples? ---
    before_epsg = before_item.properties.get("proj:epsg")
    after_epsg = after_item.properties.get("proj:epsg")
    before_tile = before_item.properties.get("s2:mgrs_tile")
    after_tile = after_item.properties.get("s2:mgrs_tile")

    if before_epsg is not None and after_epsg is not None and before_epsg != after_epsg:
        st.warning(
            f"Before (EPSG:{before_epsg}) and after (EPSG:{after_epsg}) scenes "
            "are in different UTM zones -- the after scene has been "
            "reprojected onto the before scene's grid to allow comparison, "
            "which introduces some resampling."
        )
    elif before_tile is not None and after_tile is not None and before_tile != after_tile:
        st.caption(
            f"Note: before ({before_tile}) and after ({after_tile}) scenes "
            "come from different Sentinel-2 MGRS tiles (same UTM zone, "
            "different source granule)."
        )

    # --- Sensor/baseline mismatch note ---
    # core/load.py corrects for the large, date-dependent BOA_ADD_OFFSET
    # shift automatically. It does NOT correct for the separate, smaller
    # (~1.1% on VNIR bands) documented radiometric cross-calibration
    # difference between Sentinel-2A and Sentinel-2B -- flagging when the
    # two scenes come from different platforms so that's visible rather
    # than silently unaddressed, without overstating it as a major issue.
    before_platform = before_item.properties.get("platform", "unknown")
    after_platform = after_item.properties.get("platform", "unknown")
    if before_platform != after_platform:
        st.caption(
            f"Note: before ({before_platform}) and after ({after_platform}) "
            "scenes come from different Sentinel-2 satellites. ESA applies "
            "a small (~1.1%) cross-calibration correction between them, "
            "which isn't independently corrected for here -- unlikely to "
            "be the dominant signal in a dramatic change, but worth "
            "keeping in mind for subtle comparisons."
        )

    # --- RGB previews, side by side ---
    rgb_col1, rgb_col2 = st.columns(2)
    with rgb_col1:
        st.image(
            to_rgb(bands_before),
            caption=f"Before — {before_item.datetime.date()}",
            width="stretch",
        )
    with rgb_col2:
        st.image(
            to_rgb(bands_after),
            caption=f"After — {after_item.datetime.date()}",
            width="stretch",
        )

    # --- Index per date (both now on the same grid, since bands_after was
    # aligned onto bands_before's grid right after loading) ---
    index_before = compute_indices(bands_before)[index_name]
    index_after = compute_indices(bands_after)[index_name]

    # --- Combined cloud mask: a pixel only counts if clear in BOTH dates ---
    valid = combined_valid_mask(bands_before["scl"], bands_after["scl"])
    index_before = np.where(valid, index_before, np.nan)
    index_after = np.where(valid, index_after, np.nan)

    # --- Delta + classification ---
    delta = compute_delta(index_before, index_after, preset["direction"])
    codes = preset["classify"](index_before, index_after)

    # --- Summary ---
    st.subheader("Change Summary")

    breakdown_df = class_breakdown(codes, preset["labels"])
    baseline = set(preset["baseline_codes"])
    valid_codes = codes[codes >= 0]
    total_valid = valid_codes.size
    changed_count = int(np.sum(~np.isin(valid_codes, list(baseline)))) if total_valid else 0
    changed_pct = round(100 * changed_count / total_valid, 1) if total_valid else 0.0
    mean_delta = float(np.nanmean(delta)) if total_valid else float("nan")

    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.metric("Area with detected change", f"{changed_pct}%")
    with stat_col2:
        st.metric(f"Mean \u0394{index_name.upper()} over AOI", f"{mean_delta:.3f}")

    if not breakdown_df.empty:
        st.dataframe(breakdown_df, hide_index=True, width="stretch")
        st.bar_chart(breakdown_df.set_index("Class")["Percent"])
    else:
        st.caption("No valid (cloud-free in both dates) pixels to summarize.")

    # --- Delta + classification maps ---
    map_col1, map_col2 = st.columns(2)

    finite_delta = delta[~np.isnan(delta)]
    vlim = float(np.nanpercentile(np.abs(finite_delta), 98)) if finite_delta.size else 1.0
    vlim = max(vlim, 1e-3)

    with map_col1:
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(delta, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.set_title(f"\u0394{index_name.upper()}", fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)
        st.pyplot(fig)
        st.caption(f"Red = increase in {index_name.upper()}, blue = decrease.")

        st.download_button(
            f"Download \u0394{index_name.upper()} (GeoTIFF)",
            data=_array_to_geotiff_bytes(delta, bands_before["nir"]),
            file_name=f"{before_item.id}_{after_item.id}_delta_{index_name}.tif",
            mime="image/tiff",
            key=f"{key_prefix}_download_delta",
        )

    with map_col2:
        fig2 = _plot_classification(codes, preset["labels"], preset["colors"], "Classification")
        st.pyplot(fig2)

        codes_export = codes.astype("float32")
        codes_export[codes < 0] = np.nan

        st.download_button(
            "Download Classification (GeoTIFF)",
            data=_array_to_geotiff_bytes(codes_export, bands_before["nir"]),
            file_name=f"{before_item.id}_{after_item.id}_classification.tif",
            mime="image/tiff",
            key=f"{key_prefix}_download_class",
        )

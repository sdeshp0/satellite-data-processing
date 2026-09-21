"""
Scene loading, processing, and visualization for a selected Sentinel‑2 item.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import xarray as xr
from PIL import Image
from rasterio.io import MemoryFile
from shapely.geometry import Polygon

from app.components.scene_loader import load_scene_cached
from core.indices import compute_indices
from core.viz import to_rgb, viz_scl
from core.utils import resample_bands, scale_bands, apply_cloud_mask
from core.stats import summary_table, scl_breakdown, coverage_info


# Colormap and display range chosen per index *category*, so color direction
# reflects what the index physically represents rather than one blanket
# red-green scale for everything:
#
# - vegetation: RdYlGn, high (more vegetation) -> green, low -> red.
# - burn: also RdYlGn, but for a different reason -- NBR moves *opposite*
#   of burn severity (healthy vegetation = high NBR, char/ash = low NBR),
#   so mapping low -> red / high -> green is what makes burned areas show
#   up red. This isn't a coincidence with vegetation's colormap; it's the
#   correct direction for NBR specifically.
# - moisture: BrBG (brown -> blue-green), high (more water/moisture) -> blue.
# - urban: OrRd, high (more built-up) -> red-orange.
# - soil: YlOrBr, high (more bare soil) -> brown.
CATEGORY_STYLE = {
    "vegetation": {"cmap": "RdYlGn", "vmin": -1.0, "vmax": 1.0},
    "burn":       {"cmap": "RdYlGn", "vmin": -1.0, "vmax": 1.0},
    "moisture":   {"cmap": "BrBG",   "vmin": -1.0, "vmax": 1.0},
    "urban":      {"cmap": "OrRd",   "vmin": -0.5, "vmax": 0.5},
    "soil":       {"cmap": "YlOrBr", "vmin": -0.5, "vmax": 0.5},
}

# Short, fixed explanations shown under each index chart, plus a one-line
# color legend hint since the color direction now differs per chart. Paired
# at render time with a computed mean-over-AOI value so the text is grounded
# in the specific scene being viewed, not just a generic definition.
INDEX_INFO = {
    "ndvi": {
        "label": "NDVI — Vegetation",
        "category": "vegetation",
        "desc": (
            "Higher (green) values indicate denser, healthier vegetation; "
            "values near zero or negative indicate bare soil, water, or "
            "built-up surfaces."
        ),
        "color_note": "Green = more vegetation, red = less.",
    },
    "evi": {
        "label": "EVI — Vegetation (Enhanced)",
        "category": "vegetation",
        "desc": (
            "Corrects for canopy background and atmospheric effects; less "
            "prone to saturating over dense, high-biomass vegetation than "
            "NDVI."
        ),
        "color_note": "Green = more vegetation, red = less.",
    },
    "savi": {
        "label": "SAVI — Vegetation (Soil-Adjusted)",
        "category": "vegetation",
        "desc": (
            "Like NDVI, but reduces the influence of soil brightness -- "
            "more reliable over sparsely vegetated areas."
        ),
        "color_note": "Green = more vegetation, red = less.",
    },
    "nbr": {
        "label": "NBR — Burn Severity",
        "category": "burn",
        "desc": (
            "Large drops in NBR between a before/after pair typically flag "
            "burned vegetation. Low absolute values can also indicate water "
            "or bare ground."
        ),
        "color_note": "Red = stronger burn signal, green = healthy vegetation.",
    },
    "ndmi": {
        "label": "NDMI — Vegetation Moisture",
        "category": "moisture",
        "desc": (
            "Higher values indicate higher canopy water content; useful for "
            "drought stress and fuel-moisture assessment."
        ),
        "color_note": "Blue/teal = more moisture, brown = drier.",
    },
    "ndwi": {
        "label": "NDWI — Surface Water",
        "category": "moisture",
        "desc": (
            "Higher values indicate open water; vegetation and dry soil "
            "trend negative."
        ),
        "color_note": "Blue/teal = more water, brown = drier land.",
    },
    "ndbi": {
        "label": "NDBI — Built-up Area",
        "category": "urban",
        "desc": (
            "Higher values indicate impervious / built-up surfaces; "
            "vegetation and water trend negative."
        ),
        "color_note": "Orange/red = more built-up, pale = less.",
    },
    "bsi": {
        "label": "BSI — Bare Soil",
        "category": "soil",
        "desc": (
            "Higher values indicate exposed bare soil; vegetation and water "
            "trend negative."
        ),
        "color_note": "Brown = more bare soil, pale = less.",
    },
}

# Indices shown by default; the rest are available via the multiselect.
DEFAULT_INDICES = ["ndvi", "evi", "ndmi", "nbr"]

# Charts per row in the index grid. Every row uses exactly this many
# st.columns(), even if fewer charts are placed in the last row -- see the
# note in the render loop below for why that matters.
GRID_COLS = 3


def _rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _index_to_geotiff_bytes(arr: np.ndarray, reference: xr.DataArray) -> bytes:
    """
    Package a 2D index array as a single-band, georeferenced GeoTIFF, using
    the CRS/transform already attached to a reference band (e.g. the
    resampled NIR band) so the exported file lines up correctly in GIS tools.
    """
    transform = reference.rio.transform()
    crs = reference.rio.crs
    height, width = arr.shape
    data = arr.astype("float32")

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(data, 1)
        return memfile.read()


def _style_for(name: str) -> Tuple[str, float, float]:
    """
    Resolve (cmap, vmin, vmax) for an index, based on its category. Falls
    back to the vegetation style if the index has no category mapped.
    """
    category = INDEX_INFO.get(name, {}).get("category", "vegetation")
    style = CATEGORY_STYLE.get(category, CATEGORY_STYLE["vegetation"])
    return style["cmap"], style["vmin"], style["vmax"]


def index_display(
    item: Any,
    aoi: Polygon,
    key_prefix: str = "single",
    preview_max_dim: int = 1024,
) -> None:
    """
    Load a Sentinel‑2 scene, compute spectral indices and summary
    statistics, and display results.

    Layout: scene metadata in an expander, a large RGB image, scene
    statistics (land cover breakdown + band/index stats tables), then a
    selectable grid of spectral index charts, each colored according to
    what it represents (see CATEGORY_STYLE).

    Parameters
    ----------
    item : pystac.Item
        Selected STAC item.
    aoi : shapely.geometry.Polygon
        Area of interest used for clipping.
    key_prefix : str
        Prefix for Streamlit widget/session_state keys, so this component
        can be used more than once on the same page (e.g. "before"/"after"
        on a future change-detection page) without key collisions.
    preview_max_dim : int
        Caps the longer side of each band read to this many pixels via a
        decimated read. Set to None for full 10m-resolution reads.

    Returns
    -------
    None
        Results are displayed directly in Streamlit.
    """
    st.subheader("Scene Visualization")

    # --- Scene metadata ---
    with st.expander("Scene details"):
        props = item.properties
        st.write(f"**Scene ID:** {item.id}")
        st.write(f"**Acquired:** {item.datetime}")
        st.write(f"**Cloud cover:** {props.get('eo:cloud_cover', 'N/A')}%")
        st.write(f"**Platform:** {props.get('platform', 'N/A')}")
        st.write(f"**Sun elevation:** {props.get('view:sun_elevation', 'N/A')}")
        st.write(
            f"**Processing baseline:** {props.get('s2:processing_baseline', 'N/A')} "
            "(baseline \u2265 4.00 has a radiometric offset correction "
            "automatically applied on load)"
        )
        st.write(f"**Bounding box:** {item.bbox}")

    # --- Options ---
    # Rendered in the sidebar (alongside the search controls) rather than
    # here in the main body: this is a processing/config choice made once
    # before viewing results, not something tied to a specific
    # visualization, so it fits better grouped with the other "how should
    # scenes be searched/prepared" controls (see scene_selector's cloud-
    # cover and minimum-coverage sliders for the same reasoning). Using
    # `with st.sidebar:` here rather than changing this function's
    # signature keeps the component self-contained -- Streamlit's sidebar
    # accepts new widgets from anywhere in the script, not just from code
    # that runs inside the page's own `with st.sidebar:` block.
    with st.sidebar:
        mask_clouds = st.checkbox(
            "Mask clouds / shadows using SCL",
            value=True,
            key=f"{key_prefix}_mask_clouds",
        )

    # --- Load scene ---
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
            "this scene's actual data footprint (Sentinel-2 granules aren't "
            "always fully covered by real data at swath edges). The rest is "
            "filled with nodata -- pick a different scene if this looks "
            "mostly black or empty."
        )

    # --- Process scene ---
    with st.spinner("Processing scene..."):
        bands = resample_bands(bands)
        bands = scale_bands(bands)  # "scl" is excluded from scaling
        raw_scl = bands["scl"]      # unaffected by the mask toggle below
        if mask_clouds:
            bands = apply_cloud_mask(bands)

    # --- RGB ---
    # Constrained to the same centered-column width as the AOI preview on
    # this page (see app/pages/01_Single_Scene.py), rather than stretching
    # to the full container -- at full width it dwarfed everything else.
    rgb = to_rgb(bands)
    _, rgb_col, _ = st.columns([1, 2, 1])
    with rgb_col:
        st.image(rgb, caption="RGB (2nd-98th percentile stretch)", width="stretch")

        st.download_button(
            "Download RGB (PNG)",
            data=_rgb_to_png_bytes(rgb),
            file_name=f"{item.id}_rgb.png",
            mime="image/png",
            key=f"{key_prefix}_download_rgb",
        )

    # --- SCL  ---
    # scl_fig = viz_scl(bands)
    # st.pyplot(scl_fig)

    # --- Scene Statistics ---
    st.subheader("Scene Statistics")

    coverage = coverage_info(bands["nir"])
    st.caption(
        f"AOI window: {coverage['width_px']}×{coverage['height_px']} px "
        f"at ~{coverage['resolution_m']:.0f} m resolution "
        f"(~{coverage['area_km2']:.1f} km²)"
    )

    stat_col1, stat_col2 = st.columns(2)

    with stat_col1:
        st.markdown("**Land cover breakdown (SCL)**")
        scl_df = scl_breakdown(raw_scl)
        if not scl_df.empty:
            st.bar_chart(scl_df.set_index("Class")["Percent"])
            st.dataframe(scl_df, hide_index=True, width="stretch")
        else:
            st.caption("No SCL data available for this scene.")

    with stat_col2:
        st.markdown("**Band reflectance statistics**")
        band_arrays = {
            name: da.values for name, da in bands.items() if name != "scl"
        }
        st.dataframe(summary_table(band_arrays).round(3), width="stretch")

    # --- Spectral Indices ---
    st.subheader("Spectral Indices")

    all_indices = compute_indices(bands)
    reference_band = bands["nir"]  # georeferencing source for GeoTIFF export

    selected = st.multiselect(
        "Indices to display",
        options=list(all_indices.keys()),
        default=[n for n in DEFAULT_INDICES if n in all_indices],
        format_func=lambda n: INDEX_INFO.get(n, {}).get("label", n.upper()),
        key=f"{key_prefix}_index_select",
    )

    if not selected:
        st.info("Select at least one index above to display charts.")
        return

    st.markdown("**Index statistics**")
    index_arrays = {name: all_indices[name] for name in selected}
    st.dataframe(summary_table(index_arrays).round(3), width="stretch")

    for row_start in range(0, len(selected), GRID_COLS):
        row_names = selected[row_start: row_start + GRID_COLS]

        # Always create GRID_COLS columns, even if this row has fewer charts
        # than that (e.g. the last row), so column width -- and therefore
        # chart size -- stays consistent across every row.
        cols = st.columns(GRID_COLS)

        for col, name in zip(cols, row_names):
            arr = all_indices[name]
            info = INDEX_INFO.get(name, {"label": name.upper(), "desc": "", "color_note": ""})
            cmap, vmin, vmax = _style_for(name)

            with col:
                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(info["label"], fontsize=11)
                ax.axis("off")
                fig.colorbar(im, ax=ax, shrink=0.7, label=name.upper())
                st.pyplot(fig)

                mean_val = np.nanmean(arr)
                st.caption(
                    f"{info['desc']} {info.get('color_note', '')} "
                    f"**Mean over AOI: {mean_val:.2f}**"
                )

                st.download_button(
                    f"Download {name.upper()} (GeoTIFF)",
                    data=_index_to_geotiff_bytes(arr, reference_band),
                    file_name=f"{item.id}_{name}.tif",
                    mime="image/tiff",
                    key=f"{key_prefix}_download_{name}",
                )
        # Any leftover columns in this row (when row_names is shorter than
        # GRID_COLS) are simply left empty.

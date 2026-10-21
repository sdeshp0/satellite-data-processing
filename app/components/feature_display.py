"""
Feature identification display: threshold selected indices into cleaned
masks, crosshatch them on an interactive map (with basemaps, land cover
and buildings as layers), and list each region they contain; the masks'
outlines are offered as a GeoJSON download.
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

from app.components.scene_loader import load_buildings_cached, load_scene_cached, load_worldcover_cached
from core.change import format_area, pixel_area_ha
from core.inventory import dominant_class, feature_regions, inventory_findings
from core.landcover import WORLDCOVER_CLASSES, landcover_composition, landcover_rgb
from core.utils import resample_bands, scale_bands, apply_cloud_mask, apply_nodata_mask
from core.indices import compute_indices
from core.viz import to_rgb
from core.features import (
    FEATURE_TYPES,
    threshold_mask,
    extract_contours,
    contours_to_geojson,
)
from core.webmap import feature_map, region_geojson

MAP_HEIGHT_PX = 560


def _hex_to_rgb(color: str) -> tuple:
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


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
    Threshold spectral indices into cleaned feature masks, crosshatch them
    on the scene's RGB image, and offer stats + downloads.

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
        "Thresholds a spectral index into a cleaned mask and crosshatches "
        "it on the RGB image, one colour per feature type. Works well for water bodies, urban "
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
    bands = apply_nodata_mask(bands)
    # Land cover doesn't depend on clouds, so its composition covers every
    # pixel with data, not just the cloud-free ones.
    has_data = np.isfinite(bands["nir"].values)
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
    for key in selected_features:
        info = FEATURE_TYPES[key]
        mask = threshold_mask(indices[info["index"]], thresholds[key], info["direction"], min_region_px)
        masks[key] = mask
        contours_by_feature[key] = extract_contours(mask)

    px_ha = pixel_area_ha(reference)
    clear_ha = float(np.isfinite(indices["ndvi"]).sum()) * px_ha
    landcover, landcover_year, landcover_note = _load_landcover(reference, item.datetime.year)
    building_rows = building_cols = None
    if st.checkbox(
        "Count buildings in each region", key=f"{key_prefix}_buildings",
        help="Loads Microsoft building footprints for the AOI (can take a while for large cities).",
    ):
        building_rows, building_cols = _load_buildings(reference)

    inventory = {}
    for key in selected_features:
        regions, labelled, areas = feature_regions(masks[key], reference, px_ha, building_rows, building_cols)
        cover = dominant_class(labelled, landcover, regions) if landcover is not None else {}
        inventory[key] = (regions, labelled, areas, cover)

    # --- Map, then the per-feature list that drives its zoom ---
    st.markdown("**Map**")
    map_slot = st.container()
    rows = _inventory_rows(selected_features, inventory, building_rows is not None)
    if rows:
        st.markdown("**Regions found**")
        for key in selected_features:
            regions, _, areas, _ = inventory[key]
            found = inventory_findings(key, regions, areas, clear_ha)
            st.markdown(f"*{FEATURE_TYPES[key]['label']}:* " + " ".join(f.text for f in found))
        table = st.dataframe(
            pd.DataFrame([row for row, _ in rows]), hide_index=True, width="stretch",
            on_select="rerun", selection_mode="single-row", key=f"{key_prefix}_inventory",
            column_config={"Area (ha)": st.column_config.NumberColumn(format="%.1f")},
        )
        picked = table.selection.rows
        st.caption("Select a row to zoom the map to that region. Location is the part of the AOI its centre falls in.")
    else:
        picked = []

    # A selection can outlive its row when the thresholds change.
    focus = rows[picked[0]][1] if picked and picked[0] < len(rows) else None
    with map_slot:
        rgb = to_rgb(bands)
        hatches = [(FEATURE_TYPES[k]["label"], masks[k], _hex_to_rgb(FEATURE_TYPES[k]["color"])) for k in selected_features]
        geojson_regions = _regions_geojson(selected_features, inventory, reference, building_rows is not None)
        buildings = None
        if building_rows is not None:
            buildings = np.zeros(reference.shape, dtype=bool)
            buildings[building_rows, building_cols] = True
        fmap = feature_map(
            reference, rgb, hatches, geojson_regions, _tooltip_fields(building_rows is not None, landcover is not None),
            buildings=buildings,
            landcover_rgb=None if landcover is None else landcover_rgb(np.where(has_data, landcover, 0)),
            focus=focus,
            focus_label="" if focus is None else f"#{focus.rank}, {format_area(focus.area_ha)}",
        )
        st.iframe(fmap.get_root().render(), height=MAP_HEIGHT_PX)
        st.caption(
            "Each feature is crosshatched in its colour over the scene. Use the layer list to switch to the "
            "basemap or street map, or turn on land cover"
            + (" and buildings" if buildings is not None else "")
            + ". Hover a region for its details."
        )

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

    _landcover_section(reference, has_data, landcover, landcover_year, landcover_note, key_prefix)

    st.caption(
        "Want road locations? Deriving roads from spectral reflectance "
        "isn't reliable at 10m resolution. A better approach is overlaying "
        "real road vector data (e.g. OpenStreetMap) as a reference layer "
        "rather than trying to detect roads from pixels -- not built yet."
    )


def _load_landcover(reference, scene_year: int):
    """ESA WorldCover on the reference grid, its year, and a note when it couldn't be loaded."""
    grid_key = f"{reference.rio.crs}|{tuple(reference.rio.transform())}|{reference.shape}"
    try:
        with st.spinner("Loading land cover..."):
            landcover, year = load_worldcover_cached(reference, grid_key, scene_year)
    except Exception as exc:  # network/data problems shouldn't break the page
        return None, 0, (
            "Land-cover data couldn't be loaded from Planetary Computer "
            f"({type(exc).__name__}: {str(exc)[:150]}), so this section is skipped."
        )
    if landcover is None:
        return None, 0, "No ESA WorldCover land-cover data covers this AOI."
    return landcover, year, ""


def _load_buildings(reference):
    """Building-centre pixels (rows, cols), or (None, None) with a caption when they can't be loaded."""
    grid_key = f"{reference.rio.crs}|{tuple(reference.rio.transform())}|{reference.shape}"
    try:
        with st.spinner("Loading building footprints..."):
            rows, cols, found = load_buildings_cached(reference, grid_key)
    except Exception as exc:  # network/data problems shouldn't break the page
        st.caption(f"Building footprints couldn't be loaded ({type(exc).__name__}), so they're left out.")
        return None, None
    if not found:
        st.caption("No Microsoft building footprints cover this AOI.")
        return None, None
    return rows, cols


def _inventory_rows(selected_features, inventory, with_buildings: bool):
    """(table row, Region) per listed region, features in the order selected."""
    out = []
    for key in selected_features:
        regions, _, _, cover = inventory[key]
        for r in regions:
            row = {
                "Feature": FEATURE_TYPES[key]["label"],
                "Rank": r.rank,
                "Area (ha)": round(r.area_ha, 1),
                "Location": r.sector,
                "Lat": round(r.lat, 5),
                "Lon": round(r.lon, 5),
            }
            if cover:
                row["Mostly (WorldCover)"] = WORLDCOVER_CLASSES.get(cover[r.label], ("Unknown", ""))[0]
            if with_buildings:
                row["Buildings"] = r.buildings
            out.append((row, r))
    return out


def _tooltip_fields(with_buildings: bool, with_cover: bool):
    fields = [("feature", "Feature"), ("rank", "Rank"), ("area", "Area"), ("location", "Location")]
    if with_cover:
        fields.append(("cover", "Mostly"))
    if with_buildings:
        fields.append(("buildings", "Buildings"))
    return fields


def _regions_geojson(selected_features, inventory, reference, with_buildings: bool):
    """Hover shapes for every listed region of every selected feature."""
    collection = {"type": "FeatureCollection", "features": []}
    for key in selected_features:
        regions, labelled, _, cover = inventory[key]
        if not regions:
            continue
        props = {}
        for r in regions:
            props[r.label] = {
                "feature": FEATURE_TYPES[key]["label"],
                "rank": r.rank,
                "area": format_area(r.area_ha),
                "location": r.sector,
            }
            if cover:
                props[r.label]["cover"] = WORLDCOVER_CLASSES.get(cover[r.label], ("Unknown", ""))[0]
            if with_buildings:
                props[r.label]["buildings"] = r.buildings
        collection["features"] += region_geojson(labelled, regions, reference, props)["features"]
    return collection


def _landcover_section(reference, has_data: np.ndarray, landcover, year: int, note: str, key_prefix: str) -> None:
    """ESA WorldCover composition of the AOI, with an optional class map."""
    st.markdown("**Land cover in this AOI**")
    if landcover is None:
        st.caption(note)
        return

    composition = landcover_composition(landcover, has_data, pixel_area_ha(reference))
    if composition.empty:
        st.caption("No land-cover data overlaps the scene's valid pixels.")
        return
    st.dataframe(composition, hide_index=True, width="stretch")
    st.caption(
        f"From ESA WorldCover {year} (10 m), a global land-cover map. It is a "
        "fixed annual snapshot, so it's a useful reference for the outlines "
        "above but won't reflect changes since then."
    )

    if st.checkbox("Show land-cover map", key=f"{key_prefix}_show_landcover"):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(landcover_rgb(np.where(has_data, landcover, 0)))
        ax.axis("off")
        present = set(np.unique(landcover[has_data]).tolist())
        handles = [
            Patch(color=color, label=label)
            for code, (label, color) in WORLDCOVER_CLASSES.items()
            if code in present
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
        st.pyplot(fig)

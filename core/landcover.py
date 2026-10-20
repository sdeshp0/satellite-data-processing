"""
Land-cover context for a change map: which kinds of land the detected
change fell on, from ESA WorldCover (10 m, global, on Planetary Computer).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import box, mapping

WORLDCOVER_COLLECTION = "esa-worldcover"
WORLDCOVER_NODATA = 0

# ESA WorldCover class codes -> (label, map colour from the product spec).
WORLDCOVER_CLASSES: Dict[int, Tuple[str, str]] = {
    10: ("Tree cover", "#006400"),
    20: ("Shrubland", "#ffbb22"),
    30: ("Grassland", "#ffff4c"),
    40: ("Cropland", "#f096ff"),
    50: ("Built-up", "#fa0000"),
    60: ("Bare / sparse vegetation", "#b4b4b4"),
    70: ("Snow and ice", "#f0f0f0"),
    80: ("Permanent water", "#0064c8"),
    90: ("Herbaceous wetland", "#0096a0"),
    95: ("Mangroves", "#00cf75"),
    100: ("Moss and lichen", "#fae6a0"),
}
BUILT_UP = 50


def search_worldcover(reference: xr.DataArray) -> List[Any]:
    """WorldCover items covering the reference grid, latest year only."""
    from pystac_client import Client

    bbox = transform_bounds(reference.rio.crs, "EPSG:4326", *reference.rio.bounds())
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    items = list(catalog.search(collections=[WORLDCOVER_COLLECTION], intersects=mapping(box(*bbox))).items())
    if not items:
        return []
    latest = max(item.datetime.year for item in items if item.datetime)
    return [item for item in items if item.datetime and item.datetime.year == latest]


def _signed(href: str) -> str:
    return pc.sign(href.split("?", 1)[0]) if href.startswith("http") else href


def worldcover_on_grid(items: Sequence[Any], reference: xr.DataArray) -> np.ndarray:
    """
    Read each item's "map" asset over the reference grid's footprint and
    resample it (nearest neighbour) onto that grid. Where tiles overlap the
    first one with data wins; 0 marks pixels with no land-cover data.
    """
    dst_crs = reference.rio.crs
    dst_transform = reference.rio.transform()
    shape = reference.shape[-2:]
    out = np.zeros(shape, dtype=np.uint8)

    for item in items:
        with rasterio.open(_signed(item.assets["map"].href)) as src:
            left, bottom, right, top = transform_bounds(dst_crs, src.crs, *reference.rio.bounds())
            window = from_bounds(left, bottom, right, top, src.transform).round_offsets().round_lengths()
            data = src.read(1, window=window, boundless=True, fill_value=WORLDCOVER_NODATA)
            tile = np.zeros(shape, dtype=np.uint8)
            reproject(
                source=data,
                destination=tile,
                src_transform=src.window_transform(window),
                src_crs=src.crs,
                src_nodata=WORLDCOVER_NODATA,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=WORLDCOVER_NODATA,
                resampling=Resampling.nearest,
            )
        fill = (out == WORLDCOVER_NODATA) & (tile != WORLDCOVER_NODATA)
        out[fill] = tile[fill]
    return out


def landcover_breakdown(
    landcover: np.ndarray, changed: np.ndarray, analysed: np.ndarray, pixel_area_ha: float
) -> pd.DataFrame:
    """
    For each land-cover class the change touched: the changed area, its
    share of all detected change, and the share of that class's analysed
    area that changed. Largest changed area first.
    """
    total_changed = int(changed.sum())
    rows = []
    for code, (label, _) in WORLDCOVER_CLASSES.items():
        in_class = landcover == code
        n_changed = int((in_class & changed).sum())
        if n_changed == 0:
            continue
        n_analysed = int((in_class & analysed).sum())
        rows.append({
            "Land cover": label,
            "Changed area (ha)": round(n_changed * pixel_area_ha, 1),
            "Share of detected change (%)": round(100 * n_changed / total_changed, 1),
            "Share of this land cover changed (%)": round(100 * n_changed / n_analysed, 1) if n_analysed else 0.0,
        })
    columns = ["Land cover", "Changed area (ha)", "Share of detected change (%)", "Share of this land cover changed (%)"]
    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values("Changed area (ha)", ascending=False).reset_index(drop=True)

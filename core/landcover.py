"""
Land-cover context for a change map: which kinds of land the detected
change fell on, from ESA WorldCover (10 m, global, on Planetary Computer).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def worldcover_year(item: Any) -> Optional[int]:
    """
    Product year of a WorldCover item. Its STAC `datetime` is null (the map
    covers a whole year), so fall back to `start_datetime`, then the year
    in the item id (e.g. ESA_WorldCover_10m_2021_v200_N36W123).
    """
    when = item.datetime or item.common_metadata.start_datetime
    if when is not None:
        return when.year
    match = re.search(r"_(20\d\d)_", item.id)
    return int(match.group(1)) if match else None


def search_worldcover(reference: xr.DataArray) -> Tuple[List[Any], int]:
    """WorldCover items covering the reference grid, latest year only, and that year (0 if none)."""
    from pystac_client import Client

    bbox = transform_bounds(reference.rio.crs, "EPSG:4326", *reference.rio.bounds())
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    items = list(catalog.search(collections=[WORLDCOVER_COLLECTION], intersects=mapping(box(*bbox))).items())
    if not items:
        return [], 0
    years = {id(item): worldcover_year(item) or 0 for item in items}
    latest = max(years.values())
    return [item for item in items if years[id(item)] == latest], latest


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


def landcover_composition(landcover: np.ndarray, valid: np.ndarray, pixel_area_ha: float) -> pd.DataFrame:
    """Area and share of each land-cover class over the valid pixels, largest first."""
    valid = valid & (landcover != WORLDCOVER_NODATA)
    total = int(valid.sum())
    rows = []
    for code, (label, _) in WORLDCOVER_CLASSES.items():
        n = int((valid & (landcover == code)).sum())
        if n:
            rows.append({
                "Land cover": label,
                "Area (ha)": round(n * pixel_area_ha, 1),
                "Share of AOI (%)": round(100 * n / total, 1),
            })
    df = pd.DataFrame(rows, columns=["Land cover", "Area (ha)", "Share of AOI (%)"])
    return df.sort_values("Area (ha)", ascending=False).reset_index(drop=True)


def landcover_rgb(landcover: np.ndarray) -> np.ndarray:
    """WorldCover codes as an RGB image in the product's own colours (no data = white)."""
    rgb = np.full(landcover.shape + (3,), 255, dtype=np.uint8)
    for code, (_, hex_color) in WORLDCOVER_CLASSES.items():
        rgb[landcover == code] = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return rgb

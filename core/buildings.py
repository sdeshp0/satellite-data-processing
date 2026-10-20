"""
Building exposure for a change map: how many buildings sit inside the
detected change, from Microsoft's Global ML Building Footprints
(`ms-buildings` on Planetary Computer, GeoParquet partitioned by quadkey).

A building counts as inside a class when its footprint's centroid falls in
a pixel of that class. This is exposure, not damage: a building inside a
burn scar may have survived, and the footprints come from imagery of
varying dates, so recent construction or losses may be missing.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import planetary_computer as pc
import pyarrow.parquet as pq
import shapely
import xarray as xr
from pyproj import Transformer
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping

BUILDINGS_COLLECTION = "ms-buildings"
# Each region's GeoParquet is partitioned into folders by Bing Maps
# quadkey at this zoom level (tiles ~40-80 km wide at mid-latitudes).
QUADKEY_ZOOM = 9


def _item_date(item: Any) -> str:
    """Processing date of a footprint item, from its properties or its id (e.g. "United States_2022-07-06")."""
    date = item.properties.get("msbuildings:processing-date")
    if date:
        return str(date)
    match = re.search(r"\d{4}-\d{2}-\d{2}", item.id)
    return match.group(0) if match else ""


def search_buildings(reference: xr.DataArray) -> List[Any]:
    """Footprint items covering the reference grid: each region's latest release."""
    from pystac_client import Client

    bbox = transform_bounds(reference.rio.crs, "EPSG:4326", *reference.rio.bounds())
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    items = list(catalog.search(collections=[BUILDINGS_COLLECTION], intersects=mapping(box(*bbox))).items())
    latest: Dict[str, Any] = {}
    for item in items:
        region = str(item.properties.get("msbuildings:region", ""))
        if region not in latest or _item_date(item) > _item_date(latest[region]):
            latest[region] = item
    return list(latest.values())


def quadkey(lon: float, lat: float, zoom: int = QUADKEY_ZOOM) -> str:
    """Bing Maps quadkey of the web-mercator tile holding lon/lat."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2 ** zoom
    x = min(int((lon + 180) / 360 * n), n - 1)
    y = min(int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n), n - 1)
    return "".join(str(((x >> i) & 1) + 2 * ((y >> i) & 1)) for i in range(zoom - 1, -1, -1))


def quadkeys_for_bbox(bbox: Tuple[float, float, float, float], zoom: int = QUADKEY_ZOOM) -> List[str]:
    """Quadkeys of every tile touching a lon/lat bounding box."""
    w, s, e, n = bbox
    top_left, bottom_right = quadkey(w, n, zoom), quadkey(e, s, zoom)

    def tile_xy(key: str) -> Tuple[int, int]:
        x = y = 0
        for digit in key:
            x, y = 2 * x + (int(digit) & 1), 2 * y + (int(digit) >> 1)
        return x, y

    x0, y0 = tile_xy(top_left)
    x1, y1 = tile_xy(bottom_right)
    keys = []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            keys.append("".join(str(((x >> i) & 1) + 2 * ((y >> i) & 1)) for i in range(zoom - 1, -1, -1)))
    return keys


def _filesystem(item: Any) -> Tuple[Any, str]:
    """An fsspec filesystem for the item's data asset and the asset's path on it."""
    import fsspec

    href = item.assets["data"].href
    if href.startswith("abfs://"):
        options = item.assets["data"].extra_fields.get("table:storage_options") or {}
        return fsspec.filesystem("abfs", **options), href[len("abfs://"):]
    return fsspec.filesystem("file"), href


def partition_paths(item: Any, quadkeys: Sequence[str]) -> Tuple[Any, List[str]]:
    """
    The item's quadkey partition folders for the given tiles, so only the
    AOI's part of a whole-country file is read. Tiles with no buildings have
    no folder. The folder may name the quadkey with or without leading zeros.
    """
    fs, base = _filesystem(item)
    paths = []
    for key in quadkeys:
        for name in dict.fromkeys([key, key.lstrip("0") or "0"]):
            path = f"{base}/quadkey={name}"
            if fs.exists(path):
                paths.append(path)
                break
    return fs, paths


def _read_geometries(fs: Any, path: str) -> np.ndarray:
    """One partition's footprints as shapely geometries (EPSG:4326)."""
    table = pq.read_table(path, filesystem=fs, columns=["geometry"])
    return shapely.from_wkb(table.column("geometry").to_numpy(zero_copy_only=False))


def building_pixels(items: Sequence[Any], reference: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
    """
    (rows, cols) of the grid pixel holding each building's centroid, for
    the buildings whose centroid falls inside the reference grid.
    """
    to_grid = Transformer.from_crs("EPSG:4326", reference.rio.crs, always_xy=True)
    transform = reference.rio.transform()
    height, width = reference.shape[-2:]
    w, s, e, n = transform_bounds(reference.rio.crs, "EPSG:4326", *reference.rio.bounds())

    rows, cols = [], []
    keys = quadkeys_for_bbox((w, s, e, n))
    for item in items:
        fs, paths = partition_paths(item, keys)
        for path in paths:
            geoms = _read_geometries(fs, path)
            centroids = shapely.centroid(geoms)
            lon, lat = shapely.get_x(centroids), shapely.get_y(centroids)
            near = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
            if not near.any():
                continue
            x, y = to_grid.transform(lon[near], lat[near])
            row, col = rowcol(transform, x, y)
            row, col = np.asarray(row, dtype=int), np.asarray(col, dtype=int)
            inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
            rows.append(row[inside])
            cols.append(col[inside])
    if not rows:
        return np.empty(0, int), np.empty(0, int)
    return np.concatenate(rows), np.concatenate(cols)


def building_counts(rows: np.ndarray, cols: np.ndarray, codes: np.ndarray) -> Dict[int, int]:
    """Buildings per class code at the given pixels; masked pixels (-1) are left out."""
    at = codes[rows, cols]
    values, counts = np.unique(at[at >= 0], return_counts=True)
    return {int(v): int(n) for v, n in zip(values, counts)}


def exposure_table(
    counts: Dict[int, int], labels: Dict[int, str], baseline_codes: Sequence[int]
) -> pd.DataFrame:
    """Buildings in each changed class and their share of all buildings in the change area."""
    changed = {code: counts.get(code, 0) for code in sorted(labels) if code not in baseline_codes}
    total = sum(changed.values())
    rows = [
        {
            "Class": labels[code],
            "Buildings": n,
            "Share of buildings in change area (%)": round(100 * n / total, 1),
        }
        for code, n in changed.items()
        if n
    ]
    return pd.DataFrame(rows, columns=["Class", "Buildings", "Share of buildings in change area (%)"])

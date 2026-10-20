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

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import planetary_computer as pc
import shapely
import xarray as xr
from pyproj import Transformer
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping

BUILDINGS_COLLECTION = "ms-buildings"


def search_buildings(reference: xr.DataArray) -> List[Any]:
    """Footprint partitions covering the reference grid, latest processing date only."""
    from pystac_client import Client

    bbox = transform_bounds(reference.rio.crs, "EPSG:4326", *reference.rio.bounds())
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    items = list(catalog.search(collections=[BUILDINGS_COLLECTION], intersects=mapping(box(*bbox))).items())
    if not items:
        return []
    dates = {id(item): str(item.properties.get("msbuildings:processing-date", "")) for item in items}
    latest = max(dates.values())
    return [item for item in items if dates[id(item)] == latest]


def _read_geometries(item: Any) -> np.ndarray:
    """One partition's footprints as shapely geometries (EPSG:4326)."""
    asset = item.assets["data"]
    options = asset.extra_fields.get("table:storage_options")
    kwargs = {"storage_options": options} if options else {}
    df = pd.read_parquet(asset.href, columns=["geometry"], **kwargs)
    return shapely.from_wkb(df["geometry"].to_numpy())


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
    for item in items:
        geoms = _read_geometries(item)
        centroids = shapely.centroid(geoms)
        lon, lat = shapely.get_x(centroids), shapely.get_y(centroids)
        near = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
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

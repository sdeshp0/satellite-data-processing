"""
A per-feature inventory: each connected region of a feature mask (one
water body, one built-up patch) with its area, rank by size, location and
building count, plus a plain-language summary of the whole set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyproj
import xarray as xr
from skimage import measure

from core.change import format_area
from core.findings import Finding, _SECTOR_NAMES

# What one region of each feature type is called in the summary.
REGION_NOUNS: Dict[str, Tuple[str, str]] = {
    "water": ("water body", "water bodies"),
    "urban": ("built-up patch", "built-up patches"),
    "vegetation": ("vegetated patch", "vegetated patches"),
    "bare_soil": ("bare-soil patch", "bare-soil patches"),
}
# Only the largest regions are listed; vegetation can have thousands.
MAX_REGIONS = 200
# Regions under this area are counted as "small" in the summary.
SMALL_REGION_HA = 1.0


@dataclass
class Region:
    rank: int  # 1 = largest
    label: int  # value in the labelled grid from feature_regions
    area_ha: float
    lon: float  # centroid
    lat: float
    bounds: Tuple[float, float, float, float]  # west, south, east, north (WGS84)
    sector: str  # 3x3 sector of the AOI holding the centroid, e.g. "north-west"
    buildings: Optional[int]  # building centroids inside it; None if not loaded


def _sector(row: float, col: float, shape: Tuple[int, int]) -> str:
    i = min(int(3 * row / shape[0]), 2)
    j = min(int(3 * col / shape[1]), 2)
    return _SECTOR_NAMES[i][j]


def feature_regions(
    mask: np.ndarray,
    reference: xr.DataArray,
    pixel_area_ha: float,
    building_rows: Optional[np.ndarray] = None,
    building_cols: Optional[np.ndarray] = None,
    max_regions: int = MAX_REGIONS,
) -> Tuple[List[Region], np.ndarray, np.ndarray]:
    """
    The largest `max_regions` connected regions of `mask` (8-connected),
    largest first; the labelled grid (0 = not a region); and the area in
    hectares of every region, listed or not, largest first.
    """
    labelled = measure.label(mask, connectivity=2)
    props = sorted(measure.regionprops(labelled), key=lambda p: p.area, reverse=True)
    counts = None
    if building_rows is not None and building_cols is not None:
        counts = np.bincount(labelled[building_rows, building_cols], minlength=labelled.max() + 1)

    transform = reference.rio.transform()
    to_wgs84 = pyproj.Transformer.from_crs(reference.rio.crs, "EPSG:4326", always_xy=True)

    def lonlat(row: float, col: float) -> Tuple[float, float]:
        t = transform
        x, y = t.c + col * t.a + row * t.b, t.f + col * t.d + row * t.e
        return to_wgs84.transform(x, y)

    regions = []
    for rank, p in enumerate(props[:max_regions], start=1):
        row, col = p.centroid
        lon, lat = lonlat(row + 0.5, col + 0.5)
        min_row, min_col, max_row, max_col = p.bbox
        corners = [lonlat(r, c) for r in (min_row, max_row) for c in (min_col, max_col)]
        lons, lats = zip(*corners)
        regions.append(Region(
            rank=rank,
            label=p.label,
            area_ha=float(p.area) * pixel_area_ha,
            lon=float(lon),
            lat=float(lat),
            bounds=(min(lons), min(lats), max(lons), max(lats)),
            sector=_sector(row, col, mask.shape),
            buildings=None if counts is None else int(counts[p.label]),
        ))
    return regions, labelled, np.array([p.area * pixel_area_ha for p in props], dtype=float)


def inventory_findings(
    feature_key: str, regions: List[Region], areas_ha: np.ndarray, clear_ha: float,
) -> List[Finding]:
    """One summary line for a feature's regions: count, total area, the largest, and the small ones."""
    one, many = REGION_NOUNS.get(feature_key, ("region", "regions"))
    total, total_ha = len(areas_ha), float(np.sum(areas_ha))
    if not total:
        return [Finding("headline", f"No {many} found at these settings.")]
    share = f" ({100 * total_ha / clear_ha:.0f}% of the clear AOI)" if clear_ha else ""
    if total == 1:
        return [Finding("headline", f"1 {one} covering {format_area(total_ha)}{share}, in the {regions[0].sector}.")]
    largest = regions[0]
    text = (
        f"{total:,} {many} covering {format_area(total_ha)}{share}. The largest, "
        f"{format_area(largest.area_ha)} in the {largest.sector}, is "
        f"{100 * largest.area_ha / total_ha:.0f}% of that"
    )
    small = int((areas_ha < SMALL_REGION_HA).sum())
    if small:
        text += f"; {small:,} {'is' if small == 1 else 'are'} under {SMALL_REGION_HA:g} ha"
    out = [Finding("headline", text + ".")]
    if total > len(regions):
        out.append(Finding("caveat", f"Only the {len(regions)} largest are listed below."))
    return out


def dominant_class(labelled: np.ndarray, codes: np.ndarray, regions: List[Region]) -> Dict[int, int]:
    """The most common value of `codes` (e.g. land-cover class) inside each listed region, by label."""
    out = {}
    for region in regions:
        values, counts = np.unique(codes[labelled == region.label], return_counts=True)
        out[region.label] = int(values[np.argmax(counts)])
    return out

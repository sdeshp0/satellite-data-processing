"""
Spectral feature identification: threshold a chosen index into a binary
mask, clean it up, and extract vector contours -- for highlighting things
like water bodies/rivers, urban/built-up extent, and vegetation/bare-soil
boundaries directly on top of an RGB scene.

Deliberately NOT attempting: roads, individual buildings, or anything
narrower than roughly a Sentinel-2 pixel (10m). A two-lane road is
sub-pixel at this resolution and won't reliably separate from surrounding
land cover using spectral thresholding -- that needs either much
higher-resolution imagery or a trained ML segmentation model, neither of
which this module attempts. A more realistic way to show road locations is
overlaying real road vector data (e.g. OpenStreetMap) as a reference layer
rather than deriving roads from pixels -- not implemented here.
"""

from __future__ import annotations

import inspect
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import pyproj
from scipy import ndimage
from skimage import measure, morphology


# Index + default threshold + overlay color per feature type. Colors follow
# the same convention used for the index charts elsewhere in the app
# (blue = water, orange/red = urban, green = vegetation, brown = soil).
#
# Urban and bare soil share one spectral rule ("built_or_bare": not water,
# not vegetation, NDBI above BUILT_OR_BARE_MIN_NDBI) and are told apart by
# a reference map instead (see split_built_or_bare). check_features.py
# showed why: at every test site, built-up land and bare ground had nearly
# the same NDBI and BSI, so no threshold separated them -- the old
# NDBI > 0 / BSI > 0 rules flagged ocean, fallow fields and desert as city
# and most of the city as bare soil. Their thresholds are the NDVI below
# which land counts as not vegetated.
FEATURE_TYPES: Dict[str, Dict] = {
    "water": {
        "label": "Water / Rivers",
        "index": "ndwi",
        "default_threshold": 0.0,
        "direction": "above",
        "color": "#3182bd",
    },
    "urban": {
        "label": "Urban / Built-up",
        "index": "ndvi",
        "default_threshold": 0.3,
        "direction": "below",
        "color": "#e6550d",
        "built_or_bare": True,
    },
    "vegetation": {
        "label": "Vegetation",
        "index": "ndvi",
        "default_threshold": 0.3,
        "direction": "above",
        "color": "#31a354",
    },
    "bare_soil": {
        "label": "Bare Soil",
        "index": "ndvi",
        "default_threshold": 0.3,
        "direction": "below",
        "color": "#a16928",
        "built_or_bare": True,
    },
}

# Built-up or bare land also needs NDBI above this, which drops shadows,
# snow and other dark or bright non-soil surfaces left after removing
# water and vegetation.
BUILT_OR_BARE_MIN_NDBI = -0.2
# ESA WorldCover's built-up class.
WORLDCOVER_BUILT_UP = 50

# Range an automatic (Otsu) water or vegetation threshold must fall in to
# be used; outside it the scene has no clear two-way split on that index
# (e.g. almost no water, so Otsu splits dry land in two) and the default
# threshold is used instead. Picked from check_features.py's per-class
# index percentiles: water pixels sit well above 0.35 NDWI and land well
# below -0.15; vegetation's best fixed cut-off ranged 0.2 to 0.4 by site.
AUTO_THRESHOLD_BOUNDS: Dict[str, Tuple[float, float]] = {
    "water": (-0.15, 0.35),
    "vegetation": (0.2, 0.5),
}
# Each side of an automatic split must hold at least this share of pixels.
AUTO_MIN_SIDE_SHARE = 0.02
# And the split must explain at least this share of the values' variance
# (see split_separation). A single bell curve scores about 0.64 however
# Otsu cuts it, so anything near that means there are no two groups.
AUTO_MIN_SEPARATION = 0.7


def split_separation(values: np.ndarray, threshold: float) -> float:
    """
    Share of the variance of `values` explained by splitting them at
    `threshold` (Otsu's between-class variance over total variance):
    near 1 for two tight, distant groups, about 0.64 for one bell curve.
    """
    total = float(values.var())
    if total == 0:
        return 0.0
    high = values > threshold
    share = float(high.mean())
    if share in (0.0, 1.0):
        return 0.0
    gap = float(values[high].mean() - values[~high].mean())
    return share * (1 - share) * gap * gap / total


def auto_threshold(
    values: np.ndarray, default: float, bounds: Tuple[float, float]
) -> Tuple[float, bool]:
    """
    Otsu's threshold for `values` (finite ones, clipped to -1..1), and
    True; or (`default`, False) when it falls outside `bounds`, leaves
    less than AUTO_MIN_SIDE_SHARE of the pixels on either side, or
    separates them less than AUTO_MIN_SEPARATION.
    """
    from skimage.filters import threshold_otsu

    values = np.clip(values[np.isfinite(values)], -1.0, 1.0)
    if values.size < 100 or values.min() == values.max():
        return default, False
    threshold = float(threshold_otsu(values))
    above = float(np.mean(values > threshold))
    if (
        not bounds[0] <= threshold <= bounds[1]
        or min(above, 1 - above) < AUTO_MIN_SIDE_SHARE
        or split_separation(values, threshold) < AUTO_MIN_SEPARATION
    ):
        return default, False
    return threshold, True


def scene_thresholds(indices: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, bool]]:
    """
    This scene's water (NDWI) and vegetation (NDVI) thresholds, each as
    (threshold, picked automatically). Vegetation's is picked among
    non-water pixels, so water doesn't pull it down. Urban and bare soil
    use both: not water, and NDVI below the vegetation threshold.
    """
    water = auto_threshold(
        indices["ndwi"], FEATURE_TYPES["water"]["default_threshold"], AUTO_THRESHOLD_BOUNDS["water"]
    )
    with np.errstate(invalid="ignore"):
        land_ndvi = indices["ndvi"][~(indices["ndwi"] > water[0])]
    vegetation = auto_threshold(
        land_ndvi, FEATURE_TYPES["vegetation"]["default_threshold"], AUTO_THRESHOLD_BOUNDS["vegetation"]
    )
    return {"water": water, "vegetation": vegetation}


def built_or_bare_raw(
    indices: Dict[str, np.ndarray], ndvi_below: float, water_threshold: float
) -> np.ndarray:
    """
    Pixels that are neither water (NDWI above `water_threshold`) nor
    vegetation (NDVI at or above `ndvi_below`), with NDBI above
    BUILT_OR_BARE_MIN_NDBI: built-up land or bare ground, before cleanup.
    """
    with np.errstate(invalid="ignore"):
        return (
            np.isfinite(indices["ndvi"])
            & ~(indices["ndwi"] > water_threshold)
            & (indices["ndvi"] < ndvi_below)
            & (indices["ndbi"] > BUILT_OR_BARE_MIN_NDBI)
        )


def built_reference(
    landcover: Optional[np.ndarray],
    building_rows: Optional[np.ndarray] = None,
    building_cols: Optional[np.ndarray] = None,
    radius_px: int = 1,
) -> Optional[np.ndarray]:
    """
    Where land counts as built-up: WorldCover's built-up class, plus any
    pixel within `radius_px` of a building footprint's centre (catching
    building since WorldCover's snapshot). None if neither is available.
    """
    if landcover is None and building_rows is None:
        return None
    shape = landcover.shape if landcover is not None else None
    built = landcover == WORLDCOVER_BUILT_UP if landcover is not None else None
    if building_rows is not None and building_cols is not None and len(building_rows):
        if shape is None:
            return None  # the grid's shape comes from the land cover
        near = np.zeros(shape, dtype=bool)
        near[building_rows, building_cols] = True
        near = ndimage.binary_dilation(near, np.ones((2 * radius_px + 1,) * 2, dtype=bool))
        built = built | near
    return built


def split_built_or_bare(
    raw: np.ndarray, reference: Optional[np.ndarray], feature_key: str
) -> np.ndarray:
    """
    The urban or bare-soil part of a built-or-bare mask: inside the built
    reference for "urban", outside it for "bare_soil". Without a
    reference the two can't be told apart and the whole mask is returned.
    """
    if reference is None:
        return raw
    return raw & reference if feature_key == "urban" else raw & ~reference


def threshold_mask(
    arr: np.ndarray,
    threshold: float,
    direction: str = "above",
    min_region_px: int = 20,
) -> np.ndarray:
    """
    Threshold a continuous index array into a cleaned boolean mask.

    Parameters
    ----------
    arr : np.ndarray
        Index array (e.g. NDWI, NDBI). NaNs (masked/invalid pixels) are
        excluded from the feature regardless of threshold.
    threshold : float
        Cutoff value.
    direction : str
        "above": pixels > threshold are the feature (e.g. water via NDWI).
        "below": pixels < threshold are the feature.
    min_region_px : int
        Connected regions smaller than this many pixels are dropped as
        noise, and holes up to this size are filled -- a handful of stray
        pixels crossing a threshold isn't a real river or building, just
        spectral noise.

    Returns
    -------
    np.ndarray (bool)
        Cleaned boolean mask, same shape as arr.
    """
    valid = ~np.isnan(arr)
    if direction == "below":
        raw_mask = valid & (arr < threshold)
    else:
        raw_mask = valid & (arr > threshold)
    return clean_mask(raw_mask, min_region_px)


def clean_mask(mask: np.ndarray, min_region_px: int = 20) -> np.ndarray:
    """Drop connected regions and fill holes smaller than `min_region_px` pixels."""
    cleaned = _remove_small(morphology.remove_small_objects, mask, min_region_px)
    return _remove_small(morphology.remove_small_holes, cleaned, min_region_px)


def _remove_small(func, mask: np.ndarray, min_region_px: int) -> np.ndarray:
    """
    Call a scikit-image remove_small_* function so it drops regions of
    fewer than min_region_px pixels, across scikit-image versions.

    0.26 deprecated min_size / area_threshold ("remove if < n") in favour
    of max_size ("remove if <= n"), so the new keyword gets n - 1.
    """
    if "max_size" in inspect.signature(func).parameters:
        return func(mask, max_size=max(0, min_region_px - 1))
    if func is morphology.remove_small_holes:
        return func(mask, area_threshold=min_region_px)
    return func(mask, min_size=min_region_px)


def extract_contours(mask: np.ndarray) -> List[np.ndarray]:
    """
    Extract boundary contours from a boolean mask via marching squares.

    Returns
    -------
    list of np.ndarray
        Each element is an (N, 2) array of (row, col) pixel coordinates
        tracing one contour.
    """
    return measure.find_contours(mask.astype(float), level=0.5)


def contour_to_lonlat(
    contour: np.ndarray, reference: xr.DataArray
) -> List[Tuple[float, float]]:
    """
    Convert a contour's (row, col) pixel coordinates into (lon, lat)
    geographic coordinates, using the reference band's affine transform
    and CRS.
    """
    return _contour_to_lonlat(contour, reference.rio.transform(), _to_wgs84(reference.rio.crs))


def _to_wgs84(crs) -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def _contour_to_lonlat(
    contour: np.ndarray, transform, to_wgs84: pyproj.Transformer
) -> List[Tuple[float, float]]:
    # find_contours returns pixel-centre coordinates (pixel (0, 0)'s centre
    # is (0, 0)), while the affine transform maps (0, 0) to the pixel's
    # top-left corner -- hence the half-pixel shift before transforming.
    if len(contour) == 0:
        return []
    rows = contour[:, 0] + 0.5
    cols = contour[:, 1] + 0.5
    xs = transform.c + cols * transform.a + rows * transform.b
    ys = transform.f + cols * transform.d + rows * transform.e
    lons, lats = to_wgs84.transform(xs, ys)
    return list(zip(np.asarray(lons).tolist(), np.asarray(lats).tolist()))


def contours_to_geojson(
    contours_by_feature: Dict[str, List[np.ndarray]],
    reference: xr.DataArray,
) -> Dict:
    """
    Package contours for multiple feature types into a single GeoJSON
    FeatureCollection (WGS84 LineStrings), one Feature per contour, tagged
    with a "feature_type" property.
    """
    transform = reference.rio.transform()
    to_wgs84 = _to_wgs84(reference.rio.crs)

    features = []
    for feature_key, contours in contours_by_feature.items():
        for contour in contours:
            coords = _contour_to_lonlat(contour, transform, to_wgs84)
            if len(coords) < 2:
                continue
            features.append({
                "type": "Feature",
                "properties": {"feature_type": feature_key},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lon, lat in coords],
                },
            })

    return {"type": "FeatureCollection", "features": features}


def region_stats(mask: np.ndarray, reference: xr.DataArray) -> Dict[str, float]:
    """
    Basic stats for a feature mask: number of distinct regions and total
    area in km^2, using the reference band's pixel resolution.
    """
    labeled = measure.label(mask)
    num_regions = int(labeled.max())

    res_x, res_y = reference.rio.resolution()
    pixel_area_km2 = abs(res_x * res_y) / 1e6
    total_area_km2 = float(mask.sum()) * pixel_area_km2

    return {"num_regions": num_regions, "area_km2": total_area_km2}

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

from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
import pyproj
from skimage import measure, morphology


# Index + default threshold + overlay color per feature type. Colors follow
# the same convention used for the index charts elsewhere in the app
# (blue = water, orange/red = urban, green = vegetation, brown = soil).
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
        "index": "ndbi",
        "default_threshold": 0.0,
        "direction": "above",
        "color": "#e6550d",
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
        "index": "bsi",
        "default_threshold": 0.0,
        "direction": "above",
        "color": "#a16928",
    },
}


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

    cleaned = morphology.remove_small_objects(raw_mask, min_size=min_region_px)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_region_px)
    return cleaned


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
    transform = reference.rio.transform()
    crs = reference.rio.crs
    to_wgs84 = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    points = []
    for row, col in contour:
        x, y = transform * (col, row)
        lon, lat = to_wgs84.transform(x, y)
        points.append((lon, lat))
    return points


def contours_to_geojson(
    contours_by_feature: Dict[str, List[np.ndarray]],
    reference: xr.DataArray,
) -> Dict:
    """
    Package contours for multiple feature types into a single GeoJSON
    FeatureCollection (WGS84 LineStrings), one Feature per contour, tagged
    with a "feature_type" property.
    """
    features = []
    for feature_key, contours in contours_by_feature.items():
        for contour in contours:
            coords = contour_to_lonlat(contour, reference)
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

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
from dataclasses import dataclass
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

    return clean_mask(raw_mask, min_region_px)


def clean_mask(mask: np.ndarray, min_region_px: int = 20) -> np.ndarray:
    """
    Drop connected regions smaller than min_region_px pixels and fill holes
    up to that size, so stray pixels don't read as features.
    """
    cleaned = _remove_small(morphology.remove_small_objects, mask.astype(bool), min_region_px)
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


@dataclass
class ChangeCluster:
    """An ellipse (pixel coordinates) enclosing one cluster of changed pixels."""
    row: float
    col: float
    semi_major: float
    semi_minor: float
    orientation: float  # radians, skimage regionprops convention
    pixels: int         # changed pixels in the cluster


def change_clusters(
    mask: np.ndarray,
    max_clusters: int = 5,
    min_share: float = 0.05,
    merge_frac: float = 0.05,
    pad: float = 1.25,
    min_region_px: int = 10,
) -> List[ChangeCluster]:
    """
    Group changed pixels into a few broad clusters and fit an enclosing
    ellipse to each, largest first.

    Pixels within merge_frac of the image size of each other join the same
    cluster, so fragmented change (e.g. a patchy burn) reads as one area.
    Clusters holding less than min_share of all changed pixels are dropped.
    Each ellipse matches the cluster's second moments, scaled by `pad` and
    given a minimum radius so small clusters stay visible. Specks under
    min_region_px are removed first, and a cluster spanning most of the
    image in both directions is skipped, since change spread across the
    whole AOI has no area worth circling.
    """
    mask = clean_mask(mask, min_region_px)
    total = int(mask.sum())
    if total == 0:
        return []

    size = max(mask.shape)
    merge_px = max(1, int(round(merge_frac * size)))
    grouped = morphology.dilation(
        mask, morphology.disk(merge_px, decomposition="sequence")
    )
    labels = measure.label(grouped) * mask  # cluster ids on changed pixels only

    min_radius = 0.02 * size
    clusters = []
    for region in measure.regionprops(labels):
        if region.area < min_share * total:
            continue
        height = region.bbox[2] - region.bbox[0]
        width = region.bbox[3] - region.bbox[1]
        if height > 0.8 * mask.shape[0] and width > 0.8 * mask.shape[1]:
            continue
        clusters.append(ChangeCluster(
            row=region.centroid[0],
            col=region.centroid[1],
            semi_major=max(min_radius, pad * region.axis_major_length / 2),
            semi_minor=max(min_radius, pad * region.axis_minor_length / 2),
            orientation=region.orientation,
            pixels=int(region.area),
        ))
    clusters.sort(key=lambda c: c.pixels, reverse=True)
    return clusters[:max_clusters]

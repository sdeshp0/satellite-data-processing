"""
Change hotspots: the few areas where detected change is concentrated.

Real change classifications mix coherent change areas (a burn scar, a
flooded plain) with misclassified pixels scattered across the AOI, so
hotspots are found by local density rather than connectivity: a pixel is
in a hotspot when enough of the window around it changed. Patchy change
still forms a hotspot; thin scattered noise does not.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
from skimage import measure

from core.features import _contour_to_lonlat, _to_wgs84


def _box_mean(mask: np.ndarray, radius: int) -> np.ndarray:
    """Fraction of True pixels in a (2r+1)-square window around each pixel."""
    padded = np.pad(mask.astype(np.float64), radius + 1, mode="constant")
    integral = padded.cumsum(0).cumsum(1)
    k = 2 * radius + 1
    h, w = mask.shape
    total = (
        integral[k:k + h, k:k + w] - integral[:h, k:k + w]
        - integral[k:k + h, :w] + integral[:h, :w]
    )
    return total / (k * k)


def find_hotspots(
    changed: np.ndarray,
    max_hotspots: int = 5,
    min_share: float = 0.05,
    window_frac: float = 0.03,
    min_density: float = 0.3,
) -> np.ndarray:
    """
    Label hotspot areas 1..n, largest (by changed pixels) first; 0 elsewhere.

    A pixel is in a hotspot when at least min_density of a square window
    (window_frac of the image size) around it changed, or 1.5x the AOI-wide
    changed fraction if that is higher, so widespread noise doesn't count
    as dense. Hotspots holding under min_share of the changed pixels in
    all dense areas are
    dropped, as is any spanning most of the image in both directions, since
    there's no single area to point at. Dense spots smaller than one window
    are treated as chance clumps of noise.
    """
    changed = changed.astype(bool)
    labels_out = np.zeros(changed.shape, dtype=np.int32)
    total = int(changed.sum())
    if total == 0:
        return labels_out

    radius = max(1, int(round(window_frac * max(changed.shape) / 2)))
    threshold = min(0.7, max(min_density, 1.5 * changed.mean()))
    dense = _box_mean(changed, radius) >= threshold
    labels = measure.label(dense)
    dense_total = int((dense & changed).sum())

    candidates = []
    for region in measure.regionprops(labels):
        # Chance clumps of noise make dense spots smaller than one window.
        if region.area < (2 * radius + 1) ** 2:
            continue
        n_changed = int(changed[labels == region.label].sum())
        if n_changed < min_share * dense_total:
            continue
        r0, c0, r1, c1 = region.bbox
        if r1 - r0 > 0.8 * changed.shape[0] and c1 - c0 > 0.8 * changed.shape[1]:
            continue
        candidates.append((n_changed, region.label))

    candidates.sort(reverse=True)
    for rank, (_, label) in enumerate(candidates[:max_hotspots], start=1):
        labels_out[labels == label] = rank
    return labels_out


def hotspot_table(
    hotspots: np.ndarray,
    changed: np.ndarray,
    reference: xr.DataArray,
    pixel_area_ha: float,
) -> pd.DataFrame:
    """
    One row per hotspot: changed area inside it, its share of all detected
    change, and the latitude/longitude of the changed pixels' centre.
    """
    total = int(changed.sum())
    transform = reference.rio.transform()
    to_wgs84 = _to_wgs84(reference.rio.crs)
    rows = []
    for rank in range(1, int(hotspots.max()) + 1):
        inside = (hotspots == rank) & changed
        n = int(inside.sum())
        r, c = np.nonzero(inside)
        lon, lat = _contour_to_lonlat(np.array([[r.mean(), c.mean()]]), transform, to_wgs84)[0]
        rows.append({
            "Hotspot": rank,
            "Changed area (ha)": round(n * pixel_area_ha, 1),
            "Share of detected change (%)": round(100 * n / total, 1) if total else 0.0,
            "Latitude": round(lat, 5),
            "Longitude": round(lon, 5),
        })
    return pd.DataFrame(rows)


def hotspots_geojson(hotspots: np.ndarray, reference: xr.DataArray) -> Dict:
    """Hotspot outlines as a WGS84 GeoJSON FeatureCollection of Polygons."""
    transform = reference.rio.transform()
    to_wgs84 = _to_wgs84(reference.rio.crs)
    features = []
    for rank in range(1, int(hotspots.max()) + 1):
        # Pad so outlines touching the image edge still close.
        padded = np.pad(hotspots == rank, 1).astype(float)
        for contour in measure.find_contours(padded, 0.5):
            coords = _contour_to_lonlat(contour - 1, transform, to_wgs84)
            if len(coords) < 4:
                continue
            features.append({
                "type": "Feature",
                "properties": {"hotspot": rank},
                "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in coords]]},
            })
    return {"type": "FeatureCollection", "features": features}

"""
Before/after change-detection logic: grid alignment, delta computation, and
event-specific classification using established remote-sensing methodologies.

- Wildfire: dNBR classified per USGS / Key & Benson (2006) burn-severity
  thresholds -- the standard reference scale used by USGS's Burn Severity
  Portal and the Monitoring Trends in Burn Severity (MTBS) program.
- Flood: NDWI > 0 water threshold (McFeeters, 1996 -- the original NDWI
  paper's own cutoff), applied per date, then differenced.
- Logging / Deforestation: NDVI > 0.3 vegetation threshold, applied per
  date, then differenced. This is a widely used convention, but -- unlike
  dNBR above -- not as rigidly standardized in the literature; treat it as
  a reasonable default rather than a definitive industry figure.
- Custom: statistical (k-sigma) thresholding on the raw delta -- flags
  pixels beyond k standard deviations from the AOI's own mean delta. A
  standard, index-agnostic image-differencing change-detection method,
  usable for any of the 8 indices without a domain-specific threshold.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401  # keep for .rio accessor
from rasterio.enums import Resampling


CLOUD_CLASSES = [3, 8, 9, 10, 11]  # matches core.utils.apply_cloud_mask


# --- Grid alignment ---------------------------------------------------------

def _to_georeferenced(arr: np.ndarray, reference: xr.DataArray) -> xr.DataArray:
    da = xr.DataArray(arr, dims=reference.dims, coords=reference.coords)
    da = da.rio.write_crs(reference.rio.crs)
    da = da.rio.write_transform(reference.rio.transform())
    return da


def align_to_reference(
    arr: np.ndarray,
    source_ref: xr.DataArray,
    target_ref: xr.DataArray,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """
    Reproject/resample `arr` (defined on source_ref's grid) onto
    target_ref's grid.

    Two independently loaded scenes over the "same" AOI rarely share
    pixel-for-pixel identical grids (different orbit paths, independent
    decimated reads, etc.), so a before/after pair needs this step before
    the arrays can be directly differenced.
    """
    da = _to_georeferenced(arr, source_ref)
    aligned = da.rio.reproject_match(target_ref, resampling=resampling)
    return aligned.values


def combined_valid_mask(scl_before: xr.DataArray, scl_after: xr.DataArray) -> np.ndarray:
    """
    Boolean mask (True = usable), on scl_before's grid, requiring a pixel to
    be cloud/shadow-free in BOTH dates -- a pixel that's clear before but
    cloudy after (or vice versa) can't support a real comparison.
    """
    scl_after_aligned = align_to_reference(
        scl_after.values, scl_after, scl_before, resampling=Resampling.nearest
    )
    clear_before = ~np.isin(scl_before.values.astype(int), CLOUD_CLASSES)
    clear_after = ~np.isin(scl_after_aligned.astype(int), CLOUD_CLASSES)
    return clear_before & clear_after


# --- Delta -------------------------------------------------------------------

def compute_delta(
    index_before: np.ndarray,
    index_after: np.ndarray,
    direction: str = "post_minus_pre",
) -> np.ndarray:
    """
    direction="post_minus_pre": after - before (positive = increase).
    direction="pre_minus_post": before - after (positive = decrease from
    before to after -- e.g. dNBR, where a positive value conventionally
    means more burn, since burned pixels have LOWER post-fire NBR).
    """
    if direction == "pre_minus_post":
        return index_before - index_after
    return index_after - index_before


# --- Classification: Wildfire (dNBR) -----------------------------------------

DNBR_LABELS: Dict[int, str] = {
    0: "Enhanced Regrowth, High",
    1: "Enhanced Regrowth, Low",
    2: "Unburned",
    3: "Low Severity",
    4: "Moderate-Low Severity",
    5: "Moderate-High Severity",
    6: "High Severity",
}
DNBR_COLORS: Dict[int, str] = {
    0: "#1a9850",
    1: "#66bd63",
    2: "#ffffbf",
    3: "#fee08b",
    4: "#fdae61",
    5: "#f46d43",
    6: "#a50026",
}
# (lower_bound, upper_bound, class_code] -- USGS / Key & Benson (2006)
_DNBR_BINS = [
    (-np.inf, -0.25, 0),
    (-0.25, -0.10, 1),
    (-0.10, 0.10, 2),
    (0.10, 0.27, 3),
    (0.27, 0.44, 4),
    (0.44, 0.66, 5),
    (0.66, np.inf, 6),
]


def classify_dnbr(nbr_before: np.ndarray, nbr_after: np.ndarray) -> np.ndarray:
    """
    Classify burn severity using dNBR = NBR_before - NBR_after, per the
    USGS / Key & Benson (2006) FIREMON thresholds.
    """
    dnbr = nbr_before - nbr_after
    codes = np.full(dnbr.shape, -1, dtype=int)
    for lo, hi, code in _DNBR_BINS:
        codes[(dnbr > lo) & (dnbr <= hi)] = code
    return codes


# --- Classification: Flood (NDWI) ---------------------------------------------

WATER_LABELS: Dict[int, str] = {
    0: "No Water (Either Date)",
    1: "Persistent Water",
    2: "New Water (Flood)",
    3: "Water Loss (Recession)",
}
WATER_COLORS: Dict[int, str] = {
    0: "#f0f0f0",
    1: "#3182bd",
    2: "#e31a1c",
    3: "#fdae6b",
}


def classify_water_change(
    ndwi_before: np.ndarray, ndwi_after: np.ndarray, threshold: float = 0.0
) -> np.ndarray:
    """
    Classify water/non-water per date using NDWI > 0 (McFeeters, 1996),
    then compare the two dates' classifications.
    """
    codes = np.full(ndwi_before.shape, -1, dtype=int)
    valid = ~np.isnan(ndwi_before) & ~np.isnan(ndwi_after)
    water_before = ndwi_before > threshold
    water_after = ndwi_after > threshold

    codes[valid & ~water_before & ~water_after] = 0
    codes[valid & water_before & water_after] = 1
    codes[valid & ~water_before & water_after] = 2
    codes[valid & water_before & ~water_after] = 3
    return codes


# --- Classification: Logging / Deforestation (NDVI) ----------------------------

VEG_LABELS: Dict[int, str] = {
    0: "Non-Vegetated (Stable)",
    1: "Vegetated (Stable)",
    2: "Vegetation Loss",
    3: "Vegetation Gain",
}
VEG_COLORS: Dict[int, str] = {
    0: "#d9d9d9",
    1: "#1a9850",
    2: "#d73027",
    3: "#91cf60",
}


def classify_vegetation_loss(
    ndvi_before: np.ndarray, ndvi_after: np.ndarray, threshold: float = 0.3
) -> np.ndarray:
    """
    Classify vegetated/non-vegetated per date using NDVI > 0.3 (a commonly
    used, though not rigidly standardized, cutoff for "dense vegetation"),
    then compare the two dates.
    """
    codes = np.full(ndvi_before.shape, -1, dtype=int)
    valid = ~np.isnan(ndvi_before) & ~np.isnan(ndvi_after)
    veg_before = ndvi_before > threshold
    veg_after = ndvi_after > threshold

    codes[valid & ~veg_before & ~veg_after] = 0
    codes[valid & veg_before & veg_after] = 1
    codes[valid & veg_before & ~veg_after] = 2
    codes[valid & ~veg_before & veg_after] = 3
    return codes


# --- Classification: Custom / Generic (statistical) -----------------------------

STAT_LABELS: Dict[int, str] = {
    0: "No Significant Change",
    1: "Significant Increase",
    2: "Significant Decrease",
}
STAT_COLORS: Dict[int, str] = {
    0: "#f0f0f0",
    1: "#d73027",
    2: "#4575b4",
}


def classify_statistical(
    index_before: np.ndarray, index_after: np.ndarray, k: float = 2.0
) -> np.ndarray:
    """
    Flag pixels whose delta falls beyond k standard deviations from the
    AOI's own mean delta (k-sigma thresholding).
    """
    delta = index_after - index_before
    codes = np.full(delta.shape, -1, dtype=int)
    valid = ~np.isnan(delta)
    valid_vals = delta[valid]

    if valid_vals.size == 0:
        return codes

    mean = float(np.mean(valid_vals))
    std = float(np.std(valid_vals))

    codes[valid] = 0
    codes[valid & (delta > mean + k * std)] = 1
    codes[valid & (delta < mean - k * std)] = 2
    return codes


# --- Event presets -------------------------------------------------------------
# "index": None means the user picks any of the 8 indices in the UI.
# "baseline_codes": class codes counted as "no meaningful change" when
# computing the overall % of AOI flagged as changed.

EVENT_PRESETS: Dict[str, Dict] = {
    "wildfire": {
        "label": "Wildfire — Burn Severity",
        "index": "nbr",
        "direction": "pre_minus_post",
        "methodology": (
            "dNBR (NBR before minus NBR after), classified per USGS / "
            "Key & Benson (2006) burn-severity thresholds."
        ),
        "classify": classify_dnbr,
        "labels": DNBR_LABELS,
        "colors": DNBR_COLORS,
        "baseline_codes": [2],
    },
    "flood": {
        "label": "Flood — Surface Water Change",
        "index": "ndwi",
        "direction": "post_minus_pre",
        "methodology": (
            "NDWI > 0 water threshold (McFeeters, 1996) applied per date, "
            "then differenced to find new/lost water."
        ),
        "classify": classify_water_change,
        "labels": WATER_LABELS,
        "colors": WATER_COLORS,
        "baseline_codes": [0, 1],
    },
    "logging": {
        "label": "Logging / Deforestation — Vegetation Loss",
        "index": "ndvi",
        "direction": "pre_minus_post",
        "methodology": (
            "NDVI > 0.3 vegetation threshold applied per date, then "
            "differenced to find vegetation loss/gain."
        ),
        "classify": classify_vegetation_loss,
        "labels": VEG_LABELS,
        "colors": VEG_COLORS,
        "baseline_codes": [0, 1],
    },
    "custom": {
        "label": "Custom — Any Index",
        "index": None,
        "direction": "post_minus_pre",
        "methodology": (
            "Statistical (k-sigma) thresholding: pixels beyond \u00b12 "
            "standard deviations from the AOI's mean delta."
        ),
        "classify": classify_statistical,
        "labels": STAT_LABELS,
        "colors": STAT_COLORS,
        "baseline_codes": [0],
    },
}


def class_breakdown(codes: np.ndarray, labels: Dict[int, str]) -> pd.DataFrame:
    """
    Percentage of valid (non-masked) AOI pixels in each classification code.
    """
    values = codes.ravel()
    valid = values[values >= 0]
    total = valid.size

    rows = []
    for code, label in labels.items():
        count = int(np.sum(values == code))
        if count == 0:
            continue
        rows.append({
            "Class": label,
            "Pixels": count,
            "Percent": round(100 * count / total, 1) if total else 0.0,
        })

    return pd.DataFrame(rows)

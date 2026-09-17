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

Also includes before/after *comparability* checks (core.change.comparability_checks
and comparability_score) -- a scene pair can be geometrically alignable
(see align_bands below) while still not being apples-to-apples: different
UTM zones/tiles, different platforms, different season, or different sun
angle can all introduce a signal that looks like "change" but isn't. These
checks are shared between app/components/change_display.py (surfacing
warnings for a pair the user already picked) and
app/components/sample_analyses.py / scene_selector.py (scoring/selecting
or flagging pairs before comparison even runs).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from typing import Any, Dict, List, Optional, Tuple

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


def align_bands(
    bands: Dict[str, xr.DataArray],
    target_ref: xr.DataArray,
) -> Dict[str, xr.DataArray]:
    """
    Reproject every band in `bands` onto target_ref's grid, returning a new
    dict with the same keys.

    This must run on the RAW BANDS, not just a derived index -- aligning
    only a computed index (as an earlier version of this module did) leaves
    the RGB preview built from each scene's own independent grid. Two
    scenes over the "same" AOI can come from different UTM zones, which
    reproject the same AOI polygon to different pixel dimensions/aspect
    ratios per scene -- both get force-displayed at the same width, so one
    visibly looks stretched relative to the other. Aligning bands once,
    upstream of both the RGB render and the index computation, fixes this
    at the source instead of patching around it downstream.

    Uses nearest-neighbor resampling for "scl" (categorical class codes)
    and bilinear for continuous reflectance bands.
    """
    aligned: Dict[str, xr.DataArray] = {}
    for name, da in bands.items():
        resampling = Resampling.nearest if name == "scl" else Resampling.bilinear
        values = align_to_reference(da.values, da, target_ref, resampling=resampling)
        aligned[name] = _to_georeferenced(values, target_ref)
    return aligned


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


# --- Comparability checks ---------------------------------------------------
# Whether a before/after pair is really apples-to-apples, independent of
# whether their grids can be geometrically aligned (align_bands handles
# that separately). Consolidated here (rather than left inline in
# change_display.py) so the same logic can also *score* candidate pairs
# before one is picked -- see comparability_score, used by
# sample_analyses.py -- and so scene_selector.py's picker table can surface
# it too.

@dataclass
class ComparabilityNote:
    """One comparability finding for a before/after scene pair."""
    severity: str  # "warning" (likely to matter) or "info" (worth knowing)
    message: str


# Seasonal (day-of-year) distance thresholds, in days. Below INFO, no note
# at all. Between INFO and WARNING, a caption -- worth knowing but not
# necessarily a problem (many legitimate comparisons, e.g. a multi-year
# deforestation pair, are seasonally close despite being years apart in
# raw date). At or above WARNING, normal vegetation cycling / wet-dry
# season / snow cover differences are large enough to plausibly be mistaken
# for the event itself.
SEASONAL_INFO_DAYS = 20
SEASONAL_WARNING_DAYS = 45

# Sun elevation angle (degrees) thresholds. A large difference changes
# shadow length/direction and terrain shading independent of any real
# surface change -- most visible in hilly terrain or wherever tall
# vegetation/structures cast shadows across the AOI.
SUN_ELEV_INFO_DEG = 10.0
SUN_ELEV_WARNING_DEG = 20.0


def day_of_year_distance(date_before: _date, date_after: _date) -> int:
    """
    Circular day-of-year distance between two dates, ignoring year -- e.g.
    Dec 28 vs Jan 5 is 8 days apart seasonally, not ~358. This is what
    matters for phenological/seasonal comparability, as opposed to the
    literal elapsed time between the two acquisitions (which is often
    large and *expected*, e.g. a deforestation comparison spanning several
    years by design).
    """
    doy_before = date_before.timetuple().tm_yday
    doy_after = date_after.timetuple().tm_yday
    diff = abs(doy_after - doy_before)
    return min(diff, 365 - diff)


def sun_elevation_diff(before_item, after_item) -> Optional[float]:
    """
    Absolute difference in view:sun_elevation (degrees) between two STAC
    items, or None if either item lacks the field.
    """
    before_elev = before_item.properties.get("view:sun_elevation")
    after_elev = after_item.properties.get("view:sun_elevation")
    if before_elev is None or after_elev is None:
        return None
    return abs(float(after_elev) - float(before_elev))


def comparability_checks(before_item, after_item) -> List[ComparabilityNote]:
    """
    Run all before/after comparability checks and return their findings, in
    a fixed order (roughly most-likely-to-matter first): UTM zone, MGRS
    tile, platform, seasonal (day-of-year) distance, sun elevation.

    Parameters
    ----------
    before_item, after_item : pystac.Item

    Returns
    -------
    list of ComparabilityNote
    """
    notes: List[ComparabilityNote] = []

    before_epsg = before_item.properties.get("proj:epsg")
    after_epsg = after_item.properties.get("proj:epsg")
    before_tile = before_item.properties.get("s2:mgrs_tile")
    after_tile = after_item.properties.get("s2:mgrs_tile")

    if before_epsg is not None and after_epsg is not None and before_epsg != after_epsg:
        notes.append(ComparabilityNote(
            "warning",
            f"Before (EPSG:{before_epsg}) and after (EPSG:{after_epsg}) scenes "
            "are in different UTM zones -- the after scene has been "
            "reprojected onto the before scene's grid to allow comparison, "
            "which introduces some resampling.",
        ))
    elif before_tile is not None and after_tile is not None and before_tile != after_tile:
        notes.append(ComparabilityNote(
            "info",
            f"Before ({before_tile}) and after ({after_tile}) scenes come "
            "from different Sentinel-2 MGRS tiles (same UTM zone, "
            "different source granule).",
        ))

    before_platform = before_item.properties.get("platform", "unknown")
    after_platform = after_item.properties.get("platform", "unknown")
    if before_platform != after_platform:
        notes.append(ComparabilityNote(
            "info",
            f"Before ({before_platform}) and after ({after_platform}) "
            "scenes come from different Sentinel-2 satellites. ESA applies "
            "a small (~1.1%) cross-calibration correction between them, "
            "which isn't independently corrected for here -- unlikely to "
            "be the dominant signal in a dramatic change, but worth "
            "keeping in mind for subtle comparisons.",
        ))

    doy_dist = day_of_year_distance(before_item.datetime.date(), after_item.datetime.date())
    if doy_dist >= SEASONAL_WARNING_DAYS:
        notes.append(ComparabilityNote(
            "warning",
            f"Before and after scenes are ~{doy_dist} days apart in the "
            "seasonal calendar (ignoring year). Normal vegetation cycling, "
            "snow cover, or wet/dry season differences can look like the "
            "event itself -- consider picking scenes closer in "
            "day-of-year, if available.",
        ))
    elif doy_dist >= SEASONAL_INFO_DAYS:
        notes.append(ComparabilityNote(
            "info",
            f"Before and after scenes are ~{doy_dist} days apart in the "
            "seasonal calendar (ignoring year) -- worth keeping in mind "
            "for anything sensitive to seasonal vegetation change.",
        ))

    elev_diff = sun_elevation_diff(before_item, after_item)
    if elev_diff is not None:
        if elev_diff >= SUN_ELEV_WARNING_DEG:
            notes.append(ComparabilityNote(
                "warning",
                f"Sun elevation differs by ~{elev_diff:.0f}\u00b0 between "
                "the two scenes. This changes shadow length/direction and "
                "terrain shading independent of any real surface change -- "
                "most noticeable in hilly terrain or areas with tall "
                "vegetation/structures.",
            ))
        elif elev_diff >= SUN_ELEV_INFO_DEG:
            notes.append(ComparabilityNote(
                "info",
                f"Sun elevation differs by ~{elev_diff:.0f}\u00b0 between "
                "the two scenes -- shadows will look somewhat different "
                "between dates.",
            ))

    return notes


def select_best_pair(
    before_items: List, after_items: List
) -> Optional[Tuple[Any, Any]]:
    """
    Choose the before/after pair, from two independently-searched result
    lists, with the best comparability_score: lowest seasonal (day-of-
    year) distance first, then lowest sun-elevation difference, with
    combined cloud cover as the final tiebreaker.

    Shared by two call sites: the Change Detection page uses this to
    automatically pre-select a pair as soon as both searches return
    results (the user can still override via either scene_picker table),
    and sample_analyses.py uses it for the one-click samples. Picking each
    side's lowest-cloud scene independently (the simpler alternative) can
    easily land on a pair that's individually clear but seasonally or
    illumination-mismatched -- exactly what comparability_checks warns
    about once a pair is actually loaded.

    Runs in O(len(before_items) * len(after_items)); both lists are
    typically small (a few dozen items at most from one search window), so
    this stays fast in practice.

    Returns
    -------
    tuple(pystac.Item, pystac.Item) or None
        (best_before, best_after), ranked by comparability_score, or None
        if either list is empty.
    """
    if not before_items or not after_items:
        return None

    best_pair: Optional[Tuple[Any, Any]] = None
    best_score: Optional[Tuple[float, float, float]] = None

    for before in before_items:
        for after in after_items:
            score = comparability_score(before, after)
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (before, after)

    return best_pair


def comparability_score(before_item, after_item) -> Tuple[float, float, float]:
    """
    Sortable score for ranking candidate before/after pairs -- lower is
    better on every component, and the tuple is meant to be used directly
    with min()/sorted(). Used by sample_analyses.py to pick a pair that is
    seasonally/illumination-matched, rather than picking each side's
    lowest-cloud scene independently (which could pair a summer "before"
    with a winter "after" purely because each happened to be the clearest
    scene in its own search window).

    Returns
    -------
    tuple(float, float, float)
        (day_of_year_distance, sun_elevation_diff_or_0, combined_cloud_cover)
        Seasonal and illumination match are prioritized first (they're the
        harder-to-fix, more distorting mismatches); combined cloud cover
        is the final tiebreaker among otherwise-similar pairs.
    """
    doy_dist = day_of_year_distance(before_item.datetime.date(), after_item.datetime.date())
    elev_diff = sun_elevation_diff(before_item, after_item)
    elev_component = elev_diff if elev_diff is not None else 0.0
    cloud = (
        before_item.properties.get("eo:cloud_cover", 100)
        + after_item.properties.get("eo:cloud_cover", 100)
    )
    return (float(doy_dist), float(elev_component), float(cloud))


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

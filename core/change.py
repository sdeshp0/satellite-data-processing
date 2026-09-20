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

# Sentinel-2 SCL "No Data" class. This is the genuine within-swath
# "No Data" classification AND, not coincidentally, also what
# core.load._read_band's boundless reads fill AOI pixels with when they
# fall outside a granule's real footprint (fill_value=0) -- so checking
# for this single class code catches both cases with one condition. Not
# included in CLOUD_CLASSES above (used by core.utils.apply_cloud_mask for
# the single-scene "mask clouds" toggle) because that's a separate,
# narrower concern (clouds/shadow specifically); comparison code needs the
# broader "is this pixel real data at all" check below.
NO_DATA_CLASS = 0


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


def _scl_valid_mask(codes: np.ndarray) -> np.ndarray:
    """
    True where SCL codes represent real, cloud/shadow-free data -- i.e.
    not a cloud/shadow class (CLOUD_CLASSES) and not "No Data"
    (NO_DATA_CLASS). See NO_DATA_CLASS's docstring for why that second
    condition matters as much as clouds do here: without it, AOI pixels
    that fall outside a scene's real footprint (boundless-read fill) look
    like valid near-zero-reflectance data instead of being excluded, which
    can manufacture a spurious "change" signal in exactly the region where
    one date doesn't actually cover the AOI.
    """
    return ~(np.isin(codes, CLOUD_CLASSES) | (codes == NO_DATA_CLASS))


def combined_valid_mask(scl_before: xr.DataArray, scl_after: xr.DataArray) -> np.ndarray:
    """
    Boolean mask (True = usable), on scl_before's grid, requiring a pixel to
    have real, cloud/shadow-free data in BOTH dates -- a pixel that's clear
    before but cloudy (or outside the granule's real footprint) after, or
    vice versa, can't support a real comparison. See coverage_overlap for
    the same computation broken out spatially (where the mismatch falls)
    and by category, rather than collapsed into a single AND mask.
    """
    scl_after_aligned = align_to_reference(
        scl_after.values, scl_after, scl_before, resampling=Resampling.nearest
    )
    valid_before = _scl_valid_mask(scl_before.values.astype(int))
    valid_after = _scl_valid_mask(np.rint(scl_after_aligned).astype(int))
    return valid_before & valid_after


# --- Coverage overlap (spatial) ---------------------------------------------
# Where combined_valid_mask collapses "usable in both dates" down to a
# single AND mask (exactly what's needed to compute the delta/
# classification), these three constants + coverage_overlap() keep the
# four-way breakdown -- both / before-only / after-only / neither -- and
# WHERE each falls in the AOI. Two scenes can each individually have fine
# overall coverage (core.load.load_scene's coverage_fraction) while still
# covering *different* parts of the AOI -- e.g. before misses the NW
# corner, after misses the SE corner -- which a single aggregate number
# from either scene alone can't reveal.
COVERAGE_NEITHER_CODE = 0
COVERAGE_BOTH_CODE = 1
COVERAGE_BEFORE_ONLY_CODE = 2
COVERAGE_AFTER_ONLY_CODE = 3

COVERAGE_LABELS: Dict[int, str] = {
    COVERAGE_NEITHER_CODE: "No Data (Either Date)",
    COVERAGE_BOTH_CODE: "Usable — Both Dates",
    COVERAGE_BEFORE_ONLY_CODE: "Usable — Before Only",
    COVERAGE_AFTER_ONLY_CODE: "Usable — After Only",
}
COVERAGE_COLORS: Dict[int, str] = {
    COVERAGE_NEITHER_CODE: "#f0f0f0",
    COVERAGE_BOTH_CODE: "#31a354",
    COVERAGE_BEFORE_ONLY_CODE: "#fdae6b",
    COVERAGE_AFTER_ONLY_CODE: "#6baed6",
}


def coverage_overlap(
    scl_before: xr.DataArray, scl_after: xr.DataArray
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Classify every AOI pixel (on scl_before's grid) into one of four
    coverage categories based on per-date SCL-derived usability (see
    _scl_valid_mask), and summarize the AOI-wide percentage in each.

    Parameters
    ----------
    scl_before, scl_after : xr.DataArray
        Unscaled SCL bands (see core.utils.scale_bands's exclude default).

    Returns
    -------
    tuple(np.ndarray, dict)
        codes : int ndarray, same shape as scl_before, valued per the
            COVERAGE_*_CODE constants above (see COVERAGE_LABELS/COLORS
            for display).
        percentages : dict with keys "both", "before_only", "after_only",
            "neither" -- percent of AOI pixels (by count) in each
            category. "both" is exactly the AOI fraction that
            combined_valid_mask lets through to the delta/classification.
    """
    scl_after_aligned = align_to_reference(
        scl_after.values, scl_after, scl_before, resampling=Resampling.nearest
    )
    valid_before = _scl_valid_mask(scl_before.values.astype(int))
    valid_after = _scl_valid_mask(np.rint(scl_after_aligned).astype(int))

    codes = np.full(valid_before.shape, COVERAGE_NEITHER_CODE, dtype=int)
    codes[valid_before & valid_after] = COVERAGE_BOTH_CODE
    codes[valid_before & ~valid_after] = COVERAGE_BEFORE_ONLY_CODE
    codes[~valid_before & valid_after] = COVERAGE_AFTER_ONLY_CODE

    total = codes.size
    percentages = {
        "both": round(100 * int(np.sum(codes == COVERAGE_BOTH_CODE)) / total, 1) if total else 0.0,
        "before_only": round(100 * int(np.sum(codes == COVERAGE_BEFORE_ONLY_CODE)) / total, 1) if total else 0.0,
        "after_only": round(100 * int(np.sum(codes == COVERAGE_AFTER_ONLY_CODE)) / total, 1) if total else 0.0,
        "neither": round(100 * int(np.sum(codes == COVERAGE_NEITHER_CODE)) / total, 1) if total else 0.0,
    }
    return codes, percentages


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
            "warning",
            f"Before ({before_tile}) and after ({after_tile}) scenes come "
            "from different Sentinel-2 MGRS tiles. If the AOI sits near "
            "the tile boundary, the two scenes can cover substantially "
            "different ground within it -- check the coverage-overlap "
            "map below for how much of the AOI is actually usable in "
            "both dates, not just each scene's own coverage.",
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


def granule_coverage_pct(item: Any) -> Optional[float]:
    """
    Granule-level real-data coverage for a STAC item: 100% minus
    Sentinel-2's own s2:nodata_pixel_percentage property. Returns None if
    the item doesn't report that property, rather than guessing.

    This is coverage of the WHOLE scene, not specifically any particular
    AOI -- a scene can score high here and still only partially cover a
    given AOI (or vice versa, if a small AOI happens to sit entirely in a
    mostly-empty granule's one good corner). The AOI-specific check is
    core.change.coverage_overlap, which needs the scene actually loaded;
    this function is the cheap, metadata-only proxy used for filtering
    search results *before* anything is loaded -- see scene_picker's
    "Minimum coverage" slider and select_best_pair's min_coverage_before/
    min_coverage_after below.
    """
    nodata_pct = item.properties.get("s2:nodata_pixel_percentage")
    if nodata_pct is None:
        return None
    return 100.0 - float(nodata_pct)


def _filter_by_coverage(items: List[Any], min_coverage: float) -> List[Any]:
    """
    Keep items with granule_coverage_pct >= min_coverage, or unknown
    coverage (never filtered out -- matches scene_picker's slider
    behavior: missing metadata isn't treated as a failure). If filtering
    would remove every item, falls back to the original unfiltered list
    rather than returning empty -- a caller that already confirmed items
    is non-empty should still get a best-effort pair out of
    select_best_pair, not nothing, just because every candidate happened
    to fall below the threshold.
    """
    if min_coverage <= 0:
        return items

    filtered = [
        item for item in items
        if (cov := granule_coverage_pct(item)) is None or cov >= min_coverage
    ]
    return filtered if filtered else items


DEFAULT_MAX_CLOUD_FOR_TILE_PREFERENCE = 30.0


def select_best_pair(
    before_items: List,
    after_items: List,
    min_coverage_before: float = 50.0,
    min_coverage_after: float = 50.0,
    max_cloud_for_tile_preference: float = DEFAULT_MAX_CLOUD_FOR_TILE_PREFERENCE,
) -> Optional[Tuple[Any, Any]]:
    """
    Choose the before/after pair, from two independently-searched result
    lists, with the best comparability_score (season, sun-angle, cloud --
    see that function) among a candidate pool built with a TIERED, not
    absolute, preference for matching MGRS tiles:

      1. Same-tile pairs with "reasonable" combined cloud cover (<=
         max_cloud_for_tile_preference, default 30%) -- the ideal case:
         no mosaic needed, and not badly cloudy.
      2. If none qualify, EVERY candidate pair, same-tile or not --
         core.load.load_scene's multi-tile mosaicking can compensate for
         a tile mismatch at a bounded, measured cost (~11s/sample
         observed in practice via check_sample_coverage.py), so once the
         same-tile options aren't good enough on their own, tile match
         stops constraining the search at all and the best-scoring pair
         wins outright.

    (Earlier draft of this had a middle tier -- "same-tile regardless of
    cloud" -- between these two. That's a bug, not a feature: whenever ANY
    same-tile pair exists, that middle tier is non-empty, so it would
    always win over tier 2 and cross-tile pairs would never be reached no
    matter how much clearer they were. Two tiers, not three.)

    This tiering replaced an earlier version where tile match was an
    absolute, lexicographically-first veto (a same-tile pair always beat
    ANY cross-tile pair, however much worse its cloud cover). That made
    sense before mosaicking existed, when a cross-tile pair was often
    catastrophic (60-95% AOI loss). Once mosaic could fix that, the
    absolute veto became a liability: real coverage-check runs showed it
    repeatedly locking onto a same-tile pair with ~30% combined cloud
    (comfortably passing an early, generous "reasonable" bar) while never
    even considering a cross-tile pair that might have been dramatically
    clearer -- because tuple comparison meant ANY tile match unconditionally
    beat ANY cloud/season improvement, however large. The tiered approach
    still prefers avoiding mosaic's extra cost when a same-tile option is
    genuinely fine, but stops treating "same tile" as more important than
    "actually visible" once it isn't.

    Note this can't see WHERE within a granule cloud sits relative to the
    AOI -- eo:cloud_cover is a whole-scene percentage, not AOI-local, so a
    "reasonable" 30% figure can still coincide with a cloud bank sitting
    directly over a small AOI while most of the rest of the tile is clear.
    That's a real blind spot no amount of tuning this function resolves;
    it would need an actual (costlier) per-candidate data read to fix.

    Each list is first filtered to granule_coverage_pct >=
    min_coverage_before / min_coverage_after respectively (see
    _filter_by_coverage for the "don't filter to nothing" fallback), so
    the automatic pick respects the same coverage threshold as
    scene_picker's "Minimum coverage" slider rather than being able to
    silently auto-select a scene the user would have hidden if picking
    manually. Defaults (50%) match that slider's own default; pass the
    slider's live value(s) to keep them in sync when the user adjusts it.

    Shared by two call sites: the Change Detection page uses this to
    automatically pre-select a pair as soon as both searches return
    results (the user can still override via either scene_picker table),
    and sample_analyses.py uses it for the one-click samples.

    Runs in O(len(before_items) * len(after_items)); both lists are
    typically small (a few dozen items at most from one search window), so
    this stays fast in practice.

    Returns
    -------
    tuple(pystac.Item, pystac.Item) or None
        (best_before, best_after), ranked by comparability_score within
        whichever tier was used, or None if either input list is empty.
        (Coverage filtering alone will never be the reason this returns
        None -- see _filter_by_coverage's fallback.)
    """
    if not before_items or not after_items:
        return None

    before_candidates = _filter_by_coverage(before_items, min_coverage_before)
    after_candidates = _filter_by_coverage(after_items, min_coverage_after)

    all_pairs = [(before, after) for before in before_candidates for after in after_candidates]
    if not all_pairs:
        return None

    def _same_tile(before: Any, after: Any) -> bool:
        tile_before = before.properties.get("s2:mgrs_tile")
        tile_after = after.properties.get("s2:mgrs_tile")
        return tile_before is not None and tile_before == tile_after

    def _avg_cloud(before: Any, after: Any) -> float:
        return (
            before.properties.get("eo:cloud_cover", 100)
            + after.properties.get("eo:cloud_cover", 100)
        ) / 2.0

    reasonable_same_tile_pairs = [
        (b, a) for b, a in all_pairs
        if _same_tile(b, a) and _avg_cloud(b, a) <= max_cloud_for_tile_preference
    ]

    candidate_pool = reasonable_same_tile_pairs or all_pairs

    return min(candidate_pool, key=lambda pair: comparability_score(pair[0], pair[1]))


def comparability_score(before_item, after_item) -> Tuple[float, float, float]:
    """
    Sortable score for ranking candidate before/after pairs -- lower is
    better on every component, and the tuple is meant to be used directly
    with min()/sorted(). Used within whichever tier select_best_pair has
    already narrowed candidates down to (see that function for the MGRS
    tile preference, which used to live in this score directly but is now
    a separate tiering step -- see its docstring for why).

    Returns
    -------
    tuple(float, float, float)
        (day_of_year_distance, sun_elevation_diff_or_0, combined_cloud_cover)
        Seasonal and illumination match are prioritized first (they're the
        harder-to-fix, more distorting mismatches); combined cloud cover
        is the final tiebreaker among otherwise-similar pairs. Picking
        each side's lowest-cloud scene independently (the simpler
        alternative to using this at all) can easily land on a pair
        that's individually clear but seasonally or illumination-
        mismatched -- exactly what comparability_checks warns about once
        a pair is actually loaded.
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

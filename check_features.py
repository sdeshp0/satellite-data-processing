#!/usr/bin/env python3
"""
Diagnostic for Feature Identification's thresholds (core/features.py):
how each spectral index behaves on each ESA WorldCover land-cover class in
real scenes, and how well candidate rules for each feature type (water,
urban, vegetation, bare soil) agree with WorldCover, across several
different landscapes.

For each site:

  1. The least-cloudy Sentinel-2 scene in the site's date window.
  2. The scene's bands and indices, loaded the way the app loads them,
     with each band's reflectance range. When the AOI needs more than one
     Sentinel-2 tile, each tile is also checked on its own: its processing
     baseline, the reflectance offset its own metadata declares, the
     offset the app applies, and its blue/NIR reflectance.
  3. ESA WorldCover on the same grid.
  4. Per land-cover class: pixel count and the 10th/50th/90th percentile
     of each index.
  5. Alignment: how well water and vegetation match WorldCover when
     WorldCover is shifted a few pixels each way. A best match away from
     no shift means the two grids don't line up.
  6. Per feature: candidate rules scored against WorldCover (precision =
     share of flagged pixels WorldCover agrees with; recall = share of
     WorldCover's pixels that were flagged; F1 = balance of the two), and
     the classes most of the false hits fall in.

Then a summary: each rule's F1 at every site, and the mean, so a rule
that only works in one landscape stands out.

WorldCover is a 2021 snapshot and not perfect itself, so the scores are a
guide for comparing rules, not ground truth.

Usage
-----
Run from the repository root:

    python check_features.py | tee check_features.txt     # every site
    python check_features.py --site san_francisco --site phoenix
    python check_features.py --lat 40.71 --lon -74.01 --size-km 20 --start 2023-06-01 --end 2023-09-30

Needs real network access to Microsoft Planetary Computer.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import date
from typing import Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np  # noqa: E402

from check_worldcover import _step  # noqa: E402

# Landscapes picked to stress different rules. Date windows are dry or
# leaf-on seasons with little cloud.
SITES: Dict[str, Dict] = {
    "san_francisco": {
        "lat": 37.7749, "lon": -122.4194, "size_km": 15, "start": "2023-06-01", "end": "2023-10-31",
        "tests": "dense city, deep ocean, turbid bay, dry grass hills",
    },
    "phoenix": {
        "lat": 33.4484, "lon": -112.0740, "size_km": 20, "start": "2023-03-01", "end": "2023-05-31",
        "tests": "desert city: built-up vs bare soil, irrigated green",
    },
    "cairo_nile": {
        "lat": 30.0444, "lon": 31.2357, "size_km": 20, "start": "2023-03-01", "end": "2023-05-31",
        "tests": "river, dense city, farmland, sand desert",
    },
    "new_orleans": {
        "lat": 29.9511, "lon": -90.0715, "size_km": 20, "start": "2023-10-01", "end": "2023-12-31",
        "tests": "muddy river, lakes, wetland, city",
    },
    "central_valley": {
        "lat": 36.60, "lon": -120.10, "size_km": 20, "start": "2023-07-01", "end": "2023-09-30",
        "tests": "crops vs fallow fields, small towns",
    },
    "rondonia": {
        "lat": -10.83, "lon": -62.90, "size_km": 20, "start": "2023-06-01", "end": "2023-08-31",
        "tests": "rainforest vs cleared pasture",
    },
}

# WorldCover classes each feature should match.
TARGETS = {
    "water": (80,),
    "urban": (50,),
    "vegetation": (10, 20, 30, 40, 90, 95),
    "bare_soil": (60,),
}
INDEX_NAMES = ["ndwi", "mndwi", "ndbi", "ndvi", "bsi"]


def _rules(ix: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    """Candidate rules per feature, as name -> boolean mask. The first is the app's current rule."""
    ndbi, ndvi, ndwi, mndwi, bsi = ix["ndbi"], ix["ndvi"], ix["ndwi"], ix["mndwi"], ix["bsi"]
    with np.errstate(invalid="ignore"):
        not_water = (ndwi <= 0) & (mndwi <= 0)
        return {
            "water": {
                "current: NDWI > 0": ndwi > 0,
                "MNDWI > 0": mndwi > 0,
                "MNDWI > 0.2": mndwi > 0.2,
                "NDWI > 0 or MNDWI > 0.2": (ndwi > 0) | (mndwi > 0.2),
                "MNDWI > 0, NDVI < 0.2": (mndwi > 0) & (ndvi < 0.2),
            },
            "urban": {
                "current: NDBI > 0": ndbi > 0,
                "NDBI > 0, not water": (ndbi > 0) & not_water,
                "NDBI > -0.1, not water, NDVI < 0.3": (ndbi > -0.1) & not_water & (ndvi < 0.3),
                "NDBI > -0.2, not water, NDVI < 0.3": (ndbi > -0.2) & not_water & (ndvi < 0.3),
                "NDBI - NDVI > 0, not water": (ndbi - ndvi > 0) & not_water,
                "NDBI - NDVI > -0.2, not water": (ndbi - ndvi > -0.2) & not_water,
                "not water, NDVI < 0.25, BSI < 0.1": not_water & (ndvi < 0.25) & (bsi < 0.1),
            },
            "vegetation": {
                "current: NDVI > 0.3": ndvi > 0.3,
                "NDVI > 0.2": ndvi > 0.2,
                "NDVI > 0.4": ndvi > 0.4,
            },
            "bare_soil": {
                "current: BSI > 0": bsi > 0,
                "BSI > 0, not water, NDVI < 0.3": (bsi > 0) & not_water & (ndvi < 0.3),
                "BSI > 0.1, not water, NDVI < 0.3": (bsi > 0.1) & not_water & (ndvi < 0.3),
                "BSI > 0.1, not water, NDVI < 0.2": (bsi > 0.1) & not_water & (ndvi < 0.2),
            },
        }


def _load(site: Dict, max_dim: int):
    """(indices incl. MNDWI, clear mask, WorldCover grid, scene label), or None if a step failed."""
    from shapely.geometry import box

    from core.indices import compute_indices
    from core.landcover import search_worldcover, worldcover_on_grid
    from core.load import load_scene
    from core.stac import search_sentinel2
    from core.utils import apply_cloud_mask, apply_nodata_mask, resample_bands, scale_bands

    half_lat = site["size_km"] / 2 / 111.32
    half_lon = half_lat / np.cos(np.radians(site["lat"]))
    aoi = box(site["lon"] - half_lon, site["lat"] - half_lat, site["lon"] + half_lon, site["lat"] + half_lat)

    _step("1. Scene search")
    try:
        items = search_sentinel2(aoi, date.fromisoformat(site["start"]), date.fromisoformat(site["end"]), max_cloud_cover=20)
        if not items:
            print("No scene with 20% cloud or less in the window.")
            return None
        item = min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))
        label = (f"{item.id} ({item.datetime.date()}, {item.properties.get('eo:cloud_cover', '?')}% cloud, "
                 f"processing baseline {item.properties.get('s2:processing_baseline', '?')})")
        print(f"{len(items)} scenes; using {label}")
    except Exception:
        traceback.print_exc()
        return None

    _step("2. Bands and indices")
    try:
        bands, coverage, load_info = load_scene(item, aoi, max_dim)
        bands = apply_cloud_mask(apply_nodata_mask(scale_bands(resample_bands(bands))))
        ix = compute_indices(bands)
        green, swir1 = bands["green"].values, bands["swir1"].values
        with np.errstate(invalid="ignore", divide="ignore"):
            ix["mndwi"] = (green - swir1) / (green + swir1)
        clear = np.isfinite(ix["ndvi"]) & np.isfinite(ix["mndwi"])
        print(f"grid {clear.shape}, coverage {coverage:.0%}, clear {clear.mean():.0%}")
        # Reflectance should sit roughly in 0-1; many negative values point
        # to a scaling or offset problem rather than real surfaces.
        for name in ("blue", "green", "red", "nir", "swir1", "swir2"):
            values = bands[name].values[clear]
            p1, p50, p99 = np.percentile(values, [1, 50, 99])
            print(f"  {name:<6} reflectance p1 {p1:+.3f}  p50 {p50:+.3f}  p99 {p99:+.3f}  "
                  f"negative {np.mean(values < 0):.1%}")
        if load_info.get("mosaic_attempted"):
            _check_tiles(item, aoi)
    except Exception:
        traceback.print_exc()
        return None

    _step("3. WorldCover on the grid")
    try:
        wc_items, year = search_worldcover(bands["nir"], item.datetime.year)
        if not wc_items:
            print("No WorldCover tiles cover this AOI.")
            return None
        landcover = worldcover_on_grid(wc_items, bands["nir"])
        print(f"WorldCover {year}")
    except Exception:
        traceback.print_exc()
        return None
    return ix, clear, landcover, label


def _declared_offset(item) -> str:
    """BOA_ADD_OFFSET values from the scene's own product metadata XML, or why they couldn't be read."""
    import re

    import planetary_computer as pc
    import requests

    asset = getattr(item, "assets", {}).get("product-metadata")
    if asset is None:
        return "no product-metadata asset"
    try:
        xml = requests.get(pc.sign(asset.href.split("?", 1)[0]), timeout=30).text
    except Exception as exc:
        return f"metadata unreadable ({type(exc).__name__})"
    values = sorted(set(re.findall(r"<BOA_ADD_OFFSET[^>]*>(-?\d+)</BOA_ADD_OFFSET>", xml)))
    return ", ".join(values) if values else "none declared"


def _check_tiles(item, aoi) -> None:
    """Each tile covering the AOI on its own: declared vs applied offset, and blue/NIR reflectance."""
    from core.load import _boa_offset_for_item, load_scene
    from core.stac import search_companion_tiles
    from core.utils import scale_bands

    try:
        tiles = [item] + search_companion_tiles(aoi, item, max_items=3)
    except Exception:
        traceback.print_exc()
        return
    print("  tiles covering the AOI, each loaded alone:")
    for tile in tiles:
        try:
            bands, coverage, _ = load_scene(tile, aoi, 256, enable_mosaic=False)
            valid = bands["scl"].values > 0
            bands = scale_bands(bands)
            parts = []
            for name in ("blue", "nir"):
                values = bands[name].values[valid]
                parts.append(f"{name} p50 {np.median(values):+.3f} negative {np.mean(values < 0):.0%}"
                             if values.size else f"{name} no data")
            print(f"    {tile.id}: baseline {tile.properties.get('s2:processing_baseline', '?')}, "
                  f"declared offset {_declared_offset(tile)}, applied {_boa_offset_for_item(tile):+.0f}, "
                  f"covers {coverage:.0%}; " + "; ".join(parts))
        except Exception:
            traceback.print_exc()


def _percentiles(ix, clear, landcover) -> None:
    from core.landcover import WORLDCOVER_CLASSES

    _step("4. Index percentiles (p10 / p50 / p90) per land-cover class")
    print(f"{'class':<26}{'pixels':>8}  " + "".join(f"{n.upper():>20}" for n in INDEX_NAMES))
    for code, (label, _) in WORLDCOVER_CLASSES.items():
        sel = clear & (landcover == code)
        if sel.sum() < 50:
            continue
        cells = "".join(
            "{:>20}".format("/".join(f"{v:+.2f}" for v in np.percentile(ix[n][sel], [10, 50, 90])))
            for n in INDEX_NAMES
        )
        print(f"{label:<26}{int(sel.sum()):>8}  {cells}")


def _f1(mask: np.ndarray, target: np.ndarray) -> float:
    hit = int((mask & target).sum())
    return 2 * hit / (int(mask.sum()) + int(target.sum())) if mask.any() or target.any() else 0.0


def _alignment(ix, clear, landcover, max_shift: int = 4) -> None:
    """Water and vegetation agreement with WorldCover shifted up to max_shift pixels each way."""
    _step("5. Alignment with WorldCover (F1 of the current rule, WorldCover shifted)")
    m = max_shift
    inner = (slice(m, -m), slice(m, -m))
    checks = {
        "water (NDWI > 0)": (ix["ndwi"] > 0, TARGETS["water"]),
        "vegetation (NDVI > 0.3)": (ix["ndvi"] > 0.3, TARGETS["vegetation"]),
    }
    for name, (rule, codes) in checks.items():
        target_full = np.isin(landcover, codes)
        mask = (rule & clear)[inner]
        if target_full[inner].sum() < 200:
            print(f"  {name:<26} too little of this class here to check")
            continue
        best, best_shift = -1.0, (0, 0)
        # Nearest shifts first, and a farther one must do clearly better, so
        # noise doesn't pick a shift.
        shifts = sorted(((dy, dx) for dy in range(-m, m + 1) for dx in range(-m, m + 1)),
                        key=lambda s: abs(s[0]) + abs(s[1]))
        for dy, dx in shifts:
            shifted = np.roll(target_full, (dy, dx), axis=(0, 1))[inner] & clear[inner]
            f1 = _f1(mask, shifted)
            if f1 > best + 0.005:
                best, best_shift = f1, (dy, dx)
        at_zero = _f1(mask, target_full[inner] & clear[inner])
        print(f"  {name:<26} no shift F1 {at_zero:.2f}; best F1 {best:.2f} with WorldCover moved "
              f"{best_shift[0]:+d} rows, {best_shift[1]:+d} cols")


def _score(ix, clear, landcover) -> Dict[str, Dict[str, Optional[Tuple[float, float, float]]]]:
    """(precision, recall, F1) per feature and rule; None where WorldCover has too few target pixels."""
    from core.landcover import WORLDCOVER_CLASSES

    _step("6. Candidate rules scored against WorldCover")
    scores: Dict[str, Dict] = {}
    for feature, rules in _rules(ix).items():
        target = clear & np.isin(landcover, TARGETS[feature])
        share = target.sum() / clear.sum()
        print(f"\n{feature} (WorldCover {TARGETS[feature]}: {share:.1%} of clear pixels)")
        scores[feature] = {}
        if target.sum() < 200:
            print("  too little of this class here to score")
            scores[feature] = {name: None for name in rules}
            continue
        for name, mask in rules.items():
            mask = mask & clear
            hit = int((mask & target).sum())
            precision = hit / mask.sum() if mask.sum() else 0.0
            recall = hit / target.sum()
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            scores[feature][name] = (precision, recall, f1)
            wrong = mask & ~target
            top = ""
            if wrong.any():
                codes, counts = np.unique(landcover[wrong], return_counts=True)
                order = np.argsort(counts)[::-1][:2]
                top = ", ".join(
                    f"{WORLDCOVER_CLASSES.get(int(codes[i]), (str(codes[i]),))[0]} {counts[i] / wrong.sum():.0%}"
                    for i in order
                )
            print(f"  {name:<40} flags {mask.sum() / clear.sum():>4.0%}  precision {precision:.2f}  "
                  f"recall {recall:.2f}  F1 {f1:.2f}  false hits mostly: {top}")
    return scores


def _summary(results: Dict[str, Dict]) -> None:
    _step("Summary: F1 per site (higher is better; - = too little of the class there)")
    sites = list(results)
    short = [s[:12] for s in sites]
    for feature in TARGETS:
        print(f"\n{feature}")
        print(f"  {'rule':<40}" + "".join(f"{s:>14}" for s in short) + f"{'mean':>8}")
        names = next(iter(results.values()))[feature].keys()
        for name in names:
            values = [results[s][feature][name] for s in sites]
            f1s = [v[2] for v in values if v is not None]
            cells = "".join(f"{'-' if v is None else f'{v[2]:.2f}':>14}" for v in values)
            mean = f"{np.mean(f1s):.2f}" if f1s else "-"
            print(f"  {name:<40}{cells}{mean:>8}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", action="append", choices=list(SITES), help="site to run (repeatable; default all)")
    parser.add_argument("--lat", type=float, help="custom AOI centre latitude (with --lon; replaces the sites)")
    parser.add_argument("--lon", type=float)
    parser.add_argument("--size-km", type=float, default=15.0)
    parser.add_argument("--start", default="2023-06-01", help="custom AOI window start (ISO date)")
    parser.add_argument("--end", default="2023-09-30")
    parser.add_argument("--max-dim", type=int, default=1024)
    args = parser.parse_args()

    if args.lat is not None and args.lon is not None:
        sites = {"custom": {"lat": args.lat, "lon": args.lon, "size_km": args.size_km,
                            "start": args.start, "end": args.end, "tests": "custom AOI"}}
    else:
        sites = {name: SITES[name] for name in (args.site or SITES)}

    results = {}
    for name, site in sites.items():
        print(f"\n\n######## {name}: {site['tests']}")
        print(f"AOI centre {site['lat']:.4f}, {site['lon']:.4f}, {site['size_km']} km square, "
              f"{site['start']} to {site['end']}")
        loaded = _load(site, args.max_dim)
        if loaded is None:
            continue
        ix, clear, landcover, _ = loaded
        _percentiles(ix, clear, landcover)
        _alignment(ix, clear, landcover)
        results[name] = _score(ix, clear, landcover)

    if not results:
        print("\nNo site loaded.")
        return 1
    _summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

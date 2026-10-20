#!/usr/bin/env python3
"""
Step-by-step diagnostic for the Sentinel-1 radar flood map
(core/radar.py) shown on Change Detection's flood analyses.

Builds a UTM grid over an AOI, then runs each step the app runs and prints
what comes back, with a full traceback wherever a step fails:

  1. The radar scenes found around each date: date, orbit track, assets.
  2. The pair the app would pick.
  3. Each date's VV backscatter on the grid: coverage, dB spread, share
     below the water threshold, and read time.
  4. The water-change classes from the two dates.

Usage
-----
Run from the repository root:

    python check_radar.py                                    # Sindh 2022, flood peak
    python check_radar.py --before 2022-06-19 --after 2022-09-11
    python check_radar.py --sample missouri_river_flood_2019 --before 2018-10-01 --after 2019-03-22
    python check_radar.py --lat 29.80 --lon -95.65 --size-km 20      # Addicks/Barker reservoirs

Needs real network access to Microsoft Planetary Computer.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np  # noqa: E402

from check_worldcover import _reference_grid, _step  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", default="sindh_flood_2022", help="sample analysis key (default sindh_flood_2022)")
    parser.add_argument("--before", type=date.fromisoformat, help="before date (default: the sample's radar date)")
    parser.add_argument("--after", type=date.fromisoformat, help="after date (default: the sample's radar date)")
    parser.add_argument("--lat", type=float, help="AOI centre latitude (overrides the sample's)")
    parser.add_argument("--lon", type=float, help="AOI centre longitude (overrides the sample's)")
    parser.add_argument("--size-km", type=float)
    parser.add_argument("--res", type=float, default=20.0, help="grid resolution in metres (default 20)")
    args = parser.parse_args()

    from app.components.sample_analyses import SAMPLE_ANALYSES

    if args.sample not in SAMPLE_ANALYSES:
        print(f"Unknown sample {args.sample!r}. Known: {', '.join(SAMPLE_ANALYSES)}")
        return 2
    sample = SAMPLE_ANALYSES[args.sample]
    if (args.before is None or args.after is None) and "radar_dates" not in sample:
        print(f"{args.sample} has no radar dates; pass --before and --after.")
        return 2
    before = args.before or sample["radar_dates"][0]
    after = args.after or sample["radar_dates"][1]
    lat = args.lat if args.lat is not None else sample["lat"]
    lon = args.lon if args.lon is not None else sample["lon"]
    size_km = args.size_km or max(sample.get("width_km", 10), sample.get("height_km", 10))
    print(f"centre {lat:.4f}, {lon:.4f}, {size_km} km square, {args.res} m grid")
    print(f"dates: before {before}, after {after}")

    import core.radar as rd

    reference = _reference_grid(lat, lon, size_km, args.res)
    window = timedelta(days=rd.MAX_DAYS_FROM_OPTICAL)

    _step("1. Radar scenes around each date")
    found = {}
    for label, when in (("before", before), ("after", after)):
        try:
            items = rd.search_radar(reference, when - window, when + window)
        except Exception:
            traceback.print_exc()
            return 1
        found[label] = items
        print(f"{label}: {len(items)} scene(s) within {rd.MAX_DAYS_FROM_OPTICAL} days of {when}")
        for item in sorted(items, key=lambda i: i.datetime):
            props = item.properties
            print(f"  - {item.datetime.date()} orbit {props.get('sat:relative_orbit')} "
                  f"{props.get('sat:orbit_state')} assets {sorted(item.assets)} id {item.id}")

    _step("2. Pair the app would pick")
    from rasterio.warp import transform_bounds

    bbox = transform_bounds(reference.rio.crs, "EPSG:4326", *reference.rio.bounds())
    pair = rd.pick_radar_pair(found["before"], found["after"], before, after, bbox)
    if pair is None:
        print("No pair: one of the dates has no radar scene, so the app shows no radar map.")
        return 1
    for label, items in (("before", pair.before_items), ("after", pair.after_items)):
        print(f"{label}: {[item.id for item in items]} (footprint covers {100 * rd.pass_coverage(items, bbox):.0f}% of the AOI)")
    print(f"same track: {pair.same_track}")

    _step("3. VV backscatter on the grid")
    grids = {}
    for label, items in (("before", pair.before_items), ("after", pair.after_items)):
        try:
            start = time.time()
            db = rd.backscatter_db(items, reference)
        except Exception:
            traceback.print_exc()
            return 1
        grids[label] = db
        valid = np.isfinite(db)
        print(f"{label}: read in {time.time() - start:.1f}s, {100 * valid.mean():.1f}% of the grid has data")
        if valid.any():
            p5, p50, p95 = np.percentile(db[valid], [5, 50, 95])
            water = 100 * (db[valid] < rd.WATER_THRESHOLD_DB).mean()
            print(f"  dB 5th/50th/95th percentile: {p5:.1f} / {p50:.1f} / {p95:.1f}; "
                  f"{water:.1f}% below {rd.WATER_THRESHOLD_DB} dB")

    _step("4. Water-change classes")
    from core.change import WATER_LABELS, pixel_area_ha, format_area

    codes = rd.classify_radar_water(grids["before"], grids["after"])
    px_ha = pixel_area_ha(reference)
    for code, label in WATER_LABELS.items():
        print(f"  {label:<26} {format_area((codes == code).sum() * px_ha)}")
    print(f"  {'No data':<26} {format_area((codes < 0).sum() * px_ha)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

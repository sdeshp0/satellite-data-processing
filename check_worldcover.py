#!/usr/bin/env python3
"""
Step-by-step diagnostic for the ESA WorldCover land-cover lookup
(core/landcover.py) used by Change Detection and Feature Identification.

Builds a UTM grid over an AOI, then runs each step the app runs and prints
what comes back, with a full traceback wherever a step fails:

  1. Which core/landcover.py is imported (path, modified time, git commit),
     to rule out a stale deployment.
  2. The raw STAC search: every item's id, datetime fields and assets.
  3. search_worldcover: which items and year the app would use.
  4. worldcover_on_grid: the class counts after resampling onto the grid.
  5. landcover_composition: the table Feature Identification shows.

Usage
-----
Run from the repository root:

    python check_worldcover.py                          # Camp Fire sample AOI
    python check_worldcover.py --sample lake_mead_drought
    python check_worldcover.py --lat 39.76 --lon -121.62 --size-km 25

Needs real network access to Microsoft Planetary Computer.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import xarray as xr
from pyproj import Transformer


def _step(title: str) -> None:
    print(f"\n=== {title} ===")


def _reference_grid(lat: float, lon: float, size_km: float, res_m: float) -> xr.DataArray:
    """An empty DataArray on a UTM grid centred on lat/lon, like a loaded band."""
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    x0, y0 = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    half = size_km * 500
    n = int(round(2 * half / res_m))
    xs = x0 - half + res_m * (np.arange(n) + 0.5)
    ys = y0 + half - res_m * (np.arange(n) + 0.5)
    da = xr.DataArray(np.zeros((n, n), dtype="float32"), coords={"y": ys, "x": xs}, dims=("y", "x"))
    return da.rio.write_crs(f"EPSG:{epsg}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", default="camp_fire_2018", help="sample analysis key (default camp_fire_2018)")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--size-km", type=float)
    parser.add_argument("--as-of", type=int, help="scene year; picks the WorldCover edition from then or earlier")
    parser.add_argument("--res", type=float, default=20.0, help="grid resolution in metres (default 20)")
    args = parser.parse_args()

    if args.lat is not None and args.lon is not None:
        lat, lon, size_km = args.lat, args.lon, args.size_km or 10.0
    else:
        from app.components.sample_analyses import SAMPLE_ANALYSES

        if args.sample not in SAMPLE_ANALYSES:
            print(f"Unknown sample {args.sample!r}. Known: {', '.join(SAMPLE_ANALYSES)}")
            return 2
        sample = SAMPLE_ANALYSES[args.sample]
        lat, lon = sample["lat"], sample["lon"]
        size_km = args.size_km or max(sample.get("width_km", 10), sample.get("height_km", 10))
    print(f"AOI centre {lat:.4f}, {lon:.4f}, {size_km} km square, {args.res} m grid")

    _step("1. Code version")
    import core.landcover as lc

    print("core/landcover.py:", lc.__file__)
    print("modified:", datetime.fromtimestamp(os.path.getmtime(lc.__file__)).isoformat(timespec="seconds"))
    try:
        commit = subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip()
        print("git commit:", commit or "(unknown)")
    except OSError:
        print("git commit: (git not available)")
    print("has worldcover_year (the null-datetime fix):", hasattr(lc, "worldcover_year"))

    reference = _reference_grid(lat, lon, size_km, args.res)
    print("grid:", reference.rio.crs, reference.shape, "bounds", tuple(round(b) for b in reference.rio.bounds()))

    _step("2. Raw STAC search")
    try:
        import planetary_computer as pc
        from pystac_client import Client
        from rasterio.warp import transform_bounds
        from shapely.geometry import box, mapping

        bbox = transform_bounds(reference.rio.crs, "EPSG:4326", *reference.rio.bounds())
        print("bbox (lon/lat):", tuple(round(b, 4) for b in bbox))
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
        items = list(catalog.search(collections=[lc.WORLDCOVER_COLLECTION], intersects=mapping(box(*bbox))).items())
        print(f"{len(items)} item(s)")
        for item in items:
            props = item.properties
            print(f"- {item.id}")
            print(f"    datetime={props.get('datetime')!r} start={props.get('start_datetime')!r} end={props.get('end_datetime')!r}")
            if hasattr(lc, "worldcover_year"):
                print(f"    worldcover_year -> {lc.worldcover_year(item)!r}")
            print(f"    assets: {sorted(item.assets)}")
            print(f"    property keys: {sorted(props)}")
    except Exception:
        traceback.print_exc()

    _step("3. search_worldcover (what the app uses)")
    try:
        result = lc.search_worldcover(reference, args.as_of)
        found, year = result if isinstance(result, tuple) else (result, None)
        print(f"{len(found)} item(s), year {year}: {[item.id for item in found]}")
    except Exception:
        traceback.print_exc()
        return 1
    if not found:
        print("No items, so the app would say no WorldCover data covers this AOI.")
        return 1

    _step("4. worldcover_on_grid")
    try:
        landcover = lc.worldcover_on_grid(found, reference)
    except Exception:
        traceback.print_exc()
        return 1
    codes, counts = np.unique(landcover, return_counts=True)
    for code, count in zip(codes, counts):
        label = lc.WORLDCOVER_CLASSES.get(int(code), ("no data" if code == 0 else "unknown",))[0]
        print(f"  {int(code):>3} {label:<26} {100 * count / landcover.size:5.1f}%")

    _step("5. landcover_composition")
    try:
        from core.change import pixel_area_ha

        print(lc.landcover_composition(landcover, np.ones(landcover.shape, bool), pixel_area_ha(reference)).to_string(index=False))
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

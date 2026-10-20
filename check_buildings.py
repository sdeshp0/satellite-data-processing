#!/usr/bin/env python3
"""
Step-by-step diagnostic for the Microsoft building-footprint lookup
(core/buildings.py) used by Change Detection's building exposure.

Builds a UTM grid over an AOI, then runs each step the app runs and prints
what comes back, with a full traceback wherever a step fails:

  1. The raw STAC search: every item's id, processing date and asset.
  2. search_buildings: which partitions the app would read.
  3. Which quadkey partition folders cover the AOI, and their size.
  4. building_pixels: how many building centres land on the grid.

Usage
-----
Run from the repository root:

    python check_buildings.py                          # Camp Fire sample AOI
    python check_buildings.py --sample harvey_houston_2017
    python check_buildings.py --lat 39.76 --lon -121.62 --size-km 25

Needs real network access to Microsoft Planetary Computer.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from check_worldcover import _reference_grid, _step  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", default="camp_fire_2018", help="sample analysis key (default camp_fire_2018)")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--size-km", type=float)
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

    import core.buildings as bd

    reference = _reference_grid(lat, lon, size_km, args.res)
    print("grid:", reference.rio.crs, reference.shape)

    from rasterio.warp import transform_bounds

    bbox = transform_bounds(reference.rio.crs, "EPSG:4326", *reference.rio.bounds())
    print("bbox (lon/lat):", tuple(round(b, 4) for b in bbox))

    _step("1. Raw STAC search")
    try:
        import planetary_computer as pc
        from pystac_client import Client
        from shapely.geometry import box, mapping

        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
        items = list(catalog.search(collections=[bd.BUILDINGS_COLLECTION], intersects=mapping(box(*bbox))).items())
        print(f"{len(items)} item(s)")
        for item in items:
            props = item.properties
            data = item.assets.get("data")
            print(f"- {item.id}")
            print(f"    processing-date={props.get('msbuildings:processing-date')!r} region={props.get('msbuildings:region')!r}")
            print(f"    assets: {sorted(item.assets)}")
            if data is not None:
                print(f"    data href: {data.href.split('?')[0]}")
                print(f"    storage option keys: {sorted((data.extra_fields.get('table:storage_options') or {}))}")
    except Exception:
        traceback.print_exc()

    _step("2. search_buildings (what the app uses)")
    try:
        found = bd.search_buildings(reference)
        print(f"{len(found)} item(s): {[item.id for item in found]}")
    except Exception:
        traceback.print_exc()
        return 1
    if not found:
        print("No items, so the app would say there are no mapped buildings.")
        return 1

    _step("3. Quadkey partitions for the AOI")
    try:
        import pyarrow.parquet as pq

        keys = bd.quadkeys_for_bbox(bbox)
        print(f"zoom-{bd.QUADKEY_ZOOM} tiles touching the AOI: {keys}")
        for item in found:
            fs, paths = bd.partition_paths(item, keys)
            print(f"- {item.id}: {len(paths)} partition folder(s) found")
            if not paths:
                _, base = bd._filesystem(item)
                print(f"    first entries under {base}: {fs.ls(base, detail=False)[:5]}")
            for path in paths:
                start = time.time()
                table = pq.read_table(path, filesystem=fs)
                print(f"    {path.rsplit('/', 1)[-1]}: {table.num_rows:,} rows in {time.time() - start:.1f}s; columns {table.column_names}")
    except Exception:
        traceback.print_exc()
        return 1

    _step("4. building_pixels")
    try:
        start = time.time()
        rows, cols = bd.building_pixels(found, reference)
        print(f"{len(rows):,} building centres on the grid ({time.time() - start:.1f}s)")
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

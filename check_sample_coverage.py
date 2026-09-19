#!/usr/bin/env python3
"""
Standalone coverage-health check for the curated sample analyses in
app/components/sample_analyses.py.

For each entry in SAMPLE_ANALYSES, this script:

  1. Searches for before/after scenes using the SAME progressive
     cloud-cover / date-window relaxation the app itself uses
     (CLOUD_COVER_STEPS, DATE_WINDOW_EXPANSIONS_DAYS, imported directly
     from sample_analyses.py rather than duplicated), so results match
     what clicking "Run this analysis" would actually pick.
  2. Picks the best pair with core.change.select_best_pair -- the same
     seasonal/sun-elevation-matched selection the app uses.
  3. Loads both scenes at a modest preview resolution and computes
     core.change.coverage_overlap: the actual PIXEL-LEVEL before/after
     usable-data overlap, not just each scene's own coverage_fraction.
     This is exactly the check that would have caught the original
     Ganges-Brahmaputra Delta flood sample's coverage problem before it
     showed up as a surprise in the app.
  4. Reports comparability_checks() findings alongside the coverage
     numbers (UTM/tile/platform mismatch, seasonal distance, sun
     elevation difference).
  5. Flags any sample whose "usable in both dates" percentage falls below
     a threshold (default 60%) as a likely problem sample.

Usage
-----
Run from the repository root (same directory as core/ and app/):

    python check_sample_coverage.py
    python check_sample_coverage.py --max-dim 256              # faster, coarser
    python check_sample_coverage.py --sample lake_mead_drought  # just one
    python check_sample_coverage.py --both-threshold 70
    python check_sample_coverage.py --list                     # just list keys

Requires the same Python environment as the Streamlit app
(environment.yaml / app/requirements.txt) and real network access to
Microsoft Planetary Computer -- this performs real STAC searches and real
COG reads, not a mock. Exits with status 1 if any sample is flagged (or
errors out), so it can be dropped into a CI/pre-deploy check if desired.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# --- Ensure project root is on sys.path ---
# Mirrors the same bootstrap used at the top of every app/pages/*.py file,
# so this script works whether it's run as `python check_sample_coverage.py`
# from the repo root or invoked with a full/relative path from elsewhere.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from shapely.geometry import box, Polygon

from core.stac import search_sentinel2
from core.load import load_scene
from core.utils import resample_bands
from core.change import (
    align_bands,
    coverage_overlap,
    comparability_checks,
    select_best_pair,
)

# Reuse the app's own sample definitions and relaxation schedule rather
# than duplicating them here -- if either changes in sample_analyses.py,
# this script picks up the change automatically next run. Importing this
# module pulls in streamlit as a side effect (sample_analyses.py imports
# it), but only as a plain package import -- no Streamlit app/session is
# started, and nothing here calls any st.* UI function, so this is safe to
# run as a plain script.
from app.components.sample_analyses import (
    SAMPLE_ANALYSES,
    CLOUD_COVER_STEPS,
    DATE_WINDOW_EXPANSIONS_DAYS,
)


def _build_aoi(lat: float, lon: float, width_km: float, height_km: float) -> Polygon:
    dlat = (height_km / 2) / 111.0
    dlon = (width_km / 2) / (111.0 * abs(math.cos(math.radians(lat))))
    return box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _search_with_fallback(
    aoi: Polygon, start: date, end: date
) -> Tuple[List[Any], Optional[int], Optional[int]]:
    """
    Same relaxation schedule as
    app.components.sample_analyses._search_with_fallback, but calls
    core.stac.search_sentinel2 directly instead of the Streamlit-cached
    wrapper (app.components.scene_selector._search_sentinel2_cached) --
    st.cache_data-wrapped functions work outside a running app, but the
    direct call avoids any cache-related surprises and keeps this script's
    only Streamlit dependency to the harmless SAMPLE_ANALYSES import above.
    """
    for expand_days in DATE_WINDOW_EXPANSIONS_DAYS:
        window_start = start - timedelta(days=expand_days)
        window_end = end + timedelta(days=expand_days)
        for cloud_pct in CLOUD_COVER_STEPS:
            items = search_sentinel2(aoi, window_start, window_end, max_cloud_cover=cloud_pct)
            if items:
                return items, cloud_pct, expand_days
    return [], None, None


def check_sample(sample: Dict[str, Any], max_dim: int, enable_mosaic: bool = True) -> Dict[str, Any]:
    """
    Run the full search -> pair-select -> load -> coverage-overlap
    pipeline for one sample, mirroring exactly what
    app/components/change_display.py does for a real comparison.

    Parameters
    ----------
    enable_mosaic : bool
        Passed straight through to core.load.load_scene for both scenes.
        Set False to measure the OLD single-tile-only coverage/cost for
        comparison -- see main()'s --no-mosaic flag.

    Returns a result dict on success, or a dict with an "error" key if no
    scenes could be found for one or both windows.
    """
    aoi = _build_aoi(sample["lat"], sample["lon"], sample["width_km"], sample["height_km"])
    before_start, before_end = sample["before_range"]
    after_start, after_end = sample["after_range"]

    before_items, before_cloud, before_expand = _search_with_fallback(aoi, before_start, before_end)
    after_items, after_cloud, after_expand = _search_with_fallback(aoi, after_start, after_end)

    if not before_items or not after_items:
        missing = []
        if not before_items:
            missing.append("before")
        if not after_items:
            missing.append("after")
        return {"error": f"No scenes found for the {', '.join(missing)} window(s), even after full relaxation."}

    before_item, after_item = select_best_pair(before_items, after_items)

    bands_before, before_geom_coverage, before_load_info = load_scene(
        before_item, aoi, max_dim=max_dim, enable_mosaic=enable_mosaic
    )
    bands_after, after_geom_coverage, after_load_info = load_scene(
        after_item, aoi, max_dim=max_dim, enable_mosaic=enable_mosaic
    )

    # Mirrors change_display.py's loading sequence closely enough for the
    # SCL-based coverage check -- scale_bands isn't needed here since it
    # only affects reflectance bands, not SCL (see core.utils.scale_bands's
    # exclude default), and we don't need RGB/indices for this check.
    bands_before = resample_bands(bands_before)
    bands_after = resample_bands(bands_after)
    bands_after = align_bands(bands_after, bands_before["nir"])

    coverage_codes, coverage_pct = coverage_overlap(bands_before["scl"], bands_after["scl"])
    notes = comparability_checks(before_item, after_item)

    return {
        "before_item": before_item,
        "after_item": after_item,
        "before_geom_coverage": before_geom_coverage,
        "after_geom_coverage": after_geom_coverage,
        "before_load_info": before_load_info,
        "after_load_info": after_load_info,
        "coverage_pct": coverage_pct,
        "comparability_notes": notes,
    }


class ReportWriter:
    """
    Tiny logger that both prints to the console (so you still see live
    progress while searches/loads are running -- each sample is a real
    network round trip, not instant) and accumulates every line so the
    full report -- every sample, passed or flagged, not just the
    failures -- can be saved to a text file afterward. Avoids needing to
    scroll back through terminal history to review a past run.
    """

    def __init__(self) -> None:
        self.lines: List[str] = []

    def write(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--max-dim", type=int, default=512,
        help="Decimated read size per band, in pixels (default: 512). Lower = faster but coarser SCL detail.",
    )
    parser.add_argument(
        "--both-threshold", type=float, default=60.0,
        help="Flag samples below this %% usable-in-both-dates coverage (default: 60.0).",
    )
    parser.add_argument(
        "--sample", type=str, default=None,
        help="Only check this one sample key instead of every sample in SAMPLE_ANALYSES.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available sample keys and labels, then exit without checking anything.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="sample_coverage_report.txt",
        help="Path to write the full text report to (default: sample_coverage_report.txt in the current directory).",
    )
    parser.add_argument(
        "--no-mosaic", action="store_true",
        help=(
            "Disable multi-tile mosaicking (core.load.load_scene's enable_mosaic=False), "
            "restoring the old single-tile-only coverage/cost. Run once with this flag and "
            "once without to A/B compare coverage improvement against added cost."
        ),
    )
    args = parser.parse_args()

    if args.list:
        for key, sample in SAMPLE_ANALYSES.items():
            print(f"{key:30s} {sample['label']}")
        return

    if args.sample and args.sample not in SAMPLE_ANALYSES:
        print(f"Unknown sample key: {args.sample!r}")
        print("Available keys:", ", ".join(SAMPLE_ANALYSES.keys()))
        sys.exit(2)

    keys = [args.sample] if args.sample else list(SAMPLE_ANALYSES.keys())
    enable_mosaic = not args.no_mosaic

    report = ReportWriter()
    # status is one of "PASS", "FLAGGED", "NO_SCENES", "ERROR" -- every
    # sample gets an entry here regardless of outcome, so the final summary
    # table (below) covers everything checked, not just the problems.
    summary_rows: List[Dict[str, str]] = []
    # Aggregate mosaic-cost tracking across all samples, for the "Mosaic
    # cost summary" section at the end -- the actual point of --no-mosaic
    # and this whole script's cost-measurement purpose: how much MORE
    # tiles/time did mosaicking actually add, in total, across a real
    # batch of samples, not just per-sample anecdotes.
    total_companion_tiles_used = 0
    total_companion_search_time = 0.0
    total_companion_read_time = 0.0
    samples_needing_mosaic = 0

    report.write("Sample Coverage Report")
    report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.write(
        f"Settings: max_dim={args.max_dim}, both_threshold={args.both_threshold:.0f}%, "
        f"samples={'all' if not args.sample else args.sample}, "
        f"mosaic={'enabled' if enable_mosaic else 'DISABLED (--no-mosaic)'}"
    )
    report.write(f"Checking {len(keys)} sample(s)...")
    report.write("")

    for key in keys:
        sample = SAMPLE_ANALYSES[key]
        report.write(f"=== {key} \u2014 {sample['label']} ===")
        t0 = time.time()

        try:
            result = check_sample(sample, args.max_dim, enable_mosaic=enable_mosaic)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: keep checking other samples
            report.write(f"  ERROR: {type(exc).__name__}: {exc}")
            report.write("")
            summary_rows.append({
                "key": key, "label": sample["label"], "status": "ERROR",
                "detail": f"{type(exc).__name__}: {exc}",
            })
            continue

        elapsed = time.time() - t0

        if "error" in result:
            report.write(f"  {result['error']}")
            report.write("")
            summary_rows.append({
                "key": key, "label": sample["label"], "status": "NO_SCENES",
                "detail": result["error"],
            })
            continue

        before_item = result["before_item"]
        after_item = result["after_item"]
        pct = result["coverage_pct"]

        report.write(
            f"  Before: {before_item.datetime.date()} "
            f"(cloud {before_item.properties.get('eo:cloud_cover', 0):.0f}%, "
            f"own-footprint coverage ~{result['before_geom_coverage'] * 100:.0f}%)"
        )
        report.write(
            f"  After:  {after_item.datetime.date()} "
            f"(cloud {after_item.properties.get('eo:cloud_cover', 0):.0f}%, "
            f"own-footprint coverage ~{result['after_geom_coverage'] * 100:.0f}%)"
        )

        # --- Mosaic cost/coverage, per scene ---
        sample_needed_mosaic = False
        for side_label, info in (("Before", result["before_load_info"]), ("After", result["after_load_info"])):
            if info["mosaic_attempted"]:
                sample_needed_mosaic = True
                total_companion_tiles_used += info["companions_used"]
                total_companion_search_time += info["companion_search_time_s"]
                total_companion_read_time += info["companion_read_time_s"]
                report.write(
                    f"  {side_label} mosaic: {info['primary_geom_coverage'] * 100:.1f}% "
                    f"\u2192 {info['final_coverage'] * 100:.1f}% coverage "
                    f"({info['companions_used']}/{info['companions_found']} companion tile(s) used, "
                    f"+{info['companion_search_time_s'] + info['companion_read_time_s']:.1f}s)"
                )
        if sample_needed_mosaic:
            samples_needing_mosaic += 1

        report.write(f"  Usable in BOTH dates:   {pct['both']:.1f}%")
        report.write(f"  Usable before-only:     {pct['before_only']:.1f}%")
        report.write(f"  Usable after-only:      {pct['after_only']:.1f}%")
        report.write(f"  Usable in NEITHER date: {pct['neither']:.1f}%")

        if result["comparability_notes"]:
            report.write("  Comparability notes:")
            for note in result["comparability_notes"]:
                report.write(f"    [{note.severity.upper()}] {note.message}")

        if pct["both"] < args.both_threshold:
            report.write(
                f"  >>> FLAGGED: only {pct['both']:.1f}% of the AOI is usable "
                f"in both dates (threshold {args.both_threshold:.0f}%)"
            )
            summary_rows.append({
                "key": key, "label": sample["label"], "status": "FLAGGED",
                "detail": f"only {pct['both']:.1f}% usable in both dates",
            })
        else:
            report.write(f"  OK: {pct['both']:.1f}% usable in both dates.")
            summary_rows.append({
                "key": key, "label": sample["label"], "status": "PASS",
                "detail": f"{pct['both']:.1f}% usable in both dates",
            })

        report.write(f"  ({elapsed:.1f}s)")
        report.write("")

    # --- Full summary: every sample checked, not just the flagged ones ---
    report.write("=" * 72)
    report.write("Summary (all samples)")
    report.write("=" * 72)
    for row in summary_rows:
        report.write(f"{row['status']:10s} {row['key']:30s} {row['detail']}")

    report.write("")
    flagged_or_worse = [r for r in summary_rows if r["status"] != "PASS"]
    if flagged_or_worse:
        report.write(f"{len(flagged_or_worse)} of {len(summary_rows)} sample(s) need review:")
        for row in flagged_or_worse:
            report.write(f"  - {row['key']}: [{row['status']}] {row['detail']}")
    else:
        report.write(f"All {len(summary_rows)} sample(s) passed.")

    # --- Aggregate mosaic cost summary ---
    # The actual point of --no-mosaic and this whole tracking: how much
    # MORE tiles/time did mosaicking add, in total, across a real batch --
    # not just a per-sample anecdote. Run once with --no-mosaic and once
    # without, and compare this section between the two report files
    # directly.
    report.write("")
    report.write("=" * 72)
    report.write("Mosaic cost summary")
    report.write("=" * 72)
    if not enable_mosaic:
        report.write("Mosaicking was DISABLED for this run (--no-mosaic) -- no added cost to report.")
    else:
        report.write(f"Samples needing mosaic (either side): {samples_needing_mosaic} of {len(summary_rows)}")
        report.write(f"Total companion tiles fetched and used: {total_companion_tiles_used}")
        report.write(f"Total companion search time: {total_companion_search_time:.1f}s")
        report.write(f"Total companion read time: {total_companion_read_time:.1f}s")
        report.write(
            f"Total added time from mosaicking: "
            f"{total_companion_search_time + total_companion_read_time:.1f}s"
        )

    report.save(args.output)
    print(f"\nFull report written to: {os.path.abspath(args.output)}")

    sys.exit(1 if flagged_or_worse else 0)


if __name__ == "__main__":
    main()

"""
Scene loading utilities for Sentinel-2.
Loads and clips bands to the AOI and returns rioxarray-enabled DataArrays.

Performance changes vs. the original version:
- GDAL/rasterio HTTP settings tuned for cloud-optimized GeoTIFFs (fewer
  redundant requests, merged byte-range reads, local block caching).
- Bands are read concurrently (ThreadPoolExecutor) instead of one at a time,
  since GDAL releases the GIL during network I/O.
- Optional decimated reads via `max_dim`, so callers that only need a
  preview-resolution image (e.g. RGB thumbnails, index maps for display)
  don't pay for full 10m reads over the network.

Multi-tile mosaicking (see load_scene's enable_mosaic):
Sentinel-2 tiles sit on a fixed ~110km grid, independent of any AOI a user
draws. An AOI near a tile boundary can have a real chunk of it fall
outside whichever single tile a given STAC item belongs to -- no amount of
picking a "better" scene fixes this, since the missing area is the same
regardless of date (see core.change.coverage_overlap and
comparability_score's MGRS-tile-match priority, which fixes the *pairing*
half of this problem but not a single tile's own footprint gap). When a
primary tile doesn't fully cover the AOI, load_scene now searches for
other tiles from the SAME acquisition date (core.stac.search_companion_
tiles) and fills the gap from them. This adds real cost -- more searches,
more band reads, more reprojection -- only when it's actually needed: an
AOI that doesn't straddle a boundary triggers no companion search at all,
so the common case costs exactly what it did before this feature existed.
Every load_scene call returns a `load_info` dict quantifying that cost
(tiles fetched, time spent) specifically so it CAN be measured rather than
assumed -- see app/components/change_display.py's "Load cost" expander and
check_sample_coverage.py's --no-mosaic flag for an A/B comparison.
"""

from __future__ import annotations

import time
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from shapely.geometry import Polygon
from shapely.ops import transform
from rasterio.windows import from_bounds, Window
from rasterio.enums import Resampling
from rasterio import Affine
import rasterio
import pyproj
import numpy as np
import xarray as xr
import planetary_computer as pc

from core.stac import search_companion_tiles


# GDAL settings recommended for reading Cloud-Optimized GeoTIFFs (COGs) over
# HTTP (e.g. Planetary Computer / Azure Blob Storage). These cut down on
# redundant directory-listing and per-block requests, and merge nearby byte
# ranges into fewer round trips.
GDAL_HTTP_OPTS = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    VSI_CACHE=True,
    VSI_CACHE_SIZE=200_000_000,  # 200MB local block cache per read session
    GDAL_CACHEMAX=256,
)

# "swir1" (B11) added for NDMI / NDBI / BSI. It's a 20m-native band like
# swir2 and scl, so it also needs to flow through resample_bands.
ASSET_MAP = {
    "red":   "B04",
    "green": "B03",
    "blue":  "B02",
    "nir":   "B08",
    "swir1": "B11",
    "swir2": "B12",
    "scl":   "SCL",
}

# ESA's Processing Baseline 04.00, operational from 2022-01-25, added a
# constant BOA_ADD_OFFSET (-1000) to every reflectance band's digital
# numbers, to allow encoding negative surface reflectance over dark
# surfaces without clipping. Scenes processed under an older baseline don't
# have this offset. If a before/after comparison spans this date without
# correcting for it, every reflectance value in the newer scene is shifted
# by roughly -0.1 (after the 0.0001 scale factor) relative to the older
# one -- a large, spurious, uniform artifact easily mistaken for something
# else (e.g. a sensor difference) in a delta map.
#
# Planetary Computer doesn't expose BOA_ADD_OFFSET itself as a queryable
# STAC field (it's only in the per-scene XML metadata), but it does expose
# s2:processing_baseline, which is enough to know whether the correction
# applies -- the offset has been a flat -1000 for every band at every
# baseline >= 4.00 since introduction.
BOA_OFFSET_BASELINE_THRESHOLD = 4.0
BOA_OFFSET_VALUE = -1000.0

# Mosaicking defaults -- see load_scene.
DEFAULT_MAX_COMPANION_TILES = 3
DEFAULT_MOSAIC_COVERAGE_THRESHOLD = 0.999


def _boa_offset_for_item(item: Any) -> float:
    """
    Returns -1000.0 if this item's processing baseline is >= 4.00 (meaning
    ESA's BOA_ADD_OFFSET applies and needs to be subtracted back out before
    comparing against pre-2022-baseline scenes), else 0.0.

    Defaults to 0.0 (no correction) if the baseline can't be determined --
    incorrectly subtracting 1000 from a scene that never had the offset
    applied would introduce a new artifact rather than fixing one, so an
    unknown baseline is treated as "don't touch it" rather than guessed.

    Computed per-item (not just for the primary scene) since a companion
    tile fetched for mosaicking can have a different processing baseline
    than the primary -- each tile's own offset must be corrected
    independently before their values are merged.
    """
    baseline = item.properties.get("s2:processing_baseline")
    if baseline is None:
        return 0.0
    try:
        return BOA_OFFSET_VALUE if float(baseline) >= BOA_OFFSET_BASELINE_THRESHOLD else 0.0
    except (TypeError, ValueError):
        return 0.0


def _geometric_validity_mask(
    window: Window, overlap: Window, out_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Boolean mask (True = real source data, False = boundless-read padding),
    built purely from window geometry -- no extra pixel data or network
    cost beyond what _read_band already computes.

    `window` is the full requested read window in the source's native
    pixel space; `overlap` is its intersection with the source's actual
    extent (see _read_band, which computes both already for the existing
    overlap_fraction check). Both are axis-aligned rectangles in the same
    pixel-coordinate space, so mapping `overlap`'s bounds into the
    decimated `out_shape` grid -- the same linear scaling _read_band
    already applies to build the output affine transform -- gives exactly
    which output pixels are covered by real source data versus
    fill_value=0 padding. This is what load_scene's mosaicking uses to
    decide which pixels a companion tile is allowed to fill: only ones
    that are False here, never ones already backed by real primary data.
    """
    if window.width <= 0 or window.height <= 0:
        return np.zeros(out_shape, dtype=bool)

    scale_x = out_shape[1] / window.width
    scale_y = out_shape[0] / window.height

    col_start = (overlap.col_off - window.col_off) * scale_x
    col_end = col_start + overlap.width * scale_x
    row_start = (overlap.row_off - window.row_off) * scale_y
    row_end = row_start + overlap.height * scale_y

    col_start_i = max(0, int(np.floor(col_start)))
    col_end_i = min(out_shape[1], int(np.ceil(col_end)))
    row_start_i = max(0, int(np.floor(row_start)))
    row_end_i = min(out_shape[0], int(np.ceil(row_end)))

    mask = np.zeros(out_shape, dtype=bool)
    if col_end_i > col_start_i and row_end_i > row_start_i:
        mask[row_start_i:row_end_i, col_start_i:col_end_i] = True
    return mask


def _read_band(
    name: str,
    href: str,
    aoi: Polygon,
    max_dim: Optional[int],
    offset: float = 0.0,
) -> Tuple[str, xr.DataArray, float, np.ndarray]:
    """
    Read a single band, clipped to the AOI, optionally decimated on read.

    max_dim, if given, caps the longer side of the output array to this many
    pixels by asking GDAL to decimate during the read (much cheaper than
    reading full resolution and downsampling afterwards). Pass None for a
    full-resolution read.

    offset, if nonzero, is added to the raw digital numbers immediately
    after reading (see _boa_offset_for_item) -- applied before any
    decimation-related resampling math, since a constant additive shift
    commutes with bilinear interpolation, so order doesn't matter here.

    The href is re-signed here, immediately before use, rather than relying
    on the signed href already attached to the item from search time. Items
    (and their hrefs) can sit in Streamlit's cache for up to an hour --
    signing fresh on every read means a stale cached item never causes a
    403 from an expired SAS token.

    Returns
    -------
    tuple(str, xr.DataArray, float, np.ndarray)
        (band name, data, overlap_fraction, valid_mask).
        overlap_fraction is the fraction (0-1) of the AOI window that
        actually falls within this granule's real data extent -- a single
        continuous ratio, kept for exact backward-compatible behavior with
        the pre-mosaic coverage check.
        valid_mask is the discretized, per-pixel version of the same
        geometric fact (see _geometric_validity_mask), used by load_scene
        to decide where a companion tile is allowed to fill in data.
    """
    href = pc.sign(href)

    with rasterio.Env(**GDAL_HTTP_OPTS), rasterio.open(href) as src:

        project = pyproj.Transformer.from_crs(
            "EPSG:4326", src.crs, always_xy=True
        ).transform
        aoi_proj = transform(project, aoi)

        window = from_bounds(*aoi_proj.bounds, transform=src.transform)

        # How much of the requested window actually overlaps the source
        # raster's real extent? Computed from window geometry alone (cheap,
        # no pixel data needed) BEFORE reading, so this is available even
        # when overlap is partial or zero.
        full_window = Window(0, 0, src.width, src.height)
        try:
            overlap = window.intersection(full_window)
            overlap_fraction = (overlap.width * overlap.height) / (
                window.width * window.height
            )
        except rasterio.errors.WindowError:
            overlap = Window(0, 0, 0, 0)
            overlap_fraction = 0.0

        out_shape = None
        if max_dim is not None:
            scale = min(1.0, max_dim / max(window.height, window.width))
            out_shape = (
                max(1, int(round(window.height * scale))),
                max(1, int(round(window.width * scale))),
            )
        final_out_shape = out_shape if out_shape is not None else (
            max(1, int(round(window.height))), max(1, int(round(window.width))),
        )

        # boundless=True + fill_value=0 is the key fix here: without it,
        # rasterio silently clips a partially-out-of-bounds window to the
        # dataset's actual extent, but out_shape above was computed from the
        # ORIGINAL (uncropped) window size -- asking GDAL to decimate a
        # smaller-than-expected clipped read into a larger requested
        # out_shape produced a shape/scale mismatch that showed up as a
        # repeating/tiled visual artifact. boundless=True guarantees the
        # returned array always matches out_shape (or window) exactly,
        # padding any out-of-bounds portion with fill_value instead.
        data = src.read(
            1,
            window=window,
            out_shape=out_shape,
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )

        if offset:
            # Cast to a signed dtype BEFORE subtracting -- the source data
            # is unsigned (uint16), and low-DN pixels minus 1000 would
            # silently wrap around to huge positive values instead of going
            # negative if left unsigned.
            data = data.astype("int32") + int(offset)

        win_transform = src.window_transform(window)
        if out_shape is not None:
            scale_x = window.width / out_shape[1]
            scale_y = window.height / out_shape[0]
            win_transform = win_transform * Affine.scale(scale_x, scale_y)

        res_x = win_transform.a
        res_y = win_transform.e
        start_x = win_transform.c + res_x / 2
        start_y = win_transform.f + res_y / 2

        xs = (start_x + np.arange(data.shape[1]) * res_x)
        ys = (start_y + np.arange(data.shape[0]) * res_y)

        da = xr.DataArray(
            data=data,
            dims=["y", "x"],
            coords={"y": ys, "x": xs},
            name=name,
        )
        da = da.rio.write_crs(src.crs)
        da = da.rio.write_transform(win_transform)

        valid_mask = _geometric_validity_mask(window, overlap, final_out_shape)

    return name, da, overlap_fraction, valid_mask


def _read_item_bands(
    item: Any,
    aoi: Polygon,
    max_dim: Optional[int],
    max_workers: int,
) -> Tuple[Dict[str, xr.DataArray], Dict[str, np.ndarray], float]:
    """
    Read every band in ASSET_MAP for a single STAC item, concurrently, with
    each band's BOA offset correction applied and a geometric validity mask
    computed per band (see _geometric_validity_mask).

    Shared by load_scene for both the primary scene and any companion
    tiles fetched for mosaicking, so the two code paths read bands
    identically and can't drift apart.

    Returns
    -------
    tuple(Dict[str, xr.DataArray], Dict[str, np.ndarray], float)
        (bands, valid_masks, overlap_fraction) -- valid_masks has the same
        keys as bands. overlap_fraction is the geometric AOI-window
        overlap fraction, identical across bands in practice (all bands of
        one item share a footprint); min() is taken defensively.
    """
    boa_offset = _boa_offset_for_item(item)

    jobs = [
        (
            name,
            item.assets[asset_key].href,
            aoi,
            max_dim,
            0.0 if name == "scl" else boa_offset,  # never offset the SCL classification layer
        )
        for name, asset_key in ASSET_MAP.items()
    ]

    bands: Dict[str, xr.DataArray] = {}
    valid_masks: Dict[str, np.ndarray] = {}
    overlap_fractions = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for name, da, overlap_fraction, valid_mask in ex.map(lambda job: _read_band(*job), jobs):
            bands[name] = da
            valid_masks[name] = valid_mask
            overlap_fractions.append(overlap_fraction)

    coverage_fraction = min(overlap_fractions) if overlap_fractions else 0.0
    return bands, valid_masks, coverage_fraction


def _mosaic_fill(
    primary_da: xr.DataArray,
    primary_valid: np.ndarray,
    companion_da: xr.DataArray,
    companion_valid: np.ndarray,
    resampling: Resampling,
) -> Tuple[xr.DataArray, np.ndarray]:
    """
    Fill primary_da's invalid pixels (per primary_valid) using
    companion_da, reprojected onto primary_da's grid.

    Pixels already valid in the primary are NEVER overwritten -- merging
    several companions in sequence (see load_scene's loop) only ever fills
    previously-empty pixels, so the order companions are tried in doesn't
    change the final result for any pixel more than one companion could
    have filled.

    Uses core.change.align_to_reference for the reprojection, imported
    locally (not at module level) purely to keep core.load's and
    core.change's import graph one-directional -- core.change has no
    reason to ever import core.load, and this keeps it that way even if
    core.load's own imports change later.

    Returns
    -------
    tuple(xr.DataArray, np.ndarray)
        (merged_da, merged_valid), on primary_da's grid/shape.
    """
    from core.change import align_to_reference

    companion_values_aligned = align_to_reference(
        companion_da.values, companion_da, primary_da, resampling=resampling
    )

    companion_valid_da = xr.DataArray(
        companion_valid.astype("uint8"), dims=companion_da.dims, coords=companion_da.coords
    )
    companion_valid_da = companion_valid_da.rio.write_crs(companion_da.rio.crs)
    companion_valid_da = companion_valid_da.rio.write_transform(companion_da.rio.transform())
    companion_valid_aligned = align_to_reference(
        companion_valid_da.values, companion_valid_da, primary_da, resampling=Resampling.nearest
    ) > 0

    fill_mask = (~primary_valid) & companion_valid_aligned

    merged_values = primary_da.values.copy()
    merged_values[fill_mask] = companion_values_aligned[fill_mask]
    merged_valid = primary_valid | companion_valid_aligned

    merged_da = xr.DataArray(
        merged_values, dims=primary_da.dims, coords=primary_da.coords, name=primary_da.name
    )
    merged_da = merged_da.rio.write_crs(primary_da.rio.crs)
    merged_da = merged_da.rio.write_transform(primary_da.rio.transform())
    return merged_da, merged_valid


def load_scene(
    item: Any,
    aoi: Polygon,
    max_dim: Optional[int] = None,
    max_workers: int = len(ASSET_MAP),
    enable_mosaic: bool = True,
    max_companion_tiles: int = DEFAULT_MAX_COMPANION_TILES,
    mosaic_coverage_threshold: float = DEFAULT_MOSAIC_COVERAGE_THRESHOLD,
) -> Tuple[Dict[str, xr.DataArray], float, Dict[str, Any]]:
    """
    Load Sentinel-2 bands clipped to the AOI, fetched concurrently, with
    optional multi-tile mosaicking when the AOI isn't fully covered by a
    single tile's granule (see this module's docstring for why that
    happens and why it can't be fixed by picking a different date).

    This is strictly additive to the primary tile's own data: a pixel
    already valid in the primary is never replaced by a companion's value,
    and if the primary already meets mosaic_coverage_threshold on its own,
    no companion search happens at all -- an AOI that doesn't straddle a
    tile boundary costs exactly what it did before this feature existed.

    Parameters
    ----------
    item : pystac.Item
        STAC item containing Sentinel-2 band assets (the "primary" tile).
    aoi : shapely.geometry.Polygon
        AOI in EPSG:4326.
    max_dim : int, optional
        Cap the longer side of each band to this many pixels via a decimated
        read. None (default) preserves original full-resolution behavior.
    max_workers : int
        Number of bands fetched concurrently per tile (primary or
        companion). Defaults to one thread per band in ASSET_MAP.
    enable_mosaic : bool
        Set False to restore the exact original single-tile-only behavior
        (same bands, same coverage_fraction as before this feature
        existed) -- used for A/B cost/coverage comparisons; see
        check_sample_coverage.py's --no-mosaic flag.
    max_companion_tiles : int
        Hard cap on how many extra tiles will ever be fetched for one
        load_scene call, regardless of how poor coverage still is after
        that many -- bounds worst-case added network cost.
    mosaic_coverage_threshold : float
        Stop fetching companions once the minimum per-band coverage
        fraction reaches this value (0-1, default 0.999). Doesn't need to
        reach 1.0 exactly: a small residual gap left after this is
        typically genuine within-swath nodata or cloud, not a tiling
        artifact, and isn't worth another full tile fetch to chase.

    Returns
    -------
    tuple(Dict[str, xr.DataArray], float, Dict[str, Any])
        (bands, coverage_fraction, load_info).
        coverage_fraction is the minimum AOI-coverage fraction across all
        bands, AFTER any mosaicking -- callers should warn when this is
        well below 1.0, same meaning as before this feature existed.
        load_info is a diagnostics dict for measuring this feature's cost:
          - "tiles_used": int, 1 + however many companions actually
            contributed a pixel (fetching a companion that turned out not
            to help still costs time, but only ones that helped count as
            "used").
          - "mosaic_attempted": bool, whether coverage was below threshold
            so a companion search was even tried.
          - "mosaic_applied": bool, whether any companion actually filled
            a pixel.
          - "companions_found" / "companions_used": int.
          - "primary_geom_coverage": float, coverage from the primary
            tile alone, before any mosaicking (compare against
            "final_coverage" to see the improvement).
          - "final_coverage": float, same value as the second return
            element, duplicated here for convenience.
          - "companion_search_time_s" / "companion_read_time_s": float,
            seconds spent specifically on the mosaic machinery (0.0 if
            never attempted).
          - "total_load_time_s": float, wall time for the whole call.
    """
    t_start = time.time()

    bands, valid_masks, coverage_fraction = _read_item_bands(item, aoi, max_dim, max_workers)

    load_info: Dict[str, Any] = {
        "tiles_used": 1,
        "mosaic_attempted": False,
        "mosaic_applied": False,
        "companions_found": 0,
        "companions_used": 0,
        "primary_geom_coverage": coverage_fraction,
        "final_coverage": coverage_fraction,
        "companion_search_time_s": 0.0,
        "companion_read_time_s": 0.0,
        "total_load_time_s": 0.0,
    }

    if enable_mosaic and coverage_fraction < mosaic_coverage_threshold:
        load_info["mosaic_attempted"] = True

        t_search0 = time.time()
        try:
            companions = search_companion_tiles(aoi, item, max_items=max_companion_tiles)
        except Exception:
            # A network hiccup searching for companions shouldn't break an
            # otherwise-successful primary-tile load -- fall back to
            # single-tile behavior for this call.
            companions = []
        load_info["companion_search_time_s"] = time.time() - t_search0
        load_info["companions_found"] = len(companions)

        for companion_item in companions:
            t_read0 = time.time()
            try:
                companion_bands, companion_valid_masks, _ = _read_item_bands(
                    companion_item, aoi, max_dim, max_workers
                )
            except Exception:
                # Skip a companion that fails to read rather than failing
                # the whole load -- the primary tile's data still stands.
                continue
            load_info["companion_read_time_s"] += time.time() - t_read0

            used_this_companion = False
            new_valid_fracs = []
            for name in bands:
                resampling = Resampling.nearest if name == "scl" else Resampling.bilinear
                merged_da, merged_valid = _mosaic_fill(
                    bands[name], valid_masks[name],
                    companion_bands[name], companion_valid_masks[name],
                    resampling=resampling,
                )
                if not np.array_equal(merged_valid, valid_masks[name]):
                    used_this_companion = True
                bands[name] = merged_da
                valid_masks[name] = merged_valid
                new_valid_fracs.append(float(np.mean(merged_valid)))

            if used_this_companion:
                load_info["companions_used"] += 1
                load_info["tiles_used"] += 1

            coverage_fraction = min(new_valid_fracs) if new_valid_fracs else coverage_fraction
            load_info["final_coverage"] = coverage_fraction

            if coverage_fraction >= mosaic_coverage_threshold:
                break  # good enough -- stop fetching more companions

        load_info["mosaic_applied"] = load_info["companions_used"] > 0

    load_info["total_load_time_s"] = time.time() - t_start
    return bands, coverage_fraction, load_info

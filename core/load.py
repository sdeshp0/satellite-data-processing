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
"""

from __future__ import annotations

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


def _boa_offset_for_item(item: Any) -> float:
    """
    Returns -1000.0 if this item's processing baseline is >= 4.00 (meaning
    ESA's BOA_ADD_OFFSET applies and needs to be subtracted back out before
    comparing against pre-2022-baseline scenes), else 0.0.

    Defaults to 0.0 (no correction) if the baseline can't be determined --
    incorrectly subtracting 1000 from a scene that never had the offset
    applied would introduce a new artifact rather than fixing one, so an
    unknown baseline is treated as "don't touch it" rather than guessed.
    """
    baseline = item.properties.get("s2:processing_baseline")
    if baseline is None:
        return 0.0
    try:
        return BOA_OFFSET_VALUE if float(baseline) >= BOA_OFFSET_BASELINE_THRESHOLD else 0.0
    except (TypeError, ValueError):
        return 0.0


def _read_band(
    name: str,
    href: str,
    aoi: Polygon,
    max_dim: Optional[int],
    offset: float = 0.0,
) -> Tuple[str, xr.DataArray, float]:
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
    tuple(str, xr.DataArray, float)
        (band name, data, overlap_fraction) -- overlap_fraction is the
        fraction (0-1) of the AOI window that actually falls within this
        granule's real data extent. Sentinel-2 granules aren't always
        fully covered by real data at swath edges; if the AOI mostly
        misses this particular acquisition's actual coverage,
        overlap_fraction will be well below 1.0, which is a real signal
        worth surfacing (see load_scene) rather than silently reading
        whatever happens to be there.
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
            overlap_fraction = 0.0

        out_shape = None
        if max_dim is not None:
            scale = min(1.0, max_dim / max(window.height, window.width))
            out_shape = (
                max(1, int(round(window.height * scale))),
                max(1, int(round(window.width * scale))),
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

    return name, da, overlap_fraction


def load_scene(
    item: Any,
    aoi: Polygon,
    max_dim: Optional[int] = None,
    max_workers: int = len(ASSET_MAP),
) -> Tuple[Dict[str, xr.DataArray], float]:
    """
    Load Sentinel-2 bands clipped to the AOI, fetched concurrently.

    Parameters
    ----------
    item : pystac.Item
        STAC item containing Sentinel-2 band assets.
    aoi : shapely.geometry.Polygon
        AOI in EPSG:4326.
    max_dim : int, optional
        Cap the longer side of each band to this many pixels via a decimated
        read. None (default) preserves original full-resolution behavior.
    max_workers : int
        Number of bands fetched concurrently. Defaults to one thread per
        band in ASSET_MAP.

    Returns
    -------
    tuple(Dict[str, xr.DataArray], float)
        (bands, coverage_fraction). coverage_fraction is the minimum
        AOI-overlap fraction across all bands (should be nearly identical
        band-to-band; the min is taken defensively) -- callers should warn
        when this is well below 1.0, since it means the AOI only partially
        falls within this granule's actual data footprint, not just its
        nominal tile boundary.
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
    overlap_fractions = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for name, da, overlap_fraction in ex.map(lambda job: _read_band(*job), jobs):
            bands[name] = da
            overlap_fractions.append(overlap_fraction)

    coverage_fraction = min(overlap_fractions) if overlap_fractions else 0.0
    return bands, coverage_fraction

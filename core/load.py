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
from rasterio.windows import from_bounds
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


def _read_band(
    name: str,
    href: str,
    aoi: Polygon,
    max_dim: Optional[int],
) -> Tuple[str, xr.DataArray]:
    """
    Read a single band, clipped to the AOI, optionally decimated on read.

    max_dim, if given, caps the longer side of the output array to this many
    pixels by asking GDAL to decimate during the read (much cheaper than
    reading full resolution and downsampling afterwards). Pass None for a
    full-resolution read.

    The href is re-signed here, immediately before use, rather than relying
    on the signed href already attached to the item from search time. Items
    (and their hrefs) can sit in Streamlit's cache for up to an hour --
    signing fresh on every read means a stale cached item never causes a
    403 from an expired SAS token.
    """
    href = pc.sign(href)

    with rasterio.Env(**GDAL_HTTP_OPTS), rasterio.open(href) as src:

        project = pyproj.Transformer.from_crs(
            "EPSG:4326", src.crs, always_xy=True
        ).transform
        aoi_proj = transform(project, aoi)

        window = from_bounds(*aoi_proj.bounds, transform=src.transform)

        out_shape = None
        if max_dim is not None:
            scale = min(1.0, max_dim / max(window.height, window.width))
            out_shape = (
                max(1, int(round(window.height * scale))),
                max(1, int(round(window.width * scale))),
            )

        data = src.read(
            1,
            window=window,
            out_shape=out_shape,
            resampling=Resampling.bilinear,
        )

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

    return name, da


def load_scene(
    item: Any,
    aoi: Polygon,
    max_dim: Optional[int] = None,
    max_workers: int = len(ASSET_MAP),
) -> Dict[str, xr.DataArray]:
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
    Dict[str, xr.DataArray]
        Mapping of band name -> clipped DataArray with CRS + transform.
    """
    jobs = [
        (name, item.assets[asset_key].href, aoi, max_dim)
        for name, asset_key in ASSET_MAP.items()
    ]

    bands: Dict[str, xr.DataArray] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for name, da in ex.map(lambda job: _read_band(*job), jobs):
            bands[name] = da

    return bands

"""
Scene loading utilities for Sentinel‑2.
Loads and clips bands to the AOI and returns rioxarray‑enabled DataArrays.
"""

from __future__ import annotations

from typing import Dict, Any
from shapely.geometry import Polygon
from shapely.ops import transform
from rasterio.windows import from_bounds
import rasterio
import pyproj
import numpy as np
import xarray as xr


def load_scene(item: Any, aoi: Polygon) -> Dict[str, xr.DataArray]:
    """
    Load Sentinel‑2 bands clipped to the AOI.

    Parameters
    ----------
    item : pystac.Item
        STAC item containing Sentinel‑2 band assets.
    aoi : shapely.geometry.Polygon
        AOI in EPSG:4326.

    Returns
    -------
    Dict[str, xr.DataArray]
        Dictionary of band name → clipped DataArray with CRS + transform.
    """

    # Asset mapping (unchanged behavior)
    asset_map = {
        "red":   "B04",
        "green": "B03",
        "blue":  "B02",
        "nir":   "B08",
        "swir2": "B12",
        "scl":   "SCL",
    }

    bands: Dict[str, xr.DataArray] = {}

    for name, asset_key in asset_map.items():
        href = item.assets[asset_key].href

        with rasterio.open(href) as src:

            # Reproject AOI from EPSG:4326 → raster CRS
            project = pyproj.Transformer.from_crs(
                "EPSG:4326", src.crs, always_xy=True
            ).transform
            aoi_proj = transform(project, aoi)

            # Clip window in raster CRS
            window = from_bounds(*aoi_proj.bounds, transform=src.transform)

            # Read clipped data
            data = src.read(1, window=window)

            # Compute window transform
            win_transform = src.window_transform(window)

            # Pixel resolution
            res_x = win_transform.a
            res_y = win_transform.e

            # Pixel center coordinates
            start_x = win_transform.c + res_x / 2
            start_y = win_transform.f + res_y / 2

            xs = np.arange(start_x, start_x + data.shape[1] * res_x, res_x)
            ys = np.arange(start_y, start_y + data.shape[0] * res_y, res_y)

            # Build DataArray
            da = xr.DataArray(
                data=data,
                dims=["y", "x"],
                coords={"y": ys, "x": xs},
                name=name,
            )

            # Attach CRS + transform for rioxarray
            da = da.rio.write_crs(src.crs)
            da = da.rio.write_transform(win_transform)

            bands[name] = da

    return bands

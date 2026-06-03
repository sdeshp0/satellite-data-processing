"""
Scene Loading utilities
"""

import rasterio
from rasterio.windows import from_bounds
import pyproj
from shapely.ops import transform
import xarray as xr
import numpy as np

def load_scene(item, aoi):
    """
    Load the Sentinel-2 scene bands clipped to the AOI.
    AOI is expected to be a Shapely geometry in EPSG:4326.
    """

    # Open the asset (assuming item.assets["B04"], etc.)
    red_href = item.assets["B04"].href
    green_href = item.assets["B03"].href
    blue_href = item.assets["B02"].href
    nir_href = item.assets["B08"].href
    swir2_href = item.assets["B12"].href
    scl_href = item.assets["SCL"].href

    bands = {}

    for name, href in [
        ("red", red_href),
        ("green", green_href),
        ("blue", blue_href),
        ("nir", nir_href),
        ("swir2", swir2_href),
        ("scl", scl_href)
    ]:

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

            # Get precise affine transform and bounds for this window
            win_transform = src.window_transform(window)

            # Generate X and Y pixel center coordinates

            res_x = win_transform.a  # pixel width
            res_y = win_transform.e  # pixel height

            # Calculate pixel centers to align with Xarray standard
            start_x = win_transform.c + res_x / 2
            start_y = win_transform.f + res_y / 2

            xs = np.arange(start_x, start_x + (data.shape[1] * res_x), res_x)
            ys = np.arange(start_y, start_y + (data.shape[0] * res_y), res_y)

            # Convert to Xarray DataArray
            da = xr.DataArray(
                data=data,
                dims=["y", "x"],
                coords={"y":ys, "x":xs},
                name=name
            )

            # Inject metadata to activate .rio accessors
            da = da.rio.write_crs(src.crs)
            da = da.rio.write_transform(win_transform)

            # store rioxarray-ready DataArray
            bands[name] = da

    return bands





"""
Small utilities: AOI helpers, geometry conversions, etc.
"""

from shapely.geometry import box, mapping
import rasterio
import rioxarray
import xarray as xr


def aoi_from_bounds(bounds):
    """
    Convert numeric bounds (minx, miny, maxx, maxy) to GeoJSON geometry.
    """
    geom = box(*bounds)
    return mapping(geom)


def scale_bands(bands):
    """
    Apply Scale Reflectance Factor to Bands
    """
    scale = 0.0001
    for name in bands.keys():
        data = bands[name]
        data = data.astype("float32") * scale
        bands[name] = data

    return bands


def resample_bands(bands):
    """
    Resample 20m bands (SWIR2 and SCL) to match the other 10m bands
    """
    for name in bands.keys():
        data = bands[name]

        # Resample 20m bands to match 10m bands
        if name in ["swir2", "scl"]:
            data = data.rio.reproject_match(bands["nir"])
            bands[name] = data
    return bands


def apply_cloud_mask(bands):
    """
    Apply cloud masking based on SCL band

    SCL classes to mask:
    3 = cloud shadow
    8 = medium cloud
    9 = high cloud
    10 = cirrus
    11 = snow/ice
    """
    cloud_classes = [3, 8, 9, 10, 11]  # shadow, medium cloud, high cloud, cirrus, snow

    scl = bands["scl"]
    cloud_mask = ~scl.isin(cloud_classes)

    for name in bands.keys():
        if name not in ["scl"]:
            data = bands[name]
            data = data.where(cloud_mask)
            bands[name] = data
    return bands

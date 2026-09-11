"""
Utility functions for AOI handling, band scaling, resampling,
and cloud masking for Sentinel‑2 data.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Iterable
from shapely.geometry import box, mapping
import xarray as xr
import rioxarray  # required for .rio accessor


def aoi_from_bounds(bounds: Tuple[float, float, float, float]) -> Dict[str, Any]:
    """
    Convert numeric bounds (minx, miny, maxx, maxy) into a GeoJSON geometry.

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) in EPSG:4326.

    Returns
    -------
    dict
        GeoJSON geometry mapping.
    """
    geom = box(*bounds)
    return mapping(geom)


def scale_bands(
    bands: Dict[str, xr.DataArray],
    exclude: Iterable[str] = ("scl",),
) -> Dict[str, xr.DataArray]:
    """
    Apply Sentinel‑2 reflectance scale factor (0.0001) to reflectance bands.

    Parameters
    ----------
    bands : dict
        Mapping of band name -> DataArray.
    exclude : iterable of str
        Band names to leave unscaled. Defaults to ("scl",), since the Scene
        Classification Layer holds categorical class codes (0-11), not
        reflectance -- multiplying it by the reflectance scale factor would
        silently corrupt the classification (e.g. class 4 "Vegetation"
        becomes 0.0004), breaking anything downstream that reads SCL values
        (cloud masking, land-cover breakdowns, etc.).

    Returns
    -------
    dict
        Updated mapping with reflectance bands scaled to float32; excluded
        bands passed through unchanged.
    """
    scale = 0.0001
    out = {}

    for name, da in bands.items():
        if name in exclude:
            out[name] = da
        else:
            out[name] = da.astype("float32") * scale

    return out


def resample_bands(bands: Dict[str, xr.DataArray]) -> Dict[str, xr.DataArray]:
    """
    Resample 20m bands (SWIR1, SWIR2, SCL) to match the 10m NIR band.

    Parameters
    ----------
    bands : dict
        Mapping of band name -> DataArray.

    Returns
    -------
    dict
        Updated mapping with resampled DataArrays.
    """
    out = dict(bands)

    # NIR is the 10m reference grid
    nir = bands["nir"]

    for name in ["swir1", "swir2", "scl"]:
        if name in bands:
            out[name] = bands[name].rio.reproject_match(nir)

    return out


def apply_cloud_mask(bands: Dict[str, xr.DataArray]) -> Dict[str, xr.DataArray]:
    """
    Apply cloud masking using the Sentinel‑2 Scene Classification Layer (SCL).

    Masked SCL classes:
    - 3  = cloud shadow
    - 8  = medium cloud
    - 9  = high cloud
    - 10 = cirrus
    - 11 = snow/ice

    Parameters
    ----------
    bands : dict
        Mapping of band name → DataArray.

    Returns
    -------
    dict
        Updated mapping with cloud‑masked DataArrays.
    """
    cloud_classes = [3, 8, 9, 10, 11]

    scl = bands["scl"]
    cloud_mask = ~scl.isin(cloud_classes)

    out = {}

    for name, da in bands.items():
        if name == "scl":
            out[name] = da
        else:
            out[name] = da.where(cloud_mask)

    return out

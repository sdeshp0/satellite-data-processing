"""
Utility functions for AOI handling, band scaling, resampling,
and cloud masking for Sentinel‑2 data.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Iterable
from shapely.geometry import box, mapping
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401  # required for .rio accessor


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


# SCL class 0 ("No Data"). Also what core.load's boundless reads report for
# AOI pixels outside a granule's footprint, so masking it covers both.
NO_DATA_CLASS = 0


def apply_nodata_mask(bands: Dict[str, xr.DataArray]) -> Dict[str, xr.DataArray]:
    """
    Set reflectance bands to NaN wherever SCL is "No Data" (class 0).

    Unlike apply_cloud_mask this isn't optional: those pixels hold either
    fill values or (after the processing-baseline offset) a spurious -0.1
    reflectance, so leaving them in skews statistics, index maps and
    feature masks. The SCL band itself is passed through unchanged so its
    breakdown can still report how much of the AOI had no data.
    """
    valid = bands["scl"] != NO_DATA_CLASS

    return {
        name: da if name == "scl" else da.where(valid)
        for name, da in bands.items()
    }


# SCL classes left out when checking for negative reflectance: no data,
# cloud shadow, clouds, cirrus, snow, and water (dark enough to dip just
# below zero even in a well-corrected scene).
_QUALITY_SKIP_CLASSES = (0, 3, 6, 8, 9, 10, 11)
# Above this share of negative blue or green reflectance over clear land,
# the scene's atmospheric correction has overshot (New Orleans,
# 2023-10-14: 92% of blue below zero, in both ESA processings), and colours
# and indices from it are unreliable. Well-corrected scenes sit near 0%.
NEGATIVE_REFLECTANCE_WARN_SHARE = 0.10


def negative_reflectance_share(
    bands: Dict[str, xr.DataArray], names: Iterable[str] = ("blue", "green")
) -> Dict[str, float]:
    """
    Share of clear land pixels with negative reflectance, per band, from
    scaled bands (see scale_bands). 0.0 when there is no clear land.
    """
    land = ~np.isin(bands["scl"].values, _QUALITY_SKIP_CLASSES)
    out = {}
    for name in names:
        values = bands[name].values[land]
        values = values[np.isfinite(values)]
        out[name] = float(np.mean(values < 0)) if values.size else 0.0
    return out

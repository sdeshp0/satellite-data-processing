"""
Spectral index computations for Sentinel-2.

Compute NDVI, NBR, and NDWI from Xarray DataArrays.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401  # keep for .rio accessor


def _safe_division(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Perform safe division, avoiding division by zero.
    """
    eps = 1e-6
    return numerator / (denominator + eps)


def compute_indices(bands: Dict[str, xr.DataArray]) -> Dict[str, np.ndarray]:
    """
    Compute NDVI, NBR, and NDWI from Sentinel-2 bands.

    Parameters
    ----------
    bands : dict
        Mapping of band name -> Xarray DataArray. Expected keys:
        - "red", "green", "nir", "swir2", "scl"

    Returns
    -------
    dict
        Mapping of index name -> numpy array:
        - "ndvi": (NIR - RED) / (NIR + RED)
        - "nbr":  (NIR - SWIR2) / (NIR + SWIR2)
        - "ndwi": (GREEN - NIR) / (GREEN + NIR)   [McFeeters, 1996]

    """
    red = bands["red"].values
    green = bands["green"].values
    nir = bands["nir"].values
    swir2 = bands["swir2"].values

    ndvi = _safe_division(nir - red, nir + red)
    nbr = _safe_division(nir - swir2, nir + swir2)
    ndwi = _safe_division(green - nir, green + nir)

    return {
        "ndvi": ndvi,
        "nbr": nbr,
        "ndwi": ndwi,
    }

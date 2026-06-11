"""
Spectral index computations for Sentinel‑2.

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
    Compute NDVI, NBR, and NDWI from Sentinel‑2 bands.

    Parameters
    ----------
    bands : dict
        Mapping of band name → Xarray DataArray. Expected keys:
        - "red"
        - "nir"
        - "swir2"
        - "scl"

    Returns
    -------
    dict
        Mapping of index name → numpy array:
        - "ndvi"
        - "nbr"
        - "ndwi"
    """
    # Extract numpy arrays
    red = bands["red"].values
    nir = bands["nir"].values
    swir2 = bands["swir2"].values
    scl = bands["scl"].values  # kept for API consistency, not used here

    # NDVI = (NIR - RED) / (NIR + RED)
    ndvi = _safe_division(nir - red, nir + red)

    # NBR = (NIR - SWIR2) / (NIR + SWIR2)
    nbr = _safe_division(nir - swir2, nir + swir2)

    # NDWI (variant using NIR/SWIR2)
    ndwi = _safe_division(nir - swir2, nir + swir2)

    return {
        "ndvi": ndvi,
        "nbr": nbr,
        "ndwi": ndwi,
    }

"""
Spectral index computations for Sentinel‑2.

Compute NDVI, NBR, and NDWI from pre‑loaded bands.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import rioxarray  # noqa: F401  # kept for .rio accessor on arrays


def _safe_division(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Perform safe division, avoiding division by zero.
    """
    eps = 1e-6
    return numerator / (denominator + eps)


def compute_indices(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute NDVI, NBR, and NDWI from Sentinel‑2 bands.

    Parameters
    ----------
    bands : dict
        Mapping of band name → numpy array. Expected keys:
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
    red = bands["red"]
    nir = bands["nir"]
    swir2 = bands["swir2"]
    scl = bands["scl"]

    # NDVI = (NIR - RED) / (NIR + RED)
    ndvi = _safe_division(nir - red, nir + red)

    # NBR = (NIR - SWIR2) / (NIR + SWIR2)
    nbr = _safe_division(nir - swir2, nir + swir2)

    # NDWI (one common variant) = (NIR - SWIR2) / (NIR + SWIR2)
    # You can swap to green/NIR if desired later without changing the API.
    ndwi = _safe_division(nir - swir2, nir + swir2)

    # Optional: mask out invalid SCL classes (kept minimal to preserve behavior)
    # Here we simply keep the arrays as‑is; masking is handled upstream/downstream.

    return {
        "ndvi": ndvi,
        "nbr": nbr,
        "ndwi": ndwi,
    }

"""
Spectral index computations for Sentinel-2.

Computes eight indices from Xarray DataArrays:
- NDVI  : general vegetation vigor
- EVI   : vegetation, corrected for canopy background / atmosphere
- SAVI  : vegetation, corrected for soil brightness (sparse cover)
- NBR   : burn severity
- NDMI  : vegetation moisture content
- NDWI  : surface water
- NDBI  : built-up / impervious surfaces
- BSI   : bare soil
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401  # keep for .rio accessor


# Soil brightness correction factor for SAVI. 0.5 is the standard default,
# suited for a wide range of vegetation densities.
SAVI_L = 0.5


def _safe_division(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Perform safe division, avoiding division by zero.
    """
    eps = 1e-6
    return numerator / (denominator + eps)


def compute_indices(bands: Dict[str, xr.DataArray]) -> Dict[str, np.ndarray]:
    """
    Compute spectral indices from Sentinel-2 bands.

    Parameters
    ----------
    bands : dict
        Mapping of band name -> Xarray DataArray. Expected keys:
        - "red", "green", "blue", "nir", "swir1", "swir2", "scl"
        All reflectance bands are assumed to already be scaled to
        approximately 0-1 (see core.utils.scale_bands).

    Returns
    -------
    dict
        Mapping of index name -> numpy array. All indices nominally range
        -1 to 1 unless noted otherwise:

        - "ndvi": (NIR - RED) / (NIR + RED)
        - "evi":  2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
                  Typically 0-1 over vegetation; less prone to saturating
                  over dense canopy than NDVI, and partially corrects for
                  atmospheric and canopy background effects.
        - "savi": (1 + L) * (NIR - RED) / (NIR + RED + L), L = 0.5
                  Like NDVI, but reduces the influence of exposed soil --
                  more reliable than NDVI over sparse vegetation.
        - "nbr":  (NIR - SWIR2) / (NIR + SWIR2)
        - "ndmi": (NIR - SWIR1) / (NIR + SWIR1)
                  Vegetation canopy water content; useful for drought /
                  fuel-moisture assessment.
        - "ndwi": (GREEN - NIR) / (GREEN + NIR)   [McFeeters, 1996]
        - "ndbi": (SWIR1 - NIR) / (SWIR1 + NIR)
                  Higher values indicate impervious / built-up surfaces.
        - "bsi":  ((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))
                  Higher values indicate exposed bare soil.
    """
    red = bands["red"].values
    green = bands["green"].values
    blue = bands["blue"].values
    nir = bands["nir"].values
    swir1 = bands["swir1"].values
    swir2 = bands["swir2"].values
    # scl = bands["scl"].values  # not used for index math; see core.stats

    ndvi = _safe_division(nir - red, nir + red)
    evi = 2.5 * _safe_division(nir - red, nir + 6 * red - 7.5 * blue + 1)
    savi = (1 + SAVI_L) * _safe_division(nir - red, nir + red + SAVI_L)
    nbr = _safe_division(nir - swir2, nir + swir2)
    ndmi = _safe_division(nir - swir1, nir + swir1)
    ndwi = _safe_division(green - nir, green + nir)
    ndbi = _safe_division(swir1 - nir, swir1 + nir)
    bsi = _safe_division(
        (swir1 + red) - (nir + blue),
        (swir1 + red) + (nir + blue),
    )

    return {
        "ndvi": ndvi,
        "evi": evi,
        "savi": savi,
        "nbr": nbr,
        "ndmi": ndmi,
        "ndwi": ndwi,
        "ndbi": ndbi,
        "bsi": bsi,
    }
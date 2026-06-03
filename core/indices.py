"""
spectral index computations.
"""

import numpy as np
import rioxarray


def compute_indices(bands):
    """
    Compute NDVI, NBR, NDWI.
    Returns a dict of numpy arrays.
    """
    red = bands["red"]
    nir = bands["nir"]
    swir2 = bands["swir2"]
    scl = bands["scl"]
    green = bands["green"]

    def safe_div(a, b):
        return np.where(b == 0, 0, a / b)

    ndvi = safe_div(nir - red, nir + red)
    nbr  = safe_div(nir - swir2, nir + swir2)
    ndwi = safe_div(green - nir, green + nir)

    return {
        "NDVI": ndvi,
        "NBR": nbr,
        "NDWI": ndwi
    }

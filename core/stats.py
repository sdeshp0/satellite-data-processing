"""
Summary statistics for band and index rasters, plus a breakdown of the
Sentinel-2 Scene Classification Layer (SCL) over the AOI.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
import xarray as xr


# Sentinel-2 L2A SCL class codes.
SCL_CLASSES = {
    0: "No Data",
    1: "Saturated / Defective",
    2: "Dark Area Pixels",
    3: "Cloud Shadows",
    4: "Vegetation",
    5: "Bare Soils",
    6: "Water",
    7: "Clouds (Low Prob.)",
    8: "Clouds (Medium Prob.)",
    9: "Clouds (High Prob.)",
    10: "Cirrus",
    11: "Snow / Ice",
}


def array_statistics(arr: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for a single 2D array, ignoring NaNs (e.g.
    pixels excluded by cloud masking).
    """
    valid = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr.ravel()

    if valid.size == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan,
                "max": np.nan, "p10": np.nan, "p90": np.nan}

    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "p10": float(np.percentile(valid, 10)),
        "p90": float(np.percentile(valid, 90)),
    }


def summary_table(arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Build a summary statistics table (mean/std/min/max/p10/p90) across a set
    of named 2D arrays (bands or indices).

    Parameters
    ----------
    arrays : dict
        Mapping of name -> 2D numpy array.

    Returns
    -------
    pd.DataFrame
        One row per array, columns mean/std/min/max/p10/p90.
    """
    rows = {name: array_statistics(arr) for name, arr in arrays.items()}
    return pd.DataFrame(rows).T


def scl_breakdown(scl: xr.DataArray) -> pd.DataFrame:
    """
    Compute the percentage of AOI pixels in each SCL class.

    Parameters
    ----------
    scl : xr.DataArray
        Scene Classification Layer with unscaled integer class codes (0-11).
        Must not have been passed through the reflectance scale factor (see
        core.utils.scale_bands, which now excludes "scl" by default).

    Returns
    -------
    pd.DataFrame
        Columns: "Class", "Pixels", "Percent". Only classes actually present
        in the scene are included, sorted by percentage descending.
    """
    values = np.nan_to_num(scl.values, nan=0).astype(int).ravel()
    total = values.size
    counts = np.bincount(values, minlength=12)

    rows = []
    for code, count in enumerate(counts):
        if count == 0:
            continue
        rows.append({
            "Class": SCL_CLASSES.get(code, f"Unknown ({code})"),
            "Pixels": int(count),
            "Percent": round(100 * count / total, 1),
        })

    if not rows:
        return pd.DataFrame(columns=["Class", "Pixels", "Percent"])

    return (
        pd.DataFrame(rows)
        .sort_values("Percent", ascending=False)
        .reset_index(drop=True)
    )


def coverage_info(reference: xr.DataArray) -> Dict[str, float]:
    """
    Approximate resolution / pixel dimensions / area for the loaded window,
    based on a reference band's transform (e.g. the resampled NIR band).

    Note: this describes the loaded raster window, which may be decimated
    (see load_scene's max_dim) and is a bounding rectangle around the AOI,
    not the AOI polygon's exact area.
    """
    res_x, res_y = reference.rio.resolution()
    height, width = reference.shape[-2], reference.shape[-1]
    area_km2 = abs(res_x * res_y) * height * width / 1e6

    return {
        "resolution_m": abs(res_x),
        "width_px": width,
        "height_px": height,
        "area_km2": area_km2,
    }

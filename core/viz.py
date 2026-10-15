"""
Visualization helpers for RGB rendering and SCL colorization.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def to_rgb(
    bands: Dict[str, xr.DataArray],
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """
    Convert raw Sentinel‑2 reflectance bands to an 8‑bit RGB image.

    Uses a percentile stretch (2nd-98th by default) rather than a raw
    min-max stretch. Min-max is very sensitive to single outlier pixels
    (cloud edges, sensor artifacts) which can wash out contrast across the
    whole image; clipping to a percentile range before scaling is the
    standard fix and gives a much more usable preview.

    Parameters
    ----------
    bands : dict
        Mapping of band name -> DataArray. Must contain "red", "green", "blue".
    lower_percentile, upper_percentile : float
        Percentile bounds (0-100) used to clip before scaling to 0-255.

    Returns
    -------
    np.ndarray
        RGB image as uint8 array with shape (H, W, 3).
    """
    r = bands["red"].values
    g = bands["green"].values
    b = bands["blue"].values

    stack = np.stack([r, g, b], axis=-1)

    lo = np.nanpercentile(stack, lower_percentile)
    hi = np.nanpercentile(stack, upper_percentile)

    stretched = np.clip((stack - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    # Masked (NaN) pixels render black rather than relying on an undefined
    # NaN -> uint8 cast.
    stretched = np.nan_to_num(stretched, nan=0.0)

    rgb = (stretched * 255).astype("uint8")
    return rgb


def viz_scl(bands: Dict[str, xr.DataArray]) -> plt.Figure:
    """
    Visualize the Sentinel‑2 Scene Classification Layer (SCL).

    Parameters
    ----------
    bands : dict
        Mapping of band name -> DataArray. Must contain "scl".

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure containing the SCL plot.
    """
    scl = bands["scl"]

    fig, ax = plt.subplots(figsize=(8, 6))
    scl.plot(ax=ax, cmap="tab20", add_colorbar=True)
    ax.set_title("Scene Classification Layer (SCL)")
    ax.set_axis_off()

    return fig



def draw_ellipses(
    rgb: np.ndarray,
    ellipses,
    color: tuple = (255, 212, 0),
    width_px: int | None = None,
    alpha: float = 0.85,
) -> np.ndarray:
    """
    Draw thin ellipse outlines (no fill) on a copy of a uint8 RGB image.
    `ellipses` are core.features.ChangeCluster-like objects with row, col,
    semi_major, semi_minor and orientation (radians, regionprops convention).
    """
    h, w = rgb.shape[:2]
    if width_px is None:
        # ~3 px on a 1024 px preview, so the line survives downscaling on screen.
        width_px = max(2, max(h, w) // 350)
    outline = np.zeros((h, w), dtype=bool)
    for e in ellipses:
        n = max(64, int(8 * np.pi * max(e.semi_major, e.semi_minor)))
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cos_o, sin_o = np.cos(e.orientation), np.sin(e.orientation)
        for offset in np.arange(width_px) - (width_px - 1) / 2:
            a, b = e.semi_major + offset, e.semi_minor + offset
            rows = e.row + a * np.cos(t) * cos_o - b * np.sin(t) * sin_o
            cols = e.col + a * np.cos(t) * sin_o + b * np.sin(t) * cos_o
            rows, cols = np.rint(rows).astype(int), np.rint(cols).astype(int)
            inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            outline[rows[inside], cols[inside]] = True

    out = rgb.astype(np.float32).copy()
    out[outline] = (1 - alpha) * out[outline] + alpha * np.array(color, dtype=np.float32)
    return out.astype(np.uint8)

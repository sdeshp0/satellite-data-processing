"""
Visualization helpers for RGB rendering and SCL colorization.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def to_rgb(bands: Dict[str, xr.DataArray]) -> np.ndarray:
    """
    Convert raw Sentinel‑2 reflectance bands to an 8‑bit RGB image.

    This applies a simple min‑max stretch exactly as the original code did.

    Parameters
    ----------
    bands : dict
        Mapping of band name → DataArray. Must contain:
        - "red"
        - "green"
        - "blue"

    Returns
    -------
    np.ndarray
        RGB image as uint8 array with shape (H, W, 3).
    """
    r = bands["red"].values
    g = bands["green"].values
    b = bands["blue"].values

    stack = np.stack([r, g, b], axis=-1)

    # Normalize to 0–1 (behavior preserved)
    stack = stack - np.nanmin(stack)
    stack = stack / (np.nanmax(stack) + 1e-6)

    # Convert to uint8
    rgb = (stack * 255).astype("uint8")
    return rgb


def viz_scl(bands: Dict[str, xr.DataArray]) -> plt.Figure:
    """
    Visualize the Sentinel‑2 Scene Classification Layer (SCL).

    Parameters
    ----------
    bands : dict
        Mapping of band name → DataArray. Must contain "scl".

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

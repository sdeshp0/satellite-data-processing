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


def crosshatch(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 212, 0),
    spacing_px: int | None = None,
    line_px: int | None = None,
    alpha: float = 0.8,
) -> np.ndarray:
    """
    Draw thin diagonal crosshatching over the pixels in `mask` on a copy of
    a uint8 RGB image, leaving the image visible between the lines. Spacing
    and line width scale with the image so the pattern reads the same on
    screen at any preview size.
    """
    h, w = rgb.shape[:2]
    size = max(h, w)
    spacing_px = spacing_px or max(8, size // 40)
    line_px = line_px or max(1, size // 400)
    rows, cols = np.mgrid[0:h, 0:w]
    lines = ((rows + cols) % spacing_px < line_px) | ((rows - cols) % spacing_px < line_px)
    hatch = mask.astype(bool) & lines

    out = rgb.astype(np.float32).copy()
    out[hatch] = (1 - alpha) * out[hatch] + alpha * np.array(color, dtype=np.float32)
    return out.astype(np.uint8)


def label_points(rgb: np.ndarray, points) -> np.ndarray:
    """
    Write short labels (e.g. hotspot numbers) on a copy of a uint8 RGB
    image: white text with a dark outline, centred on each (row, col, text).
    """
    from PIL import Image, ImageDraw, ImageFont

    size = max(rgb.shape[:2])
    font_px = max(12, size // 30)
    try:
        font = ImageFont.load_default(size=font_px)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for row, col, text in points:
        draw.text(
            (col, row), str(text), fill=(255, 255, 255), font=font, anchor="mm",
            stroke_width=max(1, font_px // 8), stroke_fill=(0, 0, 0),
        )
    return np.asarray(img)

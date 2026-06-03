"""
Visualization helpers for RGB and index colorization.
"""

import numpy as np
import matplotlib.pyplot as plt

def to_rgb(bands):
    """
    Convert raw Sentinel-2 reflectance to 8-bit RGB.
    Simple min-max stretch.
    """
    r = bands["red"]
    g = bands["green"]
    b = bands["blue"]

    stack = np.stack([r, g, b], axis=-1)

    # Normalize to 0–1
    stack = stack - stack.min()
    stack = stack / (stack.max() + 1e-6)

    # Convert to uint8
    rgb = (stack * 255).astype("uint8")
    return rgb


def viz_scl(bands):
    """
    Visualize scene classification layer (SCL) used for cloud masking
    """
    scl = bands["scl"]

    plt.figure(figsize=(8, 6))
    fig = scl.plot(cmap="tab20", add_colorbar=True)
    plt.title("Scene Classification Layer (SCL)")
    return fig

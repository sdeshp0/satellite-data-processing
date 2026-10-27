"""
A plain-language summary of one scene: what share of the clear part of
the AOI is water, vegetation, or built-up / bare ground.

Each clear pixel gets one surface type, checked in order: water (NDWI),
then vegetation (NDVI, split into dense and sparse), then built-up or bare
ground (the rest, unless NDBI is very low). The thresholds are the Feature
Identification defaults (core.features), so the two pages agree. Built-up
land and bare soil are one class because their Sentinel-2 indices overlap
too much to separate reliably (see check_features.py).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from core.change import format_area
from core.features import BUILT_OR_BARE_MIN_NDBI, FEATURE_TYPES
from core.findings import Finding

# NDVI above this counts as dense vegetation (closed canopy, healthy crops).
DENSE_NDVI = 0.6

SURFACE_LABELS: Dict[str, str] = {
    "water": "Water",
    "dense_vegetation": "Dense vegetation",
    "sparse_vegetation": "Sparse vegetation",
    "built_or_bare": "Built-up or bare ground",
    "other": "Other",
}


def surface_types(indices: Dict[str, np.ndarray], clear: np.ndarray) -> Dict[str, np.ndarray]:
    """One mask per SURFACE_LABELS key; together they cover the clear pixels once each."""
    with np.errstate(invalid="ignore"):
        water = clear & (indices["ndwi"] > FEATURE_TYPES["water"]["default_threshold"])
        vegetated = clear & ~water & (indices["ndvi"] > FEATURE_TYPES["vegetation"]["default_threshold"])
        dense = vegetated & (indices["ndvi"] > DENSE_NDVI)
        built_or_bare = clear & ~water & ~vegetated & (indices["ndbi"] > BUILT_OR_BARE_MIN_NDBI)
    other = clear & ~water & ~vegetated & ~built_or_bare
    return {
        "water": water,
        "dense_vegetation": dense,
        "sparse_vegetation": vegetated & ~dense,
        "built_or_bare": built_or_bare,
        "other": other,
    }


def scene_findings(types: Dict[str, np.ndarray], clear: np.ndarray, in_aoi: np.ndarray, pixel_area_ha: float) -> List[Finding]:
    """Shares of the clear AOI per surface type, and how much of the AOI was hidden."""
    n_clear, n_aoi = int(clear.sum()), int(in_aoi.sum())
    if not n_clear:
        return [Finding("caveat", "No part of the AOI has a clear view in this scene.")]
    share = {k: 100 * int(m.sum()) / n_clear for k, m in types.items()}
    vegetation = share["dense_vegetation"] + share["sparse_vegetation"]
    parts = [
        f"{vegetation:.0f}% vegetation ({share['dense_vegetation']:.0f}% dense)",
        f"{share['water']:.0f}% water",
        f"{share['built_or_bare']:.0f}% built-up or bare ground",
    ]
    if share["other"] >= 1:
        parts.append(f"{share['other']:.0f}% other")
    out = [Finding(
        "headline",
        f"Of the clear part of the AOI ({format_area(n_clear * pixel_area_ha)}): " + ", ".join(parts) + ".",
    )]
    hidden = 100 * (n_aoi - n_clear) / n_aoi if n_aoi else 0
    if hidden >= 5:
        out.append(Finding(
            "caveat",
            f"{hidden:.0f}% of the AOI is hidden by cloud, shadow or missing data, so these shares describe the rest.",
        ))
    return out

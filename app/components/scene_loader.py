"""
Shared, cached scene loading for Streamlit pages. Centralized here so
Single Scene and Change Detection use one cache instead of each defining
its own wrapper around core.load.load_scene.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import streamlit as st
from shapely import wkt as shapely_wkt

from core.load import load_scene, DEFAULT_MAX_COMPANION_TILES, DEFAULT_MOSAIC_COVERAGE_THRESHOLD


@st.cache_data(show_spinner=False, ttl=3600)
def load_scene_cached(
    _item: Any,
    aoi_wkt: str,
    item_id: str,
    max_dim: Optional[int],
    enable_mosaic: bool = True,
    max_companion_tiles: int = DEFAULT_MAX_COMPANION_TILES,
    mosaic_coverage_threshold: float = DEFAULT_MOSAIC_COVERAGE_THRESHOLD,
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    """
    Cached wrapper around load_scene.

    `_item` is prefixed with an underscore so Streamlit skips hashing the
    pystac.Item object itself (it can contain nested structures that are
    slow or unstable to hash). The actual cache key is `item_id` + `aoi_wkt`
    + `max_dim` + the mosaic parameters, all cheap and stable to hash --
    including the mosaic parameters in the key matters, since toggling
    enable_mosaic (or the thresholds) should produce a genuinely different
    cached result, not silently reuse a result computed under the other
    setting.

    Returns
    -------
    tuple(Dict[str, Any], float, Dict[str, Any])
        (bands, coverage_fraction, load_info) -- see core.load.load_scene.
        load_info carries the multi-tile-mosaic cost/coverage diagnostics
        (tiles used, time spent); callers that don't care about it can
        simply ignore the third value.
    """
    aoi = shapely_wkt.loads(aoi_wkt)
    return load_scene(
        _item,
        aoi,
        max_dim=max_dim,
        enable_mosaic=enable_mosaic,
        max_companion_tiles=max_companion_tiles,
        mosaic_coverage_threshold=mosaic_coverage_threshold,
    )

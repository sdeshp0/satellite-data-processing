"""
Shared, cached scene loading for Streamlit pages. Centralized here so
Single Scene and Change Detection use one cache instead of each defining
its own wrapper around core.load.load_scene.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import streamlit as st
from shapely import wkt as shapely_wkt

from core.load import load_scene


@st.cache_data(show_spinner=False, ttl=3600)
def load_scene_cached(
    _item: Any,
    aoi_wkt: str,
    item_id: str,
    max_dim: Optional[int],
) -> Tuple[Dict[str, Any], float]:
    """
    Cached wrapper around load_scene.

    `_item` is prefixed with an underscore so Streamlit skips hashing the
    pystac.Item object itself (it can contain nested structures that are
    slow or unstable to hash). The actual cache key is `item_id` + `aoi_wkt`
    + `max_dim`, all cheap and stable to hash.

    Returns
    -------
    tuple(Dict[str, Any], float)
        (bands, coverage_fraction) -- see core.load.load_scene. Callers
        should check coverage_fraction and warn if it's well below 1.0.
    """
    aoi = shapely_wkt.loads(aoi_wkt)
    return load_scene(_item, aoi, max_dim=max_dim)

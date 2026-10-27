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


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_worldcover_cached(_reference: Any, grid_key: str, as_of: Optional[int] = None) -> Tuple[Any, int]:
    """
    ESA WorldCover resampled onto the reference band's grid, plus the
    product year (0 if no tiles were found). `grid_key` (CRS, transform and
    shape as a string) is the cache key, since the DataArray itself isn't
    hashed. `as_of` picks the edition (see core.landcover.pick_worldcover_year).
    """
    from core.landcover import search_worldcover, worldcover_on_grid

    items, year = search_worldcover(_reference, as_of)
    if not items:
        return None, 0
    return worldcover_on_grid(items, _reference), year


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_buildings_cached(_reference: Any, grid_key: str) -> Tuple[Any, Any, int]:
    """
    Grid (rows, cols) of every Microsoft building footprint's centroid on
    the reference band's grid, plus the number of footprint partitions
    found (0 if none cover the AOI). `grid_key` is the cache key, as for
    load_worldcover_cached.
    """
    from core.buildings import building_pixels, search_buildings

    items = search_buildings(_reference)
    if not items:
        return None, None, 0
    rows, cols = building_pixels(items, _reference)
    return rows, cols, len(items)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_radar_cached(_reference: Any, grid_key: str, before_date: str, after_date: str) -> Optional[Dict[str, Any]]:
    """
    Sentinel-1 VV backscatter (dB) on the reference band's grid for the
    radar acquisitions closest to the two optical dates (ISO strings, part
    of the cache key with `grid_key`), plus which scenes were used. None
    if no radar scene covers either date.
    """
    from datetime import date

    from core.radar import backscatter_db, search_radar_pair

    pair = search_radar_pair(_reference, date.fromisoformat(before_date), date.fromisoformat(after_date))
    if pair is None:
        return None
    return {
        "before_db": backscatter_db(pair.before_items, _reference),
        "after_db": backscatter_db(pair.after_items, _reference),
        "before_date": pair.before.datetime.date().isoformat(),
        "after_date": pair.after.datetime.date().isoformat(),
        "same_track": pair.same_track,
    }


def warn_if_overcorrected(bands: Dict[str, Any], label: str = "This scene") -> None:
    """
    A warning when a scene's blue or green reflectance is often negative
    over clear land (see core.utils.negative_reflectance_share): its
    atmospheric correction overshot, so colours and indices are unreliable.
    """
    from core.utils import NEGATIVE_REFLECTANCE_WARN_SHARE, negative_reflectance_share

    shares = negative_reflectance_share(bands)
    worst = max(shares, key=shares.get)
    if shares[worst] > NEGATIVE_REFLECTANCE_WARN_SHARE:
        st.warning(
            f"{label} looks badly corrected for haze: {shares[worst]:.0%} of its clear land has {worst} "
            "reflectance below zero, which real surfaces can't have. ESA's processing removed too much "
            "atmosphere, so its colours look off and indices can run past -1 or +1. Results from it are "
            "unreliable; try a scene from another date."
        )

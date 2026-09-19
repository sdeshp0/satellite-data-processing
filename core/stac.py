"""
STAC search utilities for Sentinel-2.
"""

from __future__ import annotations

from typing import List, Any, Dict
from datetime import date

import planetary_computer as pc
from pystac_client import Client
from shapely.geometry import mapping, Polygon


def search_sentinel2(
    aoi: Polygon | Dict[str, Any],
    start_date: date,
    end_date: date,
    max_cloud_cover: int = 40,
) -> List[Any]:
    """
    Search Sentinel-2 L2A items intersecting the AOI.

    Parameters
    ----------
    aoi : shapely.geometry.Polygon or GeoJSON-like dict
        Area of interest in EPSG:4326.
    start_date : datetime.date
        Beginning of date range.
    end_date : datetime.date
        End of date range.
    max_cloud_cover : int
        Maximum eo:cloud_cover percentage to include (0-100). Was
        previously hardcoded to 40; now a parameter so callers can relax it
        (e.g. sample_analyses' progressive-relaxation fallback) rather than
        failing outright when a strict filter returns nothing -- this
        matters especially for event-driven searches, since the weather
        that causes a flood is also the weather that produces clouds.

    Returns
    -------
    List[pystac.Item]
        List of STAC items matching the query.
    """

    # Normalize AOI to GeoJSON dict
    if hasattr(aoi, "geom_type"):  # Shapely geometry
        intersects = mapping(aoi)
    else:  # Already GeoJSON-like
        intersects = aoi

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=intersects,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )

    items = list(search.get_items())
    return items


def search_companion_tiles(
    aoi: Polygon | Dict[str, Any], reference_item: Any, max_items: int = 5
) -> List[Any]:
    """
    Find other Sentinel-2 items on the SAME acquisition date as
    reference_item that also intersect the AOI -- candidates for
    mosaicking together when a single tile's granule doesn't fully cover
    the AOI (see core.load.load_scene's enable_mosaic).

    Sentinel-2 tiles sit on a fixed ~110km grid; an AOI near a tile
    boundary can straddle two (rarely more) adjacent tiles. Searching by
    exact acquisition date (rather than, say, a date range) is what finds
    the specific companion granule(s) from the same satellite pass, not
    just any other cloud-free scene of the area from a different date.

    Cloud cover is NOT filtered here (queries the full 0-100% range):
    even a cloudier companion tile can supply real data in a spot the
    primary tile lacks entirely, and the existing SCL-based masking
    (core.change.coverage_overlap) still excludes cloudy pixels downstream
    regardless of which tile they came from.

    Parameters
    ----------
    aoi : shapely.geometry.Polygon or GeoJSON-like dict
    reference_item : pystac.Item
        The already-selected "primary" scene; its acquisition date is used
        for the search, and it's excluded from the results.
    max_items : int
        Cap on how many companion candidates to return. Search results are
        usually just the handful of tiles genuinely adjacent to the
        primary, but this bounds worst-case cost regardless of how many a
        given AOI happens to intersect.

    Returns
    -------
    List[pystac.Item]
        Other items from the same acquisition date, excluding
        reference_item itself. Empty if none found (or if the reference
        item's own date can't be determined).
    """
    acquisition_date = reference_item.datetime.date()
    items = search_sentinel2(aoi, acquisition_date, acquisition_date, max_cloud_cover=100)
    companions = [it for it in items if it.id != reference_item.id]
    return companions[:max_items]

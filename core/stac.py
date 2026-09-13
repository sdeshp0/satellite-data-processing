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

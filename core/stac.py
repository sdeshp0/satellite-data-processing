"""
STAC search utilities for Sentinel-2.
"""

import planetary_computer as pc
from pystac_client import Client


def search_sentinel2(aoi, start_date, end_date):
    """
    Search Sentinel-2 L2A items intersecting the AOI.
    Returns a list of STAC Items.
    """
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=aoi,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 40}}
    )

    items = list(search.get_items())
    return items

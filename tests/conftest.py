"""
Shared fixtures for the unit tests. Everything here is synthetic -- no test
touches the network or Planetary Computer.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import rioxarray  # noqa: F401  # .rio accessor
import xarray as xr
from rasterio.transform import from_origin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

UTM_10N = "EPSG:32610"


def make_da(values, res=10.0, origin=(500_000.0, 4_400_000.0), crs=UTM_10N, name=None):
    """A georeferenced 2D DataArray with pixel-centre x/y coordinates."""
    values = np.asarray(values)
    height, width = values.shape
    transform = from_origin(origin[0], origin[1], res, res)
    xs = origin[0] + res / 2 + np.arange(width) * res
    ys = origin[1] - res / 2 - np.arange(height) * res
    da = xr.DataArray(values, dims=["y", "x"], coords={"y": ys, "x": xs}, name=name)
    da = da.rio.write_crs(crs)
    return da.rio.write_transform(transform)


def make_item(item_id, when, cloud=10.0, tile="10SFH", nodata_pct=0.0, **extra):
    """Minimal stand-in for a pystac.Item with the properties core/ reads."""
    properties = {
        "eo:cloud_cover": cloud,
        "s2:mgrs_tile": tile,
        "s2:nodata_pixel_percentage": nodata_pct,
        **extra,
    }
    return SimpleNamespace(id=item_id, datetime=datetime.fromisoformat(when), properties=properties)


@pytest.fixture
def bands():
    """Four-pixel scene: vegetation, water, bare soil, and a No Data pixel."""
    def band(values):
        return make_da(np.array(values, dtype="float32").reshape(2, 2))

    return {
        "red":   band([0.05, 0.03, 0.25, 0.0]),
        "green": band([0.08, 0.10, 0.20, 0.0]),
        "blue":  band([0.04, 0.08, 0.15, 0.0]),
        "nir":   band([0.45, 0.02, 0.30, 0.0]),
        "swir1": band([0.20, 0.01, 0.40, 0.0]),
        "swir2": band([0.10, 0.01, 0.35, 0.0]),
        "scl":   make_da(np.array([[4, 6], [5, 0]], dtype="uint8")),
    }

"""
Radar flood mapping with Sentinel-1 (`sentinel-1-rtc` on Planetary
Computer: radiometrically terrain-corrected backscatter, 10 m).

Radar sees through cloud, which is exactly when floods happen and when
optical NDWI goes blind. Calm open water reflects the radar pulse away from
the sensor, so it shows as very low VV backscatter; a threshold on VV in dB
(-18 dB by default, a widely used starting point) separates water from land
on each date, and the two dates are compared with the same water-change
classes as the optical flood preset.

Known limits: flooded vegetation and flooded streets can look bright rather
than dark (double bounce), so they are missed; wind-roughened water can
look like land; and radar shadow on steep slopes can look like water.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import planetary_computer as pc
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from scipy.ndimage import median_filter
from shapely.geometry import box, mapping

from core.change import classify_water_change, format_area
from core.findings import Finding

RADAR_COLLECTION = "sentinel-1-rtc"
WATER_THRESHOLD_DB = -18.0
# How far from an optical scene's date a radar acquisition may be to stand
# in for it. Sentinel-1 revisits a given track every 12 days (6 when two
# satellites are flying), so this usually leaves at least one candidate.
MAX_DAYS_FROM_OPTICAL = 6
# Median filter size (pixels) to suppress speckle before thresholding.
SPECKLE_FILTER_PX = 3


@dataclass
class RadarPair:
    before: Any
    after: Any
    same_track: bool


def search_radar(reference: xr.DataArray, start: date, end: date) -> List[Any]:
    """Sentinel-1 RTC items covering the reference grid between two dates."""
    from pystac_client import Client

    bbox = transform_bounds(reference.rio.crs, "EPSG:4326", *reference.rio.bounds())
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    search = catalog.search(
        collections=[RADAR_COLLECTION],
        intersects=mapping(box(*bbox)),
        datetime=f"{start.isoformat()}/{end.isoformat()}",
    )
    return [item for item in search.items() if "vv" in item.assets]


def _track(item: Any) -> Tuple[Any, Any]:
    return item.properties.get("sat:relative_orbit"), item.properties.get("sat:orbit_state")


def _days(item: Any, target: date) -> int:
    return abs((item.datetime.date() - target).days)


def pick_radar_pair(
    before_items: Sequence[Any], after_items: Sequence[Any], before_date: date, after_date: date
) -> Optional[RadarPair]:
    """
    The radar acquisitions closest to each optical date, preferring a
    before scene on the same orbit track as the after scene, so both dates
    see the ground from the same angle. None if either side has no scene.
    """
    if not before_items or not after_items:
        return None
    after = min(after_items, key=lambda item: _days(item, after_date))
    same = [item for item in before_items if _track(item) == _track(after)]
    before = min(same or before_items, key=lambda item: _days(item, before_date))
    return RadarPair(before, after, bool(same))


def search_radar_pair(reference: xr.DataArray, before_date: date, after_date: date) -> Optional[RadarPair]:
    """Search around each optical date and pick the best radar pair."""
    window = timedelta(days=MAX_DAYS_FROM_OPTICAL)
    before_items = search_radar(reference, before_date - window, before_date + window)
    after_items = search_radar(reference, after_date - window, after_date + window)
    return pick_radar_pair(before_items, after_items, before_date, after_date)


def _signed(href: str) -> str:
    return pc.sign(href.split("?", 1)[0]) if href.startswith("http") else href


def backscatter_db(item: Any, reference: xr.DataArray, asset: str = "vv") -> np.ndarray:
    """
    The item's backscatter on the reference grid, in dB, speckle-filtered.
    Averaged down to the grid's resolution; NaN where the scene has no data.
    """
    dst_crs = reference.rio.crs
    dst_transform = reference.rio.transform()
    height, width = reference.shape[-2:]

    with rasterio.open(_signed(item.assets[asset].href)) as src:
        left, bottom, right, top = transform_bounds(dst_crs, src.crs, *reference.rio.bounds())
        window = from_bounds(left, bottom, right, top, src.transform).round_offsets().round_lengths()
        # Decimate on read when the grid is coarser than the source.
        scale = max(1.0, min(window.height / height, window.width / width))
        out_shape = (max(1, int(window.height / scale)), max(1, int(window.width / scale)))
        data = src.read(
            1, window=window, out_shape=out_shape, boundless=True,
            fill_value=np.nan, resampling=Resampling.average,
        ).astype("float32")
        read_transform = src.window_transform(window) * rasterio.Affine.scale(
            window.width / out_shape[1], window.height / out_shape[0]
        )
        nodata = src.nodata

    if nodata is not None and not np.isnan(nodata):
        data[data == nodata] = np.nan
    data[data <= 0] = np.nan  # linear power; zero means no data

    power = np.full((height, width), np.nan, dtype="float32")
    reproject(
        source=data, destination=power,
        src_transform=read_transform, src_crs=src.crs, src_nodata=np.nan,
        dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    db = 10 * np.log10(power)
    if SPECKLE_FILTER_PX > 1:
        valid = np.isfinite(db)
        filled = median_filter(np.where(valid, db, 0.0), size=SPECKLE_FILTER_PX)
        db = np.where(valid, filled, np.nan).astype("float32")
    return db


def classify_radar_water(
    db_before: np.ndarray, db_after: np.ndarray, threshold_db: float = WATER_THRESHOLD_DB
) -> np.ndarray:
    """
    Water-change classes (core.change.WATER_LABELS) from two VV dB grids:
    water is backscatter below `threshold_db`. -1 where either date has no data.
    """
    return classify_water_change(-db_before, -db_after, threshold=-threshold_db)


@dataclass
class RadarFloodSummary:
    new_water_ha: float
    blind_new_water_ha: float  # radar new water where optical had no usable data
    optical_new_water_ha: float
    buildings_in_new_water: Optional[int]


def radar_flood_summary(
    radar_codes: np.ndarray,
    optical_codes: np.ndarray,
    pixel_area_ha: float,
    building_rows: Optional[np.ndarray] = None,
    building_cols: Optional[np.ndarray] = None,
) -> RadarFloodSummary:
    """New water from radar, how much of it optical couldn't see, and the buildings in it."""
    new = radar_codes == 2
    buildings = None
    if building_rows is not None and building_cols is not None:
        buildings = int(new[building_rows, building_cols].sum())
    return RadarFloodSummary(
        new_water_ha=float(new.sum()) * pixel_area_ha,
        blind_new_water_ha=float((new & (optical_codes < 0)).sum()) * pixel_area_ha,
        optical_new_water_ha=float((optical_codes == 2).sum()) * pixel_area_ha,
        buildings_in_new_water=buildings,
    )


def radar_findings(summary: RadarFloodSummary) -> List[Finding]:
    """Key-findings lines for the radar flood map."""
    if not summary.new_water_ha:
        return [Finding("detail", "Radar (Sentinel-1, which sees through cloud) shows no new open water.")]
    text = f"Radar (Sentinel-1, which sees through cloud) maps {format_area(summary.new_water_ha)} of new open water"
    if summary.blind_new_water_ha:
        text += f", {format_area(summary.blind_new_water_ha)} of it where the optical images had no clear view"
    out = [Finding("detail", text + ".")]
    if summary.buildings_in_new_water:
        n = summary.buildings_in_new_water
        out.append(Finding("detail", f"{n:,} mapped building{'s sit' if n != 1 else ' sits'} in radar-mapped new water."))
    return out

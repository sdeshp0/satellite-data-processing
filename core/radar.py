"""
Radar flood mapping with Sentinel-1 (`sentinel-1-rtc` on Planetary
Computer: radiometrically terrain-corrected backscatter, 10 m).

Radar sees through cloud, which is exactly when floods happen and when
optical NDWI goes blind. Calm open water reflects the radar pulse away from
the sensor, so it shows as very low VV backscatter; a threshold on VV in dB
(-18 dB by default, a widely used starting point) separates water from land
on each date, and the two dates are compared with the same water-change
classes as the optical flood preset.

Water under trees, shrubs or tall marsh plants does the opposite: the
pulse bounces off the water surface and back off the trunks and stems
("double bounce"), so flooded vegetation turns brighter. Vegetated pixels
(from ESA WorldCover) that brighten sharply between the two dates are
flagged as likely flooded vegetation.

Known limits: flooded streets also brighten but aren't flagged, since
buildings make urban backscatter too variable; dense, leafy canopy can
hide the water from C-band radar entirely; heavy rain or new leaves can
brighten dry vegetation too; wind-roughened water can look like land; and
radar shadow on steep slopes can look like water.
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
from rasterio.errors import WindowError
from rasterio.windows import Window, from_bounds
from scipy.ndimage import convolve, median_filter
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

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
# Rise in VV backscatter (dB) between the dates that marks likely water
# under vegetation. Seasonal leaf and moisture changes are mostly smaller.
BRIGHTENING_DB = 3.0
# WorldCover classes with trunks or stems standing above the ground: tree
# cover, shrubland, herbaceous wetland and mangroves.
VEGETATION_CLASSES = (10, 20, 90, 95)
# Code for flooded vegetation added to the radar water-change classes.
FLOODED_VEGETATION = 4


@dataclass
class RadarPair:
    """
    The radar acquisitions for each date. A satellite pass is cut into
    consecutive scenes, so an AOI near a cut needs every scene from that
    pass (same date and track) to be covered.
    """

    before_items: List[Any]
    after_items: List[Any]
    same_track: bool

    @property
    def before(self) -> Any:
        return self.before_items[0]

    @property
    def after(self) -> Any:
        return self.after_items[0]


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


def pass_coverage(items: Sequence[Any], aoi_bbox: Optional[Tuple[float, float, float, float]]) -> float:
    """
    Share of a lon/lat AOI box inside the scenes' footprints (1.0 when the
    box or the footprints are unknown). Uses the STAC geometries, so no
    pixels are read.
    """
    geoms = [shape(item.geometry) for item in items if getattr(item, "geometry", None)]
    if aoi_bbox is None or not geoms:
        return 1.0
    aoi = box(*aoi_bbox)
    return unary_union(geoms).intersection(aoi).area / aoi.area


def _best_pass(
    candidates: Sequence[Any], items: Sequence[Any], target: date, aoi_bbox: Optional[Tuple[float, float, float, float]]
) -> Any:
    """
    The candidate whose whole pass covers the most of the AOI (to the
    nearest 10%), then the one closest to the target date.
    """
    def score(item: Any) -> Tuple[float, int]:
        return (-round(pass_coverage(_same_pass(item, items), aoi_bbox), 1), _days(item, target))

    return min(candidates, key=score)


def pick_radar_pair(
    before_items: Sequence[Any],
    after_items: Sequence[Any],
    before_date: date,
    after_date: date,
    aoi_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[RadarPair]:
    """
    The radar passes for each date: on the after side the pass covering the
    most of the AOI, then the closest to the date; on the before side the
    same, among passes on the after pass's orbit track when there are any,
    so both dates see the ground from the same angle. None if either side
    has no scene.
    """
    if not before_items or not after_items:
        return None
    after = _best_pass(after_items, after_items, after_date, aoi_bbox)
    same = [item for item in before_items if _track(item) == _track(after)]
    before = _best_pass(same or before_items, before_items, before_date, aoi_bbox)
    return RadarPair(_same_pass(before, before_items), _same_pass(after, after_items), bool(same))


def _same_pass(chosen: Any, items: Sequence[Any]) -> List[Any]:
    """Every scene from the chosen scene's pass: same date and orbit track, chosen first."""
    rest = [
        item for item in items
        if item is not chosen
        and _track(item) == _track(chosen)
        and item.datetime.date() == chosen.datetime.date()
    ]
    return [chosen] + rest


def search_radar_pair(reference: xr.DataArray, before_date: date, after_date: date) -> Optional[RadarPair]:
    """Search around each optical date and pick the best radar pair."""
    window = timedelta(days=MAX_DAYS_FROM_OPTICAL)
    before_items = search_radar(reference, before_date - window, before_date + window)
    after_items = search_radar(reference, after_date - window, after_date + window)
    bbox = transform_bounds(reference.rio.crs, "EPSG:4326", *reference.rio.bounds())
    return pick_radar_pair(before_items, after_items, before_date, after_date, bbox)


def _signed(href: str) -> str:
    return pc.sign(href.split("?", 1)[0]) if href.startswith("http") else href


def backscatter_db(items: Sequence[Any], reference: xr.DataArray, asset: str = "vv") -> np.ndarray:
    """
    The backscatter of one pass's scenes on the reference grid, in dB,
    speckle-filtered. Averaged down to the grid's resolution; where scenes
    overlap the first with data wins; NaN where none has data.
    """
    power = np.full(reference.shape[-2:], np.nan, dtype="float32")
    for item in items:
        tile = _power_on_grid(item, reference, asset)
        fill = np.isnan(power) & np.isfinite(tile)
        power[fill] = tile[fill]
    db = 10 * np.log10(power)
    if SPECKLE_FILTER_PX > 1:
        valid = np.isfinite(db)
        filled = median_filter(np.where(valid, db, 0.0), size=SPECKLE_FILTER_PX)
        db = np.where(valid, filled, np.nan).astype("float32")
    return db


def _power_on_grid(item: Any, reference: xr.DataArray, asset: str) -> np.ndarray:
    """One scene's backscatter (linear power) on the reference grid; NaN where it has no data."""
    dst_crs = reference.rio.crs
    dst_transform = reference.rio.transform()
    height, width = reference.shape[-2:]

    power = np.full((height, width), np.nan, dtype="float32")
    with rasterio.open(_signed(item.assets[asset].href)) as src:
        left, bottom, right, top = transform_bounds(dst_crs, src.crs, *reference.rio.bounds())
        window = from_bounds(left, bottom, right, top, src.transform).round_offsets().round_lengths()
        # Read only the part inside the scene: GDAL's boundless (VRT) reads
        # average no-data into edge pixels, darkening them like water.
        try:
            window = window.intersection(Window(0, 0, src.width, src.height))
        except WindowError:  # the AOI misses this scene
            return power
        # Decimate on read when the grid is coarser than the source.
        scale = max(1.0, min(window.height / height, window.width / width))
        out_shape = (max(1, int(window.height / scale)), max(1, int(window.width / scale)))
        data = src.read(1, window=window, out_shape=out_shape, resampling=Resampling.average).astype("float32")
        read_transform = src.window_transform(window) * rasterio.Affine.scale(
            window.width / out_shape[1], window.height / out_shape[0]
        )
        nodata = src.nodata

    if nodata is not None and not np.isnan(nodata):
        data[data == nodata] = np.nan
    data[data <= 0] = np.nan  # linear power; zero means no data

    reproject(
        source=data, destination=power,
        src_transform=read_transform, src_crs=src.crs, src_nodata=np.nan,
        dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return power


def classify_radar_water(
    db_before: np.ndarray, db_after: np.ndarray, threshold_db: float = WATER_THRESHOLD_DB
) -> np.ndarray:
    """
    Water-change classes (core.change.WATER_LABELS) from two VV dB grids:
    water is backscatter below `threshold_db`. -1 where either date has no data.
    """
    return classify_water_change(-db_before, -db_after, threshold=-threshold_db)


def flooded_vegetation(
    db_before: np.ndarray,
    db_after: np.ndarray,
    landcover: np.ndarray,
    brightening_db: float = BRIGHTENING_DB,
    threshold_db: float = WATER_THRESHOLD_DB,
) -> np.ndarray:
    """
    Likely flooded vegetation: vegetated pixels whose backscatter rose by at
    least `brightening_db` and that aren't open water after. Pixels with
    fewer than two flagged neighbours are dropped: those are mostly
    leftover speckle.
    """
    mask = (
        (db_after - db_before >= brightening_db)
        & (db_after >= threshold_db)
        & np.isin(landcover, VEGETATION_CLASSES)
    )
    flagged_nearby = convolve(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant")
    return mask & (flagged_nearby >= 3)


def with_flooded_vegetation(radar_codes: np.ndarray, vegetation: np.ndarray) -> np.ndarray:
    """The radar water-change classes with FLOODED_VEGETATION where dry on both dates but flagged."""
    return np.where(vegetation & (radar_codes == 0), FLOODED_VEGETATION, radar_codes)


@dataclass
class RadarFloodSummary:
    new_water_ha: float
    blind_new_water_ha: float  # radar new water where optical had no usable data
    optical_new_water_ha: float
    buildings_in_new_water: Optional[int]
    flooded_vegetation_ha: Optional[float] = None  # None when it couldn't be checked


def radar_flood_summary(
    radar_codes: np.ndarray,
    optical_codes: np.ndarray,
    pixel_area_ha: float,
    building_rows: Optional[np.ndarray] = None,
    building_cols: Optional[np.ndarray] = None,
    has_vegetation_check: bool = False,
) -> RadarFloodSummary:
    """
    New water from radar, how much of it optical couldn't see, the buildings
    in it, and the flooded vegetation when `radar_codes` include it (see
    with_flooded_vegetation; `has_vegetation_check` says whether they do).
    """
    new = radar_codes == 2
    buildings = None
    if building_rows is not None and building_cols is not None:
        buildings = int(new[building_rows, building_cols].sum())
    vegetation = float((radar_codes == FLOODED_VEGETATION).sum()) * pixel_area_ha
    return RadarFloodSummary(
        new_water_ha=float(new.sum()) * pixel_area_ha,
        blind_new_water_ha=float((new & (optical_codes < 0)).sum()) * pixel_area_ha,
        optical_new_water_ha=float((optical_codes == 2).sum()) * pixel_area_ha,
        buildings_in_new_water=buildings,
        flooded_vegetation_ha=vegetation if has_vegetation_check else None,
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
    if summary.flooded_vegetation_ha:
        out.append(Finding(
            "detail",
            f"A further {format_area(summary.flooded_vegetation_ha)} of trees, shrubs or wetland brightened "
            "sharply on radar, a likely sign of water under the canopy (a separate brightening check, "
            "not counted in the new open water).",
        ))
    return out

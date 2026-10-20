"""
An interactive, zoomable web map (folium/Leaflet) of a scene's grid:
basemaps, the scene's RGB, crosshatched feature masks, building centres
and land cover as switchable layers, plus invisible region shapes that
show each region's details on hover.

Grids are reprojected from the scene's UTM grid to Web Mercator (the
basemap's projection) before they are drawn, so they line up with it;
hatching is drawn after reprojection so its lines stay crisp.
"""

from __future__ import annotations

import base64
import io
from typing import Dict, List, Optional, Sequence, Tuple

import folium
import numpy as np
import shapely
import xarray as xr
from PIL import Image
from rasterio import features as rio_features
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds, transform_geom

from core.inventory import Region
from core.viz import crosshatch

ESRI_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTRIBUTION = "Tiles &copy; Esri, Maxar, Earthstar Geographics"
BUILDING_COLOR = (255, 0, 255)


class MercatorGrid:
    """The Web Mercator grid a reference band reprojects onto, and how to move arrays to it."""

    def __init__(self, reference: xr.DataArray):
        self.src_crs = reference.rio.crs
        self.src_transform = reference.rio.transform()
        height, width = reference.shape[-2:]
        bounds = reference.rio.bounds()
        self.transform, self.width, self.height = calculate_default_transform(
            self.src_crs, "EPSG:3857", width, height, *bounds
        )
        t = self.transform
        left, top = t.c, t.f
        right, bottom = t.c + self.width * t.a, t.f + self.height * t.e
        self.bounds = transform_bounds("EPSG:3857", "EPSG:4326", left, bottom, right, top)

    def warp(self, array: np.ndarray, nodata: float = 0, resampling=Resampling.nearest) -> np.ndarray:
        """A 2D array (or H x W x bands) on the Web Mercator grid."""
        if array.ndim == 3:
            return np.dstack([self.warp(array[..., i], nodata, resampling) for i in range(array.shape[-1])])
        out = np.full((self.height, self.width), nodata, dtype=array.dtype)
        reproject(
            array, out, src_transform=self.src_transform, src_crs=self.src_crs,
            dst_transform=self.transform, dst_crs="EPSG:3857",
            src_nodata=nodata, dst_nodata=nodata, resampling=resampling,
        )
        return out

    @property
    def leaflet_bounds(self) -> List[List[float]]:
        west, south, east, north = self.bounds
        return [[south, west], [north, east]]


def _png_url(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def hatch_rgba(mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """Transparent RGBA image with crosshatching in `color` over `mask`, as Change Detection draws it."""
    blank = np.zeros(mask.shape + (3,), dtype=np.uint8)
    lines = crosshatch(blank, mask, color=(255, 255, 255), alpha=1.0) > 0
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    hatch = lines.any(axis=-1)
    rgba[hatch, :3] = color
    rgba[hatch, 3] = 255
    return rgba


def region_geojson(
    labelled: np.ndarray, regions: Sequence[Region], reference: xr.DataArray, properties: Dict[int, Dict],
) -> Dict:
    """WGS84 polygons of the listed regions, simplified to about a pixel, with `properties` per label."""
    listed = np.isin(labelled, [r.label for r in regions])
    res = abs(reference.rio.resolution()[0])
    out = []
    for geom, label in rio_features.shapes(
        labelled.astype(np.int32), mask=listed, transform=reference.rio.transform()
    ):
        shape = shapely.geometry.shape(geom).simplify(res, preserve_topology=True)
        out.append({
            "type": "Feature",
            "properties": properties[int(label)],
            "geometry": transform_geom(reference.rio.crs, "EPSG:4326", shapely.geometry.mapping(shape)),
        })
    return {"type": "FeatureCollection", "features": out}


def feature_map(
    reference: xr.DataArray,
    rgb: np.ndarray,
    hatches: Sequence[Tuple[str, np.ndarray, Tuple[int, int, int]]],
    regions_geojson: Optional[Dict] = None,
    tooltip_fields: Sequence[Tuple[str, str]] = (),
    buildings: Optional[np.ndarray] = None,
    landcover_rgb: Optional[np.ndarray] = None,
    focus: Optional[Region] = None,
    focus_label: str = "",
) -> folium.Map:
    """
    The scene's grid as a Leaflet map. `hatches` are (layer name, mask,
    colour); `buildings` is a boolean grid of pixels holding a building
    centre; `landcover_rgb` an RGB image of the land-cover classes.
    `focus` zooms to one region and pins its centre.
    """
    grid = MercatorGrid(reference)
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer(ESRI_IMAGERY, attr=ESRI_ATTRIBUTION, name="Satellite basemap (Esri)").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street map (OpenStreetMap)", show=False).add_to(m)

    def overlay(rgba: np.ndarray, name: str, show: bool = True, opacity: float = 1.0) -> None:
        folium.raster_layers.ImageOverlay(
            _png_url(rgba), bounds=grid.leaflet_bounds, name=name, show=show, opacity=opacity,
        ).add_to(m)

    has_data = grid.warp(np.ones(reference.shape[-2:], dtype=np.uint8)) > 0
    scene = np.dstack([grid.warp(rgb), (has_data * 255).astype(np.uint8)])
    overlay(scene, "Scene (true colour)")
    if landcover_rgb is not None:
        lc = grid.warp(landcover_rgb)
        overlay(np.dstack([lc, (has_data * 255).astype(np.uint8)]), "Land cover (ESA WorldCover)", show=False, opacity=0.7)
    for name, mask, color in hatches:
        overlay(hatch_rgba(grid.warp(mask.astype(np.uint8)) > 0, color), name)
    if buildings is not None:
        dots = grid.warp(buildings.astype(np.uint8)) > 0
        rgba = np.zeros(dots.shape + (4,), dtype=np.uint8)
        rgba[dots, :3] = BUILDING_COLOR
        rgba[dots, 3] = 230
        overlay(rgba, "Buildings (one dot per footprint)", show=False)

    if regions_geojson and regions_geojson["features"]:
        fields, aliases = zip(*tooltip_fields) if tooltip_fields else ((), ())
        folium.GeoJson(
            regions_geojson, name="Region details (hover)",
            style_function=lambda _: {"weight": 0, "fillOpacity": 0},
            highlight_function=lambda _: {"weight": 0, "fillColor": "#ffffff", "fillOpacity": 0.25},
            tooltip=folium.GeoJsonTooltip(fields=list(fields), aliases=list(aliases)) if fields else None,
        ).add_to(m)

    if focus is not None:
        west, south, east, north = focus.bounds
        pad_x, pad_y = max((east - west) * 0.3, 0.002), max((north - south) * 0.3, 0.002)
        m.fit_bounds([[south - pad_y, west - pad_x], [north + pad_y, east + pad_x]])
        folium.Marker(
            [focus.lat, focus.lon], tooltip=focus_label or None,
        ).add_to(m)
    else:
        m.fit_bounds(grid.leaflet_bounds)
    folium.LayerControl(collapsed=False).add_to(m)
    return m

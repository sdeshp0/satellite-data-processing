import numpy as np

from core.inventory import feature_regions
from core.webmap import MercatorGrid, feature_map, hatch_rgba, region_geojson
from tests.conftest import make_da


def test_mercator_grid_covers_the_reference_in_lonlat():
    ref = make_da(np.zeros((50, 50), dtype="float32"))
    grid = MercatorGrid(ref)
    west, south, east, north = grid.bounds
    assert -124 < west < east < -122 and 39 < south < north < 40.5
    warped = grid.warp(np.ones((50, 50), dtype=np.uint8))
    assert warped.shape == (grid.height, grid.width) and warped.sum() > 0.9 * 2500


def test_hatch_rgba_only_marks_hatch_lines_inside_the_mask():
    mask = np.zeros((80, 80), dtype=bool)
    mask[:, :40] = True
    rgba = hatch_rgba(mask, (49, 130, 189))
    drawn = rgba[..., 3] > 0
    assert drawn[:, :40].any() and not drawn[:, 40:].any()
    assert 0 < drawn[:, :40].mean() < 0.5
    assert (rgba[drawn, :3] == (49, 130, 189)).all()


def test_feature_map_builds_layers_and_region_shapes():
    ref = make_da(np.zeros((60, 60), dtype="float32"))
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:30, 10:30] = True
    regions, labelled, _ = feature_regions(mask, ref, 0.01)
    geo = region_geojson(labelled, regions, ref, {regions[0].label: {"rank": 1}})
    assert len(geo["features"]) == 1 and geo["features"][0]["properties"] == {"rank": 1}
    lon, lat = geo["features"][0]["geometry"]["coordinates"][0][0]
    assert -124 < lon < -122 and 39 < lat < 40.5

    rgb = np.zeros((60, 60, 3), dtype=np.uint8)
    html = feature_map(
        ref, rgb, [("Water", mask, (0, 0, 255))], geo, [("rank", "Rank")],
        buildings=mask, landcover_rgb=rgb, focus=regions[0], focus_label="#1",
    ).get_root().render()
    for name in ("Scene (true colour)", "Water", "Land cover (ESA WorldCover)", "Buildings", "Region details"):
        assert name in html

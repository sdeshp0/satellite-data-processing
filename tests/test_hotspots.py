import json

import numpy as np
import pytest
from skimage import draw

from core.hotspots import find_hotspots, hotspot_table, hotspots_geojson
from core.viz import crosshatch, label_points
from tests.conftest import make_da


def _scene(noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros((300, 300), dtype=bool)
    rr, cc = draw.ellipse(100, 150, 50, 90, shape=mask.shape, rotation=0.5)
    mask[rr, cc] = rng.random(rr.size) < 0.6          # patchy main area
    rr, cc = draw.disk((240, 50), 15, shape=mask.shape)
    mask[rr, cc] = True                                # smaller second area
    mask |= rng.random(mask.shape) < noise             # noise across the AOI
    return mask


@pytest.mark.parametrize("noise", [0.0, 0.15, 0.3])
def test_find_hotspots_ranks_areas_despite_noise(noise):
    hotspots = find_hotspots(_scene(noise))
    assert hotspots.max() == 2
    assert hotspots[100, 150] == 1 and hotspots[240, 50] == 2
    assert hotspots[10, 290] == 0


def test_find_hotspots_ignores_noise_only_and_empty():
    rng = np.random.default_rng(1)
    assert not find_hotspots(rng.random((300, 300)) < 0.2).any()
    assert not find_hotspots(np.zeros((50, 50), dtype=bool)).any()


def test_hotspot_table_and_geojson():
    changed = np.zeros((100, 100), dtype=bool)
    changed[20:40, 60:80] = True
    hotspots = find_hotspots(changed)
    ref = make_da(np.zeros((100, 100)), res=10.0)
    table = hotspot_table(hotspots, changed, ref, pixel_area_ha=0.01)
    assert table["Changed area (ha)"].tolist() == [4.0]
    assert table["Share of detected change (%)"].tolist() == [100.0]
    assert -125 < table["Longitude"][0] < -121 and 38 < table["Latitude"][0] < 41

    gj = json.loads(json.dumps(hotspots_geojson(hotspots, ref)))
    ring = gj["features"][0]["geometry"]["coordinates"][0]
    assert gj["features"][0]["properties"]["hotspot"] == 1
    assert ring[0] == ring[-1]  # closed


def test_crosshatch_leaves_gaps_and_outside_untouched():
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:70, 10:70] = True
    out = crosshatch(rgb, mask, color=(200, 100, 0), spacing_px=8, line_px=1, alpha=1.0)
    hatched = (out[mask] != 0).any(axis=1)
    assert 0.15 < hatched.mean() < 0.35        # lines, not a fill
    assert out[~mask].sum() == 0
    assert rgb.sum() == 0


def test_label_points_draws_text():
    out = label_points(np.zeros((100, 100, 3), dtype=np.uint8), [(50, 50, 1)])
    assert out[40:60, 40:60].max() > 200

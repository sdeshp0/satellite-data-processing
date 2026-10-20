import numpy as np

from core.scene_summary import scene_findings, surface_types


def _indices():
    # Pixels: water, dense veg, sparse veg, bare (BSI), built-up (NDBI), other, cloudy.
    return {
        "ndwi": np.array([0.3, -0.5, -0.4, -0.2, -0.1, -0.1, 0.5]),
        "ndvi": np.array([-0.2, 0.8, 0.4, 0.1, 0.1, 0.1, -0.1]),
        "ndbi": np.array([-0.3, -0.4, -0.2, -0.05, 0.1, -0.1, 0.0]),
        "bsi": np.array([-0.3, -0.3, -0.1, 0.1, -0.1, -0.1, 0.0]),
    }


def test_surface_types_label_each_clear_pixel_once():
    clear = np.array([True] * 6 + [False])
    types = surface_types(_indices(), clear)
    assert {k: np.flatnonzero(m).tolist() for k, m in types.items()} == {
        "water": [0], "dense_vegetation": [1], "sparse_vegetation": [2], "built_or_bare": [3, 4], "other": [5],
    }


def test_scene_findings_wording():
    clear = np.array([True] * 6 + [False])
    texts = [f.text for f in scene_findings(surface_types(_indices(), clear), clear, np.ones(7, bool), 10.0)]
    assert texts == [
        "Of the clear part of the AOI (60 ha): 33% vegetation (17% dense), 17% water, "
        "33% built-up or bare ground, 17% other.",
        "14% of the AOI is hidden by cloud, shadow or missing data, so these shares describe the rest.",
    ]
    none_clear = np.zeros(7, bool)
    assert "No part of the AOI" in scene_findings(surface_types(_indices(), none_clear), none_clear, np.ones(7, bool), 1)[0].text

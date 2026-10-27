import numpy as np
import pytest

from core.features import contours_to_geojson, extract_contours, region_stats, threshold_mask
from tests.conftest import make_da


def test_threshold_mask_drops_small_regions_and_nan():
    arr = np.zeros((20, 20))
    arr[2:8, 2:8] = 1.0      # 36 px region, kept
    arr[15, 15] = 1.0        # 1 px speck, dropped
    arr[3, 3] = np.nan       # NaN inside the region: a 1px hole, filled
    mask = threshold_mask(arr, 0.5, min_region_px=20)
    assert mask[2:8, 2:8].all()
    assert not mask[15, 15]
    assert mask.sum() == 36


def test_threshold_mask_keeps_region_exactly_min_size():
    arr = np.zeros((10, 10))
    arr[0:4, 0:5] = 1.0      # exactly 20 px
    assert threshold_mask(arr, 0.5, min_region_px=20).sum() == 20


def test_threshold_mask_below_direction():
    arr = np.full((10, 10), 1.0)
    arr[0:5, 0:5] = -1.0
    assert threshold_mask(arr, 0.0, direction="below", min_region_px=1).sum() == 25


def test_contours_are_pixel_centre_aligned():
    # Single feature pixel at row 2, col 3 on a 10m grid. Its outline must
    # surround that pixel's centre, not its top-left corner.
    mask = np.zeros((6, 6), dtype=bool)
    mask[2, 3] = True
    ref = make_da(np.zeros((6, 6)), res=10.0, origin=(500_000.0, 4_400_000.0))
    geojson = contours_to_geojson({"water": extract_contours(mask)}, ref)
    coords = np.array(geojson["features"][0]["geometry"]["coordinates"])

    from pyproj import Transformer
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
    xs, ys = to_utm.transform(coords[:, 0], coords[:, 1])
    centre_x = 500_000.0 + 3 * 10 + 5
    centre_y = 4_400_000.0 - 2 * 10 - 5
    assert (xs.min() + xs.max()) / 2 == pytest.approx(centre_x, abs=0.01)
    assert (ys.min() + ys.max()) / 2 == pytest.approx(centre_y, abs=0.01)


def test_region_stats_area():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:2, 0:5] = True
    mask[8:10, 8:10] = True
    stats = region_stats(mask, make_da(np.zeros((10, 10)), res=10.0))
    assert stats["num_regions"] == 2
    assert stats["area_km2"] == pytest.approx(14 * 100 / 1e6)


def test_built_or_bare_excludes_water_and_vegetation_then_splits():
    from core.features import built_or_bare_raw, built_reference, split_built_or_bare

    # Pixels: water, forest, city, desert, shadow, cloud (NaN).
    indices = {
        "ndwi": np.array([[0.5, -0.6, -0.2, -0.2, -0.1, np.nan]]),
        "ndvi": np.array([[-0.4, 0.8, 0.1, 0.06, 0.1, np.nan]]),
        "ndbi": np.array([[0.3, -0.2, 0.05, 0.1, -0.4, np.nan]]),
    }
    raw = built_or_bare_raw(indices, ndvi_below=0.3, water_threshold=0.0)
    assert raw.tolist() == [[False, False, True, True, False, False]]

    landcover = np.array([[80, 10, 50, 60, 50, 50]])
    ref = built_reference(landcover)
    assert split_built_or_bare(raw, ref, "urban").tolist() == [[False, False, True, False, False, False]]
    assert split_built_or_bare(raw, ref, "bare_soil").tolist() == [[False, False, False, True, False, False]]
    # Without a reference both get the whole built-or-bare area.
    assert split_built_or_bare(raw, None, "urban").tolist() == raw.tolist()


def test_built_reference_adds_buildings_near_footprints():
    from core.features import built_reference

    landcover = np.full((5, 5), 60)
    ref = built_reference(landcover, np.array([2]), np.array([2]))
    assert ref.sum() == 9 and ref[1:4, 1:4].all()
    assert built_reference(None) is None

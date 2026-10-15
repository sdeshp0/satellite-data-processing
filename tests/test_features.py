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


def test_clean_mask_drops_specks_and_fills_holes():
    from core.features import clean_mask
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:12, 2:12] = True
    mask[6, 6] = False      # small hole
    mask[17, 17] = True     # isolated speck
    cleaned = clean_mask(mask, min_region_px=5)
    assert cleaned[6, 6] and not cleaned[17, 17]


def test_change_clusters_groups_patches_and_drops_specks():
    from skimage import draw
    from core.features import change_clusters
    mask = np.zeros((200, 200), dtype=bool)
    # A patchy diagonal scar: many small blobs close together.
    for k in range(10):
        rr, cc = draw.disk((40 + 8 * k, 40 + 8 * k), 3, shape=mask.shape)
        mask[rr, cc] = True
    rr, cc = draw.disk((160, 40), 6, shape=mask.shape)  # second, separate area
    mask[rr, cc] = True
    mask[5, 190] = True  # speck
    clusters = change_clusters(mask)
    assert len(clusters) == 2
    scar, other = clusters
    assert abs(scar.row - 76) < 2 and abs(scar.col - 76) < 2
    assert scar.semi_major > 2 * scar.semi_minor   # elongated along the diagonal
    assert abs(other.row - 160) < 1 and abs(other.col - 40) < 1


def test_change_clusters_empty():
    from core.features import change_clusters
    assert change_clusters(np.zeros((10, 10), dtype=bool)) == []


def test_change_clusters_skips_aoi_wide_noise():
    from core.features import change_clusters
    mask = np.zeros((300, 300), dtype=bool)
    mask[::6, ::6] = True                        # scattered single pixels: removed as specks
    mask[10:290:20, 10:290:20] = True
    for r in range(10, 290, 25):                 # small blobs spread across the whole AOI
        for c in range(10, 290, 25):
            mask[r:r + 4, c:c + 4] = True
    assert change_clusters(mask) == []

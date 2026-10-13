import numpy as np
import pytest

from core.indices import compute_indices


def test_all_eight_indices_returned(bands):
    assert set(compute_indices(bands)) == {"ndvi", "evi", "savi", "nbr", "ndmi", "ndwi", "ndbi", "bsi"}


def test_ndvi_and_ndwi_values(bands):
    idx = compute_indices(bands)
    assert idx["ndvi"][0, 0] == pytest.approx((0.45 - 0.05) / (0.45 + 0.05), abs=1e-4)
    assert idx["ndwi"][0, 1] == pytest.approx((0.10 - 0.02) / (0.10 + 0.02), abs=1e-4)


def test_land_cover_signs(bands):
    idx = compute_indices(bands)
    assert idx["ndvi"][0, 0] > 0.3     # vegetation
    assert idx["ndwi"][0, 1] > 0       # water
    assert idx["bsi"][1, 0] > 0        # bare soil
    assert idx["ndwi"][0, 0] < 0       # vegetation is not water


def test_zero_denominator_does_not_produce_inf(bands):
    idx = compute_indices(bands)
    for arr in idx.values():
        assert np.isfinite(arr[1, 1])

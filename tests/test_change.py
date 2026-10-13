from datetime import date

import numpy as np
import pytest

from core import change
from tests.conftest import make_da, make_item


def test_day_of_year_distance_wraps_year_end():
    assert change.day_of_year_distance(date(2020, 12, 30), date(2021, 1, 2)) <= 5
    assert change.day_of_year_distance(date(2020, 6, 1), date(2023, 6, 1)) <= 1


def test_compute_delta_direction():
    before, after = np.array([0.5]), np.array([0.2])
    assert change.compute_delta(before, after)[0] == pytest.approx(-0.3)
    assert change.compute_delta(before, after, "pre_minus_post")[0] == pytest.approx(0.3)


@pytest.mark.parametrize("dnbr, code", [(-0.5, 0), (-0.2, 1), (0.0, 2), (0.2, 3), (0.3, 4), (0.5, 5), (0.9, 6)])
def test_classify_dnbr_bins(dnbr, code):
    assert change.classify_dnbr(np.array([0.6]), np.array([0.6 - dnbr]))[0] == code


def test_classify_water_change():
    before = np.array([-0.2, 0.3, -0.2, 0.3, np.nan])
    after = np.array([-0.2, 0.3, 0.3, -0.2, 0.3])
    assert change.classify_water_change(before, after).tolist() == [0, 1, 2, 3, -1]


def test_classify_vegetation_loss():
    before = np.array([0.1, 0.6, 0.6, 0.1])
    after = np.array([0.1, 0.6, 0.1, 0.6])
    assert change.classify_vegetation_loss(before, after).tolist() == [0, 1, 2, 3]


def test_classify_statistical_flags_outliers():
    after = np.zeros(100)
    after[0], after[1] = 5.0, -5.0
    codes = change.classify_statistical(np.zeros(100), after)
    assert codes[0] == 1 and codes[1] == 2
    assert (codes[2:] == 0).all()


def test_coverage_overlap_four_way():
    before = make_da(np.array([[4, 4], [0, 9]], dtype="uint8"))
    after = make_da(np.array([[4, 0], [4, 8]], dtype="uint8"))
    codes, pct = change.coverage_overlap(before, after)
    assert codes.tolist() == [
        [change.COVERAGE_BOTH_CODE, change.COVERAGE_BEFORE_ONLY_CODE],
        [change.COVERAGE_AFTER_ONLY_CODE, change.COVERAGE_NEITHER_CODE],
    ]
    assert pct == {"both": 25.0, "before_only": 25.0, "after_only": 25.0, "neither": 25.0}


def test_select_best_pair_prefers_clear_pair_over_season_match():
    cloudy_same_season = make_item("b1", "2020-07-01", cloud=60)
    clear_off_season = make_item("b2", "2020-04-01", cloud=5)
    after = make_item("a1", "2021-07-01", cloud=5)
    assert change.select_best_pair([cloudy_same_season, clear_off_season], [after]) == (clear_off_season, after)


def test_select_best_pair_prefers_season_among_clear_pairs():
    near = make_item("b1", "2020-07-05", cloud=10)
    far = make_item("b2", "2020-01-05", cloud=2)
    after = make_item("a1", "2021-07-01", cloud=10)
    assert change.select_best_pair([far, near], [after]) == (near, after)


def test_select_best_pair_respects_coverage_but_never_returns_nothing():
    low_cov = make_item("b1", "2020-07-01", nodata_pct=80)
    full_cov = make_item("b2", "2020-07-10", nodata_pct=0, cloud=25)
    after = make_item("a1", "2021-07-01")
    assert change.select_best_pair([low_cov, full_cov], [after])[0] is full_cov
    assert change.select_best_pair([low_cov], [after])[0] is low_cov
    assert change.select_best_pair([], [after]) is None

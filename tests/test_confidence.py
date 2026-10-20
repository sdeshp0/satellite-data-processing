import numpy as np
import pytest

from core import change
from core.confidence import AreaRange, assess_confidence, change_area_range


def _summary(codes, preset_key, px=1.0):
    return change.change_area_summary(codes, change.EVENT_PRESETS[preset_key]["baseline_codes"], px)


def test_flood_range_brackets_central_estimate():
    before = np.full(100, -0.3)
    after = np.linspace(-0.2, 0.2, 100)   # some pixels sit near the NDWI 0 threshold
    preset = change.EVENT_PRESETS["flood"]
    central = _summary(preset["classify"](before, after), "flood")
    rng = change_area_range(preset, before, after, central, 1.0)
    assert rng.low_ha < rng.central_ha < rng.high_ha


def test_clear_cut_change_has_no_spread():
    before = np.full(100, 0.8)
    after = np.where(np.arange(100) < 30, 0.0, 0.8)   # NBR drop of 0.8 or none
    preset = change.EVENT_PRESETS["wildfire"]
    central = _summary(preset["classify"](before, after), "wildfire")
    rng = change_area_range(preset, before, after, central, 1.0)
    assert rng.low_ha == rng.central_ha == rng.high_ha == 30


@pytest.mark.parametrize("all_presets", list(change.EVENT_PRESETS))
def test_every_preset_accepts_its_sensitivity_keyword(all_presets):
    preset = change.EVENT_PRESETS[all_presets]
    keyword, values = preset["sensitivity"]
    for value in values:
        preset["classify"](np.zeros(4), np.ones(4), **{keyword: value})


def _area(analysed_pct):
    return change.ChangeAreaSummary(100, analysed_pct, 10, 10.0, analysed_pct)


def test_confidence_high_when_nothing_wrong():
    conf = assess_confidence(_area(100), AreaRange(9.5, 10, 10.5))
    assert conf.rating == "High"


def test_confidence_steps_down_per_issue():
    warn = change.ComparabilityNote("warning", "m", "s")
    assert assess_confidence(_area(80), AreaRange(9.5, 10, 10.5)).rating == "Medium"
    assert assess_confidence(_area(100), AreaRange(8, 10, 12), [warn]).rating == "Low"
    low = assess_confidence(_area(50), AreaRange(5, 10, 20))
    assert low.rating == "Low" and len(low.reasons) == 2


def test_spread_handles_zero_central():
    assert AreaRange(0, 0, 0).spread == 0.0
    assert AreaRange(0, 0, 5).spread == float("inf")

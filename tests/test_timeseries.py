from datetime import date

import numpy as np

from core.timeseries import (
    RECOVERY_PRESETS,
    SamplePoint,
    add_months,
    pick_scenes,
    recovery_findings,
    recovery_summary,
    sample_point,
)
from tests.conftest import make_item


def test_add_months_wraps_years_and_clamps_day():
    assert add_months(date(2018, 11, 30), 3) == date(2019, 2, 28)
    assert add_months(date(2018, 1, 15), -1) == date(2017, 12, 15)


def test_pick_scenes_keeps_least_cloudy_per_period_and_the_pair():
    items = [
        make_item("jan-a", "2018-01-05", cloud=30), make_item("feb-b", "2018-02-10", cloud=5),
        make_item("apr-a", "2018-04-02", cloud=20), make_item("may-b", "2018-05-20", cloud=50),
        make_item("jul-a", "2018-07-01", cloud=1),
    ]
    before, after = make_item("before", "2018-04-20", cloud=40), make_item("after", "2018-05-01", cloud=40)
    picked = [i.id for i in pick_scenes(items, date(2018, 1, 1), 3, keep=[before, after])]
    # Jan-Mar: feb-b (least cloud); Apr-Jun: the pair replaces apr-a/may-b; Jul: jul-a.
    assert picked == ["feb-b", "before", "after", "jul-a"]


def test_sample_point_medians_and_cloud_cut():
    index = np.array([[0.1, 0.2, 0.6], [0.7, 0.8, np.nan]])
    affected = np.array([[True, True, False], [False, False, False]])
    control = ~affected
    valid = np.ones_like(affected)
    point = sample_point(index, valid, affected, control, date(2019, 1, 1))
    assert round(point.affected, 3) == 0.15 and point.control == 0.7 and point.clear_share == 1.0
    cloudy = valid.copy()
    cloudy[0, :2] = False
    assert sample_point(index, cloudy, affected, control, date(2019, 1, 1)).gap is None


def _points(gaps, control=0.5):
    return [SamplePoint(when, None if gap is None else control + gap, control, 1.0) for when, gap in gaps]


def test_recovery_summary_measures_share_of_gap_closed():
    points = _points([
        (date(2018, 2, 1), 0.0), (date(2018, 5, 1), 0.02), (date(2018, 11, 1), 0.0),
        (date(2018, 11, 20), -0.4), (date(2019, 6, 1), None), (date(2020, 11, 1), -0.2),
    ])
    summary = recovery_summary(points, date(2018, 11, 1), date(2018, 11, 20))
    assert summary.before.when == date(2018, 11, 1) and summary.after.when == date(2018, 11, 20)
    assert summary.latest.when == date(2020, 11, 1)
    assert round(summary.recovery_pct) == 50
    assert round(summary.event_shift, 2) == -0.4 and summary.control_shift == 0.0

    texts = [f.text for f in recovery_findings(summary, RECOVERY_PRESETS["wildfire"], "nbr")]
    assert texts[0].endswith("so the change comes from the event, not the season.")
    assert "fell by 0.40" in texts[0]
    assert texts[1] == (
        "By 2020-11-01, 2.0 years after the fire, the burned area is about 50% of the way back to its "
        "pre-event NBR, measured against nearby unburned land (a partial recovery)."
    )


def test_recovery_findings_flags_seasonal_change_and_missing_gap():
    before, after = date(2019, 1, 1), date(2019, 2, 1)
    # Both areas drop 0.2 together: the gap never opens.
    points = [SamplePoint(before, 0.6, 0.5, 1.0), SamplePoint(after, 0.4, 0.3, 1.0), SamplePoint(date(2019, 6, 1), 0.6, 0.5, 1.0)]
    summary = recovery_summary(points, before, after)
    assert summary.recovery_pct is None
    texts = [f.text for f in recovery_findings(summary, RECOVERY_PRESETS["flood"], "ndwi")]
    assert "much of this change is seasonal or regional" in texts[0]
    assert "barely stood out" in texts[1]


def test_recovery_summary_needs_clear_scenes_on_both_sides():
    points = _points([(date(2019, 3, 1), -0.3)])
    assert recovery_summary(points, date(2019, 1, 1), date(2019, 2, 1)) is None
    summary = recovery_summary(_points([(date(2019, 1, 1), 0.0), (date(2019, 2, 1), -0.3)]), date(2019, 1, 1), date(2019, 2, 1))
    assert summary.latest is None
    assert "can't be measured" in recovery_findings(summary, RECOVERY_PRESETS["logging"], "ndvi")[1].text

import numpy as np

from core import change, findings


def _run(preset, codes, px_ha=1.0, notes=()):
    codes = np.asarray(codes)
    baseline = change.EVENT_PRESETS[preset]["baseline_codes"]
    area = change.change_area_summary(codes, baseline, px_ha)
    return findings.change_findings(preset, codes, baseline, area, px_ha, notes, "ndvi")


def _texts(found, kind=None):
    return [f.text for f in found if kind is None or f.kind == kind]


def test_wildfire_headline_and_severity():
    codes = np.full((10, 10), 2)
    codes[:2] = 6   # 20 ha high severity
    codes[2:4] = 3  # 20 ha low severity
    found = _run("wildfire", codes)
    assert found[0].kind == "headline"
    assert found[0].text == "About 40 ha (40.0% of the analysed area) burned at low severity or worse."
    assert "20 ha of that is moderate-high or high severity, 50% of the burned area." in _texts(found)


def test_flood_new_and_lost_water():
    codes = np.zeros((10, 10), dtype=int)
    codes[0] = 2
    codes[1, :5] = 3
    codes[9] = 1
    texts = _texts(_run("flood", codes))
    assert texts[0] == "10 ha (10.0% of the analysed area) became newly water-covered."
    assert "5.0 ha (5.0% of the analysed area) lost surface water." in texts
    assert "10 ha was water on both dates." in texts


def test_logging_net_loss():
    codes = np.ones((10, 10), dtype=int)
    codes[:3] = 2
    codes[9] = 3
    texts = _texts(_run("logging", codes))
    assert texts[0] == "30 ha (30.0% of the analysed area) lost vegetation cover."
    assert "10 ha gained vegetation, a net loss of 20 ha." in texts


def test_negligible_change_and_no_location():
    codes = np.full((20, 20), 2)
    codes[0, 0] = 6
    found = _run("wildfire", codes)
    assert found[0].text.startswith("Little change detected: 0.2%")
    assert not any("concentrated" in t or "spread" in t for t in _texts(found))


def test_location_concentrated_and_spread():
    changed = np.zeros((30, 30), dtype=bool)
    changed[20:, 20:] = True
    assert findings.change_location(changed) == "south-east"
    assert findings.change_location(np.ones((30, 30), dtype=bool)) == "spread"
    assert findings.change_location(np.zeros((30, 30), dtype=bool)) is None


def test_caveats_for_low_coverage_and_warnings():
    codes = np.full((10, 10), -1)
    codes[:5] = 2
    codes[0, :3] = 6
    notes = [
        change.ComparabilityNote("warning", "long", "Seasonal caveat."),
        change.ComparabilityNote("info", "long", "Ignored info."),
    ]
    caveats = _texts(_run("wildfire", codes, notes=notes), "caveat")
    assert caveats[0].startswith("Only 50% of the AOI could be analysed")
    assert caveats[1:] == ["Seasonal caveat."]


def test_nothing_analysed():
    found = _run("flood", np.full((4, 4), -1))
    assert [f.kind for f in found] == ["caveat"]


def test_seasonal_warning_has_summary():
    from tests.conftest import make_item
    notes = change.comparability_checks(make_item("b", "2020-01-01"), make_item("a", "2020-07-01"))
    assert any(n.severity == "warning" and "seasonal" in n.summary for n in notes)

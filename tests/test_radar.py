from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

from core import radar
from core.radar import (
    RadarFloodSummary,
    backscatter_db,
    classify_radar_water,
    pick_radar_pair,
    radar_findings,
    radar_flood_summary,
)
from tests.conftest import make_da


def _item(when, orbit=64, state="descending", href=None):
    return SimpleNamespace(
        datetime=datetime.fromisoformat(when),
        properties={"sat:relative_orbit": orbit, "sat:orbit_state": state},
        assets={"vv": SimpleNamespace(href=href)},
    )


def test_pick_radar_pair_prefers_closest_after_and_matching_track():
    after = [_item("2017-09-02", orbit=34), _item("2017-08-29", orbit=64)]
    before = [_item("2017-08-18", orbit=34), _item("2017-08-12", orbit=64)]
    pair = pick_radar_pair(before, after, date(2017, 8, 18), date(2017, 8, 30))
    assert pair.after.datetime.date() == date(2017, 8, 29)
    # Aug 18 is closer to the optical date, but Aug 12 is on the same track.
    assert pair.before.datetime.date() == date(2017, 8, 12)
    assert pair.same_track
    assert len(pair.after_items) == 1 and len(pair.before_items) == 1


def test_pick_radar_pair_keeps_every_scene_of_the_pass():
    after = [_item("2017-08-29T00:26:20", orbit=34), _item("2017-08-29T00:26:45", orbit=34),
             _item("2017-08-30T12:22:03", orbit=143)]
    before = [_item("2017-08-05T00:26:19", orbit=34), _item("2017-08-05T00:26:44", orbit=34)]
    pair = pick_radar_pair(before, after, date(2017, 8, 5), date(2017, 8, 29))
    assert len(pair.after_items) == 2 and len(pair.before_items) == 2
    assert all(item.properties["sat:relative_orbit"] == 34 for item in pair.after_items)


def test_pick_radar_pair_falls_back_to_any_track():
    pair = pick_radar_pair([_item("2017-08-18", orbit=34)], [_item("2017-08-30", orbit=64)],
                           date(2017, 8, 18), date(2017, 8, 30))
    assert pair is not None and not pair.same_track
    assert pick_radar_pair([], [_item("2017-08-30")], date(2017, 8, 18), date(2017, 8, 30)) is None


def test_backscatter_db_resamples_onto_grid_in_db(tmp_path, monkeypatch):
    monkeypatch.setattr(radar, "SPECKLE_FILTER_PX", 1)
    reference = make_da(np.zeros((20, 20)), res=20.0)
    # 10 m source covering the reference: water (0.01 = -20 dB) on the west
    # half, land (0.1 = -10 dB) on the east, and a no-data strip in the north.
    data = np.full((60, 60), 0.1, dtype="float32")
    data[:, :30] = 0.01
    data[:7, :] = 0.0
    path = tmp_path / "vv.tif"
    left, top = 500_000.0 - 50, 4_400_000.0 + 50
    with rasterio.open(path, "w", driver="GTiff", height=60, width=60, count=1, dtype="float32",
                       crs="EPSG:32610", transform=from_origin(left, top, 10, 10), nodata=0) as dst:
        dst.write(data, 1)

    db = backscatter_db([_item("2017-08-30", href=str(path))], reference)
    assert db.shape == (20, 20)
    assert np.isnan(db[0]).all()
    assert np.allclose(db[5:, :5], -20, atol=0.1)
    assert np.allclose(db[5:, -5:], -10, atol=0.1)


def test_classify_radar_water_uses_db_threshold():
    before = np.array([[-20.0, -10.0, -20.0, np.nan]])
    after = np.array([[-20.0, -20.0, -10.0, -20.0]])
    assert classify_radar_water(before, after).tolist() == [[1, 2, 3, -1]]
    assert classify_radar_water(before, after, threshold_db=-22).tolist() == [[0, 0, 0, -1]]


def test_radar_flood_summary_counts_blind_area_and_buildings():
    radar_codes = np.array([[2, 2], [0, 2]])
    optical_codes = np.array([[2, -1], [0, -1]])
    summary = radar_flood_summary(radar_codes, optical_codes, 0.5, np.array([0, 1, 1]), np.array([1, 0, 1]))
    assert summary.new_water_ha == 1.5
    assert summary.blind_new_water_ha == 1.0
    assert summary.optical_new_water_ha == 0.5
    assert summary.buildings_in_new_water == 2


def test_radar_findings_wording():
    texts = [f.text for f in radar_findings(RadarFloodSummary(3200, 1100, 2100, 85))]
    assert texts == [
        "Radar (Sentinel-1, which sees through cloud) maps 3,200 ha of new open water, "
        "1,100 ha of it where the optical images had no clear view.",
        "85 mapped buildings sit in radar-mapped new water.",
    ]
    assert "no new open water" in radar_findings(RadarFloodSummary(0, 0, 0, None))[0].text


def test_backscatter_db_merges_scenes_from_one_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(radar, "SPECKLE_FILTER_PX", 1)
    reference = make_da(np.zeros((20, 20)), res=20.0)
    left, top = 500_000.0 - 50, 4_400_000.0 + 50
    items = []
    for name, rows in (("north", slice(0, 30)), ("south", slice(30, 60))):
        data = np.zeros((60, 60), dtype="float32")
        data[rows] = 0.1
        path = tmp_path / f"{name}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=60, width=60, count=1, dtype="float32",
                           crs="EPSG:32610", transform=from_origin(left, top, 10, 10), nodata=0) as dst:
            dst.write(data, 1)
        items.append(_item("2017-08-30", href=str(path)))

    db = backscatter_db(items, reference)
    assert np.isfinite(db).all()
    assert np.allclose(db, -10, atol=0.1)


def test_backscatter_db_leaves_area_outside_the_scene_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(radar, "SPECKLE_FILTER_PX", 1)
    reference = make_da(np.zeros((20, 20)), res=20.0)
    path = tmp_path / "west.tif"
    # Covers only the reference's western 200 m (10 of its 20 columns).
    with rasterio.open(path, "w", driver="GTiff", height=40, width=20, count=1, dtype="float32",
                       crs="EPSG:32610", transform=from_origin(500_000.0, 4_400_000.0, 10, 10), nodata=0) as dst:
        dst.write(np.full((40, 20), 0.1, dtype="float32"), 1)

    db = backscatter_db([_item("2017-08-30", href=str(path))], reference)
    assert np.allclose(db[:, :10], -10, atol=0.1)
    assert np.isnan(db[:, 10:]).all()

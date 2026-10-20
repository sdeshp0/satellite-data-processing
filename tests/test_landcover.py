from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

from core import change, findings
from core.landcover import landcover_breakdown, worldcover_on_grid
from tests.conftest import make_da


def _worldcover_tile(path, reference):
    """A 4326 GeoTIFF over the reference footprint: tree cover west, built-up east."""
    w, s, e, n = transform_bounds(reference.rio.crs, "EPSG:4326", *reference.rio.bounds())
    pad = 0.01
    w, s, e, n = w - pad, s - pad, e + pad, n + pad
    data = np.full((200, 200), 10, dtype=np.uint8)
    mid_lon = (w + e) / 2
    data[:, 100:] = 50
    with rasterio.open(
        path, "w", driver="GTiff", height=200, width=200, count=1, dtype="uint8",
        crs="EPSG:4326", transform=from_bounds(w, s, e, n, 200, 200), nodata=0,
    ) as dst:
        dst.write(data, 1)
    return mid_lon


def test_worldcover_on_grid_reprojects_onto_reference(tmp_path):
    reference = make_da(np.zeros((40, 40)), res=50.0)
    _worldcover_tile(tmp_path / "wc.tif", reference)
    item = SimpleNamespace(assets={"map": SimpleNamespace(href=str(tmp_path / "wc.tif"))})
    landcover = worldcover_on_grid([item], reference)
    assert landcover.shape == (40, 40)
    assert landcover[20, 2] == 10 and landcover[20, 37] == 50
    assert set(np.unique(landcover)) <= {10, 50}


def test_worldcover_on_grid_no_items_is_empty():
    reference = make_da(np.zeros((5, 5)))
    assert not worldcover_on_grid([], reference).any()


def _breakdown():
    landcover = np.full((10, 10), 10)
    landcover[:, 8:] = 50
    analysed = np.ones((10, 10), dtype=bool)
    changed = np.zeros((10, 10), dtype=bool)
    changed[:5, :] = True                   # 40 tree cover + 10 built-up pixels
    return landcover_breakdown(landcover, changed, analysed, pixel_area_ha=1.0)


def test_landcover_breakdown_shares():
    df = _breakdown()
    assert df["Land cover"].tolist() == ["Tree cover", "Built-up"]
    assert df["Changed area (ha)"].tolist() == [40.0, 10.0]
    assert df["Share of detected change (%)"].tolist() == [80.0, 20.0]
    assert df["Share of this land cover changed (%)"].tolist() == [50.0, 50.0]


def test_landcover_breakdown_no_change_is_empty():
    df = landcover_breakdown(np.full((3, 3), 10), np.zeros((3, 3), bool), np.ones((3, 3), bool), 1.0)
    assert df.empty and "Land cover" in df.columns


def test_findings_name_land_cover_and_built_up():
    codes = np.full((10, 10), 2)
    codes[:5] = 6
    baseline = change.EVENT_PRESETS["wildfire"]["baseline_codes"]
    area = change.change_area_summary(codes, baseline, 1.0)
    texts = [f.text for f in findings.change_findings("wildfire", codes, baseline, area, 1.0, landcover=_breakdown())]
    assert "Most of the change was on tree cover (80%), followed by built-up (20%)." in texts
    assert "10 ha of built-up land falls in the change area." in texts

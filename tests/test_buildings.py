from types import SimpleNamespace

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from rasterio.transform import xy

from core import change, findings
from core.buildings import (
    building_counts,
    building_pixels,
    exposure_table,
    partition_paths,
    quadkey,
    quadkeys_for_bbox,
)
from tests.conftest import make_da


def _partition(base, reference, pixels):
    """
    A local region dataset like ms-buildings: one quadkey folder (named
    without leading zeros) holding a small square footprint centred on each
    (row, col).
    """
    to_lonlat = Transformer.from_crs(reference.rio.crs, "EPSG:4326", always_xy=True)
    geoms = []
    for row, col in pixels:
        x, y = xy(reference.rio.transform(), row, col)
        lon, lat = to_lonlat.transform([x - 3, x + 3, x + 3, x - 3], [y - 3, y - 3, y + 3, y + 3])
        geoms.append(shapely.Polygon(zip(lon, lat)))
    key = quadkey(lon[0], lat[0])
    folder = base / f"quadkey={key.lstrip('0')}"
    folder.mkdir(parents=True)
    pd.DataFrame({"geometry": shapely.to_wkb(geoms)}).to_parquet(folder / "part-0.parquet")
    return SimpleNamespace(assets={"data": SimpleNamespace(href=str(base), extra_fields={})})


def test_building_pixels_puts_centroids_on_the_grid(tmp_path):
    reference = make_da(np.zeros((20, 30)), res=10.0)
    item = _partition(tmp_path / "region", reference, [(0, 0), (5, 7), (19, 29)])
    rows, cols = building_pixels([item], reference)
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 0), (5, 7), (19, 29)]


def test_building_pixels_drops_buildings_outside_the_grid(tmp_path):
    reference = make_da(np.zeros((20, 30)), res=10.0)
    item = _partition(tmp_path / "region", reference, [(5, 5), (-3, 5), (5, 40)])
    rows, cols = building_pixels([item], reference)
    assert list(zip(rows.tolist(), cols.tolist())) == [(5, 5)]


def test_building_counts_skip_masked_pixels():
    codes = np.array([[2, 3], [5, -1]])
    rows, cols = np.array([0, 0, 1, 1, 1]), np.array([0, 1, 0, 0, 1])
    assert building_counts(rows, cols, codes) == {2: 1, 3: 1, 5: 2}


def test_exposure_table_lists_changed_classes_only():
    df = exposure_table({2: 50, 3: 10, 6: 30}, change.DNBR_LABELS, [2])
    assert list(df["Class"]) == ["Low Severity", "High Severity"]
    assert list(df["Buildings"]) == [10, 30]
    assert list(df["Share of buildings in change area (%)"]) == [25.0, 75.0]


def test_findings_count_buildings_in_burned_area():
    codes = np.full((100, 100), 2)
    codes[:30, :30] = 6
    area = change.change_area_summary(codes, [2], 0.01)
    out = findings.change_findings(
        "wildfire", codes, [2], area, 0.01, building_counts={2: 300, 3: 20, 6: 80, 0: 5}
    )
    texts = [f.text for f in out]
    assert "100 mapped buildings sit in the burned area, 25% of the 405 in the analysed area; 80 of them in moderate-high or high severity." in texts


def test_findings_say_when_no_buildings_are_exposed():
    codes = np.full((100, 100), 0)
    codes[:30, :30] = 2
    area = change.change_area_summary(codes, [0, 1], 0.01)
    out = findings.change_findings("flood", codes, [0, 1], area, 0.01, building_counts={0: 1200})
    assert "None of the 1,200 mapped buildings in the analysed area fall in the newly flooded area." in [f.text for f in out]


def test_quadkeys_for_bbox_covers_every_tile():
    assert quadkey(-90, 45, zoom=1) == "0"
    assert quadkey(90, -45, zoom=1) == "3"
    assert sorted(quadkeys_for_bbox((-1, -1, 1, 1), zoom=1)) == ["0", "1", "2", "3"]
    # Camp Fire AOI straddles two zoom-9 tiles.
    assert quadkeys_for_bbox((-121.7698, 39.6452, -121.4735, 39.8738)) == ["023010030", "023010031"]


def test_partition_paths_only_lists_tiles_that_exist(tmp_path):
    (tmp_path / "quadkey=23010030").mkdir()
    (tmp_path / "quadkey=123").mkdir()
    item = SimpleNamespace(assets={"data": SimpleNamespace(href=str(tmp_path), extra_fields={})})
    _, paths = partition_paths(item, ["023010030", "023010031"])
    assert paths == [f"{tmp_path}/quadkey=23010030"]

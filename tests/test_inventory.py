import numpy as np

from core.inventory import dominant_class, feature_regions, inventory_findings
from tests.conftest import make_da


def _grid():
    mask = np.zeros((90, 90), dtype=bool)
    mask[5:25, 5:25] = True       # 400 px, north-west
    mask[60:70, 60:70] = True     # 100 px, south-east
    mask[40, 40] = True           # 1 px, centre
    return mask, make_da(np.zeros((90, 90), dtype="float32"))


def test_feature_regions_ranks_by_area_and_places_them():
    mask, ref = _grid()
    rows, cols = np.array([10, 12, 65, 0]), np.array([10, 12, 65, 0])
    regions, labelled, areas = feature_regions(mask, ref, 0.01, rows, cols)
    assert [r.rank for r in regions] == [1, 2, 3]
    assert [round(r.area_ha, 2) for r in regions] == [4.0, 1.0, 0.01]
    assert [r.sector for r in regions] == ["north-west", "south-east", "centre"]
    assert [r.buildings for r in regions] == [2, 1, 0]
    assert areas.tolist() == [4.0, 1.0, 0.01]
    west, south, east, north = regions[0].bounds
    assert west < regions[0].lon < east and south < regions[0].lat < north
    assert (labelled[5:25, 5:25] == regions[0].label).all()


def test_feature_regions_lists_only_the_largest():
    mask, ref = _grid()
    regions, _, areas = feature_regions(mask, ref, 0.01, max_regions=2)
    assert len(regions) == 2 and len(areas) == 3


def test_inventory_findings_summary_and_caveat():
    mask, ref = _grid()
    regions, _, areas = feature_regions(mask, ref, 0.01, max_regions=2)
    found = inventory_findings("water", regions, areas, 50.0)
    assert found[0].text == (
        "3 water bodies covering 5.0 ha (10% of the clear AOI). The largest, 4.0 ha in the north-west, "
        "is 80% of that; 1 is under 1 ha."
    )
    assert found[1].kind == "caveat" and "2 largest" in found[1].text
    assert inventory_findings("urban", [], np.array([]), 50.0)[0].text == "No built-up patches found at these settings."


def test_dominant_class_per_region():
    mask, ref = _grid()
    regions, labelled, _ = feature_regions(mask, ref, 0.01)
    codes = np.full(mask.shape, 10)
    codes[5:25, 5:20] = 80
    assert dominant_class(labelled, codes, regions)[regions[0].label] == 80
    assert dominant_class(labelled, codes, regions)[regions[1].label] == 10

from rasterio.windows import Window

import core.load as load
from tests.conftest import make_item


def test_fresh_signed_href_drops_existing_token(monkeypatch):
    seen = []
    monkeypatch.setattr(load.pc, "sign", lambda href: seen.append(href) or href + "?st=new")
    stale = "https://acct.blob.core.windows.net/c/B04.tif?st=old&se=old&sp=r&sig=x"
    assert load._fresh_signed_href(stale) == "https://acct.blob.core.windows.net/c/B04.tif?st=new"
    assert seen == ["https://acct.blob.core.windows.net/c/B04.tif"]


def test_fresh_signed_href_unsigned_href():
    # A non-blob URL is returned unchanged by pc.sign, so no network call.
    assert load._fresh_signed_href("https://example.com/a.tif") == "https://example.com/a.tif"


def test_boa_offset_by_processing_baseline():
    assert load._boa_offset_for_item(make_item("a", "2023-06-01", **{"s2:processing_baseline": "05.09"})) == -1000.0
    assert load._boa_offset_for_item(make_item("b", "2020-06-01", **{"s2:processing_baseline": "02.14"})) == 0.0
    assert load._boa_offset_for_item(make_item("c", "2020-06-01")) == 0.0
    assert load._boa_offset_for_item(make_item("d", "2020-06-01", **{"s2:processing_baseline": "n/a"})) == 0.0


def test_geometric_validity_mask_partial_overlap():
    # Window hangs 50% off the left edge of the source.
    window = Window(-10, 0, 20, 10)
    overlap = Window(0, 0, 10, 10)
    mask = load._geometric_validity_mask(window, overlap, (5, 10))
    assert mask.shape == (5, 10)
    assert not mask[:, :5].any()
    assert mask[:, 5:].all()


def test_geometric_validity_mask_no_overlap():
    mask = load._geometric_validity_mask(Window(0, 0, 10, 10), Window(0, 0, 0, 0), (4, 4))
    assert not mask.any()


def test_mosaic_fill_from_neighbouring_utm_zone_stays_in_range():
    # A companion tile in the next UTM zone sits rotated against the
    # primary's grid; pixels it doesn't cover must not be filled with the
    # reprojection's integer nodata (e.g. -2147483648).
    import numpy as np
    from pyproj import Transformer
    from rasterio.enums import Resampling

    from tests.conftest import make_da

    x16, y16 = Transformer.from_crs(4326, 32616, always_xy=True).transform(-90.15, 30.0)
    x15, y15 = Transformer.from_crs(4326, 32615, always_xy=True).transform(-90.15, 30.0)
    primary_valid = np.zeros((200, 200), bool)
    primary_valid[:, 100:] = True
    for dtype in ("int32", "uint16"):
        primary = make_da(np.full((200, 200), 500, dtype=dtype), res=100.0,
                          origin=(x16 - 10000, y16 + 10000), crs="EPSG:32616")
        companion = make_da(np.full((200, 200), 700, dtype=dtype), res=100.0,
                            origin=(x15 - 10000, y15 + 10000), crs="EPSG:32615")
        merged, valid = load._mosaic_fill(primary, primary_valid, companion, np.ones((200, 200), bool),
                                          Resampling.bilinear)
        assert merged.dtype == primary.dtype
        assert set(np.unique(merged.values[valid])) <= {500, 700}
        assert valid[:, :100].mean() > 0.9

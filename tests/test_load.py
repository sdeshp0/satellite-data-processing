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

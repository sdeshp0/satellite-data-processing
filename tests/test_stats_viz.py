import numpy as np

from core.stats import array_statistics, scl_breakdown
from core.viz import to_rgb
from tests.conftest import make_da


def test_array_statistics_ignores_nan():
    stats = array_statistics(np.array([1.0, 2.0, 3.0, np.nan]))
    assert stats["mean"] == 2.0 and stats["max"] == 3.0


def test_array_statistics_all_nan():
    assert np.isnan(array_statistics(np.array([np.nan, np.nan]))["mean"])


def test_scl_breakdown_percentages():
    df = scl_breakdown(make_da(np.array([[4, 4], [6, 0]], dtype="uint8")))
    assert df.iloc[0]["Class"] == "Vegetation"
    assert df.iloc[0]["Percent"] == 50.0
    assert df["Percent"].sum() == 100.0


def test_to_rgb_handles_masked_pixels():
    vals = np.array([[0.1, 0.2], [0.3, np.nan]], dtype="float32")
    b = {k: make_da(vals) for k in ("red", "green", "blue")}
    rgb = to_rgb(b)
    assert rgb.dtype == np.uint8 and rgb.shape == (2, 2, 3)
    assert rgb[1, 1].tolist() == [0, 0, 0]


def test_highlight_mask_shades_fill_and_outlines_edge():
    from core.viz import highlight_mask
    rgb = np.zeros((7, 7, 3), dtype=np.uint8)
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    out = highlight_mask(rgb, mask, color=(200, 100, 0), fill_alpha=0.5, outline_px=1)
    assert out[1, 1].tolist() == [200, 100, 0]   # edge: solid outline
    assert out[3, 3].tolist() == [100, 50, 0]    # interior: half-shaded
    assert out[0, 0].tolist() == [0, 0, 0]       # outside: untouched
    assert rgb.sum() == 0                         # input not modified

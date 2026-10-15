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



def test_draw_ellipses_outlines_without_fill():
    from core.features import ChangeCluster
    from core.viz import draw_ellipses
    rgb = np.zeros((50, 50, 3), dtype=np.uint8)
    e = ChangeCluster(row=25, col=25, semi_major=15, semi_minor=10, orientation=0.0, pixels=1)
    out = draw_ellipses(rgb, [e], color=(200, 100, 0), alpha=1.0, width_px=1)
    assert out[25, 25].tolist() == [0, 0, 0]         # centre untouched
    assert out[10, 25].tolist() == [200, 100, 0]     # end of major (row) axis
    assert out[25, 15].tolist() == [200, 100, 0]     # end of minor (col) axis
    assert rgb.sum() == 0

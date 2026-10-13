import numpy as np

from core.utils import apply_cloud_mask, apply_nodata_mask, scale_bands
from tests.conftest import make_da


def test_scale_bands_leaves_scl_untouched():
    raw = {"red": make_da(np.full((2, 2), 1000, dtype="uint16")),
           "scl": make_da(np.full((2, 2), 4, dtype="uint8"))}
    out = scale_bands(raw)
    assert np.allclose(out["red"].values, 0.1)
    assert out["scl"].values.tolist() == [[4, 4], [4, 4]]


def test_cloud_mask_masks_cloud_classes_only():
    scl = make_da(np.array([[4, 9], [3, 0]], dtype="uint8"))
    red = make_da(np.ones((2, 2), dtype="float32"))
    out = apply_cloud_mask({"red": red, "scl": scl})
    assert np.isnan(out["red"].values).tolist() == [[False, True], [True, False]]


def test_nodata_mask_masks_scl_zero_and_keeps_scl(bands):
    out = apply_nodata_mask(bands)
    for name in ("red", "nir", "swir2"):
        assert np.isnan(out[name].values).tolist() == [[False, False], [False, True]]
    assert out["scl"].values.tolist() == bands["scl"].values.tolist()

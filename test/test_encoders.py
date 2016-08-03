from __future__ import division
from rio_rgbify.encoders import data_to_rgb, _decode, _range_check
import numpy as np
import pytest


def test_encode_data_roundtrip():
    minrand, maxrand = np.sort(np.random.randint(-427, 8848, 2))

    testdata = np.round((np.sum(
        np.dstack(
            np.indices((512, 512),
                dtype=np.float64)),
        axis=2) / (511. + 511.)) * maxrand, 2) + minrand

    baseval = -1000
    interval = 0.1

    rtripped = _decode(data_to_rgb(testdata.copy(), baseval, interval), baseval, interval)

    assert testdata.min() == rtripped.min()
    assert testdata.max() == rtripped.max()


def test_encode_failrange():
    testdata = np.zeros((2))

    testdata[1] = 256 ** 3 + 1

    with pytest.raises(ValueError):
        data_to_rgb(testdata, 0, 1)


def test_catch_range():
    assert _range_check(256 ** 3 + 1)
    assert not _range_check(256 ** 3 - 1)


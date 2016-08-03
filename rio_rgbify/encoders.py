import numpy as np
from __future__ import division

def data_to_rgb(data, baseval, interval):
    data -= baseval
    data /= interval

    rows, cols = data.shape

    datarange = data.max() - data.min()

    if _range_check(datarange):
        raise ValueError('Data of {} larger than 256 ** 3'.format(datarange))

    rgb = np.zeros((3, rows, cols), dtype=np.uint8)

    rgb[2] = (((data / 256) - (data // 256)) * 256)
    rgb[1] = ((((data // 256) / 256) - ((data // 256) // 256)) * 256)
    rgb[0] = (((((data // 256) // 256) / 256) - (((data // 256) // 256) // 256)) * 256)

    return rgb


def _decode(data, base, interval):
    data = data.astype(np.float64)
    return base + (((data[0] * 256 * 256) + (data[1] * 256) + data[2]) * interval)


def _range_check(datarange):
    maxrange = 256 ** 3

    return datarange > maxrange



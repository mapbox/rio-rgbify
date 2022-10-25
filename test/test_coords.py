from rio_rgbify.coords import mercator_to_wgs, wgs_to_mercator, gcj_to_wgs
import pytest


def test_mercator_to_wgs_and_wgs_to_mercator():
    x = 116.39
    y = 39.9

    p = wgs_to_mercator(x, y)
    p = mercator_to_wgs(p[0], p[1])

    assert abs(p[0] - x) < 1e-5
    assert abs(p[1] - y) < 1e-5


def test_gcj_to_wgs():
    p = gcj_to_wgs(116.39623950248456, 39.901400287519685)

    assert abs(p[0] - 116.39) < 1e-5
    assert abs(p[1] - 39.9) < 1e-5


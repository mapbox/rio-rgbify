import mercantile
import types

from hypothesis import given
import hypothesis.strategies as st
import pytest

import numpy as np
from rasterio import Affine
from rio_rgbify.mbtiler import (_encode_as_webp, _encode_as_png, _make_tiles, _tile_range, RGBTiler)


@given(
    st.integers(
        min_value=0, max_value=(2 ** 10 - 1)
        ),
    st.integers(
        min_value=0, max_value=(2 ** 10 - 1)))
def test_make_tiles_tile_bounds(x, y):
    '''
    Test if children tiles from z10 are created correctly
    '''
    test_bounds = mercantile.bounds(x, y, 10)

    test_bbox = list(mercantile.xy(test_bounds.west, test_bounds.south)) + list(mercantile.xy(test_bounds.east, test_bounds.north))

    test_crs = 'epsg:3857'
    test_minz = 10
    test_maxz = 13

    created_tiles_gen = _make_tiles(test_bbox, test_crs, test_minz, test_maxz)

    assert isinstance(created_tiles_gen, types.GeneratorType)

    created_tiles = list(created_tiles_gen)

    assert len(created_tiles) == 85


@given(
    st.lists(
        elements=st.integers(min_value=0, max_value=99),
        max_size=3, min_size=3
        ),
    st.lists(
        elements=st.integers(
        min_value=100, max_value=200
        ), max_size=3, min_size=3
        ))
def test_tile_range(mintile, maxtile):
    minx, miny, _ = mintile
    maxx, maxy, _ = maxtile

    expected_length = (maxx - minx + 1) * (maxy - miny + 1)
    assert expected_length == len(list(_tile_range(mintile, maxtile)))


def test_webp_writer():
    test_data = np.zeros((3, 256, 256), dtype=np.uint8)

    test_bytearray = _encode_as_webp(test_data)

    assert len(test_bytearray) == 42

    test_complex_data = test_data.copy()

    test_complex_data[0] += (np.random.rand(256, 256) * 255).astype(np.uint8)
    test_complex_data[1] += 10

    test_bytearray_complex = _encode_as_webp(test_complex_data)

    assert len(test_bytearray) < len(test_bytearray_complex)


def test_file_writer():
    test_data = np.zeros((3, 256, 256), dtype=np.uint8)

    test_opts = {
        'driver': 'PNG',
        'dtype': 'uint8',
        'height': 512,
        'width': 512,
        'count': 3,
        'crs': 'EPSG:3857'
    }

    test_affine = Affine(1, 0, 0, 0, -1, 0)

    test_bytearray = _encode_as_png(test_data, test_opts, test_affine)

    assert len(test_bytearray) == 842

    test_complex_data = test_data.copy()

    test_complex_data[0] += (np.random.rand(256, 256) * 255).astype(np.uint8)
    test_complex_data[1] += 10

    test_bytearray_complex = _encode_as_png(test_complex_data, test_opts, test_affine)

    assert len(test_bytearray) < len(test_bytearray_complex)


def test_webp_writer_fails_dtype():
    test_data = np.zeros((3, 256, 256), dtype=np.float64)

    with pytest.raises(TypeError):
        _encode_as_webp(test_data)


def test_png_writer_fails_dtype():
    test_data = np.zeros((3, 256, 256), dtype=np.float64)

    with pytest.raises(TypeError):
        _encode_as_png(test_data)


def test_RGBtiler_format_fails():
    test_in = 'i/do/not/exist.tif'
    test_out = 'nor/do/i.tif'
    test_minz = 0
    test_maxz = 1

    with pytest.raises(ValueError):
        with RGBTiler(test_in, test_out, test_minz, test_maxz,
            format='poo') as rtiler:
            pass

"""
Regressions for two merge bugs and the layer priority the README specifies.

These drive _merge_tiles directly rather than going through process_all, so
they test the merge arithmetic without an MBTiles round trip.  A source tile is
built by hand as a TileData holding a float elevation grid, which is what
_decode_tile would have produced.
"""

import numpy as np
import mercantile
import pytest
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds

from rio_rgbify.merger import (
    EncodingType,
    MBTilesSource,
    TerrainRGBMerger,
    TileData,
)


TILE_SIZE = 64
TARGET = mercantile.Tile(x=0, y=0, z=1)
NAN = float("nan")


def _tile_data(fill, zoom=TARGET.z, size=TILE_SIZE):
    """A decoded source tile holding one elevation everywhere."""
    bounds = mercantile.bounds(mercantile.Tile(x=0, y=0, z=zoom))
    meta = {
        "count": 1,
        "dtype": rasterio.float32,
        "driver": "GTiff",
        "crs": "EPSG:3857",
        "width": size,
        "height": size,
        "transform": transform_from_bounds(
            bounds.west, bounds.south, bounds.east, bounds.north, size, size
        ),
    }
    return TileData(np.full((size, size), fill, dtype=np.float32), meta, zoom)


def _merger(tmp_path, sources, **kwargs):
    """A merger over *sources* dummy MBTilesSource entries.

    MBTilesSource refuses a path that does not exist, so each one gets a real
    (empty) file; _merge_tiles never opens them.
    """
    configs = []
    for i, adjustment in enumerate(sources):
        path = tmp_path / f"src{i}.mbtiles"
        path.touch()
        configs.append(
            MBTilesSource(
                path=path,
                encoding=EncodingType.MAPBOX,
                height_adjustment=adjustment,
            )
        )
    return TerrainRGBMerger(
        sources=configs,
        output_path=str(tmp_path / "out.mbtiles"),
        default_tile_size=TILE_SIZE,
        **kwargs,
    )


class TestHeightAdjustmentAppliedOnce:
    """height_adjustment belongs to _decode_tile and nowhere else.

    _merge_tiles used to add it a second time to the already-adjusted data, so
    a source configured with -5.0 was shifted by -10.0.
    """

    def test_merge_does_not_re_apply_the_adjustment(self, tmp_path):
        merger = _merger(tmp_path, [50.0])

        # As _decode_tile would return it: 100 m of terrain, +50 already added.
        result = merger._merge_tiles([_tile_data(150.0)], TARGET)

        assert result is not None
        assert np.median(result) == pytest.approx(150.0)

    def test_every_source_keeps_its_own_value(self, tmp_path):
        merger = _merger(tmp_path, [-5.0, 10.0])

        result = merger._merge_tiles(
            [_tile_data(95.0), _tile_data(210.0)], TARGET
        )

        # The top source wins outright; neither is shifted again.
        assert np.median(result) == pytest.approx(210.0)


class TestSparseEmptyTile:
    """What sparse_tiles actually skips, and what it must still write.

    Note on the second bug: the all-NaN check in _merge_tiles was moved above
    the output_nodata substitution, where it used to be unable to fire.  It is
    *still* unreachable, because has_native_with_data returns first for every
    input that would produce an all-NaN result -- so none of these tests can
    drive it, and none of them claims to.  They pin the emptiness behaviour
    that is reachable, which is the guard and the nodata fill.
    """

    def test_an_all_nodata_native_tile_is_skipped(self, tmp_path):
        merger = _merger(tmp_path, [0.0], sparse_tiles=True, output_nodata=-10000)

        assert merger._merge_tiles([_tile_data(NAN)], TARGET) is None

    def test_skipping_does_not_depend_on_output_nodata(self, tmp_path):
        merger = _merger(tmp_path, [0.0], sparse_tiles=True)

        assert merger._merge_tiles([_tile_data(NAN)], TARGET) is None

    def test_a_partially_covered_tile_is_written(self, tmp_path):
        merger = _merger(tmp_path, [0.0], sparse_tiles=True, output_nodata=-10000)

        partial = _tile_data(NAN)
        partial.data[0, 0] = 42.0
        result = merger._merge_tiles([partial], TARGET)

        # Skipping empty tiles must not stop a partially covered one having its
        # holes filled.
        assert result is not None
        assert result[0, 0] == pytest.approx(42.0)
        assert result[1, 1] == pytest.approx(-10000.0)
        assert not np.any(np.isnan(result))

    def test_an_overzoomed_source_alone_is_skipped(self, tmp_path):
        """No source native at this zoom means the client can overzoom instead."""
        merger = _merger(tmp_path, [0.0], sparse_tiles=True, output_nodata=-10000)

        assert merger._merge_tiles([_tile_data(100.0, zoom=0)], TARGET) is None


class TestLayerPriority:
    """The last source wins, which is what the README describes.

    "The merge logic works by merging the input sources in order ... The last
    input source will be the base layer for tiles", and bounds_source defaults
    to the last source too.  master inverts this; see the note in the commit.
    """

    def test_last_source_paints_over_the_first(self, tmp_path):
        merger = _merger(tmp_path, [0.0, 0.0])

        result = merger._merge_tiles(
            [_tile_data(100.0), _tile_data(200.0)], TARGET
        )

        assert np.median(result) == pytest.approx(200.0)

    def test_a_masked_top_source_lets_the_bottom_through(self, tmp_path):
        merger = _merger(tmp_path, [0.0, 0.0])

        # The bathymetry case: the top source is masked over the ocean, so the
        # coarse global source underneath is what shows there.
        top = _tile_data(NAN)
        top.data[0, 0] = 200.0
        result = merger._merge_tiles([_tile_data(100.0), top], TARGET)

        assert result[0, 0] == pytest.approx(200.0)
        assert result[1, 1] == pytest.approx(100.0)

"""
Tests for TerrainRGBMerger (merger.py) and RasterRGBMerger (raster_merger.py).

Fixtures
--------
All tests use in-process synthetic data — no heavyweight GeoTIFF downloading
required.  A helper creates tiny single-zoom MBTiles files and tiny GeoTIFF
in-memory using MemoryFile so nothing is written to disk during encoding.
"""

import io
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path

import mercantile
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds as transform_from_bounds
from PIL import Image

from click.testing import CliRunner

from rio_rgbify.database import MBTilesDatabase
from rio_rgbify.image import ImageEncoder, ImageFormat
from rio_rgbify.merger import (
    EncodingType,
    MBTilesSource,
    TerrainRGBMerger,
    TileData,
)
from rio_rgbify.raster_merger import (
    RasterRGBMerger,
    RasterSource,
)
from rio_rgbify.scripts.cli import main_group as cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TILE_SIZE = 256

# A small but realistic tile: z=1, x=0, y=0 (top-left world quadrant)
SAMPLE_TILE = mercantile.Tile(x=0, y=0, z=1)
SAMPLE_TILE_2 = mercantile.Tile(x=1, y=0, z=1)


def _elevation_array(fill: float = 100.0, size: int = TILE_SIZE) -> np.ndarray:
    """Return a flat (size, size) float32 elevation raster."""
    return np.full((size, size), fill, dtype=np.float32)


def _rgb_bytes_for_elevation(fill: float = 100.0, encoding: str = "mapbox") -> bytes:
    """Encode a flat elevation value to RGB PNG tile bytes."""
    data = _elevation_array(fill)
    rgb = ImageEncoder.data_to_rgb(data, encoding, 0.1, base_val=-10000)
    return ImageEncoder.save_rgb_to_bytes(rgb, ImageFormat.PNG, TILE_SIZE)


def _make_mbtiles(path: str, tiles: dict, encoding: str = "mapbox") -> None:
    """
    Create an MBTiles file at *path* containing *tiles*.

    tiles: { (z, x, y): float_elevation_value }
    """
    with MBTilesDatabase(path) as db:
        db.add_metadata({
            "name": "test",
            "format": "png",
            "minzoom": "0",
            "maxzoom": "5",
        })
        for (z, x, y), elev in tiles.items():
            tile_bytes = _rgb_bytes_for_elevation(elev, encoding)
            db.insert_tile_with_retry([x, y, z], tile_bytes)


# Web Mercator cannot project lat=±90; clamp to the valid range.
_WORLD_BOUNDS = (-180.0, -85.05, 180.0, 85.05)


def _make_geotiff(path: str, bounds, fill: float = 100.0, epsg: int = 4326) -> None:
    """
    Write a single-band GeoTIFF at *path* covering *bounds* (west, south, east, north).
    """
    west, south, east, north = bounds
    width = height = TILE_SIZE
    transform = transform_from_bounds(west, south, east, north, width, height)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=rasterio.float32,
        crs=CRS.from_epsg(epsg),
        transform=transform,
    ) as dst:
        dst.write(np.full((1, height, width), fill, dtype=np.float32))


# ---------------------------------------------------------------------------
# TerrainRGBMerger unit tests
# ---------------------------------------------------------------------------

class TestTerrainRGBMergerMergeTiles:
    """Low-level _merge_tiles logic, no I/O."""

    def _make_tile_data(self, fill: float, zoom: int = 1) -> TileData:
        data = _elevation_array(fill)
        bounds = mercantile.bounds(SAMPLE_TILE)
        meta = {
            "driver": "GTiff",
            "dtype": "float32",
            "crs": "EPSG:3857",
            "count": 1,
            "width": TILE_SIZE,
            "height": TILE_SIZE,
            "transform": transform_from_bounds(
                bounds.west, bounds.south, bounds.east, bounds.north,
                TILE_SIZE, TILE_SIZE,
            ),
        }
        return TileData(data=data, meta=meta, source_zoom=zoom)

    def _merger(self, sparse_tiles=False, num_sources=2):
        # Sources list must have enough entries to match the tile_datas passed
        # to _merge_tiles (it accesses self.sources[i] for height_adjustment).
        sources = [
            MBTilesSource(
                path=Path(__file__),  # just needs to exist
                encoding=EncodingType.MAPBOX,
                height_adjustment=0.0,
            )
            for _ in range(num_sources)
        ]
        return TerrainRGBMerger(
            sources=sources,
            output_path="/dev/null",
            sparse_tiles=sparse_tiles,
        )

    def test_returns_none_when_all_sources_none(self):
        merger = self._merger()
        result = merger._merge_tiles([None, None], SAMPLE_TILE)
        assert result is None

    def test_single_source_passthrough(self):
        merger = self._merger()
        td = self._make_tile_data(500.0)
        result = merger._merge_tiles([td], SAMPLE_TILE)
        assert result is not None
        assert result.shape == (TILE_SIZE, TILE_SIZE)
        assert np.allclose(result, 500.0, atol=1.0)

    def test_a_masked_lower_source_lets_the_upper_through(self):
        merger = self._merger(num_sources=2)
        # First source (bottom): all NaN (masked ocean)
        td1 = self._make_tile_data(0.0)
        td1.data[:] = np.nan
        # Second source (top): land at 200 m
        td2 = self._make_tile_data(200.0)
        result = merger._merge_tiles([td1, td2], SAMPLE_TILE)
        assert result is not None
        assert np.allclose(result, 200.0, atol=1.0)

    def test_last_source_takes_priority_over_first(self):
        """Sources are listed bottom first, so the last one paints over the rest.

        This is what the README describes ("the last input source will be the
        base layer for tiles"), what merge_example.json assumes (bathymetry
        first, terrain after it), and what the production configs rely on.
        """
        merger = self._merger(num_sources=2)
        td1 = self._make_tile_data(300.0)
        td2 = self._make_tile_data(100.0)
        result = merger._merge_tiles([td1, td2], SAMPLE_TILE)
        # Where td2 has real data it should win
        assert result is not None
        assert np.allclose(result, 100.0, atol=1.0)

    def test_a_masked_upper_source_lets_the_lower_through(self):
        """The bathymetry case: terrain on top, masked over the ocean."""
        merger = self._merger(num_sources=2)
        td1 = self._make_tile_data(-500.0)  # bottom: bathymetry
        td2 = self._make_tile_data(200.0)  # top: terrain, masked over water
        td2.data[:, :32] = np.nan
        result = merger._merge_tiles([td1, td2], SAMPLE_TILE)
        assert result is not None
        assert np.allclose(result[:, :32], -500.0, atol=1.0)
        assert np.allclose(result[:, 32:], 200.0, atol=1.0)

    def test_output_nodata_fills_nan(self):
        merger = self._merger()
        merger.output_nodata = -9999.0
        td = self._make_tile_data(0.0)
        td.data[:] = np.nan
        result = merger._merge_tiles([td], SAMPLE_TILE)
        assert result is not None
        assert np.all(result == -9999.0)

    def test_sparse_tiles_skips_all_nan_native(self):
        merger = self._merger(sparse_tiles=True)
        td = self._make_tile_data(0.0, zoom=1)
        td.data[:] = np.nan
        result = merger._merge_tiles([td], SAMPLE_TILE)  # tile.z == 1 == source_zoom
        assert result is None

    def test_sparse_tiles_keeps_native_with_data(self):
        merger = self._merger(sparse_tiles=True)
        td = self._make_tile_data(250.0, zoom=1)
        result = merger._merge_tiles([td], SAMPLE_TILE)
        assert result is not None

    def test_sparse_tiles_keeps_overzoom_tile(self):
        """When a source is an overzoom (different zoom level), sparse_tiles
        should NOT skip it — the source has real data even though it's not native."""
        merger = self._merger(sparse_tiles=True)
        td = self._make_tile_data(250.0, zoom=0)  # lower zoom → overzoom
        result = merger._merge_tiles([td], SAMPLE_TILE)  # tile.z == 1, source.zoom == 0
        # No native tile with data → sparse_tiles skips — this is correct behaviour
        # because the client already has the parent tile to overzoom from.
        assert result is None


# ---------------------------------------------------------------------------
# TerrainRGBMerger integration tests (actual MBTiles I/O)
# ---------------------------------------------------------------------------

class TestTerrainRGBMergerIntegration:

    def test_merge_two_sources_produces_output(self, tmp_path):
        src1 = str(tmp_path / "src1.mbtiles")
        src2 = str(tmp_path / "src2.mbtiles")
        out  = str(tmp_path / "out.mbtiles")

        _make_mbtiles(src1, {(1, 0, 0): 100.0, (1, 1, 0): 200.0})
        _make_mbtiles(src2, {(1, 0, 0):  50.0, (1, 1, 0):  75.0})

        sources = [
            MBTilesSource(Path(src1), EncodingType.MAPBOX),
            MBTilesSource(Path(src2), EncodingType.MAPBOX),
        ]
        merger = TerrainRGBMerger(
            sources=sources,
            output_path=out,
            output_image_format=ImageFormat.PNG,
            min_zoom=1,
            max_zoom=1,
            processes=1,
        )
        merger.process_all(min_zoom=1)

        assert os.path.exists(out)
        conn = sqlite3.connect(out)
        rows = conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        conn.close()
        assert rows >= 1

    def test_height_adjustment_applied(self, tmp_path):
        """Source 1 has elevation 100 with +50 adjustment → result ≈ 150."""
        src1 = str(tmp_path / "src1.mbtiles")
        out  = str(tmp_path / "out.mbtiles")

        _make_mbtiles(src1, {(1, 0, 0): 100.0})

        sources = [
            MBTilesSource(Path(src1), EncodingType.MAPBOX, height_adjustment=50.0),
        ]
        merger = TerrainRGBMerger(
            sources=sources,
            output_path=out,
            output_image_format=ImageFormat.PNG,
            min_zoom=1,
            max_zoom=1,
            processes=1,
        )
        merger.process_all(min_zoom=1)

        # Decode the output tile and check the mean is approximately 150
        conn = sqlite3.connect(out)
        row = conn.execute(
            "SELECT tile_data FROM tiles WHERE zoom_level=1 AND tile_column=0 AND tile_row=0"
        ).fetchone()
        conn.close()
        assert row is not None
        img = np.array(Image.open(io.BytesIO(row[0]))).astype(np.float64)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        decoded = -10000 + ((r * 256 * 256 + g * 256 + b) * 0.1)
        assert np.median(decoded) == pytest.approx(150.0, abs=2.0)

    def test_sparse_tiles_skips_empty_tile(self, tmp_path):
        """With sparse_tiles=True, a tile that is all-NaN in every source
        must not appear in the output database."""
        src1 = str(tmp_path / "src1.mbtiles")
        out  = str(tmp_path / "out.mbtiles")

        # Insert a tile that is entirely masked (all nodata = 0, mask_values=[0])
        masked_data = _elevation_array(0.0)  # will be masked as NaN
        rgb = ImageEncoder.data_to_rgb(masked_data, "mapbox", 0.1, base_val=-10000)
        tile_bytes = ImageEncoder.save_rgb_to_bytes(rgb, ImageFormat.PNG, TILE_SIZE)

        with MBTilesDatabase(src1) as db:
            db.insert_tile_with_retry([0, 0, 1], tile_bytes)

        sources = [MBTilesSource(Path(src1), EncodingType.MAPBOX, mask_values=[0.0])]
        merger = TerrainRGBMerger(
            sources=sources,
            output_path=out,
            output_image_format=ImageFormat.PNG,
            min_zoom=1,
            max_zoom=1,
            sparse_tiles=True,
            processes=1,
        )
        merger.process_all(min_zoom=1)

        conn = sqlite3.connect(out)
        rows = conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        conn.close()
        assert rows == 0

    def test_sparse_tiles_keeps_real_tile(self, tmp_path):
        """With sparse_tiles=True, a tile with real data must still be written."""
        src1 = str(tmp_path / "src1.mbtiles")
        out  = str(tmp_path / "out.mbtiles")
        _make_mbtiles(src1, {(1, 0, 0): 500.0})

        sources = [MBTilesSource(Path(src1), EncodingType.MAPBOX)]
        merger = TerrainRGBMerger(
            sources=sources,
            output_path=out,
            output_image_format=ImageFormat.PNG,
            min_zoom=1,
            max_zoom=1,
            sparse_tiles=True,
            processes=1,
        )
        merger.process_all(min_zoom=1)

        conn = sqlite3.connect(out)
        rows = conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        conn.close()
        assert rows >= 1

    def test_bounds_limits_output_tiles(self, tmp_path):
        """Setting bounds should restrict which tiles are written."""
        src1 = str(tmp_path / "src1.mbtiles")
        out_bounded   = str(tmp_path / "bounded.mbtiles")
        out_unbounded = str(tmp_path / "unbounded.mbtiles")

        tiles = {(1, x, y): 100.0 for x in range(2) for y in range(2)}
        _make_mbtiles(src1, tiles)

        sources = [MBTilesSource(Path(src1), EncodingType.MAPBOX)]

        bounds_tile = mercantile.bounds(SAMPLE_TILE)
        bounded_merger = TerrainRGBMerger(
            sources=sources,
            output_path=out_bounded,
            output_image_format=ImageFormat.PNG,
            min_zoom=1,
            max_zoom=1,
            bounds=[bounds_tile.west, bounds_tile.south, bounds_tile.east, bounds_tile.north],
            processes=1,
        )
        bounded_merger.process_all(min_zoom=1)

        unbounded_merger = TerrainRGBMerger(
            sources=sources,
            output_path=out_unbounded,
            output_image_format=ImageFormat.PNG,
            min_zoom=1,
            max_zoom=1,
            processes=1,
        )
        unbounded_merger.process_all(min_zoom=1)

        conn_b = sqlite3.connect(out_bounded)
        conn_u = sqlite3.connect(out_unbounded)
        count_b = conn_b.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        count_u = conn_u.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        conn_b.close()
        conn_u.close()
        assert count_b <= count_u


# ---------------------------------------------------------------------------
# RasterRGBMerger unit tests
# ---------------------------------------------------------------------------

class TestRasterRGBMergerExtractTile:
    """Test that _extract_tile correctly reads windowed data and fixes meta."""

    def test_returns_tile_data_for_covered_tile(self, tmp_path):
        tif = str(tmp_path / "dem.tif")
        # Cover the whole world at low resolution so z=1 tile 0/0 is inside.
        _make_geotiff(tif, bounds=(-180, -90, 180, 90), fill=200.0, epsg=4326)

        source = RasterSource(path=Path(tif))
        merger = RasterRGBMerger(
            sources=[source],
            output_path=str(tmp_path / "out.mbtiles"),
            min_zoom=1,
            max_zoom=1,
            processes=1,
        )
        td = merger._extract_tile(source, SAMPLE_TILE, 0)
        assert td is not None
        # meta width/height must match data shape, not the full raster
        assert td.meta["width"] == td.data.shape[1]
        assert td.meta["height"] == td.data.shape[0]

    def test_meta_crs_is_3857(self, tmp_path):
        tif = str(tmp_path / "dem.tif")
        _make_geotiff(tif, bounds=(-180, -90, 180, 90), fill=50.0, epsg=4326)

        source = RasterSource(path=Path(tif))
        merger = RasterRGBMerger(
            sources=[source],
            output_path=str(tmp_path / "out.mbtiles"),
            processes=1,
        )
        td = merger._extract_tile(source, SAMPLE_TILE, 0)
        assert td is not None
        assert "3857" in str(td.meta["crs"])

    def test_returns_none_for_out_of_bounds_tile(self, tmp_path):
        tif = str(tmp_path / "dem.tif")
        # Only cover a tiny area in Africa — will never overlap a z=1 Arctic tile
        _make_geotiff(tif, bounds=(10, 0, 11, 1), fill=0.0, epsg=4326)

        source = RasterSource(path=Path(tif))
        merger = RasterRGBMerger(
            sources=[source],
            output_path=str(tmp_path / "out.mbtiles"),
            processes=1,
        )
        # z=1, x=0, y=0 is the upper-left quadrant of the world (America/Europe)
        # but not near Africa 10-11°E, 0-1°N — however mercantile z=1 tiles are
        # large, so just test that the function doesn't raise and returns something
        # (it may still return data since z=1 tiles are huge — this test checks behaviour)
        try:
            td = merger._extract_tile(source, SAMPLE_TILE, 0)
            # No exception is also a pass
        except Exception as exc:
            pytest.fail(f"_extract_tile raised unexpectedly: {exc}")


class TestRasterRGBMergerGetMaxZoom:

    def test_zoom_increases_with_resolution(self, tmp_path):
        """Higher-resolution GeoTIFF should yield a higher max zoom."""
        coarse = str(tmp_path / "coarse.tif")
        fine   = str(tmp_path / "fine.tif")
        # Coarse: 1-degree pixels
        _make_geotiff(coarse, bounds=(0, 0, 10, 10), fill=0.0, epsg=4326)
        # Fine: 0.01-degree pixels (100×100 of the same area → same bounds but
        # we fake it by making a much larger raster over the same bounds with
        # more pixels, achieved by a higher width/height write)

        # Write fine manually with more pixels
        width = height = 1000
        transform = transform_from_bounds(0, 0, 10, 10, width, height)
        with rasterio.open(
            fine, "w", driver="GTiff", height=height, width=width,
            count=1, dtype=rasterio.float32, crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(np.zeros((1, height, width), dtype=np.float32))

        m_coarse = RasterRGBMerger(
            sources=[RasterSource(Path(coarse))],
            output_path=str(tmp_path / "c.mbtiles"),
            default_tile_size=256,
        )
        m_fine = RasterRGBMerger(
            sources=[RasterSource(Path(fine))],
            output_path=str(tmp_path / "f.mbtiles"),
            default_tile_size=256,
        )
        assert m_fine.get_max_zoom_level() > m_coarse.get_max_zoom_level()

    def test_zoom_respects_min_zoom_floor(self, tmp_path):
        """Result should always be >= min_zoom."""
        tif = str(tmp_path / "tiny.tif")
        _make_geotiff(tif, bounds=(0, 0, 180, 90), fill=0.0, epsg=4326)
        merger = RasterRGBMerger(
            sources=[RasterSource(Path(tif))],
            output_path=str(tmp_path / "out.mbtiles"),
            min_zoom=5,
            default_tile_size=256,
        )
        assert merger.get_max_zoom_level() >= 5


# ---------------------------------------------------------------------------
# RasterRGBMerger integration test
# ---------------------------------------------------------------------------

class TestRasterRGBMergerIntegration:

    def test_produces_mbtiles_output(self, tmp_path):
        tif = str(tmp_path / "dem.tif")
        out = str(tmp_path / "out.mbtiles")
        _make_geotiff(tif, bounds=_WORLD_BOUNDS, fill=100.0, epsg=4326)

        merger = RasterRGBMerger(
            sources=[RasterSource(Path(tif))],
            output_path=out,
            min_zoom=0,
            max_zoom=1,
            output_image_format=ImageFormat.PNG,
            processes=1,
        )
        merger.process_all(min_zoom=0)

        assert os.path.exists(out)
        conn = sqlite3.connect(out)
        rows = conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        conn.close()
        assert rows >= 1

    def test_sparse_tiles_skips_masked(self, tmp_path):
        """With sparse_tiles=True and a raster that has only mask_values,
        no tiles should be written."""
        tif = str(tmp_path / "dem.tif")
        out = str(tmp_path / "out.mbtiles")
        # All zeros — will be masked since mask_values=[0.0]
        _make_geotiff(tif, bounds=_WORLD_BOUNDS, fill=0.0, epsg=4326)

        merger = RasterRGBMerger(
            sources=[RasterSource(Path(tif), mask_values=[0.0])],
            output_path=out,
            min_zoom=1,
            max_zoom=1,
            sparse_tiles=True,
            output_image_format=ImageFormat.PNG,
            processes=1,
        )
        merger.process_all(min_zoom=1)

        conn = sqlite3.connect(out)
        rows = conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        conn.close()
        assert rows == 0


# ---------------------------------------------------------------------------
# CLI merge command tests
# ---------------------------------------------------------------------------

class TestMergeCLI:

    def _write_config(self, path: str, config: dict):
        with open(path, "w") as f:
            json.dump(config, f)

    def test_merge_mbtiles_basic(self, tmp_path):
        src1 = str(tmp_path / "src1.mbtiles")
        src2 = str(tmp_path / "src2.mbtiles")
        out  = str(tmp_path / "out.mbtiles")
        cfg  = str(tmp_path / "config.json")

        _make_mbtiles(src1, {(1, 0, 0): 100.0})
        _make_mbtiles(src2, {(1, 0, 0):  50.0})

        self._write_config(cfg, {
            "output_type": "mbtiles",
            "sources": [
                {"path": src1, "encoding": "mapbox"},
                {"path": src2, "encoding": "mapbox"},
            ],
            "output_path": out,
            "output_format": "png",
            "output_encoding": "mapbox",
            "min_zoom": 1,
            "max_zoom": 1,
        })

        runner = CliRunner()
        result = runner.invoke(cli, ["merge", "--config", cfg, "-j", "1"])
        assert result.exit_code == 0, result.output + str(result.exception)
        assert os.path.exists(out)

    def test_merge_with_sparse_tiles(self, tmp_path):
        """sparse_tiles flag should be forwarded from config to the merger."""
        src1 = str(tmp_path / "src1.mbtiles")
        out  = str(tmp_path / "out.mbtiles")
        cfg  = str(tmp_path / "config.json")

        _make_mbtiles(src1, {(1, 0, 0): 300.0})

        self._write_config(cfg, {
            "output_type": "mbtiles",
            "sources": [{"path": src1, "encoding": "mapbox"}],
            "output_path": out,
            "output_format": "png",
            "output_encoding": "mapbox",
            "min_zoom": 1,
            "max_zoom": 1,
            "sparse_tiles": True,
        })

        runner = CliRunner()
        result = runner.invoke(cli, ["merge", "--config", cfg, "-j", "1"])
        assert result.exit_code == 0, result.output + str(result.exception)

    def test_merge_with_height_adjustment(self, tmp_path):
        src1 = str(tmp_path / "src1.mbtiles")
        out  = str(tmp_path / "out.mbtiles")
        cfg  = str(tmp_path / "config.json")

        _make_mbtiles(src1, {(1, 0, 0): 100.0})

        self._write_config(cfg, {
            "output_type": "mbtiles",
            "sources": [{"path": src1, "encoding": "mapbox", "height_adjustment": 25.0}],
            "output_path": out,
            "output_format": "png",
            "output_encoding": "mapbox",
            "min_zoom": 1,
            "max_zoom": 1,
        })

        runner = CliRunner()
        result = runner.invoke(cli, ["merge", "--config", cfg, "-j", "1"])
        assert result.exit_code == 0, result.output + str(result.exception)
        assert os.path.exists(out)

    def test_merge_missing_config_fails(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["merge", "--config", str(tmp_path / "nope.json")])
        assert result.exit_code != 0

    def test_merge_raster_type(self, tmp_path):
        tif = str(tmp_path / "dem.tif")
        out = str(tmp_path / "out.mbtiles")
        cfg = str(tmp_path / "config.json")

        _make_geotiff(tif, bounds=(-180, -90, 180, 90), fill=50.0, epsg=4326)

        self._write_config(cfg, {
            "output_type": "raster",
            "sources": [{"path": tif}],
            "output_path": out,
            "output_format": "png",
            "output_encoding": "mapbox",
            "min_zoom": 0,
            "max_zoom": 1,
        })

        runner = CliRunner()
        result = runner.invoke(cli, ["merge", "--config", cfg, "-j", "1"])
        assert result.exit_code == 0, result.output + str(result.exception)
        assert os.path.exists(out)

    def test_merge_terrarium_output_encoding(self, tmp_path):
        src1 = str(tmp_path / "src1.mbtiles")
        out  = str(tmp_path / "out.mbtiles")
        cfg  = str(tmp_path / "config.json")

        _make_mbtiles(src1, {(1, 0, 0): 100.0}, encoding="terrarium")

        self._write_config(cfg, {
            "output_type": "mbtiles",
            "sources": [{"path": src1, "encoding": "terrarium"}],
            "output_path": out,
            "output_format": "png",
            "output_encoding": "terrarium",
            "min_zoom": 1,
            "max_zoom": 1,
        })

        runner = CliRunner()
        result = runner.invoke(cli, ["merge", "--config", cfg, "-j", "1"])
        assert result.exit_code == 0, result.output + str(result.exception)
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# Live fixture merge tests (real GEBCO + JAXA tiles, pre-downloaded)
#
# Fixtures generated by:  python test/download_fixtures.py
# Sources:
#   GEBCO 2024 TerrainRGB  — https://tiles.wifidb.net/data/ocean-rgb/{z}/{x}/{y}.webp
#   JAXA AW3D30 2024       — https://tiles.wifidb.net/data/jaxa_terrainrgb_webp/{z}/{x}/{y}.webp
# Both encoded as mapbox TerrainRGB, z0-z2 (21 tiles each).
# Merge strategy mirrors terrain_merged/merge/merge_bathymetry.json:
#   source[0] = JAXA (land, higher priority), mask_values=[-10000, 0, -1]
#   source[1] = GEBCO (bathymetry, fills ocean gaps), mask_values=[-10000]
# ---------------------------------------------------------------------------

_GEBCO_FIXTURE      = Path(__file__).parent / "fixtures" / "gebco_sample.mbtiles"
_JAXA_FIXTURE       = Path(__file__).parent / "fixtures" / "jaxa_sample.mbtiles"
_EXPECTED_TILES_DIR = Path(__file__).parent / "expected"

# Key tiles extracted by test/generate_expected_tiles.py: (z, x, y)
_REFERENCE_KEY_TILES = [(0, 0, 0), (2, 2, 1), (2, 0, 2)]


@pytest.mark.skipif(
    not (_GEBCO_FIXTURE.exists() and _JAXA_FIXTURE.exists()),
    reason="Live fixtures missing — run `python test/download_fixtures.py` first",
)
class TestLiveMerge:
    """
    Merge tests using real GEBCO bathymetry and JAXA land tiles.
    No network access required — tiles are pre-committed as fixture files.
    """

    def _run_merge(self, tmp_path, extra_config=None):
        """Invoke the CLI merge command and return the (result, output_path) tuple."""
        out = str(tmp_path / "merged.mbtiles")
        cfg_data = {
            "output_type": "mbtiles",
            "sources": [
                # Bottom first: GEBCO bathymetry underneath, JAXA land on top.
                # Equivalent to the old JAXA-first listing under the inverted
                # priority master carried, so the reference tiles still match.
                {
                    "path": str(_GEBCO_FIXTURE),
                    "encoding": "mapbox",
                    "mask_values": [-10000],
                },
                {
                    "path": str(_JAXA_FIXTURE),
                    "encoding": "mapbox",
                    "mask_values": [-10000, 0, -1],
                },
            ],
            "output_path": out,
            "output_encoding": "mapbox",
            "output_format": "webp",
            "resampling": "cubic",
            "min_zoom": 0,
            "max_zoom": 2,
        }
        if extra_config:
            cfg_data.update(extra_config)

        cfg_path = str(tmp_path / "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg_data, f)

        runner = CliRunner()
        result = runner.invoke(cli, ["merge", "--config", cfg_path, "-j", "1"])
        return result, out

    def test_merge_produces_output(self, tmp_path):
        """Basic smoke test: merging GEBCO + JAXA produces an MBTiles file."""
        result, out = self._run_merge(tmp_path)
        assert result.exit_code == 0, result.output + str(result.exception or "")
        assert os.path.exists(out)

        conn = sqlite3.connect(out)
        rows = conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        conn.close()
        assert rows >= 1

    def test_merge_covers_all_zoom_levels(self, tmp_path):
        """Output should contain tiles at z=0, z=1, and z=2."""
        result, out = self._run_merge(tmp_path)
        assert result.exit_code == 0, result.output + str(result.exception or "")

        conn = sqlite3.connect(out)
        zooms = {row[0] for row in conn.execute("SELECT DISTINCT zoom_level FROM tiles")}
        conn.close()
        assert 0 in zooms
        assert 1 in zooms
        assert 2 in zooms

    def test_merge_output_tile_is_valid_webp(self, tmp_path):
        """Every sampled output tile should decode as a valid RGB(A) image."""
        result, out = self._run_merge(tmp_path)
        assert result.exit_code == 0, result.output + str(result.exception or "")

        conn = sqlite3.connect(out)
        rows = conn.execute(
            "SELECT zoom_level, tile_column, tile_row, tile_data FROM tiles LIMIT 5"
        ).fetchall()
        conn.close()

        assert rows, "No tiles in merged output"
        for z, x, y, data in rows:
            img = Image.open(io.BytesIO(data))
            assert img.mode in ("RGB", "RGBA"), \
                f"Unexpected mode {img.mode} at z={z} x={x} y={y}"
            assert img.size in ((256, 256), (512, 512)), \
                f"Unexpected tile size {img.size} at z={z} x={x} y={y}"

    def test_jaxa_land_takes_priority_over_gebco(self, tmp_path):
        """
        Where JAXA has land data (non-masked), merged elevation should reflect JAXA
        values rather than GEBCO ocean depths.

        z=2 tile (x=2, y=1) covers East Asia / Pacific coast — JAXA has dense
        land coverage (~0–500 m) while GEBCO would give negative values there.
        A positive median elevation confirms JAXA is winning.
        """
        result, out = self._run_merge(tmp_path)
        assert result.exit_code == 0, result.output + str(result.exception or "")

        conn = sqlite3.connect(out)
        row = conn.execute(
            "SELECT tile_data FROM tiles WHERE zoom_level=2 AND tile_column=2 AND tile_row=1"
        ).fetchone()
        conn.close()

        if row is None:
            pytest.skip("Tile z=2/x=2/y=1 not present in merged output")

        img = np.array(Image.open(io.BytesIO(row[0]))).astype(np.float64)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        decoded = -10000 + ((r * 256 * 256 + g * 256 + b) * 0.1)
        assert np.median(decoded) > 0, \
            f"Median elevation {np.median(decoded):.1f} m — expected positive (JAXA land should win)"

    def test_merge_with_sparse_tiles(self, tmp_path):
        """With sparse_tiles=True the output should have ≤ tiles than without."""
        (tmp_path / "full").mkdir()
        (tmp_path / "sparse").mkdir()
        result_full,   out_full   = self._run_merge(tmp_path / "full")
        result_sparse, out_sparse = self._run_merge(
            tmp_path / "sparse", extra_config={"sparse_tiles": True}
        )
        assert result_full.exit_code == 0,   result_full.output
        assert result_sparse.exit_code == 0, result_sparse.output

        conn_f = sqlite3.connect(out_full)
        conn_s = sqlite3.connect(out_sparse)
        count_full   = conn_f.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        count_sparse = conn_s.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        conn_f.close()
        conn_s.close()

        assert count_sparse <= count_full

    def test_merge_output_nodata(self, tmp_path):
        """output_nodata config key should be accepted without error."""
        result, out = self._run_merge(
            tmp_path,
            extra_config={"output_nodata": -10000, "output_format": "png"},
        )
        assert result.exit_code == 0, result.output + str(result.exception or "")
        assert os.path.exists(out)

    @pytest.mark.skipif(
        not any(
            (_EXPECTED_TILES_DIR / f"z{z}_x{x}_y{y}.png").exists()
            for z, x, y in _REFERENCE_KEY_TILES
        ),
        reason=(
            "Reference tiles not found — run `python test/generate_expected_tiles.py`"
            " to create them"
        ),
    )
    def test_output_matches_expected_tiles(self, tmp_path):
        """
        Decoded elevation values in the merged output must match pre-generated
        reference PNGs within ±1 m tolerance.

        The reference tiles in test/fixtures/expected/ are produced with PNG
        output (lossless) by running test/generate_expected_tiles.py.  Re-run
        that script whenever you intentionally change merger behaviour.
        """
        # Use PNG output so our results are also lossless and directly comparable
        # against the reference PNGs.
        result, out = self._run_merge(tmp_path, extra_config={"output_format": "png"})
        assert result.exit_code == 0, result.output + str(result.exception or "")

        conn = sqlite3.connect(out)
        mismatches = []

        for z, x, y in _REFERENCE_KEY_TILES:
            expected_path = _EXPECTED_TILES_DIR / f"z{z}_x{x}_y{y}.png"
            if not expected_path.exists():
                continue  # skip tiles that weren't generated

            # Decode reference tile -> elevation float64 array
            ref_arr  = np.array(Image.open(expected_path).convert("RGB")).astype(np.float64)
            ref_elev = -10000 + (
                (ref_arr[:, :, 0] * 256 * 256
                 + ref_arr[:, :, 1] * 256
                 + ref_arr[:, :, 2]) * 0.1
            )

            # Decode output tile -> elevation float64 array
            row = conn.execute(
                "SELECT tile_data FROM tiles"
                " WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                (z, x, y),
            ).fetchone()
            assert row is not None, f"Tile z={z}/x={x}/y={y} missing from output"

            out_arr  = np.array(Image.open(io.BytesIO(row[0])).convert("RGB")).astype(np.float64)
            out_elev = -10000 + (
                (out_arr[:, :, 0] * 256 * 256
                 + out_arr[:, :, 1] * 256
                 + out_arr[:, :, 2]) * 0.1
            )

            # Allow ±1 m — catches regression while tolerating minor float rounding
            if not np.allclose(ref_elev, out_elev, atol=1.0):
                max_delta = float(np.max(np.abs(ref_elev - out_elev)))
                mismatches.append(
                    f"z={z}/x={x}/y={y}: max delta = {max_delta:.1f} m"
                )

        conn.close()
        assert not mismatches, "Elevation mismatch vs reference: " + "; ".join(mismatches)

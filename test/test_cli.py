from click.testing import CliRunner
import rasterio as rio
import numpy as np
from rio_rgbify.scripts.cli import rgbify

from tempfile import mkdtemp
from shutil import rmtree
import os

class TestingDir:
    def __init__(self):
        self.tmpdir = mkdtemp()
    def __enter__(self):
        return self
    def __exit__(self, a, b, c):
        rmtree(self.tmpdir)
    def mkpath(self, filename):
        return os.path.join(self.tmpdir, filename)


def test_cli_good_elev():
    in_elev_src = 'test/fixtures/elev.tif'
    expected_src = 'test/expected/elev-rgb.tif'
    with TestingDir() as tmpdir:
        out_rgb_src = tmpdir.mkpath('rgb.tif')

        runner = CliRunner()
        result = runner.invoke(rgbify, [in_elev_src, out_rgb_src, '--interval', 0.001, '--base-val', -100])

        assert result.exit_code == 0

        with rio.open(out_rgb_src) as created:
            with rio.open(expected_src) as expected:
                assert np.array_equal(created.read(), expected.read())


def test_cli_fail_elev():
    in_elev_src = 'test/fixtures/elev.tif'
    expected_src = 'test/expected/elev-rgb.tif'
    with TestingDir() as tmpdir:
        out_rgb_src = tmpdir.mkpath('rgb.tif')

        runner = CliRunner()
        result = runner.invoke(rgbify, [in_elev_src, out_rgb_src, '--interval', 0.00000001, '--base-val', -100])

        assert result.exit_code == -1

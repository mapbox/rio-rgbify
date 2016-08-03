from click.testing import CliRunner
import rasterio as rio
import numpy as np
from rio_rgbify.scripts.cli import rgbify
import click

from tempfile import mkdtemp
from shutil import rmtree
import os

from raster_tester.compare import affaux, upsample_array

class TestingDir:
    def __init__(self):
        self.tmpdir = mkdtemp()
    def __enter__(self):
        return self
    def __exit__(self, a, b, c):
        rmtree(self.tmpdir)
    def mkpath(self, filename):
        return os.path.join(self.tmpdir, filename)

def flex_compare(r1, r2, thresh=10):
    upsample = 4
    r1 = r1[::upsample]
    r2 = r2[::upsample]
    toAff, frAff = affaux(upsample)
    r1 = upsample_array(r1, upsample, frAff, toAff)
    r2 = upsample_array(r2, upsample, frAff, toAff)
    tdiff = np.abs(r1.astype(np.float64) - r2.astype(np.float64))

    click.echo('{0} values exceed the threshold difference with a max variance of {1}'.format(
        np.sum(tdiff > thresh), tdiff.max()), err=True)

    return not np.any(tdiff > thresh)


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
                carr = created.read()
                earr = expected.read()
                for a, b in zip(carr, earr):
                    assert flex_compare(a, b)


def test_cli_fail_elev():
    in_elev_src = 'test/fixtures/elev.tif'
    expected_src = 'test/expected/elev-rgb.tif'
    with TestingDir() as tmpdir:
        out_rgb_src = tmpdir.mkpath('rgb.tif')

        runner = CliRunner()
        result = runner.invoke(rgbify, [in_elev_src, out_rgb_src, '--interval', 0.00000001, '--base-val', -100])

        assert result.exit_code == -1

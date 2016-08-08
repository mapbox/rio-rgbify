from __future__ import with_statement

import traceback, itertools, sys, json, math, os

import click
import mercantile
import rasterio
import pyproj
import numpy as np
import sqlite3
from multiprocessing import Pool
from rasterio._io import virtual_file_to_buffer

from io import BytesIO
from PIL import Image

from rasterio import transform
from rasterio.warp import reproject, RESAMPLING
from rasterio.warp import transform as xform

from rio_rgbify.encoders import data_to_rgb

buffer = bytes if sys.version_info > (3,) else buffer
 
work_func = None
global_args = None
src = None


def _main_worker(inpath, g_work_func, g_args):
    """"""
    global work_func
    global global_args
    global src
    work_func = g_work_func
    global_args = g_args

    try:
        src = rasterio.open(inpath)
    except:
        return


def _tile_range(min_tile, max_tile):
    min_x, min_y, _ = min_tile
    max_x, max_y, _ = max_tile

    return itertools.product(range(min_x, max_x + 1), range(min_y, max_y + 1))

def _webp_writer(data, _):
    with BytesIO() as f:
        im = Image.fromarray(np.rollaxis(data, 0, 3))
        im.save(f, format='webp', lossless=True)

        return f.getvalue()

def _file_writer(data, dst_transform):
    kwargs = global_args['kwargs'].copy()
    kwargs['affine'] = dst_transform

    with rasterio.open('/vsimem/tileimg', 'w', **kwargs) as dst:
        dst.write(data)

    contents = bytearray(virtual_file_to_buffer('/vsimem/tileimg'))

    return contents


def _tile_worker(tile):
    x, y, z = tile

    bounds = [c for i in (mercantile.xy(*mercantile.ul(x, y + 1, z)),
            mercantile.xy(*mercantile.ul(x + 1, y, z))) for c in i]

    srcwindow = (
        list(src.index(bounds[0], bounds[3])),
        list(src.index(bounds[2], bounds[1]))
        )

    srcwindow[1][0] += 1
    srcwindow[1][1] += 1

    toaffine = transform.from_bounds(*bounds + [512, 512])

    fromaffine = src.window_transform(srcwindow)

    out = np.empty((512, 512), dtype=src.meta['dtype'])

    reproject(
        rasterio.band(src, 1), out,
        src_transform=fromaffine,
        dst_transform=toaffine,
        dst_crs="init='epsg:3857'",
        resampling=RESAMPLING.bilinear)

    out = data_to_rgb(out, global_args['base_val'], global_args['interval'])

    return tile, global_args['writer_func'](out, toaffine)


def _make_tiles(bbox, minz, maxz):
    proj = pyproj.Proj(init='epsg:3857')
    xs, ys = proj([bbox[0], bbox[2]], [bbox[1], bbox[3]], inverse=True)

    for z in range(minz, maxz + 1):
        for x, y in _tile_range(
            mercantile.tile(min(xs), max(ys), z),
            mercantile.tile(max(xs), min(ys), z)):

            yield [x, y, z]


class RGBTiler:
    def __init__(self, inpath, outpath, min_z, max_z, **kwargs):
        self.run_function = _tile_worker
        self.inpath = inpath
        self.outpath = outpath
        self.min_z = min_z
        self.max_z = max_z

        if not 'interval' in kwargs:
            kwargs['interval'] = 1

        if not 'base_val' in kwargs:
            kwargs['base_val'] = 0

        if not 'format' in kwargs:
            writer_func = _file_writer
        elif kwargs['format'].lower() == 'png':
            writer_func = _file_writer
        elif kwargs['format'].lower() == 'webp':
            writer_func = _webp_writer
        else:
            raise ValueError('{0} is not a supported filetype!'.format(kwargs['format']))

        # global kwargs not used if output  is webp
        self.global_args = {
            'kwargs':  {
                'driver': 'PNG',
                'dtype': 'uint8',
                'height': 512,
                'width': 512,
                'count': 3,
                'crs': 'EPSG:3857'
            },
            'base_val': kwargs['base_val'],
            'interval': kwargs['interval'],
            'writer_func': writer_func
        }

    def __enter__(self):
        return self
    def __exit__(self, ext_t, ext_v, trace):
        if ext_t:
            traceback.print_exc()


    def run(self, processes=4):
        with rasterio.open(self.inpath) as src:
            bbox = list(src.bounds)

        (west, east), (south, north) = xform(
                src.crs, 'EPSG:4326', src.bounds[::2], src.bounds[1::2])


        if os.path.exists(self.outpath):
            os.unlink(self.outpath)

        conn = sqlite3.connect(self.outpath)
        cur = conn.cursor()

        cur.execute(
            "CREATE TABLE tiles "
            "(zoom_level integer, tile_column integer, "
            "tile_row integer, tile_data blob);")
        cur.execute(
            "CREATE TABLE metadata (name text, value text);")


        conn.commit()

        if processes == 1:
            self.pool = MockTub(main_worker, (self.inpath, self.run_function, self.global_args))
        else:
            self.pool = Pool(processes, _main_worker, (self.inpath, self.run_function, self.global_args))

        tiles = _make_tiles(bbox, self.min_z, self.max_z)

        for tile, contents in self.pool.imap_unordered(self.run_function,
                                                       tiles):
            x, y, z = tile

            # mbtiles use inverse y indexing
            tiley = int(math.pow(2, z)) - y - 1

            cur.execute(
                "INSERT INTO tiles "
                "(zoom_level, tile_column, tile_row, tile_data) "
                "VALUES (?, ?, ?, ?);",
                (z, x, tiley, buffer(contents)))

            conn.commit()

        conn.close()

        self.pool.close()
        self.pool.join()

        return

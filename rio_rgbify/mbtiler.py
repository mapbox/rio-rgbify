from __future__ import with_statement
from __future__ import division

import traceback, itertools, sys, json, math, os

import click
import mercantile
import rasterio
import numpy as np
import sqlite3
from multiprocessing import Pool
from rasterio._io import virtual_file_to_buffer
from riomucho.single_process_pool import MockTub

from io import BytesIO
from PIL import Image

from rasterio import transform
from rasterio.warp import reproject, RESAMPLING, transform_bounds

from rio_rgbify.encoders import data_to_rgb

buffer = bytes if sys.version_info > (3,) else buffer
 
work_func = None
global_args = None
src = None


def _main_worker(inpath, g_work_func, g_args):
    """
    Util for setting global vars w/ a Pool
    """
    global work_func
    global global_args
    global src
    work_func = g_work_func
    global_args = g_args

    src = rasterio.open(inpath)


def _encode_as_webp(data, profile=None, affine=None):
    """
    Uses BytesIO + PIL to encode a (3, 512, 512)
    array into a webp bytearray.

    Parameters
    -----------
    data: ndarray
        (3 x 512 x 512) uint8 RGB array
    profile: None
        ignored
    affine: None
        ignored

    Returns
    --------
    contents: bytearray
        webp-encoded bytearray of the provided input data
    """
    with BytesIO() as f:
        im = Image.fromarray(np.rollaxis(data, 0, 3))
        im.save(f, format='webp', lossless=True)

        return f.getvalue()


def _encode_as_png(data, profile, dst_transform):
    """
    Uses rasterio's virtual file system to encode a (3, 512, 512)
    array as a png-encoded bytearray.

    Parameters
    -----------
    data: ndarray
        (3 x 512 x 512) uint8 RGB array
    profile: dictionary
        dictionary of kwargs for png writing
    affine: Affine
        affine transform for output tile

    Returns
    --------
    contents: bytearray
        png-encoded bytearray of the provided input data
    """
    profile['affine'] = dst_transform

    with rasterio.open('/vsimem/tileimg', 'w', **profile) as dst:
        dst.write(data)

    contents = bytearray(virtual_file_to_buffer('/vsimem/tileimg'))

    return contents


def _tile_worker(tile):
    """
    For each tile, and given an open rasterio src, plus a`global_args` dictionary
    with attributes of `base_val`, `interval`, and a `writer_func`,
    warp a continous single band raster to a 512 x 512 mercator tile,
    then encode this tile into RGB.

    Parameters
    -----------
    tile: list
        [x, y, z] indices of tile

    Returns
    --------
    tile, buffer
        tuple with the input tile, and a bytearray with the data encoded into
        the format created in the `writer_func`

    """
    x, y, z = tile

    bounds = [c for i in (mercantile.xy(*mercantile.ul(x, y + 1, z)),
            mercantile.xy(*mercantile.ul(x + 1, y, z))) for c in i]

    toaffine = transform.from_bounds(*bounds + [512, 512])

    out = np.empty((512, 512), dtype=src.meta['dtype'])

    reproject(
        rasterio.band(src, 1), out,
        dst_transform=toaffine,
        dst_crs="init='epsg:3857'",
        resampling=RESAMPLING.bilinear)

    out = data_to_rgb(out, global_args['base_val'], global_args['interval'])

    return tile, global_args['writer_func'](out, global_args['kwargs'].copy(), toaffine)


def _tile_range(min_tile, max_tile):
    """
    Given a min and max tile, return an iterator of
    all combinations of this tile range

    Parameters
    -----------
    min_tile: list
        [x, y, z] of minimun tile
    max_tile:
        [x, y, z] of minimun tile

    Returns
    --------
    tiles: iterator
        iterator of [x, y, z] tiles
    """
    min_x, min_y, _ = min_tile
    max_x, max_y, _ = max_tile

    return itertools.product(range(min_x, max_x + 1), range(min_y, max_y + 1))


def _make_tiles(bbox, src_crs, minz, maxz):
    '''
    Given a bounding box, zoom range, and source crs,
    find all tiles that would intersect

    Parameters
    -----------
    bbox: list
        [w, s, e, n] bounds
    src_crs: str
        the source crs of the input bbox
    minz: int
        minumum zoom to find tiles for
    maxz: int
        maximum zoom to find tiles for

    Returns
    --------
    tiles: generator
        generator of [x, y, z] tiles that intersect
        the provided bounding box
    '''
    w, s, e, n = transform_bounds(*[src_crs, 'epsg:4326'] + bbox, densify_pts=0)

    EPSILON = 1.0e-10

    w += EPSILON
    s += EPSILON
    e -= EPSILON
    n -= EPSILON

    for z in range(minz, maxz + 1):
        for x, y in _tile_range(
            mercantile.tile(w, n, z),
            mercantile.tile(e, s, z)):

            yield [x, y, z]


class RGBTiler:
    '''
    Takes continous source data of an arbitrary bit depth and encodes it
    in parallel into RGB tiles in an MBTiles file. Provided with a context manager:
    ```
    with RGBTiler(inpath, outpath, min_z, max_x, **kwargs) as tiler:
        tiler.run(processes)
    ```

    Parameters
    -----------
    inpath: string
        filepath of the source file to read and encode
    outpath: string
        filepath of the output `mbtiles`
    min_z: int
        minimum zoom level to tile
    max_z: int
        maximum zoom level to tile

    Keyword Arguments
    ------------------
    baseval: float
        the base value of the RGB numbering system.
        (will be treated as zero for this encoding)
        Default=0
    interval: float
        the interval at which to encode
        Default=1
    format: str
        output tile image format (png or webp)
        Default=png

    Returns
    --------
    None

    '''
    def __init__(self, inpath, outpath, min_z, max_z, interval=1, base_val=0, **kwargs):
        self.run_function = _tile_worker
        self.inpath = inpath
        self.outpath = outpath
        self.min_z = min_z
        self.max_z = max_z

        if not 'format' in kwargs:
            writer_func = _encode_as_png
        elif kwargs['format'].lower() == 'png':
            writer_func = _encode_as_png
        elif kwargs['format'].lower() == 'webp':
            writer_func = _encode_as_webp
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
            'base_val': base_val,
            'interval': interval,
            'writer_func': writer_func
        }

    def __enter__(self):
        return self
    def __exit__(self, ext_t, ext_v, trace):
        if ext_t:
            traceback.print_exc()


    def run(self, processes=4):
        '''
        Warp, encode, and tile
        '''

        # get the bounding box + crs of the file to tile
        with rasterio.open(self.inpath) as src:
            bbox = list(src.bounds)
            src_crs = src.crs

        # remove the output filepath if it exists
        if os.path.exists(self.outpath):
            os.unlink(self.outpath)

        # create a connection to the mbtiles file
        conn = sqlite3.connect(self.outpath)
        cur = conn.cursor()

        # create the tiles table
        cur.execute(
            "CREATE TABLE tiles "
            "(zoom_level integer, tile_column integer, "
            "tile_row integer, tile_data blob);")
        # create empty metadata
        cur.execute(
            "CREATE TABLE metadata (name text, value text);")

        conn.commit()

        if processes == 1:
            # use mock pool for profiling / debugging
            self.pool = MockTub(_main_worker, (self.inpath, self.run_function, self.global_args))
        else:
            self.pool = Pool(processes, _main_worker, (self.inpath, self.run_function, self.global_args))

        # generator of tiles to make
        tiles = _make_tiles(bbox, src_crs, self.min_z, self.max_z)


        for tile, contents in self.pool.imap_unordered(self.run_function,
                                                       tiles):
            x, y, z = tile

            # mbtiles use inverse y indexing
            tiley = int(math.pow(2, z)) - y - 1

            # insert tile object
            cur.execute(
                "INSERT INTO tiles "
                "(zoom_level, tile_column, tile_row, tile_data) "
                "VALUES (?, ?, ?, ?);",
                (z, x, tiley, buffer(contents)))

            conn.commit()

        conn.close()

        self.pool.close()
        self.pool.join()

        return None

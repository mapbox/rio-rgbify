import click

import rasterio as rio
import numpy as np
from riomucho import RioMucho
from rasterio.rio.options import creation_options

from rio_rgbify.encoders import data_to_rgb
from rio_rgbify.mbtiler import RGBTiler

def _rgb_worker(data, window, ij, g_args):
    return data_to_rgb(data[0][g_args['bidx'] - 1],
        g_args['base_val'],
        g_args['interval'])

@click.command('rgbify')
@click.argument('src_path', type=click.Path(exists=True))
@click.argument('dst_path', type=click.Path(exists=False))
@click.option('--base-val', '-b', type=float, default=0,
    help='The base value of which to base the output encoding on [DEFAULT=0]')
@click.option('--interval', '-i', type=float, default=1,
    help='Describes the precision of the output, by incrementing interval [DEFAULT=1]')
@click.option('--bidx', type=int, default=1,
    help='Band to encode [DEFAULT=1]')
@click.option('--max-z', type=int, default=None,
    help="Maximum zoom to tile (.mbtiles output only)")
@click.option('--min-z', type=int, default=None,
    help="Minimum zoom to tile (.mbtiles output only)")
@click.option('--format', type=click.Choice(['png', 'webp']), default='png',
    help="Output tile format (.mbtiles output only)")
@click.option('--workers', '-j', type=int, default=4,
    help='Workers to run [DEFAULT=4]')
@click.option('--verbose', '-v', is_flag=True, default=False)
@click.pass_context
@creation_options
def rgbify(ctx, src_path, dst_path, base_val, interval, bidx, max_z, min_z, format, workers, verbose, creation_options):
    if dst_path.split('.')[-1].lower() == 'tif':
        with rio.open(src_path) as src:
            meta = src.profile.copy()

        meta.update(
            count=3,
            dtype=np.uint8
        )

        for c in creation_options:
            meta[c] = creation_options[c]

        gargs = {
            'interval': interval,
            'base_val': base_val,
            'bidx': bidx
        }

        with RioMucho([src_path], dst_path, _rgb_worker,
            options=meta,
            global_args=gargs) as rm:

            rm.run(workers)

    elif dst_path.split('.')[-1].lower() == 'mbtiles':

        if min_z == None or max_z == None:
            raise ValueError('Zoom range must be provided for mbtile output')

        if max_z < min_z:
            raise ValueError('Max zoom {0} must be greater than min zoom {1}'.format(max_z, min_z))

        with RGBTiler(src_path, dst_path,
                      interval=interval,
                      base_val=base_val,
                      format=format,
                      max_z=max_z, min_z=min_z) as tiler:

            tiler.run(workers)

    else:
        raise ValueError('{} output filetype not supported'.format(dst_path.split('.')[-1]))

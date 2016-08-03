import click

import rasterio as rio
import numpy as np
from riomucho import RioMucho
from rasterio.rio.options import creation_options

from rio_rgbify.encoders import data_to_rgb

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
    help='Describes the precision of the output,by incrementing interval [DEFAULT=1]')
@click.option('--bidx', type=int, default=1,
    help='Band to encode [DEFAULT=1]')
@click.option('--workers', '-j', type=int, default=4,
    help='Workers to run [DEFAULT=4]')
@click.option('--verbose', '-v', is_flag=True, default=False)
@click.pass_context
@creation_options
def rgbify(ctx, src_path, dst_path, base_val, interval, bidx, workers, verbose, creation_options):
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

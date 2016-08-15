# rio-rgbify
Encode arbitrary bit depth rasters in psuedo base-256 as RGB

[![Build Status](https://travis-ci.org/mapbox/rio-rgbify.svg)](https://travis-ci.org/mapbox/rio-rgbify)[![Coverage Status](https://coveralls.io/repos/github/mapbox/rio-rgbify/badge.svg?branch=its-a-setup)](https://coveralls.io/github/mapbox/rio-rgbify)

## Installation

### From PyPi
```
pip install rio-rgbify
```
### Development
```
git clone git@github.com:mapbox/rio-rgbify.git

cd rio-rgbify

pip install -e '.[test]'

```

## CLI usage

- Input can be any raster readable by `rasterio`
- Output can be any raster format writable by `rasterio` OR
- To create tiles _directly_ from data (recommended), output to an `.mbtiles`

```
Usage: rio rgbify [OPTIONS] SRC_PATH DST_PATH

Options:
  -b, --base-val FLOAT   The base value of which to base the output encoding
                         on [DEFAULT=0]
  -i, --interval FLOAT   Describes the precision of the output, by
                         incrementing interval [DEFAULT=1]
  --bidx INTEGER         Band to encode [DEFAULT=1]
  --max-z INTEGER        Maximum zoom to tile (.mbtiles output only)
  --bounding-tile TEXT   Bounding tile '[{x}, {y}, {z}]' to limit output tiles
                         (.mbtiles output only)
  --min-z INTEGER        Minimum zoom to tile (.mbtiles output only)
  --format [png|webp]    Output tile format (.mbtiles output only)
  -j, --workers INTEGER  Workers to run [DEFAULT=4]
  -v, --verbose
  --co NAME=VALUE        Driver specific creation options.See the
                         documentation for the selected output driver for more
                         information.
  --help                 Show this message and exit.
```

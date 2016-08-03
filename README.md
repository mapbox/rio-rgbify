# rio-rgbify
Encoded arbitrary bit depth rasters in psuedo base-256 as RGB

## CLI usage

```
Usage: rio rgbify [OPTIONS] SRC_PATH DST_PATH

Options:
  -b, --base-val FLOAT
  -i, --interval FLOAT
  --bidx INTEGER
  -j, --workers INTEGER
  -v, --verbose
  --co NAME=VALUE        Driver specific creation options.See the
                         documentation for the selected output driver for more
                         information.
  --help                 Show this message and exit.
```
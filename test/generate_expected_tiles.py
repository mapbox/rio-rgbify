#!/usr/bin/env python3
"""Generate reference expected-output PNG tiles for the live merge tests.

Runs the merger against the committed GEBCO and JAXA fixture files, extracts
a small set of key output tiles as lossless PNG files, and writes them to
test/fixtures/expected/.  The comparison test (TestLiveMerge.test_output_matches_expected_tiles)
loads these PNGs, decodes to elevation values, and checks that new runs of the
merger produce the same results within 1 m tolerance.

Run this script whenever you intentionally change merger behaviour so that the
reference tiles stay in sync:

    python test/generate_expected_tiles.py
"""

import io
import json
import os
import sys
import sqlite3
import tempfile
import traceback
from pathlib import Path

from PIL import Image

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from click.testing import CliRunner
from rio_rgbify.scripts.cli import main_group as cli

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR   = Path(__file__).parent / "fixtures"
GEBCO_FIXTURE  = FIXTURES_DIR / "gebco_sample.mbtiles"
JAXA_FIXTURE   = FIXTURES_DIR / "jaxa_sample.mbtiles"
EXPECTED_DIR   = Path(__file__).parent / "expected"

# ---------------------------------------------------------------------------
# Key tiles to capture as reference output — (z, x, y, description).
#
#   z=0/x=0/y=0       global overview — always present
#   z=2/x=2/y=1       East Asia / Pacific coast — JAXA land wins over GEBCO depths
#   z=2/x=0/y=2       South Atlantic open ocean — GEBCO-only depths
# ---------------------------------------------------------------------------

KEY_TILES = [
    (0, 0, 0, "global_z0"),
    (2, 2, 1, "east_asia_z2"),
    (2, 0, 2, "south_atlantic_z2"),
]


def _decode_elevation(tile_bytes: bytes):
    """Decode mapbox-encoded RGB(A) tile bytes -> elevation float64 array."""
    img = Image.open(io.BytesIO(tile_bytes)).convert("RGB")
    arr = __import__("numpy").array(img).astype(__import__("numpy").float64)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return -10000 + ((r * 256 * 256 + g * 256 + b) * 0.1)


def main() -> int:
    if not GEBCO_FIXTURE.exists() or not JAXA_FIXTURE.exists():
        print("ERROR: Fixture files not found.")
        print("  Run `python test/download_fixtures.py` first.")
        return 1

    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        out      = os.path.join(tmp, "merged.mbtiles")
        cfg_path = os.path.join(tmp, "config.json")

        # Mirror the TestLiveMerge._run_merge config but force output_format=png
        # so the reference tiles are stored losslessly.
        cfg = {
            "output_type": "mbtiles",
            "sources": [
                # Bottom first: GEBCO bathymetry underneath, JAXA land on top.
                # Equivalent to the old JAXA-first listing under the inverted
                # priority master carried, so the reference tiles still match.
                {
                    "path": str(GEBCO_FIXTURE),
                    "encoding": "mapbox",
                    "mask_values": [-10000],
                },
                {
                    "path": str(JAXA_FIXTURE),
                    "encoding": "mapbox",
                    "mask_values": [-10000, 0, -1],
                },
            ],
            "output_path": out,
            "output_encoding": "mapbox",
            "output_format": "png",
            "resampling": "cubic",
            "min_zoom": 0,
            "max_zoom": 2,
        }

        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        print("Running merger (this may take ~60 seconds) ...")
        runner = CliRunner()
        result = runner.invoke(cli, ["merge", "--config", cfg_path, "-j", "1"])

        if result.exit_code != 0:
            print("ERROR: Merger failed:")
            print(result.output)
            if result.exception:
                traceback.print_exception(
                    type(result.exception),
                    result.exception,
                    result.exception.__traceback__,
                )
            return 1

        print("Extracting key tiles ...")
        conn = sqlite3.connect(out)
        saved = 0

        for z, x, y, desc in KEY_TILES:
            row = conn.execute(
                "SELECT tile_data FROM tiles"
                " WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                (z, x, y),
            ).fetchone()

            if row is None:
                print(f"  SKIP z={z}/x={x}/y={y} ({desc}) - tile not in output")
                continue

            fname = EXPECTED_DIR / f"z{z}_x{x}_y{y}.png"
            fname.write_bytes(row[0])
            saved += 1

            img   = Image.open(io.BytesIO(row[0]))
            elev  = _decode_elevation(row[0])
            import numpy as np
            print(
                f"  OK   z={z}/x={x}/y={y} ({desc})"
                f"  [{img.size[0]}x{img.size[1]}]"
                f"  median elev = {np.median(elev):.1f} m"
            )

        conn.close()

    print(f"\nDone. {saved} reference tiles written to {EXPECTED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""METIS Calibration Module - Entry point.

Usage:
    python -m metis.calibrate --config configs/config_airbnb.yaml --profile balanced
    python -m metis.calibrate --config configs/config_telco.yaml --iterations 5

This delegates to the unified CLI at metis.interface.cli.
"""

import sys

from metis.interface.cli import main

if __name__ == "__main__":
    sys.exit(main(["calibrate"] + sys.argv[1:]))

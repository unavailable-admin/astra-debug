"""Packaged calibration and caller-owned runtime paths."""

from pathlib import Path

ROOT = Path.cwd()
CONFIG = Path(__file__).parent / "config"
DEFAULT_URI = "ws://10.19.4.253:8081"

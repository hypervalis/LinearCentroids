#!/usr/bin/env python3
"""Verify Python dependencies and optional paper submodule."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.centroid_adapter import PAPER_CENTROIDS_FILE, has_third_party_centroid_code


def check_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Repo root: {ROOT}")

    required = ["numpy", "scipy", "sklearn", "plotly", "torch", "transformers"]
    missing = [m for m in required if not check_import(m)]
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        sys.exit(1)
    print("Core packages: OK")

    if has_third_party_centroid_code():
        print(f"Paper centroid code: found at {PAPER_CENTROIDS_FILE}")
    else:
        print("Paper centroid code: not vendored (autograd fallback in lce.centroid_adapter)")
        print("  git submodule add https://github.com/ThomasWalker1/LinearCentroidsHypothesis.git \\")
        print("      third_party/LinearCentroidsHypothesis")
        print("  git submodule update --init --recursive")


if __name__ == "__main__":
    main()

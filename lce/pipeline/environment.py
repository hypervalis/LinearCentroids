"""Environment and dependency checks."""

from __future__ import annotations

import importlib
import sys

from lce.centroid_adapter import PAPER_CENTROIDS_FILE, has_third_party_centroid_code
from lce.pipeline.paths import REPO_ROOT


def check_environment() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Repo root: {REPO_ROOT}")

    required = ["numpy", "scipy", "sklearn", "plotly", "torch", "transformers"]
    missing = [m for m in required if not _check_import(m)]
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return 1
    print("Core packages: OK")

    if has_third_party_centroid_code():
        print(f"Paper centroid code: found at {PAPER_CENTROIDS_FILE}")
    else:
        print("Paper centroid code: not vendored (autograd fallback in lce.centroid_adapter)")
        print("  git submodule add https://github.com/ThomasWalker1/LinearCentroidsHypothesis.git \\")
        print("      third_party/LinearCentroidsHypothesis")
        print("  git submodule update --init --recursive")
    return 0


def _check_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False

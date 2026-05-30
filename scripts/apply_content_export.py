#!/usr/bin/env python3
"""Apply a content-editor Export HTML JSON file to docs/index.html."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.site_content import apply_page_content_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write exported page-content JSON into docs/index.html."
    )
    parser.add_argument(
        "export_json",
        type=Path,
        help="Path to JSON from the editor's Export HTML button",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "docs" / "index.html",
        help="Target index.html (default: docs/index.html)",
    )
    args = parser.parse_args()

    raw = args.export_json.read_text(encoding="utf-8")
    path = apply_page_content_json(raw, index_path=args.index)
    print(f"Updated {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate month prompt JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT
import sys

sys.path.insert(0, str(ROOT))

from lce.concepts import DEFAULT_MONTH_TEMPLATES, MONTHS, expand_prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "prompts" / "months.json",
    )
    args = parser.parse_args()

    records = expand_prompts(MONTHS, DEFAULT_MONTH_TEMPLATES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(records)} prompts to {args.output.resolve()}")


if __name__ == "__main__":
    main()

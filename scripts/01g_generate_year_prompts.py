#!/usr/bin/env python3
"""Generate year prompt JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.concepts import YEAR_PROMPT_FAMILIES, all_year_labels, write_year_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Write year prompts as JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "prompts" / "years_neutral.jsonl",
    )
    parser.add_argument(
        "--prompt-family",
        choices=tuple(YEAR_PROMPT_FAMILIES.keys()),
        default="neutral",
    )
    args = parser.parse_args()

    path = write_year_prompts(args.output, prompt_family=args.prompt_family)
    n_years = len(all_year_labels())
    n_templates = len(YEAR_PROMPT_FAMILIES[args.prompt_family])
    print(f"Wrote {n_years * n_templates} prompts ({n_years} years) to {path.resolve()}")


if __name__ == "__main__":
    main()

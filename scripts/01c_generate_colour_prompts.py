#!/usr/bin/env python3
"""Generate colour prompt JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.concepts import COLOUR_PROMPT_FAMILIES, all_colour_labels, write_colour_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Write colour prompts as JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "prompts" / "colours_neutral.jsonl",
    )
    parser.add_argument(
        "--prompt-family",
        choices=tuple(COLOUR_PROMPT_FAMILIES.keys()),
        default="neutral",
    )
    args = parser.parse_args()

    path = write_colour_prompts(args.output, prompt_family=args.prompt_family)
    n_colours = len(all_colour_labels())
    n_templates = len(COLOUR_PROMPT_FAMILIES[args.prompt_family])
    print(f"Wrote {n_colours * n_templates} prompts ({n_colours} colours) to {path.resolve()}")


if __name__ == "__main__":
    main()

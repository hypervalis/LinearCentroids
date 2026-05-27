#!/usr/bin/env python3
"""Generate discipline prompt JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.concepts import DISCIPLINE_PROMPT_FAMILIES, all_discipline_labels, write_discipline_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Write discipline prompts as JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "prompts" / "disciplines_neutral.jsonl",
    )
    parser.add_argument(
        "--prompt-family",
        choices=tuple(DISCIPLINE_PROMPT_FAMILIES.keys()),
        default="neutral",
    )
    args = parser.parse_args()

    path = write_discipline_prompts(args.output, prompt_family=args.prompt_family)
    n_disciplines = len(all_discipline_labels())
    n_templates = len(DISCIPLINE_PROMPT_FAMILIES[args.prompt_family])
    print(
        f"Wrote {n_disciplines * n_templates} prompts ({n_disciplines} disciplines) "
        f"to {path.resolve()}"
    )


if __name__ == "__main__":
    main()

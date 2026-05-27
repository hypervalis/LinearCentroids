#!/usr/bin/env python3
"""Generate whword prompt JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.concepts import WHWORD_PROMPT_FAMILIES, all_whword_labels, write_whword_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Write whword prompts as JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "prompts" / "whwords_neutral.jsonl",
    )
    parser.add_argument(
        "--prompt-family",
        choices=tuple(WHWORD_PROMPT_FAMILIES.keys()),
        default="neutral",
    )
    args = parser.parse_args()

    path = write_whword_prompts(args.output, prompt_family=args.prompt_family)
    n_whwords = len(all_whword_labels())
    n_templates = len(WHWORD_PROMPT_FAMILIES[args.prompt_family])
    print(
        f"Wrote {n_whwords * n_templates} prompts ({n_whwords} whwords) "
        f"to {path.resolve()}"
    )


if __name__ == "__main__":
    main()

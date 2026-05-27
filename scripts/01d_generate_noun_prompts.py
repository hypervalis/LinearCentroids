#!/usr/bin/env python3
"""Generate noun prompt JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.concepts import NOUN_PROMPT_FAMILIES, all_noun_labels, write_noun_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Write noun prompts as JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "prompts" / "nouns_neutral.jsonl",
    )
    parser.add_argument(
        "--prompt-family",
        choices=tuple(NOUN_PROMPT_FAMILIES.keys()),
        default="neutral",
    )
    args = parser.parse_args()

    path = write_noun_prompts(args.output, prompt_family=args.prompt_family)
    n_nouns = len(all_noun_labels())
    n_templates = len(NOUN_PROMPT_FAMILIES[args.prompt_family])
    print(f"Wrote {n_nouns * n_templates} prompts ({n_nouns} nouns) to {path.resolve()}")


if __name__ == "__main__":
    main()

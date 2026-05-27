#!/usr/bin/env python3
"""Generate language prompt JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.concepts import LANGUAGE_PROMPT_FAMILIES, all_language_labels, write_language_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Write language prompts as JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "prompts" / "languages_neutral.jsonl",
    )
    parser.add_argument(
        "--prompt-family",
        choices=tuple(LANGUAGE_PROMPT_FAMILIES.keys()),
        default="neutral",
    )
    args = parser.parse_args()

    path = write_language_prompts(args.output, prompt_family=args.prompt_family)
    n_languages = len(all_language_labels())
    n_templates = len(LANGUAGE_PROMPT_FAMILIES[args.prompt_family])
    print(
        f"Wrote {n_languages * n_templates} prompts ({n_languages} languages) "
        f"to {path.resolve()}"
    )


if __name__ == "__main__":
    main()

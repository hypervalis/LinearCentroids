#!/usr/bin/env python3
"""Generate stopword prompt JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.concepts import STOPWORD_PROMPT_FAMILIES, all_stopword_labels, write_stopword_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Write stopword prompts as JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "prompts" / "stopwords_neutral.jsonl",
    )
    parser.add_argument(
        "--prompt-family",
        choices=tuple(STOPWORD_PROMPT_FAMILIES.keys()),
        default="neutral",
    )
    args = parser.parse_args()

    path = write_stopword_prompts(args.output, prompt_family=args.prompt_family)
    n_stopwords = len(all_stopword_labels())
    n_templates = len(STOPWORD_PROMPT_FAMILIES[args.prompt_family])
    print(
        f"Wrote {n_stopwords * n_templates} prompts ({n_stopwords} stopwords) "
        f"to {path.resolve()}"
    )


if __name__ == "__main__":
    main()

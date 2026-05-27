#!/usr/bin/env python3
"""Generate animal prompt JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.concepts import ANIMAL_PROMPT_FAMILIES, all_animal_labels, write_animal_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Write animal prompts as JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "prompts" / "animals_neutral.jsonl",
    )
    parser.add_argument(
        "--prompt-family",
        choices=tuple(ANIMAL_PROMPT_FAMILIES.keys()),
        default="neutral",
    )
    args = parser.parse_args()

    path = write_animal_prompts(args.output, prompt_family=args.prompt_family)
    n_animals = len(all_animal_labels())
    n_templates = len(ANIMAL_PROMPT_FAMILIES[args.prompt_family])
    print(f"Wrote {n_animals * n_templates} prompts ({n_animals} animals) to {path.resolve()}")


if __name__ == "__main__":
    main()

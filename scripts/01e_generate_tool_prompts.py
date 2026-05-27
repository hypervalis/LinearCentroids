#!/usr/bin/env python3
"""Generate tool prompt JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.concepts import TOOL_PROMPT_FAMILIES, all_tool_labels, write_tool_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Write tool prompts as JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "prompts" / "tools_neutral.jsonl",
    )
    parser.add_argument(
        "--prompt-family",
        choices=tuple(TOOL_PROMPT_FAMILIES.keys()),
        default="neutral",
    )
    args = parser.parse_args()

    path = write_tool_prompts(args.output, prompt_family=args.prompt_family)
    n_tools = len(all_tool_labels())
    n_templates = len(TOOL_PROMPT_FAMILIES[args.prompt_family])
    print(f"Wrote {n_tools * n_templates} prompts ({n_tools} tools) to {path.resolve()}")


if __name__ == "__main__":
    main()

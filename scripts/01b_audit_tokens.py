#!/usr/bin/env python3
"""Audit that final tokens align with month concepts for a HuggingFace tokenizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.concepts import MONTHS, expand_prompts
from lce.token_audit import TokenAuditError, audit_prompt_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = expand_prompts(MONTHS)
    try:
        results = audit_prompt_records(tokenizer, records)
    except TokenAuditError as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)

    print(f"OK: {len(results)} prompts passed for model={args.model}")
    for r in results[:3]:
        print(f"  {r.concept!r}: final_index={r.final_token_index} token={r.final_token_text!r}")
    print("  ...")


if __name__ == "__main__":
    main()

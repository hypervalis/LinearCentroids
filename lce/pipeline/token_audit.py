"""Tokenizer audit for final-token prompt templates."""

from __future__ import annotations

import sys

from lce.concepts import MONTHS, expand_prompts
from lce.token_audit import TokenAuditError, audit_prompt_records


def audit_tokens(*, model_name: str = "distilgpt2") -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    records = expand_prompts(MONTHS)
    try:
        results = audit_prompt_records(tokenizer, records)
    except TokenAuditError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"OK: {len(results)} prompts passed for model={model_name}")
    for r in results[:3]:
        print(f"  {r.concept!r}: final_index={r.final_token_index} token={r.final_token_text!r}")
    print("  ...")
    return 0

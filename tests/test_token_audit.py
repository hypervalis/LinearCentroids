"""Tests for tokenizer audit."""

from __future__ import annotations

import pytest

from lce.concepts import MONTHS, expand_prompts
from lce.token_audit import TokenAuditError, verify_final_token_is_concept


class _FakeTokenizer:
    """Minimal tokenizer: one id per word, space-prefixed tokens like GPT-2."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        words = text.split()
        return [hash(w) % 10_000 for w in words]

    def decode(self, ids: list[int]) -> str:
        return f"tok_{ids[0]}"


def test_verify_rejects_text_not_ending_with_concept():
    tok = _FakeTokenizer()
    with pytest.raises(TokenAuditError, match="does not end"):
        verify_final_token_is_concept(tok, "The month is January extra", "January")


def test_audit_all_default_prompts_with_gpt2():
    pytest.importorskip("transformers")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("distilgpt2")
    except OSError as exc:
        pytest.skip(f"distilgpt2 tokenizer unavailable offline: {exc}")

    records = expand_prompts(MONTHS)
    for record in records:
        verify_final_token_is_concept(tok, record["text"], record["concept"])

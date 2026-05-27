"""Tokenizer audit for v1 final-token extraction."""

from __future__ import annotations

from dataclasses import dataclass


class TokenAuditError(ValueError):
    """Raised when the final token does not belong to the concept string."""


@dataclass(frozen=True)
class TokenAuditResult:
    text: str
    concept: str
    token_ids: tuple[int, ...]
    final_token_index: int
    final_token_id: int
    final_token_text: str


def _normalize_piece(text: str) -> str:
    return text.replace("Ġ", " ").strip().lower()


def verify_final_token_is_concept(
    tokenizer,
    text: str,
    concept: str,
    *,
    add_special_tokens: bool = False,
) -> TokenAuditResult:
    """
    Ensure the last token of ``text`` is part of the month/concept string.

    Raises TokenAuditError if the audit fails.
    Returns metadata including the final token index to use for extraction.
    """
    token_ids = tuple(
        int(i)
        for i in tokenizer.encode(text, add_special_tokens=add_special_tokens)
    )
    if not token_ids:
        raise TokenAuditError(f"empty tokenization for text={text!r}")

    final_index = len(token_ids) - 1
    final_id = token_ids[final_index]
    final_text = tokenizer.decode([final_id])

    concept_lower = concept.lower()
    text_lower = text.lower()
    if not text_lower.endswith(concept_lower):
        raise TokenAuditError(
            f"text does not end with concept surface form: text={text!r}, concept={concept!r}"
        )

    concept_ids = tuple(
        int(i)
        for i in tokenizer.encode(concept, add_special_tokens=add_special_tokens)
    )
    if not concept_ids:
        raise TokenAuditError(f"empty tokenization for concept={concept!r}")

    # Final token id(s) must match a suffix of concept tokenization (BPE-safe).
    matched = False
    for k in range(1, len(concept_ids) + 1):
        suffix = concept_ids[-k:]
        if token_ids[-k:] == suffix:
            matched = True
            break

    if not matched:
        # Fallback: decoded final token characters appear in concept word.
        final_norm = _normalize_piece(final_text)
        if final_norm and final_norm in concept_lower:
            matched = True

    if not matched:
        raise TokenAuditError(
            "final token is not part of concept tokenization: "
            f"text={text!r}, concept={concept!r}, final_token={final_text!r}, "
            f"token_ids={token_ids}, concept_ids={concept_ids}"
        )

    return TokenAuditResult(
        text=text,
        concept=concept,
        token_ids=token_ids,
        final_token_index=final_index,
        final_token_id=final_id,
        final_token_text=final_text,
    )


def _record_concept(record: dict) -> str:
    if record.get("concept") in ("colour", "noun", "tool", "animal", "year", "discipline", "whword", "stopword", "language"):
        return str(record["value"])
    return str(record["concept"])


def _record_text(record: dict) -> str:
    text = record.get("text") or record.get("prompt")
    if not text:
        raise KeyError("prompt record missing text/prompt field")
    return str(text)


def audit_prompt_records(
    tokenizer,
    records: list[dict[str, str | object]],
) -> list[TokenAuditResult]:
    """Audit all prompt records; raise on first failure."""
    results: list[TokenAuditResult] = []
    for record in records:
        results.append(
            verify_final_token_is_concept(
                tokenizer,
                _record_text(record),
                _record_concept(record),
            )
        )
    return results

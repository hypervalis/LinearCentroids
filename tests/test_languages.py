"""Tests for programming-language ordering experiment scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

from lce.concepts import (
    LANGUAGE_ORDERINGS,
    LANGUAGE_UNIVERSAL_SET,
    expand_language_prompts,
    validate_language_orderings,
    write_language_prompts,
)


def test_language_prompt_jsonl_schema(tmp_path: Path):
    path = write_language_prompts(
        tmp_path / "languages.jsonl",
        languages=("Python", "Rust"),
        prompt_family="neutral",
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 * 5
    record = json.loads(lines[0])
    assert record["concept"] == "language"
    assert record["value"] in ("Python", "Rust")
    assert record["prompt"].endswith(record["value"])
    assert not record["prompt"].endswith(".")


def test_all_orderings_use_same_twelve_languages():
    validate_language_orderings()
    for ordering in LANGUAGE_ORDERINGS.values():
        assert len(ordering) == len(LANGUAGE_UNIVERSAL_SET)
        assert set(ordering) == set(LANGUAGE_UNIVERSAL_SET)


def test_default_language_prompts_use_universal_set():
    records = expand_language_prompts()
    values = {str(r["value"]) for r in records}
    assert values == set(LANGUAGE_UNIVERSAL_SET)


def test_list_order_matches_canonical_sequence():
    assert LANGUAGE_ORDERINGS["list_order"] == LANGUAGE_UNIVERSAL_SET
    assert LANGUAGE_UNIVERSAL_SET[0] == "Python"
    assert len(LANGUAGE_UNIVERSAL_SET) == 12

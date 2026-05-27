"""Tests for question-word ordering experiment scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

from lce.concepts import (
    WHWORD_ORDERINGS,
    WHWORD_UNIVERSAL_SET,
    expand_whword_prompts,
    validate_whword_orderings,
    write_whword_prompts,
)


def test_whword_prompt_jsonl_schema(tmp_path: Path):
    path = write_whword_prompts(
        tmp_path / "whwords.jsonl",
        whwords=("Who", "What"),
        prompt_family="neutral",
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 * 5
    record = json.loads(lines[0])
    assert record["concept"] == "whword"
    assert record["value"] in ("Who", "What")
    assert record["prompt"].endswith(record["value"])
    assert not record["prompt"].endswith(".")


def test_all_orderings_use_same_six_whwords():
    validate_whword_orderings()
    for ordering in WHWORD_ORDERINGS.values():
        assert len(ordering) == len(WHWORD_UNIVERSAL_SET)
        assert set(ordering) == set(WHWORD_UNIVERSAL_SET)


def test_default_whword_prompts_use_universal_set():
    records = expand_whword_prompts()
    values = {str(r["value"]) for r in records}
    assert values == set(WHWORD_UNIVERSAL_SET)


def test_list_order_matches_user_sequence():
    assert WHWORD_ORDERINGS["list_order"] == WHWORD_UNIVERSAL_SET
    assert WHWORD_UNIVERSAL_SET == ("Who", "What", "Where", "When", "Why", "How")

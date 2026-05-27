"""Tests for discipline ordering experiment scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

from lce.concepts import (
    DISCIPLINE_ORDERINGS,
    DISCIPLINE_UNIVERSAL_SET,
    expand_discipline_prompts,
    validate_discipline_orderings,
    write_discipline_prompts,
)


def test_discipline_prompt_jsonl_schema(tmp_path: Path):
    path = write_discipline_prompts(
        tmp_path / "disciplines.jsonl",
        disciplines=("Physics", "Biology"),
        prompt_family="neutral",
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 * 5
    record = json.loads(lines[0])
    assert record["concept"] == "discipline"
    assert record["value"] in ("Physics", "Biology")
    assert record["prompt"].endswith(record["value"])
    assert not record["prompt"].endswith(".")


def test_all_orderings_use_same_ten_disciplines():
    validate_discipline_orderings()
    for ordering in DISCIPLINE_ORDERINGS.values():
        assert len(ordering) == len(DISCIPLINE_UNIVERSAL_SET)
        assert set(ordering) == set(DISCIPLINE_UNIVERSAL_SET)


def test_default_discipline_prompts_use_universal_set():
    records = expand_discipline_prompts()
    values = {str(r["value"]) for r in records}
    assert values == set(DISCIPLINE_UNIVERSAL_SET)


def test_list_order_matches_user_sequence():
    assert DISCIPLINE_ORDERINGS["list_order"] == DISCIPLINE_UNIVERSAL_SET
    assert DISCIPLINE_UNIVERSAL_SET[0] == "Physics"
    assert DISCIPLINE_UNIVERSAL_SET[-1] == "Tautology"

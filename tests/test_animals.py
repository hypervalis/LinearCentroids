"""Tests for animal ordering experiment scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

from lce.concepts import (
    ANIMAL_ORDERINGS,
    ANIMAL_UNIVERSAL_SET,
    expand_animal_prompts,
    validate_animal_orderings,
    write_animal_prompts,
)


def test_animal_prompt_jsonl_schema(tmp_path: Path):
    path = write_animal_prompts(tmp_path / "animals.jsonl", animals=("dog", "cat"), prompt_family="neutral")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 * 5
    record = json.loads(lines[0])
    assert record["concept"] == "animal"
    assert record["value"] in ("dog", "cat")
    assert record["prompt"].endswith(record["value"])
    assert not record["prompt"].endswith(".")


def test_all_orderings_use_same_twelve_animals():
    validate_animal_orderings()
    for ordering in ANIMAL_ORDERINGS.values():
        assert len(ordering) == len(ANIMAL_UNIVERSAL_SET)
        assert set(ordering) == set(ANIMAL_UNIVERSAL_SET)


def test_default_animal_prompts_use_universal_set():
    records = expand_animal_prompts()
    values = {str(r["value"]) for r in records}
    assert values == set(ANIMAL_UNIVERSAL_SET)


def test_list_order_matches_user_sequence():
    assert ANIMAL_ORDERINGS["list_order"] == ANIMAL_UNIVERSAL_SET

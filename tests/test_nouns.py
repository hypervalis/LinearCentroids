"""Tests for random noun ordering experiment scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

from lce.concepts import (
    NOUN_ORDERINGS,
    NOUN_UNIVERSAL_SET,
    expand_noun_prompts,
    validate_noun_orderings,
    write_noun_prompts,
)
from lce.topology import evaluate_ordering_with_baseline


def test_noun_prompt_jsonl_schema(tmp_path: Path):
    path = write_noun_prompts(tmp_path / "nouns.jsonl", nouns=("hammer", "river"), prompt_family="neutral")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 * 5
    record = json.loads(lines[0])
    assert record["concept"] == "noun"
    assert record["value"] in ("hammer", "river")
    assert record["prompt"].endswith(record["value"])
    assert not record["prompt"].endswith(".")


def test_all_orderings_use_same_seven_nouns():
    validate_noun_orderings()
    for ordering in NOUN_ORDERINGS.values():
        assert len(ordering) == len(NOUN_UNIVERSAL_SET)
        assert set(ordering) == set(NOUN_UNIVERSAL_SET)


def test_default_noun_prompts_use_universal_set():
    records = expand_noun_prompts()
    values = {str(r["value"]) for r in records}
    assert values == set(NOUN_UNIVERSAL_SET)


def test_list_order_is_not_alphabetical():
    assert NOUN_ORDERINGS["list_order"] != tuple(sorted(NOUN_UNIVERSAL_SET))

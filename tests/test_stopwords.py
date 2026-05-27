"""Tests for stopword ordering experiment scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

from lce.concepts import (
    STOPWORD_ORDERINGS,
    STOPWORD_UNIVERSAL_SET,
    expand_stopword_prompts,
    validate_stopword_orderings,
    write_stopword_prompts,
)


def test_stopword_prompt_jsonl_schema(tmp_path: Path):
    path = write_stopword_prompts(
        tmp_path / "stopwords.jsonl",
        stopwords=("the", "and"),
        prompt_family="neutral",
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 * 5
    record = json.loads(lines[0])
    assert record["concept"] == "stopword"
    assert record["value"] in ("the", "and")
    assert record["prompt"].endswith(record["value"])
    assert not record["prompt"].endswith(".")


def test_all_orderings_use_same_twelve_stopwords():
    validate_stopword_orderings()
    for ordering in STOPWORD_ORDERINGS.values():
        assert len(ordering) == len(STOPWORD_UNIVERSAL_SET)
        assert set(ordering) == set(STOPWORD_UNIVERSAL_SET)


def test_default_stopword_prompts_use_universal_set():
    records = expand_stopword_prompts()
    values = {str(r["value"]) for r in records}
    assert values == set(STOPWORD_UNIVERSAL_SET)


def test_list_order_matches_canonical_sequence():
    assert STOPWORD_ORDERINGS["list_order"] == STOPWORD_UNIVERSAL_SET
    assert STOPWORD_UNIVERSAL_SET[0] == "the"
    assert len(STOPWORD_UNIVERSAL_SET) == 12

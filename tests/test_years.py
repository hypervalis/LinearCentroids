"""Tests for year ordering experiment scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

from lce.concepts import (
    YEAR_ORDERINGS,
    YEAR_UNIVERSAL_SET,
    expand_year_prompts,
    validate_year_orderings,
    write_year_prompts,
)


def test_year_prompt_jsonl_schema(tmp_path: Path):
    path = write_year_prompts(tmp_path / "years.jsonl", years=("1990", "2013"), prompt_family="neutral")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 * 5
    record = json.loads(lines[0])
    assert record["concept"] == "year"
    assert record["value"] in ("1990", "2013")
    assert record["prompt"].endswith(record["value"])
    assert not record["prompt"].endswith(".")


def test_all_orderings_use_same_twenty_four_years():
    validate_year_orderings()
    for ordering in YEAR_ORDERINGS.values():
        assert len(ordering) == len(YEAR_UNIVERSAL_SET)
        assert set(ordering) == set(YEAR_UNIVERSAL_SET)


def test_default_year_prompts_use_universal_set():
    records = expand_year_prompts()
    values = {str(r["value"]) for r in records}
    assert values == set(YEAR_UNIVERSAL_SET)


def test_list_order_is_chronological():
    assert YEAR_ORDERINGS["list_order"] == YEAR_UNIVERSAL_SET
    assert YEAR_UNIVERSAL_SET[0] == "1990"
    assert YEAR_UNIVERSAL_SET[-1] == "2013"
    assert len(YEAR_UNIVERSAL_SET) == 24

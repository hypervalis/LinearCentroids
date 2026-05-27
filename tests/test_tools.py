"""Tests for tool ordering experiment scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

from lce.concepts import (
    TOOL_ORDERINGS,
    TOOL_UNIVERSAL_SET,
    expand_tool_prompts,
    validate_tool_orderings,
    write_tool_prompts,
)


def test_tool_prompt_jsonl_schema(tmp_path: Path):
    path = write_tool_prompts(tmp_path / "tools.jsonl", tools=("hammer", "saw"), prompt_family="neutral")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 * 5
    record = json.loads(lines[0])
    assert record["concept"] == "tool"
    assert record["value"] in ("hammer", "saw")
    assert record["prompt"].endswith(record["value"])
    assert not record["prompt"].endswith(".")


def test_all_orderings_use_same_ten_tools():
    validate_tool_orderings()
    for ordering in TOOL_ORDERINGS.values():
        assert len(ordering) == len(TOOL_UNIVERSAL_SET)
        assert set(ordering) == set(TOOL_UNIVERSAL_SET)


def test_default_tool_prompts_use_universal_set():
    records = expand_tool_prompts()
    values = {str(r["value"]) for r in records}
    assert values == set(TOOL_UNIVERSAL_SET)


def test_list_order_matches_user_sequence():
    assert TOOL_ORDERINGS["list_order"] == TOOL_UNIVERSAL_SET

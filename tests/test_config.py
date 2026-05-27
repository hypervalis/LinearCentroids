"""Tests for JSON config loading."""

from __future__ import annotations

import json
from pathlib import Path

from lce.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_demo_config_extends_full():
    cfg = load_config(ROOT / "configs" / "months_distilgpt2_demo.json")
    assert cfg["model_name"] == "distilgpt2"
    assert cfg["hook"]["layers"] == [0, 3, 5]
    assert cfg["geometry_kind"] == "cyclic"
    assert len(cfg["prompt_templates"]) == 3


def test_child_overrides_output_paths(tmp_path: Path):
    parent = tmp_path / "parent.json"
    child = tmp_path / "child.json"
    parent.write_text(json.dumps({"a": 1, "nested": {"x": 1, "list": [1]}}))
    child.write_text(json.dumps({"extends": "parent", "nested": {"y": 2}, "list": [2, 3]}))
    cfg = load_config(child)
    assert cfg["a"] == 1
    assert cfg["nested"] == {"x": 1, "y": 2, "list": [1]}
    assert cfg["list"] == [2, 3]

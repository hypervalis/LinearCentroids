"""Load experiment JSON configs with optional ``extends`` merge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load config JSON, recursively merging parent configs listed in ``extends``.

    Rules:
    - child overrides parent
    - dicts merge recursively
    - lists replace (no merge)
    - ``extends`` is a filename relative to the current config file's directory
    """
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    extends = data.get("extends")
    if not extends:
        return data

    parent_path = (path.parent / f"{extends}.json").resolve()
    if not parent_path.is_file():
        parent_path = (path.parent / extends).resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"parent config not found for extends={extends!r}")

    parent = load_config(parent_path)
    return _deep_merge(parent, data)

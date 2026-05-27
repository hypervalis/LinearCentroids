"""Generate prompt files from config or concept metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lce.concepts import DEFAULT_MONTH_TEMPLATES, MONTHS, expand_prompts
from lce.pipeline.paths import repo_path
from lce.registry import concept_from_config, default_prompt_output


def generate_prompts(
    cfg: dict[str, Any],
    *,
    output: Path | None = None,
    prompt_family: str | None = None,
) -> Path:
    spec = concept_from_config(cfg)
    family = prompt_family or cfg.get("prompt_family", "neutral")
    out = repo_path(output or default_prompt_output(spec, prompt_family=family))
    out.parent.mkdir(parents=True, exist_ok=True)

    if spec.name == "months":
        records = expand_prompts(MONTHS, tuple(cfg.get("prompt_templates", DEFAULT_MONTH_TEMPLATES)))
        out.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        return out

    if spec.write_prompts is None:
        raise ValueError(f"concept {spec.name!r} has no prompt writer")
    if spec.prompt_families and family not in spec.prompt_families:
        known = ", ".join(spec.prompt_families)
        raise ValueError(f"unknown prompt_family {family!r}; known: {known}")

    return spec.write_prompts(out, prompt_family=family)

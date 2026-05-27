"""Extract ln_2 activations and MLP-local centroids."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lce.extraction import (
    extract_animals_experiment,
    extract_colours_experiment,
    extract_disciplines_experiment,
    extract_languages_experiment,
    extract_months_experiment,
    extract_nouns_experiment,
    extract_stopwords_experiment,
    extract_tools_experiment,
    extract_whwords_experiment,
    extract_years_experiment,
    resolve_device,
)
from lce.pipeline.paths import repo_path
from lce.registry import concept_from_config


def extract_experiment(cfg: dict[str, Any]) -> list:
    spec = concept_from_config(cfg)
    device = resolve_device(cfg.get("device", "cuda_if_available"))
    layers = cfg["hook"]["layers"]
    raw_dir = repo_path(cfg["output"]["raw_dir"])
    templates = tuple(cfg["prompt_templates"])
    prompt_family = cfg.get("prompt_family", "neutral")

    common = dict(
        model_name=cfg["model_name"],
        layers=layers,
        templates=templates,
        raw_dir=raw_dir,
        device=device,
    )

    if spec.name == "months":
        return extract_months_experiment(**common)
    if spec.name == "colours":
        return extract_colours_experiment(**common, prompt_family=prompt_family)
    if spec.name == "nouns":
        return extract_nouns_experiment(**common, prompt_family=prompt_family)
    if spec.name == "tools":
        return extract_tools_experiment(**common, prompt_family=prompt_family)
    if spec.name == "animals":
        return extract_animals_experiment(**common, prompt_family=prompt_family)
    if spec.name == "years":
        return extract_years_experiment(**common, prompt_family=prompt_family)
    if spec.name == "disciplines":
        return extract_disciplines_experiment(**common, prompt_family=prompt_family)
    if spec.name == "whwords":
        return extract_whwords_experiment(**common, prompt_family=prompt_family)
    if spec.name == "stopwords":
        return extract_stopwords_experiment(**common, prompt_family=prompt_family)
    if spec.name == "languages":
        return extract_languages_experiment(**common, prompt_family=prompt_family)

    raise ValueError(f"no extractor for concept {spec.name!r}")

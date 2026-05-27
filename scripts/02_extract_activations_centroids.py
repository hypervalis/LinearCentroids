#!/usr/bin/env python3
"""Extract ln_2 activations and MLP-local centroids from DistilGPT-2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.config import load_config
from lce.extraction import (
    extract_animals_experiment,
    extract_colours_experiment,
    extract_disciplines_experiment,
    extract_months_experiment,
    extract_nouns_experiment,
    extract_stopwords_experiment,
    extract_languages_experiment,
    extract_tools_experiment,
    extract_whwords_experiment,
    extract_years_experiment,
    resolve_device,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "months_distilgpt2.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "cuda_if_available"))
    layers = cfg["hook"]["layers"]
    raw_dir = ROOT / cfg["output"]["raw_dir"]
    concept = cfg.get("concept_sequence", "months")

    print(f"Config: {args.config.resolve()}")
    print(f"Concept: {concept}")
    print(f"Device: {device}")
    print(f"Layers: {layers}")
    print(f"Output: {raw_dir.resolve()}")

    if concept == "colours":
        results = extract_colours_experiment(
            model_name=cfg["model_name"],
            layers=layers,
            templates=tuple(cfg["prompt_templates"]),
            prompt_family=cfg.get("prompt_family", "neutral"),
            raw_dir=raw_dir,
            device=device,
        )
    elif concept == "nouns":
        results = extract_nouns_experiment(
            model_name=cfg["model_name"],
            layers=layers,
            templates=tuple(cfg["prompt_templates"]),
            prompt_family=cfg.get("prompt_family", "neutral"),
            raw_dir=raw_dir,
            device=device,
        )
    elif concept == "tools":
        results = extract_tools_experiment(
            model_name=cfg["model_name"],
            layers=layers,
            templates=tuple(cfg["prompt_templates"]),
            prompt_family=cfg.get("prompt_family", "neutral"),
            raw_dir=raw_dir,
            device=device,
        )
    elif concept == "animals":
        results = extract_animals_experiment(
            model_name=cfg["model_name"],
            layers=layers,
            templates=tuple(cfg["prompt_templates"]),
            prompt_family=cfg.get("prompt_family", "neutral"),
            raw_dir=raw_dir,
            device=device,
        )
    elif concept == "years":
        results = extract_years_experiment(
            model_name=cfg["model_name"],
            layers=layers,
            templates=tuple(cfg["prompt_templates"]),
            prompt_family=cfg.get("prompt_family", "neutral"),
            raw_dir=raw_dir,
            device=device,
        )
    elif concept == "disciplines":
        results = extract_disciplines_experiment(
            model_name=cfg["model_name"],
            layers=layers,
            templates=tuple(cfg["prompt_templates"]),
            prompt_family=cfg.get("prompt_family", "neutral"),
            raw_dir=raw_dir,
            device=device,
        )
    elif concept == "whwords":
        results = extract_whwords_experiment(
            model_name=cfg["model_name"],
            layers=layers,
            templates=tuple(cfg["prompt_templates"]),
            prompt_family=cfg.get("prompt_family", "neutral"),
            raw_dir=raw_dir,
            device=device,
        )
    elif concept == "stopwords":
        results = extract_stopwords_experiment(
            model_name=cfg["model_name"],
            layers=layers,
            templates=tuple(cfg["prompt_templates"]),
            prompt_family=cfg.get("prompt_family", "neutral"),
            raw_dir=raw_dir,
            device=device,
        )
    elif concept == "languages":
        results = extract_languages_experiment(
            model_name=cfg["model_name"],
            layers=layers,
            templates=tuple(cfg["prompt_templates"]),
            prompt_family=cfg.get("prompt_family", "neutral"),
            raw_dir=raw_dir,
            device=device,
        )
    else:
        results = extract_months_experiment(
            model_name=cfg["model_name"],
            layers=layers,
            templates=tuple(cfg["prompt_templates"]),
            raw_dir=raw_dir,
            device=device,
        )
    for r in results:
        print(f"  layer {r.layer}: {r.per_template_path.resolve()}")
    print(f"  template variance: {(raw_dir / 'template_variance.json').resolve()}")


if __name__ == "__main__":
    main()

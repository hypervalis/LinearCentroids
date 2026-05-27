"""Sync experiment assets into ``docs/`` for GitHub Pages."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from lce.pipeline.paths import repo_path
from lce.registry import concept_from_config


DEFAULT_EXPLORE_CONFIG = {
    "model": "distilgpt2",
    "monthsLayers": [0, 1, 2, 3, 4, 5],
    "representations": ["activation", "centroid"],
    "defaultPlots": {
        "colours_distilgpt2": {"ordering": "roygbiv", "mode": "non_cyclic"},
        "nouns_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
        "tools_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
        "animals_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
        "years_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
        "disciplines_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
        "whwords_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
        "stopwords_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
        "languages_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
    },
}


def sync_experiment_assets(cfg: dict[str, Any]) -> Path:
    """Copy plots/metrics for one experiment into ``docs/experiments/<experiment>/``."""
    spec = concept_from_config(cfg)
    plots_dir = repo_path(cfg["output"]["plots_dir"])
    metrics_dir = repo_path(cfg["output"]["metrics_dir"])
    docs_dir = repo_path("docs")
    experiment = cfg.get("experiment", "default")
    experiments_dir = docs_dir / "experiments" / experiment
    assets_plots = experiments_dir / "plots"
    assets_plots.mkdir(parents=True, exist_ok=True)

    if spec.uses_month_geometry:
        metrics_path = metrics_dir / "geometry_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing {metrics_path}; run compute-geometry first")
        shutil.copy2(metrics_path, experiments_dir / "geometry_metrics.json")
        pattern = "*_pca3d.html"
    else:
        metrics_path = metrics_dir / (spec.metrics_filename or f"{spec.name}_orderings.json")
        if metrics_path.is_file():
            shutil.copy2(metrics_path, experiments_dir / metrics_path.name)
        pattern = "layer_*.html"

    if plots_dir.is_dir():
        for old in assets_plots.glob("*.html"):
            old.unlink()
        for html in plots_dir.glob(pattern):
            shutil.copy2(html, assets_plots / html.name)
        manifest_src = plots_dir / "manifest.json"
        if manifest_src.is_file():
            shutil.copy2(manifest_src, experiments_dir / "manifest.json")

    return experiments_dir


def patch_explore_config(
    *,
    model: str = "distilgpt2",
    months_layers: list[int] | None = None,
    representations: list[str] | None = None,
) -> None:
    """Update ``window.EXPLORE_CONFIG`` in ``docs/index.html`` without touching page copy."""
    index_html = repo_path("docs/index.html")
    if not index_html.is_file():
        raise FileNotFoundError(f"missing {index_html}")

    payload = dict(DEFAULT_EXPLORE_CONFIG)
    payload["model"] = model
    if months_layers is not None:
        payload["monthsLayers"] = months_layers
    if representations is not None:
        payload["representations"] = representations

    config_json = json.dumps(payload, separators=(",", ":"))
    html = index_html.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"<script>window\.EXPLORE_CONFIG = \{.*?\};</script>",
        f"<script>window.EXPLORE_CONFIG = {config_json};</script>",
        html,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("could not locate EXPLORE_CONFIG script block in docs/index.html")
    index_html.write_text(updated, encoding="utf-8")


def build_site(cfg: dict[str, Any], *, patch_config: bool = True) -> Path:
    experiments_dir = sync_experiment_assets(cfg)
    if patch_config and concept_from_config(cfg).uses_month_geometry:
        metrics_path = repo_path(cfg["output"]["metrics_dir"]) / "geometry_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        layers = sorted({int(row["layer"]) for row in metrics.get("layer_summary", [])})
        patch_explore_config(
            model=cfg.get("model_name", "distilgpt2"),
            months_layers=layers,
        )
    return experiments_dir

"""Plotly PCA path plots for months (cyclic geometry pipeline)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from lce.concepts import MONTHS
from lce.plotting import make_pca3d_figure, save_figure_html
from lce.pipeline.paths import repo_path


def make_month_plots(cfg: dict[str, Any], *, sync_docs: bool = False) -> Path:
    raw_dir = repo_path(cfg["output"]["raw_dir"])
    plots_dir = repo_path(cfg["output"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    variance_path = raw_dir / "template_variance.json"
    variance_data = (
        json.loads(variance_path.read_text(encoding="utf-8"))
        if variance_path.is_file()
        else {}
    )

    manifest: dict = {"experiment": cfg.get("experiment"), "figures": []}
    labels = list(MONTHS)
    pca_k = cfg["geometry"]["pca_components"]
    kind = cfg.get("geometry_kind", "cyclic")

    for layer in cfg["hook"]["layers"]:
        raw_path = raw_dir / f"layer_{layer:02d}_per_template.npz"
        if not raw_path.is_file():
            raise FileNotFoundError(f"missing {raw_path}; run extraction first")
        raw = np.load(raw_path, allow_pickle=True)
        act_mean = np.mean(raw["activations"], axis=0)
        cen_mean = np.mean(raw["centroids"], axis=0)
        _ = variance_data.get("layers", {}).get(str(layer), {})

        prefix = f"L{layer} activation"
        fig = make_pca3d_figure(
            act_mean,
            labels,
            geometry_kind=kind,
            n_pca_components=pca_k,
            title_prefix=prefix,
            centroid_arrows=cen_mean,
        )
        rel = f"layer_{layer:02d}_activation_pca3d.html"
        out = save_figure_html(fig, plots_dir / rel)
        manifest["figures"].append(
            {"layer": layer, "representation": "activation", "panel": "pca3d", "path": rel}
        )
        print(out.resolve())

        prefix = f"L{layer} centroid"
        fig = make_pca3d_figure(
            cen_mean,
            labels,
            geometry_kind=kind,
            n_pca_components=pca_k,
            title_prefix=prefix,
            centroid_arrows=None,
        )
        rel = f"layer_{layer:02d}_centroid_pca3d.html"
        out = save_figure_html(fig, plots_dir / rel)
        manifest["figures"].append(
            {"layer": layer, "representation": "centroid", "panel": "pca3d", "path": rel}
        )
        print(out.resolve())

    manifest_path = plots_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path.resolve()}")

    if sync_docs:
        _sync_month_docs(cfg, plots_dir, manifest_path)

    return manifest_path


def _sync_month_docs(cfg: dict[str, Any], plots_dir: Path, manifest_path: Path) -> None:
    exp = cfg.get("experiment", "months_distilgpt2")
    docs_plots = repo_path("docs") / "experiments" / exp / "plots"
    docs_plots.mkdir(parents=True, exist_ok=True)
    for old in docs_plots.glob("*.html"):
        old.unlink()
    for html in plots_dir.glob("layer_*.html"):
        shutil.copy2(html, docs_plots / html.name)
    shutil.copy2(manifest_path, repo_path("docs") / "experiments" / exp / "manifest.json")
    print(f"Synced plots to {docs_plots.resolve()}")

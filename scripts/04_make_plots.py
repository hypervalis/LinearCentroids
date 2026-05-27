#!/usr/bin/env python3
"""Generate Plotly HTML panels for each layer and representation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.concepts import MONTHS
from lce.config import load_config
from lce.geometry import analyze_trajectory
from lce.plotting import make_pca3d_figure, save_figure_html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "months_distilgpt2.json")
    parser.add_argument("--sync-docs", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = ROOT / cfg["output"]["raw_dir"]
    geometry_dir = ROOT / cfg["output"]["geometry_dir"]
    plots_dir = ROOT / cfg["output"]["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    variance_path = raw_dir / "template_variance.json"
    variance_data = json.loads(variance_path.read_text(encoding="utf-8")) if variance_path.is_file() else {}

    manifest: dict = {"experiment": cfg.get("experiment"), "figures": []}
    labels = list(MONTHS)
    pca_k = cfg["geometry"]["pca_components"]
    kind = cfg.get("geometry_kind", "cyclic")

    for layer in cfg["hook"]["layers"]:
        raw = np.load(raw_dir / f"layer_{layer:02d}_per_template.npz", allow_pickle=True)
        act_mean = np.mean(raw["activations"], axis=0)
        cen_mean = np.mean(raw["centroids"], axis=0)
        var_layer = variance_data.get("layers", {}).get(str(layer), {})

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

    if args.sync_docs:
        exp = cfg.get("experiment", "months_distilgpt2")
        docs_plots = ROOT / "docs" / "experiments" / exp / "plots"
        docs_plots.mkdir(parents=True, exist_ok=True)
        for old in docs_plots.glob("*.html"):
            old.unlink()
        for html in plots_dir.glob("layer_*.html"):
            shutil.copy2(html, docs_plots / html.name)
        shutil.copy2(manifest_path, ROOT / "docs" / "experiments" / exp / "manifest.json")
        print(f"Synced plots to {docs_plots.resolve()}")


if __name__ == "__main__":
    main()

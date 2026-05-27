#!/usr/bin/env python3
"""Generate Plotly PCA path plots for the tool list ordering."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.concepts import TOOL_ORDERING_EVAL_MODES, TOOL_ORDERINGS
from lce.config import load_config
from lce.plotting import make_pca3d_figure, save_figure_html
from lce.topology import stack_ordered_points


def _points_by_label(raw: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    concepts = [str(c) for c in raw["concepts"]]
    n_templates, n_concepts, _ = raw["activations"].shape
    activation = {
        concepts[col]: np.mean(raw["activations"][:, col, :], axis=0) for col in range(n_concepts)
    }
    centroid = {
        concepts[col]: np.mean(raw["centroids"][:, col, :], axis=0) for col in range(n_concepts)
    }
    return activation, centroid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "tools_distilgpt2.json")
    parser.add_argument("--sync-docs", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = ROOT / cfg["output"]["raw_dir"]
    plots_dir = ROOT / cfg["output"]["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)
    pca_k = cfg["geometry"]["pca_components"]
    orderings = cfg.get("topology", {}).get("orderings", list(TOOL_ORDERINGS.keys()))

    manifest: dict = {"experiment": cfg.get("experiment"), "figures": []}

    for layer in cfg["hook"]["layers"]:
        raw_path = raw_dir / f"layer_{layer:02d}_per_template.npz"
        if not raw_path.is_file():
            print(f"Missing {raw_path}; run 02_extract first.")
            sys.exit(1)
        raw = np.load(raw_path, allow_pickle=True)
        raw_dict = {
            "activations": raw["activations"],
            "centroids": raw["centroids"],
            "concepts": list(raw["concepts"]),
        }

        act_by_label, cen_by_label = _points_by_label(raw_dict)

        for space in ("activation", "centroid"):
            for ordering_name in orderings:
                ordering = TOOL_ORDERINGS[ordering_name]
                if any(label not in act_by_label for label in ordering):
                    print(f"skip {ordering_name}: missing labels in layer {layer}")
                    continue

                for mode_name, cyclic in TOOL_ORDERING_EVAL_MODES[ordering_name]:
                    labels = list(ordering)
                    kind = "cyclic" if cyclic else "linear"
                    prefix = f"L{layer} {space} {ordering_name} ({mode_name})"

                    if space == "activation":
                        act_points = stack_ordered_points(act_by_label, ordering)
                        cen_arrows = stack_ordered_points(cen_by_label, ordering)
                        fig = make_pca3d_figure(
                            act_points,
                            labels,
                            geometry_kind=kind,
                            n_pca_components=pca_k,
                            title_prefix=prefix,
                            centroid_arrows=cen_arrows,
                        )
                    else:
                        cen_points = stack_ordered_points(cen_by_label, ordering)
                        fig = make_pca3d_figure(
                            cen_points,
                            labels,
                            geometry_kind=kind,
                            n_pca_components=pca_k,
                            title_prefix=prefix,
                            centroid_arrows=None,
                        )

                    rel = f"layer_{layer:02d}_{space}_{ordering_name}_{mode_name}.html"
                    out = save_figure_html(fig, plots_dir / rel)
                    manifest["figures"].append(
                        {
                            "layer": layer,
                            "representation": space,
                            "ordering": ordering_name,
                            "mode": mode_name,
                            "cyclic": cyclic,
                            "path": rel,
                        }
                    )
                    print(out.resolve())

    manifest_path = plots_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path.resolve()}")

    if args.sync_docs:
        exp = cfg.get("experiment", "tools_distilgpt2")
        docs_plots = ROOT / "docs" / "experiments" / exp / "plots"
        if docs_plots.is_dir():
            for old in docs_plots.glob("*.html"):
                old.unlink()
        docs_plots.mkdir(parents=True, exist_ok=True)
        for html in plots_dir.glob("layer_*.html"):
            shutil.copy2(html, docs_plots / html.name)
        shutil.copy2(manifest_path, ROOT / "docs" / "experiments" / exp / "manifest.json")
        metrics_src = ROOT / cfg["output"]["metrics_dir"] / "tools_orderings.json"
        if metrics_src.is_file():
            shutil.copy2(metrics_src, ROOT / "docs" / "experiments" / exp / "tools_orderings.json")
        print(f"Synced plots to {docs_plots.resolve()}")


if __name__ == "__main__":
    main()

"""Plotly PCA path plots for ordering concept experiments."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from lce.plotting import make_pca3d_figure, save_figure_html
from lce.pipeline.paths import repo_path
from lce.registry import concept_from_config, default_ordering_names
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


def make_ordering_plots(cfg: dict[str, Any], *, sync_docs: bool = False) -> Path:
    spec = concept_from_config(cfg)
    if spec.orderings is None or spec.ordering_eval_modes is None:
        raise ValueError(
            f"concept {spec.name!r} has no orderings; use the months geometry pipeline instead"
        )

    raw_dir = repo_path(cfg["output"]["raw_dir"])
    plots_dir = repo_path(cfg["output"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    pca_k = cfg["geometry"]["pca_components"]
    orderings = cfg.get("topology", {}).get("orderings", default_ordering_names(spec))

    manifest: dict = {"experiment": cfg.get("experiment"), "figures": []}

    for layer in cfg["hook"]["layers"]:
        raw_path = raw_dir / f"layer_{layer:02d}_per_template.npz"
        if not raw_path.is_file():
            raise FileNotFoundError(f"missing {raw_path}; run extraction first")
        raw = np.load(raw_path, allow_pickle=True)
        raw_dict = {
            "activations": raw["activations"],
            "centroids": raw["centroids"],
            "concepts": list(raw["concepts"]),
        }

        act_by_label, cen_by_label = _points_by_label(raw_dict)

        for space in ("activation", "centroid"):
            for ordering_name in orderings:
                ordering = spec.orderings[ordering_name]
                if any(label not in act_by_label for label in ordering):
                    print(f"skip {ordering_name}: missing labels in layer {layer}")
                    continue

                for mode_name, cyclic in spec.ordering_eval_modes[ordering_name]:
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

    if sync_docs:
        _sync_ordering_docs(cfg, spec, plots_dir, manifest_path)

    return manifest_path


def _sync_ordering_docs(
    cfg: dict[str, Any],
    spec,
    plots_dir: Path,
    manifest_path: Path,
) -> None:
    exp = cfg.get("experiment", f"{spec.name}_distilgpt2")
    docs_root = repo_path("docs") / "experiments" / exp
    docs_plots = docs_root / "plots"
    if docs_plots.is_dir():
        for old in docs_plots.glob("*.html"):
            old.unlink()
    docs_plots.mkdir(parents=True, exist_ok=True)
    for html in plots_dir.glob("layer_*.html"):
        shutil.copy2(html, docs_plots / html.name)
    shutil.copy2(manifest_path, docs_root / "manifest.json")

    metrics_src = repo_path(cfg["output"]["metrics_dir"]) / (
        spec.metrics_filename or f"{spec.name}_orderings.json"
    )
    if metrics_src.is_file():
        shutil.copy2(metrics_src, docs_root / metrics_src.name)
    print(f"Synced plots to {docs_plots.resolve()}")

#!/usr/bin/env python3
"""Compute geometry metrics on template-mean trajectories from extraction outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.concepts import MONTHS
from lce.config import load_config
from lce.flags import compute_structure_hints, structure_hints_to_dict
from lce.geometry import (
    analyze_trajectory,
    analysis_to_dict,
    ordinal_metrics_to_dict,
    synthetic_circle,
    synthetic_line,
    synthetic_linear_ordinal,
    synthetic_random_walk,
)
from lce.io import save_metrics_json, save_trajectory_npz


def _load_layer_raw(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "activations": data["activations"],
        "centroids": data["centroids"],
        "concepts": list(data["concepts"]),
        "templates": list(data["templates"]),
    }


def _process_representation(
    points: np.ndarray,
    labels: list[str],
    geometry_kind: str,
    pca_components: int,
) -> dict:
    analysis = analyze_trajectory(
        points, n_pca_components=pca_components, geometry_kind=geometry_kind
    )
    metrics = analysis_to_dict(analysis, labels)
    metrics["ordinal"] = ordinal_metrics_to_dict(analysis.ordinal)
    return {"analysis": analysis, "metrics": metrics}


def run_synthetic(output_dir: Path, metrics_dir: Path, dim: int, pca_k: int) -> None:
    scenarios = {
        "cyclic_line": (synthetic_line(len(MONTHS), dim, noise=0.05, seed=0), "cyclic"),
        "circle": (synthetic_circle(len(MONTHS), dim, noise=0.02, seed=1), "cyclic"),
        "random_walk": (synthetic_random_walk(len(MONTHS), dim, seed=2), "cyclic"),
        "linear_ordinal": (synthetic_linear_ordinal(len(MONTHS), dim, noise=0.02, seed=3), "linear"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    labels = list(MONTHS)
    all_metrics: dict = {"scenarios": {}}

    for name, (points, kind) in scenarios.items():
        out = _process_representation(points, labels, kind, pca_k)
        save_trajectory_npz(output_dir / f"{name}.npz", out["analysis"], labels)
        metrics = out["metrics"]
        metrics["scenario"] = name
        all_metrics["scenarios"][name] = metrics
        print(f"{name}: closure={metrics['ordinal']['closure_to_edge_ratio']}")

    path = save_metrics_json(metrics_dir / "synthetic_geometry.json", all_metrics)
    print(f"Wrote {path.resolve()}")


def run_experiment(cfg: dict) -> None:
    raw_dir = ROOT / cfg["output"]["raw_dir"]
    geometry_dir = ROOT / cfg["output"]["geometry_dir"]
    metrics_dir = ROOT / cfg["output"]["metrics_dir"]
    geometry_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    variance_path = raw_dir / "template_variance.json"
    variance_data = (
        json.loads(variance_path.read_text(encoding="utf-8"))
        if variance_path.is_file()
        else {"layers": {}}
    )

    labels = list(MONTHS)
    geometry_kind = cfg.get("geometry_kind", "cyclic")
    pca_k = cfg["geometry"]["pca_components"]

    layer_summary = []
    full_metrics: dict = {"experiment": cfg.get("experiment"), "layers": {}}

    for layer in cfg["hook"]["layers"]:
        raw_path = raw_dir / f"layer_{layer:02d}_per_template.npz"
        if not raw_path.is_file():
            print(f"Missing {raw_path}; run 02_extract first.")
            sys.exit(1)
        raw = _load_layer_raw(raw_path)
        act_mean = np.mean(raw["activations"], axis=0)
        cen_mean = np.mean(raw["centroids"], axis=0)

        layer_metrics: dict = {"layer": layer, "representations": {}}
        for name, points in (("activation", act_mean), ("centroid", cen_mean)):
            out = _process_representation(points, labels, geometry_kind, pca_k)
            save_trajectory_npz(
                geometry_dir / f"layer_{layer:02d}_{name}.npz",
                out["analysis"],
                labels,
            )
            layer_metrics["representations"][name] = out["metrics"]

        var_layer = variance_data.get("layers", {}).get(str(layer), {})
        act_var = var_layer.get("activation", {}).get("mean_template_variance")
        cen_var = var_layer.get("centroid", {}).get("mean_template_variance")
        var_ratio = (cen_var / act_var) if act_var and cen_var is not None else None

        for name in ("activation", "centroid"):
            m = layer_metrics["representations"][name]
            hints = compute_structure_hints(
                closure_to_edge_ratio=m["ordinal"]["closure_to_edge_ratio"],
                edge_length_cv=m["ordinal"]["edge_length_cv"],
                adjacent_neighbour_accuracy=m["ordinal"]["adjacent_neighbour_accuracy"],
                mean_diff_concentration=m["mean_rho_diff"],
                mean_tangent_concentration=m["mean_rho_tangent"],
                mean_normal_concentration=m["mean_rho_normal"],
                template_variance_ratio_centroid_vs_activation=var_ratio,
            )
            m["structure_hints"] = structure_hints_to_dict(hints)

        full_metrics["layers"][str(layer)] = layer_metrics
        for name in ("activation", "centroid"):
            m = layer_metrics["representations"][name]
            layer_summary.append(
                {
                    "layer": layer,
                    "representation": name,
                    "edge_length_cv": m["ordinal"]["edge_length_cv"],
                    "closure_to_edge_ratio": m["ordinal"]["closure_to_edge_ratio"],
                    "adjacent_neighbour_accuracy": m["ordinal"]["adjacent_neighbour_accuracy"],
                    "curvature_smoothness_std": m["ordinal"]["curvature_smoothness_std"],
                    "mean_rho_diff": m["mean_rho_diff"],
                    "badges": m["structure_hints"]["badges"],
                }
            )
            print(
                f"layer {layer} {name}: closure={m['ordinal']['closure_to_edge_ratio']} "
                f"neighbour={m['ordinal']['adjacent_neighbour_accuracy']:.2f} "
                f"badges={m['structure_hints']['badges']}"
            )

    full_metrics["layer_summary"] = layer_summary
    out_path = save_metrics_json(metrics_dir / "geometry_metrics.json", full_metrics)
    print(f"Wrote {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "months_distilgpt2.json")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--dim", type=int, default=32)
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic(
            ROOT / "results/geometry/synthetic",
            ROOT / "results/metrics/synthetic",
            args.dim,
            3,
        )
        return

    cfg = load_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()

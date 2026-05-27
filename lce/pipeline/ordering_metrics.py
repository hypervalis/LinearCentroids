"""Ordering topology metrics for list/permutation concept experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lce.io import save_metrics_json
from lce.pipeline.paths import repo_path
from lce.registry import concept_from_config, default_ordering_names
from lce.topology import evaluate_ordering_with_baseline, evaluation_to_dict


def _parse_layers(spec: str, available: list[int]) -> list[int]:
    if spec.strip().lower() == "all":
        return available
    wanted = [int(x.strip()) for x in spec.split(",") if x.strip()]
    missing = [layer for layer in wanted if layer not in available]
    if missing:
        raise ValueError(f"layers not found in raw data: {missing}")
    return wanted


def _parse_spaces(spec: str) -> list[str]:
    spec = spec.lower()
    if spec == "both":
        return ["activation", "centroid"]
    if spec in ("activation", "centroid"):
        return [spec]
    raise ValueError("space must be activation, centroid, or both")


def _load_layer_raw(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "activations": data["activations"],
        "centroids": data["centroids"],
        "concepts": list(data["concepts"]),
        "templates": list(data["templates"]),
    }


def _points_by_label(raw: dict, space: str) -> dict[str, np.ndarray]:
    tensor = raw["activations"] if space == "activation" else raw["centroids"]
    concepts = [str(c) for c in raw["concepts"]]
    if tensor.ndim != 3:
        raise ValueError(f"expected (templates, concepts, dim), got {tensor.shape}")
    n_templates, n_concepts, _ = tensor.shape
    return {
        concepts[col]: np.mean(tensor[:, col, :], axis=0) for col in range(n_concepts)
    }


def discover_layers(raw_dir: Path) -> list[int]:
    layers: list[int] = []
    for path in sorted(raw_dir.glob("layer_*_per_template.npz")):
        layer = int(path.stem.split("_")[1])
        layers.append(layer)
    return sorted(layers)


def compute_ordering_metrics(
    cfg: dict[str, Any],
    *,
    raw_dir: Path | None = None,
    output: Path | None = None,
    space: str = "both",
    layers_spec: str = "all",
    num_random: int | None = None,
    seed: int = 0,
    prompt_family: str | None = None,
) -> Path:
    spec = concept_from_config(cfg)
    if spec.orderings is None or spec.ordering_eval_modes is None:
        raise ValueError(
            f"concept {spec.name!r} has no orderings; use the months geometry pipeline instead"
        )

    raw_dir = repo_path(raw_dir or cfg["output"]["raw_dir"])
    metrics_dir = repo_path(cfg["output"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output = repo_path(
        output or metrics_dir / (spec.metrics_filename or f"{spec.name}_orderings.json")
    )

    orderings = cfg.get("topology", {}).get("orderings", default_ordering_names(spec))
    num_random = cfg.get("topology", {}).get("num_random", num_random or 1000)
    prompt_family = prompt_family or cfg.get("prompt_family", "neutral")

    available = discover_layers(raw_dir)
    if not available:
        raise FileNotFoundError(f"no layer_*_per_template.npz in {raw_dir}; run extraction first")

    layers = _parse_layers(layers_spec, available)
    spaces = _parse_spaces(space)

    results: dict = {
        "experiment": cfg.get("experiment") or raw_dir.name,
        "prompt_family": prompt_family,
        "num_random": num_random,
        "layers": {},
        "summary_table": [],
    }

    print("layer | space      | ordering             | cyclic | path_E% | rank   | closure")
    for layer in layers:
        raw_path = raw_dir / f"layer_{layer:02d}_per_template.npz"
        raw = _load_layer_raw(raw_path)
        results["layers"][str(layer)] = {}

        for sp in spaces:
            points_by_label = _points_by_label(raw, sp)
            results["layers"][str(layer)][sp] = {}

            for ordering_name in orderings:
                if ordering_name not in spec.orderings:
                    raise KeyError(f"unknown ordering {ordering_name!r}")
                ordering = spec.orderings[ordering_name]
                missing = [lab for lab in ordering if lab not in points_by_label]
                if missing:
                    print(f"layer {layer} {sp} {ordering_name}: skip (missing labels {missing})")
                    continue

                eval_modes = spec.ordering_eval_modes.get(ordering_name, (("default", False),))
                results["layers"][str(layer)][sp][ordering_name] = {}

                for mode_name, cyclic in eval_modes:
                    evaluation = evaluate_ordering_with_baseline(
                        points_by_label,
                        ordering,
                        cyclic=cyclic,
                        num_samples=num_random,
                        seed=seed + layer * 17 + hash((ordering_name, mode_name, sp)) % 997,
                    )
                    payload = evaluation_to_dict(evaluation)
                    payload["mode"] = mode_name
                    results["layers"][str(layer)][sp][ordering_name][mode_name] = payload

                    m = evaluation.metrics
                    b = evaluation.path_energy_baseline
                    results["summary_table"].append(
                        {
                            "layer": layer,
                            "space": sp,
                            "ordering": ordering_name,
                            "mode": mode_name,
                            "cyclic": cyclic,
                            "path_energy_percentile": b.percentile,
                            "mean_edge_rank": m.mean_edge_rank,
                            "closure_to_edge_ratio": m.closure_to_edge_ratio,
                        }
                    )

                    print(
                        f"{layer:2d} | {sp:10s} | {ordering_name:20s} | "
                        f"{'Y' if cyclic else 'N':6s} | "
                        f"{b.percentile:6.3f} | {m.mean_edge_rank:6.3f} | "
                        f"{m.closure_to_edge_ratio if m.closure_to_edge_ratio is not None else float('nan'):6.3f}"
                    )

    save_metrics_json(output, results)
    return output

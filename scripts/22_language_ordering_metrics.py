#!/usr/bin/env python3
"""Evaluate imposed language list order against random permutation baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.concepts import LANGUAGE_ORDERING_EVAL_MODES, LANGUAGE_ORDERINGS
from lce.config import load_config
from lce.io import save_metrics_json
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


def _discover_layers(raw_dir: Path) -> list[int]:
    layers: list[int] = []
    for path in sorted(raw_dir.glob("layer_*_per_template.npz")):
        layer = int(path.stem.split("_")[1])
        layers.append(layer)
    return sorted(layers)


def run_ordering_metrics(
    *,
    raw_dir: Path,
    output: Path,
    space: str,
    layers: list[int],
    num_random: int,
    seed: int,
    orderings: list[str],
    prompt_family: str,
    experiment: str | None,
) -> dict:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw dir not found: {raw_dir}")

    spaces = _parse_spaces(space)
    results: dict = {
        "experiment": experiment or raw_dir.name,
        "prompt_family": prompt_family,
        "num_random": num_random,
        "layers": {},
        "summary_table": [],
    }

    for layer in layers:
        raw_path = raw_dir / f"layer_{layer:02d}_per_template.npz"
        if not raw_path.is_file():
            raise FileNotFoundError(f"missing {raw_path}; run extraction first")
        raw = _load_layer_raw(raw_path)
        results["layers"][str(layer)] = {}

        for sp in spaces:
            points_by_label = _points_by_label(raw, sp)
            results["layers"][str(layer)][sp] = {}

            for ordering_name in orderings:
                if ordering_name not in LANGUAGE_ORDERINGS:
                    raise KeyError(f"unknown ordering {ordering_name!r}")
                ordering = LANGUAGE_ORDERINGS[ordering_name]
                missing = [lab for lab in ordering if lab not in points_by_label]
                if missing:
                    print(f"layer {layer} {sp} {ordering_name}: skip (missing labels {missing})")
                    continue

                eval_modes = LANGUAGE_ORDERING_EVAL_MODES.get(ordering_name, (("default", False),))
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
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Year ordering topology metrics.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "languages_distilgpt2.json")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--space", default="both")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--num-random", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-family", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(args.raw_dir) if args.raw_dir else ROOT / cfg["output"]["raw_dir"]
    output = (
        Path(args.output)
        if args.output
        else ROOT / cfg["output"]["metrics_dir"] / "languages_orderings.json"
    )
    orderings = cfg.get("topology", {}).get("orderings", list(LANGUAGE_ORDERINGS.keys()))
    num_random = cfg.get("topology", {}).get("num_random", args.num_random)
    prompt_family = args.prompt_family or cfg.get("prompt_family", "neutral")

    available = _discover_layers(raw_dir)
    if not available:
        print(f"No layer_*_per_template.npz in {raw_dir}; run extraction first.")
        sys.exit(1)

    layers = _parse_layers(args.layers, available)
    print("layer | space      | ordering             | cyclic | path_E% | rank   | closure")
    run_ordering_metrics(
        raw_dir=raw_dir,
        output=output,
        space=args.space,
        layers=layers,
        num_random=num_random,
        seed=args.seed,
        orderings=orderings,
        prompt_family=prompt_family,
        experiment=cfg.get("experiment"),
    )
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()

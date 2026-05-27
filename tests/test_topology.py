"""Tests for colour ordering / path topology metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lce.concepts import (
    COLOUR_ORDERINGS,
    COLOUR_UNIVERSAL_SET,
    ROYGBIV,
    expand_colour_prompts,
    validate_colour_orderings,
    write_colour_prompts,
)
from lce.topology import (
    compute_ordered_path_metrics,
    evaluate_ordering_with_baseline,
    random_ordering_baseline,
    stack_ordered_points,
)


def _line_points(labels: list[str], dim: int = 8) -> dict[str, np.ndarray]:
    direction = np.zeros(dim)
    direction[0] = 1.0
    return {label: float(i) * direction for i, label in enumerate(labels)}


def _circle_points(labels: list[str], dim: int = 8, radius: float = 1.0) -> dict[str, np.ndarray]:
    n = len(labels)
    points: dict[str, np.ndarray] = {}
    for i, label in enumerate(labels):
        angle = 2.0 * np.pi * i / n
        vec = np.zeros(dim)
        vec[0] = radius * np.cos(angle)
        vec[1] = radius * np.sin(angle)
        points[label] = vec
    return points


def test_colour_prompt_jsonl_schema(tmp_path: Path):
    path = write_colour_prompts(tmp_path / "colours.jsonl", colours=("red", "blue"), prompt_family="neutral")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 * 5
    record = json.loads(lines[0])
    assert record["concept"] == "colour"
    assert record["value"] in ("red", "blue")
    assert "index" in record
    assert record["prompt_family"] == "neutral"
    assert record["prompt"].endswith(record["value"])
    assert not record["prompt"].endswith(".")


def test_expand_colour_prompt_families():
    perceptual = expand_colour_prompts(colours=("red",), prompt_family="perceptual")
    assert perceptual[0]["template_id"] == "perceptual_00"
    assert "hue" in perceptual[0]["prompt"]


def test_line_ordering_beats_random_on_path_energy():
    labels = list(ROYGBIV)
    points = _line_points(labels)
    ordered = compute_ordered_path_metrics(points, labels, cyclic=False)
    baseline = random_ordering_baseline(
        points, labels, labels, cyclic=False, num_samples=200, seed=0
    )
    assert ordered.path_energy < baseline["path_energy"].random_mean
    assert baseline["path_energy"].percentile > 0.9


def test_circle_cyclic_hue_order_beats_random():
    labels = list(COLOUR_ORDERINGS["hue_wheel_approx"])
    assert len(labels) == len(COLOUR_UNIVERSAL_SET)
    points = _circle_points(labels)
    evaluation = evaluate_ordering_with_baseline(
        points, labels, cyclic=True, num_samples=200, seed=1
    )
    assert evaluation.path_energy_baseline.percentile > 0.8
    assert evaluation.metrics.closure_to_edge_ratio is not None
    assert abs(evaluation.metrics.closure_to_edge_ratio - 1.0) < 0.05


def test_random_cloud_no_strong_ordering_signal():
    rng = np.random.default_rng(42)
    labels = list(ROYGBIV)
    points = {label: rng.standard_normal(16) for label in labels}
    percentiles: list[float] = []
    for seed in range(12):
        perm = tuple(rng.permutation(labels))
        baseline = random_ordering_baseline(
            points, labels, perm, cyclic=False, num_samples=200, seed=seed
        )
        percentiles.append(baseline["path_energy"].percentile)
    # Random permutations on an unstructured cloud should not all look exceptional.
    assert float(np.median(percentiles)) < 0.85


def test_line_ordering_beats_shuffled_on_mean_edge_rank():
    labels = list(ROYGBIV)
    points = _line_points(labels)
    ordered = compute_ordered_path_metrics(points, labels, cyclic=False)
    shuffled = compute_ordered_path_metrics(
        points, ["green", "red", "violet", "orange", "blue", "yellow", "indigo"], cyclic=False
    )
    assert ordered.path_energy < shuffled.path_energy
    assert ordered.mean_edge_rank < shuffled.mean_edge_rank


def test_stack_ordered_points_shape():
    labels = ["red", "green", "blue"]
    points = _line_points(labels, dim=4)
    stacked = stack_ordered_points(points, labels)
    assert stacked.shape == (3, 4)


def test_all_orderings_use_same_seven_colours():
    validate_colour_orderings()
    for ordering in COLOUR_ORDERINGS.values():
        assert len(ordering) == len(COLOUR_UNIVERSAL_SET)
        assert set(ordering) == set(COLOUR_UNIVERSAL_SET)


def test_default_colour_prompts_use_universal_set():
    records = expand_colour_prompts()
    values = {str(r["value"]) for r in records}
    assert values == set(COLOUR_UNIVERSAL_SET)

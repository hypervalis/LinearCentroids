"""
Ordering / path topology metrics for categorical concepts (colours).

Unlike months (natural cyclic ordinal order), colour sets admit many plausible
imposed orderings. We ask whether a named ordering traces short edges through
activation/centroid space better than random permutations.

TODO: Add persistent homology / Vietoris-Rips H1 loop diagnostics for expanded
colour sets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations

import numpy as np

from lce.geometry import compute_closure_ratios, compute_edge_length_cv, first_differences

EPS: float = 1e-12


@dataclass(frozen=True)
class OrderedPathMetrics:
    """Metrics for one imposed ordering through a semantic point cloud."""

    ordering: tuple[str, ...]
    cyclic: bool
    path_length: float
    path_energy: float
    edge_lengths: np.ndarray
    edge_length_cv: float
    closure_edge_length: float | None
    closure_to_edge_ratio: float | None
    closure_to_path_ratio: float | None
    mean_edge_rank: float
    adjacent_edge_percentile: float
    edge_ranks: np.ndarray
    edge_rank_percentiles: np.ndarray


@dataclass(frozen=True)
class RandomBaselineComparison:
    """Observed metric vs random permutation distribution."""

    observed: float
    random_mean: float
    random_std: float
    percentile: float


@dataclass(frozen=True)
class OrderingEvaluation:
    """Path metrics plus random baselines for key statistics."""

    metrics: OrderedPathMetrics
    path_length_baseline: RandomBaselineComparison
    path_energy_baseline: RandomBaselineComparison
    mean_edge_rank_baseline: RandomBaselineComparison


def _validate_points_by_label(points_by_label: dict[str, np.ndarray]) -> None:
    if len(points_by_label) < 2:
        raise ValueError("need at least 2 labelled points")
    dim = next(iter(points_by_label.values())).shape[0]
    for label, vec in points_by_label.items():
        arr = np.asarray(vec, dtype=np.float64)
        if arr.shape != (dim,):
            raise ValueError(f"point for {label!r} has shape {arr.shape}, expected ({dim},)")


def stack_ordered_points(
    points_by_label: dict[str, np.ndarray],
    ordering: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Stack ``points_by_label[label]`` in imposed order, shape (T, D)."""
    _validate_points_by_label(points_by_label)
    missing = [label for label in ordering if label not in points_by_label]
    if missing:
        raise KeyError(f"ordering labels missing from points_by_label: {missing}")
    return np.stack([np.asarray(points_by_label[label], dtype=np.float64) for label in ordering], axis=0)


def _edge_rank_stats(
    points_by_label: dict[str, np.ndarray],
    ordering: list[str] | tuple[str, ...],
    *,
    cyclic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each ordered edge a→b, rank ``b`` among all other colours by distance from ``a``.

    Returns (ranks, percentiles) with rank 1 = nearest neighbour (best).
    Percentile in [0, 1]: 0 = best, 1 = worst.
    """
    labels = list(ordering)
    n = len(labels)
    if n < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    edges: list[tuple[str, str]] = [(labels[i], labels[i + 1]) for i in range(n - 1)]
    if cyclic:
        edges.append((labels[-1], labels[0]))

    ranks: list[float] = []
    percentiles: list[float] = []
    all_labels = list(points_by_label.keys())

    for source, target in edges:
        source_vec = np.asarray(points_by_label[source], dtype=np.float64)
        candidates = [lab for lab in all_labels if lab != source]
        if not candidates:
            continue
        dists = np.array(
            [np.linalg.norm(source_vec - np.asarray(points_by_label[lab], dtype=np.float64)) for lab in candidates]
        )
        order = np.argsort(dists)
        sorted_labels = [candidates[i] for i in order]
        rank = float(sorted_labels.index(target) + 1)
        k = len(candidates)
        percentile = 0.0 if k <= 1 else float((rank - 1) / (k - 1))
        ranks.append(rank)
        percentiles.append(percentile)

    return np.asarray(ranks, dtype=np.float64), np.asarray(percentiles, dtype=np.float64)


def compute_ordered_path_metrics(
    points_by_label: dict[str, np.ndarray],
    ordering: list[str] | tuple[str, ...],
    *,
    cyclic: bool = False,
) -> OrderedPathMetrics:
    """
    Evaluate an imposed ordering as a path through ``points_by_label``.

    Lower ``path_energy``, ``mean_edge_rank``, and ``adjacent_edge_percentile``
    indicate the ordering follows nearby geometry.
    """
    ordering = tuple(ordering)
    points = stack_ordered_points(points_by_label, ordering)
    geometry_kind = "cyclic" if cyclic else "linear"
    diffs = first_differences(points, cyclic=cyclic)
    edge_lengths = np.linalg.norm(diffs, axis=1)
    closure_len, closure_edge, closure_path = compute_closure_ratios(
        edge_lengths, geometry_kind=geometry_kind
    )
    ranks, rank_percentiles = _edge_rank_stats(points_by_label, ordering, cyclic=cyclic)

    return OrderedPathMetrics(
        ordering=ordering,
        cyclic=cyclic,
        path_length=float(np.sum(edge_lengths)),
        path_energy=float(np.sum(edge_lengths**2)),
        edge_lengths=edge_lengths,
        edge_length_cv=compute_edge_length_cv(edge_lengths),
        closure_edge_length=closure_len,
        closure_to_edge_ratio=closure_edge,
        closure_to_path_ratio=closure_path,
        mean_edge_rank=float(np.mean(ranks)) if ranks.size else float("nan"),
        adjacent_edge_percentile=float(np.mean(rank_percentiles)) if rank_percentiles.size else float("nan"),
        edge_ranks=ranks,
        edge_rank_percentiles=rank_percentiles,
    )


def _percentile_lower_is_better(observed: float, random_values: np.ndarray) -> float:
    """Fraction of random draws with value >= observed (higher percentile = better path)."""
    if random_values.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float(np.mean(random_values >= observed))


def _sample_permutations(labels: tuple[str, ...], num_samples: int, rng: np.random.Generator) -> list[tuple[str, ...]]:
    n = len(labels)
    total = math.factorial(n)
    if n <= 8 and total <= num_samples:
        perm_list = list(permutations(labels))
        rng.shuffle(perm_list)
        return perm_list[:num_samples]
    return [tuple(rng.permutation(labels)) for _ in range(num_samples)]


def random_ordering_baseline(
    points_by_label: dict[str, np.ndarray],
    labels: list[str] | tuple[str, ...],
    ordering: list[str] | tuple[str, ...],
    *,
    cyclic: bool,
    num_samples: int = 1000,
    seed: int = 0,
) -> dict[str, RandomBaselineComparison]:
    """
    Compare an observed ordering to random permutations of the same label set.

    Returns baselines for path_length, path_energy, and mean_edge_rank.
    Percentile: fraction of random permutations with metric >= observed
    (higher = observed path is shorter / lower-energy / better-ranked).
    """
    labels = tuple(labels)
    ordering = tuple(ordering)
    if len(labels) < 2:
        raise ValueError("need at least 2 labels")

    observed = compute_ordered_path_metrics(points_by_label, ordering, cyclic=cyclic)
    rng = np.random.default_rng(seed)
    samples = _sample_permutations(labels, num_samples, rng)

    path_lengths: list[float] = []
    path_energies: list[float] = []
    mean_ranks: list[float] = []
    for perm in samples:
        m = compute_ordered_path_metrics(points_by_label, perm, cyclic=cyclic)
        path_lengths.append(m.path_length)
        path_energies.append(m.path_energy)
        mean_ranks.append(m.mean_edge_rank)

    pl = np.asarray(path_lengths, dtype=np.float64)
    pe = np.asarray(path_energies, dtype=np.float64)
    mr = np.asarray(mean_ranks, dtype=np.float64)

    def _cmp(observed_val: float, samples_arr: np.ndarray) -> RandomBaselineComparison:
        return RandomBaselineComparison(
            observed=float(observed_val),
            random_mean=float(np.mean(samples_arr)),
            random_std=float(np.std(samples_arr)),
            percentile=_percentile_lower_is_better(observed_val, samples_arr),
        )

    return {
        "path_length": _cmp(observed.path_length, pl),
        "path_energy": _cmp(observed.path_energy, pe),
        "mean_edge_rank": _cmp(observed.mean_edge_rank, mr),
    }


def evaluate_ordering_with_baseline(
    points_by_label: dict[str, np.ndarray],
    ordering: list[str] | tuple[str, ...],
    *,
    cyclic: bool,
    num_samples: int = 1000,
    seed: int = 0,
) -> OrderingEvaluation:
    """Compute path metrics and random baselines for one imposed ordering."""
    ordering = tuple(ordering)
    metrics = compute_ordered_path_metrics(points_by_label, ordering, cyclic=cyclic)
    baselines = random_ordering_baseline(
        points_by_label,
        ordering,
        ordering,
        cyclic=cyclic,
        num_samples=num_samples,
        seed=seed,
    )
    return OrderingEvaluation(
        metrics=metrics,
        path_length_baseline=baselines["path_length"],
        path_energy_baseline=baselines["path_energy"],
        mean_edge_rank_baseline=baselines["mean_edge_rank"],
    )


def ordered_path_metrics_to_dict(metrics: OrderedPathMetrics) -> dict:
    return {
        "ordering": list(metrics.ordering),
        "cyclic": metrics.cyclic,
        "path_length": metrics.path_length,
        "path_energy": metrics.path_energy,
        "edge_lengths": metrics.edge_lengths.tolist(),
        "edge_length_cv": metrics.edge_length_cv,
        "closure_edge_length": metrics.closure_edge_length,
        "closure_to_edge_ratio": metrics.closure_to_edge_ratio,
        "closure_to_path_ratio": metrics.closure_to_path_ratio,
        "mean_edge_rank": metrics.mean_edge_rank,
        "adjacent_edge_percentile": metrics.adjacent_edge_percentile,
        "edge_ranks": metrics.edge_ranks.tolist(),
        "edge_rank_percentiles": metrics.edge_rank_percentiles.tolist(),
    }


def baseline_comparison_to_dict(baseline: RandomBaselineComparison) -> dict:
    return {
        "observed": baseline.observed,
        "random_mean": baseline.random_mean,
        "random_std": baseline.random_std,
        "percentile": baseline.percentile,
    }


def evaluation_to_dict(evaluation: OrderingEvaluation) -> dict:
    return {
        "metrics": ordered_path_metrics_to_dict(evaluation.metrics),
        "baseline": {
            "path_length": baseline_comparison_to_dict(evaluation.path_length_baseline),
            "path_energy": baseline_comparison_to_dict(evaluation.path_energy_baseline),
            "mean_edge_rank": baseline_comparison_to_dict(evaluation.mean_edge_rank_baseline),
        },
    }

#!/usr/bin/env python3
"""Synthetic validation of Frenet-like geometry (line, circle, random walk)."""

from __future__ import annotations

import numpy as np
import pytest

from lce.geometry import (
    analyze_trajectory,
    compute_adjacent_neighbour_accuracy,
    compute_edge_lengths,
    compute_trajectory_frame,
    first_differences,
    summarize_curvature,
    synthetic_circle,
    synthetic_line,
    synthetic_linear_ordinal,
    synthetic_random_walk,
)

N = 12
DIM = 32


# --- diffs / edges ---


def test_cyclic_differences_wrap_includes_dec_jan():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    diffs = first_differences(points, cyclic=True)
    np.testing.assert_allclose(diffs[2], points[0] - points[2])
    assert compute_edge_lengths(points, geometry_kind="cyclic").shape[0] == 3


def test_linear_diffs_shape():
    points = synthetic_linear_ordinal(N, DIM)
    assert first_differences(points, cyclic=False).shape == (N - 1, DIM)


# --- 1. Straight line (non-cyclic) ---


def test_linear_ordinal_near_zero_curvature():
    analysis = analyze_trajectory(
        synthetic_linear_ordinal(N, DIM, noise=0.0), geometry_kind="linear"
    )
    assert analysis.curvature_summary["mean"] < 1e-6


def test_linear_ordinal_high_top1_pca_capture():
    analysis = analyze_trajectory(
        synthetic_linear_ordinal(N, DIM, noise=0.0),
        n_pca_components=3,
        geometry_kind="linear",
    )
    assert np.nanmean(analysis.concentration.rho_diff) > 0.99
    assert analysis.pca.explained_variance_ratio[0] > 0.99


def test_linear_ordinal_no_closure_metrics():
    analysis = analyze_trajectory(
        synthetic_linear_ordinal(N, DIM, noise=0.0), geometry_kind="linear"
    )
    assert analysis.ordinal.closure_to_edge_ratio is None
    assert analysis.ordinal.edge_lengths.shape[0] == N - 1


def test_cyclic_line_near_zero_curvature():
    frame = compute_trajectory_frame(synthetic_line(N, DIM, noise=0.0))
    assert summarize_curvature(frame.curvature)["mean"] < 1e-6


# --- 2. Circle (cyclic) ---


def test_circle_uniform_nonzero_curvature():
    frame = compute_trajectory_frame(synthetic_circle(N, DIM, noise=0.0))
    valid = frame.curvature[np.isfinite(frame.curvature)]
    assert valid.size == N
    assert np.std(valid) < 1e-10
    assert np.mean(valid) > 0.5


def test_circle_closure_to_edge_ratio_near_one():
    analysis = analyze_trajectory(synthetic_circle(N, DIM, noise=0.0), geometry_kind="cyclic")
    assert abs(analysis.ordinal.closure_to_edge_ratio - 1.0) < 1e-6


def test_circle_low_edge_length_cv():
    analysis = analyze_trajectory(synthetic_circle(N, DIM, noise=0.0), geometry_kind="cyclic")
    assert analysis.ordinal.edge_length_cv < 1e-6


def test_circle_high_pca_concentration():
    analysis = analyze_trajectory(
        synthetic_circle(N, DIM, noise=0.0), n_pca_components=3, geometry_kind="cyclic"
    )
    assert np.nanmean(analysis.concentration.rho_diff) > 0.95
    assert np.nanmean(analysis.concentration.rho_tangent) > 0.95
    assert np.nanmean(analysis.concentration.rho_normal) > 0.90


def test_circle_high_tangent_alignment():
    analysis = analyze_trajectory(
        synthetic_circle(N, DIM, noise=0.0), n_pca_components=3, geometry_kind="cyclic"
    )
    assert np.nanmean(analysis.frame_comparison.tangent_alignment_full_vs_pca) > 0.99


def test_normal_perpendicular_to_tangent_on_circle():
    frame = compute_trajectory_frame(synthetic_circle(N, 2, noise=0.0))
    dots = np.sum(frame.normals * frame.tangents, axis=1)
    np.testing.assert_allclose(dots, 0.0, atol=1e-8)


def test_circle_perfect_adjacent_neighbours_2d():
    analysis = analyze_trajectory(synthetic_circle(N, 2, noise=0.0), geometry_kind="cyclic")
    assert analysis.ordinal.adjacent_neighbour_accuracy == 1.0


# --- 3. Random high-dimensional walk ---


def test_random_walk_lower_concentration_than_circle():
    circle = analyze_trajectory(synthetic_circle(N, DIM, noise=0.0), geometry_kind="cyclic")
    walk = analyze_trajectory(synthetic_random_walk(N, DIM, seed=7), geometry_kind="cyclic")
    assert np.nanmean(walk.concentration.rho_diff) < np.nanmean(circle.concentration.rho_diff)


def test_random_walk_higher_curvature_variability_than_circle():
    circle = analyze_trajectory(synthetic_circle(N, DIM, noise=0.0), geometry_kind="cyclic")
    walk = analyze_trajectory(synthetic_random_walk(N, DIM, seed=7), geometry_kind="cyclic")
    assert walk.curvature_summary["std"] > circle.curvature_summary["std"]


def test_random_walk_lower_normal_alignment_than_circle():
    circle = analyze_trajectory(
        synthetic_circle(N, DIM, noise=0.0), n_pca_components=3, geometry_kind="cyclic"
    )
    walk = analyze_trajectory(
        synthetic_random_walk(N, DIM, seed=7), n_pca_components=3, geometry_kind="cyclic"
    )
    # Tangents often align even under distortion; normals expose PCA fidelity loss.
    assert np.nanmean(walk.frame_comparison.normal_alignment_full_vs_pca) < np.nanmean(
        circle.frame_comparison.normal_alignment_full_vs_pca
    )


def test_random_walk_exceeds_line_curvature():
    k_line = summarize_curvature(compute_trajectory_frame(synthetic_line(N, DIM)).curvature)["mean"]
    k_walk = summarize_curvature(
        compute_trajectory_frame(synthetic_random_walk(N, DIM, seed=42)).curvature
    )["mean"]
    assert k_walk > k_line + 0.1


# --- misc ---


def test_shuffled_order_lowers_adjacent_accuracy():
    points = synthetic_circle(N, DIM, noise=0.0)
    rng = np.random.default_rng(0)
    clean = analyze_trajectory(points, geometry_kind="cyclic")
    shuffled = analyze_trajectory(points[rng.permutation(N)], geometry_kind="cyclic")
    assert shuffled.ordinal.adjacent_neighbour_accuracy < clean.ordinal.adjacent_neighbour_accuracy


def test_hidden_noise_reduces_rho_diff():
    analysis = analyze_trajectory(
        synthetic_circle(N, 50, noise=0.5, seed=0), n_pca_components=3, geometry_kind="cyclic"
    )
    assert np.nanmean(analysis.concentration.rho_diff) < 0.6


def test_npz_field_names(tmp_path):
    from lce.io import save_trajectory_npz

    analysis = analyze_trajectory(synthetic_circle(N, DIM), geometry_kind="cyclic")
    path = save_trajectory_npz(tmp_path / "t.npz", analysis, [str(i) for i in range(N)])
    data = np.load(path, allow_pickle=True)
    for key in (
        "projected_points",
        "projected_diffs",
        "projected_full_tangents",
        "projected_full_normals",
        "pca_tangents",
        "pca_normals",
        "diff_concentration",
        "tangent_concentration",
        "normal_concentration",
        "tangent_alignment_full_vs_pca",
        "normal_alignment_full_vs_pca",
        "edge_length_cv",
        "closure_to_edge_ratio",
    ):
        assert key in data

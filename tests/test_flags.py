"""Tests for heuristic structure hints."""

from __future__ import annotations

from lce.flags import compute_structure_hints, structure_hints_to_dict


def test_circle_like_hints():
    hints = compute_structure_hints(
        closure_to_edge_ratio=1.0,
        edge_length_cv=0.05,
        adjacent_neighbour_accuracy=1.0,
        mean_diff_concentration=0.9,
        mean_tangent_concentration=0.9,
        mean_normal_concentration=0.5,
        template_variance_ratio_centroid_vs_activation=0.8,
    )
    d = structure_hints_to_dict(hints)
    assert hints.cyclic_edge_hint
    assert hints.neighbour_order_hint
    assert hints.projection_reliable_hint
    assert hints.bending_visible_hint
    assert hints.prompt_stable_hint is True
    assert hints.any_structure_hint
    assert set(d["badges"]) == {
        "cyclic",
        "ordered-neighbours",
        "PCA-visible",
        "visible-bending",
        "prompt-stable",
    }


def test_no_composite_score_in_output():
    hints = compute_structure_hints(
        closure_to_edge_ratio=0.5,
        edge_length_cv=0.9,
        adjacent_neighbour_accuracy=0.1,
        mean_diff_concentration=0.1,
        mean_tangent_concentration=0.1,
        mean_normal_concentration=0.1,
    )
    assert not hints.any_structure_hint
    assert hints.prompt_stable_hint is None

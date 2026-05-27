"""Heuristic structure hints for layer summary tables (not significance tests)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructureHints:
    """Descriptive navigation flags — not claims about concept encoding."""

    cyclic_edge_hint: bool
    neighbour_order_hint: bool
    projection_reliable_hint: bool
    bending_visible_hint: bool
    prompt_stable_hint: bool | None
    any_structure_hint: bool

    def badge_labels(self) -> list[str]:
        labels: list[str] = []
        if self.cyclic_edge_hint:
            labels.append("cyclic")
        if self.neighbour_order_hint:
            labels.append("ordered-neighbours")
        if self.projection_reliable_hint:
            labels.append("PCA-visible")
        if self.bending_visible_hint:
            labels.append("visible-bending")
        if self.prompt_stable_hint:
            labels.append("prompt-stable")
        return labels


def compute_structure_hints(
    *,
    closure_to_edge_ratio: float | None,
    edge_length_cv: float,
    adjacent_neighbour_accuracy: float,
    mean_diff_concentration: float,
    mean_tangent_concentration: float,
    mean_normal_concentration: float,
    template_variance_ratio_centroid_vs_activation: float | None = None,
) -> StructureHints:
    """
    Conservative heuristic flags for exploratory navigation.

    See docs/DECISIONS.md — these are not statistical tests and must not be
    read as proof of concept encoding.
    """
    cyclic_edge = (
        closure_to_edge_ratio is not None
        and 0.7 <= closure_to_edge_ratio <= 1.3
        and edge_length_cv <= 0.40
    )
    neighbour_order = adjacent_neighbour_accuracy >= 0.50
    projection_reliable = (
        mean_diff_concentration >= 0.60 and mean_tangent_concentration >= 0.60
    )
    bending_visible = mean_normal_concentration >= 0.40

    prompt_stable: bool | None = None
    if template_variance_ratio_centroid_vs_activation is not None:
        prompt_stable = template_variance_ratio_centroid_vs_activation <= 1.0

    hints = (
        cyclic_edge,
        neighbour_order,
        projection_reliable,
        bending_visible,
        prompt_stable is True,
    )
    return StructureHints(
        cyclic_edge_hint=cyclic_edge,
        neighbour_order_hint=neighbour_order,
        projection_reliable_hint=projection_reliable,
        bending_visible_hint=bending_visible,
        prompt_stable_hint=prompt_stable,
        any_structure_hint=any(h for h in hints if h is not False),
    )


def structure_hints_to_dict(hints: StructureHints) -> dict:
    return {
        "cyclic_edge_hint": hints.cyclic_edge_hint,
        "neighbour_order_hint": hints.neighbour_order_hint,
        "projection_reliable_hint": hints.projection_reliable_hint,
        "bending_visible_hint": hints.bending_visible_hint,
        "prompt_stable_hint": hints.prompt_stable_hint,
        "any_structure_hint": hints.any_structure_hint,
        "badges": hints.badge_labels(),
    }

"""Save/load helpers for experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from lce.geometry import TrajectoryAnalysis


def save_trajectory_npz(path: str | Path, analysis: TrajectoryAnalysis, labels: list[str]) -> Path:
    """
    Save full Frenet-like geometry arrays for one trajectory.

    Full-space arrays are the primary geometry; ``projected_*`` and ``pca_*``
    are PCA shadows / contrast frames.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fc = analysis.frame_comparison
    conc = analysis.concentration
    proj = analysis.frame_projected
    pca_native = analysis.frame_pca_native
    ordm = analysis.ordinal

    np.savez_compressed(
        path,
        # full-space points and frame (true geometry)
        points=analysis.frame.points,
        diffs=analysis.frame.diffs,
        tangents=analysis.frame.tangents,
        normals=analysis.frame.normals,
        normals_raw=analysis.frame.normals_raw,
        curvature=analysis.frame.curvature,
        # PCA projection of points
        projected_points=analysis.points_pca,
        pca_projection_matrix=analysis.pca.projection_matrix,
        pca_mean=analysis.pca.mean,
        pca_explained_variance_ratio=analysis.pca.explained_variance_ratio,
        # projected full-space vectors (shadows in R^k)
        projected_diffs=proj.diffs,
        projected_full_tangents=proj.tangents,
        projected_full_normals=proj.normals,
        # PCA-native contrast frame (not true geometry)
        pca_tangents=pca_native.tangents,
        pca_normals=pca_native.normals,
        pca_curvature=pca_native.curvature,
        # concentration ρ_d, ρ_T, ρ_N
        diff_concentration=conc.rho_diff,
        tangent_concentration=conc.rho_tangent,
        normal_concentration=conc.rho_normal,
        rho_diff=conc.rho_diff,
        rho_tangent=conc.rho_tangent,
        rho_normal=conc.rho_normal,
        # frame alignment (projected full-space vs PCA-native)
        tangent_alignment=fc.tangent_alignment_full_vs_pca,
        normal_alignment=fc.normal_alignment_full_vs_pca,
        tangent_alignment_full_vs_pca=fc.tangent_alignment_full_vs_pca,
        normal_alignment_full_vs_pca=fc.normal_alignment_full_vs_pca,
        # cyclic / ordinal edge metrics
        edge_lengths=ordm.edge_lengths,
        edge_length_cv=np.array(ordm.edge_length_cv),
        closure_edge_length=np.array(ordm.closure_edge_length or np.nan),
        closure_to_edge_ratio=np.array(ordm.closure_to_edge_ratio or np.nan),
        closure_to_path_ratio=np.array(ordm.closure_to_path_ratio or np.nan),
        geometry_kind=np.array(analysis.geometry_kind),
        labels=np.array(labels, dtype=object),
        # backward-compatible aliases
        points_pca=analysis.points_pca,
        pca_components=analysis.pca.components,
    )
    return path


def save_per_template_npz(
    path: str | Path,
    *,
    activations: np.ndarray,
    centroids: np.ndarray,
    concepts: list[str],
    templates: list[str],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        activations=activations,
        centroids=centroids,
        concepts=np.array(concepts, dtype=object),
        templates=np.array(templates, dtype=object),
    )
    return path


def save_metrics_json(path: str | Path, metrics: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    return path

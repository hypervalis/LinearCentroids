"""
Frenet-like moving-frame geometry for concept trajectories.

Treats activation or centroid vectors ``x_t ∈ R^D`` as vertices of a curve in
high-dimensional space (months: cyclic; years: typically linear).

Methodological rule
-------------------
All Frenet-like quantities are computed in the **original D-dimensional space**
first. PCA is only a visualization / projection lens. A second frame computed
**after** PCA projection is a contrast control — not the true geometry.

Process (full space)
--------------------
1. Diffs: ``d_t = x_{t+1} - x_t`` (cyclic wraps last→first).
2. Unit tangents: ``T_t = d_t / ||d_t||``.
3. Tangent change: ``ΔT_t = T_{t+1} - T_t`` (cyclic wrap).
4. Normal (raw): ``N_raw_t = ΔT_t - (ΔT_t·T_t) T_t``.
5. Unit normal: ``N_t = N_raw_t / ||N_raw_t||`` (NaN if norm ≤ EPS).
6. Curvature: ``κ_t = ||N_raw_t|| / ||d_t||``.

PCA projection
--------------
Fit PCA on full-space points. With ``P_k ∈ R^{D×k}`` (columns = top PCs):

- ``z_t = P_k^T (x_t - mean)``  implemented as ``(x - mean) @ P_k``
- Project diffs/tangents/normals: ``v @ P_k``

Concentration (per step)
------------------------
- ``ρ_d(t) = ||P_k^T d_t||² / ||d_t||²``
- ``ρ_T(t) = ||P_k^T T_t||²``
- ``ρ_N(t) = ||P_k^T N_t||²``

Frame alignment (projection fidelity)
------------------------------------
Compare **normalized** projected full-space tangents/normals to the frame
recomputed natively in PCA coordinates (cosine similarity per step).

Note on binormals
-----------------
In 3D plots one may use ``B = T × N`` **after** PCA projection. There is no
unique cross-product binormal in the original ``R^D`` space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.decomposition import PCA

GeometryKind = Literal["cyclic", "linear"]
EPS: float = 1e-12


@dataclass(frozen=True)
class TrajectoryFrame:
    """Discrete Frenet-like frame in ``R^D`` (or ``R^k`` for PCA-native contrast)."""

    points: np.ndarray  # (n_steps, d) — n_steps = T (cyclic) or T with diffs T-1 (linear)
    diffs: np.ndarray  # (n_edges, d)
    tangents: np.ndarray  # (n_edges, d) unit T_t
    delta_tangents: np.ndarray  # (n_edges, d) ΔT_t
    normals_raw: np.ndarray  # (n_edges, d) before normalization
    normals: np.ndarray  # (n_edges, d) unit N_t (NaN if undefined)
    curvature: np.ndarray  # (n_edges,) κ_t
    diff_norms: np.ndarray  # (n_edges,) ||d_t||
    normal_norms: np.ndarray  # (n_edges,) ||N_raw_t||


@dataclass(frozen=True)
class PCABasis:
    """
    Top-k PCA basis fitted on trajectory points.

    ``components`` rows are sklearn PCs (k × D). ``projection_matrix`` is
    ``P_k`` with shape (D, k) so ``projected = vectors @ P_k``.
    """

    mean: np.ndarray  # (D,)
    components: np.ndarray  # (k, D) rows = principal directions
    explained_variance_ratio: np.ndarray  # (k,)

    @property
    def k(self) -> int:
        return int(self.components.shape[0])

    @property
    def projection_matrix(self) -> np.ndarray:
        """P_k with shape (D, k)."""
        return self.components.T

    def project_points(self, points: np.ndarray) -> np.ndarray:
        """z_t = P_k^T (x_t - mean), shape (n, k)."""
        centered = np.asarray(points, dtype=np.float64) - self.mean
        return centered @ self.projection_matrix

    def project_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Linear part only: v @ P_k (no mean shift)."""
        return np.asarray(vectors, dtype=np.float64) @ self.projection_matrix


@dataclass(frozen=True)
class ConcentrationScores:
    """PCA visibility of full-space geometry (ρ_d, ρ_T, ρ_N)."""

    rho_diff: np.ndarray
    rho_tangent: np.ndarray
    rho_normal: np.ndarray


@dataclass(frozen=True)
class FrameComparison:
    """Alignment of projected full-space frame vs PCA-native contrast frame."""

    tangent_cosine: np.ndarray
    normal_cosine: np.ndarray
    tangent_angle_deg: np.ndarray
    normal_angle_deg: np.ndarray

    @property
    def tangent_alignment_full_vs_pca(self) -> np.ndarray:
        return self.tangent_cosine

    @property
    def normal_alignment_full_vs_pca(self) -> np.ndarray:
        return self.normal_cosine


@dataclass(frozen=True)
class OrdinalMetrics:
    """Cycle / ordinal edge geometry (months vs years)."""

    geometry_kind: GeometryKind
    edge_lengths: np.ndarray
    edge_length_cv: float
    closure_edge_length: float | None
    closure_to_edge_ratio: float | None
    closure_to_path_ratio: float | None
    curvature_smoothness_std: float
    curvature_smoothness_step_mean: float
    adjacent_neighbour_accuracy: float


@dataclass(frozen=True)
class TrajectoryAnalysis:
    """Full pipeline output: full-space frame, PCA projection, contrast frame."""

    frame: TrajectoryFrame
    pca: PCABasis
    points_pca: np.ndarray
    frame_projected: TrajectoryFrame
    frame_pca_native: TrajectoryFrame
    concentration: ConcentrationScores
    frame_comparison: FrameComparison
    curvature_summary: dict[str, float]
    ordinal: OrdinalMetrics
    geometry_kind: GeometryKind


def first_differences(points: np.ndarray, *, cyclic: bool = True) -> np.ndarray:
    """
    First differences along the concept sequence.

    Cyclic (months): ``d_t = x_{(t+1) mod T} - x_t``, shape (T, D).
    Linear (years): ``d_t = x_{t+1} - x_t``, shape (T-1, D).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"points must be 2-D, got shape {points.shape}")
    n = points.shape[0]
    if n < 2:
        raise ValueError("need at least 2 points")
    if cyclic:
        curr = np.arange(n)
        nxt = (curr + 1) % n
        return points[nxt] - points[curr]
    return points[1:] - points[:-1]


def _unit_rows(vectors: np.ndarray, *, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise normalization; rows with norm ≤ eps become NaN."""
    norms = np.linalg.norm(vectors, axis=1)
    out = np.full_like(vectors, np.nan)
    safe = norms > eps
    out[safe] = vectors[safe] / norms[safe, None]
    return out, norms


def _remove_parallel(delta: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Remove component of ``delta`` parallel to ``tangent`` (row-wise)."""
    coeff = np.sum(delta * tangent, axis=1, keepdims=True)
    return delta - coeff * tangent


def compute_trajectory_frame(
    points: np.ndarray,
    *,
    cyclic: bool = True,
    eps: float = EPS,
) -> TrajectoryFrame:
    """
    Compute discrete Frenet-like frame in full (or PCA) space.

    Steps
    -----
    1. ``d_t`` from ``first_differences``
    2. ``T_t = d_t / ||d_t||``
    3. ``ΔT_t = T_{t+1} - T_t`` (wrap if cyclic)
    4. ``N_raw_t = ΔT_t - (ΔT_t·T_t) T_t``
    5. ``N_t = N_raw_t / ||N_raw_t||`` (NaN if degenerate)
    6. ``κ_t = ||N_raw_t|| / ||d_t||``
    """
    points = np.asarray(points, dtype=np.float64)
    diffs = first_differences(points, cyclic=cyclic)
    tangents, diff_norms = _unit_rows(diffs, eps=eps)
    n_edges = diffs.shape[0]

    delta_tangents = np.full_like(diffs, np.nan)
    if cyclic:
        idx = np.arange(n_edges)
        delta_tangents = tangents[(idx + 1) % n_edges] - tangents[idx]
    elif n_edges >= 2:
        delta_tangents[:-1] = tangents[1:] - tangents[:-1]

    normals_raw = _remove_parallel(delta_tangents, tangents)
    normals, normal_norms = _unit_rows(normals_raw, eps=eps)
    with np.errstate(invalid="ignore", divide="ignore"):
        curvature = normal_norms / diff_norms
    curvature[~np.isfinite(curvature)] = np.nan
    curvature[diff_norms <= eps] = np.nan

    return TrajectoryFrame(
        points=points,
        diffs=diffs,
        tangents=tangents,
        delta_tangents=delta_tangents,
        normals_raw=normals_raw,
        normals=normals,
        curvature=curvature,
        diff_norms=diff_norms,
        normal_norms=normal_norms,
    )


def fit_pca(points: np.ndarray, n_components: int = 3) -> PCABasis:
    """Fit PCA on full-space points; return basis with ``projection_matrix`` (D × k)."""
    points = np.asarray(points, dtype=np.float64)
    n, d = points.shape
    k = min(n_components, n, d)
    pca = PCA(n_components=k)
    pca.fit(points)
    return PCABasis(
        mean=pca.mean_.astype(np.float64),
        components=pca.components_.astype(np.float64),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float64),
    )


def summarize_curvature(curvature: np.ndarray) -> dict[str, float]:
    """Scalar curvature summaries (mean, std, …)."""
    valid = curvature[np.isfinite(curvature)]
    if valid.size == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "max": nan, "median": nan, "total": nan}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "max": float(np.max(valid)),
        "median": float(np.median(valid)),
        "total": float(np.sum(valid)),
    }


def compute_edge_lengths(points: np.ndarray, *, geometry_kind: GeometryKind = "cyclic") -> np.ndarray:
    """``||d_t||`` for each adjacent edge (cyclic includes Dec→Jan as final edge)."""
    return np.linalg.norm(first_differences(points, cyclic=(geometry_kind == "cyclic")), axis=1)


def compute_edge_length_cv(edge_lengths: np.ndarray) -> float:
    return float(np.std(edge_lengths) / np.mean(edge_lengths)) if np.mean(edge_lengths) > 0 else float("nan")


def compute_closure_ratios(
    edge_lengths: np.ndarray,
    *,
    geometry_kind: GeometryKind = "cyclic",
) -> tuple[float | None, float | None, float | None]:
    """
    Cyclic closure metrics.

    ``closure_edge_length`` = length of final edge (x_0 relative to x_{T-1} via diffs).
    ``closure_to_edge_ratio`` ≈ 1 for a clean month cycle (not ≈ 0).
    """
    if geometry_kind != "cyclic" or edge_lengths.size < 2:
        return None, None, None
    closure = float(edge_lengths[-1])
    non_closure = edge_lengths[:-1]
    mean_non = float(np.mean(non_closure))
    sum_non = float(np.sum(non_closure))
    if mean_non <= 0 or sum_non <= 0:
        return closure, float("nan"), float("nan")
    return closure, closure / mean_non, closure / sum_non


def compute_curvature_smoothness(curvature: np.ndarray, *, cyclic: bool = True) -> tuple[float, float]:
    """Return (std(κ), mean |Δκ|) with cyclic wrap on step differences."""
    valid = curvature[np.isfinite(curvature)]
    smoothness_std = float(np.std(valid)) if valid.size else float("nan")
    n = curvature.shape[0]
    if cyclic and n >= 1:
        idx = np.arange(n)
        step_diff = np.abs(curvature[(idx + 1) % n] - curvature[idx])
    elif n >= 2:
        step_diff = np.abs(np.diff(curvature))
    else:
        return smoothness_std, float("nan")
    finite = step_diff[np.isfinite(step_diff)]
    return smoothness_std, float(np.mean(finite)) if finite.size else float("nan")


def compute_adjacent_neighbour_accuracy(points: np.ndarray, *, geometry_kind: GeometryKind = "cyclic") -> float:
    """Fraction of points whose nearest neighbour is prev/next in imposed order."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return float("nan")
    hits = 0
    for i in range(n):
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf
        j = int(np.argmin(dists))
        if geometry_kind == "cyclic":
            is_adj = j in ((i - 1) % n, (i + 1) % n)
        else:
            adj = {i - 1, i + 1} & set(range(n))
            is_adj = j in adj
        hits += int(is_adj)
    return float(hits / n)


def compute_ordinal_metrics(
    points: np.ndarray,
    curvature: np.ndarray,
    *,
    geometry_kind: GeometryKind = "cyclic",
) -> OrdinalMetrics:
    edge_lengths = compute_edge_lengths(points, geometry_kind=geometry_kind)
    closure_len, closure_edge, closure_path = compute_closure_ratios(
        edge_lengths, geometry_kind=geometry_kind
    )
    curv_std, curv_step = compute_curvature_smoothness(
        curvature, cyclic=(geometry_kind == "cyclic")
    )
    return OrdinalMetrics(
        geometry_kind=geometry_kind,
        edge_lengths=edge_lengths,
        edge_length_cv=compute_edge_length_cv(edge_lengths),
        closure_edge_length=closure_len,
        closure_to_edge_ratio=closure_edge,
        closure_to_path_ratio=closure_path,
        curvature_smoothness_std=curv_std,
        curvature_smoothness_step_mean=curv_step,
        adjacent_neighbour_accuracy=compute_adjacent_neighbour_accuracy(
            points, geometry_kind=geometry_kind
        ),
    )


def ordinal_metrics_to_dict(ordinal: OrdinalMetrics) -> dict:
    return {
        "geometry_kind": ordinal.geometry_kind,
        "edge_lengths": ordinal.edge_lengths.tolist(),
        "edge_length_cv": ordinal.edge_length_cv,
        "closure_edge_length": ordinal.closure_edge_length,
        "closure_to_edge_ratio": ordinal.closure_to_edge_ratio,
        "closure_to_path_ratio": ordinal.closure_to_path_ratio,
        "curvature_smoothness_std": ordinal.curvature_smoothness_std,
        "curvature_smoothness_step_mean": ordinal.curvature_smoothness_step_mean,
        "adjacent_neighbour_accuracy": ordinal.adjacent_neighbour_accuracy,
    }


def _rowwise_cosine(a: np.ndarray, b: np.ndarray, *, eps: float = EPS) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = num / np.maximum(denom, eps)
    cos = np.clip(cos, -1.0, 1.0)
    invalid = (~np.isfinite(a).all(axis=1)) | (~np.isfinite(b).all(axis=1)) | (denom <= eps)
    cos[invalid] = np.nan
    return cos


def compute_concentration(frame: TrajectoryFrame, basis: PCABasis) -> ConcentrationScores:
    """
    Concentration scores ρ_d, ρ_T, ρ_N.

    Uses ``v @ P_k`` without forming D×D projection matrices.
    """
    proj_diff = basis.project_vectors(frame.diffs)
    proj_tangent = basis.project_vectors(frame.tangents)
    proj_normal = basis.project_vectors(frame.normals)

    diff_norm_sq = frame.diff_norms ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_diff = np.sum(proj_diff**2, axis=1) / diff_norm_sq
    rho_diff[~np.isfinite(rho_diff)] = np.nan
    rho_diff[diff_norm_sq <= 0] = np.nan

    return ConcentrationScores(
        rho_diff=rho_diff,
        rho_tangent=np.sum(proj_tangent**2, axis=1),
        rho_normal=np.sum(proj_normal**2, axis=1),
    )


def compare_frames(projected: TrajectoryFrame, pca_native: TrajectoryFrame) -> FrameComparison:
    """
    Compare projected full-space frame to PCA-native contrast frame.

    Projected full-space vectors are **re-normalized** in PCA coordinates before
    cosine similarity (direction-only comparison).
    """
    proj_t, _ = _unit_rows(projected.tangents)
    proj_n, _ = _unit_rows(projected.normals)
    t_cos = _rowwise_cosine(proj_t, pca_native.tangents)
    n_cos = _rowwise_cosine(proj_n, pca_native.normals)
    return FrameComparison(
        tangent_cosine=t_cos,
        normal_cosine=n_cos,
        tangent_angle_deg=np.degrees(np.arccos(np.clip(np.abs(t_cos), -1, 1))),
        normal_angle_deg=np.degrees(np.arccos(np.clip(np.abs(n_cos), -1, 1))),
    )


def analyze_trajectory(
    points: np.ndarray,
    *,
    n_pca_components: int = 3,
    geometry_kind: GeometryKind = "cyclic",
) -> TrajectoryAnalysis:
    """
    End-to-end Frenet-like analysis.

    1. Full-space frame (true geometry)
    2. PCA on points
    3. Project full-space diffs/tangents/normals into PCA
    4. Recompute frame on PCA coordinates (contrast only)
    5. Concentration + frame alignment + ordinal metrics
    """
    cyclic = geometry_kind == "cyclic"
    frame = compute_trajectory_frame(points, cyclic=cyclic)
    basis = fit_pca(points, n_components=n_pca_components)
    points_pca = basis.project_points(points)

    projected = TrajectoryFrame(
        points=points_pca,
        diffs=basis.project_vectors(frame.diffs),
        tangents=basis.project_vectors(frame.tangents),
        delta_tangents=basis.project_vectors(frame.delta_tangents),
        normals_raw=basis.project_vectors(frame.normals_raw),
        normals=basis.project_vectors(frame.normals),
        curvature=frame.curvature.copy(),
        diff_norms=frame.diff_norms.copy(),
        normal_norms=frame.normal_norms.copy(),
    )

    frame_pca_native = compute_trajectory_frame(points_pca, cyclic=cyclic)
    concentration = compute_concentration(frame, basis)
    comparison = compare_frames(projected, frame_pca_native)
    ordinal = compute_ordinal_metrics(points, frame.curvature, geometry_kind=geometry_kind)

    return TrajectoryAnalysis(
        frame=frame,
        pca=basis,
        points_pca=points_pca,
        frame_projected=projected,
        frame_pca_native=frame_pca_native,
        concentration=concentration,
        frame_comparison=comparison,
        curvature_summary=summarize_curvature(frame.curvature),
        ordinal=ordinal,
        geometry_kind=geometry_kind,
    )


def analysis_to_dict(analysis: TrajectoryAnalysis, labels: list[str] | None = None) -> dict:
    """JSON-serializable summary including concentration and frame alignment."""
    n = analysis.frame.points.shape[0]
    labels = labels or [str(i) for i in range(n)]
    fc = analysis.frame_comparison
    conc = analysis.concentration
    return {
        "labels": labels,
        "geometry_kind": analysis.geometry_kind,
        "curvature_summary": analysis.curvature_summary,
        "ordinal": ordinal_metrics_to_dict(analysis.ordinal),
        "pca_explained_variance_ratio": analysis.pca.explained_variance_ratio.tolist(),
        "curvature": analysis.frame.curvature.tolist(),
        "diff_concentration": conc.rho_diff.tolist(),
        "tangent_concentration": conc.rho_tangent.tolist(),
        "normal_concentration": conc.rho_normal.tolist(),
        "rho_diff": conc.rho_diff.tolist(),
        "rho_tangent": conc.rho_tangent.tolist(),
        "rho_normal": conc.rho_normal.tolist(),
        "tangent_alignment_full_vs_pca": fc.tangent_alignment_full_vs_pca.tolist(),
        "normal_alignment_full_vs_pca": fc.normal_alignment_full_vs_pca.tolist(),
        "tangent_frame_angle_deg": fc.tangent_angle_deg.tolist(),
        "normal_frame_angle_deg": fc.normal_angle_deg.tolist(),
        "mean_rho_diff": float(np.nanmean(conc.rho_diff)),
        "mean_rho_tangent": float(np.nanmean(conc.rho_tangent)),
        "mean_rho_normal": float(np.nanmean(conc.rho_normal)),
        "mean_tangent_alignment_full_vs_pca": float(np.nanmean(fc.tangent_alignment_full_vs_pca)),
        "mean_normal_alignment_full_vs_pca": float(np.nanmean(fc.normal_alignment_full_vs_pca)),
        "mean_tangent_frame_angle_deg": float(np.nanmean(fc.tangent_angle_deg)),
        "mean_normal_frame_angle_deg": float(np.nanmean(fc.normal_angle_deg)),
    }


# --- Synthetic trajectories ---


def synthetic_line(n: int, dim: int, *, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Cyclic affine progression (constant tangent → κ ≈ 0)."""
    direction = np.zeros(dim, dtype=np.float64)
    direction[0] = 1.0
    points = np.arange(n, dtype=np.float64)[:, None] * direction[None, :]
    if noise > 0:
        points += noise * np.random.default_rng(seed).standard_normal(points.shape)
    return points


def synthetic_circle(n: int, dim: int, *, radius: float = 1.0, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Circle embedded in first two coordinates (cyclic validation)."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    points = np.zeros((n, dim), dtype=np.float64)
    points[:, 0] = radius * np.cos(angles)
    points[:, 1] = radius * np.sin(angles)
    if noise > 0:
        points += noise * np.random.default_rng(seed).standard_normal(points.shape)
    return points


def synthetic_random_walk(n: int, dim: int, *, step_scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """Cyclic-indexed random walk (high-D, irregular geometry)."""
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((n, dim))
    steps /= np.maximum(np.linalg.norm(steps, axis=1, keepdims=True), EPS)
    return step_scale * np.cumsum(steps, axis=0)


def synthetic_linear_ordinal(n: int, dim: int, *, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Open ordinal line (non-cyclic; years-like)."""
    direction = np.zeros(dim, dtype=np.float64)
    direction[0] = 1.0
    points = np.arange(n, dtype=np.float64)[:, None] * direction[None, :]
    if noise > 0:
        points += noise * np.random.default_rng(seed).standard_normal(points.shape)
    return points

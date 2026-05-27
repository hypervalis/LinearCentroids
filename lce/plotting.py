"""Plotly figures for trajectory analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lce.geometry import TrajectoryAnalysis, analyze_trajectory


def _unit_rows(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = (norms > eps).reshape(-1)
    out = np.zeros_like(vectors)
    out[safe] = vectors[safe] / norms[safe]
    return out


def _pca_dirs(frame_vectors: np.ndarray, k: int) -> np.ndarray:
    """Unit directions of frame vectors in the first ``k`` PCA coordinates."""
    return _unit_rows(frame_vectors[:, :k])


def _arrow_line_segments(
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    scale: float,
) -> tuple[
    list[float | None],
    list[float | None],
    list[float | None],
    list[float],
    list[float],
    list[float],
]:
    """Build Plotly 3-D line segments and arrow-tip coordinates."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    tip_x: list[float] = []
    tip_y: list[float] = []
    tip_z: list[float] = []
    for origin, direction in zip(origins, directions):
        if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= 1e-12:
            continue
        direction = direction / np.linalg.norm(direction)
        end = origin + scale * direction
        xs.extend([float(origin[0]), float(end[0]), None])
        ys.extend([float(origin[1]), float(end[1]), None])
        zs.extend([float(origin[2]), float(end[2]), None])
        tip_x.append(float(end[0]))
        tip_y.append(float(end[1]))
        tip_z.append(float(end[2]))
    return xs, ys, zs, tip_x, tip_y, tip_z


def _arrow_scale(points: np.ndarray) -> float:
    """Arrow length relative to PCA scatter extent."""
    if len(points) < 2:
        return 0.15
    edge = np.linalg.norm(np.diff(points, axis=0), axis=1)
    base = float(np.median(edge)) if edge.size else float(np.ptp(points, axis=0).max())
    extent = float(np.ptp(points, axis=0).max()) if points.size else base
    return max(0.22 * base, 0.12 * extent, 1e-3)


def _axis_range_from_data(
    values: np.ndarray,
    *,
    padding: float = 1.5,
) -> tuple[float, float]:
    """Data min/max with a little fixed padding for labels and arrow tips."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-padding, padding)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo <= 1e-12:
        half = max(abs(lo), 1.0) * 0.05
        return lo - half, hi + half
    return lo - padding, hi + padding


def _scene_axis(title: str, axis_range: tuple[float, float]) -> dict:
    """Single 3-D scene axis with solid box faces, grid, and edge lines."""
    return dict(
        title=title,
        range=list(axis_range),
        autorange=False,
        showbackground=True,
        backgroundcolor="#E5ECF6",
        showgrid=True,
        gridcolor="white",
        gridwidth=2,
        showline=True,
        linecolor="white",
        linewidth=2,
        zeroline=False,
    )


def _scene_axis_layout(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    extra_x: list[float] | None = None,
    extra_y: list[float] | None = None,
    extra_z: list[float] | None = None,
    *,
    padding: float = 1.5,
) -> dict:
    """Axis limits: points + optional arrow tips, with small fixed padding."""
    all_x = np.concatenate([xs, np.asarray(extra_x or [], dtype=np.float64)])
    all_y = np.concatenate([ys, np.asarray(extra_y or [], dtype=np.float64)])
    all_z = np.concatenate([zs, np.asarray(extra_z or [], dtype=np.float64)])
    return {
        "xaxis": _scene_axis("PC1", _axis_range_from_data(all_x, padding=padding)),
        "yaxis": _scene_axis("PC2", _axis_range_from_data(all_y, padding=padding)),
        "zaxis": _scene_axis("PC3", _axis_range_from_data(all_z, padding=padding)),
        "domain": dict(x=(0.04, 0.96), y=(0.02, 0.92)),
        "aspectmode": "data",
    }


def trajectory_pca_3d_figure(
    analysis: TrajectoryAnalysis,
    labels: list[str],
    *,
    title: str,
    centroid_arrows: np.ndarray | None = None,
) -> go.Figure:
    """
    3-D PCA path through concept points (ρ_d on markers).

    When ``centroid_arrows`` is supplied (full-space μ(h), one row per label), draw
    those as arrows anchored at each activation point — local MLP direction in the
    PCA view. Otherwise show the path only (no Frenet tangents/normals).
    """
    pts = analysis.points_pca
    k = min(3, pts.shape[1])
    coords = np.column_stack([pts[:, 0], pts[:, 1], pts[:, 2] if k > 2 else np.zeros(len(pts))])
    rho = analysis.concentration.rho_diff
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=6, color=rho, colorscale="Viridis", showscale=True, colorbar_title="ρ_d"),
            line=dict(width=0, color="rgba(80,80,80,0.6)"),
            name="path",
        )
    )

    subtitle = "Markers coloured by ρ_d (transition visibility in PCA)"
    extra_x: list[float] = []
    extra_y: list[float] = []
    extra_z: list[float] = []
    if centroid_arrows is not None:
        centroid_arrows = np.asarray(centroid_arrows, dtype=np.float64)
        if centroid_arrows.shape[0] != len(labels):
            raise ValueError("centroid_arrows must have one row per label")
        projected = analysis.pca.project_vectors(centroid_arrows)[:, :k]
        directions = _pca_dirs(projected, k)
        scale = _arrow_scale(coords)
        ax, ay, az, tx, ty, tz = _arrow_line_segments(coords, directions, scale=scale)
        extra_x = [v for v in ax if v is not None] + tx
        extra_y = [v for v in ay if v is not None] + ty
        extra_z = [v for v in az if v is not None] + tz
        arrow_color = "#ff4500"
        if ax:
            fig.add_trace(
                go.Scatter3d(
                    x=ax,
                    y=ay,
                    z=az,
                    mode="lines",
                    line=dict(width=7, color=arrow_color),
                    name="centroid μ(h)",
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=tx,
                    y=ty,
                    z=tz,
                    mode="markers",
                    marker=dict(size=5, color=arrow_color, symbol="diamond"),
                    name="μ tip",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        subtitle = "Points = activation h · orange arrows = centroid μ(h) projected into PCA"

    fig.update_layout(
        title=f"{title}<br><sup>{subtitle}</sup>",
        height=560,
        scene=_scene_axis_layout(xs, ys, zs, extra_x, extra_y, extra_z),
        margin=dict(l=8, r=8, t=72, b=8),
        showlegend=True,
        legend=dict(x=0, y=1),
    )
    return fig


def make_pca3d_figure(
    points: np.ndarray,
    labels: list[str],
    *,
    geometry_kind: str,
    n_pca_components: int,
    title_prefix: str,
    centroid_arrows: np.ndarray | None = None,
) -> go.Figure:
    """Build PCA 3-D path plot; optional centroid arrows at each point."""
    analysis = analyze_trajectory(
        points, n_pca_components=n_pca_components, geometry_kind=geometry_kind
    )
    return trajectory_pca_3d_figure(
        analysis,
        labels,
        title=f"{title_prefix} — 3D PCA",
        centroid_arrows=centroid_arrows,
    )


def curvature_figure(analysis: TrajectoryAnalysis, labels: list[str], *, title: str) -> go.Figure:
    kappa = analysis.frame.curvature
    fig = go.Figure(
        go.Scatter(
            x=labels,
            y=kappa,
            mode="lines+markers",
            name="curvature",
        )
    )
    fig.update_layout(title=title, xaxis_title="concept", yaxis_title="κ")
    return fig


def edge_closure_figure(analysis: TrajectoryAnalysis, labels: list[str], *, title: str) -> go.Figure:
    o = analysis.ordinal
    if o.geometry_kind == "cyclic":
        edge_labels = [f"{labels[i]}→{labels[(i+1) % len(labels)]}" for i in range(len(labels))]
    else:
        edge_labels = [f"{labels[i]}→{labels[i+1]}" for i in range(len(labels) - 1)]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("adjacent edge lengths", "summary scalars"),
        row_heights=[0.65, 0.35],
    )
    fig.add_trace(go.Bar(x=edge_labels, y=o.edge_lengths, name="edge length"), row=1, col=1)
    summary_text = (
        f"edge_length_cv = {o.edge_length_cv:.4f}<br>"
        f"closure_to_edge_ratio = {o.closure_to_edge_ratio}<br>"
        f"closure_to_path_ratio = {o.closure_to_path_ratio}<br>"
        f"adjacent_neighbour_accuracy = {o.adjacent_neighbour_accuracy:.4f}"
    )
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip"),
        row=2,
        col=1,
    )
    fig.add_annotation(
        text=summary_text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.12,
        showarrow=False,
        align="left",
    )
    fig.update_layout(title=title, showlegend=False, height=520)
    fig.update_xaxes(tickangle=45, row=1, col=1)
    return fig


def template_variance_figure(
    variance: dict,
    *,
    title: str,
) -> go.Figure:
    act = variance["activation"]["variance_per_concept"]
    cen = variance["centroid"]["variance_per_concept"]
    months = list(range(1, len(act) + 1))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=months, y=act, name="activation"))
    fig.add_trace(go.Bar(x=months, y=cen, name="centroid"))
    fig.update_layout(
        title=title,
        xaxis_title="month index",
        yaxis_title="mean template variance",
        barmode="group",
    )
    return fig


def save_figure_html(fig: go.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inner = fig.to_html(include_plotlyjs="cdn", full_html=False, config={"responsive": True})
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #fff; }}
    body > div {{ width: 100%; height: 100%; }}
    .plotly-graph-div {{ width: 100% !important; height: 100% !important; min-height: 28rem; }}
  </style>
</head>
<body>
{inner}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
    return path


def make_analysis_figures(
    points: np.ndarray,
    labels: list[str],
    *,
    geometry_kind: str,
    n_pca_components: int,
    title_prefix: str,
) -> tuple[TrajectoryAnalysis, dict[str, go.Figure]]:
    analysis = analyze_trajectory(
        points, n_pca_components=n_pca_components, geometry_kind=geometry_kind
    )
    figures = {
        "pca3d": trajectory_pca_3d_figure(
            analysis, labels, title=f"{title_prefix} — 3D PCA"
        ),
        "curvature": curvature_figure(analysis, labels, title=f"{title_prefix} — curvature"),
        "edge_closure": edge_closure_figure(
            analysis, labels, title=f"{title_prefix} — edges & closure"
        ),
    }
    return analysis, figures

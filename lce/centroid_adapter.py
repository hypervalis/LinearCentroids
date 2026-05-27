"""Centroid computation via the paper repo or a local autograd fallback.

All repo code should call ``compute_centroid`` from this module only.
Do not import from ``third_party`` elsewhere in the project.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PAPER_REPO_DIR = Path(__file__).resolve().parents[1] / "third_party" / "LinearCentroidsHypothesis"
PAPER_CENTROIDS_FILE = PAPER_REPO_DIR / "centroids.py"

# ---------------------------------------------------------------------------
# TODO: third-party wrappers (extraction layer only — not used by compute_centroid)
#
# When third_party/LinearCentroidsHypothesis is vendored, the paper repo exposes:
#   - centroids.Centroids(model): forward pass + Jacobian w.r.t. model input x
#   - centroids.LocalCentroids(model): neighborhood-averaged centroids
#   - centroids.InternalModuleWrapper(model, target_module_name): hook a submodule
#
# Extraction (lce/extraction.py) may wrap those classes for full-model workflows.
# This module keeps a tensor-level API: compute_centroid(output, input).
# The paper's Centroids._compute_centroids_grad uses the same autograd pattern as
# our fallback below.
# ---------------------------------------------------------------------------


def has_third_party_centroid_code() -> bool:
    """Return True when the paper repository is present and importable."""
    if not PAPER_CENTROIDS_FILE.is_file():
        return False
    try:
        return _import_paper_centroids_module() is not None
    except Exception:
        return False


def _import_paper_centroids_module():
    """Load centroids.py from the vendored paper repo (adapter-internal only)."""
    if not PAPER_CENTROIDS_FILE.is_file():
        return None
    paper_dir = str(PAPER_REPO_DIR)
    if paper_dir not in sys.path:
        sys.path.insert(0, paper_dir)
    try:
        import centroids  # type: ignore import-not-found

        return centroids
    except ImportError:
        return None


def _validate_grad_context(component_input: torch.Tensor) -> None:
    if not torch.is_grad_enabled():
        raise RuntimeError(
            "Gradients are disabled. Enable them with:\n"
            "  with torch.enable_grad():\n"
            "      ... compute_centroid(...)"
        )
    if not component_input.requires_grad:
        raise ValueError(
            "component_input.requires_grad is False. Centroids need "
            "∇_input Σ_k f(input)_k; call component_input = "
            "component_input.detach().requires_grad_(True) (or equivalent) "
            "before the forward pass that produces component_output."
        )


def compute_centroid(
    component_output: torch.Tensor,
    component_input: torch.Tensor,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> torch.Tensor:
    """
    Compute μ(x) = ∇_x Σ_k f(x)_k for a component output f(x).

    Parameters
    ----------
    component_output:
        Tensor f(x) from the chosen sub-component.
    component_input:
        Tensor x with ``requires_grad=True`` used as the Jacobian input.
    retain_graph, create_graph:
        Passed through to ``torch.autograd.grad``.

    Returns
    -------
    torch.Tensor
        Centroid vector(s), same shape as ``component_input``.
    """
    _validate_grad_context(component_input)

    grad_outputs = torch.ones_like(component_output)
    centroid = torch.autograd.grad(
        outputs=component_output,
        inputs=component_input,
        grad_outputs=grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=False,
    )[0]
    return centroid

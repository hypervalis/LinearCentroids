"""Template-level variance and representation comparison metrics."""

from __future__ import annotations

import numpy as np


def compute_template_variance(per_template_vectors: np.ndarray) -> dict[str, float | list[float]]:
    """
    Variance across prompt templates for each concept.

    Parameters
    ----------
    per_template_vectors : (n_templates, n_concepts, dim)
        One vector per (template, concept), e.g. MLP-input activations or MLP-local centroids.
    """
    arr = np.asarray(per_template_vectors, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected shape (n_templates, n_concepts, dim), got {arr.shape}")

    var_per_concept_dim = np.var(arr, axis=0)
    var_per_concept = np.mean(var_per_concept_dim, axis=1)
    return {
        "variance_per_concept": var_per_concept.tolist(),
        "mean_template_variance": float(np.mean(var_per_concept)),
        "std_template_variance": float(np.std(var_per_concept)),
        "total_template_variance": float(np.sum(var_per_concept_dim)),
    }


def compare_template_variance(
    activation_templates: np.ndarray,
    centroid_templates: np.ndarray,
) -> dict:
    """
    Compare template variance for activations vs MLP-local centroids.

    Returns both summaries and their difference (centroid minus activation).
    Negative mean_difference => centroids vary less across templates.
    """
    act = compute_template_variance(activation_templates)
    cen = compute_template_variance(centroid_templates)
    return {
        "activation": act,
        "centroid": cen,
        "mean_difference_centroid_minus_activation": float(
            cen["mean_template_variance"] - act["mean_template_variance"]
        ),
    }

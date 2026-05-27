"""Tests for template variance metrics."""

from __future__ import annotations

import numpy as np

from lce.metrics import compare_template_variance, compute_template_variance


def test_template_variance_zero_when_identical_templates():
    # (templates, months, dim)
    vectors = np.ones((3, 12, 4))
    out = compute_template_variance(vectors)
    assert out["mean_template_variance"] == 0.0


def test_compare_template_variance():
    rng = np.random.default_rng(0)
    act = rng.standard_normal((3, 12, 8))
    cen = act + 0.1 * rng.standard_normal((3, 12, 8))
    out = compare_template_variance(act, cen)
    assert "activation" in out and "centroid" in out
    assert out["centroid"]["mean_template_variance"] > out["activation"]["mean_template_variance"]

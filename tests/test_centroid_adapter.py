"""Tests for lce.centroid_adapter."""

from __future__ import annotations

import torch
import pytest

from lce.centroid_adapter import compute_centroid, has_third_party_centroid_code


def test_has_third_party_false_when_not_vendored():
    # Repo scaffold may not include the submodule checkout.
    assert isinstance(has_third_party_centroid_code(), bool)


def test_compute_centroid_linear_map():
    x = torch.tensor([[2.0, 3.0]], requires_grad=True)
    weight = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    y = x @ weight.T
    mu = compute_centroid(y, x)
    # row-sum of Jacobian for y_i = sum_j W_ij x_j  ->  mu_j = sum_i W_ij
    expected = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(mu, expected)


def test_compute_centroid_requires_grad_false():
    x = torch.tensor([[1.0]], requires_grad=False)
    y = x * 2
    with pytest.raises(ValueError, match="requires_grad"):
        compute_centroid(y, x)


def test_compute_centroid_grad_disabled():
    x = torch.tensor([[1.0]], requires_grad=True)
    y = x * 2
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="Gradients are disabled"):
            compute_centroid(y, x)


def test_compute_centroid_retain_graph():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    y = x.pow(2)
    mu1 = compute_centroid(y, x, retain_graph=True)
    mu2 = compute_centroid(y, x)
    torch.testing.assert_close(mu1, mu2)

"""Tests for concept registry."""

from __future__ import annotations

from lce.registry import CONCEPTS, concept_from_config, get_concept


def test_registry_covers_all_config_concepts():
    expected = {
        "months",
        "colours",
        "nouns",
        "tools",
        "animals",
        "years",
        "disciplines",
        "whwords",
        "stopwords",
        "languages",
    }
    assert set(CONCEPTS) == expected


def test_months_use_geometry_pipeline_only():
    spec = get_concept("months")
    assert spec.uses_month_geometry
    assert spec.orderings is None


def test_ordering_concepts_have_metrics_filename():
    for name, spec in CONCEPTS.items():
        if name == "months":
            continue
        assert spec.orderings is not None
        assert spec.metrics_filename == f"{name}_orderings.json"


def test_concept_from_config_defaults_to_months():
    assert concept_from_config({}).name == "months"

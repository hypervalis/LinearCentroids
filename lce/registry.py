"""Concept registry: orderings, prompt writers, and pipeline metadata."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lce import concepts as c


@dataclass(frozen=True)
class ConceptSpec:
    """Pipeline metadata for one ``concept_sequence`` value in configs."""

    name: str
    orderings: Mapping[str, tuple[str, ...]] | None = None
    ordering_eval_modes: Mapping[str, tuple[tuple[str, bool], ...]] | None = None
    prompt_families: Mapping[str, tuple[str, ...]] | None = None
    write_prompts: Callable[..., Path] | None = None
    metrics_filename: str | None = None
    uses_month_geometry: bool = False


def _ordering_spec(
    name: str,
    *,
    orderings: Mapping[str, tuple[str, ...]],
    eval_modes: Mapping[str, tuple[tuple[str, bool], ...]],
    prompt_families: Mapping[str, tuple[str, ...]],
    write_prompts: Callable[..., Path],
) -> ConceptSpec:
    return ConceptSpec(
        name=name,
        orderings=orderings,
        ordering_eval_modes=eval_modes,
        prompt_families=prompt_families,
        write_prompts=write_prompts,
        metrics_filename=f"{name}_orderings.json",
    )


CONCEPTS: dict[str, ConceptSpec] = {
    "months": ConceptSpec(name="months", uses_month_geometry=True),
    "colours": _ordering_spec(
        "colours",
        orderings=c.COLOUR_ORDERINGS,
        eval_modes=c.COLOUR_ORDERING_EVAL_MODES,
        prompt_families=c.COLOUR_PROMPT_FAMILIES,
        write_prompts=c.write_colour_prompts,
    ),
    "nouns": _ordering_spec(
        "nouns",
        orderings=c.NOUN_ORDERINGS,
        eval_modes=c.NOUN_ORDERING_EVAL_MODES,
        prompt_families=c.NOUN_PROMPT_FAMILIES,
        write_prompts=c.write_noun_prompts,
    ),
    "tools": _ordering_spec(
        "tools",
        orderings=c.TOOL_ORDERINGS,
        eval_modes=c.TOOL_ORDERING_EVAL_MODES,
        prompt_families=c.TOOL_PROMPT_FAMILIES,
        write_prompts=c.write_tool_prompts,
    ),
    "animals": _ordering_spec(
        "animals",
        orderings=c.ANIMAL_ORDERINGS,
        eval_modes=c.ANIMAL_ORDERING_EVAL_MODES,
        prompt_families=c.ANIMAL_PROMPT_FAMILIES,
        write_prompts=c.write_animal_prompts,
    ),
    "years": _ordering_spec(
        "years",
        orderings=c.YEAR_ORDERINGS,
        eval_modes=c.YEAR_ORDERING_EVAL_MODES,
        prompt_families=c.YEAR_PROMPT_FAMILIES,
        write_prompts=c.write_year_prompts,
    ),
    "disciplines": _ordering_spec(
        "disciplines",
        orderings=c.DISCIPLINE_ORDERINGS,
        eval_modes=c.DISCIPLINE_ORDERING_EVAL_MODES,
        prompt_families=c.DISCIPLINE_PROMPT_FAMILIES,
        write_prompts=c.write_discipline_prompts,
    ),
    "whwords": _ordering_spec(
        "whwords",
        orderings=c.WHWORD_ORDERINGS,
        eval_modes=c.WHWORD_ORDERING_EVAL_MODES,
        prompt_families=c.WHWORD_PROMPT_FAMILIES,
        write_prompts=c.write_whword_prompts,
    ),
    "stopwords": _ordering_spec(
        "stopwords",
        orderings=c.STOPWORD_ORDERINGS,
        eval_modes=c.STOPWORD_ORDERING_EVAL_MODES,
        prompt_families=c.STOPWORD_PROMPT_FAMILIES,
        write_prompts=c.write_stopword_prompts,
    ),
    "languages": _ordering_spec(
        "languages",
        orderings=c.LANGUAGE_ORDERINGS,
        eval_modes=c.LANGUAGE_ORDERING_EVAL_MODES,
        prompt_families=c.LANGUAGE_PROMPT_FAMILIES,
        write_prompts=c.write_language_prompts,
    ),
}


def get_concept(name: str) -> ConceptSpec:
    if name not in CONCEPTS:
        known = ", ".join(sorted(CONCEPTS))
        raise KeyError(f"unknown concept_sequence {name!r}; known: {known}")
    return CONCEPTS[name]


def concept_from_config(cfg: dict[str, Any]) -> ConceptSpec:
    return get_concept(cfg.get("concept_sequence", "months"))


def default_ordering_names(spec: ConceptSpec) -> list[str]:
    if spec.orderings is None:
        return []
    return list(spec.orderings.keys())


def default_prompt_output(spec: ConceptSpec, *, prompt_family: str = "neutral") -> Path:
    if spec.name == "months":
        return Path("data/prompts/months.json")
    return Path(f"data/prompts/{spec.name}_{prompt_family}.jsonl")

"""Concept sequences, colour orderings, and prompt templates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

PromptFamily = Literal["neutral", "perceptual", "rainbow", "semantic"]

MONTHS: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

# v1: month name is the final token — no trailing punctuation or suffix text.
DEFAULT_MONTH_TEMPLATES: tuple[str, ...] = (
    "The month is {month}",
    "The calendar says {month}",
    "The event happens in {month}",
)

# --- Colour orderings (single-word labels for final-token extraction) ---

# Fixed comparison set: every named ordering is a permutation of exactly these 7 colours.
COLOUR_UNIVERSAL_SET: tuple[str, ...] = (
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "indigo",
    "violet",
)

ROYGBIV: tuple[str, ...] = COLOUR_UNIVERSAL_SET

# Extended reference lists (not used directly — orderings below permute COLOUR_UNIVERSAL_SET).
HUE_WHEEL_EXTENDED: tuple[str, ...] = (
    "red",
    "orange",
    "yellow",
    "lime",
    "green",
    "teal",
    "cyan",
    "azure",
    "blue",
    "indigo",
    "violet",
    "purple",
    "magenta",
    "pink",
)

# Each ordering: same 7 colours, different imposed order (fair path comparison).
COLOUR_ORDERINGS: dict[str, tuple[str, ...]] = {
    "roygbiv": ("red", "orange", "yellow", "green", "blue", "indigo", "violet"),
    "hue_wheel_approx": ("red", "orange", "yellow", "green", "blue", "violet", "indigo"),
    "basic_colours": ("blue", "red", "green", "yellow", "orange", "violet", "indigo"),
    "warm_to_cool": ("red", "orange", "yellow", "violet", "blue", "indigo", "green"),
    "light_to_dark_approx": ("yellow", "orange", "green", "blue", "indigo", "violet", "red"),
}

# How each named ordering should be evaluated (mode name, cyclic flag).
COLOUR_ORDERING_EVAL_MODES: dict[str, tuple[tuple[str, bool], ...]] = {
    "roygbiv": (("non_cyclic", False), ("cyclic_diagnostic", True)),
    "hue_wheel_approx": (("cyclic", True), ("non_cyclic_diagnostic", False)),
    "warm_to_cool": (("non_cyclic", False),),
    "light_to_dark_approx": (("non_cyclic", False),),
    "basic_colours": (("non_cyclic", False),),
}

COLOUR_PROMPT_FAMILIES: dict[str, tuple[str, ...]] = {
    "neutral": (
        "The colour is {value}",
        "The object is {value}",
        "The paint is {value}",
        "The card is {value}",
        "The word is {value}",
    ),
    "perceptual": (
        "The hue is {value}",
        "The light appears {value}",
        "The wavelength is perceived as {value}",
        "The colour patch looks {value}",
    ),
    "rainbow": (
        "In the rainbow, the colour is {value}",
        "The rainbow contains {value}",
        "A rainbow band is {value}",
    ),
    "semantic": (
        "The warning sign is {value}",
        "The flag colour is {value}",
        "The clothing is {value}",
        "The symbol is {value}",
    ),
}

# TODO: robust concept-token span extraction when templates are not final-token aligned.


def all_colour_labels() -> tuple[str, ...]:
    """Default extraction/comparison colour labels (fixed universal set)."""
    return COLOUR_UNIVERSAL_SET


def validate_colour_orderings() -> None:
    """Ensure every named ordering uses exactly the same colour set."""
    universal = set(COLOUR_UNIVERSAL_SET)
    n = len(COLOUR_UNIVERSAL_SET)
    for name, ordering in COLOUR_ORDERINGS.items():
        if len(ordering) != n:
            raise ValueError(f"{name}: expected {n} colours, got {len(ordering)}")
        if set(ordering) != universal:
            raise ValueError(f"{name}: must be a permutation of COLOUR_UNIVERSAL_SET")


validate_colour_orderings()


# --- Random nouns (unrelated common nouns; no natural order) ---

# Fixed comparison set: every ordering is a permutation of exactly these 7 nouns.
NOUN_UNIVERSAL_SET: tuple[str, ...] = (
    "cloud",
    "hammer",
    "ladder",
    "mirror",
    "onion",
    "pocket",
    "river",
)

# Arbitrary imposed list order (not semantically motivated).
NOUN_ORDERINGS: dict[str, tuple[str, ...]] = {
    "list_order": ("hammer", "river", "cloud", "ladder", "pocket", "mirror", "onion"),
}

NOUN_ORDERING_EVAL_MODES: dict[str, tuple[tuple[str, bool], ...]] = {
    "list_order": (("non_cyclic", False),),
}

NOUN_PROMPT_FAMILIES: dict[str, tuple[str, ...]] = {
    "neutral": (
        "The noun is {value}",
        "The word is {value}",
        "The object is {value}",
        "The thing is {value}",
        "The name is {value}",
    ),
}


def all_noun_labels() -> tuple[str, ...]:
    """Default extraction/comparison noun labels (fixed universal set)."""
    return NOUN_UNIVERSAL_SET


def validate_noun_orderings() -> None:
    """Ensure every named ordering uses exactly the same noun set."""
    universal = set(NOUN_UNIVERSAL_SET)
    n = len(NOUN_UNIVERSAL_SET)
    for name, ordering in NOUN_ORDERINGS.items():
        if len(ordering) != n:
            raise ValueError(f"{name}: expected {n} nouns, got {len(ordering)}")
        if set(ordering) != universal:
            raise ValueError(f"{name}: must be a permutation of NOUN_UNIVERSAL_SET")


validate_noun_orderings()


# --- Tools (related objects; no natural order) ---

TOOL_UNIVERSAL_SET: tuple[str, ...] = (
    "hammer",
    "saw",
    "drill",
    "wrench",
    "knife",
    "brush",
    "needle",
    "scissors",
    "pen",
    "computer",
)

TOOL_ORDERINGS: dict[str, tuple[str, ...]] = {
    "list_order": TOOL_UNIVERSAL_SET,
}

TOOL_ORDERING_EVAL_MODES: dict[str, tuple[tuple[str, bool], ...]] = {
    "list_order": (("non_cyclic", False),),
}

TOOL_PROMPT_FAMILIES: dict[str, tuple[str, ...]] = {
    "neutral": (
        "The tool is {value}",
        "The word is {value}",
        "The object is {value}",
        "The thing is {value}",
        "The item is {value}",
    ),
}


def all_tool_labels() -> tuple[str, ...]:
    """Default extraction/comparison tool labels (fixed universal set)."""
    return TOOL_UNIVERSAL_SET


def validate_tool_orderings() -> None:
    """Ensure every named ordering uses exactly the same tool set."""
    universal = set(TOOL_UNIVERSAL_SET)
    n = len(TOOL_UNIVERSAL_SET)
    for name, ordering in TOOL_ORDERINGS.items():
        if len(ordering) != n:
            raise ValueError(f"{name}: expected {n} tools, got {len(ordering)}")
        if set(ordering) != universal:
            raise ValueError(f"{name}: must be a permutation of TOOL_UNIVERSAL_SET")


validate_tool_orderings()


# --- Animals (diverse species; no natural list order) ---

ANIMAL_UNIVERSAL_SET: tuple[str, ...] = (
    "dog",
    "cat",
    "horse",
    "cow",
    "eagle",
    "shark",
    "salmon",
    "snake",
    "frog",
    "bee",
    "spider",
    "whale",
)

ANIMAL_ORDERINGS: dict[str, tuple[str, ...]] = {
    "list_order": ANIMAL_UNIVERSAL_SET,
}

ANIMAL_ORDERING_EVAL_MODES: dict[str, tuple[tuple[str, bool], ...]] = {
    "list_order": (("non_cyclic", False),),
}

ANIMAL_PROMPT_FAMILIES: dict[str, tuple[str, ...]] = {
    "neutral": (
        "The animal is {value}",
        "The word is {value}",
        "The creature is {value}",
        "The thing is {value}",
        "The name is {value}",
    ),
}


def all_animal_labels() -> tuple[str, ...]:
    """Default extraction/comparison animal labels (fixed universal set)."""
    return ANIMAL_UNIVERSAL_SET


def validate_animal_orderings() -> None:
    """Ensure every named ordering uses exactly the same animal set."""
    universal = set(ANIMAL_UNIVERSAL_SET)
    n = len(ANIMAL_UNIVERSAL_SET)
    for name, ordering in ANIMAL_ORDERINGS.items():
        if len(ordering) != n:
            raise ValueError(f"{name}: expected {n} animals, got {len(ordering)}")
        if set(ordering) != universal:
            raise ValueError(f"{name}: must be a permutation of ANIMAL_UNIVERSAL_SET")


validate_animal_orderings()


# --- Years (1990–2013; chronological list order) ---

YEAR_UNIVERSAL_SET: tuple[str, ...] = tuple(str(y) for y in range(1990, 2014))

YEAR_ORDERINGS: dict[str, tuple[str, ...]] = {
    "list_order": YEAR_UNIVERSAL_SET,
}

YEAR_ORDERING_EVAL_MODES: dict[str, tuple[tuple[str, bool], ...]] = {
    "list_order": (("non_cyclic", False),),
}

YEAR_PROMPT_FAMILIES: dict[str, tuple[str, ...]] = {
    "neutral": (
        "The year is {value}",
        "The calendar year is {value}",
        "The date year is {value}",
        "The time is {value}",
        "The number is {value}",
    ),
}


def all_year_labels() -> tuple[str, ...]:
    """Default extraction/comparison year labels (1990–2013)."""
    return YEAR_UNIVERSAL_SET


def validate_year_orderings() -> None:
    """Ensure every named ordering uses exactly the same year set."""
    universal = set(YEAR_UNIVERSAL_SET)
    n = len(YEAR_UNIVERSAL_SET)
    for name, ordering in YEAR_ORDERINGS.items():
        if len(ordering) != n:
            raise ValueError(f"{name}: expected {n} years, got {len(ordering)}")
        if set(ordering) != universal:
            raise ValueError(f"{name}: must be a permutation of YEAR_UNIVERSAL_SET")


validate_year_orderings()


# --- Disciplines (fields of study; no natural list order) ---

DISCIPLINE_UNIVERSAL_SET: tuple[str, ...] = (
    "Physics",
    "Biology",
    "Chemistry",
    "Mathematics",
    "Geology",
    "Archaeology",
    "Sociology",
    "Anthropology",
    "Genealogy",
    "Tautology",
)

DISCIPLINE_ORDERINGS: dict[str, tuple[str, ...]] = {
    "list_order": DISCIPLINE_UNIVERSAL_SET,
}

DISCIPLINE_ORDERING_EVAL_MODES: dict[str, tuple[tuple[str, bool], ...]] = {
    "list_order": (("non_cyclic", False),),
}

DISCIPLINE_PROMPT_FAMILIES: dict[str, tuple[str, ...]] = {
    "neutral": (
        "The discipline is {value}",
        "The field is {value}",
        "The subject is {value}",
        "The word is {value}",
        "The study is {value}",
    ),
}


def all_discipline_labels() -> tuple[str, ...]:
    """Default extraction/comparison discipline labels (fixed universal set)."""
    return DISCIPLINE_UNIVERSAL_SET


def validate_discipline_orderings() -> None:
    """Ensure every named ordering uses exactly the same discipline set."""
    universal = set(DISCIPLINE_UNIVERSAL_SET)
    n = len(DISCIPLINE_UNIVERSAL_SET)
    for name, ordering in DISCIPLINE_ORDERINGS.items():
        if len(ordering) != n:
            raise ValueError(f"{name}: expected {n} disciplines, got {len(ordering)}")
        if set(ordering) != universal:
            raise ValueError(f"{name}: must be a permutation of DISCIPLINE_UNIVERSAL_SET")


validate_discipline_orderings()


# --- Question words (Who, What, Where, When, Why, How) ---

WHWORD_UNIVERSAL_SET: tuple[str, ...] = (
    "Who",
    "What",
    "Where",
    "When",
    "Why",
    "How",
)

WHWORD_ORDERINGS: dict[str, tuple[str, ...]] = {
    "list_order": WHWORD_UNIVERSAL_SET,
}

WHWORD_ORDERING_EVAL_MODES: dict[str, tuple[tuple[str, bool], ...]] = {
    "list_order": (("non_cyclic", False),),
}

WHWORD_PROMPT_FAMILIES: dict[str, tuple[str, ...]] = {
    "neutral": (
        "The question word is {value}",
        "The word is {value}",
        "The interrogative is {value}",
        "The query is {value}",
        "The wh-word is {value}",
    ),
}


def all_whword_labels() -> tuple[str, ...]:
    """Default extraction/comparison question-word labels."""
    return WHWORD_UNIVERSAL_SET


def validate_whword_orderings() -> None:
    """Ensure every named ordering uses exactly the same question-word set."""
    universal = set(WHWORD_UNIVERSAL_SET)
    n = len(WHWORD_UNIVERSAL_SET)
    for name, ordering in WHWORD_ORDERINGS.items():
        if len(ordering) != n:
            raise ValueError(f"{name}: expected {n} wh-words, got {len(ordering)}")
        if set(ordering) != universal:
            raise ValueError(f"{name}: must be a permutation of WHWORD_UNIVERSAL_SET")


validate_whword_orderings()


# --- Stopwords (common function words; no natural list order) ---

STOPWORD_UNIVERSAL_SET: tuple[str, ...] = (
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "to",
    "of",
    "is",
    "it",
)

STOPWORD_ORDERINGS: dict[str, tuple[str, ...]] = {
    "list_order": STOPWORD_UNIVERSAL_SET,
}

STOPWORD_ORDERING_EVAL_MODES: dict[str, tuple[tuple[str, bool], ...]] = {
    "list_order": (("non_cyclic", False),),
}

STOPWORD_PROMPT_FAMILIES: dict[str, tuple[str, ...]] = {
    "neutral": (
        "The stopword is {value}",
        "The word is {value}",
        "The token is {value}",
        "The function word is {value}",
        "The particle is {value}",
    ),
}


def all_stopword_labels() -> tuple[str, ...]:
    """Default extraction/comparison stopword labels."""
    return STOPWORD_UNIVERSAL_SET


def validate_stopword_orderings() -> None:
    """Ensure every named ordering uses exactly the same stopword set."""
    universal = set(STOPWORD_UNIVERSAL_SET)
    n = len(STOPWORD_UNIVERSAL_SET)
    for name, ordering in STOPWORD_ORDERINGS.items():
        if len(ordering) != n:
            raise ValueError(f"{name}: expected {n} stopwords, got {len(ordering)}")
        if set(ordering) != universal:
            raise ValueError(f"{name}: must be a permutation of STOPWORD_UNIVERSAL_SET")


validate_stopword_orderings()


# --- Programming languages (no natural list order) ---

LANGUAGE_UNIVERSAL_SET: tuple[str, ...] = (
    "Python",
    "JavaScript",
    "Java",
    "Rust",
    "Go",
    "Ruby",
    "Swift",
    "Kotlin",
    "Scala",
    "Haskell",
    "Perl",
    "Lua",
)

LANGUAGE_ORDERINGS: dict[str, tuple[str, ...]] = {
    "list_order": LANGUAGE_UNIVERSAL_SET,
}

LANGUAGE_ORDERING_EVAL_MODES: dict[str, tuple[tuple[str, bool], ...]] = {
    "list_order": (("non_cyclic", False),),
}

LANGUAGE_PROMPT_FAMILIES: dict[str, tuple[str, ...]] = {
    "neutral": (
        "The language is {value}",
        "The programming language is {value}",
        "The source language is {value}",
        "The word is {value}",
        "The code is written in {value}",
    ),
}


def all_language_labels() -> tuple[str, ...]:
    """Default extraction/comparison programming language labels."""
    return LANGUAGE_UNIVERSAL_SET


def validate_language_orderings() -> None:
    """Ensure every named ordering uses exactly the same language set."""
    universal = set(LANGUAGE_UNIVERSAL_SET)
    n = len(LANGUAGE_UNIVERSAL_SET)
    for name, ordering in LANGUAGE_ORDERINGS.items():
        if len(ordering) != n:
            raise ValueError(f"{name}: expected {n} languages, got {len(ordering)}")
        if set(ordering) != universal:
            raise ValueError(f"{name}: must be a permutation of LANGUAGE_UNIVERSAL_SET")


validate_language_orderings()


def expand_prompts(
    items: tuple[str, ...] | list[str],
    templates: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, str]]:
    """Return month prompt records with concept label and filled text."""
    templates = DEFAULT_MONTH_TEMPLATES if templates is None else tuple(templates)
    records: list[dict[str, str]] = []
    for concept in items:
        for template in templates:
            records.append(
                {
                    "concept": concept,
                    "template": template,
                    "text": template.format(month=concept),
                    "token_position": "final",
                }
            )
    return records


def expand_colour_prompts(
    colours: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> list[dict[str, object]]:
    """
    Build colour prompt records (one per colour × template).

    v1: single-word colour names only; no trailing punctuation after the colour.
    """
    if prompt_family not in COLOUR_PROMPT_FAMILIES:
        raise ValueError(f"unknown prompt_family={prompt_family!r}")
    colours = COLOUR_UNIVERSAL_SET if colours is None else tuple(colours)
    templates = COLOUR_PROMPT_FAMILIES[prompt_family] if templates is None else tuple(templates)

    records: list[dict[str, object]] = []
    for index, value in enumerate(colours):
        for template_idx, template in enumerate(templates):
            records.append(
                {
                    "concept": "colour",
                    "value": value,
                    "index": index,
                    "template_id": f"{prompt_family}_{template_idx:02d}",
                    "prompt_family": prompt_family,
                    "prompt": template.format(value=value),
                    "token_position": "final",
                }
            )
    return records


def write_colour_prompts(
    path: str | Path,
    colours: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> Path:
    """Write colour prompts as JSONL (one record per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = expand_colour_prompts(colours, templates, prompt_family=prompt_family)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def expand_tool_prompts(
    tools: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> list[dict[str, object]]:
    """Build tool prompt records (one per tool × template)."""
    if prompt_family not in TOOL_PROMPT_FAMILIES:
        raise ValueError(f"unknown prompt_family={prompt_family!r}")
    tools = TOOL_UNIVERSAL_SET if tools is None else tuple(tools)
    templates = TOOL_PROMPT_FAMILIES[prompt_family] if templates is None else tuple(templates)

    records: list[dict[str, object]] = []
    for index, value in enumerate(tools):
        for template_idx, template in enumerate(templates):
            records.append(
                {
                    "concept": "tool",
                    "value": value,
                    "index": index,
                    "template_id": f"{prompt_family}_{template_idx:02d}",
                    "prompt_family": prompt_family,
                    "prompt": template.format(value=value),
                    "token_position": "final",
                }
            )
    return records


def write_tool_prompts(
    path: str | Path,
    tools: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> Path:
    """Write tool prompts as JSONL (one record per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = expand_tool_prompts(tools, templates, prompt_family=prompt_family)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def expand_animal_prompts(
    animals: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> list[dict[str, object]]:
    """Build animal prompt records (one per animal × template)."""
    if prompt_family not in ANIMAL_PROMPT_FAMILIES:
        raise ValueError(f"unknown prompt_family={prompt_family!r}")
    animals = ANIMAL_UNIVERSAL_SET if animals is None else tuple(animals)
    templates = ANIMAL_PROMPT_FAMILIES[prompt_family] if templates is None else tuple(templates)

    records: list[dict[str, object]] = []
    for index, value in enumerate(animals):
        for template_idx, template in enumerate(templates):
            records.append(
                {
                    "concept": "animal",
                    "value": value,
                    "index": index,
                    "template_id": f"{prompt_family}_{template_idx:02d}",
                    "prompt_family": prompt_family,
                    "prompt": template.format(value=value),
                    "token_position": "final",
                }
            )
    return records


def write_animal_prompts(
    path: str | Path,
    animals: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> Path:
    """Write animal prompts as JSONL (one record per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = expand_animal_prompts(animals, templates, prompt_family=prompt_family)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def expand_year_prompts(
    years: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> list[dict[str, object]]:
    """Build year prompt records (one per year × template)."""
    if prompt_family not in YEAR_PROMPT_FAMILIES:
        raise ValueError(f"unknown prompt_family={prompt_family!r}")
    years = YEAR_UNIVERSAL_SET if years is None else tuple(years)
    templates = YEAR_PROMPT_FAMILIES[prompt_family] if templates is None else tuple(templates)

    records: list[dict[str, object]] = []
    for index, value in enumerate(years):
        for template_idx, template in enumerate(templates):
            records.append(
                {
                    "concept": "year",
                    "value": value,
                    "index": index,
                    "template_id": f"{prompt_family}_{template_idx:02d}",
                    "prompt_family": prompt_family,
                    "prompt": template.format(value=value),
                    "token_position": "final",
                }
            )
    return records


def write_year_prompts(
    path: str | Path,
    years: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> Path:
    """Write year prompts as JSONL (one record per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = expand_year_prompts(years, templates, prompt_family=prompt_family)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def expand_discipline_prompts(
    disciplines: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> list[dict[str, object]]:
    """Build discipline prompt records (one per discipline × template)."""
    if prompt_family not in DISCIPLINE_PROMPT_FAMILIES:
        raise ValueError(f"unknown prompt_family={prompt_family!r}")
    disciplines = DISCIPLINE_UNIVERSAL_SET if disciplines is None else tuple(disciplines)
    templates = (
        DISCIPLINE_PROMPT_FAMILIES[prompt_family] if templates is None else tuple(templates)
    )

    records: list[dict[str, object]] = []
    for index, value in enumerate(disciplines):
        for template_idx, template in enumerate(templates):
            records.append(
                {
                    "concept": "discipline",
                    "value": value,
                    "index": index,
                    "template_id": f"{prompt_family}_{template_idx:02d}",
                    "prompt_family": prompt_family,
                    "prompt": template.format(value=value),
                    "token_position": "final",
                }
            )
    return records


def write_discipline_prompts(
    path: str | Path,
    disciplines: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> Path:
    """Write discipline prompts as JSONL (one record per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = expand_discipline_prompts(disciplines, templates, prompt_family=prompt_family)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def expand_whword_prompts(
    whwords: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> list[dict[str, object]]:
    """Build question-word prompt records (one per word × template)."""
    if prompt_family not in WHWORD_PROMPT_FAMILIES:
        raise ValueError(f"unknown prompt_family={prompt_family!r}")
    whwords = WHWORD_UNIVERSAL_SET if whwords is None else tuple(whwords)
    templates = WHWORD_PROMPT_FAMILIES[prompt_family] if templates is None else tuple(templates)

    records: list[dict[str, object]] = []
    for index, value in enumerate(whwords):
        for template_idx, template in enumerate(templates):
            records.append(
                {
                    "concept": "whword",
                    "value": value,
                    "index": index,
                    "template_id": f"{prompt_family}_{template_idx:02d}",
                    "prompt_family": prompt_family,
                    "prompt": template.format(value=value),
                    "token_position": "final",
                }
            )
    return records


def write_whword_prompts(
    path: str | Path,
    whwords: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> Path:
    """Write question-word prompts as JSONL (one record per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = expand_whword_prompts(whwords, templates, prompt_family=prompt_family)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def expand_stopword_prompts(
    stopwords: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> list[dict[str, object]]:
    """Build stopword prompt records (one per stopword × template)."""
    if prompt_family not in STOPWORD_PROMPT_FAMILIES:
        raise ValueError(f"unknown prompt_family={prompt_family!r}")
    stopwords = STOPWORD_UNIVERSAL_SET if stopwords is None else tuple(stopwords)
    templates = STOPWORD_PROMPT_FAMILIES[prompt_family] if templates is None else tuple(templates)

    records: list[dict[str, object]] = []
    for index, value in enumerate(stopwords):
        for template_idx, template in enumerate(templates):
            records.append(
                {
                    "concept": "stopword",
                    "value": value,
                    "index": index,
                    "template_id": f"{prompt_family}_{template_idx:02d}",
                    "prompt_family": prompt_family,
                    "prompt": template.format(value=value),
                    "token_position": "final",
                }
            )
    return records


def write_stopword_prompts(
    path: str | Path,
    stopwords: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> Path:
    """Write stopword prompts as JSONL (one record per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = expand_stopword_prompts(stopwords, templates, prompt_family=prompt_family)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def expand_language_prompts(
    languages: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> list[dict[str, object]]:
    """Build programming-language prompt records (one per language × template)."""
    if prompt_family not in LANGUAGE_PROMPT_FAMILIES:
        raise ValueError(f"unknown prompt_family={prompt_family!r}")
    languages = LANGUAGE_UNIVERSAL_SET if languages is None else tuple(languages)
    templates = LANGUAGE_PROMPT_FAMILIES[prompt_family] if templates is None else tuple(templates)

    records: list[dict[str, object]] = []
    for index, value in enumerate(languages):
        for template_idx, template in enumerate(templates):
            records.append(
                {
                    "concept": "language",
                    "value": value,
                    "index": index,
                    "template_id": f"{prompt_family}_{template_idx:02d}",
                    "prompt_family": prompt_family,
                    "prompt": template.format(value=value),
                    "token_position": "final",
                }
            )
    return records


def write_language_prompts(
    path: str | Path,
    languages: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> Path:
    """Write programming-language prompts as JSONL (one record per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = expand_language_prompts(languages, templates, prompt_family=prompt_family)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def expand_noun_prompts(
    nouns: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> list[dict[str, object]]:
    """Build noun prompt records (one per noun × template)."""
    if prompt_family not in NOUN_PROMPT_FAMILIES:
        raise ValueError(f"unknown prompt_family={prompt_family!r}")
    nouns = NOUN_UNIVERSAL_SET if nouns is None else tuple(nouns)
    templates = NOUN_PROMPT_FAMILIES[prompt_family] if templates is None else tuple(templates)

    records: list[dict[str, object]] = []
    for index, value in enumerate(nouns):
        for template_idx, template in enumerate(templates):
            records.append(
                {
                    "concept": "noun",
                    "value": value,
                    "index": index,
                    "template_id": f"{prompt_family}_{template_idx:02d}",
                    "prompt_family": prompt_family,
                    "prompt": template.format(value=value),
                    "token_position": "final",
                }
            )
    return records


def write_noun_prompts(
    path: str | Path,
    nouns: tuple[str, ...] | list[str] | None = None,
    templates: tuple[str, ...] | list[str] | None = None,
    *,
    prompt_family: PromptFamily = "neutral",
) -> Path:
    """Write noun prompts as JSONL (one record per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = expand_noun_prompts(nouns, templates, prompt_family=prompt_family)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path

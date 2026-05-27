"""DistilGPT-2 ln_2 activation and MLP-local centroid extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lce.centroid_adapter import compute_centroid
from lce.concepts import (
    MONTHS,
    all_colour_labels,
    all_animal_labels,
    all_discipline_labels,
    all_language_labels,
    all_noun_labels,
    all_stopword_labels,
    all_tool_labels,
    all_whword_labels,
    all_year_labels,
    expand_animal_prompts,
    expand_colour_prompts,
    expand_discipline_prompts,
    expand_language_prompts,
    expand_noun_prompts,
    expand_stopword_prompts,
    expand_tool_prompts,
    expand_whword_prompts,
    expand_year_prompts,
    expand_prompts,
)
from lce.io import save_metrics_json, save_per_template_npz
from lce.metrics import compare_template_variance
from lce.token_audit import TokenAuditError, audit_prompt_records


@dataclass(frozen=True)
class ExtractionResult:
    layer: int
    per_template_path: Path
    activations: np.ndarray
    centroids: np.ndarray
    concepts: tuple[str, ...]
    templates: tuple[str, ...]


def resolve_device(preference: str = "cuda_if_available") -> torch.device:
    if preference == "cuda_if_available" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preference)
    return torch.device("cpu")


def load_model_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def _capture_ln2_outputs(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_indices: list[int],
) -> dict[int, torch.Tensor]:
    captures: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(layer_idx: int):
        def hook(_module, _inp, out):
            captures[layer_idx] = out.detach()

        return hook

    for layer_idx in layer_indices:
        handle = model.transformer.h[layer_idx].ln_2.register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    with torch.no_grad():
        model(input_ids)

    for handle in handles:
        handle.remove()
    return captures


def _mlp_local_centroid(block: nn.Module, h_token: torch.Tensor) -> torch.Tensor:
    """Centroid w.r.t. ln_2 output h at a single token: grad sum(mlp(h))."""
    h_in = h_token.reshape(1, 1, -1).detach().requires_grad_(True)
    with torch.enable_grad():
        mlp_out = block.mlp(h_in)
        centroid = compute_centroid(mlp_out, h_in)
    return centroid.detach().reshape(-1)


def extract_prompt_at_layers(
    model: nn.Module,
    input_ids: torch.Tensor,
    token_index: int,
    layer_indices: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    device = input_ids.device
    captures = _capture_ln2_outputs(model, input_ids, layer_indices)

    activations: dict[int, np.ndarray] = {}
    centroids: dict[int, np.ndarray] = {}

    for layer_idx in layer_indices:
        h = captures[layer_idx][0, token_index].to(device)
        activations[layer_idx] = h.cpu().numpy().astype(np.float32)
        block = model.transformer.h[layer_idx]
        centroids[layer_idx] = _mlp_local_centroid(block, h).cpu().numpy().astype(np.float32)

    return activations, centroids


def _build_record_index(records: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(r["concept"], r["template"]): r for r in records}


def extract_months_experiment(
    *,
    model_name: str,
    layers: list[int],
    templates: tuple[str, ...] | list[str],
    raw_dir: str | Path,
    device: torch.device | None = None,
) -> list[ExtractionResult]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = device or resolve_device()

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    records = expand_prompts(MONTHS, templates)
    audit_prompt_records(tokenizer, records)
    record_index = _build_record_index(records)
    template_list = tuple(templates)

    n_templates = len(template_list)
    n_months = len(MONTHS)
    hidden = int(model.config.n_embd)

    layer_arrays_act = {
        layer: np.zeros((n_templates, n_months, hidden), dtype=np.float32) for layer in layers
    }
    layer_arrays_cen = {
        layer: np.zeros((n_templates, n_months, hidden), dtype=np.float32) for layer in layers
    }

    for mi, month in enumerate(tqdm(MONTHS, desc="months")):
        for ti, template in enumerate(template_list):
            record = record_index[(month, template)]
            encoded = tokenizer(record["text"], return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            audit = audit_prompt_records(tokenizer, [record])[0]
            token_index = audit.final_token_index

            activations, centroids = extract_prompt_at_layers(
                model, input_ids, token_index, layers
            )
            for layer_idx in layers:
                layer_arrays_act[layer_idx][ti, mi] = activations[layer_idx]
                layer_arrays_cen[layer_idx][ti, mi] = centroids[layer_idx]

    results: list[ExtractionResult] = []
    for layer_idx in layers:
        path = save_per_template_npz(
            raw_dir / f"layer_{layer_idx:02d}_per_template.npz",
            activations=layer_arrays_act[layer_idx],
            centroids=layer_arrays_cen[layer_idx],
            concepts=list(MONTHS),
            templates=list(template_list),
        )
        results.append(
            ExtractionResult(
                layer=layer_idx,
                per_template_path=path,
                activations=layer_arrays_act[layer_idx],
                centroids=layer_arrays_cen[layer_idx],
                concepts=MONTHS,
                templates=template_list,
            )
        )

    variance_metrics = {
        str(layer_idx): compare_template_variance(
            layer_arrays_act[layer_idx], layer_arrays_cen[layer_idx]
        )
        for layer_idx in layers
    }
    save_metrics_json(
        raw_dir / "template_variance.json",
        {"model_name": model_name, "layers": variance_metrics},
    )
    return results


def extract_colours_experiment(
    *,
    model_name: str,
    layers: list[int],
    templates: tuple[str, ...] | list[str] | None = None,
    colours: tuple[str, ...] | list[str] | None = None,
    prompt_family: str = "neutral",
    raw_dir: str | Path,
    device: torch.device | None = None,
) -> list[ExtractionResult]:
    """Extract template × colour tensors; one row per colour in ``all_colour_labels()`` order."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = device or resolve_device()

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    colour_list = tuple(colours) if colours is not None else all_colour_labels()
    records = expand_colour_prompts(colour_list, templates, prompt_family=prompt_family)
    audit_prompt_records(tokenizer, records)

    if templates is None:
        from lce.concepts import COLOUR_PROMPT_FAMILIES

        template_list = COLOUR_PROMPT_FAMILIES[prompt_family]
    else:
        template_list = tuple(templates)

    record_index: dict[tuple[str, str], dict[str, str]] = {}
    for r in records:
        value = str(r["value"])
        prompt = str(r["prompt"])
        for template in template_list:
            if prompt == template.format(value=value):
                record_index[(value, template)] = {
                    "concept": value,
                    "text": prompt,
                    "token_position": "final",
                }
                break
        else:
            raise KeyError(f"could not match prompt to template: {prompt!r}")

    n_templates = len(template_list)
    n_colours = len(colour_list)
    hidden = int(model.config.n_embd)

    layer_arrays_act = {
        layer: np.zeros((n_templates, n_colours, hidden), dtype=np.float32) for layer in layers
    }
    layer_arrays_cen = {
        layer: np.zeros((n_templates, n_colours, hidden), dtype=np.float32) for layer in layers
    }

    for ci, colour in enumerate(tqdm(colour_list, desc="colours")):
        for ti, template in enumerate(template_list):
            record = record_index[(colour, template)]
            encoded = tokenizer(record["text"], return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            audit = audit_prompt_records(tokenizer, [record])[0]
            token_index = audit.final_token_index

            activations, centroids = extract_prompt_at_layers(
                model, input_ids, token_index, layers
            )
            for layer_idx in layers:
                layer_arrays_act[layer_idx][ti, ci] = activations[layer_idx]
                layer_arrays_cen[layer_idx][ti, ci] = centroids[layer_idx]

    results: list[ExtractionResult] = []
    for layer_idx in layers:
        path = save_per_template_npz(
            raw_dir / f"layer_{layer_idx:02d}_per_template.npz",
            activations=layer_arrays_act[layer_idx],
            centroids=layer_arrays_cen[layer_idx],
            concepts=list(colour_list),
            templates=list(template_list),
        )
        results.append(
            ExtractionResult(
                layer=layer_idx,
                per_template_path=path,
                activations=layer_arrays_act[layer_idx],
                centroids=layer_arrays_cen[layer_idx],
                concepts=tuple(colour_list),
                templates=template_list,
            )
        )

    variance_metrics = {
        str(layer_idx): compare_template_variance(
            layer_arrays_act[layer_idx], layer_arrays_cen[layer_idx]
        )
        for layer_idx in layers
    }
    save_metrics_json(
        raw_dir / "template_variance.json",
        {"model_name": model_name, "layers": variance_metrics},
    )
    return results


def extract_nouns_experiment(
    *,
    model_name: str,
    layers: list[int],
    templates: tuple[str, ...] | list[str] | None = None,
    nouns: tuple[str, ...] | list[str] | None = None,
    prompt_family: str = "neutral",
    raw_dir: str | Path,
    device: torch.device | None = None,
) -> list[ExtractionResult]:
    """Extract template × noun tensors; one row per noun in ``all_noun_labels()`` order."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = device or resolve_device()

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    noun_list = tuple(nouns) if nouns is not None else all_noun_labels()
    records = expand_noun_prompts(noun_list, templates, prompt_family=prompt_family)
    audit_prompt_records(tokenizer, records)

    if templates is None:
        from lce.concepts import NOUN_PROMPT_FAMILIES

        template_list = NOUN_PROMPT_FAMILIES[prompt_family]
    else:
        template_list = tuple(templates)

    record_index: dict[tuple[str, str], dict[str, str]] = {}
    for r in records:
        value = str(r["value"])
        prompt = str(r["prompt"])
        for template in template_list:
            if prompt == template.format(value=value):
                record_index[(value, template)] = {
                    "concept": value,
                    "text": prompt,
                    "token_position": "final",
                }
                break
        else:
            raise KeyError(f"could not match prompt to template: {prompt!r}")

    n_templates = len(template_list)
    n_nouns = len(noun_list)
    hidden = int(model.config.n_embd)

    layer_arrays_act = {
        layer: np.zeros((n_templates, n_nouns, hidden), dtype=np.float32) for layer in layers
    }
    layer_arrays_cen = {
        layer: np.zeros((n_templates, n_nouns, hidden), dtype=np.float32) for layer in layers
    }

    for ni, noun in enumerate(tqdm(noun_list, desc="nouns")):
        for ti, template in enumerate(template_list):
            record = record_index[(noun, template)]
            encoded = tokenizer(record["text"], return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            audit = audit_prompt_records(tokenizer, [record])[0]
            token_index = audit.final_token_index

            activations, centroids = extract_prompt_at_layers(
                model, input_ids, token_index, layers
            )
            for layer_idx in layers:
                layer_arrays_act[layer_idx][ti, ni] = activations[layer_idx]
                layer_arrays_cen[layer_idx][ti, ni] = centroids[layer_idx]

    results: list[ExtractionResult] = []
    for layer_idx in layers:
        path = save_per_template_npz(
            raw_dir / f"layer_{layer_idx:02d}_per_template.npz",
            activations=layer_arrays_act[layer_idx],
            centroids=layer_arrays_cen[layer_idx],
            concepts=list(noun_list),
            templates=list(template_list),
        )
        results.append(
            ExtractionResult(
                layer=layer_idx,
                per_template_path=path,
                activations=layer_arrays_act[layer_idx],
                centroids=layer_arrays_cen[layer_idx],
                concepts=tuple(noun_list),
                templates=template_list,
            )
        )

    variance_metrics = {
        str(layer_idx): compare_template_variance(
            layer_arrays_act[layer_idx], layer_arrays_cen[layer_idx]
        )
        for layer_idx in layers
    }
    save_metrics_json(
        raw_dir / "template_variance.json",
        {"model_name": model_name, "layers": variance_metrics},
    )
    return results


def extract_tools_experiment(
    *,
    model_name: str,
    layers: list[int],
    templates: tuple[str, ...] | list[str] | None = None,
    tools: tuple[str, ...] | list[str] | None = None,
    prompt_family: str = "neutral",
    raw_dir: str | Path,
    device: torch.device | None = None,
) -> list[ExtractionResult]:
    """Extract template × tool tensors; one row per tool in ``all_tool_labels()`` order."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = device or resolve_device()

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    tool_list = tuple(tools) if tools is not None else all_tool_labels()
    records = expand_tool_prompts(tool_list, templates, prompt_family=prompt_family)
    audit_prompt_records(tokenizer, records)

    if templates is None:
        from lce.concepts import TOOL_PROMPT_FAMILIES

        template_list = TOOL_PROMPT_FAMILIES[prompt_family]
    else:
        template_list = tuple(templates)

    record_index: dict[tuple[str, str], dict[str, str]] = {}
    for r in records:
        value = str(r["value"])
        prompt = str(r["prompt"])
        for template in template_list:
            if prompt == template.format(value=value):
                record_index[(value, template)] = {
                    "concept": value,
                    "text": prompt,
                    "token_position": "final",
                }
                break
        else:
            raise KeyError(f"could not match prompt to template: {prompt!r}")

    n_templates = len(template_list)
    n_tools = len(tool_list)
    hidden = int(model.config.n_embd)

    layer_arrays_act = {
        layer: np.zeros((n_templates, n_tools, hidden), dtype=np.float32) for layer in layers
    }
    layer_arrays_cen = {
        layer: np.zeros((n_templates, n_tools, hidden), dtype=np.float32) for layer in layers
    }

    for tool_idx, tool in enumerate(tqdm(tool_list, desc="tools")):
        for ti, template in enumerate(template_list):
            record = record_index[(tool, template)]
            encoded = tokenizer(record["text"], return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            audit = audit_prompt_records(tokenizer, [record])[0]
            token_index = audit.final_token_index

            activations, centroids = extract_prompt_at_layers(
                model, input_ids, token_index, layers
            )
            for layer_idx in layers:
                layer_arrays_act[layer_idx][ti, tool_idx] = activations[layer_idx]
                layer_arrays_cen[layer_idx][ti, tool_idx] = centroids[layer_idx]

    results: list[ExtractionResult] = []
    for layer_idx in layers:
        path = save_per_template_npz(
            raw_dir / f"layer_{layer_idx:02d}_per_template.npz",
            activations=layer_arrays_act[layer_idx],
            centroids=layer_arrays_cen[layer_idx],
            concepts=list(tool_list),
            templates=list(template_list),
        )
        results.append(
            ExtractionResult(
                layer=layer_idx,
                per_template_path=path,
                activations=layer_arrays_act[layer_idx],
                centroids=layer_arrays_cen[layer_idx],
                concepts=tuple(tool_list),
                templates=template_list,
            )
        )

    variance_metrics = {
        str(layer_idx): compare_template_variance(
            layer_arrays_act[layer_idx], layer_arrays_cen[layer_idx]
        )
        for layer_idx in layers
    }
    save_metrics_json(
        raw_dir / "template_variance.json",
        {"model_name": model_name, "layers": variance_metrics},
    )
    return results


def extract_animals_experiment(
    *,
    model_name: str,
    layers: list[int],
    templates: tuple[str, ...] | list[str] | None = None,
    animals: tuple[str, ...] | list[str] | None = None,
    prompt_family: str = "neutral",
    raw_dir: str | Path,
    device: torch.device | None = None,
) -> list[ExtractionResult]:
    """Extract template × animal tensors; one row per animal in ``all_animal_labels()`` order."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = device or resolve_device()

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    animal_list = tuple(animals) if animals is not None else all_animal_labels()
    records = expand_animal_prompts(animal_list, templates, prompt_family=prompt_family)
    audit_prompt_records(tokenizer, records)

    if templates is None:
        from lce.concepts import ANIMAL_PROMPT_FAMILIES

        template_list = ANIMAL_PROMPT_FAMILIES[prompt_family]
    else:
        template_list = tuple(templates)

    record_index: dict[tuple[str, str], dict[str, str]] = {}
    for r in records:
        value = str(r["value"])
        prompt = str(r["prompt"])
        for template in template_list:
            if prompt == template.format(value=value):
                record_index[(value, template)] = {
                    "concept": value,
                    "text": prompt,
                    "token_position": "final",
                }
                break
        else:
            raise KeyError(f"could not match prompt to template: {prompt!r}")

    n_templates = len(template_list)
    n_animals = len(animal_list)
    hidden = int(model.config.n_embd)

    layer_arrays_act = {
        layer: np.zeros((n_templates, n_animals, hidden), dtype=np.float32) for layer in layers
    }
    layer_arrays_cen = {
        layer: np.zeros((n_templates, n_animals, hidden), dtype=np.float32) for layer in layers
    }

    for animal_idx, animal in enumerate(tqdm(animal_list, desc="animals")):
        for ti, template in enumerate(template_list):
            record = record_index[(animal, template)]
            encoded = tokenizer(record["text"], return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            audit = audit_prompt_records(tokenizer, [record])[0]
            token_index = audit.final_token_index

            activations, centroids = extract_prompt_at_layers(
                model, input_ids, token_index, layers
            )
            for layer_idx in layers:
                layer_arrays_act[layer_idx][ti, animal_idx] = activations[layer_idx]
                layer_arrays_cen[layer_idx][ti, animal_idx] = centroids[layer_idx]

    results: list[ExtractionResult] = []
    for layer_idx in layers:
        path = save_per_template_npz(
            raw_dir / f"layer_{layer_idx:02d}_per_template.npz",
            activations=layer_arrays_act[layer_idx],
            centroids=layer_arrays_cen[layer_idx],
            concepts=list(animal_list),
            templates=list(template_list),
        )
        results.append(
            ExtractionResult(
                layer=layer_idx,
                per_template_path=path,
                activations=layer_arrays_act[layer_idx],
                centroids=layer_arrays_cen[layer_idx],
                concepts=tuple(animal_list),
                templates=template_list,
            )
        )

    variance_metrics = {
        str(layer_idx): compare_template_variance(
            layer_arrays_act[layer_idx], layer_arrays_cen[layer_idx]
        )
        for layer_idx in layers
    }
    save_metrics_json(
        raw_dir / "template_variance.json",
        {"model_name": model_name, "layers": variance_metrics},
    )
    return results


def extract_years_experiment(
    *,
    model_name: str,
    layers: list[int],
    templates: tuple[str, ...] | list[str] | None = None,
    years: tuple[str, ...] | list[str] | None = None,
    prompt_family: str = "neutral",
    raw_dir: str | Path,
    device: torch.device | None = None,
) -> list[ExtractionResult]:
    """Extract template × year tensors; one row per year in ``all_year_labels()`` order."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = device or resolve_device()

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    year_list = tuple(years) if years is not None else all_year_labels()
    records = expand_year_prompts(year_list, templates, prompt_family=prompt_family)
    audit_prompt_records(tokenizer, records)

    if templates is None:
        from lce.concepts import YEAR_PROMPT_FAMILIES

        template_list = YEAR_PROMPT_FAMILIES[prompt_family]
    else:
        template_list = tuple(templates)

    record_index: dict[tuple[str, str], dict[str, str]] = {}
    for r in records:
        value = str(r["value"])
        prompt = str(r["prompt"])
        for template in template_list:
            if prompt == template.format(value=value):
                record_index[(value, template)] = {
                    "concept": value,
                    "text": prompt,
                    "token_position": "final",
                }
                break
        else:
            raise KeyError(f"could not match prompt to template: {prompt!r}")

    n_templates = len(template_list)
    n_years = len(year_list)
    hidden = int(model.config.n_embd)

    layer_arrays_act = {
        layer: np.zeros((n_templates, n_years, hidden), dtype=np.float32) for layer in layers
    }
    layer_arrays_cen = {
        layer: np.zeros((n_templates, n_years, hidden), dtype=np.float32) for layer in layers
    }

    for year_idx, year in enumerate(tqdm(year_list, desc="years")):
        for ti, template in enumerate(template_list):
            record = record_index[(year, template)]
            encoded = tokenizer(record["text"], return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            audit = audit_prompt_records(tokenizer, [record])[0]
            token_index = audit.final_token_index

            activations, centroids = extract_prompt_at_layers(
                model, input_ids, token_index, layers
            )
            for layer_idx in layers:
                layer_arrays_act[layer_idx][ti, year_idx] = activations[layer_idx]
                layer_arrays_cen[layer_idx][ti, year_idx] = centroids[layer_idx]

    results: list[ExtractionResult] = []
    for layer_idx in layers:
        path = save_per_template_npz(
            raw_dir / f"layer_{layer_idx:02d}_per_template.npz",
            activations=layer_arrays_act[layer_idx],
            centroids=layer_arrays_cen[layer_idx],
            concepts=list(year_list),
            templates=list(template_list),
        )
        results.append(
            ExtractionResult(
                layer=layer_idx,
                per_template_path=path,
                activations=layer_arrays_act[layer_idx],
                centroids=layer_arrays_cen[layer_idx],
                concepts=tuple(year_list),
                templates=template_list,
            )
        )

    variance_metrics = {
        str(layer_idx): compare_template_variance(
            layer_arrays_act[layer_idx], layer_arrays_cen[layer_idx]
        )
        for layer_idx in layers
    }
    save_metrics_json(
        raw_dir / "template_variance.json",
        {"model_name": model_name, "layers": variance_metrics},
    )
    return results


def extract_disciplines_experiment(
    *,
    model_name: str,
    layers: list[int],
    templates: tuple[str, ...] | list[str] | None = None,
    disciplines: tuple[str, ...] | list[str] | None = None,
    prompt_family: str = "neutral",
    raw_dir: str | Path,
    device: torch.device | None = None,
) -> list[ExtractionResult]:
    """Extract template × discipline tensors; one row per label in ``all_discipline_labels()`` order."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = device or resolve_device()

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    discipline_list = tuple(disciplines) if disciplines is not None else all_discipline_labels()
    records = expand_discipline_prompts(discipline_list, templates, prompt_family=prompt_family)
    audit_prompt_records(tokenizer, records)

    if templates is None:
        from lce.concepts import DISCIPLINE_PROMPT_FAMILIES

        template_list = DISCIPLINE_PROMPT_FAMILIES[prompt_family]
    else:
        template_list = tuple(templates)

    record_index: dict[tuple[str, str], dict[str, str]] = {}
    for r in records:
        value = str(r["value"])
        prompt = str(r["prompt"])
        for template in template_list:
            if prompt == template.format(value=value):
                record_index[(value, template)] = {
                    "concept": value,
                    "text": prompt,
                    "token_position": "final",
                }
                break
        else:
            raise KeyError(f"could not match prompt to template: {prompt!r}")

    n_templates = len(template_list)
    n_disciplines = len(discipline_list)
    hidden = int(model.config.n_embd)

    layer_arrays_act = {
        layer: np.zeros((n_templates, n_disciplines, hidden), dtype=np.float32)
        for layer in layers
    }
    layer_arrays_cen = {
        layer: np.zeros((n_templates, n_disciplines, hidden), dtype=np.float32)
        for layer in layers
    }

    for discipline_idx, discipline in enumerate(tqdm(discipline_list, desc="disciplines")):
        for ti, template in enumerate(template_list):
            record = record_index[(discipline, template)]
            encoded = tokenizer(record["text"], return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            audit = audit_prompt_records(tokenizer, [record])[0]
            token_index = audit.final_token_index

            activations, centroids = extract_prompt_at_layers(
                model, input_ids, token_index, layers
            )
            for layer_idx in layers:
                layer_arrays_act[layer_idx][ti, discipline_idx] = activations[layer_idx]
                layer_arrays_cen[layer_idx][ti, discipline_idx] = centroids[layer_idx]

    results: list[ExtractionResult] = []
    for layer_idx in layers:
        path = save_per_template_npz(
            raw_dir / f"layer_{layer_idx:02d}_per_template.npz",
            activations=layer_arrays_act[layer_idx],
            centroids=layer_arrays_cen[layer_idx],
            concepts=list(discipline_list),
            templates=list(template_list),
        )
        results.append(
            ExtractionResult(
                layer=layer_idx,
                per_template_path=path,
                activations=layer_arrays_act[layer_idx],
                centroids=layer_arrays_cen[layer_idx],
                concepts=tuple(discipline_list),
                templates=template_list,
            )
        )

    variance_metrics = {
        str(layer_idx): compare_template_variance(
            layer_arrays_act[layer_idx], layer_arrays_cen[layer_idx]
        )
        for layer_idx in layers
    }
    save_metrics_json(
        raw_dir / "template_variance.json",
        {"model_name": model_name, "layers": variance_metrics},
    )
    return results


def extract_whwords_experiment(
    *,
    model_name: str,
    layers: list[int],
    templates: tuple[str, ...] | list[str] | None = None,
    whwords: tuple[str, ...] | list[str] | None = None,
    prompt_family: str = "neutral",
    raw_dir: str | Path,
    device: torch.device | None = None,
) -> list[ExtractionResult]:
    """Extract template × question-word tensors; one row per ``all_whword_labels()`` order."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = device or resolve_device()

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    whword_list = tuple(whwords) if whwords is not None else all_whword_labels()
    records = expand_whword_prompts(whword_list, templates, prompt_family=prompt_family)
    audit_prompt_records(tokenizer, records)

    if templates is None:
        from lce.concepts import WHWORD_PROMPT_FAMILIES

        template_list = WHWORD_PROMPT_FAMILIES[prompt_family]
    else:
        template_list = tuple(templates)

    record_index: dict[tuple[str, str], dict[str, str]] = {}
    for r in records:
        value = str(r["value"])
        prompt = str(r["prompt"])
        for template in template_list:
            if prompt == template.format(value=value):
                record_index[(value, template)] = {
                    "concept": value,
                    "text": prompt,
                    "token_position": "final",
                }
                break
        else:
            raise KeyError(f"could not match prompt to template: {prompt!r}")

    n_templates = len(template_list)
    n_whwords = len(whword_list)
    hidden = int(model.config.n_embd)

    layer_arrays_act = {
        layer: np.zeros((n_templates, n_whwords, hidden), dtype=np.float32) for layer in layers
    }
    layer_arrays_cen = {
        layer: np.zeros((n_templates, n_whwords, hidden), dtype=np.float32) for layer in layers
    }

    for whword_idx, whword in enumerate(tqdm(whword_list, desc="whwords")):
        for ti, template in enumerate(template_list):
            record = record_index[(whword, template)]
            encoded = tokenizer(record["text"], return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            audit = audit_prompt_records(tokenizer, [record])[0]
            token_index = audit.final_token_index

            activations, centroids = extract_prompt_at_layers(
                model, input_ids, token_index, layers
            )
            for layer_idx in layers:
                layer_arrays_act[layer_idx][ti, whword_idx] = activations[layer_idx]
                layer_arrays_cen[layer_idx][ti, whword_idx] = centroids[layer_idx]

    results: list[ExtractionResult] = []
    for layer_idx in layers:
        path = save_per_template_npz(
            raw_dir / f"layer_{layer_idx:02d}_per_template.npz",
            activations=layer_arrays_act[layer_idx],
            centroids=layer_arrays_cen[layer_idx],
            concepts=list(whword_list),
            templates=list(template_list),
        )
        results.append(
            ExtractionResult(
                layer=layer_idx,
                per_template_path=path,
                activations=layer_arrays_act[layer_idx],
                centroids=layer_arrays_cen[layer_idx],
                concepts=tuple(whword_list),
                templates=template_list,
            )
        )

    variance_metrics = {
        str(layer_idx): compare_template_variance(
            layer_arrays_act[layer_idx], layer_arrays_cen[layer_idx]
        )
        for layer_idx in layers
    }
    save_metrics_json(
        raw_dir / "template_variance.json",
        {"model_name": model_name, "layers": variance_metrics},
    )
    return results


def extract_stopwords_experiment(
    *,
    model_name: str,
    layers: list[int],
    templates: tuple[str, ...] | list[str] | None = None,
    stopwords: tuple[str, ...] | list[str] | None = None,
    prompt_family: str = "neutral",
    raw_dir: str | Path,
    device: torch.device | None = None,
) -> list[ExtractionResult]:
    """Extract template × stopword tensors; one row per ``all_stopword_labels()`` order."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = device or resolve_device()

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    stopword_list = tuple(stopwords) if stopwords is not None else all_stopword_labels()
    records = expand_stopword_prompts(stopword_list, templates, prompt_family=prompt_family)
    audit_prompt_records(tokenizer, records)

    if templates is None:
        from lce.concepts import STOPWORD_PROMPT_FAMILIES

        template_list = STOPWORD_PROMPT_FAMILIES[prompt_family]
    else:
        template_list = tuple(templates)

    record_index: dict[tuple[str, str], dict[str, str]] = {}
    for r in records:
        value = str(r["value"])
        prompt = str(r["prompt"])
        for template in template_list:
            if prompt == template.format(value=value):
                record_index[(value, template)] = {
                    "concept": value,
                    "text": prompt,
                    "token_position": "final",
                }
                break
        else:
            raise KeyError(f"could not match prompt to template: {prompt!r}")

    n_templates = len(template_list)
    n_stopwords = len(stopword_list)
    hidden = int(model.config.n_embd)

    layer_arrays_act = {
        layer: np.zeros((n_templates, n_stopwords, hidden), dtype=np.float32) for layer in layers
    }
    layer_arrays_cen = {
        layer: np.zeros((n_templates, n_stopwords, hidden), dtype=np.float32) for layer in layers
    }

    for stopword_idx, stopword in enumerate(tqdm(stopword_list, desc="stopwords")):
        for ti, template in enumerate(template_list):
            record = record_index[(stopword, template)]
            encoded = tokenizer(record["text"], return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            audit = audit_prompt_records(tokenizer, [record])[0]
            token_index = audit.final_token_index

            activations, centroids = extract_prompt_at_layers(
                model, input_ids, token_index, layers
            )
            for layer_idx in layers:
                layer_arrays_act[layer_idx][ti, stopword_idx] = activations[layer_idx]
                layer_arrays_cen[layer_idx][ti, stopword_idx] = centroids[layer_idx]

    results: list[ExtractionResult] = []
    for layer_idx in layers:
        path = save_per_template_npz(
            raw_dir / f"layer_{layer_idx:02d}_per_template.npz",
            activations=layer_arrays_act[layer_idx],
            centroids=layer_arrays_cen[layer_idx],
            concepts=list(stopword_list),
            templates=list(template_list),
        )
        results.append(
            ExtractionResult(
                layer=layer_idx,
                per_template_path=path,
                activations=layer_arrays_act[layer_idx],
                centroids=layer_arrays_cen[layer_idx],
                concepts=tuple(stopword_list),
                templates=template_list,
            )
        )

    variance_metrics = {
        str(layer_idx): compare_template_variance(
            layer_arrays_act[layer_idx], layer_arrays_cen[layer_idx]
        )
        for layer_idx in layers
    }
    save_metrics_json(
        raw_dir / "template_variance.json",
        {"model_name": model_name, "layers": variance_metrics},
    )
    return results


def extract_languages_experiment(
    *,
    model_name: str,
    layers: list[int],
    templates: tuple[str, ...] | list[str] | None = None,
    languages: tuple[str, ...] | list[str] | None = None,
    prompt_family: str = "neutral",
    raw_dir: str | Path,
    device: torch.device | None = None,
) -> list[ExtractionResult]:
    """Extract template × language tensors; one row per ``all_language_labels()`` order."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = device or resolve_device()

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    language_list = tuple(languages) if languages is not None else all_language_labels()
    records = expand_language_prompts(language_list, templates, prompt_family=prompt_family)
    audit_prompt_records(tokenizer, records)

    if templates is None:
        from lce.concepts import LANGUAGE_PROMPT_FAMILIES

        template_list = LANGUAGE_PROMPT_FAMILIES[prompt_family]
    else:
        template_list = tuple(templates)

    record_index: dict[tuple[str, str], dict[str, str]] = {}
    for r in records:
        value = str(r["value"])
        prompt = str(r["prompt"])
        for template in template_list:
            if prompt == template.format(value=value):
                record_index[(value, template)] = {
                    "concept": value,
                    "text": prompt,
                    "token_position": "final",
                }
                break
        else:
            raise KeyError(f"could not match prompt to template: {prompt!r}")

    n_templates = len(template_list)
    n_languages = len(language_list)
    hidden = int(model.config.n_embd)

    layer_arrays_act = {
        layer: np.zeros((n_templates, n_languages, hidden), dtype=np.float32) for layer in layers
    }
    layer_arrays_cen = {
        layer: np.zeros((n_templates, n_languages, hidden), dtype=np.float32) for layer in layers
    }

    for language_idx, language in enumerate(tqdm(language_list, desc="languages")):
        for ti, template in enumerate(template_list):
            record = record_index[(language, template)]
            encoded = tokenizer(record["text"], return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            audit = audit_prompt_records(tokenizer, [record])[0]
            token_index = audit.final_token_index

            activations, centroids = extract_prompt_at_layers(
                model, input_ids, token_index, layers
            )
            for layer_idx in layers:
                layer_arrays_act[layer_idx][ti, language_idx] = activations[layer_idx]
                layer_arrays_cen[layer_idx][ti, language_idx] = centroids[layer_idx]

    results: list[ExtractionResult] = []
    for layer_idx in layers:
        path = save_per_template_npz(
            raw_dir / f"layer_{layer_idx:02d}_per_template.npz",
            activations=layer_arrays_act[layer_idx],
            centroids=layer_arrays_cen[layer_idx],
            concepts=list(language_list),
            templates=list(template_list),
        )
        results.append(
            ExtractionResult(
                layer=layer_idx,
                per_template_path=path,
                activations=layer_arrays_act[layer_idx],
                centroids=layer_arrays_cen[layer_idx],
                concepts=tuple(language_list),
                templates=template_list,
            )
        )

    variance_metrics = {
        str(layer_idx): compare_template_variance(
            layer_arrays_act[layer_idx], layer_arrays_cen[layer_idx]
        )
        for layer_idx in layers
    }
    save_metrics_json(
        raw_dir / "template_variance.json",
        {"model_name": model_name, "layers": variance_metrics},
    )
    return results

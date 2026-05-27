# Linear Centroid Experiments

Exploratory demo extending ideas from [**The Linear Centroids Hypothesis**](https://arxiv.org/pdf/2604.11962) (Walker et al., 2026).

We compare **MLP-input activation trajectories** and **MLP-local centroid trajectories** for simple concepts (starting with months) and analyze their geometry in full space before projecting to PCA.

> Exploratory interpretability demo — not a proof. See [docs/NOTES.md](docs/NOTES.md) and the live **[docs/index.html](docs/index.html)** viewer.

## Terminology (v1)

| Term | Meaning |
|------|---------|
| **Activation** | Pre-MLP LayerNorm (`ln_2`) output at the selected token |
| **Centroid** | **MLP-local** `μ(h) = ∇_h Σ_k MLP(h)_k` with the same `h` |

Not whole-model centroid or generic block centroid.

**Token position (v1):** final token only. Prompts end with the month name (no trailing punctuation):

```
The month is January
The calendar says January
The event happens in January
```

## What is a centroid?

For MLP input `h` and MLP output `f(h)`:

```
μ(h) = ∇_h Σ_k f(h)_k
```

Activations = *where* the token is represented; centroids = *local MLP computation* at that token.

Centroid computation: **`lce.centroid_adapter.compute_centroid` only**.

## Cyclic vs linear geometry

**Months** are **cyclic ordinal** (December→January closes the loop). For cycles, **lower curvature is not automatically better** — a clean month ring should show **stable curvature**, **even edges**, **closure ≈ typical edge**, and **high adjacent-neighbour accuracy**.

**Linear ordinal** concepts (future: years) use `geometry_kind="linear"` — no closure edge.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Third-party paper code

```bash
git submodule add https://github.com/ThomasWalker1/LinearCentroidsHypothesis.git third_party/LinearCentroidsHypothesis
git submodule update --init --recursive
```

## Full pipeline (DistilGPT-2 months)

Run extraction on a CUDA machine (recommended: Ubuntu GPU host):

```bash
python scripts/01b_audit_tokens.py
python scripts/02_extract_activations_centroids.py --config configs/months_distilgpt2.json
python scripts/03_compute_geometry.py --config configs/months_distilgpt2.json
python scripts/04_make_plots.py --config configs/months_distilgpt2.json
python scripts/05_build_site.py --config configs/months_distilgpt2.json
```

Quick demo subset (layers 0, 3, 5): use `configs/months_distilgpt2_demo.json`.

### Colour ordering topology (scaffolding)

```bash
python scripts/01c_generate_colour_prompts.py
python scripts/02_extract_activations_centroids.py --config configs/colours_distilgpt2.json
python scripts/06_colour_ordering_metrics.py --config configs/colours_distilgpt2.json
python scripts/07_colour_make_plots.py --config configs/colours_distilgpt2.json --sync-docs
```

### Random nouns (no natural order — random baseline control)

Seven unrelated nouns; one arbitrary `list_order` compared to random permutations.

```bash
python scripts/01d_generate_noun_prompts.py
python scripts/02_extract_activations_centroids.py --config configs/nouns_distilgpt2.json
python scripts/08_noun_ordering_metrics.py --config configs/nouns_distilgpt2.json
python scripts/09_noun_make_plots.py --config configs/nouns_distilgpt2.json --sync-docs
```

### Tools (imposed list order — related objects)

Ten common tools in a fixed list order, compared to random permutations.

```bash
python scripts/01e_generate_tool_prompts.py
python scripts/02_extract_activations_centroids.py --config configs/tools_distilgpt2.json
python scripts/10_tool_ordering_metrics.py --config configs/tools_distilgpt2.json
python scripts/11_tool_make_plots.py --config configs/tools_distilgpt2.json --sync-docs
```

### Animals (imposed list order — diverse species)

Twelve animals in a fixed list order, compared to random permutations.

```bash
python scripts/01f_generate_animal_prompts.py
python scripts/02_extract_activations_centroids.py --config configs/animals_distilgpt2.json
python scripts/12_animal_ordering_metrics.py --config configs/animals_distilgpt2.json
python scripts/13_animal_make_plots.py --config configs/animals_distilgpt2.json --sync-docs
```

See [docs/index.html](docs/index.html) for the interactive viewer and notes.

Outputs land under `results/`; GitHub Pages assets under `docs/experiments/<experiment>/`.

Preview locally: `python -m http.server -d docs 8000` → http://localhost:8000

## Key metrics

| Metric | Role |
|--------|------|
| `edge_lengths` | Step sizes; cyclic includes Dec→Jan |
| `edge_length_cv` | Even spacing |
| `closure_to_edge_ratio` | Dec→Jan vs mean other edges (~1 for clean cycle) |
| `closure_to_path_ratio` | Closure vs total open arc |
| `curvature_smoothness_*` | Stable bending (not “low κ”) |
| `adjacent_neighbour_accuracy` | Embedding respects month order |
| `ρ_d, ρ_T, ρ_N` | PCA visibility |
| `tangent_alignment_full_vs_pca`, `normal_alignment_full_vs_pca` | PCA geometric fidelity |
| `template_variance` | Prompt sensitivity (activation vs centroid) |

Full definitions: [docs/index.html](docs/index.html) · [docs/NOTES.md](docs/NOTES.md)

**No composite winner score** — exploratory panels only. Details: [docs/DECISIONS.md](docs/DECISIONS.md), [docs/NOTES.md](docs/NOTES.md).

## Pipeline

| Script | Purpose |
|--------|---------|
| `00_check_environment.py` | Dependencies + submodule |
| `01_generate_prompts.py` | Month prompts JSON |
| `01b_audit_tokens.py` | Tokenizer audit (fail if final token ∉ month) |
| `02_extract_activations_centroids.py` | ln_2 + MLP-local centroid on GPU (TODO) |
| `03_compute_geometry.py` | Geometry on template means |
| `04_make_plots.py` | Plotly panels (TODO) |
| `05_build_site.py` | GitHub Pages `docs/` |
| `01c_generate_colour_prompts.py` | Colour prompts JSONL |
| `01d_generate_noun_prompts.py` | Random noun prompts JSONL |
| `06_colour_ordering_metrics.py` | Colour ordering vs random baselines |
| `07_colour_make_plots.py` | Colour PCA path plots |
| `08_noun_ordering_metrics.py` | Noun list order vs random baselines |
| `09_noun_make_plots.py` | Noun PCA path plots |

Full extraction: layers `[0..5]`; demo subset `[0,3,5]` in `configs/months_distilgpt2_demo.json`.

## Citation

```bibtex
@article{walker2026lch,
  title={The Linear Centroids Hypothesis: How Deep Network Features Represent Data},
  author={Walker, Thomas and Humayun, Ahmed Imtiaz and Balestriero, Randall and Baraniuk, Richard},
  journal={arXiv:2604.11962},
  year={2026}
}
```

MIT — [LICENSE](LICENSE)

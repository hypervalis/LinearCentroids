# Linear Centroid Experiments

Exploratory demo extending ideas from [**The Linear Centroids Hypothesis**](https://arxiv.org/pdf/2604.11962) (Walker et al., 2026).

We compare **MLP-input activation trajectories** and **MLP-local centroid trajectories** for simple concepts and analyze their geometry in full space before projecting to PCA.

> Exploratory interpretability demo — not a proof. See [docs/NOTES.md](docs/NOTES.md) and the live **[docs/index.html](docs/index.html)** viewer.

## Repository layout

```
lce/                  # Core library (geometry, extraction, concepts, pipeline, CLI)
  cli/                # `python -m lce` entry points
  pipeline/           # Config-driven pipeline stages
configs/              # One JSON config per experiment
docs/                 # GitHub Pages site + generated experiment assets
scripts/              # Thin wrappers around the CLI (optional)
tests/
data/prompts/         # Generated prompt files
results/              # Local extraction outputs (mostly gitignored)
```

## Terminology (v1)

| Term | Meaning |
|------|---------|
| **Activation** | Pre-MLP LayerNorm (`ln_2`) output at the selected token |
| **Centroid** | **MLP-local** `μ(h) = ∇_h Σ_k MLP(h)_k` with the same `h` |

Not whole-model centroid or generic block centroid.

**Token position (v1):** final token only, with a tokenizer audit.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
lce check-environment
```

Optional paper submodule:

```bash
git submodule add https://github.com/ThomasWalker1/LinearCentroidsHypothesis.git third_party/LinearCentroidsHypothesis
git submodule update --init --recursive
```

## CLI

All pipeline stages are config-driven:

```bash
lce generate-prompts --config configs/months_distilgpt2.json
lce audit-tokens
lce extract --config configs/months_distilgpt2.json
lce compute-geometry --config configs/months_distilgpt2.json   # months only
lce make-plots --config configs/months_distilgpt2.json --sync-docs
lce build-site --config configs/months_distilgpt2.json
```

Ordering experiments (colours, nouns, tools, animals, years, …):

```bash
lce generate-prompts --config configs/colours_distilgpt2.json
lce extract --config configs/colours_distilgpt2.json
lce ordering-metrics --config configs/colours_distilgpt2.json
lce make-plots --config configs/colours_distilgpt2.json --sync-docs
```

Equivalent thin wrappers live under `scripts/` (e.g. `python scripts/extract.py --config …`).

Run extraction on a CUDA machine when possible. Configs live in `configs/`; outputs under `results/` and synced plots under `docs/experiments/<experiment>/`.

## GitHub Pages

The site is published from the `docs/` folder via GitHub Actions (`.github/workflows/pages.yml`).

After pushing to `main`, enable **Settings → Pages → Build and deployment → GitHub Actions** (if not already).

Live URL: **https://hypervalis.github.io/LinearCentroids/**

Local preview with save support:

```bash
python scripts/dev_site.py
# http://127.0.0.1:8765/index.html
```

Plain `python -m http.server` does not support **Save to file** in the editor.

## Key metrics

| Metric | Role |
|--------|------|
| `edge_lengths` | Step sizes; cyclic includes Dec→Jan |
| `edge_length_cv` | Even spacing |
| `closure_to_edge_ratio` | Dec→Jan vs mean other edges (~1 for clean cycle) |
| `ρ_d, ρ_T, ρ_N` | PCA visibility of full-space geometry |
| `template_variance` | Prompt sensitivity (activation vs centroid) |

Full definitions: [docs/index.html](docs/index.html) · [docs/NOTES.md](docs/NOTES.md) · [docs/DECISIONS.md](docs/DECISIONS.md)

**No composite winner score** — exploratory panels only.

## Tests

```bash
pytest
```

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

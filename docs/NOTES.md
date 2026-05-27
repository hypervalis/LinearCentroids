# Project notes

See [DECISIONS.md](DECISIONS.md) for locked v1 choices.

## Scientific framing

We compare **ln_2 activations** vs **MLP-local centroids** along imposed concept sequences.
Months are **cyclic ordinal** (December→January closes the loop).

**Important:** for cyclic concepts, **lower mean curvature is not automatically better**.
A regular circle has substantial and **uniform** curvature. What matters for months:

- coherent / stable curvature (`curvature_smoothness`)
- even step sizes (`edge_length_cv`)
- good cyclic closure (`closure_to_edge_ratio` ≈ 1)
- ordinal neighbours in embedding space (`adjacent_neighbour_accuracy`)

Linear ordinal concepts (e.g. years) use `geometry_kind="linear"`: no December→January edge.

**No composite winner score** — exploratory metric panels and narrative only.

## Terminology (v1)

| Term | Definition |
|------|------------|
| **Activation** | Pre-MLP LayerNorm output (`ln_2`) at the selected token |
| **Centroid** | `μ(h) = ∇_h Σ_k MLP(h)_k` with the same `h` (MLP-local; not whole-model) |

## Token position (v1)

Suffix-free templates; month is final:

- `The month is January`
- `The calendar says January`
- `The event happens in January`

**Final token index** + tokenizer audit (fail if final token ∉ month). TODO: span extraction.

## Pipeline data flow

```
prompts × months
    → extract per (template, month): activation h, centroid μ  [raw npz]
    → template variance metrics
    → mean over templates → 12 points per layer per representation
    → geometry + PCA metrics
    → plots → docs/
```

## Metrics

### Cyclic / ordinal geometry (`lce.geometry`)

| Metric | Meaning |
|--------|---------|
| `edge_lengths` | Step sizes; cyclic includes Dec→Jan |
| `edge_length_cv` | `std / mean` — even spacing |
| `closure_to_edge_ratio` | Dec→Jan vs mean(other edges) ≈ 1 for clean cycle |
| `closure_to_path_ratio` | closure / sum(non-closure edges) |
| `curvature_smoothness_std` | `std(κ)` |
| `curvature_smoothness_step_mean` | mean `\|κ_{t+1} − κ_t\|` |
| `adjacent_neighbour_accuracy` | NN = prev/next in order |

Plus PCA fidelity: `ρ_d`, `ρ_T`, `ρ_N`, frame angles.

### Template variance (`lce.metrics`)

From `(n_templates, n_months, dim)` before averaging. Compare activation vs centroid.

## Synthetic vs real

| | Synthetic circle | Real months |
|--|------------------|-------------|
| Purpose | Metric validation | Main experiment |
| Shape | Regular 12-gon | Unknown; not required to be circular |
| κ | Stable, nonzero | Report smoothness + closure, not “low κ” |

## Site (v1)

Single `docs/index.html`: dropdowns (concept / model / layer / activation|centroid), layer summary table with structure flags, embedded Plotly HTML fragments.

## Reading results

Prefer ordinal + closure + neighbour metrics over raw mean curvature. Pair PCA plots with `ρ` scores.

## Colour ordering topology (scaffolding)

Colours are **categorical**, not naturally ordinal like months. We impose named orderings
(ROYGBIV, hue wheel, warm→cool, etc.) and ask whether each path follows short edges in
activation/centroid space **better than random permutations**.

**Central question:** do imposed orderings trace intrinsic geometry, or arbitrary paths
through a semantic point cloud?

| Ordering | Primary mode | Notes |
|----------|--------------|-------|
| ROYGBIV | non-cyclic | cyclic closure as diagnostic only |
| Hue wheel (approx.) | cyclic | loop candidate — do not assume closure is “good” for ROYGBIV |
| Warm→cool, light→dark, basic | non-cyclic | semantic/perceptual paths |

Key metrics (`lce/topology.py`):

- `path_energy` + **percentile vs random** (higher = geometry supports ordering)
- `mean_edge_rank`, `adjacent_edge_percentile` (lower = follows nearby neighbours)
- `closure_to_edge_ratio` — cyclic diagnostics only

**Do not claim a hue circle** unless hue-wheel cyclic metrics beat random with closure ≈ 1,
low edge CV, good ranks, and stability across layers/templates.

If prompt family changes which orderings look natural, geometry may be **context-conditioned**.

**TODO:** persistent homology / Vietoris-Rips H1 loop score for expanded colour sets.

Docs: [index.html](index.html)

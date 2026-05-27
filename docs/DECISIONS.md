# Locked decisions

These choices are fixed for v1 unless explicitly revised.

## Extraction hook (DistilGPT-2)

| Field | Decision |
|-------|----------|
| **Activation site** | Pre-MLP LayerNorm output (`ln_2`) at the selected token |
| **Centroid** | `μ(h) = ∇_h Σ_k MLP(h)_k` where `h` is that same `ln_2` output |
| **Not used** | MLP output as activation; whole-model / block-level centroids |

Per transformer block `i` (0-based, six blocks in DistilGPT-2):

```
h = ln_2(block_i)(residual_after_attn)   # activation saved here
y = mlp(block_i)(h)
μ = ∇_h sum(y)
```

## Layer indexing

- **0-based** DistilGPT-2 block indices.
- **Full runs:** all six layers `[0, 1, 2, 3, 4, 5]`.
- **Quick demo:** subset `[0, 3, 5]` only (see `configs/months_distilgpt2_demo.json`).

## Averaging & template variance

- **Geometry:** computed on **template-mean** trajectories (one point per month).
- **Template variance:** computed separately from raw `(n_templates, n_months, dim)` tensors; saved and reported alongside geometry.

## Token position

- Suffix-free templates; month name is the **final surface token**.
- Use **final token index** for v1.
- Run **tokenizer audit** before extraction; **fail loudly** if the final token is not part of the month string.
- Robust concept-token span extraction: **TODO**.

## Interpretation

- **No composite “winner” score.** Exploratory only.
- Report metric panels and narrative observations — not “centroid wins activation.”

## Synthetic benchmarks

- Validate metric implementations only.
- Circle: stable **nonzero** κ, high top-2 PCA concentration, low `edge_length_cv`, `closure_to_edge_ratio` ≈ 1.
- Real months **need not** look like a perfect circle.

## GitHub Pages site

- **One page** with dropdowns: concept / model / layer / space (activation vs centroid).
- Layer **summary table** with simple flags for layers showing meaningful structure.
- Repo: **`linear-centroid-experiments`**, public, Pages from **`/docs` on `main`**.

### v1 required plots

1. 3D PCA path + tangent arrows coloured by `ρ_d`
2. Curvature plot
3. Edge / closure panel
4. Layer comparison table
5. Template variance panel

Optional: normal arrows, distance heatmaps.

## Paper third-party code

- v1 extraction uses **`lce.centroid_adapter.compute_centroid`** (autograd) only.
- Paper submodule kept for **attribution / future wrapping**; not required at runtime.

## Hardware

- Run extraction on **Ubuntu GPU box** (CUDA when available).
- Not optimized for laptop CPU.
- DistilGPT-2, all six layers for full runs, is fine.

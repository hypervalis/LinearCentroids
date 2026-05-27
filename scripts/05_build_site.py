#!/usr/bin/env python3
"""Assemble static GitHub Pages site under docs/."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lce.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "months_distilgpt2.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics_path = ROOT / cfg["output"]["metrics_dir"] / "geometry_metrics.json"
    plots_dir = ROOT / cfg["output"]["plots_dir"]
    docs_dir = ROOT / "docs"
    experiments_dir = docs_dir / "experiments" / cfg.get("experiment", "default")
    assets_plots = experiments_dir / "plots"

    if not metrics_path.is_file():
        print(f"Missing {metrics_path}; run 03_compute_geometry.py first.")
        sys.exit(1)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assets_plots.mkdir(parents=True, exist_ok=True)

    if plots_dir.is_dir():
        for html in plots_dir.glob("*_pca3d.html"):
            shutil.copy2(html, assets_plots / html.name)
        manifest_src = plots_dir / "manifest.json"
        if manifest_src.is_file():
            shutil.copy2(manifest_src, experiments_dir / "manifest.json")

    metrics_out = experiments_dir / "geometry_metrics.json"
    metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    layers = sorted({int(row["layer"]) for row in metrics.get("layer_summary", [])})
    reps = ["activation", "centroid"]

    index_html = docs_dir / "index.html"
    index_html.write_text(
        _render_index(
            model=cfg.get("model_name", "distilgpt2"),
            months_layers=layers,
            representations=reps,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {index_html.resolve()}")
    print(f"Assets: {experiments_dir.resolve()}")


def _render_index(
    *,
    model: str,
    months_layers: list[int],
    representations: list[str],
) -> str:
    rep_opts = "\n".join(f'<option value="{r}">{r}</option>' for r in representations)
    layer_opts = "\n".join(f'<option value="{l}">{l}</option>' for l in months_layers)
    explore_config = json.dumps(
        {
            "model": model,
            "monthsLayers": months_layers,
            "representations": representations,
            "defaultPlots": {
                "colours_distilgpt2": {"ordering": "roygbiv", "mode": "non_cyclic"},
                "nouns_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
                "tools_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
                "animals_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
                "years_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
                "disciplines_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
                "whwords_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
                "stopwords_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
                "languages_distilgpt2": {"ordering": "list_order", "mode": "non_cyclic"},
            },
        }
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Linear Centroid Experiments</title>
  <link rel="stylesheet" href="assets/style.css" />
  <link rel="stylesheet" href="assets/content-editor.css" />
</head>
<body>
  <main>
    <header class="page-header">
      <p class="version-tag">Version 1</p>
      <div class="page-title" data-editable="title">Linear Centroid Experiments</div>
      <div class="page-header-copy" data-editable="header-copy">
        <p>
          DistilGPT-2 is probed with short prompts (e.g. “The month is January”).
          For each concept we take the pre-MLP activation <code>h</code> and the
          local MLP centroid <code>μ(h)</code> at the concept token, average over
          template wordings, connect concepts in an imposed order, and plot the
          path in 3D PCA.
        </p>
        <p>
          Points = <code>h</code>; orange arrows = <code>μ(h)</code> (activation view).
          Marker colour <code>ρ_d</code> shows how much of each step is visible in
          the projection — not whether the imposed order is “true”.
        </p>
        <p class="meta-links">
          <a href="https://arxiv.org/pdf/2604.11962">Paper</a>
          · <a class="edit-page-link edit-entry" data-edit-on href="#">Edit text</a>
        </p>
      </div>
    </header>

    <section class="viewer">
      <div class="controls">
        <label>Concept
          <select id="concept" data-field="concept">
            <option value="months_distilgpt2">Months</option>
            <option value="colours_distilgpt2">Colours</option>
            <option value="nouns_distilgpt2">Random nouns</option>
            <option value="tools_distilgpt2">Tools</option>
            <option value="animals_distilgpt2">Animals</option>
            <option value="years_distilgpt2">Years (1990–2013)</option>
            <option value="disciplines_distilgpt2">Disciplines</option>
            <option value="whwords_distilgpt2">Question words (Who…How)</option>
            <option value="stopwords_distilgpt2">Stopwords</option>
            <option value="languages_distilgpt2">Programming languages</option>
          </select>
        </label>
        <label>Layer <select data-field="layer">{layer_opts}</select></label>
        <label>Space <select data-field="representation">{rep_opts}</select></label>
        <label class="hidden">Ordering <select data-field="ordering"></select></label>
        <label class="hidden">Mode <select data-field="mode"></select></label>
        <label class="control-checkbox">Show path
          <input type="checkbox" id="show-path" data-field="show-path" />
        </label>
      </div>

      <iframe id="plot" title="PCA plot" loading="lazy"></iframe>
    </section>

    <section class="page-section">
      <div class="page-header-copy" data-editable="plot-copy">
        <p>
          <strong>Months</strong> have a natural cycle; <strong>years</strong> a
          chronological line. <strong>Colours</strong> test imposed orders such as
          ROYGBIV against the same seven labels in other sequences. The other sets
          are list-order baselines — useful nulls, not claims of a real axis.
        </p>
        <p>
          <strong>Activation</strong> is where the representation sits;
          <strong>centroid</strong> is the local MLP map at that point (mean of
          per-template <code>μ(h)</code>, not <code>μ</code> of the mean
          activation). Geometry is computed in full space first; PCA is only the
          picture.
        </p>
      </div>
    </section>
  </main>

  <script>window.EXPLORE_CONFIG = {explore_config};</script>
  <script src="assets/explore.js"></script>
  <script src="assets/content-editor.js?v=6"></script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

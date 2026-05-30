#!/usr/bin/env python3
"""Add headings and MathJax-friendly markup to docs/index.html."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.html"

MATHJAX_HEAD = """
  <meta name="description" content="Exploratory Linear Centroid Hypothesis experiments on DistilGPT-2 activation and MLP-local centroid geometry." />
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [["\\\\(", "\\\\)"], ["$", "$"]],
        displayMath: [["\\\\[", "\\\\]"], ["$$", "$$"]],
        processEscapes: true,
      },
      options: { skipHtmlTags: ["script", "noscript", "style", "textarea", "code"] },
    };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
"""

HEADER_REPLACEMENT = """
        <p><img src="assets/content/header-lch-overview.png" alt="Concept trajectories in activation and centroid space" class="content-image"></p>

        <h2 id="background">Background</h2>
        <p>
          I came across
          <a href="https://arxiv.org/pdf/2604.11962">The Linear Centroid Hypothesis</a>
          (Walker et al., 2026). The basic idea is that a deep network at a given layer can be
          approximated locally by an affine map, and that the neighbourhood where that approximation
          holds can be studied as a local expert.
        </p>
        <p>
          The contrast is with the <strong>Linear Representation Hypothesis (LRH)</strong>, which
          postulates that dimensions in activation space correspond cleanly and globally to discrete
          conceptual features — a nice idea, but almost certainly too simple. The
          <strong>Linear Centroid Hypothesis (LCH)</strong> makes a weaker but more plausible claim:
          activation space can be subdivided into local experts characterised by roughly affine
          transformations; within each expert one can find locally Euclidean coordinates that yield
          features that are more interpretable there than in raw activation coordinates.
        </p>
        <p>
          I like this because it echoes manifolds: globally non-Euclidean, everywhere locally
          Euclidean — which hints at differential-geometric tools for activation space. For example,
          could we define parallel transport for steering vectors, or characterise global geometry?
          If LRH held globally, perhaps the relevant geometry would be \(\mathbb{R}^d\); if not,
          perhaps failure modes look more like hyperbolic or spherical structure. But that is for
          another day.
        </p>

        <h2 id="centroids">Centroid vectors</h2>
        <p>
          The main objects here are layer Jacobians. For a transformer block, back-propagate from the
          MLP output to the pre-MLP LayerNorm output \(h\), then take the row-sum of the Jacobian.
          At token \(x\), with MLP map \(f\):
        </p>
        <div class="math-block">
          \\[ \\mu(h) = \\nabla_h \\sum_k f(h)_k \\]
        </div>
        <p>
          Activations tell you <em>where</em> the representation sits; centroids summarise the
          local MLP computation at that point. A related construction appears in
          <a href="https://openreview.net/pdf?id=oDWbJsIuEp">Equivalent Mappings of Large Language Models</a>.
        </p>

        <h2 id="method">What we plot</h2>
        <p>
          I started with ordinal concepts (months), PCA paths, and Frenet-like frames, but the more
          interesting move is to treat centroid vectors as <em>tangent-ish</em> directions and compare
          them with activations across several concept lists and layers of DistilGPT-2.
        </p>
        <p>
          Geometry is computed in full hidden space first; PCA is only a 3D picture. In the
          interactive plot below, marker colour is
          \\(\\rho_d\\): the fraction of each step vector visible in the PCA projection (relevant
          mainly when a path ordering is imposed).
        </p>
        <p>
          Orange arrows (activation view) show \\(\\mu(h)\\) projected into the PCA basis fit on
          activations — not the same basis as the centroid-only view.
        </p>
        <p class="lede-close">
          The goal is not a grand theory, but to explore whether centroid trajectories carry
          geometry that activation trajectories alone do not.
        </p>
"""

PLOT_HEADING_INJECTIONS = [
    (
        "So we've got a few different concepts here",
        '<p class="section-lede">So we\'ve got a few different concepts here, and some interesting findings. Lets go through them one by one.</p><h2 id="months">Months</h2>',
    ),
    (
        "<div>Moving on. We will look at colors now.",
        '<h2 id="colours">Colours</h2><div>Moving on. We will look at colors now.',
    ),
    (
        "What we should do next is look at a set of random words",
        '<h2 id="nouns">Random nouns</h2><div>What we should do next is look at a set of random words',
    ),
    (
        "Moving on, I'm going to skip the tools concept set",
        '<h2 id="tools">Tools</h2><p class="aside">Moving on, I\'m going to skip the tools concept set',
    ),
    (
        "The next concept we are going to look at is animals.",
        '<h2 id="animals">Animals</h2><div>The next concept we are going to look at is animals.',
    ),
    (
        "Now we're going to move onto years.",
        '<h2 id="years">Years</h2><div>Now we\'re going to move onto years.',
    ),
    (
        "Beyond this, there are few more concept sets",
        '<h2 id="other-concepts">Other concept sets</h2><p>Beyond this, there are few more concept sets',
    ),
    (
        "So what does it all mean?",
        '<h2 id="closing">Closing thoughts</h2><div>So what does it all mean?',
    ),
]

# Remove duplicate lede after injection
PLOT_REMOVE = "So we've got a few different concepts here, and some interesting findings. Lets go through them one by one.<br><br><div>The first concept here is months."


def replace_div_inner(html: str, editable_id: str, new_inner: str) -> str:
    marker = f'data-editable="{editable_id}"'
    pos = html.find(marker)
    if pos == -1:
        raise KeyError(editable_id)
    open_start = html.rfind("<div", 0, pos)
    open_end = html.find(">", open_start) + 1
    depth = 1
    i = open_end
    while i < len(html) and depth:
        next_open = html.find("<div", i)
        next_close = html.find("</div>", i)
        if next_close == -1:
            raise ValueError(f"unclosed {editable_id}")
        if next_open != -1 and next_open < next_close:
            depth += 1
            i = next_open + 4
        else:
            depth -= 1
            if depth == 0:
                close_start = next_close
                break
            i = next_close + 6
    return html[:open_end] + new_inner + html[close_start:]


def main() -> None:
    html = INDEX.read_text(encoding="utf-8")

    if "mathjax@3" not in html:
        html = html.replace(
            '  <link rel="stylesheet" href="assets/content-editor.css" />',
            '  <link rel="stylesheet" href="assets/content-editor.css" />' + MATHJAX_HEAD,
        )

    html = html.replace(
        '<div class="page-title" data-editable="title">',
        '<h1 class="page-title" data-editable="title">',
    ).replace(
        '</div>\n      <div class="page-header-copy" data-editable="header-copy">',
        '</h1>\n      <div class="page-header-copy prose" data-editable="header-copy">',
    )

    html = replace_div_inner(html, "header-copy", HEADER_REPLACEMENT)

    plot_start = html.index('data-editable="plot-copy"')
    plot_open = html.find(">", plot_start) + 1
    plot_close = html.index('</div>\n      <p class="meta-links edit-entry">', plot_start)
    plot_inner = html[plot_open:plot_close]

    plot_inner = plot_inner.replace(
        "So we've got a few different concepts here, and some interesting findings. Lets go through them one by one.<br><br><div>The first concept here is months.",
        '<p class="section-lede">So we\'ve got a few different concepts here, and some interesting findings. Lets go through them one by one.</p><h2 id="months">Months</h2><div>The first concept here is months.',
    )
    for needle, repl in PLOT_HEADING_INJECTIONS[1:]:
        plot_inner = plot_inner.replace(needle, repl)

    plot_inner = plot_inner.replace(
        '<div class="page-header-copy" data-editable="plot-copy">',
        "",
    )
    html = html[:plot_open] + plot_inner + html[plot_close:]

    html = html.replace(
        '    <section class="viewer">',
        '    <section class="viewer" id="explorer">',
    )
    html = html.replace(
        '      <div class="controls">',
        '      <h2 class="viewer-heading">Interactive explorer</h2>\n      <p class="viewer-hint">DistilGPT-2 · template-mean trajectories · 3D PCA projection. Use the controls to change concept, layer, and representation.</p>\n      <div class="controls">',
    )

    html = html.replace(
        '    <section class="page-section">',
        '    <section class="page-section" id="findings">',
    )
    html = html.replace(
        '      <div class="page-header-copy" data-editable="plot-copy">',
        '      <h2 class="section-heading">Findings</h2>\n      <div class="page-header-copy prose" data-editable="plot-copy">',
    )

    html = html.replace(
        'its dynamics.</div><div><br></div></div>',
        'its dynamics.</div>',
    )

    INDEX.write_text(html, encoding="utf-8")
    print(f"Polished {INDEX}")


if __name__ == "__main__":
    main()

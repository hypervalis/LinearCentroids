/**
 * Version 1 — single viewer, concept types in one dropdown.
 */

(function () {
  const CFG = window.EXPLORE_CONFIG || {};
  const MONTHS_LAYERS = CFG.monthsLayers || [0, 1, 2, 3, 4, 5];
  const DEFAULT_PLOTS = CFG.defaultPlots || {};

  const ORDERING = {
    colours_distilgpt2: {
      slug: "colours_distilgpt2",
      manifestUrl: "experiments/colours_distilgpt2/manifest.json",
    },
    nouns_distilgpt2: {
      slug: "nouns_distilgpt2",
      manifestUrl: "experiments/nouns_distilgpt2/manifest.json",
    },
    tools_distilgpt2: {
      slug: "tools_distilgpt2",
      manifestUrl: "experiments/tools_distilgpt2/manifest.json",
    },
    animals_distilgpt2: {
      slug: "animals_distilgpt2",
      manifestUrl: "experiments/animals_distilgpt2/manifest.json",
    },
    years_distilgpt2: {
      slug: "years_distilgpt2",
      manifestUrl: "experiments/years_distilgpt2/manifest.json",
    },
    disciplines_distilgpt2: {
      slug: "disciplines_distilgpt2",
      manifestUrl: "experiments/disciplines_distilgpt2/manifest.json",
    },
    whwords_distilgpt2: {
      slug: "whwords_distilgpt2",
      manifestUrl: "experiments/whwords_distilgpt2/manifest.json",
    },
    stopwords_distilgpt2: {
      slug: "stopwords_distilgpt2",
      manifestUrl: "experiments/stopwords_distilgpt2/manifest.json",
    },
    languages_distilgpt2: {
      slug: "languages_distilgpt2",
      manifestUrl: "experiments/languages_distilgpt2/manifest.json",
    },
  };

  const orderingManifests = {};

  const $ = (sel) => document.querySelector(sel);

  function concept() {
    return $("#concept")?.value || "months_distilgpt2";
  }

  function isOrdering() {
    return concept() in ORDERING;
  }

  function pad(n) {
    return String(n).padStart(2, "0");
  }

  function field(name) {
    return $(`[data-field="${name}"]`)?.value ?? null;
  }

  function setField(name, val) {
    const el = $(`[data-field="${name}"]`);
    if (el && val != null) el.value = String(val);
  }

  function setOptions(sel, values, keep) {
    if (!sel) return;
    sel.innerHTML = values.map((v) => `<option value="${v}">${v}</option>`).join("");
    if (keep != null && values.map(String).includes(String(keep))) sel.value = String(keep);
  }

  function pathVisible() {
    return $("#show-path")?.checked ?? false;
  }

  function defaultPlotSpec(exp) {
    return (
      DEFAULT_PLOTS[exp] || {
        ordering: "list_order",
        mode: "non_cyclic",
      }
    );
  }

  function currentPlotSpec() {
    const exp = concept();
    const defaults = defaultPlotSpec(exp);
    if (!isOrdering()) return null;

    const layer = Number(field("layer"));
    const rep = field("representation");
    const figures = manifestFigures(exp);
    const pool = figures.filter((f) => f.layer === layer && f.representation === rep);
    const candidates = pool.length ? pool : figures;

    const ordering = field("ordering") || defaults.ordering;
    const mode = field("mode") || defaults.mode;
    const match = candidates.find((f) => f.ordering === ordering && f.mode === mode);
    if (match) return { ordering: match.ordering, mode: match.mode };

    const fallback =
      candidates.find(
        (f) => f.ordering === defaults.ordering && f.mode === defaults.mode
      ) || candidates[0];
    if (fallback) return { ordering: fallback.ordering, mode: fallback.mode };
    return defaults;
  }

  function manifestFigures(exp) {
    const slug = ORDERING[exp]?.slug;
    return slug ? orderingManifests[slug]?.figures || [] : [];
  }

  function syncOrderingControls() {
    const orderingEl = $('[data-field="ordering"]');
    const modeEl = $('[data-field="mode"]');
    const orderingLabel = orderingEl?.closest("label");
    const modeLabel = modeEl?.closest("label");
    const exp = concept();

    if (!isOrdering() || !orderingEl || !modeEl) {
      orderingLabel?.classList.add("hidden");
      modeLabel?.classList.add("hidden");
      return;
    }

    const layer = Number(field("layer"));
    const rep = field("representation");
    const pool = manifestFigures(exp).filter(
      (f) => f.layer === layer && f.representation === rep
    );
    const figures = pool.length ? pool : manifestFigures(exp);
    const orderings = [...new Set(figures.map((f) => f.ordering))];
    const modes = [...new Set(figures.map((f) => f.mode))];
    const showOrdering = orderings.length > 1 || modes.length > 1;

    if (!showOrdering) {
      orderingLabel?.classList.add("hidden");
      modeLabel?.classList.add("hidden");
      const defaults = defaultPlotSpec(exp);
      setOptions(orderingEl, orderings.length ? orderings : [defaults.ordering], defaults.ordering);
      const modesForDefault = modes.length ? modes : [defaults.mode];
      setOptions(modeEl, modesForDefault, defaults.mode);
      return;
    }

    orderingLabel?.classList.remove("hidden");
    modeLabel?.classList.remove("hidden");

    const defaults = defaultPlotSpec(exp);
    const keepOrdering = field("ordering") || defaults.ordering;
    setOptions(orderingEl, orderings, keepOrdering);

    const selectedOrdering = field("ordering") || orderings[0];
    const modesForOrdering = [
      ...new Set(
        figures.filter((f) => f.ordering === selectedOrdering).map((f) => f.mode)
      ),
    ];
    const keepMode = field("mode") || defaults.mode;
    setOptions(modeEl, modesForOrdering.length ? modesForOrdering : modes, keepMode);
  }

  function syncUrl() {
    const current = new URLSearchParams(location.search);
    const p = new URLSearchParams();
    p.set("concept", concept());
    p.set("layer", field("layer"));
    p.set("representation", field("representation"));
    if (isOrdering()) {
      const spec = currentPlotSpec();
      if (spec) {
        p.set("ordering", spec.ordering);
        p.set("mode", spec.mode);
      }
    }
    if (pathVisible()) p.set("path", "1");
    if (current.get("edit") === "1") p.set("edit", "1");
    history.replaceState(null, "", `?${p.toString()}`);
  }

  function applyUrl() {
    const p = new URLSearchParams(location.search);
    const exp = p.get("concept") || p.get("experiment");
    if (exp) setField("concept", exp);
    if (p.has("layer")) setField("layer", p.get("layer"));
    if (p.has("representation")) setField("representation", p.get("representation"));
    if (p.has("ordering")) setField("ordering", p.get("ordering"));
    if (p.has("mode")) setField("mode", p.get("mode"));
    const showPath = $("#show-path");
    if (showPath) showPath.checked = p.get("path") === "1";
  }

  function plotPath() {
    const layer = field("layer");
    const rep = field("representation");
    const exp = concept();
    if (exp === "months_distilgpt2") {
      return `experiments/months_distilgpt2/plots/layer_${pad(layer)}_${rep}_pca3d.html`;
    }
    const meta = ORDERING[exp];
    const spec = currentPlotSpec();
    return `experiments/${meta.slug}/plots/layer_${pad(layer)}_${rep}_${spec.ordering}_${spec.mode}.html`;
  }

  function prepareIframeDocument(doc) {
    if (!doc) return;
    doc.documentElement.style.height = "100%";
    doc.body.style.margin = "0";
    doc.body.style.height = "100%";
    doc.body.style.overflow = "hidden";
  }

  function waitForPlotLayout(gd, timeoutMs = 4000) {
    return new Promise((resolve) => {
      const start = Date.now();
      const tick = () => {
        if (gd._fullLayout?.scene?.xaxis && gd.data?.length) resolve(true);
        else if (Date.now() - start > timeoutMs) resolve(false);
        else setTimeout(tick, 40);
      };
      tick();
    });
  }

  function applyPathVisibility(gd, Plotly) {
    const pathIdx = gd.data.findIndex((t) => t.name === "path");
    if (pathIdx < 0) return;

    const show = pathVisible();
    Plotly.restyle(
      gd,
      {
        mode: show ? "lines+markers+text" : "markers+text",
        "line.width": show ? 4 : 0,
        "line.color": show ? "rgba(80,80,80,0.6)" : "rgba(80,80,80,0)",
      },
      [pathIdx]
    );
  }

  async function applyPlotView(retries = 20) {
    const iframe = $("#plot");
    const doc = iframe?.contentDocument;
    const win = iframe?.contentWindow;
    if (!doc || !win) return;

    prepareIframeDocument(doc);

    const gd = doc.querySelector(".plotly-graph-div");
    const Plotly = win.Plotly;
    if (!gd || !Plotly) {
      if (retries > 0) setTimeout(() => applyPlotView(retries - 1), 60);
      return;
    }

    const ready = await waitForPlotLayout(gd);
    if (!ready && retries > 0) {
      setTimeout(() => applyPlotView(retries - 1), 60);
      return;
    }

    await new Promise((resolve) => setTimeout(resolve, 150));

    try {
      applyPathVisibility(gd, Plotly);
      if (Plotly.Plots?.resize) Plotly.Plots.resize(gd);
    } catch (_) {
      if (retries > 0) setTimeout(() => applyPlotView(retries - 1), 60);
    }
  }

  function onConceptChange() {
    const exp = concept();

    if (isOrdering()) {
      const meta = ORDERING[exp];
      const man = orderingManifests[meta.slug] || { figures: [] };
      const layers =
        man.figures.length > 0
          ? [...new Set(man.figures.map((f) => f.layer))].sort((a, b) => a - b)
          : MONTHS_LAYERS;
      setOptions($('[data-field="layer"]'), layers, field("layer"));
      syncOrderingControls();
    } else {
      setOptions($('[data-field="layer"]'), MONTHS_LAYERS, field("layer") || "3");
    }

    refreshAll();
  }

  function refreshAll() {
    const iframe = $("#plot");
    if (iframe) {
      iframe.onload = () => applyPlotView();
      iframe.src = plotPath();
    }
    syncUrl();
  }

  Promise.all([
    fetch(ORDERING.colours_distilgpt2.manifestUrl).then((r) => r.json()).catch(() => ({ figures: [] })),
    fetch(ORDERING.nouns_distilgpt2.manifestUrl).then((r) => r.json()).catch(() => ({ figures: [] })),
    fetch(ORDERING.tools_distilgpt2.manifestUrl).then((r) => r.json()).catch(() => ({ figures: [] })),
    fetch(ORDERING.animals_distilgpt2.manifestUrl).then((r) => r.json()).catch(() => ({ figures: [] })),
    fetch(ORDERING.years_distilgpt2.manifestUrl).then((r) => r.json()).catch(() => ({ figures: [] })),
    fetch(ORDERING.disciplines_distilgpt2.manifestUrl).then((r) => r.json()).catch(() => ({ figures: [] })),
    fetch(ORDERING.whwords_distilgpt2.manifestUrl).then((r) => r.json()).catch(() => ({ figures: [] })),
    fetch(ORDERING.stopwords_distilgpt2.manifestUrl).then((r) => r.json()).catch(() => ({ figures: [] })),
    fetch(ORDERING.languages_distilgpt2.manifestUrl).then((r) => r.json()).catch(() => ({ figures: [] })),
  ]).then(([cMan, nMan, tMan, aMan, yMan, dMan, wMan, sMan, lMan]) => {
    orderingManifests.colours_distilgpt2 = cMan;
    orderingManifests.nouns_distilgpt2 = nMan;
    orderingManifests.tools_distilgpt2 = tMan;
    orderingManifests.animals_distilgpt2 = aMan;
    orderingManifests.years_distilgpt2 = yMan;
    orderingManifests.disciplines_distilgpt2 = dMan;
    orderingManifests.whwords_distilgpt2 = wMan;
    orderingManifests.stopwords_distilgpt2 = sMan;
    orderingManifests.languages_distilgpt2 = lMan;

    applyUrl();

    $("#concept")?.addEventListener("change", onConceptChange);
    $("#show-path")?.addEventListener("change", () => {
      applyPlotView();
      syncUrl();
    });
    document.querySelectorAll("[data-field]").forEach((el) => {
      el.addEventListener("change", () => {
        if (el.dataset.field === "concept") return;
        refreshAll();
      });
    });

    onConceptChange();
  });

  window.addEventListener("resize", () => {
    const iframe = $("#plot");
    const gd = iframe?.contentDocument?.querySelector(".plotly-graph-div");
    const Plotly = iframe?.contentWindow?.Plotly;
    if (gd && Plotly?.Plots?.resize) Plotly.Plots.resize(gd);
  });
})();

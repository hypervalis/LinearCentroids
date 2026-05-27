"""Unified command-line interface for LCE pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lce.config import load_config
from lce.pipeline.build_site import build_site
from lce.pipeline.environment import check_environment
from lce.pipeline.extract import extract_experiment
from lce.pipeline.generate_prompts import generate_prompts
from lce.pipeline.geometry import compute_geometry, run_synthetic
from lce.pipeline.month_plots import make_month_plots
from lce.pipeline.ordering_metrics import compute_ordering_metrics
from lce.pipeline.ordering_plots import make_ordering_plots
from lce.pipeline.paths import REPO_ROOT
from lce.pipeline.token_audit import audit_tokens
from lce.registry import concept_from_config


def _config_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def cmd_check_environment(_: argparse.Namespace) -> int:
    return check_environment()


def cmd_audit_tokens(args: argparse.Namespace) -> int:
    return audit_tokens(model_name=args.model)


def cmd_generate_prompts(args: argparse.Namespace) -> int:
    cfg = load_config(_config_path(args.config))
    path = generate_prompts(
        cfg,
        output=args.output,
        prompt_family=args.prompt_family,
    )
    print(f"Wrote {path.resolve()}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    cfg = load_config(_config_path(args.config))
    spec = concept_from_config(cfg)
    print(f"Config: {_config_path(args.config).resolve()}")
    print(f"Concept: {spec.name}")
    print(f"Output: {REPO_ROOT / cfg['output']['raw_dir']}")
    results = extract_experiment(cfg)
    raw_dir = REPO_ROOT / cfg["output"]["raw_dir"]
    for r in results:
        print(f"  layer {r.layer}: {r.per_template_path.resolve()}")
    print(f"  template variance: {(raw_dir / 'template_variance.json').resolve()}")
    return 0


def cmd_compute_geometry(args: argparse.Namespace) -> int:
    if args.synthetic:
        path = run_synthetic(dim=args.dim, pca_k=3)
        print(f"Wrote {path.resolve()}")
        return 0
    cfg = load_config(_config_path(args.config))
    path = compute_geometry(cfg)
    print(f"Wrote {path.resolve()}")
    return 0


def cmd_ordering_metrics(args: argparse.Namespace) -> int:
    cfg = load_config(_config_path(args.config))
    path = compute_ordering_metrics(
        cfg,
        raw_dir=args.raw_dir,
        output=args.output,
        space=args.space,
        layers_spec=args.layers,
        num_random=args.num_random,
        seed=args.seed,
        prompt_family=args.prompt_family,
    )
    print(f"Wrote {path.resolve()}")
    return 0


def cmd_make_plots(args: argparse.Namespace) -> int:
    cfg = load_config(_config_path(args.config))
    spec = concept_from_config(cfg)
    if spec.uses_month_geometry:
        make_month_plots(cfg, sync_docs=args.sync_docs)
    else:
        make_ordering_plots(cfg, sync_docs=args.sync_docs)
    return 0


def cmd_build_site(args: argparse.Namespace) -> int:
    cfg = load_config(_config_path(args.config))
    path = build_site(cfg, patch_config=not args.no_patch_config)
    print(f"Synced assets to {path.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lce",
        description="Linear Centroid Experiments — extraction, geometry, and site pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-environment", help="Verify dependencies and optional submodule.")

    p = sub.add_parser("audit-tokens", help="Audit month prompt final tokens.")
    p.add_argument("--model", default="distilgpt2")

    p = sub.add_parser("generate-prompts", help="Write prompt JSON/JSONL from config.")
    p.add_argument("--config", default="configs/months_distilgpt2.json")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--prompt-family", default=None)

    p = sub.add_parser("extract", help="Extract activations and MLP-local centroids.")
    p.add_argument("--config", default="configs/months_distilgpt2.json")

    p = sub.add_parser("compute-geometry", help="Months cyclic geometry on template means.")
    p.add_argument("--config", default="configs/months_distilgpt2.json")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--dim", type=int, default=32)

    p = sub.add_parser("ordering-metrics", help="Ordering topology metrics vs random baselines.")
    p.add_argument("--config", default="configs/nouns_distilgpt2.json")
    p.add_argument("--raw-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--space", default="both")
    p.add_argument("--layers", default="all")
    p.add_argument("--num-random", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt-family", default=None)

    p = sub.add_parser("make-plots", help="Generate Plotly PCA path plots.")
    p.add_argument("--config", default="configs/months_distilgpt2.json")
    p.add_argument("--sync-docs", action="store_true")

    p = sub.add_parser("build-site", help="Sync experiment assets into docs/ for GitHub Pages.")
    p.add_argument("--config", default="configs/months_distilgpt2.json")
    p.add_argument(
        "--no-patch-config",
        action="store_true",
        help="Do not rewrite window.EXPLORE_CONFIG in docs/index.html.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "check-environment": cmd_check_environment,
        "audit-tokens": cmd_audit_tokens,
        "generate-prompts": cmd_generate_prompts,
        "extract": cmd_extract,
        "compute-geometry": cmd_compute_geometry,
        "ordering-metrics": cmd_ordering_metrics,
        "make-plots": cmd_make_plots,
        "build-site": cmd_build_site,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

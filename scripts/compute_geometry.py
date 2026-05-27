#!/usr/bin/env python3
"""Thin wrapper: ``lce compute-geometry``."""

from lce.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["compute-geometry", *__import__("sys").argv[1:]]))

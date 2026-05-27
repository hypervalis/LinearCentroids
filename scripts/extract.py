#!/usr/bin/env python3
"""Thin wrapper: ``lce extract``."""

from lce.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["extract", *__import__("sys").argv[1:]]))

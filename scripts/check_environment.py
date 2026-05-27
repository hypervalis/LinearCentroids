#!/usr/bin/env python3
"""Thin wrapper: ``lce check-environment``."""

from lce.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["check-environment", *__import__("sys").argv[1:]]))

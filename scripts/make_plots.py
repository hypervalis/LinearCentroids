#!/usr/bin/env python3
"""Thin wrapper: ``lce make-plots``."""

from lce.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["make-plots", *__import__("sys").argv[1:]]))

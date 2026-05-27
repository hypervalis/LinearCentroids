#!/usr/bin/env python3
"""Thin wrapper: ``lce build-site``."""

from lce.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["build-site", *__import__("sys").argv[1:]]))

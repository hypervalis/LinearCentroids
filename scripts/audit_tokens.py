#!/usr/bin/env python3
"""Thin wrapper: ``lce audit-tokens``."""

from lce.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["audit-tokens", *__import__("sys").argv[1:]]))

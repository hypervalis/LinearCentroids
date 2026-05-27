#!/usr/bin/env python3
"""Thin wrapper: ``lce ordering-metrics``."""

from lce.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["ordering-metrics", *__import__("sys").argv[1:]]))

#!/usr/bin/env python3
"""Thin wrapper: ``lce generate-prompts``."""

from lce.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["generate-prompts", *__import__("sys").argv[1:]]))

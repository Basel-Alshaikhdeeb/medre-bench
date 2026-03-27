#!/usr/bin/env python3
"""Convenience script to run a single experiment."""

import sys

from medre_bench.cli import app

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "train"] + sys.argv[1:]
    app()

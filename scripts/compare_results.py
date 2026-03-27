#!/usr/bin/env python3
"""Convenience script to compare results."""

import sys

from medre_bench.cli import app

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "compare"] + sys.argv[1:]
    app()

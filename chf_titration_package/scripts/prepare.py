#!/usr/bin/env python
"""One-shot setup: generate synthetic patient-weeks and train the classifier bundle.

Idempotent — if the outputs already exist, it's a no-op. Pass --force to regenerate.

Usage:
    python scripts/prepare.py                      # defaults: 10,000 weeks, mimic mode if parquet present
    python scripts/prepare.py --n 5000             # faster setup with fewer weeks
    python scripts/prepare.py --force              # regenerate + retrain from scratch
"""
from chf_titration.cli import prepare_cli

if __name__ == "__main__":
    prepare_cli()

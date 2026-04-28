#!/usr/bin/env python
"""Generate synthetic patient-weeks for training.

Usage:
    python scripts/generate_data.py --n 10000 --out data/synthetic_patient_weeks.parquet
"""
from chf_titration.cli import generate_data_cli

if __name__ == "__main__":
    generate_data_cli()

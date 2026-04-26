#!/usr/bin/env python
"""Train the 11-flag LightGBM classifier.

Usage:
    python scripts/train.py --data data/synthetic_patient_weeks.parquet --out models/chf_classifier_lgbm.pkl
"""
from chf_titration.cli import train_cli

if __name__ == "__main__":
    train_cli()

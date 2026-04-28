#!/usr/bin/env python
"""Launch the Streamlit UI on localhost:8501.

Usage:
    python scripts/run_ui.py
"""
from chf_titration.cli import ui_cli

if __name__ == "__main__":
    ui_cli()

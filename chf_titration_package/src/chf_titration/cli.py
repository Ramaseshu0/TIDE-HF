"""Console-script entry points (invoked via pyproject.toml [project.scripts])."""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_DATA_PATH = "data/synthetic_patient_weeks.parquet"
DEFAULT_BUNDLE_PATH = "models/chf_classifier_lgbm.pkl"
DEFAULT_PREP_N = 10000


def generate_data_cli():
    ap = argparse.ArgumentParser(description="Generate synthetic patient-weeks for training.")
    ap.add_argument("--n", type=int, default=10000, help="number of patient-weeks")
    ap.add_argument("--out", default=DEFAULT_DATA_PATH)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["auto", "distribution", "mimic"], default="auto",
                    help="auto (default) uses mimic when the parquet is present, else distribution")
    ap.add_argument("--source-parquet", default=None,
                    help="override path to synthetic_CHF_visits_v2.parquet (default: data/)")
    args = ap.parse_args()
    from .data_gen import generate_and_save
    generate_and_save(
        n=args.n, out_path=args.out, seed=args.seed,
        mode=args.mode, source_parquet=args.source_parquet,
    )


def train_cli():
    ap = argparse.ArgumentParser(description="Train the 11-flag LightGBM classifier.")
    ap.add_argument("--data", default=DEFAULT_DATA_PATH)
    ap.add_argument("--out", default=DEFAULT_BUNDLE_PATH)
    ap.add_argument("--n-estimators", type=int, default=500)
    args = ap.parse_args()
    from .classifier import train_classifier
    train_classifier(args.data, args.out, n_estimators=args.n_estimators)


def ensure_bundle(
    data_path: str = DEFAULT_DATA_PATH,
    bundle_path: str = DEFAULT_BUNDLE_PATH,
    n: int = DEFAULT_PREP_N,
    seed: int = 42,
    mode: str = "auto",
    n_estimators: int = 500,
    source_parquet: str | None = None,
    force: bool = False,
) -> Path:
    """First-run setup: generate synthetic weeks (if missing) and train (if missing).

    Idempotent. Returns the path to the ready-to-load bundle. Intended to be called
    from run_ui.py so the first UI launch after `pip install -e .` works with no
    extra steps — subsequent launches skip this because both artefacts are cached.
    """
    from .data_gen import generate_and_save
    from .classifier import train_classifier

    bp = Path(bundle_path)
    dp = Path(data_path)

    def _usable(p: Path, min_bytes: int = 1024) -> bool:
        """A file is considered usable if it exists and is bigger than a smoke-test
        threshold. Protects against partial writes from an earlier crashed run
        (pyarrow leaves behind a 0-byte parquet if to_parquet fails mid-write)."""
        try: return p.exists() and p.stat().st_size >= min_bytes
        except OSError: return False

    if _usable(bp) and not force:
        print(f"[chf-prepare] bundle already present at {bp} — skipping setup.")
        return bp
    # bundle unusable but present → remove the stub so a failed train won't leave a mess either
    if bp.exists() and not _usable(bp):
        print(f"[chf-prepare] removing incomplete bundle at {bp}")
        try: bp.unlink()
        except OSError: pass

    if force or not _usable(dp):
        if dp.exists() and not _usable(dp):
            print(f"[chf-prepare] removing incomplete data file at {dp}")
            try: dp.unlink()
            except OSError: pass
        print(f"[chf-prepare] generating {n:,} synthetic patient-weeks (mode={mode}) → {dp}")
        generate_and_save(n=n, out_path=str(dp), seed=seed, mode=mode, source_parquet=source_parquet)
    else:
        print(f"[chf-prepare] reusing existing synthetic weeks at {dp}")

    print(f"[chf-prepare] training LightGBM classifier → {bp}")
    train_classifier(str(dp), str(bp), n_estimators=n_estimators)
    print(f"[chf-prepare] done.")
    return bp


def prepare_cli():
    ap = argparse.ArgumentParser(
        description="One-shot setup: generate synthetic weeks + train the LightGBM classifier bundle.",
    )
    ap.add_argument("--n", type=int, default=DEFAULT_PREP_N)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["auto", "distribution", "mimic"], default="auto")
    ap.add_argument("--data", default=DEFAULT_DATA_PATH)
    ap.add_argument("--out", default=DEFAULT_BUNDLE_PATH)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--source-parquet", default=None)
    ap.add_argument("--force", action="store_true", help="Regenerate data + retrain even if files exist.")
    args = ap.parse_args()
    ensure_bundle(
        data_path=args.data, bundle_path=args.out,
        n=args.n, seed=args.seed, mode=args.mode,
        n_estimators=args.n_estimators,
        source_parquet=args.source_parquet,
        force=args.force,
    )


def ui_cli():
    """Launch the Streamlit UI on localhost. Auto-prepares the classifier bundle on first run."""
    ensure_bundle()
    ui_path = Path(__file__).parent / "ui.py"
    print(f"[chf-ui] launching Streamlit at http://localhost:8501 ...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)])

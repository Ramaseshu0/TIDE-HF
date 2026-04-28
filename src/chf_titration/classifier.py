"""Classifier: training, loading, and prediction. Requires a trained bundle."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import LABEL_COLS, CLASSES_ALL, RHYTHMS, VITAL_KEYS, ECG_SCALAR_KEYS
from .featurize import featurize_week, build_feature_row
from .strategy import TARGETS


def load_bundle(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def predict_flags(patient: dict, tps: list[dict], bundle: dict,
                  threshold: float = 0.5) -> tuple[dict, dict]:
    """Classifier prediction. Requires a trained bundle.

    Raises ValueError if bundle is None (no silent fallback to pseudo-classifier).
    """
    if bundle is None:
        raise ValueError(
            "predict_flags requires a trained classifier bundle. "
            "Run `python scripts/generate_data.py` then `python scripts/train.py` to create one."
        )
    patient_with_targets = {**patient, "targets": TARGETS}
    x = build_feature_row(patient_with_targets, tps, feature_cols=bundle["feature_cols"])
    flags, probs = {}, {}
    for lab in bundle["label_cols"]:
        p = float(bundle["models"][lab].predict_proba(x)[0, 1])
        probs[lab] = p; flags[lab] = bool(p >= threshold)
    return flags, probs


def train_classifier(parquet_path: str | Path, out_path: str | Path,
                     n_estimators: int = 500, early_stopping_rounds: int = 30,
                     random_state: int = 42) -> dict:
    """Train 11 LightGBM boosters (one per flag) + save a bundle pickle."""
    import lightgbm as lgb
    from sklearn.model_selection import GroupShuffleSplit

    df = pd.read_parquet(parquet_path)
    print(f"[train] loaded {len(df):,} weeks × {len(df.columns)} cols")

    print("[train] featurizing...")
    rows = []
    for _, r in df.iterrows():
        feats = featurize_week(list(r["timepoints"]))
        feats["gender_M"] = 1 if str(r.get("gender")) == "M" else 0
        feats["anchor_age"] = float(r.get("anchor_age") or 65)
        for cls in CLASSES_ALL:
            feats[f"on_{cls}"] = int(bool(r.get(f"on_{cls}")))
            for p in ("dose_ratio", "target_dose_mg", "actual_dose_mg"):
                col = f"{p}_{cls}"
                v = r.get(col)
                feats[col] = float(v) if v is not None and not pd.isna(v) else 0.0
        rows.append(feats)
    X = pd.DataFrame(rows)
    feature_cols = list(X.columns)
    print(f"[train]   X shape {X.shape}")

    Y = df[LABEL_COLS].astype(int)

    groups = df["subject_id"].values if "subject_id" in df.columns else np.arange(len(df))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
    tv_idx, test_idx = next(gss1.split(X, Y, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=random_state)
    tr_rel, val_rel = next(gss2.split(X.iloc[tv_idx], Y.iloc[tv_idx], groups=groups[tv_idx]))
    train_idx = tv_idx[tr_rel]; val_idx = tv_idx[val_rel]

    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    Y_train, Y_val, Y_test = Y.iloc[train_idx], Y.iloc[val_idx], Y.iloc[test_idx]
    print(f"[train]   split train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    models = {}; metrics = {}
    for i, lab in enumerate(LABEL_COLS):
        m = lgb.LGBMClassifier(
            n_estimators=n_estimators, learning_rate=0.05, num_leaves=63,
            min_child_samples=30, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.85, reg_lambda=1.0, n_jobs=-1, verbose=-1,
            random_state=random_state,
        )
        m.fit(X_train, Y_train[lab],
              eval_set=[(X_val, Y_val[lab])], eval_metric="auc",
              callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)])
        models[lab] = m
        from sklearn.metrics import roc_auc_score
        p = m.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(Y_test[lab], p) if len(set(Y_test[lab])) > 1 else float("nan")
        metrics[lab] = auroc
        print(f"  [{i+1:2d}/11] {lab:<34} test AUROC={auroc:.3f}  best_iter={m.best_iteration_}")

    bundle = dict(
        models=models, feature_cols=feature_cols, label_cols=LABEL_COLS,
        rhythms=RHYTHMS, vital_keys=VITAL_KEYS, ecg_keys=ECG_SCALAR_KEYS,
        test_metrics=metrics, random_state=random_state,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[train] saved → {out_path}")
    return bundle

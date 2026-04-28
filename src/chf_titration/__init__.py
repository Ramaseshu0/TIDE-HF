"""CHF GDMT titration pipeline."""

__version__ = "0.1.0"

from .constants import CLASSES, RHYTHMS, CLASS_TO_REP_DRUG, LABEL_COLS
from .featurize import featurize_week, build_feature_row
from .synthesize import synthesize_week, PATIENTS
from .engine import TitrationEngine, derive_contraindications
from .strategy import apply_strategy, STRATEGIES, LADDERS, TARGETS
from .classifier import load_bundle, predict_flags, train_classifier

__all__ = [
    "CLASSES", "RHYTHMS", "CLASS_TO_REP_DRUG", "LABEL_COLS",
    "featurize_week", "build_feature_row",
    "synthesize_week", "PATIENTS",
    "TitrationEngine", "derive_contraindications",
    "apply_strategy", "STRATEGIES", "LADDERS", "TARGETS",
    "load_bundle", "predict_flags", "train_classifier",
]

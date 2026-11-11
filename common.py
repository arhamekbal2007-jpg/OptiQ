"""Shared helpers: paths, seeds, data loading, feature engineering, metrics."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "bank_campaign_participant_release_upgraded.xlsx"
OUT_DIR = ROOT  # the three result files live at the top level of the package

# ---- master seed list (every stochastic component uses one of these) ----
SEEDS = {
    "data_split_cv": 42,        # RepeatedStratifiedKFold in Stage 1
    "stage1_models": 42,        # PCA / logistic regression / kernel model
    "stage2_qaoa_init": 42,     # QAOA parameter initialisation + sampling
    "stage2_qaoa_sampling": 7,
    "stage3_shots": 2026,       # finite-shot sampling of the Stage 3 circuit
    "stage3_monte_carlo": 11,   # Monte Carlo validation
}

NUMERIC_RAW = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]


def load_data():
    tr = pd.read_excel(DATA_FILE, "Stage1_Train")
    te = pd.read_excel(DATA_FILE, "Stage1_Test")
    s2 = pd.read_excel(DATA_FILE, "Stage2_Data")
    cons = pd.read_excel(DATA_FILE, "Stage2_Constraints")
    tr["y"] = (tr["y"].astype(str).str.lower() == "yes").astype(int)
    return tr, te, s2, cons


def load_constraints(cons: pd.DataFrame) -> dict:
    c = {str(r["symbol"]).strip(): r["value"] for _, r in cons.iterrows()}
    return {"N": int(c["N"]), "K": int(c["K"]), "budget": float(c["C"]), "capacity": float(c["H"])}


class Engineer(BaseEstimator, TransformerMixin):
    """Stateless numeric transforms (no information is learned, so no leakage)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = pd.DataFrame(index=X.index)
        d["age"] = X["age"]
        d["balance_slog"] = np.sign(X["balance"]) * np.log1p(np.abs(X["balance"]))
        d["day"] = X["day"]
        d["campaign_log"] = np.log1p(X["campaign"])
        d["previous_log"] = np.log1p(X["previous"])
        d["was_contacted"] = (X["pdays"] >= 0).astype(float)
        d["pdays_log"] = np.log1p(X["pdays"].clip(lower=0))
        for c in CATEGORICAL:
            d[c] = X[c].astype(str)
        return d


NUM_ENGINEERED = ["age", "balance_slog", "day", "campaign_log", "previous_log", "was_contacted", "pdays_log"]


def make_preprocessor():
    """Engineer -> (scale numerics, one-hot categoricals). Always fitted on training folds only."""
    from sklearn.pipeline import Pipeline

    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_ENGINEERED),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    return Pipeline([("eng", Engineer()), ("ct", ct)])


# ---------------------------- metrics ----------------------------
def top_k_metrics(y, p, k):
    order = np.argsort(-p, kind="stable")[:k]
    hits = int(np.sum(np.asarray(y)[order]))
    return hits / k, hits / max(int(np.sum(y)), 1)


def calibration_slope_intercept(y, p):
    from sklearn.linear_model import LogisticRegression

    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def all_metrics(y, p, top_frac=0.10):
    y = np.asarray(y)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    k = max(1, int(round(top_frac * len(y))))
    prec, rec = top_k_metrics(y, p, k)
    slope, icpt = calibration_slope_intercept(y, p)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "precision_at_top10pct": prec,
        "recall_at_top10pct": rec,
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "calib_slope": slope,
        "calib_intercept": icpt,
    }


def environment_info() -> dict:
    import importlib

    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "openpyxl", "qiskit", "qiskit_aer"]:
        try:
            m = importlib.import_module(pkg)
            info[pkg] = getattr(m, "__version__", "unknown")
        except Exception:
            info[pkg] = "not installed"
    return info


def dump_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    Path(path).write_text(json.dumps(obj, indent=2, default=conv))

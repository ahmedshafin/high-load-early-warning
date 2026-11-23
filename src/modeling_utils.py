"""
Helpers for loading the model-ready tables in our experiments (Colab-friendly).

    from modeling_utils import load_split, scale_fit_on_train
    Xtr, ytr, Xva, yva, Xte, yte, feats = load_split("D1", thr=4, H=30)

Rules baked in:
  * rows of the idle router r7 are excluded by default (it never carries load - it only inflates
    true negatives / lowers the false-positive rate);
  * 'purge' rows (the 60 s band between splits) are never returned;
  * scaling is NOT stored in the files - use scale_fit_on_train() so the scaler sees TRAIN rows only.
"""
import numpy as np
import pandas as pd

from config import MODEL_DIR
from features import feature_groups


def load_model_ready(ds, exclude_idle_router=True):
    d = pd.read_parquet(MODEL_DIR / f"{ds}_model_ready.parquet")
    return d[d.host != "r7"] if exclude_idle_router else d


def feature_columns(df, groups=None):
    g = feature_groups(df)
    keys = groups or list(g)
    return [c for k in keys for c in g[k]]


def load_split(ds, thr=4, H=30, groups=None, exclude_idle_router=True):
    """Within-dataset chronological split -> (Xtr, ytr, Xva, yva, Xte, yte, feature_names)."""
    d = load_model_ready(ds, exclude_idle_router)
    feats, y = feature_columns(d, groups), f"y_ge{thr}_h{H}"
    out = []
    for s in ("train", "val", "test"):
        part = d[d.split == s]
        out += [part[feats], part[y]]
    return (*out, feats)


def load_full(ds, thr=4, H=30, groups=None, exclude_idle_router=True):
    """Whole dataset as (X, y) - use as the TEST set in cross-dataset experiments."""
    d = load_model_ready(ds, exclude_idle_router)
    d = d[d.split != "purge"]
    feats = feature_columns(d, groups)
    return d[feats], d[f"y_ge{thr}_h{H}"]


def scale_fit_on_train(Xtr, *others, log_cols=None):
    """log1p the heavy-tailed columns, then z-score using TRAIN statistics only (for LR / NN models)."""
    Xtr = Xtr.copy()
    log_cols = log_cols or [c for c in Xtr.columns if (Xtr[c] >= 0).all() and Xtr[c].max() > 1000]
    for X in (Xtr, *others):
        X.loc[:, log_cols] = np.log1p(X[log_cols])
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
    return [(X - mu) / sd for X in (Xtr, *others)]


def val_prediction_frame(df, target_col, horizon, probs, y_val, model_name, split="val"):
    """Row-level validation probabilities (same rows/order as the model's validation X, y).

    Saved by the model scripts so 13_fixed_budget_thresholds.py can choose thresholds
    without retraining anything.
    """
    part = df.loc[df["split"].eq(split)]
    target = pd.to_numeric(part[target_col], errors="coerce")
    keep = target.isin([0, 1])
    frame = part.loc[keep, ["dataset", "host", "ts_unix", "time", "label"]].reset_index(drop=True)
    y_val = np.asarray(y_val)
    if len(frame) != len(y_val) or not np.array_equal(target.loc[keep].astype("int8").to_numpy(), y_val):
        raise RuntimeError("Validation metadata does not align with the model's validation rows.")
    frame["horizon_seconds"] = horizon
    frame["target_column"] = target_col
    frame["actual_onset"] = y_val
    frame["predicted_probability"] = np.asarray(probs, dtype=float)
    frame["model"] = model_name
    return frame

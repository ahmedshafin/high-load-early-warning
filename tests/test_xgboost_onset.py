"""Light checks for 12_xgboost_onset.py (skipped when xgboost is not installed)."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("xgboost")

SRC = Path(__file__).resolve().parents[1] / "src"


def load():
    spec = importlib.util.spec_from_file_location("xgb_onset", SRC / "12_xgboost_onset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_uses_same_protocol_helpers_as_random_forest():
    x = load()
    assert x.TRAIN_DATASET == "D1"
    assert x.TEST_DATASETS == ("D1", "D2", "D3")
    assert x.XGB_PARAMS["early_stopping_rounds"] > 0
    assert x.XGB_PARAMS["random_state"] == 42


def test_label_derived_features_are_excluded():
    import pandas as pd
    x = load()
    cols = ["bw_max", "run_len_ge4", "steps_since_ge4", "ctx_net_n_ge4", "n_ge4_last30s", "bw_max_lag1"]
    df = pd.DataFrame({c: [0.0, 1.0] for c in cols})
    feats, _ = x.select_telemetry_features(df)
    assert "run_len_ge4" not in feats and "ctx_net_n_ge4" not in feats and "n_ge4_last30s" not in feats

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_SRC))

spec = importlib.util.spec_from_file_location(
    "rule_based_onset_baseline", PROJECT_SRC / "08_rule_based_onset_baseline.py"
)
rule_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rule_module
spec.loader.exec_module(rule_module)


def test_predict_rule_uses_bandwidth_or_rise_only():
    frame = pd.DataFrame({
        "bw_max": [10.0, 50.0, 20.0, 45.0],
        "bw_max_diff1": [1.0, 1.0, 12.0, 5.0],
    })
    got = rule_module.predict_rule(frame, bw_threshold=40.0, rise_threshold=10.0)
    assert np.array_equal(got, np.array([0, 1, 1, 1], dtype=np.int8))


def test_classification_metrics_confusion_counts():
    metrics = rule_module.classification_metrics(
        np.array([0, 0, 1, 1], dtype=np.int8),
        np.array([0, 1, 1, 0], dtype=np.int8),
    )
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["f1"] == 0.5
    assert metrics["false_positive_rate"] == 0.5


def test_choose_rule_selects_using_validation_rows():
    train = pd.DataFrame({
        "bw_max": [10, 20, 30, 40, 50, 60, 70, 80],
        "bw_max_diff1": [0, 2, 4, 6, 8, 10, 12, 14],
        "y_onset_h10": [0, 0, 0, 0, 0, 1, 1, 1],
    })
    validation = pd.DataFrame({
        "bw_max": [15, 25, 45, 55, 75, 85],
        "bw_max_diff1": [1, 3, 5, 9, 11, 15],
        "y_onset_h10": [0, 0, 0, 1, 1, 1],
    })
    selected, val_metrics = rule_module.choose_rule(train, validation, "y_onset_h10")
    assert selected["bw_threshold"] in rule_module.BW_THRESHOLD_CANDIDATES
    assert val_metrics["n_eligible"] == len(validation)
    assert 0.0 <= val_metrics["f1"] <= 1.0

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_SRC))

spec = importlib.util.spec_from_file_location(
    "random_forest_onset", PROJECT_SRC / "10_random_forest_onset.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_feature_selection_excludes_label_state_and_targets():
    frame = pd.DataFrame(columns=[
        "bw_max", "bw_max_diff1", "run_len_ge4", "n_ge4_last30s",
        "ctx_net_n_ge4", "label", "cur_ge4", "y_onset_h10",
    ])
    original = module.feature_groups
    module.feature_groups = lambda _: {
        "telemetry": ["bw_max", "bw_max_diff1"],
        "label_state": ["run_len_ge4", "n_ge4_last30s", "ctx_net_n_ge4", "cur_ge4", "label", "y_onset_h10"],
    }
    try:
        selected, audit = module.select_telemetry_features(frame)
    finally:
        module.feature_groups = original

    assert selected == ["bw_max", "bw_max_diff1"]
    assert len(audit) == 8


def test_partition_xy_excludes_minus_one_and_purge_rows():
    frame = pd.DataFrame({
        "split": ["train", "train", "train", "purge", "val"],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y_onset_h10": [0, 1, -1, 0, 1],
    })
    X, y = module.partition_xy(frame, "y_onset_h10", ["x1"], split="train")
    assert X.x1.tolist() == [1.0, 2.0]
    assert y.tolist() == [0, 1]

    X_all, y_all = module.partition_xy(frame, "y_onset_h10", ["x1"])
    assert X_all.x1.tolist() == [1.0, 2.0, 5.0]
    assert y_all.tolist() == [0, 1, 1]


def test_auc_metrics_use_probabilities_and_return_confusion_counts():
    metrics = module.binary_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 0]),
        np.array([0.1, 0.8, 0.9, 0.2]),
    )
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["f1"] == 0.5
    assert metrics["roc_auc"] == 0.75


def test_threshold_is_selected_from_validation_probabilities():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    p = np.array([0.05, 0.10, 0.25, 0.65, 0.80, 0.95])
    threshold, metrics = module.choose_probability_threshold(y, p)
    assert 0.0 <= threshold <= 1.0
    assert metrics["f1"] == 1.0

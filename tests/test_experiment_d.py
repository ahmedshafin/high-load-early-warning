"""Tests for the pooled validation scoring and the regime comparison in 15_experiment_d_multi_scenario.py."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

SRC = Path(__file__).resolve().parents[1] / "src"
_spec = importlib.util.spec_from_file_location("experiment_d", SRC / "15_experiment_d_multi_scenario.py")
ed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ed)

H, STARTS = 30, [120, 500, 730, 900, 1135]


def make_scorer(dataset):
    ts = np.arange(0, 1310, 10)
    rows = pd.DataFrame({"dataset": dataset, "host": "r1", "ts_unix": ts})
    rows["actual_onset"] = [int(any(t < s <= t + H for s in STARTS)) for t in ts]
    events = pd.DataFrame({"dataset": dataset, "host": "r1", "start_ts_unix": STARTS,
                           "event_number_within_host": range(1, 6)})
    return ed.fb.Scorer(rows, events, dataset, H, hours=1.0)


def probs_for(scorer, warn_ts):
    return np.isin(scorer.p["ts_unix"].to_numpy(), warn_ts).astype(float)


def test_pooled_score_sums_counts_and_exposure():
    a, b = make_scorer("D1"), make_scorer("D2")
    warn = [100, 110, 300, 700, 900, 1100]                  # 2 of 5 events detected, 3 false alarms, per dataset
    single = a.score(probs_for(a, warn) >= 0.5)
    pooled = ed.pooled_score([a, b], [probs_for(a, warn), probs_for(b, warn)], 0.5)
    assert pooled["eligible_events"] == 10 and pooled["detected_events"] == 4
    assert pooled["false_alarms"] == 6 and pooled["monitoring_router_hours"] == 2.0
    assert pooled["false_alarms_per_router_hour"] == single["false_alarms_per_router_hour"] == 3.0
    assert pooled["alert_active_time_pct"] == pytest.approx(single["alert_active_time_pct"], abs=1e-2)


def test_pooled_score_weights_scenarios_by_exposure():
    a, b = make_scorer("D1"), make_scorer("D2")
    warn = [100, 110, 300, 700, 900, 1100]
    pooled = ed.pooled_score([a, b], [probs_for(a, warn), np.zeros(len(b.p))], 0.5)   # D2: never warns
    assert pooled["detected_events"] == 2 and pooled["eligible_events"] == 10
    assert pooled["event_detection_rate_pct"] == 20.0
    assert pooled["false_alarms_per_router_hour"] == 1.5     # 3 false alarms over 2 router-hours


def test_comparison_is_multi_minus_base():
    base = dict(config="D_all", horizon_seconds=20, budget_fa_per_router_hour=3.0, test_set="D3", n_features=56,
                alert_precision_pct=70.0, fa_realized_over_budget=1.0)
    results = pd.DataFrame([
        {**base, "regime": "train_D1", "event_detection_rate_pct": 70.0, "false_alarms_per_router_hour": 4.0},
        {**base, "regime": "train_D1_D2", "event_detection_rate_pct": 78.0, "false_alarms_per_router_hour": 3.0},
    ])
    pr = pd.DataFrame([
        {"regime": "train_D1", "config": "D_all", "horizon_seconds": 20, "test_set": "D3", "pr_auc": 0.60},
        {"regime": "train_D1_D2", "config": "D_all", "horizon_seconds": 20, "test_set": "D3", "pr_auc": 0.65},
    ])
    out = ed.build_comparison(results, pr).iloc[0]
    assert out["delta_event_detection_rate_pct"] == 8.0
    assert out["delta_false_alarms_per_router_hour"] == -1.0
    assert out["delta_pr_auc"] == pytest.approx(0.05)

"""Tests for 16_bootstrap_confidence_intervals.py: weighted AP, per-block statistics, resampling identities."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

SRC = Path(__file__).resolve().parents[1] / "src"
_spec = importlib.util.spec_from_file_location("bootstrap_ci", SRC / "16_bootstrap_confidence_intervals.py")
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)

H, STARTS = 30, [120, 500, 730, 900, 1135]
WARN_TS = [100, 110, 300, 700, 900, 1100]


def make_scorer():
    ts = np.arange(0, 1310, 10)
    rows = pd.DataFrame({"dataset": "T", "host": "r1", "ts_unix": ts})
    rows["actual_onset"] = [int(any(t < s <= t + H for s in STARTS)) for t in ts]
    events = pd.DataFrame({"dataset": "T", "host": "r1", "start_ts_unix": STARTS,
                           "event_number_within_host": range(1, 6)})
    return bs.fb.Scorer(rows, events, "T", H, hours=1300 / 3600)


def test_weighted_ap_matches_sklearn_including_ties_and_weights():
    rng = np.random.default_rng(1)
    n = 4000
    y = (rng.random(n) < 0.15).astype(int)
    prob = np.round(np.clip(0.3 * y + rng.random(n) * 0.7, 0, 1), 2)        # many exact ties
    blocks = np.arange(n) // 40
    ap = bs.WeightedAP(y, prob, blocks)
    ones = np.ones(blocks.max() + 1)
    assert ap(ones) == pytest.approx(average_precision_score(y, prob), abs=1e-9)
    counts = rng.integers(0, 4, size=blocks.max() + 1).astype(float)          # a bootstrap-style draw
    expected = average_precision_score(y, prob, sample_weight=counts[blocks])
    assert ap(counts) == pytest.approx(expected, abs=1e-9)


def test_detail_totals_equal_score_totals():
    scorer = make_scorer()
    warn = np.isin(scorer.p["ts_unix"].to_numpy(), WARN_TS)
    alerts, events = scorer.detail(warn)
    r = scorer.score(warn)
    assert len(events) == r["eligible_events"] and int(events["detected"].sum()) == r["detected_events"]
    assert len(alerts) == r["alert_episodes"] and int(alerts["matched"].sum()) == r["matched_alerts"]
    assert float((alerts["end_ts"] - alerts["start_ts"]).sum()) == pytest.approx(
        r["alert_active_time_pct"] / 100 * scorer.hours * 3600, abs=0.1)


def test_block_sums_reproduce_totals_for_any_block_size():
    scorer = make_scorer()
    warn = np.isin(scorer.p["ts_unix"].to_numpy(), WARN_TS)
    r = scorer.score(warn)
    t0 = int(scorer.ts.min())
    for block_seconds in (60, 200, 10_000):
        n_blocks = int((scorer.ts.max() - t0) // block_seconds) + 1
        s = bs.block_stats(scorer, warn, t0, block_seconds, n_blocks)
        assert s["detected"].sum() == r["detected_events"] and s["eligible"].sum() == r["eligible_events"]
        assert s["false_alarms"].sum() == r["false_alarms"] and s["alerts"].sum() == r["alert_episodes"]
        assert s["hours"].sum() == pytest.approx(scorer.hours)
        m = bs.metrics_from_sums({k: v.sum() for k, v in s.items()})
        assert m["event_detection_rate_pct"] == pytest.approx(r["event_detection_rate_pct"], abs=1e-3)
        assert m["false_alarms_per_router_hour"] == pytest.approx(r["false_alarms_per_router_hour"], abs=1e-3)


def test_resampling_all_blocks_once_reproduces_the_point_estimate():
    scorer = make_scorer()
    warn = np.isin(scorer.p["ts_unix"].to_numpy(), WARN_TS)
    t0 = int(scorer.ts.min())
    n_blocks = int((scorer.ts.max() - t0) // 120) + 1
    s = bs.block_stats(scorer, warn, t0, 120, n_blocks)
    once = np.ones(n_blocks)
    assert (once @ s["detected"]) == s["detected"].sum()
    # a draw that doubles every block leaves all ratios unchanged
    twice = 2 * once
    m1 = bs.metrics_from_sums({k: once @ v for k, v in s.items()})
    m2 = bs.metrics_from_sums({k: twice @ v for k, v in s.items()})
    for key in m1:
        assert m1[key] == pytest.approx(m2[key])

"""Tests for 13_fixed_budget_thresholds.py: operating-point selection and event-level scoring."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
_spec = importlib.util.spec_from_file_location("fixed_budget", SRC / "13_fixed_budget_thresholds.py")
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)

CURVE = pd.DataFrame({
    "threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    "event_detection_rate_pct": [90.0, 80.0, 80.0, 50.0, 0.0],
    "false_alarms_per_router_hour": [6.0, 3.0, 2.5, 1.0, 0.0],
    "alert_active_time_pct": [8.0, 6.0, 5.0, 3.0, 0.0],
})
CURVE["tiebreak"] = CURVE["threshold"]


def test_picks_highest_detection_within_budget_then_fewest_false_alarms():
    pos, feasible = fb.choose_operating_point(CURVE, 3.0)
    assert feasible and CURVE.loc[pos, "threshold"] == 0.7          # 80 % detection, 2.5 < 3.0 alarms
    pos, feasible = fb.choose_operating_point(CURVE, 1.0)
    assert feasible and CURVE.loc[pos, "threshold"] == 0.8          # the budget is inclusive (<=)
    pos, feasible = fb.choose_operating_point(CURVE, 10.0)
    assert CURVE.loc[pos, "threshold"] == 0.5


def test_infeasible_budget_falls_back_to_fewest_false_alarms():
    curve = CURVE.assign(false_alarms_per_router_hour=[9.0, 8.0, 7.5, 7.0, 6.0])
    pos, feasible = fb.choose_operating_point(curve, 1.0)
    assert not feasible and curve.loc[pos, "threshold"] == 0.9


def test_tie_prefers_higher_threshold():
    curve = pd.DataFrame({"threshold": [0.4, 0.6], "event_detection_rate_pct": [70.0, 70.0],
                          "false_alarms_per_router_hour": [2.0, 2.0], "alert_active_time_pct": [4.0, 4.0]})
    curve["tiebreak"] = curve["threshold"]
    pos, _ = fb.choose_operating_point(curve, 5.0)
    assert curve.loc[pos, "threshold"] == 0.6


def test_candidate_thresholds_cover_never_warn_and_are_sorted_unique():
    thr = fb.candidate_thresholds(np.random.default_rng(0).random(5000))
    assert thr[-1] > 1.0 and np.all(np.diff(thr) > 0)


def test_scorer_reproduces_hand_checked_timeline():
    H, starts = 30, [120, 500, 730, 900, 1135]
    ts = np.arange(0, 1310, 10)
    rows = pd.DataFrame({"dataset": "T", "host": "r1", "ts_unix": ts})
    rows["actual_onset"] = [int(any(t < s <= t + H for s in starts)) for t in ts]
    events = pd.DataFrame({"dataset": "T", "host": "r1", "start_ts_unix": starts,
                           "event_number_within_host": range(1, 6)})
    scorer = fb.Scorer(rows, events, "T", H, hours=1.0)
    warn = np.isin(scorer.p["ts_unix"].to_numpy(), [100, 110, 300, 700, 900, 1100])
    r = scorer.score(warn)
    # same timeline as test_event_level_evaluation: 2 of 5 events detected (leads 20 s and 30 s),
    # alerts = 5 (2 matched, 3 false alarms)
    assert (r["eligible_events"], r["detected_events"]) == (5, 2)
    assert (r["alert_episodes"], r["matched_alerts"], r["false_alarms"]) == (5, 2, 3)
    assert r["false_alarms_per_router_hour"] == 3.0
    assert r["median_lead_time_seconds"] == 25.0 and r["mean_lead_time_seconds"] == 25.0
    # time under alert: 20 + 30 + 30 + 30 + 30 = 140 s of 3600 s; ideal alarm = 5 events x 30 s = 150 s
    assert r["alert_active_time_pct"] == round(140 / 3600 * 100, 3)
    assert r["oracle_active_pct"] == round(150 / 3600 * 100, 3)
    # never warning detects nothing and raises no alarms
    none = scorer.score(np.zeros(len(warn), dtype=bool))
    assert none["detected_events"] == 0 and none["alert_episodes"] == 0


def test_always_on_alarm_is_excluded_by_the_alert_time_cap():
    # the loophole: warning constantly => ~100 % detection, almost no "false alarms" (one endless alert)
    curve = pd.DataFrame({
        "threshold": [0.05, 0.6, 0.8],
        "event_detection_rate_pct": [100.0, 82.0, 60.0],
        "false_alarms_per_router_hour": [0.1, 2.5, 0.9],
        "alert_active_time_pct": [99.0, 9.0, 5.0],
    })
    curve["tiebreak"] = curve["threshold"]
    pos, _ = fb.choose_operating_point(curve, 3.0)                       # no cap -> the degenerate point
    assert curve.loc[pos, "threshold"] == 0.05
    pos, feasible = fb.choose_operating_point(curve, 3.0, max_active_pct=10.0)
    assert feasible and curve.loc[pos, "threshold"] == 0.6
    # tight budget + cap: only 0.8 qualifies on false alarms
    pos, feasible = fb.choose_operating_point(curve, 1.0, max_active_pct=10.0)
    assert feasible and curve.loc[pos, "threshold"] == 0.8


def test_active_cap_is_a_multiple_of_the_ideal_alarm():
    curve = pd.DataFrame({"oracle_active_pct": [2.5, 2.5]})
    assert fb.active_cap(curve, 2.0) == 5.0
    assert fb.active_cap(curve, 0) is None


def test_always_warning_scores_perfectly_on_false_alarms_but_saturates_alert_time():
    H, starts = 30, [120, 500, 730, 900, 1135]
    ts = np.arange(0, 1310, 10)
    rows = pd.DataFrame({"dataset": "T", "host": "r1", "ts_unix": ts})
    rows["actual_onset"] = [int(any(t < s <= t + H for s in starts)) for t in ts]
    events = pd.DataFrame({"dataset": "T", "host": "r1", "start_ts_unix": starts,
                           "event_number_within_host": range(1, 6)})
    scorer = fb.Scorer(rows, events, "T", H, hours=1300 / 3600)
    r = scorer.score(np.ones(len(ts), dtype=bool))
    assert r["detected_events"] == 5 and r["false_alarms"] <= 1       # looks perfect ...
    assert r["alert_active_time_pct"] > 95                            # ... but the alarm is on ~all the time

"""Hand-checkable test of the alert state machine in 11_event_level_evaluation.py."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
_spec = importlib.util.spec_from_file_location("event_level", SRC / "11_event_level_evaluation.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

H = 30


def run(warning_ts, event_ts):
    warnings = pd.DataFrame({"ts_unix": warning_ts, "predicted_warning": 1})
    events = pd.DataFrame({"start_ts_unix": event_ts,
                           "event_number_within_host": range(1, len(event_ts) + 1)})
    alerts, matches = ev.evaluate_host(warnings, events, "T", "r1", H)
    return pd.DataFrame(alerts), matches


def test_alert_matching_false_alarms_and_lead_times():
    # event 1 @120: warned @100 and renewed @110          -> detected, lead 20 s, ONE alert
    # warning @300, nothing follows (expires @330)         -> false alarm
    # event 2 @500: no warning                             -> missed
    # event 3 @730: warning @700 (event exactly at expiry) -> detected, lead 30 s
    # event 4 @900 with a warning at the SAME timestamp    -> missed (not an advance warning);
    #                                                         that warning then expires -> false alarm
    # event 5 @1135: warning @1100 expired @1130           -> missed; alert is a false alarm
    alerts, m = run([100, 110, 300, 700, 900, 1100], [120, 500, 730, 900, 1135])

    assert len(alerts) == 5
    assert (alerts["alert_status"] == "matched").sum() == 2
    assert (alerts["alert_status"] == "false_alarm").sum() == 3
    assert alerts.loc[alerts["alert_status"] == "matched", "n_positive_predictions"].tolist() == [2, 1]

    assert [m[i]["event_detected"] for i in range(1, 6)] == [1, 0, 1, 0, 0]
    assert m[1]["lead_time_seconds"] == 20
    assert m[3]["lead_time_seconds"] == 30
    assert np.isnan(m[2]["lead_time_seconds"])


def test_no_warnings_means_all_events_missed_and_no_alerts():
    alerts, m = run([], [100, 200])
    assert len(alerts) == 0
    assert [m[1]["event_detected"], m[2]["event_detected"]] == [0, 0]


def test_each_alert_matches_at_most_one_event():
    # one warning @100 covers events @110 and @120, but an alert may be matched only once
    alerts, m = run([100], [110, 120])
    assert (alerts["alert_status"] == "matched").sum() == 1
    assert [m[1]["event_detected"], m[2]["event_detected"]] == [1, 0]

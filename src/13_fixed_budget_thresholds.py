"""Fixed false-alarm-budget operating points (event level), for every model.

Why this script exists
----------------------
Scripts 08-10 and 12 pick each model's threshold by maximizing *sample-level* F1 on D1
validation, so every model ends up at a different point on its own detection / false-alarm
curve and the event-level numbers are not directly comparable. Here every model is held to
the SAME budget instead:

    choose, on D1 VALIDATION only, the operating point that maximizes event detection
    subject to  false alarms per router-hour <= budget
    AND         time under alert <= max_active_ratio x (time an ideal alarm would need),

then apply that frozen operating point to the D1 test split, D2 and D3 and report what
detection and false-alarm rate it actually delivers there. The gap between the budget and the
realized false-alarm rate on D2/D3 measures how well each model's operating point transfers
across scenarios (calibration drift).

* ML models (Logistic Regression, Random Forest, XGBoost): the operating point is a
  probability threshold, applied to the saved D1-validation / test probabilities. Nothing is
  retrained. Requires the *_onset_val_predictions.csv files written by scripts 09, 10 and 12.
* Rule-based baseline: the operating point is a (bandwidth threshold, rise threshold) pair
  from the same candidate grid script 08 uses, chosen under the same budget.

Why the second constraint? Under script 11's alert policy an alert is renewed by every further
warning, so a model that warns on every sample holds ONE never-ending alert that each event
matches: ~100 % detection with almost no "false alarms". The false-alarm count alone can
therefore be gamed (this shows up at H = 60 s). Limiting the share of router-time spent under
an active alert closes that loophole. The "ideal alarm" fires exactly H seconds before each
event, i.e. it is active for eligible_events x H seconds; the default cap is 2x that. Use
--max-active-ratio 0 to switch the cap off.

Event-level scoring (alerts, matching, lead time, eligibility) is imported from
11_event_level_evaluation.py, so these numbers are directly comparable with its output. For
verification the script also re-scores every model at its original F1-optimal operating point
and checks that the result equals 11's summary exactly.

Run from the repository root, after scripts 08-12:
    python src/13_fixed_budget_thresholds.py
    python src/13_fixed_budget_thresholds.py --budgets 1 2 3 5 --max-active-ratio 2

Outputs (reports/tables/)
-------------------------
fixed_budget_curves_validation.csv        detection vs false-alarm curve on D1 validation
fixed_budget_selected_operating_points.csv chosen operating point per model / horizon / budget
fixed_budget_event_level_summary.csv       test results (D1 test, D2, D3) per operating point
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import HORIZONS_SEC, MODEL_DIR, TAB_DIR  # noqa: E402
from modeling_utils import load_model_ready  # noqa: E402


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load("event_level", "11_event_level_evaluation.py")
rule = _load("rule_based", "08_rule_based_onset_baseline.py")

DATASETS = ev.DATASETS
DEFAULT_BUDGETS = [1.0, 2.0, 3.0, 5.0]          # false alarms per router-hour
ML_MODELS = [
    ("logistic_regression", "LogisticRegression"),
    ("random_forest", "RandomForest"),
    ("xgboost", "XGBoost"),
]
RULE_NAME = "RuleBased"


# ----------------------------------------------------------------------------- data
def load_events() -> pd.DataFrame:
    events = pd.read_csv(TAB_DIR / "label4_event_audit.csv")
    events["host"] = events["host"].astype(str)
    events["start_ts_unix"] = pd.to_numeric(events["start_ts_unix"], errors="raise").astype("int64")
    keep = pd.to_numeric(events["onset_confirmed_from_observed_transition"], errors="coerce").eq(1)
    return events.loc[keep].copy()


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["host"] = df["host"].astype(str)
    df["ts_unix"] = pd.to_numeric(df["ts_unix"], errors="raise").astype("int64")
    df["actual_onset"] = pd.to_numeric(df["actual_onset"], errors="raise").astype("int8")
    return df


def router_hours(dataset: str, split: str | None) -> float:
    """Continuously observed router-hours (10 s steps only), idle router excluded.

    split=None -> all non-purge rows (same definition script 11 uses for D2/D3).
    """
    df = pd.read_parquet(MODEL_DIR / f"{dataset}_model_ready.parquet", columns=["host", "ts_unix", "split"])
    df["host"] = df["host"].astype(str)
    df = df[df["host"] != ev.IDLE_ROUTER]
    df = df[df["split"] == split] if split else df[df["split"] != "purge"]
    total = 0
    for _, group in df.groupby("host"):
        times = np.sort(group["ts_unix"].unique().astype(np.int64))
        gaps = np.diff(times)
        total += int(gaps[gaps == ev.SAMPLING_SEC].sum())
    return total / 3600.0


# --------------------------------------------------------------------------- scoring
class Scorer:
    """Event-level scoring of arbitrary warning decisions on a fixed set of prediction rows.

    Event eligibility is independent of the threshold, so it is computed once; each call to
    score() only re-runs the (cheap) alert state machine from script 11.
    """

    def __init__(self, rows: pd.DataFrame, events: pd.DataFrame, dataset: str, horizon: int, hours: float):
        self.p = rows.sort_values(["host", "ts_unix"]).reset_index(drop=True)
        self.dataset, self.horizon, self.hours = dataset, horizon, hours
        event_eval = ev.determine_event_eligibility(self.p, events[events["dataset"] == dataset].copy(), horizon)
        self.eligible = event_eval[event_eval["eligible_for_evaluation"]]
        hosts = self.p["host"].to_numpy()
        self.ts = self.p["ts_unix"].to_numpy(dtype=np.int64)
        self.hosts = sorted(set(self.p["host"]) | set(self.eligible["host"]))
        self.row_index = {h: np.flatnonzero(hosts == h) for h in self.hosts}
        self.host_events = {h: self.eligible[self.eligible["host"] == h] for h in self.hosts}

    def score(self, warn: np.ndarray) -> dict:
        warn = np.asarray(warn, dtype=bool)
        detected = alerts_n = matched = false_alarms = 0
        active_seconds = 0.0
        leads: list[float] = []
        for host in self.hosts:
            idx = self.row_index[host]
            chosen = idx[warn[idx]]
            warning_rows = pd.DataFrame({"ts_unix": self.ts[chosen], "predicted_warning": 1})
            alert_records, matches = ev.evaluate_host(
                warning_rows, self.host_events[host], self.dataset, host, self.horizon)
            alerts_n += len(alert_records)
            for alert in alert_records:
                active_seconds += float(alert["alert_end_ts"] - alert["alert_start_ts"])
                if alert["alert_status"] == "matched":
                    matched += 1
                else:
                    false_alarms += 1
            for result in matches.values():
                if result["event_detected"] == 1:
                    detected += 1
                    leads.append(float(result["lead_time_seconds"]))
        eligible = len(self.eligible)
        exposure = self.hours * 3600.0
        return {
            "eligible_events": eligible,
            "detected_events": detected,
            "event_detection_rate_pct": round(detected / eligible * 100, 3) if eligible else np.nan,
            "alert_episodes": alerts_n,
            "matched_alerts": matched,
            "false_alarms": false_alarms,
            "alert_precision_pct": round(matched / alerts_n * 100, 3) if alerts_n else np.nan,
            "monitoring_router_hours": round(self.hours, 3),
            "false_alarms_per_router_hour": round(false_alarms / self.hours, 4) if self.hours > 0 else np.nan,
            "median_lead_time_seconds": round(float(np.median(leads)), 3) if leads else np.nan,
            "mean_lead_time_seconds": round(float(np.mean(leads)), 3) if leads else np.nan,
            "alert_active_time_pct": round(active_seconds / exposure * 100, 3) if exposure > 0 else np.nan,
            "oracle_active_pct": round(eligible * self.horizon / exposure * 100, 3) if exposure > 0 else np.nan,
        }


def _scorer_detail(self, warn: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-alert and per-event records (same state machine as score()); used for block bootstraps.

    alerts: host, start_ts, end_ts, matched (bool)      events: host, ts, detected (bool)
    """
    warn = np.asarray(warn, dtype=bool)
    alert_rows, event_rows = [], []
    for host in self.hosts:
        idx = self.row_index[host]
        chosen = idx[warn[idx]]
        warning_rows = pd.DataFrame({"ts_unix": self.ts[chosen], "predicted_warning": 1})
        host_events = self.host_events[host]
        alert_records, matches = ev.evaluate_host(
            warning_rows, host_events, self.dataset, host, self.horizon)
        for alert in alert_records:
            alert_rows.append((host, int(alert["alert_start_ts"]), int(alert["alert_end_ts"]),
                               alert["alert_status"] == "matched"))
        event_ts = dict(zip(host_events["event_number_within_host"], host_events["start_ts_unix"]))
        for number, result in matches.items():
            event_rows.append((host, int(event_ts[number]), result["event_detected"] == 1))
    alerts = pd.DataFrame(alert_rows, columns=["host", "start_ts", "end_ts", "matched"])
    events = pd.DataFrame(event_rows, columns=["host", "ts", "detected"])
    return alerts, events


Scorer.detail = _scorer_detail


def choose_operating_point(curve: pd.DataFrame, budget: float,
                          max_active_pct: float | None = None) -> tuple[int, bool]:
    """Index (position) of the best operating point under the budget, and feasibility flag.

    Feasible = false_alarms_per_router_hour <= budget AND (if max_active_pct is given)
    alert_active_time_pct <= max_active_pct.
    Best = highest event detection among feasible points; ties -> fewer false alarms, then higher
    `tiebreak` (higher threshold = fewer alerts).
    If nothing is feasible, return (feasible=False) the point with the fewest false alarms among
    points that respect the alert-time cap (or, failing that, the point with least alert time).
    """
    fa = curve["false_alarms_per_router_hour"].to_numpy(dtype=float)
    active = curve["alert_active_time_pct"].to_numpy(dtype=float)
    within_cap = np.ones(len(curve), dtype=bool) if max_active_pct is None else active <= max_active_pct
    ok = np.flatnonzero((fa <= budget) & within_cap)
    if len(ok) == 0:
        pool = np.flatnonzero(within_cap)
        if len(pool) == 0:
            return int(np.argmin(active)), False
        return int(pool[np.argmin(fa[pool])]), False
    sub = curve.iloc[ok].sort_values(
        ["event_detection_rate_pct", "false_alarms_per_router_hour", "tiebreak"],
        ascending=[False, True, False])
    return int(np.flatnonzero(curve.index == sub.index[0])[0]), True


def active_cap(curve: pd.DataFrame, ratio: float) -> float | None:
    """Alert-time cap (% of router-time) = ratio x what an ideal alarm needs on validation."""
    if ratio is None or ratio <= 0:
        return None
    return round(float(ratio) * float(curve["oracle_active_pct"].iloc[0]), 3)


def candidate_thresholds(prob: np.ndarray) -> np.ndarray:
    """Candidate probability thresholds: a regular grid plus quantiles that are densest in the
    upper tail (budgets of a few false alarms per hour live at the top ~1-5% of scores)."""
    tail = 1.0 - np.geomspace(0.5, 0.0005, 160)            # quantile levels 0.50 ... 0.9995
    upper = np.quantile(prob, tail)
    grid = np.linspace(0.05, 0.99, 95)
    return np.unique(np.round(np.concatenate([upper, grid, [1.000001]]), 6))   # last = never warn


# ------------------------------------------------------------------------- ML models
def run_ml_model(prefix, display, events, hours_val, hours_test, budgets, active_ratio):
    val_path = TAB_DIR / f"{prefix}_onset_val_predictions.csv"
    test_path = TAB_DIR / f"{prefix}_onset_predictions.csv"
    for path in (val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found. Rerun the model scripts (09, 10, 12).")
    val_all, test_all = read_predictions(val_path), read_predictions(test_path)

    curves, selected, summary = [], [], []
    for horizon in HORIZONS_SEC:
        print(f"  {display}  H={horizon}s: sweeping validation thresholds ...", flush=True)
        val_rows = val_all[val_all["horizon_seconds"] == horizon]
        val_scorer = Scorer(val_rows, events, "D1", horizon, hours_val)
        prob = val_scorer.p["predicted_probability"].to_numpy(dtype=float)

        curve = pd.DataFrame([
            {"threshold": thr, **val_scorer.score(prob >= thr)} for thr in candidate_thresholds(prob)])
        curve["tiebreak"] = curve["threshold"]
        curves.append(curve.assign(model=display, horizon_seconds=horizon))

        test_scorers = {}
        for ds in DATASETS:
            rows = test_all[(test_all["dataset"] == ds) & (test_all["horizon_seconds"] == horizon)]
            test_scorers[ds] = Scorer(rows, events, ds, horizon, hours_test[ds])

        # operating points: the original F1-optimal threshold (for verification) + each budget
        points = [("f1_optimal", np.nan,
                   float(test_all.loc[test_all["horizon_seconds"] == horizon,
                                      "selected_probability_threshold"].iloc[0]), None)]
        cap = active_cap(curve, active_ratio)
        for budget in budgets:
            pos, feasible = choose_operating_point(curve, budget, cap)
            row = curve.iloc[pos]
            points.append((f"budget_{budget:g}", budget, float(row["threshold"]), (row, feasible)))
            selected.append({
                "model": display, "horizon_seconds": horizon, "budget_fa_per_router_hour": budget,
                "max_active_pct_cap": cap, "val_alert_active_time_pct": row["alert_active_time_pct"],
                "feasible": feasible, "threshold": float(row["threshold"]),
                "val_detection_rate_pct": row["event_detection_rate_pct"],
                "val_false_alarms_per_router_hour": row["false_alarms_per_router_hour"],
                "val_alert_precision_pct": row["alert_precision_pct"],
                "val_eligible_events": int(row["eligible_events"]),
            })

        for name, budget, thr, _ in points:
            for ds in DATASETS:
                sc = test_scorers[ds]
                result = sc.score(sc.p["predicted_probability"].to_numpy(dtype=float) >= thr)
                summary.append({"model": display, "operating_point": name,
                                "budget_fa_per_router_hour": budget, "dataset": ds,
                                "horizon_seconds": horizon, "threshold": thr, **result})
    return curves, selected, summary


# --------------------------------------------------------------------- rule-based model
def run_rule_model(events, hours_val, hours_test, budgets, active_ratio):
    datasets = {ds: load_model_ready(ds, exclude_idle_router=True) for ds in DATASETS}
    d1 = datasets[rule.TRAIN_DATASET]
    curves, selected, summary = [], [], []

    for horizon in HORIZONS_SEC:
        print(f"  {RULE_NAME}  H={horizon}s: sweeping validation rules ...", flush=True)
        target = f"y_onset_h{horizon}"
        train = rule.eligible_rows(d1, target, split="train")
        val = rule.eligible_rows(d1, target, split="val")
        val = val.assign(actual_onset=val[target].astype("int8"))
        val_scorer = Scorer(val, events, "D1", horizon, hours_val)

        candidates = [(float(bw), rise)
                      for bw in rule.BW_THRESHOLD_CANDIDATES
                      for rise in rule.make_rise_thresholds(train, target)]
        curve_rows = []
        for bw, rise in candidates:
            warn = rule.predict_rule(val_scorer.p, bw, rise).astype(bool)
            curve_rows.append({"bw_threshold_mbps": bw,
                               "rise_threshold_mbps_per_10s": np.nan if rise is None else float(rise),
                               **val_scorer.score(warn)})
        curve = pd.DataFrame(curve_rows)
        curve["tiebreak"] = curve["bw_threshold_mbps"]
        curves.append(curve.assign(model=RULE_NAME, horizon_seconds=horizon))

        test_scorers = {}
        for ds in DATASETS:
            part = datasets[ds]
            part = part[part["split"] == "test"] if ds == rule.TRAIN_DATASET else part[part["split"] != "purge"]
            rows = rule.eligible_rows(part, target)
            rows = rows.assign(actual_onset=rows[target].astype("int8"))
            test_scorers[ds] = Scorer(rows, events, ds, horizon, hours_test[ds])

        f1_rule, _ = rule.choose_rule(train, rule.eligible_rows(d1, target, split="val"), target)
        points = [("f1_optimal", np.nan, float(f1_rule["bw_threshold"]), f1_rule["rise_threshold"])]
        cap = active_cap(curve, active_ratio)
        for budget in budgets:
            pos, feasible = choose_operating_point(curve, budget, cap)
            row = curve.iloc[pos]
            rise = None if pd.isna(row["rise_threshold_mbps_per_10s"]) else float(row["rise_threshold_mbps_per_10s"])
            points.append((f"budget_{budget:g}", budget, float(row["bw_threshold_mbps"]), rise))
            selected.append({
                "model": RULE_NAME, "horizon_seconds": horizon, "budget_fa_per_router_hour": budget,
                "max_active_pct_cap": cap, "val_alert_active_time_pct": row["alert_active_time_pct"],
                "feasible": feasible, "bw_threshold_mbps": float(row["bw_threshold_mbps"]),
                "rise_threshold_mbps_per_10s": row["rise_threshold_mbps_per_10s"],
                "val_detection_rate_pct": row["event_detection_rate_pct"],
                "val_false_alarms_per_router_hour": row["false_alarms_per_router_hour"],
                "val_alert_precision_pct": row["alert_precision_pct"],
                "val_eligible_events": int(row["eligible_events"]),
            })

        for name, budget, bw, rise in points:
            for ds in DATASETS:
                sc = test_scorers[ds]
                result = sc.score(rule.predict_rule(sc.p, bw, rise).astype(bool))
                summary.append({"model": RULE_NAME, "operating_point": name,
                                "budget_fa_per_router_hour": budget, "dataset": ds,
                                "horizon_seconds": horizon, "bw_threshold_mbps": bw,
                                "rise_threshold_mbps_per_10s": np.nan if rise is None else rise, **result})
    return curves, selected, summary


# ------------------------------------------------------------------------------ main
def verify_against_script_11(summary: pd.DataFrame) -> None:
    path = TAB_DIR / "event_level_summary_all_models.csv"
    if not path.exists():
        print("\n[verify] event_level_summary_all_models.csv not found; skipping cross-check with script 11.")
        return
    ref = pd.read_csv(path)
    mine = summary[summary["operating_point"] == "f1_optimal"]
    merged = mine.merge(ref, on=["model", "dataset", "horizon_seconds"], suffixes=("", "_ref"))
    if merged.empty:
        print("\n[verify] no overlapping rows with script 11's summary; skipping.")
        return
    mismatch = ((merged["detected_events"] != merged["detected_events_ref"])
                | (merged["false_alarms"] != merged["false_alarms_ref"])
                | (merged["eligible_events"] != merged["eligible_events_ref"]))
    if "alert_active_time_pct_ref" in merged.columns:        # present once script 11 has been rerun
        mismatch |= (merged["alert_active_time_pct"] - merged["alert_active_time_pct_ref"]).abs() > 1e-3
    bad = merged[mismatch]
    if len(bad):
        print(bad[["model", "dataset", "horizon_seconds", "detected_events", "detected_events_ref",
                   "false_alarms", "false_alarms_ref"]].to_string(index=False))
        raise RuntimeError("Re-scoring at the F1-optimal operating point does NOT reproduce script 11.")
    print(f"\n[verify] OK: re-scoring at the F1-optimal operating points reproduces script 11 exactly "
          f"({len(merged)} model/dataset/horizon rows).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed false-alarm-budget operating points.")
    parser.add_argument("--budgets", type=float, nargs="+", default=DEFAULT_BUDGETS,
                        help="false alarms per router-hour allowed on D1 validation (default: 1 2 3 5)")
    parser.add_argument("--max-active-ratio", type=float, default=2.0,
                        help="max time under alert as a multiple of the ideal alarm's (default 2; 0 = off)")
    args = parser.parse_args()
    budgets = sorted(set(args.budgets))
    active_ratio = args.max_active_ratio

    print("\nFIXED FALSE-ALARM-BUDGET OPERATING POINTS")
    print("=" * 78)
    print(f"Budgets (false alarms per router-hour, chosen on D1 validation): {budgets}")
    print("Alert-time cap: " + (f"{active_ratio:g} x the ideal alarm's time under alert"
                                if active_ratio > 0 else "off"))

    events = load_events()
    hours_val = router_hours("D1", "val")
    hours_test = {ds: ev.calculate_monitoring_hours(ds) for ds in DATASETS}
    print(f"D1 validation exposure: {hours_val:.2f} router-hours | test exposure: "
          + ", ".join(f"{ds}={h:.1f}" for ds, h in hours_test.items()))

    all_curves, all_selected, all_summary = [], [], []
    for prefix, display in ML_MODELS:
        if not (TAB_DIR / f"{prefix}_onset_val_predictions.csv").exists():
            print(f"\n[skip] {display}: validation predictions not found (rerun its script).")
            continue
        c, s, m = run_ml_model(prefix, display, events, hours_val, hours_test, budgets, active_ratio)
        all_curves += c
        all_selected += s
        all_summary += m
    c, s, m = run_rule_model(events, hours_val, hours_test, budgets, active_ratio)
    all_curves += c
    all_selected += s
    all_summary += m

    curves = pd.concat(all_curves, ignore_index=True).drop(columns=["tiebreak"])
    selected = pd.DataFrame(all_selected)
    summary = pd.DataFrame(all_summary)
    summary["fa_realized_over_budget"] = (
        summary["false_alarms_per_router_hour"] / summary["budget_fa_per_router_hour"]).round(3)
    summary["active_time_over_ideal"] = (
        summary["alert_active_time_pct"] / summary["oracle_active_pct"]).round(3)

    curves.to_csv(TAB_DIR / "fixed_budget_curves_validation.csv", index=False)
    selected.to_csv(TAB_DIR / "fixed_budget_selected_operating_points.csv", index=False)
    summary.to_csv(TAB_DIR / "fixed_budget_event_level_summary.csv", index=False)

    verify_against_script_11(summary)

    budget_rows = summary[summary["operating_point"] != "f1_optimal"]
    for ds in ("D1", "D3"):
        print(f"\n{'=' * 78}\n{ds}: event detection % / false alarms per router-hour at each budget\n{'=' * 78}")
        for horizon in HORIZONS_SEC:
            part = budget_rows[(budget_rows["dataset"] == ds) & (budget_rows["horizon_seconds"] == horizon)]
            table = part.assign(cell=part.apply(
                lambda r: f"{r['event_detection_rate_pct']:.1f} / {r['false_alarms_per_router_hour']:.1f}", axis=1)
            ).pivot(index="model", columns="budget_fa_per_router_hour", values="cell")
            table.columns = [f"budget {c:g}" for c in table.columns]
            print(f"\nH = {horizon} s")
            print(table.to_string())
    infeasible = selected[~selected["feasible"]]
    if len(infeasible):
        print("\nBudgets that could not be met on validation (lowest-false-alarm point used instead):")
        print(infeasible[["model", "horizon_seconds", "budget_fa_per_router_hour",
                          "val_false_alarms_per_router_hour"]].to_string(index=False))
    print("\nSaved: fixed_budget_curves_validation.csv, fixed_budget_selected_operating_points.csv,"
          " fixed_budget_event_level_summary.csv")


if __name__ == "__main__":
    main()

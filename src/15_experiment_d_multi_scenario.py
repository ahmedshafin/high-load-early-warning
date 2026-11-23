"""Experiment D: does training on a second day (D1 + D2) improve transfer to the bursty day D3?

Two training regimes are compared with the SAME model code (XGBoost, same hyperparameters,
same feature subsets as 12/14) and the SAME budget-based operating-point selection as 13/14:

    train_D1      train on D1-train;        tune on D1-validation       (the setup of scripts 12-14)
    train_D1_D2   train on D1-train+D2-train; tune on D1-val + D2-val pooled

Tuning means early stopping and choosing the probability threshold for each false-alarm budget
(max event detection subject to false alarms per router-hour <= budget and the alert-time cap).
D3 is NEVER used for training or tuning in either regime.

Every regime is evaluated on identical test sets:
    D1_test   the untouched chronological test split of D1
    D2_test   the untouched chronological test split of D2
    D3        all non-purge rows of D3 (the held-out bursty day)
(D2_test is held out in BOTH regimes, so the D2 comparison is fair; script 13 used all of D2.)

What to look at: D3 detection / false alarms / PR-AUC for the two regimes, and whether the
false-alarm rate on D3 lands closer to the budget when D2 is part of training.

Run from the repository root, after scripts 02, 08-14:
    python src/15_experiment_d_multi_scenario.py
    python src/15_experiment_d_multi_scenario.py --configs D_all A_bandwidth --horizons 20 30

Outputs (reports/tables/)
-------------------------
experiment_d_results.csv      event-level results per regime / config / horizon / budget / test set
experiment_d_pr_auc.csv       threshold-free row-level PR-AUC per regime / config / horizon / test set
experiment_d_comparison.csv   train_D1_D2 minus train_D1 differences (detection, false alarms, PR-AUC)
experiment_d_test_predictions.parquet  test-set probabilities per regime/config (input to script 16; gitignored)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import HORIZONS_SEC, TAB_DIR  # noqa: E402
from features import feature_groups  # noqa: E402
from modeling_utils import load_model_ready  # noqa: E402


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ab = _load("feature_ablation", "14_feature_ablation.py")
fb = ab.fb
XGBClassifier, XGB_PARAMS = ab.XGBClassifier, ab.XGB_PARAMS
prepare_xy, select_telemetry_features = ab.prepare_xy, ab.select_telemetry_features

REGIMES = {"train_D1": ["D1"], "train_D1_D2": ["D1", "D2"]}
BASE_REGIME, MULTI_REGIME = "train_D1", "train_D1_D2"
TEST_SETS = [("D1_test", "D1", "test"), ("D2_test", "D2", "test"), ("D3", "D3", None)]
DATASETS = ("D1", "D2", "D3")
OVER_BUDGET_FACTOR = ab.OVER_BUDGET_FACTOR


# ------------------------------------------------------------------ pooled validation
def pooled_score(scorers, probs, threshold: float) -> dict:
    """Event-level score pooled over several validation scenarios (counts and exposure are summed)."""
    detected = eligible = alerts = matched = false_alarms = 0
    hours = active_s = oracle_s = 0.0
    for scorer, prob in zip(scorers, probs):
        r = scorer.score(prob >= threshold)
        detected += r["detected_events"]
        eligible += r["eligible_events"]
        alerts += r["alert_episodes"]
        matched += r["matched_alerts"]
        false_alarms += r["false_alarms"]
        hours += scorer.hours
        active_s += r["alert_active_time_pct"] / 100.0 * scorer.hours * 3600.0
        oracle_s += r["oracle_active_pct"] / 100.0 * scorer.hours * 3600.0
    seconds = hours * 3600.0
    return {
        "eligible_events": eligible,
        "detected_events": detected,
        "event_detection_rate_pct": round(detected / eligible * 100, 3) if eligible else np.nan,
        "alert_episodes": alerts,
        "matched_alerts": matched,
        "false_alarms": false_alarms,
        "alert_precision_pct": round(matched / alerts * 100, 3) if alerts else np.nan,
        "monitoring_router_hours": round(hours, 3),
        "false_alarms_per_router_hour": round(false_alarms / hours, 4) if hours > 0 else np.nan,
        "alert_active_time_pct": round(active_s / seconds * 100, 3) if seconds > 0 else np.nan,
        "oracle_active_pct": round(oracle_s / seconds * 100, 3) if seconds > 0 else np.nan,
    }


# ------------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment D: multi-scenario training (D1+D2 -> D3).")
    parser.add_argument("--horizons", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--budgets", type=float, nargs="+", default=[1.0, 3.0])
    parser.add_argument("--configs", nargs="+", default=["D_all"],
                        help="feature configurations from 14_feature_ablation.py (default: D_all)")
    parser.add_argument("--max-active-ratio", type=float, default=2.0)
    args = parser.parse_args()
    bad = [h for h in args.horizons if h not in HORIZONS_SEC]
    if bad:
        raise SystemExit(f"Unsupported horizons {bad}; available: {HORIZONS_SEC}")
    budgets = sorted(set(args.budgets))

    print("\nEXPERIMENT D: single-scenario vs multi-scenario training (XGBoost, fixed budgets)")
    print("=" * 78)
    events = fb.load_events()
    datasets = {ds: load_model_ready(ds, exclude_idle_router=True) for ds in DATASETS}
    d1 = datasets["D1"]
    feature_cols, _ = select_telemetry_features(d1)
    all_configs = ab.build_configs(d1, feature_cols, extras=True)
    unknown = [c for c in args.configs if c not in all_configs]
    if unknown:
        raise SystemExit(f"Unknown configs {unknown}; available: {list(all_configs)}")
    configs = {name: all_configs[name] for name in args.configs}

    hours = {("D1", "val"): fb.router_hours("D1", "val"), ("D2", "val"): fb.router_hours("D2", "val")}
    for _, ds, split in TEST_SETS:
        hours[(ds, split)] = fb.router_hours(ds, split)
    print("Exposure (router-hours): " + ", ".join(f"{ds}/{sp or 'non-purge'}={h:.1f}" for (ds, sp), h in hours.items()))

    result_rows, pr_rows = [], []
    pred_frames: list[pd.DataFrame] = []      # test-set probabilities, saved for 16_bootstrap_confidence_intervals.py
    for horizon in args.horizons:
        target = f"y_onset_h{horizon}"
        print(f"\n{'-' * 70}\nHorizon {horizon}s\n{'-' * 70}", flush=True)

        test_parts, test_scorers = {}, {}
        for name, ds, split in TEST_SETS:
            part = ab.eligible_part(datasets[ds], ds, target, split)
            test_parts[name] = part
            test_scorers[name] = ab.make_scorer(part, events, ds, horizon, hours[(ds, split)])
        wide = {name: test_parts[name][["host", "ts_unix", "actual_onset"]].copy() for name, _, _ in TEST_SETS}

        for regime, train_ds in REGIMES.items():
            parts = [prepare_xy(datasets[ds], target, feature_cols, split="train") for ds in train_ds]
            X_train = pd.concat([p[0] for p in parts], ignore_index=True)
            y_train = np.concatenate([p[1] for p in parts])
            imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_train)
            X_train_imp = imputer.transform(X_train)
            scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

            val_parts = [ab.eligible_part(datasets[ds], ds, target, "val") for ds in train_ds]
            val_X = [ab.design_matrix(p, feature_cols, imputer) for p in val_parts]
            val_scorers = [ab.make_scorer(p, events, ds, horizon, hours[(ds, "val")])
                           for p, ds in zip(val_parts, train_ds)]
            X_val_all = np.vstack(val_X)
            y_val_all = np.concatenate([p["actual_onset"].to_numpy() for p in val_parts])
            test_X = {name: ab.design_matrix(test_parts[name], feature_cols, imputer) for name, _, _ in TEST_SETS}

            for cfg_name, cols in configs.items():
                idx = [feature_cols.index(c) for c in cols]
                model = XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
                model.fit(X_train_imp[:, idx], y_train, eval_set=[(X_val_all[:, idx], y_val_all)], verbose=False)
                print(f"  {regime} / {cfg_name}: trained on {len(y_train):,} rows, {len(cols)} features; "
                      "sweeping pooled validation thresholds ...", flush=True)

                val_probs = [model.predict_proba(x[:, idx])[:, 1] for x in val_X]
                curve = pd.DataFrame([
                    {"threshold": t, **pooled_score(val_scorers, val_probs, t)}
                    for t in fb.candidate_thresholds(np.concatenate(val_probs))])
                curve["tiebreak"] = curve["threshold"]
                cap = fb.active_cap(curve, args.max_active_ratio)

                test_probs = {name: model.predict_proba(test_X[name][:, idx])[:, 1] for name, _, _ in TEST_SETS}
                for name, _, _ in TEST_SETS:
                    wide[name][f"prob__{cfg_name}__{regime}"] = test_probs[name]
                for name, _, _ in TEST_SETS:
                    y = test_parts[name]["actual_onset"].to_numpy()
                    pr_rows.append({"regime": regime, "config": cfg_name, "horizon_seconds": horizon,
                                    "test_set": name,
                                    "pr_auc": round(float(average_precision_score(y, test_probs[name])), 4),
                                    "prevalence_pct": round(float(y.mean() * 100), 3)})

                for budget in budgets:
                    pos, feasible = fb.choose_operating_point(curve, budget, cap)
                    chosen = curve.iloc[pos]
                    for name, _, _ in TEST_SETS:
                        score = test_scorers[name].score(test_probs[name] >= float(chosen["threshold"]))
                        result_rows.append({
                            "regime": regime, "config": cfg_name, "n_features": len(cols),
                            "horizon_seconds": horizon, "budget_fa_per_router_hour": budget,
                            "test_set": name, "threshold": float(chosen["threshold"]), "feasible": feasible,
                            "max_active_pct_cap": cap,
                            "val_detection_rate_pct": chosen["event_detection_rate_pct"],
                            "val_false_alarms_per_router_hour": chosen["false_alarms_per_router_hour"],
                            **score})

        for name, _, _ in TEST_SETS:
            pred_frames.append(wide[name].assign(horizon_seconds=horizon, test_set=name))

    pd.concat(pred_frames, ignore_index=True).to_parquet(
        TAB_DIR / "experiment_d_test_predictions.parquet", index=False)

    results = pd.DataFrame(result_rows)
    results["fa_realized_over_budget"] = (
        results["false_alarms_per_router_hour"] / results["budget_fa_per_router_hour"]).round(3)
    results["over_budget"] = results["fa_realized_over_budget"] > OVER_BUDGET_FACTOR
    pr_auc = pd.DataFrame(pr_rows)
    comparison = build_comparison(results, pr_auc)

    results.to_csv(TAB_DIR / "experiment_d_results.csv", index=False)
    pr_auc.to_csv(TAB_DIR / "experiment_d_pr_auc.csv", index=False)
    comparison.to_csv(TAB_DIR / "experiment_d_comparison.csv", index=False)

    verify_against_script_13(results)
    print_summary(results, pr_auc, comparison, budgets, args.horizons)


# ----------------------------------------------------------------------- reporting
def build_comparison(results: pd.DataFrame, pr_auc: pd.DataFrame) -> pd.DataFrame:
    keys = ["config", "horizon_seconds", "budget_fa_per_router_hour", "test_set"]
    cols = ["event_detection_rate_pct", "false_alarms_per_router_hour", "fa_realized_over_budget",
            "alert_precision_pct"]
    base = results[results["regime"] == BASE_REGIME].set_index(keys)[cols]
    multi = results[results["regime"] == MULTI_REGIME].set_index(keys)[cols]
    diff = (multi - base).add_prefix("delta_").round(3)
    out = base.add_prefix("train_D1_").join(multi.add_prefix("train_D1_D2_")).join(diff).reset_index()

    pk = ["config", "horizon_seconds", "test_set"]
    pbase = pr_auc[pr_auc["regime"] == BASE_REGIME].set_index(pk)["pr_auc"]
    pmulti = pr_auc[pr_auc["regime"] == MULTI_REGIME].set_index(pk)["pr_auc"]
    pdiff = (pmulti - pbase).round(4).rename("delta_pr_auc")
    return out.merge(pdiff.reset_index(), on=pk, how="left")


def verify_against_script_13(results: pd.DataFrame) -> None:
    path = TAB_DIR / "fixed_budget_event_level_summary.csv"
    if not path.exists():
        print("\n[check] fixed_budget_event_level_summary.csv not found; skipping the train_D1 reproduction check.")
        return
    ref = pd.read_csv(path)
    ref = ref[(ref["model"] == "XGBoost") & (ref["operating_point"] != "f1_optimal")]
    ref["test_set"] = ref["dataset"].map({"D1": "D1_test", "D3": "D3"})
    mine = results[(results["regime"] == BASE_REGIME) & (results["config"] == "D_all")]
    merged = mine.merge(ref.dropna(subset=["test_set"]),
                        on=["horizon_seconds", "budget_fa_per_router_hour", "test_set"], suffixes=("", "_s13"))
    if merged.empty:
        print("\n[check] no train_D1 / D_all rows overlap with script 13; skipping.")
        return
    d_det = (merged["event_detection_rate_pct"] - merged["event_detection_rate_pct_s13"]).abs().max()
    d_fa = (merged["false_alarms_per_router_hour"] - merged["false_alarms_per_router_hour_s13"]).abs().max()
    verdict = "OK" if d_det < 1.0 and d_fa < 0.25 else "CHECK (larger than expected)"
    print(f"\n[check] train_D1 / D_all vs script 13 XGBoost rows on D1_test and D3 ({len(merged)} rows): "
          f"max |detection diff| = {d_det:.2f} points, max |false-alarm diff| = {d_fa:.3f} -> {verdict}")


def _cell(r) -> str:
    star = "*" if r["over_budget"] else " "
    return f"{r['event_detection_rate_pct']:5.1f} / {r['false_alarms_per_router_hour']:4.1f}{star}"


def print_summary(results, pr_auc, comparison, budgets, horizons) -> None:
    print(f"\n{'=' * 78}\nD3 (held-out bursty day): event detection % / false alarms per router-hour\n{'=' * 78}")
    print(f"* = false-alarm rate more than {int((OVER_BUDGET_FACTOR - 1) * 100)}% above the budget "
          "(detection bought with extra alarms).")
    for horizon in horizons:
        part = results[(results["test_set"] == "D3") & (results["horizon_seconds"] == horizon)]
        part = part.assign(cell=part.apply(_cell, axis=1), row=part["config"] + " | " + part["regime"])
        table = part.pivot(index="row", columns="budget_fa_per_router_hour", values="cell")
        table.columns = [f"budget {c:g}" for c in table.columns]
        print(f"\nH = {horizon} s")
        print(table.to_string())

    print(f"\n{'=' * 78}\nRow-level PR-AUC (threshold-free)\n{'=' * 78}")
    for horizon in horizons:
        part = pr_auc[pr_auc["horizon_seconds"] == horizon]
        table = part.assign(row=part["config"] + " | " + part["regime"]).pivot(
            index="row", columns="test_set", values="pr_auc")[["D1_test", "D2_test", "D3"]]
        table["gap D1_test->D3"] = (table["D1_test"] - table["D3"]).round(4)
        print(f"\nH = {horizon} s")
        print(table.to_string())

    print(f"\n{'=' * 78}\nEffect of adding D2 to training (train_D1_D2 minus train_D1)\n{'=' * 78}")
    show = comparison[comparison["test_set"] == "D3"][
        ["config", "horizon_seconds", "budget_fa_per_router_hour", "delta_event_detection_rate_pct",
         "delta_false_alarms_per_router_hour", "train_D1_fa_realized_over_budget",
         "train_D1_D2_fa_realized_over_budget", "delta_pr_auc"]].rename(columns={
        "delta_event_detection_rate_pct": "d_detection_pts", "delta_false_alarms_per_router_hour": "d_false_alarms",
        "train_D1_fa_realized_over_budget": "FA/budget (D1)", "train_D1_D2_fa_realized_over_budget": "FA/budget (D1+D2)",
        "delta_pr_auc": "d_PR-AUC"})
    print("\nOn D3:")
    print(show.to_string(index=False))
    print("\nSaved: experiment_d_results.csv, experiment_d_pr_auc.csv, experiment_d_comparison.csv")


if __name__ == "__main__":
    main()

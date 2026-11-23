"""Feature ablation for the XGBoost onset model, scored at fixed false-alarm budgets.

Question answered: WHICH telemetry makes the model work? Each configuration trains the same
XGBoost model (same hyperparameters, same data, same early stopping as 12_xgboost_onset.py) on
a different subset of the telemetry features, then is scored exactly like script 13: operating
points are chosen on D1 VALIDATION for each false-alarm budget (with the same alert-time cap),
frozen, and applied to the D1 test split, D2 and D3 with script 11's event-level scoring.

Cumulative signal-source ablation (the four configurations)
    A_bandwidth            current bandwidth + the history (lags, trends, trailing windows) of bandwidth
    B_plus_packets         A + packet/error counters and their trailing windows
    C_plus_tcp_udp         B + TCP/UDP counters
    D_all                  C + network context (other routers' load) + meta flags
                           = the exact feature set used by scripts 09, 10 and 12

Extra diagnostics (skip with --no-extras)
    E_no_history           D_all minus every lag / trend / trailing-window feature (a snapshot model)
    F_no_context           D_all minus the network-context features (other routers' load)
    G_bandwidth_plus_context  A + network context (does context substitute for packet counters?)

Everything is computed from saved model-ready data; no label-derived feature is ever used
(select_telemetry_features from script 10 removes them first).

Run from the repository root, after scripts 02, 08-13:
    python src/14_feature_ablation.py
    python src/14_feature_ablation.py --horizons 20 30 --budgets 1 3 --max-active-ratio 2

Outputs (reports/tables/)
-------------------------
feature_ablation_results.csv             event-level results per config / horizon / budget / dataset
feature_ablation_pr_auc.csv              threshold-free row-level PR-AUC per config / horizon / dataset
feature_ablation_feature_importance.csv  XGBoost gain importance per config, feature and feature group
feature_ablation_configs.csv             which features each configuration contains
feature_ablation_test_predictions.parquet  test-set probabilities per config (input to script 16; gitignored)
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

from config import HORIZONS_SEC, SEED, TAB_DIR  # noqa: E402
from features import feature_groups  # noqa: E402
from modeling_utils import load_model_ready  # noqa: E402


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


xgb12 = _load("xgb_onset", "12_xgboost_onset.py")       # hyperparameters, protocol helpers
fb = _load("fixed_budget", "13_fixed_budget_thresholds.py")   # Scorer, operating-point selection

XGBClassifier = xgb12.XGBClassifier
XGB_PARAMS = xgb12.XGB_PARAMS
select_telemetry_features = xgb12.select_telemetry_features
prepare_xy = xgb12.prepare_xy
TRAIN_DATASET = xgb12.TRAIN_DATASET
DATASETS = fb.DATASETS

BANDWIDTH_PREFIXES = ("bw_",)
PACKET_PREFIXES = ("eth_", "ucast_", "nucast_", "discards_", "rmon_")
TCPUDP_PREFIXES = ("tcp", "udp")

OVER_BUDGET_FACTOR = 1.25      # realized false alarms > 1.25 x budget => flagged with "*" in the tables
DEFAULT_HORIZONS = [20, 30]
DEFAULT_BUDGETS = [1.0, 3.0]


# ------------------------------------------------------------------ configurations
def signal_source(column: str) -> str:
    if column.startswith(BANDWIDTH_PREFIXES):
        return "bandwidth"
    if column.startswith(PACKET_PREFIXES):
        return "packets"
    if column.startswith(TCPUDP_PREFIXES):
        return "tcp_udp"
    raise ValueError(f"Cannot assign feature '{column}' to a signal source; update the ablation configuration.")


def build_configs(df: pd.DataFrame, feature_cols: list[str], extras: bool = True) -> dict[str, list[str]]:
    """Feature subsets for each configuration, in the column order of `feature_cols`."""
    allowed = set(feature_cols)
    groups = {g: [c for c in cols if c in allowed] for g, cols in feature_groups(df).items()}

    current = {"bandwidth": [], "packets": [], "tcp_udp": []}
    for col in groups["current_bandwidth"]:
        current["bandwidth"].append(col)
    for col in groups["current_packets_errors"]:
        current["packets"].append(col)
    for col in groups["current_tcp_udp"]:
        current["tcp_udp"].append(col)

    history = {"bandwidth": [], "packets": [], "tcp_udp": []}
    for col in groups["lags_trend"] + groups["trailing_windows"]:
        history[signal_source(col)].append(col)

    context = groups["network_context"]
    meta = groups["meta_flags"]
    all_history = [c for src in history.values() for c in src]

    a = current["bandwidth"] + history["bandwidth"]
    b = a + current["packets"] + history["packets"]
    c = b + current["tcp_udp"] + history["tcp_udp"]
    d = c + context + meta

    if set(d) != allowed:
        raise ValueError("Ablation configurations do not cover the feature set; unassigned: "
                         f"{sorted(allowed - set(d))}, unexpected: {sorted(set(d) - allowed)}")

    configs = {"A_bandwidth": a, "B_plus_packets": b, "C_plus_tcp_udp": c, "D_all": d}
    if extras:
        configs["E_no_history"] = [x for x in d if x not in set(all_history)]
        configs["F_no_context"] = [x for x in d if x not in set(context)]
        configs["G_bandwidth_plus_context"] = a + context
    order = {col: i for i, col in enumerate(feature_cols)}
    return {name: sorted(dict.fromkeys(cols), key=order.__getitem__) for name, cols in configs.items()}


# ------------------------------------------------------------------------ data prep
def eligible_part(df: pd.DataFrame, dataset: str, target: str, split: str | None) -> pd.DataFrame:
    """Rows with a valid (0/1) onset target, sorted by (host, time) -- the order Scorer uses."""
    if split is not None:
        part = df[df["split"] == split]
    elif dataset == TRAIN_DATASET:
        part = df[df["split"] == "test"]
    else:
        part = df[df["split"] != "purge"]
    y = pd.to_numeric(part[target], errors="coerce")
    part = part.loc[y.isin([0, 1])].copy()
    part["host"] = part["host"].astype(str)
    part["ts_unix"] = part["ts_unix"].astype("int64")
    part["actual_onset"] = pd.to_numeric(part[target]).astype("int8")
    return part.sort_values(["host", "ts_unix"]).reset_index(drop=True)


def design_matrix(part: pd.DataFrame, cols: list[str], imputer: SimpleImputer) -> np.ndarray:
    X = part[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return imputer.transform(X)


def make_scorer(part, events, dataset, horizon, hours):
    scorer = fb.Scorer(part[["dataset", "host", "ts_unix", "actual_onset"]], events, dataset, horizon, hours)
    if not (np.array_equal(scorer.ts, part["ts_unix"].to_numpy())
            and np.array_equal(scorer.p["host"].to_numpy(), part["host"].to_numpy())):
        raise RuntimeError("Scorer row order differs from the design-matrix row order.")
    return scorer


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost feature ablation at fixed false-alarm budgets.")
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--budgets", type=float, nargs="+", default=DEFAULT_BUDGETS,
                        help="false alarms per router-hour chosen on D1 validation (default: 1 3)")
    parser.add_argument("--max-active-ratio", type=float, default=2.0,
                        help="max time under alert as a multiple of the ideal alarm's (0 = off)")
    parser.add_argument("--no-extras", action="store_true", help="only the four cumulative configurations")
    args = parser.parse_args()
    bad = [h for h in args.horizons if h not in HORIZONS_SEC]
    if bad:
        raise SystemExit(f"Unsupported horizons {bad}; available: {HORIZONS_SEC}")
    budgets = sorted(set(args.budgets))

    print("\nFEATURE ABLATION (XGBoost, fixed false-alarm budgets)")
    print("=" * 78)
    events = fb.load_events()
    hours_val = fb.router_hours("D1", "val")
    hours_test = {ds: fb.ev.calculate_monitoring_hours(ds) for ds in DATASETS}
    datasets = {ds: load_model_ready(ds, exclude_idle_router=True) for ds in DATASETS}
    d1 = datasets[TRAIN_DATASET]

    feature_cols, _ = select_telemetry_features(d1)
    configs = build_configs(d1, feature_cols, extras=not args.no_extras)
    group_of = {c: g for g, cols in feature_groups(d1).items() for c in cols}

    print(f"Telemetry features available: {len(feature_cols)}")
    for name, cols in configs.items():
        print(f"  {name:26s} {len(cols):3d} features")
    pd.DataFrame([{"config": n, "n_features": len(c), "feature": f, "feature_group": group_of.get(f, "other")}
                  for n, c in configs.items() for f in c]).to_csv(
        TAB_DIR / "feature_ablation_configs.csv", index=False)

    result_rows, pr_rows, importance_rows = [], [], []
    pred_frames: list[pd.DataFrame] = []      # test-set probabilities, saved for 16_bootstrap_confidence_intervals.py

    for horizon in args.horizons:
        target = f"y_onset_h{horizon}"
        print(f"\n{'-' * 70}\nHorizon {horizon}s\n{'-' * 70}", flush=True)

        X_train, y_train = prepare_xy(d1, target, feature_cols, split="train")
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_train)
        X_train_imp = imputer.transform(X_train)
        scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

        val_part = eligible_part(d1, TRAIN_DATASET, target, "val")
        X_val_imp = design_matrix(val_part, feature_cols, imputer)
        y_val = val_part["actual_onset"].to_numpy()
        val_scorer = make_scorer(val_part, events, "D1", horizon, hours_val)

        test_parts, test_X, test_scorers = {}, {}, {}
        for ds in DATASETS:
            part = eligible_part(datasets[ds], ds, target, None)
            test_parts[ds] = part
            test_X[ds] = design_matrix(part, feature_cols, imputer)
            test_scorers[ds] = make_scorer(part, events, ds, horizon, hours_test[ds])

        wide = {ds: test_parts[ds][["host", "ts_unix", "actual_onset"]].copy() for ds in DATASETS}
        cache: dict[tuple, dict] = {}       # identical column sets are trained only once
        for name, cols in configs.items():
            signature = tuple(cols)
            if signature in cache:
                out = cache[signature]
                print(f"  {name}: same features as an earlier configuration; reusing its model", flush=True)
            else:
                idx = [feature_cols.index(c) for c in cols]
                model = XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
                model.fit(X_train_imp[:, idx], y_train, eval_set=[(X_val_imp[:, idx], y_val)], verbose=False)
                out = {
                    "val_prob": model.predict_proba(X_val_imp[:, idx])[:, 1],
                    "test_prob": {ds: model.predict_proba(test_X[ds][:, idx])[:, 1] for ds in DATASETS},
                    "importance": dict(zip(cols, np.asarray(model.feature_importances_, dtype=float))),
                }
                cache[signature] = out
                print(f"  {name}: trained on {len(cols)} features; sweeping validation thresholds ...", flush=True)

            val_prob = out["val_prob"]
            curve = pd.DataFrame([{"threshold": t, **val_scorer.score(val_prob >= t)}
                                  for t in fb.candidate_thresholds(val_prob)])
            curve["tiebreak"] = curve["threshold"]
            cap = fb.active_cap(curve, args.max_active_ratio)

            for ds in DATASETS:
                wide[ds][f"prob__{name}"] = out["test_prob"][ds]

            for feature, value in out["importance"].items():
                importance_rows.append({"config": name, "horizon_seconds": horizon, "feature": feature,
                                        "feature_group": group_of.get(feature, "other"), "importance": value})

            for ds in DATASETS:
                pr_rows.append({"config": name, "n_features": len(cols), "horizon_seconds": horizon,
                                "dataset": ds, "pr_auc": round(float(average_precision_score(
                                    test_parts[ds]["actual_onset"], out["test_prob"][ds])), 4),
                                "prevalence_pct": round(float(test_parts[ds]["actual_onset"].mean() * 100), 3)})

            for budget in budgets:
                pos, feasible = fb.choose_operating_point(curve, budget, cap)
                chosen = curve.iloc[pos]
                for ds in DATASETS:
                    score = test_scorers[ds].score(out["test_prob"][ds] >= float(chosen["threshold"]))
                    result_rows.append({
                        "config": name, "n_features": len(cols), "horizon_seconds": horizon,
                        "budget_fa_per_router_hour": budget, "dataset": ds,
                        "threshold": float(chosen["threshold"]), "feasible": feasible,
                        "max_active_pct_cap": cap,
                        "val_detection_rate_pct": chosen["event_detection_rate_pct"],
                        "val_false_alarms_per_router_hour": chosen["false_alarms_per_router_hour"],
                        **score})

        for ds in DATASETS:
            pred_frames.append(wide[ds].assign(horizon_seconds=horizon, dataset=ds))

    pd.concat(pred_frames, ignore_index=True).to_parquet(
        TAB_DIR / "feature_ablation_test_predictions.parquet", index=False)

    results = pd.DataFrame(result_rows)
    base = results[results["config"] == "D_all"].set_index(
        ["horizon_seconds", "budget_fa_per_router_hour", "dataset"])["event_detection_rate_pct"]
    key = list(zip(results["horizon_seconds"], results["budget_fa_per_router_hour"], results["dataset"]))
    results["detection_delta_vs_D_all"] = (
        results["event_detection_rate_pct"].to_numpy() - base.reindex(key).to_numpy()).round(3)
    results["fa_realized_over_budget"] = (
        results["false_alarms_per_router_hour"] / results["budget_fa_per_router_hour"]).round(3)
    base_fa = results[results["config"] == "D_all"].set_index(
        ["horizon_seconds", "budget_fa_per_router_hour", "dataset"])["false_alarms_per_router_hour"]
    results["false_alarm_delta_vs_D_all"] = (
        results["false_alarms_per_router_hour"].to_numpy() - base_fa.reindex(key).to_numpy()).round(4)
    # A detection gain bought with many more false alarms is not a real gain: flag overshoots.
    results["over_budget"] = results["fa_realized_over_budget"] > OVER_BUDGET_FACTOR

    results.to_csv(TAB_DIR / "feature_ablation_results.csv", index=False)
    pr_auc = pd.DataFrame(pr_rows)
    base_pr = pr_auc[pr_auc["config"] == "D_all"].set_index(["horizon_seconds", "dataset"])["pr_auc"]
    pr_auc["pr_auc_delta_vs_D_all"] = (
        pr_auc["pr_auc"].to_numpy() - base_pr.reindex(list(zip(pr_auc["horizon_seconds"], pr_auc["dataset"]))).to_numpy()
    ).round(4)
    pr_auc.to_csv(TAB_DIR / "feature_ablation_pr_auc.csv", index=False)
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(TAB_DIR / "feature_ablation_feature_importance.csv", index=False)

    verify_against_script_13(results)
    print_summary(results, pr_auc, importance, budgets, args.horizons)


# ----------------------------------------------------------------------- reporting
def verify_against_script_13(results: pd.DataFrame) -> None:
    path = TAB_DIR / "fixed_budget_event_level_summary.csv"
    if not path.exists():
        print("\n[check] fixed_budget_event_level_summary.csv not found; skipping the D_all reproduction check.")
        return
    ref = pd.read_csv(path)
    ref = ref[(ref["model"] == "XGBoost") & (ref["operating_point"] != "f1_optimal")]
    mine = results[results["config"] == "D_all"]
    merged = mine.merge(ref, left_on=["horizon_seconds", "budget_fa_per_router_hour", "dataset"],
                        right_on=["horizon_seconds", "budget_fa_per_router_hour", "dataset"], suffixes=("", "_s13"))
    if merged.empty:
        print("\n[check] no overlapping budgets/horizons with script 13's XGBoost rows; skipping.")
        return
    d_det = (merged["event_detection_rate_pct"] - merged["event_detection_rate_pct_s13"]).abs().max()
    d_fa = (merged["false_alarms_per_router_hour"] - merged["false_alarms_per_router_hour_s13"]).abs().max()
    verdict = "OK" if d_det < 1.0 and d_fa < 0.25 else "CHECK (larger than expected)"
    print(f"\n[check] D_all vs script 13 XGBoost rows ({len(merged)} rows): max |detection diff| = "
          f"{d_det:.2f} points, max |false-alarm diff| = {d_fa:.3f}/router-hour -> {verdict}")
    print("        (D_all uses the same features and hyperparameters, so it should reproduce them; "
          "tiny differences can come from multi-threaded training.)")


def print_summary(results, pr_auc, importance, budgets, horizons) -> None:
    for ds in ("D1", "D3"):
        print(f"\n{'=' * 78}\n{ds}: event detection % / false alarms per router-hour"
              f"   (change in detection vs D_all)\n{'=' * 78}")
        print(f"A trailing * means the false-alarm rate is more than {int((OVER_BUDGET_FACTOR - 1) * 100)}% above "
              "the budget: its detection is bought with extra alarms,\nso do not read its change vs D_all as a "
              "real gain. Compare configurations on the PR-AUC tables below.")
        for horizon in horizons:
            part = results[(results["dataset"] == ds) & (results["horizon_seconds"] == horizon)]
            part = part.assign(cell=part.apply(
                lambda r: f"{r['event_detection_rate_pct']:.1f} / {r['false_alarms_per_router_hour']:.1f}"
                          f"{'*' if r['over_budget'] else ' '} ({r['detection_delta_vs_D_all']:+.1f})", axis=1))
            table = part.pivot(index="config", columns="budget_fa_per_router_hour", values="cell")
            table.columns = [f"budget {c:g}" for c in table.columns]
            order = [c for c in part["config"].drop_duplicates()]
            print(f"\nH = {horizon} s")
            print(table.reindex(order).to_string())

    print(f"\n{'=' * 78}\nRow-level PR-AUC (threshold-free), by dataset   [change vs D_all]\n{'=' * 78}")
    for horizon in horizons:
        part = pr_auc[pr_auc["horizon_seconds"] == horizon]
        part = part.assign(cell=part.apply(
            lambda r: f"{r['pr_auc']:.3f} [{r['pr_auc_delta_vs_D_all']:+.3f}]", axis=1))
        table = part.pivot(index="config", columns="dataset", values="cell")
        order = [c for c in part["config"].drop_duplicates()]
        print(f"\nH = {horizon} s")
        print(table.reindex(order).to_string())

    print(f"\n{'=' * 78}\nWhere D_all's importance sits (share of total XGBoost gain by feature group)\n{'=' * 78}")
    for horizon in horizons:
        part = importance[(importance["config"] == "D_all") & (importance["horizon_seconds"] == horizon)]
        share = part.groupby("feature_group")["importance"].sum()
        share = (share / share.sum() * 100).sort_values(ascending=False).round(1)
        print(f"\nH = {horizon} s")
        print(share.to_string())
    print("\nSaved: feature_ablation_results.csv, feature_ablation_pr_auc.csv, "
          "feature_ablation_feature_importance.csv, feature_ablation_configs.csv")


if __name__ == "__main__":
    main()

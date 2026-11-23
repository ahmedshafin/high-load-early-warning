"""Time-block bootstrap confidence intervals for the headline comparisons.

Why blocks?  Onsets are not independent: a traffic wave raises several routers at once and
consecutive samples are strongly dependent, so resampling single rows or events would give
intervals that are far too narrow. Instead the recording of each test set is cut into fixed
time blocks (default 10 minutes, ALL routers together, so simultaneous onsets stay in one
block) and whole blocks are resampled with replacement. The same resampled blocks are used for
every system in a comparison (a paired bootstrap), so differences between systems get their own
intervals.

Three analyses, each re-using the operating points already chosen on validation data
(nothing is retrained and no threshold is re-tuned):

  models        the four-model budget comparison of script 13 (reference system: XGBoost)
  ablation      the feature configurations of script 14 (reference system: D_all)
  experiment_d  training on D1 vs D1+D2 from script 15 (train_D1_D2 minus train_D1)

Metrics (all recomputed from per-block sums, so they equal the original tables on the full data):
  event detection rate, false alarms per router-hour, alert precision, time under alert,
  and row-level PR-AUC (threshold-free; weighted bootstrap of the precision-recall curve).
Alerts and events are assigned to the block in which they start; the alert state machine is run
once on the full series (script 11's logic), so alerts that straddle a block edge are unaffected.

Run from the repository root, after scripts 08-15 (14 and 15 must have been run after they
were updated to save their test-set probabilities):
    python src/16_bootstrap_confidence_intervals.py
    python src/16_bootstrap_confidence_intervals.py --analysis models --n-boot 2000 --block-minutes 10

Outputs (reports/tables/)
-------------------------
bootstrap_summary.csv       estimate and CI for every system / horizon / budget / test set / metric
bootstrap_differences.csv   paired differences vs the reference system, with CIs
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

from config import TAB_DIR  # noqa: E402
from modeling_utils import load_model_ready  # noqa: E402


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fb = _load("fixed_budget", "13_fixed_budget_thresholds.py")
rule = fb.rule
ev = fb.ev

DATASETS = ev.DATASETS
ML_MODELS = [("logistic_regression", "LogisticRegression"), ("random_forest", "RandomForest"),
             ("xgboost", "XGBoost")]
EVENT_METRICS = ["event_detection_rate_pct", "false_alarms_per_router_hour",
                 "alert_precision_pct", "alert_active_time_pct"]
TEST_SET_DATASET = {"D1": ("D1", "test"), "D2": ("D2", None), "D3": ("D3", None),
                    "D1_test": ("D1", "test"), "D2_test": ("D2", "test")}


# ----------------------------------------------------------------- blocks and statistics
def block_stats(scorer, warn: np.ndarray, t0: int, block_seconds: int, n_blocks: int) -> dict:
    """Per-block sufficient statistics of one system at one operating point."""
    alerts, events = scorer.detail(warn)

    def blocks(ts):
        return np.clip((np.asarray(ts, dtype=np.int64) - t0) // block_seconds, 0, n_blocks - 1).astype(np.int64)

    ev_b = blocks(events["ts"]) if len(events) else np.zeros(0, dtype=np.int64)
    al_b = blocks(alerts["start_ts"]) if len(alerts) else np.zeros(0, dtype=np.int64)
    matched = alerts["matched"].to_numpy(dtype=bool) if len(alerts) else np.zeros(0, dtype=bool)
    detected = events["detected"].to_numpy(dtype=bool) if len(events) else np.zeros(0, dtype=bool)
    duration = (alerts["end_ts"] - alerts["start_ts"]).to_numpy(dtype=float) if len(alerts) else np.zeros(0)

    row_blocks = blocks(scorer.ts)
    rows_per_block = np.bincount(row_blocks, minlength=n_blocks).astype(float)
    hours = rows_per_block / rows_per_block.sum() * scorer.hours      # exposure apportioned by rows

    n_alerts = np.bincount(al_b, minlength=n_blocks).astype(float)
    n_matched = np.bincount(al_b[matched], minlength=n_blocks).astype(float)
    return {
        "eligible": np.bincount(ev_b, minlength=n_blocks).astype(float),
        "detected": np.bincount(ev_b[detected], minlength=n_blocks).astype(float),
        "alerts": n_alerts,
        "matched": n_matched,
        "false_alarms": n_alerts - n_matched,
        "active_s": np.bincount(al_b, weights=duration, minlength=n_blocks).astype(float),
        "hours": hours,
    }


def metrics_from_sums(s: dict) -> dict:
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "event_detection_rate_pct": 100.0 * s["detected"] / s["eligible"],
            "false_alarms_per_router_hour": s["false_alarms"] / s["hours"],
            "alert_precision_pct": 100.0 * s["matched"] / s["alerts"],
            "alert_active_time_pct": 100.0 * s["active_s"] / (s["hours"] * 3600.0),
        }


class WeightedAP:
    """Average precision (as sklearn defines it) for block-resampled rows via integer row weights."""

    def __init__(self, y: np.ndarray, prob: np.ndarray, row_blocks: np.ndarray):
        order = np.argsort(-prob, kind="stable")
        self.y = np.asarray(y, dtype=float)[order]
        self.blocks = np.asarray(row_blocks)[order]
        sorted_prob = np.asarray(prob, dtype=float)[order]
        change = np.flatnonzero(np.diff(sorted_prob) != 0)
        self.ends = np.append(change, len(sorted_prob) - 1)

    def __call__(self, block_counts: np.ndarray) -> float:
        w = block_counts[self.blocks]
        tp = np.cumsum(w * self.y)[self.ends]
        predicted = np.cumsum(w)[self.ends]
        positives = tp[-1]
        if positives <= 0:
            return float("nan")
        delta_tp = np.diff(tp, prepend=0.0)
        precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
        return float(np.sum(delta_tp / positives * precision))


# ------------------------------------------------------------------------- data classes
class System:
    """One system under comparison: rows sorted by (host, ts_unix) with columns host, ts_unix,
    actual_onset; `prob` aligned with the rows (None for the rule-based system); `warn` maps each
    false-alarm budget to the boolean warning decisions at its frozen operating point."""

    def __init__(self, name: str, rows: pd.DataFrame, prob, warn: dict):
        self.name, self.rows, self.prob, self.warn = name, rows, prob, warn


def sorted_frame(df: pd.DataFrame, prob: np.ndarray | None):
    df = df.reset_index(drop=True).copy()
    df["host"] = df["host"].astype(str)
    df["ts_unix"] = df["ts_unix"].astype("int64")
    df["actual_onset"] = df["actual_onset"].astype("int8")
    order = df.sort_values(["host", "ts_unix"], kind="stable").index.to_numpy()
    df = df.loc[order].reset_index(drop=True)
    return df, (None if prob is None else np.asarray(prob, dtype=float)[order])


def threshold_lookup(table: pd.DataFrame, keys: dict) -> float:
    sel = table
    for column, value in keys.items():
        sel = sel[sel[column] == value]
    if sel.empty:
        raise KeyError(f"No stored operating point for {keys}")
    return float(sel["threshold"].iloc[0])


# ------------------------------------------------------------------------- the engine
def process_group(analysis, horizon, test_set, systems, references, budgets, events, hours, args, store):
    dataset = TEST_SET_DATASET[test_set][0]
    block_seconds = max(int(round(args.block_minutes * 60)), 1)

    scorers: dict[tuple, object] = {}

    def scorer_for(system: System):
        key = (len(system.rows), hash(system.rows["host"].to_numpy().tobytes()),
               hash(system.rows["ts_unix"].to_numpy().tobytes()))
        if key not in scorers:
            scorers[key] = fb.Scorer(system.rows[["host", "ts_unix", "actual_onset"]].assign(dataset=dataset),
                                     events, dataset, horizon, hours)
        return scorers[key]

    base_ts = scorer_for(systems[0]).ts
    t0 = int(base_ts.min())
    n_blocks = int((base_ts.max() - t0) // block_seconds) + 1
    seed_seq = np.random.SeedSequence([args.seed, horizon, sorted(TEST_SET_DATASET).index(test_set)])
    rng = np.random.default_rng(seed_seq)
    draws = rng.integers(0, n_blocks, size=(args.n_boot, n_blocks))
    counts = np.zeros((args.n_boot, n_blocks))
    np.add.at(counts, (np.repeat(np.arange(args.n_boot), n_blocks), draws.ravel()), 1.0)

    point, boot, totals = {}, {}, []
    for system in systems:
        scorer = scorer_for(system)
        row_blocks = np.clip((scorer.ts - t0) // block_seconds, 0, n_blocks - 1).astype(np.int64)
        for budget in budgets:
            stats = block_stats(scorer, system.warn[budget], t0, block_seconds, n_blocks)
            sums_point = {k: v.sum() for k, v in stats.items()}
            sums_boot = {k: counts @ v for k, v in stats.items()}
            m_point, m_boot = metrics_from_sums(sums_point), metrics_from_sums(sums_boot)
            for metric in EVENT_METRICS:
                point[(system.name, budget, metric)] = float(m_point[metric])
                boot[(system.name, budget, metric)] = m_boot[metric]
            totals.append({"system": system.name, "horizon_seconds": horizon, "budget": budget,
                           "test_set": test_set, "detected_events": int(sums_point["detected"]),
                           "eligible_events": int(sums_point["eligible"]),
                           "false_alarms": int(sums_point["false_alarms"])})
        if system.prob is not None:
            ap = WeightedAP(system.rows["actual_onset"].to_numpy(), system.prob, row_blocks)
            point[(system.name, np.nan, "pr_auc")] = ap(np.ones(n_blocks))
            boot[(system.name, np.nan, "pr_auc")] = np.array([ap(c) for c in counts])

    alpha = (100.0 - args.ci) / 2.0
    for (name, budget, metric), estimate in point.items():
        samples = boot[(name, budget, metric)]
        lo, hi = np.nanpercentile(samples, [alpha, 100.0 - alpha])
        store["summary"].append({
            "analysis": analysis, "system": name, "horizon_seconds": horizon, "budget_fa_per_router_hour": budget,
            "test_set": test_set, "metric": metric, "estimate": estimate, "ci_low": lo, "ci_high": hi,
            "n_blocks": n_blocks, "n_boot": args.n_boot, "block_minutes": args.block_minutes})
    for name, reference in references:
        for (sys_name, budget, metric), estimate in point.items():
            if sys_name != name or (reference, budget, metric) not in point:
                continue
            diff_boot = boot[(name, budget, metric)] - boot[(reference, budget, metric)]
            lo, hi = np.nanpercentile(diff_boot, [alpha, 100.0 - alpha])
            store["differences"].append({
                "analysis": analysis, "system": name, "reference": reference, "horizon_seconds": horizon,
                "budget_fa_per_router_hour": budget, "test_set": test_set, "metric": metric,
                "difference": estimate - point[(reference, budget, metric)], "ci_low": lo, "ci_high": hi,
                "ci_excludes_zero": bool(lo > 0 or hi < 0), "n_blocks": n_blocks, "n_boot": args.n_boot})
    store["totals"][analysis].extend(totals)


# ----------------------------------------------------------------------------- analyses
def analysis_models(args, events, hours_test, store):
    ops_path = TAB_DIR / "fixed_budget_selected_operating_points.csv"
    if not ops_path.exists():
        print("[skip] models: run 13_fixed_budget_thresholds.py first.")
        return
    ops = pd.read_csv(ops_path)
    predictions = {}
    for prefix, display in ML_MODELS:
        path = TAB_DIR / f"{prefix}_onset_predictions.csv"
        if path.exists():
            predictions[display] = pd.read_csv(path)
    rule_data = {ds: load_model_ready(ds, exclude_idle_router=True) for ds in DATASETS}
    reference = "XGBoost"

    for horizon in args.horizons:
        target = f"y_onset_h{horizon}"
        for dataset in DATASETS:
            systems = []
            for display, pred in predictions.items():
                part = pred[(pred["dataset"] == dataset) & (pred["horizon_seconds"] == horizon)]
                rows, prob = sorted_frame(part[["host", "ts_unix", "actual_onset"]], part["predicted_probability"].to_numpy())
                warn = {b: prob >= threshold_lookup(ops, {"model": display, "horizon_seconds": horizon,
                                                          "budget_fa_per_router_hour": b}) for b in args.budgets}
                systems.append(System(display, rows, prob, warn))
            part = rule_data[dataset]
            part = part[part["split"] == "test"] if dataset == rule.TRAIN_DATASET else part[part["split"] != "purge"]
            elig = rule.eligible_rows(part, target)
            elig = elig.assign(host=elig["host"].astype(str)).sort_values(
                ["host", "ts_unix"], kind="stable").reset_index(drop=True)
            rows, _ = sorted_frame(elig[["host", "ts_unix"]].assign(actual_onset=elig[target].astype("int8")), None)
            warn = {}
            for b in args.budgets:
                sel = ops[(ops["model"] == "RuleBased") & (ops["horizon_seconds"] == horizon)
                          & (ops["budget_fa_per_router_hour"] == b)].iloc[0]
                rise = None if pd.isna(sel["rise_threshold_mbps_per_10s"]) else float(sel["rise_threshold_mbps_per_10s"])
                warn[b] = rule.predict_rule(elig, float(sel["bw_threshold_mbps"]), rise).astype(bool)
            systems.append(System("RuleBased", rows, None, warn))
            if not any(s.name == reference for s in systems):
                continue
            refs = [(s.name, reference) for s in systems if s.name != reference]
            print(f"  models  H={horizon}s  {dataset}: {len(systems)} systems", flush=True)
            process_group("models", horizon, dataset, systems, refs, args.budgets, events,
                          hours_test[dataset], args, store)


def analysis_ablation(args, events, hours_test, store):
    pred_path, res_path = TAB_DIR / "feature_ablation_test_predictions.parquet", TAB_DIR / "feature_ablation_results.csv"
    if not (pred_path.exists() and res_path.exists()):
        print("[skip] ablation: rerun 14_feature_ablation.py (it now saves test probabilities).")
        return
    pred, res = pd.read_parquet(pred_path), pd.read_csv(res_path)
    configs = [c[len("prob__"):] for c in pred.columns if c.startswith("prob__")]
    for horizon in args.horizons:
        for dataset in DATASETS:
            part = pred[(pred["horizon_seconds"] == horizon) & (pred["dataset"] == dataset)]
            if part.empty:
                continue
            systems = []
            for config in configs:
                rows, prob = sorted_frame(part[["host", "ts_unix", "actual_onset"]], part[f"prob__{config}"].to_numpy())
                warn = {b: prob >= threshold_lookup(res, {"config": config, "horizon_seconds": horizon,
                                                          "budget_fa_per_router_hour": b}) for b in args.budgets}
                systems.append(System(config, rows, prob, warn))
            refs = [(c, "D_all") for c in configs if c != "D_all"]
            print(f"  ablation  H={horizon}s  {dataset}: {len(systems)} systems", flush=True)
            process_group("ablation", horizon, dataset, systems, refs, args.budgets, events,
                          hours_test[dataset], args, store)


def analysis_experiment_d(args, events, hours_test, store):
    pred_path, res_path = TAB_DIR / "experiment_d_test_predictions.parquet", TAB_DIR / "experiment_d_results.csv"
    if not (pred_path.exists() and res_path.exists()):
        print("[skip] experiment_d: rerun 15_experiment_d_multi_scenario.py (it now saves test probabilities).")
        return
    pred, res = pd.read_parquet(pred_path), pd.read_csv(res_path)
    columns = [c[len("prob__"):] for c in pred.columns if c.startswith("prob__")]
    configs = sorted({c.split("__")[0] for c in columns})
    for horizon in args.horizons:
        for test_set in ("D1_test", "D2_test", "D3"):
            part = pred[(pred["horizon_seconds"] == horizon) & (pred["test_set"] == test_set)]
            if part.empty:
                continue
            dataset, split = TEST_SET_DATASET[test_set]
            hours = fb.router_hours(dataset, split)
            systems = []
            for column in columns:
                config, regime = column.split("__")
                rows, prob = sorted_frame(part[["host", "ts_unix", "actual_onset"]], part[f"prob__{column}"].to_numpy())
                warn = {b: prob >= threshold_lookup(res, {"config": config, "regime": regime, "horizon_seconds": horizon,
                                                          "budget_fa_per_router_hour": b, "test_set": test_set})
                        for b in args.budgets}
                systems.append(System(column, rows, prob, warn))
            refs = [(f"{c}__train_D1_D2", f"{c}__train_D1") for c in configs]
            print(f"  experiment_d  H={horizon}s  {test_set}: {len(systems)} systems", flush=True)
            process_group("experiment_d", horizon, test_set, systems, refs, args.budgets, events, hours, args, store)


# ---------------------------------------------------------------------- verification
def verify_totals(store) -> None:
    sources = {
        "models": (TAB_DIR / "fixed_budget_event_level_summary.csv", "model", "dataset"),
        "ablation": (TAB_DIR / "feature_ablation_results.csv", "config", "dataset"),
        "experiment_d": (TAB_DIR / "experiment_d_results.csv", None, "test_set"),
    }
    for analysis, rows in store["totals"].items():
        if not rows:
            continue
        path, name_col, set_col = sources[analysis]
        ours = pd.DataFrame(rows)
        ref = pd.read_csv(path)
        if analysis == "models":
            ref = ref[ref["operating_point"].str.startswith("budget_")].rename(columns={"model": "system"})
        elif analysis == "ablation":
            ref = ref.rename(columns={"config": "system"})
        else:
            ref["system"] = ref["config"] + "__" + ref["regime"]
        ref = ref.rename(columns={"budget_fa_per_router_hour": "budget", set_col: "test_set"})
        merged = ours.merge(ref, on=["system", "horizon_seconds", "budget", "test_set"], suffixes=("", "_src"))
        if merged.empty:
            print(f"[verify] {analysis}: no overlapping rows with the original results.")
            continue
        d_det = (merged["detected_events"] - merged["detected_events_src"]).abs().max()
        d_fa = (merged["false_alarms"] - merged["false_alarms_src"]).abs().max()
        status = "OK" if d_det == 0 and d_fa == 0 else "MISMATCH"
        print(f"[verify] {analysis}: {len(merged)} rows, max |detected diff| = {d_det}, max |false alarms diff| = {d_fa} -> {status}")
        if status != "OK":
            raise RuntimeError(f"{analysis}: bootstrap point estimates differ from the original results.")


# --------------------------------------------------------------------------- reporting
def fmt(est, lo, hi, digits=1):
    return f"{est:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def print_report(summary: pd.DataFrame, diffs: pd.DataFrame, args) -> None:
    for analysis in summary["analysis"].unique():
        sub = summary[summary["analysis"] == analysis]
        print(f"\n{'=' * 90}\n{analysis.upper()}: estimates with {args.ci:g}% time-block bootstrap intervals\n{'=' * 90}")
        for horizon in args.horizons:
            for test_set in sub["test_set"].unique():
                cell = sub[(sub["horizon_seconds"] == horizon) & (sub["test_set"] == test_set)]
                if cell.empty:
                    continue
                pr = cell[cell["metric"] == "pr_auc"]
                print(f"\n-- H = {horizon} s, test set {test_set} --")
                table = pd.DataFrame(index=pd.Index(cell["system"].drop_duplicates()))
                table["PR-AUC"] = pd.Series(
                    [fmt(r.estimate, r.ci_low, r.ci_high, 3) for r in pr.itertuples()], index=pr["system"])
                for metric, label, digits in (("event_detection_rate_pct", "detect %", 1),
                                              ("false_alarms_per_router_hour", "FA/hr", 2)):
                    for budget in args.budgets:
                        m = cell[(cell["metric"] == metric) & (cell["budget_fa_per_router_hour"] == budget)]
                        table[f"{label} @b{budget:g}"] = pd.Series(
                            [fmt(r.estimate, r.ci_low, r.ci_high, digits) for r in m.itertuples()], index=m["system"])
                print(table.fillna("-").to_string())

    if len(diffs):
        print(f"\n{'=' * 90}\nPAIRED DIFFERENCES (system minus reference); * = interval excludes 0\n{'=' * 90}")
        for analysis in diffs["analysis"].unique():
            d = diffs[(diffs["analysis"] == analysis) & (diffs["metric"].isin(["pr_auc", "event_detection_rate_pct"]))]
            for horizon in args.horizons:
                for test_set in d["test_set"].unique():
                    cell = d[(d["horizon_seconds"] == horizon) & (d["test_set"] == test_set)]
                    if cell.empty:
                        continue
                    ref = cell["reference"].iloc[0] if cell["reference"].nunique() == 1 else "per-config baseline"
                    print(f"\n{analysis}  H = {horizon} s  {test_set}   (reference: {ref})")
                    rows = {}
                    for r in cell.itertuples():
                        key = r.system
                        col = "PR-AUC" if r.metric == "pr_auc" else f"detect pts @b{r.budget_fa_per_router_hour:g}"
                        digits = 3 if r.metric == "pr_auc" else 1
                        rows.setdefault(key, {})[col] = fmt(r.difference, r.ci_low, r.ci_high, digits) + (
                            "*" if r.ci_excludes_zero else "")
                    print(pd.DataFrame(rows).T.fillna("-").to_string())
    print("\nCaution: false-alarm and detection differences between systems are only meaningful together, "
          "because each system sits at its own realized false-alarm rate; PR-AUC differences do not have that problem.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-block bootstrap confidence intervals.")
    parser.add_argument("--analysis", nargs="+", choices=["models", "ablation", "experiment_d"],
                        default=["models", "ablation", "experiment_d"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--budgets", type=float, nargs="+", default=[1.0, 3.0])
    parser.add_argument("--block-minutes", type=float, default=10.0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("\nTIME-BLOCK BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 78)
    print(f"block = {args.block_minutes:g} min (all routers together), {args.n_boot} resamples, "
          f"{args.ci:g}% percentile intervals, seed {args.seed}")
    events = fb.load_events()
    hours_test = {ds: ev.calculate_monitoring_hours(ds) for ds in DATASETS}
    store = {"summary": [], "differences": [], "totals": {"models": [], "ablation": [], "experiment_d": []}}

    runners = {"models": analysis_models, "ablation": analysis_ablation, "experiment_d": analysis_experiment_d}
    for name in args.analysis:
        runners[name](args, events, hours_test, store)
    if not store["summary"]:
        raise SystemExit("Nothing was analysed. Run scripts 13, 14 and 15 (updated versions) first.")

    summary, diffs = pd.DataFrame(store["summary"]), pd.DataFrame(store["differences"])
    summary.to_csv(TAB_DIR / "bootstrap_summary.csv", index=False)
    diffs.to_csv(TAB_DIR / "bootstrap_differences.csv", index=False)
    verify_totals(store)
    print_report(summary, diffs, args)
    print("\nSaved: bootstrap_summary.csv, bootstrap_differences.csv")


if __name__ == "__main__":
    main()

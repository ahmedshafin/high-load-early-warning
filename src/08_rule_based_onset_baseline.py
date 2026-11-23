"""Rule-based baseline for LABEL=4 event-onset prediction.

This is the first M4 baseline for the improved CSE552 outline.

Rule (features available at prediction time t only):
    warn if bw_max >= B
       OR bw_max_diff1 >= R (when a trend threshold is enabled)

B and R are selected on D1's validation split only. Candidate trend thresholds
are derived from D1's training split only. The selected rule is then evaluated
on D1's untouched test split and on D2/D3 (excluding their purge rows).

Rows with y_onset_h{H} == -1 are ineligible and excluded from both tuning and
metric calculation. The script does not use LABEL, current/future targets, or
any other target-derived column as model input.

Run from the repository root:
    python src/08_rule_based_onset_baseline.py

Outputs:
    reports/tables/rule_based_onset_results.csv
    reports/tables/rule_based_onset_selected_rules.csv
    reports/tables/rule_based_onset_predictions.csv   (row-level warnings, for event-level evaluation)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from config import DATASETS, HORIZONS_SEC, MODEL_DIR, TAB_DIR  # noqa: E402
from modeling_utils import load_model_ready  # noqa: E402

# Domain-informed candidates for current bandwidth immediately before an event.
# LABEL=4 begins at >=90 Mbps, so thresholds remain below that boundary.
BW_THRESHOLD_CANDIDATES = [30.0, 40.0, 50.0, 60.0, 70.0, 75.0, 80.0, 85.0, 89.0]
TREND_QUANTILES = [0.50, 0.65, 0.75, 0.85, 0.90, 0.95]
FEATURES = ["bw_max", "bw_max_diff1"]
TRAIN_DATASET = "D1"
TEST_DATASETS = ["D1", "D2", "D3"]
IDLE_ROUTER = "r7"


def make_rise_thresholds(train_df: pd.DataFrame, target_col: str) -> list[Optional[float]]:
    """Create trend candidates from eligible D1 training rows only."""
    y = pd.to_numeric(train_df[target_col], errors="coerce")
    eligible = train_df.loc[y.isin([0, 1]), "bw_max_diff1"]
    positive_rises = pd.to_numeric(eligible, errors="coerce")
    positive_rises = positive_rises[np.isfinite(positive_rises) & (positive_rises > 0)]

    thresholds: list[Optional[float]] = [None]  # None disables the trend clause.
    if not positive_rises.empty:
        thresholds.extend(
            float(positive_rises.quantile(q))
            for q in TREND_QUANTILES
        )

    # Remove duplicate quantiles (common with discretized telemetry).
    unique: list[Optional[float]] = []
    seen = set()
    for value in thresholds:
        key = None if value is None else round(value, 8)
        if key not in seen:
            unique.append(value)
            seen.add(key)
    return unique


def predict_rule(df: pd.DataFrame, bw_threshold: float,
                 rise_threshold: Optional[float]) -> np.ndarray:
    """Return 0/1 warning decisions from current bandwidth and its 1-step rise."""
    bw = pd.to_numeric(df["bw_max"], errors="coerce").to_numpy(dtype=float)
    rise = pd.to_numeric(df["bw_max_diff1"], errors="coerce").to_numpy(dtype=float)
    prediction = bw >= bw_threshold
    if rise_threshold is not None:
        prediction |= rise >= rise_threshold
    return prediction.astype(np.int8)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Sample-level metrics for a deterministic rule; no probability AUC is reported."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_eligible": int(len(y_true)),
        "positive_prevalence_pct": round(float(y_true.mean() * 100), 4) if len(y_true) else np.nan,
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)), 4) if (fp + tn) else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "predicted_warning_rows": int(y_pred.sum()),
    }


def eligible_rows(df: pd.DataFrame, target_col: str, split: Optional[str] = None) -> pd.DataFrame:
    """Filter unknown/ineligible targets and optional chronological split."""
    part = df
    if split is not None:
        part = part.loc[part["split"].eq(split)]
    elif "split" in part.columns:
        part = part.loc[~part["split"].eq("purge")]

    y = pd.to_numeric(part[target_col], errors="coerce")
    valid = y.isin([0, 1])
    for col in FEATURES:
        x = pd.to_numeric(part[col], errors="coerce")
        valid &= np.isfinite(x)
    return part.loc[valid].copy()


def choose_rule(train_df: pd.DataFrame, val_df: pd.DataFrame,
                target_col: str) -> tuple[dict, dict]:
    """Tune rule thresholds on D1 validation rows; derive candidates from D1 train."""
    if train_df.empty:
        raise ValueError(f"No eligible D1 training rows for {target_col}")
    if val_df.empty:
        raise ValueError(f"No eligible D1 validation rows for {target_col}")

    rise_candidates = make_rise_thresholds(train_df, target_col)
    y_val = val_df[target_col].astype("int8").to_numpy()
    candidates = []

    for bw_threshold in BW_THRESHOLD_CANDIDATES:
        for rise_threshold in rise_candidates:
            pred = predict_rule(val_df, bw_threshold, rise_threshold)
            metrics = classification_metrics(y_val, pred)
            candidates.append({
                "bw_threshold_mbps": float(bw_threshold),
                "rise_threshold_mbps_per_10s": (
                    np.nan if rise_threshold is None else float(rise_threshold)
                ),
                "trend_clause_enabled": rise_threshold is not None,
                **metrics,
            })

    # Maximize validation F1; ties prefer lower FPR, higher precision, then fewer alerts.
    best = max(
        candidates,
        key=lambda row: (
            row["f1"],
            -(row["false_positive_rate"] if np.isfinite(row["false_positive_rate"]) else 1.0),
            row["precision"],
            -row["predicted_warning_rows"],
        ),
    )
    selected = {
        "bw_threshold": best["bw_threshold_mbps"],
        "rise_threshold": (
            None if not best["trend_clause_enabled"]
            else best["rise_threshold_mbps_per_10s"]
        ),
    }
    return selected, best


def evaluate_split(df: pd.DataFrame, target_col: str, selected: dict) -> dict:
    eligible = eligible_rows(df, target_col)
    if eligible.empty:
        return {"n_eligible": 0, "positive_prevalence_pct": np.nan,
                "precision": np.nan, "recall": np.nan, "f1": np.nan,
                "false_positive_rate": np.nan, "tn": 0, "fp": 0,
                "fn": 0, "tp": 0, "predicted_warning_rows": 0}
    y = eligible[target_col].astype("int8").to_numpy()
    pred = predict_rule(eligible, selected["bw_threshold"], selected["rise_threshold"])
    return classification_metrics(y, pred)


def main() -> None:
    print("\nRULE-BASED EVENT-ONSET BASELINE")
    print("=" * 76)
    print(f"Training/tuning scenario: {TRAIN_DATASET}")
    print("Rule inputs: bw_max and bw_max_diff1 (current/past telemetry only)")
    print("Tuning: D1 train-derived candidate values; D1 validation selects thresholds")
    print("Evaluation: D1 test, D2 non-purge, D3 non-purge")
    print("Ineligible target sentinel -1 is excluded")

    datasets = {}
    for ds in TEST_DATASETS:
        path = MODEL_DIR / f"{ds}_model_ready.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run python src/02_build_features.py first."
            )
        d = load_model_ready(ds, exclude_idle_router=True)
        missing = [c for c in FEATURES + ["split"] if c not in d.columns]
        if missing:
            raise KeyError(f"{ds} model-ready table is missing columns: {missing}")
        datasets[ds] = d
        print(f"Loaded {ds}: {len(d):,} rows; hosts={d['host'].nunique()}")

    d1 = datasets[TRAIN_DATASET]
    result_rows = []
    selected_rows = []
    prediction_frames = []

    for horizon in HORIZONS_SEC:
        target_col = f"y_onset_h{horizon}"
        if target_col not in d1.columns:
            raise KeyError(f"Missing target {target_col}; rebuild model-ready tables.")

        train = eligible_rows(d1, target_col, split="train")
        val = eligible_rows(d1, target_col, split="val")
        selected, val_metrics = choose_rule(train, val, target_col)

        selected_row = {
            "target": target_col,
            "horizon_seconds": horizon,
            "tuning_dataset": TRAIN_DATASET,
            "selected_bw_threshold_mbps": selected["bw_threshold"],
            "selected_rise_threshold_mbps_per_10s": (
                np.nan if selected["rise_threshold"] is None else selected["rise_threshold"]
            ),
            "trend_clause_enabled": selected["rise_threshold"] is not None,
            "validation_n": val_metrics["n_eligible"],
            "validation_prevalence_pct": val_metrics["positive_prevalence_pct"],
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "validation_fpr": val_metrics["false_positive_rate"],
        }
        selected_rows.append(selected_row)

        for test_ds in TEST_DATASETS:
            test_data = datasets[test_ds]
            if test_ds == TRAIN_DATASET:
                test = test_data.loc[test_data["split"].eq("test")]
                test_definition = "chronological_test_split"
            else:
                test = test_data.loc[~test_data["split"].eq("purge")]
                test_definition = "all_non_purge_rows_in_scenario"

            metrics = evaluate_split(test, target_col, selected)

            # Save row-level warnings so 11_event_level_evaluation.py can score this rule
            # exactly like the ML models (same eligible rows as the sample-level metrics).
            elig = eligible_rows(test, target_col)
            frame = elig[["dataset", "host", "ts_unix", "time", "label"]].copy()
            frame["horizon_seconds"] = horizon
            frame["target_column"] = target_col
            frame["actual_onset"] = elig[target_col].astype("int8").to_numpy()
            frame["predicted_probability"] = np.nan          # a rule has no probability
            frame["predicted_warning"] = predict_rule(
                elig, selected["bw_threshold"], selected["rise_threshold"])
            frame["selected_probability_threshold"] = np.nan
            frame["test_definition"] = test_definition
            frame["model"] = "RuleBased"
            prediction_frames.append(frame)
            result_rows.append({
                "target": target_col,
                "horizon_seconds": horizon,
                "train_dataset": TRAIN_DATASET,
                "validation_dataset": TRAIN_DATASET,
                "test_dataset": test_ds,
                "experiment": (
                    "same_scenario" if test_ds == TRAIN_DATASET
                    else "cross_scenario"
                ),
                "test_definition": test_definition,
                "bw_threshold_mbps": selected["bw_threshold"],
                "rise_threshold_mbps_per_10s": (
                    np.nan if selected["rise_threshold"] is None else selected["rise_threshold"]
                ),
                "trend_clause_enabled": selected["rise_threshold"] is not None,
                **metrics,
            })

            print(
                f"H={horizon:2d}s | D1->{test_ds} | "
                f"P={metrics['precision']:.3f} "
                f"R={metrics['recall']:.3f} "
                f"F1={metrics['f1']:.3f} "
                f"FPR={metrics['false_positive_rate']:.3f} "
                f"N={metrics['n_eligible']:,}"
            )

    results = pd.DataFrame(result_rows)
    selected = pd.DataFrame(selected_rows)
    results_path = TAB_DIR / "rule_based_onset_results.csv"
    selected_path = TAB_DIR / "rule_based_onset_selected_rules.csv"
    results.to_csv(results_path, index=False)
    selected.to_csv(selected_path, index=False)
    predictions_path = TAB_DIR / "rule_based_onset_predictions.csv"
    pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False)

    print("\nSelected thresholds (chosen using D1 validation only):")
    print(selected[[
        "target", "selected_bw_threshold_mbps",
        "selected_rise_threshold_mbps_per_10s", "trend_clause_enabled",
        "validation_f1", "validation_fpr"
    ]].to_string(index=False))
    print(f"\nSaved: {results_path}")
    print(f"Saved: {selected_path}")
    print(f"Saved: {predictions_path}")
    print("\nRule-based event-onset baseline completed.")


if __name__ == "__main__":
    main()

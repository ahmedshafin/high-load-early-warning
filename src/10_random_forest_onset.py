"""Random Forest baseline for LABEL=4 event-onset prediction.

Protocol
--------
* Fit one horizon-specific Random Forest on D1's chronological TRAIN rows.
* Fit median imputation on D1 TRAIN only; Random Forest does not need scaling.
* Select the probability threshold using D1 VALIDATION only (maximize F1).
* Evaluate once on D1 TEST and on all non-purge eligible rows of D2 and D3.
* Exclude onset target sentinel -1; exclude idle router r7, matching the LR run.
* Exclude features derived from current/historical LABEL state.

Run from repository root:
    python src/10_random_forest_onset.py

Outputs
-------
reports/tables/random_forest_onset_results.csv
reports/tables/random_forest_onset_selected_thresholds.csv
reports/tables/random_forest_onset_feature_audit.csv
reports/tables/random_forest_onset_feature_importance.csv
reports/tables/random_forest_onset_predictions.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import HORIZONS_SEC, MODEL_DIR, SEED, TAB_DIR  # noqa: E402
from features import feature_groups  # noqa: E402
from modeling_utils import load_model_ready  # noqa: E402

TRAIN_DATASET = "D1"
TEST_DATASETS = ("D1", "D2", "D3")
IDLE_ROUTER = "r7"

# Keep the original project's Random Forest settings for continuity.
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": SEED,
}

LABEL_DERIVED_EXCLUSIONS = {
    "run_len_ge3",
    "run_len_ge4",
    "steps_since_ge3",
    "steps_since_ge4",
    "ctx_net_n_ge4",
}


def select_telemetry_features(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Use the same 56 telemetry feature selection policy as Logistic Regression."""
    feature_to_group: dict[str, str] = {}
    for group_name, columns in feature_groups(df).items():
        for col in columns:
            feature_to_group.setdefault(col, group_name)

    included: list[str] = []
    audit_rows: list[dict] = []
    for col, group_name in feature_to_group.items():
        keep = col in df.columns
        reason = "Included: telemetry or telemetry-derived feature available at prediction time."
        if not keep:
            reason = "Excluded: configured feature is absent from the model-ready table."
        elif col in LABEL_DERIVED_EXCLUSIONS:
            keep = False
            reason = "Excluded: derived from current/historical LABEL state, not raw telemetry."
        elif col.startswith("n_ge4_last"):
            keep = False
            reason = "Excluded: historical rolling count derived from LABEL=4 state."
        elif col in {"label", "cur_ge3", "cur_ge4"} or col.startswith("y_"):
            keep = False
            reason = "Excluded: label/current-state/target column is not a model input."

        if keep:
            included.append(col)
        audit_rows.append({
            "feature": col,
            "feature_group": group_name,
            "included_in_random_forest": keep,
            "reason": reason,
        })

    if not included:
        raise ValueError("No usable telemetry features were found.")
    return included, pd.DataFrame(audit_rows)


def partition_xy(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    split: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return eligible binary target rows; -1 is never treated as class 0."""
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    part = df
    if split is not None:
        part = part.loc[part["split"].eq(split)]
    elif "split" in part.columns:
        part = part.loc[~part["split"].eq("purge")]

    y = pd.to_numeric(part[target_col], errors="coerce")
    mask = y.isin([0, 1])
    X = part.loc[mask, feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y_array = y.loc[mask].astype("int8").to_numpy()

    if len(X) == 0:
        raise ValueError(f"No eligible rows for {target_col} (split={split!r}).")
    if np.unique(y_array).size != 2:
        raise ValueError(
            f"Both classes are required for {target_col} (split={split!r}); "
            f"found {np.unique(y_array).tolist()}."
        )
    return X, y_array


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute sample-level metrics; AUC values use predicted probabilities."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    if np.unique(y_true).size == 2:
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
    else:
        pr_auc = float("nan")
        roc_auc = float("nan")

    return {
        "n_eligible": int(len(y_true)),
        "positive_prevalence_pct": round(float(y_true.mean() * 100), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)), 4) if (fp + tn) else np.nan,
        "pr_auc_average_precision": round(pr_auc, 4) if np.isfinite(pr_auc) else np.nan,
        "roc_auc": round(roc_auc, 4) if np.isfinite(roc_auc) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_warning_rows": int(y_pred.sum()),
    }


def choose_probability_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, dict]:
    """Choose threshold on D1 validation only; tie-break by lower FPR/alerts."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=float)
    quantiles = np.quantile(y_prob, np.linspace(0.01, 0.99, 199))
    regular_grid = np.linspace(0.05, 0.95, 181)
    candidates = np.unique(np.concatenate(([0.0, 1.0], quantiles, regular_grid)))

    best_threshold: float | None = None
    best_key: tuple | None = None
    for threshold in candidates:
        pred = (y_prob >= threshold).astype(np.int8)
        tp = int(np.sum((y_true == 1) & (pred == 1)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        fn = int(np.sum((y_true == 1) & (pred == 0)))
        tn = int(np.sum((y_true == 0) & (pred == 0)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        fpr = fp / (fp + tn) if (fp + tn) else 1.0
        key = (f1, -fpr, precision, -int(pred.sum()), float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)

    if best_threshold is None:
        raise RuntimeError("Could not select a probability threshold.")
    best_pred = (y_prob >= best_threshold).astype(np.int8)
    best_metrics = binary_metrics(y_true, best_pred, y_prob)
    return best_threshold, best_metrics


def prepare_xy(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    split: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    X, y = partition_xy(df, target_col, feature_cols, split=split)
    return X, y


def main() -> None:
    print("\nRANDOM FOREST — EVENT-ONSET BASELINE")
    print("=" * 78)
    print("Training/tuning scenario: D1")
    print(f"Estimator: SimpleImputer(median) + RandomForestClassifier({RF_PARAMS})")
    print("Imputation: fitted on D1 train only; no feature scaling required")
    print("Threshold tuning: D1 validation only; maximize validation F1")
    print("Evaluation: D1 untouched test, D2 non-purge, D3 non-purge")
    print("Target sentinel -1 is excluded; idle router r7 is excluded")
    print("Label-derived input features are excluded")

    datasets: dict[str, pd.DataFrame] = {}
    for ds in TEST_DATASETS:
        path = MODEL_DIR / f"{ds}_model_ready.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run python src/02_build_features.py first."
            )
        datasets[ds] = load_model_ready(ds, exclude_idle_router=True)
        print(f"Loaded {ds}: {len(datasets[ds]):,} rows; hosts={datasets[ds]['host'].nunique()}")

    feature_cols, feature_audit = select_telemetry_features(datasets[TRAIN_DATASET])
    feature_audit_path = TAB_DIR / "random_forest_onset_feature_audit.csv"
    feature_audit.to_csv(feature_audit_path, index=False)
    print(f"\nTelemetry features used: {len(feature_cols)}")
    print("Excluded label-derived inputs:", int((~feature_audit["included_in_random_forest"]).sum()))

    results_rows: list[dict] = []
    threshold_rows: list[dict] = []
    importance_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    d1 = datasets[TRAIN_DATASET]

    for horizon in HORIZONS_SEC:
        target_col = f"y_onset_h{horizon}"
        if target_col not in d1.columns:
            raise KeyError(f"Missing {target_col}. Rebuild with python src/02_build_features.py.")

        print(f"\n{'-' * 68}\nHorizon: {horizon}s | target: {target_col}\n{'-' * 68}")
        X_train, y_train = prepare_xy(d1, target_col, feature_cols, split="train")
        X_val, y_val = prepare_xy(d1, target_col, feature_cols, split="val")

        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)

        model = RandomForestClassifier(**RF_PARAMS)
        start = time.perf_counter()
        model.fit(X_train_imp, y_train)
        fit_seconds = time.perf_counter() - start

        val_prob = model.predict_proba(X_val_imp)[:, 1]
        threshold, val_metrics = choose_probability_threshold(y_val, val_prob)
        threshold_rows.append({
            "target": target_col,
            "horizon_seconds": horizon,
            "train_dataset": TRAIN_DATASET,
            "validation_dataset": TRAIN_DATASET,
            **{f"rf_{k}": v for k, v in RF_PARAMS.items() if k != "n_jobs"},
            "n_jobs": RF_PARAMS["n_jobs"],
            "selected_probability_threshold": threshold,
            "threshold_selection_metric": "validation_F1; ties lower FPR, higher precision, fewer alerts",
            "n_train": int(len(y_train)),
            "n_validation": int(len(y_val)),
            "fit_seconds": round(fit_seconds, 3),
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "validation_fpr": val_metrics["false_positive_rate"],
            "validation_pr_auc_average_precision": val_metrics["pr_auc_average_precision"],
        })
        print(
            f"Selected validation threshold={threshold:.4f} | "
            f"validation F1={val_metrics['f1']:.3f} | "
            f"validation FPR={val_metrics['false_positive_rate']:.3f} | "
            f"fit={fit_seconds:.1f}s"
        )

        # Impurity-based importance is useful as a first look, but correlated
        # features can divide or distort importance. Later M6 adds ablation and
        # validation-based permutation importance.
        for feature, importance in zip(feature_cols, model.feature_importances_):
            group = next((g for g, cols in feature_groups(d1).items() if feature in cols), "other")
            importance_rows.append({
                "target": target_col,
                "horizon_seconds": horizon,
                "feature": feature,
                "feature_group": group,
                "importance_type": "random_forest_impurity_mean_decrease",
                "importance": float(importance),
            })

        for test_ds in TEST_DATASETS:
            test_df = datasets[test_ds]
            if test_ds == TRAIN_DATASET:
                test_split = "test"
                test_definition = "chronological_test_split"
            else:
                test_split = None
                test_definition = "all_non_purge_rows_in_scenario"

            X_test, y_test = prepare_xy(test_df, target_col, feature_cols, split=test_split)
            X_test_imp = imputer.transform(X_test)
            y_prob = model.predict_proba(X_test_imp)[:, 1]
            y_pred = (y_prob >= threshold).astype(np.int8)
            if test_split is not None:
                metadata_part = test_df.loc[
                    test_df["split"].eq(test_split)
                ].copy()
            else:
                metadata_part = test_df.loc[
                    ~test_df["split"].eq("purge")
                ].copy()

            # Apply exactly the same target eligibility condition.
            target_values = pd.to_numeric(
                metadata_part[target_col], errors="coerce"
            )

            eligible_mask = target_values.isin([0, 1])

            # Keep identifiers and actual timestamps.
            prediction_meta = metadata_part.loc[
                eligible_mask,
                ["dataset", "host", "ts_unix", "time", "label"]
            ].copy()

            # Verify that metadata and model predictions have identical ordering.
            metadata_y = (
                target_values.loc[eligible_mask]
                .astype("int8")
                .to_numpy()
            )

            if len(prediction_meta) != len(y_test):
                raise RuntimeError(
                    "Prediction metadata does not align with y_test."
                )

            if not np.array_equal(metadata_y, y_test):
                raise RuntimeError(
                    "Actual targets are not aligned with model predictions."
                )

            # Construct the prediction records.
            prediction_frame = prediction_meta.reset_index(drop=True)

            prediction_frame["horizon_seconds"] = horizon
            prediction_frame["target_column"] = target_col
            prediction_frame["actual_onset"] = y_test
            prediction_frame["predicted_probability"] = y_prob
            prediction_frame["predicted_warning"] = y_pred
            prediction_frame["selected_probability_threshold"] = threshold
            prediction_frame["test_definition"] = test_definition

            prediction_frames.append(prediction_frame)
            metrics = binary_metrics(y_test, y_pred, y_prob)

            results_rows.append({
                "model": "RandomForest",
                "target": target_col,
                "horizon_seconds": horizon,
                "train_dataset": TRAIN_DATASET,
                "validation_dataset": TRAIN_DATASET,
                "test_dataset": test_ds,
                "experiment": "same_scenario" if test_ds == TRAIN_DATASET else "cross_scenario",
                "test_definition": test_definition,
                **{f"rf_{k}": v for k, v in RF_PARAMS.items() if k != "n_jobs"},
                "n_jobs": RF_PARAMS["n_jobs"],
                "selected_probability_threshold": threshold,
                "n_features": len(feature_cols),
                "fit_seconds": round(fit_seconds, 3),
                **metrics,
            })
            print(
                f"D1->{test_ds} | P={metrics['precision']:.3f} "
                f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
                f"FPR={metrics['false_positive_rate']:.3f} "
                f"PR-AUC={metrics['pr_auc_average_precision']:.3f} "
                f"N={metrics['n_eligible']:,}"
            )

    results = pd.DataFrame(results_rows)
    thresholds = pd.DataFrame(threshold_rows)
    importances = pd.DataFrame(importance_rows)
    predictions = pd.concat(
        prediction_frames,
        ignore_index=True
    )
    results_path = TAB_DIR / "random_forest_onset_results.csv"
    thresholds_path = TAB_DIR / "random_forest_onset_selected_thresholds.csv"
    importances_path = TAB_DIR / "random_forest_onset_feature_importance.csv"
    predictions_path = TAB_DIR / "random_forest_onset_predictions.csv"

    results.to_csv(results_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)
    importances.to_csv(importances_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print(f"Saved: {predictions_path}")

    print("\nResults summary — F1 by horizon and test scenario:")
    print(results.pivot(index="horizon_seconds", columns="test_dataset", values="f1").round(3).to_string())
    print(f"\nSaved: {results_path}")
    print(f"Saved: {thresholds_path}")
    print(f"Saved: {feature_audit_path}")
    print(f"Saved: {importances_path}")
    print("\nRandom Forest event-onset baseline completed.")


if __name__ == "__main__":
    main()

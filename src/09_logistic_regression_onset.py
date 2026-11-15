"""Logistic Regression baseline for LABEL=4 event-onset prediction.

Protocol
--------
* Train each horizon-specific model on D1's chronological TRAIN split only.
* Fit median imputation and standardization on D1 TRAIN only.
* Use D1 VALIDATION only to choose the probability decision threshold.
* Report on D1's untouched TEST split and on all non-purge rows of D2 and D3.
* Exclude y_onset == -1 rows from fitting, threshold selection, and evaluation.
* Exclude features derived from LABEL/current label states (for example,
  run_len_ge4, n_ge4_last30s, and ctx_net_n_ge4) to keep inputs telemetry-based.

Run from the repository root:
    python src/09_logistic_regression_onset.py

Outputs
-------
reports/tables/logistic_regression_onset_results.csv
reports/tables/logistic_regression_onset_selected_thresholds.csv
reports/tables/logistic_regression_onset_feature_audit.csv
reports/tables/logistic_regression_onset_coefficients.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import HORIZONS_SEC, MODEL_DIR, SEED, TAB_DIR  # noqa: E402
from features import feature_groups  # noqa: E402
from modeling_utils import load_model_ready  # noqa: E402

TRAIN_DATASET = "D1"
TEST_DATASETS = ("D1", "D2", "D3")
IDLE_ROUTER = "r7"
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 3000,
    "solver": "lbfgs",
    "random_state": SEED,
}

# These columns are deterministic functions of current/past LABEL values,
# not direct SNMP/network telemetry. Do not use them as model inputs.
LABEL_DERIVED_EXCLUSIONS = {
    "run_len_ge3",
    "run_len_ge4",
    "steps_since_ge3",
    "steps_since_ge4",
    "ctx_net_n_ge4",
}


def select_telemetry_features(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Return safe feature names and an inclusion/exclusion audit table."""
    groups = feature_groups(df)
    feature_to_group: dict[str, str] = {}
    for group_name, columns in groups.items():
        for col in columns:
            feature_to_group.setdefault(col, group_name)

    included: list[str] = []
    audit_rows: list[dict] = []

    for col, group_name in feature_to_group.items():
        reason = "Included: telemetry or telemetry-derived feature available at prediction time."
        keep = True
        if col in LABEL_DERIVED_EXCLUSIONS:
            keep = False
            reason = "Excluded: derived from current/historical LABEL state, not raw telemetry."
        elif col.startswith("n_ge4_last"):
            keep = False
            reason = "Excluded: historical rolling count derived from LABEL=4 state."
        elif col in {"label", "cur_ge3", "cur_ge4"} or col.startswith("y_"):
            keep = False
            reason = "Excluded: label/current-state/target column is not a model input."

        if col not in df.columns:
            keep = False
            reason = "Excluded: configured feature is absent from the model-ready table."

        if keep:
            included.append(col)
        audit_rows.append({
            "feature": col,
            "feature_group": group_name,
            "included_in_logistic_regression": keep,
            "reason": reason,
        })

    if not included:
        raise ValueError("No usable telemetry feature columns were found.")
    return included, pd.DataFrame(audit_rows)


def make_pipeline() -> Pipeline:
    """Create a train-fitted imputation/scaling + logistic-regression pipeline."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)),
    ])


def partition_xy(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    split: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Select eligible binary target rows and telemetry columns.

    For split=None, all non-purge rows are returned (used for D2/D3 transfer
    evaluation). Ineligible -1 targets are never treated as negative labels.
    """
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    part = df
    if split is not None:
        part = part.loc[part["split"].eq(split)]
    elif "split" in part.columns:
        part = part.loc[~part["split"].eq("purge")]

    y = pd.to_numeric(part[target_col], errors="coerce")
    mask = y.isin([0, 1])
    X = part.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[mask].copy()
    y_array = y.loc[mask].astype("int8").to_numpy()

    if len(X) == 0:
        raise ValueError(f"No eligible rows for {target_col} (split={split!r}).")
    if np.unique(y_array).size < 2:
        raise ValueError(
            f"Both classes are required for {target_col} (split={split!r}); "
            f"found classes {np.unique(y_array).tolist()}."
        )
    return X, y_array


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Return sample-level metrics, including probability-based PR-AUC/ROC-AUC."""
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
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "predicted_warning_rows": int(y_pred.sum()),
    }


def choose_probability_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, dict]:
    """Choose a decision threshold using validation F1 only.

    Candidate thresholds use validation probability quantiles plus a regular
    grid. Ties are resolved by lower FPR, higher precision, then fewer alerts.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=float)
    quantiles = np.quantile(y_prob, np.linspace(0.01, 0.99, 199))
    regular_grid = np.linspace(0.05, 0.95, 181)
    candidates = np.unique(np.concatenate(([0.0, 1.0], quantiles, regular_grid)))

    best_threshold: float | None = None
    best_metrics: dict | None = None
    best_key: tuple | None = None

    for threshold in candidates:
        pred = (y_prob >= threshold).astype(np.int8)
        metrics = binary_metrics(y_true, pred, y_prob)
        fpr = metrics["false_positive_rate"]
        key = (
            metrics["f1"],
            -(fpr if np.isfinite(fpr) else 1.0),
            metrics["precision"],
            -metrics["predicted_warning_rows"],
            float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    assert best_threshold is not None and best_metrics is not None
    return best_threshold, best_metrics


def fit_and_tune(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> tuple[Pipeline, float, dict, int, int]:
    """Fit on D1 train and tune only the probability threshold on D1 validation."""
    X_train, y_train = partition_xy(train_df, target_col, feature_cols, split="train")
    X_val, y_val = partition_xy(val_df, target_col, feature_cols, split="val")

    model = make_pipeline()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
    convergence_messages = [str(w.message) for w in caught if "converg" in str(w.message).lower()]
    if convergence_messages:
        warnings.warn(
            f"Logistic Regression may not have converged for {target_col}: "
            + " | ".join(convergence_messages),
            RuntimeWarning,
        )

    val_prob = model.predict_proba(X_val)[:, 1]
    threshold, val_metrics = choose_probability_threshold(y_val, val_prob)
    return model, threshold, val_metrics, len(X_train), len(X_val)


def evaluate_model(
    model: Pipeline,
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    threshold: float,
    split: str | None = None,
) -> dict:
    X, y = partition_xy(df, target_col, feature_cols, split=split)
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(np.int8)
    return binary_metrics(y, predictions, probabilities)


def main() -> None:
    print("\nLOGISTIC REGRESSION — EVENT-ONSET BASELINE")
    print("=" * 78)
    print("Training/tuning scenario: D1")
    print("Estimator: StandardScaler + LogisticRegression(class_weight='balanced', C=1.0)")
    print("Imputation/scaling: fitted on D1 train only")
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
    feature_audit_path = TAB_DIR / "logistic_regression_onset_feature_audit.csv"
    feature_audit.to_csv(feature_audit_path, index=False)
    print(f"\nTelemetry features used: {len(feature_cols)}")
    print("Excluded label-derived inputs:", int((~feature_audit["included_in_logistic_regression"]).sum()))

    result_rows: list[dict] = []
    threshold_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    d1 = datasets[TRAIN_DATASET]

    for horizon in HORIZONS_SEC:
        target_col = f"y_onset_h{horizon}"
        if target_col not in d1.columns:
            raise KeyError(f"Missing {target_col}. Rebuild with python src/02_build_features.py.")

        print(f"\n{'-' * 68}\nHorizon: {horizon}s | target: {target_col}\n{'-' * 68}")
        model, threshold, val_metrics, n_train, n_val = fit_and_tune(
            d1, d1, target_col, feature_cols
        )

        threshold_rows.append({
            "target": target_col,
            "horizon_seconds": horizon,
            "train_dataset": TRAIN_DATASET,
            "validation_dataset": TRAIN_DATASET,
            "model_C": LOGISTIC_REGRESSION_PARAMS["C"],
            "class_weight": LOGISTIC_REGRESSION_PARAMS["class_weight"],
            "selected_probability_threshold": threshold,
            "threshold_selection_metric": "validation_F1; ties lower FPR, higher precision, fewer alerts",
            "n_train": n_train,
            "n_validation": n_val,
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "validation_fpr": val_metrics["false_positive_rate"],
            "validation_pr_auc_average_precision": val_metrics["pr_auc_average_precision"],
        })
        print(
            f"Selected validation threshold={threshold:.4f} | "
            f"validation F1={val_metrics['f1']:.3f} | "
            f"validation FPR={val_metrics['false_positive_rate']:.3f}"
        )

        coefs = model.named_steps["model"].coef_[0]
        for feature, coefficient in zip(feature_cols, coefs):
            coefficient_rows.append({
                "target": target_col,
                "horizon_seconds": horizon,
                "feature": feature,
                "standardized_coefficient": float(coefficient),
                "absolute_coefficient": float(abs(coefficient)),
            })

        for test_ds in TEST_DATASETS:
            test_df = datasets[test_ds]
            if test_ds == TRAIN_DATASET:
                test_split = "test"
                test_definition = "chronological_test_split"
            else:
                test_split = None
                test_definition = "all_non_purge_rows_in_scenario"

            metrics = evaluate_model(
                model, test_df, target_col, feature_cols, threshold, split=test_split
            )
            if test_split is not None:
                metadata_part = test_df.loc[
                    test_df["split"].eq(test_split)
                ].copy()
            else:
                metadata_part = test_df.loc[
                    ~test_df["split"].eq("purge")
                ].copy()

            # Apply the same target eligibility condition.
            target_values = pd.to_numeric(
                metadata_part[target_col], errors="coerce"
            )

            eligible_mask = target_values.isin([0, 1])

            # Retain the original observation identifiers.
            prediction_meta = metadata_part.loc[
                eligible_mask,
                ["dataset", "host", "ts_unix", "time", "label"]
            ].copy().reset_index(drop=True)

            # Generate individual predictions using the trained model.
            X_test, y_test = partition_xy(
                test_df, target_col, feature_cols, split=test_split
            )

            probabilities = model.predict_proba(X_test)[:, 1]

            predictions = (
                probabilities >= threshold
            ).astype(np.int8)

            # Verify row alignment.
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
                    "Actual targets are not aligned with predictions."
                )

            # Construct prediction records.
            prediction_frame = prediction_meta.copy()

            prediction_frame["horizon_seconds"] = horizon
            prediction_frame["target_column"] = target_col
            prediction_frame["actual_onset"] = y_test
            prediction_frame["predicted_probability"] = probabilities
            prediction_frame["predicted_warning"] = predictions
            prediction_frame["selected_probability_threshold"] = threshold
            prediction_frame["test_definition"] = test_definition
            prediction_frame["model"] = "LogisticRegression"

            prediction_frames.append(prediction_frame)
            result_rows.append({
                "model": "LogisticRegression",
                "target": target_col,
                "horizon_seconds": horizon,
                "train_dataset": TRAIN_DATASET,
                "validation_dataset": TRAIN_DATASET,
                "test_dataset": test_ds,
                "experiment": "same_scenario" if test_ds == TRAIN_DATASET else "cross_scenario",
                "test_definition": test_definition,
                "model_C": LOGISTIC_REGRESSION_PARAMS["C"],
                "class_weight": LOGISTIC_REGRESSION_PARAMS["class_weight"],
                "selected_probability_threshold": threshold,
                "n_features": len(feature_cols),
                **metrics,
            })
            print(
                f"D1->{test_ds} | P={metrics['precision']:.3f} "
                f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
                f"FPR={metrics['false_positive_rate']:.3f} "
                f"PR-AUC={metrics['pr_auc_average_precision']:.3f} "
                f"N={metrics['n_eligible']:,}"
            )

    results = pd.DataFrame(result_rows)
    thresholds = pd.DataFrame(threshold_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    predictions = pd.concat(
        prediction_frames,
        ignore_index=True
    )
    results_path = TAB_DIR / "logistic_regression_onset_results.csv"
    thresholds_path = TAB_DIR / "logistic_regression_onset_selected_thresholds.csv"
    coefficients_path = TAB_DIR / "logistic_regression_onset_coefficients.csv"
    predictions_path = TAB_DIR / "logistic_regression_onset_predictions.csv"

    results.to_csv(results_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print(f"Saved: {predictions_path}")
    print("\nSelected thresholds (D1 validation only):")
    print(thresholds[[
        "target", "selected_probability_threshold", "validation_f1", "validation_fpr"
    ]].to_string(index=False))
    print(f"\nSaved: {results_path}")
    print(f"Saved: {thresholds_path}")
    print(f"Saved: {feature_audit_path}")
    print(f"Saved: {coefficients_path}")
    print("\nLogistic Regression event-onset baseline completed.")


if __name__ == "__main__":
    main()

"""XGBoost for LABEL=4 event-onset prediction (primary boosted-tree model).

Protocol (identical to 09 Logistic Regression and 10 Random Forest)
-------------------------------------------------------------------
* Train one horizon-specific model on D1's chronological TRAIN rows only.
* Same telemetry features, same median imputation (fitted on D1 TRAIN only).
* Class imbalance handled with scale_pos_weight computed from D1 TRAIN only.
* D1 VALIDATION is used for (a) early stopping and (b) choosing the probability
  threshold (maximize validation F1) -- never the test data.
* Evaluate on D1 TEST and on all non-purge eligible rows of D2 and D3.
* Rows with y_onset == -1, purge rows and the idle router r7 are excluded.

The feature selection, target filtering, metrics and threshold rule are imported from
10_random_forest_onset.py so every model is compared under exactly the same rules.

Hyperparameters are fixed (not searched); only the number of trees is chosen by early
stopping on D1 validation. Requires: xgboost >= 2.0.

Run from the repository root:
    python src/12_xgboost_onset.py

Outputs
-------
reports/tables/xgboost_onset_results.csv
reports/tables/xgboost_onset_selected_thresholds.csv
reports/tables/xgboost_onset_feature_audit.csv
reports/tables/xgboost_onset_feature_importance.csv
reports/tables/xgboost_onset_predictions.csv      (gitignored; read by script 11)
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import HORIZONS_SEC, MODEL_DIR, SEED, TAB_DIR  # noqa: E402
from features import feature_groups  # noqa: E402
from modeling_utils import load_model_ready, val_prediction_frame  # noqa: E402


def _load_rf_module():
    """Import 10_random_forest_onset.py (its name starts with a digit)."""
    spec = importlib.util.spec_from_file_location("rf_onset", SRC_DIR / "10_random_forest_onset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rf = _load_rf_module()
select_telemetry_features = _rf.select_telemetry_features
prepare_xy = _rf.prepare_xy
binary_metrics = _rf.binary_metrics
choose_probability_threshold = _rf.choose_probability_threshold

TRAIN_DATASET = _rf.TRAIN_DATASET
TEST_DATASETS = _rf.TEST_DATASETS

XGB_PARAMS = {
    "n_estimators": 500,            # upper bound; early stopping picks the final number
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "early_stopping_rounds": 30,
    "n_jobs": -1,
    "random_state": SEED,
}


def main() -> None:
    print("\nXGBOOST — EVENT-ONSET MODEL")
    print("=" * 78)
    print("Training/tuning scenario: D1")
    print(f"Estimator: SimpleImputer(median) + XGBClassifier({XGB_PARAMS})")
    print("Imbalance: scale_pos_weight from D1 TRAIN only")
    print("Early stopping + threshold: D1 validation only; test data never used for tuning")
    print("Evaluation: D1 untouched test, D2 non-purge, D3 non-purge")

    datasets: dict[str, pd.DataFrame] = {}
    for ds in TEST_DATASETS:
        path = MODEL_DIR / f"{ds}_model_ready.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run python src/02_build_features.py first.")
        datasets[ds] = load_model_ready(ds, exclude_idle_router=True)
        print(f"Loaded {ds}: {len(datasets[ds]):,} rows; hosts={datasets[ds]['host'].nunique()}")

    feature_cols, feature_audit = select_telemetry_features(datasets[TRAIN_DATASET])
    feature_audit = feature_audit.rename(
        columns={"included_in_random_forest": "included_in_xgboost"})
    feature_audit_path = TAB_DIR / "xgboost_onset_feature_audit.csv"
    feature_audit.to_csv(feature_audit_path, index=False)
    print(f"\nTelemetry features used: {len(feature_cols)}")

    d1 = datasets[TRAIN_DATASET]
    group_of = {c: g for g, cols in feature_groups(d1).items() for c in cols}

    results_rows: list[dict] = []
    threshold_rows: list[dict] = []
    importance_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []

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

        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        model = XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
        start = time.perf_counter()
        model.fit(X_train_imp, y_train, eval_set=[(X_val_imp, y_val)], verbose=False)
        fit_seconds = time.perf_counter() - start
        best_iteration = getattr(model, "best_iteration", None)
        n_trees_used = int(best_iteration) + 1 if best_iteration is not None else XGB_PARAMS["n_estimators"]

        val_prob = model.predict_proba(X_val_imp)[:, 1]
        threshold, val_metrics = choose_probability_threshold(y_val, val_prob)
        validation_frames.append(val_prediction_frame(
            d1, target_col, horizon, val_prob, y_val, "XGBoost"))
        threshold_rows.append({
            "target": target_col,
            "horizon_seconds": horizon,
            "train_dataset": TRAIN_DATASET,
            "validation_dataset": TRAIN_DATASET,
            **{f"xgb_{k}": v for k, v in XGB_PARAMS.items()},
            "scale_pos_weight": round(scale_pos_weight, 4),
            "n_trees_used": n_trees_used,
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
            f"trees={n_trees_used} | scale_pos_weight={scale_pos_weight:.1f} | "
            f"threshold={threshold:.4f} | val F1={val_metrics['f1']:.3f} | "
            f"val FPR={val_metrics['false_positive_rate']:.3f} | fit={fit_seconds:.1f}s"
        )

        for feature, importance in zip(feature_cols, model.feature_importances_):
            importance_rows.append({
                "target": target_col,
                "horizon_seconds": horizon,
                "feature": feature,
                "feature_group": group_of.get(feature, "other"),
                "importance_type": "xgboost_gain",
                "importance": float(importance),
            })

        for test_ds in TEST_DATASETS:
            test_df = datasets[test_ds]
            if test_ds == TRAIN_DATASET:
                test_split, test_definition = "test", "chronological_test_split"
            else:
                test_split, test_definition = None, "all_non_purge_rows_in_scenario"

            X_test, y_test = prepare_xy(test_df, target_col, feature_cols, split=test_split)
            y_prob = model.predict_proba(imputer.transform(X_test))[:, 1]
            y_pred = (y_prob >= threshold).astype(np.int8)

            part = (test_df.loc[test_df["split"].eq(test_split)] if test_split is not None
                    else test_df.loc[~test_df["split"].eq("purge")])
            target_values = pd.to_numeric(part[target_col], errors="coerce")
            eligible = target_values.isin([0, 1])
            meta = part.loc[eligible, ["dataset", "host", "ts_unix", "time", "label"]].copy()
            meta_y = target_values.loc[eligible].astype("int8").to_numpy()
            if len(meta) != len(y_test) or not np.array_equal(meta_y, y_test):
                raise RuntimeError("Prediction metadata is not aligned with the model's test rows.")

            frame = meta.reset_index(drop=True)
            frame["horizon_seconds"] = horizon
            frame["target_column"] = target_col
            frame["actual_onset"] = y_test
            frame["predicted_probability"] = y_prob
            frame["predicted_warning"] = y_pred
            frame["selected_probability_threshold"] = threshold
            frame["test_definition"] = test_definition
            frame["model"] = "XGBoost"
            prediction_frames.append(frame)

            metrics = binary_metrics(y_test, y_pred, y_prob)
            results_rows.append({
                "model": "XGBoost",
                "target": target_col,
                "horizon_seconds": horizon,
                "train_dataset": TRAIN_DATASET,
                "validation_dataset": TRAIN_DATASET,
                "test_dataset": test_ds,
                "experiment": "same_scenario" if test_ds == TRAIN_DATASET else "cross_scenario",
                "test_definition": test_definition,
                **{f"xgb_{k}": v for k, v in XGB_PARAMS.items() if k not in ("n_jobs", "early_stopping_rounds")},
                "scale_pos_weight": round(scale_pos_weight, 4),
                "n_trees_used": n_trees_used,
                "selected_probability_threshold": threshold,
                "n_features": len(feature_cols),
                "fit_seconds": round(fit_seconds, 3),
                **metrics,
            })
            print(
                f"D1->{test_ds} | P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                f"F1={metrics['f1']:.3f} FPR={metrics['false_positive_rate']:.3f} "
                f"PR-AUC={metrics['pr_auc_average_precision']:.3f} N={metrics['n_eligible']:,}"
            )

    results = pd.DataFrame(results_rows)
    results_path = TAB_DIR / "xgboost_onset_results.csv"
    thresholds_path = TAB_DIR / "xgboost_onset_selected_thresholds.csv"
    importances_path = TAB_DIR / "xgboost_onset_feature_importance.csv"
    predictions_path = TAB_DIR / "xgboost_onset_predictions.csv"

    results.to_csv(results_path, index=False)
    pd.DataFrame(threshold_rows).to_csv(thresholds_path, index=False)
    pd.DataFrame(importance_rows).to_csv(importances_path, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False)
    val_path = TAB_DIR / "xgboost_onset_val_predictions.csv"
    pd.concat(validation_frames, ignore_index=True).to_csv(val_path, index=False)

    print("\nResults summary — F1 by horizon and test scenario:")
    print(results.pivot(index="horizon_seconds", columns="test_dataset", values="f1").round(3).to_string())
    for path in (results_path, thresholds_path, feature_audit_path, importances_path, predictions_path, val_path):
        print(f"Saved: {path}")
    print("\nXGBoost event-onset model completed.")


if __name__ == "__main__":
    main()

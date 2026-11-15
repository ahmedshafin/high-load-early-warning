"""Validate event-onset target columns in all model-ready datasets.

Run from the repository root:
    python src/07_validate_onset_targets.py

This script reads existing Parquet outputs and writes a summary CSV. It does not
modify the model-ready data or any source datasets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATASETS, MODEL_DIR, TAB_DIR, HORIZONS_SEC


def validate_dataset(ds: str) -> tuple[pd.DataFrame, list[str]]:
    path = MODEL_DIR / f"{ds}_model_ready.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python src/02_build_features.py` first."
        )

    df = pd.read_parquet(path)
    required = {"host", "ts_unix", "label"}
    target_cols = [f"y_onset_h{h}" for h in HORIZONS_SEC]
    missing = sorted((required | set(target_cols)) - set(df.columns))
    if missing:
        raise KeyError(f"{ds}: missing required columns: {missing}")

    errors: list[str] = []
    labels = pd.to_numeric(df["label"], errors="coerce")
    current4 = labels.eq(4).to_numpy()

    # Every current LABEL=4 row must be ineligible for *new* event-onset prediction.
    for col in target_cols:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy()
        observed_values = set(np.unique(y[~pd.isna(y)]).tolist())
        if not observed_values.issubset({-1, 0, 1}):
            errors.append(f"{ds}/{col}: unexpected target values {sorted(observed_values)}")

        wrong_current4 = int(np.sum(current4 & (y != -1)))
        if wrong_current4:
            errors.append(
                f"{ds}/{col}: {wrong_current4} current LABEL=4 rows are not marked -1"
            )

    # Horizon nesting: a positive at a shorter horizon must not become a negative
    # at a longer horizon. A longer horizon may be -1 if its full future window
    # is no longer observed (for example, near a data boundary or a gap).
    nested_violations = {}
    for short_h, long_h in zip(HORIZONS_SEC[:-1], HORIZONS_SEC[1:]):
        short = df[f"y_onset_h{short_h}"].to_numpy()
        longer = df[f"y_onset_h{long_h}"].to_numpy()
        count = int(np.sum((short == 1) & (longer == 0)))
        nested_violations[f"h{short_h}_positive_but_h{long_h}_negative"] = count
        if count:
            errors.append(
                f"{ds}: {count} rows are positive at {short_h}s but negative at {long_h}s"
            )

    rows = []
    current4_count = int(current4.sum())
    for h in HORIZONS_SEC:
        col = f"y_onset_h{h}"
        y = df[col].to_numpy()
        pos = int(np.sum(y == 1))
        neg = int(np.sum(y == 0))
        ineligible = int(np.sum(y == -1))
        eligible = pos + neg
        non4_ineligible = int(np.sum((y == -1) & (~current4)))
        rows.append({
            "dataset": ds,
            "horizon_seconds": h,
            "rows": len(df),
            "positive_rows": pos,
            "negative_rows": neg,
            "ineligible_rows_minus1": ineligible,
            "eligible_rows": eligible,
            "positive_rate_among_eligible_pct": round(100 * pos / eligible, 4) if eligible else np.nan,
            "current_LABEL4_rows": current4_count,
            "non4_rows_ineligible_due_to_boundary_or_gap": non4_ineligible,
            "current_LABEL4_rows_not_minus1": int(np.sum(current4 & (y != -1))),
        })

    return pd.DataFrame(rows), errors


def main() -> int:
    all_rows = []
    all_errors = []

    print("EVENT-ONSET TARGET VALIDATION")
    print("=" * 72)
    print(f"Model-ready directory: {MODEL_DIR}")
    print(f"Horizons: {HORIZONS_SEC}")

    for ds in DATASETS:
        print(f"\nValidating {ds} ...")
        summary, errors = validate_dataset(ds)
        all_rows.append(summary)
        if errors:
            all_errors.extend(errors)
            print("  FAIL")
            for error in errors:
                print(f"  - {error}")
        else:
            print("  PASS: target values, LABEL=4 eligibility, and horizon nesting")
        print(summary[[
            "horizon_seconds", "positive_rows", "negative_rows",
            "ineligible_rows_minus1", "positive_rate_among_eligible_pct",
            "non4_rows_ineligible_due_to_boundary_or_gap",
        ]].to_string(index=False))

    result = pd.concat(all_rows, ignore_index=True)
    out = TAB_DIR / "onset_target_validation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print("\n" + "=" * 72)
    print(f"Saved: {out}")
    if all_errors:
        print(f"VALIDATION FAILED: {len(all_errors)} issue(s) found.")
        return 1

    print("VALIDATION PASSED for all available datasets.")
    print("Note: -1 is an ineligible/unknown sentinel, not a negative class.")
    print("Use only y_onset_h{H} values >= 0 when training/evaluating onset models.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

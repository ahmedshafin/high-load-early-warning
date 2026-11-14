"""\
M1 - Reproducible audit of the three original Network Digital Twin CSV files.

Run from the repository root with:
    python src/00_dataset_audit.py

This script READS the original files in data/raw/ and does not modify them.
It writes audit tables to reports/tables/.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import DATASETS, RAW_DIR, TAB_DIR, SAMPLING_SEC, VALID_TS_RANGE, raw_path

# Raw source-column names used by this project.
RAW_RENAMES = {
    "Host Name": "host",
    "Timestamp": "ts_unix",
    "LABEL": "label",
    "Nº de interfacesdetectadas": "n_interfaces_detected",
}

BANDWIDTH_PREFIX = "bw_"
COUNTER_PREFIXES = (
    "ifIn", "ifOut", "etherStats", "tcp", "udp", "hostIn", "hostOut"
)


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read a source CSV, accommodating a UTF-8 BOM or common Windows encoding."""
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="cp1252")


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same basic naming convention as the existing cleaning script."""
    out = df.rename(columns=RAW_RENAMES).copy()
    out.columns = [str(c).replace(".", "_").strip() for c in out.columns]
    return out


def utc_text(value: float) -> str:
    if pd.isna(value):
        return ""
    return pd.to_datetime(value, unit="s", utc=True).isoformat()


def get_timestamp_bounds() -> Tuple[float, float]:
    lo, hi = [pd.Timestamp(x, tz="UTC").timestamp() for x in VALID_TS_RANGE]
    return lo, hi


def prepare_keys_and_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact frame used for host/time overlap comparisons."""
    out = pd.DataFrame(index=df.index)
    out["host"] = df["host"].astype("string") if "host" in df else pd.NA
    out["ts_unix"] = pd.to_numeric(df["ts_unix"], errors="coerce") if "ts_unix" in df else np.nan
    out["label"] = pd.to_numeric(df["label"], errors="coerce") if "label" in df else np.nan

    bw_cols = [c for c in df.columns if c.startswith(BANDWIDTH_PREFIX) and
               (c.startswith("bw_rx_") or c.startswith("bw_tx_"))]
    if bw_cols:
        bw = df[bw_cols].apply(pd.to_numeric, errors="coerce")
        out["bw_max_mbps"] = bw.max(axis=1, skipna=True)
    else:
        out["bw_max_mbps"] = np.nan
    return out


def audit_one_dataset(ds_id: str) -> Dict[str, object]:
    path = raw_path(ds_id)
    raw = read_raw_csv(path)
    df = standardize_column_names(raw)

    # Basic schema and missingness.
    missing_rows = []
    for col in df.columns:
        missing_rows.append({
            "dataset": ds_id,
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": round(float(df[col].isna().mean() * 100), 4),
            "unique_non_null": int(df[col].nunique(dropna=True)),
        })

    if "host" not in df.columns or "ts_unix" not in df.columns or "label" not in df.columns:
        missing = [c for c in ("host", "ts_unix", "label") if c not in df.columns]
        raise ValueError(
            f"{ds_id}: required columns missing after standardization: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    # Timestamp validity; preserve the original values and do not snap/regularize.
    ts_num = pd.to_numeric(df["ts_unix"], errors="coerce")
    lo, hi = get_timestamp_bounds()
    ts_valid = ts_num.notna() & ts_num.between(lo, hi)
    host_valid = df["host"].notna() & df["host"].astype(str).str.strip().ne("")
    label_num = pd.to_numeric(df["label"], errors="coerce")

    # Raw bandwidth fields and the maximum observed interface bandwidth per row.
    bw_cols = [c for c in df.columns if c.startswith("bw_rx_") or c.startswith("bw_tx_")]
    bw_numeric = df[bw_cols].apply(pd.to_numeric, errors="coerce") if bw_cols else pd.DataFrame(index=df.index)
    bw_max = bw_numeric.max(axis=1, skipna=True) if bw_cols else pd.Series(np.nan, index=df.index)

    # Duplicate checks.
    duplicate_full_rows = int(df.duplicated().sum())
    valid_keys = pd.DataFrame({
        "host": df.loc[ts_valid & host_valid, "host"].astype(str),
        "ts_unix": ts_num.loc[ts_valid & host_valid].astype("int64"),
    })
    duplicate_host_ts = int(valid_keys.duplicated(["host", "ts_unix"]).sum())

    # Dataset overview.
    valid_ts_values = ts_num.loc[ts_valid]
    valid_hosts = sorted(df.loc[host_valid, "host"].astype(str).unique().tolist())
    overview = {
        "dataset": ds_id,
        "source_file": path.name,
        "file_size_mb": round(path.stat().st_size / (1024 ** 2), 3),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "hosts_count": int(len(valid_hosts)),
        "hosts": ";".join(valid_hosts),
        "timestamp_valid_rows": int(ts_valid.sum()),
        "timestamp_invalid_or_missing_rows": int((~ts_valid).sum()),
        "host_missing_rows": int((~host_valid).sum()),
        "label_missing_or_non_numeric_rows": int(label_num.isna().sum()),
        "duplicate_full_rows": duplicate_full_rows,
        "duplicate_host_timestamp_rows": duplicate_host_ts,
        "timestamp_start_utc": utc_text(valid_ts_values.min()) if len(valid_ts_values) else "",
        "timestamp_end_utc": utc_text(valid_ts_values.max()) if len(valid_ts_values) else "",
        "unique_label_values": ";".join(map(str, sorted(label_num.dropna().unique().tolist()))),
    }

    # Actual time-gap audit within each host. No resampling or timestamp snapping.
    time_frame = pd.DataFrame({
        "host": df.loc[ts_valid & host_valid, "host"].astype(str),
        "ts_unix": ts_num.loc[ts_valid & host_valid].astype("int64"),
    }).sort_values(["host", "ts_unix"])
    gap_rows = []
    for host, group in time_frame.groupby("host", sort=True):
        gaps = group["ts_unix"].diff().dropna()
        counts = gaps.value_counts().sort_index()
        gap_rows.append({
            "dataset": ds_id,
            "host": host,
            "observations": int(len(group)),
            "start_utc": utc_text(group["ts_unix"].min()),
            "end_utc": utc_text(group["ts_unix"].max()),
            "median_gap_s": float(gaps.median()) if len(gaps) else np.nan,
            "mode_gap_s": float(gaps.mode().iloc[0]) if len(gaps) and len(gaps.mode()) else np.nan,
            "exact_nominal_gaps": int((gaps == SAMPLING_SEC).sum()),
            "shorter_than_nominal_gaps": int((gaps < SAMPLING_SEC).sum()),
            "longer_than_nominal_gaps": int((gaps > SAMPLING_SEC).sum()),
            "max_gap_s": float(gaps.max()) if len(gaps) else np.nan,
            "distinct_gap_values_s": json.dumps({str(k): int(v) for k, v in counts.items()}),
        })

    # LABEL counts and observed bandwidth ranges per supplied label.
    label_rows = []
    label_valid = label_num.notna()
    labels_all = sorted(label_num.loc[label_valid].unique().tolist())
    counts = label_num.loc[label_valid].value_counts()
    total_labels = int(counts.sum())
    for label_value in labels_all:
        mask = label_num.eq(label_value)
        values = bw_max.loc[mask].dropna()
        label_rows.append({
            "dataset": ds_id,
            "label": label_value,
            "count": int(counts.get(label_value, 0)),
            "percentage": round(float(counts.get(label_value, 0) / total_labels * 100), 4) if total_labels else np.nan,
            "bw_max_min_mbps": float(values.min()) if len(values) else np.nan,
            "bw_max_median_mbps": float(values.median()) if len(values) else np.nan,
            "bw_max_max_mbps": float(values.max()) if len(values) else np.nan,
            "rows_without_bandwidth_value": int(mask.sum() - len(values)),
        })

    # LABEL=4 episodes. An episode is split whenever consecutive observations are
    # not exactly SAMPLING_SEC apart, so missing intervals are never silently bridged.
    event_rows = []
    event_frame = pd.DataFrame({
        "host": df.loc[ts_valid & host_valid & label_valid, "host"].astype(str),
        "ts_unix": ts_num.loc[ts_valid & host_valid & label_valid].astype("int64"),
        "label": label_num.loc[ts_valid & host_valid & label_valid].astype(float),
    }).sort_values(["host", "ts_unix"])

    for host, group in event_frame.groupby("host", sort=True):
        group = group.reset_index(drop=True)
        delta = group["ts_unix"].diff()
        prev_label = group["label"].shift(1)
        is4 = group["label"].eq(4)
        # A new segment begins when LABEL first becomes 4 or when a time gap breaks continuity.
        starts = is4 & (prev_label.ne(4) | delta.ne(SAMPLING_SEC))
        segment_id = starts.cumsum()
        events = group.loc[is4].assign(_event_segment=segment_id.loc[is4].values)
        for event_number, (_, ev) in enumerate(events.groupby("_event_segment", sort=True), start=1):
            start_ts = int(ev["ts_unix"].min())
            end_ts = int(ev["ts_unix"].max())
            start_index = int(ev.index.min())
            # A confirmed onset requires an immediately preceding observed sample
            # exactly one nominal interval earlier and with a non-4 label.
            has_previous = start_index > 0
            previous_is_adjacent = bool(has_previous and (start_ts - int(group.loc[start_index - 1, "ts_unix"]) == SAMPLING_SEC))
            previous_is_non4 = bool(has_previous and group.loc[start_index - 1, "label"] != 4)
            if not has_previous:
                onset_type = "left_censored_first_observation"
            elif not previous_is_adjacent:
                onset_type = "start_after_timestamp_gap"
            elif previous_is_non4:
                onset_type = "observed_non4_to_4_transition"
            else:
                onset_type = "unclassified_segment_boundary"

            n_samples = int(len(ev))
            event_rows.append({
                "dataset": ds_id,
                "host": host,
                "event_number_within_host": event_number,
                "event_start_utc": utc_text(start_ts),
                "event_end_utc": utc_text(end_ts),
                "start_ts_unix": start_ts,
                "end_ts_unix": end_ts,
                "sample_count": n_samples,
                "timestamp_span_s": end_ts - start_ts,
                "estimated_coverage_s_at_nominal_interval": n_samples * SAMPLING_SEC,
                "onset_type": onset_type,
                "onset_confirmed_from_observed_transition": int(onset_type == "observed_non4_to_4_transition"),
            })

    # A compact frame for cross-file overlap comparison.
    compact = prepare_keys_and_measurements(df)
    compact = compact.loc[
        pd.to_numeric(compact["ts_unix"], errors="coerce").between(lo, hi) & compact["host"].notna()
    ].copy()
    compact["ts_unix"] = pd.to_numeric(compact["ts_unix"], errors="coerce").astype("int64")
    compact["host"] = compact["host"].astype(str)
    compact = compact.drop_duplicates(["host", "ts_unix"], keep="first")

    # SNMP-style field profile. The project data dictionary identifies these as
    # per-interval deltas; this table profiles values but does not infer/rewrite units.
    counter_rows = []
    counter_cols = [c for c in df.columns if c.startswith(COUNTER_PREFIXES)]
    for col in counter_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        non_null = values.dropna()
        counter_rows.append({
            "dataset": ds_id,
            "field": col,
            "non_null_count": int(non_null.size),
            "missing_count": int(values.isna().sum()),
            "negative_count": int((non_null < 0).sum()),
            "zero_pct": round(float((non_null == 0).mean() * 100), 4) if len(non_null) else np.nan,
            "minimum": float(non_null.min()) if len(non_null) else np.nan,
            "median": float(non_null.median()) if len(non_null) else np.nan,
            "p95": float(non_null.quantile(0.95)) if len(non_null) else np.nan,
            "maximum": float(non_null.max()) if len(non_null) else np.nan,
            "interpretation_from_project_data_dictionary": "per-interval delta; verify field-specific units",
        })

    return {
        "overview": overview,
        "missingness": missing_rows,
        "timestamp_gaps": gap_rows,
        "label_bandwidth": label_rows,
        "events": event_rows,
        "compact": compact,
        "counter_profile": counter_rows,
    }


def build_overlap_report(compact_by_dataset: Dict[str, pd.DataFrame]) -> List[dict]:
    rows = []
    for ds_a, ds_b in itertools.combinations(compact_by_dataset.keys(), 2):
        a = compact_by_dataset[ds_a]
        b = compact_by_dataset[ds_b]
        timestamps_a = set(a["ts_unix"].unique().tolist())
        timestamps_b = set(b["ts_unix"].unique().tolist())
        shared_timestamps = timestamps_a.intersection(timestamps_b)

        merged = a.merge(b, on=["host", "ts_unix"], how="inner", suffixes=("_a", "_b"))
        label_both = merged["label_a"].notna() & merged["label_b"].notna()
        bw_both = merged["bw_max_mbps_a"].notna() & merged["bw_max_mbps_b"].notna()
        label_equal = int((merged.loc[label_both, "label_a"] == merged.loc[label_both, "label_b"]).sum())
        bw_equal = int(np.isclose(
            merged.loc[bw_both, "bw_max_mbps_a"].astype(float),
            merged.loc[bw_both, "bw_max_mbps_b"].astype(float),
            rtol=0, atol=1e-9, equal_nan=False,
        ).sum())
        rows.append({
            "dataset_a": ds_a,
            "dataset_b": ds_b,
            "shared_timestamp_count_any_host": int(len(shared_timestamps)),
            "shared_host_timestamp_keys": int(len(merged)),
            "shared_keys_with_label_compared": int(label_both.sum()),
            "shared_keys_with_same_label": label_equal,
            "shared_keys_with_different_label": int(label_both.sum() - label_equal),
            "shared_keys_with_bandwidth_compared": int(bw_both.sum()),
            "shared_keys_with_same_bw_max": bw_equal,
            "shared_keys_with_different_bw_max": int(bw_both.sum() - bw_equal),
        })
    return rows


def write_csv(rows: List[dict], filename: str) -> None:
    path = TAB_DIR / filename
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  saved: {path}")


def main() -> None:
    print("=" * 72)
    print("RAW DATASET AUDIT - M1")
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Output directory:   {TAB_DIR}")
    print("Original CSV files will not be modified.")
    print("=" * 72)

    results = {}
    for ds_id in DATASETS:
        print(f"\nAuditing {ds_id} ...")
        try:
            result = audit_one_dataset(ds_id)
        except FileNotFoundError:
            print(f"ERROR: no CSV matched {DATASETS[ds_id]} in {RAW_DIR}")
            raise
        results[ds_id] = result
        ov = result["overview"]
        print(f"  file: {ov['source_file']}")
        print(f"  rows/columns: {ov['rows']:,} / {ov['columns']}")
        print(f"  hosts: {ov['hosts_count']} ({ov['hosts']})")
        print(f"  invalid timestamps: {ov['timestamp_invalid_or_missing_rows']:,}")
        print(f"  duplicate host/timestamp rows: {ov['duplicate_host_timestamp_rows']:,}")
        print(f"  LABEL=4 episodes found: {len(result['events']):,}")

    # Write separate, easy-to-inspect CSV reports.
    write_csv([r["overview"] for r in results.values()], "dataset_audit_overview.csv")
    write_csv([row for r in results.values() for row in r["missingness"]], "column_missingness.csv")
    write_csv([row for r in results.values() for row in r["timestamp_gaps"]], "timestamp_gap_audit.csv")
    write_csv([row for r in results.values() for row in r["label_bandwidth"]], "label_bandwidth_audit.csv")
    write_csv([row for r in results.values() for row in r["events"]], "label4_event_audit.csv")
    write_csv([row for r in results.values() for row in r["counter_profile"]], "snmp_counter_profile.csv")

    overlap = build_overlap_report({ds: results[ds]["compact"] for ds in results})
    write_csv(overlap, "cross_dataset_overlap.csv")

    print("\nAudit complete. Review the CSV files in reports/tables/.")
    print("Important: event durations are sample-based estimates at the nominal polling interval.")
    print("Events beginning at the first observation or after a timestamp gap are flagged as unconfirmed onsets.")
    print("SNMP field profiles report values; they do not convert counters into rates.")


if __name__ == "__main__":
    main()

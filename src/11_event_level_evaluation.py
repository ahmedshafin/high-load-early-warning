
"""
Event-level evaluation for Random Forest onset predictions.

Uses:
- random_forest_onset_predictions.csv
- label4_event_audit.csv
- model_ready CSV files for monitoring-time exposure

No model training is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TAB_DIR = ROOT / "reports" / "tables"
MODEL_DIR = ROOT / "data" / "model_ready"

DATASETS = ["D1", "D2", "D3"]
HORIZONS = [10, 20, 30, 60]

IDLE_ROUTER = "r7"
SAMPLING_SEC = 10


def load_inputs():
    """Load saved predictions and confirmed LABEL=4 events."""

    pred_path = TAB_DIR / "random_forest_onset_predictions.csv"
    event_path = TAB_DIR / "label4_event_audit.csv"

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    if not event_path.exists():
        raise FileNotFoundError(event_path)

    predictions = pd.read_csv(pred_path)
    events = pd.read_csv(event_path)

    # Normalize identifiers and timestamps.
    predictions["host"] = predictions["host"].astype(str)
    predictions["ts_unix"] = pd.to_numeric(
        predictions["ts_unix"], errors="raise"
    ).astype("int64")

    events["host"] = events["host"].astype(str)
    events["start_ts_unix"] = pd.to_numeric(
        events["start_ts_unix"], errors="raise"
    ).astype("int64")

    predictions["actual_onset"] = pd.to_numeric(
        predictions["actual_onset"], errors="raise"
    ).astype("int8")

    predictions["predicted_warning"] = pd.to_numeric(
        predictions["predicted_warning"], errors="raise"
    ).astype("int8")

    # Only confirmed observed non-4 -> 4 transitions.
    events = events[
        pd.to_numeric(
            events["onset_confirmed_from_observed_transition"],
            errors="coerce"
        ).eq(1)
    ].copy()

    return predictions, events


def determine_event_eligibility(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Determine whether every confirmed event had at least one
    valid positive target opportunity within its preceding horizon.

    Events outside the evaluation window are not counted as misses.
    """

    prediction_lookup = {}

    for (dataset, host), group in predictions.groupby(
        ["dataset", "host"]
    ):

        group = group.sort_values("ts_unix")

        prediction_lookup[(dataset, host)] = (
            group["ts_unix"].to_numpy(dtype=np.int64),
            group["actual_onset"].to_numpy(dtype=np.int8),
        )

    records = []

    for _, event in events.iterrows():

        dataset = event["dataset"]
        host = event["host"]
        start = int(event["start_ts_unix"])

        key = (dataset, host)

        times, targets = prediction_lookup.get(
            key,
            (np.array([], dtype=np.int64),
             np.array([], dtype=np.int8))
        )

        # Target window: [event_start - H, event_start)
        left = np.searchsorted(
            times, start - horizon, side="left"
        )

        right = np.searchsorted(
            times, start, side="left"
        )

        prior_targets = targets[left:right]

        n_prior = int(len(prior_targets))
        n_positive = int(np.sum(prior_targets == 1))

        # A valid prediction opportunity before this event
        # must identify this event in its future onset window.
        if n_prior > 0 and n_positive == 0:
            raise RuntimeError(
                f"Target/event mismatch: {dataset}, {host}, "
                f"event {event['event_number_within_host']}, "
                f"horizon {horizon}s"
            )

        row = event.to_dict()

        row["horizon_seconds"] = horizon
        row["n_prior_prediction_rows"] = n_prior
        row["n_positive_target_rows"] = n_positive
        row["eligible_for_evaluation"] = n_positive > 0

        records.append(row)

    return pd.DataFrame(records)


def evaluate_host(
    warning_rows: pd.DataFrame,
    host_events: pd.DataFrame,
    dataset: str,
    host: str,
    horizon: int,
):
    """
    Match alert episodes to individual events chronologically.

    Alert policy:
    - A positive prediction opens an alert.
    - Further positive predictions renew an active alert.
    - An actual event closes and matches the active alert.
    - An expired unmatched alert is a false alarm.
    - Each alert and event can be matched only once.
    """

    warnings = (
        warning_rows[
            warning_rows["predicted_warning"] == 1
        ]
        .sort_values("ts_unix")
        .to_dict("records")
    )

    events = (
        host_events.sort_values("start_ts_unix")
        .to_dict("records")
    )

    alert_records = []
    matches = {}

    wi = 0
    ei = 0
    alert_number = 0

    active = None

    def close_alert(
        alert,
        status,
        end_time,
        matched_event=None,
        first_valid_warning=None,
        lead_time=None,
    ):

        alert_records.append({
            "dataset": dataset,
            "host": host,
            "horizon_seconds": horizon,
            "alert_id": alert["alert_id"],
            "alert_start_ts": alert["start_ts"],
            "last_warning_ts": alert["last_warning_ts"],
            "alert_expiry_ts": alert["expires_at"],
            "alert_end_ts": int(end_time),
            "n_positive_predictions": len(
                alert["warning_times"]
            ),
            "alert_status": status,
            "matched_event_number": (
                matched_event if matched_event is not None
                else np.nan
            ),
            "first_valid_warning_ts": (
                first_valid_warning
                if first_valid_warning is not None
                else np.nan
            ),
            "lead_time_seconds": (
                lead_time
                if lead_time is not None
                else np.nan
            ),
        })

    # Process warnings and events in chronological order.
    while (
        wi < len(warnings)
        or ei < len(events)
        or active is not None
    ):

        warning_time = (
            int(warnings[wi]["ts_unix"])
            if wi < len(warnings)
            else float("inf")
        )

        event_time = (
            int(events[ei]["start_ts_unix"])
            if ei < len(events)
            else float("inf")
        )

        next_time = min(warning_time, event_time)

        # Expire an alert only when its horizon has passed.
        # An event exactly at the expiry time still counts.
        if active is not None and next_time > active["expires_at"]:

            close_alert(
                active,
                status="false_alarm",
                end_time=active["expires_at"],
            )

            active = None
            continue

        # Process events before warnings at identical timestamps.
        # A prediction at the event onset is not an advance warning.
        if event_time <= warning_time:

            event = events[ei]

            event_number = int(
                event["event_number_within_host"]
            )

            detected = False

            if active is not None:

                valid_warning_times = [
                    t for t in active["warning_times"]
                    if event_time - horizon <= t < event_time
                ]

                if not valid_warning_times:
                    raise RuntimeError(
                        "Active alert has no valid warning "
                        f"for event {event_number}"
                    )

                # First warning that actually falls inside
                # this event's H-second warning window.
                first_valid = min(valid_warning_times)

                lead_time = event_time - first_valid

                close_alert(
                    active,
                    status="matched",
                    end_time=event_time,
                    matched_event=event_number,
                    first_valid_warning=first_valid,
                    lead_time=lead_time,
                )

                matches[event_number] = {
                    "event_detected": 1,
                    "matched_alert_id": active["alert_id"],
                    "first_valid_warning_ts": first_valid,
                    "lead_time_seconds": lead_time,
                }

                active = None
                detected = True

            if not detected:
                matches[event_number] = {
                    "event_detected": 0,
                    "matched_alert_id": None,
                    "first_valid_warning_ts": np.nan,
                    "lead_time_seconds": np.nan,
                }

            ei += 1

        else:

            warning = warnings[wi]
            t = int(warning["ts_unix"])

            if active is None:

                alert_number += 1

                active = {
                    "alert_id": (
                        f"{dataset}_H{horizon}_{host}_"
                        f"A{alert_number}"
                    ),
                    "alert_start_ts": t,
                    "start_ts": t,
                    "last_warning_ts": t,
                    "expires_at": t + horizon,
                    "warning_times": [t],
                }

            else:

                # Existing active alert receives a new warning.
                active["warning_times"].append(t)
                active["last_warning_ts"] = t
                active["expires_at"] = max(
                    active["expires_at"],
                    t + horizon,
                )

            wi += 1

    # Close any remaining alert that expired without an event.
    if active is not None:

        close_alert(
            active,
            status="false_alarm",
            end_time=active["expires_at"],
        )

    return alert_records, matches


def calculate_monitoring_hours(dataset: str) -> float:
    """
    Calculate observed monitoring duration in router-hours.

    D1: chronological test split only.
    D2/D3: all non-purge rows.

    Exclude gaps larger than the nominal 10-second interval.
    """

    path = MODEL_DIR / f"{dataset}_model_ready.csv.gz"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        compression="gzip",
        usecols=["host", "ts_unix", "split"],
    )

    df = df[df["host"] != IDLE_ROUTER].copy()

    if dataset == "D1":
        df = df[df["split"] == "test"]
    else:
        df = df[df["split"] != "purge"]

    total_seconds = 0

    for _, group in df.groupby("host"):

        times = np.sort(
            group["ts_unix"].unique().astype(np.int64)
        )

        gaps = np.diff(times)

        # Count only continuously observed intervals.
        total_seconds += int(
            gaps[gaps == SAMPLING_SEC].sum()
        )

    return total_seconds / 3600.0


def main():

    print("\nRANDOM FOREST — EVENT-LEVEL EVALUATION")
    print("=" * 78)

    predictions, all_events = load_inputs()

    print("Prediction rows:", len(predictions))
    print("Confirmed actual events:", len(all_events))

    summary_rows = []
    event_output_frames = []
    alert_output_frames = []

    monitoring_hours = {
        ds: calculate_monitoring_hours(ds)
        for ds in DATASETS
    }

    for horizon in HORIZONS:

        print(
            f"\n{'-' * 65}\n"
            f"HORIZON: {horizon} seconds\n"
            f"{'-' * 65}"
        )

        for dataset in DATASETS:

            p = predictions[
                (predictions["dataset"] == dataset)
                & (predictions["horizon_seconds"] == horizon)
            ].copy()

            e = all_events[
                all_events["dataset"] == dataset
            ].copy()

            if p.empty:
                raise ValueError(
                    f"No predictions for {dataset}, H={horizon}"
                )

            # Determine which confirmed events have valid
            # prediction opportunities.
            event_eval = determine_event_eligibility(
                p, e, horizon
            )

            event_eval["event_detected"] = np.nan
            event_eval["matched_alert_id"] = None
            event_eval["first_valid_warning_ts"] = np.nan
            event_eval["lead_time_seconds"] = np.nan

            event_eval.loc[
                event_eval["eligible_for_evaluation"],
                "event_detected"
            ] = 0

            alerts_for_dataset = []

            hosts = sorted(
                set(p["host"]) |
                set(event_eval["host"])
            )

            for host in hosts:

                warning_rows = p[p["host"] == host]

                host_events = event_eval[
                    (event_eval["host"] == host)
                    & (event_eval["eligible_for_evaluation"])
                ]

                alert_records, matches = evaluate_host(
                    warning_rows,
                    host_events,
                    dataset,
                    host,
                    horizon,
                )

                alerts_for_dataset.extend(alert_records)

                # Attach event-level detection results.
                for idx, event in host_events.iterrows():

                    event_number = int(
                        event["event_number_within_host"]
                    )

                    result = matches[event_number]

                    event_eval.at[
                        idx, "event_detected"
                    ] = result["event_detected"]

                    event_eval.at[
                        idx, "matched_alert_id"
                    ] = result["matched_alert_id"]

                    event_eval.at[
                        idx, "first_valid_warning_ts"
                    ] = result["first_valid_warning_ts"]

                    event_eval.at[
                        idx, "lead_time_seconds"
                    ] = result["lead_time_seconds"]

            alert_df = pd.DataFrame(alerts_for_dataset)

            eligible_events = event_eval[
                event_eval["eligible_for_evaluation"]
            ]

            detected_events = int(
                (eligible_events["event_detected"] == 1).sum()
            )

            eligible_count = len(eligible_events)
            missed_events = eligible_count - detected_events

            alert_count = len(alert_df)

            false_alarms = (
                int((alert_df["alert_status"] == "false_alarm").sum())
                if alert_count else 0
            )

            matched_alerts = (
                int((alert_df["alert_status"] == "matched").sum())
                if alert_count else 0
            )

            router_hours = monitoring_hours[dataset]

            false_alarm_rate = (
                false_alarms / router_hours
                if router_hours > 0
                else np.nan
            )

            lead_times = pd.to_numeric(
                eligible_events.loc[
                    eligible_events["event_detected"] == 1,
                    "lead_time_seconds"
                ],
                errors="coerce"
            ).dropna()

            detection_rate = (
                detected_events / eligible_count * 100
                if eligible_count else np.nan
            )

            summary_rows.append({
                "model": "RandomForest",
                "dataset": dataset,
                "horizon_seconds": horizon,
                "confirmed_events": len(event_eval),
                "eligible_events": eligible_count,
                "detected_events": detected_events,
                "missed_events": missed_events,
                "event_detection_rate_pct": round(
                    detection_rate, 3
                ) if np.isfinite(detection_rate) else np.nan,
                "total_alert_episodes": alert_count,
                "matched_alerts": matched_alerts,
                "false_alarms": false_alarms,
                "monitoring_router_hours": round(
                    router_hours, 3
                ),
                "false_alarms_per_router_hour": round(
                    false_alarm_rate, 4
                ) if np.isfinite(false_alarm_rate) else np.nan,
                "median_lead_time_seconds": (
                    round(float(lead_times.median()), 3)
                    if not lead_times.empty else np.nan
                ),
                "mean_lead_time_seconds": (
                    round(float(lead_times.mean()), 3)
                    if not lead_times.empty else np.nan
                ),
            })

            event_output_frames.append(event_eval)
            alert_output_frames.append(alert_df)

            print(f"\n{dataset}:")
            print("Confirmed events:", len(event_eval))
            print("Eligible events:", eligible_count)
            print("Detected events:", detected_events)
            print("Missed events:", missed_events)
            print(f"Detection rate: {detection_rate:.2f}%")
            print("Alert episodes:", alert_count)
            print("Matched alerts:", matched_alerts)
            print("False alarms:", false_alarms)
            print(
                "False alarms/router-hour:",
                round(false_alarm_rate, 4)
            )
            print(
                "Median lead time:",
                round(float(lead_times.median()), 2)
                if not lead_times.empty else "N/A",
                "seconds"
            )

    # Combine all results.
    summary = pd.DataFrame(summary_rows)

    event_output = pd.concat(
        event_output_frames,
        ignore_index=True
    )

    alert_output = pd.concat(
        alert_output_frames,
        ignore_index=True
    )

    # Save reproducible results.
    summary_path = (
        TAB_DIR / "random_forest_event_level_summary.csv"
    )

    events_path = (
        TAB_DIR / "random_forest_event_level_events.csv"
    )

    alerts_path = (
        TAB_DIR / "random_forest_alert_episodes.csv"
    )

    summary.to_csv(summary_path, index=False)
    event_output.to_csv(events_path, index=False)
    alert_output.to_csv(alerts_path, index=False)

    print("\n" + "=" * 78)
    print("EVENT-LEVEL RESULTS SUMMARY")
    print("=" * 78)

    display_columns = [
        "dataset",
        "horizon_seconds",
        "eligible_events",
        "detected_events",
        "event_detection_rate_pct",
        "false_alarms",
        "false_alarms_per_router_hour",
        "median_lead_time_seconds",
    ]

    print(
        summary[display_columns]
        .to_string(index=False)
    )

    print("\nSaved files:")
    print(summary_path)
    print(events_path)
    print(alerts_path)

    print("\nEvent-level evaluation completed successfully.")


if __name__ == "__main__":
    main()

"""Unit tests for the original pipeline and new event-onset targets."""
import numpy as np
import pandas as pd

from common import label_from_bw, run_lengths
from config import HORIZONS_SEC, SAMPLING_SEC
from features import build, feature_groups, add_event_onset_targets

T0 = 1_748_860_136


def make_clean(n=600, hosts=("r1", "r2", "r3"), seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for h in hosts:
        load = np.zeros(n)
        i = 0
        while i < n:
            dur = int(rng.integers(1, 12))
            load[i:i + dur] = rng.choice([0, 5, 30, 60, 80, 100, 130])
            i += dur
        tx, rx = load, load * rng.uniform(.4, .6, size=n)
        df = pd.DataFrame({"dataset": "T", "host": h, "ts_unix": T0 + SAMPLING_SEC * np.arange(n),
                           "bw_rx_3": rx, "bw_tx_3": tx})
        df["bw_max"] = df[["bw_rx_3", "bw_tx_3"]].max(axis=1)
        df["label"] = label_from_bw(df["bw_max"]).astype("int8")
        df["has_if6"] = df["any_imputed"] = df["n_imputed"] = df["idle_tail"] = 0
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_label_bins():
    x = pd.Series([0, 9.9, 10, 39.9, 40, 69.9, 70, 89.9, 90, 150])
    assert label_from_bw(x).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_run_lengths():
    assert run_lengths(np.array([0, 1, 1, 0, 1, 0, 1, 1, 1])).tolist() == [2, 1, 3]
    assert run_lengths(np.array([0, 0])).tolist() == []


def test_features_are_causal():
    """Features at t must not change when everything after t is removed."""
    clean = make_clean()
    cut_ts = T0 + SAMPLING_SEC * 300
    full = build(clean.copy(), drop_idle_tail=False)
    cut = build(clean[clean.ts_unix <= cut_ts].copy(), drop_idle_tail=False)
    feats = [c for v in feature_groups(full).values() for c in v]
    a = full[full.ts_unix <= cut_ts].sort_values(["host", "ts_unix"]).reset_index(drop=True)
    b = cut.sort_values(["host", "ts_unix"]).reset_index(drop=True)
    assert len(a) == len(b)
    for c in feats:
        assert np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True), f"{c} uses future data"


def test_targets_equal_future_label():
    clean = make_clean()
    out = build(clean.copy(), drop_idle_tail=False)
    lk = clean.set_index(["host", "ts_unix"])["label"]
    for H in HORIZONS_SEC:
        ok = out[f"y_cls_h{H}"].notna()
        key = pd.MultiIndex.from_arrays([out.loc[ok, "host"], out.loc[ok, "ts_unix"] + H])
        assert (lk.reindex(key).values == out.loc[ok, f"y_cls_h{H}"].values).all()
        assert (out.loc[ok, "y_ge4_h{H}".format(H=H)] == (out.loc[ok, f"y_cls_h{H}"] >= 4)).all()


def test_build_includes_event_onset_targets_and_preserves_future_state_targets():
    out = build(make_clean(n=120), drop_idle_tail=False)
    for H in HORIZONS_SEC:
        onset_col = f"y_onset_h{H}"
        assert onset_col in out.columns
        assert set(out[onset_col].unique()).issubset({-1, 0, 1})
        assert (out.loc[out["label"] == 4, onset_col] == -1).all()
        assert f"y_cls_h{H}" in out.columns
        assert f"y_ge4_h{H}" in out.columns


def test_event_onset_targets_use_new_transition_not_current_state():
    labels = [0, 0, 4, 4, 0, 0, 0, 4, 4, 0]
    df = pd.DataFrame({"host": "r1", "ts_unix": T0 + SAMPLING_SEC * np.arange(len(labels)),
                       "label": labels})
    out = add_event_onset_targets(df)

    assert out["y_onset_h10"].tolist() == [0, 1, -1, -1, 0, 0, 1, -1, -1, -1]
    assert out["y_onset_h20"].tolist() == [1, 1, -1, -1, 0, 1, 1, -1, -1, -1]


def test_event_onset_targets_mark_incomplete_or_gapped_horizons_unknown():
    # There is a 20-second gap between observations at t=10 and t=30.
    labels = [0, 0, 4, 0, 0, 4]
    timestamps = T0 + np.array([0, 10, 30, 40, 50, 60])
    df = pd.DataFrame({"host": "r1", "ts_unix": timestamps, "label": labels})
    out = add_event_onset_targets(df)

    # t=10 cannot be labelled because its next expected observation at t=20 is absent.
    assert out.loc[1, "y_onset_h10"] == -1
    # After the gap, a normal non-4 -> 4 transition is still a confirmed onset.
    assert out.loc[4, "y_onset_h10"] == 1
    # Current LABEL 4 and the final incomplete horizon are unknown/ineligible.
    assert out.loc[2, "y_onset_h10"] == -1
    assert out.loc[5, "y_onset_h10"] == -1


def test_event_onset_targets_are_host_specific():
    ts = T0 + SAMPLING_SEC * np.arange(5)
    df = pd.concat([
        pd.DataFrame({"host": "r1", "ts_unix": ts, "label": [0, 0, 4, 4, 0]}),
        pd.DataFrame({"host": "r2", "ts_unix": ts, "label": [0, 0, 0, 0, 0]}),
    ], ignore_index=True)
    out = add_event_onset_targets(df)
    assert out.loc[(out.host == "r1") & (out.ts_unix == ts[1]), "y_onset_h10"].iloc[0] == 1
    assert out.loc[(out.host == "r2") & (out.ts_unix == ts[1]), "y_onset_h10"].iloc[0] == 0


def test_split_has_purge_gap():
    out = build(make_clean(), drop_idle_tail=False)
    mx = max(HORIZONS_SEC)
    t = {s: out.loc[out.split == s, "ts_unix"] for s in ("train", "val", "test")}
    assert t["train"].max() + mx < t["val"].min() + 1
    assert t["val"].max() + mx < t["test"].min() + 1


def test_idle_tail_removed_but_targets_use_it():
    clean = make_clean()
    last = clean.ts_unix.max()
    clean.loc[clean.ts_unix > last - SAMPLING_SEC * 100, "idle_tail"] = 1
    out = build(clean, drop_idle_tail=True)
    assert out.ts_unix.max() <= last - SAMPLING_SEC * 100
    # Rows right before the tail still get a defined future-state target (computed on full series).
    assert out.loc[out.ts_unix == out.ts_unix.max(), "y_cls_h60"].notna().all()

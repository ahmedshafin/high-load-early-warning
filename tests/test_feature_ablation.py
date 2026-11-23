"""Tests for the feature-ablation configurations in 14_feature_ablation.py (needs xgboost)."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("xgboost")

SRC = Path(__file__).resolve().parents[1] / "src"
_spec = importlib.util.spec_from_file_location("feature_ablation", SRC / "14_feature_ablation.py")
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)

COLUMNS = [
    "bw_rx_3", "bw_tx_3", "bw_rx_sum", "bw_tx_sum", "bw_rx_share", "bw_max", "bw_total",
    "eth_pkts_sum", "eth_octets_sum", "ucast_in_sum", "discards_sum", "rmon_err_sum",
    "tcpOutRsts", "tcpInSegs", "udpInDatagrams",
    "bw_max_lag1", "bw_max_diff1", "bw_max_mean30s", "bw_total_std30s",
    "eth_pkts_sum_mean30s", "discards_sum_mean60s",
    "ctx_core_bw_max_mean", "ctx_net_bw_max_max", "has_if6", "any_imputed",
]


def frame():
    return pd.DataFrame({c: [0.0, 1.0] for c in COLUMNS})


def test_cumulative_configs_are_nested_and_d_is_everything():
    df = frame()
    cfg = ab.build_configs(df, COLUMNS)
    a, b, c, d = (set(cfg[k]) for k in ("A_bandwidth", "B_plus_packets", "C_plus_tcp_udp", "D_all"))
    assert a < b < c < d and d == set(COLUMNS)


def test_signal_sources_are_separated():
    cfg = ab.build_configs(frame(), COLUMNS)
    a, b, c = set(cfg["A_bandwidth"]), set(cfg["B_plus_packets"]), set(cfg["C_plus_tcp_udp"])
    assert all(x.startswith("bw_") for x in a)                       # bandwidth + its history only
    assert {"eth_pkts_sum", "eth_pkts_sum_mean30s", "discards_sum_mean60s"} <= b - a
    assert not any(x.startswith(("tcp", "udp")) for x in b)
    assert {"tcpOutRsts", "tcpInSegs", "udpInDatagrams"} <= c - b
    assert not any(x.startswith("ctx_") for x in c)                  # context only enters in D_all


def test_extras_remove_exactly_what_they_say():
    cfg = ab.build_configs(frame(), COLUMNS)
    d = set(cfg["D_all"])
    assert d - set(cfg["F_no_context"]) == {"ctx_core_bw_max_mean", "ctx_net_bw_max_max"}
    assert d - set(cfg["E_no_history"]) == {"bw_max_lag1", "bw_max_diff1", "bw_max_mean30s", "bw_total_std30s",
                                            "eth_pkts_sum_mean30s", "discards_sum_mean60s"}
    assert set(cfg["G_bandwidth_plus_context"]) == set(cfg["A_bandwidth"]) | {"ctx_core_bw_max_mean", "ctx_net_bw_max_max"}
    assert list(ab.build_configs(frame(), COLUMNS, extras=False)) == [
        "A_bandwidth", "B_plus_packets", "C_plus_tcp_udp", "D_all"]


def test_unclassifiable_history_feature_raises():
    cols = COLUMNS + ["mystery_mean30s"]
    with pytest.raises(ValueError):
        ab.build_configs(pd.DataFrame({c: [0.0, 1.0] for c in cols}), cols)


def test_config_columns_keep_the_original_feature_order():
    cfg = ab.build_configs(frame(), COLUMNS)
    for cols in cfg.values():
        assert cols == sorted(cols, key=COLUMNS.index)

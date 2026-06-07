from pathlib import Path

import pandas as pd

from tsm_data_quality_engine import build_checks, determine_status


def _base_frames(adj_close_values=None):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    raw = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1000, 1100, 1200],
            "adj_close": adj_close_values or [100.5, 102.0, 103.0],
            "data_source": ["unit"] * 3,
            "data_quality_note": ["ok"] * 3,
        }
    )
    enriched = raw.copy()
    for col in [
        "spy_adj_close",
        "smh_adj_close",
        "qqq_adj_close",
        "relative_return_vs_spy_60d",
        "relative_return_vs_smh_60d",
        "relative_return_vs_qqq_60d",
    ]:
        enriched[col] = 1.0
    signals = raw[["date", "open", "high", "low", "close", "volume", "adj_close"]].copy()
    signals["atr_14"] = 2.0
    signals["atr_14_pct"] = signals["atr_14"] / signals["close"]
    signals["score_price_algo_total"] = 60.0
    signals["entry_trigger"] = "NONE"
    signals["trade_action"] = "NO_TRADE"
    signals["risk_pct_2atr"] = 2 * signals["atr_14"] / signals["close"]
    return raw, enriched, signals


def test_data_quality_passes_clean_inputs():
    raw, enriched, signals = _base_frames()
    existing = Path(__file__)

    checks = build_checks(
        raw,
        enriched,
        signals,
        existing,
        existing,
        existing,
        "2024-01-05",
        7,
    )

    assert determine_status(checks) == "PASS"


def test_data_quality_fails_stale_latest_signal():
    raw, enriched, signals = _base_frames()
    existing = Path(__file__)

    checks = build_checks(
        raw,
        enriched,
        signals,
        existing,
        existing,
        existing,
        "2024-01-20",
        7,
    )

    assert determine_status(checks) == "FAIL"
    stale = checks.loc[checks["check"].eq("latest_signal_within_stale_limit")].iloc[0]
    assert bool(stale["passed"]) is False
    assert stale["severity"] == "CRITICAL"


def test_data_quality_warns_when_adjusted_close_equals_close():
    raw, enriched, signals = _base_frames(adj_close_values=[101.0, 102.0, 103.0])
    existing = Path(__file__)

    checks = build_checks(
        raw,
        enriched,
        signals,
        existing,
        existing,
        existing,
        "2024-01-05",
        7,
    )

    assert determine_status(checks) == "WARN"
    failure = checks.loc[checks["check"].eq("corporate_action_adjustment_independent")].iloc[0]
    assert bool(failure["passed"]) is False
    assert failure["severity"] == "WARN"


def test_data_quality_allows_intentionally_disabled_benchmarks():
    raw, enriched, signals = _base_frames()
    enriched = enriched.drop(
        columns=[
            "spy_adj_close",
            "smh_adj_close",
            "qqq_adj_close",
            "relative_return_vs_spy_60d",
            "relative_return_vs_smh_60d",
            "relative_return_vs_qqq_60d",
        ]
    )
    existing = Path(__file__)

    checks = build_checks(
        raw,
        enriched,
        signals,
        existing,
        existing,
        existing,
        "2024-01-05",
        7,
        benchmark_mode="disabled",
    )

    assert determine_status(checks) == "PASS"
    disabled = checks.loc[checks["check"].eq("benchmark_join_columns_disabled")].iloc[0]
    assert bool(disabled["passed"]) is True
    assert disabled["severity"] == "CRITICAL"

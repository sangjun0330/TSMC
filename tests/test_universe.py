from __future__ import annotations

import pandas as pd
import pytest

from run_universe_market_data_update import selected_bar_specs
from run_pooled_universe_update import preferred_source_for_row
from tsm_intraday_provider_adapter import select_intraday_provider
from tsm_pooled_dataset_builder import add_cross_sectional_features
from tsm_pooled_model_engine import filter_latest_predictions_for_output_scope, filter_latest_predictions_to_decision_scope
from tsm_dashboard import with_rule_fallback_universe_latest_predictions
from tsm_core.currency import normalize_ohlcv_to_engine_currency
from tsm_core.universe import (
    DEFAULT_DECISION_UNIVERSE_CONFIG,
    DEFAULT_RESEARCH_UNIVERSE_CONFIG,
    build_universe_scope_audit,
    load_decision_universe_members,
    load_research_universe_members,
    load_universe_members,
    market_region_for_symbol,
    read_universe_config,
    symbol_set,
)


def test_universe_loader_filters_disabled_and_preserves_paths(tmp_path):
    config_path = tmp_path / "universe.csv"
    pd.DataFrame(
        [
            {
                "symbol": "tsm",
                "symbol_group": "semiconductor",
                "symbol_stooq": "TSM.US",
                "symbol_yahoo": "TSM",
                "data_outdir": str(tmp_path / "output" / "TSM"),
                "rule_outdir": str(tmp_path / "rules" / "TSM"),
                "enabled": True,
                "paper_enabled": True,
            },
            {
                "symbol": "NVDA",
                "symbol_group": "semiconductor",
                "symbol_stooq": "NVDA.US",
                "symbol_yahoo": "NVDA",
                "data_outdir": str(tmp_path / "output" / "NVDA"),
                "rule_outdir": str(tmp_path / "rules" / "NVDA"),
                "enabled": False,
                "paper_enabled": True,
            },
        ]
    ).to_csv(config_path, index=False)

    members = load_universe_members(config_path)

    assert [member.symbol for member in members] == ["TSM"]
    assert members[0].paths.signals.name == "tsm_daily_algorithmic_signals.csv"


def test_universe_loader_rejects_missing_required_columns(tmp_path):
    config_path = tmp_path / "bad_universe.csv"
    pd.DataFrame([{"symbol": "TSM"}]).to_csv(config_path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        read_universe_config(config_path)


def test_default_decision_universe_is_top10_plus_kr_memory_names():
    members = load_decision_universe_members(DEFAULT_DECISION_UNIVERSE_CONFIG)
    symbols = symbol_set(members)
    by_symbol = {member.symbol: member for member in members}

    assert len(members) == 12
    assert len(symbols) == 12
    assert {"005930.KS", "000660.KS"}.issubset(symbols)
    assert by_symbol["005930.KS"].listing_currency == "KRW"
    assert by_symbol["005930.KS"].display_currency == "KRW"
    assert by_symbol["005930.KS"].engine_currency == "USD"
    assert by_symbol["005930.KS"].fx_pair == "KRW=X"
    assert by_symbol["005930.KS"].market_region == "KR"
    assert by_symbol["000660.KS"].market_region == "KR"
    assert by_symbol["NVDA"].market_region == "OVERSEAS"


def test_market_region_inference_and_kr_preferred_source_split():
    args = type("Args", (), {"preferred_source": "stooq", "kr_preferred_source": "yahoo"})()
    kr_row = pd.Series({"symbol": "005930.KS", "symbol_yahoo": "005930.KS"})
    us_row = pd.Series({"symbol": "NVDA", "symbol_yahoo": "NVDA"})

    assert market_region_for_symbol("000660.KS", "000660.KS") == "KR"
    assert market_region_for_symbol("NVDA", "NVDA") == "OVERSEAS"
    assert preferred_source_for_row(kr_row, args) == "yahoo"
    assert preferred_source_for_row(us_row, args) == "stooq"


def test_research_universe_contains_top10_without_changing_decision_scope():
    decision = load_decision_universe_members(DEFAULT_DECISION_UNIVERSE_CONFIG)
    research = load_research_universe_members(DEFAULT_DECISION_UNIVERSE_CONFIG, DEFAULT_RESEARCH_UNIVERSE_CONFIG)
    audit = build_universe_scope_audit(decision, research)

    assert symbol_set(decision).issubset(symbol_set(research))
    assert len(research) > len(decision)
    assert audit[audit["is_decision_universe"]]["symbol"].nunique() == 12
    assert set(audit["training_scope"]) == {"universal_research_pool"}


def test_universe_market_data_default_specs_cover_1h_5m_1m():
    args = type("Args", (), {"bar_scope": "both", "model_minute_interval": "5m", "execution_minute_interval": "1m"})()

    assert selected_bar_specs(args) == ["hourly", "minute_model", "minute_execution"]


def test_cross_sectional_features_group_by_market_region_and_date():
    features = pd.DataFrame(
        [
            {"symbol": "NVDA", "market_region": "OVERSEAS", "date": "2026-06-01", "return_20d": 0.10, "close": 120.0, "sma_50": 100.0, "sma_200": 90.0},
            {"symbol": "TSM", "market_region": "OVERSEAS", "date": "2026-06-01", "return_20d": 0.20, "close": 110.0, "sma_50": 100.0, "sma_200": 90.0},
            {"symbol": "005930.KS", "market_region": "KR", "date": "2026-06-02", "return_20d": 0.30, "close": 80.0, "sma_50": 100.0, "sma_200": 90.0},
            {"symbol": "000660.KS", "market_region": "KR", "date": "2026-06-02", "return_20d": 0.40, "close": 130.0, "sma_50": 100.0, "sma_200": 90.0},
        ]
    )

    out = add_cross_sectional_features(features)
    by_symbol = out.set_index("symbol")

    assert by_symbol.loc["005930.KS", "relative_return_vs_universe_median_20d"] == pytest.approx(-0.05)
    assert by_symbol.loc["000660.KS", "relative_return_vs_universe_median_20d"] == pytest.approx(0.05)
    assert by_symbol.loc["NVDA", "relative_return_vs_universe_median_20d"] == pytest.approx(-0.05)
    assert by_symbol.loc["TSM", "relative_return_vs_universe_median_20d"] == pytest.approx(0.05)
    assert by_symbol.loc["005930.KS", "universe_above_sma50_ratio"] == 0.5
    assert by_symbol.loc["NVDA", "universe_above_sma50_ratio"] == 1.0


def test_intraday_provider_auto_falls_back_to_yahoo_without_keys(monkeypatch):
    for key in ["ALPHAVANTAGE_API_KEY", "POLYGON_API_KEY", "EODHD_API_KEY", "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"]:
        monkeypatch.delenv(key, raising=False)

    provider, attempts = select_intraday_provider("auto", "5m")

    assert provider == "yahoo"
    assert attempts[-1].provider == "yahoo"
    assert attempts[-1].selected is True


def test_latest_pooled_predictions_filter_to_top10_decision_scope():
    latest = pd.DataFrame(
        [
            {"symbol": "TSM", "is_decision_universe": True, "training_scope": "universal_research_pool"},
            {"symbol": "RESEARCHX", "is_decision_universe": False, "training_scope": "universal_research_pool"},
        ]
    )

    filtered = filter_latest_predictions_to_decision_scope(latest)

    assert filtered["symbol"].tolist() == ["TSM"]
    assert set(filtered["training_scope"]) == {"universal_research_pool"}


def test_latest_prediction_output_scope_keeps_expanded_universe_when_not_decision_only():
    latest = pd.DataFrame(
        [
            {"symbol": "TSM", "is_decision_universe": True, "training_scope": "universal_research_pool"},
            {"symbol": "RESEARCHX", "is_decision_universe": False, "training_scope": "universal_research_pool"},
        ]
    )

    universe = filter_latest_predictions_for_output_scope(latest, decision_only=False)
    top10 = filter_latest_predictions_for_output_scope(latest, decision_only=True)

    assert universe["symbol"].tolist() == ["TSM", "RESEARCHX"]
    assert top10["symbol"].tolist() == ["TSM"]


def test_dashboard_latest_predictions_do_not_add_rule_fallback_rows():
    latest = pd.DataFrame([{"symbol": "NVDA", "prediction_source": "pooled_model"}])
    decision_config = pd.DataFrame([{"symbol": "NVDA"}, {"symbol": "000660.KS"}])

    out = with_rule_fallback_universe_latest_predictions(latest, decision_config)

    assert out["symbol"].tolist() == ["NVDA"]
    assert "000660.KS" not in set(out["symbol"])


def test_krw_ohlcv_preserves_native_prices_and_normalizes_engine_usd():
    prices = pd.DataFrame(
        [
            {"date": "2026-05-21", "open": 1500.0, "high": 1650.0, "low": 1450.0, "close": 1600.0, "adj_close": 1600.0, "volume": 10},
        ]
    )
    fx = pd.DataFrame([{"date": "2026-05-21", "fx_pair": "KRW=X", "usdkrw": 1600.0, "fx_rate_to_usd": 1 / 1600.0}])

    normalized = normalize_ohlcv_to_engine_currency(
        prices,
        symbol="005930.KS",
        symbol_yahoo="005930.KS",
        listing_currency="KRW",
        display_currency="KRW",
        engine_currency="USD",
        fx_pair="KRW=X",
        fx_rates=fx,
    )

    row = normalized.iloc[0]
    assert row["close_native"] == 1600.0
    assert row["close"] == 1.0
    assert row["close_usd"] == 1.0
    assert row["display_currency"] == "KRW"
    assert row["engine_currency"] == "USD"

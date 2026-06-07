from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest


SYMBOLS = ["NVDA", "TSM", "AVGO", "AMD", "INTC", "MU", "TXN", "LRCX", "AMAT", "QCOM", "005930.KS", "000660.KS"]


def _write_base_fixture(output_dir, rule_dir) -> None:
    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "symbol_group": "memory_storage" if symbol in {"000660.KS", "MU"} else "fabless_ai_analog",
                "display_currency": "KRW" if symbol.endswith(".KS") else "USD",
                "enabled": True,
            }
            for symbol in SYMBOLS
        ]
        + [{"symbol": "SNPS", "symbol_group": "eda_ip", "display_currency": "USD", "enabled": True}]
    ).to_csv(rule_dir / "tsm_decision_universe_config.csv", index=False)

    signal_rows = []
    for idx, symbol in enumerate(SYMBOLS):
        is_krw = symbol.endswith(".KS")
        signal_rows.append(
            {
                "symbol": symbol,
                "symbol_group": "memory_storage" if symbol in {"000660.KS", "MU"} else "fabless_ai_analog",
                "date": "2026-05-29",
                "close": 55.0 if is_krw else 100.0 + idx,
                "close_native": 75000.0 if symbol == "005930.KS" else 180000.0 if symbol == "000660.KS" else 100.0 + idx,
                "close_usd": 55.0 if is_krw else 100.0 + idx,
                "display_currency": "KRW" if is_krw else "USD",
                "fx_rate_to_usd": 0.000733 if is_krw else 1.0,
                "decision_tier": "AVOID_OR_WAIT",
                "suggested_action": "AVOID_OR_WAIT",
                "suggested_weight": 0.0,
                "trade_action": "NO_TRADE",
                "next_check_condition": "WAIT_FOR_NEW_SETUP",
                "risk_pct_2atr": 0.08,
                "stop_price_1_8atr": 50.0 if is_krw else 92.0 + idx,
                "take_profit_2R": 65.0 if is_krw else 118.0 + idx,
                "score_price_algo_total": 55.0,
                "research_signal_score": 60.0,
            }
        )
    pd.DataFrame(signal_rows).to_csv(rule_dir / "tsm_universe_latest_signals.csv", index=False)

    pred_rows = [
        {
            "symbol": symbol,
            "symbol_group": "memory_storage" if symbol in {"000660.KS", "MU"} else "fabless_ai_analog",
            "date": "2026-05-29",
            "is_decision_universe": symbol in SYMBOLS,
            "latest_trade_ready": False,
            "p_success_5d": 0.45,
            "p_stop_hit_5d": 0.42,
            "p_stop_hit_raw_5d": 0.47,
            "p_stop_hit_calibrated_5d": 0.42,
            "p_stop_hit_raw_minus_calibrated_5d": 0.05,
            "p_stop_hit_oos_percentile_5d": 0.72,
            "strict_stop_risk_gap_5d": 0.07,
            "paper_stop_risk_gap_5d": 0.02,
            "expected_r_5d": 0.08,
            "decision_score_5d": 0.07,
            "paper_decision_score_5d": 0.09,
            "threshold_5d": 0.19,
            "p_success_20d": 0.4,
            "p_stop_hit_20d": 0.5,
            "p_stop_hit_raw_20d": 0.58,
            "p_stop_hit_calibrated_20d": 0.5,
            "p_stop_hit_raw_minus_calibrated_20d": 0.08,
            "p_stop_hit_oos_percentile_20d": 0.91,
            "strict_stop_risk_gap_20d": 0.15,
            "paper_stop_risk_gap_20d": 0.10,
            "stop_risk_calibration_warning": "STOP_RAW_CALIBRATED_GAP_GT_5PCT",
            "expected_r_20d": 0.1,
            "decision_score_20d": 0.05,
            "paper_decision_score_20d": 0.1,
            "threshold_20d": 0.2,
            "p_success_60d": 0.55,
            "p_stop_hit_60d": 0.6,
            "p_stop_hit_raw_60d": 0.64,
            "p_stop_hit_calibrated_60d": 0.6,
            "p_stop_hit_raw_minus_calibrated_60d": 0.04,
            "p_stop_hit_oos_percentile_60d": 0.96,
            "strict_stop_risk_gap_60d": 0.25,
            "paper_stop_risk_gap_60d": 0.20,
            "expected_r_60d": 0.22,
            "decision_score_60d": 0.12,
            "paper_decision_score_60d": 0.16,
            "threshold_60d": 0.24,
            "decision_support_allowed": False,
            "paper_decision_support_allowed": False,
            "block_reasons": "LATEST_NOT_TRADE_READY",
            "model_name": "pooled_lgbm_classifier",
            "prediction_source": "pooled_model",
            "live_trading_status": "DISABLED_BY_DESIGN",
        }
        for symbol in SYMBOLS + ["SNPS"]
    ]
    pd.DataFrame(pred_rows).to_csv(rule_dir / "tsm_universe_latest_predictions.csv", index=False)
    pd.DataFrame(
        [
            {"field": "next_day_prediction_asof_date", "value": "2026-05-29"},
            {"field": "next_day_best_model_1d", "value": "empirical_bayes_group_rate"},
            {"field": "next_day_p_up_1d", "value": 0.52},
            {"field": "next_day_p_down_1d", "value": 0.48},
            {"field": "next_day_threshold_1d", "value": 0.49},
            {"field": "next_day_confidence_band_1d", "value": "LOW_POSITIVE"},
            {"field": "next_day_prediction_signal_status", "value": "DISPLAY_ONLY_MODEL_QUALITY_NOT_PASSED"},
            {"field": "next_day_model_quality_status", "value": "NO_BRIER_IMPROVEMENT"},
            {"field": "next_day_prediction_quality_pass_1d", "value": False},
        ]
    ).to_csv(rule_dir / "tsm_next_day_up_latest_snapshot.csv", index=False)
    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "symbol_group": "memory_storage" if symbol in {"000660.KS", "MU"} else "fabless_ai_analog",
                "date": "2026-06-01" if symbol.endswith(".KS") else "2026-05-29",
                "next_day_prediction_asof_date": "2026-06-01" if symbol.endswith(".KS") else "2026-05-29",
                "next_day_prediction_scope": "next_day_up_all",
                "next_day_best_model_1d": "empirical_bayes_group_rate",
                "next_day_p_up_1d": 0.52 if symbol == "TSM" else 0.51,
                "next_day_p_down_1d": 0.48 if symbol == "TSM" else 0.49,
                "next_day_threshold_1d": 0.49,
                "next_day_confidence_band_1d": "LOW_POSITIVE",
                "next_day_prediction_signal_status": "DISPLAY_ONLY_MODEL_QUALITY_NOT_PASSED",
                "next_day_model_quality_status": "NO_BRIER_IMPROVEMENT",
                "next_day_prediction_quality_pass_1d": False,
            }
            for symbol in SYMBOLS
        ]
    ).to_csv(rule_dir / "tsm_next_day_up_universe_latest_predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "symbol_group": "memory_storage" if symbol in {"000660.KS", "MU"} else "fabless_ai_analog",
                "date": "2026-06-01" if symbol.endswith(".KS") else "2026-05-29",
                "next_close_prediction_asof_date": "2026-06-01" if symbol.endswith(".KS") else "2026-05-29",
                "next_close_prediction_scope": "next_close_forecast_top12",
                "next_close_model_quality_status": "DISPLAY_ONLY_FORECAST_MODEL_QUALITY_NOT_PASSED",
                "next_close_best_model_1d": "validation_weighted_blend",
                "next_close_predicted_close_1d": 76000.0 if symbol == "005930.KS" else 102.5,
                "next_close_predicted_close_engine_1d": 55.7 if symbol.endswith(".KS") else 102.5,
                "next_close_predicted_close_usd_1d": 55.7 if symbol.endswith(".KS") else 102.5,
                "next_close_predicted_close_native_1d": 76000.0 if symbol == "005930.KS" else 102.5,
                "next_close_predicted_return_pct_1d": 1.2,
                "next_close_lower_80_1d": 74000.0 if symbol == "005930.KS" else 99.5,
                "next_close_upper_80_1d": 78000.0 if symbol == "005930.KS" else 105.5,
                "next_close_performance_quality_pass_1d": False,
                "next_close_oos_event_count_1d": 5000,
                "next_close_best_model_5d": "validation_weighted_blend",
                "next_close_predicted_close_5d": 78000.0 if symbol == "005930.KS" else 105.0,
                "next_close_predicted_return_pct_5d": 3.5,
                "next_close_lower_80_5d": 73000.0 if symbol == "005930.KS" else 98.0,
                "next_close_upper_80_5d": 83000.0 if symbol == "005930.KS" else 111.0,
                "next_close_performance_quality_pass_5d": False,
                "next_close_oos_event_count_5d": 5000,
                "next_close_best_model_20d": "validation_weighted_blend",
                "next_close_predicted_close_20d": 82000.0 if symbol == "005930.KS" else 110.0,
                "next_close_predicted_return_pct_20d": 7.0,
                "next_close_lower_80_20d": 70000.0 if symbol == "005930.KS" else 94.0,
                "next_close_upper_80_20d": 90000.0 if symbol == "005930.KS" else 120.0,
                "next_close_performance_quality_pass_20d": False,
                "next_close_oos_event_count_20d": 5000,
            }
            for symbol in SYMBOLS
        ]
    ).to_csv(rule_dir / "tsm_next_close_universe_latest_predictions.csv", index=False)

    portfolio_rows = []
    for symbol in SYMBOLS:
        approved = symbol in {"QCOM", "TXN"}
        portfolio_rows.append(
            {
                "symbol": symbol,
                "symbol_group": "fabless_ai_analog",
                "asof_date": "2026-05-29",
                "portfolio_status": "APPROVED" if approved else "REJECTED",
                "rank": 1 if symbol == "QCOM" else 2 if symbol == "TXN" else "",
                "requested_weight": 0.03 if approved else 0.0,
                "approved_weight": 0.025 if symbol == "QCOM" else 0.03 if symbol == "TXN" else 0.0,
                "decision_tier": "BREAKOUT_EXTENSION_TINY" if approved else "AVOID_OR_WAIT",
                "suggested_action": "HIGH_VOL_TINY_EXTENSION" if approved else "AVOID_OR_WAIT",
                "block_reason": "PASS" if approved else "REQUESTED_WEIGHT_ZERO",
                "warning_reasons": "PASS",
                "p_success_20d": 0.4,
                "expected_r_20d": 0.1,
                "p_stop_hit_20d": 0.5,
            }
        )
    pd.DataFrame(portfolio_rows).to_csv(rule_dir / "tsm_portfolio_risk_order_decisions.csv", index=False)
    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "asof_date": "2026-05-29",
                "status": "APPROVED" if symbol in {"QCOM", "TXN"} else "REJECTED",
                "target_weight": 0.03 if symbol in {"QCOM", "TXN"} else 0.0,
                "live_trading_status": "DISABLED_BY_DESIGN",
            }
            for symbol in SYMBOLS
        ]
    ).to_csv(rule_dir / "tsm_order_intents.csv", index=False)

    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "symbol_group": "memory_storage" if symbol in {"000660.KS", "MU"} else "fabless_ai_analog",
                "bar_type": "daily",
                "status": "OK",
                "end_timestamp": "2026-05-29",
                "latest_close_native": 75000.0 if symbol == "005930.KS" else 180000.0 if symbol == "000660.KS" else 100.0,
                "latest_close_usd": 55.0 if symbol.endswith(".KS") else 100.0,
                "display_currency": "KRW" if symbol.endswith(".KS") else "USD",
                "fx_rate_to_usd": 0.000733 if symbol.endswith(".KS") else 1.0,
            }
            for symbol in SYMBOLS
        ]
    ).to_csv(output_dir / "tsm_universe_market_data_latest.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "fx_pair": "KRW=X",
                "base_currency": "USD",
                "quote_currency": "KRW",
                "fx_close": 1508.64,
                "usdkrw": 1508.64,
                "fx_rate_to_usd": 0.0006628486583943154,
                "data_source": "frankfurter_ecb_fx",
                "source_url": "https://api.frankfurter.app/2016-05-12..2026-06-02?from=USD&to=KRW",
                "generated_at_utc": "2026-06-02T08:15:35+00:00",
            }
        ]
    ).to_csv(output_dir / "tsm_fx_latest.csv", index=False)


def _patch_dirs(monkeypatch, tmp_path):
    import tsm_dashboard

    output_dir = tmp_path / "output"
    rule_dir = tmp_path / "rules"
    output_dir.mkdir()
    rule_dir.mkdir()
    _write_base_fixture(output_dir, rule_dir)
    monkeypatch.setattr(tsm_dashboard, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(tsm_dashboard, "RULE_DIR", rule_dir)
    monkeypatch.setattr(tsm_dashboard, "ALLOWED_FILE_ROOTS", [output_dir.resolve(), rule_dir.resolve()])
    return tsm_dashboard


def test_investor_dashboard_returns_operational_12_and_ranks_approved_actions(tmp_path, monkeypatch):
    tsm_dashboard = _patch_dirs(monkeypatch, tmp_path)

    data = tsm_dashboard.build_investor_dashboard()

    assert [row["symbol"] for row in data["symbols"]] == SYMBOLS
    assert "SNPS" not in [row["symbol"] for row in data["symbols"]]
    assert data["action_queue"][0]["symbol"] == "QCOM"
    assert data["action_queue"][0]["signal_profile"]["basis"] == "20d_prediction"
    assert data["action_queue"][0]["signal_profile"]["tier"] == "defensive"
    assert data["action_queue"][0]["action_level"] == "avoid"


def test_investor_dashboard_exposes_usdkrw_exchange_rate(tmp_path, monkeypatch):
    tsm_dashboard = _patch_dirs(monkeypatch, tmp_path)

    data = tsm_dashboard.build_investor_dashboard()

    assert data["fx"]["date"] == "2026-06-01"
    assert data["fx"]["fx_pair"] == "KRW=X"
    assert data["fx"]["base_currency"] == "USD"
    assert data["fx"]["quote_currency"] == "KRW"
    assert data["fx"]["usdkrw"] == 1508.64


def test_koreaexim_fx_refresh_parses_usd_deal_base_rate(tmp_path, monkeypatch):
    tsm_dashboard = _patch_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("KOREAEXIM_AUTHKEY", "test-key")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                [
                    {"cur_unit": "JPY(100)", "deal_bas_r": "950.12", "cur_nm": "일본 옌"},
                    {"cur_unit": "USD", "deal_bas_r": "1,356.78", "ttb": "1,343.21", "tts": "1,370.11", "cur_nm": "미국 달러"},
                ]
            ).encode("utf-8")

    monkeypatch.setattr(tsm_dashboard, "urlopen", lambda request, timeout=20: FakeResponse())

    result = tsm_dashboard.refresh_koreaexim_fx("2026-06-03")

    assert result["fx"]["date"] == "2026-06-03"
    assert result["fx"]["usdkrw"] == 1356.78
    assert result["fx"]["fx_rate_to_usd"] == pytest.approx(1 / 1356.78)
    assert result["fx"]["data_source"] == "koreaexim_current_exchange_api"


def test_koreaexim_fx_refresh_requires_authkey(tmp_path, monkeypatch):
    tsm_dashboard = _patch_dirs(monkeypatch, tmp_path)
    monkeypatch.delenv("KOREAEXIM_AUTHKEY", raising=False)
    monkeypatch.delenv("KOREAEXIM_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="KOREAEXIM_AUTHKEY"):
        tsm_dashboard.fetch_koreaexim_usdkrw("2026-06-03")


def test_koreaexim_fx_refresh_falls_back_to_recent_available_date(tmp_path, monkeypatch):
    tsm_dashboard = _patch_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("KOREAEXIM_AUTHKEY", "test-key")
    monkeypatch.setattr(tsm_dashboard, "_koreaexim_query_dates", lambda search_date=None: ["20260603", "20260602"])

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(request, timeout=20):
        url = request.full_url
        if "searchdate=20260603" in url:
            return FakeResponse([])
        return FakeResponse([{"cur_unit": "USD", "deal_bas_r": "1,350.50", "cur_nm": "미국 달러"}])

    monkeypatch.setattr(tsm_dashboard, "urlopen", fake_urlopen)

    result = tsm_dashboard.fetch_koreaexim_usdkrw()

    assert result["date"] == "2026-06-02"
    assert result["usdkrw"] == 1350.50


def test_investor_dashboard_keeps_blocked_predictions_display_only(tmp_path, monkeypatch):
    tsm_dashboard = _patch_dirs(monkeypatch, tmp_path)

    data = tsm_dashboard.build_investor_dashboard()

    assert data["system"]["model_status"] == "표시 전용"
    assert data["system"]["main_model"] == "20일 예측"
    assert all(row["prediction"]["use_status"] != "판단 가능" for row in data["symbols"])


def test_investor_dashboard_explains_new_buy_block_reasons(tmp_path, monkeypatch):
    tsm_dashboard = _patch_dirs(monkeypatch, tmp_path)

    data = tsm_dashboard.build_investor_dashboard()
    nvda = next(row for row in data["symbols"] if row["symbol"] == "NVDA")
    qcom = next(row for row in data["symbols"] if row["symbol"] == "QCOM")

    assert nvda["action_text"] == "새 매수 금지"
    assert nvda["signal_profile"]["basis"] == "20d_prediction"
    assert nvda["signal_profile"]["tier_label"] == "매수금지"
    assert {row["code"] for row in nvda["new_buy_block_reasons"]} >= {
        "REQUESTED_WEIGHT_ZERO",
        "PORTFOLIO_REJECTED",
        "RULE_AVOID_OR_WAIT",
    }
    assert any(row["code"] == "EXPECTED_R_LOW" for row in qcom["new_buy_block_reasons"])


def test_investor_dashboard_uses_next_day_prediction_as_all_symbol_main_model(tmp_path, monkeypatch):
    tsm_dashboard = _patch_dirs(monkeypatch, tmp_path)

    data = tsm_dashboard.build_investor_dashboard()
    tsm = next(row for row in data["symbols"] if row["symbol"] == "TSM")
    nvda = next(row for row in data["symbols"] if row["symbol"] == "NVDA")
    hynix = next(row for row in data["symbols"] if row["symbol"] == "000660.KS")

    assert tsm["prediction"]["horizon"] == "내일"
    assert tsm["prediction"]["probability_label"] == "상승 확률"
    assert tsm["prediction"]["probability_kind"] == "next_close_up"
    assert tsm["prediction"]["up_probability"] == 0.52
    assert tsm["predictions"]["1d"]["model_name"] == "empirical_bayes_group_rate"
    assert nvda["prediction"]["horizon"] == "내일"
    assert nvda["predictions"]["1d"]["up_probability"] == 0.51
    assert hynix["prediction"]["horizon"] == "내일"
    assert hynix["main_prediction_as_of"] == "2026-06-01"


def test_investor_dashboard_exposes_next_close_forecasts(tmp_path, monkeypatch):
    tsm_dashboard = _patch_dirs(monkeypatch, tmp_path)

    data = tsm_dashboard.build_investor_dashboard()
    nvda = next(row for row in data["symbols"] if row["symbol"] == "NVDA")
    samsung = next(row for row in data["symbols"] if row["symbol"] == "005930.KS")
    hynix = next(row for row in data["symbols"] if row["symbol"] == "000660.KS")

    assert nvda["close_forecasts"]["1d"]["predicted_close"] == 102.5
    assert nvda["close_forecasts"]["5d"]["predicted_return_pct"] == 3.5
    assert nvda["close_forecasts"]["20d"]["lower_80"] == 94.0
    assert samsung["close_forecasts"]["1d"]["predicted_close"] == pytest.approx(55.7 * 1508.64)
    assert samsung["close_forecasts"]["1d"]["predicted_close_usd"] == 55.7
    assert hynix["close_forecasts"]["1d"]["predicted_close"] == pytest.approx(55.7 * 1508.64)
    assert hynix["close_forecasts"]["1d"]["lower_80"] == pytest.approx((99.5 / 102.5) * 55.7 * 1508.64)


def test_investor_dashboard_exposes_raw_calibrated_stop_risk(tmp_path, monkeypatch):
    tsm_dashboard = _patch_dirs(monkeypatch, tmp_path)

    data = tsm_dashboard.build_investor_dashboard()
    nvda = next(row for row in data["symbols"] if row["symbol"] == "NVDA")
    pred5 = nvda["predictions"]["5d"]
    pred20 = nvda["predictions"]["20d"]
    pred60 = nvda["predictions"]["60d"]

    assert pred5["probability_label"] == "성공 확률"
    assert pred5["probability_kind"] == "meta_label_success"
    assert pred5["up_probability"] == 0.45
    assert pred5["down_risk"] == 0.42
    assert pred5["threshold"] == 0.19
    assert pred20["probability_label"] == "성공 확률"
    assert pred20["down_risk"] == 0.5
    assert pred20["down_risk_calibrated"] == 0.5
    assert pred20["down_risk_raw"] == 0.58
    assert pred20["down_risk_oos_percentile"] == 0.91
    assert pred20["strict_stop_risk_gap"] == 0.15
    assert pred20["stop_risk_calibration_warning"] == "STOP_RAW_CALIBRATED_GAP_GT_5PCT"
    assert pred60["up_probability"] == 0.55
    assert pred60["down_risk"] == 0.6
    assert pred60["paper_stop_risk_gap"] == 0.2


def test_investor_dashboard_uses_krw_display_prices(tmp_path, monkeypatch):
    tsm_dashboard = _patch_dirs(monkeypatch, tmp_path)

    data = tsm_dashboard.build_investor_dashboard()
    samsung = next(row for row in data["symbols"] if row["symbol"] == "005930.KS")

    assert samsung["currency"] == "KRW"
    assert samsung["price"] == pytest.approx(55.0 * 1508.64)
    assert samsung["risk"]["stop_price"] == pytest.approx(50.0 * 1508.64)


def test_investor_dashboard_applies_latest_fx_to_krw_display_without_model_rerun(tmp_path, monkeypatch):
    tsm_dashboard = _patch_dirs(monkeypatch, tmp_path)

    first = tsm_dashboard.build_investor_dashboard()
    output_dir = tsm_dashboard.OUTPUT_DIR
    pd.DataFrame(
        [
            {
                "date": "2026-06-02",
                "fx_pair": "KRW=X",
                "base_currency": "USD",
                "quote_currency": "KRW",
                "fx_close": 1400.0,
                "usdkrw": 1400.0,
                "fx_rate_to_usd": 1 / 1400.0,
                "data_source": "test_latest_fx",
            }
        ]
    ).to_csv(output_dir / "tsm_fx_latest.csv", index=False)
    second = tsm_dashboard.build_investor_dashboard()

    first_samsung = next(row for row in first["symbols"] if row["symbol"] == "005930.KS")
    second_samsung = next(row for row in second["symbols"] if row["symbol"] == "005930.KS")
    assert first_samsung["price"] == pytest.approx(55.0 * 1508.64)
    assert second_samsung["price"] == pytest.approx(55.0 * 1400.0)
    assert second_samsung["close_forecasts"]["1d"]["predicted_close"] == pytest.approx(55.7 * 1400.0)


def test_investor_dashboard_prefers_kr2_latest_outputs(tmp_path, monkeypatch):
    tsm_dashboard = _patch_dirs(monkeypatch, tmp_path)
    output_dir = tsm_dashboard.OUTPUT_DIR
    rule_dir = tsm_dashboard.RULE_DIR

    kr2_dir = output_dir / "kr2_automation"
    kr2_dir.mkdir()
    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "symbol_group": "memory_storage",
                "bar_type": bar_type,
                "status": "OK",
                "end_timestamp": "2026-06-01",
                "latest_close_native": 80000.0 if symbol == "005930.KS" else 190000.0,
                "latest_close_usd": 60.0 if symbol == "005930.KS" else 140.0,
                "display_currency": "KRW",
                "engine_currency": "USD",
                "fx_rate_to_usd": 0.00075,
                "generated_at_utc": "2026-06-01T07:35:11+00:00",
            }
            for symbol in ["005930.KS", "000660.KS"]
            for bar_type in ["daily", "hourly", "minute_model", "minute_execution"]
        ]
    ).to_csv(kr2_dir / "tsm_universe_market_data_latest.csv", index=False)

    for symbol in ["005930.KS", "000660.KS"]:
        symbol_rule = rule_dir / "universe" / symbol
        symbol_rule.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "symbol_group": "memory_storage",
                    "date": "2026-06-01",
                    "close": 60.0 if symbol == "005930.KS" else 140.0,
                    "close_native": 80000.0 if symbol == "005930.KS" else 190000.0,
                    "close_usd": 60.0 if symbol == "005930.KS" else 140.0,
                    "display_currency": "KRW",
                    "engine_currency": "USD",
                    "fx_rate_to_usd": 0.00075,
                    "decision_tier": "WATCH_TREND",
                    "suggested_action": "WATCH_FOR_5_20D_SETUP",
                    "trade_action": "REDUCE_OR_DO_NOT_CHASE",
                    "risk_pct_2atr": 0.08,
                    "stop_price_1_8atr": 55.0,
                    "take_profit_2R": 70.0,
                    "score_price_algo_total": 65.0,
                    "research_signal_score": 62.0,
                }
            ]
        ).to_csv(symbol_rule / "tsm_daily_algorithmic_signals.csv", index=False)

    data = tsm_dashboard.build_investor_dashboard()

    assert [row["symbol"] for row in data["symbols"]] == SYMBOLS
    assert data["system"]["latest_data_date"] == "2026-06-01"
    samsung = next(row for row in data["symbols"] if row["symbol"] == "005930.KS")
    hynix = next(row for row in data["symbols"] if row["symbol"] == "000660.KS")
    nvda = next(row for row in data["symbols"] if row["symbol"] == "NVDA")
    assert samsung["as_of"] == "2026-06-01"
    assert samsung["signal_as_of"] == "2026-06-01"
    assert hynix["as_of"] == "2026-06-01"
    assert hynix["signal_as_of"] == "2026-06-01"
    assert nvda["as_of"] == "2026-05-29"
    assert nvda["signal_as_of"] == "2026-05-29"


def test_investor_dashboard_marks_stale_data(tmp_path, monkeypatch):
    tsm_dashboard = _patch_dirs(monkeypatch, tmp_path)

    class FakeDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 6, 20)

    monkeypatch.setattr(tsm_dashboard, "date", FakeDate)

    data = tsm_dashboard.build_investor_dashboard()

    assert data["system"]["data_status"] == "오래됨"

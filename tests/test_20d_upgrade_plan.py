import numpy as np
import pandas as pd

from tsm_core.schemas import validate_no_future_features
from tsm_backtest_feedback_feature_engine import build_feedback_features
from tsm_external_feature_engine import add_revenue_features
from tsm_ml_overlay_backtest import build_non_overlap_event_portfolio
from tsm_prediction_engine import OPTIONAL_NEWS_SIGNAL_DEFAULTS, normalize_optional_signal_columns
from tsm_validation_engine import build_deflated_sharpe_report


def test_missing_news_columns_get_defaulted_before_pooled_load():
    signals = pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "close": [100.0]})
    out = normalize_optional_signal_columns(signals)

    for col, default in OPTIONAL_NEWS_SIGNAL_DEFAULTS.items():
        assert col in out.columns
        assert out[col].iloc[0] == default


def test_tsmc_revenue_features_do_not_backfill_before_release_date():
    base = pd.DataFrame(
        {
            "symbol": ["TSM", "TSM", "TSM"],
            "symbol_group": ["foundry"] * 3,
            "date": pd.to_datetime(["2025-02-07", "2025-02-10", "2025-02-11"]),
        }
    )
    revenue = pd.DataFrame(
        {
            "revenue_period_month": [pd.Timestamp("2025-01-31")],
            "revenue_release_date": [pd.Timestamp("2025-02-10")],
            "tsmc_monthly_revenue_ntd_m": [293288.0],
            "tsmc_monthly_revenue_yoy_pct": [35.9],
            "tsmc_monthly_revenue_mom_pct": [np.nan],
            "tsmc_revenue_yoy_3m_avg_pct": [35.9],
            "tsmc_revenue_12m_cumulative_yoy_pct": [np.nan],
            "tsmc_revenue_source": ["TSMC_OFFICIAL_MONTHLY_REVENUE"],
        }
    )

    out = add_revenue_features(base, revenue, "OK")

    assert pd.isna(out.loc[out["date"].eq(pd.Timestamp("2025-02-07")), "tsmc_monthly_revenue_ntd_m"].iloc[0])
    assert out.loc[out["date"].eq(pd.Timestamp("2025-02-10")), "tsmc_monthly_revenue_ntd_m"].iloc[0] == 293288.0


def test_schema_blocks_new_future_leakage_patterns():
    leaks = validate_no_future_features(["score_price_algo_total", "actual_return_20d", "next_earnings_return", "future_alpha"])

    assert "actual_return_20d" in leaks
    assert "next_earnings_return" in leaks
    assert "future_alpha" in leaks
    assert "score_price_algo_total" not in leaks


def test_backtest_feedback_waits_for_label_availability_date():
    ledger = pd.DataFrame(
        {
            "symbol": ["TSM", "TSM"],
            "symbol_group": ["foundry", "foundry"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-10"]),
            "exit_date": pd.to_datetime(["2024-01-30", "2024-01-31"]),
            "strategy_id": ["S", "S"],
            "strategy_group": ["SG", "SG"],
            "variant_id": ["V1", "V2"],
            "entry_trigger": ["A", "A"],
            "candidate_tier": ["decision_trade_ready", "decision_trade_ready"],
            "net_return_pct": [10.0, -10.0],
            "realized_r_multiple": [1.0, -1.0],
            "stop_hit": [False, True],
        }
    )

    features = build_feedback_features(ledger, pd.DataFrame())

    assert pd.isna(features.loc[0, "fb_trigger_success_rate_ewm_60"])
    assert pd.isna(features.loc[1, "fb_trigger_success_rate_ewm_60"])


def test_oof_feedback_waits_for_20d_label_availability():
    ledger = pd.DataFrame(
        {
            "symbol": ["TSM", "TSM"],
            "symbol_group": ["foundry", "foundry"],
            "date": pd.to_datetime(["2024-01-29", "2024-02-01"]),
            "exit_date": pd.to_datetime(["2024-01-29", "2024-02-01"]),
            "strategy_id": ["S", "S"],
            "strategy_group": ["SG", "SG"],
            "variant_id": ["V1", "V2"],
            "entry_trigger": ["A", "A"],
            "candidate_tier": ["decision_trade_ready", "decision_trade_ready"],
            "net_return_pct": [0.0, 0.0],
            "realized_r_multiple": [0.0, 0.0],
            "stop_hit": [False, False],
        }
    )
    oof = pd.DataFrame(
        {
            "symbol": ["TSM"],
            "date": [pd.Timestamp("2024-01-02")],
            "p_success": [0.25],
            "label_success_20d": [1],
            "label_net_return_pct_20d": [5.0],
            "selected_by_threshold": [True],
        }
    )

    features = build_feedback_features(ledger, oof)

    assert pd.isna(features.loc[0, "fb_model_calibration_residual_ewm_120"])
    assert features.loc[1, "fb_model_calibration_residual_ewm_120"] == 0.75


def test_deflated_sharpe_uses_period_sharpe_and_bootstrap_gate():
    rng = np.random.default_rng(7)
    returns = rng.normal(loc=0.00002, scale=0.01, size=252)
    curves = pd.DataFrame(
        {
            "strategy_id": ["LOW_SR"] * len(returns),
            "strategy_name": ["Low Sharpe"] * len(returns),
            "date": pd.date_range("2024-01-01", periods=len(returns), freq="B"),
            "daily_return": returns,
        }
    )

    report = build_deflated_sharpe_report(curves, trial_count=100)

    assert not bool(report["dsr_pass"].iloc[0])
    assert "observed_period_sharpe" in report.columns
    assert "bootstrap_sharpe_ci_lower_90" in report.columns


def test_non_overlap_event_portfolio_skips_same_symbol_open_event():
    oos = pd.DataFrame(
        {
            "symbol": ["TSM", "TSM"],
            "candidate_scope": ["trade_ready_entry", "trade_ready_entry"],
            "horizon_days": [20, 20],
            "model_name": ["model", "model"],
            "signal_idx": [1, 2],
            "date": pd.to_datetime(["2025-01-02", "2025-01-06"]),
            "label_entry_date": pd.to_datetime(["2025-01-03", "2025-01-07"]),
            "label_exit_date": pd.to_datetime(["2025-01-31", "2025-01-20"]),
            "selected_by_threshold": [True, True],
            "label_net_return_pct": [5.0, 4.0],
            "label_exit_reason": ["HORIZON_20D", "HORIZON_20D"],
        }
    )

    portfolio = build_non_overlap_event_portfolio(oos, event_weight=0.1)
    selected = portfolio[portfolio["policy"].eq("ML_SELECTED_NON_OVERLAP")].sort_values("signal_idx")

    assert bool(selected.iloc[0]["accepted"])
    assert not bool(selected.iloc[1]["accepted"])
    assert selected.iloc[1]["skip_reason"] == "OVERLAPPING_OPEN_EVENT"

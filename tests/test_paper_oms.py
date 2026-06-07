from __future__ import annotations

import pytest
import pandas as pd

from tsm_core.config import AutomationConfig, PaperExecutionConfig, PaperFeedbackConfig, PaperOmsConfig, PortfolioConstructionConfig, PortfolioRiskConfig
from tsm_core.execution import ExecutionBlockReason, IntentStatus, OrderStatus
from tsm_automation_scheduler import build_plan as build_automation_plan, build_quality as build_automation_quality
from tsm_execution_feedback_engine import build_feedback_events, build_label_table
from tsm_fill_model_calibration_engine import build_calibration
from tsm_order_intent_engine import build_latest_intent, upsert_intent, upsert_intents
from tsm_order_state_machine import build_state_events
from tsm_paper_execution_engine import build_order_and_fills, build_positions, filter_symbols, resolve_same_bar_stop_target
from tsm_portfolio_risk_engine import build_decision, build_portfolio_decisions, filter_decision_universe
from tsm_position_reconciler import build_reconciliation


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-20"),
                "open": 99.0,
                "high": 103.0,
                "low": 97.0,
                "close": 100.0,
                "entry_trigger": "BREAKOUT_20D",
                "trade_action": "ENTRY_ALLOWED",
                "research_signal_action": "NO_ACTION",
                "position_weight_if_0_5pct_account_risk": 0.12,
                "paper_tracking_weight": 0.02,
                "atr_14": 5.0,
                "atr_stop_2x": 90.0,
                "beta_vs_spy_252d": 1.0,
                "beta_vs_smh_252d": 0.5,
                "dollar_volume_ma_20": 1_000_000_000.0,
            },
            {
                "date": pd.Timestamp("2026-05-21"),
                "open": 101.0,
                "high": 104.0,
                "low": 99.0,
                "close": 102.0,
                "entry_trigger": "NO_ENTRY_TRIGGER",
                "trade_action": "NO_TRADE",
                "research_signal_action": "NO_ACTION",
                "position_weight_if_0_5pct_account_risk": 0.0,
                "paper_tracking_weight": 0.0,
                "atr_14": 5.0,
                "atr_stop_2x": 92.0,
                "atr_14_pct": 0.05,
                "open_gap_pct": 0.01,
                "beta_vs_spy_252d": 1.0,
                "beta_vs_smh_252d": 0.5,
                "dollar_volume_ma_20": 1_000_000_000.0,
            },
        ]
    )


def _approved_intent(**overrides: object) -> pd.Series:
    base = {
        "intent_id": "intent_test",
        "signal_id": "sig_test",
        "symbol": "TSM",
        "asof_date": "2026-05-20",
        "side": "BUY",
        "order_type": "MARKET",
        "target_weight": 0.10,
        "max_notional": 10_000.0,
        "entry_reference_price": 100.0,
        "stop_price": 90.0,
        "limit_price": None,
        "time_in_force": "DAY",
        "status": IntentStatus.APPROVED.value,
        "reason": "STRICT_RULE_ENTRY",
    }
    base.update(overrides)
    return pd.Series(base)


def _approved_risk(**overrides: object) -> pd.Series:
    base = {
        "intent_id": "intent_test",
        "symbol": "TSM",
        "asof_date": "2026-05-20",
        "portfolio_status": IntentStatus.APPROVED.value,
        "approved_weight": 0.10,
        "block_reason": "PASS",
    }
    base.update(overrides)
    return pd.Series(base)


def test_order_intent_idempotency():
    latest = build_latest_intent(
        _signals().iloc[:1],
        {"risk_state": "ENTRY_RISK_ALLOWED", "final_recommended_max_weight": 0.10},
        {},
        {},
        PaperOmsConfig(),
        "TSM",
    )
    ledger = upsert_intent(pd.DataFrame(), latest)
    ledger = upsert_intent(ledger, latest)

    assert len(ledger) == 1
    assert ledger.iloc[0]["intent_id"] == latest["intent_id"]


def test_batch_order_intent_upsert_replaces_stale_symbol_rows():
    old = pd.DataFrame(
        [
            {
                "intent_id": "old_nvda",
                "signal_id": "old_sig",
                "symbol": "NVDA",
                "asof_date": "2026-05-20",
                "status": IntentStatus.APPROVED.value,
            },
            {
                "intent_id": "old_tsm",
                "signal_id": "old_sig_tsm",
                "symbol": "TSM",
                "asof_date": "2026-05-20",
                "status": IntentStatus.APPROVED.value,
            },
        ]
    )
    latest = {"intent_id": "new_nvda", "signal_id": "new_sig", "symbol": "NVDA", "asof_date": "2026-05-21", "status": IntentStatus.APPROVED.value}

    out = upsert_intents(old, [latest])

    assert out["symbol"].tolist() == ["TSM", "NVDA"]
    assert out[out["symbol"].eq("NVDA")]["intent_id"].tolist() == ["new_nvda"]


def test_order_intent_blocks_when_live_disabled():
    latest = build_latest_intent(
        _signals().iloc[:1],
        {"risk_state": "ENTRY_RISK_ALLOWED", "final_recommended_max_weight": 0.10},
        {},
        {},
        PaperOmsConfig(),
        "TSM",
    )

    assert latest["live_trading_status"] == "DISABLED_BY_DESIGN"
    assert latest["live_order_blocked"] is True
    assert latest["live_block_reason"] == ExecutionBlockReason.LIVE_TRADING_DISABLED.value


def test_order_intent_accepts_v2_tiny_extension_but_keeps_live_disabled():
    signals = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-27"),
                "open": 110.0,
                "high": 125.0,
                "low": 109.0,
                "close": 122.0,
                "entry_trigger": "NONE",
                "trade_action": "REDUCE_OR_DO_NOT_CHASE",
                "research_signal_action": "NO_ACTION",
                "position_weight_if_0_5pct_account_risk": 0.05,
                "paper_tracking_weight": 0.0,
                "atr_14": 5.0,
                "atr_stop_2x": 112.0,
                "decision_tier": "BREAKOUT_EXTENSION_TINY",
                "suggested_action": "HIGH_VOL_TINY_EXTENSION",
                "sizing_tier": "TINY_0_5_3",
                "suggested_weight": 0.03,
                "raw_entry_event": "RAW_60D_BREAKOUT",
                "semi_momentum_regime": "MEMORY_AI_LEADERSHIP",
                "next_check_condition": "HOLD_ONLY_IF_CLOSE_ABOVE_5D_LOW_AND_10D_EMA",
            }
        ]
    )
    latest = build_latest_intent(
        signals,
        {"risk_state": "V2_TINY_TREND_RISK_ALLOWED", "final_recommended_max_weight": 2.0},
        {},
        {},
        PaperOmsConfig(),
        "000660.KS",
    )

    assert latest["status"] == IntentStatus.APPROVED.value
    assert latest["reason"] == "HIGH_VOL_TINY_EXTENSION"
    assert latest["target_weight"] == pytest.approx(0.02)
    assert latest["decision_tier"] == "BREAKOUT_EXTENSION_TINY"
    assert latest["live_trading_status"] == "DISABLED_BY_DESIGN"


def test_order_intent_uses_v2_stop_and_invalidation_levels():
    signals = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-27"),
                "open": 110.0,
                "high": 125.0,
                "low": 109.0,
                "close": 122.0,
                "entry_trigger": "NONE",
                "trade_action": "REDUCE_OR_DO_NOT_CHASE",
                "research_signal_action": "NO_ACTION",
                "position_weight_if_0_5pct_account_risk": 0.05,
                "paper_tracking_weight": 0.0,
                "atr_14": 5.0,
                "atr_stop_2x": 112.0,
                "stop_price_1_8atr": 113.0,
                "invalidation_5d_low": 108.0,
                "invalidation_ema10": 111.0,
                "decision_tier": "BREAKOUT_EXTENSION_TINY",
                "suggested_action": "HIGH_VOL_TINY_EXTENSION",
                "suggested_weight": 0.03,
            }
        ]
    )

    latest = build_latest_intent(
        signals,
        {"risk_state": "V2_TINY_TREND_RISK_ALLOWED", "final_recommended_max_weight": 3.0},
        {},
        {},
        PaperOmsConfig(),
        "000660.KS",
    )

    assert latest["stop_price"] == pytest.approx(113.0)
    assert latest["stop_price_1_8atr"] == pytest.approx(113.0)
    assert latest["invalidation_5d_low"] == pytest.approx(108.0)
    assert latest["invalidation_ema10"] == pytest.approx(111.0)


def test_order_intent_signal_id_includes_symbol_when_missing_source_id():
    tsm = build_latest_intent(
        _signals().iloc[:1],
        {"risk_state": "ENTRY_RISK_ALLOWED", "final_recommended_max_weight": 0.10},
        {},
        {},
        PaperOmsConfig(),
        "TSM",
    )
    nvda = build_latest_intent(
        _signals().iloc[:1],
        {"risk_state": "ENTRY_RISK_ALLOWED", "final_recommended_max_weight": 0.10},
        {},
        {},
        PaperOmsConfig(),
        "NVDA",
    )

    assert tsm["signal_id"] != nvda["signal_id"]


def test_portfolio_risk_caps_weight():
    decision = build_decision(
        _approved_intent(target_weight=0.12, max_notional=12_000.0),
        {"beta_vs_spy_252d": 1.0, "beta_vs_smh_252d": 0.5, "dollar_volume_ma_20": 1_000_000_000.0},
        PortfolioRiskConfig(max_single_name_weight=0.10),
    )

    assert decision["portfolio_status"] == IntentStatus.APPROVED.value
    assert decision["approved_weight"] == 0.10


def test_portfolio_risk_smh_beta_warn_does_not_block():
    decision = build_decision(
        _approved_intent(target_weight=0.10, max_notional=10_000.0),
        {"beta_vs_spy_252d": 1.0, "beta_vs_smh_252d": 1.4, "dollar_volume_ma_20": 1_000_000_000.0},
        PortfolioRiskConfig(max_single_name_weight=0.10, max_beta_to_smh=0.80, smh_beta_limit_mode="WARN"),
    )

    assert decision["portfolio_status"] == IntentStatus.APPROVED.value
    assert decision["approved_weight"] == 0.10
    assert decision["block_reason"] == "PASS"
    assert decision["beta_to_smh_warning"] is True
    assert decision["warning_reasons"] == "BETA_TO_SMH_WARN"


def test_portfolio_risk_spy_beta_still_blocks():
    decision = build_decision(
        _approved_intent(target_weight=0.10, max_notional=10_000.0),
        {"beta_vs_spy_252d": 1.4, "beta_vs_smh_252d": 0.5, "dollar_volume_ma_20": 1_000_000_000.0},
        PortfolioRiskConfig(max_single_name_weight=0.10, max_beta_to_spy=1.20, smh_beta_limit_mode="WARN"),
    )

    assert decision["portfolio_status"] == IntentStatus.REJECTED.value
    assert decision["approved_weight"] == 0.0
    assert "BETA_TO_SPY_LIMIT" in decision["block_reason"]


def test_portfolio_risk_v2_spy_beta_warns_and_reduces_instead_of_blocking():
    decision = build_decision(
        _approved_intent(
            target_weight=0.03,
            max_notional=3_000.0,
            decision_tier="BREAKOUT_EXTENSION_TINY",
            suggested_action="HIGH_VOL_TINY_EXTENSION",
            suggested_weight=0.03,
        ),
        {"beta_vs_spy_252d": 2.4, "beta_vs_smh_252d": 0.5, "dollar_volume_ma_20": 1_000_000_000.0},
        PortfolioRiskConfig(max_single_name_weight=0.10, max_beta_to_spy=1.20),
    )

    assert decision["portfolio_status"] == IntentStatus.APPROVED.value
    assert decision["approved_weight"] == pytest.approx(0.0225)
    assert decision["block_reason"] == "PASS"
    assert decision["beta_to_spy_warning"] is True
    assert "BETA_TO_SPY_WARN" in decision["warning_reasons"]


def test_portfolio_risk_intraday_low_coverage_reduces_caps():
    decision = build_decision(
        _approved_intent(target_weight=0.10, max_notional=10_000.0),
        {
            "beta_vs_spy_252d": 1.0,
            "beta_vs_smh_252d": 0.5,
            "dollar_volume_ma_20": 1_000_000_000.0,
            "timeframe_coverage_score": 0.0,
        },
        PortfolioRiskConfig(max_single_name_weight=0.10, max_order_adv_pct=0.05),
    )

    assert decision["approved_weight"] == pytest.approx(0.075)
    assert decision["max_order_adv_pct"] == pytest.approx(0.0375)
    assert "INTRADAY_COVERAGE_LOW" in decision["warning_reasons"]


def test_portfolio_risk_preserves_v2_decision_context():
    decision = build_decision(
        _approved_intent(
            target_weight=0.03,
            max_notional=3_000.0,
            raw_entry_event="RAW_60D_BREAKOUT",
            decision_tier="BREAKOUT_EXTENSION_TINY",
            sizing_tier="TINY_0_5_3",
            suggested_action="HIGH_VOL_TINY_EXTENSION",
            suggested_weight=0.03,
            semi_momentum_regime="MEMORY_AI_LEADERSHIP",
            stop_price_1_8atr=113.0,
            invalidation_5d_low=108.0,
            invalidation_ema10=111.0,
            next_check_condition="HOLD_ONLY_IF_CLOSE_ABOVE_5D_LOW_AND_10D_EMA",
        ),
        {"beta_vs_spy_252d": 1.0, "beta_vs_smh_252d": 0.5, "dollar_volume_ma_20": 1_000_000_000.0},
        PortfolioRiskConfig(max_single_name_weight=0.10),
    )

    assert decision["decision_tier"] == "BREAKOUT_EXTENSION_TINY"
    assert decision["suggested_action"] == "HIGH_VOL_TINY_EXTENSION"
    assert decision["suggested_weight"] == pytest.approx(0.03)
    assert decision["semi_momentum_regime"] == "MEMORY_AI_LEADERSHIP"
    assert decision["stop_price_1_8atr"] == pytest.approx(113.0)
    assert decision["next_check_condition"] == "HOLD_ONLY_IF_CLOSE_ABOVE_5D_LOW_AND_10D_EMA"


def test_paper_execution_uses_1m_path_for_target_first():
    minute_bars = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-05-21 09:30"), "open": 101.0, "high": 104.0, "low": 100.8, "close": 103.8},
            {"date": pd.Timestamp("2026-05-21 09:31"), "open": 103.8, "high": 104.2, "low": 99.0, "close": 100.0},
        ]
    )
    order, fills, slippage = build_order_and_fills(
        _approved_intent(stop_price=100.0),
        _approved_risk(),
        _signals(),
        PaperOmsConfig(),
        PaperExecutionConfig(half_spread_bps=0.0, volatility_bps_multiplier=0.0, gap_penalty_bps=0.0, participation_bps_multiplier=0.0),
        commission_bps=0.0,
        intraday_context=pd.Series({"execution_minute_feature_status": "OK", "m1_realized_range_pct": 0.001}),
        minute_bars=minute_bars,
    )

    assert order["status"] == OrderStatus.FILLED.value
    assert fills[1]["fill_type"] == "TARGET"
    assert fills[1]["reason"] == "INTRADAY_1M_TARGET_FIRST"
    assert slippage["intraday_path_source"] == "INTRADAY_FEATURE_CONTEXT"


def test_portfolio_risk_rank_cap_allocates_multiple_symbols():
    intents = pd.DataFrame(
        [
            _approved_intent(intent_id="i_nvda", symbol="NVDA", target_weight=0.20, max_notional=20_000.0, paper_decision_score_20d=0.90, p_success_20d=0.70, expected_r_20d=0.80, p_stop_hit_20d=0.20),
            _approved_intent(intent_id="i_tsm", symbol="TSM", target_weight=0.20, max_notional=20_000.0, paper_decision_score_20d=0.80, p_success_20d=0.65, expected_r_20d=0.70, p_stop_hit_20d=0.22),
            _approved_intent(intent_id="i_asml", symbol="ASML", target_weight=0.20, max_notional=20_000.0, paper_decision_score_20d=0.70, p_success_20d=0.60, expected_r_20d=0.60, p_stop_hit_20d=0.25),
            _approved_intent(intent_id="i_amd", symbol="AMD", target_weight=0.20, max_notional=20_000.0, paper_decision_score_20d=0.60, p_success_20d=0.55, expected_r_20d=0.50, p_stop_hit_20d=0.30),
        ]
    )
    contexts = {symbol: {"beta_vs_spy_252d": 1.0, "beta_vs_smh_252d": 0.5, "dollar_volume_ma_20": 1_000_000_000.0} for symbol in ["NVDA", "TSM", "ASML", "AMD"]}
    decisions = build_portfolio_decisions(
        intents,
        contexts,
        PortfolioRiskConfig(max_single_name_weight=0.10),
        PortfolioConstructionConfig(max_open_positions=5, max_total_gross_weight=0.25, max_single_name_weight=0.10, max_group_weight=0.25),
    )

    approved = decisions[decisions["portfolio_status"].eq(IntentStatus.APPROVED.value)].sort_values("rank")
    rejected = decisions[decisions["portfolio_status"].eq(IntentStatus.REJECTED.value)]

    assert approved["symbol"].tolist() == ["NVDA", "TSM", "ASML"]
    assert approved["approved_weight"].tolist() == pytest.approx([0.10, 0.10, 0.05])
    assert rejected.iloc[0]["symbol"] == "AMD"
    assert rejected.iloc[0]["block_reason"] == "PORTFOLIO_CAP_EXHAUSTED"


def test_portfolio_construction_preserves_pre_portfolio_v2_beta_reduction():
    intents = pd.DataFrame(
        [
            _approved_intent(
                intent_id="i_amd",
                symbol="AMD",
                target_weight=0.03,
                max_notional=3_000.0,
                decision_tier="BREAKOUT_EXTENSION_TINY",
                suggested_action="HIGH_VOL_TINY_EXTENSION",
                suggested_weight=0.03,
            )
        ]
    )
    decisions = build_portfolio_decisions(
        intents,
        {"AMD": {"beta_vs_spy_252d": 2.4, "beta_vs_smh_252d": 0.5, "dollar_volume_ma_20": 1_000_000_000.0}},
        PortfolioRiskConfig(max_single_name_weight=0.10, max_beta_to_spy=1.20),
        PortfolioConstructionConfig(max_open_positions=5, max_total_gross_weight=0.35, max_single_name_weight=0.10, max_group_weight=0.35),
    )

    assert decisions.iloc[0]["approved_weight"] == pytest.approx(0.0225)
    assert decisions.iloc[0]["warning_reasons"] == "BETA_TO_SPY_WARN"


def test_portfolio_risk_filters_to_decision_universe_before_approval():
    intents = pd.DataFrame(
        [
            _approved_intent(intent_id="i_tsm", symbol="TSM"),
            _approved_intent(intent_id="i_research", symbol="RESEARCHX"),
        ]
    )

    filtered = filter_decision_universe(intents, {"TSM"})

    assert filtered["symbol"].tolist() == ["TSM"]


def test_paper_execution_filters_existing_rows_to_decision_universe():
    rows = pd.DataFrame(
        [
            {"symbol": "TSM", "order_id": "o1"},
            {"symbol": "RESEARCHX", "order_id": "o2"},
        ]
    )

    filtered = filter_symbols(rows, {"TSM"})

    assert filtered["order_id"].tolist() == ["o1"]


def test_next_open_fill_not_same_day_close():
    order, fills, _ = build_order_and_fills(
        _approved_intent(),
        _approved_risk(),
        _signals(),
        PaperOmsConfig(account_equity=100_000.0),
        PaperExecutionConfig(),
        commission_bps=1.0,
    )

    assert order["status"] == OrderStatus.FILLED.value
    assert fills
    assert fills[0]["fill_date"] == "2026-05-21"
    assert pd.Timestamp(fills[0]["fill_date"]) > pd.Timestamp(order["signal_asof_date"])


def test_limit_order_can_expire_unfilled():
    order, fills, _ = build_order_and_fills(
        _approved_intent(order_type="LIMIT", limit_price=100.0),
        _approved_risk(),
        _signals(),
        PaperOmsConfig(account_equity=100_000.0),
        PaperExecutionConfig(),
        commission_bps=1.0,
    )

    assert order["status"] == OrderStatus.EXPIRED.value
    assert fills == []


def test_stop_gap_through_uses_open():
    reason, price = resolve_same_bar_stop_target({"open": 85.0, "high": 90.0, "low": 80.0}, stop=90.0, target=110.0, side="BUY")

    assert reason == "STOP"
    assert price == 85.0


def test_same_bar_stop_target_uses_stop_first():
    reason, price = resolve_same_bar_stop_target({"open": 100.0, "high": 120.0, "low": 90.0}, stop=95.0, target=110.0, side="BUY")

    assert reason == "STOP"
    assert price == 95.0


def test_paper_position_updates_after_fill():
    fills = pd.DataFrame(
        [
            {
                "symbol": "TSM",
                "side": "BUY",
                "fill_date": "2026-05-21",
                "fill_price": 100.0,
                "quantity": 10.0,
            }
        ]
    )
    positions = build_positions(fills, _signals(), account_equity=100_000.0)

    latest = positions.iloc[0]
    assert latest["position_state"] == "LONG"
    assert latest["quantity"] == 10.0
    assert latest["weight"] > 0


def test_paper_positions_use_account_level_equity_for_multiple_symbols():
    fills = pd.DataFrame(
        [
            {"symbol": "TSM", "side": "BUY", "fill_date": "2026-05-21", "fill_price": 100.0, "quantity": 10.0},
            {"symbol": "NVDA", "side": "BUY", "fill_date": "2026-05-21", "fill_price": 50.0, "quantity": 20.0},
        ]
    )
    tsm_signals = pd.DataFrame([{"date": pd.Timestamp("2026-05-21"), "close": 110.0}])
    nvda_signals = pd.DataFrame([{"date": pd.Timestamp("2026-05-21"), "close": 60.0}])
    positions = build_positions(fills, tsm_signals, account_equity=100_000.0, signals_by_symbol={"TSM": tsm_signals, "NVDA": nvda_signals})

    assert set(positions["symbol"]) == {"TSM", "NVDA"}
    assert positions["equity"].nunique() == 1
    assert positions["market_value"].sum() == pytest.approx(2_300.0)
    assert positions["weight"].sum() == pytest.approx(2_300.0 / 100_300.0)


def test_reconciliation_detects_position_mismatch():
    positions = pd.DataFrame(
        [
            {
                "symbol": "TSM",
                "asof_date": "2026-05-21",
                "quantity": 10.0,
                "market_price": 110.0,
                "market_value": 1_100.0,
            }
        ]
    )
    fills = pd.DataFrame(
        [
            {
                "symbol": "TSM",
                "side": "BUY",
                "fill_date": "2026-05-21",
                "fill_price": 100.0,
                "quantity": 5.0,
            }
        ]
    )
    report = build_reconciliation(positions, fills)

    assert report.iloc[0]["status"] == "FAIL"
    assert report.iloc[0]["details"] == "POSITION_MISMATCH"


def test_order_state_reconciled_after_fill_and_reconciliation_pass():
    order, fills, _ = build_order_and_fills(
        _approved_intent(),
        _approved_risk(),
        _signals(),
        PaperOmsConfig(account_equity=100_000.0),
        PaperExecutionConfig(),
        commission_bps=1.0,
    )
    reconciliation = pd.DataFrame([{"status": "PASS", "details": "PASS", "mismatch_count": 0}])
    events = build_state_events(pd.DataFrame([_approved_intent()]), pd.DataFrame([_approved_risk()]), pd.DataFrame([order]), pd.DataFrame(fills), reconciliation)

    assert events.iloc[-1]["lifecycle_state"] == "RECONCILED"
    assert events.iloc[-1]["live_trading_status"] == "DISABLED_BY_DESIGN"


def test_order_state_rejected_intent_is_not_paper_submitted():
    rejected = _approved_intent(status=IntentStatus.REJECTED.value, reason="LOW_EXPECTANCY")
    events = build_state_events(pd.DataFrame([rejected]), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    assert events.iloc[-1]["lifecycle_state"] == "REJECTED"
    assert events.iloc[-1]["block_reason"] == "INTENT_REJECTED"


def test_execution_feedback_labels_mature_only_after_horizon():
    order, fills, _ = build_order_and_fills(
        _approved_intent(),
        _approved_risk(),
        _signals(),
        PaperOmsConfig(account_equity=100_000.0),
        PaperExecutionConfig(),
        commission_bps=1.0,
    )
    short_horizon = build_feedback_events(pd.DataFrame([order]), pd.DataFrame(fills), _signals(), PaperFeedbackConfig(horizon_days=1))
    long_horizon = build_feedback_events(pd.DataFrame([order]), pd.DataFrame(fills), _signals(), PaperFeedbackConfig(horizon_days=20))
    labels = build_label_table(short_horizon)

    assert bool(short_horizon.iloc[0]["label_matured"]) is True
    assert bool(long_horizon.iloc[0]["label_matured"]) is False
    assert bool(labels.iloc[0]["label_include_for_model"]) is True


def test_execution_feedback_tracks_expired_unfilled_order():
    order, fills, _ = build_order_and_fills(
        _approved_intent(order_type="LIMIT", limit_price=100.0),
        _approved_risk(),
        _signals(),
        PaperOmsConfig(account_equity=100_000.0),
        PaperExecutionConfig(),
        commission_bps=1.0,
    )
    events = build_feedback_events(pd.DataFrame([order]), pd.DataFrame(fills), _signals(), PaperFeedbackConfig(lambda_unfilled=0.5))

    assert bool(events.iloc[0]["expired_unfilled_flag"]) is True
    assert events.iloc[0]["execution_adjusted_utility"] == -0.5


def test_fill_calibration_research_only_when_insufficient_events():
    events = pd.DataFrame(
        [
            {
                "feedback_id": "fb1",
                "fill_model": "next_open_with_spread",
                "filled_flag": True,
                "expired_unfilled_flag": False,
                "label_matured": True,
                "realized_slippage_bps": 5.0,
                "slippage_error_bps": 1.0,
                "live_trading_status": "DISABLED_BY_DESIGN",
            }
        ]
    )
    calibration = build_calibration(events, PaperFeedbackConfig(min_matured_events_for_calibration=30))

    assert calibration.iloc[0]["calibration_status"] == "RESEARCH_ONLY_INSUFFICIENT_EVENTS"
    assert calibration.iloc[0]["matured_event_count"] == 1


def test_automation_plan_has_no_live_submit():
    plan = build_automation_plan(AutomationConfig(), {"lifecycle_state": "RECONCILED"})
    quality = build_automation_quality(plan)

    assert not plan["run_mode"].astype(str).str.contains("live", case=False, na=False).any()
    assert plan["live_trading_status"].eq("DISABLED_BY_DESIGN").all()
    assert bool(quality[quality["check"].eq("automation_no_live_submit_task")].iloc[0]["passed"]) is True


def test_dashboard_summary_contains_paper_oms_tables(tmp_path, monkeypatch):
    import tsm_dashboard

    output_dir = tmp_path / "output"
    rule_dir = tmp_path / "rules"
    output_dir.mkdir()
    rule_dir.mkdir()
    pd.DataFrame(
        [
            {
                "intent_id": "intent_test",
                "signal_id": "sig_test",
                "symbol": "TSM",
                "asof_date": "2026-05-20",
                "status": "APPROVED",
                "live_trading_status": "DISABLED_BY_DESIGN",
            }
        ]
    ).to_csv(rule_dir / "tsm_order_intents.csv", index=False)
    pd.DataFrame([{"field": "portfolio_risk_status", "value": "APPROVED"}]).to_csv(rule_dir / "tsm_portfolio_risk_snapshot.csv", index=False)
    pd.DataFrame([{"symbol": "TSM", "asof_date": "2026-05-21", "quantity": 0, "position_state": "CASH"}]).to_csv(rule_dir / "tsm_paper_positions.csv", index=False)
    pd.DataFrame([{"symbol": "TSM", "asof_date": "2026-05-21", "status": "PASS", "live_trading_status": "DISABLED_BY_DESIGN"}]).to_csv(rule_dir / "tsm_paper_reconciliation_report.csv", index=False)

    monkeypatch.setattr(tsm_dashboard, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(tsm_dashboard, "RULE_DIR", rule_dir)
    monkeypatch.setattr(tsm_dashboard, "ALLOWED_FILE_ROOTS", [output_dir.resolve(), rule_dir.resolve()])

    summary = tsm_dashboard.build_summary()

    assert summary["snapshots"]["order_intent"]["intent_id"] == "intent_test"
    assert "order_intents" in summary["tables"]
    assert "paper_orders" in summary["tables"]
    assert "paper_fills" in summary["tables"]
    assert "paper_positions" in summary["tables"]
    assert "paper_slippage" in summary["tables"]
    assert "paper_reconciliation" in summary["tables"]
    assert "order_state_events" in summary["tables"]
    assert "execution_feedback_events" in summary["tables"]
    assert "execution_feedback_labels" in summary["tables"]
    assert "fill_model_calibration" in summary["tables"]
    assert "automation_plan" in summary["tables"]

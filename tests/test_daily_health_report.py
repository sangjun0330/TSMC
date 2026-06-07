import pandas as pd

from tsm_daily_health_report import asof_alignment_summary, build_snapshot, component_row, derive_component_rows, derive_overall_status


def test_daily_health_warns_when_prediction_is_blocked_but_research_and_paper_ready():
    blocks = derive_component_rows(
        data_quality={"data_quality_status": "PASS", "decision_support_data_gate": "PASS", "block_reasons": "PASS"},
        system_state={
            "research_ready": True,
            "paper_ready": True,
            "research_block_reasons": "PASS",
            "paper_block_reasons": "PASS",
            "prediction_block_reasons": "NO_MODEL",
            "paper_trading_status": "RULE_BASED_READY_PREDICTION_DISPLAY_ONLY",
        },
        prediction={"prediction_use_status": "DISPLAY_ONLY_NO_MODEL_CANDIDATE", "prediction_signal_status": "NO_MODEL_CANDIDATE"},
        model_gate={"model_gate_status": "BLOCKED", "failed_gate_groups": "local_latest", "failed_gate_count": 1},
        risk={"risk_state": "WAIT_FOR_TRIGGER", "final_recommended_max_weight": 2.0},
        shadow_quality=pd.DataFrame([{"check": "shadow_no_order_execution", "passed": True, "severity": "CRITICAL"}]),
        operational_quality=pd.DataFrame([{"check": "data_quality_gate_not_fail", "passed": True}]),
    )

    assert derive_overall_status(blocks) == "WARN"
    prediction = blocks.loc[blocks["component"].eq("prediction_decision_support")].iloc[0]
    assert prediction["status"] == "BLOCKED"
    assert "NO_MODEL" in prediction["block_reasons"]


def test_daily_health_treats_model_quality_pass_latest_blocked_as_model_gate_pass():
    blocks = derive_component_rows(
        data_quality={"data_quality_status": "PASS", "decision_support_data_gate": "PASS", "block_reasons": "PASS"},
        system_state={
            "research_ready": True,
            "paper_ready": True,
            "research_block_reasons": "PASS",
            "paper_block_reasons": "PASS",
            "prediction_block_reasons": "LATEST_NOT_TRADE_READY",
            "paper_trading_status": "RULE_BASED_READY_PREDICTION_DISPLAY_ONLY",
        },
        prediction={"prediction_use_status": "DISPLAY_ONLY_RULE_FILTERED", "prediction_signal_status": "LATEST_RULE_FILTERED_NOT_TRADE_READY"},
        model_gate={
            "model_gate_status": "PASS_MODEL_QUALITY_SIGNAL_STANDBY",
            "pooled_system_quality_pass": True,
            "pooled_latest_signal_pass": False,
            "blocking_failed_gate_count": 0,
            "blocking_failed_gate_groups": "PASS",
            "warning_failed_gate_count": 700,
            "performance_gate_status": "PASS_ACTIVE_MODEL_PERFORMANCE_DIAGNOSTIC_WARNINGS_ONLY",
            "active_performance_failed_gate_count": 0,
            "active_performance_blocking_failed_gate_count": 0,
            "active_metric_performance_failed_gate_count": 0,
            "diagnostic_performance_warning_gate_count": 619,
            "active_performance_failed_gate_groups": "PASS",
            "next_required_performance_action": "PASS_ACTIVE_MODEL_PERFORMANCE",
            "performance_gate_interpretation": "ACTIVE_PERFORMANCE_PASS_DIAGNOSTIC_WARNINGS_NONBLOCKING",
        },
        risk={"risk_state": "WAIT_FOR_TRIGGER", "final_recommended_max_weight": 2.0},
        shadow_quality=pd.DataFrame([{"check": "shadow_no_order_execution", "passed": True, "severity": "CRITICAL"}]),
        operational_quality=pd.DataFrame([{"check": "data_quality_gate_not_fail", "passed": True}]),
    )

    assert derive_overall_status(blocks) == "WARN"
    model_gate = blocks.loc[blocks["component"].eq("model_gate")].iloc[0]
    performance_gate = blocks.loc[blocks["component"].eq("model_performance_gate")].iloc[0]
    prediction = blocks.loc[blocks["component"].eq("prediction_decision_support")].iloc[0]
    assert model_gate["status"] == "PASS"
    assert model_gate["block_reasons"] == "PASS"
    assert "blocking_failed_gates=0" in model_gate["details"]
    assert performance_gate["status"] == "PASS"
    assert performance_gate["block_reasons"] == "PASS"
    assert "active_failed=0" in performance_gate["details"]
    assert "active_metric_failed=0" in performance_gate["details"]
    assert "diagnostic_warnings=619" in performance_gate["details"]
    assert "ACTIVE_PERFORMANCE_PASS_DIAGNOSTIC_WARNINGS_NONBLOCKING" in performance_gate["details"]
    assert prediction["status"] == "BLOCKED"


def test_daily_health_fails_on_active_model_performance_gate_failure():
    blocks = derive_component_rows(
        data_quality={"data_quality_status": "PASS", "decision_support_data_gate": "PASS", "block_reasons": "PASS"},
        system_state={
            "research_ready": True,
            "paper_ready": True,
            "research_block_reasons": "PASS",
            "paper_block_reasons": "PASS",
            "prediction_block_reasons": "PASS",
            "paper_trading_status": "PREDICTION_PAPER_ALPHA_READY",
        },
        prediction={"prediction_use_status": "DECISION_SUPPORT_ALLOWED", "prediction_signal_status": "PREDICTION_CONFIRMED"},
        model_gate={
            "model_gate_status": "PASS",
            "pooled_system_quality_pass": True,
            "blocking_failed_gate_count": 0,
            "blocking_failed_gate_groups": "PASS",
            "performance_gate_status": "BLOCKED_ACTIVE_MODEL_PERFORMANCE",
            "active_performance_failed_gate_count": 2,
            "active_performance_blocking_failed_gate_count": 1,
            "active_performance_failed_gate_groups": "pooled_model_quality",
            "next_required_performance_action": "REPAIR_ACTIVE_MODEL_PERFORMANCE",
        },
        risk={"risk_state": "WAIT_FOR_TRIGGER", "final_recommended_max_weight": 2.0},
        shadow_quality=pd.DataFrame([{"check": "shadow_no_order_execution", "passed": True, "severity": "CRITICAL"}]),
        operational_quality=pd.DataFrame([{"check": "data_quality_gate_not_fail", "passed": True}]),
    )

    assert derive_overall_status(blocks) == "FAIL"
    performance_gate = blocks.loc[blocks["component"].eq("model_performance_gate")].iloc[0]
    assert performance_gate["status"] == "FAIL"
    assert performance_gate["block_reasons"] == "pooled_model_quality"
    assert "active_blocking=1" in performance_gate["details"]


def test_daily_health_snapshot_propagates_rank_policy_diagnostics():
    blocks = pd.DataFrame(
        [
            {"component": "model_performance_gate", "status": "PASS", "block_reasons": "PASS", "details": ""},
        ]
    )
    snapshot = build_snapshot(
        blocks,
        {"system_state": "RESEARCH_READY", "rank_policy_diagnostic_pass_count": 3, "prediction_performance_block_reasons": "PASS"},
        {"data_quality_status": "PASS"},
        {"prediction_use_status": "DISPLAY_ONLY_NO_ENTRY_TRIGGER"},
        {
            "model_gate_status": "PASS_MODEL_QUALITY_SIGNAL_STANDBY",
            "pooled_model_quality_pass": True,
            "pooled_system_quality_pass": True,
            "pooled_latest_signal_pass": False,
            "pooled_latest_block_reasons": "LATEST_NOT_TRADE_READY|POOLED_STOP_RISK_GT_0_35",
            "pooled_decision_block_reasons": "LATEST_NOT_TRADE_READY|POOLED_EXPECTED_R_LT_0_35",
            "paper_gate_status": "DISPLAY_ONLY_NO_LATEST_TRADE_READY",
            "paper_gate_block_reasons": "LATEST_NOT_TRADE_READY|POOLED_STOP_RISK_GT_0_40",
            "rank_uplift_diagnostic_count": 78,
            "rank_uplift_positive_diagnostic_count": 51,
            "rank_policy_diagnostic_pass_count": 20,
            "rank_policy_best_se_lower_pct": 3.096,
            "rank_policy_supported_performance_warning_count": 159,
            "rank_policy_supported_threshold_warning_count": 86,
            "rank_policy_supported_threshold_metric_warning_count": 62,
            "unresolved_diagnostic_metric_performance_warning_gate_count": 298,
            "classified_diagnostic_performance_warning_count": 619,
            "unresolved_diagnostic_performance_warning_count": 0,
            "performance_warning_resolution_status": "PASS_ALL_NONACTIVE_WARNINGS_CLASSIFIED",
            "next_required_evidence_action": "await_latest_trade_ready_signal",
        },
        pd.DataFrame(
            [
                {
                    "check": "decision_score_at_or_above_threshold",
                    "passed": False,
                    "gap_to_pass": 0.1694,
                    "recommended_action": "wait_for_score_to_clear_locked_oof_threshold",
                },
                {
                    "check": "stop_risk_within_strict_limit",
                    "passed": False,
                    "gap_to_pass": 0.21492,
                    "recommended_action": "wait_for_stop_risk_to_fall_below_strict_limit",
                },
                {
                    "check": "expected_r_at_or_above_min",
                    "passed": False,
                    "gap_to_pass": 0.372126,
                    "recommended_action": "wait_for_expected_r_to_recover_above_minimum",
                },
                {
                    "check": "paper_model_gate_pass",
                    "passed": True,
                    "gap_to_pass": "",
                    "recommended_action": "PASS",
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "symbol": "TSM",
                    "date": "2026-05-29",
                    "is_decision_universe": True,
                    "is_event_candidate": True,
                    "is_actionable_entry_candidate": False,
                    "is_trade_ready_entry_candidate": False,
                    "entry_gate_status": "NO_TRIGGER_WATCHLIST",
                    "score_price_algo_total": 68.15,
                },
                {
                    "symbol": "QCOM",
                    "date": "2026-05-29",
                    "is_decision_universe": True,
                    "is_event_candidate": True,
                    "is_actionable_entry_candidate": True,
                    "is_trade_ready_entry_candidate": False,
                    "entry_gate_status": "NO_ENTRY_TRIGGER",
                    "score_price_algo_total": 66.32,
                },
                {
                    "symbol": "OLD",
                    "date": "2026-05-28",
                    "is_decision_universe": False,
                    "is_event_candidate": True,
                    "is_actionable_entry_candidate": True,
                    "is_trade_ready_entry_candidate": True,
                    "entry_gate_status": "STRICT_ENTRY_ALLOWED",
                    "score_price_algo_total": 90.0,
                },
            ]
        ),
    )
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert values["prediction_block_category"] == "LATEST_MARKET_STATE_BLOCKED"
    assert values["latest_market_state_status"] == "LATEST_SIGNAL_BLOCKED"
    assert values["pooled_model_quality_pass"] is True
    assert values["pooled_system_quality_pass"] is True
    assert values["pooled_latest_signal_pass"] is False
    assert values["pooled_latest_block_reasons"] == "LATEST_NOT_TRADE_READY|POOLED_STOP_RISK_GT_0_35"
    assert values["paper_gate_status"] == "DISPLAY_ONLY_NO_LATEST_TRADE_READY"
    assert values["paper_gate_block_reasons"] == "LATEST_NOT_TRADE_READY|POOLED_STOP_RISK_GT_0_40"
    assert values["next_required_evidence_action"] == "await_latest_trade_ready_signal"
    assert values["rank_uplift_diagnostic_count"] == 78
    assert values["rank_uplift_positive_diagnostic_count"] == 51
    assert values["rank_policy_diagnostic_pass_count"] == 20
    assert values["rank_policy_best_se_lower_pct"] == 3.096
    assert values["rank_policy_supported_performance_warning_count"] == 159
    assert values["rank_policy_supported_threshold_warning_count"] == 86
    assert values["rank_policy_supported_threshold_metric_warning_count"] == 62
    assert values["unresolved_diagnostic_metric_performance_warning_gate_count"] == 298
    assert values["classified_diagnostic_performance_warning_count"] == 619
    assert values["unresolved_diagnostic_performance_warning_count"] == 0
    assert values["performance_warning_resolution_status"] == "PASS_ALL_NONACTIVE_WARNINGS_CLASSIFIED"
    assert values["prediction_performance_block_reasons"] == "PASS"
    assert values["latest_blocker_failed_check_count"] == 3
    assert values["latest_blocker_numeric_gap_count"] == 3
    assert values["latest_score_gap_to_threshold"] == 0.1694
    assert values["latest_stop_risk_gap_to_strict_limit"] == 0.21492
    assert values["latest_expected_r_gap_to_min"] == 0.372126
    assert "decision_score_at_or_above_threshold=0.169400" in values["latest_blocker_numeric_gap_summary"]
    assert "wait_for_score_to_clear_locked_oof_threshold" in values["latest_blocker_recommended_actions"]
    assert values["latest_universe_asof_date"] == "2026-05-29"
    assert values["latest_universe_asof_by_region"] == "OVERSEAS:2026-05-29:3"
    assert values["latest_universe_symbol_count"] == 3
    assert values["latest_decision_universe_symbol_count"] == 2
    assert values["latest_event_candidate_count"] == 3
    assert values["latest_actionable_candidate_count"] == 2
    assert values["latest_trade_ready_candidate_count"] == 1
    assert "QCOM:NO_ENTRY_TRIGGER:66.32" in values["latest_actionable_candidate_summary"]
    assert "OLD:STRICT_ENTRY_ALLOWED:90.00" in values["latest_trade_ready_candidate_summary"]


def test_daily_health_warns_when_market_data_latest_lags_prediction_asof():
    pooled_features = pd.DataFrame(
        [
            {"symbol": "TSM", "date": "2026-05-29"},
            {"symbol": "NVDA", "date": "2026-05-29"},
        ]
    )
    market_latest = pd.DataFrame(
        [
            {"symbol": "TSM", "bar_type": "daily", "end_timestamp": "2026-05-27", "generated_at_utc": "2026-05-28T08:53:24+00:00"},
            {"symbol": "NVDA", "bar_type": "daily", "end_timestamp": "2026-05-27", "generated_at_utc": "2026-05-28T08:53:25+00:00"},
        ]
    )

    alignment = asof_alignment_summary(pooled_features, market_latest)
    blocks = pd.DataFrame(
        [
            {"component": "model_performance_gate", "status": "PASS", "block_reasons": "PASS", "details": ""},
            component_row("asof_alignment", alignment["asof_alignment_status"], alignment["asof_alignment_block_reasons"], ""),
        ]
    )
    snapshot = build_snapshot(
        blocks,
        {"system_state": "RESEARCH_READY"},
        {"data_quality_status": "PASS"},
        {"prediction_use_status": "DISPLAY_ONLY_NO_ENTRY_TRIGGER"},
        {},
        pd.DataFrame(),
        pooled_features,
        alignment,
    )
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert alignment["asof_alignment_status"] == "WARN"
    assert alignment["asof_alignment_block_reasons"] == "MARKET_DATA_LATEST_LAGS_PREDICTION_ASOF"
    assert values["daily_health_status"] == "WARN"
    assert values["prediction_asof_date_for_alignment"] == "2026-05-29"
    assert values["market_data_latest_daily_max_date"] == "2026-05-27"
    assert values["market_data_latest_lag_days"] == 2
    assert "asof_alignment" in values["failed_or_blocked_components"]


def test_daily_health_allows_korea_one_day_ahead_when_each_symbol_is_fresh():
    pooled_features = pd.DataFrame(
        [
            {"symbol": "TSM", "market_region": "OVERSEAS", "date": "2026-06-01"},
            {"symbol": "NVDA", "market_region": "OVERSEAS", "date": "2026-06-01"},
            {"symbol": "005930.KS", "market_region": "KR", "date": "2026-06-02"},
            {"symbol": "000660.KS", "market_region": "KR", "date": "2026-06-02"},
        ]
    )
    market_latest = pd.DataFrame(
        [
            {"symbol": "TSM", "market_region": "OVERSEAS", "bar_type": "daily", "end_timestamp": "2026-06-01"},
            {"symbol": "NVDA", "market_region": "OVERSEAS", "bar_type": "daily", "end_timestamp": "2026-06-01"},
            {"symbol": "005930.KS", "market_region": "KR", "bar_type": "daily", "end_timestamp": "2026-06-02"},
            {"symbol": "000660.KS", "market_region": "KR", "bar_type": "daily", "end_timestamp": "2026-06-02"},
        ]
    )

    alignment = asof_alignment_summary(pooled_features, market_latest)

    assert alignment["asof_alignment_status"] == "PASS"
    assert alignment["market_data_latest_lag_days"] == 0
    assert alignment["prediction_asof_date_for_alignment"] == "2026-06-02"
    assert alignment["prediction_asof_by_region"] == "KR:2026-06-02:2|OVERSEAS:2026-06-01:2"
    assert alignment["market_data_latest_daily_max_by_region"] == "KR:2026-06-02:2|OVERSEAS:2026-06-01:2"


def test_daily_health_fails_on_data_quality_fail():
    blocks = derive_component_rows(
        data_quality={"data_quality_status": "FAIL", "decision_support_data_gate": "BLOCK", "block_reasons": "latest_signal_within_stale_limit"},
        system_state={"research_ready": True, "paper_ready": True, "research_block_reasons": "PASS", "paper_block_reasons": "PASS"},
        prediction={"prediction_use_status": "DECISION_SUPPORT_ALLOWED", "prediction_signal_status": "PREDICTION_CONFIRMED"},
        model_gate={"model_gate_status": "PASS", "failed_gate_groups": "PASS", "failed_gate_count": 0},
        risk={"risk_state": "ENTRY_RISK_ALLOWED", "final_recommended_max_weight": 5.0},
        shadow_quality=pd.DataFrame([{"check": "shadow_no_order_execution", "passed": True, "severity": "CRITICAL"}]),
        operational_quality=pd.DataFrame([{"check": "data_quality_gate_not_fail", "passed": False}]),
    )
    snapshot = build_snapshot(blocks, {"system_state": "RESEARCH_BLOCKED_DATA_QUALITY"}, {"data_quality_status": "FAIL"}, {"prediction_use_status": "DECISION_SUPPORT_ALLOWED"})
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert derive_overall_status(blocks) == "FAIL"
    assert values["daily_health_status"] == "FAIL"
    assert "data_quality" in values["failed_or_blocked_components"]

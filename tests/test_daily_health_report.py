import pandas as pd

from tsm_daily_health_report import build_snapshot, derive_component_rows, derive_overall_status


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

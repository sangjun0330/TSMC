#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily-data system state and readiness engine for TSM.

This engine does not approve live trading. It creates a conservative readiness
score for research/paper-trading operation using only existing daily outputs.

Outputs:
- tsm_system_readiness_scorecard.csv
- tsm_latest_system_state.csv
- tsm_system_readiness_report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from tsm_core.decision_schema import is_prediction_decision_support


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def load_csv(path: Path, required_cols: Iterable[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = strip_bom_columns(pd.read_csv(path))
    if required_cols and not df.empty:
        require_columns(df, required_cols, str(path))
    return df


def load_snapshot(path: Path) -> Dict[str, str]:
    df = load_csv(path, required_cols=["field", "value"])
    if df.empty:
        return {}
    return dict(zip(df["field"], df["value"]))


def bool_series_all_true(df: pd.DataFrame, col: str = "passed") -> bool:
    if df.empty or col not in df.columns:
        return False
    return df[col].astype(str).str.lower().isin(["true", "1", "yes"]).all()


def critical_checks_pass(df: pd.DataFrame) -> bool:
    if df.empty or "passed" not in df.columns:
        return False
    target = df[df["severity"].astype(str).str.upper().eq("CRITICAL")] if "severity" in df.columns else df
    if target.empty:
        target = df
    return bool_series_all_true(target)


def numeric(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def prediction_pipeline_ready_from_model_gate(model_gate_snapshot: Dict[str, str] | None) -> bool:
    if not model_gate_snapshot:
        return False
    if "prediction_pipeline_ready" in model_gate_snapshot:
        return to_bool(model_gate_snapshot.get("prediction_pipeline_ready"))
    model_quality_pass = to_bool(model_gate_snapshot.get("pooled_system_quality_pass", False))
    active_failed = int(numeric(model_gate_snapshot.get("active_performance_failed_gate_count", 0), 0))
    active_blocking = int(numeric(model_gate_snapshot.get("active_performance_blocking_failed_gate_count", 0), 0))
    model_blocking = int(numeric(model_gate_snapshot.get("model_quality_blocking_failed_gate_count", 0), 0))
    performance_status = str(model_gate_snapshot.get("performance_gate_status", "UNKNOWN"))
    return bool(model_quality_pass and active_failed == 0 and active_blocking == 0 and model_blocking == 0 and performance_status.startswith("PASS"))


def domain_score(passed: bool, weight: float, note: str) -> Dict:
    return {"passed": bool(passed), "weight": weight, "score": weight if passed else 0.0, "note": note}


def select_evidence_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, str, float, float]:
    """Prefer alpha-research evidence, then live-like evidence, and only then diagnostic fallback."""
    if frame.empty:
        return frame, "missing", np.nan, np.nan
    candidate = frame.copy()
    if "strategy_group" in candidate.columns and candidate["strategy_group"].astype(str).eq("alpha_research").any():
        return candidate[candidate["strategy_group"].astype(str).eq("alpha_research")].copy(), "alpha-research", 3.0, -25.0
    if "strategy_group" in candidate.columns and candidate["strategy_group"].astype(str).eq("live_like").any():
        return candidate[candidate["strategy_group"].astype(str).eq("live_like")].copy(), "live-like", 1.0, -15.0
    if "strategy_id" in candidate.columns and candidate["strategy_id"].astype(str).str.startswith("L").any():
        return candidate[candidate["strategy_id"].astype(str).str.startswith("L")].copy(), "live-like-id", 1.0, -15.0
    return candidate, "diagnostic-fallback", 0.0, -60.0


def build_scorecard(
    operational_quality: pd.DataFrame,
    integrity_quality: pd.DataFrame,
    validation_quality: pd.DataFrame,
    prediction_quality: pd.DataFrame,
    prediction_snapshot: Dict[str, str],
    risk_snapshot: Dict[str, str],
    stress_snapshot: Dict[str, str],
    backtest_summary: pd.DataFrame,
    walk_forward: pd.DataFrame,
    causal_walk_forward: pd.DataFrame | None = None,
    paper_oms_quality: pd.DataFrame | None = None,
    paper_reconciliation_quality: pd.DataFrame | None = None,
    order_state_quality: pd.DataFrame | None = None,
    execution_feedback_quality: pd.DataFrame | None = None,
    fill_calibration_quality: pd.DataFrame | None = None,
    automation_quality: pd.DataFrame | None = None,
    model_gate_snapshot: Dict[str, str] | None = None,
) -> pd.DataFrame:
    quality_ok = bool_series_all_true(operational_quality)
    if integrity_quality.empty or "severity" not in integrity_quality.columns:
        integrity_ok = False
    else:
        critical = integrity_quality[integrity_quality["severity"] == "CRITICAL"]
        integrity_ok = bool_series_all_true(critical if not critical.empty else integrity_quality)
    validation_ok = bool_series_all_true(validation_quality)
    if prediction_quality.empty or "passed" not in prediction_quality.columns:
        prediction_ok = False
        prediction_note = "Missing prediction quality checks."
    else:
        critical = prediction_quality[prediction_quality["severity"] == "CRITICAL"] if "severity" in prediction_quality.columns else prediction_quality
        critical_ok = bool_series_all_true(critical if not critical.empty else prediction_quality)
        use_status = prediction_snapshot.get("prediction_use_status", "UNKNOWN")
        latest_decision_support = is_prediction_decision_support(use_status)
        pipeline_ready = prediction_pipeline_ready_from_model_gate(model_gate_snapshot)
        if not critical_ok:
            prediction_ok = False
            prediction_note = "Prediction engine critical checks failed; research readiness is scored separately."
        elif pipeline_ready:
            prediction_ok = True
            if latest_decision_support:
                prediction_note = "Prediction pipeline and latest entry signal both pass decision-support gates."
            else:
                prediction_note = f"Prediction pipeline passes active model gates; latest entry signal is standby, use_status={use_status}."
        elif latest_decision_support:
            prediction_ok = True
            prediction_note = "Prediction overlay passed quality gates and may be used as decision support."
        else:
            prediction_ok = False
            prediction_note = f"Prediction overlay is display-only; use_status={use_status}."

    risk_state = risk_snapshot.get("risk_state", "UNKNOWN")
    risk_ok = risk_state not in {"NO_NEW_RISK", "UNKNOWN"}
    if risk_state == "OBSERVATION_OR_TINY_SIZE_ONLY":
        risk_note = "Risk engine allows only observation/tiny sizing."
    elif risk_ok:
        risk_note = "Risk engine allows nonzero research sizing."
    else:
        risk_note = "Risk engine blocks new risk."

    stress_status = stress_snapshot.get("stress_status", "UNKNOWN")
    stress_ok = stress_status in {"LOW_STRESS_FOR_CURRENT_SIZE", "MODERATE_STRESS_FOR_CURRENT_SIZE"}

    if backtest_summary.empty:
        best_strategy_ok = False
        best_note = "Missing backtest summary."
    else:
        eligible, basis, min_cagr, mdd_floor = select_evidence_frame(backtest_summary)
        best = eligible.sort_values("cagr_pct", ascending=False).iloc[0]
        best_strategy_ok = numeric(best.get("cagr_pct")) > min_cagr and numeric(best.get("max_drawdown_pct")) > mdd_floor
        best_note = f"Best {basis} strategy {best.get('strategy_id')} CAGR={best.get('cagr_pct')}, MDD={best.get('max_drawdown_pct')}."

    wf_source = causal_walk_forward if causal_walk_forward is not None and not causal_walk_forward.empty else walk_forward
    if wf_source.empty:
        wf_ok = False
        wf_note = "Missing walk-forward summary."
    else:
        wf, wf_basis, wf_min_cagr, _ = select_evidence_frame(wf_source)
        if causal_walk_forward is not None and not causal_walk_forward.empty:
            wf_basis = f"causal {wf_basis}"
        positive_rate = wf.groupby("strategy_id")["test_positive"].apply(
            lambda x: x.astype(str).str.lower().isin(["true", "1", "yes"]).mean()
        ).max()
        median_test = wf.groupby("strategy_id")["test_cagr_pct"].median().max()
        wf_ok = positive_rate >= 0.50 and median_test > wf_min_cagr
        wf_note = f"Best {wf_basis} positive test rate={positive_rate * 100:.1f}%, best median test CAGR={median_test:.2f}%."

    domains = {
        "operational_quality": domain_score(quality_ok, 10, "All operational file/date/data checks pass."),
        "daily_integrity": domain_score(integrity_ok, 15, "Daily OHLC, formula, signal, risk, trade, and curve contracts pass."),
        "validation_quality": domain_score(validation_ok, 15, "Look-ahead/cost/overlap quality checks pass."),
        "prediction_decision_support": domain_score(prediction_ok, 15, prediction_note),
        "risk_policy": domain_score(risk_ok, 10, risk_note),
        "stress_tolerance": domain_score(stress_ok, 10, f"Stress status={stress_status}."),
        "full_period_backtest": domain_score(best_strategy_ok, 15, best_note),
        "walk_forward_evidence": domain_score(wf_ok, 10, wf_note),
    }
    if (
        paper_oms_quality is not None
        or paper_reconciliation_quality is not None
        or order_state_quality is not None
        or execution_feedback_quality is not None
        or fill_calibration_quality is not None
        or automation_quality is not None
    ):
        paper_oms_ok = critical_checks_pass(paper_oms_quality if paper_oms_quality is not None else pd.DataFrame())
        paper_recon_ok = critical_checks_pass(paper_reconciliation_quality if paper_reconciliation_quality is not None else pd.DataFrame())
        order_state_ok = critical_checks_pass(order_state_quality if order_state_quality is not None else pd.DataFrame())
        execution_feedback_ok = critical_checks_pass(execution_feedback_quality if execution_feedback_quality is not None else pd.DataFrame())
        fill_calibration_ok = critical_checks_pass(fill_calibration_quality if fill_calibration_quality is not None else pd.DataFrame())
        automation_ok = critical_checks_pass(automation_quality if automation_quality is not None else pd.DataFrame())
        all_paper_ops_ok = all([paper_oms_ok, paper_recon_ok, order_state_ok, execution_feedback_ok, fill_calibration_ok, automation_ok])
        domains["paper_oms_quality"] = domain_score(
            all_paper_ops_ok,
            0,
            "Paper OMS checks="
            f"{'PASS' if paper_oms_ok else 'FAIL'}, reconciliation={'PASS' if paper_recon_ok else 'FAIL'}, "
            f"lifecycle={'PASS' if order_state_ok else 'FAIL'}, feedback={'PASS' if execution_feedback_ok else 'FAIL'}, "
            f"calibration={'PASS' if fill_calibration_ok else 'FAIL'}, automation={'PASS' if automation_ok else 'FAIL'}.",
        )

    rows = []
    for domain, values in domains.items():
        rows.append(
            {
                "domain": domain,
                "passed": values["passed"],
                "weight": values["weight"],
                "score": values["score"],
                "note": values["note"],
            }
        )
    return pd.DataFrame(rows)


def readiness_state(total_score: float, scorecard: pd.DataFrame, risk_snapshot: Dict[str, str]) -> str:
    passed_by_domain = dict(zip(scorecard["domain"], scorecard["passed"]))
    if not passed_by_domain.get("operational_quality", False):
        return "RESEARCH_BLOCKED_OPERATIONAL_QUALITY"
    if not passed_by_domain.get("daily_integrity", False):
        return "RESEARCH_BLOCKED_INTEGRITY"
    if not passed_by_domain.get("validation_quality", False):
        return "RESEARCH_BLOCKED_VALIDATION"
    if not passed_by_domain.get("prediction_decision_support", False):
        return "RESEARCH_READY_PREDICTION_BLOCKED"
    if not passed_by_domain.get("full_period_backtest", False) or not passed_by_domain.get("walk_forward_evidence", False):
        return "RESEARCH_READY_ALPHA_NOT_READY"
    if total_score >= 85 and risk_snapshot.get("risk_state") == "ENTRY_RISK_ALLOWED":
        return "PAPER_READY_ENTRY_ALLOWED"
    if total_score >= 75:
        return "PAPER_READY_WAITING_FOR_SIGNAL"
    if total_score >= 60:
        return "RESEARCH_READY_NOT_PAPER_READY"
    return "RESEARCH_ONLY_REVIEW_REQUIRED"


def readiness_component_score(scorecard: pd.DataFrame, domains: Dict[str, float]) -> float:
    if scorecard.empty:
        return 0.0
    rows = scorecard.set_index("domain")
    total_weight = float(sum(domains.values()))
    if total_weight <= 0:
        return 0.0
    earned = 0.0
    for domain, weight in domains.items():
        if domain in rows.index and bool(rows.loc[domain, "passed"]):
            earned += float(weight)
    return earned / total_weight * 100.0


def readiness_block_reasons(scorecard: pd.DataFrame, domain_map: Dict[str, str]) -> str:
    if scorecard.empty:
        return "MISSING_SCORECARD"
    failed = []
    passed_by_domain = dict(zip(scorecard["domain"], scorecard["passed"]))
    for domain, reason in domain_map.items():
        if not bool(passed_by_domain.get(domain, False)):
            failed.append(reason)
    return "|".join(failed) if failed else "PASS"


def active_performance_block_reason_from_model_gate(model_gate_snapshot: Dict[str, str] | None) -> str:
    if not model_gate_snapshot:
        return "UNKNOWN"
    performance_status = str(model_gate_snapshot.get("performance_gate_status", "UNKNOWN"))
    active_failed = int(numeric(model_gate_snapshot.get("active_performance_failed_gate_count", 0), 0))
    active_blocking = int(numeric(model_gate_snapshot.get("active_performance_blocking_failed_gate_count", 0), 0))
    if performance_status.startswith("PASS") and active_failed == 0 and active_blocking == 0:
        return "PASS"
    reasons = str(model_gate_snapshot.get("active_performance_failed_gate_groups", "") or "").strip()
    if reasons and reasons.upper() != "PASS":
        return reasons
    return performance_status if performance_status and performance_status != "UNKNOWN" else "UNKNOWN"


def latest_block_reason_from_model_gate(model_gate_snapshot: Dict[str, str] | None) -> str:
    if not model_gate_snapshot:
        return ""
    for field in ("pooled_latest_block_reasons", "pooled_decision_block_reasons", "paper_gate_block_reasons"):
        value = str(model_gate_snapshot.get(field, "") or "").strip()
        if value and value.upper() not in {"PASS", "UNKNOWN", "NAN"}:
            return f"POOLED_LATEST:{value}"
    return ""


def latest_signal_block_reason_from_snapshot(prediction_snapshot: Dict[str, str]) -> str:
    for field in ("latest_signal_block_reasons", "paper_alpha_block_reasons"):
        value = str(prediction_snapshot.get(field, "") or "").strip()
        if value and value.upper() != "PASS":
            return value
    signal_status = str(prediction_snapshot.get("prediction_signal_status", "") or "").strip()
    if signal_status:
        return signal_status
    use_status = str(prediction_snapshot.get("prediction_use_status", "") or "").strip()
    return use_status if use_status else "PREDICTION_NOT_DECISION_SUPPORT"


def prediction_block_reason_from_snapshot(
    prediction_snapshot: Dict[str, str],
    prediction_ready: bool,
    model_gate_snapshot: Dict[str, str] | None = None,
) -> str:
    if prediction_ready:
        return "PASS"
    performance_reasons = active_performance_block_reason_from_model_gate(model_gate_snapshot)
    if performance_reasons == "PASS":
        pooled_latest_reason = latest_block_reason_from_model_gate(model_gate_snapshot)
        if pooled_latest_reason:
            return pooled_latest_reason
        use_status = str(prediction_snapshot.get("prediction_use_status", "") or "").upper()
        signal_status = str(prediction_snapshot.get("prediction_signal_status", "") or "").upper()
        if use_status.startswith("DISPLAY_ONLY") or signal_status.startswith("NO_ENTRY_TRIGGER"):
            return f"LATEST_SIGNAL:{latest_signal_block_reason_from_snapshot(prediction_snapshot)}"
    reasons = []
    pooled_quality_pass = to_bool(
        prediction_snapshot.get(
            "pooled_model_quality_pass",
            model_gate_snapshot.get("pooled_model_quality_pass", False) if model_gate_snapshot else False,
        )
    )

    def add_reason(value: object, prefix: str = "") -> None:
        text = str(value or "").strip()
        if not text or text.upper() == "PASS":
            return
        reasons.append(f"{prefix}{text}" if prefix else text)

    local = str(prediction_snapshot.get("model_quality_block_reasons", "") or "").strip()
    if local and not pooled_quality_pass:
        add_reason(local)
    add_reason(prediction_snapshot.get("pooled_model_quality_block_reasons", ""), "POOLED:")
    add_reason(prediction_snapshot.get("pooled_decision_block_reasons", ""), "POOLED_LATEST:")
    if pooled_quality_pass and not reasons:
        add_reason(prediction_snapshot.get("pooled_latest_block_reasons", ""), "POOLED_LATEST:")
    return "|".join(reasons) if reasons else "PREDICTION_NOT_DECISION_SUPPORT"


def build_latest_state(
    scorecard: pd.DataFrame,
    risk_snapshot: Dict[str, str],
    stress_snapshot: Dict[str, str],
    prediction_snapshot: Dict[str, str],
    model_gate_snapshot: Dict[str, str] | None = None,
) -> pd.DataFrame:
    model_gate_snapshot = model_gate_snapshot or {}
    composite_gate_score = float(scorecard["score"].sum())
    passed_by_domain = dict(zip(scorecard["domain"], scorecard["passed"]))
    research_score = readiness_component_score(
        scorecard,
        {
            "operational_quality": 10,
            "daily_integrity": 15,
            "validation_quality": 15,
        },
    )
    state = readiness_state(composite_gate_score, scorecard, risk_snapshot)
    prediction_pipeline_ready = bool(passed_by_domain.get("prediction_decision_support", False))
    prediction_score = 100.0 if prediction_pipeline_ready else 0.0
    alpha_score = readiness_component_score(
        scorecard,
        {
            "full_period_backtest": 35,
            "walk_forward_evidence": 35,
            "prediction_decision_support": 20,
            "risk_policy": 5,
            "stress_tolerance": 5,
        },
    )
    paper_score = min(
        100.0,
        research_score * 0.50
        + (20.0 if passed_by_domain.get("risk_policy", False) else 0.0)
        + (20.0 if passed_by_domain.get("stress_tolerance", False) else 0.0)
        + (10.0 if passed_by_domain.get("prediction_decision_support", False) else 0.0),
    )
    live_score = 0.0
    research_ready = bool(
        passed_by_domain.get("operational_quality", False)
        and passed_by_domain.get("daily_integrity", False)
        and passed_by_domain.get("validation_quality", False)
    )
    prediction_decision_support = is_prediction_decision_support(prediction_snapshot.get("prediction_use_status", "UNKNOWN"))
    paper_prediction_decision_support = to_bool(
        prediction_snapshot.get(
            "pooled_paper_decision_support_allowed",
            prediction_snapshot.get("paper_decision_support_allowed", False),
        )
    )
    paper_oms_ready = bool(passed_by_domain.get("paper_oms_quality", True))
    alpha_ready = bool(alpha_score >= 75.0 and prediction_pipeline_ready)
    prediction_ready = bool(prediction_score >= 100.0)
    paper_ready = bool(((paper_score >= 75.0 and research_ready) or paper_prediction_decision_support) and paper_oms_ready)
    live_ready = False
    research_block_reasons = readiness_block_reasons(
        scorecard,
        {
            "operational_quality": "OPERATIONAL_QUALITY_FAILED",
            "daily_integrity": "DAILY_INTEGRITY_FAILED",
            "validation_quality": "VALIDATION_QUALITY_FAILED",
        },
    )
    alpha_block_reasons = readiness_block_reasons(
        scorecard,
        {
            "full_period_backtest": "ALPHA_RESEARCH_BACKTEST_WEAK",
            "walk_forward_evidence": "ALPHA_RESEARCH_WALK_FORWARD_WEAK",
            "prediction_decision_support": "PREDICTION_NOT_DECISION_SUPPORT",
        },
    )
    prediction_block_reasons = prediction_block_reason_from_snapshot(prediction_snapshot, prediction_ready, model_gate_snapshot)
    prediction_performance_block_reasons = active_performance_block_reason_from_model_gate(model_gate_snapshot)
    latest_block_reasons = latest_block_reason_from_model_gate(model_gate_snapshot)
    if not latest_block_reasons and prediction_block_reasons.startswith(("LATEST_SIGNAL:", "POOLED_LATEST:")):
        latest_block_reasons = prediction_block_reasons
    model_quality_pass = to_bool(model_gate_snapshot.get("pooled_model_quality_pass", False))
    system_quality_pass = to_bool(model_gate_snapshot.get("pooled_system_quality_pass", False))
    latest_signal_pass = to_bool(model_gate_snapshot.get("pooled_latest_signal_pass", False))
    if prediction_decision_support:
        latest_market_state_status = "DECISION_SUPPORT_ALLOWED"
    elif system_quality_pass and not latest_signal_pass:
        latest_market_state_status = "LATEST_SIGNAL_BLOCKED"
    elif not system_quality_pass:
        latest_market_state_status = "MODEL_QUALITY_BLOCKED"
    else:
        latest_market_state_status = "PREDICTION_NOT_DECISION_SUPPORT"
    live_block_reasons = "NO_LIVE_BROKER_BY_DESIGN"
    rows = [
        {"field": "system_readiness_score", "value": research_score},
        {"field": "composite_gate_score", "value": composite_gate_score},
        {"field": "research_readiness_score", "value": research_score},
        {"field": "alpha_readiness_score", "value": alpha_score},
        {"field": "prediction_readiness_score", "value": prediction_score},
        {"field": "paper_readiness_score", "value": paper_score},
        {"field": "live_readiness_score", "value": live_score},
        {"field": "system_state", "value": state},
        {"field": "research_ready", "value": research_ready},
        {"field": "alpha_ready", "value": alpha_ready},
        {"field": "prediction_ready", "value": prediction_ready},
        {"field": "prediction_pipeline_ready", "value": prediction_pipeline_ready},
        {"field": "live_ready", "value": live_ready},
        {"field": "prediction_decision_support", "value": prediction_decision_support},
        {"field": "paper_prediction_decision_support", "value": paper_prediction_decision_support},
        {"field": "paper_oms_ready", "value": paper_oms_ready},
        {"field": "paper_ready", "value": paper_ready},
        {"field": "research_block_reasons", "value": research_block_reasons},
        {"field": "alpha_block_reasons", "value": alpha_block_reasons},
        {"field": "prediction_block_reasons", "value": prediction_block_reasons},
        {"field": "prediction_performance_block_reasons", "value": prediction_performance_block_reasons},
        {"field": "paper_block_reasons", "value": "PASS" if paper_ready else ("PAPER_OMS_QUALITY_FAILED" if not paper_oms_ready else "PAPER_READINESS_SCORE_LT_75")},
        {"field": "live_block_reasons", "value": live_block_reasons},
        {"field": "risk_state", "value": risk_snapshot.get("risk_state", "UNKNOWN")},
        {"field": "final_recommended_max_weight_pct", "value": risk_snapshot.get("final_recommended_max_weight", "NA")},
        {"field": "stress_status", "value": stress_snapshot.get("stress_status", "UNKNOWN")},
        {"field": "prediction_signal_status", "value": prediction_snapshot.get("prediction_signal_status", "UNKNOWN")},
        {"field": "prediction_use_status", "value": prediction_snapshot.get("prediction_use_status", "UNKNOWN")},
        {"field": "latest_market_state_status", "value": latest_market_state_status},
        {"field": "prediction_latest_block_reasons", "value": latest_block_reasons or "PASS"},
        {"field": "pooled_model_quality_pass", "value": model_quality_pass},
        {"field": "pooled_system_quality_pass", "value": system_quality_pass},
        {"field": "pooled_latest_signal_pass", "value": latest_signal_pass},
        {"field": "pooled_latest_block_reasons", "value": model_gate_snapshot.get("pooled_latest_block_reasons", "UNKNOWN")},
        {"field": "pooled_decision_block_reasons", "value": model_gate_snapshot.get("pooled_decision_block_reasons", "UNKNOWN")},
        {
            "field": "paper_gate_status",
            "value": model_gate_snapshot.get(
                "paper_gate_status",
                prediction_snapshot.get("pooled_paper_gate_status", prediction_snapshot.get("paper_gate_status", "UNKNOWN")),
            ),
        },
        {"field": "paper_gate_block_reasons", "value": model_gate_snapshot.get("paper_gate_block_reasons", "UNKNOWN")},
        {"field": "prediction_scope_used", "value": prediction_snapshot.get("prediction_scope_used", "UNKNOWN")},
        {"field": "prediction_entry_gate_status", "value": prediction_snapshot.get("latest_entry_gate_status", "UNKNOWN")},
        {"field": "model_gate_status", "value": model_gate_snapshot.get("model_gate_status", "UNKNOWN")},
        {"field": "performance_gate_status", "value": model_gate_snapshot.get("performance_gate_status", "UNKNOWN")},
        {"field": "active_performance_failed_gate_count", "value": model_gate_snapshot.get("active_performance_failed_gate_count", 0)},
        {"field": "active_performance_blocking_failed_gate_count", "value": model_gate_snapshot.get("active_performance_blocking_failed_gate_count", 0)},
        {"field": "active_metric_performance_failed_gate_count", "value": model_gate_snapshot.get("active_metric_performance_failed_gate_count", 0)},
        {"field": "metric_performance_failed_gate_count", "value": model_gate_snapshot.get("metric_performance_failed_gate_count", 0)},
        {"field": "performance_evidence_gap_count", "value": model_gate_snapshot.get("performance_evidence_gap_count", 0)},
        {"field": "aggregate_performance_quality_flag_count", "value": model_gate_snapshot.get("aggregate_performance_quality_flag_count", 0)},
        {"field": "diagnostic_performance_warning_gate_count", "value": model_gate_snapshot.get("diagnostic_performance_warning_gate_count", 0)},
        {"field": "diagnostic_metric_performance_warning_gate_count", "value": model_gate_snapshot.get("diagnostic_metric_performance_warning_gate_count", 0)},
        {"field": "rank_policy_supported_performance_warning_count", "value": model_gate_snapshot.get("rank_policy_supported_performance_warning_count", 0)},
        {"field": "rank_policy_supported_threshold_warning_count", "value": model_gate_snapshot.get("rank_policy_supported_threshold_warning_count", 0)},
        {"field": "rank_policy_supported_metric_warning_count", "value": model_gate_snapshot.get("rank_policy_supported_metric_warning_count", 0)},
        {"field": "rank_policy_supported_threshold_metric_warning_count", "value": model_gate_snapshot.get("rank_policy_supported_threshold_metric_warning_count", 0)},
        {"field": "evidence_limited_metric_warning_count", "value": model_gate_snapshot.get("evidence_limited_metric_warning_count", 0)},
        {"field": "rejected_model_metric_warning_count", "value": model_gate_snapshot.get("rejected_model_metric_warning_count", 0)},
        {"field": "strategy_diagnostic_metric_warning_count", "value": model_gate_snapshot.get("strategy_diagnostic_metric_warning_count", 0)},
        {
            "field": "unresolved_diagnostic_metric_performance_warning_gate_count",
            "value": model_gate_snapshot.get("unresolved_diagnostic_metric_performance_warning_gate_count", 0),
        },
        {
            "field": "classified_diagnostic_performance_warning_count",
            "value": model_gate_snapshot.get("classified_diagnostic_performance_warning_count", 0),
        },
        {
            "field": "unresolved_diagnostic_performance_warning_count",
            "value": model_gate_snapshot.get("unresolved_diagnostic_performance_warning_count", 0),
        },
        {"field": "uncategorized_root_cause_count", "value": model_gate_snapshot.get("uncategorized_root_cause_count", 0)},
        {"field": "uncategorized_warning_gate_count", "value": model_gate_snapshot.get("uncategorized_warning_gate_count", 0)},
        {"field": "uncategorized_root_causes", "value": model_gate_snapshot.get("uncategorized_root_causes", "PASS")},
        {
            "field": "performance_warning_resolution_status",
            "value": model_gate_snapshot.get("performance_warning_resolution_status", "UNKNOWN"),
        },
        {"field": "performance_failed_gate_groups", "value": model_gate_snapshot.get("performance_failed_gate_groups", "PASS")},
        {"field": "active_performance_failed_gate_groups", "value": model_gate_snapshot.get("active_performance_failed_gate_groups", "PASS")},
        {"field": "metric_performance_failed_families", "value": model_gate_snapshot.get("metric_performance_failed_families", "PASS")},
        {"field": "diagnostic_performance_failed_gate_groups", "value": model_gate_snapshot.get("diagnostic_performance_failed_gate_groups", "PASS")},
        {"field": "next_required_evidence_action", "value": model_gate_snapshot.get("next_required_evidence_action", "UNKNOWN")},
        {"field": "next_required_performance_action", "value": model_gate_snapshot.get("next_required_performance_action", "UNKNOWN")},
        {"field": "performance_gate_interpretation", "value": model_gate_snapshot.get("performance_gate_interpretation", "UNKNOWN")},
        {"field": "rank_uplift_diagnostic_count", "value": model_gate_snapshot.get("rank_uplift_diagnostic_count", 0)},
        {"field": "rank_uplift_positive_diagnostic_count", "value": model_gate_snapshot.get("rank_uplift_positive_diagnostic_count", 0)},
        {"field": "rank_policy_diagnostic_pass_count", "value": model_gate_snapshot.get("rank_policy_diagnostic_pass_count", 0)},
        {"field": "rank_policy_best_se_lower_pct", "value": model_gate_snapshot.get("rank_policy_best_se_lower_pct", "NA")},
        {"field": "live_trading_status", "value": "DISABLED_BY_DESIGN"},
        {
            "field": "paper_trading_status",
            "value": (
                "PREDICTION_PAPER_SIGNAL_READY"
                if paper_prediction_decision_support
                else ("PREDICTION_PIPELINE_READY_WAITING_FOR_SIGNAL" if prediction_ready else "RULE_BASED_READY_PREDICTION_DISPLAY_ONLY")
            )
            if paper_ready
            else "NOT_READY",
        },
    ]
    return pd.DataFrame(rows)


def write_report(outdir: Path, scorecard: pd.DataFrame, latest: pd.DataFrame) -> None:
    latest_map = dict(zip(latest["field"], latest["value"]))
    lines = [
        "# Top10 System Readiness Report",
        "",
        f"- System state: {latest_map.get('system_state', 'NA')}",
        f"- Legacy system readiness score (research stack only): {latest_map.get('system_readiness_score', 'NA')} / 100",
        f"- Composite gate score: {latest_map.get('composite_gate_score', 'NA')} / 100",
        f"- Research readiness score: {latest_map.get('research_readiness_score', 'NA')} / 100",
        f"- Alpha readiness score: {latest_map.get('alpha_readiness_score', 'NA')} / 100",
        f"- Prediction readiness score: {latest_map.get('prediction_readiness_score', 'NA')} / 100",
        f"- Paper readiness score: {latest_map.get('paper_readiness_score', 'NA')} / 100",
        f"- Live readiness score: {latest_map.get('live_readiness_score', 'NA')} / 100",
        f"- Research ready: {latest_map.get('research_ready', 'NA')}",
        f"- Alpha ready: {latest_map.get('alpha_ready', 'NA')}",
        f"- Prediction ready: {latest_map.get('prediction_ready', 'NA')}",
        f"- Prediction pipeline ready: {latest_map.get('prediction_pipeline_ready', 'NA')}",
        f"- Live ready: {latest_map.get('live_ready', 'NA')}",
        f"- Prediction decision support: {latest_map.get('prediction_decision_support', 'NA')}",
        f"- Paper prediction decision support: {latest_map.get('paper_prediction_decision_support', 'NA')}",
        f"- Paper OMS ready: {latest_map.get('paper_oms_ready', 'NA')}",
        f"- Paper gate status: {latest_map.get('paper_gate_status', 'NA')}",
        f"- Paper trading status: {latest_map.get('paper_trading_status', 'NA')}",
        f"- Live trading status: {latest_map.get('live_trading_status', 'NA')}",
        f"- Alpha block reasons: {latest_map.get('alpha_block_reasons', 'NA')}",
        f"- Prediction block reasons: {latest_map.get('prediction_block_reasons', 'NA')}",
        f"- Prediction performance block reasons: {latest_map.get('prediction_performance_block_reasons', 'NA')}",
        f"- Live block reasons: {latest_map.get('live_block_reasons', 'NA')}",
        f"- Risk state: {latest_map.get('risk_state', 'NA')}",
        f"- Stress status: {latest_map.get('stress_status', 'NA')}",
        f"- Prediction status: {latest_map.get('prediction_signal_status', 'NA')}",
        f"- Prediction use status: {latest_map.get('prediction_use_status', 'NA')}",
        f"- Latest market state status: {latest_map.get('latest_market_state_status', 'NA')}",
        f"- Latest prediction block reasons: {latest_map.get('prediction_latest_block_reasons', 'NA')}",
        f"- Pooled model quality pass: {latest_map.get('pooled_model_quality_pass', 'NA')}",
        f"- Pooled system quality pass: {latest_map.get('pooled_system_quality_pass', 'NA')}",
        f"- Pooled latest signal pass: {latest_map.get('pooled_latest_signal_pass', 'NA')}",
        f"- Pooled latest block reasons: {latest_map.get('pooled_latest_block_reasons', 'NA')}",
        f"- Paper gate block reasons: {latest_map.get('paper_gate_block_reasons', 'NA')}",
        f"- Prediction scope: {latest_map.get('prediction_scope_used', 'NA')}",
        f"- Prediction entry gate: {latest_map.get('prediction_entry_gate_status', 'NA')}",
        f"- Model gate status: {latest_map.get('model_gate_status', 'NA')}",
        f"- Performance gate status: {latest_map.get('performance_gate_status', 'NA')}",
        f"- Active performance failed gates: {latest_map.get('active_performance_failed_gate_count', 'NA')}",
        f"- Active performance blocking failed gates: {latest_map.get('active_performance_blocking_failed_gate_count', 'NA')}",
        f"- Active metric performance failed gates: {latest_map.get('active_metric_performance_failed_gate_count', 'NA')}",
        f"- Metric performance failed gates: {latest_map.get('metric_performance_failed_gate_count', 'NA')}",
        f"- Performance evidence gap count: {latest_map.get('performance_evidence_gap_count', 'NA')}",
        f"- Aggregate performance quality flags: {latest_map.get('aggregate_performance_quality_flag_count', 'NA')}",
        f"- Diagnostic performance warnings: {latest_map.get('diagnostic_performance_warning_gate_count', 'NA')}",
        f"- Diagnostic metric performance warnings: {latest_map.get('diagnostic_metric_performance_warning_gate_count', 'NA')}",
        f"- Rank-policy supported metric warnings: {latest_map.get('rank_policy_supported_metric_warning_count', 'NA')}",
        f"- Evidence-limited metric warnings: {latest_map.get('evidence_limited_metric_warning_count', 'NA')}",
        f"- Rejected-model metric warnings: {latest_map.get('rejected_model_metric_warning_count', 'NA')}",
        f"- Strategy diagnostic metric warnings: {latest_map.get('strategy_diagnostic_metric_warning_count', 'NA')}",
        f"- Classified diagnostic performance warnings: {latest_map.get('classified_diagnostic_performance_warning_count', 'NA')}",
        f"- Unresolved diagnostic performance warnings: {latest_map.get('unresolved_diagnostic_performance_warning_count', 'NA')}",
        f"- Uncategorized root causes: {latest_map.get('uncategorized_root_cause_count', 'NA')}",
        f"- Uncategorized warning gate rows: {latest_map.get('uncategorized_warning_gate_count', 'NA')}",
        f"- Uncategorized root cause names: {latest_map.get('uncategorized_root_causes', 'NA')}",
        f"- Performance warning resolution status: {latest_map.get('performance_warning_resolution_status', 'NA')}",
        f"- Next required evidence action: {latest_map.get('next_required_evidence_action', 'NA')}",
        f"- Next required performance action: {latest_map.get('next_required_performance_action', 'NA')}",
        f"- Performance gate interpretation: {latest_map.get('performance_gate_interpretation', 'NA')}",
        "",
        "## Scorecard",
        "",
        "| Domain | Passed | Score | Weight | Note |",
        "|---|---:|---:|---:|---|",
    ]
    for _, row in scorecard.iterrows():
        lines.append(f"| {row['domain']} | {row['passed']} | {row['score']:.1f} | {row['weight']:.1f} | {row['note']} |")

    lines.extend(
        [
            "",
            "## Operating Rule",
            "- Research readiness means the daily-data research stack is internally consistent.",
            "- Alpha readiness additionally requires live-like economic evidence and a trusted 20D prediction pipeline.",
            "- Current entry permission is reported separately as the latest market signal; no-entry days do not fail model readiness.",
            "- Live trading remains disabled by design until broker integration, reconciliation, kill switch, and intraday execution audit exist.",
            "- No readiness score means the next trade will be profitable.",
            "",
            "This report is research tooling, not investment advice.",
        ]
    )
    (outdir / "tsm_system_readiness_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily-data system readiness scorecard.")
    parser.add_argument("--operational-quality", default="tsm_price_rule_output/tsm_operational_quality_checks.csv")
    parser.add_argument("--integrity-quality", default="tsm_price_rule_output/tsm_daily_integrity_checks.csv")
    parser.add_argument("--validation-quality", default="tsm_price_rule_output/tsm_validation_quality_checks.csv")
    parser.add_argument("--prediction-quality", default="tsm_price_rule_output/tsm_prediction_quality_checks.csv")
    parser.add_argument("--prediction-snapshot", default="tsm_price_rule_output/tsm_latest_prediction_snapshot.csv")
    parser.add_argument("--model-gate-snapshot", default="tsm_price_rule_output/tsm_model_gate_snapshot.csv")
    parser.add_argument("--risk-snapshot", default="tsm_price_rule_output/tsm_latest_risk_snapshot.csv")
    parser.add_argument("--stress-snapshot", default="tsm_price_rule_output/tsm_latest_stress_snapshot.csv")
    parser.add_argument("--backtest-summary", default="tsm_price_rule_output/tsm_backtest_strategy_summary.csv")
    parser.add_argument("--walk-forward", default="tsm_price_rule_output/tsm_validation_walk_forward_summary.csv")
    parser.add_argument("--causal-walk-forward", default="tsm_price_rule_output/tsm_validation_causal_walk_forward_summary.csv")
    parser.add_argument("--paper-oms-quality", default="")
    parser.add_argument("--paper-reconciliation-quality", default="")
    parser.add_argument("--order-state-quality", default="")
    parser.add_argument("--execution-feedback-quality", default="")
    parser.add_argument("--fill-calibration-quality", default="")
    parser.add_argument("--automation-quality", default="")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    operational_quality = load_csv(Path(args.operational_quality))
    integrity_quality = load_csv(Path(args.integrity_quality))
    validation_quality = load_csv(Path(args.validation_quality))
    prediction_quality = load_csv(Path(args.prediction_quality))
    prediction_snapshot = load_snapshot(Path(args.prediction_snapshot))
    model_gate_snapshot = load_snapshot(Path(args.model_gate_snapshot))
    risk_snapshot = load_snapshot(Path(args.risk_snapshot))
    stress_snapshot = load_snapshot(Path(args.stress_snapshot))
    backtest_summary = load_csv(Path(args.backtest_summary))
    walk_forward = load_csv(Path(args.walk_forward))
    causal_walk_forward = load_csv(Path(args.causal_walk_forward))
    paper_oms_quality = load_csv(Path(args.paper_oms_quality)) if args.paper_oms_quality else None
    paper_reconciliation_quality = load_csv(Path(args.paper_reconciliation_quality)) if args.paper_reconciliation_quality else None
    order_state_quality = load_csv(Path(args.order_state_quality)) if args.order_state_quality else None
    execution_feedback_quality = load_csv(Path(args.execution_feedback_quality)) if args.execution_feedback_quality else None
    fill_calibration_quality = load_csv(Path(args.fill_calibration_quality)) if args.fill_calibration_quality else None
    automation_quality = load_csv(Path(args.automation_quality)) if args.automation_quality else None

    scorecard = build_scorecard(
        operational_quality,
        integrity_quality,
        validation_quality,
        prediction_quality,
        prediction_snapshot,
        risk_snapshot,
        stress_snapshot,
        backtest_summary,
        walk_forward,
        causal_walk_forward,
        paper_oms_quality,
        paper_reconciliation_quality,
        order_state_quality,
        execution_feedback_quality,
        fill_calibration_quality,
        automation_quality,
        model_gate_snapshot,
    )
    latest = build_latest_state(scorecard, risk_snapshot, stress_snapshot, prediction_snapshot, model_gate_snapshot)

    scorecard.to_csv(outdir / "tsm_system_readiness_scorecard.csv", index=False)
    latest.to_csv(outdir / "tsm_latest_system_state.csv", index=False)
    write_report(outdir, scorecard, latest)

    print("완료: system readiness outputs =", outdir.resolve())
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()

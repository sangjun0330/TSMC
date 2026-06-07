#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate daily health report for the TSM research and paper-trading system.

This report is intentionally broker-free. It summarizes whether the current
daily run is suitable for research, paper/shadow tracking, and prediction
decision support.

Outputs:
- tsm_daily_health_snapshot.csv
- tsm_system_block_reasons.csv
- tsm_daily_health_report.md
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tsm_core.decision_schema import is_prediction_decision_support
from tsm_core.io import strip_bom_columns
from tsm_core.universe import market_region_for_symbol


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return strip_bom_columns(pd.read_csv(path, low_memory=False))


def read_snapshot(path: Path) -> dict[str, object]:
    df = read_csv_if_exists(path)
    if df.empty or not {"field", "value"}.issubset(df.columns):
        return {}
    return dict(zip(df["field"].astype(str), df["value"]))


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def join_unique(values: pd.Series, limit: int = 8) -> str:
    seen: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text.lower() == "nan" or text in seen:
            continue
        seen.append(text)
    if len(seen) > limit:
        return "|".join(seen[:limit] + [f"+{len(seen) - limit}"])
    return "|".join(seen) if seen else "PASS"


def latest_blocker_gap_summary(blocker_audit: pd.DataFrame) -> dict[str, object]:
    defaults: dict[str, object] = {
        "latest_blocker_failed_check_count": 0,
        "latest_blocker_numeric_gap_count": 0,
        "latest_blocker_failed_checks": "PASS",
        "latest_blocker_recommended_actions": "PASS",
        "latest_score_gap_to_threshold": "NA",
        "latest_stop_risk_gap_to_strict_limit": "NA",
        "latest_expected_r_gap_to_min": "NA",
        "paper_stop_risk_gap_to_limit": "NA",
        "latest_blocker_numeric_gap_summary": "PASS",
    }
    if blocker_audit.empty or not {"check", "passed"}.issubset(blocker_audit.columns):
        return defaults
    work = blocker_audit.copy()
    passed = work["passed"].astype(str).str.lower().isin(["true", "1", "yes"])
    failed = work[~passed].copy()
    defaults["latest_blocker_failed_check_count"] = int(len(failed))
    if failed.empty:
        return defaults
    defaults["latest_blocker_failed_checks"] = join_unique(failed["check"])
    if "recommended_action" in failed.columns:
        defaults["latest_blocker_recommended_actions"] = join_unique(failed["recommended_action"], limit=6)

    numeric = failed.copy()
    numeric["gap_to_pass_numeric"] = pd.to_numeric(numeric.get("gap_to_pass", pd.Series(dtype=float)), errors="coerce")
    numeric = numeric[numeric["gap_to_pass_numeric"].notna()]
    defaults["latest_blocker_numeric_gap_count"] = int(len(numeric))
    if not numeric.empty:
        defaults["latest_blocker_numeric_gap_summary"] = "|".join(
            f"{row['check']}={row['gap_to_pass_numeric']:.6f}" for _, row in numeric.iterrows()
        )
    gap_by_check = dict(zip(numeric["check"].astype(str), numeric["gap_to_pass_numeric"]))
    defaults["latest_score_gap_to_threshold"] = gap_by_check.get("decision_score_at_or_above_threshold", "NA")
    defaults["latest_stop_risk_gap_to_strict_limit"] = gap_by_check.get("stop_risk_within_strict_limit", "NA")
    defaults["latest_expected_r_gap_to_min"] = gap_by_check.get("expected_r_at_or_above_min", "NA")
    defaults["paper_stop_risk_gap_to_limit"] = gap_by_check.get("stop_risk_within_paper_limit", "NA")
    return defaults


def latest_universe_scope_summary(pooled_features: pd.DataFrame) -> dict[str, object]:
    defaults: dict[str, object] = {
        "latest_universe_asof_date": "NA",
        "latest_universe_asof_by_region": "NA",
        "latest_universe_symbol_count": 0,
        "latest_decision_universe_symbol_count": 0,
        "latest_event_candidate_count": 0,
        "latest_actionable_candidate_count": 0,
        "latest_trade_ready_candidate_count": 0,
        "latest_actionable_candidate_summary": "PASS",
        "latest_trade_ready_candidate_summary": "PASS",
    }
    if pooled_features.empty or not {"symbol", "date"}.issubset(pooled_features.columns):
        return defaults
    work = pooled_features.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    if work["date"].dropna().empty:
        return defaults
    if "market_region" not in work.columns:
        work["market_region"] = work["symbol"].map(lambda symbol: market_region_for_symbol(symbol))
    work["market_region"] = work["market_region"].fillna("").astype(str).map(
        lambda value: market_region_for_symbol("", "", value)
    )
    latest = (
        work.dropna(subset=["symbol", "date"])
        .sort_values(["symbol", "date"])
        .groupby("symbol", as_index=False, dropna=False)
        .tail(1)
        .reset_index(drop=True)
    )
    if latest.empty:
        return defaults
    latest_date = latest["date"].max()
    defaults["latest_universe_asof_date"] = latest_date.date().isoformat()
    region_parts = []
    for region, group in latest.groupby("market_region", dropna=False):
        region_dates = group["date"].dropna()
        if region_dates.empty:
            continue
        region_parts.append(
            f"{str(region).upper()}:{region_dates.max().date().isoformat()}:{group['symbol'].astype(str).str.upper().nunique()}"
        )
    defaults["latest_universe_asof_by_region"] = "|".join(sorted(region_parts)) if region_parts else "NA"
    defaults["latest_universe_symbol_count"] = int(latest["symbol"].astype(str).str.upper().nunique())
    if "is_decision_universe" in latest.columns:
        defaults["latest_decision_universe_symbol_count"] = int(
            latest.loc[latest["is_decision_universe"].map(to_bool), "symbol"].astype(str).str.upper().nunique()
        )
    for column, field in [
        ("is_event_candidate", "latest_event_candidate_count"),
        ("is_actionable_entry_candidate", "latest_actionable_candidate_count"),
        ("is_trade_ready_entry_candidate", "latest_trade_ready_candidate_count"),
    ]:
        if column in latest.columns:
            defaults[field] = int(latest[column].map(to_bool).sum())

    def summarize_candidates(mask_col: str) -> str:
        if mask_col not in latest.columns:
            return "PASS"
        subset = latest[latest[mask_col].map(to_bool)].copy()
        if subset.empty:
            return "PASS"
        subset["score_numeric"] = pd.to_numeric(subset.get("score_price_algo_total", pd.Series(dtype=float)), errors="coerce")
        subset = subset.sort_values(["score_numeric", "symbol"], ascending=[False, True])
        parts = []
        for _, row in subset.head(8).iterrows():
            score = row.get("score_numeric")
            score_text = f"{float(score):.2f}" if pd.notna(score) else "NA"
            parts.append(f"{str(row.get('symbol', '')).upper()}:{row.get('entry_gate_status', 'NA')}:{score_text}")
        if len(subset) > 8:
            parts.append(f"+{len(subset) - 8}")
        return "|".join(parts)

    defaults["latest_actionable_candidate_summary"] = summarize_candidates("is_actionable_entry_candidate")
    defaults["latest_trade_ready_candidate_summary"] = summarize_candidates("is_trade_ready_entry_candidate")
    return defaults


def asof_alignment_summary(pooled_features: pd.DataFrame, market_data_latest: pd.DataFrame) -> dict[str, object]:
    defaults: dict[str, object] = {
        "asof_alignment_status": "PASS",
        "asof_alignment_block_reasons": "PASS",
        "prediction_asof_date_for_alignment": "NA",
        "prediction_asof_by_region": "NA",
        "market_data_latest_daily_max_date": "NA",
        "market_data_latest_daily_max_by_region": "NA",
        "market_data_latest_lag_days": "NA",
        "market_data_latest_generated_at_utc": "NA",
        "market_data_latest_stale_symbol_summary": "PASS",
    }
    if pooled_features.empty or "date" not in pooled_features.columns:
        defaults["asof_alignment_status"] = "WARN"
        defaults["asof_alignment_block_reasons"] = "MISSING_POOLED_FEATURE_MATRIX_ASOF"
        return defaults
    prediction_dates = pd.to_datetime(pooled_features["date"], errors="coerce").dropna()
    if prediction_dates.empty:
        defaults["asof_alignment_status"] = "WARN"
        defaults["asof_alignment_block_reasons"] = "MISSING_PREDICTION_ASOF_DATE"
        return defaults
    prediction_asof = prediction_dates.max().normalize()
    defaults["prediction_asof_date_for_alignment"] = prediction_asof.date().isoformat()
    pred = pooled_features.copy()
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce").dt.normalize()
    if "symbol" in pred.columns:
        if "market_region" not in pred.columns:
            pred["market_region"] = pred["symbol"].map(lambda symbol: market_region_for_symbol(symbol))
        pred["market_region"] = pred["market_region"].fillna("").astype(str).map(
            lambda value: market_region_for_symbol("", "", value)
        )
        pred_latest = (
            pred.dropna(subset=["symbol", "date"])
            .sort_values(["symbol", "date"])
            .groupby("symbol", as_index=False, dropna=False)
            .tail(1)
            .reset_index(drop=True)
        )
    else:
        pred_latest = pd.DataFrame()
    if not pred_latest.empty:
        pred_parts = []
        for region, group in pred_latest.groupby("market_region", dropna=False):
            dates = group["date"].dropna()
            if dates.empty:
                continue
            pred_parts.append(
                f"{str(region).upper()}:{dates.max().date().isoformat()}:{group['symbol'].astype(str).str.upper().nunique()}"
            )
        defaults["prediction_asof_by_region"] = "|".join(sorted(pred_parts)) if pred_parts else "NA"
    if market_data_latest.empty or "end_timestamp" not in market_data_latest.columns:
        defaults["asof_alignment_status"] = "WARN"
        defaults["asof_alignment_block_reasons"] = "MISSING_MARKET_DATA_LATEST_ASOF"
        return defaults
    market = market_data_latest.copy()
    if "bar_type" in market.columns:
        daily = market[market["bar_type"].astype(str).str.lower().eq("daily")].copy()
        if not daily.empty:
            market = daily
    market["end_date"] = pd.to_datetime(market["end_timestamp"], errors="coerce").dt.normalize()
    if "symbol" in market.columns:
        if "market_region" not in market.columns:
            market["market_region"] = market["symbol"].map(lambda symbol: market_region_for_symbol(symbol))
        market["market_region"] = market["market_region"].fillna("").astype(str).map(
            lambda value: market_region_for_symbol("", "", value)
        )
    market_dates = market["end_date"].dropna()
    if market_dates.empty:
        defaults["asof_alignment_status"] = "WARN"
        defaults["asof_alignment_block_reasons"] = "MISSING_MARKET_DATA_LATEST_END_TIMESTAMP"
        return defaults
    market_max = market_dates.max()
    lag_days = int((prediction_asof - market_max).days)
    defaults["market_data_latest_daily_max_date"] = market_max.date().isoformat()
    if "market_region" in market.columns:
        market_parts = []
        for region, group in market.groupby("market_region", dropna=False):
            dates = group["end_date"].dropna()
            if dates.empty:
                continue
            symbol_count = group["symbol"].astype(str).str.upper().nunique() if "symbol" in group.columns else len(group)
            market_parts.append(f"{str(region).upper()}:{dates.max().date().isoformat()}:{symbol_count}")
        defaults["market_data_latest_daily_max_by_region"] = "|".join(sorted(market_parts)) if market_parts else "NA"
    if "generated_at_utc" in market.columns:
        generated = pd.to_datetime(market["generated_at_utc"], errors="coerce").dropna()
        if not generated.empty:
            defaults["market_data_latest_generated_at_utc"] = generated.max().isoformat()
    if not pred_latest.empty and "symbol" in market.columns:
        market_latest = (
            market.dropna(subset=["symbol", "end_date"])
            .sort_values(["symbol", "end_date"])
            .groupby("symbol", as_index=False, dropna=False)
            .tail(1)
            .reset_index(drop=True)
        )
        alignment = pred_latest[["symbol", "date", "market_region"]].merge(
            market_latest[["symbol", "end_date"]],
            on="symbol",
            how="left",
        )
        alignment["lag_days"] = (alignment["date"] - alignment["end_date"]).dt.days
        alignment.loc[alignment["end_date"].isna(), "lag_days"] = 9999
        lag_values = pd.to_numeric(alignment["lag_days"], errors="coerce").dropna()
        defaults["market_data_latest_lag_days"] = int(lag_values.max()) if not lag_values.empty else "NA"
        stale = alignment[alignment["lag_days"].fillna(9999).gt(0)].copy()
    else:
        defaults["market_data_latest_lag_days"] = lag_days
        stale = market[market["end_date"].notna() & market["end_date"].lt(prediction_asof)].copy()
    if not stale.empty and "symbol" in stale.columns:
        if "lag_days" in stale.columns:
            stale = stale.sort_values(["lag_days", "symbol"], ascending=[False, True])
        else:
            stale = stale.sort_values(["end_date", "symbol"])
        defaults["market_data_latest_stale_symbol_summary"] = "|".join(
            f"{str(row.get('symbol', '')).upper()}:{pd.to_datetime(row.get('end_date'), errors='coerce').date().isoformat() if pd.notna(pd.to_datetime(row.get('end_date'), errors='coerce')) else 'MISSING'}"
            for _, row in stale.head(8).iterrows()
        )
        if len(stale) > 8:
            defaults["market_data_latest_stale_symbol_summary"] += f"|+{len(stale) - 8}"
    lag_value = defaults["market_data_latest_lag_days"]
    if lag_value != "NA" and int(lag_value) > 0:
        defaults["asof_alignment_status"] = "WARN"
        defaults["asof_alignment_block_reasons"] = "MARKET_DATA_LATEST_LAGS_PREDICTION_ASOF"
    return defaults


def checks_critical_pass(checks: pd.DataFrame) -> tuple[bool, str]:
    if checks.empty or "passed" not in checks.columns:
        return False, "MISSING_CHECKS"
    target = checks[checks["severity"].astype(str).eq("CRITICAL")] if "severity" in checks.columns else checks
    if target.empty:
        target = checks
    passed = target["passed"].astype(str).str.lower().isin(["true", "1", "yes"]).all()
    failed = target[~target["passed"].astype(str).str.lower().isin(["true", "1", "yes"])]
    reasons = "|".join(failed["check"].astype(str)) if not failed.empty and "check" in failed.columns else "PASS"
    return bool(passed), reasons


def component_row(component: str, status: str, block_reasons: str, details: str = "") -> dict[str, object]:
    return {"component": component, "status": status, "block_reasons": block_reasons or "PASS", "details": details}


def derive_performance_gate_row(model_gate: dict[str, object]) -> dict[str, object]:
    if not model_gate:
        return component_row(
            "model_performance_gate",
            "FAIL",
            "MISSING_MODEL_GATE",
            "performance_status=MISSING, active_failed=NA, active_blocking=NA, diagnostic_warnings=NA, next_action=UNKNOWN",
        )

    performance_status = str(model_gate.get("performance_gate_status", "UNKNOWN"))
    active_failed = to_int(model_gate.get("active_performance_failed_gate_count", 0))
    active_blocking = to_int(model_gate.get("active_performance_blocking_failed_gate_count", 0))
    diagnostic_warnings = to_int(model_gate.get("diagnostic_performance_warning_gate_count", 0))
    if active_blocking > 0 or performance_status.startswith(("BLOCKED", "FAIL")):
        status = "FAIL"
    elif active_failed > 0 or performance_status.startswith("WARN") or performance_status == "UNKNOWN":
        status = "WARN"
    elif performance_status.startswith("PASS"):
        status = "PASS"
    else:
        status = "WARN"

    block_reasons = "PASS" if status == "PASS" else str(
        model_gate.get(
            "active_performance_failed_gate_groups",
            model_gate.get("performance_failed_gate_groups", "MISSING_PERFORMANCE_GATE"),
        )
    )
    details = (
        f"performance_status={performance_status}, active_failed={active_failed}, "
        f"active_blocking={active_blocking}, diagnostic_warnings={diagnostic_warnings}, "
        f"active_metric_failed={model_gate.get('active_metric_performance_failed_gate_count', 'NA')}, "
        f"rank_policy_supported_warnings={model_gate.get('rank_policy_supported_performance_warning_count', 'NA')}, "
        f"unresolved_diagnostic_metric_warnings={model_gate.get('unresolved_diagnostic_metric_performance_warning_gate_count', 'NA')}, "
        f"next_action={model_gate.get('next_required_performance_action', 'UNKNOWN')}, "
        f"interpretation={model_gate.get('performance_gate_interpretation', 'UNKNOWN')}"
    )
    return component_row("model_performance_gate", status, block_reasons, details)


def derive_component_rows(
    data_quality: dict[str, object],
    system_state: dict[str, object],
    prediction: dict[str, object],
    model_gate: dict[str, object],
    risk: dict[str, object],
    shadow_quality: pd.DataFrame,
    operational_quality: pd.DataFrame,
    paper_oms_quality: pd.DataFrame | None = None,
    paper_reconciliation_quality: pd.DataFrame | None = None,
    order_state_quality: pd.DataFrame | None = None,
    execution_feedback_quality: pd.DataFrame | None = None,
    fill_calibration_quality: pd.DataFrame | None = None,
    automation_quality: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    data_status = str(data_quality.get("data_quality_status", "MISSING"))
    data_gate = str(data_quality.get("decision_support_data_gate", "UNKNOWN"))
    rows.append(
        component_row(
            "data_quality",
            "FAIL" if data_status in {"FAIL", "MISSING"} else ("WARN" if data_status == "WARN" else "PASS"),
            str(data_quality.get("block_reasons", "MISSING_DATA_QUALITY")),
            f"gate={data_gate}, latest_signal_date={data_quality.get('latest_signal_date', 'NA')}",
        )
    )

    operational_ok, operational_reasons = checks_critical_pass(operational_quality)
    rows.append(component_row("operational_quality", "PASS" if operational_ok else "FAIL", operational_reasons))

    research_ready = to_bool(system_state.get("research_ready", False))
    rows.append(
        component_row(
            "research_readiness",
            "PASS" if research_ready else "FAIL",
            str(system_state.get("research_block_reasons", "MISSING_SYSTEM_STATE")),
            f"score={system_state.get('research_readiness_score', system_state.get('system_readiness_score', 'NA'))}",
        )
    )

    prediction_signal_ready = is_prediction_decision_support(prediction.get("prediction_use_status", "UNKNOWN"))
    prediction_pipeline_ready = to_bool(system_state.get("prediction_pipeline_ready", model_gate.get("prediction_pipeline_ready", False)))
    prediction_block_reasons = str(
        system_state.get("prediction_block_reasons", prediction.get("model_quality_block_reasons", "PREDICTION_NOT_DECISION_SUPPORT"))
        or ""
    ).strip()
    if prediction_pipeline_ready and not prediction_signal_ready:
        prediction_block_reasons = "PASS"
    if prediction_block_reasons.upper() in {"", "UNKNOWN", "PREDICTION_NOT_DECISION_SUPPORT"}:
        pooled_latest_reasons = str(
            model_gate.get("pooled_latest_block_reasons", model_gate.get("pooled_decision_block_reasons", ""))
            or ""
        ).strip()
        if pooled_latest_reasons and pooled_latest_reasons.upper() not in {"PASS", "UNKNOWN", "NAN"}:
            prediction_block_reasons = f"POOLED_LATEST:{pooled_latest_reasons}"
    if not prediction_block_reasons:
        prediction_block_reasons = "PREDICTION_NOT_DECISION_SUPPORT"
    rows.append(
        component_row(
            "prediction_decision_support",
            "PASS" if prediction_pipeline_ready else "BLOCKED",
            prediction_block_reasons,
            (
                f"pipeline_ready={prediction_pipeline_ready}, "
                f"use_status={prediction.get('prediction_use_status', 'NA')}, "
                f"signal_status={prediction.get('prediction_signal_status', 'NA')}, "
                f"latest_market_state={system_state.get('latest_market_state_status', 'NA')}"
            ),
        )
    )

    gate_status = str(model_gate.get("model_gate_status", "MISSING"))
    blocking_failed_count = to_int(model_gate.get("blocking_failed_gate_count", model_gate.get("critical_failed_gate_count", 0)))
    pooled_system_quality_pass = to_bool(model_gate.get("pooled_system_quality_pass", False))
    model_quality_standby_statuses = {"MODEL_QUALITY_PASS_LATEST_BLOCKED", "PASS_MODEL_QUALITY_SIGNAL_STANDBY"}
    if gate_status == "PASS" or gate_status.startswith("PASS_MODEL_QUALITY"):
        model_gate_status = "PASS"
    elif gate_status in model_quality_standby_statuses and blocking_failed_count == 0 and pooled_system_quality_pass:
        model_gate_status = "PASS"
    elif gate_status in {"BLOCKED", "WARN", "MODEL_QUALITY_PASS_LATEST_BLOCKED"}:
        model_gate_status = "BLOCKED"
    else:
        model_gate_status = "FAIL"
    model_gate_reasons = (
        str(model_gate.get("blocking_failed_gate_groups", "PASS"))
        if model_gate_status == "PASS"
        else str(model_gate.get("blocking_failed_gate_groups", model_gate.get("failed_gate_groups", "MISSING_MODEL_GATE")))
    )
    model_gate_details = (
        f"status={gate_status}, blocking_failed_gates={blocking_failed_count}, "
        f"warning_gates={model_gate.get('warning_failed_gate_count', 'NA')}, "
        f"latest_signal_pass={model_gate.get('pooled_latest_signal_pass', 'NA')}"
    )
    rows.append(
        component_row(
            "model_gate",
            model_gate_status,
            model_gate_reasons,
            model_gate_details,
        )
    )
    rows.append(derive_performance_gate_row(model_gate))

    risk_state = str(risk.get("risk_state", "UNKNOWN"))
    risk_ok = risk_state not in {"UNKNOWN", "NO_NEW_RISK"}
    rows.append(
        component_row(
            "risk_policy",
            "PASS" if risk_ok else "FAIL",
            "PASS" if risk_ok else risk_state,
            f"risk_state={risk_state}, max_weight={risk.get('final_recommended_max_weight', 'NA')}",
        )
    )

    paper_ready = to_bool(system_state.get("paper_ready", False))
    shadow_ok, shadow_reasons = checks_critical_pass(shadow_quality)
    paper_status = "PASS" if paper_ready and shadow_ok else "FAIL"
    rows.append(
        component_row(
            "paper_shadow",
            paper_status,
            "PASS" if paper_status == "PASS" else f"{system_state.get('paper_block_reasons', 'PAPER_NOT_READY')}|{shadow_reasons}",
            f"paper_status={system_state.get('paper_trading_status', 'NA')}",
        )
    )

    if paper_oms_quality is not None:
        paper_oms_ok, paper_oms_reasons = checks_critical_pass(paper_oms_quality)
        rows.append(
            component_row(
                "paper_oms",
                "PASS" if paper_oms_ok else "FAIL",
                paper_oms_reasons,
                "Broker-free paper order/fill/position ledger checks.",
            )
        )

    if paper_reconciliation_quality is not None:
        paper_recon_ok, paper_recon_reasons = checks_critical_pass(paper_reconciliation_quality)
        rows.append(
            component_row(
                "paper_reconciliation",
                "PASS" if paper_recon_ok else "FAIL",
                paper_recon_reasons,
                "Internal paper position reconciliation checks.",
            )
        )

    if order_state_quality is not None:
        order_state_ok, order_state_reasons = checks_critical_pass(order_state_quality)
        rows.append(component_row("paper_order_lifecycle", "PASS" if order_state_ok else "FAIL", order_state_reasons, "Paper order state machine checks."))

    if execution_feedback_quality is not None:
        feedback_ok, feedback_reasons = checks_critical_pass(execution_feedback_quality)
        rows.append(component_row("paper_execution_feedback", "PASS" if feedback_ok else "FAIL", feedback_reasons, "Paper execution feedback feature/label checks."))

    if fill_calibration_quality is not None:
        calibration_ok, calibration_reasons = checks_critical_pass(fill_calibration_quality)
        rows.append(component_row("paper_fill_calibration", "PASS" if calibration_ok else "FAIL", calibration_reasons, "Fill model calibration output checks."))

    if automation_quality is not None:
        automation_ok, automation_reasons = checks_critical_pass(automation_quality)
        rows.append(component_row("paper_automation", "PASS" if automation_ok else "FAIL", automation_reasons, "Broker-free automation plan checks."))

    rows.append(
        component_row(
            "live_trading",
            "NOT_APPLICABLE",
            str(system_state.get("live_block_reasons", "NO_LIVE_BROKER_BY_DESIGN")),
            "Broker execution is excluded by design.",
        )
    )
    return pd.DataFrame(rows)


def derive_overall_status(blocks: pd.DataFrame) -> str:
    if blocks.empty:
        return "FAIL"
    hard_components = blocks[
        blocks["component"].isin(
            [
                "data_quality",
                "operational_quality",
                "research_readiness",
                "model_performance_gate",
                "risk_policy",
                "paper_shadow",
                "paper_oms",
                "paper_reconciliation",
                "paper_order_lifecycle",
                "paper_execution_feedback",
                "paper_fill_calibration",
                "paper_automation",
            ]
        )
    ]
    if hard_components["status"].isin(["FAIL"]).any():
        return "FAIL"
    if blocks["status"].isin(["BLOCKED", "WARN"]).any():
        return "WARN"
    return "PASS"


def build_snapshot(
    blocks: pd.DataFrame,
    system_state: dict[str, object],
    data_quality: dict[str, object],
    prediction: dict[str, object],
    model_gate: dict[str, object] | None = None,
    latest_blocker_audit: pd.DataFrame | None = None,
    pooled_feature_matrix: pd.DataFrame | None = None,
    asof_alignment: dict[str, object] | None = None,
) -> pd.DataFrame:
    model_gate = model_gate or {}
    asof_alignment = asof_alignment or {
        "asof_alignment_status": "PASS",
        "asof_alignment_block_reasons": "PASS",
        "prediction_asof_date_for_alignment": "NA",
        "market_data_latest_daily_max_date": "NA",
        "market_data_latest_lag_days": "NA",
        "market_data_latest_generated_at_utc": "NA",
        "market_data_latest_stale_symbol_summary": "PASS",
    }
    latest_gaps = latest_blocker_gap_summary(latest_blocker_audit if latest_blocker_audit is not None else pd.DataFrame())
    latest_universe = latest_universe_scope_summary(pooled_feature_matrix if pooled_feature_matrix is not None else pd.DataFrame())
    status = derive_overall_status(blocks)
    failed_or_blocked = blocks[blocks["status"].isin(["FAIL", "BLOCKED", "WARN"])]
    prediction_support = is_prediction_decision_support(prediction.get("prediction_use_status", "UNKNOWN"))
    pooled_model_quality_pass = to_bool(model_gate.get("pooled_model_quality_pass", system_state.get("pooled_model_quality_pass", False)))
    pooled_system_quality_pass = to_bool(model_gate.get("pooled_system_quality_pass", system_state.get("pooled_system_quality_pass", False)))
    pooled_latest_signal_pass = to_bool(model_gate.get("pooled_latest_signal_pass", system_state.get("pooled_latest_signal_pass", False)))
    prediction_pipeline_ready = to_bool(system_state.get("prediction_pipeline_ready", model_gate.get("prediction_pipeline_ready", False)))
    active_failed = to_int(model_gate.get("active_performance_failed_gate_count", system_state.get("active_performance_failed_gate_count", 0)))
    active_blocking = to_int(model_gate.get("active_performance_blocking_failed_gate_count", system_state.get("active_performance_blocking_failed_gate_count", 0)))
    latest_block_reasons = str(
        system_state.get(
            "prediction_latest_block_reasons",
            model_gate.get("pooled_latest_block_reasons", model_gate.get("pooled_decision_block_reasons", "PASS")),
        )
        or "PASS"
    )
    if prediction_support:
        prediction_block_category = "PASS"
    elif active_failed > 0 or active_blocking > 0:
        prediction_block_category = "MODEL_PERFORMANCE_BLOCKED"
    elif prediction_pipeline_ready and not pooled_latest_signal_pass:
        prediction_block_category = "NO_ENTRY_SIGNAL_STANDBY"
    elif pooled_system_quality_pass and not pooled_latest_signal_pass:
        prediction_block_category = "LATEST_MARKET_STATE_BLOCKED"
    elif not pooled_system_quality_pass:
        prediction_block_category = "MODEL_QUALITY_BLOCKED"
    else:
        prediction_block_category = "PREDICTION_NOT_DECISION_SUPPORT"
    latest_market_state_status = system_state.get(
        "latest_market_state_status",
        "DECISION_SUPPORT_ALLOWED" if prediction_support else ("LATEST_SIGNAL_BLOCKED" if prediction_block_category == "LATEST_MARKET_STATE_BLOCKED" else prediction_block_category),
    )
    rows = [
        {"field": "daily_health_status", "value": status},
        {"field": "system_state", "value": system_state.get("system_state", "UNKNOWN")},
        {"field": "data_quality_status", "value": data_quality.get("data_quality_status", "UNKNOWN")},
        {"field": "research_ready", "value": system_state.get("research_ready", False)},
        {"field": "paper_ready", "value": system_state.get("paper_ready", False)},
        {"field": "prediction_pipeline_ready", "value": prediction_pipeline_ready},
        {"field": "prediction_decision_support", "value": prediction_support},
        {"field": "prediction_use_status", "value": prediction.get("prediction_use_status", "UNKNOWN")},
        {"field": "prediction_block_category", "value": prediction_block_category},
        {"field": "latest_market_state_status", "value": latest_market_state_status},
        {"field": "prediction_latest_block_reasons", "value": latest_block_reasons},
        {"field": "latest_blocker_failed_check_count", "value": latest_gaps["latest_blocker_failed_check_count"]},
        {"field": "latest_blocker_numeric_gap_count", "value": latest_gaps["latest_blocker_numeric_gap_count"]},
        {"field": "latest_blocker_failed_checks", "value": latest_gaps["latest_blocker_failed_checks"]},
        {"field": "latest_blocker_recommended_actions", "value": latest_gaps["latest_blocker_recommended_actions"]},
        {"field": "latest_score_gap_to_threshold", "value": latest_gaps["latest_score_gap_to_threshold"]},
        {"field": "latest_stop_risk_gap_to_strict_limit", "value": latest_gaps["latest_stop_risk_gap_to_strict_limit"]},
        {"field": "latest_expected_r_gap_to_min", "value": latest_gaps["latest_expected_r_gap_to_min"]},
        {"field": "paper_stop_risk_gap_to_limit", "value": latest_gaps["paper_stop_risk_gap_to_limit"]},
        {"field": "latest_blocker_numeric_gap_summary", "value": latest_gaps["latest_blocker_numeric_gap_summary"]},
        {"field": "latest_universe_asof_date", "value": latest_universe["latest_universe_asof_date"]},
        {"field": "latest_universe_asof_by_region", "value": latest_universe["latest_universe_asof_by_region"]},
        {"field": "latest_universe_symbol_count", "value": latest_universe["latest_universe_symbol_count"]},
        {"field": "latest_decision_universe_symbol_count", "value": latest_universe["latest_decision_universe_symbol_count"]},
        {"field": "latest_event_candidate_count", "value": latest_universe["latest_event_candidate_count"]},
        {"field": "latest_actionable_candidate_count", "value": latest_universe["latest_actionable_candidate_count"]},
        {"field": "latest_trade_ready_candidate_count", "value": latest_universe["latest_trade_ready_candidate_count"]},
        {"field": "latest_actionable_candidate_summary", "value": latest_universe["latest_actionable_candidate_summary"]},
        {"field": "latest_trade_ready_candidate_summary", "value": latest_universe["latest_trade_ready_candidate_summary"]},
        {"field": "asof_alignment_status", "value": asof_alignment.get("asof_alignment_status", "PASS")},
        {"field": "asof_alignment_block_reasons", "value": asof_alignment.get("asof_alignment_block_reasons", "PASS")},
        {"field": "prediction_asof_date_for_alignment", "value": asof_alignment.get("prediction_asof_date_for_alignment", "NA")},
        {"field": "prediction_asof_by_region", "value": asof_alignment.get("prediction_asof_by_region", "NA")},
        {"field": "market_data_latest_daily_max_date", "value": asof_alignment.get("market_data_latest_daily_max_date", "NA")},
        {"field": "market_data_latest_daily_max_by_region", "value": asof_alignment.get("market_data_latest_daily_max_by_region", "NA")},
        {"field": "market_data_latest_lag_days", "value": asof_alignment.get("market_data_latest_lag_days", "NA")},
        {"field": "market_data_latest_generated_at_utc", "value": asof_alignment.get("market_data_latest_generated_at_utc", "NA")},
        {"field": "market_data_latest_stale_symbol_summary", "value": asof_alignment.get("market_data_latest_stale_symbol_summary", "PASS")},
        {"field": "pooled_model_quality_pass", "value": pooled_model_quality_pass},
        {"field": "pooled_system_quality_pass", "value": pooled_system_quality_pass},
        {"field": "pooled_latest_signal_pass", "value": pooled_latest_signal_pass},
        {
            "field": "pooled_latest_block_reasons",
            "value": model_gate.get("pooled_latest_block_reasons", system_state.get("pooled_latest_block_reasons", "UNKNOWN")),
        },
        {
            "field": "pooled_decision_block_reasons",
            "value": model_gate.get("pooled_decision_block_reasons", system_state.get("pooled_decision_block_reasons", "UNKNOWN")),
        },
        {"field": "paper_gate_status", "value": model_gate.get("paper_gate_status", system_state.get("paper_gate_status", "UNKNOWN"))},
        {"field": "paper_gate_block_reasons", "value": model_gate.get("paper_gate_block_reasons", system_state.get("paper_gate_block_reasons", "UNKNOWN"))},
        {
            "field": "prediction_performance_block_reasons",
            "value": system_state.get(
                "prediction_performance_block_reasons",
                model_gate.get("active_performance_failed_gate_groups", "UNKNOWN"),
            ),
        },
        {"field": "model_gate_status", "value": model_gate.get("model_gate_status", system_state.get("model_gate_status", "UNKNOWN"))},
        {"field": "performance_gate_status", "value": model_gate.get("performance_gate_status", system_state.get("performance_gate_status", "UNKNOWN"))},
        {
            "field": "active_performance_failed_gate_count",
            "value": model_gate.get("active_performance_failed_gate_count", system_state.get("active_performance_failed_gate_count", 0)),
        },
        {
            "field": "active_performance_blocking_failed_gate_count",
            "value": model_gate.get(
                "active_performance_blocking_failed_gate_count",
                system_state.get("active_performance_blocking_failed_gate_count", 0),
            ),
        },
        {
            "field": "diagnostic_performance_warning_gate_count",
            "value": model_gate.get("diagnostic_performance_warning_gate_count", system_state.get("diagnostic_performance_warning_gate_count", 0)),
        },
        {
            "field": "metric_performance_failed_gate_count",
            "value": model_gate.get("metric_performance_failed_gate_count", system_state.get("metric_performance_failed_gate_count", 0)),
        },
        {
            "field": "performance_evidence_gap_count",
            "value": model_gate.get("performance_evidence_gap_count", system_state.get("performance_evidence_gap_count", 0)),
        },
        {
            "field": "aggregate_performance_quality_flag_count",
            "value": model_gate.get("aggregate_performance_quality_flag_count", system_state.get("aggregate_performance_quality_flag_count", 0)),
        },
        {
            "field": "active_metric_performance_failed_gate_count",
            "value": model_gate.get("active_metric_performance_failed_gate_count", system_state.get("active_metric_performance_failed_gate_count", 0)),
        },
        {
            "field": "diagnostic_metric_performance_warning_gate_count",
            "value": model_gate.get(
                "diagnostic_metric_performance_warning_gate_count",
                system_state.get("diagnostic_metric_performance_warning_gate_count", 0),
            ),
        },
        {
            "field": "rank_policy_supported_performance_warning_count",
            "value": model_gate.get(
                "rank_policy_supported_performance_warning_count",
                system_state.get("rank_policy_supported_performance_warning_count", 0),
            ),
        },
        {
            "field": "rank_policy_supported_threshold_warning_count",
            "value": model_gate.get(
                "rank_policy_supported_threshold_warning_count",
                system_state.get("rank_policy_supported_threshold_warning_count", 0),
            ),
        },
        {
            "field": "rank_policy_supported_metric_warning_count",
            "value": model_gate.get(
                "rank_policy_supported_metric_warning_count",
                system_state.get("rank_policy_supported_metric_warning_count", 0),
            ),
        },
        {
            "field": "rank_policy_supported_threshold_metric_warning_count",
            "value": model_gate.get(
                "rank_policy_supported_threshold_metric_warning_count",
                system_state.get("rank_policy_supported_threshold_metric_warning_count", 0),
            ),
        },
        {
            "field": "evidence_limited_metric_warning_count",
            "value": model_gate.get(
                "evidence_limited_metric_warning_count",
                system_state.get("evidence_limited_metric_warning_count", 0),
            ),
        },
        {
            "field": "rejected_model_metric_warning_count",
            "value": model_gate.get(
                "rejected_model_metric_warning_count",
                system_state.get("rejected_model_metric_warning_count", 0),
            ),
        },
        {
            "field": "strategy_diagnostic_metric_warning_count",
            "value": model_gate.get(
                "strategy_diagnostic_metric_warning_count",
                system_state.get("strategy_diagnostic_metric_warning_count", 0),
            ),
        },
        {
            "field": "unresolved_diagnostic_metric_performance_warning_gate_count",
            "value": model_gate.get(
                "unresolved_diagnostic_metric_performance_warning_gate_count",
                system_state.get("unresolved_diagnostic_metric_performance_warning_gate_count", 0),
            ),
        },
        {
            "field": "classified_diagnostic_performance_warning_count",
            "value": model_gate.get(
                "classified_diagnostic_performance_warning_count",
                system_state.get("classified_diagnostic_performance_warning_count", 0),
            ),
        },
        {
            "field": "unresolved_diagnostic_performance_warning_count",
            "value": model_gate.get(
                "unresolved_diagnostic_performance_warning_count",
                system_state.get("unresolved_diagnostic_performance_warning_count", 0),
            ),
        },
        {"field": "uncategorized_root_cause_count", "value": model_gate.get("uncategorized_root_cause_count", 0)},
        {"field": "uncategorized_warning_gate_count", "value": model_gate.get("uncategorized_warning_gate_count", 0)},
        {"field": "uncategorized_root_causes", "value": model_gate.get("uncategorized_root_causes", "PASS")},
        {
            "field": "performance_warning_resolution_status",
            "value": model_gate.get(
                "performance_warning_resolution_status",
                system_state.get("performance_warning_resolution_status", "UNKNOWN"),
            ),
        },
        {
            "field": "next_required_performance_action",
            "value": model_gate.get("next_required_performance_action", system_state.get("next_required_performance_action", "UNKNOWN")),
        },
        {
            "field": "next_required_evidence_action",
            "value": model_gate.get("next_required_evidence_action", system_state.get("next_required_evidence_action", "UNKNOWN")),
        },
        {
            "field": "performance_gate_interpretation",
            "value": model_gate.get("performance_gate_interpretation", system_state.get("performance_gate_interpretation", "UNKNOWN")),
        },
        {
            "field": "rank_uplift_diagnostic_count",
            "value": model_gate.get("rank_uplift_diagnostic_count", system_state.get("rank_uplift_diagnostic_count", 0)),
        },
        {
            "field": "rank_uplift_positive_diagnostic_count",
            "value": model_gate.get("rank_uplift_positive_diagnostic_count", system_state.get("rank_uplift_positive_diagnostic_count", 0)),
        },
        {
            "field": "rank_policy_diagnostic_pass_count",
            "value": model_gate.get("rank_policy_diagnostic_pass_count", system_state.get("rank_policy_diagnostic_pass_count", 0)),
        },
        {
            "field": "rank_policy_best_se_lower_pct",
            "value": model_gate.get("rank_policy_best_se_lower_pct", system_state.get("rank_policy_best_se_lower_pct", "NA")),
        },
        {"field": "failed_or_blocked_components", "value": "|".join(failed_or_blocked["component"].astype(str)) if not failed_or_blocked.empty else "PASS"},
        {"field": "live_trading_status", "value": system_state.get("live_trading_status", "DISABLED_BY_DESIGN")},
        {"field": "generated_at_utc", "value": now_utc_iso()},
    ]
    return pd.DataFrame(rows)


def write_report(outdir: Path, snapshot: pd.DataFrame, blocks: pd.DataFrame) -> None:
    snap = dict(zip(snapshot["field"], snapshot["value"]))
    lines = [
        "# Top10 Daily Health Report",
        "",
        f"- Daily health status: {snap.get('daily_health_status', 'NA')}",
        f"- System state: {snap.get('system_state', 'NA')}",
        f"- Data quality status: {snap.get('data_quality_status', 'NA')}",
        f"- Research ready: {snap.get('research_ready', 'NA')}",
        f"- Paper ready: {snap.get('paper_ready', 'NA')}",
        f"- Prediction pipeline ready: {snap.get('prediction_pipeline_ready', 'NA')}",
        f"- Prediction decision support: {snap.get('prediction_decision_support', 'NA')}",
        f"- Prediction use status: {snap.get('prediction_use_status', 'NA')}",
        f"- Prediction block category: {snap.get('prediction_block_category', 'NA')}",
        f"- Latest market state status: {snap.get('latest_market_state_status', 'NA')}",
        f"- Latest prediction block reasons: {snap.get('prediction_latest_block_reasons', 'NA')}",
        f"- Latest blocker failed checks: {snap.get('latest_blocker_failed_check_count', 'NA')}",
        f"- Latest blocker failed check names: {snap.get('latest_blocker_failed_checks', 'NA')}",
        f"- Latest blocker recommended actions: {snap.get('latest_blocker_recommended_actions', 'NA')}",
        f"- Latest score gap to threshold: {snap.get('latest_score_gap_to_threshold', 'NA')}",
        f"- Latest stop-risk gap to strict limit: {snap.get('latest_stop_risk_gap_to_strict_limit', 'NA')}",
        f"- Latest expected-R gap to minimum: {snap.get('latest_expected_r_gap_to_min', 'NA')}",
        f"- Paper stop-risk gap to limit: {snap.get('paper_stop_risk_gap_to_limit', 'NA')}",
        f"- Latest blocker numeric gap summary: {snap.get('latest_blocker_numeric_gap_summary', 'NA')}",
        f"- Latest universe as-of date: {snap.get('latest_universe_asof_date', 'NA')}",
        f"- Latest universe as-of by region: {snap.get('latest_universe_asof_by_region', 'NA')}",
        f"- Latest universe symbols: {snap.get('latest_universe_symbol_count', 'NA')}",
        f"- Latest decision-universe symbols: {snap.get('latest_decision_universe_symbol_count', 'NA')}",
        f"- Latest event/actionable/trade-ready candidates: {snap.get('latest_event_candidate_count', 'NA')}/{snap.get('latest_actionable_candidate_count', 'NA')}/{snap.get('latest_trade_ready_candidate_count', 'NA')}",
        f"- Latest actionable candidate summary: {snap.get('latest_actionable_candidate_summary', 'NA')}",
        f"- Latest trade-ready candidate summary: {snap.get('latest_trade_ready_candidate_summary', 'NA')}",
        f"- As-of alignment status: {snap.get('asof_alignment_status', 'NA')}",
        f"- As-of alignment block reasons: {snap.get('asof_alignment_block_reasons', 'NA')}",
        f"- Prediction as-of date: {snap.get('prediction_asof_date_for_alignment', 'NA')}",
        f"- Prediction as-of by region: {snap.get('prediction_asof_by_region', 'NA')}",
        f"- Market data latest daily max date: {snap.get('market_data_latest_daily_max_date', 'NA')}",
        f"- Market data latest daily max by region: {snap.get('market_data_latest_daily_max_by_region', 'NA')}",
        f"- Market data latest lag days: {snap.get('market_data_latest_lag_days', 'NA')}",
        f"- Market data latest generated at UTC: {snap.get('market_data_latest_generated_at_utc', 'NA')}",
        f"- Market data stale symbol summary: {snap.get('market_data_latest_stale_symbol_summary', 'NA')}",
        f"- Pooled model quality pass: {snap.get('pooled_model_quality_pass', 'NA')}",
        f"- Pooled system quality pass: {snap.get('pooled_system_quality_pass', 'NA')}",
        f"- Pooled latest signal pass: {snap.get('pooled_latest_signal_pass', 'NA')}",
        f"- Pooled latest block reasons: {snap.get('pooled_latest_block_reasons', 'NA')}",
        f"- Paper gate status: {snap.get('paper_gate_status', 'NA')}",
        f"- Paper gate block reasons: {snap.get('paper_gate_block_reasons', 'NA')}",
        f"- Prediction performance block reasons: {snap.get('prediction_performance_block_reasons', 'NA')}",
        f"- Model gate status: {snap.get('model_gate_status', 'NA')}",
        f"- Performance gate status: {snap.get('performance_gate_status', 'NA')}",
        f"- Active performance failed gates: {snap.get('active_performance_failed_gate_count', 'NA')}",
        f"- Active performance blocking failed gates: {snap.get('active_performance_blocking_failed_gate_count', 'NA')}",
        f"- Active metric performance failed gates: {snap.get('active_metric_performance_failed_gate_count', 'NA')}",
        f"- Metric performance failed gates: {snap.get('metric_performance_failed_gate_count', 'NA')}",
        f"- Performance evidence gap count: {snap.get('performance_evidence_gap_count', 'NA')}",
        f"- Aggregate performance quality flags: {snap.get('aggregate_performance_quality_flag_count', 'NA')}",
        f"- Diagnostic performance warnings: {snap.get('diagnostic_performance_warning_gate_count', 'NA')}",
        f"- Diagnostic metric performance warnings: {snap.get('diagnostic_metric_performance_warning_gate_count', 'NA')}",
        f"- Rank-policy supported performance warnings: {snap.get('rank_policy_supported_performance_warning_count', 'NA')}",
        f"- Rank-policy supported threshold warnings: {snap.get('rank_policy_supported_threshold_warning_count', 'NA')}",
        f"- Rank-policy supported metric warnings: {snap.get('rank_policy_supported_metric_warning_count', 'NA')}",
        f"- Rank-policy supported threshold metric warnings: {snap.get('rank_policy_supported_threshold_metric_warning_count', 'NA')}",
        f"- Evidence-limited metric warnings: {snap.get('evidence_limited_metric_warning_count', 'NA')}",
        f"- Rejected-model metric warnings: {snap.get('rejected_model_metric_warning_count', 'NA')}",
        f"- Strategy diagnostic metric warnings: {snap.get('strategy_diagnostic_metric_warning_count', 'NA')}",
        f"- Unresolved diagnostic metric performance warnings: {snap.get('unresolved_diagnostic_metric_performance_warning_gate_count', 'NA')}",
        f"- Classified diagnostic performance warnings: {snap.get('classified_diagnostic_performance_warning_count', 'NA')}",
        f"- Unresolved diagnostic performance warnings: {snap.get('unresolved_diagnostic_performance_warning_count', 'NA')}",
        f"- Uncategorized root causes: {snap.get('uncategorized_root_cause_count', 'NA')}",
        f"- Uncategorized warning gate rows: {snap.get('uncategorized_warning_gate_count', 'NA')}",
        f"- Uncategorized root cause names: {snap.get('uncategorized_root_causes', 'NA')}",
        f"- Performance warning resolution status: {snap.get('performance_warning_resolution_status', 'NA')}",
        f"- Next required evidence action: {snap.get('next_required_evidence_action', 'NA')}",
        f"- Next required performance action: {snap.get('next_required_performance_action', 'NA')}",
        f"- Performance gate interpretation: {snap.get('performance_gate_interpretation', 'NA')}",
        f"- Rank uplift diagnostics: {snap.get('rank_uplift_diagnostic_count', 'NA')}",
        f"- Rank uplift positive diagnostics: {snap.get('rank_uplift_positive_diagnostic_count', 'NA')}",
        f"- Rank policy diagnostic passes: {snap.get('rank_policy_diagnostic_pass_count', 'NA')}",
        f"- Best rank policy fold-lower uplift: {snap.get('rank_policy_best_se_lower_pct', 'NA')}",
        f"- Live trading status: {snap.get('live_trading_status', 'NA')}",
        "",
        "## Components",
        "",
        "| Component | Status | Block Reasons | Details |",
        "|---|---|---|---|",
    ]
    for _, row in blocks.iterrows():
        lines.append(f"| {row['component']} | {row['status']} | {row['block_reasons']} | {row['details']} |")
    lines.extend(["", "This report is research tooling, not investment advice."])
    (outdir / "tsm_daily_health_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily health report for TSM research outputs.")
    parser.add_argument("--data-quality", default="tsm_price_rule_output/tsm_latest_data_quality_snapshot.csv")
    parser.add_argument("--system-state", default="tsm_price_rule_output/tsm_latest_system_state.csv")
    parser.add_argument("--prediction", default="tsm_price_rule_output/tsm_latest_prediction_snapshot.csv")
    parser.add_argument("--model-gate", default="tsm_price_rule_output/tsm_model_gate_snapshot.csv")
    parser.add_argument("--latest-blocker-audit", default="tsm_price_rule_output/tsm_latest_prediction_blocker_audit.csv")
    parser.add_argument("--pooled-feature-matrix", default="tsm_price_rule_output/tsm_prediction_pooled_feature_matrix.csv")
    parser.add_argument("--market-data-latest", default="output/tsm_universe_market_data_latest.csv")
    parser.add_argument("--risk", default="tsm_price_rule_output/tsm_latest_risk_snapshot.csv")
    parser.add_argument("--shadow-quality", default="tsm_price_rule_output/tsm_shadow_paper_quality_checks.csv")
    parser.add_argument("--operational-quality", default="tsm_price_rule_output/tsm_operational_quality_checks.csv")
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

    data_quality = read_snapshot(Path(args.data_quality))
    system_state = read_snapshot(Path(args.system_state))
    prediction = read_snapshot(Path(args.prediction))
    model_gate = read_snapshot(Path(args.model_gate))
    latest_blocker_audit = read_csv_if_exists(Path(args.latest_blocker_audit))
    pooled_feature_matrix = read_csv_if_exists(Path(args.pooled_feature_matrix))
    market_data_latest = read_csv_if_exists(Path(args.market_data_latest))
    risk = read_snapshot(Path(args.risk))
    shadow_quality = read_csv_if_exists(Path(args.shadow_quality))
    operational_quality = read_csv_if_exists(Path(args.operational_quality))
    paper_oms_quality = read_csv_if_exists(Path(args.paper_oms_quality)) if args.paper_oms_quality else None
    paper_reconciliation_quality = read_csv_if_exists(Path(args.paper_reconciliation_quality)) if args.paper_reconciliation_quality else None
    order_state_quality = read_csv_if_exists(Path(args.order_state_quality)) if args.order_state_quality else None
    execution_feedback_quality = read_csv_if_exists(Path(args.execution_feedback_quality)) if args.execution_feedback_quality else None
    fill_calibration_quality = read_csv_if_exists(Path(args.fill_calibration_quality)) if args.fill_calibration_quality else None
    automation_quality = read_csv_if_exists(Path(args.automation_quality)) if args.automation_quality else None

    blocks = derive_component_rows(
        data_quality,
        system_state,
        prediction,
        model_gate,
        risk,
        shadow_quality,
        operational_quality,
        paper_oms_quality,
        paper_reconciliation_quality,
        order_state_quality,
        execution_feedback_quality,
        fill_calibration_quality,
        automation_quality,
    )
    asof_alignment = asof_alignment_summary(pooled_feature_matrix, market_data_latest)
    if str(asof_alignment.get("asof_alignment_status", "PASS")).upper() != "PASS":
        blocks = pd.concat(
            [
                blocks,
                pd.DataFrame(
                    [
                        component_row(
                            "asof_alignment",
                            str(asof_alignment.get("asof_alignment_status", "WARN")),
                            str(asof_alignment.get("asof_alignment_block_reasons", "UNKNOWN_ASOF_ALIGNMENT")),
                            (
                                f"prediction_asof={asof_alignment.get('prediction_asof_date_for_alignment', 'NA')}, "
                                f"market_daily_max={asof_alignment.get('market_data_latest_daily_max_date', 'NA')}, "
                                f"lag_days={asof_alignment.get('market_data_latest_lag_days', 'NA')}"
                            ),
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )
    snapshot = build_snapshot(blocks, system_state, data_quality, prediction, model_gate, latest_blocker_audit, pooled_feature_matrix, asof_alignment)
    snapshot.to_csv(outdir / "tsm_daily_health_snapshot.csv", index=False)
    blocks.to_csv(outdir / "tsm_system_block_reasons.csv", index=False)
    write_report(outdir, snapshot, blocks)
    print("completed: daily health outputs =", outdir.resolve())
    print(snapshot.to_string(index=False))


if __name__ == "__main__":
    main()

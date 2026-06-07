#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the full TSM research pipeline without placing orders.

Default flow:
1) refresh price/features
2) rebuild rule-engine signals
3) rebuild backtests
4) rebuild risk policy
5) rebuild validation outputs
6) rebuild stress, integrity, and prediction outputs
7) rebuild system state and daily decision report
8) write an operational manifest and data-quality report
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tsm_core.config import apply_config_defaults, load_run_config


@dataclass
class StepResult:
    step: str
    command: str
    started_at_utc: str
    ended_at_utc: str
    duration_sec: float
    returncode: int
    status: str
    stdout_tail: str
    stderr_tail: str


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def tail_text(value: str, max_chars: int = 4000) -> str:
    value = value or ""
    return value[-max_chars:]


def run_step(step: str, command: List[str], cwd: Path) -> StepResult:
    started = now_utc_iso()
    t0 = time.perf_counter()
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    duration = time.perf_counter() - t0
    ended = now_utc_iso()
    status = "PASS" if proc.returncode == 0 else "FAIL"
    return StepResult(
        step=step,
        command=" ".join(command),
        started_at_utc=started,
        ended_at_utc=ended,
        duration_sec=duration,
        returncode=proc.returncode,
        status=status,
        stdout_tail=tail_text(proc.stdout),
        stderr_tail=tail_text(proc.stderr),
    )


def write_manifest(outdir: Path, results: List[StepResult]) -> None:
    rows = [r.__dict__ for r in results]
    pd.DataFrame(rows).to_csv(outdir / "tsm_daily_update_manifest.csv", index=False)


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def file_check(path: Path, required: bool = True) -> dict:
    exists = path.exists()
    return {
        "check": f"file_exists:{path.name}",
        "passed": bool(exists or not required),
        "value": str(path),
        "details": "required" if required else "optional",
    }


def safe_read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return strip_bom_columns(pd.read_csv(path, **kwargs))


def max_validation_mismatch(validation: Optional[pd.DataFrame]) -> float:
    if validation is None or validation.empty or "mismatch_count" not in validation.columns:
        return np.nan
    return float(pd.to_numeric(validation["mismatch_count"], errors="coerce").fillna(0).max())


def format_percent_value(value) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{parsed:.2f}%"


def build_data_quality_report(output_dir: Path, rule_outdir: Path, run_date: str, include_system_outputs: bool = True) -> pd.DataFrame:
    checks = []
    required_files = [
        output_dir / "tsm_daily_10y_raw.csv",
        output_dir / "tsm_daily_10y_enriched.csv",
        output_dir / "tsm_daily_10y_summary.csv",
        rule_outdir / "tsm_daily_algorithmic_signals.csv",
        rule_outdir / "tsm_latest_decision_snapshot.csv",
        rule_outdir / "tsm_data_quality_checks.csv",
        rule_outdir / "tsm_data_quality_issues.csv",
        rule_outdir / "tsm_latest_data_quality_snapshot.csv",
        rule_outdir / "tsm_data_quality_report.md",
        rule_outdir / "tsm_backtest_strategy_summary.csv",
        rule_outdir / "tsm_risk_policy_daily.csv",
        rule_outdir / "tsm_validation_quality_checks.csv",
        rule_outdir / "tsm_validation_causal_walk_forward_summary.csv",
        rule_outdir / "tsm_model_trials_log.csv",
        rule_outdir / "tsm_pbo_report.csv",
        rule_outdir / "tsm_cscv_pbo_report.csv",
        rule_outdir / "tsm_deflated_sharpe_report.csv",
        rule_outdir / "tsm_daily_integrity_checks.csv",
        rule_outdir / "tsm_latest_integrity_snapshot.csv",
        rule_outdir / "tsm_prediction_model_comparison.csv",
        rule_outdir / "tsm_prediction_oos_predictions.csv",
        rule_outdir / "tsm_prediction_calibration_summary.csv",
        rule_outdir / "tsm_prediction_candidate_scope_stats.csv",
        rule_outdir / "tsm_prediction_label_diagnostics.csv",
        rule_outdir / "tsm_prediction_feature_selection_report.csv",
        rule_outdir / "tsm_prediction_model_audit.csv",
        rule_outdir / "tsm_prediction_reliability_report.md",
        rule_outdir / "tsm_prediction_quality_checks.csv",
        rule_outdir / "tsm_schema_quality_checks.csv",
        rule_outdir / "tsm_prediction_feature_contract.csv",
        rule_outdir / "tsm_prediction_fold_manifest.csv",
        rule_outdir / "tsm_prediction_policy_audit.csv",
        rule_outdir / "tsm_latest_prediction_snapshot.csv",
        rule_outdir / "tsm_next_day_up_model_comparison.csv",
        rule_outdir / "tsm_next_day_up_oos_predictions.csv",
        rule_outdir / "tsm_next_day_up_latest_snapshot.csv",
        rule_outdir / "tsm_next_day_up_quality_checks.csv",
        rule_outdir / "tsm_next_day_up_report.md",
        rule_outdir / "tsm_intraday_daily_features.csv",
        rule_outdir / "tsm_top10_timeframe_coverage.csv",
        rule_outdir / "tsm_intraday_feature_quality_checks.csv",
        rule_outdir / "tsm_ml_overlay_summary.csv",
        rule_outdir / "tsm_ml_overlay_equity_curves.csv",
        rule_outdir / "tsm_ml_overlay_quality_checks.csv",
        rule_outdir / "tsm_ml_overlay_report.md",
        rule_outdir / "tsm_prediction_pooled_feature_matrix.csv",
        rule_outdir / "tsm_prediction_pooled_schema.csv",
        rule_outdir / "tsm_prediction_pooled_quality_checks.csv",
        rule_outdir / "tsm_prediction_pooled_dataset_report.md",
        rule_outdir / "tsm_decision_universe_config.csv",
        rule_outdir / "tsm_research_pool_audit.csv",
        rule_outdir / "tsm_next_close_label_dataset.csv",
        rule_outdir / "tsm_next_close_feature_matrix.csv",
        rule_outdir / "tsm_next_close_feature_selection_report.csv",
        rule_outdir / "tsm_next_close_walk_forward_metrics.csv",
        rule_outdir / "tsm_next_close_oos_predictions.csv",
        rule_outdir / "tsm_next_close_model_comparison.csv",
        rule_outdir / "tsm_next_close_interval_calibration.csv",
        rule_outdir / "tsm_next_close_latest_snapshot.csv",
        rule_outdir / "tsm_next_close_universe_latest_predictions.csv",
        rule_outdir / "tsm_next_close_quality_checks.csv",
        rule_outdir / "tsm_next_close_report.md",
        rule_outdir / "tsm_pooled_model_comparison.csv",
        rule_outdir / "tsm_pooled_model_oos_predictions.csv",
        rule_outdir / "tsm_pooled_model_oof_predictions.csv",
        rule_outdir / "tsm_pooled_model_slice_diagnostics.csv",
        rule_outdir / "tsm_pooled_tsm_calibration.csv",
        rule_outdir / "tsm_pooled_latest_prediction_overlay.csv",
        rule_outdir / "tsm_universe_latest_predictions.csv",
        rule_outdir / "tsm_top10_latest_predictions.csv",
        rule_outdir / "tsm_pooled_model_quality_checks.csv",
        rule_outdir / "tsm_pooled_model_report.md",
        rule_outdir / "tsm_model_gate_audit.csv",
        rule_outdir / "tsm_model_gate_snapshot.csv",
        rule_outdir / "tsm_model_gate_report.md",
        rule_outdir / "tsm_prediction_model_registry.csv",
        rule_outdir / "tsm_prediction_experiment_log.csv",
        rule_outdir / "tsm_prediction_model_registry_report.md",
        rule_outdir / "tsm_shadow_paper_predictions.csv",
        rule_outdir / "tsm_shadow_paper_quality_checks.csv",
        rule_outdir / "tsm_shadow_paper_report.md",
        rule_outdir / "tsm_order_intents.csv",
        rule_outdir / "tsm_order_intent_quality_checks.csv",
        rule_outdir / "tsm_order_intent_report.md",
        rule_outdir / "tsm_portfolio_risk_order_decisions.csv",
        rule_outdir / "tsm_portfolio_risk_snapshot.csv",
        rule_outdir / "tsm_portfolio_targets.csv",
        rule_outdir / "tsm_portfolio_snapshot.csv",
        rule_outdir / "tsm_portfolio_risk_checks.csv",
        rule_outdir / "tsm_portfolio_risk_report.md",
        rule_outdir / "tsm_paper_orders.csv",
        rule_outdir / "tsm_paper_fills.csv",
        rule_outdir / "tsm_paper_positions.csv",
        rule_outdir / "tsm_paper_slippage_report.csv",
        rule_outdir / "tsm_paper_oms_quality_checks.csv",
        rule_outdir / "tsm_paper_oms_report.md",
        rule_outdir / "tsm_paper_reconciliation_report.csv",
        rule_outdir / "tsm_paper_reconciliation_quality_checks.csv",
        rule_outdir / "tsm_paper_reconciliation_report.md",
        rule_outdir / "tsm_order_state_events.csv",
        rule_outdir / "tsm_order_lifecycle_snapshot.csv",
        rule_outdir / "tsm_order_state_quality_checks.csv",
        rule_outdir / "tsm_order_state_report.md",
        rule_outdir / "tsm_execution_feedback_events.csv",
        rule_outdir / "tsm_execution_feedback_features.csv",
        rule_outdir / "tsm_execution_feedback_labels.csv",
        rule_outdir / "tsm_execution_feedback_quality_checks.csv",
        rule_outdir / "tsm_execution_feedback_report.md",
        rule_outdir / "tsm_fill_model_calibration.csv",
        rule_outdir / "tsm_fill_model_calibration_quality_checks.csv",
        rule_outdir / "tsm_fill_model_calibration_report.md",
        rule_outdir / "tsm_automation_plan.csv",
        rule_outdir / "tsm_automation_quality_checks.csv",
        rule_outdir / "tsm_automation_report.md",
        rule_outdir / "tsm_daily_stress_scenarios.csv",
        rule_outdir / "tsm_latest_stress_snapshot.csv",
        output_dir / "tsm_news_raw_articles.csv",
        output_dir / "tsm_news_events_normalized.csv",
        output_dir / "tsm_news_event_clusters.csv",
        output_dir / "tsm_price_news_matches.csv",
        rule_outdir / "tsm_news_integrated_daily.csv",
        rule_outdir / "tsm_news_causal_event_report.md",
        rule_outdir / "tsm_news_cause_forward_return_stats.csv",
    ]
    if include_system_outputs:
        required_files.extend(
            [
                rule_outdir / "tsm_system_readiness_scorecard.csv",
                rule_outdir / "tsm_latest_system_state.csv",
                rule_outdir / "tsm_daily_trading_plan.md",
                rule_outdir / "tsm_daily_health_snapshot.csv",
                rule_outdir / "tsm_system_block_reasons.csv",
                rule_outdir / "tsm_daily_health_report.md",
            ]
        )
    checks.extend(file_check(path) for path in required_files)

    raw = safe_read_csv(output_dir / "tsm_daily_10y_raw.csv", parse_dates=["date"])
    enriched = safe_read_csv(output_dir / "tsm_daily_10y_enriched.csv", parse_dates=["date"])
    signals = safe_read_csv(rule_outdir / "tsm_daily_algorithmic_signals.csv", parse_dates=["date"])
    validation = safe_read_csv(rule_outdir / "tsm_data_validation_raw_vs_enriched.csv")
    quality = safe_read_csv(rule_outdir / "tsm_validation_quality_checks.csv")
    integrity = safe_read_csv(rule_outdir / "tsm_daily_integrity_checks.csv")
    prediction_quality = safe_read_csv(rule_outdir / "tsm_prediction_quality_checks.csv")
    data_quality_snapshot = safe_read_csv(rule_outdir / "tsm_latest_data_quality_snapshot.csv")
    model_gate_snapshot = safe_read_csv(rule_outdir / "tsm_model_gate_snapshot.csv")
    news_daily = safe_read_csv(rule_outdir / "tsm_news_integrated_daily.csv", parse_dates=["date"])
    news_matches = safe_read_csv(output_dir / "tsm_price_news_matches.csv", parse_dates=["date"])

    if raw is not None:
        checks.append({"check": "raw_rows_positive", "passed": len(raw) > 0, "value": len(raw), "details": ""})
        checks.append({"check": "raw_dates_unique", "passed": not raw["date"].duplicated().any(), "value": len(raw), "details": ""})
    if enriched is not None:
        checks.append({"check": "enriched_rows_positive", "passed": len(enriched) > 0, "value": len(enriched), "details": ""})
        checks.append({"check": "enriched_dates_unique", "passed": not enriched["date"].duplicated().any(), "value": len(enriched), "details": ""})
    if raw is not None and enriched is not None:
        checks.append({"check": "raw_enriched_row_count_match", "passed": len(raw) == len(enriched), "value": f"{len(raw)} vs {len(enriched)}", "details": ""})
    if signals is not None:
        latest_signal_date = signals["date"].max()
        days_since_latest = (pd.Timestamp(run_date) - latest_signal_date.normalize()).days
        checks.append({"check": "signals_rows_positive", "passed": len(signals) > 0, "value": len(signals), "details": ""})
        checks.append({"check": "signals_dates_unique", "passed": not signals["date"].duplicated().any(), "value": len(signals), "details": ""})
        checks.append({"check": "latest_signal_not_future", "passed": latest_signal_date.normalize() <= pd.Timestamp(run_date), "value": latest_signal_date.date().isoformat(), "details": f"run_date={run_date}"})
        checks.append({"check": "latest_signal_within_7_calendar_days", "passed": days_since_latest <= 7, "value": days_since_latest, "details": "weekends/holidays can create short gaps"})
    mismatch = max_validation_mismatch(validation)
    if pd.notna(mismatch):
        checks.append({"check": "raw_vs_enriched_no_mismatches", "passed": mismatch == 0, "value": mismatch, "details": "from tsm_data_validation_raw_vs_enriched.csv"})
    if quality is not None and not quality.empty and "passed" in quality.columns:
        passed = quality["passed"].astype(str).str.lower().isin(["true", "1", "yes"]).all()
        checks.append({"check": "validation_quality_checks_pass", "passed": bool(passed), "value": f"{quality['passed'].sum() if quality['passed'].dtype != object else ''}", "details": "from tsm_validation_quality_checks.csv"})
    if integrity is not None and not integrity.empty and "passed" in integrity.columns:
        if "severity" in integrity.columns:
            critical = integrity[integrity["severity"] == "CRITICAL"]
            target = critical if not critical.empty else integrity
        else:
            target = integrity
        passed = target["passed"].astype(str).str.lower().isin(["true", "1", "yes"]).all()
        checks.append({"check": "daily_integrity_critical_checks_pass", "passed": bool(passed), "value": len(target), "details": "from tsm_daily_integrity_checks.csv"})
    if prediction_quality is not None and not prediction_quality.empty and "passed" in prediction_quality.columns:
        if "severity" in prediction_quality.columns:
            critical = prediction_quality[prediction_quality["severity"] == "CRITICAL"]
            target = critical if not critical.empty else prediction_quality
        else:
            target = prediction_quality
        passed = target["passed"].astype(str).str.lower().isin(["true", "1", "yes"]).all()
        checks.append({"check": "prediction_quality_critical_checks_pass", "passed": bool(passed), "value": len(target), "details": "from tsm_prediction_quality_checks.csv"})
    if data_quality_snapshot is not None and not data_quality_snapshot.empty and {"field", "value"}.issubset(data_quality_snapshot.columns):
        data_quality = dict(zip(data_quality_snapshot["field"], data_quality_snapshot["value"]))
        status = str(data_quality.get("data_quality_status", "UNKNOWN"))
        checks.append(
            {
                "check": "data_quality_gate_not_fail",
                "passed": status != "FAIL",
                "value": status,
                "details": str(data_quality.get("block_reasons", "")),
            }
        )
    if model_gate_snapshot is not None and not model_gate_snapshot.empty and {"field", "value"}.issubset(model_gate_snapshot.columns):
        model_gate = dict(zip(model_gate_snapshot["field"], model_gate_snapshot["value"]))
        status = str(model_gate.get("model_gate_status", "UNKNOWN"))
        known_model_gate_statuses = {
            "PASS",
            "WARN",
            "BLOCKED",
            "PAPER_DECISION_SUPPORT_ALLOWED",
            "MODEL_QUALITY_PASS_LATEST_BLOCKED",
            "PASS_MODEL_QUALITY_SIGNAL_STANDBY",
        }
        checks.append(
            {
                "check": "model_gate_audit_ran",
                "passed": status in known_model_gate_statuses,
                "value": status,
                "details": str(model_gate.get("blocking_failed_gate_groups", model_gate.get("failed_gate_groups", ""))),
            }
        )
    if news_daily is not None and not news_daily.empty:
        latest_news_date = news_daily["date"].max()
        checks.append(
            {
                "check": "news_integrated_rows_positive",
                "passed": len(news_daily) > 0,
                "value": len(news_daily),
                "details": f"latest={latest_news_date.date().isoformat() if pd.notna(latest_news_date) else 'NA'}",
            }
        )
        if "news_coverage_status" in news_daily.columns:
            coverage_values = "|".join(sorted(set(news_daily["news_coverage_status"].dropna().astype(str).tail(20))))
            checks.append({"check": "news_direct_web_collection_recorded", "passed": True, "value": coverage_values or "UNKNOWN", "details": "direct web/RSS collection is best-effort"})
    if news_matches is not None and not news_matches.empty:
        checks.append({"check": "news_price_matches_rows_positive", "passed": len(news_matches) > 0, "value": len(news_matches), "details": "price-to-news cause matching output"})

    report = pd.DataFrame(checks)
    report.to_csv(rule_outdir / "tsm_operational_quality_checks.csv", index=False)
    return report


def write_operational_report(rule_outdir: Path, manifest: List[StepResult], quality: pd.DataFrame) -> None:
    passed_steps = all(r.status in {"PASS", "PARTIAL_INTRADAY"} for r in manifest)
    passed_quality = bool(quality["passed"].all()) if not quality.empty else False
    latest_plan = rule_outdir / "tsm_daily_trading_plan.md"
    latest_risk = rule_outdir / "tsm_latest_risk_snapshot.csv"
    latest_stress = rule_outdir / "tsm_latest_stress_snapshot.csv"
    latest_system = rule_outdir / "tsm_latest_system_state.csv"
    latest_prediction = rule_outdir / "tsm_latest_prediction_snapshot.csv"
    latest_health = rule_outdir / "tsm_daily_health_snapshot.csv"
    latest_news = rule_outdir / "tsm_news_integrated_daily.csv"

    lines = [
        "# Top10 Daily Update Operational Report",
        "",
        f"- Pipeline status: {'PASS' if passed_steps else 'FAIL'}",
        f"- Quality status: {'PASS' if passed_quality else 'FAIL'}",
        f"- Generated at UTC: {now_utc_iso()}",
        "",
        "## Step Manifest",
        "",
        "| Step | Status | Duration Sec |",
        "|---|---:|---:|",
    ]
    for result in manifest:
        lines.append(f"| {result.step} | {result.status} | {result.duration_sec:.2f} |")

    lines.extend(["", "## Quality Checks", "", "| Check | Passed | Value |", "|---|---:|---:|"])
    for _, row in quality.iterrows():
        lines.append(f"| {row['check']} | {row['passed']} | {row['value']} |")

    if latest_risk.exists():
        risk = pd.read_csv(latest_risk)
        risk_map = dict(zip(risk["field"], risk["value"]))
        lines.extend(
            [
                "",
                "## Latest Risk",
                f"- Date: {risk_map.get('date', 'NA')}",
                f"- Risk state: {risk_map.get('risk_state', 'NA')}",
                f"- Final max weight: {risk_map.get('final_recommended_max_weight', 'NA')}%",
                f"- Limiting reason: {risk_map.get('limiting_reason', 'NA')}",
            ]
        )

    if latest_stress.exists():
        stress = pd.read_csv(latest_stress)
        stress_map = dict(zip(stress["field"], stress["value"]))
        lines.extend(
            [
                "",
                "## Latest Stress",
                f"- Stress status: {stress_map.get('stress_status', 'NA')}",
                f"- Worst scenario: {stress_map.get('worst_current_weight_scenario', 'NA')}",
                f"- Estimated portfolio impact: {format_percent_value(stress_map.get('worst_current_weight_portfolio_impact_pct'))}",
            ]
        )

    if latest_prediction.exists():
        prediction = pd.read_csv(latest_prediction)
        prediction_map = dict(zip(prediction["field"], prediction["value"]))
        lines.extend(
            [
                "",
                "## Latest Prediction",
                f"- Prediction status: {prediction_map.get('prediction_signal_status', 'NA')}",
                f"- Use status: {prediction_map.get('prediction_use_status', 'NA')}",
                f"- 20D probability: {prediction_map.get('p_success_20d', 'NA')}",
                f"- 20D threshold: {prediction_map.get('threshold_20d', 'NA')}",
                f"- Next-day up probability: {prediction_map.get('next_day_p_up_1d', 'NA')}",
                f"- Next-day up status: {prediction_map.get('next_day_prediction_signal_status', 'NA')}",
            ]
        )

    if latest_system.exists():
        system = pd.read_csv(latest_system)
        system_map = dict(zip(system["field"], system["value"]))
        lines.extend(
            [
                "",
                "## System State",
                f"- System state: {system_map.get('system_state', 'NA')}",
                f"- Readiness score: {system_map.get('system_readiness_score', 'NA')}/100",
                f"- Paper trading status: {system_map.get('paper_trading_status', 'NA')}",
                f"- Live trading status: {system_map.get('live_trading_status', 'NA')}",
            ]
        )

    if latest_health.exists():
        health = pd.read_csv(latest_health)
        health_map = dict(zip(health["field"], health["value"]))
        lines.extend(
            [
                "",
                "## Daily Health",
                f"- Health status: {health_map.get('daily_health_status', 'NA')}",
                f"- Failed or blocked components: {health_map.get('failed_or_blocked_components', 'NA')}",
            ]
        )

    if latest_news.exists():
        news = pd.read_csv(latest_news)
        if not news.empty:
            last_news = news.tail(1).iloc[0]
            lines.extend(
                [
                    "",
                    "## News Causes",
                    f"- Latest news date: {last_news.get('date', 'NA')}",
                    f"- Coverage: {last_news.get('news_coverage_status', 'NA')}",
                    f"- Primary cause: {last_news.get('news_primary_cause_type', 'NA')}",
                    f"- Confidence: {last_news.get('news_match_confidence', 'NA')}",
                ]
            )

    lines.extend(
        [
            "",
            "## Decision Report",
            f"- Latest trading plan: {latest_plan.name if latest_plan.exists() else 'missing'}",
            "",
            "No orders are placed by this pipeline.",
        ]
    )
    (rule_outdir / "tsm_daily_update_operational_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Top10 daily research update without placing orders.")
    parser.add_argument("--config", default="config/tsm_research.toml")
    parser.add_argument("--start", default="2016-05-12")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--preferred-source", choices=["stooq", "yahoo"], default="stooq")
    parser.add_argument("--kr-preferred-source", choices=["stooq", "yahoo"], default="yahoo")
    parser.add_argument("--events", default="tsm_events_seed.csv")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--rule-outdir", default="tsm_price_rule_output")
    parser.add_argument("--fx-rates", default="output/tsm_fx_rates_daily.csv", help="Daily FX rates used for non-USD listings.")
    parser.add_argument("--skip-data-refresh", action="store_true", help="Use existing output/*.csv files and rerun downstream engines only.")
    parser.add_argument("--skip-news-refresh", action="store_true", help="Use existing news cause files and skip direct-web news collection.")
    parser.add_argument("--news-backfill", action="store_true", help="Collect direct-web news using the configured historical backfill start.")
    parser.add_argument("--news-enabled", action="store_true", default=True)
    parser.add_argument("--news-sources", default="tsmc_press,tsmc_investor,google_news_rss,yahoo_finance")
    parser.add_argument("--news-lookback-days", type=int, default=14)
    parser.add_argument("--news-backfill-start", default="2016-05-12")
    parser.add_argument("--news-politeness-delay-sec", type=float, default=1.0)
    parser.add_argument("--news-market-timezone", default="America/New_York")
    parser.add_argument("--skip-charts", action="store_true")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stop-multiple", type=float, default=2.0)
    parser.add_argument("--score-threshold", type=float, default=75.0)
    parser.add_argument("--schema-strict", action="store_true")
    parser.add_argument("--prediction-policy", choices=["conservative", "research"], default="conservative")
    parser.add_argument("--universe-mode", choices=["universe", "hybrid"], default="hybrid")
    parser.add_argument("--universe-config", default="config/semiconductor_universe_top10.csv")
    parser.add_argument("--decision-universe-config", default="config/semiconductor_universe_top10.csv")
    parser.add_argument("--research-universe-config", default="config/semiconductor_universe_expanded.csv")
    parser.add_argument("--skip-universe-symbol-build", action="store_true", help="Rebuild pooled aggregates from existing per-symbol universe outputs.")
    parser.add_argument("--skip-universe-symbol-diagnostics", action="store_true", help="Skip per-symbol diagnostics during universe symbol builds.")
    parser.add_argument("--skip-external-web", action="store_true", help="Build external features from local files only.")
    parser.add_argument("--enable-feedback-features", action="store_true", help="Build prefix-only backtest feedback feature outputs.")
    parser.add_argument("--enable-auto-research", action="store_true", help="Run automatic research trial ledger after pooled model outputs.")
    parser.add_argument("--max-trials", type=int, default=100, help="Maximum auto-research trials when enabled.")
    args = parser.parse_args()
    return apply_config_defaults(args, parser, load_run_config(args.config))


def main() -> None:
    args = parse_args()
    cwd = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    rule_outdir = Path(args.rule_outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rule_outdir.mkdir(parents=True, exist_ok=True)
    decision_universe_config = str(args.decision_universe_config or args.universe_config)
    research_universe_config = str(args.research_universe_config)

    py = sys.executable
    steps: List[tuple[str, List[str]]] = []

    if not args.skip_data_refresh:
        steps.append(
            (
                "fx_rate_engine",
                [
                    py,
                    "tsm_fx_rate_engine.py",
                    "--start",
                    args.start,
                    "--end",
                    args.end,
                    "--outdir",
                    str(Path(args.fx_rates).parent),
                    "--pairs",
                    "KRW=X",
                ],
            )
        )
        cmd = [
            py,
            "tsm_daily_quant_pipeline.py",
            "--start",
            args.start,
            "--end",
            args.end,
            "--preferred-source",
            args.preferred_source,
            "--events",
            args.events,
            "--outdir",
            str(output_dir),
            "--fx-rates",
            str(args.fx_rates),
        ]
        if args.skip_charts:
            cmd.append("--skip-charts")
        if args.skip_benchmarks:
            cmd.append("--skip-benchmarks")
        steps.append(("data_pipeline", cmd))

    if args.news_enabled and not args.skip_news_refresh:
        news_mode = "backfill" if args.news_backfill else "daily"
        news_start = args.news_backfill_start if args.news_backfill else args.start
        steps.append(
            (
                "news_causal_engine",
                [
                    py,
                    "tsm_news_causal_engine.py",
                    "--enriched",
                    str(output_dir / "tsm_daily_10y_enriched.csv"),
                    "--events",
                    str(args.events),
                    "--start",
                    str(news_start),
                    "--end",
                    str(args.end),
                    "--mode",
                    news_mode,
                    "--lookback-days",
                    str(args.news_lookback_days),
                    "--sources",
                    str(args.news_sources),
                    "--outdir",
                    str(output_dir),
                    "--rule-outdir",
                    str(rule_outdir),
                    "--market-timezone",
                    str(args.news_market_timezone),
                    "--politeness-delay-sec",
                    str(args.news_politeness_delay_sec),
                ],
            )
        )

    steps.extend(
        [
            (
                "rule_engine",
                [
                    py,
                    "tsm_price_rule_engine.py",
                    "--enriched",
                    str(output_dir / "tsm_daily_10y_enriched.csv"),
                    "--raw",
                    str(output_dir / "tsm_daily_10y_raw.csv"),
                    "--summary",
                    str(output_dir / "tsm_daily_10y_summary.csv"),
                    "--events",
                    str(output_dir / "tsm_event_impact_10y.csv"),
                    "--news-daily",
                    str(rule_outdir / "tsm_news_integrated_daily.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--config",
                    str(args.config),
                    "--symbol",
                    "TSM",
                    "--symbol-group",
                    "semiconductor",
                ],
            ),
            (
                "data_quality_engine",
                [
                    py,
                    "tsm_data_quality_engine.py",
                    "--raw",
                    str(output_dir / "tsm_daily_10y_raw.csv"),
                    "--enriched",
                    str(output_dir / "tsm_daily_10y_enriched.csv"),
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--run-date",
                    str(args.end),
                ],
            ),
            (
                "backtest_engine",
                [
                    py,
                    "tsm_backtest_engine.py",
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--enriched",
                    str(output_dir / "tsm_daily_10y_enriched.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--commission-bps",
                    str(args.commission_bps),
                    "--slippage-bps",
                    str(args.slippage_bps),
                    "--stop-multiple",
                    str(args.stop_multiple),
                    "--score-threshold",
                    str(args.score_threshold),
                ],
            ),
            (
                "risk_engine",
                [
                    py,
                    "tsm_risk_engine.py",
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--root-symbol",
                    "TSM",
                    "--root-signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                ],
            ),
            (
                "backtest_event_ledger",
                [
                    py,
                    "tsm_backtest_event_ledger.py",
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--trade-log",
                    str(rule_outdir / "tsm_backtest_trade_log.csv"),
                    "--enriched",
                    str(output_dir / "tsm_daily_10y_enriched.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--symbol",
                    "TSM",
                    "--commission-bps",
                    str(args.commission_bps),
                    "--slippage-bps",
                    str(args.slippage_bps),
                ],
            ),
            (
                "validation_engine",
                [
                    py,
                    "tsm_validation_engine.py",
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--enriched",
                    str(output_dir / "tsm_daily_10y_enriched.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--baseline-stop-multiple",
                    str(args.stop_multiple),
                    "--baseline-score-threshold",
                    str(args.score_threshold),
                    "--commission-bps",
                    str(args.commission_bps),
                    "--slippage-bps",
                    str(args.slippage_bps),
                    "--model-trial-count",
                    "60",
                ],
            ),
            (
                "stress_engine",
                [
                    py,
                    "tsm_daily_stress_engine.py",
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--risk-policy",
                    str(rule_outdir / "tsm_risk_policy_daily.csv"),
                    "--drawdowns",
                    str(rule_outdir / "tsm_drawdown_episodes.csv"),
                    "--equity-curves",
                    str(rule_outdir / "tsm_backtest_equity_curves.csv"),
                    "--intraday-features",
                    str(rule_outdir / "tsm_intraday_daily_features.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
            (
                "integrity_engine",
                [
                    py,
                    "tsm_daily_integrity_engine.py",
                    "--raw",
                    str(output_dir / "tsm_daily_10y_raw.csv"),
                    "--enriched",
                    str(output_dir / "tsm_daily_10y_enriched.csv"),
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--risk-policy",
                    str(rule_outdir / "tsm_risk_policy_daily.csv"),
                    "--trade-log",
                    str(rule_outdir / "tsm_backtest_trade_log.csv"),
                    "--equity-curves",
                    str(rule_outdir / "tsm_backtest_equity_curves.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
            (
                "external_feature_engine",
                [
                    py,
                    "tsm_external_feature_engine.py",
                    "--outdir",
                    str(rule_outdir),
                    "--universe-config",
                    research_universe_config,
                    "--news-events",
                    str(output_dir / "tsm_news_events_normalized.csv"),
                    "--start",
                    str(args.start),
                    "--end",
                    str(args.end),
                    *(["--skip-web"] if args.skip_external_web else []),
                ],
            ),
            *(
                [
                    (
                        "universe_daily_refresh",
                        [
                            py,
                            "run_pooled_universe_update.py",
                            "--symbol-build-only",
                            *(["--skip-symbol-diagnostics"] if args.skip_universe_symbol_diagnostics else []),
                            "--universe-config",
                            decision_universe_config,
                            "--decision-universe-config",
                            decision_universe_config,
                            "--research-universe-config",
                            research_universe_config,
                            "--outdir",
                            str(rule_outdir),
                            "--start",
                            str(args.start),
                            "--end",
                            str(args.end),
                            "--preferred-source",
                            str(args.preferred_source),
                            "--kr-preferred-source",
                            str(args.kr_preferred_source),
                            "--fx-rates",
                            str(args.fx_rates),
                            "--commission-bps",
                            str(args.commission_bps),
                            "--slippage-bps",
                            str(args.slippage_bps),
                            "--stop-multiple",
                            str(args.stop_multiple),
                        ],
                    )
                ]
                if not args.skip_universe_symbol_build and not args.skip_data_refresh
                else []
            ),
            (
                "top10_intraday_market_data",
                [
                    py,
                    "run_universe_market_data_update.py",
                    "--universe-config",
                    decision_universe_config,
                    "--outdir",
                    str(output_dir),
                    "--start",
                    str(args.start),
                    "--end",
                    str(args.end),
                    "--bar-scope",
                    "both",
                    "--model-minute-interval",
                    "5m",
                    "--execution-minute-interval",
                    "1m",
                    "--provider",
                    "auto",
                    "--provider-order",
                    "alpha_vantage,polygon,eodhd,alpaca,yahoo",
                    "--fx-rates",
                    str(args.fx_rates),
                    "--skip-charts",
                    "--continue-on-error",
                ],
            ),
            (
                "intraday_feature_engine",
                [
                    py,
                    "tsm_intraday_feature_engine.py",
                    "--decision-universe-config",
                    decision_universe_config,
                    "--research-universe-config",
                    research_universe_config,
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
            (
                "prediction_engine",
                [
                    py,
                    "tsm_prediction_engine.py",
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--trade-log",
                    str(rule_outdir / "tsm_backtest_trade_log.csv"),
                    "--risk-policy",
                    str(rule_outdir / "tsm_risk_policy_daily.csv"),
                    "--enriched",
                    str(output_dir / "tsm_daily_10y_enriched.csv"),
                    "--external-features",
                    str(rule_outdir / "tsm_external_daily_features.csv"),
                    "--intraday-features",
                    str(rule_outdir / "tsm_intraday_daily_features.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--commission-bps",
                    str(args.commission_bps),
                    "--slippage-bps",
                    str(args.slippage_bps),
                    "--stop-multiple",
                    str(args.stop_multiple),
                    "--config",
                    str(args.config),
                    "--prediction-policy",
                    str(args.prediction_policy),
                    "--symbol",
                    "TSM",
                    *(["--schema-strict"] if args.schema_strict else []),
                ],
            ),
            (
                "next_day_up_model_engine",
                [
                    py,
                    "tsm_next_day_up_model_engine.py",
                    "--aggregate-universe",
                    "--universe-config",
                    decision_universe_config,
                    "--universe-rule-root",
                    str(rule_outdir / "universe"),
                    "--universe-data-root",
                    str(output_dir / "universe"),
                    "--external-features",
                    str(rule_outdir / "tsm_external_daily_features.csv"),
                    "--intraday-features",
                    str(rule_outdir / "tsm_intraday_daily_features.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--latest-prediction",
                    str(rule_outdir / "tsm_latest_prediction_snapshot.csv"),
                    "--symbol",
                    "TSM",
                ],
            ),
            (
                "ml_overlay_backtest",
                [
                    py,
                    "tsm_ml_overlay_backtest.py",
                    "--oos-predictions",
                    str(rule_outdir / "tsm_prediction_oos_predictions.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--enriched",
                    str(output_dir / "tsm_daily_10y_enriched.csv"),
                    "--smh-enriched",
                    "output/universe/SMH/tsm_daily_10y_enriched.csv",
                    "--soxx-enriched",
                    "output/universe/SOXX/tsm_daily_10y_enriched.csv",
                ],
            ),
            (
                "pooled_dataset_builder",
                [
                    py,
                    "run_pooled_universe_update.py",
                    "--skip-symbol-build",
                    *(["--skip-symbol-diagnostics"] if args.skip_universe_symbol_diagnostics else []),
                    "--universe-config",
                    decision_universe_config,
                    "--decision-universe-config",
                    decision_universe_config,
                    "--research-universe-config",
                    research_universe_config,
                    "--outdir",
                    str(rule_outdir),
                    "--start",
                    str(args.start),
                    "--end",
                    str(args.end),
                    "--preferred-source",
                    str(args.preferred_source),
                    "--kr-preferred-source",
                    str(args.kr_preferred_source),
                    "--fx-rates",
                    str(args.fx_rates),
                    "--commission-bps",
                    str(args.commission_bps),
                    "--slippage-bps",
                    str(args.slippage_bps),
                    "--stop-multiple",
                    str(args.stop_multiple),
                    "--external-features",
                    str(rule_outdir / "tsm_external_daily_features.csv"),
                    "--intraday-features",
                    str(rule_outdir / "tsm_intraday_daily_features.csv"),
                ],
            ),
            (
                "next_close_forecast_engine",
                [
                    py,
                    "tsm_next_close_forecast_engine.py",
                    "--pooled-feature-matrix",
                    str(rule_outdir / "tsm_prediction_pooled_feature_matrix.csv"),
                    "--decision-universe-config",
                    decision_universe_config,
                    "--latest-prediction",
                    str(rule_outdir / "tsm_latest_prediction_snapshot.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--symbol",
                    "TSM",
                ],
            ),
            (
                "pooled_model_engine",
                [
                    py,
                    "tsm_pooled_model_engine.py",
                    "--pooled-feature-matrix",
                    str(rule_outdir / "tsm_prediction_pooled_feature_matrix.csv"),
                    "--pooled-quality",
                    str(rule_outdir / "tsm_prediction_pooled_quality_checks.csv"),
                    "--latest-prediction",
                    str(rule_outdir / "tsm_latest_prediction_snapshot.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
            (
                "model_gate_engine",
                [
                    py,
                    "tsm_model_gate_engine.py",
                    "--comparison",
                    str(rule_outdir / "tsm_prediction_model_comparison.csv"),
                    "--latest-prediction",
                    str(rule_outdir / "tsm_latest_prediction_snapshot.csv"),
                    "--pooled-comparison",
                    str(rule_outdir / "tsm_pooled_model_comparison.csv"),
                    "--pooled-latest",
                    str(rule_outdir / "tsm_pooled_latest_prediction_overlay.csv"),
                    "--tsm-calibration",
                    str(rule_outdir / "tsm_pooled_tsm_calibration_metrics.csv"),
                    "--cpcv-strategy-distribution",
                    str(rule_outdir / "tsm_cpcv_strategy_distribution.csv"),
                    "--cpcv-model-distribution",
                    str(rule_outdir / "tsm_cpcv_model_distribution.csv"),
                    "--pbo-report",
                    str(rule_outdir / "tsm_pbo_report.csv"),
                    "--cscv-pbo-report",
                    str(rule_outdir / "tsm_cscv_pbo_report.csv"),
                    "--dsr-report",
                    str(rule_outdir / "tsm_deflated_sharpe_report.csv"),
                    "--paper-gate",
                    str(rule_outdir / "tsm_paper_gate_snapshot.csv"),
                    "--next-close-comparison",
                    str(rule_outdir / "tsm_next_close_model_comparison.csv"),
                    "--next-close-quality",
                    str(rule_outdir / "tsm_next_close_quality_checks.csv"),
                    "--next-close-latest",
                    str(rule_outdir / "tsm_next_close_latest_snapshot.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
            (
                "model_registry_engine",
                [
                    py,
                    "tsm_model_registry_engine.py",
                    "--outdir",
                    str(rule_outdir),
                    "--config",
                    str(args.config),
                    "--label-dataset",
                    str(rule_outdir / "tsm_prediction_label_dataset.csv"),
                    "--feature-matrix",
                    str(rule_outdir / "tsm_prediction_feature_matrix.csv"),
                    "--feature-contract",
                    str(rule_outdir / "tsm_prediction_feature_contract.csv"),
                    "--fold-manifest",
                    str(rule_outdir / "tsm_prediction_fold_manifest.csv"),
                    "--schema-quality",
                    str(rule_outdir / "tsm_schema_quality_checks.csv"),
                    "--prediction-engine-source",
                    "tsm_prediction_engine.py",
                ],
            ),
            (
                "shadow_paper_engine",
                [
                    py,
                    "tsm_shadow_paper_engine.py",
                    "--prediction",
                    str(rule_outdir / "tsm_latest_prediction_snapshot.csv"),
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--ledger",
                    str(rule_outdir / "tsm_shadow_paper_predictions.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--commission-bps",
                    str(args.commission_bps),
                    "--slippage-bps",
                    str(args.slippage_bps),
                    "--stop-multiple",
                    str(args.stop_multiple),
                ],
            ),
            (
                "pre_paper_system_state_engine",
                [
                    py,
                    "tsm_system_state_engine.py",
                    "--operational-quality",
                    str(rule_outdir / "tsm_operational_quality_checks.csv"),
                    "--integrity-quality",
                    str(rule_outdir / "tsm_daily_integrity_checks.csv"),
                    "--validation-quality",
                    str(rule_outdir / "tsm_validation_quality_checks.csv"),
                    "--prediction-quality",
                    str(rule_outdir / "tsm_prediction_quality_checks.csv"),
                    "--prediction-snapshot",
                    str(rule_outdir / "tsm_latest_prediction_snapshot.csv"),
                    "--risk-snapshot",
                    str(rule_outdir / "tsm_latest_risk_snapshot.csv"),
                    "--stress-snapshot",
                    str(rule_outdir / "tsm_latest_stress_snapshot.csv"),
                    "--backtest-summary",
                    str(rule_outdir / "tsm_backtest_strategy_summary.csv"),
                    "--walk-forward",
                    str(rule_outdir / "tsm_validation_walk_forward_summary.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
            (
                "order_intent_engine",
                [
                    py,
                    "tsm_order_intent_engine.py",
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--risk",
                    str(rule_outdir / "tsm_latest_risk_snapshot.csv"),
                    "--prediction",
                    str(rule_outdir / "tsm_latest_prediction_snapshot.csv"),
                    "--system-state",
                    str(rule_outdir / "tsm_latest_system_state.csv"),
                    "--ledger",
                    str(rule_outdir / "tsm_order_intents.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--universe-config",
                    decision_universe_config,
                    "--latest-predictions",
                    str(rule_outdir / "tsm_universe_latest_predictions.csv"),
                    "--signals-root",
                    str(rule_outdir),
                    "--config",
                    str(args.config),
                ],
            ),
            (
                "portfolio_risk_engine",
                [
                    py,
                    "tsm_portfolio_risk_engine.py",
                    "--intents",
                    str(rule_outdir / "tsm_order_intents.csv"),
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--latest-signals",
                    str(rule_outdir / "tsm_universe_latest_signals.csv"),
                    "--intraday-features",
                    str(rule_outdir / "tsm_intraday_daily_features.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--config",
                    str(args.config),
                    "--decision-universe-config",
                    decision_universe_config,
                ],
            ),
            (
                "paper_execution_engine",
                [
                    py,
                    "tsm_paper_execution_engine.py",
                    "--intents",
                    str(rule_outdir / "tsm_order_intents.csv"),
                    "--portfolio-decisions",
                    str(rule_outdir / "tsm_portfolio_risk_order_decisions.csv"),
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--signals-root",
                    str(rule_outdir),
                    "--universe-config",
                    decision_universe_config,
                    "--intraday-features",
                    str(rule_outdir / "tsm_intraday_daily_features.csv"),
                    "--orders",
                    str(rule_outdir / "tsm_paper_orders.csv"),
                    "--fills",
                    str(rule_outdir / "tsm_paper_fills.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--config",
                    str(args.config),
                    "--commission-bps",
                    str(args.commission_bps),
                ],
            ),
            (
                "position_reconciler",
                [
                    py,
                    "tsm_position_reconciler.py",
                    "--positions",
                    str(rule_outdir / "tsm_paper_positions.csv"),
                    "--fills",
                    str(rule_outdir / "tsm_paper_fills.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
            (
                "order_state_machine",
                [
                    py,
                    "tsm_order_state_machine.py",
                    "--intents",
                    str(rule_outdir / "tsm_order_intents.csv"),
                    "--portfolio-decisions",
                    str(rule_outdir / "tsm_portfolio_risk_order_decisions.csv"),
                    "--orders",
                    str(rule_outdir / "tsm_paper_orders.csv"),
                    "--fills",
                    str(rule_outdir / "tsm_paper_fills.csv"),
                    "--reconciliation",
                    str(rule_outdir / "tsm_paper_reconciliation_report.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
            (
                "execution_feedback_engine",
                [
                    py,
                    "tsm_execution_feedback_engine.py",
                    "--orders",
                    str(rule_outdir / "tsm_paper_orders.csv"),
                    "--fills",
                    str(rule_outdir / "tsm_paper_fills.csv"),
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--config",
                    str(args.config),
                ],
            ),
            (
                "fill_model_calibration_engine",
                [
                    py,
                    "tsm_fill_model_calibration_engine.py",
                    "--feedback-events",
                    str(rule_outdir / "tsm_execution_feedback_events.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--config",
                    str(args.config),
                ],
            ),
            (
                "automation_scheduler",
                [
                    py,
                    "tsm_automation_scheduler.py",
                    "--lifecycle-snapshot",
                    str(rule_outdir / "tsm_order_lifecycle_snapshot.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--config",
                    str(args.config),
                ],
            ),
            (
                "system_state_engine",
                [
                    py,
                    "tsm_system_state_engine.py",
                    "--operational-quality",
                    str(rule_outdir / "tsm_operational_quality_checks.csv"),
                    "--integrity-quality",
                    str(rule_outdir / "tsm_daily_integrity_checks.csv"),
                    "--validation-quality",
                    str(rule_outdir / "tsm_validation_quality_checks.csv"),
                    "--prediction-quality",
                    str(rule_outdir / "tsm_prediction_quality_checks.csv"),
                    "--prediction-snapshot",
                    str(rule_outdir / "tsm_latest_prediction_snapshot.csv"),
                    "--risk-snapshot",
                    str(rule_outdir / "tsm_latest_risk_snapshot.csv"),
                    "--stress-snapshot",
                    str(rule_outdir / "tsm_latest_stress_snapshot.csv"),
                    "--backtest-summary",
                    str(rule_outdir / "tsm_backtest_strategy_summary.csv"),
                    "--walk-forward",
                    str(rule_outdir / "tsm_validation_walk_forward_summary.csv"),
                    "--paper-oms-quality",
                    str(rule_outdir / "tsm_paper_oms_quality_checks.csv"),
                    "--paper-reconciliation-quality",
                    str(rule_outdir / "tsm_paper_reconciliation_quality_checks.csv"),
                    "--order-state-quality",
                    str(rule_outdir / "tsm_order_state_quality_checks.csv"),
                    "--execution-feedback-quality",
                    str(rule_outdir / "tsm_execution_feedback_quality_checks.csv"),
                    "--fill-calibration-quality",
                    str(rule_outdir / "tsm_fill_model_calibration_quality_checks.csv"),
                    "--automation-quality",
                    str(rule_outdir / "tsm_automation_quality_checks.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
            (
                "daily_trading_report",
                [
                    py,
                    "tsm_daily_trading_report.py",
                    "--latest",
                    str(rule_outdir / "tsm_latest_decision_snapshot.csv"),
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--risk",
                    str(rule_outdir / "tsm_latest_risk_snapshot.csv"),
                    "--backtest-summary",
                    str(rule_outdir / "tsm_backtest_strategy_summary.csv"),
                    "--validation-quality",
                    str(rule_outdir / "tsm_validation_quality_checks.csv"),
                    "--walk-forward",
                    str(rule_outdir / "tsm_validation_walk_forward_summary.csv"),
                    "--stress",
                    str(rule_outdir / "tsm_latest_stress_snapshot.csv"),
                    "--system-state",
                    str(rule_outdir / "tsm_latest_system_state.csv"),
                    "--prediction",
                    str(rule_outdir / "tsm_latest_prediction_snapshot.csv"),
                    "--order-intents",
                    str(rule_outdir / "tsm_order_intents.csv"),
                    "--universe-latest-predictions",
                    str(rule_outdir / "tsm_universe_latest_predictions.csv"),
                    "--portfolio-targets",
                    str(rule_outdir / "tsm_portfolio_targets.csv"),
                    "--paper-positions",
                    str(rule_outdir / "tsm_paper_positions.csv"),
                    "--paper-reconciliation",
                    str(rule_outdir / "tsm_paper_reconciliation_report.csv"),
                    "--order-lifecycle",
                    str(rule_outdir / "tsm_order_lifecycle_snapshot.csv"),
                    "--execution-feedback",
                    str(rule_outdir / "tsm_execution_feedback_events.csv"),
                    "--fill-calibration",
                    str(rule_outdir / "tsm_fill_model_calibration.csv"),
                    "--automation-plan",
                    str(rule_outdir / "tsm_automation_plan.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
            (
                "daily_health_report",
                [
                    py,
                    "tsm_daily_health_report.py",
                    "--data-quality",
                    str(rule_outdir / "tsm_latest_data_quality_snapshot.csv"),
                    "--system-state",
                    str(rule_outdir / "tsm_latest_system_state.csv"),
                    "--prediction",
                    str(rule_outdir / "tsm_latest_prediction_snapshot.csv"),
                    "--model-gate",
                    str(rule_outdir / "tsm_model_gate_snapshot.csv"),
                    "--risk",
                    str(rule_outdir / "tsm_latest_risk_snapshot.csv"),
                    "--shadow-quality",
                    str(rule_outdir / "tsm_shadow_paper_quality_checks.csv"),
                    "--operational-quality",
                    str(rule_outdir / "tsm_operational_quality_checks.csv"),
                    "--paper-oms-quality",
                    str(rule_outdir / "tsm_paper_oms_quality_checks.csv"),
                    "--paper-reconciliation-quality",
                    str(rule_outdir / "tsm_paper_reconciliation_quality_checks.csv"),
                    "--order-state-quality",
                    str(rule_outdir / "tsm_order_state_quality_checks.csv"),
                    "--execution-feedback-quality",
                    str(rule_outdir / "tsm_execution_feedback_quality_checks.csv"),
                    "--fill-calibration-quality",
                    str(rule_outdir / "tsm_fill_model_calibration_quality_checks.csv"),
                    "--automation-quality",
                    str(rule_outdir / "tsm_automation_quality_checks.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
        ]
    )

    def insert_before(step_name: str, new_step: tuple[str, List[str]]) -> None:
        for idx, (name, _) in enumerate(steps):
            if name == step_name:
                steps.insert(idx, new_step)
                return
        steps.append(new_step)

    if args.enable_feedback_features:
        insert_before(
            "model_gate_engine",
            (
                "backtest_feedback_feature_engine",
                [
                    py,
                    "tsm_backtest_feedback_feature_engine.py",
                    "--ledger",
                    str(rule_outdir / "tsm_backtest_event_ledger.csv"),
                    "--oof-predictions",
                    str(rule_outdir / "tsm_pooled_model_oof_predictions.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
        )

    if args.enable_auto_research:
        insert_before(
            "model_gate_engine",
            (
                "auto_research_engine",
                [
                    py,
                    "tsm_auto_research_engine.py",
                    "--pooled-comparison",
                    str(rule_outdir / "tsm_pooled_model_comparison.csv"),
                    "--local-comparison",
                    str(rule_outdir / "tsm_prediction_model_comparison.csv"),
                    "--pbo-report",
                    str(rule_outdir / "tsm_pbo_report.csv"),
                    "--dsr-report",
                    str(rule_outdir / "tsm_deflated_sharpe_report.csv"),
                    "--cpcv-model-distribution",
                    str(rule_outdir / "tsm_cpcv_model_distribution.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--storage",
                    f"sqlite:///{rule_outdir / 'tsm_research_trials.sqlite'}",
                    "--max-trials",
                    str(args.max_trials),
                ],
            ),
        )

    results: List[StepResult] = []
    research_warning_steps = {"external_feature_engine", "universe_daily_refresh", "pooled_dataset_builder", "next_close_forecast_engine", "pooled_model_engine"}
    partial_intraday_steps = {"top10_intraday_market_data", "intraday_feature_engine"}
    for step, command in steps:
        result = run_step(step, command, cwd)
        if step in partial_intraday_steps and result.returncode != 0:
            result.status = "PARTIAL_INTRADAY"
        results.append(result)
        write_manifest(rule_outdir, results)
        print(f"{step}: {result.status} ({result.duration_sec:.2f}s)")
        if step == "news_causal_engine" and result.returncode != 0:
            print("WARNING: news causal engine failed; continuing with existing or empty news outputs.")
            continue
        if step in partial_intraday_steps and result.returncode != 0:
            print(f"WARNING: {step} failed; continuing with PARTIAL_INTRADAY status and latest existing intraday features.")
            continue
        if step in research_warning_steps and result.returncode != 0:
            print(f"WARNING: {step} failed; continuing because this is a model research/support stage.")
            continue
        if result.returncode != 0 and not args.continue_on_error:
            quality = build_data_quality_report(output_dir, rule_outdir, args.end)
            write_operational_report(rule_outdir, results, quality)
            raise SystemExit(result.returncode)
        if step == "automation_scheduler" and result.returncode == 0:
            build_data_quality_report(output_dir, rule_outdir, args.end, include_system_outputs=False)

    quality = build_data_quality_report(output_dir, rule_outdir, args.end, include_system_outputs=True)
    write_manifest(rule_outdir, results)
    write_operational_report(rule_outdir, results, quality)

    if not quality["passed"].all():
        print("WARNING: operational quality checks failed. See tsm_operational_quality_checks.csv")
    print("완료: daily update pipeline =", rule_outdir.resolve())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local dashboard for the Top10 semiconductor rule-engine package.

The dashboard keeps the existing research scripts as the source of truth. It
only reads their CSV/Markdown outputs and provides a small local API for
refreshing the full pipeline or individual engines.
"""

from __future__ import annotations

import argparse
import errno
import io
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd

from tsm_env import load_project_env


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "dashboard"
OUTPUT_DIR = ROOT / "output"
RULE_DIR = ROOT / "tsm_price_rule_output"
ALLOWED_FILE_ROOTS = [OUTPUT_DIR.resolve(), RULE_DIR.resolve()]
KR2_AUTOMATION_SYMBOLS = {"005930.KS", "000660.KS"}

DEFAULT_START = "2016-05-12"
DEFAULT_END = date.today().isoformat()
SUMMARY_CACHE_TTL_SECONDS = 30.0
MAX_DASHBOARD_LINE_COUNT_BYTES = 5 * 1024 * 1024
KOREAEXIM_EXCHANGE_URL = "https://oapi.koreaexim.go.kr/site/program/financial/exchangeJSON"

load_project_env()

DASHBOARD_POOLED_OOS_COLUMNS = [
    "symbol",
    "symbol_group",
    "date",
    "split",
    "candidate_tier",
    "is_model_training_candidate",
    "is_decision_entry_candidate",
    "is_trade_ready_entry_candidate",
    "label_success_20d",
    "label_net_return_pct_20d",
    "p_success_base",
    "p_success_eb",
    "p_success_logistic",
    "p_success_lgbm",
    "p_success_xgb",
    "p_success_stack_raw",
    "p_success_calibrated",
    "p_success_tsm_calibrated",
    "p_stop_hit",
    "p_stop_hit_lgbm",
    "p_stop_hit_raw",
    "p_stop_hit_calibrated",
    "p_stop_hit_raw_minus_calibrated",
    "p_stop_hit_oos_percentile",
    "stop_risk_calibration_warning",
    "expected_r_net",
    "expected_r_lgbm",
    "expected_net_return_pct",
    "decision_score",
    "decision_score_tsm_calibrated",
    "utility_score",
    "raw_utility_score",
    "threshold",
    "selected_by_threshold",
    "threshold_reason",
    "threshold_decision_eligible",
    "entry_trigger",
    "trend_regime",
    "vol_regime",
    "drawdown_bucket",
]
DASHBOARD_POOLED_OOF_COLUMNS = [
    "fold_id",
    "split",
    "validation_design",
    "model_name",
    "evaluation_scope",
    "symbol",
    "symbol_group",
    "date",
    "candidate_tier",
    "entry_trigger",
    "trend_regime",
    "vol_regime",
    "label_success_20d",
    "label_net_return_pct_20d",
    "p_success",
    "utility_score",
    "threshold",
    "selected_by_threshold",
    "threshold_reason",
    "threshold_decision_eligible",
]
DASHBOARD_EVENT_LEDGER_COLUMNS = [
    "symbol",
    "symbol_group",
    "date",
    "strategy_id",
    "variant_id",
    "horizon_days",
    "stop_multiple",
    "candidate_tier",
    "entry_trigger",
    "trade_action",
    "strict_signal_stage",
    "research_signal_stage",
    "research_signal_action",
    "research_signal_score",
    "entry_signal_pass",
    "entry_failure_reason",
    "next_open_available",
    "target_weight_pct",
    "position_overlap_flag",
    "matched_trade_flag",
    "exit_reason",
    "realized_r_multiple",
    "net_return_pct",
    "stop_hit",
    "hit_1r",
    "hit_2r",
    "label_overlap_count",
    "sample_uniqueness_weight",
]
DASHBOARD_FEEDBACK_COLUMNS = [
    "symbol",
    "date",
    "strategy_id",
    "variant_id",
    "candidate_tier",
    "entry_trigger",
    "fb_trigger_success_rate_ewm_60",
    "fb_trigger_expected_r_ewm_60",
    "fb_candidate_tier_success_rate_ewm_120",
    "fb_symbol_stop_rate_252",
    "fb_symbol_expected_r_20",
    "fb_symbol_expected_r_60",
    "fb_symbol_expected_r_120",
    "fb_model_calibration_residual_ewm_120",
    "fb_threshold_selected_uplift_ewm_120",
    "fb_strategy_drawdown_sensitivity_252",
]


RUN_LOCK = threading.Lock()
SUMMARY_CACHE_LOCK = threading.Lock()
RUN_PROCESS: subprocess.Popen[str] | None = None
RUN_STATE: dict[str, Any] = {
    "running": False,
    "mode": None,
    "command": [],
    "started_at": None,
    "ended_at": None,
    "duration_sec": None,
    "returncode": None,
    "status": "IDLE",
    "log": [],
}
SUMMARY_CACHE: dict[str, Any] = {"expires_at": 0.0, "data": None}
DISPLAY_CONTEXT_CACHE: dict[str, dict[str, Any]] = {}
# The investor-dashboard payload reads many large CSVs and can take ~40s to
# build on a slow/external volume. Cache it (single-flight build) so repeat
# loads, refreshes, and re-opens are instant; a background pre-warm at startup
# kicks the build off immediately so the window populates as soon as possible.
INVESTOR_CACHE_TTL_SECONDS = 600.0
INVESTOR_CACHE_LOCK = threading.Lock()
INVESTOR_BUILD_LOCK = threading.Lock()
INVESTOR_CACHE: dict[str, Any] = {"expires_at": 0.0, "data": None}


MODE_LABELS = {
    "full": "전체 업데이트",
    "downstream": "데이터 유지 후 전체 재계산",
    "research_expansion_update": "연구 확장 업데이트",
    "hourly": "시간봉 데이터 갱신",
    "intraday": "분봉 데이터 갱신",
    "data_pipeline": "일봉 데이터/차트",
    "news_causal_engine": "뉴스 원인 수집",
    "rule_engine": "룰 엔진",
    "backtest_engine": "백테스트",
    "risk_engine": "리스크 정책",
    "validation_engine": "검증",
    "stress_engine": "스트레스",
    "integrity_engine": "무결성",
    "prediction_engine": "예측",
    "next_day_up_model_engine": "내일 상승 예측",
    "next_close_forecast_engine": "종가 예측",
    "external_feature_engine": "외부 피처",
    "ml_overlay_backtest": "ML 오버레이 백테스트",
    "pooled_dataset_builder": "Pooled 데이터셋",
    "pooled_model_engine": "Pooled 모델",
    "data_quality_engine": "데이터 품질",
    "model_gate_engine": "모델 게이트 감사",
    "backtest_event_ledger": "백테스트 이벤트 원장",
    "backtest_feedback_feature_engine": "백테스트 피드백 피처",
    "auto_research_engine": "자동 연구 Trial",
    "model_registry_engine": "모델 레지스트리",
    "shadow_paper_engine": "섀도 페이퍼",
    "pre_paper_system_state_engine": "Paper 전 시스템 준비도",
    "order_intent_engine": "주문 의도",
    "portfolio_risk_engine": "포트폴리오 리스크",
    "paper_execution_engine": "Paper OMS",
    "position_reconciler": "Paper 정합성",
    "order_state_machine": "주문 생애주기",
    "execution_feedback_engine": "실행 피드백",
    "fill_model_calibration_engine": "체결모델 보정",
    "automation_scheduler": "Paper 자동화 계획",
    "universe_market_data_update": "유니버스 시간/분봉 갱신",
    "pooled_universe_update": "반도체 유니버스 업데이트",
    "system_state_engine": "시스템 준비도",
    "daily_trading_report": "매매 계획",
    "daily_health_report": "일일 헬스 리포트",
}

DEFAULT_UNIVERSE_CONFIG = "config/semiconductor_universe_top10.csv"
DEFAULT_RESEARCH_UNIVERSE_CONFIG = "config/semiconductor_universe_expanded.csv"
DEFAULT_EXTERNAL_FEATURES = "tsm_price_rule_output/tsm_external_daily_features.csv"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def read_csv_file(path: Path, **kwargs: Any) -> pd.DataFrame:
    kwargs.setdefault("low_memory", False)
    return strip_bom_columns(pd.read_csv(path, **kwargs))


def _is_blank_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"", "na", "n/a", "none", "null"}
    return False


def _as_float(value: Any) -> float | None:
    if _is_blank_value(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, bool):
        return None
    try:
        return float(str(value).replace("%", "").replace(",", "").strip())
    except ValueError:
        return None


def _clamp_float(value: float | None, low: float, high: float) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return max(low, min(high, value))


def _set_alias(data: dict[str, Any], target: str, source: str) -> None:
    if not _is_blank_value(data.get(target)) and target in data:
        return
    if _is_blank_value(data.get(source)):
        return
    data[target] = data[source]


def _normalize_risk_snapshot_fields(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    final_weight = _as_float(row.get("final_recommended_max_weight"))
    final_weight_pct = _as_float(row.get("final_recommended_max_weight_pct"))

    if final_weight is None and final_weight_pct is not None:
        # New systems store max weight as percentage, old systems store ratio.
        row["final_recommended_max_weight"] = final_weight_pct / 100 if abs(final_weight_pct) > 1 else final_weight_pct
    elif final_weight is not None and _is_blank_value(row.get("final_recommended_max_weight_pct")):
        # Keep percent-form display field even for ratio-based outputs.
        row["final_recommended_max_weight_pct"] = final_weight * 100 if abs(final_weight) <= 1 else final_weight

    _set_alias(row, "latest_entry_gate_status", "prediction_entry_gate_status")
    _set_alias(row, "prediction_entry_gate_status", "latest_entry_gate_status")
    return row


def _normalize_system_snapshot_fields(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    _set_alias(row, "latest_entry_gate_status", "prediction_entry_gate_status")
    _set_alias(row, "prediction_entry_gate_status", "latest_entry_gate_status")

    final_weight = _as_float(row.get("final_recommended_max_weight"))
    final_weight_pct = _as_float(row.get("final_recommended_max_weight_pct"))

    if final_weight is None and final_weight_pct is not None:
        row["final_recommended_max_weight"] = final_weight_pct / 100 if abs(final_weight_pct) > 1 else final_weight_pct
    elif final_weight is not None and _is_blank_value(row.get("final_recommended_max_weight_pct")) and abs(final_weight) <= 1:
        row["final_recommended_max_weight_pct"] = final_weight * 100

    return row


def _normalize_prediction_snapshot_fields(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    _set_alias(row, "latest_entry_gate_status", "prediction_entry_gate_status")
    _set_alias(row, "prediction_entry_gate_status", "latest_entry_gate_status")
    return row


def _normalize_next_day_prediction_snapshot_fields(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    _set_alias(row, "asof_date", "next_day_prediction_asof_date")
    _set_alias(row, "prediction_scope", "next_day_prediction_scope")
    _set_alias(row, "label_definition", "next_day_label_definition")
    _set_alias(row, "best_model", "next_day_best_model_1d")
    _set_alias(row, "p_up", "next_day_p_up_1d")
    _set_alias(row, "p_down", "next_day_p_down_1d")
    _set_alias(row, "threshold", "next_day_threshold_1d")
    _set_alias(row, "confidence_band", "next_day_confidence_band_1d")
    _set_alias(row, "p_up_lower_80", "next_day_p_up_lower_80_1d")
    _set_alias(row, "p_up_upper_80", "next_day_p_up_upper_80_1d")
    _set_alias(row, "prediction_signal_status", "next_day_prediction_signal_status")
    _set_alias(row, "model_quality_status", "next_day_model_quality_status")
    _set_alias(row, "model_quality_block_reasons", "next_day_model_quality_block_reasons_1d")
    _set_alias(row, "prediction_quality_pass", "next_day_prediction_quality_pass_1d")
    _set_alias(row, "performance_quality_pass", "next_day_performance_quality_pass_1d")
    _set_alias(row, "performance_quality_block_reasons", "next_day_performance_quality_block_reasons_1d")
    _set_alias(row, "oos_event_count", "next_day_oos_event_count_1d")
    _set_alias(row, "selected_oos_event_count", "next_day_selected_oos_event_count_1d")
    _set_alias(row, "brier_improvement_pct", "next_day_brier_improvement_pct_1d")
    _set_alias(row, "ece", "next_day_ece_1d")
    _set_alias(row, "pr_auc", "next_day_pr_auc_1d")
    _set_alias(row, "expectancy_improvement_pct", "next_day_expectancy_improvement_pct_1d")
    return row


def _normalize_risk_weight_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "final_recommended_max_weight" not in df.columns and "final_recommended_max_weight_pct" not in df.columns:
        return df

    out = df.copy()
    if "final_recommended_max_weight" in out.columns:
        weight = pd.to_numeric(out["final_recommended_max_weight"], errors="coerce")
        # Older/newer runs may carry percent in this column; normalize to ratio.
        weight = weight.mask(weight > 1, weight / 100)
        out["final_recommended_max_weight"] = weight
    else:
        out["final_recommended_max_weight"] = pd.to_numeric(out["final_recommended_max_weight_pct"], errors="coerce").mask(
            lambda s: s > 1, lambda s: s / 100
        )
    out["final_recommended_max_weight"] = out["final_recommended_max_weight"].clip(lower=0)

    if "final_recommended_max_weight_pct" not in out.columns:
        out["final_recommended_max_weight_pct"] = out["final_recommended_max_weight"] * 100
    return out


def safe_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return read_csv_file(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def safe_read_csv_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    wanted = set(columns)
    return safe_read_csv(path, usecols=lambda col: col in wanted)


def safe_read_csv_tail(path: Path, rows: int, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        header = path.open("r", encoding="utf-8-sig", errors="replace").readline()
        raw = subprocess.check_output(
            ["tail", "-n", str(max(1, rows)), str(path)],
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if not raw.strip():
            return pd.DataFrame(columns=columns or None)
        csv_text = raw if raw.startswith(header) else header + raw
        kwargs: dict[str, Any] = {"low_memory": False}
        if columns is not None:
            wanted = set(columns)
            kwargs["usecols"] = lambda col: col in wanted
        return strip_bom_columns(pd.read_csv(io.StringIO(csv_text), **kwargs))
    except Exception:
        return pd.DataFrame()


def ensure_v2_signal_columns(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty or "decision_tier" in signals.columns:
        return signals
    try:
        from tsm_price_rule_engine import add_semiconductor_momentum_v2_layers

        return add_semiconductor_momentum_v2_layers(signals)
    except Exception:
        return signals


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if hasattr(value, "item"):
        try:
            return clean_json(value.item())
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def records_from_df(df: pd.DataFrame, columns: list[str] | None = None, tail: int | None = None) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = df.copy()
    if columns:
        existing = [c for c in columns if c in out.columns]
        out = out[existing]
    if tail:
        out = out.tail(tail)
    return clean_json(out.to_dict(orient="records"))


def field_value_map(path: Path) -> dict[str, Any]:
    df = safe_read_csv(path)
    if df.empty:
        return {}
    if {"field", "value"}.issubset(df.columns):
        return clean_json(dict(zip(df["field"], df["value"])))
    if {"metric", "value"}.issubset(df.columns):
        return clean_json(dict(zip(df["metric"], df["value"])))
    return {}


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "pass"])


def quality_summary(path: Path, critical_only: bool = False) -> dict[str, Any]:
    df = safe_read_csv(path)
    if df.empty or "passed" not in df.columns:
        return {
            "file": relative_name(path),
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate_pct": None,
            "failed_rows": [],
        }
    target = df.copy()
    if critical_only and "severity" in target.columns:
        critical = target[target["severity"].astype(str).str.upper() == "CRITICAL"]
        if not critical.empty:
            target = critical
    passed_mask = bool_series(target["passed"])
    total = int(len(target))
    passed = int(passed_mask.sum())
    failed_rows = target.loc[~passed_mask].head(20)
    return {
        "file": relative_name(path),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate_pct": round((passed / total) * 100, 2) if total else None,
        "failed_rows": records_from_df(failed_rows),
    }


def _quality_failure_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "passed" not in df.columns:
        return {"critical_failed": 0, "warning_failed": 0, "failed": 0}
    passed = bool_series(df["passed"])
    severity = (
        df["severity"].astype(str).str.upper()
        if "severity" in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    failed = ~passed
    return {
        "critical_failed": int((failed & severity.eq("CRITICAL")).sum()),
        "warning_failed": int((failed & severity.isin(["WARN", "WARNING"])).sum()),
        "failed": int(failed.sum()),
    }


def _file_timestamp(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _symbol_coverage(df: pd.DataFrame, symbols: list[str]) -> dict[str, Any]:
    expected = {symbol.upper() for symbol in symbols}
    if df.empty or "symbol" not in df.columns:
        return {"present": 0, "expected": len(expected), "missing": sorted(expected)}
    present = {str(symbol).strip().upper() for symbol in df["symbol"].dropna().tolist()}
    return {
        "present": len(expected & present),
        "expected": len(expected),
        "missing": sorted(expected - present),
    }


def _status_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    return clean_json(df[column].dropna().astype(str).value_counts().to_dict())


def _latest_dates_by_symbol(symbols: list[str]) -> dict[str, str | None]:
    dates: dict[str, str | None] = {}
    for symbol in symbols:
        df = safe_read_csv(OUTPUT_DIR / "universe" / symbol / "tsm_daily_10y_enriched.csv")
        if df.empty or "date" not in df.columns:
            dates[symbol] = None
            continue
        parsed = pd.to_datetime(df["date"], errors="coerce").dropna()
        dates[symbol] = parsed.max().date().isoformat() if not parsed.empty else None
    return dates


def _latest_log_path() -> Path | None:
    log_dir = RULE_DIR / "logs"
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("daily_update_*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def build_daily_update_system_summary(
    decision_config: pd.DataFrame | None = None,
    market_data: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    portfolio_decisions: pd.DataFrame | None = None,
    paper_orders: pd.DataFrame | None = None,
    paper_fills: pd.DataFrame | None = None,
    paper_positions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    overseas_symbols = ["NVDA", "TSM", "AVGO", "AMD", "INTC", "MU", "TXN", "LRCX", "AMAT", "QCOM"]
    kr_symbols = ["005930.KS", "000660.KS"]
    top12_symbols = overseas_symbols + kr_symbols

    config = decision_config if decision_config is not None else safe_read_csv(RULE_DIR / "tsm_decision_universe_config.csv")
    if config.empty:
        config = safe_read_csv(ROOT / "config" / "semiconductor_universe_top10.csv")
    enabled_symbols: list[str] = []
    if not config.empty and "symbol" in config.columns:
        enabled = config
        if "enabled" in enabled.columns:
            enabled = enabled[bool_series(enabled["enabled"])]
        enabled_symbols = [str(symbol).strip().upper() for symbol in enabled["symbol"].dropna().tolist()]
    if not enabled_symbols:
        enabled_symbols = top12_symbols

    market = market_data if market_data is not None else safe_read_csv(OUTPUT_DIR / "tsm_universe_market_data_latest.csv")
    latest_predictions = predictions if predictions is not None else safe_read_csv(RULE_DIR / "tsm_universe_latest_predictions.csv")
    risk_decisions = portfolio_decisions if portfolio_decisions is not None else safe_read_csv(RULE_DIR / "tsm_portfolio_risk_order_decisions.csv")
    orders = paper_orders if paper_orders is not None else safe_read_csv(RULE_DIR / "tsm_paper_orders.csv")
    fills = paper_fills if paper_fills is not None else safe_read_csv(RULE_DIR / "tsm_paper_fills.csv")
    positions = paper_positions if paper_positions is not None else safe_read_csv(RULE_DIR / "tsm_paper_positions.csv")
    pooled_feature = safe_read_csv(RULE_DIR / "tsm_prediction_pooled_feature_matrix.csv")
    pooled_label = safe_read_csv(RULE_DIR / "tsm_prediction_pooled_label_dataset.csv")
    threshold_grid = safe_read_csv(RULE_DIR / "tsm_rule_threshold_sensitivity.csv")
    threshold_summary = safe_read_csv(RULE_DIR / "tsm_rule_threshold_sensitivity_summary.csv")
    full_audit = safe_read_csv(RULE_DIR / "tsm_full_daily_update_audit.csv")

    quality = {}
    for key, file_name in {
        "pooled_dataset": "tsm_prediction_pooled_quality_checks.csv",
        "pooled_model": "tsm_pooled_model_quality_checks.csv",
        "threshold_sensitivity": "tsm_rule_threshold_sensitivity_quality_checks.csv",
        "automation": "tsm_automation_quality_checks.csv",
        "paper_oms": "tsm_paper_oms_quality_checks.csv",
        "portfolio_risk": "tsm_portfolio_risk_checks.csv",
    }.items():
        df = safe_read_csv(RULE_DIR / file_name)
        quality[key] = {"rows": int(len(df)), **_quality_failure_counts(df)}

    audit_failures: list[dict[str, Any]] = []
    audit_status = "UNKNOWN"
    if not full_audit.empty and "status" in full_audit.columns:
        failed = full_audit[full_audit["status"].astype(str).str.upper().eq("FAIL")].copy()
        audit_failures = records_from_df(failed, ["component", "status", "details"], tail=8)
        audit_status = "FAIL" if not failed.empty else "PASS"

    fallback_rows = 0
    source_values: list[str] = []
    live_values: list[str] = []
    if not latest_predictions.empty and "symbol" in latest_predictions.columns:
        decision_predictions = latest_predictions[
            latest_predictions["symbol"].astype(str).str.upper().isin({s.upper() for s in enabled_symbols})
        ].copy()
        source_cols = [col for col in decision_predictions.columns if "source" in col.lower()]
        if source_cols:
            fallback_mask = pd.Series(False, index=decision_predictions.index)
            for col in source_cols:
                fallback_mask |= decision_predictions[col].astype(str).str.contains("fallback", case=False, na=False)
            fallback_rows = int(fallback_mask.sum())
        if "prediction_source" in latest_predictions.columns:
            source_values = sorted(latest_predictions["prediction_source"].dropna().astype(str).unique().tolist())
        if "live_trading_status" in latest_predictions.columns:
            live_values = sorted(latest_predictions["live_trading_status"].dropna().astype(str).unique().tolist())

    if not market.empty and "symbol" in market.columns:
        top12_market = market[market["symbol"].astype(str).str.upper().isin({s.upper() for s in top12_symbols})].copy()
    else:
        top12_market = pd.DataFrame()

    if not risk_decisions.empty and "symbol" in risk_decisions.columns:
        risk_subset = risk_decisions[
            risk_decisions["symbol"].astype(str).str.upper().isin({s.upper() for s in overseas_symbols})
        ].copy()
    else:
        risk_subset = pd.DataFrame()

    latest_log = _latest_log_path()
    return clean_json(
        {
            "run_date": date.today().isoformat(),
            "latest_log": str(latest_log.relative_to(ROOT)) if latest_log else None,
            "latest_log_updated_at_utc": _file_timestamp(latest_log),
            "full_audit_status": audit_status,
            "full_audit_required_failures": audit_failures,
            "audit_required_failures": audit_failures,
            "overseas_symbols": overseas_symbols,
            "kr_symbols": kr_symbols,
            "enabled_symbol_count": len(enabled_symbols),
            "top12_expected": len(top12_symbols),
            "daily_latest_by_symbol": _latest_dates_by_symbol(top12_symbols),
            "market_data": {
                "top12_rows": int(len(top12_market)),
                "status_counts": _status_counts(top12_market, "status"),
                "bar_counts": _status_counts(top12_market, "bar_type"),
                "latest_by_bar": (
                    clean_json(top12_market.groupby("bar_type")["end_timestamp"].max().to_dict())
                    if not top12_market.empty and {"bar_type", "end_timestamp"}.issubset(top12_market.columns)
                    else {}
                ),
            },
            "coverage": {
                "pooled_feature_overseas": _symbol_coverage(pooled_feature, overseas_symbols),
                "pooled_label_overseas": _symbol_coverage(pooled_label, overseas_symbols),
                "latest_prediction_overseas": _symbol_coverage(latest_predictions, overseas_symbols),
                "latest_prediction_top12": _symbol_coverage(latest_predictions, top12_symbols),
            },
            "latest_predictions": {
                "rows": int(len(latest_predictions)),
                "foreign_rows": _symbol_coverage(latest_predictions, overseas_symbols).get("present", 0),
                "top12_rows": _symbol_coverage(latest_predictions, top12_symbols).get("present", 0),
                "fallback_rows": fallback_rows,
                "sources": source_values,
                "live_trading_statuses": live_values,
            },
            "quality": quality,
            "threshold_sensitivity": {
                "grid_rows": int(len(threshold_grid)),
                "summary_rows": int(len(threshold_summary)),
                "recommendation": (
                    str(threshold_summary.iloc[0].get("recommendation"))
                    if not threshold_summary.empty and "recommendation" in threshold_summary.columns
                    else None
                ),
                "reason": (
                    str(threshold_summary.iloc[0].get("reason"))
                    if not threshold_summary.empty and "reason" in threshold_summary.columns
                    else None
                ),
            },
            "portfolio_risk": {
                "rows": int(len(risk_decisions)),
                "overseas_rows": int(len(risk_subset)),
                "approved_overseas": (
                    int(risk_subset["portfolio_status"].astype(str).str.upper().eq("APPROVED").sum())
                    if not risk_subset.empty and "portfolio_status" in risk_subset.columns
                    else 0
                ),
                "rejected_overseas": (
                    int(risk_subset["portfolio_status"].astype(str).str.upper().eq("REJECTED").sum())
                    if not risk_subset.empty and "portfolio_status" in risk_subset.columns
                    else 0
                ),
                "smh_beta_limit_modes": (
                    sorted(risk_decisions["smh_beta_limit_mode"].dropna().astype(str).unique().tolist())
                    if "smh_beta_limit_mode" in risk_decisions.columns
                    else []
                ),
                "has_block_reason": "block_reason" in risk_decisions.columns,
                "has_warning_reasons": "warning_reasons" in risk_decisions.columns,
            },
            "paper_oms": {
                "order_rows": int(len(orders)),
                "order_status_counts": _status_counts(orders, "status"),
                "fill_rows": int(len(fills)),
                "position_rows": int(len(positions)),
                "position_state_counts": _status_counts(positions, "position_state"),
                "live_trading_statuses": (
                    sorted(orders["live_trading_status"].dropna().astype(str).unique().tolist())
                    if "live_trading_status" in orders.columns
                    else ["DISABLED_BY_DESIGN"]
                ),
            },
        }
    )


def mismatch_summary(path: Path) -> dict[str, Any]:
    df = safe_read_csv(path)
    if df.empty or "mismatch_count" not in df.columns:
        return {
            "file": relative_name(path),
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate_pct": None,
            "failed_rows": [],
        }
    target = df.copy()
    mismatches = pd.to_numeric(target["mismatch_count"], errors="coerce").fillna(0)
    passed_mask = mismatches.eq(0)
    total = int(len(target))
    passed = int(passed_mask.sum())
    failed_rows = target.loc[~passed_mask].head(20)
    return {
        "file": relative_name(path),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate_pct": round((passed / total) * 100, 2) if total else None,
        "failed_rows": records_from_df(failed_rows),
    }


def relative_name(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.name


def file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".md":
        return "report"
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg"}:
        return "image"
    return suffix.lstrip(".") or "file"


def line_count(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    try:
        if path.stat().st_size > MAX_DASHBOARD_LINE_COUNT_BYTES:
            return None
        with path.open("rb") as f:
            count = sum(1 for _ in f)
        return max(0, count - 1)
    except Exception:
        return None


def output_scope(path: Path, base: Path) -> tuple[str, str | None]:
    try:
        parts = path.relative_to(base).parts
    except Exception:
        return "core", None
    if len(parts) >= 3 and parts[0] == "universe":
        return "universe", parts[1]
    return "core", None


def list_output_files() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in [OUTPUT_DIR, RULE_DIR]:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if any(part.startswith(".") for part in path.relative_to(base).parts):
                continue
            stat = path.stat()
            rel = relative_name(path)
            scope, symbol = output_scope(path, base)
            rows.append(
                {
                    "path": rel,
                    "name": path.name,
                    "kind": file_kind(path),
                    "scope": scope,
                    "symbol": symbol,
                    "source_dir": base.name,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    "rows": None if scope == "universe" else line_count(path),
                    "url": f"/api/file?file={rel}",
                }
            )
    return rows


def latest_row_map(path: Path) -> dict[str, Any]:
    df = safe_read_csv(path)
    if df.empty:
        return {}
    row = df.tail(1).to_dict(orient="records")[0]
    return clean_json(row)


def latest_record_map(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    return clean_json(df.tail(1).to_dict(orient="records")[0])


def _context_value_from_row(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    return None if _is_blank_value(value) else value


def _latest_fx_context() -> dict[str, Any]:
    latest_fx = latest_row_map(OUTPUT_DIR / "tsm_fx_latest.csv")
    fx_rate = _as_float(latest_fx.get("fx_rate_to_usd"))
    usdkrw = _as_float(latest_fx.get("usdkrw") or latest_fx.get("fx_close"))
    if fx_rate is None and usdkrw and usdkrw > 0:
        fx_rate = 1.0 / usdkrw
    if usdkrw is None and fx_rate and fx_rate > 0:
        usdkrw = 1.0 / fx_rate
    return {
        "date": latest_fx.get("date"),
        "fx_pair": latest_fx.get("fx_pair") or "KRW=X",
        "fx_close": latest_fx.get("fx_close"),
        "base_currency": latest_fx.get("base_currency") or "USD",
        "quote_currency": latest_fx.get("quote_currency") or "KRW",
        "fx_rate_to_usd": fx_rate,
        "usdkrw": usdkrw,
        "data_source": latest_fx.get("data_source"),
        "source_url": latest_fx.get("source_url"),
        "generated_at_utc": latest_fx.get("generated_at_utc"),
    }


def _parse_koreaexim_number(value: Any) -> float | None:
    text = str(value or "").replace(",", "").strip()
    return _as_float(text)


def _koreaexim_authkey() -> str:
    return str(os.getenv("KOREAEXIM_AUTHKEY") or os.getenv("KOREAEXIM_API_KEY") or "").strip()


def _koreaexim_query_dates(search_date: str | None = None, fallback_days: int = 7) -> list[str]:
    if search_date:
        query_date = search_date.replace("-", "")
        if len(query_date) != 8 or not query_date.isdigit():
            raise ValueError("search_date는 YYYYMMDD 또는 YYYY-MM-DD 형식이어야 합니다.")
        return [query_date]
    today = date.today()
    return [(today - timedelta(days=offset)).strftime("%Y%m%d") for offset in range(max(1, int(fallback_days)))]


def _fetch_koreaexim_payload(authkey: str, query_date: str, timeout: int) -> Any:
    url = f"{KOREAEXIM_EXCHANGE_URL}?{urlencode({'authkey': authkey, 'searchdate': query_date, 'data': 'AP01'})}"
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8-sig"))


def fetch_koreaexim_usdkrw(search_date: str | None = None, timeout: int = 20) -> dict[str, Any]:
    authkey = _koreaexim_authkey()
    if not authkey:
        raise RuntimeError("KOREAEXIM_AUTHKEY 또는 KOREAEXIM_API_KEY 환경변수가 필요합니다.")
    errors: list[str] = []
    usd_row: dict[str, Any] | None = None
    used_date = ""
    for query_date in _koreaexim_query_dates(search_date):
        payload = _fetch_koreaexim_payload(authkey, query_date, timeout)
        if not isinstance(payload, list):
            errors.append(f"{query_date}: 응답 형식 오류")
            continue
        usd_row = next((row for row in payload if str(row.get("cur_unit", "")).upper() == "USD"), None)
        if usd_row:
            used_date = query_date
            break
        errors.append(f"{query_date}: USD 행 없음")
    if not usd_row:
        raise RuntimeError(f"수출입은행 환율 API에서 최근 환율을 찾지 못했습니다. {'; '.join(errors)}")
    usdkrw = _parse_koreaexim_number(usd_row.get("deal_bas_r"))
    if usdkrw is None or usdkrw <= 0:
        raise RuntimeError("수출입은행 USD 매매기준율을 파싱하지 못했습니다.")
    return {
        "date": f"{used_date[:4]}-{used_date[4:6]}-{used_date[6:]}",
        "fx_pair": "KRW=X",
        "fx_close": usdkrw,
        "base_currency": "USD",
        "quote_currency": "KRW",
        "fx_rate_to_usd": 1.0 / usdkrw,
        "usdkrw": usdkrw,
        "data_source": "koreaexim_current_exchange_api",
        "source_url": KOREAEXIM_EXCHANGE_URL,
        "generated_at_utc": now_iso(),
        "cur_unit": usd_row.get("cur_unit"),
        "cur_nm": usd_row.get("cur_nm"),
        "ttb": usd_row.get("ttb"),
        "tts": usd_row.get("tts"),
        "deal_bas_r": usd_row.get("deal_bas_r"),
    }


def refresh_koreaexim_fx(search_date: str | None = None) -> dict[str, Any]:
    fx = fetch_koreaexim_usdkrw(search_date=search_date)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([fx]).to_csv(OUTPUT_DIR / "tsm_fx_latest.csv", index=False, encoding="utf-8-sig")
    return clean_json({"fx": _latest_fx_context(), "raw": fx})


def _display_fx_rate_for_investor(currency: str, signal: dict[str, Any], market: dict[str, Any], member: dict[str, Any]) -> Any:
    if str(currency).upper() == "KRW":
        latest_fx = _latest_fx_context()
        latest_rate = _as_float(latest_fx.get("fx_rate_to_usd"))
        if latest_rate and latest_rate > 0:
            return latest_rate
    return _first_text(signal.get("fx_rate_to_usd"), market.get("fx_rate_to_usd"), member.get("fx_rate_to_usd"), default="")


def display_context_for_symbol(symbol: Any, latest_price: dict[str, Any] | None = None) -> dict[str, Any]:
    sym = str(symbol or "").strip().upper()
    latest_price = dict(latest_price or {})
    cache_allowed = not latest_price
    if cache_allowed and sym in DISPLAY_CONTEXT_CACHE:
        return dict(DISPLAY_CONTEXT_CACHE[sym])
    config = safe_read_csv(RULE_DIR / "tsm_decision_universe_config.csv")
    if config.empty or "symbol" not in config.columns:
        config = safe_read_csv(ROOT / DEFAULT_UNIVERSE_CONFIG)
    config_row: dict[str, Any] = {}
    if not config.empty and "symbol" in config.columns:
        rows = config[config["symbol"].astype(str).str.upper() == sym]
        if not rows.empty:
            config_row = clean_json(rows.iloc[-1].to_dict())

    if not latest_price:
        latest_price = latest_row_map(OUTPUT_DIR / "universe" / sym / "tsm_daily_10y_enriched.csv")
    if not latest_price and sym == "TSM":
        latest_price = latest_row_map(OUTPUT_DIR / "tsm_daily_10y_enriched.csv")

    listing = (
        _context_value_from_row(latest_price, "listing_currency")
        or _context_value_from_row(config_row, "listing_currency")
        or ("KRW" if sym.endswith((".KS", ".KQ")) else "USD")
    )
    display = (
        _context_value_from_row(latest_price, "display_currency")
        or _context_value_from_row(config_row, "display_currency")
        or listing
    )
    engine = (
        _context_value_from_row(latest_price, "engine_currency")
        or _context_value_from_row(config_row, "engine_currency")
        or "USD"
    )
    fx_pair = (
        _context_value_from_row(latest_price, "fx_pair")
        or _context_value_from_row(config_row, "fx_pair")
        or ("KRW=X" if str(display).upper() == "KRW" else "")
    )
    fx_rate = _as_float(latest_price.get("fx_rate_to_usd"))
    usdkrw = _as_float(latest_price.get("usdkrw"))
    if str(display).upper() == "KRW" and (fx_rate is None or usdkrw is None):
        latest_fx = _latest_fx_context()
        fx_rate = fx_rate if fx_rate is not None else _as_float(latest_fx.get("fx_rate_to_usd"))
        usdkrw = usdkrw if usdkrw is not None else _as_float(latest_fx.get("usdkrw"))

    context = clean_json(
        {
            "symbol": sym,
            "listing_currency": str(listing).upper(),
            "display_currency": str(display).upper(),
            "engine_currency": str(engine).upper(),
            "fx_pair": fx_pair,
            "fx_rate_to_usd": fx_rate,
            "usdkrw": usdkrw,
        }
    )
    if cache_allowed:
        DISPLAY_CONTEXT_CACHE[sym] = dict(context)
    return context


def with_display_context(row: dict[str, Any], symbol: Any, latest_price: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(row or {})
    context = display_context_for_symbol(symbol, latest_price)
    for key, value in context.items():
        if _is_blank_value(out.get(key)):
            out[key] = value
    return out


def enrich_display_context(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "symbol" not in df.columns:
        return df
    out = df.copy()
    for col in ["listing_currency", "display_currency", "engine_currency", "fx_pair", "fx_rate_to_usd", "usdkrw"]:
        if col not in out.columns:
            out[col] = None
    symbols = sorted({str(sym).strip().upper() for sym in out["symbol"].dropna().tolist() if str(sym).strip()})
    for sym in symbols:
        context = display_context_for_symbol(sym)
        mask = out["symbol"].astype(str).str.upper() == sym
        for col in ["listing_currency", "display_currency", "engine_currency", "fx_pair", "fx_rate_to_usd", "usdkrw"]:
            blank_idx = out.loc[mask, col].map(_is_blank_value)
            if blank_idx.any():
                target_index = blank_idx[blank_idx].index
                out.loc[target_index, col] = context.get(col)
    return out


def universe_validation_snapshot(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "symbol" not in df.columns:
        return {}
    loaded = bool_series(df["loaded"]) if "loaded" in df.columns else pd.Series(False, index=df.index)
    strict = bool_series(df["strict_eligible"]) if "strict_eligible" in df.columns else pd.Series(False, index=df.index)
    target_loaded = int(loaded.sum())
    target_strict = int(strict.sum())
    short_history = (
        int(df.get("eligibility_status", pd.Series("", index=df.index)).astype(str).eq("SHORT_HISTORY_RESEARCH_ONLY").sum())
        if "eligibility_status" in df.columns
        else 0
    )
    missing = (
        int(df.get("eligibility_status", pd.Series("", index=df.index)).astype(str).eq("MISSING_ENRICHED").sum())
        if "eligibility_status" in df.columns
        else 0
    )
    return clean_json(
        {
            "candidate_symbols": int(len(df)),
            "loaded_symbols": target_loaded,
            "strict_eligible_symbols": target_strict,
            "loaded_target_pass": target_loaded >= 45,
            "strict_eligible_target_pass": target_strict >= 35,
            "short_history_research_only": short_history,
            "missing_enriched_symbols": missing,
        }
    )


def sample_audit_snapshot(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    numeric_cols = [
        "loaded",
        "strict_eligible",
        "row_count",
        "trade_ready_20d_labeled",
        "model_training_20d_labeled",
        "test_holdout_trade_ready_20d",
        "tsm_like_effective_n_contribution",
    ]
    totals: dict[str, Any] = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if col in {"loaded", "strict_eligible"}:
            totals[col] = int(bool_series(df[col]).sum())
        else:
            totals[col] = float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
    totals["trade_ready_target_pass"] = totals.get("trade_ready_20d_labeled", 0) >= 10000
    totals["model_training_target_pass"] = totals.get("model_training_20d_labeled", 0) >= 100000
    return clean_json(totals)


def research_expansion_snapshot(
    event_ledger: pd.DataFrame,
    feedback_features: pd.DataFrame,
    auto_trials: pd.DataFrame,
    best_candidates: pd.DataFrame,
    cpcv_strategy: pd.DataFrame,
    cpcv_model: pd.DataFrame,
    model_gate: dict[str, Any],
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "event_ledger_rows": int(len(event_ledger)),
        "feedback_feature_rows": int(len(feedback_features)),
        "auto_research_trial_count": int(len(auto_trials)),
        "research_validation_pass": model_gate.get("research_validation_pass"),
        "pooled_system_quality_pass": model_gate.get("pooled_system_quality_pass"),
    }
    if not event_ledger.empty:
        snapshot["event_ledger_symbol_count"] = int(event_ledger["symbol"].nunique()) if "symbol" in event_ledger.columns else 0
        snapshot["event_ledger_strategy_count"] = int(event_ledger["strategy_id"].nunique()) if "strategy_id" in event_ledger.columns else 0
        snapshot["event_ledger_variant_count"] = int(event_ledger["variant_id"].nunique()) if "variant_id" in event_ledger.columns else 0
        if "candidate_tier" in event_ledger.columns:
            tiers = event_ledger["candidate_tier"].astype(str)
            snapshot["event_ledger_trade_ready_rows"] = int(tiers.eq("decision_trade_ready").sum())
            snapshot["event_ledger_near_miss_rows"] = int(tiers.eq("near_miss_no_trade").sum())
            snapshot["event_ledger_research_candidate_rows"] = int(tiers.ne("none").sum())
    if not feedback_features.empty:
        feedback_cols = [c for c in feedback_features.columns if str(c).startswith("fb_")]
        snapshot["feedback_feature_count"] = int(len(feedback_cols))
        snapshot["feedback_symbol_count"] = int(feedback_features["symbol"].nunique()) if "symbol" in feedback_features.columns else 0
    if not best_candidates.empty and "objective_score" in best_candidates.columns:
        best = best_candidates.sort_values("objective_score", ascending=False).head(1).to_dict(orient="records")
        if best:
            snapshot["best_research_model"] = best[0].get("model_name")
            snapshot["best_research_family"] = best[0].get("model_family")
            snapshot["best_research_objective_score"] = best[0].get("objective_score")
    if not cpcv_strategy.empty and "test_uplift_pct_median" in cpcv_strategy.columns:
        row = cpcv_strategy.sort_values("test_uplift_pct_median", ascending=False).head(1).to_dict(orient="records")
        if row:
            snapshot["best_cpcv_strategy"] = row[0].get("strategy_id")
            snapshot["best_cpcv_strategy_median_uplift_pct"] = row[0].get("test_uplift_pct_median")
            snapshot["best_cpcv_strategy_q25_uplift_pct"] = row[0].get("test_uplift_pct_q25")
    if not cpcv_model.empty and "selected_minus_all_pct_median" in cpcv_model.columns:
        row = cpcv_model.sort_values("selected_minus_all_pct_median", ascending=False).head(1).to_dict(orient="records")
        if row:
            snapshot["best_cpcv_model"] = row[0].get("model_name")
            snapshot["best_cpcv_model_scope"] = row[0].get("candidate_scope")
            snapshot["best_cpcv_model_median_uplift_pct"] = row[0].get("selected_minus_all_pct_median")
            snapshot["best_cpcv_model_q25_uplift_pct"] = row[0].get("selected_minus_all_pct_q25")
    return clean_json(snapshot)


def build_summary() -> dict[str, Any]:
    DISPLAY_CONTEXT_CACHE.clear()
    enriched = safe_read_csv(OUTPUT_DIR / "tsm_daily_10y_enriched.csv")
    signals = ensure_v2_signal_columns(safe_read_csv(RULE_DIR / "tsm_daily_algorithmic_signals.csv"))
    equity = safe_read_csv(RULE_DIR / "tsm_backtest_equity_curves.csv")
    risk = _normalize_risk_weight_columns(safe_read_csv(RULE_DIR / "tsm_risk_policy_daily.csv"))
    backtest = safe_read_csv(RULE_DIR / "tsm_backtest_strategy_summary.csv")
    trade_log = safe_read_csv(RULE_DIR / "tsm_backtest_trade_log.csv")
    yearly = safe_read_csv(RULE_DIR / "tsm_backtest_yearly_returns.csv")
    prediction_comparison = safe_read_csv(RULE_DIR / "tsm_prediction_model_comparison.csv")
    prediction_audit = safe_read_csv(RULE_DIR / "tsm_prediction_model_audit.csv")
    prediction_near_pass = safe_read_csv(RULE_DIR / "tsm_prediction_near_pass_candidates.csv")
    prediction_performance_gap = safe_read_csv(RULE_DIR / "tsm_prediction_performance_gap_summary.csv")
    prediction_feature_association = safe_read_csv(RULE_DIR / "tsm_prediction_feature_association_summary.csv")
    prediction_brier_decomposition = safe_read_csv(RULE_DIR / "tsm_prediction_brier_decomposition_summary.csv")
    calibration = safe_read_csv(RULE_DIR / "tsm_prediction_calibration_bins.csv")
    candidate_scope = safe_read_csv(RULE_DIR / "tsm_prediction_candidate_scope_stats.csv")
    label_diagnostics = safe_read_csv(RULE_DIR / "tsm_prediction_label_diagnostics.csv")
    feature_selection = safe_read_csv(RULE_DIR / "tsm_prediction_feature_selection_report.csv")
    walk_metrics = safe_read_csv(RULE_DIR / "tsm_prediction_walk_forward_metrics.csv")
    threshold_policy = safe_read_csv(RULE_DIR / "tsm_prediction_threshold_policy.csv")
    walk_forward = safe_read_csv(RULE_DIR / "tsm_validation_walk_forward_summary.csv")
    causal_walk_forward = safe_read_csv(RULE_DIR / "tsm_validation_causal_walk_forward_summary.csv")
    validation_segments = safe_read_csv(RULE_DIR / "tsm_validation_segment_summary.csv")
    sensitivity = safe_read_csv(RULE_DIR / "tsm_validation_parameter_sensitivity.csv")
    rule_threshold_sensitivity = safe_read_csv(RULE_DIR / "tsm_rule_threshold_sensitivity.csv")
    rule_threshold_sensitivity_summary = safe_read_csv(RULE_DIR / "tsm_rule_threshold_sensitivity_summary.csv")
    rule_threshold_sensitivity_quality = safe_read_csv(RULE_DIR / "tsm_rule_threshold_sensitivity_quality_checks.csv")
    stress_scenarios = safe_read_csv(RULE_DIR / "tsm_daily_stress_scenarios.csv")
    strategy_stress = safe_read_csv(RULE_DIR / "tsm_strategy_stress_summary.csv")
    readiness = safe_read_csv(RULE_DIR / "tsm_system_readiness_scorecard.csv")
    trading_plan = safe_read_csv(RULE_DIR / "tsm_daily_trading_plan.csv")
    manifest = safe_read_csv(RULE_DIR / "tsm_daily_update_manifest.csv")
    full_daily_audit = safe_read_csv(RULE_DIR / "tsm_full_daily_update_audit.csv")
    data_validation = safe_read_csv(RULE_DIR / "tsm_data_validation_raw_vs_enriched.csv")
    yearly_price = safe_read_csv(RULE_DIR / "tsm_yearly_price_volatility_stats.csv")
    monthly_price = safe_read_csv(RULE_DIR / "tsm_monthly_price_volatility_stats.csv")
    extreme_moves = safe_read_csv(RULE_DIR / "tsm_extreme_daily_moves.csv")
    event_analysis = safe_read_csv(RULE_DIR / "tsm_event_integrated_analysis.csv")
    news_daily = safe_read_csv(RULE_DIR / "tsm_news_integrated_daily.csv")
    news_matches = safe_read_csv(OUTPUT_DIR / "tsm_price_news_matches.csv")
    news_clusters = safe_read_csv(OUTPUT_DIR / "tsm_news_event_clusters.csv")
    news_cause_forward = safe_read_csv(RULE_DIR / "tsm_news_cause_forward_return_stats.csv")
    prediction_features = safe_read_csv(RULE_DIR / "tsm_prediction_feature_matrix.csv")
    regime_forward = safe_read_csv(RULE_DIR / "tsm_regime_forward_return_stats.csv")
    rule_forward = safe_read_csv(RULE_DIR / "tsm_rule_forward_return_stats.csv")
    pbo = safe_read_csv(RULE_DIR / "tsm_pbo_report.csv")
    dsr = safe_read_csv(RULE_DIR / "tsm_deflated_sharpe_report.csv")
    model_trials = safe_read_csv(RULE_DIR / "tsm_model_trials_log.csv")
    universe_market_data_latest = safe_read_csv(OUTPUT_DIR / "tsm_universe_market_data_latest.csv")
    universe_intraday_manifest = safe_read_csv(OUTPUT_DIR / "tsm_universe_intraday_update_manifest.csv")
    prediction_oos = safe_read_csv(RULE_DIR / "tsm_prediction_oos_predictions.csv")
    prediction_calibration_summary = safe_read_csv(RULE_DIR / "tsm_prediction_calibration_summary.csv")
    prediction_policy_audit = safe_read_csv(RULE_DIR / "tsm_prediction_policy_audit.csv")
    prediction_feature_contract = safe_read_csv(RULE_DIR / "tsm_prediction_feature_contract.csv")
    prediction_fold_manifest = safe_read_csv(RULE_DIR / "tsm_prediction_fold_manifest.csv")
    prediction_model_registry = safe_read_csv(RULE_DIR / "tsm_prediction_model_registry.csv")
    prediction_experiment_log = safe_read_csv(RULE_DIR / "tsm_prediction_experiment_log.csv")
    next_day_latest_snapshot = field_value_map(RULE_DIR / "tsm_next_day_up_latest_snapshot.csv")
    next_day_universe_latest_predictions = safe_read_csv(RULE_DIR / "tsm_next_day_up_universe_latest_predictions.csv")
    next_day_model_comparison = safe_read_csv(RULE_DIR / "tsm_next_day_up_model_comparison.csv")
    next_day_oos_predictions = safe_read_csv_tail(RULE_DIR / "tsm_next_day_up_oos_predictions.csv", 500)
    next_day_calibration_summary = safe_read_csv(RULE_DIR / "tsm_next_day_up_calibration_summary.csv")
    next_day_threshold_policy = safe_read_csv(RULE_DIR / "tsm_next_day_up_threshold_policy.csv")
    next_day_quality = safe_read_csv(RULE_DIR / "tsm_next_day_up_quality_checks.csv")
    next_close_latest_snapshot = field_value_map(RULE_DIR / "tsm_next_close_latest_snapshot.csv")
    next_close_universe_latest_predictions = safe_read_csv(RULE_DIR / "tsm_next_close_universe_latest_predictions.csv")
    next_close_model_comparison = safe_read_csv(RULE_DIR / "tsm_next_close_model_comparison.csv")
    next_close_oos_predictions = safe_read_csv_tail(RULE_DIR / "tsm_next_close_oos_predictions.csv", 500)
    next_close_interval_calibration = safe_read_csv(RULE_DIR / "tsm_next_close_interval_calibration.csv")
    next_close_quality = safe_read_csv(RULE_DIR / "tsm_next_close_quality_checks.csv")
    schema_quality = safe_read_csv(RULE_DIR / "tsm_schema_quality_checks.csv")
    cscv_pbo = safe_read_csv(RULE_DIR / "tsm_cscv_pbo_report.csv")
    ml_overlay_summary = safe_read_csv(RULE_DIR / "tsm_ml_overlay_summary.csv")
    ml_overlay_quality = safe_read_csv(RULE_DIR / "tsm_ml_overlay_quality_checks.csv")
    ml_overlay_equity = safe_read_csv(RULE_DIR / "tsm_ml_overlay_equity_curves.csv")
    pooled_model_comparison = safe_read_csv(RULE_DIR / "tsm_pooled_model_comparison.csv")
    pooled_model_quality = safe_read_csv(RULE_DIR / "tsm_pooled_model_quality_checks.csv")
    pooled_tsm_calibration = safe_read_csv(RULE_DIR / "tsm_pooled_tsm_calibration.csv")
    pooled_tsm_calibration_metrics = safe_read_csv(RULE_DIR / "tsm_pooled_tsm_calibration_metrics.csv")
    stop_risk_baseline_summary = safe_read_csv(RULE_DIR / "tsm_stop_risk_baseline_summary.csv")
    stop_risk_calibration_bins = safe_read_csv(RULE_DIR / "tsm_stop_risk_calibration_bins.csv")
    stop_risk_slice_diagnostics = safe_read_csv(RULE_DIR / "tsm_stop_risk_slice_diagnostics.csv")
    stop_risk_latest_distribution = safe_read_csv(RULE_DIR / "tsm_stop_risk_latest_distribution.csv")
    pooled_threshold_policy = safe_read_csv(RULE_DIR / "tsm_pooled_model_threshold_policy.csv")
    pooled_uplift_bootstrap = safe_read_csv(RULE_DIR / "tsm_pooled_uplift_bootstrap_report.csv")
    pooled_oos_predictions = safe_read_csv_tail(RULE_DIR / "tsm_pooled_model_oos_predictions.csv", 500, DASHBOARD_POOLED_OOS_COLUMNS)
    pooled_oof_predictions = safe_read_csv_tail(RULE_DIR / "tsm_pooled_model_oof_predictions.csv", 800, DASHBOARD_POOLED_OOF_COLUMNS)
    pooled_slice_diagnostics = safe_read_csv(RULE_DIR / "tsm_pooled_model_slice_diagnostics.csv")
    pooled_dataset_quality = safe_read_csv(RULE_DIR / "tsm_prediction_pooled_quality_checks.csv")
    pooled_scope_stats = safe_read_csv(RULE_DIR / "tsm_prediction_pooled_scope_stats.csv")
    pooled_schema = safe_read_csv(RULE_DIR / "tsm_prediction_pooled_schema.csv")
    pooled_input_failures = safe_read_csv(RULE_DIR / "tsm_prediction_pooled_input_failures.csv")
    pooled_universe_config = safe_read_csv(RULE_DIR / "tsm_prediction_pooled_universe_config.csv")
    decision_universe_config = safe_read_csv(RULE_DIR / "tsm_decision_universe_config.csv")
    research_pool_audit = safe_read_csv(RULE_DIR / "tsm_research_pool_audit.csv")
    top10_timeframe_coverage = safe_read_csv(RULE_DIR / "tsm_top10_timeframe_coverage.csv")
    research_pool_timeframe_coverage = safe_read_csv(RULE_DIR / "tsm_research_pool_timeframe_coverage.csv")
    intraday_daily_features = safe_read_csv_tail(RULE_DIR / "tsm_intraday_daily_features.csv", 500)
    intraday_feature_quality = safe_read_csv(RULE_DIR / "tsm_intraday_feature_quality_checks.csv")
    pooled_split_manifest = safe_read_csv(RULE_DIR / "tsm_prediction_pooled_split_manifest.csv")
    pooled_universe_manifest = safe_read_csv(RULE_DIR / "tsm_pooled_universe_update_manifest.csv")
    universe_validation = safe_read_csv(RULE_DIR / "tsm_universe_validation_report.csv")
    universe_latest_signals = ensure_v2_signal_columns(safe_read_csv(RULE_DIR / "tsm_universe_latest_signals.csv"))
    universe_latest_predictions = safe_read_csv(RULE_DIR / "tsm_universe_latest_predictions.csv")
    universe_rule_fallback_diagnostics = rule_fallback_universe_prediction_diagnostics(
        universe_latest_predictions,
        decision_universe_config,
    )
    portfolio_targets = safe_read_csv(RULE_DIR / "tsm_portfolio_targets.csv")
    portfolio_snapshot = safe_read_csv(RULE_DIR / "tsm_portfolio_snapshot.csv")
    pooled_sample_audit = safe_read_csv(RULE_DIR / "tsm_prediction_pooled_sample_audit.csv")
    tsm_like_calibration_pool = safe_read_csv(RULE_DIR / "tsm_tsm_like_calibration_pool.csv")
    tsm_like_calibration_metrics = safe_read_csv(RULE_DIR / "tsm_tsm_like_calibration_metrics.csv")
    paper_gate_snapshot_table = safe_read_csv(RULE_DIR / "tsm_paper_gate_snapshot.csv")
    pooled_learning_curve = safe_read_csv(RULE_DIR / "tsm_pooled_learning_curve_report.csv")
    prediction_benchmark = safe_read_csv(RULE_DIR / "tsm_prediction_benchmark_comparison.csv")
    event_portfolio = safe_read_csv(RULE_DIR / "tsm_event_portfolio_backtest.csv")
    external_feature_schema = safe_read_csv(RULE_DIR / "tsm_external_feature_schema.csv")
    shadow_predictions = safe_read_csv(RULE_DIR / "tsm_shadow_paper_predictions.csv")
    shadow_quality = safe_read_csv(RULE_DIR / "tsm_shadow_paper_quality_checks.csv")
    order_intents = safe_read_csv(RULE_DIR / "tsm_order_intents.csv")
    order_intent_quality = safe_read_csv(RULE_DIR / "tsm_order_intent_quality_checks.csv")
    portfolio_risk_decisions = safe_read_csv(RULE_DIR / "tsm_portfolio_risk_order_decisions.csv")
    portfolio_risk_checks = safe_read_csv(RULE_DIR / "tsm_portfolio_risk_checks.csv")
    paper_orders = safe_read_csv(RULE_DIR / "tsm_paper_orders.csv")
    paper_fills = safe_read_csv(RULE_DIR / "tsm_paper_fills.csv")
    paper_positions = safe_read_csv(RULE_DIR / "tsm_paper_positions.csv")
    paper_slippage = safe_read_csv(RULE_DIR / "tsm_paper_slippage_report.csv")
    paper_oms_quality = safe_read_csv(RULE_DIR / "tsm_paper_oms_quality_checks.csv")
    paper_reconciliation = safe_read_csv(RULE_DIR / "tsm_paper_reconciliation_report.csv")
    paper_reconciliation_quality = safe_read_csv(RULE_DIR / "tsm_paper_reconciliation_quality_checks.csv")
    order_state_events = safe_read_csv(RULE_DIR / "tsm_order_state_events.csv")
    order_lifecycle_snapshot = safe_read_csv(RULE_DIR / "tsm_order_lifecycle_snapshot.csv")
    order_state_quality = safe_read_csv(RULE_DIR / "tsm_order_state_quality_checks.csv")
    execution_feedback_events = safe_read_csv(RULE_DIR / "tsm_execution_feedback_events.csv")
    execution_feedback_features = safe_read_csv(RULE_DIR / "tsm_execution_feedback_features.csv")
    execution_feedback_labels = safe_read_csv(RULE_DIR / "tsm_execution_feedback_labels.csv")
    execution_feedback_quality = safe_read_csv(RULE_DIR / "tsm_execution_feedback_quality_checks.csv")
    fill_model_calibration = safe_read_csv(RULE_DIR / "tsm_fill_model_calibration.csv")
    fill_model_calibration_quality = safe_read_csv(RULE_DIR / "tsm_fill_model_calibration_quality_checks.csv")
    automation_plan = safe_read_csv(RULE_DIR / "tsm_automation_plan.csv")
    automation_quality = safe_read_csv(RULE_DIR / "tsm_automation_quality_checks.csv")
    data_quality_checks = safe_read_csv(RULE_DIR / "tsm_data_quality_checks.csv")
    data_quality_issues = safe_read_csv(RULE_DIR / "tsm_data_quality_issues.csv")
    model_gate_audit = safe_read_csv(RULE_DIR / "tsm_model_gate_audit.csv")
    model_gate_root_causes = safe_read_csv(RULE_DIR / "tsm_model_gate_root_causes.csv")
    model_candidate_disposition = safe_read_csv(RULE_DIR / "tsm_model_candidate_disposition_summary.csv")
    model_rank_policy_watchlist = safe_read_csv(RULE_DIR / "tsm_model_rank_policy_promotion_watchlist.csv")
    model_warning_resolution = safe_read_csv(RULE_DIR / "tsm_model_performance_warning_resolution_summary.csv")
    model_unresolved_performance_priorities = safe_read_csv(RULE_DIR / "tsm_model_unresolved_performance_priorities.csv")
    system_block_reasons = safe_read_csv(RULE_DIR / "tsm_system_block_reasons.csv")
    backtest_event_ledger = safe_read_csv_tail(RULE_DIR / "tsm_backtest_event_ledger.csv", 800, DASHBOARD_EVENT_LEDGER_COLUMNS)
    backtest_feedback_features = safe_read_csv_tail(RULE_DIR / "tsm_backtest_feedback_features.csv", 800, DASHBOARD_FEEDBACK_COLUMNS)
    backtest_feedback_quality = safe_read_csv(RULE_DIR / "tsm_backtest_feedback_quality_checks.csv")
    cpcv_path_summary = safe_read_csv(RULE_DIR / "tsm_cpcv_path_summary.csv")
    cpcv_strategy_distribution = safe_read_csv(RULE_DIR / "tsm_cpcv_strategy_distribution.csv")
    cpcv_model_distribution = safe_read_csv(RULE_DIR / "tsm_cpcv_model_distribution.csv")
    auto_research_trials = safe_read_csv(RULE_DIR / "tsm_auto_research_trial_ledger.csv")
    auto_research_best = safe_read_csv(RULE_DIR / "tsm_auto_research_best_candidates.csv")
    research_expansion_manifest = safe_read_csv(RULE_DIR / "tsm_research_expansion_manifest.csv")

    order_intents = enrich_display_context(order_intents)
    paper_orders = enrich_display_context(paper_orders)
    paper_fills = enrich_display_context(paper_fills)
    paper_positions = enrich_display_context(paper_positions)
    paper_reconciliation = enrich_display_context(paper_reconciliation)

    price_columns = [
        "date",
        "close",
        "close_native",
        "close_usd",
        "listing_currency",
        "display_currency",
        "engine_currency",
        "fx_pair",
        "fx_rate_to_usd",
        "usdkrw",
        "sma_20",
        "sma_50",
        "sma_200",
        "drawdown_from_ath",
        "vol_20d_ann",
        "vol_63d_ann",
        "atr_14_pct",
        "volume",
        "close_change_pct",
        "rsi_14",
        "relative_return_vs_spy_60d",
        "relative_return_vs_smh_60d",
        "beta_vs_spy_252d",
    ]
    signal_columns = [
        "date",
        "close",
        "close_native",
        "close_usd",
        "display_currency",
        "engine_currency",
        "fx_rate_to_usd",
        "usdkrw",
        "score_price_algo_total",
        "entry_trigger",
        "trade_action",
        "raw_entry_event",
        "decision_tier",
        "sizing_tier",
        "suggested_action",
        "suggested_weight",
        "semi_momentum_regime",
        "strict_signal_stage",
        "research_signal_stage",
        "research_signal_action",
        "research_signal_score",
        "atr_stop_2x",
        "take_profit_2R",
        "risk_pct_2atr",
        "position_weight_if_0_5pct_account_risk",
    ]
    risk_columns = [
        "date",
        "close",
        "close_native",
        "close_usd",
        "display_currency",
        "engine_currency",
        "fx_rate_to_usd",
        "usdkrw",
        "final_recommended_max_weight",
        "final_recommended_max_weight_pct",
        "risk_state",
        "limiting_reason",
        "vol_limit_weight",
        "trend_limit_weight",
        "drawdown_limit_weight",
        "score_limit_weight",
        "account_risk_limit_weight",
        "secondary_account_risk_limit_weight",
    ]
    equity_columns = [
        "date",
        "strategy_id",
        "strategy_name",
        "equity",
        "daily_return_pct",
        "drawdown_pct",
        "position_weight_pct",
        "position_state",
    ]
    trade_columns = [
        "strategy_id",
        "strategy_group",
        "trade_event",
        "entry_date",
        "entry_price",
        "target_weight_pct",
        "exit_date",
        "exit_price",
        "exit_reason",
        "holding_trading_days",
        "net_return_pct",
        "portfolio_return_pct",
        "r_multiple",
    ]
    prediction_oos_columns = [
        "horizon_days",
        "candidate_scope",
        "fold_id",
        "model_name",
        "date",
        "entry_trigger",
        "trade_action",
        "entry_gate_status",
        "prediction_universe",
        "p_success",
        "p_stop_survival",
        "threshold",
        "selected_by_threshold",
        "label_success",
        "label_net_return_pct",
        "label_expected_r",
        "label_exit_reason",
        "calibration_method",
        "threshold_source",
    ]
    registry_columns = [
        "candidate_scope",
        "champion_scope",
        "horizon_days",
        "model_name",
        "model_family",
        "model_policy",
        "promotion_status",
        "prediction_quality_pass",
        "model_quality_pass",
        "latest_signal_pass",
        "pooled_decision_support_allowed",
        "oos_event_count",
        "selected_oos_event_count",
        "selected_fraction",
        "brier_improvement_pct",
        "ece",
        "pr_auc",
        "average_precision",
        "selected_signal_expectancy_pct",
        "expectancy_improvement_pct",
        "rank_score",
        "feature_count",
        "dependency_versions",
        "quality_block_reasons",
        "selection_rate_pct",
        "overlay_mean_return_improvement_pct",
        "top_selected_features",
    ]
    experiment_columns = [
        "experiment_id",
        "candidate_scope",
        "horizon_days",
        "model_name",
        "status",
        "block_reasons",
        "oos_event_count",
        "brier_improvement_pct",
        "ece",
        "pr_auc",
        "expectancy_improvement_pct",
        "overlay_mean_return_improvement_pct",
    ]
    overlay_summary_columns = [
        "candidate_scope",
        "horizon_days",
        "model_name",
        "policy",
        "selection_rate_pct",
        "event_count",
        "success_rate_pct",
        "mean_net_return_pct",
        "profit_factor",
        "stop_rate_pct",
        "cumulative_weighted_return_pct",
        "max_event_curve_drawdown_pct",
    ]
    overlay_equity_columns = [
        "date",
        "candidate_scope",
        "horizon_days",
        "model_name",
        "policy",
        "equity",
        "drawdown_pct",
    ]
    pooled_comparison_columns = [
        "model_name",
        "model_family",
        "is_champion",
        "validation_design",
        "split",
        "evaluation_scope",
        "event_count",
        "symbol_count",
        "success_rate",
        "mean_return_pct",
        "brier_improvement_pct",
        "average_precision",
        "base_rate_average_precision",
        "ece",
        "threshold",
        "selection_score_col",
        "probability_col",
        "selected_event_count",
        "selected_fraction",
        "selected_success_rate",
        "selected_mean_return_pct",
        "selected_minus_all_pct",
        "selected_minus_all_ci_lower_pct",
        "selected_minus_all_ci_lower_pct_paired",
        "selected_minus_score_baseline_ci_lower_pct",
        "selected_minus_score_baseline_ci_lower_pct_paired",
        "uplift_bootstrap_p_value",
        "uplift_bootstrap_p_value_paired",
        "bootstrap_method",
        "bootstrap_block_col",
        "threshold_decision_eligible",
        "selected_expectancy_ci_lower_pct",
        "selected_stop_rate",
        "utility_weight_label",
        "calibration_method",
    ]
    pooled_uplift_columns = [
        "model_name",
        "model_family",
        "validation_design",
        "split",
        "evaluation_scope",
        "event_count",
        "selected_event_count",
        "selected_fraction",
        "selected_mean_return_pct",
        "mean_return_pct",
        "selected_minus_all_pct",
        "selected_minus_all_ci_lower_pct",
        "selected_minus_all_ci_lower_pct_independent",
        "selected_minus_all_ci_lower_pct_paired",
        "score_baseline_mean_return_pct",
        "selected_minus_score_baseline_pct",
        "selected_minus_score_baseline_ci_lower_pct",
        "selected_minus_score_baseline_ci_lower_pct_independent",
        "selected_minus_score_baseline_ci_lower_pct_paired",
        "uplift_bootstrap_p_value",
        "uplift_bootstrap_p_value_independent",
        "uplift_bootstrap_p_value_paired",
        "bootstrap_method",
        "bootstrap_block_col",
        "positive_expectancy_fold_count",
        "fold_positive_uplift_count",
        "fold_positive_score_baseline_uplift_count",
        "fold_uplift_ci_lower_min",
        "fold_score_baseline_ci_lower_min",
        "uplift_pass",
        "uplift_failure_reasons",
        "uplift_failure_reasons_detail",
        "score_baseline_policy",
        "trial_count",
        "bootstrap_iterations",
        "is_champion",
    ]
    pooled_slice_columns = [
        "model_name",
        "is_champion",
        "evaluation_scope",
        "split",
        "slice_dimension",
        "slice_value",
        "event_count",
        "success_rate",
        "mean_return_pct",
        "brier_improvement_pct",
        "ece",
        "threshold",
        "selected_event_count",
        "selected_fraction",
        "selected_success_rate",
        "selected_mean_return_pct",
        "selected_minus_all_pct",
        "selected_ci_lower_pct",
        "selected_stop_rate",
        "threshold_reason",
    ]
    pooled_oos_columns = DASHBOARD_POOLED_OOS_COLUMNS
    pooled_oof_columns = DASHBOARD_POOLED_OOF_COLUMNS
    shadow_columns = [
        "symbol",
        "prediction_asof_date",
        "horizon_days",
        "prediction_scope_used",
        "best_model",
        "p_success",
        "threshold",
        "prediction_quality_pass",
        "prediction_signal_status",
        "prediction_use_status",
        "decision_permission",
        "final_trade_decision",
        "paper_action",
        "signal_entry_trigger",
        "signal_trade_action",
        "realized_status",
        "realized_success",
        "realized_net_return_pct",
        "realized_exit_reason",
    ]
    order_intent_columns = [
        "intent_id",
        "signal_id",
        "symbol",
        "symbol_group",
        "asof_date",
        "decision_tier",
        "suggested_action",
        "suggested_weight",
        "semi_momentum_regime",
        "raw_entry_event",
        "stop_price_1_8atr",
        "invalidation_5d_low",
        "invalidation_ema10",
        "next_check_condition",
        "side",
        "order_type",
        "target_weight",
        "max_notional",
        "entry_reference_price",
        "stop_price",
        "limit_price",
        "display_currency",
        "fx_rate_to_usd",
        "usdkrw",
        "status",
        "reason",
        "risk_state",
        "decision_source",
        "live_trading_status",
        "live_order_blocked",
        "live_block_reason",
    ]
    portfolio_risk_columns = [
        "intent_id",
        "symbol",
        "symbol_group",
        "asof_date",
        "decision_tier",
        "suggested_action",
        "suggested_weight",
        "semi_momentum_regime",
        "raw_entry_event",
        "stop_price_1_8atr",
        "invalidation_5d_low",
        "invalidation_ema10",
        "next_check_condition",
        "input_status",
        "portfolio_status",
        "requested_weight",
        "approved_weight",
        "max_single_name_weight",
        "beta_vs_spy_252d",
        "beta_vs_smh_252d",
        "smh_beta_limit_mode",
        "beta_to_spy_warning",
        "beta_to_smh_warning",
        "beta_max_weight_multiplier",
        "order_adv_pct",
        "max_order_adv_pct",
        "block_reason",
        "warning_reasons",
        "risk_policy_version",
    ]
    paper_order_columns = [
        "order_id",
        "intent_id",
        "symbol",
        "side",
        "order_type",
        "status",
        "target_weight",
        "target_notional",
        "quantity",
        "reference_price",
        "limit_price",
        "stop_price",
        "display_currency",
        "fx_rate_to_usd",
        "usdkrw",
        "signal_asof_date",
        "expected_fill_date",
        "fill_model",
        "live_trading_status",
    ]
    paper_fill_columns = [
        "fill_id",
        "order_id",
        "intent_id",
        "symbol",
        "side",
        "fill_type",
        "status",
        "fill_date",
        "fill_price",
        "raw_price",
        "display_currency",
        "fx_rate_to_usd",
        "usdkrw",
        "quantity",
        "gross_notional",
        "commission_bps",
        "slippage_bps",
        "total_cost_bps",
        "reason",
    ]
    paper_position_columns = [
        "symbol",
        "asof_date",
        "quantity",
        "average_price",
        "market_price",
        "market_value",
        "cash",
        "equity",
        "display_currency",
        "fx_rate_to_usd",
        "usdkrw",
        "weight",
        "realized_pnl",
        "unrealized_pnl",
        "position_state",
    ]
    paper_reconciliation_columns = [
        "symbol",
        "asof_date",
        "internal_quantity",
        "recomputed_quantity",
        "quantity_diff",
        "internal_market_value",
        "recomputed_market_value",
        "market_value_diff",
        "display_currency",
        "fx_rate_to_usd",
        "usdkrw",
        "mismatch_count",
        "status",
        "details",
        "live_trading_status",
    ]
    order_state_columns = [
        "state_event_id",
        "intent_id",
        "order_id",
        "symbol",
        "asof_date",
        "lifecycle_state",
        "previous_state",
        "intent_status",
        "portfolio_status",
        "order_status",
        "fill_status",
        "reconciliation_status",
        "block_reason",
        "live_trading_status",
    ]
    execution_feedback_columns = [
        "feedback_id",
        "intent_id",
        "order_id",
        "fill_id",
        "symbol",
        "signal_asof_date",
        "fill_date",
        "horizon_days",
        "order_status",
        "fill_status",
        "fill_model",
        "filled_flag",
        "expired_unfilled_flag",
        "expected_slippage_bps",
        "realized_slippage_bps",
        "slippage_error_bps",
        "post_fill_return_pct",
        "mfe_pct",
        "mae_pct",
        "stop_hit",
        "target_hit",
        "label_matured",
        "execution_adjusted_utility",
        "live_trading_status",
    ]
    execution_label_columns = [
        "feedback_id",
        "intent_id",
        "order_id",
        "symbol",
        "fill_date",
        "horizon_days",
        "label_matured",
        "label_include_for_model",
        "label_success",
        "label_post_fill_return_pct",
        "label_execution_adjusted_utility",
        "label_expired_unfilled",
        "label_stop_hit",
        "label_target_hit",
        "live_trading_status",
    ]
    fill_calibration_columns = [
        "fill_model",
        "event_count",
        "matured_event_count",
        "fill_rate",
        "limit_expiry_rate",
        "mean_realized_slippage_bps",
        "median_realized_slippage_bps",
        "p90_realized_slippage_bps",
        "mean_slippage_error_bps",
        "suggested_half_spread_bps",
        "suggested_gap_penalty_bps",
        "suggested_event_day_penalty_bps",
        "calibration_status",
        "live_trading_status",
    ]
    automation_plan_columns = [
        "task_id",
        "task_name",
        "cadence",
        "local_time",
        "timezone",
        "run_mode",
        "required_inputs",
        "expected_outputs",
        "automation_status",
        "live_trading_status",
        "notes",
    ]
    event_ledger_columns = DASHBOARD_EVENT_LEDGER_COLUMNS
    feedback_columns = DASHBOARD_FEEDBACK_COLUMNS
    cpcv_strategy_columns = [
        "strategy_id",
        "strategy_name",
        "cpcv_path_count",
        "test_uplift_pct_median",
        "test_uplift_pct_q25",
        "test_uplift_pct_min",
        "positive_path_rate_pct",
        "worst_test_drawdown_pct",
        "cpcv_median_uplift_pass",
        "cpcv_worst_quartile_pass",
    ]
    cpcv_model_columns = [
        "model_name",
        "candidate_scope",
        "fold_count",
        "selected_event_count",
        "min_selected_events_per_fold",
        "selected_minus_all_pct_median",
        "selected_minus_all_pct_q25",
        "selected_minus_all_pct_min",
        "positive_fold_rate_pct",
        "cpcv_model_median_uplift_pass",
        "cpcv_model_worst_quartile_pass",
        "source_file",
    ]
    auto_research_columns = [
        "trial_id",
        "trial_source",
        "objective_score",
        "model_family",
        "model_name",
        "candidate_scope",
        "split",
        "horizon_days",
        "pbo_penalty",
        "dsr_penalty",
        "cpcv_penalty",
        "feature_set_hash",
        "label_config_hash",
        "split_config_hash",
        "data_snapshot_hash",
        "tested_at_utc",
    ]
    news_feature_columns = [
        "date",
        "news_event_count_1d",
        "news_event_count_3d",
        "news_sentiment_score_1d",
        "news_primary_cause_type",
        "news_primary_cluster_id",
        "news_match_confidence",
        "news_match_confidence_score",
        "news_coverage_status",
        "news_source_count",
        "news_penalty_event",
        "hist_news_category_count_20d",
        "hist_news_category_success_rate_20d",
        "hist_news_category_mean_return_20d",
        "hist_news_category_count_60d",
        "hist_news_category_success_rate_60d",
        "hist_news_category_mean_return_60d",
        "score_price_algo_total",
        "trade_action",
        "entry_trigger",
    ]

    if not equity.empty and "strategy_id" in equity.columns:
        equity_tail = equity.groupby("strategy_id", group_keys=False).tail(650)
    else:
        equity_tail = equity
    if not ml_overlay_equity.empty:
        overlay_group_keys = [
            col
            for col in ["candidate_scope", "horizon_days", "model_name", "policy"]
            if col in ml_overlay_equity.columns
        ]
        if overlay_group_keys:
            ml_overlay_equity_tail = ml_overlay_equity.groupby(overlay_group_keys, group_keys=False).tail(220)
        else:
            ml_overlay_equity_tail = ml_overlay_equity.tail(1200)
    else:
        ml_overlay_equity_tail = ml_overlay_equity

    latest_price = latest_row_map(OUTPUT_DIR / "tsm_daily_10y_enriched.csv")
    latest_prediction_features = {}
    if not prediction_features.empty:
        latest_feature_cols = [c for c in news_feature_columns if c in prediction_features.columns]
        if latest_feature_cols:
            latest_prediction_features = clean_json(prediction_features[latest_feature_cols].tail(1).to_dict(orient="records")[0])
    news_matches_focus = pd.DataFrame()
    if not news_matches.empty:
        focus = news_matches.copy()
        for col in ["close_change_pct", "open_gap_pct", "volume_ratio_20", "news_match_confidence_score"]:
            if col in focus.columns:
                focus[col] = pd.to_numeric(focus[col], errors="coerce")
        confidence = focus.get("news_match_confidence", pd.Series("", index=focus.index)).astype(str)
        shock = (
            focus.get("close_change_pct", pd.Series(0, index=focus.index)).abs().ge(0.03)
            | focus.get("open_gap_pct", pd.Series(0, index=focus.index)).abs().ge(0.025)
            | focus.get("algo_event_shock_day", pd.Series(False, index=focus.index)).astype(str).str.lower().isin(["true", "1"])
        )
        news_matches_focus = focus[confidence.ne("NO_MATCH") | shock].copy()
    news_feature_contract = pd.DataFrame()
    if not prediction_feature_contract.empty:
        contract_col = prediction_feature_contract.get("column", pd.Series("", index=prediction_feature_contract.index)).astype(str)
        feature_group = prediction_feature_contract.get("feature_group", pd.Series("", index=prediction_feature_contract.index)).astype(str)
        news_feature_contract = prediction_feature_contract[
            contract_col.str.startswith("news_") | contract_col.str.contains("news_category") | feature_group.eq("news_causal_context")
        ].copy()
    news_penalty_summary = {}
    news_penalty_events = pd.DataFrame()
    if not news_daily.empty and "news_penalty_event" in news_daily.columns:
        penalty_frame = news_daily.copy()
        penalty_values = pd.to_numeric(penalty_frame["news_penalty_event"], errors="coerce").fillna(0.0)
        news_penalty_events = penalty_frame[penalty_values.ne(0)].copy()
        recent_penalty_values = penalty_values.tail(120)
        latest_penalty = (
            clean_json(news_penalty_events.tail(1).to_dict(orient="records")[0])
            if not news_penalty_events.empty
            else {}
        )
        news_penalty_summary = clean_json(
            {
                "all_days": int(len(penalty_frame)),
                "penalty_days": int(penalty_values.ne(0).sum()),
                "recent_120_penalty_days": int(recent_penalty_values.ne(0).sum()),
                "total_penalty": float(penalty_values.sum()),
                "max_penalty": float(penalty_values.max()) if len(penalty_values) else 0.0,
                "latest_penalty_date": latest_penalty.get("date"),
                "latest_penalty_cause": latest_penalty.get("news_primary_cause_type"),
                "latest_penalty_summary": latest_penalty.get("news_cause_summary"),
            }
        )
    files = list_output_files()
    reports = [row for row in files if row["kind"] == "report"]
    images = [row for row in files if row["kind"] == "image"]

    model_gate_snapshot = field_value_map(RULE_DIR / "tsm_model_gate_snapshot.csv")
    top10_ontology_snapshot = {
        "title": "Top10+2 Ontology Command Center",
        "decision_scope": "top10",
        "training_scope": "universal_research_pool",
        "decision_symbol_count": int(decision_universe_config["symbol"].nunique()) if not decision_universe_config.empty and "symbol" in decision_universe_config.columns else 0,
        "research_pool_symbol_count": int(research_pool_audit["symbol"].nunique()) if not research_pool_audit.empty and "symbol" in research_pool_audit.columns else 0,
        "top10_timeframe_rows": int(len(top10_timeframe_coverage)),
        "top10_timeframe_ok_rows": int(top10_timeframe_coverage["status"].astype(str).eq("OK").sum()) if not top10_timeframe_coverage.empty and "status" in top10_timeframe_coverage.columns else 0,
    }
    daily_update_summary = build_daily_update_system_summary(
        decision_config=decision_universe_config,
        market_data=universe_market_data_latest,
        predictions=universe_latest_predictions,
        portfolio_decisions=portfolio_risk_decisions,
        paper_orders=paper_orders,
        paper_fills=paper_fills,
        paper_positions=paper_positions,
    )

    return clean_json(
        {
            "generated_at": now_iso(),
            "paths": {
                "root": str(ROOT),
                "output_dir": "output",
                "rule_outdir": "tsm_price_rule_output",
            },
            "snapshots": {
                "summary": field_value_map(OUTPUT_DIR / "tsm_daily_10y_summary.csv"),
                "integrated_price": field_value_map(RULE_DIR / "tsm_integrated_price_summary.csv"),
                "decision": field_value_map(RULE_DIR / "tsm_latest_decision_snapshot.csv"),
                "system": _normalize_system_snapshot_fields(field_value_map(RULE_DIR / "tsm_latest_system_state.csv")),
                "daily_health": field_value_map(RULE_DIR / "tsm_daily_health_snapshot.csv"),
                "data_quality": field_value_map(RULE_DIR / "tsm_latest_data_quality_snapshot.csv"),
                "model_gate": model_gate_snapshot,
                "research_expansion": research_expansion_snapshot(
                    backtest_event_ledger,
                    backtest_feedback_features,
                    auto_research_trials,
                    auto_research_best,
                    cpcv_strategy_distribution,
                    cpcv_model_distribution,
                    model_gate_snapshot,
                ),
                "top10_ontology": top10_ontology_snapshot,
                "daily_update": daily_update_summary,
                "paper_gate": field_value_map(RULE_DIR / "tsm_paper_gate_snapshot.csv"),
                "universe_portfolio": field_value_map(RULE_DIR / "tsm_portfolio_snapshot.csv"),
                "order_intent": latest_record_map(order_intents),
                "portfolio_risk": field_value_map(RULE_DIR / "tsm_portfolio_risk_snapshot.csv"),
                "paper_position": latest_record_map(paper_positions),
                "paper_reconciliation": latest_record_map(paper_reconciliation),
                "order_lifecycle": field_value_map(RULE_DIR / "tsm_order_lifecycle_snapshot.csv"),
                "execution_feedback": latest_row_map(RULE_DIR / "tsm_execution_feedback_events.csv"),
                "fill_model_calibration": latest_row_map(RULE_DIR / "tsm_fill_model_calibration.csv"),
                "automation": latest_row_map(RULE_DIR / "tsm_automation_plan.csv"),
                "universe_validation": universe_validation_snapshot(universe_validation),
                "pooled_sample_audit": sample_audit_snapshot(pooled_sample_audit),
                "risk": _normalize_risk_snapshot_fields(field_value_map(RULE_DIR / "tsm_latest_risk_snapshot.csv")),
                "prediction": _normalize_prediction_snapshot_fields(field_value_map(RULE_DIR / "tsm_latest_prediction_snapshot.csv")),
                "next_day_prediction": _normalize_next_day_prediction_snapshot_fields(
                    next_day_latest_snapshot
                    or _latest_symbol_record(next_day_universe_latest_predictions, "TSM", ("next_day_prediction_asof_date", "date"))
                    or field_value_map(RULE_DIR / "tsm_latest_prediction_snapshot.csv")
                ),
                "next_close_forecast": next_close_latest_snapshot
                or _latest_symbol_record(next_close_universe_latest_predictions, "TSM", ("next_close_prediction_asof_date", "date"))
                or {},
                "pooled_prediction": field_value_map(RULE_DIR / "tsm_pooled_latest_prediction_overlay.csv"),
                "stress": field_value_map(RULE_DIR / "tsm_latest_stress_snapshot.csv"),
                "integrity": field_value_map(RULE_DIR / "tsm_latest_integrity_snapshot.csv"),
                "latest_news": latest_row_map(RULE_DIR / "tsm_news_integrated_daily.csv"),
                "latest_prediction_features": latest_prediction_features,
                "news_penalty_summary": news_penalty_summary,
                "latest_fx": _latest_fx_context(),
                "latest_price": latest_price,
            },
            "quality": {
                "operational": quality_summary(RULE_DIR / "tsm_operational_quality_checks.csv"),
                "integrity": quality_summary(RULE_DIR / "tsm_daily_integrity_checks.csv", critical_only=True),
                "validation": quality_summary(RULE_DIR / "tsm_validation_quality_checks.csv"),
                "prediction": quality_summary(RULE_DIR / "tsm_prediction_quality_checks.csv", critical_only=True),
                "next_day_prediction": quality_summary(RULE_DIR / "tsm_next_day_up_quality_checks.csv", critical_only=True),
                "next_close_forecast": quality_summary(RULE_DIR / "tsm_next_close_quality_checks.csv", critical_only=True),
                "data_contract": mismatch_summary(RULE_DIR / "tsm_data_validation_raw_vs_enriched.csv"),
                "schema": quality_summary(RULE_DIR / "tsm_schema_quality_checks.csv", critical_only=True),
                "ml_overlay": quality_summary(RULE_DIR / "tsm_ml_overlay_quality_checks.csv", critical_only=True),
                "pooled_dataset": quality_summary(RULE_DIR / "tsm_prediction_pooled_quality_checks.csv", critical_only=True),
                "pooled_model": quality_summary(RULE_DIR / "tsm_pooled_model_quality_checks.csv", critical_only=True),
                "shadow_paper": quality_summary(RULE_DIR / "tsm_shadow_paper_quality_checks.csv", critical_only=True),
                "order_intent": quality_summary(RULE_DIR / "tsm_order_intent_quality_checks.csv", critical_only=True),
                "portfolio_risk": quality_summary(RULE_DIR / "tsm_portfolio_risk_checks.csv", critical_only=True),
                "paper_execution": quality_summary(RULE_DIR / "tsm_paper_oms_quality_checks.csv", critical_only=True),
                "paper_reconciliation": quality_summary(RULE_DIR / "tsm_paper_reconciliation_quality_checks.csv", critical_only=True),
                "order_state": quality_summary(RULE_DIR / "tsm_order_state_quality_checks.csv", critical_only=True),
                "execution_feedback": quality_summary(RULE_DIR / "tsm_execution_feedback_quality_checks.csv", critical_only=True),
                "fill_calibration": quality_summary(RULE_DIR / "tsm_fill_model_calibration_quality_checks.csv", critical_only=True),
                "automation": quality_summary(RULE_DIR / "tsm_automation_quality_checks.csv", critical_only=True),
                "data_quality": quality_summary(RULE_DIR / "tsm_data_quality_checks.csv", critical_only=True),
                "model_gate": quality_summary(RULE_DIR / "tsm_model_gate_audit.csv", critical_only=True),
                "backtest_feedback": quality_summary(RULE_DIR / "tsm_backtest_feedback_quality_checks.csv", critical_only=True),
                "full_daily_update": quality_summary(RULE_DIR / "tsm_full_daily_update_audit.csv", critical_only=True),
            },
            "series": {
                "price": records_from_df(enriched, price_columns, tail=650),
                "signals": records_from_df(signals, signal_columns, tail=650),
                "equity": records_from_df(equity_tail, equity_columns),
                "risk": records_from_df(risk, risk_columns, tail=650),
                "ml_overlay_equity": records_from_df(ml_overlay_equity_tail, overlay_equity_columns),
            },
            "tables": {
                "trading_plan": records_from_df(trading_plan),
                "backtest_summary": records_from_df(backtest),
                "trade_log": records_from_df(trade_log.tail(250) if not trade_log.empty else trade_log, trade_columns),
                "yearly_returns": records_from_df(yearly),
                "prediction_comparison": records_from_df(prediction_comparison),
                "prediction_audit": records_from_df(prediction_audit),
                "prediction_near_pass_candidates": records_from_df(prediction_near_pass),
                "prediction_performance_gap_summary": records_from_df(prediction_performance_gap),
                "prediction_feature_association_summary": records_from_df(prediction_feature_association),
                "prediction_brier_decomposition_summary": records_from_df(prediction_brier_decomposition),
                "prediction_calibration": records_from_df(calibration),
                "prediction_candidate_scope": records_from_df(candidate_scope),
                "prediction_label_diagnostics": records_from_df(label_diagnostics),
                "prediction_feature_selection": records_from_df(feature_selection.tail(400) if not feature_selection.empty else feature_selection),
                "prediction_walk_metrics": records_from_df(walk_metrics.tail(500) if not walk_metrics.empty else walk_metrics),
                "prediction_threshold_policy": records_from_df(threshold_policy.tail(400) if not threshold_policy.empty else threshold_policy),
                "prediction_oos_predictions": records_from_df(prediction_oos.tail(500) if not prediction_oos.empty else prediction_oos, prediction_oos_columns),
                "prediction_calibration_summary": records_from_df(prediction_calibration_summary),
                "prediction_policy_audit": records_from_df(prediction_policy_audit),
                "prediction_feature_contract": records_from_df(prediction_feature_contract),
                "prediction_fold_manifest": records_from_df(prediction_fold_manifest),
                "prediction_model_registry": records_from_df(prediction_model_registry, registry_columns),
                "prediction_experiment_log": records_from_df(prediction_experiment_log, experiment_columns),
                "next_day_model_comparison": records_from_df(next_day_model_comparison),
                "next_day_universe_latest_predictions": records_from_df(next_day_universe_latest_predictions),
                "next_day_oos_predictions": records_from_df(next_day_oos_predictions.tail(500) if not next_day_oos_predictions.empty else next_day_oos_predictions),
                "next_day_calibration_summary": records_from_df(next_day_calibration_summary),
                "next_day_threshold_policy": records_from_df(next_day_threshold_policy.tail(400) if not next_day_threshold_policy.empty else next_day_threshold_policy),
                "next_day_quality_checks": records_from_df(next_day_quality),
                "next_close_model_comparison": records_from_df(next_close_model_comparison),
                "next_close_universe_latest_predictions": records_from_df(next_close_universe_latest_predictions),
                "next_close_oos_predictions": records_from_df(next_close_oos_predictions.tail(500) if not next_close_oos_predictions.empty else next_close_oos_predictions),
                "next_close_interval_calibration": records_from_df(next_close_interval_calibration),
                "next_close_quality_checks": records_from_df(next_close_quality),
                "validation_walk_forward": records_from_df(walk_forward),
                "validation_causal_walk_forward": records_from_df(causal_walk_forward),
                "validation_segments": records_from_df(validation_segments),
                "validation_sensitivity": records_from_df(sensitivity),
                "rule_threshold_sensitivity": records_from_df(rule_threshold_sensitivity.tail(800) if not rule_threshold_sensitivity.empty else rule_threshold_sensitivity),
                "rule_threshold_sensitivity_summary": records_from_df(rule_threshold_sensitivity_summary),
                "rule_threshold_sensitivity_quality_checks": records_from_df(rule_threshold_sensitivity_quality),
                "stress_scenarios": records_from_df(stress_scenarios),
                "strategy_stress": records_from_df(strategy_stress),
                "readiness_scorecard": records_from_df(readiness),
                "manifest": records_from_df(manifest.tail(40) if not manifest.empty else manifest),
                "full_daily_update_audit": records_from_df(full_daily_audit.tail(300) if not full_daily_audit.empty else full_daily_audit),
                "data_validation": records_from_df(data_validation),
                "yearly_price": records_from_df(yearly_price),
                "monthly_price": records_from_df(monthly_price.tail(60) if not monthly_price.empty else monthly_price),
                "extreme_moves": records_from_df(extreme_moves),
                "event_analysis": records_from_df(event_analysis),
                "news_daily": records_from_df(news_daily.tail(120) if not news_daily.empty else news_daily),
                "news_matches": records_from_df(news_matches.tail(160) if not news_matches.empty else news_matches),
                "news_matches_focus": records_from_df(news_matches_focus.tail(500) if not news_matches_focus.empty else news_matches_focus),
                "news_high_matches": records_from_df(news_matches[news_matches["news_match_confidence"].astype(str).eq("HIGH")] if not news_matches.empty and "news_match_confidence" in news_matches.columns else pd.DataFrame()),
                "news_penalty_events": records_from_df(news_penalty_events.tail(100) if not news_penalty_events.empty else news_penalty_events),
                "news_clusters": records_from_df(news_clusters.tail(120) if not news_clusters.empty else news_clusters),
                "news_cause_forward": records_from_df(news_cause_forward),
                "news_feature_matrix": records_from_df(prediction_features.tail(120) if not prediction_features.empty else prediction_features, news_feature_columns),
                "news_feature_contract": records_from_df(news_feature_contract),
                "regime_forward": records_from_df(regime_forward),
                "rule_forward": records_from_df(rule_forward),
                "pbo_report": records_from_df(pbo),
                "deflated_sharpe": records_from_df(dsr),
                "model_trials": records_from_df(model_trials.tail(120) if not model_trials.empty else model_trials),
                "schema_quality_checks": records_from_df(schema_quality),
                "cscv_pbo_report": records_from_df(cscv_pbo),
                "ml_overlay_summary": records_from_df(ml_overlay_summary, overlay_summary_columns),
                "ml_overlay_quality_checks": records_from_df(ml_overlay_quality),
                "pooled_model_comparison": records_from_df(pooled_model_comparison, pooled_comparison_columns),
                "pooled_model_quality_checks": records_from_df(pooled_model_quality),
                "pooled_tsm_calibration": records_from_df(pooled_tsm_calibration),
                "pooled_tsm_calibration_metrics": records_from_df(pooled_tsm_calibration_metrics, pooled_comparison_columns),
                "stop_risk_baseline_summary": records_from_df(stop_risk_baseline_summary),
                "stop_risk_calibration_bins": records_from_df(stop_risk_calibration_bins),
                "stop_risk_slice_diagnostics": records_from_df(stop_risk_slice_diagnostics.tail(1200) if not stop_risk_slice_diagnostics.empty else stop_risk_slice_diagnostics),
                "stop_risk_latest_distribution": records_from_df(stop_risk_latest_distribution),
                "pooled_threshold_policy": records_from_df(pooled_threshold_policy),
                "pooled_uplift_bootstrap": records_from_df(pooled_uplift_bootstrap, pooled_uplift_columns),
                "pooled_oos_predictions": records_from_df(pooled_oos_predictions.tail(500) if not pooled_oos_predictions.empty else pooled_oos_predictions, pooled_oos_columns),
                "pooled_oof_predictions": records_from_df(pooled_oof_predictions.tail(800) if not pooled_oof_predictions.empty else pooled_oof_predictions, pooled_oof_columns),
                "pooled_slice_diagnostics": records_from_df(pooled_slice_diagnostics.tail(1200) if not pooled_slice_diagnostics.empty else pooled_slice_diagnostics, pooled_slice_columns),
                "pooled_dataset_quality_checks": records_from_df(pooled_dataset_quality),
                "pooled_scope_stats": records_from_df(pooled_scope_stats.tail(300) if not pooled_scope_stats.empty else pooled_scope_stats),
                "pooled_schema": records_from_df(pooled_schema.tail(300) if not pooled_schema.empty else pooled_schema),
                "pooled_input_failures": records_from_df(pooled_input_failures),
                "pooled_universe_config": records_from_df(pooled_universe_config),
                "decision_universe_config": records_from_df(decision_universe_config),
                "universe_risk_latest": universe_risk_latest_rows(decision_universe_config),
                "research_pool_audit": records_from_df(research_pool_audit),
                "top10_timeframe_coverage": records_from_df(top10_timeframe_coverage),
                "research_pool_timeframe_coverage": records_from_df(research_pool_timeframe_coverage.tail(500) if not research_pool_timeframe_coverage.empty else research_pool_timeframe_coverage),
                "intraday_daily_features": records_from_df(intraday_daily_features.tail(500) if not intraday_daily_features.empty else intraday_daily_features),
                "intraday_feature_quality_checks": records_from_df(intraday_feature_quality),
                "pooled_split_manifest": records_from_df(pooled_split_manifest),
                "pooled_universe_manifest": records_from_df(pooled_universe_manifest.tail(80) if not pooled_universe_manifest.empty else pooled_universe_manifest),
                "universe_validation": records_from_df(universe_validation),
                "universe_latest_signals": records_from_df(
                    universe_latest_signals,
                    [
                        "symbol",
                        "symbol_group",
                        "date",
                        "close",
                        "score_price_algo_total",
                        "raw_entry_event",
                        "decision_tier",
                        "sizing_tier",
                        "suggested_action",
                        "suggested_weight",
                        "semi_momentum_regime",
                        "semi_group_momentum_score",
                        "memory_ai_regime_score",
                        "positive_thrust_day",
                        "negative_shock_day",
                        "stop_price_1_8atr",
                        "invalidation_5d_low",
                        "invalidation_ema10",
                        "next_check_condition",
                        "execution_status",
                    ],
                ),
                "universe_latest_predictions": records_from_df(universe_latest_predictions),
                "universe_rule_fallback_diagnostics": records_from_df(universe_rule_fallback_diagnostics),
                "portfolio_targets": records_from_df(portfolio_targets),
                "portfolio_snapshot": records_from_df(portfolio_snapshot),
                "pooled_sample_audit": records_from_df(pooled_sample_audit),
                "tsm_like_calibration_pool": records_from_df(tsm_like_calibration_pool.tail(800) if not tsm_like_calibration_pool.empty else tsm_like_calibration_pool),
                "tsm_like_calibration_metrics": records_from_df(tsm_like_calibration_metrics),
                "paper_gate_snapshot": records_from_df(paper_gate_snapshot_table),
                "pooled_learning_curve": records_from_df(pooled_learning_curve),
                "prediction_benchmark": records_from_df(prediction_benchmark),
                "event_portfolio_backtest": records_from_df(event_portfolio.tail(700) if not event_portfolio.empty else event_portfolio),
                "external_feature_schema": records_from_df(external_feature_schema),
                "shadow_predictions": records_from_df(shadow_predictions, shadow_columns),
                "shadow_quality_checks": records_from_df(shadow_quality),
                "order_intents": records_from_df(order_intents, order_intent_columns),
                "order_intent_quality_checks": records_from_df(order_intent_quality),
                "portfolio_risk_order_decisions": records_from_df(portfolio_risk_decisions, portfolio_risk_columns),
                "portfolio_risk_checks": records_from_df(portfolio_risk_checks),
                "paper_orders": records_from_df(paper_orders, paper_order_columns),
                "paper_fills": records_from_df(paper_fills.tail(250) if not paper_fills.empty else paper_fills, paper_fill_columns),
                "paper_positions": records_from_df(paper_positions, paper_position_columns),
                "paper_slippage": records_from_df(paper_slippage),
                "paper_oms_quality_checks": records_from_df(paper_oms_quality),
                "paper_reconciliation": records_from_df(paper_reconciliation, paper_reconciliation_columns),
                "paper_reconciliation_quality_checks": records_from_df(paper_reconciliation_quality),
                "order_state_events": records_from_df(order_state_events, order_state_columns),
                "order_lifecycle_snapshot": records_from_df(order_lifecycle_snapshot),
                "order_state_quality_checks": records_from_df(order_state_quality),
                "execution_feedback_events": records_from_df(execution_feedback_events.tail(250) if not execution_feedback_events.empty else execution_feedback_events, execution_feedback_columns),
                "execution_feedback_features": records_from_df(execution_feedback_features.tail(250) if not execution_feedback_features.empty else execution_feedback_features),
                "execution_feedback_labels": records_from_df(execution_feedback_labels.tail(250) if not execution_feedback_labels.empty else execution_feedback_labels, execution_label_columns),
                "execution_feedback_quality_checks": records_from_df(execution_feedback_quality),
                "fill_model_calibration": records_from_df(fill_model_calibration, fill_calibration_columns),
                "fill_model_calibration_quality_checks": records_from_df(fill_model_calibration_quality),
                "automation_plan": records_from_df(automation_plan, automation_plan_columns),
                "automation_quality_checks": records_from_df(automation_quality),
                "data_quality_checks": records_from_df(data_quality_checks),
                "data_quality_issues": records_from_df(data_quality_issues),
                "model_gate_audit": records_from_df(model_gate_audit.tail(500) if not model_gate_audit.empty else model_gate_audit),
                "model_gate_root_causes": records_from_df(model_gate_root_causes),
                "model_candidate_disposition_summary": records_from_df(model_candidate_disposition),
                "model_rank_policy_promotion_watchlist": records_from_df(model_rank_policy_watchlist),
                "model_performance_warning_resolution_summary": records_from_df(model_warning_resolution),
                "model_unresolved_performance_priorities": records_from_df(model_unresolved_performance_priorities),
                "system_block_reasons": records_from_df(system_block_reasons),
                "backtest_event_ledger": records_from_df(backtest_event_ledger.tail(800) if not backtest_event_ledger.empty else backtest_event_ledger, event_ledger_columns),
                "backtest_feedback_features": records_from_df(backtest_feedback_features.tail(800) if not backtest_feedback_features.empty else backtest_feedback_features, feedback_columns),
                "backtest_feedback_quality_checks": records_from_df(backtest_feedback_quality),
                "cpcv_path_summary": records_from_df(cpcv_path_summary.tail(500) if not cpcv_path_summary.empty else cpcv_path_summary),
                "cpcv_strategy_distribution": records_from_df(cpcv_strategy_distribution, cpcv_strategy_columns),
                "cpcv_model_distribution": records_from_df(cpcv_model_distribution, cpcv_model_columns),
                "auto_research_trial_ledger": records_from_df(auto_research_trials.tail(500) if not auto_research_trials.empty else auto_research_trials, auto_research_columns),
                "auto_research_best_candidates": records_from_df(auto_research_best, auto_research_columns),
                "research_expansion_manifest": records_from_df(research_expansion_manifest.tail(80) if not research_expansion_manifest.empty else research_expansion_manifest),
                "universe_market_data_latest": records_from_df(universe_market_data_latest),
                "universe_intraday_manifest": records_from_df(universe_intraday_manifest.tail(120) if not universe_intraday_manifest.empty else universe_intraday_manifest),
            },
            "files": files,
            "reports": reports,
            "images": images,
            "run": get_run_state(),
        }
    )


def clear_summary_cache() -> None:
    with SUMMARY_CACHE_LOCK:
        SUMMARY_CACHE["expires_at"] = 0.0
        SUMMARY_CACHE["data"] = None
    DISPLAY_CONTEXT_CACHE.clear()


def cached_build_summary() -> dict[str, Any]:
    now = time.monotonic()
    with SUMMARY_CACHE_LOCK:
        cached = SUMMARY_CACHE.get("data")
        if cached is not None and now < float(SUMMARY_CACHE.get("expires_at", 0.0)):
            return clean_json(cached)
    data = build_summary()
    with SUMMARY_CACHE_LOCK:
        SUMMARY_CACHE["data"] = data
        SUMMARY_CACHE["expires_at"] = time.monotonic() + SUMMARY_CACHE_TTL_SECONDS
    return data


def cached_build_investor_dashboard() -> dict[str, Any]:
    """Cached, single-flight wrapper around build_investor_dashboard().

    The first build is unavoidably slow (~40s on a slow volume); subsequent
    loads within the TTL return instantly. The build lock ensures concurrent
    requests (e.g. a startup pre-warm plus the window's first fetch) reuse one
    build instead of stampeding."""
    now = time.monotonic()
    with INVESTOR_CACHE_LOCK:
        cached = INVESTOR_CACHE.get("data")
        if cached is not None and now < float(INVESTOR_CACHE.get("expires_at", 0.0)):
            return clean_json(cached)
    with INVESTOR_BUILD_LOCK:
        now = time.monotonic()
        with INVESTOR_CACHE_LOCK:
            cached = INVESTOR_CACHE.get("data")
            if cached is not None and now < float(INVESTOR_CACHE.get("expires_at", 0.0)):
                return clean_json(cached)
        data = build_investor_dashboard()
        with INVESTOR_CACHE_LOCK:
            INVESTOR_CACHE["data"] = data
            INVESTOR_CACHE["expires_at"] = time.monotonic() + INVESTOR_CACHE_TTL_SECONDS
        return data


def prewarm_investor_dashboard() -> None:
    """Build the investor-dashboard cache in the background so the UI populates
    as soon as possible after launch. Safe to call from a daemon thread."""
    try:
        cached_build_investor_dashboard()
    except Exception:
        pass


def build_live_quotes() -> dict[str, Any]:
    """Real-time price layer for the investor dashboard (price-derived only).

    Returns live current price / intraday change / market state per symbol from
    tsm_live_quote_engine (Finnhub for US, Yahoo for KR). This is intentionally
    separate from the daily model payload: only PRICE updates intra-session;
    predictions stay as-of the last close."""
    try:
        import tsm_live_quote_engine

        decision_config = RULE_DIR / "tsm_decision_universe_config.csv"
        config_path = decision_config if decision_config.exists() else DEFAULT_UNIVERSE_CONFIG
        return clean_json(tsm_live_quote_engine.fetch_live_quotes(config_path))
    except Exception as exc:  # pragma: no cover - network/runtime guard
        return {"error": str(exc), "quotes": {}}


def _allowed_universe_symbols() -> dict[str, str]:
    """Map of allowed UPPER symbol -> symbol_group from the decision universe."""
    config = safe_read_csv(RULE_DIR / "tsm_decision_universe_config.csv")
    if config.empty or "symbol" not in config.columns:
        config = safe_read_csv(Path(DEFAULT_UNIVERSE_CONFIG))
    out: dict[str, str] = {}
    if not config.empty and "symbol" in config.columns:
        for _, row in config.iterrows():
            sym = str(row.get("symbol", "")).strip().upper()
            if sym:
                out[sym] = str(row.get("symbol_group", "")).strip()
    return out


SYMBOL_GROUP_LABELS = {
    "fabless_ai_analog": "팹리스 · AI/아날로그",
    "foundry_idm": "파운드리 · IDM",
    "memory_foundry_idm": "메모리 · 파운드리",
    "memory_storage": "메모리",
    "semicap_osat": "장비 · OSAT",
}

SYMBOL_DISPLAY_NAMES = {
    "TSM": "TSMC",
    "NVDA": "NVIDIA",
    "AVGO": "Broadcom",
    "AMD": "Advanced Micro Devices",
    "INTC": "Intel",
    "MU": "Micron Technology",
    "TXN": "Texas Instruments",
    "LRCX": "Lam Research",
    "AMAT": "Applied Materials",
    "QCOM": "Qualcomm",
    "005930.KS": "Samsung Electronics",
    "000660.KS": "SK hynix",
}


POOLED_DASHBOARD_HORIZONS = (5, 20, 60)
POOLED_HORIZON_PREDICTION_FIELDS = (
    "p_success",
    "p_stop_hit",
    "p_stop_hit_raw",
    "p_stop_hit_calibrated",
    "p_stop_hit_raw_minus_calibrated",
    "p_stop_hit_oos_percentile",
    "strict_stop_risk_gap",
    "paper_stop_risk_gap",
    "expected_r",
    "decision_score",
    "paper_decision_score",
    "threshold",
)


def _pooled_horizon_prediction_columns() -> list[str]:
    return [
        f"{field}_{horizon}d"
        for horizon in POOLED_DASHBOARD_HORIZONS
        for field in POOLED_HORIZON_PREDICTION_FIELDS
    ]


UNIVERSE_LATEST_PREDICTION_COLUMNS = [
    "symbol",
    "symbol_group",
    "date",
    "latest_trade_ready",
    *_pooled_horizon_prediction_columns(),
    "stop_risk_calibration_warning",
    "decision_support_allowed",
    "paper_decision_support_allowed",
    "block_reasons",
    "model_name",
    "prediction_source",
    "live_trading_status",
]


def symbol_display_name(symbol: Any) -> str:
    key = str(symbol or "").strip().upper()
    return SYMBOL_DISPLAY_NAMES.get(key, key)


def universe_risk_latest_rows(decision_config: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-symbol risk-engine recommended max weight (the real 'risk cap').

    Reads each decision symbol's tiny tsm_latest_risk_snapshot.csv (key/value),
    where final_recommended_max_weight is stored in PERCENT (e.g. 7.0 == 7%).
    """
    rows: list[dict[str, Any]] = []
    if decision_config.empty or "symbol" not in decision_config.columns:
        return rows
    for _, member in decision_config.iterrows():
        sym = str(member.get("symbol", "")).strip().upper()
        if not sym:
            continue
        snap = field_value_map(RULE_DIR / "universe" / sym / "tsm_latest_risk_snapshot.csv")
        if not snap:
            continue
        rows.append(
            {
                "symbol": sym,
                "symbol_group": str(member.get("symbol_group", "")),
                "date": snap.get("date"),
                "final_recommended_max_weight_pct": _as_float(snap.get("final_recommended_max_weight")),
                "risk_state": snap.get("risk_state"),
                "limiting_reason": snap.get("limiting_reason"),
            }
        )
    return rows


def _decision_config_or_default(decision_config: pd.DataFrame) -> pd.DataFrame:
    if not decision_config.empty and "symbol" in decision_config.columns:
        return decision_config
    return safe_read_csv(ROOT / DEFAULT_UNIVERSE_CONFIG)


def _risk_allows_decision(risk_state: Any) -> bool:
    state = str(risk_state or "").strip().upper()
    if not state:
        return True
    blocking_tokens = ("OBSERVATION", "TINY", "BLOCK", "AVOID", "SUSPEND", "DISABLE")
    return not any(token in state for token in blocking_tokens)


INVESTOR_DASHBOARD_SYMBOL_ORDER = [
    "NVDA",
    "TSM",
    "AVGO",
    "AMD",
    "INTC",
    "MU",
    "TXN",
    "LRCX",
    "AMAT",
    "QCOM",
    "005930.KS",
    "000660.KS",
]

INVESTOR_ACTION_PRIORITY = {
    "execute_candidate": 0,
    "watch": 1,
    "reduce": 2,
    "avoid": 3,
    "blocked": 4,
}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "pass", "allowed", "approved"}


def _first_text(*values: Any, default: str = "") -> str:
    for value in values:
        if _is_blank_value(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null", "na", "n/a"}:
            return text
    return default


def _ratio_value(value: Any) -> float | None:
    parsed = _as_float(value)
    if parsed is None:
        return None
    if abs(parsed) > 1.0 and abs(parsed) <= 100.0:
        parsed = parsed / 100.0
    return parsed


def _date_value(value: Any) -> date | None:
    if _is_blank_value(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _latest_symbol_record(df: pd.DataFrame, symbol: str, date_columns: tuple[str, ...] = ("date", "asof_date", "prediction_asof_date")) -> dict[str, Any]:
    if df.empty or "symbol" not in df.columns:
        return {}
    symbol_key = symbol.upper()
    rows = df[df["symbol"].astype(str).str.upper() == symbol_key].copy()
    if rows.empty:
        return {}
    for col in date_columns:
        if col in rows.columns:
            rows["_sort_date"] = pd.to_datetime(rows[col], errors="coerce")
            rows = rows.sort_values("_sort_date")
            break
    return clean_json(rows.iloc[-1].drop(labels=["_sort_date"], errors="ignore").to_dict())


def _latest_record_date(row: dict[str, Any], date_columns: tuple[str, ...]) -> date | None:
    for col in date_columns:
        parsed = _date_value(row.get(col))
        if parsed is not None:
            return parsed
    return None


def _freshest_symbol_record(
    aggregate_df: pd.DataFrame,
    symbol: str,
    symbol_path: Path,
    date_columns: tuple[str, ...] = ("date", "asof_date", "prediction_asof_date"),
) -> dict[str, Any]:
    aggregate = _latest_symbol_record(aggregate_df, symbol, date_columns)
    per_symbol = latest_row_map(symbol_path)
    if not per_symbol:
        return aggregate
    aggregate_date = _latest_record_date(aggregate, date_columns)
    per_symbol_date = _latest_record_date(per_symbol, date_columns)
    if aggregate_date is None or (per_symbol_date is not None and per_symbol_date >= aggregate_date):
        return per_symbol
    return aggregate


def _freshest_date_text(*values: Any) -> str:
    dated: list[tuple[date, str]] = []
    fallback = ""
    for value in values:
        text = _first_text(value, default="")
        if not text:
            continue
        if not fallback:
            fallback = text
        parsed = _date_value(text)
        if parsed is not None:
            dated.append((parsed, text))
    if not dated:
        return fallback
    dated.sort(key=lambda item: item[0])
    return dated[-1][1]


def _investor_market_data_latest() -> pd.DataFrame:
    """Return market latest rows, with fresher KR2 automation rows overriding globals."""
    global_market = safe_read_csv(OUTPUT_DIR / "tsm_universe_market_data_latest.csv")
    kr2_market = safe_read_csv(OUTPUT_DIR / "kr2_automation" / "tsm_universe_market_data_latest.csv")
    if kr2_market.empty or "symbol" not in kr2_market.columns:
        return global_market
    if global_market.empty or "symbol" not in global_market.columns:
        return kr2_market

    kr2_symbols = {str(sym).strip().upper() for sym in kr2_market["symbol"].dropna().tolist()}
    base = global_market[
        ~global_market["symbol"].astype(str).str.upper().isin(kr2_symbols)
    ].copy()
    return pd.concat([base, kr2_market], ignore_index=True, sort=False)


def _decision_dashboard_members() -> list[dict[str, Any]]:
    config = _decision_config_or_default(safe_read_csv(RULE_DIR / "tsm_decision_universe_config.csv"))
    members: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not config.empty and "symbol" in config.columns:
        for _, row in config.iterrows():
            symbol = str(row.get("symbol", "")).strip().upper()
            if not symbol or symbol in seen:
                continue
            if "enabled" in config.columns and not _truthy(row.get("enabled", True)):
                continue
            if symbol not in INVESTOR_DASHBOARD_SYMBOL_ORDER:
                continue
            seen.add(symbol)
            members.append(clean_json(row.to_dict()))
    order_index = {symbol: idx for idx, symbol in enumerate(INVESTOR_DASHBOARD_SYMBOL_ORDER)}
    if len(members) < len(INVESTOR_DASHBOARD_SYMBOL_ORDER):
        for symbol in INVESTOR_DASHBOARD_SYMBOL_ORDER:
            if symbol not in seen:
                members.append({"symbol": symbol, "symbol_group": "", "display_currency": "KRW" if symbol.endswith(".KS") else "USD"})
                seen.add(symbol)
    members.sort(key=lambda row: order_index.get(str(row.get("symbol", "")).strip().upper(), 999))
    return members


def _display_currency_for(member: dict[str, Any], signal: dict[str, Any], market: dict[str, Any]) -> str:
    currency = _first_text(
        signal.get("display_currency"),
        market.get("display_currency"),
        member.get("display_currency"),
        member.get("listing_currency"),
        default="USD",
    ).upper()
    return "KRW" if currency == "KRW" else "USD"


def _display_price_from_engine(value: Any, currency: str, fx_rate_to_usd: Any = None) -> float | None:
    parsed = _as_float(value)
    if parsed is None:
        return None
    if currency == "KRW":
        fx = _as_float(fx_rate_to_usd)
        if fx and fx > 0:
            return parsed / fx
    return parsed


def _latest_display_price(signal: dict[str, Any], market: dict[str, Any], currency: str, fx_rate_to_usd: Any = None) -> float | None:
    if currency == "KRW":
        engine_price = (
            _as_float(signal.get("close_usd"))
            or _as_float(market.get("latest_close_usd"))
            or _as_float(signal.get("close"))
            or _as_float(market.get("latest_close"))
        )
        converted = _display_price_from_engine(engine_price, currency, fx_rate_to_usd)
        return converted or _as_float(signal.get("close_native")) or _as_float(market.get("latest_close_native"))
    return _as_float(signal.get("close_usd")) or _as_float(market.get("latest_close_usd")) or _as_float(signal.get("close")) or _as_float(market.get("latest_close"))


def _price_series_for_symbol(symbol: str, currency: str, default_fx_rate_to_usd: Any) -> list[dict[str, Any]]:
    df = safe_read_csv(OUTPUT_DIR / "universe" / symbol / "tsm_daily_10y_enriched.csv")
    if df.empty or "date" not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in df.tail(120).iterrows():
        if currency == "KRW":
            engine_price = _as_float(row.get("close_usd")) or _as_float(row.get("close"))
            price = _display_price_from_engine(engine_price, currency, default_fx_rate_to_usd) or _as_float(row.get("close_native"))
        else:
            price = _as_float(row.get("close_usd")) or _as_float(row.get("close"))
        if price is None:
            continue
        rows.append({"date": str(row.get("date", ""))[:10], "price": price})
    return rows


def _translate_next_check(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return "장 마감 후 재확인"
    if "WAIT_FOR_RAW_BREAKOUT_OR_PULLBACK_REENTRY" in text:
        return "돌파 또는 눌림목 재진입 확인"
    if "HOLD_ONLY_IF_CLOSE_ABOVE_5D_LOW_AND_10D_EMA" in text:
        return "5일 저점과 10일 평균선 위에서만 유지"
    if "WAIT_FOR_NEW_SETUP" in text:
        return "새 신호가 생길 때까지 대기"
    if "WAIT" in text:
        return "조건이 다시 맞는지 확인"
    return text.replace("_", " ").strip()


def _reason_tokens(*values: Any) -> list[str]:
    tokens: list[str] = []
    for value in values:
        text = str(value or "").strip().upper()
        if not text or text in {"PASS", "NONE", "NAN"}:
            continue
        for token in re.split(r"[|,;]\s*|\s{2,}", text):
            token = token.strip()
            if token and token not in {"PASS", "NONE", "NAN"}:
                tokens.append(token)
    return list(dict.fromkeys(tokens))


def _new_buy_block_reasons(
    signal: dict[str, Any],
    prediction: dict[str, Any],
    portfolio: dict[str, Any],
    intent: dict[str, Any],
    close_forecast_20d: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    reasons: list[dict[str, str]] = []
    raw_tokens = _reason_tokens(
        portfolio.get("block_reason"),
        portfolio.get("warning_reasons"),
        intent.get("reason"),
        prediction.get("block_reasons"),
        signal.get("block_reason"),
        signal.get("warning_reasons"),
    )

    def add(code: str, label: str, detail: str, severity: str = "block") -> None:
        if not any(item["code"] == code for item in reasons):
            reasons.append({"code": code, "label": label, "detail": detail, "severity": severity})

    token_labels = {
        "REQUESTED_WEIGHT_ZERO": ("권장 비중 0%", "포트폴리오 리스크 엔진이 신규 매수 비중을 0으로 제한했습니다."),
        "NO_NEW_RISK": ("신규 리스크 한도 없음", "현재 계좌/유니버스 조건에서 새 위험 예산을 배정하지 않았습니다."),
        "LATEST_NOT_TRADE_READY": ("최신 행 미준비", "최신 20일 예측 행이 trade-ready 진입 후보가 아닙니다."),
        "MAX_OPEN_POSITIONS": ("열린 후보 수 한도", "동시에 열 수 있는 후보 수 한도에 걸렸습니다."),
        "BETA_TO_SPY_LIMIT": ("시장 베타 한도", "SPY 대비 민감도가 신규 진입 한도를 넘었습니다."),
        "BETA_TO_SMH_LIMIT": ("반도체 베타 한도", "SMH 대비 민감도가 신규 진입 한도를 넘었습니다."),
        "INTRADAY_VOL_HIGH": ("장중 변동성 높음", "분봉 변동성이 높아 신규 진입을 보류합니다."),
        "INTRADAY_COVERAGE_LOW": ("장중 데이터 부족", "분봉 커버리지가 낮아 신규 진입 확인력이 부족합니다."),
        "NO_ACTIONABLE_SIGNAL": ("실행 신호 없음", "가격 룰과 예측이 동시에 실행 조건을 만들지 못했습니다."),
    }
    for token in raw_tokens:
        if token in token_labels:
            label, detail = token_labels[token]
            add(token, label, detail)

    portfolio_status = str(portfolio.get("portfolio_status") or intent.get("status") or "").upper()
    decision_tier = str(signal.get("decision_tier") or portfolio.get("decision_tier") or intent.get("decision_tier") or "").upper()
    suggested_action = str(signal.get("suggested_action") or portfolio.get("suggested_action") or intent.get("suggested_action") or "").upper()
    trade_action = str(signal.get("trade_action") or "").upper()
    if portfolio_status == "REJECTED":
        add("PORTFOLIO_REJECTED", "포트폴리오 한도 차단", "포트폴리오 리스크 엔진의 최신 결정이 REJECTED입니다.")
    if "AVOID_OR_WAIT" in {decision_tier, suggested_action}:
        add("RULE_AVOID_OR_WAIT", "룰 엔진 대기/회피", "가격 룰이 신규 진입보다 대기를 우선합니다.")
    if "REDUCE_OR_DO_NOT_CHASE" in trade_action:
        add("DO_NOT_CHASE", "추격매수 금지", "가격이 과열 또는 고변동 구간이라 새 진입 대신 눌림목 확인이 필요합니다.")

    p_success = _ratio_value(prediction.get("p_success_20d"))
    threshold = _ratio_value(prediction.get("threshold_20d"))
    stop_risk = _ratio_value(prediction.get("p_stop_hit_calibrated_20d", prediction.get("p_stop_hit_20d")))
    expected_r = _as_float(prediction.get("expected_r_20d"))
    forecast_return = _as_float((close_forecast_20d or {}).get("predicted_return_pct"))
    if p_success is None:
        add("P_SUCCESS_20D_MISSING", "20일 성공확률 없음", "20일 예측 확률이 없어 신규 매수 판단을 보수적으로 처리합니다.", "warn")
    elif threshold is not None and p_success < threshold:
        add("P_SUCCESS_BELOW_THRESHOLD", "20일 임계값 미달", f"성공확률 {p_success * 100:.1f}%가 통과 기준 {threshold * 100:.1f}%보다 낮습니다.")
    elif p_success < 0.45:
        add("P_SUCCESS_LOW", "20일 성공확률 낮음", f"성공확률 {p_success * 100:.1f}%가 방어 기준 45.0%보다 낮습니다.", "warn")
    if stop_risk is not None and stop_risk >= 0.55:
        add("STOP_RISK_HIGH", "20일 손절위험 높음", f"보정 손절위험이 {stop_risk * 100:.1f}%로 높습니다.")
    if expected_r is not None and expected_r < 0.20:
        add("EXPECTED_R_LOW", "20일 기대값 낮음", f"기대값이 {expected_r:.2f}R로 최소 확인 기준 0.20R보다 낮습니다.")
    if forecast_return is not None and forecast_return < -3.0:
        add("FORECAST_RETURN_NEGATIVE", "20일 예측수익률 약세", f"20일 예측수익률이 {forecast_return:.1f}%입니다.")
    if not _truthy(prediction.get("decision_support_allowed")) and not _truthy(prediction.get("paper_decision_support_allowed")):
        add("MODEL_DISPLAY_ONLY", "모델 표시 전용", "예측 모델이 아직 판단/가상 실행 게이트를 통과하지 못했습니다.", "warn")
    return reasons


def _plain_warning(row: dict[str, Any], prediction: dict[str, Any]) -> str:
    warnings: list[str] = []
    stop_risk = _ratio_value(prediction.get("p_stop_hit_calibrated_20d", prediction.get("p_stop_hit_20d")))
    expected_r = _as_float(prediction.get("expected_r_20d"))
    raw_warning = str(row.get("warning_reasons") or row.get("block_reason") or prediction.get("block_reasons") or "")
    if stop_risk is not None and stop_risk >= 0.55:
        warnings.append("손절 위험이 큼")
    if expected_r is not None and expected_r < 0.35:
        warnings.append("기대값이 낮음")
    if "BETA" in raw_warning:
        warnings.append("시장 민감도 경고")
    if "MAX_OPEN_POSITIONS" in raw_warning:
        warnings.append("열린 후보 수 한도 초과")
    if "REQUESTED_WEIGHT_ZERO" in raw_warning or "NO_NEW_RISK" in raw_warning:
        warnings.append("새 매수 금지")
    return " · ".join(dict.fromkeys(warnings)) if warnings else "손절 기준을 지키며 확인"


def _twenty_day_signal_profile(
    signal: dict[str, Any],
    prediction: dict[str, Any],
    portfolio: dict[str, Any],
    intent: dict[str, Any],
    close_forecast_20d: dict[str, Any] | None = None,
) -> dict[str, Any]:
    p_success = _ratio_value(prediction.get("p_success_20d"))
    threshold = _ratio_value(prediction.get("threshold_20d"))
    stop_risk = _ratio_value(prediction.get("p_stop_hit_calibrated_20d", prediction.get("p_stop_hit_20d")))
    expected_r = _as_float(prediction.get("expected_r_20d"))
    forecast_return = _as_float((close_forecast_20d or {}).get("predicted_return_pct"))
    decision_score = _as_float(prediction.get("decision_score_20d"))
    paper_score = _as_float(prediction.get("paper_decision_score_20d"))
    approved_weight = _ratio_value(portfolio.get("approved_weight")) or 0.0
    block_reasons = _new_buy_block_reasons(signal, prediction, portfolio, intent, close_forecast_20d)
    blocking_codes = {row["code"] for row in block_reasons if row.get("severity") == "block"}

    threshold_ok = threshold is None or (p_success is not None and p_success >= threshold)
    forecast_ok = forecast_return is None or forecast_return >= 0.0
    risk_ok = stop_risk is None or stop_risk < 0.55
    expected_ok = expected_r is None or expected_r >= 0.20
    positive = (
        p_success is not None
        and threshold_ok
        and p_success >= 0.55
        and (expected_r is None or expected_r >= 0.35)
        and (stop_risk is None or stop_risk < 0.45)
        and (forecast_return is None or forecast_return >= 3.0)
        and not blocking_codes.intersection({"REQUESTED_WEIGHT_ZERO", "NO_NEW_RISK", "PORTFOLIO_REJECTED", "P_SUCCESS_BELOW_THRESHOLD", "STOP_RISK_HIGH", "EXPECTED_R_LOW", "FORECAST_RETURN_NEGATIVE"})
    )
    defensive = (
        p_success is None
        or not threshold_ok
        or not risk_ok
        or not expected_ok
        or not forecast_ok
        or bool(blocking_codes.intersection({"REQUESTED_WEIGHT_ZERO", "NO_NEW_RISK", "PORTFOLIO_REJECTED", "RULE_AVOID_OR_WAIT", "DO_NOT_CHASE"}))
    )

    if positive:
        tier = "positive"
        label = "실행후보"
        action_level = "execute_candidate"
        headline = "20일 예측 우위, 비중 확인"
        summary = "성공확률·손절위험·기대값이 신규 진입 조건을 통과"
    elif defensive:
        tier = "defensive"
        label = "매수금지"
        action_level = "avoid"
        headline = "새 매수 금지"
        summary = block_reasons[0]["label"] if block_reasons else "20일 예측 기준이 신규 진입에 불리"
    else:
        tier = "neutral"
        label = "관찰"
        action_level = "watch"
        headline = "조건 확인 후 대기"
        summary = "20일 예측이 강한 매수 우위나 명확한 방어 신호는 아님"

    metrics = {
        "p_success_20d": p_success,
        "threshold_20d": threshold,
        "p_stop_hit_20d": stop_risk,
        "expected_r_20d": expected_r,
        "forecast_return_20d": forecast_return,
        "decision_score_20d": decision_score,
        "paper_decision_score_20d": paper_score,
        "approved_weight": approved_weight,
    }
    return {
        "basis": "20d_prediction",
        "tier": tier,
        "tier_label": label,
        "action_level": action_level,
        "headline": headline,
        "summary": summary,
        "new_buy_allowed": tier == "positive",
        "new_buy_block_reasons": block_reasons,
        "metrics": metrics,
    }


def _prediction_use_status(prediction: dict[str, Any]) -> str:
    if _truthy(prediction.get("decision_support_allowed")):
        return "판단 가능"
    if _truthy(prediction.get("paper_decision_support_allowed")):
        return "가상 실행 가능"
    if not prediction:
        return "예측 없음"
    return "표시 전용"


def _prediction_confidence(prediction: dict[str, Any], horizon: int = 20) -> str:
    if not prediction:
        return "낮음"
    if _truthy(prediction.get("decision_support_allowed")):
        return "높음"
    if _truthy(prediction.get("paper_decision_support_allowed")):
        return "보통"
    score = _as_float(prediction.get(f"paper_decision_score_{horizon}d", prediction.get("paper_decision_score_20d")))
    stop_risk = _ratio_value(
        prediction.get(
            f"p_stop_hit_calibrated_{horizon}d",
            prediction.get(f"p_stop_hit_{horizon}d", prediction.get("p_stop_hit_calibrated_20d", prediction.get("p_stop_hit_20d"))),
        )
    )
    if score is not None and score >= 0.15 and (stop_risk is None or stop_risk < 0.55):
        return "보통"
    return "낮음"


def _prediction_payload(prediction: dict[str, Any], horizon: int) -> dict[str, Any]:
    calibrated_stop = _ratio_value(
        prediction.get(f"p_stop_hit_calibrated_{horizon}d", prediction.get(f"p_stop_hit_{horizon}d"))
    )
    return {
        "horizon": f"{horizon}거래일",
        "probability_label": "성공 확률",
        "probability_kind": "meta_label_success",
        "probability_definition": "손절 회피와 비용 차감 후 순수익 기준의 메타라벨 성공확률",
        "up_probability": _ratio_value(prediction.get(f"p_success_{horizon}d")),
        "down_risk": calibrated_stop,
        "down_risk_raw": _ratio_value(prediction.get(f"p_stop_hit_raw_{horizon}d")),
        "down_risk_calibrated": calibrated_stop,
        "down_risk_raw_minus_calibrated": _ratio_value(prediction.get(f"p_stop_hit_raw_minus_calibrated_{horizon}d")),
        "down_risk_oos_percentile": _ratio_value(prediction.get(f"p_stop_hit_oos_percentile_{horizon}d")),
        "strict_stop_risk_gap": _ratio_value(prediction.get(f"strict_stop_risk_gap_{horizon}d")),
        "paper_stop_risk_gap": _ratio_value(prediction.get(f"paper_stop_risk_gap_{horizon}d")),
        "stop_risk_calibration_warning": prediction.get("stop_risk_calibration_warning"),
        "expected_r": _as_float(prediction.get(f"expected_r_{horizon}d")),
        "threshold": _ratio_value(prediction.get(f"threshold_{horizon}d")),
        "confidence": "낮음" if _as_float(prediction.get(f"p_success_{horizon}d")) is None else _prediction_confidence(prediction, horizon),
        "use_status": _prediction_use_status(prediction),
    }


def _next_day_prediction_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    if not snapshot:
        return {}
    source = _normalize_next_day_prediction_snapshot_fields(snapshot)
    p_up = _ratio_value(source.get("p_up"))
    if p_up is None:
        return {}
    p_down = _ratio_value(source.get("p_down"))
    quality_pass = _truthy(source.get("prediction_quality_pass"))
    return {
        "horizon": "내일",
        "horizon_days": 1,
        "probability_label": "상승 확률",
        "probability_kind": "next_close_up",
        "probability_definition": "다음 거래일 종가가 오늘 종가보다 높을 확률",
        "up_probability": p_up,
        "down_probability": p_down,
        "down_risk": p_down,
        "down_risk_raw": None,
        "down_risk_calibrated": p_down,
        "down_risk_oos_percentile": None,
        "strict_stop_risk_gap": None,
        "paper_stop_risk_gap": None,
        "expected_r": None,
        "expected_net_return_pct": _as_float(source.get("expectancy_improvement_pct")),
        "threshold": _ratio_value(source.get("threshold")),
        "confidence": str(source.get("confidence_band") or ("보통" if quality_pass else "낮음")),
        "use_status": "판단 가능" if quality_pass else "표시 전용",
        "model_name": source.get("best_model"),
        "asof_date": source.get("asof_date"),
        "model_quality_status": source.get("model_quality_status"),
        "quality_block_reasons": source.get("model_quality_block_reasons"),
        "prediction_signal_status": source.get("prediction_signal_status"),
        "oos_event_count": _as_float(source.get("oos_event_count")),
        "selected_oos_event_count": _as_float(source.get("selected_oos_event_count")),
        "is_main_model": True,
    }


def _next_day_prediction_snapshot_for_symbol(universe_next_day: pd.DataFrame, symbol: str) -> dict[str, Any]:
    symbol_key = str(symbol).strip().upper()
    candidates: list[dict[str, Any]] = []
    aggregate = _latest_symbol_record(universe_next_day, symbol_key, ("next_day_prediction_asof_date", "date", "asof_date"))
    if aggregate:
        candidates.append(aggregate)
    per_symbol = field_value_map(RULE_DIR / "universe" / symbol_key / "tsm_next_day_up_latest_snapshot.csv")
    if per_symbol:
        per_symbol["symbol"] = symbol_key
        candidates.append(per_symbol)
    if symbol_key == "TSM":
        root = field_value_map(RULE_DIR / "tsm_next_day_up_latest_snapshot.csv") or field_value_map(RULE_DIR / "tsm_latest_prediction_snapshot.csv")
        if root:
            root["symbol"] = symbol_key
            candidates.append(root)
    if not candidates:
        return {}

    def sort_date(row: dict[str, Any]) -> date:
        parsed = _date_value(row.get("next_day_prediction_asof_date") or row.get("date") or row.get("asof_date"))
        return parsed or date.min

    return max(candidates, key=sort_date)


def _next_close_forecast_snapshot_for_symbol(universe_next_close: pd.DataFrame, symbol: str) -> dict[str, Any]:
    symbol_key = str(symbol).strip().upper()
    candidates: list[dict[str, Any]] = []
    aggregate = _latest_symbol_record(universe_next_close, symbol_key, ("next_close_prediction_asof_date", "date", "asof_date"))
    if aggregate:
        candidates.append(aggregate)
    if symbol_key == "TSM":
        root = field_value_map(RULE_DIR / "tsm_next_close_latest_snapshot.csv")
        if root:
            root["symbol"] = symbol_key
            candidates.append(root)
    if not candidates:
        return {}

    def sort_date(row: dict[str, Any]) -> date:
        parsed = _date_value(row.get("next_close_prediction_asof_date") or row.get("date") or row.get("asof_date"))
        return parsed or date.min

    return max(candidates, key=sort_date)


def _plausible_native_price(native: float | None, usd: float | None) -> bool:
    if native is None or native <= 0:
        return False
    if usd is None or usd <= 0:
        return True
    return native > usd * 20


def _next_close_forecast_payload(
    snapshot: dict[str, Any],
    horizon: int,
    currency: str = "USD",
    current_display_price: Any = None,
    fx_rate_to_usd: Any = None,
) -> dict[str, Any]:
    if not snapshot:
        return {}
    suffix = f"{int(horizon)}d"
    predicted_return_pct = _as_float(snapshot.get(f"next_close_predicted_return_pct_{suffix}"))
    raw_predicted_close = _as_float(snapshot.get(f"next_close_predicted_close_{suffix}"))
    predicted_close = raw_predicted_close
    if predicted_close is None:
        predicted_close = _as_float(snapshot.get(f"next_close_predicted_close_engine_{suffix}"))
    predicted_close_usd = _as_float(snapshot.get(f"next_close_predicted_close_usd_{suffix}"))
    predicted_close_native = _as_float(snapshot.get(f"next_close_predicted_close_native_{suffix}"))
    if str(currency).upper() == "KRW":
        current_price = _as_float(current_display_price)
        engine_prediction = predicted_close_usd or _as_float(snapshot.get(f"next_close_predicted_close_engine_{suffix}"))
        converted_prediction = _display_price_from_engine(engine_prediction, "KRW", fx_rate_to_usd)
        if converted_prediction is not None:
            predicted_close = converted_prediction
        elif current_price is not None and predicted_return_pct is not None:
            predicted_close = current_price * (1.0 + predicted_return_pct / 100.0)
        elif _plausible_native_price(predicted_close_native, predicted_close_usd):
            predicted_close = predicted_close_native
        else:
            predicted_close = _display_price_from_engine(predicted_close, "KRW", fx_rate_to_usd)
    if predicted_close is None:
        return {}

    def display_interval(side: str) -> float | None:
        raw_value = snapshot.get(f"next_close_{side}_80_{suffix}")
        parsed = _as_float(raw_value)
        if parsed is None:
            return None
        if str(currency).upper() != "KRW":
            return parsed
        engine_value = _as_float(snapshot.get(f"next_close_{side}_80_engine_{suffix}"))
        converted = _display_price_from_engine(engine_value, "KRW", fx_rate_to_usd)
        if converted is not None:
            return converted
        if raw_predicted_close and raw_predicted_close > 0 and predicted_close and predicted_close > 0:
            ratio = parsed / raw_predicted_close
            if 0 < ratio < 5:
                return predicted_close * ratio
        return _display_price_from_engine(parsed, "KRW", fx_rate_to_usd)

    return {
        "horizon": suffix.upper(),
        "horizon_days": int(horizon),
        "predicted_close": predicted_close,
        "predicted_close_engine": _as_float(snapshot.get(f"next_close_predicted_close_engine_{suffix}")),
        "predicted_close_usd": predicted_close_usd,
        "predicted_close_native": predicted_close_native,
        "predicted_return_pct": predicted_return_pct,
        "lower_80": display_interval("lower"),
        "upper_80": display_interval("upper"),
        "model_name": snapshot.get(f"next_close_best_model_{suffix}"),
        "asof_date": snapshot.get("next_close_prediction_asof_date") or snapshot.get("date"),
        "model_quality_status": snapshot.get("next_close_model_quality_status"),
        "quality_pass": _truthy(snapshot.get(f"next_close_performance_quality_pass_{suffix}")),
        "quality_block_reasons": snapshot.get(f"next_close_quality_block_reasons_{suffix}"),
        "oos_event_count": _as_float(snapshot.get(f"next_close_oos_event_count_{suffix}")),
        "interval_coverage_80": _as_float(snapshot.get(f"next_close_interval_coverage_80_{suffix}")),
    }


def _scenario_for_signal(signal: dict[str, Any], action_level: str) -> dict[str, str]:
    trigger = str(signal.get("entry_trigger") or signal.get("raw_entry_event") or "").upper()
    decision = str(signal.get("decision_tier") or signal.get("suggested_action") or signal.get("trade_action") or "").upper()
    if action_level == "execute_candidate":
        rise_if = "신호가 유지되고 손절 기준을 지키면 상승 추적"
    elif "BREAKOUT" in trigger or "BREAKOUT" in decision:
        rise_if = "20/60일 고점 돌파가 유지될 때"
    elif "WATCH" in decision:
        rise_if = "돌파 또는 눌림목 재진입이 확인될 때"
    else:
        rise_if = "새 상승 신호가 다시 생길 때"
    if "REDUCE" in decision:
        fall_if = "과열 구간에서 가격이 밀리면 하락 위험 확대"
    elif "AVOID" in decision:
        fall_if = "추세 회복 전 변동성이 커지면 하락 위험 확대"
    else:
        fall_if = "손절 기준가 이탈 또는 급격한 변동성 확대"
    return {
        "rise_if": rise_if,
        "fall_if": fall_if,
        "next_check": _translate_next_check(signal.get("next_check_condition")),
    }


def _action_from_rows(
    signal: dict[str, Any],
    prediction: dict[str, Any],
    portfolio: dict[str, Any],
    intent: dict[str, Any],
    close_forecast_20d: dict[str, Any] | None = None,
) -> tuple[str, str, str, dict[str, Any]]:
    profile = _twenty_day_signal_profile(signal, prediction, portfolio, intent, close_forecast_20d)
    return profile["action_level"], profile["headline"], profile["summary"], profile


def _system_status_for_investor(symbols: list[dict[str, Any]], market_data: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    latest_dates = [_date_value(row.get("as_of")) for row in symbols]
    latest_dates = [value for value in latest_dates if value is not None]
    today = date.today()
    max_age = max((today - value).days for value in latest_dates) if latest_dates else None
    daily_market = market_data[market_data.get("bar_type", pd.Series(dtype=str)).astype(str).eq("daily")] if not market_data.empty and "bar_type" in market_data.columns else market_data
    non_ok = int((~daily_market.get("status", pd.Series(dtype=str)).astype(str).str.upper().eq("OK")).sum()) if not daily_market.empty and "status" in daily_market.columns else 0
    if max_age is None:
        data_status = "주의"
    elif max_age > 7:
        data_status = "오래됨"
    elif max_age > 3 or non_ok > 0:
        data_status = "주의"
    else:
        data_status = "정상"

    model_quality = quality_summary(RULE_DIR / "tsm_pooled_model_quality_checks.csv", critical_only=True)
    if predictions.empty:
        model_status = "차단"
    elif any(row.get("prediction", {}).get("use_status") == "판단 가능" for row in symbols):
        model_status = "판단 가능"
    elif model_quality.get("failed", 0):
        model_status = "표시 전용"
    else:
        model_status = "표시 전용"

    return {
        "data_status": data_status,
        "model_status": model_status,
        "main_model": "20일 예측",
        "live_trading_status": "DISABLED_BY_DESIGN",
        "latest_data_date": max(value.isoformat() for value in latest_dates) if latest_dates else None,
        "max_data_age_days": max_age,
        "daily_data_issue_count": non_ok,
        "model_quality_failed": model_quality.get("failed", 0),
    }


def build_investor_dashboard() -> dict[str, Any]:
    members = _decision_dashboard_members()
    signals = ensure_v2_signal_columns(safe_read_csv(RULE_DIR / "tsm_universe_latest_signals.csv"))
    predictions = safe_read_csv(RULE_DIR / "tsm_universe_latest_predictions.csv")
    next_day_predictions = safe_read_csv(RULE_DIR / "tsm_next_day_up_universe_latest_predictions.csv")
    next_close_predictions = safe_read_csv(RULE_DIR / "tsm_next_close_universe_latest_predictions.csv")
    portfolio_decisions = safe_read_csv(RULE_DIR / "tsm_portfolio_risk_order_decisions.csv")
    intents = safe_read_csv(RULE_DIR / "tsm_order_intents.csv")
    market_data = _investor_market_data_latest()

    symbols: list[dict[str, Any]] = []
    for member in members:
        symbol = str(member.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        signal = _freshest_symbol_record(
            signals,
            symbol,
            RULE_DIR / "universe" / symbol / "tsm_daily_algorithmic_signals.csv",
            ("date", "asof_date"),
        )
        prediction = _latest_symbol_record(predictions, symbol, ("date", "prediction_asof_date"))
        portfolio = _latest_symbol_record(portfolio_decisions, symbol, ("asof_date", "date"))
        intent = _latest_symbol_record(intents, symbol, ("asof_date", "date"))
        daily_market = pd.DataFrame()
        if not market_data.empty and "symbol" in market_data.columns:
            daily_market = market_data[
                (market_data["symbol"].astype(str).str.upper() == symbol)
                & (
                    market_data.get("bar_type", pd.Series("", index=market_data.index)).astype(str).eq("daily")
                    if "bar_type" in market_data.columns
                    else True
                )
            ].copy()
        market = clean_json(daily_market.iloc[-1].to_dict()) if not daily_market.empty else {}
        currency = _display_currency_for(member, signal, market)
        fx_rate = _display_fx_rate_for_investor(currency, signal, market, member)
        price = _latest_display_price(signal, market, currency, fx_rate)
        stop_price = _display_price_from_engine(
            _first_text(signal.get("stop_price_1_8atr"), signal.get("atr_stop_2x"), portfolio.get("stop_price_1_8atr"), default=""),
            currency,
            fx_rate,
        )
        target_price = _display_price_from_engine(_first_text(signal.get("take_profit_2R"), default=""), currency, fx_rate)
        pred5 = _prediction_payload(prediction, 5)
        pred20 = _prediction_payload(prediction, 20)
        pred60 = _prediction_payload(prediction, 60)
        pred1 = _next_day_prediction_payload(_next_day_prediction_snapshot_for_symbol(next_day_predictions, symbol))
        close_forecast_snapshot = _next_close_forecast_snapshot_for_symbol(next_close_predictions, symbol)
        close_forecasts = {
            "1d": _next_close_forecast_payload(close_forecast_snapshot, 1, currency, price, fx_rate),
            "5d": _next_close_forecast_payload(close_forecast_snapshot, 5, currency, price, fx_rate),
            "20d": _next_close_forecast_payload(close_forecast_snapshot, 20, currency, price, fx_rate),
        }
        action_level, action_text, main_reason, signal_profile = _action_from_rows(
            signal,
            prediction,
            portfolio,
            intent,
            close_forecasts.get("20d"),
        )
        main_prediction = pred1 or pred5 or pred20
        as_of = _freshest_date_text(prediction.get("date"), signal.get("date"), market.get("end_timestamp"), market.get("generated_at_utc"))
        as_of_date = _date_value(as_of)
        symbol_payload = {
            "symbol": symbol,
            "name": symbol_display_name(symbol),
            "group": str(member.get("symbol_group") or signal.get("symbol_group") or prediction.get("symbol_group") or ""),
            "group_label": SYMBOL_GROUP_LABELS.get(str(member.get("symbol_group") or signal.get("symbol_group") or prediction.get("symbol_group") or ""), str(member.get("symbol_group") or "")),
            "price": price,
            "currency": currency,
            "as_of": as_of_date.isoformat() if as_of_date else str(as_of)[:10],
            "signal_as_of": str(signal.get("date", ""))[:10],
            "prediction_as_of": str(prediction.get("date", ""))[:10],
            "main_prediction_as_of": str(main_prediction.get("asof_date") or prediction.get("date", ""))[:10],
            "action_level": action_level,
            "action_text": action_text,
            "main_reason": main_reason,
            "warning": _plain_warning(portfolio or intent or signal, prediction),
            "signal_profile": signal_profile,
            "new_buy_block_reasons": signal_profile.get("new_buy_block_reasons", []),
            "prediction": main_prediction,
            "predictions": {
                "1d": pred1,
                "5d": pred5,
                "20d": pred20,
                "60d": pred60,
            },
            "close_forecasts": close_forecasts,
            "scenario": _scenario_for_signal(signal, action_level),
            "risk": {
                "approved_weight": _ratio_value(portfolio.get("approved_weight")) or 0.0,
                "suggested_weight": _ratio_value(signal.get("suggested_weight") or portfolio.get("suggested_weight")) or 0.0,
                "requested_weight": _ratio_value(portfolio.get("requested_weight") or intent.get("target_weight")) or 0.0,
                "rank": _as_float(portfolio.get("rank")),
                "stop_price": stop_price,
                "target_price": target_price,
                "risk_pct": _ratio_value(signal.get("risk_pct_2atr")),
            },
            "scores": {
                "rule_score": _as_float(signal.get("score_price_algo_total")),
                "research_score": _as_float(signal.get("research_signal_score")),
                "decision_score": _as_float(prediction.get("decision_score_20d")),
                "paper_decision_score": _as_float(prediction.get("paper_decision_score_20d")),
            },
            "raw_status": {
                "portfolio_status": portfolio.get("portfolio_status"),
                "intent_status": intent.get("status"),
                "live_trading_status": _first_text(portfolio.get("live_trading_status"), intent.get("live_trading_status"), prediction.get("live_trading_status"), default="DISABLED_BY_DESIGN"),
            },
            "price_series": _price_series_for_symbol(symbol, currency, fx_rate),
        }
        symbols.append(clean_json(symbol_payload))

    action_candidates = []
    for row in symbols:
        priority = INVESTOR_ACTION_PRIORITY.get(str(row.get("action_level")), 9)
        approved_weight = _ratio_value(row.get("risk", {}).get("approved_weight")) or 0.0
        portfolio_rank = _as_float(row.get("risk", {}).get("rank")) or 9999.0
        paper_score = _as_float(row.get("scores", {}).get("paper_decision_score")) or -999.0
        action_candidates.append((priority, portfolio_rank, -approved_weight, -paper_score, row))
    action_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], str(item[4].get("symbol"))))
    action_queue = [
        {
            "symbol": row["symbol"],
            "name": row["name"],
            "rank": idx + 1,
            "action": row["action_text"],
            "action_level": row["action_level"],
            "max_weight": row["risk"]["approved_weight"],
            "main_reason": row["main_reason"],
            "warning": row["warning"],
            "signal_profile": row.get("signal_profile", {}),
            "new_buy_block_reasons": row.get("new_buy_block_reasons", []),
        }
        for idx, (_, _, _, _, row) in enumerate(action_candidates)
    ]

    return clean_json(
        {
            "generated_at": now_iso(),
            "fx": _latest_fx_context(),
            "system": {
                **_system_status_for_investor(symbols, market_data, predictions),
                "daily_update": build_daily_update_system_summary(
                    decision_config=safe_read_csv(RULE_DIR / "tsm_decision_universe_config.csv"),
                    market_data=market_data,
                    predictions=predictions,
                    portfolio_decisions=portfolio_decisions,
                    paper_orders=safe_read_csv(RULE_DIR / "tsm_paper_orders.csv"),
                    paper_fills=safe_read_csv(RULE_DIR / "tsm_paper_fills.csv"),
                    paper_positions=safe_read_csv(RULE_DIR / "tsm_paper_positions.csv"),
                ),
            },
            "action_queue": action_queue,
            "symbols": symbols,
            "run": get_run_state(),
        }
    )


# ---------------------------------------------------------------------------
# Manual portfolio execution cockpit (broker-free).
#
# The user records their own buy/sell transactions; we reuse the existing paper
# position math + model outputs (signals, risk, portfolio-risk approved_weight,
# predictions) to produce holdings, weights, returns and rebalance execution
# tickets. Values are kept in engine USD; the client converts to KRW using the
# supplied fx context (KRW = usd * usdkrw). No live broker is ever called.
# ---------------------------------------------------------------------------

PORTFOLIO_DIR = OUTPUT_DIR / "portfolio"
MANUAL_TRANSACTIONS_PATH = PORTFOLIO_DIR / "manual_transactions.csv"
MANUAL_CASH_PATH = PORTFOLIO_DIR / "manual_cash_ledger.csv"


def _run_manual_portfolio_engine() -> dict[str, pd.DataFrame]:
    import tsm_manual_portfolio_engine as mpe

    return mpe.run_manual_portfolio(ROOT, ROOT / "config" / "tsm_research.toml", PORTFOLIO_DIR)


def _holding_return_pct(avg: float | None, price: float | None) -> float | None:
    if avg and avg > 0 and price is not None:
        return (price - avg) / avg * 100.0
    return None


def _overlay_live_portfolio(
    holdings: list[dict[str, Any]],
    totals: dict[str, Any],
    usdkrw: float | None,
    rebalance_queue: list[dict[str, Any]] | None = None,
) -> str | None:
    """Recompute holding prices/values/weights/P&L and totals from live quotes so
    the portfolio tracks intraday prices. Definitions match the engine exactly
    (equity = market_value + cash; return = (mv - cost)/cost; weight = mv/equity).

    The rebalance tickets' *price-derived arithmetic* is also re-priced live
    (current_weight, weight_gap, ref_price_usd, target/delta shares, delta
    notional) using the same formulas as ``build_rebalance_tickets``. The
    *model decision* fields stay fixed at the last close because they require
    re-running the model/risk pipeline: action/action_level, target_weight,
    stop/target price, p_success, reason, warning, current_shares (held qty).

    Returns the live as-of timestamp, or None if no live quote could be applied
    (fall back to daily)."""
    try:
        import tsm_live_quote_engine

        live = tsm_live_quote_engine.fetch_live_quotes()
    except Exception:
        return None
    quotes = (live or {}).get("quotes") or {}
    if not quotes:
        return None

    def _to_usd(native: float | None, region: str) -> float | None:
        if native is None:
            return None
        if region == "KR" and usdkrw:
            return native / usdkrw
        return native

    applied = False
    daily_pnl = 0.0
    for h in holdings:
        q = quotes.get(str(h.get("symbol")))
        if not q:
            continue
        region = str(q.get("region") or "US")
        price_usd = _to_usd(_as_float(q.get("price")), region)
        if price_usd is None:
            continue
        shares = _as_float(h.get("shares")) or 0.0
        avg = _as_float(h.get("avg_price_usd"))
        h["current_price_usd"] = price_usd
        h["market_value_usd"] = price_usd * shares
        if avg is not None:
            h["unrealized_pnl_usd"] = (price_usd - avg) * shares
        h["return_pct"] = _holding_return_pct(avg, price_usd)
        h["intraday_pct"] = _as_float(q.get("change_pct"))
        h["live"] = True
        prev_usd = _to_usd(_as_float(q.get("prev_close")), region)
        if prev_usd is not None:
            daily_pnl += (price_usd - prev_usd) * shares
        applied = True
    if not applied:
        return None

    mv_total = sum(_as_float(h.get("market_value_usd")) or 0.0 for h in holdings)
    cash = _as_float(totals.get("cash_usd")) or 0.0
    cost = _as_float(totals.get("cost_basis_usd")) or 0.0
    equity = mv_total + cash
    for h in holdings:
        h["weight_pct"] = ((_as_float(h.get("market_value_usd")) or 0.0) / equity * 100.0) if equity else 0.0
    holdings.sort(key=lambda h: h.get("market_value_usd") or 0.0, reverse=True)
    totals["market_value_usd"] = mv_total
    totals["equity_usd"] = equity
    totals["total_return_pct"] = ((mv_total - cost) / cost * 100.0) if cost > 0 else 0.0
    totals["daily_pnl_usd"] = daily_pnl

    # Re-price the rebalance tickets' price-derived arithmetic from the same live
    # quotes / live equity. Formulas mirror build_rebalance_tickets exactly; only
    # current_shares (held qty) and the model decision fields stay fixed.
    if rebalance_queue:
        mv_by_symbol = {str(h.get("symbol")): (_as_float(h.get("market_value_usd")) or 0.0) for h in holdings}
        for t in rebalance_queue:
            q = quotes.get(str(t.get("symbol")))
            if not q:
                continue
            price = _to_usd(_as_float(q.get("price")), str(q.get("region") or "US"))
            if price is None or price <= 0:
                continue
            target_weight = _as_float(t.get("target_weight")) or 0.0
            current_shares = _as_float(t.get("current_shares")) or 0.0  # held qty — fixed
            mv = mv_by_symbol.get(str(t.get("symbol")), price * current_shares)
            current_weight = (mv / equity) if equity else 0.0
            # EXIT/SELL liquidates the whole position; others close the weight gap.
            if str(t.get("action") or "").upper() == "EXIT" or str(t.get("action_level") or "").upper() == "SELL":
                target_shares, delta_shares = 0.0, -current_shares
            else:
                target_shares = max((target_weight * equity) / price, 0.0) if equity > 0 else 0.0
                delta_shares = target_shares - current_shares
            t["current_weight"] = current_weight
            t["weight_gap"] = target_weight - current_weight
            t["target_shares"] = max(target_shares, 0.0)
            t["delta_shares"] = delta_shares
            t["delta_notional_usd"] = delta_shares * price
            t["ref_price_usd"] = price
            t["live"] = True

    return (live or {}).get("generated_at")


def build_manual_portfolio_dashboard() -> dict[str, Any]:
    result = _run_manual_portfolio_engine()
    positions = result["positions"]
    tickets = result["tickets"]
    snapshot = result["snapshot"]
    transactions = result["transactions"]
    reconciliation = result["reconciliation"]

    fx_ctx = _latest_fx_context()
    snap = snapshot.iloc[0].to_dict() if not snapshot.empty else {}
    usdkrw = _as_float(snap.get("usdkrw")) or _as_float(fx_ctx.get("usdkrw"))
    fx_rate = _as_float(snap.get("fx_rate_to_usd")) or _as_float(fx_ctx.get("fx_rate_to_usd"))

    ticket_by_symbol: dict[str, dict[str, Any]] = {}
    if not tickets.empty:
        for _, t in tickets.iterrows():
            ticket_by_symbol[str(t.get("symbol"))] = t.to_dict()

    holdings: list[dict[str, Any]] = []
    for _, p in positions.iterrows():
        sym = str(p.get("symbol", ""))
        if sym.upper() == "CASH":
            continue
        qty = _as_float(p.get("quantity")) or 0.0
        if qty <= 1e-9:
            continue
        avg = _as_float(p.get("average_price"))
        price = _as_float(p.get("market_price"))
        t = ticket_by_symbol.get(sym, {})
        holdings.append(
            {
                "symbol": sym,
                "name": symbol_display_name(sym),
                "shares": qty,
                "avg_price_usd": avg,
                "current_price_usd": price,
                "market_value_usd": _as_float(p.get("market_value")),
                "weight_pct": (_as_float(p.get("weight")) or 0.0) * 100.0,
                "unrealized_pnl_usd": _as_float(p.get("unrealized_pnl")),
                "realized_pnl_usd": _as_float(p.get("realized_pnl")),
                "return_pct": _holding_return_pct(avg, price),
                "recommendation": t.get("action"),
                "action_level": t.get("action_level"),
                "delta_shares": _as_float(t.get("delta_shares")),
                "target_weight_pct": (_as_float(t.get("target_weight")) or 0.0) * 100.0,
                "stop_usd": _as_float(t.get("stop_price_usd")),
                "target_usd": _as_float(t.get("target_price_usd")),
                "p_success_20d": _as_float(t.get("p_success_20d")),
                "reason": t.get("reason"),
                "warning": t.get("warning"),
                "data_status": t.get("data_status"),
            }
        )
    holdings.sort(key=lambda h: h.get("market_value_usd") or 0.0, reverse=True)

    rebalance_queue = [r for r in (clean_json(rec) for rec in tickets.to_dict("records"))]
    recon_status = "PASS"
    if not reconciliation.empty and "status" in reconciliation.columns:
        recon_status = "PASS" if reconciliation["status"].astype(str).eq("PASS").all() else "FAIL"

    totals = {
        "equity_usd": _as_float(snap.get("equity_usd")) or 0.0,
        "cost_basis_usd": _as_float(snap.get("cost_basis_usd")) or 0.0,
        "market_value_usd": _as_float(snap.get("market_value_usd")) or 0.0,
        "cash_usd": _as_float(snap.get("cash_usd")) or 0.0,
        "total_return_pct": _as_float(snap.get("total_return_pct")) or 0.0,
        "daily_pnl_usd": _as_float(snap.get("daily_pnl_usd")) or 0.0,
        "n_holdings": int(_as_float(snap.get("n_holdings")) or 0),
    }

    # Real-time overlay: re-price holdings/totals AND the tickets' price-derived
    # arithmetic from live quotes (live USD/KRW already resolved above). Only the
    # model decision fields on tickets keep the engine's daily computation.
    live_usdkrw = usdkrw
    try:
        live_fx = build_live_quotes().get("fx") or {}
        live_usdkrw = _as_float(live_fx.get("usdkrw")) or usdkrw
    except Exception:
        live_fx = {}
    live_as_of = _overlay_live_portfolio(holdings, totals, live_usdkrw, rebalance_queue)

    return clean_json(
        {
            "generated_at": now_iso(),
            "live_as_of": live_as_of,
            "currency_default": "KRW",
            "fx": {"fx_rate_to_usd": fx_rate, "usdkrw": live_usdkrw or usdkrw, "date": fx_ctx.get("date"), "fx_pair": fx_ctx.get("fx_pair")},
            "totals": totals,
            "holdings": holdings,
            "rebalance_queue": rebalance_queue,
            "transactions": [clean_json(rec) for rec in transactions.to_dict("records")],
            "reconciliation_status": recon_status,
            "universe": list(__import__("tsm_manual_portfolio_engine").DECISION_UNIVERSE),
            "live_trading_status": "DISABLED_BY_DESIGN",
        }
    )


def _read_manual_transactions() -> pd.DataFrame:
    import tsm_manual_portfolio_engine as mpe

    df = safe_read_csv(MANUAL_TRANSACTIONS_PATH)
    for col in mpe.TRANSACTION_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[mpe.TRANSACTION_COLUMNS].copy() if not df.empty else pd.DataFrame(columns=mpe.TRANSACTION_COLUMNS)


def _write_manual_transactions(df: pd.DataFrame) -> None:
    import tsm_manual_portfolio_engine as mpe

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MANUAL_TRANSACTIONS_PATH.with_suffix(".csv.tmp")
    df[mpe.TRANSACTION_COLUMNS].to_csv(tmp, index=False)
    tmp.replace(MANUAL_TRANSACTIONS_PATH)


def _validate_transaction(tx: dict[str, Any]) -> dict[str, Any]:
    import tsm_manual_portfolio_engine as mpe

    symbol = mpe.clean_symbol(tx.get("symbol"))
    if not symbol:
        raise ValueError("symbol is required")
    side = str(tx.get("side", "")).strip().upper()
    if side not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    qty = _as_float(tx.get("quantity"))
    if qty is None or qty <= 0:
        raise ValueError("quantity must be a positive number")
    price = _as_float(tx.get("price"))
    if price is None or price <= 0:
        raise ValueError("price must be a positive number")
    currency = str(tx.get("price_currency", "") or "").strip().upper() or "USD"
    if currency not in ("USD", "KRW", "NATIVE"):
        raise ValueError("price_currency must be USD, KRW or NATIVE")
    trade_date = str(tx.get("trade_date", "") or "").strip()
    if pd.isna(pd.to_datetime(trade_date, errors="coerce")):
        raise ValueError("trade_date must be a valid date (YYYY-MM-DD)")
    return {
        "transaction_id": str(tx.get("transaction_id") or "").strip(),
        "trade_date": trade_date[:10],
        "symbol": symbol,
        "side": side,
        "quantity": float(qty),
        "price": float(price),
        "price_currency": currency,
        "fees": float(_as_float(tx.get("fees")) or 0.0),
        "note": str(tx.get("note", "") or ""),
    }


def apply_portfolio_mutation(payload: dict[str, Any]) -> dict[str, Any]:
    import tsm_manual_portfolio_engine as mpe

    action = str(payload.get("action", "add")).strip().lower()
    df = _read_manual_transactions()

    if action == "delete":
        tx_id = str(payload.get("transaction_id") or (payload.get("transaction") or {}).get("transaction_id") or "").strip()
        if not tx_id:
            raise ValueError("transaction_id is required to delete")
        df = df[df["transaction_id"].astype(str) != tx_id].copy()
    elif action in ("add", "update"):
        tx = _validate_transaction(payload.get("transaction") or payload)
        if not tx["transaction_id"]:
            tx["transaction_id"] = mpe.stable_id(tx["symbol"], tx["side"], tx["trade_date"], tx["quantity"], tx["price"], now_iso(), prefix="mtx")
        tx["created_at_utc"] = now_iso()
        df = df[df["transaction_id"].astype(str) != tx["transaction_id"]].copy()
        df = pd.concat([df, pd.DataFrame([tx])], ignore_index=True)
    else:
        raise ValueError(f"unknown action: {action}")

    _write_manual_transactions(df)
    return build_manual_portfolio_dashboard()


def _rule_fallback_universe_prediction_row(member: pd.Series) -> dict[str, Any] | None:
    sym = str(member.get("symbol", "")).strip().upper()
    if not sym:
        return None

    signal = latest_row_map(RULE_DIR / "universe" / sym / "tsm_daily_algorithmic_signals.csv")
    risk = field_value_map(RULE_DIR / "universe" / sym / "tsm_latest_risk_snapshot.csv")
    if not signal and not risk:
        return None

    raw_score = _as_float(signal.get("score_price_algo_total") or risk.get("score_price_algo_total"))
    raw_research_score = _as_float(signal.get("research_signal_score"))
    score = _clamp_float(raw_score / 100, 0.0, 1.0) if raw_score is not None else None
    paper_score = (
        _clamp_float(raw_research_score / 100, 0.0, 1.0)
        if raw_research_score is not None
        else score
    )
    threshold = 0.65

    risk_pct = _as_float(risk.get("risk_pct_2atr") or signal.get("risk_pct_2atr"))
    if risk_pct is not None and abs(risk_pct) <= 1:
        risk_pct *= 100
    stop_hit_proxy = _clamp_float(risk_pct / 40, 0.0, 0.95) if risk_pct is not None else None
    expected_r = _clamp_float((score * 2) - 1, -1.0, 2.0) if score is not None else None

    entry_trigger = str(signal.get("entry_trigger") or risk.get("entry_trigger") or "").strip().upper()
    trade_action = str(signal.get("trade_action") or risk.get("trade_action") or "").strip().upper()
    strict_stage = str(signal.get("strict_signal_stage") or "").strip().upper()
    latest_trade_ready = (
        entry_trigger not in {"", "NONE", "NA", "N/A", "NULL"}
        and trade_action not in {"", "NONE", "NO_ACTION", "REDUCE_OR_DO_NOT_CHASE"}
        and "NO_ENTRY" not in strict_stage
    )

    risk_state = risk.get("risk_state") or signal.get("risk_state")
    risk_ok = _risk_allows_decision(risk_state)
    decision_allowed = False
    paper_allowed = False

    block_reasons = ["POOLED_PREDICTION_MISSING_RULE_FALLBACK", "DIAGNOSTIC_ONLY_NO_ACTIONABILITY"]
    if not latest_trade_ready:
        block_reasons.append("LATEST_NOT_TRADE_READY")
    if not risk_ok:
        risk_reason = str(risk_state or risk.get("limiting_reason") or "RISK_LIMITED").strip()
        block_reasons.append(f"RISK:{risk_reason}")
    if score is None:
        block_reasons.append("RULE_SCORE_MISSING")
    elif score < threshold:
        block_reasons.append("RULE_SCORE_BELOW_65")
    if stop_hit_proxy is not None and stop_hit_proxy > 0.35:
        block_reasons.append("STOP_RISK_PROXY_GT_0_35")

    row = {
        "symbol": sym,
        "symbol_group": str(member.get("symbol_group", signal.get("symbol_group", "")) or ""),
        "date": signal.get("date") or risk.get("date"),
        "latest_trade_ready": latest_trade_ready,
        "p_success_20d": score,
        "p_stop_hit_20d": stop_hit_proxy,
        "p_stop_hit_raw_20d": stop_hit_proxy,
        "p_stop_hit_calibrated_20d": stop_hit_proxy,
        "p_stop_hit_raw_minus_calibrated_20d": 0.0 if stop_hit_proxy is not None else None,
        "p_stop_hit_oos_percentile_20d": None,
        "strict_stop_risk_gap_20d": stop_hit_proxy - 0.35 if stop_hit_proxy is not None else None,
        "paper_stop_risk_gap_20d": stop_hit_proxy - 0.40 if stop_hit_proxy is not None else None,
        "stop_risk_calibration_warning": "RULE_FALLBACK_NO_STOP_CALIBRATION",
        "expected_r_20d": expected_r,
        "decision_score_20d": score,
        "paper_decision_score_20d": paper_score,
        "threshold_20d": threshold,
        "decision_support_allowed": decision_allowed,
        "paper_decision_support_allowed": paper_allowed,
        "block_reasons": "|".join(block_reasons),
        "model_name": "rule_score_fallback_no_pooled_prediction",
        "live_trading_status": "DISABLED_BY_DESIGN",
        "decision_scope": "top10",
        "training_scope": "universal_research_pool",
        "prediction_source": "rule_fallback_diagnostic_only",
    }
    return row


def rule_fallback_universe_prediction_diagnostics(
    predictions: pd.DataFrame,
    decision_config: pd.DataFrame,
) -> pd.DataFrame:
    """Diagnostic-only fallback rows for decision symbols missing pooled predictions."""
    members = _decision_config_or_default(decision_config)
    if members.empty or "symbol" not in members.columns:
        return pd.DataFrame(columns=UNIVERSE_LATEST_PREDICTION_COLUMNS)

    existing = predictions.copy() if not predictions.empty else pd.DataFrame(columns=UNIVERSE_LATEST_PREDICTION_COLUMNS)
    if "symbol" not in existing.columns:
        existing["symbol"] = pd.Series(dtype=str)

    existing_symbols = {str(sym).strip().upper() for sym in existing["symbol"].dropna().tolist()}
    fallback_rows: list[dict[str, Any]] = []
    for _, member in members.iterrows():
        sym = str(member.get("symbol", "")).strip().upper()
        if not sym or sym in existing_symbols:
            continue
        row = _rule_fallback_universe_prediction_row(member)
        if row:
            fallback_rows.append(row)

    diagnostics = pd.DataFrame(fallback_rows)
    if diagnostics.empty:
        return pd.DataFrame(columns=UNIVERSE_LATEST_PREDICTION_COLUMNS)

    for col in UNIVERSE_LATEST_PREDICTION_COLUMNS:
        if col not in diagnostics.columns:
            diagnostics[col] = None

    order = {
        str(row.get("symbol", "")).strip().upper(): idx
        for idx, row in members.reset_index(drop=True).iterrows()
        if str(row.get("symbol", "")).strip()
    }
    diagnostics["_decision_order"] = diagnostics["symbol"].astype(str).str.upper().map(order).fillna(len(order) + 999)
    diagnostics = diagnostics.sort_values(["_decision_order", "symbol"]).drop(columns=["_decision_order"])
    return diagnostics.reset_index(drop=True)


def with_rule_fallback_universe_latest_predictions(
    predictions: pd.DataFrame,
    decision_config: pd.DataFrame,
) -> pd.DataFrame:
    """Deprecated compatibility wrapper: actionability feeds must remain pooled-only."""
    existing = predictions.copy() if not predictions.empty else pd.DataFrame(columns=UNIVERSE_LATEST_PREDICTION_COLUMNS)
    for col in UNIVERSE_LATEST_PREDICTION_COLUMNS:
        if col not in existing.columns:
            existing[col] = None
    return existing[UNIVERSE_LATEST_PREDICTION_COLUMNS].copy()


def _universe_prediction_row(symbol: str) -> dict[str, Any]:
    """Latest pooled prediction row for a symbol from the universe-level feed."""
    preds = safe_read_csv(RULE_DIR / "tsm_universe_latest_predictions.csv")
    if preds.empty or "symbol" not in preds.columns:
        return {}
    rows = preds[preds["symbol"].astype(str).str.upper() == symbol]
    if rows.empty:
        return {}
    if "date" in rows.columns:
        rows = rows.sort_values("date")
    return clean_json(rows.iloc[-1].to_dict())


def _decision_member_row(symbol: str) -> pd.Series:
    config = _decision_config_or_default(safe_read_csv(RULE_DIR / "tsm_decision_universe_config.csv"))
    if config.empty or "symbol" not in config.columns:
        return pd.Series({"symbol": symbol})
    rows = config[config["symbol"].astype(str).str.upper() == symbol]
    if rows.empty:
        return pd.Series({"symbol": symbol})
    return rows.iloc[-1]


def build_symbol_summary(raw_symbol: str) -> dict[str, Any]:
    """Per-symbol payload shaped like build_summary so the ontology buildModel works.

    Powers the main-page Object View drill-down (/api/symbol?sym=NVDA). Reads the
    symbol's own outputs under universe/<SYM>/ plus the symbol's row from the
    universe-level pooled prediction feed.
    """
    symbol = str(raw_symbol or "").strip().upper()
    allowed = _allowed_universe_symbols()
    if symbol not in allowed:
        return {"error": f"unknown symbol: {raw_symbol}", "allowed": sorted(allowed)}

    group = allowed.get(symbol, "")
    out = OUTPUT_DIR / "universe" / symbol
    rule = RULE_DIR / "universe" / symbol

    enriched = safe_read_csv(out / "tsm_daily_10y_enriched.csv")
    signals = ensure_v2_signal_columns(safe_read_csv(rule / "tsm_daily_algorithmic_signals.csv"))
    risk = _normalize_risk_weight_columns(safe_read_csv(rule / "tsm_risk_policy_daily.csv"))
    equity = safe_read_csv(rule / "tsm_backtest_equity_curves.csv")

    price_columns = [
        "date", "close", "close_native", "close_usd", "listing_currency", "display_currency", "engine_currency", "fx_pair", "fx_rate_to_usd", "usdkrw",
        "sma_20", "sma_50", "sma_200", "drawdown_from_ath",
        "vol_20d_ann", "vol_63d_ann", "atr_14_pct", "volume", "close_change_pct",
        "rsi_14", "relative_return_vs_spy_60d", "relative_return_vs_smh_60d",
        "beta_vs_spy_252d",
    ]
    signal_columns = [
        "date", "close", "close_native", "close_usd", "display_currency", "engine_currency", "fx_rate_to_usd", "usdkrw",
        "score_price_algo_total", "entry_trigger", "trade_action",
        "raw_entry_event", "decision_tier", "sizing_tier", "suggested_action", "suggested_weight", "semi_momentum_regime",
        "strict_signal_stage", "research_signal_stage", "research_signal_action",
        "research_signal_score", "atr_stop_2x", "take_profit_2R", "risk_pct_2atr",
        "position_weight_if_0_5pct_account_risk",
    ]
    risk_columns = [
        "date", "close", "close_native", "close_usd", "display_currency", "engine_currency", "fx_rate_to_usd", "usdkrw",
        "final_recommended_max_weight", "final_recommended_max_weight_pct",
        "risk_state", "limiting_reason", "vol_limit_weight", "trend_limit_weight",
        "drawdown_limit_weight", "score_limit_weight", "account_risk_limit_weight",
        "secondary_account_risk_limit_weight",
    ]
    equity_tail = equity.tail(1500) if not equity.empty else equity

    pred_row = _universe_prediction_row(symbol)
    rule_fallback_diagnostic = {}
    if not pred_row:
        fallback_row = _rule_fallback_universe_prediction_row(_decision_member_row(symbol))
        rule_fallback_diagnostic = clean_json(fallback_row or {})
    pooled_prediction = {
        "symbol": symbol,
        "symbol_group": group,
        "asof_date": pred_row.get("date"),
        "p_success_20d": pred_row.get("p_success_20d"),
        "p_stop_hit_20d": pred_row.get("p_stop_hit_20d"),
        "p_stop_hit_raw_20d": pred_row.get("p_stop_hit_raw_20d"),
        "p_stop_hit_calibrated_20d": pred_row.get("p_stop_hit_calibrated_20d"),
        "p_stop_hit_raw_minus_calibrated_20d": pred_row.get("p_stop_hit_raw_minus_calibrated_20d"),
        "p_stop_hit_oos_percentile_20d": pred_row.get("p_stop_hit_oos_percentile_20d"),
        "strict_stop_risk_gap_20d": pred_row.get("strict_stop_risk_gap_20d"),
        "paper_stop_risk_gap_20d": pred_row.get("paper_stop_risk_gap_20d"),
        "stop_risk_calibration_warning": pred_row.get("stop_risk_calibration_warning"),
        "expected_r_20d": pred_row.get("expected_r_20d"),
        "expected_r_net_20d": pred_row.get("expected_r_20d"),
        "decision_score_20d": pred_row.get("decision_score_20d"),
        "paper_decision_score_20d": pred_row.get("paper_decision_score_20d"),
        "threshold_20d": pred_row.get("threshold_20d"),
        "decision_support_allowed": pred_row.get("decision_support_allowed"),
        "paper_decision_support_allowed": pred_row.get("paper_decision_support_allowed"),
        "latest_trade_ready": pred_row.get("latest_trade_ready"),
        "block_reasons": pred_row.get("block_reasons"),
        "model_name": pred_row.get("model_name"),
        "prediction_source": pred_row.get("prediction_source"),
        "live_trading_status": pred_row.get("live_trading_status"),
    }
    for horizon in POOLED_DASHBOARD_HORIZONS:
        suffix = f"{horizon}d"
        for field in POOLED_HORIZON_PREDICTION_FIELDS:
            pooled_prediction[f"{field}_{suffix}"] = pred_row.get(f"{field}_{suffix}")
        pooled_prediction[f"expected_r_net_{suffix}"] = pred_row.get(f"expected_r_{suffix}")
    paper_gate = {
        "paper_decision_support_allowed": pred_row.get("paper_decision_support_allowed"),
        "strict_decision_support_allowed": pred_row.get("decision_support_allowed"),
        "paper_decision_score_20d": pred_row.get("paper_decision_score_20d"),
        "paper_gate_status": "PAPER_DECISION_SUPPORT_ALLOWED"
        if str(pred_row.get("paper_decision_support_allowed", "")).strip().lower() in {"true", "1", "yes"}
        else "BLOCKED",
    }

    latest_price = {}
    if not enriched.empty:
        latest_price = clean_json(enriched.iloc[-1][[c for c in price_columns if c in enriched.columns]].to_dict())
    display_context = display_context_for_symbol(symbol, latest_price)

    def _display_currency(row: Any = None) -> str:
        if row is not None:
            try:
                value = row.get("display_currency")
                if value and str(value).strip().lower() not in {"nan", "none"}:
                    return str(value).strip().upper()
            except Exception:
                pass
        value = latest_price.get("display_currency") or latest_price.get("listing_currency")
        if value and str(value).strip().lower() not in {"nan", "none"}:
            return str(value).strip().upper()
        return "KRW" if symbol.endswith(".KS") else "USD"

    def _fmt_price_value(value: Any, currency: str) -> str:
        n = _as_float(value)
        if n is None:
            return "없음"
        if currency == "KRW":
            return f"₩{n:,.0f}"
        return f"${n:,.2f}"

    def _price_for_display(row: Any, key: str) -> Any:
        currency = _display_currency(row)
        if currency == "KRW":
            try:
                native = row.get(f"{key}_native")
                if _as_float(native) is not None:
                    return native
                value = row.get(key)
                fx_rate = _as_float(row.get("fx_rate_to_usd"))
                value_num = _as_float(value)
                if value_num is not None and fx_rate and fx_rate > 0:
                    return value_num / fx_rate
            except Exception:
                return None
        try:
            usd = row.get(f"{key}_usd")
            if _as_float(usd) is not None:
                return usd
            return row.get(key)
        except Exception:
            return None

    def _fmt_price(row: Any, key: str) -> str:
        return _fmt_price_value(_price_for_display(row, key), _display_currency(row))

    def _fmt_engine_price(value: Any, row: Any = None) -> str:
        n = _as_float(value)
        if n is None:
            return "없음"
        currency = _display_currency(row)
        if currency == "KRW":
            fx_rate = None
            if row is not None:
                try:
                    fx_rate = _as_float(row.get("fx_rate_to_usd"))
                except Exception:
                    fx_rate = None
            fx_rate = fx_rate or _as_float(latest_price.get("fx_rate_to_usd")) or _as_float(display_context.get("fx_rate_to_usd"))
            if fx_rate and fx_rate > 0:
                n = n / fx_rate
        return _fmt_price_value(n, currency)

    def _fmt_usd(value: Any) -> str:
        return _fmt_price_value(value, "USD")

    def _fmt_num(value: Any, digits: int = 1) -> str:
        n = _as_float(value)
        return f"{n:,.{digits}f}" if n is not None else "없음"

    def _fmt_pct(value: Any, digits: int = 2) -> str:
        n = _as_float(value)
        if n is None:
            return "없음"
        pct = n * 100 if abs(n) <= 1 else n
        return f"{pct:,.{digits}f}%"

    def _trigger_focus_snapshot() -> dict[str, Any]:
        if signals.empty:
            return with_display_context({}, symbol, latest_price)
        s = signals.iloc[-1]
        high_values = pd.to_numeric(signals.get("high", pd.Series(dtype=float)), errors="coerce")
        prev_20d_high = high_values.shift(1).rolling(20, min_periods=10).max().iloc[-1]
        prev_60d_high = high_values.shift(1).rolling(60, min_periods=30).max().iloc[-1]
        close = _as_float(s.get("close"))
        sma_50 = _as_float(s.get("sma_50"))
        pullback_low = sma_50 * 0.97 if sma_50 is not None else None
        pullback_high = sma_50 * 1.05 if sma_50 is not None else None
        raw_breakout_20d = bool(close is not None and _as_float(prev_20d_high) is not None and close > _as_float(prev_20d_high))
        raw_breakout_60d = bool(close is not None and _as_float(prev_60d_high) is not None and close > _as_float(prev_60d_high))

        def flag_value(name: str) -> bool:
            value = s.get(name)
            if value is None:
                return False
            return str(value).strip().lower() in {"true", "1", "yes"}

        clean_breakout_20d = flag_value("trigger_breakout_20d")
        clean_breakout_60d = flag_value("trigger_breakout_60d")
        v2_tier = str(s.get("decision_tier", "") or "")
        v2_action = str(s.get("suggested_action", "") or "")
        v2_actionable = v2_tier in {"STRICT_ENTRY_ALLOWED", "AGGRESSIVE_TREND_ENTRY", "BREAKOUT_EXTENSION_TINY", "PULLBACK_REENTRY"}
        raw_breakout_blocked = (raw_breakout_20d or raw_breakout_60d) and not (clean_breakout_20d or clean_breakout_60d)
        block_reasons: list[str] = []
        if raw_breakout_blocked:
            if flag_value("algo_vol_extreme"):
                block_reasons.append("극단 변동성")
            if flag_value("algo_overextended_highvol"):
                block_reasons.append("과열/고변동")
            if flag_value("algo_event_shock_day"):
                block_reasons.append("이벤트 쇼크")
            if not flag_value("algo_trend_up_loose"):
                block_reasons.append("추세 필터")
            score = _as_float(s.get("score_price_algo_total"))
            if score is not None and score < 60:
                block_reasons.append("점수 60 미만")
        price_breakout_block_reason = "|".join(block_reasons) if block_reasons else ("리스크 필터" if raw_breakout_blocked else "PASS")
        trigger_candidates = [
            ("20일 고점 돌파", _as_float(prev_20d_high), "가격 기준입니다. 실제 신호는 추세/변동성 필터까지 통과해야 합니다."),
            ("60일 고점 돌파", _as_float(prev_60d_high), "가격 기준입니다. 실제 신호는 추세/변동성 필터까지 통과해야 합니다."),
        ]
        if close is not None and pullback_low is not None and pullback_high is not None:
            if pullback_low <= close <= pullback_high:
                trigger_candidates.append(("50일선 눌림목", close, "현재가가 50일선 눌림목 밴드 안에 있음"))
            else:
                nearest_pullback = pullback_high if close > pullback_high else pullback_low
                trigger_candidates.append(("50일선 눌림목", nearest_pullback, "50일선 -3% ~ +5% 눌림목 밴드 재진입 기준"))
        viable = [
            (label, price, note)
            for label, price, note in trigger_candidates
            if price is not None and math.isfinite(price) and close is not None
        ]
        if viable:
            nearest_label, nearest_price, nearest_note = min(viable, key=lambda item: abs((item[1] - close) / close) if close else math.inf)
        else:
            nearest_label, nearest_price, nearest_note = ("매수 트리거", None, "트리거 기준가 계산 대기")
        trigger_distance_pct = ((nearest_price / close) - 1.0) * 100 if close and nearest_price is not None else None
        focus = {
            "symbol": symbol,
            "date": s.get("date"),
            "close": close,
            "entry_trigger": s.get("entry_trigger"),
            "trade_action": s.get("trade_action"),
            "score_price_algo_total": s.get("score_price_algo_total"),
            "today_decision": v2_action if v2_action else s.get("trade_action"),
            "decision_tier": s.get("decision_tier"),
            "sizing_tier": s.get("sizing_tier"),
            "suggested_action": s.get("suggested_action"),
            "suggested_weight": s.get("suggested_weight"),
            "execution_status": s.get("execution_status"),
            "raw_entry_event": s.get("raw_entry_event"),
            "raw_52w_high_near": s.get("raw_52w_high_near"),
            "semi_momentum_regime": s.get("semi_momentum_regime"),
            "semi_group_momentum_score": s.get("semi_group_momentum_score"),
            "memory_ai_regime_score": s.get("memory_ai_regime_score"),
            "positive_thrust_day": s.get("positive_thrust_day"),
            "negative_shock_day": s.get("negative_shock_day"),
            "next_check_condition": s.get("next_check_condition"),
            "v2_actionable": v2_actionable,
            "raw_breakout_20d": raw_breakout_20d,
            "raw_breakout_60d": raw_breakout_60d,
            "raw_breakout_blocked": raw_breakout_blocked,
            "price_breakout_block_reason": price_breakout_block_reason,
            "trigger_breakout_20d": clean_breakout_20d,
            "trigger_breakout_60d": clean_breakout_60d,
            "trigger_pullback_50d": s.get("trigger_pullback_50d"),
            "algo_vol_extreme": s.get("algo_vol_extreme"),
            "algo_overextended_highvol": s.get("algo_overextended_highvol"),
            "algo_event_shock_day": s.get("algo_event_shock_day"),
            "breakout_20d_price": _as_float(prev_20d_high),
            "breakout_60d_price": _as_float(prev_60d_high),
            "pullback_50d_low": pullback_low,
            "pullback_50d_high": pullback_high,
            "nearest_trigger_label": nearest_label,
            "nearest_trigger_price": nearest_price,
            "nearest_trigger_note": nearest_note,
            "nearest_trigger_distance_pct": trigger_distance_pct,
            "stop_price_2atr": s.get("atr_stop_2x"),
            "stop_price_1_8atr": s.get("stop_price_1_8atr"),
            "invalidation_5d_low": s.get("invalidation_5d_low"),
            "invalidation_ema10": s.get("invalidation_ema10"),
            "risk_per_share_2atr": (2.0 * _as_float(s.get("atr_14"))) if _as_float(s.get("atr_14")) is not None else None,
            "risk_pct_2atr": s.get("risk_pct_2atr"),
            "take_profit_2R": s.get("take_profit_2R"),
            "take_profit_3R": s.get("take_profit_3R"),
        }
        return with_display_context(clean_json(focus), symbol, latest_price)

    trading_plan: list[dict[str, Any]] = []
    if not signals.empty:
        s = signals.iloc[-1]
        high_values = pd.to_numeric(signals.get("high", pd.Series(dtype=float)), errors="coerce")
        prev_20d_high = high_values.shift(1).rolling(20, min_periods=10).max().iloc[-1]
        prev_60d_high = high_values.shift(1).rolling(60, min_periods=30).max().iloc[-1]
        pullback_low = _as_float(s.get("sma_50")) * 0.97 if _as_float(s.get("sma_50")) is not None else None
        pullback_high = _as_float(s.get("sma_50")) * 1.05 if _as_float(s.get("sma_50")) is not None else None
        risk_per_share = 2.0 * _as_float(s.get("atr_14")) if _as_float(s.get("atr_14")) is not None else None
        trading_plan = [
            {"section": "가격", "item": "최근 종가", "value": _fmt_price(s, "close"), "unit": "", "notes": f"기준일 {s.get('date', '')}"},
            {"section": "가격", "item": "현재 종가", "value": _fmt_price(s, "close"), "unit": "", "notes": f"기준일 {s.get('date', '')}"},
            {"section": "매수", "item": "매수 신호", "value": str(s.get("entry_trigger") or "NONE"), "unit": "", "notes": "규칙 기반 매수 트리거"},
            {"section": "매수", "item": "매수 점수", "value": _fmt_num(s.get("score_price_algo_total"), 1), "unit": "점", "notes": "75점 이상이면 매수 후보"},
            {"section": "매수", "item": "엄격 행동", "value": str(s.get("trade_action") or "-"), "unit": "", "notes": "엄격 규칙이 내린 행동"},
            {"section": "진입", "item": "20일 고점 돌파 기준가", "value": _fmt_engine_price(prev_20d_high, s), "unit": "", "notes": "가격 기준입니다. 실제 신호는 추세/변동성 필터까지 통과해야 합니다."},
            {"section": "진입", "item": "60일 고점 돌파 기준가", "value": _fmt_engine_price(prev_60d_high, s), "unit": "", "notes": "가격 기준입니다. 실제 신호는 추세/변동성 필터까지 통과해야 합니다."},
            {"section": "진입", "item": "50일선 눌림목 구간", "value": f"{_fmt_engine_price(pullback_low, s)} ~ {_fmt_engine_price(pullback_high, s)}", "unit": "", "notes": "50일선 -3% ~ +5% 구간"},
            {"section": "위험", "item": "손절 기준가", "value": _fmt_price(s, "atr_stop_2x"), "unit": "", "notes": "2x ATR 손절가"},
            {"section": "리스크", "item": "2ATR 손절가", "value": _fmt_price(s, "atr_stop_2x"), "unit": "", "notes": "현재 종가 기준 룰 엔진 손절가"},
            {"section": "리스크", "item": "1주당 2ATR 리스크", "value": _fmt_engine_price(risk_per_share, s), "unit": "", "notes": "진입가 기준 실제 손절폭 산정에 사용"},
            {"section": "위험", "item": "2차 목표가", "value": _fmt_price(s, "take_profit_2R"), "unit": "", "notes": "2R 목표가"},
            {"section": "목표", "item": "1차 목표가 2R", "value": _fmt_price(s, "take_profit_2R"), "unit": "", "notes": "현재가 기준 2R"},
            {"section": "목표", "item": "2차 목표가 3R", "value": _fmt_price(s, "take_profit_3R"), "unit": "", "notes": "현재가 기준 3R"},
            {"section": "위험", "item": "손절까지 거리", "value": _fmt_pct(s.get("risk_pct_2atr")), "unit": "", "notes": "현재가 대비 손절 거리"},
            {"section": "위험", "item": "계좌위험 0.5% 기준 비중", "value": _fmt_pct(s.get("position_weight_if_0_5pct_account_risk")), "unit": "", "notes": "계좌 0.5% 위험 기준 매수 비중"},
        ]

    return clean_json(
        {
            "generated_at": now_iso(),
            "scope": "symbol",
            "meta": {
                "symbol": symbol,
                "symbol_name": symbol_display_name(symbol),
                "symbol_display_name": symbol_display_name(symbol),
                "symbol_group": group,
                "group_label": SYMBOL_GROUP_LABELS.get(group, group),
                "listing_currency": latest_price.get("listing_currency") or _display_currency(),
                "display_currency": _display_currency(),
                "engine_currency": latest_price.get("engine_currency") or "USD",
                "fx_pair": latest_price.get("fx_pair") or ("KRW=X" if _display_currency() == "KRW" else ""),
                "fx_rate_to_usd": display_context.get("fx_rate_to_usd"),
                "usdkrw": display_context.get("usdkrw"),
                "as_of": pred_row.get("date") or latest_price.get("date"),
            },
            "snapshots": {
                "decision": with_display_context(field_value_map(rule / "tsm_latest_decision_snapshot.csv"), symbol, latest_price),
                "risk": _normalize_risk_snapshot_fields(with_display_context(field_value_map(rule / "tsm_latest_risk_snapshot.csv"), symbol, latest_price)),
                "prediction": _normalize_prediction_snapshot_fields(dict(pooled_prediction)),
                "pooled_prediction": pooled_prediction,
                "rule_fallback_diagnostic": rule_fallback_diagnostic,
                "paper_gate": paper_gate,
                "stress": with_display_context(field_value_map(rule / "tsm_latest_stress_snapshot.csv"), symbol, latest_price),
                "integrity": field_value_map(rule / "tsm_latest_integrity_snapshot.csv"),
                "data_quality": field_value_map(rule / "tsm_latest_data_quality_snapshot.csv"),
                "integrated_price": with_display_context(field_value_map(rule / "tsm_integrated_price_summary.csv"), symbol, latest_price),
                "latest_price": with_display_context(latest_price, symbol, latest_price),
                "trigger_focus": _trigger_focus_snapshot(),
            },
            "quality": {
                "integrity": quality_summary(rule / "tsm_daily_integrity_checks.csv", critical_only=True),
                "data_quality": quality_summary(rule / "tsm_data_quality_checks.csv", critical_only=True),
                "data_contract": mismatch_summary(rule / "tsm_data_validation_raw_vs_enriched.csv"),
            },
            "series": {
                "price": records_from_df(enriched, price_columns, tail=400),
                "signals": records_from_df(signals, signal_columns, tail=400),
                "risk": records_from_df(risk, risk_columns, tail=400),
                "equity": records_from_df(equity_tail),
            },
            "tables": {
                "trading_plan": trading_plan,
                "backtest_summary": records_from_df(safe_read_csv(rule / "tsm_backtest_strategy_summary.csv")),
                "trade_log": records_from_df(
                    enrich_display_context(
                        safe_read_csv(rule / "tsm_backtest_trade_log.csv")
                        .tail(250)
                        .assign(symbol=symbol, **{k: v for k, v in display_context.items() if k != "symbol"})
                    )
                ),
                "yearly_returns": records_from_df(safe_read_csv(rule / "tsm_backtest_yearly_returns.csv")),
                "rule_forward": records_from_df(safe_read_csv(rule / "tsm_rule_forward_return_stats.csv")),
                "regime_forward": records_from_df(safe_read_csv(rule / "tsm_regime_forward_return_stats.csv")),
                "event_analysis": records_from_df(safe_read_csv(rule / "tsm_event_integrated_analysis.csv")),
                "stress_scenarios": records_from_df(enrich_display_context(safe_read_csv(rule / "tsm_daily_stress_scenarios.csv").assign(symbol=symbol, **{k: v for k, v in display_context.items() if k != "symbol"}))),
                "strategy_stress": records_from_df(safe_read_csv(rule / "tsm_strategy_stress_summary.csv")),
                "extreme_moves": records_from_df(safe_read_csv(rule / "tsm_extreme_daily_moves.csv")),
                "yearly_price": records_from_df(safe_read_csv(rule / "tsm_yearly_price_volatility_stats.csv")),
                "monthly_price": records_from_df(safe_read_csv(rule / "tsm_monthly_price_volatility_stats.csv").tail(60)),
                "data_validation": records_from_df(safe_read_csv(rule / "tsm_data_validation_raw_vs_enriched.csv")),
                "news_cause_forward": records_from_df(safe_read_csv(rule / "tsm_news_cause_forward_return_stats.csv")),
                "drawdown_episodes": records_from_df(safe_read_csv(rule / "tsm_drawdown_episodes.csv")),
            },
        }
    )


def resolve_allowed_file(raw_name: str) -> Path:
    if not raw_name:
        raise ValueError("missing file")
    raw_name = unquote(raw_name)
    rel = Path(raw_name)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("invalid file path")
    path = (ROOT / rel).resolve()
    if not any(path == base or base in path.parents for base in ALLOWED_FILE_ROOTS):
        raise ValueError("file is outside allowed output directories")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(raw_name)
    return path


def read_table_response(file_name: str, limit: int = 250, offset: int = 0, tail: bool = False) -> dict[str, Any]:
    path = resolve_allowed_file(file_name)
    if path.suffix.lower() != ".csv":
        raise ValueError("only csv files can be viewed as tables")
    df = read_csv_file(path)
    total = len(df)
    if tail:
        view = df.tail(limit)
    else:
        offset = max(0, offset)
        view = df.iloc[offset : offset + limit]
    return clean_json(
        {
            "file": relative_name(path),
            "total_rows": total,
            "columns": list(df.columns),
            "rows": view.to_dict(orient="records"),
        }
    )


def read_report_response(file_name: str) -> dict[str, Any]:
    path = resolve_allowed_file(file_name)
    if path.suffix.lower() != ".md":
        raise ValueError("only markdown reports can be read here")
    return {
        "file": relative_name(path),
        "content": path.read_text(encoding="utf-8", errors="replace"),
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
    }


def float_payload(payload: dict[str, Any], name: str, default: float) -> float:
    value = payload.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def str_payload(payload: dict[str, Any], name: str, default: str) -> str:
    value = payload.get(name, default)
    if value is None or value == "":
        return default
    return str(value)


def bool_payload(payload: dict[str, Any], name: str, default: bool = False) -> bool:
    if name not in payload:
        return default
    value = payload.get(name)
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "on"}


def command_for_mode(payload: dict[str, Any]) -> tuple[str, list[str]]:
    mode = str_payload(payload, "mode", "downstream")
    py = sys.executable
    start = str_payload(payload, "start", DEFAULT_START)
    end = str_payload(payload, "end", DEFAULT_END)
    preferred_source = str_payload(payload, "preferred_source", "stooq")
    kr_preferred_source = str_payload(payload, "kr_preferred_source", "yahoo")
    provider = str_payload(payload, "provider", "auto")
    interval = str_payload(payload, "interval", "1m")
    config = str_payload(payload, "config", "config/tsm_research.toml")
    universe_config = str_payload(payload, "universe_config", DEFAULT_UNIVERSE_CONFIG)
    decision_universe_config = str_payload(payload, "decision_universe_config", universe_config)
    research_universe_config = str_payload(payload, "research_universe_config", DEFAULT_RESEARCH_UNIVERSE_CONFIG)
    # Model-feature minute bars are fixed at 5m (the coverage/feature contract expects
    # tsm_5min_available_enriched.csv); do NOT fall back to the form's generic `interval`
    # selector, which defaults to 1m and would leave the 5m layer unfetched.
    model_minute_interval = str_payload(payload, "model_minute_interval", "5m")
    execution_minute_interval = str_payload(payload, "execution_minute_interval", "1m")
    universe_mode = str_payload(payload, "universe_mode", "hybrid")
    if universe_mode not in {"universe", "hybrid"}:
        universe_mode = "hybrid"
    external_features = str_payload(payload, "external_features", DEFAULT_EXTERNAL_FEATURES)
    prediction_policy = str_payload(payload, "prediction_policy", "conservative")
    news_mode = str_payload(payload, "news_mode", "daily")
    if news_mode not in {"daily", "backfill"}:
        news_mode = "daily"
    news_lookback_days = str(int(float_payload(payload, "news_lookback_days", 14)))
    output_dir = "output"
    rule_outdir = "tsm_price_rule_output"
    commission_bps = str(float_payload(payload, "commission_bps", 1.0))
    slippage_bps = str(float_payload(payload, "slippage_bps", 5.0))
    stop_multiple = str(float_payload(payload, "stop_multiple", 2.0))
    score_threshold = str(float_payload(payload, "score_threshold", 75.0))
    max_trials = str(int(float_payload(payload, "max_trials", 100)))

    if mode in {"full", "downstream"}:
        cmd = [
            py,
            "run_daily_update.py",
            "--config",
            config,
            "--universe-config",
            universe_config,
            "--decision-universe-config",
            decision_universe_config,
            "--research-universe-config",
            research_universe_config,
            "--universe-mode",
            universe_mode,
            "--start",
            start,
            "--end",
            end,
            "--preferred-source",
            preferred_source,
            "--kr-preferred-source",
            kr_preferred_source,
            "--output-dir",
            output_dir,
            "--rule-outdir",
            rule_outdir,
            "--commission-bps",
            commission_bps,
            "--slippage-bps",
            slippage_bps,
            "--stop-multiple",
            stop_multiple,
            "--score-threshold",
            score_threshold,
            "--prediction-policy",
            prediction_policy,
        ]
        if mode == "downstream":
            cmd.append("--skip-data-refresh")
        if bool_payload(payload, "schema_strict"):
            cmd.append("--schema-strict")
        if bool_payload(payload, "skip_charts"):
            cmd.append("--skip-charts")
        if bool_payload(payload, "skip_benchmarks"):
            cmd.append("--skip-benchmarks")
        if bool_payload(payload, "skip_news_refresh"):
            cmd.append("--skip-news-refresh")
        if news_mode == "backfill":
            cmd.append("--news-backfill")
        if bool_payload(payload, "skip_external_web"):
            cmd.append("--skip-external-web")
        if bool_payload(payload, "skip_data_refresh") and mode == "full":
            cmd.append("--skip-data-refresh")
        if bool_payload(payload, "skip_symbol_build"):
            cmd.append("--skip-universe-symbol-build")
        if bool_payload(payload, "skip_symbol_diagnostics"):
            cmd.append("--skip-universe-symbol-diagnostics")
        if bool_payload(payload, "enable_feedback_features"):
            cmd.append("--enable-feedback-features")
        if bool_payload(payload, "enable_auto_research"):
            cmd.extend(["--enable-auto-research", "--max-trials", max_trials])
        cmd.extend(["--news-lookback-days", news_lookback_days])
        if bool_payload(payload, "continue_on_error"):
            cmd.append("--continue-on-error")
        return mode, cmd

    if mode == "research_expansion_update":
        cmd = [
            py,
            "run_research_expansion_update.py",
            "--start",
            start,
            "--end",
            end,
            "--preferred-source",
            preferred_source,
            "--output-dir",
            output_dir,
            "--rule-outdir",
            rule_outdir,
            "--universe-config",
            universe_config,
            "--commission-bps",
            commission_bps,
            "--slippage-bps",
            slippage_bps,
            "--stop-multiple",
            stop_multiple,
            "--score-threshold",
            score_threshold,
            "--max-trials",
            max_trials,
        ]
        if bool_payload(payload, "skip_data_refresh"):
            cmd.append("--skip-data-refresh")
        if bool_payload(payload, "skip_news_refresh"):
            cmd.append("--skip-news-refresh")
        if bool_payload(payload, "skip_external_web"):
            cmd.append("--skip-external-web")
        if bool_payload(payload, "continue_on_error"):
            cmd.append("--continue-on-error")
        if bool_payload(payload, "enable_feedback_features", True):
            cmd.append("--enable-feedback-features")
        if bool_payload(payload, "enable_auto_research", True):
            cmd.append("--enable-auto-research")
        return mode, cmd

    if mode == "hourly":
        cmd = [
            py,
            "run_universe_market_data_update.py",
            "--universe-config",
            decision_universe_config,
            "--outdir",
            output_dir,
            "--start",
            start,
            "--end",
            end,
            "--bar-scope",
            "hourly",
            "--provider",
            provider,
        ]
        if bool_payload(payload, "skip_charts"):
            cmd.append("--skip-charts")
        if bool_payload(payload, "continue_on_error", True):
            cmd.append("--continue-on-error")
        return mode, cmd

    if mode == "intraday":
        cmd = [
            py,
            "run_universe_market_data_update.py",
            "--universe-config",
            decision_universe_config,
            "--outdir",
            output_dir,
            "--start",
            start,
            "--end",
            end,
            "--bar-scope",
            "minute",
            "--model-minute-interval",
            model_minute_interval,
            "--execution-minute-interval",
            execution_minute_interval or interval,
            "--provider",
            provider,
        ]
        if bool_payload(payload, "skip_charts"):
            cmd.append("--skip-charts")
        if bool_payload(payload, "continue_on_error", True):
            cmd.append("--continue-on-error")
        return mode, cmd

    if mode == "universe_market_data_update":
        cmd = [
            py,
            "run_universe_market_data_update.py",
            "--universe-config",
            decision_universe_config,
            "--outdir",
            output_dir,
            "--start",
            start,
            "--end",
            end,
            "--bar-scope",
            "both",
            "--model-minute-interval",
            model_minute_interval,
            "--execution-minute-interval",
            execution_minute_interval,
            "--provider",
            provider,
        ]
        if bool_payload(payload, "skip_charts", True):
            cmd.append("--skip-charts")
        if bool_payload(payload, "continue_on_error", True):
            cmd.append("--continue-on-error")
        return mode, cmd

    commands: dict[str, list[str]] = {
        "data_pipeline": [
            py,
            "tsm_daily_quant_pipeline.py",
            "--start",
            start,
            "--end",
            end,
            "--preferred-source",
            preferred_source,
            "--events",
            "tsm_events_seed.csv",
            "--outdir",
            output_dir,
        ],
        "rule_engine": [
            py,
            "tsm_price_rule_engine.py",
            "--enriched",
            f"{output_dir}/tsm_daily_10y_enriched.csv",
            "--raw",
            f"{output_dir}/tsm_daily_10y_raw.csv",
            "--summary",
            f"{output_dir}/tsm_daily_10y_summary.csv",
            "--events",
            f"{output_dir}/tsm_event_impact_10y.csv",
            "--news-daily",
            f"{rule_outdir}/tsm_news_integrated_daily.csv",
            "--outdir",
            rule_outdir,
            "--config",
            config,
        ],
        "news_causal_engine": [
            py,
            "tsm_news_causal_engine.py",
            "--enriched",
            f"{output_dir}/tsm_daily_10y_enriched.csv",
            "--events",
            "tsm_events_seed.csv",
            "--start",
            start,
            "--end",
            end,
            "--mode",
            news_mode,
            "--lookback-days",
            news_lookback_days,
            "--outdir",
            output_dir,
            "--rule-outdir",
            rule_outdir,
        ],
        "backtest_engine": [
            py,
            "tsm_backtest_engine.py",
            "--signals",
            f"{rule_outdir}/tsm_daily_algorithmic_signals.csv",
            "--enriched",
            f"{output_dir}/tsm_daily_10y_enriched.csv",
            "--outdir",
            rule_outdir,
            "--commission-bps",
            commission_bps,
            "--slippage-bps",
            slippage_bps,
            "--stop-multiple",
            stop_multiple,
            "--score-threshold",
            score_threshold,
        ],
        "backtest_event_ledger": [
            py,
            "tsm_backtest_event_ledger.py",
            "--signals",
            f"{rule_outdir}/tsm_daily_algorithmic_signals.csv",
            "--trade-log",
            f"{rule_outdir}/tsm_backtest_trade_log.csv",
            "--enriched",
            f"{output_dir}/tsm_daily_10y_enriched.csv",
            "--outdir",
            rule_outdir,
            "--commission-bps",
            commission_bps,
            "--slippage-bps",
            slippage_bps,
        ],
        "risk_engine": [
            py,
            "tsm_risk_engine.py",
            "--signals",
            f"{rule_outdir}/tsm_daily_algorithmic_signals.csv",
            "--outdir",
            rule_outdir,
        ],
        "validation_engine": [
            py,
            "tsm_validation_engine.py",
            "--signals",
            f"{rule_outdir}/tsm_daily_algorithmic_signals.csv",
            "--enriched",
            f"{output_dir}/tsm_daily_10y_enriched.csv",
            "--outdir",
            rule_outdir,
            "--baseline-stop-multiple",
            stop_multiple,
            "--baseline-score-threshold",
            score_threshold,
            "--commission-bps",
            commission_bps,
            "--slippage-bps",
            slippage_bps,
            "--model-trial-count",
            "60",
        ],
        "stress_engine": [
            py,
            "tsm_daily_stress_engine.py",
            "--signals",
            f"{rule_outdir}/tsm_daily_algorithmic_signals.csv",
            "--risk-policy",
            f"{rule_outdir}/tsm_risk_policy_daily.csv",
            "--drawdowns",
            f"{rule_outdir}/tsm_drawdown_episodes.csv",
            "--equity-curves",
            f"{rule_outdir}/tsm_backtest_equity_curves.csv",
            "--outdir",
            rule_outdir,
        ],
        "integrity_engine": [
            py,
            "tsm_daily_integrity_engine.py",
            "--raw",
            f"{output_dir}/tsm_daily_10y_raw.csv",
            "--enriched",
            f"{output_dir}/tsm_daily_10y_enriched.csv",
            "--signals",
            f"{rule_outdir}/tsm_daily_algorithmic_signals.csv",
            "--risk-policy",
            f"{rule_outdir}/tsm_risk_policy_daily.csv",
            "--trade-log",
            f"{rule_outdir}/tsm_backtest_trade_log.csv",
            "--equity-curves",
            f"{rule_outdir}/tsm_backtest_equity_curves.csv",
            "--outdir",
            rule_outdir,
        ],
        "prediction_engine": [
            py,
            "tsm_prediction_engine.py",
            "--config",
            config,
            "--signals",
            f"{rule_outdir}/tsm_daily_algorithmic_signals.csv",
            "--trade-log",
            f"{rule_outdir}/tsm_backtest_trade_log.csv",
            "--risk-policy",
            f"{rule_outdir}/tsm_risk_policy_daily.csv",
            "--enriched",
            f"{output_dir}/tsm_daily_10y_enriched.csv",
            "--outdir",
            rule_outdir,
            "--commission-bps",
            commission_bps,
            "--slippage-bps",
            slippage_bps,
            "--stop-multiple",
            stop_multiple,
            "--prediction-policy",
            prediction_policy,
            "--external-features",
            external_features,
            "--intraday-features",
            f"{rule_outdir}/tsm_intraday_daily_features.csv",
        ],
        "next_day_up_model_engine": [
            py,
            "tsm_next_day_up_model_engine.py",
            "--aggregate-universe",
            "--universe-config",
            decision_universe_config,
            "--universe-rule-root",
            f"{rule_outdir}/universe",
            "--universe-data-root",
            f"{output_dir}/universe",
            "--external-features",
            external_features,
            "--intraday-features",
            f"{rule_outdir}/tsm_intraday_daily_features.csv",
            "--outdir",
            rule_outdir,
            "--latest-prediction",
            f"{rule_outdir}/tsm_latest_prediction_snapshot.csv",
            "--symbol",
            "TSM",
        ],
        "next_close_forecast_engine": [
            py,
            "tsm_next_close_forecast_engine.py",
            "--pooled-feature-matrix",
            f"{rule_outdir}/tsm_prediction_pooled_feature_matrix.csv",
            "--decision-universe-config",
            decision_universe_config,
            "--latest-prediction",
            f"{rule_outdir}/tsm_latest_prediction_snapshot.csv",
            "--outdir",
            rule_outdir,
            "--symbol",
            "TSM",
        ],
        "external_feature_engine": [
            py,
            "tsm_external_feature_engine.py",
            "--outdir",
            rule_outdir,
            "--universe-config",
            research_universe_config,
            "--start",
            start,
            "--end",
            end,
        ],
        "ml_overlay_backtest": [
            py,
            "tsm_ml_overlay_backtest.py",
            "--oos-predictions",
            f"{rule_outdir}/tsm_prediction_oos_predictions.csv",
            "--outdir",
            rule_outdir,
        ],
        "pooled_dataset_builder": [
            py,
            "run_pooled_universe_update.py",
            "--skip-symbol-build",
            "--universe-config",
            decision_universe_config,
            "--decision-universe-config",
            decision_universe_config,
            "--research-universe-config",
            research_universe_config,
            "--outdir",
            rule_outdir,
            "--commission-bps",
            commission_bps,
            "--slippage-bps",
            slippage_bps,
            "--stop-multiple",
            stop_multiple,
            "--external-features",
            external_features,
            "--intraday-features",
            f"{rule_outdir}/tsm_intraday_daily_features.csv",
        ],
        "pooled_model_engine": [
            py,
            "tsm_pooled_model_engine.py",
            "--pooled-feature-matrix",
            f"{rule_outdir}/tsm_prediction_pooled_feature_matrix.csv",
            "--pooled-quality",
            f"{rule_outdir}/tsm_prediction_pooled_quality_checks.csv",
            "--latest-prediction",
            f"{rule_outdir}/tsm_latest_prediction_snapshot.csv",
            "--outdir",
            rule_outdir,
        ],
        "backtest_feedback_feature_engine": [
            py,
            "tsm_backtest_feedback_feature_engine.py",
            "--ledger",
            f"{rule_outdir}/tsm_backtest_event_ledger.csv",
            "--oof-predictions",
            f"{rule_outdir}/tsm_pooled_model_oof_predictions.csv",
            "--outdir",
            rule_outdir,
        ],
        "auto_research_engine": [
            py,
            "tsm_auto_research_engine.py",
            "--pooled-comparison",
            f"{rule_outdir}/tsm_pooled_model_comparison.csv",
            "--local-comparison",
            f"{rule_outdir}/tsm_prediction_model_comparison.csv",
            "--pbo-report",
            f"{rule_outdir}/tsm_pbo_report.csv",
            "--dsr-report",
            f"{rule_outdir}/tsm_deflated_sharpe_report.csv",
            "--cpcv-model-distribution",
            f"{rule_outdir}/tsm_cpcv_model_distribution.csv",
            "--outdir",
            rule_outdir,
            "--storage",
            f"sqlite:///{rule_outdir}/tsm_research_trials.sqlite",
            "--max-trials",
            max_trials,
        ],
        "data_quality_engine": [
            py,
            "tsm_data_quality_engine.py",
            "--raw",
            f"{output_dir}/tsm_daily_10y_raw.csv",
            "--enriched",
            f"{output_dir}/tsm_daily_10y_enriched.csv",
            "--signals",
            f"{rule_outdir}/tsm_daily_algorithmic_signals.csv",
            "--outdir",
            rule_outdir,
            "--run-date",
            end,
        ],
        "model_gate_engine": [
            py,
            "tsm_model_gate_engine.py",
            "--comparison",
            f"{rule_outdir}/tsm_prediction_model_comparison.csv",
            "--latest-prediction",
            f"{rule_outdir}/tsm_latest_prediction_snapshot.csv",
            "--pooled-comparison",
            f"{rule_outdir}/tsm_pooled_model_comparison.csv",
            "--pooled-latest",
            f"{rule_outdir}/tsm_pooled_latest_prediction_overlay.csv",
            "--tsm-calibration",
            f"{rule_outdir}/tsm_pooled_tsm_calibration_metrics.csv",
            "--cpcv-strategy-distribution",
            f"{rule_outdir}/tsm_cpcv_strategy_distribution.csv",
            "--cpcv-model-distribution",
            f"{rule_outdir}/tsm_cpcv_model_distribution.csv",
            "--pbo-report",
            f"{rule_outdir}/tsm_pbo_report.csv",
            "--cscv-pbo-report",
            f"{rule_outdir}/tsm_cscv_pbo_report.csv",
            "--dsr-report",
            f"{rule_outdir}/tsm_deflated_sharpe_report.csv",
            "--paper-gate",
            f"{rule_outdir}/tsm_paper_gate_snapshot.csv",
            "--next-close-comparison",
            f"{rule_outdir}/tsm_next_close_model_comparison.csv",
            "--next-close-quality",
            f"{rule_outdir}/tsm_next_close_quality_checks.csv",
            "--next-close-latest",
            f"{rule_outdir}/tsm_next_close_latest_snapshot.csv",
            "--outdir",
            rule_outdir,
        ],
        "model_registry_engine": [
            py,
            "tsm_model_registry_engine.py",
            "--outdir",
            rule_outdir,
            "--config",
            config,
            "--label-dataset",
            f"{rule_outdir}/tsm_prediction_label_dataset.csv",
            "--feature-matrix",
            f"{rule_outdir}/tsm_prediction_feature_matrix.csv",
            "--feature-contract",
            f"{rule_outdir}/tsm_prediction_feature_contract.csv",
            "--fold-manifest",
            f"{rule_outdir}/tsm_prediction_fold_manifest.csv",
            "--schema-quality",
            f"{rule_outdir}/tsm_schema_quality_checks.csv",
            "--prediction-engine-source",
            "tsm_prediction_engine.py",
        ],
        "shadow_paper_engine": [
            py,
            "tsm_shadow_paper_engine.py",
            "--prediction",
            f"{rule_outdir}/tsm_latest_prediction_snapshot.csv",
            "--signals",
            f"{rule_outdir}/tsm_daily_algorithmic_signals.csv",
            "--ledger",
            f"{rule_outdir}/tsm_shadow_paper_predictions.csv",
            "--outdir",
            rule_outdir,
            "--commission-bps",
            commission_bps,
            "--slippage-bps",
            slippage_bps,
            "--stop-multiple",
            stop_multiple,
            "--external-features",
            external_features,
        ],
        "order_intent_engine": [
            py,
            "tsm_order_intent_engine.py",
            "--signals",
            f"{rule_outdir}/tsm_daily_algorithmic_signals.csv",
            "--risk",
            f"{rule_outdir}/tsm_latest_risk_snapshot.csv",
            "--prediction",
            f"{rule_outdir}/tsm_latest_prediction_snapshot.csv",
            "--system-state",
            f"{rule_outdir}/tsm_latest_system_state.csv",
            "--ledger",
            f"{rule_outdir}/tsm_order_intents.csv",
            "--outdir",
            rule_outdir,
            "--config",
            config,
            "--universe-config",
            decision_universe_config,
            "--latest-predictions",
            f"{rule_outdir}/tsm_universe_latest_predictions.csv",
            "--signals-root",
            rule_outdir,
        ],
        "portfolio_risk_engine": [
            py,
            "tsm_portfolio_risk_engine.py",
            "--intents",
            f"{rule_outdir}/tsm_order_intents.csv",
            "--signals",
            f"{rule_outdir}/tsm_daily_algorithmic_signals.csv",
            "--outdir",
            rule_outdir,
            "--config",
            config,
            "--decision-universe-config",
            decision_universe_config,
            "--latest-signals",
            f"{rule_outdir}/tsm_universe_latest_signals.csv",
        ],
        "paper_execution_engine": [
            py,
            "tsm_paper_execution_engine.py",
            "--intents",
            f"{rule_outdir}/tsm_order_intents.csv",
            "--portfolio-decisions",
            f"{rule_outdir}/tsm_portfolio_risk_order_decisions.csv",
            "--signals",
            f"{rule_outdir}/tsm_daily_algorithmic_signals.csv",
            "--orders",
            f"{rule_outdir}/tsm_paper_orders.csv",
            "--fills",
            f"{rule_outdir}/tsm_paper_fills.csv",
            "--outdir",
            rule_outdir,
            "--config",
            config,
            "--commission-bps",
            commission_bps,
            "--signals-root",
            rule_outdir,
            "--universe-config",
            decision_universe_config,
        ],
        "position_reconciler": [
            py,
            "tsm_position_reconciler.py",
            "--positions",
            f"{rule_outdir}/tsm_paper_positions.csv",
            "--fills",
            f"{rule_outdir}/tsm_paper_fills.csv",
            "--outdir",
            rule_outdir,
        ],
        "order_state_machine": [
            py,
            "tsm_order_state_machine.py",
            "--intents",
            f"{rule_outdir}/tsm_order_intents.csv",
            "--portfolio-decisions",
            f"{rule_outdir}/tsm_portfolio_risk_order_decisions.csv",
            "--orders",
            f"{rule_outdir}/tsm_paper_orders.csv",
            "--fills",
            f"{rule_outdir}/tsm_paper_fills.csv",
            "--reconciliation",
            f"{rule_outdir}/tsm_paper_reconciliation_report.csv",
            "--outdir",
            rule_outdir,
        ],
        "execution_feedback_engine": [
            py,
            "tsm_execution_feedback_engine.py",
            "--orders",
            f"{rule_outdir}/tsm_paper_orders.csv",
            "--fills",
            f"{rule_outdir}/tsm_paper_fills.csv",
            "--signals",
            f"{rule_outdir}/tsm_daily_algorithmic_signals.csv",
            "--outdir",
            rule_outdir,
            "--config",
            config,
        ],
        "fill_model_calibration_engine": [
            py,
            "tsm_fill_model_calibration_engine.py",
            "--feedback-events",
            f"{rule_outdir}/tsm_execution_feedback_events.csv",
            "--outdir",
            rule_outdir,
            "--config",
            config,
        ],
        "automation_scheduler": [
            py,
            "tsm_automation_scheduler.py",
            "--lifecycle-snapshot",
            f"{rule_outdir}/tsm_order_lifecycle_snapshot.csv",
            "--outdir",
            rule_outdir,
            "--config",
            config,
        ],
        "pooled_universe_update": [
            py,
            "run_pooled_universe_update.py",
            "--universe-config",
            decision_universe_config,
            "--decision-universe-config",
            decision_universe_config,
            "--research-universe-config",
            research_universe_config,
            "--outdir",
            rule_outdir,
            "--start",
            start,
            "--end",
            end,
            "--preferred-source",
            preferred_source,
            "--kr-preferred-source",
            kr_preferred_source,
            "--commission-bps",
            commission_bps,
            "--slippage-bps",
            slippage_bps,
            "--stop-multiple",
            stop_multiple,
            "--intraday-features",
            f"{rule_outdir}/tsm_intraday_daily_features.csv",
        ],
        "system_state_engine": [
            py,
            "tsm_system_state_engine.py",
            "--operational-quality",
            f"{rule_outdir}/tsm_operational_quality_checks.csv",
            "--integrity-quality",
            f"{rule_outdir}/tsm_daily_integrity_checks.csv",
            "--validation-quality",
            f"{rule_outdir}/tsm_validation_quality_checks.csv",
            "--prediction-quality",
            f"{rule_outdir}/tsm_prediction_quality_checks.csv",
            "--prediction-snapshot",
            f"{rule_outdir}/tsm_latest_prediction_snapshot.csv",
            "--risk-snapshot",
            f"{rule_outdir}/tsm_latest_risk_snapshot.csv",
            "--stress-snapshot",
            f"{rule_outdir}/tsm_latest_stress_snapshot.csv",
            "--backtest-summary",
            f"{rule_outdir}/tsm_backtest_strategy_summary.csv",
            "--walk-forward",
            f"{rule_outdir}/tsm_validation_walk_forward_summary.csv",
            "--causal-walk-forward",
            f"{rule_outdir}/tsm_validation_causal_walk_forward_summary.csv",
            "--paper-oms-quality",
            f"{rule_outdir}/tsm_paper_oms_quality_checks.csv",
            "--paper-reconciliation-quality",
            f"{rule_outdir}/tsm_paper_reconciliation_quality_checks.csv",
            "--order-state-quality",
            f"{rule_outdir}/tsm_order_state_quality_checks.csv",
            "--execution-feedback-quality",
            f"{rule_outdir}/tsm_execution_feedback_quality_checks.csv",
            "--fill-calibration-quality",
            f"{rule_outdir}/tsm_fill_model_calibration_quality_checks.csv",
            "--automation-quality",
            f"{rule_outdir}/tsm_automation_quality_checks.csv",
            "--outdir",
            rule_outdir,
        ],
        "daily_trading_report": [
            py,
            "tsm_daily_trading_report.py",
            "--latest",
            f"{rule_outdir}/tsm_latest_decision_snapshot.csv",
            "--signals",
            f"{rule_outdir}/tsm_daily_algorithmic_signals.csv",
            "--risk",
            f"{rule_outdir}/tsm_latest_risk_snapshot.csv",
            "--backtest-summary",
            f"{rule_outdir}/tsm_backtest_strategy_summary.csv",
            "--validation-quality",
            f"{rule_outdir}/tsm_validation_quality_checks.csv",
            "--walk-forward",
            f"{rule_outdir}/tsm_validation_walk_forward_summary.csv",
            "--stress",
            f"{rule_outdir}/tsm_latest_stress_snapshot.csv",
            "--system-state",
            f"{rule_outdir}/tsm_latest_system_state.csv",
            "--prediction",
            f"{rule_outdir}/tsm_latest_prediction_snapshot.csv",
            "--order-intents",
            f"{rule_outdir}/tsm_order_intents.csv",
            "--paper-positions",
            f"{rule_outdir}/tsm_paper_positions.csv",
            "--paper-reconciliation",
            f"{rule_outdir}/tsm_paper_reconciliation_report.csv",
            "--order-lifecycle",
            f"{rule_outdir}/tsm_order_lifecycle_snapshot.csv",
            "--execution-feedback",
            f"{rule_outdir}/tsm_execution_feedback_events.csv",
            "--fill-calibration",
            f"{rule_outdir}/tsm_fill_model_calibration.csv",
            "--automation-plan",
            f"{rule_outdir}/tsm_automation_plan.csv",
            "--outdir",
            rule_outdir,
        ],
        "daily_health_report": [
            py,
            "tsm_daily_health_report.py",
            "--data-quality",
            f"{rule_outdir}/tsm_latest_data_quality_snapshot.csv",
            "--system-state",
            f"{rule_outdir}/tsm_latest_system_state.csv",
            "--prediction",
            f"{rule_outdir}/tsm_latest_prediction_snapshot.csv",
            "--model-gate",
            f"{rule_outdir}/tsm_model_gate_snapshot.csv",
            "--risk",
            f"{rule_outdir}/tsm_latest_risk_snapshot.csv",
            "--shadow-quality",
            f"{rule_outdir}/tsm_shadow_paper_quality_checks.csv",
            "--paper-oms-quality",
            f"{rule_outdir}/tsm_paper_oms_quality_checks.csv",
            "--paper-reconciliation-quality",
            f"{rule_outdir}/tsm_paper_reconciliation_quality_checks.csv",
            "--order-state-quality",
            f"{rule_outdir}/tsm_order_state_quality_checks.csv",
            "--execution-feedback-quality",
            f"{rule_outdir}/tsm_execution_feedback_quality_checks.csv",
            "--fill-calibration-quality",
            f"{rule_outdir}/tsm_fill_model_calibration_quality_checks.csv",
            "--automation-quality",
            f"{rule_outdir}/tsm_automation_quality_checks.csv",
            "--operational-quality",
            f"{rule_outdir}/tsm_operational_quality_checks.csv",
            "--outdir",
            rule_outdir,
        ],
    }
    if mode == "data_pipeline":
        if bool_payload(payload, "skip_charts"):
            commands[mode].append("--skip-charts")
        if bool_payload(payload, "skip_benchmarks"):
            commands[mode].append("--skip-benchmarks")
    if mode == "prediction_engine" and bool_payload(payload, "schema_strict"):
        commands[mode].append("--schema-strict")
    if mode == "pooled_universe_update":
        if bool_payload(payload, "skip_benchmarks"):
            commands[mode].append("--skip-benchmarks")
        if bool_payload(payload, "continue_on_error"):
            commands[mode].append("--continue-on-error")
        if bool_payload(payload, "skip_symbol_build"):
            commands[mode].append("--skip-symbol-build")
        if bool_payload(payload, "skip_symbol_diagnostics"):
            commands[mode].append("--skip-symbol-diagnostics")
    if mode not in commands:
        raise ValueError(f"unknown run mode: {mode}")
    return mode, commands[mode]


def append_run_log(line: str) -> None:
    with RUN_LOCK:
        log = RUN_STATE.setdefault("log", [])
        log.append(line.rstrip("\n"))
        if len(log) > 500:
            del log[: len(log) - 500]


def get_run_state() -> dict[str, Any]:
    with RUN_LOCK:
        return clean_json(dict(RUN_STATE))


def run_command_thread(mode: str, command: list[str]) -> None:
    global RUN_PROCESS
    started = time.perf_counter()
    clear_summary_cache()
    try:
        with RUN_LOCK:
            RUN_STATE.update(
                {
                    "running": True,
                    "mode": mode,
                    "mode_label": MODE_LABELS.get(mode, mode),
                    "command": command,
                    "started_at": now_iso(),
                    "ended_at": None,
                    "duration_sec": None,
                    "returncode": None,
                    "status": "RUNNING",
                    "log": [f"$ {' '.join(command)}"],
                }
            )
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=os.environ.copy(),
        )
        with RUN_LOCK:
            RUN_PROCESS = proc
        if proc.stdout:
            for line in proc.stdout:
                append_run_log(line)
        returncode = proc.wait()
        status = "PASS" if returncode == 0 else "FAIL"
    except Exception as exc:
        returncode = -1
        status = "FAIL"
        append_run_log(f"ERROR: {exc}")
    finally:
        clear_summary_cache()
        with RUN_LOCK:
            RUN_PROCESS = None
            RUN_STATE.update(
                {
                    "running": False,
                    "ended_at": now_iso(),
                    "duration_sec": round(time.perf_counter() - started, 2),
                    "returncode": returncode,
                    "status": status,
                }
            )


def start_run(payload: dict[str, Any]) -> dict[str, Any]:
    mode, command = command_for_mode(payload)
    with RUN_LOCK:
        if RUN_STATE.get("running"):
            raise RuntimeError("a pipeline run is already active")
    thread = threading.Thread(target=run_command_thread, args=(mode, command), daemon=True)
    thread.start()
    time.sleep(0.05)
    return get_run_state()


def stop_run() -> dict[str, Any]:
    with RUN_LOCK:
        proc = RUN_PROCESS
    if proc and proc.poll() is None:
        append_run_log("Stopping process...")
        proc.terminate()
    return get_run_state()


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "TSMDashboard/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(clean_json(data), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_error_json(self, message: str, status: int = 400) -> None:
        self.send_json({"error": message}, status=status)

    def send_static(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        raw = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        if not raw:
            return {}
        return json.loads(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path in {"/", "/index.html"}:
                self.send_static(STATIC_DIR / "index.html")
                return
            if path == "/styles.css":
                self.send_static(STATIC_DIR / "styles.css")
                return
            if path == "/app.js":
                self.send_static(STATIC_DIR / "app.js")
                return
            if path == "/ontology.js":
                self.send_static(STATIC_DIR / "ontology.js")
                return
            if path == "/ontology_renderers.js":
                self.send_static(STATIC_DIR / "ontology_renderers.js")
                return
            if path in {"/api/summary", "/api/data"}:
                self.send_json(cached_build_summary())
                return
            if path == "/api/investor-dashboard":
                self.send_json(cached_build_investor_dashboard())
                return
            if path == "/api/live-quotes":
                self.send_json(build_live_quotes())
                return
            if path == "/api/portfolio":
                self.send_json(build_manual_portfolio_dashboard())
                return
            if path == "/api/files":
                self.send_json({"files": list_output_files()})
                return
            if path == "/api/run-status":
                self.send_json(get_run_state())
                return
            if path == "/api/table":
                file_name = query.get("file", [""])[0]
                limit = int(query.get("limit", ["250"])[0])
                offset = int(query.get("offset", ["0"])[0])
                tail = query.get("tail", ["false"])[0].lower() == "true"
                self.send_json(read_table_response(file_name, limit=max(1, min(limit, 1000)), offset=offset, tail=tail))
                return
            if path == "/api/report":
                file_name = query.get("file", [""])[0]
                self.send_json(read_report_response(file_name))
                return
            if path == "/api/symbol":
                sym = query.get("sym", [""])[0]
                self.send_json(build_symbol_summary(sym))
                return
            if path == "/api/file":
                file_name = query.get("file", [""])[0]
                file_path = resolve_allowed_file(file_name)
                self.send_static(file_path)
                return
            self.send_error(404)
        except FileNotFoundError as exc:
            self.send_error_json(str(exc), status=404)
        except Exception as exc:
            self.send_error_json(str(exc), status=400)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/run":
                self.send_json(start_run(self.read_json_body()), status=202)
                return
            if parsed.path == "/api/stop":
                self.send_json(stop_run())
                return
            if parsed.path == "/api/fx/refresh":
                payload = self.read_json_body()
                self.send_json(refresh_koreaexim_fx(payload.get("searchdate") or payload.get("search_date")))
                return
            if parsed.path == "/api/portfolio/transaction":
                self.send_json(apply_portfolio_mutation(self.read_json_body()))
                return
            self.send_error(404)
        except RuntimeError as exc:
            self.send_error_json(str(exc), status=409)
        except Exception as exc:
            self.send_error_json(str(exc), status=400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Top10 semiconductor dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--strict-port", action="store_true", help="Fail instead of trying the next port when the requested port is busy.")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in the default browser.")
    return parser.parse_args()


def bind_server(host: str, port: int, strict_port: bool) -> tuple[ThreadingHTTPServer, int]:
    candidate_ports = [port] if strict_port else list(range(port, port + 25))
    last_error: OSError | None = None
    for candidate in candidate_ports:
        try:
            return ThreadingHTTPServer((host, candidate), DashboardHandler), candidate
        except OSError as exc:
            last_error = exc
            if exc.errno != errno.EADDRINUSE:
                raise
            if strict_port:
                raise
            print(f"Port {candidate} is already in use; trying {candidate + 1}...")
    if last_error:
        raise last_error
    raise RuntimeError("No port candidates available.")


def main() -> None:
    args = parse_args()
    server, port = bind_server(args.host, args.port, args.strict_port)
    url = f"http://{args.host}:{port}"
    print(f"Top10 dashboard running at {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

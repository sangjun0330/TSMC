#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local dashboard for the TSMC price algorithm rule-engine package.

The dashboard keeps the existing research scripts as the source of truth. It
only reads their CSV/Markdown outputs and provides a small local API for
refreshing the full pipeline or individual engines.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import mimetypes
import os
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "dashboard"
OUTPUT_DIR = ROOT / "output"
RULE_DIR = ROOT / "tsm_price_rule_output"
ALLOWED_FILE_ROOTS = [OUTPUT_DIR.resolve(), RULE_DIR.resolve()]

DEFAULT_START = "2016-05-12"
DEFAULT_END = date.today().isoformat()


RUN_LOCK = threading.Lock()
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
    "pooled_universe_update": "반도체 유니버스 업데이트",
    "system_state_engine": "시스템 준비도",
    "daily_trading_report": "매매 계획",
    "daily_health_report": "일일 헬스 리포트",
}

DEFAULT_UNIVERSE_CONFIG = "config/semiconductor_universe_expanded.csv"
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
                    "rows": line_count(path),
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
    enriched = safe_read_csv(OUTPUT_DIR / "tsm_daily_10y_enriched.csv")
    hourly = safe_read_csv(OUTPUT_DIR / "tsm_hourly_available_enriched.csv")
    minute = safe_read_csv(OUTPUT_DIR / "tsm_minute_available_enriched.csv")
    signals = safe_read_csv(RULE_DIR / "tsm_daily_algorithmic_signals.csv")
    equity = safe_read_csv(RULE_DIR / "tsm_backtest_equity_curves.csv")
    risk = _normalize_risk_weight_columns(safe_read_csv(RULE_DIR / "tsm_risk_policy_daily.csv"))
    backtest = safe_read_csv(RULE_DIR / "tsm_backtest_strategy_summary.csv")
    trade_log = safe_read_csv(RULE_DIR / "tsm_backtest_trade_log.csv")
    yearly = safe_read_csv(RULE_DIR / "tsm_backtest_yearly_returns.csv")
    prediction_comparison = safe_read_csv(RULE_DIR / "tsm_prediction_model_comparison.csv")
    prediction_audit = safe_read_csv(RULE_DIR / "tsm_prediction_model_audit.csv")
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
    stress_scenarios = safe_read_csv(RULE_DIR / "tsm_daily_stress_scenarios.csv")
    strategy_stress = safe_read_csv(RULE_DIR / "tsm_strategy_stress_summary.csv")
    readiness = safe_read_csv(RULE_DIR / "tsm_system_readiness_scorecard.csv")
    trading_plan = safe_read_csv(RULE_DIR / "tsm_daily_trading_plan.csv")
    manifest = safe_read_csv(RULE_DIR / "tsm_daily_update_manifest.csv")
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
    hourly_audit = safe_read_csv(OUTPUT_DIR / "tsm_hourly_10y_source_audit.csv")
    minute_audit = safe_read_csv(OUTPUT_DIR / "tsm_minute_10y_source_audit.csv")
    prediction_oos = safe_read_csv(RULE_DIR / "tsm_prediction_oos_predictions.csv")
    prediction_calibration_summary = safe_read_csv(RULE_DIR / "tsm_prediction_calibration_summary.csv")
    prediction_policy_audit = safe_read_csv(RULE_DIR / "tsm_prediction_policy_audit.csv")
    prediction_feature_contract = safe_read_csv(RULE_DIR / "tsm_prediction_feature_contract.csv")
    prediction_fold_manifest = safe_read_csv(RULE_DIR / "tsm_prediction_fold_manifest.csv")
    prediction_model_registry = safe_read_csv(RULE_DIR / "tsm_prediction_model_registry.csv")
    prediction_experiment_log = safe_read_csv(RULE_DIR / "tsm_prediction_experiment_log.csv")
    schema_quality = safe_read_csv(RULE_DIR / "tsm_schema_quality_checks.csv")
    cscv_pbo = safe_read_csv(RULE_DIR / "tsm_cscv_pbo_report.csv")
    ml_overlay_summary = safe_read_csv(RULE_DIR / "tsm_ml_overlay_summary.csv")
    ml_overlay_quality = safe_read_csv(RULE_DIR / "tsm_ml_overlay_quality_checks.csv")
    ml_overlay_equity = safe_read_csv(RULE_DIR / "tsm_ml_overlay_equity_curves.csv")
    pooled_model_comparison = safe_read_csv(RULE_DIR / "tsm_pooled_model_comparison.csv")
    pooled_model_quality = safe_read_csv(RULE_DIR / "tsm_pooled_model_quality_checks.csv")
    pooled_tsm_calibration = safe_read_csv(RULE_DIR / "tsm_pooled_tsm_calibration.csv")
    pooled_tsm_calibration_metrics = safe_read_csv(RULE_DIR / "tsm_pooled_tsm_calibration_metrics.csv")
    pooled_threshold_policy = safe_read_csv(RULE_DIR / "tsm_pooled_model_threshold_policy.csv")
    pooled_uplift_bootstrap = safe_read_csv(RULE_DIR / "tsm_pooled_uplift_bootstrap_report.csv")
    pooled_oos_predictions = safe_read_csv(RULE_DIR / "tsm_pooled_model_oos_predictions.csv")
    pooled_oof_predictions = safe_read_csv(RULE_DIR / "tsm_pooled_model_oof_predictions.csv")
    pooled_slice_diagnostics = safe_read_csv(RULE_DIR / "tsm_pooled_model_slice_diagnostics.csv")
    pooled_dataset_quality = safe_read_csv(RULE_DIR / "tsm_prediction_pooled_quality_checks.csv")
    pooled_scope_stats = safe_read_csv(RULE_DIR / "tsm_prediction_pooled_scope_stats.csv")
    pooled_schema = safe_read_csv(RULE_DIR / "tsm_prediction_pooled_schema.csv")
    pooled_input_failures = safe_read_csv(RULE_DIR / "tsm_prediction_pooled_input_failures.csv")
    pooled_universe_config = safe_read_csv(RULE_DIR / "tsm_prediction_pooled_universe_config.csv")
    pooled_split_manifest = safe_read_csv(RULE_DIR / "tsm_prediction_pooled_split_manifest.csv")
    pooled_universe_manifest = safe_read_csv(RULE_DIR / "tsm_pooled_universe_update_manifest.csv")
    universe_validation = safe_read_csv(RULE_DIR / "tsm_universe_validation_report.csv")
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
    data_quality_checks = safe_read_csv(RULE_DIR / "tsm_data_quality_checks.csv")
    data_quality_issues = safe_read_csv(RULE_DIR / "tsm_data_quality_issues.csv")
    model_gate_audit = safe_read_csv(RULE_DIR / "tsm_model_gate_audit.csv")
    model_gate_root_causes = safe_read_csv(RULE_DIR / "tsm_model_gate_root_causes.csv")
    system_block_reasons = safe_read_csv(RULE_DIR / "tsm_system_block_reasons.csv")
    backtest_event_ledger = safe_read_csv(RULE_DIR / "tsm_backtest_event_ledger.csv")
    backtest_feedback_features = safe_read_csv(RULE_DIR / "tsm_backtest_feedback_features.csv")
    backtest_feedback_quality = safe_read_csv(RULE_DIR / "tsm_backtest_feedback_quality_checks.csv")
    cpcv_path_summary = safe_read_csv(RULE_DIR / "tsm_cpcv_path_summary.csv")
    cpcv_strategy_distribution = safe_read_csv(RULE_DIR / "tsm_cpcv_strategy_distribution.csv")
    cpcv_model_distribution = safe_read_csv(RULE_DIR / "tsm_cpcv_model_distribution.csv")
    auto_research_trials = safe_read_csv(RULE_DIR / "tsm_auto_research_trial_ledger.csv")
    auto_research_best = safe_read_csv(RULE_DIR / "tsm_auto_research_best_candidates.csv")
    research_expansion_manifest = safe_read_csv(RULE_DIR / "tsm_research_expansion_manifest.csv")

    price_columns = [
        "date",
        "close",
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
    intraday_price_columns = [
        "date",
        "close",
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
        "trend_regime",
        "vol_regime",
    ]
    signal_columns = [
        "date",
        "close",
        "score_price_algo_total",
        "entry_trigger",
        "trade_action",
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
    pooled_oos_columns = [
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
    pooled_oof_columns = [
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
    event_ledger_columns = [
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
    feedback_columns = [
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
    latest_hourly = latest_row_map(OUTPUT_DIR / "tsm_hourly_available_enriched.csv")
    latest_minute = latest_row_map(OUTPUT_DIR / "tsm_minute_available_enriched.csv")
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
                "hourly_summary": field_value_map(OUTPUT_DIR / "tsm_hourly_available_summary.csv"),
                "minute_summary": field_value_map(OUTPUT_DIR / "tsm_minute_available_summary.csv"),
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
                "paper_gate": field_value_map(RULE_DIR / "tsm_paper_gate_snapshot.csv"),
                "universe_validation": universe_validation_snapshot(universe_validation),
                "pooled_sample_audit": sample_audit_snapshot(pooled_sample_audit),
                "risk": _normalize_risk_snapshot_fields(field_value_map(RULE_DIR / "tsm_latest_risk_snapshot.csv")),
                "prediction": _normalize_prediction_snapshot_fields(field_value_map(RULE_DIR / "tsm_latest_prediction_snapshot.csv")),
                "pooled_prediction": field_value_map(RULE_DIR / "tsm_pooled_latest_prediction_overlay.csv"),
                "stress": field_value_map(RULE_DIR / "tsm_latest_stress_snapshot.csv"),
                "integrity": field_value_map(RULE_DIR / "tsm_latest_integrity_snapshot.csv"),
                "latest_news": latest_row_map(RULE_DIR / "tsm_news_integrated_daily.csv"),
                "latest_prediction_features": latest_prediction_features,
                "news_penalty_summary": news_penalty_summary,
                "latest_price": latest_price,
                "latest_hourly": latest_hourly,
                "latest_minute": latest_minute,
            },
            "quality": {
                "operational": quality_summary(RULE_DIR / "tsm_operational_quality_checks.csv"),
                "integrity": quality_summary(RULE_DIR / "tsm_daily_integrity_checks.csv", critical_only=True),
                "validation": quality_summary(RULE_DIR / "tsm_validation_quality_checks.csv"),
                "prediction": quality_summary(RULE_DIR / "tsm_prediction_quality_checks.csv", critical_only=True),
                "data_contract": mismatch_summary(RULE_DIR / "tsm_data_validation_raw_vs_enriched.csv"),
                "schema": quality_summary(RULE_DIR / "tsm_schema_quality_checks.csv", critical_only=True),
                "ml_overlay": quality_summary(RULE_DIR / "tsm_ml_overlay_quality_checks.csv", critical_only=True),
                "pooled_dataset": quality_summary(RULE_DIR / "tsm_prediction_pooled_quality_checks.csv", critical_only=True),
                "pooled_model": quality_summary(RULE_DIR / "tsm_pooled_model_quality_checks.csv", critical_only=True),
                "shadow_paper": quality_summary(RULE_DIR / "tsm_shadow_paper_quality_checks.csv", critical_only=True),
                "data_quality": quality_summary(RULE_DIR / "tsm_data_quality_checks.csv", critical_only=True),
                "model_gate": quality_summary(RULE_DIR / "tsm_model_gate_audit.csv", critical_only=True),
                "backtest_feedback": quality_summary(RULE_DIR / "tsm_backtest_feedback_quality_checks.csv", critical_only=True),
            },
            "series": {
                "price": records_from_df(enriched, price_columns, tail=650),
                "hourly": records_from_df(hourly, intraday_price_columns, tail=650),
                "minute": records_from_df(minute, intraday_price_columns, tail=780),
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
                "validation_walk_forward": records_from_df(walk_forward),
                "validation_causal_walk_forward": records_from_df(causal_walk_forward),
                "validation_segments": records_from_df(validation_segments),
                "validation_sensitivity": records_from_df(sensitivity),
                "stress_scenarios": records_from_df(stress_scenarios),
                "strategy_stress": records_from_df(strategy_stress),
                "readiness_scorecard": records_from_df(readiness),
                "manifest": records_from_df(manifest.tail(40) if not manifest.empty else manifest),
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
                "pooled_split_manifest": records_from_df(pooled_split_manifest),
                "pooled_universe_manifest": records_from_df(pooled_universe_manifest.tail(80) if not pooled_universe_manifest.empty else pooled_universe_manifest),
                "universe_validation": records_from_df(universe_validation),
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
                "data_quality_checks": records_from_df(data_quality_checks),
                "data_quality_issues": records_from_df(data_quality_issues),
                "model_gate_audit": records_from_df(model_gate_audit.tail(500) if not model_gate_audit.empty else model_gate_audit),
                "model_gate_root_causes": records_from_df(model_gate_root_causes),
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
                "hourly_source_audit": records_from_df(hourly_audit),
                "minute_source_audit": records_from_df(minute_audit),
            },
            "files": files,
            "reports": reports,
            "images": images,
            "run": get_run_state(),
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
    provider = str_payload(payload, "provider", "auto")
    interval = str_payload(payload, "interval", "1m")
    config = str_payload(payload, "config", "config/tsm_research.toml")
    universe_config = str_payload(payload, "universe_config", DEFAULT_UNIVERSE_CONFIG)
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
            "tsm_hourly_quant_pipeline.py",
            "--start",
            start,
            "--end",
            end,
            "--provider",
            provider,
            "--outdir",
            output_dir,
        ]
        if bool_payload(payload, "skip_charts"):
            cmd.append("--skip-charts")
        return mode, cmd

    if mode == "intraday":
        cmd = [
            py,
            "tsm_intraday_quant_pipeline.py",
            "--start",
            start,
            "--end",
            end,
            "--interval",
            interval,
            "--outdir",
            output_dir,
        ]
        if bool_payload(payload, "skip_charts"):
            cmd.append("--skip-charts")
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
        ],
        "external_feature_engine": [
            py,
            "tsm_external_feature_engine.py",
            "--outdir",
            rule_outdir,
            "--universe-config",
            universe_config,
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
            universe_config,
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
        "pooled_universe_update": [
            py,
            "run_pooled_universe_update.py",
            "--universe-config",
            universe_config,
            "--outdir",
            rule_outdir,
            "--start",
            start,
            "--end",
            end,
            "--preferred-source",
            preferred_source,
            "--commission-bps",
            commission_bps,
            "--slippage-bps",
            slippage_bps,
            "--stop-multiple",
            stop_multiple,
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
            if path == "/api/summary":
                self.send_json(build_summary())
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
            self.send_error(404)
        except RuntimeError as exc:
            self.send_error_json(str(exc), status=409)
        except Exception as exc:
            self.send_error_json(str(exc), status=400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local TSMC dashboard.")
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
    print(f"TSMC dashboard running at {url}")
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

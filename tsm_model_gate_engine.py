#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model gate audit for TSM prediction and pooled overlays.

This engine does not train models. It reads the model comparison artifacts and
latest prediction snapshots, then expands each decision-support requirement
into explicit pass/fail rows.

Outputs:
- tsm_model_gate_audit.csv
- tsm_model_gate_snapshot.csv
- tsm_model_gate_root_causes.csv
- tsm_model_gate_report.md
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from tsm_core.decision_schema import is_prediction_decision_support
from tsm_core.io import strip_bom_columns


MIN_DECISION_OOS_EVENTS = 100
MIN_SELECTED_OOS_EVENTS = 50
MIN_SELECTED_EVENTS_PER_FOLD = 10
DECISION_ECE_THRESHOLD = 0.10
MIN_POSITIVE_EXPECTANCY_FOLDS = 4
MIN_CALIBRATION_BIN_N = 30
MAX_THRESHOLD_IQR = 0.10
MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT = 0.0
MIN_EXPECTANCY_IMPROVEMENT_PCT = 0.0

MIN_POOLED_EVAL_EVENTS = 150
MIN_POOLED_SELECTED_EVENTS = 50
MAX_POOLED_ECE = 0.10
MAX_TSM_ECE = 0.15
MAX_STOP_HIT_FOR_LATEST = 0.35
MIN_EXPECTED_R_FOR_LATEST = 0.35
MIN_TSM_LIKE_EFFECTIVE_CALIBRATION_N = 500
MAX_PBO = 0.50
MAX_CPCV_WORST_QUARTILE_DRAWDOWN_PCT = -35.0

ROOT_CAUSE_GUIDANCE = {
    "NOT_20D_TRADE_READY_DECISION_SCOPE": (
        1,
        "system_sample_evidence",
        "20일 매수 판단 후보 진단 행을 먼저 확인하고, 결정 사용은 충분한 표본 확보 뒤에만 허용합니다.",
    ),
    "OOS_EVENT_COUNT_LT_MIN": (
        1,
        "system_sample_evidence",
        "더 긴 기간, 여러 종목 표본, 소표본 진단으로 학습 외 검증 사례를 먼저 늘립니다.",
    ),
    "OOS_EVENT_COUNT_LT_100": (
        1,
        "system_sample_evidence",
        "local TSM 단독 20D OOS가 100건 미만이면 의사결정용으로 승격하지 않습니다.",
    ),
    "SELECTED_OOS_EVENT_COUNT_LT_MIN": (
        1,
        "system_sample_evidence",
        "예측이 고른 테스트 사례가 최소 기준을 채운 뒤에만 선택 기준을 조정합니다.",
    ),
    "SELECTED_OOS_EVENT_COUNT_LT_50": (
        1,
        "system_sample_evidence",
        "local TSM 선택 OOS가 50건 미만이면 선택 threshold 근거가 부족합니다.",
    ),
    "SELECTED_EVENTS_PER_FOLD_LT_MIN": (
        1,
        "system_sample_evidence",
        "각 시간순 검증 묶음에서 선택 사례가 충분히 쌓이기 전까지 기준 성과를 신뢰하지 않습니다.",
    ),
    "SELECTED_EVENTS_PER_FOLD_LT_10": (
        1,
        "system_sample_evidence",
        "fold별 선택 사례가 10건 미만이면 시간순 반복성이 부족합니다.",
    ),
    "CALIBRATION_MIN_BIN_N_LT_MIN": (
        1,
        "system_calibration",
        "확률 구간별 표본을 늘리고, 충분히 채워질 때까지 결정 게이트는 차단 상태로 둡니다.",
    ),
    "CALIBRATION_MIN_BIN_N_LT_30": (
        1,
        "system_calibration",
        "decision calibration bin의 최소 표본이 30건 미만이면 확률 신뢰도가 부족합니다.",
    ),
    "FIXED_WIDTH_CALIBRATION_WARN": (
        3,
        "system_calibration",
        "fixed-width calibration bin 실패는 진단 경고로 남기되, 판단 gate는 adaptive bin 기준을 사용합니다.",
    ),
    "ECE_GT_LIMIT": (
        2,
        "system_calibration",
        "p_success를 판단 확률로 쓰기 전에 확률 보정을 먼저 개선합니다.",
    ),
    "ECE_GT_0_10": (
        2,
        "system_calibration",
        "decision ECE가 0.10 이하로 내려오기 전까지 확률 gate를 통과시키지 않습니다.",
    ),
    "PREDICTION_QUALITY_FALSE": (
        2,
        "system_sample_evidence",
        "모델 비교 행에서 어떤 세부 기준이 품질 미달을 만든 것인지 확인합니다.",
    ),
    "LOCAL_MODEL_COMPARISON_MISSING": (
        1,
        "system_sample_evidence",
        "local model comparison 산출물이 없으면 단독 TSM 모델 gate를 평가할 수 없습니다.",
    ),
    "DIAGNOSTIC_ONLY_INSUFFICIENT_SAMPLE": (
        1,
        "system_sample_evidence",
        "local TSM 단독 표본이 부족한 동안 해당 모델은 display-only 진단 경로로 유지합니다.",
    ),
    "TREE_OR_FULL_FEATURE_MODEL_RESEARCH_ONLY_SMALL_SAMPLE": (
        2,
        "system_sample_evidence",
        "소표본에서 복잡한 feature 모델은 연구용으로만 유지하고 decision support에는 쓰지 않습니다.",
    ),
    "NO_BRIER_IMPROVEMENT": (
        2,
        "system_economic_uplift",
        "단순 기본 확률보다 좋아질 때까지 모델은 진단용으로만 둡니다.",
    ),
    "PR_AUC_NOT_ABOVE_BASE": (
        2,
        "system_economic_uplift",
        "미래 정보가 없는 유효 특징을 보강하거나, 구분력이 개선될 때까지 단순 모델을 우선합니다.",
    ),
    "ML_SELECTED_MINUS_RULE_ALL_LE_0": (
        2,
        "system_economic_uplift",
        "예측으로 고른 결과가 기본 규칙 전체보다 좋아질 때까지 필터로 쓰지 않습니다.",
    ),
    "SELECTED_EXPECTANCY_CI_LOWER_LE_0": (
        2,
        "system_economic_uplift",
        "보수적으로 본 기대수익 하단이 양수로 확인될 때까지 판단 지원을 열지 않습니다.",
    ),
    "POSITIVE_EXPECTANCY_FOLDS_LT_MIN": (
        2,
        "system_economic_uplift",
        "한 구간의 우연한 수익이 아니라 여러 검증 구간에서 반복되는지 확인합니다.",
    ),
    "POSITIVE_EXPECTANCY_FOLDS_LT_4": (
        2,
        "system_economic_uplift",
        "양수 기대수익 fold가 4개 미만이면 반복성 근거가 부족합니다.",
    ),
    "THRESHOLD_IQR_GT_LIMIT": (
        3,
        "system_threshold_stability",
        "검증 구간별 선택 기준이 안정될 때까지 고정 기준이나 진단용 기준으로만 봅니다.",
    ),
    "THRESHOLD_IQR_GT_0_10": (
        3,
        "system_threshold_stability",
        "threshold IQR이 0.10을 넘으면 fold별 기준이 불안정하므로 진단용으로만 둡니다.",
    ),
    "POOLED_NO_BRIER_IMPROVEMENT": (
        2,
        "system_economic_uplift",
        "여러 종목 모델을 다시 보정하거나 더 보수적인 사전확률을 사용합니다.",
    ),
    "POOLED_EVENT_COUNT_LT_MIN": (
        1,
        "system_sample_evidence",
        "여러 종목 OOF 평가 표본이 최소 기준을 채울 때까지 pooled 모델을 판단 지원으로 승격하지 않습니다.",
    ),
    "POOLED_SELECTED_EVENT_COUNT_LT_MIN": (
        1,
        "system_sample_evidence",
        "검증 구간에서 모델이 선택한 사건 수가 최소 기준을 채운 뒤에만 threshold를 판단 기준으로 사용합니다.",
    ),
    "POOLED_ECE_GT_LIMIT": (
        2,
        "system_calibration",
        "여러 종목 확률 오차를 낮추기 전까지 판단 지원으로 쓰지 않습니다.",
    ),
    "POOLED_DECISION_CALIBRATION_BIN_N_LT_30": (
        2,
        "system_calibration",
        "adaptive equal-frequency calibration bin이 최소 30개 표본을 채우기 전까지 확률 gate를 통과시키지 않습니다.",
    ),
    "POOLED_SELECTED_MINUS_ALL_LE_0": (
        2,
        "system_economic_uplift",
        "여러 종목 선택 결과가 전체 결과보다 좋아질 때까지 필터 사용을 차단합니다.",
    ),
    "POOLED_SELECTED_MINUS_ALL_CI_LOWER_LE_0": (
        2,
        "system_economic_uplift",
        "bootstrap 하단 기준에서도 선택 결과가 전체 rule events보다 나아질 때까지 필터 사용을 차단합니다.",
    ),
    "POOLED_SELECTED_MINUS_SCORE_BASELINE_CI_LOWER_LE_0": (
        2,
        "system_economic_uplift",
        "score-only baseline 대비 bootstrap 하단 uplift가 양수가 될 때까지 모델 필터 승격을 보류합니다.",
    ),
    "POOLED_SELECTED_EXPECTANCY_CI_LOWER_LE_0": (
        2,
        "system_economic_uplift",
        "선택된 pooled events 자체의 보수적 기대수익 하단이 양수가 될 때까지 판단 지원을 열지 않습니다.",
    ),
    "POOLED_POSITIVE_EXPECTANCY_FOLDS_LT_4": (
        2,
        "system_economic_uplift",
        "워크포워드 fold 대부분에서 양수 기대수익 하단이 반복될 때까지 판단 지원으로 쓰지 않습니다.",
    ),
    "POOLED_TRIAL_LEDGER_MISSING": (
        2,
        "system_economic_uplift",
        "threshold, model, utility-weight 탐색 횟수 ledger가 기록되어야 backtest overfitting 위험을 추적할 수 있습니다.",
    ),
    "POOLED_UPLIFT_NOT_PASSED": (
        2,
        "system_economic_uplift",
        "all-events와 score-only baseline 대비 bootstrap 하단 및 fold 반복성이 모두 양수가 될 때까지 판단 지원을 보류합니다.",
    ),
    "POOLED_SELECTED_FRACTION_OUT_OF_RANGE": (
        2,
        "system_threshold_stability",
        "선택 비율이 30%~60% 범위에 들어오도록 decision score 기준을 재조정합니다.",
    ),
    "POOLED_THRESHOLD_NOT_DECISION_ELIGIBLE": (
        2,
        "system_threshold_stability",
        "strict 후보 전용 threshold가 독립 검증 구간에서 통과하기 전까지 최신 판단 지원을 차단합니다.",
    ),
    "POOLED_THRESHOLD_STABILITY_FAILED": (
        2,
        "system_threshold_stability",
        "OOF fold별 선택 수, 선택 비율, threshold IQR이 안정 조건을 모두 통과해야 threshold를 판단용으로 인정합니다.",
    ),
    "POOLED_OOF_FOLD_SELECTED_EVENTS_LT_10": (
        2,
        "system_threshold_stability",
        "워크포워드 각 테스트 fold에서 선택 사례가 최소 10개 이상 나오도록 threshold 안정성을 먼저 개선합니다.",
    ),
    "TSM_CALIBRATION_ECE_GT_LIMIT": (
        1,
        "system_calibration",
        "TSMC 전용 확률 조정을 믿기 전에 TSMC 보류 검증 사례를 더 확보합니다.",
    ),
    "TSM_CALIBRATION_ECE_GT_0_15": (
        1,
        "system_calibration",
        "TSMC 전용 calibration ECE가 0.15를 넘으면 pooled overlay를 판단 지원으로 승격하지 않습니다.",
    ),
    "TSM_CALIBRATION_ROUTE_NOT_PASSED": (
        1,
        "system_calibration",
        "POOLED_ONLY, group/logit/Platt/blend route 중 pre-test에서 선택된 route가 ECE/Brier/holdout 조건을 통과할 때만 gate primary로 사용합니다.",
    ),
    "NO_MODEL": (
        1,
        "system_sample_evidence",
        "신뢰 가능한 20일 매수 판단 모델이 생길 때까지 최신 단독 예측은 표시용으로 둡니다.",
    ),
    "POOLED_MODEL_QUALITY_NOT_PASSED": (
        2,
        "system_sample_evidence",
        "여러 종목 모델의 확률 보정과 기대수익 기준이 통과될 때까지 참고용으로만 봅니다.",
    ),
    "POOLED_MODEL_COMPARISON_MISSING": (
        1,
        "system_sample_evidence",
        "pooled model comparison 산출물이 없으면 품질 gate를 평가할 수 없습니다.",
    ),
    "TSM_CALIBRATION_MISSING": (
        1,
        "system_calibration",
        "TSM 전용 calibration metrics 산출물이 없으면 pooled overlay 신뢰도를 판단할 수 없습니다.",
    ),
    "TSM_LIKE_EFFECTIVE_N_LT_500": (
        1,
        "system_calibration",
        "TSM-like calibration 유효 표본이 500건 이상 쌓이기 전까지 pooled overlay를 판단 지원으로 승격하지 않습니다.",
    ),
    "CPCV_DISTRIBUTION_MISSING": (
        1,
        "system_backtest_overfit_control",
        "CPCV path/model 분포 산출물을 만든 뒤 여러 OOS 경로에서 uplift가 반복되는지 확인합니다.",
    ),
    "CPCV_MEDIAN_UPLIFT_LE_0": (
        1,
        "system_backtest_overfit_control",
        "CPCV 경로 중앙값 uplift가 양수로 확인될 때까지 threshold/model 승격을 보류합니다.",
    ),
    "CPCV_WORST_QUARTILE_UPLIFT_LT_0": (
        1,
        "system_backtest_overfit_control",
        "CPCV 하위 25% 경로에서 손익 또는 drawdown 방어가 확인될 때까지 판단 지원으로 쓰지 않습니다.",
    ),
    "PBO_GE_0_50": (
        1,
        "system_backtest_overfit_control",
        "전략/모델 trial 수를 반영한 PBO가 0.5 미만으로 내려오기 전까지 과최적화 위험을 차단합니다.",
    ),
    "DSR_NOT_PASSED": (
        1,
        "system_backtest_overfit_control",
        "Deflated Sharpe가 통과되지 않으면 여러 trial을 거친 성과를 보수적으로 차단합니다.",
    ),
    "LATEST_NOT_TRADE_READY": (
        3,
        "price_timing_latest_signal",
        "최신 행이 매수 준비 신호가 아니면 예측값을 매매 판단으로 승격하지 않습니다.",
    ),
    "LATEST_CONTEXT_ONLY_NO_ENTRY_TRIGGER": (
        3,
        "price_timing_latest_signal",
        "최신 행에 매수 트리거가 없으면 모델 품질과 별개로 최신 판단 지원을 열지 않습니다.",
    ),
    "LATEST_RULE_FILTERED_NOT_TRADE_READY": (
        3,
        "price_timing_latest_signal",
        "최신 가격 신호가 rule gate에서 trade-ready가 아니면 시스템 품질 통과와 별개로 진입을 보류합니다.",
    ),
    "NO_ENTRY_TRIGGER": (
        3,
        "price_timing_latest_signal",
        "최신 가격 신호가 진입 트리거를 만들지 않았으므로 모델 차단과 분리해서 표시합니다.",
    ),
    "POOLED_STOP_RISK_GT_0_35": (
        2,
        "price_timing_latest_signal",
        "손절 확률이 위험 한도보다 높으면 여러 종목 최신 예측을 차단합니다.",
    ),
    "POOLED_STOP_RISK_GT_LIMIT": (
        2,
        "price_timing_latest_signal",
        "손절 확률이 위험 한도보다 높으면 여러 종목 최신 예측을 차단합니다.",
    ),
    "POOLED_EXPECTED_R_LT_0_35": (
        2,
        "price_timing_latest_signal",
        "기대 R이 최소 기준보다 낮으면 여러 종목 최신 예측을 차단합니다.",
    ),
    "POOLED_EXPECTED_R_LT_MIN": (
        2,
        "price_timing_latest_signal",
        "기대 R이 최소 기준보다 낮으면 여러 종목 최신 예측을 차단합니다.",
    ),
    "POOLED_DECISION_SCORE_BELOW_THRESHOLD": (
        2,
        "price_timing_latest_signal",
        "결정 점수가 검증에서 선택한 기준보다 낮으면 최신 예측을 차단합니다.",
    ),
    "POOLED_LATEST_SIGNAL_BLOCKED": (
        3,
        "price_timing_latest_signal",
        "pooled 모델 품질과 별개로 최신 가격/점수/위험 조건 중 하나가 통과하지 못했습니다.",
    ),
    "POOLED_LATEST_NOT_DECISION_SUPPORT": (
        3,
        "price_timing_latest_signal",
        "최신 pooled overlay가 decision_support_allowed 상태가 아니므로 표시용으로만 유지합니다.",
    ),
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return strip_bom_columns(pd.read_csv(path))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_snapshot(path: Path) -> dict[str, object]:
    df = read_csv(path)
    if df.empty or not {"field", "value"}.issubset(df.columns):
        return {}
    return dict(zip(df["field"].astype(str), df["value"]))


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def as_float(value: object, default: float = np.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def gate_row(
    gate_group: str,
    gate: str,
    passed: bool,
    value: object,
    threshold: object,
    block_reason: str,
    *,
    severity: str = "CRITICAL",
    candidate_scope: str = "",
    horizon_days: object = "",
    model_name: str = "",
    split: str = "",
) -> dict[str, object]:
    return {
        "gate_group": gate_group,
        "candidate_scope": candidate_scope,
        "horizon_days": horizon_days,
        "model_name": model_name,
        "split": split,
        "gate": gate,
        "passed": bool(passed),
        "severity": severity,
        "value": value,
        "threshold": threshold,
        "block_reason": "PASS" if passed else block_reason,
    }


def row_float(row: pd.Series, column: str, default: float = np.nan) -> float:
    return as_float(row.get(column, default), default)


def row_bool(row: pd.Series, column: str) -> bool:
    return to_bool(row.get(column, False))


def row_selected_fraction(row: pd.Series) -> float:
    value = row_float(row, "selected_fraction")
    if math.isfinite(value):
        return value
    selected_count = row_float(row, "selected_event_count")
    event_count = row_float(row, "event_count")
    if math.isfinite(selected_count) and math.isfinite(event_count) and event_count > 0:
        return selected_count / event_count
    return np.nan


def evaluate_local_model_row(row: pd.Series) -> list[dict[str, object]]:
    scope = str(row.get("candidate_scope", ""))
    horizon = row.get("horizon_days", "")
    model_name = str(row.get("model_name", ""))
    min_oos = row_float(row, "min_decision_oos_events", MIN_DECISION_OOS_EVENTS)
    min_selected = row_float(row, "min_selected_oos_events", MIN_SELECTED_OOS_EVENTS)
    base_pr_auc = row_float(row, "base_rate_pr_auc")
    pr_auc = row_float(row, "pr_auc")
    decision_ece = row_float(row, "decision_ece", row_float(row, "ece"))
    decision_min_bin_n = row_float(row, "decision_min_calibration_bin_n", row_float(row, "min_calibration_bin_n"))
    rows = [
        gate_row("local_prediction", "decision_scope_eligible", row_bool(row, "decision_scope_eligible"), row.get("decision_scope_eligible"), "True", "NOT_20D_TRADE_READY_DECISION_SCOPE", candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row("local_prediction", "prediction_quality_pass", row_bool(row, "prediction_quality_pass"), row.get("prediction_quality_pass"), "True", "PREDICTION_QUALITY_FALSE", candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row("local_prediction", "oos_event_count", row_float(row, "oos_event_count") >= min_oos, row_float(row, "oos_event_count"), f">={min_oos:.0f}", "OOS_EVENT_COUNT_LT_MIN", candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row("local_prediction", "selected_oos_event_count", row_float(row, "selected_oos_event_count") >= min_selected, row_float(row, "selected_oos_event_count"), f">={min_selected:.0f}", "SELECTED_OOS_EVENT_COUNT_LT_MIN", candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row("local_prediction", "min_selected_events_per_fold", row_float(row, "min_selected_events_per_fold") >= MIN_SELECTED_EVENTS_PER_FOLD, row_float(row, "min_selected_events_per_fold"), f">={MIN_SELECTED_EVENTS_PER_FOLD}", "SELECTED_EVENTS_PER_FOLD_LT_MIN", candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row("local_prediction", "brier_improvement_positive", row_float(row, "brier_improvement_pct") > 0.0, row_float(row, "brier_improvement_pct"), ">0", "NO_BRIER_IMPROVEMENT", candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row("local_prediction", "decision_ece_within_limit", decision_ece <= DECISION_ECE_THRESHOLD, decision_ece, f"<={DECISION_ECE_THRESHOLD}", "ECE_GT_LIMIT", candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row("local_prediction", "pr_auc_above_base", pd.notna(pr_auc) and pd.notna(base_pr_auc) and pr_auc > base_pr_auc, pr_auc, f">{base_pr_auc}", "PR_AUC_NOT_ABOVE_BASE", candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row("local_prediction", "selected_minus_rule_all_positive", row_float(row, "selected_minus_rule_all_pct") > MIN_EXPECTANCY_IMPROVEMENT_PCT, row_float(row, "selected_minus_rule_all_pct"), f">{MIN_EXPECTANCY_IMPROVEMENT_PCT}", "ML_SELECTED_MINUS_RULE_ALL_LE_0", candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row("local_prediction", "selected_expectancy_ci_lower_positive", row_float(row, "selected_signal_expectancy_ci_lower_pct") > MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT, row_float(row, "selected_signal_expectancy_ci_lower_pct"), f">{MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT}", "SELECTED_EXPECTANCY_CI_LOWER_LE_0", candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row("local_prediction", "positive_expectancy_folds", row_float(row, "positive_expectancy_folds") >= MIN_POSITIVE_EXPECTANCY_FOLDS, row_float(row, "positive_expectancy_folds"), f">={MIN_POSITIVE_EXPECTANCY_FOLDS}", "POSITIVE_EXPECTANCY_FOLDS_LT_MIN", candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row("local_prediction", "decision_min_calibration_bin_n", decision_min_bin_n >= MIN_CALIBRATION_BIN_N, decision_min_bin_n, f">={MIN_CALIBRATION_BIN_N}", "CALIBRATION_MIN_BIN_N_LT_MIN", candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row("local_prediction", "fixed_width_calibration_bin_n_diagnostic", row_float(row, "fixed_width_min_calibration_bin_n", decision_min_bin_n) >= MIN_CALIBRATION_BIN_N, row_float(row, "fixed_width_min_calibration_bin_n", decision_min_bin_n), f">={MIN_CALIBRATION_BIN_N}", "FIXED_WIDTH_CALIBRATION_WARN", severity="WARN", candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row("local_prediction", "threshold_iqr", row_float(row, "threshold_iqr") <= MAX_THRESHOLD_IQR, row_float(row, "threshold_iqr"), f"<={MAX_THRESHOLD_IQR}", "THRESHOLD_IQR_GT_LIMIT", candidate_scope=scope, horizon_days=horizon, model_name=model_name),
    ]
    return rows


def evaluate_local_models(comparison: pd.DataFrame, latest_prediction: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if comparison.empty:
        rows.append(gate_row("local_prediction", "comparison_available", False, "missing", "present", "LOCAL_MODEL_COMPARISON_MISSING"))
    else:
        for _, row in comparison.iterrows():
            rows.extend(evaluate_local_model_row(row))
    use_status = latest_prediction.get("prediction_use_status", "UNKNOWN")
    rows.append(
        gate_row(
            "local_latest",
            "latest_prediction_decision_support",
            is_prediction_decision_support(use_status),
            use_status,
            "DECISION_SUPPORT_ALLOWED",
            str(latest_prediction.get("model_quality_block_reasons", "LOCAL_LATEST_NOT_DECISION_SUPPORT")),
        )
    )
    return rows


def select_combined_pooled_row(pooled_comparison: pd.DataFrame) -> pd.DataFrame:
    if pooled_comparison.empty:
        return pooled_comparison
    preferred = pooled_comparison[pooled_comparison["split"].astype(str).eq("combined_test_holdout")]
    if "evaluation_scope" in preferred.columns:
        strict_scope = preferred[preferred["evaluation_scope"].astype(str).eq("trade_ready_entry_only")]
        if not strict_scope.empty:
            preferred = strict_scope
    if "validation_design" in preferred.columns:
        oof = preferred[preferred["validation_design"].astype(str).eq("walk_forward_oof")]
        if not oof.empty:
            preferred = oof
    if "is_champion" in preferred.columns:
        champion = preferred[preferred["is_champion"].map(to_bool)]
        if not champion.empty:
            return champion
    return preferred if not preferred.empty else pooled_comparison.tail(1)


def select_tsm_calibration_gate_row(tsm_calibration: pd.DataFrame) -> pd.DataFrame:
    if tsm_calibration.empty:
        return tsm_calibration
    preferred = tsm_calibration.copy()
    if "is_selected_tsm_calibration_route" in preferred.columns:
        selected = preferred[preferred["is_selected_tsm_calibration_route"].map(to_bool)].copy()
        if not selected.empty:
            preferred = selected
    if "split" in preferred.columns:
        combined = preferred[preferred["split"].astype(str).eq("tsm_combined_test_holdout")]
        if not combined.empty:
            return combined
    return preferred.tail(1)


def pooled_oof_fold_selection_stats(pooled_comparison: pd.DataFrame, row: pd.Series) -> tuple[float, str]:
    stored_min = row_float(row, "min_selected_events_per_fold")
    stored_weak = str(row.get("weak_oof_folds", row.get("weak_selected_event_folds", "")))
    if math.isfinite(stored_min):
        return stored_min, stored_weak
    if str(row.get("validation_design", "")) != "walk_forward_oof" or pooled_comparison.empty:
        return np.nan, ""
    model_name = str(row.get("model_name", ""))
    evaluation_scope = str(row.get("evaluation_scope", ""))
    split_values = pooled_comparison["split"].astype(str) if "split" in pooled_comparison.columns else pd.Series(dtype=str)
    fold_rows = pooled_comparison[
        split_values.str.startswith("oof_test_")
        & pooled_comparison.get("model_name", pd.Series("", index=pooled_comparison.index)).astype(str).eq(model_name)
        & pooled_comparison.get("evaluation_scope", pd.Series("", index=pooled_comparison.index)).astype(str).eq(evaluation_scope)
    ].copy()
    fold_rows = fold_rows[pd.to_numeric(fold_rows.get("event_count", pd.Series(dtype=float)), errors="coerce") > 0]
    if fold_rows.empty:
        return np.nan, ""
    selected_counts = pd.to_numeric(fold_rows.get("selected_event_count", pd.Series(dtype=float)), errors="coerce")
    weak_folds = fold_rows.loc[selected_counts < MIN_SELECTED_EVENTS_PER_FOLD, "split"].astype(str).tolist()
    return float(selected_counts.min()) if selected_counts.notna().any() else np.nan, "|".join(weak_folds)


def evaluate_pooled_models(pooled_comparison: pd.DataFrame, pooled_latest: dict[str, object], tsm_calibration: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selected = select_combined_pooled_row(pooled_comparison)
    if selected.empty:
        rows.append(gate_row("pooled_model", "pooled_comparison_available", False, "missing", "present", "POOLED_MODEL_COMPARISON_MISSING"))
    else:
        row = selected.iloc[0]
        model_name = str(row.get("model_name", ""))
        split = str(row.get("split", ""))
        selected_fraction = row_selected_fraction(row)
        min_selected_fold, weak_folds = pooled_oof_fold_selection_stats(pooled_comparison, row)
        validation_design = str(row.get("validation_design", ""))
        decision_ece = row_float(row, "decision_ece", row_float(row, "ece"))
        decision_min_bin_n = row_float(row, "decision_min_calibration_bin_n", row_float(row, "min_calibration_bin_n"))
        selected_minus_all_lower = row_float(row, "selected_minus_all_ci_lower_pct_paired", row_float(row, "selected_minus_all_ci_lower_pct"))
        selected_minus_score_lower = row_float(
            row,
            "selected_minus_score_baseline_ci_lower_pct_paired",
            row_float(row, "selected_minus_score_baseline_ci_lower_pct"),
        )
        uplift_pass = row_bool(row, "uplift_pass") if "uplift_pass" in row.index else (
            selected_minus_all_lower > 0.0
            and selected_minus_score_lower > 0.0
            and (validation_design != "walk_forward_oof" or row_float(row, "positive_expectancy_fold_count") >= MIN_POSITIVE_EXPECTANCY_FOLDS)
        )
        rows.extend(
            [
                gate_row("pooled_model", "event_count", row_float(row, "event_count") >= MIN_POOLED_EVAL_EVENTS, row_float(row, "event_count"), f">={MIN_POOLED_EVAL_EVENTS}", "POOLED_EVENT_COUNT_LT_MIN", model_name=model_name, split=split),
                gate_row("pooled_model", "selected_event_count", row_float(row, "selected_event_count") >= MIN_POOLED_SELECTED_EVENTS, row_float(row, "selected_event_count"), f">={MIN_POOLED_SELECTED_EVENTS}", "POOLED_SELECTED_EVENT_COUNT_LT_MIN", model_name=model_name, split=split),
                gate_row("pooled_model", "brier_improvement_positive", row_float(row, "brier_improvement_pct") > 0.0, row_float(row, "brier_improvement_pct"), ">0", "POOLED_NO_BRIER_IMPROVEMENT", model_name=model_name, split=split),
                gate_row("pooled_model", "decision_ece_within_limit", decision_ece <= MAX_POOLED_ECE, decision_ece, f"<={MAX_POOLED_ECE}", "POOLED_ECE_GT_LIMIT", model_name=model_name, split=split),
                gate_row("pooled_model", "decision_min_calibration_bin_n", decision_min_bin_n >= MIN_CALIBRATION_BIN_N, decision_min_bin_n, f">={MIN_CALIBRATION_BIN_N}", "POOLED_DECISION_CALIBRATION_BIN_N_LT_30", model_name=model_name, split=split),
                gate_row("pooled_model_diagnostic", "fixed_width_calibration_bin_n_diagnostic", row_float(row, "fixed_width_min_calibration_bin_n", decision_min_bin_n) >= MIN_CALIBRATION_BIN_N, row_float(row, "fixed_width_min_calibration_bin_n", decision_min_bin_n), f">={MIN_CALIBRATION_BIN_N}", "FIXED_WIDTH_CALIBRATION_WARN", severity="WARN", model_name=model_name, split=split),
                gate_row("pooled_model", "selected_fraction_30_to_60_pct", 0.30 <= selected_fraction <= 0.60, selected_fraction, "0.30..0.60", "POOLED_SELECTED_FRACTION_OUT_OF_RANGE", model_name=model_name, split=split),
                gate_row("pooled_model", "selected_minus_all_positive", row_float(row, "selected_minus_all_pct") > 0.0, row_float(row, "selected_minus_all_pct"), ">0", "POOLED_SELECTED_MINUS_ALL_LE_0", model_name=model_name, split=split),
                gate_row("pooled_model", "selected_minus_all_ci_lower_positive", selected_minus_all_lower > 0.0, selected_minus_all_lower, ">0 paired/block", "POOLED_SELECTED_MINUS_ALL_CI_LOWER_LE_0", model_name=model_name, split=split),
                gate_row("pooled_model", "selected_minus_score_baseline_ci_lower_positive", selected_minus_score_lower > 0.0, selected_minus_score_lower, ">0 paired/block", "POOLED_SELECTED_MINUS_SCORE_BASELINE_CI_LOWER_LE_0", model_name=model_name, split=split),
                gate_row("pooled_model", "selected_expectancy_ci_lower_positive", row_float(row, "selected_expectancy_ci_lower_pct") > 0.0, row_float(row, "selected_expectancy_ci_lower_pct"), ">0", "POOLED_SELECTED_EXPECTANCY_CI_LOWER_LE_0", model_name=model_name, split=split),
                gate_row("pooled_model", "economic_uplift_pass", uplift_pass, row.get("uplift_failure_reasons", uplift_pass), "True", "POOLED_UPLIFT_NOT_PASSED", model_name=model_name, split=split),
                gate_row("pooled_model", "threshold_decision_eligible", row_bool(row, "threshold_decision_eligible"), row.get("threshold_decision_eligible", False), "True", "POOLED_THRESHOLD_NOT_DECISION_ELIGIBLE", model_name=model_name, split=split),
                gate_row("pooled_model", "positive_expectancy_fold_count", validation_design != "walk_forward_oof" or row_float(row, "positive_expectancy_fold_count") >= MIN_POSITIVE_EXPECTANCY_FOLDS, row_float(row, "positive_expectancy_fold_count"), f">={MIN_POSITIVE_EXPECTANCY_FOLDS}", "POOLED_POSITIVE_EXPECTANCY_FOLDS_LT_4", model_name=model_name, split=split),
                gate_row("pooled_model", "trial_count_recorded", row_float(row, "trial_count", 0) > 0, row_float(row, "trial_count", 0), ">0", "POOLED_TRIAL_LEDGER_MISSING", model_name=model_name, split=split),
            ]
        )
        if validation_design == "walk_forward_oof":
            rows.append(
                gate_row(
                    "pooled_model",
                    "threshold_stability_pass",
                    row_bool(row, "threshold_stability_pass"),
                    f"{row.get('threshold_stability_pass', False)}; weak={row.get('weak_oof_folds', weak_folds)}",
                    "True",
                    "POOLED_THRESHOLD_STABILITY_FAILED",
                    model_name=model_name,
                    split=split,
                )
            )
            rows.append(
                gate_row(
                    "pooled_model",
                    "min_selected_events_per_oof_fold",
                    math.isfinite(min_selected_fold) and min_selected_fold >= MIN_SELECTED_EVENTS_PER_FOLD,
                    f"{min_selected_fold:g}; weak={weak_folds}" if math.isfinite(min_selected_fold) else "missing",
                    f">={MIN_SELECTED_EVENTS_PER_FOLD}",
                    "POOLED_OOF_FOLD_SELECTED_EVENTS_LT_10",
                    model_name=model_name,
                    split=split,
                )
            )

    if tsm_calibration.empty:
        rows.append(gate_row("pooled_tsm_calibration", "tsm_calibration_available", False, "missing", "present", "TSM_CALIBRATION_MISSING"))
    else:
        selected_tsm = select_tsm_calibration_gate_row(tsm_calibration)
        row = selected_tsm.iloc[0]
        tsm_ece = row_float(row, "decision_ece", row_float(row, "ece"))
        rows.extend(
            [
                gate_row(
                    "pooled_tsm_calibration",
                    "tsm_calibration_route_pass",
                    row_bool(row, "tsm_calibration_route_pass") if "tsm_calibration_route_pass" in row.index else tsm_ece <= MAX_TSM_ECE,
                    row.get("tsm_calibration_route_failure_reasons", row.get("tsm_calibration_route_pass", "missing")),
                    "True",
                    "TSM_CALIBRATION_ROUTE_NOT_PASSED",
                    model_name=str(row.get("model_name", "")),
                    split=str(row.get("split", "")),
                ),
                gate_row(
                    "pooled_tsm_calibration",
                    "tsm_ece_within_limit",
                    tsm_ece <= MAX_TSM_ECE,
                    tsm_ece,
                    f"<={MAX_TSM_ECE}",
                    "TSM_CALIBRATION_ECE_GT_LIMIT",
                    model_name=str(row.get("model_name", "")),
                    split=str(row.get("split", "")),
                ),
            ]
        )

    latest_signal_pass = to_bool(pooled_latest.get("latest_signal_pass", False))
    decision_allowed = to_bool(pooled_latest.get("decision_support_allowed", False))
    rows.extend(
        [
            gate_row("pooled_latest", "latest_signal_pass", latest_signal_pass, pooled_latest.get("latest_signal_pass", "missing"), "True", str(pooled_latest.get("latest_block_reasons", "POOLED_LATEST_SIGNAL_BLOCKED"))),
            gate_row("pooled_latest", "pooled_prediction_decision_support", decision_allowed, pooled_latest.get("decision_support_allowed", "missing"), "True", str(pooled_latest.get("latest_block_reasons", "POOLED_LATEST_NOT_DECISION_SUPPORT"))),
            gate_row("pooled_latest", "latest_stop_risk_within_limit", as_float(pooled_latest.get("p_stop_hit_20d")) <= MAX_STOP_HIT_FOR_LATEST, as_float(pooled_latest.get("p_stop_hit_20d")), f"<={MAX_STOP_HIT_FOR_LATEST}", "POOLED_STOP_RISK_GT_LIMIT"),
            gate_row("pooled_latest", "latest_expected_r_above_min", as_float(pooled_latest.get("expected_r_net_20d")) >= MIN_EXPECTED_R_FOR_LATEST, as_float(pooled_latest.get("expected_r_net_20d")), f">={MIN_EXPECTED_R_FOR_LATEST}", "POOLED_EXPECTED_R_LT_MIN"),
        ]
    )
    return rows


def selected_cpcv_distribution_row(cpcv_strategy: pd.DataFrame, cpcv_model: pd.DataFrame) -> tuple[str, pd.Series | None]:
    if not cpcv_model.empty:
        preferred = cpcv_model.copy()
        if "candidate_scope" in preferred.columns:
            strict = preferred[preferred["candidate_scope"].astype(str).str.contains("trade_ready|entry", case=False, na=False)]
            if not strict.empty:
                preferred = strict
        sort_col = "selected_minus_all_pct_median" if "selected_minus_all_pct_median" in preferred.columns else preferred.columns[-1]
        return "model", preferred.sort_values(sort_col, ascending=False).iloc[0]
    if not cpcv_strategy.empty:
        preferred = cpcv_strategy.copy()
        sort_col = "test_uplift_pct_median" if "test_uplift_pct_median" in preferred.columns else preferred.columns[-1]
        return "strategy", preferred.sort_values(sort_col, ascending=False).iloc[0]
    return "", None


def evaluate_research_validation(
    cpcv_strategy: pd.DataFrame,
    cpcv_model: pd.DataFrame,
    pbo_report: pd.DataFrame,
    cscv_pbo_report: pd.DataFrame,
    dsr_report: pd.DataFrame,
    tsm_calibration: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source, row = selected_cpcv_distribution_row(cpcv_strategy, cpcv_model)
    if row is None:
        rows.append(gate_row("research_validation", "cpcv_distribution_available", False, "missing", "present", "CPCV_DISTRIBUTION_MISSING"))
    else:
        if source == "model":
            median_uplift = row_float(row, "selected_minus_all_pct_median")
            q25_uplift = row_float(row, "selected_minus_all_pct_q25")
            drawdown = np.nan
            model_name = str(row.get("model_name", ""))
            scope = str(row.get("candidate_scope", ""))
        else:
            median_uplift = row_float(row, "test_uplift_pct_median")
            q25_uplift = row_float(row, "test_uplift_pct_q25")
            drawdown = row_float(row, "worst_test_drawdown_pct")
            model_name = str(row.get("strategy_id", ""))
            scope = "strategy"
        worst_quartile_or_drawdown_pass = (
            math.isfinite(q25_uplift)
            and q25_uplift >= 0.0
            or math.isfinite(drawdown)
            and drawdown >= MAX_CPCV_WORST_QUARTILE_DRAWDOWN_PCT
        )
        rows.extend(
            [
                gate_row("research_validation", "cpcv_distribution_available", True, source, "present", "CPCV_DISTRIBUTION_MISSING", candidate_scope=scope, model_name=model_name),
                gate_row("research_validation", "cpcv_path_median_uplift_positive", math.isfinite(median_uplift) and median_uplift > 0.0, median_uplift, ">0", "CPCV_MEDIAN_UPLIFT_LE_0", candidate_scope=scope, model_name=model_name),
                gate_row(
                    "research_validation",
                    "cpcv_worst_quartile_uplift_nonnegative_or_drawdown_ok",
                    worst_quartile_or_drawdown_pass,
                    f"q25={q25_uplift}; drawdown={drawdown}",
                    f"q25>=0 or drawdown>={MAX_CPCV_WORST_QUARTILE_DRAWDOWN_PCT}",
                    "CPCV_WORST_QUARTILE_UPLIFT_LT_0",
                    candidate_scope=scope,
                    model_name=model_name,
                ),
            ]
        )

    pbo_values: list[float] = []
    if not pbo_report.empty:
        for column in ["pbo_proxy", "pbo_cscv"]:
            if column in pbo_report.columns:
                pbo_values.extend(pd.to_numeric(pbo_report[column], errors="coerce").dropna().tolist())
    if not cscv_pbo_report.empty and "pbo_cscv" in cscv_pbo_report.columns:
        pbo_values.extend(pd.to_numeric(cscv_pbo_report["pbo_cscv"], errors="coerce").dropna().tolist())
    max_pbo = float(max(pbo_values)) if pbo_values else np.nan
    rows.append(
        gate_row(
            "research_validation",
            "pbo_below_0_50",
            math.isfinite(max_pbo) and max_pbo < MAX_PBO,
            max_pbo if math.isfinite(max_pbo) else "missing",
            f"<{MAX_PBO}",
            "PBO_GE_0_50",
        )
    )
    dsr_pass = bool(not dsr_report.empty and dsr_report.get("dsr_pass", pd.Series(dtype=object)).map(to_bool).any())
    rows.append(
        gate_row(
            "research_validation",
            "dsr_pass",
            dsr_pass,
            "any_strategy_pass" if dsr_pass else "missing_or_false",
            "True",
            "DSR_NOT_PASSED",
        )
    )
    if tsm_calibration is None or tsm_calibration.empty:
        rows.append(gate_row("research_validation", "tsm_like_effective_calibration_n", False, "missing", f">={MIN_TSM_LIKE_EFFECTIVE_CALIBRATION_N}", "TSM_LIKE_EFFECTIVE_N_LT_500"))
    else:
        selected_tsm = select_tsm_calibration_gate_row(tsm_calibration)
        row = selected_tsm.iloc[0]
        effective_n = np.nan
        for column in [
            "tsm_like_effective_calibration_n",
            "tsm_like_effective_train_validation_n",
            "effective_calibration_n",
            "effective_n",
            "holdout_event_count",
            "event_count",
        ]:
            if column in row.index:
                effective_n = row_float(row, column)
                if math.isfinite(effective_n):
                    break
        rows.append(
            gate_row(
                "research_validation",
                "tsm_like_effective_calibration_n",
                math.isfinite(effective_n) and effective_n >= MIN_TSM_LIKE_EFFECTIVE_CALIBRATION_N,
                effective_n if math.isfinite(effective_n) else "missing",
                f">={MIN_TSM_LIKE_EFFECTIVE_CALIBRATION_N}",
                "TSM_LIKE_EFFECTIVE_N_LT_500",
                model_name=str(row.get("model_name", "")),
                split=str(row.get("split", "")),
            )
        )
    return rows


def build_audit(
    comparison: pd.DataFrame,
    latest_prediction: dict[str, object],
    pooled_comparison: pd.DataFrame,
    pooled_latest: dict[str, object],
    tsm_calibration: pd.DataFrame,
    cpcv_strategy: pd.DataFrame | None = None,
    cpcv_model: pd.DataFrame | None = None,
    pbo_report: pd.DataFrame | None = None,
    cscv_pbo_report: pd.DataFrame | None = None,
    dsr_report: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    rows.extend(evaluate_local_models(comparison, latest_prediction))
    rows.extend(evaluate_pooled_models(pooled_comparison, pooled_latest, tsm_calibration))
    research_inputs_provided = any(x is not None for x in [cpcv_strategy, cpcv_model, pbo_report, cscv_pbo_report, dsr_report])
    if research_inputs_provided:
        rows.extend(
            evaluate_research_validation(
                cpcv_strategy if cpcv_strategy is not None else pd.DataFrame(),
                cpcv_model if cpcv_model is not None else pd.DataFrame(),
                pbo_report if pbo_report is not None else pd.DataFrame(),
                cscv_pbo_report if cscv_pbo_report is not None else pd.DataFrame(),
                dsr_report if dsr_report is not None else pd.DataFrame(),
                tsm_calibration,
            )
        )
    return pd.DataFrame(rows)


def unique_join(values: Iterable[object], limit: int = 8) -> str:
    clean = [str(v) for v in values if str(v) and str(v).lower() != "nan"]
    unique = sorted(set(clean))
    if len(unique) > limit:
        return "|".join(unique[:limit]) + f"|+{len(unique) - limit}"
    return "|".join(unique)


def build_root_causes(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    failed = audit[~audit["passed"].astype(bool)].copy()
    if failed.empty:
        return pd.DataFrame(
            [
                {
                    "root_cause": "PASS",
                    "priority": 0,
                    "failed_gate_count": 0,
                    "category": "pass",
                    "gate_groups": "",
                    "gates": "",
                    "candidate_scopes": "",
                    "horizons": "",
                    "models": "",
                    "splits": "",
                    "value_min": np.nan,
                    "value_max": np.nan,
                    "recommended_action": "No failed model gates.",
                }
            ]
        )
    exploded_rows: list[dict[str, object]] = []
    for _, row in failed.iterrows():
        raw_reasons = str(row.get("block_reason", "")).split("|")
        reasons = [reason[6:] if reason.startswith("MODEL:") else reason for reason in raw_reasons if reason and reason != "PASS"]
        if not reasons:
            reasons = ["UNKNOWN_FAILED_GATE"]
        for reason in reasons:
            out = row.to_dict()
            out["root_cause"] = reason
            exploded_rows.append(out)
    exploded = pd.DataFrame(exploded_rows)
    rows = []
    for reason, group in exploded.groupby("root_cause", dropna=False):
        priority, category, action = ROOT_CAUSE_GUIDANCE.get(
            str(reason),
            (3, "uncategorized_gate_failure", "Inspect the failed gate rows for this uncategorized reason."),
        )
        numeric_values = pd.to_numeric(group.get("value", pd.Series(dtype=float)), errors="coerce").dropna()
        rows.append(
            {
                "root_cause": reason,
                "priority": priority,
                "failed_gate_count": int(len(group)),
                "category": category,
                "gate_groups": unique_join(group.get("gate_group", pd.Series(dtype=object))),
                "gates": unique_join(group.get("gate", pd.Series(dtype=object))),
                "candidate_scopes": unique_join(group.get("candidate_scope", pd.Series(dtype=object))),
                "horizons": unique_join(group.get("horizon_days", pd.Series(dtype=object))),
                "models": unique_join(group.get("model_name", pd.Series(dtype=object))),
                "splits": unique_join(group.get("split", pd.Series(dtype=object))),
                "value_min": float(numeric_values.min()) if not numeric_values.empty else np.nan,
                "value_max": float(numeric_values.max()) if not numeric_values.empty else np.nan,
                "recommended_action": action,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["priority", "failed_gate_count", "root_cause"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def build_snapshot(audit: pd.DataFrame, latest_prediction: dict[str, object], pooled_latest: dict[str, object]) -> pd.DataFrame:
    def gate_pass(gate_group: str, gate: str, default: bool = False) -> bool:
        if audit.empty:
            return default
        rows = audit[audit["gate_group"].astype(str).eq(gate_group) & audit["gate"].astype(str).eq(gate)]
        if rows.empty:
            return default
        return bool(rows["passed"].astype(bool).all())

    if audit.empty:
        status = "FAIL"
        failed = pd.DataFrame()
        pooled_model_quality_pass = False
        pooled_latest_signal_pass = False
        pooled_support = False
        local_support = False
    else:
        failed = audit[~audit["passed"].astype(bool)]
        critical_failed = failed[failed["severity"].astype(str).eq("CRITICAL")]
        local_support = is_prediction_decision_support(latest_prediction.get("prediction_use_status", "UNKNOWN"))
        pooled_model_groups = audit[audit["gate_group"].astype(str).isin(["pooled_model", "pooled_tsm_calibration"])].copy()
        pooled_model_quality_pass = bool(not pooled_model_groups.empty and pooled_model_groups["passed"].astype(bool).all())
        research_validation_groups = audit[audit["gate_group"].astype(str).eq("research_validation")].copy()
        research_validation_pass = bool(research_validation_groups.empty or research_validation_groups["passed"].astype(bool).all())
        pooled_system_quality_pass = bool(pooled_model_quality_pass and research_validation_pass)
        pooled_latest_signal_pass = to_bool(pooled_latest.get("latest_signal_pass", False))
        pooled_support = to_bool(pooled_latest.get("decision_support_allowed", False))
        pooled_paper_support = to_bool(pooled_latest.get("paper_decision_support_allowed", False))
        if local_support or (pooled_support and pooled_system_quality_pass):
            status = "PASS"
        elif pooled_paper_support:
            status = "PAPER_DECISION_SUPPORT_ALLOWED"
        elif pooled_system_quality_pass and not pooled_latest_signal_pass:
            status = "MODEL_QUALITY_PASS_LATEST_BLOCKED"
        elif critical_failed.empty:
            status = "WARN"
        else:
            status = "BLOCKED"
    if audit.empty:
        research_validation_pass = False
        pooled_system_quality_pass = False
    model_quality_failures = failed[failed["gate_group"].astype(str).isin(["pooled_model", "pooled_tsm_calibration"])] if not failed.empty else pd.DataFrame()
    latest_failures = (
        failed[
            failed["gate_group"].astype(str).eq("pooled_latest")
            & ~failed["gate"].astype(str).eq("pooled_prediction_decision_support")
        ]
        if not failed.empty
        else pd.DataFrame()
    )
    pooled_model_quality_reasons = (
        unique_join(model_quality_failures["block_reason"]) if not model_quality_failures.empty else pooled_latest.get("model_quality_block_reasons", "PASS")
    )
    reported_latest_reasons = str(pooled_latest.get("latest_block_reasons", ""))
    pooled_latest_reasons = (
        reported_latest_reasons
        if reported_latest_reasons and reported_latest_reasons.lower() != "nan"
        else unique_join(latest_failures["block_reason"]) if not latest_failures.empty else "PASS"
    )
    local_failures = (
        failed[failed["gate_group"].astype(str).isin(["local_prediction", "local_latest"])]
        if not failed.empty
        else pd.DataFrame()
    )
    local_diagnostic_block_reasons = unique_join(local_failures["block_reason"]) if not local_failures.empty else "PASS"
    local_status = str(latest_prediction.get("prediction_use_status", "UNKNOWN"))
    local_tsm_diagnostic_only = bool((not local_support) and ("DISPLAY" in local_status or "DIAGNOSTIC" in local_status or "NO_MODEL" in local_status))
    pooled_threshold_stability_pass = gate_pass("pooled_model", "threshold_stability_pass", to_bool(pooled_latest.get("threshold_stability_pass", False)))
    pooled_economic_uplift_pass = gate_pass("pooled_model", "economic_uplift_pass", to_bool(pooled_latest.get("uplift_pass", False)))
    pooled_tsm_calibration_pass = bool(
        not audit.empty
        and not audit[audit["gate_group"].astype(str).eq("pooled_tsm_calibration")].empty
        and audit[audit["gate_group"].astype(str).eq("pooled_tsm_calibration")]["passed"].astype(bool).all()
    )
    research_validation_pass = bool(
        not audit.empty
        and (
            audit[audit["gate_group"].astype(str).eq("research_validation")].empty
            or audit[audit["gate_group"].astype(str).eq("research_validation")]["passed"].astype(bool).all()
        )
    )
    pooled_system_quality_pass = bool(pooled_model_quality_pass and research_validation_pass)
    pooled_threshold_failure_summary = pooled_latest.get("pooled_threshold_failure_summary") or pooled_latest.get("threshold_failure_summary")
    if not pooled_threshold_failure_summary:
        threshold_failures = model_quality_failures[model_quality_failures["gate"].astype(str).str.contains("threshold|min_selected_events_per_oof_fold", na=False)] if not model_quality_failures.empty else pd.DataFrame()
        pooled_threshold_failure_summary = unique_join(threshold_failures["block_reason"]) if not threshold_failures.empty else "PASS"
    pooled_uplift_failure_summary = pooled_latest.get("pooled_uplift_failure_summary") or pooled_latest.get("uplift_failure_summary")
    if not pooled_uplift_failure_summary:
        uplift_failures = model_quality_failures[model_quality_failures["gate"].astype(str).str.contains("uplift|selected_minus|positive_expectancy", na=False)] if not model_quality_failures.empty else pd.DataFrame()
        pooled_uplift_failure_summary = unique_join(uplift_failures["block_reason"]) if not uplift_failures.empty else "PASS"
    pooled_tsm_calibration_failure_summary = pooled_latest.get("pooled_tsm_calibration_failure_summary") or pooled_latest.get("tsm_calibration_failure_summary")
    if not pooled_tsm_calibration_failure_summary:
        tsm_failures = model_quality_failures[model_quality_failures["gate_group"].astype(str).eq("pooled_tsm_calibration")] if not model_quality_failures.empty else pd.DataFrame()
        pooled_tsm_calibration_failure_summary = unique_join(tsm_failures["block_reason"]) if not tsm_failures.empty else "PASS"
    next_required_evidence_action = pooled_latest.get("next_required_evidence_action", "")
    if not next_required_evidence_action:
        actions = []
        if not pooled_threshold_stability_pass:
            actions.append("improve_locked_risk_adjusted_rank_policy_or_expand_fold_trade_ready_events")
        if not pooled_economic_uplift_pass:
            actions.append("increase_oof_lower_bound_evidence_with_more_semiconductor_events")
        if not pooled_tsm_calibration_pass:
            actions.append("improve_tsm_route_calibration_or_expand_tsm_like_calibration_sample")
        next_required_evidence_action = "|".join(actions) if actions else "PASS"
    rows = [
        {"field": "model_gate_status", "value": status},
        {"field": "local_prediction_decision_support", "value": is_prediction_decision_support(latest_prediction.get("prediction_use_status", "UNKNOWN"))},
        {"field": "local_prediction_use_status", "value": latest_prediction.get("prediction_use_status", "UNKNOWN")},
        {"field": "local_tsm_diagnostic_only", "value": local_tsm_diagnostic_only},
        {"field": "local_diagnostic_block_reasons", "value": local_diagnostic_block_reasons},
        {"field": "pooled_model_quality_pass", "value": pooled_model_quality_pass},
        {"field": "research_validation_pass", "value": research_validation_pass},
        {"field": "pooled_system_quality_pass", "value": pooled_system_quality_pass},
        {"field": "pooled_threshold_stability_pass", "value": pooled_threshold_stability_pass},
        {"field": "pooled_economic_uplift_pass", "value": pooled_economic_uplift_pass},
        {"field": "pooled_tsm_calibration_pass", "value": pooled_tsm_calibration_pass},
        {"field": "pooled_latest_signal_pass", "value": pooled_latest_signal_pass},
        {"field": "pooled_prediction_decision_support", "value": bool(pooled_system_quality_pass and pooled_latest_signal_pass and pooled_support)},
        {"field": "paper_prediction_decision_support", "value": to_bool(pooled_latest.get("paper_decision_support_allowed", False))},
        {"field": "paper_gate_status", "value": pooled_latest.get("paper_gate_status", "UNKNOWN")},
        {"field": "paper_gate_block_reasons", "value": pooled_latest.get("paper_gate_block_reasons", "UNKNOWN")},
        {"field": "pooled_model_quality_block_reasons", "value": pooled_model_quality_reasons},
        {"field": "pooled_latest_block_reasons", "value": pooled_latest_reasons},
        {"field": "pooled_decision_block_reasons", "value": pooled_latest.get("decision_block_reasons", "UNKNOWN")},
        {"field": "pooled_threshold_failure_summary", "value": pooled_threshold_failure_summary},
        {"field": "pooled_uplift_failure_summary", "value": pooled_uplift_failure_summary},
        {"field": "pooled_tsm_calibration_failure_summary", "value": pooled_tsm_calibration_failure_summary},
        {"field": "next_required_evidence_action", "value": next_required_evidence_action},
        {"field": "failed_gate_count", "value": len(failed)},
        {"field": "critical_failed_gate_count", "value": int((failed["severity"].astype(str).eq("CRITICAL")).sum()) if not failed.empty else 0},
        {"field": "failed_gate_groups", "value": "|".join(sorted(set(failed["gate_group"].astype(str)))) if not failed.empty else "PASS"},
        {"field": "generated_at_utc", "value": now_utc_iso()},
    ]
    return pd.DataFrame(rows)


def write_report(outdir: Path, audit: pd.DataFrame, snapshot: pd.DataFrame, root_causes: pd.DataFrame) -> None:
    snap = dict(zip(snapshot["field"], snapshot["value"]))
    failed = audit[~audit["passed"].astype(bool)] if not audit.empty else pd.DataFrame()
    lines = [
        "# TSMC Model Gate Audit Report",
        "",
        f"- Model gate status: {snap.get('model_gate_status', 'NA')}",
        f"- Local prediction decision support: {snap.get('local_prediction_decision_support', 'NA')}",
        f"- Local prediction use status: {snap.get('local_prediction_use_status', 'NA')}",
        f"- Pooled model quality pass: {snap.get('pooled_model_quality_pass', 'NA')}",
        f"- Research validation pass: {snap.get('research_validation_pass', 'NA')}",
        f"- Pooled system quality pass: {snap.get('pooled_system_quality_pass', 'NA')}",
        f"- Pooled latest signal pass: {snap.get('pooled_latest_signal_pass', 'NA')}",
        f"- Pooled prediction decision support: {snap.get('pooled_prediction_decision_support', 'NA')}",
        f"- Paper prediction decision support: {snap.get('paper_prediction_decision_support', 'NA')}",
        f"- Paper gate status: {snap.get('paper_gate_status', 'NA')}",
        f"- Failed gate count: {snap.get('failed_gate_count', 'NA')}",
        f"- Critical failed gate count: {snap.get('critical_failed_gate_count', 'NA')}",
        "",
        "## Failed Gates",
        "",
        "| Group | Scope | Horizon | Model | Split | Gate | Value | Threshold | Block Reason |",
        "|---|---|---:|---|---|---|---:|---:|---|",
    ]
    if failed.empty:
        lines.append("| PASS | NA | NA | NA | NA | NA | NA | NA | PASS |")
    else:
        for _, row in failed.iterrows():
            lines.append(
                f"| {row['gate_group']} | {row['candidate_scope']} | {row['horizon_days']} | {row['model_name']} | {row['split']} | "
                f"{row['gate']} | {row['value']} | {row['threshold']} | {row['block_reason']} |"
            )
    lines.extend(["", "## Root Cause Summary", ""])
    if root_causes.empty:
        lines.append("- No root-cause summary available.")
    else:
        lines.extend(
            [
                "| Priority | Root Cause | Failed Gates | Category | Recommended Action |",
                "|---:|---|---:|---|---|",
            ]
        )
        for _, row in root_causes.iterrows():
            lines.append(
                f"| {row['priority']} | {row['root_cause']} | {row['failed_gate_count']} | "
                f"{row['category']} | {row['recommended_action']} |"
            )
    lines.extend(["", "This report is research tooling, not investment advice."])
    (outdir / "tsm_model_gate_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model gate audit for TSM prediction outputs.")
    parser.add_argument("--comparison", default="tsm_price_rule_output/tsm_prediction_model_comparison.csv")
    parser.add_argument("--latest-prediction", default="tsm_price_rule_output/tsm_latest_prediction_snapshot.csv")
    parser.add_argument("--pooled-comparison", default="tsm_price_rule_output/tsm_pooled_model_comparison.csv")
    parser.add_argument("--pooled-latest", default="tsm_price_rule_output/tsm_pooled_latest_prediction_overlay.csv")
    parser.add_argument("--tsm-calibration", default="tsm_price_rule_output/tsm_pooled_tsm_calibration_metrics.csv")
    parser.add_argument("--cpcv-strategy-distribution", default="tsm_price_rule_output/tsm_cpcv_strategy_distribution.csv")
    parser.add_argument("--cpcv-model-distribution", default="tsm_price_rule_output/tsm_cpcv_model_distribution.csv")
    parser.add_argument("--pbo-report", default="tsm_price_rule_output/tsm_pbo_report.csv")
    parser.add_argument("--cscv-pbo-report", default="tsm_price_rule_output/tsm_cscv_pbo_report.csv")
    parser.add_argument("--dsr-report", default="tsm_price_rule_output/tsm_deflated_sharpe_report.csv")
    parser.add_argument("--paper-gate", default="tsm_price_rule_output/tsm_paper_gate_snapshot.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    comparison = read_csv(Path(args.comparison))
    latest_prediction = read_snapshot(Path(args.latest_prediction))
    pooled_comparison = read_csv(Path(args.pooled_comparison))
    pooled_latest = read_snapshot(Path(args.pooled_latest))
    tsm_calibration = read_csv(Path(args.tsm_calibration))
    cpcv_strategy = read_csv(Path(args.cpcv_strategy_distribution))
    cpcv_model = read_csv(Path(args.cpcv_model_distribution))
    pbo_report = read_csv(Path(args.pbo_report))
    cscv_pbo_report = read_csv(Path(args.cscv_pbo_report))
    dsr_report = read_csv(Path(args.dsr_report))
    paper_gate = read_snapshot(Path(args.paper_gate))
    for key, value in paper_gate.items():
        pooled_latest.setdefault(key, value)

    audit = build_audit(
        comparison,
        latest_prediction,
        pooled_comparison,
        pooled_latest,
        tsm_calibration,
        cpcv_strategy,
        cpcv_model,
        pbo_report,
        cscv_pbo_report,
        dsr_report,
    )
    snapshot = build_snapshot(audit, latest_prediction, pooled_latest)
    root_causes = build_root_causes(audit)
    audit.to_csv(outdir / "tsm_model_gate_audit.csv", index=False)
    snapshot.to_csv(outdir / "tsm_model_gate_snapshot.csv", index=False)
    root_causes.to_csv(outdir / "tsm_model_gate_root_causes.csv", index=False)
    write_report(outdir, audit, snapshot, root_causes)
    print("completed: model gate outputs =", outdir.resolve())
    print(snapshot.to_string(index=False))


if __name__ == "__main__":
    main()

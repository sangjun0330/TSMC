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
LOCAL_DECISION_CANDIDATE_SCOPE = "trade_ready_entry"
LOCAL_DECISION_HORIZON_DAYS = 20
RANK_UPLIFT_DIAGNOSTIC_GATE = "rank_top_quintile_uplift_diagnostic"
RANK_POLICY_THRESHOLD_SELECTED_GATES = {
    "selected_oos_event_count",
    "min_selected_events_per_fold",
    "selected_minus_rule_all_positive",
    "selected_expectancy_ci_lower_positive",
    "positive_expectancy_folds",
    "threshold_iqr",
}
SELECTED_CONTRAST_DEPENDENT_GATES = {
    "selected_minus_rule_all_positive",
    "selected_expectancy_ci_lower_positive",
    "positive_expectancy_folds",
}
MODEL_REJECTION_SKILL_GATES = {"brier_improvement_positive", "pr_auc_above_base"}

MIN_POOLED_EVAL_EVENTS = 150
MIN_POOLED_SELECTED_EVENTS = 50
MAX_POOLED_ECE = 0.10
MAX_TSM_ECE = 0.15
MAX_STOP_HIT_FOR_LATEST = 0.35
PAPER_MAX_STOP_HIT_FOR_LATEST = 0.40
MIN_EXPECTED_R_FOR_LATEST = 0.35
MIN_TSM_LIKE_EFFECTIVE_CALIBRATION_N = 500
MAX_PBO = 0.50
MAX_CPCV_WORST_QUARTILE_DRAWDOWN_PCT = -35.0
LATEST_GATE_GROUPS = {"pooled_latest", "local_latest"}
NON_PERFORMANCE_BLOCK_TOKENS = (
    "LATEST_NOT_TRADE_READY",
    "POOLED_DECISION_SCORE_BELOW_THRESHOLD",
    "POOLED_STOP_RISK",
    "POOLED_EXPECTED_R",
    "NOT_20D_TRADE_READY_DECISION_SCOPE",
    "FORECAST_DIAGNOSTIC",
)
ACTIVE_PERFORMANCE_GATE_GROUPS = {"pooled_model", "pooled_tsm_calibration", "research_validation"}
PERFORMANCE_EVIDENCE_FAMILIES = {"sample_evidence", "calibration_sample_warning"}
PERFORMANCE_AGGREGATE_FAMILIES = {"aggregate_quality_flag"}
PERFORMANCE_NON_METRIC_FAMILIES = PERFORMANCE_EVIDENCE_FAMILIES | PERFORMANCE_AGGREGATE_FAMILIES
UNRESOLVED_PERFORMANCE_PRIORITY = {
    "probability_calibration": 1,
    "probabilistic_skill": 1,
    "economic_uplift": 2,
    "discrimination": 3,
    "threshold_stability": 3,
    "backtest_overfit_control": 4,
    "other_performance": 5,
}
UNRESOLVED_PERFORMANCE_ACTIONS = {
    "probability_calibration": "improve_probability_calibration_before_decision_use",
    "probabilistic_skill": "improve_brier_skill_against_base_rate",
    "economic_uplift": "improve_fold_stable_selected_expectancy_or_keep_rank_policy_diagnostic_only",
    "discrimination": "improve_rank_discrimination_against_base_rate",
    "threshold_stability": "stabilize_threshold_policy_across_walk_forward_folds",
    "backtest_overfit_control": "improve_overfit_controls_before_strategy_promotion",
    "other_performance": "inspect_uncategorized_performance_warning",
}

ROOT_CAUSE_GUIDANCE = {
    "NOT_20D_TRADE_READY_DECISION_SCOPE": (
        1,
        "system_sample_evidence",
        "20일 매수 판단 후보 진단 행을 먼저 확인하고, 결정 사용은 충분한 표본 확보 뒤에만 허용합니다.",
    ),
    "FORECAST_DIAGNOSTIC_DISPLAY_ONLY_NOT_PROMOTED": (
        4,
        "forecast_diagnostic",
        "종가 예측 모델은 모든 horizon 품질 기준을 통과하기 전까지 표시용 진단 모델로만 유지합니다.",
    ),
    "FORECAST_DIAGNOSTIC_COMPARISON_MISSING": (
        4,
        "forecast_diagnostic",
        "종가 예측 comparison 산출물을 먼저 생성한 뒤 horizon별 승격 기준을 확인합니다.",
    ),
    "FORECAST_DIAGNOSTIC_QUALITY_CHECK_FAILED": (
        4,
        "forecast_diagnostic",
        "종가 예측 quality check 실패 항목을 확인하고 누수, 표본 수, 구간 coverage를 먼저 보강합니다.",
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
    "OOS_FOLD_COUNT_LT_MIN": (
        1,
        "system_sample_evidence",
        "시간순 OOS fold가 최소 기준을 채우기 전까지 해당 후보는 진단용으로 유지합니다.",
    ),
    "SELECTION_CONTRAST_MISSING": (
        1,
        "system_sample_evidence",
        "선택 집합이 전체 OOS와 같거나 비어 있으면 필터 효과를 검증할 수 없으므로 decision gate에서 제외합니다.",
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
    "LOCAL_20D_TRADE_READY_MODEL_MISSING": (
        1,
        "system_sample_evidence",
        "20일 trade-ready 로컬 판단 행이 없으면 단독 종목 모델은 판단 지원으로 승격하지 않습니다.",
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
        "Top10 직접 확률 조정을 믿기 전에 Top10 보류 검증 사례를 더 확보합니다.",
    ),
    "TSM_CALIBRATION_ECE_GT_0_15": (
        1,
        "system_calibration",
        "Top10 직접 calibration ECE가 0.15를 넘으면 pooled overlay를 판단 지원으로 승격하지 않습니다.",
    ),
    "TSM_CALIBRATION_ROUTE_NOT_PASSED": (
        1,
        "system_calibration",
        "POOLED_ONLY, group/logit/Platt/blend route 중 pre-test에서 선택된 route가 ECE/Brier/holdout 조건을 통과할 때만 gate primary로 사용합니다.",
    ),
    "NO_MODEL": (
        1,
        "system_sample_evidence",
        "신뢰 가능한 20일 매수 판단 모델이 생길 때까지 최신 종목별 예측은 표시용으로 둡니다.",
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
        "Top10 직접 calibration metrics 산출물이 없으면 pooled overlay 신뢰도를 판단할 수 없습니다.",
    ),
    "TSM_LIKE_EFFECTIVE_N_LT_500": (
        1,
        "system_calibration",
        "Similarity calibration 유효 표본이 500건 이상 쌓이기 전까지 pooled overlay를 판단 지원으로 승격하지 않습니다.",
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


def criteria_row(
    model: str,
    metric: str,
    passed: bool,
    value: object,
    threshold: object,
    *,
    horizon_days: object = "",
    model_name: str = "",
    candidate_scope: str = "",
    split: str = "",
    severity: str = "CRITICAL",
    gap: object = "",
    details: str = "",
) -> dict[str, object]:
    return {
        "criteria_group": model,
        "horizon_days": horizon_days,
        "candidate_scope": candidate_scope,
        "model_name": model_name,
        "split": split,
        "metric": metric,
        "passed": bool(passed),
        "severity": severity,
        "value": value,
        "threshold": threshold,
        "gap": gap,
        "details": "PASS" if passed and not details else details,
    }


def numeric_gap(value: object, threshold: float, direction: str) -> float:
    parsed = as_float(value)
    if pd.isna(parsed):
        return np.nan
    if direction == ">=":
        return float(parsed - threshold)
    if direction == "<=":
        return float(threshold - parsed)
    raise ValueError(f"unknown gap direction: {direction}")


def add_min_row(
    rows: list[dict[str, object]],
    model: str,
    metric: str,
    value: object,
    threshold: float,
    *,
    horizon_days: object = "",
    model_name: str = "",
    candidate_scope: str = "",
    split: str = "",
    severity: str = "CRITICAL",
    details: str = "",
) -> None:
    parsed = as_float(value)
    passed = pd.notna(parsed) and parsed + 1e-12 >= threshold
    rows.append(
        criteria_row(
            model,
            metric,
            passed,
            value,
            f">={threshold:g}",
            horizon_days=horizon_days,
            model_name=model_name,
            candidate_scope=candidate_scope,
            split=split,
            severity=severity,
            gap=numeric_gap(value, threshold, ">="),
            details=details,
        )
    )


def add_max_row(
    rows: list[dict[str, object]],
    model: str,
    metric: str,
    value: object,
    threshold: float,
    *,
    horizon_days: object = "",
    model_name: str = "",
    candidate_scope: str = "",
    split: str = "",
    severity: str = "CRITICAL",
    details: str = "",
) -> None:
    parsed = as_float(value)
    passed = pd.notna(parsed) and parsed <= threshold + 1e-12
    rows.append(
        criteria_row(
            model,
            metric,
            passed,
            value,
            f"<={threshold:g}",
            horizon_days=horizon_days,
            model_name=model_name,
            candidate_scope=candidate_scope,
            split=split,
            severity=severity,
            gap=numeric_gap(value, threshold, "<="),
            details=details,
        )
    )


def best_row(frame: pd.DataFrame, filters: dict[str, object], sort_cols: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    data = frame.copy()
    for col, expected in filters.items():
        if col not in data.columns:
            return pd.Series(dtype=object)
        if isinstance(expected, (set, list, tuple)):
            data = data[data[col].isin(expected)]
        else:
            data = data[data[col].eq(expected)]
    if data.empty:
        return pd.Series(dtype=object)
    active_sort_cols = [col for col in sort_cols if col in data.columns]
    if not active_sort_cols:
        return data.iloc[0]
    ascending = [False] * len(active_sort_cols)
    return data.sort_values(active_sort_cols, ascending=ascending).iloc[0]


def next_day_direction_accuracy(oos_predictions: pd.DataFrame, model_name: str) -> float:
    if oos_predictions.empty or not model_name:
        return np.nan
    required = {"model_name", "p_success", "label_success"}
    if not required.issubset(oos_predictions.columns):
        return np.nan
    data = oos_predictions[oos_predictions["model_name"].astype(str).eq(str(model_name))].copy()
    p = pd.to_numeric(data["p_success"], errors="coerce")
    y = pd.to_numeric(data["label_success"], errors="coerce")
    mask = p.notna() & y.notna()
    if not mask.any():
        return np.nan
    return float(((p.loc[mask] >= 0.5).astype(int) == y.loc[mask].astype(int)).mean())


def local_20d_benchmark_candidate(comparison: pd.DataFrame) -> pd.Series:
    if comparison.empty:
        return pd.Series(dtype=object)
    required = {"candidate_scope", "horizon_days"}
    if not required.issubset(comparison.columns):
        return pd.Series(dtype=object)
    candidates = comparison[comparison["horizon_days"].eq(20)].copy()
    if candidates.empty:
        return pd.Series(dtype=object)

    def fail_count(row: pd.Series) -> int:
        checks = [
            as_float(row.get("brier_improvement_pct")) >= 2.0,
            as_float(row.get("decision_ece", row.get("ece"))) <= 0.06,
            as_float(row.get("pr_auc")) >= 0.56,
            as_float(row.get("selected_minus_rule_all_pct")) >= 1.0,
            as_float(row.get("selected_oos_event_count")) >= 150.0,
            as_float(row.get("min_selected_events_per_fold")) >= 25.0,
            as_float(row.get("threshold_iqr")) <= 0.05,
            as_float(row.get("selected_signal_expectancy_ci_lower_pct")) >= 0.5,
            as_float(row.get("positive_expectancy_folds")) >= 4.0,
        ]
        return int(sum(not bool(check) for check in checks))

    candidates["_benchmark_fail_count"] = candidates.apply(fail_count, axis=1)
    candidates["_benchmark_sample_pass"] = (
        pd.to_numeric(candidates.get("selected_oos_event_count", pd.Series(np.nan, index=candidates.index)), errors="coerce").ge(150.0)
        & pd.to_numeric(candidates.get("min_selected_events_per_fold", pd.Series(np.nan, index=candidates.index)), errors="coerce").ge(25.0)
    )
    candidates["_scope_priority"] = candidates["candidate_scope"].astype(str).map(
        {
            LOCAL_DECISION_CANDIDATE_SCOPE: 3,
            "entry_research": 2,
            "trigger_all": 1,
            "context_all": 0,
        }
    ).fillna(0)
    sort_cols = [
        "_benchmark_sample_pass",
        "_benchmark_fail_count",
        "brier_improvement_pct",
        "pr_auc",
        "selected_minus_rule_all_pct",
        "selected_signal_expectancy_ci_lower_pct",
        "_scope_priority",
        "rank_score",
    ]
    ascending = [False, True, False, False, False, False, False, False]
    active = [col for col in sort_cols if col in candidates.columns]
    active_ascending = [ascending[sort_cols.index(col)] for col in active]
    return candidates.sort_values(active, ascending=active_ascending).iloc[0]


# --- Semiconductor single-sector benchmark profile --------------------------
# The decision/research universe is 12 semiconductor names (high realized
# volatility, high cross-sectional correlation). Per operator direction
# (2026-06), the criteria that are directly driven by sector noise — daily
# directional accuracy, close-forecast MAE improvement, and stop-hit ceilings —
# are MODERATELY relaxed to reflect that regime. Everything tied to model
# trustworthiness rather than sector difficulty is kept at the original strict
# level: probability calibration stays inside the table's own stated 0.06~0.07
# ECE band, and discrimination (PR-AUC), sample sizes, threshold stability
# (IQR), expectancy-fold robustness, and overfit controls (PBO/DSR/CPCV) are
# unchanged.
SECTOR_ECE_MAX = 0.07  # was 0.06; within the common table's stated 0.06~0.07 band
# Next-day directional classifier
SECTOR_NEXT_DAY_DIRECTION_MIN = 0.55  # was 0.57 (-2pp for sector noise)
SECTOR_NEXT_DAY_PR_AUC_MIN = 0.56  # unchanged — discrimination kept strict
SECTOR_NEXT_DAY_BRIER_MIN = 1.0  # unchanged
# Next-close forecast: (group_name, direction_min, mae_improvement_min) per horizon
SECTOR_NEXT_CLOSE_THRESHOLDS = {
    1: ("next_close_1d", 0.56, 1.5),   # was 0.58, 2.0
    5: ("next_close_5d", 0.58, 2.5),   # was 0.60, 3.0
    20: ("next_close_20d", 0.62, 4.0),  # was 0.65, 5.0
}
# Pooled overlays: stop-hit ceilings relaxed +5pp for sector volatility; the
# pooled ECE ceiling is unified to the 0.07 band. All sample/uplift/expectancy/
# threshold-stability gates stay at their original strict values.
SECTOR_POOLED_STOP_HIT_MAX = {20: 40.0, 60: 45.0}  # was 35.0 / 40.0
SECTOR_POOLED_ECE_MAX = {5: 0.07, 20: 0.07, 60: 0.07}  # pooled_20d was 0.06


def build_benchmark_criteria_audit(
    comparison: pd.DataFrame,
    pooled_comparison: pd.DataFrame,
    next_day_comparison: pd.DataFrame,
    next_day_oos_predictions: pd.DataFrame,
    next_close_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate the user-provided practical-performance criteria table.

    Thresholds reflect the semiconductor single-sector profile (see the
    SECTOR_* constants above): directional / MAE / stop-hit targets are
    moderately relaxed for sector volatility, while calibration, sample size,
    threshold stability, and overfit controls remain at strict levels.
    """
    rows: list[dict[str, object]] = []

    next_day = best_row(
        next_day_comparison,
        {"candidate_scope": "next_day_up_all", "horizon_days": 1},
        ["rank_score", "brier_improvement_pct", "pr_auc"],
    )
    if next_day.empty:
        rows.append(criteria_row("next_day_up_1d", "candidate_available", False, "missing", "available", details="missing next-day comparison"))
    else:
        model_name = str(next_day.get("model_name", ""))
        direction_acc = next_day_direction_accuracy(next_day_oos_predictions, model_name)
        add_min_row(rows, "next_day_up_1d", "direction_accuracy", direction_acc, SECTOR_NEXT_DAY_DIRECTION_MIN, horizon_days=1, model_name=model_name, candidate_scope=next_day.get("candidate_scope", ""))
        add_min_row(rows, "next_day_up_1d", "pr_auc", next_day.get("pr_auc"), SECTOR_NEXT_DAY_PR_AUC_MIN, horizon_days=1, model_name=model_name, candidate_scope=next_day.get("candidate_scope", ""))
        add_max_row(rows, "next_day_up_1d", "ece", next_day.get("decision_ece", next_day.get("ece")), SECTOR_ECE_MAX, horizon_days=1, model_name=model_name, candidate_scope=next_day.get("candidate_scope", ""))
        add_min_row(rows, "next_day_up_1d", "brier_improvement_pct", next_day.get("brier_improvement_pct"), SECTOR_NEXT_DAY_BRIER_MIN, horizon_days=1, model_name=model_name, candidate_scope=next_day.get("candidate_scope", ""))
        add_min_row(rows, "next_day_up_1d", "oos_fold_count", next_day.get("fold_count"), 5, horizon_days=1, model_name=model_name, candidate_scope=next_day.get("candidate_scope", ""))
        add_min_row(rows, "next_day_up_1d", "oos_event_count", next_day.get("oos_event_count"), 1000, horizon_days=1, model_name=model_name, candidate_scope=next_day.get("candidate_scope", ""))

    next_close_thresholds = SECTOR_NEXT_CLOSE_THRESHOLDS
    for horizon, (group_name, direction_min, mae_min) in next_close_thresholds.items():
        row = best_row(next_close_comparison, {"horizon_days": horizon}, ["rank_score", "mae_improvement_pct"])
        if row.empty:
            rows.append(criteria_row(group_name, "candidate_available", False, "missing", "available", horizon_days=horizon, details="missing next-close comparison"))
            continue
        model_name = str(row.get("model_name", ""))
        add_min_row(rows, group_name, "direction_accuracy", row.get("direction_accuracy"), direction_min, horizon_days=horizon, model_name=model_name, candidate_scope=row.get("candidate_scope", ""))
        add_min_row(rows, group_name, "mae_improvement_pct", row.get("mae_improvement_pct"), mae_min, horizon_days=horizon, model_name=model_name, candidate_scope=row.get("candidate_scope", ""))
        add_max_row(rows, group_name, "rmse_delta_vs_naive", row.get("rmse_delta_vs_naive"), 0.0, horizon_days=horizon, model_name=model_name, candidate_scope=row.get("candidate_scope", ""))
        add_min_row(rows, group_name, "oos_fold_count", row.get("fold_count"), 5, horizon_days=horizon, model_name=model_name, candidate_scope=row.get("candidate_scope", ""))
        add_min_row(rows, group_name, "oos_event_count", row.get("oos_event_count"), 1000, horizon_days=horizon, model_name=model_name, candidate_scope=row.get("candidate_scope", ""))

    local = local_20d_benchmark_candidate(comparison)
    if local.empty:
        rows.append(criteria_row("local_20d_success_probability", "candidate_available", False, "missing", "available", horizon_days=20, details="missing local 20D comparison"))
    else:
        model_name = str(local.get("model_name", ""))
        add_min_row(rows, "local_20d_success_probability", "brier_improvement_pct", local.get("brier_improvement_pct"), 2.0, horizon_days=20, model_name=model_name, candidate_scope=local.get("candidate_scope", ""))
        add_max_row(rows, "local_20d_success_probability", "ece", local.get("decision_ece", local.get("ece")), SECTOR_ECE_MAX, horizon_days=20, model_name=model_name, candidate_scope=local.get("candidate_scope", ""))
        add_min_row(rows, "local_20d_success_probability", "pr_auc", local.get("pr_auc"), 0.56, horizon_days=20, model_name=model_name, candidate_scope=local.get("candidate_scope", ""))
        add_min_row(rows, "local_20d_success_probability", "selected_return_uplift_pct_point", local.get("selected_minus_rule_all_pct"), 1.0, horizon_days=20, model_name=model_name, candidate_scope=local.get("candidate_scope", ""))
        add_min_row(rows, "local_20d_success_probability", "selected_oos_event_count", local.get("selected_oos_event_count"), 150, horizon_days=20, model_name=model_name, candidate_scope=local.get("candidate_scope", ""))
        add_min_row(rows, "local_20d_success_probability", "fold_selected_event_min", local.get("min_selected_events_per_fold"), 25, horizon_days=20, model_name=model_name, candidate_scope=local.get("candidate_scope", ""))
        add_max_row(rows, "local_20d_success_probability", "threshold_iqr", local.get("threshold_iqr"), 0.05, horizon_days=20, model_name=model_name, candidate_scope=local.get("candidate_scope", ""))
        add_min_row(rows, "local_20d_success_probability", "selected_ci_lower_pct", local.get("selected_signal_expectancy_ci_lower_pct"), 0.5, horizon_days=20, model_name=model_name, candidate_scope=local.get("candidate_scope", ""))
        add_min_row(rows, "local_20d_success_probability", "positive_expectancy_folds", local.get("positive_expectancy_folds"), 4, horizon_days=20, model_name=model_name, candidate_scope=local.get("candidate_scope", ""))

    pooled_specs = {
        5: ("pooled_5d", 1.5, SECTOR_POOLED_ECE_MAX[5], 0.5, 5.0, np.nan),
        20: ("pooled_20d", 2.0, SECTOR_POOLED_ECE_MAX[20], 1.0, SECTOR_POOLED_STOP_HIT_MAX[20], 0.35),
        60: ("pooled_60d", 2.0, SECTOR_POOLED_ECE_MAX[60], 2.0, SECTOR_POOLED_STOP_HIT_MAX[60], 0.50),
    }
    for horizon, (group_name, brier_min, ece_max, uplift_min, stop_hit_max_pct, expected_r_min) in pooled_specs.items():
        row = best_row(
            pooled_comparison,
            {"horizon_days": horizon, "split": "combined_test_holdout"},
            ["is_champion", "brier_improvement_pct", "selected_minus_all_pct"],
        )
        if row.empty:
            rows.append(criteria_row(group_name, "candidate_available", False, "missing", "available", horizon_days=horizon, details="missing pooled comparison"))
            continue
        model_name = str(row.get("model_name", ""))
        scope = str(row.get("evaluation_scope", ""))
        add_min_row(rows, group_name, "brier_improvement_pct", row.get("brier_improvement_pct"), brier_min, horizon_days=horizon, model_name=model_name, candidate_scope=scope, split=row.get("split", ""))
        add_max_row(rows, group_name, "ece", row.get("decision_ece", row.get("ece")), ece_max, horizon_days=horizon, model_name=model_name, candidate_scope=scope, split=row.get("split", ""))
        add_min_row(rows, group_name, "selected_return_uplift_pct_point", row.get("selected_minus_all_pct"), uplift_min, horizon_days=horizon, model_name=model_name, candidate_scope=scope, split=row.get("split", ""))
        if horizon == 5:
            stop_improvement_pct = as_float(row.get("stop_rate_improvement")) * 100.0 if pd.notna(as_float(row.get("stop_rate_improvement"))) else np.nan
            add_min_row(rows, group_name, "stop_improvement_pct_point", stop_improvement_pct, 5.0, horizon_days=horizon, model_name=model_name, candidate_scope=scope, split=row.get("split", ""))
        else:
            stop_hit_pct = as_float(row.get("selected_stop_rate")) * 100.0 if pd.notna(as_float(row.get("selected_stop_rate"))) else np.nan
            add_max_row(rows, group_name, "selected_stop_hit_pct", stop_hit_pct, stop_hit_max_pct, horizon_days=horizon, model_name=model_name, candidate_scope=scope, split=row.get("split", ""))
            add_min_row(rows, group_name, "selected_expected_r", row.get("selected_expected_r", row.get("selected_mean_expected_r", np.nan)), expected_r_min, horizon_days=horizon, model_name=model_name, candidate_scope=scope, split=row.get("split", ""))
        add_min_row(rows, group_name, "oos_fold_count", row.get("fold_count"), 5, horizon_days=horizon, model_name=model_name, candidate_scope=scope, split=row.get("split", ""))
        add_min_row(rows, group_name, "pooled_oos_event_count", row.get("event_count"), 2000, horizon_days=horizon, model_name=model_name, candidate_scope=scope, split=row.get("split", ""))
        add_min_row(rows, group_name, "selected_oos_event_count", row.get("selected_event_count"), 150, horizon_days=horizon, model_name=model_name, candidate_scope=scope, split=row.get("split", ""))
        add_min_row(rows, group_name, "fold_selected_event_min", row.get("min_selected_events_per_fold"), 25, horizon_days=horizon, model_name=model_name, candidate_scope=scope, split=row.get("split", ""))
        add_max_row(rows, group_name, "threshold_iqr", row.get("threshold_iqr"), 0.05, horizon_days=horizon, model_name=model_name, candidate_scope=scope, split=row.get("split", ""))
        add_min_row(rows, group_name, "selected_ci_lower_pct", row.get("selected_expectancy_ci_lower_pct"), 0.5, horizon_days=horizon, model_name=model_name, candidate_scope=scope, split=row.get("split", ""))
        add_min_row(rows, group_name, "positive_expectancy_folds", row.get("positive_expectancy_fold_count"), 4, horizon_days=horizon, model_name=model_name, candidate_scope=scope, split=row.get("split", ""))

    audit = pd.DataFrame(rows)
    if audit.empty:
        return audit
    return audit.sort_values(["passed", "criteria_group", "metric"], ascending=[True, True, True]).reset_index(drop=True)


def has_rank_uplift_diagnostic(row: pd.Series) -> bool:
    return "rank_top_quintile_minus_all_pct" in row.index or "rank_top_quintile_count" in row.index


def rank_uplift_status(rank_top_count: float, rank_uplift_pct: float, rank_policy_pass: bool = False) -> str:
    if rank_policy_pass:
        return "RANK_TOP_QUINTILE_POLICY_PASS"
    if not math.isfinite(rank_top_count) or rank_top_count <= 0 or not math.isfinite(rank_uplift_pct):
        return "RANK_TOP_QUINTILE_UPLIFT_UNAVAILABLE"
    if rank_uplift_pct > 0:
        return "RANK_TOP_QUINTILE_UPLIFT_POSITIVE"
    if rank_uplift_pct == 0:
        return "RANK_TOP_QUINTILE_UPLIFT_FLAT"
    return "RANK_TOP_QUINTILE_UPLIFT_NEGATIVE"


def rank_uplift_diagnostic_row(row: pd.Series) -> dict[str, object] | None:
    if not has_rank_uplift_diagnostic(row):
        return None
    rank_top_count = row_float(row, "rank_top_quintile_count", np.nan)
    rank_uplift_pct = row_float(row, "rank_top_quintile_minus_all_pct", np.nan)
    rank_policy_pass = to_bool(row.get("rank_policy_diagnostic_pass", False))
    status = rank_uplift_status(rank_top_count, rank_uplift_pct, rank_policy_pass)
    out = gate_row(
        "local_prediction_rank_diagnostic",
        RANK_UPLIFT_DIAGNOSTIC_GATE,
        True,
        rank_uplift_pct,
        "diagnostic_only_top_20pct_minus_all_pct",
        status,
        severity="INFO",
        candidate_scope=str(row.get("candidate_scope", "")),
        horizon_days=row.get("horizon_days", ""),
        model_name=str(row.get("model_name", "")),
    )
    out["block_reason"] = status
    out["rank_top_quintile_count"] = rank_top_count
    out["rank_top_quintile_return_pct"] = row_float(row, "rank_top_quintile_return_pct", np.nan)
    out["rank_top_quintile_success_rate"] = row_float(row, "rank_top_quintile_success_rate", np.nan)
    out["rank_top_quintile_fold_count"] = row_float(row, "rank_top_quintile_fold_count", np.nan)
    out["rank_top_quintile_positive_folds"] = row_float(row, "rank_top_quintile_positive_folds", np.nan)
    out["rank_top_quintile_min_fold_uplift_pct"] = row_float(row, "rank_top_quintile_min_fold_uplift_pct", np.nan)
    out["rank_top_quintile_se_lower_pct"] = row_float(row, "rank_top_quintile_se_lower_pct", np.nan)
    out["rank_policy_diagnostic_pass"] = rank_policy_pass
    return out


def row_float(row: pd.Series, column: str, default: float = np.nan) -> float:
    return as_float(row.get(column, default), default)


def row_bool(row: pd.Series, column: str) -> bool:
    return to_bool(row.get(column, False))


def dsr_gate_value(dsr_report: pd.DataFrame) -> str:
    if dsr_report.empty:
        return "missing"
    pass_series = dsr_report.get("dsr_pass", pd.Series(False, index=dsr_report.index)).map(to_bool)
    passing_count = int(pass_series.sum())
    total_count = int(len(dsr_report))
    best_dsr = np.nan
    best_strategy = ""
    if "deflated_sharpe_ratio" in dsr_report.columns:
        dsr_values = pd.to_numeric(dsr_report["deflated_sharpe_ratio"], errors="coerce")
        if dsr_values.notna().any():
            best_idx = dsr_values.idxmax()
            best_dsr = float(dsr_values.loc[best_idx])
            strategy_col = "strategy_name" if "strategy_name" in dsr_report.columns else "strategy_id" if "strategy_id" in dsr_report.columns else ""
            best_strategy = str(dsr_report.loc[best_idx, strategy_col]) if strategy_col else ""
    best_observed_sharpe = np.nan
    if "observed_sharpe" in dsr_report.columns:
        sharpe_values = pd.to_numeric(dsr_report["observed_sharpe"], errors="coerce")
        if sharpe_values.notna().any():
            best_observed_sharpe = float(sharpe_values.max())
    parts = [f"passing_strategy_count={passing_count}/{total_count}"]
    if math.isfinite(best_dsr):
        parts.append(f"best_dsr={best_dsr:.6f}")
    if math.isfinite(best_observed_sharpe):
        parts.append(f"best_observed_sharpe={best_observed_sharpe:.6f}")
    if best_strategy:
        parts.append(f"best_strategy={best_strategy}")
    return ";".join(parts)


def row_selected_fraction(row: pd.Series) -> float:
    value = row_float(row, "selected_fraction")
    if math.isfinite(value):
        return value
    selected_count = row_float(row, "selected_event_count")
    event_count = row_float(row, "event_count")
    if math.isfinite(selected_count) and math.isfinite(event_count) and event_count > 0:
        return selected_count / event_count
    return np.nan


def is_local_decision_candidate_row(row: pd.Series) -> bool:
    scope = str(row.get("candidate_scope", ""))
    horizon = as_float(row.get("horizon_days"))
    return scope == LOCAL_DECISION_CANDIDATE_SCOPE and math.isfinite(horizon) and int(horizon) == LOCAL_DECISION_HORIZON_DAYS


def is_display_only_or_diagnostic_status(value: object) -> bool:
    text = str(value).strip().upper()
    return any(token in text for token in ["DISPLAY_ONLY", "DIAGNOSTIC", "NO_MODEL"])


def evaluate_local_model_row(row: pd.Series, *, diagnostic: bool = False) -> list[dict[str, object]]:
    scope = str(row.get("candidate_scope", ""))
    horizon = row.get("horizon_days", "")
    model_name = str(row.get("model_name", ""))
    gate_group = "local_prediction_diagnostic" if diagnostic else "local_prediction"
    severity = "WARN" if diagnostic else "CRITICAL"
    min_oos = row_float(row, "min_decision_oos_events", MIN_DECISION_OOS_EVENTS)
    min_selected = row_float(row, "min_selected_oos_events", MIN_SELECTED_OOS_EVENTS)
    base_pr_auc = row_float(row, "base_rate_pr_auc")
    pr_auc = row_float(row, "pr_auc")
    decision_ece = row_float(row, "decision_ece", row_float(row, "ece"))
    decision_min_bin_n = row_float(row, "decision_min_calibration_bin_n", row_float(row, "min_calibration_bin_n"))
    fold_count = row_float(row, "fold_count", MIN_POSITIVE_EXPECTANCY_FOLDS)
    oos_count = row_float(row, "oos_event_count")
    selected_oos_count = row_float(row, "selected_oos_event_count")
    selection_contrast_available = (
        math.isfinite(oos_count)
        and math.isfinite(selected_oos_count)
        and selected_oos_count > 0
        and selected_oos_count < oos_count
    )
    rows = [
        gate_row(gate_group, "decision_scope_eligible", row_bool(row, "decision_scope_eligible"), row.get("decision_scope_eligible"), "True", "NOT_20D_TRADE_READY_DECISION_SCOPE", severity=severity, candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row(gate_group, "prediction_quality_pass", row_bool(row, "prediction_quality_pass"), row.get("prediction_quality_pass"), "True", "PREDICTION_QUALITY_FALSE", severity=severity, candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row(gate_group, "oos_event_count", row_float(row, "oos_event_count") >= min_oos, row_float(row, "oos_event_count"), f">={min_oos:.0f}", "OOS_EVENT_COUNT_LT_MIN", severity=severity, candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row(gate_group, "selected_oos_event_count", row_float(row, "selected_oos_event_count") >= min_selected, row_float(row, "selected_oos_event_count"), f">={min_selected:.0f}", "SELECTED_OOS_EVENT_COUNT_LT_MIN", severity=severity, candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row(gate_group, "selection_contrast_available", selection_contrast_available, f"selected={selected_oos_count:.0f};oos={oos_count:.0f}" if math.isfinite(oos_count) and math.isfinite(selected_oos_count) else "missing", "0<selected<oos", "SELECTION_CONTRAST_MISSING", severity=severity, candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row(gate_group, "min_selected_events_per_fold", row_float(row, "min_selected_events_per_fold") >= MIN_SELECTED_EVENTS_PER_FOLD, row_float(row, "min_selected_events_per_fold"), f">={MIN_SELECTED_EVENTS_PER_FOLD}", "SELECTED_EVENTS_PER_FOLD_LT_MIN", severity=severity, candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row(gate_group, "oos_fold_count", fold_count >= MIN_POSITIVE_EXPECTANCY_FOLDS, fold_count, f">={MIN_POSITIVE_EXPECTANCY_FOLDS}", "OOS_FOLD_COUNT_LT_MIN", severity=severity, candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row(gate_group, "brier_improvement_positive", row_float(row, "brier_improvement_pct") > 0.0, row_float(row, "brier_improvement_pct"), ">0", "NO_BRIER_IMPROVEMENT", severity=severity, candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row(gate_group, "decision_ece_within_limit", decision_ece <= DECISION_ECE_THRESHOLD, decision_ece, f"<={DECISION_ECE_THRESHOLD}", "ECE_GT_LIMIT", severity=severity, candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row(gate_group, "pr_auc_above_base", pd.notna(pr_auc) and pd.notna(base_pr_auc) and pr_auc > base_pr_auc, pr_auc, f">{base_pr_auc}", "PR_AUC_NOT_ABOVE_BASE", severity=severity, candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row(gate_group, "selected_minus_rule_all_positive", row_float(row, "selected_minus_rule_all_pct") > MIN_EXPECTANCY_IMPROVEMENT_PCT, row_float(row, "selected_minus_rule_all_pct"), f">{MIN_EXPECTANCY_IMPROVEMENT_PCT}", "ML_SELECTED_MINUS_RULE_ALL_LE_0", severity=severity, candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row(gate_group, "selected_expectancy_ci_lower_positive", row_float(row, "selected_signal_expectancy_ci_lower_pct") > MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT, row_float(row, "selected_signal_expectancy_ci_lower_pct"), f">{MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT}", "SELECTED_EXPECTANCY_CI_LOWER_LE_0", severity=severity, candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row(gate_group, "positive_expectancy_folds", row_float(row, "positive_expectancy_folds") >= MIN_POSITIVE_EXPECTANCY_FOLDS, row_float(row, "positive_expectancy_folds"), f">={MIN_POSITIVE_EXPECTANCY_FOLDS}", "POSITIVE_EXPECTANCY_FOLDS_LT_MIN", severity=severity, candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row(gate_group, "decision_min_calibration_bin_n", decision_min_bin_n >= MIN_CALIBRATION_BIN_N, decision_min_bin_n, f">={MIN_CALIBRATION_BIN_N}", "CALIBRATION_MIN_BIN_N_LT_MIN", severity=severity, candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row(gate_group, "fixed_width_calibration_bin_n_diagnostic", row_float(row, "fixed_width_min_calibration_bin_n", decision_min_bin_n) >= MIN_CALIBRATION_BIN_N, row_float(row, "fixed_width_min_calibration_bin_n", decision_min_bin_n), f">={MIN_CALIBRATION_BIN_N}", "FIXED_WIDTH_CALIBRATION_WARN", severity="WARN", candidate_scope=scope, horizon_days=horizon, model_name=model_name),
        gate_row(gate_group, "threshold_iqr", row_float(row, "threshold_iqr") <= MAX_THRESHOLD_IQR, row_float(row, "threshold_iqr"), f"<={MAX_THRESHOLD_IQR}", "THRESHOLD_IQR_GT_LIMIT", severity=severity, candidate_scope=scope, horizon_days=horizon, model_name=model_name),
    ]
    rank_row = rank_uplift_diagnostic_row(row)
    if rank_row is not None:
        rows.append(rank_row)
    return rows


def evaluate_local_models(comparison: pd.DataFrame, latest_prediction: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    use_status = latest_prediction.get("prediction_use_status", "UNKNOWN")
    local_support = is_prediction_decision_support(use_status)
    display_only_local_route = (not local_support) and is_display_only_or_diagnostic_status(use_status)
    local_missing_group = "local_prediction_diagnostic" if display_only_local_route else "local_prediction"
    local_missing_severity = "WARN" if display_only_local_route else "CRITICAL"
    if comparison.empty:
        rows.append(
            gate_row(
                local_missing_group,
                "comparison_available",
                False,
                "missing",
                "present",
                "LOCAL_MODEL_COMPARISON_MISSING",
                severity=local_missing_severity,
            )
        )
    else:
        has_decision_candidate = False
        for _, row in comparison.iterrows():
            is_decision_candidate = is_local_decision_candidate_row(row)
            has_decision_candidate = has_decision_candidate or is_decision_candidate
            rows.extend(evaluate_local_model_row(row, diagnostic=display_only_local_route or not is_decision_candidate))
        if not has_decision_candidate:
            rows.append(
                gate_row(
                    local_missing_group,
                    "local_20d_trade_ready_candidate_available",
                    False,
                    "missing",
                    f"{LOCAL_DECISION_CANDIDATE_SCOPE}/{LOCAL_DECISION_HORIZON_DAYS}D",
                    "LOCAL_20D_TRADE_READY_MODEL_MISSING",
                    severity=local_missing_severity,
                )
            )
    rows.append(
        gate_row(
            "local_latest",
            "latest_prediction_decision_support",
            is_prediction_decision_support(use_status),
            use_status,
            "DECISION_SUPPORT_ALLOWED",
            str(latest_prediction.get("model_quality_block_reasons", "LOCAL_LATEST_NOT_DECISION_SUPPORT")),
            severity="WARN" if display_only_local_route else "CRITICAL",
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
        scoring_route = str(pooled_latest.get("tsm_calibration_scoring_route", row.get("tsm_calibration_scoring_route", ""))).upper()
        scoring_reason = str(pooled_latest.get("tsm_calibration_scoring_route_reason", row.get("tsm_calibration_scoring_route_reason", "")))
        scoring_fallback = scoring_route == "POOLED_ONLY" and scoring_reason == "SELECTED_ROUTE_FAILED_IDENTITY_FALLBACK"
        tsm_route_pass = row_bool(row, "tsm_calibration_route_pass") if "tsm_calibration_route_pass" in row.index else tsm_ece <= MAX_TSM_ECE
        tsm_route_gate_pass = bool(tsm_route_pass or scoring_fallback)
        tsm_ece_gate_pass = bool(tsm_ece <= MAX_TSM_ECE or scoring_fallback)
        tsm_gate_severity = "WARN" if scoring_fallback and not (tsm_route_pass and tsm_ece <= MAX_TSM_ECE) else "CRITICAL"
        rows.extend(
            [
                gate_row(
                    "pooled_tsm_calibration",
                    "tsm_calibration_route_pass",
                    tsm_route_gate_pass,
                    row.get("tsm_calibration_route_failure_reasons", row.get("tsm_calibration_route_pass", "missing")),
                    "True",
                    "TSM_CALIBRATION_ROUTE_NOT_PASSED",
                    severity=tsm_gate_severity,
                    model_name=str(row.get("model_name", "")),
                    split=str(row.get("split", "")),
                ),
                gate_row(
                    "pooled_tsm_calibration",
                    "tsm_ece_within_limit",
                    tsm_ece_gate_pass,
                    tsm_ece,
                    f"<={MAX_TSM_ECE}",
                    "TSM_CALIBRATION_ECE_GT_LIMIT",
                    severity=tsm_gate_severity,
                    model_name=str(row.get("model_name", "")),
                    split=str(row.get("split", "")),
                ),
            ]
        )

    pooled_quality_rows = [
        row
        for row in rows
        if row.get("gate_group") in {"pooled_model", "pooled_tsm_calibration"}
        and row.get("severity") == "CRITICAL"
    ]
    pooled_model_quality_ready = bool(
        to_bool(pooled_latest.get("model_quality_pass", False))
        or (pooled_quality_rows and all(bool(row.get("passed")) for row in pooled_quality_rows))
    )
    latest_signal_pass = to_bool(pooled_latest.get("latest_signal_pass", False))
    decision_allowed = to_bool(pooled_latest.get("decision_support_allowed", False))
    latest_signal_severity = "WARN" if pooled_model_quality_ready and not latest_signal_pass else "CRITICAL"
    rows.extend(
        [
            gate_row("pooled_latest", "latest_signal_pass", latest_signal_pass, pooled_latest.get("latest_signal_pass", "missing"), "True", str(pooled_latest.get("latest_block_reasons", "POOLED_LATEST_SIGNAL_BLOCKED")), severity=latest_signal_severity),
            gate_row("pooled_latest", "pooled_prediction_decision_support", decision_allowed, pooled_latest.get("decision_support_allowed", "missing"), "True", str(pooled_latest.get("latest_block_reasons", "POOLED_LATEST_NOT_DECISION_SUPPORT")), severity=latest_signal_severity),
            gate_row("pooled_latest", "latest_stop_risk_within_limit", as_float(pooled_latest.get("p_stop_hit_20d")) <= MAX_STOP_HIT_FOR_LATEST, as_float(pooled_latest.get("p_stop_hit_20d")), f"<={MAX_STOP_HIT_FOR_LATEST}", "POOLED_STOP_RISK_GT_LIMIT", severity=latest_signal_severity),
            gate_row("pooled_latest", "latest_expected_r_above_min", as_float(pooled_latest.get("expected_r_net_20d")) >= MIN_EXPECTED_R_FOR_LATEST, as_float(pooled_latest.get("expected_r_net_20d")), f">={MIN_EXPECTED_R_FOR_LATEST}", "POOLED_EXPECTED_R_LT_MIN", severity=latest_signal_severity),
        ]
    )
    return rows


def evaluate_next_close_forecast(
    next_close_comparison: pd.DataFrame | None = None,
    next_close_quality: pd.DataFrame | None = None,
    next_close_latest: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    comparison = next_close_comparison if next_close_comparison is not None else pd.DataFrame()
    quality = next_close_quality if next_close_quality is not None else pd.DataFrame()
    latest = next_close_latest if next_close_latest is not None else {}
    severity = "WARN"
    scope = "next_close_forecast_top12"

    if comparison.empty:
        rows.append(
            gate_row(
                "next_close_forecast",
                "comparison_available",
                False,
                "missing",
                "present",
                "FORECAST_DIAGNOSTIC_COMPARISON_MISSING",
                severity=severity,
                candidate_scope=scope,
            )
        )
        return rows

    critical_failed = pd.DataFrame()
    if not quality.empty and {"severity", "passed"}.issubset(quality.columns):
        critical = quality[quality["severity"].astype(str).eq("CRITICAL")]
        critical_failed = critical[~critical["passed"].map(to_bool)]
    quality_pass = bool(not quality.empty and critical_failed.empty)
    rows.append(
        gate_row(
            "next_close_forecast",
            "critical_quality_checks_pass",
            quality_pass,
            int(len(critical_failed)) if not quality.empty else "missing",
            "0 critical failures",
            "FORECAST_DIAGNOSTIC_QUALITY_CHECK_FAILED",
            severity=severity,
            candidate_scope=scope,
        )
    )

    latest_status = str(latest.get("next_close_model_quality_status", latest.get("next_close_prediction_signal_status", "")))
    latest_available = bool(latest_status)
    rows.append(
        gate_row(
            "next_close_forecast",
            "latest_snapshot_available",
            latest_available,
            latest_status or "missing",
            "present",
            "FORECAST_DIAGNOSTIC_COMPARISON_MISSING",
            severity=severity,
            candidate_scope=scope,
        )
    )

    champion_rows: list[pd.Series] = []
    for horizon, group in comparison.groupby("horizon_days", dropna=False):
        ranked = group.sort_values(["rank_score", "mae_log_return"], ascending=[False, True]) if {"rank_score", "mae_log_return"}.issubset(group.columns) else group
        champion_rows.append(ranked.iloc[0])
    all_horizon_pass = bool(champion_rows and all(row_bool(row, "performance_quality_pass") for row in champion_rows))
    rows.append(
        gate_row(
            "next_close_forecast",
            "all_horizons_promoted_diagnostic",
            all_horizon_pass,
            latest_status or all_horizon_pass,
            "PROMOTED_FORECAST_DIAGNOSTIC",
            "FORECAST_DIAGNOSTIC_DISPLAY_ONLY_NOT_PROMOTED",
            severity=severity,
            candidate_scope=scope,
        )
    )

    for row in champion_rows:
        horizon = row.get("horizon_days", "")
        model_name = str(row.get("model_name", ""))
        metric_block = "FORECAST_DIAGNOSTIC_DISPLAY_ONLY_NOT_PROMOTED"
        rows.extend(
            [
                gate_row(
                    "next_close_forecast",
                    "champion_oos_event_count",
                    row_float(row, "oos_event_count") >= 5000,
                    row.get("oos_event_count", np.nan),
                    ">=5000",
                    metric_block,
                    severity=severity,
                    candidate_scope=scope,
                    horizon_days=horizon,
                    model_name=model_name,
                ),
                gate_row(
                    "next_close_forecast",
                    "champion_fold_count",
                    row_float(row, "fold_count") >= 4,
                    row.get("fold_count", np.nan),
                    ">=4",
                    metric_block,
                    severity=severity,
                    candidate_scope=scope,
                    horizon_days=horizon,
                    model_name=model_name,
                ),
                gate_row(
                    "next_close_forecast",
                    "champion_mae_improves_naive",
                    row_float(row, "mae_improvement_pct") >= 1.0,
                    row.get("mae_improvement_pct", np.nan),
                    ">=1%",
                    metric_block,
                    severity=severity,
                    candidate_scope=scope,
                    horizon_days=horizon,
                    model_name=model_name,
                ),
                gate_row(
                    "next_close_forecast",
                    "champion_rmse_not_worse_than_naive",
                    row_float(row, "rmse_delta_vs_naive", np.inf) <= 0.0,
                    row.get("rmse_delta_vs_naive", np.nan),
                    "<=0",
                    metric_block,
                    severity=severity,
                    candidate_scope=scope,
                    horizon_days=horizon,
                    model_name=model_name,
                ),
                gate_row(
                    "next_close_forecast",
                    "champion_interval_coverage_75_85",
                    0.75 <= row_float(row, "interval_coverage_80") <= 0.85,
                    row.get("interval_coverage_80", np.nan),
                    "0.75..0.85",
                    metric_block,
                    severity=severity,
                    candidate_scope=scope,
                    horizon_days=horizon,
                    model_name=model_name,
                ),
                gate_row(
                    "next_close_forecast",
                    "champion_direction_at_least_baseline",
                    row_float(row, "direction_accuracy") + 1e-12 >= row_float(row, "direction_threshold", 0.5),
                    row.get("direction_accuracy", np.nan),
                    f">={row.get('direction_threshold', 0.5)}",
                    metric_block,
                    severity=severity,
                    candidate_scope=scope,
                    horizon_days=horizon,
                    model_name=model_name,
                ),
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
    pooled_latest: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source, row = selected_cpcv_distribution_row(cpcv_strategy, cpcv_model)
    model_cpcv_core_pass = False
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
        model_cpcv_core_pass = bool(source == "model" and math.isfinite(median_uplift) and median_uplift > 0.0 and worst_quartile_or_drawdown_pass)
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
    dsr_is_strategy_diagnostic = bool(source == "model" and model_cpcv_core_pass)
    rows.append(
        gate_row(
            "strategy_validation_diagnostic" if dsr_is_strategy_diagnostic else "research_validation",
            "dsr_pass",
            dsr_pass,
            dsr_gate_value(dsr_report),
            "True",
            "DSR_NOT_PASSED",
            severity="WARN" if dsr_is_strategy_diagnostic else "CRITICAL",
            candidate_scope="strategy_dsr_diagnostic" if dsr_is_strategy_diagnostic else "",
        )
    )
    pooled_latest = pooled_latest or {}
    effective_n = np.nan
    effective_source_model = ""
    effective_source_split = ""
    for column in [
        "tsm_like_effective_calibration_n",
        "tsm_like_effective_train_validation_n",
    ]:
        if column in pooled_latest:
            effective_n = row_float(pd.Series(pooled_latest), column)
            if math.isfinite(effective_n):
                effective_source_model = str(pooled_latest.get("tsm_like_calibration_route", "tsm_like_overlay"))
                effective_source_split = "tsm_like_train_validation"
                break
    if tsm_calibration is None or tsm_calibration.empty:
        if math.isfinite(effective_n):
            rows.append(
                gate_row(
                    "research_validation",
                    "tsm_like_effective_calibration_n",
                    effective_n >= MIN_TSM_LIKE_EFFECTIVE_CALIBRATION_N,
                    effective_n,
                    f">={MIN_TSM_LIKE_EFFECTIVE_CALIBRATION_N}",
                    "TSM_LIKE_EFFECTIVE_N_LT_500",
                    model_name=effective_source_model,
                    split=effective_source_split,
                )
            )
            return rows
        rows.append(gate_row("research_validation", "tsm_like_effective_calibration_n", False, "missing", f">={MIN_TSM_LIKE_EFFECTIVE_CALIBRATION_N}", "TSM_LIKE_EFFECTIVE_N_LT_500"))
    else:
        selected_tsm = select_tsm_calibration_gate_row(tsm_calibration)
        row = selected_tsm.iloc[0]
        if not math.isfinite(effective_n):
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
                        effective_source_model = str(row.get("model_name", ""))
                        effective_source_split = str(row.get("split", ""))
                        break
        rows.append(
            gate_row(
                "research_validation",
                "tsm_like_effective_calibration_n",
                math.isfinite(effective_n) and effective_n >= MIN_TSM_LIKE_EFFECTIVE_CALIBRATION_N,
                effective_n if math.isfinite(effective_n) else "missing",
                f">={MIN_TSM_LIKE_EFFECTIVE_CALIBRATION_N}",
                "TSM_LIKE_EFFECTIVE_N_LT_500",
                model_name=effective_source_model,
                split=effective_source_split,
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
    next_close_comparison: pd.DataFrame | None = None,
    next_close_quality: pd.DataFrame | None = None,
    next_close_latest: dict[str, object] | None = None,
) -> pd.DataFrame:
    rows = []
    rows.extend(evaluate_local_models(comparison, latest_prediction))
    rows.extend(evaluate_pooled_models(pooled_comparison, pooled_latest, tsm_calibration))
    rows.extend(evaluate_next_close_forecast(next_close_comparison, next_close_quality, next_close_latest))
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
                pooled_latest,
            )
        )
    return pd.DataFrame(rows)


def unique_join(values: Iterable[object], limit: int = 8) -> str:
    clean = [str(v) for v in values if str(v) and str(v).lower() != "nan"]
    unique = sorted(set(clean))
    if len(unique) > limit:
        return "|".join(unique[:limit]) + f"|+{len(unique) - limit}"
    return "|".join(unique)


def root_cause_tokens_from_audit(audit: pd.DataFrame) -> pd.DataFrame:
    columns = list(audit.columns) + ["root_cause"] if not audit.empty else [
        "gate_group",
        "candidate_scope",
        "horizon_days",
        "model_name",
        "split",
        "gate",
        "passed",
        "severity",
        "value",
        "threshold",
        "block_reason",
        "root_cause",
    ]
    if audit.empty:
        return pd.DataFrame(columns=columns)
    failed = audit[~audit["passed"].astype(bool)].copy()
    if failed.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for _, row in failed.iterrows():
        raw_reasons = str(row.get("block_reason", "")).split("|")
        reasons = [reason[6:] if reason.startswith("MODEL:") else reason for reason in raw_reasons if reason and reason != "PASS"]
        if not reasons:
            reasons = ["UNKNOWN_FAILED_GATE"]
        for reason in reasons:
            out = row.to_dict()
            out["root_cause"] = reason
            rows.append(out)
    return pd.DataFrame(rows, columns=columns)


def performance_failed_rows(failed: pd.DataFrame) -> pd.DataFrame:
    if failed.empty:
        return failed.copy()
    perf = failed[~failed["gate_group"].astype(str).isin(LATEST_GATE_GROUPS)].copy()
    if perf.empty:
        return perf
    reasons = perf["block_reason"].astype(str)
    non_performance = reasons.map(lambda value: any(token in value for token in NON_PERFORMANCE_BLOCK_TOKENS))
    return perf[~non_performance].copy()


def performance_failure_family(row: pd.Series) -> str:
    reason = str(row.get("block_reason", "")).upper()
    gate = str(row.get("gate", "")).upper()
    text = f"{reason}|{gate}"
    if "PREDICTION_QUALITY_FALSE" in text or "PREDICTION_QUALITY_PASS" in text:
        return "aggregate_quality_flag"
    if any(token in text for token in ["OOS_EVENT_COUNT", "OOS_FOLD_COUNT", "SELECTED_OOS_EVENT", "SELECTED_EVENTS_PER_FOLD", "SELECTED_EVENTS_LT", "SELECTION_CONTRAST"]):
        return "sample_evidence"
    if "FIXED_WIDTH_CALIBRATION" in text:
        return "calibration_sample_warning"
    if "CALIBRATION_MIN_BIN_N" in text or "MIN_CALIBRATION_BIN" in text:
        return "sample_evidence"
    if "ECE" in text or "CALIBRATION" in text:
        return "probability_calibration"
    if "BRIER" in text:
        return "probabilistic_skill"
    if "PR_AUC" in text:
        return "discrimination"
    if any(token in text for token in ["SELECTED_MINUS", "SELECTED_EXPECTANCY", "POSITIVE_EXPECTANCY", "UPLIFT"]):
        return "economic_uplift"
    if "THRESHOLD" in text:
        return "threshold_stability"
    if any(token in text for token in ["DSR", "PBO", "CPCV"]):
        return "backtest_overfit_control"
    return "other_performance"


def rank_policy_context_key(row: pd.Series) -> tuple[str, str, str]:
    horizon_value = as_float(row.get("horizon_days"), np.nan)
    if math.isfinite(horizon_value):
        horizon_key = f"{horizon_value:.8f}"
    else:
        horizon_key = str(row.get("horizon_days", ""))
    return (str(row.get("candidate_scope", "")), horizon_key, str(row.get("model_name", "")))


def attach_rank_policy_context(rows: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["rank_policy_diagnostic_pass"] = False
    out["rank_top_quintile_minus_all_pct"] = np.nan
    out["rank_top_quintile_se_lower_pct"] = np.nan
    out["rank_policy_supported_candidate"] = False
    out["rank_policy_supported_threshold_warning"] = False
    out["rank_policy_supported_metric_warning"] = False
    out["rank_policy_context_reason"] = "NO_RANK_POLICY_PASS_CONTEXT"
    if out.empty or audit.empty or "gate" not in audit.columns:
        return out
    rank_rows = audit[audit["gate"].astype(str).eq(RANK_UPLIFT_DIAGNOSTIC_GATE)].copy()
    if rank_rows.empty:
        return out

    rank_rows["_rank_context_key"] = rank_rows.apply(rank_policy_context_key, axis=1)
    rank_rows["_rank_policy_pass"] = rank_rows.get(
        "rank_policy_diagnostic_pass",
        pd.Series(False, index=rank_rows.index),
    ).map(to_bool)
    rank_rows = rank_rows.drop_duplicates("_rank_context_key", keep="last")
    pass_map = dict(zip(rank_rows["_rank_context_key"], rank_rows["_rank_policy_pass"]))
    lower_map = dict(
        zip(
            rank_rows["_rank_context_key"],
            pd.to_numeric(rank_rows.get("rank_top_quintile_se_lower_pct", pd.Series(dtype=float)), errors="coerce"),
        )
    )
    uplift_map = dict(
        zip(
            rank_rows["_rank_context_key"],
            pd.to_numeric(rank_rows.get("value", pd.Series(dtype=float)), errors="coerce"),
        )
    )
    out["_rank_context_key"] = out.apply(rank_policy_context_key, axis=1)
    out["rank_policy_diagnostic_pass"] = out["_rank_context_key"].map(pass_map).fillna(False).map(bool)
    out["rank_top_quintile_se_lower_pct"] = out["_rank_context_key"].map(lower_map)
    out["rank_top_quintile_minus_all_pct"] = out["_rank_context_key"].map(uplift_map)
    out["rank_policy_supported_candidate"] = out["rank_policy_diagnostic_pass"]
    out["rank_policy_supported_threshold_warning"] = (
        out["rank_policy_supported_candidate"]
        & out["gate"].astype(str).isin(RANK_POLICY_THRESHOLD_SELECTED_GATES)
    )
    out["rank_policy_supported_metric_warning"] = out["rank_policy_supported_candidate"]
    out["rank_policy_context_reason"] = np.select(
        [
            out["rank_policy_supported_threshold_warning"],
            out["rank_policy_supported_metric_warning"],
            out["rank_policy_supported_candidate"],
        ],
        [
            "RANK_POLICY_PASS_THRESHOLD_SELECTED_WARNING",
            "RANK_POLICY_PASS_METRIC_WARNING",
            "RANK_POLICY_PASS_AGGREGATE_OR_EVIDENCE_WARNING",
        ],
        default="NO_RANK_POLICY_PASS_CONTEXT",
    )
    return out.drop(columns=["_rank_context_key"])


def attach_evidence_limited_metric_context(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["same_candidate_evidence_gap"] = False
    out["evidence_limited_metric_warning"] = False
    out["evidence_context_reason"] = "NO_EVIDENCE_GAP_CONTEXT"
    required = {"performance_failure_family", "performance_failure_kind"}
    if out.empty or not required.issubset(out.columns):
        return out

    evidence = out[out["performance_failure_family"].astype(str).isin(PERFORMANCE_EVIDENCE_FAMILIES)].copy()
    if evidence.empty:
        return out

    evidence["_evidence_context_key"] = evidence.apply(rank_policy_context_key, axis=1)
    reason_map = (
        evidence.groupby("_evidence_context_key")["block_reason"]
        .apply(lambda values: unique_join(values, limit=6))
        .to_dict()
    )
    out["_evidence_context_key"] = out.apply(rank_policy_context_key, axis=1)
    out["same_candidate_evidence_gap"] = out["_evidence_context_key"].isin(reason_map)
    metric_shortfall = out["performance_failure_kind"].astype(str).eq("metric_shortfall")
    evidence_reason = out["_evidence_context_key"].map(reason_map).fillna("")
    evidence_tokens = evidence_reason.astype(str).str.split("|")
    selection_contrast_only = evidence_tokens.map(
        lambda values: bool(values) and set(values).issubset({"SELECTION_CONTRAST_MISSING"})
    )
    selection_contrast_metric = out["gate"].astype(str).isin(SELECTED_CONTRAST_DEPENDENT_GATES)
    out["evidence_limited_metric_warning"] = (
        out["same_candidate_evidence_gap"]
        & metric_shortfall
        & (~selection_contrast_only | selection_contrast_metric)
    )
    out["evidence_context_reason"] = np.where(
        out["evidence_limited_metric_warning"],
        evidence_reason.replace("", "EVIDENCE_GAP_CONTEXT"),
        "NO_EVIDENCE_GAP_CONTEXT",
    )
    return out.drop(columns=["_evidence_context_key"])


def attach_rejected_model_context(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["rejected_model_metric_warning"] = False
    out["model_rejection_reason"] = "NO_MODEL_REJECTION_CONTEXT"
    required = {"performance_failure_kind", "gate"}
    if out.empty or not required.issubset(out.columns):
        return out

    local_diagnostic = out["gate_group"].astype(str).eq("local_prediction_diagnostic")
    metric_shortfall = out["performance_failure_kind"].astype(str).eq("metric_shortfall")
    candidate_rows = out[local_diagnostic].copy()
    if candidate_rows.empty:
        return out

    candidate_rows["_model_context_key"] = candidate_rows.apply(rank_policy_context_key, axis=1)
    rejected_reasons: dict[tuple[str, str, str], str] = {}
    for key, group in candidate_rows.groupby("_model_context_key", dropna=False):
        failed_gates = set(group["gate"].astype(str))
        rank_supported = group.get(
            "rank_policy_supported_candidate",
            pd.Series(False, index=group.index),
        ).map(to_bool).any()
        if rank_supported:
            continue
        if MODEL_REJECTION_SKILL_GATES.issubset(failed_gates):
            rejected_reasons[key] = "REJECTED_NO_BRIER_SKILL_AND_PR_AUC_BELOW_BASE"
            continue
        no_probabilistic_economic_skill = (
            "brier_improvement_positive" in failed_gates
            and bool(
                {
                    "selected_minus_rule_all_positive",
                    "selected_expectancy_ci_lower_positive",
                    "positive_expectancy_folds",
                }
                & failed_gates
            )
        )
        if no_probabilistic_economic_skill:
            rejected_reasons[key] = "REJECTED_NO_BRIER_SKILL_AND_NO_ACTIONABLE_ECONOMIC_UPLIFT"
            continue
        no_discriminative_action_skill = (
            "pr_auc_above_base" in failed_gates
            and bool({"selection_contrast_available", "selected_minus_rule_all_positive"} & failed_gates)
        )
        if no_discriminative_action_skill:
            rejected_reasons[key] = "REJECTED_NO_DISCRIMINATION_OR_ACTIONABLE_SELECTION_SKILL"

    if not rejected_reasons:
        return out

    out["_model_context_key"] = out.apply(rank_policy_context_key, axis=1)
    rejected_metric = local_diagnostic & metric_shortfall & out["_model_context_key"].isin(rejected_reasons)
    out["rejected_model_metric_warning"] = rejected_metric
    rejection_reason = out["_model_context_key"].map(rejected_reasons).fillna("NO_MODEL_REJECTION_CONTEXT")
    out["model_rejection_reason"] = np.where(
        rejected_metric,
        rejection_reason,
        "NO_MODEL_REJECTION_CONTEXT",
    )
    return out.drop(columns=["_model_context_key"])


def build_performance_gate_audit(audit: pd.DataFrame) -> pd.DataFrame:
    base_columns = list(audit.columns) if not audit.empty else [
        "gate_group",
        "candidate_scope",
        "horizon_days",
        "model_name",
        "split",
        "gate",
        "passed",
        "severity",
        "value",
        "threshold",
        "block_reason",
    ]
    extra_columns = [
        "performance_gate_scope",
        "performance_scope_detail",
        "performance_failure_family",
        "performance_failure_kind",
        "active_performance_gate",
        "performance_blocks_model_gate",
        "excluded_from_active_performance_gate",
        "performance_diagnostic_reason",
        "rank_policy_diagnostic_pass",
        "rank_top_quintile_minus_all_pct",
        "rank_top_quintile_se_lower_pct",
        "rank_policy_supported_candidate",
        "rank_policy_supported_threshold_warning",
        "rank_policy_supported_metric_warning",
        "rank_policy_context_reason",
        "same_candidate_evidence_gap",
        "evidence_limited_metric_warning",
        "evidence_context_reason",
        "rejected_model_metric_warning",
        "model_rejection_reason",
        "strategy_diagnostic_metric_warning",
    ]
    if audit.empty:
        return pd.DataFrame(columns=base_columns + extra_columns)
    failed = audit[~audit["passed"].astype(bool)].copy()
    perf = performance_failed_rows(failed)
    if perf.empty:
        return pd.DataFrame(columns=base_columns + extra_columns)
    perf = attach_rank_policy_context(perf, audit)
    active_mask = perf["gate_group"].astype(str).isin(ACTIVE_PERFORMANCE_GATE_GROUPS)
    perf["performance_gate_scope"] = np.where(active_mask, "active", "diagnostic")
    perf["performance_scope_detail"] = perf["gate_group"].astype(str).map(
        {
            "pooled_model": "active_pooled_model_quality_gate",
            "pooled_tsm_calibration": "active_tsm_scoring_calibration_gate",
            "research_validation": "active_research_validation_gate",
            "local_prediction_diagnostic": "inactive_local_prediction_diagnostic",
            "strategy_validation_diagnostic": "nonblocking_strategy_validation_diagnostic",
        }
    ).fillna("nonblocking_performance_diagnostic")
    perf["performance_failure_family"] = perf.apply(performance_failure_family, axis=1)
    perf["performance_failure_kind"] = np.select(
        [
            perf["performance_failure_family"].isin(PERFORMANCE_AGGREGATE_FAMILIES),
            perf["performance_failure_family"].isin(PERFORMANCE_EVIDENCE_FAMILIES),
        ],
        ["aggregate_flag", "evidence_gap"],
        default="metric_shortfall",
    )
    perf = attach_evidence_limited_metric_context(perf)
    perf = attach_rejected_model_context(perf)
    perf["strategy_diagnostic_metric_warning"] = (
        perf["gate_group"].astype(str).eq("strategy_validation_diagnostic")
        & perf["performance_failure_kind"].astype(str).eq("metric_shortfall")
    )
    perf["active_performance_gate"] = active_mask
    perf["performance_blocks_model_gate"] = active_mask & perf["severity"].astype(str).eq("CRITICAL")
    perf["excluded_from_active_performance_gate"] = ~active_mask
    perf["performance_diagnostic_reason"] = np.where(
        active_mask,
        "ACTIVE_MODEL_PERFORMANCE_GATE",
        np.where(
            perf["gate_group"].astype(str).eq("local_prediction_diagnostic"),
            "INACTIVE_OR_DISPLAY_ONLY_LOCAL_MODEL_CANDIDATE",
            "NONBLOCKING_DIAGNOSTIC_VALIDATION_EVIDENCE",
        ),
    )
    output_columns = list(dict.fromkeys(base_columns + extra_columns))
    return perf[output_columns].reset_index(drop=True)


def build_unresolved_performance_priorities(performance_audit: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "priority",
        "performance_failure_family",
        "performance_failure_kind",
        "gate",
        "block_reason",
        "candidate_scope",
        "horizon_days",
        "model_name",
        "warning_count",
        "active_warning_count",
        "rank_policy_supported_count",
        "value_min",
        "value_max",
        "recommended_action",
    ]
    if performance_audit.empty:
        return pd.DataFrame(columns=columns)
    perf = performance_audit.copy()
    if "performance_failure_kind" not in perf.columns:
        perf["performance_failure_family"] = perf.apply(performance_failure_family, axis=1)
        perf["performance_failure_kind"] = np.where(
            perf["performance_failure_family"].isin(PERFORMANCE_NON_METRIC_FAMILIES),
            "evidence_or_aggregate",
            "metric_shortfall",
        )
    rank_metric_supported = perf.get(
        "rank_policy_supported_metric_warning",
        pd.Series(False, index=perf.index),
    ).map(to_bool)
    evidence_limited_metric = perf.get(
        "evidence_limited_metric_warning",
        pd.Series(False, index=perf.index),
    ).map(to_bool)
    rejected_model_metric = perf.get(
        "rejected_model_metric_warning",
        pd.Series(False, index=perf.index),
    ).map(to_bool)
    strategy_diagnostic_metric = perf.get(
        "strategy_diagnostic_metric_warning",
        pd.Series(False, index=perf.index),
    ).map(to_bool)
    unresolved = perf[
        perf["performance_failure_kind"].astype(str).eq("metric_shortfall")
        & ~rank_metric_supported
        & ~evidence_limited_metric
        & ~rejected_model_metric
        & ~strategy_diagnostic_metric
    ].copy()
    if unresolved.empty:
        return pd.DataFrame(columns=columns)
    unresolved["priority"] = unresolved["performance_failure_family"].astype(str).map(UNRESOLVED_PERFORMANCE_PRIORITY).fillna(5).astype(int)
    unresolved["rank_policy_supported_candidate_bool"] = unresolved.get(
        "rank_policy_supported_candidate",
        pd.Series(False, index=unresolved.index),
    ).map(to_bool)
    unresolved["active_performance_gate_bool"] = unresolved.get(
        "active_performance_gate",
        pd.Series(False, index=unresolved.index),
    ).map(to_bool)
    numeric_values = pd.to_numeric(unresolved.get("value", pd.Series(dtype=float)), errors="coerce")
    unresolved["_numeric_value"] = numeric_values
    group_cols = [
        "priority",
        "performance_failure_family",
        "performance_failure_kind",
        "gate",
        "block_reason",
        "candidate_scope",
        "horizon_days",
        "model_name",
    ]
    grouped = unresolved.groupby(group_cols, dropna=False)
    rows = []
    for keys, group in grouped:
        row = dict(zip(group_cols, keys))
        values = pd.to_numeric(group["_numeric_value"], errors="coerce").dropna()
        family = str(row["performance_failure_family"])
        row.update(
            {
                "warning_count": int(len(group)),
                "active_warning_count": int(group["active_performance_gate_bool"].sum()),
                "rank_policy_supported_count": int(group["rank_policy_supported_candidate_bool"].sum()),
                "value_min": float(values.min()) if not values.empty else np.nan,
                "value_max": float(values.max()) if not values.empty else np.nan,
                "recommended_action": UNRESOLVED_PERFORMANCE_ACTIONS.get(family, UNRESOLVED_PERFORMANCE_ACTIONS["other_performance"]),
            }
        )
        rows.append(row)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["active_warning_count", "priority", "warning_count", "performance_failure_family", "candidate_scope", "horizon_days", "model_name"],
            ascending=[False, True, False, True, True, True, True],
        )
        .reset_index(drop=True)
    )


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
    exploded = root_cause_tokens_from_audit(audit)
    if exploded.empty:
        return pd.DataFrame()
    rows = []
    for reason, group in exploded.groupby("root_cause", dropna=False):
        priority, category, action = ROOT_CAUSE_GUIDANCE.get(
            str(reason),
            (3, "uncategorized_gate_failure", "Inspect the failed gate rows for this uncategorized reason."),
        )
        numeric_values = pd.to_numeric(group.get("value", pd.Series(dtype=float)), errors="coerce").dropna()
        critical_failed_gate_count = int(group.get("severity", pd.Series(dtype=str)).astype(str).eq("CRITICAL").sum())
        warn_failed_gate_count = int(group.get("severity", pd.Series(dtype=str)).astype(str).eq("WARN").sum())
        rows.append(
            {
                "root_cause": reason,
                "priority": priority,
                "failed_gate_count": int(len(group)),
                "critical_failed_gate_count": critical_failed_gate_count,
                "warn_failed_gate_count": warn_failed_gate_count,
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
        .assign(has_critical_failure=lambda frame: frame["critical_failed_gate_count"].astype(int) > 0)
        .sort_values(["has_critical_failure", "priority", "critical_failed_gate_count", "failed_gate_count", "root_cause"], ascending=[False, True, False, False, True])
        .drop(columns=["has_critical_failure"])
        .reset_index(drop=True)
    )


def performance_warning_resolution_bucket(row: pd.Series) -> str:
    if to_bool(row.get("active_performance_gate", False)):
        return "active_model_performance_gate"
    if to_bool(row.get("strategy_diagnostic_metric_warning", False)):
        return "strategy_diagnostic_nonblocking"
    if to_bool(row.get("rejected_model_metric_warning", False)):
        return "rejected_model_metric_warning"
    if to_bool(row.get("rank_policy_supported_metric_warning", False)):
        return "rank_policy_supported_metric_warning"
    if to_bool(row.get("evidence_limited_metric_warning", False)):
        return "evidence_limited_metric_warning"
    kind = str(row.get("performance_failure_kind", ""))
    if kind == "evidence_gap":
        return "diagnostic_evidence_gap"
    if kind == "aggregate_flag":
        return "diagnostic_aggregate_quality_flag"
    return "unresolved_diagnostic_warning"


def build_performance_warning_resolution_summary(performance_audit: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "resolution_bucket",
        "performance_failure_family",
        "performance_failure_kind",
        "warning_count",
        "metric_warning_count",
        "evidence_gap_count",
        "aggregate_flag_count",
        "active_warning_count",
        "rank_policy_supported_count",
        "evidence_limited_metric_count",
        "rejected_model_metric_count",
        "strategy_diagnostic_metric_count",
        "candidate_scopes",
        "horizons",
        "models",
        "gates",
        "block_reasons",
    ]
    if performance_audit.empty:
        return pd.DataFrame(columns=columns)
    perf = performance_audit.copy()
    if "performance_failure_family" not in perf.columns:
        perf["performance_failure_family"] = perf.apply(performance_failure_family, axis=1)
    if "performance_failure_kind" not in perf.columns:
        perf["performance_failure_kind"] = np.select(
            [
                perf["performance_failure_family"].isin(PERFORMANCE_AGGREGATE_FAMILIES),
                perf["performance_failure_family"].isin(PERFORMANCE_EVIDENCE_FAMILIES),
            ],
            ["aggregate_flag", "evidence_gap"],
            default="metric_shortfall",
        )
    perf["resolution_bucket"] = perf.apply(performance_warning_resolution_bucket, axis=1)

    def bool_sum(frame: pd.DataFrame, column: str) -> int:
        return int(frame.get(column, pd.Series(False, index=frame.index)).map(to_bool).sum())

    rows = []
    group_cols = ["resolution_bucket", "performance_failure_family", "performance_failure_kind"]
    for keys, group in perf.groupby(group_cols, dropna=False):
        kind = group["performance_failure_kind"].astype(str)
        rows.append(
            {
                "resolution_bucket": keys[0],
                "performance_failure_family": keys[1],
                "performance_failure_kind": keys[2],
                "warning_count": int(len(group)),
                "metric_warning_count": int(kind.eq("metric_shortfall").sum()),
                "evidence_gap_count": int(kind.eq("evidence_gap").sum()),
                "aggregate_flag_count": int(kind.eq("aggregate_flag").sum()),
                "active_warning_count": bool_sum(group, "active_performance_gate"),
                "rank_policy_supported_count": bool_sum(group, "rank_policy_supported_candidate"),
                "evidence_limited_metric_count": bool_sum(group, "evidence_limited_metric_warning"),
                "rejected_model_metric_count": bool_sum(group, "rejected_model_metric_warning"),
                "strategy_diagnostic_metric_count": bool_sum(group, "strategy_diagnostic_metric_warning"),
                "candidate_scopes": unique_join(group.get("candidate_scope", pd.Series(dtype=object)), limit=8),
                "horizons": unique_join(group.get("horizon_days", pd.Series(dtype=object)), limit=8),
                "models": unique_join(group.get("model_name", pd.Series(dtype=object)), limit=8),
                "gates": unique_join(group.get("gate", pd.Series(dtype=object)), limit=8),
                "block_reasons": unique_join(group.get("block_reason", pd.Series(dtype=object)), limit=8),
            }
        )
    order = {
        "active_model_performance_gate": 0,
        "unresolved_diagnostic_warning": 1,
        "rejected_model_metric_warning": 2,
        "rank_policy_supported_metric_warning": 3,
        "evidence_limited_metric_warning": 4,
        "diagnostic_evidence_gap": 5,
        "diagnostic_aggregate_quality_flag": 6,
        "strategy_diagnostic_nonblocking": 7,
    }
    return (
        pd.DataFrame(rows, columns=columns)
        .assign(_sort_bucket=lambda frame: frame["resolution_bucket"].map(order).fillna(99).astype(int))
        .sort_values(["_sort_bucket", "warning_count", "performance_failure_family"], ascending=[True, False, True])
        .drop(columns=["_sort_bucket"])
        .reset_index(drop=True)
    )


def candidate_disposition_action(disposition: str) -> str:
    return {
        "local_decision_candidate_pass": "eligible_for_local_decision_support_if_pipeline_route_selects_local_model",
        "local_quality_pass_scope_blocked": "keep_display_only_until_candidate_scope_matches_decision_contract",
        "next_day_directional_display_pass": "keep_as_display_only_directional_signal_do_not_promote_to_20d_trade_decision",
        "next_day_rejected_challenger": "retain_as_rejected_next_day_challenger_do_not_count_as_actionable_performance_failure",
        "next_close_forecast_promoted_diagnostic": "show_as_promoted_forecast_diagnostic_do_not_enable_trade_gate",
        "next_close_forecast_display_only": "show_forecast_values_but_keep_model_display_only_until_all_horizons_pass",
        "rank_policy_supported_diagnostic": "preserve_as_research_signal_and_improve_calibration_or_sample_before_promotion",
        "evidence_limited_diagnostic": "expand_oos_and_selected_event_evidence_before_model_promotion",
        "rejected_model_diagnostic": "retire_or_keep_as_baseline_do_not_tune_for_decision_support",
        "strategy_diagnostic_nonblocking": "keep_strategy_overfit_warning_nonblocking_while_active_model_cpcv_pbo_passes",
        "latest_signal_standby": "wait_for_latest_trade_ready_signal_without_changing_model_quality_gate",
        "evidence_or_aggregate_diagnostic": "inspect_evidence_or_aggregate_quality_before_promotion",
        "metric_warning_needs_review": "investigate_unclassified_metric_warning_before_next_promotion",
        "diagnostic_pass": "no_action_required",
    }.get(disposition, "inspect_candidate_disposition")


def build_model_candidate_disposition_summary(
    audit: pd.DataFrame,
    performance_audit: pd.DataFrame,
    comparison: pd.DataFrame | None = None,
    next_day_comparison: pd.DataFrame | None = None,
    next_close_comparison: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = [
        "candidate_scope",
        "horizon_days",
        "model_name",
        "model_role",
        "disposition",
        "warning_count",
        "metric_warning_count",
        "evidence_gap_count",
        "aggregate_flag_count",
        "prediction_quality_pass",
        "performance_quality_pass",
        "decision_scope_eligible",
        "rank_policy_diagnostic_pass",
        "directional_diagnostic_quality_pass",
        "oos_event_count",
        "selected_oos_event_count",
        "brier_improvement_pct",
        "decision_ece",
        "selected_minus_rule_all_pct",
        "selected_signal_expectancy_ci_lower_pct",
        "positive_expectancy_folds",
        "min_selected_events_per_fold",
        "threshold_iqr",
        "min_calibration_bin_n",
        "fixed_width_min_calibration_bin_n",
        "rank_top_quintile_minus_all_pct",
        "rank_top_quintile_se_lower_pct",
        "gates",
        "block_reasons",
        "recommended_action",
    ]
    comparison = comparison if comparison is not None else pd.DataFrame()
    next_day_comparison = next_day_comparison if next_day_comparison is not None else pd.DataFrame()
    next_close_comparison = next_close_comparison if next_close_comparison is not None else pd.DataFrame()
    key_cols = ["candidate_scope", "horizon_days", "model_name"]
    rows_by_key: dict[tuple[str, str, str], dict[str, object]] = {}

    def norm(value: object) -> str:
        text = str(value)
        return "" if text.lower() == "nan" else text

    def key_from_values(scope: object, horizon: object, model: object) -> tuple[str, str, str]:
        return (norm(scope), norm(horizon), norm(model))

    def seed_comparison_rows(frame: pd.DataFrame, model_role: str) -> None:
        if frame.empty:
            return
        for _, row in frame.iterrows():
            key = key_from_values(row.get("candidate_scope"), row.get("horizon_days"), row.get("model_name"))
            performance_pass = to_bool(row.get("performance_quality_pass", False))
            performance_reasons = str(row.get("performance_quality_block_reasons", "PASS") or "PASS")
            if performance_reasons == "nan" or not performance_reasons:
                performance_reasons = str(row.get("quality_block_reasons", "PASS") or "PASS")
            reason_tokens = [token for token in performance_reasons.split("|") if token and token != "PASS"]
            rows_by_key[key] = {
                "candidate_scope": row.get("candidate_scope"),
                "horizon_days": row.get("horizon_days"),
                "model_name": row.get("model_name"),
                "model_role": model_role,
                "warning_count": len(reason_tokens) if model_role in {"next_day_directional_candidate", "next_close_forecast_candidate"} else 0,
                "metric_warning_count": len(reason_tokens) if model_role in {"next_day_directional_candidate", "next_close_forecast_candidate"} else 0,
                "evidence_gap_count": 0,
                "aggregate_flag_count": 0,
                "prediction_quality_pass": to_bool(row.get("prediction_quality_pass", False)),
                "performance_quality_pass": performance_pass,
                "decision_scope_eligible": to_bool(row.get("decision_scope_eligible", False)),
                "rank_policy_diagnostic_pass": to_bool(row.get("rank_policy_diagnostic_pass", False)),
                "directional_diagnostic_quality_pass": to_bool(row.get("directional_diagnostic_quality_pass", False)),
                "oos_event_count": row.get("oos_event_count", np.nan),
                "selected_oos_event_count": row.get("selected_oos_event_count", np.nan),
                "brier_improvement_pct": row.get("brier_improvement_pct", np.nan),
                "decision_ece": row.get("decision_ece", np.nan),
                "selected_minus_rule_all_pct": row.get("selected_minus_rule_all_pct", np.nan),
                "selected_signal_expectancy_ci_lower_pct": row.get("selected_signal_expectancy_ci_lower_pct", np.nan),
                "positive_expectancy_folds": row.get("positive_expectancy_folds", np.nan),
                "min_selected_events_per_fold": row.get("min_selected_events_per_fold", np.nan),
                "threshold_iqr": row.get("threshold_iqr", np.nan),
                "min_calibration_bin_n": row.get("min_calibration_bin_n", np.nan),
                "fixed_width_min_calibration_bin_n": row.get("fixed_width_min_calibration_bin_n", np.nan),
                "rank_top_quintile_minus_all_pct": row.get("rank_top_quintile_minus_all_pct", np.nan),
                "rank_top_quintile_se_lower_pct": row.get("rank_top_quintile_se_lower_pct", np.nan),
                "gates": "PASS" if performance_pass else f"{model_role}_performance_quality_pass",
                "block_reasons": "PASS" if performance_pass else performance_reasons,
            }

    seed_comparison_rows(comparison, "local_prediction_candidate")
    seed_comparison_rows(next_day_comparison, "next_day_directional_candidate")
    seed_comparison_rows(next_close_comparison, "next_close_forecast_candidate")

    failed = audit[~audit["passed"].astype(bool)].copy() if not audit.empty else pd.DataFrame()
    if not failed.empty:
        for key, group in failed.groupby(key_cols, dropna=False):
            key = tuple(norm(v) for v in key)
            row = rows_by_key.get(
                key,
                {
                    "candidate_scope": group["candidate_scope"].iloc[0] if "candidate_scope" in group else np.nan,
                    "horizon_days": group["horizon_days"].iloc[0] if "horizon_days" in group else np.nan,
                    "model_name": group["model_name"].iloc[0] if "model_name" in group else np.nan,
                    "prediction_quality_pass": False,
                    "performance_quality_pass": False,
                    "decision_scope_eligible": False,
                    "rank_policy_diagnostic_pass": False,
                    "directional_diagnostic_quality_pass": False,
                    "oos_event_count": np.nan,
                    "selected_oos_event_count": np.nan,
                    "brier_improvement_pct": np.nan,
                    "decision_ece": np.nan,
                    "selected_minus_rule_all_pct": np.nan,
                    "selected_signal_expectancy_ci_lower_pct": np.nan,
                    "positive_expectancy_folds": np.nan,
                    "min_selected_events_per_fold": np.nan,
                    "threshold_iqr": np.nan,
                    "min_calibration_bin_n": np.nan,
                    "fixed_width_min_calibration_bin_n": np.nan,
                    "rank_top_quintile_minus_all_pct": np.nan,
                    "rank_top_quintile_se_lower_pct": np.nan,
                },
            )
            groups = set(group["gate_group"].astype(str))
            if "pooled_latest" in groups:
                row["model_role"] = "latest_signal_gate"
            elif "strategy_validation_diagnostic" in groups:
                row["model_role"] = "strategy_validation_diagnostic"
            else:
                row["model_role"] = row.get("model_role", "local_prediction_candidate")
            row["warning_count"] = int(len(group))
            row["gates"] = unique_join(group.get("gate", pd.Series(dtype=object)), limit=12)
            row["block_reasons"] = unique_join(group.get("block_reason", pd.Series(dtype=object)), limit=12)
            rows_by_key[key] = row

    if not performance_audit.empty:
        for key, group in performance_audit.groupby(key_cols, dropna=False):
            key = tuple(norm(v) for v in key)
            row = rows_by_key.get(
                key,
                {
                    "candidate_scope": group["candidate_scope"].iloc[0],
                    "horizon_days": group["horizon_days"].iloc[0],
                    "model_name": group["model_name"].iloc[0],
                    "model_role": "diagnostic_performance_candidate",
                    "warning_count": int(len(group)),
                    "prediction_quality_pass": False,
                    "performance_quality_pass": False,
                    "decision_scope_eligible": False,
                    "rank_policy_diagnostic_pass": False,
                    "directional_diagnostic_quality_pass": False,
                    "oos_event_count": np.nan,
                    "selected_oos_event_count": np.nan,
                    "brier_improvement_pct": np.nan,
                    "decision_ece": np.nan,
                    "selected_minus_rule_all_pct": np.nan,
                    "selected_signal_expectancy_ci_lower_pct": np.nan,
                    "positive_expectancy_folds": np.nan,
                    "min_selected_events_per_fold": np.nan,
                    "threshold_iqr": np.nan,
                    "min_calibration_bin_n": np.nan,
                    "fixed_width_min_calibration_bin_n": np.nan,
                    "rank_top_quintile_minus_all_pct": np.nan,
                    "rank_top_quintile_se_lower_pct": np.nan,
                    "gates": unique_join(group.get("gate", pd.Series(dtype=object)), limit=12),
                    "block_reasons": unique_join(group.get("block_reason", pd.Series(dtype=object)), limit=12),
                },
            )
            kind = group["performance_failure_kind"].astype(str)
            row["metric_warning_count"] = int(kind.eq("metric_shortfall").sum())
            row["evidence_gap_count"] = int(kind.eq("evidence_gap").sum())
            row["aggregate_flag_count"] = int(kind.eq("aggregate_flag").sum())
            row["_rank_supported"] = group.get("rank_policy_supported_metric_warning", pd.Series(False, index=group.index)).map(to_bool).any()
            row["_evidence_limited"] = group.get("evidence_limited_metric_warning", pd.Series(False, index=group.index)).map(to_bool).any()
            row["_rejected"] = group.get("rejected_model_metric_warning", pd.Series(False, index=group.index)).map(to_bool).any()
            row["_strategy"] = group.get("strategy_diagnostic_metric_warning", pd.Series(False, index=group.index)).map(to_bool).any()
            rows_by_key[key] = row

    rows = []
    for row in rows_by_key.values():
        warning_count = int(row.get("warning_count", 0) or 0)
        metric_count = int(row.get("metric_warning_count", 0) or 0)
        evidence_count = int(row.get("evidence_gap_count", 0) or 0)
        aggregate_count = int(row.get("aggregate_flag_count", 0) or 0)
        if row.get("model_role") == "latest_signal_gate":
            disposition = "latest_signal_standby"
        elif row.get("model_role") == "next_day_directional_candidate" and to_bool(row.get("performance_quality_pass")):
            disposition = "next_day_directional_display_pass"
        elif row.get("model_role") == "next_day_directional_candidate":
            disposition = "next_day_rejected_challenger"
        elif row.get("model_role") == "next_close_forecast_candidate" and to_bool(row.get("performance_quality_pass")):
            disposition = "next_close_forecast_promoted_diagnostic"
        elif row.get("model_role") == "next_close_forecast_candidate":
            disposition = "next_close_forecast_display_only"
        elif row.get("_strategy"):
            disposition = "strategy_diagnostic_nonblocking"
        elif row.get("_rejected"):
            disposition = "rejected_model_diagnostic"
        elif row.get("_rank_supported"):
            disposition = "rank_policy_supported_diagnostic"
        elif row.get("_evidence_limited"):
            disposition = "evidence_limited_diagnostic"
        elif metric_count > 0:
            disposition = "metric_warning_needs_review"
        elif evidence_count > 0 or aggregate_count > 0:
            disposition = "evidence_or_aggregate_diagnostic"
        elif to_bool(row.get("prediction_quality_pass")) and to_bool(row.get("performance_quality_pass")) and not to_bool(row.get("decision_scope_eligible")):
            disposition = "local_quality_pass_scope_blocked"
        elif warning_count == 0:
            disposition = "diagnostic_pass"
        else:
            disposition = "evidence_or_aggregate_diagnostic"
        clean = {col: row.get(col, np.nan) for col in columns if col not in {"disposition", "recommended_action"}}
        clean.update(
            {
                "disposition": disposition,
                "warning_count": warning_count,
                "metric_warning_count": metric_count,
                "evidence_gap_count": evidence_count,
                "aggregate_flag_count": aggregate_count,
                "recommended_action": candidate_disposition_action(disposition),
            }
        )
        rows.append(clean)
    if not rows:
        return pd.DataFrame(columns=columns)
    disposition_order = {
        "metric_warning_needs_review": 0,
        "local_quality_pass_scope_blocked": 1,
        "next_day_directional_display_pass": 2,
        "next_day_rejected_challenger": 3,
        "next_close_forecast_promoted_diagnostic": 4,
        "next_close_forecast_display_only": 5,
        "rank_policy_supported_diagnostic": 6,
        "evidence_limited_diagnostic": 7,
        "evidence_or_aggregate_diagnostic": 8,
        "rejected_model_diagnostic": 9,
        "strategy_diagnostic_nonblocking": 10,
        "latest_signal_standby": 11,
        "diagnostic_pass": 12,
    }
    return (
        pd.DataFrame(rows, columns=columns)
        .assign(_sort=lambda frame: frame["disposition"].map(disposition_order).fillna(99).astype(int))
        .sort_values(["_sort", "warning_count", "candidate_scope", "horizon_days", "model_name"], ascending=[True, False, True, True, True])
        .drop(columns=["_sort"])
        .reset_index(drop=True)
    )


def numeric_gap_floor(value: object, floor: float) -> float:
    value_num = row_float(pd.Series({"value": value}), "value", np.nan)
    if not math.isfinite(value_num):
        return np.nan
    return max(0.0, floor - value_num)


def numeric_gap_ceiling(value: object, ceiling: float) -> float:
    value_num = row_float(pd.Series({"value": value}), "value", np.nan)
    if not math.isfinite(value_num):
        return np.nan
    return max(0.0, value_num - ceiling)


def rank_policy_promotion_action(row: pd.Series) -> str:
    if numeric_gap_floor(row.get("brier_improvement_pct"), 0.0) > 0:
        return "improve_probability_skill_before_rank_signal_promotion"
    if numeric_gap_ceiling(row.get("decision_ece"), DECISION_ECE_THRESHOLD) > 0:
        return "calibrate_rank_supported_probability_before_promotion"
    if numeric_gap_floor(row.get("selected_minus_rule_all_pct"), 0.0) > 0:
        return "convert_rank_signal_into_positive_selected_uplift"
    if numeric_gap_floor(row.get("selected_signal_expectancy_ci_lower_pct"), MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT) > 0:
        return "increase_selected_expectancy_lower_bound_before_promotion"
    if numeric_gap_floor(row.get("positive_expectancy_folds"), MIN_POSITIVE_EXPECTANCY_FOLDS) > 0:
        return "increase_fold_repetition_of_positive_expectancy"
    if numeric_gap_floor(row.get("min_selected_events_per_fold"), MIN_SELECTED_EVENTS_PER_FOLD) > 0:
        return "collect_more_selected_events_per_fold"
    if not to_bool(row.get("decision_scope_eligible", False)):
        return "keep_rank_signal_research_only_until_decision_scope_contract_matches"
    return "candidate_close_to_promotion_recheck_full_gate"


def build_rank_policy_promotion_watchlist(candidate_disposition_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_scope",
        "horizon_days",
        "model_name",
        "warning_count",
        "oos_event_count",
        "selected_oos_event_count",
        "rank_top_quintile_minus_all_pct",
        "rank_top_quintile_se_lower_pct",
        "brier_improvement_pct",
        "brier_gap_to_zero_pct",
        "decision_ece",
        "decision_ece_gap_to_limit",
        "selected_minus_rule_all_pct",
        "selected_minus_all_gap_pct",
        "selected_signal_expectancy_ci_lower_pct",
        "selected_ci_gap_pct",
        "positive_expectancy_folds",
        "positive_expectancy_fold_gap",
        "min_selected_events_per_fold",
        "selected_events_per_fold_gap",
        "threshold_iqr",
        "threshold_iqr_gap",
        "min_calibration_bin_n",
        "calibration_bin_gap",
        "fixed_width_min_calibration_bin_n",
        "fixed_width_calibration_bin_gap",
        "decision_scope_eligible",
        "promotion_blocker_count",
        "promotion_gap_score",
        "promotion_blockers",
        "recommended_action",
    ]
    if candidate_disposition_summary.empty:
        return pd.DataFrame(columns=columns)
    watch = candidate_disposition_summary[
        candidate_disposition_summary["disposition"].astype(str).eq("rank_policy_supported_diagnostic")
    ].copy()
    if watch.empty:
        return pd.DataFrame(columns=columns)
    if "selected_signal_expectancy_ci_lower_pct" not in watch.columns:
        watch["selected_signal_expectancy_ci_lower_pct"] = np.nan
    watch["brier_gap_to_zero_pct"] = watch["brier_improvement_pct"].map(lambda value: numeric_gap_floor(value, 0.0))
    watch["decision_ece_gap_to_limit"] = watch["decision_ece"].map(lambda value: numeric_gap_ceiling(value, DECISION_ECE_THRESHOLD))
    watch["selected_minus_all_gap_pct"] = watch["selected_minus_rule_all_pct"].map(lambda value: numeric_gap_floor(value, 0.0))
    watch["selected_ci_gap_pct"] = watch["selected_signal_expectancy_ci_lower_pct"].map(
        lambda value: numeric_gap_floor(value, MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT)
    )
    watch["positive_expectancy_fold_gap"] = watch["positive_expectancy_folds"].map(lambda value: numeric_gap_floor(value, MIN_POSITIVE_EXPECTANCY_FOLDS))
    watch["selected_events_per_fold_gap"] = watch["min_selected_events_per_fold"].map(lambda value: numeric_gap_floor(value, MIN_SELECTED_EVENTS_PER_FOLD))
    watch["threshold_iqr_gap"] = watch["threshold_iqr"].map(lambda value: numeric_gap_ceiling(value, MAX_THRESHOLD_IQR))
    watch["calibration_bin_gap"] = watch["min_calibration_bin_n"].map(lambda value: numeric_gap_floor(value, MIN_CALIBRATION_BIN_N))
    watch["fixed_width_calibration_bin_gap"] = watch["fixed_width_min_calibration_bin_n"].map(lambda value: numeric_gap_floor(value, MIN_CALIBRATION_BIN_N))

    def blockers(row: pd.Series) -> str:
        out = []
        if numeric_gap_floor(row.get("brier_improvement_pct"), 0.0) > 0:
            out.append("BRIER_SKILL")
        if numeric_gap_ceiling(row.get("decision_ece"), DECISION_ECE_THRESHOLD) > 0:
            out.append("DECISION_ECE")
        if numeric_gap_floor(row.get("selected_minus_rule_all_pct"), 0.0) > 0:
            out.append("SELECTED_UPLIFT")
        if numeric_gap_floor(row.get("selected_signal_expectancy_ci_lower_pct"), MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT) > 0:
            out.append("SELECTED_CI")
        if numeric_gap_floor(row.get("positive_expectancy_folds"), MIN_POSITIVE_EXPECTANCY_FOLDS) > 0:
            out.append("POSITIVE_FOLDS")
        if numeric_gap_floor(row.get("min_selected_events_per_fold"), MIN_SELECTED_EVENTS_PER_FOLD) > 0:
            out.append("FOLD_SAMPLE")
        if numeric_gap_ceiling(row.get("threshold_iqr"), MAX_THRESHOLD_IQR) > 0:
            out.append("THRESHOLD_STABILITY")
        if numeric_gap_floor(row.get("min_calibration_bin_n"), MIN_CALIBRATION_BIN_N) > 0:
            out.append("CALIBRATION_SAMPLE")
        if not to_bool(row.get("decision_scope_eligible", False)):
            out.append("DECISION_SCOPE")
        return "|".join(out) if out else "PASS"

    watch["promotion_blockers"] = watch.apply(blockers, axis=1)
    watch["promotion_blocker_count"] = watch["promotion_blockers"].map(
        lambda value: 0 if str(value) == "PASS" else len([part for part in str(value).split("|") if part])
    )
    gap_cols = [
        "brier_gap_to_zero_pct",
        "decision_ece_gap_to_limit",
        "selected_minus_all_gap_pct",
        "selected_ci_gap_pct",
        "positive_expectancy_fold_gap",
        "selected_events_per_fold_gap",
        "threshold_iqr_gap",
        "calibration_bin_gap",
        "fixed_width_calibration_bin_gap",
    ]
    watch["promotion_gap_score"] = watch[gap_cols].apply(
        lambda row: float(pd.to_numeric(row, errors="coerce").fillna(0.0).sum()),
        axis=1,
    )
    watch["recommended_action"] = watch.apply(rank_policy_promotion_action, axis=1)
    return (
        watch[columns]
        .sort_values(
            [
                "promotion_blocker_count",
                "promotion_gap_score",
                "rank_top_quintile_se_lower_pct",
                "rank_top_quintile_minus_all_pct",
                "decision_ece_gap_to_limit",
                "brier_gap_to_zero_pct",
                "warning_count",
            ],
            ascending=[True, True, False, False, True, True, True],
        )
        .reset_index(drop=True)
    )


def build_latest_prediction_blocker_audit(
    latest_prediction: dict[str, object],
    pooled_latest: dict[str, object],
) -> pd.DataFrame:
    columns = [
        "scope",
        "check",
        "passed",
        "value",
        "threshold",
        "gap_to_pass",
        "direction",
        "block_reason",
        "recommended_action",
    ]

    def finite(value: object) -> float:
        return as_float(value)

    def row(
        scope: str,
        check: str,
        passed: bool,
        value: object,
        threshold: object,
        gap_to_pass: object,
        direction: str,
        block_reason: str,
        recommended_action: str,
    ) -> dict[str, object]:
        return {
            "scope": scope,
            "check": check,
            "passed": bool(passed),
            "value": value,
            "threshold": threshold,
            "gap_to_pass": gap_to_pass,
            "direction": direction,
            "block_reason": "PASS" if passed else block_reason,
            "recommended_action": "PASS" if passed else recommended_action,
        }

    local_use_status = latest_prediction.get("prediction_use_status", "UNKNOWN")
    local_signal_status = latest_prediction.get("prediction_signal_status", "UNKNOWN")
    local_support = is_prediction_decision_support(local_use_status)
    latest_trade_ready = to_bool(pooled_latest.get("latest_trade_ready", False))
    latest_signal_pass = to_bool(pooled_latest.get("latest_signal_pass", False))
    decision_allowed = to_bool(pooled_latest.get("decision_support_allowed", False))
    paper_allowed = to_bool(pooled_latest.get("paper_decision_support_allowed", False))
    decision_score = finite(pooled_latest.get("decision_score_20d"))
    threshold = finite(pooled_latest.get("threshold_20d"))
    stop_hit = finite(pooled_latest.get("p_stop_hit_20d"))
    expected_r = finite(pooled_latest.get("expected_r_net_20d", pooled_latest.get("expected_r_20d")))
    paper_model_pass = to_bool(pooled_latest.get("paper_model_gate_pass", False))
    paper_oof_pass = to_bool(pooled_latest.get("paper_oof_selection_evidence_pass", paper_model_pass))

    rows = [
        row(
            "local_latest",
            "local_prediction_decision_support",
            local_support,
            local_use_status,
            "DECISION_SUPPORT_ALLOWED",
            "",
            "equals",
            str(latest_prediction.get("latest_signal_block_reasons", local_signal_status)),
            "await_or_generate_latest_entry_trigger_with_model_quality_pass",
        ),
        row(
            "pooled_latest",
            "pooled_decision_support_allowed",
            decision_allowed,
            pooled_latest.get("decision_support_allowed", "UNKNOWN"),
            "True",
            "",
            "equals",
            str(pooled_latest.get("decision_block_reasons", "POOLED_LATEST_NOT_DECISION_SUPPORT")),
            "resolve_all_failed_pooled_latest_checks",
        ),
        row(
            "pooled_latest",
            "latest_signal_pass",
            latest_signal_pass,
            pooled_latest.get("latest_signal_pass", "UNKNOWN"),
            "True",
            "",
            "equals",
            str(pooled_latest.get("latest_block_reasons", "POOLED_LATEST_SIGNAL_BLOCKED")),
            "await_latest_trade_ready_signal",
        ),
        row(
            "pooled_latest",
            "latest_trade_ready",
            latest_trade_ready,
            pooled_latest.get("latest_trade_ready", "UNKNOWN"),
            "True",
            "",
            "equals",
            "LATEST_NOT_TRADE_READY",
            "await_latest_trade_ready_signal",
        ),
        row(
            "pooled_latest",
            "decision_score_at_or_above_threshold",
            bool(math.isfinite(decision_score) and math.isfinite(threshold) and decision_score >= threshold),
            decision_score,
            threshold,
            threshold - decision_score if math.isfinite(decision_score) and math.isfinite(threshold) else np.nan,
            "higher_is_better",
            "POOLED_DECISION_SCORE_BELOW_THRESHOLD",
            "wait_for_score_to_clear_locked_oof_threshold",
        ),
        row(
            "pooled_latest",
            "stop_risk_within_strict_limit",
            bool(math.isfinite(stop_hit) and stop_hit <= MAX_STOP_HIT_FOR_LATEST),
            stop_hit,
            MAX_STOP_HIT_FOR_LATEST,
            stop_hit - MAX_STOP_HIT_FOR_LATEST if math.isfinite(stop_hit) else np.nan,
            "lower_is_better",
            "POOLED_STOP_RISK_GT_0_35",
            "wait_for_stop_risk_to_fall_below_strict_limit",
        ),
        row(
            "pooled_latest",
            "expected_r_at_or_above_min",
            bool(math.isfinite(expected_r) and expected_r >= MIN_EXPECTED_R_FOR_LATEST),
            expected_r,
            MIN_EXPECTED_R_FOR_LATEST,
            MIN_EXPECTED_R_FOR_LATEST - expected_r if math.isfinite(expected_r) else np.nan,
            "higher_is_better",
            "POOLED_EXPECTED_R_LT_0_35",
            "wait_for_expected_r_to_recover_above_minimum",
        ),
        row(
            "paper_latest",
            "paper_decision_support_allowed",
            paper_allowed,
            pooled_latest.get("paper_decision_support_allowed", "UNKNOWN"),
            "True",
            "",
            "equals",
            str(pooled_latest.get("paper_gate_block_reasons", "PAPER_GATE_BLOCKED")),
            "resolve_paper_latest_market_state_blocks",
        ),
        row(
            "paper_model",
            "paper_model_gate_pass",
            paper_model_pass,
            pooled_latest.get("paper_model_gate_pass", "UNKNOWN"),
            "True",
            "",
            "equals",
            str(pooled_latest.get("paper_model_block_reasons", "PAPER_MODEL_GATE_BLOCKED")),
            "repair_paper_model_evidence_before_paper_decision_support",
        ),
        row(
            "paper_model",
            "paper_oof_selection_evidence_pass",
            paper_oof_pass,
            pooled_latest.get("paper_oof_selection_evidence_pass", "UNKNOWN"),
            "True",
            "",
            "equals",
            "POOLED_OOF_FOLD_SELECTED_EVENTS_LT_25",
            "expand_or_merge_oof_evidence_without_weakening_event_thresholds",
        ),
        row(
            "paper_latest",
            "stop_risk_within_paper_limit",
            bool(math.isfinite(stop_hit) and stop_hit <= PAPER_MAX_STOP_HIT_FOR_LATEST),
            stop_hit,
            PAPER_MAX_STOP_HIT_FOR_LATEST,
            stop_hit - PAPER_MAX_STOP_HIT_FOR_LATEST if math.isfinite(stop_hit) else np.nan,
            "lower_is_better",
            "POOLED_STOP_RISK_GT_0_40",
            "wait_for_stop_risk_to_fall_below_paper_limit",
        ),
    ]
    return pd.DataFrame(rows, columns=columns)


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
        research_validation_critical = research_validation_groups[research_validation_groups["severity"].astype(str).eq("CRITICAL")]
        research_validation_pass = bool(research_validation_critical.empty or research_validation_critical["passed"].astype(bool).all())
        pooled_system_quality_pass = bool(pooled_model_quality_pass and research_validation_pass)
        pooled_latest_signal_pass = to_bool(pooled_latest.get("latest_signal_pass", False))
        pooled_support = to_bool(pooled_latest.get("decision_support_allowed", False))
        pooled_paper_support = to_bool(pooled_latest.get("paper_decision_support_allowed", False))
        if local_support or (pooled_support and pooled_system_quality_pass):
            status = "PASS"
        elif pooled_paper_support:
            status = "PAPER_DECISION_SUPPORT_ALLOWED"
        elif pooled_system_quality_pass and not pooled_latest_signal_pass:
            status = "PASS_MODEL_QUALITY_SIGNAL_STANDBY"
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
        "PASS"
        if pooled_model_quality_pass
        else unique_join(model_quality_failures["block_reason"])
        if not model_quality_failures.empty
        else pooled_latest.get("model_quality_block_reasons", "PASS")
    )
    reported_latest_reasons = str(pooled_latest.get("latest_block_reasons", ""))
    pooled_latest_reasons = (
        reported_latest_reasons
        if reported_latest_reasons and reported_latest_reasons.lower() != "nan"
        else unique_join(latest_failures["block_reason"]) if not latest_failures.empty else "PASS"
    )
    local_failures = (
        failed[failed["gate_group"].astype(str).isin(["local_prediction", "local_prediction_diagnostic", "local_latest"])]
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
            or audit[
                audit["gate_group"].astype(str).eq("research_validation")
                & audit["severity"].astype(str).eq("CRITICAL")
            ]["passed"].astype(bool).all()
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
    if pooled_tsm_calibration_pass and str(pooled_tsm_calibration_failure_summary).upper() != "PASS":
        pooled_tsm_calibration_failure_summary = "DIAGNOSTIC_ONLY_FAILED_TSM_ROUTE_NOT_USED_FOR_SCORING"
    next_required_evidence_action = pooled_latest.get("next_required_evidence_action", "")
    if pooled_system_quality_pass and not pooled_latest_signal_pass:
        next_required_evidence_action = "await_latest_trade_ready_signal"
    if not next_required_evidence_action:
        actions = []
        if not pooled_threshold_stability_pass:
            actions.append("improve_locked_risk_adjusted_rank_policy_or_expand_fold_trade_ready_events")
        if not pooled_economic_uplift_pass:
            actions.append("increase_oof_lower_bound_evidence_with_more_semiconductor_events")
        if not pooled_tsm_calibration_pass:
            actions.append("improve_tsm_route_calibration_or_expand_tsm_like_calibration_sample")
        next_required_evidence_action = "|".join(actions) if actions else "PASS"
    critical_failed_gate_count = int((failed["severity"].astype(str).eq("CRITICAL")).sum()) if not failed.empty else 0
    warning_failed_gate_count = int((failed["severity"].astype(str).eq("WARN")).sum()) if not failed.empty else 0
    info_failed_gate_count = int((failed["severity"].astype(str).eq("INFO")).sum()) if not failed.empty else 0
    model_quality_failed = (
        failed[failed["gate_group"].astype(str).isin(["pooled_model", "pooled_tsm_calibration", "research_validation"])]
        if not failed.empty
        else pd.DataFrame()
    )
    model_quality_blocking_failed_gate_count = (
        int(model_quality_failed["severity"].astype(str).eq("CRITICAL").sum())
        if not model_quality_failed.empty
        else 0
    )
    model_quality_warning_gate_count = (
        int(model_quality_failed["severity"].astype(str).eq("WARN").sum())
        if not model_quality_failed.empty
        else 0
    )
    strategy_warning_gate_count = int(
        failed["gate_group"].astype(str).str.startswith("strategy_validation").astype(bool).sum()
    ) if not failed.empty else 0
    performance_failed = performance_failed_rows(failed)
    performance_failed = attach_rank_policy_context(performance_failed, audit) if not performance_failed.empty else performance_failed
    performance_blocking_failed_gate_count = (
        int(performance_failed["severity"].astype(str).eq("CRITICAL").sum())
        if not performance_failed.empty
        else 0
    )
    performance_warning_gate_count = (
        int(performance_failed["severity"].astype(str).eq("WARN").sum())
        if not performance_failed.empty
        else 0
    )
    active_performance_failed = (
        performance_failed[performance_failed["gate_group"].astype(str).isin(ACTIVE_PERFORMANCE_GATE_GROUPS)]
        if not performance_failed.empty
        else pd.DataFrame()
    )
    diagnostic_performance_failed = (
        performance_failed[~performance_failed["gate_group"].astype(str).isin(ACTIVE_PERFORMANCE_GATE_GROUPS)]
        if not performance_failed.empty
        else pd.DataFrame()
    )
    performance_audit = build_performance_gate_audit(audit)
    unresolved_performance_priorities = build_unresolved_performance_priorities(performance_audit)
    performance_warning_resolution_summary = build_performance_warning_resolution_summary(performance_audit)
    if not performance_failed.empty:
        performance_families = performance_failed.apply(performance_failure_family, axis=1)
        performance_metric_mask = ~performance_families.isin(PERFORMANCE_NON_METRIC_FAMILIES)
        active_performance_families = active_performance_failed.apply(performance_failure_family, axis=1) if not active_performance_failed.empty else pd.Series(dtype=object)
        diagnostic_performance_families = (
            diagnostic_performance_failed.apply(performance_failure_family, axis=1)
            if not diagnostic_performance_failed.empty
            else pd.Series(dtype=object)
        )
    else:
        performance_families = pd.Series(dtype=object)
        performance_metric_mask = pd.Series(dtype=bool)
        active_performance_families = pd.Series(dtype=object)
        diagnostic_performance_families = pd.Series(dtype=object)
    metric_performance_failed_gate_count = int(performance_metric_mask.sum()) if len(performance_metric_mask) else 0
    performance_evidence_gap_count = int(performance_families.isin(PERFORMANCE_EVIDENCE_FAMILIES).sum()) if len(performance_families) else 0
    aggregate_performance_quality_flag_count = int(performance_families.isin(PERFORMANCE_AGGREGATE_FAMILIES).sum()) if len(performance_families) else 0
    active_metric_performance_failed_gate_count = int((~active_performance_families.isin(PERFORMANCE_NON_METRIC_FAMILIES)).sum()) if len(active_performance_families) else 0
    active_performance_evidence_gap_count = int(active_performance_families.isin(PERFORMANCE_EVIDENCE_FAMILIES).sum()) if len(active_performance_families) else 0
    active_aggregate_performance_quality_flag_count = int(active_performance_families.isin(PERFORMANCE_AGGREGATE_FAMILIES).sum()) if len(active_performance_families) else 0
    diagnostic_metric_performance_warning_gate_count = (
        int((~diagnostic_performance_families.isin(PERFORMANCE_NON_METRIC_FAMILIES)).sum()) if len(diagnostic_performance_families) else 0
    )
    diagnostic_performance_evidence_gap_count = (
        int(diagnostic_performance_families.isin(PERFORMANCE_EVIDENCE_FAMILIES).sum()) if len(diagnostic_performance_families) else 0
    )
    diagnostic_aggregate_performance_quality_flag_count = (
        int(diagnostic_performance_families.isin(PERFORMANCE_AGGREGATE_FAMILIES).sum()) if len(diagnostic_performance_families) else 0
    )
    rank_policy_supported_performance_warning_count = (
        int(performance_failed.get("rank_policy_supported_candidate", pd.Series(False, index=performance_failed.index)).map(to_bool).sum())
        if not performance_failed.empty
        else 0
    )
    rank_policy_supported_threshold_warning_count = (
        int(performance_failed.get("rank_policy_supported_threshold_warning", pd.Series(False, index=performance_failed.index)).map(to_bool).sum())
        if not performance_failed.empty
        else 0
    )
    rank_policy_supported_metric_warning_count = 0
    rank_policy_supported_threshold_metric_warning_count = 0
    evidence_limited_metric_warning_count = 0
    rejected_model_metric_warning_count = 0
    strategy_diagnostic_metric_warning_count = 0
    unresolved_diagnostic_metric_performance_warning_gate_count = diagnostic_metric_performance_warning_gate_count
    if not performance_audit.empty:
        audit_families = performance_audit["performance_failure_family"].astype(str)
        audit_diagnostic_metric = (
            ~performance_audit["gate_group"].astype(str).isin(ACTIVE_PERFORMANCE_GATE_GROUPS)
            & ~audit_families.isin(PERFORMANCE_NON_METRIC_FAMILIES)
        )
        rank_supported_metric = performance_audit.get(
            "rank_policy_supported_metric_warning",
            pd.Series(False, index=performance_audit.index),
        ).map(to_bool)
        rank_supported_threshold = performance_audit.get(
            "rank_policy_supported_threshold_warning",
            pd.Series(False, index=performance_audit.index),
        ).map(to_bool)
        evidence_limited_metric = performance_audit.get(
            "evidence_limited_metric_warning",
            pd.Series(False, index=performance_audit.index),
        ).map(to_bool)
        rejected_model_metric = performance_audit.get(
            "rejected_model_metric_warning",
            pd.Series(False, index=performance_audit.index),
        ).map(to_bool)
        strategy_diagnostic_metric = performance_audit.get(
            "strategy_diagnostic_metric_warning",
            pd.Series(False, index=performance_audit.index),
        ).map(to_bool)
        diagnostic_rank_supported_metric = rank_supported_metric & audit_diagnostic_metric
        diagnostic_rank_supported_threshold_metric = rank_supported_threshold & audit_diagnostic_metric
        diagnostic_evidence_limited_metric = evidence_limited_metric & audit_diagnostic_metric
        diagnostic_rejected_model_metric = rejected_model_metric & audit_diagnostic_metric
        diagnostic_strategy_metric = strategy_diagnostic_metric & audit_diagnostic_metric
        rank_policy_supported_metric_warning_count = int(diagnostic_rank_supported_metric.sum())
        rank_policy_supported_threshold_metric_warning_count = int(diagnostic_rank_supported_threshold_metric.sum())
        evidence_limited_metric_warning_count = int(diagnostic_evidence_limited_metric.sum())
        rejected_model_metric_warning_count = int(diagnostic_rejected_model_metric.sum())
        strategy_diagnostic_metric_warning_count = int(diagnostic_strategy_metric.sum())
        resolved_diagnostic_metric = (
            diagnostic_rank_supported_metric
            | diagnostic_evidence_limited_metric
            | diagnostic_rejected_model_metric
            | diagnostic_strategy_metric
        )
        unresolved_diagnostic_metric_performance_warning_gate_count = max(
            0,
            diagnostic_metric_performance_warning_gate_count - int(resolved_diagnostic_metric.sum()),
        )
    active_performance_failed_gate_count = len(active_performance_failed)
    active_performance_blocking_failed_gate_count = (
        int(active_performance_failed["severity"].astype(str).eq("CRITICAL").sum())
        if not active_performance_failed.empty
        else 0
    )
    diagnostic_performance_warning_gate_count = (
        int(
            performance_failed[
                ~performance_failed["gate_group"].astype(str).isin(ACTIVE_PERFORMANCE_GATE_GROUPS)
                & performance_failed["severity"].astype(str).eq("WARN")
            ].shape[0]
        )
        if not performance_failed.empty
        else 0
    )
    unresolved_diagnostic_performance_warning_count = 0
    classified_diagnostic_performance_warning_count = diagnostic_performance_warning_gate_count
    performance_warning_resolution_status = (
        "PASS_NO_PERFORMANCE_WARNINGS"
        if diagnostic_performance_warning_gate_count == 0 and active_performance_failed_gate_count == 0
        else "PASS_ALL_NONACTIVE_WARNINGS_CLASSIFIED"
    )
    if not performance_warning_resolution_summary.empty:
        resolution_counts = dict(
            zip(
                performance_warning_resolution_summary["resolution_bucket"].astype(str),
                pd.to_numeric(performance_warning_resolution_summary["warning_count"], errors="coerce").fillna(0).astype(int),
            )
        )
        unresolved_diagnostic_performance_warning_count = int(resolution_counts.get("unresolved_diagnostic_warning", 0))
        classified_diagnostic_performance_warning_count = max(
            0,
            diagnostic_performance_warning_gate_count - unresolved_diagnostic_performance_warning_count,
        )
    if active_performance_failed_gate_count > 0:
        performance_warning_resolution_status = "ACTION_REQUIRED_ACTIVE_PERFORMANCE_FAILURES"
    elif unresolved_diagnostic_performance_warning_count > 0:
        performance_warning_resolution_status = "WARN_UNRESOLVED_DIAGNOSTIC_WARNINGS"
    actionable_performance_failed_gate_count = active_performance_failed_gate_count + unresolved_diagnostic_performance_warning_count
    actionable_metric_performance_failed_gate_count = (
        active_metric_performance_failed_gate_count
        + unresolved_diagnostic_metric_performance_warning_gate_count
    )
    resolved_diagnostic_performance_warning_count = classified_diagnostic_performance_warning_count
    actionable_performance_gate_status = (
        "PASS_ACTIONABLE_PERFORMANCE"
        if actionable_performance_failed_gate_count == 0
        else "ACTION_REQUIRED_PERFORMANCE"
    )
    performance_failed_groups = (
        "|".join(sorted(set(performance_failed["gate_group"].astype(str))))
        if not performance_failed.empty
        else "PASS"
    )
    active_performance_failed_groups = (
        "|".join(sorted(set(active_performance_failed["gate_group"].astype(str))))
        if not active_performance_failed.empty
        else "PASS"
    )
    metric_performance_failed_families = (
        "|".join(sorted(set(performance_families[performance_metric_mask].astype(str))))
        if metric_performance_failed_gate_count
        else "PASS"
    )
    diagnostic_performance_failed_groups = (
        "|".join(sorted(set(diagnostic_performance_failed["gate_group"].astype(str))))
        if not diagnostic_performance_failed.empty
        else "PASS"
    )
    if active_performance_blocking_failed_gate_count > 0:
        performance_gate_status = "BLOCKED_ACTIVE_MODEL_PERFORMANCE"
    elif active_performance_failed_gate_count > 0:
        performance_gate_status = "WARN_ACTIVE_MODEL_PERFORMANCE"
    elif diagnostic_performance_warning_gate_count > 0:
        performance_gate_status = "PASS_ACTIVE_MODEL_PERFORMANCE_DIAGNOSTIC_WARNINGS_ONLY"
    else:
        performance_gate_status = "PASS_ACTIVE_MODEL_PERFORMANCE"
    next_required_performance_action = (
        unique_join(active_performance_failed["block_reason"])
        if not active_performance_failed.empty
        else "PASS_ACTIVE_MODEL_PERFORMANCE"
    )
    if active_performance_failed.empty and diagnostic_performance_failed.empty:
        performance_gate_interpretation = "ACTIVE_AND_DIAGNOSTIC_PERFORMANCE_PASS"
    elif active_performance_failed.empty:
        performance_gate_interpretation = "ACTIVE_PERFORMANCE_PASS_DIAGNOSTIC_WARNINGS_NONBLOCKING"
    else:
        performance_gate_interpretation = "ACTIVE_PERFORMANCE_REQUIRES_REPAIR"
    prediction_pipeline_ready = bool(
        pooled_system_quality_pass
        and model_quality_blocking_failed_gate_count == 0
        and actionable_performance_failed_gate_count == 0
        and active_performance_blocking_failed_gate_count == 0
        and performance_gate_status.startswith("PASS")
    )
    critical_failed_groups = (
        "|".join(sorted(set(failed.loc[failed["severity"].astype(str).eq("CRITICAL"), "gate_group"].astype(str))))
        if (not failed.empty and failed["severity"].astype(str).eq("CRITICAL").any())
        else "PASS"
    )
    warning_failed_groups = (
        "|".join(sorted(set(failed.loc[failed["severity"].astype(str).eq("WARN"), "gate_group"].astype(str))))
        if (not failed.empty and failed["severity"].astype(str).eq("WARN").any())
        else "PASS"
    )
    rank_diagnostic = (
        audit[audit["gate"].astype(str).eq(RANK_UPLIFT_DIAGNOSTIC_GATE)].copy()
        if (not audit.empty and "gate" in audit.columns)
        else pd.DataFrame()
    )
    root_cause_tokens = root_cause_tokens_from_audit(audit)
    if root_cause_tokens.empty:
        uncategorized_root_cause_count = 0
        uncategorized_warning_gate_count = 0
        uncategorized_root_causes = "PASS"
    else:
        uncategorized = root_cause_tokens[
            ~root_cause_tokens["root_cause"].astype(str).isin(ROOT_CAUSE_GUIDANCE)
        ].copy()
        uncategorized_root_cause_count = int(uncategorized["root_cause"].astype(str).nunique()) if not uncategorized.empty else 0
        uncategorized_warning_gate_count = int(len(uncategorized))
        uncategorized_root_causes = unique_join(uncategorized["root_cause"], limit=12) if not uncategorized.empty else "PASS"
    if rank_diagnostic.empty:
        rank_uplift_diagnostic_count = 0
        rank_uplift_positive_count = 0
        rank_policy_diagnostic_pass_count = 0
        rank_policy_best_se_lower_pct = np.nan
    else:
        rank_values = pd.to_numeric(rank_diagnostic.get("value", pd.Series(dtype=float)), errors="coerce")
        rank_lower = pd.to_numeric(rank_diagnostic.get("rank_top_quintile_se_lower_pct", pd.Series(dtype=float)), errors="coerce")
        rank_policy_pass = rank_diagnostic.get("rank_policy_diagnostic_pass", pd.Series(False, index=rank_diagnostic.index)).map(to_bool)
        rank_uplift_diagnostic_count = int(len(rank_diagnostic))
        rank_uplift_positive_count = int((rank_values > 0.0).sum())
        rank_policy_diagnostic_pass_count = int(rank_policy_pass.sum())
        rank_policy_lower = rank_lower[rank_policy_pass]
        rank_policy_best_se_lower_pct = float(rank_policy_lower.max()) if rank_policy_lower.notna().any() else np.nan
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
        {
            "field": "selected_tsm_calibration_route_pass",
            "value": pooled_latest.get("selected_tsm_calibration_route_pass", pooled_latest.get("tsm_calibration_route_pass", "UNKNOWN")),
        },
        {"field": "effective_tsm_scoring_route", "value": pooled_latest.get("effective_tsm_scoring_route", pooled_latest.get("tsm_calibration_scoring_route", "UNKNOWN"))},
        {
            "field": "effective_tsm_scoring_route_pass",
            "value": to_bool(pooled_latest.get("effective_tsm_scoring_route_pass", pooled_tsm_calibration_pass)),
        },
        {
            "field": "effective_tsm_scoring_route_reason",
            "value": pooled_latest.get("effective_tsm_scoring_route_reason", pooled_latest.get("tsm_calibration_scoring_route_reason", "UNKNOWN")),
        },
        {"field": "pooled_latest_signal_pass", "value": pooled_latest_signal_pass},
        {"field": "prediction_pipeline_ready", "value": prediction_pipeline_ready},
        {"field": "pooled_prediction_decision_support", "value": bool(pooled_system_quality_pass and pooled_latest_signal_pass and pooled_support)},
        {"field": "paper_prediction_decision_support", "value": to_bool(pooled_latest.get("paper_decision_support_allowed", False))},
        {"field": "latest_stop_hit_raw_20d", "value": pooled_latest.get("p_stop_hit_raw_20d", pooled_latest.get("p_stop_hit_20d", np.nan))},
        {"field": "latest_stop_hit_calibrated_20d", "value": pooled_latest.get("p_stop_hit_calibrated_20d", pooled_latest.get("p_stop_hit_20d", np.nan))},
        {"field": "latest_stop_hit_oos_percentile_20d", "value": pooled_latest.get("p_stop_hit_oos_percentile_20d", np.nan)},
        {"field": "latest_stop_hit_raw_minus_calibrated_20d", "value": pooled_latest.get("p_stop_hit_raw_minus_calibrated_20d", np.nan)},
        {"field": "latest_stop_risk_calibration_warning", "value": pooled_latest.get("stop_risk_calibration_warning", "PASS")},
        {"field": "paper_gate_status", "value": pooled_latest.get("paper_gate_status", "UNKNOWN")},
        {"field": "paper_gate_block_reasons", "value": pooled_latest.get("paper_gate_block_reasons", "UNKNOWN")},
        {"field": "pooled_model_quality_block_reasons", "value": pooled_model_quality_reasons},
        {"field": "pooled_latest_block_reasons", "value": pooled_latest_reasons},
        {"field": "pooled_decision_block_reasons", "value": pooled_latest_reasons if pooled_system_quality_pass and not pooled_latest_signal_pass else pooled_latest.get("decision_block_reasons", "UNKNOWN")},
        {"field": "pooled_threshold_failure_summary", "value": pooled_threshold_failure_summary},
        {"field": "pooled_uplift_failure_summary", "value": pooled_uplift_failure_summary},
        {"field": "pooled_tsm_calibration_failure_summary", "value": pooled_tsm_calibration_failure_summary},
        {"field": "next_required_evidence_action", "value": next_required_evidence_action},
        {"field": "failed_gate_count", "value": len(failed)},
        {"field": "blocking_failed_gate_count", "value": critical_failed_gate_count},
        {"field": "critical_failed_gate_count", "value": critical_failed_gate_count},
        {"field": "warning_failed_gate_count", "value": warning_failed_gate_count},
        {"field": "diagnostic_warning_gate_count", "value": warning_failed_gate_count},
        {"field": "info_failed_gate_count", "value": info_failed_gate_count},
        {"field": "model_quality_failed_gate_count", "value": model_quality_blocking_failed_gate_count},
        {"field": "model_quality_blocking_failed_gate_count", "value": model_quality_blocking_failed_gate_count},
        {"field": "model_quality_warning_gate_count", "value": model_quality_warning_gate_count},
        {"field": "strategy_warning_gate_count", "value": strategy_warning_gate_count},
        {"field": "performance_gate_status", "value": performance_gate_status},
        {"field": "actionable_performance_gate_status", "value": actionable_performance_gate_status},
        {"field": "actionable_performance_failed_gate_count", "value": actionable_performance_failed_gate_count},
        {"field": "actionable_metric_performance_failed_gate_count", "value": actionable_metric_performance_failed_gate_count},
        {"field": "performance_failed_gate_count", "value": len(performance_failed)},
        {"field": "performance_blocking_failed_gate_count", "value": performance_blocking_failed_gate_count},
        {"field": "performance_warning_gate_count", "value": performance_warning_gate_count},
        {"field": "metric_performance_failed_gate_count", "value": metric_performance_failed_gate_count},
        {"field": "performance_evidence_gap_count", "value": performance_evidence_gap_count},
        {"field": "aggregate_performance_quality_flag_count", "value": aggregate_performance_quality_flag_count},
        {"field": "active_performance_failed_gate_count", "value": active_performance_failed_gate_count},
        {"field": "active_performance_blocking_failed_gate_count", "value": active_performance_blocking_failed_gate_count},
        {"field": "active_metric_performance_failed_gate_count", "value": active_metric_performance_failed_gate_count},
        {"field": "active_performance_evidence_gap_count", "value": active_performance_evidence_gap_count},
        {"field": "active_aggregate_performance_quality_flag_count", "value": active_aggregate_performance_quality_flag_count},
        {"field": "diagnostic_performance_warning_gate_count", "value": diagnostic_performance_warning_gate_count},
        {"field": "diagnostic_metric_performance_warning_gate_count", "value": diagnostic_metric_performance_warning_gate_count},
        {"field": "diagnostic_performance_evidence_gap_count", "value": diagnostic_performance_evidence_gap_count},
        {"field": "diagnostic_aggregate_performance_quality_flag_count", "value": diagnostic_aggregate_performance_quality_flag_count},
        {"field": "rank_policy_supported_performance_warning_count", "value": rank_policy_supported_performance_warning_count},
        {"field": "rank_policy_supported_threshold_warning_count", "value": rank_policy_supported_threshold_warning_count},
        {"field": "rank_policy_supported_metric_warning_count", "value": rank_policy_supported_metric_warning_count},
        {"field": "rank_policy_supported_threshold_metric_warning_count", "value": rank_policy_supported_threshold_metric_warning_count},
        {"field": "evidence_limited_metric_warning_count", "value": evidence_limited_metric_warning_count},
        {"field": "rejected_model_metric_warning_count", "value": rejected_model_metric_warning_count},
        {"field": "strategy_diagnostic_metric_warning_count", "value": strategy_diagnostic_metric_warning_count},
        {"field": "unresolved_diagnostic_metric_performance_warning_gate_count", "value": unresolved_diagnostic_metric_performance_warning_gate_count},
        {"field": "resolved_diagnostic_performance_warning_count", "value": resolved_diagnostic_performance_warning_count},
        {"field": "classified_diagnostic_performance_warning_count", "value": classified_diagnostic_performance_warning_count},
        {"field": "unresolved_diagnostic_performance_warning_count", "value": unresolved_diagnostic_performance_warning_count},
        {"field": "performance_warning_resolution_status", "value": performance_warning_resolution_status},
        {"field": "uncategorized_root_cause_count", "value": uncategorized_root_cause_count},
        {"field": "uncategorized_warning_gate_count", "value": uncategorized_warning_gate_count},
        {"field": "uncategorized_root_causes", "value": uncategorized_root_causes},
        {"field": "performance_failed_gate_groups", "value": performance_failed_groups},
        {"field": "active_performance_failed_gate_groups", "value": active_performance_failed_groups},
        {"field": "metric_performance_failed_families", "value": metric_performance_failed_families},
        {"field": "diagnostic_performance_failed_gate_groups", "value": diagnostic_performance_failed_groups},
        {"field": "next_required_performance_action", "value": next_required_performance_action},
        {"field": "performance_gate_interpretation", "value": performance_gate_interpretation},
        {"field": "rank_uplift_diagnostic_count", "value": rank_uplift_diagnostic_count},
        {"field": "rank_uplift_positive_diagnostic_count", "value": rank_uplift_positive_count},
        {"field": "rank_policy_diagnostic_pass_count", "value": rank_policy_diagnostic_pass_count},
        {"field": "rank_policy_best_se_lower_pct", "value": rank_policy_best_se_lower_pct},
        {"field": "failed_gate_groups", "value": "|".join(sorted(set(failed["gate_group"].astype(str)))) if not failed.empty else "PASS"},
        {"field": "blocking_failed_gate_groups", "value": critical_failed_groups},
        {"field": "critical_failed_gate_groups", "value": critical_failed_groups},
        {"field": "warning_failed_gate_groups", "value": warning_failed_groups},
        {"field": "generated_at_utc", "value": now_utc_iso()},
    ]
    return pd.DataFrame(rows)


def write_report(
    outdir: Path,
    audit: pd.DataFrame,
    snapshot: pd.DataFrame,
    root_causes: pd.DataFrame,
    prediction_performance_gap_summary: pd.DataFrame | None = None,
    candidate_disposition_summary: pd.DataFrame | None = None,
) -> None:
    snap = dict(zip(snapshot["field"], snapshot["value"]))
    prediction_performance_gap_summary = (
        prediction_performance_gap_summary if prediction_performance_gap_summary is not None else pd.DataFrame()
    )
    candidate_disposition_summary = candidate_disposition_summary if candidate_disposition_summary is not None else pd.DataFrame()
    rank_policy_promotion_watchlist = build_rank_policy_promotion_watchlist(candidate_disposition_summary)
    failed = audit[~audit["passed"].astype(bool)] if not audit.empty else pd.DataFrame()
    blocking_failed = failed[failed["severity"].astype(str).eq("CRITICAL")] if not failed.empty else pd.DataFrame()
    warnings = failed[failed["severity"].astype(str).eq("WARN")] if not failed.empty else pd.DataFrame()
    performance_failed = performance_failed_rows(failed)
    active_performance_failed = (
        performance_failed[performance_failed["gate_group"].astype(str).isin(ACTIVE_PERFORMANCE_GATE_GROUPS)]
        if not performance_failed.empty
        else pd.DataFrame()
    )
    diagnostic_performance_failed = (
        performance_failed[~performance_failed["gate_group"].astype(str).isin(ACTIVE_PERFORMANCE_GATE_GROUPS)]
        if not performance_failed.empty
        else pd.DataFrame()
    )
    performance_audit = build_performance_gate_audit(audit)
    unresolved_performance_priorities = build_unresolved_performance_priorities(performance_audit)
    performance_warning_resolution_summary = build_performance_warning_resolution_summary(performance_audit)

    def append_gate_rows(lines: list[str], rows: pd.DataFrame) -> None:
        lines.extend(
            [
                "| Group | Severity | Scope | Horizon | Model | Split | Gate | Value | Threshold | Block Reason |",
                "|---|---|---|---:|---|---|---|---:|---:|---|",
            ]
        )
        if rows.empty:
            lines.append("| PASS | NA | NA | NA | NA | NA | NA | NA | NA | PASS |")
            return
        for _, row in rows.iterrows():
            lines.append(
                f"| {row['gate_group']} | {row['severity']} | {row['candidate_scope']} | {row['horizon_days']} | {row['model_name']} | {row['split']} | "
                f"{row['gate']} | {row['value']} | {row['threshold']} | {row['block_reason']} |"
            )

    def append_rank_uplift_diagnostics(lines: list[str]) -> None:
        lines.extend(["", "## Ranking Uplift Diagnostics", ""])
        if audit.empty or "gate" not in audit.columns:
            lines.append("- No rank-top-20% diagnostic rows available.")
            return
        rank_rows = audit[audit["gate"].astype(str).eq(RANK_UPLIFT_DIAGNOSTIC_GATE)].copy()
        if rank_rows.empty:
            lines.append("- No rank-top-20% diagnostic rows available.")
            return
        rank_rows["rank_uplift_value"] = pd.to_numeric(rank_rows.get("value", pd.Series(dtype=float)), errors="coerce")
        rank_rows["rank_top_count_value"] = pd.to_numeric(rank_rows.get("rank_top_quintile_count", pd.Series(dtype=float)), errors="coerce")
        rank_rows["rank_fold_lower_value"] = pd.to_numeric(rank_rows.get("rank_top_quintile_se_lower_pct", pd.Series(dtype=float)), errors="coerce")
        rank_rows["rank_positive_folds_value"] = pd.to_numeric(rank_rows.get("rank_top_quintile_positive_folds", pd.Series(dtype=float)), errors="coerce")
        rank_rows["rank_policy_pass_value"] = rank_rows.get("rank_policy_diagnostic_pass", pd.Series(False, index=rank_rows.index)).map(to_bool)
        available = rank_rows["rank_uplift_value"].notna() & (rank_rows["rank_top_count_value"].fillna(0) > 0)
        positive = available & (rank_rows["rank_uplift_value"] > 0)
        nonpositive = available & (rank_rows["rank_uplift_value"] <= 0)
        unavailable = ~available
        robust = available & rank_rows["rank_policy_pass_value"]
        lines.extend(
            [
                f"- Diagnostic rows: {len(rank_rows)}",
                f"- Positive top-20% minus all: {int(positive.sum())}",
                f"- Robust rank-policy pass rows: {int(robust.sum())}",
                f"- Nonpositive top-20% minus all: {int(nonpositive.sum())}",
                f"- Unavailable rank diagnostics: {int(unavailable.sum())}",
                "",
                "| Scope | Horizon | Model | Top 20% Count | Top 20% Minus All | Fold Lower | Positive Folds | Rank Policy | Status |",
                "|---|---:|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        display = rank_rows[available].sort_values(
            ["rank_policy_pass_value", "rank_fold_lower_value", "rank_uplift_value", "rank_top_count_value", "candidate_scope", "horizon_days", "model_name"],
            ascending=[False, False, False, False, True, True, True],
        )
        if display.empty:
            lines.append("| NA | NA | NA | NA | NA | NA | NA | False | RANK_TOP_QUINTILE_UPLIFT_UNAVAILABLE |")
            return
        for _, row in display.head(15).iterrows():
            fold_lower = row["rank_fold_lower_value"]
            fold_lower_text = f"{fold_lower:.2f}%" if pd.notna(fold_lower) else "NA"
            positive_folds = row["rank_positive_folds_value"]
            positive_folds_text = str(int(positive_folds)) if pd.notna(positive_folds) else "NA"
            lines.append(
                f"| {row['candidate_scope']} | {row['horizon_days']} | {row['model_name']} | "
                f"{int(row['rank_top_count_value'])} | {row['rank_uplift_value']:.2f}% | {fold_lower_text} | "
                f"{positive_folds_text} | {bool(row['rank_policy_pass_value'])} | {row['block_reason']} |"
            )

    lines = [
        "# Top10 Model Gate Audit Report",
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
        f"- Blocking failed gate count: {snap.get('blocking_failed_gate_count', snap.get('critical_failed_gate_count', 'NA'))}",
        f"- Warning failed gate count: {snap.get('warning_failed_gate_count', 'NA')}",
        f"- Model-quality warning gate count: {snap.get('model_quality_warning_gate_count', 'NA')}",
        f"- Strategy warning gate count: {snap.get('strategy_warning_gate_count', 'NA')}",
        f"- Performance gate status: {snap.get('performance_gate_status', 'NA')}",
        f"- Actionable performance gate status: {snap.get('actionable_performance_gate_status', 'NA')}",
        f"- Actionable performance failed gates: {snap.get('actionable_performance_failed_gate_count', 'NA')}",
        f"- Actionable metric performance failed gates: {snap.get('actionable_metric_performance_failed_gate_count', 'NA')}",
        f"- Raw performance warning/failure rows excluding latest/no-signal: {snap.get('performance_failed_gate_count', 'NA')}",
        f"- Raw metric performance warning/failure rows: {snap.get('metric_performance_failed_gate_count', 'NA')}",
        f"- Performance evidence gap count: {snap.get('performance_evidence_gap_count', 'NA')}",
        f"- Aggregate performance quality flags: {snap.get('aggregate_performance_quality_flag_count', 'NA')}",
        f"- Active performance failed gates: {snap.get('active_performance_failed_gate_count', 'NA')}",
        f"- Active performance blocking gates: {snap.get('active_performance_blocking_failed_gate_count', 'NA')}",
        f"- Active metric performance failed gates: {snap.get('active_metric_performance_failed_gate_count', 'NA')}",
        f"- Diagnostic performance warning gates: {snap.get('diagnostic_performance_warning_gate_count', 'NA')}",
        f"- Diagnostic metric performance warnings: {snap.get('diagnostic_metric_performance_warning_gate_count', 'NA')}",
        f"- Rank-policy supported performance warnings: {snap.get('rank_policy_supported_performance_warning_count', 'NA')}",
        f"- Rank-policy supported threshold warnings: {snap.get('rank_policy_supported_threshold_warning_count', 'NA')}",
        f"- Rank-policy supported metric warnings: {snap.get('rank_policy_supported_metric_warning_count', 'NA')}",
        f"- Rank-policy supported threshold metric warnings: {snap.get('rank_policy_supported_threshold_metric_warning_count', 'NA')}",
        f"- Evidence-limited metric warnings: {snap.get('evidence_limited_metric_warning_count', 'NA')}",
        f"- Rejected-model metric warnings: {snap.get('rejected_model_metric_warning_count', 'NA')}",
        f"- Strategy diagnostic metric warnings: {snap.get('strategy_diagnostic_metric_warning_count', 'NA')}",
        f"- Unresolved diagnostic metric performance warnings: {snap.get('unresolved_diagnostic_metric_performance_warning_gate_count', 'NA')}",
        f"- Resolved diagnostic performance warnings: {snap.get('resolved_diagnostic_performance_warning_count', 'NA')}",
        f"- Classified diagnostic performance warnings: {snap.get('classified_diagnostic_performance_warning_count', 'NA')}",
        f"- Unresolved diagnostic performance warnings: {snap.get('unresolved_diagnostic_performance_warning_count', 'NA')}",
        f"- Performance warning resolution status: {snap.get('performance_warning_resolution_status', 'NA')}",
        f"- Next-day directional display pass candidates: {snap.get('next_day_directional_display_pass_count', 'NA')}",
        f"- Next-day rejected challengers: {snap.get('next_day_rejected_challenger_count', 'NA')}",
        f"- Next-day actionable performance failures: {snap.get('next_day_actionable_performance_failed_count', 'NA')}",
        f"- Uncategorized root causes: {snap.get('uncategorized_root_cause_count', 'NA')}",
        f"- Uncategorized warning gate rows: {snap.get('uncategorized_warning_gate_count', 'NA')}",
        f"- Uncategorized root cause names: {snap.get('uncategorized_root_causes', 'NA')}",
        f"- Next required performance action: {snap.get('next_required_performance_action', 'NA')}",
        f"- Performance gate interpretation: {snap.get('performance_gate_interpretation', 'NA')}",
        f"- Blocking failed gate groups: {snap.get('blocking_failed_gate_groups', snap.get('critical_failed_gate_groups', 'NA'))}",
        f"- Warning failed gate groups: {snap.get('warning_failed_gate_groups', 'NA')}",
        "",
        "## Blocking Failed Gates",
        "",
    ]
    append_gate_rows(lines, blocking_failed)
    lines.extend(["", "## Performance-Only Gate Summary", ""])
    if performance_failed.empty:
        lines.append("- No failed performance gates after excluding latest/no-signal and trade-readiness-scope blocks.")
    else:
        if active_performance_failed.empty:
            lines.append("- Active performance failures: PASS")
        else:
            lines.extend(["- Active performance failures: ACTION_REQUIRED", "", "### Active Performance Failures", ""])
            active_summary = (
                active_performance_failed.groupby(["gate_group", "severity", "gate", "block_reason"], dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values(["severity", "count", "gate_group", "gate"], ascending=[True, False, True, True])
            )
            lines.extend(
                [
                    "| Group | Severity | Gate | Count | Block Reason |",
                    "|---|---|---|---:|---|",
                ]
            )
            for _, row in active_summary.head(40).iterrows():
                lines.append(
                    f"| {row['gate_group']} | {row['severity']} | {row['gate']} | {int(row['count'])} | {row['block_reason']} |"
                )
        performance_family_summary = (
            performance_failed.assign(performance_failure_family=performance_failed.apply(performance_failure_family, axis=1))
            .groupby(["performance_failure_family"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "performance_failure_family"], ascending=[False, True])
        )
        lines.extend(["", "### Performance Failure Families", ""])
        lines.extend(["| Family | Count |", "|---|---:|"])
        for _, row in performance_family_summary.iterrows():
            lines.append(f"| {row['performance_failure_family']} | {int(row['count'])} |")
        lines.extend(["", "### Performance Warning Resolution Summary", ""])
        if performance_warning_resolution_summary.empty:
            lines.append("- No performance warning resolution summary available.")
        else:
            lines.extend(
                [
                    "| Resolution Bucket | Family | Kind | Count | Metric | Evidence Gap | Aggregate | Rank Supported | Evidence Limited | Rejected | Strategy |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for _, row in performance_warning_resolution_summary.head(30).iterrows():
                lines.append(
                    f"| {row['resolution_bucket']} | {row['performance_failure_family']} | {row['performance_failure_kind']} | "
                    f"{int(row['warning_count'])} | {int(row['metric_warning_count'])} | {int(row['evidence_gap_count'])} | "
                    f"{int(row['aggregate_flag_count'])} | {int(row['rank_policy_supported_count'])} | "
                    f"{int(row['evidence_limited_metric_count'])} | {int(row['rejected_model_metric_count'])} | "
                    f"{int(row['strategy_diagnostic_metric_count'])} |"
                )
        lines.extend(["", "### Prediction Performance Gap Summary", ""])
        if prediction_performance_gap_summary.empty:
            lines.append("- No prediction performance gap summary available.")
        else:
            display = prediction_performance_gap_summary.head(12)
            lines.extend(
                [
                    "| Reason | Candidates | One-Block Candidates | Recommended Action | Top Candidate | Median Brier Gap | Max ECE Gap | Max Selected CI Gap |",
                    "|---|---:|---:|---|---|---:|---:|---:|",
                ]
            )
            for _, row in display.iterrows():
                top_horizon = row.get("top_candidate_horizon_days", np.nan)
                top_horizon_text = f"{int(top_horizon)}d" if pd.notna(top_horizon) else "NA"
                top_candidate = f"{row.get('top_candidate_scope', 'NA')}/{top_horizon_text}/{row.get('top_candidate_model_name', 'NA')}"
                lines.append(
                    f"| {row['performance_block_reason']} | {int(row['candidate_count'])} | "
                    f"{int(row['one_block_candidate_count'])} | {row['recommended_action']} | {top_candidate} | "
                    f"{row.get('median_brier_gap_pct', np.nan):.2f}% | "
                    f"{row.get('max_ece_gap', np.nan):.4f} | "
                    f"{row.get('max_selected_ci_gap_pct', np.nan):.2f}% |"
                )
        lines.extend(["", "### Candidate Disposition Summary", ""])
        if candidate_disposition_summary.empty:
            lines.append("- No candidate disposition summary available.")
        else:
            counts = (
                candidate_disposition_summary.groupby(["disposition", "recommended_action"], dropna=False)["warning_count"]
                .agg(candidate_count="size", warning_count="sum")
                .reset_index()
                .sort_values(["warning_count", "candidate_count", "disposition"], ascending=[False, False, True])
            )
            lines.extend(["| Disposition | Candidates | Warnings | Recommended Action |", "|---|---:|---:|---|"])
            for _, row in counts.iterrows():
                lines.append(
                    f"| {row['disposition']} | {int(row['candidate_count'])} | {int(row['warning_count'])} | {row['recommended_action']} |"
                )
            display = candidate_disposition_summary[
                ~candidate_disposition_summary["disposition"].astype(str).eq("diagnostic_pass")
            ].head(15)
            if not display.empty:
                lines.extend(["", "#### Top Candidate Dispositions", ""])
                lines.extend(
                    [
                        "| Disposition | Scope | Horizon | Model | Warnings | Metric | Evidence | Aggregate | Action |",
                        "|---|---|---:|---|---:|---:|---:|---:|---|",
                    ]
                )
                for _, row in display.iterrows():
                    lines.append(
                        f"| {row['disposition']} | {row['candidate_scope']} | {row['horizon_days']} | {row['model_name']} | "
                        f"{int(row['warning_count'])} | {int(row['metric_warning_count'])} | {int(row['evidence_gap_count'])} | "
                        f"{int(row['aggregate_flag_count'])} | {row['recommended_action']} |"
                    )
        lines.extend(["", "### Rank-Policy Promotion Watchlist", ""])
        if rank_policy_promotion_watchlist.empty:
            lines.append("- No rank-policy supported diagnostic candidates require promotion tracking.")
        else:
            lines.extend(
                [
                    "| Scope | Horizon | Model | Rank Lower | Rank Uplift | ECE Gap | Brier Gap | Uplift Gap | Fold Gap | Blockers | Action |",
                    "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
                ]
            )
            for _, row in rank_policy_promotion_watchlist.head(15).iterrows():
                lines.append(
                    f"| {row['candidate_scope']} | {row['horizon_days']} | {row['model_name']} | "
                    f"{row.get('rank_top_quintile_se_lower_pct', np.nan):.2f}% | "
                    f"{row.get('rank_top_quintile_minus_all_pct', np.nan):.2f}% | "
                    f"{row.get('decision_ece_gap_to_limit', np.nan):.4f} | "
                    f"{row.get('brier_gap_to_zero_pct', np.nan):.2f}% | "
                    f"{row.get('selected_minus_all_gap_pct', np.nan):.2f}% | "
                    f"{row.get('positive_expectancy_fold_gap', np.nan):.0f} | "
                    f"{row.get('promotion_blockers', 'NA')} | {row.get('recommended_action', 'NA')} |"
                )
        lines.extend(["", "### Nonblocking Diagnostic Performance Warnings", ""])
        performance_summary = (
            diagnostic_performance_failed.groupby(["gate_group", "severity", "gate", "block_reason"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["severity", "count", "gate_group", "gate"], ascending=[True, False, True, True])
        )
        if performance_summary.empty:
            lines.append("- No nonblocking diagnostic performance warnings.")
        else:
            lines.extend(
                [
                    "| Group | Severity | Gate | Count | Block Reason |",
                    "|---|---|---|---:|---|",
                ]
            )
            for _, row in performance_summary.head(40).iterrows():
                lines.append(
                    f"| {row['gate_group']} | {row['severity']} | {row['gate']} | {int(row['count'])} | {row['block_reason']} |"
                )
    lines.extend(["", "### Unresolved Diagnostic Performance Priorities", ""])
    if unresolved_performance_priorities.empty:
        lines.append("- No unresolved diagnostic metric performance warnings after rank-policy, evidence-limited, rejected-model, and strategy diagnostic context.")
    else:
        family_priority_summary = (
            unresolved_performance_priorities.groupby(
                ["priority", "performance_failure_family", "recommended_action"],
                dropna=False,
            )["warning_count"]
            .sum()
            .reset_index()
            .sort_values(["priority", "warning_count", "performance_failure_family"], ascending=[True, False, True])
        )
        lines.extend(
            [
                "| Priority | Family | Count | Recommended Action |",
                "|---:|---|---:|---|",
            ]
        )
        for _, row in family_priority_summary.iterrows():
            lines.append(
                f"| {row['priority']} | {row['performance_failure_family']} | {int(row['warning_count'])} | {row['recommended_action']} |"
            )
        lines.extend(["", "#### Top Unresolved Rows", ""])
        lines.extend(
            [
                "| Priority | Family | Scope | Horizon | Model | Gate | Count | Rank Supported | Recommended Action |",
                "|---:|---|---|---:|---|---|---:|---:|---|",
            ]
        )
        for _, row in unresolved_performance_priorities.head(25).iterrows():
            lines.append(
                f"| {row['priority']} | {row['performance_failure_family']} | {row['candidate_scope']} | {row['horizon_days']} | "
                f"{row['model_name']} | {row['gate']} | {int(row['warning_count'])} | "
                f"{int(row['rank_policy_supported_count'])} | {row['recommended_action']} |"
            )
    append_rank_uplift_diagnostics(lines)
    lines.extend(["", "## Diagnostic Warning Summary", ""])
    if warnings.empty:
        lines.append("- No diagnostic warnings.")
    else:
        warning_summary = (
            warnings.groupby(["gate_group", "gate", "block_reason"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "gate_group", "gate"], ascending=[False, True, True])
        )
        lines.extend(
            [
                f"- Diagnostic warning rows: {len(warnings)}",
                "- Full warning detail is preserved in `tsm_model_gate_audit.csv`; this report summarizes repeated diagnostic rows.",
                "",
                "| Group | Gate | Count | Block Reason |",
                "|---|---|---:|---|",
            ]
        )
        for _, row in warning_summary.head(40).iterrows():
            lines.append(
                f"| {row['gate_group']} | {row['gate']} | {int(row['count'])} | {row['block_reason']} |"
            )
        if len(warning_summary) > 40:
            lines.append(f"| ... | ... | {len(warning_summary) - 40} more warning groups | See audit CSV |")
    lines.extend(["", "## Root Cause Summary", ""])
    if root_causes.empty:
        lines.append("- No root-cause summary available.")
    else:
        lines.extend(
            [
                "| Priority | Root Cause | Failed Gates | Critical | Warn | Category | Recommended Action |",
                "|---:|---|---:|---:|---:|---|---|",
            ]
        )
        for _, row in root_causes.iterrows():
            lines.append(
                f"| {row['priority']} | {row['root_cause']} | {row['failed_gate_count']} | "
                f"{row.get('critical_failed_gate_count', 0)} | {row.get('warn_failed_gate_count', 0)} | "
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
    parser.add_argument("--prediction-performance-gap-summary", default="tsm_price_rule_output/tsm_prediction_performance_gap_summary.csv")
    parser.add_argument("--next-day-comparison", default="tsm_price_rule_output/tsm_next_day_up_model_comparison.csv")
    parser.add_argument("--next-day-oos-predictions", default="tsm_price_rule_output/tsm_next_day_up_oos_predictions.csv")
    parser.add_argument("--next-close-comparison", default="tsm_price_rule_output/tsm_next_close_model_comparison.csv")
    parser.add_argument("--next-close-quality", default="tsm_price_rule_output/tsm_next_close_quality_checks.csv")
    parser.add_argument("--next-close-latest", default="tsm_price_rule_output/tsm_next_close_latest_snapshot.csv")
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
    prediction_performance_gap_summary = read_csv(Path(args.prediction_performance_gap_summary))
    next_day_comparison = read_csv(Path(args.next_day_comparison))
    next_day_oos_predictions = read_csv(Path(args.next_day_oos_predictions))
    next_close_comparison = read_csv(Path(args.next_close_comparison))
    next_close_quality = read_csv(Path(args.next_close_quality))
    next_close_latest = read_snapshot(Path(args.next_close_latest))
    for key, value in paper_gate.items():
        pooled_latest[key] = value

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
        next_close_comparison,
        next_close_quality,
        next_close_latest,
    )
    snapshot = build_snapshot(audit, latest_prediction, pooled_latest)
    root_causes = build_root_causes(audit)
    performance_audit = build_performance_gate_audit(audit)
    unresolved_performance_priorities = build_unresolved_performance_priorities(performance_audit)
    performance_warning_resolution_summary = build_performance_warning_resolution_summary(performance_audit)
    candidate_disposition_summary = build_model_candidate_disposition_summary(audit, performance_audit, comparison, next_day_comparison, next_close_comparison)
    rank_policy_promotion_watchlist = build_rank_policy_promotion_watchlist(candidate_disposition_summary)
    latest_blocker_audit = build_latest_prediction_blocker_audit(latest_prediction, pooled_latest)
    benchmark_criteria_audit = build_benchmark_criteria_audit(
        comparison,
        pooled_comparison,
        next_day_comparison,
        next_day_oos_predictions,
        next_close_comparison,
    )
    benchmark_failed = benchmark_criteria_audit[~benchmark_criteria_audit["passed"].map(to_bool)] if not benchmark_criteria_audit.empty else pd.DataFrame()
    next_day_dispositions = (
        candidate_disposition_summary[candidate_disposition_summary["candidate_scope"].astype(str).eq("next_day_up_all")]
        if not candidate_disposition_summary.empty
        else pd.DataFrame()
    )
    if not next_day_dispositions.empty:
        next_day_display_pass_count = int(next_day_dispositions["disposition"].astype(str).eq("next_day_directional_display_pass").sum())
        next_day_rejected_challenger_count = int(next_day_dispositions["disposition"].astype(str).eq("next_day_rejected_challenger").sum())
        next_day_actionable_performance_failed_count = int(
            (~next_day_dispositions["disposition"].astype(str).isin({"next_day_directional_display_pass", "next_day_rejected_challenger"})).sum()
        )
    else:
        next_day_display_pass_count = 0
        next_day_rejected_challenger_count = 0
        next_day_actionable_performance_failed_count = 0
    next_close_dispositions = (
        candidate_disposition_summary[candidate_disposition_summary["model_role"].astype(str).eq("next_close_forecast_candidate")]
        if not candidate_disposition_summary.empty and "model_role" in candidate_disposition_summary.columns
        else pd.DataFrame()
    )
    if not next_close_dispositions.empty:
        next_close_promoted_count = int(next_close_dispositions["disposition"].astype(str).eq("next_close_forecast_promoted_diagnostic").sum())
        next_close_display_only_count = int(next_close_dispositions["disposition"].astype(str).eq("next_close_forecast_display_only").sum())
    else:
        next_close_promoted_count = 0
        next_close_display_only_count = 0
    snapshot = pd.concat(
        [
            snapshot,
            pd.DataFrame(
                [
                    {"field": "next_day_directional_display_pass_count", "value": next_day_display_pass_count},
                    {"field": "next_day_rejected_challenger_count", "value": next_day_rejected_challenger_count},
                    {"field": "next_day_actionable_performance_failed_count", "value": next_day_actionable_performance_failed_count},
                    {"field": "next_close_forecast_promoted_diagnostic_count", "value": next_close_promoted_count},
                    {"field": "next_close_forecast_display_only_count", "value": next_close_display_only_count},
                    {"field": "benchmark_criteria_status", "value": "PASS" if benchmark_failed.empty and not benchmark_criteria_audit.empty else "BLOCKED"},
                    {"field": "benchmark_criteria_failed_count", "value": int(len(benchmark_failed))},
                    {"field": "benchmark_criteria_total_count", "value": int(len(benchmark_criteria_audit))},
                ]
            ),
        ],
        ignore_index=True,
    )
    audit.to_csv(outdir / "tsm_model_gate_audit.csv", index=False)
    snapshot.to_csv(outdir / "tsm_model_gate_snapshot.csv", index=False)
    root_causes.to_csv(outdir / "tsm_model_gate_root_causes.csv", index=False)
    performance_audit.to_csv(outdir / "tsm_model_performance_gate_audit.csv", index=False)
    unresolved_performance_priorities.to_csv(outdir / "tsm_model_unresolved_performance_priorities.csv", index=False)
    performance_warning_resolution_summary.to_csv(outdir / "tsm_model_performance_warning_resolution_summary.csv", index=False)
    candidate_disposition_summary.to_csv(outdir / "tsm_model_candidate_disposition_summary.csv", index=False)
    rank_policy_promotion_watchlist.to_csv(outdir / "tsm_model_rank_policy_promotion_watchlist.csv", index=False)
    latest_blocker_audit.to_csv(outdir / "tsm_latest_prediction_blocker_audit.csv", index=False)
    benchmark_criteria_audit.to_csv(outdir / "tsm_benchmark_criteria_audit.csv", index=False)
    write_report(outdir, audit, snapshot, root_causes, prediction_performance_gap_summary, candidate_disposition_summary)
    print("completed: model gate outputs =", outdir.resolve())
    print(snapshot.to_string(index=False))


if __name__ == "__main__":
    main()

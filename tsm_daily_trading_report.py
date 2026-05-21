#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the latest TSM trading plan from rule-engine outputs.

Inputs:
- tsm_price_rule_output/tsm_latest_decision_snapshot.csv
- tsm_price_rule_output/tsm_daily_algorithmic_signals.csv
- optional risk/backtest/validation outputs

Outputs:
- tsm_daily_trading_plan.csv
- tsm_daily_trading_plan.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def to_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def as_float(value, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def usd(value: float) -> str:
    return "NA" if pd.isna(value) else f"${value:,.2f}"


def pct(value: float, digits: int = 2) -> str:
    return "NA" if pd.isna(value) else f"{value * 100:.{digits}f}%"


def pct_from_percent_value(value: float, digits: int = 2) -> str:
    return "NA" if pd.isna(value) else f"{value:.{digits}f}%"


def pct_string_value(value, digits: int = 2) -> str:
    parsed = as_float(value)
    return "NA" if pd.isna(parsed) else f"{parsed:.{digits}f}%"


def load_snapshot(path: Path) -> Dict[str, str]:
    snapshot = pd.read_csv(path)
    require_columns(snapshot, ["field", "value"], str(path))
    return dict(zip(snapshot["field"], snapshot["value"]))


def load_optional_snapshot(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    snapshot = pd.read_csv(path)
    if "field" not in snapshot.columns or "value" not in snapshot.columns:
        return {}
    return dict(zip(snapshot["field"], snapshot["value"]))


def load_optional_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return strip_bom_columns(pd.read_csv(path, **kwargs))


def load_signals(path: Path) -> pd.DataFrame:
    df = strip_bom_columns(pd.read_csv(path, parse_dates=["date"]))
    df = df.sort_values("date").reset_index(drop=True)
    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_20",
        "sma_50",
        "sma_200",
        "atr_14",
        "atr_14_pct",
        "vol_20d_ann",
        "drawdown_from_ath",
        "dist_close_sma_50_pct",
        "score_price_algo_total",
        "entry_trigger",
        "trade_action",
        "atr_stop_2x",
        "atr_trailing_stop_3x",
        "take_profit_2R",
        "take_profit_3R",
        "position_weight_if_0_5pct_account_risk",
        "position_weight_if_1pct_account_risk",
        "algo_vol_high",
        "algo_vol_extreme",
        "algo_overextended_highvol",
        "algo_deep_downtrend_avoid",
        "algo_event_shock_day",
        "trigger_breakout_20d",
        "trigger_breakout_60d",
        "trigger_pullback_50d",
        "trigger_deep_dd_recovery",
    ]
    require_columns(df, required, str(path))
    for col in [
        "algo_vol_high",
        "algo_vol_extreme",
        "algo_overextended_highvol",
        "algo_deep_downtrend_avoid",
        "algo_event_shock_day",
        "trigger_breakout_20d",
        "trigger_breakout_60d",
        "trigger_pullback_50d",
        "trigger_deep_dd_recovery",
    ]:
        df[col] = df[col].map(to_bool)
    return df


def derive_max_weight(row: pd.Series) -> float:
    if to_bool(row.get("algo_deep_downtrend_avoid")):
        return 0.00
    if as_float(row.get("close")) < as_float(row.get("sma_200")):
        return 0.03
    if to_bool(row.get("algo_vol_extreme")):
        return 0.04
    if to_bool(row.get("algo_vol_high")):
        return 0.07
    return 0.12


def action_ko(action: str) -> str:
    mapping = {
        "ENTRY_ALLOWED": "매수 가능",
        "HOLD_OR_WAIT_TRIGGER": "보유 또는 트리거 대기",
        "WATCHLIST_PULLBACK_ONLY": "관찰: 눌림목만 대기",
        "STRICT_LIVE_ENTRY": "엄격 기준 실전 후보",
        "STRICT_TRIGGER_WAITING_FOR_SCORE_OR_RISK": "엄격 기준 보류",
        "STRICT_NO_ENTRY": "엄격 기준 신규 진입 없음",
        "LIVE_ENTRY_ALREADY_ALLOWED": "실전 신호 이미 발생",
        "PAPER_BUY_SETUP": "공격형 Paper 후보",
        "EARLY_BULLISH_WATCH": "조기 상승 관찰 강화",
        "WATCHLIST_ONLY": "관찰 전용",
        "RESEARCH_BLOCKED_RISK": "연구 신호도 위험 차단",
        "NO_RESEARCH_SIGNAL": "연구 신호 없음",
        "PAPER_TRACK_LONG_SETUP": "Paper 전용 롱 추적",
        "WATCH_ONLY_EARLY_BULLISH": "조기 관찰만",
        "WATCH_ONLY": "관찰만",
        "NO_ACTION": "행동 없음",
        "REDUCE_OR_DO_NOT_CHASE": "축소 또는 추격매수 금지",
        "AVOID_TREND_LONG": "추세 롱 회피",
        "NO_NEW_LONG_BELOW_200D": "200일선 아래 신규 롱 금지",
        "RESEARCH_ONLY_DEEP_DD": "연구 전용: 깊은 낙폭 회복",
        "NO_TRADE": "거래 없음",
    }
    return mapping.get(action, action)


def conclusion_ko(row: pd.Series) -> str:
    action = str(row.get("trade_action", "NO_TRADE"))
    score = as_float(row.get("score_price_algo_total"))
    trigger = str(row.get("entry_trigger", "NONE"))

    if action == "ENTRY_ALLOWED":
        return f"진입 조건이 충족되었습니다. 트리거는 {trigger}이며 2ATR 손절 기준으로만 접근합니다."
    if action == "WATCHLIST_PULLBACK_ONLY":
        return "추격매수 금지. 50일선 눌림목 또는 20일/60일 고점 재돌파를 기다리는 구간입니다."
    if action == "REDUCE_OR_DO_NOT_CHASE":
        return "과이격 또는 고변동 위험이 큽니다. 신규 추격매수보다 축소/대기가 우선입니다."
    if action == "AVOID_TREND_LONG":
        return "깊은 하락장 또는 고변동 하락 추세입니다. 펀더멘털 재검증 전까지 추세 롱은 회피합니다."
    if action == "NO_NEW_LONG_BELOW_200D":
        return "200일선 아래입니다. 신규 롱은 금지하고 회복 확인 전까지 대기합니다."
    if action == "RESEARCH_ONLY_DEEP_DD":
        return "깊은 낙폭 회복 신호는 현재 자동 진입 대상이 아닙니다. 연구/관찰 신호로만 기록합니다."
    if action == "HOLD_OR_WAIT_TRIGGER":
        return "점수는 높지만 명확한 진입 트리거가 없습니다. 보유자는 관리, 신규 매수자는 트리거를 대기합니다."
    research_stage = str(row.get("research_signal_stage", "NO_RESEARCH_SIGNAL"))
    if research_stage == "PAPER_BUY_SETUP":
        return "엄격한 실전 진입은 아니지만 공격형 Paper 후보입니다. 실거래가 아니라 가상 기록으로 성과를 추적합니다."
    if research_stage == "EARLY_BULLISH_WATCH":
        return "실전 매수 신호는 아니지만 추세와 가격 구조가 개선되어 조기 상승 관찰을 강화합니다."
    if research_stage == "WATCHLIST_ONLY":
        return "실전 매수 신호는 아니며 관심 목록에서 다음 트리거를 기다립니다."
    if score < 60:
        return "알고리즘 점수가 낮습니다. 신규 매수보다 대기 또는 비중 축소가 우선입니다."
    return "명확한 신규 진입 조건이 없습니다. 다음 트리거가 나올 때까지 대기합니다."


def build_prohibition_list(row: pd.Series) -> List[str]:
    conditions = []
    if to_bool(row.get("algo_vol_extreme")):
        conditions.append("극단 변동성 구간: 신규 진입 금지")
    if as_float(row.get("close")) < as_float(row.get("sma_200")):
        conditions.append("종가가 200일선 아래: 추세추종 신규 롱 금지")
    if to_bool(row.get("algo_overextended_highvol")):
        conditions.append("50일선 대비 과이격 + 고변동: 추격매수 금지")
    if to_bool(row.get("algo_event_shock_day")):
        conditions.append("이벤트/충격일: 변동성 안정화 전 비중 확대 금지")
    if str(row.get("entry_trigger", "NONE")) == "NONE":
        conditions.append("진입 트리거 없음: 돌파 또는 눌림목 반등 확인 필요")
    return conditions or ["명시적 매수 금지 조건 없음"]


def validation_passed(quality: pd.DataFrame) -> str:
    if quality.empty or "passed" not in quality.columns:
        return "NA"
    passed = quality["passed"].astype(str).str.lower().isin(["true", "1", "yes"]).all()
    return "PASS" if passed else "FAIL"


def best_backtest_line(summary: pd.DataFrame) -> tuple[str, str, str]:
    if summary.empty or "cagr_pct" not in summary.columns:
        return "NA", "NA", "NA"
    ranked = summary.sort_values("cagr_pct", ascending=False).iloc[0]
    return (
        str(ranked.get("strategy_id", "NA")),
        pct_from_percent_value(as_float(ranked.get("cagr_pct"))),
        pct_from_percent_value(as_float(ranked.get("max_drawdown_pct"))),
    )


def walk_forward_line(walk_forward: pd.DataFrame, strategy_id: str) -> tuple[str, str]:
    if walk_forward.empty or strategy_id == "NA":
        return "NA", "NA"
    part = walk_forward[walk_forward["strategy_id"] == strategy_id].copy()
    if part.empty:
        return "NA", "NA"
    median_cagr = part["test_cagr_pct"].median()
    positive_rate = part["test_positive"].astype(str).str.lower().isin(["true", "1", "yes"]).mean() * 100
    return pct_from_percent_value(median_cagr), pct_from_percent_value(positive_rate)


def build_plan_rows(
    snapshot: Dict[str, str],
    signals: pd.DataFrame,
    risk_snapshot: Dict[str, str],
    prediction_snapshot: Dict[str, str],
    backtest_summary: pd.DataFrame,
    validation_quality: pd.DataFrame,
    walk_forward: pd.DataFrame,
    stress_snapshot: Dict[str, str],
    system_state: Dict[str, str],
) -> pd.DataFrame:
    last = signals.iloc[-1]
    prev_20d_high = signals["high"].shift(1).rolling(20, min_periods=10).max().iloc[-1]
    prev_60d_high = signals["high"].shift(1).rolling(60, min_periods=30).max().iloc[-1]
    max_weight = derive_max_weight(last)
    pullback_low = as_float(last["sma_50"]) * 0.97
    pullback_high = as_float(last["sma_50"]) * 1.05
    risk_per_share = 2.0 * as_float(last["atr_14"])
    prohibitions = build_prohibition_list(last)
    best_strategy, best_cagr, best_mdd = best_backtest_line(backtest_summary)
    best_wf_cagr, best_wf_positive_rate = walk_forward_line(walk_forward, best_strategy)

    rows = [
        ("판단", "기준일", last["date"].date().isoformat(), "", "최신 신호표 기준"),
        ("판단", "오늘 행동", action_ko(str(last["trade_action"])), "", str(last["trade_action"])),
        ("판단", "엄격 신호 단계", action_ko(str(last.get("strict_signal_stage", "STRICT_NO_ENTRY"))), "", str(last.get("strict_signal_stage", "STRICT_NO_ENTRY"))),
        ("판단", "연구/Paper 신호 단계", action_ko(str(last.get("research_signal_stage", "NO_RESEARCH_SIGNAL"))), "", str(last.get("research_signal_reason", ""))),
        ("판단", "연구/Paper 행동", action_ko(str(last.get("research_signal_action", "NO_ACTION"))), "", "실거래 주문이 아닌 관찰/가상 기록 단계"),
        ("판단", "연구 신호 점수", f"{as_float(last.get('research_signal_score')):.1f}", "점", f"level={last.get('research_signal_level', 'NA')}"),
        ("판단", "Paper 추적 비중", pct(as_float(last.get("paper_tracking_weight", 0.0))), "%", "Paper 후보일 때만 사용하는 가상 추적 비중"),
        ("판단", "알고리즘 점수", f"{as_float(last['score_price_algo_total']):.1f}", "점", "75 이상이면 트리거 동반 진입 후보"),
        ("판단", "진입 트리거", str(last["entry_trigger"]), "", "NONE이면 신규 진입 대기"),
        ("판단", "20D 돌파 트리거", str(bool(last.get("trigger_breakout_20d", False))), "", "분리된 bool 트리거"),
        ("판단", "60D 돌파 트리거", str(bool(last.get("trigger_breakout_60d", False))), "", "primary trigger에서 20D보다 우선"),
        ("판단", "50D 눌림목 트리거", str(bool(last.get("trigger_pullback_50d", False))), "", "분리된 bool 트리거"),
        ("판단", "Deep DD 회복 트리거", str(bool(last.get("trigger_deep_dd_recovery", False))), "", "자동 진입이 아닌 연구 전용"),
        ("가격", "현재 종가", usd(as_float(last["close"])), "USD", "최신 종가"),
        ("가격", "20일선", usd(as_float(last["sma_20"])), "USD", ""),
        ("가격", "50일선", usd(as_float(last["sma_50"])), "USD", ""),
        ("가격", "200일선", usd(as_float(last["sma_200"])), "USD", ""),
        ("진입", "20일 고점 돌파 기준가", usd(as_float(prev_20d_high)), "USD", "종가가 이 가격 위에서 마감하면 20D 돌파 확인"),
        ("진입", "60일 고점 돌파 기준가", usd(as_float(prev_60d_high)), "USD", "종가가 이 가격 위에서 마감하면 60D 돌파 확인"),
        ("진입", "50일선 눌림목 구간", f"{usd(pullback_low)} ~ {usd(pullback_high)}", "USD", "기존 알고리즘의 50일선 -3% ~ +5% 구간"),
        ("리스크", "2ATR 손절가", usd(as_float(last["atr_stop_2x"])), "USD", "현재 종가 기준 룰 엔진 손절가"),
        ("리스크", "3ATR 트레일링 손절가", usd(as_float(last["atr_trailing_stop_3x"])), "USD", "고변동 구간 보조 관리선"),
        ("리스크", "1주당 2ATR 리스크", usd(risk_per_share), "USD", "진입가 기준 실제 손절폭 산정에 사용"),
        ("목표", "1차 목표가 2R", usd(as_float(last["take_profit_2R"])), "USD", "현재 종가 기준 2R"),
        ("목표", "2차 목표가 3R", usd(as_float(last["take_profit_3R"])), "USD", "현재 종가 기준 3R"),
        ("비중", "현재 변동성 기준 최대 비중", pct(max_weight), "%", "평상 12%, 고변동 7%, 극단 4%, 200일선 아래 3%, 깊은 하락장 0%"),
        ("비중", "0.5% 계좌위험 기준 비중", pct(as_float(last["position_weight_if_0_5pct_account_risk"])), "%", "2ATR 손절폭 기준"),
        ("비중", "1.0% 계좌위험 기준 비중", pct(as_float(last["position_weight_if_1pct_account_risk"])), "%", "2ATR 손절폭 기준"),
        ("변동성", "ATR14", pct(as_float(last["atr_14_pct"])), "%", "종가 대비 ATR14"),
        ("변동성", "20일 연율화 변동성", pct(as_float(last["vol_20d_ann"])), "%", ""),
        ("위험", "고점 대비 낙폭", pct(as_float(last["drawdown_from_ath"])), "%", ""),
        ("위험", "50일선 대비 이격", pct(as_float(last["dist_close_sma_50_pct"])), "%", "과이격이면 추격매수 위험 증가"),
    ]

    if risk_snapshot:
        rows.extend(
            [
                ("리스크 엔진", "리스크 상태", risk_snapshot.get("risk_state", "NA"), "", "tsm_latest_risk_snapshot.csv"),
                ("리스크 엔진", "최종 권장 최대 비중", f"{risk_snapshot.get('final_recommended_max_weight', 'NA')}%", "%", "변동성/추세/낙폭/점수/계좌위험 제한의 최솟값"),
                ("리스크 엔진", "비중 제한 이유", risk_snapshot.get("limiting_reason", "NA"), "", ""),
            ]
        )

    if stress_snapshot:
        rows.extend(
            [
                ("스트레스", "스트레스 상태", stress_snapshot.get("stress_status", "NA"), "", "일봉 기반 시나리오"),
                ("스트레스", "최악 현재비중 시나리오", stress_snapshot.get("worst_current_weight_scenario", "NA"), "", ""),
                ("스트레스", "최악 현재비중 손실 추정", pct_string_value(stress_snapshot.get("worst_current_weight_portfolio_impact_pct")), "%", "최종 권장 비중 기준"),
                ("스트레스", "최악 시나리오 TSMC 가격", f"${as_float(stress_snapshot.get('worst_current_weight_implied_price')):,.2f}", "USD", ""),
            ]
        )

    if system_state:
        rows.extend(
            [
                ("시스템", "시스템 상태", system_state.get("system_state", "NA"), "", "일봉 시스템 운영 준비도"),
                ("시스템", "시스템 준비도 점수", f"{as_float(system_state.get('system_readiness_score')):.1f}", "점", "100점 만점"),
                ("시스템", "페이퍼 트레이딩 상태", system_state.get("paper_trading_status", "NA"), "", ""),
                ("시스템", "라이브 트레이딩 상태", system_state.get("live_trading_status", "NA"), "", "현재 주문 기능 없음"),
            ]
        )

    if prediction_snapshot:
        rows.extend(
            [
                ("예측", "예측 상태", prediction_snapshot.get("prediction_signal_status", "NA"), "", "기존 행동을 바꾸지 않는 보조 메타-라벨"),
                ("예측", "예측 사용 상태", prediction_snapshot.get("prediction_use_status", "NA"), "", "품질 기준 미달이면 표시 전용"),
                ("예측", "예측 적용 범위", prediction_snapshot.get("prediction_scope_used", "NA"), "", "trade_ready_entry, trigger_all, context_all 중 현재 행에 맞는 범위"),
                ("예측", "최신 후보 유형", prediction_snapshot.get("latest_candidate_scope", "NA"), "", "진입 트리거 여부와 행동 상태 기준"),
                ("예측", "진입 게이트 상태", prediction_snapshot.get("latest_entry_gate_status", "NA"), "", "룰 엔진이 실제 진입을 허용했는지 표시"),
                ("예측", "20일 성공확률", pct(as_float(prediction_snapshot.get("p_success_20d"))), "%", "현재 적용 범위의 2ATR 손절 회피 + 비용 차감 후 순수익 기준"),
                ("예측", "20일 손절 회피확률", pct(as_float(prediction_snapshot.get("p_stop_survival_20d"))), "%", "2단계 모델 1단계: 2ATR 손절을 피할 확률"),
                ("예측", "20일 손절 확률", pct(as_float(prediction_snapshot.get("p_stop_hit_20d"))), "%", "1 - 손절 회피확률"),
                ("예측", "20일 1R 도달확률", pct(as_float(prediction_snapshot.get("p_hit_1r_20d"))), "%", "메타 라벨 기반 1R 도달 확률"),
                ("예측", "20일 2R 도달확률", pct(as_float(prediction_snapshot.get("p_hit_2r_20d"))), "%", "메타 라벨 기반 2R 도달 확률"),
                ("예측", "20일 기대 R", f"{as_float(prediction_snapshot.get('expected_r_20d')):.2f}", "R", "손절폭 1R 기준 기대값"),
                ("예측", "20일 기대 순수익률", pct_string_value(prediction_snapshot.get("expected_net_return_20d")), "%", "비용 차감 후 label 평균 기반"),
                ("예측", "20일 조건부 수익확률", pct(as_float(prediction_snapshot.get("p_positive_given_survival_20d"))), "%", "2단계 모델 2단계: 손절 회피 조건에서 순수익 확률"),
                ("예측", "20일 성공확률 80% 하단", pct(as_float(prediction_snapshot.get("p_success_lower_80_20d"))), "%", "OOS 표본수 기반 보수적 Wilson 구간"),
                ("예측", "20일 성공확률 80% 상단", pct(as_float(prediction_snapshot.get("p_success_upper_80_20d"))), "%", "OOS 표본수 기반 보수적 Wilson 구간"),
                ("예측", "20일 유효 OOS 이벤트 수", prediction_snapshot.get("effective_oos_event_count_20d", "NA"), "건", "100건 미만이면 의사결정 보조 불가"),
                ("예측", "20일 확률 임계값", pct(as_float(prediction_snapshot.get("threshold_20d"))), "%", "walk-forward validation에서 선택"),
                ("예측", "60일 성공확률", pct(as_float(prediction_snapshot.get("p_success_60d"))), "%", "동일 기준 60거래일"),
                ("예측", "20일 모델", prediction_snapshot.get("best_model_20d", "NA"), "", ""),
                ("예측", "20일 Brier 개선", pct_string_value(prediction_snapshot.get("brier_improvement_pct_20d")), "%", "base rate 대비"),
                ("예측", "20일 ECE", f"{as_float(prediction_snapshot.get('ece_20d')):.4f}", "", "calibration error"),
                ("예측", "20일 기대값 개선", pct_string_value(prediction_snapshot.get("expectancy_improvement_pct_20d")), "%", "prediction-filtered OOS net expectancy 개선"),
                ("예측", "Decision permission", prediction_snapshot.get("decision_permission", "NA"), "", "DISPLAY_ONLY, PAPER_ONLY, LIVE_ALLOWED"),
                ("예측", "최종 예측 기반 결정", prediction_snapshot.get("final_trade_decision", "NA"), "", "룰 엔진 후보 위에 얹는 메타 결정"),
                ("예측", "모델 차단 사유", prediction_snapshot.get("model_quality_block_reasons", "NA"), "", "PASS가 아니면 실전 의사결정 금지"),
                ("예측", "Pooled 20D 모델", prediction_snapshot.get("pooled_model_name", "NA"), "", "12-symbol pooled trade_ready model"),
                ("예측", "Pooled 20D 성공확률", pct(as_float(prediction_snapshot.get("pooled_p_success_20d"))), "%", "TSMC 전용 보정 계층 적용"),
                ("예측", "Pooled 20D 임계값", pct(as_float(prediction_snapshot.get("pooled_threshold_20d"))), "%", "pooled validation 기준"),
                ("예측", "Pooled decision support", prediction_snapshot.get("pooled_decision_support_allowed", "NA"), "", "통과해도 live trading은 비활성"),
                ("예측", "Pooled 모델 차단 사유", prediction_snapshot.get("pooled_model_quality_block_reasons", "NA"), "", "pooled model gate"),
                ("예측", "Pooled 최신 기준 이하 사유", prediction_snapshot.get("pooled_decision_block_reasons", "NA"), "", "최신 TSMC 게이트"),
            ]
        )

    rows.extend(
        [
            ("검증", "백테스트 최상위 전략", best_strategy, "", "전체 기간 CAGR 기준"),
            ("검증", "최상위 전략 CAGR", best_cagr, "%", "거래비용 포함"),
            ("검증", "최상위 전략 MDD", best_mdd, "%", "거래비용 포함"),
            ("검증", "최상위 전략 WF 중앙 CAGR", best_wf_cagr, "%", "rolling test window 기준"),
            ("검증", "최상위 전략 WF 양수 비율", best_wf_positive_rate, "%", "test window total return > 0 비율"),
            ("검증", "기계적 품질 체크", validation_passed(validation_quality), "", "룩어헤드/비용/중복 포지션 등"),
        ]
    )

    for idx, condition in enumerate(prohibitions, start=1):
        rows.append(("매수 금지 조건", f"조건 {idx}", condition, "", ""))

    rows.append(("결론", "현재 해석", conclusion_ko(last), "", "교육/리서치용 판단"))

    # Keep the snapshot dependency visible for auditability.
    if snapshot:
        rows.append(("검증", "latest_snapshot_date", snapshot.get("date", ""), "", "스냅샷 파일에서 읽은 기준일"))

    return pd.DataFrame(rows, columns=["section", "item", "value", "unit", "notes"])


def write_markdown(outdir: Path, plan: pd.DataFrame, signals: pd.DataFrame) -> None:
    last = signals.iloc[-1]
    conclusion = plan.loc[(plan["section"] == "결론") & (plan["item"] == "현재 해석"), "value"].iloc[0]

    def value_of(item: str) -> str:
        s = plan.loc[plan["item"] == item, "value"]
        return s.iloc[0] if len(s) else "NA"

    prohibitions = plan.loc[plan["section"] == "매수 금지 조건", "value"].tolist()
    prohibition_text = "\n".join([f"- {p}" for p in prohibitions])

    lines = [
        "# TSMC Daily Trading Plan",
        "",
        f"- 기준일: {last['date'].date().isoformat()}",
        f"- 오늘 행동: {value_of('오늘 행동')}",
        f"- 엄격 신호 단계: {value_of('엄격 신호 단계')}",
        f"- 연구/Paper 신호 단계: {value_of('연구/Paper 신호 단계')}",
        f"- 연구/Paper 행동: {value_of('연구/Paper 행동')}",
        f"- 알고리즘 점수: {value_of('알고리즘 점수')}",
        f"- 결론: {conclusion}",
        "",
        "## Key Levels",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| 현재 종가 | {value_of('현재 종가')} |",
        f"| 20일 고점 돌파 기준가 | {value_of('20일 고점 돌파 기준가')} |",
        f"| 60일 고점 돌파 기준가 | {value_of('60일 고점 돌파 기준가')} |",
        f"| 50일선 눌림목 구간 | {value_of('50일선 눌림목 구간')} |",
        f"| 2ATR 손절가 | {value_of('2ATR 손절가')} |",
        f"| 1차 목표가 2R | {value_of('1차 목표가 2R')} |",
        f"| 2차 목표가 3R | {value_of('2차 목표가 3R')} |",
        f"| 현재 변동성 기준 최대 비중 | {value_of('현재 변동성 기준 최대 비중')} |",
        f"| 리스크 엔진 최종 권장 최대 비중 | {value_of('최종 권장 최대 비중')} |",
        "",
        "## Buy Restrictions",
        prohibition_text,
        "",
        "## Risk Policy",
        "",
        f"- 리스크 상태: {value_of('리스크 상태')}",
        f"- 비중 제한 이유: {value_of('비중 제한 이유')}",
        "",
        "## Daily Stress",
        "",
        f"- 스트레스 상태: {value_of('스트레스 상태')}",
        f"- 최악 현재비중 시나리오: {value_of('최악 현재비중 시나리오')}",
        f"- 최악 현재비중 손실 추정: {value_of('최악 현재비중 손실 추정')}",
        f"- 최악 시나리오 TSMC 가격: {value_of('최악 시나리오 TSMC 가격')}",
        "",
        "## Prediction Overlay",
        "",
        f"- 예측 상태: {value_of('예측 상태')}",
        f"- 예측 사용 상태: {value_of('예측 사용 상태')}",
        f"- 예측 적용 범위: {value_of('예측 적용 범위')}",
        f"- 최신 후보 유형: {value_of('최신 후보 유형')}",
        f"- 진입 게이트 상태: {value_of('진입 게이트 상태')}",
        f"- 20일 성공확률 / 임계값: {value_of('20일 성공확률')} / {value_of('20일 확률 임계값')}",
        f"- 20일 손절 회피확률 / 조건부 수익확률: {value_of('20일 손절 회피확률')} / {value_of('20일 조건부 수익확률')}",
        f"- 20일 성공확률 80% 구간: {value_of('20일 성공확률 80% 하단')} ~ {value_of('20일 성공확률 80% 상단')}",
        f"- 20일 유효 OOS 이벤트 수: {value_of('20일 유효 OOS 이벤트 수')}",
        f"- 60일 성공확률: {value_of('60일 성공확률')}",
        f"- 20일 모델: {value_of('20일 모델')}",
        f"- Brier 개선 / ECE: {value_of('20일 Brier 개선')} / {value_of('20일 ECE')}",
        f"- 기대값 개선: {value_of('20일 기대값 개선')}",
        "",
        "## Validation Evidence",
        "",
        f"- 백테스트 최상위 전략: {value_of('백테스트 최상위 전략')}",
        f"- 최상위 전략 CAGR / MDD: {value_of('최상위 전략 CAGR')} / {value_of('최상위 전략 MDD')}",
        f"- 워크포워드 중앙 CAGR / 양수 비율: {value_of('최상위 전략 WF 중앙 CAGR')} / {value_of('최상위 전략 WF 양수 비율')}",
        f"- 기계적 품질 체크: {value_of('기계적 품질 체크')}",
        "",
        "## System State",
        "",
        f"- 시스템 상태: {value_of('시스템 상태')}",
        f"- 시스템 준비도 점수: {value_of('시스템 준비도 점수')}",
        f"- 페이퍼 트레이딩 상태: {value_of('페이퍼 트레이딩 상태')}",
        f"- 라이브 트레이딩 상태: {value_of('라이브 트레이딩 상태')}",
        "",
        "This plan is research tooling, not investment advice.",
    ]

    (outdir / "tsm_daily_trading_plan.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the latest TSMC daily trading plan.")
    parser.add_argument("--latest", default="tsm_price_rule_output/tsm_latest_decision_snapshot.csv")
    parser.add_argument("--signals", default="tsm_price_rule_output/tsm_daily_algorithmic_signals.csv")
    parser.add_argument("--risk", default="tsm_price_rule_output/tsm_latest_risk_snapshot.csv")
    parser.add_argument("--backtest-summary", default="tsm_price_rule_output/tsm_backtest_strategy_summary.csv")
    parser.add_argument("--validation-quality", default="tsm_price_rule_output/tsm_validation_quality_checks.csv")
    parser.add_argument("--walk-forward", default="tsm_price_rule_output/tsm_validation_walk_forward_summary.csv")
    parser.add_argument("--stress", default="tsm_price_rule_output/tsm_latest_stress_snapshot.csv")
    parser.add_argument("--system-state", default="tsm_price_rule_output/tsm_latest_system_state.csv")
    parser.add_argument("--prediction", default="tsm_price_rule_output/tsm_latest_prediction_snapshot.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    snapshot = load_snapshot(Path(args.latest))
    signals = load_signals(Path(args.signals))
    risk_snapshot = load_optional_snapshot(Path(args.risk))
    prediction_snapshot = load_optional_snapshot(Path(args.prediction))
    backtest_summary = load_optional_csv(Path(args.backtest_summary))
    validation_quality = load_optional_csv(Path(args.validation_quality))
    walk_forward = load_optional_csv(Path(args.walk_forward))
    stress_snapshot = load_optional_snapshot(Path(args.stress))
    system_state = load_optional_snapshot(Path(args.system_state))
    plan = build_plan_rows(
        snapshot,
        signals,
        risk_snapshot,
        prediction_snapshot,
        backtest_summary,
        validation_quality,
        walk_forward,
        stress_snapshot,
        system_state,
    )
    plan.to_csv(outdir / "tsm_daily_trading_plan.csv", index=False)
    write_markdown(outdir, plan, signals)

    print("완료: daily trading plan =", outdir.resolve())
    print(plan.loc[plan["section"].isin(["판단", "결론"]), ["item", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()

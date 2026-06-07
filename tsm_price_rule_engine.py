#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top10 호환 일별 주가 통합 분석 + 규칙/알고리즘 수치화 엔진

입력 파일
- tsm_daily_10y_enriched.csv : 핵심 기준 파일. OHLCV + 수익률/변동성/추세/모멘텀/유동성/벤치마크 지표 포함
- tsm_daily_10y_raw.csv      : 원시 OHLCV 검증용
- tsm_daily_10y_summary.csv  : 기존 전체 요약값
- tsm_event_impact_10y.csv   : 이벤트 전후 수익률

출력 파일
- tsm_integrated_price_summary.csv
- tsm_yearly_price_volatility_stats.csv
- tsm_monthly_price_volatility_stats.csv
- tsm_extreme_daily_moves.csv
- tsm_drawdown_episodes.csv
- tsm_event_integrated_analysis.csv
- tsm_regime_forward_return_stats.csv
- tsm_rule_forward_return_stats.csv
- tsm_daily_algorithmic_signals.csv
- tsm_latest_decision_snapshot.csv
- tsm_algorithmic_rulebook.md

주의
- 본 코드는 투자 권유가 아니라 분석·검증용입니다.
- enriched 파일의 *_pct 컬럼은 대부분 비율 형태(예: 0.038 = 3.8%)입니다. 출력 파일은 pct 단위로 변환합니다.
- 신호는 당일 종가 기준으로 계산되며 실제 체결은 다음 거래일 시가/지정가 검증이 필요합니다.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from causal_utils import causal_rolling_quantile, rolling_percentile_rank, rolling_winsorize
from tsm_core.config import RuleEngineConfig, load_run_config

TRADING_DAYS = 252


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_id(*parts: object, prefix: str = "sig") -> str:
    text = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


# -----------------------------
# 공통 유틸
# -----------------------------

def pct(x: pd.Series | float) -> pd.Series | float:
    """ratio -> percent"""
    return x * 100


def causal_score(
    s: pd.Series,
    high_good: bool = True,
    window: int = 756,
    min_periods: int = 252,
) -> pd.Series:
    clipped = rolling_winsorize(s, window=window, min_periods=min_periods)
    return rolling_percentile_rank(clipped, window=window, min_periods=min_periods, high_good=high_good)


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")


def numeric_series(df: pd.DataFrame, columns: str | Iterable[str], default: float = np.nan) -> pd.Series:
    """Return a numeric Series for the first available column, preserving df.index."""
    if isinstance(columns, str):
        column_names = (columns,)
    else:
        column_names = tuple(columns)
    for column in column_names:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


NEWS_DAILY_COLUMNS = [
    "news_event_count_1d",
    "news_event_count_3d",
    "news_sentiment_score_1d",
    "news_primary_cause_type",
    "news_primary_cluster_id",
    "news_match_confidence",
    "news_match_confidence_score",
    "news_coverage_status",
    "news_source_count",
    "news_primary_source_url",
    "news_cause_summary",
    "news_penalty_event",
]

NO_HIGH_CONFIDENCE_CAUSE = "NO_HIGH_CONFIDENCE_NEWS"
NO_CLUSTER = "NO_CLUSTER"


def integrate_news_daily(df: pd.DataFrame, news_daily_path: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    defaults: dict[str, object] = {
        "news_event_count_1d": 0,
        "news_event_count_3d": 0,
        "news_sentiment_score_1d": 0.0,
        "news_primary_cause_type": NO_HIGH_CONFIDENCE_CAUSE,
        "news_primary_cluster_id": NO_CLUSTER,
        "news_match_confidence": "NO_MATCH",
        "news_match_confidence_score": 0.0,
        "news_coverage_status": "NOT_RUN",
        "news_source_count": 0,
        "news_primary_source_url": "",
        "news_cause_summary": "",
        "news_penalty_event": 0.0,
    }
    if news_daily_path and Path(news_daily_path).exists():
        news = pd.read_csv(news_daily_path)
        if not news.empty and "date" in news.columns:
            news = news.copy()
            news["date"] = pd.to_datetime(news["date"], errors="coerce")
            keep = ["date", *[c for c in NEWS_DAILY_COLUMNS if c in news.columns]]
            out = out.merge(news[keep].dropna(subset=["date"]).drop_duplicates("date", keep="last"), on="date", how="left")
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = out[col].fillna(default)
    for col in ["news_event_count_1d", "news_event_count_3d", "news_source_count"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in ["news_sentiment_score_1d", "news_match_confidence_score", "news_penalty_event"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


# -----------------------------
# 데이터 로드/검증
# -----------------------------

def load_inputs(enriched: str, raw: str, summary: Optional[str], events: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    df = pd.read_csv(enriched, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    raw_df = pd.read_csv(raw, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    summary_df = pd.read_csv(summary) if summary and Path(summary).exists() else None
    events_df = None
    if events and Path(events).exists():
        events_df = pd.read_csv(events)
        for c in ["event_date", "trading_date_used"]:
            if c in events_df.columns:
                events_df[c] = pd.to_datetime(events_df[c])

    required = [
        "date", "open", "high", "low", "close", "adj_close", "volume",
        "close_change_pct", "open_gap_pct", "open_to_close_pct", "intraday_range_pct_prev_close",
        "atr_14", "atr_14_pct", "vol_20d_ann", "vol_63d_ann", "vol_252d_ann",
        "drawdown_from_ath", "sma_20", "sma_50", "sma_200", "dist_close_sma_50_pct",
        "rsi_14", "return_20d", "return_60d", "return_126d", "return_252d",
        "breakout_20d_high", "breakout_60d_high", "dollar_volume", "dollar_volume_ma_20"
    ]
    require_columns(df, required)
    require_columns(raw_df, ["date", "open", "high", "low", "close", "adj_close", "volume"])
    return df, raw_df, summary_df, events_df


def validate_raw_vs_enriched(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    common = raw_df[["date", "open", "high", "low", "close", "adj_close", "volume"]].merge(
        df[["date", "open", "high", "low", "close", "adj_close", "volume"]],
        on="date", suffixes=("_raw", "_enriched"), how="inner"
    )
    out = []
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        diff = (common[f"{c}_raw"] - common[f"{c}_enriched"]).abs()
        max_diff = diff.max()
        mismatch_count = int((diff > (1e-6 if c != "volume" else 0)).sum())
        out.append({
            "check": f"raw_vs_enriched_{c}",
            "rows_compared": len(common),
            "mismatch_count": mismatch_count,
            "max_abs_diff": max_diff,
        })
    out.append({
        "check": "date_range",
        "rows_compared": len(df),
        "mismatch_count": 0,
        "max_abs_diff": np.nan,
        "note": f"{df['date'].min().date()} ~ {df['date'].max().date()}"
    })
    return pd.DataFrame(out)


# -----------------------------
# 지표/분석
# -----------------------------

def add_forward_returns(df: pd.DataFrame, horizons: Iterable[int] = (1, 2, 3, 5, 10, 20, 21, 40, 60, 63, 120, 126, 252)) -> pd.DataFrame:
    df = df.copy()
    for h in horizons:
        df[f"fwd_{h}d_ret"] = df["close"].shift(-h) / df["close"] - 1
    return df


def add_discovered_states(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = df.copy()
    df["atr14_q10_hist"] = causal_rolling_quantile(df["atr_14_pct"], 0.10)
    df["atr14_q25_hist"] = causal_rolling_quantile(df["atr_14_pct"], 0.25)
    df["atr14_q50_hist"] = causal_rolling_quantile(df["atr_14_pct"], 0.50)
    df["atr14_q75_hist"] = causal_rolling_quantile(df["atr_14_pct"], 0.75)
    df["atr14_q90_hist"] = causal_rolling_quantile(df["atr_14_pct"], 0.90)
    df["atr14_q95_hist"] = causal_rolling_quantile(df["atr_14_pct"], 0.95)
    df["vol20_q10_hist"] = causal_rolling_quantile(df["vol_20d_ann"], 0.10)
    df["vol20_q25_hist"] = causal_rolling_quantile(df["vol_20d_ann"], 0.25)
    df["vol20_q50_hist"] = causal_rolling_quantile(df["vol_20d_ann"], 0.50)
    df["vol20_q75_hist"] = causal_rolling_quantile(df["vol_20d_ann"], 0.75)
    df["vol20_q90_hist"] = causal_rolling_quantile(df["vol_20d_ann"], 0.90)
    df["vol20_q95_hist"] = causal_rolling_quantile(df["vol_20d_ann"], 0.95)
    df["range_q75_hist"] = causal_rolling_quantile(df["intraday_range_pct_prev_close"], 0.75)
    df["range_q90_hist"] = causal_rolling_quantile(df["intraday_range_pct_prev_close"], 0.90)
    df["range_q95_hist"] = causal_rolling_quantile(df["intraday_range_pct_prev_close"], 0.95)

    def latest_or_nan(col: str) -> float:
        valid = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(valid.iloc[-1]) if not valid.empty else np.nan

    q = {
        "atr10": latest_or_nan("atr14_q10_hist"),
        "atr25": latest_or_nan("atr14_q25_hist"),
        "atr50": latest_or_nan("atr14_q50_hist"),
        "atr75": latest_or_nan("atr14_q75_hist"),
        "atr90": latest_or_nan("atr14_q90_hist"),
        "atr95": latest_or_nan("atr14_q95_hist"),
        "vol20_10": latest_or_nan("vol20_q10_hist"),
        "vol20_25": latest_or_nan("vol20_q25_hist"),
        "vol20_50": latest_or_nan("vol20_q50_hist"),
        "vol20_75": latest_or_nan("vol20_q75_hist"),
        "vol20_90": latest_or_nan("vol20_q90_hist"),
        "vol20_95": latest_or_nan("vol20_q95_hist"),
        "range75": latest_or_nan("range_q75_hist"),
        "range90": latest_or_nan("range_q90_hist"),
        "range95": latest_or_nan("range_q95_hist"),
    }

    # 규칙 상태: 전부 당일 종가까지의 정보만 사용
    df["algo_trend_up_clean"] = (
        (df["close"] > df["sma_20"]) &
        (df["sma_20"] > df["sma_50"]) &
        (df["sma_50"] > df["sma_200"]) &
        (df["sma_50_slope_5d_pct"] > 0) &
        (df["sma_200_slope_5d_pct"] >= 0)
    )
    df["algo_trend_up_loose"] = (df["close"] > df["sma_50"]) & (df["close"] > df["sma_200"]) & (df["sma_50"] > df["sma_200"])
    df["algo_trend_down_clean"] = (df["close"] < df["sma_20"]) & (df["sma_20"] < df["sma_50"]) & (df["sma_50"] < df["sma_200"])

    df["algo_vol_extreme"] = (df["atr_14_pct"] > df["atr14_q90_hist"]) | (df["vol_20d_ann"] > df["vol20_q90_hist"])
    df["algo_vol_high"] = (df["atr_14_pct"] > df["atr14_q75_hist"]) | (df["vol_20d_ann"] > df["vol20_q75_hist"])
    df["algo_vol_high"] = df["algo_vol_high"].fillna(False)
    df["algo_vol_extreme"] = df["algo_vol_extreme"].fillna(False)
    df["algo_vol_low_normal"] = ~df["algo_vol_high"]

    df["algo_breakout20_clean"] = (df["breakout_20d_high"].astype(bool)) & df["algo_trend_up_loose"] & (~df["algo_vol_extreme"])
    df["algo_breakout60_clean"] = (df["breakout_60d_high"].astype(bool)) & df["algo_trend_up_loose"] & (~df["algo_vol_extreme"])

    # 상승 추세 눌림목 반등: 200일선 위 + 50일선 근처 + 단기 회복 + RSI 과열 아님
    df["algo_pullback_to_50_bounce"] = (
        df["algo_trend_up_loose"] &
        df["dist_close_sma_50_pct"].between(-0.03, 0.05) &
        (df["return_5d"] > 0) &
        (df["close_change_pct"] > 0) &
        df["rsi_14"].between(40, 65)
    )

    # 과열 경고: 상승 추세라도 50일선 대비 15% 이상 이격 + 고변동
    df["algo_overextended_highvol"] = df["algo_trend_up_loose"] & (df["dist_close_sma_50_pct"] > 0.15) & df["algo_vol_high"]

    # 급락/이벤트성 충격일: 단기 새 진입 금지 후보
    df["algo_event_shock_day"] = (
        (df["close_change_pct"].abs() >= 0.05) |
        (df["open_gap_pct"].abs() >= 0.04) |
        (df["intraday_range_pct_prev_close"] >= df["range_q95_hist"])
    )
    df["algo_event_shock_day"] = df["algo_event_shock_day"].fillna(False)

    # 깊은 하락장: 단기 반등 가능성은 있지만 일반 추세추종 신규매수는 금지
    df["algo_deep_downtrend_avoid"] = (df["close"] < df["sma_200"]) & (df["drawdown_from_ath"] < -0.25) & df["algo_vol_high"]

    # 상대강도: SPY/SMH 대비 60일 상대수익률이 모두 양수일 때만 강하게 인정
    if "relative_return_vs_spy_60d" in df.columns and "relative_return_vs_smh_60d" in df.columns:
        df["algo_relative_strength"] = df["algo_trend_up_loose"] & (df["relative_return_vs_spy_60d"] > 0) & (df["relative_return_vs_smh_60d"] > 0)
    else:
        df["algo_relative_strength"] = df["algo_trend_up_loose"]

    # 회복장: 200일선 아래/깊은 DD에서 강한 반등. 트레이딩은 가능하지만 포지션 축소형.
    df["algo_regime_recovery"] = (df["close"] > df["sma_50"]) & (df["sma_50_slope_5d_pct"] > 0) & (df["drawdown_from_ath"] < -0.20) & (df["return_20d"] > 0.08)

    return df, q


def add_scores_and_signals(
    df: pd.DataFrame,
    q: Dict[str, float],
    config: RuleEngineConfig | None = None,
    symbol: str = "TSM",
    symbol_group: str = "semiconductor",
) -> pd.DataFrame:
    df = df.copy()
    config = config or RuleEngineConfig()

    # 점수화: 같은 종목의 10년 분포 내 상대 점수. 0~100.
    # 모멘텀은 최근 20/60/126/252일 수익률 + 12-1 모멘텀을 결합.
    df["score_mom_20"] = causal_score(df["return_20d"], True)
    df["score_mom_60"] = causal_score(df["return_60d"], True)
    df["score_mom_126"] = causal_score(df["return_126d"], True)
    df["score_mom_252"] = causal_score(df["return_252d"], True)
    df["score_mom_12m_ex_1m"] = causal_score(df.get("momentum_12m_ex_1m", df["return_252d"]), True)
    df["score_momentum"] = (
        0.15 * df["score_mom_20"] +
        0.25 * df["score_mom_60"] +
        0.25 * df["score_mom_126"] +
        0.20 * df["score_mom_252"] +
        0.15 * df["score_mom_12m_ex_1m"]
    )

    # 추세 점수: 이동평균 배열 + 이격/기울기 + DD 상태
    trend_parts = []
    trend_parts.append((df["close"] > df["sma_200"]).astype(float) * 18)
    trend_parts.append((df["close"] > df["sma_50"]).astype(float) * 15)
    trend_parts.append((df["sma_50"] > df["sma_200"]).astype(float) * 15)
    trend_parts.append((df["close"] > df["sma_20"]).astype(float) * 10)
    trend_parts.append((df["sma_20"] > df["sma_50"]).astype(float) * 10)
    trend_parts.append((df["sma_50_slope_5d_pct"] > 0).astype(float) * 10)
    trend_parts.append((df["sma_200_slope_5d_pct"] >= 0).astype(float) * 8)
    trend_parts.append((df["return_60d"] > 0).astype(float) * 7)
    trend_parts.append((df["drawdown_from_ath"] > -0.10).astype(float) * 7)
    df["score_trend"] = np.sum(trend_parts, axis=0)

    # 변동성 점수: 낮을수록 높게. 단 TSM은 성장주라 너무 낮은 변동성만 최고로 보지 않고 극단고변동을 강하게 감점.
    df["score_low_vol"] = (
        0.35 * causal_score(df["atr_14_pct"], False) +
        0.35 * causal_score(df["vol_20d_ann"], False) +
        0.20 * causal_score(df["intraday_range_pct_prev_close"].rolling(20).mean(), False) +
        0.10 * causal_score(df["drawdown_from_ath"].abs(), False)
    )

    # 상대강도 점수: SPY/SMH/QQQ 대비 우위. 없는 컬럼은 중립 처리.
    rel_cols = [c for c in ["relative_return_vs_spy_60d", "relative_return_vs_smh_60d", "relative_return_vs_qqq_60d"] if c in df.columns]
    if rel_cols:
        rel_scores = [causal_score(df[c], True) for c in rel_cols]
        df["score_relative_strength"] = pd.concat(rel_scores, axis=1).mean(axis=1, skipna=True).fillna(50)
    else:
        df["score_relative_strength"] = 50

    # 유동성 점수: 달러 거래대금은 높을수록 좋고, 거래량 급증은 돌파 확인에는 좋지만 과열에서는 이벤트 페널티로 분리.
    df["score_liquidity"] = causal_score(df["dollar_volume_ma_20"], True)

    # 이벤트/과열 페널티
    df["penalty_event"] = 0.0
    df.loc[df["algo_event_shock_day"], "penalty_event"] += 8
    df.loc[df["algo_overextended_highvol"], "penalty_event"] += 8
    df.loc[df["algo_deep_downtrend_avoid"], "penalty_event"] += 12
    df.loc[df["atr_14_pct"] > df["atr14_q95_hist"], "penalty_event"] += 5
    df.loc[df["vol_20d_ann"] > df["vol20_q95_hist"], "penalty_event"] += 5
    if "news_penalty_event" not in df.columns:
        df["news_penalty_event"] = 0.0
    df["news_penalty_event"] = pd.to_numeric(df["news_penalty_event"], errors="coerce").fillna(0.0)

    # 종목별 트레이딩 총점: 펀더멘털 없이 가격/위험/유동성 중심
    df["score_price_algo_total"] = (
        0.30 * df["score_trend"] +
        0.25 * df["score_momentum"] +
        0.20 * df["score_relative_strength"] +
        0.15 * df["score_low_vol"] +
        0.10 * df["score_liquidity"] -
        df["penalty_event"] -
        df["news_penalty_event"]
    ).clip(0, 100)

    # 진입/보유/축소 규칙
    df["trigger_breakout_20d"] = df["algo_breakout20_clean"].astype(bool)
    df["trigger_breakout_60d"] = df["algo_breakout60_clean"].astype(bool)
    df["trigger_pullback_50d"] = df["algo_pullback_to_50_bounce"].astype(bool)
    df["trigger_deep_dd_recovery"] = df["algo_regime_recovery"].astype(bool)

    df["entry_trigger"] = np.select(
        [
            df["trigger_breakout_60d"],
            df["trigger_breakout_20d"],
            df["trigger_pullback_50d"],
            df["trigger_deep_dd_recovery"],
        ],
        ["60D_BREAKOUT", "20D_BREAKOUT", "50D_PULLBACK_BOUNCE", "DEEP_DD_RECOVERY"],
        default="NONE",
    )

    df["trade_action"] = "NO_TRADE"
    entry_threshold = float(config.score_entry_threshold)
    watchlist_threshold = float(config.watchlist_threshold)
    stop_atr_multiple = float(config.stop_atr_multiple)
    take_profit_r_multiple = float(config.take_profit_r_multiple)

    df.loc[(df["score_price_algo_total"] >= entry_threshold) & (df["entry_trigger"] != "NONE") & (~df["algo_vol_extreme"]), "trade_action"] = "ENTRY_ALLOWED"
    df.loc[df["entry_trigger"] == "DEEP_DD_RECOVERY", "trade_action"] = "RESEARCH_ONLY_DEEP_DD"
    df.loc[(df["score_price_algo_total"] >= entry_threshold) & (df["entry_trigger"] == "NONE") & df["algo_trend_up_loose"], "trade_action"] = "HOLD_OR_WAIT_TRIGGER"
    df.loc[(df["score_price_algo_total"].between(watchlist_threshold, entry_threshold)) & df["algo_trend_up_loose"], "trade_action"] = "WATCHLIST_PULLBACK_ONLY"
    df.loc[df["algo_overextended_highvol"], "trade_action"] = "REDUCE_OR_DO_NOT_CHASE"
    df.loc[df["algo_deep_downtrend_avoid"], "trade_action"] = "AVOID_TREND_LONG"
    df.loc[df["close"] < df["sma_200"], "trade_action"] = np.where(
        df.loc[df["close"] < df["sma_200"], "trade_action"].eq("AVOID_TREND_LONG"),
        "AVOID_TREND_LONG",
        "NO_NEW_LONG_BELOW_200D"
    )
    df.loc[df["entry_trigger"] == "DEEP_DD_RECOVERY", "trade_action"] = "RESEARCH_ONLY_DEEP_DD"

    # 리스크/주문 수치화
    df["atr_stop_2x"] = df["close"] - stop_atr_multiple * df["atr_14"]
    df["atr_trailing_stop_3x"] = df["close"] - 3.0 * df["atr_14"]
    df["risk_per_share_2atr"] = stop_atr_multiple * df["atr_14"]
    df["risk_pct_2atr"] = df["risk_per_share_2atr"] / df["close"]

    # 위험에 따른 최대 비중: 평상 12%, 고변동 7%, 극단 4%, 200일선 아래 0~3% 관찰
    df["max_weight_by_vol"] = float(config.max_weight_normal)
    df.loc[df["algo_vol_high"], "max_weight_by_vol"] = float(config.max_weight_high_vol)
    df.loc[df["algo_vol_extreme"], "max_weight_by_vol"] = float(config.max_weight_extreme_vol)
    df.loc[df["close"] < df["sma_200"], "max_weight_by_vol"] = float(config.max_weight_below_200d)
    df.loc[df["algo_deep_downtrend_avoid"], "max_weight_by_vol"] = 0.00

    # 1% 계좌위험 기준 이론 포지션 비중 = 1% / 손절폭%. 다만 max_weight_by_vol로 제한.
    df["position_weight_if_1pct_account_risk"] = np.minimum(0.01 / df["risk_pct_2atr"], df["max_weight_by_vol"])
    df["position_weight_if_0_5pct_account_risk"] = np.minimum(0.005 / df["risk_pct_2atr"], df["max_weight_by_vol"])

    # 익절 기준: 2R, 3R
    df["take_profit_2R"] = df["close"] + take_profit_r_multiple * df["risk_per_share_2atr"]
    df["take_profit_3R"] = df["close"] + 3 * df["risk_per_share_2atr"]

    df = add_two_stage_signal_layers(df)
    df = add_semiconductor_momentum_v2_layers(df)

    df["signal_version"] = str(config.signal_version)
    df["rule_family"] = str(config.rule_family)
    df["hypothesis_id"] = str(config.hypothesis_id)
    df["decision_policy_version"] = str(config.decision_policy_version)
    df["symbol"] = str(symbol).upper()
    df["symbol_group"] = str(symbol_group)
    df["available_at_utc"] = df["date"].dt.strftime("%Y-%m-%dT21:00:00+00:00") if "date" in df.columns else now_utc_iso()
    df["signal_id"] = [
        stable_id(symbol, date.date().isoformat() if hasattr(date, "date") else date, version, trigger, action, prefix="sig")
        for symbol, date, version, trigger, action in zip(
            df["symbol"],
            df["date"] if "date" in df.columns else pd.Series(range(len(df)), index=df.index),
            df["signal_version"],
            df["entry_trigger"],
            df["trade_action"],
        )
    ]

    return df


V2_ACTIONABLE_TIERS = {"STRICT_ENTRY_ALLOWED", "AGGRESSIVE_TREND_ENTRY", "BREAKOUT_EXTENSION_TINY", "PULLBACK_REENTRY"}
MEMORY_AI_GROUP_TOKENS = ("memory", "fabless_ai", "ai")


def add_semiconductor_momentum_v2_layers(df: pd.DataFrame) -> pd.DataFrame:
    """Add semiconductor-specific 5-20D momentum decision columns.

    The v1 strict signal is preserved. V2 records raw momentum events even when
    volatility or extension would make v1 suppress the clean trigger.
    """
    out = df.copy()
    close = numeric_series(out, "close")
    high = numeric_series(out, ("high", "close"))
    sma50 = numeric_series(out, "sma_50")
    sma200 = numeric_series(out, "sma_200")
    ema10 = numeric_series(out, ("ema_10", "sma_10"))
    ret20 = numeric_series(out, "return_20d", default=0.0).fillna(0.0)
    ret60 = numeric_series(out, "return_60d", default=0.0).fillna(0.0)
    close_change = numeric_series(out, "close_change_pct", default=0.0).fillna(0.0)
    open_gap = numeric_series(out, "open_gap_pct", default=0.0).fillna(0.0)
    open_to_close = numeric_series(out, "open_to_close_pct", default=0.0).fillna(0.0)
    score = numeric_series(out, "score_price_algo_total", default=0.0).fillna(0.0)
    score_momentum = numeric_series(out, "score_momentum", default=50.0).fillna(50.0)
    score_relative = numeric_series(out, "score_relative_strength", default=50.0).fillna(50.0)
    dist50 = numeric_series(out, "dist_close_sma_50_pct", default=0.0).fillna(0.0)
    rel_smh = numeric_series(out, ("relative_return_vs_smh_20d", "relative_return_vs_smh_60d"))
    rel_qqq = numeric_series(out, ("relative_return_vs_qqq_20d", "relative_return_vs_qqq_60d"))
    relative_strength_vs_smh_qqq = pd.concat([rel_smh, rel_qqq], axis=1).mean(axis=1, skipna=True).fillna(0.0)
    symbol_group = out.get("symbol_group", pd.Series("semiconductor", index=out.index)).fillna("semiconductor").astype(str).str.lower()

    prev_20d_high = pd.to_numeric(out.get("prev_20d_high"), errors="coerce") if "prev_20d_high" in out.columns else high.shift(1).rolling(20, min_periods=10).max()
    prev_60d_high = pd.to_numeric(out.get("prev_60d_high"), errors="coerce") if "prev_60d_high" in out.columns else high.shift(1).rolling(60, min_periods=30).max()
    prev_252d_high = pd.to_numeric(out.get("prev_252d_high"), errors="coerce") if "prev_252d_high" in out.columns else high.shift(1).rolling(252, min_periods=126).max()

    out["raw_breakout_20d"] = close.gt(prev_20d_high).fillna(False)
    out["raw_breakout_60d"] = close.gt(prev_60d_high).fillna(False)
    out["raw_52w_high_near"] = (close.ge(prev_252d_high * 0.97) | out.get("breakout_252d_high", pd.Series(False, index=out.index)).fillna(False).astype(bool)).fillna(False)
    out["positive_thrust_day"] = ((close_change >= 0.03) | (open_gap >= 0.04) | (open_to_close >= 0.025)).fillna(False)
    out["negative_shock_day"] = ((close_change <= -0.04) | (open_gap <= -0.04) | (open_to_close <= -0.035)).fillna(False)
    out["relative_strength_vs_smh_qqq"] = relative_strength_vs_smh_qqq

    memory_ai_group = symbol_group.apply(lambda g: any(token in g for token in MEMORY_AI_GROUP_TOKENS))
    raw_breakout_any = out["raw_breakout_20d"] | out["raw_breakout_60d"] | out["raw_52w_high_near"]
    trend_ok = out.get("algo_trend_up_loose", pd.Series(False, index=out.index)).fillna(False).astype(bool) & close.ge(sma50) & close.ge(sma200)
    strong_relative = (relative_strength_vs_smh_qqq > 0.0) | out.get("algo_relative_strength", pd.Series(False, index=out.index)).fillna(False).astype(bool) | score_relative.ge(60.0)

    out["semi_group_momentum_score"] = (
        ret20.gt(0).astype(float) * 18.0
        + ret60.gt(0).astype(float) * 18.0
        + raw_breakout_any.astype(float) * 22.0
        + strong_relative.astype(float) * 18.0
        + score_momentum.ge(70.0).astype(float) * 14.0
        + out["positive_thrust_day"].astype(float) * 10.0
    ).clip(0, 100)
    out["memory_ai_regime_score"] = (
        out["semi_group_momentum_score"] * 0.65
        + memory_ai_group.astype(float) * 20.0
        + ret20.gt(0.08).astype(float) * 8.0
        + ret60.gt(0.15).astype(float) * 7.0
    ).clip(0, 100)
    out["semi_momentum_regime"] = np.select(
        [
            out["memory_ai_regime_score"].ge(70.0),
            out["semi_group_momentum_score"].ge(65.0),
            out["semi_group_momentum_score"].ge(45.0),
        ],
        ["MEMORY_AI_LEADERSHIP", "SEMI_MOMENTUM_UP", "SEMI_NEUTRAL"],
        default="SEMI_WEAK",
    )
    out["raw_entry_event"] = np.select(
        [
            out["raw_breakout_60d"],
            out["raw_breakout_20d"],
            out["raw_52w_high_near"],
            out["positive_thrust_day"],
        ],
        ["RAW_60D_BREAKOUT", "RAW_20D_BREAKOUT", "RAW_52W_HIGH_NEAR", "POSITIVE_THRUST_DAY"],
        default="NONE",
    )

    vol_extreme = out.get("algo_vol_extreme", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    overextended = out.get("algo_overextended_highvol", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    event_shock = out.get("algo_event_shock_day", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    below_200d = close.lt(sma200).fillna(False)
    avoid = below_200d | out.get("algo_deep_downtrend_avoid", pd.Series(False, index=out.index)).fillna(False).astype(bool) | out["negative_shock_day"] | out.get("algo_trend_down_clean", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    strong_semi = out["semi_group_momentum_score"].ge(65.0) | out["memory_ai_regime_score"].ge(70.0)
    high_vol_or_extension = vol_extreme | overextended | event_shock | dist50.gt(0.20)
    strict_entry = out.get("trade_action", pd.Series("", index=out.index)).fillna("").astype(str).eq("ENTRY_ALLOWED")
    aggressive_entry = (~avoid) & raw_breakout_any & strong_semi & strong_relative & score.ge(60.0) & (~vol_extreme) & (~overextended)
    extension_tiny = (~avoid) & raw_breakout_any & out["positive_thrust_day"] & strong_semi & score.ge(50.0) & high_vol_or_extension
    pullback_reentry = (~avoid) & out.get("algo_pullback_to_50_bounce", pd.Series(False, index=out.index)).fillna(False).astype(bool) & strong_semi & score.ge(55.0)
    watch_trend = (~avoid) & trend_ok & strong_semi & score.ge(55.0)

    out["decision_tier"] = np.select(
        [strict_entry, aggressive_entry, extension_tiny, pullback_reentry, watch_trend, avoid],
        ["STRICT_ENTRY_ALLOWED", "AGGRESSIVE_TREND_ENTRY", "BREAKOUT_EXTENSION_TINY", "PULLBACK_REENTRY", "WATCH_TREND", "AVOID_OR_WAIT"],
        default="AVOID_OR_WAIT",
    )
    base_weight = numeric_series(out, "position_weight_if_0_5pct_account_risk", default=0.0).fillna(0.0).clip(lower=0.0)
    strict_weight = np.minimum(np.maximum(base_weight, 0.04), 0.07)
    aggressive_weight = np.minimum(np.maximum(base_weight, 0.04), 0.07)
    reduced_weight = np.minimum(np.maximum(base_weight, 0.02), 0.04)
    tiny_weight = np.minimum(np.maximum(base_weight, 0.005), 0.03)
    out["suggested_weight"] = np.select(
        [
            out["decision_tier"].eq("STRICT_ENTRY_ALLOWED"),
            out["decision_tier"].eq("AGGRESSIVE_TREND_ENTRY"),
            out["decision_tier"].eq("BREAKOUT_EXTENSION_TINY"),
            out["decision_tier"].eq("PULLBACK_REENTRY"),
        ],
        [strict_weight, aggressive_weight, tiny_weight, reduced_weight],
        default=0.0,
    ).astype(float)
    out["sizing_tier"] = np.select(
        [
            out["decision_tier"].eq("STRICT_ENTRY_ALLOWED"),
            out["decision_tier"].eq("AGGRESSIVE_TREND_ENTRY"),
            out["decision_tier"].eq("BREAKOUT_EXTENSION_TINY"),
            out["decision_tier"].eq("PULLBACK_REENTRY"),
            out["decision_tier"].eq("WATCH_TREND"),
        ],
        ["STANDARD_4_7", "STANDARD_4_7", "TINY_0_5_3", "REDUCED_2_4", "WATCH_0"],
        default="NO_SIZE",
    )
    out["suggested_action"] = np.select(
        [
            out["decision_tier"].eq("STRICT_ENTRY_ALLOWED"),
            out["decision_tier"].eq("AGGRESSIVE_TREND_ENTRY"),
            out["decision_tier"].eq("BREAKOUT_EXTENSION_TINY"),
            out["decision_tier"].eq("PULLBACK_REENTRY"),
            out["decision_tier"].eq("WATCH_TREND"),
        ],
        ["STRICT_BUY_ALLOWED", "SEMI_MOMENTUM_ENTRY", "HIGH_VOL_TINY_EXTENSION", "PULLBACK_REENTRY", "WATCH_FOR_5_20D_SETUP"],
        default="AVOID_OR_WAIT",
    )
    out["execution_status"] = np.where(out["decision_tier"].isin(V2_ACTIONABLE_TIERS), "PAPER_INTENT_ALLOWED_LIVE_DISABLED", "DISPLAY_ONLY")
    out["stop_price_1_8atr"] = close - 1.8 * numeric_series(out, "atr_14")
    out["invalidation_5d_low"] = numeric_series(out, ("low", "close")).rolling(5, min_periods=1).min()
    out["invalidation_ema10"] = ema10
    out["next_check_condition"] = np.select(
        [
            out["decision_tier"].eq("BREAKOUT_EXTENSION_TINY"),
            out["decision_tier"].eq("AGGRESSIVE_TREND_ENTRY"),
            out["decision_tier"].eq("PULLBACK_REENTRY"),
            out["decision_tier"].eq("WATCH_TREND"),
        ],
        [
            "HOLD_ONLY_IF_CLOSE_ABOVE_5D_LOW_AND_10D_EMA",
            "TRAIL_WITH_1_8ATR_OR_10D_EMA",
            "INVALIDATE_IF_CLOSE_BELOW_50D_BAND",
            "WAIT_FOR_RAW_BREAKOUT_OR_PULLBACK_REENTRY",
        ],
        default="WAIT_FOR_NEW_SETUP",
    )
    return out


def add_two_stage_signal_layers(df: pd.DataFrame) -> pd.DataFrame:
    """Add a softer research/paper signal layer without changing strict live signals."""
    out = df.copy()
    score = numeric_series(out, "score_price_algo_total", default=0.0).fillna(0.0)
    close = numeric_series(out, "close")
    sma20 = numeric_series(out, "sma_20")
    sma50 = numeric_series(out, "sma_50")
    sma200 = numeric_series(out, "sma_200")
    ret20 = numeric_series(out, "return_20d", default=0.0).fillna(0.0)
    drawdown = numeric_series(out, "drawdown_from_ath", default=-1.0).fillna(-1.0)
    entry_trigger = out.get("entry_trigger", pd.Series("NONE", index=out.index)).fillna("NONE").astype(str)
    trade_action = out.get("trade_action", pd.Series("NO_TRADE", index=out.index)).fillna("NO_TRADE").astype(str)

    def flag(name: str) -> pd.Series:
        if name not in out.columns:
            return pd.Series(False, index=out.index)
        return out[name].fillna(False).astype(bool)

    has_strict_trigger = entry_trigger.ne("NONE")
    live_entry = trade_action.eq("ENTRY_ALLOWED")
    trend_ok = flag("algo_trend_up_loose") & close.ge(sma200) & close.ge(sma50)
    constructive_price = close.ge(sma20) & ret20.gt(0.0) & drawdown.gt(-0.12)
    risk_block = (
        close.lt(sma200)
        | flag("algo_deep_downtrend_avoid")
        | flag("algo_overextended_highvol")
        | flag("algo_event_shock_day")
        | flag("algo_vol_extreme")
    )
    paper_setup = (~live_entry) & (~risk_block) & trend_ok & has_strict_trigger & score.ge(65.0)
    early_watch = (~live_entry) & (~paper_setup) & (~risk_block) & trend_ok & constructive_price & score.ge(60.0)
    watchlist = (~live_entry) & (~paper_setup) & (~early_watch) & (~risk_block) & trend_ok & score.ge(55.0)

    out["strict_signal_stage"] = np.select(
        [
            live_entry,
            has_strict_trigger,
        ],
        [
            "STRICT_LIVE_ENTRY",
            "STRICT_TRIGGER_WAITING_FOR_SCORE_OR_RISK",
        ],
        default="STRICT_NO_ENTRY",
    )
    out["research_signal_stage"] = np.select(
        [
            live_entry,
            paper_setup,
            early_watch,
            watchlist,
            risk_block,
        ],
        [
            "LIVE_ENTRY_ALREADY_ALLOWED",
            "PAPER_BUY_SETUP",
            "EARLY_BULLISH_WATCH",
            "WATCHLIST_ONLY",
            "RESEARCH_BLOCKED_RISK",
        ],
        default="NO_RESEARCH_SIGNAL",
    )
    out["research_signal_action"] = np.select(
        [
            live_entry,
            paper_setup,
            early_watch,
            watchlist,
        ],
        [
            "USE_STRICT_LIVE_SIGNAL",
            "PAPER_TRACK_LONG_SETUP",
            "WATCH_ONLY_EARLY_BULLISH",
            "WATCH_ONLY",
        ],
        default="NO_ACTION",
    )
    out["research_signal_level"] = np.select(
        [live_entry, paper_setup, early_watch, watchlist],
        [4, 3, 2, 1],
        default=0,
    ).astype(int)
    out["research_signal_score"] = (
        score
        + trend_ok.astype(float) * 8.0
        + constructive_price.astype(float) * 6.0
        + has_strict_trigger.astype(float) * 10.0
        - risk_block.astype(float) * 25.0
    ).clip(0, 100)
    out["research_signal_reason"] = np.select(
        [
            live_entry,
            paper_setup,
            early_watch,
            watchlist,
            risk_block,
        ],
        [
            "STRICT_ENTRY_ALLOWED",
            "TRIGGER_WITH_SCORE65_AND_RISK_OK",
            "SCORE60_TREND_OK_PRICE_ABOVE_20D",
            "TREND_OK_SCORE55_WATCHLIST",
            "RISK_BLOCK_VOL_EVENT_OR_TREND",
        ],
        default="NO_TRIGGER_OR_SCORE_TOO_LOW",
    )
    base_paper_weight = numeric_series(out, "position_weight_if_0_5pct_account_risk", default=0.0).fillna(0.0)
    out["paper_tracking_weight"] = 0.0
    out.loc[paper_setup, "paper_tracking_weight"] = np.minimum(base_paper_weight.loc[paper_setup], 0.02)
    return out


def integrated_summary(df: pd.DataFrame, validation: pd.DataFrame, q: Dict[str, float]) -> pd.DataFrame:
    d = df.dropna(subset=["close_change_pct"]).copy()
    start = df.iloc[0]
    end = df.iloc[-1]
    total_close_ret = end["close"] / start["close"] - 1
    total_adj_ret = end["adj_close"] / start["adj_close"] - 1
    years = (end["date"] - start["date"]).days / 365.25
    max_up = d.loc[d["close_change_pct"].idxmax()]
    max_down = d.loc[d["close_change_pct"].idxmin()]

    rows = [
        ("period", f"{start['date'].date()} ~ {end['date'].date()}"),
        ("trading_days", len(df)),
        ("listing_currency", end.get("listing_currency", "USD")),
        ("display_currency", end.get("display_currency", "USD")),
        ("engine_currency", end.get("engine_currency", "USD")),
        ("fx_pair", end.get("fx_pair", "")),
        ("latest_fx_rate_to_usd", end.get("fx_rate_to_usd", 1.0)),
        ("latest_usdkrw", end.get("usdkrw", np.nan)),
        ("start_close_usd", start["close"]),
        ("end_close_usd", end["close"]),
        ("start_close_native", start.get("close_native", start["close"])),
        ("end_close_native", end.get("close_native", end["close"])),
        ("latest_close_native", end.get("close_native", end["close"])),
        ("price_total_return_pct", pct(total_close_ret)),
        ("adj_total_return_pct", pct(total_adj_ret)),
        ("price_cagr_pct", pct((1 + total_close_ret) ** (1 / years) - 1)),
        ("adj_cagr_pct", pct((1 + total_adj_ret) ** (1 / years) - 1)),
        ("mean_daily_return_pct", pct(d["close_change_pct"].mean())),
        ("median_daily_return_pct", pct(d["close_change_pct"].median())),
        ("daily_vol_ann_pct", pct(d["close_change_pct"].std() * math.sqrt(TRADING_DAYS))),
        ("downside_vol_ann_pct", pct(d.loc[d["close_change_pct"] < 0, "close_change_pct"].std() * math.sqrt(TRADING_DAYS))),
        ("avg_abs_daily_return_pct", pct(d["close_change_pct"].abs().mean())),
        ("median_abs_daily_return_pct", pct(d["close_change_pct"].abs().median())),
        ("avg_intraday_range_pct", pct(d["intraday_range_pct_prev_close"].mean())),
        ("median_intraday_range_pct", pct(d["intraday_range_pct_prev_close"].median())),
        ("max_drawdown_from_ath_pct", pct(d["drawdown_from_ath"].min())),
        ("positive_day_ratio_pct", pct((d["close_change_pct"] > 0).mean())),
        ("negative_day_ratio_pct", pct((d["close_change_pct"] < 0).mean())),
        ("days_up_3pct_or_more", int((d["close_change_pct"] >= 0.03).sum())),
        ("days_down_3pct_or_more", int((d["close_change_pct"] <= -0.03).sum())),
        ("days_up_5pct_or_more", int((d["close_change_pct"] >= 0.05).sum())),
        ("days_down_5pct_or_more", int((d["close_change_pct"] <= -0.05).sum())),
        ("days_up_8pct_or_more", int((d["close_change_pct"] >= 0.08).sum())),
        ("days_down_8pct_or_more", int((d["close_change_pct"] <= -0.08).sum())),
        ("best_daily_return_pct", pct(max_up["close_change_pct"])),
        ("best_daily_return_date", str(max_up["date"].date())),
        ("worst_daily_return_pct", pct(max_down["close_change_pct"])),
        ("worst_daily_return_date", str(max_down["date"].date())),
        ("atr14_latest_hist_median_pct", pct(q["atr50"])),
        ("atr14_latest_hist_75pct_threshold_pct", pct(q["atr75"])),
        ("atr14_latest_hist_90pct_threshold_pct", pct(q["atr90"])),
        ("vol20_latest_hist_median_ann_pct", pct(q["vol20_50"])),
        ("vol20_latest_hist_75pct_threshold_ann_pct", pct(q["vol20_75"])),
        ("vol20_latest_hist_90pct_threshold_ann_pct", pct(q["vol20_90"])),
        ("latest_close_usd", end["close"]),
        ("latest_return_20d_pct", pct(end["return_20d"])),
        ("latest_return_60d_pct", pct(end["return_60d"])),
        ("latest_return_126d_pct", pct(end["return_126d"])),
        ("latest_return_252d_pct", pct(end["return_252d"])),
        ("latest_atr14_pct", pct(end["atr_14_pct"])),
        ("latest_vol20_ann_pct", pct(end["vol_20d_ann"])),
        ("latest_drawdown_from_ath_pct", pct(end["drawdown_from_ath"])),
        ("latest_trend_regime", end.get("trend_regime", "NA")),
        ("latest_vol_regime", end.get("vol_regime", "NA")),
        ("latest_algo_action", end.get("trade_action", "NA")),
        ("latest_strict_signal_stage", end.get("strict_signal_stage", "NA")),
        ("latest_research_signal_stage", end.get("research_signal_stage", "NA")),
        ("latest_research_signal_action", end.get("research_signal_action", "NA")),
        ("latest_research_signal_level", end.get("research_signal_level", np.nan)),
        ("latest_research_signal_score", end.get("research_signal_score", np.nan)),
        ("latest_algo_score", end.get("score_price_algo_total", np.nan)),
    ]

    # validation 요약도 포함
    for _, r in validation.iterrows():
        rows.append((f"validation_{r['check']}_mismatches", r.get("mismatch_count", np.nan)))
    return pd.DataFrame(rows, columns=["metric", "value"])


def yearly_stats(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["close_change_pct"]).copy()
    d["year"] = d["date"].dt.year
    rows = []
    for y, g in d.groupby("year"):
        first = df[df["date"].dt.year == y].iloc[0]
        last = df[df["date"].dt.year == y].iloc[-1]
        rows.append({
            "year": y,
            "days": len(g),
            "start_close": first["close"],
            "end_close": last["close"],
            "year_return_pct": pct(last["close"] / first["close"] - 1),
            "mean_daily_pct": pct(g["close_change_pct"].mean()),
            "median_daily_pct": pct(g["close_change_pct"].median()),
            "avg_abs_daily_pct": pct(g["close_change_pct"].abs().mean()),
            "ann_vol_pct": pct(g["close_change_pct"].std() * math.sqrt(TRADING_DAYS)),
            "avg_intraday_range_pct": pct(g["intraday_range_pct_prev_close"].mean()),
            "avg_atr14_pct": pct(g["atr_14_pct"].mean()),
            "max_drawdown_pct": pct(g["drawdown_from_ath"].min()),
            "max_up_day_pct": pct(g["close_change_pct"].max()),
            "max_down_day_pct": pct(g["close_change_pct"].min()),
            "up_days": int((g["close_change_pct"] > 0).sum()),
            "down_days": int((g["close_change_pct"] < 0).sum()),
            "up_5pct_count": int((g["close_change_pct"] >= 0.05).sum()),
            "down_5pct_count": int((g["close_change_pct"] <= -0.05).sum()),
        })
    return pd.DataFrame(rows)


def monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["close_change_pct"]).copy()
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    rows = []
    for (y, m), g in d.groupby(["year", "month"]):
        rows.append({
            "year": y,
            "month": m,
            "days": len(g),
            "month_return_pct": pct(g["close"].iloc[-1] / g["close"].iloc[0] - 1),
            "mean_daily_pct": pct(g["close_change_pct"].mean()),
            "avg_abs_daily_pct": pct(g["close_change_pct"].abs().mean()),
            "ann_vol_pct": pct(g["close_change_pct"].std() * math.sqrt(TRADING_DAYS)) if len(g) > 1 else np.nan,
            "avg_intraday_range_pct": pct(g["intraday_range_pct_prev_close"].mean()),
            "max_up_day_pct": pct(g["close_change_pct"].max()),
            "max_down_day_pct": pct(g["close_change_pct"].min()),
            "avg_atr14_pct": pct(g["atr_14_pct"].mean()),
            "high_vol_days": int((g["algo_vol_high"]).sum()),
            "event_shock_days": int((g["algo_event_shock_day"]).sum()),
        })
    return pd.DataFrame(rows)


def extreme_daily_moves(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "date", "open", "high", "low", "close", "volume",
        "close_change_pct", "open_gap_pct", "open_to_close_pct", "intraday_range_pct_prev_close",
        "volume_ratio_20", "atr_14_pct", "vol_20d_ann", "drawdown_from_ath", "trend_regime",
        "entry_trigger", "trade_action", "score_price_algo_total"
    ]
    rows = []
    specs = [
        ("largest_up_close_to_close", df.nlargest(20, "close_change_pct")),
        ("largest_down_close_to_close", df.nsmallest(20, "close_change_pct")),
        ("largest_positive_gap", df.nlargest(20, "open_gap_pct")),
        ("largest_negative_gap", df.nsmallest(20, "open_gap_pct")),
        ("widest_intraday_range", df.nlargest(20, "intraday_range_pct_prev_close")),
    ]
    for label, part in specs:
        tmp = part[base_cols].copy()
        tmp.insert(0, "extreme_type", label)
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    pct_cols = [c for c in out.columns if c.endswith("_pct") or c in ["vol_20d_ann", "drawdown_from_ath"]]
    for c in pct_cols:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pct(out[c])
    return out


def drawdown_episodes(df: pd.DataFrame, threshold: float = -0.10) -> pd.DataFrame:
    d = df.copy().sort_values("date").reset_index(drop=True)
    d["running_max_close"] = d["close"].cummax()
    d["dd"] = d["close"] / d["running_max_close"] - 1
    episodes = []
    in_ep = False
    start_idx = trough_idx = None
    peak_date = None
    peak_price = None

    for i, row in d.iterrows():
        if (not in_ep) and row["dd"] <= threshold:
            peak_idx = d.loc[:i, "close"].idxmax()
            in_ep = True
            start_idx = i
            trough_idx = i
            peak_date = d.loc[peak_idx, "date"]
            peak_price = d.loc[peak_idx, "close"]
        if in_ep:
            if row["dd"] < d.loc[trough_idx, "dd"]:
                trough_idx = i
            if row["close"] >= peak_price and i > start_idx:
                episodes.append({
                    "peak_date": peak_date,
                    "drawdown_start_date": d.loc[start_idx, "date"],
                    "trough_date": d.loc[trough_idx, "date"],
                    "recovery_date": row["date"],
                    "peak_close": peak_price,
                    "trough_close": d.loc[trough_idx, "close"],
                    "trough_drawdown_pct": pct(d.loc[trough_idx, "dd"]),
                    "calendar_days_peak_to_trough": (d.loc[trough_idx, "date"] - peak_date).days,
                    "calendar_days_peak_to_recovery": (row["date"] - peak_date).days,
                })
                in_ep = False
    if in_ep and start_idx is not None:
        episodes.append({
            "peak_date": peak_date,
            "drawdown_start_date": d.loc[start_idx, "date"],
            "trough_date": d.loc[trough_idx, "date"],
            "recovery_date": pd.NaT,
            "peak_close": peak_price,
            "trough_close": d.loc[trough_idx, "close"],
            "trough_drawdown_pct": pct(d.loc[trough_idx, "dd"]),
            "calendar_days_peak_to_trough": (d.loc[trough_idx, "date"] - peak_date).days,
            "calendar_days_peak_to_recovery": np.nan,
        })
    out = pd.DataFrame(episodes)
    if out.empty:
        return out
    return out.sort_values("trough_drawdown_pct")


def event_integrated_analysis(events_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if events_df is None or events_df.empty:
        return pd.DataFrame()
    out = events_df.copy()
    # 핵심 퍼센트 컬럼은 이미 pct 단위로 저장된 것으로 보이는 파일이므로 그대로 둔다.
    # 이벤트 분류 스코어: 당일 충격 + 5/20/60일 지속성.
    out["event_impact_score"] = (
        out["return_on_event_day_pct"].abs().fillna(0) * 0.35 +
        out["post_5d_return_pct"].abs().fillna(0) * 0.20 +
        out["post_20d_return_pct"].abs().fillna(0) * 0.25 +
        out["post_60d_return_pct"].abs().fillna(0) * 0.20
    )
    out["event_direction_class"] = np.select(
        [
            (out["return_on_event_day_pct"] < -3) & (out["post_20d_return_pct"] < 0),
            (out["return_on_event_day_pct"] > 3) & (out["post_20d_return_pct"] > 0),
            (out["return_on_event_day_pct"].abs() >= 3) & (out["post_20d_return_pct"].abs() < 2),
        ],
        ["NEGATIVE_PERSISTENT", "POSITIVE_PERSISTENT", "ONE_DAY_SHOCK_MEAN_REVERT"],
        default="MIXED_OR_LOW_IMPACT"
    )
    return out.sort_values("event_impact_score", ascending=False)


def forward_stats_for_mask(df: pd.DataFrame, mask: pd.Series, rule_name: str, horizons=(1, 5, 20, 60, 120)) -> List[dict]:
    rows = []
    for h in horizons:
        col = f"fwd_{h}d_ret"
        s = df.loc[mask.fillna(False) & df[col].notna(), col]
        if len(s) == 0:
            continue
        rows.append({
            "rule": rule_name,
            "horizon_days": h,
            "n_signals": int(len(s)),
            "mean_fwd_return_pct": pct(s.mean()),
            "median_fwd_return_pct": pct(s.median()),
            "win_rate_pct": pct((s > 0).mean()),
            "p10_fwd_return_pct": pct(s.quantile(0.10)),
            "p90_fwd_return_pct": pct(s.quantile(0.90)),
            "worst_fwd_return_pct": pct(s.min()),
            "best_fwd_return_pct": pct(s.max()),
        })
    return rows


def rule_forward_return_stats(df: pd.DataFrame) -> pd.DataFrame:
    rules = {
        "trend_up_clean": df["algo_trend_up_clean"],
        "trend_up_loose": df["algo_trend_up_loose"],
        "trend_down_clean": df["algo_trend_down_clean"],
        "vol_low_normal": df["algo_vol_low_normal"],
        "vol_high": df["algo_vol_high"],
        "vol_extreme": df["algo_vol_extreme"],
        "breakout20_clean": df["algo_breakout20_clean"],
        "breakout60_clean": df["algo_breakout60_clean"],
        "trigger_breakout_20d": df.get("trigger_breakout_20d", df["algo_breakout20_clean"]),
        "trigger_breakout_60d": df.get("trigger_breakout_60d", df["algo_breakout60_clean"]),
        "trigger_pullback_50d": df.get("trigger_pullback_50d", df["algo_pullback_to_50_bounce"]),
        "trigger_deep_dd_recovery": df.get("trigger_deep_dd_recovery", df["algo_regime_recovery"]),
        "v2_aggressive_trend_entry": (df.get("decision_tier", pd.Series("", index=df.index)) == "AGGRESSIVE_TREND_ENTRY"),
        "v2_breakout_extension_tiny": (df.get("decision_tier", pd.Series("", index=df.index)) == "BREAKOUT_EXTENSION_TINY"),
        "v2_pullback_reentry": (df.get("decision_tier", pd.Series("", index=df.index)) == "PULLBACK_REENTRY"),
        "pullback_to_50_bounce": df["algo_pullback_to_50_bounce"],
        "overextended_highvol": df["algo_overextended_highvol"],
        "deep_downtrend_avoid": df["algo_deep_downtrend_avoid"],
        "relative_strength": df["algo_relative_strength"],
        "event_shock_day": df["algo_event_shock_day"],
        "entry_allowed_score75": (df["trade_action"] == "ENTRY_ALLOWED"),
        "hold_or_wait_score75": (df["trade_action"] == "HOLD_OR_WAIT_TRIGGER"),
        "research_early_bullish_watch": (df.get("research_signal_stage", pd.Series("", index=df.index)) == "EARLY_BULLISH_WATCH"),
        "research_paper_buy_setup": (df.get("research_signal_stage", pd.Series("", index=df.index)) == "PAPER_BUY_SETUP"),
    }
    rows = []
    for name, m in rules.items():
        rows.extend(forward_stats_for_mask(df, m, name))
    return pd.DataFrame(rows)


def regime_forward_return_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["trend_regime", "vol_regime", "trade_action", "entry_trigger", "decision_tier", "semi_momentum_regime"]:
        if col not in df.columns:
            continue
        for val, g in df.groupby(col, dropna=False):
            mask = df[col].eq(val)
            for r in forward_stats_for_mask(df, mask, f"{col}={val}", horizons=(20, 60)):
                rows.append(r)
    return pd.DataFrame(rows)


def news_cause_forward_return_stats(df: pd.DataFrame) -> pd.DataFrame:
    if "news_primary_cause_type" not in df.columns:
        return pd.DataFrame()
    rows = []
    cause = df["news_primary_cause_type"].fillna("").astype(str).str.strip()
    confident = df.get("news_match_confidence", pd.Series("", index=df.index)).fillna("").astype(str).eq("HIGH")
    for value, mask in {
        "ALL_HIGH_CONFIDENCE_NEWS": confident & cause.ne(""),
        **{f"cause_type={name}": confident & cause.eq(name) for name in sorted(cause[cause.ne("")].unique())},
    }.items():
        if not mask.any():
            continue
        rows.extend(forward_stats_for_mask(df, mask, value, horizons=(1, 5, 20, 60, 120)))
    return pd.DataFrame(rows)


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    last = df.iloc[-1]
    fields = [
        "date", "symbol", "symbol_group", "open", "high", "low", "close", "adj_close", "volume",
        "open_native", "high_native", "low_native", "close_native", "adj_close_native",
        "open_usd", "high_usd", "low_usd", "close_usd", "adj_close_usd",
        "listing_currency", "display_currency", "engine_currency", "fx_pair", "fx_rate_to_usd", "usdkrw", "fx_date",
        "close_change_pct", "return_20d", "return_60d", "return_126d", "return_252d",
        "sma_20", "sma_50", "sma_200", "dist_close_sma_20_pct", "dist_close_sma_50_pct", "dist_close_sma_200_pct",
        "rsi_14", "atr_14", "atr_14_pct", "vol_20d_ann", "vol_63d_ann", "vol_252d_ann",
        "drawdown_from_ath", "trend_regime", "vol_regime", "momentum_signal",
        "beta_vs_spy_252d", "beta_vs_smh_252d", "relative_return_vs_spy_60d", "relative_return_vs_smh_60d",
        "atr14_q75_hist", "atr14_q90_hist", "atr14_q95_hist", "vol20_q75_hist", "vol20_q90_hist", "vol20_q95_hist", "range_q95_hist",
        "score_trend", "score_momentum", "score_relative_strength", "score_low_vol", "score_liquidity", "penalty_event", "news_penalty_event", "score_price_algo_total",
        "news_event_count_1d", "news_event_count_3d", "news_sentiment_score_1d", "news_primary_cause_type", "news_primary_cluster_id",
        "news_match_confidence", "news_match_confidence_score", "news_coverage_status", "news_source_count", "news_primary_source_url", "news_cause_summary",
        "trigger_breakout_20d", "trigger_breakout_60d", "trigger_pullback_50d", "trigger_deep_dd_recovery",
        "entry_trigger", "trade_action", "atr_stop_2x", "atr_trailing_stop_3x", "risk_pct_2atr",
        "position_weight_if_0_5pct_account_risk", "position_weight_if_1pct_account_risk", "take_profit_2R", "take_profit_3R",
        "strict_signal_stage", "research_signal_stage", "research_signal_action", "research_signal_level",
        "research_signal_score", "research_signal_reason", "paper_tracking_weight",
        "raw_entry_event", "raw_breakout_20d", "raw_breakout_60d", "raw_52w_high_near",
        "positive_thrust_day", "negative_shock_day", "relative_strength_vs_smh_qqq",
        "semi_group_momentum_score", "memory_ai_regime_score", "semi_momentum_regime",
        "decision_tier", "sizing_tier", "suggested_action", "suggested_weight",
        "execution_status", "stop_price_1_8atr", "invalidation_5d_low", "invalidation_ema10", "next_check_condition",
    ]
    rows = []
    ratio_like = {
        "close_change_pct", "return_20d", "return_60d", "return_126d", "return_252d",
        "dist_close_sma_20_pct", "dist_close_sma_50_pct", "dist_close_sma_200_pct",
        "atr_14_pct", "vol_20d_ann", "vol_63d_ann", "vol_252d_ann", "drawdown_from_ath",
        "relative_return_vs_spy_60d", "relative_return_vs_smh_60d", "risk_pct_2atr",
        "position_weight_if_0_5pct_account_risk", "position_weight_if_1pct_account_risk",
        "paper_tracking_weight",
        "suggested_weight",
        "atr14_q75_hist", "atr14_q90_hist", "atr14_q95_hist",
        "vol20_q75_hist", "vol20_q90_hist", "vol20_q95_hist", "range_q95_hist",
    }
    for f in fields:
        if f not in df.columns:
            continue
        v = last[f]
        rows.append({"field": f, "value": pct(v) if f in ratio_like and pd.notna(v) and not isinstance(v, str) else v})
    return pd.DataFrame(rows)


def write_rulebook_md(outdir: Path, df: pd.DataFrame, q: Dict[str, float], summary_df: pd.DataFrame, rule_stats: pd.DataFrame) -> None:
    last = df.iloc[-1]
    # 편의값
    def metric(name):
        s = summary_df.loc[summary_df["metric"] == name, "value"]
        return s.iloc[0] if len(s) else "NA"

    # 20일 평균 결과 추출
    def rule_line(rule: str, h: int = 20) -> str:
        r = rule_stats[(rule_stats["rule"] == rule) & (rule_stats["horizon_days"] == h)]
        if r.empty:
            return "NA"
        r = r.iloc[0]
        return f"n={int(r['n_signals'])}, 평균 {r['mean_fwd_return_pct']:.2f}%, 중앙값 {r['median_fwd_return_pct']:.2f}%, 승률 {r['win_rate_pct']:.1f}%"

    lines = []
    lines.append("# Top10 호환 일별 주가 기반 규칙/알고리즘 정리본")
    lines.append("")
    lines.append("## 1. 데이터 기준")
    lines.append(f"- 기간: {metric('period')}")
    lines.append(f"- 거래일 수: {metric('trading_days')}")
    lines.append(f"- 시작 종가/최종 종가: ${float(metric('start_close_usd')):.2f} → ${float(metric('end_close_usd')):.2f}")
    lines.append(f"- 가격 기준 누적 상승률: {float(metric('price_total_return_pct')):.2f}%")
    lines.append(f"- 수정주가 기준 누적 상승률: {float(metric('adj_total_return_pct')):.2f}%")
    lines.append(f"- 수정주가 기준 CAGR: {float(metric('adj_cagr_pct')):.2f}%")
    lines.append(f"- 연율화 변동성: {float(metric('daily_vol_ann_pct')):.2f}%")
    lines.append(f"- 최대 낙폭(MDD): {float(metric('max_drawdown_from_ath_pct')):.2f}%")
    lines.append("")
    lines.append("## 2. Causal 변동성 기준선")
    lines.append("- 아래 기준선은 전체 10년 분포가 아니라 최신일 기준 과거 rolling window로 계산한 historical threshold입니다.")
    lines.append(f"- ATR14 rolling 중앙값: {q['atr50']*100:.2f}%, 75% 기준: {q['atr75']*100:.2f}%, 90% 기준: {q['atr90']*100:.2f}%, 95% 기준: {q['atr95']*100:.2f}%")
    lines.append(f"- 20일 연율화 변동성 rolling 중앙값: {q['vol20_50']*100:.2f}%, 75% 기준: {q['vol20_75']*100:.2f}%, 90% 기준: {q['vol20_90']*100:.2f}%, 95% 기준: {q['vol20_95']*100:.2f}%")
    lines.append(
        f"- 실전 규칙: ATR14가 {q['atr75']*100:.2f}%를 넘으면 고변동, "
        f"{q['atr90']*100:.2f}%를 넘으면 극단 변동성으로 분류한다. "
        "극단 변동성에서는 신규 추격매수를 금지하고 포지션 한도를 낮춘다."
    )
    lines.append("")
    lines.append("## 3. 발견된 핵심 패턴")
    lines.append(f"- 20일 고점 돌파 + 상승추세 + 비극단 변동성: {rule_line('breakout20_clean', 20)}")
    lines.append(f"- 60일 고점 돌파 + 상승추세 + 비극단 변동성: {rule_line('breakout60_clean', 20)}")
    lines.append(f"- 50일선 눌림목 반등: {rule_line('pullback_to_50_bounce', 20)}")
    lines.append(f"- 50일선 대비 15% 이상 이격 + 고변동 과열: {rule_line('overextended_highvol', 20)}")
    lines.append(f"- 이벤트/충격일: {rule_line('event_shock_day', 20)}")
    lines.append("해석: TSM은 장기 우상향 종목이라 깊은 하락장 뒤 반등 통계가 좋아 보일 수 있지만, 이는 큰 변동성과 손절 난도가 동반되는 구간이다. 일반 추세추종 시스템에서는 200일선 아래의 깊은 하락장은 신규 롱 금지 또는 아주 작은 관찰 비중으로만 취급한다.")
    lines.append("")
    lines.append("## 4. 최종 알고리즘")
    lines.append("```text")
    lines.append("1) 데이터 검증: raw와 enriched의 OHLCV 날짜/값 일치 확인")
    lines.append("2) 변동성 분류: ATR14%, 20일 연율화 변동성, 일중 변동폭으로 LOW/NORMAL/HIGH/EXTREME 결정")
    lines.append("3) 추세 분류: Close, SMA20, SMA50, SMA200 배열과 SMA 기울기로 상승/혼합/하락 분류")
    lines.append("4) 모멘텀 점수: 20/60/126/252일 수익률 + 12-1M 모멘텀을 rolling causal percentile로 0~100점화")
    lines.append("5) 상대강도 점수: SPY/SMH/QQQ 대비 60일 상대수익률을 rolling causal percentile로 점수화")
    lines.append("6) 총점 = 0.30*추세 + 0.25*모멘텀 + 0.20*상대강도 + 0.15*저변동성 + 0.10*유동성 - 이벤트/과열 페널티")
    lines.append("7) 진입 후보: bool trigger를 먼저 분리하고 primary trigger는 60D 돌파, 20D 돌파, 50D 눌림목, deep DD 회복 순으로 지정")
    lines.append("8) 청산/축소: Close<SMA50이면 축소, Close<SMA200이면 신규 롱 금지/청산, ATR 급등+이격 과다이면 추격 금지")
    lines.append("9) 손절: Entry - 2*ATR14. 고변동 구간은 2.5~3*ATR 또는 비중 축소")
    lines.append("10) 비중: 계좌위험 0.5~1.0% / (2*ATR14/가격), 단 평상 12%, 고변동 7%, 극단 4%, 200일선 아래 0~3% 한도")
    lines.append("```")
    lines.append("")
    lines.append("## 5. 최신 상태")
    lines.append(f"- 최신일: {last['date'].date()}")
    lines.append(f"- 종가: ${last['close']:.2f}")
    lines.append(f"- 20/50/200일선: ${last['sma_20']:.2f} / ${last['sma_50']:.2f} / ${last['sma_200']:.2f}")
    lines.append(f"- 20/60/126/252일 수익률: {last['return_20d']*100:.2f}% / {last['return_60d']*100:.2f}% / {last['return_126d']*100:.2f}% / {last['return_252d']*100:.2f}%")
    lines.append(f"- ATR14: {last['atr_14_pct']*100:.2f}%, 20일 연율화 변동성: {last['vol_20d_ann']*100:.2f}%")
    lines.append(f"- 고점 대비 낙폭: {last['drawdown_from_ath']*100:.2f}%")
    lines.append(f"- 알고리즘 점수: {last['score_price_algo_total']:.1f}, 행동: {last['trade_action']}, primary 트리거: {last['entry_trigger']}")
    lines.append(f"- 트리거 bool: 20D={bool(last.get('trigger_breakout_20d', False))}, 60D={bool(last.get('trigger_breakout_60d', False))}, 50D={bool(last.get('trigger_pullback_50d', False))}, deepDD={bool(last.get('trigger_deep_dd_recovery', False))}")
    lines.append(f"- 2ATR 손절가: ${last['atr_stop_2x']:.2f}, 2R 목표가: ${last['take_profit_2R']:.2f}, 3R 목표가: ${last['take_profit_3R']:.2f}")
    lines.append("")
    lines.append("## 6. 산출물")
    lines.append("- tsm_daily_algorithmic_signals.csv: 모든 거래일별 점수, 상태, 행동, 손절가, 목표가, 비중 계산")
    lines.append("- tsm_rule_forward_return_stats.csv: 규칙별 1/5/20/60/120일 사후수익률 통계")
    lines.append("- tsm_extreme_daily_moves.csv: 상승폭/하락폭/갭/일중변동폭 극단일")
    lines.append("- tsm_drawdown_episodes.csv: 10% 이상 주요 낙폭 구간")
    lines.append("- tsm_event_integrated_analysis.csv: 이벤트별 단기/중기 충격 분석")
    lines.append("- tsm_news_cause_forward_return_stats.csv: 뉴스 원인 카테고리별 이후 수익률 분석")
    (outdir / "tsm_algorithmic_rulebook.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# 메인
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched", default="tsm_daily_10y_enriched.csv")
    parser.add_argument("--raw", default="tsm_daily_10y_raw.csv")
    parser.add_argument("--summary", default="tsm_daily_10y_summary.csv")
    parser.add_argument("--events", default="tsm_event_impact_10y.csv")
    parser.add_argument("--news-daily", default="")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--config", default="config/tsm_research.toml")
    parser.add_argument("--symbol", default="TSM")
    parser.add_argument("--symbol-group", default="semiconductor")
    args = parser.parse_args()
    run_config = load_run_config(args.config)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, raw_df, _, events_df = load_inputs(args.enriched, args.raw, args.summary, args.events)
    validation = validate_raw_vs_enriched(df, raw_df)
    df = add_forward_returns(df)
    df, q = add_discovered_states(df)
    df = integrate_news_daily(df, args.news_daily)
    df = add_scores_and_signals(df, q, run_config.rule_engine, symbol=args.symbol, symbol_group=args.symbol_group)

    summary_out = integrated_summary(df, validation, q)
    yearly_out = yearly_stats(df)
    monthly_out = monthly_stats(df)
    extreme_out = extreme_daily_moves(df)
    dd_out = drawdown_episodes(df)
    event_out = event_integrated_analysis(events_df)
    rule_stats_out = rule_forward_return_stats(df)
    regime_stats_out = regime_forward_return_stats(df)
    news_cause_stats_out = news_cause_forward_return_stats(df)
    latest_out = latest_snapshot(df)

    # 일별 신호 파일은 너무 크지 않게 핵심 컬럼만 저장
    signal_cols = [
        "date", "open", "high", "low", "close", "adj_close", "volume", "dollar_volume", "dollar_volume_ma_20",
        "open_native", "high_native", "low_native", "close_native", "adj_close_native", "open_usd", "high_usd", "low_usd", "close_usd", "adj_close_usd",
        "listing_currency", "display_currency", "engine_currency", "fx_pair", "fx_rate_to_usd", "usdkrw", "fx_date",
        "symbol", "symbol_group",
        "close_change_pct", "open_gap_pct", "open_to_close_pct", "intraday_range_pct_prev_close",
        "return_20d", "return_60d", "return_126d", "return_252d", "momentum_12m_ex_1m",
        "sma_20", "sma_50", "sma_200", "dist_close_sma_20_pct", "dist_close_sma_50_pct", "dist_close_sma_200_pct",
        "rsi_14", "atr_14", "atr_14_pct", "vol_20d_ann", "vol_63d_ann", "vol_252d_ann", "drawdown_from_ath",
        "atr14_q75_hist", "atr14_q90_hist", "atr14_q95_hist", "vol20_q75_hist", "vol20_q90_hist", "vol20_q95_hist", "range_q95_hist",
        "trend_regime", "vol_regime", "momentum_signal",
        "relative_return_vs_spy_60d", "relative_return_vs_smh_60d", "relative_return_vs_qqq_60d",
        "beta_vs_spy_252d", "beta_vs_smh_252d", "beta_vs_qqq_252d",
        "algo_trend_up_clean", "algo_trend_up_loose", "algo_vol_high", "algo_vol_extreme", "algo_breakout20_clean", "algo_breakout60_clean",
        "algo_pullback_to_50_bounce", "algo_overextended_highvol", "algo_deep_downtrend_avoid", "algo_event_shock_day", "algo_relative_strength",
        "trigger_breakout_20d", "trigger_breakout_60d", "trigger_pullback_50d", "trigger_deep_dd_recovery",
        "raw_entry_event", "raw_breakout_20d", "raw_breakout_60d", "raw_52w_high_near",
        "positive_thrust_day", "negative_shock_day", "relative_strength_vs_smh_qqq",
        "semi_group_momentum_score", "memory_ai_regime_score", "semi_momentum_regime",
        "decision_tier", "sizing_tier", "suggested_action", "suggested_weight",
        "execution_status", "stop_price_1_8atr", "invalidation_5d_low", "invalidation_ema10", "next_check_condition",
        "score_trend", "score_momentum", "score_relative_strength", "score_low_vol", "score_liquidity", "penalty_event", "news_penalty_event", "score_price_algo_total",
        "news_event_count_1d", "news_event_count_3d", "news_sentiment_score_1d", "news_primary_cause_type", "news_primary_cluster_id",
        "news_match_confidence", "news_match_confidence_score", "news_coverage_status", "news_source_count", "news_primary_source_url", "news_cause_summary",
        "entry_trigger", "trade_action", "atr_stop_2x", "atr_trailing_stop_3x", "risk_pct_2atr", "position_weight_if_0_5pct_account_risk", "position_weight_if_1pct_account_risk", "take_profit_2R", "take_profit_3R",
        "strict_signal_stage", "research_signal_stage", "research_signal_action", "research_signal_level", "research_signal_score", "research_signal_reason", "paper_tracking_weight",
        "signal_id", "signal_version", "rule_family", "hypothesis_id", "available_at_utc", "decision_policy_version"
    ]
    signal_cols = [c for c in signal_cols if c in df.columns]
    signals_out = df[signal_cols].copy()

    # CSV 출력: ratio 계열 컬럼은 유지한다. 파일명/컬럼명에 pct가 붙은 원본은 ratio이므로 code users가 후속 계산하기 좋게 유지.
    validation.to_csv(outdir / "tsm_data_validation_raw_vs_enriched.csv", index=False)
    summary_out.to_csv(outdir / "tsm_integrated_price_summary.csv", index=False)
    yearly_out.to_csv(outdir / "tsm_yearly_price_volatility_stats.csv", index=False)
    monthly_out.to_csv(outdir / "tsm_monthly_price_volatility_stats.csv", index=False)
    extreme_out.to_csv(outdir / "tsm_extreme_daily_moves.csv", index=False)
    dd_out.to_csv(outdir / "tsm_drawdown_episodes.csv", index=False)
    event_out.to_csv(outdir / "tsm_event_integrated_analysis.csv", index=False)
    regime_stats_out.to_csv(outdir / "tsm_regime_forward_return_stats.csv", index=False)
    rule_stats_out.to_csv(outdir / "tsm_rule_forward_return_stats.csv", index=False)
    news_cause_stats_out.to_csv(outdir / "tsm_news_cause_forward_return_stats.csv", index=False)
    signals_out.to_csv(outdir / "tsm_daily_algorithmic_signals.csv", index=False)
    latest_out.to_csv(outdir / "tsm_latest_decision_snapshot.csv", index=False)

    write_rulebook_md(outdir, df, q, summary_out, rule_stats_out)

    print("완료: output directory =", outdir.resolve())
    print("latest action:", df.iloc[-1][["date", "close", "score_price_algo_total", "entry_trigger", "trade_action", "atr_stop_2x", "take_profit_2R", "position_weight_if_0_5pct_account_risk"]].to_dict())


if __name__ == "__main__":
    main()

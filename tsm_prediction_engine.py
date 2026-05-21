#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSM conservative prediction accuracy engine.

This engine does not predict the next price. It estimates the probability that
an existing rule-engine event would be successful if entered at the next
session open under the same daily-data assumptions as the backtest engine.

Outputs:
- tsm_prediction_label_dataset.csv
- tsm_prediction_feature_matrix.csv
- tsm_prediction_candidate_scope_stats.csv
- tsm_prediction_walk_forward_metrics.csv
- tsm_prediction_model_comparison.csv
- tsm_prediction_calibration_bins.csv
- tsm_prediction_threshold_policy.csv
- tsm_latest_prediction_snapshot.csv
- tsm_prediction_quality_checks.csv
- tsm_prediction_report.md
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tsm_core.audit import build_policy_audit_rows
from tsm_core.config import apply_config_defaults, load_run_config
from tsm_core.decision_schema import DecisionPermission, FinalTradeDecision, PredictionUseStatus
from tsm_core.metrics import (
    adaptive_calibration_bins as adaptive_calibration_bins_core,
    calibration_binning_primary as calibration_binning_primary_core,
)
from tsm_core.schemas import build_schema_quality_checks, is_future_leakage_feature
from tsm_core.splits import PurgedEventTimeSplit

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.impute import SimpleImputer
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - exercised by runtime quality check
    SKLEARN_IMPORT_ERROR = exc
    ConvergenceWarning = Warning


EVENT_ACTIONS = {"ENTRY_ALLOWED", "HOLD_OR_WAIT_TRIGGER", "WATCHLIST_PULLBACK_ONLY"}
NEAR_MISS_RESEARCH_TIER = "near_miss_no_trade"
ENTRY_RESEARCH_TIERS = {
    "decision_trade_ready",
    "relaxed_trigger_score65",
    "relaxed_trigger_score60",
    "setup_context_score65",
    NEAR_MISS_RESEARCH_TIER,
}
RISK_RESEARCH_TIER = "risk_blocked_research"
MODEL_TRAINING_MIN_TSM_20D_LABELS = 750
HORIZONS = (5, 10, 20, 40, 60, 120)
TRADING_DAYS = 252
PRIOR_STRENGTH = 20.0
MIN_DECISION_OOS_EVENTS = 100
MIN_SELECTED_OOS_EVENTS = 50
MIN_SELECTED_EVENTS_PER_FOLD = 10
MIN_TREE_DECISION_OOS_EVENTS = 500
MIN_FOLD_TRAIN_EVENTS = 120
MIN_FOLD_VALIDATION_EVENTS = 40
MIN_FOLD_TEST_EVENTS = 40
MIN_CALIBRATION_CLASS_COUNT = 10
MIN_THRESHOLD_SELECTED_EVENTS = 20
DECISION_ECE_THRESHOLD = 0.10
MIN_POSITIVE_EXPECTANCY_FOLDS = 4
MIN_CALIBRATION_BIN_N = 30
MAX_THRESHOLD_IQR = 0.10
MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT = 0.0
MIN_EXPECTANCY_IMPROVEMENT_PCT = 0.0
MIN_SPARSE_TRADE_READY_DIAGNOSTIC_EVENTS = 60
MIN_SPARSE_TRADE_READY_TRAIN_EVENTS = 25
MIN_SPARSE_TRADE_READY_VALIDATION_EVENTS = 15
MIN_SPARSE_TRADE_READY_TEST_EVENTS = 20
SPARSE_TRADE_READY_TEST_FRACTION = 0.35
SPARSE_DIAGNOSTIC_SCOPES = {"trade_ready_entry", "trigger_all"}
MISSING_FEATURE_THRESHOLD = 0.30
CORRELATION_FEATURE_THRESHOLD = 0.90
MAX_NUMERIC_FEATURES = 25
DECISION_MODELS = {"empirical_bayes_group_rate", "base_rate_by_trigger_regime", "score_logistic", "elastic_net_logistic"}
RESEARCH_ONLY_MODELS = {"logistic_balanced", "random_forest_fixed", "hist_gradient_boosting_fixed"}

OPTIONAL_NEWS_SIGNAL_DEFAULTS = {
    "news_penalty_event": False,
    "news_event_count_1d": 0,
    "news_event_count_3d": 0,
    "news_sentiment_score_1d": 0.0,
    "news_match_confidence_score": 0.0,
    "news_source_count": 0,
    "news_primary_cause_type": "NO_NEWS",
    "news_match_confidence": "NO_MATCH",
    "news_coverage_status": "NO_COVERAGE",
}

NUMERIC_FEATURES = [
    "close_change_pct",
    "open_gap_pct",
    "open_to_close_pct",
    "intraday_range_pct_prev_close",
    "return_20d",
    "return_60d",
    "return_126d",
    "return_252d",
    "momentum_12m_ex_1m",
    "dist_close_sma_20_pct",
    "dist_close_sma_50_pct",
    "dist_close_sma_200_pct",
    "rsi_14",
    "atr_14_pct",
    "vol_20d_ann",
    "vol_63d_ann",
    "vol_252d_ann",
    "drawdown_from_ath",
    "relative_return_vs_spy_60d",
    "relative_return_vs_smh_60d",
    "relative_return_vs_qqq_60d",
    "beta_vs_spy_252d",
    "beta_vs_smh_252d",
    "beta_vs_qqq_252d",
    "score_trend",
    "score_momentum",
    "score_relative_strength",
    "score_low_vol",
    "score_liquidity",
    "penalty_event",
    "news_penalty_event",
    "score_price_algo_total",
    "news_event_count_1d",
    "news_event_count_3d",
    "news_sentiment_score_1d",
    "news_match_confidence_score",
    "news_source_count",
    "risk_pct_2atr",
    "position_weight_if_0_5pct_account_risk",
    "position_weight_if_1pct_account_risk",
    "final_recommended_max_weight",
    "volume_ratio_20",
    "atr_percentile_252d",
    "vol_compression_20_252",
    "vol_expansion_20_63",
    "breakout_quality_20",
    "breakout_quality_60",
    "volume_confirmation_20",
    "score_vol_interaction",
    "dist50_slope_interaction",
    "trend_slope_alignment",
    "risk_weight_score_interaction",
    "hist_success_rate_trigger_trend",
    "hist_success_rate_gate",
    "hist_success_rate_universe",
    "hist_mean_return_gate",
    "hist_stop_rate_gate",
    "hist_news_category_count_20d",
    "hist_news_category_success_rate_20d",
    "hist_news_category_mean_return_20d",
    "hist_news_category_count_60d",
    "hist_news_category_success_rate_60d",
    "hist_news_category_mean_return_60d",
]

ENRICHED_NUMERIC_FEATURES = [
    "atr_5_pct",
    "atr_10_pct",
    "atr_20_pct",
    "atr_50_pct",
    "atr_63_pct",
    "atr_100_pct",
    "atr_252_pct",
    "downside_vol_20d_ann",
    "downside_vol_60d_ann",
    "downside_vol_126d_ann",
    "downside_vol_252d_ann",
    "parkinson_vol_20d_ann",
    "parkinson_vol_60d_ann",
    "parkinson_vol_120d_ann",
    "parkinson_vol_252d_ann",
    "garman_klass_vol_20d_ann",
    "garman_klass_vol_60d_ann",
    "garman_klass_vol_120d_ann",
    "garman_klass_vol_252d_ann",
    "sma_20_slope_5d_pct",
    "sma_50_slope_5d_pct",
    "sma_100_slope_5d_pct",
    "sma_200_slope_5d_pct",
    "boll_width_20_pct",
    "boll_z_20",
    "macd_12_26",
    "macd_signal_9",
    "macd_hist",
    "volume_ratio_5",
    "volume_ratio_10",
    "volume_ratio_50",
    "dollar_volume_ratio_5",
    "dollar_volume_ratio_10",
    "dollar_volume_ratio_20",
    "dollar_volume_ratio_50",
    "amihud_20d",
    "amihud_60d",
    "amihud_252d",
    "drawdown_from_20d_high",
    "drawdown_from_60d_high",
    "drawdown_from_120d_high",
    "drawdown_from_252d_high",
    "max_drawdown_20d",
    "max_drawdown_60d",
    "max_drawdown_120d",
    "max_drawdown_252d",
    "close_change_pct_z_20d",
    "close_change_pct_z_60d",
    "close_change_pct_z_252d",
    "intraday_range_pct_prev_close_z_20d",
    "intraday_range_pct_prev_close_z_60d",
    "intraday_range_pct_prev_close_z_252d",
    "volume_z_20d",
    "volume_z_60d",
    "volume_z_252d",
    "dollar_volume_z_20d",
    "dollar_volume_z_60d",
    "dollar_volume_z_252d",
    "atr_14_pct_z_20d",
    "atr_14_pct_z_60d",
    "atr_14_pct_z_252d",
    "vol20_rank_252d",
    "up_3pct_count_252d",
    "up_5pct_count_252d",
    "down_3pct_count_252d",
    "down_5pct_count_252d",
    "relative_return_vs_spy_20d",
    "relative_return_vs_spy_120d",
    "relative_return_vs_spy_252d",
    "relative_return_vs_smh_20d",
    "relative_return_vs_smh_120d",
    "relative_return_vs_smh_252d",
    "relative_return_vs_qqq_20d",
    "relative_return_vs_qqq_120d",
    "relative_return_vs_qqq_252d",
    "beta_vs_spy_60d",
    "beta_vs_spy_126d",
    "beta_vs_smh_60d",
    "beta_vs_smh_126d",
    "beta_vs_qqq_60d",
    "beta_vs_qqq_126d",
]

NUMERIC_FEATURES = list(dict.fromkeys([*NUMERIC_FEATURES, *ENRICHED_NUMERIC_FEATURES]))

BOOL_FEATURES = [
    "algo_trend_up_clean",
    "algo_trend_up_loose",
    "algo_vol_high",
    "algo_vol_extreme",
    "algo_breakout20_clean",
    "algo_breakout60_clean",
    "algo_pullback_to_50_bounce",
    "algo_overextended_highvol",
    "algo_deep_downtrend_avoid",
    "algo_event_shock_day",
    "algo_relative_strength",
    "trigger_breakout_20d",
    "trigger_breakout_60d",
    "trigger_pullback_50d",
    "trigger_deep_dd_recovery",
]

CATEGORICAL_FEATURES = [
    "trend_regime",
    "vol_regime",
    "drawdown_bucket",
    "momentum_signal",
    "entry_trigger",
    "trade_action",
    "risk_state",
    "entry_gate_status",
    "candidate_scope",
    "prediction_universe",
    "candidate_tier",
    "news_primary_cause_type",
    "news_match_confidence",
    "news_coverage_status",
]

EXTERNAL_NUMERIC_FEATURES = [
    "tsmc_monthly_revenue_ntd_m",
    "tsmc_monthly_revenue_yoy_pct",
    "tsmc_monthly_revenue_mom_pct",
    "tsmc_revenue_yoy_3m_avg_pct",
    "tsmc_revenue_12m_cumulative_yoy_pct",
    "days_since_tsmc_revenue_release",
    "days_since_tsmc_earnings",
    "market_smh_return_20d",
    "market_smh_return_60d",
    "market_soxx_return_20d",
    "market_soxx_return_60d",
    "market_qqq_return_20d",
    "market_qqq_return_60d",
    "market_spy_return_20d",
    "market_spy_return_60d",
    "external_relative_return_vs_qqq_20d",
    "external_relative_return_vs_spy_20d",
    "universe_external_above_sma50_ratio",
    "universe_external_above_sma200_ratio",
    "peer_return_20d_rank",
    "peer_return_60d_rank",
    "peer_return_126d_rank",
    "peer_group_return_20d_rank",
    "fx_usdtwd_return_20d",
    "fx_usdtwd_return_60d",
]

EXTERNAL_BOOL_FEATURES = [
    "tsmc_earnings_pre_5d_window",
    "tsmc_earnings_post_5d_window",
    "tsmc_earnings_event_day",
    "fx_data_available",
]

EXTERNAL_CATEGORICAL_FEATURES = [
    "external_feature_status",
    "tsmc_revenue_source",
]

NUMERIC_FEATURES = list(dict.fromkeys([*NUMERIC_FEATURES, *EXTERNAL_NUMERIC_FEATURES]))
BOOL_FEATURES = list(dict.fromkeys([*BOOL_FEATURES, *EXTERNAL_BOOL_FEATURES]))
CATEGORICAL_FEATURES = list(dict.fromkeys([*CATEGORICAL_FEATURES, *EXTERNAL_CATEGORICAL_FEATURES]))

FORBIDDEN_FEATURE_PATTERNS = ("fwd_return", "forward", "future", "label_", "exit_", "next_", "actual_return", "net_return", "gross_return", "r_multiple")


@dataclass(frozen=True)
class ModelResult:
    name: str
    probabilities: np.ndarray
    fitted_model: object | None
    fallback_reason: str


def pct(x: float) -> float:
    return float(x) * 100.0 if pd.notna(x) else np.nan


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return strip_bom_columns(pd.read_csv(path, **kwargs))


def normalize_optional_signal_columns(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    for col, default in OPTIONAL_NEWS_SIGNAL_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
    return out


def load_external_features(path: Optional[Path], symbol: str = "TSM") -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    external = read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    require_columns(external, ["symbol", "date"], str(path))
    external["symbol"] = external["symbol"].astype(str).str.upper()
    symbol_upper = str(symbol).upper()
    external = external[external["symbol"].eq(symbol_upper)].copy()
    if external.empty:
        return pd.DataFrame()
    return external.drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)


def merge_external_features(signals: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    if external.empty:
        return signals.copy()
    out = signals.copy()
    merge_cols = [c for c in external.columns if c != "symbol"]
    external_for_merge = external[merge_cols].copy()
    overlap = [c for c in external_for_merge.columns if c in out.columns and c != "date"]
    if overlap:
        external_for_merge = external_for_merge.rename(columns={c: f"external_{c}" for c in overlap})
    return out.merge(external_for_merge, on="date", how="left")


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


def valid_price(value) -> bool:
    return pd.notna(value) and float(value) > 0


def bounded_probability(value: float) -> float:
    if pd.isna(value):
        return np.nan
    return float(np.clip(value, 1e-6, 1.0 - 1e-6))


def rolling_percentile_current(s: pd.Series) -> float:
    values = pd.to_numeric(s, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.rank(pct=True).iloc[-1])


def wilson_interval(p: float, n: int, z: float = 1.2815515655446004) -> Tuple[float, float]:
    if pd.isna(p) or n <= 0:
        return np.nan, np.nan
    p = bounded_probability(p)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denom
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def actionable_entry_candidate(row: pd.Series) -> bool:
    return str(row.get("entry_trigger", "NONE")) != "NONE"


def above_200d(row: pd.Series) -> bool:
    close = as_float(row.get("close"))
    sma_200 = as_float(row.get("sma_200"))
    return pd.notna(close) and pd.notna(sma_200) and close >= sma_200


def trade_ready_entry_candidate(row: pd.Series) -> bool:
    return (
        str(row.get("entry_trigger", "NONE")) != "NONE"
        and str(row.get("trade_action", "NO_TRADE")) == "ENTRY_ALLOWED"
        and not to_bool(row.get("algo_vol_extreme"))
        and above_200d(row)
    )


def risk_blocked_research_candidate(row: pd.Series) -> bool:
    trigger = str(row.get("entry_trigger", "NONE"))
    action = str(row.get("trade_action", "NO_TRADE"))
    return trigger != "NONE" and (
        not above_200d(row)
        or to_bool(row.get("algo_vol_extreme"))
        or to_bool(row.get("algo_overextended_highvol"))
        or to_bool(row.get("algo_deep_downtrend_avoid"))
        or trigger == "DEEP_DD_RECOVERY"
        or action == "RESEARCH_ONLY_DEEP_DD"
    )


def near_miss_research_candidate(row: pd.Series) -> bool:
    trigger = str(row.get("entry_trigger", "NONE"))
    action = str(row.get("trade_action", "NO_TRADE"))
    score = as_float(row.get("score_price_algo_total"), default=-np.inf)
    if trigger != "NONE":
        return False
    if to_bool(row.get("algo_vol_extreme")):
        return False
    if not above_200d(row):
        return False
    loose_trend = to_bool(row.get("algo_trend_up_loose")) or to_bool(row.get("algo_trend_up_clean"))
    context_wait = action in {"HOLD_OR_WAIT_TRIGGER", "WATCHLIST_PULLBACK_ONLY", "NO_TRADE"}
    return bool(context_wait and loose_trend and 55 <= score < 65)


def candidate_tier(row: pd.Series) -> str:
    if trade_ready_entry_candidate(row):
        return "decision_trade_ready"
    trigger = str(row.get("entry_trigger", "NONE"))
    action = str(row.get("trade_action", "NO_TRADE"))
    score = as_float(row.get("score_price_algo_total"), default=-np.inf)
    if trigger != "NONE" and score >= 65 and not to_bool(row.get("algo_vol_extreme")) and above_200d(row):
        return "relaxed_trigger_score65"
    if trigger != "NONE" and score >= 60 and not to_bool(row.get("algo_vol_extreme")) and above_200d(row):
        return "relaxed_trigger_score60"
    if (
        trigger == "NONE"
        and action in {"HOLD_OR_WAIT_TRIGGER", "WATCHLIST_PULLBACK_ONLY"}
        and score >= 65
        and to_bool(row.get("algo_trend_up_loose"))
        and not to_bool(row.get("algo_vol_extreme"))
    ):
        return "setup_context_score65"
    if near_miss_research_candidate(row):
        return NEAR_MISS_RESEARCH_TIER
    if risk_blocked_research_candidate(row):
        return RISK_RESEARCH_TIER
    return "none"


def entry_research_candidate(row: pd.Series) -> bool:
    tier = row.get("candidate_tier")
    if pd.isna(tier):
        tier = candidate_tier(row)
    tier = str(tier)
    return tier in ENTRY_RESEARCH_TIERS


def risk_research_candidate(row: pd.Series) -> bool:
    tier = row.get("candidate_tier")
    if pd.isna(tier):
        tier = candidate_tier(row)
    tier = str(tier)
    return tier == RISK_RESEARCH_TIER


def event_candidate(row: pd.Series) -> bool:
    return (
        str(row.get("entry_trigger", "NONE")) != "NONE"
        or str(row.get("trade_action", "NO_TRADE")) in EVENT_ACTIONS
        or entry_research_candidate(row)
        or risk_research_candidate(row)
    )


def candidate_scope(row: pd.Series) -> str:
    trigger = str(row.get("entry_trigger", "NONE"))
    action = str(row.get("trade_action", "NO_TRADE"))
    if trigger != "NONE" and action == "ENTRY_ALLOWED":
        return "ACTIONABLE_ENTRY_ALLOWED"
    if trigger != "NONE":
        return "ACTIONABLE_ENTRY_FILTERED"
    if action == "HOLD_OR_WAIT_TRIGGER":
        return "CONTEXT_HIGH_SCORE_WAIT"
    if action == "WATCHLIST_PULLBACK_ONLY":
        return "CONTEXT_WATCHLIST"
    return "NOT_CANDIDATE"


def entry_gate_status(row: pd.Series) -> str:
    trigger = str(row.get("entry_trigger", "NONE"))
    action = str(row.get("trade_action", "NO_TRADE"))
    if trigger == "NONE":
        if action == "HOLD_OR_WAIT_TRIGGER":
            return "NO_TRIGGER_HIGH_SCORE_WAIT"
        if action == "WATCHLIST_PULLBACK_ONLY":
            return "NO_TRIGGER_WATCHLIST"
        return "NO_ENTRY_TRIGGER"
    if trade_ready_entry_candidate(row):
        return "TRADE_READY"
    if trigger == "DEEP_DD_RECOVERY" or action == "RESEARCH_ONLY_DEEP_DD":
        return "RESEARCH_ONLY_DEEP_DD"
    if as_float(row.get("close")) < as_float(row.get("sma_200")):
        return "BLOCKED_BELOW_200D"
    if to_bool(row.get("algo_vol_extreme")):
        return "BLOCKED_EXTREME_VOL"
    if to_bool(row.get("algo_overextended_highvol")):
        return "BLOCKED_OVEREXTENDED_HIGHVOL"
    if to_bool(row.get("algo_deep_downtrend_avoid")):
        return "BLOCKED_DEEP_DOWNTREND"
    if as_float(row.get("score_price_algo_total")) < 75:
        return "BLOCKED_SCORE_BELOW_75"
    return f"BLOCKED_{action}"


def prediction_universe(row: pd.Series) -> str:
    if trade_ready_entry_candidate(row):
        return "trade_ready_entry"
    if entry_research_candidate(row):
        return "entry_research"
    if actionable_entry_candidate(row):
        return "trigger_all"
    if event_candidate(row):
        return "context_all"
    return "none"


def drawdown_bucket(value) -> str:
    dd = as_float(value)
    if pd.isna(dd):
        return "dd_unknown"
    if dd >= -0.05:
        return "dd_0_5"
    if dd >= -0.10:
        return "dd_5_10"
    if dd >= -0.20:
        return "dd_10_20"
    if dd >= -0.35:
        return "dd_20_35"
    return "dd_gt_35"


def usable_numeric_columns(frame: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    cols: List[str] = []
    for col in candidates:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().any():
            cols.append(col)
    return list(dict.fromkeys(cols))


def usable_model_columns(frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = usable_numeric_columns(frame, NUMERIC_FEATURES)
    bool_cols = [c for c in BOOL_FEATURES if c in frame.columns]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in frame.columns]
    return list(dict.fromkeys([*numeric_cols, *bool_cols])), categorical_cols


def add_daily_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr_percentile_252d"] = pd.to_numeric(df["atr_14_pct"], errors="coerce").rolling(252, min_periods=60).apply(rolling_percentile_current, raw=False)
    df["vol_compression_20_252"] = pd.to_numeric(df["vol_20d_ann"], errors="coerce") / pd.to_numeric(df["vol_252d_ann"], errors="coerce").replace(0, np.nan)
    df["vol_expansion_20_63"] = pd.to_numeric(df["vol_20d_ann"], errors="coerce") / pd.to_numeric(df["vol_63d_ann"], errors="coerce").replace(0, np.nan)
    prev_20_high = pd.to_numeric(df["high"], errors="coerce").shift(1).rolling(20, min_periods=10).max()
    prev_60_high = pd.to_numeric(df["high"], errors="coerce").shift(1).rolling(60, min_periods=30).max()
    close = pd.to_numeric(df["close"], errors="coerce")
    df["breakout_quality_20"] = close / prev_20_high - 1.0
    df["breakout_quality_60"] = close / prev_60_high - 1.0
    df["volume_confirmation_20"] = pd.to_numeric(df["volume_ratio_20"], errors="coerce") * pd.to_numeric(df["close_change_pct"], errors="coerce")
    df["score_vol_interaction"] = pd.to_numeric(df["score_price_algo_total"], errors="coerce") * (1.0 - pd.to_numeric(df["atr_percentile_252d"], errors="coerce"))
    slope_50 = pd.to_numeric(df.get("sma_50_slope_5d_pct", np.nan), errors="coerce")
    slope_200 = pd.to_numeric(df.get("sma_200_slope_5d_pct", np.nan), errors="coerce")
    df["dist50_slope_interaction"] = pd.to_numeric(df["dist_close_sma_50_pct"], errors="coerce") * slope_50
    df["trend_slope_alignment"] = slope_50 + slope_200
    df["risk_weight_score_interaction"] = pd.to_numeric(df["final_recommended_max_weight"], errors="coerce") * pd.to_numeric(df["score_price_algo_total"], errors="coerce")
    df = add_news_category_history_features(df)
    return df


def add_news_category_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "news_primary_cause_type" not in out.columns:
        for horizon in HORIZONS:
            out[f"hist_news_category_count_{horizon}d"] = 0.0
            out[f"hist_news_category_success_rate_{horizon}d"] = np.nan
            out[f"hist_news_category_mean_return_{horizon}d"] = np.nan
        return out

    cause = out["news_primary_cause_type"].fillna("").astype(str).str.strip()
    valid_cause = cause.ne("") & ~cause.str.startswith("NO_")
    close = pd.to_numeric(out["close"], errors="coerce")
    for horizon in HORIZONS:
        counts: list[float] = []
        success_rates: list[float] = []
        mean_returns: list[float] = []
        history: dict[str, list[float]] = {}
        fwd_return = close.shift(-horizon) / close - 1
        for idx, category in enumerate(cause):
            bucket = history.get(category, []) if valid_cause.iloc[idx] else []
            counts.append(float(len(bucket)))
            success_rates.append(float(np.mean([value > 0 for value in bucket])) if bucket else np.nan)
            mean_returns.append(float(np.mean(bucket)) if bucket else np.nan)
            known_idx = idx - horizon - 1
            if known_idx >= 0:
                known_category = cause.iloc[known_idx]
                known_return = fwd_return.iloc[known_idx]
                if valid_cause.iloc[known_idx] and pd.notna(known_return):
                    history.setdefault(known_category, []).append(float(known_return))
        out[f"hist_news_category_count_{horizon}d"] = counts
        out[f"hist_news_category_success_rate_{horizon}d"] = success_rates
        out[f"hist_news_category_mean_return_{horizon}d"] = mean_returns
    return out


def cost_rate(commission_bps: float, slippage_bps: float) -> float:
    return (commission_bps + slippage_bps) / 10000.0


def load_inputs(
    signals_path: Path,
    risk_path: Path,
    trade_log_path: Path,
    enriched_path: Optional[Path] = None,
    external_features_path: Optional[Path] = None,
    symbol: str = "TSM",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    signals = read_csv(signals_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    signals = normalize_optional_signal_columns(signals)
    required_signals = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "sma_200",
        "atr_14",
        "entry_trigger",
        "trade_action",
        "dollar_volume",
        "dollar_volume_ma_20",
        "close_change_pct",
        "open_gap_pct",
        "open_to_close_pct",
        "intraday_range_pct_prev_close",
        "return_20d",
        "return_60d",
        "return_126d",
        "return_252d",
        "momentum_12m_ex_1m",
        "dist_close_sma_20_pct",
        "dist_close_sma_50_pct",
        "dist_close_sma_200_pct",
        "rsi_14",
        "atr_14_pct",
        "vol_20d_ann",
        "vol_63d_ann",
        "vol_252d_ann",
        "drawdown_from_ath",
        "score_trend",
        "score_momentum",
        "score_relative_strength",
        "score_low_vol",
        "score_liquidity",
        "penalty_event",
        "news_penalty_event",
        "score_price_algo_total",
        "news_event_count_1d",
        "news_event_count_3d",
        "news_sentiment_score_1d",
        "news_match_confidence_score",
        "news_source_count",
        "risk_pct_2atr",
        "position_weight_if_0_5pct_account_risk",
        "position_weight_if_1pct_account_risk",
        *[c for c in BOOL_FEATURES if c not in EXTERNAL_BOOL_FEATURES],
        *[
            c
            for c in CATEGORICAL_FEATURES
            if c not in {"risk_state", "entry_gate_status", "candidate_scope", "prediction_universe", "drawdown_bucket", "candidate_tier", *EXTERNAL_CATEGORICAL_FEATURES}
        ],
    ]
    require_columns(signals, sorted(set(required_signals)), str(signals_path))
    if signals["date"].duplicated().any():
        raise ValueError("signals has duplicated dates")
    if not signals["date"].is_monotonic_increasing:
        raise ValueError("signals dates must be sorted ascending")

    for col in BOOL_FEATURES:
        if col in signals.columns:
            signals[col] = signals[col].map(to_bool)

    risk = read_csv(risk_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    require_columns(risk, ["date", "risk_state", "final_recommended_max_weight"], str(risk_path))
    risk_extra = risk[["date", "risk_state", "final_recommended_max_weight"]].copy()
    df = signals.merge(risk_extra, on="date", how="left")
    df["risk_state"] = df["risk_state"].fillna("UNKNOWN")
    df["final_recommended_max_weight"] = pd.to_numeric(df["final_recommended_max_weight"], errors="coerce")
    df["volume_ratio_20"] = pd.to_numeric(df["dollar_volume"], errors="coerce") / pd.to_numeric(df["dollar_volume_ma_20"], errors="coerce")
    df["signal_idx"] = np.arange(len(df))
    if enriched_path and enriched_path.exists():
        enriched = read_csv(enriched_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        extra_cols = [c for c in ENRICHED_NUMERIC_FEATURES if c in enriched.columns and c not in df.columns]
        if extra_cols:
            df = df.merge(enriched[["date", *extra_cols]], on="date", how="left")
    external = load_external_features(external_features_path, symbol=symbol)
    df = merge_external_features(df, external)
    for col in EXTERNAL_BOOL_FEATURES:
        if col in df.columns:
            df[col] = df[col].map(to_bool)
    df = add_daily_prediction_features(df)
    df["is_actionable_entry_candidate"] = df.apply(actionable_entry_candidate, axis=1)
    df["is_trade_ready_entry_candidate"] = df.apply(trade_ready_entry_candidate, axis=1)
    df["candidate_tier"] = df.apply(candidate_tier, axis=1)
    df["is_entry_research_candidate"] = df["candidate_tier"].isin(ENTRY_RESEARCH_TIERS)
    df["is_risk_research_candidate"] = df["candidate_tier"].eq(RISK_RESEARCH_TIER)
    df["is_model_training_candidate"] = df["is_entry_research_candidate"]
    df["is_decision_entry_candidate"] = df["is_trade_ready_entry_candidate"]
    df["is_event_candidate"] = df.apply(event_candidate, axis=1)
    df["entry_gate_status"] = df.apply(entry_gate_status, axis=1)
    df["candidate_scope"] = df.apply(candidate_scope, axis=1)
    df["prediction_universe"] = df.apply(prediction_universe, axis=1)
    df["drawdown_bucket"] = df["drawdown_from_ath"].map(drawdown_bucket)

    if trade_log_path.exists():
        trades = read_csv(trade_log_path)
    else:
        trades = pd.DataFrame()
    return df, trades


def label_event_horizon(
    df: pd.DataFrame,
    idx: int,
    horizon: int,
    commission_bps: float,
    slippage_bps: float,
    stop_multiple: float,
) -> Dict:
    def unavailable(status: str) -> Dict:
        return {
            f"label_status_{horizon}d": status,
            f"label_success_{horizon}d": np.nan,
            f"label_stop_survival_{horizon}d": np.nan,
            f"label_hit_1r_before_stop_{horizon}d": np.nan,
            f"label_hit_2r_before_stop_{horizon}d": np.nan,
            f"label_positive_return_{horizon}d": np.nan,
            f"label_expected_r_{horizon}d": np.nan,
            f"label_mfe_r_{horizon}d": np.nan,
            f"label_mae_r_{horizon}d": np.nan,
            f"label_time_to_1r_{horizon}d": np.nan,
            f"label_time_to_2r_{horizon}d": np.nan,
            f"label_time_to_stop_{horizon}d": np.nan,
            f"label_ambiguous_stop_1r_same_day_{horizon}d": np.nan,
            f"label_ambiguous_stop_2r_same_day_{horizon}d": np.nan,
            f"label_gap_through_stop_{horizon}d": np.nan,
            f"label_entry_gap_pct_{horizon}d": np.nan,
            f"label_first_touch_type_{horizon}d": status,
            f"label_time_to_first_touch_{horizon}d": np.nan,
            f"label_mfe_before_stop_{horizon}d": np.nan,
            f"label_mae_before_profit_{horizon}d": np.nan,
            f"label_event_regime_at_entry_{horizon}d": status,
            f"label_net_return_pct_{horizon}d": np.nan,
            f"label_gross_return_pct_{horizon}d": np.nan,
            f"label_exit_reason_{horizon}d": status,
            f"label_exit_type_clean_{horizon}d": status,
            f"label_entry_date_{horizon}d": "",
            f"label_exit_date_{horizon}d": "",
            f"label_entry_price_{horizon}d": np.nan,
            f"label_exit_price_{horizon}d": np.nan,
            f"label_stop_price_{horizon}d": np.nan,
            f"label_1r_price_{horizon}d": np.nan,
            f"label_2r_price_{horizon}d": np.nan,
            f"label_holding_trading_days_{horizon}d": np.nan,
        }

    row = df.iloc[idx]
    if not bool(row["is_event_candidate"]):
        return unavailable("NOT_EVENT")

    entry_idx = idx + 1
    horizon_exit_idx = idx + horizon
    if entry_idx >= len(df):
        status = "UNAVAILABLE_NEXT_OPEN"
    elif horizon_exit_idx >= len(df):
        status = "UNAVAILABLE_FUTURE_WINDOW"
    else:
        status = "LABELED"
    if status != "LABELED":
        return unavailable(status)

    entry = df.iloc[entry_idx]
    entry_price = as_float(entry["open"])
    atr = as_float(row["atr_14"])
    if not valid_price(entry_price) or atr <= 0:
        return unavailable("INVALID_ENTRY_DATA")

    cr = cost_rate(commission_bps, slippage_bps)
    stop_price = entry_price - stop_multiple * atr
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        return unavailable("INVALID_RISK_DISTANCE")
    target_1r = entry_price + stop_multiple * atr
    target_2r = entry_price + 2.0 * stop_multiple * atr
    exit_idx = horizon_exit_idx
    exit_price = as_float(df.iloc[horizon_exit_idx]["close"])
    exit_reason = f"HORIZON_{horizon}D"
    stop_hit = False
    hit_1r = False
    hit_2r = False
    mfe = 0.0
    mae = 0.0
    ambiguous_stop_1r_same_day = False
    ambiguous_stop_2r_same_day = False
    gap_through_stop = False
    first_touch_type = "NONE"
    time_to_first_touch = np.nan
    mfe_before_stop = np.nan
    mae_before_profit = np.nan
    time_to_1r = np.nan
    time_to_2r = np.nan
    time_to_stop = np.nan
    signal_close = as_float(row.get("close"))
    entry_gap_pct = entry_price / signal_close - 1.0 if valid_price(signal_close) else np.nan
    entry_trend_regime = str(entry.get("trend_regime", row.get("trend_regime", "UNKNOWN")))
    entry_vol_regime = str(entry.get("vol_regime", row.get("vol_regime", "UNKNOWN")))
    event_regime_at_entry = f"{entry_trend_regime}|{entry_vol_regime}"

    for j in range(entry_idx, horizon_exit_idx + 1):
        low = as_float(df.iloc[j]["low"])
        high = as_float(df.iloc[j]["high"])
        open_price = as_float(df.iloc[j]["open"])
        prior_mfe = mfe
        prior_mae = mae
        stop_touched = valid_price(low) and low <= stop_price
        hit_1r_touched = valid_price(high) and high >= target_1r
        hit_2r_touched = valid_price(high) and high >= target_2r
        if stop_touched and hit_1r_touched:
            ambiguous_stop_1r_same_day = True
        if stop_touched and hit_2r_touched:
            ambiguous_stop_2r_same_day = True
        if pd.isna(time_to_first_touch) and (stop_touched or hit_1r_touched or hit_2r_touched):
            time_to_first_touch = j - entry_idx
            if stop_touched and hit_2r_touched:
                first_touch_type = "AMBIGUOUS_STOP_2R_SAME_DAY"
            elif stop_touched and hit_1r_touched:
                first_touch_type = "AMBIGUOUS_STOP_1R_SAME_DAY"
            elif stop_touched:
                first_touch_type = "STOP"
            elif hit_2r_touched:
                first_touch_type = "TARGET_2R"
            else:
                first_touch_type = "TARGET_1R"
        if stop_touched and pd.isna(mfe_before_stop):
            mfe_before_stop = prior_mfe
        if (hit_1r_touched or hit_2r_touched) and pd.isna(mae_before_profit):
            mae_before_profit = prior_mae
        if valid_price(high):
            mfe = max(mfe, (high - entry_price) / risk_per_share)
        if valid_price(low):
            mae = min(mae, (low - entry_price) / risk_per_share)
        if stop_touched:
            exit_idx = j
            if valid_price(open_price) and open_price <= stop_price:
                gap_through_stop = True
            exit_price = open_price if valid_price(open_price) and open_price <= stop_price else stop_price
            exit_reason = f"ATR_STOP_{stop_multiple:g}X"
            stop_hit = True
            time_to_stop = j - entry_idx
            break
        if hit_1r_touched:
            hit_1r = True
            if pd.isna(time_to_1r):
                time_to_1r = j - entry_idx
        if hit_2r_touched:
            hit_2r = True
            if pd.isna(time_to_2r):
                time_to_2r = j - entry_idx

    if not valid_price(exit_price):
        return unavailable("INVALID_EXIT_DATA")

    entry_after_cost = entry_price * (1.0 + cr)
    exit_after_cost = exit_price * (1.0 - cr)
    gross_return = exit_price / entry_price - 1.0
    net_return = exit_after_cost / entry_after_cost - 1.0
    expected_r = (exit_after_cost - entry_after_cost) / risk_per_share
    success = (not stop_hit) and net_return > 0
    positive_return = net_return > 0
    if stop_hit:
        exit_type_clean = "STOP"
    elif hit_2r:
        exit_type_clean = "TARGET_2R_OR_TIME"
    elif hit_1r:
        exit_type_clean = "TARGET_1R_OR_TIME"
    else:
        exit_type_clean = "TIME"

    return {
        f"label_status_{horizon}d": "LABELED",
        f"label_success_{horizon}d": int(success),
        f"label_stop_survival_{horizon}d": int(not stop_hit),
        f"label_hit_1r_before_stop_{horizon}d": int(hit_1r),
        f"label_hit_2r_before_stop_{horizon}d": int(hit_2r),
        f"label_positive_return_{horizon}d": int(positive_return),
        f"label_expected_r_{horizon}d": expected_r,
        f"label_mfe_r_{horizon}d": mfe,
        f"label_mae_r_{horizon}d": mae,
        f"label_time_to_1r_{horizon}d": time_to_1r,
        f"label_time_to_2r_{horizon}d": time_to_2r,
        f"label_time_to_stop_{horizon}d": time_to_stop,
        f"label_ambiguous_stop_1r_same_day_{horizon}d": int(ambiguous_stop_1r_same_day),
        f"label_ambiguous_stop_2r_same_day_{horizon}d": int(ambiguous_stop_2r_same_day),
        f"label_gap_through_stop_{horizon}d": int(gap_through_stop),
        f"label_entry_gap_pct_{horizon}d": pct(entry_gap_pct),
        f"label_first_touch_type_{horizon}d": first_touch_type,
        f"label_time_to_first_touch_{horizon}d": time_to_first_touch,
        f"label_mfe_before_stop_{horizon}d": mfe_before_stop,
        f"label_mae_before_profit_{horizon}d": mae_before_profit,
        f"label_event_regime_at_entry_{horizon}d": event_regime_at_entry,
        f"label_net_return_pct_{horizon}d": pct(net_return),
        f"label_gross_return_pct_{horizon}d": pct(gross_return),
        f"label_exit_reason_{horizon}d": exit_reason,
        f"label_exit_type_clean_{horizon}d": exit_type_clean,
        f"label_entry_date_{horizon}d": df.iloc[entry_idx]["date"],
        f"label_exit_date_{horizon}d": df.iloc[exit_idx]["date"],
        f"label_entry_price_{horizon}d": entry_price,
        f"label_exit_price_{horizon}d": exit_price,
        f"label_stop_price_{horizon}d": stop_price,
        f"label_1r_price_{horizon}d": target_1r,
        f"label_2r_price_{horizon}d": target_2r,
        f"label_holding_trading_days_{horizon}d": exit_idx - entry_idx,
    }


def add_label_overlap_uniqueness(labels: pd.DataFrame) -> pd.DataFrame:
    out = labels.copy()
    for horizon in HORIZONS:
        status_col = f"label_status_{horizon}d"
        entry_col = f"label_entry_date_{horizon}d"
        exit_col = f"label_exit_date_{horizon}d"
        overlap_col = f"label_overlap_count_{horizon}d"
        weight_col = f"sample_uniqueness_weight_{horizon}d"
        out[overlap_col] = 0
        out[weight_col] = 0.0
        if not {status_col, entry_col, exit_col}.issubset(out.columns):
            continue
        labeled_mask = out[status_col].astype(str).eq("LABELED")
        if not labeled_mask.any():
            continue
        starts = pd.to_datetime(out.loc[labeled_mask, entry_col], errors="coerce")
        ends = pd.to_datetime(out.loc[labeled_mask, exit_col], errors="coerce")
        valid_mask = starts.notna() & ends.notna()
        if not valid_mask.any():
            continue
        valid_index = starts.loc[valid_mask].index
        start_values = starts.loc[valid_mask].astype("int64").to_numpy(dtype=np.int64)
        end_values = ends.loc[valid_mask].astype("int64").to_numpy(dtype=np.int64)
        ordered_starts = np.sort(start_values)
        ordered_ends = np.sort(end_values)
        active_by_end = np.searchsorted(ordered_starts, end_values, side="right")
        ended_before_start = np.searchsorted(ordered_ends, start_values, side="left")
        overlap_counts = np.maximum(active_by_end - ended_before_start, 1)
        out.loc[valid_index, overlap_col] = overlap_counts.astype(int)
        out.loc[valid_index, weight_col] = 1.0 / overlap_counts
        if horizon == 20:
            out["label_overlap_count"] = out[overlap_col]
            out["sample_uniqueness_weight"] = out[weight_col]
    return out


def build_label_dataset(
    df: pd.DataFrame,
    commission_bps: float,
    slippage_bps: float,
    stop_multiple: float,
) -> pd.DataFrame:
    base_cols = [
        "date",
        "signal_idx",
        "is_event_candidate",
        "is_actionable_entry_candidate",
        "is_trade_ready_entry_candidate",
        "is_entry_research_candidate",
        "is_risk_research_candidate",
        "is_model_training_candidate",
        "is_decision_entry_candidate",
        "candidate_tier",
        "entry_gate_status",
        "candidate_scope",
        "prediction_universe",
        "entry_trigger",
        "trade_action",
        "close",
        "atr_14",
        "score_price_algo_total",
        "risk_state",
        "news_primary_cause_type",
        "news_match_confidence",
        "news_coverage_status",
        "news_cause_summary",
    ]
    rows = []
    for idx, row in df.iterrows():
        out = {c: row.get(c) for c in base_cols}
        for horizon in HORIZONS:
            out.update(label_event_horizon(df, idx, horizon, commission_bps, slippage_bps, stop_multiple))
        rows.append(out)
    return add_label_overlap_uniqueness(pd.DataFrame(rows))


def build_feature_matrix(signals: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [*NUMERIC_FEATURES, *BOOL_FEATURES, *CATEGORICAL_FEATURES]
    meta = [
        "date",
        "signal_idx",
        "is_event_candidate",
        "is_actionable_entry_candidate",
        "is_trade_ready_entry_candidate",
        "is_entry_research_candidate",
        "is_risk_research_candidate",
        "is_model_training_candidate",
        "is_decision_entry_candidate",
        "candidate_tier",
        "entry_gate_status",
        "candidate_scope",
        "prediction_universe",
    ]
    available = [c for c in feature_cols if c in signals.columns and c not in set(meta)]
    label_cols = [c for c in labels.columns if c.startswith("label_")]
    features = signals[[*meta, *available]].copy()
    for col in BOOL_FEATURES:
        if col in features.columns:
            features[col] = features[col].map(lambda x: 1 if to_bool(x) else 0)
    merged = features.merge(labels[["date", "signal_idx", *label_cols]], on=["date", "signal_idx"], how="left")
    return merged[[*meta, *available, *label_cols]]


def build_candidate_scope_stats(labels: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    group_specs = [
        ("prediction_universe", "prediction_universe"),
        ("candidate_tier", "candidate_tier"),
        ("model_training_flag", "is_model_training_candidate"),
        ("decision_entry_flag", "is_decision_entry_candidate"),
        ("candidate_scope", "candidate_scope"),
        ("entry_gate_status", "entry_gate_status"),
        ("entry_trigger", "entry_trigger"),
        ("trade_action", "trade_action"),
    ]
    for horizon in HORIZONS:
        status_col = f"label_status_{horizon}d"
        success_col = f"label_success_{horizon}d"
        return_col = f"label_net_return_pct_{horizon}d"
        expected_r_col = f"label_expected_r_{horizon}d"
        mfe_col = f"label_mfe_r_{horizon}d"
        mae_col = f"label_mae_r_{horizon}d"
        hit_1r_col = f"label_hit_1r_before_stop_{horizon}d"
        hit_2r_col = f"label_hit_2r_before_stop_{horizon}d"
        exit_col = f"label_exit_reason_{horizon}d"
        for group_type, group_col in group_specs:
            if group_col not in labels.columns:
                continue
            for group_value, group in labels.groupby(group_col, dropna=False):
                labeled = group[group[status_col] == "LABELED"].copy()
                returns = pd.to_numeric(labeled[return_col], errors="coerce")
                expected_r = pd.to_numeric(labeled.get(expected_r_col), errors="coerce")
                rows.append(
                    {
                        "horizon_days": horizon,
                        "group_type": group_type,
                        "group_value": group_value,
                        "total_count": len(group),
                        "labeled_count": len(labeled),
                        "unavailable_count": int(group[status_col].astype(str).str.startswith("UNAVAILABLE").sum()),
                        "success_rate_pct": pct(labeled[success_col].astype(float).mean()) if len(labeled) else np.nan,
                        "mean_net_return_pct": float(returns.mean()) if len(labeled) else np.nan,
                        "median_net_return_pct": float(returns.median()) if len(labeled) else np.nan,
                        "mean_expected_r": float(expected_r.mean()) if len(labeled) else np.nan,
                        "median_expected_r": float(expected_r.median()) if len(labeled) else np.nan,
                        "mean_mfe_r": float(pd.to_numeric(labeled.get(mfe_col), errors="coerce").mean()) if len(labeled) else np.nan,
                        "mean_mae_r": float(pd.to_numeric(labeled.get(mae_col), errors="coerce").mean()) if len(labeled) else np.nan,
                        "hit_1r_rate_pct": pct(labeled[hit_1r_col].astype(float).mean()) if len(labeled) and hit_1r_col in labeled.columns else np.nan,
                        "hit_2r_rate_pct": pct(labeled[hit_2r_col].astype(float).mean()) if len(labeled) and hit_2r_col in labeled.columns else np.nan,
                        "profit_factor": profit_factor(returns) if len(labeled) else np.nan,
                        "atr_stop_rate_pct": pct(labeled[exit_col].astype(str).str.startswith("ATR_STOP").mean()) if len(labeled) else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(["horizon_days", "group_type", "labeled_count"], ascending=[True, True, False]).reset_index(drop=True)


def build_label_diagnostics(labels: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for horizon in HORIZONS:
        status_col = f"label_status_{horizon}d"
        exit_col = f"label_exit_reason_{horizon}d"
        labeled = labels[labels[status_col] == "LABELED"].copy()
        for universe, group in labeled.groupby("prediction_universe", dropna=False):
            rows.append(
                {
                    "horizon_days": horizon,
                    "prediction_universe": universe,
                    "labeled_count": len(group),
                    "success_rate_pct": pct(group[f"label_success_{horizon}d"].astype(float).mean()) if len(group) else np.nan,
                    "stop_survival_rate_pct": pct(group[f"label_stop_survival_{horizon}d"].astype(float).mean()) if len(group) else np.nan,
                    "positive_return_rate_pct": pct(group[f"label_positive_return_{horizon}d"].astype(float).mean()) if len(group) else np.nan,
                    "hit_1r_before_stop_rate_pct": pct(group[f"label_hit_1r_before_stop_{horizon}d"].astype(float).mean()) if len(group) else np.nan,
                    "hit_2r_before_stop_rate_pct": pct(group[f"label_hit_2r_before_stop_{horizon}d"].astype(float).mean()) if len(group) else np.nan,
                    "atr_stop_rate_pct": pct(group[exit_col].astype(str).str.startswith("ATR_STOP").mean()) if len(group) else np.nan,
                    "time_barrier_rate_pct": pct(group[exit_col].astype(str).str.startswith("HORIZON").mean()) if len(group) else np.nan,
                    "mean_holding_days": pd.to_numeric(group[f"label_holding_trading_days_{horizon}d"], errors="coerce").mean(),
                    "mean_net_return_pct": pd.to_numeric(group[f"label_net_return_pct_{horizon}d"], errors="coerce").mean(),
                    "mean_expected_r": pd.to_numeric(group.get(f"label_expected_r_{horizon}d"), errors="coerce").mean(),
                    "median_expected_r": pd.to_numeric(group.get(f"label_expected_r_{horizon}d"), errors="coerce").median(),
                    "mean_mfe_r": pd.to_numeric(group.get(f"label_mfe_r_{horizon}d"), errors="coerce").mean(),
                    "mean_mae_r": pd.to_numeric(group.get(f"label_mae_r_{horizon}d"), errors="coerce").mean(),
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon_days", "prediction_universe"]).reset_index(drop=True)


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, list(numeric_cols)),
            ("cat", categorical_pipeline, list(categorical_cols)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_model(model_name: str, numeric_cols: Sequence[str], categorical_cols: Sequence[str], elastic_params: Optional[Dict[str, float]] = None):
    if SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError(f"scikit-learn is required for prediction engine: {SKLEARN_IMPORT_ERROR}")
    if model_name == "score_logistic":
        preprocessor = build_preprocessor(["score_price_algo_total"], [])
        model = LogisticRegression(class_weight="balanced", solver="lbfgs", l1_ratio=0.0, max_iter=1000, random_state=42)
    else:
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        if model_name == "logistic_balanced":
            model = LogisticRegression(class_weight="balanced", solver="lbfgs", l1_ratio=0.0, max_iter=1000, random_state=42)
        elif model_name == "elastic_net_logistic":
            params = elastic_params or {"C": 0.1, "l1_ratio": 0.5}
            model = LogisticRegression(
                solver="saga",
                class_weight="balanced",
                C=float(params["C"]),
                l1_ratio=float(params["l1_ratio"]),
                max_iter=800,
                tol=1e-3,
                random_state=42,
            )
        elif model_name == "random_forest_fixed":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )
        elif model_name == "hist_gradient_boosting_fixed":
            model = HistGradientBoostingClassifier(
                max_iter=150,
                max_leaf_nodes=15,
                learning_rate=0.05,
                l2_regularization=1.0,
                random_state=42,
            )
        else:
            raise ValueError(f"unknown model: {model_name}")
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def shrinkage_rate(successes: float, count: float, global_rate: float, prior_strength: float = PRIOR_STRENGTH) -> float:
    if count <= 0 or pd.isna(count):
        return bounded_probability(global_rate)
    return bounded_probability((successes + prior_strength * global_rate) / (count + prior_strength))


def base_rate_predict_with_effective_n(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    global_rate = bounded_probability(train[target_col].mean())
    group_cols = ["entry_trigger", "trend_regime", "entry_gate_status"]
    if not all(c in train.columns for c in group_cols):
        return np.full(len(test), global_rate), np.zeros(len(test), dtype=float)
    grouped = train.groupby(group_cols)[target_col].agg(["sum", "count"]).to_dict("index")
    preds: List[float] = []
    effective_n: List[float] = []
    for _, row in test.iterrows():
        key = tuple(row.get(c) for c in group_cols)
        stats = grouped.get(key)
        if stats is None:
            preds.append(global_rate)
            effective_n.append(0.0)
        else:
            preds.append(shrinkage_rate(float(stats["sum"]), float(stats["count"]), global_rate))
            effective_n.append(float(stats["count"]))
    return np.array(preds, dtype=float), np.array(effective_n, dtype=float)


def grouped_rate_predict_with_effective_n(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    group_cols: Sequence[str],
    prior_strength: float = PRIOR_STRENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    clean = train.dropna(subset=[target_col]).copy()
    if clean.empty:
        return np.full(len(test), np.nan), np.zeros(len(test), dtype=float)
    global_rate = bounded_probability(pd.to_numeric(clean[target_col], errors="coerce").mean())
    if not all(c in clean.columns and c in test.columns for c in group_cols):
        return np.full(len(test), global_rate), np.zeros(len(test), dtype=float)
    grouped = clean.groupby(list(group_cols), dropna=False)[target_col].agg(["sum", "count"]).to_dict("index")
    preds: List[float] = []
    effective_n: List[float] = []
    for _, row in test.iterrows():
        stats = grouped.get(tuple(row.get(c) for c in group_cols))
        if stats is None:
            preds.append(global_rate)
            effective_n.append(0.0)
        else:
            count = float(stats["count"])
            preds.append(shrinkage_rate(float(stats["sum"]), count, global_rate, prior_strength=prior_strength))
            effective_n.append(count)
    return np.array(preds, dtype=float), np.array(effective_n, dtype=float)


def base_rate_predict(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> np.ndarray:
    return base_rate_predict_with_effective_n(train, test, target_col)[0]


def shrinkage_mean(values: pd.Series, count: float, global_mean: float, prior_strength: float = PRIOR_STRENGTH) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if count <= 0 or clean.empty or pd.isna(global_mean):
        return float(global_mean) if pd.notna(global_mean) else np.nan
    return float((clean.sum() + prior_strength * global_mean) / (count + prior_strength))


def grouped_latest_rate(train: pd.DataFrame, latest_row: pd.DataFrame, target_col: str) -> float:
    if train.empty or target_col not in train.columns:
        return np.nan
    global_rate = bounded_probability(pd.to_numeric(train[target_col], errors="coerce").mean())
    pred, _ = base_rate_predict_with_effective_n(train.dropna(subset=[target_col]), latest_row, target_col)
    return float(pred[0]) if len(pred) else global_rate


def grouped_latest_mean(train: pd.DataFrame, latest_row: pd.DataFrame, target_col: str) -> float:
    if train.empty or target_col not in train.columns:
        return np.nan
    values = pd.to_numeric(train[target_col], errors="coerce")
    global_mean = float(values.mean()) if values.notna().any() else np.nan
    group_cols = ["entry_trigger", "trend_regime", "entry_gate_status"]
    if not all(c in train.columns for c in group_cols):
        return global_mean
    key = tuple(latest_row.iloc[0].get(c) for c in group_cols)
    mask = pd.Series(True, index=train.index)
    for col, value in zip(group_cols, key):
        mask &= train[col].eq(value)
    group_values = values[mask]
    return shrinkage_mean(group_values, float(group_values.notna().sum()), global_mean)


def model_quality_block_reasons(row: pd.Series) -> str:
    reasons: List[str] = []
    model_name = str(row.get("model_name", ""))
    if not to_bool(row.get("decision_scope_eligible", False)):
        reasons.append("NOT_20D_TRADE_READY_DECISION_SCOPE")
    if model_name in RESEARCH_ONLY_MODELS:
        reasons.append("TREE_OR_FULL_FEATURE_MODEL_RESEARCH_ONLY_SMALL_SAMPLE")
    if int(as_float(row.get("oos_event_count", 0), 0)) < MIN_DECISION_OOS_EVENTS:
        reasons.append("OOS_EVENT_COUNT_LT_100")
    if int(as_float(row.get("selected_oos_event_count", 0), 0)) < MIN_SELECTED_OOS_EVENTS:
        reasons.append("SELECTED_OOS_EVENT_COUNT_LT_50")
    if int(as_float(row.get("min_selected_events_per_fold", 0), 0)) < MIN_SELECTED_EVENTS_PER_FOLD:
        reasons.append("SELECTED_EVENTS_PER_FOLD_LT_10")
    if pd.isna(row.get("brier_score")) or pd.isna(row.get("base_rate_brier_score")) or row.get("brier_score") >= row.get("base_rate_brier_score"):
        reasons.append("NO_BRIER_IMPROVEMENT")
    ece_value = as_float(row.get("decision_ece", row.get("ece")))
    if pd.isna(ece_value) or ece_value > DECISION_ECE_THRESHOLD:
        reasons.append("ECE_GT_0_10")
    if pd.notna(row.get("pr_auc")) and pd.notna(row.get("base_rate_pr_auc")) and row.get("pr_auc") < row.get("base_rate_pr_auc"):
        reasons.append("PR_AUC_NOT_ABOVE_BASE")
    if pd.isna(row.get("expectancy_improvement_pct")) or row.get("expectancy_improvement_pct") <= 0:
        reasons.append("ML_SELECTED_MINUS_RULE_ALL_LE_0")
    if pd.isna(row.get("selected_signal_expectancy_ci_lower_pct")) or row.get("selected_signal_expectancy_ci_lower_pct") <= MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT:
        reasons.append("SELECTED_EXPECTANCY_CI_LOWER_LE_0")
    if int(as_float(row.get("positive_expectancy_folds", 0), 0)) < MIN_POSITIVE_EXPECTANCY_FOLDS:
        reasons.append("POSITIVE_EXPECTANCY_FOLDS_LT_4")
    min_bin_n = int(as_float(row.get("decision_min_calibration_bin_n", row.get("min_calibration_bin_n")), 0))
    if min_bin_n < MIN_CALIBRATION_BIN_N:
        reasons.append("CALIBRATION_MIN_BIN_N_LT_30")
    if pd.isna(row.get("threshold_iqr")) or row.get("threshold_iqr") > MAX_THRESHOLD_IQR:
        reasons.append("THRESHOLD_IQR_GT_0_10")
    return "|".join(reasons) if reasons else "PASS"


def feature_group_for_column(col: str) -> str:
    lower = col.lower()
    if lower.startswith(("tsmc_", "market_", "peer_", "fx_", "external_", "universe_external_")) or "earnings" in lower:
        return "external_context"
    if lower.startswith("hist_"):
        return "history_priors"
    if lower.startswith("news_") or "news_category" in lower:
        return "news_causal_context"
    if lower.startswith("score_") or "score" in lower or lower in {"rsi_14"}:
        return "score_core"
    if "vol" in lower or "atr" in lower or "risk" in lower or "drawdown" in lower or "mae" in lower:
        return "risk_vol"
    if "momentum" in lower or "return" in lower or "sma" in lower or "trend" in lower or "breakout" in lower or "relative" in lower or "beta" in lower:
        return "trend_momentum"
    return "market_context"


def build_feature_contract(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_allowlist = set(NUMERIC_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES)
    for col in feature_matrix.columns:
        if col in {"date", "signal_idx"}:
            role = "metadata"
            leakage_policy = "allowed_metadata"
            available_at = "signal_close"
        elif col.startswith("label_"):
            role = "label"
            leakage_policy = "blocked_from_model_input"
            available_at = "future_window_close"
        elif col in {
            "is_event_candidate",
            "is_actionable_entry_candidate",
            "is_trade_ready_entry_candidate",
            "is_entry_research_candidate",
            "is_risk_research_candidate",
            "is_model_training_candidate",
            "is_decision_entry_candidate",
        }:
            role = "candidate_flag"
            leakage_policy = "allowed_filter_only"
            available_at = "signal_close"
        elif col in model_allowlist:
            role = "model_feature"
            leakage_policy = "allowed_train_only_selection" if not is_future_leakage_feature(col) else "blocked_from_model_input"
            if feature_group_for_column(col) == "external_context":
                available_at = "signal_close_prior_external_release"
            else:
                available_at = "signal_close_prior_web_collection" if col.startswith("news_") or "news_category" in col else "signal_close"
        else:
            role = "unknown"
            leakage_policy = "review_required"
            available_at = "unknown"
        rows.append(
            {
                "column": col,
                "role": role,
                "available_at": available_at,
                "leakage_policy": leakage_policy,
                "feature_group": feature_group_for_column(col),
            }
        )
    return pd.DataFrame(rows)


def _prior_global_rate(values: pd.Series, default: float = 0.5) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    prior_sum = numeric.fillna(0.0).cumsum().shift(1).fillna(0.0)
    prior_count = numeric.notna().astype(float).cumsum().shift(1).fillna(0.0)
    out = prior_sum / prior_count.replace(0, np.nan)
    return out.fillna(default).clip(1e-6, 1.0 - 1e-6)


def _prior_group_rate(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    target_col: str,
    out_col: str,
    default: float = 0.5,
) -> pd.Series:
    target = pd.to_numeric(frame[target_col], errors="coerce")
    global_prior = _prior_global_rate(target, default=default)
    keys = [frame[c] for c in group_cols]
    group_sum_cum = target.fillna(0.0).groupby(keys, dropna=False).cumsum()
    group_count_cum = target.notna().astype(float).groupby(keys, dropna=False).cumsum()
    group_sum = group_sum_cum.groupby(keys, dropna=False).shift(1).fillna(0.0)
    group_count = group_count_cum.groupby(keys, dropna=False).shift(1).fillna(0.0)
    encoded = (group_sum + PRIOR_STRENGTH * global_prior) / (group_count + PRIOR_STRENGTH)
    return encoded.map(bounded_probability).rename(out_col)


def _prior_group_mean(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    out_col: str,
    default: float = 0.0,
) -> pd.Series:
    values = pd.to_numeric(frame[value_col], errors="coerce")
    prior_sum = values.fillna(0.0).cumsum().shift(1).fillna(0.0)
    prior_count = values.notna().astype(float).cumsum().shift(1).fillna(0.0)
    global_prior = (prior_sum / prior_count.replace(0, np.nan)).fillna(default)
    group_sum_cum = values.fillna(0.0).groupby(frame[group_col], dropna=False).cumsum()
    group_count_cum = values.notna().astype(float).groupby(frame[group_col], dropna=False).cumsum()
    group_sum = group_sum_cum.groupby(frame[group_col], dropna=False).shift(1).fillna(0.0)
    group_count = group_count_cum.groupby(frame[group_col], dropna=False).shift(1).fillna(0.0)
    encoded = (group_sum + PRIOR_STRENGTH * global_prior) / (group_count + PRIOR_STRENGTH)
    return encoded.rename(out_col)


def _full_group_rate(train: pd.DataFrame, pred: pd.DataFrame, group_cols: Sequence[str], target_col: str, default: float) -> pd.Series:
    global_rate = bounded_probability(pd.to_numeric(train[target_col], errors="coerce").mean()) if target_col in train.columns else default
    if pd.isna(global_rate):
        global_rate = default
    if not all(c in train.columns and c in pred.columns for c in group_cols):
        return pd.Series(global_rate, index=pred.index, dtype=float)
    stats = train.groupby(list(group_cols), dropna=False)[target_col].agg(["sum", "count"]).to_dict("index")

    def value(row: pd.Series) -> float:
        entry = stats.get(tuple(row.get(c) for c in group_cols))
        if entry is None:
            return float(global_rate)
        return shrinkage_rate(float(entry["sum"]), float(entry["count"]), float(global_rate))

    return pred.apply(value, axis=1)


def _full_group_mean(train: pd.DataFrame, pred: pd.DataFrame, group_col: str, value_col: str, default: float) -> pd.Series:
    values = pd.to_numeric(train[value_col], errors="coerce") if value_col in train.columns else pd.Series(dtype=float)
    global_mean = float(values.mean()) if values.notna().any() else default
    if group_col not in train.columns or group_col not in pred.columns:
        return pd.Series(global_mean, index=pred.index, dtype=float)
    stats = values.groupby(train[group_col], dropna=False).agg(["sum", "count"]).to_dict("index")

    def value(row: pd.Series) -> float:
        entry = stats.get(row.get(group_col))
        if entry is None:
            return global_mean
        return shrinkage_mean(pd.Series([entry["sum"]]), float(entry["count"]), global_mean)

    return pred.apply(value, axis=1)


def add_train_history_features(train: pd.DataFrame, pred: pd.DataFrame, target_col: str, return_col: str, exit_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    pred = pred.copy()

    if target_col in train.columns:
        for group_cols, out_col in [
            (["entry_trigger", "trend_regime"], "hist_success_rate_trigger_trend"),
            (["entry_gate_status"], "hist_success_rate_gate"),
            (["prediction_universe"], "hist_success_rate_universe"),
        ]:
            if all(c in train.columns for c in group_cols):
                train[out_col] = _prior_group_rate(train, group_cols, target_col, out_col)
            else:
                train[out_col] = _prior_global_rate(train[target_col])
            pred[out_col] = _full_group_rate(train, pred, group_cols, target_col, default=0.5)
    else:
        for out_col in ["hist_success_rate_trigger_trend", "hist_success_rate_gate", "hist_success_rate_universe"]:
            train[out_col] = 0.5
            pred[out_col] = 0.5

    if "entry_gate_status" in train.columns and return_col in train.columns:
        train["hist_mean_return_gate"] = _prior_group_mean(train, "entry_gate_status", return_col, "hist_mean_return_gate", default=0.0)
        pred["hist_mean_return_gate"] = _full_group_mean(train, pred, "entry_gate_status", return_col, default=0.0)
    else:
        train["hist_mean_return_gate"] = 0.0
        pred["hist_mean_return_gate"] = 0.0

    if "entry_gate_status" in train.columns and exit_col in train.columns:
        stop_values = train[exit_col].astype(str).str.startswith("ATR_STOP").astype(float)
        stop_frame = train.copy()
        stop_frame["_hist_stop_target"] = stop_values
        train["hist_stop_rate_gate"] = _prior_group_mean(stop_frame, "entry_gate_status", "_hist_stop_target", "hist_stop_rate_gate", default=0.5).clip(0.0, 1.0)
        pred["hist_stop_rate_gate"] = _full_group_mean(stop_frame, pred, "entry_gate_status", "_hist_stop_target", default=float(stop_values.mean()) if len(stop_values) else 0.5).clip(0.0, 1.0)
    else:
        train["hist_stop_rate_gate"] = 0.5
        pred["hist_stop_rate_gate"] = 0.5
    return train, pred


def select_fold_features(
    train: pd.DataFrame,
    numeric_candidates: Sequence[str],
    categorical_candidates: Sequence[str],
    horizon: int,
    candidate_scope_name: str,
    fold_id: int,
    target_name: str,
) -> Tuple[List[str], List[str], List[Dict]]:
    selected_numeric: List[str] = []
    rows: List[Dict] = []
    numeric_data: Dict[str, pd.Series] = {}
    group_priority = {"score_core": 0, "risk_vol": 1, "trend_momentum": 2, "history_priors": 3, "market_context": 4}
    for col in numeric_candidates:
        if col not in train.columns:
            continue
        feature_group = feature_group_for_column(col)
        common = {
            "horizon_days": horizon,
            "candidate_scope": candidate_scope_name,
            "fold_id": fold_id,
            "target_name": target_name,
            "feature": col,
            "feature_type": "numeric",
            "feature_group": feature_group,
            "feature_selection_source": "train_only",
        }
        if is_future_leakage_feature(col):
            rows.append({**common, "decision": "excluded", "reason": "future_or_label_feature_blocked", "missing_rate": np.nan})
            continue
        values = pd.to_numeric(train[col], errors="coerce")
        missing_rate = float(values.isna().mean())
        non_na = values.dropna()
        if missing_rate > MISSING_FEATURE_THRESHOLD:
            rows.append({**common, "decision": "excluded", "reason": "missing_rate_gt_30pct", "missing_rate": missing_rate})
            continue
        if non_na.nunique() <= 1:
            rows.append({**common, "decision": "excluded", "reason": "zero_variance", "missing_rate": missing_rate})
            continue
        selected_numeric.append(col)
        numeric_data[col] = values
        rows.append({**common, "decision": "selected_pre_corr", "reason": "", "missing_rate": missing_rate})

    dropped_corr: set[str] = set()
    if len(selected_numeric) > 1:
        numeric_frame = pd.DataFrame(numeric_data, index=train.index)
        corr = numeric_frame[selected_numeric].corr().abs()
        for i, col in enumerate(selected_numeric):
            if col in dropped_corr:
                continue
            for other in selected_numeric[i + 1 :]:
                if other in dropped_corr:
                    continue
                value = corr.loc[col, other]
                if pd.notna(value) and value > CORRELATION_FEATURE_THRESHOLD:
                    dropped_corr.add(other)
                    rows.append(
                        {
                            "horizon_days": horizon,
                            "candidate_scope": candidate_scope_name,
                            "fold_id": fold_id,
                            "target_name": target_name,
                            "feature": other,
                            "feature_type": "numeric",
                            "feature_group": feature_group_for_column(other),
                            "feature_selection_source": "train_only",
                            "decision": "excluded",
                            "reason": f"corr_gt_{CORRELATION_FEATURE_THRESHOLD}_with:{col}",
                            "missing_rate": float(pd.to_numeric(train[other], errors="coerce").isna().mean()),
                        }
                    )

    final_numeric = [c for c in selected_numeric if c not in dropped_corr]
    if len(final_numeric) > MAX_NUMERIC_FEATURES:
        stats = []
        for col in final_numeric:
            values = pd.to_numeric(train[col], errors="coerce")
            stats.append(
                {
                    "feature": col,
                    "group_rank": group_priority.get(feature_group_for_column(col), 9),
                    "missing_rate": float(values.isna().mean()),
                    "variance": float(values.var(skipna=True)) if values.notna().any() else 0.0,
                }
            )
        ranked = pd.DataFrame(stats).sort_values(["group_rank", "missing_rate", "variance", "feature"], ascending=[True, True, False, True])
        keep = set(ranked.head(MAX_NUMERIC_FEATURES)["feature"])
        cap_dropped = [c for c in final_numeric if c not in keep]
        final_numeric = [c for c in final_numeric if c in keep]
        for col in cap_dropped:
            rows.append(
                {
                    "horizon_days": horizon,
                    "candidate_scope": candidate_scope_name,
                    "fold_id": fold_id,
                    "target_name": target_name,
                    "feature": col,
                    "feature_type": "numeric",
                    "feature_group": feature_group_for_column(col),
                    "feature_selection_source": "train_only",
                    "decision": "excluded",
                    "reason": "max_25_numeric_features",
                    "missing_rate": float(pd.to_numeric(train[col], errors="coerce").isna().mean()),
                }
            )
    selected_categorical: List[str] = []
    for col in categorical_candidates:
        if col not in train.columns:
            continue
        feature_group = feature_group_for_column(col)
        common = {
            "horizon_days": horizon,
            "candidate_scope": candidate_scope_name,
            "fold_id": fold_id,
            "target_name": target_name,
            "feature": col,
            "feature_type": "categorical",
            "feature_group": feature_group,
            "feature_selection_source": "train_only",
        }
        if is_future_leakage_feature(col):
            rows.append({**common, "decision": "excluded", "reason": "future_or_label_feature_blocked", "missing_rate": np.nan})
            continue
        non_na = train[col].dropna()
        if non_na.nunique() <= 1:
            rows.append({**common, "decision": "excluded", "reason": "single_category", "missing_rate": float(train[col].isna().mean())})
            continue
        selected_categorical.append(col)
        rows.append({**common, "decision": "selected", "reason": "", "missing_rate": float(train[col].isna().mean())})

    for row in rows:
        if row["decision"] == "selected_pre_corr":
            row["decision"] = "selected" if row["feature"] in final_numeric else "excluded"
            if row["feature"] not in final_numeric and not row["reason"]:
                row["reason"] = "correlation_filter" if row["feature"] in dropped_corr else "max_25_numeric_features"
    return final_numeric, selected_categorical, rows


def select_elastic_net_params(train: pd.DataFrame, target_col: str, numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> Dict[str, float]:
    y = train[target_col].astype(int)
    if len(train) < 60 or y.nunique() < 2:
        return {"C": 0.1, "l1_ratio": 0.5}
    split_at = max(1, int(len(train) * 0.8))
    train_core = train.iloc[:split_at].copy()
    validation = train.iloc[split_at:].copy()
    if len(validation) < 10 or train_core[target_col].nunique() < 2 or validation[target_col].nunique() < 2:
        return {"C": 0.1, "l1_ratio": 0.5}
    best = {"C": 0.1, "l1_ratio": 0.5}
    best_loss = np.inf
    for c_value in (0.05, 0.1, 0.3, 1.0):
        for l1_ratio in (0.1, 0.5, 0.9):
            try:
                model = make_model("elastic_net_logistic", numeric_cols, categorical_cols, {"C": c_value, "l1_ratio": l1_ratio})
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="'penalty' was deprecated.*", category=FutureWarning)
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(train_core, train_core[target_col].astype(int))
                    prob = model.predict_proba(validation)[:, 1]
                loss = safe_log_loss(validation[target_col], prob)
            except Exception:
                continue
            if pd.notna(loss) and loss < best_loss:
                best_loss = loss
                best = {"C": c_value, "l1_ratio": l1_ratio}
    return best


def fit_predict_model(
    model_name: str,
    train: pd.DataFrame,
    pred: pd.DataFrame,
    target_col: str,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> ModelResult:
    y = train[target_col].astype(int)
    if len(y) == 0:
        return ModelResult(model_name, np.full(len(pred), np.nan), None, "empty_train")
    if y.nunique() < 2:
        return ModelResult(model_name, np.full(len(pred), bounded_probability(y.mean())), None, "single_class_train")
    elastic_params = select_elastic_net_params(train, target_col, numeric_cols, categorical_cols) if model_name == "elastic_net_logistic" else None
    model = make_model(model_name, numeric_cols, categorical_cols, elastic_params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'penalty' was deprecated.*", category=FutureWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(train, y)
        probabilities = model.predict_proba(pred)[:, 1]
    probabilities = np.array([bounded_probability(x) for x in probabilities], dtype=float)
    return ModelResult(model_name, probabilities, model, "")


def two_stage_predict_model(
    model_name: str,
    train: pd.DataFrame,
    pred: pd.DataFrame,
    horizon: int,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> Dict[str, object]:
    success_col = f"label_success_{horizon}d"
    survival_col = f"label_stop_survival_{horizon}d"
    positive_col = f"label_positive_return_{horizon}d"
    return_col = f"label_net_return_pct_{horizon}d"
    exit_col = f"label_exit_reason_{horizon}d"

    train_aug, pred_aug = add_train_history_features(train, pred, success_col, return_col, exit_col)
    if model_name in {"base_rate_by_trigger_regime", "empirical_bayes_group_rate"}:
        if model_name == "empirical_bayes_group_rate":
            group_cols = ["entry_trigger", "trend_regime", "vol_regime", "drawdown_bucket"]
            p_survival, n_survival = grouped_rate_predict_with_effective_n(train_aug, pred_aug, survival_col, group_cols)
        else:
            p_survival, n_survival = base_rate_predict_with_effective_n(train_aug, pred_aug, survival_col)
        survived_train = train_aug[train_aug[survival_col].astype(int) == 1].copy()
        if len(survived_train) < 10:
            if model_name == "empirical_bayes_group_rate":
                p_positive, n_positive = grouped_rate_predict_with_effective_n(train_aug, pred_aug, positive_col, group_cols)
            else:
                p_positive, n_positive = base_rate_predict_with_effective_n(train_aug, pred_aug, positive_col)
        else:
            if model_name == "empirical_bayes_group_rate":
                p_positive, n_positive = grouped_rate_predict_with_effective_n(survived_train, pred_aug, positive_col, group_cols)
            else:
                p_positive, n_positive = base_rate_predict_with_effective_n(survived_train, pred_aug, positive_col)
        final_prob = np.array([bounded_probability(a * b) for a, b in zip(p_survival, p_positive)], dtype=float)
        return {
            "probabilities": final_prob,
            "p_stop_survival": p_survival,
            "p_positive_given_survival": p_positive,
            "effective_n": np.minimum(n_survival, n_positive),
            "fallback_reason": "",
            "train_aug": train_aug,
            "pred_aug": pred_aug,
        }

    survival_result = fit_predict_model(model_name, train_aug.dropna(subset=[survival_col]), pred_aug, survival_col, numeric_cols, categorical_cols)
    survived_train = train_aug[train_aug[survival_col].astype(int) == 1].copy()
    if len(survived_train) < 30 or survived_train[positive_col].nunique() < 2:
        p_positive, n_positive = base_rate_predict_with_effective_n(train_aug, pred_aug, positive_col)
        positive_fallback = "positive_stage_shrinkage_base_rate"
    else:
        positive_result = fit_predict_model(model_name, survived_train.dropna(subset=[positive_col]), pred_aug, positive_col, numeric_cols, categorical_cols)
        p_positive = positive_result.probabilities
        n_positive = np.full(len(pred_aug), len(survived_train), dtype=float)
        positive_fallback = positive_result.fallback_reason
    final_prob = np.array([bounded_probability(a * b) for a, b in zip(survival_result.probabilities, p_positive)], dtype=float)
    fallback = "|".join([x for x in [survival_result.fallback_reason, positive_fallback] if x])
    return {
        "probabilities": final_prob,
        "p_stop_survival": survival_result.probabilities,
        "p_positive_given_survival": p_positive,
        "effective_n": n_positive,
        "fallback_reason": fallback,
        "train_aug": train_aug,
        "pred_aug": pred_aug,
    }


def fit_sigmoid_probability_calibrator(probabilities: np.ndarray, y_true: pd.Series):
    y = y_true.astype(int)
    valid = pd.Series(probabilities).notna().to_numpy()
    if valid.sum() < MIN_FOLD_VALIDATION_EVENTS or y[valid].nunique() < 2:
        return None
    calibrator = LogisticRegression(solver="lbfgs", l1_ratio=0.0, max_iter=1000, random_state=42)
    calibrator.fit(np.asarray(probabilities)[valid].reshape(-1, 1), y[valid])
    return calibrator


def fit_isotonic_probability_calibrator(probabilities: np.ndarray, y_true: pd.Series):
    y = y_true.astype(int)
    valid = pd.Series(probabilities).notna().to_numpy()
    if valid.sum() < 1000 or y[valid].nunique() < 2:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.asarray(probabilities)[valid], y[valid])
    return calibrator


def validation_class_counts(y_true: pd.Series) -> Tuple[int, int, int]:
    clean = pd.to_numeric(y_true, errors="coerce").dropna().astype(int)
    positives = int(clean.sum())
    negatives = int(len(clean) - positives)
    return int(len(clean)), positives, negatives


def validation_sample_ok(y_true: pd.Series) -> bool:
    n, positives, negatives = validation_class_counts(y_true)
    return n >= MIN_FOLD_VALIDATION_EVENTS and positives >= MIN_CALIBRATION_CLASS_COUNT and negatives >= MIN_CALIBRATION_CLASS_COUNT


def fit_probability_calibrator(probabilities: np.ndarray, y_true: pd.Series) -> Tuple[object | None, str]:
    y = y_true.astype(int)
    valid = pd.Series(probabilities).notna().to_numpy()
    n = int(valid.sum())
    valid_y = y[valid]
    positives = int(valid_y.sum()) if n else 0
    negatives = int(n - positives)
    if n < MIN_FOLD_VALIDATION_EVENTS or positives < MIN_CALIBRATION_CLASS_COUNT or negatives < MIN_CALIBRATION_CLASS_COUNT:
        return None, "INSUFFICIENT_CALIBRATION_SAMPLE"
    sigmoid = fit_sigmoid_probability_calibrator(probabilities, y_true)
    if n < 1000:
        return sigmoid, "validation_sigmoid_fixed" if sigmoid is not None else "none"
    isotonic = fit_isotonic_probability_calibrator(probabilities, y_true)
    if isotonic is not None:
        return isotonic, "validation_isotonic_fixed_n_ge_1000"
    return sigmoid, "validation_sigmoid_fixed" if sigmoid is not None else "none"


def apply_probability_calibrator(probabilities: np.ndarray, calibrator) -> np.ndarray:
    if calibrator is None:
        return np.array([bounded_probability(x) for x in probabilities], dtype=float)
    if isinstance(calibrator, IsotonicRegression):
        calibrated = calibrator.predict(np.asarray(probabilities))
        return np.array([bounded_probability(x) for x in calibrated], dtype=float)
    calibrated = calibrator.predict_proba(np.asarray(probabilities).reshape(-1, 1))[:, 1]
    return np.array([bounded_probability(x) for x in calibrated], dtype=float)


def fit_stage_probability_calibrators(stage_result: Dict[str, object], validation: pd.DataFrame, horizon: int) -> Dict[str, object]:
    survival_col = f"label_stop_survival_{horizon}d"
    positive_col = f"label_positive_return_{horizon}d"
    survival_calibrator, survival_method = fit_probability_calibrator(
        np.asarray(stage_result["p_stop_survival"], dtype=float),
        validation[survival_col],
    )
    survived = validation[survival_col].astype(int).to_numpy() == 1
    if int(survived.sum()) < MIN_FOLD_VALIDATION_EVENTS or not validation_sample_ok(validation.loc[survived, positive_col]):
        positive_calibrator = None
        positive_method = "INSUFFICIENT_CALIBRATION_SAMPLE"
    else:
        positive_calibrator, positive_method = fit_probability_calibrator(
            np.asarray(stage_result["p_positive_given_survival"], dtype=float)[survived],
            validation.loc[survived, positive_col],
        )
    return {
        "survival": survival_calibrator,
        "positive": positive_calibrator,
        "method": f"stage_survival:{survival_method};stage_positive:{positive_method}",
    }


def apply_stage_probability_calibrators(stage_result: Dict[str, object], calibrators: Optional[Dict[str, object]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    survival_calibrator = calibrators.get("survival") if calibrators else None
    positive_calibrator = calibrators.get("positive") if calibrators else None
    p_survival = apply_probability_calibrator(np.asarray(stage_result["p_stop_survival"], dtype=float), survival_calibrator)
    p_positive = apply_probability_calibrator(np.asarray(stage_result["p_positive_given_survival"], dtype=float), positive_calibrator)
    p_success = np.array([bounded_probability(a * b) for a, b in zip(p_survival, p_positive)], dtype=float)
    return p_success, p_survival, p_positive


def safe_log_loss(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return np.nan
    return float(log_loss(y_true.astype(int), np.clip(y_prob, 1e-6, 1.0 - 1e-6), labels=[0, 1]))


def safe_average_precision(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return np.nan
    positives = int(y_true.astype(int).sum())
    if positives == 0:
        return 0.0
    if positives == len(y_true):
        return 1.0
    return float(average_precision_score(y_true.astype(int), y_prob))


def safe_nanmean(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if len(arr) else np.nan


def profit_factor(returns_pct: pd.Series) -> float:
    returns = pd.to_numeric(returns_pct, errors="coerce").dropna()
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if losses.empty:
        return np.inf if not wins.empty else np.nan
    return float(wins.sum() / abs(losses.sum()))


def calibration_bins(
    y_true: pd.Series,
    y_prob: np.ndarray,
    n_bins: int,
    horizon: int,
    model_name: str,
    fold_id: int,
    candidate_scope_name: str,
) -> pd.DataFrame:
    rows = []
    tmp = pd.DataFrame({"y": y_true.astype(int).to_numpy(), "p": np.clip(y_prob, 0, 1)})
    tmp["bin"] = np.minimum((tmp["p"] * n_bins).astype(int), n_bins - 1)
    for bin_id, group in tmp.groupby("bin"):
        mean_p = float(group["p"].mean())
        observed = float(group["y"].mean())
        lower, upper = wilson_interval(observed, len(group))
        rows.append(
            {
                "horizon_days": horizon,
                "candidate_scope": candidate_scope_name,
                "model_name": model_name,
                "fold_id": fold_id,
                "bin_id": int(bin_id),
                "n": len(group),
                "mean_predicted_probability": mean_p,
                "observed_success_rate": observed,
                "observed_success_lower_80": lower,
                "observed_success_upper_80": upper,
                "abs_calibration_error": abs(mean_p - observed),
                "calibration_error_lower_adjusted": float(max(lower - mean_p, mean_p - upper, 0.0)),
                "binning": "fixed_width",
            }
        )
    return pd.DataFrame(rows)


def equal_frequency_calibration_bins(
    y_true: pd.Series,
    y_prob: np.ndarray,
    n_bins: int,
    horizon: int,
    model_name: str,
    fold_id: int,
    candidate_scope_name: str,
) -> pd.DataFrame:
    tmp = pd.DataFrame({"y": y_true.astype(int).to_numpy(), "p": np.clip(y_prob, 0, 1)})
    tmp = tmp.dropna(subset=["p"]).sort_values("p").reset_index(drop=True)
    if tmp.empty:
        return pd.DataFrame()
    bin_count = max(1, min(int(n_bins), len(tmp)))
    tmp["bin"] = pd.qcut(tmp.index, q=bin_count, labels=False, duplicates="drop")
    rows = []
    for bin_id, group in tmp.groupby("bin"):
        mean_p = float(group["p"].mean())
        observed = float(group["y"].mean())
        lower, upper = wilson_interval(observed, len(group))
        rows.append(
            {
                "horizon_days": horizon,
                "candidate_scope": candidate_scope_name,
                "model_name": model_name,
                "fold_id": fold_id,
                "bin_id": int(bin_id),
                "n": len(group),
                "min_predicted_probability": group["p"].min(),
                "max_predicted_probability": group["p"].max(),
                "mean_predicted_probability": mean_p,
                "observed_success_rate": observed,
                "observed_success_lower_80": lower,
                "observed_success_upper_80": upper,
                "abs_calibration_error": abs(mean_p - observed),
                "calibration_error_lower_adjusted": float(max(lower - mean_p, mean_p - upper, 0.0)),
                "binning": "equal_frequency",
            }
        )
    return pd.DataFrame(rows)


def adaptive_calibration_bins(
    y_true: pd.Series,
    y_prob: np.ndarray,
    horizon: int,
    model_name: str,
    fold_id: int,
    candidate_scope_name: str,
) -> pd.DataFrame:
    bins = adaptive_calibration_bins_core(
        y_true,
        y_prob,
        target_min_bin_n=MIN_CALIBRATION_BIN_N,
        max_bins=10,
    )
    if bins.empty:
        return bins
    bins = bins.copy()
    bins.insert(0, "horizon_days", horizon)
    bins.insert(1, "candidate_scope", candidate_scope_name)
    bins.insert(2, "model_name", model_name)
    bins.insert(3, "fold_id", fold_id)
    return bins


def expected_calibration_error(bins: pd.DataFrame) -> float:
    if bins.empty or bins["n"].sum() == 0:
        return np.nan
    return float((bins["abs_calibration_error"] * bins["n"]).sum() / bins["n"].sum())


def choose_threshold(
    y_prob: np.ndarray,
    returns_pct: pd.Series,
    thresholds: Sequence[float],
    min_trades: int,
    default_threshold: float,
) -> Tuple[float, float, int, str]:
    returns = pd.to_numeric(returns_pct, errors="coerce")
    required_trades = max(int(min_trades), MIN_THRESHOLD_SELECTED_EVENTS)
    best_threshold = np.nan
    best_expectancy = np.nan
    best_lower_bound = -np.inf
    best_count = 0
    for threshold in thresholds:
        mask = y_prob >= threshold
        count = int(mask.sum())
        if count < required_trades:
            continue
        selected_returns = returns[mask].dropna()
        if len(selected_returns) < required_trades:
            continue
        expectancy = float(selected_returns.mean())
        std = float(selected_returns.std(ddof=1)) if len(selected_returns) > 1 else 0.0
        lower_bound = expectancy - (std / math.sqrt(len(selected_returns)))
        if pd.notna(lower_bound) and lower_bound > best_lower_bound:
            best_lower_bound = lower_bound
            best_expectancy = expectancy
            best_threshold = float(threshold)
            best_count = count
    if best_count == 0:
        return np.nan, np.nan, 0, "INSUFFICIENT_VALIDATION_SELECTION"
    return best_threshold, best_expectancy, best_count, "TRAIN_VALIDATION_EXPECTANCY_LOWER_BOUND"


def purge_history_against_eval(
    history: pd.DataFrame,
    evaluation: pd.DataFrame,
    horizon: int,
    embargo_days: int,
    signal_idx_col: str = "signal_idx",
) -> Tuple[pd.DataFrame, int]:
    if history.empty or evaluation.empty:
        return history.copy(), 0
    history_signal = pd.to_numeric(history[signal_idx_col], errors="coerce")
    eval_signal = pd.to_numeric(evaluation[signal_idx_col], errors="coerce")
    eval_start = int(eval_signal.min())
    eval_end = int(eval_signal.max())
    history_label_start = history_signal + 1
    history_label_end = history_signal + int(horizon)
    eval_label_start = eval_start + 1
    eval_label_end = eval_end + int(horizon)
    overlaps_eval = (history_label_start <= eval_label_end) & (history_label_end >= eval_label_start)
    in_embargo = history_signal >= eval_start - int(embargo_days)
    keep = ~(overlaps_eval | in_embargo)
    return history[keep].copy(), int((~keep).sum())


def sparse_trade_ready_diagnostic_split(
    available: pd.DataFrame,
    horizon: int,
    gap_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]] | None:
    if len(available) < MIN_SPARSE_TRADE_READY_DIAGNOSTIC_EVENTS:
        return None
    frame = available.sort_values("signal_idx").reset_index(drop=True)
    embargo_days = max(int(gap_days), int(horizon))
    max_test_count = len(frame) - MIN_SPARSE_TRADE_READY_TRAIN_EVENTS
    if max_test_count < MIN_SPARSE_TRADE_READY_TEST_EVENTS:
        return None
    initial_test_count = min(
        max(MIN_SPARSE_TRADE_READY_TEST_EVENTS, int(math.ceil(len(frame) * SPARSE_TRADE_READY_TEST_FRACTION))),
        max_test_count,
    )

    for test_count in range(initial_test_count, MIN_SPARSE_TRADE_READY_TEST_EVENTS - 1, -1):
        test = frame.tail(test_count).copy().reset_index(drop=True)
        pre_test = frame.iloc[:-test_count].copy().reset_index(drop=True)
        pre_test, purged_test = purge_history_against_eval(pre_test, test, horizon, embargo_days)
        pre_test = pre_test.sort_values("signal_idx").reset_index(drop=True)
        if len(pre_test) < MIN_SPARSE_TRADE_READY_TRAIN_EVENTS:
            continue
        validation_count = min(
            MIN_SPARSE_TRADE_READY_VALIDATION_EVENTS,
            max(0, len(pre_test) - MIN_SPARSE_TRADE_READY_TRAIN_EVENTS),
        )
        if validation_count > 0:
            validation = pre_test.tail(validation_count).copy().reset_index(drop=True)
            train = pre_test.iloc[:-validation_count].copy().reset_index(drop=True)
            train, purged_validation = purge_history_against_eval(train, validation, horizon, embargo_days)
            train = train.sort_values("signal_idx").reset_index(drop=True)
        else:
            validation = pd.DataFrame(columns=pre_test.columns)
            train = pre_test
            purged_validation = 0
        if len(train) < MIN_SPARSE_TRADE_READY_TRAIN_EVENTS:
            continue
        metadata = {
            "embargo_days": embargo_days,
            "purged_train_count": purged_test + purged_validation,
            "test_count": len(test),
        }
        return train, validation, test, metadata
    return None


def evaluate_sparse_trade_ready_diagnostic(
    available: pd.DataFrame,
    horizon: int,
    candidate_scope_name: str,
    thresholds: Sequence[float],
    min_validation_trades: int,
    default_threshold: float,
    n_bins: int,
    gap_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = sparse_trade_ready_diagnostic_split(available, horizon, gap_days)
    if split is None:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty
    train, validation, test, split_meta = split
    target_col = f"label_success_{horizon}d"
    survival_col = f"label_stop_survival_{horizon}d"
    positive_col = f"label_positive_return_{horizon}d"
    return_col = f"label_net_return_pct_{horizon}d"
    exit_col = f"label_exit_reason_{horizon}d"
    model_names = ["empirical_bayes_group_rate", "base_rate_by_trigger_regime"]
    if candidate_scope_name == "trigger_all" and "score_price_algo_total" in available.columns:
        model_names.append("score_logistic")
    sparse_reason = f"SPARSE_{candidate_scope_name.upper()}_DIAGNOSTIC_ONLY"
    sparse_threshold_prefix = "SPARSE_TRADE_READY" if candidate_scope_name == "trade_ready_entry" else f"SPARSE_{candidate_scope_name.upper()}"

    metric_rows: List[Dict] = []
    threshold_rows: List[Dict] = []
    calibration_rows: List[pd.DataFrame] = []
    prediction_rows: List[Dict] = []
    validation_n, validation_pos, validation_neg = validation_class_counts(validation[target_col]) if not validation.empty else (0, 0, 0)
    validation_ok = validation_sample_ok(validation[target_col]) if not validation.empty else False
    fold_id = 1

    def int_or_nan(value: object) -> object:
        return int(value) if pd.notna(value) else np.nan

    for model_name in model_names:
        model_numeric_cols = ["score_price_algo_total"] if model_name == "score_logistic" else []
        model_categorical_cols: List[str] = []
        threshold = np.nan
        val_expectancy = np.nan
        val_count = 0
        threshold_source = f"{sparse_threshold_prefix}_DEFAULT_THRESHOLD_NO_VALIDATION"
        if len(validation) >= MIN_THRESHOLD_SELECTED_EVENTS:
            val_result = two_stage_predict_model(model_name, train, validation, horizon, model_numeric_cols, model_categorical_cols)
            val_prob = np.asarray(val_result["probabilities"], dtype=float)
            threshold, val_expectancy, val_count, threshold_source = choose_threshold(
                val_prob,
                validation[return_col],
                thresholds,
                min_validation_trades,
                default_threshold,
            )
        if pd.isna(threshold):
            threshold = float(default_threshold)
            threshold_source = f"{sparse_threshold_prefix}_DEFAULT_THRESHOLD_AFTER_{threshold_source}"

        test_result = two_stage_predict_model(model_name, train, test, horizon, model_numeric_cols, model_categorical_cols)
        test_prob = np.asarray(test_result["probabilities"], dtype=float)
        test_survival = np.asarray(test_result["p_stop_survival"], dtype=float)
        test_positive = np.asarray(test_result["p_positive_given_survival"], dtype=float)
        effective_n = np.asarray(test_result["effective_n"], dtype=float)
        y_test = test[target_col].astype(int)
        returns = pd.to_numeric(test[return_col], errors="coerce")
        selected = test_prob >= threshold
        selected_returns = returns[selected]
        selected_return_std = float(selected_returns.std(ddof=1)) if len(selected_returns) > 1 else 0.0
        selected_expectancy_ci_lower = (
            float(selected_returns.mean()) - selected_return_std / math.sqrt(len(selected_returns))
            if len(selected_returns)
            else np.nan
        )
        bins = calibration_bins(y_test, test_prob, n_bins, horizon, model_name, fold_id, candidate_scope_name)
        calibration_rows.append(bins)
        eq_bins = equal_frequency_calibration_bins(y_test, test_prob, n_bins, horizon, model_name, fold_id, candidate_scope_name)
        if not eq_bins.empty:
            calibration_rows.append(eq_bins)
        decision_bins = adaptive_calibration_bins(y_test, test_prob, horizon, model_name, fold_id, candidate_scope_name)
        if not decision_bins.empty:
            calibration_rows.append(decision_bins)
        decision_ece_value = expected_calibration_error(decision_bins)
        decision_min_bin_n = int(decision_bins["n"].min()) if not decision_bins.empty else 0
        fixed_width_ece = expected_calibration_error(bins)
        fixed_width_min_bin_n = int(bins["n"].min()) if not bins.empty else 0
        base_prob = np.full(len(test), bounded_probability(train[target_col].mean()))

        for row_pos, (_, test_row) in enumerate(test.iterrows()):
            prediction_rows.append(
                {
                    "horizon_days": horizon,
                    "candidate_scope": candidate_scope_name,
                    "fold_id": fold_id,
                    "model_name": model_name,
                    "signal_idx": int(test_row["signal_idx"]),
                    "date": test_row["date"],
                    "entry_trigger": test_row.get("entry_trigger", ""),
                    "trade_action": test_row.get("trade_action", ""),
                    "entry_gate_status": test_row.get("entry_gate_status", ""),
                    "prediction_universe": test_row.get("prediction_universe", ""),
                    "p_success": float(test_prob[row_pos]) if row_pos < len(test_prob) else np.nan,
                    "p_stop_survival": float(test_survival[row_pos]) if row_pos < len(test_survival) and pd.notna(test_survival[row_pos]) else np.nan,
                    "p_stop_hit": float(1.0 - test_survival[row_pos]) if row_pos < len(test_survival) and pd.notna(test_survival[row_pos]) else np.nan,
                    "p_positive_given_survival": float(test_positive[row_pos]) if row_pos < len(test_positive) and pd.notna(test_positive[row_pos]) else np.nan,
                    "effective_sample_size": float(effective_n[row_pos]) if row_pos < len(effective_n) and pd.notna(effective_n[row_pos]) else np.nan,
                    "threshold": threshold,
                    "selected_by_threshold": bool(selected[row_pos]) if row_pos < len(selected) else False,
                    "label_success": int(test_row[target_col]),
                    "label_stop_survival": int_or_nan(test_row.get(survival_col)),
                    "label_positive_return": int_or_nan(test_row.get(positive_col)),
                    "label_net_return_pct": as_float(test_row.get(return_col)),
                    "label_expected_r": as_float(test_row.get(f"label_expected_r_{horizon}d")),
                    "label_hit_1r_before_stop": int_or_nan(test_row.get(f"label_hit_1r_before_stop_{horizon}d")),
                    "label_hit_2r_before_stop": int_or_nan(test_row.get(f"label_hit_2r_before_stop_{horizon}d")),
                    "label_ambiguous_stop_1r_same_day": int_or_nan(test_row.get(f"label_ambiguous_stop_1r_same_day_{horizon}d")),
                    "label_ambiguous_stop_2r_same_day": int_or_nan(test_row.get(f"label_ambiguous_stop_2r_same_day_{horizon}d")),
                    "label_gap_through_stop": int_or_nan(test_row.get(f"label_gap_through_stop_{horizon}d")),
                    "label_first_touch_type": test_row.get(f"label_first_touch_type_{horizon}d", ""),
                    "label_exit_reason": test_row.get(exit_col, ""),
                    "label_entry_date": test_row.get(f"label_entry_date_{horizon}d", ""),
                    "label_exit_date": test_row.get(f"label_exit_date_{horizon}d", ""),
                    "label_entry_price": as_float(test_row.get(f"label_entry_price_{horizon}d")),
                    "label_exit_price": as_float(test_row.get(f"label_exit_price_{horizon}d")),
                    "label_stop_price": as_float(test_row.get(f"label_stop_price_{horizon}d")),
                    "train_start_date": train["date"].min(),
                    "train_end_date": train["date"].max(),
                    "test_start_date": test["date"].min(),
                    "test_end_date": test["date"].max(),
                    "calibration_method": "none_sparse_diagnostic",
                    "threshold_source": threshold_source,
                    "fallback_reason": sparse_reason,
                }
            )

        metric_rows.append(
            {
                "horizon_days": horizon,
                "candidate_scope": candidate_scope_name,
                "fold_id": fold_id,
                "model_name": model_name,
                "train_start_date": train["date"].min(),
                "train_end_date": train["date"].max(),
                "test_start_date": test["date"].min(),
                "test_end_date": test["date"].max(),
                "train_end_idx": int(pd.to_numeric(train["signal_idx"], errors="coerce").max()),
                "validation_start_idx": int(pd.to_numeric(validation["signal_idx"], errors="coerce").min()) if not validation.empty else np.nan,
                "validation_end_idx": int(pd.to_numeric(validation["signal_idx"], errors="coerce").max()) if not validation.empty else np.nan,
                "test_start_idx": int(pd.to_numeric(test["signal_idx"], errors="coerce").min()),
                "purge_gap_days": split_meta["embargo_days"],
                "purged_train_count": split_meta["purged_train_count"],
                "train_event_count": len(train),
                "validation_event_count": len(validation),
                "validation_positive_count": validation_pos,
                "validation_negative_count": validation_neg,
                "validation_sample_sufficient": validation_ok,
                "test_event_count": len(test),
                "test_positive_rate": float(y_test.mean()),
                "brier_score": float(brier_score_loss(y_test, test_prob)),
                "base_rate_brier_score": float(brier_score_loss(y_test, base_prob)),
                "log_loss": safe_log_loss(y_test, test_prob),
                "pr_auc": safe_average_precision(y_test, test_prob),
                "base_rate_pr_auc": float(y_test.mean()),
                "ece": decision_ece_value,
                "decision_ece": decision_ece_value,
                "calibration_min_bin_n": decision_min_bin_n,
                "decision_min_calibration_bin_n": decision_min_bin_n,
                "calibration_binning_primary": calibration_binning_primary_core(),
                "fixed_width_ece": fixed_width_ece,
                "fixed_width_min_calibration_bin_n": fixed_width_min_bin_n,
                "threshold": threshold,
                "threshold_source": threshold_source,
                "calibration_method": "none_sparse_diagnostic",
                "fallback_reason": sparse_reason,
                "model_feature_count": len(model_numeric_cols) + len(model_categorical_cols),
                "selected_model_feature_count": len(model_numeric_cols) + len(model_categorical_cols),
                "mean_p_stop_survival": safe_nanmean(test_survival),
                "mean_p_positive_given_survival": safe_nanmean(test_positive),
                "mean_effective_sample_size": safe_nanmean(effective_n),
                "all_signal_expectancy_pct": float(returns.mean()),
                "selected_signal_count": int(selected.sum()),
                "selected_signal_expectancy_pct": float(selected_returns.mean()) if len(selected_returns) else np.nan,
                "selected_signal_expectancy_ci_lower_pct": selected_expectancy_ci_lower,
                "selected_signal_return_std_pct": selected_return_std if len(selected_returns) else np.nan,
                "selected_signal_win_rate_pct": pct((selected_returns > 0).mean()) if len(selected_returns) else np.nan,
                "selected_signal_profit_factor": profit_factor(selected_returns) if len(selected_returns) else np.nan,
                "expectancy_improvement_pct": (float(selected_returns.mean()) - float(returns.mean())) if len(selected_returns) else np.nan,
            }
        )
        threshold_rows.append(
            {
                "horizon_days": horizon,
                "candidate_scope": candidate_scope_name,
                "fold_id": fold_id,
                "model_name": model_name,
                "threshold": threshold,
                "threshold_source": threshold_source,
                "validation_expectancy_pct": val_expectancy,
                "validation_selected_count": val_count,
                "validation_event_count": validation_n,
                "validation_positive_count": validation_pos,
                "validation_negative_count": validation_neg,
                "validation_sample_sufficient": validation_ok,
                "min_validation_trades": min_validation_trades,
            }
        )

    fold_manifest = pd.DataFrame(
        [
            {
                "fold_id": fold_id,
                "train_start_idx": int(pd.to_numeric(train["signal_idx"], errors="coerce").min()),
                "train_end_idx": int(pd.to_numeric(train["signal_idx"], errors="coerce").max()),
                "validation_start_idx": int(pd.to_numeric(validation["signal_idx"], errors="coerce").min()) if not validation.empty else np.nan,
                "validation_end_idx": int(pd.to_numeric(validation["signal_idx"], errors="coerce").max()) if not validation.empty else np.nan,
                "test_start_idx": int(pd.to_numeric(test["signal_idx"], errors="coerce").min()),
                "test_end_idx": int(pd.to_numeric(test["signal_idx"], errors="coerce").max()),
                "horizon_days": horizon,
                "embargo_days": split_meta["embargo_days"],
                "purged_train_count": split_meta["purged_train_count"],
                "train_event_count": len(train),
                "validation_event_count": len(validation),
                "test_event_count": len(test),
                "candidate_scope": candidate_scope_name,
                "status": "SPARSE_DIAGNOSTIC_USED",
            }
        ]
    )
    metrics = pd.DataFrame(metric_rows)
    thresholds_out = pd.DataFrame(threshold_rows)
    calibration = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    feature_selection = pd.DataFrame()
    oos_predictions = pd.DataFrame(prediction_rows)
    return metrics, thresholds_out, calibration, feature_selection, oos_predictions, fold_manifest


def build_folds(max_signal_idx: int, train_days: int, test_days: int, step_days: int, gap_days: int) -> List[Dict]:
    folds = []
    train_end = train_days - 1
    fold_id = 1
    while True:
        test_start = train_end + gap_days + 1
        test_end = test_start + test_days - 1
        if test_start > max_signal_idx:
            break
        folds.append(
            {
                "fold_id": fold_id,
                "train_start_idx": 0,
                "train_end_idx": train_end,
                "test_start_idx": test_start,
                "test_end_idx": min(test_end, max_signal_idx),
                "gap_days": gap_days,
            }
        )
        fold_id += 1
        train_end += step_days
    return folds


def evaluate_prediction_stream(
    frame: pd.DataFrame,
    horizon: int,
    candidate_scope_name: str,
    candidate_col: str,
    train_days: int,
    validation_days: int,
    test_days: int,
    step_days: int,
    gap_days: int,
    thresholds: Sequence[float],
    min_validation_trades: int,
    default_threshold: float,
    n_bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_col = f"label_success_{horizon}d"
    survival_col = f"label_stop_survival_{horizon}d"
    positive_col = f"label_positive_return_{horizon}d"
    return_col = f"label_net_return_pct_{horizon}d"
    status_col = f"label_status_{horizon}d"
    exit_col = f"label_exit_reason_{horizon}d"
    available = frame[(frame[candidate_col]) & (frame[status_col] == "LABELED")].copy()
    available = available.dropna(subset=[target_col, return_col]).sort_values("signal_idx").reset_index(drop=True)

    numeric_cols, categorical_cols = usable_model_columns(available)
    numeric_cols = list(
        dict.fromkeys(
            [
                *numeric_cols,
                "hist_success_rate_trigger_trend",
                "hist_success_rate_gate",
                "hist_success_rate_universe",
                "hist_mean_return_gate",
                "hist_stop_rate_gate",
            ]
        )
    )
    splitter = PurgedEventTimeSplit(
        horizon_days=horizon,
        train_days=train_days,
        validation_days=max(MIN_FOLD_VALIDATION_EVENTS, int(min_validation_trades), int(validation_days)),
        test_days=test_days,
        step_days=step_days,
        embargo_days=max(gap_days, horizon),
        min_train_events=MIN_FOLD_TRAIN_EVENTS,
        min_validation_events=MIN_FOLD_VALIDATION_EVENTS,
        min_test_events=MIN_FOLD_TEST_EVENTS,
    )
    metric_rows: List[Dict] = []
    threshold_rows: List[Dict] = []
    calibration_rows: List[pd.DataFrame] = []
    feature_rows: List[Dict] = []
    prediction_rows: List[Dict] = []
    fold_rows: List[Dict] = []
    if candidate_scope_name == "trade_ready_entry":
        model_names = ["empirical_bayes_group_rate", "base_rate_by_trigger_regime", "score_logistic"]
    elif candidate_scope_name == "entry_research":
        model_names = [
            "empirical_bayes_group_rate",
            "base_rate_by_trigger_regime",
            "score_logistic",
            "elastic_net_logistic",
        ]
    elif candidate_scope_name == "trigger_all":
        model_names = [
            "empirical_bayes_group_rate",
            "base_rate_by_trigger_regime",
            "score_logistic",
            "elastic_net_logistic",
            "logistic_balanced",
        ]
    else:
        model_names = ["empirical_bayes_group_rate", "base_rate_by_trigger_regime"]

    for raw_spec in splitter.iter_specs(int(frame["signal_idx"].max())):
        train, validation, test, fold_spec = splitter.apply(available, raw_spec)
        fold_row = {
            **fold_spec.to_row(),
            "candidate_scope": candidate_scope_name,
            "status": "USED",
        }
        if (
            len(train) < MIN_FOLD_TRAIN_EVENTS
            or len(validation) < MIN_FOLD_VALIDATION_EVENTS
            or len(test) < MIN_FOLD_TEST_EVENTS
        ):
            fold_row["status"] = "SKIPPED_INSUFFICIENT_FOLD_EVENTS"
            fold_rows.append(fold_row)
            continue
        fold_rows.append(fold_row)
        train_core = train.copy()
        validation_ok = validation_sample_ok(validation[target_col])
        validation_n, validation_pos, validation_neg = validation_class_counts(validation[target_col])
        train_feature_basis, _ = add_train_history_features(train, train, target_col, return_col, exit_col)

        for model_name in model_names:
            selected_numeric, selected_categorical, selected_rows = select_fold_features(
                train_feature_basis,
                numeric_cols,
                categorical_cols,
                horizon,
                candidate_scope_name,
                fold_spec.fold_id,
                "success",
            )
            feature_rows.extend(selected_rows)
            if model_name == "score_logistic":
                selected_numeric, selected_categorical = ["score_price_algo_total"], []
            elif model_name in {"base_rate_by_trigger_regime", "empirical_bayes_group_rate"}:
                selected_numeric, selected_categorical = [], []

            if model_name in {"base_rate_by_trigger_regime", "empirical_bayes_group_rate"}:
                calibration_method = "none"
                if not validation_ok:
                    val_prob = np.full(0, np.nan)
                    threshold, val_expectancy, val_count, threshold_source = np.nan, np.nan, 0, "INSUFFICIENT_VALIDATION_CLASS_BALANCE"
                else:
                    val_result = two_stage_predict_model(model_name, train_core, validation, horizon, selected_numeric, selected_categorical)
                    val_prob = val_result["probabilities"]
                    threshold, val_expectancy, val_count, threshold_source = choose_threshold(
                        val_prob,
                        validation[return_col],
                        thresholds,
                        min_validation_trades,
                        default_threshold,
                    )
                test_result = two_stage_predict_model(model_name, train, test, horizon, selected_numeric, selected_categorical)
                test_prob = test_result["probabilities"]
                test_survival = test_result["p_stop_survival"]
                test_positive = test_result["p_positive_given_survival"]
                effective_n = test_result["effective_n"]
                fallback_reason = ""
            else:
                calibration_method = "none"
                stage_calibrators: Optional[Dict[str, object]] = None
                if not validation_ok:
                    val_prob = base_rate_predict(train_core, validation, target_col)
                    calibration_method = "INSUFFICIENT_CALIBRATION_SAMPLE"
                    threshold, val_expectancy, val_count, threshold_source = np.nan, np.nan, 0, "INSUFFICIENT_VALIDATION_CLASS_BALANCE"
                else:
                    train_core_feature_basis, _ = add_train_history_features(train_core, train_core, target_col, return_col, exit_col)
                    core_numeric, core_categorical, _ = select_fold_features(
                        train_core_feature_basis,
                        numeric_cols,
                        categorical_cols,
                        horizon,
                        candidate_scope_name,
                        fold_spec.fold_id,
                        "success_validation_core",
                    )
                    if model_name == "score_logistic":
                        core_numeric, core_categorical = ["score_price_algo_total"], []
                    val_result = two_stage_predict_model(model_name, train_core, validation, horizon, core_numeric, core_categorical)
                    stage_calibrators = fit_stage_probability_calibrators(val_result, validation, horizon)
                    calibration_method = str(stage_calibrators["method"])
                    val_prob, _, _ = apply_stage_probability_calibrators(val_result, stage_calibrators)
                    threshold, val_expectancy, val_count, threshold_source = choose_threshold(
                        val_prob,
                        validation[return_col],
                        thresholds,
                        min_validation_trades,
                        default_threshold,
                    )
                if calibration_method == "INSUFFICIENT_CALIBRATION_SAMPLE":
                    base_prob, base_n = base_rate_predict_with_effective_n(train, test, target_col)
                    test_prob = base_prob
                    test_survival = np.full(len(test), np.nan)
                    test_positive = np.full(len(test), np.nan)
                    effective_n = base_n
                    fallback_reason = "INSUFFICIENT_CALIBRATION_SAMPLE_BASE_RATE"
                else:
                    test_result = two_stage_predict_model(model_name, train, test, horizon, selected_numeric, selected_categorical)
                    if validation.empty:
                        stage_calibrators = None
                    test_prob, test_survival, test_positive = apply_stage_probability_calibrators(test_result, stage_calibrators)
                    effective_n = test_result["effective_n"]
                    fallback_reason = test_result["fallback_reason"]

            y_test = test[target_col].astype(int)
            returns = pd.to_numeric(test[return_col], errors="coerce")
            selected = test_prob >= threshold if pd.notna(threshold) else np.zeros(len(test_prob), dtype=bool)
            selected_returns = returns[selected]
            selected_return_std = float(selected_returns.std(ddof=1)) if len(selected_returns) > 1 else 0.0
            selected_expectancy_ci_lower = (
                float(selected_returns.mean()) - selected_return_std / math.sqrt(len(selected_returns))
                if len(selected_returns)
                else np.nan
            )
            bins = calibration_bins(y_test, test_prob, n_bins, horizon, model_name, fold_spec.fold_id, candidate_scope_name)
            calibration_rows.append(bins)
            eq_bins = equal_frequency_calibration_bins(y_test, test_prob, n_bins, horizon, model_name, fold_spec.fold_id, candidate_scope_name)
            if not eq_bins.empty:
                calibration_rows.append(eq_bins)
            decision_bins = adaptive_calibration_bins(y_test, test_prob, horizon, model_name, fold_spec.fold_id, candidate_scope_name)
            if not decision_bins.empty:
                calibration_rows.append(decision_bins)
            decision_ece_value = expected_calibration_error(decision_bins)
            decision_min_bin_n = int(decision_bins["n"].min()) if not decision_bins.empty else 0
            fixed_width_ece = expected_calibration_error(bins)
            fixed_width_min_bin_n = int(bins["n"].min()) if not bins.empty else 0
            base_prob = np.full(len(test), bounded_probability(train[target_col].mean()))

            for row_pos, (_, test_row) in enumerate(test.iterrows()):
                prediction_rows.append(
                    {
                        "horizon_days": horizon,
                        "candidate_scope": candidate_scope_name,
                        "fold_id": fold_spec.fold_id,
                        "model_name": model_name,
                        "signal_idx": int(test_row["signal_idx"]),
                        "date": test_row["date"],
                        "entry_trigger": test_row.get("entry_trigger", ""),
                        "trade_action": test_row.get("trade_action", ""),
                        "entry_gate_status": test_row.get("entry_gate_status", ""),
                        "prediction_universe": test_row.get("prediction_universe", ""),
                        "p_success": float(test_prob[row_pos]) if row_pos < len(test_prob) else np.nan,
                        "p_stop_survival": float(test_survival[row_pos]) if row_pos < len(test_survival) and pd.notna(test_survival[row_pos]) else np.nan,
                        "p_stop_hit": float(1.0 - test_survival[row_pos]) if row_pos < len(test_survival) and pd.notna(test_survival[row_pos]) else np.nan,
                        "p_positive_given_survival": float(test_positive[row_pos]) if row_pos < len(test_positive) and pd.notna(test_positive[row_pos]) else np.nan,
                        "effective_sample_size": float(effective_n[row_pos]) if row_pos < len(effective_n) and pd.notna(effective_n[row_pos]) else np.nan,
                        "threshold": threshold,
                        "selected_by_threshold": bool(selected[row_pos]) if row_pos < len(selected) else False,
                        "label_success": int(test_row[target_col]),
                        "label_stop_survival": int(test_row[survival_col]) if pd.notna(test_row.get(survival_col)) else np.nan,
                        "label_positive_return": int(test_row[positive_col]) if pd.notna(test_row.get(positive_col)) else np.nan,
                        "label_net_return_pct": as_float(test_row.get(return_col)),
                        "label_expected_r": as_float(test_row.get(f"label_expected_r_{horizon}d")),
                        "label_hit_1r_before_stop": int(test_row[f"label_hit_1r_before_stop_{horizon}d"]) if pd.notna(test_row.get(f"label_hit_1r_before_stop_{horizon}d")) else np.nan,
                        "label_hit_2r_before_stop": int(test_row[f"label_hit_2r_before_stop_{horizon}d"]) if pd.notna(test_row.get(f"label_hit_2r_before_stop_{horizon}d")) else np.nan,
                        "label_ambiguous_stop_1r_same_day": int(test_row[f"label_ambiguous_stop_1r_same_day_{horizon}d"]) if pd.notna(test_row.get(f"label_ambiguous_stop_1r_same_day_{horizon}d")) else np.nan,
                        "label_ambiguous_stop_2r_same_day": int(test_row[f"label_ambiguous_stop_2r_same_day_{horizon}d"]) if pd.notna(test_row.get(f"label_ambiguous_stop_2r_same_day_{horizon}d")) else np.nan,
                        "label_gap_through_stop": int(test_row[f"label_gap_through_stop_{horizon}d"]) if pd.notna(test_row.get(f"label_gap_through_stop_{horizon}d")) else np.nan,
                        "label_first_touch_type": test_row.get(f"label_first_touch_type_{horizon}d", ""),
                        "label_exit_reason": test_row.get(exit_col, ""),
                        "label_entry_date": test_row.get(f"label_entry_date_{horizon}d", ""),
                        "label_exit_date": test_row.get(f"label_exit_date_{horizon}d", ""),
                        "label_entry_price": as_float(test_row.get(f"label_entry_price_{horizon}d")),
                        "label_exit_price": as_float(test_row.get(f"label_exit_price_{horizon}d")),
                        "label_stop_price": as_float(test_row.get(f"label_stop_price_{horizon}d")),
                        "train_start_date": train["date"].min(),
                        "train_end_date": train["date"].max(),
                        "test_start_date": test["date"].min(),
                        "test_end_date": test["date"].max(),
                        "calibration_method": calibration_method,
                        "threshold_source": threshold_source,
                        "fallback_reason": fallback_reason,
                    }
                )

            metric_rows.append(
                {
                    "horizon_days": horizon,
                    "candidate_scope": candidate_scope_name,
                    "fold_id": fold_spec.fold_id,
                    "model_name": model_name,
                    "train_start_date": train["date"].min(),
                    "train_end_date": train["date"].max(),
                    "test_start_date": test["date"].min(),
                    "test_end_date": test["date"].max(),
                    "train_end_idx": fold_spec.train_end_idx,
                    "validation_start_idx": fold_spec.validation_start_idx,
                    "validation_end_idx": fold_spec.validation_end_idx,
                    "test_start_idx": fold_spec.test_start_idx,
                    "purge_gap_days": fold_spec.embargo_days,
                    "purged_train_count": fold_spec.purged_train_count,
                    "train_event_count": len(train),
                    "validation_event_count": len(validation),
                    "validation_positive_count": validation_pos,
                    "validation_negative_count": validation_neg,
                    "validation_sample_sufficient": validation_ok,
                    "test_event_count": len(test),
                    "test_positive_rate": float(y_test.mean()),
                    "brier_score": float(brier_score_loss(y_test, test_prob)),
                    "base_rate_brier_score": float(brier_score_loss(y_test, base_prob)),
                    "log_loss": safe_log_loss(y_test, test_prob),
                    "pr_auc": safe_average_precision(y_test, test_prob),
                    "base_rate_pr_auc": float(y_test.mean()),
                    "ece": decision_ece_value,
                    "decision_ece": decision_ece_value,
                    "calibration_min_bin_n": decision_min_bin_n,
                    "decision_min_calibration_bin_n": decision_min_bin_n,
                    "calibration_binning_primary": calibration_binning_primary_core(),
                    "fixed_width_ece": fixed_width_ece,
                    "fixed_width_min_calibration_bin_n": fixed_width_min_bin_n,
                    "threshold": threshold,
                    "threshold_source": threshold_source,
                    "calibration_method": calibration_method,
                    "fallback_reason": fallback_reason,
                    "model_feature_count": len(numeric_cols) + len(categorical_cols),
                    "selected_model_feature_count": len(selected_numeric) + len(selected_categorical),
                    "mean_p_stop_survival": safe_nanmean(test_survival),
                    "mean_p_positive_given_survival": safe_nanmean(test_positive),
                    "mean_effective_sample_size": safe_nanmean(effective_n),
                    "all_signal_expectancy_pct": float(returns.mean()),
                    "selected_signal_count": int(selected.sum()),
                    "selected_signal_expectancy_pct": float(selected_returns.mean()) if len(selected_returns) else np.nan,
                    "selected_signal_expectancy_ci_lower_pct": selected_expectancy_ci_lower,
                    "selected_signal_return_std_pct": selected_return_std if len(selected_returns) else np.nan,
                    "selected_signal_win_rate_pct": pct((selected_returns > 0).mean()) if len(selected_returns) else np.nan,
                    "selected_signal_profit_factor": profit_factor(selected_returns) if len(selected_returns) else np.nan,
                    "expectancy_improvement_pct": (float(selected_returns.mean()) - float(returns.mean())) if len(selected_returns) else np.nan,
                }
            )
            threshold_rows.append(
                {
                    "horizon_days": horizon,
                    "candidate_scope": candidate_scope_name,
                    "fold_id": fold_spec.fold_id,
                    "model_name": model_name,
                    "threshold": threshold,
                    "threshold_source": threshold_source,
                    "validation_expectancy_pct": val_expectancy,
                    "validation_selected_count": val_count,
                    "validation_event_count": validation_n,
                    "validation_positive_count": validation_pos,
                    "validation_negative_count": validation_neg,
                    "validation_sample_sufficient": validation_ok,
                    "min_validation_trades": min_validation_trades,
                }
            )

    if not metric_rows and candidate_scope_name in SPARSE_DIAGNOSTIC_SCOPES and len(available) >= MIN_SPARSE_TRADE_READY_DIAGNOSTIC_EVENTS:
        return evaluate_sparse_trade_ready_diagnostic(
            available,
            horizon,
            candidate_scope_name,
            thresholds,
            min_validation_trades,
            default_threshold,
            n_bins,
            gap_days,
        )

    metrics = pd.DataFrame(metric_rows)
    thresholds_out = pd.DataFrame(threshold_rows)
    calibration = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    feature_selection = pd.DataFrame(feature_rows)
    oos_predictions = pd.DataFrame(prediction_rows)
    fold_manifest = pd.DataFrame(fold_rows)
    return metrics, thresholds_out, calibration, feature_selection, oos_predictions, fold_manifest


def aggregate_model_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for (candidate_scope_name, horizon, model_name), group in metrics.groupby(["candidate_scope", "horizon_days", "model_name"]):
        weights = group["test_event_count"].astype(float)
        weight_sum = weights.sum()
        def weighted(col: str) -> float:
            if col not in group.columns:
                return np.nan
            return float((group[col] * weights).sum() / weight_sum) if weight_sum else np.nan

        oos_events = int(group["test_event_count"].sum())
        selected_count = int(group["selected_signal_count"].sum())
        brier = weighted("brier_score")
        base_brier = weighted("base_rate_brier_score")
        brier_improvement = (base_brier - brier) / base_brier * 100.0 if base_brier > 0 else np.nan
        pr_auc = weighted("pr_auc")
        base_pr_auc = weighted("base_rate_pr_auc")
        decision_ece_value = weighted("decision_ece") if "decision_ece" in group.columns else weighted("ece")
        ece = decision_ece_value
        fixed_width_ece = weighted("fixed_width_ece") if "fixed_width_ece" in group.columns else np.nan
        expectancy_all = weighted("all_signal_expectancy_pct")
        if selected_count:
            selected_expectancy = float(
                (group["selected_signal_expectancy_pct"].fillna(0) * group["selected_signal_count"]).sum() / selected_count
            )
        else:
            selected_expectancy = np.nan
        expectancy_improvement = selected_expectancy - expectancy_all if pd.notna(selected_expectancy) else np.nan
        if selected_count:
            selected_ci_values = pd.to_numeric(group.get("selected_signal_expectancy_ci_lower_pct", pd.Series(dtype=float)), errors="coerce").dropna()
            selected_expectancy_ci_lower = float(selected_ci_values.min()) if not selected_ci_values.empty else np.nan
        else:
            selected_expectancy_ci_lower = np.nan
        positive_expectancy_folds = int((group["selected_signal_expectancy_pct"] > 0).sum())
        selected_positive_folds = int((group["selected_signal_count"] > 0).sum())
        min_selected_events_per_fold = int(pd.to_numeric(group.get("selected_signal_count", pd.Series([0])), errors="coerce").fillna(0).min())
        decision_min_series = pd.to_numeric(
            group.get("decision_min_calibration_bin_n", group.get("calibration_min_bin_n", pd.Series([0]))),
            errors="coerce",
        ).fillna(0)
        min_calibration_bin_n = int(decision_min_series.min())
        fixed_width_min_bin_n = int(
            pd.to_numeric(group.get("fixed_width_min_calibration_bin_n", pd.Series([0])), errors="coerce").fillna(0).min()
        )
        thresholds = pd.to_numeric(group.get("threshold", pd.Series(dtype=float)), errors="coerce").dropna()
        threshold_iqr = float(thresholds.quantile(0.75) - thresholds.quantile(0.25)) if not thresholds.empty else np.nan
        is_20d_trade_ready = candidate_scope_name == "trade_ready_entry" and int(horizon) == 20
        if not is_20d_trade_ready:
            model_policy = "DIAGNOSTIC_ONLY_NON_DECISION_SCOPE"
        elif oos_events < MIN_DECISION_OOS_EVENTS:
            model_policy = "DIAGNOSTIC_ONLY_INSUFFICIENT_SAMPLE"
        elif model_name in RESEARCH_ONLY_MODELS or (model_name not in DECISION_MODELS):
            model_policy = "RESEARCH_ONLY"
        else:
            model_policy = "DECISION_CANDIDATE"
        quality_pass = (
            model_name in DECISION_MODELS
            and oos_events >= MIN_DECISION_OOS_EVENTS
            and selected_count >= MIN_SELECTED_OOS_EVENTS
            and min_selected_events_per_fold >= MIN_SELECTED_EVENTS_PER_FOLD
            and pd.notna(brier_improvement)
            and brier < base_brier
            and pd.notna(ece)
            and ece <= DECISION_ECE_THRESHOLD
            and pd.notna(pr_auc)
            and pr_auc >= base_pr_auc
            and pd.notna(expectancy_improvement)
            and expectancy_improvement > MIN_EXPECTANCY_IMPROVEMENT_PCT
            and pd.notna(selected_expectancy_ci_lower)
            and selected_expectancy_ci_lower > MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT
            and positive_expectancy_folds >= MIN_POSITIVE_EXPECTANCY_FOLDS
            and min_calibration_bin_n >= MIN_CALIBRATION_BIN_N
            and pd.notna(threshold_iqr)
            and threshold_iqr <= MAX_THRESHOLD_IQR
        )
        rows.append(
            {
                "candidate_scope": candidate_scope_name,
                "horizon_days": horizon,
                "model_name": model_name,
                "fold_count": group["fold_id"].nunique(),
                "oos_event_count": oos_events,
                "selected_oos_event_count": selected_count,
                "brier_score": brier,
                "base_rate_brier_score": base_brier,
                "brier_improvement_pct": brier_improvement,
                "log_loss": weighted("log_loss"),
                "pr_auc": pr_auc,
                "base_rate_pr_auc": base_pr_auc,
                "ece": ece,
                "decision_ece": decision_ece_value,
                "decision_min_calibration_bin_n": min_calibration_bin_n,
                "calibration_binning_primary": calibration_binning_primary_core(),
                "fixed_width_ece": fixed_width_ece,
                "fixed_width_min_calibration_bin_n": fixed_width_min_bin_n,
                "all_signal_expectancy_pct": expectancy_all,
                "selected_signal_expectancy_pct": selected_expectancy,
                "selected_signal_expectancy_ci_lower_pct": selected_expectancy_ci_lower,
                "expectancy_improvement_pct": expectancy_improvement,
                "selected_minus_rule_all_pct": expectancy_improvement,
                "positive_expectancy_folds": positive_expectancy_folds,
                "selected_positive_folds": selected_positive_folds,
                "min_selected_events_per_fold": min_selected_events_per_fold,
                "threshold_iqr": threshold_iqr,
                "min_calibration_bin_n": min_calibration_bin_n,
                "mean_effective_sample_size": weighted("mean_effective_sample_size") if "mean_effective_sample_size" in group.columns else np.nan,
                "min_decision_oos_events": MIN_DECISION_OOS_EVENTS,
                "min_selected_oos_events": MIN_SELECTED_OOS_EVENTS,
                "decision_scope_eligible": bool(is_20d_trade_ready),
                "model_policy": model_policy,
                "prediction_quality_pass": bool(quality_pass),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_score"] = (
        out["prediction_quality_pass"].astype(int) * 1000
        + out["expectancy_improvement_pct"].fillna(-999)
        + out["brier_improvement_pct"].fillna(-999) / 10.0
        - out["ece"].fillna(1.0) * 10.0
    )
    return out.sort_values(["candidate_scope", "horizon_days", "rank_score"], ascending=[True, True, False]).reset_index(drop=True)


def aggregate_calibration_summary(calibration: pd.DataFrame) -> pd.DataFrame:
    if calibration.empty:
        return pd.DataFrame()
    frame = calibration.copy()
    if "binning" not in frame.columns:
        frame["binning"] = "fixed_width"
    rows = []
    for (candidate_scope_name, horizon, model_name, binning), group in frame.groupby(["candidate_scope", "horizon_days", "model_name", "binning"]):
        total = float(pd.to_numeric(group["n"], errors="coerce").sum())
        if total <= 0:
            continue
        rows.append(
            {
                "candidate_scope": candidate_scope_name,
                "horizon_days": horizon,
                "model_name": model_name,
                "binning": binning,
                "bin_count": int(group["bin_id"].nunique()),
                "oos_event_count": int(total),
                "ece": float((group["abs_calibration_error"] * group["n"]).sum() / total),
                "mean_predicted_probability": float((group["mean_predicted_probability"] * group["n"]).sum() / total),
                "observed_success_rate": float((group["observed_success_rate"] * group["n"]).sum() / total),
                "max_abs_calibration_error": float(group["abs_calibration_error"].max()),
                "min_bin_n": int(group["n"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values(["candidate_scope", "horizon_days", "model_name", "binning"]).reset_index(drop=True)


def build_model_audit(metrics: pd.DataFrame, comparison: pd.DataFrame, calibration: pd.DataFrame, threshold_policy: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    rows = []
    for _, row in comparison.iterrows():
        part = metrics[
            (metrics["candidate_scope"] == row["candidate_scope"])
            & (metrics["horizon_days"] == row["horizon_days"])
            & (metrics["model_name"] == row["model_name"])
        ]
        bins = calibration[
            (calibration["candidate_scope"] == row["candidate_scope"])
            & (calibration["horizon_days"] == row["horizon_days"])
            & (calibration["model_name"] == row["model_name"])
        ]
        if "binning" in bins.columns:
            reliability_bins = bins[bins["binning"].fillna("fixed_width").eq("fixed_width")]
        else:
            reliability_bins = bins
        thresholds = threshold_policy[
            (threshold_policy["candidate_scope"] == row["candidate_scope"])
            & (threshold_policy["horizon_days"] == row["horizon_days"])
            & (threshold_policy["model_name"] == row["model_name"])
        ]
        if reliability_bins.empty or reliability_bins["n"].sum() == 0:
            reliability = resolution = uncertainty = np.nan
        else:
            total = float(reliability_bins["n"].sum())
            y_bar = float((reliability_bins["observed_success_rate"] * reliability_bins["n"]).sum() / total)
            reliability = float((((reliability_bins["mean_predicted_probability"] - reliability_bins["observed_success_rate"]) ** 2) * reliability_bins["n"]).sum() / total)
            resolution = float((((reliability_bins["observed_success_rate"] - y_bar) ** 2) * reliability_bins["n"]).sum() / total)
            uncertainty = float(y_bar * (1.0 - y_bar))
        block_reasons = []
        if not to_bool(row.get("decision_scope_eligible", False)):
            block_reasons.append("NOT_20D_TRADE_READY_DECISION_SCOPE")
        if str(row.get("model_name")) in RESEARCH_ONLY_MODELS or str(row.get("model_policy")) == "RESEARCH_ONLY":
            block_reasons.append("TREE_OR_FULL_FEATURE_MODEL_RESEARCH_ONLY_SMALL_SAMPLE")
        if str(row.get("model_policy")) == "DIAGNOSTIC_ONLY_INSUFFICIENT_SAMPLE":
            block_reasons.append("DIAGNOSTIC_ONLY_INSUFFICIENT_SAMPLE")
        if int(as_float(row.get("oos_event_count", 0), 0)) < MIN_DECISION_OOS_EVENTS:
            block_reasons.append("OOS_EVENT_COUNT_LT_100")
        if int(as_float(row.get("selected_oos_event_count", 0), 0)) < MIN_SELECTED_OOS_EVENTS:
            block_reasons.append("SELECTED_OOS_EVENT_COUNT_LT_50")
        if int(as_float(row.get("min_selected_events_per_fold", 0), 0)) < MIN_SELECTED_EVENTS_PER_FOLD:
            block_reasons.append("SELECTED_EVENTS_PER_FOLD_LT_10")
        if pd.isna(row.get("brier_score")) or pd.isna(row.get("base_rate_brier_score")) or row.get("brier_score") >= row.get("base_rate_brier_score"):
            block_reasons.append("NO_BRIER_IMPROVEMENT")
        row_ece = as_float(row.get("decision_ece", row.get("ece")))
        if pd.isna(row_ece) or row_ece > DECISION_ECE_THRESHOLD:
            block_reasons.append("ECE_GT_0_10")
        if pd.notna(row.get("pr_auc")) and pd.notna(row.get("base_rate_pr_auc")) and row.get("pr_auc") < row.get("base_rate_pr_auc"):
            block_reasons.append("PR_AUC_NOT_ABOVE_BASE")
        if pd.isna(row.get("expectancy_improvement_pct")) or row.get("expectancy_improvement_pct") <= 0:
            block_reasons.append("ML_SELECTED_MINUS_RULE_ALL_LE_0")
        if pd.isna(row.get("selected_signal_expectancy_ci_lower_pct")) or row.get("selected_signal_expectancy_ci_lower_pct") <= MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT:
            block_reasons.append("SELECTED_EXPECTANCY_CI_LOWER_LE_0")
        if int(as_float(row.get("positive_expectancy_folds", 0), 0)) < MIN_POSITIVE_EXPECTANCY_FOLDS:
            block_reasons.append("POSITIVE_EXPECTANCY_FOLDS_LT_4")
        row_min_bin_n = int(as_float(row.get("decision_min_calibration_bin_n", row.get("min_calibration_bin_n")), 0))
        if row_min_bin_n < MIN_CALIBRATION_BIN_N:
            block_reasons.append("CALIBRATION_MIN_BIN_N_LT_30")
        if pd.isna(row.get("threshold_iqr")) or row.get("threshold_iqr") > MAX_THRESHOLD_IQR:
            block_reasons.append("THRESHOLD_IQR_GT_0_10")
        rows.append(
            {
                "candidate_scope": row["candidate_scope"],
                "horizon_days": row["horizon_days"],
                "model_name": row["model_name"],
                "model_policy": row.get("model_policy", "UNKNOWN"),
                "prediction_quality_pass": row["prediction_quality_pass"],
                "quality_block_reasons": "|".join(block_reasons) if block_reasons else "PASS",
                "fold_count": row["fold_count"],
                "oos_event_count": row["oos_event_count"],
                "selected_oos_event_count": row["selected_oos_event_count"],
                "threshold_median": thresholds["threshold"].median() if not thresholds.empty else np.nan,
                "threshold_std": thresholds["threshold"].std(ddof=0) if len(thresholds) else np.nan,
                "calibration_methods": "|".join(sorted(part["calibration_method"].dropna().astype(str).unique())) if not part.empty else "",
                "brier_score": row["brier_score"],
                "brier_reliability": reliability,
                "brier_resolution": resolution,
                "brier_uncertainty": uncertainty,
                "brier_improvement_pct": row["brier_improvement_pct"],
                "ece": row["ece"],
                "decision_ece": row.get("decision_ece", row.get("ece", np.nan)),
                "decision_min_calibration_bin_n": row.get("decision_min_calibration_bin_n", row.get("min_calibration_bin_n", 0)),
                "calibration_binning_primary": row.get("calibration_binning_primary", calibration_binning_primary_core()),
                "fixed_width_ece": row.get("fixed_width_ece", np.nan),
                "fixed_width_min_calibration_bin_n": row.get("fixed_width_min_calibration_bin_n", np.nan),
                "pr_auc": row["pr_auc"],
                "base_rate_pr_auc": row["base_rate_pr_auc"],
                "expectancy_improvement_pct": row["expectancy_improvement_pct"],
                "selected_signal_expectancy_ci_lower_pct": row.get("selected_signal_expectancy_ci_lower_pct", np.nan),
                "selected_minus_rule_all_pct": row.get("selected_minus_rule_all_pct", np.nan),
                "positive_expectancy_folds": row.get("positive_expectancy_folds", 0),
                "min_selected_events_per_fold": row.get("min_selected_events_per_fold", 0),
                "threshold_iqr": row.get("threshold_iqr", np.nan),
                "min_calibration_bin_n": row.get("min_calibration_bin_n", 0),
                "mean_effective_sample_size": row.get("mean_effective_sample_size", np.nan),
            }
        )
    return pd.DataFrame(rows).sort_values(["candidate_scope", "horizon_days", "prediction_quality_pass", "brier_improvement_pct"], ascending=[True, True, False, False]).reset_index(drop=True)


def best_model_for_horizon(comparison: pd.DataFrame, horizon: int, candidate_scope_name: str) -> Optional[pd.Series]:
    if comparison.empty or not {"horizon_days", "candidate_scope", "model_name"}.issubset(comparison.columns):
        return None
    part = comparison[(comparison["horizon_days"] == horizon) & (comparison["candidate_scope"] == candidate_scope_name)].copy()
    if part.empty:
        return None
    decision_part = part[part["model_name"].isin(DECISION_MODELS)].copy()
    if not decision_part.empty:
        part = decision_part
    return part.sort_values("rank_score", ascending=False).iloc[0]


def latest_prediction_for_horizon(
    feature_matrix: pd.DataFrame,
    comparison: pd.DataFrame,
    threshold_policy: pd.DataFrame,
    horizon: int,
    latest_row: pd.DataFrame,
    candidate_scope_name: str,
    candidate_col: str,
    output_prefix: str = "",
) -> Dict:
    prefix = f"{output_prefix}_" if output_prefix else ""
    best = best_model_for_horizon(comparison, horizon, candidate_scope_name)
    if best is None:
        return {
            f"{prefix}best_model_{horizon}d": "NA",
            f"{prefix}p_success_{horizon}d": np.nan,
            f"{prefix}p_stop_survival_{horizon}d": np.nan,
            f"{prefix}p_stop_hit_{horizon}d": np.nan,
            f"{prefix}p_positive_given_survival_{horizon}d": np.nan,
            f"{prefix}p_hit_1r_{horizon}d": np.nan,
            f"{prefix}p_hit_2r_{horizon}d": np.nan,
            f"{prefix}expected_r_{horizon}d": np.nan,
            f"{prefix}expected_net_return_{horizon}d": np.nan,
            f"{prefix}decision_score_{horizon}d": np.nan,
            f"{prefix}p_success_lower_80_{horizon}d": np.nan,
            f"{prefix}p_success_upper_80_{horizon}d": np.nan,
            f"{prefix}threshold_{horizon}d": np.nan,
            f"{prefix}confidence_band_{horizon}d": "NA",
            f"{prefix}prediction_quality_pass_{horizon}d": False,
            f"{prefix}oos_event_count_{horizon}d": 0,
            f"{prefix}effective_oos_event_count_{horizon}d": 0,
            f"{prefix}selected_oos_event_count_{horizon}d": 0,
            f"{prefix}selected_expectancy_ci_lower_pct_{horizon}d": np.nan,
            f"{prefix}selected_minus_rule_all_pct_{horizon}d": np.nan,
            f"{prefix}threshold_iqr_{horizon}d": np.nan,
            f"{prefix}min_selected_events_per_fold_{horizon}d": 0,
            f"{prefix}model_quality_block_reasons_{horizon}d": "NO_MODEL",
        }

    target_col = f"label_success_{horizon}d"
    status_col = f"label_status_{horizon}d"
    train = feature_matrix[(feature_matrix[candidate_col]) & (feature_matrix[status_col] == "LABELED")].copy()
    train = train.dropna(subset=[target_col]).sort_values("signal_idx").reset_index(drop=True)
    numeric_cols, categorical_cols = usable_model_columns(train)
    numeric_cols = list(
        dict.fromkeys(
            [
                *numeric_cols,
                "hist_success_rate_trigger_trend",
                "hist_success_rate_gate",
                "hist_success_rate_universe",
                "hist_mean_return_gate",
                "hist_stop_rate_gate",
            ]
        )
    )
    train_feature_basis, _ = add_train_history_features(train, train, target_col, f"label_net_return_pct_{horizon}d", f"label_exit_reason_{horizon}d")
    selected_numeric, selected_categorical, _ = select_fold_features(train_feature_basis, numeric_cols, categorical_cols, horizon, candidate_scope_name, 0, "latest_success")
    model_name = str(best["model_name"])
    p_stop_survival = np.nan
    p_positive_given_survival = np.nan
    effective_n = int(best.get("oos_event_count", 0))
    if train.empty:
        probability = np.nan
    else:
        if model_name == "score_logistic":
            selected_numeric, selected_categorical = ["score_price_algo_total"], []
        elif model_name in {"base_rate_by_trigger_regime", "empirical_bayes_group_rate"}:
            selected_numeric, selected_categorical = [], []
        stage_calibrators: Optional[Dict[str, object]] = None
        calibration_sample_status = "none"
        if len(train) >= 60 and model_name not in {"base_rate_by_trigger_regime", "empirical_bayes_group_rate"}:
            split_at = max(1, int(len(train) * 0.8))
            train_core = train.iloc[:split_at].copy()
            validation = train.iloc[split_at:].copy()
            if not validation_sample_ok(validation[target_col]):
                calibration_sample_status = "INSUFFICIENT_CALIBRATION_SAMPLE"
            else:
                train_core_feature_basis, _ = add_train_history_features(train_core, train_core, target_col, f"label_net_return_pct_{horizon}d", f"label_exit_reason_{horizon}d")
                core_numeric, core_categorical, _ = select_fold_features(train_core_feature_basis, numeric_cols, categorical_cols, horizon, candidate_scope_name, 0, "latest_validation_core")
                if model_name == "score_logistic":
                    core_numeric, core_categorical = ["score_price_algo_total"], []
                val_result = two_stage_predict_model(model_name, train_core, validation, horizon, core_numeric, core_categorical)
                stage_calibrators = fit_stage_probability_calibrators(val_result, validation, horizon)
                calibration_sample_status = str(stage_calibrators["method"])
        if calibration_sample_status == "INSUFFICIENT_CALIBRATION_SAMPLE" and model_name not in {"base_rate_by_trigger_regime", "empirical_bayes_group_rate"}:
            base_prob, base_n = base_rate_predict_with_effective_n(train, latest_row, target_col)
            probability = base_prob[0] if len(base_prob) else np.nan
            effective_n = int(base_n[0]) if len(base_n) else effective_n
        else:
            result = two_stage_predict_model(model_name, train, latest_row, horizon, selected_numeric, selected_categorical)
            prob, p_survival, p_positive = apply_stage_probability_calibrators(result, stage_calibrators)
            probability = prob[0] if len(prob) else np.nan
            p_stop_survival = p_survival[0] if len(p_survival) else np.nan
            p_positive_given_survival = p_positive[0] if len(p_positive) else np.nan
            effective_values = result["effective_n"]
            if len(effective_values) and pd.notna(effective_values[0]):
                effective_n = int(effective_values[0])

    if pd.isna(p_stop_survival):
        p_stop_survival = grouped_latest_rate(train, latest_row, f"label_stop_survival_{horizon}d")
    p_stop_hit = 1.0 - p_stop_survival if pd.notna(p_stop_survival) else np.nan
    p_hit_1r = grouped_latest_rate(train, latest_row, f"label_hit_1r_before_stop_{horizon}d")
    p_hit_2r = grouped_latest_rate(train, latest_row, f"label_hit_2r_before_stop_{horizon}d")
    expected_r = grouped_latest_mean(train, latest_row, f"label_expected_r_{horizon}d")
    expected_net_return = grouped_latest_mean(train, latest_row, f"label_net_return_pct_{horizon}d")

    part_thresholds = threshold_policy[
        (threshold_policy["candidate_scope"] == candidate_scope_name)
        & (threshold_policy["horizon_days"] == horizon)
        & (threshold_policy["model_name"] == model_name)
    ]
    valid_thresholds = pd.to_numeric(part_thresholds["threshold"], errors="coerce").dropna() if not part_thresholds.empty else pd.Series(dtype=float)
    threshold = float(valid_thresholds.median()) if not valid_thresholds.empty else np.nan
    if pd.isna(probability):
        band = "NA"
    elif probability >= 0.70:
        band = "HIGH"
    elif probability >= 0.60:
        band = "MEDIUM"
    elif probability >= 0.50:
        band = "LOW_POSITIVE"
    else:
        band = "LOW_NEGATIVE"
    lower_80, upper_80 = wilson_interval(probability, max(int(best.get("oos_event_count", 0)), 1))
    return {
        f"{prefix}best_model_{horizon}d": model_name,
        f"{prefix}p_success_{horizon}d": probability,
        f"{prefix}p_stop_survival_{horizon}d": p_stop_survival,
        f"{prefix}p_stop_hit_{horizon}d": p_stop_hit,
        f"{prefix}p_positive_given_survival_{horizon}d": p_positive_given_survival,
        f"{prefix}p_hit_1r_{horizon}d": p_hit_1r,
        f"{prefix}p_hit_2r_{horizon}d": p_hit_2r,
        f"{prefix}expected_r_{horizon}d": expected_r,
        f"{prefix}expected_net_return_{horizon}d": expected_net_return,
        f"{prefix}decision_score_{horizon}d": fixed_20d_decision_score(probability, p_stop_hit, expected_r) if horizon == 20 else np.nan,
        f"{prefix}p_success_lower_80_{horizon}d": lower_80,
        f"{prefix}p_success_upper_80_{horizon}d": upper_80,
        f"{prefix}threshold_{horizon}d": threshold,
        f"{prefix}confidence_band_{horizon}d": band,
        f"{prefix}prediction_quality_pass_{horizon}d": bool(best["prediction_quality_pass"]),
        f"{prefix}oos_event_count_{horizon}d": int(best["oos_event_count"]),
        f"{prefix}effective_oos_event_count_{horizon}d": effective_n,
        f"{prefix}brier_score_{horizon}d": best["brier_score"],
            f"{prefix}brier_improvement_pct_{horizon}d": best["brier_improvement_pct"],
            f"{prefix}ece_{horizon}d": best["ece"],
            f"{prefix}pr_auc_{horizon}d": best["pr_auc"],
            f"{prefix}expectancy_improvement_pct_{horizon}d": best["expectancy_improvement_pct"],
            f"{prefix}selected_oos_event_count_{horizon}d": int(best.get("selected_oos_event_count", 0)),
            f"{prefix}selected_expectancy_ci_lower_pct_{horizon}d": best.get("selected_signal_expectancy_ci_lower_pct", np.nan),
            f"{prefix}selected_minus_rule_all_pct_{horizon}d": best.get("selected_minus_rule_all_pct", np.nan),
            f"{prefix}threshold_iqr_{horizon}d": best.get("threshold_iqr", np.nan),
            f"{prefix}min_selected_events_per_fold_{horizon}d": best.get("min_selected_events_per_fold", 0),
            f"{prefix}model_quality_block_reasons_{horizon}d": model_quality_block_reasons(best),
        }


def prediction_signal_status(probability: float, threshold: float, oos_event_count: int) -> str:
    if oos_event_count < MIN_DECISION_OOS_EVENTS:
        return "INSUFFICIENT_OOS_EVIDENCE"
    if pd.isna(probability) or pd.isna(threshold):
        return "INSUFFICIENT_DATA"
    if probability >= threshold:
        return "PREDICTION_CONFIRMED"
    if probability >= 0.50:
        return "PREDICTION_NEUTRAL"
    return "PREDICTION_FILTERED"


def fixed_20d_decision_score(p_success: float, p_stop_hit: float, expected_r: float) -> float:
    if pd.isna(p_success) or pd.isna(p_stop_hit) or pd.isna(expected_r):
        return np.nan
    return float(0.70 * p_success - 0.40 * p_stop_hit + 0.20 * math.tanh(expected_r / 1.5))


def paper_alpha_sizing(values: Dict[str, object], latest_row: pd.DataFrame) -> Dict[str, object]:
    if values.get("prediction_use_status") != PredictionUseStatus.DECISION_SUPPORT_ALLOWED.value:
        return {
            "paper_alpha_weight_pct": 0.0,
            "paper_alpha_size_reason": "PREDICTION_GATE_BLOCKED",
            "paper_alpha_block_reasons": values.get("latest_signal_block_reasons", "PREDICTION_NOT_DECISION_SUPPORT"),
        }
    row = latest_row.iloc[0]
    vol = as_float(row.get("vol_20d_ann"))
    risk_pct = as_float(row.get("risk_pct_2atr"))
    decision_score = as_float(values.get("decision_score_20d"))
    threshold = as_float(values.get("threshold_20d"))
    block_reasons: List[str] = []
    if pd.isna(vol) or vol <= 0:
        block_reasons.append("VOL_20D_INVALID")
    if pd.isna(risk_pct) or risk_pct <= 0:
        block_reasons.append("RISK_PCT_2ATR_INVALID")
    if pd.isna(decision_score) or pd.isna(threshold):
        block_reasons.append("DECISION_SCORE_OR_THRESHOLD_INVALID")
    if block_reasons:
        return {
            "paper_alpha_weight_pct": 0.0,
            "paper_alpha_size_reason": "SIZE_INPUT_INVALID",
            "paper_alpha_block_reasons": "|".join(block_reasons),
        }
    vol_cap = min(0.25, 0.12 / vol)
    stop_cap = min(0.25, 0.01 / risk_pct)
    confidence = float(np.clip((decision_score - threshold) / 0.20, 0.0, 1.0))
    weight = max(0.0, min(vol_cap, stop_cap) * confidence)
    if weight <= 0:
        block_reasons.append("CONFIDENCE_AT_OR_BELOW_THRESHOLD")
    return {
        "paper_alpha_weight_pct": pct(weight),
        "paper_alpha_size_reason": f"vol_cap={vol_cap:.4f};stop_cap={stop_cap:.4f};confidence={confidence:.4f}",
        "paper_alpha_block_reasons": "|".join(block_reasons) if block_reasons else "PASS",
    }


def final_trade_decision_from_prediction(values: Dict[str, object], latest_row: pd.DataFrame) -> str:
    row = latest_row.iloc[0]
    if not bool(row.get("is_trade_ready_entry_candidate", False)):
        if str(row.get("entry_trigger", "NONE")) == "NONE":
            return FinalTradeDecision.NO_TRADE_NO_TRIGGER.value
        return FinalTradeDecision.RULE_BASED_SMALL_OR_PAPER_ONLY.value
    if not bool(values.get("prediction_quality_pass_20d", False)):
        return FinalTradeDecision.NO_TRADE_MODEL_NOT_TRUSTED.value
    if as_float(values.get("p_stop_hit_20d")) > 0.35:
        return FinalTradeDecision.NO_TRADE_STOP_RISK.value
    if as_float(values.get("expected_r_20d")) < 0.35:
        return FinalTradeDecision.NO_TRADE_LOW_EXPECTANCY.value
    threshold = as_float(values.get("threshold_20d"))
    probability = as_float(values.get("p_success_20d"))
    if pd.isna(threshold) or pd.isna(probability) or probability < threshold:
        return FinalTradeDecision.NO_TRADE_BELOW_PROBABILITY_THRESHOLD.value
    if as_float(values.get("expected_r_20d")) >= 0.50 and as_float(values.get("p_hit_1r_20d")) >= 0.60:
        return FinalTradeDecision.ALPHA_RESEARCH_LONG_ALLOWED.value
    return FinalTradeDecision.ALPHA_RESEARCH_SMALL_LONG_ALLOWED.value


def build_latest_snapshot(feature_matrix: pd.DataFrame, comparison: pd.DataFrame, threshold_policy: pd.DataFrame) -> pd.DataFrame:
    latest_row = feature_matrix.iloc[[-1]].copy()
    latest_is_actionable = bool(latest_row["is_actionable_entry_candidate"].iloc[0])
    latest_is_trade_ready = bool(latest_row["is_trade_ready_entry_candidate"].iloc[0])
    latest_is_model_training = bool(latest_row.get("is_model_training_candidate", pd.Series([latest_is_trade_ready])).iloc[0])
    latest_is_event = bool(latest_row["is_event_candidate"].iloc[0])
    if latest_is_trade_ready:
        scope_used = "trade_ready_entry"
        scope_col = "is_trade_ready_entry_candidate"
    elif latest_is_model_training:
        scope_used = "entry_research"
        scope_col = "is_model_training_candidate"
    elif latest_is_actionable:
        scope_used = "trigger_all"
        scope_col = "is_actionable_entry_candidate"
    elif latest_is_event:
        scope_used = "context_all"
        scope_col = "is_event_candidate"
    else:
        scope_used = "none"
        scope_col = "is_event_candidate"
    values: Dict[str, object] = {
        "prediction_asof_date": latest_row["date"].iloc[0],
        "latest_is_event_candidate": latest_is_event,
        "latest_is_actionable_entry_candidate": latest_is_actionable,
        "latest_is_trade_ready_entry_candidate": latest_is_trade_ready,
        "latest_is_model_training_candidate": latest_is_model_training,
        "latest_entry_gate_status": latest_row["entry_gate_status"].iloc[0],
        "latest_candidate_scope": latest_row["candidate_scope"].iloc[0],
        "latest_candidate_tier": latest_row["candidate_tier"].iloc[0] if "candidate_tier" in latest_row.columns else "NA",
        "prediction_scope_used": scope_used,
    }
    for horizon in HORIZONS:
        values.update(
            latest_prediction_for_horizon(
                feature_matrix,
                comparison,
                threshold_policy,
                horizon,
                latest_row,
                scope_used,
                scope_col,
            )
        )
        values.update(
            latest_prediction_for_horizon(
                feature_matrix,
                comparison,
                threshold_policy,
                horizon,
                latest_row,
                "trade_ready_entry",
                "is_trade_ready_entry_candidate",
                output_prefix="trade_ready",
            )
        )
        values.update(
            latest_prediction_for_horizon(
                feature_matrix,
                comparison,
                threshold_policy,
                horizon,
                latest_row,
                "entry_research",
                "is_model_training_candidate",
                output_prefix="entry_research",
            )
        )
        values.update(
            latest_prediction_for_horizon(
                feature_matrix,
                comparison,
                threshold_policy,
                horizon,
                latest_row,
                "trigger_all",
                "is_actionable_entry_candidate",
                output_prefix="trigger",
            )
        )
        values.update(
            latest_prediction_for_horizon(
                feature_matrix,
                comparison,
                threshold_policy,
                horizon,
                latest_row,
                "context_all",
                "is_event_candidate",
                output_prefix="context",
            )
        )

    p20 = as_float(values.get("p_success_20d"))
    threshold20 = as_float(values.get("threshold_20d"))
    oos20 = int(as_float(values.get("oos_event_count_20d"), 0))
    latest_signal_block_reasons: List[str] = []
    if not latest_is_event:
        status = "NO_LATEST_EVENT_CANDIDATE"
        latest_signal_block_reasons.append("LATEST_NOT_EVENT_CANDIDATE")
    elif latest_is_actionable and not latest_is_trade_ready:
        status = "ENTRY_TRIGGER_FILTERED_BY_RULES"
        latest_signal_block_reasons.append("LATEST_RULE_FILTERED_NOT_TRADE_READY")
    elif not latest_is_actionable:
        status = "NO_ENTRY_TRIGGER_CONTEXT_ONLY"
        latest_signal_block_reasons.append("LATEST_CONTEXT_ONLY_NO_ENTRY_TRIGGER")
    else:
        status = prediction_signal_status(p20, threshold20, oos20)
        if status != "PREDICTION_CONFIRMED":
            latest_signal_block_reasons.append(status)
    quality_ok = bool(values.get("prediction_quality_pass_20d", False))
    use_status = PredictionUseStatus.DISPLAY_ONLY_QUALITY_NOT_PASSED.value
    if status == "NO_ENTRY_TRIGGER_CONTEXT_ONLY":
        use_status = PredictionUseStatus.DISPLAY_ONLY_NO_ENTRY_TRIGGER.value
    elif status == "ENTRY_TRIGGER_FILTERED_BY_RULES":
        use_status = PredictionUseStatus.DISPLAY_ONLY_RULE_FILTERED.value
    elif status in {"NO_MODEL_CANDIDATE", "NO_LATEST_EVENT_CANDIDATE"}:
        use_status = PredictionUseStatus.DISPLAY_ONLY_NO_MODEL_CANDIDATE.value
    elif status == "INSUFFICIENT_DATA":
        use_status = PredictionUseStatus.DISPLAY_ONLY_INSUFFICIENT_DATA.value
    elif status == "INSUFFICIENT_OOS_EVIDENCE":
        use_status = PredictionUseStatus.DISPLAY_ONLY_INSUFFICIENT_OOS_EVIDENCE.value
    trade_gate_ok = (
        quality_ok
        and latest_is_trade_ready
        and status == "PREDICTION_CONFIRMED"
        and as_float(values.get("p_stop_hit_20d")) <= 0.35
        and as_float(values.get("expected_r_20d")) >= 0.35
    )
    if trade_gate_ok:
        use_status = PredictionUseStatus.DECISION_SUPPORT_ALLOWED.value
    elif quality_ok and latest_is_trade_ready:
        use_status = PredictionUseStatus.DISPLAY_ONLY_RULE_FILTERED.value

    values["prediction_signal_status"] = status
    values["latest_signal_block_reasons"] = "|".join(latest_signal_block_reasons) if latest_signal_block_reasons else "PASS"
    values["prediction_use_status"] = use_status
    values["model_support_route"] = (
        "DISPLAY_ONLY_NO_MODEL_CANDIDATE"
        if use_status == PredictionUseStatus.DISPLAY_ONLY_NO_MODEL_CANDIDATE.value or scope_used == "none"
        else "LOCAL_TSM_DIAGNOSTIC"
    )
    values["decision_permission"] = (
        DecisionPermission.DECISION_SUPPORT_ONLY.value
        if use_status == PredictionUseStatus.DECISION_SUPPORT_ALLOWED.value
        else (DecisionPermission.PAPER_ONLY_RULE_BASED.value if latest_is_trade_ready else DecisionPermission.DISPLAY_ONLY.value)
    )
    values["model_health_scope"] = "trade_ready_entry_20d"
    values["model_health_quality_pass_20d"] = values.get("trade_ready_prediction_quality_pass_20d", False)
    values["model_health_best_model_20d"] = values.get("trade_ready_best_model_20d", "NA")
    values["model_health_oos_event_count_20d"] = values.get("trade_ready_oos_event_count_20d", 0)
    values["model_health_selected_oos_event_count_20d"] = values.get("trade_ready_selected_oos_event_count_20d", 0)
    values["model_health_block_reasons_20d"] = values.get("trade_ready_model_quality_block_reasons_20d", values.get("model_quality_block_reasons_20d", "UNKNOWN"))
    values["model_quality_block_reasons"] = values["model_health_block_reasons_20d"]
    values["final_trade_decision"] = final_trade_decision_from_prediction(values, latest_row)
    values.update(paper_alpha_sizing(values, latest_row))
    values["live_trading_status"] = "DISABLED_BY_DESIGN"
    return pd.DataFrame([{"field": k, "value": v} for k, v in values.items()])


def check_row(check: str, passed: bool, severity: str, value, tolerance="", details="") -> Dict:
    return {
        "check": check,
        "passed": bool(passed),
        "severity": severity,
        "value": value,
        "tolerance": tolerance,
        "details": details,
    }


def label_self_test_checks(commission_bps: float, slippage_bps: float, stop_multiple: float) -> List[Dict]:
    rows = []
    base = pd.DataFrame(
        [
            {"date": pd.Timestamp("2020-01-01"), "open": 100, "high": 101, "low": 99, "close": 100, "atr_14": 2, "entry_trigger": "20D_BREAKOUT", "trade_action": "ENTRY_ALLOWED"},
            {"date": pd.Timestamp("2020-01-02"), "open": 100, "high": 103, "low": 99, "close": 102, "atr_14": 2, "entry_trigger": "NONE", "trade_action": "NO_TRADE"},
            {"date": pd.Timestamp("2020-01-03"), "open": 102, "high": 104, "low": 101, "close": 103, "atr_14": 2, "entry_trigger": "NONE", "trade_action": "NO_TRADE"},
            {"date": pd.Timestamp("2020-01-06"), "open": 103, "high": 105, "low": 102, "close": 104, "atr_14": 2, "entry_trigger": "NONE", "trade_action": "NO_TRADE"},
            {"date": pd.Timestamp("2020-01-07"), "open": 104, "high": 106, "low": 103, "close": 105, "atr_14": 2, "entry_trigger": "NONE", "trade_action": "NO_TRADE"},
        ]
    )
    base["is_event_candidate"] = base.apply(event_candidate, axis=1)
    pos = label_event_horizon(base, 0, 4, commission_bps, slippage_bps, stop_multiple)
    rows.append(check_row("label_self_test_positive_no_stop", pos["label_success_4d"] == 1, "CRITICAL", pos["label_success_4d"]))
    rows.append(check_row("label_self_test_1r_hit_before_stop", pos["label_hit_1r_before_stop_4d"] == 1 and pos["label_hit_2r_before_stop_4d"] == 0, "CRITICAL", f"1R={pos['label_hit_1r_before_stop_4d']},2R={pos['label_hit_2r_before_stop_4d']}"))

    stop = base.copy()
    stop.loc[1, "low"] = 95
    stop.loc[1, "high"] = 110
    stop["is_event_candidate"] = stop.apply(event_candidate, axis=1)
    stopped = label_event_horizon(stop, 0, 4, commission_bps, slippage_bps, stop_multiple)
    rows.append(check_row("label_self_test_stop_first", stopped["label_success_4d"] == 0 and str(stopped["label_exit_reason_4d"]).startswith("ATR_STOP"), "CRITICAL", stopped["label_exit_reason_4d"]))
    rows.append(check_row("label_self_test_same_day_stop_profit_stop_first", stopped["label_hit_1r_before_stop_4d"] == 0 and stopped["label_stop_survival_4d"] == 0, "CRITICAL", f"survival={stopped['label_stop_survival_4d']},1R={stopped['label_hit_1r_before_stop_4d']}"))

    neg = base.copy()
    neg.loc[4, "close"] = 99
    neg["is_event_candidate"] = neg.apply(event_candidate, axis=1)
    negative = label_event_horizon(neg, 0, 4, commission_bps, slippage_bps, stop_multiple)
    rows.append(check_row("label_self_test_negative_horizon", negative["label_success_4d"] == 0, "CRITICAL", negative["label_success_4d"]))

    unavailable_base = base.copy()
    unavailable_base.loc[2, "entry_trigger"] = "20D_BREAKOUT"
    unavailable_base.loc[2, "trade_action"] = "ENTRY_ALLOWED"
    unavailable_base["is_event_candidate"] = unavailable_base.apply(event_candidate, axis=1)
    unavailable = label_event_horizon(unavailable_base, 2, 4, commission_bps, slippage_bps, stop_multiple)
    rows.append(check_row("label_self_test_unavailable_future", unavailable["label_status_4d"] == "UNAVAILABLE_FUTURE_WINDOW", "CRITICAL", unavailable["label_status_4d"]))
    return rows


def history_feature_self_test_checks() -> List[Dict]:
    rows = []
    train = pd.DataFrame(
        [
            {"signal_idx": 1, "entry_trigger": "A", "trend_regime": "UP", "entry_gate_status": "READY", "prediction_universe": "trigger_all", "label_success_20d": 1, "label_net_return_pct_20d": 4.0, "label_exit_reason_20d": "HORIZON_20D"},
            {"signal_idx": 2, "entry_trigger": "B", "trend_regime": "UP", "entry_gate_status": "FILTERED", "prediction_universe": "trigger_all", "label_success_20d": 0, "label_net_return_pct_20d": -2.0, "label_exit_reason_20d": "ATR_STOP_2X"},
            {"signal_idx": 3, "entry_trigger": "A", "trend_regime": "UP", "entry_gate_status": "READY", "prediction_universe": "trigger_all", "label_success_20d": 0, "label_net_return_pct_20d": -1.0, "label_exit_reason_20d": "ATR_STOP_2X"},
        ]
    )
    pred = train.iloc[[-1]].copy()
    train_aug, pred_aug = add_train_history_features(train, pred, "label_success_20d", "label_net_return_pct_20d", "label_exit_reason_20d")
    first_value = as_float(train_aug["hist_success_rate_trigger_trend"].iloc[0])
    third_value = as_float(train_aug["hist_success_rate_trigger_trend"].iloc[2])
    pred_value = as_float(pred_aug["hist_success_rate_trigger_trend"].iloc[0])
    rows.append(check_row("history_feature_first_row_uses_neutral_prior", abs(first_value - 0.5) < 1e-12, "CRITICAL", first_value))
    rows.append(check_row("history_feature_train_row_uses_prior_same_group", third_value > 0.5, "CRITICAL", third_value, "", "Third row should see the first A/UP result but not its own label."))
    rows.append(check_row("history_feature_prediction_uses_full_train_history", pred_value < third_value, "CRITICAL", pred_value, "", "Prediction rows may use the fully observed training window."))
    return rows


def build_quality_checks(
    feature_matrix: pd.DataFrame,
    labels: pd.DataFrame,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    threshold_policy: pd.DataFrame,
    oos_predictions: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    feature_contract: pd.DataFrame,
    latest: pd.DataFrame,
    trades: pd.DataFrame,
    commission_bps: float,
    slippage_bps: float,
    stop_multiple: float,
) -> pd.DataFrame:
    rows = []
    rows.append(check_row("sklearn_available", SKLEARN_IMPORT_ERROR is None, "CRITICAL", "ok" if SKLEARN_IMPORT_ERROR is None else str(SKLEARN_IMPORT_ERROR)))
    rows.append(check_row("trade_log_available", not trades.empty, "WARN", len(trades), "", "Trade log is an input audit dependency; labels are rebuilt independently from OHLC."))
    rows.extend(label_self_test_checks(commission_bps, slippage_bps, stop_multiple))
    rows.extend(history_feature_self_test_checks())
    feature_cols = [c for c in feature_matrix.columns if c in NUMERIC_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES]
    forbidden = [c for c in feature_cols if any(pattern in c.lower() for pattern in FORBIDDEN_FEATURE_PATTERNS)]
    rows.append(check_row("feature_allowlist_has_no_forbidden_columns", not forbidden, "CRITICAL", ",".join(forbidden) if forbidden else "ok"))
    rows.append(check_row("feature_allowlist_has_expanded_daily_features", len(feature_cols) >= 80, "CRITICAL", len(feature_cols), ">=80"))
    external_present = any(c in feature_matrix.columns for c in EXTERNAL_NUMERIC_FEATURES + EXTERNAL_BOOL_FEATURES + EXTERNAL_CATEGORICAL_FEATURES)
    rows.append(check_row("external_features_present", external_present, "WARN", "ok" if external_present else "EXTERNAL_FEATURES_MISSING"))
    event_count = int(labels["is_event_candidate"].sum()) if "is_event_candidate" in labels.columns else 0
    actionable_count = int(labels["is_actionable_entry_candidate"].sum()) if "is_actionable_entry_candidate" in labels.columns else 0
    trade_ready_count = int(labels["is_trade_ready_entry_candidate"].sum()) if "is_trade_ready_entry_candidate" in labels.columns else 0
    model_training_count = int(labels["is_model_training_candidate"].sum()) if "is_model_training_candidate" in labels.columns else trade_ready_count
    decision_entry_count = int(labels["is_decision_entry_candidate"].sum()) if "is_decision_entry_candidate" in labels.columns else trade_ready_count
    rows.append(check_row("event_candidate_count_positive", event_count > 0, "CRITICAL", event_count))
    rows.append(check_row("trigger_all_candidate_count_positive", actionable_count > 0, "CRITICAL", actionable_count))
    rows.append(check_row("trade_ready_entry_candidate_count_positive", trade_ready_count > 0, "CRITICAL", trade_ready_count))
    rows.append(check_row("model_training_candidate_count_positive", model_training_count > 0, "CRITICAL", model_training_count))
    rows.append(check_row("decision_entry_alias_matches_trade_ready", decision_entry_count == trade_ready_count, "CRITICAL", decision_entry_count, trade_ready_count))
    for horizon in HORIZONS:
        v2_cols = [
            f"label_stop_survival_{horizon}d",
            f"label_hit_1r_before_stop_{horizon}d",
            f"label_hit_2r_before_stop_{horizon}d",
            f"label_positive_return_{horizon}d",
            f"label_expected_r_{horizon}d",
            f"label_mfe_r_{horizon}d",
            f"label_mae_r_{horizon}d",
            f"label_time_to_stop_{horizon}d",
            f"label_ambiguous_stop_1r_same_day_{horizon}d",
            f"label_ambiguous_stop_2r_same_day_{horizon}d",
            f"label_gap_through_stop_{horizon}d",
            f"label_entry_gap_pct_{horizon}d",
            f"label_first_touch_type_{horizon}d",
            f"label_time_to_first_touch_{horizon}d",
            f"label_mfe_before_stop_{horizon}d",
            f"label_mae_before_profit_{horizon}d",
            f"label_event_regime_at_entry_{horizon}d",
            f"label_overlap_count_{horizon}d",
            f"sample_uniqueness_weight_{horizon}d",
        ]
        rows.append(check_row(f"v2_triple_barrier_label_columns_{horizon}d_present", all(c in labels.columns for c in v2_cols), "CRITICAL", ",".join([c for c in v2_cols if c in labels.columns])))
        available = int((labels[f"label_status_{horizon}d"] == "LABELED").sum())
        rows.append(check_row(f"available_labels_{horizon}d_at_least_30", available >= 30, "CRITICAL", available))
        labeled_weights = pd.to_numeric(labels.loc[labels[f"label_status_{horizon}d"].eq("LABELED"), f"sample_uniqueness_weight_{horizon}d"], errors="coerce")
        rows.append(
            check_row(
                f"sample_uniqueness_weight_{horizon}d_bounded",
                bool(not labeled_weights.empty and labeled_weights.between(0.0, 1.0, inclusive="both").all()),
                "CRITICAL",
                f"min={labeled_weights.min() if not labeled_weights.empty else 'NA'} max={labeled_weights.max() if not labeled_weights.empty else 'NA'}",
                "0..1",
            )
        )
        actionable_available = int(((labels[f"label_status_{horizon}d"] == "LABELED") & labels["is_actionable_entry_candidate"]).sum())
        trade_ready_available = int(((labels[f"label_status_{horizon}d"] == "LABELED") & labels["is_trade_ready_entry_candidate"]).sum())
        model_training_available = (
            int(((labels[f"label_status_{horizon}d"] == "LABELED") & labels["is_model_training_candidate"]).sum())
            if "is_model_training_candidate" in labels.columns
            else trade_ready_available
        )
        rows.append(check_row(f"available_trigger_all_labels_{horizon}d_at_least_30", actionable_available >= 30, "CRITICAL", actionable_available))
        rows.append(check_row(f"available_trade_ready_labels_{horizon}d_at_least_30", trade_ready_available >= 30, "CRITICAL", trade_ready_available))
        if horizon == 20:
            rows.append(
                check_row(
                    "available_model_training_labels_20d_at_least_750",
                    model_training_available >= MODEL_TRAINING_MIN_TSM_20D_LABELS,
                    "CRITICAL",
                    model_training_available,
                    MODEL_TRAINING_MIN_TSM_20D_LABELS,
                )
            )
    rows.append(check_row("walk_forward_metrics_non_empty", not metrics.empty, "CRITICAL", len(metrics)))
    rows.append(check_row("oos_prediction_ledger_non_empty", not oos_predictions.empty, "CRITICAL", len(oos_predictions)))
    if not oos_predictions.empty and {"date", "train_end_date"}.issubset(oos_predictions.columns):
        oos_dates = pd.to_datetime(oos_predictions["date"], errors="coerce")
        train_end = pd.to_datetime(oos_predictions["train_end_date"], errors="coerce")
        rows.append(check_row("oos_predictions_after_train_end_date", bool((oos_dates > train_end).all()), "CRITICAL", f"violations={int((oos_dates <= train_end).sum())}"))
    else:
        rows.append(check_row("oos_predictions_after_train_end_date", False, "CRITICAL", "missing_columns"))
    rows.append(check_row("equal_frequency_calibration_summary_available", bool(not calibration_summary.empty and calibration_summary.get("binning", pd.Series(dtype=str)).astype(str).eq("equal_frequency").any()), "CRITICAL", len(calibration_summary)))
    rows.append(check_row("model_comparison_non_empty", not comparison.empty, "CRITICAL", len(comparison)))
    if not comparison.empty and "candidate_scope" in comparison.columns:
        scopes = set(comparison["candidate_scope"].dropna().astype(str))
        trigger_manifest = (
            fold_manifest[fold_manifest["candidate_scope"].astype(str).eq("trigger_all")].copy()
            if not fold_manifest.empty and {"candidate_scope", "status"}.issubset(fold_manifest.columns)
            else pd.DataFrame()
        )
        trigger_statuses = set(trigger_manifest["status"].dropna().astype(str)) if not trigger_manifest.empty else set()
        trigger_documented_skip = bool(trigger_statuses and all(status.startswith("SKIPPED") for status in trigger_statuses))
        rows.append(check_row("model_comparison_has_context_scope", "context_all" in scopes, "CRITICAL", ",".join(sorted(scopes))))
        rows.append(check_row("model_comparison_has_entry_research_scope", "entry_research" in scopes, "CRITICAL", ",".join(sorted(scopes))))
        rows.append(
            check_row(
                "model_comparison_has_trigger_scope_or_documented_skip",
                "trigger_all" in scopes or trigger_documented_skip,
                "CRITICAL",
                ",".join(sorted(scopes)),
                details=f"trigger_fold_statuses={','.join(sorted(trigger_statuses)) if trigger_statuses else 'missing'}",
            )
        )
        rows.append(check_row("model_comparison_has_trigger_context_scopes", {"trigger_all", "context_all", "entry_research"}.issubset(scopes), "WARN", ",".join(sorted(scopes))))
        rows.append(check_row("model_comparison_has_trade_ready_scope", "trade_ready_entry" in scopes, "WARN", ",".join(sorted(scopes)), "", "Trade-ready may remain diagnostic-only until OOS events reach 100."))
        decision_scope = comparison[comparison.get("decision_scope_eligible", pd.Series(False, index=comparison.index)).map(to_bool)].copy()
        bad_decision_scope = decision_scope[
            ~(
                decision_scope["candidate_scope"].astype(str).eq("trade_ready_entry")
                & pd.to_numeric(decision_scope["horizon_days"], errors="coerce").eq(20)
            )
        ]
        rows.append(check_row("prediction_decision_scope_is_20d_trade_ready_only", bad_decision_scope.empty, "CRITICAL", len(bad_decision_scope), "0"))
        if not decision_scope.empty:
            selected_series = pd.to_numeric(
                decision_scope["selected_oos_event_count"] if "selected_oos_event_count" in decision_scope.columns else pd.Series(0, index=decision_scope.index),
                errors="coerce",
            ).fillna(0)
            ci_series = pd.to_numeric(
                decision_scope["selected_signal_expectancy_ci_lower_pct"] if "selected_signal_expectancy_ci_lower_pct" in decision_scope.columns else pd.Series(np.nan, index=decision_scope.index),
                errors="coerce",
            ).dropna()
            selected_ok = bool((selected_series >= MIN_SELECTED_OOS_EVENTS).all())
            ci_ok = bool((ci_series > MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT).all()) if not ci_series.empty else False
            rows.append(check_row("decision_scope_selected_oos_gate_applied", selected_ok, "WARN", int(selected_series.min())))
            rows.append(check_row("decision_scope_expectancy_ci_gate_applied", ci_ok, "WARN", "ok" if ci_ok else "blocked"))
    else:
        rows.append(check_row("model_comparison_has_context_scope", False, "CRITICAL", "missing"))
        rows.append(check_row("model_comparison_has_entry_research_scope", False, "CRITICAL", "missing"))
        rows.append(check_row("model_comparison_has_trigger_scope_or_documented_skip", False, "CRITICAL", "missing"))
        rows.append(check_row("model_comparison_has_trigger_context_scopes", False, "WARN", "missing"))
        rows.append(check_row("model_comparison_has_trade_ready_scope", False, "WARN", "missing"))
    if not metrics.empty and {"train_end_idx", "test_start_idx", "horizon_days"}.issubset(metrics.columns):
        observed_gap = pd.to_numeric(metrics["test_start_idx"], errors="coerce") - pd.to_numeric(metrics["train_end_idx"], errors="coerce") - 1
        horizon = pd.to_numeric(metrics["horizon_days"], errors="coerce")
        rows.append(check_row("purged_walk_forward_gap_covers_label_horizon", bool((observed_gap >= horizon).all()), "CRITICAL", f"min_gap={observed_gap.min()}, max_horizon={horizon.max()}"))
    else:
        rows.append(check_row("purged_walk_forward_gap_covers_label_horizon", False, "CRITICAL", "missing_metric_columns"))
    rows.append(
        check_row(
            "threshold_policy_not_from_test",
            bool(not threshold_policy.empty and "threshold_source" in threshold_policy.columns and not threshold_policy["threshold_source"].astype(str).str.contains("TEST", case=False, na=False).any()),
            "CRITICAL",
            "ok" if not threshold_policy.empty else "missing",
        )
    )
    rows.append(check_row("fold_manifest_available", not fold_manifest.empty, "CRITICAL", len(fold_manifest)))
    if not fold_manifest.empty and {"status", "embargo_days", "horizon_days"}.issubset(fold_manifest.columns):
        used = fold_manifest[fold_manifest["status"].astype(str).isin(["USED", "SPARSE_DIAGNOSTIC_USED"])].copy()
        rows.append(check_row("purged_event_time_split_used", not used.empty, "CRITICAL", len(used)))
        rows.append(check_row("fold_embargo_covers_horizon", bool((pd.to_numeric(fold_manifest["embargo_days"], errors="coerce") >= pd.to_numeric(fold_manifest["horizon_days"], errors="coerce")).all()), "CRITICAL", "ok"))
    rows.append(check_row("feature_contract_available", not feature_contract.empty, "CRITICAL", len(feature_contract)))
    if not feature_contract.empty and {"role", "leakage_policy"}.issubset(feature_contract.columns):
        label_allowed = feature_contract[(feature_contract["role"] == "label") & ~feature_contract["leakage_policy"].astype(str).str.contains("blocked", na=False)]
        rows.append(check_row("feature_contract_blocks_label_features", label_allowed.empty, "CRITICAL", len(label_allowed)))
    latest_map = dict(zip(latest["field"], latest["value"])) if not latest.empty else {}
    rows.append(check_row("latest_prediction_snapshot_available", bool(latest_map), "CRITICAL", "ok" if latest_map else "missing"))
    use_status = latest_map.get("prediction_use_status", "UNKNOWN")
    rows.append(check_row("prediction_decision_usable_20d", use_status == "DECISION_SUPPORT_ALLOWED", "WARN", use_status, "", "False means display-only, not a pipeline failure."))
    return pd.DataFrame(rows)


def write_report(
    outdir: Path,
    labels: pd.DataFrame,
    scope_stats: pd.DataFrame,
    comparison: pd.DataFrame,
    latest: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    latest_map = dict(zip(latest["field"], latest["value"])) if not latest.empty else {}
    critical = quality[quality["severity"] == "CRITICAL"] if not quality.empty else pd.DataFrame()
    critical_passed = bool(critical["passed"].all()) if not critical.empty else False
    lines = [
        "# TSMC Prediction Accuracy Report",
        "",
        f"- Critical quality checks passed: {critical_passed}",
        f"- Event candidates: {int(labels['is_event_candidate'].sum()) if not labels.empty else 0}",
        f"- Trade-ready entry candidates: {int(labels['is_trade_ready_entry_candidate'].sum()) if 'is_trade_ready_entry_candidate' in labels.columns else 0}",
        f"- Model-training research candidates: {int(labels['is_model_training_candidate'].sum()) if 'is_model_training_candidate' in labels.columns else 0}",
        f"- Trigger-all candidates: {int(labels['is_actionable_entry_candidate'].sum()) if 'is_actionable_entry_candidate' in labels.columns else 0}",
        f"- Latest prediction scope: {latest_map.get('prediction_scope_used', 'NA')}",
        f"- Latest prediction status: {latest_map.get('prediction_signal_status', 'NA')}",
        f"- Prediction use status: {latest_map.get('prediction_use_status', 'NA')}",
        f"- 20D probability / threshold: {latest_map.get('p_success_20d', 'NA')} / {latest_map.get('threshold_20d', 'NA')}",
        f"- 20D fixed decision score: {latest_map.get('decision_score_20d', 'NA')}",
        f"- 20D stop hit / 1R / 2R / expected R: {latest_map.get('p_stop_hit_20d', 'NA')} / {latest_map.get('p_hit_1r_20d', 'NA')} / {latest_map.get('p_hit_2r_20d', 'NA')} / {latest_map.get('expected_r_20d', 'NA')}",
        f"- 20D 80% probability interval: {latest_map.get('p_success_lower_80_20d', 'NA')} ~ {latest_map.get('p_success_upper_80_20d', 'NA')}",
        f"- 20D effective OOS events: {latest_map.get('effective_oos_event_count_20d', 'NA')}",
        f"- Decision permission / final decision: {latest_map.get('decision_permission', 'NA')} / {latest_map.get('final_trade_decision', 'NA')}",
        f"- Paper alpha weight: {latest_map.get('paper_alpha_weight_pct', 'NA')}%",
        f"- Model quality block reasons: {latest_map.get('model_quality_block_reasons', 'NA')}",
        f"- 60D probability / threshold: {latest_map.get('p_success_60d', 'NA')} / {latest_map.get('threshold_60d', 'NA')}",
        "",
        "## Candidate Scope Base Evidence",
        "",
        "| Scope | Horizon | Labeled | Success Rate | Mean Net Return | Mean Expected R | Median Expected R | ATR Stop Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base_scope = scope_stats[
        (scope_stats["group_type"] == "prediction_universe")
        & (scope_stats["group_value"].isin(["trade_ready_entry", "entry_research", "trigger_all", "context_all"]))
    ]
    if base_scope.empty:
        lines.append("| NA | NA | 0 | NA | NA | NA | NA |")
    else:
        for _, row in base_scope.sort_values(["horizon_days", "group_value"]).iterrows():
            lines.append(
                f"| {row['group_value']} | {int(row['horizon_days'])} | {int(row['labeled_count'])} | "
                f"{row['success_rate_pct']:.2f}% | {row['mean_net_return_pct']:.2f}% | "
                f"{row.get('mean_expected_r', np.nan):.2f} | {row.get('median_expected_r', np.nan):.2f} | {row['atr_stop_rate_pct']:.2f}% |"
            )
    lines.extend(
        [
        "",
        "## Model Comparison",
        "",
        "| Scope | Horizon | Model | Policy | Quality Pass | OOS | Selected OOS | Brier Improvement | ECE | PR AUC | Selected Minus Rule | Selected CI Lower | Threshold IQR |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if comparison.empty:
        lines.append("| NA | NA | NA | False | 0 | NA | NA | NA | NA |")
    else:
        for _, row in comparison.sort_values(["candidate_scope", "horizon_days", "rank_score"], ascending=[True, True, False]).iterrows():
            lines.append(
                f"| {row['candidate_scope']} | {int(row['horizon_days'])} | {row['model_name']} | {row.get('model_policy', 'UNKNOWN')} | {row['prediction_quality_pass']} | "
                f"{int(row['oos_event_count'])} | {int(row.get('selected_oos_event_count', 0))} | {row['brier_improvement_pct']:.2f}% | "
                f"{row['ece']:.4f} | {row['pr_auc']:.4f} | {row.get('selected_minus_rule_all_pct', np.nan):.2f}% | "
                f"{row.get('selected_signal_expectancy_ci_lower_pct', np.nan):.2f}% | {row.get('threshold_iqr', np.nan):.4f} |"
            )
    failed = quality[~quality["passed"]] if not quality.empty else pd.DataFrame()
    lines.extend(["", "## Failed or Advisory Checks", "", "| Check | Severity | Value | Details |", "|---|---|---:|---|"])
    if failed.empty:
        lines.append("| none | NA | NA | NA |")
    else:
        for _, row in failed.iterrows():
            lines.append(f"| {row['check']} | {row['severity']} | {row['value']} | {row.get('details', '')} |")
    lines.extend(
        [
            "",
            "## Operating Rule",
            "- Prediction is a meta-label overlay on top of the existing rule engine.",
            "- It does not change trade_action by itself.",
            "- If prediction_use_status is not DECISION_SUPPORT_ALLOWED, probabilities are display-only.",
            "",
            "This report is research tooling, not investment advice.",
        ]
    )
    (outdir / "tsm_prediction_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_reliability_report(outdir: Path, latest: pd.DataFrame, audit: pd.DataFrame, calibration: pd.DataFrame) -> None:
    latest_map = dict(zip(latest["field"], latest["value"])) if not latest.empty else {}
    lines = [
        "# TSMC Prediction Reliability Report",
        "",
        f"- Latest scope: {latest_map.get('prediction_scope_used', 'NA')}",
        f"- Latest signal status: {latest_map.get('prediction_signal_status', 'NA')}",
        f"- Latest use status: {latest_map.get('prediction_use_status', 'NA')}",
        f"- 20D p_success: {latest_map.get('p_success_20d', 'NA')}",
        f"- 20D p_stop_survival: {latest_map.get('p_stop_survival_20d', 'NA')}",
        f"- 20D p_positive_given_survival: {latest_map.get('p_positive_given_survival_20d', 'NA')}",
        f"- 20D 80% interval: {latest_map.get('p_success_lower_80_20d', 'NA')} ~ {latest_map.get('p_success_upper_80_20d', 'NA')}",
        "",
        "## Model Audit",
        "",
        "| Scope | Horizon | Model | Pass | OOS | Block Reasons | Brier Improvement | ECE | PR AUC |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    if audit.empty:
        lines.append("| NA | NA | NA | False | 0 | missing | NA | NA | NA |")
    else:
        for _, row in audit.iterrows():
            lines.append(
                f"| {row['candidate_scope']} | {int(row['horizon_days'])} | {row['model_name']} | {row['prediction_quality_pass']} | "
                f"{int(row['oos_event_count'])} | {row['quality_block_reasons']} | {row['brier_improvement_pct']:.2f}% | {row['ece']:.4f} | {row['pr_auc']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Reliability Bins",
            "",
            "| Scope | Horizon | Model | Binning | Fold | Bin | N | Mean Predicted | Observed | Abs Error |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if calibration.empty:
        lines.append("| NA | NA | NA | NA | NA | NA | 0 | NA | NA | NA |")
    else:
        best_keys = set()
        if not audit.empty:
            for _, row in audit.sort_values(["candidate_scope", "horizon_days", "brier_improvement_pct"], ascending=[True, True, False]).groupby(["candidate_scope", "horizon_days"]).head(1).iterrows():
                best_keys.add((row["candidate_scope"], row["horizon_days"], row["model_name"]))
        display = calibration[
            calibration.apply(lambda r: (r["candidate_scope"], r["horizon_days"], r["model_name"]) in best_keys, axis=1)
        ] if best_keys else calibration.head(0)
        for _, row in display.sort_values(["candidate_scope", "horizon_days", "model_name", "binning", "fold_id", "bin_id"]).iterrows():
            lines.append(
                f"| {row['candidate_scope']} | {int(row['horizon_days'])} | {row['model_name']} | {row.get('binning', 'fixed_width')} | {int(row['fold_id'])} | {int(row['bin_id'])} | "
                f"{int(row['n'])} | {row['mean_predicted_probability']:.4f} | {row['observed_success_rate']:.4f} | {row['abs_calibration_error']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Operating Rule",
            "- A model is decision-support eligible only when OOS count, Brier improvement, ECE, PR AUC, and expectancy-fold gates pass.",
            "- If the latest row is not trade_ready_entry, the prediction overlay remains display-only.",
            "",
            "This report is research tooling, not investment advice.",
        ]
    )
    (outdir / "tsm_prediction_reliability_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_thresholds(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TSM meta-label prediction accuracy outputs.")
    parser.add_argument("--config", default="config/tsm_research.toml")
    parser.add_argument("--signals", default="tsm_price_rule_output/tsm_daily_algorithmic_signals.csv")
    parser.add_argument("--trade-log", default="tsm_price_rule_output/tsm_backtest_trade_log.csv")
    parser.add_argument("--risk-policy", default="tsm_price_rule_output/tsm_risk_policy_daily.csv")
    parser.add_argument("--enriched", default="output/tsm_daily_10y_enriched.csv")
    parser.add_argument("--external-features", default="", help="Optional symbol/date external feature CSV produced by tsm_external_feature_engine.py.")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stop-multiple", type=float, default=2.0)
    parser.add_argument("--train-days", type=int, default=756)
    parser.add_argument("--validation-days", type=int, default=40)
    parser.add_argument("--test-days", type=int, default=252)
    parser.add_argument("--step-days", type=int, default=252)
    parser.add_argument("--gap-days", type=int, default=60)
    parser.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--default-threshold", type=float, default=0.60)
    parser.add_argument("--min-validation-trades", type=int, default=10)
    parser.add_argument("--calibration-bins", type=int, default=5)
    parser.add_argument("--schema-strict", action="store_true")
    parser.add_argument("--prediction-policy", choices=["conservative", "research"], default="conservative")
    args = parser.parse_args()
    return apply_config_defaults(args, parser, load_run_config(args.config))


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if SKLEARN_IMPORT_ERROR is not None:
        raise SystemExit(f"scikit-learn is required. Install dependencies with: python -m pip install -r requirements.txt. Error: {SKLEARN_IMPORT_ERROR}")

    external_path = Path(args.external_features) if str(args.external_features).strip() else None
    signals, trades = load_inputs(Path(args.signals), Path(args.risk_policy), Path(args.trade_log), Path(args.enriched), external_path, symbol="TSM")
    labels = build_label_dataset(signals, args.commission_bps, args.slippage_bps, args.stop_multiple)
    feature_matrix = build_feature_matrix(signals, labels)
    scope_stats = build_candidate_scope_stats(labels)
    label_diagnostics = build_label_diagnostics(labels)

    metric_parts = []
    threshold_parts = []
    calibration_parts = []
    feature_selection_parts = []
    oos_prediction_parts = []
    fold_manifest_parts = []
    for candidate_scope_name, candidate_col in [
        ("trade_ready_entry", "is_trade_ready_entry_candidate"),
        ("entry_research", "is_model_training_candidate"),
        ("trigger_all", "is_actionable_entry_candidate"),
        ("context_all", "is_event_candidate"),
    ]:
        for horizon in HORIZONS:
            metrics, thresholds, calibration, feature_selection, oos_predictions, fold_manifest = evaluate_prediction_stream(
                feature_matrix,
                horizon,
                candidate_scope_name,
                candidate_col,
                args.train_days,
                args.validation_days,
                args.test_days,
                args.step_days,
                args.gap_days,
                parse_thresholds(args.thresholds),
                args.min_validation_trades,
                args.default_threshold,
                args.calibration_bins,
            )
            metric_parts.append(metrics)
            threshold_parts.append(thresholds)
            calibration_parts.append(calibration)
            feature_selection_parts.append(feature_selection)
            oos_prediction_parts.append(oos_predictions)
            fold_manifest_parts.append(fold_manifest)

    metrics = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    threshold_policy = pd.concat(threshold_parts, ignore_index=True) if threshold_parts else pd.DataFrame()
    calibration = pd.concat(calibration_parts, ignore_index=True) if calibration_parts else pd.DataFrame()
    feature_selection_report = pd.concat(feature_selection_parts, ignore_index=True) if feature_selection_parts else pd.DataFrame()
    oos_predictions = pd.concat(oos_prediction_parts, ignore_index=True) if oos_prediction_parts else pd.DataFrame()
    fold_manifest = pd.concat(fold_manifest_parts, ignore_index=True) if fold_manifest_parts else pd.DataFrame()
    calibration_summary = aggregate_calibration_summary(calibration)
    comparison = aggregate_model_comparison(metrics)
    model_audit = build_model_audit(metrics, comparison, calibration, threshold_policy)
    latest = build_latest_snapshot(feature_matrix, comparison, threshold_policy)
    feature_contract = build_feature_contract(feature_matrix)
    schema_quality = build_schema_quality_checks(
        labels=labels,
        feature_matrix=feature_matrix,
        comparison=comparison,
        latest_snapshot=latest,
        strict=bool(args.schema_strict),
    )
    policy_audit = build_policy_audit_rows(args.prediction_policy, latest, comparison)
    quality = build_quality_checks(
        feature_matrix,
        labels,
        metrics,
        comparison,
        threshold_policy,
        oos_predictions,
        calibration_summary,
        fold_manifest,
        feature_contract,
        latest,
        trades,
        args.commission_bps,
        args.slippage_bps,
        args.stop_multiple,
    )
    quality = pd.concat([quality, schema_quality], ignore_index=True)

    labels.to_csv(outdir / "tsm_prediction_label_dataset.csv", index=False)
    feature_matrix.to_csv(outdir / "tsm_prediction_feature_matrix.csv", index=False)
    scope_stats.to_csv(outdir / "tsm_prediction_candidate_scope_stats.csv", index=False)
    label_diagnostics.to_csv(outdir / "tsm_prediction_label_diagnostics.csv", index=False)
    metrics.to_csv(outdir / "tsm_prediction_walk_forward_metrics.csv", index=False)
    oos_predictions.to_csv(outdir / "tsm_prediction_oos_predictions.csv", index=False)
    comparison.to_csv(outdir / "tsm_prediction_model_comparison.csv", index=False)
    calibration.to_csv(outdir / "tsm_prediction_calibration_bins.csv", index=False)
    calibration_summary.to_csv(outdir / "tsm_prediction_calibration_summary.csv", index=False)
    threshold_policy.to_csv(outdir / "tsm_prediction_threshold_policy.csv", index=False)
    feature_selection_report.to_csv(outdir / "tsm_prediction_feature_selection_report.csv", index=False)
    feature_contract.to_csv(outdir / "tsm_prediction_feature_contract.csv", index=False)
    fold_manifest.to_csv(outdir / "tsm_prediction_fold_manifest.csv", index=False)
    policy_audit.to_csv(outdir / "tsm_prediction_policy_audit.csv", index=False)
    schema_quality.to_csv(outdir / "tsm_schema_quality_checks.csv", index=False)
    model_audit.to_csv(outdir / "tsm_prediction_model_audit.csv", index=False)
    latest.to_csv(outdir / "tsm_latest_prediction_snapshot.csv", index=False)
    quality.to_csv(outdir / "tsm_prediction_quality_checks.csv", index=False)
    write_report(outdir, labels, scope_stats, comparison, latest, quality)
    write_reliability_report(outdir, latest, model_audit, calibration)

    print("완료: prediction outputs =", outdir.resolve())
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()

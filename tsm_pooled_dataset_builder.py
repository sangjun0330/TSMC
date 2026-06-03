#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a pooled cross-sectional prediction dataset.

The default config builds a one-symbol TSM pool from the current outputs. To add
more symbols, pass a CSV with columns:
symbol,signals,risk_policy,trade_log,enriched

Outputs:
- tsm_prediction_pooled_label_dataset.csv
- tsm_prediction_pooled_feature_matrix.csv
- tsm_prediction_pooled_scope_stats.csv
- tsm_prediction_pooled_schema.csv
- tsm_prediction_pooled_quality_checks.csv
- tsm_prediction_pooled_dataset_report.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from tsm_core.universe import market_region_for_symbol
from tsm_prediction_engine import (
    BOOL_FEATURES,
    CATEGORICAL_FEATURES,
    HORIZONS,
    NUMERIC_FEATURES,
    add_label_overlap_uniqueness,
    as_float,
    build_candidate_scope_stats,
    build_feature_matrix,
    cost_rate,
    load_inputs,
    pct,
    to_bool,
    valid_price,
)


SEMICONDUCTOR_UNIVERSE = [
    ("TSM", "foundry"),
    ("NVDA", "ai_accelerator"),
    ("AMD", "ai_accelerator"),
    ("AVGO", "ai_accelerator"),
    ("ASML", "semicap"),
    ("AMAT", "semicap"),
    ("LRCX", "semicap"),
    ("KLAC", "semicap"),
    ("MU", "memory"),
    ("005930.KS", "memory_foundry_idm"),
    ("000660.KS", "memory_storage"),
    ("QCOM", "semiconductor"),
    ("SMH", "semiconductor_etf"),
    ("SOXX", "semiconductor_etf"),
]


REQUIRED_DECISION_SYMBOL_COUNT = 12
ENTRY_SCORE_GRID = (70.0, 72.5, 75.0, 77.5, 80.0)
WATCHLIST_SCORE_GRID = (60.0, 62.5, 65.0, 67.5, 70.0)
OBSERVATION_SCORE_GRID = (55.0, 57.5, 60.0, 62.5, 65.0)
OVEREXTENDED_SMA50_GRID = (0.10, 0.15, 0.20, 0.25, 0.30)
BASELINE_ENTRY_SCORE = 75.0
BASELINE_WATCHLIST_SCORE = 65.0
BASELINE_OBSERVATION_SCORE = 60.0
BASELINE_OVEREXTENDED_SMA50 = 0.15


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def default_config() -> pd.DataFrame:
    rows: List[Dict] = []
    for symbol, group in SEMICONDUCTOR_UNIVERSE:
        if symbol == "TSM":
            rows.append(
                {
                    "symbol": symbol,
                    "symbol_group": group,
                    "signals": "tsm_price_rule_output/tsm_daily_algorithmic_signals.csv",
                    "risk_policy": "tsm_price_rule_output/tsm_risk_policy_daily.csv",
                    "trade_log": "tsm_price_rule_output/tsm_backtest_trade_log.csv",
                    "enriched": "output/tsm_daily_10y_enriched.csv",
                }
            )
        else:
            rows.append(
                {
                    "symbol": symbol,
                    "symbol_group": group,
                    "signals": f"tsm_price_rule_output/universe/{symbol}/tsm_daily_algorithmic_signals.csv",
                    "risk_policy": f"tsm_price_rule_output/universe/{symbol}/tsm_risk_policy_daily.csv",
                    "trade_log": f"tsm_price_rule_output/universe/{symbol}/tsm_backtest_trade_log.csv",
                    "enriched": f"output/universe/{symbol}/tsm_daily_10y_enriched.csv",
                }
            )
    return pd.DataFrame(rows)


def read_config(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return default_config()
    config = strip_bom_columns(pd.read_csv(path))
    required = ["symbol", "signals", "risk_policy", "trade_log", "enriched"]
    missing = [c for c in required if c not in config.columns]
    if missing:
        raise ValueError(f"pooled config missing columns: {missing}")
    if "symbol_group" not in config.columns:
        config["symbol_group"] = "semiconductor"
    optional = [
        c
        for c in ["enabled", "paper_enabled", "strict_eligible", "eligibility_status", "is_decision_universe", "decision_scope", "training_scope", "market_region"]
        if c in config.columns
    ]
    return config[["symbol", "symbol_group", *required[1:], *optional]].copy()


def _unavailable_label_fields(horizon: int, status: str) -> Dict[str, object]:
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


def build_pooled_label_dataset(
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
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=base_cols)

    dates = df["date"].to_numpy()
    open_values = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=float)
    high_values = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low_values = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    close_values = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    atr_values = pd.to_numeric(df["atr_14"], errors="coerce").to_numpy(dtype=float)
    trend_values = df["trend_regime"].fillna("UNKNOWN").astype(str).to_numpy() if "trend_regime" in df.columns else np.full(n, "UNKNOWN", dtype=object)
    vol_values = df["vol_regime"].fillna("UNKNOWN").astype(str).to_numpy() if "vol_regime" in df.columns else np.full(n, "UNKNOWN", dtype=object)
    event_values = df["is_event_candidate"].map(to_bool).to_numpy(dtype=bool)
    cr = cost_rate(commission_bps, slippage_bps)

    rows: list[Dict[str, object]] = []
    for idx, base in enumerate(df[base_cols].to_dict("records")):
        out = dict(base)
        for horizon in HORIZONS:
            if not event_values[idx]:
                out.update(_unavailable_label_fields(horizon, "NOT_EVENT"))
                continue

            entry_idx = idx + 1
            horizon_exit_idx = idx + horizon
            if entry_idx >= n:
                out.update(_unavailable_label_fields(horizon, "UNAVAILABLE_NEXT_OPEN"))
                continue
            if horizon_exit_idx >= n:
                out.update(_unavailable_label_fields(horizon, "UNAVAILABLE_FUTURE_WINDOW"))
                continue

            entry_price = as_float(open_values[entry_idx])
            atr = as_float(atr_values[idx])
            if not valid_price(entry_price) or atr <= 0:
                out.update(_unavailable_label_fields(horizon, "INVALID_ENTRY_DATA"))
                continue

            stop_price = entry_price - stop_multiple * atr
            risk_per_share = entry_price - stop_price
            if risk_per_share <= 0:
                out.update(_unavailable_label_fields(horizon, "INVALID_RISK_DISTANCE"))
                continue

            target_1r = entry_price + stop_multiple * atr
            target_2r = entry_price + 2.0 * stop_multiple * atr
            exit_idx = horizon_exit_idx
            exit_price = as_float(close_values[horizon_exit_idx])
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
            signal_close = as_float(close_values[idx])
            entry_gap_pct = entry_price / signal_close - 1.0 if valid_price(signal_close) else np.nan
            event_regime_at_entry = f"{trend_values[entry_idx]}|{vol_values[entry_idx]}"

            for j in range(entry_idx, horizon_exit_idx + 1):
                low = as_float(low_values[j])
                high = as_float(high_values[j])
                open_price = as_float(open_values[j])
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
                out.update(_unavailable_label_fields(horizon, "INVALID_EXIT_DATA"))
                continue

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

            out.update(
                {
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
                    f"label_entry_date_{horizon}d": dates[entry_idx],
                    f"label_exit_date_{horizon}d": dates[exit_idx],
                    f"label_entry_price_{horizon}d": entry_price,
                    f"label_exit_price_{horizon}d": exit_price,
                    f"label_stop_price_{horizon}d": stop_price,
                    f"label_1r_price_{horizon}d": target_1r,
                    f"label_2r_price_{horizon}d": target_2r,
                    f"label_holding_trading_days_{horizon}d": exit_idx - entry_idx,
                }
            )
        rows.append(out)
    return add_label_overlap_uniqueness(pd.DataFrame(rows))


def build_symbol_dataset(
    row: pd.Series,
    commission_bps: float,
    slippage_bps: float,
    stop_multiple: float,
    external_features_path: Path | None = None,
    intraday_features_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbol = str(row["symbol"]).upper()
    symbol_group = str(row.get("symbol_group", "semiconductor"))
    market_region = market_region_for_symbol(symbol, "", row.get("market_region"))
    decision_flag = row.get("is_decision_universe", row.get("enabled", "true"))
    is_decision_universe = str(decision_flag).strip().lower() in {"true", "1", "yes", "y", "t"}
    decision_scope = str(row.get("decision_scope", "top10"))
    training_scope = str(row.get("training_scope", "universal_research_pool"))
    for col in ["signals", "risk_policy", "trade_log", "enriched"]:
        if not Path(row[col]).exists():
            raise FileNotFoundError(f"{symbol} missing {col}: {row[col]}")
    signals, _ = load_inputs(
        Path(row["signals"]),
        Path(row["risk_policy"]),
        Path(row["trade_log"]),
        Path(row["enriched"]),
        external_features_path,
        intraday_features_path,
        symbol=symbol,
    )
    labels = build_pooled_label_dataset(signals, commission_bps, slippage_bps, stop_multiple)
    features = build_feature_matrix(signals, labels)
    scope_stats = build_candidate_scope_stats(labels)
    for frame in [labels, features, scope_stats]:
        frame.insert(0, "symbol", symbol)
        frame.insert(1, "symbol_group", symbol_group)
        frame.insert(2, "market_region", market_region)
        frame["is_decision_universe"] = is_decision_universe
        frame["decision_scope"] = decision_scope
        frame["training_scope"] = training_scope
    return labels, features, scope_stats


def add_cross_sectional_features(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty or not {"symbol", "date"}.issubset(features.columns):
        return features.copy()
    out = features.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "market_region" not in out.columns:
        out["market_region"] = out["symbol"].map(lambda symbol: market_region_for_symbol(symbol))
    out["market_region"] = out["market_region"].fillna("").astype(str).map(
        lambda value: market_region_for_symbol("", "", value)
    )
    group_keys = ["market_region", "date"]
    group_values = [out[key] for key in group_keys]
    rank_cols = [
        "return_20d",
        "return_60d",
        "return_126d",
        "rsi_14",
        "atr_percentile_252d",
        "drawdown_from_ath",
        "volume_ratio_20",
        "score_price_algo_total",
    ]
    for col in rank_cols:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        out[f"cs_rank_{col}"] = values.groupby(group_values, dropna=False).rank(pct=True, method="average")
        mean = values.groupby(group_values, dropna=False).transform("mean")
        std = values.groupby(group_values, dropna=False).transform("std").replace(0, np.nan)
        out[f"cs_z_{col}"] = (values - mean) / std

    if "return_20d" in out.columns:
        ret20 = pd.to_numeric(out["return_20d"], errors="coerce")
        universe_median = ret20.groupby(group_values, dropna=False).transform("median")
        out["relative_return_vs_universe_median_20d"] = ret20 - universe_median
        for benchmark in ["SMH", "SOXX"]:
            bench_col = f"{benchmark.lower()}_return_20d"
            if bench_col not in out.columns:
                bench = (
                    out[out["symbol"].astype(str).str.upper().eq(benchmark)]
                    .groupby("date", as_index=False)["return_20d"]
                    .last()
                    .rename(columns={"return_20d": bench_col})
                )
                out = out.merge(bench, on="date", how="left")
            out[f"relative_return_vs_{benchmark.lower()}_20d"] = ret20.to_numpy() - pd.to_numeric(out[bench_col], errors="coerce")
    if {"close", "sma_50"}.issubset(out.columns):
        above_50 = pd.to_numeric(out["close"], errors="coerce") >= pd.to_numeric(out["sma_50"], errors="coerce")
        out["universe_above_sma50_ratio"] = above_50.astype(float).groupby(group_values, dropna=False).transform("mean")
    if {"close", "sma_200"}.issubset(out.columns):
        above_200 = pd.to_numeric(out["close"], errors="coerce") >= pd.to_numeric(out["sma_200"], errors="coerce")
        out["universe_above_sma200_ratio"] = above_200.astype(float).groupby(group_values, dropna=False).transform("mean")
    if "is_model_training_candidate" in out.columns:
        model_training = out["is_model_training_candidate"].astype(str).str.lower().isin(["true", "1", "yes"])
        out["universe_model_training_candidate_ratio"] = model_training.astype(float).groupby(group_values, dropna=False).transform("mean")
    if "is_decision_entry_candidate" in out.columns:
        decision_entry = out["is_decision_entry_candidate"].astype(str).str.lower().isin(["true", "1", "yes"])
        out["universe_decision_entry_candidate_ratio"] = decision_entry.astype(float).groupby(group_values, dropna=False).transform("mean")
    return out


def add_signal_cluster_features(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty or not {"symbol", "date"}.issubset(features.columns):
        return features.copy()
    out = features.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["symbol", "date", "signal_idx" if "signal_idx" in out.columns else "date"]).reset_index(drop=True)
    if "is_event_candidate" in out.columns:
        event_signal = out["is_event_candidate"].astype(str).str.lower().isin(["true", "1", "yes"])
    elif "entry_trigger" in out.columns:
        event_signal = out["entry_trigger"].astype(str).ne("NONE")
    else:
        event_signal = pd.Series(False, index=out.index)
    if "is_decision_entry_candidate" in out.columns:
        strict_signal = out["is_decision_entry_candidate"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        strict_signal = event_signal.copy()
    out["_pooled_event_signal"] = event_signal.astype(float)
    out["_pooled_strict_signal"] = strict_signal.astype(float)
    out["days_since_prev_signal"] = np.nan
    for _, group in out.groupby("symbol", sort=False):
        dated = group.dropna(subset=["date"]).copy()
        if dated.empty:
            continue
        event_dates = dated["date"].where(out.loc[dated.index, "_pooled_event_signal"].astype(bool)).ffill().shift()
        out.loc[dated.index, "days_since_prev_signal"] = (dated["date"] - event_dates).dt.days
        rolling_basis = out.loc[dated.index, ["date", "_pooled_event_signal", "_pooled_strict_signal"]].set_index("date")
        for window in [20, 60, 120]:
            out.loc[dated.index, f"signal_count_{window}d"] = (
                rolling_basis["_pooled_event_signal"].rolling(f"{window}D", min_periods=1).sum().to_numpy()
            )
        for window in [20, 60]:
            out.loc[dated.index, f"strict_signal_count_{window}d"] = (
                rolling_basis["_pooled_strict_signal"].rolling(f"{window}D", min_periods=1).sum().to_numpy()
            )
    return out.drop(columns=["_pooled_event_signal", "_pooled_strict_signal"])


def add_regime_interaction_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for a, b in [
        ("entry_trigger", "trend_regime"),
        ("entry_trigger", "vol_regime"),
        ("trend_regime", "drawdown_bucket"),
        ("risk_state", "vol_regime"),
        ("candidate_tier", "trend_regime"),
        ("candidate_tier", "vol_regime"),
        ("news_primary_cause_type", "trend_regime"),
        ("news_primary_cause_type", "vol_regime"),
    ]:
        if a in out.columns and b in out.columns:
            out[f"{a}__x__{b}"] = out[a].fillna("UNKNOWN").astype(str) + "__" + out[b].fillna("UNKNOWN").astype(str)
    return out


def add_pooled_feature_engineering(features: pd.DataFrame) -> pd.DataFrame:
    out = add_cross_sectional_features(features)
    out = add_signal_cluster_features(out)
    out = add_regime_interaction_features(out)
    return out


def schema_rows(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    feature_sets = {
        "numeric_feature": set(NUMERIC_FEATURES),
        "bool_feature": set(BOOL_FEATURES),
        "categorical_feature": set(CATEGORICAL_FEATURES),
    }
    for col in features.columns:
        if col in {"symbol", "symbol_group", "date", "signal_idx"}:
            role = "entity_key" if col in {"symbol", "symbol_group"} else "time_key"
        elif col.startswith("label_"):
            role = "label"
        else:
            role = "metadata"
            for candidate_role, values in feature_sets.items():
                if col in values:
                    role = candidate_role
                    break
            if role == "metadata":
                lower = str(col).lower()
                if lower in {"fx_data_available", "vix_data_available", "tsmc_earnings_pre_5d_window", "tsmc_earnings_post_5d_window", "tsmc_earnings_event_day"}:
                    role = "bool_feature"
                elif lower.startswith(("cs_rank_", "cs_z_", "relative_return_vs_", "signal_count_", "strict_signal_count_", "universe_", "tsmc_", "market_", "peer_", "fx_", "vix_", "external_")) or lower in {"days_since_prev_signal", "days_since_tsmc_earnings"}:
                    role = "numeric_feature"
                elif "__x__" in lower:
                    role = "categorical_feature"
        rows.append(
            {
                "column": col,
                "role": role,
                "dtype": str(features[col].dtype),
                "missing_rate": float(features[col].isna().mean()) if len(features) else np.nan,
                "non_null_count": int(features[col].notna().sum()) if len(features) else 0,
            }
        )
    for col in labels.columns:
        if col.startswith("label_") and col not in features.columns:
            rows.append(
                {
                    "column": col,
                    "role": "label_detail",
                    "dtype": str(labels[col].dtype),
                    "missing_rate": float(labels[col].isna().mean()) if len(labels) else np.nan,
                    "non_null_count": int(labels[col].notna().sum()) if len(labels) else 0,
                }
            )
    return pd.DataFrame(rows).sort_values(["role", "column"]).reset_index(drop=True)


def check_row(check: str, passed: bool, severity: str, value, details: str = "") -> Dict:
    return {"check": check, "passed": bool(passed), "severity": severity, "value": value, "details": details}


def to_bool_series(values: pd.Series, default: bool = False) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"}).fillna(default)


def decision_symbols(config: pd.DataFrame) -> list[str]:
    if config.empty or "symbol" not in config.columns:
        return []
    if "is_decision_universe" in config.columns:
        mask = to_bool_series(config["is_decision_universe"])
    elif "enabled" in config.columns:
        mask = to_bool_series(config["enabled"])
    else:
        return []
    return sorted(config.loc[mask, "symbol"].astype(str).str.upper().unique().tolist())


def _symbol_set(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "symbol" not in frame.columns:
        return set()
    return {str(symbol).upper() for symbol in frame["symbol"].dropna().tolist()}


def assign_date_split(date_value) -> str:
    date = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(date):
        return "unknown"
    if date.year <= 2022:
        return "train_2016_2022"
    if date.year == 2023:
        return "validation_2023"
    if date.year == 2024:
        return "test_2024"
    return "final_holdout_2025_2026"


def build_split_manifest(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty or not {"symbol", "date"}.issubset(features.columns):
        return pd.DataFrame(
            columns=[
                "fold_id",
                "split",
                "symbol_count",
                "row_count",
                "trade_ready_event_count",
                "selected_candidate_count",
                "weak_symbol_groups",
                "start_date",
                "end_date",
            ]
        )
    optional_cols = [
        c
        for c in [
            "symbol_group",
            "is_decision_universe",
            "is_trade_ready_entry_candidate",
            "is_decision_entry_candidate",
        ]
        if c in features.columns
    ]
    frame = features[["symbol", "date", *optional_cols]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["split"] = frame["date"].map(assign_date_split)
    if "is_trade_ready_entry_candidate" not in frame.columns:
        frame["is_trade_ready_entry_candidate"] = False
    if "is_decision_entry_candidate" not in frame.columns:
        frame["is_decision_entry_candidate"] = frame["is_trade_ready_entry_candidate"]
    frame["is_trade_ready_entry_candidate"] = frame["is_trade_ready_entry_candidate"].astype(bool)
    frame["is_decision_entry_candidate"] = frame["is_decision_entry_candidate"].astype(bool)

    rows = []
    for split, group in frame.groupby("split", dropna=False):
        weak_symbol_groups = ""
        if "symbol_group" in group.columns:
            group_counts = (
                group.groupby("symbol_group", dropna=False)
                .agg(
                    trade_ready_event_count=("is_trade_ready_entry_candidate", "sum"),
                    selected_candidate_count=("is_decision_entry_candidate", "sum"),
                )
                .reset_index()
            )
            weak = group_counts[
                (group_counts["trade_ready_event_count"].astype(int) == 0)
                | (group_counts["selected_candidate_count"].astype(int) == 0)
            ]["symbol_group"].dropna().astype(str)
            weak_symbol_groups = "|".join(sorted(set(weak)))
        rows.append(
            {
                "fold_id": str(split),
                "split": split,
                "symbol_count": int(group["symbol"].nunique()),
                "row_count": int(len(group)),
                "trade_ready_event_count": int(group["is_trade_ready_entry_candidate"].sum()),
                "selected_candidate_count": int(group["is_decision_entry_candidate"].sum()),
                "weak_symbol_groups": weak_symbol_groups,
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


TSM_DIRECT_SYMBOLS = {"TSM"}
TSM_LIKE_FOUNDRY_IDM_SYMBOLS = {"TSM", "UMC", "GFS", "INTC", "STM", "TSEM", "005930.KS"}
TSM_MEMORY_SUPPLY_SYMBOLS = {"MU", "005930.KS", "000660.KS"}
TSM_SUPPLY_CHAIN_SYMBOLS = {"ASML", "AMAT", "LRCX", "KLAC", "TER", "ENTG", "AMKR", "PLAB"}
SEMI_BREADTH_REGIME_SYMBOLS = {"SMH", "SOXX", "SOXQ", "XSD", "PSI", "FTXL"}
TSM_LIKE_BASE_GROUP_WEIGHTS = {
    "TSM_DIRECT": 1.00,
    "TSM_LIKE_FOUNDRY_IDM": 0.80,
    "TSM_MEMORY_SUPPLY": 0.65,
    "TSM_SUPPLY_CHAIN": 0.55,
    "SEMI_BREADTH_REGIME": 0.40,
    "OTHER_SEMI": 0.25,
}


def tsm_like_group_for_symbol(symbol: str, symbol_group: str = "") -> str:
    value = str(symbol).upper()
    group = str(symbol_group).lower()
    if value in TSM_DIRECT_SYMBOLS:
        return "TSM_DIRECT"
    if value in TSM_LIKE_FOUNDRY_IDM_SYMBOLS or group in {"foundry", "foundry_idm", "memory_foundry_idm"}:
        return "TSM_LIKE_FOUNDRY_IDM"
    if value in TSM_MEMORY_SUPPLY_SYMBOLS or group in {"memory", "memory_storage"}:
        return "TSM_MEMORY_SUPPLY"
    if value in TSM_SUPPLY_CHAIN_SYMBOLS or group in {"semicap", "semicap_osat"}:
        return "TSM_SUPPLY_CHAIN"
    if value in SEMI_BREADTH_REGIME_SYMBOLS or group in {"semiconductor_etf", "semi_breadth_regime"}:
        return "SEMI_BREADTH_REGIME"
    return "OTHER_SEMI"


def build_sample_audit(config: pd.DataFrame, labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    label_frame = labels.copy()
    feature_frame = features.copy()
    if "date" in label_frame.columns:
        label_frame["date"] = pd.to_datetime(label_frame["date"], errors="coerce")
    for _, cfg in config.iterrows():
        symbol = str(cfg.get("symbol", "")).upper()
        symbol_group = str(cfg.get("symbol_group", "semiconductor"))
        symbol_labels = label_frame[label_frame.get("symbol", pd.Series(dtype=object)).astype(str).str.upper().eq(symbol)].copy() if not label_frame.empty and "symbol" in label_frame.columns else pd.DataFrame()
        symbol_features = feature_frame[feature_frame.get("symbol", pd.Series(dtype=object)).astype(str).str.upper().eq(symbol)].copy() if not feature_frame.empty and "symbol" in feature_frame.columns else pd.DataFrame()
        loaded = bool(not symbol_features.empty)
        status_20d = symbol_labels.get("label_status_20d", pd.Series(dtype=object)).astype(str).eq("LABELED") if not symbol_labels.empty else pd.Series(dtype=bool)
        trade_ready = symbol_labels.get("is_trade_ready_entry_candidate", pd.Series(False, index=symbol_labels.index)).astype(bool) if not symbol_labels.empty else pd.Series(dtype=bool)
        model_training_col = "is_model_training_candidate" if "is_model_training_candidate" in symbol_labels.columns else "is_trade_ready_entry_candidate"
        model_training = symbol_labels.get(model_training_col, pd.Series(False, index=symbol_labels.index)).astype(bool) if not symbol_labels.empty else pd.Series(dtype=bool)
        split_dates = pd.to_datetime(symbol_labels.get("date", pd.Series(pd.NaT, index=symbol_labels.index)), errors="coerce") if not symbol_labels.empty else pd.Series(dtype="datetime64[ns]")
        test_holdout = split_dates.ge(pd.Timestamp("2024-01-01")) if not symbol_labels.empty else pd.Series(dtype=bool)
        tsm_like_group = tsm_like_group_for_symbol(symbol, symbol_group)
        base_weight = TSM_LIKE_BASE_GROUP_WEIGHTS[tsm_like_group]
        strict_eligible = cfg.get("strict_eligible", np.nan)
        rows.append(
            {
                "symbol": symbol,
                "symbol_group": symbol_group,
                "market_region": market_region_for_symbol(symbol, "", cfg.get("market_region")),
                "is_decision_universe": str(cfg.get("is_decision_universe", "")).strip(),
                "decision_scope": str(cfg.get("decision_scope", "")),
                "training_scope": str(cfg.get("training_scope", "")),
                "strict_eligible": strict_eligible,
                "loaded": loaded,
                "row_count": int(len(symbol_features)),
                "trade_ready_20d_labeled": int((trade_ready & status_20d).sum()) if not symbol_labels.empty else 0,
                "model_training_20d_labeled": int((model_training & status_20d).sum()) if not symbol_labels.empty else 0,
                "test_holdout_trade_ready_20d": int((trade_ready & status_20d & test_holdout).sum()) if not symbol_labels.empty else 0,
                "tsm_like_group": tsm_like_group,
                "tsm_like_weight_mean": base_weight if loaded else np.nan,
                "tsm_like_effective_n_contribution": float(base_weight * base_weight * int((trade_ready & status_20d).sum())) if not symbol_labels.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _combined_feature_labels(labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    key_cols = [c for c in ["symbol", "date", "signal_idx"] if c in features.columns and c in labels.columns]
    label_cols = [
        c
        for c in labels.columns
        if c.startswith("label_") or c in {"symbol", "date", "signal_idx"}
    ]
    if labels.empty or not key_cols or not label_cols:
        return features.copy()
    label_part = labels[list(dict.fromkeys(label_cols))].copy()
    return features.merge(label_part, on=key_cols, how="left", suffixes=("", "_label"))


def _nonempty_trigger(values: pd.Series) -> pd.Series:
    return ~values.astype(str).str.strip().str.upper().isin({"", "NONE", "NO_ENTRY_TRIGGER", "NA", "N/A", "NULL"})


def _candidate_masks(frame: pd.DataFrame, entry: float, watch: float, observation: float, overextended_line: float) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    score = pd.to_numeric(frame.get("score_price_algo_total", pd.Series(np.nan, index=frame.index)), errors="coerce")
    dist50 = pd.to_numeric(frame.get("dist_close_sma_50_pct", pd.Series(np.nan, index=frame.index)), errors="coerce")
    trend = to_bool_series(frame.get("algo_trend_up_loose", pd.Series(False, index=frame.index))).reindex(frame.index, fill_value=False)
    high_vol = to_bool_series(frame.get("algo_vol_high", pd.Series(False, index=frame.index))).reindex(frame.index, fill_value=False)
    extreme_vol = to_bool_series(frame.get("algo_vol_extreme", pd.Series(False, index=frame.index))).reindex(frame.index, fill_value=False)
    trigger = _nonempty_trigger(frame.get("entry_trigger", pd.Series("", index=frame.index))).reindex(frame.index, fill_value=False)
    overextended = trend & high_vol & (dist50 > overextended_line)
    trade = (score >= entry) & trigger & (~extreme_vol) & (~overextended)
    research = (score >= watch) & trend & (~overextended)
    observe = (score >= observation) & trend
    return trade.fillna(False), research.fillna(False), observe.fillna(False), overextended.fillna(False)


def _mean_pct(frame: pd.DataFrame, col: str) -> float:
    if col not in frame.columns or frame.empty:
        return np.nan
    return float(pd.to_numeric(frame[col], errors="coerce").mean())


def _selected_max_drawdown_pct(selected: pd.DataFrame, return_col: str) -> float:
    if selected.empty or return_col not in selected.columns or "date" not in selected.columns:
        return np.nan
    daily = selected.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["_ret"] = pd.to_numeric(daily[return_col], errors="coerce") / 100.0
    grouped = daily.dropna(subset=["date"]).groupby("date")["_ret"].mean().sort_index()
    if grouped.empty:
        return np.nan
    equity = (1.0 + grouped.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min() * 100.0)


def _selected_sharpe(selected: pd.DataFrame, return_col: str) -> float:
    if selected.empty or return_col not in selected.columns or "date" not in selected.columns:
        return np.nan
    daily = selected.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["_ret"] = pd.to_numeric(daily[return_col], errors="coerce") / 100.0
    grouped = daily.dropna(subset=["date"]).groupby("date")["_ret"].mean().dropna().sort_index()
    std = grouped.std(ddof=0)
    return float(grouped.mean() / std * np.sqrt(252)) if std and std > 0 else np.nan


def _latest_changed_symbols(
    latest_frame: pd.DataFrame,
    trade: pd.Series,
    research: pd.Series,
    observe: pd.Series,
    baseline_trade: pd.Series,
    baseline_research: pd.Series,
    baseline_observe: pd.Series,
) -> str:
    if latest_frame.empty or "symbol" not in latest_frame.columns:
        return ""
    changed = latest_frame[
        (trade.reindex(latest_frame.index, fill_value=False) != baseline_trade.reindex(latest_frame.index, fill_value=False))
        | (research.reindex(latest_frame.index, fill_value=False) != baseline_research.reindex(latest_frame.index, fill_value=False))
        | (observe.reindex(latest_frame.index, fill_value=False) != baseline_observe.reindex(latest_frame.index, fill_value=False))
    ]
    return "|".join(sorted(changed["symbol"].astype(str).str.upper().tolist()))


def _latest_rows_for_change_detection(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns or "date" not in frame.columns:
        return pd.DataFrame()
    work = frame[["symbol", "date"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    latest_index = work.sort_values(["symbol", "date"]).groupby(work["symbol"].astype(str).str.upper(), dropna=False).tail(1).index
    return frame.loc[latest_index].copy()


def build_rule_threshold_sensitivity(labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    frame = _combined_feature_labels(labels, features)
    if frame.empty:
        return pd.DataFrame()
    rows: list[Dict[str, object]] = []
    unique_dates = max(int(pd.to_datetime(frame.get("date", pd.Series(dtype=object)), errors="coerce").nunique()), 1)
    latest_frame = _latest_rows_for_change_detection(frame)
    baseline_trade, baseline_research, baseline_observe, _ = _candidate_masks(
        frame,
        BASELINE_ENTRY_SCORE,
        BASELINE_WATCHLIST_SCORE,
        BASELINE_OBSERVATION_SCORE,
        BASELINE_OVEREXTENDED_SMA50,
    )
    for entry in ENTRY_SCORE_GRID:
        for watch in WATCHLIST_SCORE_GRID:
            for observation in OBSERVATION_SCORE_GRID:
                for over_line in OVEREXTENDED_SMA50_GRID:
                    trade, research, observe, overextended = _candidate_masks(frame, entry, watch, observation, over_line)
                    selected = frame[trade].copy()
                    stop_survival = pd.to_numeric(selected.get("label_stop_survival_20d", pd.Series(dtype=float)), errors="coerce")
                    stop_hit_rate = float((1.0 - stop_survival).mean() * 100.0) if len(stop_survival.dropna()) else np.nan
                    rows.append(
                        {
                            "entry_score_threshold": entry,
                            "watchlist_score_threshold": watch,
                            "observation_score_threshold": observation,
                            "overextended_sma50_pct": over_line * 100.0,
                            "is_current_baseline": bool(
                                entry == BASELINE_ENTRY_SCORE
                                and watch == BASELINE_WATCHLIST_SCORE
                                and observation == BASELINE_OBSERVATION_SCORE
                                and over_line == BASELINE_OVEREXTENDED_SMA50
                            ),
                            "selected_symbol_count": int(selected["symbol"].nunique()) if "symbol" in selected.columns else 0,
                            "trade_candidate_count": int(trade.sum()),
                            "research_candidate_count": int(research.sum()),
                            "observation_candidate_count": int(observe.sum()),
                            "overextended_event_count": int(overextended.sum()),
                            "mean_fwd_return_20d_pct": _mean_pct(selected, "label_net_return_pct_20d"),
                            "mean_fwd_return_60d_pct": _mean_pct(selected, "label_net_return_pct_60d"),
                            "stop_hit_rate_20d_pct": stop_hit_rate,
                            "mean_expected_r_20d": _mean_pct(selected, "label_expected_r_20d"),
                            "max_drawdown_pct": _selected_max_drawdown_pct(selected, "label_net_return_pct_20d"),
                            "sharpe_20d_zero_rf": _selected_sharpe(selected, "label_net_return_pct_20d"),
                            "turnover_events_per_date": float(trade.sum() / unique_dates),
                            "latest_affected_symbols": _latest_changed_symbols(
                                latest_frame,
                                trade,
                                research,
                                observe,
                                baseline_trade,
                                baseline_research,
                                baseline_observe,
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def build_rule_threshold_sensitivity_summary(sensitivity: pd.DataFrame) -> pd.DataFrame:
    if sensitivity.empty:
        return pd.DataFrame(
            [
                {
                    "recommendation": "KEEP_CURRENT",
                    "baseline_entry_score_threshold": BASELINE_ENTRY_SCORE,
                    "baseline_watchlist_score_threshold": BASELINE_WATCHLIST_SCORE,
                    "baseline_observation_score_threshold": BASELINE_OBSERVATION_SCORE,
                    "baseline_overextended_sma50_pct": BASELINE_OVEREXTENDED_SMA50 * 100.0,
                    "reason": "no_sensitivity_rows",
                }
            ]
        )
    baseline_rows = sensitivity[sensitivity["is_current_baseline"].astype(bool)]
    baseline = baseline_rows.iloc[0] if not baseline_rows.empty else pd.Series(dtype=object)
    baseline_sharpe = float(pd.to_numeric(pd.Series([baseline.get("sharpe_20d_zero_rf", np.nan)]), errors="coerce").iloc[0])
    baseline_return = float(pd.to_numeric(pd.Series([baseline.get("mean_fwd_return_20d_pct", np.nan)]), errors="coerce").iloc[0])
    baseline_dd = float(pd.to_numeric(pd.Series([baseline.get("max_drawdown_pct", np.nan)]), errors="coerce").iloc[0])
    baseline_count = int(pd.to_numeric(pd.Series([baseline.get("trade_candidate_count", 0)]), errors="coerce").fillna(0).iloc[0])
    candidates = sensitivity.copy()
    candidates["_trade_count"] = pd.to_numeric(candidates["trade_candidate_count"], errors="coerce").fillna(0)
    candidates["_sharpe"] = pd.to_numeric(candidates["sharpe_20d_zero_rf"], errors="coerce")
    candidates["_return"] = pd.to_numeric(candidates["mean_fwd_return_20d_pct"], errors="coerce")
    candidates["_dd"] = pd.to_numeric(candidates["max_drawdown_pct"], errors="coerce")
    min_count = max(30, int(baseline_count * 0.5))
    viable = candidates[candidates["_trade_count"] >= min_count].dropna(subset=["_sharpe", "_return"])
    if viable.empty or pd.isna(baseline_sharpe):
        best = baseline
        recommendation = "KEEP_CURRENT"
        reason = "insufficient_viable_alternative"
    else:
        viable["_score"] = viable["_sharpe"] + 0.02 * viable["_return"] + 0.005 * viable["_dd"].fillna(0.0)
        best = viable.sort_values(["_score", "_trade_count"], ascending=[False, False]).iloc[0]
        materially_better = (
            float(best["_sharpe"]) > baseline_sharpe + 0.10
            and float(best["_return"]) >= baseline_return
            and (pd.isna(baseline_dd) or pd.isna(best["_dd"]) or float(best["_dd"]) >= baseline_dd - 5.0)
        )
        if not materially_better:
            recommendation = "KEEP_CURRENT"
            reason = "no_materially_better_grid_point"
        elif float(best["entry_score_threshold"]) < BASELINE_ENTRY_SCORE or float(best["overextended_sma50_pct"]) > BASELINE_OVEREXTENDED_SMA50 * 100.0:
            recommendation = "REVIEW_LOOSER"
            reason = "alternative_has_better_risk_adjusted_metrics_with_looser_entry_or_chase_line"
        elif float(best["entry_score_threshold"]) > BASELINE_ENTRY_SCORE or float(best["overextended_sma50_pct"]) < BASELINE_OVEREXTENDED_SMA50 * 100.0:
            recommendation = "REVIEW_TIGHTER"
            reason = "alternative_has_better_risk_adjusted_metrics_with_tighter_entry_or_chase_line"
        else:
            recommendation = "KEEP_CURRENT"
            reason = "best_grid_point_matches_current_threshold_family"
    return pd.DataFrame(
        [
            {
                "recommendation": recommendation,
                "reason": reason,
                "baseline_entry_score_threshold": BASELINE_ENTRY_SCORE,
                "baseline_watchlist_score_threshold": BASELINE_WATCHLIST_SCORE,
                "baseline_observation_score_threshold": BASELINE_OBSERVATION_SCORE,
                "baseline_overextended_sma50_pct": BASELINE_OVEREXTENDED_SMA50 * 100.0,
                "baseline_trade_candidate_count": baseline_count,
                "baseline_mean_fwd_return_20d_pct": baseline_return,
                "baseline_sharpe_20d_zero_rf": baseline_sharpe,
                "baseline_max_drawdown_pct": baseline_dd,
                "best_entry_score_threshold": best.get("entry_score_threshold", np.nan),
                "best_watchlist_score_threshold": best.get("watchlist_score_threshold", np.nan),
                "best_observation_score_threshold": best.get("observation_score_threshold", np.nan),
                "best_overextended_sma50_pct": best.get("overextended_sma50_pct", np.nan),
                "best_trade_candidate_count": best.get("trade_candidate_count", np.nan),
                "best_mean_fwd_return_20d_pct": best.get("mean_fwd_return_20d_pct", np.nan),
                "best_sharpe_20d_zero_rf": best.get("sharpe_20d_zero_rf", np.nan),
                "best_max_drawdown_pct": best.get("max_drawdown_pct", np.nan),
                "config_mutation_allowed": False,
            }
        ]
    )


def build_rule_threshold_sensitivity_quality_checks(sensitivity: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    expected_grid_size = len(ENTRY_SCORE_GRID) * len(WATCHLIST_SCORE_GRID) * len(OBSERVATION_SCORE_GRID) * len(OVEREXTENDED_SMA50_GRID)
    baseline_present = bool(not sensitivity.empty and sensitivity.get("is_current_baseline", pd.Series(dtype=bool)).astype(bool).any())
    summary_recommendation = str(summary.iloc[0].get("recommendation", "")) if not summary.empty else ""
    return pd.DataFrame(
        [
            check_row("rule_threshold_sensitivity_rows_positive", not sensitivity.empty, "CRITICAL", len(sensitivity)),
            check_row("rule_threshold_sensitivity_grid_complete", len(sensitivity) == expected_grid_size, "CRITICAL", len(sensitivity), f"expected={expected_grid_size}"),
            check_row("rule_threshold_sensitivity_baseline_present", baseline_present, "CRITICAL", baseline_present),
            check_row("rule_threshold_sensitivity_recommendation_valid", summary_recommendation in {"KEEP_CURRENT", "REVIEW_LOOSER", "REVIEW_TIGHTER"}, "CRITICAL", summary_recommendation),
            check_row("rule_threshold_sensitivity_no_config_mutation", True, "CRITICAL", "advisory_only"),
        ]
    )


def build_quality_checks(config: pd.DataFrame, labels: pd.DataFrame, features: pd.DataFrame, schema: pd.DataFrame, failures: pd.DataFrame, split_manifest: pd.DataFrame) -> pd.DataFrame:
    loaded_symbol_count = int(features["symbol"].nunique()) if "symbol" in features.columns and not features.empty else 0
    configured_decision_symbols = decision_symbols(config)
    loaded_symbols = _symbol_set(features)
    labeled_symbols = _symbol_set(labels)
    missing_feature_symbols = sorted(set(configured_decision_symbols) - loaded_symbols)
    missing_label_symbols = sorted(set(configured_decision_symbols) - labeled_symbols)
    trade_ready_20d = 0
    model_training_20d = 0
    if not labels.empty and {"is_trade_ready_entry_candidate", "label_status_20d"}.issubset(labels.columns):
        trade_ready_20d = int((labels["is_trade_ready_entry_candidate"].astype(bool) & labels["label_status_20d"].eq("LABELED")).sum())
    if not labels.empty and "label_status_20d" in labels.columns:
        training_col = "is_model_training_candidate" if "is_model_training_candidate" in labels.columns else "is_trade_ready_entry_candidate"
        if training_col in labels.columns:
            model_training_20d = int((labels[training_col].astype(bool) & labels["label_status_20d"].eq("LABELED")).sum())
    rows = [
        check_row("pooled_config_symbol_count_positive", config["symbol"].nunique() > 0, "CRITICAL", config["symbol"].nunique()),
        check_row(
            "pooled_decision_symbol_count_eq_required",
            not configured_decision_symbols or len(configured_decision_symbols) == REQUIRED_DECISION_SYMBOL_COUNT,
            "CRITICAL",
            len(configured_decision_symbols) if configured_decision_symbols else "not_configured",
            f"required={REQUIRED_DECISION_SYMBOL_COUNT}",
        ),
        check_row(
            "pooled_decision_symbols_loaded_all",
            not configured_decision_symbols or (not missing_feature_symbols and not missing_label_symbols),
            "CRITICAL",
            "PASS" if not missing_feature_symbols and not missing_label_symbols else f"missing_features={','.join(missing_feature_symbols)};missing_labels={','.join(missing_label_symbols)}",
            "Every decision universe symbol must be present in pooled feature and label datasets.",
        ),
        check_row("pooled_loaded_symbol_count_at_least_10", loaded_symbol_count >= 10, "CRITICAL", loaded_symbol_count, "Need at least 10 loaded symbols for pooled semiconductor ML."),
        check_row("pooled_training_scope_recorded", "training_scope" in features.columns if not features.empty else False, "CRITICAL", "training_scope"),
        check_row("pooled_decision_scope_recorded", "decision_scope" in features.columns if not features.empty else False, "CRITICAL", "decision_scope"),
        check_row("pooled_input_failures_absent", failures.empty, "WARN", len(failures), "Missing per-symbol outputs are expected until universe runner is executed."),
        check_row("pooled_labels_non_empty", not labels.empty, "CRITICAL", len(labels)),
        check_row("pooled_features_non_empty", not features.empty, "CRITICAL", len(features)),
        check_row("pooled_schema_non_empty", not schema.empty, "CRITICAL", len(schema)),
        check_row("pooled_has_symbol_columns", {"symbol", "symbol_group"}.issubset(features.columns) and {"symbol", "symbol_group"}.issubset(labels.columns), "CRITICAL", "symbol,symbol_group"),
        check_row("pooled_trade_ready_20d_labeled_at_least_500", trade_ready_20d >= 500, "CRITICAL", trade_ready_20d, "Minimum target for prediction decision research."),
        check_row("pooled_model_training_20d_labeled_at_least_10000", model_training_20d >= 10000, "CRITICAL", model_training_20d, "Minimum target for pooled model training research."),
        check_row("research_target_trade_ready_20d_labeled_at_least_10000", trade_ready_20d >= 10000, "WARN", trade_ready_20d, "Expansion target for robust decision evidence."),
        check_row("research_target_model_training_20d_labeled_at_least_100000", model_training_20d >= 100000, "WARN", model_training_20d, "Expansion target for robust pooled model training."),
        check_row("pooled_date_based_split_manifest_available", not split_manifest.empty, "CRITICAL", len(split_manifest), "Splits are assigned by calendar date, never by symbol-random split."),
    ]
    if not labels.empty:
        for horizon in HORIZONS:
            status_col = f"label_status_{horizon}d"
            labeled = int(labels[status_col].eq("LABELED").sum()) if status_col in labels.columns else 0
            rows.append(check_row(f"pooled_labeled_events_{horizon}d_positive", labeled > 0, "CRITICAL", labeled))
    if not features.empty:
        duplicate_keys = int(features.duplicated(["symbol", "date", "signal_idx"]).sum()) if {"symbol", "date", "signal_idx"}.issubset(features.columns) else -1
        rows.append(check_row("pooled_symbol_date_signal_unique", duplicate_keys == 0, "CRITICAL", duplicate_keys))
        feature_count = int(schema["role"].isin(["numeric_feature", "bool_feature", "categorical_feature"]).sum()) if not schema.empty else 0
        rows.append(check_row("pooled_feature_count_at_least_80", feature_count >= 80, "CRITICAL", feature_count, "Current per-symbol schema should expose the expanded feature set."))
        external_cols = [c for c in features.columns if str(c).lower().startswith(("tsmc_", "market_", "peer_", "fx_", "vix_", "external_"))]
        rows.append(check_row("pooled_external_features_present", bool(external_cols), "WARN", len(external_cols), "EXTERNAL_FEATURES_MISSING if zero."))
        intraday_cols = [
            c
            for c in features.columns
            if str(c).lower().startswith(("hourly_", "model_minute_", "execution_minute_", "m5_", "m1_", "intraday_", "timeframe_"))
        ]
        rows.append(check_row("pooled_intraday_features_present", bool(intraday_cols), "WARN", len(intraday_cols), "INTRADAY_FEATURES_MISSING if zero."))
    return pd.DataFrame(rows)


def write_report(outdir: Path, config: pd.DataFrame, labels: pd.DataFrame, features: pd.DataFrame, quality: pd.DataFrame, failures: pd.DataFrame, split_manifest: pd.DataFrame) -> None:
    lines = [
        "# Top10 Pooled Prediction Dataset Report",
        "",
        f"- Symbols: {', '.join(config['symbol'].astype(str).str.upper())}",
        f"- Loaded symbols: {features['symbol'].nunique() if 'symbol' in features.columns and not features.empty else 0}",
        f"- Label rows: {len(labels)}",
        f"- Feature rows: {len(features)}",
        "",
        "## Quality",
        "",
        "| Check | Passed | Severity | Value |",
        "|---|---:|---|---:|",
    ]
    for _, row in quality.iterrows():
        lines.append(f"| {row['check']} | {row['passed']} | {row['severity']} | {row['value']} |")
    lines.extend(["", "## Date-Based Splits", "", "| Split | Symbols | Rows | Start | End |", "|---|---:|---:|---|---|"])
    if split_manifest.empty:
        lines.append("| NA | 0 | 0 | NA | NA |")
    else:
        for _, row in split_manifest.iterrows():
            lines.append(f"| {row['split']} | {int(row['symbol_count'])} | {int(row['row_count'])} | {row['start_date']} | {row['end_date']} |")
    lines.extend(["", "## Missing Symbol Inputs", "", "| Symbol | Status | Details |", "|---|---|---|"])
    if failures.empty:
        lines.append("| none | OK |  |")
    else:
        for _, row in failures.iterrows():
            lines.append(f"| {row['symbol']} | {row['status']} | {row['details']} |")
    lines.extend(
        [
            "",
            "## Expansion Contract",
            "- Add more symbols by providing a config CSV with the same rule-engine output paths per symbol.",
            "- Pooled training must split by date and may group/calibrate by symbol; symbol-random split is not allowed.",
            "- Default research split: train 2016-2022, validation 2023, test 2024, final holdout 2025-2026.",
            "",
            "This report is research tooling, not investment advice.",
        ]
    )
    (outdir / "tsm_prediction_pooled_dataset_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pooled cross-sectional prediction dataset.")
    parser.add_argument("--config", default="")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stop-multiple", type=float, default=2.0)
    parser.add_argument("--external-features", default="", help="Optional symbol/date external feature CSV produced by tsm_external_feature_engine.py.")
    parser.add_argument("--intraday-features", default="", help="Optional symbol/date intraday feature CSV produced by tsm_intraday_feature_engine.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config) if args.config else None
    config = read_config(config_path)
    external_features_path = Path(args.external_features) if str(args.external_features).strip() else None
    intraday_features_path = Path(args.intraday_features) if str(args.intraday_features).strip() else None

    label_parts = []
    feature_parts = []
    scope_parts = []
    failures: List[Dict] = []
    for _, row in config.iterrows():
        if "strict_eligible" in row.index:
            strict_value = str(row.get("strict_eligible", "")).strip().lower()
            if strict_value in {"false", "0", "no"}:
                failures.append(
                    {
                        "symbol": str(row.get("symbol", "UNKNOWN")).upper(),
                        "status": "SKIPPED_STRICT_INELIGIBLE",
                        "details": str(row.get("eligibility_status", "STRICT_ELIGIBILITY_FALSE")),
                    }
                )
                continue
        try:
            labels, features, scope_stats = build_symbol_dataset(
                row,
                args.commission_bps,
                args.slippage_bps,
                args.stop_multiple,
                external_features_path,
                intraday_features_path,
            )
        except Exception as exc:
            failures.append({"symbol": str(row.get("symbol", "UNKNOWN")).upper(), "status": "SKIPPED_INPUT_UNAVAILABLE", "details": str(exc)})
            continue
        label_parts.append(labels)
        feature_parts.append(features)
        scope_parts.append(scope_stats)

    labels = pd.concat(label_parts, ignore_index=True) if label_parts else pd.DataFrame()
    features = pd.concat(feature_parts, ignore_index=True) if feature_parts else pd.DataFrame()
    features = add_pooled_feature_engineering(features) if not features.empty else features
    scope_stats = pd.concat(scope_parts, ignore_index=True) if scope_parts else pd.DataFrame()
    schema = schema_rows(features, labels) if not features.empty else pd.DataFrame()
    failure_df = pd.DataFrame(failures)
    split_manifest = build_split_manifest(features)
    sample_audit = build_sample_audit(config, labels, features)
    quality = build_quality_checks(config, labels, features, schema, failure_df, split_manifest)
    threshold_sensitivity = build_rule_threshold_sensitivity(labels, features)
    threshold_sensitivity_summary = build_rule_threshold_sensitivity_summary(threshold_sensitivity)
    threshold_sensitivity_quality = build_rule_threshold_sensitivity_quality_checks(threshold_sensitivity, threshold_sensitivity_summary)

    labels.to_csv(outdir / "tsm_prediction_pooled_label_dataset.csv", index=False)
    features.to_csv(outdir / "tsm_prediction_pooled_feature_matrix.csv", index=False)
    scope_stats.to_csv(outdir / "tsm_prediction_pooled_scope_stats.csv", index=False)
    schema.to_csv(outdir / "tsm_prediction_pooled_schema.csv", index=False)
    split_manifest.to_csv(outdir / "tsm_prediction_pooled_split_manifest.csv", index=False)
    sample_audit.to_csv(outdir / "tsm_prediction_pooled_sample_audit.csv", index=False)
    failure_df.to_csv(outdir / "tsm_prediction_pooled_input_failures.csv", index=False)
    quality.to_csv(outdir / "tsm_prediction_pooled_quality_checks.csv", index=False)
    threshold_sensitivity.to_csv(outdir / "tsm_rule_threshold_sensitivity.csv", index=False)
    threshold_sensitivity_summary.to_csv(outdir / "tsm_rule_threshold_sensitivity_summary.csv", index=False)
    threshold_sensitivity_quality.to_csv(outdir / "tsm_rule_threshold_sensitivity_quality_checks.csv", index=False)
    write_report(outdir, config, labels, features, quality, failure_df, split_manifest)
    print("완료: pooled prediction dataset outputs =", outdir.resolve())
    print(quality.to_string(index=False))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

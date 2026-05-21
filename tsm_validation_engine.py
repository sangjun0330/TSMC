#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSM strategy validation engine.

This adds reliability checks around the backtest:
- regime/period performance
- rolling walk-forward audit windows
- parameter sensitivity for stop multiples and score thresholds
- mechanical quality checks for look-ahead, final-row fills, costs, and overlap

Outputs:
- tsm_validation_segment_summary.csv
- tsm_validation_walk_forward_summary.csv
- tsm_validation_parameter_sensitivity.csv
- tsm_validation_quality_checks.csv
- tsm_validation_report.md
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from causal_utils import assert_prefix_stability
from tsm_price_rule_engine import add_discovered_states, add_forward_returns, add_scores_and_signals
from tsm_backtest_engine import (
    STRATEGIES,
    derive_max_weight,
    load_inputs,
    run_backtests,
    summarize_strategy,
    simulate_strategy,
)
from tsm_core.splits import CombinatorialPurgedEventSplit


TRADING_DAYS = 252


REGIME_WINDOWS: Tuple[Tuple[str, str, str], ...] = (
    ("2016_2019_pre_covid", "2016-01-01", "2019-12-31"),
    ("2020_covid_liquidity_shock", "2020-01-01", "2020-12-31"),
    ("2021_2022_peak_to_downcycle", "2021-01-01", "2022-12-31"),
    ("2023_2024_ai_reacceleration", "2023-01-01", "2024-12-31"),
    ("2025_2026_recent_high_vol", "2025-01-01", "2026-12-31"),
)


def pct(x: float) -> float:
    return float(x) * 100.0 if pd.notna(x) else np.nan


def fmt_pct(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}%"


def fmt_num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    if value == np.inf:
        return "inf"
    return f"{value:.{digits}f}"


def as_float(value, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def summarize_curve_period(
    curve: pd.DataFrame,
    strategy_id: str,
    strategy_name: str,
    period_type: str,
    period_name: str,
) -> Dict:
    g = curve.sort_values("date").copy()
    if len(g) < 2:
        return {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "period_type": period_type,
            "period_name": period_name,
            "start_date": g["date"].iloc[0].date().isoformat() if len(g) else "",
            "end_date": g["date"].iloc[-1].date().isoformat() if len(g) else "",
            "trading_days": len(g),
            "total_return_pct": np.nan,
            "cagr_pct": np.nan,
            "max_drawdown_pct": np.nan,
            "sharpe_zero_rf": np.nan,
            "exposure_days_pct": np.nan,
        }

    start_equity = float(g["equity"].iloc[0])
    end_equity = float(g["equity"].iloc[-1])
    years = max((g["date"].iloc[-1] - g["date"].iloc[0]).days / 365.25, 1 / 365.25)
    total_return = end_equity / start_equity - 1.0 if start_equity > 0 else np.nan
    cagr = (end_equity / start_equity) ** (1.0 / years) - 1.0 if start_equity > 0 and end_equity > 0 else np.nan
    dd = g["equity"] / g["equity"].cummax() - 1.0
    daily_ret = g["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    daily_std = daily_ret.std(ddof=0)
    sharpe = daily_ret.mean() / daily_std * math.sqrt(TRADING_DAYS) if daily_std > 0 else np.nan
    position_col = "position_weight" if "position_weight" in g.columns else "position_weight_pct" if "position_weight_pct" in g.columns else ""
    exposure_days = (pd.to_numeric(g[position_col], errors="coerce") > 0).mean() if position_col else np.nan

    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "period_type": period_type,
        "period_name": period_name,
        "start_date": g["date"].iloc[0].date().isoformat(),
        "end_date": g["date"].iloc[-1].date().isoformat(),
        "trading_days": len(g),
        "total_return_pct": pct(total_return),
        "cagr_pct": pct(cagr),
        "max_drawdown_pct": pct(dd.min()),
        "sharpe_zero_rf": sharpe,
        "exposure_days_pct": pct(exposure_days),
    }


def build_baseline_curves(
    signals: pd.DataFrame,
    initial_capital: float,
    commission_bps: float,
    slippage_bps: float,
    stop_multiple: float,
    score_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cost_rate = (commission_bps + slippage_bps) / 10000.0
    curves = []
    trades = []
    for spec in STRATEGIES:
        curve, trade_log = simulate_strategy(
            signals,
            spec,
            initial_capital,
            cost_rate,
            stop_multiple=stop_multiple,
            score_threshold=score_threshold,
        )
        curves.append(curve)
        if not trade_log.empty:
            trades.append(trade_log)
    summary, trade_out, _, _ = run_backtests(
        signals,
        initial_capital=initial_capital,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        stop_multiple=stop_multiple,
        score_threshold=score_threshold,
    )
    curve_out = pd.concat(curves, ignore_index=True)
    trade_raw = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return summary, curve_out, trade_raw if not trade_raw.empty else trade_out


def build_segment_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy_id, strategy_name), group in curves.groupby(["strategy_id", "strategy_name"]):
        for period_name, start, end in REGIME_WINDOWS:
            mask = group["date"].between(pd.Timestamp(start), pd.Timestamp(end))
            part = group.loc[mask]
            if not part.empty:
                rows.append(
                    summarize_curve_period(
                        part,
                        strategy_id,
                        strategy_name,
                        period_type="fixed_regime",
                        period_name=period_name,
                    )
                )
    return pd.DataFrame(rows)


def build_walk_forward_summary(
    curves: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
) -> pd.DataFrame:
    rows = []
    for (strategy_id, strategy_name), group in curves.groupby(["strategy_id", "strategy_name"]):
        g = group.sort_values("date").reset_index(drop=True)
        window_id = 1
        start = 0
        while start + train_days + 20 < len(g):
            train_start = start
            train_end = min(start + train_days, len(g))
            test_start = train_end
            test_end = min(test_start + test_days, len(g))
            if test_end - test_start < 20:
                break

            train = summarize_curve_period(
                g.iloc[train_start:train_end],
                strategy_id,
                strategy_name,
                period_type="walk_forward_train",
                period_name=f"wf_{window_id}",
            )
            test = summarize_curve_period(
                g.iloc[test_start:test_end],
                strategy_id,
                strategy_name,
                period_type="walk_forward_test",
                period_name=f"wf_{window_id}",
            )
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "strategy_name": strategy_name,
                    "window_id": window_id,
                    "train_start_date": train["start_date"],
                    "train_end_date": train["end_date"],
                    "test_start_date": test["start_date"],
                    "test_end_date": test["end_date"],
                    "train_total_return_pct": train["total_return_pct"],
                    "test_total_return_pct": test["total_return_pct"],
                    "train_cagr_pct": train["cagr_pct"],
                    "test_cagr_pct": test["cagr_pct"],
                    "train_max_drawdown_pct": train["max_drawdown_pct"],
                    "test_max_drawdown_pct": test["max_drawdown_pct"],
                    "train_sharpe_zero_rf": train["sharpe_zero_rf"],
                    "test_sharpe_zero_rf": test["sharpe_zero_rf"],
                    "test_exposure_days_pct": test["exposure_days_pct"],
                    "test_positive": bool(pd.notna(test["total_return_pct"]) and test["total_return_pct"] > 0),
                }
            )
            window_id += 1
            start += step_days
    return pd.DataFrame(rows)


def make_causal_signals(raw_enriched: pd.DataFrame) -> pd.DataFrame:
    signals = add_forward_returns(raw_enriched.copy())
    signals, q = add_discovered_states(signals)
    signals = add_scores_and_signals(signals, q)
    signals["derived_max_weight_by_vol"] = derive_max_weight(signals)
    return signals


def build_causal_walk_forward_summary(
    raw_enriched: pd.DataFrame,
    initial_capital: float,
    commission_bps: float,
    slippage_bps: float,
    stop_multiple: float,
    score_threshold: float,
    train_days: int,
    test_days: int,
    step_days: int,
) -> pd.DataFrame:
    rows: List[Dict] = []
    if raw_enriched.empty:
        return pd.DataFrame()
    cost_rate = (commission_bps + slippage_bps) / 10000.0
    start = train_days
    window_id = 1
    while start < len(raw_enriched):
        train_end = start
        test_start = train_end
        test_end = min(test_start + test_days, len(raw_enriched))
        if test_end - test_start < 20:
            break
        history_plus_test = raw_enriched.iloc[:test_end].copy()
        signals = make_causal_signals(history_plus_test)
        test_signals = signals.iloc[test_start:test_end].copy().reset_index(drop=True)
        if test_signals.empty:
            break
        for spec in STRATEGIES:
            curve, trades = simulate_strategy(
                test_signals,
                spec,
                initial_capital,
                cost_rate,
                stop_multiple=stop_multiple,
                score_threshold=score_threshold,
            )
            summary = summarize_strategy(
                curve,
                trades,
                spec,
                initial_capital,
                stop_multiple=stop_multiple,
                score_threshold=score_threshold,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
            )
            rows.append(
                {
                    "strategy_id": spec.strategy_id,
                    "strategy_name": spec.name,
                    "strategy_group": spec.strategy_group,
                    "window_id": window_id,
                    "train_start_date": raw_enriched.iloc[0]["date"],
                    "train_end_date": raw_enriched.iloc[train_end - 1]["date"],
                    "test_start_date": raw_enriched.iloc[test_start]["date"],
                    "test_end_date": raw_enriched.iloc[test_end - 1]["date"],
                    "test_trading_days": len(test_signals),
                    "test_total_return_pct": summary["total_return_pct"],
                    "test_cagr_pct": summary["cagr_pct"],
                    "test_max_drawdown_pct": summary["max_drawdown_pct"],
                    "test_sharpe_zero_rf": summary["sharpe_zero_rf"],
                    "test_trade_count": summary["trade_count"],
                    "test_exposure_days_pct": summary["exposure_days_pct"],
                    "test_positive": bool(pd.notna(summary["total_return_pct"]) and summary["total_return_pct"] > 0),
                }
            )
        start += step_days
        window_id += 1
    return pd.DataFrame(rows)


def build_cpcv_path_summary(
    curves: pd.DataFrame,
    horizon_days: int = 20,
    n_groups: int = 6,
    test_group_count: int = 2,
) -> pd.DataFrame:
    rows: List[Dict] = []
    if curves.empty:
        return pd.DataFrame()
    splitter = CombinatorialPurgedEventSplit(
        horizon_days=horizon_days,
        n_groups=n_groups,
        test_group_count=test_group_count,
        min_train_events=60,
        min_test_events=20,
        max_paths=60,
    )
    for (strategy_id, strategy_name), group in curves.groupby(["strategy_id", "strategy_name"]):
        g = group.sort_values("date").reset_index(drop=True).copy()
        g["signal_idx"] = np.arange(len(g))
        for train, test, spec in splitter.split(g, signal_idx_col="signal_idx"):
            train_summary = summarize_curve_period(
                train,
                strategy_id,
                strategy_name,
                period_type="cpcv_train",
                period_name=f"cpcv_{spec.path_id}",
            )
            test_summary = summarize_curve_period(
                test,
                strategy_id,
                strategy_name,
                period_type="cpcv_test",
                period_name=f"cpcv_{spec.path_id}",
            )
            rows.append(
                {
                    "path_id": spec.path_id,
                    "strategy_id": strategy_id,
                    "strategy_name": strategy_name,
                    "test_group_ids": "|".join(str(v) for v in spec.test_group_ids),
                    "train_group_ids": "|".join(str(v) for v in spec.train_group_ids),
                    "horizon_days": spec.horizon_days,
                    "embargo_days": spec.embargo_days,
                    "purged_train_count": spec.purged_train_count,
                    "train_event_count": spec.train_event_count,
                    "test_event_count": spec.test_event_count,
                    "train_start_date": train_summary["start_date"],
                    "train_end_date": train_summary["end_date"],
                    "test_start_date": test_summary["start_date"],
                    "test_end_date": test_summary["end_date"],
                    "train_total_return_pct": train_summary["total_return_pct"],
                    "test_total_return_pct": test_summary["total_return_pct"],
                    "test_uplift_pct": test_summary["total_return_pct"],
                    "train_cagr_pct": train_summary["cagr_pct"],
                    "test_cagr_pct": test_summary["cagr_pct"],
                    "test_max_drawdown_pct": test_summary["max_drawdown_pct"],
                    "test_sharpe_zero_rf": test_summary["sharpe_zero_rf"],
                    "test_positive": bool(pd.notna(test_summary["total_return_pct"]) and test_summary["total_return_pct"] > 0),
                }
            )
    return pd.DataFrame(rows)


def build_cpcv_strategy_distribution(cpcv_paths: pd.DataFrame) -> pd.DataFrame:
    if cpcv_paths.empty:
        return pd.DataFrame()
    rows: List[Dict] = []
    for (strategy_id, strategy_name), group in cpcv_paths.groupby(["strategy_id", "strategy_name"]):
        uplift = pd.to_numeric(group["test_uplift_pct"], errors="coerce")
        drawdown = pd.to_numeric(group["test_max_drawdown_pct"], errors="coerce")
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "cpcv_path_count": int(len(group)),
                "test_uplift_pct_median": float(uplift.median()) if uplift.notna().any() else np.nan,
                "test_uplift_pct_q25": float(uplift.quantile(0.25)) if uplift.notna().any() else np.nan,
                "test_uplift_pct_min": float(uplift.min()) if uplift.notna().any() else np.nan,
                "positive_path_rate_pct": pct((uplift > 0).mean()) if uplift.notna().any() else np.nan,
                "worst_test_drawdown_pct": float(drawdown.min()) if drawdown.notna().any() else np.nan,
                "cpcv_median_uplift_pass": bool(uplift.notna().any() and uplift.median() > 0.0),
                "cpcv_worst_quartile_pass": bool(uplift.notna().any() and uplift.quantile(0.25) >= 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["cpcv_median_uplift_pass", "test_uplift_pct_median"], ascending=[False, False]).reset_index(drop=True)


def build_cpcv_model_distribution(outdir: Path) -> pd.DataFrame:
    prediction_path = outdir / "tsm_pooled_model_oof_predictions.csv"
    if not prediction_path.exists():
        prediction_path = outdir / "tsm_prediction_oos_predictions.csv"
    if not prediction_path.exists():
        return pd.DataFrame()
    try:
        predictions = pd.read_csv(prediction_path, parse_dates=["date"])
    except Exception:
        return pd.DataFrame()
    if predictions.empty:
        return pd.DataFrame()
    return_col = next((c for c in ["label_net_return_pct_20d", "label_net_return_pct", "actual_return_pct", "net_return_pct"] if c in predictions.columns), "")
    selected_col = next((c for c in ["selected_by_threshold", "is_selected", "model_selected", "selected"] if c in predictions.columns), "")
    model_col = "model_name" if "model_name" in predictions.columns else ""
    scope_col = "candidate_scope" if "candidate_scope" in predictions.columns else "prediction_universe" if "prediction_universe" in predictions.columns else ""
    fold_col = "fold_id" if "fold_id" in predictions.columns else "split" if "split" in predictions.columns else ""
    if not return_col or not selected_col:
        return pd.DataFrame()
    frame = predictions.copy()
    frame["_return"] = pd.to_numeric(frame[return_col], errors="coerce")
    frame["_selected"] = frame[selected_col].map(lambda x: str(x).strip().lower() in {"true", "1", "yes", "y"})
    if model_col:
        frame["_model"] = frame[model_col].astype(str)
    else:
        frame["_model"] = "model"
    if scope_col:
        frame["_scope"] = frame[scope_col].astype(str)
    else:
        frame["_scope"] = "all"
    if fold_col:
        frame["_fold"] = frame[fold_col].astype(str)
    else:
        frame["_fold"] = "all"
    rows: List[Dict] = []
    for (model_name, scope), group in frame.dropna(subset=["_return"]).groupby(["_model", "_scope"]):
        fold_uplifts: List[float] = []
        selected_counts: List[int] = []
        for fold, fold_group in group.groupby("_fold"):
            selected = fold_group[fold_group["_selected"]]
            if selected.empty or fold_group.empty:
                continue
            uplift = float(selected["_return"].mean() - fold_group["_return"].mean())
            fold_uplifts.append(uplift)
            selected_counts.append(int(len(selected)))
        if not fold_uplifts:
            continue
        uplift_series = pd.Series(fold_uplifts, dtype=float)
        rows.append(
            {
                "model_name": model_name,
                "candidate_scope": scope,
                "fold_count": int(len(uplift_series)),
                "selected_event_count": int(sum(selected_counts)),
                "min_selected_events_per_fold": int(min(selected_counts)),
                "selected_minus_all_pct_median": float(uplift_series.median()),
                "selected_minus_all_pct_q25": float(uplift_series.quantile(0.25)),
                "selected_minus_all_pct_min": float(uplift_series.min()),
                "positive_fold_rate_pct": pct((uplift_series > 0).mean()),
                "cpcv_model_median_uplift_pass": bool(uplift_series.median() > 0.0),
                "cpcv_model_worst_quartile_pass": bool(uplift_series.quantile(0.25) >= 0.0),
                "source_file": prediction_path.name,
            }
        )
    return pd.DataFrame(rows).sort_values(["cpcv_model_median_uplift_pass", "selected_minus_all_pct_median"], ascending=[False, False]).reset_index(drop=True) if rows else pd.DataFrame()


def build_parameter_sensitivity(
    signals: pd.DataFrame,
    initial_capital: float,
    commission_bps: float,
    slippage_bps: float,
    stop_multiples: Iterable[float],
    score_thresholds: Iterable[float],
) -> pd.DataFrame:
    rows = []

    for stop_multiple in stop_multiples:
        summary, _, _, _ = run_backtests(
            signals,
            initial_capital=initial_capital,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            stop_multiple=stop_multiple,
            score_threshold=75.0,
        )
        summary = summary.copy()
        summary.insert(0, "sensitivity_type", "stop_multiple")
        summary.insert(1, "parameter_value", stop_multiple)
        rows.append(summary)

    for score_threshold in score_thresholds:
        summary, _, _, _ = run_backtests(
            signals,
            initial_capital=initial_capital,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            stop_multiple=2.0,
            score_threshold=score_threshold,
        )
        summary = summary[summary["strategy_id"].isin(["C_SCORE75_TRIGGER_2ATR", "E_VOL_ADJUSTED_SCORE75"])].copy()
        summary.insert(0, "sensitivity_type", "score_threshold")
        summary.insert(1, "parameter_value", score_threshold)
        rows.append(summary)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_model_trials_log(baseline_summary: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    tested_at = datetime.now(timezone.utc).isoformat()
    rows: List[Dict] = []
    for _, row in baseline_summary.iterrows():
        rows.append(
            {
                "strategy_id": row.get("strategy_id"),
                "strategy_group": row.get("strategy_group", "diagnostic"),
                "trial_source": "baseline",
                "parameters": {
                    "stop_multiple": row.get("stop_multiple"),
                    "score_threshold": row.get("score_threshold"),
                    "commission_bps": row.get("commission_bps"),
                    "slippage_bps": row.get("slippage_bps"),
                },
                "sharpe": row.get("sharpe_zero_rf"),
                "cagr": row.get("cagr_pct"),
                "mdd": row.get("max_drawdown_pct"),
                "tested_at": tested_at,
            }
        )
    if not sensitivity.empty:
        for _, row in sensitivity.iterrows():
            rows.append(
                {
                    "strategy_id": row.get("strategy_id"),
                    "strategy_group": row.get("strategy_group", "diagnostic"),
                    "trial_source": row.get("sensitivity_type"),
                    "parameters": {
                        "parameter_value": row.get("parameter_value"),
                        "stop_multiple": row.get("stop_multiple"),
                        "score_threshold": row.get("score_threshold"),
                        "commission_bps": row.get("commission_bps"),
                        "slippage_bps": row.get("slippage_bps"),
                    },
                    "sharpe": row.get("sharpe_zero_rf"),
                    "cagr": row.get("cagr_pct"),
                    "mdd": row.get("max_drawdown_pct"),
                    "tested_at": tested_at,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["parameters"] = out["parameters"].map(str)
    return out


def build_pbo_report(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame()
    rows: List[Dict] = []
    for strategy_id, group in trials.groupby("strategy_id"):
        sharpe = pd.to_numeric(group["sharpe"], errors="coerce")
        cagr = pd.to_numeric(group["cagr"], errors="coerce")
        valid = group.loc[sharpe.notna()].copy()
        if valid.empty:
            continue
        valid_sharpe = pd.to_numeric(valid["sharpe"], errors="coerce").to_numpy()
        best = valid.iloc[int(np.nanargmax(valid_sharpe))]
        median_sharpe = float(pd.to_numeric(valid["sharpe"], errors="coerce").median())
        best_sharpe = as_float(best.get("sharpe"))
        best_cagr = as_float(best.get("cagr"))
        pbo_proxy = float(((sharpe > sharpe.median()) & (cagr <= cagr.median())).mean()) if len(valid) > 1 else np.nan
        rows.append(
            {
                "strategy_id": strategy_id,
                "trial_count": len(valid),
                "best_trial_source": best.get("trial_source"),
                "best_trial_parameters": best.get("parameters"),
                "best_sharpe": best_sharpe,
                "median_sharpe": median_sharpe,
                "best_cagr_pct": best_cagr,
                "pbo_proxy": pbo_proxy,
                "overfit_warning": bool(pd.notna(pbo_proxy) and pbo_proxy >= 0.50),
            }
        )
    return pd.DataFrame(rows)


def build_cscv_pbo_report(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty or "sharpe" not in trials.columns:
        return pd.DataFrame()
    rows: List[Dict] = []
    for strategy_id, group in trials.groupby("strategy_id"):
        valid = group.copy()
        valid["_sharpe"] = pd.to_numeric(valid["sharpe"], errors="coerce")
        valid = valid.dropna(subset=["_sharpe"]).reset_index(drop=True)
        n = len(valid)
        if n < 4:
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "trial_count": n,
                    "cscv_combinations": 0,
                    "pbo_cscv": np.nan,
                    "mean_oos_rank_pct": np.nan,
                    "details": "insufficient_trials_for_cscv",
                }
            )
            continue
        half = n // 2
        combo_iter = list(combinations(range(n), half))
        original_combo_count = len(combo_iter)
        if len(combo_iter) > 252:
            step = max(1, len(combo_iter) // 252)
            combo_iter = combo_iter[::step][:252]
        overfit = 0
        rank_pcts: List[float] = []
        used = 0
        all_indices = set(range(n))
        for train_indices in combo_iter:
            train_set = set(train_indices)
            test_indices = sorted(all_indices - train_set)
            train = valid.iloc[list(train_indices)]
            test = valid.iloc[test_indices]
            if train.empty or test.empty:
                continue
            best_local_idx = int(train["_sharpe"].idxmax())
            if best_local_idx not in test.index:
                test_rank_frame = pd.concat([test, valid.iloc[[best_local_idx]]], ignore_index=True)
            else:
                test_rank_frame = test
            ranks = test_rank_frame["_sharpe"].rank(method="average", pct=True)
            best_rank_pct = float(ranks.iloc[-1]) if best_local_idx not in test.index else float(ranks.loc[best_local_idx])
            rank_pcts.append(best_rank_pct)
            if best_rank_pct <= 0.50:
                overfit += 1
            used += 1
        rows.append(
            {
                "strategy_id": strategy_id,
                "trial_count": n,
                "cscv_combinations": used,
                "pbo_cscv": float(overfit / used) if used else np.nan,
                "mean_oos_rank_pct": float(np.mean(rank_pcts)) if rank_pcts else np.nan,
                "details": "sampled_combinations" if original_combo_count > used else "all_combinations",
            }
        )
    return pd.DataFrame(rows)


def build_deflated_sharpe_report(curves: pd.DataFrame, trial_count: int) -> pd.DataFrame:
    rows: List[Dict] = []
    norm = NormalDist()
    effective_trials = max(int(trial_count), 1)
    for (strategy_id, strategy_name), group in curves.groupby(["strategy_id", "strategy_name"]):
        daily_ret = pd.to_numeric(group.sort_values("date")["daily_return"], errors="coerce").dropna()
        n = len(daily_ret)
        if n < 3 or daily_ret.std(ddof=0) <= 0:
            continue
        sr_period = float(daily_ret.mean() / daily_ret.std(ddof=0))
        sr = sr_period * math.sqrt(TRADING_DAYS)
        skew = float(daily_ret.skew())
        kurt = float(daily_ret.kurtosis() + 3.0)
        sr_std_term = max(1e-12, 1.0 - skew * sr_period + ((kurt - 1.0) / 4.0) * sr_period * sr_period)
        sr_std = math.sqrt(sr_std_term / max(n - 1, 1))
        psr = norm.cdf(sr_period / sr_std)
        trial_adjustment = norm.inv_cdf(1.0 - 1.0 / max(effective_trials, 2))
        deflated_benchmark_period = max(0.0, trial_adjustment * sr_std)
        dsr = norm.cdf((sr_period - deflated_benchmark_period) / sr_std)
        rng = np.random.default_rng(42)
        boot = []
        daily_values = daily_ret.to_numpy(dtype=float)
        for _ in range(400):
            sample = rng.choice(daily_values, size=n, replace=True)
            sample_std = float(np.std(sample, ddof=0))
            if sample_std > 0:
                boot.append(float(np.mean(sample) / sample_std * math.sqrt(TRADING_DAYS)))
        bootstrap_lower = float(np.nanpercentile(boot, 5)) if boot else np.nan
        bootstrap_upper = float(np.nanpercentile(boot, 95)) if boot else np.nan
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "trial_count": effective_trials,
                "observations": n,
                "observed_sharpe": sr,
                "observed_period_sharpe": sr_period,
                "skew": skew,
                "kurtosis": kurt,
                "probabilistic_sharpe_ratio": psr,
                "deflated_sharpe_ratio": dsr,
                "deflated_benchmark_sharpe": deflated_benchmark_period * math.sqrt(TRADING_DAYS),
                "bootstrap_sharpe_ci_lower_90": bootstrap_lower,
                "bootstrap_sharpe_ci_upper_90": bootstrap_upper,
                "dsr_pass": bool(dsr >= 0.95 and pd.notna(bootstrap_lower) and bootstrap_lower > 0.0),
            }
        )
    return pd.DataFrame(rows)


def no_overlapping_trades(trades: pd.DataFrame) -> bool:
    if trades.empty:
        return True
    if "trade_event" in trades.columns:
        trades = trades[trades["trade_event"].eq("AGGREGATE_EXIT")].copy()
    if trades.empty:
        return True
    for _, g in trades.sort_values(["strategy_id", "entry_date"]).groupby("strategy_id"):
        previous_exit = None
        for _, row in g.iterrows():
            if previous_exit is not None and row["entry_date"] < previous_exit:
                return False
            previous_exit = row["exit_date"]
    return True


def build_quality_checks(signals: pd.DataFrame, trades: pd.DataFrame, curves: pd.DataFrame, raw_enriched: pd.DataFrame | None = None) -> pd.DataFrame:
    last_signal_date = signals["date"].max()
    rows = []

    def add(check: str, passed: bool, detail: str) -> None:
        rows.append({"check": check, "passed": bool(passed), "detail": detail})

    add("signals_dates_sorted", signals["date"].is_monotonic_increasing, "signals date index must be ascending")
    add("signals_no_duplicate_dates", not signals["date"].duplicated().any(), "signals must have one row per trading date")
    add("curves_have_all_strategies", curves["strategy_id"].nunique() == len(STRATEGIES), "one equity curve per strategy")
    if raw_enriched is not None and not raw_enriched.empty:
        try:
            assert_prefix_stability(make_causal_signals, raw_enriched, check_rows=min(1000, len(raw_enriched)))
            add("rule_engine_prefix_stability_pass", True, "causal rule engine matches full-vs-prefix recomputation")
        except AssertionError as exc:
            add("rule_engine_prefix_stability_pass", False, str(exc))
    else:
        add("rule_engine_prefix_stability_pass", False, "raw enriched input unavailable for prefix stability check")

    if trades.empty:
        add("trade_log_non_empty", False, "baseline trade log is empty")
        add("entry_after_signal", True, "not applicable because trade log is empty")
        add("no_last_row_signal_fill", True, "not applicable because trade log is empty")
        add("net_return_after_cost_not_above_gross", True, "not applicable because trade log is empty")
        add("no_overlapping_positions", True, "not applicable because trade log is empty")
        add("exit_reasons_present", False, "trade log is empty")
    else:
        add("trade_log_non_empty", True, f"{len(trades)} trades")
        add(
            "entry_after_signal",
            bool((trades["entry_date"] > trades["entry_signal_date"]).all()),
            "entries should fill after close-based signals",
        )
        add(
            "no_last_row_signal_fill",
            not (trades["entry_signal_date"] == last_signal_date).any(),
            "last-row signals have no next open and should not fill",
        )
        add(
            "net_return_after_cost_not_above_gross",
            bool((trades["net_return_pct"] <= trades["gross_return_pct"] + 1e-9).all()),
            "cost-adjusted trade returns should not exceed gross returns",
        )
        add(
            "no_overlapping_positions",
            no_overlapping_trades(trades),
            "single-position strategy should not stack overlapping entries",
        )
        add(
            "exit_reasons_present",
            bool(trades["exit_reason"].nunique() >= 2),
            "expect more than one exit reason in baseline test",
        )

    return pd.DataFrame(rows)


def write_report(
    outdir: Path,
    baseline_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    walk_forward: pd.DataFrame,
    causal_walk_forward: pd.DataFrame,
    sensitivity: pd.DataFrame,
    cpcv_strategy_distribution: pd.DataFrame,
    cpcv_model_distribution: pd.DataFrame,
    pbo_report: pd.DataFrame,
    cscv_pbo_report: pd.DataFrame,
    dsr_report: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    ranked = baseline_summary.sort_values("cagr_pct", ascending=False)
    best = ranked.iloc[0]

    weak_segments = (
        segment_summary.assign(is_negative=lambda x: x["total_return_pct"] < 0)
        .groupby("strategy_id")["is_negative"]
        .sum()
        .reset_index(name="negative_segment_count")
    )
    wf = walk_forward.groupby("strategy_id").agg(
        median_test_cagr_pct=("test_cagr_pct", "median"),
        positive_test_rate_pct=("test_positive", lambda x: pct(x.mean())),
        worst_test_drawdown_pct=("test_max_drawdown_pct", "min"),
    ).reset_index()
    quality_passed = bool(quality["passed"].all())

    lines = [
        "# TSMC Strategy Validation Report",
        "",
        "## Validation Principles",
        "- Avoid judging a strategy only by one full-period equity curve.",
        "- Use out-of-sample style windows and regime slices to expose instability.",
        "- Keep transaction costs in the tested return stream.",
        "- Treat parameter sweeps as robustness checks, not as permission to pick the best-looking value.",
        "",
        "Research basis used for this implementation:",
        "- Bailey, Borwein, Lopez de Prado, and Zhu: Probability of Backtest Overfitting.",
        "- CFA Institute: Backtesting & Simulation guidance.",
        "- Novy-Marx/NBER: multiple-signal backtests can amplify overfitting bias.",
        "- Sharpe: risk-adjusted return should be interpreted with the risk stream, not raw return alone.",
        "",
        "## Baseline Ranking",
        "",
        "| Strategy | CAGR | MDD | Sharpe | Trades |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in ranked.iterrows():
        lines.append(
            f"| {row['strategy_id']} | {fmt_pct(row['cagr_pct'])} | {fmt_pct(row['max_drawdown_pct'])} | "
            f"{fmt_num(row['sharpe_zero_rf'])} | {int(row['trade_count'])} |"
        )

    lines.extend(
        [
            "",
            "## Best Full-Period Strategy",
            f"- Strategy: {best['strategy_id']}",
            f"- CAGR: {fmt_pct(best['cagr_pct'])}",
            f"- Max drawdown: {fmt_pct(best['max_drawdown_pct'])}",
            "",
            "## Regime Weakness Count",
            "",
            "| Strategy | Negative fixed regimes |",
            "|---|---:|",
        ]
    )
    for _, row in weak_segments.iterrows():
        lines.append(f"| {row['strategy_id']} | {int(row['negative_segment_count'])} |")

    lines.extend(["", "## Walk-Forward Test Summary", "", "| Strategy | Median test CAGR | Positive test rate | Worst test MDD |", "|---|---:|---:|---:|"])
    for _, row in wf.iterrows():
        lines.append(
            f"| {row['strategy_id']} | {fmt_pct(row['median_test_cagr_pct'])} | "
            f"{fmt_pct(row['positive_test_rate_pct'])} | {fmt_pct(row['worst_test_drawdown_pct'])} |"
        )

    if not causal_walk_forward.empty:
        causal_wf = causal_walk_forward.groupby("strategy_id").agg(
            median_test_cagr_pct=("test_cagr_pct", "median"),
            positive_test_rate_pct=("test_positive", lambda x: pct(x.mean())),
            worst_test_drawdown_pct=("test_max_drawdown_pct", "min"),
        ).reset_index()
        lines.extend(["", "## Causal Recomputed Walk-Forward", "", "| Strategy | Median test CAGR | Positive test rate | Worst test MDD |", "|---|---:|---:|---:|"])
        for _, row in causal_wf.iterrows():
            lines.append(
                f"| {row['strategy_id']} | {fmt_pct(row['median_test_cagr_pct'])} | "
                f"{fmt_pct(row['positive_test_rate_pct'])} | {fmt_pct(row['worst_test_drawdown_pct'])} |"
            )

    if not cpcv_strategy_distribution.empty:
        lines.extend(["", "## CPCV Strategy Distribution", "", "| Strategy | Paths | Median uplift | Q25 uplift | Worst MDD | Positive paths |", "|---|---:|---:|---:|---:|---:|"])
        for _, row in cpcv_strategy_distribution.iterrows():
            lines.append(
                f"| {row['strategy_id']} | {int(row['cpcv_path_count'])} | {fmt_pct(row['test_uplift_pct_median'])} | "
                f"{fmt_pct(row['test_uplift_pct_q25'])} | {fmt_pct(row['worst_test_drawdown_pct'])} | {fmt_pct(row['positive_path_rate_pct'])} |"
            )

    if not cpcv_model_distribution.empty:
        lines.extend(["", "## CPCV Model Distribution", "", "| Model | Scope | Folds | Selected | Median selected-minus-all | Q25 |", "|---|---|---:|---:|---:|---:|"])
        for _, row in cpcv_model_distribution.iterrows():
            lines.append(
                f"| {row['model_name']} | {row['candidate_scope']} | {int(row['fold_count'])} | "
                f"{int(row['selected_event_count'])} | {fmt_pct(row['selected_minus_all_pct_median'])} | {fmt_pct(row['selected_minus_all_pct_q25'])} |"
            )

    lines.extend(
        [
            "",
            "## Mechanical Quality Checks",
            f"- Overall: {'PASS' if quality_passed else 'FAIL'}",
            "",
            "| Check | Passed | Detail |",
            "|---|---:|---|",
        ]
    )
    for _, row in quality.iterrows():
        lines.append(f"| {row['check']} | {row['passed']} | {row['detail']} |")

    if not sensitivity.empty:
        sens_best = sensitivity.sort_values("cagr_pct", ascending=False).head(5)
        lines.extend(["", "## Top Sensitivity Runs", "", "| Type | Value | Strategy | CAGR | MDD |", "|---|---:|---|---:|---:|"])
        for _, row in sens_best.iterrows():
            lines.append(
                f"| {row['sensitivity_type']} | {row['parameter_value']} | {row['strategy_id']} | "
                f"{fmt_pct(row['cagr_pct'])} | {fmt_pct(row['max_drawdown_pct'])} |"
            )

    if not pbo_report.empty:
        lines.extend(["", "## PBO Proxy", "", "| Strategy | Trials | PBO Proxy | Overfit Warning |", "|---|---:|---:|---:|"])
        for _, row in pbo_report.iterrows():
            lines.append(f"| {row['strategy_id']} | {int(row['trial_count'])} | {fmt_num(row['pbo_proxy'])} | {row['overfit_warning']} |")

    if not cscv_pbo_report.empty:
        lines.extend(["", "## CSCV PBO", "", "| Strategy | Trials | Combinations | PBO | Mean OOS Rank |", "|---|---:|---:|---:|---:|"])
        for _, row in cscv_pbo_report.iterrows():
            lines.append(
                f"| {row['strategy_id']} | {int(row['trial_count'])} | {int(row['cscv_combinations'])} | "
                f"{fmt_num(row['pbo_cscv'])} | {fmt_num(row['mean_oos_rank_pct'])} |"
            )

    if not dsr_report.empty:
        lines.extend(["", "## Deflated Sharpe", "", "| Strategy | Sharpe | PSR | DSR | Bootstrap 90% CI Lower | Pass |", "|---|---:|---:|---:|---:|---:|"])
        for _, row in dsr_report.iterrows():
            lines.append(
                f"| {row['strategy_id']} | {fmt_num(row['observed_sharpe'])} | "
                f"{fmt_num(row['probabilistic_sharpe_ratio'], 3)} | {fmt_num(row['deflated_sharpe_ratio'], 3)} | "
                f"{fmt_num(row.get('bootstrap_sharpe_ci_lower_90'))} | {row['dsr_pass']} |"
            )

    lines.extend(
        [
            "",
            "## Output files",
            "- tsm_validation_segment_summary.csv",
            "- tsm_validation_walk_forward_summary.csv",
            "- tsm_validation_causal_walk_forward_summary.csv",
            "- tsm_cpcv_path_summary.csv",
            "- tsm_cpcv_strategy_distribution.csv",
            "- tsm_cpcv_model_distribution.csv",
            "- tsm_validation_parameter_sensitivity.csv",
            "- tsm_model_trials_log.csv",
            "- tsm_pbo_report.csv",
            "- tsm_cscv_pbo_report.csv",
            "- tsm_deflated_sharpe_report.csv",
            "- tsm_validation_quality_checks.csv",
            "",
            "This report is research tooling, not investment advice.",
        ]
    )
    (outdir / "tsm_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TSM strategy robustness.")
    parser.add_argument("--signals", default="tsm_price_rule_output/tsm_daily_algorithmic_signals.csv")
    parser.add_argument("--enriched", default="output/tsm_daily_10y_enriched.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--initial-capital", type=float, default=1.0)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--baseline-stop-multiple", type=float, default=2.0)
    parser.add_argument("--baseline-score-threshold", type=float, default=75.0)
    parser.add_argument("--walk-forward-train-days", type=int, default=756)
    parser.add_argument("--walk-forward-test-days", type=int, default=252)
    parser.add_argument("--walk-forward-step-days", type=int, default=252)
    parser.add_argument("--stop-multiples", default="1.5,2.0,2.5,3.0")
    parser.add_argument("--score-thresholds", default="65,70,75,80")
    parser.add_argument("--model-trial-count", type=int, default=0, help="Additional prediction model/feature/threshold trials to include in DSR deflation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    signals = load_inputs(Path(args.signals), Path(args.enriched) if args.enriched else None)
    raw_enriched = pd.read_csv(args.enriched, parse_dates=["date"]).sort_values("date").reset_index(drop=True) if args.enriched and Path(args.enriched).exists() else pd.DataFrame()
    baseline_summary, curves, trades = build_baseline_curves(
        signals=signals,
        initial_capital=args.initial_capital,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        stop_multiple=args.baseline_stop_multiple,
        score_threshold=args.baseline_score_threshold,
    )
    segment_summary = build_segment_summary(curves)
    walk_forward = build_walk_forward_summary(
        curves,
        train_days=args.walk_forward_train_days,
        test_days=args.walk_forward_test_days,
        step_days=args.walk_forward_step_days,
    )
    causal_walk_forward = build_causal_walk_forward_summary(
        raw_enriched=raw_enriched,
        initial_capital=args.initial_capital,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        stop_multiple=args.baseline_stop_multiple,
        score_threshold=args.baseline_score_threshold,
        train_days=args.walk_forward_train_days,
        test_days=args.walk_forward_test_days,
        step_days=args.walk_forward_step_days,
    )
    sensitivity = build_parameter_sensitivity(
        signals=signals,
        initial_capital=args.initial_capital,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        stop_multiples=parse_float_list(args.stop_multiples),
        score_thresholds=parse_float_list(args.score_thresholds),
    )
    trials_log = build_model_trials_log(baseline_summary, sensitivity)
    pbo_report = build_pbo_report(trials_log)
    cscv_pbo_report = build_cscv_pbo_report(trials_log)
    dsr_report = build_deflated_sharpe_report(curves, len(trials_log) + max(int(args.model_trial_count), 0))
    cpcv_path_summary = build_cpcv_path_summary(curves, horizon_days=20)
    cpcv_strategy_distribution = build_cpcv_strategy_distribution(cpcv_path_summary)
    cpcv_model_distribution = build_cpcv_model_distribution(outdir)
    quality = build_quality_checks(signals, trades, curves, raw_enriched=raw_enriched)

    segment_summary.to_csv(outdir / "tsm_validation_segment_summary.csv", index=False)
    walk_forward.to_csv(outdir / "tsm_validation_walk_forward_summary.csv", index=False)
    causal_walk_forward.to_csv(outdir / "tsm_validation_causal_walk_forward_summary.csv", index=False)
    cpcv_path_summary.to_csv(outdir / "tsm_cpcv_path_summary.csv", index=False)
    cpcv_strategy_distribution.to_csv(outdir / "tsm_cpcv_strategy_distribution.csv", index=False)
    cpcv_model_distribution.to_csv(outdir / "tsm_cpcv_model_distribution.csv", index=False)
    sensitivity.to_csv(outdir / "tsm_validation_parameter_sensitivity.csv", index=False)
    trials_log.to_csv(outdir / "tsm_model_trials_log.csv", index=False)
    pbo_report.to_csv(outdir / "tsm_pbo_report.csv", index=False)
    cscv_pbo_report.to_csv(outdir / "tsm_cscv_pbo_report.csv", index=False)
    dsr_report.to_csv(outdir / "tsm_deflated_sharpe_report.csv", index=False)
    quality.to_csv(outdir / "tsm_validation_quality_checks.csv", index=False)
    write_report(
        outdir,
        baseline_summary,
        segment_summary,
        walk_forward,
        causal_walk_forward,
        sensitivity,
        cpcv_strategy_distribution,
        cpcv_model_distribution,
        pbo_report,
        cscv_pbo_report,
        dsr_report,
        quality,
    )

    print("완료: validation outputs =", outdir.resolve())
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()

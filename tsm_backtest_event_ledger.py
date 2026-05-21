#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest event ledger builder.

This ledger is candidate-centric, not fill-centric. It records trade-ready,
relaxed, risk-blocked, and near-miss events even when no order would have been
filled by the strategy. The output is meant for research feedback features and
auditability, not for placing orders.

Outputs:
- tsm_backtest_event_ledger.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from tsm_backtest_engine import STRATEGIES, as_float, entry_signal, load_inputs, target_weight_for_entry, to_bool, valid_price
from tsm_prediction_engine import candidate_tier


DEFAULT_HORIZONS = (5, 10, 20, 40, 60, 120)
DEFAULT_STOP_MULTIPLES = (1.5, 2.0, 2.5, 3.0)


def cost_rate(commission_bps: float, slippage_bps: float) -> float:
    return (float(commission_bps) + float(slippage_bps)) / 10000.0


def parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in str(value).split(",") if v.strip()]


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        out = pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    out.columns = [str(c).lstrip("\ufeff") for c in out.columns]
    return out


def candidate_event(row: pd.Series) -> bool:
    trigger = str(row.get("entry_trigger", "NONE"))
    action = str(row.get("trade_action", "NO_TRADE"))
    tier = str(row.get("candidate_tier", "none"))
    return trigger != "NONE" or action in {"ENTRY_ALLOWED", "HOLD_OR_WAIT_TRIGGER", "WATCHLIST_PULLBACK_ONLY"} or tier != "none"


def entry_failure_reason(row: pd.Series, spec, entry_ok: bool, target_weight: float, next_open_ok: bool) -> str:
    if not next_open_ok:
        return "NO_NEXT_OPEN"
    if target_weight <= 0:
        return "TARGET_WEIGHT_ZERO"
    if entry_ok:
        return "PASS"
    reasons: List[str] = []
    if to_bool(row.get("algo_vol_extreme")):
        reasons.append("VOL_EXTREME")
    close = as_float(row.get("close"))
    sma_200 = as_float(row.get("sma_200"))
    if not valid_price(close) or not valid_price(sma_200) or close <= sma_200:
        reasons.append("BELOW_OR_MISSING_200D")
    atr = as_float(row.get("atr_14"))
    if not valid_price(atr):
        reasons.append("INVALID_ATR")
    trigger = str(row.get("entry_trigger", "NONE"))
    score = as_float(row.get("score_price_algo_total"), default=np.nan)
    if spec.entry_type == "breakout20" and not (to_bool(row.get("algo_breakout20_clean")) or trigger == "20D_BREAKOUT"):
        reasons.append("NO_20D_BREAKOUT")
    elif spec.entry_type == "breakout60" and not (to_bool(row.get("algo_breakout60_clean")) or trigger == "60D_BREAKOUT"):
        reasons.append("NO_60D_BREAKOUT")
    elif spec.entry_type == "pullback50" and not (to_bool(row.get("algo_pullback_to_50_bounce")) or trigger == "50D_PULLBACK_BOUNCE"):
        reasons.append("NO_50D_PULLBACK")
    elif spec.entry_type == "score75_trigger" and not (pd.notna(score) and score >= 75 and trigger not in {"NONE", "DEEP_DD_RECOVERY"}):
        reasons.append("SCORE_OR_TRIGGER_BLOCKED")
    return "|".join(reasons) if reasons else "ENTRY_RULE_FALSE"


def horizon_exit_metrics(
    df: pd.DataFrame,
    idx: int,
    horizon_days: int,
    stop_multiple: float,
    commission_bps: float,
    slippage_bps: float,
) -> Dict[str, object]:
    entry_idx = idx + 1
    exit_limit_idx = idx + int(horizon_days)
    if entry_idx >= len(df):
        return {"entry_available": False, "entry_block_reason": "NO_NEXT_OPEN"}
    if exit_limit_idx >= len(df):
        return {"entry_available": False, "entry_block_reason": "NO_FUTURE_WINDOW"}
    signal = df.iloc[idx]
    entry = df.iloc[entry_idx]
    entry_price = as_float(entry.get("open"))
    atr = as_float(signal.get("atr_14"))
    if not valid_price(entry_price) or not valid_price(atr):
        return {"entry_available": False, "entry_block_reason": "INVALID_ENTRY_DATA"}
    cr = cost_rate(commission_bps, slippage_bps)
    stop_price = entry_price - float(stop_multiple) * atr
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        return {"entry_available": False, "entry_block_reason": "INVALID_RISK_DISTANCE"}
    target_1r = entry_price + risk_per_share
    target_2r = entry_price + 2.0 * risk_per_share
    exit_idx = exit_limit_idx
    exit_price = as_float(df.iloc[exit_idx].get("close"))
    exit_reason = f"HORIZON_{horizon_days}D"
    stop_hit = False
    hit_1r = False
    hit_2r = False
    first_touch_type = "NONE"
    mfe_r = 0.0
    mae_r = 0.0

    for j in range(entry_idx, exit_limit_idx + 1):
        row = df.iloc[j]
        low = as_float(row.get("low"))
        high = as_float(row.get("high"))
        open_price = as_float(row.get("open"))
        stop_touched = valid_price(low) and low <= stop_price
        hit_1r_touched = valid_price(high) and high >= target_1r
        hit_2r_touched = valid_price(high) and high >= target_2r
        if first_touch_type == "NONE" and (stop_touched or hit_1r_touched or hit_2r_touched):
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
        if valid_price(high):
            mfe_r = max(mfe_r, (high - entry_price) / risk_per_share)
        if valid_price(low):
            mae_r = min(mae_r, (low - entry_price) / risk_per_share)
        if stop_touched:
            exit_idx = j
            exit_price = open_price if valid_price(open_price) and open_price <= stop_price else stop_price
            exit_reason = f"ATR_STOP_{stop_multiple:g}X"
            stop_hit = True
            break
        if hit_1r_touched:
            hit_1r = True
        if hit_2r_touched:
            hit_2r = True
    if not valid_price(exit_price):
        return {"entry_available": False, "entry_block_reason": "INVALID_EXIT_DATA"}
    entry_after_cost = entry_price * (1.0 + cr)
    exit_after_cost = exit_price * (1.0 - cr)
    gross_return = exit_price / entry_price - 1.0
    net_return = exit_after_cost / entry_after_cost - 1.0
    realized_r = (exit_after_cost - entry_after_cost) / risk_per_share
    return {
        "entry_available": True,
        "entry_block_reason": "PASS",
        "entry_date": df.iloc[entry_idx]["date"],
        "exit_date": df.iloc[exit_idx]["date"],
        "next_open": entry_price,
        "entry_price_after_cost": entry_after_cost,
        "exit_price": exit_price,
        "exit_price_after_cost": exit_after_cost,
        "stop_price": stop_price,
        "target_1r_price": target_1r,
        "target_2r_price": target_2r,
        "exit_reason": exit_reason,
        "realized_r_multiple": realized_r,
        "gross_return_pct": gross_return * 100.0,
        "net_return_pct": net_return * 100.0,
        "stop_hit": stop_hit,
        "hit_1r": hit_1r,
        "hit_2r": hit_2r,
        "first_touch_type": first_touch_type,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "holding_days": int(exit_idx - entry_idx),
    }


def aggregate_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    if "trade_event" in out.columns:
        out = out[out["trade_event"].astype(str).eq("AGGREGATE_EXIT")].copy()
    if out.empty:
        return pd.DataFrame()
    for col in ["entry_signal_date", "entry_date", "exit_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out.sort_values(["strategy_id", "entry_signal_date", "entry_date"]).drop_duplicates(["strategy_id", "entry_signal_date"], keep="first")


def add_overlap_weights(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    out = ledger.copy()
    out["label_overlap_count"] = 0
    out["sample_uniqueness_weight"] = 0.0
    for _, indexer in out.groupby(["symbol", "horizon_days"]).groups.items():
        idx = list(indexer)
        starts = pd.to_datetime(out.loc[idx, "entry_date"], errors="coerce")
        ends = pd.to_datetime(out.loc[idx, "exit_date"], errors="coerce")
        valid = starts.notna() & ends.notna()
        if not valid.any():
            continue
        valid_index = starts.loc[valid].index
        start_values = starts.loc[valid].astype("int64").to_numpy(dtype=np.int64)
        end_values = ends.loc[valid].astype("int64").to_numpy(dtype=np.int64)
        ordered_starts = np.sort(start_values)
        ordered_ends = np.sort(end_values)
        active_by_end = np.searchsorted(ordered_starts, end_values, side="right")
        ended_before_start = np.searchsorted(ordered_ends, start_values, side="left")
        overlap_counts = np.maximum(active_by_end - ended_before_start, 1)
        out.loc[valid_index, "label_overlap_count"] = overlap_counts.astype(int)
        out.loc[valid_index, "sample_uniqueness_weight"] = 1.0 / overlap_counts
    return out


def build_symbol_ledger(
    symbol: str,
    symbol_group: str,
    signals_path: Path,
    enriched_path: Path | None,
    trade_log_path: Path,
    horizons: Iterable[int],
    stop_multiples: Iterable[float],
    commission_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    if not signals_path.exists():
        return pd.DataFrame()
    signals = load_inputs(signals_path, enriched_path if enriched_path and enriched_path.exists() else None)
    signals["candidate_tier"] = signals.apply(candidate_tier, axis=1)
    signals["date"] = pd.to_datetime(signals["date"], errors="coerce")
    trades = aggregate_trades(read_csv(trade_log_path))
    trade_lookup = {}
    if not trades.empty:
        trade_lookup = {
            (str(row["strategy_id"]), pd.Timestamp(row["entry_signal_date"]).normalize()): row
            for _, row in trades.iterrows()
            if pd.notna(row.get("entry_signal_date"))
        }
    rows: List[Dict[str, object]] = []
    metric_cache: dict[tuple[int, int, float], dict[str, object]] = {}
    candidate_mask = signals.apply(candidate_event, axis=1)
    candidate_indices = signals.index[candidate_mask].tolist()
    for idx in candidate_indices:
        row = signals.loc[idx]
        signal_date = pd.Timestamp(row["date"]).normalize()
        next_open_ok = idx + 1 < len(signals) and valid_price(signals.loc[idx + 1, "open"])
        for spec in STRATEGIES:
            entry_ok = entry_signal(spec, row)
            target_weight = target_weight_for_entry(spec, row) if next_open_ok else 0.0
            failure_reason = entry_failure_reason(row, spec, entry_ok, target_weight, next_open_ok)
            matched_trade = trade_lookup.get((spec.strategy_id, signal_date))
            for horizon in horizons:
                for stop_multiple in stop_multiples:
                    metric_key = (int(idx), int(horizon), float(stop_multiple))
                    cached_metrics = metric_cache.get(metric_key)
                    if cached_metrics is None:
                        cached_metrics = horizon_exit_metrics(
                            signals,
                            idx,
                            int(horizon),
                            float(stop_multiple),
                            commission_bps,
                            slippage_bps,
                        )
                        metric_cache[metric_key] = cached_metrics
                    metrics = dict(cached_metrics)
                    entry_available = bool(metrics.get("entry_available", False))
                    variant_id = f"{spec.strategy_id}_{int(horizon)}D_{float(stop_multiple):g}ATR"
                    out = {
                        "symbol": symbol,
                        "symbol_group": symbol_group,
                        "date": row["date"],
                        "signal_idx": int(idx),
                        "strategy_id": spec.strategy_id,
                        "strategy_name": spec.name,
                        "strategy_group": spec.strategy_group,
                        "variant_id": variant_id,
                        "horizon_days": int(horizon),
                        "stop_multiple": float(stop_multiple),
                        "candidate_tier": row.get("candidate_tier", "none"),
                        "entry_trigger": row.get("entry_trigger", "NONE"),
                        "trade_action": row.get("trade_action", "NO_TRADE"),
                        "strict_signal_stage": row.get("strict_signal_stage", ""),
                        "research_signal_stage": row.get("research_signal_stage", ""),
                        "research_signal_action": row.get("research_signal_action", ""),
                        "research_signal_level": row.get("research_signal_level", np.nan),
                        "research_signal_score": as_float(row.get("research_signal_score")),
                        "research_signal_reason": row.get("research_signal_reason", ""),
                        "paper_tracking_weight": as_float(row.get("paper_tracking_weight")),
                        "score_price_algo_total": as_float(row.get("score_price_algo_total")),
                        "entry_signal_pass": bool(entry_ok),
                        "entry_failure_reason": failure_reason if failure_reason != "PASS" else metrics.get("entry_block_reason", "PASS"),
                        "next_open_available": bool(next_open_ok and entry_available),
                        "target_weight_pct": float(target_weight * 100.0),
                        "position_overlap_flag": False,
                        "matched_trade_flag": matched_trade is not None,
                        "matched_trade_exit_reason": matched_trade.get("exit_reason") if matched_trade is not None else "",
                        "matched_trade_net_return_pct": matched_trade.get("net_return_pct") if matched_trade is not None else np.nan,
                        "matched_trade_r_multiple": matched_trade.get("r_multiple") if matched_trade is not None else np.nan,
                    }
                    out.update(metrics)
                    rows.append(out)
    ledger = pd.DataFrame(rows)
    if not ledger.empty and not trades.empty:
        trade_ranges = trades[["strategy_id", "entry_date", "exit_date"]].dropna().copy()
        ledger_entry_dates = pd.to_datetime(ledger["entry_date"], errors="coerce")
        for strategy_id, trade_group in trade_ranges.groupby("strategy_id"):
            ledger_mask = ledger["strategy_id"].astype(str).eq(str(strategy_id)) & ledger_entry_dates.notna()
            if not ledger_mask.any():
                continue
            starts = pd.to_datetime(trade_group["entry_date"], errors="coerce").dropna().astype("int64").to_numpy(dtype=np.int64)
            ends = pd.to_datetime(trade_group["exit_date"], errors="coerce").dropna().astype("int64").to_numpy(dtype=np.int64)
            if len(starts) == 0 or len(ends) == 0:
                continue
            starts = np.sort(starts)
            ends = np.sort(ends)
            values = ledger_entry_dates.loc[ledger_mask].astype("int64").to_numpy(dtype=np.int64)
            active_by_date = np.searchsorted(starts, values, side="right")
            ended_before_date = np.searchsorted(ends, values, side="left")
            overlaps = active_by_date > ended_before_date
            overlap_index = ledger.loc[ledger_mask].index[overlaps]
            if len(overlap_index):
                ledger.loc[overlap_index, "position_overlap_flag"] = ~ledger.loc[overlap_index, "matched_trade_flag"].astype(bool)
    return add_overlap_weights(ledger)


def build_universe_jobs(args: argparse.Namespace) -> List[Dict[str, object]]:
    if args.universe_config:
        config = read_csv(Path(args.universe_config))
        if not config.empty and {"symbol", "signals", "trade_log"}.issubset(config.columns):
            rows = []
            for _, row in config.iterrows():
                rows.append(
                    {
                        "symbol": str(row.get("symbol")),
                        "symbol_group": str(row.get("symbol_group", "")),
                        "signals": Path(str(row.get("signals"))),
                        "trade_log": Path(str(row.get("trade_log"))),
                        "enriched": Path(str(row.get("enriched"))) if str(row.get("enriched", "")).strip() else None,
                    }
                )
            return rows
    return [
        {
            "symbol": args.symbol,
            "symbol_group": args.symbol_group,
            "signals": Path(args.signals),
            "trade_log": Path(args.trade_log),
            "enriched": Path(args.enriched) if args.enriched else None,
        }
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate-level backtest event ledger.")
    parser.add_argument("--signals", default="tsm_price_rule_output/tsm_daily_algorithmic_signals.csv")
    parser.add_argument("--trade-log", default="tsm_price_rule_output/tsm_backtest_trade_log.csv")
    parser.add_argument("--enriched", default="output/tsm_daily_10y_enriched.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--output-name", default="tsm_backtest_event_ledger.csv")
    parser.add_argument("--symbol", default="TSM")
    parser.add_argument("--symbol-group", default="foundry_idm")
    parser.add_argument("--universe-config", default="")
    parser.add_argument("--horizons", default="5,10,20,40,60,120")
    parser.add_argument("--stop-multiples", default="1.5,2.0,2.5,3.0")
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    horizons = parse_int_list(args.horizons) or list(DEFAULT_HORIZONS)
    stop_multiples = parse_float_list(args.stop_multiples) or list(DEFAULT_STOP_MULTIPLES)
    frames = []
    for job in build_universe_jobs(args):
        ledger = build_symbol_ledger(
            symbol=str(job["symbol"]),
            symbol_group=str(job.get("symbol_group", "")),
            signals_path=Path(job["signals"]),
            enriched_path=Path(job["enriched"]) if job.get("enriched") else None,
            trade_log_path=Path(job["trade_log"]),
            horizons=horizons,
            stop_multiples=stop_multiples,
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
        )
        if not ledger.empty:
            frames.append(ledger)
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output.to_csv(outdir / args.output_name, index=False)
    print("completed: backtest event ledger =", (outdir / args.output_name).resolve(), "rows=", len(output))


if __name__ == "__main__":
    main()

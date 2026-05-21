#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSM single-name backtest engine.

The rule engine creates close-based signals. This script tests those signals
with next-session open fills, ATR stops, cash/share accounting, and explicit
transaction costs.

Outputs:
- tsm_backtest_strategy_summary.csv
- tsm_backtest_trade_log.csv
- tsm_backtest_equity_curves.csv
- tsm_backtest_yearly_returns.csv
- tsm_backtest_report.md
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


TRADING_DAYS = 252
ALPHA_RESEARCH_TARGET_VOL = 0.10
ALPHA_RESEARCH_MAX_WEIGHT = 0.25
ALPHA_RESEARCH_ACCOUNT_RISK = 0.01


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    name: str
    description: str
    entry_type: str
    exit_on_sma50: bool
    sizing_mode: str
    strategy_group: str = "diagnostic"


@dataclass(frozen=True)
class ExitPolicy:
    stop_multiple: float = 2.0
    take_profit_1r_pct: float = 0.30
    take_profit_2r_pct: float = 0.30
    trail_remaining_pct: float = 0.40
    trailing_atr_multiple: float = 3.0
    exit_on_sma50: bool = True
    exit_on_sma200: bool = True


STRATEGIES: Tuple[StrategySpec, ...] = (
    StrategySpec(
        strategy_id="A_20D_BREAKOUT_2ATR",
        name="Strategy A - 20D breakout + 2ATR stop",
        description="20-day clean breakout, no extreme volatility, above 200D, 2ATR stop.",
        entry_type="breakout20",
        exit_on_sma50=True,
        sizing_mode="full",
    ),
    StrategySpec(
        strategy_id="B_50D_PULLBACK_2ATR",
        name="Strategy B - 50D pullback bounce + 2ATR stop",
        description="50-day moving-average pullback bounce, no extreme volatility, above 200D, 2ATR stop.",
        entry_type="pullback50",
        exit_on_sma50=True,
        sizing_mode="full",
    ),
    StrategySpec(
        strategy_id="C_SCORE75_TRIGGER_2ATR",
        name="Strategy C - score >= 75 + trigger + 2ATR stop",
        description="Algorithm score >= 75 with any entry trigger, no extreme volatility, above 200D, 2ATR stop.",
        entry_type="score75_trigger",
        exit_on_sma50=True,
        sizing_mode="full",
    ),
    StrategySpec(
        strategy_id="D_200D_TREND_2ATR",
        name="Strategy D - 200D trend hold + 2ATR stop",
        description="Hold only when close is above SMA200; exit below SMA200. 2ATR stop still applies.",
        entry_type="above200",
        exit_on_sma50=False,
        sizing_mode="full",
    ),
    StrategySpec(
        strategy_id="E_VOL_ADJUSTED_SCORE75",
        name="Strategy E - vol-adjusted score >= 75",
        description="Strategy C entry logic with position sizing from the rule engine risk columns.",
        entry_type="score75_trigger",
        exit_on_sma50=True,
        sizing_mode="vol_adjusted",
        strategy_group="diagnostic",
    ),
    StrategySpec(
        strategy_id="L1_60D_BREAKOUT_LIVE_LIKE",
        name="Live-like 1 - 60D breakout risk-sized",
        description="60-day breakout with 0.5% account-risk sizing, risk caps, partial profit taking, and trailing management.",
        entry_type="breakout60",
        exit_on_sma50=True,
        sizing_mode="live_like",
        strategy_group="live_like",
    ),
    StrategySpec(
        strategy_id="L2_50D_PULLBACK_LIVE_LIKE",
        name="Live-like 2 - 50D pullback risk-sized",
        description="50-day pullback bounce with 0.5% account-risk sizing, risk caps, partial profit taking, and trailing management.",
        entry_type="pullback50",
        exit_on_sma50=True,
        sizing_mode="live_like",
        strategy_group="live_like",
    ),
    StrategySpec(
        strategy_id="L3_SCORE75_TRIGGER_LIVE_LIKE",
        name="Live-like 3 - score >= 75 + trigger risk-sized",
        description="Score-trigger entry excluding deep drawdown research-only signals, risk-sized with partial profit taking.",
        entry_type="score75_trigger",
        exit_on_sma50=True,
        sizing_mode="live_like",
        strategy_group="live_like",
    ),
    StrategySpec(
        strategy_id="AR1_200D_TREND_ALPHA_RESEARCH",
        name="Alpha Research 1 - 200D trend core allocation",
        description="Alpha research mode: 200D trend core allocation with 1% account-risk, 10% target strategy vol, and 25% max weight. Prediction overlay remains a separate gate.",
        entry_type="above200",
        exit_on_sma50=False,
        sizing_mode="alpha_research",
        strategy_group="alpha_research",
    ),
)


TRADE_LOG_COLUMNS = [
    "strategy_id",
    "strategy_name",
    "strategy_group",
    "trade_event",
    "entry_signal_date",
    "entry_date",
    "entry_price",
    "entry_price_after_cost",
    "entry_atr_14",
    "stop_price",
    "stop_multiple",
    "target_weight_pct",
    "remaining_position_pct",
    "exit_signal_date",
    "exit_date",
    "exit_price",
    "exit_price_after_cost",
    "exit_reason",
    "holding_trading_days",
    "holding_calendar_days",
    "gross_return_pct",
    "net_return_pct",
    "portfolio_return_pct",
    "r_multiple",
    "entry_equity",
    "exit_equity",
    "net_pnl",
]


def pct(x: float) -> float:
    return float(x) * 100.0 if pd.notna(x) else np.nan


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


def valid_price(value) -> bool:
    return pd.notna(value) and float(value) > 0


def atr_stop_reason(stop_multiple: float) -> str:
    if abs(stop_multiple - round(stop_multiple)) < 1e-9:
        return f"ATR_STOP_{int(round(stop_multiple))}X"
    return f"ATR_STOP_{str(stop_multiple).replace('.', '_')}X"


def load_inputs(signals_path: Path, enriched_path: Optional[Path]) -> pd.DataFrame:
    signals = strip_bom_columns(pd.read_csv(signals_path, parse_dates=["date"]))
    signals = signals.sort_values("date").reset_index(drop=True)

    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "atr_14",
        "sma_50",
        "sma_200",
        "algo_vol_high",
        "algo_vol_extreme",
        "algo_breakout20_clean",
        "algo_breakout60_clean",
        "algo_pullback_to_50_bounce",
        "algo_deep_downtrend_avoid",
        "score_price_algo_total",
        "entry_trigger",
        "position_weight_if_0_5pct_account_risk",
    ]
    require_columns(signals, required, str(signals_path))

    if signals["date"].duplicated().any():
        dupes = signals.loc[signals["date"].duplicated(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"signals has duplicated dates: {dupes[:5]}")
    if not signals["date"].is_monotonic_increasing:
        raise ValueError("signals dates are not sorted ascending")

    for col in [
        "algo_vol_high",
        "algo_vol_extreme",
        "algo_breakout20_clean",
        "algo_breakout60_clean",
        "algo_pullback_to_50_bounce",
        "algo_deep_downtrend_avoid",
    ]:
        signals[col] = signals[col].map(to_bool)

    if enriched_path and enriched_path.exists():
        enriched = strip_bom_columns(pd.read_csv(enriched_path, parse_dates=["date"]))
        require_columns(enriched, ["date", "open", "high", "low", "close", "adj_close"], str(enriched_path))
        enriched = enriched.sort_values("date").reset_index(drop=True)
        if enriched["date"].duplicated().any():
            raise ValueError("enriched input has duplicated dates")

        compare_cols = ["open", "high", "low", "close", "adj_close"]
        check = signals[["date", *compare_cols]].merge(
            enriched[["date", *compare_cols]],
            on="date",
            how="inner",
            suffixes=("_signals", "_enriched"),
        )
        if check.empty:
            raise ValueError("signals and enriched inputs have no overlapping dates")
        for col in compare_cols:
            diff = (check[f"{col}_signals"] - check[f"{col}_enriched"]).abs().max()
            if pd.notna(diff) and diff > 1e-5:
                raise ValueError(f"signals/enriched mismatch for {col}; max_abs_diff={diff}")

    signals["derived_max_weight_by_vol"] = derive_max_weight(signals)
    return signals


def derive_max_weight(df: pd.DataFrame) -> pd.Series:
    weight = pd.Series(0.12, index=df.index, dtype=float)
    weight.loc[df["algo_vol_high"].map(to_bool)] = 0.07
    weight.loc[df["algo_vol_extreme"].map(to_bool)] = 0.04
    weight.loc[df["close"] < df["sma_200"]] = 0.03
    if "algo_deep_downtrend_avoid" in df.columns:
        weight.loc[df["algo_deep_downtrend_avoid"].map(to_bool)] = 0.00
    return weight.clip(lower=0.0, upper=1.0)


def common_entry_ok(row: pd.Series) -> bool:
    if to_bool(row.get("algo_vol_extreme")):
        return False
    if not valid_price(row.get("close")) or not valid_price(row.get("sma_200")):
        return False
    if float(row["close"]) <= float(row["sma_200"]):
        return False
    if not valid_price(row.get("atr_14")):
        return False
    return True


def entry_signal(spec: StrategySpec, row: pd.Series, score_threshold: float = 75.0) -> bool:
    if not common_entry_ok(row):
        return False

    entry_trigger = str(row.get("entry_trigger", "NONE"))
    score = as_float(row.get("score_price_algo_total"))

    if spec.entry_type == "breakout20":
        return to_bool(row.get("algo_breakout20_clean")) or entry_trigger == "20D_BREAKOUT"
    if spec.entry_type == "breakout60":
        return to_bool(row.get("algo_breakout60_clean")) or entry_trigger == "60D_BREAKOUT"
    if spec.entry_type == "pullback50":
        return to_bool(row.get("algo_pullback_to_50_bounce")) or entry_trigger == "50D_PULLBACK_BOUNCE"
    if spec.entry_type == "score75_trigger":
        return score >= score_threshold and entry_trigger not in {"NONE", "DEEP_DD_RECOVERY"}
    if spec.entry_type == "above200":
        return True
    raise ValueError(f"unknown entry_type: {spec.entry_type}")


def target_weight_for_entry(spec: StrategySpec, row: pd.Series) -> float:
    if spec.sizing_mode == "full":
        return 1.0

    risk_weight = as_float(row.get("position_weight_if_0_5pct_account_risk"), default=np.nan)
    max_weight = as_float(row.get("derived_max_weight_by_vol"), default=0.0)
    if pd.isna(risk_weight):
        risk_weight = max_weight
    if spec.sizing_mode == "alpha_research":
        risk_pct_2atr = as_float(row.get("risk_pct_2atr"), default=np.nan)
        alpha_risk_weight = ALPHA_RESEARCH_ACCOUNT_RISK / risk_pct_2atr if pd.notna(risk_pct_2atr) and risk_pct_2atr > 0 else np.nan
        vol_20d = as_float(row.get("vol_20d_ann"), default=np.nan)
        vol_target_weight = ALPHA_RESEARCH_TARGET_VOL / vol_20d if pd.notna(vol_20d) and vol_20d > 0 else ALPHA_RESEARCH_MAX_WEIGHT
        if pd.isna(alpha_risk_weight):
            alpha_risk_weight = ALPHA_RESEARCH_MAX_WEIGHT
        return float(np.clip(min(alpha_risk_weight, vol_target_weight, ALPHA_RESEARCH_MAX_WEIGHT), 0.0, 1.0))
    if spec.sizing_mode == "live_like":
        strategy_cap = 0.08
        return float(np.clip(min(risk_weight, max_weight, strategy_cap), 0.0, 1.0))
    return float(np.clip(min(risk_weight, max_weight), 0.0, 1.0))


def next_open_available(df: pd.DataFrame, idx: int) -> bool:
    if idx + 1 >= len(df):
        return False
    return valid_price(df.loc[idx + 1, "open"])


def max_consecutive_losses(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    max_run = 0
    current = 0
    for pnl in trades["net_pnl"]:
        if pd.notna(pnl) and pnl < 0:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def simulate_strategy(
    df: pd.DataFrame,
    spec: StrategySpec,
    initial_capital: float,
    cost_rate: float,
    stop_multiple: float = 2.0,
    score_threshold: float = 75.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    exit_policy = ExitPolicy(stop_multiple=stop_multiple, exit_on_sma50=spec.exit_on_sma50)
    tp1_fraction = min(exit_policy.take_profit_1r_pct, max(0.0, 1.0 - exit_policy.trail_remaining_pct))
    tp2_fraction = min(
        exit_policy.take_profit_2r_pct,
        max(0.0, 1.0 - exit_policy.trail_remaining_pct - tp1_fraction),
    )
    cash = float(initial_capital)
    shares = 0.0
    position: Optional[Dict] = None
    pending_entry: Optional[Dict] = None
    pending_exit: Optional[Dict] = None
    trade_rows: List[Dict] = []
    equity_rows: List[Dict] = []

    def execute_entry(fill_row: pd.Series, signal: Dict, idx: int) -> None:
        nonlocal cash, shares, position

        fill_price = as_float(fill_row["open"])
        atr = as_float(signal["atr_14"])
        target_weight = as_float(signal["target_weight"], default=0.0)
        equity_before = cash

        if not valid_price(fill_price) or atr <= 0 or target_weight <= 0 or equity_before <= 0:
            return

        fill_price_after_cost = fill_price * (1.0 + cost_rate)
        target_notional = equity_before * target_weight
        shares = target_notional / fill_price_after_cost
        total_cost = shares * fill_price_after_cost
        cash -= total_cost

        stop_price = fill_price - stop_multiple * atr
        risk_per_share = fill_price - stop_price
        position = {
            "strategy_id": spec.strategy_id,
            "strategy_name": spec.name,
            "entry_signal_date": signal["signal_date"],
            "entry_idx": idx,
            "entry_date": fill_row["date"],
            "entry_price": fill_price,
            "entry_price_after_cost": fill_price_after_cost,
            "entry_atr_14": atr,
            "stop_price": stop_price,
            "stop_multiple": stop_multiple,
            "target_weight": target_weight,
            "entry_equity": equity_before,
            "entry_total_cost": total_cost,
            "initial_shares": shares,
            "remaining_fraction": 1.0,
            "realized_proceeds": 0.0,
            "realized_net_pnl": 0.0,
            "realized_gross_pnl": 0.0,
            "tp1_done": False,
            "tp2_done": False,
            "risk_per_share": risk_per_share,
            "target_1r": fill_price + risk_per_share,
            "target_2r": fill_price + 2.0 * risk_per_share,
        }

    def append_trade_row(
        exit_row: pd.Series,
        exit_price: float,
        exit_price_after_cost: float,
        exit_reason: str,
        exit_signal_date,
        idx: int,
        trade_event: str,
        net_pnl: float,
        entry_cost_basis: float,
        remaining_fraction: float,
        gross_return_override: Optional[float] = None,
    ) -> None:
        if position is None:
            return
        gross_return = gross_return_override if gross_return_override is not None else exit_price / position["entry_price"] - 1.0
        net_return = net_pnl / entry_cost_basis if entry_cost_basis > 0 else np.nan
        portfolio_return = net_pnl / position["entry_equity"] if position["entry_equity"] else np.nan
        risk_per_share = position.get("risk_per_share", position["entry_price"] - position["stop_price"])
        if trade_event == "AGGREGATE_EXIT":
            risk_per_share = position.get("risk_per_share", risk_per_share)
            r_multiple = (net_pnl / position["initial_shares"]) / risk_per_share if risk_per_share > 0 and position["initial_shares"] > 0 else np.nan
        else:
            r_multiple = (exit_price - position["entry_price"]) / risk_per_share if risk_per_share > 0 else np.nan
        holding_calendar_days = (exit_row["date"] - position["entry_date"]).days
        holding_trading_days = idx - int(position["entry_idx"])

        trade_rows.append(
            {
                "strategy_id": spec.strategy_id,
                "strategy_name": spec.name,
                "strategy_group": spec.strategy_group,
                "trade_event": trade_event,
                "entry_signal_date": position["entry_signal_date"],
                "entry_date": position["entry_date"],
                "entry_price": position["entry_price"],
                "entry_price_after_cost": position["entry_price_after_cost"],
                "entry_atr_14": position["entry_atr_14"],
                "stop_price": position["stop_price"],
                "stop_multiple": position["stop_multiple"],
                "target_weight_pct": pct(position["target_weight"]),
                "remaining_position_pct": pct(remaining_fraction),
                "exit_signal_date": exit_signal_date,
                "exit_date": exit_row["date"],
                "exit_price": exit_price,
                "exit_price_after_cost": exit_price_after_cost,
                "exit_reason": exit_reason,
                "holding_trading_days": holding_trading_days,
                "holding_calendar_days": holding_calendar_days,
                "gross_return_pct": pct(gross_return),
                "net_return_pct": pct(net_return),
                "portfolio_return_pct": pct(portfolio_return),
                "r_multiple": r_multiple,
                "entry_equity": position["entry_equity"],
                "exit_equity": cash,
                "net_pnl": net_pnl,
            }
        )

    def execute_partial_exit(
        exit_row: pd.Series,
        exit_price: float,
        exit_reason: str,
        idx: int,
        fraction_of_initial: float,
    ) -> None:
        nonlocal cash, shares, position
        if position is None or shares <= 0:
            return
        shares_to_close = min(shares, position["initial_shares"] * fraction_of_initial)
        if shares_to_close <= 0:
            return
        exit_price_after_cost = exit_price * (1.0 - cost_rate)
        proceeds = shares_to_close * exit_price_after_cost
        entry_cost_basis = shares_to_close * position["entry_price_after_cost"]
        net_pnl = proceeds - entry_cost_basis
        gross_pnl = shares_to_close * (exit_price - position["entry_price"])
        cash += proceeds
        shares -= shares_to_close
        position["realized_proceeds"] += proceeds
        position["realized_net_pnl"] += net_pnl
        position["realized_gross_pnl"] += gross_pnl
        position["remaining_fraction"] = shares / position["initial_shares"] if position["initial_shares"] else 0.0
        append_trade_row(
            exit_row=exit_row,
            exit_price=exit_price,
            exit_price_after_cost=exit_price_after_cost,
            exit_reason=exit_reason,
            exit_signal_date=exit_row["date"],
            idx=idx,
            trade_event="PARTIAL_EXIT",
            net_pnl=net_pnl,
            entry_cost_basis=entry_cost_basis,
            remaining_fraction=position["remaining_fraction"],
        )

    def execute_exit(
        exit_row: pd.Series,
        exit_price: float,
        exit_reason: str,
        exit_signal_date,
        idx: int,
    ) -> None:
        nonlocal cash, shares, position, pending_exit

        if position is None or shares <= 0:
            return

        exit_price = as_float(exit_price)
        if not valid_price(exit_price):
            return

        exit_price_after_cost = exit_price * (1.0 - cost_rate)
        proceeds = shares * exit_price_after_cost
        cash += proceeds
        remaining_entry_cost = shares * position["entry_price_after_cost"]
        final_leg_pnl = proceeds - remaining_entry_cost
        net_pnl = position.get("realized_net_pnl", 0.0) + final_leg_pnl
        final_leg_gross_pnl = shares * (exit_price - position["entry_price"])
        gross_pnl = position.get("realized_gross_pnl", 0.0) + final_leg_gross_pnl
        gross_basis = position["initial_shares"] * position["entry_price"]
        gross_return = gross_pnl / gross_basis if gross_basis > 0 else np.nan
        append_trade_row(
            exit_row=exit_row,
            exit_price=exit_price,
            exit_price_after_cost=exit_price_after_cost,
            exit_reason=exit_reason,
            exit_signal_date=exit_signal_date,
            idx=idx,
            trade_event="AGGREGATE_EXIT",
            net_pnl=net_pnl,
            entry_cost_basis=position["entry_total_cost"],
            remaining_fraction=0.0,
            gross_return_override=gross_return,
        )

        shares = 0.0
        position = None
        pending_exit = None

    for idx, row in df.iterrows():
        if pending_exit is not None and position is not None:
            execute_exit(
                exit_row=row,
                exit_price=as_float(row["open"]),
                exit_reason=pending_exit["reason"],
                exit_signal_date=pending_exit["signal_date"],
                idx=idx,
            )
            pending_exit = None

        if pending_entry is not None and position is None:
            execute_entry(row, pending_entry, idx)
            pending_entry = None

        if position is not None:
            low = as_float(row["low"])
            high = as_float(row["high"])
            open_price = as_float(row["open"])
            stop_price = as_float(position["stop_price"])
            if valid_price(low) and valid_price(stop_price) and low <= stop_price:
                stop_fill = open_price if valid_price(open_price) and open_price <= stop_price else stop_price
                execute_exit(
                    exit_row=row,
                    exit_price=stop_fill,
                    exit_reason=atr_stop_reason(stop_multiple),
                    exit_signal_date=row["date"],
                    idx=idx,
                )

        if position is not None and spec.sizing_mode == "live_like":
            high = as_float(row["high"])
            if valid_price(high) and (not position.get("tp1_done")) and high >= position["target_1r"]:
                execute_partial_exit(row, position["target_1r"], "TAKE_PROFIT_1R_PARTIAL", idx, tp1_fraction)
                if position is not None:
                    position["tp1_done"] = True
            if valid_price(high) and position is not None and (not position.get("tp2_done")) and high >= position["target_2r"]:
                execute_partial_exit(row, position["target_2r"], "TAKE_PROFIT_2R_PARTIAL", idx, tp2_fraction)
                if position is not None:
                    position["tp2_done"] = True
            close_for_trail = as_float(row.get("close"))
            atr_for_trail = as_float(row.get("atr_14"))
            if position is not None and valid_price(close_for_trail) and atr_for_trail > 0 and (position.get("tp1_done") or position.get("tp2_done")):
                trail_stop = close_for_trail - exit_policy.trailing_atr_multiple * atr_for_trail
                position["stop_price"] = max(as_float(position["stop_price"]), trail_stop)

        close_price = as_float(row["close"])
        if position is not None and valid_price(close_price):
            equity = cash + shares * close_price
            position_weight = shares * close_price / equity if equity else 0.0
            position_state = "LONG"
        else:
            equity = cash
            position_weight = 0.0
            position_state = "CASH"

        equity_rows.append(
            {
                "date": row["date"],
                "strategy_id": spec.strategy_id,
                "strategy_name": spec.name,
                "strategy_group": spec.strategy_group,
                "equity": equity,
                "position_weight": position_weight,
                "position_state": position_state,
                "cash": cash,
                "shares": shares,
            }
        )

        if idx >= len(df) - 1:
            continue

        if position is not None and pending_exit is None:
            if exit_policy.exit_on_sma200 and valid_price(row.get("sma_200")) and valid_price(row.get("close")) and row["close"] < row["sma_200"]:
                if next_open_available(df, idx):
                    pending_exit = {"reason": "CLOSE_BELOW_200D", "signal_date": row["date"]}
            elif (
                spec.exit_on_sma50
                and valid_price(row.get("sma_50"))
                and valid_price(row.get("close"))
                and row["close"] < row["sma_50"]
            ):
                if next_open_available(df, idx):
                    pending_exit = {"reason": "CLOSE_BELOW_50D", "signal_date": row["date"]}

        if (
            position is None
            and pending_entry is None
            and entry_signal(spec, row, score_threshold=score_threshold)
            and next_open_available(df, idx)
        ):
            weight = target_weight_for_entry(spec, row)
            if weight > 0:
                pending_entry = {
                    "signal_date": row["date"],
                    "atr_14": row["atr_14"],
                    "target_weight": weight,
                }

    if position is not None:
        last = df.iloc[-1]
        execute_exit(
            exit_row=last,
            exit_price=as_float(last["close"]),
            exit_reason="END_OF_DATA_LIQUIDATION",
            exit_signal_date=last["date"],
            idx=len(df) - 1,
        )
        equity_rows[-1]["equity"] = cash
        equity_rows[-1]["position_weight"] = 0.0
        equity_rows[-1]["position_state"] = "CASH"
        equity_rows[-1]["cash"] = cash
        equity_rows[-1]["shares"] = 0.0

    curve = pd.DataFrame(equity_rows)
    curve["daily_return"] = curve["equity"].pct_change().fillna(0.0)
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0

    trades = pd.DataFrame(trade_rows, columns=TRADE_LOG_COLUMNS)
    return curve, trades


def summarize_strategy(
    curve: pd.DataFrame,
    trades: pd.DataFrame,
    spec: StrategySpec,
    initial_capital: float,
    stop_multiple: float = 2.0,
    score_threshold: float = 75.0,
    commission_bps: float = 1.0,
    slippage_bps: float = 5.0,
) -> Dict:
    if curve.empty:
        raise ValueError(f"empty equity curve for {spec.strategy_id}")

    start_date = curve["date"].iloc[0]
    end_date = curve["date"].iloc[-1]
    years = max((end_date - start_date).days / 365.25, 1 / 365.25)
    start_equity = float(curve["equity"].iloc[0])
    end_equity = float(curve["equity"].iloc[-1])
    total_return = end_equity / initial_capital - 1.0
    cagr = (end_equity / initial_capital) ** (1.0 / years) - 1.0 if end_equity > 0 else np.nan
    daily_ret = curve["daily_return"].replace([np.inf, -np.inf], np.nan).dropna()
    ann_vol = daily_ret.std(ddof=0) * math.sqrt(TRADING_DAYS) if len(daily_ret) > 1 else np.nan
    sharpe = daily_ret.mean() / daily_ret.std(ddof=0) * math.sqrt(TRADING_DAYS) if daily_ret.std(ddof=0) > 0 else np.nan
    downside = daily_ret[daily_ret < 0]
    sortino = daily_ret.mean() / downside.std(ddof=0) * math.sqrt(TRADING_DAYS) if len(downside) > 1 and downside.std(ddof=0) > 0 else np.nan
    max_dd = curve["drawdown"].min()
    calmar = cagr / abs(max_dd) if pd.notna(max_dd) and max_dd < 0 else np.nan

    aggregate_trades = trades
    if not trades.empty and "trade_event" in trades.columns:
        aggregate_trades = trades[trades["trade_event"].eq("AGGREGATE_EXIT")].copy()

    if aggregate_trades.empty:
        win_rate = np.nan
        profit_factor = np.nan
        avg_gain = np.nan
        avg_loss = np.nan
        avg_hold = np.nan
        max_loss_run = 0
    else:
        wins = aggregate_trades.loc[aggregate_trades["net_pnl"] > 0, "net_pnl"]
        losses = aggregate_trades.loc[aggregate_trades["net_pnl"] < 0, "net_pnl"]
        win_rate = (aggregate_trades["net_pnl"] > 0).mean()
        profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else np.inf
        avg_gain = aggregate_trades.loc[aggregate_trades["net_return_pct"] > 0, "net_return_pct"].mean()
        avg_loss = aggregate_trades.loc[aggregate_trades["net_return_pct"] < 0, "net_return_pct"].mean()
        avg_hold = aggregate_trades["holding_trading_days"].mean()
        max_loss_run = max_consecutive_losses(aggregate_trades)

    return {
        "strategy_id": spec.strategy_id,
        "strategy_name": spec.name,
        "strategy_group": spec.strategy_group,
        "description": spec.description,
        "stop_multiple": stop_multiple,
        "score_threshold": score_threshold,
        "commission_bps": commission_bps,
        "slippage_bps": slippage_bps,
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "start_equity": start_equity,
        "end_equity": end_equity,
        "total_return_pct": pct(total_return),
        "cagr_pct": pct(cagr),
        "annualized_volatility_pct": pct(ann_vol),
        "max_drawdown_pct": pct(max_dd),
        "sharpe_zero_rf": sharpe,
        "sortino_zero_rf": sortino,
        "calmar_ratio": calmar,
        "trade_count": int(len(aggregate_trades)),
        "win_rate_pct": pct(win_rate),
        "profit_factor": profit_factor,
        "avg_trade_gain_pct": avg_gain,
        "avg_trade_loss_pct": avg_loss,
        "avg_holding_trading_days": avg_hold,
        "max_consecutive_losses": int(max_loss_run),
        "exposure_days_pct": pct((curve["position_weight"] > 0).mean()),
        "avg_position_weight_pct": pct(curve["position_weight"].mean()),
    }


def build_yearly_returns(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy_id, strategy_name), group in curves.groupby(["strategy_id", "strategy_name"]):
        g = group.sort_values("date").copy()
        g["year"] = g["date"].dt.year
        for year, part in g.groupby("year"):
            year_return = (1.0 + part["daily_return"]).prod() - 1.0
            year_drawdown = (part["equity"] / part["equity"].cummax() - 1.0).min()
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "strategy_name": strategy_name,
                    "year": int(year),
                    "year_return_pct": pct(year_return),
                    "year_max_drawdown_pct": pct(year_drawdown),
                    "exposure_days_pct": pct((part["position_weight"] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def make_output_curves(curves: pd.DataFrame) -> pd.DataFrame:
    out = curves.copy()
    out["daily_return_pct"] = out["daily_return"].map(pct)
    out["drawdown_pct"] = out["drawdown"].map(pct)
    out["position_weight_pct"] = out["position_weight"].map(pct)
    return out[
        [
            "date",
            "strategy_id",
            "strategy_name",
            "strategy_group",
            "equity",
            "daily_return_pct",
            "drawdown_pct",
            "position_weight_pct",
            "position_state",
            "cash",
            "shares",
        ]
    ]


def fmt_pct(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    if value == np.inf:
        return "inf"
    return f"{value:.{digits}f}%"


def fmt_num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    if value == np.inf:
        return "inf"
    return f"{value:.{digits}f}"


def write_report(
    outdir: Path,
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    cost_rate: float,
    commission_bps: float,
    slippage_bps: float,
    stop_multiple: float,
    score_threshold: float,
) -> None:
    ranked = summary.sort_values("cagr_pct", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0] if not ranked.empty else None

    lines = [
        "# TSMC Backtest Report",
        "",
        "## Execution assumptions",
        "- Signals are generated from same-day close data.",
        "- Entries and moving-average exits are filled at the next trading day's open.",
        "- ATR stops are filled intraday at the stop price, or at the open if the market gaps below the stop.",
        f"- ATR stop multiple: {stop_multiple:.2f}x.",
        f"- Score-trigger strategies use score threshold: {score_threshold:.1f}.",
        f"- Alpha research sizing uses target vol {ALPHA_RESEARCH_TARGET_VOL * 100:.1f}%, max weight {ALPHA_RESEARCH_MAX_WEIGHT * 100:.1f}%, and account risk {ALPHA_RESEARCH_ACCOUNT_RISK * 100:.1f}% per 2ATR stop.",
        f"- Round-trip costs are modeled per side: commission {commission_bps:.2f} bps + slippage {slippage_bps:.2f} bps = {cost_rate * 10000:.2f} bps.",
        "- Dividends, tax, FX, and ADR-specific fees are excluded.",
        "",
        "## Strategy summary",
        "",
        "| Group | Strategy | CAGR | Total Return | MDD | Sharpe | Win Rate | Profit Factor | Trades | Exposure |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in ranked.iterrows():
        lines.append(
            "| "
            f"{row.get('strategy_group', 'diagnostic')} | "
            f"{row['strategy_id']} | "
            f"{fmt_pct(row['cagr_pct'])} | "
            f"{fmt_pct(row['total_return_pct'])} | "
            f"{fmt_pct(row['max_drawdown_pct'])} | "
            f"{fmt_num(row['sharpe_zero_rf'])} | "
            f"{fmt_pct(row['win_rate_pct'])} | "
            f"{fmt_num(row['profit_factor'])} | "
            f"{int(row['trade_count'])} | "
            f"{fmt_pct(row['exposure_days_pct'])} |"
        )

    if best is not None:
        lines.extend(
            [
                "",
                "## Highest CAGR strategy",
                f"- Strategy: {best['strategy_id']}",
                f"- CAGR: {fmt_pct(best['cagr_pct'])}",
                f"- Max drawdown: {fmt_pct(best['max_drawdown_pct'])}",
                f"- Trades: {int(best['trade_count'])}",
            ]
        )

    if not trades.empty:
        exit_counts = trades.groupby(["strategy_id", "trade_event", "exit_reason"]).size().reset_index(name="count")
        lines.extend(["", "## Exit reason counts", "", "| Strategy | Exit reason | Count |", "|---|---|---:|"])
        for _, row in exit_counts.iterrows():
            lines.append(f"| {row['strategy_id']} | {row['trade_event']}:{row['exit_reason']} | {int(row['count'])} |")

    lines.extend(
        [
            "",
            "## Output files",
            "- tsm_backtest_strategy_summary.csv",
            "- tsm_backtest_trade_log.csv",
            "- tsm_backtest_equity_curves.csv",
            "- tsm_backtest_yearly_returns.csv",
            "",
            "This report is research tooling, not investment advice.",
        ]
    )

    (outdir / "tsm_backtest_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_backtests(
    signals: pd.DataFrame,
    initial_capital: float,
    commission_bps: float,
    slippage_bps: float,
    stop_multiple: float = 2.0,
    score_threshold: float = 75.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cost_rate = (commission_bps + slippage_bps) / 10000.0
    curve_parts = []
    trade_parts = []
    summary_rows = []

    for spec in STRATEGIES:
        curve, trades = simulate_strategy(
            signals,
            spec,
            initial_capital,
            cost_rate,
            stop_multiple=stop_multiple,
            score_threshold=score_threshold,
        )
        curve_parts.append(curve)
        if not trades.empty:
            trade_parts.append(trades)
        summary_rows.append(
            summarize_strategy(
                curve,
                trades,
                spec,
                initial_capital,
                stop_multiple=stop_multiple,
                score_threshold=score_threshold,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
            )
        )

    curves = pd.concat(curve_parts, ignore_index=True)
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    summary = pd.DataFrame(summary_rows)
    yearly = build_yearly_returns(curves)
    return summary, trades, make_output_curves(curves), yearly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest TSM rule-engine strategies.")
    parser.add_argument("--signals", default="tsm_price_rule_output/tsm_daily_algorithmic_signals.csv")
    parser.add_argument("--enriched", default="output/tsm_daily_10y_enriched.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--initial-capital", type=float, default=1.0)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stop-multiple", type=float, default=2.0)
    parser.add_argument("--score-threshold", type=float, default=75.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    signals = load_inputs(Path(args.signals), Path(args.enriched) if args.enriched else None)
    summary, trades, curves, yearly = run_backtests(
        signals=signals,
        initial_capital=args.initial_capital,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        stop_multiple=args.stop_multiple,
        score_threshold=args.score_threshold,
    )

    summary.to_csv(outdir / "tsm_backtest_strategy_summary.csv", index=False)
    trades.to_csv(outdir / "tsm_backtest_trade_log.csv", index=False)
    curves.to_csv(outdir / "tsm_backtest_equity_curves.csv", index=False)
    yearly.to_csv(outdir / "tsm_backtest_yearly_returns.csv", index=False)

    write_report(
        outdir=outdir,
        summary=summary,
        trades=trades,
        cost_rate=(args.commission_bps + args.slippage_bps) / 10000.0,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        stop_multiple=args.stop_multiple,
        score_threshold=args.score_threshold,
    )

    print("완료: backtest outputs =", outdir.resolve())
    print(summary[["strategy_id", "cagr_pct", "max_drawdown_pct", "trade_count", "win_rate_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()

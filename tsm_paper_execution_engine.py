#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Broker-free paper execution engine.

This engine simulates paper orders from approved order intents and portfolio
risk decisions. It never calls a live broker and always records that live order
submission is disabled by design.

Outputs:
- tsm_paper_orders.csv
- tsm_paper_fills.csv
- tsm_paper_positions.csv
- tsm_paper_slippage_report.csv
- tsm_paper_oms_quality_checks.csv
- tsm_paper_oms_report.md
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tsm_core.config import PaperExecutionConfig, PaperOmsConfig, load_run_config
from tsm_core.execution import FillStatus, IntentStatus, OrderStatus, PaperFill, PaperOrder, PaperPosition
from tsm_core.io import as_float, check_row, read_csv, require_columns, strip_bom_columns
from tsm_core.universe import load_universe_members


ORDER_COLUMNS = [
    "order_id",
    "intent_id",
    "symbol",
    "side",
    "order_type",
    "status",
    "target_weight",
    "target_notional",
    "quantity",
    "reference_price",
    "limit_price",
    "stop_price",
    "time_in_force",
    "submitted_at_utc",
    "signal_asof_date",
    "expected_fill_date",
    "fill_model",
    "live_trading_status",
]

FILL_COLUMNS = [
    "fill_id",
    "order_id",
    "intent_id",
    "symbol",
    "side",
    "fill_type",
    "status",
    "fill_date",
    "fill_price",
    "raw_price",
    "quantity",
    "gross_notional",
    "commission_bps",
    "slippage_bps",
    "total_cost_bps",
    "reason",
    "created_at_utc",
]

POSITION_COLUMNS = [
    "symbol",
    "asof_date",
    "quantity",
    "average_price",
    "market_price",
    "market_value",
    "cash",
    "equity",
    "weight",
    "realized_pnl",
    "unrealized_pnl",
    "position_state",
    "updated_at_utc",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_id(*parts: object, prefix: str) -> str:
    text = "|".join("" if part is None else str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def clean_symbol(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    text = str(value).strip().upper()
    return text if text and text != "NAN" else default


def read_csv_if_exists(path: Path, columns: list[str] | None = None, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    df = strip_bom_columns(pd.read_csv(path, **kwargs))
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[columns].copy()
    return df


def normalize_weight(value: object) -> float:
    parsed = as_float(value)
    if pd.isna(parsed):
        return 0.0
    return float(parsed / 100.0 if abs(parsed) > 1 else parsed)


def load_signals(path: Path) -> pd.DataFrame:
    df = read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    require_columns(df, ["date", "open", "high", "low", "close"], str(path))
    return df


def estimate_slippage_bps(
    side: str,
    order_notional: float,
    adv: float,
    spread_bps: float,
    vol_bps: float,
    open_gap_bps: float = 0.0,
    event_day: bool = False,
    config: PaperExecutionConfig | None = None,
) -> float:
    cfg = config or PaperExecutionConfig()
    participation = 0.0 if adv <= 0 else max(0.0, order_notional / adv) * float(cfg.participation_bps_multiplier) * 100.0
    volatility_component = max(0.0, vol_bps / 10000.0) * float(cfg.volatility_bps_multiplier)
    gap_component = abs(open_gap_bps) / 10000.0 * float(cfg.gap_penalty_bps)
    event_component = float(cfg.event_day_penalty_bps) if event_day else 0.0
    return float(max(0.0, spread_bps + volatility_component + participation + gap_component + event_component))


def resolve_same_bar_stop_target(bar: pd.Series | dict, stop: float, target: float, side: str = "BUY") -> tuple[str, float]:
    open_price = as_float(bar.get("open"))
    high = as_float(bar.get("high"))
    low = as_float(bar.get("low"))
    if str(side).upper() in {"BUY", "LONG"}:
        stop_hit = pd.notna(low) and pd.notna(stop) and low <= stop
        target_hit = pd.notna(high) and pd.notna(target) and high >= target
        if stop_hit:
            return "STOP", float(open_price if pd.notna(open_price) and open_price <= stop else stop)
        if target_hit:
            return "TARGET", float(target)
    return "HOLD", np.nan


def resolve_intraday_stop_target(minute_bars: pd.DataFrame | None, fill_date: str, stop: float, target: float) -> tuple[str, float, str]:
    if minute_bars is None or minute_bars.empty or not fill_date:
        return "HOLD", np.nan, "NO_INTRADAY_PATH"
    bars = minute_bars.copy()
    if "date" not in bars.columns:
        return "HOLD", np.nan, "NO_INTRADAY_PATH"
    dates = pd.to_datetime(bars["date"], errors="coerce")
    session = bars[dates.dt.date.astype(str).eq(str(fill_date))].copy()
    if session.empty:
        return "HOLD", np.nan, "NO_INTRADAY_SESSION"
    session["_ts"] = pd.to_datetime(session["date"], errors="coerce")
    session = session.sort_values("_ts")
    for _, bar in session.iterrows():
        high = as_float(bar.get("high"))
        low = as_float(bar.get("low"))
        open_price = as_float(bar.get("open"))
        stop_hit = pd.notna(low) and pd.notna(stop) and low <= stop
        target_hit = pd.notna(high) and pd.notna(target) and high >= target
        if stop_hit and target_hit:
            raw = float(open_price if pd.notna(open_price) and open_price <= stop else stop)
            return "STOP", raw, "AMBIGUOUS_INTRADAY_BAR_STOP_CONSERVATIVE"
        if stop_hit:
            raw = float(open_price if pd.notna(open_price) and open_price <= stop else stop)
            return "STOP", raw, "INTRADAY_1M_STOP_FIRST"
        if target_hit:
            return "TARGET", float(target), "INTRADAY_1M_TARGET_FIRST"
    return "HOLD", np.nan, "INTRADAY_1M_NO_TOUCH"


def next_bar_after(signals: pd.DataFrame, asof_date: str) -> pd.Series | None:
    asof = pd.Timestamp(asof_date).normalize()
    future = signals[signals["date"].dt.normalize() > asof]
    if future.empty:
        return None
    return future.iloc[0]


def upsert_rows(existing: pd.DataFrame, rows: list[dict[str, object]], key: str, columns: list[str]) -> pd.DataFrame:
    new = pd.DataFrame(rows)
    for col in columns:
        if col not in new.columns:
            new[col] = np.nan
    if existing.empty:
        return new[columns].copy()
    keys = set(new[key].dropna().astype(str))
    kept = existing[~existing[key].astype(str).isin(keys)].copy()
    return pd.concat([kept, new[columns]], ignore_index=True)[columns].copy()


def build_order_and_fills(
    intent: pd.Series,
    risk_decision: pd.Series,
    signals: pd.DataFrame,
    oms_config: PaperOmsConfig,
    execution_config: PaperExecutionConfig,
    commission_bps: float,
    intraday_context: pd.Series | None = None,
    minute_bars: pd.DataFrame | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    intent_id = str(intent.get("intent_id", ""))
    symbol = clean_symbol(intent.get("symbol"), clean_symbol(risk_decision.get("symbol")))
    order_id = stable_id(intent_id, "paper", prefix="pord")
    approved_weight = normalize_weight(risk_decision.get("approved_weight", intent.get("target_weight", 0.0)))
    reference_price = as_float(intent.get("entry_reference_price"))
    target_notional = float(oms_config.account_equity) * approved_weight
    estimated_quantity = target_notional / reference_price if pd.notna(reference_price) and reference_price > 0 else 0.0
    next_bar = next_bar_after(signals, str(intent.get("asof_date", "")))
    expected_fill_date = pd.Timestamp(next_bar["date"]).date().isoformat() if next_bar is not None else ""
    status = OrderStatus.PAPER_SUBMITTED.value if approved_weight > 0 else OrderStatus.BLOCKED.value
    reason = str(risk_decision.get("block_reason", "PASS"))
    fills: list[dict[str, object]] = []
    intraday_status = str(intraday_context.get("execution_minute_feature_status", "")) if intraday_context is not None else ""
    intraday_path_source = "DAILY_OHLC_FALLBACK"
    intraday_slippage_bps = np.nan

    if str(intent.get("status")) != IntentStatus.APPROVED.value or str(risk_decision.get("portfolio_status")) != IntentStatus.APPROVED.value:
        status = OrderStatus.BLOCKED.value
    elif next_bar is None:
        status = OrderStatus.PAPER_SUBMITTED.value
        reason = "AWAITING_NEXT_OPEN"
    else:
        raw_open = as_float(next_bar.get("open"))
        limit_price = as_float(intent.get("limit_price"))
        order_type = str(intent.get("order_type", "MARKET")).upper()
        if order_type == "LIMIT" and pd.notna(limit_price) and raw_open > limit_price:
            status = OrderStatus.EXPIRED.value
            reason = "EXPIRED_UNFILLED_LIMIT_OPEN_OR_CANCEL"
        elif raw_open > 0 and approved_weight > 0:
            adv = as_float(next_bar.get("dollar_volume_ma_20"), execution_config.default_adv)
            vol_bps = abs(as_float(next_bar.get("atr_14_pct"), 0.0)) * 10000.0
            if intraday_context is not None:
                intraday_vol = as_float(intraday_context.get("m1_realized_range_pct"), as_float(intraday_context.get("execution_minute_realized_range_pct")))
                if pd.notna(intraday_vol):
                    intraday_slippage_bps = max(0.0, float(intraday_vol) * 10000.0)
                    vol_bps = max(vol_bps, intraday_slippage_bps)
                    intraday_path_source = "INTRADAY_FEATURE_CONTEXT"
            open_gap_bps = abs(as_float(next_bar.get("open_gap_pct"), 0.0)) * 10000.0
            event_day = str(next_bar.get("algo_event_shock_day", "False")).lower() in {"true", "1", "yes"}
            slippage_bps = estimate_slippage_bps(
                str(intent.get("side", "BUY")),
                target_notional,
                adv,
                float(execution_config.half_spread_bps),
                vol_bps,
                open_gap_bps,
                event_day,
                execution_config,
            )
            total_cost_bps = float(commission_bps) + slippage_bps
            fill_price = raw_open * (1.0 + total_cost_bps / 10000.0)
            quantity = target_notional / fill_price if fill_price > 0 else 0.0
            entry_fill = PaperFill(
                fill_id=stable_id(order_id, "ENTRY", expected_fill_date, prefix="pfl"),
                order_id=order_id,
                intent_id=intent_id,
                symbol=symbol,
                side="BUY",
                fill_type="ENTRY",
                status=FillStatus.FILLED.value,
                fill_date=expected_fill_date,
                fill_price=float(fill_price),
                raw_price=float(raw_open),
                quantity=float(quantity),
                gross_notional=float(quantity * fill_price),
                commission_bps=float(commission_bps),
                slippage_bps=float(slippage_bps),
                total_cost_bps=float(total_cost_bps),
                reason=str(execution_config.fill_mode),
                created_at_utc=now_utc_iso(),
            ).to_dict()
            fills.append(entry_fill)
            status = OrderStatus.FILLED.value
            stop_price = as_float(intent.get("stop_price"))
            target_price = fill_price + 2.0 * (fill_price - stop_price) if pd.notna(stop_price) and fill_price > stop_price else np.nan
            exit_reason, exit_raw, first_touch_source = resolve_intraday_stop_target(minute_bars, expected_fill_date, stop_price, target_price)
            if exit_reason == "HOLD":
                exit_reason, exit_raw = resolve_same_bar_stop_target(next_bar, stop_price, target_price, "BUY")
                if exit_reason != "HOLD":
                    daily_stop_hit = pd.notna(as_float(next_bar.get("low"))) and pd.notna(stop_price) and as_float(next_bar.get("low")) <= stop_price
                    daily_target_hit = pd.notna(as_float(next_bar.get("high"))) and pd.notna(target_price) and as_float(next_bar.get("high")) >= target_price
                    first_touch_source = "AMBIGUOUS_DAILY_PATH" if daily_stop_hit and daily_target_hit else "DAILY_OHLC_FALLBACK"
            if exit_reason != "HOLD" and pd.notna(exit_raw):
                exit_fill_price = exit_raw * (1.0 - total_cost_bps / 10000.0)
                fills.append(
                    PaperFill(
                        fill_id=stable_id(order_id, exit_reason, expected_fill_date, prefix="pfl"),
                        order_id=order_id,
                        intent_id=intent_id,
                        symbol=symbol,
                        side="SELL",
                        fill_type=exit_reason,
                        status=FillStatus.FILLED.value,
                        fill_date=expected_fill_date,
                        fill_price=float(exit_fill_price),
                        raw_price=float(exit_raw),
                        quantity=float(quantity),
                        gross_notional=float(quantity * exit_fill_price),
                        commission_bps=float(commission_bps),
                        slippage_bps=float(slippage_bps),
                        total_cost_bps=float(total_cost_bps),
                        reason=first_touch_source,
                        created_at_utc=now_utc_iso(),
                    ).to_dict()
                )

    order = PaperOrder(
        order_id=order_id,
        intent_id=intent_id,
        symbol=symbol,
        side=str(intent.get("side", "BUY")),
        order_type=str(intent.get("order_type", "MARKET")),
        status=status,
        target_weight=float(approved_weight),
        target_notional=float(target_notional),
        quantity=float(estimated_quantity),
        reference_price=float(reference_price) if pd.notna(reference_price) else np.nan,
        limit_price=as_float(intent.get("limit_price")) if pd.notna(as_float(intent.get("limit_price"))) else None,
        stop_price=as_float(intent.get("stop_price")) if pd.notna(as_float(intent.get("stop_price"))) else None,
        time_in_force=str(intent.get("time_in_force", "DAY")),
        submitted_at_utc=now_utc_iso(),
        signal_asof_date=str(intent.get("asof_date", "")),
        expected_fill_date=expected_fill_date,
        fill_model=str(execution_config.fill_mode),
        live_trading_status="DISABLED_BY_DESIGN",
    ).to_dict()
    slippage = {
        "order_id": order_id,
        "intent_id": intent_id,
        "symbol": symbol,
        "fill_model": execution_config.fill_mode,
        "status": status,
        "expected_fill_date": expected_fill_date,
        "slippage_bps": fills[0]["slippage_bps"] if fills else np.nan,
        "intraday_execution_status": intraday_status,
        "intraday_path_source": intraday_path_source,
        "intraday_slippage_bps": intraday_slippage_bps,
        "commission_bps": commission_bps,
        "total_cost_bps": fills[0]["total_cost_bps"] if fills else np.nan,
        "reason": reason,
        "live_trading_status": "DISABLED_BY_DESIGN",
    }
    return order, fills, slippage


def latest_market_context(signals: pd.DataFrame) -> tuple[str, float]:
    if signals.empty:
        return "", np.nan
    last = signals.sort_values("date").iloc[-1] if "date" in signals.columns else signals.iloc[-1]
    market_date = pd.Timestamp(last["date"]).date().isoformat() if "date" in last.index and pd.notna(last.get("date")) else ""
    return market_date, as_float(last.get("close"))


def build_positions(
    fills: pd.DataFrame,
    signals: pd.DataFrame,
    account_equity: float,
    signals_by_symbol: dict[str, pd.DataFrame] | None = None,
    universe_symbols: list[str] | None = None,
) -> pd.DataFrame:
    default_market_date, default_market_price = latest_market_context(signals)
    if fills.empty:
        symbol = (universe_symbols or [])[0] if universe_symbols else ""
        if not symbol and "symbol" in signals.columns and not signals.empty:
            symbol = str(signals["symbol"].dropna().astype(str).iloc[-1])
        row = PaperPosition(symbol or "CASH", default_market_date, 0.0, np.nan, default_market_price, 0.0, float(account_equity), float(account_equity), 0.0, 0.0, 0.0, "CASH", now_utc_iso()).to_dict()
        return pd.DataFrame([row], columns=POSITION_COLUMNS)

    position_state: dict[str, dict[str, float]] = {}
    for symbol, group in fills.groupby("symbol", dropna=False):
        qty = 0.0
        cost_basis = 0.0
        realized = 0.0
        for _, fill in group.sort_values("fill_date").iterrows():
            fill_qty = as_float(fill.get("quantity"), 0.0)
            fill_price = as_float(fill.get("fill_price"), 0.0)
            side = str(fill.get("side", "")).upper()
            if side == "BUY":
                qty += fill_qty
                cost_basis += fill_qty * fill_price
            elif side == "SELL":
                avg = cost_basis / qty if qty > 0 else 0.0
                realized += fill_qty * (fill_price - avg)
                qty -= fill_qty
                cost_basis -= min(cost_basis, fill_qty * avg)
        position_state[str(symbol)] = {"quantity": max(float(qty), 0.0), "cost_basis": float(cost_basis), "realized_pnl": float(realized)}

    cash = float(account_equity)
    for _, fill in fills.sort_values("fill_date").iterrows():
        qty = as_float(fill.get("quantity"), 0.0)
        price = as_float(fill.get("fill_price"), 0.0)
        if str(fill.get("side", "")).upper() == "BUY":
            cash -= qty * price
        elif str(fill.get("side", "")).upper() == "SELL":
            cash += qty * price

    rows = []
    total_market_value = 0.0
    market_by_symbol: dict[str, tuple[str, float]] = {}
    for symbol in position_state:
        symbol_signals = (signals_by_symbol or {}).get(str(symbol).upper(), signals)
        market_date, market_price = latest_market_context(symbol_signals)
        market_by_symbol[str(symbol)] = (market_date or default_market_date, market_price)
        total_market_value += position_state[symbol]["quantity"] * market_price if pd.notna(market_price) else 0.0
    equity = cash + total_market_value
    for symbol, state in position_state.items():
        qty = state["quantity"]
        cost_basis = state["cost_basis"]
        realized = state["realized_pnl"]
        market_date, market_price = market_by_symbol.get(symbol, (default_market_date, default_market_price))
        avg_price = cost_basis / qty if qty > 1e-12 else np.nan
        mkt_value = qty * market_price if pd.notna(market_price) else 0.0
        unrealized = qty * (market_price - avg_price) if qty > 1e-12 and pd.notna(avg_price) and pd.notna(market_price) else 0.0
        rows.append(
            PaperPosition(
                symbol=str(symbol),
                asof_date=market_date,
                quantity=float(max(qty, 0.0)),
                average_price=float(avg_price) if pd.notna(avg_price) else np.nan,
                market_price=float(market_price) if pd.notna(market_price) else np.nan,
                market_value=float(mkt_value),
                cash=float(cash),
                equity=float(equity),
                weight=float(mkt_value / equity) if equity else 0.0,
                realized_pnl=float(realized),
                unrealized_pnl=float(unrealized),
                position_state="LONG" if qty > 1e-12 else "CASH",
                updated_at_utc=now_utc_iso(),
            ).to_dict()
        )
    return pd.DataFrame(rows, columns=POSITION_COLUMNS)


def build_quality(orders: pd.DataFrame, fills: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    fill_after_signal = True
    if not fills.empty and not orders.empty:
        joined = fills.merge(orders[["order_id", "signal_asof_date"]], on="order_id", how="left")
        fill_after_signal = bool((pd.to_datetime(joined["fill_date"], errors="coerce") > pd.to_datetime(joined["signal_asof_date"], errors="coerce")).fillna(True).all())
    rows = [
        check_row("paper_no_live_order_submission", True, "CRITICAL", "paper_only"),
        check_row("paper_orders_contract", set(ORDER_COLUMNS).issubset(orders.columns), "CRITICAL", len(orders.columns)),
        check_row("paper_fills_contract", set(FILL_COLUMNS).issubset(fills.columns), "CRITICAL", len(fills.columns)),
        check_row("paper_positions_contract", set(POSITION_COLUMNS).issubset(positions.columns), "CRITICAL", len(positions.columns)),
        check_row("paper_order_ids_unique", not orders.get("order_id", pd.Series(dtype=str)).duplicated().any(), "CRITICAL", int(orders.get("order_id", pd.Series(dtype=str)).duplicated().sum())),
        check_row("paper_fill_ids_unique", not fills.get("fill_id", pd.Series(dtype=str)).duplicated().any(), "CRITICAL", int(fills.get("fill_id", pd.Series(dtype=str)).duplicated().sum())),
        check_row("paper_next_open_fill_not_same_day_close", fill_after_signal, "CRITICAL", "fill_date > signal_asof_date"),
    ]
    return pd.DataFrame(rows)


def write_report(outdir: Path, orders: pd.DataFrame, fills: pd.DataFrame, positions: pd.DataFrame, quality: pd.DataFrame) -> None:
    latest_order = orders.tail(1).to_dict(orient="records")[0] if not orders.empty else {}
    latest_position = positions.tail(1).to_dict(orient="records")[0] if not positions.empty else {}
    lines = [
        "# Top10 Paper OMS Report",
        "",
        f"- Latest order status: {latest_order.get('status', 'NA')}",
        f"- Latest order ID: {latest_order.get('order_id', 'NA')}",
        f"- Fill rows: {len(fills)}",
        f"- Position state: {latest_position.get('position_state', 'NA')}",
        f"- Position weight: {normalize_weight(latest_position.get('weight', 0.0)) * 100:.2f}%",
        f"- Live trading status: {latest_order.get('live_trading_status', 'DISABLED_BY_DESIGN')}",
        "",
        "## Quality Checks",
        "",
        "| Check | Passed | Value |",
        "|---|---:|---:|",
    ]
    for _, row in quality.iterrows():
        lines.append(f"| {row['check']} | {row['passed']} | {row['value']} |")
    lines.extend(["", "This report is a broker-free paper execution ledger."])
    (outdir / "tsm_paper_oms_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_signal_map(universe_config: str, root_signals_path: Path) -> dict[str, pd.DataFrame]:
    signals_by_symbol: dict[str, pd.DataFrame] = {}
    if universe_config:
        for member in load_universe_members(universe_config, paper_only=True):
            path = member.paths.signals if member.paths.signals.exists() else (root_signals_path if member.symbol == "TSM" and root_signals_path.exists() else member.paths.signals)
            if path.exists():
                try:
                    signals_by_symbol[member.symbol.upper()] = load_signals(path)
                except Exception:
                    continue
    if not signals_by_symbol and root_signals_path.exists():
        root = load_signals(root_signals_path)
        symbol = str(root.get("symbol", pd.Series(["TSM"])).dropna().astype(str).iloc[-1]) if "symbol" in root.columns and not root.empty else "TSM"
        signals_by_symbol[symbol.upper()] = root
    return signals_by_symbol


def load_intraday_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = strip_bom_columns(pd.read_csv(path, parse_dates=["date"]))
    except Exception:
        return pd.DataFrame()
    if frame.empty or not {"symbol", "date"}.issubset(frame.columns):
        return pd.DataFrame()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame.dropna(subset=["date"]).sort_values(["symbol", "date"])


def intraday_context_for(features: pd.DataFrame, symbol: str, asof_date: object) -> pd.Series | None:
    if features.empty or not symbol:
        return None
    asof = pd.Timestamp(asof_date).normalize()
    subset = features[(features["symbol"].astype(str).str.upper() == symbol.upper()) & (features["date"] <= asof)]
    if subset.empty:
        return None
    return subset.iloc[-1]


def load_execution_minute_map(universe_config: str) -> dict[str, pd.DataFrame]:
    minute_by_symbol: dict[str, pd.DataFrame] = {}
    if not universe_config:
        return minute_by_symbol
    for member in load_universe_members(universe_config, paper_only=True):
        path = member.data_outdir / "tsm_minute_available_enriched.csv"
        if not path.exists():
            continue
        try:
            frame = strip_bom_columns(pd.read_csv(path, parse_dates=["date"]))
        except Exception:
            continue
        if not frame.empty and {"date", "open", "high", "low", "close"}.issubset(frame.columns):
            minute_by_symbol[member.symbol.upper()] = frame.sort_values("date").reset_index(drop=True)
    return minute_by_symbol


def filter_symbols(frame: pd.DataFrame, symbols: set[str]) -> pd.DataFrame:
    if frame.empty or not symbols or "symbol" not in frame.columns:
        return frame.copy()
    return frame[frame["symbol"].astype(str).str.upper().isin(symbols)].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate paper-only order execution.")
    parser.add_argument("--intents", default="tsm_price_rule_output/tsm_order_intents.csv")
    parser.add_argument("--portfolio-decisions", default="tsm_price_rule_output/tsm_portfolio_risk_order_decisions.csv")
    parser.add_argument("--signals", default="tsm_price_rule_output/tsm_daily_algorithmic_signals.csv")
    parser.add_argument("--signals-root", default="tsm_price_rule_output")
    parser.add_argument("--universe-config", default="")
    parser.add_argument("--intraday-features", default="tsm_price_rule_output/tsm_intraday_daily_features.csv")
    parser.add_argument("--orders", default="tsm_price_rule_output/tsm_paper_orders.csv")
    parser.add_argument("--fills", default="tsm_price_rule_output/tsm_paper_fills.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--config", default="config/tsm_research.toml")
    parser.add_argument("--commission-bps", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_config = load_run_config(args.config)
    oms_config = run_config.paper_oms
    execution_config = run_config.paper_execution
    signals_root = Path(args.signals_root)
    root_signals_path = signals_root / "tsm_daily_algorithmic_signals.csv" if signals_root.is_dir() or signals_root.suffix == "" else Path(args.signals)
    signals_by_symbol = build_signal_map(str(args.universe_config), root_signals_path)
    intraday_features = load_intraday_features(Path(args.intraday_features))
    minute_bars_by_symbol = load_execution_minute_map(str(args.universe_config))
    decision_symbols = set(signals_by_symbol)
    signals = next(iter(signals_by_symbol.values())) if signals_by_symbol else load_signals(Path(args.signals))
    intents = read_csv_if_exists(Path(args.intents))
    risk_decisions = read_csv_if_exists(Path(args.portfolio_decisions))
    existing_orders = read_csv_if_exists(Path(args.orders), ORDER_COLUMNS)
    existing_fills = read_csv_if_exists(Path(args.fills), FILL_COLUMNS)
    intents = filter_symbols(intents, decision_symbols)
    risk_decisions = filter_symbols(risk_decisions, decision_symbols)
    existing_orders = filter_symbols(existing_orders, decision_symbols)
    existing_fills = filter_symbols(existing_fills, decision_symbols)
    order_rows: list[dict[str, object]] = []
    fill_rows: list[dict[str, object]] = []
    slippage_rows: list[dict[str, object]] = []
    approved_risk = risk_decisions[risk_decisions.get("portfolio_status", pd.Series(dtype=str)).astype(str).eq(IntentStatus.APPROVED.value)].copy() if not risk_decisions.empty else pd.DataFrame()
    if not intents.empty and not approved_risk.empty:
        for _, latest_risk in approved_risk.iterrows():
            intent_id = str(latest_risk.get("intent_id", ""))
            matched = intents[intents.get("intent_id", pd.Series(dtype=str)).astype(str).eq(intent_id)]
            if matched.empty:
                continue
            latest_intent = matched.iloc[-1]
            symbol = clean_symbol(latest_intent.get("symbol"), clean_symbol(latest_risk.get("symbol")))
            symbol_signals = signals_by_symbol.get(symbol, signals)
            intraday_context = intraday_context_for(intraday_features, symbol, latest_intent.get("asof_date"))
            order, fills, slippage = build_order_and_fills(
                latest_intent,
                latest_risk,
                symbol_signals,
                oms_config,
                execution_config,
                args.commission_bps,
                intraday_context=intraday_context,
                minute_bars=minute_bars_by_symbol.get(symbol),
            )
            order_rows.append(order)
            fill_rows.extend(fills)
            slippage_rows.append(slippage)
    elif not intents.empty and not risk_decisions.empty:
        latest_intent = intents.sort_values("asof_date").iloc[-1] if "asof_date" in intents.columns else intents.iloc[-1]
        latest_risk = risk_decisions.sort_values("asof_date").iloc[-1] if "asof_date" in risk_decisions.columns else risk_decisions.iloc[-1]
        symbol = clean_symbol(latest_intent.get("symbol"), clean_symbol(latest_risk.get("symbol")))
        symbol_signals = signals_by_symbol.get(symbol, signals)
        intraday_context = intraday_context_for(intraday_features, symbol, latest_intent.get("asof_date"))
        order, fills, slippage = build_order_and_fills(
            latest_intent,
            latest_risk,
            symbol_signals,
            oms_config,
            execution_config,
            args.commission_bps,
            intraday_context=intraday_context,
            minute_bars=minute_bars_by_symbol.get(symbol),
        )
        order_rows.append(order)
        fill_rows.extend(fills)
        slippage_rows.append(slippage)

    orders = upsert_rows(existing_orders, order_rows, "order_id", ORDER_COLUMNS) if order_rows else existing_orders
    fills = upsert_rows(existing_fills, fill_rows, "fill_id", FILL_COLUMNS) if fill_rows else existing_fills
    positions = build_positions(fills, signals, oms_config.account_equity, signals_by_symbol=signals_by_symbol, universe_symbols=list(signals_by_symbol))
    slippage = pd.DataFrame(slippage_rows)
    quality = build_quality(orders, fills, positions)

    orders.to_csv(outdir / "tsm_paper_orders.csv", index=False)
    fills.to_csv(outdir / "tsm_paper_fills.csv", index=False)
    positions.to_csv(outdir / "tsm_paper_positions.csv", index=False)
    slippage.to_csv(outdir / "tsm_paper_slippage_report.csv", index=False)
    quality.to_csv(outdir / "tsm_paper_oms_quality_checks.csv", index=False)
    write_report(outdir, orders, fills, positions, quality)
    print("completed: paper OMS outputs =", outdir.resolve())
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()

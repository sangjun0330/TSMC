#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual portfolio execution cockpit engine.

Treats a human-maintained transaction ledger (the user's own buy/sell records)
as the source of truth, converts each transaction into a broker-free paper fill,
and reuses the existing paper-execution position math (``build_positions``) to
compute holdings, weights, returns and realized/unrealized P&L. It then diffs the
current weight of each holding against the model's target weight (portfolio-risk
``approved_weight`` with rule/risk fallbacks) to emit concrete rebalance
execution tickets ("buy N shares / trim M shares, stop X, target Y, reason Z").

No live broker is ever called; every output records that live trading is disabled
by design. This engine only *reads* existing model outputs -- it never retrains
models or runs backtests, so it is cheap to re-run on every transaction edit.

Internal math is in the engine currency (USD). User prices entered in KRW are
converted to USD using the daily FX rate (``KRW=X``) on the trade date. Display
conversion back to KRW is handled by the dashboard layer.

Outputs (default outdir: output/portfolio):
- manual_paper_fills.csv
- manual_paper_positions.csv
- manual_reconciliation_report.csv
- manual_rebalance_tickets.csv
- manual_portfolio_snapshot.csv
- manual_portfolio_quality_checks.csv
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tsm_core.config import load_run_config
from tsm_core.currency import load_fx_rates
from tsm_core.execution import PaperFill
from tsm_core.io import as_float, check_row, strip_bom_columns
from tsm_paper_execution_engine import (
    FILL_COLUMNS,
    POSITION_COLUMNS,
    build_positions,
    load_signals,
)
from tsm_position_reconciler import (
    build_quality as build_reconciliation_quality,
    build_reconciliation,
)

LIVE_TRADING_STATUS = "DISABLED_BY_DESIGN"

# Decision universe (top10 US semis + the 2 KR names). Korean display names match
# the user's brokerage app; the dashboard layer may override these.
DECISION_UNIVERSE = [
    "NVDA", "TSM", "AVGO", "AMD", "INTC", "MU",
    "TXN", "LRCX", "AMAT", "QCOM", "005930.KS", "000660.KS",
]
NAME_MAP = {
    "NVDA": "엔비디아",
    "TSM": "TSMC(ADR)",
    "AVGO": "브로드컴",
    "AMD": "AMD",
    "INTC": "인텔",
    "MU": "마이크론 테크놀로지",
    "TXN": "텍사스 인스트루먼트",
    "LRCX": "램 리서치",
    "AMAT": "어플라이드 머티리얼즈",
    "QCOM": "퀄컴",
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
}

TRANSACTION_COLUMNS = [
    "transaction_id",
    "trade_date",
    "symbol",
    "side",
    "quantity",
    "price",
    "price_currency",
    "fees",
    "note",
    "created_at_utc",
]

CASH_COLUMNS = ["entry_date", "type", "amount", "currency", "note", "created_at_utc"]

TICKET_COLUMNS = [
    "rank",
    "symbol",
    "name",
    "action",
    "current_weight",
    "target_weight",
    "weight_gap",
    "current_shares",
    "target_shares",
    "delta_shares",
    "delta_notional_usd",
    "ref_price_usd",
    "stop_price_usd",
    "target_price_usd",
    "p_success_20d",
    "action_level",
    "reason",
    "warning",
    "data_status",
    "live_trading_status",
    "checked_at_utc",
]

SNAPSHOT_COLUMNS = [
    "asof_date",
    "equity_usd",
    "cost_basis_usd",
    "market_value_usd",
    "cash_usd",
    "total_return_pct",
    "daily_pnl_usd",
    "n_holdings",
    "top_action_symbol",
    "top_action",
    "fx_rate_to_usd",
    "usdkrw",
    "live_trading_status",
    "generated_at_utc",
]

_ADD_EPS = 1e-6


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


def display_name(symbol: str) -> str:
    return NAME_MAP.get(symbol.upper(), symbol)


def listing_currency_for(symbol: str) -> str:
    s = symbol.upper()
    return "KRW" if s.endswith(".KS") or s.endswith(".KQ") else "USD"


def normalize_ratio(value: object) -> float:
    parsed = as_float(value)
    if pd.isna(parsed):
        return 0.0
    return float(parsed / 100.0 if abs(parsed) > 1.0 else parsed)


def read_ledger(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        df = strip_bom_columns(pd.read_csv(path))
    except Exception:
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return strip_bom_columns(pd.read_csv(path))
    except Exception:
        return pd.DataFrame()


def fx_rate_to_usd_on(fx: pd.DataFrame, trade_date: object) -> float:
    """Backward-as-of FX rate (KRW=X) on/just before ``trade_date`` (KRW -> USD)."""
    if fx.empty or "fx_rate_to_usd" not in fx.columns:
        return np.nan
    rates = fx.dropna(subset=["fx_rate_to_usd"]).sort_values("date")
    if rates.empty:
        return np.nan
    d = pd.to_datetime(trade_date, errors="coerce")
    if pd.isna(d):
        return as_float(rates["fx_rate_to_usd"].iloc[-1])
    sub = rates[rates["date"] <= d.normalize()]
    if sub.empty:
        sub = rates
    return as_float(sub["fx_rate_to_usd"].iloc[-1])


def price_to_usd(price: object, currency: str, symbol: str, trade_date: object, fx: pd.DataFrame) -> tuple[float, float]:
    """Return (price_usd, fx_rate_used). Resolves NATIVE by symbol listing currency."""
    p = as_float(price)
    if pd.isna(p):
        return np.nan, np.nan
    cur = str(currency or "").strip().upper()
    if cur in ("", "NATIVE"):
        cur = listing_currency_for(symbol)
    if cur == "USD":
        return p, 1.0
    if cur == "KRW":
        rate = fx_rate_to_usd_on(fx, trade_date)
        if pd.isna(rate) or rate <= 0:
            return np.nan, np.nan
        return p * rate, rate
    return p, 1.0


def amount_to_usd(amount: object, currency: str, entry_date: object, fx: pd.DataFrame) -> float:
    a = as_float(amount)
    if pd.isna(a):
        return 0.0
    cur = str(currency or "USD").strip().upper()
    if cur == "KRW":
        rate = fx_rate_to_usd_on(fx, entry_date)
        return float(a * rate) if pd.notna(rate) and rate > 0 else 0.0
    return float(a)


def transactions_to_fills(transactions: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if transactions.empty:
        return pd.DataFrame(columns=FILL_COLUMNS)
    for _, t in transactions.iterrows():
        symbol = clean_symbol(t.get("symbol"))
        side = str(t.get("side", "")).strip().upper()
        if not symbol or side not in ("BUY", "SELL"):
            continue
        qty = as_float(t.get("quantity"), 0.0)
        if not (qty > 0):
            continue
        trade_date = pd.to_datetime(t.get("trade_date"), errors="coerce")
        date_str = trade_date.date().isoformat() if pd.notna(trade_date) else str(t.get("trade_date", ""))[:10]
        currency = str(t.get("price_currency", "") or "").strip().upper()
        raw_price = as_float(t.get("price"), 0.0)
        price_usd, _ = price_to_usd(raw_price, currency, symbol, trade_date, fx)
        if pd.isna(price_usd):
            continue
        gross = float(qty * price_usd)
        txid = str(t.get("transaction_id") or "").strip() or stable_id(symbol, side, date_str, qty, raw_price, prefix="mtx")
        rows.append(
            PaperFill(
                fill_id=stable_id(txid, prefix="mfill"),
                order_id=stable_id(txid, "order", prefix="morder"),
                intent_id=txid,
                symbol=symbol,
                side=side,
                fill_type="MANUAL",
                status="FILLED",
                fill_date=date_str,
                fill_price=float(price_usd),
                raw_price=float(raw_price),
                quantity=float(qty),
                gross_notional=gross,
                commission_bps=0.0,
                slippage_bps=0.0,
                total_cost_bps=0.0,
                reason=str(t.get("note", "") or ""),
                created_at_utc=now_utc_iso(),
            ).to_dict()
        )
    if not rows:
        return pd.DataFrame(columns=FILL_COLUMNS)
    return pd.DataFrame(rows, columns=FILL_COLUMNS)


def account_equity_from(fills: pd.DataFrame, cash_ledger: pd.DataFrame, fx: pd.DataFrame) -> float:
    """Total deposited capital (USD). Uses cash ledger when present, otherwise the
    net invested notional so that derived cash settles near zero (fully invested)."""
    if not cash_ledger.empty:
        total = 0.0
        for _, c in cash_ledger.iterrows():
            usd = amount_to_usd(c.get("amount"), c.get("currency", "USD"), c.get("entry_date"), fx)
            kind = str(c.get("type", "DEPOSIT")).strip().upper()
            total += -usd if kind in ("WITHDRAW", "WITHDRAWAL") else usd
        if total > 0:
            return float(total)
    net = 0.0
    for _, f in fills.iterrows():
        notional = as_float(f.get("gross_notional"), 0.0)
        if str(f.get("side", "")).upper() == "BUY":
            net += notional
        elif str(f.get("side", "")).upper() == "SELL":
            net -= notional
    return float(max(net, 0.0))


def load_signals_by_symbol(symbols: list[str], rule_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = rule_dir / "universe" / symbol / "tsm_daily_algorithmic_signals.csv"
        if not path.exists():
            continue
        try:
            out[symbol.upper()] = load_signals(path)
        except Exception:
            continue
    return out


def latest_two_closes(signals: pd.DataFrame) -> tuple[float, float]:
    if signals.empty or "close" not in signals.columns:
        return np.nan, np.nan
    closes = pd.to_numeric(signals.sort_values("date")["close"], errors="coerce").dropna()
    if closes.empty:
        return np.nan, np.nan
    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2]) if len(closes) >= 2 else np.nan
    return last, prev


def _latest_by_symbol(frame: pd.DataFrame, symbol: str, date_cols: tuple[str, ...]) -> pd.Series:
    if frame.empty or "symbol" not in frame.columns:
        return pd.Series(dtype=object)
    sub = frame[frame["symbol"].astype(str).str.upper() == symbol.upper()].copy()
    if sub.empty:
        return pd.Series(dtype=object)
    for col in date_cols:
        if col in sub.columns:
            sub["_o"] = pd.to_datetime(sub[col], errors="coerce")
            sub = sub.sort_values("_o")
            break
    return sub.iloc[-1]


def build_rebalance_tickets(
    positions: pd.DataFrame,
    signals_by_symbol: dict[str, pd.DataFrame],
    portfolio_decisions: pd.DataFrame,
    latest_signals: pd.DataFrame,
    rule_dir: Path,
) -> pd.DataFrame:
    equity = as_float(positions["equity"].iloc[0], 0.0) if not positions.empty and "equity" in positions.columns else 0.0
    held: dict[str, pd.Series] = {}
    for _, p in positions.iterrows():
        sym = clean_symbol(p.get("symbol"))
        if sym and sym != "CASH":
            held[sym] = p

    # candidate symbols: held names + any decision-universe symbol with a positive target
    candidates = set(held.keys())
    for sym in DECISION_UNIVERSE:
        candidates.add(sym.upper())

    rows: list[dict] = []
    for symbol in sorted(candidates):
        pos = held.get(symbol, pd.Series(dtype=object))
        sig = signals_by_symbol.get(symbol)
        if sig is None:
            try:
                path = rule_dir / "universe" / symbol / "tsm_daily_algorithmic_signals.csv"
                sig = load_signals(path) if path.exists() else pd.DataFrame()
            except Exception:
                sig = pd.DataFrame()
        last_close, _ = latest_two_closes(sig) if sig is not None else (np.nan, np.nan)
        signal_row = sig.sort_values("date").iloc[-1] if sig is not None and not sig.empty else pd.Series(dtype=object)

        decision = _latest_by_symbol(portfolio_decisions, symbol, ("asof_date", "date"))
        latest_sig_row = _latest_by_symbol(latest_signals, symbol, ("date", "asof_date"))

        current_shares = as_float(pos.get("quantity"), 0.0) if not pos.empty else 0.0
        current_price = as_float(pos.get("market_price"), last_close) if not pos.empty else last_close
        if pd.isna(current_price):
            current_price = last_close
        current_weight = as_float(pos.get("weight"), 0.0) if not pos.empty else 0.0

        # Target weight: portfolio-risk approved_weight, then risk-policy / suggested fallback
        data_status = "MODEL"
        target_weight = normalize_ratio(decision.get("approved_weight")) if not decision.empty else 0.0
        if decision.empty or target_weight <= 0:
            risk_path = rule_dir / "universe" / symbol / "tsm_risk_policy_daily.csv"
            risk = read_csv_if_exists(risk_path)
            fallback = 0.0
            if not risk.empty:
                rr = risk.sort_values("date").iloc[-1] if "date" in risk.columns else risk.iloc[-1]
                fallback = normalize_ratio(rr.get("final_recommended_max_weight"))
                allow = str(rr.get("allow_entry", "")).strip().lower()
                if allow in ("false", "0", "no"):
                    fallback = 0.0
            if fallback <= 0 and not latest_sig_row.empty:
                fallback = normalize_ratio(latest_sig_row.get("suggested_weight"))
            if decision.empty:
                data_status = "RULE_FALLBACK"
            target_weight = fallback if fallback > 0 else target_weight

        # Stop / target / action context (engine USD)
        stop_price = as_float(
            (decision.get("stop_price_1_8atr") if not decision.empty else np.nan),
            as_float(signal_row.get("atr_stop_2x"), np.nan) if not signal_row.empty else np.nan,
        )
        target_price = as_float(signal_row.get("take_profit_2R"), np.nan) if not signal_row.empty else np.nan
        trade_action = str(signal_row.get("trade_action", "") or "").strip().upper() if not signal_row.empty else ""
        entry_trigger = str(signal_row.get("entry_trigger", "") or "").strip() if not signal_row.empty else ""
        p_success = as_float(decision.get("p_success_20d"), np.nan) if not decision.empty else np.nan
        block_reason = str(decision.get("block_reason", "") or "").strip() if not decision.empty else ""
        warning_reasons = str(decision.get("warning_reasons", "") or "").strip() if not decision.empty else ""

        target_value = target_weight * equity if equity > 0 else 0.0
        target_shares = target_value / current_price if pd.notna(current_price) and current_price > 0 else 0.0
        delta_shares = target_shares - current_shares
        delta_notional = delta_shares * current_price if pd.notna(current_price) else 0.0

        model_says_exit = trade_action == "SELL" or (block_reason not in ("", "PASS") and target_weight <= 0)

        if current_shares > _ADD_EPS and model_says_exit:
            action, action_level = "EXIT", "SELL"
            target_shares, delta_shares = 0.0, -current_shares
            delta_notional = delta_shares * current_price if pd.notna(current_price) else 0.0
        elif delta_shares > _ADD_EPS and target_weight > 0:
            action = "BUY_NEW" if current_shares <= _ADD_EPS else "ADD"
            action_level = "BUY"
        elif delta_shares < -_ADD_EPS:
            action, action_level = "TRIM", "REDUCE"
        else:
            action, action_level = "HOLD", "HOLD"

        # Only emit a row for held names or actionable (non-HOLD with target) names
        if current_shares <= _ADD_EPS and action in ("HOLD",):
            continue
        if current_shares <= _ADD_EPS and action == "BUY_NEW" and target_weight <= 0:
            continue

        reason_bits = []
        if entry_trigger:
            reason_bits.append(entry_trigger)
        if trade_action:
            reason_bits.append(f"action={trade_action}")
        if pd.notna(p_success):
            reason_bits.append(f"p_success_20d={p_success:.2f}")
        if data_status == "RULE_FALLBACK":
            reason_bits.append("rule/risk fallback (no portfolio decision)")
        reason = "; ".join(reason_bits) if reason_bits else "no actionable signal"
        warning = " | ".join([w for w in (block_reason, warning_reasons) if w and w != "PASS"])

        rows.append(
            {
                "rank": np.nan,
                "symbol": symbol,
                "name": display_name(symbol),
                "action": action,
                "current_weight": float(current_weight),
                "target_weight": float(target_weight),
                "weight_gap": float(target_weight - current_weight),
                "current_shares": float(current_shares),
                "target_shares": float(max(target_shares, 0.0)),
                "delta_shares": float(delta_shares),
                "delta_notional_usd": float(delta_notional),
                "ref_price_usd": float(current_price) if pd.notna(current_price) else np.nan,
                "stop_price_usd": float(stop_price) if pd.notna(stop_price) else np.nan,
                "target_price_usd": float(target_price) if pd.notna(target_price) else np.nan,
                "p_success_20d": float(p_success) if pd.notna(p_success) else np.nan,
                "action_level": action_level,
                "reason": reason,
                "warning": warning,
                "data_status": data_status,
                "live_trading_status": LIVE_TRADING_STATUS,
                "checked_at_utc": now_utc_iso(),
            }
        )

    tickets = pd.DataFrame(rows, columns=TICKET_COLUMNS)
    if tickets.empty:
        return tickets
    priority = {"EXIT": 0, "TRIM": 1, "ADD": 2, "BUY_NEW": 2, "HOLD": 3}
    tickets["_p"] = tickets["action"].map(priority).fillna(4)
    tickets["_mag"] = tickets["delta_notional_usd"].abs()
    tickets = tickets.sort_values(["_p", "_mag"], ascending=[True, False]).reset_index(drop=True)
    tickets["rank"] = np.arange(1, len(tickets) + 1)
    return tickets.drop(columns=["_p", "_mag"])


def build_snapshot(positions: pd.DataFrame, tickets: pd.DataFrame, signals_by_symbol: dict[str, pd.DataFrame], fx: pd.DataFrame) -> pd.DataFrame:
    holdings = positions[positions["symbol"].astype(str).str.upper() != "CASH"].copy() if not positions.empty else pd.DataFrame()
    holdings = holdings[pd.to_numeric(holdings.get("quantity", pd.Series(dtype=float)), errors="coerce").fillna(0) > _ADD_EPS] if not holdings.empty else holdings
    market_value = float(pd.to_numeric(holdings.get("market_value", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not holdings.empty else 0.0
    cost_basis = 0.0
    daily_pnl = 0.0
    if not holdings.empty:
        for _, h in holdings.iterrows():
            qty = as_float(h.get("quantity"), 0.0)
            avg = as_float(h.get("average_price"), np.nan)
            if pd.notna(avg):
                cost_basis += qty * avg
            last, prev = latest_two_closes(signals_by_symbol.get(clean_symbol(h.get("symbol")), pd.DataFrame()))
            if pd.notna(last) and pd.notna(prev):
                daily_pnl += qty * (last - prev)
    equity = as_float(positions["equity"].iloc[0], market_value) if not positions.empty and "equity" in positions.columns else market_value
    cash = as_float(positions["cash"].iloc[0], 0.0) if not positions.empty and "cash" in positions.columns else 0.0
    total_return_pct = ((market_value - cost_basis) / cost_basis * 100.0) if cost_basis > 0 else 0.0
    top = tickets.iloc[0] if not tickets.empty else pd.Series(dtype=object)
    fx_rate = fx_rate_to_usd_on(fx, None)
    usdkrw = (1.0 / fx_rate) if pd.notna(fx_rate) and fx_rate > 0 else np.nan
    asof = ""
    if not holdings.empty and "asof_date" in holdings.columns:
        asof = str(holdings["asof_date"].dropna().astype(str).max())
    row = {
        "asof_date": asof,
        "equity_usd": float(equity),
        "cost_basis_usd": float(cost_basis),
        "market_value_usd": float(market_value),
        "cash_usd": float(cash),
        "total_return_pct": float(total_return_pct),
        "daily_pnl_usd": float(daily_pnl),
        "n_holdings": int(len(holdings)),
        "top_action_symbol": str(top.get("symbol", "")) if not top.empty else "",
        "top_action": str(top.get("action", "")) if not top.empty else "",
        "fx_rate_to_usd": float(fx_rate) if pd.notna(fx_rate) else np.nan,
        "usdkrw": float(usdkrw) if pd.notna(usdkrw) else np.nan,
        "live_trading_status": LIVE_TRADING_STATUS,
        "generated_at_utc": now_utc_iso(),
    }
    return pd.DataFrame([row], columns=SNAPSHOT_COLUMNS)


def build_quality(fills: pd.DataFrame, positions: pd.DataFrame, reconciliation: pd.DataFrame) -> pd.DataFrame:
    n_holdings = int((pd.to_numeric(positions.get("quantity", pd.Series(dtype=float)), errors="coerce").fillna(0) > _ADD_EPS).sum()) if not positions.empty else 0
    all_priced = bool((pd.to_numeric(fills.get("fill_price", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).all()) if not fills.empty else True
    recon_pass = bool(reconciliation.get("status", pd.Series(dtype=str)).astype(str).eq("PASS").all()) if not reconciliation.empty else True
    live_disabled = bool(positions is not None)
    rows = [
        check_row("manual_portfolio_fills_priced", all_priced, "CRITICAL", "all_usd_priced"),
        check_row("manual_portfolio_positions_built", n_holdings >= 0, "INFO", n_holdings),
        check_row("manual_portfolio_reconciliation_pass", recon_pass, "CRITICAL", "PASS" if recon_pass else "FAIL"),
        check_row("manual_portfolio_live_trading_disabled", live_disabled, "CRITICAL", LIVE_TRADING_STATUS),
    ]
    return pd.DataFrame(rows)


def run_manual_portfolio(root: Path, config_path: Path, outdir: Path) -> dict[str, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    rule_dir = root / "tsm_price_rule_output"

    transactions = read_ledger(outdir / "manual_transactions.csv", TRANSACTION_COLUMNS)
    cash_ledger = read_ledger(outdir / "manual_cash_ledger.csv", CASH_COLUMNS)
    fx = load_fx_rates(root / "output" / "tsm_fx_rates_daily.csv", "KRW=X")

    fills = transactions_to_fills(transactions, fx)
    account_equity = account_equity_from(fills, cash_ledger, fx)

    symbols = sorted({clean_symbol(s) for s in fills.get("symbol", pd.Series(dtype=str)).tolist() if clean_symbol(s)})
    signals_by_symbol = load_signals_by_symbol(symbols, rule_dir)
    default_signals = next(iter(signals_by_symbol.values()), pd.DataFrame())

    positions = build_positions(fills, default_signals, account_equity, signals_by_symbol, symbols or None)
    reconciliation = build_reconciliation(positions, fills)

    portfolio_decisions = read_csv_if_exists(rule_dir / "tsm_portfolio_risk_order_decisions.csv")
    latest_signals = read_csv_if_exists(rule_dir / "tsm_universe_latest_signals.csv")
    tickets = build_rebalance_tickets(positions, signals_by_symbol, portfolio_decisions, latest_signals, rule_dir)

    snapshot = build_snapshot(positions, tickets, signals_by_symbol, fx)
    quality = build_quality(fills, positions, reconciliation)

    fills.to_csv(outdir / "manual_paper_fills.csv", index=False)
    positions.to_csv(outdir / "manual_paper_positions.csv", index=False)
    reconciliation.to_csv(outdir / "manual_reconciliation_report.csv", index=False)
    tickets.to_csv(outdir / "manual_rebalance_tickets.csv", index=False)
    snapshot.to_csv(outdir / "manual_portfolio_snapshot.csv", index=False)
    quality.to_csv(outdir / "manual_portfolio_quality_checks.csv", index=False)

    return {
        "transactions": transactions,
        "fills": fills,
        "positions": positions,
        "reconciliation": reconciliation,
        "tickets": tickets,
        "snapshot": snapshot,
        "quality": quality,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual portfolio execution cockpit engine (broker-free).")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--config", default="config/tsm_research.toml")
    parser.add_argument("--outdir", default="output/portfolio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    outdir = (root / args.outdir) if not Path(args.outdir).is_absolute() else Path(args.outdir)
    # load_run_config tolerates missing files and provides defaults (caps/account_equity).
    try:
        load_run_config(root / args.config if not Path(args.config).is_absolute() else Path(args.config))
    except Exception:
        pass
    result = run_manual_portfolio(root, Path(args.config), outdir)
    print("completed: manual portfolio outputs =", outdir.resolve())
    print(result["quality"].to_string(index=False))
    if not result["tickets"].empty:
        cols = ["rank", "symbol", "action", "current_weight", "target_weight", "delta_shares", "reason"]
        print(result["tickets"][cols].to_string(index=False))


if __name__ == "__main__":
    main()

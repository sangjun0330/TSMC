#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Broker-free order lifecycle state machine.

The Paper OMS deliberately stops before broker connectivity. This module
normalizes the daily CSV ledgers into a single lifecycle state that downstream
monitoring, reports, and model feedback can consume without inferring status
from several files independently.

Outputs:
- tsm_order_state_events.csv
- tsm_order_lifecycle_snapshot.csv
- tsm_order_state_quality_checks.csv
- tsm_order_state_report.md
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tsm_core.execution import IntentStatus, OrderLifecycleState, OrderStateEvent, OrderStatus
from tsm_core.io import check_row, strip_bom_columns


EVENT_COLUMNS = [
    "state_event_id",
    "intent_id",
    "order_id",
    "symbol",
    "asof_date",
    "lifecycle_state",
    "previous_state",
    "intent_status",
    "portfolio_status",
    "order_status",
    "fill_status",
    "reconciliation_status",
    "block_reason",
    "live_trading_status",
    "event_created_at_utc",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_id(*parts: object, prefix: str) -> str:
    text = "|".join("" if part is None else str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return strip_bom_columns(pd.read_csv(path))


def latest_matching(frame: pd.DataFrame, key: str, value: object) -> pd.Series:
    if frame.empty or key not in frame.columns:
        return pd.Series(dtype=object)
    matched = frame[frame[key].astype(str).eq(str(value))]
    if matched.empty:
        return pd.Series(dtype=object)
    sort_cols = [col for col in ["asof_date", "expected_fill_date", "fill_date", "submitted_at_utc", "checked_at_utc"] if col in matched.columns]
    return matched.sort_values(sort_cols).iloc[-1] if sort_cols else matched.iloc[-1]


def latest_reconciliation_status(reconciliation: pd.DataFrame) -> tuple[str, str]:
    if reconciliation.empty:
        return "MISSING", "MISSING_RECONCILIATION"
    status_col = reconciliation.get("status", pd.Series("", index=reconciliation.index)).astype(str)
    if status_col.str.upper().eq("FAIL").any():
        failed = reconciliation[status_col.str.upper().eq("FAIL")].tail(1).iloc[0]
        return "FAIL", str(failed.get("details", "POSITION_MISMATCH"))
    latest = reconciliation.tail(1).iloc[0]
    return str(latest.get("status", "PASS")), str(latest.get("details", "PASS"))


def determine_lifecycle_state(
    intent_status: str,
    portfolio_status: str,
    order_status: str,
    fill_status: str,
    reconciliation_status: str,
) -> tuple[str, str]:
    intent_status = intent_status.upper()
    portfolio_status = portfolio_status.upper()
    order_status = order_status.upper()
    fill_status = fill_status.upper()
    reconciliation_status = reconciliation_status.upper()

    if intent_status in {"", "MISSING"}:
        return OrderLifecycleState.MISSING_INTENT.value, "MISSING_INTENT"
    if intent_status == IntentStatus.REJECTED.value:
        return OrderLifecycleState.REJECTED.value, "INTENT_REJECTED"
    if intent_status == IntentStatus.BLOCKED.value:
        return OrderLifecycleState.BLOCKED.value, "INTENT_BLOCKED"
    if portfolio_status in {IntentStatus.REJECTED.value, IntentStatus.BLOCKED.value}:
        return OrderLifecycleState.RISK_REJECTED.value, "PORTFOLIO_RISK_REJECTED"
    if order_status == OrderStatus.EXPIRED.value:
        return OrderLifecycleState.EXPIRED_UNFILLED.value, "EXPIRED_UNFILLED"
    if order_status == OrderStatus.CANCELED.value:
        return OrderLifecycleState.CANCELED.value, "ORDER_CANCELED"
    if order_status == OrderStatus.PARTIALLY_FILLED.value or fill_status == OrderStatus.PARTIALLY_FILLED.value:
        return OrderLifecycleState.PARTIALLY_FILLED.value, "PARTIAL_FILL"
    if order_status == OrderStatus.FILLED.value or fill_status == OrderStatus.FILLED.value:
        if reconciliation_status == "PASS":
            return OrderLifecycleState.RECONCILED.value, "PASS"
        return OrderLifecycleState.RECONCILIATION_REQUIRED.value, "RECONCILIATION_REQUIRED"
    if order_status == OrderStatus.PAPER_SUBMITTED.value:
        return OrderLifecycleState.AWAITING_FILL.value, "AWAITING_FILL"
    if portfolio_status == IntentStatus.APPROVED.value:
        return OrderLifecycleState.APPROVED.value, "PORTFOLIO_APPROVED"
    if intent_status == IntentStatus.APPROVED.value:
        return OrderLifecycleState.PROPOSED.value, "AWAITING_PORTFOLIO_RISK"
    return OrderLifecycleState.PROPOSED.value, "PENDING"


def build_state_events(
    intents: pd.DataFrame,
    portfolio_decisions: pd.DataFrame,
    orders: pd.DataFrame,
    fills: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> pd.DataFrame:
    if intents.empty:
        event = OrderStateEvent(
            state_event_id=stable_id("missing", now_utc_iso(), prefix="ose"),
            intent_id="",
            order_id="",
            symbol="",
            asof_date="",
            lifecycle_state=OrderLifecycleState.MISSING_INTENT.value,
            previous_state="",
            intent_status="MISSING",
            portfolio_status="MISSING",
            order_status="MISSING",
            fill_status="MISSING",
            reconciliation_status="MISSING",
            block_reason="MISSING_INTENT",
            live_trading_status="DISABLED_BY_DESIGN",
            event_created_at_utc=now_utc_iso(),
        )
        return pd.DataFrame([event.to_dict()], columns=EVENT_COLUMNS)

    rec_status, rec_reason = latest_reconciliation_status(reconciliation)
    rows: list[dict[str, object]] = []
    sort_cols = [col for col in ["asof_date", "created_at_utc"] if col in intents.columns]
    sorted_intents = intents.sort_values(sort_cols) if sort_cols else intents
    previous_state = ""
    for _, intent in sorted_intents.iterrows():
        intent_id = str(intent.get("intent_id", ""))
        decision = latest_matching(portfolio_decisions, "intent_id", intent_id)
        order = latest_matching(orders, "intent_id", intent_id)
        order_id = str(order.get("order_id", "")) if not order.empty else ""
        fill = latest_matching(fills, "order_id", order_id) if order_id else latest_matching(fills, "intent_id", intent_id)
        intent_status = str(intent.get("status", "MISSING"))
        portfolio_status = str(decision.get("portfolio_status", "MISSING")) if not decision.empty else "MISSING"
        order_status = str(order.get("status", "MISSING")) if not order.empty else "MISSING"
        fill_status = str(fill.get("status", "MISSING")) if not fill.empty else "MISSING"
        state, block_reason = determine_lifecycle_state(intent_status, portfolio_status, order_status, fill_status, rec_status)
        if block_reason == "PASS":
            block_reason = rec_reason if rec_reason != "PASS" else "PASS"
        event = OrderStateEvent(
            state_event_id=stable_id(intent_id, order_id, state, rec_status, prefix="ose"),
            intent_id=intent_id,
            order_id=order_id,
            symbol=str(intent.get("symbol", order.get("symbol", "") if not order.empty else "")),
            asof_date=str(intent.get("asof_date", "")),
            lifecycle_state=state,
            previous_state=previous_state,
            intent_status=intent_status,
            portfolio_status=portfolio_status,
            order_status=order_status,
            fill_status=fill_status,
            reconciliation_status=rec_status,
            block_reason=block_reason,
            live_trading_status=str(intent.get("live_trading_status", order.get("live_trading_status", "DISABLED_BY_DESIGN") if not order.empty else "DISABLED_BY_DESIGN")),
            event_created_at_utc=now_utc_iso(),
        )
        rows.append(event.to_dict())
        previous_state = state
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def build_snapshot(events: pd.DataFrame) -> pd.DataFrame:
    latest = events.tail(1).iloc[0] if not events.empty else pd.Series(dtype=object)
    fields = [
        ("lifecycle_state", latest.get("lifecycle_state", "MISSING_INTENT")),
        ("intent_id", latest.get("intent_id", "")),
        ("order_id", latest.get("order_id", "")),
        ("symbol", latest.get("symbol", "")),
        ("asof_date", latest.get("asof_date", "")),
        ("intent_status", latest.get("intent_status", "MISSING")),
        ("portfolio_status", latest.get("portfolio_status", "MISSING")),
        ("order_status", latest.get("order_status", "MISSING")),
        ("fill_status", latest.get("fill_status", "MISSING")),
        ("reconciliation_status", latest.get("reconciliation_status", "MISSING")),
        ("block_reason", latest.get("block_reason", "MISSING")),
        ("live_trading_status", latest.get("live_trading_status", "DISABLED_BY_DESIGN")),
        ("event_created_at_utc", latest.get("event_created_at_utc", now_utc_iso())),
    ]
    return pd.DataFrame([{"field": field, "value": value} for field, value in fields])


def build_quality(events: pd.DataFrame) -> pd.DataFrame:
    allowed_states = {state.value for state in OrderLifecycleState}
    duplicate_events = int(events.get("state_event_id", pd.Series(dtype=str)).duplicated().sum()) if not events.empty else 0
    duplicate_orders = int(events.get("order_id", pd.Series(dtype=str)).replace("", np.nan).dropna().duplicated().sum()) if not events.empty else 0
    live_disabled = bool(events.get("live_trading_status", pd.Series(["DISABLED_BY_DESIGN"])).astype(str).eq("DISABLED_BY_DESIGN").all()) if not events.empty else True
    legal_states = bool(events.get("lifecycle_state", pd.Series(dtype=str)).astype(str).isin(allowed_states).all()) if not events.empty else False
    return pd.DataFrame(
        [
            check_row("order_state_rows_positive", not events.empty, "CRITICAL", len(events)),
            check_row("order_state_ids_unique", duplicate_events == 0, "CRITICAL", duplicate_events),
            check_row("order_state_legal_lifecycle_state", legal_states, "CRITICAL", ",".join(sorted(allowed_states))),
            check_row("order_state_no_duplicate_order_ids", duplicate_orders == 0, "WARN", duplicate_orders),
            check_row("order_state_live_trading_disabled", live_disabled, "CRITICAL", "DISABLED_BY_DESIGN"),
        ]
    )


def write_report(outdir: Path, events: pd.DataFrame, quality: pd.DataFrame) -> None:
    latest = events.tail(1).to_dict(orient="records")[0] if not events.empty else {}
    lines = [
        "# Top10 Paper OMS Lifecycle Report",
        "",
        f"- Lifecycle state: {latest.get('lifecycle_state', 'MISSING_INTENT')}",
        f"- Intent ID: {latest.get('intent_id', 'NA')}",
        f"- Order ID: {latest.get('order_id', 'NA')}",
        f"- Block reason: {latest.get('block_reason', 'NA')}",
        f"- Live trading status: {latest.get('live_trading_status', 'DISABLED_BY_DESIGN')}",
        "",
        "## Quality Checks",
        "",
        "| Check | Passed | Value |",
        "|---|---:|---:|",
    ]
    for _, row in quality.iterrows():
        lines.append(f"| {row['check']} | {row['passed']} | {row['value']} |")
    lines.extend(["", "This lifecycle state is paper-only and never submits broker orders."])
    (outdir / "tsm_order_state_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build broker-free Paper OMS lifecycle state.")
    parser.add_argument("--intents", default="tsm_price_rule_output/tsm_order_intents.csv")
    parser.add_argument("--portfolio-decisions", default="tsm_price_rule_output/tsm_portfolio_risk_order_decisions.csv")
    parser.add_argument("--orders", default="tsm_price_rule_output/tsm_paper_orders.csv")
    parser.add_argument("--fills", default="tsm_price_rule_output/tsm_paper_fills.csv")
    parser.add_argument("--reconciliation", default="tsm_price_rule_output/tsm_paper_reconciliation_report.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    events = build_state_events(
        read_csv_if_exists(Path(args.intents)),
        read_csv_if_exists(Path(args.portfolio_decisions)),
        read_csv_if_exists(Path(args.orders)),
        read_csv_if_exists(Path(args.fills)),
        read_csv_if_exists(Path(args.reconciliation)),
    )
    snapshot = build_snapshot(events)
    quality = build_quality(events)
    events.to_csv(outdir / "tsm_order_state_events.csv", index=False)
    snapshot.to_csv(outdir / "tsm_order_lifecycle_snapshot.csv", index=False)
    quality.to_csv(outdir / "tsm_order_state_quality_checks.csv", index=False)
    write_report(outdir, events, quality)
    print("completed: order lifecycle outputs =", outdir.resolve())
    print(snapshot.to_string(index=False))


if __name__ == "__main__":
    main()

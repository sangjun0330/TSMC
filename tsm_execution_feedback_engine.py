#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build execution feedback features and labels from the broker-free Paper OMS.

The engine turns simulated orders/fills into model-feedback rows while keeping
immature outcomes out of model-promotion evidence. It records unfilled/expired
orders separately so execution quality can improve without pretending that a
missed limit order was a market outcome.

Outputs:
- tsm_execution_feedback_events.csv
- tsm_execution_feedback_features.csv
- tsm_execution_feedback_labels.csv
- tsm_execution_feedback_quality_checks.csv
- tsm_execution_feedback_report.md
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tsm_core.config import PaperFeedbackConfig, load_run_config
from tsm_core.execution import ExecutionFeedbackEvent, FillStatus, OrderStatus
from tsm_core.io import as_float, check_row, strip_bom_columns, to_bool


EVENT_COLUMNS = [
    "feedback_id",
    "intent_id",
    "order_id",
    "fill_id",
    "symbol",
    "signal_asof_date",
    "fill_date",
    "horizon_days",
    "order_status",
    "fill_status",
    "fill_model",
    "filled_flag",
    "partial_fill_flag",
    "expired_unfilled_flag",
    "raw_price",
    "fill_price",
    "expected_slippage_bps",
    "realized_slippage_bps",
    "slippage_error_bps",
    "post_fill_return_pct",
    "mfe_pct",
    "mae_pct",
    "stop_hit",
    "target_hit",
    "label_matured",
    "execution_adjusted_utility",
    "live_trading_status",
    "created_at_utc",
]

FEATURE_COLUMNS = [
    "feedback_id",
    "intent_id",
    "order_id",
    "symbol",
    "signal_asof_date",
    "fill_date",
    "fill_model",
    "order_status",
    "filled_flag",
    "expired_unfilled_flag",
    "raw_price",
    "fill_price",
    "expected_slippage_bps",
    "realized_slippage_bps",
    "slippage_error_bps",
    "mfe_pct",
    "mae_pct",
    "stop_hit",
    "target_hit",
    "live_trading_status",
]

LABEL_COLUMNS = [
    "feedback_id",
    "intent_id",
    "order_id",
    "symbol",
    "fill_date",
    "horizon_days",
    "label_matured",
    "label_include_for_model",
    "label_success",
    "label_post_fill_return_pct",
    "label_execution_adjusted_utility",
    "label_expired_unfilled",
    "label_stop_hit",
    "label_target_hit",
    "live_trading_status",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_id(*parts: object, prefix: str) -> str:
    text = "|".join("" if part is None else str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return strip_bom_columns(pd.read_csv(path, **kwargs))


def load_signals(path: Path) -> pd.DataFrame:
    signals = read_csv_if_exists(path, parse_dates=["date"])
    if signals.empty:
        return signals
    return signals.sort_values("date").reset_index(drop=True)


def first_fill_for_order(fills: pd.DataFrame, order_id: str) -> pd.Series:
    if fills.empty or "order_id" not in fills.columns:
        return pd.Series(dtype=object)
    rows = fills[fills["order_id"].astype(str).eq(str(order_id))]
    if rows.empty:
        return pd.Series(dtype=object)
    if "fill_type" in rows.columns:
        entry = rows[rows["fill_type"].astype(str).str.upper().eq("ENTRY")]
        if not entry.empty:
            rows = entry
    sort_cols = [col for col in ["fill_date", "created_at_utc"] if col in rows.columns]
    return (rows.sort_values(sort_cols) if sort_cols else rows).iloc[0]


def signal_window(signals: pd.DataFrame, fill_date: str, horizon_days: int) -> pd.DataFrame:
    if signals.empty or "date" not in signals.columns or not fill_date:
        return pd.DataFrame()
    fill_ts = pd.Timestamp(fill_date).normalize()
    dates = pd.to_datetime(signals["date"], errors="coerce").dt.normalize()
    return signals[dates >= fill_ts].head(int(horizon_days) + 1).copy()


def forward_outcome(order: pd.Series, fill: pd.Series, signals: pd.DataFrame, config: PaperFeedbackConfig) -> dict[str, object]:
    fill_price = as_float(fill.get("fill_price"))
    raw_price = as_float(fill.get("raw_price"))
    fill_date = str(fill.get("fill_date", ""))
    side = str(fill.get("side", order.get("side", "BUY"))).upper()
    window = signal_window(signals, fill_date, int(config.horizon_days))
    label_matured = len(window) >= int(config.horizon_days)
    if pd.isna(fill_price) or fill_price <= 0 or window.empty:
        return {
            "post_fill_return_pct": np.nan,
            "mfe_pct": np.nan,
            "mae_pct": np.nan,
            "stop_hit": False,
            "target_hit": False,
            "label_matured": False,
            "execution_adjusted_utility": np.nan,
            "realized_slippage_bps": np.nan,
        }

    close = as_float(window.tail(1).iloc[0].get("close"))
    high = pd.to_numeric(window.get("high", pd.Series(dtype=float)), errors="coerce").max()
    low = pd.to_numeric(window.get("low", pd.Series(dtype=float)), errors="coerce").min()
    long_side = side in {"BUY", "LONG"}
    if long_side:
        post_return = close / fill_price - 1.0 if pd.notna(close) else np.nan
        mfe = high / fill_price - 1.0 if pd.notna(high) else np.nan
        mae = low / fill_price - 1.0 if pd.notna(low) else np.nan
        realized_slippage = (fill_price / raw_price - 1.0) * 10000.0 if pd.notna(raw_price) and raw_price > 0 else np.nan
        stop_price = as_float(order.get("stop_price"))
        target_price = fill_price + float(config.target_r_multiple) * (fill_price - stop_price) if pd.notna(stop_price) and fill_price > stop_price else np.nan
        stop_hit = bool(pd.notna(stop_price) and pd.notna(low) and low <= stop_price)
        target_hit = bool(pd.notna(target_price) and pd.notna(high) and high >= target_price)
    else:
        post_return = 1.0 - close / fill_price if pd.notna(close) else np.nan
        mfe = 1.0 - low / fill_price if pd.notna(low) else np.nan
        mae = 1.0 - high / fill_price if pd.notna(high) else np.nan
        realized_slippage = (raw_price / fill_price - 1.0) * 10000.0 if pd.notna(raw_price) and raw_price > 0 and fill_price > 0 else np.nan
        stop_hit = False
        target_hit = False
    expected_slippage = as_float(fill.get("slippage_bps"), 0.0)
    utility = (
        post_return
        - float(config.lambda_mae) * abs(min(mae, 0.0) if pd.notna(mae) else 0.0)
        - float(config.lambda_slippage) * max(realized_slippage - expected_slippage, 0.0) / 10000.0
    )
    return {
        "post_fill_return_pct": float(post_return),
        "mfe_pct": float(mfe) if pd.notna(mfe) else np.nan,
        "mae_pct": float(mae) if pd.notna(mae) else np.nan,
        "stop_hit": stop_hit,
        "target_hit": target_hit,
        "label_matured": bool(label_matured),
        "execution_adjusted_utility": float(utility) if pd.notna(utility) else np.nan,
        "realized_slippage_bps": float(realized_slippage) if pd.notna(realized_slippage) else np.nan,
    }


def build_feedback_events(orders: pd.DataFrame, fills: pd.DataFrame, signals: pd.DataFrame, config: PaperFeedbackConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if orders.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    sort_cols = [col for col in ["signal_asof_date", "expected_fill_date", "submitted_at_utc"] if col in orders.columns]
    ordered = orders.sort_values(sort_cols) if sort_cols else orders
    for _, order in ordered.iterrows():
        order_id = str(order.get("order_id", ""))
        fill = first_fill_for_order(fills, order_id)
        order_status = str(order.get("status", "MISSING"))
        expired_unfilled = order_status == OrderStatus.EXPIRED.value
        filled = not fill.empty and str(fill.get("status", "")).upper() == FillStatus.FILLED.value
        partial = order_status == OrderStatus.PARTIALLY_FILLED.value
        if fill.empty:
            feedback_id = stable_id(order_id, "UNFILLED", config.horizon_days, prefix="xfb")
            event = ExecutionFeedbackEvent(
                feedback_id=feedback_id,
                intent_id=str(order.get("intent_id", "")),
                order_id=order_id,
                fill_id="",
                symbol=str(order.get("symbol", "")),
                signal_asof_date=str(order.get("signal_asof_date", "")),
                fill_date=str(order.get("expected_fill_date", "")),
                horizon_days=int(config.horizon_days),
                order_status=order_status,
                fill_status="UNFILLED" if expired_unfilled else "MISSING",
                fill_model=str(order.get("fill_model", "")),
                filled_flag=False,
                partial_fill_flag=bool(partial),
                expired_unfilled_flag=bool(expired_unfilled),
                raw_price=np.nan,
                fill_price=np.nan,
                expected_slippage_bps=np.nan,
                realized_slippage_bps=np.nan,
                slippage_error_bps=np.nan,
                post_fill_return_pct=np.nan,
                mfe_pct=np.nan,
                mae_pct=np.nan,
                stop_hit=False,
                target_hit=False,
                label_matured=bool(expired_unfilled),
                execution_adjusted_utility=-float(config.lambda_unfilled) if expired_unfilled else np.nan,
                live_trading_status=str(order.get("live_trading_status", "DISABLED_BY_DESIGN")),
                created_at_utc=now_utc_iso(),
            )
            rows.append(event.to_dict())
            continue

        outcome = forward_outcome(order, fill, signals, config)
        expected_slippage = as_float(fill.get("slippage_bps"), np.nan)
        realized_slippage = as_float(outcome.get("realized_slippage_bps"), np.nan)
        feedback_id = stable_id(order_id, fill.get("fill_id", ""), config.horizon_days, prefix="xfb")
        event = ExecutionFeedbackEvent(
            feedback_id=feedback_id,
            intent_id=str(order.get("intent_id", fill.get("intent_id", ""))),
            order_id=order_id,
            fill_id=str(fill.get("fill_id", "")),
            symbol=str(order.get("symbol", fill.get("symbol", ""))),
            signal_asof_date=str(order.get("signal_asof_date", "")),
            fill_date=str(fill.get("fill_date", "")),
            horizon_days=int(config.horizon_days),
            order_status=order_status,
            fill_status=str(fill.get("status", "")),
            fill_model=str(order.get("fill_model", "")),
            filled_flag=bool(filled),
            partial_fill_flag=bool(partial),
            expired_unfilled_flag=False,
            raw_price=as_float(fill.get("raw_price"), np.nan),
            fill_price=as_float(fill.get("fill_price"), np.nan),
            expected_slippage_bps=expected_slippage,
            realized_slippage_bps=realized_slippage,
            slippage_error_bps=realized_slippage - expected_slippage if pd.notna(realized_slippage) and pd.notna(expected_slippage) else np.nan,
            post_fill_return_pct=as_float(outcome.get("post_fill_return_pct"), np.nan),
            mfe_pct=as_float(outcome.get("mfe_pct"), np.nan),
            mae_pct=as_float(outcome.get("mae_pct"), np.nan),
            stop_hit=to_bool(outcome.get("stop_hit", False)),
            target_hit=to_bool(outcome.get("target_hit", False)),
            label_matured=to_bool(outcome.get("label_matured", False)),
            execution_adjusted_utility=as_float(outcome.get("execution_adjusted_utility"), np.nan),
            live_trading_status=str(order.get("live_trading_status", "DISABLED_BY_DESIGN")),
            created_at_utc=now_utc_iso(),
        )
        rows.append(event.to_dict())
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def build_feature_table(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return events[[col for col in FEATURE_COLUMNS if col in events.columns]].copy()


def build_label_table(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    labels = pd.DataFrame()
    labels["feedback_id"] = events["feedback_id"]
    labels["intent_id"] = events["intent_id"]
    labels["order_id"] = events["order_id"]
    labels["symbol"] = events["symbol"]
    labels["fill_date"] = events["fill_date"]
    labels["horizon_days"] = events["horizon_days"]
    labels["label_matured"] = events["label_matured"]
    filled = events["filled_flag"].astype(bool)
    matured = events["label_matured"].astype(bool)
    expired = events["expired_unfilled_flag"].astype(bool)
    labels["label_include_for_model"] = filled & matured
    labels["label_success"] = filled & matured & (pd.to_numeric(events["post_fill_return_pct"], errors="coerce") > 0)
    labels["label_post_fill_return_pct"] = events["post_fill_return_pct"]
    labels["label_execution_adjusted_utility"] = events["execution_adjusted_utility"]
    labels["label_expired_unfilled"] = expired
    labels["label_stop_hit"] = events["stop_hit"]
    labels["label_target_hit"] = events["target_hit"]
    labels["live_trading_status"] = events["live_trading_status"]
    return labels[LABEL_COLUMNS].copy()


def build_quality(events: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    live_disabled = bool(events.get("live_trading_status", pd.Series(["DISABLED_BY_DESIGN"])).astype(str).eq("DISABLED_BY_DESIGN").all()) if not events.empty else True
    duplicate_events = int(events.get("feedback_id", pd.Series(dtype=str)).duplicated().sum()) if not events.empty else 0
    matured_count = int(labels.get("label_matured", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1", "yes"]).sum()) if not labels.empty else 0
    return pd.DataFrame(
        [
            check_row("execution_feedback_events_contract", set(EVENT_COLUMNS).issubset(events.columns), "CRITICAL", len(events.columns)),
            check_row("execution_feedback_features_contract", set(FEATURE_COLUMNS).issubset(features.columns), "CRITICAL", len(features.columns)),
            check_row("execution_feedback_labels_contract", set(LABEL_COLUMNS).issubset(labels.columns), "CRITICAL", len(labels.columns)),
            check_row("execution_feedback_ids_unique", duplicate_events == 0, "CRITICAL", duplicate_events),
            check_row("execution_feedback_live_trading_disabled", live_disabled, "CRITICAL", "DISABLED_BY_DESIGN"),
            check_row("execution_feedback_matured_label_count", matured_count >= 0, "INFO", matured_count),
        ]
    )


def write_report(outdir: Path, events: pd.DataFrame, labels: pd.DataFrame, quality: pd.DataFrame) -> None:
    matured = int(labels.get("label_matured", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1", "yes"]).sum()) if not labels.empty else 0
    included = int(labels.get("label_include_for_model", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1", "yes"]).sum()) if not labels.empty else 0
    lines = [
        "# Top10 Paper Execution Feedback Report",
        "",
        f"- Feedback events: {len(events)}",
        f"- Matured labels: {matured}",
        f"- Model-eligible labels: {included}",
        "- Live trading status: DISABLED_BY_DESIGN",
        "",
        "## Quality Checks",
        "",
        "| Check | Passed | Value |",
        "|---|---:|---:|",
    ]
    for _, row in quality.iterrows():
        lines.append(f"| {row['check']} | {row['passed']} | {row['value']} |")
    lines.extend(["", "Immature paper outcomes are tracked but excluded from model-improvement labels."])
    (outdir / "tsm_execution_feedback_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper execution feedback features and labels.")
    parser.add_argument("--orders", default="tsm_price_rule_output/tsm_paper_orders.csv")
    parser.add_argument("--fills", default="tsm_price_rule_output/tsm_paper_fills.csv")
    parser.add_argument("--signals", default="tsm_price_rule_output/tsm_daily_algorithmic_signals.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--config", default="config/tsm_research.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = load_run_config(args.config).paper_feedback
    events = build_feedback_events(
        read_csv_if_exists(Path(args.orders)),
        read_csv_if_exists(Path(args.fills)),
        load_signals(Path(args.signals)),
        config,
    )
    features = build_feature_table(events)
    labels = build_label_table(events)
    quality = build_quality(events, features, labels)
    events.to_csv(outdir / "tsm_execution_feedback_events.csv", index=False)
    features.to_csv(outdir / "tsm_execution_feedback_features.csv", index=False)
    labels.to_csv(outdir / "tsm_execution_feedback_labels.csv", index=False)
    quality.to_csv(outdir / "tsm_execution_feedback_quality_checks.csv", index=False)
    write_report(outdir, events, labels, quality)
    print("completed: execution feedback outputs =", outdir.resolve())
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()

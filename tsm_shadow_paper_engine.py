#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maintain a shadow paper-trading ledger for prediction snapshots.

This engine records the prediction decision that would have been available at
the latest as-of date. It does not place orders. When enough future bars exist,
it realizes the paper prediction with the same horizon labeler used by the ML
prediction engine.

Outputs:
- tsm_shadow_paper_predictions.csv
- tsm_shadow_paper_quality_checks.csv
- tsm_shadow_paper_report.md
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from tsm_core.decision_schema import (
    PaperAction,
    is_decision_support_permission,
    is_long_allowed_decision,
)
from tsm_prediction_engine import (
    HORIZONS,
    actionable_entry_candidate,
    as_float,
    candidate_scope,
    entry_gate_status,
    event_candidate,
    label_event_horizon,
    prediction_universe,
    to_bool,
    trade_ready_entry_candidate,
)


LEDGER_COLUMNS = [
    "symbol",
    "prediction_asof_date",
    "horizon_days",
    "recorded_at_utc",
    "latest_is_event_candidate",
    "latest_is_actionable_entry_candidate",
    "latest_is_trade_ready_entry_candidate",
    "latest_entry_gate_status",
    "latest_candidate_scope",
    "prediction_scope_used",
    "best_model",
    "p_success",
    "p_success_lower_80",
    "p_success_upper_80",
    "threshold",
    "selected_by_threshold",
    "confidence_band",
    "prediction_quality_pass",
    "oos_event_count",
    "effective_oos_event_count",
    "p_stop_hit",
    "p_hit_1r",
    "p_hit_2r",
    "expected_r",
    "expected_net_return",
    "model_quality_block_reasons",
    "trade_ready_p_success",
    "trigger_p_success",
    "context_p_success",
    "prediction_signal_status",
    "prediction_use_status",
    "decision_permission",
    "final_trade_decision",
    "paper_action",
    "signal_entry_trigger",
    "signal_trade_action",
    "signal_close",
    "signal_atr_14",
    "realized_status",
    "realized_success",
    "realized_net_return_pct",
    "realized_gross_return_pct",
    "realized_expected_r",
    "realized_stop_survival",
    "realized_hit_1r_before_stop",
    "realized_hit_2r_before_stop",
    "realized_exit_reason",
    "realized_exit_type_clean",
    "realized_entry_date",
    "realized_exit_date",
    "realized_entry_price",
    "realized_exit_price",
    "realized_holding_trading_days",
    "realized_at_utc",
]

TEXT_LEDGER_COLUMNS = [
    "symbol",
    "prediction_asof_date",
    "recorded_at_utc",
    "latest_entry_gate_status",
    "latest_candidate_scope",
    "prediction_scope_used",
    "best_model",
    "confidence_band",
    "model_quality_block_reasons",
    "prediction_signal_status",
    "prediction_use_status",
    "decision_permission",
    "final_trade_decision",
    "paper_action",
    "signal_entry_trigger",
    "signal_trade_action",
    "realized_status",
    "realized_exit_reason",
    "realized_exit_type_clean",
    "realized_entry_date",
    "realized_exit_date",
    "realized_at_utc",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def read_snapshot(path: Path) -> Dict[str, object]:
    df = strip_bom_columns(pd.read_csv(path))
    require_columns(df, ["field", "value"], str(path))
    return dict(zip(df["field"].astype(str), df["value"]))


def read_signals(path: Path) -> pd.DataFrame:
    df = strip_bom_columns(pd.read_csv(path, parse_dates=["date"])).sort_values("date").reset_index(drop=True)
    require_columns(df, ["date", "open", "high", "low", "close", "atr_14", "entry_trigger", "trade_action"], str(path))
    if df["date"].duplicated().any():
        raise ValueError("signals has duplicated dates")
    df = df.copy()
    df["is_event_candidate"] = df.apply(event_candidate, axis=1)
    df["is_actionable_entry_candidate"] = df.apply(actionable_entry_candidate, axis=1)
    df["is_trade_ready_entry_candidate"] = df.apply(trade_ready_entry_candidate, axis=1)
    df["entry_gate_status"] = df.apply(entry_gate_status, axis=1)
    df["candidate_scope"] = df.apply(candidate_scope, axis=1)
    df["prediction_universe"] = df.apply(prediction_universe, axis=1)
    return df


def read_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    df = strip_bom_columns(pd.read_csv(path))
    for col in LEDGER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[LEDGER_COLUMNS].copy()
    for col in TEXT_LEDGER_COLUMNS:
        df[col] = df[col].astype("object")
    return df


def parse_date(value) -> pd.Timestamp:
    out = pd.to_datetime(value, errors="coerce")
    if pd.isna(out):
        raise ValueError(f"Invalid date value: {value}")
    return pd.Timestamp(out).normalize()


def clean_scalar(value):
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() == "nan":
            return np.nan
    return value


def snapshot_float(snapshot: Dict[str, object], key: str) -> float:
    return as_float(clean_scalar(snapshot.get(key)))


def snapshot_bool(snapshot: Dict[str, object], key: str) -> bool:
    return to_bool(clean_scalar(snapshot.get(key)))


def snapshot_text(snapshot: Dict[str, object], key: str, default: str = "") -> str:
    value = clean_scalar(snapshot.get(key, default))
    if pd.isna(value):
        return default
    return str(value)


def find_signal_row(signals: pd.DataFrame, asof_date: pd.Timestamp) -> pd.Series:
    matches = signals[signals["date"].dt.normalize().eq(asof_date)]
    if matches.empty:
        prior = signals[signals["date"].dt.normalize().le(asof_date)]
        if prior.empty:
            raise ValueError(f"No signal row on or before prediction_asof_date={asof_date.date().isoformat()}")
        return prior.iloc[-1]
    return matches.iloc[-1]


def finite_probability(value: float) -> bool:
    return pd.notna(value) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def paper_action(row: Dict[str, object]) -> str:
    if not bool(row["latest_is_event_candidate"]):
        return PaperAction.NO_SIGNAL.value
    if not is_decision_support_permission(row["decision_permission"]):
        return PaperAction.DISPLAY_ONLY_NO_PAPER_TRADE.value
    if not bool(row["prediction_quality_pass"]):
        return PaperAction.RULE_ONLY_QUALITY_BLOCKED.value
    if finite_probability(row["p_success"]) and finite_probability(row["threshold"]) and float(row["p_success"]) >= float(row["threshold"]):
        if is_long_allowed_decision(row["final_trade_decision"]):
            return PaperAction.PAPER_LONG_CONFIRMED.value
        return PaperAction.PAPER_WATCH_CONFIRMED.value
    return PaperAction.PAPER_FILTERED.value


def build_latest_rows(snapshot: Dict[str, object], signals: pd.DataFrame, symbol: str) -> pd.DataFrame:
    asof_date = parse_date(snapshot.get("prediction_asof_date"))
    signal_row = find_signal_row(signals, asof_date)
    recorded_at = now_utc_iso()
    rows: List[Dict[str, object]] = []
    for horizon in HORIZONS:
        row = {
            "symbol": symbol,
            "prediction_asof_date": asof_date.date().isoformat(),
            "horizon_days": int(horizon),
            "recorded_at_utc": recorded_at,
            "latest_is_event_candidate": snapshot_bool(snapshot, "latest_is_event_candidate"),
            "latest_is_actionable_entry_candidate": snapshot_bool(snapshot, "latest_is_actionable_entry_candidate"),
            "latest_is_trade_ready_entry_candidate": snapshot_bool(snapshot, "latest_is_trade_ready_entry_candidate"),
            "latest_entry_gate_status": snapshot_text(snapshot, "latest_entry_gate_status"),
            "latest_candidate_scope": snapshot_text(snapshot, "latest_candidate_scope"),
            "prediction_scope_used": snapshot_text(snapshot, "prediction_scope_used"),
            "best_model": snapshot_text(snapshot, f"best_model_{horizon}d"),
            "p_success": snapshot_float(snapshot, f"p_success_{horizon}d"),
            "p_success_lower_80": snapshot_float(snapshot, f"p_success_lower_80_{horizon}d"),
            "p_success_upper_80": snapshot_float(snapshot, f"p_success_upper_80_{horizon}d"),
            "threshold": snapshot_float(snapshot, f"threshold_{horizon}d"),
            "confidence_band": snapshot_text(snapshot, f"confidence_band_{horizon}d"),
            "prediction_quality_pass": snapshot_bool(snapshot, f"prediction_quality_pass_{horizon}d"),
            "oos_event_count": snapshot_float(snapshot, f"oos_event_count_{horizon}d"),
            "effective_oos_event_count": snapshot_float(snapshot, f"effective_oos_event_count_{horizon}d"),
            "p_stop_hit": snapshot_float(snapshot, f"p_stop_hit_{horizon}d"),
            "p_hit_1r": snapshot_float(snapshot, f"p_hit_1r_{horizon}d"),
            "p_hit_2r": snapshot_float(snapshot, f"p_hit_2r_{horizon}d"),
            "expected_r": snapshot_float(snapshot, f"expected_r_{horizon}d"),
            "expected_net_return": snapshot_float(snapshot, f"expected_net_return_{horizon}d"),
            "model_quality_block_reasons": snapshot_text(snapshot, f"model_quality_block_reasons_{horizon}d"),
            "trade_ready_p_success": snapshot_float(snapshot, f"trade_ready_p_success_{horizon}d"),
            "trigger_p_success": snapshot_float(snapshot, f"trigger_p_success_{horizon}d"),
            "context_p_success": snapshot_float(snapshot, f"context_p_success_{horizon}d"),
            "prediction_signal_status": snapshot_text(snapshot, "prediction_signal_status"),
            "prediction_use_status": snapshot_text(snapshot, "prediction_use_status"),
            "decision_permission": snapshot_text(snapshot, "decision_permission"),
            "final_trade_decision": snapshot_text(snapshot, "final_trade_decision"),
            "signal_entry_trigger": signal_row.get("entry_trigger", ""),
            "signal_trade_action": signal_row.get("trade_action", ""),
            "signal_close": as_float(signal_row.get("close")),
            "signal_atr_14": as_float(signal_row.get("atr_14")),
            "realized_status": "PENDING",
        }
        row["selected_by_threshold"] = bool(finite_probability(row["p_success"]) and finite_probability(row["threshold"]) and row["p_success"] >= row["threshold"])
        row["paper_action"] = paper_action(row)
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in LEDGER_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[LEDGER_COLUMNS].copy()


def upsert_latest_rows(ledger: pd.DataFrame, latest_rows: pd.DataFrame) -> pd.DataFrame:
    if latest_rows.empty:
        return ledger
    ledger = ledger.copy()
    latest_keys = set(zip(latest_rows["symbol"], latest_rows["prediction_asof_date"], latest_rows["horizon_days"].astype(int)))
    if not ledger.empty:
        ledger_keys = list(zip(ledger["symbol"], ledger["prediction_asof_date"], pd.to_numeric(ledger["horizon_days"], errors="coerce").fillna(-1).astype(int)))
        keep = [key not in latest_keys for key in ledger_keys]
        ledger = ledger.loc[keep].copy()
    out = pd.concat([ledger, latest_rows], ignore_index=True)
    return out[LEDGER_COLUMNS].copy()


def signal_index_by_date(signals: pd.DataFrame) -> Dict[pd.Timestamp, int]:
    return {pd.Timestamp(row["date"]).normalize(): int(i) for i, row in signals.iterrows()}


def update_realized_outcomes(
    ledger: pd.DataFrame,
    signals: pd.DataFrame,
    commission_bps: float,
    slippage_bps: float,
    stop_multiple: float,
) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    out = ledger.copy()
    for col in TEXT_LEDGER_COLUMNS:
        out[col] = out[col].astype("object")
    idx_by_date = signal_index_by_date(signals)
    realized_at = now_utc_iso()
    for i, row in out.iterrows():
        asof = parse_date(row["prediction_asof_date"])
        horizon = int(row["horizon_days"])
        signal_idx = idx_by_date.get(asof)
        if signal_idx is None:
            out.loc[i, "realized_status"] = "MISSING_SIGNAL_DATE"
            continue
        labels = label_event_horizon(signals, signal_idx, horizon, commission_bps, slippage_bps, stop_multiple)
        status = labels.get(f"label_status_{horizon}d", "UNKNOWN")
        out.loc[i, "realized_status"] = status
        out.loc[i, "realized_success"] = labels.get(f"label_success_{horizon}d")
        out.loc[i, "realized_net_return_pct"] = labels.get(f"label_net_return_pct_{horizon}d")
        out.loc[i, "realized_gross_return_pct"] = labels.get(f"label_gross_return_pct_{horizon}d")
        out.loc[i, "realized_expected_r"] = labels.get(f"label_expected_r_{horizon}d")
        out.loc[i, "realized_stop_survival"] = labels.get(f"label_stop_survival_{horizon}d")
        out.loc[i, "realized_hit_1r_before_stop"] = labels.get(f"label_hit_1r_before_stop_{horizon}d")
        out.loc[i, "realized_hit_2r_before_stop"] = labels.get(f"label_hit_2r_before_stop_{horizon}d")
        out.loc[i, "realized_exit_reason"] = labels.get(f"label_exit_reason_{horizon}d")
        out.loc[i, "realized_exit_type_clean"] = labels.get(f"label_exit_type_clean_{horizon}d")
        out.loc[i, "realized_entry_date"] = labels.get(f"label_entry_date_{horizon}d")
        out.loc[i, "realized_exit_date"] = labels.get(f"label_exit_date_{horizon}d")
        out.loc[i, "realized_entry_price"] = labels.get(f"label_entry_price_{horizon}d")
        out.loc[i, "realized_exit_price"] = labels.get(f"label_exit_price_{horizon}d")
        out.loc[i, "realized_holding_trading_days"] = labels.get(f"label_holding_trading_days_{horizon}d")
        if status == "LABELED":
            out.loc[i, "realized_at_utc"] = realized_at
    return out[LEDGER_COLUMNS].copy()


def check_row(check: str, passed: bool, severity: str, value, details: str = "") -> Dict[str, object]:
    return {"check": check, "passed": bool(passed), "severity": severity, "value": value, "details": details}


def build_quality_checks(ledger: pd.DataFrame, latest_rows: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = [
        check_row("shadow_ledger_non_empty", not ledger.empty, "CRITICAL", len(ledger)),
        check_row("shadow_latest_rows_recorded_for_all_horizons", len(latest_rows) == len(HORIZONS), "CRITICAL", len(latest_rows)),
        check_row("shadow_no_order_execution", True, "CRITICAL", "paper_only"),
    ]
    if not ledger.empty:
        duplicate_count = int(ledger.duplicated(["symbol", "prediction_asof_date", "horizon_days"]).sum())
        rows.append(check_row("shadow_no_duplicate_symbol_date_horizon", duplicate_count == 0, "CRITICAL", duplicate_count))
        known_statuses = {
            "PENDING",
            "LABELED",
            "NOT_EVENT",
            "UNAVAILABLE_NEXT_OPEN",
            "UNAVAILABLE_FUTURE_WINDOW",
            "INVALID_ENTRY_DATA",
            "INVALID_RISK_DISTANCE",
            "INVALID_EXIT_DATA",
            "MISSING_SIGNAL_DATE",
        }
        unknown_statuses = sorted(set(ledger["realized_status"].dropna().astype(str)) - known_statuses)
        rows.append(check_row("shadow_realized_status_contract", not unknown_statuses, "CRITICAL", "|".join(unknown_statuses)))
        latest_key_count = int(
            ledger.merge(
                latest_rows[["symbol", "prediction_asof_date", "horizon_days"]],
                on=["symbol", "prediction_asof_date", "horizon_days"],
                how="inner",
            ).shape[0]
        )
        rows.append(check_row("shadow_latest_upsert_visible", latest_key_count == len(latest_rows), "CRITICAL", latest_key_count))
        latest_dates = pd.to_datetime(ledger["prediction_asof_date"], errors="coerce")
        rows.append(check_row("shadow_prediction_dates_not_after_signal_end", bool((latest_dates <= signals["date"].max().normalize()).all()), "CRITICAL", signals["date"].max().date().isoformat()))
        labeled = ledger[ledger["realized_status"].eq("LABELED")].copy()
        if not labeled.empty:
            success_values_valid = pd.to_numeric(labeled["realized_success"], errors="coerce").isin([0, 1]).all()
            rows.append(check_row("shadow_labeled_success_binary", bool(success_values_valid), "CRITICAL", len(labeled)))
        else:
            rows.append(check_row("shadow_labeled_rows_available", False, "WARN", 0, "Expected to be 0 until enough future bars accrue."))
    if not latest_rows.empty:
        no_signal_rows = latest_rows[latest_rows["latest_is_event_candidate"].eq(False)]
        if not no_signal_rows.empty:
            rows.append(check_row("shadow_no_signal_rows_have_no_signal_action", bool(no_signal_rows["paper_action"].eq("NO_SIGNAL").all()), "CRITICAL", "|".join(no_signal_rows["paper_action"].astype(str).unique())))
    return pd.DataFrame(rows)


def fmt(value, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(outdir: Path, ledger: pd.DataFrame, quality: pd.DataFrame) -> None:
    latest = ledger.sort_values(["prediction_asof_date", "horizon_days"]).tail(len(HORIZONS)) if not ledger.empty else pd.DataFrame()
    realized = ledger[ledger["realized_status"].eq("LABELED")].copy() if not ledger.empty else pd.DataFrame()
    lines = [
        "# TSMC Shadow Paper Prediction Report",
        "",
        "This ledger records prediction decisions only. It does not place orders.",
        "",
        "## Quality",
        "",
        "| Check | Passed | Severity | Value |",
        "|---|---:|---|---:|",
    ]
    for _, row in quality.iterrows():
        lines.append(f"| {row['check']} | {row['passed']} | {row['severity']} | {row['value']} |")

    lines.extend(
        [
            "",
            "## Latest Paper Rows",
            "",
            "| As Of | Horizon | Action | Scope | Model | P(success) | Threshold | Status | Realized |",
            "|---|---:|---|---|---|---:|---:|---|---|",
        ]
    )
    if latest.empty:
        lines.append("| NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    else:
        for _, row in latest.iterrows():
            lines.append(
                f"| {row['prediction_asof_date']} | {int(row['horizon_days'])} | {row['paper_action']} | "
                f"{row['prediction_scope_used']} | {row['best_model']} | {fmt(row['p_success'], 4)} | "
                f"{fmt(row['threshold'], 4)} | {row['prediction_use_status']} | {row['realized_status']} |"
            )

    lines.extend(
        [
            "",
            "## Realized Summary",
            f"- Ledger rows: {len(ledger)}",
            f"- Realized labeled rows: {len(realized)}",
        ]
    )
    if not realized.empty:
        lines.append(f"- Realized success rate: {fmt(pd.to_numeric(realized['realized_success'], errors='coerce').mean() * 100.0)}%")
        lines.append(f"- Mean realized net return: {fmt(pd.to_numeric(realized['realized_net_return_pct'], errors='coerce').mean())}%")
    lines.extend(["", "This report is research tooling, not investment advice."])
    (outdir / "tsm_shadow_paper_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maintain shadow paper prediction ledger.")
    parser.add_argument("--prediction", default="tsm_price_rule_output/tsm_latest_prediction_snapshot.csv")
    parser.add_argument("--signals", default="tsm_price_rule_output/tsm_daily_algorithmic_signals.csv")
    parser.add_argument("--ledger", default="tsm_price_rule_output/tsm_shadow_paper_predictions.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--symbol", default="TSM")
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stop-multiple", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    snapshot = read_snapshot(Path(args.prediction))
    signals = read_signals(Path(args.signals))
    latest_rows = build_latest_rows(snapshot, signals, args.symbol.upper())
    ledger_path = Path(args.ledger)
    ledger = read_ledger(ledger_path)
    ledger = upsert_latest_rows(ledger, latest_rows)
    ledger = update_realized_outcomes(ledger, signals, args.commission_bps, args.slippage_bps, args.stop_multiple)
    quality = build_quality_checks(ledger, latest_rows, signals)

    ledger.to_csv(outdir / "tsm_shadow_paper_predictions.csv", index=False)
    quality.to_csv(outdir / "tsm_shadow_paper_quality_checks.csv", index=False)
    write_report(outdir, ledger, quality)
    print("completed: shadow paper outputs =", outdir.resolve())
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()

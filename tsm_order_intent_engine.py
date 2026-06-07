#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build broker-free paper order intents from the latest TSM signal stack.

This engine stops before any broker boundary. It creates an auditable intent
ledger for paper execution only and keeps live trading disabled by design.

Outputs:
- tsm_order_intents.csv
- tsm_order_intent_quality_checks.csv
- tsm_order_intent_report.md
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tsm_core.config import PaperOmsConfig, load_run_config
from tsm_core.execution import ExecutionBlockReason, IntentStatus, OrderIntent
from tsm_core.io import as_float, check_row, read_csv, require_columns, strip_bom_columns, to_bool
from tsm_core.universe import UniverseMember, load_universe_members


INTENT_COLUMNS = [
    "intent_id",
    "signal_id",
    "signal_version",
    "rule_family",
    "hypothesis_id",
    "model_version",
    "risk_policy_version",
    "symbol",
    "symbol_group",
    "asof_date",
    "side",
    "order_type",
    "target_weight",
    "max_notional",
    "entry_reference_price",
    "stop_price",
    "limit_price",
    "time_in_force",
    "max_slippage_bps",
    "strategy_id",
    "decision_source",
    "available_at_utc",
    "created_at_utc",
    "status",
    "reason",
    "live_trading_status",
    "live_order_blocked",
    "live_block_reason",
    "risk_state",
    "prediction_use_status",
    "final_trade_decision",
    "paper_decision_score_20d",
    "p_success_20d",
    "expected_r_20d",
    "p_stop_hit_20d",
    "research_signal_score",
    "raw_entry_event",
    "decision_tier",
    "sizing_tier",
    "suggested_action",
    "suggested_weight",
    "semi_momentum_regime",
    "stop_price_1_8atr",
    "invalidation_5d_low",
    "invalidation_ema10",
    "next_check_condition",
    "prediction_block_reasons",
    "paper_oms_enabled",
    "kill_switch_active",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_id(*parts: object, prefix: str) -> str:
    text = "|".join("" if part is None else str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def read_snapshot(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    df = strip_bom_columns(pd.read_csv(path))
    if df.empty or not {"field", "value"}.issubset(df.columns):
        return {}
    return dict(zip(df["field"].astype(str), df["value"]))


def read_existing_intents(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=INTENT_COLUMNS)
    df = strip_bom_columns(pd.read_csv(path))
    for col in INTENT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[INTENT_COLUMNS].copy()


def normalize_weight(value: object) -> float:
    parsed = as_float(value)
    if pd.isna(parsed):
        return 0.0
    return float(parsed / 100.0 if abs(parsed) > 1 else parsed)


def load_signals(path: Path) -> pd.DataFrame:
    df = read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    require_columns(df, ["date", "close", "entry_trigger", "trade_action", "atr_14"], str(path))
    return df


def first_text(*values: object, default: str = "") -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "na", "n/a"}:
            return text
    return default


def latest_signal_id(row: pd.Series, symbol: str = "TSM") -> str:
    existing = first_text(row.get("signal_id"))
    if existing:
        return existing
    return stable_id(symbol, row.get("date"), row.get("entry_trigger"), row.get("trade_action"), prefix="sig")


def actionable_signal(row: pd.Series, prediction: dict[str, object], system_state: dict[str, object]) -> tuple[bool, str, float]:
    trade_action = str(row.get("trade_action", "NO_TRADE"))
    research_action = str(row.get("research_signal_action", "NO_ACTION"))
    decision_tier = str(row.get("decision_tier", ""))
    suggested_action = str(row.get("suggested_action", ""))
    final_decision = str(prediction.get("final_trade_decision", ""))
    paper_ready = to_bool(system_state.get("paper_ready", False))
    if trade_action == "ENTRY_ALLOWED":
        return True, "STRICT_RULE_ENTRY", normalize_weight(row.get("position_weight_if_0_5pct_account_risk", 0.0))
    if decision_tier == "AGGRESSIVE_TREND_ENTRY":
        return True, "SEMI_MOMENTUM_ENTRY", normalize_weight(row.get("suggested_weight", 0.04))
    if decision_tier == "BREAKOUT_EXTENSION_TINY":
        return True, "HIGH_VOL_TINY_EXTENSION", normalize_weight(row.get("suggested_weight", 0.01))
    if decision_tier == "PULLBACK_REENTRY":
        return True, "PULLBACK_REENTRY", normalize_weight(row.get("suggested_weight", 0.02))
    if suggested_action in {"SEMI_MOMENTUM_ENTRY", "HIGH_VOL_TINY_EXTENSION", "PULLBACK_REENTRY"}:
        return True, suggested_action, normalize_weight(row.get("suggested_weight", 0.0))
    if research_action == "PAPER_TRACK_LONG_SETUP":
        return True, "RULE_RESEARCH_PAPER_SETUP", normalize_weight(row.get("paper_tracking_weight", 0.02))
    if to_bool(prediction.get("paper_decision_support_allowed", False)):
        return True, "POOLED_PAPER_DECISION", normalize_weight(row.get("paper_tracking_weight", row.get("position_weight_if_0_5pct_account_risk", 0.02)))
    if to_bool(prediction.get("decision_support_allowed", False)):
        return True, "POOLED_DECISION_SUPPORT", normalize_weight(row.get("paper_tracking_weight", row.get("position_weight_if_0_5pct_account_risk", 0.02)))
    if final_decision in {"ALPHA_RESEARCH_LONG_ALLOWED", "ALPHA_RESEARCH_SMALL_LONG_ALLOWED"} and paper_ready:
        return True, "PREDICTION_PAPER_DECISION", normalize_weight(row.get("paper_tracking_weight", row.get("position_weight_if_0_5pct_account_risk", 0.02)))
    return False, "NO_ACTIONABLE_SIGNAL", 0.0


def build_latest_intent(
    signals: pd.DataFrame,
    risk_snapshot: dict[str, object],
    prediction_snapshot: dict[str, object],
    system_state: dict[str, object],
    config: PaperOmsConfig,
    symbol: str,
) -> dict[str, object]:
    row = signals.iloc[-1]
    asof_date = pd.Timestamp(row["date"]).date().isoformat()
    symbol = str(symbol).upper()
    signal_id = latest_signal_id(row, symbol=symbol)
    risk_state = first_text(risk_snapshot.get("risk_state"), default="UNKNOWN")
    risk_weight = normalize_weight(risk_snapshot.get("final_recommended_max_weight", row.get("position_weight_if_0_5pct_account_risk", 0.0)))
    has_signal, signal_reason, signal_weight = actionable_signal(row, prediction_snapshot, system_state)
    target_weight = max(0.0, min(weight for weight in [risk_weight, signal_weight, 1.0] if pd.notna(weight)))
    close = as_float(row.get("close"))
    stop_price = as_float(row.get("stop_price_1_8atr"), as_float(row.get("atr_stop_2x"), close - 2.0 * as_float(row.get("atr_14", 0.0))))
    max_notional = float(config.account_equity) * target_weight

    status = IntentStatus.APPROVED.value
    reason = signal_reason
    if not bool(config.enabled):
        status = IntentStatus.BLOCKED.value
        reason = ExecutionBlockReason.PAPER_OMS_DISABLED.value
        target_weight = 0.0
        max_notional = 0.0
    elif bool(config.kill_switch_active):
        status = IntentStatus.BLOCKED.value
        reason = ExecutionBlockReason.KILL_SWITCH_ACTIVE.value
        target_weight = 0.0
        max_notional = 0.0
    elif risk_state in {"UNKNOWN", "NO_NEW_RISK", "RESEARCH_ONLY_NO_NEW_RISK"} or risk_weight <= 0:
        status = IntentStatus.REJECTED.value
        reason = f"{ExecutionBlockReason.RISK_LIMIT_EXCEEDED.value}:{risk_state}"
        target_weight = 0.0
        max_notional = 0.0
    elif not has_signal or target_weight <= 0:
        status = IntentStatus.REJECTED.value
        reason = signal_reason
        target_weight = 0.0
        max_notional = 0.0

    order_type = str(config.default_order_type).upper()
    limit_price = close if order_type == "LIMIT" and pd.notna(close) else np.nan
    intent_id = stable_id(symbol, asof_date, signal_id, order_type, round(target_weight, 8), prefix="intent")
    intent = OrderIntent(
        intent_id=intent_id,
        signal_id=signal_id,
        signal_version=first_text(row.get("signal_version"), default="tsm_rule_v1"),
        rule_family=first_text(row.get("rule_family"), default="price_momentum_risk"),
        hypothesis_id=first_text(row.get("hypothesis_id"), default="top10_price_momentum_rule_v1"),
        model_version=first_text(prediction_snapshot.get("best_model_20d"), prediction_snapshot.get("pooled_model_name"), default="rule_only"),
        risk_policy_version="risk_policy_v1",
        symbol=symbol,
        asof_date=asof_date,
        side="BUY",
        order_type=order_type,
        target_weight=float(target_weight),
        max_notional=float(max_notional),
        entry_reference_price=float(close) if pd.notna(close) else np.nan,
        stop_price=float(stop_price) if pd.notna(stop_price) else np.nan,
        limit_price=float(limit_price) if pd.notna(limit_price) else None,
        time_in_force=str(config.time_in_force).upper(),
        max_slippage_bps=float(config.max_slippage_bps),
        strategy_id=first_text(row.get("decision_tier"), row.get("entry_trigger"), row.get("research_signal_stage"), default="NO_SIGNAL"),
        decision_source=reason,
        available_at_utc=first_text(row.get("available_at_utc"), default=f"{asof_date}T21:00:00+00:00"),
        created_at_utc=now_utc_iso(),
        status=status,
        reason=reason,
        live_trading_status="DISABLED_BY_DESIGN",
        live_order_blocked=True,
        live_block_reason=ExecutionBlockReason.LIVE_TRADING_DISABLED.value,
    ).to_dict()
    intent.update(
        {
            "symbol_group": first_text(row.get("symbol_group"), prediction_snapshot.get("symbol_group"), default="semiconductor"),
            "risk_state": risk_state,
            "prediction_use_status": first_text(prediction_snapshot.get("prediction_use_status"), default="UNKNOWN"),
            "final_trade_decision": first_text(prediction_snapshot.get("final_trade_decision"), default="UNKNOWN"),
            "paper_decision_score_20d": as_float(prediction_snapshot.get("paper_decision_score_20d"), np.nan),
            "p_success_20d": as_float(prediction_snapshot.get("p_success_20d"), np.nan),
            "expected_r_20d": as_float(prediction_snapshot.get("expected_r_20d", prediction_snapshot.get("expected_r_net_20d")), np.nan),
            "p_stop_hit_20d": as_float(prediction_snapshot.get("p_stop_hit_20d"), np.nan),
            "research_signal_score": as_float(row.get("research_signal_score"), np.nan),
            "raw_entry_event": first_text(row.get("raw_entry_event"), default="NONE"),
            "decision_tier": first_text(row.get("decision_tier"), default="LEGACY_ONLY"),
            "sizing_tier": first_text(row.get("sizing_tier"), default="LEGACY_ONLY"),
            "suggested_action": first_text(row.get("suggested_action"), default="LEGACY_ONLY"),
            "suggested_weight": normalize_weight(row.get("suggested_weight", 0.0)),
            "semi_momentum_regime": first_text(row.get("semi_momentum_regime"), default="UNKNOWN"),
            "stop_price_1_8atr": as_float(row.get("stop_price_1_8atr"), np.nan),
            "invalidation_5d_low": as_float(row.get("invalidation_5d_low"), np.nan),
            "invalidation_ema10": as_float(row.get("invalidation_ema10"), np.nan),
            "next_check_condition": first_text(row.get("next_check_condition"), default=""),
            "prediction_block_reasons": first_text(prediction_snapshot.get("block_reasons"), prediction_snapshot.get("decision_block_reasons"), default=""),
            "paper_oms_enabled": bool(config.enabled),
            "kill_switch_active": bool(config.kill_switch_active),
        }
    )
    return intent


def upsert_intent(ledger: pd.DataFrame, latest: dict[str, object]) -> pd.DataFrame:
    latest_df = pd.DataFrame([latest])
    for col in INTENT_COLUMNS:
        if col not in latest_df.columns:
            latest_df[col] = np.nan
    if ledger.empty:
        out = latest_df[INTENT_COLUMNS].copy()
    else:
        kept = ledger[ledger["intent_id"].astype(str) != str(latest["intent_id"])].copy()
        out = pd.concat([kept, latest_df[INTENT_COLUMNS]], ignore_index=True)
    return out[INTENT_COLUMNS].copy()


def upsert_intents(ledger: pd.DataFrame, latest_rows: list[dict[str, object]]) -> pd.DataFrame:
    if not latest_rows:
        return ledger[INTENT_COLUMNS].copy() if not ledger.empty else pd.DataFrame(columns=INTENT_COLUMNS)
    latest_df = pd.DataFrame(latest_rows)
    for col in INTENT_COLUMNS:
        if col not in latest_df.columns:
            latest_df[col] = np.nan
    if ledger.empty:
        return latest_df[INTENT_COLUMNS].copy()
    latest_symbols = set(latest_df["symbol"].dropna().astype(str).str.upper())
    if latest_symbols and "symbol" in ledger.columns:
        kept = ledger[~ledger["symbol"].astype(str).str.upper().isin(latest_symbols)].copy()
    else:
        latest_keys = set(latest_df["intent_id"].dropna().astype(str))
        kept = ledger[~ledger["intent_id"].astype(str).isin(latest_keys)].copy()
    return pd.concat([kept, latest_df[INTENT_COLUMNS]], ignore_index=True)[INTENT_COLUMNS].copy()


def build_quality_checks(intents: pd.DataFrame, latest: dict[str, object]) -> pd.DataFrame:
    duplicate_count = int(intents["intent_id"].duplicated().sum()) if not intents.empty and "intent_id" in intents.columns else 0
    rows = [
        check_row("order_intent_file_contract", set(INTENT_COLUMNS).issubset(intents.columns), "CRITICAL", len(intents.columns), details="required output columns present"),
        check_row("order_intent_latest_recorded", bool(latest.get("intent_id")), "CRITICAL", latest.get("intent_id", "")),
        check_row("order_intent_no_duplicate_intent_id", duplicate_count == 0, "CRITICAL", duplicate_count),
        check_row("order_intent_idempotent_latest", duplicate_count == 0, "CRITICAL", latest.get("intent_id", "")),
        check_row("order_intent_live_trading_disabled", str(latest.get("live_trading_status")) == "DISABLED_BY_DESIGN", "CRITICAL", latest.get("live_trading_status")),
        check_row("order_intent_live_order_blocked", to_bool(latest.get("live_order_blocked", True)), "CRITICAL", latest.get("live_block_reason")),
        check_row("order_intent_has_signal_id", bool(str(latest.get("signal_id", "")).strip()), "CRITICAL", latest.get("signal_id", "")),
        check_row("order_intent_target_weight_nonnegative", as_float(latest.get("target_weight"), 0.0) >= 0.0, "CRITICAL", latest.get("target_weight")),
    ]
    return pd.DataFrame(rows)


def write_report(outdir: Path, latest: dict[str, object], quality: pd.DataFrame) -> None:
    lines = [
        "# Top10 Order Intent Report",
        "",
        f"- Intent ID: {latest.get('intent_id', 'NA')}",
        f"- Signal ID: {latest.get('signal_id', 'NA')}",
        f"- Status: {latest.get('status', 'NA')}",
        f"- Reason: {latest.get('reason', 'NA')}",
        f"- Target weight: {as_float(latest.get('target_weight'), 0.0) * 100:.2f}%",
        f"- Live trading status: {latest.get('live_trading_status', 'DISABLED_BY_DESIGN')}",
        "",
        "## Quality Checks",
        "",
        "| Check | Passed | Value |",
        "|---|---:|---:|",
    ]
    for _, row in quality.iterrows():
        lines.append(f"| {row['check']} | {row['passed']} | {row['value']} |")
    lines.extend(["", "This report is paper OMS tooling. It does not submit broker orders."])
    (outdir / "tsm_order_intent_report.md").write_text("\n".join(lines), encoding="utf-8")


def latest_prediction_by_symbol(path: Path, symbol: str) -> dict[str, object]:
    if not path.exists():
        return {}
    frame = strip_bom_columns(pd.read_csv(path))
    if frame.empty or "symbol" not in frame.columns:
        return {}
    matched = frame[frame["symbol"].astype(str).str.upper().eq(str(symbol).upper())].copy()
    if matched.empty:
        return {}
    if "date" in matched.columns:
        matched["date"] = pd.to_datetime(matched["date"], errors="coerce")
        matched = matched.sort_values("date")
    return matched.tail(1).iloc[0].to_dict()


def member_signal_path(member: UniverseMember, root_signals: Path) -> Path:
    if member.paths.signals.exists():
        return member.paths.signals
    if member.symbol.upper() == "TSM" and root_signals.exists():
        return root_signals
    return member.paths.signals


def member_risk_path(member: UniverseMember, root_risk: Path) -> Path:
    if member.paths.risk_snapshot.exists():
        return member.paths.risk_snapshot
    if member.symbol.upper() == "TSM" and root_risk.exists():
        return root_risk
    return member.paths.risk_snapshot


def rejected_missing_intent(member: UniverseMember, reason: str, config: PaperOmsConfig) -> dict[str, object]:
    asof_date = ""
    signal_id = stable_id(member.symbol, "missing", reason, prefix="sig")
    intent_id = stable_id(member.symbol, asof_date, signal_id, reason, prefix="intent")
    intent = OrderIntent(
        intent_id=intent_id,
        signal_id=signal_id,
        symbol=member.symbol,
        asof_date=asof_date,
        side="BUY",
        order_type=str(config.default_order_type).upper(),
        target_weight=0.0,
        max_notional=0.0,
        entry_reference_price=np.nan,
        stop_price=np.nan,
        limit_price=None,
        time_in_force=str(config.time_in_force).upper(),
        max_slippage_bps=float(config.max_slippage_bps),
        strategy_id="MISSING",
        decision_source=reason,
        available_at_utc="",
        created_at_utc=now_utc_iso(),
        status=IntentStatus.REJECTED.value,
        reason=reason,
        live_trading_status="DISABLED_BY_DESIGN",
        live_order_blocked=True,
        live_block_reason=ExecutionBlockReason.LIVE_TRADING_DISABLED.value,
    ).to_dict()
    intent.update(
        {
            "symbol_group": member.symbol_group,
            "risk_state": "UNKNOWN",
            "prediction_use_status": "UNKNOWN",
            "final_trade_decision": "UNKNOWN",
            "paper_decision_score_20d": np.nan,
            "p_success_20d": np.nan,
            "expected_r_20d": np.nan,
            "p_stop_hit_20d": np.nan,
            "research_signal_score": np.nan,
            "raw_entry_event": "NONE",
            "decision_tier": "MISSING",
            "sizing_tier": "MISSING",
            "suggested_action": "NO_DATA",
            "suggested_weight": 0.0,
            "semi_momentum_regime": "UNKNOWN",
            "stop_price_1_8atr": np.nan,
            "invalidation_5d_low": np.nan,
            "invalidation_ema10": np.nan,
            "next_check_condition": reason,
            "prediction_block_reasons": reason,
            "paper_oms_enabled": bool(config.enabled),
            "kill_switch_active": bool(config.kill_switch_active),
        }
    )
    return intent


def build_batch_intents(
    members: list[UniverseMember],
    root_signals: Path,
    root_risk: Path,
    latest_predictions: Path,
    system_state: dict[str, object],
    config: PaperOmsConfig,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    intents: list[dict[str, object]] = []
    latest_signal_rows: list[dict[str, object]] = []
    latest_risk_rows: list[dict[str, object]] = []
    for member in members:
        signal_path = member_signal_path(member, root_signals)
        if not signal_path.exists():
            intents.append(rejected_missing_intent(member, "MISSING_SIGNALS", config))
            continue
        try:
            signals = load_signals(signal_path)
        except Exception:
            intents.append(rejected_missing_intent(member, "INVALID_SIGNALS", config))
            continue
        if signals.empty:
            intents.append(rejected_missing_intent(member, "EMPTY_SIGNALS", config))
            continue
        if "symbol" not in signals.columns:
            signals["symbol"] = member.symbol
        if "symbol_group" not in signals.columns:
            signals["symbol_group"] = member.symbol_group
        latest_signal_rows.append(signals.tail(1).iloc[0].to_dict())
        risk_snapshot = read_snapshot(member_risk_path(member, root_risk))
        if risk_snapshot:
            latest_risk_rows.append({"symbol": member.symbol, "symbol_group": member.symbol_group, **risk_snapshot})
        prediction_snapshot = latest_prediction_by_symbol(latest_predictions, member.symbol)
        intents.append(build_latest_intent(signals, risk_snapshot, prediction_snapshot, system_state, config, member.symbol))
    return intents, pd.DataFrame(latest_signal_rows), pd.DataFrame(latest_risk_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-only order intents.")
    parser.add_argument("--signals", default="tsm_price_rule_output/tsm_daily_algorithmic_signals.csv")
    parser.add_argument("--risk", default="tsm_price_rule_output/tsm_latest_risk_snapshot.csv")
    parser.add_argument("--prediction", default="tsm_price_rule_output/tsm_latest_prediction_snapshot.csv")
    parser.add_argument("--system-state", default="tsm_price_rule_output/tsm_latest_system_state.csv")
    parser.add_argument("--ledger", default="tsm_price_rule_output/tsm_order_intents.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--symbol", default="TSM")
    parser.add_argument("--universe-config", default="")
    parser.add_argument("--latest-predictions", default="tsm_price_rule_output/tsm_universe_latest_predictions.csv")
    parser.add_argument("--signals-root", default="tsm_price_rule_output")
    parser.add_argument("--config", default="config/tsm_research.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = load_run_config(args.config).paper_oms
    system_state = read_snapshot(Path(args.system_state))
    ledger = read_existing_intents(Path(args.ledger))
    if str(args.universe_config).strip():
        members = load_universe_members(args.universe_config, paper_only=True)
        decision_symbols = {member.symbol.upper() for member in members}
        if not ledger.empty and "symbol" in ledger.columns:
            ledger = ledger[ledger["symbol"].astype(str).str.upper().isin(decision_symbols)].copy()
        signals_root = Path(args.signals_root)
        root_signals = signals_root / "tsm_daily_algorithmic_signals.csv" if signals_root.is_dir() or signals_root.suffix == "" else signals_root
        latest_rows, latest_signals, latest_risk = build_batch_intents(
            members,
            root_signals,
            Path(args.risk),
            Path(args.latest_predictions),
            system_state,
            config,
        )
        latest = latest_rows[-1] if latest_rows else {}
        intents = upsert_intents(ledger, latest_rows)
        if not intents.empty and "symbol" in intents.columns:
            intents = intents[intents["symbol"].astype(str).str.upper().isin(decision_symbols)].copy()
        if not latest_signals.empty:
            latest_signals.to_csv(outdir / "tsm_universe_latest_signals.csv", index=False)
        if not latest_risk.empty:
            latest_risk.to_csv(outdir / "tsm_universe_latest_risk.csv", index=False)
    else:
        signals = load_signals(Path(args.signals))
        risk_snapshot = read_snapshot(Path(args.risk))
        prediction_snapshot = read_snapshot(Path(args.prediction))
        latest = build_latest_intent(signals, risk_snapshot, prediction_snapshot, system_state, config, str(args.symbol).upper())
        intents = upsert_intent(ledger, latest)
    quality = build_quality_checks(intents, latest)

    intents.to_csv(outdir / "tsm_order_intents.csv", index=False)
    quality.to_csv(outdir / "tsm_order_intent_quality_checks.csv", index=False)
    write_report(outdir, latest, quality)
    print("completed: order intents =", (outdir / "tsm_order_intents.csv").resolve(), "rows=", len(intents))
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()

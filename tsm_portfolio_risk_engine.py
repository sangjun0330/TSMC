#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply broker-free portfolio risk gates to paper order intents.

This engine does not submit or simulate orders. It caps or rejects candidate
paper intents and emits a risk decision ledger for the paper execution engine.

Outputs:
- tsm_portfolio_risk_order_decisions.csv
- tsm_portfolio_risk_snapshot.csv
- tsm_portfolio_risk_checks.csv
- tsm_portfolio_risk_report.md
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tsm_core.config import PortfolioConstructionConfig, PortfolioRiskConfig, load_run_config
from tsm_core.execution import ExecutionBlockReason, IntentStatus
from tsm_core.io import as_float, check_row, strip_bom_columns
from tsm_core.universe import load_decision_universe_members


V2_ACTIONABLE_TIERS = {"STRICT_ENTRY_ALLOWED", "AGGRESSIVE_TREND_ENTRY", "BREAKOUT_EXTENSION_TINY", "PULLBACK_REENTRY"}


DECISION_COLUMNS = [
    "intent_id",
    "symbol",
    "symbol_group",
    "asof_date",
    "input_status",
    "portfolio_status",
    "rank",
    "requested_weight",
    "approved_weight",
    "max_single_name_weight",
    "max_semiconductor_weight",
    "max_total_gross_weight",
    "max_group_weight",
    "allocation_policy",
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
    "paper_decision_score_20d",
    "p_success_20d",
    "expected_r_20d",
    "p_stop_hit_20d",
    "research_signal_score",
    "max_beta_to_spy",
    "max_beta_to_smh",
    "smh_beta_limit_mode",
    "beta_vs_spy_252d",
    "beta_vs_smh_252d",
    "beta_to_spy_warning",
    "beta_to_smh_warning",
    "beta_max_weight_multiplier",
    "order_adv_pct",
    "max_order_adv_pct",
    "intraday_coverage_score",
    "intraday_vol_bps",
    "intraday_max_weight_multiplier",
    "intraday_order_adv_cap_multiplier",
    "block_reason",
    "warning_reasons",
    "risk_policy_version",
    "checked_at_utc",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return strip_bom_columns(pd.read_csv(path))


def latest_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    if "asof_date" in df.columns:
        return df.sort_values("asof_date").iloc[-1]
    return df.iloc[-1]


def normalize_weight(value: object) -> float:
    parsed = as_float(value)
    if pd.isna(parsed):
        return 0.0
    return float(parsed / 100.0 if abs(parsed) > 1 else parsed)


def clean_symbol(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    text = str(value).strip().upper()
    return text if text and text != "NAN" else default


def latest_signal_context(signals: pd.DataFrame) -> dict[str, float]:
    if signals.empty:
        return {"beta_vs_spy_252d": np.nan, "beta_vs_smh_252d": np.nan, "dollar_volume_ma_20": np.nan}
    row = signals.sort_values("date").iloc[-1] if "date" in signals.columns else signals.iloc[-1]
    return {
        "beta_vs_spy_252d": as_float(row.get("beta_vs_spy_252d")),
        "beta_vs_smh_252d": as_float(row.get("beta_vs_smh_252d")),
        "dollar_volume_ma_20": as_float(row.get("dollar_volume_ma_20")),
        "timeframe_coverage_score": as_float(row.get("timeframe_coverage_score")),
        "m1_realized_range_pct": as_float(row.get("m1_realized_range_pct")),
        "execution_minute_realized_range_pct": as_float(row.get("execution_minute_realized_range_pct")),
    }


def build_decision(intent: pd.Series, context: dict[str, float], config: PortfolioRiskConfig) -> dict[str, object]:
    requested = normalize_weight(intent.get("target_weight"))
    status = str(intent.get("status", "MISSING"))
    beta_spy = context.get("beta_vs_spy_252d", np.nan)
    beta_smh = context.get("beta_vs_smh_252d", np.nan)
    adv = context.get("dollar_volume_ma_20", np.nan)
    block_reasons: list[str] = []
    warning_reasons: list[str] = []
    decision_tier = str(intent.get("decision_tier", ""))
    v2_actionable = decision_tier in V2_ACTIONABLE_TIERS and requested > 0 and status == IntentStatus.APPROVED.value
    intraday_coverage = as_float(context.get("timeframe_coverage_score"), np.nan)
    intraday_vol_bps = as_float(context.get("m1_realized_range_pct"), as_float(context.get("execution_minute_realized_range_pct"), np.nan))
    intraday_vol_bps = float(intraday_vol_bps * 10000.0) if pd.notna(intraday_vol_bps) else np.nan
    intraday_weight_multiplier = 1.0
    intraday_adv_multiplier = 1.0
    beta_weight_multiplier = 1.0
    if pd.notna(intraday_coverage) and intraday_coverage < 0.34:
        intraday_weight_multiplier *= 0.75
        intraday_adv_multiplier *= 0.75
        warning_reasons.append("INTRADAY_COVERAGE_LOW")
    if pd.notna(intraday_vol_bps) and intraday_vol_bps > 250.0:
        intraday_weight_multiplier *= 0.75
        intraday_adv_multiplier *= 0.75
        warning_reasons.append("INTRADAY_VOL_HIGH")
    max_weight = min(float(config.max_single_name_weight), requested)
    max_weight *= intraday_weight_multiplier
    max_notional = as_float(intent.get("max_notional"), 0.0)
    order_adv_pct = max_notional / adv if pd.notna(adv) and adv > 0 else 0.0
    max_order_adv_pct = float(config.max_order_adv_pct) * intraday_adv_multiplier
    smh_beta_limit_mode = str(getattr(config, "smh_beta_limit_mode", "WARN") or "WARN").strip().upper()

    if status != IntentStatus.APPROVED.value:
        block_reasons.append(str(intent.get("reason", "INTENT_NOT_APPROVED")))
    if requested <= 0:
        block_reasons.append("REQUESTED_WEIGHT_ZERO")
    if pd.notna(beta_spy) and abs(beta_spy) > float(config.max_beta_to_spy):
        if v2_actionable:
            beta_weight_multiplier *= 0.75
            warning_reasons.append("BETA_TO_SPY_WARN")
        else:
            block_reasons.append("BETA_TO_SPY_LIMIT")
    if pd.notna(beta_smh) and abs(beta_smh) > float(config.max_beta_to_smh):
        if smh_beta_limit_mode in {"BLOCK", "HARD", "HARD_BLOCK", "REJECT"}:
            block_reasons.append("BETA_TO_SMH_LIMIT")
        else:
            warning_reasons.append("BETA_TO_SMH_WARN")
    if order_adv_pct > max_order_adv_pct:
        block_reasons.append(ExecutionBlockReason.ORDER_CAPACITY_EXCEEDED.value)

    max_weight *= beta_weight_multiplier
    approved = max_weight if not block_reasons else 0.0
    portfolio_status = IntentStatus.APPROVED.value if approved > 0 else IntentStatus.REJECTED.value
    return {
        "intent_id": intent.get("intent_id", ""),
        "symbol": clean_symbol(intent.get("symbol")),
        "symbol_group": intent.get("symbol_group", "semiconductor"),
        "asof_date": intent.get("asof_date", ""),
        "input_status": status,
        "portfolio_status": portfolio_status,
        "rank": np.nan,
        "requested_weight": requested,
        "approved_weight": approved,
        "max_single_name_weight": float(config.max_single_name_weight),
        "max_semiconductor_weight": float(config.max_semiconductor_weight),
        "max_total_gross_weight": np.nan,
        "max_group_weight": np.nan,
        "allocation_policy": "per_symbol_cap",
        "raw_entry_event": intent.get("raw_entry_event", "NONE"),
        "decision_tier": intent.get("decision_tier", "LEGACY_ONLY"),
        "sizing_tier": intent.get("sizing_tier", "LEGACY_ONLY"),
        "suggested_action": intent.get("suggested_action", "LEGACY_ONLY"),
        "suggested_weight": normalize_weight(intent.get("suggested_weight")),
        "semi_momentum_regime": intent.get("semi_momentum_regime", "UNKNOWN"),
        "stop_price_1_8atr": as_float(intent.get("stop_price_1_8atr"), np.nan),
        "invalidation_5d_low": as_float(intent.get("invalidation_5d_low"), np.nan),
        "invalidation_ema10": as_float(intent.get("invalidation_ema10"), np.nan),
        "next_check_condition": intent.get("next_check_condition", ""),
        "paper_decision_score_20d": as_float(intent.get("paper_decision_score_20d"), np.nan),
        "p_success_20d": as_float(intent.get("p_success_20d"), np.nan),
        "expected_r_20d": as_float(intent.get("expected_r_20d"), np.nan),
        "p_stop_hit_20d": as_float(intent.get("p_stop_hit_20d"), np.nan),
        "research_signal_score": as_float(intent.get("research_signal_score"), np.nan),
        "max_beta_to_spy": float(config.max_beta_to_spy),
        "max_beta_to_smh": float(config.max_beta_to_smh),
        "smh_beta_limit_mode": smh_beta_limit_mode,
        "beta_vs_spy_252d": beta_spy,
        "beta_vs_smh_252d": beta_smh,
        "beta_to_spy_warning": "BETA_TO_SPY_WARN" in warning_reasons,
        "beta_to_smh_warning": "BETA_TO_SMH_WARN" in warning_reasons,
        "beta_max_weight_multiplier": beta_weight_multiplier,
        "order_adv_pct": order_adv_pct,
        "max_order_adv_pct": max_order_adv_pct,
        "intraday_coverage_score": intraday_coverage,
        "intraday_vol_bps": intraday_vol_bps,
        "intraday_max_weight_multiplier": intraday_weight_multiplier,
        "intraday_order_adv_cap_multiplier": intraday_adv_multiplier,
        "block_reason": "|".join(block_reasons) if block_reasons else "PASS",
        "warning_reasons": "|".join(warning_reasons) if warning_reasons else "PASS",
        "risk_policy_version": "portfolio_risk_v1",
        "checked_at_utc": now_utc_iso(),
    }


def _ranking_frame(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return decisions.copy()
    out = decisions.copy()
    for col in ["paper_decision_score_20d", "p_success_20d", "expected_r_20d", "research_signal_score", "p_stop_hit_20d"]:
        out[col] = pd.to_numeric(out.get(col, pd.Series(np.nan, index=out.index)), errors="coerce")
    out["_rank_score"] = out["paper_decision_score_20d"].fillna(-np.inf)
    out["_rank_p_success"] = out["p_success_20d"].fillna(-np.inf)
    out["_rank_expected_r"] = out["expected_r_20d"].fillna(-np.inf)
    out["_rank_research_score"] = out["research_signal_score"].fillna(-np.inf)
    out["_rank_stop"] = out["p_stop_hit_20d"].fillna(np.inf)
    return out.sort_values(
        ["_rank_score", "_rank_p_success", "_rank_expected_r", "_rank_research_score", "_rank_stop", "symbol"],
        ascending=[False, False, False, False, True, True],
    ).drop(columns=["_rank_score", "_rank_p_success", "_rank_expected_r", "_rank_research_score", "_rank_stop"])


def build_portfolio_decisions(
    intents: pd.DataFrame,
    contexts: dict[str, dict[str, float]],
    risk_config: PortfolioRiskConfig,
    construction: PortfolioConstructionConfig,
) -> pd.DataFrame:
    if intents.empty:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    base_rows = []
    for _, intent in intents.iterrows():
        symbol = clean_symbol(intent.get("symbol"))
        base_rows.append(build_decision(intent, contexts.get(symbol, {}), risk_config))
    base = pd.DataFrame(base_rows)
    if base.empty:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    candidates = _ranking_frame(base[base["portfolio_status"].astype(str).eq(IntentStatus.APPROVED.value)].copy())
    rejected = base[~base["portfolio_status"].astype(str).eq(IntentStatus.APPROVED.value)].copy()
    approved_rows: list[dict[str, object]] = []
    gross_used = 0.0
    group_used: dict[str, float] = {}
    rank = 1
    for _, row in candidates.iterrows():
        risk_capped_weight = normalize_weight(row.get("approved_weight"))
        group = str(row.get("symbol_group", "semiconductor"))
        remaining_gross = max(0.0, float(construction.max_total_gross_weight) - gross_used)
        remaining_group = max(0.0, float(construction.max_group_weight) - group_used.get(group, 0.0))
        remaining_positions = rank <= int(construction.max_open_positions)
        approved = min(risk_capped_weight, float(construction.max_single_name_weight), remaining_group, remaining_gross)
        row = row.to_dict()
        row.update(
            {
                "rank": rank,
                "max_single_name_weight": float(construction.max_single_name_weight),
                "max_total_gross_weight": float(construction.max_total_gross_weight),
                "max_group_weight": float(construction.max_group_weight),
                "allocation_policy": str(construction.allocation_policy),
            }
        )
        if not remaining_positions:
            row["portfolio_status"] = IntentStatus.REJECTED.value
            row["approved_weight"] = 0.0
            row["block_reason"] = "MAX_OPEN_POSITIONS_EXCEEDED"
        elif approved <= 0:
            row["portfolio_status"] = IntentStatus.REJECTED.value
            row["approved_weight"] = 0.0
            row["block_reason"] = "PORTFOLIO_CAP_EXHAUSTED"
        else:
            row["approved_weight"] = float(approved)
            row["block_reason"] = "PASS"
            gross_used += float(approved)
            group_used[group] = group_used.get(group, 0.0) + float(approved)
        approved_rows.append(row)
        rank += 1
    if not rejected.empty:
        rejected = rejected.copy()
        rejected["rank"] = np.nan
        rejected["max_total_gross_weight"] = float(construction.max_total_gross_weight)
        rejected["max_group_weight"] = float(construction.max_group_weight)
        rejected["allocation_policy"] = str(construction.allocation_policy)
    out = pd.concat([pd.DataFrame(approved_rows), rejected], ignore_index=True) if approved_rows or not rejected.empty else pd.DataFrame(columns=DECISION_COLUMNS)
    for col in DECISION_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[DECISION_COLUMNS].copy()


def contexts_from_latest_signals(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    if frame.empty:
        return {}
    contexts: dict[str, dict[str, float]] = {}
    work = frame.copy()
    if "symbol" not in work.columns:
        work["symbol"] = "TSM"
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.sort_values(["symbol", "date"])
    for symbol, group in work.groupby(work["symbol"].astype(str).str.upper(), dropna=False):
        contexts[str(symbol)] = latest_signal_context(group)
    return contexts


def merge_intraday_contexts(contexts: dict[str, dict[str, float]], path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return contexts
    try:
        frame = strip_bom_columns(pd.read_csv(path, parse_dates=["date"], low_memory=False))
    except Exception:
        return contexts
    if frame.empty or "symbol" not in frame.columns:
        return contexts
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values(["symbol", "date"])
    wanted = [
        "timeframe_coverage_score",
        "m1_realized_range_pct",
        "execution_minute_realized_range_pct",
        "intraday_feature_freshness_minutes",
    ]
    out = {symbol: dict(values) for symbol, values in contexts.items()}
    for symbol, group in frame.groupby("symbol", dropna=False):
        latest = group.iloc[-1]
        context = out.setdefault(str(symbol), {})
        for col in wanted:
            if col in latest.index:
                context[col] = as_float(latest.get(col), np.nan)
    return out


def filter_decision_universe(frame: pd.DataFrame, decision_symbols: set[str]) -> pd.DataFrame:
    if frame.empty or not decision_symbols or "symbol" not in frame.columns:
        return frame.copy()
    return frame[frame["symbol"].astype(str).str.upper().isin(decision_symbols)].copy()


def build_snapshot(decisions: pd.DataFrame) -> pd.DataFrame:
    approved = decisions[decisions.get("portfolio_status", pd.Series(dtype=str)).astype(str).eq(IntentStatus.APPROVED.value)] if not decisions.empty else pd.DataFrame()
    latest = latest_row(decisions)
    total_approved = float(pd.to_numeric(approved.get("approved_weight", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not approved.empty else 0.0
    total_requested = float(pd.to_numeric(decisions.get("requested_weight", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not decisions.empty else 0.0
    status = IntentStatus.APPROVED.value if total_approved > 0 else latest.get("portfolio_status", "MISSING")
    rows = [
        {"field": "portfolio_risk_status", "value": status},
        {"field": "portfolio_risk_asof_date", "value": latest.get("asof_date", "")},
        {"field": "portfolio_approved_weight", "value": total_approved},
        {"field": "portfolio_requested_weight", "value": total_requested},
        {"field": "portfolio_approved_count", "value": int(len(approved))},
        {"field": "portfolio_candidate_count", "value": int(len(decisions))},
        {"field": "portfolio_risk_block_reason", "value": latest.get("block_reason", "MISSING")},
        {"field": "portfolio_risk_warning_reasons", "value": latest.get("warning_reasons", "PASS")},
        {"field": "paper_live_trading_status", "value": "DISABLED_BY_DESIGN"},
        {"field": "generated_at_utc", "value": now_utc_iso()},
    ]
    return pd.DataFrame(rows)


def build_checks(decisions: pd.DataFrame) -> pd.DataFrame:
    latest = latest_row(decisions)
    rows = [
        check_row("portfolio_risk_decision_rows_positive", not decisions.empty, "CRITICAL", len(decisions)),
        check_row("portfolio_risk_weight_cap_applied", normalize_weight(latest.get("approved_weight")) <= normalize_weight(latest.get("max_single_name_weight", 0.0)) + 1e-12, "CRITICAL", latest.get("approved_weight", 0.0)),
        check_row("portfolio_risk_no_live_order_submission", True, "CRITICAL", "paper_only"),
        check_row("portfolio_risk_no_duplicate_intent_id", not decisions.get("intent_id", pd.Series(dtype=str)).duplicated().any(), "CRITICAL", int(decisions.get("intent_id", pd.Series(dtype=str)).duplicated().sum())),
    ]
    return pd.DataFrame(rows)


def write_report(outdir: Path, snapshot: pd.DataFrame, checks: pd.DataFrame) -> None:
    snap = dict(zip(snapshot["field"], snapshot["value"]))
    lines = [
        "# Top10 Portfolio Risk Report",
        "",
        f"- Status: {snap.get('portfolio_risk_status', 'NA')}",
        f"- Approved weight: {normalize_weight(snap.get('portfolio_approved_weight')) * 100:.2f}%",
        f"- Block reason: {snap.get('portfolio_risk_block_reason', 'NA')}",
        f"- Warning reasons: {snap.get('portfolio_risk_warning_reasons', 'PASS')}",
        f"- Live trading status: {snap.get('paper_live_trading_status', 'DISABLED_BY_DESIGN')}",
        "",
        "## Checks",
        "",
        "| Check | Passed | Value |",
        "|---|---:|---:|",
    ]
    for _, row in checks.iterrows():
        lines.append(f"| {row['check']} | {row['passed']} | {row['value']} |")
    lines.extend(["", "This report applies paper-only portfolio gates."])
    (outdir / "tsm_portfolio_risk_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply portfolio risk gates to paper order intents.")
    parser.add_argument("--intents", default="tsm_price_rule_output/tsm_order_intents.csv")
    parser.add_argument("--signals", default="tsm_price_rule_output/tsm_daily_algorithmic_signals.csv")
    parser.add_argument("--latest-signals", default="tsm_price_rule_output/tsm_universe_latest_signals.csv")
    parser.add_argument("--intraday-features", default="tsm_price_rule_output/tsm_intraday_daily_features.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--config", default="config/tsm_research.toml")
    parser.add_argument("--decision-universe-config", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_config = load_run_config(args.config)
    config = run_config.portfolio_risk
    construction = run_config.portfolio_construction
    intents = read_csv_if_exists(Path(args.intents))
    decision_config = str(args.decision_universe_config or run_config.universe.decision_universe_config).strip()
    decision_symbols: set[str] = set()
    if decision_config:
        decision_symbols = {member.symbol.upper() for member in load_decision_universe_members(decision_config, paper_only=True, require_count=False)}
    intents = filter_decision_universe(intents, decision_symbols)
    latest_signals = read_csv_if_exists(Path(args.latest_signals))
    contexts = contexts_from_latest_signals(latest_signals)
    contexts = merge_intraday_contexts(contexts, Path(args.intraday_features))
    if not contexts:
        signals = read_csv_if_exists(Path(args.signals))
        fallback_symbol = "TSM"
        if not intents.empty and "symbol" in intents.columns:
            fallback_symbol = clean_symbol(latest_row(intents).get("symbol"), fallback_symbol)
        contexts = {fallback_symbol: latest_signal_context(signals)}
    decisions = build_portfolio_decisions(intents, contexts, config, construction) if not intents.empty else pd.DataFrame(
        [
            {
                "portfolio_status": IntentStatus.REJECTED.value,
                "block_reason": "MISSING_INTENT",
                "approved_weight": 0.0,
                "requested_weight": 0.0,
                "checked_at_utc": now_utc_iso(),
            }
        ]
    )
    for col in DECISION_COLUMNS:
        if col not in decisions.columns:
            decisions[col] = np.nan
    decisions = decisions[DECISION_COLUMNS].copy()
    snapshot = build_snapshot(decisions)
    checks = build_checks(decisions)
    if decision_symbols:
        non_decision = (
            int((~decisions["symbol"].astype(str).str.upper().isin(decision_symbols)).sum())
            if not decisions.empty and "symbol" in decisions.columns
            else 0
        )
        checks = pd.concat(
            [checks, pd.DataFrame([check_row("portfolio_risk_decision_universe_top10_only", non_decision == 0, "CRITICAL", non_decision, "0")])],
            ignore_index=True,
        )
    decisions.to_csv(outdir / "tsm_portfolio_risk_order_decisions.csv", index=False)
    decisions[decisions["portfolio_status"].astype(str).eq(IntentStatus.APPROVED.value)].to_csv(outdir / "tsm_portfolio_targets.csv", index=False)
    snapshot.to_csv(outdir / "tsm_portfolio_risk_snapshot.csv", index=False)
    snapshot.to_csv(outdir / "tsm_portfolio_snapshot.csv", index=False)
    checks.to_csv(outdir / "tsm_portfolio_risk_checks.csv", index=False)
    write_report(outdir, snapshot, checks)
    print("completed: portfolio risk outputs =", outdir.resolve())
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()

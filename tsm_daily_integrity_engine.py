#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily-data integrity audit for the TSM research system.

This is a contract checker. It verifies that the daily-data outputs are
internally consistent before the system state/reporting layers rely on them.

Outputs:
- tsm_daily_integrity_checks.csv
- tsm_latest_integrity_snapshot.csv
- tsm_daily_integrity_report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return strip_bom_columns(pd.read_csv(path, **kwargs))


def check_row(check: str, passed: bool, severity: str, value, tolerance="", details="") -> Dict:
    return {
        "check": check,
        "passed": bool(passed),
        "severity": severity,
        "value": value,
        "tolerance": tolerance,
        "details": details,
    }


def max_abs_diff(a: pd.Series, b: pd.Series) -> float:
    diff = (pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")).abs()
    return float(diff.max()) if not diff.dropna().empty else np.nan


def ohlc_checks(df: pd.DataFrame, label: str) -> List[Dict]:
    rows = []
    if df.empty:
        rows.append(check_row(f"{label}_exists", False, "CRITICAL", "missing", "", "CSV is empty or missing"))
        return rows

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    rows.append(check_row(f"{label}_required_columns", not missing, "CRITICAL", ",".join(missing) if missing else "ok"))
    if missing:
        return rows

    rows.append(check_row(f"{label}_rows_positive", len(df) > 0, "CRITICAL", len(df)))
    rows.append(check_row(f"{label}_dates_sorted", df["date"].is_monotonic_increasing, "CRITICAL", "ascending"))
    rows.append(check_row(f"{label}_dates_unique", not df["date"].duplicated().any(), "CRITICAL", int(df["date"].duplicated().sum())))

    price_cols = ["open", "high", "low", "close"]
    prices = df[price_cols].apply(pd.to_numeric, errors="coerce")
    rows.append(check_row(f"{label}_no_missing_ohlc", not prices.isna().any().any(), "CRITICAL", int(prices.isna().sum().sum())))
    rows.append(check_row(f"{label}_positive_prices", bool((prices > 0).all().all()), "CRITICAL", int((prices <= 0).sum().sum())))

    high_ok = prices["high"] >= prices[["open", "low", "close"]].max(axis=1)
    low_ok = prices["low"] <= prices[["open", "high", "close"]].min(axis=1)
    rows.append(check_row(f"{label}_high_low_consistent", bool((high_ok & low_ok).all()), "CRITICAL", int((~(high_ok & low_ok)).sum())))
    volume = pd.to_numeric(df["volume"], errors="coerce")
    rows.append(check_row(f"{label}_volume_nonnegative", bool((volume.fillna(-1) >= 0).all()), "CRITICAL", int((volume.fillna(-1) < 0).sum())))
    return rows


def date_alignment_checks(raw: pd.DataFrame, enriched: pd.DataFrame, signals: pd.DataFrame, risk: pd.DataFrame) -> List[Dict]:
    rows = []
    frames = {"raw": raw, "enriched": enriched, "signals": signals, "risk": risk}
    nonempty = {name: df for name, df in frames.items() if not df.empty and "date" in df.columns}
    if len(nonempty) < 2:
        rows.append(check_row("date_alignment_available", False, "CRITICAL", len(nonempty), "", "Need at least two dated outputs"))
        return rows

    base_name, base_df = next(iter(nonempty.items()))
    base_dates = set(pd.to_datetime(base_df["date"]).dt.normalize())
    for name, df in nonempty.items():
        dates = set(pd.to_datetime(df["date"]).dt.normalize())
        missing_from_name = len(base_dates - dates)
        extra_in_name = len(dates - base_dates)
        rows.append(
            check_row(
                f"date_alignment_{base_name}_vs_{name}",
                missing_from_name == 0 and extra_in_name == 0,
                "CRITICAL",
                f"missing={missing_from_name}, extra={extra_in_name}",
            )
        )
    return rows


def latest_feature_checks(signals: pd.DataFrame) -> List[Dict]:
    rows = []
    if signals.empty:
        return [check_row("latest_signal_features_available", False, "CRITICAL", "missing")]
    required_latest = [
        "close",
        "sma_20",
        "sma_50",
        "sma_200",
        "atr_14",
        "atr_14_pct",
        "vol_20d_ann",
        "score_price_algo_total",
        "entry_trigger",
        "trade_action",
        "risk_pct_2atr",
    ]
    latest = signals.iloc[-1]
    missing_cols = [c for c in required_latest if c not in signals.columns]
    rows.append(check_row("latest_required_signal_columns", not missing_cols, "CRITICAL", ",".join(missing_cols) if missing_cols else "ok"))
    if missing_cols:
        return rows
    missing_values = [c for c in required_latest if pd.isna(latest[c])]
    rows.append(check_row("latest_required_signal_values_not_null", not missing_values, "CRITICAL", ",".join(missing_values) if missing_values else "ok"))
    return rows


def formula_checks(signals: pd.DataFrame) -> List[Dict]:
    rows = []
    if signals.empty:
        return [check_row("signal_formula_checks_available", False, "CRITICAL", "missing")]
    required = ["close", "atr_14", "risk_pct_2atr", "atr_stop_2x", "atr_trailing_stop_3x", "take_profit_2R", "take_profit_3R"]
    missing = [c for c in required if c not in signals.columns]
    rows.append(check_row("signal_formula_columns", not missing, "CRITICAL", ",".join(missing) if missing else "ok"))
    if missing:
        return rows

    tol = 1e-8
    close = pd.to_numeric(signals["close"], errors="coerce")
    atr = pd.to_numeric(signals["atr_14"], errors="coerce")
    rows.append(check_row("formula_risk_pct_2atr", max_abs_diff(signals["risk_pct_2atr"], 2 * atr / close) <= tol, "CRITICAL", max_abs_diff(signals["risk_pct_2atr"], 2 * atr / close), tol))
    rows.append(check_row("formula_atr_stop_2x", max_abs_diff(signals["atr_stop_2x"], close - 2 * atr) <= tol, "CRITICAL", max_abs_diff(signals["atr_stop_2x"], close - 2 * atr), tol))
    rows.append(check_row("formula_atr_trailing_stop_3x", max_abs_diff(signals["atr_trailing_stop_3x"], close - 3 * atr) <= tol, "CRITICAL", max_abs_diff(signals["atr_trailing_stop_3x"], close - 3 * atr), tol))
    rows.append(check_row("formula_take_profit_2R", max_abs_diff(signals["take_profit_2R"], close + 4 * atr) <= tol, "CRITICAL", max_abs_diff(signals["take_profit_2R"], close + 4 * atr), tol))
    rows.append(check_row("formula_take_profit_3R", max_abs_diff(signals["take_profit_3R"], close + 6 * atr) <= tol, "CRITICAL", max_abs_diff(signals["take_profit_3R"], close + 6 * atr), tol))
    return rows


def enriched_indicator_checks(enriched: pd.DataFrame) -> List[Dict]:
    rows = []
    if enriched.empty:
        return [check_row("enriched_indicator_checks_available", False, "CRITICAL", "missing")]
    required = ["date", "high", "close", "breakout_20d_high", "breakout_60d_high", "atr_14", "atr_14_pct"]
    missing = [c for c in required if c not in enriched.columns]
    rows.append(check_row("enriched_indicator_columns", not missing, "CRITICAL", ",".join(missing) if missing else "ok"))
    if missing:
        return rows

    close = pd.to_numeric(enriched["close"], errors="coerce")
    high = pd.to_numeric(enriched["high"], errors="coerce")
    prev20 = high.shift(1).rolling(20, min_periods=10).max()
    prev60 = high.shift(1).rolling(60, min_periods=30).max()
    calc20 = close > prev20
    calc60 = close > prev60
    stored20 = enriched["breakout_20d_high"].astype(str).str.lower().isin(["true", "1", "yes"])
    stored60 = enriched["breakout_60d_high"].astype(str).str.lower().isin(["true", "1", "yes"])
    valid20 = prev20.notna()
    valid60 = prev60.notna()
    mismatch20 = int((calc20[valid20] != stored20[valid20]).sum())
    mismatch60 = int((calc60[valid60] != stored60[valid60]).sum())
    rows.append(check_row("breakout_20d_recomputed_match", mismatch20 == 0, "CRITICAL", mismatch20))
    rows.append(check_row("breakout_60d_recomputed_match", mismatch60 == 0, "CRITICAL", mismatch60))
    rows.append(check_row("formula_atr_14_pct_enriched", max_abs_diff(enriched["atr_14_pct"], enriched["atr_14"] / close) <= 1e-10, "CRITICAL", max_abs_diff(enriched["atr_14_pct"], enriched["atr_14"] / close), "1e-10"))
    return rows


def trade_and_curve_checks(trades: pd.DataFrame, curves: pd.DataFrame, signals: pd.DataFrame) -> List[Dict]:
    rows = []
    if trades.empty:
        rows.append(check_row("trade_log_non_empty", False, "WARN", 0, "", "Some strategies may legitimately have zero trades, but current baseline is expected to trade."))
    else:
        trades = trades.copy()
        for col in ["entry_signal_date", "entry_date", "exit_date"]:
            trades[col] = pd.to_datetime(trades[col])
        rows.append(check_row("trade_entry_after_signal", bool((trades["entry_date"] > trades["entry_signal_date"]).all()), "CRITICAL", "all"))
        rows.append(check_row("trade_exit_not_before_entry", bool((trades["exit_date"] >= trades["entry_date"]).all()), "CRITICAL", "all"))
        rows.append(check_row("trade_net_return_not_above_gross", bool((trades["net_return_pct"] <= trades["gross_return_pct"] + 1e-9).all()), "CRITICAL", "all"))
        overlap_trades = trades[trades["trade_event"].eq("AGGREGATE_EXIT")].copy() if "trade_event" in trades.columns else trades
        overlap_count = 0
        for _, group in overlap_trades.sort_values(["strategy_id", "entry_date"]).groupby("strategy_id"):
            previous_exit = None
            for _, row in group.iterrows():
                if previous_exit is not None and row["entry_date"] < previous_exit:
                    overlap_count += 1
                previous_exit = row["exit_date"]
        rows.append(check_row("trade_no_overlapping_positions", overlap_count == 0, "CRITICAL", overlap_count))

    if curves.empty:
        rows.append(check_row("equity_curves_non_empty", False, "CRITICAL", 0))
    else:
        curves = curves.copy()
        rows.append(check_row("equity_positive", bool((pd.to_numeric(curves["equity"], errors="coerce") > 0).all()), "CRITICAL", "all"))
        if not signals.empty:
            expected = len(signals)
            per_strategy_counts = curves.groupby("strategy_id")["date"].count()
            rows.append(check_row("equity_curve_rows_per_strategy_match_signals", bool((per_strategy_counts == expected).all()), "CRITICAL", per_strategy_counts.to_dict(), f"expected={expected}"))
    return rows


def risk_checks(risk: pd.DataFrame, signals: pd.DataFrame) -> List[Dict]:
    rows = []
    if risk.empty:
        return [check_row("risk_policy_available", False, "CRITICAL", "missing")]
    required = ["date", "final_recommended_max_weight", "account_risk_limit_weight", "risk_state"]
    missing = [c for c in required if c not in risk.columns]
    rows.append(check_row("risk_policy_columns", not missing, "CRITICAL", ",".join(missing) if missing else "ok"))
    if missing:
        return rows
    weights = pd.to_numeric(risk["final_recommended_max_weight"], errors="coerce")
    account = pd.to_numeric(risk["account_risk_limit_weight"], errors="coerce")
    rows.append(check_row("risk_final_weight_between_0_and_1", bool(weights.between(0, 1).all()), "CRITICAL", f"min={weights.min()}, max={weights.max()}"))
    rows.append(check_row("risk_account_limit_no_nan_after_warmup", bool(account.iloc[260:].notna().all()), "CRITICAL", int(account.iloc[260:].isna().sum()), "after 260 rows"))
    if not signals.empty:
        rows.append(check_row("risk_latest_date_matches_signals", pd.to_datetime(risk["date"]).max() == pd.to_datetime(signals["date"]).max(), "CRITICAL", f"risk={pd.to_datetime(risk['date']).max()}, signals={pd.to_datetime(signals['date']).max()}"))
    return rows


def snapshot(rows: List[Dict]) -> pd.DataFrame:
    checks = pd.DataFrame(rows)
    critical = checks[checks["severity"] == "CRITICAL"]
    failed = checks[~checks["passed"]]
    critical_failed = critical[~critical["passed"]]
    return pd.DataFrame(
        [
            {"field": "all_checks_passed", "value": bool(failed.empty)},
            {"field": "all_critical_checks_passed", "value": bool(critical_failed.empty)},
            {"field": "n_checks", "value": len(checks)},
            {"field": "n_failed", "value": len(failed)},
            {"field": "n_critical_failed", "value": len(critical_failed)},
        ]
    )


def write_report(outdir: Path, checks: pd.DataFrame, snap: pd.DataFrame) -> None:
    snap_map = dict(zip(snap["field"], snap["value"]))
    failed = checks[~checks["passed"]]
    lines = [
        "# Top10 Daily Integrity Report",
        "",
        f"- All checks passed: {snap_map.get('all_checks_passed')}",
        f"- All critical checks passed: {snap_map.get('all_critical_checks_passed')}",
        f"- Checks: {snap_map.get('n_checks')}",
        f"- Failed: {snap_map.get('n_failed')}",
        f"- Critical failed: {snap_map.get('n_critical_failed')}",
        "",
    ]
    if failed.empty:
        lines.append("No failed checks.")
    else:
        lines.extend(["## Failed Checks", "", "| Check | Severity | Value | Details |", "|---|---|---:|---|"])
        for _, row in failed.iterrows():
            lines.append(f"| {row['check']} | {row['severity']} | {row['value']} | {row['details']} |")

    lines.extend(["", "## All Checks", "", "| Check | Passed | Severity | Value |", "|---|---:|---|---:|"])
    for _, row in checks.iterrows():
        lines.append(f"| {row['check']} | {row['passed']} | {row['severity']} | {row['value']} |")

    (outdir / "tsm_daily_integrity_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit daily-data integrity for the TSM system.")
    parser.add_argument("--raw", default="output/tsm_daily_10y_raw.csv")
    parser.add_argument("--enriched", default="output/tsm_daily_10y_enriched.csv")
    parser.add_argument("--signals", default="tsm_price_rule_output/tsm_daily_algorithmic_signals.csv")
    parser.add_argument("--risk-policy", default="tsm_price_rule_output/tsm_risk_policy_daily.csv")
    parser.add_argument("--trade-log", default="tsm_price_rule_output/tsm_backtest_trade_log.csv")
    parser.add_argument("--equity-curves", default="tsm_price_rule_output/tsm_backtest_equity_curves.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = read_csv_if_exists(Path(args.raw), parse_dates=["date"])
    enriched = read_csv_if_exists(Path(args.enriched), parse_dates=["date"])
    signals = read_csv_if_exists(Path(args.signals), parse_dates=["date"])
    risk = read_csv_if_exists(Path(args.risk_policy), parse_dates=["date"])
    trades = read_csv_if_exists(Path(args.trade_log))
    curves = read_csv_if_exists(Path(args.equity_curves), parse_dates=["date"])

    rows: List[Dict] = []
    rows.extend(ohlc_checks(raw, "raw"))
    rows.extend(ohlc_checks(enriched, "enriched"))
    rows.extend(date_alignment_checks(raw, enriched, signals, risk))
    rows.extend(latest_feature_checks(signals))
    rows.extend(formula_checks(signals))
    rows.extend(enriched_indicator_checks(enriched))
    rows.extend(trade_and_curve_checks(trades, curves, signals))
    rows.extend(risk_checks(risk, signals))

    checks = pd.DataFrame(rows)
    snap = snapshot(rows)
    checks.to_csv(outdir / "tsm_daily_integrity_checks.csv", index=False)
    snap.to_csv(outdir / "tsm_latest_integrity_snapshot.csv", index=False)
    write_report(outdir, checks, snap)

    print("완료: integrity outputs =", outdir.resolve())
    print(snap.to_string(index=False))


if __name__ == "__main__":
    main()

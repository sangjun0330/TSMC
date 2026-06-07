#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily data-quality gate for the TSM research and paper-trading system.

This engine checks whether the daily OHLCV, enriched features, benchmark joins,
and rule-engine signal rows are safe enough for downstream research and
decision-support reporting. It does not place orders.

Outputs:
- tsm_data_quality_checks.csv
- tsm_data_quality_issues.csv
- tsm_latest_data_quality_snapshot.csv
- tsm_data_quality_report.md
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable
import pandas as pd

from tsm_core.io import check_row, require_columns, strip_bom_columns


OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
BENCHMARK_COLUMNS = [
    "spy_adj_close",
    "smh_adj_close",
    "qqq_adj_close",
    "relative_return_vs_spy_60d",
    "relative_return_vs_smh_60d",
    "relative_return_vs_qqq_60d",
]
LATEST_SIGNAL_COLUMNS = [
    "date",
    "close",
    "atr_14",
    "atr_14_pct",
    "score_price_algo_total",
    "entry_trigger",
    "trade_action",
    "risk_pct_2atr",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return strip_bom_columns(pd.read_csv(path, **kwargs))


def to_numeric_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return df[list(columns)].apply(pd.to_numeric, errors="coerce")


def bool_from_series(series: pd.Series) -> bool:
    return bool(series.fillna(False).all())


def has_required_columns(df: pd.DataFrame, required: Iterable[str]) -> tuple[bool, str]:
    missing = [c for c in required if c not in df.columns]
    return not missing, ",".join(missing) if missing else "ok"


def frame_checks(df: pd.DataFrame, label: str, path: Path) -> list[dict]:
    rows: list[dict] = []
    rows.append(check_row(f"{label}_file_exists", path.exists(), "CRITICAL", str(path)))
    if df.empty:
        rows.append(check_row(f"{label}_rows_positive", False, "CRITICAL", 0, details="CSV is missing or empty."))
        return rows

    ok, missing = has_required_columns(df, OHLCV_COLUMNS)
    rows.append(check_row(f"{label}_required_ohlcv_columns", ok, "CRITICAL", missing))
    if not ok:
        return rows

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    rows.append(check_row(f"{label}_dates_parseable", bool(parsed_dates.notna().all()), "CRITICAL", int(parsed_dates.isna().sum())))
    if parsed_dates.notna().any():
        rows.append(check_row(f"{label}_dates_sorted", bool(parsed_dates.is_monotonic_increasing), "CRITICAL", "ascending"))
        rows.append(check_row(f"{label}_dates_unique", not parsed_dates.duplicated().any(), "CRITICAL", int(parsed_dates.duplicated().sum())))
        max_gap = int(parsed_dates.sort_values().diff().dt.days.dropna().max()) if len(parsed_dates.dropna()) > 1 else 0
        rows.append(
            check_row(
                f"{label}_calendar_gap_not_extreme",
                max_gap <= 7,
                "WARN",
                max_gap,
                "<=7 calendar days",
                "Long gaps may be holidays or missing data; inspect if this fails.",
            )
        )

    prices = to_numeric_frame(df, ["open", "high", "low", "close"])
    rows.append(check_row(f"{label}_ohlc_numeric", not prices.isna().any().any(), "CRITICAL", int(prices.isna().sum().sum())))
    rows.append(check_row(f"{label}_prices_positive", bool((prices > 0).all().all()), "CRITICAL", int((prices <= 0).sum().sum())))
    high_ok = prices["high"] >= prices[["open", "low", "close"]].max(axis=1)
    low_ok = prices["low"] <= prices[["open", "high", "close"]].min(axis=1)
    rows.append(check_row(f"{label}_ohlc_bounds_valid", bool_from_series(high_ok & low_ok), "CRITICAL", int((~(high_ok & low_ok)).sum())))

    volume = pd.to_numeric(df["volume"], errors="coerce")
    rows.append(check_row(f"{label}_volume_numeric", bool(volume.notna().all()), "CRITICAL", int(volume.isna().sum())))
    rows.append(check_row(f"{label}_volume_nonnegative", bool((volume.fillna(-1) >= 0).all()), "CRITICAL", int((volume.fillna(-1) < 0).sum())))
    zero_volume = int((volume.fillna(0) == 0).sum())
    rows.append(check_row(f"{label}_zero_volume_absent", zero_volume == 0, "WARN", zero_volume))

    returns = prices["close"].pct_change()
    extreme_moves = int((returns.abs() > 0.25).sum())
    rows.append(check_row(f"{label}_single_day_return_outliers_under_25pct", extreme_moves == 0, "WARN", extreme_moves))
    return rows


def alignment_checks(raw: pd.DataFrame, enriched: pd.DataFrame, signals: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if raw.empty or enriched.empty or signals.empty:
        rows.append(check_row("date_alignment_inputs_available", False, "CRITICAL", "missing_input"))
        return rows
    for label, df in [("raw", raw), ("enriched", enriched), ("signals", signals)]:
        require_columns(df, ["date"], label)
    raw_dates = set(pd.to_datetime(raw["date"], errors="coerce").dt.normalize())
    enriched_dates = set(pd.to_datetime(enriched["date"], errors="coerce").dt.normalize())
    signal_dates = set(pd.to_datetime(signals["date"], errors="coerce").dt.normalize())
    rows.append(check_row("raw_vs_enriched_dates_match", raw_dates == enriched_dates, "CRITICAL", f"raw_only={len(raw_dates - enriched_dates)}, enriched_only={len(enriched_dates - raw_dates)}"))
    rows.append(check_row("enriched_vs_signals_dates_match", enriched_dates == signal_dates, "CRITICAL", f"enriched_only={len(enriched_dates - signal_dates)}, signals_only={len(signal_dates - enriched_dates)}"))

    common_cols = ["open", "high", "low", "close", "volume"]
    merged = raw[["date", *common_cols]].merge(enriched[["date", *common_cols]], on="date", suffixes=("_raw", "_enriched"))
    mismatches = 0
    max_abs = 0.0
    for col in common_cols:
        left = pd.to_numeric(merged[f"{col}_raw"], errors="coerce")
        right = pd.to_numeric(merged[f"{col}_enriched"], errors="coerce")
        diff = (left - right).abs()
        mismatches += int((diff > 1e-8).sum())
        max_abs = max(max_abs, float(diff.max()) if diff.notna().any() else 0.0)
    rows.append(check_row("raw_enriched_ohlcv_values_match", mismatches == 0, "CRITICAL", f"mismatches={mismatches}, max_abs={max_abs}"))
    return rows


def latest_checks(
    enriched: pd.DataFrame,
    signals: pd.DataFrame,
    run_date: str,
    max_stale_days: int,
    benchmark_mode: str,
) -> list[dict]:
    rows: list[dict] = []
    if enriched.empty or signals.empty:
        rows.append(check_row("latest_inputs_available", False, "CRITICAL", "missing_input"))
        return rows

    ok, missing = has_required_columns(signals, LATEST_SIGNAL_COLUMNS)
    rows.append(check_row("latest_signal_required_columns", ok, "CRITICAL", missing))
    if not ok:
        return rows

    run_ts = pd.Timestamp(run_date).normalize()
    signal_dates = pd.to_datetime(signals["date"], errors="coerce").dt.normalize()
    latest_signal_date = signal_dates.max()
    stale_days = int((run_ts - latest_signal_date).days) if pd.notna(latest_signal_date) else 9999
    rows.append(check_row("latest_signal_not_future", bool(latest_signal_date <= run_ts), "CRITICAL", latest_signal_date.date().isoformat() if pd.notna(latest_signal_date) else "NA", f"run_date={run_date}"))
    rows.append(check_row("latest_signal_within_stale_limit", stale_days <= max_stale_days, "CRITICAL", stale_days, f"<={max_stale_days} calendar days"))

    latest = signals.loc[signal_dates.eq(latest_signal_date)].iloc[-1]
    missing_values = [c for c in LATEST_SIGNAL_COLUMNS if pd.isna(latest[c])]
    rows.append(check_row("latest_signal_required_values_present", not missing_values, "CRITICAL", ",".join(missing_values) if missing_values else "ok"))

    if benchmark_mode == "disabled":
        present = [c for c in BENCHMARK_COLUMNS if c in enriched.columns]
        rows.append(
            check_row(
                "benchmark_join_columns_disabled",
                not present,
                "CRITICAL",
                ",".join(present) if present else "ok",
                details="Benchmarks are intentionally disabled for this run.",
            )
        )
        return rows

    ok, missing = has_required_columns(enriched, BENCHMARK_COLUMNS)
    severity = "CRITICAL" if benchmark_mode == "required" else "INFO"
    rows.append(check_row("benchmark_join_columns_present", ok, severity, missing))
    if ok:
        latest_enriched = enriched[pd.to_datetime(enriched["date"], errors="coerce").dt.normalize().eq(latest_signal_date)].tail(1)
        if latest_enriched.empty:
            rows.append(check_row("latest_benchmark_row_present", False, severity, latest_signal_date.date().isoformat()))
        else:
            latest_bench = latest_enriched.iloc[0]
            missing_bench = [c for c in BENCHMARK_COLUMNS if pd.isna(latest_bench[c])]
            rows.append(check_row("latest_benchmark_values_present", not missing_bench, severity, ",".join(missing_bench) if missing_bench else "ok"))
    return rows


def corporate_action_checks(raw: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if raw.empty or "adj_close" not in raw.columns or "close" not in raw.columns:
        rows.append(check_row("corporate_action_adjusted_close_available", False, "WARN", "missing"))
        return rows
    close = pd.to_numeric(raw["close"], errors="coerce")
    adj = pd.to_numeric(raw["adj_close"], errors="coerce")
    diff = (close - adj).abs()
    all_equal = bool((diff.fillna(0) <= 1e-10).all())
    rows.append(
        check_row(
            "corporate_action_adjustment_independent",
            not all_equal,
            "WARN",
            "adj_close_equals_close_for_all_rows" if all_equal else "adjusted_close_differs_somewhere",
            details="If the symbol had known splits or dividends in the sample, validate adjusted-close handling before decision support.",
        )
    )
    if "data_quality_note" in raw.columns:
        notes = sorted(set(raw["data_quality_note"].dropna().astype(str)))
        rows.append(check_row("source_data_quality_notes_available", bool(notes), "INFO", "|".join(notes[:5]) if notes else "none"))
    if "data_source" in raw.columns:
        sources = sorted(set(raw["data_source"].dropna().astype(str)))
        rows.append(check_row("source_vendor_recorded", bool(sources), "CRITICAL", "|".join(sources) if sources else "missing"))
    return rows


def determine_status(checks: pd.DataFrame) -> str:
    if checks.empty:
        return "FAIL"
    failed = checks[~checks["passed"].astype(bool)]
    if not failed[failed["severity"].eq("CRITICAL")].empty:
        return "FAIL"
    if not failed[failed["severity"].eq("WARN")].empty:
        return "WARN"
    return "PASS"


def build_snapshot(checks: pd.DataFrame, raw: pd.DataFrame, signals: pd.DataFrame, run_date: str) -> pd.DataFrame:
    status = determine_status(checks)
    failed = checks[~checks["passed"].astype(bool)] if not checks.empty else pd.DataFrame()
    critical_failed = failed[failed["severity"].eq("CRITICAL")] if not failed.empty else pd.DataFrame()
    warn_failed = failed[failed["severity"].eq("WARN")] if not failed.empty else pd.DataFrame()
    latest_date = "NA"
    if not signals.empty and "date" in signals.columns:
        parsed = pd.to_datetime(signals["date"], errors="coerce")
        latest = parsed.max()
        latest_date = latest.date().isoformat() if pd.notna(latest) else "NA"
    source = "NA"
    if not raw.empty and "data_source" in raw.columns:
        source = "|".join(sorted(set(raw["data_source"].dropna().astype(str))))
    rows = [
        {"field": "data_quality_status", "value": status},
        {"field": "decision_support_data_gate", "value": "BLOCK" if status == "FAIL" else ("PASS_WITH_WARNINGS" if status == "WARN" else "PASS")},
        {"field": "run_date", "value": run_date},
        {"field": "latest_signal_date", "value": latest_date},
        {"field": "data_source", "value": source},
        {"field": "checks", "value": len(checks)},
        {"field": "failed_checks", "value": len(failed)},
        {"field": "critical_failed_checks", "value": len(critical_failed)},
        {"field": "warn_failed_checks", "value": len(warn_failed)},
        {"field": "generated_at_utc", "value": now_utc_iso()},
        {"field": "block_reasons", "value": "|".join(critical_failed["check"].astype(str)) if not critical_failed.empty else "PASS"},
        {"field": "warning_reasons", "value": "|".join(warn_failed["check"].astype(str)) if not warn_failed.empty else "PASS"},
    ]
    return pd.DataFrame(rows)


def build_checks(
    raw: pd.DataFrame,
    enriched: pd.DataFrame,
    signals: pd.DataFrame,
    raw_path: Path,
    enriched_path: Path,
    signals_path: Path,
    run_date: str,
    max_stale_days: int,
    benchmark_mode: str = "required",
) -> pd.DataFrame:
    rows: list[dict] = []
    rows.extend(frame_checks(raw, "raw", raw_path))
    rows.extend(frame_checks(enriched, "enriched", enriched_path))
    rows.extend(frame_checks(signals, "signals", signals_path))
    rows.extend(alignment_checks(raw, enriched, signals))
    rows.extend(latest_checks(enriched, signals, run_date, max_stale_days, benchmark_mode))
    rows.extend(corporate_action_checks(raw))
    return pd.DataFrame(rows)


def write_report(outdir: Path, checks: pd.DataFrame, snapshot: pd.DataFrame) -> None:
    snap = dict(zip(snapshot["field"], snapshot["value"]))
    failed = checks[~checks["passed"].astype(bool)] if not checks.empty else pd.DataFrame()
    lines = [
        "# Top10 Data Quality Report",
        "",
        f"- Data quality status: {snap.get('data_quality_status', 'NA')}",
        f"- Decision-support data gate: {snap.get('decision_support_data_gate', 'NA')}",
        f"- Run date: {snap.get('run_date', 'NA')}",
        f"- Latest signal date: {snap.get('latest_signal_date', 'NA')}",
        f"- Data source: {snap.get('data_source', 'NA')}",
        f"- Failed checks: {snap.get('failed_checks', 'NA')}",
        f"- Critical failed checks: {snap.get('critical_failed_checks', 'NA')}",
        f"- Warning failed checks: {snap.get('warn_failed_checks', 'NA')}",
        "",
        "## Failed Checks",
        "",
        "| Check | Severity | Value | Details |",
        "|---|---|---:|---|",
    ]
    if failed.empty:
        lines.append("| PASS | NA | 0 | No failed checks. |")
    else:
        for _, row in failed.iterrows():
            lines.append(f"| {row['check']} | {row['severity']} | {row['value']} | {row.get('details', '')} |")
    lines.extend(["", "This report is research tooling, not investment advice."])
    (outdir / "tsm_data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily data-quality checks for TSM research outputs.")
    parser.add_argument("--raw", default="output/tsm_daily_10y_raw.csv")
    parser.add_argument("--enriched", default="output/tsm_daily_10y_enriched.csv")
    parser.add_argument("--signals", default="tsm_price_rule_output/tsm_daily_algorithmic_signals.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--run-date", default=date.today().isoformat())
    parser.add_argument("--max-stale-days", type=int, default=7)
    parser.add_argument(
        "--benchmark-mode",
        choices=["required", "optional", "disabled"],
        default="required",
        help="Whether benchmark join columns are required, optional, or intentionally absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw)
    enriched_path = Path(args.enriched)
    signals_path = Path(args.signals)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = read_csv_if_exists(raw_path, parse_dates=["date"])
    enriched = read_csv_if_exists(enriched_path, parse_dates=["date"])
    signals = read_csv_if_exists(signals_path, parse_dates=["date"])
    checks = build_checks(
        raw,
        enriched,
        signals,
        raw_path,
        enriched_path,
        signals_path,
        args.run_date,
        args.max_stale_days,
        args.benchmark_mode,
    )
    snapshot = build_snapshot(checks, raw, signals, args.run_date)
    issues = checks[~checks["passed"].astype(bool)].copy()

    checks.to_csv(outdir / "tsm_data_quality_checks.csv", index=False)
    issues.to_csv(outdir / "tsm_data_quality_issues.csv", index=False)
    snapshot.to_csv(outdir / "tsm_latest_data_quality_snapshot.csv", index=False)
    write_report(outdir, checks, snapshot)
    print("completed: data quality outputs =", outdir.resolve())
    print(snapshot.to_string(index=False))


if __name__ == "__main__":
    main()

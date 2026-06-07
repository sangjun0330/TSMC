#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate expanded semiconductor universe inputs for pooled prediction research.

This validator only audits data eligibility. It does not download data, fit
models, place orders, or relax any strict model gate.

Outputs:
- tsm_universe_validation_report.csv
- tsm_universe_validation_report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from tsm_core.universe import load_universe_members, members_to_frame


MIN_DAILY_ROWS = 1260
MAX_STRICT_START_DATE = pd.Timestamp("2019-01-01")
MAX_OHLC_MISSING_RATE = 0.01
MIN_RETURN_20D_ROWS = 1000
ETF_GROUPS = {"semiconductor_etf", "etf", "semi_breadth_regime"}


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def read_universe(path: Path) -> pd.DataFrame:
    config = members_to_frame(load_universe_members(path, include_disabled=True))
    config["symbol"] = config["symbol"].astype(str).str.upper()
    config["symbol_group"] = config["symbol_group"].fillna("semiconductor").astype(str)
    return config


def _safe_missing_rate(frame: pd.DataFrame, columns: Iterable[str]) -> float:
    present = [col for col in columns if col in frame.columns]
    if not present or frame.empty:
        return 1.0
    return float(frame[present].isna().any(axis=1).mean())


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def validate_symbol_frame(row: pd.Series, enriched: pd.DataFrame | None) -> Dict[str, object]:
    symbol = str(row.get("symbol", "")).upper()
    symbol_group = str(row.get("symbol_group", "semiconductor"))
    market_region = str(row.get("market_region", "OVERSEAS"))
    is_etf = symbol_group.lower() in ETF_GROUPS
    if enriched is None or enriched.empty:
        return {
            "symbol": symbol,
            "symbol_group": symbol_group,
            "market_region": market_region,
            "loaded": False,
            "strict_eligible": False,
            "eligibility_status": "MISSING_ENRICHED",
            "row_count": 0,
            "start_date": "",
            "end_date": "",
            "duplicate_date_count": np.nan,
            "ohlc_missing_rate": np.nan,
            "return_20d_calculable_rows": 0,
            "median_dollar_volume": np.nan,
            "is_etf_exception": is_etf,
            "failure_reasons": "MISSING_ENRICHED",
        }
    frame = strip_bom_columns(enriched)
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce")
    row_count = int(len(frame))
    start_date = frame["date"].min()
    end_date = frame["date"].max()
    duplicate_dates = int(frame["date"].duplicated().sum()) if "date" in frame.columns else row_count
    ohlc_missing_rate = _safe_missing_rate(frame, ["open", "high", "low", "close"])
    if "is_missing_ohlc" in frame.columns:
        ohlc_missing_rate = max(ohlc_missing_rate, float(_truthy(frame["is_missing_ohlc"]).mean()))
    if "return_20d" in frame.columns:
        return_20d_rows = int(pd.to_numeric(frame["return_20d"], errors="coerce").notna().sum())
    elif "close" in frame.columns:
        return_20d_rows = int(pd.to_numeric(frame["close"], errors="coerce").pct_change(20).notna().sum())
    else:
        return_20d_rows = 0
    if "dollar_volume" in frame.columns:
        median_dollar_volume = float(pd.to_numeric(frame["dollar_volume"], errors="coerce").median())
    elif {"close", "volume"}.issubset(frame.columns):
        median_dollar_volume = float((pd.to_numeric(frame["close"], errors="coerce") * pd.to_numeric(frame["volume"], errors="coerce")).median())
    else:
        median_dollar_volume = np.nan

    failures: list[str] = []
    if row_count < MIN_DAILY_ROWS:
        failures.append("DAILY_ROWS_LT_1260")
    if pd.isna(start_date) or start_date > MAX_STRICT_START_DATE:
        failures.append("START_DATE_AFTER_2019_01_01")
    if duplicate_dates != 0:
        failures.append("DUPLICATE_DATES_GT_0")
    if pd.isna(ohlc_missing_rate) or ohlc_missing_rate >= MAX_OHLC_MISSING_RATE:
        failures.append("OHLC_MISSING_RATE_GTE_1PCT")
    if return_20d_rows < MIN_RETURN_20D_ROWS:
        failures.append("RETURN_20D_ROWS_LT_1000")
    if not is_etf and (pd.isna(median_dollar_volume) or median_dollar_volume <= 0):
        failures.append("MEDIAN_DOLLAR_VOLUME_MISSING")
    short_history = any(reason in failures for reason in ["DAILY_ROWS_LT_1260", "START_DATE_AFTER_2019_01_01", "RETURN_20D_ROWS_LT_1000"])
    if failures:
        status = "SHORT_HISTORY_RESEARCH_ONLY" if short_history else "RESEARCH_ONLY_DATA_QUALITY_FAILED"
    else:
        status = "STRICT_TRAINING_ELIGIBLE"
    return {
        "symbol": symbol,
        "symbol_group": symbol_group,
        "market_region": market_region,
        "loaded": True,
        "strict_eligible": not failures,
        "eligibility_status": status,
        "row_count": row_count,
        "start_date": start_date.date().isoformat() if pd.notna(start_date) else "",
        "end_date": end_date.date().isoformat() if pd.notna(end_date) else "",
        "duplicate_date_count": duplicate_dates,
        "ohlc_missing_rate": ohlc_missing_rate,
        "return_20d_calculable_rows": return_20d_rows,
        "median_dollar_volume": median_dollar_volume,
        "is_etf_exception": is_etf,
        "failure_reasons": "|".join(failures) if failures else "PASS",
    }


def validate_universe(config: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    duplicate_symbols = set(config.loc[config["symbol"].duplicated(keep=False), "symbol"].astype(str))
    duplicate_data_paths = set(config.loc[config["data_outdir"].astype(str).duplicated(keep=False), "data_outdir"].astype(str))
    duplicate_rule_paths = set(config.loc[config["rule_outdir"].astype(str).duplicated(keep=False), "rule_outdir"].astype(str))
    for _, row in config.iterrows():
        path = Path(str(row.get("enriched", "")))
        if path.exists():
            try:
                enriched = pd.read_csv(path)
            except Exception:
                enriched = None
        else:
            enriched = None
        result = validate_symbol_frame(row, enriched)
        end_date = pd.to_datetime(result.get("end_date"), errors="coerce")
        freshness_days = int((pd.Timestamp.utcnow().tz_localize(None).normalize() - end_date).days) if pd.notna(end_date) else np.nan
        symbol = str(row.get("symbol", "")).upper()
        data_path = str(row.get("data_outdir", ""))
        rule_path = str(row.get("rule_outdir", ""))
        path_collision = data_path in duplicate_data_paths or rule_path in duplicate_rule_paths
        duplicate_symbol = symbol in duplicate_symbols
        enabled = _truthy(pd.Series([row.get("enabled", True)])).iloc[0]
        paper_enabled = _truthy(pd.Series([row.get("paper_enabled", True)])).iloc[0]
        freshness_ok = bool(pd.notna(freshness_days) and freshness_days <= 7)
        extra_failures = []
        if duplicate_symbol:
            extra_failures.append("DUPLICATE_SYMBOL")
        if path_collision:
            extra_failures.append("PATH_COLLISION")
        if not enabled:
            extra_failures.append("SYMBOL_DISABLED")
        if not freshness_ok:
            extra_failures.append("DATA_NOT_FRESH")
        result.update(
            {
                "enabled": bool(enabled),
                "paper_enabled": bool(paper_enabled),
                "duplicate_symbol": bool(duplicate_symbol),
                "path_collision": bool(path_collision),
                "freshness_days": freshness_days,
                "freshness_ok": bool(freshness_ok),
                "paper_eligible": bool(enabled and paper_enabled and not duplicate_symbol and not path_collision and freshness_ok and result.get("loaded", False)),
            }
        )
        if extra_failures:
            base = str(result.get("failure_reasons", "PASS"))
            result["failure_reasons"] = "|".join([reason for reason in [base if base != "PASS" else "", *extra_failures] if reason])
            if result.get("eligibility_status") == "STRICT_TRAINING_ELIGIBLE":
                result["eligibility_status"] = "RESEARCH_ONLY_OPERATIONAL_CHECK_FAILED"
            result["strict_eligible"] = False
        rows.append(result)
    return pd.DataFrame(rows)


def write_markdown_report(outdir: Path, config: pd.DataFrame, report: pd.DataFrame) -> None:
    loaded = int(report["loaded"].astype(bool).sum()) if "loaded" in report.columns else 0
    strict = int(report["strict_eligible"].astype(bool).sum()) if "strict_eligible" in report.columns else 0
    candidates = int(config["symbol"].nunique()) if "symbol" in config.columns else 0
    lines = [
        "# Semiconductor Universe Validation Report",
        "",
        "This audit validates free-data eligibility before symbols are allowed into strict pooled model training.",
        "",
        f"- Candidate symbols: {candidates}",
        f"- Loaded symbols: {loaded}",
        f"- Strict eligible symbols: {strict}",
        f"- Loaded target pass: {loaded >= 45} (target >=45)",
        f"- Strict eligible target pass: {strict >= 35} (target >=35)",
        "",
        "## Group Summary",
        "",
        "| Group | Candidates | Loaded | Strict Eligible |",
        "|---|---:|---:|---:|",
    ]
    if report.empty:
        lines.append("| NA | 0 | 0 | 0 |")
    else:
        summary = (
            report.groupby("symbol_group", dropna=False)
            .agg(candidates=("symbol", "nunique"), loaded=("loaded", "sum"), strict_eligible=("strict_eligible", "sum"))
            .reset_index()
            .sort_values("symbol_group")
        )
        for _, row in summary.iterrows():
            lines.append(f"| {row['symbol_group']} | {int(row['candidates'])} | {int(row['loaded'])} | {int(row['strict_eligible'])} |")
    lines.extend(
        [
            "",
            "## Symbol Detail",
            "",
            "| Symbol | Group | Loaded | Strict | Status | Rows | Start | End | Failure Reasons |",
            "|---|---|---:|---:|---|---:|---|---|---|",
        ]
    )
    if report.empty:
        lines.append("| NA | NA | False | False | MISSING | 0 |  |  | MISSING |")
    else:
        for _, row in report.sort_values(["symbol_group", "symbol"]).iterrows():
            lines.append(
                f"| {row['symbol']} | {row['symbol_group']} | {row['loaded']} | {row['strict_eligible']} | "
                f"{row['eligibility_status']} | {int(row['row_count'])} | {row['start_date']} | {row['end_date']} | {row['failure_reasons']} |"
            )
    lines.extend(
        [
            "",
            "## Limitation",
            "- This is an active expanded universe and therefore has survivorship-bias risk. Treat strict eligibility as a data-quality gate, not proof of tradable edge.",
            "- Short-history names such as ARM stay research-only until they meet the full strict history threshold.",
        ]
    )
    (outdir / "tsm_universe_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate expanded semiconductor universe data eligibility.")
    parser.add_argument("--universe-config", default="config/semiconductor_universe_expanded.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = read_universe(Path(args.universe_config))
    report = validate_universe(config)
    report.to_csv(outdir / "tsm_universe_validation_report.csv", index=False)
    write_markdown_report(outdir, config, report)
    loaded = int(report["loaded"].astype(bool).sum()) if not report.empty else 0
    strict = int(report["strict_eligible"].astype(bool).sum()) if not report.empty else 0
    print(f"completed: universe validation loaded_symbols={loaded} strict_eligible_symbols={strict} outdir={outdir.resolve()}")


if __name__ == "__main__":
    main()

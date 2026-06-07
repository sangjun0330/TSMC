#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refresh intraday market data for every enabled universe member.

Daily per-symbol data is built by run_pooled_universe_update.py. This runner
fills the matching hourly/minute layer under each member's data_outdir and
writes root-level manifests so dashboard/audit code can verify coverage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tsm_core.io import strip_bom_columns
from tsm_core.universe import UniverseMember, load_universe_members
from tsm_intraday_provider_adapter import DEFAULT_PROVIDER_ORDER, SUPPORTED_PROVIDERS

INTRADAY_INTERVAL_LABEL = {
    "1m": "minute",
    "2m": "2min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "hourly",
    "1h": "hourly",
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_step(cmd: list[str], continue_on_error: bool) -> dict[str, Any]:
    started = now_utc_iso()
    start_time = time.monotonic()
    result = subprocess.run(cmd, text=True, capture_output=True)
    ended = now_utc_iso()
    row = {
        "returncode": result.returncode,
        "status": "OK" if result.returncode == 0 else "FAILED",
        "started_at_utc": started,
        "ended_at_utc": ended,
        "duration_sec": round(time.monotonic() - start_time, 3),
        "stdout_tail": result.stdout[-1200:],
        "stderr_tail": result.stderr[-1200:],
    }
    if result.returncode != 0 and not continue_on_error:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return row


def read_field_value(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        df = strip_bom_columns(pd.read_csv(path))
    except Exception:
        return {}
    if not {"metric", "value"}.issubset(df.columns):
        return {}
    return dict(zip(df["metric"].astype(str), df["value"]))


def latest_from_enriched(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        wanted = {
            "date",
            "close",
            "close_native",
            "close_usd",
            "listing_currency",
            "display_currency",
            "engine_currency",
            "fx_pair",
            "fx_rate_to_usd",
            "usdkrw",
        }
        df = strip_bom_columns(pd.read_csv(path, usecols=lambda col: col in wanted))
    except Exception:
        return {}
    if df.empty or "date" not in df.columns:
        return {}
    dates = pd.to_datetime(df["date"], errors="coerce")
    idx = dates.idxmax()
    return {
        "latest_timestamp": dates.loc[idx].isoformat() if pd.notna(dates.loc[idx]) else "",
        "latest_close": df.loc[idx, "close"] if "close" in df.columns else "",
        "latest_close_native": df.loc[idx, "close_native"] if "close_native" in df.columns else df.loc[idx, "close"] if "close" in df.columns else "",
        "latest_close_usd": df.loc[idx, "close_usd"] if "close_usd" in df.columns else df.loc[idx, "close"] if "close" in df.columns else "",
        "listing_currency": df.loc[idx, "listing_currency"] if "listing_currency" in df.columns else "",
        "display_currency": df.loc[idx, "display_currency"] if "display_currency" in df.columns else "",
        "engine_currency": df.loc[idx, "engine_currency"] if "engine_currency" in df.columns else "",
        "fx_pair": df.loc[idx, "fx_pair"] if "fx_pair" in df.columns else "",
        "fx_rate_to_usd": df.loc[idx, "fx_rate_to_usd"] if "fx_rate_to_usd" in df.columns else "",
        "usdkrw": df.loc[idx, "usdkrw"] if "usdkrw" in df.columns else "",
        "rows": len(df),
    }


def output_paths(member: UniverseMember, bar_type: str, interval: str = "") -> dict[str, Path]:
    if bar_type == "daily":
        prefix = "tsm_daily_10y"
        return {
            "raw_path": member.data_outdir / f"{prefix}_raw.csv",
            "enriched_path": member.data_outdir / f"{prefix}_enriched.csv",
            "summary_path": member.data_outdir / f"{prefix}_summary.csv",
            "audit_path": member.data_outdir / "tsm_daily_columns_dictionary.csv",
        }
    if bar_type == "hourly":
        prefix = "tsm_hourly"
    else:
        label = INTRADAY_INTERVAL_LABEL.get(interval, str(interval).replace("m", "min"))
        prefix = "tsm_minute" if interval == "1m" else f"tsm_{label}"
    return {
        "raw_path": member.data_outdir / f"{prefix}_available_raw.csv",
        "enriched_path": member.data_outdir / f"{prefix}_available_enriched.csv",
        "summary_path": member.data_outdir / f"{prefix}_available_summary.csv",
        "audit_path": member.data_outdir / f"{prefix}_10y_source_audit.csv",
    }


def command_for_member(member: UniverseMember, args: argparse.Namespace, bar_type: str) -> tuple[list[str], str]:
    py = sys.executable
    if bar_type == "hourly":
        cmd = [
            py,
            "tsm_hourly_quant_pipeline.py",
            "--symbol",
            member.symbol_yahoo,
            "--start",
            args.start,
            "--end",
            args.end,
            "--outdir",
            str(member.data_outdir),
            "--provider",
            args.provider,
            "--provider-order",
            args.provider_order,
            "--listing-currency",
            member.listing_currency,
            "--display-currency",
            member.display_currency,
            "--engine-currency",
            member.engine_currency,
            "--fx-pair",
            member.fx_pair,
            "--fx-rates",
            args.fx_rates,
        ]
        interval = "1h"
    elif bar_type in {"minute", "minute_model", "minute_execution"}:
        interval = args.model_minute_interval if bar_type == "minute_model" else args.execution_minute_interval
        cmd = [
            py,
            "tsm_intraday_quant_pipeline.py",
            "--symbol",
            member.symbol_yahoo,
            "--interval",
            interval,
            "--start",
            args.start,
            "--end",
            args.end,
            "--outdir",
            str(member.data_outdir),
            "--provider",
            args.provider,
            "--provider-order",
            args.provider_order,
            "--listing-currency",
            member.listing_currency,
            "--display-currency",
            member.display_currency,
            "--engine-currency",
            member.engine_currency,
            "--fx-pair",
            member.fx_pair,
            "--fx-rates",
            args.fx_rates,
        ]
    else:
        raise ValueError(f"unsupported bar_type: {bar_type}")
    if args.skip_charts:
        cmd.append("--skip-charts")
    return cmd, interval


def manifest_row(member: UniverseMember, bar_type: str, interval: str, cmd: list[str], run: dict[str, Any]) -> dict[str, Any]:
    paths = output_paths(member, bar_type, interval)
    return {
        "symbol": member.symbol,
        "symbol_group": member.symbol_group,
        "market_region": member.market_region,
        "listing_currency": member.listing_currency,
        "display_currency": member.display_currency,
        "engine_currency": member.engine_currency,
        "fx_pair": member.fx_pair,
        "bar_type": bar_type,
        "interval": interval,
        "data_outdir": str(member.data_outdir),
        "command": " ".join(cmd),
        **{key: str(value) for key, value in paths.items()},
        **run,
    }


def latest_row(member: UniverseMember, bar_type: str, interval: str, status: str = "") -> dict[str, Any]:
    paths = output_paths(member, bar_type, interval)
    summary = read_field_value(paths["summary_path"])
    latest = latest_from_enriched(paths["enriched_path"])
    start_key = "start_timestamp" if bar_type != "daily" else "start_date"
    end_key = "end_timestamp" if bar_type != "daily" else "end_date"
    return {
        "symbol": member.symbol,
        "symbol_group": member.symbol_group,
        "market_region": member.market_region,
        "listing_currency": member.listing_currency,
        "display_currency": member.display_currency,
        "engine_currency": member.engine_currency,
        "fx_pair": member.fx_pair,
        "bar_type": bar_type,
        "interval": interval,
        "data_outdir": str(member.data_outdir),
        "status": status or ("OK" if paths["enriched_path"].exists() else "MISSING"),
        "data_source": summary.get("data_source", ""),
        "start_timestamp": summary.get(start_key, ""),
        "end_timestamp": summary.get(end_key, latest.get("latest_timestamp", "")),
        "bars": summary.get("bars", summary.get("trading_days", latest.get("rows", ""))),
        "complete_requested_coverage": summary.get("complete_requested_coverage", bar_type == "daily"),
        "latest_close": latest.get("latest_close_native", latest.get("latest_close", "")),
        "latest_close_native": latest.get("latest_close_native", ""),
        "latest_close_usd": latest.get("latest_close_usd", latest.get("latest_close", "")),
        "latest_engine_close": latest.get("latest_close", ""),
        "fx_rate_to_usd": latest.get("fx_rate_to_usd", ""),
        "usdkrw": latest.get("usdkrw", ""),
        "enriched_path": str(paths["enriched_path"]),
        "summary_path": str(paths["summary_path"]),
        "audit_path": str(paths["audit_path"]),
        "generated_at_utc": now_utc_iso(),
    }


def selected_bar_types(scope: str) -> list[str]:
    if scope == "hourly":
        return ["hourly"]
    if scope == "minute":
        return ["minute_model", "minute_execution"]
    return ["hourly", "minute_model", "minute_execution"]


def selected_bar_specs(args: argparse.Namespace) -> list[str]:
    specs: list[str] = []
    seen_intervals: set[str] = set()
    for bar_type in selected_bar_types(args.bar_scope):
        interval = "1h"
        if bar_type == "minute_model":
            interval = args.model_minute_interval
        elif bar_type == "minute_execution":
            interval = args.execution_minute_interval
        dedupe_key = interval if bar_type != "hourly" else "1h_hourly"
        if bar_type != "hourly" and dedupe_key in seen_intervals:
            continue
        seen_intervals.add(dedupe_key)
        specs.append(bar_type)
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh universe hourly/minute market data.")
    parser.add_argument("--universe-config", default="config/semiconductor_universe_top10.csv")
    parser.add_argument("--outdir", default="output", help="Root output directory for aggregate manifests.")
    parser.add_argument("--start", default="2016-05-12")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--bar-scope", choices=["hourly", "minute", "both"], default="both")
    parser.add_argument("--minute-interval", choices=["1m", "2m", "5m", "15m", "30m", "60m", "1h"], default="", help="Legacy minute interval; overrides execution-minute-interval when provided.")
    parser.add_argument("--model-minute-interval", choices=["1m", "2m", "5m", "15m", "30m", "60m", "1h"], default="5m")
    parser.add_argument("--execution-minute-interval", choices=["1m", "2m", "5m", "15m", "30m", "60m", "1h"], default="1m")
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, default="auto")
    parser.add_argument("--provider-order", default=",".join(DEFAULT_PROVIDER_ORDER))
    parser.add_argument("--fx-rates", default="output/tsm_fx_rates_daily.csv", help="Daily FX rates used for non-USD listings.")
    parser.add_argument("--skip-charts", action="store_true", help="Skip per-symbol intraday PNG charts.")
    parser.add_argument("--manifest-only", action="store_true", help="Refresh root latest coverage manifest from existing per-symbol files without downloading.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--paper-only", action="store_true", help="Only update paper-enabled universe members.")
    parser.add_argument("--max-symbols", type=int, default=0, help="Testing guard; 0 means all enabled symbols.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.minute_interval).strip():
        args.execution_minute_interval = args.minute_interval
    root_outdir = Path(args.outdir)
    root_outdir.mkdir(parents=True, exist_ok=True)
    members = load_universe_members(args.universe_config, include_disabled=args.include_disabled, paper_only=args.paper_only)
    if args.max_symbols > 0:
        members = members[: args.max_symbols]
    fx_pairs = sorted({member.fx_pair for member in members if member.fx_pair})
    if fx_pairs:
        run_step(
            [
                sys.executable,
                "tsm_fx_rate_engine.py",
                "--start",
                str(args.start),
                "--end",
                str(args.end),
                "--outdir",
                str(Path(args.fx_rates).parent),
                "--pairs",
                ",".join(fx_pairs),
            ],
            args.continue_on_error,
        )

    manifest_rows: list[dict[str, Any]] = []
    latest_rows: list[dict[str, Any]] = []
    for member in members:
        member.data_outdir.mkdir(parents=True, exist_ok=True)
        latest_rows.append(latest_row(member, "daily", "1d"))
        for bar_type in selected_bar_specs(args):
            if args.manifest_only:
                interval = "1h"
                if bar_type == "minute_model":
                    interval = args.model_minute_interval
                elif bar_type == "minute_execution":
                    interval = args.execution_minute_interval
                latest_rows.append(latest_row(member, bar_type, interval))
                continue
            cmd, interval = command_for_member(member, args, bar_type)
            run = run_step(cmd, args.continue_on_error)
            manifest_rows.append(manifest_row(member, bar_type, interval, cmd, run))
            latest_rows.append(latest_row(member, bar_type, interval, str(run["status"])))

    manifest = pd.DataFrame(manifest_rows)
    latest = pd.DataFrame(latest_rows)
    manifest_path = root_outdir / "tsm_universe_intraday_update_manifest.csv"
    latest_path = root_outdir / "tsm_universe_market_data_latest.csv"
    if not args.manifest_only:
        manifest.to_csv(manifest_path, index=False)
    latest.to_csv(latest_path, index=False)
    print("completed: universe intraday manifest =", manifest_path.resolve())
    print("completed: universe market data latest =", latest_path.resolve())
    if not manifest.empty:
        print(manifest[["symbol", "bar_type", "status", "duration_sec"]].tail(20).to_string(index=False))


if __name__ == "__main__":
    main()

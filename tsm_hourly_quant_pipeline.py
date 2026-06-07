#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compatibility entrypoint for per-symbol hourly data.

The shared intraday pipeline already supports 60m/1h bars and writes the
expected `tsm_hourly_*` output files. This wrapper keeps the documented
`tsm_hourly_quant_pipeline.py` command working for automations and README
examples.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import tsm_intraday_quant_pipeline
from tsm_intraday_provider_adapter import DEFAULT_PROVIDER_ORDER, SUPPORTED_PROVIDERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hourly OHLCV files with daily-compatible schema for one requested symbol.")
    parser.add_argument("--symbol", default="TSM", help="Ticker symbol, default: TSM")
    parser.add_argument("--start", default="2016-05-12", help="Requested start date YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), help="Requested end date YYYY-MM-DD")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, default="auto", help="Intraday data provider adapter; auto falls back to Yahoo when needed.")
    parser.add_argument("--provider-order", default=",".join(DEFAULT_PROVIDER_ORDER), help="Comma-separated provider preference order used when --provider auto.")
    parser.add_argument("--listing-currency", default="", help="Listing/native price currency, inferred from ticker when empty.")
    parser.add_argument("--display-currency", default="", help="Dashboard display currency, defaults to listing currency.")
    parser.add_argument("--engine-currency", default="USD", help="Canonical engine/model currency.")
    parser.add_argument("--fx-pair", default="", help="Yahoo FX pair used for native->engine conversion, e.g. KRW=X.")
    parser.add_argument("--fx-rates", default="output/tsm_fx_rates_daily.csv", help="Daily FX rates CSV.")
    parser.add_argument("--skip-charts", action="store_true", help="Skip chart generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    translated = [
        sys.argv[0],
        "--symbol",
        args.symbol,
        "--start",
        args.start,
        "--end",
        args.end,
        "--interval",
        "1h",
        "--provider",
        args.provider,
        "--provider-order",
        args.provider_order,
        "--outdir",
        args.outdir,
        "--listing-currency",
        args.listing_currency,
        "--display-currency",
        args.display_currency,
        "--engine-currency",
        args.engine_currency,
        "--fx-pair",
        args.fx_pair,
        "--fx-rates",
        args.fx_rates,
    ]
    if args.skip_charts:
        translated.append("--skip-charts")

    original_argv = sys.argv
    try:
        sys.argv = translated
        tsm_intraday_quant_pipeline.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()

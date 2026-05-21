#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compatibility entrypoint for TSMC hourly data.

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TSMC hourly OHLCV files with daily-compatible schema.")
    parser.add_argument("--symbol", default="TSM", help="Ticker symbol, default: TSM")
    parser.add_argument("--start", default="2016-05-12", help="Requested start date YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), help="Requested end date YYYY-MM-DD")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--provider", choices=["auto", "yahoo"], default="auto", help="Accepted for automation compatibility; public Yahoo data is used.")
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
        "--outdir",
        args.outdir,
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

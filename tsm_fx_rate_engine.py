#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch daily FX rates used to normalize non-USD listings into USD engine units."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from tsm_core.currency import ensure_fx_rate_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch daily FX rates for multi-currency universe normalization.")
    parser.add_argument("--start", default="2016-05-12", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date YYYY-MM-DD")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--pairs", default="KRW=X", help="Comma-separated Yahoo FX pairs, default: KRW=X")
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    pairs = [pair.strip() for pair in str(args.pairs).split(",") if pair.strip()]
    path = outdir / "tsm_fx_rates_daily.csv"
    rates = ensure_fx_rate_file(path, pairs, start=args.start, end=args.end, timeout=args.timeout)
    latest = pd.DataFrame()
    if not rates.empty and {"date", "fx_pair"}.issubset(rates.columns):
        latest = (
            rates.sort_values(["fx_pair", "date"])
            .groupby("fx_pair", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
    latest_path = outdir / "tsm_fx_latest.csv"
    latest.to_csv(latest_path, index=False, encoding="utf-8-sig")
    print(f"FX rates saved: {path.resolve()}")
    print(f"Latest FX saved: {latest_path.resolve()}")
    if not latest.empty:
        cols = [col for col in ["fx_pair", "date", "usdkrw", "fx_rate_to_usd"] if col in latest.columns]
        print(latest[cols].to_string(index=False))


if __name__ == "__main__":
    main()


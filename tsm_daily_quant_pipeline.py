#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-symbol daily quantitative price-analysis pipeline for Top10 workflows.

What this script does
---------------------
1) Downloads daily OHLCV data for the requested symbol.
2) Cleans and validates the data.
3) Computes detailed daily price movement, return, drawdown, volatility,
   ATR, trend, momentum, liquidity, relative-strength, and event-window metrics.
4) Saves enriched CSV files and detailed PNG charts.

Default output files
--------------------
output/tsm_daily_10y_enriched.csv
output/tsm_daily_10y_summary.csv
output/tsm_event_impact_10y.csv
output/charts/tsm_price_ma_drawdown.png
output/charts/tsm_daily_return_distribution.png
output/charts/tsm_rolling_vol_atr.png
output/charts/tsm_daily_move_heatmap.png
output/charts/tsm_event_impact.png

Install
-------
pip install -r requirements.txt

Run
---
python tsm_daily_quant_pipeline.py --start 2016-05-12 --end 2026-05-12 --outdir output

Notes
-----
- This is educational research tooling, not investment advice.
- If Yahoo is rate-limited, the script falls back to Stooq.
- Stooq provides OHLCV without an explicit Adj Close column. In that case,
  adj_close is set equal to close and the data_quality_note column marks it.
- For precise total-return backtests, confirm dividend/split handling with your broker
  or a paid corporate-actions data source.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import time
from dataclasses import dataclass
from datetime import date, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from tsm_core.currency import (
    currency_profile,
    ensure_fx_rate_file,
    load_fx_rates,
    normalize_ohlcv_to_engine_currency,
    requires_fx_conversion,
)

# matplotlib is imported lazily in make_charts() so that CSV generation still works
# on machines without a display backend.


# -----------------------------
# Configuration
# -----------------------------

DEFAULT_SYMBOL_STOOQ = "tsm.us"
DEFAULT_SYMBOL_YAHOO = "TSM"
DEFAULT_BENCHMARKS = {
    "SPY": "spy.us",     # broad US equity benchmark
    "SMH": "smh.us",     # semiconductor ETF benchmark
    "QQQ": "qqq.us",     # Nasdaq-heavy benchmark
}


@dataclass
class DownloadResult:
    df: pd.DataFrame
    source: str
    url: str
    note: str


# -----------------------------
# Utility functions
# -----------------------------


def ensure_outdirs(outdir: Path) -> Dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    charts_dir = outdir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    return {"outdir": outdir, "charts": charts_dir}


def to_yyyymmdd(d: str) -> str:
    return pd.to_datetime(d).strftime("%Y%m%d")


def unix_seconds(d: str) -> int:
    dt = pd.to_datetime(d).to_pydatetime()
    # Yahoo period2 is exclusive; add one day for inclusive end handling.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def pct(x: float) -> float:
    return float(x) * 100.0 if pd.notna(x) else np.nan


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace({0: np.nan})


def annualize_vol(series: pd.Series, periods: int = 252) -> float:
    return float(series.dropna().std(ddof=0) * math.sqrt(periods)) if series.dropna().shape[0] > 1 else np.nan


def percentile_rank_last(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return float(s.rank(pct=True).iloc[-1])


# -----------------------------
# Data download
# -----------------------------


def download_stooq(symbol: str, start: str, end: str, timeout: int = 30) -> DownloadResult:
    """Download daily OHLCV from Stooq CSV endpoint."""
    d1 = to_yyyymmdd(start)
    d2 = to_yyyymmdd(end)
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&d1={d1}&d2={d2}&i=d"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; TSM-Quant-Research/1.0)",
        "Accept": "text/csv,text/plain,*/*",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    text = r.text.strip()
    if not text or "Date" not in text[:50]:
        raise RuntimeError(f"Stooq returned empty or invalid CSV: {text[:200]}")

    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        raise RuntimeError("Stooq returned an empty dataset")
    # Expected: Date,Open,High,Low,Close,Volume
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Stooq missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["adj_close"] = df["close"]
    df["data_source"] = "stooq"
    df["data_quality_note"] = "Stooq OHLCV; adj_close set equal to close unless you validate corporate actions separately."
    return DownloadResult(df=df, source="stooq", url=url, note="OHLCV downloaded from Stooq daily CSV endpoint.")


def download_yahoo_chart(symbol: str, start: str, end: str, timeout: int = 30) -> DownloadResult:
    """Download daily OHLCV and adjusted close from Yahoo chart JSON.

    This avoids the older CSV download endpoint that often requires cookies/crumb.
    """
    p1 = unix_seconds(start)
    p2 = unix_seconds((pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    base_urls = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={p2}&interval=1d&events=history%7Cdiv%7Csplit",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={p2}&interval=1d&events=history%7Cdiv%7Csplit",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }
    last_error = None
    for url in base_urls:
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            chart = payload.get("chart", {})
            if chart.get("error"):
                raise RuntimeError(str(chart["error"]))
            result = chart.get("result", [None])[0]
            if not result:
                raise RuntimeError("Yahoo chart JSON has no result")
            timestamps = result.get("timestamp", [])
            quote = result.get("indicators", {}).get("quote", [{}])[0]
            adj = result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")
            if not timestamps or not quote:
                raise RuntimeError("Yahoo chart JSON has no timestamps/quotes")
            df = pd.DataFrame({
                "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume"),
                "adj_close": adj if adj is not None else quote.get("close"),
            })
            df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
            df = df.reset_index(drop=True)
            if df.empty:
                raise RuntimeError("Yahoo chart returned empty OHLC data")
            df["data_source"] = "yahoo_chart"
            df["data_quality_note"] = "Yahoo chart JSON with adjusted close when provided."
            return DownloadResult(df=df, source="yahoo_chart", url=url, note="OHLCV downloaded from Yahoo chart JSON.")
        except Exception as e:
            last_error = e
            time.sleep(1)
    raise RuntimeError(f"Yahoo download failed: {last_error}")


def download_prices(
    stooq_symbol: str,
    yahoo_symbol: str,
    start: str,
    end: str,
    preferred_source: str = "stooq",
) -> DownloadResult:
    """Download price data with fallback."""
    preferred_source = preferred_source.lower()
    errors: List[str] = []
    if preferred_source == "yahoo":
        try:
            return download_yahoo_chart(yahoo_symbol, start, end)
        except Exception as e:
            errors.append(f"Yahoo failed: {e}")
        try:
            return download_stooq(stooq_symbol, start, end)
        except Exception as e:
            errors.append(f"Stooq failed: {e}")
    else:
        try:
            return download_stooq(stooq_symbol, start, end)
        except Exception as e:
            errors.append(f"Stooq failed: {e}")
        try:
            return download_yahoo_chart(yahoo_symbol, start, end)
        except Exception as e:
            errors.append(f"Yahoo failed: {e}")
    raise RuntimeError("; ".join(errors))


# -----------------------------
# Feature engineering
# -----------------------------


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and add data-quality flags."""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_missing_ohlc"] = df[["open", "high", "low", "close"]].isna().any(axis=1)
    df["is_bad_high_low"] = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    df["is_zero_or_negative_price"] = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    df["is_zero_volume"] = df["volume"].fillna(0) <= 0
    df["trading_day_number"] = np.arange(1, len(df) + 1)
    return df


def add_core_price_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    df["ticker"] = ticker.upper().replace(".US", "")

    df["prev_close"] = df["close"].shift(1)
    df["prev_adj_close"] = df["adj_close"].shift(1)

    # Close-to-close returns and changes.
    df["close_change_usd"] = df["close"] - df["prev_close"]
    df["close_change_pct"] = safe_div(df["close"], df["prev_close"]) - 1
    df["adj_close_change_pct"] = safe_div(df["adj_close"], df["prev_adj_close"]) - 1
    df["log_return"] = np.log(safe_div(df["adj_close"], df["prev_adj_close"]))

    # Gap and intraday movement.
    df["open_gap_usd"] = df["open"] - df["prev_close"]
    df["open_gap_pct"] = safe_div(df["open"], df["prev_close"]) - 1
    df["open_to_close_usd"] = df["close"] - df["open"]
    df["open_to_close_pct"] = safe_div(df["close"], df["open"]) - 1
    df["high_minus_prev_close_pct"] = safe_div(df["high"], df["prev_close"]) - 1
    df["low_minus_prev_close_pct"] = safe_div(df["low"], df["prev_close"]) - 1
    df["intraday_range_usd"] = df["high"] - df["low"]
    df["intraday_range_pct_close"] = safe_div(df["intraday_range_usd"], df["close"])
    df["intraday_range_pct_prev_close"] = safe_div(df["intraday_range_usd"], df["prev_close"])
    df["high_low_ratio"] = safe_div(df["high"], df["low"])
    df["high_low_pct"] = df["high_low_ratio"] - 1

    # Candlestick anatomy.
    df["candle_body_usd"] = (df["close"] - df["open"]).abs()
    df["candle_body_pct"] = safe_div(df["candle_body_usd"], df["open"])
    df["upper_shadow_usd"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_shadow_usd"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["upper_shadow_pct"] = safe_div(df["upper_shadow_usd"], df["open"])
    df["lower_shadow_pct"] = safe_div(df["lower_shadow_usd"], df["open"])
    df["close_position_in_range"] = safe_div(df["close"] - df["low"], df["high"] - df["low"])
    df["direction"] = np.select(
        [df["close_change_pct"] > 0, df["close_change_pct"] < 0],
        ["up", "down"],
        default="flat",
    )
    df["intraday_direction"] = np.select(
        [df["close"] > df["open"], df["close"] < df["open"]],
        ["bull_candle", "bear_candle"],
        default="doji",
    )
    df["gap_direction"] = np.select(
        [df["open_gap_pct"] > 0, df["open_gap_pct"] < 0],
        ["gap_up", "gap_down"],
        default="no_gap",
    )
    return df


def add_rolling_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    windows = [2, 3, 5, 10, 20, 21, 50, 60, 63, 100, 120, 126, 200, 252]
    for w in windows:
        df[f"return_{w}d"] = safe_div(df["adj_close"], df["adj_close"].shift(w)) - 1
        df[f"close_change_{w}d_usd"] = df["close"] - df["close"].shift(w)

    # Classic momentum variants excluding the most recent 1 month.
    df["momentum_3m_ex_1m"] = safe_div(df["adj_close"].shift(21), df["adj_close"].shift(63)) - 1
    df["momentum_6m_ex_1m"] = safe_div(df["adj_close"].shift(21), df["adj_close"].shift(126)) - 1
    df["momentum_12m_ex_1m"] = safe_div(df["adj_close"].shift(21), df["adj_close"].shift(252)) - 1

    # Cumulative return from first valid day.
    first = df["adj_close"].dropna().iloc[0]
    df["cumulative_return_from_start"] = safe_div(df["adj_close"], pd.Series(first, index=df.index)) - 1
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # True range and ATR.
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["prev_close"]).abs()
    tr3 = (df["low"] - df["prev_close"]).abs()
    df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    for w in [5, 10, 14, 20, 50, 63, 100, 252]:
        df[f"atr_{w}"] = df["true_range"].rolling(w, min_periods=max(2, int(w * 0.5))).mean()
        df[f"atr_{w}_pct"] = safe_div(df[f"atr_{w}"], df["close"])

    # Close-to-close volatility.
    for w in [5, 10, 20, 21, 50, 60, 63, 100, 120, 126, 200, 252]:
        df[f"vol_{w}d_daily"] = df["log_return"].rolling(w, min_periods=max(2, int(w * 0.5))).std(ddof=0)
        df[f"vol_{w}d_ann"] = df[f"vol_{w}d_daily"] * np.sqrt(252)
        downside = df["log_return"].where(df["log_return"] < 0, np.nan)
        df[f"downside_vol_{w}d_ann"] = downside.rolling(w, min_periods=max(2, int(w * 0.5))).std(ddof=0) * np.sqrt(252)

    # Range-based volatility estimators.
    log_hl = np.log(safe_div(df["high"], df["low"]))
    log_co = np.log(safe_div(df["close"], df["open"]))
    parkinson_daily_var = (log_hl ** 2) / (4 * np.log(2))
    gk_daily_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    gk_daily_var = gk_daily_var.clip(lower=0)
    for w in [10, 20, 60, 63, 120, 252]:
        df[f"parkinson_vol_{w}d_ann"] = np.sqrt(parkinson_daily_var.rolling(w, min_periods=max(2, int(w * 0.5))).mean() * 252)
        df[f"garman_klass_vol_{w}d_ann"] = np.sqrt(gk_daily_var.rolling(w, min_periods=max(2, int(w * 0.5))).mean() * 252)

    # Tail-risk frequency.
    for threshold in [-0.03, -0.05, -0.08, -0.10]:
        name = str(abs(int(threshold * 100)))
        df[f"is_down_{name}pct_or_more"] = df["close_change_pct"] <= threshold
        df[f"down_{name}pct_count_252d"] = df[f"is_down_{name}pct_or_more"].rolling(252, min_periods=20).sum()
    for threshold in [0.03, 0.05, 0.08, 0.10]:
        name = str(abs(int(threshold * 100)))
        df[f"is_up_{name}pct_or_more"] = df["close_change_pct"] >= threshold
        df[f"up_{name}pct_count_252d"] = df[f"is_up_{name}pct_or_more"].rolling(252, min_periods=20).sum()

    return df


def add_trend_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for w in [5, 10, 20, 21, 50, 60, 100, 120, 200, 252]:
        df[f"sma_{w}"] = df["close"].rolling(w, min_periods=max(2, int(w * 0.5))).mean()
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False, min_periods=max(2, int(w * 0.5))).mean()
        df[f"dist_close_sma_{w}_pct"] = safe_div(df["close"], df[f"sma_{w}"]) - 1
        df[f"sma_{w}_slope_5d_pct"] = safe_div(df[f"sma_{w}"], df[f"sma_{w}"].shift(5)) - 1

    # RSI(14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace({0: np.nan})
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD 12/26/9
    ema12 = df["close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = df["close"].ewm(span=26, adjust=False, min_periods=26).mean()
    df["macd_12_26"] = ema12 - ema26
    df["macd_signal_9"] = df["macd_12_26"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_hist"] = df["macd_12_26"] - df["macd_signal_9"]

    # Bollinger 20.
    df["boll_mid_20"] = df["close"].rolling(20, min_periods=10).mean()
    df["boll_std_20"] = df["close"].rolling(20, min_periods=10).std(ddof=0)
    df["boll_upper_20_2"] = df["boll_mid_20"] + 2 * df["boll_std_20"]
    df["boll_lower_20_2"] = df["boll_mid_20"] - 2 * df["boll_std_20"]
    df["boll_width_20_pct"] = safe_div(df["boll_upper_20_2"] - df["boll_lower_20_2"], df["boll_mid_20"])
    df["boll_z_20"] = safe_div(df["close"] - df["boll_mid_20"], df["boll_std_20"])

    # Breakout and breakdown flags. Use previous rolling high/low to avoid look-ahead.
    for w in [20, 50, 60, 100, 120, 200, 252]:
        prev_high = df["high"].shift(1).rolling(w, min_periods=max(2, int(w * 0.5))).max()
        prev_low = df["low"].shift(1).rolling(w, min_periods=max(2, int(w * 0.5))).min()
        df[f"breakout_{w}d_high"] = df["close"] > prev_high
        df[f"breakdown_{w}d_low"] = df["close"] < prev_low
        df[f"prev_{w}d_high"] = prev_high
        df[f"prev_{w}d_low"] = prev_low

    df["trend_regime"] = np.select(
        [
            (df["close"] > df["sma_50"]) & (df["sma_50"] > df["sma_200"]),
            (df["close"] < df["sma_50"]) & (df["sma_50"] < df["sma_200"]),
            (df["close"] > df["sma_200"]),
            (df["close"] < df["sma_200"]),
        ],
        ["strong_uptrend", "strong_downtrend", "above_200d_mixed", "below_200d_mixed"],
        default="insufficient_data",
    )
    df["momentum_signal"] = np.select(
        [
            (df["return_20d"] > 0) & (df["return_63d"] > 0) & (df["return_126d"] > 0),
            (df["return_20d"] < 0) & (df["return_63d"] < 0) & (df["return_126d"] < 0),
        ],
        ["positive_multi_horizon", "negative_multi_horizon"],
        default="mixed",
    )
    return df


def add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dollar_volume"] = df["close"] * df["volume"]
    for w in [5, 10, 20, 21, 50, 60, 63, 100, 120, 252]:
        df[f"volume_ma_{w}"] = df["volume"].rolling(w, min_periods=max(2, int(w * 0.5))).mean()
        df[f"dollar_volume_ma_{w}"] = df["dollar_volume"].rolling(w, min_periods=max(2, int(w * 0.5))).mean()
        df[f"volume_ratio_{w}"] = safe_div(df["volume"], df[f"volume_ma_{w}"])
        df[f"dollar_volume_ratio_{w}"] = safe_div(df["dollar_volume"], df[f"dollar_volume_ma_{w}"])
    # Amihud illiquidity proxy: |return| / dollar volume.
    df["amihud_daily"] = safe_div(df["close_change_pct"].abs(), df["dollar_volume"])
    for w in [20, 60, 252]:
        df[f"amihud_{w}d"] = df["amihud_daily"].rolling(w, min_periods=max(2, int(w * 0.5))).mean()
    return df


def add_drawdown_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["running_high_close"] = df["adj_close"].cummax()
    df["drawdown_from_ath"] = safe_div(df["adj_close"], df["running_high_close"]) - 1
    df["is_new_ath"] = df["adj_close"] >= df["running_high_close"]
    for w in [20, 60, 120, 252, 504, 756, 1260, 2520]:
        roll_high = df["adj_close"].rolling(w, min_periods=max(2, int(w * 0.5))).max()
        df[f"drawdown_from_{w}d_high"] = safe_div(df["adj_close"], roll_high) - 1
        # Rolling max drawdown over window. This is heavier but OK for one ticker.
        df[f"max_drawdown_{w}d"] = df["adj_close"].rolling(w, min_periods=max(10, int(w * 0.5))).apply(
            lambda x: float((x / np.maximum.accumulate(x) - 1).min()),
            raw=True,
        )
    return df


def add_zscore_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["close_change_pct", "log_return", "intraday_range_pct_prev_close", "volume", "dollar_volume", "atr_14_pct"]:
        if col not in df.columns:
            continue
        for w in [20, 60, 252]:
            mean = df[col].rolling(w, min_periods=max(5, int(w * 0.5))).mean()
            std = df[col].rolling(w, min_periods=max(5, int(w * 0.5))).std(ddof=0)
            df[f"{col}_z_{w}d"] = safe_div(df[col] - mean, std)
    # Volatility regime based on rolling 252-day percentile of 20d annualized volatility.
    df["vol20_rank_252d"] = df["vol_20d_ann"].rolling(252, min_periods=60).apply(percentile_rank_last, raw=False)
    df["vol_regime"] = np.select(
        [df["vol20_rank_252d"] >= 0.80, df["vol20_rank_252d"] <= 0.20],
        ["high_vol", "low_vol"],
        default="normal_vol",
    )
    return df


def add_benchmark_features(df: pd.DataFrame, benchmarks: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add relative returns and rolling beta versus benchmarks."""
    df = df.copy()
    for name, bdf in benchmarks.items():
        if bdf is None or bdf.empty:
            continue
        b = bdf[["date", "adj_close"]].copy()
        b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
        b = b.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
        b = b.rename(columns={"adj_close": f"{name.lower()}_adj_close"})
        order_col = "_benchmark_merge_order"
        left = df.copy()
        left[order_col] = np.arange(len(left))
        left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
        df = (
            pd.merge_asof(left.sort_values("date"), b, on="date", direction="backward")
            .sort_values(order_col)
            .drop(columns=[order_col])
            .reset_index(drop=True)
        )
        bcol = f"{name.lower()}_adj_close"
        df[f"{name.lower()}_return_1d"] = safe_div(df[bcol], df[bcol].shift(1)) - 1
        for w in [20, 21, 60, 63, 120, 126, 252]:
            df[f"{name.lower()}_return_{w}d"] = safe_div(df[bcol], df[bcol].shift(w)) - 1
            df[f"relative_return_vs_{name.lower()}_{w}d"] = df[f"return_{w}d"] - df[f"{name.lower()}_return_{w}d"]
        # Rolling beta using daily returns.
        for w in [60, 126, 252]:
            cov = df["close_change_pct"].rolling(w, min_periods=max(20, int(w * 0.5))).cov(df[f"{name.lower()}_return_1d"])
            var = df[f"{name.lower()}_return_1d"].rolling(w, min_periods=max(20, int(w * 0.5))).var(ddof=0)
            df[f"beta_vs_{name.lower()}_{w}d"] = cov / var.replace({0: np.nan})
    return df


def add_risk_trade_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Add ATR-based risk levels for trading-system design."""
    df = df.copy()
    # Hypothetical next-day risk levels based on current close/ATR.
    for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
        key = str(k).replace(".", "_")
        df[f"atr14_stop_long_k_{key}"] = df["close"] - k * df["atr_14"]
        df[f"atr14_takeprofit_2r_k_{key}"] = df["close"] + 2 * k * df["atr_14"]
        df[f"risk_pct_atr14_k_{key}"] = safe_div(k * df["atr_14"], df["close"])
    # Position notional if account risk is 0.5% or 1.0% and stop is ATR14 x 2.
    stop_risk_pct = safe_div(2 * df["atr_14"], df["close"])
    df["position_notional_for_0_5pct_account_risk_using_atr2"] = safe_div(pd.Series(0.005, index=df.index), stop_risk_pct)
    df["position_notional_for_1pct_account_risk_using_atr2"] = safe_div(pd.Series(0.010, index=df.index), stop_risk_pct)
    return df


def enrich_prices(df: pd.DataFrame, ticker: str, benchmarks: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    df = validate_ohlcv(df)
    df = add_core_price_features(df, ticker)
    df = add_rolling_returns(df)
    df = add_volatility_features(df)
    df = add_trend_momentum_features(df)
    df = add_liquidity_features(df)
    df = add_drawdown_features(df)
    df = add_zscore_regime_features(df)
    if benchmarks:
        df = add_benchmark_features(df, benchmarks)
    df = add_risk_trade_levels(df)
    return df


# -----------------------------
# Event analysis
# -----------------------------


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["event_date", "event_name", "event_type", "source_url", "notes"])
    events = pd.read_csv(path)
    if events.empty:
        return pd.DataFrame(columns=["event_date", "event_name", "event_type", "source_url", "notes"])
    events["event_date"] = pd.to_datetime(events["event_date"])
    return events


def nearest_trading_day(dates: pd.Series, event_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    if dates.empty:
        return None
    future = dates[dates >= event_date]
    if len(future) > 0:
        return future.iloc[0]
    return dates.iloc[-1]


def compute_event_impact(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    trading_dates = df["date"]
    idx_by_date = {d: i for i, d in enumerate(df["date"])}
    for _, ev in events.iterrows():
        event_date = pd.to_datetime(ev["event_date"])
        td = nearest_trading_day(trading_dates, event_date)
        if td is None or td not in idx_by_date:
            continue
        i = idx_by_date[td]
        row = {
            "event_date": event_date.date().isoformat(),
            "trading_date_used": td.date().isoformat(),
            "event_name": ev.get("event_name", ""),
            "event_type": ev.get("event_type", ""),
            "source_url": ev.get("source_url", ""),
            "notes": ev.get("notes", ""),
            "close_on_event_trading_day": df.loc[i, "close"],
            "return_on_event_day_pct": pct(df.loc[i, "close_change_pct"]),
            "gap_on_event_day_pct": pct(df.loc[i, "open_gap_pct"]),
            "intraday_range_event_day_pct": pct(df.loc[i, "intraday_range_pct_prev_close"]),
            "vol20_ann_on_event_day_pct": pct(df.loc[i, "vol_20d_ann"]),
            "atr14_pct_on_event_day_pct": pct(df.loc[i, "atr_14_pct"]),
            "drawdown_from_ath_pct": pct(df.loc[i, "drawdown_from_ath"]),
        }
        for h in [1, 2, 3, 5, 10, 20, 21, 60, 63]:
            if i + h < len(df):
                row[f"post_{h}d_return_pct"] = pct(df.loc[i + h, "adj_close"] / df.loc[i, "adj_close"] - 1)
            else:
                row[f"post_{h}d_return_pct"] = np.nan
            if i - h >= 0:
                row[f"pre_{h}d_return_pct"] = pct(df.loc[i, "adj_close"] / df.loc[i - h, "adj_close"] - 1)
            else:
                row[f"pre_{h}d_return_pct"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------
# Summary stats
# -----------------------------


def compute_summary(df: pd.DataFrame, source_info: DownloadResult) -> pd.DataFrame:
    ret = df["adj_close_change_pct"].dropna()
    log_ret = df["log_return"].dropna()
    start_date = df["date"].iloc[0]
    end_date = df["date"].iloc[-1]
    start_price = df["adj_close"].iloc[0]
    end_price = df["adj_close"].iloc[-1]
    years = (end_date - start_date).days / 365.25
    total_return = end_price / start_price - 1
    cagr = (end_price / start_price) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = annualize_vol(log_ret)
    mdd = float(df["drawdown_from_ath"].min())
    sharpe_zero_rf = cagr / ann_vol if ann_vol and not np.isnan(ann_vol) and ann_vol != 0 else np.nan
    sortino = cagr / annualize_vol(log_ret[log_ret < 0]) if len(log_ret[log_ret < 0]) > 1 else np.nan
    calmar = cagr / abs(mdd) if mdd and mdd < 0 else np.nan

    rows = [
        ("ticker", df["ticker"].iloc[0]),
        ("listing_currency", df["listing_currency"].iloc[-1] if "listing_currency" in df.columns else "USD"),
        ("display_currency", df["display_currency"].iloc[-1] if "display_currency" in df.columns else "USD"),
        ("engine_currency", df["engine_currency"].iloc[-1] if "engine_currency" in df.columns else "USD"),
        ("fx_pair", df["fx_pair"].iloc[-1] if "fx_pair" in df.columns else ""),
        ("latest_fx_rate_to_usd", df["fx_rate_to_usd"].iloc[-1] if "fx_rate_to_usd" in df.columns else 1.0),
        ("latest_usdkrw", df["usdkrw"].iloc[-1] if "usdkrw" in df.columns else np.nan),
        ("data_source", source_info.source),
        ("data_url", source_info.url),
        ("data_note", source_info.note),
        ("start_date", start_date.date().isoformat()),
        ("end_date", end_date.date().isoformat()),
        ("trading_days", len(df)),
        ("start_adj_close", start_price),
        ("end_adj_close", end_price),
        ("start_adj_close_native", df["adj_close_native"].iloc[0] if "adj_close_native" in df.columns else start_price),
        ("end_adj_close_native", df["adj_close_native"].iloc[-1] if "adj_close_native" in df.columns else end_price),
        ("start_close_native", df["close_native"].iloc[0] if "close_native" in df.columns else df["close"].iloc[0]),
        ("end_close_native", df["close_native"].iloc[-1] if "close_native" in df.columns else df["close"].iloc[-1]),
        ("total_return_pct", pct(total_return)),
        ("cagr_pct", pct(cagr)),
        ("annualized_volatility_pct", pct(ann_vol)),
        ("max_drawdown_pct", pct(mdd)),
        ("sharpe_zero_rf", sharpe_zero_rf),
        ("sortino_zero_rf", sortino),
        ("calmar_ratio", calmar),
        ("best_daily_return_pct", pct(ret.max())),
        ("worst_daily_return_pct", pct(ret.min())),
        ("avg_daily_return_pct", pct(ret.mean())),
        ("median_daily_return_pct", pct(ret.median())),
        ("positive_day_ratio_pct", pct((ret > 0).mean())),
        ("negative_day_ratio_pct", pct((ret < 0).mean())),
        ("avg_intraday_range_pct", pct(df["intraday_range_pct_prev_close"].mean())),
        ("median_intraday_range_pct", pct(df["intraday_range_pct_prev_close"].median())),
        ("avg_20d_ann_vol_pct", pct(df["vol_20d_ann"].mean())),
        ("latest_20d_ann_vol_pct", pct(df["vol_20d_ann"].iloc[-1])),
        ("latest_63d_ann_vol_pct", pct(df["vol_63d_ann"].iloc[-1])),
        ("latest_252d_ann_vol_pct", pct(df["vol_252d_ann"].iloc[-1])),
        ("latest_atr14_pct", pct(df["atr_14_pct"].iloc[-1])),
        ("latest_drawdown_from_ath_pct", pct(df["drawdown_from_ath"].iloc[-1])),
        ("latest_trend_regime", df["trend_regime"].iloc[-1]),
        ("latest_momentum_signal", df["momentum_signal"].iloc[-1]),
        ("latest_vol_regime", df["vol_regime"].iloc[-1]),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


# -----------------------------
# Chart generation
# -----------------------------


def make_charts(df: pd.DataFrame, event_impact: pd.DataFrame, charts_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) Price + moving averages + drawdown.
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot(df["date"], df["close"], label="Close", linewidth=1.2)
    for w in [20, 50, 200]:
        ax1.plot(df["date"], df[f"sma_{w}"], label=f"SMA {w}", linewidth=1.0)
    ax1.set_title("Per-symbol daily close with 20/50/200-day moving averages")
    engine_currency = df["engine_currency"].iloc[-1] if "engine_currency" in df.columns and not df.empty else "USD"
    ax1.set_ylabel(f"Price, {engine_currency}")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.fill_between(df["date"], df["drawdown_from_ath"] * 100, 0, alpha=0.2, label="Drawdown from ATH (%)")
    ax2.set_ylabel("Drawdown, %")
    fig.tight_layout()
    fig.savefig(charts_dir / "tsm_price_ma_drawdown.png", dpi=160)
    plt.close(fig)

    # 2) Rolling volatility and ATR.
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot(df["date"], df["vol_20d_ann"] * 100, label="20D annualized vol", linewidth=1.0)
    ax1.plot(df["date"], df["vol_63d_ann"] * 100, label="63D annualized vol", linewidth=1.0)
    ax1.plot(df["date"], df["vol_252d_ann"] * 100, label="252D annualized vol", linewidth=1.0)
    ax1.set_ylabel("Annualized volatility, %")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["atr_14_pct"] * 100, label="ATR14 %", linewidth=1.0)
    ax2.set_ylabel("ATR14, % of close")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Per-symbol rolling volatility and ATR")
    fig.tight_layout()
    fig.savefig(charts_dir / "tsm_rolling_vol_atr.png", dpi=160)
    plt.close(fig)

    # 3) Daily return distribution.
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.hist(df["close_change_pct"].dropna() * 100, bins=80)
    ax.axvline(0, linewidth=1.0)
    ax.set_title("Per-symbol daily close-to-close return distribution")
    ax.set_xlabel("Daily return, %")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts_dir / "tsm_daily_return_distribution.png", dpi=160)
    plt.close(fig)

    # 4) Daily returns + volume on same timeline.
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.bar(df["date"], df["close_change_pct"] * 100, width=1.0, label="Daily return %")
    ax1.set_ylabel("Daily return, %")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["volume_ma_20"] / 1_000_000, label="20D avg volume, M", linewidth=1.0)
    ax2.set_ylabel("Volume, million shares")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Per-symbol daily return bars and 20-day average volume")
    fig.tight_layout()
    fig.savefig(charts_dir / "tsm_daily_return_volume.png", dpi=160)
    plt.close(fig)

    # 5) Daily move heatmap by year/month.
    tmp = df.copy()
    tmp["year"] = tmp["date"].dt.year
    tmp["month"] = tmp["date"].dt.month
    pivot = tmp.pivot_table(index="year", columns="month", values="close_change_pct", aggfunc=lambda x: np.nanmean(np.abs(x)) * 100)
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title("Per-symbol average absolute daily move by month, %")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([str(i) for i in range(1, 13)])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    fig.colorbar(im, ax=ax, label="Average abs daily move, %")
    fig.tight_layout()
    fig.savefig(charts_dir / "tsm_daily_move_heatmap.png", dpi=160)
    plt.close(fig)

    # 6) Event impact if available.
    if event_impact is not None and not event_impact.empty:
        plot_df = event_impact.dropna(subset=["post_5d_return_pct"]).copy()
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(14, 8))
            labels = [f"{d}\n{name[:28]}" for d, name in zip(plot_df["trading_date_used"], plot_df["event_name"])]
            x = np.arange(len(plot_df))
            ax.bar(x, plot_df["return_on_event_day_pct"], label="Event-day return %")
            ax.plot(x, plot_df["post_5d_return_pct"], marker="o", label="Post 5D return %")
            ax.plot(x, plot_df["post_20d_return_pct"], marker="o", label="Post 20D return %")
            ax.axhline(0, linewidth=1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=70, ha="right")
            ax.set_title("Per-symbol event-day and post-event returns")
            ax.set_ylabel("Return, %")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(charts_dir / "tsm_event_impact.png", dpi=160)
            plt.close(fig)


# -----------------------------
# Column dictionary
# -----------------------------


def write_column_dictionary(path: Path) -> None:
    rows = [
        ("date", "거래일"),
        ("open/high/low/close", "일별 OHLC 원시 가격"),
        ("adj_close", "수정 종가. Stooq 사용 시 close와 동일하게 설정, Yahoo 사용 시 조정종가"),
        ("close_change_usd", "전일 종가 대비 달러 상승/하락폭"),
        ("close_change_pct", "전일 종가 대비 상승률/하락률"),
        ("log_return", "로그 수익률"),
        ("open_gap_pct", "전일 종가 대비 당일 시가 갭 비율"),
        ("open_to_close_pct", "당일 시가 대비 종가 변화율"),
        ("intraday_range_pct_prev_close", "고가-저가 범위를 전일 종가로 나눈 일중 변동폭"),
        ("close_position_in_range", "종가가 당일 저가~고가 범위 중 어디에 위치하는지. 1에 가까울수록 고가 마감"),
        ("return_20d/63d/126d/252d", "각 기간 누적 수익률"),
        ("momentum_12m_ex_1m", "최근 1개월 제외 12개월 모멘텀"),
        ("vol_20d_ann/63d_ann/252d_ann", "일별 로그수익률 기반 연율화 변동성"),
        ("downside_vol_*", "하락일 수익률만 사용한 연율화 하락 변동성"),
        ("true_range", "고가-저가와 갭을 반영한 진폭"),
        ("atr_14", "14일 평균 True Range"),
        ("atr_14_pct", "ATR14를 종가로 나눈 변동성 비율"),
        ("parkinson_vol_*", "고가/저가 범위를 활용한 Parkinson 변동성"),
        ("garman_klass_vol_*", "OHLC를 활용한 Garman-Klass 변동성"),
        ("sma_20/50/200", "이동평균"),
        ("dist_close_sma_*_pct", "종가가 해당 이동평균 대비 얼마나 위/아래인지"),
        ("rsi_14", "RSI 14"),
        ("macd_12_26/macd_signal_9/macd_hist", "MACD 지표"),
        ("boll_z_20", "20일 볼린저 기준 종가 Z-score"),
        ("breakout_*d_high", "이전 N일 고점 돌파 여부"),
        ("breakdown_*d_low", "이전 N일 저점 이탈 여부"),
        ("running_high_close", "기간 중 누적 최고 수정종가"),
        ("drawdown_from_ath", "누적 고점 대비 낙폭"),
        ("max_drawdown_*d", "해당 롤링 기간의 최대낙폭"),
        ("dollar_volume", "종가 × 거래량"),
        ("volume_ratio_20", "당일 거래량 / 20일 평균 거래량"),
        ("amihud_daily", "절대수익률 / 달러거래대금. 낮을수록 유동성 양호"),
        ("beta_vs_spy_252d", "SPY 대비 252일 롤링 베타"),
        ("relative_return_vs_smh_63d", "SMH 대비 63일 상대수익률"),
        ("trend_regime", "50/200일 이동평균 기반 추세 국면"),
        ("momentum_signal", "20/63/126일 수익률 기반 복수기간 모멘텀 판정"),
        ("vol_regime", "20일 변동성의 252일 내 분위 기반 변동성 국면"),
        ("atr14_stop_long_k_*", "현재 종가 기준 ATR 손절가"),
        ("position_notional_for_*", "ATR2 손절 기준 계좌위험을 맞추기 위한 이론상 포지션 비중"),
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["column_pattern", "description_ko"])
        writer.writerows(rows)


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build detailed daily quant price dataset and charts for one requested symbol.")
    parser.add_argument("--symbol-stooq", default=DEFAULT_SYMBOL_STOOQ, help="Stooq ticker, default: tsm.us")
    parser.add_argument("--symbol-yahoo", default=DEFAULT_SYMBOL_YAHOO, help="Yahoo ticker, default: TSM")
    parser.add_argument("--start", default="2016-05-12", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date YYYY-MM-DD")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--preferred-source", choices=["stooq", "yahoo"], default="stooq", help="Primary data source")
    parser.add_argument("--events", default="tsm_events_seed.csv", help="Event CSV path")
    parser.add_argument("--listing-currency", default="", help="Listing/native price currency, inferred from ticker when empty.")
    parser.add_argument("--display-currency", default="", help="Dashboard display currency, defaults to listing currency.")
    parser.add_argument("--engine-currency", default="USD", help="Canonical engine/model currency.")
    parser.add_argument("--fx-pair", default="", help="Yahoo FX pair used for native->engine conversion, e.g. KRW=X.")
    parser.add_argument("--fx-rates", default="output/tsm_fx_rates_daily.csv", help="Daily FX rates CSV.")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip SPY/SMH/QQQ benchmark downloads")
    parser.add_argument("--skip-charts", action="store_true", help="Skip PNG chart generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    dirs = ensure_outdirs(outdir)

    print(f"Downloading {args.symbol_stooq}/{args.symbol_yahoo} from {args.start} to {args.end}...")
    source_info = download_prices(
        stooq_symbol=args.symbol_stooq,
        yahoo_symbol=args.symbol_yahoo,
        start=args.start,
        end=args.end,
        preferred_source=args.preferred_source,
    )
    profile = currency_profile(
        args.symbol_stooq,
        args.symbol_yahoo,
        listing_currency=args.listing_currency,
        display_currency=args.display_currency,
        engine_currency=args.engine_currency,
        fx_pair=args.fx_pair,
    )
    fx_rates = pd.DataFrame()
    if requires_fx_conversion(profile.listing_currency, profile.engine_currency):
        ensure_fx_rate_file(args.fx_rates, [profile.fx_pair], start=args.start, end=args.end)
        fx_rates = load_fx_rates(args.fx_rates, profile.fx_pair)
    raw = normalize_ohlcv_to_engine_currency(
        source_info.df,
        symbol=args.symbol_stooq,
        symbol_yahoo=args.symbol_yahoo,
        listing_currency=profile.listing_currency,
        display_currency=profile.display_currency,
        engine_currency=profile.engine_currency,
        fx_pair=profile.fx_pair,
        fx_rates=fx_rates,
    )
    raw_path = outdir / "tsm_daily_10y_raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"Raw data saved: {raw_path}")

    benchmarks: Dict[str, pd.DataFrame] = {}
    if not args.skip_benchmarks:
        for bname, bstooq in DEFAULT_BENCHMARKS.items():
            try:
                bres = download_prices(
                    stooq_symbol=bstooq,
                    yahoo_symbol=bname,
                    start=args.start,
                    end=args.end,
                    preferred_source=args.preferred_source,
                )
                benchmarks[bname] = bres.df[["date", "adj_close"]].copy()
                print(f"Benchmark {bname} downloaded from {bres.source}.")
            except Exception as e:
                print(f"WARNING: benchmark {bname} download failed: {e}", file=sys.stderr)

    enriched = enrich_prices(raw, ticker=args.symbol_yahoo, benchmarks=benchmarks)
    enriched_path = outdir / "tsm_daily_10y_enriched.csv"
    enriched.to_csv(enriched_path, index=False, encoding="utf-8-sig")
    print(f"Enriched daily data saved: {enriched_path}")

    # Event impact.
    event_path = Path(args.events)
    if not event_path.exists():
        # Also try path relative to script directory.
        script_event_path = Path(__file__).resolve().parent / args.events
        if script_event_path.exists():
            event_path = script_event_path
    events = load_events(event_path)
    event_impact = compute_event_impact(enriched, events)
    event_impact_path = outdir / "tsm_event_impact_10y.csv"
    event_impact.to_csv(event_impact_path, index=False, encoding="utf-8-sig")
    print(f"Event impact saved: {event_impact_path}")

    summary = compute_summary(enriched, source_info)
    summary_path = outdir / "tsm_daily_10y_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Summary saved: {summary_path}")

    write_column_dictionary(outdir / "tsm_daily_columns_dictionary.csv")
    print(f"Column dictionary saved: {outdir / 'tsm_daily_columns_dictionary.csv'}")

    if not args.skip_charts:
        make_charts(enriched, event_impact, dirs["charts"])
        print(f"Charts saved under: {dirs['charts']}")

    print("Done.")


if __name__ == "__main__":
    main()

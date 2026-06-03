#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build free/official external daily features for the TSM prediction stack.

The output is keyed by symbol/date and is safe to left-join into the local or
pooled prediction feature matrices. TSMC monthly revenue features are only
available after their release date, never on the target revenue month itself.

Outputs:
- tsm_external_daily_features.csv
- tsm_external_feature_schema.csv
"""

from __future__ import annotations

import argparse
import calendar
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
PEER_SYMBOLS = {"TSM", "NVDA", "AMD", "AVGO", "ASML", "AMAT", "LRCX", "KLAC", "MU", "QCOM", "005930.KS", "000660.KS"}
BENCHMARK_YAHOO = {"QQQ": "QQQ", "SPY": "SPY", "USDTWD": "TWD=X"}
TSMC_MONTHLY_REVENUE_URL = "https://investor.tsmc.com/english/monthly-revenue/{year}"


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return strip_bom_columns(pd.read_csv(path, **kwargs))


def read_universe(path: Path) -> pd.DataFrame:
    if path.exists():
        config = read_csv(path)
    else:
        config = pd.DataFrame(
            [
                {"symbol": "TSM", "symbol_group": "foundry", "enriched": "output/universe/TSM/tsm_daily_10y_enriched.csv"},
                {"symbol": "NVDA", "symbol_group": "ai_accelerator", "enriched": "output/universe/NVDA/tsm_daily_10y_enriched.csv"},
                {"symbol": "AMD", "symbol_group": "ai_accelerator", "enriched": "output/universe/AMD/tsm_daily_10y_enriched.csv"},
                {"symbol": "AVGO", "symbol_group": "ai_accelerator", "enriched": "output/universe/AVGO/tsm_daily_10y_enriched.csv"},
                {"symbol": "ASML", "symbol_group": "semicap", "enriched": "output/universe/ASML/tsm_daily_10y_enriched.csv"},
                {"symbol": "AMAT", "symbol_group": "semicap", "enriched": "output/universe/AMAT/tsm_daily_10y_enriched.csv"},
                {"symbol": "LRCX", "symbol_group": "semicap", "enriched": "output/universe/LRCX/tsm_daily_10y_enriched.csv"},
                {"symbol": "KLAC", "symbol_group": "semicap", "enriched": "output/universe/KLAC/tsm_daily_10y_enriched.csv"},
                {"symbol": "MU", "symbol_group": "memory", "enriched": "output/universe/MU/tsm_daily_10y_enriched.csv"},
                {"symbol": "005930.KS", "symbol_group": "memory_foundry_idm", "enriched": "output/universe/005930.KS/tsm_daily_10y_enriched.csv"},
                {"symbol": "000660.KS", "symbol_group": "memory_storage", "enriched": "output/universe/000660.KS/tsm_daily_10y_enriched.csv"},
                {"symbol": "QCOM", "symbol_group": "semiconductor", "enriched": "output/universe/QCOM/tsm_daily_10y_enriched.csv"},
                {"symbol": "SMH", "symbol_group": "semiconductor_etf", "enriched": "output/universe/SMH/tsm_daily_10y_enriched.csv"},
                {"symbol": "SOXX", "symbol_group": "semiconductor_etf", "enriched": "output/universe/SOXX/tsm_daily_10y_enriched.csv"},
            ]
        )
    if "enriched" not in config.columns and "data_outdir" in config.columns:
        config["enriched"] = config["data_outdir"].astype(str).str.rstrip("/") + "/tsm_daily_10y_enriched.csv"
    if "symbol_group" not in config.columns:
        config["symbol_group"] = "semiconductor"
    return config[["symbol", "symbol_group", "enriched"]].copy()


def fetch_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def html_to_lines(text: str) -> list[str]:
    text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", text)
    text = re.sub(r"(?is)<br\s*/?>|</p>|</div>|</tr>|</td>|</th>|</li>|</h[1-6]>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    lines = [re.sub(r"\s+", " ", html.unescape(line)).strip() for line in text.splitlines()]
    return [line for line in lines if line]


def parse_tsmc_monthly_revenue_html(year: int, text: str) -> pd.DataFrame:
    lines = html_to_lines(text)
    rows: list[dict] = []
    for idx, line in enumerate(lines):
        key = line.lower().strip(".")
        if key not in MONTH_MAP:
            continue
        payload = " ".join(lines[idx + 1 : idx + 5])
        match = re.search(r"([0-9][0-9,]*)\s+(-?[0-9]+(?:\.[0-9]+)?)%", payload)
        if not match:
            continue
        month = MONTH_MAP[key]
        revenue = float(match.group(1).replace(",", ""))
        yoy = float(match.group(2))
        period_month = pd.Timestamp(year=year, month=month, day=calendar.monthrange(year, month)[1])
        release_date = period_month + pd.Timedelta(days=10)
        rows.append(
            {
                "revenue_period_month": period_month,
                "tsmc_monthly_revenue_ntd_m": revenue,
                "tsmc_monthly_revenue_yoy_pct": yoy,
                "revenue_release_date": release_date.normalize(),
                "tsmc_revenue_source": "TSMC_OFFICIAL_MONTHLY_REVENUE",
            }
        )
    return pd.DataFrame(rows)


def parse_month_year(value: str) -> tuple[int | None, int | None]:
    lower = str(value).lower()
    month = None
    for token, month_no in MONTH_MAP.items():
        if re.search(rf"\b{re.escape(token)}[a-z.]*\b", lower):
            month = month_no
            break
    year_match = re.search(r"\b(20[0-9]{2}|19[0-9]{2})\b", lower)
    year = int(year_match.group(1)) if year_match else None
    return year, month


def revenue_release_overrides(news_events: Path) -> pd.DataFrame:
    if not news_events.exists():
        return pd.DataFrame()
    events = read_csv(news_events, parse_dates=["event_date", "available_for_signal_date"])
    if "cause_type" not in events.columns:
        return pd.DataFrame()
    part = events[events["cause_type"].astype(str).eq("monthly_revenue")].copy()
    rows = []
    for _, row in part.iterrows():
        year, month = parse_month_year(row.get("event_name", ""))
        if year is None or month is None:
            continue
        period = pd.Timestamp(year=year, month=month, day=calendar.monthrange(year, month)[1])
        release = row.get("available_for_signal_date")
        if pd.isna(release):
            release = row.get("event_date")
        if pd.notna(release):
            rows.append({"revenue_period_month": period, "revenue_release_date": pd.Timestamp(release).normalize()})
    return pd.DataFrame(rows)


def build_tsmc_revenue_releases(start_year: int, end_year: int, skip_web: bool, news_events: Path) -> tuple[pd.DataFrame, str]:
    parts = []
    status = "OK"
    if not skip_web:
        for year in range(start_year, end_year + 1):
            try:
                text = fetch_text(TSMC_MONTHLY_REVENUE_URL.format(year=year))
                parsed = parse_tsmc_monthly_revenue_html(year, text)
                if not parsed.empty:
                    parts.append(parsed)
                time.sleep(0.05)
            except Exception:
                status = "TSMC_REVENUE_FETCH_PARTIAL_OR_FAILED"
    revenue = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if revenue.empty:
        return revenue, "NO_TSMC_REVENUE"
    overrides = revenue_release_overrides(news_events)
    if not overrides.empty:
        revenue = revenue.merge(overrides, on="revenue_period_month", how="left", suffixes=("", "_override"))
        revenue["revenue_release_date"] = revenue["revenue_release_date_override"].combine_first(revenue["revenue_release_date"])
        revenue = revenue.drop(columns=[c for c in revenue.columns if c.endswith("_override")])
    revenue = revenue.sort_values("revenue_period_month").drop_duplicates("revenue_period_month", keep="last")
    revenue["tsmc_monthly_revenue_mom_pct"] = revenue["tsmc_monthly_revenue_ntd_m"].pct_change() * 100.0
    revenue["tsmc_revenue_yoy_3m_avg_pct"] = revenue["tsmc_monthly_revenue_yoy_pct"].rolling(3, min_periods=1).mean()
    rolling_12 = revenue["tsmc_monthly_revenue_ntd_m"].rolling(12, min_periods=12).sum()
    revenue["tsmc_revenue_12m_cumulative_yoy_pct"] = (rolling_12 / rolling_12.shift(12) - 1.0) * 100.0
    return revenue, status


def fetch_yahoo_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    period1 = int(pd.Timestamp(start).timestamp())
    period2 = int((pd.Timestamp(end) + pd.Timedelta(days=3)).timestamp())
    query = urllib.parse.urlencode({"period1": period1, "period2": period2, "interval": "1d", "events": "history"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?{query}"
    raw = json.loads(fetch_text(url))
    result = raw.get("chart", {}).get("result", [])
    if not result:
        return pd.DataFrame()
    item = result[0]
    timestamps = item.get("timestamp", [])
    close = item.get("indicators", {}).get("quote", [{}])[0].get("close", [])
    frame = pd.DataFrame({"date": pd.to_datetime(timestamps, unit="s").normalize(), "close": close})
    return frame.dropna().sort_values("date").reset_index(drop=True)


def load_enriched_frames(config: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, row in config.iterrows():
        path = Path(row["enriched"])
        if not path.exists() and str(row["symbol"]).upper() == "TSM" and Path("output/tsm_daily_10y_enriched.csv").exists():
            path = Path("output/tsm_daily_10y_enriched.csv")
        if not path.exists():
            continue
        frame = read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        frame.insert(0, "symbol", str(row["symbol"]).upper())
        frame.insert(1, "symbol_group", str(row.get("symbol_group", "semiconductor")))
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def add_revenue_features(base: pd.DataFrame, revenue: pd.DataFrame, revenue_status: str) -> pd.DataFrame:
    out = base.sort_values("date").copy()
    if revenue.empty:
        for col in [
            "tsmc_monthly_revenue_ntd_m",
            "tsmc_monthly_revenue_yoy_pct",
            "tsmc_monthly_revenue_mom_pct",
            "tsmc_revenue_yoy_3m_avg_pct",
            "tsmc_revenue_12m_cumulative_yoy_pct",
            "days_since_tsmc_revenue_release",
        ]:
            out[col] = np.nan
        out["tsmc_revenue_source"] = "NO_TSMC_REVENUE"
        return out
    rev = revenue.sort_values("revenue_release_date").copy()
    merged = pd.merge_asof(out.sort_values("date"), rev, left_on="date", right_on="revenue_release_date", direction="backward")
    merged["days_since_tsmc_revenue_release"] = (merged["date"] - merged["revenue_release_date"]).dt.days
    merged["tsmc_revenue_source"] = merged["tsmc_revenue_source"].fillna(revenue_status)
    return merged.drop(columns=[c for c in ["revenue_period_month", "revenue_release_date"] if c in merged.columns])


def add_earnings_features(base: pd.DataFrame, news_events: Path) -> pd.DataFrame:
    out = base.sort_values("date").copy()
    if not news_events.exists():
        out["tsmc_earnings_pre_5d_window"] = False
        out["tsmc_earnings_post_5d_window"] = False
        out["tsmc_earnings_event_day"] = False
        out["days_since_tsmc_earnings"] = np.nan
        return out
    events = read_csv(news_events, parse_dates=["event_date", "available_for_signal_date"])
    earnings = events[events.get("cause_type", pd.Series(dtype=str)).astype(str).eq("earnings_results")].copy()
    if earnings.empty:
        out["tsmc_earnings_pre_5d_window"] = False
        out["tsmc_earnings_post_5d_window"] = False
        out["tsmc_earnings_event_day"] = False
        out["days_since_tsmc_earnings"] = np.nan
        return out
    earnings["signal_date"] = earnings["available_for_signal_date"].combine_first(earnings["event_date"]).dt.normalize()
    event_dates = pd.Series(sorted(earnings["signal_date"].dropna().unique()))
    prev_event = pd.merge_asof(out[["date"]].sort_values("date"), pd.DataFrame({"signal_date": event_dates}), left_on="date", right_on="signal_date", direction="backward")
    next_event = pd.merge_asof(out[["date"]].sort_values("date"), pd.DataFrame({"signal_date": event_dates}), left_on="date", right_on="signal_date", direction="forward")
    days_since = (out["date"].to_numpy(dtype="datetime64[ns]") - prev_event["signal_date"].to_numpy(dtype="datetime64[ns]")) / np.timedelta64(1, "D")
    days_until = (next_event["signal_date"].to_numpy(dtype="datetime64[ns]") - out["date"].to_numpy(dtype="datetime64[ns]")) / np.timedelta64(1, "D")
    out["days_since_tsmc_earnings"] = days_since
    out["tsmc_earnings_event_day"] = pd.Series(days_since, index=out.index).eq(0)
    out["tsmc_earnings_post_5d_window"] = pd.Series(days_since, index=out.index).between(0, 7, inclusive="both")
    out["tsmc_earnings_pre_5d_window"] = pd.Series(days_until, index=out.index).between(0, 7, inclusive="both")
    return out


def benchmark_feature_frame(local_frames: pd.DataFrame, skip_web: bool, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.DataFrame({"date": sorted(local_frames["date"].dropna().unique())})
    out = dates.copy()
    for symbol, prefix in [("SMH", "market_smh"), ("SOXX", "market_soxx")]:
        part = local_frames[local_frames["symbol"].eq(symbol)].copy()
        if part.empty:
            continue
        cols = ["date"]
        rename = {}
        for window in [5, 20, 60]:
            col = f"return_{window}d"
            if col in part.columns:
                cols.append(col)
                rename[col] = f"{prefix}_return_{window}d"
        out = out.merge(part[cols].rename(columns=rename), on="date", how="left")
    if not skip_web:
        for label, yahoo_symbol in [("qqq", "QQQ"), ("spy", "SPY")]:
            try:
                frame = fetch_yahoo_close(yahoo_symbol, start, end)
                merge_cols = ["date"]
                for window in [5, 20, 60]:
                    name = f"market_{label}_return_{window}d"
                    frame[name] = frame["close"].pct_change(window) * 100.0
                    merge_cols.append(name)
                out = out.merge(frame[merge_cols], on="date", how="left")
            except Exception:
                for window in [5, 20, 60]:
                    out[f"market_{label}_return_{window}d"] = np.nan
    return out


def fx_feature_frame(dates: pd.DataFrame, skip_web: bool, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    out = dates.copy()
    out["fx_usdtwd_return_20d"] = np.nan
    out["fx_usdtwd_return_60d"] = np.nan
    out["fx_data_available"] = False
    if skip_web:
        return out
    try:
        fx = fetch_yahoo_close(BENCHMARK_YAHOO["USDTWD"], start, end)
        if fx.empty:
            return out
        fx["fx_usdtwd_return_20d"] = fx["close"].pct_change(20) * 100.0
        fx["fx_usdtwd_return_60d"] = fx["close"].pct_change(60) * 100.0
        fx["fx_data_available"] = True
        return out[["date"]].merge(fx[["date", "fx_usdtwd_return_20d", "fx_usdtwd_return_60d", "fx_data_available"]], on="date", how="left").fillna({"fx_data_available": False})
    except Exception:
        return out


def vix_feature_frame(dates: pd.DataFrame, skip_web: bool, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """CBOE VIX implied-volatility regime features (leak-free: each row uses the
    VIX close known at end of that date). Research-backed short-horizon
    directional signal for equities. Values are NaN when web data is skipped or
    unavailable, so models simply ignore the block instead of breaking."""
    out = dates[["date"]].copy()
    vix_cols = [
        "vix_level",
        "vix_change_1d_pct",
        "vix_change_5d_pct",
        "vix_vs_ma20_pct",
        "vix_zscore_60d",
    ]
    for col in vix_cols:
        out[col] = np.nan
    out["vix_data_available"] = False
    if skip_web:
        return out
    try:
        vix = fetch_yahoo_close("^VIX", start, end)
        if vix.empty:
            return out
        vix = vix.sort_values("date").reset_index(drop=True)
        level = pd.to_numeric(vix["close"], errors="coerce")
        vix["vix_level"] = level
        vix["vix_change_1d_pct"] = level.pct_change(1) * 100.0
        vix["vix_change_5d_pct"] = level.pct_change(5) * 100.0
        ma20 = level.rolling(20, min_periods=10).mean()
        vix["vix_vs_ma20_pct"] = (level / ma20 - 1.0) * 100.0
        roll_mean = level.rolling(60, min_periods=20).mean()
        roll_std = level.rolling(60, min_periods=20).std(ddof=0)
        vix["vix_zscore_60d"] = (level - roll_mean) / roll_std.replace(0.0, np.nan)
        vix["vix_data_available"] = True
        keep = ["date", *vix_cols, "vix_data_available"]
        return out[["date"]].merge(vix[keep], on="date", how="left").fillna({"vix_data_available": False})
    except Exception:
        return out


def add_cross_symbol_features(base: pd.DataFrame, local_frames: pd.DataFrame) -> pd.DataFrame:
    cols = ["symbol", "symbol_group", "date", "return_20d", "return_60d", "return_126d", "close", "sma_50", "sma_200"]
    available = [c for c in cols if c in local_frames.columns]
    slim = local_frames[available].copy()
    peer = slim[slim["symbol"].isin(PEER_SYMBOLS)].copy()
    for window in [20, 60, 126]:
        col = f"return_{window}d"
        if col in peer.columns:
            peer[f"peer_return_{window}d_rank"] = pd.to_numeric(peer[col], errors="coerce").groupby(peer["date"]).rank(pct=True)
            peer[f"peer_group_return_{window}d_rank"] = pd.to_numeric(peer[col], errors="coerce").groupby([peer["date"], peer["symbol_group"]]).rank(pct=True)
    if {"close", "sma_50"}.issubset(peer.columns):
        peer["universe_external_above_sma50_ratio"] = (pd.to_numeric(peer["close"], errors="coerce") >= pd.to_numeric(peer["sma_50"], errors="coerce")).astype(float).groupby(peer["date"]).transform("mean")
    if {"close", "sma_200"}.issubset(peer.columns):
        peer["universe_external_above_sma200_ratio"] = (pd.to_numeric(peer["close"], errors="coerce") >= pd.to_numeric(peer["sma_200"], errors="coerce")).astype(float).groupby(peer["date"]).transform("mean")
    keep = [
        c
        for c in [
            "symbol",
            "date",
            "peer_return_20d_rank",
            "peer_return_60d_rank",
            "peer_return_126d_rank",
            "peer_group_return_20d_rank",
            "universe_external_above_sma50_ratio",
            "universe_external_above_sma200_ratio",
        ]
        if c in peer.columns
    ]
    out = base.merge(peer[keep], on=["symbol", "date"], how="left")
    return out


def build_external_features(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = read_universe(Path(args.universe_config))
    local_frames = load_enriched_frames(config)
    if local_frames.empty:
        raise ValueError("no enriched universe files available")
    local_frames["date"] = pd.to_datetime(local_frames["date"], errors="coerce").dt.normalize()
    start = pd.to_datetime(args.start) if args.start else local_frames["date"].min()
    end = pd.to_datetime(args.end) if args.end else local_frames["date"].max()
    local_frames = local_frames[(local_frames["date"] >= start) & (local_frames["date"] <= end)].copy()
    base = local_frames[["symbol", "symbol_group", "date"]].drop_duplicates().sort_values(["symbol", "date"]).reset_index(drop=True)
    revenue, revenue_status = build_tsmc_revenue_releases(int(args.revenue_start_year), int(args.revenue_end_year), bool(args.skip_web), Path(args.news_events))
    base = add_revenue_features(base, revenue, revenue_status)
    base = add_earnings_features(base, Path(args.news_events))
    bench = benchmark_feature_frame(local_frames, bool(args.skip_web), start, end)
    base = base.merge(bench, on="date", how="left")
    fx = fx_feature_frame(pd.DataFrame({"date": sorted(local_frames["date"].dropna().unique())}), bool(args.skip_web), start, end)
    base = base.merge(fx, on="date", how="left")
    vix = vix_feature_frame(pd.DataFrame({"date": sorted(local_frames["date"].dropna().unique())}), bool(args.skip_web), start, end)
    base = base.merge(vix, on="date", how="left")
    base = add_cross_symbol_features(base, local_frames)
    if {"return_20d", "symbol", "date"}.issubset(local_frames.columns):
        own = local_frames[["symbol", "date", "return_20d"]].rename(columns={"return_20d": "_own_return_20d"})
        base = base.merge(own, on=["symbol", "date"], how="left")
        if "market_qqq_return_20d" in base.columns:
            base["external_relative_return_vs_qqq_20d"] = base["_own_return_20d"] - base["market_qqq_return_20d"]
        if "market_spy_return_20d" in base.columns:
            base["external_relative_return_vs_spy_20d"] = base["_own_return_20d"] - base["market_spy_return_20d"]
        base = base.drop(columns=["_own_return_20d"])
    base["external_feature_status"] = np.where(base["tsmc_monthly_revenue_ntd_m"].notna(), "OK", revenue_status)
    schema = build_schema(base)
    return base.sort_values(["symbol", "date"]).reset_index(drop=True), schema


def build_schema(features: pd.DataFrame) -> pd.DataFrame:
    bool_cols = {"tsmc_earnings_pre_5d_window", "tsmc_earnings_post_5d_window", "tsmc_earnings_event_day", "fx_data_available"}
    categorical_cols = {"external_feature_status", "tsmc_revenue_source"}
    rows = []
    for col in features.columns:
        if col in {"symbol", "date"}:
            role = "key"
        elif col in bool_cols:
            role = "bool_feature"
        elif col in categorical_cols:
            role = "categorical_feature"
        elif pd.api.types.is_numeric_dtype(features[col]):
            role = "numeric_feature"
        else:
            role = "metadata"
        rows.append(
            {
                "column": col,
                "role": role,
                "dtype": str(features[col].dtype),
                "missing_rate": float(features[col].isna().mean()) if len(features) else np.nan,
                "source": "semiconductor_official_or_free_public_data",
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build free/official external features for TSM prediction.")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--universe-config", default="config/semiconductor_universe.csv")
    parser.add_argument("--news-events", default="output/tsm_news_events_normalized.csv")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--revenue-start-year", type=int, default=2016)
    parser.add_argument("--revenue-end-year", type=int, default=pd.Timestamp.today().year)
    parser.add_argument("--skip-web", action="store_true", help="Use only local files; web-sourced features become unavailable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    features, schema = build_external_features(args)
    features.to_csv(outdir / "tsm_external_daily_features.csv", index=False)
    schema.to_csv(outdir / "tsm_external_feature_schema.csv", index=False)
    print("completed: external features =", (outdir / "tsm_external_daily_features.csv").resolve())
    print(schema.to_string(index=False))


if __name__ == "__main__":
    main()

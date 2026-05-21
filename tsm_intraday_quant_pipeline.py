#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSM intraday downloader for minute/hour bars with the same raw schema as the
daily CSV.

Default mode targets 1-minute bars because "분봉" usually means minute candles.
Public Yahoo chart data cannot provide 10 years of minute bars:

- 1m: only about the last 8 calendar days
- 2m/5m/15m/30m: must be within the last 60 days
- 60m/1h: must be within the last 730 days

This script records that source audit, downloads the maximum public Yahoo window
available for the requested interval, and writes daily-compatible raw/enriched
CSV files plus a compact report.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from tsm_daily_quant_pipeline import DownloadResult, enrich_prices, pct


DEFAULT_SYMBOL = "TSM"
DEFAULT_START = "2016-05-12"
DEFAULT_OUTDIR = "output"
RAW_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
    "data_source",
    "data_quality_note",
]
YAHOO_PUBLIC_LIMIT_DAYS = {
    "1m": 7,
    "2m": 59,
    "5m": 59,
    "15m": 59,
    "30m": 59,
    "60m": 729,
    "1h": 729,
}
YAHOO_LIMIT_LABEL = {
    "1m": "Yahoo public 1m requests allow only about 8 calendar days; 7 days is used to avoid inclusive end-time rejection.",
    "2m": "Yahoo public 2m requests must be within the last 60 days; 59 days is used to avoid inclusive end-time rejection.",
    "5m": "Yahoo public 5m requests must be within the last 60 days; 59 days is used to avoid inclusive end-time rejection.",
    "15m": "Yahoo public 15m requests must be within the last 60 days; 59 days is used to avoid inclusive end-time rejection.",
    "30m": "Yahoo public 30m requests must be within the last 60 days; 59 days is used to avoid inclusive end-time rejection.",
    "60m": "Yahoo public 60m/1h requests must be within the last 730 days; 729 days is used to avoid inclusive end-time rejection.",
    "1h": "Yahoo public 60m/1h requests must be within the last 730 days; 729 days is used to avoid inclusive end-time rejection.",
}
INTERVAL_LABEL = {
    "1m": "minute",
    "2m": "2min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "hourly",
    "1h": "hourly",
}


@dataclass
class IntradayDownloadResult:
    source_info: DownloadResult
    requested_start: str
    requested_end: str
    provider: str
    interval: str
    complete_requested_coverage: bool
    provider_limit_note: str


def ensure_outdirs(outdir: Path) -> Dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    charts_dir = outdir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    return {"outdir": outdir, "charts": charts_dir}


def parse_utc_day(value: str) -> datetime:
    return pd.to_datetime(value).to_pydatetime().replace(tzinfo=timezone.utc)


def ymd(value: datetime) -> str:
    return value.date().isoformat()


def unix_seconds(value: str) -> int:
    return int(parse_utc_day(value).timestamp())


def yahoo_chart_url(symbol: str, start: str, end: str, interval: str) -> str:
    period1 = unix_seconds(start)
    period2 = int((parse_utc_day(end) + timedelta(days=1)).timestamp())
    query = urlencode({
        "period1": period1,
        "period2": period2,
        "interval": interval,
        "events": "history|div|split",
    })
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"


def request_json(url: str, timeout: int = 30) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def check_yahoo_full_range_limit(symbol: str, start: str, end: str, interval: str) -> str:
    url = yahoo_chart_url(symbol, start, end, interval)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        payload = response.json()
        error = payload.get("chart", {}).get("error")
        if error:
            return f"HTTP {response.status_code}: {error.get('description', error)}"
        result = payload.get("chart", {}).get("result") or []
        rows = len(result[0].get("timestamp", [])) if result else 0
        return f"HTTP {response.status_code}: returned {rows} rows"
    except Exception as exc:
        return f"Yahoo full-range check failed: {exc}"


def fetch_stooq_evidence(interval: str) -> str:
    stooq_interval = interval.rstrip("m").replace("1h", "60")
    if interval == "60m":
        stooq_interval = "60"
    url = f"https://stooq.com/q/d/l/?s=tsm.us&i={stooq_interval}"
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        text = " ".join(response.text.strip().split())
        return f"HTTP {response.status_code}: {text[:240]}"
    except Exception as exc:
        return f"Stooq check failed: {exc}"


def yahoo_available_start(requested_start: str, requested_end: str, interval: str) -> str:
    start_dt = parse_utc_day(requested_start)
    end_dt = parse_utc_day(requested_end)
    max_start = end_dt - timedelta(days=YAHOO_PUBLIC_LIMIT_DAYS[interval])
    return ymd(max(start_dt, max_start))


def download_yahoo_intraday(symbol: str, requested_start: str, requested_end: str, interval: str) -> IntradayDownloadResult:
    start = yahoo_available_start(requested_start, requested_end, interval)
    url = yahoo_chart_url(symbol, start, requested_end, interval)
    payload = request_json(url)
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(chart["error"])
    result = chart.get("result", [None])[0]
    if not result:
        raise RuntimeError("Yahoo chart JSON has no result")

    timestamps = result.get("timestamp", [])
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    if not timestamps or not quote:
        raise RuntimeError("Yahoo chart JSON has no timestamp/quote data")

    exchange_tz = result.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
    dt = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(exchange_tz).tz_localize(None)
    df = pd.DataFrame({
        "date": dt,
        "open": quote.get("open"),
        "high": quote.get("high"),
        "low": quote.get("low"),
        "close": quote.get("close"),
        "volume": quote.get("volume"),
    })
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Yahoo chart returned empty intraday OHLC data")

    df["adj_close"] = df["close"]
    df["data_source"] = f"yahoo_chart_{interval}"
    df["data_quality_note"] = (
        f"Yahoo chart JSON {interval} bars. date is America/New_York exchange-local timestamp; "
        f"intraday adjusted close is not supplied, so adj_close is set equal to close. "
        f"{YAHOO_LIMIT_LABEL[interval]}"
    )
    complete = pd.to_datetime(df["date"].iloc[0]).date() <= pd.to_datetime(requested_start).date()
    source = DownloadResult(
        df=df[RAW_COLUMNS].copy(),
        source=f"yahoo_chart_{interval}",
        url=url,
        note=f"Best public no-key {interval} coverage from Yahoo chart JSON.",
    )
    return IntradayDownloadResult(
        source_info=source,
        requested_start=requested_start,
        requested_end=requested_end,
        provider="yahoo",
        interval=interval,
        complete_requested_coverage=bool(complete),
        provider_limit_note=YAHOO_LIMIT_LABEL[interval],
    )


def observed_bars_per_session(df: pd.DataFrame) -> float:
    sessions = pd.to_datetime(df["date"]).dt.date
    counts = sessions.value_counts()
    if counts.empty:
        return 1.0
    fullish = counts[counts >= counts.quantile(0.50)]
    return float(fullish.median() if not fullish.empty else counts.median())


def annualization_bars_per_year(df: pd.DataFrame) -> float:
    return observed_bars_per_session(df) * 252.0


def rescale_annualized_columns(enriched: pd.DataFrame, bars_per_year: float) -> pd.DataFrame:
    out = enriched.copy()
    scale = math.sqrt(bars_per_year / 252.0)
    ann_cols = [c for c in out.columns if c.endswith("_ann") or ("_vol_" in c and c.endswith("d_ann"))]
    ann_cols += [c for c in out.columns if c.startswith("parkinson_vol_") or c.startswith("garman_klass_vol_")]
    for col in sorted(set(ann_cols)):
        if col in out.columns:
            out[col] = out[col] * scale
    return out


def compute_summary(df: pd.DataFrame, result: IntradayDownloadResult, bars_per_year: float) -> pd.DataFrame:
    ret = df["adj_close_change_pct"].dropna()
    log_ret = df["log_return"].dropna()
    start_ts = pd.to_datetime(df["date"].iloc[0])
    end_ts = pd.to_datetime(df["date"].iloc[-1])
    start_price = float(df["adj_close"].iloc[0])
    end_price = float(df["adj_close"].iloc[-1])
    years = max((end_ts - start_ts).total_seconds() / (365.25 * 24 * 3600), np.nan)
    total_return = end_price / start_price - 1
    cagr = (end_price / start_price) ** (1 / years) - 1 if years and years > 0 else np.nan
    ann_vol = float(log_ret.std(ddof=0) * math.sqrt(bars_per_year)) if len(log_ret) > 1 else np.nan
    downside = log_ret[log_ret < 0]
    downside_ann = float(downside.std(ddof=0) * math.sqrt(bars_per_year)) if len(downside) > 1 else np.nan
    mdd = float(df["drawdown_from_ath"].min())
    requested_start_date = pd.to_datetime(result.requested_start).date()
    actual_start_date = start_ts.date()
    coverage_gap_days = max((actual_start_date - requested_start_date).days, 0)
    bar_name = INTERVAL_LABEL[result.interval]

    rows = [
        ("ticker", df["ticker"].iloc[0]),
        ("bar_interval", result.interval),
        ("bar_label", bar_name),
        ("data_source", result.source_info.source),
        ("data_url", result.source_info.url),
        ("data_note", result.source_info.note),
        ("provider_limit_note", result.provider_limit_note),
        ("requested_start_date", result.requested_start),
        ("requested_end_date", result.requested_end),
        ("complete_requested_coverage", result.complete_requested_coverage),
        ("coverage_gap_start_days", coverage_gap_days),
        ("start_timestamp", start_ts.isoformat(sep=" ")),
        ("end_timestamp", end_ts.isoformat(sep=" ")),
        ("bars", len(df)),
        ("zero_volume_bars", int((df["volume"].fillna(0) <= 0).sum())),
        ("bad_high_low_bars", int(df["is_bad_high_low"].sum()) if "is_bad_high_low" in df.columns else np.nan),
        ("observed_median_bars_per_session", observed_bars_per_session(df)),
        ("annualization_bars_per_year", bars_per_year),
        ("start_adj_close", start_price),
        ("end_adj_close", end_price),
        ("total_return_pct", pct(total_return)),
        ("cagr_pct", pct(cagr)),
        ("annualized_volatility_pct", pct(ann_vol)),
        ("max_drawdown_pct", pct(mdd)),
        ("sharpe_zero_rf", cagr / ann_vol if ann_vol and not np.isnan(ann_vol) and ann_vol != 0 else np.nan),
        ("sortino_zero_rf", cagr / downside_ann if downside_ann and not np.isnan(downside_ann) and downside_ann != 0 else np.nan),
        (f"best_{bar_name}_return_pct", pct(ret.max())),
        (f"worst_{bar_name}_return_pct", pct(ret.min())),
        (f"avg_{bar_name}_return_pct", pct(ret.mean())),
        (f"median_{bar_name}_return_pct", pct(ret.median())),
        ("positive_bar_ratio_pct", pct((ret > 0).mean())),
        ("negative_bar_ratio_pct", pct((ret < 0).mean())),
        (f"avg_{bar_name}_range_pct", pct(df["intraday_range_pct_prev_close"].mean())),
        (f"median_{bar_name}_range_pct", pct(df["intraday_range_pct_prev_close"].median())),
        ("latest_20bar_ann_vol_pct", pct(df["vol_20d_ann"].iloc[-1])),
        ("latest_63bar_ann_vol_pct", pct(df["vol_63d_ann"].iloc[-1])),
        ("latest_252bar_ann_vol_pct", pct(df["vol_252d_ann"].iloc[-1])),
        ("latest_atr14_pct", pct(df["atr_14_pct"].iloc[-1])),
        ("latest_drawdown_from_ath_pct", pct(df["drawdown_from_ath"].iloc[-1])),
        ("latest_trend_regime", df["trend_regime"].iloc[-1]),
        ("latest_momentum_signal", df["momentum_signal"].iloc[-1]),
        ("latest_vol_regime", df["vol_regime"].iloc[-1]),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def write_source_audit(
    path: Path,
    result: IntradayDownloadResult,
    yahoo_full_range_evidence: str,
    stooq_evidence: str,
) -> None:
    df = result.source_info.df
    actual_start = pd.to_datetime(df["date"].iloc[0]).isoformat(sep=" ")
    actual_end = pd.to_datetime(df["date"].iloc[-1]).isoformat(sep=" ")
    polygon_key = "present" if os.getenv("POLYGON_API_KEY") else "missing"
    alpha_key = "present" if os.getenv("ALPHAVANTAGE_API_KEY") else "missing"
    eodhd_key = "present" if os.getenv("EODHD_API_KEY") else "missing"
    rows = [
        {
            "provider": "yahoo_chart",
            "interval": result.interval,
            "auth_status": "no_key_required",
            "request_status": "downloaded",
            "supports_complete_requested_10y": result.complete_requested_coverage,
            "actual_start": actual_start,
            "actual_end": actual_end,
            "rows": len(df),
            "evidence": yahoo_full_range_evidence,
            "source_url": "https://query1.finance.yahoo.com/v8/finance/chart/TSM",
        },
        {
            "provider": "stooq_csv",
            "interval": result.interval,
            "auth_status": "apikey_required_by_response",
            "request_status": "not_downloaded",
            "supports_complete_requested_10y": "unverified_for_TSM_US_intraday",
            "actual_start": "",
            "actual_end": "",
            "rows": "",
            "evidence": stooq_evidence,
            "source_url": "https://stooq.com/q/d/l/?s=tsm.us",
        },
        {
            "provider": "polygon_flat_files",
            "interval": "1 minute",
            "auth_status": f"POLYGON_API_KEY_{polygon_key}",
            "request_status": "skipped",
            "supports_complete_requested_10y": "plan_required",
            "actual_start": "",
            "actual_end": "",
            "rows": "",
            "evidence": "Official flat-file docs list U.S. equity minute aggregates and plan history up to 10 years on Developer and all history on Advanced.",
            "source_url": "https://polygon.io/docs/flat-files/stocks/minute-aggregates/2020",
        },
        {
            "provider": "alpha_vantage",
            "interval": "1min/5min/15min/30min/60min",
            "auth_status": f"ALPHAVANTAGE_API_KEY_{alpha_key}",
            "request_status": "skipped",
            "supports_complete_requested_10y": "premium_key_required",
            "actual_start": "",
            "actual_end": "",
            "rows": "",
            "evidence": "Official docs state TIME_SERIES_INTRADAY has 20+ years of historical intraday depth, month=YYYY-MM support, and is a premium endpoint.",
            "source_url": "https://www.alphavantage.co/documentation/",
        },
        {
            "provider": "eodhd_intraday",
            "interval": "1m/5m/1h",
            "auth_status": f"EODHD_API_KEY_{eodhd_key}",
            "request_status": "skipped",
            "supports_complete_requested_10y": "key_or_plan_required",
            "actual_start": "",
            "actual_end": "",
            "rows": "",
            "evidence": "EODHD docs note intraday API range windows and historical availability; API token is required.",
            "source_url": "https://eodhd.com/financial-academy/how-to-get-stocks-data-examples/how-to-get-stocks-intraday-historical-data-on-python",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def make_charts(df: pd.DataFrame, charts_dir: Path, prefix: str, interval: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot(df["date"], df["close"], label=f"{interval} close", linewidth=0.8)
    for w in [20, 50, 200]:
        ax1.plot(df["date"], df[f"sma_{w}"], label=f"SMA {w} bars", linewidth=0.8)
    ax1.set_title(f"TSMC {interval} close with moving averages")
    ax1.set_ylabel("Price, USD")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.fill_between(df["date"], df["drawdown_from_ath"] * 100, 0, alpha=0.18)
    ax2.set_ylabel("Drawdown, %")
    fig.tight_layout()
    fig.savefig(charts_dir / f"{prefix}_price_ma_drawdown.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.hist(df["close_change_pct"].dropna() * 100, bins=100)
    ax.axvline(0, linewidth=1.0)
    ax.set_title(f"TSMC {interval} return distribution")
    ax.set_xlabel("Bar return, %")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(charts_dir / f"{prefix}_return_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot(df["date"], df["vol_20d_ann"] * 100, label="20-bar annualized vol", linewidth=0.8)
    ax1.plot(df["date"], df["vol_63d_ann"] * 100, label="63-bar annualized vol", linewidth=0.8)
    ax1.plot(df["date"], df["vol_252d_ann"] * 100, label="252-bar annualized vol", linewidth=0.8)
    ax1.set_ylabel("Annualized volatility, %")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["atr_14_pct"] * 100, label="ATR14 %", linewidth=0.8)
    ax2.set_ylabel("ATR14, % of close")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"TSMC {interval} rolling volatility and ATR")
    fig.tight_layout()
    fig.savefig(charts_dir / f"{prefix}_rolling_vol_atr.png", dpi=160)
    plt.close(fig)


def write_report(path: Path, summary: pd.DataFrame, audit: pd.DataFrame, prefix: str) -> None:
    metrics = dict(zip(summary["metric"], summary["value"]))
    bar_label = metrics.get("bar_label")
    lines = [
        f"# TSMC {metrics.get('bar_interval')} intraday 데이터 소스 점검 및 분석 요약",
        "",
        "## 결론",
        "",
        f"- 요청 범위: {metrics.get('requested_start_date')} ~ {metrics.get('requested_end_date')}",
        f"- 실제 확보 소스: `{metrics.get('data_source')}`",
        f"- 실제 확보 범위: {metrics.get('start_timestamp')} ~ {metrics.get('end_timestamp')}",
        f"- 확보한 봉 수: {metrics.get('bars')}",
        f"- 거래량 0 봉 수: {metrics.get('zero_volume_bars')}",
        f"- 요청한 10년 범위 완성 여부: `{metrics.get('complete_requested_coverage')}`",
        f"- 시작 구간 미확보 일수: {metrics.get('coverage_gap_start_days')}",
        "",
        "공개 Yahoo chart API만으로는 10년 분봉을 완성할 수 없습니다. 이 리포트는 실제 오류 응답과 공식 문서 기반 대체 공급처를 함께 기록하고, 인증 없이 확보 가능한 최대 범위를 일봉 raw와 같은 컬럼 스키마로 저장합니다.",
        "",
        "## 핵심 통계",
        "",
        f"- 전체 수익률: {float(metrics.get('total_return_pct')):.2f}%",
        f"- CAGR: {float(metrics.get('cagr_pct')):.2f}%",
        f"- 연율화 변동성: {float(metrics.get('annualized_volatility_pct')):.2f}%",
        f"- 최대 낙폭: {float(metrics.get('max_drawdown_pct')):.2f}%",
        f"- 최신 추세 국면: `{metrics.get('latest_trend_regime')}`",
        f"- 최신 변동성 국면: `{metrics.get('latest_vol_regime')}`",
        "",
        "## 생성 파일",
        "",
        f"- `output/{prefix}_available_raw.csv`",
        f"- `output/{prefix}_available_enriched.csv`",
        f"- `output/{prefix}_available_summary.csv`",
        f"- `output/{prefix}_10y_source_audit.csv`",
        f"- `output/{prefix}_data_report.md`",
        f"- `output/charts/{prefix}_price_ma_drawdown.png`",
        f"- `output/charts/{prefix}_return_distribution.png`",
        f"- `output/charts/{prefix}_rolling_vol_atr.png`",
        "",
        "## 주의",
        "",
        f"- `*_20d`, `*_63d`, `*_252d` 컬럼명은 스키마 호환을 위해 유지했지만, 이 파일에서는 20/63/252개 {bar_label} 봉 기준입니다.",
        "- Yahoo intraday에는 별도 조정종가가 없어 `adj_close = close`로 저장했습니다.",
        "- 10년 완성 분봉은 Polygon Developer 이상, Alpha Vantage premium, EODHD 같은 키 기반 공급처가 필요합니다.",
        "",
        "## 소스 감사",
        "",
    ]
    for _, row in audit.iterrows():
        lines.append(f"- {row['provider']} ({row['interval']}): {row['request_status']}, {row['supports_complete_requested_10y']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TSMC intraday OHLCV files with daily-compatible schema.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Ticker symbol, default: TSM")
    parser.add_argument("--start", default=DEFAULT_START, help="Requested start date YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), help="Requested end date YYYY-MM-DD")
    parser.add_argument("--interval", choices=sorted(YAHOO_PUBLIC_LIMIT_DAYS), default="1m", help="Yahoo interval")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--skip-charts", action="store_true", help="Skip chart generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    dirs = ensure_outdirs(outdir)
    label = INTERVAL_LABEL[args.interval]
    prefix = "tsm_minute" if args.interval == "1m" else f"tsm_{label}"

    yahoo_evidence = check_yahoo_full_range_limit(args.symbol, args.start, args.end, args.interval)
    stooq_evidence = fetch_stooq_evidence(args.interval)
    result = download_yahoo_intraday(args.symbol, args.start, args.end, args.interval)

    raw = result.source_info.df
    raw_path = outdir / f"{prefix}_available_raw.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"Raw intraday data saved: {raw_path}")

    enriched = enrich_prices(raw, ticker=args.symbol, benchmarks=None)
    bars_per_year = annualization_bars_per_year(enriched)
    enriched = rescale_annualized_columns(enriched, bars_per_year)
    enriched_path = outdir / f"{prefix}_available_enriched.csv"
    enriched.to_csv(enriched_path, index=False, encoding="utf-8-sig")
    print(f"Enriched intraday data saved: {enriched_path}")

    summary = compute_summary(enriched, result, bars_per_year)
    summary_path = outdir / f"{prefix}_available_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Intraday summary saved: {summary_path}")

    audit_path = outdir / f"{prefix}_10y_source_audit.csv"
    write_source_audit(audit_path, result, yahoo_evidence, stooq_evidence)
    print(f"Intraday source audit saved: {audit_path}")

    audit = pd.read_csv(audit_path)
    report_path = outdir / f"{prefix}_data_report.md"
    write_report(report_path, summary, audit, prefix)
    print(f"Intraday report saved: {report_path}")

    if not args.skip_charts:
        make_charts(enriched, dirs["charts"], prefix, args.interval)
        print(f"Intraday charts saved: {dirs['charts']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

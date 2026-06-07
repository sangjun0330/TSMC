#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-time quote engine for the investor dashboard.

Fetches the current price, previous close, intraday change, and market state
for the decision universe. US names use Finnhub (real-time-ish, free tier);
Korea (.KS/.KQ) uses Yahoo, which Finnhub free does not cover. Everything is
best-effort and per-symbol fault-tolerant: a failed symbol simply omits its
quote rather than breaking the whole payload. Results are cached briefly so
many polling clients share one upstream round-trip.

Note: this only provides LIVE PRICE-derived data. Model predictions
(up-probability, close forecast, etc.) are daily/multi-day horizon and are NOT
recomputed here — see the dashboard which keeps them labelled as as-of close.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

ROOT = Path(__file__).resolve().parent
DEFAULT_UNIVERSE_CONFIG = ROOT / "config" / "semiconductor_universe_top10.csv"
CACHE_TTL_SECONDS = 20.0

_CACHE: dict[str, object] = {"ts": 0.0, "data": None}
_LOCK = threading.Lock()


def _load_env_value(name: str) -> str:
    env_path = ROOT / ".env"
    try:
        for line in env_path.read_text().splitlines():
            s = line.strip()
            if s.startswith("#") or "=" not in s:
                continue
            key, value = s.split("=", 1)
            if key.strip() == name:
                return value.strip()
    except Exception:
        pass
    return ""


def _http_json(url: str, timeout: int = 8) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _finnhub_quote(symbol: str, key: str) -> dict | None:
    url = f"https://finnhub.io/api/v1/quote?symbol={urllib.parse.quote(symbol)}&token={key}"
    data = _http_json(url)
    current = data.get("c")
    if not current:
        return None
    prev = data.get("pc")
    return {
        "price": float(current),
        "prev_close": float(prev) if prev else None,
        "change": float(data.get("d") or 0.0),
        "change_pct": float(data.get("dp") or 0.0),
        "quote_time": int(data.get("t") or 0),
        "source": "finnhub",
        "market_state": None,
    }


def _yahoo_quote(yahoo_symbol: str) -> dict | None:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(yahoo_symbol)}?range=1d&interval=1m"
    )
    data = _http_json(url)
    result = (data.get("chart", {}) or {}).get("result") or []
    if not result:
        return None
    meta = result[0].get("meta", {}) or {}
    price = meta.get("regularMarketPrice")
    if price is None:
        return None
    prev = meta.get("previousClose") or meta.get("chartPreviousClose")
    change = (float(price) - float(prev)) if prev else 0.0
    change_pct = (change / float(prev) * 100.0) if prev else 0.0
    return {
        "price": float(price),
        "prev_close": float(prev) if prev else None,
        "change": float(change),
        "change_pct": float(change_pct),
        "quote_time": int(meta.get("regularMarketTime") or 0),
        "source": "yahoo",
        "market_state": meta.get("marketState"),
    }


def _market_state(zone: str, open_min: int, close_min: int) -> str:
    if ZoneInfo is None:
        return "UNKNOWN"
    now = datetime.now(ZoneInfo(zone))
    if now.weekday() >= 5:
        return "CLOSED"
    minutes = now.hour * 60 + now.minute
    if open_min <= minutes < close_min:
        return "REGULAR"
    if zone == "America/New_York":
        if 4 * 60 <= minutes < open_min:
            return "PRE"
        if close_min <= minutes < 20 * 60:
            return "POST"
    return "CLOSED"


def _us_market_state() -> str:
    # Pre 04:00, regular 09:30-16:00, after 16:00-20:00 (ET) — all treated as live.
    return _market_state("America/New_York", 9 * 60 + 30, 16 * 60)


def _kr_market_state() -> str:
    # Korea (KRX) treated as live from the opening auction (08:30) through the
    # after-hours single-price session (18:00):
    #   08:30-09:00 PRE (장전 동시호가) · 09:00-15:30 REGULAR (정규장)
    #   15:30-18:00 POST (시간외 종가/단일가) · else CLOSED
    if ZoneInfo is None:
        return "UNKNOWN"
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    if now.weekday() >= 5:
        return "CLOSED"
    minutes = now.hour * 60 + now.minute
    if 8 * 60 + 30 <= minutes < 9 * 60:
        return "PRE"
    if 9 * 60 <= minutes < 15 * 60 + 30:
        return "REGULAR"
    if 15 * 60 + 30 <= minutes < 18 * 60:
        return "POST"
    return "CLOSED"


def _yahoo_fx_usdkrw() -> dict | None:
    """Live USD/KRW from Yahoo (KRW=X trades ~24h on weekdays)."""
    q = _yahoo_quote("KRW=X")
    if not q:
        return None
    return {
        "usdkrw": q["price"],
        "prev_close": q.get("prev_close"),
        "change_pct": q.get("change_pct"),
        "source": "yahoo-live",
        "data_source": "실시간",
    }


def _is_korea(symbol: str, currency: str) -> bool:
    sym = symbol.upper()
    return sym.endswith(".KS") or sym.endswith(".KQ") or currency.upper() == "KRW"


def fetch_live_quotes(config_path: str | Path = DEFAULT_UNIVERSE_CONFIG, use_cache: bool = True) -> dict:
    """Return {generated_at, quotes:{symbol:{price, prev_close, change, change_pct,
    market_state, currency, source, quote_time}}} for the decision universe."""
    now = time.time()
    if use_cache:
        with _LOCK:
            cached = _CACHE.get("data")
            if cached is not None and now - float(_CACHE.get("ts", 0.0)) < CACHE_TTL_SECONDS:
                return cached  # type: ignore[return-value]

    key = _load_env_value("FINNHUB_API_KEY")
    try:
        config = pd.read_csv(config_path)
    except Exception:
        config = pd.DataFrame()

    us_state = _us_market_state()
    kr_state = _kr_market_state()

    def _one(row) -> tuple[str, dict] | None:
        symbol = str(row.get("symbol", "")).strip()
        if not symbol:
            return None
        yahoo_symbol = str(row.get("symbol_yahoo", symbol) or symbol).strip()
        currency = str(row.get("listing_currency", "USD") or "USD").strip()
        korea = _is_korea(symbol, currency)
        quote: dict | None = None
        try:
            if korea:
                quote = _yahoo_quote(yahoo_symbol)
            else:
                if key:
                    try:
                        quote = _finnhub_quote(symbol, key)
                    except Exception:
                        quote = None
                if quote is None:
                    quote = _yahoo_quote(yahoo_symbol)
        except Exception:
            quote = None
        if not quote:
            return None
        # Korea: force our extended-hours state (08:30-18:00). US: Finnhub gives
        # no state, so fall back to our pre/regular/after windows.
        quote["market_state"] = kr_state if korea else (quote.get("market_state") or us_state)
        quote["currency"] = currency
        quote["region"] = "KR" if korea else "US"
        return symbol, quote

    quotes: dict[str, dict] = {}
    fx: dict | None = None
    rows = [row for _, row in config.iterrows()]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(12, max(1, len(rows))) + 1) as pool:
        fx_future = pool.submit(_yahoo_fx_usdkrw)
        for res in pool.map(_one, rows):
            if res:
                quotes[res[0]] = res[1]
        try:
            fx = fx_future.result()
        except Exception:
            fx = None

    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "us_market_state": us_state,
        "kr_market_state": kr_state,
        "fx": fx,
        "quotes": quotes,
    }
    with _LOCK:
        _CACHE["data"] = payload
        _CACHE["ts"] = time.time()
    return payload


if __name__ == "__main__":
    import sys

    out = fetch_live_quotes(use_cache=False)
    print(f"US={out['us_market_state']} KR={out['kr_market_state']} quotes={len(out['quotes'])}")
    for sym, q in out["quotes"].items():
        print(f"  {sym:12s} {q['price']:>12.2f} {q['change_pct']:+6.2f}%  {q['market_state']:8s} {q['source']}")
    sys.exit(0)

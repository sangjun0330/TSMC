from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests


ENGINE_CURRENCY = "USD"
DEFAULT_FX_PAIR_BY_CURRENCY = {
    "KRW": "KRW=X",
}
PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")


@dataclass(frozen=True)
class CurrencyProfile:
    listing_currency: str
    display_currency: str
    engine_currency: str
    fx_pair: str


def normalize_currency(value: object, default: str = "USD") -> str:
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE", "NA", "N/A"}:
        return default.upper()
    return text


def clean_optional_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "n/a"}:
        return ""
    return text


def infer_listing_currency(symbol: object = "", symbol_yahoo: object = "") -> str:
    values = [str(symbol or "").strip().upper(), str(symbol_yahoo or "").strip().upper()]
    if any(value.endswith(".KS") or value.endswith(".KQ") for value in values):
        return "KRW"
    return "USD"


def currency_profile(
    symbol: object = "",
    symbol_yahoo: object = "",
    *,
    listing_currency: object = "",
    display_currency: object = "",
    engine_currency: object = ENGINE_CURRENCY,
    fx_pair: object = "",
) -> CurrencyProfile:
    listing = normalize_currency(listing_currency, infer_listing_currency(symbol, symbol_yahoo))
    engine = normalize_currency(engine_currency, ENGINE_CURRENCY)
    display = normalize_currency(display_currency, listing)
    pair = clean_optional_text(fx_pair)
    if listing != engine and not pair:
        pair = DEFAULT_FX_PAIR_BY_CURRENCY.get(listing, "")
    return CurrencyProfile(
        listing_currency=listing,
        display_currency=display,
        engine_currency=engine,
        fx_pair=pair,
    )


def requires_fx_conversion(listing_currency: object, engine_currency: object = ENGINE_CURRENCY) -> bool:
    return normalize_currency(listing_currency) != normalize_currency(engine_currency, ENGINE_CURRENCY)


def yahoo_chart_url(symbol: str, start: str, end: str, interval: str = "1d") -> str:
    start_ts = int(pd.to_datetime(start, utc=True).timestamp())
    end_ts = int((pd.to_datetime(end, utc=True) + pd.Timedelta(days=1)).timestamp())
    query = urlencode(
        {
            "period1": start_ts,
            "period2": end_ts,
            "interval": interval,
            "events": "history",
        }
    )
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}?{query}"


def fetch_yahoo_fx_rates(pair: str, start: str, end: str, timeout: int = 30) -> pd.DataFrame:
    pair = str(pair or "").strip()
    if not pair:
        return pd.DataFrame(columns=["date", "fx_pair", "usdkrw", "fx_rate_to_usd", "data_source", "source_url"])
    url = yahoo_chart_url(pair, start, end, "1d")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/121 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo FX download failed for {pair}: {chart['error']}")
    result = chart.get("result", [None])[0]
    if not result:
        raise RuntimeError(f"Yahoo FX download returned no result for {pair}")
    timestamps = result.get("timestamp", [])
    quote_block = result.get("indicators", {}).get("quote", [{}])[0]
    close = quote_block.get("close", [])
    if not timestamps or not close:
        raise RuntimeError(f"Yahoo FX download returned no daily close for {pair}")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).date.astype(str),
            "fx_close": close,
        }
    )
    frame["fx_close"] = pd.to_numeric(frame["fx_close"], errors="coerce")
    frame = frame.dropna(subset=["fx_close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"Yahoo FX download produced empty cleaned frame for {pair}")
    frame["fx_pair"] = pair
    if pair.upper() == "KRW=X":
        frame["base_currency"] = "USD"
        frame["quote_currency"] = "KRW"
        frame["usdkrw"] = frame["fx_close"]
        frame["fx_rate_to_usd"] = 1.0 / frame["usdkrw"]
    else:
        frame["base_currency"] = ""
        frame["quote_currency"] = ""
        frame["usdkrw"] = np.nan
        frame["fx_rate_to_usd"] = np.nan
    frame["data_source"] = "yahoo_chart_fx"
    frame["source_url"] = url
    frame["generated_at_utc"] = pd.Timestamp.utcnow().replace(microsecond=0).isoformat()
    return frame[
        [
            "date",
            "fx_pair",
            "base_currency",
            "quote_currency",
            "fx_close",
            "usdkrw",
            "fx_rate_to_usd",
            "data_source",
            "source_url",
            "generated_at_utc",
        ]
    ]


def fetch_frankfurter_fx_rates(pair: str, start: str, end: str, timeout: int = 30) -> pd.DataFrame:
    pair = str(pair or "").strip().upper()
    if pair != "KRW=X":
        raise ValueError(f"Frankfurter fallback only supports KRW=X in this pipeline, got {pair}.")
    url = f"https://api.frankfurter.app/{start}..{end}?from=USD&to=KRW"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    rates = payload.get("rates", {})
    rows = []
    for day, value in rates.items():
        krw = value.get("KRW") if isinstance(value, dict) else None
        if krw is None:
            continue
        rows.append({"date": day, "usdkrw": krw})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Frankfurter fallback returned no USD/KRW rows.")
    frame["usdkrw"] = pd.to_numeric(frame["usdkrw"], errors="coerce")
    frame = frame.dropna(subset=["usdkrw"]).sort_values("date").reset_index(drop=True)
    frame["fx_pair"] = "KRW=X"
    frame["base_currency"] = "USD"
    frame["quote_currency"] = "KRW"
    frame["fx_close"] = frame["usdkrw"]
    frame["fx_rate_to_usd"] = 1.0 / frame["usdkrw"]
    frame["data_source"] = "frankfurter_ecb_fx"
    frame["source_url"] = url
    frame["generated_at_utc"] = pd.Timestamp.utcnow().replace(microsecond=0).isoformat()
    return frame[
        [
            "date",
            "fx_pair",
            "base_currency",
            "quote_currency",
            "fx_close",
            "usdkrw",
            "fx_rate_to_usd",
            "data_source",
            "source_url",
            "generated_at_utc",
        ]
    ]


def fetch_fx_rates(pair: str, start: str, end: str, timeout: int = 30) -> pd.DataFrame:
    try:
        return fetch_yahoo_fx_rates(pair, start, end, timeout=timeout)
    except Exception:
        return fetch_frankfurter_fx_rates(pair, start, end, timeout=timeout)


def load_fx_rates(path: str | Path, fx_pair: str = "") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.columns = [str(col).lstrip("\ufeff") for col in frame.columns]
    if "date" not in frame.columns:
        return pd.DataFrame()
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    frame = frame.dropna(subset=["date"])
    if fx_pair and "fx_pair" in frame.columns:
        frame = frame[frame["fx_pair"].astype(str).str.upper().eq(str(fx_pair).upper())].copy()
    if "fx_rate_to_usd" in frame.columns:
        frame["fx_rate_to_usd"] = pd.to_numeric(frame["fx_rate_to_usd"], errors="coerce")
    if "usdkrw" in frame.columns:
        frame["usdkrw"] = pd.to_numeric(frame["usdkrw"], errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def ensure_fx_rate_file(
    path: str | Path,
    pairs: Iterable[str],
    *,
    start: str,
    end: str,
    timeout: int = 30,
) -> pd.DataFrame:
    path = Path(path)
    existing = load_fx_rates(path)
    requested = [clean_optional_text(pair) for pair in pairs if clean_optional_text(pair)]
    if not requested:
        return existing
    frames: list[pd.DataFrame] = []
    for pair in requested:
        frame = pd.DataFrame()
        if not existing.empty and "fx_pair" in existing.columns:
            cached = existing[existing["fx_pair"].astype(str).str.upper().eq(pair.upper())].copy()
            if not cached.empty:
                min_date = pd.to_datetime(cached["date"], errors="coerce").min()
                max_date = pd.to_datetime(cached["date"], errors="coerce").max()
                if pd.notna(min_date) and pd.notna(max_date) and min_date.date() <= pd.to_datetime(start).date() and max_date.date() >= pd.to_datetime(end).date():
                    frame = cached
        if frame.empty:
            frame = fetch_fx_rates(pair, start, end, timeout=timeout)
            time.sleep(0.15)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else existing
    if not combined.empty:
        combined = combined.drop_duplicates(["date", "fx_pair"], keep="last").sort_values(["fx_pair", "date"])
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path, index=False, encoding="utf-8-sig")
    return combined


def _date_key(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize().astype("datetime64[ns]")


def repair_ohlc_bounds(frame: pd.DataFrame, *, suffix: str = "") -> pd.DataFrame:
    names = {
        key: f"{key}_{suffix}" if suffix else key
        for key in ("open", "high", "low", "close")
    }
    if not all(col in frame.columns for col in names.values()):
        return frame
    values = frame[[names["open"], names["high"], names["low"], names["close"]]].apply(pd.to_numeric, errors="coerce")
    frame[names["high"]] = values.max(axis=1)
    frame[names["low"]] = values.min(axis=1)
    return frame


def normalize_ohlcv_to_engine_currency(
    frame: pd.DataFrame,
    *,
    symbol: object = "",
    symbol_yahoo: object = "",
    listing_currency: object = "",
    display_currency: object = "",
    engine_currency: object = ENGINE_CURRENCY,
    fx_pair: object = "",
    fx_rates: pd.DataFrame | None = None,
    date_column: str = "date",
) -> pd.DataFrame:
    profile = currency_profile(
        symbol,
        symbol_yahoo,
        listing_currency=listing_currency,
        display_currency=display_currency,
        engine_currency=engine_currency,
        fx_pair=fx_pair,
    )
    out = frame.copy()
    if out.empty:
        for col, value in {
            "listing_currency": profile.listing_currency,
            "display_currency": profile.display_currency,
            "engine_currency": profile.engine_currency,
            "fx_pair": profile.fx_pair,
        }.items():
            out[col] = value
        return out

    for col in PRICE_COLUMNS:
        if col in out.columns:
            native_col = f"{col}_native"
            out[col] = pd.to_numeric(out[col], errors="coerce")
            if native_col not in out.columns:
                out[native_col] = out[col]
            else:
                out[native_col] = pd.to_numeric(out[native_col], errors="coerce").fillna(out[col])
    repair_ohlc_bounds(out, suffix="native")

    out["listing_currency"] = profile.listing_currency
    out["display_currency"] = profile.display_currency
    out["engine_currency"] = profile.engine_currency
    out["fx_pair"] = profile.fx_pair

    if profile.listing_currency == profile.engine_currency:
        out["native_to_engine_fx_rate"] = 1.0
        out["fx_rate_to_usd"] = 1.0 if profile.engine_currency == "USD" else np.nan
        out["usdkrw"] = np.nan
        out["fx_date"] = pd.to_datetime(out[date_column], errors="coerce").dt.date.astype(str)
        suffix = profile.engine_currency.lower()
        for col in PRICE_COLUMNS:
            if col in out.columns:
                out[f"{col}_{suffix}"] = out[col]
        repair_ohlc_bounds(out)
        return out

    if profile.listing_currency == "KRW" and profile.engine_currency == "USD":
        fx = fx_rates.copy() if fx_rates is not None else pd.DataFrame()
        if fx.empty:
            raise ValueError("KRW to USD normalization requires USD/KRW FX rates.")
        fx = fx.copy()
        fx["date"] = pd.to_datetime(fx["date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
        if "fx_pair" in fx.columns and profile.fx_pair:
            fx = fx[fx["fx_pair"].astype(str).str.upper().eq(profile.fx_pair.upper())].copy()
        fx["fx_rate_to_usd"] = pd.to_numeric(fx["fx_rate_to_usd"], errors="coerce")
        if "usdkrw" in fx.columns:
            fx["usdkrw"] = pd.to_numeric(fx["usdkrw"], errors="coerce")
        fx = fx.dropna(subset=["date", "fx_rate_to_usd"]).sort_values("date")
        if fx.empty:
            raise ValueError(f"No usable FX rates found for {profile.fx_pair}.")

        key = pd.DataFrame(
            {
                "_row": np.arange(len(out)),
                "_fx_lookup_date": _date_key(out[date_column]),
            }
        ).sort_values("_fx_lookup_date")
        rate_cols = ["date", "fx_rate_to_usd"]
        if "usdkrw" in fx.columns:
            rate_cols.append("usdkrw")
        aligned = pd.merge_asof(
            key,
            fx[rate_cols].sort_values("date"),
            left_on="_fx_lookup_date",
            right_on="date",
            direction="backward",
        ).sort_values("_row")
        rates = aligned["fx_rate_to_usd"].to_numpy()
        if np.isnan(rates).any():
            missing = int(np.isnan(rates).sum())
            raise ValueError(f"Missing FX rates for {missing} rows while normalizing {symbol_yahoo or symbol}.")
        out["native_to_engine_fx_rate"] = rates
        out["fx_rate_to_usd"] = rates
        out["usdkrw"] = aligned["usdkrw"].to_numpy() if "usdkrw" in aligned.columns else 1.0 / rates
        out["fx_date"] = pd.to_datetime(aligned["date"], errors="coerce").dt.date.astype(str).to_numpy()
        for col in PRICE_COLUMNS:
            if col in out.columns:
                out[f"{col}_usd"] = out[f"{col}_native"] * out["native_to_engine_fx_rate"]
                out[col] = out[f"{col}_usd"]
        repair_ohlc_bounds(out, suffix="usd")
        repair_ohlc_bounds(out)
        return out

    raise ValueError(f"Unsupported currency normalization: {profile.listing_currency} -> {profile.engine_currency}")

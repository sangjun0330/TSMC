#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Top10 intraday coverage and daily-aligned intraday features."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from tsm_core.io import check_row, strip_bom_columns
from tsm_core.universe import (
    DECISION_SCOPE_TOP10,
    DEFAULT_DECISION_UNIVERSE_CONFIG,
    DEFAULT_RESEARCH_UNIVERSE_CONFIG,
    REQUIRED_DECISION_SYMBOL_COUNT,
    TRAINING_SCOPE_UNIVERSAL_RESEARCH_POOL,
    UniverseMember,
    add_scope_columns,
    build_universe_scope_audit,
    load_decision_universe_members,
    load_research_universe_members,
    members_to_frame,
    symbol_set,
)


INTERVAL_LABEL = {
    "1m": "minute",
    "2m": "2min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "hourly",
    "1h": "hourly",
}
TIMEFRAME_SPECS = [
    ("1d", "daily", "daily"),
    ("1h", "hourly", "hourly"),
    ("5m", "model_minute", "model_feature"),
    ("1m", "execution_minute", "execution_slippage"),
]
INTRADAY_BASE_FEATURES = [
    "close",
    "return_20bar",
    "vol_20bar_ann",
    "atr14_pct",
    "bars_available",
    "realized_vol_20bar_ann",
    "realized_vol_78bar_ann",
    "realized_vol_390bar_ann",
    "har_rv_daily_lag",
    "har_rv_weekly_lag",
    "har_rv_monthly_lag",
    "intraday_trend_20bar",
    "last_hour_return",
    "close_position_in_range",
    "realized_range_pct",
    "volume_ratio_20bar",
    "volume_z_20bar",
    "liquidity_dollar_volume_20bar",
    "minutes_since_bar",
    "last_bar_timestamp",
    "source_provider",
]
FEATURE_ALIASES = {
    "model_minute": "m5",
    "execution_minute": "m1",
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return strip_bom_columns(pd.read_csv(path, **kwargs))


def intraday_path(member: UniverseMember, interval: str) -> Path:
    label = INTERVAL_LABEL.get(interval, str(interval).replace("m", "min"))
    prefix = "tsm_minute" if interval == "1m" else f"tsm_{label}"
    return member.data_outdir / f"{prefix}_available_enriched.csv"


def timeframe_path(member: UniverseMember, interval: str) -> Path:
    if interval == "1d":
        return member.paths.enriched
    if interval == "1h":
        return member.data_outdir / "tsm_hourly_available_enriched.csv"
    return intraday_path(member, interval)


def latest_timestamp(path: Path) -> tuple[str, int, str]:
    if not path.exists():
        return "", 0, "MISSING"
    try:
        frame = read_csv(path, usecols=lambda col: col in {"date", "close"})
    except Exception as exc:
        return "", 0, f"INVALID:{type(exc).__name__}"
    if frame.empty or "date" not in frame.columns:
        return "", 0, "EMPTY"
    dates = pd.to_datetime(frame["date"], errors="coerce")
    valid = dates.dropna()
    if valid.empty:
        return "", len(frame), "INVALID_DATE"
    return valid.max().isoformat(), len(frame), "OK"


def build_coverage_rows(members: Iterable[UniverseMember], decision_symbols: set[str]) -> pd.DataFrame:
    rows = []
    for member in members:
        for interval, feature_prefix, usage in TIMEFRAME_SPECS:
            path = timeframe_path(member, interval)
            latest, rows_count, status = latest_timestamp(path)
            rows.append(
                {
                    "symbol": member.symbol,
                    "symbol_group": member.symbol_group,
                    "market_region": member.market_region,
                    "interval": interval,
                    "feature_prefix": feature_prefix,
                    "usage": usage,
                    "is_decision_universe": member.symbol.upper() in decision_symbols,
                    "decision_scope": DECISION_SCOPE_TOP10,
                    "training_scope": TRAINING_SCOPE_UNIVERSAL_RESEARCH_POOL,
                    "status": status,
                    "rows": rows_count,
                    "latest_timestamp": latest,
                    "path": str(path),
                    "generated_at_utc": now_utc_iso(),
                }
            )
    return pd.DataFrame(rows)


def daily_source(member: UniverseMember, root_symbol: str = "", root_signals: Path | None = None) -> pd.DataFrame:
    source = member.paths.signals if member.paths.signals.exists() else member.paths.enriched
    if root_symbol and member.symbol.upper() == root_symbol.upper() and root_signals and root_signals.exists():
        source = root_signals
    if not source.exists():
        return pd.DataFrame(columns=["date", "symbol", "symbol_group", "asof_ts"])
    frame = read_csv(source, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if "date" not in frame.columns:
        return pd.DataFrame(columns=["date", "symbol", "symbol_group", "asof_ts"])
    out = frame[["date"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).drop_duplicates(["date"], keep="last")
    out["symbol"] = member.symbol
    out["symbol_group"] = member.symbol_group
    out["market_region"] = member.market_region
    out["asof_ts"] = out["date"] + pd.Timedelta(hours=23, minutes=59, seconds=59)
    out["daily_signal_available"] = 1.0
    out["daily_source_path"] = str(source)
    return out


def realized_vol(log_returns: pd.Series, window: int, annualization: float) -> pd.Series:
    rv = log_returns.rolling(window, min_periods=max(2, min(10, window // 2))).apply(
        lambda values: float(np.sqrt(np.nansum(np.square(values)))),
        raw=True,
    )
    return rv * np.sqrt(float(annualization) / float(max(window, 1)))


def prepare_intraday_features(path: Path, prefix: str, annualization: float) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp"])
    try:
        frame = read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["timestamp"])
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame(columns=["timestamp"])
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    high = pd.to_numeric(frame.get("high"), errors="coerce") if "high" in frame.columns else close
    low = pd.to_numeric(frame.get("low"), errors="coerce") if "low" in frame.columns else close
    volume = pd.to_numeric(frame.get("volume"), errors="coerce") if "volume" in frame.columns else pd.Series(np.nan, index=frame.index)
    returns = close.pct_change()
    log_returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    bar_range = (high - low).abs()
    range_denom = (high - low).replace(0, np.nan)
    rolling_volume = volume.rolling(20, min_periods=5)
    rv20 = realized_vol(log_returns, 20, annualization)
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["date"], errors="coerce"),
            f"{prefix}_close": close,
            f"{prefix}_return_20bar": close / close.shift(20) - 1.0,
            f"{prefix}_vol_20bar_ann": returns.rolling(20, min_periods=10).std() * np.sqrt(annualization),
            f"{prefix}_bars_available": close.rolling(20, min_periods=1).count(),
            f"{prefix}_realized_vol_20bar_ann": rv20,
            f"{prefix}_realized_vol_78bar_ann": realized_vol(log_returns, 78, annualization),
            f"{prefix}_realized_vol_390bar_ann": realized_vol(log_returns, 390, annualization),
            f"{prefix}_har_rv_daily_lag": rv20.shift(1),
            f"{prefix}_har_rv_weekly_lag": rv20.shift(1).rolling(5, min_periods=2).mean(),
            f"{prefix}_har_rv_monthly_lag": rv20.shift(1).rolling(20, min_periods=5).mean(),
            f"{prefix}_intraday_trend_20bar": close / close.rolling(20, min_periods=5).mean() - 1.0,
            f"{prefix}_last_hour_return": close / close.shift(12) - 1.0,
            f"{prefix}_close_position_in_range": (close - low) / range_denom,
            f"{prefix}_realized_range_pct": bar_range / close.replace(0, np.nan),
            f"{prefix}_volume_ratio_20bar": volume / rolling_volume.mean().replace(0, np.nan),
            f"{prefix}_volume_z_20bar": (volume - rolling_volume.mean()) / rolling_volume.std(ddof=0).replace(0, np.nan),
            f"{prefix}_liquidity_dollar_volume_20bar": (close * volume).rolling(20, min_periods=5).mean(),
            f"{prefix}_source_provider": frame.get("data_source", pd.Series("", index=frame.index)).astype(str),
        }
    )
    if "atr_14_pct" in frame.columns:
        out[f"{prefix}_atr14_pct"] = pd.to_numeric(frame["atr_14_pct"], errors="coerce")
    elif {"high", "low", "close"}.issubset(frame.columns):
        prev_close = close.shift(1)
        true_range = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        out[f"{prefix}_atr14_pct"] = true_range.rolling(14, min_periods=5).mean() / close.replace(0, np.nan)
    else:
        out[f"{prefix}_atr14_pct"] = np.nan
    return out.dropna(subset=["timestamp"])


def expected_feature_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{name}" for name in INTRADAY_BASE_FEATURES]


def merge_asof_feature(daily: pd.DataFrame, features: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if daily.empty:
        return daily.copy()
    out = daily.sort_values("asof_ts").copy()
    if features.empty:
        for col in expected_feature_columns(prefix):
            out[col] = np.nan
        out[f"{prefix}_feature_status"] = "MISSING"
        return out
    merged = pd.merge_asof(out, features.sort_values("timestamp"), left_on="asof_ts", right_on="timestamp", direction="backward")
    merged[f"{prefix}_minutes_since_bar"] = (merged["asof_ts"] - merged["timestamp"]).dt.total_seconds() / 60.0
    merged[f"{prefix}_last_bar_timestamp"] = merged["timestamp"]
    merged[f"{prefix}_feature_status"] = np.where(merged["timestamp"].notna(), "OK", "MISSING")
    return merged.drop(columns=["timestamp"])


def add_alias_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for source_prefix, alias_prefix in FEATURE_ALIASES.items():
        for name in INTRADAY_BASE_FEATURES:
            src = f"{source_prefix}_{name}"
            dst = f"{alias_prefix}_{name}"
            if src in out.columns and dst not in out.columns:
                out[dst] = out[src]
        status_src = f"{source_prefix}_feature_status"
        status_dst = f"{alias_prefix}_feature_status"
        if status_src in out.columns and status_dst not in out.columns:
            out[status_dst] = out[status_src]
    return out


def build_member_features(
    member: UniverseMember,
    decision_symbols: set[str],
    root_symbol: str = "",
    root_signals: Path | None = None,
) -> pd.DataFrame:
    daily = daily_source(member, root_symbol=root_symbol, root_signals=root_signals)
    if daily.empty:
        return pd.DataFrame()
    hourly = prepare_intraday_features(timeframe_path(member, "1h"), "hourly", annualization=252 * 6.5)
    model_minute = prepare_intraday_features(timeframe_path(member, "5m"), "model_minute", annualization=252 * 78)
    execution_minute = prepare_intraday_features(timeframe_path(member, "1m"), "execution_minute", annualization=252 * 390)
    out = merge_asof_feature(daily, hourly, "hourly")
    out = merge_asof_feature(out, model_minute, "model_minute")
    out = merge_asof_feature(out, execution_minute, "execution_minute")
    out = add_alias_columns(out)
    status_cols = ["hourly_feature_status", "model_minute_feature_status", "execution_minute_feature_status"]
    out["timeframe_coverage_score"] = out[status_cols].eq("OK").sum(axis=1) / float(len(status_cols))
    out["intraday_any_coverage"] = out[status_cols].eq("OK").any(axis=1).astype(float)
    out["intraday_full_coverage"] = out[status_cols].eq("OK").all(axis=1).astype(float)
    out["intraday_model_ready"] = out[["hourly_feature_status", "model_minute_feature_status"]].eq("OK").all(axis=1).astype(float)
    out["intraday_execution_ready"] = out["execution_minute_feature_status"].eq("OK").astype(float)
    out["intraday_coverage_class"] = np.select(
        [
            out["intraday_full_coverage"].eq(1.0),
            out["intraday_model_ready"].eq(1.0),
            out["intraday_any_coverage"].eq(1.0),
        ],
        ["FULL", "MODEL_READY", "PARTIAL"],
        default="MISSING",
    )
    out["intraday_feature_freshness_minutes"] = out[
        ["hourly_minutes_since_bar", "model_minute_minutes_since_bar", "execution_minute_minutes_since_bar"]
    ].min(axis=1, skipna=True)
    out["intraday_feature_status"] = np.where(out["timeframe_coverage_score"].eq(1.0), "OK", "PARTIAL_OR_MISSING")
    out = out.drop(columns=["asof_ts"])
    return add_scope_columns(out, decision_symbols)


def build_quality(coverage: pd.DataFrame, features: pd.DataFrame, decision_symbols: set[str]) -> pd.DataFrame:
    top10_coverage = coverage[coverage["is_decision_universe"].astype(bool)] if not coverage.empty else pd.DataFrame()
    required_pairs = {(symbol, interval) for symbol in decision_symbols for interval, _, _ in TIMEFRAME_SPECS}
    observed_pairs = set(zip(top10_coverage.get("symbol", pd.Series(dtype=str)).astype(str), top10_coverage.get("interval", pd.Series(dtype=str)).astype(str)))
    ok_pairs = set(
        zip(
            top10_coverage[top10_coverage.get("status", pd.Series(dtype=str)).astype(str).eq("OK")].get("symbol", pd.Series(dtype=str)).astype(str),
            top10_coverage[top10_coverage.get("status", pd.Series(dtype=str)).astype(str).eq("OK")].get("interval", pd.Series(dtype=str)).astype(str),
        )
    )
    return pd.DataFrame(
        [
            check_row(
                "top10_decision_symbol_count_matches_required",
                len(decision_symbols) == REQUIRED_DECISION_SYMBOL_COUNT,
                "CRITICAL",
                len(decision_symbols),
                str(REQUIRED_DECISION_SYMBOL_COUNT),
            ),
            check_row("top10_timeframe_rows_complete", required_pairs.issubset(observed_pairs), "CRITICAL", len(observed_pairs), len(required_pairs)),
            check_row("top10_timeframe_coverage_ok", required_pairs.issubset(ok_pairs), "WARN", len(ok_pairs), len(required_pairs), "Public Yahoo limits can leave 1m/5m partial or missing; this gates freshness diagnostics."),
            check_row("intraday_daily_features_non_empty", not features.empty, "WARN", len(features)),
            check_row("intraday_features_tag_decision_scope", "decision_scope" in features.columns if not features.empty else False, "CRITICAL", "decision_scope"),
            check_row("intraday_features_tag_training_scope", "training_scope" in features.columns if not features.empty else False, "CRITICAL", "training_scope"),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Top10/research intraday feature coverage.")
    parser.add_argument("--decision-universe-config", default=DEFAULT_DECISION_UNIVERSE_CONFIG)
    parser.add_argument("--research-universe-config", default=DEFAULT_RESEARCH_UNIVERSE_CONFIG)
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--root-symbol", default="TSM", help="Symbol that can use the freshly generated root daily signal file.")
    parser.add_argument("--root-signals", default="", help="Optional root daily signal file used to avoid root/universe date mismatch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    decision_members = load_decision_universe_members(args.decision_universe_config, include_disabled=args.include_disabled)
    research_members = load_research_universe_members(
        args.decision_universe_config,
        args.research_universe_config,
        include_disabled=args.include_disabled,
    )
    decision_symbols = symbol_set(decision_members)
    top10_coverage = build_coverage_rows(decision_members, decision_symbols)
    research_coverage = build_coverage_rows(research_members, decision_symbols)
    research_audit = build_universe_scope_audit(decision_members, research_members)
    root_signals = Path(args.root_signals) if str(args.root_signals).strip() else None
    feature_parts = [
        build_member_features(member, decision_symbols, root_symbol=args.root_symbol, root_signals=root_signals)
        for member in research_members
    ]
    features = pd.concat([part for part in feature_parts if not part.empty], ignore_index=True) if feature_parts else pd.DataFrame()
    quality = build_quality(top10_coverage, features, decision_symbols)

    features.to_csv(outdir / "tsm_intraday_daily_features.csv", index=False)
    add_scope_columns(members_to_frame(decision_members), decision_symbols).to_csv(outdir / "tsm_decision_universe_config.csv", index=False)
    top10_coverage.to_csv(outdir / "tsm_top10_timeframe_coverage.csv", index=False)
    research_coverage.to_csv(outdir / "tsm_research_pool_timeframe_coverage.csv", index=False)
    research_audit.to_csv(outdir / "tsm_research_pool_audit.csv", index=False)
    quality.to_csv(outdir / "tsm_intraday_feature_quality_checks.csv", index=False)
    print("completed: intraday feature outputs =", outdir.resolve())
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()

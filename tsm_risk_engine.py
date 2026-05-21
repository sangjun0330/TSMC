#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSM risk engine.

This script separates position sizing and risk limits from entry signals.
It does not decide whether the signal is alpha-positive; it decides how much
risk the current market state can carry if a valid signal exists.

Outputs:
- tsm_risk_policy_daily.csv
- tsm_latest_risk_snapshot.csv
- tsm_risk_policy_report.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def pct(x: float) -> float:
    return float(x) * 100.0 if pd.notna(x) else np.nan


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def to_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def as_float(value, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def load_signals(path: Path) -> pd.DataFrame:
    df = strip_bom_columns(pd.read_csv(path, parse_dates=["date"]))
    df = df.sort_values("date").reset_index(drop=True)
    required = [
        "date",
        "close",
        "sma_50",
        "sma_200",
        "atr_14",
        "atr_14_pct",
        "vol_20d_ann",
        "drawdown_from_ath",
        "dist_close_sma_50_pct",
        "score_price_algo_total",
        "entry_trigger",
        "trade_action",
        "risk_pct_2atr",
        "algo_vol_high",
        "algo_vol_extreme",
        "algo_overextended_highvol",
        "algo_deep_downtrend_avoid",
        "algo_event_shock_day",
    ]
    require_columns(df, required, str(path))
    bool_cols = [
        "algo_vol_high",
        "algo_vol_extreme",
        "algo_overextended_highvol",
        "algo_deep_downtrend_avoid",
        "algo_event_shock_day",
    ]
    for col in bool_cols:
        df[col] = df[col].map(to_bool)
    return df


def vol_limit(row: pd.Series) -> tuple[float, str]:
    if to_bool(row["algo_vol_extreme"]):
        return 0.04, "EXTREME_VOL_LIMIT"
    if to_bool(row["algo_vol_high"]):
        return 0.07, "HIGH_VOL_LIMIT"
    return 0.12, "NORMAL_VOL_LIMIT"


def trend_limit(row: pd.Series) -> tuple[float, str]:
    close = as_float(row["close"])
    sma200 = as_float(row["sma_200"])
    if to_bool(row["algo_deep_downtrend_avoid"]):
        return 0.00, "DEEP_DOWNTREND_ZERO"
    if close < sma200:
        return 0.03, "BELOW_200D_LIMIT"
    return 0.12, "ABOVE_200D_LIMIT"


def drawdown_limit(row: pd.Series) -> tuple[float, str]:
    dd = as_float(row["drawdown_from_ath"])
    if dd <= -0.40:
        return 0.00, "DD_OVER_40_WAIT_FOR_FUNDAMENTAL_RECHECK"
    if dd <= -0.30:
        return 0.04, "DD_30_TO_40_CRISIS_LIMIT"
    if dd <= -0.20:
        return 0.06, "DD_20_TO_30_RECOVERY_LIMIT"
    if dd <= -0.15:
        return 0.08, "DD_15_TO_20_LIMIT"
    if dd >= -0.05:
        return 0.08, "NEAR_HIGH_NO_CHASE_LIMIT"
    return 0.12, "NORMAL_DD_LIMIT"


def score_limit(row: pd.Series) -> tuple[float, str]:
    score = as_float(row["score_price_algo_total"])
    trigger = str(row["entry_trigger"])
    if trigger == "DEEP_DD_RECOVERY" or str(row.get("trade_action")) == "RESEARCH_ONLY_DEEP_DD":
        return 0.00, "DEEP_DD_RESEARCH_ONLY"
    if score >= 75 and trigger != "NONE":
        return 0.12, "SCORE75_WITH_TRIGGER"
    if score >= 75:
        return 0.08, "SCORE75_WAIT_TRIGGER"
    if score >= 65:
        return 0.04, "SCORE65_WATCHLIST_ONLY"
    if score >= 60:
        return 0.02, "SCORE60_SMALL_OBSERVATION_ONLY"
    return 0.00, "SCORE_BELOW_60_NO_NEW_RISK"


def account_risk_limit(row: pd.Series, account_risk_pct: float) -> tuple[float, str]:
    risk_pct_2atr = as_float(row["risk_pct_2atr"])
    if pd.isna(risk_pct_2atr) or risk_pct_2atr <= 0:
        return 0.00, "INVALID_2ATR_RISK"
    return min(account_risk_pct / risk_pct_2atr, 1.0), f"ACCOUNT_RISK_{account_risk_pct * 100:.2f}PCT"


def risk_state(row: pd.Series, final_weight: float) -> str:
    if final_weight <= 0:
        return "NO_NEW_RISK"
    if str(row.get("trade_action")) == "RESEARCH_ONLY_DEEP_DD":
        return "RESEARCH_ONLY_NO_NEW_RISK"
    if to_bool(row["algo_vol_extreme"]):
        return "OBSERVATION_OR_TINY_SIZE_ONLY"
    if str(row["trade_action"]) == "ENTRY_ALLOWED":
        return "ENTRY_RISK_ALLOWED"
    if str(row["trade_action"]) == "WATCHLIST_PULLBACK_ONLY":
        return "WAIT_FOR_PULLBACK"
    if str(row["entry_trigger"]) == "NONE":
        return "WAIT_FOR_TRIGGER"
    return "MANAGE_EXISTING_OR_SMALL_SIZE"


def limiting_reason(limits: Dict[str, tuple[float, str]]) -> str:
    min_value = min(v[0] for v in limits.values())
    reasons = [reason for value, reason in limits.values() if abs(value - min_value) < 1e-12]
    return "|".join(reasons)


def build_risk_policy(
    signals: pd.DataFrame,
    account_risk_pct: float,
    secondary_account_risk_pct: float,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, row in signals.iterrows():
        limits = {
            "vol": vol_limit(row),
            "trend": trend_limit(row),
            "drawdown": drawdown_limit(row),
            "score": score_limit(row),
            "account": account_risk_limit(row, account_risk_pct),
        }
        secondary_account = account_risk_limit(row, secondary_account_risk_pct)
        final_weight = max(0.0, min(value for value, _ in limits.values()))
        risk_per_share = 2.0 * as_float(row["atr_14"])
        stop_price = as_float(row["close"]) - risk_per_share
        rows.append(
            {
                "date": row["date"],
                "close": row["close"],
                "trade_action": row["trade_action"],
                "entry_trigger": row["entry_trigger"],
                "score_price_algo_total": row["score_price_algo_total"],
                "atr_14": row["atr_14"],
                "atr_14_pct": row["atr_14_pct"],
                "vol_20d_ann": row["vol_20d_ann"],
                "drawdown_from_ath": row["drawdown_from_ath"],
                "dist_close_sma_50_pct": row["dist_close_sma_50_pct"],
                "risk_pct_2atr": row["risk_pct_2atr"],
                "risk_per_share_2atr": risk_per_share,
                "stop_price_2atr": stop_price,
                "vol_limit_weight": limits["vol"][0],
                "trend_limit_weight": limits["trend"][0],
                "drawdown_limit_weight": limits["drawdown"][0],
                "score_limit_weight": limits["score"][0],
                "account_risk_limit_weight": limits["account"][0],
                "secondary_account_risk_limit_weight": secondary_account[0],
                "final_recommended_max_weight": final_weight,
                "limiting_reason": limiting_reason(limits),
                "risk_state": risk_state(row, final_weight),
                "algo_vol_high": row["algo_vol_high"],
                "algo_vol_extreme": row["algo_vol_extreme"],
                "algo_overextended_highvol": row["algo_overextended_highvol"],
                "algo_deep_downtrend_avoid": row["algo_deep_downtrend_avoid"],
                "algo_event_shock_day": row["algo_event_shock_day"],
            }
        )
    return pd.DataFrame(rows)


def write_latest_snapshot(outdir: Path, policy: pd.DataFrame) -> None:
    last = policy.iloc[-1]
    ratio_cols = {
        "atr_14_pct",
        "vol_20d_ann",
        "drawdown_from_ath",
        "dist_close_sma_50_pct",
        "risk_pct_2atr",
        "vol_limit_weight",
        "trend_limit_weight",
        "drawdown_limit_weight",
        "score_limit_weight",
        "account_risk_limit_weight",
        "secondary_account_risk_limit_weight",
        "final_recommended_max_weight",
    }
    rows = []
    for col in policy.columns:
        value = last[col]
        if col in ratio_cols and pd.notna(value):
            value = pct(value)
        rows.append({"field": col, "value": value})
    pd.DataFrame(rows).to_csv(outdir / "tsm_latest_risk_snapshot.csv", index=False)


def fmt_pct(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.{digits}f}%"


def write_report(outdir: Path, policy: pd.DataFrame, account_risk_pct: float, secondary_account_risk_pct: float) -> None:
    last = policy.iloc[-1]
    counts = policy["risk_state"].value_counts().reset_index()
    counts.columns = ["risk_state", "days"]

    lines = [
        "# TSM Risk Policy Report",
        "",
        "## Policy",
        f"- Primary account risk budget: {account_risk_pct * 100:.2f}% per trade at a 2ATR stop.",
        f"- Secondary reference risk budget: {secondary_account_risk_pct * 100:.2f}% per trade at a 2ATR stop.",
        "- Final recommended max weight is the minimum of volatility, trend, drawdown, score, and account-risk limits.",
        "- This engine sizes risk only. It does not override signal validity.",
        "",
        "## Latest Risk Snapshot",
        f"- Date: {last['date'].date().isoformat()}",
        f"- Close: ${last['close']:.2f}",
        f"- Trade action: {last['trade_action']}",
        f"- Risk state: {last['risk_state']}",
        f"- Final recommended max weight: {fmt_pct(last['final_recommended_max_weight'])}",
        f"- Limiting reason: {last['limiting_reason']}",
        f"- 2ATR stop price: ${last['stop_price_2atr']:.2f}",
        f"- 2ATR risk per share: ${last['risk_per_share_2atr']:.2f}",
        "",
        "## Risk State Distribution",
        "",
        "| Risk state | Days |",
        "|---|---:|",
    ]
    for _, row in counts.iterrows():
        lines.append(f"| {row['risk_state']} | {int(row['days'])} |")

    lines.extend(
        [
            "",
            "## Output files",
            "- tsm_risk_policy_daily.csv",
            "- tsm_latest_risk_snapshot.csv",
            "",
            "This report is research tooling, not investment advice.",
        ]
    )
    (outdir / "tsm_risk_policy_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TSM risk policy outputs.")
    parser.add_argument("--signals", default="tsm_price_rule_output/tsm_daily_algorithmic_signals.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--account-risk-pct", type=float, default=0.005)
    parser.add_argument("--secondary-account-risk-pct", type=float, default=0.010)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    signals = load_signals(Path(args.signals))
    policy = build_risk_policy(
        signals=signals,
        account_risk_pct=args.account_risk_pct,
        secondary_account_risk_pct=args.secondary_account_risk_pct,
    )
    policy.to_csv(outdir / "tsm_risk_policy_daily.csv", index=False)
    write_latest_snapshot(outdir, policy)
    write_report(outdir, policy, args.account_risk_pct, args.secondary_account_risk_pct)

    last = policy.iloc[-1]
    print("완료: risk policy outputs =", outdir.resolve())
    print(
        {
            "date": last["date"].date().isoformat(),
            "risk_state": last["risk_state"],
            "final_recommended_max_weight_pct": pct(last["final_recommended_max_weight"]),
            "limiting_reason": last["limiting_reason"],
        }
    )


if __name__ == "__main__":
    main()

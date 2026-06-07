#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily and intraday-aware stress and scenario engine for TSM.

It estimates how current risk sizing would behave under historical drawdowns,
daily return shocks, realized intraday volatility shocks, and strategy equity
drawdowns.

Outputs:
- tsm_daily_stress_scenarios.csv
- tsm_strategy_stress_summary.csv
- tsm_latest_stress_snapshot.csv
- tsm_daily_stress_report.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def pct(x: float) -> float:
    return float(x) * 100.0 if pd.notna(x) else np.nan


def as_float(value, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt_pct(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}%"


def load_csv(path: Path, required_cols: Iterable[str] | None = None, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = strip_bom_columns(pd.read_csv(path, **kwargs))
    if required_cols:
        require_columns(df, required_cols, str(path))
    return df


def latest_risk_weight(risk_policy: pd.DataFrame) -> float:
    if risk_policy.empty:
        return 0.0
    return float(np.clip(as_float(risk_policy.iloc[-1]["final_recommended_max_weight"], 0.0), 0.0, 1.0))


def build_daily_shock_scenarios(signals: pd.DataFrame, latest_close: float, final_weight: float) -> List[Dict]:
    ret = pd.to_numeric(signals["close_change_pct"], errors="coerce").dropna()
    specs = [
        ("worst_1d_close_to_close", ret.min(), "Historical worst daily close-to-close return."),
        ("p01_1d_close_to_close", ret.quantile(0.01), "1st percentile daily close-to-close return."),
        ("p05_1d_close_to_close", ret.quantile(0.05), "5th percentile daily close-to-close return."),
        ("minus_10pct_manual_shock", -0.10, "Manual single-day shock."),
        ("minus_20pct_manual_shock", -0.20, "Manual gap/crash shock."),
    ]
    rows = []
    for name, shock, notes in specs:
        rows.append(
            {
                "scenario_type": "daily_return_shock",
                "scenario_name": name,
                "source": "tsm_daily_algorithmic_signals",
                "shock_return_pct": pct(shock),
                "latest_close": latest_close,
                "implied_price": latest_close * (1.0 + shock),
                "final_recommended_max_weight_pct": pct(final_weight),
                "estimated_portfolio_impact_pct": pct(final_weight * shock),
                "notes": notes,
            }
        )
    return rows


def build_drawdown_scenarios(drawdowns: pd.DataFrame, latest_close: float, final_weight: float) -> List[Dict]:
    if drawdowns.empty:
        return []
    d = drawdowns.copy()
    d["shock"] = pd.to_numeric(d["trough_drawdown_pct"], errors="coerce") / 100.0
    d = d.sort_values("shock").head(8)
    rows = []
    for _, row in d.iterrows():
        shock = as_float(row["shock"])
        rows.append(
            {
                "scenario_type": "historical_drawdown_replay",
                "scenario_name": f"peak_{row.get('peak_date')}_to_trough_{row.get('trough_date')}",
                "source": "tsm_drawdown_episodes",
                "shock_return_pct": pct(shock),
                "latest_close": latest_close,
                "implied_price": latest_close * (1.0 + shock),
                "final_recommended_max_weight_pct": pct(final_weight),
                "estimated_portfolio_impact_pct": pct(final_weight * shock),
                "notes": f"Historical peak-to-trough drawdown; recovery_date={row.get('recovery_date', '')}",
            }
        )
    return rows


def build_atr_scenarios(signals: pd.DataFrame, latest_close: float, final_weight: float) -> List[Dict]:
    last = signals.iloc[-1]
    atr = as_float(last["atr_14"])
    specs = [
        ("minus_1atr", -atr / latest_close, "One ATR adverse move."),
        ("minus_2atr_stop", -2 * atr / latest_close, "Current 2ATR stop distance."),
        ("minus_3atr_tail", -3 * atr / latest_close, "Three ATR adverse move."),
    ]
    rows = []
    for name, shock, notes in specs:
        rows.append(
            {
                "scenario_type": "atr_shock",
                "scenario_name": name,
                "source": "latest_atr_14",
                "shock_return_pct": pct(shock),
                "latest_close": latest_close,
                "implied_price": latest_close * (1.0 + shock),
                "final_recommended_max_weight_pct": pct(final_weight),
                "estimated_portfolio_impact_pct": pct(final_weight * shock),
                "notes": notes,
            }
        )
    return rows


def build_intraday_scenarios(intraday_features: pd.DataFrame, latest_close: float, final_weight: float) -> List[Dict]:
    if intraday_features.empty:
        return []
    frame = intraday_features.copy()
    if "symbol" in frame.columns:
        frame = frame[frame["symbol"].astype(str).str.upper().eq("TSM")]
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values("date")
    if frame.empty:
        return []
    latest = frame.iloc[-1]
    rows: list[dict] = []
    vol = as_float(latest.get("m5_realized_vol_20bar_ann"), as_float(latest.get("model_minute_realized_vol_20bar_ann"), np.nan))
    realized_range = as_float(latest.get("m1_realized_range_pct"), as_float(latest.get("execution_minute_realized_range_pct"), np.nan))
    coverage = as_float(latest.get("timeframe_coverage_score"), np.nan)
    if pd.notna(vol):
        for multiple in [1.0, 2.0]:
            shock = -abs(vol) / math.sqrt(252.0) * multiple
            rows.append(
                {
                    "scenario_type": "intraday_realized_vol_shock",
                    "scenario_name": f"minus_{multiple:g}x_intraday_realized_daily_vol",
                    "source": "tsm_intraday_daily_features",
                    "shock_return_pct": pct(shock),
                    "latest_close": latest_close,
                    "implied_price": latest_close * (1.0 + shock),
                    "final_recommended_max_weight_pct": pct(final_weight),
                    "estimated_portfolio_impact_pct": pct(final_weight * shock),
                    "notes": f"coverage_score={coverage}",
                }
            )
    if pd.notna(realized_range):
        shock = -abs(realized_range)
        rows.append(
            {
                "scenario_type": "gap_plus_intraday_range",
                "scenario_name": "gap_down_plus_latest_1m_range",
                "source": "tsm_intraday_daily_features",
                "shock_return_pct": pct(shock),
                "latest_close": latest_close,
                "implied_price": latest_close * (1.0 + shock),
                "final_recommended_max_weight_pct": pct(final_weight),
                "estimated_portfolio_impact_pct": pct(final_weight * shock),
                "notes": "Execution path stress uses 1m realized range when available.",
            }
        )
    rows.append(
        {
            "scenario_type": "first_touch_ambiguity_stress",
            "scenario_name": "same_day_stop_target_order_uncertain",
            "source": "paper_oms_path_policy",
            "shock_return_pct": np.nan,
            "latest_close": latest_close,
            "implied_price": np.nan,
            "final_recommended_max_weight_pct": pct(final_weight),
            "estimated_portfolio_impact_pct": np.nan,
            "notes": "When 1m replay is missing, paper OMS records AMBIGUOUS_DAILY_PATH and applies conservative daily fallback.",
        }
    )
    return rows


def build_strategy_stress_summary(curves: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        curves,
        ["date", "strategy_id", "strategy_name", "equity", "daily_return_pct", "drawdown_pct", "position_weight_pct"],
        "equity curves",
    )
    rows = []
    curves = curves.copy()
    curves["daily_return"] = pd.to_numeric(curves["daily_return_pct"], errors="coerce") / 100.0
    curves["drawdown"] = pd.to_numeric(curves["drawdown_pct"], errors="coerce") / 100.0
    curves["position_weight"] = pd.to_numeric(curves["position_weight_pct"], errors="coerce") / 100.0
    for (strategy_id, strategy_name), group in curves.groupby(["strategy_id", "strategy_name"]):
        g = group.sort_values("date").copy()
        rolling_20 = (1.0 + g["daily_return"].fillna(0)).rolling(20, min_periods=5).apply(np.prod, raw=True) - 1.0
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "worst_daily_return_pct": pct(g["daily_return"].min()),
                "worst_20d_return_pct": pct(rolling_20.min()),
                "max_drawdown_pct": pct(g["drawdown"].min()),
                "latest_equity": g["equity"].iloc[-1],
                "latest_drawdown_pct": pct(g["drawdown"].iloc[-1]),
                "latest_position_weight_pct": pct(g["position_weight"].iloc[-1]),
                "exposure_days_pct": pct((g["position_weight"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_latest_stress_snapshot(scenarios: pd.DataFrame, strategy_stress: pd.DataFrame) -> pd.DataFrame:
    if scenarios.empty:
        worst_scenario = pd.Series(dtype=object)
    else:
        worst_scenario = scenarios.sort_values("estimated_portfolio_impact_pct").iloc[0]
    if strategy_stress.empty:
        worst_strategy = pd.Series(dtype=object)
    else:
        worst_strategy = strategy_stress.sort_values("max_drawdown_pct").iloc[0]

    rows = [
        {"field": "worst_current_weight_scenario", "value": worst_scenario.get("scenario_name", "NA")},
        {"field": "worst_current_weight_portfolio_impact_pct", "value": worst_scenario.get("estimated_portfolio_impact_pct", np.nan)},
        {"field": "worst_current_weight_implied_price", "value": worst_scenario.get("implied_price", np.nan)},
        {"field": "worst_strategy_by_mdd", "value": worst_strategy.get("strategy_id", "NA")},
        {"field": "worst_strategy_mdd_pct", "value": worst_strategy.get("max_drawdown_pct", np.nan)},
        {"field": "stress_status", "value": classify_stress_status(worst_scenario.get("estimated_portfolio_impact_pct", np.nan))},
    ]
    return pd.DataFrame(rows)


def classify_stress_status(worst_portfolio_impact_pct: float) -> str:
    impact = as_float(worst_portfolio_impact_pct)
    if pd.isna(impact):
        return "UNKNOWN"
    if impact <= -5:
        return "HIGH_STRESS_FOR_CURRENT_SIZE"
    if impact <= -2:
        return "MODERATE_STRESS_FOR_CURRENT_SIZE"
    return "LOW_STRESS_FOR_CURRENT_SIZE"


def write_report(outdir: Path, scenarios: pd.DataFrame, strategy_stress: pd.DataFrame, snapshot: pd.DataFrame) -> None:
    snapshot_map = dict(zip(snapshot["field"], snapshot["value"]))
    worst = scenarios.sort_values("estimated_portfolio_impact_pct").head(5)
    strategy = strategy_stress.sort_values("max_drawdown_pct")

    lines = [
        "# Top10 Daily Stress Report",
        "",
        "## Latest Stress Snapshot",
        f"- Stress status: {snapshot_map.get('stress_status', 'NA')}",
        f"- Worst current-weight scenario: {snapshot_map.get('worst_current_weight_scenario', 'NA')}",
        f"- Estimated portfolio impact: {fmt_pct(as_float(snapshot_map.get('worst_current_weight_portfolio_impact_pct')))}",
        f"- Implied reference price: ${as_float(snapshot_map.get('worst_current_weight_implied_price')):,.2f}",
        "",
        "## Worst Current-Weight Scenarios",
        "",
        "| Scenario | Shock | Implied Price | Portfolio Impact |",
        "|---|---:|---:|---:|",
    ]
    for _, row in worst.iterrows():
        lines.append(
            f"| {row['scenario_name']} | {fmt_pct(row['shock_return_pct'])} | "
            f"${row['implied_price']:,.2f} | {fmt_pct(row['estimated_portfolio_impact_pct'])} |"
        )

    lines.extend(["", "## Strategy Stress Summary", "", "| Strategy | MDD | Worst 20D | Worst 1D |", "|---|---:|---:|---:|"])
    for _, row in strategy.iterrows():
        lines.append(
            f"| {row['strategy_id']} | {fmt_pct(row['max_drawdown_pct'])} | "
            f"{fmt_pct(row['worst_20d_return_pct'])} | {fmt_pct(row['worst_daily_return_pct'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- Daily data cannot tell the intraday sequence of stop and target touches.",
            "- This stress engine is for portfolio-level daily risk framing, not execution replay.",
            "- Position size should be checked against both signal validity and stress tolerance.",
            "",
            "This report is research tooling, not investment advice.",
        ]
    )
    (outdir / "tsm_daily_stress_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily-data stress scenarios for TSM.")
    parser.add_argument("--signals", default="tsm_price_rule_output/tsm_daily_algorithmic_signals.csv")
    parser.add_argument("--risk-policy", default="tsm_price_rule_output/tsm_risk_policy_daily.csv")
    parser.add_argument("--drawdowns", default="tsm_price_rule_output/tsm_drawdown_episodes.csv")
    parser.add_argument("--equity-curves", default="tsm_price_rule_output/tsm_backtest_equity_curves.csv")
    parser.add_argument("--intraday-features", default="tsm_price_rule_output/tsm_intraday_daily_features.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    signals = load_csv(
        Path(args.signals),
        required_cols=["date", "close", "close_change_pct", "atr_14"],
        parse_dates=["date"],
    )
    risk_policy = load_csv(Path(args.risk_policy), required_cols=["date", "final_recommended_max_weight"], parse_dates=["date"])
    drawdowns = load_csv(Path(args.drawdowns)) if Path(args.drawdowns).exists() else pd.DataFrame()
    curves = load_csv(Path(args.equity_curves), parse_dates=["date"])
    intraday_features = load_csv(Path(args.intraday_features), parse_dates=["date"]) if Path(args.intraday_features).exists() else pd.DataFrame()

    latest = signals.iloc[-1]
    latest_close = as_float(latest["close"])
    final_weight = latest_risk_weight(risk_policy)

    rows = []
    rows.extend(build_daily_shock_scenarios(signals, latest_close, final_weight))
    rows.extend(build_drawdown_scenarios(drawdowns, latest_close, final_weight))
    rows.extend(build_atr_scenarios(signals, latest_close, final_weight))
    rows.extend(build_intraday_scenarios(intraday_features, latest_close, final_weight))
    scenarios = pd.DataFrame(rows)
    strategy_stress = build_strategy_stress_summary(curves)
    snapshot = build_latest_stress_snapshot(scenarios, strategy_stress)

    scenarios.to_csv(outdir / "tsm_daily_stress_scenarios.csv", index=False)
    strategy_stress.to_csv(outdir / "tsm_strategy_stress_summary.csv", index=False)
    snapshot.to_csv(outdir / "tsm_latest_stress_snapshot.csv", index=False)
    write_report(outdir, scenarios, strategy_stress, snapshot)

    print("완료: daily stress outputs =", outdir.resolve())
    print(snapshot.to_string(index=False))


if __name__ == "__main__":
    main()

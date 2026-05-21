#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML overlay backtest for OOS meta-label predictions.

This does not place trades. It evaluates whether the prediction layer improves
the event stream selected by the rule engine in walk-forward OOS predictions.

Outputs:
- tsm_ml_overlay_summary.csv
- tsm_ml_overlay_equity_curves.csv
- tsm_ml_overlay_quality_checks.csv
- tsm_ml_overlay_report.md
"""

from __future__ import annotations

import argparse
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


def read_oos_predictions(path: Path) -> pd.DataFrame:
    df = strip_bom_columns(pd.read_csv(path, parse_dates=["date", "train_end_date", "test_start_date", "test_end_date"]))
    required = [
        "horizon_days",
        "candidate_scope",
        "fold_id",
        "model_name",
        "signal_idx",
        "date",
        "p_success",
        "threshold",
        "selected_by_threshold",
        "label_success",
        "label_net_return_pct",
        "label_expected_r",
        "label_exit_reason",
        "train_end_date",
    ]
    require_columns(df, required, str(path))
    for col in ["label_entry_date", "label_exit_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "symbol" not in df.columns:
        df["symbol"] = "TSM"
    df["selected_by_threshold"] = df["selected_by_threshold"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df.sort_values(["candidate_scope", "horizon_days", "model_name", "date", "signal_idx"]).reset_index(drop=True)


def profit_factor(returns_pct: pd.Series) -> float:
    returns = pd.to_numeric(returns_pct, errors="coerce").dropna()
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if losses.empty:
        return np.inf if not wins.empty else np.nan
    return float(wins.sum() / abs(losses.sum()))


def event_curve(part: pd.DataFrame, event_weight: float) -> pd.DataFrame:
    rows: List[Dict] = []
    equity = 1.0
    peak = 1.0
    for _, row in part.sort_values(["date", "signal_idx"]).iterrows():
        event_return = pd.to_numeric(pd.Series([row["label_net_return_pct"]]), errors="coerce").iloc[0]
        if pd.isna(event_return):
            continue
        weighted_return = float(event_return) / 100.0 * event_weight
        equity *= 1.0 + weighted_return
        peak = max(peak, equity)
        rows.append(
            {
                "date": row["date"],
                "candidate_scope": row["candidate_scope"],
                "horizon_days": int(row["horizon_days"]),
                "model_name": row["model_name"],
                "policy": row["policy"],
                "signal_idx": int(row["signal_idx"]),
                "event_return_pct": event_return,
                "event_weight": event_weight,
                "weighted_event_return_pct": pct(weighted_return),
                "equity": equity,
                "drawdown_pct": pct(equity / peak - 1.0),
            }
        )
    return pd.DataFrame(rows)


def summarize_policy(part: pd.DataFrame, event_weight: float) -> Dict:
    returns = pd.to_numeric(part["label_net_return_pct"], errors="coerce").dropna()
    expected_r = pd.to_numeric(part.get("label_expected_r"), errors="coerce")
    curve = event_curve(part, event_weight)
    if returns.empty:
        return {
            "event_count": 0,
            "success_rate_pct": np.nan,
            "mean_net_return_pct": np.nan,
            "median_net_return_pct": np.nan,
            "mean_expected_r": np.nan,
            "profit_factor": np.nan,
            "cumulative_weighted_return_pct": np.nan,
            "max_event_curve_drawdown_pct": np.nan,
        }
    return {
        "event_count": int(len(returns)),
        "success_rate_pct": pct(pd.to_numeric(part.loc[returns.index, "label_success"], errors="coerce").mean()),
        "mean_net_return_pct": float(returns.mean()),
        "median_net_return_pct": float(returns.median()),
        "mean_expected_r": float(expected_r.loc[returns.index].mean()) if not expected_r.empty else np.nan,
        "profit_factor": profit_factor(returns),
        "stop_rate_pct": pct(part.loc[returns.index, "label_exit_reason"].astype(str).str.startswith("ATR_STOP").mean()),
        "hit_1r_rate_pct": pct(pd.to_numeric(part.loc[returns.index, "label_hit_1r_before_stop"], errors="coerce").mean()) if "label_hit_1r_before_stop" in part.columns else np.nan,
        "hit_2r_rate_pct": pct(pd.to_numeric(part.loc[returns.index, "label_hit_2r_before_stop"], errors="coerce").mean()) if "label_hit_2r_before_stop" in part.columns else np.nan,
        "cumulative_weighted_return_pct": pct(curve["equity"].iloc[-1] - 1.0) if not curve.empty else np.nan,
        "max_event_curve_drawdown_pct": float(curve["drawdown_pct"].min()) if not curve.empty else np.nan,
    }


def build_overlay_outputs(oos: pd.DataFrame, event_weight: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict] = []
    curve_parts: List[pd.DataFrame] = []
    group_cols = ["candidate_scope", "horizon_days", "model_name"]
    for keys, group in oos.groupby(group_cols, dropna=False):
        candidate_scope, horizon, model_name = keys
        policies = [
            ("RULE_ALL_OOS_EVENTS", group),
            ("ML_THRESHOLD_SELECTED", group[group["selected_by_threshold"]].copy()),
        ]
        policy_stats: Dict[str, Dict] = {}
        for policy, part in policies:
            part = part.copy()
            part["policy"] = policy
            stats = summarize_policy(part, event_weight)
            policy_stats[policy] = stats
            summary_rows.append(
                {
                    "candidate_scope": candidate_scope,
                    "horizon_days": int(horizon),
                    "model_name": model_name,
                    "policy": policy,
                    "event_weight": event_weight,
                    "selection_rate_pct": pct(len(part) / len(group)) if len(group) else np.nan,
                    **stats,
                }
            )
            curve = event_curve(part, event_weight)
            if not curve.empty:
                curve_parts.append(curve)

        base = policy_stats["RULE_ALL_OOS_EVENTS"]
        selected = policy_stats["ML_THRESHOLD_SELECTED"]
        summary_rows.append(
            {
                "candidate_scope": candidate_scope,
                "horizon_days": int(horizon),
                "model_name": model_name,
                "policy": "ML_SELECTED_MINUS_RULE_ALL",
                "event_weight": event_weight,
                "selection_rate_pct": pct(selected["event_count"] / base["event_count"]) if base["event_count"] else np.nan,
                "event_count": selected["event_count"],
                "success_rate_pct": selected.get("success_rate_pct", np.nan) - base.get("success_rate_pct", np.nan),
                "mean_net_return_pct": selected.get("mean_net_return_pct", np.nan) - base.get("mean_net_return_pct", np.nan),
                "median_net_return_pct": selected.get("median_net_return_pct", np.nan) - base.get("median_net_return_pct", np.nan),
                "mean_expected_r": selected.get("mean_expected_r", np.nan) - base.get("mean_expected_r", np.nan),
                "profit_factor": np.nan,
                "stop_rate_pct": selected.get("stop_rate_pct", np.nan) - base.get("stop_rate_pct", np.nan),
                "hit_1r_rate_pct": selected.get("hit_1r_rate_pct", np.nan) - base.get("hit_1r_rate_pct", np.nan),
                "hit_2r_rate_pct": selected.get("hit_2r_rate_pct", np.nan) - base.get("hit_2r_rate_pct", np.nan),
                "cumulative_weighted_return_pct": selected.get("cumulative_weighted_return_pct", np.nan) - base.get("cumulative_weighted_return_pct", np.nan),
                "max_event_curve_drawdown_pct": selected.get("max_event_curve_drawdown_pct", np.nan) - base.get("max_event_curve_drawdown_pct", np.nan),
            }
        )
    summary = pd.DataFrame(summary_rows)
    curves = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame()
    return summary, curves


def event_exit_date(row: pd.Series) -> pd.Timestamp:
    if "label_exit_date" in row and pd.notna(row.get("label_exit_date")):
        return pd.Timestamp(row["label_exit_date"]).normalize()
    entry = pd.Timestamp(row.get("label_entry_date")) if "label_entry_date" in row and pd.notna(row.get("label_entry_date")) else pd.Timestamp(row["date"]) + pd.offsets.BDay(1)
    return (entry + pd.offsets.BDay(int(row.get("horizon_days", 20)))).normalize()


def event_entry_date(row: pd.Series) -> pd.Timestamp:
    if "label_entry_date" in row and pd.notna(row.get("label_entry_date")):
        return pd.Timestamp(row["label_entry_date"]).normalize()
    return (pd.Timestamp(row["date"]) + pd.offsets.BDay(1)).normalize()


def build_non_overlap_event_portfolio(oos: pd.DataFrame, event_weight: float) -> pd.DataFrame:
    rows: List[Dict] = []
    group_cols = ["candidate_scope", "horizon_days", "model_name"]
    for keys, group in oos.groupby(group_cols, dropna=False):
        candidate_scope, horizon, model_name = keys
        for policy, part in [
            ("RULE_NON_OVERLAP", group),
            ("ML_SELECTED_NON_OVERLAP", group[group["selected_by_threshold"]].copy()),
        ]:
            if part.empty:
                continue
            equity = 1.0
            peak = 1.0
            open_until: Dict[str, pd.Timestamp] = {}
            part = part.copy()
            part["_entry_date"] = part.apply(event_entry_date, axis=1)
            part["_exit_date"] = part.apply(event_exit_date, axis=1)
            for _, row in part.sort_values(["_entry_date", "date", "signal_idx"]).iterrows():
                symbol = str(row.get("symbol", "TSM")).upper()
                entry_date = pd.Timestamp(row["_entry_date"]).normalize()
                exit_date = pd.Timestamp(row["_exit_date"]).normalize()
                previous_exit = open_until.get(symbol)
                accepted = previous_exit is None or entry_date > previous_exit
                event_return = pd.to_numeric(pd.Series([row["label_net_return_pct"]]), errors="coerce").iloc[0]
                weighted_return = 0.0
                skip_reason = ""
                if accepted and pd.notna(event_return):
                    weighted_return = float(event_return) / 100.0 * event_weight
                    equity *= 1.0 + weighted_return
                    peak = max(peak, equity)
                    open_until[symbol] = exit_date
                elif not accepted:
                    skip_reason = "OVERLAPPING_OPEN_EVENT"
                else:
                    skip_reason = "MISSING_EVENT_RETURN"
                rows.append(
                    {
                        "symbol": symbol,
                        "candidate_scope": candidate_scope,
                        "horizon_days": int(horizon),
                        "model_name": model_name,
                        "policy": policy,
                        "signal_idx": int(row["signal_idx"]),
                        "signal_date": row["date"],
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "accepted": bool(accepted and pd.notna(event_return)),
                        "skip_reason": skip_reason if skip_reason else "PASS",
                        "label_exit_reason": row.get("label_exit_reason", ""),
                        "event_return_pct": event_return,
                        "event_weight": event_weight,
                        "weighted_event_return_pct": pct(weighted_return),
                        "equity": equity,
                        "drawdown_pct": pct(equity / peak - 1.0),
                    }
                )
    return pd.DataFrame(rows)


def summarize_portfolio_curve(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    if portfolio.empty:
        return pd.DataFrame()
    for keys, group in portfolio.groupby(["candidate_scope", "horizon_days", "model_name", "policy"], dropna=False):
        accepted = group[group["accepted"].astype(bool)].copy()
        returns = pd.to_numeric(accepted["event_return_pct"], errors="coerce").dropna()
        rows.append(
            {
                "benchmark": "|".join([str(v) for v in keys]),
                "benchmark_type": "event_portfolio",
                "event_count": int(len(accepted)),
                "total_return_pct": pct(group["equity"].iloc[-1] - 1.0) if not group.empty else np.nan,
                "max_drawdown_pct": float(pd.to_numeric(group["drawdown_pct"], errors="coerce").min()) if not group.empty else np.nan,
                "mean_event_return_pct": float(returns.mean()) if not returns.empty else np.nan,
                "profit_factor": profit_factor(returns),
            }
        )
    return pd.DataFrame(rows)


def buy_hold_benchmark(path: Path, label: str) -> Dict[str, object]:
    if not path.exists():
        return {"benchmark": label, "benchmark_type": "buy_hold", "event_count": 0, "total_return_pct": np.nan, "max_drawdown_pct": np.nan, "mean_event_return_pct": np.nan, "profit_factor": np.nan}
    frame = strip_bom_columns(pd.read_csv(path, parse_dates=["date"])).sort_values("date")
    if frame.empty or "close" not in frame.columns:
        return {"benchmark": label, "benchmark_type": "buy_hold", "event_count": 0, "total_return_pct": np.nan, "max_drawdown_pct": np.nan, "mean_event_return_pct": np.nan, "profit_factor": np.nan}
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if close.empty:
        return {"benchmark": label, "benchmark_type": "buy_hold", "event_count": 0, "total_return_pct": np.nan, "max_drawdown_pct": np.nan, "mean_event_return_pct": np.nan, "profit_factor": np.nan}
    equity = close / close.iloc[0]
    drawdown = equity / equity.cummax() - 1.0
    returns = close.pct_change().dropna() * 100.0
    return {
        "benchmark": label,
        "benchmark_type": "buy_hold",
        "event_count": int(len(returns)),
        "total_return_pct": pct(equity.iloc[-1] - 1.0),
        "max_drawdown_pct": pct(drawdown.min()),
        "mean_event_return_pct": float(returns.mean()) if not returns.empty else np.nan,
        "profit_factor": profit_factor(returns),
    }


def build_benchmark_comparison(portfolio: pd.DataFrame, enriched: Path, smh_enriched: Path, soxx_enriched: Path) -> pd.DataFrame:
    rows = [
        buy_hold_benchmark(enriched, "TSM_BUY_HOLD"),
        buy_hold_benchmark(smh_enriched, "SMH_BUY_HOLD"),
        buy_hold_benchmark(soxx_enriched, "SOXX_BUY_HOLD"),
    ]
    portfolio_summary = summarize_portfolio_curve(portfolio)
    if not portfolio_summary.empty:
        rows.extend(portfolio_summary.to_dict("records"))
    return pd.DataFrame(rows)


def check_row(check: str, passed: bool, severity: str, value, details: str = "") -> Dict:
    return {"check": check, "passed": bool(passed), "severity": severity, "value": value, "details": details}


def build_quality_checks(oos: pd.DataFrame, summary: pd.DataFrame, curves: pd.DataFrame, portfolio: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = [
        check_row("oos_predictions_non_empty", not oos.empty, "CRITICAL", len(oos)),
        check_row("overlay_summary_non_empty", not summary.empty, "CRITICAL", len(summary)),
        check_row("overlay_curves_non_empty", not curves.empty, "WARN", len(curves)),
        check_row("non_overlap_event_portfolio_non_empty", portfolio is not None and not portfolio.empty, "WARN", 0 if portfolio is None else len(portfolio)),
    ]
    if not oos.empty:
        rows.append(check_row("oos_dates_after_train_end", bool((oos["date"] > oos["train_end_date"]).all()), "CRITICAL", int((oos["date"] <= oos["train_end_date"]).sum())))
        selected = int(oos["selected_by_threshold"].sum())
        rows.append(check_row("ml_selected_events_positive", selected > 0, "WARN", selected, "Some models may select no events; this is a warning unless every policy is empty."))
    if not summary.empty:
        delta = summary[summary["policy"].eq("ML_SELECTED_MINUS_RULE_ALL")].copy()
        improved = int((pd.to_numeric(delta["mean_net_return_pct"], errors="coerce") > 0).sum()) if not delta.empty else 0
        rows.append(check_row("at_least_one_overlay_mean_return_improvement", improved > 0, "WARN", improved))
    if portfolio is not None and not portfolio.empty:
        accepted = portfolio[portfolio["accepted"].astype(bool)]
        rows.append(check_row("non_overlap_accepted_events_positive", not accepted.empty, "WARN", len(accepted)))
    return pd.DataFrame(rows)


def fmt(value, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    if value == np.inf:
        return "inf"
    return f"{float(value):.{digits}f}"


def write_report(outdir: Path, summary: pd.DataFrame, quality: pd.DataFrame) -> None:
    lines = [
        "# TSMC ML Overlay Backtest Report",
        "",
        "This report evaluates OOS prediction-filtered event streams. It is not an order engine.",
        "",
        "## Quality",
        "",
        "| Check | Passed | Severity | Value |",
        "|---|---:|---|---:|",
    ]
    for _, row in quality.iterrows():
        lines.append(f"| {row['check']} | {row['passed']} | {row['severity']} | {row['value']} |")

    display = summary[summary["policy"].isin(["RULE_ALL_OOS_EVENTS", "ML_THRESHOLD_SELECTED"])].copy() if not summary.empty else pd.DataFrame()
    lines.extend(
        [
            "",
            "## Policy Summary",
            "",
            "| Scope | Horizon | Model | Policy | Events | Mean Return | Success | Stop | Profit Factor | Weighted Return | MDD |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if display.empty:
        lines.append("| NA | NA | NA | NA | 0 | NA | NA | NA | NA | NA | NA |")
    else:
        for _, row in display.sort_values(["candidate_scope", "horizon_days", "model_name", "policy"]).iterrows():
            lines.append(
                f"| {row['candidate_scope']} | {int(row['horizon_days'])} | {row['model_name']} | {row['policy']} | "
                f"{int(row['event_count'])} | {fmt(row['mean_net_return_pct'])}% | {fmt(row['success_rate_pct'])}% | "
                f"{fmt(row.get('stop_rate_pct'))}% | {fmt(row['profit_factor'])} | {fmt(row['cumulative_weighted_return_pct'])}% | "
                f"{fmt(row['max_event_curve_drawdown_pct'])}% |"
            )
    lines.extend(["", "This report is research tooling, not investment advice."])
    (outdir / "tsm_ml_overlay_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ML overlay OOS event performance.")
    parser.add_argument("--oos-predictions", default="tsm_price_rule_output/tsm_prediction_oos_predictions.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--event-weight", type=float, default=0.08)
    parser.add_argument("--enriched", default="output/tsm_daily_10y_enriched.csv")
    parser.add_argument("--smh-enriched", default="output/universe/SMH/tsm_daily_10y_enriched.csv")
    parser.add_argument("--soxx-enriched", default="output/universe/SOXX/tsm_daily_10y_enriched.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    oos = read_oos_predictions(Path(args.oos_predictions))
    event_weight = float(np.clip(args.event_weight, 0.0, 1.0))
    summary, curves = build_overlay_outputs(oos, event_weight)
    portfolio = build_non_overlap_event_portfolio(oos, event_weight)
    benchmark = build_benchmark_comparison(portfolio, Path(args.enriched), Path(args.smh_enriched), Path(args.soxx_enriched))
    quality = build_quality_checks(oos, summary, curves, portfolio)
    summary.to_csv(outdir / "tsm_ml_overlay_summary.csv", index=False)
    curves.to_csv(outdir / "tsm_ml_overlay_equity_curves.csv", index=False)
    portfolio.to_csv(outdir / "tsm_event_portfolio_backtest.csv", index=False)
    benchmark.to_csv(outdir / "tsm_prediction_benchmark_comparison.csv", index=False)
    quality.to_csv(outdir / "tsm_ml_overlay_quality_checks.csv", index=False)
    write_report(outdir, summary, quality)
    print("완료: ML overlay outputs =", outdir.resolve())
    if not summary.empty:
        print(summary[["candidate_scope", "horizon_days", "model_name", "policy", "event_count", "mean_net_return_pct"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()

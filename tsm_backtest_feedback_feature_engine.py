#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prefix-only feedback feature engine.

Inputs:
- tsm_backtest_event_ledger.csv
- optional tsm_pooled_model_oof_predictions.csv
- optional validation outputs

Outputs:
- tsm_backtest_feedback_features.csv
- tsm_backtest_feedback_quality_checks.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    kwargs.setdefault("low_memory", False)
    try:
        out = pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    out.columns = [str(c).lstrip("\ufeff") for c in out.columns]
    return out


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def check_row(check: str, passed: bool, value: object = "", threshold: object = "", detail: str = "") -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "value": value, "threshold": threshold, "detail": detail}


def prefix_ewm(frame: pd.DataFrame, group_cols: list[str], value_col: str, output_col: str, span: int) -> pd.Series:
    return (
        frame.groupby(group_cols, dropna=False)[value_col]
        .transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).ewm(span=span, min_periods=1, adjust=False).mean())
        .rename(output_col)
    )


def prefix_rolling_mean(frame: pd.DataFrame, group_cols: list[str], value_col: str, output_col: str, window: int, min_periods: int = 1) -> pd.Series:
    return (
        frame.groupby(group_cols, dropna=False)[value_col]
        .transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window, min_periods=min_periods).mean())
        .rename(output_col)
    )


def prefix_rolling_min(frame: pd.DataFrame, group_cols: list[str], value_col: str, output_col: str, window: int, min_periods: int = 1) -> pd.Series:
    return (
        frame.groupby(group_cols, dropna=False)[value_col]
        .transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window, min_periods=min_periods).min())
        .rename(output_col)
    )


def add_business_days(dates: pd.Series, days: pd.Series | int, default_days: int = 20) -> pd.Series:
    date_values = pd.to_datetime(dates, errors="coerce")
    day_values = pd.Series(days, index=date_values.index) if not isinstance(days, pd.Series) else days.reindex(date_values.index)
    out = []
    for date_value, day_value in zip(date_values, day_values):
        if pd.isna(date_value):
            out.append(pd.NaT)
            continue
        try:
            offset_days = int(day_value) if pd.notna(day_value) else int(default_days)
        except Exception:
            offset_days = int(default_days)
        out.append(pd.Timestamp(date_value) + pd.offsets.BDay(max(offset_days, 1)))
    return pd.Series(out, index=date_values.index)


def feedback_available_date(frame: pd.DataFrame, default_horizon_days: int = 20) -> pd.Series:
    if "exit_date" in frame.columns:
        exit_date = pd.to_datetime(frame["exit_date"], errors="coerce")
    elif "label_exit_date_20d" in frame.columns:
        exit_date = pd.to_datetime(frame["label_exit_date_20d"], errors="coerce")
    elif "label_exit_date" in frame.columns:
        exit_date = pd.to_datetime(frame["label_exit_date"], errors="coerce")
    else:
        exit_date = pd.Series(pd.NaT, index=frame.index)
    horizon = pd.to_numeric(frame.get("horizon_days", default_horizon_days), errors="coerce") if "horizon_days" in frame.columns else default_horizon_days
    fallback = add_business_days(frame["date"], horizon, default_days=default_horizon_days)
    return exit_date.fillna(fallback)


def group_key(frame: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    if not group_cols:
        return pd.Series("", index=frame.index)
    return frame[group_cols].fillna("UNKNOWN").astype(str).agg("\x1f".join, axis=1)


def merge_available_history(frame: pd.DataFrame, history: pd.DataFrame, group_cols: list[str], feature_col: str) -> pd.Series:
    values = pd.Series(np.nan, index=frame.index, dtype=float)
    if frame.empty or history.empty:
        return values.rename(feature_col)
    left = frame[["_row_id", "date", *group_cols]].copy()
    right = history[["_feedback_available_date", feature_col, *group_cols]].dropna(subset=["_feedback_available_date"]).copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").astype("datetime64[ns]")
    right["_feedback_available_date"] = pd.to_datetime(right["_feedback_available_date"], errors="coerce").astype("datetime64[ns]")
    if right.empty:
        return values.rename(feature_col)
    left["_feedback_group_key"] = group_key(left, group_cols)
    right["_feedback_group_key"] = group_key(right, group_cols)
    for key, left_group in left.groupby("_feedback_group_key", dropna=False):
        right_group = right[right["_feedback_group_key"].eq(key)].sort_values("_feedback_available_date")
        if right_group.empty:
            continue
        merged = pd.merge_asof(
            left_group.sort_values("date"),
            right_group[["_feedback_available_date", feature_col]].rename(columns={"_feedback_available_date": "date"}),
            on="date",
            direction="backward",
            allow_exact_matches=False,
        )
        values.loc[merged["_row_id"].to_numpy()] = pd.to_numeric(merged[feature_col], errors="coerce").to_numpy()
    return values.rename(feature_col)


def availability_ewm(frame: pd.DataFrame, group_cols: list[str], value_col: str, output_col: str, span: int) -> pd.Series:
    history = frame[["_feedback_available_date", value_col, *group_cols]].dropna(subset=["_feedback_available_date"]).copy()
    if history.empty:
        return pd.Series(np.nan, index=frame.index, name=output_col)
    history = history.sort_values([*group_cols, "_feedback_available_date"], kind="mergesort").reset_index(drop=True)
    history[output_col] = history.groupby(group_cols, dropna=False)[value_col].transform(
        lambda s: pd.to_numeric(s, errors="coerce").ewm(span=span, min_periods=1, adjust=False).mean()
    )
    return merge_available_history(frame, history, group_cols, output_col)


def availability_rolling_mean(frame: pd.DataFrame, group_cols: list[str], value_col: str, output_col: str, window: int, min_periods: int = 1) -> pd.Series:
    history = frame[["_feedback_available_date", value_col, *group_cols]].dropna(subset=["_feedback_available_date"]).copy()
    if history.empty:
        return pd.Series(np.nan, index=frame.index, name=output_col)
    history = history.sort_values([*group_cols, "_feedback_available_date"], kind="mergesort").reset_index(drop=True)
    history[output_col] = history.groupby(group_cols, dropna=False)[value_col].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(window, min_periods=min_periods).mean()
    )
    return merge_available_history(frame, history, group_cols, output_col)


def availability_rolling_min(frame: pd.DataFrame, group_cols: list[str], value_col: str, output_col: str, window: int, min_periods: int = 1) -> pd.Series:
    history = frame[["_feedback_available_date", value_col, *group_cols]].dropna(subset=["_feedback_available_date"]).copy()
    if history.empty:
        return pd.Series(np.nan, index=frame.index, name=output_col)
    history = history.sort_values([*group_cols, "_feedback_available_date"], kind="mergesort").reset_index(drop=True)
    history[output_col] = history.groupby(group_cols, dropna=False)[value_col].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(window, min_periods=min_periods).min()
    )
    return merge_available_history(frame, history, group_cols, output_col)


def prepare_oof_daily_feedback(oof: pd.DataFrame) -> pd.DataFrame:
    if oof.empty:
        return pd.DataFrame()
    frame = oof.copy()
    if "date" not in frame.columns:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if "symbol" not in frame.columns:
        frame["symbol"] = "TSM"
    prob_col = next((c for c in ["p_success_20d", "p_success", "decision_score", "p_success_calibrated"] if c in frame.columns), "")
    label_col = next((c for c in ["label_success_20d", "actual_success", "label_success"] if c in frame.columns), "")
    return_col = next((c for c in ["label_net_return_pct_20d", "actual_return_pct", "net_return_pct"] if c in frame.columns), "")
    selected_col = next((c for c in ["selected_by_threshold", "is_selected", "model_selected", "selected"] if c in frame.columns), "")
    if prob_col and label_col:
        frame["_prob"] = pd.to_numeric(frame[prob_col], errors="coerce")
        frame["_label"] = pd.to_numeric(frame[label_col], errors="coerce")
        frame["_calibration_residual"] = frame["_label"] - frame["_prob"]
    else:
        frame["_calibration_residual"] = np.nan
    if return_col and selected_col:
        frame["_return"] = pd.to_numeric(frame[return_col], errors="coerce")
        frame["_selected"] = frame[selected_col].map(to_bool)
        daily_all = frame.groupby(["symbol", "date"])["_return"].mean().rename("_all_return")
        daily_selected = frame[frame["_selected"]].groupby(["symbol", "date"])["_return"].mean().rename("_selected_return")
        uplift = pd.concat([daily_all, daily_selected], axis=1).reset_index()
        uplift["_selected_uplift"] = uplift["_selected_return"] - uplift["_all_return"]
    else:
        uplift = pd.DataFrame(columns=["symbol", "date", "_selected_uplift"])
    residual = frame.groupby(["symbol", "date"])["_calibration_residual"].mean().reset_index()
    daily = residual.merge(uplift[["symbol", "date", "_selected_uplift"]], on=["symbol", "date"], how="outer")
    daily["date"] = feedback_available_date(daily, default_horizon_days=20)
    daily = daily.sort_values(["symbol", "date"]).reset_index(drop=True)
    daily["fb_model_calibration_residual_ewm_120"] = (
        daily.groupby("symbol")["_calibration_residual"].transform(lambda s: pd.to_numeric(s, errors="coerce").ewm(span=120, min_periods=1, adjust=False).mean())
    )
    daily["fb_threshold_selected_uplift_ewm_120"] = (
        daily.groupby("symbol")["_selected_uplift"].transform(lambda s: pd.to_numeric(s, errors="coerce").ewm(span=120, min_periods=1, adjust=False).mean())
    )
    return daily[["symbol", "date", "fb_model_calibration_residual_ewm_120", "fb_threshold_selected_uplift_ewm_120"]]


def merge_prior_daily_feedback(ledger: pd.DataFrame, daily_feedback: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or daily_feedback.empty:
        out = ledger.copy()
        out["fb_model_calibration_residual_ewm_120"] = np.nan
        out["fb_threshold_selected_uplift_ewm_120"] = np.nan
        return out
    frames: List[pd.DataFrame] = []
    feedback = daily_feedback.sort_values(["symbol", "date"]).copy()
    feedback["date"] = pd.to_datetime(feedback["date"], errors="coerce").astype("datetime64[ns]")
    for symbol, group in ledger.groupby("symbol", dropna=False):
        left = group.sort_values("date").copy()
        left["date"] = pd.to_datetime(left["date"], errors="coerce").astype("datetime64[ns]")
        right = feedback[feedback["symbol"].astype(str).eq(str(symbol))].sort_values("date")
        if right.empty:
            left["fb_model_calibration_residual_ewm_120"] = np.nan
            left["fb_threshold_selected_uplift_ewm_120"] = np.nan
        else:
            left = pd.merge_asof(
                left,
                right.drop(columns=["symbol"]),
                on="date",
                direction="backward",
                allow_exact_matches=False,
            )
        frames.append(left)
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date", "strategy_id", "variant_id"]).reset_index(drop=True)


def build_feedback_features(ledger: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    out = ledger.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "symbol" not in out.columns:
        out["symbol"] = "TSM"
    for col in ["symbol_group", "entry_trigger", "candidate_tier", "strategy_id", "strategy_group"]:
        if col not in out.columns:
            out[col] = "UNKNOWN"
        out[col] = out[col].fillna("UNKNOWN").astype(str)
    net_return = pd.to_numeric(out.get("net_return_pct", np.nan), errors="coerce")
    out["_event_success"] = np.where(net_return.notna(), (net_return > 0).astype(float), np.nan)
    out["_stop_hit"] = out.get("stop_hit", False).map(to_bool).astype(float) if "stop_hit" in out.columns else 0.0
    out["_expected_r"] = pd.to_numeric(out.get("realized_r_multiple", np.nan), errors="coerce")
    out["_net_return_pct"] = pd.to_numeric(out.get("net_return_pct", np.nan), errors="coerce")
    out = out.sort_values(["symbol", "date", "strategy_id", "variant_id"]).reset_index(drop=True)
    out["_row_id"] = np.arange(len(out))
    out["_feedback_available_date"] = feedback_available_date(out)

    out["fb_trigger_success_rate_ewm_60"] = availability_ewm(out, ["symbol_group", "entry_trigger"], "_event_success", "fb_trigger_success_rate_ewm_60", 60)
    out["fb_trigger_expected_r_ewm_60"] = availability_ewm(out, ["symbol_group", "entry_trigger"], "_expected_r", "fb_trigger_expected_r_ewm_60", 60)
    out["fb_candidate_tier_success_rate_ewm_120"] = availability_ewm(out, ["symbol_group", "candidate_tier"], "_event_success", "fb_candidate_tier_success_rate_ewm_120", 120)
    out["fb_symbol_stop_rate_252"] = availability_rolling_mean(out, ["symbol"], "_stop_hit", "fb_symbol_stop_rate_252", 252)
    out["fb_symbol_expected_r_20"] = availability_rolling_mean(out, ["symbol"], "_expected_r", "fb_symbol_expected_r_20", 20)
    out["fb_symbol_expected_r_60"] = availability_rolling_mean(out, ["symbol"], "_expected_r", "fb_symbol_expected_r_60", 60)
    out["fb_symbol_expected_r_120"] = availability_rolling_mean(out, ["symbol"], "_expected_r", "fb_symbol_expected_r_120", 120)
    out["fb_strategy_drawdown_sensitivity_252"] = availability_rolling_min(out, ["symbol", "strategy_id"], "_net_return_pct", "fb_strategy_drawdown_sensitivity_252", 252)

    daily_feedback = prepare_oof_daily_feedback(oof)
    out = merge_prior_daily_feedback(out, daily_feedback)
    helper_cols = [c for c in out.columns if c.startswith("_")]
    return out.drop(columns=helper_cols)


def prefix_self_test() -> list[dict[str, object]]:
    test = pd.DataFrame(
        {
            "symbol": ["X", "X", "X"],
            "symbol_group": ["G", "G", "G"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "strategy_id": ["S", "S", "S"],
            "strategy_group": ["SG", "SG", "SG"],
            "variant_id": ["V", "V", "V"],
            "entry_trigger": ["A", "A", "A"],
            "candidate_tier": ["decision_trade_ready", "decision_trade_ready", "decision_trade_ready"],
            "net_return_pct": [10.0, -10.0, 20.0],
            "realized_r_multiple": [1.0, -1.0, 2.0],
            "stop_hit": [False, True, False],
            "exit_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        }
    )
    features = build_feedback_features(test, pd.DataFrame())
    first = features["fb_trigger_success_rate_ewm_60"].iloc[0]
    second = features["fb_trigger_success_rate_ewm_60"].iloc[1]
    third_expected_r = features["fb_symbol_expected_r_20"].iloc[2]
    return [
        check_row("feedback_prefix_first_row_no_own_label", pd.isna(first), first, "NaN"),
        check_row("feedback_prefix_second_row_uses_prior_label", math.isclose(float(second), 1.0, rel_tol=1e-9), second, "1.0"),
        check_row("feedback_prefix_third_row_uses_prior_mean", math.isclose(float(third_expected_r), 0.0, abs_tol=1e-9), third_expected_r, "0.0"),
    ]


def build_quality_checks(ledger: pd.DataFrame, features: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    rows = [
        check_row("event_ledger_non_empty", not ledger.empty, len(ledger), ">0"),
        check_row("feedback_features_non_empty", not features.empty, len(features), ">0"),
        check_row("oof_predictions_available_or_optional", True, len(oof), ">0 preferred", "Ledger-only feedback remains valid when OOF predictions are unavailable."),
    ]
    rows.extend(prefix_self_test())
    expected_cols = [
        "fb_trigger_success_rate_ewm_60",
        "fb_symbol_stop_rate_252",
        "fb_symbol_expected_r_20",
        "fb_symbol_expected_r_60",
        "fb_symbol_expected_r_120",
        "fb_model_calibration_residual_ewm_120",
        "fb_threshold_selected_uplift_ewm_120",
        "fb_strategy_drawdown_sensitivity_252",
    ]
    rows.append(check_row("feedback_feature_columns_present", all(c in features.columns for c in expected_cols), ",".join([c for c in expected_cols if c in features.columns]), ",".join(expected_cols)))
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prefix-only backtest feedback features.")
    parser.add_argument("--ledger", default="tsm_price_rule_output/tsm_backtest_event_ledger.csv")
    parser.add_argument("--oof-predictions", default="tsm_price_rule_output/tsm_pooled_model_oof_predictions.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--output-name", default="tsm_backtest_feedback_features.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ledger = read_csv(Path(args.ledger), parse_dates=["date"])
    oof = read_csv(Path(args.oof_predictions), parse_dates=["date"])
    features = build_feedback_features(ledger, oof)
    quality = build_quality_checks(ledger, features, oof)
    features.to_csv(outdir / args.output_name, index=False)
    quality.to_csv(outdir / "tsm_backtest_feedback_quality_checks.csv", index=False)
    print("completed: backtest feedback features =", (outdir / args.output_name).resolve(), "rows=", len(features))
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()

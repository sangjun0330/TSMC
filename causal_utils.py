from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def rolling_winsorize(
    s: pd.Series,
    window: int = 756,
    min_periods: int = 252,
    p_low: float = 0.01,
    p_high: float = 0.99,
) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce")
    lo = values.rolling(window=window, min_periods=min_periods).quantile(p_low).shift(1)
    hi = values.rolling(window=window, min_periods=min_periods).quantile(p_high).shift(1)
    return values.clip(lower=lo, upper=hi)


def rolling_percentile_rank(
    s: pd.Series,
    window: int = 756,
    min_periods: int = 252,
    high_good: bool = True,
    neutral_value: float = 50.0,
) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce")

    def _rank_last(x: pd.Series) -> float:
        arr = pd.Series(x).dropna()
        if len(arr) < min_periods:
            return np.nan
        last = arr.iloc[-1]
        pct = (arr <= last).mean() * 100.0
        return float(pct if high_good else 100.0 - pct)

    out = values.rolling(window=window, min_periods=min_periods).apply(_rank_last, raw=False)
    return out.fillna(neutral_value)


def causal_rolling_quantile(
    s: pd.Series,
    q: float,
    window: int = 756,
    min_periods: int = 252,
) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce")
    return values.rolling(window=window, min_periods=min_periods).quantile(q).shift(1)


def assert_prefix_stability(
    make_signals_fn: Callable[[pd.DataFrame], pd.DataFrame],
    df: pd.DataFrame,
    check_rows: int = 1000,
    check_cols: list[str] | None = None,
) -> bool:
    if check_cols is None:
        check_cols = [
            "algo_vol_extreme",
            "algo_vol_high",
            "algo_event_shock_day",
            "score_momentum",
            "score_low_vol",
            "score_relative_strength",
            "score_liquidity",
            "score_price_algo_total",
            "entry_trigger",
            "trade_action",
        ]
    full = make_signals_fn(df).iloc[:check_rows].reset_index(drop=True)
    trunc = make_signals_fn(df.iloc[:check_rows].copy()).reset_index(drop=True)
    failures: dict[str, dict[str, int]] = {}
    for col in check_cols:
        if col not in full.columns or col not in trunc.columns:
            continue
        a = full[col]
        b = trunc[col]
        if pd.api.types.is_bool_dtype(a) or pd.api.types.is_object_dtype(a):
            diff = a.astype(str) != b.astype(str)
        elif pd.api.types.is_numeric_dtype(a):
            diff = ((a - b).abs() > 1e-9) & ~(a.isna() & b.isna())
        else:
            diff = a.astype(str) != b.astype(str)
        count = int(diff.sum())
        if count > 0:
            first_idx = int(np.where(diff.to_numpy())[0][0])
            failures[col] = {"diff_count": count, "first_idx": first_idx}
    if failures:
        raise AssertionError(f"Prefix stability failed: {failures}")
    return True

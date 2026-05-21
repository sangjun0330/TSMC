from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


def bounded_probability(value: float) -> float:
    if pd.isna(value):
        return np.nan
    return float(np.clip(value, 1e-6, 1.0 - 1e-6))


def wilson_interval(p: float, n: int, z: float = 1.2815515655446004) -> tuple[float, float]:
    if pd.isna(p) or n <= 0:
        return np.nan, np.nan
    p = bounded_probability(p)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denom
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def safe_nanmean(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if len(arr) else np.nan


def expected_calibration_error(bins: pd.DataFrame) -> float:
    if bins.empty or "n" not in bins.columns or bins["n"].sum() == 0:
        return np.nan
    return float((bins["abs_calibration_error"] * bins["n"]).sum() / bins["n"].sum())


def adaptive_calibration_bins(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_prob: Sequence[float] | np.ndarray | pd.Series,
    target_min_bin_n: int = 30,
    max_bins: int = 10,
) -> pd.DataFrame:
    """Equal-frequency calibration bins sized for decision-gate sample support."""
    tmp = pd.DataFrame(
        {
            "y": pd.to_numeric(pd.Series(y_true), errors="coerce"),
            "p": pd.to_numeric(pd.Series(y_prob), errors="coerce").clip(0.0, 1.0),
        }
    ).dropna(subset=["y", "p"])
    if tmp.empty:
        return pd.DataFrame(
            columns=[
                "bin_id",
                "n",
                "min_predicted_probability",
                "max_predicted_probability",
                "mean_predicted_probability",
                "observed_success_rate",
                "observed_success_lower_80",
                "observed_success_upper_80",
                "abs_calibration_error",
                "calibration_error_lower_adjusted",
                "binning",
            ]
        )
    tmp = tmp.sort_values("p").reset_index(drop=True)
    target_min_bin_n = max(1, int(target_min_bin_n))
    max_bins = max(1, int(max_bins))
    bin_count = max(1, min(max_bins, len(tmp) // target_min_bin_n))
    if bin_count == 1:
        tmp["bin"] = 0
    else:
        tmp["bin"] = pd.qcut(tmp.index, q=bin_count, labels=False, duplicates="drop")
    rows = []
    for bin_id, group in tmp.groupby("bin", dropna=False):
        n = int(len(group))
        mean_p = float(group["p"].mean())
        observed = float(group["y"].mean())
        lower, upper = wilson_interval(observed, n)
        rows.append(
            {
                "bin_id": int(bin_id),
                "n": n,
                "min_predicted_probability": float(group["p"].min()),
                "max_predicted_probability": float(group["p"].max()),
                "mean_predicted_probability": mean_p,
                "observed_success_rate": observed,
                "observed_success_lower_80": lower,
                "observed_success_upper_80": upper,
                "abs_calibration_error": abs(mean_p - observed),
                "calibration_error_lower_adjusted": float(max(lower - mean_p, mean_p - upper, 0.0)),
                "binning": "adaptive_equal_frequency",
            }
        )
    return pd.DataFrame(rows)


def decision_ece(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_prob: Sequence[float] | np.ndarray | pd.Series,
    target_min_bin_n: int = 30,
    max_bins: int = 10,
) -> float:
    return expected_calibration_error(adaptive_calibration_bins(y_true, y_prob, target_min_bin_n, max_bins))


def decision_min_calibration_bin_n(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_prob: Sequence[float] | np.ndarray | pd.Series,
    target_min_bin_n: int = 30,
    max_bins: int = 10,
) -> int:
    bins = adaptive_calibration_bins(y_true, y_prob, target_min_bin_n, max_bins)
    if bins.empty:
        return 0
    return int(pd.to_numeric(bins["n"], errors="coerce").min())


def calibration_binning_primary() -> str:
    return "adaptive_equal_frequency"

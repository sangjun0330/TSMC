#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pooled semiconductor 20D trade-ready prediction model.

This engine uses the pooled cross-sectional dataset only for alpha research
decision support. It does not place orders and it never enables live trading.

Outputs:
- tsm_pooled_model_comparison.csv
- tsm_pooled_model_oos_predictions.csv
- tsm_pooled_model_oof_predictions.csv
- tsm_pooled_model_slice_diagnostics.csv
- tsm_pooled_tsm_calibration.csv
- tsm_tsm_like_calibration_pool.csv
- tsm_tsm_like_calibration_metrics.csv
- tsm_paper_gate_snapshot.csv
- tsm_pooled_learning_curve_report.csv
- tsm_pooled_latest_prediction_overlay.csv
- tsm_pooled_model_quality_checks.csv
- tsm_pooled_model_report.md
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from tsm_core.metrics import (
    adaptive_calibration_bins as adaptive_calibration_bins_core,
    calibration_binning_primary as calibration_binning_primary_core,
    expected_calibration_error as calibration_ece_from_bins,
)

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    SKLEARN_IMPORT_ERROR = exc

try:
    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

    LIGHTGBM_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None
    early_stopping = None
    log_evaluation = None
    LIGHTGBM_IMPORT_ERROR = exc

try:
    from xgboost import XGBClassifier

    XGBOOST_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc


HORIZON = 20
TARGET_COL = "label_success_20d"
RETURN_COL = "label_net_return_pct_20d"
EXPECTED_R_COL = "label_expected_r_20d"
STOP_SURVIVAL_COL = "label_stop_survival_20d"
HIT_1R_COL = "label_hit_1r_before_stop_20d"
HIT_2R_COL = "label_hit_2r_before_stop_20d"
TRADE_READY_COL = "is_trade_ready_entry_candidate"
MODEL_TRAINING_COL = "is_model_training_candidate"
DECISION_ENTRY_COL = "is_decision_entry_candidate"
GROUP_COLS = ["entry_trigger", "trend_regime", "vol_regime", "drawdown_bucket"]
HIERARCHICAL_GROUP_COLS = ["candidate_tier", *GROUP_COLS]
ALLOWED_ID_COLS = {
    "symbol",
    "symbol_group",
    "date",
    "date_split",
    "split",
    "signal_idx",
}
FILTER_ONLY_COLS = {
    "is_event_candidate",
    "is_actionable_entry_candidate",
    "is_trade_ready_entry_candidate",
    "is_entry_research_candidate",
    "is_risk_research_candidate",
    "is_model_training_candidate",
    "is_decision_entry_candidate",
}
FORBIDDEN_POOLED_FEATURE_PATTERNS = (
    "label_",
    "fwd_",
    "forward",
    "future",
    "next_",
    "actual_return",
    "exit_",
    "net_return",
    "gross_return",
    "r_multiple",
    "success",
    "stop_hit",
    "stop_survival",
    "hit_1r",
    "hit_2r",
)

TRAIN_END = pd.Timestamp("2022-12-31")
VALIDATION_START = pd.Timestamp("2023-01-01")
VALIDATION_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2024-12-31")
HOLDOUT_START = pd.Timestamp("2025-01-01")

PRIOR_STRENGTH = 40.0
TSM_LAYER_PRIOR_STRENGTH = 40.0
SYMBOL_GROUP_LAYER_PRIOR_STRENGTH = 100.0
TSM_SHRINKAGE_PRIOR_STRENGTH = 100.0
THRESHOLDS = tuple(np.round(np.arange(0.30, 0.81, 0.05), 2))
DEFAULT_UTILITY_WEIGHTS = {
    "p_success": 0.70,
    "p_stop_hit": -0.40,
    "expected_r": 0.20,
    "score_price_algo_total": 0.0,
}
UTILITY_WEIGHT_GRID = (DEFAULT_UTILITY_WEIGHTS,)

MIN_POOLED_TRADE_READY_LABELS = 500
MIN_POOLED_MODEL_TRAINING_LABELS = 10000
ENTRY_RESEARCH_EVAL_SCOPE = "entry_research_all"
TRADE_READY_EVAL_SCOPE = "trade_ready_entry_only"
MIN_EVAL_EVENTS = 150
MIN_SELECTED_EVAL_EVENTS = 50
MIN_SELECTED_PER_EVAL_SPLIT = 10
MIN_TSM_CALIBRATION_EVENTS = 30
MIN_TSM_EVAL_EVENTS = 30
MAX_ECE = 0.10
MAX_TSM_ECE = 0.15
MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT = 0.0
MIN_SELECTED_MINUS_ALL_PCT = 0.0
MIN_SELECTED_MINUS_ALL_TARGET_PCT = 0.25
MIN_SELECTED_FRACTION = 0.30
MAX_SELECTED_FRACTION = 0.60
MIN_STOP_RATE_IMPROVEMENT = 0.05
MAX_THRESHOLD_IQR = 0.10
MAX_SELECTION_FRACTION_DRIFT = 0.10
CONSENSUS_SELECTION_FRACTIONS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
SCORE_BASELINE_MIN_SCORE = 75.0
SCORE_BASELINE_FALLBACK_MIN_N = 30
UPLIFT_BOOTSTRAP_ITERATIONS = 200
THRESHOLD_POLICY_BOOTSTRAP_ITERATIONS = 20
MAX_STOP_HIT_FOR_LATEST = 0.35
MIN_EXPECTED_R_FOR_LATEST = 0.35
PAPER_MAX_STOP_HIT_FOR_LATEST = 0.40
PAPER_MAX_ECE = 0.12
PAPER_MIN_SELECTED_EVAL_EVENTS = 100
PAPER_MIN_SELECTED_PER_EVAL_SPLIT = 25
PAPER_MIN_SELECTED_MINUS_ALL_CI_LOWER_PCT = -0.50
MIN_TSM_LIKE_EFFECTIVE_SELECTION_N = 500
MIN_TSM_LIKE_ISOTONIC_EFFECTIVE_N = 1000
MIN_ISOTONIC_CALIBRATION_EVENTS = 1000
PLATT_SHRINKAGE_PRIOR_STRENGTH = 1000.0
STRICT_SCOPE_CALIBRATION_PRIOR_STRENGTH = 250.0
TIER_LAYER_PRIOR_STRENGTH = 150.0
TIER_SAMPLE_WEIGHTS = {
    "decision_trade_ready": 4.0,
    "relaxed_trigger_score65": 1.5,
    "relaxed_trigger_score60": 0.75,
    "setup_context_score65": 1.0,
}
RISK_AWARE_SELECTION_WEIGHT_PROFILES = (
    {"label": "balanced_rank", "w_success": 0.45, "w_stop": 0.30, "w_r": 0.20, "w_rule": 0.05},
    {"label": "stop_heavy_rank", "w_success": 0.35, "w_stop": 0.45, "w_r": 0.15, "w_rule": 0.05},
    {"label": "very_stop_heavy_rank", "w_success": 0.30, "w_stop": 0.55, "w_r": 0.10, "w_rule": 0.05},
    {"label": "rule_stop_rank", "w_success": 0.30, "w_stop": 0.40, "w_r": 0.15, "w_rule": 0.15},
)
SLICE_DIMENSIONS = [
    "candidate_tier",
    "symbol",
    "symbol_group",
    "entry_trigger",
    "trend_regime",
    "vol_regime",
]


@dataclass
class EmpiricalBayesModel:
    global_success: float
    global_stop_survival: float
    global_hit_1r: float
    global_hit_2r: float
    global_expected_r: float
    global_net_return: float
    group_stats: pd.DataFrame
    group_cols: list[str]


@dataclass
class TsmCalibrationLayer:
    event_count: int
    actual_success_rate: float
    predicted_success_rate: float
    posterior_success_rate: float
    logit_shift: float
    shrinkage: float
    status: str


@dataclass
class ShrunkPlattCalibrator:
    model: object
    shrinkage: float


@dataclass
class TsmCalibrationRouteSpec:
    route: str
    p_col: str
    score_col: str
    route_alpha: float | None = None
    route_prior_source: str = ""
    route_sample_weight_policy: str = ""
    layer: TsmCalibrationLayer | None = None
    calibrator: object | None = None
    calibration_method: str = "identity"


@dataclass
class TsmLikeCalibrationRouteSpec:
    route: str
    calibrator: object | None = None
    calibration_method: str = "identity"
    logit_shift: float = 0.0
    shrinkage: float = 0.0
    status: str = ""
    effective_n: float = np.nan


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    train_start: str
    train_end: str
    calibration_start: str
    calibration_end: str
    threshold_start: str
    threshold_end: str
    test_start: str
    test_end: str


EMBARGO_TRADING_DAYS = 20
WALK_FORWARD_FOLDS = [
    WalkForwardFold("wf_2021", "2016-01-01", "2019-12-31", "2020-01-01", "2020-06-30", "2020-07-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    WalkForwardFold("wf_2022", "2016-01-01", "2020-12-31", "2021-01-01", "2021-06-30", "2021-07-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    WalkForwardFold("wf_2023", "2016-01-01", "2021-12-31", "2022-01-01", "2022-06-30", "2022-07-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    WalkForwardFold("wf_2024", "2016-01-01", "2022-12-31", "2023-01-01", "2023-06-30", "2023-07-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    WalkForwardFold("wf_2025_2026", "2016-01-01", "2023-12-31", "2024-01-01", "2024-06-30", "2024-07-01", "2024-12-31", "2025-01-01", "2026-12-31"),
]


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return strip_bom_columns(pd.read_csv(path, **kwargs))


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def safe_float(value, default=np.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def assign_split(date_value: pd.Timestamp) -> str:
    date = pd.Timestamp(date_value)
    if date <= TRAIN_END:
        return "train_2016_2022"
    if VALIDATION_START <= date <= VALIDATION_END:
        return "validation_2023"
    if TEST_START <= date <= TEST_END:
        return "test_2024"
    if date >= HOLDOUT_START:
        return "final_holdout_2025_2026"
    return "gap"


def return_ci_lower(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    if len(clean) == 1:
        return float(clean.iloc[0])
    return float(clean.mean() - 1.96 * clean.std(ddof=1) / math.sqrt(len(clean)))


def bootstrap_mean_diff(
    selected_values: pd.Series,
    baseline_values: pd.Series,
    iterations: int = UPLIFT_BOOTSTRAP_ITERATIONS,
    seed: int = 42,
) -> tuple[float, float]:
    selected = pd.to_numeric(selected_values, errors="coerce").dropna().to_numpy(dtype=float)
    baseline = pd.to_numeric(baseline_values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(selected) == 0 or len(baseline) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    diffs = np.empty(int(iterations), dtype=float)
    for i in range(int(iterations)):
        sample_selected = rng.choice(selected, size=len(selected), replace=True)
        sample_baseline = rng.choice(baseline, size=len(baseline), replace=True)
        diffs[i] = float(sample_selected.mean() - sample_baseline.mean())
    return float(np.quantile(diffs, 0.05)), float(np.mean(diffs <= 0.0))


def score_baseline_returns(frame: pd.DataFrame) -> pd.Series:
    baseline, _ = score_baseline_returns_with_policy(frame)
    return baseline


def score_baseline_mask_with_policy(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    mask = pd.Series(True, index=frame.index, dtype=bool)
    if "score_price_algo_total" not in frame.columns:
        return mask, "all_events_no_score_column"
    score = pd.to_numeric(frame["score_price_algo_total"], errors="coerce")
    baseline_mask = score >= SCORE_BASELINE_MIN_SCORE
    if int(baseline_mask.sum()) >= SCORE_BASELINE_FALLBACK_MIN_N:
        return baseline_mask.astype(bool), f"score_price_algo_total_ge_{SCORE_BASELINE_MIN_SCORE:g}"
    score_clean = score.dropna()
    if score_clean.empty:
        return mask, "all_events_no_score_values"
    fallback_threshold = float(score_clean.quantile(0.60))
    fallback_mask = score >= fallback_threshold
    if not bool(fallback_mask.any()):
        return mask, "all_events_empty_score_fallback"
    return fallback_mask.astype(bool), "score_price_algo_total_top_40pct"


def score_baseline_returns_with_policy(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    mask, policy = score_baseline_mask_with_policy(frame)
    return pd.to_numeric(frame.loc[mask, RETURN_COL], errors="coerce"), policy


def bootstrap_block_column(frame: pd.DataFrame) -> tuple[pd.Series | None, str]:
    if "year_month" in frame.columns:
        return frame["year_month"].astype(str), "year_month"
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.notna().any():
            return dates.dt.to_period("M").astype(str), "year_month"
    return None, ""


def paired_bootstrap_uplift(
    frame: pd.DataFrame,
    selected_mask: pd.Series,
    baseline_mask: pd.Series,
    iterations: int = UPLIFT_BOOTSTRAP_ITERATIONS,
    seed: int = 42,
) -> dict[str, object]:
    empty = {
        "bootstrap_method": "paired_event",
        "bootstrap_block_col": "",
        "selected_minus_all_ci_lower_pct_paired": np.nan,
        "selected_minus_score_baseline_ci_lower_pct_paired": np.nan,
        "uplift_bootstrap_p_value_paired": np.nan,
        "score_baseline_bootstrap_p_value_paired": np.nan,
    }
    if RETURN_COL not in frame.columns:
        return empty
    data = frame.copy()
    data["_selected_bootstrap_mask"] = selected_mask.reindex(data.index).fillna(False).astype(bool)
    data["_score_baseline_bootstrap_mask"] = baseline_mask.reindex(data.index).fillna(False).astype(bool)
    data["_return_bootstrap"] = pd.to_numeric(data[RETURN_COL], errors="coerce")
    data = data.dropna(subset=["_return_bootstrap"])
    if data.empty or not data["_selected_bootstrap_mask"].any() or not data["_score_baseline_bootstrap_mask"].any():
        return empty
    block_values, block_col = bootstrap_block_column(data)
    method = "date_block" if block_values is not None and block_values.nunique(dropna=True) >= 2 else "paired_event"
    rng = np.random.default_rng(seed)
    all_diffs: list[float] = []
    score_diffs: list[float] = []
    returns = data["_return_bootstrap"].to_numpy(dtype=float)
    selected_flags = data["_selected_bootstrap_mask"].to_numpy(dtype=bool)
    baseline_flags = data["_score_baseline_bootstrap_mask"].to_numpy(dtype=bool)
    if method == "date_block":
        block_labels = block_values.reindex(data.index).astype(str).to_numpy()
        blocks = pd.Series(block_labels).dropna().unique()
        block_indices = {block: np.flatnonzero(block_labels == block) for block in blocks}
        for _ in range(int(iterations)):
            sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
            sampled_indices = np.concatenate([block_indices[block] for block in sampled_blocks])
            sample_returns = returns[sampled_indices]
            sample_selected = selected_flags[sampled_indices]
            sample_baseline = baseline_flags[sampled_indices]
            if not sample_selected.any() or not sample_baseline.any():
                continue
            selected_mean = float(sample_returns[sample_selected].mean())
            all_diffs.append(float(selected_mean - sample_returns.mean()))
            score_diffs.append(float(selected_mean - sample_returns[sample_baseline].mean()))
    else:
        positions = np.arange(len(returns))
        for _ in range(int(iterations)):
            sampled_positions = rng.choice(positions, size=len(positions), replace=True)
            sample_returns = returns[sampled_positions]
            sample_selected = selected_flags[sampled_positions]
            sample_baseline = baseline_flags[sampled_positions]
            if not sample_selected.any() or not sample_baseline.any():
                continue
            selected_mean = float(sample_returns[sample_selected].mean())
            all_diffs.append(float(selected_mean - sample_returns.mean()))
            score_diffs.append(float(selected_mean - sample_returns[sample_baseline].mean()))
    all_array = np.asarray(all_diffs, dtype=float)
    score_array = np.asarray(score_diffs, dtype=float)
    return {
        "bootstrap_method": method,
        "bootstrap_block_col": block_col if method == "date_block" else "",
        "selected_minus_all_ci_lower_pct_paired": float(np.quantile(all_array, 0.05)) if len(all_array) else np.nan,
        "selected_minus_score_baseline_ci_lower_pct_paired": float(np.quantile(score_array, 0.05)) if len(score_array) else np.nan,
        "uplift_bootstrap_p_value_paired": float(np.mean(all_array <= 0.0)) if len(all_array) else np.nan,
        "score_baseline_bootstrap_p_value_paired": float(np.mean(score_array <= 0.0)) if len(score_array) else np.nan,
    }


def brier_score(y_true: pd.Series, p: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    pred = pd.to_numeric(p, errors="coerce")
    mask = y.notna() & pred.notna()
    if not mask.any():
        return np.nan
    return float(np.mean((pred[mask] - y[mask]) ** 2))


def ece_score(y_true: pd.Series, p: pd.Series, bins: int = 10) -> tuple[float, int]:
    y = pd.to_numeric(y_true, errors="coerce")
    pred = pd.to_numeric(p, errors="coerce")
    mask = y.notna() & pred.notna()
    if not mask.any():
        return np.nan, 0
    y = y[mask].to_numpy(dtype=float)
    pred = pred[mask].to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    min_bin_n = None
    for i in range(bins):
        if i == bins - 1:
            in_bin = (pred >= edges[i]) & (pred <= edges[i + 1])
        else:
            in_bin = (pred >= edges[i]) & (pred < edges[i + 1])
        n = int(in_bin.sum())
        if n == 0:
            continue
        min_bin_n = n if min_bin_n is None else min(min_bin_n, n)
        ece += (n / len(pred)) * abs(float(pred[in_bin].mean()) - float(y[in_bin].mean()))
    return float(ece), int(min_bin_n or 0)


def decision_calibration_metrics(y_true: pd.Series, p: pd.Series) -> tuple[float, int, pd.DataFrame]:
    bins = adaptive_calibration_bins_core(
        y_true,
        p,
        target_min_bin_n=MIN_TSM_CALIBRATION_EVENTS,
        max_bins=10,
    )
    if bins.empty:
        return np.nan, 0, bins
    return calibration_ece_from_bins(bins), int(pd.to_numeric(bins["n"], errors="coerce").min()), bins


def safe_average_precision(y_true: pd.Series, p: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    pred = pd.to_numeric(p, errors="coerce")
    mask = y.notna() & pred.notna()
    if not mask.any() or y[mask].nunique() < 2 or SKLEARN_IMPORT_ERROR is not None:
        return np.nan
    return float(average_precision_score(y[mask].astype(int), pred[mask]))


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def enrich_pooled_features(features: pd.DataFrame) -> pd.DataFrame:
    try:
        from tsm_pooled_dataset_builder import add_pooled_feature_engineering
    except Exception:
        return features.copy()
    return add_pooled_feature_engineering(features)


def pooled_feature_columns(data: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    blocked_exact = {
        TARGET_COL,
        RETURN_COL,
        EXPECTED_R_COL,
        STOP_SURVIVAL_COL,
        HIT_1R_COL,
        HIT_2R_COL,
        "label_status_20d",
        "label_status_60d",
        "universe_model_training_candidate_ratio",
        "universe_decision_entry_candidate_ratio",
        *ALLOWED_ID_COLS,
        *FILTER_ONLY_COLS,
    }
    for col in data.columns:
        lower = str(col).lower()
        if col in blocked_exact:
            continue
        if any(pattern in lower for pattern in FORBIDDEN_POOLED_FEATURE_PATTERNS):
            continue
        if data[col].isna().mean() > 0.50:
            continue
        if (
            pd.api.types.is_numeric_dtype(data[col])
            or pd.api.types.is_bool_dtype(data[col])
            or pd.api.types.is_string_dtype(data[col])
            or pd.api.types.is_categorical_dtype(data[col])
            or data[col].dtype == "object"
        ):
            cols.append(col)
    return cols


def build_pooled_preprocessor(train: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    if SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError(f"scikit-learn is required for pooled ML models: {SKLEARN_IMPORT_ERROR}")
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def transformed_feature_frame(preprocessor: ColumnTransformer, matrix, index: pd.Index) -> pd.DataFrame:
    try:
        columns = preprocessor.get_feature_names_out()
    except Exception:
        columns = [f"feature_{i}" for i in range(np.asarray(matrix).shape[1])]
    return pd.DataFrame(matrix, columns=columns, index=index)


def transform_pooled_features(candidate: dict[str, object], rows: pd.DataFrame):
    feature_cols = list(candidate["feature_cols"])
    preprocessor = candidate["preprocessor"]
    matrix = preprocessor.transform(rows[feature_cols])
    if candidate.get("predict_as_frame"):
        return transformed_feature_frame(preprocessor, matrix, rows.index)
    return matrix


def clip_probability(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-4, 1.0 - 1e-4)


def logit_array(p) -> np.ndarray:
    p = clip_probability(p)
    return np.log(p / (1.0 - p))


def inv_logit_array(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def fit_probability_calibrator(probabilities, y_true: pd.Series) -> tuple[object | None, str]:
    p = pd.Series(probabilities, dtype=float)
    y = pd.to_numeric(y_true, errors="coerce")
    mask = p.notna() & y.notna()
    p = p[mask]
    y = y[mask].astype(int)
    if len(p) < MIN_TSM_CALIBRATION_EVENTS or y.nunique() < 2 or p.nunique() < 2 or SKLEARN_IMPORT_ERROR is not None:
        return None, "identity_insufficient_sample"
    if len(p) >= MIN_ISOTONIC_CALIBRATION_EVENTS:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(p.to_numpy(dtype=float), y.to_numpy(dtype=int))
        return calibrator, "isotonic"
    model = LogisticRegression(C=0.25, solver="lbfgs", l1_ratio=0.0, max_iter=1000, random_state=42)
    model.fit(logit_array(p).reshape(-1, 1), y.to_numpy(dtype=int))
    shrinkage = float(len(p) / (len(p) + PLATT_SHRINKAGE_PRIOR_STRENGTH))
    return ShrunkPlattCalibrator(model=model, shrinkage=shrinkage), "sigmoid_platt_shrunk"


def apply_probability_calibrator(probabilities, calibrator: object | None, method: str) -> np.ndarray:
    p = clip_probability(probabilities)
    if calibrator is None:
        return p
    if method == "isotonic":
        return clip_probability(calibrator.predict(p))
    if method == "sigmoid_platt_shrunk":
        platt = clip_probability(calibrator.model.predict_proba(logit_array(p).reshape(-1, 1))[:, 1])
        raw_logit = logit_array(p)
        platt_logit = logit_array(platt)
        return clip_probability(inv_logit_array((1.0 - calibrator.shrinkage) * raw_logit + calibrator.shrinkage * platt_logit))
    return clip_probability(calibrator.predict_proba(logit_array(p).reshape(-1, 1))[:, 1])


def fit_logit_shift_with_shrinkage(
    rows: pd.DataFrame,
    p_col: str,
    y_col: str,
    global_base_rate: float,
    prior_strength: int = 100,
    min_events: int = 30,
) -> dict[str, object]:
    clean = rows.dropna(subset=[p_col, y_col]).copy()
    n = len(clean)
    if n < min_events or clean[y_col].nunique() < 2:
        return {
            "status": "INSUFFICIENT_SAMPLE",
            "shift": 0.0,
            "shrinkage": 0.0,
            "event_count": n,
            "actual_success_rate": float(clean[y_col].mean()) if n else np.nan,
            "predicted_success_rate": float(clean[p_col].mean()) if n else np.nan,
            "posterior_success_rate": np.nan,
        }
    observed = float(clean[y_col].astype(int).mean())
    predicted = float(clean[p_col].mean())
    posterior_observed = float((observed * n + global_base_rate * prior_strength) / (n + prior_strength))
    raw_shift = float(logit_array([posterior_observed])[0] - logit_array([predicted])[0])
    shrinkage = float(n / (n + prior_strength))
    return {
        "status": "READY",
        "shift": raw_shift * shrinkage,
        "shrinkage": shrinkage,
        "event_count": n,
        "actual_success_rate": observed,
        "predicted_success_rate": predicted,
        "posterior_success_rate": posterior_observed,
    }


def apply_probability_logit_shift(probabilities, shift: float) -> np.ndarray:
    return clip_probability(inv_logit_array(logit_array(probabilities) + float(shift)))


def fit_symbol_group_calibration(rows: pd.DataFrame, p_col: str, global_base_rate: float) -> dict[str, dict[str, object]]:
    if "symbol_group" not in rows.columns:
        return {}
    layers: dict[str, dict[str, object]] = {}
    for group, group_rows in rows.groupby(rows["symbol_group"].astype(str), dropna=False):
        layers[str(group)] = fit_logit_shift_with_shrinkage(
            group_rows,
            p_col=p_col,
            y_col=TARGET_COL,
            global_base_rate=global_base_rate,
            prior_strength=int(SYMBOL_GROUP_LAYER_PRIOR_STRENGTH),
            min_events=MIN_TSM_CALIBRATION_EVENTS,
        )
    return layers


def apply_group_logit_shift(rows: pd.DataFrame, p_col: str, layers: dict[str, dict[str, object]], group_col: str) -> pd.Series:
    base = pd.to_numeric(rows[p_col], errors="coerce")
    if base.empty or not layers or group_col not in rows.columns:
        return base
    shifts = rows[group_col].astype(str).map(lambda group: safe_float(layers.get(group, {}).get("shift"), 0.0))
    adjusted = clip_probability(inv_logit_array(logit_array(base) + shifts.to_numpy(dtype=float)))
    return pd.Series(adjusted, index=rows.index)


def apply_symbol_group_calibration(rows: pd.DataFrame, p_col: str, layers: dict[str, dict[str, object]]) -> pd.Series:
    return apply_group_logit_shift(rows, p_col, layers, "symbol_group")


def fit_candidate_tier_calibration(rows: pd.DataFrame, p_col: str, global_base_rate: float) -> dict[str, dict[str, object]]:
    if "candidate_tier" not in rows.columns:
        return {}
    layers: dict[str, dict[str, object]] = {}
    for tier, tier_rows in rows.groupby(rows["candidate_tier"].fillna("MISSING").astype(str), dropna=False):
        layers[str(tier)] = fit_logit_shift_with_shrinkage(
            tier_rows,
            p_col=p_col,
            y_col=TARGET_COL,
            global_base_rate=global_base_rate,
            prior_strength=int(TIER_LAYER_PRIOR_STRENGTH),
            min_events=MIN_TSM_CALIBRATION_EVENTS,
        )
    return layers


def apply_candidate_tier_calibration(rows: pd.DataFrame, p_col: str, layers: dict[str, dict[str, object]]) -> pd.Series:
    return apply_group_logit_shift(rows, p_col, layers, "candidate_tier")


def model_family(model_name: str) -> str:
    lower = str(model_name).lower()
    if "stack" in lower:
        return "stacked_calibrated"
    if "lgbm" in lower:
        return "lightgbm"
    if "xgb" in lower:
        return "xgboost"
    if "elastic" in lower or "logistic" in lower:
        return "linear_logistic"
    if "hist_gradient" in lower:
        return "hist_gradient_boosting"
    if "empirical_bayes" in lower:
        return "empirical_bayes"
    if "tsm_specific" in lower:
        return "tsm_calibration"
    return "other"


def utility_score_frame(
    frame: pd.DataFrame,
    p_col: str,
    stop_col: str,
    expected_r_col: str,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    active_weights = weights or DEFAULT_UTILITY_WEIGHTS
    p = pd.to_numeric(frame[p_col], errors="coerce")
    stop = pd.to_numeric(frame[stop_col], errors="coerce")
    expected_r = pd.to_numeric(frame[expected_r_col], errors="coerce")
    score_total = pd.to_numeric(frame.get("score_price_algo_total", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    score_scaled = (score_total.clip(lower=0.0, upper=100.0) / 100.0).astype(float)
    return (
        active_weights["p_success"] * p
        + active_weights["p_stop_hit"] * stop
        + active_weights["expected_r"] * np.tanh(expected_r / 1.5)
        + active_weights["score_price_algo_total"] * score_scaled
    )


def decision_score_frame(frame: pd.DataFrame, p_col: str, stop_col: str, expected_r_col: str) -> pd.Series:
    return utility_score_frame(frame, p_col, stop_col, expected_r_col, DEFAULT_UTILITY_WEIGHTS)


def prepare_dataset(features: pd.DataFrame) -> pd.DataFrame:
    required = [
        "symbol",
        "symbol_group",
        "date",
        TRADE_READY_COL,
        TARGET_COL,
        RETURN_COL,
        EXPECTED_R_COL,
        STOP_SURVIVAL_COL,
        HIT_1R_COL,
        HIT_2R_COL,
        "label_status_20d",
        *GROUP_COLS,
    ]
    require_columns(features, required, "pooled feature matrix")
    frame = enrich_pooled_features(features)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if MODEL_TRAINING_COL not in frame.columns:
        frame[MODEL_TRAINING_COL] = frame[TRADE_READY_COL]
    if DECISION_ENTRY_COL not in frame.columns:
        frame[DECISION_ENTRY_COL] = frame[TRADE_READY_COL]
    for col in FILTER_ONLY_COLS:
        if col in frame.columns:
            frame[col] = frame[col].map(to_bool)
    for col in [TARGET_COL, RETURN_COL, EXPECTED_R_COL, STOP_SURVIVAL_COL, HIT_1R_COL, HIT_2R_COL]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in GROUP_COLS + ["symbol", "symbol_group"]:
        frame[col] = frame[col].fillna("UNKNOWN").astype(str)
    frame["date_split"] = frame["date"].map(assign_split)
    labeled = frame[(frame[MODEL_TRAINING_COL]) & (frame["label_status_20d"].eq("LABELED"))].copy()
    labeled = labeled.dropna(subset=["date", TARGET_COL, RETURN_COL]).sort_values(["date", "symbol", "signal_idx"]).reset_index(drop=True)
    return labeled


def fit_empirical_bayes(train: pd.DataFrame, prior_strength: float = PRIOR_STRENGTH, group_cols: list[str] | None = None) -> EmpiricalBayesModel:
    if train.empty:
        raise ValueError("cannot fit pooled model with empty training frame")
    active_group_cols = [col for col in (group_cols or GROUP_COLS) if col in train.columns]
    if not active_group_cols:
        active_group_cols = GROUP_COLS
    global_success = float(train[TARGET_COL].mean())
    global_stop = float(train[STOP_SURVIVAL_COL].mean())
    global_hit_1r = float(train[HIT_1R_COL].mean())
    global_hit_2r = float(train[HIT_2R_COL].mean())
    global_expected_r = float(train[EXPECTED_R_COL].mean())
    global_net_return = float(train[RETURN_COL].mean())
    grouped = (
        train.groupby(active_group_cols, dropna=False)
        .agg(
            group_count=(TARGET_COL, "size"),
            success_sum=(TARGET_COL, "sum"),
            stop_survival_sum=(STOP_SURVIVAL_COL, "sum"),
            hit_1r_sum=(HIT_1R_COL, "sum"),
            hit_2r_sum=(HIT_2R_COL, "sum"),
            expected_r_mean=(EXPECTED_R_COL, "mean"),
            net_return_mean=(RETURN_COL, "mean"),
        )
        .reset_index()
    )
    denom = grouped["group_count"] + prior_strength
    grouped["p_success_base"] = (grouped["success_sum"] + prior_strength * global_success) / denom
    grouped["p_stop_survival"] = (grouped["stop_survival_sum"] + prior_strength * global_stop) / denom
    grouped["p_hit_1r"] = (grouped["hit_1r_sum"] + prior_strength * global_hit_1r) / denom
    grouped["p_hit_2r"] = (grouped["hit_2r_sum"] + prior_strength * global_hit_2r) / denom
    grouped["expected_r_net"] = (grouped["expected_r_mean"] * grouped["group_count"] + prior_strength * global_expected_r) / denom
    grouped["expected_net_return_pct"] = (grouped["net_return_mean"] * grouped["group_count"] + prior_strength * global_net_return) / denom
    return EmpiricalBayesModel(
        global_success=global_success,
        global_stop_survival=global_stop,
        global_hit_1r=global_hit_1r,
        global_hit_2r=global_hit_2r,
        global_expected_r=global_expected_r,
        global_net_return=global_net_return,
        group_stats=grouped,
        group_cols=active_group_cols,
    )


def predict_empirical_bayes(model: EmpiricalBayesModel, rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy().reset_index(drop=True)
    group_cols = [col for col in model.group_cols if col in out.columns]
    merged = out[group_cols].merge(model.group_stats, on=group_cols, how="left")
    out["p_success_base"] = pd.to_numeric(merged["p_success_base"], errors="coerce").fillna(model.global_success).clip(0.0, 1.0).to_numpy()
    out["p_stop_survival"] = pd.to_numeric(merged["p_stop_survival"], errors="coerce").fillna(model.global_stop_survival).clip(0.0, 1.0).to_numpy()
    out["p_stop_hit"] = 1.0 - out["p_stop_survival"]
    out["p_hit_1r"] = pd.to_numeric(merged["p_hit_1r"], errors="coerce").fillna(model.global_hit_1r).clip(0.0, 1.0).to_numpy()
    out["p_hit_2r"] = pd.to_numeric(merged["p_hit_2r"], errors="coerce").fillna(model.global_hit_2r).clip(0.0, 1.0).to_numpy()
    out["expected_r_net"] = pd.to_numeric(merged["expected_r_net"], errors="coerce").fillna(model.global_expected_r).to_numpy()
    out["expected_net_return_pct"] = pd.to_numeric(merged["expected_net_return_pct"], errors="coerce").fillna(model.global_net_return).to_numpy()
    out["effective_group_n"] = pd.to_numeric(merged["group_count"], errors="coerce").fillna(0).astype(int).to_numpy()
    return out


def fit_pooled_elastic_net_candidate(train: pd.DataFrame, feature_cols: list[str], target_col: str = TARGET_COL) -> dict[str, object]:
    preprocessor = build_pooled_preprocessor(train, feature_cols)
    model = LogisticRegression(
        solver="saga",
        class_weight="balanced",
        C=0.20,
        l1_ratio=0.25,
        max_iter=2500,
        tol=1e-3,
        random_state=42,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(train[feature_cols], train[target_col].astype(int))
    return {"name": "pooled_elastic_net_logistic", "model": pipeline, "feature_cols": feature_cols, "kind": "classifier"}


def candidate_training_weights(frame: pd.DataFrame) -> pd.Series:
    if "candidate_tier" in frame.columns:
        weights = frame["candidate_tier"].astype(str).map(TIER_SAMPLE_WEIGHTS).fillna(1.0).astype(float)
    else:
        weights = pd.Series(1.0, index=frame.index, dtype=float)
    if DECISION_ENTRY_COL in frame.columns:
        weights = np.where(frame[DECISION_ENTRY_COL].map(to_bool), np.maximum(weights, TIER_SAMPLE_WEIGHTS["decision_trade_ready"]), weights)
    return pd.Series(weights, index=frame.index, dtype=float)


def fit_pooled_weighted_elastic_net_candidate(train: pd.DataFrame, feature_cols: list[str], target_col: str = TARGET_COL) -> dict[str, object]:
    preprocessor = build_pooled_preprocessor(train, feature_cols)
    model = LogisticRegression(
        solver="saga",
        C=0.15,
        l1_ratio=0.35,
        max_iter=3000,
        tol=1e-3,
        random_state=43,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(train[feature_cols], train[target_col].astype(int), model__sample_weight=candidate_training_weights(train).to_numpy(dtype=float))
    return {"name": "pooled_weighted_elastic_net_logistic", "model": pipeline, "feature_cols": feature_cols, "kind": "classifier"}


def fit_strict_elastic_net_candidate(train: pd.DataFrame, feature_cols: list[str], target_col: str = TARGET_COL) -> dict[str, object] | None:
    if DECISION_ENTRY_COL not in train.columns:
        return None
    strict_train = train[train[DECISION_ENTRY_COL].map(to_bool)].copy()
    if len(strict_train) < MIN_POOLED_TRADE_READY_LABELS or strict_train[target_col].nunique() < 2:
        return None
    preprocessor = build_pooled_preprocessor(strict_train, feature_cols)
    model = LogisticRegression(
        solver="saga",
        class_weight="balanced",
        C=0.35,
        l1_ratio=0.25,
        max_iter=3000,
        tol=1e-3,
        random_state=44,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(strict_train[feature_cols], strict_train[target_col].astype(int))
    return {"name": "pooled_strict_elastic_net_logistic", "model": pipeline, "feature_cols": feature_cols, "kind": "classifier"}


def fit_pooled_hist_gbm_candidate(train: pd.DataFrame, feature_cols: list[str], target_col: str = TARGET_COL) -> dict[str, object]:
    preprocessor = build_pooled_preprocessor(train, feature_cols)
    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(train[feature_cols], train[target_col].astype(int))
    return {"name": "pooled_hist_gradient_boosting", "model": pipeline, "feature_cols": feature_cols, "kind": "classifier"}


def fit_pooled_lgbm_success_candidate(train: pd.DataFrame, validation: pd.DataFrame, feature_cols: list[str]) -> dict[str, object] | None:
    if LGBMClassifier is None:
        return None
    preprocessor = build_pooled_preprocessor(train, feature_cols)
    X_train_raw = preprocessor.fit_transform(train[feature_cols])
    X_valid_raw = preprocessor.transform(validation[feature_cols])
    X_train = transformed_feature_frame(preprocessor, X_train_raw, train.index)
    X_valid = transformed_feature_frame(preprocessor, X_valid_raw, validation.index)
    y_train = train[TARGET_COL].astype(int)
    y_valid = validation[TARGET_COL].astype(int)
    pos = max(1, int(y_train.sum()))
    neg = max(1, int(len(y_train) - pos))
    model = LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=4,
        min_data_in_leaf=50,
        min_sum_hessian_in_leaf=5.0,
        feature_fraction=0.75,
        bagging_fraction=0.75,
        bagging_freq=1,
        lambda_l1=0.10,
        lambda_l2=2.00,
        scale_pos_weight=neg / pos,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(stopping_rounds=100, first_metric_only=True, verbose=False), log_evaluation(period=0)],
    )
    return {"name": "pooled_lgbm_classifier", "preprocessor": preprocessor, "model": model, "feature_cols": feature_cols, "kind": "classifier", "predict_as_frame": True}


def fit_pooled_lgbm_aux_heads(train: pd.DataFrame, validation: pd.DataFrame, feature_cols: list[str]) -> dict[str, object]:
    heads: dict[str, object] = {}
    if LGBMClassifier is not None:
        stop_train = 1 - train[STOP_SURVIVAL_COL].astype(int)
        stop_valid = 1 - validation[STOP_SURVIVAL_COL].astype(int)
        preprocessor = build_pooled_preprocessor(train, feature_cols)
        X_train_raw = preprocessor.fit_transform(train[feature_cols])
        X_valid_raw = preprocessor.transform(validation[feature_cols])
        X_train = transformed_feature_frame(preprocessor, X_train_raw, train.index)
        X_valid = transformed_feature_frame(preprocessor, X_valid_raw, validation.index)
        pos = max(1, int(stop_train.sum()))
        neg = max(1, int(len(stop_train) - pos))
        stop_model = LGBMClassifier(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.025,
            num_leaves=15,
            max_depth=4,
            min_data_in_leaf=50,
            lambda_l2=2.0,
            scale_pos_weight=neg / pos,
            random_state=43,
            verbosity=-1,
        )
        stop_model.fit(
            X_train,
            stop_train,
            eval_set=[(X_valid, stop_valid)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(stopping_rounds=80, first_metric_only=True, verbose=False), log_evaluation(period=0)],
        )
        r_model = LGBMRegressor(
            objective="regression",
            n_estimators=1200,
            learning_rate=0.025,
            num_leaves=15,
            max_depth=4,
            min_data_in_leaf=50,
            lambda_l2=2.0,
            random_state=44,
            verbosity=-1,
        )
        r_model.fit(
            X_train,
            pd.to_numeric(train[EXPECTED_R_COL], errors="coerce").fillna(0.0),
            eval_set=[(X_valid, pd.to_numeric(validation[EXPECTED_R_COL], errors="coerce").fillna(0.0))],
            eval_metric="l2",
            callbacks=[early_stopping(stopping_rounds=80, first_metric_only=True, verbose=False), log_evaluation(period=0)],
        )
        heads["stop"] = {"preprocessor": preprocessor, "model": stop_model, "feature_cols": feature_cols, "kind": "classifier", "predict_as_frame": True}
        heads["expected_r"] = {"preprocessor": preprocessor, "model": r_model, "feature_cols": feature_cols, "kind": "regressor", "predict_as_frame": True}
        return heads

    preprocessor = build_pooled_preprocessor(train, feature_cols)
    stop_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    min_samples_leaf=40,
                    l2_regularization=1.0,
                    random_state=43,
                ),
            ),
        ]
    )
    stop_pipeline.fit(train[feature_cols], 1 - train[STOP_SURVIVAL_COL].astype(int))
    r_pipeline = Pipeline(
        [
            ("preprocessor", build_pooled_preprocessor(train, feature_cols)),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    min_samples_leaf=40,
                    l2_regularization=1.0,
                    random_state=44,
                ),
            ),
        ]
    )
    r_pipeline.fit(train[feature_cols], pd.to_numeric(train[EXPECTED_R_COL], errors="coerce").fillna(0.0))
    heads["stop"] = {"model": stop_pipeline, "feature_cols": feature_cols, "kind": "classifier"}
    heads["expected_r"] = {"model": r_pipeline, "feature_cols": feature_cols, "kind": "regressor"}
    return heads


def fit_pooled_xgb_candidate(train: pd.DataFrame, validation: pd.DataFrame, feature_cols: list[str]) -> dict[str, object] | None:
    if XGBClassifier is None:
        return None
    preprocessor = build_pooled_preprocessor(train, feature_cols)
    X_train = preprocessor.fit_transform(train[feature_cols])
    X_valid = preprocessor.transform(validation[feature_cols])
    y_train = train[TARGET_COL].astype(int)
    y_valid = validation[TARGET_COL].astype(int)
    pos = max(1, int(y_train.sum()))
    neg = max(1, int(len(y_train) - pos))
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=3,
        min_child_weight=8,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=2.0,
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return {"name": "pooled_xgb_classifier", "preprocessor": preprocessor, "model": model, "feature_cols": feature_cols, "kind": "classifier"}


def predict_pooled_candidate(candidate: dict[str, object], rows: pd.DataFrame) -> np.ndarray:
    feature_cols = list(candidate["feature_cols"])
    model = candidate["model"]
    if "preprocessor" in candidate:
        X = transform_pooled_features(candidate, rows)
        if candidate.get("kind") == "regressor":
            return np.asarray(model.predict(X), dtype=float)
        return clip_probability(model.predict_proba(X)[:, 1])
    if candidate.get("kind") == "regressor":
        return np.asarray(model.predict(rows[feature_cols]), dtype=float)
    return clip_probability(model.predict_proba(rows[feature_cols])[:, 1])


STACK_FEATURES = [
    "p_success_eb",
    "p_success_hier_eb",
    "p_success_logistic",
    "p_success_weighted_logistic",
    "p_success_strict_logistic",
    "p_success_lgbm",
    "p_success_xgb",
    "p_success_hist_gbm",
    "p_stop_hit_lgbm",
    "expected_r_lgbm",
    "effective_group_n",
    "score_price_algo_total",
    "atr_percentile_252d",
    "drawdown_from_ath",
]


def available_stack_features(frame: pd.DataFrame) -> list[str]:
    return [c for c in STACK_FEATURES if c in frame.columns and pd.to_numeric(frame[c], errors="coerce").notna().any()]


def fit_stack_calibrator(validation_predictions: pd.DataFrame) -> dict[str, object] | None:
    if SKLEARN_IMPORT_ERROR is not None:
        return None
    cols = available_stack_features(validation_predictions)
    if len(cols) < 2 or validation_predictions[TARGET_COL].nunique() < 2:
        return None
    X = validation_predictions[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = validation_predictions[TARGET_COL].astype(int)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=0.25,
                    solver="lbfgs",
                    l1_ratio=0.0,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return {"name": "pooled_stack_calibrated", "model": model, "feature_cols": cols}


def predict_stack_candidate(candidate: dict[str, object], rows: pd.DataFrame) -> np.ndarray:
    cols = list(candidate["feature_cols"])
    X = rows[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return clip_probability(candidate["model"].predict_proba(X)[:, 1])


def fit_tsm_calibration_layer(train_val_tsm: pd.DataFrame, pred_col: str, global_success: float) -> TsmCalibrationLayer:
    frame = train_val_tsm.dropna(subset=[TARGET_COL, pred_col]).copy()
    n = len(frame)
    if n == 0:
        return TsmCalibrationLayer(0, np.nan, np.nan, np.nan, 0.0, 0.0, "NO_TSM_CALIBRATION_EVENTS")
    actual = float(frame[TARGET_COL].mean())
    predicted = float(frame[pred_col].mean())
    posterior = float((frame[TARGET_COL].sum() + TSM_SHRINKAGE_PRIOR_STRENGTH * global_success) / (n + TSM_SHRINKAGE_PRIOR_STRENGTH))
    if n < MIN_TSM_CALIBRATION_EVENTS or frame[TARGET_COL].nunique() < 2:
        return TsmCalibrationLayer(n, actual, predicted, posterior, 0.0, 0.0, "TSM_CALIBRATION_LAYER_INSUFFICIENT_SAMPLE")
    shrinkage = float(n / (n + TSM_SHRINKAGE_PRIOR_STRENGTH))
    shift = float((logit(posterior) - logit(predicted)) * shrinkage)
    status = "TSM_CALIBRATION_LAYER_READY"
    return TsmCalibrationLayer(n, actual, predicted, posterior, shift, shrinkage, status)


def apply_tsm_layer(predictions: pd.DataFrame, layer: TsmCalibrationLayer, source_col: str = "p_success_base") -> pd.Series:
    return pd.to_numeric(predictions[source_col], errors="coerce").map(lambda p: inv_logit(logit(p) + layer.logit_shift) if pd.notna(p) else np.nan)


TSM_CALIBRATION_ROUTES = (
    "POOLED_ONLY",
    "SEMI_GROUP_LOGIT_SHIFT",
    "TSM_STATIC_LOGIT_SHIFT",
    "TSM_SHRUNK_LOGIT_SHIFT",
    "TSM_TIME_DECAY_LOGIT_SHIFT",
    "TSM_PLATT_SIGMOID",
    "TSM_CONVEX_BLEND",
)
TSM_ROUTE_PRIORITY = {
    "POOLED_ONLY": 0,
    "SEMI_GROUP_LOGIT_SHIFT": 1,
    "TSM_SHRUNK_LOGIT_SHIFT": 2,
    "TSM_CONVEX_BLEND": 3,
    "TSM_PLATT_SIGMOID": 4,
    "TSM_TIME_DECAY_LOGIT_SHIFT": 5,
    "TSM_STATIC_LOGIT_SHIFT": 6,
    "TSM_LOGIT_SHIFT": 6,
}
TSM_DIRECT_SYMBOLS = {"TSM"}
TSM_LIKE_FOUNDRY_IDM_SYMBOLS = {"TSM", "UMC", "GFS", "INTC", "STM", "TSEM"}
TSM_SUPPLY_CHAIN_SYMBOLS = {"ASML", "AMAT", "LRCX", "KLAC", "TER", "ENTG", "AMKR", "PLAB"}
SEMI_BREADTH_REGIME_SYMBOLS = {"SMH", "SOXX", "SOXQ", "XSD", "PSI", "FTXL"}
TSM_LIKE_BASE_GROUP_WEIGHTS = {
    "TSM_DIRECT": 1.00,
    "TSM_LIKE_FOUNDRY_IDM": 0.80,
    "TSM_SUPPLY_CHAIN": 0.55,
    "SEMI_BREADTH_REGIME": 0.40,
    "OTHER_SEMI": 0.25,
}
TSM_LIKE_ROUTE_PRIORITY = {
    "TSM_DIRECT_ONLY": 0,
    "TSM_DIRECT_PLUS_LIKE_SHRINKAGE": 1,
    "TSM_LIKE_WEIGHTED_LOGIT_SHIFT": 2,
    "TSM_LIKE_WEIGHTED_PLATT": 3,
    "TSM_LIKE_WEIGHTED_ISOTONIC": 4,
}


def shrunk_tsm_layer(layer: TsmCalibrationLayer, shrink_factor: float = 0.50) -> TsmCalibrationLayer:
    factor = max(0.0, min(1.0, float(shrink_factor)))
    return TsmCalibrationLayer(
        event_count=layer.event_count,
        actual_success_rate=layer.actual_success_rate,
        predicted_success_rate=layer.predicted_success_rate,
        posterior_success_rate=layer.posterior_success_rate,
        logit_shift=layer.logit_shift * factor,
        shrinkage=layer.shrinkage * factor,
        status=f"{layer.status}_SHRUNK_{factor:.2f}",
    )


def apply_tsm_calibration_route(
    predictions: pd.DataFrame,
    route: str,
    tsm_layer: TsmCalibrationLayer,
    shrunk_layer: TsmCalibrationLayer | None = None,
    source_col: str = "p_success_calibrated",
) -> pd.Series:
    route = str(route)
    if route == "POOLED_ONLY":
        return pd.to_numeric(predictions[source_col], errors="coerce")
    if route == "TSM_SHRUNK_LOGIT_SHIFT":
        return apply_tsm_layer(predictions, shrunk_layer or shrunk_tsm_layer(tsm_layer), source_col=source_col)
    if route == "TSM_STATIC_LOGIT_SHIFT":
        return apply_tsm_layer(predictions, tsm_layer, source_col=source_col)
    return apply_tsm_layer(predictions, tsm_layer, source_col=source_col)


def fit_route_logit_layer(rows: pd.DataFrame, pred_col: str, global_success: float, prior_strength: float, status_prefix: str) -> TsmCalibrationLayer:
    frame = rows.dropna(subset=[TARGET_COL, pred_col]).copy()
    n = len(frame)
    if n == 0:
        return TsmCalibrationLayer(0, np.nan, np.nan, np.nan, 0.0, 0.0, f"NO_{status_prefix}_EVENTS")
    actual = float(frame[TARGET_COL].mean())
    predicted = float(frame[pred_col].mean())
    posterior = float((frame[TARGET_COL].sum() + prior_strength * global_success) / (n + prior_strength))
    if n < MIN_TSM_CALIBRATION_EVENTS or frame[TARGET_COL].nunique() < 2:
        return TsmCalibrationLayer(n, actual, predicted, posterior, 0.0, 0.0, f"{status_prefix}_INSUFFICIENT_SAMPLE")
    shrinkage = float(n / (n + prior_strength))
    shift = float((logit(posterior) - logit(predicted)) * shrinkage)
    return TsmCalibrationLayer(n, actual, predicted, posterior, shift, shrinkage, f"{status_prefix}_READY")


def fit_time_decay_route_layer(
    rows: pd.DataFrame,
    pred_col: str,
    global_success: float,
    prior_strength: float = TSM_SHRINKAGE_PRIOR_STRENGTH,
    half_life_days: float = 504.0,
) -> TsmCalibrationLayer:
    frame = rows.dropna(subset=[TARGET_COL, pred_col]).copy()
    n = len(frame)
    if n == 0:
        return TsmCalibrationLayer(0, np.nan, np.nan, np.nan, 0.0, 0.0, "TSM_TIME_DECAY_LOGIT_SHIFT_NO_EVENTS")
    dates = pd.to_datetime(frame.get("date", pd.Series(pd.NaT, index=frame.index)), errors="coerce")
    if dates.notna().any():
        age_days = (dates.max() - dates).dt.days.clip(lower=0).fillna(0.0)
        weights = np.power(0.5, age_days.to_numpy(dtype=float) / float(half_life_days))
    else:
        weights = np.ones(len(frame), dtype=float)
    y = pd.to_numeric(frame[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(frame[pred_col], errors="coerce").to_numpy(dtype=float)
    weight_sum = float(np.sum(weights))
    actual = float(np.average(y, weights=weights)) if weight_sum > 0 else float(np.nanmean(y))
    predicted = float(np.average(p, weights=weights)) if weight_sum > 0 else float(np.nanmean(p))
    effective_n = float((weight_sum ** 2) / np.sum(np.square(weights))) if np.sum(np.square(weights)) > 0 else float(n)
    posterior = float((np.sum(weights * y) + prior_strength * global_success) / (weight_sum + prior_strength))
    if n < MIN_TSM_CALIBRATION_EVENTS or pd.Series(y).nunique() < 2:
        return TsmCalibrationLayer(n, actual, predicted, posterior, 0.0, 0.0, "TSM_TIME_DECAY_LOGIT_SHIFT_INSUFFICIENT_SAMPLE")
    shrinkage = float(effective_n / (effective_n + prior_strength))
    shift = float((logit(posterior) - logit(predicted)) * shrinkage)
    return TsmCalibrationLayer(n, actual, predicted, posterior, shift, shrinkage, "TSM_TIME_DECAY_LOGIT_SHIFT_READY")


def fit_platt_sigmoid_route(rows: pd.DataFrame, pred_col: str) -> tuple[object | None, str]:
    frame = rows.dropna(subset=[TARGET_COL, pred_col]).copy()
    if len(frame) < MIN_TSM_CALIBRATION_EVENTS or frame[TARGET_COL].nunique() < 2 or frame[pred_col].nunique() < 2 or SKLEARN_IMPORT_ERROR is not None:
        return None, "sigmoid_platt_insufficient_sample"
    model = LogisticRegression(C=0.25, solver="lbfgs", l1_ratio=0.0, max_iter=1000, random_state=42)
    model.fit(logit_array(frame[pred_col]).reshape(-1, 1), frame[TARGET_COL].astype(int).to_numpy(dtype=int))
    shrinkage = float(len(frame) / (len(frame) + PLATT_SHRINKAGE_PRIOR_STRENGTH))
    return ShrunkPlattCalibrator(model=model, shrinkage=shrinkage), "sigmoid_platt_shrunk"


def apply_tsm_calibration_route_spec(predictions: pd.DataFrame, spec: TsmCalibrationRouteSpec, source_col: str = "p_success_calibrated") -> pd.Series:
    source = pd.to_numeric(predictions[source_col], errors="coerce")
    route = str(spec.route)
    if route == "POOLED_ONLY":
        return source
    if route in {"SEMI_GROUP_LOGIT_SHIFT", "TSM_STATIC_LOGIT_SHIFT", "TSM_SHRUNK_LOGIT_SHIFT", "TSM_TIME_DECAY_LOGIT_SHIFT"}:
        return apply_tsm_layer(predictions, spec.layer or TsmCalibrationLayer(0, np.nan, np.nan, np.nan, 0.0, 0.0, "NO_LAYER"), source_col=source_col)
    if route == "TSM_PLATT_SIGMOID":
        return pd.Series(apply_probability_calibrator(source, spec.calibrator, spec.calibration_method), index=predictions.index)
    if route == "TSM_CONVEX_BLEND":
        alpha = safe_float(spec.route_alpha, 0.0)
        shifted = apply_tsm_layer(predictions, spec.layer or TsmCalibrationLayer(0, np.nan, np.nan, np.nan, 0.0, 0.0, "NO_LAYER"), source_col=source_col)
        return pd.Series(clip_probability((1.0 - alpha) * source + alpha * shifted), index=predictions.index)
    return source


def choose_convex_blend_alpha(selection_rows: pd.DataFrame, source_col: str, layer: TsmCalibrationLayer, base_rate: float) -> float:
    if selection_rows.empty:
        return 0.0
    candidates: list[dict[str, object]] = []
    shifted = apply_tsm_layer(selection_rows, layer, source_col=source_col)
    source = pd.to_numeric(selection_rows[source_col], errors="coerce")
    for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
        p = pd.Series(clip_probability((1.0 - alpha) * source + alpha * shifted), index=selection_rows.index)
        ece, _, _ = decision_calibration_metrics(selection_rows[TARGET_COL], p)
        brier = brier_score(selection_rows[TARGET_COL], p)
        base_brier = brier_score(selection_rows[TARGET_COL], pd.Series(base_rate, index=selection_rows.index))
        improvement = float((base_brier - brier) / base_brier * 100.0) if pd.notna(base_brier) and base_brier > 0 and pd.notna(brier) else np.nan
        fail_count = int(pd.isna(ece) or ece > MAX_TSM_ECE) + int(pd.isna(improvement) or improvement <= 0.0)
        candidates.append({"alpha": alpha, "fail_count": fail_count, "decision_ece": ece, "brier_improvement_pct": improvement})
    ranked = pd.DataFrame(candidates).sort_values(["fail_count", "decision_ece", "brier_improvement_pct", "alpha"], ascending=[True, True, False, True])
    return float(ranked.iloc[0]["alpha"]) if not ranked.empty else 0.0


def build_tsm_route_specs(
    selection_rows: pd.DataFrame,
    all_rows: pd.DataFrame,
    source_col: str,
    global_success: float,
    tsm_layer: TsmCalibrationLayer,
    shrunk_layer: TsmCalibrationLayer,
) -> list[TsmCalibrationRouteSpec]:
    tsm_groups = set(selection_rows.get("symbol_group", pd.Series(dtype=object)).dropna().astype(str))
    group_rows = all_rows[all_rows.get("symbol_group", pd.Series(dtype=object)).astype(str).isin(tsm_groups)].copy() if tsm_groups and "symbol_group" in all_rows.columns else pd.DataFrame()
    if "symbol" in group_rows.columns:
        non_tsm_group_rows = group_rows[~group_rows["symbol"].astype(str).eq("TSM")].copy()
        if not non_tsm_group_rows.empty:
            group_rows = non_tsm_group_rows
    if group_rows.empty:
        group_rows = all_rows.copy()
    group_layer = fit_route_logit_layer(group_rows, source_col, global_success, SYMBOL_GROUP_LAYER_PRIOR_STRENGTH, "SEMI_GROUP_LOGIT_SHIFT")
    time_decay_layer = fit_time_decay_route_layer(selection_rows, source_col, global_success)
    platt_calibrator, platt_method = fit_platt_sigmoid_route(selection_rows, source_col)
    alpha = choose_convex_blend_alpha(selection_rows, source_col, shrunk_layer, global_success)
    specs = [
        TsmCalibrationRouteSpec("POOLED_ONLY", "p_success_tsm_route_pooled_only", "decision_score_tsm_route_pooled_only", route_prior_source="pooled_trade_ready_probability", route_sample_weight_policy="none"),
        TsmCalibrationRouteSpec("SEMI_GROUP_LOGIT_SHIFT", "p_success_tsm_route_semi_group_logit_shift", "decision_score_tsm_route_semi_group_logit_shift", route_prior_source="semiconductor_symbol_group", route_sample_weight_policy="unweighted", layer=group_layer),
        TsmCalibrationRouteSpec("TSM_STATIC_LOGIT_SHIFT", "p_success_tsm_route_tsm_static_logit_shift", "decision_score_tsm_route_tsm_static_logit_shift", route_prior_source="tsm_train_validation", route_sample_weight_policy="unweighted", layer=tsm_layer),
        TsmCalibrationRouteSpec("TSM_SHRUNK_LOGIT_SHIFT", "p_success_tsm_route_tsm_shrunk_logit_shift", "decision_score_tsm_route_tsm_shrunk_logit_shift", route_prior_source="tsm_train_validation", route_sample_weight_policy="unweighted_shrunk_0_50", layer=shrunk_layer),
        TsmCalibrationRouteSpec("TSM_TIME_DECAY_LOGIT_SHIFT", "p_success_tsm_route_tsm_time_decay_logit_shift", "decision_score_tsm_route_tsm_time_decay_logit_shift", route_prior_source="tsm_train_validation", route_sample_weight_policy="time_decay_half_life_504d", layer=time_decay_layer),
        TsmCalibrationRouteSpec("TSM_CONVEX_BLEND", "p_success_tsm_route_tsm_convex_blend", "decision_score_tsm_route_tsm_convex_blend", route_alpha=alpha, route_prior_source="tsm_train_validation", route_sample_weight_policy="validation_alpha_grid", layer=shrunk_layer),
    ]
    if platt_calibrator is not None:
        specs.append(
            TsmCalibrationRouteSpec(
                "TSM_PLATT_SIGMOID",
                "p_success_tsm_route_tsm_platt_sigmoid",
                "decision_score_tsm_route_tsm_platt_sigmoid",
                route_prior_source="tsm_train_validation",
                route_sample_weight_policy="sigmoid_platt_no_isotonic",
                calibrator=platt_calibrator,
                calibration_method=platt_method,
            )
        )
    return specs


def choose_tsm_calibration_route(route_metrics: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    if route_metrics.empty or "tsm_calibration_route" not in route_metrics.columns:
        return "POOLED_ONLY", pd.DataFrame()
    summaries: list[dict[str, object]] = []
    for route, rows in route_metrics.groupby("tsm_calibration_route", dropna=False):
        route_name = str(route)
        combined = rows[rows["split"].astype(str).eq("tsm_combined_test_holdout")]
        test_holdout = rows[rows["split"].astype(str).isin(["tsm_test_2024", "tsm_final_holdout_2025_2026"])]
        combined_row = combined.iloc[0] if not combined.empty else pd.Series(dtype=object)
        combined_ece = safe_float(combined_row.get("decision_ece", combined_row.get("ece")))
        combined_brier_improvement = safe_float(combined_row.get("brier_improvement_pct"))
        split_ece = pd.to_numeric(test_holdout.get("decision_ece", test_holdout.get("ece", pd.Series(dtype=float))), errors="coerce")
        max_split_ece = float(split_ece.max()) if split_ece.notna().any() else np.nan
        failures: list[str] = []
        if pd.isna(combined_ece) or combined_ece > MAX_TSM_ECE:
            failures.append("TSM_COMBINED_ECE_GT_0_15")
        if pd.isna(combined_brier_improvement) or combined_brier_improvement <= 0:
            failures.append("TSM_COMBINED_BRIER_IMPROVEMENT_LE_0")
        if pd.isna(max_split_ece) or max_split_ece > 0.20:
            failures.append("TSM_TEST_OR_HOLDOUT_ECE_GT_0_20")
        summaries.append(
            {
                "tsm_calibration_route": route_name,
                "tsm_calibration_route_pass": not failures,
                "tsm_calibration_route_failure_reasons": "|".join(failures) if failures else "PASS",
                "combined_decision_ece": combined_ece,
                "combined_brier_improvement_pct": combined_brier_improvement,
                "max_test_holdout_ece": max_split_ece,
                "route_fail_count": len(failures),
                "route_priority": TSM_ROUTE_PRIORITY.get(route_name, 9),
            }
        )
    summary = pd.DataFrame(summaries)
    if summary.empty:
        return "POOLED_ONLY", summary
    passed = summary[summary["tsm_calibration_route_pass"].map(to_bool)].copy()
    if not passed.empty:
        best = passed.sort_values(["combined_decision_ece", "route_priority"], ascending=[True, True]).iloc[0]
        pooled = passed[passed["tsm_calibration_route"].astype(str).eq("POOLED_ONLY")]
        if not pooled.empty and safe_float(pooled.iloc[0].get("combined_decision_ece")) <= safe_float(best.get("combined_decision_ece")) + 0.01:
            chosen_route = "POOLED_ONLY"
        else:
            chosen_route = str(best["tsm_calibration_route"])
    else:
        ranked = summary.sort_values(
            ["route_fail_count", "combined_decision_ece", "max_test_holdout_ece", "combined_brier_improvement_pct", "route_priority"],
            ascending=[True, True, True, False, True],
        )
        best = ranked.iloc[0]
        pooled = summary[summary["tsm_calibration_route"].astype(str).eq("POOLED_ONLY")]
        if (
            not pooled.empty
            and int(safe_float(pooled.iloc[0].get("route_fail_count"), 99)) == int(safe_float(best.get("route_fail_count"), 99))
            and safe_float(pooled.iloc[0].get("combined_decision_ece")) <= safe_float(best.get("combined_decision_ece")) + 0.01
        ):
            chosen_route = "POOLED_ONLY"
        else:
            chosen_route = str(best["tsm_calibration_route"])
    summary["is_selected_tsm_calibration_route"] = summary["tsm_calibration_route"].astype(str).eq(chosen_route)
    return chosen_route, summary


def choose_tsm_calibration_route_v2(route_metrics: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    if route_metrics.empty or "tsm_calibration_route" not in route_metrics.columns:
        return "POOLED_ONLY", pd.DataFrame()
    summaries: list[dict[str, object]] = []
    for route, rows in route_metrics.groupby("tsm_calibration_route", dropna=False):
        route_name = str(route)
        selection = rows[rows["split"].astype(str).eq("tsm_train_validation")]
        combined = rows[rows["split"].astype(str).eq("tsm_combined_test_holdout")]
        test_holdout = rows[rows["split"].astype(str).isin(["tsm_test_2024", "tsm_final_holdout_2025_2026"])]
        selection_row = selection.iloc[0] if not selection.empty else pd.Series(dtype=object)
        combined_row = combined.iloc[0] if not combined.empty else pd.Series(dtype=object)
        selection_ece = safe_float(selection_row.get("decision_ece", selection_row.get("ece")))
        selection_brier_improvement = safe_float(selection_row.get("brier_improvement_pct"))
        combined_ece = safe_float(combined_row.get("decision_ece", combined_row.get("ece")))
        combined_brier_improvement = safe_float(combined_row.get("brier_improvement_pct"))
        split_ece = pd.to_numeric(test_holdout.get("decision_ece", test_holdout.get("ece", pd.Series(dtype=float))), errors="coerce")
        max_split_ece = float(split_ece.max()) if split_ece.notna().any() else np.nan
        selection_failures: list[str] = []
        if pd.isna(selection_ece) or selection_ece > MAX_TSM_ECE:
            selection_failures.append("ROUTE_SELECTION_ECE_GT_0_15")
        if pd.isna(selection_brier_improvement) or selection_brier_improvement <= 0:
            selection_failures.append("ROUTE_SELECTION_BRIER_IMPROVEMENT_LE_0")
        final_failures: list[str] = []
        if pd.isna(combined_ece) or combined_ece > MAX_TSM_ECE:
            final_failures.append("TSM_COMBINED_ECE_GT_0_15")
        if pd.isna(combined_brier_improvement) or combined_brier_improvement <= 0:
            final_failures.append("TSM_COMBINED_BRIER_IMPROVEMENT_LE_0")
        if pd.isna(max_split_ece) or max_split_ece > 0.20:
            final_failures.append("TSM_TEST_OR_HOLDOUT_ECE_GT_0_20")
        route_meta = rows.iloc[0]
        summaries.append(
            {
                "tsm_calibration_route": route_name,
                "tsm_calibration_route_v2": route_name,
                "tsm_calibration_route_pass": not final_failures,
                "tsm_calibration_route_failure_reasons": "|".join(final_failures) if final_failures else "PASS",
                "route_selection_pass": not selection_failures,
                "route_selection_failure_reasons": "|".join(selection_failures) if selection_failures else "PASS",
                "selection_decision_ece": selection_ece,
                "selection_brier_improvement_pct": selection_brier_improvement,
                "combined_decision_ece": combined_ece,
                "combined_brier_improvement_pct": combined_brier_improvement,
                "max_test_holdout_ece": max_split_ece,
                "route_fail_count": len(final_failures),
                "route_selection_fail_count": len(selection_failures),
                "route_priority": TSM_ROUTE_PRIORITY.get(route_name, 9),
                "route_selection_window": "train_2016_2022_plus_validation_2023",
                "route_evaluation_window": "test_2024_plus_final_holdout_2025_2026",
                "route_selection_provenance_valid": True,
                "route_alpha": route_meta.get("route_alpha", np.nan),
                "route_prior_source": route_meta.get("route_prior_source", ""),
                "route_sample_weight_policy": route_meta.get("route_sample_weight_policy", ""),
            }
        )
    summary = pd.DataFrame(summaries)
    if summary.empty:
        return "POOLED_ONLY", summary
    selection_passed = summary[summary["route_selection_pass"].map(to_bool)].copy()
    if not selection_passed.empty:
        best = selection_passed.sort_values(
            ["selection_decision_ece", "selection_brier_improvement_pct", "route_priority"],
            ascending=[True, False, True],
        ).iloc[0]
        pooled = selection_passed[selection_passed["tsm_calibration_route"].astype(str).eq("POOLED_ONLY")]
        if not pooled.empty and safe_float(pooled.iloc[0].get("selection_decision_ece")) <= safe_float(best.get("selection_decision_ece")) + 0.01:
            chosen_route = "POOLED_ONLY"
        else:
            chosen_route = str(best["tsm_calibration_route"])
    else:
        ranked = summary.sort_values(
            ["route_selection_fail_count", "selection_decision_ece", "selection_brier_improvement_pct", "route_priority"],
            ascending=[True, True, False, True],
        )
        best = ranked.iloc[0]
        pooled = summary[summary["tsm_calibration_route"].astype(str).eq("POOLED_ONLY")]
        if (
            not pooled.empty
            and int(safe_float(pooled.iloc[0].get("route_selection_fail_count"), 99)) == int(safe_float(best.get("route_selection_fail_count"), 99))
            and safe_float(pooled.iloc[0].get("selection_decision_ece")) <= safe_float(best.get("selection_decision_ece")) + 0.01
        ):
            chosen_route = "POOLED_ONLY"
        else:
            chosen_route = str(best["tsm_calibration_route"])
    summary["is_selected_tsm_calibration_route"] = summary["tsm_calibration_route"].astype(str).eq(chosen_route)
    summary["selected_route_is_simplest_close_candidate"] = summary["is_selected_tsm_calibration_route"].map(to_bool) & summary["tsm_calibration_route"].astype(str).eq("POOLED_ONLY")
    return chosen_route, summary


def tsm_like_group_for_symbol(symbol: str, symbol_group: str = "") -> str:
    value = str(symbol).upper()
    group = str(symbol_group).lower()
    if value in TSM_DIRECT_SYMBOLS:
        return "TSM_DIRECT"
    if value in TSM_LIKE_FOUNDRY_IDM_SYMBOLS or group in {"foundry", "foundry_idm"}:
        return "TSM_LIKE_FOUNDRY_IDM"
    if value in TSM_SUPPLY_CHAIN_SYMBOLS or group in {"semicap", "semicap_osat"}:
        return "TSM_SUPPLY_CHAIN"
    if value in SEMI_BREADTH_REGIME_SYMBOLS or group in {"semiconductor_etf", "semi_breadth_regime"}:
        return "SEMI_BREADTH_REGIME"
    return "OTHER_SEMI"


def effective_sample_size(weights) -> float:
    values = pd.to_numeric(pd.Series(weights), errors="coerce").dropna().to_numpy(dtype=float)
    values = values[values > 0]
    denom = float(np.sum(np.square(values)))
    if len(values) == 0 or denom <= 0:
        return 0.0
    return float(np.sum(values) ** 2 / denom)


def fit_tsm_like_weight_table(reference_rows: pd.DataFrame) -> pd.DataFrame:
    if reference_rows.empty or "symbol" not in reference_rows.columns:
        return pd.DataFrame(columns=["symbol", "tsm_like_group", "base_group_weight", "dynamic_weight", "tsm_like_weight"])
    ref = reference_rows.copy()
    ref["symbol"] = ref["symbol"].astype(str).str.upper()
    if "symbol_group" not in ref.columns:
        ref["symbol_group"] = "semiconductor"
    tsm = ref[ref["symbol"].eq("TSM")].copy()
    tsm_return = None
    if not tsm.empty and "return_20d" in ref.columns:
        tsm_return = (
            tsm[["date", "return_20d"]]
            .dropna()
            .groupby("date", as_index=False)["return_20d"]
            .mean()
            .rename(columns={"return_20d": "tsm_return_20d"})
        )
    tsm_beta = safe_float(pd.to_numeric(tsm.get("beta_vs_smh_126d", pd.Series(dtype=float)), errors="coerce").median(), np.nan)
    tsm_vol = safe_float(pd.to_numeric(tsm.get("vol_20d_ann", pd.Series(dtype=float)), errors="coerce").median(), np.nan)
    rows: list[dict[str, object]] = []
    for symbol, group in ref.groupby("symbol", dropna=False):
        symbol_group = str(group.get("symbol_group", pd.Series(["semiconductor"])).dropna().astype(str).iloc[0]) if "symbol_group" in group.columns and group["symbol_group"].notna().any() else "semiconductor"
        tsm_like_group = tsm_like_group_for_symbol(str(symbol), symbol_group)
        base_weight = TSM_LIKE_BASE_GROUP_WEIGHTS[tsm_like_group]
        if str(symbol).upper() == "TSM":
            corr_score = beta_similarity = vol_similarity = 1.0
        else:
            corr_score = 0.50
            if tsm_return is not None and "return_20d" in group.columns:
                merged = group[["date", "return_20d"]].dropna().merge(tsm_return, on="date", how="inner")
                if len(merged) >= 30:
                    corr = pd.to_numeric(merged["return_20d"], errors="coerce").corr(pd.to_numeric(merged["tsm_return_20d"], errors="coerce"))
                    corr_score = float(np.clip((safe_float(corr, 0.0) + 1.0) / 2.0, 0.0, 1.0))
            beta = safe_float(pd.to_numeric(group.get("beta_vs_smh_126d", pd.Series(dtype=float)), errors="coerce").median(), np.nan)
            beta_similarity = 0.50 if pd.isna(beta) or pd.isna(tsm_beta) else float(np.clip(1.0 - abs(beta - tsm_beta) / 2.0, 0.0, 1.0))
            vol = safe_float(pd.to_numeric(group.get("vol_20d_ann", pd.Series(dtype=float)), errors="coerce").median(), np.nan)
            vol_den = max(abs(tsm_vol), 0.01) if pd.notna(tsm_vol) else np.nan
            vol_similarity = 0.50 if pd.isna(vol) or pd.isna(vol_den) else float(np.clip(1.0 - abs(vol - tsm_vol) / vol_den, 0.0, 1.0))
        dynamic_weight = float(np.clip(0.5 * corr_score + 0.3 * beta_similarity + 0.2 * vol_similarity, 0.10, 1.00))
        final_weight = 1.0 if str(symbol).upper() == "TSM" else float(np.clip(base_weight * dynamic_weight, 0.10, 1.00))
        rows.append(
            {
                "symbol": str(symbol).upper(),
                "symbol_group": symbol_group,
                "tsm_like_group": tsm_like_group,
                "base_group_weight": base_weight,
                "corr_similarity_to_tsm": corr_score,
                "beta_similarity_to_tsm": beta_similarity,
                "vol_similarity_to_tsm": vol_similarity,
                "dynamic_weight": dynamic_weight,
                "tsm_like_weight": final_weight,
            }
        )
    return pd.DataFrame(rows)


def attach_tsm_like_weights(rows: pd.DataFrame, weight_table: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if out.empty:
        return out
    out["symbol"] = out.get("symbol", pd.Series("", index=out.index)).astype(str).str.upper()
    if "symbol_group" not in out.columns:
        out["symbol_group"] = "semiconductor"
    if weight_table.empty:
        out["tsm_like_group"] = [tsm_like_group_for_symbol(s, g) for s, g in zip(out["symbol"], out["symbol_group"])]
        out["tsm_like_weight"] = [TSM_LIKE_BASE_GROUP_WEIGHTS[g] for g in out["tsm_like_group"]]
        return out
    merge_cols = ["symbol", "tsm_like_group", "base_group_weight", "dynamic_weight", "tsm_like_weight"]
    out = out.merge(weight_table[[c for c in merge_cols if c in weight_table.columns]], on="symbol", how="left")
    missing = out["tsm_like_group"].isna()
    if missing.any():
        fallback_groups = [tsm_like_group_for_symbol(s, g) for s, g in zip(out.loc[missing, "symbol"], out.loc[missing, "symbol_group"])]
        out.loc[missing, "tsm_like_group"] = fallback_groups
        out.loc[missing, "tsm_like_weight"] = [TSM_LIKE_BASE_GROUP_WEIGHTS[g] for g in fallback_groups]
    out["tsm_like_weight"] = pd.to_numeric(out["tsm_like_weight"], errors="coerce").fillna(0.10).clip(0.10, 1.00)
    return out


def weighted_brier_score(y_true: pd.Series, p: pd.Series, weights: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    probs = pd.to_numeric(p, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = y.notna() & probs.notna() & w.notna() & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.average(np.square(y[mask].to_numpy(dtype=float) - probs[mask].to_numpy(dtype=float)), weights=w[mask].to_numpy(dtype=float)))


def weighted_ece_score(y_true: pd.Series, p: pd.Series, weights: pd.Series, bins: int = 10) -> tuple[float, int]:
    frame = pd.DataFrame(
        {
            "y": pd.to_numeric(y_true, errors="coerce"),
            "p": pd.to_numeric(p, errors="coerce"),
            "w": pd.to_numeric(weights, errors="coerce"),
        }
    ).dropna()
    frame = frame[frame["w"] > 0].copy()
    if frame.empty:
        return np.nan, 0
    frame["bin"] = pd.cut(frame["p"].clip(0.0, 1.0), bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True, labels=False)
    total_weight = float(frame["w"].sum())
    ece = 0.0
    min_bin_n = np.inf
    for _, group in frame.groupby("bin", dropna=True):
        if group.empty:
            continue
        group_weight = float(group["w"].sum())
        actual = float(np.average(group["y"], weights=group["w"]))
        predicted = float(np.average(group["p"], weights=group["w"]))
        ece += (group_weight / total_weight) * abs(actual - predicted)
        min_bin_n = min(min_bin_n, len(group))
    return float(ece), int(min_bin_n if np.isfinite(min_bin_n) else 0)


def fit_tsm_like_logit_shift(rows: pd.DataFrame, p_col: str, base_rate: float, weight_col: str = "tsm_like_weight") -> TsmLikeCalibrationRouteSpec:
    frame = rows.dropna(subset=[TARGET_COL, p_col, weight_col]).copy()
    frame = frame[pd.to_numeric(frame[weight_col], errors="coerce") > 0].copy()
    eff_n = effective_sample_size(frame[weight_col]) if not frame.empty else 0.0
    if frame.empty:
        return TsmLikeCalibrationRouteSpec("TSM_LIKE_WEIGHTED_LOGIT_SHIFT", status="NO_TSM_LIKE_EVENTS", effective_n=0.0)
    y = pd.to_numeric(frame[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(frame[p_col], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(frame[weight_col], errors="coerce").to_numpy(dtype=float)
    actual = float(np.average(y, weights=w))
    predicted = float(np.average(p, weights=w))
    if eff_n < MIN_TSM_CALIBRATION_EVENTS or pd.Series(y).nunique() < 2:
        return TsmLikeCalibrationRouteSpec("TSM_LIKE_WEIGHTED_LOGIT_SHIFT", status="TSM_LIKE_WEIGHTED_LOGIT_SHIFT_INSUFFICIENT_SAMPLE", effective_n=eff_n)
    posterior = float((np.sum(w * y) + TSM_SHRINKAGE_PRIOR_STRENGTH * base_rate) / (np.sum(w) + TSM_SHRINKAGE_PRIOR_STRENGTH))
    shrinkage = float(eff_n / (eff_n + TSM_SHRINKAGE_PRIOR_STRENGTH))
    shift = float((logit(posterior) - logit(predicted)) * shrinkage)
    return TsmLikeCalibrationRouteSpec("TSM_LIKE_WEIGHTED_LOGIT_SHIFT", logit_shift=shift, shrinkage=shrinkage, status="TSM_LIKE_WEIGHTED_LOGIT_SHIFT_READY", effective_n=eff_n)


def fit_tsm_like_platt(rows: pd.DataFrame, p_col: str, route: str = "TSM_LIKE_WEIGHTED_PLATT", weight_col: str = "tsm_like_weight") -> TsmLikeCalibrationRouteSpec:
    frame = rows.dropna(subset=[TARGET_COL, p_col, weight_col]).copy()
    frame = frame[pd.to_numeric(frame[weight_col], errors="coerce") > 0].copy()
    eff_n = effective_sample_size(frame[weight_col]) if not frame.empty else 0.0
    if len(frame) < MIN_TSM_CALIBRATION_EVENTS or frame[TARGET_COL].nunique() < 2 or frame[p_col].nunique() < 2 or SKLEARN_IMPORT_ERROR is not None:
        return TsmLikeCalibrationRouteSpec(route, status=f"{route}_INSUFFICIENT_SAMPLE", effective_n=eff_n)
    model = LogisticRegression(C=0.25, solver="lbfgs", l1_ratio=0.0, max_iter=1000, random_state=42)
    model.fit(
        logit_array(frame[p_col]).reshape(-1, 1),
        frame[TARGET_COL].astype(int).to_numpy(dtype=int),
        sample_weight=pd.to_numeric(frame[weight_col], errors="coerce").to_numpy(dtype=float),
    )
    shrinkage = float(eff_n / (eff_n + PLATT_SHRINKAGE_PRIOR_STRENGTH))
    return TsmLikeCalibrationRouteSpec(route, calibrator=ShrunkPlattCalibrator(model=model, shrinkage=shrinkage), calibration_method="sigmoid_platt_shrunk", shrinkage=shrinkage, status=f"{route}_READY", effective_n=eff_n)


def fit_tsm_like_isotonic(rows: pd.DataFrame, p_col: str, weight_col: str = "tsm_like_weight") -> TsmLikeCalibrationRouteSpec:
    route = "TSM_LIKE_WEIGHTED_ISOTONIC"
    frame = rows.dropna(subset=[TARGET_COL, p_col, weight_col]).copy()
    frame = frame[pd.to_numeric(frame[weight_col], errors="coerce") > 0].copy()
    eff_n = effective_sample_size(frame[weight_col]) if not frame.empty else 0.0
    if eff_n < MIN_TSM_LIKE_ISOTONIC_EFFECTIVE_N or frame[TARGET_COL].nunique() < 2 or frame[p_col].nunique() < 2 or SKLEARN_IMPORT_ERROR is not None:
        return TsmLikeCalibrationRouteSpec(route, status=f"{route}_INSUFFICIENT_EFFECTIVE_N", effective_n=eff_n)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(
        pd.to_numeric(frame[p_col], errors="coerce").to_numpy(dtype=float),
        frame[TARGET_COL].astype(int).to_numpy(dtype=int),
        sample_weight=pd.to_numeric(frame[weight_col], errors="coerce").to_numpy(dtype=float),
    )
    return TsmLikeCalibrationRouteSpec(route, calibrator=calibrator, calibration_method="isotonic", status=f"{route}_READY", effective_n=eff_n)


def apply_tsm_like_calibration(rows: pd.DataFrame, p_col: str, spec: TsmLikeCalibrationRouteSpec) -> pd.Series:
    source = pd.to_numeric(rows[p_col], errors="coerce") if p_col in rows.columns else pd.Series(np.nan, index=rows.index)
    if spec.route in {"TSM_DIRECT_ONLY", "TSM_LIKE_WEIGHTED_LOGIT_SHIFT", "TSM_DIRECT_PLUS_LIKE_SHRINKAGE"}:
        return pd.Series(clip_probability(inv_logit_array(logit_array(source) + safe_float(spec.logit_shift, 0.0))), index=rows.index)
    if spec.route in {"TSM_LIKE_WEIGHTED_PLATT", "TSM_LIKE_WEIGHTED_ISOTONIC"}:
        return pd.Series(apply_probability_calibrator(source, spec.calibrator, spec.calibration_method), index=rows.index)
    return source


def tsm_like_metric_row(split_name: str, rows: pd.DataFrame, p_col: str, route: str, base_rate: float) -> dict[str, object]:
    frame = rows.dropna(subset=[TARGET_COL, p_col, "tsm_like_weight"]).copy() if p_col in rows.columns else pd.DataFrame()
    if frame.empty:
        return {
            "tsm_like_calibration_route": route,
            "split": split_name,
            "event_count": 0,
            "weighted_event_count": 0.0,
            "effective_n": 0.0,
            "decision_ece": np.nan,
            "brier_improvement_pct": np.nan,
            "status": "NO_EVENTS",
        }
    weights = pd.to_numeric(frame["tsm_like_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    weighted_events = float(weights.sum())
    eff_n = effective_sample_size(weights)
    ece, min_bin_n = weighted_ece_score(frame[TARGET_COL], frame[p_col], weights)
    brier = weighted_brier_score(frame[TARGET_COL], frame[p_col], weights)
    base_brier = weighted_brier_score(frame[TARGET_COL], pd.Series(base_rate, index=frame.index), weights)
    return {
        "tsm_like_calibration_route": route,
        "split": split_name,
        "event_count": int(len(frame)),
        "weighted_event_count": weighted_events,
        "effective_n": eff_n,
        "decision_ece": ece,
        "ece": ece,
        "decision_min_calibration_bin_n": min_bin_n,
        "brier_score": brier,
        "base_rate_brier_score": base_brier,
        "brier_improvement_pct": float((base_brier - brier) / base_brier * 100.0) if pd.notna(base_brier) and base_brier > 0 and pd.notna(brier) else np.nan,
        "weighted_success_rate": float(np.average(pd.to_numeric(frame[TARGET_COL], errors="coerce"), weights=weights)) if weighted_events > 0 else np.nan,
        "weighted_predicted_success_rate": float(np.average(pd.to_numeric(frame[p_col], errors="coerce"), weights=weights)) if weighted_events > 0 else np.nan,
        "status": "OK",
    }


def choose_tsm_like_route(metrics: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    if metrics.empty:
        return "TSM_DIRECT_ONLY", pd.DataFrame()
    rows = []
    for route, group in metrics.groupby("tsm_like_calibration_route", dropna=False):
        route_name = str(route)
        selection = group[group["split"].astype(str).eq("tsm_like_train_validation")]
        combined = group[group["split"].astype(str).eq("tsm_like_combined_test_holdout")]
        selection_row = selection.iloc[0] if not selection.empty else pd.Series(dtype=object)
        combined_row = combined.iloc[0] if not combined.empty else pd.Series(dtype=object)
        selection_ece = safe_float(selection_row.get("decision_ece"))
        selection_brier = safe_float(selection_row.get("brier_improvement_pct"))
        selection_eff_n = safe_float(selection_row.get("effective_n"), 0.0)
        failures = []
        if selection_eff_n < MIN_TSM_LIKE_EFFECTIVE_SELECTION_N:
            failures.append("TSM_LIKE_EFFECTIVE_TRAIN_VALIDATION_N_LT_500")
        if pd.isna(selection_ece) or selection_ece > MAX_TSM_ECE:
            failures.append("TSM_LIKE_SELECTION_ECE_GT_0_15")
        if pd.isna(selection_brier) or selection_brier <= 0:
            failures.append("TSM_LIKE_SELECTION_BRIER_IMPROVEMENT_LE_0")
        rows.append(
            {
                "tsm_like_calibration_route": route_name,
                "tsm_like_route_selection_pass": not failures,
                "tsm_like_route_selection_failure_reasons": "|".join(failures) if failures else "PASS",
                "selection_decision_ece": selection_ece,
                "selection_brier_improvement_pct": selection_brier,
                "selection_effective_n": selection_eff_n,
                "combined_decision_ece": safe_float(combined_row.get("decision_ece")),
                "combined_brier_improvement_pct": safe_float(combined_row.get("brier_improvement_pct")),
                "route_priority": TSM_LIKE_ROUTE_PRIORITY.get(route_name, 9),
                "route_selection_window": "train_2016_2022_plus_validation_2023",
                "route_evaluation_window": "test_2024_plus_final_holdout_2025_2026",
                "route_selection_provenance_valid": True,
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return "TSM_DIRECT_ONLY", summary
    passed = summary[summary["tsm_like_route_selection_pass"].map(to_bool)].copy()
    rank_source = passed if not passed.empty else summary
    selected = rank_source.sort_values(
        ["tsm_like_route_selection_pass", "selection_decision_ece", "selection_brier_improvement_pct", "route_priority"],
        ascending=[False, True, False, True],
    ).iloc[0]
    route = str(selected["tsm_like_calibration_route"])
    summary["is_selected_tsm_like_route"] = summary["tsm_like_calibration_route"].astype(str).eq(route)
    return route, summary


def build_tsm_like_calibration_artifacts(
    predictions: pd.DataFrame,
    source_col: str,
    base_rate: float,
    trade_weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, TsmLikeCalibrationRouteSpec]:
    if predictions.empty or source_col not in predictions.columns:
        empty_spec = TsmLikeCalibrationRouteSpec("TSM_DIRECT_ONLY", status="MISSING_PREDICTIONS")
        return pd.DataFrame(), pd.DataFrame(), empty_spec
    decision_mask = predictions[DECISION_ENTRY_COL].map(to_bool) if DECISION_ENTRY_COL in predictions.columns else pd.Series(True, index=predictions.index)
    reference = predictions[decision_mask & predictions["split"].isin(["train_2016_2022", "validation_2023"])].copy()
    weight_table = fit_tsm_like_weight_table(reference)
    pool = attach_tsm_like_weights(predictions[decision_mask].copy(), weight_table)
    selection = pool[pool["split"].isin(["train_2016_2022", "validation_2023"])].copy()
    direct = selection[selection["symbol"].astype(str).str.upper().eq("TSM")].copy()
    direct_layer = fit_tsm_calibration_layer(direct, source_col, base_rate)
    direct_spec = TsmLikeCalibrationRouteSpec("TSM_DIRECT_ONLY", logit_shift=direct_layer.logit_shift, shrinkage=direct_layer.shrinkage, status=direct_layer.status, effective_n=float(direct_layer.event_count))
    like_logit = fit_tsm_like_logit_shift(selection, source_col, base_rate)
    like_platt = fit_tsm_like_platt(selection, source_col)
    like_isotonic = fit_tsm_like_isotonic(selection, source_col)
    direct_eff = float(direct_layer.event_count)
    like_eff = safe_float(like_logit.effective_n, 0.0)
    direct_alpha = direct_eff / (direct_eff + like_eff) if (direct_eff + like_eff) > 0 else 0.0
    shrinkage_spec = TsmLikeCalibrationRouteSpec(
        "TSM_DIRECT_PLUS_LIKE_SHRINKAGE",
        logit_shift=direct_alpha * direct_spec.logit_shift + (1.0 - direct_alpha) * like_logit.logit_shift,
        shrinkage=direct_alpha * direct_spec.shrinkage + (1.0 - direct_alpha) * like_logit.shrinkage,
        status="TSM_DIRECT_PLUS_LIKE_SHRINKAGE_READY" if direct_eff >= MIN_TSM_CALIBRATION_EVENTS and like_eff >= MIN_TSM_LIKE_EFFECTIVE_SELECTION_N else "TSM_DIRECT_PLUS_LIKE_SHRINKAGE_ADVISORY",
        effective_n=direct_eff + like_eff,
    )
    route_specs = [direct_spec, like_platt, like_logit, like_isotonic, shrinkage_spec]
    metric_rows: list[dict[str, object]] = []
    for spec in route_specs:
        route_p_col = f"p_success_{spec.route.lower()}"
        pool[route_p_col] = apply_tsm_like_calibration(pool, source_col, spec)
        pool[f"decision_score_{spec.route.lower()}"] = utility_score_frame(pool, route_p_col, "p_stop_hit_lgbm", "expected_r_lgbm", trade_weights)
        for split_name, frame in [
            ("tsm_like_train_validation", pool[pool["split"].isin(["train_2016_2022", "validation_2023"])]),
            ("tsm_like_test_2024", pool[pool["split"].eq("test_2024")]),
            ("tsm_like_final_holdout_2025_2026", pool[pool["split"].eq("final_holdout_2025_2026")]),
            ("tsm_like_combined_test_holdout", pool[pool["split"].isin(["test_2024", "final_holdout_2025_2026"])]),
        ]:
            row = tsm_like_metric_row(split_name, frame, route_p_col, spec.route, base_rate)
            row["route_status"] = spec.status
            row["route_shrinkage"] = spec.shrinkage
            row["route_logit_shift"] = spec.logit_shift
            row["route_selection_provenance_valid"] = True
            metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    selected_route, route_summary = choose_tsm_like_route(metrics)
    if not route_summary.empty:
        metrics = metrics.merge(route_summary, on="tsm_like_calibration_route", how="left", suffixes=("", "_selection"))
    selected_spec = next((spec for spec in route_specs if spec.route == selected_route), direct_spec)
    pool["p_success_tsm_like_calibrated"] = apply_tsm_like_calibration(pool, source_col, selected_spec)
    pool["decision_score_tsm_like_calibrated"] = utility_score_frame(pool, "p_success_tsm_like_calibrated", "p_stop_hit_lgbm", "expected_r_lgbm", trade_weights)
    pool["selected_tsm_like_calibration_route"] = selected_route
    keep_cols = [
        "symbol",
        "symbol_group",
        "date",
        "split",
        "signal_idx",
        DECISION_ENTRY_COL,
        TRADE_READY_COL,
        TARGET_COL,
        RETURN_COL,
        source_col,
        "p_success_tsm_like_calibrated",
        "decision_score_tsm_like_calibrated",
        "p_stop_hit_lgbm",
        "expected_r_lgbm",
        "tsm_like_group",
        "base_group_weight",
        "dynamic_weight",
        "tsm_like_weight",
        "selected_tsm_like_calibration_route",
    ]
    pool = pool[[c for c in keep_cols if c in pool.columns]].copy()
    return pool, metrics, selected_spec


def scope_prior_strength(scope_name: str, event_count: int, default: float = STRICT_SCOPE_CALIBRATION_PRIOR_STRENGTH) -> float:
    if scope_name != TRADE_READY_EVAL_SCOPE:
        return default
    if event_count < 200:
        return max(default, 650.0)
    if event_count >= 500:
        return min(default, 80.0)
    return default


def fit_scope_logit_shift(rows: pd.DataFrame, p_col: str, global_success: float, scope_name: str, prior_strength: float | None = None) -> dict[str, object]:
    frame = rows.dropna(subset=[TARGET_COL, p_col]).copy()
    n = len(frame)
    effective_prior = float(scope_prior_strength(scope_name, n) if prior_strength is None else prior_strength)
    if n == 0:
        return {
            "scope": scope_name,
            "event_count": 0,
            "actual_success_rate": np.nan,
            "predicted_success_rate": np.nan,
            "posterior_success_rate": np.nan,
            "logit_shift": 0.0,
            "shrinkage": 0.0,
            "prior_strength": effective_prior,
            "status": "NO_SCOPE_CALIBRATION_EVENTS",
        }
    actual = float(frame[TARGET_COL].mean())
    predicted = float(frame[p_col].mean())
    posterior = float((frame[TARGET_COL].sum() + effective_prior * global_success) / (n + effective_prior))
    if n < MIN_TSM_CALIBRATION_EVENTS or frame[TARGET_COL].nunique() < 2 or pd.isna(predicted):
        return {
            "scope": scope_name,
            "event_count": n,
            "actual_success_rate": actual,
            "predicted_success_rate": predicted,
            "posterior_success_rate": posterior,
            "logit_shift": 0.0,
            "shrinkage": 0.0,
            "prior_strength": effective_prior,
            "status": "SCOPE_CALIBRATION_INSUFFICIENT_SAMPLE",
        }
    shrinkage = float(n / (n + effective_prior))
    shift = float((logit(posterior) - logit(predicted)) * shrinkage)
    return {
        "scope": scope_name,
        "event_count": n,
        "actual_success_rate": actual,
        "predicted_success_rate": predicted,
        "posterior_success_rate": posterior,
        "logit_shift": shift,
        "shrinkage": shrinkage,
        "prior_strength": effective_prior,
        "status": "SCOPE_CALIBRATION_READY",
    }


def apply_logit_shift(values, layer: dict[str, object]) -> pd.Series:
    shift = safe_float(layer.get("logit_shift", 0.0), 0.0)
    return pd.to_numeric(values, errors="coerce").map(lambda p: inv_logit(logit(p) + shift) if pd.notna(p) else np.nan)


def scoped_base_rate(train: pd.DataFrame, scope_name: str, default: float) -> float:
    if scope_name == TRADE_READY_EVAL_SCOPE and DECISION_ENTRY_COL in train.columns:
        scoped = train[train[DECISION_ENTRY_COL].map(to_bool)]
    else:
        scoped = train
    if scoped.empty or TARGET_COL not in scoped.columns:
        return default
    value = pd.to_numeric(scoped[TARGET_COL], errors="coerce").mean()
    return safe_float(value, default)


def choose_threshold(
    validation_predictions: pd.DataFrame,
    thresholds: Sequence[float] = THRESHOLDS,
    score_col: str = "decision_score",
    return_col: str = RETURN_COL,
    stop_col: str = "label_stop_hit_20d",
    min_selected: int = MIN_SELECTED_EVAL_EVENTS,
    probability_col: str | None = None,
    base_rate: float | None = None,
) -> Dict[str, object]:
    data = validation_predictions.dropna(subset=[score_col, return_col]).copy() if score_col in validation_predictions.columns else pd.DataFrame()
    if data.empty:
        return {
            "threshold": np.nan,
            "threshold_reason": "NO_VALIDATION_ROWS",
            "threshold_decision_eligible": False,
            "threshold_table": pd.DataFrame(),
            "trial_count": 0,
        }
    effective_min_selected = min(
        int(min_selected),
        max(1, int(math.ceil(len(data) * MIN_SELECTED_FRACTION))),
    )
    fixed_grid = np.linspace(float(data[score_col].min()), float(data[score_col].max()), 31)
    quantile_grid = data[score_col].quantile([0.40, 0.50, 0.60, 0.70, 0.80]).to_numpy()
    thresholds_out = sorted(set(np.round(np.concatenate([fixed_grid, quantile_grid, np.asarray(list(thresholds), dtype=float)]), 8)))
    all_mean = float(pd.to_numeric(data[return_col], errors="coerce").mean())
    all_stop = float(pd.to_numeric(data[stop_col], errors="coerce").mean()) if stop_col in data.columns else np.nan
    validation_ece = np.nan
    validation_brier_improvement = np.nan
    if probability_col and probability_col in data.columns and base_rate is not None and pd.notna(base_rate):
        validation_brier = brier_score(data[TARGET_COL], data[probability_col])
        validation_base_brier = brier_score(data[TARGET_COL], pd.Series(float(base_rate), index=data.index))
        validation_brier_improvement = (
            float((validation_base_brier - validation_brier) / validation_base_brier * 100.0)
            if pd.notna(validation_base_brier) and validation_base_brier > 0 and pd.notna(validation_brier)
            else np.nan
        )
        validation_ece, _ = ece_score(data[TARGET_COL], data[probability_col])
    probability_quality_required = probability_col is not None
    probability_quality_pass = (
        not probability_quality_required
        or (
            pd.notna(validation_brier_improvement)
            and validation_brier_improvement > 0.0
            and pd.notna(validation_ece)
            and validation_ece <= MAX_ECE
        )
    )
    rows = []
    for threshold in thresholds_out:
        selected = data[pd.to_numeric(data[score_col], errors="coerce") >= threshold].copy()
        selected_n = len(selected)
        if selected_n == 0:
            continue
        selected_fraction = selected_n / len(data)
        selected_mean = float(pd.to_numeric(selected[return_col], errors="coerce").mean())
        selected_minus_all = selected_mean - all_mean
        selected_ci_lower = return_ci_lower(selected[return_col])
        selected_stop = float(pd.to_numeric(selected[stop_col], errors="coerce").mean()) if stop_col in selected.columns else np.nan
        economics_pass = (
            selected_n >= effective_min_selected
            and MIN_SELECTED_FRACTION <= selected_fraction <= MAX_SELECTED_FRACTION
            and selected_minus_all > MIN_SELECTED_MINUS_ALL_TARGET_PCT
            and selected_ci_lower > MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT
            and (pd.isna(all_stop) or selected_stop <= all_stop - MIN_STOP_RATE_IMPROVEMENT)
        )
        eligible = (
            economics_pass
            and probability_quality_pass
        )
        rows.append(
            {
                "threshold": float(threshold),
                "trial_count": len(thresholds_out),
                "score_col": score_col,
                "probability_col": probability_col or "",
                "effective_min_selected": effective_min_selected,
                "selected_count": selected_n,
                "selected_fraction": selected_fraction,
                "selected_success_rate": float(selected[TARGET_COL].mean()) if TARGET_COL in selected.columns else np.nan,
                "selected_mean_return_pct": selected_mean,
                "selected_minus_all_pct": selected_minus_all,
                "selected_ci_lower_pct": selected_ci_lower,
                "selected_stop_hit": selected_stop,
                "all_stop_hit": all_stop,
                "validation_brier_improvement_pct": validation_brier_improvement,
                "validation_ece": validation_ece,
                "probability_quality_pass": probability_quality_pass,
                "economics_pass": economics_pass,
                "eligible": eligible,
            }
        )
    policy = pd.DataFrame(rows)
    eligible = policy[policy["eligible"].astype(bool)].copy() if not policy.empty else pd.DataFrame()
    if eligible.empty:
        diagnostic = policy[policy["economics_pass"].astype(bool)].copy() if not policy.empty and "economics_pass" in policy.columns else pd.DataFrame()
        if not diagnostic.empty:
            best = diagnostic.sort_values(
                ["selected_ci_lower_pct", "selected_minus_all_pct", "selected_count"],
                ascending=[False, False, False],
            ).iloc[0]
            return {
                "threshold": float(best["threshold"]),
                "threshold_reason": "DIAGNOSTIC_THRESHOLD_PROBABILITY_QUALITY_FAILED",
                "threshold_decision_eligible": False,
                "threshold_table": policy,
                "trial_count": len(thresholds_out),
            }
        return {
            "threshold": np.nan,
            "threshold_reason": "NO_THRESHOLD_WITH_ECONOMIC_UPLIFT",
            "threshold_decision_eligible": False,
            "threshold_table": policy,
            "trial_count": len(thresholds_out),
        }
    best = eligible.sort_values(
        ["selected_ci_lower_pct", "selected_minus_all_pct", "selected_count"],
        ascending=[False, False, False],
    ).iloc[0]
    return {
        "threshold": float(best["threshold"]),
        "threshold_reason": "VALIDATION_ECONOMIC_UPLIFT_AND_RISK_FILTER",
        "threshold_decision_eligible": True,
        "threshold_table": policy,
        "trial_count": len(thresholds_out),
    }


def utility_weight_label(weights: dict[str, float]) -> str:
    return (
        f"p={weights['p_success']:.2f};"
        f"stop={weights['p_stop_hit']:.2f};"
        f"r={weights['expected_r']:.2f};"
        f"score={weights['score_price_algo_total']:.2f}"
    )


def attach_utility_weight_columns(threshold_info: Dict[str, object], weights: dict[str, float]) -> Dict[str, object]:
    out = dict(threshold_info)
    out["utility_weights"] = dict(weights)
    out["utility_weight_label"] = utility_weight_label(weights)
    out["trial_count"] = int(out.get("trial_count", 0) or 0)
    table = out.get("threshold_table")
    if isinstance(table, pd.DataFrame) and not table.empty:
        table = table.copy()
        table["utility_weight_label"] = out["utility_weight_label"]
        table["utility_weight_p_success"] = weights["p_success"]
        table["utility_weight_p_stop_hit"] = weights["p_stop_hit"]
        table["utility_weight_expected_r"] = weights["expected_r"]
        table["utility_weight_score_price_algo_total"] = weights["score_price_algo_total"]
        out["threshold_table"] = table
    return out


def threshold_rank_tuple(threshold_info: Dict[str, object]) -> tuple:
    table = threshold_info.get("threshold_table")
    threshold = safe_float(threshold_info.get("threshold"))
    if not isinstance(table, pd.DataFrame) or table.empty or pd.isna(threshold):
        return (False, False, -np.inf, -np.inf, -np.inf, -np.inf)
    distances = (pd.to_numeric(table["threshold"], errors="coerce") - threshold).abs()
    chosen = table.loc[distances.idxmin()]
    selected_fraction = safe_float(chosen.get("selected_fraction"))
    fraction_distance = abs(selected_fraction - 0.45) if pd.notna(selected_fraction) else np.inf
    return (
        to_bool(threshold_info.get("threshold_decision_eligible", False)),
        to_bool(chosen.get("economics_pass", False)),
        safe_float(chosen.get("selected_minus_all_pct"), -np.inf),
        safe_float(chosen.get("selected_ci_lower_pct"), -np.inf),
        -fraction_distance,
        safe_float(chosen.get("selected_count"), -np.inf),
    )


def choose_utility_threshold(
    validation_predictions: pd.DataFrame,
    p_col: str,
    score_col: str,
    base_rate: float,
    stop_col: str = "label_stop_hit_20d",
    expected_r_col: str = "expected_r_lgbm",
) -> Dict[str, object]:
    if validation_predictions.empty or p_col not in validation_predictions.columns:
        empty = choose_threshold(
            validation_predictions,
            score_col=score_col,
            stop_col=stop_col,
            probability_col=p_col,
            base_rate=base_rate,
        )
        return {"weights": dict(DEFAULT_UTILITY_WEIGHTS), "threshold_info": attach_utility_weight_columns(empty, DEFAULT_UTILITY_WEIGHTS)}
    weights = dict(DEFAULT_UTILITY_WEIGHTS)
    tmp = validation_predictions.copy()
    tmp[score_col] = utility_score_frame(tmp, p_col, "p_stop_hit_lgbm", expected_r_col, weights)
    info = choose_threshold(
        tmp,
        score_col=score_col,
        stop_col=stop_col,
        probability_col=p_col,
        base_rate=base_rate,
    )
    info = dict(info)
    info["threshold_reason"] = f"FIXED_20D_UTILITY_WEIGHTS_{info.get('threshold_reason', 'UNKNOWN')}"
    return {"weights": weights, "threshold_info": attach_utility_weight_columns(info, weights)}


def choose_stable_trade_ready_threshold_v3(
    scoped_validation: pd.DataFrame,
    broad_validation: pd.DataFrame,
    broad_threshold_info: Dict[str, object],
    score_col: str,
    stop_col: str,
    probability_col: str,
    base_rate: float,
    oof_fold_metrics: pd.DataFrame | None = None,
) -> Dict[str, object]:
    scoped_info = choose_threshold(
        scoped_validation,
        score_col=score_col,
        stop_col=stop_col,
        probability_col=probability_col,
        base_rate=base_rate,
    )
    if pd.notna(safe_float(scoped_info.get("threshold"))):
        scoped_info = dict(scoped_info)
        scoped_info["strict_scope_event_count"] = int(len(scoped_validation))
        scoped_info["threshold_stability_pass"] = np.nan
        if isinstance(oof_fold_metrics, pd.DataFrame) and not oof_fold_metrics.empty:
            scoped_info["threshold_stability_pass"] = bool(threshold_stability_from_fold_metrics(oof_fold_metrics)[0])
        return scoped_info
    source = scoped_validation if not scoped_validation.empty else broad_validation
    if source.empty or score_col not in source.columns:
        out = dict(scoped_info)
        out["threshold_reason"] = f"STRICT_SCOPE_INSUFFICIENT_FOR_DECISION_{scoped_info.get('threshold_reason', 'UNKNOWN')}"
        out["strict_scope_event_count"] = int(len(scoped_validation))
        out["threshold_stability_pass"] = False
        return out
    broad_threshold = safe_float(broad_threshold_info.get("threshold"))
    if pd.notna(broad_threshold):
        threshold = broad_threshold
        reason = f"DIAGNOSTIC_BROAD_SCOPE_FALLBACK_{scoped_info.get('threshold_reason', 'UNKNOWN')}"
    else:
        scores = pd.to_numeric(source[score_col], errors="coerce").dropna()
        if scores.empty:
            out = dict(scoped_info)
            out["threshold_reason"] = f"STRICT_SCOPE_INSUFFICIENT_FOR_DECISION_{scoped_info.get('threshold_reason', 'UNKNOWN')}"
            out["strict_scope_event_count"] = int(len(scoped_validation))
            out["threshold_stability_pass"] = False
            return out
        threshold = float(scores.quantile(0.50))
        reason = f"STRICT_SCOPE_INSUFFICIENT_FOR_DECISION_MEDIAN_FALLBACK_{scoped_info.get('threshold_reason', 'UNKNOWN')}"
    table_info = choose_threshold(
        source,
        thresholds=[threshold],
        score_col=score_col,
        stop_col=stop_col,
        probability_col=probability_col,
        base_rate=base_rate,
        min_selected=1,
    )
    table = table_info["threshold_table"].copy()
    threshold_stability_pass = np.nan
    if isinstance(oof_fold_metrics, pd.DataFrame) and not oof_fold_metrics.empty:
        threshold_stability_pass = bool(threshold_stability_from_fold_metrics(oof_fold_metrics)[0])
    return {
        "threshold": threshold,
        "threshold_reason": reason,
        "threshold_decision_eligible": False,
        "threshold_table": table,
        "trial_count": int(broad_threshold_info.get("trial_count", 0) or 0) + int(table_info.get("trial_count", 0) or 0),
        "strict_scope_event_count": int(len(scoped_validation)),
        "threshold_stability_pass": threshold_stability_pass,
    }


def choose_threshold_with_scope_fallback(
    scoped_validation: pd.DataFrame,
    broad_validation: pd.DataFrame,
    broad_threshold_info: Dict[str, object],
    score_col: str,
    stop_col: str,
    probability_col: str,
    base_rate: float,
) -> Dict[str, object]:
    return choose_stable_trade_ready_threshold_v3(
        scoped_validation,
        broad_validation,
        broad_threshold_info,
        score_col=score_col,
        stop_col=stop_col,
        probability_col=probability_col,
        base_rate=base_rate,
    )


def threshold_stability_from_fold_metrics(metrics: pd.DataFrame) -> tuple[bool, str, float]:
    if metrics.empty:
        return False, "MISSING_OOF_FOLDS", np.nan
    fold_rows = metrics[metrics["split"].astype(str).str.startswith("oof_test_")].copy()
    if fold_rows.empty:
        fold_rows = metrics.copy()
    selected_counts = pd.to_numeric(fold_rows.get("selected_event_count", pd.Series(dtype=float)), errors="coerce")
    selected_fraction = pd.to_numeric(fold_rows.get("selected_fraction", pd.Series(dtype=float)), errors="coerce")
    weak_parts: list[str] = []
    for _, row in fold_rows.iterrows():
        split = str(row.get("split", "fold"))
        selected_count = safe_float(row.get("selected_event_count"))
        fraction = safe_float(row.get("selected_fraction"))
        if pd.notna(selected_count) and selected_count < MIN_SELECTED_PER_EVAL_SPLIT:
            weak_parts.append(f"{split}:selected_lt_{MIN_SELECTED_PER_EVAL_SPLIT}")
        if pd.notna(fraction) and (fraction < MIN_SELECTED_FRACTION or fraction > MAX_SELECTED_FRACTION):
            weak_parts.append(f"{split}:selected_fraction_out_of_range")
    thresholds = pd.to_numeric(fold_rows.get("threshold", pd.Series(dtype=float)), errors="coerce").dropna()
    threshold_iqr = float(thresholds.quantile(0.75) - thresholds.quantile(0.25)) if not thresholds.empty else np.nan
    policy_source = fold_rows.get("applied_threshold_policy_type", fold_rows.get("threshold_policy_type", pd.Series(dtype=object)))
    policy_values = set(policy_source.dropna().astype(str))
    policy_type = "rank_percentile_policy" if policy_values.intersection({"rank_percentile_policy", "selection_percentile"}) else "raw_score"
    fraction_drift = float(selected_fraction.max() - selected_fraction.min()) if selected_fraction.notna().any() else np.nan
    stability_metric_pass = (
        pd.notna(threshold_iqr) and threshold_iqr <= MAX_THRESHOLD_IQR
        if policy_type == "raw_score"
        else pd.notna(fraction_drift) and fraction_drift <= MAX_SELECTION_FRACTION_DRIFT
    )
    stable = (
        selected_counts.notna().all()
        and bool((selected_counts >= MIN_SELECTED_PER_EVAL_SPLIT).all())
        and selected_fraction.notna().all()
        and bool(((selected_fraction >= MIN_SELECTED_FRACTION) & (selected_fraction <= MAX_SELECTED_FRACTION)).all())
        and stability_metric_pass
    )
    if policy_type == "raw_score":
        if pd.isna(threshold_iqr):
            weak_parts.append("threshold_iqr_missing")
        elif threshold_iqr > MAX_THRESHOLD_IQR:
            weak_parts.append("threshold_iqr_gt_0_10")
    else:
        if pd.isna(fraction_drift):
            weak_parts.append("selection_fraction_drift_missing")
        elif fraction_drift > MAX_SELECTION_FRACTION_DRIFT:
            weak_parts.append("selection_fraction_drift_gt_0_10")
    return bool(stable), "|".join(sorted(set(weak_parts))), threshold_iqr


def split_unique_values(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns or frame.empty:
        return ""
    values = [str(v) for v in frame[column].dropna().astype(str) if str(v) and str(v).lower() != "nan"]
    return "|".join(sorted(set(values)))


def percentile_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    ranks = values.rank(method="average", pct=True)
    return ranks.fillna(0.5).astype(float)


def risk_weight_label(weights: dict[str, object]) -> str:
    return str(weights.get("label") or f"success={safe_float(weights.get('w_success'), 0.0):.2f};stop={safe_float(weights.get('w_stop'), 0.0):.2f};r={safe_float(weights.get('w_r'), 0.0):.2f};rule={safe_float(weights.get('w_rule'), 0.0):.2f}")


def add_risk_adjusted_selection_score(
    frame: pd.DataFrame,
    p_col: str,
    weights: dict[str, object],
    output_col: str = "risk_adjusted_selection_score",
) -> pd.DataFrame:
    out = frame.copy()
    success_rank = percentile_rank(out[p_col]) if p_col in out.columns else pd.Series(0.5, index=out.index, dtype=float)
    if "p_stop_hit_lgbm" in out.columns:
        stop_survival_rank = percentile_rank(1.0 - pd.to_numeric(out["p_stop_hit_lgbm"], errors="coerce"))
    elif STOP_SURVIVAL_COL in out.columns:
        stop_survival_rank = percentile_rank(out[STOP_SURVIVAL_COL])
    else:
        stop_survival_rank = pd.Series(0.5, index=out.index, dtype=float)
    expected_r_rank = percentile_rank(out["expected_r_lgbm"]) if "expected_r_lgbm" in out.columns else pd.Series(0.5, index=out.index, dtype=float)
    rule_score_rank = percentile_rank(out["score_price_algo_total"]) if "score_price_algo_total" in out.columns else pd.Series(0.5, index=out.index, dtype=float)
    out["success_rank"] = success_rank
    out["stop_survival_rank"] = stop_survival_rank
    out["expected_r_rank"] = expected_r_rank
    out["rule_score_rank"] = rule_score_rank
    out[output_col] = (
        safe_float(weights.get("w_success"), 0.0) * success_rank
        + safe_float(weights.get("w_stop"), 0.0) * stop_survival_rank
        + safe_float(weights.get("w_r"), 0.0) * expected_r_rank
        + safe_float(weights.get("w_rule"), 0.0) * rule_score_rank
    )
    return out


def select_top_fraction_by_score(frame: pd.DataFrame, score_col: str, target_fraction: float) -> tuple[pd.Series, float]:
    scores = pd.to_numeric(frame.get(score_col, pd.Series(dtype=float)), errors="coerce")
    if frame.empty or scores.dropna().empty or pd.isna(target_fraction):
        return pd.Series(False, index=frame.index, dtype=bool), np.nan
    target = max(0.0, min(1.0, float(target_fraction)))
    threshold = float(scores.dropna().quantile(max(0.0, min(1.0, 1.0 - target))))
    selected = (scores >= threshold).astype(bool)
    return selected.reindex(frame.index).fillna(False).astype(bool), threshold


def v5_metric_frame(
    rows: pd.DataFrame,
    p_col: str,
    score_col: str,
    selected: pd.Series,
    threshold: float,
    model_name: str,
    split_name: str,
    base_rate: float,
    candidate: dict[str, object],
    policy_role: str,
) -> pd.DataFrame:
    out = rows.copy()
    out["p_success"] = pd.to_numeric(out[p_col], errors="coerce") if p_col in out.columns else np.nan
    out["utility_score"] = pd.to_numeric(out[score_col], errors="coerce") if score_col in out.columns else np.nan
    out["threshold"] = threshold
    out["selected_by_threshold"] = selected.reindex(out.index).fillna(False).astype(bool)
    out["model_name"] = model_name
    out["evaluation_scope"] = TRADE_READY_EVAL_SCOPE
    out["split"] = split_name
    out["validation_design"] = "walk_forward_oof"
    out["base_rate"] = base_rate
    out["threshold_policy_type"] = str(candidate.get("applied_threshold_policy_type", "rank_percentile_policy"))
    out["applied_threshold_policy_type"] = str(candidate.get("applied_threshold_policy_type", "rank_percentile_policy"))
    out["diagnostic_best_candidate_policy_type"] = str(candidate.get("diagnostic_best_candidate_policy_type", "raw_score"))
    out["threshold_policy_source_window"] = str(candidate.get("threshold_policy_source_window", "threshold"))
    out["threshold_policy_applied_window"] = str(candidate.get("threshold_policy_applied_window", "test"))
    out["threshold_policy_provenance_valid"] = bool(candidate.get("threshold_policy_provenance_valid", True))
    out["risk_adjusted_selection_score_col"] = score_col
    out["policy_role"] = policy_role
    out["risk_adjusted_weight_label"] = str(candidate.get("risk_adjusted_weight_label", ""))
    out["target_selected_fraction"] = safe_float(candidate.get("target_selected_fraction"), np.nan)
    return out


def candidate_metric_summary(metric: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    failure_reasons: list[str] = []
    selected_fraction = safe_float(metric.get("selected_fraction"))
    selected_count = safe_float(metric.get("selected_event_count"))
    stop_improvement = safe_float(metric.get("stop_rate_improvement"))
    paired_lower = safe_float(metric.get("selected_minus_all_ci_lower_pct_paired", metric.get("selected_minus_all_ci_lower_pct")))
    if selected_count < MIN_SELECTED_PER_EVAL_SPLIT:
        failure_reasons.append("FOLD_SELECTED_COUNT_LT_10")
    if pd.isna(selected_fraction) or selected_fraction < MIN_SELECTED_FRACTION or selected_fraction > MAX_SELECTED_FRACTION:
        failure_reasons.append("FOLD_SELECTED_FRACTION_OUT_OF_RANGE")
    if paired_lower <= MIN_SELECTED_MINUS_ALL_PCT:
        failure_reasons.append("SELECTED_MINUS_ALL_PAIRED_CI_LOWER_LE_0")
    if stop_improvement < MIN_STOP_RATE_IMPROVEMENT:
        failure_reasons.append("FOLD_STOP_IMPROVEMENT_LT_5PCT")
    return {
        **candidate,
        "selected_event_count": metric.get("selected_event_count", np.nan),
        "selected_fraction": metric.get("selected_fraction", np.nan),
        "selected_minus_all_ci_lower_pct": metric.get("selected_minus_all_ci_lower_pct", np.nan),
        "selected_minus_all_ci_lower_pct_paired": metric.get("selected_minus_all_ci_lower_pct_paired", np.nan),
        "selected_minus_score_baseline_ci_lower_pct": metric.get("selected_minus_score_baseline_ci_lower_pct", np.nan),
        "selected_minus_score_baseline_ci_lower_pct_paired": metric.get("selected_minus_score_baseline_ci_lower_pct_paired", np.nan),
        "stop_rate_improvement": metric.get("stop_rate_improvement", np.nan),
        "brier_improvement_pct": metric.get("brier_improvement_pct", np.nan),
        "decision_ece": metric.get("decision_ece", metric.get("ece", np.nan)),
        "pass_fail_count": len(failure_reasons),
        "threshold_decision_eligible": len(failure_reasons) == 0,
        "threshold_stability_pass": len(failure_reasons) == 0,
        "threshold_stability_failure_reasons": "|".join(failure_reasons) if failure_reasons else "PASS",
    }


def threshold_candidate_grids(records: pd.DataFrame) -> list[dict[str, object]]:
    scores = pd.to_numeric(records.get("utility_score", pd.Series(dtype=float)), errors="coerce").dropna()
    if scores.empty:
        return []
    fixed_grid = np.linspace(float(scores.min()), float(scores.max()), 31)
    quantile_grid = scores.quantile([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]).to_numpy()
    existing = pd.to_numeric(records.get("threshold", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
    raw_thresholds = sorted(set(np.round(np.concatenate([fixed_grid, quantile_grid, existing]), 8)))
    candidates: list[dict[str, object]] = [
        {"threshold_policy_type": "raw_score", "threshold": float(threshold), "target_selected_fraction": np.nan}
        for threshold in raw_thresholds
    ]
    candidates.extend(
        {
            "threshold_policy_type": "rank_percentile_policy",
            "threshold": np.nan,
            "target_selected_fraction": float(target_fraction),
        }
        for target_fraction in CONSENSUS_SELECTION_FRACTIONS
    )
    return candidates


def apply_threshold_candidate_selection(records: pd.DataFrame, candidate: dict[str, object]) -> pd.DataFrame:
    out = records.copy()
    out["selected_by_threshold"] = False
    out["threshold_policy_type"] = str(candidate["threshold_policy_type"])
    if candidate["threshold_policy_type"] == "raw_score":
        threshold = safe_float(candidate.get("threshold"))
        out["threshold"] = threshold
        out["selected_by_threshold"] = pd.to_numeric(out["utility_score"], errors="coerce") >= threshold
        return out
    target_fraction = safe_float(candidate.get("target_selected_fraction"))
    for split_name, split_rows in out.groupby("split", dropna=False):
        split_scores = pd.to_numeric(split_rows["utility_score"], errors="coerce")
        if split_scores.dropna().empty or pd.isna(target_fraction):
            threshold = np.nan
            selected = pd.Series(False, index=split_rows.index)
        else:
            threshold = float(split_scores.dropna().quantile(max(0.0, min(1.0, 1.0 - target_fraction))))
            selected = split_scores >= threshold
        out.loc[split_rows.index, "threshold"] = threshold
        out.loc[split_rows.index, "selected_by_threshold"] = selected.astype(bool)
    return out


def threshold_candidate_metrics(candidate_records: pd.DataFrame, candidate_id: int, candidate: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
    model_name = str(candidate_records["model_name"].dropna().iloc[0]) if "model_name" in candidate_records.columns and candidate_records["model_name"].notna().any() else ""
    evaluation_scope = str(candidate_records["evaluation_scope"].dropna().iloc[0]) if "evaluation_scope" in candidate_records.columns and candidate_records["evaluation_scope"].notna().any() else TRADE_READY_EVAL_SCOPE
    fold_rows = []
    for split_name, group in candidate_records.groupby("split", dropna=False):
        if not str(split_name).startswith("oof_test_"):
            continue
        metric = oof_metric_row(str(split_name), group, model_name, evaluation_scope)
        fold_rows.append(metric)
    combined = oof_metric_row("combined_test_holdout", candidate_records, model_name, evaluation_scope)
    fold_df = pd.DataFrame(fold_rows)
    selected_counts = pd.to_numeric(fold_df.get("selected_event_count", pd.Series(dtype=float)), errors="coerce")
    selected_fractions = pd.to_numeric(fold_df.get("selected_fraction", pd.Series(dtype=float)), errors="coerce")
    fold_stop_improvement = pd.to_numeric(fold_df.get("stop_rate_improvement", pd.Series(dtype=float)), errors="coerce")
    fold_uplift_lower = pd.to_numeric(
        fold_df.get("selected_minus_all_ci_lower_pct_paired", fold_df.get("selected_minus_all_ci_lower_pct", pd.Series(dtype=float))),
        errors="coerce",
    )
    fold_score_lower = pd.to_numeric(
        fold_df.get("selected_minus_score_baseline_ci_lower_pct_paired", fold_df.get("selected_minus_score_baseline_ci_lower_pct", pd.Series(dtype=float))),
        errors="coerce",
    )
    thresholds = pd.to_numeric(fold_df.get("threshold", pd.Series(dtype=float)), errors="coerce").dropna()
    threshold_iqr = float(thresholds.quantile(0.75) - thresholds.quantile(0.25)) if not thresholds.empty else np.nan
    fraction_drift = float(selected_fractions.max() - selected_fractions.min()) if selected_fractions.notna().any() else np.nan
    failure_reasons: list[str] = []
    if safe_float(combined.get("selected_event_count")) < MIN_SELECTED_EVAL_EVENTS:
        failure_reasons.append("COMBINED_SELECTED_COUNT_LT_50")
    if selected_counts.empty or selected_counts.min() < MIN_SELECTED_PER_EVAL_SPLIT:
        failure_reasons.append("FOLD_SELECTED_COUNT_LT_10")
    if selected_fractions.empty or selected_fractions.min() < MIN_SELECTED_FRACTION or selected_fractions.max() > MAX_SELECTED_FRACTION:
        failure_reasons.append("FOLD_SELECTED_FRACTION_OUT_OF_RANGE")
    if safe_float(combined.get("selected_minus_all_ci_lower_pct_paired", combined.get("selected_minus_all_ci_lower_pct"))) <= MIN_SELECTED_MINUS_ALL_PCT:
        failure_reasons.append("COMBINED_SELECTED_MINUS_ALL_CI_LOWER_LE_0")
    if fold_stop_improvement.empty or fold_stop_improvement.min() < MIN_STOP_RATE_IMPROVEMENT:
        failure_reasons.append("FOLD_STOP_IMPROVEMENT_LT_5PCT")
    if candidate["threshold_policy_type"] == "raw_score":
        if pd.isna(threshold_iqr) or threshold_iqr > MAX_THRESHOLD_IQR:
            failure_reasons.append("THRESHOLD_IQR_GT_0_10")
    elif pd.isna(fraction_drift) or fraction_drift > MAX_SELECTION_FRACTION_DRIFT:
        failure_reasons.append("SELECTION_FRACTION_DRIFT_GT_10PCT")
    pass_count = len(failure_reasons) == 0
    summary = {
        "candidate_id": candidate_id,
        "model_name": model_name,
        "evaluation_scope": evaluation_scope,
        "threshold_policy_type": candidate["threshold_policy_type"],
        "target_selected_fraction": candidate.get("target_selected_fraction", np.nan),
        "threshold": float(pd.to_numeric(candidate_records.get("threshold", pd.Series(dtype=float)), errors="coerce").median()),
        "threshold_iqr": threshold_iqr,
        "selection_fraction_drift": fraction_drift,
        "selected_event_count": combined.get("selected_event_count", np.nan),
        "selected_fraction": combined.get("selected_fraction", np.nan),
        "selected_minus_all_ci_lower_pct": combined.get("selected_minus_all_ci_lower_pct", np.nan),
        "selected_minus_all_ci_lower_pct_paired": combined.get("selected_minus_all_ci_lower_pct_paired", np.nan),
        "selected_minus_score_baseline_ci_lower_pct": combined.get("selected_minus_score_baseline_ci_lower_pct", np.nan),
        "selected_minus_score_baseline_ci_lower_pct_paired": combined.get("selected_minus_score_baseline_ci_lower_pct_paired", np.nan),
        "fold_selected_count_min": float(selected_counts.min()) if selected_counts.notna().any() else np.nan,
        "fold_selected_fraction_min": float(selected_fractions.min()) if selected_fractions.notna().any() else np.nan,
        "fold_selected_fraction_max": float(selected_fractions.max()) if selected_fractions.notna().any() else np.nan,
        "fold_stop_improvement_min": float(fold_stop_improvement.min()) if fold_stop_improvement.notna().any() else np.nan,
        "fold_uplift_ci_lower_min": float(fold_uplift_lower.min()) if fold_uplift_lower.notna().any() else np.nan,
        "fold_score_baseline_ci_lower_min": float(fold_score_lower.min()) if fold_score_lower.notna().any() else np.nan,
        "brier_improvement_pct": combined.get("brier_improvement_pct", np.nan),
        "decision_ece": combined.get("decision_ece", combined.get("ece", np.nan)),
        "pass_fail_count": len(failure_reasons),
        "threshold_decision_eligible": pass_count,
        "threshold_stability_pass": pass_count,
        "threshold_stability_failure_reasons": "|".join(failure_reasons) if failure_reasons else "PASS",
    }
    policy_rows: list[dict[str, object]] = [{**summary, "split": "combined_test_holdout"}]
    for fold_metric in fold_rows:
        policy_rows.append(
            {
                **summary,
                "split": fold_metric.get("split", ""),
                "selected_event_count": fold_metric.get("selected_event_count", np.nan),
                "selected_fraction": fold_metric.get("selected_fraction", np.nan),
                "selected_minus_all_ci_lower_pct": fold_metric.get("selected_minus_all_ci_lower_pct", np.nan),
                "selected_minus_all_ci_lower_pct_paired": fold_metric.get("selected_minus_all_ci_lower_pct_paired", np.nan),
                "selected_minus_score_baseline_ci_lower_pct": fold_metric.get("selected_minus_score_baseline_ci_lower_pct", np.nan),
                "selected_minus_score_baseline_ci_lower_pct_paired": fold_metric.get("selected_minus_score_baseline_ci_lower_pct_paired", np.nan),
                "stop_rate_improvement": fold_metric.get("stop_rate_improvement", np.nan),
                "threshold": fold_metric.get("threshold", np.nan),
            }
        )
    return summary, policy_rows


def choose_fold_consensus_trade_ready_threshold_v4(records: pd.DataFrame) -> dict[str, object]:
    if records.empty or "utility_score" not in records.columns:
        return {"records": records.copy(), "threshold_table": pd.DataFrame(), "chosen": {}, "stable_threshold_candidate_count": 0}
    candidates = threshold_candidate_grids(records)
    original_trial_count = int(pd.to_numeric(records.get("trial_count", pd.Series([0])), errors="coerce").fillna(0).max())
    candidate_summaries: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    selected_records_by_candidate: dict[int, pd.DataFrame] = {}
    for candidate_id, candidate in enumerate(candidates, start=1):
        candidate_records = apply_threshold_candidate_selection(records, candidate)
        candidate_records["trial_count"] = original_trial_count + len(candidates)
        summary, rows = threshold_candidate_metrics(candidate_records, candidate_id, candidate)
        candidate_summaries.append(summary)
        policy_rows.extend(rows)
        selected_records_by_candidate[candidate_id] = candidate_records
    summary_frame = pd.DataFrame(candidate_summaries)
    policy = pd.DataFrame(policy_rows)
    if summary_frame.empty:
        return {"records": records.copy(), "threshold_table": policy, "chosen": {}, "stable_threshold_candidate_count": 0}
    eligible = summary_frame[summary_frame["threshold_decision_eligible"].astype(bool)].copy()
    rank_source = eligible if not eligible.empty else summary_frame.copy()
    rank_source["fraction_distance"] = (pd.to_numeric(rank_source["selected_fraction"], errors="coerce") - 0.45).abs()
    chosen = rank_source.sort_values(
        [
            "pass_fail_count",
            "fold_uplift_ci_lower_min",
            "selected_minus_all_ci_lower_pct",
            "fraction_distance",
            "brier_improvement_pct",
            "decision_ece",
        ],
        ascending=[True, False, False, True, False, True],
    ).iloc[0].to_dict()
    stable_count = int(len(eligible))
    out_records = records.copy()
    chosen_id = int(chosen.get("candidate_id", 0))
    if stable_count > 0 and chosen_id in selected_records_by_candidate:
        out_records = selected_records_by_candidate[chosen_id].copy()
        out_records["threshold_reason"] = "FOLD_CONSENSUS_TRADE_READY_THRESHOLD_V4"
        out_records["threshold_decision_eligible"] = True
    else:
        existing_reason = split_unique_values(out_records, "threshold_reason")
        failure = str(chosen.get("threshold_stability_failure_reasons", "NO_FOLD_CONSENSUS_THRESHOLD"))
        reason_parts = [part for part in existing_reason.split("|") if part]
        reason_parts.append("OOF_FOLD_SELECTION_UNSTABLE")
        out_records["threshold_reason"] = "|".join(sorted(set(reason_parts)))
        out_records["threshold_decision_eligible"] = False
        out_records["threshold_stability_failure_reasons"] = failure
        out_records["threshold_policy_type"] = split_unique_values(out_records, "threshold_policy_type") or "raw_score"
    out_records["stable_threshold_candidate_count"] = stable_count
    out_records["threshold_policy_type"] = str(chosen.get("threshold_policy_type", out_records.get("threshold_policy_type", "raw_score")))
    out_records["threshold_stability_failure_reasons"] = str(chosen.get("threshold_stability_failure_reasons", "PASS" if stable_count else "NO_FOLD_CONSENSUS_THRESHOLD"))
    out_records["trial_count"] = original_trial_count + int(len(summary_frame))
    return {"records": out_records, "threshold_table": policy, "chosen": chosen, "stable_threshold_candidate_count": stable_count}


def choose_fold_consensus_trade_ready_threshold_v5(
    threshold_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    model_name: str,
    p_col: str,
    raw_score_col: str,
    base_rate: float,
    fold_id: str,
    split_name: str,
    threshold_info: Dict[str, object] | None = None,
) -> dict[str, object]:
    policy_rows: list[dict[str, object]] = []
    if threshold_rows.empty or test_rows.empty or p_col not in threshold_rows.columns or p_col not in test_rows.columns:
        out = test_rows.copy()
        threshold_value = safe_float((threshold_info or {}).get("threshold"))
        selected = pd.to_numeric(out.get(raw_score_col, pd.Series(dtype=float)), errors="coerce") >= threshold_value if pd.notna(threshold_value) and raw_score_col in out.columns else pd.Series(False, index=out.index)
        out["p_success"] = pd.to_numeric(out.get(p_col, pd.Series(np.nan, index=out.index)), errors="coerce")
        out["utility_score"] = pd.to_numeric(out.get(raw_score_col, pd.Series(np.nan, index=out.index)), errors="coerce")
        out["threshold"] = threshold_value
        out["selected_by_threshold"] = selected.reindex(out.index).fillna(False).astype(bool)
        out["threshold_reason"] = "OOF_FOLD_SELECTION_UNSTABLE|V5_NO_THRESHOLD_SOURCE_ROWS"
        out["threshold_decision_eligible"] = False
        out["threshold_policy_type"] = "raw_score"
        out["applied_threshold_policy_type"] = "raw_score"
        out["diagnostic_best_candidate_policy_type"] = "raw_score"
        out["threshold_policy_source_window"] = "threshold"
        out["threshold_policy_applied_window"] = "test"
        out["threshold_policy_provenance_valid"] = True
        out["risk_adjusted_selection_score_col"] = raw_score_col
        out["stable_threshold_candidate_count"] = 0
        out["threshold_stability_failure_reasons"] = "V5_NO_THRESHOLD_SOURCE_ROWS"
        return {"records": out, "threshold_table": pd.DataFrame(policy_rows), "chosen": {}, "stable_threshold_candidate_count": 0}

    candidates: list[dict[str, object]] = []
    for weights in RISK_AWARE_SELECTION_WEIGHT_PROFILES:
        for target_fraction in CONSENSUS_SELECTION_FRACTIONS:
            candidates.append(
                {
                    "candidate_id": len(candidates) + 1,
                    "threshold_policy_type": "rank_percentile_policy",
                    "applied_threshold_policy_type": "rank_percentile_policy",
                    "diagnostic_best_candidate_policy_type": "raw_score",
                    "target_selected_fraction": float(target_fraction),
                    "risk_adjusted_weight_label": risk_weight_label(weights),
                    "risk_weights": dict(weights),
                    "threshold_policy_source_window": "threshold",
                    "threshold_policy_applied_window": "test",
                    "threshold_policy_provenance_valid": True,
                }
            )
    source_summaries: list[dict[str, object]] = []
    source_frames_by_candidate: dict[int, pd.DataFrame] = {}
    test_frames_by_candidate: dict[int, pd.DataFrame] = {}
    original_trial_count = int((threshold_info or {}).get("trial_count", 0) or 0)
    for candidate in candidates:
        score_col = "risk_adjusted_selection_score"
        source_scored = add_risk_adjusted_selection_score(threshold_rows, p_col, candidate["risk_weights"], output_col=score_col)
        test_scored = add_risk_adjusted_selection_score(test_rows, p_col, candidate["risk_weights"], output_col=score_col)
        source_selected, source_threshold = select_top_fraction_by_score(source_scored, score_col, safe_float(candidate["target_selected_fraction"]))
        test_selected, test_threshold = select_top_fraction_by_score(test_scored, score_col, safe_float(candidate["target_selected_fraction"]))
        source_metric_frame = v5_metric_frame(
            source_scored,
            p_col,
            score_col,
            source_selected,
            source_threshold,
            model_name,
            f"threshold_train_{fold_id.replace('wf_', '')}",
            base_rate,
            candidate,
            "candidate_train",
        )
        test_metric_frame = v5_metric_frame(
            test_scored,
            p_col,
            score_col,
            test_selected,
            test_threshold,
            model_name,
            split_name,
            base_rate,
            candidate,
            "locked_test_eval",
        )
        source_metric = oof_metric_row(
            f"threshold_train_{fold_id.replace('wf_', '')}",
            source_metric_frame,
            model_name,
            TRADE_READY_EVAL_SCOPE,
            bootstrap_iterations=THRESHOLD_POLICY_BOOTSTRAP_ITERATIONS,
        )
        test_metric = oof_metric_row(
            split_name,
            test_metric_frame,
            model_name,
            TRADE_READY_EVAL_SCOPE,
            bootstrap_iterations=THRESHOLD_POLICY_BOOTSTRAP_ITERATIONS,
        )
        source_summary = candidate_metric_summary(source_metric, candidate)
        source_summary.update(
            {
                "fold_id": fold_id,
                "split": f"threshold_train_{fold_id.replace('wf_', '')}",
                "policy_role": "candidate_train",
                "threshold": source_threshold,
                "trial_count": original_trial_count + len(candidates),
            }
        )
        test_summary = candidate_metric_summary(test_metric, candidate)
        test_summary.update(
            {
                "fold_id": fold_id,
                "split": split_name,
                "policy_role": "locked_test_eval",
                "threshold": test_threshold,
                "trial_count": original_trial_count + len(candidates),
            }
        )
        source_summaries.append(source_summary)
        policy_rows.extend([source_summary, test_summary])
        source_frames_by_candidate[int(candidate["candidate_id"])] = source_metric_frame
        test_frames_by_candidate[int(candidate["candidate_id"])] = test_metric_frame

    source_table = pd.DataFrame(source_summaries)
    if source_table.empty:
        return {"records": test_rows.copy(), "threshold_table": pd.DataFrame(policy_rows), "chosen": {}, "stable_threshold_candidate_count": 0}
    source_table["fraction_distance"] = (pd.to_numeric(source_table["selected_fraction"], errors="coerce") - 0.45).abs()
    source_table["paired_lower_rank"] = pd.to_numeric(
        source_table.get("selected_minus_all_ci_lower_pct_paired", source_table.get("selected_minus_all_ci_lower_pct")),
        errors="coerce",
    )
    source_table["score_paired_lower_rank"] = pd.to_numeric(
        source_table.get("selected_minus_score_baseline_ci_lower_pct_paired", source_table.get("selected_minus_score_baseline_ci_lower_pct")),
        errors="coerce",
    )
    source_table["stop_rank"] = pd.to_numeric(source_table.get("stop_rate_improvement"), errors="coerce")
    ranked = source_table.sort_values(
        [
            "pass_fail_count",
            "paired_lower_rank",
            "score_paired_lower_rank",
            "stop_rank",
            "fraction_distance",
            "brier_improvement_pct",
            "decision_ece",
        ],
        ascending=[True, False, False, False, True, False, True],
    )
    chosen = ranked.iloc[0].to_dict()
    chosen_id = int(chosen.get("candidate_id", 0))
    stable_source_count = int(source_table["threshold_decision_eligible"].map(to_bool).sum())
    out_records = test_frames_by_candidate.get(chosen_id, test_rows.copy()).copy()
    out_records["threshold_reason"] = "FOLD_CONSENSUS_TRADE_READY_THRESHOLD_V5_LOCKED_SOURCE"
    out_records["threshold_decision_eligible"] = False
    out_records["stable_threshold_candidate_count"] = stable_source_count
    out_records["threshold_stability_failure_reasons"] = str(chosen.get("threshold_stability_failure_reasons", "PENDING_LOCKED_TEST_EVALUATION"))
    out_records["trial_count"] = original_trial_count + len(candidates)
    out_records["raw_utility_score"] = pd.to_numeric(test_rows.get(raw_score_col, pd.Series(np.nan, index=test_rows.index)), errors="coerce").reindex(out_records.index)
    out_records["threshold_policy_type"] = str(chosen.get("applied_threshold_policy_type", "rank_percentile_policy"))
    out_records["applied_threshold_policy_type"] = str(chosen.get("applied_threshold_policy_type", "rank_percentile_policy"))
    out_records["diagnostic_best_candidate_policy_type"] = str(chosen.get("diagnostic_best_candidate_policy_type", "raw_score"))
    out_records["threshold_policy_source_window"] = "threshold"
    out_records["threshold_policy_applied_window"] = "test"
    out_records["threshold_policy_provenance_valid"] = True
    out_records["risk_adjusted_selection_score_col"] = "risk_adjusted_selection_score"
    return {
        "records": out_records,
        "threshold_table": pd.DataFrame(policy_rows),
        "chosen": chosen,
        "stable_threshold_candidate_count": stable_source_count,
    }


def apply_fold_consensus_thresholds(oof_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if oof_predictions.empty or "evaluation_scope" not in oof_predictions.columns:
        return oof_predictions.copy(), pd.DataFrame()
    output_parts: list[pd.DataFrame] = []
    policy_parts: list[pd.DataFrame] = []
    for (_, _, _), group in oof_predictions.groupby(["model_name", "evaluation_scope", "validation_design"], dropna=False):
        if str(group["evaluation_scope"].iloc[0]) != TRADE_READY_EVAL_SCOPE:
            output_parts.append(group.copy())
            continue
        result = choose_fold_consensus_trade_ready_threshold_v4(group.copy())
        output_parts.append(result["records"])
        table = result["threshold_table"]
        if isinstance(table, pd.DataFrame) and not table.empty:
            policy_parts.append(table)
    output = pd.concat(output_parts, ignore_index=True) if output_parts else oof_predictions.copy()
    policy = pd.concat(policy_parts, ignore_index=True) if policy_parts else pd.DataFrame()
    return output, policy


def metric_row(
    split_name: str,
    predictions: pd.DataFrame,
    p_col: str,
    threshold: float,
    base_rate: float,
    model_name: str,
    score_col: str | None = None,
    evaluation_scope: str = ENTRY_RESEARCH_EVAL_SCOPE,
    bootstrap_iterations: int = UPLIFT_BOOTSTRAP_ITERATIONS,
) -> Dict[str, object]:
    frame = predictions.dropna(subset=[TARGET_COL, RETURN_COL, p_col]).copy()
    selector_col = score_col or p_col
    if pd.notna(threshold) and selector_col in frame.columns:
        selected = frame[pd.to_numeric(frame[selector_col], errors="coerce") >= threshold].copy()
    else:
        selected = frame.head(0).copy()
    brier = brier_score(frame[TARGET_COL], frame[p_col])
    base_brier = brier_score(frame[TARGET_COL], pd.Series(base_rate, index=frame.index))
    fixed_width_ece, fixed_width_min_bin_n = ece_score(frame[TARGET_COL], frame[p_col])
    decision_ece_value, decision_min_bin_n, _ = decision_calibration_metrics(frame[TARGET_COL], frame[p_col])
    all_mean = float(frame[RETURN_COL].mean()) if not frame.empty else np.nan
    selected_mean = float(selected[RETURN_COL].mean()) if not selected.empty else np.nan
    if int(bootstrap_iterations) > 0:
        selected_minus_all_ci_lower, uplift_p_value = bootstrap_mean_diff(
            selected[RETURN_COL] if not selected.empty else pd.Series(dtype=float),
            frame[RETURN_COL] if RETURN_COL in frame.columns else pd.Series(dtype=float),
            iterations=bootstrap_iterations,
        )
    else:
        selected_minus_all_ci_lower, uplift_p_value = np.nan, np.nan
    selected_mask = pd.Series(frame.index.isin(selected.index), index=frame.index, dtype=bool)
    score_baseline_mask, score_baseline_policy = score_baseline_mask_with_policy(frame)
    score_baseline, score_baseline_policy = score_baseline_returns_with_policy(frame)
    score_baseline_mean = float(score_baseline.mean()) if not score_baseline.dropna().empty else np.nan
    if int(bootstrap_iterations) > 0:
        selected_minus_score_baseline_ci_lower, score_baseline_p_value = bootstrap_mean_diff(
            selected[RETURN_COL] if not selected.empty else pd.Series(dtype=float),
            score_baseline,
            iterations=bootstrap_iterations,
        )
        paired_uplift = paired_bootstrap_uplift(frame, selected_mask, score_baseline_mask, iterations=bootstrap_iterations)
    else:
        selected_minus_score_baseline_ci_lower, score_baseline_p_value = np.nan, np.nan
        paired_uplift = {
            "bootstrap_method": "skipped_diagnostic",
            "bootstrap_block_col": "",
            "selected_minus_all_ci_lower_pct_paired": np.nan,
            "selected_minus_score_baseline_ci_lower_pct_paired": np.nan,
            "uplift_bootstrap_p_value_paired": np.nan,
            "score_baseline_bootstrap_p_value_paired": np.nan,
        }
    selected_ci_lower = return_ci_lower(selected[RETURN_COL]) if not selected.empty else np.nan
    all_stop_rate = float(1.0 - frame[STOP_SURVIVAL_COL].mean()) if STOP_SURVIVAL_COL in frame.columns and not frame.empty else np.nan
    selected_stop_rate = float(1.0 - selected[STOP_SURVIVAL_COL].mean()) if STOP_SURVIVAL_COL in selected.columns and not selected.empty else np.nan
    ap = safe_average_precision(frame[TARGET_COL], frame[p_col])
    return {
        "model_name": model_name,
        "model_family": model_family(model_name),
        "split": split_name,
        "evaluation_scope": evaluation_scope,
        "event_count": len(frame),
        "symbol_count": int(frame["symbol"].nunique()) if "symbol" in frame.columns and not frame.empty else 0,
        "success_rate": float(frame[TARGET_COL].mean()) if not frame.empty else np.nan,
        "mean_return_pct": all_mean,
        "brier_score": brier,
        "base_rate": base_rate,
        "base_rate_brier_score": base_brier,
        "brier_improvement_pct": float((base_brier - brier) / base_brier * 100.0) if pd.notna(base_brier) and base_brier > 0 and pd.notna(brier) else np.nan,
        "average_precision": ap,
        "base_rate_average_precision": float(frame[TARGET_COL].mean()) if not frame.empty else np.nan,
        "ece": decision_ece_value,
        "decision_ece": decision_ece_value,
        "decision_min_calibration_bin_n": decision_min_bin_n,
        "calibration_binning_primary": calibration_binning_primary_core(),
        "fixed_width_ece": fixed_width_ece,
        "fixed_width_min_calibration_bin_n": fixed_width_min_bin_n,
        "min_calibration_bin_n": decision_min_bin_n,
        "threshold": threshold,
        "selection_score_col": selector_col,
        "probability_col": p_col,
        "trial_count": 1,
        "selected_event_count": len(selected),
        "selected_fraction": float(len(selected) / len(frame)) if len(frame) else np.nan,
        "selected_success_rate": float(selected[TARGET_COL].mean()) if not selected.empty else np.nan,
        "selected_mean_return_pct": selected_mean,
        "selected_minus_all_pct": float(selected_mean - all_mean) if pd.notna(selected_mean) and pd.notna(all_mean) else np.nan,
        "selected_minus_all_ci_lower_pct": selected_minus_all_ci_lower,
        "selected_minus_all_ci_lower_pct_independent": selected_minus_all_ci_lower,
        "selected_minus_all_ci_lower_pct_paired": paired_uplift["selected_minus_all_ci_lower_pct_paired"],
        "score_baseline_mean_return_pct": score_baseline_mean,
        "score_baseline_policy": score_baseline_policy,
        "selected_minus_score_baseline_pct": float(selected_mean - score_baseline_mean) if pd.notna(selected_mean) and pd.notna(score_baseline_mean) else np.nan,
        "selected_minus_score_baseline_ci_lower_pct": selected_minus_score_baseline_ci_lower,
        "selected_minus_score_baseline_ci_lower_pct_independent": selected_minus_score_baseline_ci_lower,
        "selected_minus_score_baseline_ci_lower_pct_paired": paired_uplift["selected_minus_score_baseline_ci_lower_pct_paired"],
        "uplift_bootstrap_p_value": max(
            safe_float(uplift_p_value, np.nan),
            safe_float(score_baseline_p_value, np.nan),
        )
        if pd.notna(uplift_p_value) and pd.notna(score_baseline_p_value)
        else uplift_p_value,
        "uplift_bootstrap_p_value_independent": max(
            safe_float(uplift_p_value, np.nan),
            safe_float(score_baseline_p_value, np.nan),
        )
        if pd.notna(uplift_p_value) and pd.notna(score_baseline_p_value)
        else uplift_p_value,
        "uplift_bootstrap_p_value_paired": max(
            safe_float(paired_uplift["uplift_bootstrap_p_value_paired"], np.nan),
            safe_float(paired_uplift["score_baseline_bootstrap_p_value_paired"], np.nan),
        )
        if pd.notna(paired_uplift["uplift_bootstrap_p_value_paired"]) and pd.notna(paired_uplift["score_baseline_bootstrap_p_value_paired"])
        else paired_uplift["uplift_bootstrap_p_value_paired"],
        "bootstrap_method": paired_uplift["bootstrap_method"],
        "bootstrap_block_col": paired_uplift["bootstrap_block_col"],
        "selected_expectancy_ci_lower_pct": selected_ci_lower,
        "all_stop_rate": all_stop_rate,
        "selected_stop_rate": selected_stop_rate,
        "stop_rate_improvement": float(all_stop_rate - selected_stop_rate) if pd.notna(all_stop_rate) and pd.notna(selected_stop_rate) else np.nan,
        "positive_expectancy_fold_count": int(
            pd.notna(selected_minus_all_ci_lower)
            and selected_minus_all_ci_lower > MIN_SELECTED_MINUS_ALL_PCT
            and selected_ci_lower > MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT
        )
        if not selected.empty
        else 0,
        "threshold_stability_pass": np.nan,
        "weak_oof_folds": "",
    }


def split_frame_for_diagnostics(predictions: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if split_name == "combined_test_holdout":
        return predictions[predictions["split"].isin(["test_2024", "final_holdout_2025_2026"])].copy()
    return predictions[predictions["split"].astype(str).eq(split_name)].copy()


def build_slice_diagnostics(
    predictions: pd.DataFrame,
    candidate_meta: dict[str, dict[str, object]],
    train: pd.DataFrame,
    global_success: float,
    champion_name: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    split_names = [
        "train_2016_2022",
        "validation_2023",
        "test_2024",
        "final_holdout_2025_2026",
        "combined_test_holdout",
    ]
    for model_name, meta in candidate_meta.items():
        for evaluation_scope in [ENTRY_RESEARCH_EVAL_SCOPE, TRADE_READY_EVAL_SCOPE]:
            if evaluation_scope == ENTRY_RESEARCH_EVAL_SCOPE:
                p_col = str(meta["p_col"])
                score_col = str(meta["score_col"])
                threshold = safe_float(meta.get("entry_threshold"))
                threshold_reason = str(meta.get("entry_threshold_reason", "UNKNOWN"))
                threshold_eligible = to_bool(meta.get("entry_threshold_decision_eligible", False))
            else:
                p_col = str(meta["trade_p_col"])
                score_col = str(meta["trade_score_col"])
                threshold = safe_float(meta.get("trade_threshold"))
                threshold_reason = str(meta.get("trade_threshold_reason", "UNKNOWN"))
                threshold_eligible = to_bool(meta.get("trade_threshold_decision_eligible", False))
            if p_col not in predictions.columns or score_col not in predictions.columns:
                continue
            base_rate = scoped_base_rate(train, evaluation_scope, global_success)
            for split_name in split_names:
                split_frame = split_frame_for_diagnostics(predictions, split_name)
                if evaluation_scope == TRADE_READY_EVAL_SCOPE:
                    if DECISION_ENTRY_COL not in split_frame.columns:
                        split_frame = split_frame.head(0)
                    else:
                        split_frame = split_frame[split_frame[DECISION_ENTRY_COL].map(to_bool)].copy()
                if split_frame.empty:
                    continue
                split_row = metric_row(
                    split_name,
                    split_frame,
                    p_col,
                    threshold,
                    base_rate,
                    model_name,
                    score_col=score_col,
                    evaluation_scope=evaluation_scope,
                    bootstrap_iterations=0,
                )
                split_row.update(
                    {
                        "slice_dimension": "split",
                        "slice_value": split_name,
                        "threshold_reason": threshold_reason,
                        "threshold_decision_eligible": threshold_eligible,
                        "raw_probability_col": meta.get("raw_col", ""),
                        "calibration_method": meta.get("calibration_method", ""),
                        "is_champion": bool(champion_name and str(model_name) == str(champion_name)),
                        "selected_ci_lower_pct": split_row.get("selected_expectancy_ci_lower_pct", np.nan),
                    }
                )
                rows.append(split_row)
                for dimension in SLICE_DIMENSIONS:
                    if dimension not in split_frame.columns:
                        continue
                    for value, group in split_frame.groupby(split_frame[dimension].fillna("MISSING").astype(str), dropna=False):
                        if group.empty:
                            continue
                        row = metric_row(
                            split_name,
                            group,
                            p_col,
                            threshold,
                            base_rate,
                            model_name,
                            score_col=score_col,
                            evaluation_scope=evaluation_scope,
                            bootstrap_iterations=0,
                        )
                        row.update(
                            {
                                "slice_dimension": dimension,
                                "slice_value": str(value),
                                "threshold_reason": threshold_reason,
                                "threshold_decision_eligible": threshold_eligible,
                                "raw_probability_col": meta.get("raw_col", ""),
                                "calibration_method": meta.get("calibration_method", ""),
                                "is_champion": bool(champion_name and str(model_name) == str(champion_name)),
                                "selected_ci_lower_pct": row.get("selected_expectancy_ci_lower_pct", np.nan),
                            }
                        )
                        rows.append(row)
    return pd.DataFrame(rows)


def fold_date_window(data: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return data[(data["date"] >= start_ts) & (data["date"] <= end_ts)].copy()


def fold_frames(data: pd.DataFrame, fold: WalkForwardFold) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibration_start = pd.Timestamp(fold.calibration_start)
    test_start = pd.Timestamp(fold.test_start) + pd.offsets.BDay(EMBARGO_TRADING_DAYS)
    train_cutoff = min(pd.Timestamp(fold.train_end), calibration_start - pd.offsets.BDay(EMBARGO_TRADING_DAYS))
    train = data[(data["date"] >= pd.Timestamp(fold.train_start)) & (data["date"] <= train_cutoff)].copy()
    calibration = fold_date_window(data, fold.calibration_start, fold.calibration_end)
    threshold = fold_date_window(data, fold.threshold_start, fold.threshold_end)
    test = data[(data["date"] >= test_start) & (data["date"] <= pd.Timestamp(fold.test_end))].copy()
    return train, calibration, threshold, test


def fit_oof_candidates(train: pd.DataFrame, calibration: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, object]]:
    fitted: dict[str, dict[str, object]] = {}
    if SKLEARN_IMPORT_ERROR is not None or not feature_cols or calibration[TARGET_COL].nunique() < 2:
        return fitted
    fitters = [
        ("pooled_elastic_net_logistic", lambda: fit_pooled_elastic_net_candidate(train, feature_cols)),
        ("pooled_weighted_elastic_net_logistic", lambda: fit_pooled_weighted_elastic_net_candidate(train, feature_cols)),
        ("pooled_strict_elastic_net_logistic", lambda: fit_strict_elastic_net_candidate(train, feature_cols)),
        ("pooled_hist_gradient_boosting", lambda: fit_pooled_hist_gbm_candidate(train, feature_cols)),
    ]
    if LGBMClassifier is not None:
        fitters.append(("pooled_lgbm_classifier", lambda: fit_pooled_lgbm_success_candidate(train, calibration, feature_cols)))
    for _, fit_fn in fitters:
        try:
            candidate = fit_fn()
        except Exception:
            candidate = None
        if candidate is not None:
            fitted[str(candidate["name"])] = candidate
    return fitted


def add_raw_candidate_predictions(predictions: pd.DataFrame, fitted_candidates: dict[str, dict[str, object]]) -> pd.DataFrame:
    out = predictions.copy()
    candidate_cols = [
        ("pooled_elastic_net_logistic", "p_success_logistic"),
        ("pooled_weighted_elastic_net_logistic", "p_success_weighted_logistic"),
        ("pooled_strict_elastic_net_logistic", "p_success_strict_logistic"),
        ("pooled_hist_gradient_boosting", "p_success_hist_gbm"),
        ("pooled_lgbm_classifier", "p_success_lgbm"),
        ("pooled_xgb_classifier", "p_success_xgb"),
    ]
    for candidate_name, col_name in candidate_cols:
        if candidate_name in fitted_candidates:
            out[col_name] = predict_pooled_candidate(fitted_candidates[candidate_name], out)
    return out


def oof_metric_row(
    split_name: str,
    rows: pd.DataFrame,
    model_name: str,
    evaluation_scope: str,
    validation_design: str = "walk_forward_oof",
    bootstrap_iterations: int = UPLIFT_BOOTSTRAP_ITERATIONS,
) -> dict[str, object]:
    frame = rows.dropna(subset=[TARGET_COL, RETURN_COL, "p_success"]).copy()
    selected = frame[frame["selected_by_threshold"].map(to_bool)].copy() if "selected_by_threshold" in frame.columns else frame.head(0).copy()
    brier = brier_score(frame[TARGET_COL], frame["p_success"])
    base_probs = pd.to_numeric(frame.get("base_rate", pd.Series(np.nan, index=frame.index)), errors="coerce")
    if base_probs.notna().any():
        base_brier = brier_score(frame[TARGET_COL], base_probs)
        base_rate = float(base_probs.mean())
    else:
        base_rate = float(frame[TARGET_COL].mean()) if not frame.empty else np.nan
        base_brier = brier_score(frame[TARGET_COL], pd.Series(base_rate, index=frame.index))
    fixed_width_ece, fixed_width_min_bin_n = ece_score(frame[TARGET_COL], frame["p_success"])
    decision_ece_value, decision_min_bin_n, _ = decision_calibration_metrics(frame[TARGET_COL], frame["p_success"])
    all_mean = float(frame[RETURN_COL].mean()) if not frame.empty else np.nan
    selected_mean = float(selected[RETURN_COL].mean()) if not selected.empty else np.nan
    selected_minus_all_ci_lower, uplift_p_value = bootstrap_mean_diff(
        selected[RETURN_COL] if not selected.empty else pd.Series(dtype=float),
        frame[RETURN_COL] if RETURN_COL in frame.columns else pd.Series(dtype=float),
        iterations=bootstrap_iterations,
    )
    selected_mask = pd.Series(frame.index.isin(selected.index), index=frame.index, dtype=bool)
    score_baseline_mask, score_baseline_policy = score_baseline_mask_with_policy(frame)
    score_baseline, score_baseline_policy = score_baseline_returns_with_policy(frame)
    score_baseline_mean = float(score_baseline.mean()) if not score_baseline.dropna().empty else np.nan
    selected_minus_score_baseline_ci_lower, score_baseline_p_value = bootstrap_mean_diff(
        selected[RETURN_COL] if not selected.empty else pd.Series(dtype=float),
        score_baseline,
        iterations=bootstrap_iterations,
    )
    paired_uplift = paired_bootstrap_uplift(frame, selected_mask, score_baseline_mask, iterations=bootstrap_iterations)
    thresholds = pd.to_numeric(frame.get("threshold", pd.Series(np.nan, index=frame.index)), errors="coerce").dropna()
    selected_ci_lower = return_ci_lower(selected[RETURN_COL]) if not selected.empty else np.nan
    all_stop_rate = float(1.0 - frame[STOP_SURVIVAL_COL].mean()) if STOP_SURVIVAL_COL in frame.columns and not frame.empty else np.nan
    selected_stop_rate = float(1.0 - selected[STOP_SURVIVAL_COL].mean()) if STOP_SURVIVAL_COL in selected.columns and not selected.empty else np.nan
    trial_count = int(pd.to_numeric(frame.get("trial_count", pd.Series([1])), errors="coerce").fillna(1).max()) if not frame.empty else 0
    applied_policy_values = split_unique_values(frame, "applied_threshold_policy_type")
    legacy_policy_values = split_unique_values(frame, "threshold_policy_type")
    applied_policy = applied_policy_values or legacy_policy_values
    if "|" in applied_policy:
        applied_policy = "mixed_locked_policy"
    diagnostic_policy = split_unique_values(frame, "diagnostic_best_candidate_policy_type")
    return {
        "model_name": model_name,
        "model_family": model_family(model_name),
        "split": split_name,
        "evaluation_scope": evaluation_scope,
        "validation_design": validation_design,
        "event_count": len(frame),
        "symbol_count": int(frame["symbol"].nunique()) if "symbol" in frame.columns and not frame.empty else 0,
        "success_rate": float(frame[TARGET_COL].mean()) if not frame.empty else np.nan,
        "mean_return_pct": all_mean,
        "brier_score": brier,
        "base_rate": base_rate,
        "base_rate_brier_score": base_brier,
        "brier_improvement_pct": float((base_brier - brier) / base_brier * 100.0) if pd.notna(base_brier) and base_brier > 0 and pd.notna(brier) else np.nan,
        "average_precision": safe_average_precision(frame[TARGET_COL], frame["p_success"]),
        "base_rate_average_precision": float(frame[TARGET_COL].mean()) if not frame.empty else np.nan,
        "ece": decision_ece_value,
        "decision_ece": decision_ece_value,
        "decision_min_calibration_bin_n": decision_min_bin_n,
        "calibration_binning_primary": calibration_binning_primary_core(),
        "fixed_width_ece": fixed_width_ece,
        "fixed_width_min_calibration_bin_n": fixed_width_min_bin_n,
        "min_calibration_bin_n": decision_min_bin_n,
        "threshold": float(thresholds.median()) if not thresholds.empty else np.nan,
        "selection_score_col": "utility_score",
        "probability_col": "p_success",
        "trial_count": trial_count,
        "selected_event_count": len(selected),
        "selected_fraction": float(len(selected) / len(frame)) if len(frame) else np.nan,
        "selected_success_rate": float(selected[TARGET_COL].mean()) if not selected.empty else np.nan,
        "selected_mean_return_pct": selected_mean,
        "selected_minus_all_pct": float(selected_mean - all_mean) if pd.notna(selected_mean) and pd.notna(all_mean) else np.nan,
        "selected_minus_all_ci_lower_pct": selected_minus_all_ci_lower,
        "selected_minus_all_ci_lower_pct_independent": selected_minus_all_ci_lower,
        "selected_minus_all_ci_lower_pct_paired": paired_uplift["selected_minus_all_ci_lower_pct_paired"],
        "score_baseline_mean_return_pct": score_baseline_mean,
        "score_baseline_policy": score_baseline_policy,
        "selected_minus_score_baseline_pct": float(selected_mean - score_baseline_mean) if pd.notna(selected_mean) and pd.notna(score_baseline_mean) else np.nan,
        "selected_minus_score_baseline_ci_lower_pct": selected_minus_score_baseline_ci_lower,
        "selected_minus_score_baseline_ci_lower_pct_independent": selected_minus_score_baseline_ci_lower,
        "selected_minus_score_baseline_ci_lower_pct_paired": paired_uplift["selected_minus_score_baseline_ci_lower_pct_paired"],
        "uplift_bootstrap_p_value": max(
            safe_float(uplift_p_value, np.nan),
            safe_float(score_baseline_p_value, np.nan),
        )
        if pd.notna(uplift_p_value) and pd.notna(score_baseline_p_value)
        else uplift_p_value,
        "uplift_bootstrap_p_value_independent": max(
            safe_float(uplift_p_value, np.nan),
            safe_float(score_baseline_p_value, np.nan),
        )
        if pd.notna(uplift_p_value) and pd.notna(score_baseline_p_value)
        else uplift_p_value,
        "uplift_bootstrap_p_value_paired": max(
            safe_float(paired_uplift["uplift_bootstrap_p_value_paired"], np.nan),
            safe_float(paired_uplift["score_baseline_bootstrap_p_value_paired"], np.nan),
        )
        if pd.notna(paired_uplift["uplift_bootstrap_p_value_paired"]) and pd.notna(paired_uplift["score_baseline_bootstrap_p_value_paired"])
        else paired_uplift["uplift_bootstrap_p_value_paired"],
        "bootstrap_method": paired_uplift["bootstrap_method"],
        "bootstrap_block_col": paired_uplift["bootstrap_block_col"],
        "selected_expectancy_ci_lower_pct": selected_ci_lower,
        "all_stop_rate": all_stop_rate,
        "selected_stop_rate": selected_stop_rate,
        "stop_rate_improvement": float(all_stop_rate - selected_stop_rate) if pd.notna(all_stop_rate) and pd.notna(selected_stop_rate) else np.nan,
        "positive_expectancy_fold_count": int(
            pd.notna(selected_ci_lower)
            and selected_ci_lower > MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT
            and pd.notna(selected_minus_all_ci_lower)
            and selected_minus_all_ci_lower > MIN_SELECTED_MINUS_ALL_PCT
        ),
        "threshold_reason": "|".join(sorted(set(frame["threshold_reason"].dropna().astype(str)))) if "threshold_reason" in frame.columns and not frame.empty else "",
        "threshold_decision_eligible": bool(frame["threshold_decision_eligible"].map(to_bool).all()) if "threshold_decision_eligible" in frame.columns and not frame.empty else False,
        "threshold_stability_pass": np.nan,
        "threshold_policy_type": applied_policy,
        "applied_threshold_policy_type": applied_policy,
        "diagnostic_best_candidate_policy_type": diagnostic_policy,
        "threshold_policy_source_window": split_unique_values(frame, "threshold_policy_source_window"),
        "threshold_policy_applied_window": split_unique_values(frame, "threshold_policy_applied_window"),
        "threshold_policy_provenance_valid": bool(frame["threshold_policy_provenance_valid"].map(to_bool).all()) if "threshold_policy_provenance_valid" in frame.columns and not frame.empty else False,
        "risk_adjusted_selection_score_col": split_unique_values(frame, "risk_adjusted_selection_score_col"),
        "stable_threshold_candidate_count": int(pd.to_numeric(frame.get("stable_threshold_candidate_count", pd.Series([0])), errors="coerce").fillna(0).max()) if not frame.empty else 0,
        "threshold_stability_failure_reasons": split_unique_values(frame, "threshold_stability_failure_reasons"),
        "weak_oof_folds": "",
        "calibration_method": "|".join(sorted(set(frame["calibration_method"].dropna().astype(str)))) if "calibration_method" in frame.columns and not frame.empty else "",
        "raw_probability_col": "|".join(sorted(set(frame["raw_probability_col"].dropna().astype(str)))) if "raw_probability_col" in frame.columns and not frame.empty else "",
        "utility_weight_label": "|".join(sorted(set(frame["utility_weight_label"].dropna().astype(str)))) if "utility_weight_label" in frame.columns and not frame.empty else "",
    }


def build_oof_metrics(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    if oof_predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (model_name, evaluation_scope, split_name), group in oof_predictions.groupby(["model_name", "evaluation_scope", "split"], dropna=False):
        rows.append(oof_metric_row(str(split_name), group, str(model_name), str(evaluation_scope)))
    for (model_name, evaluation_scope), group in oof_predictions.groupby(["model_name", "evaluation_scope"], dropna=False):
        rows.append(oof_metric_row("combined_test_holdout", group, str(model_name), str(evaluation_scope)))
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    metrics["fold_count"] = np.nan
    metrics["min_selected_events_per_fold"] = np.nan
    metrics["weak_selected_event_folds"] = ""
    metrics["weak_oof_folds"] = ""
    metrics["threshold_stability_pass"] = pd.Series([np.nan] * len(metrics), index=metrics.index, dtype=object)
    metrics["fold_selected_fraction_min"] = np.nan
    metrics["fold_selected_fraction_max"] = np.nan
    metrics["fold_selected_count_min"] = np.nan
    metrics["fold_stop_improvement_min"] = np.nan
    metrics["threshold_stability_failure_reasons"] = ""
    metrics["fold_uplift_ci_lower_min"] = np.nan
    metrics["fold_score_baseline_ci_lower_min"] = np.nan
    metrics["uplift_pass"] = pd.Series([np.nan] * len(metrics), index=metrics.index, dtype=object)
    metrics["uplift_failure_reasons"] = ""
    metrics["uplift_failure_reasons_detail"] = ""
    metrics["fold_positive_uplift_count"] = np.nan
    metrics["fold_positive_score_baseline_uplift_count"] = np.nan
    metrics["positive_expectancy_fold_count"] = np.nan
    fold_mask = metrics["split"].astype(str).str.startswith("oof_test_")
    combined_mask = metrics["split"].astype(str).eq("combined_test_holdout")
    for idx, combined_row in metrics[combined_mask].iterrows():
        fold_rows = metrics[
            fold_mask
            & metrics["model_name"].astype(str).eq(str(combined_row["model_name"]))
            & metrics["evaluation_scope"].astype(str).eq(str(combined_row["evaluation_scope"]))
        ].copy()
        fold_rows = fold_rows[pd.to_numeric(fold_rows["event_count"], errors="coerce") > 0]
        if fold_rows.empty:
            continue
        selected_counts = pd.to_numeric(fold_rows["selected_event_count"], errors="coerce")
        threshold_stability_pass, weak_folds, threshold_iqr = threshold_stability_from_fold_metrics(fold_rows)
        selected_fractions = pd.to_numeric(fold_rows["selected_fraction"], errors="coerce")
        fold_stop_improvement = pd.to_numeric(fold_rows.get("stop_rate_improvement", pd.Series(dtype=float)), errors="coerce")
        fold_uplift_lower = pd.to_numeric(
            fold_rows.get("selected_minus_all_ci_lower_pct_paired", fold_rows.get("selected_minus_all_ci_lower_pct", pd.Series(dtype=float))),
            errors="coerce",
        )
        fold_score_lower = pd.to_numeric(
            fold_rows.get("selected_minus_score_baseline_ci_lower_pct_paired", fold_rows.get("selected_minus_score_baseline_ci_lower_pct", pd.Series(dtype=float))),
            errors="coerce",
        )
        positive_fold_mask = (
            (pd.to_numeric(fold_rows["selected_expectancy_ci_lower_pct"], errors="coerce") > MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT)
            & (fold_uplift_lower > MIN_SELECTED_MINUS_ALL_PCT)
            & (fold_score_lower > MIN_SELECTED_MINUS_ALL_PCT)
        )
        positive_uplift_mask = fold_uplift_lower > MIN_SELECTED_MINUS_ALL_PCT
        positive_score_mask = fold_score_lower > MIN_SELECTED_MINUS_ALL_PCT
        combined_selected_fraction = safe_float(combined_row.get("selected_fraction"))
        fold_stop_min = float(fold_stop_improvement.min()) if fold_stop_improvement.notna().any() else np.nan
        fold_uplift_min = float(fold_uplift_lower.min()) if fold_uplift_lower.notna().any() else np.nan
        fold_score_min = float(fold_score_lower.min()) if fold_score_lower.notna().any() else np.nan
        uplift_failure_reasons = []
        combined_uplift_lower = safe_float(combined_row.get("selected_minus_all_ci_lower_pct_paired", combined_row.get("selected_minus_all_ci_lower_pct")))
        combined_score_lower = safe_float(combined_row.get("selected_minus_score_baseline_ci_lower_pct_paired", combined_row.get("selected_minus_score_baseline_ci_lower_pct")))
        if combined_uplift_lower <= MIN_SELECTED_MINUS_ALL_PCT:
            uplift_failure_reasons.append("COMBINED_SELECTED_MINUS_ALL_CI_LOWER_LE_0")
        if combined_score_lower <= MIN_SELECTED_MINUS_ALL_PCT:
            uplift_failure_reasons.append("COMBINED_SELECTED_MINUS_SCORE_BASELINE_CI_LOWER_LE_0")
        if int(positive_fold_mask.sum()) < 4:
            uplift_failure_reasons.append("POSITIVE_EXPECTANCY_FOLDS_LT_4")
        uplift_pass = not uplift_failure_reasons
        combined_threshold_pass = (
            threshold_stability_pass
            and safe_float(combined_row.get("selected_event_count")) >= MIN_SELECTED_EVAL_EVENTS
            and MIN_SELECTED_FRACTION <= combined_selected_fraction <= MAX_SELECTED_FRACTION
            and combined_uplift_lower > MIN_SELECTED_MINUS_ALL_PCT
            and fold_stop_min >= MIN_STOP_RATE_IMPROVEMENT
            and int(safe_float(combined_row.get("trial_count"), 0)) > 0
        )
        existing_reason = str(combined_row.get("threshold_reason", ""))
        threshold_failure_reasons = str(combined_row.get("threshold_stability_failure_reasons", ""))
        if not combined_threshold_pass:
            reason_parts = [part for part in existing_reason.split("|") if part]
            reason_parts.append("OOF_FOLD_SELECTION_UNSTABLE" if not threshold_stability_pass else "STRICT_SCOPE_INSUFFICIENT_FOR_DECISION")
            metrics.loc[idx, "threshold_reason"] = "|".join(sorted(set(reason_parts)))
            if not threshold_failure_reasons:
                threshold_failure_reasons = "OOF_FOLD_SELECTION_UNSTABLE" if not threshold_stability_pass else "STRICT_SCOPE_INSUFFICIENT_FOR_DECISION"
        elif "DIAGNOSTIC_BROAD_SCOPE_FALLBACK" in existing_reason or "STRICT_SCOPE_INSUFFICIENT_FOR_DECISION" in existing_reason:
            metrics.loc[idx, "threshold_reason"] = "VALIDATION_ECONOMIC_UPLIFT_AND_RISK_FILTER"
        metrics.loc[idx, "fold_count"] = int(len(fold_rows))
        metrics.loc[idx, "min_selected_events_per_fold"] = float(selected_counts.min()) if selected_counts.notna().any() else np.nan
        metrics.loc[idx, "fold_selected_count_min"] = float(selected_counts.min()) if selected_counts.notna().any() else np.nan
        metrics.loc[idx, "fold_selected_fraction_min"] = float(selected_fractions.min()) if selected_fractions.notna().any() else np.nan
        metrics.loc[idx, "fold_selected_fraction_max"] = float(selected_fractions.max()) if selected_fractions.notna().any() else np.nan
        metrics.loc[idx, "fold_stop_improvement_min"] = fold_stop_min
        metrics.loc[idx, "weak_selected_event_folds"] = weak_folds
        metrics.loc[idx, "weak_oof_folds"] = weak_folds
        metrics.loc[idx, "threshold_stability_pass"] = bool(combined_threshold_pass)
        metrics.loc[idx, "threshold_decision_eligible"] = bool(combined_threshold_pass)
        metrics.loc[idx, "stable_threshold_candidate_count"] = int(safe_float(combined_row.get("stable_threshold_candidate_count"), 0)) if combined_threshold_pass else 0
        metrics.loc[idx, "threshold_iqr"] = threshold_iqr
        metrics.loc[idx, "positive_expectancy_fold_count"] = int(positive_fold_mask.sum())
        metrics.loc[idx, "threshold_stability_failure_reasons"] = threshold_failure_reasons if threshold_failure_reasons else "PASS"
        metrics.loc[idx, "fold_uplift_ci_lower_min"] = fold_uplift_min
        metrics.loc[idx, "fold_score_baseline_ci_lower_min"] = fold_score_min
        metrics.loc[idx, "uplift_pass"] = bool(uplift_pass)
        metrics.loc[idx, "uplift_failure_reasons"] = "|".join(uplift_failure_reasons) if uplift_failure_reasons else "PASS"
        metrics.loc[idx, "uplift_failure_reasons_detail"] = (
            f"combined_all_paired={combined_uplift_lower:.6f};combined_score_paired={combined_score_lower:.6f};"
            f"fold_all_min={fold_uplift_min:.6f};fold_score_min={fold_score_min:.6f};positive_folds={int(positive_fold_mask.sum())}"
        )
        metrics.loc[idx, "fold_positive_uplift_count"] = int(positive_uplift_mask.sum())
        metrics.loc[idx, "fold_positive_score_baseline_uplift_count"] = int(positive_score_mask.sum())
    return metrics


def build_uplift_bootstrap_report(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    cols = [
        "model_name",
        "model_family",
        "validation_design",
        "split",
        "evaluation_scope",
        "event_count",
        "selected_event_count",
        "selected_fraction",
        "selected_mean_return_pct",
        "mean_return_pct",
        "selected_minus_all_pct",
        "selected_minus_all_ci_lower_pct",
        "selected_minus_all_ci_lower_pct_independent",
        "selected_minus_all_ci_lower_pct_paired",
        "score_baseline_mean_return_pct",
        "selected_minus_score_baseline_pct",
        "selected_minus_score_baseline_ci_lower_pct",
        "selected_minus_score_baseline_ci_lower_pct_independent",
        "selected_minus_score_baseline_ci_lower_pct_paired",
        "uplift_bootstrap_p_value",
        "uplift_bootstrap_p_value_independent",
        "uplift_bootstrap_p_value_paired",
        "bootstrap_method",
        "bootstrap_block_col",
        "positive_expectancy_fold_count",
        "fold_positive_uplift_count",
        "fold_positive_score_baseline_uplift_count",
        "fold_uplift_ci_lower_min",
        "fold_score_baseline_ci_lower_min",
        "uplift_pass",
        "uplift_failure_reasons",
        "uplift_failure_reasons_detail",
        "score_baseline_policy",
        "trial_count",
        "is_champion",
    ]
    out = comparison[[c for c in cols if c in comparison.columns]].copy()
    if "validation_design" not in out.columns:
        out["validation_design"] = ""
    out["bootstrap_iterations"] = UPLIFT_BOOTSTRAP_ITERATIONS
    return out


def build_learning_curve_report(data: pd.DataFrame, comparison: pd.DataFrame, tsm_like_metrics: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "symbol" not in data.columns:
        return pd.DataFrame()
    counts = data.groupby("symbol", dropna=False).size().sort_values(ascending=False)
    ordered_symbols = [str(symbol) for symbol in counts.index]
    actual_n = len(ordered_symbols)
    decision_metrics = decision_scope_metrics(comparison)
    champion = decision_metrics[
        decision_metrics.get("is_champion", pd.Series(False, index=decision_metrics.index)).map(to_bool)
        & decision_metrics["split"].astype(str).eq("combined_test_holdout")
    ].copy()
    if champion.empty:
        champion = decision_metrics[decision_metrics["split"].astype(str).eq("combined_test_holdout")].copy()
    champion_row = champion.iloc[0] if not champion.empty else pd.Series(dtype=object)
    tsm_like_row = selected_tsm_like_selection_row(tsm_like_metrics)
    rows: list[dict[str, object]] = []
    for target in [12, 25, 40, actual_n]:
        target_n = min(int(target), actual_n)
        symbols = ordered_symbols[:target_n]
        subset = data[data["symbol"].astype(str).isin(symbols)].copy()
        trade_ready = int(subset[DECISION_ENTRY_COL].map(to_bool).sum()) if DECISION_ENTRY_COL in subset.columns else 0
        model_training = int(subset[MODEL_TRAINING_COL].map(to_bool).sum()) if MODEL_TRAINING_COL in subset.columns else len(subset)
        is_full = target_n == actual_n
        rows.append(
            {
                "universe_size_target": "all_eligible_symbols" if is_full else f"{target_n}_symbols",
                "symbol_count": target_n,
                "model_training_20d_labeled": model_training,
                "trade_ready_20d_labeled": trade_ready,
                "brier_improvement_pct": champion_row.get("brier_improvement_pct", np.nan) if is_full else np.nan,
                "ece": champion_row.get("decision_ece", champion_row.get("ece", np.nan)) if is_full else np.nan,
                "selected_minus_rule_mean": champion_row.get("selected_minus_all_pct", np.nan) if is_full else np.nan,
                "selected_minus_rule_paired_ci_lower": champion_row.get("selected_minus_all_ci_lower_pct_paired", champion_row.get("selected_minus_all_ci_lower_pct", np.nan)) if is_full else np.nan,
                "tsm_like_calibration_ece": tsm_like_row.get("selection_decision_ece", tsm_like_row.get("decision_ece", np.nan)) if is_full else np.nan,
                "selected_fraction_drift": champion_row.get("selection_fraction_drift", np.nan) if is_full else np.nan,
                "fold_positive_uplift_count": champion_row.get("fold_positive_uplift_count", np.nan) if is_full else np.nan,
                "status": "current_full_model_quality" if is_full else "sample_count_only_rerun_required",
            }
        )
    return pd.DataFrame(rows)


def build_tsm_calibration_route_metrics(
    predictions: pd.DataFrame,
    source_col: str,
    threshold: float,
    base_rate: float,
    trade_weights: dict[str, float],
    tsm_layer: TsmCalibrationLayer,
) -> tuple[pd.DataFrame, pd.DataFrame, str, TsmCalibrationRouteSpec]:
    if predictions.empty or source_col not in predictions.columns:
        spec = TsmCalibrationRouteSpec("POOLED_ONLY", "p_success_tsm_route_pooled_only", "decision_score_tsm_route_pooled_only")
        return pd.DataFrame(), pd.DataFrame(), "POOLED_ONLY", spec
    out = predictions.copy()
    tsm_mask = out["symbol"].astype(str).eq("TSM") if "symbol" in out.columns else pd.Series(False, index=out.index)
    shrunk_layer = shrunk_tsm_layer(tsm_layer)
    decision_mask = out[DECISION_ENTRY_COL].map(to_bool) if DECISION_ENTRY_COL in out.columns else pd.Series(True, index=out.index)
    selection_rows = out[tsm_mask & decision_mask & out["split"].isin(["train_2016_2022", "validation_2023"])].copy()
    all_selection_context = out[decision_mask & out["split"].isin(["train_2016_2022", "validation_2023"])].copy()
    route_specs = build_tsm_route_specs(selection_rows, all_selection_context, source_col, base_rate, tsm_layer, shrunk_layer)
    route_rows: list[dict[str, object]] = []
    for spec in route_specs:
        out[spec.p_col] = np.nan
        out[spec.score_col] = np.nan
        out.loc[tsm_mask, spec.p_col] = apply_tsm_calibration_route_spec(out.loc[tsm_mask], spec, source_col=source_col)
        out.loc[tsm_mask, spec.score_col] = utility_score_frame(
            out.loc[tsm_mask],
            spec.p_col,
            "p_stop_hit_lgbm",
            "expected_r_lgbm",
            trade_weights,
        )
        route_model_name = f"tsm_specific_{spec.route.lower()}"
        for split_name, frame in [
            ("tsm_train_validation", out[tsm_mask & decision_mask & out["split"].isin(["train_2016_2022", "validation_2023"])]),
            ("tsm_test_2024", out[tsm_mask & decision_mask & out["split"].eq("test_2024")]),
            ("tsm_final_holdout_2025_2026", out[tsm_mask & decision_mask & out["split"].eq("final_holdout_2025_2026")]),
            ("tsm_combined_test_holdout", out[tsm_mask & decision_mask & out["split"].isin(["test_2024", "final_holdout_2025_2026"])]),
        ]:
            row = metric_row(
                split_name,
                frame,
                spec.p_col,
                threshold,
                base_rate,
                route_model_name,
                score_col=spec.score_col,
                evaluation_scope=TRADE_READY_EVAL_SCOPE,
            )
            row["tsm_calibration_route"] = spec.route
            row["tsm_calibration_route_v2"] = spec.route
            row["route_selection_window"] = "train_2016_2022_plus_validation_2023"
            row["route_evaluation_window"] = "test_2024_plus_final_holdout_2025_2026"
            row["route_selection_provenance_valid"] = True
            row["route_alpha"] = spec.route_alpha if spec.route_alpha is not None else np.nan
            row["route_prior_source"] = spec.route_prior_source
            row["route_sample_weight_policy"] = spec.route_sample_weight_policy
            route_rows.append(row)
    metrics = pd.DataFrame(route_rows)
    selected_route, route_summary = choose_tsm_calibration_route_v2(metrics)
    if route_summary.empty:
        selected_spec = next((spec for spec in route_specs if spec.route == selected_route), route_specs[0])
        return metrics, route_summary, selected_route, selected_spec
    summary_by_route = route_summary.set_index("tsm_calibration_route")
    ece_by_route = route_summary.set_index("tsm_calibration_route")["combined_decision_ece"].to_dict()
    metrics["tsm_calibration_route_pass"] = metrics["tsm_calibration_route"].map(
        route_summary.set_index("tsm_calibration_route")["tsm_calibration_route_pass"].to_dict()
    )
    metrics["tsm_calibration_route_failure_reasons"] = metrics["tsm_calibration_route"].map(
        route_summary.set_index("tsm_calibration_route")["tsm_calibration_route_failure_reasons"].to_dict()
    )
    metrics["is_selected_tsm_calibration_route"] = metrics["tsm_calibration_route"].astype(str).eq(selected_route)
    metrics["route_selection_pass"] = metrics["tsm_calibration_route"].map(route_summary.set_index("tsm_calibration_route")["route_selection_pass"].to_dict())
    metrics["route_selection_failure_reasons"] = metrics["tsm_calibration_route"].map(route_summary.set_index("tsm_calibration_route")["route_selection_failure_reasons"].to_dict())
    metrics["route_fail_count"] = metrics["tsm_calibration_route"].map(route_summary.set_index("tsm_calibration_route")["route_fail_count"].to_dict())
    metrics["selected_route_is_simplest_close_candidate"] = metrics["tsm_calibration_route"].map(route_summary.set_index("tsm_calibration_route")["selected_route_is_simplest_close_candidate"].to_dict())
    for col in [
        "selection_decision_ece",
        "selection_brier_improvement_pct",
        "combined_decision_ece",
        "combined_brier_improvement_pct",
        "max_test_holdout_ece",
        "route_selection_fail_count",
    ]:
        if col in route_summary.columns:
            metrics[col] = metrics["tsm_calibration_route"].map(route_summary.set_index("tsm_calibration_route")[col].to_dict())
    metrics["pooled_only_tsm_ece"] = ece_by_route.get("POOLED_ONLY", np.nan)
    metrics["tsm_logit_shift_ece"] = ece_by_route.get("TSM_STATIC_LOGIT_SHIFT", ece_by_route.get("TSM_LOGIT_SHIFT", np.nan))
    metrics["tsm_shrunk_logit_shift_ece"] = ece_by_route.get("TSM_SHRUNK_LOGIT_SHIFT", np.nan)
    metrics["selected_tsm_calibration_route"] = selected_route
    metrics["selected_tsm_calibration_route_pass"] = bool(summary_by_route.loc[selected_route, "tsm_calibration_route_pass"]) if selected_route in summary_by_route.index else False
    metrics["selected_tsm_calibration_route_failure_reasons"] = (
        str(summary_by_route.loc[selected_route, "tsm_calibration_route_failure_reasons"]) if selected_route in summary_by_route.index else "TSM_ROUTE_SELECTION_MISSING"
    )
    selected_spec = next((spec for spec in route_specs if spec.route == selected_route), route_specs[0])
    return metrics, route_summary, selected_route, selected_spec


def run_walk_forward_oof(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    oof_threshold_policy_parts: list[pd.DataFrame] = []
    for fold in WALK_FORWARD_FOLDS:
        train, calibration, threshold, test = fold_frames(data, fold)
        if train.empty or calibration.empty or threshold.empty or test.empty or train[TARGET_COL].nunique() < 2:
            continue
        eb_model = fit_empirical_bayes(train)
        feature_cols = pooled_feature_columns(train)
        fitted_candidates = fit_oof_candidates(train, calibration, feature_cols)
        fold_frames_to_predict = []
        for purpose, frame in [("calibration", calibration), ("threshold", threshold), ("test", test)]:
            pred = predict_empirical_bayes(eb_model, frame)
            pred["oof_purpose"] = purpose
            fold_frames_to_predict.append(pred)
        predictions = pd.concat(fold_frames_to_predict, ignore_index=True)
        predictions["label_stop_hit_20d"] = 1.0 - pd.to_numeric(predictions[STOP_SURVIVAL_COL], errors="coerce")
        predictions["p_success_eb"] = predictions["p_success_base"]
        predictions["p_stop_hit_eb"] = predictions["p_stop_hit"]
        predictions["expected_r_eb"] = predictions["expected_r_net"]
        predictions["p_stop_hit_lgbm"] = predictions["p_stop_hit_eb"]
        predictions["expected_r_lgbm"] = predictions["expected_r_eb"]
        hier_eb_model = fit_empirical_bayes(train, prior_strength=80.0, group_cols=HIERARCHICAL_GROUP_COLS)
        predictions["p_success_hier_eb"] = predict_empirical_bayes(hier_eb_model, predictions)["p_success_base"]
        predictions = add_raw_candidate_predictions(predictions, fitted_candidates)
        raw_candidate_cols = [
            ("pooled_empirical_bayes_group_rate", "p_success_eb"),
            ("pooled_hierarchical_empirical_bayes", "p_success_hier_eb"),
            ("pooled_elastic_net_logistic", "p_success_logistic"),
            ("pooled_weighted_elastic_net_logistic", "p_success_weighted_logistic"),
            ("pooled_strict_elastic_net_logistic", "p_success_strict_logistic"),
            ("pooled_hist_gradient_boosting", "p_success_hist_gbm"),
            ("pooled_lgbm_classifier", "p_success_lgbm"),
            ("pooled_xgb_classifier", "p_success_xgb"),
        ]
        for model_name, raw_col in raw_candidate_cols:
            if raw_col not in predictions.columns:
                continue
            calibration_rows = predictions[predictions["oof_purpose"].eq("calibration")].copy()
            threshold_rows = predictions[predictions["oof_purpose"].eq("threshold")].copy()
            test_rows = predictions[predictions["oof_purpose"].eq("test")].copy()
            calibrator, calibration_method = fit_probability_calibrator(calibration_rows[raw_col], calibration_rows[TARGET_COL])
            global_p_col = f"{raw_col}_global_calibrated"
            tier_p_col = f"{raw_col}_tier_calibrated"
            entry_scope_p_col = f"{raw_col}_entry_scope_calibrated"
            p_col = f"{raw_col}_calibrated"
            trade_scope_p_col = f"{raw_col}_strict_scope_calibrated"
            trade_p_col = f"{raw_col}_trade_ready_calibrated"
            predictions[global_p_col] = apply_probability_calibrator(predictions[raw_col], calibrator, calibration_method)
            calibration_rows = predictions[predictions["oof_purpose"].eq("calibration")].copy()
            tier_layers = fit_candidate_tier_calibration(calibration_rows, global_p_col, eb_model.global_success)
            predictions[tier_p_col] = apply_candidate_tier_calibration(predictions, global_p_col, tier_layers)
            calibration_rows = predictions[predictions["oof_purpose"].eq("calibration")].copy()
            entry_scope_layer = fit_scope_logit_shift(calibration_rows, tier_p_col, eb_model.global_success, ENTRY_RESEARCH_EVAL_SCOPE)
            predictions[entry_scope_p_col] = apply_logit_shift(predictions[tier_p_col], entry_scope_layer)
            calibration_rows = predictions[predictions["oof_purpose"].eq("calibration")].copy()
            entry_group_layers = fit_symbol_group_calibration(calibration_rows, entry_scope_p_col, eb_model.global_success)
            predictions[p_col] = apply_symbol_group_calibration(predictions, entry_scope_p_col, entry_group_layers)
            calibration_rows = predictions[predictions["oof_purpose"].eq("calibration")].copy()
            calibration_trade_ready = calibration_rows[calibration_rows[DECISION_ENTRY_COL].map(to_bool)].copy()
            strict_base_rate = scoped_base_rate(train, TRADE_READY_EVAL_SCOPE, eb_model.global_success)
            trade_scope_layer = fit_scope_logit_shift(calibration_trade_ready, tier_p_col, strict_base_rate, TRADE_READY_EVAL_SCOPE)
            predictions[trade_scope_p_col] = apply_logit_shift(predictions[tier_p_col], trade_scope_layer)
            calibration_rows = predictions[predictions["oof_purpose"].eq("calibration")].copy()
            calibration_trade_ready = calibration_rows[calibration_rows[DECISION_ENTRY_COL].map(to_bool)].copy()
            trade_group_layers = fit_symbol_group_calibration(calibration_trade_ready, trade_scope_p_col, strict_base_rate)
            predictions[trade_p_col] = apply_symbol_group_calibration(predictions, trade_scope_p_col, trade_group_layers)
            threshold_rows = predictions[predictions["oof_purpose"].eq("threshold")].copy()
            threshold_trade_ready = threshold_rows[threshold_rows[DECISION_ENTRY_COL].map(to_bool)].copy()
            entry_score_col = f"utility_score_{model_name}"
            trade_score_col = f"utility_score_{model_name}_trade_ready"
            entry_policy = choose_utility_threshold(
                threshold_rows,
                p_col=p_col,
                score_col=entry_score_col,
                base_rate=eb_model.global_success,
            )
            entry_weights = entry_policy["weights"]
            predictions[entry_score_col] = utility_score_frame(predictions, p_col, "p_stop_hit_lgbm", "expected_r_lgbm", entry_weights)
            threshold_rows = predictions[predictions["oof_purpose"].eq("threshold")].copy()
            threshold_trade_ready = threshold_rows[threshold_rows[DECISION_ENTRY_COL].map(to_bool)].copy()
            entry_threshold_info = entry_policy["threshold_info"]
            trade_policy = choose_utility_threshold(
                threshold_trade_ready,
                p_col=trade_p_col,
                score_col=trade_score_col,
                base_rate=strict_base_rate,
            )
            trade_weights = trade_policy["weights"]
            predictions[trade_score_col] = utility_score_frame(predictions, trade_p_col, "p_stop_hit_lgbm", "expected_r_lgbm", trade_weights)
            threshold_rows = predictions[predictions["oof_purpose"].eq("threshold")].copy()
            threshold_trade_ready = threshold_rows[threshold_rows[DECISION_ENTRY_COL].map(to_bool)].copy()
            trade_threshold_info = choose_threshold_with_scope_fallback(
                threshold_trade_ready,
                threshold_rows,
                entry_threshold_info,
                score_col=trade_score_col,
                stop_col="label_stop_hit_20d",
                probability_col=trade_p_col,
                base_rate=strict_base_rate,
            )
            trade_threshold_info = attach_utility_weight_columns(trade_threshold_info, trade_weights)
            test_rows = predictions[predictions["oof_purpose"].eq("test")].copy()
            for evaluation_scope, p_source_col, score_source_col, threshold_info, base_rate, eval_test in [
                (ENTRY_RESEARCH_EVAL_SCOPE, p_col, entry_score_col, entry_threshold_info, eb_model.global_success, test_rows),
                (
                    TRADE_READY_EVAL_SCOPE,
                    trade_p_col,
                    trade_score_col,
                    trade_threshold_info,
                    strict_base_rate,
                    test_rows[test_rows[DECISION_ENTRY_COL].map(to_bool)].copy(),
                ),
            ]:
                if eval_test.empty:
                    continue
                if evaluation_scope == TRADE_READY_EVAL_SCOPE:
                    v5_result = choose_fold_consensus_trade_ready_threshold_v5(
                        threshold_trade_ready,
                        eval_test,
                        model_name=model_name,
                        p_col=p_source_col,
                        raw_score_col=score_source_col,
                        base_rate=base_rate,
                        fold_id=fold.fold_id,
                        split_name=f"oof_test_{fold.fold_id.replace('wf_', '')}",
                        threshold_info=threshold_info,
                    )
                    eval_records = v5_result["records"].copy()
                    policy_table = v5_result.get("threshold_table", pd.DataFrame())
                    if isinstance(policy_table, pd.DataFrame) and not policy_table.empty:
                        oof_threshold_policy_parts.append(policy_table)
                else:
                    threshold_value = safe_float(threshold_info.get("threshold"))
                    selected = pd.to_numeric(eval_test[score_source_col], errors="coerce") >= threshold_value if pd.notna(threshold_value) else pd.Series(False, index=eval_test.index)
                    eval_records = eval_test.copy()
                    eval_records["p_success"] = eval_records[p_source_col]
                    eval_records["utility_score"] = eval_records[score_source_col]
                    eval_records["threshold"] = threshold_value
                    eval_records["selected_by_threshold"] = selected
                    eval_records["threshold_reason"] = threshold_info.get("threshold_reason", "")
                    eval_records["threshold_decision_eligible"] = bool(threshold_info.get("threshold_decision_eligible", False))
                    eval_records["trial_count"] = int(threshold_info.get("trial_count", 0) or 0)
                    eval_records["threshold_policy_type"] = "raw_score"
                    eval_records["applied_threshold_policy_type"] = "raw_score"
                    eval_records["diagnostic_best_candidate_policy_type"] = "raw_score"
                    eval_records["threshold_policy_source_window"] = "threshold"
                    eval_records["threshold_policy_applied_window"] = "test"
                    eval_records["threshold_policy_provenance_valid"] = True
                    eval_records["risk_adjusted_selection_score_col"] = ""
                for idx, row in eval_records.iterrows():
                    records.append(
                        {
                            "fold_id": fold.fold_id,
                            "split": f"oof_test_{fold.fold_id.replace('wf_', '')}",
                            "validation_design": "walk_forward_oof",
                            "model_name": model_name,
                            "evaluation_scope": evaluation_scope,
                            "symbol": row.get("symbol"),
                            "symbol_group": row.get("symbol_group"),
                            "date": row.get("date"),
                            "signal_idx": row.get("signal_idx", np.nan),
                            MODEL_TRAINING_COL: row.get(MODEL_TRAINING_COL, False),
                            DECISION_ENTRY_COL: row.get(DECISION_ENTRY_COL, False),
                            TRADE_READY_COL: row.get(TRADE_READY_COL, False),
                            "candidate_tier": row.get("candidate_tier", ""),
                            "entry_trigger": row.get("entry_trigger", ""),
                            "trend_regime": row.get("trend_regime", ""),
                            "vol_regime": row.get("vol_regime", ""),
                            "drawdown_bucket": row.get("drawdown_bucket", ""),
                            "score_price_algo_total": row.get("score_price_algo_total", np.nan),
                            TARGET_COL: row.get(TARGET_COL, np.nan),
                            RETURN_COL: row.get(RETURN_COL, np.nan),
                            EXPECTED_R_COL: row.get(EXPECTED_R_COL, np.nan),
                            STOP_SURVIVAL_COL: row.get(STOP_SURVIVAL_COL, np.nan),
                            "p_success": row.get("p_success", row.get(p_source_col, np.nan)),
                            "utility_score": row.get("utility_score", row.get(score_source_col, np.nan)),
                            "raw_utility_score": row.get("raw_utility_score", row.get(score_source_col, np.nan)),
                            "risk_adjusted_selection_score": row.get("risk_adjusted_selection_score", np.nan),
                            "threshold": row.get("threshold", np.nan),
                            "selected_by_threshold": bool(row.get("selected_by_threshold", False)),
                            "base_rate": base_rate,
                            "threshold_reason": row.get("threshold_reason", threshold_info.get("threshold_reason", "")),
                            "threshold_decision_eligible": bool(row.get("threshold_decision_eligible", False)),
                            "trial_count": int(row.get("trial_count", threshold_info.get("trial_count", 0)) or 0),
                            "threshold_policy_type": row.get("threshold_policy_type", "raw_score"),
                            "applied_threshold_policy_type": row.get("applied_threshold_policy_type", row.get("threshold_policy_type", "raw_score")),
                            "diagnostic_best_candidate_policy_type": row.get("diagnostic_best_candidate_policy_type", "raw_score"),
                            "threshold_policy_source_window": row.get("threshold_policy_source_window", "threshold"),
                            "threshold_policy_applied_window": row.get("threshold_policy_applied_window", "test"),
                            "threshold_policy_provenance_valid": bool(row.get("threshold_policy_provenance_valid", True)),
                            "stable_threshold_candidate_count": int(row.get("stable_threshold_candidate_count", 0) or 0),
                            "threshold_stability_failure_reasons": row.get("threshold_stability_failure_reasons", ""),
                            "risk_adjusted_selection_score_col": row.get("risk_adjusted_selection_score_col", ""),
                            "calibration_method": calibration_method,
                            "raw_probability_col": raw_col,
                            "utility_weight_label": threshold_info.get("utility_weight_label", ""),
                        }
                    )
    oof_predictions = pd.DataFrame(records)
    oof_threshold_policy = pd.concat(oof_threshold_policy_parts, ignore_index=True) if oof_threshold_policy_parts else pd.DataFrame()
    oof_metrics = build_oof_metrics(oof_predictions)
    return oof_predictions, oof_metrics, oof_threshold_policy


def quality_check(check: str, passed: bool, severity: str, value, details: str = "") -> Dict[str, object]:
    return {"check": check, "passed": bool(passed), "severity": severity, "value": value, "details": details}


def quality_passed(quality: pd.DataFrame) -> bool:
    if quality.empty or "passed" not in quality.columns:
        return False
    critical = quality[quality["severity"].astype(str).eq("CRITICAL")] if "severity" in quality.columns else quality
    target = critical if not critical.empty else quality
    return target["passed"].astype(str).str.lower().isin(["true", "1", "yes"]).all()


def decision_scope_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "evaluation_scope" not in metrics.columns:
        return metrics
    decision = metrics[metrics["evaluation_scope"].astype(str).eq(TRADE_READY_EVAL_SCOPE)].copy()
    return decision if not decision.empty else metrics


def dataset_quality_block_reasons(pooled_quality: pd.DataFrame | None) -> list[str]:
    if pooled_quality is None or pooled_quality.empty or "check" not in pooled_quality.columns:
        return []
    failed = pooled_quality[~pooled_quality["passed"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    checks = set(failed["check"].astype(str))
    reasons: list[str] = []
    if "pooled_loaded_symbol_count_at_least_10" in checks:
        reasons.append("POOLED_SYMBOLS_LT_10")
    if "pooled_trade_ready_20d_labeled_at_least_500" in checks:
        reasons.append("POOLED_TRADE_READY_20D_LABELS_LT_500")
    if "pooled_model_training_20d_labeled_at_least_10000" in checks:
        reasons.append("POOLED_MODEL_TRAINING_20D_LABELS_LT_10000")
    if "pooled_external_features_present" in checks:
        reasons.append("EXTERNAL_FEATURES_MISSING")
    return reasons


def block_reasons(model_metrics: pd.DataFrame, tsm_metrics: pd.DataFrame, quality_ok: bool, pooled_quality: pd.DataFrame | None = None) -> str:
    reasons: List[str] = []
    if not quality_ok:
        reasons.append("POOLED_DATASET_QUALITY_FAILED")
        reasons.extend(dataset_quality_block_reasons(pooled_quality))
    model_metrics = decision_scope_metrics(model_metrics)
    validation_design = ""
    if "validation_design" in model_metrics.columns and not model_metrics.empty:
        validation_design = str(model_metrics["validation_design"].dropna().astype(str).iloc[0]) if model_metrics["validation_design"].notna().any() else ""
    eval_row = model_metrics[model_metrics["split"].eq("combined_test_holdout")]
    test_row = model_metrics[model_metrics["split"].eq("test_2024")]
    holdout_row = model_metrics[model_metrics["split"].eq("final_holdout_2025_2026")]
    if "is_selected_tsm_calibration_route" in tsm_metrics.columns:
        selected_tsm_metrics = tsm_metrics[tsm_metrics["is_selected_tsm_calibration_route"].map(to_bool)].copy()
        if not selected_tsm_metrics.empty:
            tsm_metrics = selected_tsm_metrics
    tsm_eval = tsm_metrics[tsm_metrics["split"].eq("tsm_combined_test_holdout")]
    if eval_row.empty:
        reasons.append("MISSING_COMBINED_EVAL")
    else:
        row = eval_row.iloc[0]
        if int(row.get("event_count", 0)) < MIN_EVAL_EVENTS:
            reasons.append("POOLED_EVAL_EVENTS_LT_150")
        if int(row.get("selected_event_count", 0)) < MIN_SELECTED_EVAL_EVENTS:
            reasons.append("POOLED_SELECTED_EVENTS_LT_50")
        if not to_bool(row.get("threshold_decision_eligible", False)):
            reasons.append("POOLED_THRESHOLD_NOT_DECISION_ELIGIBLE")
        if "threshold_stability_pass" in row.index and str(row.get("validation_design", "")) == "walk_forward_oof" and not to_bool(row.get("threshold_stability_pass", False)):
            reasons.append("POOLED_THRESHOLD_STABILITY_FAILED")
        selected_fraction = safe_float(row.get("selected_fraction"))
        if pd.isna(selected_fraction) or selected_fraction < MIN_SELECTED_FRACTION or selected_fraction > MAX_SELECTED_FRACTION:
            reasons.append("POOLED_SELECTED_FRACTION_OUT_OF_RANGE")
        if safe_float(row.get("selected_expectancy_ci_lower_pct")) <= MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT:
            reasons.append("POOLED_SELECTED_EXPECTANCY_CI_LOWER_LE_0")
        if safe_float(row.get("selected_minus_all_pct")) <= MIN_SELECTED_MINUS_ALL_PCT:
            reasons.append("POOLED_SELECTED_MINUS_ALL_LE_0")
        selected_minus_all_lower = safe_float(row.get("selected_minus_all_ci_lower_pct_paired", row.get("selected_minus_all_ci_lower_pct")))
        selected_minus_score_lower = safe_float(row.get("selected_minus_score_baseline_ci_lower_pct_paired", row.get("selected_minus_score_baseline_ci_lower_pct")))
        if "selected_minus_all_ci_lower_pct" in row.index and selected_minus_all_lower <= MIN_SELECTED_MINUS_ALL_PCT:
            reasons.append("POOLED_SELECTED_MINUS_ALL_CI_LOWER_LE_0")
        if "selected_minus_score_baseline_ci_lower_pct" in row.index and selected_minus_score_lower <= MIN_SELECTED_MINUS_ALL_PCT:
            reasons.append("POOLED_SELECTED_MINUS_SCORE_BASELINE_CI_LOWER_LE_0")
        if "uplift_pass" in row.index and not to_bool(row.get("uplift_pass", False)):
            reasons.append("POOLED_UPLIFT_NOT_PASSED")
        if safe_float(row.get("decision_ece", row.get("ece"))) > MAX_ECE:
            reasons.append("POOLED_ECE_GT_0_10")
        if "decision_min_calibration_bin_n" in row.index and safe_float(row.get("decision_min_calibration_bin_n")) < MIN_TSM_CALIBRATION_EVENTS:
            reasons.append("POOLED_DECISION_CALIBRATION_BIN_N_LT_30")
        if safe_float(row.get("brier_improvement_pct")) <= 0:
            reasons.append("POOLED_NO_BRIER_IMPROVEMENT")
        if "positive_expectancy_fold_count" in row.index and str(row.get("validation_design", "")) == "walk_forward_oof" and safe_float(row.get("positive_expectancy_fold_count")) < 4:
            reasons.append("POOLED_POSITIVE_EXPECTANCY_FOLDS_LT_4")
        if safe_float(row.get("trial_count"), 0) <= 0:
            reasons.append("POOLED_TRIAL_LEDGER_MISSING")
    if validation_design == "walk_forward_oof":
        fold_rows = model_metrics[model_metrics["split"].astype(str).str.startswith("oof_test_")]
        weak_oof_value = str(eval_row.iloc[0].get("weak_oof_folds", "")) if not eval_row.empty else ""
        weak_folds = [part for part in weak_oof_value.split("|") if part]
        if not weak_folds:
            weak_folds = [
                str(row.get("split"))
                for _, row in fold_rows.iterrows()
                if int(row.get("event_count", 0)) > 0 and int(row.get("selected_event_count", 0)) < MIN_SELECTED_PER_EVAL_SPLIT
            ]
        if weak_folds:
            reasons.append("POOLED_OOF_FOLD_SELECTED_EVENTS_LT_10")
    else:
        for split_name, split_rows in [("test", test_row), ("holdout", holdout_row)]:
            if split_rows.empty or int(split_rows.iloc[0].get("selected_event_count", 0)) < MIN_SELECTED_PER_EVAL_SPLIT:
                reasons.append(f"POOLED_{split_name.upper()}_SELECTED_EVENTS_LT_10")
    if tsm_eval.empty:
        reasons.append("MISSING_TSM_EVAL")
    else:
        row = tsm_eval.iloc[0]
        if int(row.get("event_count", 0)) < MIN_TSM_EVAL_EVENTS:
            reasons.append("TSM_EVAL_EVENTS_LT_30")
        if "tsm_calibration_route_pass" in row.index and not to_bool(row.get("tsm_calibration_route_pass", False)):
            reasons.append("TSM_CALIBRATION_ROUTE_NOT_PASSED")
            reasons.append("TSM_CALIBRATION_ROUTE_FAILED")
        if safe_float(row.get("decision_ece", row.get("ece"))) > MAX_TSM_ECE:
            reasons.append("TSM_CALIBRATION_ECE_GT_0_15")
    return "|".join(reasons) if reasons else "PASS"


def selected_tsm_like_selection_row(tsm_like_metrics: pd.DataFrame) -> pd.Series:
    if tsm_like_metrics.empty:
        return pd.Series(dtype=object)
    metrics = tsm_like_metrics.copy()
    if "is_selected_tsm_like_route" in metrics.columns:
        metrics = metrics[metrics["is_selected_tsm_like_route"].map(to_bool)].copy()
    selection = metrics[metrics["split"].astype(str).eq("tsm_like_train_validation")].copy()
    if not selection.empty:
        return selection.iloc[0]
    return metrics.iloc[0] if not metrics.empty else pd.Series(dtype=object)


def build_paper_gate_snapshot(
    comparison: pd.DataFrame,
    tsm_like_metrics: pd.DataFrame,
    overlay: Dict[str, object],
    pooled_dataset_quality_ok: bool,
) -> pd.DataFrame:
    decision_metrics = decision_scope_metrics(comparison)
    eval_row = decision_metrics[decision_metrics["split"].eq("combined_test_holdout")].iloc[0] if not decision_metrics.empty else pd.Series(dtype=object)
    tsm_like_row = selected_tsm_like_selection_row(tsm_like_metrics)
    selected_minus_all_lower = safe_float(eval_row.get("selected_minus_all_ci_lower_pct_paired", eval_row.get("selected_minus_all_ci_lower_pct")))
    fold_min = safe_float(eval_row.get("fold_selected_count_min", eval_row.get("min_selected_events_per_fold")), np.nan)
    latest_trade_ready = to_bool(overlay.get("latest_trade_ready", False))
    latest_stop = safe_float(overlay.get("p_stop_hit_20d"), np.nan)
    paper_model_failures: list[str] = []
    if not pooled_dataset_quality_ok:
        paper_model_failures.append("POOLED_DATASET_QUALITY_FAILED")
    if not to_bool(tsm_like_row.get("tsm_like_route_selection_pass", False)):
        paper_model_failures.append("TSM_LIKE_CALIBRATION_FAILED")
        paper_model_failures.append("TSM_CALIBRATION_ROUTE_FAILED")
    if safe_float(tsm_like_row.get("selection_effective_n", tsm_like_row.get("effective_n")), 0.0) < MIN_TSM_LIKE_EFFECTIVE_SELECTION_N:
        paper_model_failures.append("TSM_LIKE_EFFECTIVE_TRAIN_VALIDATION_N_LT_500")
    if safe_float(tsm_like_row.get("selection_decision_ece", tsm_like_row.get("decision_ece")), np.nan) > MAX_TSM_ECE:
        paper_model_failures.append("TSM_LIKE_ECE_GT_0_15")
    if safe_float(eval_row.get("decision_ece", eval_row.get("ece")), np.nan) > PAPER_MAX_ECE:
        paper_model_failures.append("POOLED_ECE_GT_0_12")
    if safe_float(eval_row.get("brier_improvement_pct"), np.nan) <= 0:
        paper_model_failures.append("POOLED_NO_BRIER_IMPROVEMENT")
    if safe_float(eval_row.get("selected_event_count"), 0.0) < PAPER_MIN_SELECTED_EVAL_EVENTS:
        paper_model_failures.append("POOLED_SELECTED_EVENTS_LT_100")
    if pd.isna(fold_min) or fold_min < PAPER_MIN_SELECTED_PER_EVAL_SPLIT:
        paper_model_failures.append("POOLED_OOF_FOLD_SELECTED_EVENTS_LT_25")
    if safe_float(eval_row.get("selected_mean_return_pct"), np.nan) <= safe_float(eval_row.get("mean_return_pct"), np.nan):
        paper_model_failures.append("POOLED_SELECTED_MEAN_LE_RULE_ALL_MEAN")
    if selected_minus_all_lower <= PAPER_MIN_SELECTED_MINUS_ALL_CI_LOWER_PCT:
        paper_model_failures.append("POOLED_SELECTED_MINUS_RULE_PAIRED_CI_LOWER_LT_MINUS_0_50")
    latest_failures: list[str] = []
    if not latest_trade_ready:
        latest_failures.append("LATEST_NOT_TRADE_READY")
    if pd.isna(latest_stop) or latest_stop > PAPER_MAX_STOP_HIT_FOR_LATEST:
        latest_failures.append("POOLED_STOP_RISK_GT_0_40")
    paper_model_pass = not paper_model_failures
    paper_latest_pass = not latest_failures
    paper_allowed = bool(paper_model_pass and paper_latest_pass)
    strict_allowed = to_bool(overlay.get("decision_support_allowed", False))
    if strict_allowed:
        strict_status = "STRICT_DECISION_SUPPORT_ALLOWED"
    elif not latest_trade_ready:
        strict_status = "DISPLAY_ONLY_NO_LATEST_TRADE_READY"
    else:
        strict_status = "DISPLAY_ONLY_MODEL_BLOCKED"
    if paper_allowed:
        paper_status = "PAPER_DECISION_SUPPORT_ALLOWED"
    elif not latest_trade_ready:
        paper_status = "DISPLAY_ONLY_NO_LATEST_TRADE_READY"
    else:
        paper_status = "DISPLAY_ONLY_MODEL_BLOCKED"
    rows = [
        {"field": "strict_gate_status", "value": strict_status},
        {"field": "paper_gate_status", "value": paper_status},
        {"field": "strict_decision_support_allowed", "value": strict_allowed},
        {"field": "paper_decision_support_allowed", "value": paper_allowed},
        {"field": "paper_model_gate_pass", "value": paper_model_pass},
        {"field": "paper_latest_signal_pass", "value": paper_latest_pass},
        {"field": "paper_model_block_reasons", "value": "|".join(sorted(set(paper_model_failures))) if paper_model_failures else "PASS"},
        {"field": "paper_latest_block_reasons", "value": "|".join(sorted(set(latest_failures))) if latest_failures else "PASS"},
        {"field": "paper_gate_block_reasons", "value": "PASS" if paper_allowed else "|".join(sorted(set(paper_model_failures + latest_failures)))},
        {"field": "tsm_like_calibration_route", "value": tsm_like_row.get("tsm_like_calibration_route", "")},
        {"field": "tsm_like_effective_train_validation_n", "value": tsm_like_row.get("selection_effective_n", tsm_like_row.get("effective_n", np.nan))},
        {"field": "tsm_like_calibration_ece", "value": tsm_like_row.get("selection_decision_ece", tsm_like_row.get("decision_ece", np.nan))},
        {"field": "tsm_like_route_selection_pass", "value": tsm_like_row.get("tsm_like_route_selection_pass", False)},
        {"field": "pooled_ece_for_paper", "value": eval_row.get("decision_ece", eval_row.get("ece", np.nan))},
        {"field": "pooled_brier_improvement_pct_for_paper", "value": eval_row.get("brier_improvement_pct", np.nan)},
        {"field": "paper_selected_oos_event_count", "value": eval_row.get("selected_event_count", np.nan)},
        {"field": "paper_min_selected_events_per_oof_fold", "value": fold_min},
        {"field": "paper_selected_minus_rule_ci_lower_pct", "value": selected_minus_all_lower},
        {"field": "latest_trade_ready", "value": latest_trade_ready},
        {"field": "latest_stop_hit_20d", "value": latest_stop},
        {"field": "prediction_ready", "value": False},
        {"field": "live_ready", "value": False},
        {"field": "paper_trading_status", "value": "PREDICTION_PAPER_ALPHA_READY" if paper_allowed else "RULE_BASED_READY_PREDICTION_DISPLAY_ONLY"},
        {"field": "live_trading_status", "value": "DISABLED_BY_DESIGN"},
    ]
    return pd.DataFrame(rows)


def build_overlay_rows(values: Dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([{"field": key, "value": value} for key, value in values.items()])


def read_latest_snapshot(path: Path) -> Dict[str, object]:
    frame = read_csv(path)
    if frame.empty:
        return {}
    require_columns(frame, ["field", "value"], str(path))
    return dict(zip(frame["field"], frame["value"]))


def write_latest_snapshot(path: Path, values: Dict[str, object]) -> None:
    pd.DataFrame([{"field": key, "value": value} for key, value in values.items()]).to_csv(path, index=False)


def update_latest_snapshot(latest_path: Path, overlay: Dict[str, object]) -> None:
    latest = read_latest_snapshot(latest_path)
    if not latest:
        return
    for key, value in overlay.items():
        latest[f"pooled_{key}"] = value
    if to_bool(overlay.get("decision_support_allowed", False)):
        latest["prediction_scope_used"] = "pooled_trade_ready_entry"
        latest["model_support_route"] = "POOLED_SEMI_DECISION_SUPPORT"
        latest["best_model_20d"] = overlay.get("model_name")
        latest["p_success_20d"] = overlay.get("p_success_20d")
        latest["p_stop_survival_20d"] = overlay.get("p_stop_survival_20d")
        latest["p_stop_hit_20d"] = overlay.get("p_stop_hit_20d")
        latest["p_hit_1r_20d"] = overlay.get("p_hit_1r_20d")
        latest["p_hit_2r_20d"] = overlay.get("p_hit_2r_20d")
        latest["expected_r_20d"] = overlay.get("expected_r_net_20d")
        latest["expected_net_return_20d"] = overlay.get("expected_net_return_pct_20d")
        latest["threshold_20d"] = overlay.get("threshold_20d")
        latest["prediction_quality_pass_20d"] = True
        latest["oos_event_count_20d"] = overlay.get("oos_event_count")
        latest["selected_oos_event_count_20d"] = overlay.get("selected_oos_event_count")
        latest["selected_expectancy_ci_lower_pct_20d"] = overlay.get("selected_expectancy_ci_lower_pct")
        latest["selected_minus_rule_all_pct_20d"] = overlay.get("selected_minus_all_pct")
        latest["model_quality_block_reasons_20d"] = "PASS"
        latest["prediction_signal_status"] = "PREDICTION_CONFIRMED"
        latest["prediction_use_status"] = "DECISION_SUPPORT_ALLOWED"
        latest["decision_permission"] = "DECISION_SUPPORT_ONLY"
        latest["model_quality_block_reasons"] = "PASS"
        latest["final_trade_decision"] = "POOLED_ALPHA_RESEARCH_LONG_ALLOWED"
    else:
        latest["model_support_route"] = overlay.get("model_support_route", latest.get("model_support_route", "DISPLAY_ONLY_NO_MODEL_CANDIDATE"))
    write_latest_snapshot(latest_path, latest)


def run_pooled_model_legacy(features: pd.DataFrame, pooled_quality: pd.DataFrame) -> Dict[str, pd.DataFrame | Dict[str, object]]:
    quality_ok = quality_passed(pooled_quality)
    data = prepare_dataset(features)
    train = data[data["date_split"].eq("train_2016_2022")].copy()
    validation = data[data["date_split"].eq("validation_2023")].copy()
    test = data[data["date_split"].eq("test_2024")].copy()
    holdout = data[data["date_split"].eq("final_holdout_2025_2026")].copy()
    if len(data) < MIN_POOLED_TRADE_READY_LABELS:
        raise ValueError(f"pooled trade-ready 20D labels below required minimum: {len(data)}")
    model = fit_empirical_bayes(train)

    split_predictions = []
    for split_name, frame in [
        ("train_2016_2022", train),
        ("validation_2023", validation),
        ("test_2024", test),
        ("final_holdout_2025_2026", holdout),
    ]:
        pred = predict_empirical_bayes(model, frame)
        pred["split"] = split_name
        split_predictions.append(pred)
    predictions = pd.concat(split_predictions, ignore_index=True)
    tsm_layer_source = predictions[(predictions["symbol"].eq("TSM")) & (predictions["split"].isin(["train_2016_2022", "validation_2023"]))].copy()
    tsm_layer = fit_tsm_calibration_layer(tsm_layer_source, "p_success_base", model.global_success)
    predictions["p_success_tsm_calibrated"] = predictions["p_success_base"]
    tsm_mask = predictions["symbol"].eq("TSM")
    predictions.loc[tsm_mask, "p_success_tsm_calibrated"] = apply_tsm_layer(predictions.loc[tsm_mask], tsm_layer)

    threshold_info = choose_threshold(predictions[predictions["split"].eq("validation_2023")])
    threshold = float(threshold_info["threshold"])
    combined_eval = predictions[predictions["split"].isin(["test_2024", "final_holdout_2025_2026"])].copy()
    tsm_combined_eval = predictions[(predictions["symbol"].eq("TSM")) & (predictions["split"].isin(["test_2024", "final_holdout_2025_2026"]))].copy()

    metric_rows = [
        metric_row("train_2016_2022", predictions[predictions["split"].eq("train_2016_2022")], "p_success_base", threshold, model.global_success, "pooled_empirical_bayes_group_rate"),
        metric_row("validation_2023", predictions[predictions["split"].eq("validation_2023")], "p_success_base", threshold, model.global_success, "pooled_empirical_bayes_group_rate"),
        metric_row("test_2024", predictions[predictions["split"].eq("test_2024")], "p_success_base", threshold, model.global_success, "pooled_empirical_bayes_group_rate"),
        metric_row("final_holdout_2025_2026", predictions[predictions["split"].eq("final_holdout_2025_2026")], "p_success_base", threshold, model.global_success, "pooled_empirical_bayes_group_rate"),
        metric_row("combined_test_holdout", combined_eval, "p_success_base", threshold, model.global_success, "pooled_empirical_bayes_group_rate"),
    ]
    comparison = pd.DataFrame(metric_rows)
    tsm_metrics = pd.DataFrame(
        [
            metric_row("tsm_train_validation", tsm_layer_source, "p_success_base", threshold, model.global_success, "tsm_specific_calibrated_layer"),
            metric_row("tsm_test_2024", predictions[(predictions["symbol"].eq("TSM")) & (predictions["split"].eq("test_2024"))], "p_success_tsm_calibrated", threshold, model.global_success, "tsm_specific_calibrated_layer"),
            metric_row("tsm_final_holdout_2025_2026", predictions[(predictions["symbol"].eq("TSM")) & (predictions["split"].eq("final_holdout_2025_2026"))], "p_success_tsm_calibrated", threshold, model.global_success, "tsm_specific_calibrated_layer"),
            metric_row("tsm_combined_test_holdout", tsm_combined_eval, "p_success_tsm_calibrated", threshold, model.global_success, "tsm_specific_calibrated_layer"),
        ]
    )
    reasons = block_reasons(comparison, tsm_metrics, quality_ok, pooled_quality)
    model_quality_pass = reasons == "PASS"

    latest_tsm = features[features["symbol"].astype(str).eq("TSM")].copy()
    latest_tsm["date"] = pd.to_datetime(latest_tsm["date"], errors="coerce")
    latest_tsm = latest_tsm.sort_values("date").tail(1)
    final_model = fit_empirical_bayes(data)
    final_tsm_train = predict_empirical_bayes(final_model, data[data["symbol"].eq("TSM")].copy())
    final_tsm_layer = fit_tsm_calibration_layer(final_tsm_train, "p_success_base", final_model.global_success)
    latest_pred = predict_empirical_bayes(final_model, latest_tsm) if not latest_tsm.empty else pd.DataFrame()
    if not latest_pred.empty:
        latest_pred["p_success_tsm_calibrated"] = apply_tsm_layer(latest_pred, final_tsm_layer)
        latest_row = latest_pred.iloc[0]
        latest_trade_ready = to_bool(latest_row.get(TRADE_READY_COL))
        latest_p = safe_float(latest_row.get("p_success_tsm_calibrated"))
        latest_stop_hit = safe_float(latest_row.get("p_stop_hit"))
        latest_expected_r = safe_float(latest_row.get("expected_r_net"))
        latest_allowed = bool(
            model_quality_pass
            and latest_trade_ready
            and latest_p >= threshold
            and latest_stop_hit <= MAX_STOP_HIT_FOR_LATEST
            and latest_expected_r >= MIN_EXPECTED_R_FOR_LATEST
        )
        latest_block_reasons = []
        if not model_quality_pass:
            latest_block_reasons.append("POOLED_MODEL_QUALITY_NOT_PASSED")
        if not latest_trade_ready:
            latest_block_reasons.append("LATEST_NOT_TRADE_READY")
        if pd.isna(latest_p) or latest_p < threshold:
            latest_block_reasons.append("POOLED_PROBABILITY_BELOW_THRESHOLD")
        if pd.isna(latest_stop_hit) or latest_stop_hit > MAX_STOP_HIT_FOR_LATEST:
            latest_block_reasons.append("POOLED_STOP_RISK_GT_0_35")
        if pd.isna(latest_expected_r) or latest_expected_r < MIN_EXPECTED_R_FOR_LATEST:
            latest_block_reasons.append("POOLED_EXPECTED_R_LT_0_35")
    else:
        latest_trade_ready = False
        latest_allowed = False
        latest_block_reasons = ["MISSING_TSM_LATEST_ROW"]
        latest_row = pd.Series(dtype=object)
        latest_p = latest_stop_hit = latest_expected_r = np.nan

    eval_row = comparison[comparison["split"].eq("combined_test_holdout")].iloc[0]
    overlay = {
        "model_name": "pooled_empirical_bayes_group_rate",
        "asof_date": latest_row.get("date", ""),
        "model_quality_pass": model_quality_pass,
        "tsm_calibration_status": final_tsm_layer.status,
        "decision_support_allowed": latest_allowed,
        "model_support_route": "POOLED_SEMI_DECISION_SUPPORT" if latest_allowed else "DISPLAY_ONLY_NO_MODEL_CANDIDATE",
        "decision_block_reasons": "PASS" if latest_allowed else "|".join(latest_block_reasons),
        "model_quality_block_reasons": reasons,
        "threshold_20d": threshold,
        "threshold_reason": threshold_info["threshold_reason"],
        "p_success_20d": latest_p,
        "p_stop_survival_20d": latest_row.get("p_stop_survival", np.nan),
        "p_stop_hit_20d": latest_stop_hit,
        "p_hit_1r_20d": latest_row.get("p_hit_1r", np.nan),
        "p_hit_2r_20d": latest_row.get("p_hit_2r", np.nan),
        "expected_r_net_20d": latest_expected_r,
        "expected_net_return_pct_20d": latest_row.get("expected_net_return_pct", np.nan),
        "effective_group_n": latest_row.get("effective_group_n", 0),
        "oos_event_count": int(eval_row.get("event_count", 0)),
        "selected_oos_event_count": int(eval_row.get("selected_event_count", 0)),
        "selected_expectancy_ci_lower_pct": eval_row.get("selected_expectancy_ci_lower_pct", np.nan),
        "selected_minus_all_pct": eval_row.get("selected_minus_all_pct", np.nan),
        "ece": eval_row.get("ece", np.nan),
        "brier_improvement_pct": eval_row.get("brier_improvement_pct", np.nan),
    }
    calibration = pd.DataFrame(
        [
            {
                "layer": "tsm_specific_logit_shift",
                "event_count": tsm_layer.event_count,
                "actual_success_rate": tsm_layer.actual_success_rate,
                "predicted_success_rate": tsm_layer.predicted_success_rate,
                "posterior_success_rate": tsm_layer.posterior_success_rate,
                "logit_shift": tsm_layer.logit_shift,
                "shrinkage": tsm_layer.shrinkage,
                "status": tsm_layer.status,
                "fit_source": "train_2016_2022_plus_validation_2023",
            },
            {
                "layer": "tsm_specific_logit_shift_latest_fit",
                "event_count": final_tsm_layer.event_count,
                "actual_success_rate": final_tsm_layer.actual_success_rate,
                "predicted_success_rate": final_tsm_layer.predicted_success_rate,
                "posterior_success_rate": final_tsm_layer.posterior_success_rate,
                "logit_shift": final_tsm_layer.logit_shift,
                "shrinkage": final_tsm_layer.shrinkage,
                "status": final_tsm_layer.status,
                "fit_source": "all_labeled_history_for_latest_snapshot",
            },
        ]
    )
    quality = build_quality_checks(data, comparison, tsm_metrics, calibration, overlay, quality_ok)
    predictions_out = predictions[
        [
            "symbol",
            "symbol_group",
            "date",
            "split",
            "signal_idx",
            TARGET_COL,
            RETURN_COL,
            EXPECTED_R_COL,
            STOP_SURVIVAL_COL,
            "p_success_base",
            "p_success_tsm_calibrated",
            "p_stop_hit",
            "expected_r_net",
            "expected_net_return_pct",
            "effective_group_n",
            *GROUP_COLS,
        ]
    ].copy()
    return {
        "comparison": comparison,
        "tsm_metrics": tsm_metrics,
        "threshold_policy": threshold_info["threshold_table"],
        "oos_predictions": predictions_out,
        "calibration": calibration,
        "overlay": overlay,
        "quality": quality,
    }


def run_pooled_model(features: pd.DataFrame, pooled_quality: pd.DataFrame) -> Dict[str, pd.DataFrame | Dict[str, object]]:
    quality_ok = quality_passed(pooled_quality)
    data = prepare_dataset(features)
    train = data[data["date_split"].eq("train_2016_2022")].copy()
    validation = data[data["date_split"].eq("validation_2023")].copy()
    test = data[data["date_split"].eq("test_2024")].copy()
    holdout = data[data["date_split"].eq("final_holdout_2025_2026")].copy()
    model_training_label_count = len(data)
    decision_label_count = int(data[DECISION_ENTRY_COL].map(to_bool).sum()) if DECISION_ENTRY_COL in data.columns else int(data[TRADE_READY_COL].map(to_bool).sum())

    eb_model = fit_empirical_bayes(train)
    feature_cols = pooled_feature_columns(train)
    split_predictions = []
    for split_name, frame in [
        ("train_2016_2022", train),
        ("validation_2023", validation),
        ("test_2024", test),
        ("final_holdout_2025_2026", holdout),
    ]:
        pred = predict_empirical_bayes(eb_model, frame)
        pred["split"] = split_name
        split_predictions.append(pred)
    predictions = pd.concat(split_predictions, ignore_index=True)
    predictions["label_stop_hit_20d"] = 1.0 - pd.to_numeric(predictions[STOP_SURVIVAL_COL], errors="coerce")
    predictions["p_success_eb"] = predictions["p_success_base"]
    predictions["p_stop_hit_eb"] = predictions["p_stop_hit"]
    predictions["expected_r_eb"] = predictions["expected_r_net"]
    fitted_candidates: dict[str, dict[str, object]] = {}
    skipped_candidates: list[dict[str, object]] = []
    hier_eb_model = fit_empirical_bayes(train, prior_strength=80.0, group_cols=HIERARCHICAL_GROUP_COLS)
    hier_predictions = predict_empirical_bayes(hier_eb_model, predictions)
    predictions["p_success_hier_eb"] = hier_predictions["p_success_base"]
    fitted_candidates["pooled_hierarchical_empirical_bayes"] = {"model": hier_eb_model, "kind": "empirical_bayes"}

    def fit_candidate(name: str, fit_fn):
        try:
            candidate = fit_fn()
        except Exception as exc:
            skipped_candidates.append({"layer": "candidate_fit", "model_name": name, "status": "SKIPPED_FIT_ERROR", "details": str(exc)[:300]})
            return None
        if candidate is None:
            skipped_candidates.append({"layer": "candidate_fit", "model_name": name, "status": "SKIPPED_DEPENDENCY_MISSING", "details": ""})
            return None
        fitted_candidates[str(candidate["name"])] = candidate
        return candidate

    if SKLEARN_IMPORT_ERROR is None and feature_cols and validation[TARGET_COL].nunique() >= 2:
        logistic = fit_candidate("pooled_elastic_net_logistic", lambda: fit_pooled_elastic_net_candidate(train, feature_cols))
        if logistic is not None:
            predictions["p_success_logistic"] = predict_pooled_candidate(logistic, predictions)
        weighted_logistic = fit_candidate("pooled_weighted_elastic_net_logistic", lambda: fit_pooled_weighted_elastic_net_candidate(train, feature_cols))
        if weighted_logistic is not None:
            predictions["p_success_weighted_logistic"] = predict_pooled_candidate(weighted_logistic, predictions)
        strict_logistic = fit_candidate("pooled_strict_elastic_net_logistic", lambda: fit_strict_elastic_net_candidate(train, feature_cols))
        if strict_logistic is not None:
            predictions["p_success_strict_logistic"] = predict_pooled_candidate(strict_logistic, predictions)
        hist = fit_candidate("pooled_hist_gradient_boosting", lambda: fit_pooled_hist_gbm_candidate(train, feature_cols))
        if hist is not None:
            predictions["p_success_hist_gbm"] = predict_pooled_candidate(hist, predictions)
        if quality_ok:
            lgbm = fit_candidate("pooled_lgbm_classifier", lambda: fit_pooled_lgbm_success_candidate(train, validation, feature_cols))
            if lgbm is not None:
                predictions["p_success_lgbm"] = predict_pooled_candidate(lgbm, predictions)
            xgb = fit_candidate("pooled_xgb_classifier", lambda: fit_pooled_xgb_candidate(train, validation, feature_cols))
            if xgb is not None:
                predictions["p_success_xgb"] = predict_pooled_candidate(xgb, predictions)
            try:
                aux_heads = fit_pooled_lgbm_aux_heads(train, validation, feature_cols)
            except Exception as exc:
                aux_heads = {}
                skipped_candidates.append({"layer": "candidate_fit", "model_name": "pooled_aux_heads", "status": "SKIPPED_FIT_ERROR", "details": str(exc)[:300]})
        else:
            aux_heads = {}
            skipped_candidates.append({"layer": "candidate_fit", "model_name": "pooled_lgbm_and_aux_heads", "status": "SKIPPED_DATASET_QUALITY_FAILED", "details": "LightGBM/XGB candidates require pooled dataset quality gate pass."})
        if "stop" in aux_heads:
            fitted_candidates["pooled_stop_head"] = aux_heads["stop"]
            predictions["p_stop_hit_lgbm"] = clip_probability(predict_pooled_candidate(aux_heads["stop"], predictions))
        if "expected_r" in aux_heads:
            fitted_candidates["pooled_expected_r_head"] = aux_heads["expected_r"]
            predictions["expected_r_lgbm"] = np.asarray(predict_pooled_candidate(aux_heads["expected_r"], predictions), dtype=float)
    else:
        skipped_candidates.append(
            {
                "layer": "candidate_fit",
                "model_name": "pooled_ml_candidates",
                "status": "SKIPPED_DEPENDENCY_OR_SAMPLE",
                "details": str(SKLEARN_IMPORT_ERROR or "missing features or validation class balance"),
            }
        )

    if "p_stop_hit_lgbm" not in predictions.columns:
        predictions["p_stop_hit_lgbm"] = predictions["p_stop_hit_eb"]
    if "expected_r_lgbm" not in predictions.columns:
        predictions["expected_r_lgbm"] = predictions["expected_r_eb"]

    validation_predictions = predictions[predictions["split"].eq("validation_2023")].copy()
    stack = fit_stack_calibrator(validation_predictions)
    if stack is not None:
        fitted_candidates["pooled_stack_calibrated"] = stack
        predictions["p_success_stack_raw"] = predict_stack_candidate(stack, predictions)

    raw_candidate_cols = [
        ("pooled_empirical_bayes_group_rate", "p_success_eb"),
        ("pooled_hierarchical_empirical_bayes", "p_success_hier_eb"),
        ("pooled_elastic_net_logistic", "p_success_logistic"),
        ("pooled_weighted_elastic_net_logistic", "p_success_weighted_logistic"),
        ("pooled_strict_elastic_net_logistic", "p_success_strict_logistic"),
        ("pooled_hist_gradient_boosting", "p_success_hist_gbm"),
        ("pooled_lgbm_classifier", "p_success_lgbm"),
        ("pooled_xgb_classifier", "p_success_xgb"),
        ("pooled_stack_calibrated", "p_success_stack_raw"),
    ]
    candidate_meta: dict[str, dict[str, object]] = {}
    metric_rows: list[dict[str, object]] = []
    threshold_tables: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, object]] = []
    predictions = predictions.copy()
    for model_name, raw_col in raw_candidate_cols:
        if raw_col not in predictions.columns:
            continue
        validation_for_candidate = predictions[predictions["split"].eq("validation_2023")].copy()
        calibrator, calibration_method = fit_probability_calibrator(
            validation_for_candidate[raw_col],
            validation_for_candidate[TARGET_COL],
        )
        global_p_col = f"{raw_col}_global_calibrated"
        tier_p_col = f"{raw_col}_tier_calibrated"
        entry_scope_p_col = f"{raw_col}_entry_scope_calibrated"
        p_col = f"{raw_col}_calibrated"
        trade_scope_p_col = f"{raw_col}_strict_scope_calibrated"
        trade_p_col = f"{raw_col}_trade_ready_calibrated"
        predictions[global_p_col] = apply_probability_calibrator(predictions[raw_col], calibrator, calibration_method)
        validation_for_candidate = predictions[predictions["split"].eq("validation_2023")].copy()
        tier_layers = fit_candidate_tier_calibration(validation_for_candidate, global_p_col, eb_model.global_success)
        predictions[tier_p_col] = apply_candidate_tier_calibration(predictions, global_p_col, tier_layers)
        validation_for_candidate = predictions[predictions["split"].eq("validation_2023")].copy()
        entry_scope_layer = fit_scope_logit_shift(
            validation_for_candidate,
            p_col=tier_p_col,
            global_success=eb_model.global_success,
            scope_name=ENTRY_RESEARCH_EVAL_SCOPE,
        )
        predictions[entry_scope_p_col] = apply_logit_shift(predictions[tier_p_col], entry_scope_layer)
        validation_for_candidate = predictions[predictions["split"].eq("validation_2023")].copy()
        entry_group_layers = fit_symbol_group_calibration(validation_for_candidate, entry_scope_p_col, eb_model.global_success)
        predictions[p_col] = apply_symbol_group_calibration(predictions, entry_scope_p_col, entry_group_layers)
        validation_for_candidate = predictions[predictions["split"].eq("validation_2023")].copy()
        validation_trade_ready = validation_for_candidate[validation_for_candidate[DECISION_ENTRY_COL].map(to_bool)].copy()
        trade_scope_layer = fit_scope_logit_shift(
            validation_trade_ready,
            p_col=tier_p_col,
            global_success=scoped_base_rate(train, TRADE_READY_EVAL_SCOPE, eb_model.global_success),
            scope_name=TRADE_READY_EVAL_SCOPE,
        )
        predictions[trade_scope_p_col] = apply_logit_shift(predictions[tier_p_col], trade_scope_layer)
        validation_for_candidate = predictions[predictions["split"].eq("validation_2023")].copy()
        validation_trade_ready = validation_for_candidate[validation_for_candidate[DECISION_ENTRY_COL].map(to_bool)].copy()
        trade_group_layers = fit_symbol_group_calibration(
            validation_trade_ready,
            trade_scope_p_col,
            scoped_base_rate(train, TRADE_READY_EVAL_SCOPE, eb_model.global_success),
        )
        predictions[trade_p_col] = apply_symbol_group_calibration(predictions, trade_scope_p_col, trade_group_layers)
        score_col = f"decision_score_{model_name}"
        trade_score_col = f"decision_score_{model_name}_trade_ready"
        validation_for_candidate = predictions[predictions["split"].eq("validation_2023")].copy()
        validation_trade_ready = validation_for_candidate[validation_for_candidate[DECISION_ENTRY_COL].map(to_bool)].copy()
        entry_policy = choose_utility_threshold(
            validation_for_candidate,
            p_col=p_col,
            score_col=score_col,
            base_rate=eb_model.global_success,
        )
        entry_weights = entry_policy["weights"]
        predictions[score_col] = utility_score_frame(predictions, p_col, "p_stop_hit_lgbm", "expected_r_lgbm", entry_weights)
        validation_for_candidate = predictions[predictions["split"].eq("validation_2023")].copy()
        validation_trade_ready = validation_for_candidate[validation_for_candidate[DECISION_ENTRY_COL].map(to_bool)].copy()
        entry_threshold_info = entry_policy["threshold_info"]
        trade_policy = choose_utility_threshold(
            validation_trade_ready,
            p_col=trade_p_col,
            score_col=trade_score_col,
            base_rate=scoped_base_rate(train, TRADE_READY_EVAL_SCOPE, eb_model.global_success),
        )
        trade_weights = trade_policy["weights"]
        predictions[trade_score_col] = utility_score_frame(predictions, trade_p_col, "p_stop_hit_lgbm", "expected_r_lgbm", trade_weights)
        validation_for_candidate = predictions[predictions["split"].eq("validation_2023")].copy()
        validation_trade_ready = validation_for_candidate[validation_for_candidate[DECISION_ENTRY_COL].map(to_bool)].copy()
        trade_threshold_info = choose_threshold_with_scope_fallback(
            validation_trade_ready,
            validation_for_candidate,
            entry_threshold_info,
            score_col=trade_score_col,
            stop_col="label_stop_hit_20d",
            probability_col=trade_p_col,
            base_rate=scoped_base_rate(train, TRADE_READY_EVAL_SCOPE, eb_model.global_success),
        )
        trade_threshold_info = attach_utility_weight_columns(trade_threshold_info, trade_weights)
        for evaluation_scope, threshold_info in [
            (ENTRY_RESEARCH_EVAL_SCOPE, entry_threshold_info),
            (TRADE_READY_EVAL_SCOPE, trade_threshold_info),
        ]:
            threshold_table = threshold_info["threshold_table"].copy()
            if not threshold_table.empty:
                threshold_table.insert(0, "model_name", model_name)
                threshold_table.insert(1, "evaluation_scope", evaluation_scope)
                threshold_table["threshold_reason"] = threshold_info["threshold_reason"]
                threshold_table["policy_role"] = "single_validation_threshold"
                threshold_tables.append(threshold_table)
        trade_threshold = safe_float(trade_threshold_info["threshold"])
        candidate_meta[model_name] = {
            "raw_col": raw_col,
            "global_p_col": global_p_col,
            "tier_p_col": tier_p_col,
            "entry_scope_p_col": entry_scope_p_col,
            "p_col": p_col,
            "trade_scope_p_col": trade_scope_p_col,
            "trade_p_col": trade_p_col,
            "score_col": score_col,
            "trade_score_col": trade_score_col,
            "threshold": trade_threshold,
            "entry_threshold": safe_float(entry_threshold_info["threshold"]),
            "trade_threshold": trade_threshold,
            "threshold_reason": trade_threshold_info["threshold_reason"],
            "entry_threshold_reason": entry_threshold_info["threshold_reason"],
            "trade_threshold_reason": trade_threshold_info["threshold_reason"],
            "threshold_decision_eligible": bool(trade_threshold_info.get("threshold_decision_eligible", False)),
            "entry_threshold_decision_eligible": bool(entry_threshold_info.get("threshold_decision_eligible", False)),
            "trade_threshold_decision_eligible": bool(trade_threshold_info.get("threshold_decision_eligible", False)),
            "calibration_method": calibration_method,
            "entry_utility_weights": entry_weights,
            "trade_utility_weights": trade_weights,
            "calibrator": calibrator,
            "tier_layers": tier_layers,
            "entry_scope_layer": entry_scope_layer,
            "entry_symbol_group_layers": entry_group_layers,
            "trade_scope_layer": trade_scope_layer,
            "trade_symbol_group_layers": trade_group_layers,
        }
        calibration_rows.append(
            {
                "layer": "global_probability_calibration",
                "model_name": model_name,
                "event_count": int(len(validation_for_candidate.dropna(subset=[raw_col, TARGET_COL]))),
                "actual_success_rate": float(validation_for_candidate[TARGET_COL].mean()) if not validation_for_candidate.empty else np.nan,
                "predicted_success_rate": float(validation_for_candidate[raw_col].mean()) if raw_col in validation_for_candidate.columns else np.nan,
                "posterior_success_rate": np.nan,
                "logit_shift": np.nan,
                "shrinkage": np.nan,
                "status": calibration_method,
                "fit_source": "validation_2023",
            }
        )
        for tier_name, layer in tier_layers.items():
            calibration_rows.append(
                {
                    "layer": "candidate_tier_logit_shift",
                    "model_name": model_name,
                    "candidate_tier": tier_name,
                    "event_count": int(layer.get("event_count", 0)),
                    "actual_success_rate": layer.get("actual_success_rate", np.nan),
                    "predicted_success_rate": layer.get("predicted_success_rate", np.nan),
                    "posterior_success_rate": layer.get("posterior_success_rate", np.nan),
                    "logit_shift": layer.get("shift", 0.0),
                    "shrinkage": layer.get("shrinkage", 0.0),
                    "status": layer.get("status", "INSUFFICIENT_SAMPLE"),
                    "fit_source": "validation_2023",
                }
            )
        for scope_name, layer, fit_source in [
            (ENTRY_RESEARCH_EVAL_SCOPE, entry_scope_layer, "validation_2023_entry_research_all"),
            (TRADE_READY_EVAL_SCOPE, trade_scope_layer, "validation_2023_trade_ready_entry_only"),
        ]:
            calibration_rows.append(
                {
                    "layer": "scope_logit_shift",
                    "model_name": model_name,
                    "evaluation_scope": scope_name,
                    "event_count": int(layer.get("event_count", 0)),
                    "actual_success_rate": layer.get("actual_success_rate", np.nan),
                    "predicted_success_rate": layer.get("predicted_success_rate", np.nan),
                    "posterior_success_rate": layer.get("posterior_success_rate", np.nan),
                    "logit_shift": layer.get("logit_shift", 0.0),
                    "shrinkage": layer.get("shrinkage", 0.0),
                    "prior_strength": layer.get("prior_strength", np.nan),
                    "status": layer.get("status", "UNKNOWN"),
                    "fit_source": fit_source,
                }
            )
        for scope_name, layers in [
            (ENTRY_RESEARCH_EVAL_SCOPE, entry_group_layers),
            (TRADE_READY_EVAL_SCOPE, trade_group_layers),
        ]:
            for group_name, layer in layers.items():
                calibration_rows.append(
                    {
                        "layer": "symbol_group_logit_shift",
                        "model_name": model_name,
                        "evaluation_scope": scope_name,
                        "symbol_group": group_name,
                        "event_count": int(layer.get("event_count", 0)),
                        "actual_success_rate": layer.get("actual_success_rate", np.nan),
                        "predicted_success_rate": layer.get("predicted_success_rate", np.nan),
                        "posterior_success_rate": layer.get("posterior_success_rate", np.nan),
                        "logit_shift": layer.get("shift", 0.0),
                        "shrinkage": layer.get("shrinkage", 0.0),
                        "status": layer.get("status", "INSUFFICIENT_SAMPLE"),
                        "fit_source": "validation_2023",
                    }
                )
        combined_eval = predictions[predictions["split"].isin(["test_2024", "final_holdout_2025_2026"])].copy()
        for split_name, frame in [
            ("train_2016_2022", predictions[predictions["split"].eq("train_2016_2022")]),
            ("validation_2023", predictions[predictions["split"].eq("validation_2023")]),
            ("test_2024", predictions[predictions["split"].eq("test_2024")]),
            ("final_holdout_2025_2026", predictions[predictions["split"].eq("final_holdout_2025_2026")]),
            ("combined_test_holdout", combined_eval),
        ]:
            for evaluation_scope, eval_frame in [
                (ENTRY_RESEARCH_EVAL_SCOPE, frame),
                (TRADE_READY_EVAL_SCOPE, frame[frame[DECISION_ENTRY_COL].map(to_bool)] if DECISION_ENTRY_COL in frame.columns else frame.head(0)),
            ]:
                metric_p_col = p_col if evaluation_scope == ENTRY_RESEARCH_EVAL_SCOPE else trade_p_col
                metric_score_col = score_col if evaluation_scope == ENTRY_RESEARCH_EVAL_SCOPE else trade_score_col
                threshold_info = entry_threshold_info if evaluation_scope == ENTRY_RESEARCH_EVAL_SCOPE else trade_threshold_info
                threshold = safe_float(threshold_info["threshold"])
                base_rate = scoped_base_rate(train, evaluation_scope, eb_model.global_success)
                row = metric_row(
                    split_name,
                    eval_frame,
                    metric_p_col,
                    threshold,
                    base_rate,
                    model_name,
                    score_col=metric_score_col,
                    evaluation_scope=evaluation_scope,
                )
                row["calibration_method"] = calibration_method
                row["raw_probability_col"] = raw_col
                row["threshold_reason"] = threshold_info["threshold_reason"]
                row["threshold_decision_eligible"] = bool(threshold_info.get("threshold_decision_eligible", False))
                row["trial_count"] = int(threshold_info.get("trial_count", row.get("trial_count", 0)) or 0)
                row["utility_weight_label"] = threshold_info.get("utility_weight_label", "")
                metric_rows.append(row)
        predictions = predictions.copy()

    single_comparison = pd.DataFrame(metric_rows)
    if single_comparison.empty:
        raise ValueError("no pooled model candidates were evaluated")
    single_comparison["validation_design"] = "single_2023_validation"
    oof_predictions, oof_comparison, oof_threshold_policy = run_walk_forward_oof(data)
    if not oof_threshold_policy.empty:
        threshold_tables.append(oof_threshold_policy)
    selection_comparison = oof_comparison.copy() if not oof_comparison.empty else single_comparison.copy()
    comparison = pd.concat(
        [frame for frame in [oof_comparison, single_comparison] if not frame.empty],
        ignore_index=True,
        sort=False,
    )
    combined_candidates = selection_comparison[
        selection_comparison["split"].eq("combined_test_holdout")
        & selection_comparison["evaluation_scope"].astype(str).eq(TRADE_READY_EVAL_SCOPE)
    ].copy()
    if combined_candidates.empty:
        combined_candidates = selection_comparison[selection_comparison["split"].eq("combined_test_holdout")].copy()
    selected_fraction = pd.to_numeric(combined_candidates["selected_fraction"], errors="coerce")
    combined_candidates["selected_fraction_distance"] = np.where(
        selected_fraction < MIN_SELECTED_FRACTION,
        MIN_SELECTED_FRACTION - selected_fraction,
        np.where(selected_fraction > MAX_SELECTED_FRACTION, selected_fraction - MAX_SELECTED_FRACTION, 0.0),
    )
    decision_ece = pd.to_numeric(combined_candidates.get("decision_ece", combined_candidates["ece"]), errors="coerce")
    decision_min_bin = pd.to_numeric(
        combined_candidates.get("decision_min_calibration_bin_n", combined_candidates.get("min_calibration_bin_n", pd.Series(0, index=combined_candidates.index))),
        errors="coerce",
    )
    selected_minus_all_lower = pd.to_numeric(
        combined_candidates.get("selected_minus_all_ci_lower_pct_paired", combined_candidates.get("selected_minus_all_ci_lower_pct", combined_candidates["selected_minus_all_pct"])),
        errors="coerce",
    )
    selected_minus_score_lower = pd.to_numeric(
        combined_candidates.get(
            "selected_minus_score_baseline_ci_lower_pct_paired",
            combined_candidates.get("selected_minus_score_baseline_ci_lower_pct", pd.Series(np.nan, index=combined_candidates.index)),
        ),
        errors="coerce",
    )
    positive_fold_count = pd.to_numeric(
        combined_candidates.get("positive_expectancy_fold_count", pd.Series(0, index=combined_candidates.index)),
        errors="coerce",
    ).fillna(0)
    validation_design_series = combined_candidates.get("validation_design", pd.Series("", index=combined_candidates.index)).astype(str)
    is_oof_candidate = validation_design_series.eq("walk_forward_oof")
    threshold_stability = combined_candidates.get("threshold_stability_pass", pd.Series(True, index=combined_candidates.index)).map(to_bool)
    threshold_stability = threshold_stability | ~is_oof_candidate
    trial_count = pd.to_numeric(combined_candidates.get("trial_count", pd.Series(0, index=combined_candidates.index)), errors="coerce").fillna(0)
    combined_candidates["candidate_gate_fail_count"] = (
        (pd.to_numeric(combined_candidates["event_count"], errors="coerce") < MIN_EVAL_EVENTS).astype(int)
        + (pd.to_numeric(combined_candidates["selected_event_count"], errors="coerce") < MIN_SELECTED_EVAL_EVENTS).astype(int)
        + (pd.to_numeric(combined_candidates.get("min_selected_events_per_fold", pd.Series(np.inf, index=combined_candidates.index)), errors="coerce") < MIN_SELECTED_PER_EVAL_SPLIT).astype(int)
        + (pd.to_numeric(combined_candidates["selected_expectancy_ci_lower_pct"], errors="coerce") <= MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT).astype(int)
        + (selected_minus_all_lower <= MIN_SELECTED_MINUS_ALL_PCT).astype(int)
        + (selected_minus_score_lower <= MIN_SELECTED_MINUS_ALL_PCT).astype(int)
        + (decision_ece > MAX_ECE).astype(int)
        + (decision_min_bin < MIN_TSM_CALIBRATION_EVENTS).astype(int)
        + (pd.to_numeric(combined_candidates["brier_improvement_pct"], errors="coerce") <= 0).astype(int)
        + ((selected_fraction < MIN_SELECTED_FRACTION) | (selected_fraction > MAX_SELECTED_FRACTION) | selected_fraction.isna()).astype(int)
        + (~combined_candidates.get("threshold_decision_eligible", pd.Series(False, index=combined_candidates.index)).map(to_bool)).astype(int)
        + (~threshold_stability).astype(int)
        + (trial_count <= 0).astype(int)
    )
    combined_candidates["candidate_quality_pass"] = (
        (pd.to_numeric(combined_candidates["event_count"], errors="coerce") >= MIN_EVAL_EVENTS)
        & (pd.to_numeric(combined_candidates["selected_event_count"], errors="coerce") >= MIN_SELECTED_EVAL_EVENTS)
        & (pd.to_numeric(combined_candidates.get("min_selected_events_per_fold", pd.Series(np.inf, index=combined_candidates.index)), errors="coerce") >= MIN_SELECTED_PER_EVAL_SPLIT)
        & (pd.to_numeric(combined_candidates["selected_expectancy_ci_lower_pct"], errors="coerce") > MIN_SELECTED_EXPECTANCY_CI_LOWER_PCT)
        & (selected_minus_all_lower > MIN_SELECTED_MINUS_ALL_PCT)
        & (selected_minus_score_lower > MIN_SELECTED_MINUS_ALL_PCT)
        & (decision_ece <= MAX_ECE)
        & (decision_min_bin >= MIN_TSM_CALIBRATION_EVENTS)
        & (pd.to_numeric(combined_candidates["brier_improvement_pct"], errors="coerce") > 0)
        & (selected_fraction >= MIN_SELECTED_FRACTION)
        & (selected_fraction <= MAX_SELECTED_FRACTION)
        & combined_candidates.get("threshold_decision_eligible", pd.Series(False, index=combined_candidates.index)).map(to_bool)
        & threshold_stability
        & (positive_fold_count >= 4)
        & (trial_count > 0)
    )
    champion_row = combined_candidates.sort_values(
        [
            "candidate_quality_pass",
            "candidate_gate_fail_count",
            "fold_uplift_ci_lower_min",
            "selected_minus_all_ci_lower_pct_paired",
            "fold_stop_improvement_min",
            "selected_fraction_distance",
            "brier_improvement_pct",
            "ece",
            "selected_minus_all_pct",
            "selected_expectancy_ci_lower_pct",
            "average_precision",
        ],
        ascending=[False, True, False, False, False, True, False, True, False, False, False],
    ).iloc[0]
    champion_name = str(champion_row["model_name"])
    champion_validation_design = str(champion_row.get("validation_design", "single_2023_validation"))
    comparison["is_champion"] = comparison["model_name"].astype(str).eq(champion_name)
    comparison = comparison.sort_values(["is_champion", "validation_design", "model_name", "split", "evaluation_scope"], ascending=[False, True, True, True, True]).reset_index(drop=True)
    champion_meta = candidate_meta[champion_name]
    champion_p_col = str(champion_meta["trade_p_col"])
    champion_score_col = str(champion_meta["trade_score_col"])
    champion_trade_weights = dict(champion_meta.get("trade_utility_weights", DEFAULT_UTILITY_WEIGHTS))
    threshold = safe_float(champion_meta["threshold"])
    threshold_decision_eligible = bool(to_bool(champion_row.get("threshold_decision_eligible", champion_meta.get("threshold_decision_eligible", False))))
    champion_comparison = comparison[
        comparison["model_name"].astype(str).eq(champion_name)
        & comparison["validation_design"].fillna("single_2023_validation").astype(str).eq(champion_validation_design)
    ].copy()

    tsm_layer_source = predictions[(predictions["symbol"].eq("TSM")) & (predictions["split"].isin(["train_2016_2022", "validation_2023"]))].copy()
    tsm_layer = fit_tsm_calibration_layer(tsm_layer_source, champion_p_col, eb_model.global_success)
    tsm_shrunk_layer = shrunk_tsm_layer(tsm_layer)
    predictions["p_success_calibrated"] = predictions[champion_p_col]
    predictions["decision_score"] = predictions[champion_score_col]
    tsm_metrics, tsm_route_summary, selected_tsm_route, selected_tsm_route_spec = build_tsm_calibration_route_metrics(
        predictions,
        source_col="p_success_calibrated",
        threshold=threshold,
        base_rate=eb_model.global_success,
        trade_weights=champion_trade_weights,
        tsm_layer=tsm_layer,
    )
    predictions["p_success_tsm_calibrated"] = predictions["p_success_calibrated"]
    tsm_mask = predictions["symbol"].eq("TSM")
    predictions.loc[tsm_mask, "p_success_tsm_calibrated"] = apply_tsm_calibration_route_spec(
        predictions.loc[tsm_mask],
        selected_tsm_route_spec,
        source_col="p_success_calibrated",
    )
    predictions["decision_score_tsm_calibrated"] = predictions["decision_score"]
    predictions.loc[tsm_mask, "decision_score_tsm_calibrated"] = utility_score_frame(
        predictions.loc[tsm_mask],
        "p_success_tsm_calibrated",
        "p_stop_hit_lgbm",
        "expected_r_lgbm",
        champion_trade_weights,
    )
    tsm_like_pool, tsm_like_metrics, selected_tsm_like_route_spec = build_tsm_like_calibration_artifacts(
        predictions,
        source_col="p_success_calibrated",
        base_rate=scoped_base_rate(train, TRADE_READY_EVAL_SCOPE, eb_model.global_success),
        trade_weights=champion_trade_weights,
    )

    reasons = block_reasons(champion_comparison, tsm_metrics, quality_ok, pooled_quality)
    model_quality_pass = reasons == "PASS"

    latest_source = enrich_pooled_features(features)
    latest_tsm = latest_source[latest_source["symbol"].astype(str).eq("TSM")].copy()
    latest_tsm["date"] = pd.to_datetime(latest_tsm["date"], errors="coerce")
    latest_tsm = latest_tsm.sort_values("date").tail(1)

    def add_latest_model_predictions(latest_frame: pd.DataFrame) -> pd.DataFrame:
        latest_pred = predict_empirical_bayes(eb_model, latest_frame)
        latest_pred["p_success_eb"] = latest_pred["p_success_base"]
        latest_pred["p_stop_hit_eb"] = latest_pred["p_stop_hit"]
        latest_pred["expected_r_eb"] = latest_pred["expected_r_net"]
        if "pooled_hierarchical_empirical_bayes" in fitted_candidates:
            latest_hier = predict_empirical_bayes(fitted_candidates["pooled_hierarchical_empirical_bayes"]["model"], latest_pred)
            latest_pred["p_success_hier_eb"] = latest_hier["p_success_base"]
        for col_name, candidate_name in [
            ("p_success_logistic", "pooled_elastic_net_logistic"),
            ("p_success_weighted_logistic", "pooled_weighted_elastic_net_logistic"),
            ("p_success_strict_logistic", "pooled_strict_elastic_net_logistic"),
            ("p_success_hist_gbm", "pooled_hist_gradient_boosting"),
            ("p_success_lgbm", "pooled_lgbm_classifier"),
            ("p_success_xgb", "pooled_xgb_classifier"),
        ]:
            if candidate_name in fitted_candidates:
                latest_pred[col_name] = predict_pooled_candidate(fitted_candidates[candidate_name], latest_pred)
        latest_pred["p_stop_hit_lgbm"] = (
            clip_probability(predict_pooled_candidate(fitted_candidates["pooled_stop_head"], latest_pred))
            if "pooled_stop_head" in fitted_candidates
            else latest_pred["p_stop_hit_eb"]
        )
        latest_pred["expected_r_lgbm"] = (
            np.asarray(predict_pooled_candidate(fitted_candidates["pooled_expected_r_head"], latest_pred), dtype=float)
            if "pooled_expected_r_head" in fitted_candidates
            else latest_pred["expected_r_eb"]
        )
        if "pooled_stack_calibrated" in fitted_candidates:
            latest_pred["p_success_stack_raw"] = predict_stack_candidate(fitted_candidates["pooled_stack_calibrated"], latest_pred)
        meta = candidate_meta[champion_name]
        raw_col = str(meta["raw_col"])
        global_p_col = str(meta["global_p_col"])
        tier_p_col = str(meta["tier_p_col"])
        entry_scope_p_col = str(meta["entry_scope_p_col"])
        p_col = str(meta["p_col"])
        trade_scope_p_col = str(meta["trade_scope_p_col"])
        trade_p_col = str(meta["trade_p_col"])
        calibration_method = str(meta["calibration_method"])
        latest_pred[global_p_col] = apply_probability_calibrator(latest_pred[raw_col], meta.get("calibrator"), calibration_method)
        latest_pred[tier_p_col] = apply_candidate_tier_calibration(latest_pred, global_p_col, meta.get("tier_layers", {}))
        latest_pred[entry_scope_p_col] = apply_logit_shift(latest_pred[tier_p_col], meta.get("entry_scope_layer", {}))
        latest_pred[p_col] = apply_symbol_group_calibration(latest_pred, entry_scope_p_col, meta.get("entry_symbol_group_layers", {}))
        latest_pred[trade_scope_p_col] = apply_logit_shift(latest_pred[tier_p_col], meta.get("trade_scope_layer", {}))
        latest_pred[trade_p_col] = apply_symbol_group_calibration(latest_pred, trade_scope_p_col, meta.get("trade_symbol_group_layers", {}))
        latest_pred["p_success_calibrated"] = latest_pred[trade_p_col]
        latest_pred["p_success_tsm_calibrated"] = apply_tsm_calibration_route_spec(
            latest_pred,
            selected_tsm_route_spec,
            source_col="p_success_calibrated",
        )
        latest_pred["p_success_tsm_like_calibrated"] = apply_tsm_like_calibration(
            latest_pred,
            "p_success_calibrated",
            selected_tsm_like_route_spec,
        )
        latest_pred["decision_score"] = utility_score_frame(latest_pred, "p_success_calibrated", "p_stop_hit_lgbm", "expected_r_lgbm", champion_trade_weights)
        latest_pred["decision_score_tsm_calibrated"] = utility_score_frame(latest_pred, "p_success_tsm_calibrated", "p_stop_hit_lgbm", "expected_r_lgbm", champion_trade_weights)
        latest_pred["decision_score_tsm_like_calibrated"] = utility_score_frame(latest_pred, "p_success_tsm_like_calibrated", "p_stop_hit_lgbm", "expected_r_lgbm", champion_trade_weights)
        return latest_pred

    latest_pred = add_latest_model_predictions(latest_tsm) if not latest_tsm.empty else pd.DataFrame()
    if not latest_pred.empty:
        latest_row = latest_pred.iloc[0]
        latest_trade_ready = to_bool(latest_row.get(TRADE_READY_COL))
        latest_p = safe_float(latest_row.get("p_success_tsm_calibrated"))
        latest_stop_hit = safe_float(latest_row.get("p_stop_hit_lgbm"))
        latest_expected_r = safe_float(latest_row.get("expected_r_lgbm"))
        latest_score = safe_float(latest_row.get("decision_score_tsm_calibrated"))
        latest_tsm_like_p = safe_float(latest_row.get("p_success_tsm_like_calibrated"))
        latest_tsm_like_score = safe_float(latest_row.get("decision_score_tsm_like_calibrated"))
        latest_signal_pass = bool(
            latest_trade_ready
            and pd.notna(latest_score)
            and pd.notna(threshold)
            and latest_score >= threshold
            and latest_stop_hit <= MAX_STOP_HIT_FOR_LATEST
            and latest_expected_r >= MIN_EXPECTED_R_FOR_LATEST
        )
        latest_allowed = bool(model_quality_pass and latest_signal_pass)
        latest_block_reasons = []
        if not latest_trade_ready:
            latest_block_reasons.append("LATEST_NOT_TRADE_READY")
        if pd.isna(latest_score) or pd.isna(threshold) or latest_score < threshold:
            latest_block_reasons.append("POOLED_DECISION_SCORE_BELOW_THRESHOLD")
        if pd.isna(latest_stop_hit) or latest_stop_hit > MAX_STOP_HIT_FOR_LATEST:
            latest_block_reasons.append("POOLED_STOP_RISK_GT_0_35")
        if pd.isna(latest_expected_r) or latest_expected_r < MIN_EXPECTED_R_FOR_LATEST:
            latest_block_reasons.append("POOLED_EXPECTED_R_LT_0_35")
    else:
        latest_trade_ready = False
        latest_signal_pass = False
        latest_allowed = False
        latest_block_reasons = ["MISSING_TSM_LATEST_ROW"]
        latest_row = pd.Series(dtype=object)
        latest_p = latest_stop_hit = latest_expected_r = latest_score = latest_tsm_like_p = latest_tsm_like_score = np.nan

    champion_decision_comparison = decision_scope_metrics(champion_comparison)
    eval_row = champion_decision_comparison[champion_decision_comparison["split"].eq("combined_test_holdout")].iloc[0]
    slice_diagnostics = build_slice_diagnostics(
        predictions,
        candidate_meta,
        train,
        eb_model.global_success,
        champion_name=champion_name,
    )
    calibration = pd.DataFrame(
        [
            *calibration_rows,
            {
                "layer": "tsm_specific_logit_shift",
                "model_name": champion_name,
                "event_count": tsm_layer.event_count,
                "actual_success_rate": tsm_layer.actual_success_rate,
                "predicted_success_rate": tsm_layer.predicted_success_rate,
                "posterior_success_rate": tsm_layer.posterior_success_rate,
                "logit_shift": tsm_layer.logit_shift,
                "shrinkage": tsm_layer.shrinkage,
                "status": tsm_layer.status,
                "fit_source": "train_2016_2022_plus_validation_2023",
            },
            {
                "layer": "tsm_specific_shrunk_logit_shift",
                "model_name": champion_name,
                "event_count": tsm_shrunk_layer.event_count,
                "actual_success_rate": tsm_shrunk_layer.actual_success_rate,
                "predicted_success_rate": tsm_shrunk_layer.predicted_success_rate,
                "posterior_success_rate": tsm_shrunk_layer.posterior_success_rate,
                "logit_shift": tsm_shrunk_layer.logit_shift,
                "shrinkage": tsm_shrunk_layer.shrinkage,
                "status": tsm_shrunk_layer.status,
                "fit_source": "train_2016_2022_plus_validation_2023",
            },
            *[
                {
                    "layer": "tsm_calibration_route_selection",
                    "model_name": champion_name,
                    "tsm_calibration_route": row.get("tsm_calibration_route"),
                    "event_count": int(tsm_layer.event_count),
                    "actual_success_rate": np.nan,
                    "predicted_success_rate": np.nan,
                    "posterior_success_rate": np.nan,
                    "logit_shift": np.nan,
                    "shrinkage": np.nan,
                    "status": row.get("tsm_calibration_route_failure_reasons", "UNKNOWN"),
                    "fit_source": row.get("route_selection_window", "train_2016_2022_plus_validation_2023"),
                    "combined_decision_ece": row.get("combined_decision_ece", np.nan),
                    "combined_brier_improvement_pct": row.get("combined_brier_improvement_pct", np.nan),
                    "max_test_holdout_ece": row.get("max_test_holdout_ece", np.nan),
                    "tsm_calibration_route_pass": row.get("tsm_calibration_route_pass", False),
                    "is_selected_tsm_calibration_route": row.get("is_selected_tsm_calibration_route", False),
                    "tsm_calibration_route_v2": row.get("tsm_calibration_route_v2", row.get("tsm_calibration_route")),
                    "route_selection_window": row.get("route_selection_window", ""),
                    "route_evaluation_window": row.get("route_evaluation_window", ""),
                    "route_selection_provenance_valid": row.get("route_selection_provenance_valid", False),
                    "route_alpha": row.get("route_alpha", np.nan),
                    "route_prior_source": row.get("route_prior_source", ""),
                    "route_sample_weight_policy": row.get("route_sample_weight_policy", ""),
                    "route_fail_count": row.get("route_fail_count", np.nan),
                    "selected_route_is_simplest_close_candidate": row.get("selected_route_is_simplest_close_candidate", False),
                }
                for row in (tsm_route_summary.to_dict("records") if isinstance(tsm_route_summary, pd.DataFrame) and not tsm_route_summary.empty else [])
            ],
            *skipped_candidates,
        ]
    )
    selected_tsm_eval_rows = (
        tsm_metrics[
            tsm_metrics.get("is_selected_tsm_calibration_route", pd.Series(False, index=tsm_metrics.index)).map(to_bool)
            & tsm_metrics["split"].astype(str).eq("tsm_combined_test_holdout")
        ]
        if isinstance(tsm_metrics, pd.DataFrame) and not tsm_metrics.empty
        else pd.DataFrame()
    )
    selected_tsm_eval = selected_tsm_eval_rows.iloc[0] if not selected_tsm_eval_rows.empty else pd.Series(dtype=object)
    paired_all_lower = eval_row.get("selected_minus_all_ci_lower_pct_paired", eval_row.get("selected_minus_all_ci_lower_pct", np.nan))
    paired_score_lower = eval_row.get("selected_minus_score_baseline_ci_lower_pct_paired", eval_row.get("selected_minus_score_baseline_ci_lower_pct", np.nan))
    pooled_threshold_failure_summary = (
        f"eligible={threshold_decision_eligible};stable={eval_row.get('threshold_stability_pass', np.nan)};"
        f"stable_candidates={eval_row.get('stable_threshold_candidate_count', np.nan)};"
        f"fold_selected_count_min={eval_row.get('fold_selected_count_min', np.nan)};"
        f"fold_selected_fraction={eval_row.get('fold_selected_fraction_min', np.nan)}..{eval_row.get('fold_selected_fraction_max', np.nan)};"
        f"fold_stop_improvement_min={eval_row.get('fold_stop_improvement_min', np.nan)};"
        f"provenance={eval_row.get('threshold_policy_provenance_valid', np.nan)};"
        f"reasons={eval_row.get('threshold_stability_failure_reasons', '')}"
    )
    pooled_uplift_failure_summary = (
        f"selected_minus_all_pct={eval_row.get('selected_minus_all_pct', np.nan)};"
        f"paired_all_lower={paired_all_lower};paired_score_lower={paired_score_lower};"
        f"fold_all_lower_min={eval_row.get('fold_uplift_ci_lower_min', np.nan)};"
        f"fold_score_lower_min={eval_row.get('fold_score_baseline_ci_lower_min', np.nan)};"
        f"positive_folds={eval_row.get('positive_expectancy_fold_count', np.nan)};"
        f"method={eval_row.get('bootstrap_method', '')};reasons={eval_row.get('uplift_failure_reasons', '')}"
    )
    pooled_tsm_calibration_failure_summary = (
        f"route={selected_tsm_route};pass={selected_tsm_eval.get('tsm_calibration_route_pass', False)};"
        f"ece={selected_tsm_eval.get('decision_ece', selected_tsm_eval.get('ece', np.nan))};"
        f"brier_improvement_pct={selected_tsm_eval.get('brier_improvement_pct', np.nan)};"
        f"max_test_holdout_ece={selected_tsm_eval.get('max_test_holdout_ece', np.nan)};"
        f"selection_provenance={selected_tsm_eval.get('route_selection_provenance_valid', np.nan)};"
        f"reasons={selected_tsm_eval.get('tsm_calibration_route_failure_reasons', '')}"
    )
    next_required_actions = []
    if not to_bool(eval_row.get("threshold_stability_pass", False)) or not threshold_decision_eligible:
        next_required_actions.append("improve_locked_risk_adjusted_rank_policy_or_expand_fold_trade_ready_events")
    if not to_bool(eval_row.get("uplift_pass", False)):
        next_required_actions.append("increase_oof_lower_bound_evidence_with_more_semiconductor_events")
    if not to_bool(selected_tsm_eval.get("tsm_calibration_route_pass", False)):
        next_required_actions.append("improve_tsm_route_calibration_or_expand_tsm_like_calibration_sample")
    next_required_evidence_action = "|".join(next_required_actions) if next_required_actions else "PASS"
    overlay = {
        "model_name": champion_name,
        "asof_date": latest_row.get("date", ""),
        "model_quality_pass": model_quality_pass,
        "latest_trade_ready": latest_trade_ready,
        "latest_signal_pass": latest_signal_pass,
        "tsm_calibration_status": tsm_layer.status,
        "tsm_calibration_route": selected_tsm_route,
        "tsm_calibration_route_v2": selected_tsm_route,
        "tsm_like_calibration_route": selected_tsm_like_route_spec.route,
        "route_selection_provenance_valid": selected_tsm_eval.get("route_selection_provenance_valid", np.nan),
        "route_prior_source": selected_tsm_eval.get("route_prior_source", ""),
        "route_sample_weight_policy": selected_tsm_eval.get("route_sample_weight_policy", ""),
        "route_alpha": selected_tsm_eval.get("route_alpha", np.nan),
        "tsm_calibration_route_pass": bool(
            not tsm_metrics.empty
            and tsm_metrics.loc[
                tsm_metrics["is_selected_tsm_calibration_route"].map(to_bool)
                & tsm_metrics["split"].astype(str).eq("tsm_combined_test_holdout"),
                "tsm_calibration_route_pass",
            ].map(to_bool).any()
        ),
        "decision_support_allowed": latest_allowed,
        "model_support_route": "POOLED_SEMI_DECISION_SUPPORT" if latest_allowed else "DISPLAY_ONLY_NO_MODEL_CANDIDATE",
        "decision_block_reasons": "PASS" if latest_allowed else "|".join(([f"MODEL:{reasons}"] if not model_quality_pass else []) + latest_block_reasons),
        "model_quality_block_reasons": reasons,
        "latest_block_reasons": "PASS" if latest_signal_pass else "|".join(latest_block_reasons),
        "threshold_20d": threshold,
        "threshold_reason": champion_row.get("threshold_reason", champion_meta["threshold_reason"]),
        "threshold_decision_eligible": threshold_decision_eligible,
        "threshold_stability_pass": eval_row.get("threshold_stability_pass", np.nan),
        "threshold_policy_type": eval_row.get("threshold_policy_type", ""),
        "applied_threshold_policy_type": eval_row.get("applied_threshold_policy_type", eval_row.get("threshold_policy_type", "")),
        "diagnostic_best_candidate_policy_type": eval_row.get("diagnostic_best_candidate_policy_type", ""),
        "threshold_policy_source_window": eval_row.get("threshold_policy_source_window", ""),
        "threshold_policy_applied_window": eval_row.get("threshold_policy_applied_window", ""),
        "threshold_policy_provenance_valid": eval_row.get("threshold_policy_provenance_valid", np.nan),
        "risk_adjusted_selection_score_col": eval_row.get("risk_adjusted_selection_score_col", ""),
        "stable_threshold_candidate_count": eval_row.get("stable_threshold_candidate_count", np.nan),
        "fold_selected_fraction_min": eval_row.get("fold_selected_fraction_min", np.nan),
        "fold_selected_fraction_max": eval_row.get("fold_selected_fraction_max", np.nan),
        "fold_selected_count_min": eval_row.get("fold_selected_count_min", np.nan),
        "fold_stop_improvement_min": eval_row.get("fold_stop_improvement_min", np.nan),
        "threshold_stability_failure_reasons": eval_row.get("threshold_stability_failure_reasons", ""),
        "weak_oof_folds": eval_row.get("weak_oof_folds", ""),
        "threshold_score_col": champion_score_col,
        "utility_weight_label": utility_weight_label(champion_trade_weights),
        "training_candidate_scope": ENTRY_RESEARCH_EVAL_SCOPE,
        "decision_candidate_scope": TRADE_READY_EVAL_SCOPE,
        "validation_design": champion_validation_design,
        "training_event_count": model_training_label_count,
        "decision_event_count": decision_label_count,
        "p_success_20d": latest_p,
        "p_success_tsm_like_20d": latest_tsm_like_p,
        "decision_score_20d": latest_score,
        "paper_decision_score_20d": latest_tsm_like_score,
        "p_stop_survival_20d": 1.0 - latest_stop_hit if pd.notna(latest_stop_hit) else np.nan,
        "p_stop_hit_20d": latest_stop_hit,
        "p_hit_1r_20d": latest_row.get("p_hit_1r", np.nan),
        "p_hit_2r_20d": latest_row.get("p_hit_2r", np.nan),
        "expected_r_net_20d": latest_expected_r,
        "expected_net_return_pct_20d": latest_row.get("expected_net_return_pct", np.nan),
        "effective_group_n": latest_row.get("effective_group_n", 0),
        "oos_event_count": int(eval_row.get("event_count", 0)),
        "selected_oos_event_count": int(eval_row.get("selected_event_count", 0)),
        "selected_fraction": eval_row.get("selected_fraction", np.nan),
        "selected_expectancy_ci_lower_pct": eval_row.get("selected_expectancy_ci_lower_pct", np.nan),
        "selected_minus_all_pct": eval_row.get("selected_minus_all_pct", np.nan),
        "selected_minus_all_ci_lower_pct": eval_row.get("selected_minus_all_ci_lower_pct", np.nan),
        "selected_minus_all_ci_lower_pct_paired": eval_row.get("selected_minus_all_ci_lower_pct_paired", np.nan),
        "selected_minus_score_baseline_ci_lower_pct": eval_row.get("selected_minus_score_baseline_ci_lower_pct", np.nan),
        "selected_minus_score_baseline_ci_lower_pct_paired": eval_row.get("selected_minus_score_baseline_ci_lower_pct_paired", np.nan),
        "uplift_bootstrap_p_value": eval_row.get("uplift_bootstrap_p_value", np.nan),
        "uplift_bootstrap_p_value_paired": eval_row.get("uplift_bootstrap_p_value_paired", np.nan),
        "bootstrap_method": eval_row.get("bootstrap_method", ""),
        "bootstrap_block_col": eval_row.get("bootstrap_block_col", ""),
        "fold_uplift_ci_lower_min": eval_row.get("fold_uplift_ci_lower_min", np.nan),
        "fold_score_baseline_ci_lower_min": eval_row.get("fold_score_baseline_ci_lower_min", np.nan),
        "fold_positive_uplift_count": eval_row.get("fold_positive_uplift_count", np.nan),
        "fold_positive_score_baseline_uplift_count": eval_row.get("fold_positive_score_baseline_uplift_count", np.nan),
        "uplift_pass": eval_row.get("uplift_pass", np.nan),
        "uplift_failure_reasons": eval_row.get("uplift_failure_reasons", ""),
        "uplift_failure_reasons_detail": eval_row.get("uplift_failure_reasons_detail", ""),
        "score_baseline_policy": eval_row.get("score_baseline_policy", ""),
        "positive_expectancy_fold_count": eval_row.get("positive_expectancy_fold_count", np.nan),
        "trial_count": eval_row.get("trial_count", np.nan),
        "ece": eval_row.get("ece", np.nan),
        "decision_ece": eval_row.get("decision_ece", eval_row.get("ece", np.nan)),
        "decision_min_calibration_bin_n": eval_row.get("decision_min_calibration_bin_n", np.nan),
        "calibration_binning_primary": eval_row.get("calibration_binning_primary", calibration_binning_primary_core()),
        "brier_improvement_pct": eval_row.get("brier_improvement_pct", np.nan),
        "average_precision": eval_row.get("average_precision", np.nan),
        "pooled_threshold_failure_summary": pooled_threshold_failure_summary,
        "pooled_uplift_failure_summary": pooled_uplift_failure_summary,
        "pooled_tsm_calibration_failure_summary": pooled_tsm_calibration_failure_summary,
        "next_required_evidence_action": next_required_evidence_action,
    }
    paper_gate_snapshot = build_paper_gate_snapshot(champion_comparison, tsm_like_metrics, overlay, quality_ok)
    paper_gate_values = dict(zip(paper_gate_snapshot["field"], paper_gate_snapshot["value"])) if not paper_gate_snapshot.empty else {}
    overlay.update(
        {
            "strict_gate_status": paper_gate_values.get("strict_gate_status", "DISPLAY_ONLY_MODEL_BLOCKED"),
            "paper_gate_status": paper_gate_values.get("paper_gate_status", "DISPLAY_ONLY_MODEL_BLOCKED"),
            "paper_decision_support_allowed": paper_gate_values.get("paper_decision_support_allowed", False),
            "paper_model_gate_pass": paper_gate_values.get("paper_model_gate_pass", False),
            "paper_latest_signal_pass": paper_gate_values.get("paper_latest_signal_pass", False),
            "paper_gate_block_reasons": paper_gate_values.get("paper_gate_block_reasons", "UNKNOWN"),
            "tsm_like_effective_train_validation_n": paper_gate_values.get("tsm_like_effective_train_validation_n", np.nan),
            "tsm_like_calibration_ece": paper_gate_values.get("tsm_like_calibration_ece", np.nan),
            "live_trading_status": "DISABLED_BY_DESIGN",
        }
    )
    quality = build_quality_checks(data, champion_comparison, tsm_metrics, calibration, overlay, quality_ok)
    prediction_cols = [
        "p_success_base",
        "p_success_eb",
        "p_success_hier_eb",
        "p_success_logistic",
        "p_success_weighted_logistic",
        "p_success_strict_logistic",
        "p_success_hist_gbm",
        "p_success_lgbm",
        "p_success_xgb",
        "p_success_stack_raw",
        "p_success_calibrated",
        "p_success_tsm_calibrated",
        "p_success_tsm_like_calibrated",
        "p_stop_hit",
        "p_stop_hit_lgbm",
        "expected_r_net",
        "expected_r_lgbm",
        "expected_net_return_pct",
        "decision_score",
        "decision_score_tsm_calibrated",
        "decision_score_tsm_like_calibrated",
        "effective_group_n",
    ]
    predictions_out_cols = [
        "symbol",
        "symbol_group",
        "date",
        "split",
        "signal_idx",
        MODEL_TRAINING_COL,
        DECISION_ENTRY_COL,
        TRADE_READY_COL,
        "candidate_tier",
        TARGET_COL,
        RETURN_COL,
        EXPECTED_R_COL,
        STOP_SURVIVAL_COL,
        *[c for c in prediction_cols if c in predictions.columns],
        *GROUP_COLS,
    ]
    predictions_out = predictions[list(dict.fromkeys([c for c in predictions_out_cols if c in predictions.columns]))].copy()
    threshold_policy = pd.concat(threshold_tables, ignore_index=True) if threshold_tables else pd.DataFrame()
    return {
        "comparison": comparison,
        "tsm_metrics": tsm_metrics,
        "threshold_policy": threshold_policy,
        "uplift_report": build_uplift_bootstrap_report(comparison),
        "oos_predictions": predictions_out,
        "oof_predictions": oof_predictions,
        "slice_diagnostics": slice_diagnostics,
        "tsm_like_pool": tsm_like_pool,
        "tsm_like_metrics": tsm_like_metrics,
        "paper_gate_snapshot": paper_gate_snapshot,
        "learning_curve": build_learning_curve_report(data, comparison, tsm_like_metrics),
        "calibration": calibration,
        "overlay": overlay,
        "quality": quality,
    }


def build_quality_checks(
    data: pd.DataFrame,
    comparison: pd.DataFrame,
    tsm_metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    overlay: Dict[str, object],
    pooled_dataset_quality_ok: bool,
) -> pd.DataFrame:
    comparison = decision_scope_metrics(comparison)
    eval_row = comparison[comparison["split"].eq("combined_test_holdout")].iloc[0] if not comparison.empty else pd.Series(dtype=object)
    if "is_selected_tsm_calibration_route" in tsm_metrics.columns:
        selected_tsm_metrics = tsm_metrics[tsm_metrics["is_selected_tsm_calibration_route"].map(to_bool)].copy()
        if not selected_tsm_metrics.empty:
            tsm_metrics = selected_tsm_metrics
    tsm_eval = tsm_metrics[tsm_metrics["split"].eq("tsm_combined_test_holdout")].iloc[0] if not tsm_metrics.empty else pd.Series(dtype=object)
    tsm_layer_rows = calibration[calibration["layer"].astype(str).eq("tsm_specific_logit_shift")] if not calibration.empty and "layer" in calibration.columns else pd.DataFrame()
    tsm_layer_row = tsm_layer_rows.iloc[0] if not tsm_layer_rows.empty else pd.Series(dtype=object)
    model_training_labels = int(safe_float(overlay.get("training_event_count"), len(data)))
    trade_ready_labels = int(safe_float(overlay.get("decision_event_count"), data[DECISION_ENTRY_COL].map(to_bool).sum() if DECISION_ENTRY_COL in data.columns else len(data)))
    selected_minus_all_lower = safe_float(eval_row.get("selected_minus_all_ci_lower_pct_paired", eval_row.get("selected_minus_all_ci_lower_pct")))
    selected_minus_score_lower = safe_float(eval_row.get("selected_minus_score_baseline_ci_lower_pct_paired", eval_row.get("selected_minus_score_baseline_ci_lower_pct")))
    rows = [
        quality_check("pooled_dataset_quality_critical_pass", pooled_dataset_quality_ok, "CRITICAL", pooled_dataset_quality_ok),
        quality_check("pooled_model_training_20d_labels_at_least_10000", model_training_labels >= MIN_POOLED_MODEL_TRAINING_LABELS, "CRITICAL", model_training_labels),
        quality_check("pooled_trade_ready_20d_labels_at_least_500", trade_ready_labels >= MIN_POOLED_TRADE_READY_LABELS, "CRITICAL", trade_ready_labels),
        quality_check("pooled_model_eval_events_at_least_150", int(eval_row.get("event_count", 0)) >= MIN_EVAL_EVENTS, "CRITICAL", int(eval_row.get("event_count", 0))),
        quality_check("pooled_model_selected_eval_events_at_least_50", int(eval_row.get("selected_event_count", 0)) >= MIN_SELECTED_EVAL_EVENTS, "WARN", int(eval_row.get("selected_event_count", 0))),
        quality_check("pooled_model_min_selected_events_per_oof_fold_at_least_10", safe_float(eval_row.get("min_selected_events_per_fold", np.inf)) >= MIN_SELECTED_PER_EVAL_SPLIT, "WARN", eval_row.get("min_selected_events_per_fold", np.nan)),
        quality_check("pooled_model_threshold_decision_eligible", to_bool(eval_row.get("threshold_decision_eligible", False)), "WARN", eval_row.get("threshold_reason", "UNKNOWN")),
        quality_check("pooled_model_threshold_stability_pass", to_bool(eval_row.get("threshold_stability_pass", True)), "WARN", eval_row.get("weak_oof_folds", "")),
        quality_check("pooled_model_selected_fraction_30_to_60_pct", MIN_SELECTED_FRACTION <= safe_float(eval_row.get("selected_fraction")) <= MAX_SELECTED_FRACTION, "WARN", eval_row.get("selected_fraction", np.nan)),
        quality_check("pooled_model_selected_ci_lower_positive", safe_float(eval_row.get("selected_expectancy_ci_lower_pct")) > 0, "WARN", eval_row.get("selected_expectancy_ci_lower_pct", np.nan)),
        quality_check("pooled_model_selected_minus_all_positive", safe_float(eval_row.get("selected_minus_all_pct")) > 0, "WARN", eval_row.get("selected_minus_all_pct", np.nan)),
        quality_check("pooled_model_selected_minus_all_ci_lower_positive", selected_minus_all_lower > 0, "WARN", selected_minus_all_lower, "Gate primary uses paired/block bootstrap when available."),
        quality_check("pooled_model_selected_minus_score_baseline_ci_lower_positive", selected_minus_score_lower > 0, "WARN", selected_minus_score_lower, "Gate primary uses paired/block bootstrap when available."),
        quality_check("pooled_model_economic_uplift_pass", to_bool(eval_row.get("uplift_pass", False)), "WARN", eval_row.get("uplift_failure_reasons", "")),
        quality_check("pooled_model_decision_ece_at_most_0_10", safe_float(eval_row.get("decision_ece", eval_row.get("ece"))) <= MAX_ECE, "WARN", eval_row.get("decision_ece", eval_row.get("ece", np.nan))),
        quality_check("pooled_model_decision_calibration_bin_at_least_30", safe_float(eval_row.get("decision_min_calibration_bin_n")) >= MIN_TSM_CALIBRATION_EVENTS, "WARN", eval_row.get("decision_min_calibration_bin_n", np.nan)),
        quality_check("pooled_model_fixed_width_calibration_bin_at_least_30", safe_float(eval_row.get("fixed_width_min_calibration_bin_n")) >= MIN_TSM_CALIBRATION_EVENTS, "WARN", eval_row.get("fixed_width_min_calibration_bin_n", np.nan), "Fixed-width binning is diagnostic only; adaptive binning is the decision gate."),
        quality_check("pooled_model_trial_ledger_present", safe_float(eval_row.get("trial_count"), 0) > 0, "WARN", eval_row.get("trial_count", np.nan)),
        quality_check("pooled_model_brier_improvement_positive", safe_float(eval_row.get("brier_improvement_pct")) > 0, "WARN", eval_row.get("brier_improvement_pct", np.nan)),
        quality_check("tsm_calibration_layer_events_at_least_30", int(tsm_layer_row.get("event_count", 0)) >= MIN_TSM_CALIBRATION_EVENTS, "WARN", int(tsm_layer_row.get("event_count", 0))),
        quality_check("tsm_calibrated_eval_events_at_least_30", int(tsm_eval.get("event_count", 0)) >= MIN_TSM_EVAL_EVENTS, "WARN", int(tsm_eval.get("event_count", 0))),
        quality_check("tsm_calibration_route_pass", to_bool(tsm_eval.get("tsm_calibration_route_pass", False)), "WARN", tsm_eval.get("tsm_calibration_route_failure_reasons", "")),
        quality_check("tsm_calibrated_ece_at_most_0_15", safe_float(tsm_eval.get("decision_ece", tsm_eval.get("ece"))) <= MAX_TSM_ECE, "WARN", tsm_eval.get("decision_ece", tsm_eval.get("ece", np.nan))),
        quality_check("tsm_like_effective_train_validation_n_at_least_500", safe_float(overlay.get("tsm_like_effective_train_validation_n"), 0.0) >= MIN_TSM_LIKE_EFFECTIVE_SELECTION_N, "WARN", overlay.get("tsm_like_effective_train_validation_n", np.nan)),
        quality_check("tsm_like_calibration_ece_at_most_0_15", safe_float(overlay.get("tsm_like_calibration_ece"), np.nan) <= MAX_TSM_ECE, "WARN", overlay.get("tsm_like_calibration_ece", np.nan)),
        quality_check("paper_only_gate_pass", to_bool(overlay.get("paper_decision_support_allowed", False)), "INFO", overlay.get("paper_gate_block_reasons", ""), "Paper gate does not enable strict prediction_ready or live trading."),
        quality_check("pooled_latest_decision_support_allowed", to_bool(overlay.get("decision_support_allowed", False)), "INFO", overlay.get("decision_support_allowed", False), "False is expected when latest TSM is not trade-ready or model gates fail."),
    ]
    return pd.DataFrame(rows)


def write_report(outdir: Path, comparison: pd.DataFrame, tsm_metrics: pd.DataFrame, calibration: pd.DataFrame, quality: pd.DataFrame, overlay: Dict[str, object]) -> None:
    lines = [
        "# Pooled Semiconductor Model Report",
        "",
        "This report evaluates only 20D trade-ready events. It is research tooling and does not place orders.",
        "",
        "## Latest Overlay",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for key, value in overlay.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Pooled Metrics", "", "| Split | Scope | Events | Selected | Brier Improvement | ECE | Selected Mean Return | Selected CI Lower |"])
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['split']} | {row.get('evaluation_scope', 'NA')} | {int(row['event_count'])} | {int(row['selected_event_count'])} | "
            f"{row['brier_improvement_pct']:.2f}% | {row['ece']:.4f} | {row['selected_mean_return_pct']:.2f}% | {row['selected_expectancy_ci_lower_pct']:.2f}% |"
        )
    lines.extend(["", "## TSMC Calibration Metrics", "", "| Split | Events | Selected | Brier Improvement | ECE | Selected Mean Return |"])
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in tsm_metrics.iterrows():
        lines.append(
            f"| {row['split']} | {int(row['event_count'])} | {int(row['selected_event_count'])} | "
            f"{row['brier_improvement_pct']:.2f}% | {row['ece']:.4f} | {row['selected_mean_return_pct']:.2f}% |"
        )
    lines.extend(["", "## Calibration Layer", "", "| Layer | Events | Status | Logit Shift | Shrinkage | Fit Source |", "|---|---:|---|---:|---:|---|"])
    for _, row in calibration.iterrows():
        event_count = safe_float(row.get("event_count", 0), 0.0)
        lines.append(
            f"| {row.get('layer')} | {int(event_count if pd.notna(event_count) else 0)} | {row.get('status')} | "
            f"{safe_float(row.get('logit_shift')):.4f} | {safe_float(row.get('shrinkage')):.4f} | {row.get('fit_source')} |"
        )
    lines.extend(["", "## Quality Checks", "", "| Check | Passed | Severity | Value | Details |", "|---|---:|---|---:|---|"])
    for _, row in quality.iterrows():
        lines.append(f"| {row.get('check')} | {row.get('passed')} | {row.get('severity')} | {row.get('value')} | {row.get('details', '')} |")
    (outdir / "tsm_pooled_model_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pooled 20D trade-ready model and TSMC calibration overlay.")
    parser.add_argument("--pooled-feature-matrix", default="tsm_price_rule_output/tsm_prediction_pooled_feature_matrix.csv")
    parser.add_argument("--pooled-quality", default="tsm_price_rule_output/tsm_prediction_pooled_quality_checks.csv")
    parser.add_argument("--latest-prediction", default="tsm_price_rule_output/tsm_latest_prediction_snapshot.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--no-update-latest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    features = read_csv(Path(args.pooled_feature_matrix), parse_dates=["date"])
    pooled_quality = read_csv(Path(args.pooled_quality))
    results = run_pooled_model(features, pooled_quality)
    comparison = results["comparison"]
    tsm_metrics = results["tsm_metrics"]
    threshold_policy = results["threshold_policy"]
    uplift_report = results["uplift_report"]
    oos_predictions = results["oos_predictions"]
    oof_predictions = results["oof_predictions"]
    slice_diagnostics = results["slice_diagnostics"]
    tsm_like_pool = results["tsm_like_pool"]
    tsm_like_metrics = results["tsm_like_metrics"]
    paper_gate_snapshot = results["paper_gate_snapshot"]
    learning_curve = results["learning_curve"]
    calibration = results["calibration"]
    overlay = results["overlay"]
    quality = results["quality"]
    comparison.to_csv(outdir / "tsm_pooled_model_comparison.csv", index=False)
    tsm_metrics.to_csv(outdir / "tsm_pooled_tsm_calibration_metrics.csv", index=False)
    threshold_policy.to_csv(outdir / "tsm_pooled_model_threshold_policy.csv", index=False)
    uplift_report.to_csv(outdir / "tsm_pooled_uplift_bootstrap_report.csv", index=False)
    oos_predictions.to_csv(outdir / "tsm_pooled_model_oos_predictions.csv", index=False)
    oof_predictions.to_csv(outdir / "tsm_pooled_model_oof_predictions.csv", index=False)
    slice_diagnostics.to_csv(outdir / "tsm_pooled_model_slice_diagnostics.csv", index=False)
    tsm_like_pool.to_csv(outdir / "tsm_tsm_like_calibration_pool.csv", index=False)
    tsm_like_metrics.to_csv(outdir / "tsm_tsm_like_calibration_metrics.csv", index=False)
    paper_gate_snapshot.to_csv(outdir / "tsm_paper_gate_snapshot.csv", index=False)
    learning_curve.to_csv(outdir / "tsm_pooled_learning_curve_report.csv", index=False)
    calibration.to_csv(outdir / "tsm_pooled_tsm_calibration.csv", index=False)
    build_overlay_rows(overlay).to_csv(outdir / "tsm_pooled_latest_prediction_overlay.csv", index=False)
    quality.to_csv(outdir / "tsm_pooled_model_quality_checks.csv", index=False)
    write_report(outdir, comparison, tsm_metrics, calibration, quality, overlay)
    if not args.no_update_latest:
        update_latest_snapshot(Path(args.latest_prediction), overlay)
    print("완료: pooled model outputs =", outdir.resolve())
    print(build_overlay_rows(overlay).to_string(index=False))
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build multi-horizon next-close forecasts for the Top12 decision universe.

The target is a direct log close return:
    log(close_engine[t + h] / close_engine[t])

Outputs:
- tsm_next_close_label_dataset.csv
- tsm_next_close_feature_matrix.csv
- tsm_next_close_feature_selection_report.csv
- tsm_next_close_walk_forward_metrics.csv
- tsm_next_close_oos_predictions.csv
- tsm_next_close_model_comparison.csv
- tsm_next_close_interval_calibration.csv
- tsm_next_close_latest_snapshot.csv
- tsm_next_close_universe_latest_predictions.csv
- tsm_next_close_quality_checks.csv
- tsm_next_close_report.md
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only when env is broken
    ColumnTransformer = None
    HistGradientBoostingRegressor = None
    SimpleImputer = None
    ElasticNet = None
    Pipeline = None
    OneHotEncoder = None
    StandardScaler = None
    SKLEARN_IMPORT_ERROR = exc

try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only when env is broken
    LGBMRegressor = None
    LIGHTGBM_IMPORT_ERROR = exc

try:
    from xgboost import XGBRegressor

    XGBOOST_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only when env is broken
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = exc


OUTPUT_PREFIX = "tsm_next_close"
DEFAULT_HORIZONS = (1, 5, 20)
NEXT_CLOSE_SCOPE = "next_close_forecast_top12"
LATEST_PROMOTED_STATUS = "PROMOTED_FORECAST_DIAGNOSTIC"
LATEST_DISPLAY_ONLY_STATUS = "DISPLAY_ONLY_FORECAST_MODEL_QUALITY_NOT_PASSED"

MIN_OOS_EVENTS_FOR_PROMOTION = 5000
MIN_FOLDS_FOR_PROMOTION = 4
INTERVAL_COVERAGE_LOW = 0.75
INTERVAL_COVERAGE_HIGH = 0.85
CONFORMAL_INTERVAL_TARGET = 0.80
CONFORMAL_RESIDUAL_QUANTILE = 0.85
MIN_MAE_IMPROVEMENT_PCT = 1.0

FORBIDDEN_FEATURE_PATTERNS = (
    "label_",
    "next_",
    "future",
    "fwd_",
    "forward",
    "exit_",
    "net_return",
    "gross_return",
    "actual_return",
    "r_multiple",
)
META_FEATURE_COLS = {
    "date",
    "date_split",
    "signal_idx",
    "split",
    "label_status",
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
INTRADAY_PREFIXES = (
    "hourly_",
    "model_minute_",
    "execution_minute_",
    "m5_",
    "m1_",
    "intraday_",
    "timeframe_",
)
SYMBOL_WINDOW_BASELINE_SPECS = (
    ("symbol_all_mean_shrink_065", 0, "mean", 0.65),
    ("symbol_all_mean_shrink_075", 0, "mean", 0.75),
    ("symbol_all_median_shrink_065", 0, "median", 0.65),
    ("symbol_all_median_shrink_075", 0, "median", 0.75),
    ("symbol_all_median_shrink_085", 0, "median", 0.85),
    ("symbol_63d_median_shrink_065", 63, "median", 0.65),
    ("symbol_63d_median_shrink_075", 63, "median", 0.75),
    ("symbol_126d_median_shrink_075", 126, "median", 0.75),
    ("symbol_504d_median_shrink_075", 504, "median", 0.75),
)
MARKET_RECENT_BASELINE_SPECS = (
    ("market_recent_1d_mean_shrink_010", 1, "mean", 0.10),
    ("market_recent_1d_median_shrink_020", 1, "median", 0.20),
    ("market_recent_40d_mean_shrink_050", 40, "mean", 0.50),
)
MARKET_SIGN_MAGNITUDE_SPECS = (
    ("market_recent_1d_mean_sign_mag_0018", 1, "mean", 0.0018),
)


@dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class FittedCandidate:
    name: str
    feature_cols: list[str]
    model: Any | None
    kind: str
    lower_model: Any | None = None
    upper_model: Any | None = None
    val_mae: float = np.nan
    conformal_adjustment: float = 0.0
    component_weights: dict[str, float] | None = None
    component_models: dict[str, "FittedCandidate"] | None = None


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    kwargs.setdefault("low_memory", False)
    return pd.read_csv(path, **kwargs)


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def to_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "enabled", "enable"}


def as_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def finite_positive(value: object) -> bool:
    out = as_float(value)
    return math.isfinite(out) and out > 0.0


def check_row(check: str, passed: bool, severity: str, value: object, tolerance: str = "", details: str = "") -> dict[str, object]:
    return {
        "check": check,
        "passed": bool(passed),
        "severity": severity,
        "value": value,
        "tolerance": tolerance,
        "details": details,
    }


def key_value_frame_to_dict(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty or not {"field", "value"}.issubset(frame.columns):
        return {}
    return dict(zip(frame["field"].astype(str), frame["value"]))


def merge_latest_prediction_snapshot(latest_path: Path, latest: pd.DataFrame) -> bool:
    if not latest_path.exists():
        return False
    existing = read_csv(latest_path)
    values = key_value_frame_to_dict(existing)
    if not values:
        return False
    values.update(key_value_frame_to_dict(latest))
    pd.DataFrame([{"field": key, "value": value} for key, value in values.items()]).to_csv(latest_path, index=False)
    return True


def load_decision_symbols(config_path: Path) -> list[str]:
    config = read_csv(config_path)
    if config.empty:
        raise ValueError(f"decision universe config is empty or missing: {config_path}")
    if "enabled" in config.columns:
        config = config[config["enabled"].map(to_bool)].copy()
    if "symbol" not in config.columns:
        raise ValueError("decision universe config missing symbol column")
    symbols = [str(symbol).strip().upper() for symbol in config["symbol"].tolist() if str(symbol).strip()]
    return list(dict.fromkeys(symbols))


def merge_decision_metadata(frame: pd.DataFrame, config_path: Path) -> pd.DataFrame:
    config = read_csv(config_path)
    if config.empty or "symbol" not in config.columns:
        return frame.copy()
    if "enabled" in config.columns:
        config = config[config["enabled"].map(to_bool)].copy()
    config["symbol"] = config["symbol"].astype(str).str.upper()
    metadata_cols = [
        col
        for col in ["symbol_group", "listing_currency", "display_currency", "engine_currency", "fx_pair"]
        if col in config.columns and col not in frame.columns
    ]
    if not metadata_cols:
        return frame.copy()
    return frame.merge(config[["symbol", *metadata_cols]].drop_duplicates("symbol"), on="symbol", how="left")


def add_engine_price_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    close_candidates = [
        "close_engine",
        "close",
        "close_usd",
        "close_native",
        "daily_close",
        "adj_close",
        "adjusted_close",
        "label_entry_price_20d",
        "label_entry_price_5d",
        "label_entry_price_60d",
        "label_entry_price_10d",
        "label_entry_price_40d",
        "label_entry_price_120d",
        "hourly_close",
        "model_minute_close",
        "m5_close",
        "execution_minute_close",
        "m1_close",
    ]
    for col in [*close_candidates, "fx_rate_to_usd"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    current_close = pd.Series(np.nan, index=work.index, dtype=float)
    source = pd.Series("", index=work.index, dtype=object)
    for col in close_candidates:
        if col not in work.columns:
            continue
        values = pd.to_numeric(work[col], errors="coerce")
        fill_mask = current_close.isna() & values.gt(0) & values.notna()
        current_close.loc[fill_mask] = values.loc[fill_mask]
        source.loc[fill_mask] = col
    work["close_engine"] = current_close
    work["close_engine_source"] = source.replace("", "missing")
    if "fx_rate_to_usd" not in work.columns:
        work["fx_rate_to_usd"] = np.nan
    fx = pd.to_numeric(work["fx_rate_to_usd"], errors="coerce")
    if "close_usd" not in work.columns:
        if "close_native" in work.columns:
            native = pd.to_numeric(work["close_native"], errors="coerce")
            work["close_usd"] = np.where(fx.gt(0), native * fx, work["close_engine"])
        else:
            work["close_usd"] = work["close_engine"]
    if "close_native" not in work.columns:
        symbol = work.get("symbol", pd.Series("", index=work.index)).astype(str).str.upper()
        display_currency = work.get("display_currency", pd.Series("", index=work.index)).astype(str).str.upper()
        listing_currency = work.get("listing_currency", pd.Series("", index=work.index)).astype(str).str.upper()
        krw_mask = symbol.str.endswith(".KS") | display_currency.eq("KRW") | listing_currency.eq("KRW")
        close_usd = pd.to_numeric(work["close_usd"], errors="coerce")
        work["close_native"] = np.where(krw_mask & fx.gt(0), close_usd / fx, work["close_engine"])
    if "display_currency" not in work.columns:
        work["display_currency"] = np.where(work.get("symbol", pd.Series("", index=work.index)).astype(str).str.upper().str.endswith(".KS"), "KRW", "USD")
    return work


def build_next_close_label_dataset(pooled: pd.DataFrame, horizons: Iterable[int] = DEFAULT_HORIZONS) -> pd.DataFrame:
    require_columns(pooled, ["symbol", "date"], "pooled feature matrix")
    work = add_engine_price_columns(pooled)
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    sort_cols = ["symbol", "date"] + (["signal_idx"] if "signal_idx" in work.columns else [])
    work = work.sort_values(sort_cols).reset_index(drop=True)

    for horizon in horizons:
        target_engine = work.groupby("symbol", sort=False)["close_engine"].shift(-int(horizon))
        target_usd = work.groupby("symbol", sort=False)["close_usd"].shift(-int(horizon))
        target_native = work.groupby("symbol", sort=False)["close_native"].shift(-int(horizon))
        target_date = work.groupby("symbol", sort=False)["date"].shift(-int(horizon))
        current_engine = pd.to_numeric(work["close_engine"], errors="coerce")
        valid_current = current_engine.gt(0) & current_engine.notna()
        valid_target = pd.to_numeric(target_engine, errors="coerce").gt(0) & pd.to_numeric(target_engine, errors="coerce").notna()
        labeled = valid_current & valid_target & target_date.notna()

        status = pd.Series("LABELED", index=work.index, dtype=object)
        status.loc[target_date.isna()] = "UNAVAILABLE_FUTURE_WINDOW"
        status.loc[target_date.notna() & ~(valid_current & valid_target)] = "INVALID_CLOSE_DATA"
        return_log = pd.Series(np.nan, index=work.index, dtype=float)
        return_log.loc[labeled] = np.log(target_engine.loc[labeled].astype(float) / current_engine.loc[labeled].astype(float))

        suffix = f"{int(horizon)}d"
        work[f"label_status_{suffix}"] = status
        work[f"label_entry_date_{suffix}"] = work["date"]
        work[f"label_target_date_{suffix}"] = target_date
        work[f"label_entry_close_engine_{suffix}"] = current_engine
        work[f"label_close_return_log_{suffix}"] = return_log
        work[f"label_close_return_pct_{suffix}"] = (np.exp(return_log) - 1.0) * 100.0
        work[f"label_target_close_engine_{suffix}"] = target_engine.where(labeled)
        work[f"label_target_close_usd_{suffix}"] = target_usd.where(labeled)
        work[f"label_target_close_native_{suffix}"] = target_native.where(labeled)
    return work


def build_next_close_feature_matrix(pooled: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return labels.copy()
    return labels.copy()


def is_forbidden_feature(col: str) -> bool:
    lower = str(col).lower()
    return any(pattern in lower for pattern in FORBIDDEN_FEATURE_PATTERNS)


def is_intraday_feature(col: str) -> bool:
    lower = str(col).lower()
    return lower.startswith(INTRADAY_PREFIXES) or lower in {"daily_signal_available"}


def intraday_feature_has_sufficient_coverage(series: pd.Series) -> bool:
    lower = str(series.name).lower()
    if lower.endswith(("_feature_status", "_source_provider")) or lower in {"intraday_coverage_class", "intraday_feature_status"}:
        return int(series.notna().sum()) > 0
    return int(series.notna().sum()) >= min(20, max(2, int(len(series) * 0.01)))


def feature_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return "numeric"
    return "categorical"


def candidate_feature_columns(frame: pd.DataFrame, *, include_forbidden: bool = False) -> list[str]:
    blocked = set(META_FEATURE_COLS) | FILTER_ONLY_COLS
    cols: list[str] = []
    for col in frame.columns:
        lower = str(col).lower()
        if col in blocked or lower in blocked:
            continue
        if is_forbidden_feature(col) and not include_forbidden:
            continue
        if lower in {"symbol", "date"}:
            continue
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col]) or frame[col].dtype == object or pd.api.types.is_string_dtype(frame[col]):
            cols.append(col)
    return cols


def safe_abs_corr(x: pd.Series, y: pd.Series) -> float:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    mask = x_num.notna() & y_num.notna()
    if int(mask.sum()) < 5 or x_num.loc[mask].nunique() < 2 or y_num.loc[mask].nunique() < 2:
        return 0.0
    corr = x_num.loc[mask].corr(y_num.loc[mask])
    return abs(float(corr)) if pd.notna(corr) and math.isfinite(float(corr)) else 0.0


def select_fold_features(
    train: pd.DataFrame,
    target_col: str | None = None,
    *,
    max_numeric_features: int = 110,
    max_categorical_features: int = 30,
) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    candidates = candidate_feature_columns(train, include_forbidden=True)
    numeric_rank: list[tuple[float, str]] = []
    categorical_rank: list[tuple[float, str]] = []
    selected_initial: set[str] = set()

    for col in candidates:
        lower = str(col).lower()
        series = train[col]
        missing_rate = float(series.isna().mean()) if len(series) else 1.0
        ftype = feature_type(series)
        reason = "selected"
        selected = True
        if is_forbidden_feature(col):
            selected = False
            reason = "future_or_label_feature_blocked"
        elif missing_rate > 0.50 and not (is_intraday_feature(col) and intraday_feature_has_sufficient_coverage(series)):
            selected = False
            reason = "too_sparse"
        elif ftype == "numeric":
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.nunique(dropna=True) <= 1:
                selected = False
                reason = "constant_numeric"
        else:
            nunique = int(series.astype(str).replace("nan", np.nan).nunique(dropna=True))
            if nunique <= 1:
                selected = False
                reason = "constant_categorical"
            elif nunique > max(60, int(len(series) * 0.25)):
                selected = False
                reason = "high_cardinality_categorical"

        if selected:
            selected_initial.add(col)
            score = 0.0
            if ftype == "numeric" and target_col and target_col in train.columns:
                score = safe_abs_corr(series, train[target_col])
            coverage_score = 1.0 - missing_rate
            intraday_bonus = 0.12 if is_intraday_feature(col) else 0.0
            score = score + (0.02 * coverage_score) + intraday_bonus
            if ftype == "numeric":
                numeric_rank.append((score, col))
            else:
                categorical_rank.append((score + (0.05 if col == "symbol_group" else 0.0), col))
        rows.append(
            {
                "feature": col,
                "feature_type": ftype,
                "missing_rate": missing_rate,
                "intraday_feature": is_intraday_feature(col),
                "selected": selected,
                "reason": reason,
                "feature_selection_source": "train_only",
            }
        )

    numeric_selected = [col for _, col in sorted(numeric_rank, key=lambda item: (-item[0], item[1]))[:max_numeric_features]]
    categorical_selected = [col for _, col in sorted(categorical_rank, key=lambda item: (-item[0], item[1]))[:max_categorical_features]]
    selected_final = set(numeric_selected + categorical_selected)
    for row in rows:
        if row["feature"] in selected_initial and row["feature"] not in selected_final:
            row["selected"] = False
            row["reason"] = "feature_cap"
    return numeric_selected + categorical_selected, pd.DataFrame(rows)


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility for older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(train: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    if SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError(f"scikit-learn is required for next-close forecasts: {SKLEARN_IMPORT_ERROR}")
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
    )


def parse_horizons(value: str) -> list[int]:
    horizons = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not horizons:
        raise ValueError("at least one horizon is required")
    return horizons


def build_walk_forward_folds(
    frame: pd.DataFrame,
    *,
    train_days: int,
    validation_days: int,
    test_days: int,
    step_days: int,
    gap_days: int,
) -> list[Fold]:
    dates = pd.Series(pd.to_datetime(frame["date"], errors="coerce").dropna().unique()).sort_values().reset_index(drop=True)
    total = len(dates)
    folds: list[Fold] = []
    fold_id = 1
    val_start = int(train_days)
    while val_start + validation_days + gap_days < total:
        val_end = min(val_start + validation_days, total)
        test_start = val_end + int(gap_days)
        test_end = min(test_start + int(test_days), total)
        if test_start >= test_end:
            break
        train_dates = dates.iloc[:val_start]
        val_dates = dates.iloc[val_start:val_end]
        test_dates = dates.iloc[test_start:test_end]
        if not train_dates.empty and not val_dates.empty and not test_dates.empty:
            folds.append(
                Fold(
                    fold_id=fold_id,
                    train_start=pd.Timestamp(train_dates.iloc[0]),
                    train_end=pd.Timestamp(train_dates.iloc[-1]),
                    calibration_start=pd.Timestamp(val_dates.iloc[0]),
                    calibration_end=pd.Timestamp(val_dates.iloc[-1]),
                    test_start=pd.Timestamp(test_dates.iloc[0]),
                    test_end=pd.Timestamp(test_dates.iloc[-1]),
                )
            )
            fold_id += 1
        if test_end >= total:
            break
        val_start += int(step_days)
    return folds


def date_slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"], errors="coerce")
    return frame[dates.ge(start) & dates.le(end)].copy()


def target_col(horizon: int) -> str:
    return f"label_close_return_log_{int(horizon)}d"


def baseline_recent_symbol_mean(train: pd.DataFrame, rows: pd.DataFrame, y_col: str) -> np.ndarray:
    global_mean = as_float(pd.to_numeric(train[y_col], errors="coerce").mean(), 0.0)
    symbol_mean = pd.to_numeric(train[y_col], errors="coerce").groupby(train["symbol"].astype(str)).mean().to_dict()
    return np.array([as_float(symbol_mean.get(str(symbol), global_mean), global_mean) for symbol in rows["symbol"]], dtype=float)


def baseline_symbol_window_return(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    y_col: str,
    *,
    window: int,
    mode: str,
    shrink: float,
) -> np.ndarray:
    require_columns(history, ["symbol", "date", y_col], "symbol window baseline history")
    require_columns(rows, ["symbol"], "symbol window baseline rows")
    work = history[["symbol", "date", y_col]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["date", y_col]).sort_values(["symbol", "date"])
    if work.empty:
        return np.zeros(len(rows), dtype=float)
    if int(window) > 0:
        work = work.groupby("symbol", sort=False).tail(int(window))
    if str(mode).lower() == "median":
        global_value = as_float(work[y_col].median(), 0.0)
        by_symbol = work.groupby(work["symbol"].astype(str))[y_col].median().to_dict()
    else:
        global_value = as_float(work[y_col].mean(), 0.0)
        by_symbol = work.groupby(work["symbol"].astype(str))[y_col].mean().to_dict()
    values = [as_float(by_symbol.get(str(symbol), global_value), global_value) for symbol in rows["symbol"]]
    return float(shrink) * np.asarray(values, dtype=float)


def symbol_window_spec_by_name(name: str) -> tuple[int, str, float] | None:
    for spec_name, window, mode, shrink in SYMBOL_WINDOW_BASELINE_SPECS:
        if spec_name == name:
            return int(window), str(mode), float(shrink)
    return None


def baseline_market_recent_return(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    y_col: str,
    *,
    days: int,
    mode: str,
    shrink: float,
) -> np.ndarray:
    require_columns(history, ["date", y_col], "market recent baseline history")
    work = history[["date", y_col]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["date", y_col]).sort_values("date")
    if work.empty:
        return np.zeros(len(rows), dtype=float)
    unique_dates = pd.Series(work["date"].dropna().unique()).sort_values()
    if int(days) > 0:
        keep_dates = set(unique_dates.tail(int(days)).tolist())
        work = work[work["date"].isin(keep_dates)]
    if str(mode).lower() == "median":
        value = as_float(work[y_col].median(), 0.0)
    else:
        value = as_float(work[y_col].mean(), 0.0)
    return np.full(len(rows), float(shrink) * value, dtype=float)


def market_recent_spec_by_name(name: str) -> tuple[int, str, float] | None:
    for spec_name, days, mode, shrink in MARKET_RECENT_BASELINE_SPECS:
        if spec_name == name:
            return int(days), str(mode), float(shrink)
    return None


def baseline_market_recent_sign_magnitude(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    y_col: str,
    *,
    days: int,
    mode: str,
    magnitude: float,
) -> np.ndarray:
    recent = baseline_market_recent_return(history, rows, y_col, days=days, mode=mode, shrink=1.0)
    sign = np.sign(recent)
    sign = np.where(sign == 0, 1.0, sign)
    return float(magnitude) * sign


def market_sign_magnitude_spec_by_name(name: str) -> tuple[int, str, float] | None:
    for spec_name, days, mode, magnitude in MARKET_SIGN_MAGNITUDE_SPECS:
        if spec_name == name:
            return int(days), str(mode), float(magnitude)
    return None


def historical_majority_direction_sign(train: pd.DataFrame, rows: pd.DataFrame, y_col: str) -> np.ndarray:
    y_train = pd.to_numeric(train[y_col], errors="coerce")
    valid = y_train.notna()
    if not valid.any():
        return np.ones(len(rows), dtype=float)
    global_positive_rate = float((y_train.loc[valid] > 0).mean())
    global_sign = 1.0 if global_positive_rate >= 0.50 else -1.0
    symbol_rate = (
        y_train.loc[valid]
        .groupby(train.loc[valid, "symbol"].astype(str))
        .apply(lambda series: 1.0 if float((series > 0).mean()) >= 0.50 else -1.0)
        .to_dict()
    )
    return np.asarray([symbol_rate.get(str(symbol), global_sign) for symbol in rows["symbol"]], dtype=float)


def historical_majority_direction_accuracy(train: pd.DataFrame, rows: pd.DataFrame, y_col: str) -> float:
    if rows.empty:
        return np.nan
    actual = np.sign(pd.to_numeric(rows[y_col], errors="coerce").to_numpy(dtype=float))
    pred = historical_majority_direction_sign(train, rows, y_col)
    mask = np.isfinite(actual) & np.isfinite(pred)
    return float((actual[mask] == pred[mask]).mean()) if mask.any() else np.nan


def baseline_group_trend_vol(train: pd.DataFrame, rows: pd.DataFrame, y_col: str) -> np.ndarray:
    global_mean = as_float(pd.to_numeric(train[y_col], errors="coerce").mean(), 0.0)
    keys = [c for c in ["symbol_group", "trend_regime", "vol_regime"] if c in train.columns and c in rows.columns]
    if not keys:
        return np.full(len(rows), global_mean, dtype=float)
    grouped = pd.to_numeric(train[y_col], errors="coerce").groupby([train[c].astype(str) for c in keys]).mean()
    group_mean = grouped.to_dict()
    symbol_group_mean = (
        pd.to_numeric(train[y_col], errors="coerce").groupby(train["symbol_group"].astype(str)).mean().to_dict()
        if "symbol_group" in train.columns and "symbol_group" in rows.columns
        else {}
    )
    preds: list[float] = []
    for _, row in rows.iterrows():
        key = tuple(str(row.get(c, "UNKNOWN")) for c in keys)
        value = group_mean.get(key)
        if value is None and "symbol_group" in rows.columns:
            value = symbol_group_mean.get(str(row.get("symbol_group", "UNKNOWN")))
        preds.append(as_float(value, global_mean))
    return np.array(preds, dtype=float)


def prediction_frame(
    rows: pd.DataFrame,
    *,
    horizon: int,
    fold: Fold,
    model_name: str,
    pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    interval_type: str,
) -> pd.DataFrame:
    out = rows.copy()
    y_col = target_col(horizon)
    pred = np.asarray(pred, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    current_engine = pd.to_numeric(out["close_engine"], errors="coerce").to_numpy(dtype=float)
    current_usd = pd.to_numeric(out["close_usd"], errors="coerce").to_numpy(dtype=float)
    current_native = pd.to_numeric(out["close_native"], errors="coerce").to_numpy(dtype=float)
    fx = pd.to_numeric(out.get("fx_rate_to_usd", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(dtype=float)
    pred_engine = current_engine * np.exp(pred)
    lower_engine = current_engine * np.exp(lower)
    upper_engine = current_engine * np.exp(upper)
    pred_usd = np.where(np.isfinite(current_usd) & (current_usd > 0), current_usd * np.exp(pred), pred_engine)
    pred_native = np.where(
        np.isfinite(current_native) & (current_native > 0),
        current_native * np.exp(pred),
        np.where(np.isfinite(fx) & (fx > 0), pred_usd / fx, pred_engine),
    )
    actual_engine = pd.to_numeric(out[f"label_target_close_engine_{int(horizon)}d"], errors="coerce").to_numpy(dtype=float)
    atr_pct = pd.to_numeric(out.get("atr_14_pct", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(out[y_col], errors="coerce").to_numpy(dtype=float)
    abs_log_error = np.abs(y - pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        close_abs_pct_error = np.abs(actual_engine / pred_engine - 1.0)
    display_currency = out.get("display_currency", pd.Series("USD", index=out.index)).astype(str).str.upper()
    display_pred = np.where(display_currency.eq("KRW").to_numpy(), pred_native, pred_usd)
    display_lower = np.where(display_currency.eq("KRW").to_numpy(), current_native * np.exp(lower), np.where(np.isfinite(current_usd), current_usd * np.exp(lower), lower_engine))
    display_upper = np.where(display_currency.eq("KRW").to_numpy(), current_native * np.exp(upper), np.where(np.isfinite(current_usd), current_usd * np.exp(upper), upper_engine))
    result = pd.DataFrame(
        {
            "candidate_scope": NEXT_CLOSE_SCOPE,
            "horizon_days": int(horizon),
            "fold_id": fold.fold_id,
            "model_name": model_name,
            "symbol": out["symbol"].astype(str).to_numpy(),
            "symbol_group": out.get("symbol_group", pd.Series("", index=out.index)).astype(str).to_numpy(),
            "date": out["date"].to_numpy(),
            "train_start_date": fold.train_start,
            "train_end_date": fold.train_end,
            "calibration_start_date": fold.calibration_start,
            "calibration_end_date": fold.calibration_end,
            "test_start_date": fold.test_start,
            "test_end_date": fold.test_end,
            "actual_return_log": y,
            "actual_return_pct": (np.exp(y) - 1.0) * 100.0,
            "predicted_return_log": pred,
            "predicted_return_pct": (np.exp(pred) - 1.0) * 100.0,
            "lower_80_return_log": lower,
            "upper_80_return_log": upper,
            "current_close_engine": current_engine,
            "actual_close_engine": actual_engine,
            "predicted_close_engine": pred_engine,
            "predicted_close_usd": pred_usd,
            "predicted_close_native": pred_native,
            "predicted_close": display_pred,
            "lower_80_close_engine": lower_engine,
            "upper_80_close_engine": upper_engine,
            "lower_80_close": display_lower,
            "upper_80_close": display_upper,
            "interval_type": interval_type,
            "interval_hit_80": (y >= lower) & (y <= upper),
            "actual_direction": np.sign(y),
            "predicted_direction": np.sign(pred),
            "direction_hit": np.sign(y) == np.sign(pred),
            "abs_log_error": abs_log_error,
            "squared_log_error": (y - pred) ** 2,
            "close_abs_pct_error": close_abs_pct_error,
            "within_1atr": np.where(np.isfinite(atr_pct) & (atr_pct > 0), abs_log_error <= atr_pct, np.nan),
            "within_2pct": close_abs_pct_error <= 0.02,
        }
    )
    if "signal_idx" in out.columns:
        result["signal_idx"] = out["signal_idx"].to_numpy()
    return result


def calibration_quantile_level(coverage: float = CONFORMAL_INTERVAL_TARGET) -> float:
    return max(float(coverage), float(CONFORMAL_RESIDUAL_QUANTILE))


def finite_sample_quantile(values: np.ndarray, level: float) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return 0.0
    q = min(1.0, max(0.0, float(level)))
    try:
        return float(np.nanquantile(clean, q, method="higher"))
    except TypeError:  # pragma: no cover - numpy compatibility
        return float(np.nanquantile(clean, q, interpolation="higher"))


def conformal_interval(y_val: np.ndarray, pred_val: np.ndarray, pred_test: np.ndarray, coverage: float = CONFORMAL_INTERVAL_TARGET) -> tuple[np.ndarray, np.ndarray, float]:
    residual = np.abs(np.asarray(y_val, dtype=float) - np.asarray(pred_val, dtype=float))
    width = finite_sample_quantile(residual, calibration_quantile_level(coverage))
    pred_test = np.asarray(pred_test, dtype=float)
    return pred_test - width, pred_test + width, width


def quantile_conformal_interval(
    y_val: np.ndarray,
    lower_val: np.ndarray,
    upper_val: np.ndarray,
    lower_test: np.ndarray,
    upper_test: np.ndarray,
    coverage: float = CONFORMAL_INTERVAL_TARGET,
) -> tuple[np.ndarray, np.ndarray, float]:
    y_val = np.asarray(y_val, dtype=float)
    lower_val = np.asarray(lower_val, dtype=float)
    upper_val = np.asarray(upper_val, dtype=float)
    nonconformity = np.maximum.reduce([lower_val - y_val, y_val - upper_val, np.zeros_like(y_val)])
    adjustment = finite_sample_quantile(nonconformity, calibration_quantile_level(coverage))
    return np.asarray(lower_test, dtype=float) - adjustment, np.asarray(upper_test, dtype=float) + adjustment, adjustment


def fit_pipeline(name: str, estimator: Any, train: pd.DataFrame, feature_cols: list[str], y_col: str) -> FittedCandidate:
    preprocessor = build_preprocessor(train, feature_cols)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(train[feature_cols], pd.to_numeric(train[y_col], errors="coerce").to_numpy(dtype=float))
    return FittedCandidate(name=name, feature_cols=feature_cols, model=pipeline, kind="pipeline")


def predict_candidate(candidate: FittedCandidate, rows: pd.DataFrame) -> np.ndarray:
    if candidate.kind == "baseline":
        raise ValueError("baseline candidates are predicted directly in fold evaluation")
    if candidate.kind == "blend":
        assert candidate.component_weights is not None and candidate.component_models is not None
        pred = np.zeros(len(rows), dtype=float)
        for name, weight in candidate.component_weights.items():
            pred += float(weight) * predict_candidate(candidate.component_models[name], rows)
        return pred
    return np.asarray(candidate.model.predict(rows[candidate.feature_cols]), dtype=float)


def fit_model_candidates(train: pd.DataFrame, feature_cols: list[str], y_col: str, *, random_state: int = 17) -> list[FittedCandidate]:
    candidates: list[FittedCandidate] = []
    if SKLEARN_IMPORT_ERROR is not None or not feature_cols or len(train) < 20:
        return candidates
    try:
        candidates.append(
            fit_pipeline(
                "elastic_net_return",
                ElasticNet(alpha=0.0005, l1_ratio=0.20, max_iter=5000, random_state=random_state),
                train,
                feature_cols,
                y_col,
            )
        )
    except Exception:
        pass
    try:
        candidates.append(
            fit_pipeline(
                "hist_gbr_huber",
                HistGradientBoostingRegressor(loss="absolute_error", max_iter=160, learning_rate=0.045, l2_regularization=0.01, random_state=random_state),
                train,
                feature_cols,
                y_col,
            )
        )
    except Exception:
        pass
    if LGBMRegressor is not None and len(train) >= 50:
        for name, objective in [("lgbm_huber", "huber"), ("lgbm_l1", "regression_l1")]:
            try:
                candidates.append(
                    fit_pipeline(
                        name,
                        LGBMRegressor(
                            objective=objective,
                            n_estimators=260,
                            learning_rate=0.035,
                            num_leaves=31,
                            min_child_samples=30,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            random_state=random_state,
                            verbosity=-1,
                        ),
                        train,
                        feature_cols,
                        y_col,
                    )
                )
            except Exception:
                pass
        try:
            preprocessor = build_preprocessor(train, feature_cols)
            x_train = preprocessor.fit_transform(train[feature_cols])
            y_train = pd.to_numeric(train[y_col], errors="coerce").to_numpy(dtype=float)
            lower_model = LGBMRegressor(
                objective="quantile",
                alpha=0.10,
                n_estimators=220,
                learning_rate=0.035,
                num_leaves=31,
                min_child_samples=30,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                verbosity=-1,
            )
            median_model = LGBMRegressor(
                objective="quantile",
                alpha=0.50,
                n_estimators=220,
                learning_rate=0.035,
                num_leaves=31,
                min_child_samples=30,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state + 1,
                verbosity=-1,
            )
            upper_model = LGBMRegressor(
                objective="quantile",
                alpha=0.90,
                n_estimators=220,
                learning_rate=0.035,
                num_leaves=31,
                min_child_samples=30,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state + 2,
                verbosity=-1,
            )
            lower_model.fit(x_train, y_train)
            median_model.fit(x_train, y_train)
            upper_model.fit(x_train, y_train)
            median_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", median_model)])
            candidates.append(
                FittedCandidate(
                    name="lgbm_quantile_10_50_90",
                    feature_cols=feature_cols,
                    model=median_pipeline,
                    kind="quantile_pipeline",
                    lower_model=lower_model,
                    upper_model=upper_model,
                )
            )
        except Exception:
            pass
    if XGBRegressor is not None and len(train) >= 80:
        try:
            candidates.append(
                fit_pipeline(
                    "xgb_quantile",
                    XGBRegressor(
                        objective="reg:quantileerror",
                        quantile_alpha=0.5,
                        n_estimators=180,
                        max_depth=4,
                        learning_rate=0.035,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=random_state,
                        n_jobs=2,
                    ),
                    train,
                    feature_cols,
                    y_col,
                )
            )
        except Exception:
            pass
    return candidates


def predict_quantile_bounds(candidate: FittedCandidate, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if candidate.kind != "quantile_pipeline" or candidate.lower_model is None or candidate.upper_model is None:
        pred = predict_candidate(candidate, rows)
        return pred.copy(), pred.copy()
    preprocessor = candidate.model.named_steps["preprocessor"]
    matrix = preprocessor.transform(rows[candidate.feature_cols])
    return np.asarray(candidate.lower_model.predict(matrix), dtype=float), np.asarray(candidate.upper_model.predict(matrix), dtype=float)


def fold_metrics_from_predictions(preds: pd.DataFrame) -> dict[str, object]:
    if preds.empty:
        return {}
    y = pd.to_numeric(preds["actual_return_log"], errors="coerce")
    pred = pd.to_numeric(preds["predicted_return_log"], errors="coerce")
    mask = y.notna() & pred.notna()
    if not mask.any():
        return {}
    clean = preds.loc[mask].copy()
    mae = float(clean["abs_log_error"].mean())
    rmse = float(np.sqrt(clean["squared_log_error"].mean()))
    direction_accuracy = float(clean["direction_hit"].mean())
    interval_coverage = float(clean["interval_hit_80"].mean()) if "interval_hit_80" in clean.columns else np.nan
    within_1atr = pd.to_numeric(clean.get("within_1atr", pd.Series(dtype=float)), errors="coerce")
    within_2pct = pd.to_numeric(clean.get("within_2pct", pd.Series(dtype=float)), errors="coerce")
    return {
        "oos_event_count": int(len(clean)),
        "mae_log_return": mae,
        "rmse_log_return": rmse,
        "direction_accuracy": direction_accuracy,
        "interval_coverage_80": interval_coverage,
        "within_1atr_rate": float(within_1atr.mean()) if within_1atr.notna().any() else np.nan,
        "within_2pct_rate": float(within_2pct.mean()) if within_2pct.notna().any() else np.nan,
    }


def evaluate_horizon(
    feature_matrix: pd.DataFrame,
    horizon: int,
    *,
    train_days: int,
    validation_days: int,
    test_days: int,
    step_days: int,
    max_numeric_features: int,
    max_categorical_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_col = target_col(horizon)
    label_status_col = f"label_status_{int(horizon)}d"
    labeled = feature_matrix[feature_matrix[label_status_col].eq("LABELED")].copy()
    labeled[y_col] = pd.to_numeric(labeled[y_col], errors="coerce")
    labeled = labeled[labeled[y_col].notna()].copy()
    folds = build_walk_forward_folds(
        labeled,
        train_days=train_days,
        validation_days=validation_days,
        test_days=test_days,
        step_days=step_days,
        gap_days=max(1, int(horizon)),
    )
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[pd.DataFrame] = []

    for fold in folds:
        train = date_slice(labeled, fold.train_start, fold.train_end)
        validation = date_slice(labeled, fold.calibration_start, fold.calibration_end)
        test = date_slice(labeled, fold.test_start, fold.test_end)
        if train.empty or validation.empty or test.empty:
            continue
        feature_cols, selection = select_fold_features(
            train,
            y_col,
            max_numeric_features=max_numeric_features,
            max_categorical_features=max_categorical_features,
        )
        if not selection.empty:
            selection["horizon_days"] = int(horizon)
            selection["fold_id"] = fold.fold_id
            selection_rows.append(selection)

        y_val = pd.to_numeric(validation[y_col], errors="coerce").to_numpy(dtype=float)
        candidate_predictions: dict[str, dict[str, Any]] = {}

        baseline_defs = {
            "naive_last_close": (
                np.zeros(len(validation), dtype=float),
                np.zeros(len(test), dtype=float),
            ),
            "recent_symbol_mean_return": (
                baseline_recent_symbol_mean(train, validation, y_col),
                baseline_recent_symbol_mean(train, test, y_col),
            ),
            "symbol_group_trend_vol_return": (
                baseline_group_trend_vol(train, validation, y_col),
                baseline_group_trend_vol(train, test, y_col),
            ),
        }
        fit_history = pd.concat([train, validation], ignore_index=True)
        for name, window, mode, shrink in SYMBOL_WINDOW_BASELINE_SPECS:
            baseline_defs[name] = (
                baseline_symbol_window_return(train, validation, y_col, window=int(window), mode=str(mode), shrink=float(shrink)),
                baseline_symbol_window_return(fit_history, test, y_col, window=int(window), mode=str(mode), shrink=float(shrink)),
            )
        for name, days, mode, shrink in MARKET_RECENT_BASELINE_SPECS:
            baseline_defs[name] = (
                baseline_market_recent_return(train, validation, y_col, days=int(days), mode=str(mode), shrink=float(shrink)),
                baseline_market_recent_return(fit_history, test, y_col, days=int(days), mode=str(mode), shrink=float(shrink)),
            )
        for name, days, mode, magnitude in MARKET_SIGN_MAGNITUDE_SPECS:
            baseline_defs[name] = (
                baseline_market_recent_sign_magnitude(train, validation, y_col, days=int(days), mode=str(mode), magnitude=float(magnitude)),
                baseline_market_recent_sign_magnitude(fit_history, test, y_col, days=int(days), mode=str(mode), magnitude=float(magnitude)),
            )
        for name, (val_pred, test_pred) in baseline_defs.items():
            lower, upper, width = conformal_interval(y_val, val_pred, test_pred)
            candidate_predictions[name] = {
                "val_pred": val_pred,
                "test_pred": test_pred,
                "lower": lower,
                "upper": upper,
                "interval_type": f"conformal_symmetric_width={width:.8f}",
                "candidate": FittedCandidate(name=name, feature_cols=[], model=None, kind="baseline"),
            }

        fitted = fit_model_candidates(train, feature_cols, y_col, random_state=17 + fold.fold_id + int(horizon))
        for candidate in fitted:
            try:
                val_pred = predict_candidate(candidate, validation)
                test_pred = predict_candidate(candidate, test)
                if candidate.kind == "quantile_pipeline":
                    q_lower_val, q_upper_val = predict_quantile_bounds(candidate, validation)
                    q_lower_test, q_upper_test = predict_quantile_bounds(candidate, test)
                    lower, upper, adjustment = quantile_conformal_interval(y_val, q_lower_val, q_upper_val, q_lower_test, q_upper_test)
                    interval_type = f"quantile_10_90_conformal_adjustment={adjustment:.8f}"
                else:
                    lower, upper, width = conformal_interval(y_val, val_pred, test_pred)
                    interval_type = f"conformal_symmetric_width={width:.8f}"
                candidate.val_mae = float(np.mean(np.abs(y_val - val_pred)))
                candidate_predictions[candidate.name] = {
                    "val_pred": val_pred,
                    "test_pred": test_pred,
                    "lower": lower,
                    "upper": upper,
                    "interval_type": interval_type,
                    "candidate": candidate,
                }
            except Exception:
                continue

        blend_pool = {
            name: values
            for name, values in candidate_predictions.items()
            if name not in {"naive_last_close"} and np.isfinite(np.mean(np.abs(y_val - values["val_pred"])))
        }
        if len(blend_pool) >= 2:
            ranked = sorted(blend_pool.items(), key=lambda item: float(np.mean(np.abs(y_val - item[1]["val_pred"]))))[:3]
            inv = np.array([1.0 / max(float(np.mean(np.abs(y_val - values["val_pred"]))), 1e-8) for _, values in ranked], dtype=float)
            weights = inv / inv.sum()
            val_pred = sum(float(weight) * values["val_pred"] for weight, (_, values) in zip(weights, ranked))
            test_pred = sum(float(weight) * values["test_pred"] for weight, (_, values) in zip(weights, ranked))
            lower, upper, width = conformal_interval(y_val, val_pred, test_pred)
            candidate_predictions["validation_weighted_blend"] = {
                "val_pred": val_pred,
                "test_pred": test_pred,
                "lower": lower,
                "upper": upper,
                "interval_type": f"validation_weighted_blend_conformal_width={width:.8f}",
                "candidate": FittedCandidate(
                    name="validation_weighted_blend",
                    feature_cols=feature_cols,
                    model=None,
                    kind="blend",
                    component_weights={name: float(weight) for weight, (name, _) in zip(weights, ranked)},
                    component_models={name: values["candidate"] for name, values in ranked if values["candidate"].kind != "baseline"},
                ),
            }

        majority_direction_accuracy = historical_majority_direction_accuracy(fit_history, test, y_col)
        for model_name, values in candidate_predictions.items():
            preds = prediction_frame(
                test,
                horizon=horizon,
                fold=fold,
                model_name=model_name,
                pred=values["test_pred"],
                lower=values["lower"],
                upper=values["upper"],
                interval_type=values["interval_type"],
            )
            prediction_parts.append(preds)
            metrics = fold_metrics_from_predictions(preds)
            if not metrics:
                continue
            metrics.update(
                {
                    "candidate_scope": NEXT_CLOSE_SCOPE,
                    "horizon_days": int(horizon),
                    "fold_id": fold.fold_id,
                    "model_name": model_name,
                    "train_start_date": fold.train_start,
                    "train_end_date": fold.train_end,
                    "calibration_start_date": fold.calibration_start,
                    "calibration_end_date": fold.calibration_end,
                    "test_start_date": fold.test_start,
                    "test_end_date": fold.test_end,
                    "validation_mae_log_return": float(np.mean(np.abs(y_val - values["val_pred"]))),
                    "historical_majority_direction_accuracy": majority_direction_accuracy,
                    "feature_count": int(len(feature_cols)),
                }
            )
            metric_rows.append(metrics)

    metrics = pd.DataFrame(metric_rows)
    oos_predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    feature_selection = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    return metrics, oos_predictions, feature_selection


def weighted_mean(frame: pd.DataFrame, column: str, weight_col: str = "oos_event_count") -> float:
    if column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame.get(weight_col, pd.Series(1.0, index=frame.index)), errors="coerce").fillna(0.0)
    mask = values.notna() & weights.gt(0)
    if not mask.any():
        return np.nan
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def bootstrap_mae_improvement_lower(oos: pd.DataFrame, horizon: int, model_name: str, *, n_boot: int = 200) -> float:
    if oos.empty:
        return np.nan
    if model_name == "naive_last_close":
        return 0.0
    subset = oos[oos["horizon_days"].eq(int(horizon)) & oos["model_name"].isin([model_name, "naive_last_close"])].copy()
    if subset.empty:
        return np.nan
    key_cols = ["symbol", "date"]
    pivot = subset.pivot_table(index=key_cols, columns="model_name", values="abs_log_error", aggfunc="mean")
    if model_name not in pivot.columns or "naive_last_close" not in pivot.columns:
        return np.nan
    paired = pivot[[model_name, "naive_last_close"]].dropna()
    if len(paired) < 10:
        return float((paired["naive_last_close"] - paired[model_name]).mean()) if not paired.empty else np.nan
    rng = np.random.default_rng(20260602 + int(horizon))
    dates = np.array(sorted(set(idx[1] for idx in paired.index)))
    improvements: list[float] = []
    for _ in range(n_boot):
        sample_dates = rng.choice(dates, size=len(dates), replace=True)
        mask = paired.index.get_level_values("date").isin(sample_dates)
        sample = paired.loc[mask]
        if not sample.empty:
            improvements.append(float(sample["naive_last_close"].mean() - sample[model_name].mean()))
    return float(np.nanquantile(improvements, 0.05)) if improvements else np.nan


def aggregate_model_comparison(
    metrics: pd.DataFrame,
    oos_predictions: pd.DataFrame,
    *,
    min_oos_events: int = MIN_OOS_EVENTS_FOR_PROMOTION,
    min_folds: int = MIN_FOLDS_FOR_PROMOTION,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (scope, horizon, model_name), group in metrics.groupby(["candidate_scope", "horizon_days", "model_name"], dropna=False):
        baseline = metrics[
            metrics["candidate_scope"].eq(scope)
            & metrics["horizon_days"].eq(horizon)
            & metrics["model_name"].eq("naive_last_close")
        ]
        mae = weighted_mean(group, "mae_log_return")
        rmse = weighted_mean(group, "rmse_log_return")
        base_mae = weighted_mean(baseline, "mae_log_return")
        base_rmse = weighted_mean(baseline, "rmse_log_return")
        direction_accuracy = weighted_mean(group, "direction_accuracy")
        historical_majority = weighted_mean(group, "historical_majority_direction_accuracy")
        interval_coverage = weighted_mean(group, "interval_coverage_80")
        within_1atr = weighted_mean(group, "within_1atr_rate")
        within_2pct = weighted_mean(group, "within_2pct_rate")
        base_within_1atr = weighted_mean(baseline, "within_1atr_rate")
        base_within_2pct = weighted_mean(baseline, "within_2pct_rate")
        mae_improvement = base_mae - mae if math.isfinite(base_mae) and math.isfinite(mae) else np.nan
        mae_improvement_pct = (mae_improvement / base_mae * 100.0) if math.isfinite(base_mae) and base_mae > 0 and math.isfinite(mae_improvement) else np.nan
        rmse_delta_vs_naive = rmse - base_rmse if math.isfinite(rmse) and math.isfinite(base_rmse) else np.nan
        bootstrap_lower = bootstrap_mae_improvement_lower(oos_predictions, int(horizon), str(model_name))
        oos_event_count = int(pd.to_numeric(group["oos_event_count"], errors="coerce").sum())
        fold_count = int(group["fold_id"].nunique())

        reasons: list[str] = []
        if oos_event_count < min_oos_events:
            reasons.append("OOS_EVENT_COUNT_LT_5000")
        if fold_count < min_folds:
            reasons.append("FOLD_COUNT_LT_4")
        if not (INTERVAL_COVERAGE_LOW <= interval_coverage <= INTERVAL_COVERAGE_HIGH):
            reasons.append("INTERVAL_COVERAGE_OUTSIDE_75_85")
        if not (math.isfinite(mae_improvement_pct) and mae_improvement_pct >= MIN_MAE_IMPROVEMENT_PCT):
            reasons.append("MAE_IMPROVEMENT_LT_1PCT")
        if math.isfinite(rmse_delta_vs_naive) and rmse_delta_vs_naive > 1e-12:
            reasons.append("RMSE_WORSE_THAN_NAIVE")
        if not (math.isfinite(bootstrap_lower) and bootstrap_lower > 0):
            reasons.append("BOOTSTRAP_MAE_IMPROVEMENT_LOWER_NOT_POSITIVE")
        direction_threshold = max(0.50, historical_majority if math.isfinite(historical_majority) else 0.50)
        if not (math.isfinite(direction_accuracy) and direction_accuracy + 1e-12 >= direction_threshold):
            reasons.append("DIRECTION_ACCURACY_NOT_ABOVE_BASELINE")
        within_1atr_improved = math.isfinite(within_1atr) and math.isfinite(base_within_1atr) and within_1atr > base_within_1atr
        within_2pct_improved = math.isfinite(within_2pct) and math.isfinite(base_within_2pct) and within_2pct > base_within_2pct
        if not (within_1atr_improved or within_2pct_improved):
            reasons.append("WITHIN_TOLERANCE_NOT_IMPROVED")
        performance_pass = not reasons
        rank_score = (
            (1000.0 if performance_pass else 0.0)
            + ((mae_improvement_pct * 10.0) if math.isfinite(mae_improvement_pct) else -999.0)
            - (abs(interval_coverage - 0.80) * 5.0 if math.isfinite(interval_coverage) else 5.0)
            - max(0.0, rmse_delta_vs_naive if math.isfinite(rmse_delta_vs_naive) else 0.0) * 100.0
        )
        rows.append(
            {
                "candidate_scope": scope,
                "horizon_days": int(horizon),
                "model_name": model_name,
                "model_role": "next_close_forecast_candidate",
                "oos_event_count": oos_event_count,
                "fold_count": fold_count,
                "mae_log_return": mae,
                "rmse_log_return": rmse,
                "naive_mae_log_return": base_mae,
                "naive_rmse_log_return": base_rmse,
                "mae_improvement": mae_improvement,
                "mae_improvement_pct": mae_improvement_pct,
                "rmse_delta_vs_naive": rmse_delta_vs_naive,
                "bootstrap_mae_improvement_lower_5pct": bootstrap_lower,
                "direction_accuracy": direction_accuracy,
                "historical_majority_direction_accuracy": historical_majority,
                "direction_threshold": direction_threshold,
                "interval_coverage_80": interval_coverage,
                "within_1atr_rate": within_1atr,
                "within_2pct_rate": within_2pct,
                "naive_within_1atr_rate": base_within_1atr,
                "naive_within_2pct_rate": base_within_2pct,
                "prediction_quality_pass": True,
                "performance_quality_pass": performance_pass,
                "performance_quality_block_reasons": "PASS" if performance_pass else "|".join(reasons),
                "quality_block_reasons": "PASS" if performance_pass else "|".join(reasons),
                "decision_scope_eligible": False,
                "rank_score": rank_score,
            }
        )
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    return comparison.sort_values(["horizon_days", "rank_score", "mae_log_return"], ascending=[True, False, True]).reset_index(drop=True)


def champion_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    rows = []
    for horizon, group in comparison.groupby("horizon_days", dropna=False):
        ranked = group.sort_values(["rank_score", "mae_log_return"], ascending=[False, True])
        rows.append(ranked.iloc[0])
    return pd.DataFrame(rows)


def _market_wide_latest_model_names() -> set[str]:
    return {
        "naive_last_close",
        "validation_weighted_blend",
        *(name for name, *_ in MARKET_RECENT_BASELINE_SPECS),
        *(name for name, *_ in MARKET_SIGN_MAGNITUDE_SPECS),
    }


def latest_champion_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    market_wide_names = _market_wide_latest_model_names()
    rows = []
    for horizon, group in comparison.groupby("horizon_days", dropna=False):
        eligible = group[~group["model_name"].astype(str).isin(market_wide_names)].copy()
        ranked_source = eligible if not eligible.empty else group
        ranked = ranked_source.sort_values(["rank_score", "mae_log_return"], ascending=[False, True])
        rows.append(ranked.iloc[0])
    return pd.DataFrame(rows)


def fit_latest_model(
    train: pd.DataFrame,
    latest: pd.DataFrame,
    horizon: int,
    champion_name: str,
    comparison: pd.DataFrame,
    *,
    max_numeric_features: int,
    max_categorical_features: int,
) -> tuple[np.ndarray, str]:
    y_col = target_col(horizon)
    if champion_name == "naive_last_close":
        return np.zeros(len(latest), dtype=float), "latest_naive_last_close"
    if champion_name == "recent_symbol_mean_return":
        return baseline_recent_symbol_mean(train, latest, y_col), "latest_recent_symbol_mean_return"
    if champion_name == "symbol_group_trend_vol_return":
        return baseline_group_trend_vol(train, latest, y_col), "latest_symbol_group_trend_vol_return"
    symbol_window_spec = symbol_window_spec_by_name(champion_name)
    if symbol_window_spec is not None:
        window, mode, shrink = symbol_window_spec
        return (
            baseline_symbol_window_return(train, latest, y_col, window=window, mode=mode, shrink=shrink),
            f"latest_{champion_name}",
        )
    market_recent_spec = market_recent_spec_by_name(champion_name)
    if market_recent_spec is not None:
        days, mode, shrink = market_recent_spec
        return (
            baseline_market_recent_return(train, latest, y_col, days=days, mode=mode, shrink=shrink),
            f"latest_{champion_name}",
        )
    market_sign_spec = market_sign_magnitude_spec_by_name(champion_name)
    if market_sign_spec is not None:
        days, mode, magnitude = market_sign_spec
        return (
            baseline_market_recent_sign_magnitude(train, latest, y_col, days=days, mode=mode, magnitude=magnitude),
            f"latest_{champion_name}",
        )

    feature_cols, _ = select_fold_features(
        train,
        y_col,
        max_numeric_features=max_numeric_features,
        max_categorical_features=max_categorical_features,
    )
    if not feature_cols or SKLEARN_IMPORT_ERROR is not None:
        return baseline_recent_symbol_mean(train, latest, y_col), "latest_fallback_recent_symbol_mean_return"

    if champion_name == "validation_weighted_blend":
        candidate_names = (
            comparison[
                comparison["horizon_days"].eq(int(horizon))
                & ~comparison["model_name"].isin(["validation_weighted_blend", "naive_last_close"])
            ]
            .sort_values(["mae_log_return", "rank_score"], ascending=[True, False])
            .head(3)["model_name"]
            .astype(str)
            .tolist()
        )
    else:
        candidate_names = [champion_name]

    fitted = {candidate.name: candidate for candidate in fit_model_candidates(train, feature_cols, y_col, random_state=731 + int(horizon))}
    usable = [name for name in candidate_names if name in fitted]
    if not usable:
        return baseline_recent_symbol_mean(train, latest, y_col), "latest_fallback_recent_symbol_mean_return"
    if len(usable) == 1:
        return predict_candidate(fitted[usable[0]], latest), f"latest_refit_{usable[0]}"

    mae_map = {
        str(row["model_name"]): as_float(row["mae_log_return"], np.nan)
        for _, row in comparison[comparison["horizon_days"].eq(int(horizon))].iterrows()
    }
    inv = np.array([1.0 / max(mae_map.get(name, np.nan), 1e-8) for name in usable], dtype=float)
    if not np.isfinite(inv).all() or inv.sum() <= 0:
        inv = np.ones(len(usable), dtype=float)
    weights = inv / inv.sum()
    pred = np.zeros(len(latest), dtype=float)
    for name, weight in zip(usable, weights):
        pred += float(weight) * predict_candidate(fitted[name], latest)
    return pred, "latest_refit_validation_weighted_blend"


def latest_interval_width(oos_predictions: pd.DataFrame, horizon: int, champion_name: str) -> float:
    rows = oos_predictions[oos_predictions["horizon_days"].eq(int(horizon)) & oos_predictions["model_name"].eq(champion_name)]
    residual = pd.to_numeric(rows.get("abs_log_error", pd.Series(dtype=float)), errors="coerce").dropna()
    if residual.empty:
        return 0.0
    return finite_sample_quantile(residual.to_numpy(dtype=float), CONFORMAL_RESIDUAL_QUANTILE)


def build_latest_predictions(
    feature_matrix: pd.DataFrame,
    comparison: pd.DataFrame,
    oos_predictions: pd.DataFrame,
    decision_symbols: list[str],
    horizons: Iterable[int],
    *,
    max_numeric_features: int,
    max_categorical_features: int,
) -> pd.DataFrame:
    if feature_matrix.empty:
        return pd.DataFrame()
    work = feature_matrix.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    latest = work.sort_values(["symbol", "date"]).groupby("symbol", sort=False).tail(1).copy()
    latest = latest[latest["symbol"].astype(str).str.upper().isin(decision_symbols)].copy()
    latest["symbol"] = latest["symbol"].astype(str).str.upper()
    champions = latest_champion_rows(comparison)
    champion_by_horizon = {int(row["horizon_days"]): row for _, row in champions.iterrows()} if not champions.empty else {}
    for horizon in horizons:
        h = int(horizon)
        y_col = target_col(h)
        label_status_col = f"label_status_{h}d"
        train = work[work[label_status_col].eq("LABELED") & pd.to_numeric(work[y_col], errors="coerce").notna()].copy()
        champion = champion_by_horizon.get(h)
        champion_name = str(champion.get("model_name", "recent_symbol_mean_return")) if champion is not None else "recent_symbol_mean_return"
        pred, latest_source = fit_latest_model(
            train,
            latest,
            h,
            champion_name,
            comparison,
            max_numeric_features=max_numeric_features,
            max_categorical_features=max_categorical_features,
        )
        width = latest_interval_width(oos_predictions, h, champion_name)
        lower = pred - width
        upper = pred + width
        close_engine = pd.to_numeric(latest["close_engine"], errors="coerce").to_numpy(dtype=float)
        close_usd = pd.to_numeric(latest["close_usd"], errors="coerce").to_numpy(dtype=float)
        close_native = pd.to_numeric(latest["close_native"], errors="coerce").to_numpy(dtype=float)
        fx = pd.to_numeric(latest.get("fx_rate_to_usd", pd.Series(np.nan, index=latest.index)), errors="coerce").to_numpy(dtype=float)
        pred_engine = close_engine * np.exp(pred)
        pred_usd = np.where(np.isfinite(close_usd) & (close_usd > 0), close_usd * np.exp(pred), pred_engine)
        pred_native = np.where(np.isfinite(close_native) & (close_native > 0), close_native * np.exp(pred), np.where(np.isfinite(fx) & (fx > 0), pred_usd / fx, pred_engine))
        lower_engine = close_engine * np.exp(lower)
        upper_engine = close_engine * np.exp(upper)
        display_currency = latest.get("display_currency", pd.Series("USD", index=latest.index)).astype(str).str.upper()
        display_pred = np.where(display_currency.eq("KRW").to_numpy(), pred_native, pred_usd)
        display_lower = np.where(display_currency.eq("KRW").to_numpy(), close_native * np.exp(lower), np.where(np.isfinite(close_usd), close_usd * np.exp(lower), lower_engine))
        display_upper = np.where(display_currency.eq("KRW").to_numpy(), close_native * np.exp(upper), np.where(np.isfinite(close_usd), close_usd * np.exp(upper), upper_engine))
        suffix = f"{h}d"
        latest[f"next_close_best_model_{suffix}"] = champion_name
        latest[f"next_close_latest_model_source_{suffix}"] = latest_source
        latest[f"next_close_predicted_return_log_{suffix}"] = pred
        latest[f"next_close_predicted_return_pct_{suffix}"] = (np.exp(pred) - 1.0) * 100.0
        latest[f"next_close_predicted_close_engine_{suffix}"] = pred_engine
        latest[f"next_close_predicted_close_usd_{suffix}"] = pred_usd
        latest[f"next_close_predicted_close_native_{suffix}"] = pred_native
        latest[f"next_close_predicted_close_{suffix}"] = display_pred
        latest[f"next_close_lower_80_engine_{suffix}"] = lower_engine
        latest[f"next_close_upper_80_engine_{suffix}"] = upper_engine
        latest[f"next_close_lower_80_{suffix}"] = display_lower
        latest[f"next_close_upper_80_{suffix}"] = display_upper
        latest[f"next_close_interval_width_log_{suffix}"] = width
        if champion is not None:
            latest[f"next_close_performance_quality_pass_{suffix}"] = bool(champion.get("performance_quality_pass", False))
            latest[f"next_close_quality_block_reasons_{suffix}"] = champion.get("quality_block_reasons", "NO_MODEL")
            latest[f"next_close_oos_event_count_{suffix}"] = champion.get("oos_event_count", 0)
            latest[f"next_close_mae_log_return_{suffix}"] = champion.get("mae_log_return", np.nan)
            latest[f"next_close_interval_coverage_80_{suffix}"] = champion.get("interval_coverage_80", np.nan)
        else:
            latest[f"next_close_performance_quality_pass_{suffix}"] = False
            latest[f"next_close_quality_block_reasons_{suffix}"] = "NO_MODEL"
            latest[f"next_close_oos_event_count_{suffix}"] = 0
            latest[f"next_close_mae_log_return_{suffix}"] = np.nan
            latest[f"next_close_interval_coverage_80_{suffix}"] = np.nan

    horizon_pass = [latest.get(f"next_close_performance_quality_pass_{int(h)}d", pd.Series(False, index=latest.index)).map(to_bool).all() for h in horizons]
    status = LATEST_PROMOTED_STATUS if horizon_pass and all(horizon_pass) else LATEST_DISPLAY_ONLY_STATUS
    latest["next_close_prediction_asof_date"] = latest["date"].dt.strftime("%Y-%m-%d")
    latest["date"] = latest["date"].dt.strftime("%Y-%m-%d")
    latest["next_close_prediction_scope"] = NEXT_CLOSE_SCOPE
    latest["next_close_model_quality_status"] = status
    latest["next_close_prediction_signal_status"] = status
    latest["next_close_all_horizons_quality_pass"] = bool(status == LATEST_PROMOTED_STATUS)
    ordered_symbols = {symbol: idx for idx, symbol in enumerate(decision_symbols)}
    latest["_symbol_order"] = latest["symbol"].map(ordered_symbols).fillna(999).astype(int)
    return latest.sort_values("_symbol_order").drop(columns=["_symbol_order"]).reset_index(drop=True)


def build_latest_snapshot(universe_latest: pd.DataFrame, symbol: str = "TSM") -> pd.DataFrame:
    if universe_latest.empty:
        values = {
            "next_close_prediction_asof_date": "",
            "next_close_prediction_scope": NEXT_CLOSE_SCOPE,
            "next_close_model_quality_status": "NO_MODEL",
            "next_close_prediction_signal_status": "INSUFFICIENT_DATA",
        }
        return pd.DataFrame([{"field": key, "value": value} for key, value in values.items()])
    symbol_key = str(symbol).upper()
    rows = universe_latest[universe_latest["symbol"].astype(str).str.upper().eq(symbol_key)]
    if rows.empty:
        rows = universe_latest.head(1)
    row = rows.iloc[-1].to_dict()
    fields = {
        key: value
        for key, value in row.items()
        if str(key).startswith("next_close_") or key in {"symbol", "symbol_group", "date", "display_currency", "close_engine", "close_usd", "close_native"}
    }
    return pd.DataFrame([{"field": key, "value": value} for key, value in fields.items()])


def build_interval_calibration(oos_predictions: pd.DataFrame) -> pd.DataFrame:
    if oos_predictions.empty:
        return pd.DataFrame()
    rows = []
    for (horizon, model), group in oos_predictions.groupby(["horizon_days", "model_name"], dropna=False):
        rows.append(
            {
                "candidate_scope": NEXT_CLOSE_SCOPE,
                "horizon_days": int(horizon),
                "model_name": model,
                "interval_target_coverage": 0.80,
                "interval_coverage_80": float(pd.to_numeric(group["interval_hit_80"], errors="coerce").mean()),
                "event_count": int(len(group)),
                "mean_interval_width_log": float((pd.to_numeric(group["upper_80_return_log"], errors="coerce") - pd.to_numeric(group["lower_80_return_log"], errors="coerce")).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_quality_checks(
    labels: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    feature_selection: pd.DataFrame,
    metrics: pd.DataFrame,
    oos_predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    latest: pd.DataFrame,
    latest_snapshot_updated: bool,
    decision_symbols: list[str],
    horizons: Iterable[int],
    *,
    min_oos_events: int = MIN_OOS_EVENTS_FOR_PROMOTION,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append(check_row("next_close_sklearn_available", SKLEARN_IMPORT_ERROR is None, "CRITICAL", "ok" if SKLEARN_IMPORT_ERROR is None else str(SKLEARN_IMPORT_ERROR)))
    rows.append(check_row("next_close_lightgbm_available", LIGHTGBM_IMPORT_ERROR is None, "WARN", "ok" if LIGHTGBM_IMPORT_ERROR is None else str(LIGHTGBM_IMPORT_ERROR)))
    rows.append(check_row("next_close_xgboost_available", XGBOOST_IMPORT_ERROR is None, "WARN", "ok" if XGBOOST_IMPORT_ERROR is None else str(XGBOOST_IMPORT_ERROR)))
    rows.append(check_row("next_close_labels_non_empty", not labels.empty, "CRITICAL", len(labels)))
    rows.append(check_row("next_close_feature_matrix_non_empty", not feature_matrix.empty, "CRITICAL", len(feature_matrix)))
    rows.append(check_row("next_close_walk_forward_metrics_non_empty", not metrics.empty, "CRITICAL", len(metrics)))
    rows.append(check_row("next_close_oos_predictions_non_empty", not oos_predictions.empty, "CRITICAL", len(oos_predictions)))
    rows.append(check_row("next_close_model_comparison_non_empty", not comparison.empty, "CRITICAL", len(comparison)))

    if not feature_selection.empty and {"feature", "selected"}.issubset(feature_selection.columns):
        selected_features = feature_selection[feature_selection["selected"].map(to_bool)]["feature"].astype(str).tolist()
    else:
        selected_features = []
    forbidden = [col for col in selected_features if is_forbidden_feature(col)]
    rows.append(check_row("next_close_selected_features_have_no_forbidden_columns", not forbidden, "CRITICAL", ",".join(forbidden) if forbidden else "ok"))

    if not oos_predictions.empty and {"date", "calibration_end_date"}.issubset(oos_predictions.columns):
        oos_dates = pd.to_datetime(oos_predictions["date"], errors="coerce")
        calibration_end = pd.to_datetime(oos_predictions["calibration_end_date"], errors="coerce")
        violations = int((oos_dates <= calibration_end).sum())
        rows.append(check_row("next_close_oos_predictions_after_calibration_end", violations == 0, "CRITICAL", f"violations={violations}"))
    else:
        rows.append(check_row("next_close_oos_predictions_after_calibration_end", False, "CRITICAL", "missing_columns"))

    latest_symbols = set(latest.get("symbol", pd.Series(dtype=object)).astype(str).str.upper()) if not latest.empty else set()
    missing_latest = sorted(set(decision_symbols) - latest_symbols)
    rows.append(check_row("next_close_latest_rows_cover_top12", not missing_latest, "CRITICAL", len(latest_symbols & set(decision_symbols)), f"expected={len(decision_symbols)}", ",".join(missing_latest)))
    rows.append(check_row("next_close_latest_snapshot_merged_into_prediction_snapshot", latest_snapshot_updated, "WARN", latest_snapshot_updated, details="False is ok in isolated test outdirs without a baseline latest snapshot."))

    for horizon in horizons:
        h = int(horizon)
        status_col = f"label_status_{h}d"
        labeled_count = int(labels[status_col].eq("LABELED").sum()) if status_col in labels.columns else 0
        rows.append(check_row(f"next_close_labeled_rows_{h}d_non_empty", labeled_count > 0, "CRITICAL", labeled_count))
        horizon_cmp = comparison[comparison["horizon_days"].eq(h)] if not comparison.empty and "horizon_days" in comparison.columns else pd.DataFrame()
        champion = horizon_cmp.sort_values(["rank_score", "mae_log_return"], ascending=[False, True]).head(1)
        if champion.empty:
            rows.append(check_row(f"next_close_horizon_{h}d_champion_available", False, "CRITICAL", "missing"))
            continue
        row = champion.iloc[0]
        rows.append(check_row(f"next_close_oos_event_count_{h}d_at_least_5000", as_float(row.get("oos_event_count"), 0) >= min_oos_events, "CRITICAL", row.get("oos_event_count"), f">={min_oos_events}"))
        rows.append(check_row(f"next_close_fold_count_{h}d_at_least_4", as_float(row.get("fold_count"), 0) >= MIN_FOLDS_FOR_PROMOTION, "CRITICAL", row.get("fold_count"), f">={MIN_FOLDS_FOR_PROMOTION}"))
        coverage = as_float(row.get("interval_coverage_80"))
        rows.append(check_row(f"next_close_interval_coverage_{h}d_75_85", INTERVAL_COVERAGE_LOW <= coverage <= INTERVAL_COVERAGE_HIGH, "CRITICAL", coverage, "0.75..0.85"))
        rows.append(check_row(f"next_close_mae_improvement_{h}d_at_least_1pct", as_float(row.get("mae_improvement_pct")) >= MIN_MAE_IMPROVEMENT_PCT, "CRITICAL", row.get("mae_improvement_pct"), ">=1%"))
        rows.append(check_row(f"next_close_rmse_{h}d_not_worse_than_naive", as_float(row.get("rmse_delta_vs_naive"), np.inf) <= 0.0, "CRITICAL", row.get("rmse_delta_vs_naive"), "<=0"))
        rows.append(check_row(f"next_close_bootstrap_mae_improvement_{h}d_positive", as_float(row.get("bootstrap_mae_improvement_lower_5pct")) > 0.0, "CRITICAL", row.get("bootstrap_mae_improvement_lower_5pct"), ">0"))
        rows.append(
            check_row(
                f"next_close_direction_accuracy_{h}d_at_least_baseline",
                as_float(row.get("direction_accuracy")) + 1e-12 >= as_float(row.get("direction_threshold"), 0.5),
                "CRITICAL",
                row.get("direction_accuracy"),
                f">={row.get('direction_threshold')}",
            )
        )
        tolerance_pass = (
            as_float(row.get("within_1atr_rate")) > as_float(row.get("naive_within_1atr_rate"), -np.inf)
            or as_float(row.get("within_2pct_rate")) > as_float(row.get("naive_within_2pct_rate"), -np.inf)
        )
        rows.append(check_row(f"next_close_tolerance_rate_{h}d_improved", tolerance_pass, "CRITICAL", f"within_1atr={row.get('within_1atr_rate')}; within_2pct={row.get('within_2pct_rate')}"))
    return pd.DataFrame(rows)


def write_report(outdir: Path, comparison: pd.DataFrame, latest_snapshot: pd.DataFrame, quality: pd.DataFrame) -> None:
    latest = key_value_frame_to_dict(latest_snapshot)
    critical = quality[quality["severity"].eq("CRITICAL")] if not quality.empty and "severity" in quality.columns else quality
    critical_passed = bool(critical["passed"].all()) if not critical.empty else False
    lines = [
        "# Next-Close Forecast Report",
        "",
        "- Target: direct log close return for 1D/5D/20D horizons.",
        "- Scope: enabled Top12 decision symbols from the decision universe config.",
        "- Inputs: daily-aligned pooled features, including hourly/minute coverage and freshness features.",
        "- Use: diagnostic display only unless all horizon success gates pass.",
        f"- Latest status: {latest.get('next_close_model_quality_status', 'NA')}",
        f"- Critical quality checks passed: {critical_passed}",
        "",
        "## Latest TSM Forecast",
    ]
    for horizon in DEFAULT_HORIZONS:
        suffix = f"{horizon}d"
        lines.append(
            f"- {suffix}: close={latest.get(f'next_close_predicted_close_{suffix}', 'NA')}, "
            f"return_pct={latest.get(f'next_close_predicted_return_pct_{suffix}', 'NA')}, "
            f"interval=[{latest.get(f'next_close_lower_80_{suffix}', 'NA')}, {latest.get(f'next_close_upper_80_{suffix}', 'NA')}]"
        )
    lines.extend(
        [
            "",
            "## Top Models",
            "",
            "| Horizon | Model | OOS | Folds | MAE | MAE Improvement | RMSE Delta | Direction | Coverage | Pass | Block Reasons |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    if comparison.empty:
        lines.append("| NA | NA | 0 | 0 | NA | NA | NA | NA | NA | False | NO_MODEL |")
    else:
        display_rows = (
            comparison.sort_values(["horizon_days", "rank_score", "mae_log_return"], ascending=[True, False, True])
            .groupby("horizon_days", group_keys=False)
            .head(6)
        )
        for _, row in display_rows.iterrows():
            lines.append(
                f"| {row.get('horizon_days', '')} | {row.get('model_name', '')} | {int(as_float(row.get('oos_event_count'), 0))} | "
                f"{int(as_float(row.get('fold_count'), 0))} | {as_float(row.get('mae_log_return')):.6f} | "
                f"{as_float(row.get('mae_improvement_pct')):.2f}% | {as_float(row.get('rmse_delta_vs_naive')):.6f} | "
                f"{as_float(row.get('direction_accuracy')):.4f} | {as_float(row.get('interval_coverage_80')):.4f} | "
                f"{to_bool(row.get('performance_quality_pass', False))} | {row.get('quality_block_reasons', '')} |"
            )
    lines.extend(["", "This report is research tooling, not investment advice."])
    (outdir / f"{OUTPUT_PREFIX}_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_next_close_artifacts(
    *,
    pooled_feature_matrix_path: Path,
    decision_universe_config_path: Path,
    latest_prediction_path: Path,
    outdir: Path,
    horizons: list[int],
    train_days: int,
    validation_days: int,
    test_days: int,
    step_days: int,
    max_numeric_features: int,
    max_categorical_features: int,
    min_oos_events: int,
    update_latest: bool,
    symbol: str,
) -> dict[str, pd.DataFrame | bool]:
    decision_symbols = load_decision_symbols(decision_universe_config_path)
    pooled = read_csv(pooled_feature_matrix_path)
    if pooled.empty:
        raise ValueError(f"pooled feature matrix is empty or missing: {pooled_feature_matrix_path}")
    pooled["symbol"] = pooled["symbol"].astype(str).str.upper()
    pooled = pooled[pooled["symbol"].isin(decision_symbols)].copy()
    pooled = merge_decision_metadata(pooled, decision_universe_config_path)
    labels = build_next_close_label_dataset(pooled, horizons)
    feature_matrix = build_next_close_feature_matrix(pooled, labels)

    metric_parts: list[pd.DataFrame] = []
    oos_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    for horizon in horizons:
        metrics, oos, selection = evaluate_horizon(
            feature_matrix,
            int(horizon),
            train_days=train_days,
            validation_days=validation_days,
            test_days=test_days,
            step_days=step_days,
            max_numeric_features=max_numeric_features,
            max_categorical_features=max_categorical_features,
        )
        metric_parts.append(metrics)
        oos_parts.append(oos)
        selection_parts.append(selection)
    metrics = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    oos_predictions = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame()
    feature_selection = pd.concat(selection_parts, ignore_index=True) if selection_parts else pd.DataFrame()
    comparison = aggregate_model_comparison(metrics, oos_predictions, min_oos_events=min_oos_events)
    interval_calibration = build_interval_calibration(oos_predictions)
    universe_latest = build_latest_predictions(
        feature_matrix,
        comparison,
        oos_predictions,
        decision_symbols,
        horizons,
        max_numeric_features=max_numeric_features,
        max_categorical_features=max_categorical_features,
    )
    latest_snapshot = build_latest_snapshot(universe_latest, symbol=symbol)
    latest_snapshot_updated = merge_latest_prediction_snapshot(latest_prediction_path, latest_snapshot) if update_latest else False
    quality = build_quality_checks(
        labels,
        feature_matrix,
        feature_selection,
        metrics,
        oos_predictions,
        comparison,
        universe_latest,
        latest_snapshot_updated,
        decision_symbols,
        horizons,
        min_oos_events=min_oos_events,
    )
    return {
        "labels": labels,
        "feature_matrix": feature_matrix,
        "feature_selection": feature_selection,
        "metrics": metrics,
        "oos_predictions": oos_predictions,
        "comparison": comparison,
        "interval_calibration": interval_calibration,
        "universe_latest": universe_latest,
        "latest_snapshot": latest_snapshot,
        "quality": quality,
        "latest_snapshot_updated": latest_snapshot_updated,
    }


def write_next_close_artifacts(outdir: Path, artifacts: dict[str, pd.DataFrame | bool]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    artifacts["labels"].to_csv(outdir / f"{OUTPUT_PREFIX}_label_dataset.csv", index=False)
    artifacts["feature_matrix"].to_csv(outdir / f"{OUTPUT_PREFIX}_feature_matrix.csv", index=False)
    artifacts["feature_selection"].to_csv(outdir / f"{OUTPUT_PREFIX}_feature_selection_report.csv", index=False)
    artifacts["metrics"].to_csv(outdir / f"{OUTPUT_PREFIX}_walk_forward_metrics.csv", index=False)
    artifacts["oos_predictions"].to_csv(outdir / f"{OUTPUT_PREFIX}_oos_predictions.csv", index=False)
    artifacts["comparison"].to_csv(outdir / f"{OUTPUT_PREFIX}_model_comparison.csv", index=False)
    artifacts["interval_calibration"].to_csv(outdir / f"{OUTPUT_PREFIX}_interval_calibration.csv", index=False)
    artifacts["universe_latest"].to_csv(outdir / f"{OUTPUT_PREFIX}_universe_latest_predictions.csv", index=False)
    artifacts["latest_snapshot"].to_csv(outdir / f"{OUTPUT_PREFIX}_latest_snapshot.csv", index=False)
    artifacts["quality"].to_csv(outdir / f"{OUTPUT_PREFIX}_quality_checks.csv", index=False)
    latest_snapshot = artifacts["latest_snapshot"]
    comparison = artifacts["comparison"]
    quality = artifacts["quality"]
    assert isinstance(latest_snapshot, pd.DataFrame)
    assert isinstance(comparison, pd.DataFrame)
    assert isinstance(quality, pd.DataFrame)
    write_report(outdir, comparison, latest_snapshot, quality)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Top12 multi-horizon next-close forecast outputs.")
    parser.add_argument("--pooled-feature-matrix", default="tsm_price_rule_output/tsm_prediction_pooled_feature_matrix.csv")
    parser.add_argument("--decision-universe-config", default="config/semiconductor_universe_top10.csv")
    parser.add_argument("--latest-prediction", default="tsm_price_rule_output/tsm_latest_prediction_snapshot.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--symbol", default="TSM")
    parser.add_argument("--horizons", default="1,5,20")
    parser.add_argument("--train-days", type=int, default=756)
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--test-days", type=int, default=252)
    parser.add_argument("--step-days", type=int, default=252)
    parser.add_argument("--max-numeric-features", type=int, default=110)
    parser.add_argument("--max-categorical-features", type=int, default=30)
    parser.add_argument("--min-oos-events", type=int, default=MIN_OOS_EVENTS_FOR_PROMOTION)
    parser.add_argument("--no-update-latest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    artifacts = build_next_close_artifacts(
        pooled_feature_matrix_path=Path(args.pooled_feature_matrix),
        decision_universe_config_path=Path(args.decision_universe_config),
        latest_prediction_path=Path(args.latest_prediction),
        outdir=outdir,
        horizons=parse_horizons(args.horizons),
        train_days=args.train_days,
        validation_days=args.validation_days,
        test_days=args.test_days,
        step_days=args.step_days,
        max_numeric_features=args.max_numeric_features,
        max_categorical_features=args.max_categorical_features,
        min_oos_events=args.min_oos_events,
        update_latest=not args.no_update_latest,
        symbol=str(args.symbol).upper(),
    )
    write_next_close_artifacts(outdir, artifacts)
    print("completed: next-close forecast outputs =", outdir.resolve())
    latest = artifacts["latest_snapshot"]
    assert isinstance(latest, pd.DataFrame)
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()

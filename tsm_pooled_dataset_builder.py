#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a pooled cross-sectional prediction dataset.

The default config builds a one-symbol TSM pool from the current outputs. To add
more symbols, pass a CSV with columns:
symbol,signals,risk_policy,trade_log,enriched

Outputs:
- tsm_prediction_pooled_label_dataset.csv
- tsm_prediction_pooled_feature_matrix.csv
- tsm_prediction_pooled_scope_stats.csv
- tsm_prediction_pooled_schema.csv
- tsm_prediction_pooled_quality_checks.csv
- tsm_prediction_pooled_dataset_report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from tsm_prediction_engine import (
    BOOL_FEATURES,
    CATEGORICAL_FEATURES,
    HORIZONS,
    NUMERIC_FEATURES,
    build_candidate_scope_stats,
    build_feature_matrix,
    build_label_dataset,
    load_inputs,
)


SEMICONDUCTOR_UNIVERSE = [
    ("TSM", "foundry"),
    ("NVDA", "ai_accelerator"),
    ("AMD", "ai_accelerator"),
    ("AVGO", "ai_accelerator"),
    ("ASML", "semicap"),
    ("AMAT", "semicap"),
    ("LRCX", "semicap"),
    ("KLAC", "semicap"),
    ("MU", "memory"),
    ("QCOM", "semiconductor"),
    ("SMH", "semiconductor_etf"),
    ("SOXX", "semiconductor_etf"),
]


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def default_config() -> pd.DataFrame:
    rows: List[Dict] = []
    for symbol, group in SEMICONDUCTOR_UNIVERSE:
        if symbol == "TSM":
            rows.append(
                {
                    "symbol": symbol,
                    "symbol_group": group,
                    "signals": "tsm_price_rule_output/tsm_daily_algorithmic_signals.csv",
                    "risk_policy": "tsm_price_rule_output/tsm_risk_policy_daily.csv",
                    "trade_log": "tsm_price_rule_output/tsm_backtest_trade_log.csv",
                    "enriched": "output/tsm_daily_10y_enriched.csv",
                }
            )
        else:
            rows.append(
                {
                    "symbol": symbol,
                    "symbol_group": group,
                    "signals": f"tsm_price_rule_output/universe/{symbol}/tsm_daily_algorithmic_signals.csv",
                    "risk_policy": f"tsm_price_rule_output/universe/{symbol}/tsm_risk_policy_daily.csv",
                    "trade_log": f"tsm_price_rule_output/universe/{symbol}/tsm_backtest_trade_log.csv",
                    "enriched": f"output/universe/{symbol}/tsm_daily_10y_enriched.csv",
                }
            )
    return pd.DataFrame(rows)


def read_config(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return default_config()
    config = strip_bom_columns(pd.read_csv(path))
    required = ["symbol", "signals", "risk_policy", "trade_log", "enriched"]
    missing = [c for c in required if c not in config.columns]
    if missing:
        raise ValueError(f"pooled config missing columns: {missing}")
    if "symbol_group" not in config.columns:
        config["symbol_group"] = "semiconductor"
    optional = [c for c in ["strict_eligible", "eligibility_status"] if c in config.columns]
    return config[["symbol", "symbol_group", *required[1:], *optional]].copy()


def build_symbol_dataset(
    row: pd.Series,
    commission_bps: float,
    slippage_bps: float,
    stop_multiple: float,
    external_features_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbol = str(row["symbol"]).upper()
    symbol_group = str(row.get("symbol_group", "semiconductor"))
    for col in ["signals", "risk_policy", "trade_log", "enriched"]:
        if not Path(row[col]).exists():
            raise FileNotFoundError(f"{symbol} missing {col}: {row[col]}")
    signals, _ = load_inputs(
        Path(row["signals"]),
        Path(row["risk_policy"]),
        Path(row["trade_log"]),
        Path(row["enriched"]),
        external_features_path,
        symbol=symbol,
    )
    labels = build_label_dataset(signals, commission_bps, slippage_bps, stop_multiple)
    features = build_feature_matrix(signals, labels)
    scope_stats = build_candidate_scope_stats(labels)
    for frame in [labels, features, scope_stats]:
        frame.insert(0, "symbol", symbol)
        frame.insert(1, "symbol_group", symbol_group)
    return labels, features, scope_stats


def add_cross_sectional_features(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty or not {"symbol", "date"}.issubset(features.columns):
        return features.copy()
    out = features.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    rank_cols = [
        "return_20d",
        "return_60d",
        "return_126d",
        "rsi_14",
        "atr_percentile_252d",
        "drawdown_from_ath",
        "volume_ratio_20",
        "score_price_algo_total",
    ]
    by_date = out.groupby("date", dropna=False)
    for col in rank_cols:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        out[f"cs_rank_{col}"] = values.groupby(out["date"], dropna=False).rank(pct=True, method="average")
        mean = by_date[col].transform(lambda s: pd.to_numeric(s, errors="coerce").mean())
        std = by_date[col].transform(lambda s: pd.to_numeric(s, errors="coerce").std()).replace(0, np.nan)
        out[f"cs_z_{col}"] = (values - mean) / std

    if "return_20d" in out.columns:
        ret20 = pd.to_numeric(out["return_20d"], errors="coerce")
        universe_median = ret20.groupby(out["date"], dropna=False).transform("median")
        out["relative_return_vs_universe_median_20d"] = ret20 - universe_median
        for benchmark in ["SMH", "SOXX"]:
            bench_col = f"{benchmark.lower()}_return_20d"
            if bench_col not in out.columns:
                bench = (
                    out[out["symbol"].astype(str).str.upper().eq(benchmark)]
                    .groupby("date", as_index=False)["return_20d"]
                    .last()
                    .rename(columns={"return_20d": bench_col})
                )
                out = out.merge(bench, on="date", how="left")
            out[f"relative_return_vs_{benchmark.lower()}_20d"] = ret20.to_numpy() - pd.to_numeric(out[bench_col], errors="coerce")
    if {"close", "sma_50"}.issubset(out.columns):
        above_50 = pd.to_numeric(out["close"], errors="coerce") >= pd.to_numeric(out["sma_50"], errors="coerce")
        out["universe_above_sma50_ratio"] = above_50.astype(float).groupby(out["date"], dropna=False).transform("mean")
    if {"close", "sma_200"}.issubset(out.columns):
        above_200 = pd.to_numeric(out["close"], errors="coerce") >= pd.to_numeric(out["sma_200"], errors="coerce")
        out["universe_above_sma200_ratio"] = above_200.astype(float).groupby(out["date"], dropna=False).transform("mean")
    if "is_model_training_candidate" in out.columns:
        model_training = out["is_model_training_candidate"].astype(str).str.lower().isin(["true", "1", "yes"])
        out["universe_model_training_candidate_ratio"] = model_training.astype(float).groupby(out["date"], dropna=False).transform("mean")
    if "is_decision_entry_candidate" in out.columns:
        decision_entry = out["is_decision_entry_candidate"].astype(str).str.lower().isin(["true", "1", "yes"])
        out["universe_decision_entry_candidate_ratio"] = decision_entry.astype(float).groupby(out["date"], dropna=False).transform("mean")
    return out


def add_signal_cluster_features(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty or not {"symbol", "date"}.issubset(features.columns):
        return features.copy()
    out = features.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["symbol", "date", "signal_idx" if "signal_idx" in out.columns else "date"]).reset_index(drop=True)
    if "is_event_candidate" in out.columns:
        event_signal = out["is_event_candidate"].astype(str).str.lower().isin(["true", "1", "yes"])
    elif "entry_trigger" in out.columns:
        event_signal = out["entry_trigger"].astype(str).ne("NONE")
    else:
        event_signal = pd.Series(False, index=out.index)
    if "is_decision_entry_candidate" in out.columns:
        strict_signal = out["is_decision_entry_candidate"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        strict_signal = event_signal.copy()
    out["_pooled_event_signal"] = event_signal.astype(float)
    out["_pooled_strict_signal"] = strict_signal.astype(float)
    out["days_since_prev_signal"] = np.nan
    for _, group in out.groupby("symbol", sort=False):
        last_event_date = None
        for idx, row in group.iterrows():
            date = row["date"]
            if last_event_date is not None and pd.notna(date):
                out.at[idx, "days_since_prev_signal"] = int((date - last_event_date).days)
            if bool(out.at[idx, "_pooled_event_signal"]) and pd.notna(date):
                last_event_date = date
        dated = group.dropna(subset=["date"]).copy()
        if dated.empty:
            continue
        rolling_basis = out.loc[dated.index, ["date", "_pooled_event_signal", "_pooled_strict_signal"]].set_index("date")
        for window in [20, 60, 120]:
            out.loc[dated.index, f"signal_count_{window}d"] = (
                rolling_basis["_pooled_event_signal"].rolling(f"{window}D", min_periods=1).sum().to_numpy()
            )
        for window in [20, 60]:
            out.loc[dated.index, f"strict_signal_count_{window}d"] = (
                rolling_basis["_pooled_strict_signal"].rolling(f"{window}D", min_periods=1).sum().to_numpy()
            )
    return out.drop(columns=["_pooled_event_signal", "_pooled_strict_signal"])


def add_regime_interaction_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for a, b in [
        ("entry_trigger", "trend_regime"),
        ("entry_trigger", "vol_regime"),
        ("trend_regime", "drawdown_bucket"),
        ("risk_state", "vol_regime"),
        ("candidate_tier", "trend_regime"),
        ("candidate_tier", "vol_regime"),
        ("news_primary_cause_type", "trend_regime"),
        ("news_primary_cause_type", "vol_regime"),
    ]:
        if a in out.columns and b in out.columns:
            out[f"{a}__x__{b}"] = out[a].fillna("UNKNOWN").astype(str) + "__" + out[b].fillna("UNKNOWN").astype(str)
    return out


def add_pooled_feature_engineering(features: pd.DataFrame) -> pd.DataFrame:
    out = add_cross_sectional_features(features)
    out = add_signal_cluster_features(out)
    out = add_regime_interaction_features(out)
    return out


def schema_rows(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    feature_sets = {
        "numeric_feature": set(NUMERIC_FEATURES),
        "bool_feature": set(BOOL_FEATURES),
        "categorical_feature": set(CATEGORICAL_FEATURES),
    }
    for col in features.columns:
        if col in {"symbol", "symbol_group", "date", "signal_idx"}:
            role = "entity_key" if col in {"symbol", "symbol_group"} else "time_key"
        elif col.startswith("label_"):
            role = "label"
        else:
            role = "metadata"
            for candidate_role, values in feature_sets.items():
                if col in values:
                    role = candidate_role
                    break
            if role == "metadata":
                lower = str(col).lower()
                if lower in {"fx_data_available", "tsmc_earnings_pre_5d_window", "tsmc_earnings_post_5d_window", "tsmc_earnings_event_day"}:
                    role = "bool_feature"
                elif lower.startswith(("cs_rank_", "cs_z_", "relative_return_vs_", "signal_count_", "strict_signal_count_", "universe_", "tsmc_", "market_", "peer_", "fx_", "external_")) or lower in {"days_since_prev_signal", "days_since_tsmc_earnings"}:
                    role = "numeric_feature"
                elif "__x__" in lower:
                    role = "categorical_feature"
        rows.append(
            {
                "column": col,
                "role": role,
                "dtype": str(features[col].dtype),
                "missing_rate": float(features[col].isna().mean()) if len(features) else np.nan,
                "non_null_count": int(features[col].notna().sum()) if len(features) else 0,
            }
        )
    for col in labels.columns:
        if col.startswith("label_") and col not in features.columns:
            rows.append(
                {
                    "column": col,
                    "role": "label_detail",
                    "dtype": str(labels[col].dtype),
                    "missing_rate": float(labels[col].isna().mean()) if len(labels) else np.nan,
                    "non_null_count": int(labels[col].notna().sum()) if len(labels) else 0,
                }
            )
    return pd.DataFrame(rows).sort_values(["role", "column"]).reset_index(drop=True)


def check_row(check: str, passed: bool, severity: str, value, details: str = "") -> Dict:
    return {"check": check, "passed": bool(passed), "severity": severity, "value": value, "details": details}


def assign_date_split(date_value) -> str:
    date = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(date):
        return "unknown"
    if date.year <= 2022:
        return "train_2016_2022"
    if date.year == 2023:
        return "validation_2023"
    if date.year == 2024:
        return "test_2024"
    return "final_holdout_2025_2026"


def build_split_manifest(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty or not {"symbol", "date"}.issubset(features.columns):
        return pd.DataFrame(
            columns=[
                "fold_id",
                "split",
                "symbol_count",
                "row_count",
                "trade_ready_event_count",
                "selected_candidate_count",
                "weak_symbol_groups",
                "start_date",
                "end_date",
            ]
        )
    optional_cols = [
        c
        for c in [
            "symbol_group",
            "is_trade_ready_entry_candidate",
            "is_decision_entry_candidate",
        ]
        if c in features.columns
    ]
    frame = features[["symbol", "date", *optional_cols]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["split"] = frame["date"].map(assign_date_split)
    if "is_trade_ready_entry_candidate" not in frame.columns:
        frame["is_trade_ready_entry_candidate"] = False
    if "is_decision_entry_candidate" not in frame.columns:
        frame["is_decision_entry_candidate"] = frame["is_trade_ready_entry_candidate"]
    frame["is_trade_ready_entry_candidate"] = frame["is_trade_ready_entry_candidate"].astype(bool)
    frame["is_decision_entry_candidate"] = frame["is_decision_entry_candidate"].astype(bool)

    rows = []
    for split, group in frame.groupby("split", dropna=False):
        weak_symbol_groups = ""
        if "symbol_group" in group.columns:
            group_counts = (
                group.groupby("symbol_group", dropna=False)
                .agg(
                    trade_ready_event_count=("is_trade_ready_entry_candidate", "sum"),
                    selected_candidate_count=("is_decision_entry_candidate", "sum"),
                )
                .reset_index()
            )
            weak = group_counts[
                (group_counts["trade_ready_event_count"].astype(int) == 0)
                | (group_counts["selected_candidate_count"].astype(int) == 0)
            ]["symbol_group"].dropna().astype(str)
            weak_symbol_groups = "|".join(sorted(set(weak)))
        rows.append(
            {
                "fold_id": str(split),
                "split": split,
                "symbol_count": int(group["symbol"].nunique()),
                "row_count": int(len(group)),
                "trade_ready_event_count": int(group["is_trade_ready_entry_candidate"].sum()),
                "selected_candidate_count": int(group["is_decision_entry_candidate"].sum()),
                "weak_symbol_groups": weak_symbol_groups,
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


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


def build_sample_audit(config: pd.DataFrame, labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    label_frame = labels.copy()
    feature_frame = features.copy()
    if "date" in label_frame.columns:
        label_frame["date"] = pd.to_datetime(label_frame["date"], errors="coerce")
    for _, cfg in config.iterrows():
        symbol = str(cfg.get("symbol", "")).upper()
        symbol_group = str(cfg.get("symbol_group", "semiconductor"))
        symbol_labels = label_frame[label_frame.get("symbol", pd.Series(dtype=object)).astype(str).str.upper().eq(symbol)].copy() if not label_frame.empty and "symbol" in label_frame.columns else pd.DataFrame()
        symbol_features = feature_frame[feature_frame.get("symbol", pd.Series(dtype=object)).astype(str).str.upper().eq(symbol)].copy() if not feature_frame.empty and "symbol" in feature_frame.columns else pd.DataFrame()
        loaded = bool(not symbol_features.empty)
        status_20d = symbol_labels.get("label_status_20d", pd.Series(dtype=object)).astype(str).eq("LABELED") if not symbol_labels.empty else pd.Series(dtype=bool)
        trade_ready = symbol_labels.get("is_trade_ready_entry_candidate", pd.Series(False, index=symbol_labels.index)).astype(bool) if not symbol_labels.empty else pd.Series(dtype=bool)
        model_training_col = "is_model_training_candidate" if "is_model_training_candidate" in symbol_labels.columns else "is_trade_ready_entry_candidate"
        model_training = symbol_labels.get(model_training_col, pd.Series(False, index=symbol_labels.index)).astype(bool) if not symbol_labels.empty else pd.Series(dtype=bool)
        split_dates = pd.to_datetime(symbol_labels.get("date", pd.Series(pd.NaT, index=symbol_labels.index)), errors="coerce") if not symbol_labels.empty else pd.Series(dtype="datetime64[ns]")
        test_holdout = split_dates.ge(pd.Timestamp("2024-01-01")) if not symbol_labels.empty else pd.Series(dtype=bool)
        tsm_like_group = tsm_like_group_for_symbol(symbol, symbol_group)
        base_weight = TSM_LIKE_BASE_GROUP_WEIGHTS[tsm_like_group]
        strict_eligible = cfg.get("strict_eligible", np.nan)
        rows.append(
            {
                "symbol": symbol,
                "symbol_group": symbol_group,
                "strict_eligible": strict_eligible,
                "loaded": loaded,
                "row_count": int(len(symbol_features)),
                "trade_ready_20d_labeled": int((trade_ready & status_20d).sum()) if not symbol_labels.empty else 0,
                "model_training_20d_labeled": int((model_training & status_20d).sum()) if not symbol_labels.empty else 0,
                "test_holdout_trade_ready_20d": int((trade_ready & status_20d & test_holdout).sum()) if not symbol_labels.empty else 0,
                "tsm_like_group": tsm_like_group,
                "tsm_like_weight_mean": base_weight if loaded else np.nan,
                "tsm_like_effective_n_contribution": float(base_weight * base_weight * int((trade_ready & status_20d).sum())) if not symbol_labels.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_quality_checks(config: pd.DataFrame, labels: pd.DataFrame, features: pd.DataFrame, schema: pd.DataFrame, failures: pd.DataFrame, split_manifest: pd.DataFrame) -> pd.DataFrame:
    loaded_symbol_count = int(features["symbol"].nunique()) if "symbol" in features.columns and not features.empty else 0
    trade_ready_20d = 0
    model_training_20d = 0
    if not labels.empty and {"is_trade_ready_entry_candidate", "label_status_20d"}.issubset(labels.columns):
        trade_ready_20d = int((labels["is_trade_ready_entry_candidate"].astype(bool) & labels["label_status_20d"].eq("LABELED")).sum())
    if not labels.empty and "label_status_20d" in labels.columns:
        training_col = "is_model_training_candidate" if "is_model_training_candidate" in labels.columns else "is_trade_ready_entry_candidate"
        if training_col in labels.columns:
            model_training_20d = int((labels[training_col].astype(bool) & labels["label_status_20d"].eq("LABELED")).sum())
    rows = [
        check_row("pooled_config_symbol_count_positive", config["symbol"].nunique() > 0, "CRITICAL", config["symbol"].nunique()),
        check_row("pooled_loaded_symbol_count_at_least_10", loaded_symbol_count >= 10, "CRITICAL", loaded_symbol_count, "Need at least 10 loaded symbols for pooled semiconductor ML."),
        check_row("pooled_input_failures_absent", failures.empty, "WARN", len(failures), "Missing per-symbol outputs are expected until universe runner is executed."),
        check_row("pooled_labels_non_empty", not labels.empty, "CRITICAL", len(labels)),
        check_row("pooled_features_non_empty", not features.empty, "CRITICAL", len(features)),
        check_row("pooled_schema_non_empty", not schema.empty, "CRITICAL", len(schema)),
        check_row("pooled_has_symbol_columns", {"symbol", "symbol_group"}.issubset(features.columns) and {"symbol", "symbol_group"}.issubset(labels.columns), "CRITICAL", "symbol,symbol_group"),
        check_row("pooled_trade_ready_20d_labeled_at_least_500", trade_ready_20d >= 500, "CRITICAL", trade_ready_20d, "Minimum target for prediction decision research."),
        check_row("pooled_model_training_20d_labeled_at_least_10000", model_training_20d >= 10000, "CRITICAL", model_training_20d, "Minimum target for pooled model training research."),
        check_row("research_target_trade_ready_20d_labeled_at_least_10000", trade_ready_20d >= 10000, "WARN", trade_ready_20d, "Expansion target for robust decision evidence."),
        check_row("research_target_model_training_20d_labeled_at_least_100000", model_training_20d >= 100000, "WARN", model_training_20d, "Expansion target for robust pooled model training."),
        check_row("pooled_date_based_split_manifest_available", not split_manifest.empty, "CRITICAL", len(split_manifest), "Splits are assigned by calendar date, never by symbol-random split."),
    ]
    if not labels.empty:
        for horizon in HORIZONS:
            status_col = f"label_status_{horizon}d"
            labeled = int(labels[status_col].eq("LABELED").sum()) if status_col in labels.columns else 0
            rows.append(check_row(f"pooled_labeled_events_{horizon}d_positive", labeled > 0, "CRITICAL", labeled))
    if not features.empty:
        duplicate_keys = int(features.duplicated(["symbol", "date", "signal_idx"]).sum()) if {"symbol", "date", "signal_idx"}.issubset(features.columns) else -1
        rows.append(check_row("pooled_symbol_date_signal_unique", duplicate_keys == 0, "CRITICAL", duplicate_keys))
        feature_count = int(schema["role"].isin(["numeric_feature", "bool_feature", "categorical_feature"]).sum()) if not schema.empty else 0
        rows.append(check_row("pooled_feature_count_at_least_80", feature_count >= 80, "CRITICAL", feature_count, "Current single-symbol schema should expose the expanded feature set."))
        external_cols = [c for c in features.columns if str(c).lower().startswith(("tsmc_", "market_", "peer_", "fx_", "external_"))]
        rows.append(check_row("pooled_external_features_present", bool(external_cols), "WARN", len(external_cols), "EXTERNAL_FEATURES_MISSING if zero."))
    return pd.DataFrame(rows)


def write_report(outdir: Path, config: pd.DataFrame, labels: pd.DataFrame, features: pd.DataFrame, quality: pd.DataFrame, failures: pd.DataFrame, split_manifest: pd.DataFrame) -> None:
    lines = [
        "# TSMC Pooled Prediction Dataset Report",
        "",
        f"- Symbols: {', '.join(config['symbol'].astype(str).str.upper())}",
        f"- Loaded symbols: {features['symbol'].nunique() if 'symbol' in features.columns and not features.empty else 0}",
        f"- Label rows: {len(labels)}",
        f"- Feature rows: {len(features)}",
        "",
        "## Quality",
        "",
        "| Check | Passed | Severity | Value |",
        "|---|---:|---|---:|",
    ]
    for _, row in quality.iterrows():
        lines.append(f"| {row['check']} | {row['passed']} | {row['severity']} | {row['value']} |")
    lines.extend(["", "## Date-Based Splits", "", "| Split | Symbols | Rows | Start | End |", "|---|---:|---:|---|---|"])
    if split_manifest.empty:
        lines.append("| NA | 0 | 0 | NA | NA |")
    else:
        for _, row in split_manifest.iterrows():
            lines.append(f"| {row['split']} | {int(row['symbol_count'])} | {int(row['row_count'])} | {row['start_date']} | {row['end_date']} |")
    lines.extend(["", "## Missing Symbol Inputs", "", "| Symbol | Status | Details |", "|---|---|---|"])
    if failures.empty:
        lines.append("| none | OK |  |")
    else:
        for _, row in failures.iterrows():
            lines.append(f"| {row['symbol']} | {row['status']} | {row['details']} |")
    lines.extend(
        [
            "",
            "## Expansion Contract",
            "- Add more symbols by providing a config CSV with the same rule-engine output paths per symbol.",
            "- Pooled training must split by date and may group/calibrate by symbol; symbol-random split is not allowed.",
            "- Default research split: train 2016-2022, validation 2023, test 2024, final holdout 2025-2026.",
            "",
            "This report is research tooling, not investment advice.",
        ]
    )
    (outdir / "tsm_prediction_pooled_dataset_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pooled cross-sectional prediction dataset.")
    parser.add_argument("--config", default="")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stop-multiple", type=float, default=2.0)
    parser.add_argument("--external-features", default="", help="Optional symbol/date external feature CSV produced by tsm_external_feature_engine.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config) if args.config else None
    config = read_config(config_path)
    external_features_path = Path(args.external_features) if str(args.external_features).strip() else None

    label_parts = []
    feature_parts = []
    scope_parts = []
    failures: List[Dict] = []
    for _, row in config.iterrows():
        if "strict_eligible" in row.index:
            strict_value = str(row.get("strict_eligible", "")).strip().lower()
            if strict_value in {"false", "0", "no"}:
                failures.append(
                    {
                        "symbol": str(row.get("symbol", "UNKNOWN")).upper(),
                        "status": "SKIPPED_STRICT_INELIGIBLE",
                        "details": str(row.get("eligibility_status", "STRICT_ELIGIBILITY_FALSE")),
                    }
                )
                continue
        try:
            labels, features, scope_stats = build_symbol_dataset(row, args.commission_bps, args.slippage_bps, args.stop_multiple, external_features_path)
        except Exception as exc:
            failures.append({"symbol": str(row.get("symbol", "UNKNOWN")).upper(), "status": "SKIPPED_INPUT_UNAVAILABLE", "details": str(exc)})
            continue
        label_parts.append(labels)
        feature_parts.append(features)
        scope_parts.append(scope_stats)

    labels = pd.concat(label_parts, ignore_index=True) if label_parts else pd.DataFrame()
    features = pd.concat(feature_parts, ignore_index=True) if feature_parts else pd.DataFrame()
    features = add_pooled_feature_engineering(features) if not features.empty else features
    scope_stats = pd.concat(scope_parts, ignore_index=True) if scope_parts else pd.DataFrame()
    schema = schema_rows(features, labels) if not features.empty else pd.DataFrame()
    failure_df = pd.DataFrame(failures)
    split_manifest = build_split_manifest(features)
    sample_audit = build_sample_audit(config, labels, features)
    quality = build_quality_checks(config, labels, features, schema, failure_df, split_manifest)

    labels.to_csv(outdir / "tsm_prediction_pooled_label_dataset.csv", index=False)
    features.to_csv(outdir / "tsm_prediction_pooled_feature_matrix.csv", index=False)
    scope_stats.to_csv(outdir / "tsm_prediction_pooled_scope_stats.csv", index=False)
    schema.to_csv(outdir / "tsm_prediction_pooled_schema.csv", index=False)
    split_manifest.to_csv(outdir / "tsm_prediction_pooled_split_manifest.csv", index=False)
    sample_audit.to_csv(outdir / "tsm_prediction_pooled_sample_audit.csv", index=False)
    failure_df.to_csv(outdir / "tsm_prediction_pooled_input_failures.csv", index=False)
    quality.to_csv(outdir / "tsm_prediction_pooled_quality_checks.csv", index=False)
    write_report(outdir, config, labels, features, quality, failure_df, split_manifest)
    print("완료: pooled prediction dataset outputs =", outdir.resolve())
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()

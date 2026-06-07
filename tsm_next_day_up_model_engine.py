#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a next-trading-day close-to-close up probability model.

This engine is intentionally separate from the 20D/60D rule-candidate
meta-label model. It predicts whether the next trading day's close is above
today's close using only data available on the current signal row.

Outputs:
- tsm_next_day_up_label_dataset.csv
- tsm_next_day_up_feature_matrix.csv
- tsm_next_day_up_walk_forward_metrics.csv
- tsm_next_day_up_oos_predictions.csv
- tsm_next_day_up_model_comparison.csv
- tsm_next_day_up_calibration_bins.csv
- tsm_next_day_up_calibration_summary.csv
- tsm_next_day_up_threshold_policy.csv
- tsm_next_day_up_feature_selection_report.csv
- tsm_next_day_up_latest_snapshot.csv
- tsm_next_day_up_quality_checks.csv
- tsm_next_day_up_report.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from tsm_prediction_engine import (
    BOOL_FEATURES,
    CATEGORICAL_FEATURES,
    FORBIDDEN_FEATURE_PATTERNS,
    NEXT_DAY_UP_SCOPE,
    NUMERIC_FEATURES,
    aggregate_calibration_summary,
    aggregate_model_comparison,
    as_float,
    bounded_probability,
    build_feature_association_summary,
    evaluate_prediction_stream,
    latest_prediction_for_horizon,
    load_inputs,
    pct,
    read_csv,
    to_bool,
)


HORIZON_DAYS = 1
NEXT_DAY_CANDIDATE_COL = "is_next_day_up_model_candidate"
OUTPUT_PREFIX = "tsm_next_day_up"


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def check_row(check: str, passed: bool, severity: str, value, tolerance: str = "", details: str = "") -> Dict[str, object]:
    return {
        "check": check,
        "passed": bool(passed),
        "severity": severity,
        "value": value,
        "tolerance": tolerance,
        "details": details,
    }


def valid_price(value: object) -> bool:
    return pd.notna(value) and math.isfinite(float(value)) and float(value) > 0


def unavailable_label_fields(status: str) -> Dict[str, object]:
    horizon = HORIZON_DAYS
    return {
        f"label_status_{horizon}d": status,
        f"label_success_{horizon}d": np.nan,
        f"label_stop_survival_{horizon}d": np.nan,
        f"label_hit_1r_before_stop_{horizon}d": np.nan,
        f"label_hit_2r_before_stop_{horizon}d": np.nan,
        f"label_positive_return_{horizon}d": np.nan,
        f"label_expected_r_{horizon}d": np.nan,
        f"label_mfe_r_{horizon}d": np.nan,
        f"label_mae_r_{horizon}d": np.nan,
        f"label_time_to_1r_{horizon}d": np.nan,
        f"label_time_to_2r_{horizon}d": np.nan,
        f"label_time_to_stop_{horizon}d": np.nan,
        f"label_ambiguous_stop_1r_same_day_{horizon}d": np.nan,
        f"label_ambiguous_stop_2r_same_day_{horizon}d": np.nan,
        f"label_gap_through_stop_{horizon}d": np.nan,
        f"label_entry_gap_pct_{horizon}d": np.nan,
        f"label_first_touch_type_{horizon}d": status,
        f"label_time_to_first_touch_{horizon}d": np.nan,
        f"label_mfe_before_stop_{horizon}d": np.nan,
        f"label_mae_before_profit_{horizon}d": np.nan,
        f"label_event_regime_at_entry_{horizon}d": status,
        f"label_net_return_pct_{horizon}d": np.nan,
        f"label_gross_return_pct_{horizon}d": np.nan,
        f"label_exit_reason_{horizon}d": status,
        f"label_exit_type_clean_{horizon}d": status,
        f"label_entry_date_{horizon}d": "",
        f"label_exit_date_{horizon}d": "",
        f"label_entry_price_{horizon}d": np.nan,
        f"label_exit_price_{horizon}d": np.nan,
        f"label_stop_price_{horizon}d": np.nan,
        f"label_1r_price_{horizon}d": np.nan,
        f"label_2r_price_{horizon}d": np.nan,
        f"label_holding_trading_days_{horizon}d": np.nan,
        f"label_overlap_count_{horizon}d": 0,
        f"sample_uniqueness_weight_{horizon}d": 0.0,
        "next_day_close_to_close_return_pct": np.nan,
    }


def build_next_day_up_label_dataset(signals: pd.DataFrame) -> pd.DataFrame:
    require_columns(signals, ["date", "signal_idx", "close"], "signals")
    work = signals.sort_values("signal_idx").reset_index(drop=True).copy()
    base_cols = [
        "date",
        "signal_idx",
        "symbol",
        "is_event_candidate",
        "is_actionable_entry_candidate",
        "is_trade_ready_entry_candidate",
        "is_entry_research_candidate",
        "is_risk_research_candidate",
        "is_model_training_candidate",
        "is_decision_entry_candidate",
        "candidate_tier",
        "entry_gate_status",
        "candidate_scope",
        "prediction_universe",
        "entry_trigger",
        "trade_action",
        "close",
        "atr_14",
        "atr_14_pct",
        "score_price_algo_total",
        "risk_state",
        "trend_regime",
        "vol_regime",
        "drawdown_bucket",
    ]
    rows: List[Dict[str, object]] = []
    close = pd.to_numeric(work["close"], errors="coerce")
    high = pd.to_numeric(work["high"], errors="coerce") if "high" in work.columns else pd.Series(np.nan, index=work.index)
    low = pd.to_numeric(work["low"], errors="coerce") if "low" in work.columns else pd.Series(np.nan, index=work.index)
    atr_pct = pd.to_numeric(work["atr_14_pct"], errors="coerce") if "atr_14_pct" in work.columns else pd.Series(np.nan, index=work.index)

    for idx, source in work.iterrows():
        out = {c: source.get(c) for c in base_cols if c in work.columns}
        out[NEXT_DAY_CANDIDATE_COL] = True
        next_idx = idx + 1
        if next_idx >= len(work):
            out.update(unavailable_label_fields("UNAVAILABLE_FUTURE_WINDOW"))
            rows.append(out)
            continue
        current_close = close.iloc[idx]
        next_close = close.iloc[next_idx]
        if not valid_price(current_close) or not valid_price(next_close):
            out.update(unavailable_label_fields("INVALID_CLOSE_DATA"))
            rows.append(out)
            continue

        return_pct = float(next_close / current_close - 1.0)
        success = int(return_pct > 0.0)
        atr_pct_value = float(atr_pct.iloc[idx]) if pd.notna(atr_pct.iloc[idx]) and float(atr_pct.iloc[idx]) > 0 else np.nan
        expected_r = return_pct / atr_pct_value if pd.notna(atr_pct_value) else np.nan
        next_high = high.iloc[next_idx]
        next_low = low.iloc[next_idx]
        mfe_r = ((float(next_high) / float(current_close) - 1.0) / atr_pct_value) if pd.notna(next_high) and pd.notna(atr_pct_value) else np.nan
        mae_r = ((float(next_low) / float(current_close) - 1.0) / atr_pct_value) if pd.notna(next_low) and pd.notna(atr_pct_value) else np.nan
        exit_reason = "NEXT_CLOSE_UP" if success else "NEXT_CLOSE_DOWN_OR_FLAT"
        event_regime = f"{source.get('trend_regime', 'UNKNOWN')}|{source.get('vol_regime', 'UNKNOWN')}"
        horizon = HORIZON_DAYS
        out.update(
            {
                f"label_status_{horizon}d": "LABELED",
                f"label_success_{horizon}d": success,
                f"label_stop_survival_{horizon}d": 1,
                f"label_hit_1r_before_stop_{horizon}d": 0,
                f"label_hit_2r_before_stop_{horizon}d": 0,
                f"label_positive_return_{horizon}d": success,
                f"label_expected_r_{horizon}d": expected_r,
                f"label_mfe_r_{horizon}d": mfe_r,
                f"label_mae_r_{horizon}d": mae_r,
                f"label_time_to_1r_{horizon}d": np.nan,
                f"label_time_to_2r_{horizon}d": np.nan,
                f"label_time_to_stop_{horizon}d": np.nan,
                f"label_ambiguous_stop_1r_same_day_{horizon}d": 0,
                f"label_ambiguous_stop_2r_same_day_{horizon}d": 0,
                f"label_gap_through_stop_{horizon}d": 0,
                f"label_entry_gap_pct_{horizon}d": 0.0,
                f"label_first_touch_type_{horizon}d": exit_reason,
                f"label_time_to_first_touch_{horizon}d": np.nan,
                f"label_mfe_before_stop_{horizon}d": np.nan,
                f"label_mae_before_profit_{horizon}d": np.nan,
                f"label_event_regime_at_entry_{horizon}d": event_regime,
                f"label_net_return_pct_{horizon}d": pct(return_pct),
                f"label_gross_return_pct_{horizon}d": pct(return_pct),
                f"label_exit_reason_{horizon}d": exit_reason,
                f"label_exit_type_clean_{horizon}d": "NEXT_CLOSE",
                f"label_entry_date_{horizon}d": source.get("date"),
                f"label_exit_date_{horizon}d": work.iloc[next_idx].get("date"),
                f"label_entry_price_{horizon}d": float(current_close),
                f"label_exit_price_{horizon}d": float(next_close),
                f"label_stop_price_{horizon}d": np.nan,
                f"label_1r_price_{horizon}d": np.nan,
                f"label_2r_price_{horizon}d": np.nan,
                f"label_holding_trading_days_{horizon}d": 1,
                f"label_overlap_count_{horizon}d": 1,
                f"sample_uniqueness_weight_{horizon}d": 1.0,
                "next_day_close_to_close_return_pct": pct(return_pct),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def build_next_day_up_feature_matrix(signals: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [*NUMERIC_FEATURES, *BOOL_FEATURES, *CATEGORICAL_FEATURES]
    meta = [
        "date",
        "signal_idx",
        "symbol",
        NEXT_DAY_CANDIDATE_COL,
        "is_event_candidate",
        "is_actionable_entry_candidate",
        "is_trade_ready_entry_candidate",
        "is_entry_research_candidate",
        "is_risk_research_candidate",
        "is_model_training_candidate",
        "is_decision_entry_candidate",
        "candidate_tier",
        "entry_gate_status",
        "candidate_scope",
        "prediction_universe",
    ]
    work = signals.copy()
    work[NEXT_DAY_CANDIDATE_COL] = True
    available = [c for c in feature_cols if c in work.columns and c not in set(meta)]
    present_meta = [c for c in meta if c in work.columns]
    features = work[[*present_meta, *available]].copy()
    for col in BOOL_FEATURES:
        if col in features.columns:
            features[col] = features[col].map(lambda x: 1 if to_bool(x) else 0)
    label_cols = [c for c in labels.columns if c.startswith("label_") or c.startswith("next_day_")]
    merged = features.merge(labels[["date", "signal_idx", *label_cols]], on=["date", "signal_idx"], how="left")
    return merged[[*present_meta, *available, *label_cols]]


def parse_thresholds(value: str) -> List[float]:
    return [float(v.strip()) for v in str(value).split(",") if v.strip()]


def key_value_frame_to_dict(frame: pd.DataFrame) -> Dict[str, object]:
    if frame.empty or not {"field", "value"}.issubset(frame.columns):
        return {}
    return dict(zip(frame["field"].astype(str), frame["value"]))


def latest_status(probability: float, threshold: float, oos_event_count: int, performance_pass: bool) -> str:
    if oos_event_count < 100:
        return "INSUFFICIENT_OOS_EVIDENCE"
    if pd.isna(probability) or pd.isna(threshold):
        return "INSUFFICIENT_DATA"
    if not performance_pass:
        return "DISPLAY_ONLY_MODEL_QUALITY_NOT_PASSED"
    if probability >= threshold:
        return "UP_BIAS"
    if probability >= 0.50:
        return "MARGINAL_UP_PROBABILITY"
    return "NO_UP_EDGE"


def build_latest_snapshot(feature_matrix: pd.DataFrame, comparison: pd.DataFrame, threshold_policy: pd.DataFrame) -> pd.DataFrame:
    if feature_matrix.empty:
        values = {
            "next_day_prediction_asof_date": "",
            "next_day_prediction_scope": NEXT_DAY_UP_SCOPE,
            "next_day_best_model_1d": "NA",
            "next_day_p_up_1d": np.nan,
            "next_day_p_down_1d": np.nan,
            "next_day_threshold_1d": np.nan,
            "next_day_prediction_signal_status": "INSUFFICIENT_DATA",
            "next_day_model_quality_status": "NO_MODEL",
            "next_day_label_definition": "next_trading_day_close_gt_today_close",
        }
        return pd.DataFrame([{"field": k, "value": v} for k, v in values.items()])

    latest_row = feature_matrix.iloc[[-1]].copy()
    raw = latest_prediction_for_horizon(
        feature_matrix,
        comparison,
        threshold_policy,
        HORIZON_DAYS,
        latest_row,
        NEXT_DAY_UP_SCOPE,
        NEXT_DAY_CANDIDATE_COL,
    )
    p_up = as_float(raw.get("p_success_1d"))
    p_down = 1.0 - p_up if pd.notna(p_up) else np.nan
    threshold = as_float(raw.get("threshold_1d"))
    oos_count = int(as_float(raw.get("oos_event_count_1d"), 0))
    performance_pass = to_bool(raw.get("performance_quality_pass_1d", False))
    model_quality_reasons = str(raw.get("model_quality_block_reasons_1d", "NO_MODEL"))
    model_quality_status = "PERFORMANCE_PASS_DISPLAY_ONLY" if performance_pass else model_quality_reasons
    values = {
        "next_day_prediction_asof_date": latest_row["date"].iloc[0],
        "next_day_prediction_scope": NEXT_DAY_UP_SCOPE,
        "next_day_label_definition": "next_trading_day_close_gt_today_close",
        "next_day_best_model_1d": raw.get("best_model_1d", "NA"),
        "next_day_p_up_1d": p_up,
        "next_day_p_down_1d": p_down,
        "next_day_threshold_1d": threshold,
        "next_day_confidence_band_1d": raw.get("confidence_band_1d", "NA"),
        "next_day_p_up_lower_80_1d": raw.get("p_success_lower_80_1d", np.nan),
        "next_day_p_up_upper_80_1d": raw.get("p_success_upper_80_1d", np.nan),
        "next_day_prediction_signal_status": latest_status(p_up, threshold, oos_count, performance_pass),
        "next_day_model_quality_status": model_quality_status,
        "next_day_model_quality_block_reasons_1d": model_quality_reasons,
        "next_day_prediction_quality_pass_1d": raw.get("prediction_quality_pass_1d", False),
        "next_day_performance_quality_pass_1d": performance_pass,
        "next_day_performance_quality_block_reasons_1d": raw.get("performance_quality_block_reasons_1d", "NO_MODEL"),
        "next_day_oos_event_count_1d": oos_count,
        "next_day_selected_oos_event_count_1d": raw.get("selected_oos_event_count_1d", 0),
        "next_day_brier_score_1d": raw.get("brier_score_1d", np.nan),
        "next_day_brier_improvement_pct_1d": raw.get("brier_improvement_pct_1d", np.nan),
        "next_day_ece_1d": raw.get("ece_1d", np.nan),
        "next_day_pr_auc_1d": raw.get("pr_auc_1d", np.nan),
        "next_day_expectancy_improvement_pct_1d": raw.get("expectancy_improvement_pct_1d", np.nan),
        "next_day_selected_minus_all_pct_1d": raw.get("selected_minus_rule_all_pct_1d", np.nan),
        "next_day_threshold_iqr_1d": raw.get("threshold_iqr_1d", np.nan),
    }
    return pd.DataFrame([{"field": k, "value": v} for k, v in values.items()])


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


def build_quality_checks(
    labels: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    threshold_policy: pd.DataFrame,
    oos_predictions: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    latest: pd.DataFrame,
    latest_snapshot_updated: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    status_col = "label_status_1d"
    labeled_count = int(labels[status_col].eq("LABELED").sum()) if status_col in labels.columns else 0
    rows.append(check_row("next_day_up_labels_non_empty", not labels.empty, "CRITICAL", len(labels)))
    rows.append(check_row("next_day_up_labeled_rows_at_least_500", labeled_count >= 500, "CRITICAL", labeled_count, ">=500"))
    if "label_success_1d" in labels.columns:
        clean = pd.to_numeric(labels.loc[labels[status_col].eq("LABELED"), "label_success_1d"], errors="coerce").dropna()
        rows.append(check_row("next_day_up_label_binary", bool(clean.isin([0, 1]).all() and not clean.empty), "CRITICAL", len(clean)))
        rows.append(check_row("next_day_up_label_has_two_classes", clean.nunique() == 2, "CRITICAL", int(clean.nunique())))
    feature_cols = [c for c in feature_matrix.columns if c in NUMERIC_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES]
    forbidden = [c for c in feature_cols if any(pattern in c.lower() for pattern in FORBIDDEN_FEATURE_PATTERNS)]
    rows.append(check_row("next_day_up_feature_allowlist_has_no_forbidden_columns", not forbidden, "CRITICAL", ",".join(forbidden) if forbidden else "ok"))
    rows.append(check_row("next_day_up_feature_matrix_non_empty", not feature_matrix.empty, "CRITICAL", len(feature_matrix)))
    rows.append(check_row("next_day_up_walk_forward_metrics_non_empty", not metrics.empty, "CRITICAL", len(metrics)))
    rows.append(check_row("next_day_up_oos_predictions_non_empty", not oos_predictions.empty, "CRITICAL", len(oos_predictions)))
    if not oos_predictions.empty and {"date", "train_end_date"}.issubset(oos_predictions.columns):
        oos_dates = pd.to_datetime(oos_predictions["date"], errors="coerce")
        train_end = pd.to_datetime(oos_predictions["train_end_date"], errors="coerce")
        rows.append(check_row("next_day_up_oos_predictions_after_train_end_date", bool((oos_dates > train_end).all()), "CRITICAL", f"violations={int((oos_dates <= train_end).sum())}"))
    else:
        rows.append(check_row("next_day_up_oos_predictions_after_train_end_date", False, "CRITICAL", "missing_columns"))
    rows.append(check_row("next_day_up_model_comparison_non_empty", not comparison.empty, "CRITICAL", len(comparison)))
    rows.append(check_row("next_day_up_calibration_summary_available", not calibration_summary.empty, "CRITICAL", len(calibration_summary)))
    rows.append(
        check_row(
            "next_day_up_threshold_policy_not_from_test",
            bool(not threshold_policy.empty and "threshold_source" in threshold_policy.columns and not threshold_policy["threshold_source"].astype(str).str.contains("TEST", case=False, na=False).any()),
            "CRITICAL",
            "ok" if not threshold_policy.empty else "missing",
        )
    )
    latest_map = key_value_frame_to_dict(latest)
    rows.append(check_row("next_day_up_latest_snapshot_available", bool(latest_map), "CRITICAL", "ok" if latest_map else "missing"))
    rows.append(check_row("next_day_up_merged_into_latest_prediction_snapshot", latest_snapshot_updated, "WARN", latest_snapshot_updated, details="False is ok when running in an isolated outdir without a baseline latest snapshot."))
    return pd.DataFrame(rows)


def write_report(
    outdir: Path,
    labels: pd.DataFrame,
    comparison: pd.DataFrame,
    latest: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    latest_map = key_value_frame_to_dict(latest)
    critical = quality[quality["severity"].eq("CRITICAL")] if not quality.empty and "severity" in quality.columns else quality
    critical_passed = bool(critical["passed"].all()) if not critical.empty else False
    lines = [
        "# Next-Day Up Prediction Report",
        "",
        "- Label: next trading day close > current close.",
        "- Scope: all daily signal rows with a next trading day close.",
        "- Use: diagnostic probability only; it does not enable live trading or 20D decision support.",
        "- Directional diagnostic pass allows only near-neutral Brier loss when calibration, PR-AUC edge, selected uplift, fold stability, and sample-size gates all pass.",
        f"- Labeled rows: {int(labels['label_status_1d'].eq('LABELED').sum()) if 'label_status_1d' in labels.columns else 0}",
        f"- Critical quality checks passed: {critical_passed}",
        "",
        "## Latest",
        f"- As of: {latest_map.get('next_day_prediction_asof_date', 'NA')}",
        f"- Model: {latest_map.get('next_day_best_model_1d', 'NA')}",
        f"- P(up): {latest_map.get('next_day_p_up_1d', 'NA')}",
        f"- Threshold: {latest_map.get('next_day_threshold_1d', 'NA')}",
        f"- Status: {latest_map.get('next_day_prediction_signal_status', 'NA')}",
        f"- Quality: {latest_map.get('next_day_model_quality_status', 'NA')}",
        "",
        "## Top Models",
        "",
        "| Scope | Horizon | Model | OOS | Brier Improvement | ECE | PR AUC | Directional Pass | Perf Pass | Block Reasons |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    if comparison.empty:
        lines.append("| NA | NA | NA | 0 | NA | NA | NA | False | False | NO_MODEL |")
    else:
        top = comparison.sort_values("rank_score", ascending=False).head(10) if "rank_score" in comparison.columns else comparison.head(10)
        for _, row in top.iterrows():
            lines.append(
                f"| {row.get('candidate_scope', '')} | {row.get('horizon_days', '')} | {row.get('model_name', '')} | "
                f"{int(as_float(row.get('oos_event_count'), 0))} | {as_float(row.get('brier_improvement_pct')):.2f}% | "
                f"{as_float(row.get('ece')):.4f} | {as_float(row.get('pr_auc')):.4f} | "
                f"{to_bool(row.get('directional_diagnostic_quality_pass', False))} | "
                f"{to_bool(row.get('performance_quality_pass', False))} | {row.get('quality_block_reasons', '')} |"
            )
    lines.extend(["", "This report is research tooling, not investment advice."])
    (outdir / f"{OUTPUT_PREFIX}_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_next_day_artifacts(
    *,
    signals_path: Path,
    risk_policy_path: Path,
    trade_log_path: Path,
    enriched_path: Path,
    external_features_path: Optional[Path],
    intraday_features_path: Optional[Path],
    outdir: Path,
    latest_prediction_path: Path,
    symbol: str,
    train_days: int,
    validation_days: int,
    test_days: int,
    step_days: int,
    gap_days: int,
    thresholds: List[float],
    min_validation_trades: int,
    default_threshold: float,
    calibration_bins: int,
    update_latest: bool,
) -> Dict[str, pd.DataFrame | bool]:
    signals, _ = load_inputs(
        signals_path,
        risk_policy_path,
        trade_log_path,
        enriched_path,
        external_features_path,
        intraday_features_path,
        symbol=str(symbol).upper(),
    )
    labels = build_next_day_up_label_dataset(signals)
    feature_matrix = build_next_day_up_feature_matrix(signals, labels)
    metrics, threshold_policy, calibration, feature_selection, oos_predictions, fold_manifest = evaluate_prediction_stream(
        feature_matrix,
        HORIZON_DAYS,
        NEXT_DAY_UP_SCOPE,
        NEXT_DAY_CANDIDATE_COL,
        train_days,
        validation_days,
        test_days,
        step_days,
        gap_days,
        thresholds,
        min_validation_trades,
        default_threshold,
        calibration_bins,
    )
    calibration_summary = aggregate_calibration_summary(calibration)
    comparison = aggregate_model_comparison(metrics)
    feature_association_summary = build_feature_association_summary(feature_selection)
    latest = build_latest_snapshot(feature_matrix, comparison, threshold_policy)
    latest_snapshot_updated = False
    if update_latest:
        latest_snapshot_updated = merge_latest_prediction_snapshot(latest_prediction_path, latest)
    quality = build_quality_checks(
        labels,
        feature_matrix,
        metrics,
        comparison,
        threshold_policy,
        oos_predictions,
        calibration_summary,
        latest,
        latest_snapshot_updated,
    )
    return {
        "labels": labels,
        "feature_matrix": feature_matrix,
        "metrics": metrics,
        "oos_predictions": oos_predictions,
        "comparison": comparison,
        "calibration": calibration,
        "calibration_summary": calibration_summary,
        "threshold_policy": threshold_policy,
        "feature_selection": feature_selection,
        "feature_association_summary": feature_association_summary,
        "fold_manifest": fold_manifest,
        "latest": latest,
        "quality": quality,
        "latest_snapshot_updated": latest_snapshot_updated,
    }


def write_next_day_artifacts(outdir: Path, artifacts: Dict[str, pd.DataFrame | bool]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    labels = artifacts["labels"]
    comparison = artifacts["comparison"]
    latest = artifacts["latest"]
    quality = artifacts["quality"]
    assert isinstance(labels, pd.DataFrame)
    assert isinstance(comparison, pd.DataFrame)
    assert isinstance(latest, pd.DataFrame)
    assert isinstance(quality, pd.DataFrame)
    artifacts["labels"].to_csv(outdir / f"{OUTPUT_PREFIX}_label_dataset.csv", index=False)
    artifacts["feature_matrix"].to_csv(outdir / f"{OUTPUT_PREFIX}_feature_matrix.csv", index=False)
    artifacts["metrics"].to_csv(outdir / f"{OUTPUT_PREFIX}_walk_forward_metrics.csv", index=False)
    artifacts["oos_predictions"].to_csv(outdir / f"{OUTPUT_PREFIX}_oos_predictions.csv", index=False)
    artifacts["comparison"].to_csv(outdir / f"{OUTPUT_PREFIX}_model_comparison.csv", index=False)
    artifacts["calibration"].to_csv(outdir / f"{OUTPUT_PREFIX}_calibration_bins.csv", index=False)
    artifacts["calibration_summary"].to_csv(outdir / f"{OUTPUT_PREFIX}_calibration_summary.csv", index=False)
    artifacts["threshold_policy"].to_csv(outdir / f"{OUTPUT_PREFIX}_threshold_policy.csv", index=False)
    artifacts["feature_selection"].to_csv(outdir / f"{OUTPUT_PREFIX}_feature_selection_report.csv", index=False)
    artifacts["feature_association_summary"].to_csv(outdir / f"{OUTPUT_PREFIX}_feature_association_summary.csv", index=False)
    artifacts["fold_manifest"].to_csv(outdir / f"{OUTPUT_PREFIX}_fold_manifest.csv", index=False)
    artifacts["latest"].to_csv(outdir / f"{OUTPUT_PREFIX}_latest_snapshot.csv", index=False)
    artifacts["quality"].to_csv(outdir / f"{OUTPUT_PREFIX}_quality_checks.csv", index=False)
    write_report(outdir, labels, comparison, latest, quality)


def latest_snapshot_row(latest: pd.DataFrame, symbol: str, symbol_group: str = "") -> Dict[str, object]:
    values = key_value_frame_to_dict(latest)
    values["symbol"] = str(symbol).upper()
    values["symbol_group"] = symbol_group
    values["date"] = values.get("next_day_prediction_asof_date", "")
    return values


def build_fast_symbol_latest_snapshot(signals_path: Path, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = read_csv(signals_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(signals.get("close"), errors="coerce")
    next_close = close.shift(-1)
    labeled = signals.copy()
    labeled["label_success_1d"] = (next_close > close).astype(float)
    valid = close.notna() & next_close.notna() & (close > 0) & (next_close > 0)
    labeled = labeled[valid].copy()
    latest = signals.iloc[-1] if not signals.empty else pd.Series(dtype=object)
    overall_rate = float(labeled["label_success_1d"].mean()) if not labeled.empty else np.nan
    recent = labeled.tail(252)
    recent_rate = float(recent["label_success_1d"].mean()) if not recent.empty else overall_rate
    trend = str(latest.get("trend_regime", "UNKNOWN"))
    vol = str(latest.get("vol_regime", "UNKNOWN"))
    group = labeled[
        labeled.get("trend_regime", pd.Series("", index=labeled.index)).astype(str).eq(trend)
        & labeled.get("vol_regime", pd.Series("", index=labeled.index)).astype(str).eq(vol)
    ]
    group_rate = float(group["label_success_1d"].mean()) if not group.empty else np.nan
    prior = 40.0
    base = recent_rate if pd.notna(recent_rate) else overall_rate
    if pd.notna(group_rate) and pd.notna(base):
        p_up = (float(group["label_success_1d"].sum()) + prior * base) / (len(group) + prior)
        model_name = "empirical_bayes_trend_vol_rate"
        effective_n = len(group)
    else:
        p_up = base
        model_name = "recent_symbol_base_rate"
        effective_n = len(recent)
    p_up = bounded_probability(p_up)
    p_down = 1.0 - p_up if pd.notna(p_up) else np.nan
    threshold = bounded_probability(max(0.49, min(0.55, overall_rate))) if pd.notna(overall_rate) else 0.50
    if pd.isna(p_up):
        status = "INSUFFICIENT_DATA"
        band = "NA"
    elif p_up >= threshold:
        status = "DISPLAY_ONLY_UP_BIAS"
        band = "LOW_POSITIVE" if p_up < 0.60 else "MEDIUM"
    elif p_up >= 0.50:
        status = "DISPLAY_ONLY_MARGINAL_UP_PROBABILITY"
        band = "LOW_POSITIVE"
    else:
        status = "DISPLAY_ONLY_NO_UP_EDGE"
        band = "LOW_NEGATIVE"
    values = {
        "next_day_prediction_asof_date": latest.get("date", ""),
        "next_day_prediction_scope": NEXT_DAY_UP_SCOPE,
        "next_day_label_definition": "next_trading_day_close_gt_today_close",
        "next_day_best_model_1d": model_name,
        "next_day_p_up_1d": p_up,
        "next_day_p_down_1d": p_down,
        "next_day_threshold_1d": threshold,
        "next_day_confidence_band_1d": band,
        "next_day_prediction_signal_status": status,
        "next_day_model_quality_status": "DISPLAY_ONLY_FAST_EMPIRICAL_UNIVERSE",
        "next_day_model_quality_block_reasons_1d": "DISPLAY_ONLY_FAST_EMPIRICAL_UNIVERSE",
        "next_day_prediction_quality_pass_1d": False,
        "next_day_performance_quality_pass_1d": False,
        "next_day_performance_quality_block_reasons_1d": "DISPLAY_ONLY_FAST_EMPIRICAL_UNIVERSE",
        "next_day_oos_event_count_1d": int(len(labeled)),
        "next_day_selected_oos_event_count_1d": int(effective_n),
        "next_day_brier_improvement_pct_1d": np.nan,
        "next_day_ece_1d": np.nan,
        "next_day_pr_auc_1d": np.nan,
        "next_day_expectancy_improvement_pct_1d": np.nan,
    }
    quality = pd.DataFrame(
        [
            check_row("next_day_up_fast_signals_non_empty", not signals.empty, "CRITICAL", len(signals)),
            check_row("next_day_up_fast_labeled_rows_at_least_500", len(labeled) >= 500, "CRITICAL", len(labeled), ">=500"),
            check_row("next_day_up_fast_latest_snapshot_available", bool(values["next_day_prediction_asof_date"]), "CRITICAL", values["next_day_prediction_asof_date"]),
        ]
    )
    return pd.DataFrame([{"field": key, "value": value} for key, value in values.items()]), quality


def write_fast_symbol_artifacts(outdir: Path, latest: pd.DataFrame, quality: pd.DataFrame) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    latest.to_csv(outdir / f"{OUTPUT_PREFIX}_latest_snapshot.csv", index=False)
    quality.to_csv(outdir / f"{OUTPUT_PREFIX}_quality_checks.csv", index=False)


def _config_path_value(row: pd.Series, key: str, fallback: Path) -> Path:
    value = row.get(key)
    if pd.notna(value) and str(value).strip():
        return Path(str(value))
    return fallback


def run_universe(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    config = read_csv(Path(args.universe_config))
    if "enabled" in config.columns:
        config = config[config["enabled"].map(to_bool)].copy()
    rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    thresholds = parse_thresholds(args.thresholds)
    external_path = Path(args.external_features) if str(args.external_features).strip() else None
    intraday_path = Path(args.intraday_features) if str(args.intraday_features).strip() else None
    mirror_symbol = str(args.symbol).upper()

    for _, row in config.iterrows():
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        symbol_group = str(row.get("symbol_group", ""))
        symbol_rule_dir = _config_path_value(row, "rule_outdir", Path(args.universe_rule_root) / symbol)
        symbol_data_dir = _config_path_value(row, "data_outdir", Path(args.universe_data_root) / symbol)
        signals_path = _config_path_value(row, "signals", symbol_rule_dir / "tsm_daily_algorithmic_signals.csv")
        risk_policy_path = _config_path_value(row, "risk_policy", symbol_rule_dir / "tsm_risk_policy_daily.csv")
        trade_log_path = _config_path_value(row, "trade_log", symbol_rule_dir / "tsm_backtest_trade_log.csv")
        enriched_path = _config_path_value(row, "enriched", symbol_data_dir / "tsm_daily_10y_enriched.csv")
        latest_prediction_path = symbol_rule_dir / "tsm_latest_prediction_snapshot.csv"
        required = [signals_path, risk_policy_path, trade_log_path, enriched_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            manifest_rows.append(
                {
                    "symbol": symbol,
                    "status": "SKIPPED_MISSING_INPUT",
                    "details": "|".join(missing),
                }
            )
            continue
        try:
            if args.aggregate_full_models:
                artifacts = build_next_day_artifacts(
                    signals_path=signals_path,
                    risk_policy_path=risk_policy_path,
                    trade_log_path=trade_log_path,
                    enriched_path=enriched_path,
                    external_features_path=external_path,
                    intraday_features_path=intraday_path,
                    outdir=symbol_rule_dir,
                    latest_prediction_path=latest_prediction_path,
                    symbol=symbol,
                    train_days=args.train_days,
                    validation_days=args.validation_days,
                    test_days=args.test_days,
                    step_days=args.step_days,
                    gap_days=args.gap_days,
                    thresholds=thresholds,
                    min_validation_trades=args.min_validation_trades,
                    default_threshold=args.default_threshold,
                    calibration_bins=args.calibration_bins,
                    update_latest=not args.no_update_latest,
                )
                write_next_day_artifacts(symbol_rule_dir, artifacts)
                latest = artifacts["latest"]
                assert isinstance(latest, pd.DataFrame)
            else:
                latest, quality = build_fast_symbol_latest_snapshot(signals_path, symbol)
                write_fast_symbol_artifacts(symbol_rule_dir, latest, quality)
                if not args.no_update_latest:
                    merge_latest_prediction_snapshot(latest_prediction_path, latest)
            rows.append(latest_snapshot_row(latest, symbol, symbol_group))
            if symbol == mirror_symbol:
                if args.aggregate_full_models:
                    write_next_day_artifacts(outdir, artifacts)
                else:
                    write_fast_symbol_artifacts(outdir, latest, quality)
                if not args.no_update_latest:
                    merge_latest_prediction_snapshot(Path(args.latest_prediction), latest)
            manifest_rows.append(
                {
                    "symbol": symbol,
                    "status": "OK",
                    "details": str(symbol_rule_dir),
                }
            )
        except Exception as exc:
            manifest_rows.append(
                {
                    "symbol": symbol,
                    "status": "FAILED",
                    "details": str(exc),
                }
            )

    pd.DataFrame(rows).to_csv(outdir / f"{OUTPUT_PREFIX}_universe_latest_predictions.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(outdir / f"{OUTPUT_PREFIX}_universe_manifest.csv", index=False)
    print("completed: universe next-day up prediction outputs =", outdir.resolve())
    if rows:
        print(pd.DataFrame(rows)[["symbol", "date", "next_day_p_up_1d", "next_day_prediction_signal_status"]].to_string(index=False))
    else:
        print("no universe next-day predictions generated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build next-day close-to-close up probability outputs.")
    parser.add_argument("--signals", default="tsm_price_rule_output/tsm_daily_algorithmic_signals.csv")
    parser.add_argument("--trade-log", default="tsm_price_rule_output/tsm_backtest_trade_log.csv")
    parser.add_argument("--risk-policy", default="tsm_price_rule_output/tsm_risk_policy_daily.csv")
    parser.add_argument("--enriched", default="output/tsm_daily_10y_enriched.csv")
    parser.add_argument("--external-features", default="")
    parser.add_argument("--intraday-features", default="")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--latest-prediction", default="tsm_price_rule_output/tsm_latest_prediction_snapshot.csv")
    parser.add_argument("--symbol", default="TSM")
    parser.add_argument("--aggregate-universe", action="store_true", help="Build next-day outputs for every enabled symbol in the universe config.")
    parser.add_argument("--aggregate-full-models", action="store_true", help="Use the full walk-forward model loop for each universe symbol. Slower; default aggregate mode writes fast display snapshots.")
    parser.add_argument("--universe-config", default="", help="Universe config with per-symbol signal/risk/enriched paths.")
    parser.add_argument("--universe-rule-root", default="tsm_price_rule_output/universe")
    parser.add_argument("--universe-data-root", default="output/universe")
    parser.add_argument("--train-days", type=int, default=756)
    parser.add_argument("--validation-days", type=int, default=40)
    parser.add_argument("--test-days", type=int, default=252)
    parser.add_argument("--step-days", type=int, default=252)
    parser.add_argument("--gap-days", type=int, default=1)
    parser.add_argument("--thresholds", default="0.50,0.525,0.55,0.575,0.60")
    parser.add_argument("--default-threshold", type=float, default=0.55)
    parser.add_argument("--min-validation-trades", type=int, default=40)
    parser.add_argument("--calibration-bins", type=int, default=5)
    parser.add_argument("--no-update-latest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_universe:
        if not str(args.universe_config).strip():
            raise ValueError("--aggregate-universe requires --universe-config")
        run_universe(args)
        return

    external_path = Path(args.external_features) if str(args.external_features).strip() else None
    intraday_path = Path(args.intraday_features) if str(args.intraday_features).strip() else None
    artifacts = build_next_day_artifacts(
        signals_path=Path(args.signals),
        risk_policy_path=Path(args.risk_policy),
        trade_log_path=Path(args.trade_log),
        enriched_path=Path(args.enriched),
        external_features_path=external_path,
        intraday_features_path=intraday_path,
        outdir=outdir,
        latest_prediction_path=Path(args.latest_prediction),
        symbol=str(args.symbol).upper(),
        train_days=args.train_days,
        validation_days=args.validation_days,
        test_days=args.test_days,
        step_days=args.step_days,
        gap_days=args.gap_days,
        thresholds=parse_thresholds(args.thresholds),
        min_validation_trades=args.min_validation_trades,
        default_threshold=args.default_threshold,
        calibration_bins=args.calibration_bins,
        update_latest=not args.no_update_latest,
    )
    write_next_day_artifacts(outdir, artifacts)
    latest = artifacts["latest"]
    assert isinstance(latest, pd.DataFrame)

    print("completed: next-day up prediction outputs =", outdir.resolve())
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()

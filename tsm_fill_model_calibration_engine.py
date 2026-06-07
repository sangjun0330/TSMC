#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate paper fill assumptions from execution feedback.

This module does not change config automatically. It writes suggested fill-model
parameters and keeps them in research-only status until enough matured paper
events exist for a defensible calibration review.

Outputs:
- tsm_fill_model_calibration.csv
- tsm_fill_model_calibration_quality_checks.csv
- tsm_fill_model_calibration_report.md
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tsm_core.config import PaperFeedbackConfig, load_run_config
from tsm_core.execution import FillModelCalibration
from tsm_core.io import as_float, check_row, strip_bom_columns, to_bool


CALIBRATION_COLUMNS = [
    "calibration_id",
    "fill_model",
    "event_count",
    "matured_event_count",
    "fill_rate",
    "limit_expiry_rate",
    "mean_realized_slippage_bps",
    "median_realized_slippage_bps",
    "p90_realized_slippage_bps",
    "mean_slippage_error_bps",
    "suggested_half_spread_bps",
    "suggested_gap_penalty_bps",
    "suggested_event_day_penalty_bps",
    "calibration_status",
    "live_trading_status",
    "calibrated_at_utc",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_id(*parts: object, prefix: str) -> str:
    text = "|".join("" if part is None else str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return strip_bom_columns(pd.read_csv(path))


def quantile(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(clean.quantile(q))


def build_calibration(events: pd.DataFrame, config: PaperFeedbackConfig) -> pd.DataFrame:
    if events.empty:
        row = FillModelCalibration(
            calibration_id=stable_id("missing", config.horizon_days, prefix="fmc"),
            fill_model="missing",
            event_count=0,
            matured_event_count=0,
            fill_rate=0.0,
            limit_expiry_rate=0.0,
            mean_realized_slippage_bps=np.nan,
            median_realized_slippage_bps=np.nan,
            p90_realized_slippage_bps=np.nan,
            mean_slippage_error_bps=np.nan,
            suggested_half_spread_bps=np.nan,
            suggested_gap_penalty_bps=np.nan,
            suggested_event_day_penalty_bps=np.nan,
            calibration_status="RESEARCH_ONLY_INSUFFICIENT_EVENTS",
            live_trading_status="DISABLED_BY_DESIGN",
            calibrated_at_utc=now_utc_iso(),
        )
        return pd.DataFrame([row.to_dict()], columns=CALIBRATION_COLUMNS)

    rows: list[dict[str, object]] = []
    events = events.copy()
    if "fill_model" not in events.columns:
        events["fill_model"] = "unknown"
    for fill_model, group in events.groupby(events["fill_model"].fillna("unknown").astype(str), dropna=False):
        filled = group.get("filled_flag", pd.Series(False, index=group.index)).map(to_bool)
        expired = group.get("expired_unfilled_flag", pd.Series(False, index=group.index)).map(to_bool)
        matured = group.get("label_matured", pd.Series(False, index=group.index)).map(to_bool)
        realized = pd.to_numeric(group.get("realized_slippage_bps", pd.Series(dtype=float)), errors="coerce")
        error = pd.to_numeric(group.get("slippage_error_bps", pd.Series(dtype=float)), errors="coerce")
        matured_count = int(matured.sum())
        event_count = int(len(group))
        mean_realized = as_float(realized.dropna().mean(), np.nan)
        median_realized = as_float(realized.dropna().median(), np.nan)
        p90_realized = quantile(realized, 0.90)
        mean_error = as_float(error.dropna().mean(), np.nan)
        positive_error = max(mean_error, 0.0) if pd.notna(mean_error) else np.nan
        status = "CALIBRATION_READY" if matured_count >= int(config.min_matured_events_for_calibration) else "RESEARCH_ONLY_INSUFFICIENT_EVENTS"
        suggested_half_spread = max(0.5, median_realized / 2.0) if pd.notna(median_realized) else np.nan
        suggested_gap_penalty = max(0.0, positive_error * 0.5) if pd.notna(positive_error) else np.nan
        suggested_event_penalty = max(0.0, positive_error * 0.25) if pd.notna(positive_error) else np.nan
        rows.append(
            FillModelCalibration(
                calibration_id=stable_id(fill_model, event_count, matured_count, prefix="fmc"),
                fill_model=str(fill_model),
                event_count=event_count,
                matured_event_count=matured_count,
                fill_rate=float(filled.mean()) if event_count else 0.0,
                limit_expiry_rate=float(expired.mean()) if event_count else 0.0,
                mean_realized_slippage_bps=mean_realized,
                median_realized_slippage_bps=median_realized,
                p90_realized_slippage_bps=p90_realized,
                mean_slippage_error_bps=mean_error,
                suggested_half_spread_bps=float(suggested_half_spread) if pd.notna(suggested_half_spread) else np.nan,
                suggested_gap_penalty_bps=float(suggested_gap_penalty) if pd.notna(suggested_gap_penalty) else np.nan,
                suggested_event_day_penalty_bps=float(suggested_event_penalty) if pd.notna(suggested_event_penalty) else np.nan,
                calibration_status=status,
                live_trading_status="DISABLED_BY_DESIGN",
                calibrated_at_utc=now_utc_iso(),
            ).to_dict()
        )
    return pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)


def build_quality(calibration: pd.DataFrame) -> pd.DataFrame:
    live_disabled = bool(calibration.get("live_trading_status", pd.Series(["DISABLED_BY_DESIGN"])).astype(str).eq("DISABLED_BY_DESIGN").all()) if not calibration.empty else True
    duplicate_ids = int(calibration.get("calibration_id", pd.Series(dtype=str)).duplicated().sum()) if not calibration.empty else 0
    ready_count = int(calibration.get("calibration_status", pd.Series(dtype=str)).astype(str).eq("CALIBRATION_READY").sum()) if not calibration.empty else 0
    return pd.DataFrame(
        [
            check_row("fill_model_calibration_contract", set(CALIBRATION_COLUMNS).issubset(calibration.columns), "CRITICAL", len(calibration.columns)),
            check_row("fill_model_calibration_rows_positive", not calibration.empty, "CRITICAL", len(calibration)),
            check_row("fill_model_calibration_ids_unique", duplicate_ids == 0, "CRITICAL", duplicate_ids),
            check_row("fill_model_calibration_live_trading_disabled", live_disabled, "CRITICAL", "DISABLED_BY_DESIGN"),
            check_row("fill_model_calibration_ready_count", ready_count >= 0, "INFO", ready_count),
        ]
    )


def write_report(outdir: Path, calibration: pd.DataFrame, quality: pd.DataFrame) -> None:
    latest = calibration.tail(1).to_dict(orient="records")[0] if not calibration.empty else {}
    lines = [
        "# Top10 Paper Fill Model Calibration Report",
        "",
        f"- Fill model: {latest.get('fill_model', 'NA')}",
        f"- Status: {latest.get('calibration_status', 'NA')}",
        f"- Events / matured: {latest.get('event_count', 0)} / {latest.get('matured_event_count', 0)}",
        f"- Suggested half spread bps: {latest.get('suggested_half_spread_bps', 'NA')}",
        f"- Live trading status: {latest.get('live_trading_status', 'DISABLED_BY_DESIGN')}",
        "",
        "## Quality Checks",
        "",
        "| Check | Passed | Value |",
        "|---|---:|---:|",
    ]
    for _, row in quality.iterrows():
        lines.append(f"| {row['check']} | {row['passed']} | {row['value']} |")
    lines.extend(["", "Suggested parameters require human review before config changes."])
    (outdir / "tsm_fill_model_calibration_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate broker-free paper fill model assumptions.")
    parser.add_argument("--feedback-events", default="tsm_price_rule_output/tsm_execution_feedback_events.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--config", default="config/tsm_research.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = load_run_config(args.config).paper_feedback
    calibration = build_calibration(read_csv_if_exists(Path(args.feedback_events)), config)
    quality = build_quality(calibration)
    calibration.to_csv(outdir / "tsm_fill_model_calibration.csv", index=False)
    quality.to_csv(outdir / "tsm_fill_model_calibration_quality_checks.csv", index=False)
    write_report(outdir, calibration, quality)
    print("completed: fill model calibration outputs =", outdir.resolve())
    print(calibration.to_string(index=False))


if __name__ == "__main__":
    main()

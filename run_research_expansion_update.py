#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Research expansion pipeline.

This script runs the normal daily system and then refreshes research-only
artifacts that need the expanded universe, event ledger, OOF predictions, CPCV,
DSR/PBO, and automatic trial ledger to exist together.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class StepResult:
    step: str
    status: str
    returncode: int
    duration_sec: float
    command: str


def run_step(step: str, command: List[str], cwd: Path, continue_on_error: bool) -> StepResult:
    start = time.time()
    print(f"\n=== {step} ===")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=str(cwd), text=True)
    duration = time.time() - start
    status = "OK" if completed.returncode == 0 else "FAIL"
    result = StepResult(step=step, status=status, returncode=completed.returncode, duration_sec=duration, command=" ".join(command))
    if completed.returncode != 0 and not continue_on_error:
        raise SystemExit(completed.returncode)
    return result


def write_manifest(outdir: Path, results: List[StepResult]) -> None:
    pd.DataFrame([asdict(r) for r in results]).to_csv(outdir / "tsm_research_expansion_manifest.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanded universe research update.")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument("--universe-config", default="config/semiconductor_universe_expanded.csv")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--rule-outdir", default="tsm_price_rule_output")
    parser.add_argument("--preferred-source", choices=["stooq", "yahoo"], default="stooq")
    parser.add_argument("--skip-data-refresh", action="store_true")
    parser.add_argument("--skip-news-refresh", action="store_true")
    parser.add_argument("--skip-external-web", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stop-multiple", type=float, default=2.0)
    parser.add_argument("--score-threshold", type=float, default=75.0)
    parser.add_argument("--enable-feedback-features", action="store_true")
    parser.add_argument("--enable-auto-research", action="store_true")
    parser.add_argument("--max-trials", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path(__file__).resolve().parent
    py = sys.executable
    rule_outdir = Path(args.rule_outdir)
    output_dir = Path(args.output_dir)
    rule_outdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_cmd = [
        py,
        "run_daily_update.py",
        "--start",
        str(args.start),
        "--end",
        str(args.end),
        "--preferred-source",
        str(args.preferred_source),
        "--output-dir",
        str(output_dir),
        "--rule-outdir",
        str(rule_outdir),
        "--universe-config",
        str(args.universe_config),
        "--commission-bps",
        str(args.commission_bps),
        "--slippage-bps",
        str(args.slippage_bps),
        "--stop-multiple",
        str(args.stop_multiple),
        "--score-threshold",
        str(args.score_threshold),
        "--prediction-policy",
        "research",
    ]
    if args.skip_data_refresh:
        daily_cmd.append("--skip-data-refresh")
    if args.skip_news_refresh:
        daily_cmd.append("--skip-news-refresh")
    if args.skip_external_web:
        daily_cmd.append("--skip-external-web")
    if args.continue_on_error:
        daily_cmd.append("--continue-on-error")
    if args.enable_feedback_features:
        daily_cmd.append("--enable-feedback-features")
    if args.enable_auto_research:
        daily_cmd.extend(["--enable-auto-research", "--max-trials", str(args.max_trials)])

    steps: List[tuple[str, List[str]]] = [
        ("daily_update_base_pipeline", daily_cmd),
        (
            "universe_backtest_event_ledger",
            [
                py,
                "tsm_backtest_event_ledger.py",
                "--universe-config",
                str(args.universe_config),
                "--outdir",
                str(rule_outdir),
                "--output-name",
                "tsm_backtest_event_ledger.csv",
                "--commission-bps",
                str(args.commission_bps),
                "--slippage-bps",
                str(args.slippage_bps),
            ],
        ),
    ]
    if args.enable_feedback_features:
        steps.append(
            (
                "feedback_feature_engine_universe",
                [
                    py,
                    "tsm_backtest_feedback_feature_engine.py",
                    "--ledger",
                    str(rule_outdir / "tsm_backtest_event_ledger.csv"),
                    "--oof-predictions",
                    str(rule_outdir / "tsm_pooled_model_oof_predictions.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            )
        )
    steps.extend(
        [
            (
                "post_pooled_cpcv_validation",
                [
                    py,
                    "tsm_validation_engine.py",
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--enriched",
                    str(output_dir / "tsm_daily_10y_enriched.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--baseline-stop-multiple",
                    str(args.stop_multiple),
                    "--baseline-score-threshold",
                    str(args.score_threshold),
                    "--commission-bps",
                    str(args.commission_bps),
                    "--slippage-bps",
                    str(args.slippage_bps),
                    "--model-trial-count",
                    "100",
                ],
            ),
        ]
    )
    if args.enable_auto_research:
        steps.append(
            (
                "auto_research_engine",
                [
                    py,
                    "tsm_auto_research_engine.py",
                    "--pooled-comparison",
                    str(rule_outdir / "tsm_pooled_model_comparison.csv"),
                    "--local-comparison",
                    str(rule_outdir / "tsm_prediction_model_comparison.csv"),
                    "--pbo-report",
                    str(rule_outdir / "tsm_pbo_report.csv"),
                    "--dsr-report",
                    str(rule_outdir / "tsm_deflated_sharpe_report.csv"),
                    "--cpcv-model-distribution",
                    str(rule_outdir / "tsm_cpcv_model_distribution.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--storage",
                    f"sqlite:///{rule_outdir / 'tsm_research_trials.sqlite'}",
                    "--max-trials",
                    str(args.max_trials),
                ],
            )
        )
    steps.extend(
        [
            (
                "model_gate_engine_research",
                [
                    py,
                    "tsm_model_gate_engine.py",
                    "--comparison",
                    str(rule_outdir / "tsm_prediction_model_comparison.csv"),
                    "--latest-prediction",
                    str(rule_outdir / "tsm_latest_prediction_snapshot.csv"),
                    "--pooled-comparison",
                    str(rule_outdir / "tsm_pooled_model_comparison.csv"),
                    "--pooled-latest",
                    str(rule_outdir / "tsm_pooled_latest_prediction_overlay.csv"),
                    "--tsm-calibration",
                    str(rule_outdir / "tsm_pooled_tsm_calibration_metrics.csv"),
                    "--cpcv-strategy-distribution",
                    str(rule_outdir / "tsm_cpcv_strategy_distribution.csv"),
                    "--cpcv-model-distribution",
                    str(rule_outdir / "tsm_cpcv_model_distribution.csv"),
                    "--pbo-report",
                    str(rule_outdir / "tsm_pbo_report.csv"),
                    "--cscv-pbo-report",
                    str(rule_outdir / "tsm_cscv_pbo_report.csv"),
                    "--dsr-report",
                    str(rule_outdir / "tsm_deflated_sharpe_report.csv"),
                    "--paper-gate",
                    str(rule_outdir / "tsm_paper_gate_snapshot.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
            (
                "daily_health_report_refresh",
                [
                    py,
                    "tsm_daily_health_report.py",
                    "--data-quality",
                    str(rule_outdir / "tsm_latest_data_quality_snapshot.csv"),
                    "--system-state",
                    str(rule_outdir / "tsm_latest_system_state.csv"),
                    "--prediction",
                    str(rule_outdir / "tsm_latest_prediction_snapshot.csv"),
                    "--model-gate",
                    str(rule_outdir / "tsm_model_gate_snapshot.csv"),
                    "--risk",
                    str(rule_outdir / "tsm_latest_risk_snapshot.csv"),
                    "--shadow-quality",
                    str(rule_outdir / "tsm_shadow_paper_quality_checks.csv"),
                    "--operational-quality",
                    str(rule_outdir / "tsm_operational_quality_checks.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
        ]
    )

    results: List[StepResult] = []
    for step, command in steps:
        results.append(run_step(step, command, cwd, args.continue_on_error))
        write_manifest(rule_outdir, results)
    print("\ncompleted: research expansion update =", rule_outdir.resolve())


if __name__ == "__main__":
    main()

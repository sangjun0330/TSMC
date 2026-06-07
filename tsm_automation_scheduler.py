#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the broker-free automation plan for daily Paper OMS operation.

This is intentionally a plan/checklist generator, not a daemon and not a live
order scheduler. It defines the repeatable paper workflow that run_daily_update
can materialize and that the dashboard can audit.

Outputs:
- tsm_automation_plan.csv
- tsm_automation_quality_checks.csv
- tsm_automation_report.md
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tsm_core.config import AutomationConfig, load_run_config
from tsm_core.io import check_row, strip_bom_columns


PLAN_COLUMNS = [
    "task_id",
    "task_name",
    "cadence",
    "local_time",
    "timezone",
    "run_mode",
    "required_inputs",
    "expected_outputs",
    "automation_status",
    "live_trading_status",
    "notes",
    "generated_at_utc",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_snapshot(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    df = strip_bom_columns(pd.read_csv(path))
    if not {"field", "value"}.issubset(df.columns):
        return {}
    return dict(zip(df["field"].astype(str), df["value"]))


def build_plan(config: AutomationConfig, lifecycle_snapshot: dict[str, object] | None = None) -> pd.DataFrame:
    lifecycle_snapshot = lifecycle_snapshot or {}
    lifecycle_state = str(lifecycle_snapshot.get("lifecycle_state", "UNKNOWN"))
    generated = now_utc_iso()
    tasks = [
        {
            "task_id": "post_close_data_refresh",
            "task_name": "Post-close data refresh",
            "cadence": "DAILY",
            "local_time": config.post_close_local_time,
            "run_mode": "downstream",
            "required_inputs": "daily vendor data|prior ledgers",
            "expected_outputs": "signals|risk|prediction|order intent",
            "notes": "Refresh data and rebuild research stack after the US close.",
        },
        {
            "task_id": "universe_daily_refresh",
            "task_name": "Universe daily refresh",
            "cadence": "DAILY",
            "local_time": config.post_close_local_time,
            "run_mode": "pooled_universe_update",
            "required_inputs": "universe config|daily vendor data",
            "expected_outputs": "output/universe/* daily files|per-symbol signals|pooled dataset",
            "notes": "Refresh each enabled universe member with the same daily data/rule/backtest/risk path.",
        },
        {
            "task_id": "universe_hourly_refresh",
            "task_name": "Universe hourly refresh",
            "cadence": "DAILY",
            "local_time": config.post_close_local_time,
            "run_mode": "universe_market_data_update",
            "required_inputs": "universe config|Yahoo chart hourly data",
            "expected_outputs": "output/universe/*/tsm_hourly_available_*.csv",
            "notes": "Refresh public hourly bars for every enabled universe member.",
        },
        {
            "task_id": "universe_minute_refresh",
            "task_name": "Universe minute refresh",
            "cadence": "DAILY",
            "local_time": config.post_close_local_time,
            "run_mode": "universe_market_data_update",
            "required_inputs": "universe config|Yahoo chart 1m data",
            "expected_outputs": "output/universe/*/tsm_minute_available_*.csv",
            "notes": "Refresh public 1-minute bars for every enabled universe member within source limits.",
        },
        {
            "task_id": "generate_order_intent",
            "task_name": "Generate order intent",
            "cadence": "DAILY",
            "local_time": config.post_close_local_time,
            "run_mode": "order_intent_engine",
            "required_inputs": "latest signals|risk snapshot|prediction snapshot|system state",
            "expected_outputs": "tsm_order_intents.csv",
            "notes": "Create broker-free proposed/approved order intent with live trading disabled.",
        },
        {
            "task_id": "portfolio_risk_gate",
            "task_name": "Portfolio risk gate",
            "cadence": "DAILY",
            "local_time": config.post_close_local_time,
            "run_mode": "portfolio_risk_engine",
            "required_inputs": "order intents|latest signals",
            "expected_outputs": "portfolio risk decisions|risk checks",
            "notes": "Cap paper target weight before any simulated order is submitted.",
        },
        {
            "task_id": "premarket_recheck",
            "task_name": "Premarket recheck",
            "cadence": "DAILY",
            "local_time": config.premarket_local_time,
            "run_mode": "data_quality_engine",
            "required_inputs": "latest daily outputs|market calendar",
            "expected_outputs": "freshness checks",
            "notes": "Confirm the daily inputs are still valid before simulated next-open fills.",
        },
        {
            "task_id": "paper_submit",
            "task_name": "Dry run / paper submit",
            "cadence": "DAILY",
            "local_time": config.premarket_local_time,
            "run_mode": "paper_execution_engine",
            "required_inputs": "approved intent|portfolio gate|next open bar",
            "expected_outputs": "paper orders|paper fills|paper positions|slippage report",
            "notes": "Simulate next-open paper execution. No broker call is allowed.",
        },
        {
            "task_id": "open_reconciliation",
            "task_name": "Open reconciliation",
            "cadence": "DAILY",
            "local_time": config.open_reconciliation_local_time,
            "run_mode": "position_reconciler",
            "required_inputs": "paper fills|paper positions",
            "expected_outputs": "paper reconciliation report",
            "notes": "Block the workflow if internal paper ledgers do not reconcile.",
        },
        {
            "task_id": "lifecycle_update",
            "task_name": "Lifecycle update",
            "cadence": "DAILY",
            "local_time": config.open_reconciliation_local_time,
            "run_mode": "order_state_machine",
            "required_inputs": "intents|risk decisions|orders|fills|reconciliation",
            "expected_outputs": "order lifecycle snapshot",
            "notes": f"Current lifecycle state is {lifecycle_state}.",
        },
        {
            "task_id": "eod_feedback_update",
            "task_name": "End-of-day feedback update",
            "cadence": "DAILY",
            "local_time": config.eod_feedback_local_time,
            "run_mode": "execution_feedback_engine",
            "required_inputs": "paper orders|paper fills|daily signals",
            "expected_outputs": "execution feedback features|execution feedback labels",
            "notes": "Mature only labels whose forward horizon is available.",
        },
        {
            "task_id": "fill_model_calibration",
            "task_name": "Fill model calibration",
            "cadence": "DAILY",
            "local_time": config.eod_feedback_local_time,
            "run_mode": "fill_model_calibration_engine",
            "required_inputs": "execution feedback events",
            "expected_outputs": "fill model calibration suggestions",
            "notes": "Write research-only suggested execution-cost parameters.",
        },
        {
            "task_id": "weekly_model_retrain_candidate",
            "task_name": "Weekly model retrain candidate",
            "cadence": f"WEEKLY_{config.weekly_review_day}",
            "local_time": config.eod_feedback_local_time,
            "run_mode": "prediction_engine",
            "required_inputs": "matured feedback labels|pooled dataset|feature contracts",
            "expected_outputs": "candidate model diagnostics",
            "notes": "Review feedback labels as candidates for future model training inputs.",
        },
        {
            "task_id": "monthly_promotion_review",
            "task_name": "Monthly promotion review",
            "cadence": f"MONTHLY_DAY_{config.monthly_promotion_review_day}",
            "local_time": config.eod_feedback_local_time,
            "run_mode": "model_gate_engine",
            "required_inputs": "calibration|paper slippage|model registry|paper feedback labels",
            "expected_outputs": "promotion/rejection decision evidence",
            "notes": "Promotion remains manual; live trading is disabled by design.",
        },
    ]
    rows = []
    for task in tasks:
        rows.append(
            {
                **task,
                "timezone": config.timezone,
                "automation_status": "ACTIVE_PLAN" if config.enabled else "PAUSED_PLAN",
                "live_trading_status": "DISABLED_BY_DESIGN",
                "generated_at_utc": generated,
            }
        )
    return pd.DataFrame(rows, columns=PLAN_COLUMNS)


def build_quality(plan: pd.DataFrame) -> pd.DataFrame:
    required = {
        "post_close_data_refresh",
        "universe_daily_refresh",
        "universe_hourly_refresh",
        "universe_minute_refresh",
        "generate_order_intent",
        "portfolio_risk_gate",
        "paper_submit",
        "open_reconciliation",
        "lifecycle_update",
        "eod_feedback_update",
        "fill_model_calibration",
    }
    task_ids = set(plan.get("task_id", pd.Series(dtype=str)).astype(str))
    live_disabled = bool(plan.get("live_trading_status", pd.Series(["DISABLED_BY_DESIGN"])).astype(str).eq("DISABLED_BY_DESIGN").all()) if not plan.empty else True
    no_live_modes = not plan.get("run_mode", pd.Series(dtype=str)).astype(str).str.contains("live", case=False, na=False).any() if not plan.empty else True
    return pd.DataFrame(
        [
            check_row("automation_plan_contract", set(PLAN_COLUMNS).issubset(plan.columns), "CRITICAL", len(plan.columns)),
            check_row("automation_required_tasks_present", required.issubset(task_ids), "CRITICAL", "|".join(sorted(required - task_ids)) or "PASS"),
            check_row("automation_no_live_submit_task", no_live_modes, "CRITICAL", "no live run modes"),
            check_row("automation_live_trading_disabled", live_disabled, "CRITICAL", "DISABLED_BY_DESIGN"),
        ]
    )


def write_report(outdir: Path, plan: pd.DataFrame, quality: pd.DataFrame) -> None:
    lines = [
        "# Top10 Paper OMS Automation Plan",
        "",
        f"- Task count: {len(plan)}",
        "- Live trading status: DISABLED_BY_DESIGN",
        "- Scope: broker-free daily paper workflow plan",
        "",
        "## Tasks",
        "",
        "| Task | Cadence | Time | Run Mode |",
        "|---|---|---:|---|",
    ]
    for _, row in plan.iterrows():
        lines.append(f"| {row['task_name']} | {row['cadence']} | {row['local_time']} {row['timezone']} | {row['run_mode']} |")
    lines.extend(["", "## Quality Checks", "", "| Check | Passed | Value |", "|---|---:|---:|"])
    for _, row in quality.iterrows():
        lines.append(f"| {row['check']} | {row['passed']} | {row['value']} |")
    lines.extend(["", "This plan does not schedule or submit live broker orders."])
    (outdir / "tsm_automation_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate broker-free Paper OMS automation plan.")
    parser.add_argument("--lifecycle-snapshot", default="tsm_price_rule_output/tsm_order_lifecycle_snapshot.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--config", default="config/tsm_research.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = load_run_config(args.config).automation
    plan = build_plan(config, read_snapshot(Path(args.lifecycle_snapshot)))
    quality = build_quality(plan)
    plan.to_csv(outdir / "tsm_automation_plan.csv", index=False)
    quality.to_csv(outdir / "tsm_automation_quality_checks.csv", index=False)
    write_report(outdir, plan, quality)
    print("completed: automation plan outputs =", outdir.resolve())
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate daily health report for the TSM research and paper-trading system.

This report is intentionally broker-free. It summarizes whether the current
daily run is suitable for research, paper/shadow tracking, and prediction
decision support.

Outputs:
- tsm_daily_health_snapshot.csv
- tsm_system_block_reasons.csv
- tsm_daily_health_report.md
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tsm_core.decision_schema import is_prediction_decision_support
from tsm_core.io import strip_bom_columns


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return strip_bom_columns(pd.read_csv(path))


def read_snapshot(path: Path) -> dict[str, object]:
    df = read_csv_if_exists(path)
    if df.empty or not {"field", "value"}.issubset(df.columns):
        return {}
    return dict(zip(df["field"].astype(str), df["value"]))


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def checks_critical_pass(checks: pd.DataFrame) -> tuple[bool, str]:
    if checks.empty or "passed" not in checks.columns:
        return False, "MISSING_CHECKS"
    target = checks[checks["severity"].astype(str).eq("CRITICAL")] if "severity" in checks.columns else checks
    if target.empty:
        target = checks
    passed = target["passed"].astype(str).str.lower().isin(["true", "1", "yes"]).all()
    failed = target[~target["passed"].astype(str).str.lower().isin(["true", "1", "yes"])]
    reasons = "|".join(failed["check"].astype(str)) if not failed.empty and "check" in failed.columns else "PASS"
    return bool(passed), reasons


def component_row(component: str, status: str, block_reasons: str, details: str = "") -> dict[str, object]:
    return {"component": component, "status": status, "block_reasons": block_reasons or "PASS", "details": details}


def derive_component_rows(
    data_quality: dict[str, object],
    system_state: dict[str, object],
    prediction: dict[str, object],
    model_gate: dict[str, object],
    risk: dict[str, object],
    shadow_quality: pd.DataFrame,
    operational_quality: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    data_status = str(data_quality.get("data_quality_status", "MISSING"))
    data_gate = str(data_quality.get("decision_support_data_gate", "UNKNOWN"))
    rows.append(
        component_row(
            "data_quality",
            "FAIL" if data_status in {"FAIL", "MISSING"} else ("WARN" if data_status == "WARN" else "PASS"),
            str(data_quality.get("block_reasons", "MISSING_DATA_QUALITY")),
            f"gate={data_gate}, latest_signal_date={data_quality.get('latest_signal_date', 'NA')}",
        )
    )

    operational_ok, operational_reasons = checks_critical_pass(operational_quality)
    rows.append(component_row("operational_quality", "PASS" if operational_ok else "FAIL", operational_reasons))

    research_ready = to_bool(system_state.get("research_ready", False))
    rows.append(
        component_row(
            "research_readiness",
            "PASS" if research_ready else "FAIL",
            str(system_state.get("research_block_reasons", "MISSING_SYSTEM_STATE")),
            f"score={system_state.get('research_readiness_score', system_state.get('system_readiness_score', 'NA'))}",
        )
    )

    prediction_ready = is_prediction_decision_support(prediction.get("prediction_use_status", "UNKNOWN"))
    rows.append(
        component_row(
            "prediction_decision_support",
            "PASS" if prediction_ready else "BLOCKED",
            str(system_state.get("prediction_block_reasons", prediction.get("model_quality_block_reasons", "PREDICTION_NOT_DECISION_SUPPORT"))),
            f"use_status={prediction.get('prediction_use_status', 'NA')}, signal_status={prediction.get('prediction_signal_status', 'NA')}",
        )
    )

    gate_status = str(model_gate.get("model_gate_status", "MISSING"))
    if gate_status == "PASS":
        model_gate_status = "PASS"
    elif gate_status in {"BLOCKED", "WARN"}:
        model_gate_status = "BLOCKED"
    else:
        model_gate_status = "FAIL"
    rows.append(
        component_row(
            "model_gate",
            model_gate_status,
            str(model_gate.get("failed_gate_groups", "MISSING_MODEL_GATE")),
            f"status={gate_status}, failed_gates={model_gate.get('failed_gate_count', 'NA')}",
        )
    )

    risk_state = str(risk.get("risk_state", "UNKNOWN"))
    risk_ok = risk_state not in {"UNKNOWN", "NO_NEW_RISK"}
    rows.append(
        component_row(
            "risk_policy",
            "PASS" if risk_ok else "FAIL",
            "PASS" if risk_ok else risk_state,
            f"risk_state={risk_state}, max_weight={risk.get('final_recommended_max_weight', 'NA')}",
        )
    )

    paper_ready = to_bool(system_state.get("paper_ready", False))
    shadow_ok, shadow_reasons = checks_critical_pass(shadow_quality)
    paper_status = "PASS" if paper_ready and shadow_ok else "FAIL"
    rows.append(
        component_row(
            "paper_shadow",
            paper_status,
            "PASS" if paper_status == "PASS" else f"{system_state.get('paper_block_reasons', 'PAPER_NOT_READY')}|{shadow_reasons}",
            f"paper_status={system_state.get('paper_trading_status', 'NA')}",
        )
    )

    rows.append(
        component_row(
            "live_trading",
            "NOT_APPLICABLE",
            str(system_state.get("live_block_reasons", "NO_LIVE_BROKER_BY_DESIGN")),
            "Broker execution is excluded by design.",
        )
    )
    return pd.DataFrame(rows)


def derive_overall_status(blocks: pd.DataFrame) -> str:
    if blocks.empty:
        return "FAIL"
    hard_components = blocks[blocks["component"].isin(["data_quality", "operational_quality", "research_readiness", "risk_policy", "paper_shadow"])]
    if hard_components["status"].isin(["FAIL"]).any():
        return "FAIL"
    if blocks["status"].isin(["BLOCKED", "WARN"]).any():
        return "WARN"
    return "PASS"


def build_snapshot(blocks: pd.DataFrame, system_state: dict[str, object], data_quality: dict[str, object], prediction: dict[str, object]) -> pd.DataFrame:
    status = derive_overall_status(blocks)
    failed_or_blocked = blocks[blocks["status"].isin(["FAIL", "BLOCKED", "WARN"])]
    rows = [
        {"field": "daily_health_status", "value": status},
        {"field": "system_state", "value": system_state.get("system_state", "UNKNOWN")},
        {"field": "data_quality_status", "value": data_quality.get("data_quality_status", "UNKNOWN")},
        {"field": "research_ready", "value": system_state.get("research_ready", False)},
        {"field": "paper_ready", "value": system_state.get("paper_ready", False)},
        {"field": "prediction_decision_support", "value": is_prediction_decision_support(prediction.get("prediction_use_status", "UNKNOWN"))},
        {"field": "prediction_use_status", "value": prediction.get("prediction_use_status", "UNKNOWN")},
        {"field": "failed_or_blocked_components", "value": "|".join(failed_or_blocked["component"].astype(str)) if not failed_or_blocked.empty else "PASS"},
        {"field": "live_trading_status", "value": system_state.get("live_trading_status", "DISABLED_BY_DESIGN")},
        {"field": "generated_at_utc", "value": now_utc_iso()},
    ]
    return pd.DataFrame(rows)


def write_report(outdir: Path, snapshot: pd.DataFrame, blocks: pd.DataFrame) -> None:
    snap = dict(zip(snapshot["field"], snapshot["value"]))
    lines = [
        "# TSMC Daily Health Report",
        "",
        f"- Daily health status: {snap.get('daily_health_status', 'NA')}",
        f"- System state: {snap.get('system_state', 'NA')}",
        f"- Data quality status: {snap.get('data_quality_status', 'NA')}",
        f"- Research ready: {snap.get('research_ready', 'NA')}",
        f"- Paper ready: {snap.get('paper_ready', 'NA')}",
        f"- Prediction decision support: {snap.get('prediction_decision_support', 'NA')}",
        f"- Prediction use status: {snap.get('prediction_use_status', 'NA')}",
        f"- Live trading status: {snap.get('live_trading_status', 'NA')}",
        "",
        "## Components",
        "",
        "| Component | Status | Block Reasons | Details |",
        "|---|---|---|---|",
    ]
    for _, row in blocks.iterrows():
        lines.append(f"| {row['component']} | {row['status']} | {row['block_reasons']} | {row['details']} |")
    lines.extend(["", "This report is research tooling, not investment advice."])
    (outdir / "tsm_daily_health_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily health report for TSM research outputs.")
    parser.add_argument("--data-quality", default="tsm_price_rule_output/tsm_latest_data_quality_snapshot.csv")
    parser.add_argument("--system-state", default="tsm_price_rule_output/tsm_latest_system_state.csv")
    parser.add_argument("--prediction", default="tsm_price_rule_output/tsm_latest_prediction_snapshot.csv")
    parser.add_argument("--model-gate", default="tsm_price_rule_output/tsm_model_gate_snapshot.csv")
    parser.add_argument("--risk", default="tsm_price_rule_output/tsm_latest_risk_snapshot.csv")
    parser.add_argument("--shadow-quality", default="tsm_price_rule_output/tsm_shadow_paper_quality_checks.csv")
    parser.add_argument("--operational-quality", default="tsm_price_rule_output/tsm_operational_quality_checks.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_quality = read_snapshot(Path(args.data_quality))
    system_state = read_snapshot(Path(args.system_state))
    prediction = read_snapshot(Path(args.prediction))
    model_gate = read_snapshot(Path(args.model_gate))
    risk = read_snapshot(Path(args.risk))
    shadow_quality = read_csv_if_exists(Path(args.shadow_quality))
    operational_quality = read_csv_if_exists(Path(args.operational_quality))

    blocks = derive_component_rows(data_quality, system_state, prediction, model_gate, risk, shadow_quality, operational_quality)
    snapshot = build_snapshot(blocks, system_state, data_quality, prediction)
    snapshot.to_csv(outdir / "tsm_daily_health_snapshot.csv", index=False)
    blocks.to_csv(outdir / "tsm_system_block_reasons.csv", index=False)
    write_report(outdir, snapshot, blocks)
    print("completed: daily health outputs =", outdir.resolve())
    print(snapshot.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a final audit for the full daily automation wrapper.

The daily wrapper runs more than run_daily_update.py: pooled universe, daily
system, hourly bars, and minute bars. This audit is the final cross-system
inventory so the scheduled automation can report one source of truth.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DATE_COLUMNS = [
    "date",
    "asof_date",
    "expected_fill_date",
    "created_at_utc",
    "generated_at_utc",
    "ended_at_utc",
    "timestamp",
]

OPTIONAL_CSV_READ_LIMIT_BYTES = 64 * 1024 * 1024
OPTIONAL_CSV_PARSE_LABELS = {"tsm_model_gate_audit"}


REQUIRED_FILES = [
    ("daily_raw", "output/tsm_daily_10y_raw.csv", "data"),
    ("daily_enriched", "output/tsm_daily_10y_enriched.csv", "data"),
    ("daily_summary", "output/tsm_daily_10y_summary.csv", "data"),
    ("hourly_enriched", "output/tsm_hourly_available_enriched.csv", "data"),
    ("hourly_summary", "output/tsm_hourly_available_summary.csv", "data"),
    ("hourly_source_audit", "output/tsm_hourly_10y_source_audit.csv", "data"),
    ("minute_enriched", "output/tsm_minute_available_enriched.csv", "data"),
    ("minute_summary", "output/tsm_minute_available_summary.csv", "data"),
    ("minute_source_audit", "output/tsm_minute_10y_source_audit.csv", "data"),
    ("universe_market_data_latest", "output/tsm_universe_market_data_latest.csv", "data"),
    ("universe_intraday_manifest", "output/tsm_universe_intraday_update_manifest.csv", "data"),
    ("news_integrated_daily", "tsm_price_rule_output/tsm_news_integrated_daily.csv", "news"),
    ("pooled_universe_manifest", "tsm_price_rule_output/tsm_pooled_universe_update_manifest.csv", "pooled"),
    ("daily_update_manifest", "tsm_price_rule_output/tsm_daily_update_manifest.csv", "pipeline"),
    ("operational_quality", "tsm_price_rule_output/tsm_operational_quality_checks.csv", "quality"),
    ("daily_integrity", "tsm_price_rule_output/tsm_daily_integrity_checks.csv", "quality"),
    ("validation_quality", "tsm_price_rule_output/tsm_validation_quality_checks.csv", "quality"),
    ("prediction_quality", "tsm_price_rule_output/tsm_prediction_quality_checks.csv", "quality"),
    ("model_gate_snapshot", "tsm_price_rule_output/tsm_model_gate_snapshot.csv", "system"),
    ("tsm_model_performance_gate_audit", "tsm_price_rule_output/tsm_model_performance_gate_audit.csv", "system"),
    ("latest_system_state", "tsm_price_rule_output/tsm_latest_system_state.csv", "system"),
    ("daily_health_snapshot", "tsm_price_rule_output/tsm_daily_health_snapshot.csv", "system"),
    ("latest_decision_snapshot", "tsm_price_rule_output/tsm_latest_decision_snapshot.csv", "decision"),
    ("latest_risk_snapshot", "tsm_price_rule_output/tsm_latest_risk_snapshot.csv", "decision"),
    ("daily_trading_plan_csv", "tsm_price_rule_output/tsm_daily_trading_plan.csv", "decision"),
    ("daily_trading_plan_md", "tsm_price_rule_output/tsm_daily_trading_plan.md", "decision"),
    ("backtest_trade_log", "tsm_price_rule_output/tsm_backtest_trade_log.csv", "backtest"),
    ("shadow_paper_predictions", "tsm_price_rule_output/tsm_shadow_paper_predictions.csv", "paper"),
    ("order_intents", "tsm_price_rule_output/tsm_order_intents.csv", "paper"),
    ("portfolio_risk_decisions", "tsm_price_rule_output/tsm_portfolio_risk_order_decisions.csv", "paper"),
    ("paper_orders", "tsm_price_rule_output/tsm_paper_orders.csv", "paper"),
    ("paper_fills", "tsm_price_rule_output/tsm_paper_fills.csv", "paper"),
    ("paper_positions", "tsm_price_rule_output/tsm_paper_positions.csv", "paper"),
    ("paper_reconciliation", "tsm_price_rule_output/tsm_paper_reconciliation_report.csv", "paper"),
    ("order_lifecycle_snapshot", "tsm_price_rule_output/tsm_order_lifecycle_snapshot.csv", "paper"),
    ("execution_feedback_events", "tsm_price_rule_output/tsm_execution_feedback_events.csv", "paper"),
    ("fill_model_calibration", "tsm_price_rule_output/tsm_fill_model_calibration.csv", "paper"),
    ("automation_plan", "tsm_price_rule_output/tsm_automation_plan.csv", "automation"),
    ("automation_quality", "tsm_price_rule_output/tsm_automation_quality_checks.csv", "automation"),
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "pass", "ok"}


def read_csv(path: Path) -> pd.DataFrame:
    return strip_bom_columns(pd.read_csv(path, low_memory=False))


def snapshot_map(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        df = read_csv(path)
    except Exception:
        return {}
    if {"field", "value"}.issubset(df.columns):
        return dict(zip(df["field"].astype(str), df["value"]))
    return {}


def first_value(*values: Any, default: str = "NA") -> str:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "na", "n/a"}:
            return text
    return default


def latest_date_from_frame(df: pd.DataFrame, snap: dict[str, Any] | None = None) -> str:
    snap = snap or {}
    for key in ["date", "asof_date", "latest_signal_date", "generated_at_utc", "ended_at_utc"]:
        if key in snap:
            parsed = pd.to_datetime(snap[key], errors="coerce")
            if pd.notna(parsed):
                return parsed.isoformat()
    for col in DATE_COLUMNS:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            latest = parsed.max()
            if pd.notna(latest):
                return latest.isoformat()
    return ""


def quality_pass(df: pd.DataFrame, *, critical_only: bool = False) -> tuple[str, str]:
    if df.empty:
        return "", ""
    if "passed" in df.columns:
        target = df
        if "severity" in df.columns:
            critical = df[df["severity"].astype(str).str.upper().eq("CRITICAL")]
            if not critical.empty or critical_only:
                target = critical
        ok = target["passed"].map(as_bool).all()
        failed = target.loc[~target["passed"].map(as_bool), "check"].astype(str).tolist() if "check" in target.columns else []
        return ("PASS" if ok else "FAIL", "|".join(failed[:20]))
    if "status" in df.columns:
        statuses = df["status"].dropna().astype(str).str.upper()
        if statuses.empty:
            return "", ""
        ok = statuses.isin(["PASS", "OK", "SUCCESS", "SKIPPED"]).all()
        failed = statuses.loc[~statuses.isin(["PASS", "OK", "SUCCESS", "SKIPPED"])].astype(str).tolist()
        return ("PASS" if ok else "FAIL", "|".join(failed[:20]))
    return "", ""


def performance_gate_quality_pass(df: pd.DataFrame) -> tuple[str, str]:
    if df.empty:
        return "PASS", "active_performance_failures=0;diagnostic_rows=0"
    if "active_performance_gate" in df.columns:
        active = df[df["active_performance_gate"].map(as_bool)].copy()
    elif "performance_gate_scope" in df.columns:
        active = df[df["performance_gate_scope"].astype(str).str.lower().eq("active")].copy()
    else:
        return "FAIL", "missing_active_performance_gate"
    diagnostic_count = len(df) - len(active)
    if active.empty:
        return "PASS", f"active_performance_failures=0;diagnostic_rows={diagnostic_count}"
    if "passed" not in active.columns:
        return "FAIL", "active_performance_rows_missing_passed"
    failed = active[~active["passed"].map(as_bool)]
    if failed.empty:
        return "PASS", f"active_performance_failures=0;diagnostic_rows={diagnostic_count}"
    detail_cols = [c for c in ["gate_group", "gate", "block_reason"] if c in failed.columns]
    if detail_cols:
        details = "|".join(failed.head(20)[detail_cols].fillna("").astype(str).agg(":".join, axis=1).tolist())
    else:
        details = f"active_performance_failures={len(failed)}"
    return "FAIL", details


def metadata_row(root: Path, label: str, rel_path: str, category: str, required: bool) -> dict[str, Any]:
    path = (root / rel_path).resolve()
    generated = now_utc_iso()
    row: dict[str, Any] = {
        "component": label,
        "category": category,
        "path": str(path),
        "required": required,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "rows": "",
        "columns": "",
        "latest_date": "",
        "status": "MISSING" if required else "OPTIONAL_MISSING",
        "passed": False if required else True,
        "details": "",
        "generated_at_utc": generated,
    }
    if not path.exists():
        return row
    row["status"] = "PRESENT"
    row["passed"] = True
    if path.suffix.lower() != ".csv":
        return row
    if not required and label not in OPTIONAL_CSV_PARSE_LABELS:
        row["latest_date"] = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        if row["size_bytes"] > OPTIONAL_CSV_READ_LIMIT_BYTES:
            row["details"] = f"csv_read_skipped_size_bytes_gt_{OPTIONAL_CSV_READ_LIMIT_BYTES}"
        else:
            row["details"] = "csv_read_skipped_optional_inventory"
        return row
    if not required and row["size_bytes"] > OPTIONAL_CSV_READ_LIMIT_BYTES:
        row["latest_date"] = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        row["details"] = f"csv_read_skipped_size_bytes_gt_{OPTIONAL_CSV_READ_LIMIT_BYTES}"
        return row
    try:
        df = read_csv(path)
    except Exception as exc:
        row["status"] = "READ_ERROR"
        row["passed"] = False
        row["details"] = f"{type(exc).__name__}: {exc}"
        return row
    snap = dict(zip(df["field"].astype(str), df["value"])) if {"field", "value"}.issubset(df.columns) else {}
    status, details = ("", "")
    if label == "tsm_model_performance_gate_audit":
        status, details = performance_gate_quality_pass(df)
    elif "passed" in df.columns or "manifest" in label:
        status, details = quality_pass(df, critical_only=label == "tsm_model_gate_audit")
    latest_date = latest_date_from_frame(df, snap)
    if not latest_date:
        latest_date = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    row.update(
        {
            "rows": len(df),
            "columns": len(df.columns),
            "latest_date": latest_date,
            "status": status or "PRESENT",
            "passed": (status != "FAIL") if status else True,
            "details": details,
        }
    )
    return row


def inventory_paths(root: Path, output_dir: Path, rule_outdir: Path) -> list[tuple[str, str, str, bool]]:
    known = {(Path(rel).as_posix()) for _, rel, _ in REQUIRED_FILES}
    rows = [(label, rel, category, True) for label, rel, category in REQUIRED_FILES]
    for base, category in [(output_dir, "inventory"), (rule_outdir, "inventory")]:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith("._"):
                continue
            if "/logs/" in path.as_posix() or ".daily_update.lock" in path.as_posix():
                continue
            if path.suffix.lower() not in {".csv", ".md", ".png", ".sqlite"}:
                continue
            rel = path.resolve().relative_to(root.resolve()).as_posix()
            if rel in known:
                continue
            rows.append((path.stem, rel, category, False))
    return rows


def freshness_rows(audit: pd.DataFrame, run_date: str, generated_at: str) -> list[dict[str, Any]]:
    rows = []
    run_ts = pd.Timestamp(run_date).normalize()
    limits = {
        "daily_enriched": 7,
        "hourly_enriched": 14,
        "minute_enriched": 14,
        "universe_market_data_latest": 14,
        "news_integrated_daily": 7,
        "latest_decision_snapshot": 7,
        "latest_system_state": 7,
    }
    by_component = audit.set_index("component", drop=False)
    for component, max_days in limits.items():
        if component not in by_component.index:
            continue
        row = by_component.loc[component]
        latest = pd.to_datetime(row.get("latest_date"), errors="coerce")
        if pd.isna(latest) and row.get("exists", False):
            try:
                latest = pd.Timestamp(datetime.fromtimestamp(Path(str(row.get("path"))).stat().st_mtime, tz=timezone.utc))
            except Exception:
                latest = pd.NaT
        if pd.notna(latest) and getattr(latest, "tzinfo", None) is not None:
            latest = latest.tz_convert(None)
        age = (run_ts - latest.normalize()).days if pd.notna(latest) else 9999
        passed = bool(pd.notna(latest) and latest.normalize() <= run_ts and age <= max_days)
        rows.append(
            {
                "component": f"{component}_freshness",
                "category": "freshness",
                "path": row.get("path", ""),
                "required": True,
                "exists": row.get("exists", False),
                "size_bytes": row.get("size_bytes", 0),
                "rows": row.get("rows", ""),
                "columns": row.get("columns", ""),
                "latest_date": latest.isoformat() if pd.notna(latest) else "",
                "status": "PASS" if passed else "FAIL",
                "passed": passed,
                "details": f"age_days={age};max_days={max_days};run_date={run_date}",
                "generated_at_utc": generated_at,
            }
        )
    return rows


def latest_log_path(log_file: str, rule_outdir: Path) -> str:
    if log_file:
        return str(Path(log_file).resolve())
    logs = sorted((rule_outdir / "logs").glob("daily_update_*.log"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return str(logs[-1].resolve()) if logs else ""


def status_map_value(path: Path, *keys: str) -> str:
    snap = snapshot_map(path)
    return first_value(*(snap.get(key) for key in keys))


def write_report(root: Path, rule_outdir: Path, audit: pd.DataFrame, run_date: str, log_file: str) -> None:
    decision = snapshot_map(root / "tsm_price_rule_output/tsm_latest_decision_snapshot.csv")
    risk = snapshot_map(root / "tsm_price_rule_output/tsm_latest_risk_snapshot.csv")
    system = snapshot_map(root / "tsm_price_rule_output/tsm_latest_system_state.csv")
    health = snapshot_map(root / "tsm_price_rule_output/tsm_daily_health_snapshot.csv")
    model_gate = snapshot_map(root / "tsm_price_rule_output/tsm_model_gate_snapshot.csv")
    required = audit[audit["required"].map(as_bool)]
    failed = required[~required["passed"].map(as_bool)]
    latest_log = latest_log_path(log_file, rule_outdir)

    summary_rows = [
        ("Daily data", "daily_enriched"),
        ("Hourly data", "hourly_enriched"),
        ("Minute data", "minute_enriched"),
        ("News", "news_integrated_daily"),
        ("System", "latest_system_state"),
        ("Health", "daily_health_snapshot"),
    ]
    by_component = audit.set_index("component", drop=False)
    lines = [
        "# Top10 Full Daily Update Audit",
        "",
        f"- Run date: {run_date}",
        f"- Overall status: {'PASS' if failed.empty else 'FAIL'}",
        f"- Required checks: {len(required) - len(failed)} / {len(required)} passed",
        f"- Generated at UTC: {now_utc_iso()}",
        f"- Latest log: {latest_log or 'NA'}",
        "",
        "## Latest Data",
        "",
        "| Scope | Latest Date | Rows | Status |",
        "|---|---:|---:|---|",
    ]
    for label, component in summary_rows:
        row = by_component.loc[component] if component in by_component.index else {}
        lines.append(
            f"| {label} | {first_value(row.get('latest_date') if isinstance(row, pd.Series) else '')} | "
            f"{first_value(row.get('rows') if isinstance(row, pd.Series) else '')} | "
            f"{first_value(row.get('status') if isinstance(row, pd.Series) else '')} |"
        )
    lines.extend(
        [
            "",
            "## System Gates",
            "",
            f"- System state: {first_value(system.get('system_state'))}",
            f"- Daily health: {first_value(health.get('daily_health_status'))}",
            f"- Model gate: {first_value(model_gate.get('model_gate_status'))}",
            f"- Research ready: {first_value(system.get('research_ready'))}",
            f"- Paper ready: {first_value(system.get('paper_ready'))}",
            f"- Live trading: {first_value(system.get('live_trading_status'), default='DISABLED_BY_DESIGN')}",
            f"- Block reasons: {first_value(system.get('paper_block_reasons'), system.get('prediction_block_reasons'), system.get('live_block_reasons'))}",
            "",
            "## Latest Decision",
            "",
            f"- Date: {first_value(decision.get('date'))}",
            f"- Trade action: {first_value(decision.get('trade_action'), decision.get('strict_signal_stage'))}",
            f"- Entry trigger: {first_value(decision.get('entry_trigger'))}",
            f"- Recommended max weight: {first_value(risk.get('final_recommended_max_weight'), risk.get('final_recommended_max_weight_pct'))}%",
            f"- Risk state: {first_value(risk.get('risk_state'))}",
            f"- Stop price: {first_value(decision.get('atr_stop_2x'), decision.get('stop_price'))}",
            f"- Target price: {first_value(decision.get('take_profit_2R'), decision.get('take_profit_3R'))}",
            f"- Paper trading: {first_value(system.get('paper_trading_status'))}",
            f"- Shadow/paper gate: {first_value(system.get('paper_gate_status'))}",
            "",
            "## Required Failures",
            "",
        ]
    )
    if failed.empty:
        lines.append("No required audit failures.")
    else:
        lines.extend(["| Component | Status | Details |", "|---|---|---|"])
        for _, row in failed.head(40).iterrows():
            lines.append(f"| {row['component']} | {row['status']} | {row['details']} |")
    lines.extend(["", "Broker/live order submission remains disabled by design."])
    (rule_outdir / "tsm_full_daily_update_audit.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final full daily update audit.")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--rule-outdir", default="tsm_price_rule_output")
    parser.add_argument("--run-date", default=date.today().isoformat())
    parser.add_argument("--log-file", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    output_dir = (root / args.output_dir).resolve()
    rule_outdir = (root / args.rule_outdir).resolve()
    generated = now_utc_iso()
    rows = [
        metadata_row(root, label, rel, category, required)
        for label, rel, category, required in inventory_paths(root, output_dir, rule_outdir)
    ]
    audit = pd.DataFrame(rows)
    audit = pd.concat([audit, pd.DataFrame(freshness_rows(audit, args.run_date, generated))], ignore_index=True)
    audit.to_csv(rule_outdir / "tsm_full_daily_update_audit.csv", index=False)
    write_report(root, rule_outdir, audit, args.run_date, args.log_file)
    required = audit[audit["required"].map(as_bool)]
    failed = required[~required["passed"].map(as_bool)]
    print("completed: full daily update audit =", (rule_outdir / "tsm_full_daily_update_audit.csv").resolve())
    print(f"required_passed={len(required) - len(failed)}/{len(required)}")
    if not failed.empty:
        print(failed[["component", "status", "details"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

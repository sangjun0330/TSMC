#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconcile broker-free paper positions against paper fills.

There is no external broker in this phase. The reconciler verifies that the
internal paper position ledger matches the fills ledger and blocks live trading
by design.

Outputs:
- tsm_paper_reconciliation_report.csv
- tsm_paper_reconciliation_quality_checks.csv
- tsm_paper_reconciliation_report.md
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tsm_core.execution import PaperReconciliation
from tsm_core.io import as_float, check_row, strip_bom_columns


REPORT_COLUMNS = [
    "symbol",
    "asof_date",
    "internal_quantity",
    "recomputed_quantity",
    "quantity_diff",
    "internal_market_value",
    "recomputed_market_value",
    "market_value_diff",
    "mismatch_count",
    "status",
    "details",
    "checked_at_utc",
    "live_trading_status",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return strip_bom_columns(pd.read_csv(path))


def recompute_from_fills(fills: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return pd.DataFrame(columns=["symbol", "recomputed_quantity", "recomputed_market_value"])
    rows = []
    for symbol, group in fills.groupby("symbol", dropna=False):
        buys = group[group["side"].astype(str).str.upper().eq("BUY")]
        sells = group[group["side"].astype(str).str.upper().eq("SELL")]
        qty = pd.to_numeric(buys.get("quantity", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() - pd.to_numeric(sells.get("quantity", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        latest_price = as_float(group.sort_values("fill_date").tail(1).iloc[0].get("fill_price")) if not group.empty else np.nan
        rows.append({"symbol": str(symbol), "recomputed_quantity": max(float(qty), 0.0), "recomputed_market_value": max(float(qty), 0.0) * latest_price if pd.notna(latest_price) else 0.0})
    return pd.DataFrame(rows)


def build_reconciliation(positions: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        positions = pd.DataFrame([{"symbol": "CASH", "asof_date": "", "quantity": 0.0, "market_value": 0.0}])
    recomputed = recompute_from_fills(fills)
    rows = []
    for _, pos in positions.iterrows():
        symbol = str(pos.get("symbol", "CASH"))
        rec = recomputed[recomputed["symbol"].astype(str).eq(symbol)]
        rec_row = rec.iloc[0] if not rec.empty else pd.Series({"recomputed_quantity": 0.0, "recomputed_market_value": 0.0})
        internal_qty = as_float(pos.get("quantity"), 0.0)
        recomputed_qty = as_float(rec_row.get("recomputed_quantity"), 0.0)
        internal_mv = as_float(pos.get("market_value"), 0.0)
        market_price = as_float(pos.get("market_price"), np.nan)
        recomputed_mv = recomputed_qty * market_price if pd.notna(market_price) else as_float(rec_row.get("recomputed_market_value"), 0.0)
        qty_diff = internal_qty - recomputed_qty
        mv_diff = internal_mv - recomputed_mv
        mismatch_count = int(abs(qty_diff) > 1e-8) + int(abs(mv_diff) > max(1.0, abs(internal_mv) * 0.01))
        status = "PASS" if mismatch_count == 0 else "FAIL"
        rows.append(
            PaperReconciliation(
                symbol=symbol,
                asof_date=str(pos.get("asof_date", "")),
                internal_quantity=float(internal_qty),
                recomputed_quantity=float(recomputed_qty),
                quantity_diff=float(qty_diff),
                internal_market_value=float(internal_mv),
                recomputed_market_value=float(recomputed_mv),
                market_value_diff=float(mv_diff),
                mismatch_count=mismatch_count,
                status=status,
                details="PASS" if status == "PASS" else "POSITION_MISMATCH",
                checked_at_utc=now_utc_iso(),
            ).to_dict()
        )
    out = pd.DataFrame(rows)
    out["live_trading_status"] = "DISABLED_BY_DESIGN"
    return out[REPORT_COLUMNS].copy()


def build_quality(report: pd.DataFrame) -> pd.DataFrame:
    mismatch_count = int(pd.to_numeric(report.get("mismatch_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not report.empty else 0
    rows = [
        check_row("paper_reconciliation_rows_positive", not report.empty, "CRITICAL", len(report)),
        check_row("paper_reconciliation_no_position_mismatch", mismatch_count == 0, "CRITICAL", mismatch_count),
        check_row("paper_reconciliation_no_live_broker_required", True, "CRITICAL", "paper_only"),
        check_row("paper_reconciliation_live_trading_disabled", bool(report["live_trading_status"].astype(str).eq("DISABLED_BY_DESIGN").all()) if not report.empty else True, "CRITICAL", "DISABLED_BY_DESIGN"),
    ]
    return pd.DataFrame(rows)


def write_report_md(outdir: Path, report: pd.DataFrame, quality: pd.DataFrame) -> None:
    mismatch_count = int(pd.to_numeric(report.get("mismatch_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not report.empty else 0
    lines = [
        "# Top10 Paper Reconciliation Report",
        "",
        f"- Status: {'PASS' if mismatch_count == 0 else 'FAIL'}",
        f"- Mismatch count: {mismatch_count}",
        "- Live trading status: DISABLED_BY_DESIGN",
        "",
        "## Quality Checks",
        "",
        "| Check | Passed | Value |",
        "|---|---:|---:|",
    ]
    for _, row in quality.iterrows():
        lines.append(f"| {row['check']} | {row['passed']} | {row['value']} |")
    lines.extend(["", "This report reconciles internal paper ledgers only."])
    (outdir / "tsm_paper_reconciliation_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile broker-free paper positions.")
    parser.add_argument("--positions", default="tsm_price_rule_output/tsm_paper_positions.csv")
    parser.add_argument("--fills", default="tsm_price_rule_output/tsm_paper_fills.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    positions = read_csv_if_exists(Path(args.positions))
    fills = read_csv_if_exists(Path(args.fills))
    report = build_reconciliation(positions, fills)
    quality = build_quality(report)
    report.to_csv(outdir / "tsm_paper_reconciliation_report.csv", index=False)
    quality.to_csv(outdir / "tsm_paper_reconciliation_quality_checks.csv", index=False)
    write_report_md(outdir, report, quality)
    print("completed: paper reconciliation outputs =", outdir.resolve())
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()

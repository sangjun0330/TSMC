#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build per-symbol research outputs for the semiconductor pooled prediction set.

This runner intentionally does not place orders. It creates the file layout
expected by tsm_pooled_dataset_builder.py:

- output/universe/<SYMBOL>/
- tsm_price_rule_output/universe/<SYMBOL>/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def run_step(name: str, cmd: List[str], continue_on_error: bool) -> Dict[str, object]:
    result = subprocess.run(cmd, text=True, capture_output=True)
    ok = result.returncode == 0
    if not ok and not continue_on_error:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return {
        "step": name,
        "returncode": result.returncode,
        "status": "OK" if ok else "FAILED",
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    }


def read_universe(path: Path) -> pd.DataFrame:
    config = strip_bom_columns(pd.read_csv(path))
    required = ["symbol", "symbol_stooq", "symbol_yahoo", "data_outdir", "rule_outdir"]
    missing = [c for c in required if c not in config.columns]
    if missing:
        raise ValueError(f"universe config missing columns: {missing}")
    return config


def builder_config_rows(config: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in config.iterrows():
        symbol = str(row["symbol"]).upper()
        group = str(row.get("symbol_group", "semiconductor"))
        data_outdir = Path(row["data_outdir"])
        rule_outdir = Path(row["rule_outdir"])
        rows.append(
            {
                "symbol": symbol,
                "symbol_group": group,
                "strict_eligible": row.get("strict_eligible", ""),
                "eligibility_status": row.get("eligibility_status", ""),
                "signals": rule_outdir / "tsm_daily_algorithmic_signals.csv",
                "risk_policy": rule_outdir / "tsm_risk_policy_daily.csv",
                "trade_log": rule_outdir / "tsm_backtest_trade_log.csv",
                "enriched": data_outdir / "tsm_daily_10y_enriched.csv",
            }
        )
    return pd.DataFrame(rows)


def append_cost_args(cmd: List[str], args: argparse.Namespace) -> List[str]:
    return [
        *cmd,
        "--commission-bps",
        str(args.commission_bps),
        "--slippage-bps",
        str(args.slippage_bps),
        "--stop-multiple",
        str(args.stop_multiple),
    ]


def run_symbol(row: pd.Series, args: argparse.Namespace) -> List[Dict[str, object]]:
    py = sys.executable
    symbol = str(row["symbol"]).upper()
    data_outdir = Path(row["data_outdir"])
    rule_outdir = Path(row["rule_outdir"])
    data_outdir.mkdir(parents=True, exist_ok=True)
    rule_outdir.mkdir(parents=True, exist_ok=True)
    steps: List[Dict[str, object]] = []

    daily_cmd = [
        py,
        "tsm_daily_quant_pipeline.py",
        "--symbol-stooq",
        str(row["symbol_stooq"]),
        "--symbol-yahoo",
        str(row["symbol_yahoo"]),
        "--start",
        args.start,
        "--end",
        args.end,
        "--outdir",
        str(data_outdir),
        "--preferred-source",
        args.preferred_source,
        "--skip-charts",
    ]
    if args.skip_benchmarks:
        daily_cmd.append("--skip-benchmarks")
    steps.append(run_step(f"{symbol}:daily_data", daily_cmd, args.continue_on_error))
    if steps[-1]["status"] != "OK":
        return steps

    rule_cmd = [
        py,
        "tsm_price_rule_engine.py",
        "--enriched",
        str(data_outdir / "tsm_daily_10y_enriched.csv"),
        "--raw",
        str(data_outdir / "tsm_daily_10y_raw.csv"),
        "--summary",
        str(data_outdir / "tsm_daily_10y_summary.csv"),
        "--events",
        str(data_outdir / "tsm_event_impact_10y.csv"),
        "--outdir",
        str(rule_outdir),
    ]
    steps.append(run_step(f"{symbol}:rule_engine", rule_cmd, args.continue_on_error))
    if steps[-1]["status"] != "OK":
        return steps

    backtest_cmd = append_cost_args(
        [
            py,
            "tsm_backtest_engine.py",
            "--signals",
            str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
            "--enriched",
            str(data_outdir / "tsm_daily_10y_enriched.csv"),
            "--outdir",
            str(rule_outdir),
        ],
        args,
    )
    steps.append(run_step(f"{symbol}:backtest", backtest_cmd, args.continue_on_error))
    if steps[-1]["status"] != "OK":
        return steps

    risk_cmd = [
        py,
        "tsm_risk_engine.py",
        "--signals",
        str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
        "--outdir",
        str(rule_outdir),
    ]
    steps.append(run_step(f"{symbol}:risk", risk_cmd, args.continue_on_error))
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pooled semiconductor research outputs without placing orders.")
    parser.add_argument("--universe-config", default="config/semiconductor_universe_expanded.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--start", default="2016-05-12")
    parser.add_argument("--end", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument("--preferred-source", choices=["stooq", "yahoo"], default="stooq")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stop-multiple", type=float, default=2.0)
    parser.add_argument("--skip-symbol-build", action="store_true", help="Only rebuild pooled dataset from existing per-symbol outputs.")
    parser.add_argument("--external-features", default="", help="Optional symbol/date external feature CSV.")
    parser.add_argument("--include-research-only-symbols", action="store_true", help="Include symbols that failed strict eligibility in the pooled builder config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = read_universe(Path(args.universe_config))
    manifest_rows: List[Dict[str, object]] = []
    if not args.skip_symbol_build:
        for _, row in config.iterrows():
            manifest_rows.extend(run_symbol(row, args))

    manifest_rows.append(
        run_step(
            "universe_validator",
            [
                sys.executable,
                "tsm_universe_validator.py",
                "--universe-config",
                str(args.universe_config),
                "--outdir",
                str(outdir),
            ],
            args.continue_on_error,
        )
    )
    validation_path = outdir / "tsm_universe_validation_report.csv"
    if validation_path.exists():
        validation = strip_bom_columns(pd.read_csv(validation_path))
        if not validation.empty and {"symbol", "strict_eligible", "eligibility_status"}.issubset(validation.columns):
            config = config.merge(
                validation[["symbol", "strict_eligible", "eligibility_status"]],
                on="symbol",
                how="left",
            )

    builder_config = builder_config_rows(config)
    builder_config_path = outdir / "tsm_prediction_pooled_universe_config.csv"
    builder_config.to_csv(builder_config_path, index=False)
    manifest_rows.append(
        run_step(
            "pooled_dataset_builder",
            [
                sys.executable,
                "tsm_pooled_dataset_builder.py",
                "--config",
                str(builder_config_path),
                "--outdir",
                str(outdir),
                "--commission-bps",
                str(args.commission_bps),
                "--slippage-bps",
                str(args.slippage_bps),
                "--stop-multiple",
                str(args.stop_multiple),
                *(
                    ["--external-features", str(args.external_features)]
                    if str(args.external_features).strip()
                    else []
                ),
            ],
            args.continue_on_error,
        )
    )
    pd.DataFrame(manifest_rows).to_csv(outdir / "tsm_pooled_universe_update_manifest.csv", index=False)
    print("completed: pooled universe outputs =", outdir.resolve())


if __name__ == "__main__":
    main()

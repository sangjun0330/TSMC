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

from tsm_core.universe import (
    DEFAULT_DECISION_UNIVERSE_CONFIG,
    DEFAULT_RESEARCH_UNIVERSE_CONFIG,
    add_scope_columns,
    build_universe_scope_audit,
    load_decision_universe_members,
    load_research_universe_members,
    load_universe_members,
    market_region_for_symbol,
    members_to_frame,
    symbol_set,
)


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
    return members_to_frame(load_universe_members(path))


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
                "market_region": row.get("market_region", market_region_for_symbol(symbol, row.get("symbol_yahoo", ""))),
                "is_decision_universe": bool(row.get("is_decision_universe", False)),
                "decision_scope": row.get("decision_scope", "top10"),
                "training_scope": row.get("training_scope", "universal_research_pool"),
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


def preferred_source_for_row(row: pd.Series, args: argparse.Namespace) -> str:
    region = market_region_for_symbol(row.get("symbol"), row.get("symbol_yahoo"), row.get("market_region"))
    if region == "KR":
        return str(getattr(args, "kr_preferred_source", "yahoo") or "yahoo")
    return str(getattr(args, "preferred_source", "stooq") or "stooq")


def run_symbol(row: pd.Series, args: argparse.Namespace) -> List[Dict[str, object]]:
    py = sys.executable
    symbol = str(row["symbol"]).upper()
    symbol_group = str(row.get("symbol_group", "semiconductor"))
    market_region = market_region_for_symbol(symbol, row.get("symbol_yahoo"), row.get("market_region"))
    preferred_source = preferred_source_for_row(row, args)
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
        preferred_source,
        "--listing-currency",
        str(row.get("listing_currency", "")),
        "--display-currency",
        str(row.get("display_currency", "")),
        "--engine-currency",
        str(row.get("engine_currency", "USD")),
        "--fx-pair",
        str(row.get("fx_pair", "")),
        "--fx-rates",
        str(args.fx_rates),
        "--skip-charts",
    ]
    if args.skip_benchmarks:
        daily_cmd.append("--skip-benchmarks")
    steps.append(run_step(f"{symbol}:{market_region}:daily_data:{preferred_source}", daily_cmd, args.continue_on_error))
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
        "--symbol",
        symbol,
        "--symbol-group",
        symbol_group,
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
    if steps[-1]["status"] != "OK":
        return steps

    if not args.skip_symbol_diagnostics:
        diagnostic_steps = [
            (
                "data_quality",
                [
                    py,
                    "tsm_data_quality_engine.py",
                    "--raw",
                    str(data_outdir / "tsm_daily_10y_raw.csv"),
                    "--enriched",
                    str(data_outdir / "tsm_daily_10y_enriched.csv"),
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--run-date",
                    args.end,
                ],
            ),
            (
                "backtest_event_ledger",
                [
                    py,
                    "tsm_backtest_event_ledger.py",
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--trade-log",
                    str(rule_outdir / "tsm_backtest_trade_log.csv"),
                    "--enriched",
                    str(data_outdir / "tsm_daily_10y_enriched.csv"),
                    "--outdir",
                    str(rule_outdir),
                    "--symbol",
                    symbol,
                    "--symbol-group",
                    symbol_group,
                    "--commission-bps",
                    str(args.commission_bps),
                    "--slippage-bps",
                    str(args.slippage_bps),
                ],
            ),
            (
                "stress",
                [
                    py,
                    "tsm_daily_stress_engine.py",
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--risk-policy",
                    str(rule_outdir / "tsm_risk_policy_daily.csv"),
                    "--drawdowns",
                    str(rule_outdir / "tsm_drawdown_episodes.csv"),
                    "--equity-curves",
                    str(rule_outdir / "tsm_backtest_equity_curves.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
            (
                "integrity",
                [
                    py,
                    "tsm_daily_integrity_engine.py",
                    "--raw",
                    str(data_outdir / "tsm_daily_10y_raw.csv"),
                    "--enriched",
                    str(data_outdir / "tsm_daily_10y_enriched.csv"),
                    "--signals",
                    str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                    "--risk-policy",
                    str(rule_outdir / "tsm_risk_policy_daily.csv"),
                    "--trade-log",
                    str(rule_outdir / "tsm_backtest_trade_log.csv"),
                    "--equity-curves",
                    str(rule_outdir / "tsm_backtest_equity_curves.csv"),
                    "--outdir",
                    str(rule_outdir),
                ],
            ),
        ]
        for diagnostic_name, diagnostic_cmd in diagnostic_steps:
            steps.append(run_step(f"{symbol}:{diagnostic_name}", diagnostic_cmd, args.continue_on_error))
            if steps[-1]["status"] != "OK":
                return steps

    if args.run_local_prediction:
        prediction_cmd = append_cost_args(
            [
                py,
                "tsm_prediction_engine.py",
                "--signals",
                str(rule_outdir / "tsm_daily_algorithmic_signals.csv"),
                "--trade-log",
                str(rule_outdir / "tsm_backtest_trade_log.csv"),
                "--risk-policy",
                str(rule_outdir / "tsm_risk_policy_daily.csv"),
                "--enriched",
                str(data_outdir / "tsm_daily_10y_enriched.csv"),
                "--outdir",
                str(rule_outdir),
                "--symbol",
                symbol,
            ],
            args,
        )
        steps.append(run_step(f"{symbol}:local_prediction", prediction_cmd, args.continue_on_error))
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pooled semiconductor research outputs without placing orders.")
    parser.add_argument("--universe-config", default=DEFAULT_DECISION_UNIVERSE_CONFIG, help="Legacy alias for the Top10 decision universe config.")
    parser.add_argument("--decision-universe-config", default="", help="Top10 operational universe used for intent/risk/execution/dashboard decisions.")
    parser.add_argument("--research-universe-config", default=DEFAULT_RESEARCH_UNIVERSE_CONFIG, help="Universal research pool used for pooled model training/calibration.")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--start", default="2016-05-12")
    parser.add_argument("--end", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument("--preferred-source", choices=["stooq", "yahoo"], default="stooq")
    parser.add_argument("--kr-preferred-source", choices=["stooq", "yahoo"], default="yahoo", help="Primary daily source for .KS/.KQ symbols; Korean data often becomes available before US data.")
    parser.add_argument("--fx-rates", default="output/tsm_fx_rates_daily.csv", help="Daily FX rates used for non-USD listings.")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stop-multiple", type=float, default=2.0)
    parser.add_argument("--skip-symbol-build", action="store_true", help="Only rebuild pooled dataset from existing per-symbol outputs.")
    parser.add_argument("--symbol-build-only", action="store_true", help="Refresh per-symbol universe daily outputs and stop before pooled dataset build.")
    parser.add_argument("--skip-symbol-diagnostics", action="store_true", help="Skip per-symbol data quality, event ledger, stress, and integrity diagnostics.")
    parser.add_argument("--run-local-prediction", action="store_true", help="Also run per-symbol local prediction outputs after risk.")
    parser.add_argument("--external-features", default="", help="Optional symbol/date external feature CSV.")
    parser.add_argument("--intraday-features", default="", help="Optional symbol/date intraday feature CSV.")
    parser.add_argument("--include-research-only-symbols", action="store_true", help="Include symbols that failed strict eligibility in the pooled builder config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    decision_config_path = Path(args.decision_universe_config or args.universe_config)
    research_config_path = Path(args.research_universe_config or DEFAULT_RESEARCH_UNIVERSE_CONFIG)
    decision_members = load_decision_universe_members(decision_config_path)
    research_members = load_research_universe_members(decision_config_path, research_config_path)
    decision_symbols = symbol_set(decision_members)
    decision_config = add_scope_columns(members_to_frame(decision_members), decision_symbols)
    config = add_scope_columns(members_to_frame(research_members), decision_symbols)
    decision_config.to_csv(outdir / "tsm_decision_universe_config.csv", index=False)
    build_universe_scope_audit(decision_members, research_members).to_csv(outdir / "tsm_research_pool_audit.csv", index=False)
    manifest_rows: List[Dict[str, object]] = []
    manifest_rows.append(
        run_step(
            "fx_rate_engine",
            [
                sys.executable,
                "tsm_fx_rate_engine.py",
                "--start",
                str(args.start),
                "--end",
                str(args.end),
                "--outdir",
                str(Path(args.fx_rates).parent),
            ],
            args.continue_on_error,
        )
    )
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
                str(decision_config_path),
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
            validation = validation[["symbol", "strict_eligible", "eligibility_status"]].copy()
            config = config.drop(columns=[c for c in ["strict_eligible", "eligibility_status"] if c in config.columns])
            config = config.merge(
                validation,
                on="symbol",
                how="left",
            )
            config["strict_eligible"] = config["strict_eligible"].fillna(True)
            config["eligibility_status"] = config["eligibility_status"].fillna("RESEARCH_POOL_NOT_VALIDATED")

    builder_config = builder_config_rows(config)
    builder_config_path = outdir / "tsm_prediction_pooled_universe_config.csv"
    builder_config.to_csv(builder_config_path, index=False)
    if args.symbol_build_only:
        manifest_rows.append(
            {
                "step": "pooled_dataset_builder",
                "returncode": 0,
                "status": "SKIPPED_SYMBOL_BUILD_ONLY",
                "duration_sec": 0.0,
                "stdout_tail": "",
                "stderr_tail": "",
            }
        )
        pd.DataFrame(manifest_rows).to_csv(outdir / "tsm_pooled_universe_update_manifest.csv", index=False)
        print("completed: pooled universe symbol build outputs =", outdir.resolve())
        return
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
                *(
                    ["--intraday-features", str(args.intraday_features)]
                    if str(args.intraday_features).strip()
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

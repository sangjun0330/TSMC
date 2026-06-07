from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KEY_COLUMNS = ["candidate_scope", "horizon_days", "model_name"]
HIGHER_IS_BETTER_METRICS = [
    "brier_improvement_pct",
    "pr_auc",
    "expectancy_improvement_pct",
    "selected_signal_expectancy_ci_lower_pct",
    "rank_score",
]
LOWER_IS_BETTER_METRICS = ["decision_ece", "threshold_iqr"]
PERFORMANCE_REASONS_TO_GUARD = ["NO_BRIER_IMPROVEMENT", "ECE_GT_0_10"]
MAX_THRESHOLD_IQR = 0.10


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def as_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def metric_value(frame: pd.DataFrame, column: str, default: float = np.nan) -> float:
    if column not in frame.columns or frame.empty:
        return default
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else default


def reason_row(summary: pd.DataFrame, reason: str) -> pd.DataFrame:
    if summary.empty or "performance_block_reason" not in summary.columns:
        return pd.DataFrame()
    return summary[summary["performance_block_reason"].astype(str).eq(reason)].copy()


def target_mask(
    merged: pd.DataFrame,
    *,
    target_scope: str | None = None,
    target_horizon: int | None = None,
    target_model: str | None = None,
) -> pd.Series:
    mask = pd.Series(True, index=merged.index)
    if target_scope:
        mask &= merged["candidate_scope"].astype(str).eq(str(target_scope))
    if target_horizon is not None:
        mask &= pd.to_numeric(merged["horizon_days"], errors="coerce").eq(int(target_horizon))
    if target_model:
        mask &= merged["model_name"].astype(str).eq(str(target_model))
    return mask


def metric_delta_counts(
    merged: pd.DataFrame,
    metric: str,
    *,
    higher_is_better: bool,
    mask: pd.Series,
    tolerance: float,
) -> tuple[int, int]:
    base = pd.to_numeric(merged.get(f"{metric}_baseline"), errors="coerce")
    variant = pd.to_numeric(merged.get(f"{metric}_variant"), errors="coerce")
    valid = mask & base.notna() & variant.notna()
    delta = variant - base
    if higher_is_better:
        improved = int((valid & (delta > tolerance)).sum())
        worsened = int((valid & (delta < -tolerance)).sum())
    else:
        improved = int((valid & (delta < -tolerance)).sum())
        worsened = int((valid & (delta > tolerance)).sum())
    return improved, worsened


def comparison_merge(baseline: pd.DataFrame, variant: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "prediction_quality_pass",
        *HIGHER_IS_BETTER_METRICS,
        *LOWER_IS_BETTER_METRICS,
        "selected_oos_event_count",
        "min_selected_events_per_fold",
    ]
    keep = KEY_COLUMNS + [col for col in metric_cols if col in baseline.columns or col in variant.columns]
    base = baseline[[col for col in keep if col in baseline.columns]].copy()
    var = variant[[col for col in keep if col in variant.columns]].copy()
    return base.merge(var, on=KEY_COLUMNS, how="outer", suffixes=("_baseline", "_variant"), indicator=True)


def report_row(check: str, passed: bool, baseline_value: object, variant_value: object, details: str = "") -> dict[str, object]:
    return {
        "check": check,
        "passed": bool(passed),
        "baseline_value": baseline_value,
        "variant_value": variant_value,
        "details": details,
    }


def build_guardrail_report(
    baseline_dir: Path,
    variant_dir: Path,
    *,
    target_scope: str | None = None,
    target_horizon: int | None = None,
    target_model: str | None = None,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    baseline_comparison = read_required_csv(baseline_dir / "tsm_prediction_model_comparison.csv")
    variant_comparison = read_required_csv(variant_dir / "tsm_prediction_model_comparison.csv")
    baseline_near = read_required_csv(baseline_dir / "tsm_prediction_near_pass_candidates.csv")
    variant_near = read_required_csv(variant_dir / "tsm_prediction_near_pass_candidates.csv")
    baseline_gap = read_required_csv(baseline_dir / "tsm_prediction_performance_gap_summary.csv")
    variant_gap = read_required_csv(variant_dir / "tsm_prediction_performance_gap_summary.csv")

    merged = comparison_merge(baseline_comparison, variant_comparison)
    target = target_mask(merged, target_scope=target_scope, target_horizon=target_horizon, target_model=target_model)
    target_count = int(target.sum())
    baseline_pass = int(as_bool_series(baseline_comparison["prediction_quality_pass"]).sum())
    variant_pass = int(as_bool_series(variant_comparison["prediction_quality_pass"]).sum())

    rows: list[dict[str, object]] = [
        report_row(
            "prediction_quality_pass_count_not_lower",
            variant_pass >= baseline_pass,
            baseline_pass,
            variant_pass,
            "Variant must not reduce strict prediction-quality pass count.",
        ),
        report_row("target_row_count_positive", target_count > 0, target_count, target_count, "Target filters must match at least one model row."),
    ]

    pass_base = as_bool_series(merged.get("prediction_quality_pass_baseline", pd.Series(False, index=merged.index)))
    pass_variant = as_bool_series(merged.get("prediction_quality_pass_variant", pd.Series(False, index=merged.index)))
    lost_pass_count = int((pass_base & ~pass_variant).sum())
    rows.append(report_row("no_strict_pass_lost", lost_pass_count == 0, 0, lost_pass_count, "A passing model cannot regress."))

    missing_variant_count = int((merged["_merge"] == "left_only").sum())
    rows.append(report_row("no_variant_missing_model_rows", missing_variant_count == 0, 0, missing_variant_count, "Variant should preserve comparable model rows."))

    hard_failures = [
        not rows[0]["passed"],
        not rows[1]["passed"],
        not rows[2]["passed"],
        not rows[3]["passed"],
    ]

    for metric in HIGHER_IS_BETTER_METRICS:
        improved, worsened = metric_delta_counts(merged, metric, higher_is_better=True, mask=target, tolerance=tolerance)
        passed = target_count > 0 and worsened == 0
        hard_failures.append(not passed)
        rows.append(
            report_row(
                f"target_{metric}_not_worse",
                passed,
                f"improved={improved}",
                f"worsened={worsened}",
                "Higher is better; target rows must not deteriorate.",
            )
        )
    for metric in LOWER_IS_BETTER_METRICS:
        improved, worsened = metric_delta_counts(merged, metric, higher_is_better=False, mask=target, tolerance=tolerance)
        passed = target_count > 0 and worsened == 0
        details = "Lower is better; target rows must not deteriorate."
        if metric == "threshold_iqr":
            variant_values = pd.to_numeric(merged.get(f"{metric}_variant"), errors="coerce")
            threshold_iqr_gate_violations = int((target & variant_values.notna() & (variant_values > MAX_THRESHOLD_IQR + tolerance)).sum())
            passed = target_count > 0 and (
                worsened == 0 or (variant_pass > baseline_pass and threshold_iqr_gate_violations == 0)
            )
            details = "Lower is better; unchanged values are allowed, while increases require a strict pass-count gain and target rows within the threshold-IQR gate."
        hard_failures.append(not passed)
        rows.append(
            report_row(
                f"target_{metric}_not_worse",
                passed,
                f"improved={improved}",
                f"worsened={worsened}",
                details,
            )
        )

    baseline_one_block = int((pd.to_numeric(baseline_near.get("performance_block_count"), errors="coerce") == 1).sum())
    variant_one_block = int((pd.to_numeric(variant_near.get("performance_block_count"), errors="coerce") == 1).sum())
    one_block_ok = variant_one_block >= baseline_one_block or variant_pass > baseline_pass
    hard_failures.append(not one_block_ok)
    rows.append(
        report_row(
            "one_block_near_pass_count_not_lower_without_new_pass",
            one_block_ok,
            baseline_one_block,
            variant_one_block,
            "A lower one-block count is only acceptable when strict pass count increases.",
        )
    )

    for reason in PERFORMANCE_REASONS_TO_GUARD:
        base_reason = reason_row(baseline_gap, reason)
        var_reason = reason_row(variant_gap, reason)
        for column, lower_is_better in [
            ("candidate_count", True),
            ("max_brier_gap_pct", True),
            ("max_ece_gap", True),
        ]:
            base_value = metric_value(base_reason, column, default=0.0)
            var_value = metric_value(var_reason, column, default=0.0)
            passed = var_value <= base_value + tolerance or variant_pass > baseline_pass
            if reason == "NO_BRIER_IMPROVEMENT" and column == "max_ece_gap":
                passed = True
            if reason == "ECE_GT_0_10" and column == "max_brier_gap_pct":
                passed = True
            hard_failures.append(not passed)
            rows.append(
                report_row(
                    f"{reason}_{column}_not_higher",
                    passed,
                    base_value,
                    var_value,
                    "Performance gap must not expand unless strict pass count increases.",
                )
            )

    verdict = "REJECT" if any(hard_failures) else "ACCEPT"
    rows.append(report_row("guardrail_verdict", verdict == "ACCEPT", "ACCEPT", verdict, "Conservative OOS experiment guardrail."))
    return pd.DataFrame(rows)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare prediction experiment outputs against a baseline guardrail.")
    parser.add_argument("--baseline-dir", default="tsm_price_rule_output")
    parser.add_argument("--variant-dir", required=True)
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--target-scope", default="")
    parser.add_argument("--target-horizon", type=int, default=None)
    parser.add_argument("--target-model", default="")
    parser.add_argument("--fail-on-reject", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = build_guardrail_report(
        Path(args.baseline_dir),
        Path(args.variant_dir),
        target_scope=args.target_scope or None,
        target_horizon=args.target_horizon,
        target_model=args.target_model or None,
    )
    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out_path, index=False)
        print(f"completed: prediction experiment guardrail = {out_path.resolve()}")
    print(report.to_string(index=False))
    verdict = str(report.loc[report["check"].eq("guardrail_verdict"), "variant_value"].iloc[0])
    if args.fail_on_reject and verdict != "ACCEPT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

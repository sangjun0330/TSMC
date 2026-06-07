#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a reproducible model registry snapshot from prediction outputs.

Outputs:
- tsm_prediction_model_registry.csv
- tsm_prediction_experiment_log.csv
- tsm_prediction_model_registry_report.md
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


DEPENDENCY_PACKAGES = ("pandas", "numpy", "scikit-learn", "lightgbm", "xgboost", "optuna")


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return strip_bom_columns(pd.read_csv(path, low_memory=False))


def file_sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_versions() -> str:
    rows = []
    for package in DEPENDENCY_PACKAGES:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "missing"
        rows.append(f"{package}={version}")
    return "|".join(rows)


def pooled_feature_count(pooled_schema: pd.DataFrame, pooled_features: pd.DataFrame) -> int:
    if not pooled_schema.empty and {"column", "role"}.issubset(pooled_schema.columns):
        roles = pooled_schema["role"].astype(str)
        return int(roles.str.contains("feature", case=False, na=False).sum())
    if not pooled_features.empty:
        non_feature_cols = {"symbol", "symbol_group", "date", "date_split", "split", "signal_idx"}
        return int(len([c for c in pooled_features.columns if c not in non_feature_cols and not str(c).startswith("label_")]))
    return 0


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def registry_id(row: pd.Series) -> str:
    basis = (
        f"{row.get('candidate_scope')}|{row.get('horizon_days')}|{row.get('model_name')}|"
        f"{row.get('split', '')}|{row.get('evaluation_scope', '')}|{row.get('validation_design', '')}|"
        f"{row.get('champion_scope', '')}|{row.get('generated_at_utc')}"
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def key_value_frame_to_dict(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty or not {"field", "value"}.issubset(frame.columns):
        return {}
    return dict(zip(frame["field"].astype(str), frame["value"]))


def selected_feature_counts(feature_selection: pd.DataFrame) -> pd.DataFrame:
    if feature_selection.empty:
        return pd.DataFrame(columns=["candidate_scope", "horizon_days", "selected_feature_count_median", "top_selected_features"])
    selected = feature_selection[feature_selection["decision"].eq("selected")].copy()
    if selected.empty:
        return pd.DataFrame(columns=["candidate_scope", "horizon_days", "selected_feature_count_median", "top_selected_features"])
    counts = selected.groupby(["candidate_scope", "horizon_days", "fold_id"]).size().reset_index(name="selected_feature_count")
    top = (
        selected.groupby(["candidate_scope", "horizon_days", "feature"])
        .size()
        .reset_index(name="n")
        .sort_values(["candidate_scope", "horizon_days", "n"], ascending=[True, True, False])
    )
    top_features = top.groupby(["candidate_scope", "horizon_days"]).head(10).groupby(["candidate_scope", "horizon_days"])["feature"].apply(lambda x: "|".join(x)).reset_index(name="top_selected_features")
    out = counts.groupby(["candidate_scope", "horizon_days"])["selected_feature_count"].median().reset_index(name="selected_feature_count_median")
    return out.merge(top_features, on=["candidate_scope", "horizon_days"], how="left")


def overlay_deltas(overlay: pd.DataFrame) -> pd.DataFrame:
    if overlay.empty:
        return pd.DataFrame()
    delta = overlay[overlay["policy"].eq("ML_SELECTED_MINUS_RULE_ALL")].copy()
    keep = [
        "candidate_scope",
        "horizon_days",
        "model_name",
        "event_count",
        "selection_rate_pct",
        "mean_net_return_pct",
        "success_rate_pct",
        "stop_rate_pct",
        "cumulative_weighted_return_pct",
        "max_event_curve_drawdown_pct",
    ]
    return delta[[c for c in keep if c in delta.columns]].rename(
        columns={
            "event_count": "overlay_selected_event_count",
            "mean_net_return_pct": "overlay_mean_return_improvement_pct",
            "success_rate_pct": "overlay_success_rate_improvement_pct",
            "stop_rate_pct": "overlay_stop_rate_delta_pct",
            "cumulative_weighted_return_pct": "overlay_weighted_return_delta_pct",
            "max_event_curve_drawdown_pct": "overlay_mdd_delta_pct",
        }
    )


def build_registry(
    comparison: pd.DataFrame,
    audit: pd.DataFrame,
    thresholds: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    feature_selection: pd.DataFrame,
    overlay: pd.DataFrame,
    source_hashes: Dict[str, str],
) -> pd.DataFrame:
    require_columns(comparison, ["candidate_scope", "horizon_days", "model_name"], "model comparison")
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    registry = comparison.copy()
    registry["generated_at_utc"] = generated_at

    if not audit.empty:
        audit_keep = [
            "candidate_scope",
            "horizon_days",
            "model_name",
            "quality_block_reasons",
            "brier_reliability",
            "brier_resolution",
            "brier_uncertainty",
            "threshold_median",
            "threshold_std",
            "calibration_methods",
        ]
        registry = registry.merge(audit[[c for c in audit_keep if c in audit.columns]], on=["candidate_scope", "horizon_days", "model_name"], how="left")

    if not thresholds.empty:
        threshold_agg = thresholds.groupby(["candidate_scope", "horizon_days", "model_name"]).agg(
            threshold_policy_median=("threshold", "median"),
            threshold_policy_std=("threshold", "std"),
            validation_selected_count_sum=("validation_selected_count", "sum"),
        ).reset_index()
        registry = registry.merge(threshold_agg, on=["candidate_scope", "horizon_days", "model_name"], how="left")

    if not calibration_summary.empty:
        cal = calibration_summary.pivot_table(
            index=["candidate_scope", "horizon_days", "model_name"],
            columns="binning",
            values="ece",
            aggfunc="mean",
        ).reset_index()
        cal.columns = [str(c) if c in {"candidate_scope", "horizon_days", "model_name"} else f"ece_{c}" for c in cal.columns]
        registry = registry.merge(cal, on=["candidate_scope", "horizon_days", "model_name"], how="left")

    registry = registry.merge(selected_feature_counts(feature_selection), on=["candidate_scope", "horizon_days"], how="left")
    od = overlay_deltas(overlay)
    if not od.empty:
        registry = registry.merge(od, on=["candidate_scope", "horizon_days", "model_name"], how="left")

    registry["source_hash_prediction_oos"] = source_hashes.get("oos_predictions", "")
    registry["source_hash_model_comparison"] = source_hashes.get("comparison", "")
    registry["source_hash_overlay"] = source_hashes.get("overlay", "")
    registry["schema_version"] = "tsm_prediction_schema_v3"
    registry["config_hash"] = source_hashes.get("config", "")
    registry["dataset_hash"] = source_hashes.get("dataset", "")
    registry["feature_contract_hash"] = source_hashes.get("feature_contract", "")
    registry["fold_manifest_hash"] = source_hashes.get("fold_manifest", "")
    registry["source_hash_schema_quality"] = source_hashes.get("schema_quality", "")
    registry["source_hash_prediction_engine"] = source_hashes.get("prediction_engine_source", "")
    registry["registry_id"] = registry.apply(registry_id, axis=1)
    first_cols = ["registry_id", "generated_at_utc", "candidate_scope", "horizon_days", "model_name", "model_policy", "prediction_quality_pass"]
    return registry[[c for c in first_cols if c in registry.columns] + [c for c in registry.columns if c not in first_cols]]


def build_experiment_log(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame()
    rows = []
    for _, row in registry.iterrows():
        if bool(row.get("prediction_quality_pass", False)):
            status = "decision_support_candidate"
        elif str(row.get("model_policy", "")) == "RESEARCH_ONLY":
            status = "research_only"
        else:
            status = "blocked"
        rows.append(
            {
                "experiment_id": row["registry_id"],
                "generated_at_utc": row["generated_at_utc"],
                "candidate_scope": row["candidate_scope"],
                "horizon_days": row["horizon_days"],
                "model_name": row["model_name"],
                "status": status,
                "block_reasons": row.get("quality_block_reasons", ""),
                "oos_event_count": row.get("oos_event_count", np.nan),
                "brier_improvement_pct": row.get("brier_improvement_pct", np.nan),
                "ece": row.get("ece", np.nan),
                "pr_auc": row.get("pr_auc", np.nan),
                "expectancy_improvement_pct": row.get("expectancy_improvement_pct", np.nan),
                "overlay_mean_return_improvement_pct": row.get("overlay_mean_return_improvement_pct", np.nan),
            }
        )
    return pd.DataFrame(rows)


def build_pooled_registry(
    pooled_comparison: pd.DataFrame,
    pooled_latest: pd.DataFrame,
    pooled_schema: pd.DataFrame,
    pooled_features: pd.DataFrame,
    model_gate_snapshot: pd.DataFrame,
    source_hashes: Dict[str, str],
) -> pd.DataFrame:
    if pooled_comparison.empty:
        return pd.DataFrame()
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    latest = key_value_frame_to_dict(pooled_latest)
    gate = key_value_frame_to_dict(model_gate_snapshot)
    pooled_system_quality_pass = to_bool(gate.get("pooled_system_quality_pass", False))
    pooled_prediction_decision_support = to_bool(gate.get("pooled_prediction_decision_support", False))
    dependency_version_text = dependency_versions()
    feature_count = pooled_feature_count(pooled_schema, pooled_features)
    frame = pooled_comparison[pooled_comparison["split"].astype(str).eq("combined_test_holdout")].copy()
    if frame.empty:
        frame = pooled_comparison.copy()
    rows = []
    for _, row in frame.iterrows():
        is_champion = to_bool(row.get("is_champion", False))
        model_quality_pass = to_bool(latest.get("model_quality_pass", False)) if is_champion else False
        latest_signal_pass = to_bool(latest.get("latest_signal_pass", False)) if is_champion else False
        diagnostic_champion = bool(is_champion)
        promotable_model = bool(is_champion and pooled_system_quality_pass)
        decision_support_allowed = bool(is_champion and pooled_prediction_decision_support)
        model_policy = (
            "POOLED_PROMOTABLE_CHAMPION"
            if promotable_model
            else "POOLED_DIAGNOSTIC_CHAMPION_BLOCKED"
            if diagnostic_champion
            else "POOLED_CANDIDATE"
        )
        out = row.to_dict()
        out.update(
            {
                "generated_at_utc": generated_at,
                "candidate_scope": "pooled_trade_ready_entry",
                "champion_scope": "pooled_trade_ready_20d",
                "horizon_days": 20,
                "model_policy": model_policy,
                "diagnostic_champion": diagnostic_champion,
                "promotable_model": promotable_model,
                "decision_support_allowed": decision_support_allowed,
                "prediction_quality_pass": promotable_model,
                "model_quality_pass": bool(model_quality_pass),
                "pooled_system_quality_pass": bool(is_champion and pooled_system_quality_pass),
                "oos_event_count": row.get("event_count", np.nan),
                "selected_oos_event_count": row.get("selected_event_count", np.nan),
                "expectancy_improvement_pct": row.get("selected_minus_all_pct", np.nan),
                "pr_auc": row.get("average_precision", np.nan),
                "promotion_status": "PROMOTABLE" if promotable_model else "BLOCKED",
                "latest_signal_pass": bool(latest_signal_pass),
                "pooled_decision_support_allowed": decision_support_allowed,
                "quality_block_reasons": latest.get("model_quality_block_reasons", "") if is_champion else "NOT_CHAMPION",
                "source_hash_pooled_comparison": source_hashes.get("pooled_comparison", ""),
                "source_hash_pooled_oos": source_hashes.get("pooled_oos_predictions", ""),
                "source_hash_pooled_threshold_policy": source_hashes.get("pooled_threshold_policy", ""),
                "source_hash_pooled_quality": source_hashes.get("pooled_quality", ""),
                "source_hash_pooled_latest": source_hashes.get("pooled_latest", ""),
                "source_hash_model_gate_snapshot": source_hashes.get("model_gate_snapshot", ""),
                "source_hash_pooled_schema": source_hashes.get("pooled_schema", ""),
                "source_hash_pooled_feature_matrix": source_hashes.get("pooled_feature_matrix", ""),
                "dependency_versions": dependency_version_text,
                "feature_count": feature_count,
                "schema_version": "tsm_pooled_prediction_schema_v1",
            }
        )
        out["rank_score"] = (
            int(is_champion) * 1000
            + float(pd.to_numeric(pd.Series([out.get("selected_minus_all_pct")]), errors="coerce").iloc[0] if pd.notna(out.get("selected_minus_all_pct")) else -999)
            + float(pd.to_numeric(pd.Series([out.get("brier_improvement_pct")]), errors="coerce").iloc[0] if pd.notna(out.get("brier_improvement_pct")) else -999) / 10.0
            - float(pd.to_numeric(pd.Series([out.get("ece")]), errors="coerce").iloc[0] if pd.notna(out.get("ece")) else 1.0) * 10.0
        )
        rows.append(out)
    pooled = pd.DataFrame(rows)
    if pooled.empty:
        return pooled
    pooled["registry_id"] = pooled.apply(registry_id, axis=1)
    return pooled


def write_report(outdir: Path, registry: pd.DataFrame, experiment_log: pd.DataFrame) -> None:
    def display_value(row: pd.Series, column: str, default: object = False) -> object:
        value = row.get(column, default)
        if pd.isna(value):
            return default
        return value

    lines = [
        "# Top10 Prediction Model Registry Report",
        "",
        f"- Registered model rows: {len(registry)}",
        f"- Experiment log rows: {len(experiment_log)}",
        "",
        "## Current Registry",
        "",
        "| Scope | Horizon | Model | Policy | Promotion | Promotable | Diagnostic | Model Pass | Latest Pass | Decision Support | OOS | Brier Improvement | ECE | Block Reasons |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    if registry.empty:
        lines.append("| NA | NA | NA | NA | BLOCKED | False | False | False | False | False | 0 | NA | NA | missing |")
    else:
        for _, row in registry.sort_values(["candidate_scope", "horizon_days", "rank_score"], ascending=[True, True, False]).iterrows():
            oos_event_count = pd.to_numeric(pd.Series([row.get("oos_event_count", 0)]), errors="coerce").fillna(0).iloc[0]
            lines.append(
                f"| {row.get('candidate_scope')} | {int(row.get('horizon_days'))} | {row.get('model_name')} | {row.get('model_policy')} | "
                f"{display_value(row, 'promotion_status', 'BLOCKED')} | {display_value(row, 'promotable_model', display_value(row, 'prediction_quality_pass', False))} | "
                f"{display_value(row, 'diagnostic_champion', False)} | {display_value(row, 'model_quality_pass', display_value(row, 'prediction_quality_pass', False))} | "
                f"{display_value(row, 'latest_signal_pass', False)} | {display_value(row, 'pooled_decision_support_allowed', display_value(row, 'decision_support_allowed', False))} | "
                f"{int(oos_event_count)} | "
                f"{row.get('brier_improvement_pct', np.nan):.2f}% | {row.get('ece', np.nan):.4f} | "
                f"{row.get('quality_block_reasons', '')} |"
            )
    lines.extend(["", "This report is research tooling, not investment advice."])
    (outdir / "tsm_prediction_model_registry_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model registry snapshot.")
    parser.add_argument("--comparison", default="tsm_price_rule_output/tsm_prediction_model_comparison.csv")
    parser.add_argument("--audit", default="tsm_price_rule_output/tsm_prediction_model_audit.csv")
    parser.add_argument("--thresholds", default="tsm_price_rule_output/tsm_prediction_threshold_policy.csv")
    parser.add_argument("--calibration-summary", default="tsm_price_rule_output/tsm_prediction_calibration_summary.csv")
    parser.add_argument("--feature-selection", default="tsm_price_rule_output/tsm_prediction_feature_selection_report.csv")
    parser.add_argument("--overlay", default="tsm_price_rule_output/tsm_ml_overlay_summary.csv")
    parser.add_argument("--oos-predictions", default="tsm_price_rule_output/tsm_prediction_oos_predictions.csv")
    parser.add_argument("--label-dataset", default="tsm_price_rule_output/tsm_prediction_label_dataset.csv")
    parser.add_argument("--feature-matrix", default="tsm_price_rule_output/tsm_prediction_feature_matrix.csv")
    parser.add_argument("--feature-contract", default="tsm_price_rule_output/tsm_prediction_feature_contract.csv")
    parser.add_argument("--fold-manifest", default="tsm_price_rule_output/tsm_prediction_fold_manifest.csv")
    parser.add_argument("--schema-quality", default="tsm_price_rule_output/tsm_schema_quality_checks.csv")
    parser.add_argument("--config", default="config/tsm_research.toml")
    parser.add_argument("--prediction-engine-source", default="tsm_prediction_engine.py")
    parser.add_argument("--pooled-comparison", default="tsm_price_rule_output/tsm_pooled_model_comparison.csv")
    parser.add_argument("--pooled-oos-predictions", default="tsm_price_rule_output/tsm_pooled_model_oos_predictions.csv")
    parser.add_argument("--pooled-threshold-policy", default="tsm_price_rule_output/tsm_pooled_model_threshold_policy.csv")
    parser.add_argument("--pooled-quality", default="tsm_price_rule_output/tsm_pooled_model_quality_checks.csv")
    parser.add_argument("--pooled-latest", default="tsm_price_rule_output/tsm_pooled_latest_prediction_overlay.csv")
    parser.add_argument("--model-gate-snapshot", default="tsm_price_rule_output/tsm_model_gate_snapshot.csv")
    parser.add_argument("--pooled-schema", default="tsm_price_rule_output/tsm_prediction_pooled_schema.csv")
    parser.add_argument("--pooled-feature-matrix", default="tsm_price_rule_output/tsm_prediction_pooled_feature_matrix.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    comparison_path = Path(args.comparison)
    overlay_path = Path(args.overlay)
    oos_path = Path(args.oos_predictions)
    label_path = Path(args.label_dataset)
    feature_path = Path(args.feature_matrix)
    feature_contract_path = Path(args.feature_contract)
    fold_manifest_path = Path(args.fold_manifest)
    schema_quality_path = Path(args.schema_quality)
    config_path = Path(args.config)
    prediction_source_path = Path(args.prediction_engine_source)
    pooled_comparison_path = Path(args.pooled_comparison)
    pooled_oos_path = Path(args.pooled_oos_predictions)
    pooled_threshold_path = Path(args.pooled_threshold_policy)
    pooled_quality_path = Path(args.pooled_quality)
    pooled_latest_path = Path(args.pooled_latest)
    model_gate_snapshot_path = Path(args.model_gate_snapshot)
    pooled_schema_path = Path(args.pooled_schema)
    pooled_feature_matrix_path = Path(args.pooled_feature_matrix)
    registry = build_registry(
        read_csv(comparison_path),
        read_csv(Path(args.audit)),
        read_csv(Path(args.thresholds)),
        read_csv(Path(args.calibration_summary)),
        read_csv(Path(args.feature_selection)),
        read_csv(overlay_path),
        {
            "comparison": file_sha256(comparison_path),
            "overlay": file_sha256(overlay_path),
            "oos_predictions": file_sha256(oos_path),
            "dataset": hashlib.sha256((file_sha256(label_path) + file_sha256(feature_path)).encode("utf-8")).hexdigest(),
            "feature_contract": file_sha256(feature_contract_path),
            "fold_manifest": file_sha256(fold_manifest_path),
            "schema_quality": file_sha256(schema_quality_path),
            "config": file_sha256(config_path),
            "prediction_engine_source": file_sha256(prediction_source_path),
        },
    )
    pooled_registry = build_pooled_registry(
        read_csv(pooled_comparison_path),
        read_csv(pooled_latest_path),
        read_csv(pooled_schema_path),
        read_csv(pooled_feature_matrix_path),
        read_csv(model_gate_snapshot_path),
        {
            "pooled_comparison": file_sha256(pooled_comparison_path),
            "pooled_oos_predictions": file_sha256(pooled_oos_path),
            "pooled_threshold_policy": file_sha256(pooled_threshold_path),
            "pooled_quality": file_sha256(pooled_quality_path),
            "pooled_latest": file_sha256(pooled_latest_path),
            "model_gate_snapshot": file_sha256(model_gate_snapshot_path),
            "pooled_schema": file_sha256(pooled_schema_path),
            "pooled_feature_matrix": file_sha256(pooled_feature_matrix_path),
        },
    )
    if not pooled_registry.empty:
        registry = pd.concat([registry, pooled_registry], ignore_index=True, sort=False)
    experiment_log = build_experiment_log(registry)
    registry.to_csv(outdir / "tsm_prediction_model_registry.csv", index=False)
    experiment_log.to_csv(outdir / "tsm_prediction_experiment_log.csv", index=False)
    write_report(outdir, registry, experiment_log)
    print("완료: model registry outputs =", outdir.resolve())
    print(experiment_log.head(30).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic research trial ledger.

This engine records research trials against existing OOF/model comparison
artifacts. It intentionally does not maximize CAGR directly. The score rewards
calibrated probability and conservative selected-event uplift, while penalizing
ECE, threshold instability, fold drift, PBO, and DSR failures.

Outputs:
- tsm_auto_research_trial_ledger.csv
- tsm_auto_research_best_candidates.csv
- tsm_auto_research_report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import optuna

    OPTUNA_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local env
    optuna = None
    OPTUNA_IMPORT_ERROR = exc


ALLOWED_MODEL_FAMILIES = ("empirical_bayes", "elastic_net", "lightgbm", "xgboost", "stacked_calibrated")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        out = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    out.columns = [str(c).lstrip("\ufeff") for c in out.columns]
    return out


def as_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def hash_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def infer_model_family(model_name: object) -> str:
    name = str(model_name).lower()
    if "elastic" in name:
        return "elastic_net"
    if "lightgbm" in name or "lgbm" in name:
        return "lightgbm"
    if "xgboost" in name or "xgb" in name:
        return "xgboost"
    if "stack" in name or "blend" in name or "calibrated" in name:
        return "stacked_calibrated"
    return "empirical_bayes"


def normalize_candidate_frame(pooled: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if not pooled.empty:
        p = pooled.copy()
        p["source_scope"] = "pooled"
        frames.append(p)
    if not local.empty:
        local_frame = local.copy()
        local_frame["source_scope"] = "local"
        frames.append(local_frame)
    if not frames:
        return pd.DataFrame()
    candidates = pd.concat(frames, ignore_index=True, sort=False)
    candidates["model_family"] = candidates.get("model_name", "model").map(infer_model_family)
    candidates = candidates[candidates["model_family"].isin(ALLOWED_MODEL_FAMILIES)].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates["_brier_improvement"] = pd.to_numeric(candidates.get("brier_improvement_pct", 0.0), errors="coerce").fillna(0.0)
    candidates["_uplift_all_lower"] = pd.to_numeric(
        candidates.get("selected_minus_all_ci_lower_pct_paired", candidates.get("selected_minus_all_ci_lower_pct", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    candidates["_uplift_score_lower"] = pd.to_numeric(
        candidates.get("selected_minus_score_baseline_ci_lower_pct_paired", candidates.get("selected_minus_score_baseline_ci_lower_pct", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    candidates["_ece"] = pd.to_numeric(candidates.get("decision_ece", candidates.get("ece", np.nan)), errors="coerce").fillna(1.0)
    candidates["_threshold_iqr"] = pd.to_numeric(candidates.get("threshold_iqr", np.nan), errors="coerce").fillna(1.0)
    selected_fraction = pd.to_numeric(candidates.get("selected_fraction", np.nan), errors="coerce")
    candidates["_fold_selection_drift"] = (selected_fraction - 0.45).abs().fillna(0.25)
    return candidates.reset_index(drop=True)


def validation_penalties(pbo: pd.DataFrame, dsr: pd.DataFrame, cpcv_model: pd.DataFrame) -> dict[str, float]:
    pbo_values = []
    for col in ["pbo_proxy", "pbo_cscv"]:
        if col in pbo.columns:
            pbo_values.extend(pd.to_numeric(pbo[col], errors="coerce").dropna().tolist())
    pbo_penalty = float(max(pbo_values)) if pbo_values else 0.50
    dsr_pass = bool(not dsr.empty and dsr.get("dsr_pass", pd.Series(dtype=bool)).map(to_bool).any())
    cpcv_q25_values = []
    if not cpcv_model.empty and "selected_minus_all_pct_q25" in cpcv_model.columns:
        cpcv_q25_values = pd.to_numeric(cpcv_model["selected_minus_all_pct_q25"], errors="coerce").dropna().tolist()
    cpcv_penalty = 0.0 if cpcv_q25_values and max(cpcv_q25_values) >= 0.0 else 0.25
    return {
        "pbo_penalty": pbo_penalty,
        "dsr_penalty": 0.0 if dsr_pass else 1.0,
        "cpcv_penalty": cpcv_penalty,
    }


def score_candidates(candidates: pd.DataFrame, weights: dict[str, float], penalties: dict[str, float]) -> pd.DataFrame:
    scored = candidates.copy()
    scored["objective_score"] = (
        weights["brier"] * scored["_brier_improvement"]
        + weights["uplift_all"] * scored["_uplift_all_lower"]
        + weights["uplift_score"] * scored["_uplift_score_lower"]
        - weights["ece"] * scored["_ece"] * 100.0
        - weights["threshold"] * scored["_threshold_iqr"] * 100.0
        - weights["fold_drift"] * scored["_fold_selection_drift"] * 100.0
        - weights["pbo"] * penalties["pbo_penalty"] * 100.0
        - weights["dsr"] * penalties["dsr_penalty"] * 100.0
        - weights["cpcv"] * penalties["cpcv_penalty"] * 100.0
    )
    return scored.sort_values("objective_score", ascending=False).reset_index(drop=True)


def deterministic_trials(candidates: pd.DataFrame, penalties: dict[str, float], max_trials: int, metadata: dict[str, object]) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: List[Dict[str, object]] = []
    for trial_id in range(max(int(max_trials), 1)):
        weights = {
            "brier": float(rng.uniform(0.8, 1.5)),
            "uplift_all": float(rng.uniform(0.8, 1.8)),
            "uplift_score": float(rng.uniform(0.8, 1.8)),
            "ece": float(rng.uniform(0.6, 1.6)),
            "threshold": float(rng.uniform(0.4, 1.2)),
            "fold_drift": float(rng.uniform(0.2, 1.0)),
            "pbo": float(rng.uniform(0.6, 1.6)),
            "dsr": float(rng.uniform(0.6, 1.6)),
            "cpcv": float(rng.uniform(0.5, 1.4)),
        }
        scored = score_candidates(candidates, weights, penalties)
        best = scored.iloc[0]
        rows.append(
            {
                "trial_id": trial_id,
                "trial_source": "deterministic_sampler",
                "objective_score": best["objective_score"],
                "model_family": best.get("model_family"),
                "model_name": best.get("model_name"),
                "candidate_scope": best.get("candidate_scope", best.get("evaluation_scope", "")),
                "split": best.get("split", ""),
                "horizon_days": best.get("horizon_days", ""),
                "weights_json": json.dumps(weights, sort_keys=True),
                "pbo_penalty": penalties["pbo_penalty"],
                "dsr_penalty": penalties["dsr_penalty"],
                "cpcv_penalty": penalties["cpcv_penalty"],
                **metadata,
            }
        )
    return pd.DataFrame(rows)


def optuna_trials(candidates: pd.DataFrame, penalties: dict[str, float], max_trials: int, storage: str, metadata: dict[str, object]) -> pd.DataFrame:
    if optuna is None:
        return deterministic_trials(candidates, penalties, max_trials, metadata)
    study = optuna.create_study(
        study_name="tsm_auto_research_objective",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial) -> float:
        weights = {
            "brier": trial.suggest_float("brier_weight", 0.8, 1.5),
            "uplift_all": trial.suggest_float("uplift_all_weight", 0.8, 1.8),
            "uplift_score": trial.suggest_float("uplift_score_weight", 0.8, 1.8),
            "ece": trial.suggest_float("ece_weight", 0.6, 1.6),
            "threshold": trial.suggest_float("threshold_weight", 0.4, 1.2),
            "fold_drift": trial.suggest_float("fold_drift_weight", 0.2, 1.0),
            "pbo": trial.suggest_float("pbo_weight", 0.6, 1.6),
            "dsr": trial.suggest_float("dsr_weight", 0.6, 1.6),
            "cpcv": trial.suggest_float("cpcv_weight", 0.5, 1.4),
        }
        scored = score_candidates(candidates, weights, penalties)
        best = scored.iloc[0]
        trial.set_user_attr("model_family", str(best.get("model_family", "")))
        trial.set_user_attr("model_name", str(best.get("model_name", "")))
        trial.set_user_attr("candidate_scope", str(best.get("candidate_scope", best.get("evaluation_scope", ""))))
        trial.set_user_attr("split", str(best.get("split", "")))
        trial.set_user_attr("weights_json", json.dumps(weights, sort_keys=True))
        return float(best["objective_score"])

    if max_trials > 0:
        study.optimize(objective, n_trials=int(max_trials), show_progress_bar=False)
    rows = []
    for t in study.trials:
        rows.append(
            {
                "trial_id": t.number,
                "trial_source": "optuna",
                "objective_score": t.value,
                "model_family": t.user_attrs.get("model_family", ""),
                "model_name": t.user_attrs.get("model_name", ""),
                "candidate_scope": t.user_attrs.get("candidate_scope", ""),
                "split": t.user_attrs.get("split", ""),
                "horizon_days": "",
                "weights_json": t.user_attrs.get("weights_json", ""),
                "pbo_penalty": penalties["pbo_penalty"],
                "dsr_penalty": penalties["dsr_penalty"],
                "cpcv_penalty": penalties["cpcv_penalty"],
                **metadata,
            }
        )
    return pd.DataFrame(rows)


def build_report(outdir: Path, trial_ledger: pd.DataFrame, best: pd.DataFrame, optuna_status: str) -> None:
    lines = [
        "# TSM Auto Research Report",
        "",
        f"- Generated at UTC: {now_utc_iso()}",
        f"- Optuna status: {optuna_status}",
        f"- Trial rows: {len(trial_ledger)}",
        "",
        "## Objective",
        "- Reward Brier improvement and conservative selected-event uplift.",
        "- Penalize ECE, threshold IQR, fold selection drift, PBO, DSR failure, and CPCV weak-quartile failure.",
        "- Deep models are intentionally excluded until the effective event sample is much larger.",
        "",
        "## Best Candidates",
        "",
        "| Rank | Model family | Model | Scope | Split | Objective |",
        "|---:|---|---|---|---|---:|",
    ]
    if best.empty:
        lines.append("| NA | NA | NA | NA | NA | NA |")
    else:
        for rank, (_, row) in enumerate(best.head(10).iterrows(), start=1):
            lines.append(
                f"| {rank} | {row.get('model_family', '')} | {row.get('model_name', '')} | "
                f"{row.get('candidate_scope', '')} | {row.get('split', '')} | {as_float(row.get('objective_score')):.4f} |"
            )
    lines.extend(["", "This report is research tooling, not investment advice."])
    (outdir / "tsm_auto_research_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build automatic research trial ledger.")
    parser.add_argument("--pooled-comparison", default="tsm_price_rule_output/tsm_pooled_model_comparison.csv")
    parser.add_argument("--local-comparison", default="tsm_price_rule_output/tsm_prediction_model_comparison.csv")
    parser.add_argument("--pbo-report", default="tsm_price_rule_output/tsm_pbo_report.csv")
    parser.add_argument("--dsr-report", default="tsm_price_rule_output/tsm_deflated_sharpe_report.csv")
    parser.add_argument("--cpcv-model-distribution", default="tsm_price_rule_output/tsm_cpcv_model_distribution.csv")
    parser.add_argument("--outdir", default="tsm_price_rule_output")
    parser.add_argument("--storage", default="sqlite:///tsm_price_rule_output/tsm_research_trials.sqlite")
    parser.add_argument("--max-trials", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pooled_path = Path(args.pooled_comparison)
    local_path = Path(args.local_comparison)
    pooled = read_csv(pooled_path)
    local = read_csv(local_path)
    pbo = read_csv(Path(args.pbo_report))
    dsr = read_csv(Path(args.dsr_report))
    cpcv_model = read_csv(Path(args.cpcv_model_distribution))
    candidates = normalize_candidate_frame(pooled, local)
    metadata = {
        "tested_at_utc": now_utc_iso(),
        "feature_set_hash": file_hash(outdir / "tsm_prediction_feature_contract.csv"),
        "label_config_hash": hash_text({"horizons": [5, 10, 20, 40, 60, 120], "stop_multiples": [1.5, 2.0, 2.5, 3.0]}),
        "split_config_hash": hash_text({"purged": True, "cpcv_groups": 6, "embargo_days": 20}),
        "data_snapshot_hash": hash_text({"pooled": file_hash(pooled_path), "local": file_hash(local_path)}),
    }
    if candidates.empty:
        trial_ledger = pd.DataFrame()
        best = pd.DataFrame()
        optuna_status = f"no_candidates; optuna_error={OPTUNA_IMPORT_ERROR}" if OPTUNA_IMPORT_ERROR else "no_candidates"
    else:
        penalties = validation_penalties(pbo, dsr, cpcv_model)
        if OPTUNA_IMPORT_ERROR is None:
            trial_ledger = optuna_trials(candidates, penalties, args.max_trials, args.storage, metadata)
            optuna_status = "enabled"
        else:
            trial_ledger = deterministic_trials(candidates, penalties, args.max_trials, metadata)
            optuna_status = f"fallback_sampler; optuna_error={OPTUNA_IMPORT_ERROR}"
        best = trial_ledger.sort_values("objective_score", ascending=False).head(25).reset_index(drop=True) if not trial_ledger.empty else pd.DataFrame()
    trial_ledger.to_csv(outdir / "tsm_auto_research_trial_ledger.csv", index=False)
    best.to_csv(outdir / "tsm_auto_research_best_candidates.csv", index=False)
    build_report(outdir, trial_ledger, best, optuna_status)
    print("completed: auto research ledger =", (outdir / "tsm_auto_research_trial_ledger.csv").resolve(), "rows=", len(trial_ledger))


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .io import check_row

try:
    import pandera.pandas as pa
except Exception:  # pragma: no cover - optional dependency in existing envs
    try:
        import pandera as pa
    except Exception:
        pa = None


FUTURE_FEATURE_PREFIXES = ("label_", "fwd_", "exit_", "next_")
FUTURE_FEATURE_SUBSTRINGS = ("forward", "future", "actual_return", "net_return", "gross_return", "r_multiple")


@dataclass(frozen=True)
class FeatureContract:
    column: str
    role: str
    available_at: str
    leakage_policy: str
    feature_group: str


def is_future_leakage_feature(column: str) -> bool:
    lower = column.lower()
    return lower.startswith(FUTURE_FEATURE_PREFIXES) or any(token in lower for token in FUTURE_FEATURE_SUBSTRINGS)


def validate_no_future_features(columns: Iterable[str]) -> list[str]:
    return [str(col) for col in columns if is_future_leakage_feature(str(col))]


def _required_schema(required: list[str]):
    if pa is None:
        return None
    return pa.DataFrameSchema({col: pa.Column(nullable=True, required=True) for col in required}, coerce=False, strict=False)


def _validate_required(frame: pd.DataFrame, required: list[str], name: str, strict: bool) -> dict:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        if strict:
            raise ValueError(f"{name} missing required columns: {missing}")
        return check_row(f"schema_{name}_required_columns", False, "CRITICAL", ",".join(missing), details="missing required columns")
    schema = _required_schema(required)
    if schema is not None:
        try:
            schema.validate(frame, lazy=True)
        except Exception as exc:
            if strict:
                raise
            return check_row(f"schema_{name}_required_columns", False, "CRITICAL", "pandera_validation_failed", details=str(exc)[:500])
    return check_row(f"schema_{name}_required_columns", True, "CRITICAL", "ok")


def build_schema_quality_checks(
    *,
    labels: pd.DataFrame | None = None,
    feature_matrix: pd.DataFrame | None = None,
    comparison: pd.DataFrame | None = None,
    latest_snapshot: pd.DataFrame | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    rows = []
    if labels is not None:
        rows.append(_validate_required(labels, ["date", "signal_idx", "prediction_universe"], "prediction_labels", strict))
    if feature_matrix is not None:
        rows.append(_validate_required(feature_matrix, ["date", "signal_idx", "is_event_candidate"], "prediction_features", strict))
        model_input_cols = [
            c
            for c in feature_matrix.columns
            if c
            not in {
                "date",
                "signal_idx",
                "is_event_candidate",
                "is_actionable_entry_candidate",
                "is_trade_ready_entry_candidate",
                "is_entry_research_candidate",
                "is_risk_research_candidate",
                "is_model_training_candidate",
                "is_decision_entry_candidate",
            }
            and not str(c).startswith("label_")
        ]
        leaks = validate_no_future_features(model_input_cols)
        passed = not leaks
        if strict and not passed:
            raise ValueError(f"future/leakage feature columns are not allowed in model inputs: {leaks}")
        rows.append(check_row("schema_model_inputs_block_future_features", passed, "CRITICAL", ",".join(leaks) if leaks else "ok"))
    if comparison is not None:
        rows.append(_validate_required(comparison, ["candidate_scope", "horizon_days", "model_name", "prediction_quality_pass"], "model_comparison", strict))
    if latest_snapshot is not None:
        rows.append(_validate_required(latest_snapshot, ["field", "value"], "latest_prediction_snapshot", strict))
    rows.append(
        check_row(
            "pandera_available",
            pa is not None,
            "WARN",
            "ok" if pa is not None else "missing",
            details="Install dev dependency pandera for full schema diagnostics.",
        )
    )
    return pd.DataFrame(rows)

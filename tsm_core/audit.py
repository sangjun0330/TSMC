from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ModelGateResult:
    passed: bool
    use_status: str
    block_reasons: str


def file_sha256(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    payload = frame.sort_index(axis=1).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_policy_audit_rows(prediction_policy: str, latest: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    latest_map = dict(zip(latest["field"], latest["value"])) if not latest.empty else {}
    rows = [
        {
            "audit_item": "prediction_policy",
            "status": prediction_policy,
            "details": "conservative blocks rule-filtered rows from decision support",
        },
        {
            "audit_item": "latest_use_status",
            "status": latest_map.get("prediction_use_status", "UNKNOWN"),
            "details": latest_map.get("model_quality_block_reasons", ""),
        },
        {
            "audit_item": "decision_candidate_count",
            "status": int(comparison["prediction_quality_pass"].sum()) if "prediction_quality_pass" in comparison.columns else 0,
            "details": "number of model rows passing decision-support gates",
        },
    ]
    return pd.DataFrame(rows)

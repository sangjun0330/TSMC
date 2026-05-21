from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lstrip("\ufeff") for c in out.columns]
    return out


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return strip_bom_columns(pd.read_csv(path, **kwargs))


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def to_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def as_float(value, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def check_row(check: str, passed: bool, severity: str, value, tolerance: str = "", details: str = "") -> dict:
    return {
        "check": check,
        "passed": bool(passed),
        "severity": severity,
        "value": value,
        "tolerance": tolerance,
        "details": details,
    }

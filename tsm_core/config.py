from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass(frozen=True)
class CostConfig:
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    stop_multiple: float = 2.0


@dataclass(frozen=True)
class PredictionConfig:
    policy: str = "conservative"
    train_days: int = 756
    validation_days: int = 63
    test_days: int = 126
    step_days: int = 63
    gap_days: int = 20
    thresholds: str = "0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70"
    default_threshold: float = 0.55
    min_validation_trades: int = 20
    calibration_bins: int = 5


@dataclass(frozen=True)
class ValidationConfig:
    schema_strict: bool = False
    baseline_score_threshold: float = 75.0


@dataclass(frozen=True)
class NewsConfig:
    enabled: bool = True
    direct_web_only: bool = True
    sources: str = "tsmc_press,tsmc_investor,google_news_rss,yahoo_finance"
    daily_lookback_days: int = 14
    backfill_start: str = "2016-05-12"
    politeness_delay_sec: float = 1.0
    market_timezone: str = "America/New_York"
    schedule_local_time: str = "07:30"


@dataclass(frozen=True)
class RunConfig:
    cost: CostConfig = field(default_factory=CostConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    news: NewsConfig = field(default_factory=NewsConfig)


def _coerce_dataclass(cls, values: dict[str, Any]):
    names = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in values.items() if k in names})


def load_run_config(path: str | Path | None) -> RunConfig:
    if not path:
        return RunConfig()
    config_path = Path(path)
    if not config_path.exists():
        return RunConfig()
    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)
    return RunConfig(
        cost=_coerce_dataclass(CostConfig, raw.get("cost", {})),
        prediction=_coerce_dataclass(PredictionConfig, raw.get("prediction", {})),
        validation=_coerce_dataclass(ValidationConfig, raw.get("validation", {})),
        news=_coerce_dataclass(NewsConfig, raw.get("news", {})),
    )


def config_hash(config: RunConfig | dict[str, Any]) -> str:
    payload = asdict(config) if hasattr(config, "__dataclass_fields__") else config
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply_config_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser, config: RunConfig) -> argparse.Namespace:
    mapping = {
        "commission_bps": config.cost.commission_bps,
        "slippage_bps": config.cost.slippage_bps,
        "stop_multiple": config.cost.stop_multiple,
        "train_days": config.prediction.train_days,
        "validation_days": config.prediction.validation_days,
        "test_days": config.prediction.test_days,
        "step_days": config.prediction.step_days,
        "gap_days": config.prediction.gap_days,
        "thresholds": config.prediction.thresholds,
        "default_threshold": config.prediction.default_threshold,
        "min_validation_trades": config.prediction.min_validation_trades,
        "calibration_bins": config.prediction.calibration_bins,
        "prediction_policy": config.prediction.policy,
        "schema_strict": config.validation.schema_strict,
        "score_threshold": config.validation.baseline_score_threshold,
        "news_enabled": config.news.enabled,
        "news_sources": config.news.sources,
        "news_lookback_days": config.news.daily_lookback_days,
        "news_backfill_start": config.news.backfill_start,
        "news_politeness_delay_sec": config.news.politeness_delay_sec,
        "news_market_timezone": config.news.market_timezone,
    }
    for attr, value in mapping.items():
        if not hasattr(args, attr):
            continue
        try:
            default = parser.get_default(attr)
        except Exception:
            default = None
        current = getattr(args, attr)
        if current == default:
            setattr(args, attr, value)
    return args

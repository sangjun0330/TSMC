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
    external_features: str = "tsm_price_rule_output/tsm_external_daily_features.csv"
    intraday_features: str = "tsm_price_rule_output/tsm_intraday_daily_features.csv"
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
class RuleEngineConfig:
    score_entry_threshold: float = 75.0
    watchlist_threshold: float = 65.0
    observation_threshold: float = 60.0
    stop_atr_multiple: float = 2.0
    take_profit_r_multiple: float = 2.0
    max_weight_normal: float = 0.12
    max_weight_high_vol: float = 0.07
    max_weight_extreme_vol: float = 0.04
    max_weight_below_200d: float = 0.03
    signal_version: str = "tsm_rule_v1"
    rule_family: str = "price_momentum_risk"
    hypothesis_id: str = "top10_price_momentum_rule_v1"
    decision_policy_version: str = "decision_policy_v1"


@dataclass(frozen=True)
class PaperOmsConfig:
    enabled: bool = True
    account_equity: float = 100000.0
    default_order_type: str = "MARKET"
    time_in_force: str = "DAY"
    max_slippage_bps: float = 25.0
    kill_switch_active: bool = False


@dataclass(frozen=True)
class PaperExecutionConfig:
    fill_mode: str = "next_open_with_spread"
    half_spread_bps: float = 2.0
    volatility_bps_multiplier: float = 10.0
    participation_bps_multiplier: float = 5.0
    gap_penalty_bps: float = 3.0
    event_day_penalty_bps: float = 5.0
    default_adv: float = 1000000000.0


@dataclass(frozen=True)
class PortfolioRiskConfig:
    max_single_name_weight: float = 0.10
    max_semiconductor_weight: float = 0.35
    max_beta_to_spy: float = 1.20
    max_beta_to_smh: float = 0.80
    smh_beta_limit_mode: str = "WARN"
    max_daily_loss: float = -0.02
    max_weekly_loss: float = -0.05
    max_order_adv_pct: float = 0.01


@dataclass(frozen=True)
class UniverseConfig:
    enabled: bool = True
    default_universe_config: str = "config/semiconductor_universe_top10.csv"
    decision_universe_config: str = "config/semiconductor_universe_top10.csv"
    research_universe_config: str = "config/semiconductor_universe_expanded.csv"
    required_decision_symbol_count: int = 12
    decision_scope: str = "top10"
    training_scope: str = "universal_research_pool"


@dataclass(frozen=True)
class PortfolioConstructionConfig:
    max_open_positions: int = 5
    max_total_gross_weight: float = 0.35
    max_single_name_weight: float = 0.10
    max_group_weight: float = 0.35
    allocation_policy: str = "rank_cap"


@dataclass(frozen=True)
class PaperFeedbackConfig:
    horizon_days: int = 20
    lambda_mae: float = 0.25
    lambda_slippage: float = 0.01
    lambda_unfilled: float = 0.50
    stop_multiple: float = 2.0
    target_r_multiple: float = 2.0
    min_matured_events_for_calibration: int = 30


@dataclass(frozen=True)
class AutomationConfig:
    enabled: bool = True
    timezone: str = "America/New_York"
    post_close_local_time: str = "17:30"
    premarket_local_time: str = "08:45"
    open_reconciliation_local_time: str = "09:45"
    eod_feedback_local_time: str = "18:15"
    weekly_review_day: str = "FRI"
    monthly_promotion_review_day: int = 1


@dataclass(frozen=True)
class RunConfig:
    cost: CostConfig = field(default_factory=CostConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    rule_engine: RuleEngineConfig = field(default_factory=RuleEngineConfig)
    paper_oms: PaperOmsConfig = field(default_factory=PaperOmsConfig)
    paper_execution: PaperExecutionConfig = field(default_factory=PaperExecutionConfig)
    portfolio_risk: PortfolioRiskConfig = field(default_factory=PortfolioRiskConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    portfolio_construction: PortfolioConstructionConfig = field(default_factory=PortfolioConstructionConfig)
    paper_feedback: PaperFeedbackConfig = field(default_factory=PaperFeedbackConfig)
    automation: AutomationConfig = field(default_factory=AutomationConfig)


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
        rule_engine=_coerce_dataclass(RuleEngineConfig, raw.get("rule_engine", {})),
        paper_oms=_coerce_dataclass(PaperOmsConfig, raw.get("paper_oms", {})),
        paper_execution=_coerce_dataclass(PaperExecutionConfig, raw.get("paper_execution", {})),
        portfolio_risk=_coerce_dataclass(PortfolioRiskConfig, raw.get("portfolio_risk", {})),
        universe=_coerce_dataclass(UniverseConfig, raw.get("universe", {})),
        portfolio_construction=_coerce_dataclass(PortfolioConstructionConfig, raw.get("portfolio_construction", {})),
        paper_feedback=_coerce_dataclass(PaperFeedbackConfig, raw.get("paper_feedback", {})),
        automation=_coerce_dataclass(AutomationConfig, raw.get("automation", {})),
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
        "external_features": config.prediction.external_features,
        "intraday_features": config.prediction.intraday_features,
        "prediction_policy": config.prediction.policy,
        "schema_strict": config.validation.schema_strict,
        "score_threshold": config.validation.baseline_score_threshold,
        "universe_config": config.universe.default_universe_config,
        "decision_universe_config": config.universe.decision_universe_config,
        "research_universe_config": config.universe.research_universe_config,
        "max_open_positions": config.portfolio_construction.max_open_positions,
        "max_total_gross_weight": config.portfolio_construction.max_total_gross_weight,
        "max_single_name_weight": config.portfolio_construction.max_single_name_weight,
        "max_group_weight": config.portfolio_construction.max_group_weight,
        "allocation_policy": config.portfolio_construction.allocation_policy,
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

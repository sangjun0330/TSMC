from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ExecutionBlockReason(str, Enum):
    LIVE_TRADING_DISABLED = "LIVE_TRADING_DISABLED"
    STALE_DATA = "STALE_DATA"
    DATA_NOT_FRESH = "DATA_NOT_FRESH"
    MODEL_NOT_TRUSTED = "MODEL_NOT_TRUSTED"
    LOW_EXPECTANCY = "LOW_EXPECTANCY"
    STOP_RISK_TOO_HIGH = "STOP_RISK_TOO_HIGH"
    RISK_LIMIT_EXCEEDED = "RISK_LIMIT_EXCEEDED"
    MISSING_BROKER = "MISSING_BROKER"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"
    PAPER_OMS_DISABLED = "PAPER_OMS_DISABLED"
    DUPLICATE_INTENT = "DUPLICATE_INTENT"
    ORDER_CAPACITY_EXCEEDED = "ORDER_CAPACITY_EXCEEDED"
    PAPER_FILL_UNAVAILABLE = "PAPER_FILL_UNAVAILABLE"
    POSITION_MISMATCH = "POSITION_MISMATCH"


class IntentStatus(str, Enum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PAPER_SUBMITTED = "PAPER_SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    BLOCKED = "BLOCKED"


class OrderStatus(str, Enum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PAPER_SUBMITTED = "PAPER_SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    BLOCKED = "BLOCKED"


class FillStatus(str, Enum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PAPER_SUBMITTED = "PAPER_SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    EXPIRED = "EXPIRED"
    CANCELED = "CANCELED"
    BLOCKED = "BLOCKED"


class OrderLifecycleState(str, Enum):
    MISSING_INTENT = "MISSING_INTENT"
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    RISK_REJECTED = "RISK_REJECTED"
    PAPER_SUBMITTED = "PAPER_SUBMITTED"
    AWAITING_FILL = "AWAITING_FILL"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    EXPIRED_UNFILLED = "EXPIRED_UNFILLED"
    CANCELED = "CANCELED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    RECONCILED = "RECONCILED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class OrderIntent:
    intent_id: str = ""
    signal_id: str = ""
    signal_version: str = "tsm_rule_v1"
    rule_family: str = "price_momentum_risk"
    hypothesis_id: str = "top10_price_momentum_rule_v1"
    model_version: str = ""
    risk_policy_version: str = "risk_policy_v1"
    symbol: str = ""
    asof_date: str = ""
    side: str = ""
    order_type: str = ""
    target_weight: float = 0.0
    max_notional: float = 0.0
    entry_reference_price: float = 0.0
    stop_price: float = 0.0
    limit_price: float | None = None
    time_in_force: str = "DAY"
    max_slippage_bps: float = 15.0
    strategy_id: str = ""
    decision_source: str = ""
    available_at_utc: str = ""
    created_at_utc: str = ""
    status: str = IntentStatus.PROPOSED.value
    reason: str = ""
    live_trading_status: str = "DISABLED_BY_DESIGN"
    live_order_blocked: bool = True
    live_block_reason: str = ExecutionBlockReason.LIVE_TRADING_DISABLED.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperOrder:
    order_id: str
    intent_id: str
    symbol: str
    side: str
    order_type: str
    status: str
    target_weight: float
    target_notional: float
    quantity: float
    reference_price: float
    limit_price: float | None
    stop_price: float | None
    time_in_force: str
    submitted_at_utc: str
    signal_asof_date: str
    expected_fill_date: str
    fill_model: str
    live_trading_status: str = "DISABLED_BY_DESIGN"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperFill:
    fill_id: str
    order_id: str
    intent_id: str
    symbol: str
    side: str
    fill_type: str
    status: str
    fill_date: str
    fill_price: float
    raw_price: float
    quantity: float
    gross_notional: float
    commission_bps: float
    slippage_bps: float
    total_cost_bps: float
    reason: str
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperPosition:
    symbol: str
    asof_date: str
    quantity: float
    average_price: float
    market_price: float
    market_value: float
    cash: float
    equity: float
    weight: float
    realized_pnl: float
    unrealized_pnl: float
    position_state: str
    updated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperReconciliation:
    symbol: str
    asof_date: str
    internal_quantity: float
    recomputed_quantity: float
    quantity_diff: float
    internal_market_value: float
    recomputed_market_value: float
    market_value_diff: float
    mismatch_count: int
    status: str
    details: str
    checked_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperOmsQualityCheck:
    check: str
    passed: bool
    severity: str
    value: Any
    tolerance: str = ""
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskCheckResult:
    passed: bool
    target_weight: float
    max_allowed_weight: float
    block_reason: ExecutionBlockReason | str
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["block_reason"] = str(self.block_reason.value if isinstance(self.block_reason, ExecutionBlockReason) else self.block_reason)
        return out


@dataclass(frozen=True)
class OrderStateEvent:
    state_event_id: str
    intent_id: str
    order_id: str
    symbol: str
    asof_date: str
    lifecycle_state: str
    previous_state: str
    intent_status: str
    portfolio_status: str
    order_status: str
    fill_status: str
    reconciliation_status: str
    block_reason: str
    live_trading_status: str
    event_created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionFeedbackEvent:
    feedback_id: str
    intent_id: str
    order_id: str
    fill_id: str
    symbol: str
    signal_asof_date: str
    fill_date: str
    horizon_days: int
    order_status: str
    fill_status: str
    fill_model: str
    filled_flag: bool
    partial_fill_flag: bool
    expired_unfilled_flag: bool
    raw_price: float
    fill_price: float
    expected_slippage_bps: float
    realized_slippage_bps: float
    slippage_error_bps: float
    post_fill_return_pct: float
    mfe_pct: float
    mae_pct: float
    stop_hit: bool
    target_hit: bool
    label_matured: bool
    execution_adjusted_utility: float
    live_trading_status: str
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FillModelCalibration:
    calibration_id: str
    fill_model: str
    event_count: int
    matured_event_count: int
    fill_rate: float
    limit_expiry_rate: float
    mean_realized_slippage_bps: float
    median_realized_slippage_bps: float
    p90_realized_slippage_bps: float
    mean_slippage_error_bps: float
    suggested_half_spread_bps: float
    suggested_gap_penalty_bps: float
    suggested_event_day_penalty_bps: float
    calibration_status: str
    live_trading_status: str
    calibrated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

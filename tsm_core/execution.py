from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ExecutionBlockReason(str, Enum):
    LIVE_TRADING_DISABLED = "LIVE_TRADING_DISABLED"
    STALE_DATA = "STALE_DATA"
    MODEL_NOT_TRUSTED = "MODEL_NOT_TRUSTED"
    LOW_EXPECTANCY = "LOW_EXPECTANCY"
    STOP_RISK_TOO_HIGH = "STOP_RISK_TOO_HIGH"
    RISK_LIMIT_EXCEEDED = "RISK_LIMIT_EXCEEDED"
    MISSING_BROKER = "MISSING_BROKER"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    asof_date: str
    side: str
    order_type: str
    target_weight: float
    entry_reference_price: float
    stop_price: float
    time_in_force: str
    strategy_id: str
    decision_source: str
    live_trading_status: str = "DISABLED_BY_DESIGN"

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

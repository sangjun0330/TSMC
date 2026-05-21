# Execution Contract

Live trading remains disabled by design. The current execution contract is only
for research and paper-trading handoff.

## Flow

Signal Engine -> Order Intent -> Risk Check -> Blocked Execution Audit

Broker order submission, fill listeners, reconciliation, and kill-switch
automation are intentionally not implemented in this phase.

## Required Objects

- `OrderIntent`: symbol, as-of date, side, order type, target weight, reference entry price, stop price, time in force, strategy id, and decision source.
- `RiskCheckResult`: pass/fail, requested target weight, max allowed weight, block reason, and details.
- `ExecutionBlockReason`: controlled reason codes for stale data, untrusted model, low expectancy, stop risk, risk limit, missing broker, reconciliation, kill switch, and live-trading disabled.

All live order paths must continue to emit `DISABLED_BY_DESIGN` until broker
integration, fill reconciliation, stale-data checks, and kill switch behavior
exist and are tested.

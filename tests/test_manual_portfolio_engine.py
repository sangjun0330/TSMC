from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tsm_manual_portfolio_engine import (
    account_equity_from,
    build_rebalance_tickets,
    price_to_usd,
    run_manual_portfolio,
    transactions_to_fills,
)


def _fx() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-06-02"]),
            "fx_pair": ["KRW=X"] * 3,
            "fx_rate_to_usd": [0.00070, 0.00069, 0.00066],
            "usdkrw": [1428.57, 1449.27, 1515.15],
        }
    )


def _write_signals(root: Path, symbol: str, closes: list[float], *, trade_action: str = "HOLD", entry: str = "NONE") -> None:
    rule_dir = root / "tsm_price_rule_output" / "universe" / symbol
    rule_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2026-05-25", periods=len(closes), freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "atr_stop_2x": [c * 0.9 for c in closes],
            "take_profit_2R": [c * 1.2 for c in closes],
            "trade_action": trade_action,
            "entry_trigger": entry,
            "suggested_weight": 0.05,
        }
    )
    frame.to_csv(rule_dir / "tsm_daily_algorithmic_signals.csv", index=False)


def _write_portfolio_decisions(root: Path, rows: list[dict]) -> None:
    rule_dir = root / "tsm_price_rule_output"
    rule_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(rule_dir / "tsm_portfolio_risk_order_decisions.csv", index=False)


def _write_fx(root: Path) -> None:
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    _fx().to_csv(out / "tsm_fx_rates_daily.csv", index=False)


def _write_transactions(root: Path, rows: list[dict]) -> Path:
    outdir = root / "output" / "portfolio"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "manual_transactions.csv", index=False)
    return outdir


def test_price_to_usd_krw_conversion():
    fx = _fx()
    # KRW price on 2026-04-01 uses fx_rate_to_usd 0.00070
    usd, rate = price_to_usd(100000.0, "KRW", "005930.KS", "2026-04-01", fx)
    assert rate == pytest.approx(0.00070)
    assert usd == pytest.approx(70.0)
    # USD stays USD
    usd2, rate2 = price_to_usd(150.0, "USD", "LRCX", "2026-04-01", fx)
    assert usd2 == pytest.approx(150.0) and rate2 == pytest.approx(1.0)
    # NATIVE resolves to KRW for a .KS symbol
    usd3, _ = price_to_usd(100000.0, "NATIVE", "000660.KS", "2026-04-02", fx)
    assert usd3 == pytest.approx(69.0)


def test_average_cost_and_market_value(tmp_path):
    root = tmp_path
    _write_fx(root)
    _write_signals(root, "LRCX", [300.0, 300.0])  # latest close 300
    tx = [
        dict(transaction_id="a", trade_date="2026-04-01", symbol="LRCX", side="BUY", quantity=2, price=100.0, price_currency="USD", fees=0, note=""),
        dict(transaction_id="b", trade_date="2026-04-02", symbol="LRCX", side="BUY", quantity=2, price=200.0, price_currency="USD", fees=0, note=""),
    ]
    _write_transactions(root, tx)
    res = run_manual_portfolio(root, Path("config/tsm_research.toml"), root / "output" / "portfolio")
    pos = res["positions"]
    lrcx = pos[pos["symbol"] == "LRCX"].iloc[0]
    assert lrcx["quantity"] == pytest.approx(4.0)
    assert lrcx["average_price"] == pytest.approx(150.0)  # (200 + 400) / 4
    assert lrcx["market_value"] == pytest.approx(1200.0)  # 4 * 300
    assert lrcx["unrealized_pnl"] == pytest.approx(600.0)  # 4 * (300 - 150)
    # fully invested: weight ~ 1.0
    assert lrcx["weight"] == pytest.approx(1.0, abs=1e-6)


def test_partial_sell_realized_pnl(tmp_path):
    root = tmp_path
    _write_fx(root)
    _write_signals(root, "MU", [400.0])
    tx = [
        dict(transaction_id="a", trade_date="2026-04-01", symbol="MU", side="BUY", quantity=4, price=150.0, price_currency="USD", fees=0, note=""),
        dict(transaction_id="b", trade_date="2026-04-02", symbol="MU", side="SELL", quantity=1, price=200.0, price_currency="USD", fees=0, note=""),
    ]
    _write_transactions(root, tx)
    res = run_manual_portfolio(root, Path("config/tsm_research.toml"), root / "output" / "portfolio")
    pos = res["positions"]
    mu = pos[pos["symbol"] == "MU"].iloc[0]
    assert mu["quantity"] == pytest.approx(3.0)
    assert mu["average_price"] == pytest.approx(150.0)
    assert mu["realized_pnl"] == pytest.approx(50.0)  # 1 * (200 - 150)


def test_krw_transaction_becomes_usd_fill(tmp_path):
    root = tmp_path
    fx = _fx()
    tx = pd.DataFrame(
        [dict(transaction_id="k", trade_date="2026-04-01", symbol="005930.KS", side="BUY", quantity=1, price=100000.0, price_currency="KRW", fees=0, note="")]
    )
    fills = transactions_to_fills(tx, fx)
    assert len(fills) == 1
    assert fills.iloc[0]["fill_price"] == pytest.approx(70.0)  # 100000 * 0.00070
    assert fills.iloc[0]["raw_price"] == pytest.approx(100000.0)
    # account equity (no cash ledger) = net invested notional in USD
    eq = account_equity_from(fills, pd.DataFrame(), fx)
    assert eq == pytest.approx(70.0)


def test_rebalance_tickets_trim_buynew_exit(tmp_path):
    root = tmp_path
    _write_fx(root)
    # Held LRCX heavily overweight vs a small model target -> TRIM
    _write_signals(root, "LRCX", [300.0], trade_action="HOLD", entry="60D_BREAKOUT")
    # AMD not held but model target positive -> BUY_NEW
    _write_signals(root, "AMD", [120.0], trade_action="ENTRY_ALLOWED", entry="BREAKOUT_20D")
    # TSM held and model says SELL -> EXIT
    _write_signals(root, "TSM", [200.0], trade_action="SELL", entry="NONE")
    _write_portfolio_decisions(
        root,
        [
            dict(symbol="LRCX", asof_date="2026-06-02", approved_weight=0.05, p_success_20d=0.6, block_reason="PASS", warning_reasons="PASS", stop_price_1_8atr=270.0),
            dict(symbol="AMD", asof_date="2026-06-02", approved_weight=0.08, p_success_20d=0.7, block_reason="PASS", warning_reasons="PASS", stop_price_1_8atr=108.0),
            dict(symbol="TSM", asof_date="2026-06-02", approved_weight=0.0, p_success_20d=0.2, block_reason="DEEP_DOWNTREND", warning_reasons="PASS", stop_price_1_8atr=180.0),
        ],
    )
    tx = [
        dict(transaction_id="l", trade_date="2026-04-01", symbol="LRCX", side="BUY", quantity=1, price=100.0, price_currency="USD", fees=0, note=""),
        dict(transaction_id="t", trade_date="2026-04-01", symbol="TSM", side="BUY", quantity=1, price=100.0, price_currency="USD", fees=0, note=""),
    ]
    _write_transactions(root, tx)
    res = run_manual_portfolio(root, Path("config/tsm_research.toml"), root / "output" / "portfolio")
    tickets = res["tickets"].set_index("symbol")

    assert tickets.loc["LRCX", "action"] == "TRIM"
    assert tickets.loc["LRCX", "delta_shares"] < 0

    assert tickets.loc["AMD", "action"] == "BUY_NEW"
    assert tickets.loc["AMD", "delta_shares"] > 0
    assert tickets.loc["AMD", "target_weight"] == pytest.approx(0.08)

    assert tickets.loc["TSM", "action"] == "EXIT"
    assert tickets.loc["TSM", "delta_shares"] == pytest.approx(-1.0)
    # safety: never live trading
    assert (res["tickets"]["live_trading_status"] == "DISABLED_BY_DESIGN").all()

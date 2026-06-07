from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from tsm_core.currency import currency_profile
from tsm_core.io import strip_bom_columns, to_bool


DEFAULT_UNIVERSE_CONFIG = "config/semiconductor_universe_top10.csv"
DEFAULT_DECISION_UNIVERSE_CONFIG = DEFAULT_UNIVERSE_CONFIG
DEFAULT_RESEARCH_UNIVERSE_CONFIG = "config/semiconductor_universe_expanded.csv"
DECISION_SCOPE_TOP10 = "top10"
TRAINING_SCOPE_UNIVERSAL_RESEARCH_POOL = "universal_research_pool"
REQUIRED_DECISION_SYMBOL_COUNT = 12
REQUIRED_UNIVERSE_COLUMNS = [
    "symbol",
    "symbol_group",
    "symbol_stooq",
    "symbol_yahoo",
    "data_outdir",
    "rule_outdir",
]
KOREAN_MARKET_SUFFIXES = (".KS", ".KQ")
MARKET_REGION_KR = "KR"
MARKET_REGION_OVERSEAS = "OVERSEAS"


@dataclass(frozen=True)
class UniversePaths:
    data_outdir: Path
    rule_outdir: Path
    enriched: Path
    raw: Path
    summary: Path
    events: Path
    signals: Path
    risk_policy: Path
    risk_snapshot: Path
    trade_log: Path


@dataclass(frozen=True)
class UniverseMember:
    symbol: str
    symbol_group: str
    symbol_stooq: str
    symbol_yahoo: str
    data_outdir: Path
    rule_outdir: Path
    listing_currency: str = "USD"
    display_currency: str = "USD"
    engine_currency: str = "USD"
    fx_pair: str = ""
    market_region: str = MARKET_REGION_OVERSEAS
    enabled: bool = True
    paper_enabled: bool = True
    strict_eligible: str = ""
    eligibility_status: str = ""

    @property
    def paths(self) -> UniversePaths:
        return build_universe_paths(self.data_outdir, self.rule_outdir)


def build_universe_paths(data_outdir: str | Path, rule_outdir: str | Path) -> UniversePaths:
    data_path = Path(data_outdir)
    rule_path = Path(rule_outdir)
    return UniversePaths(
        data_outdir=data_path,
        rule_outdir=rule_path,
        enriched=data_path / "tsm_daily_10y_enriched.csv",
        raw=data_path / "tsm_daily_10y_raw.csv",
        summary=data_path / "tsm_daily_10y_summary.csv",
        events=data_path / "tsm_event_impact_10y.csv",
        signals=rule_path / "tsm_daily_algorithmic_signals.csv",
        risk_policy=rule_path / "tsm_risk_policy_daily.csv",
        risk_snapshot=rule_path / "tsm_latest_risk_snapshot.csv",
        trade_log=rule_path / "tsm_backtest_trade_log.csv",
    )


def default_tsm_member(output_dir: str | Path = "output", rule_outdir: str | Path = "tsm_price_rule_output") -> UniverseMember:
    return UniverseMember(
        symbol="TSM",
        symbol_group="semiconductor",
        symbol_stooq="TSM.US",
        symbol_yahoo="TSM",
        data_outdir=Path(output_dir),
        rule_outdir=Path(rule_outdir),
        listing_currency="USD",
        display_currency="USD",
        engine_currency="USD",
        fx_pair="",
        market_region=MARKET_REGION_OVERSEAS,
        enabled=True,
        paper_enabled=True,
        strict_eligible="",
        eligibility_status="TSM_COMPAT_FALLBACK",
    )


def _text(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "n/a"}:
        return default
    return text


def _path(value: object, default: str | Path) -> Path:
    return Path(_text(value, str(default)))


def is_korean_symbol(symbol: object = "", symbol_yahoo: object = "") -> bool:
    values = [str(symbol or "").strip().upper(), str(symbol_yahoo or "").strip().upper()]
    return any(value.endswith(KOREAN_MARKET_SUFFIXES) for value in values)


def market_region_for_symbol(symbol: object = "", symbol_yahoo: object = "", market_region: object = "") -> str:
    configured = _text(market_region).upper()
    if configured in {MARKET_REGION_KR, MARKET_REGION_OVERSEAS}:
        return configured
    return MARKET_REGION_KR if is_korean_symbol(symbol, symbol_yahoo) else MARKET_REGION_OVERSEAS


def read_universe_config(path: str | Path = DEFAULT_UNIVERSE_CONFIG) -> pd.DataFrame:
    config_path = Path(path)
    if not config_path.exists():
        return pd.DataFrame([member_to_config_row(default_tsm_member())])
    config = strip_bom_columns(pd.read_csv(config_path))
    missing = [col for col in REQUIRED_UNIVERSE_COLUMNS if col not in config.columns]
    if missing:
        raise ValueError(f"universe config missing columns: {missing}")
    config = config.copy()
    config["symbol"] = config["symbol"].astype(str).str.strip().str.upper()
    config["symbol_group"] = config["symbol_group"].fillna("semiconductor").astype(str).str.strip()
    if "enabled" not in config.columns:
        config["enabled"] = True
    if "paper_enabled" not in config.columns:
        config["paper_enabled"] = True
    for col in ["listing_currency", "display_currency", "engine_currency", "fx_pair"]:
        if col not in config.columns:
            config[col] = ""
    if "market_region" not in config.columns:
        config["market_region"] = ""
    profiles = [
        currency_profile(
            row.get("symbol"),
            row.get("symbol_yahoo"),
            listing_currency=row.get("listing_currency"),
            display_currency=row.get("display_currency"),
            engine_currency=row.get("engine_currency") or "USD",
            fx_pair=row.get("fx_pair"),
        )
        for _, row in config.iterrows()
    ]
    config["listing_currency"] = [profile.listing_currency for profile in profiles]
    config["display_currency"] = [profile.display_currency for profile in profiles]
    config["engine_currency"] = [profile.engine_currency for profile in profiles]
    config["fx_pair"] = [profile.fx_pair for profile in profiles]
    config["market_region"] = [
        market_region_for_symbol(row.get("symbol"), row.get("symbol_yahoo"), row.get("market_region"))
        for _, row in config.iterrows()
    ]
    return config


def member_to_config_row(member: UniverseMember) -> dict[str, object]:
    return {
        "symbol": member.symbol,
        "symbol_group": member.symbol_group,
        "symbol_stooq": member.symbol_stooq,
        "symbol_yahoo": member.symbol_yahoo,
        "data_outdir": str(member.data_outdir),
        "rule_outdir": str(member.rule_outdir),
        "listing_currency": member.listing_currency,
        "display_currency": member.display_currency,
        "engine_currency": member.engine_currency,
        "fx_pair": member.fx_pair,
        "market_region": member.market_region,
        "enabled": member.enabled,
        "paper_enabled": member.paper_enabled,
        "strict_eligible": member.strict_eligible,
        "eligibility_status": member.eligibility_status,
        "signals": str(member.paths.signals),
        "risk_policy": str(member.paths.risk_policy),
        "trade_log": str(member.paths.trade_log),
        "enriched": str(member.paths.enriched),
    }


def row_to_member(row: pd.Series) -> UniverseMember:
    symbol = _text(row.get("symbol"), "TSM").upper()
    profile = currency_profile(
        symbol,
        row.get("symbol_yahoo"),
        listing_currency=row.get("listing_currency"),
        display_currency=row.get("display_currency"),
        engine_currency=row.get("engine_currency") or "USD",
        fx_pair=row.get("fx_pair"),
    )
    return UniverseMember(
        symbol=symbol,
        symbol_group=_text(row.get("symbol_group"), "semiconductor"),
        symbol_stooq=_text(row.get("symbol_stooq"), f"{symbol}.US"),
        symbol_yahoo=_text(row.get("symbol_yahoo"), symbol),
        data_outdir=_path(row.get("data_outdir"), Path("output") / "universe" / symbol),
        rule_outdir=_path(row.get("rule_outdir"), Path("tsm_price_rule_output") / "universe" / symbol),
        listing_currency=profile.listing_currency,
        display_currency=profile.display_currency,
        engine_currency=profile.engine_currency,
        fx_pair=profile.fx_pair,
        market_region=market_region_for_symbol(symbol, row.get("symbol_yahoo"), row.get("market_region")),
        enabled=to_bool(row.get("enabled", True)),
        paper_enabled=to_bool(row.get("paper_enabled", True)),
        strict_eligible=_text(row.get("strict_eligible")),
        eligibility_status=_text(row.get("eligibility_status")),
    )


def load_universe_members(
    path: str | Path = DEFAULT_UNIVERSE_CONFIG,
    *,
    include_disabled: bool = False,
    paper_only: bool = False,
) -> list[UniverseMember]:
    config = read_universe_config(path)
    members = [row_to_member(row) for _, row in config.iterrows()]
    if not include_disabled:
        members = [member for member in members if member.enabled]
    if paper_only:
        members = [member for member in members if member.paper_enabled]
    return members


def symbol_set(members: Iterable[UniverseMember]) -> set[str]:
    return {member.symbol.upper() for member in members}


def load_decision_universe_members(
    path: str | Path = DEFAULT_DECISION_UNIVERSE_CONFIG,
    *,
    include_disabled: bool = False,
    paper_only: bool = False,
    require_count: bool = True,
) -> list[UniverseMember]:
    members = load_universe_members(path, include_disabled=include_disabled, paper_only=paper_only)
    if require_count and len(members) != REQUIRED_DECISION_SYMBOL_COUNT:
        raise ValueError(
            f"decision universe must contain exactly {REQUIRED_DECISION_SYMBOL_COUNT} enabled symbols; "
            f"got {len(members)} from {path}"
        )
    return members


def load_research_universe_members(
    decision_path: str | Path = DEFAULT_DECISION_UNIVERSE_CONFIG,
    research_path: str | Path = DEFAULT_RESEARCH_UNIVERSE_CONFIG,
    *,
    include_disabled: bool = False,
    paper_only: bool = False,
) -> list[UniverseMember]:
    decision_members = load_decision_universe_members(
        decision_path,
        include_disabled=include_disabled,
        paper_only=paper_only,
        require_count=False,
    )
    research_members = load_universe_members(research_path, include_disabled=include_disabled, paper_only=paper_only)
    merged: dict[str, UniverseMember] = {}
    for member in [*decision_members, *research_members]:
        merged.setdefault(member.symbol.upper(), member)
    return list(merged.values())


def add_scope_columns(
    frame: pd.DataFrame,
    decision_symbols: Iterable[str],
    *,
    decision_scope: str = DECISION_SCOPE_TOP10,
    training_scope: str = TRAINING_SCOPE_UNIVERSAL_RESEARCH_POOL,
) -> pd.DataFrame:
    out = frame.copy()
    decision_set = {str(symbol).upper() for symbol in decision_symbols}
    if "symbol" not in out.columns:
        out["symbol"] = ""
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["is_decision_universe"] = out["symbol"].isin(decision_set)
    out["decision_scope"] = decision_scope
    out["training_scope"] = training_scope
    return out


def filter_to_decision_symbols(frame: pd.DataFrame, decision_symbols: Iterable[str]) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns:
        return frame.copy()
    decision_set = {str(symbol).upper() for symbol in decision_symbols}
    out = frame.copy()
    return out[out["symbol"].astype(str).str.upper().isin(decision_set)].copy()


def build_universe_scope_audit(
    decision_members: Iterable[UniverseMember],
    research_members: Iterable[UniverseMember],
    *,
    decision_scope: str = DECISION_SCOPE_TOP10,
    training_scope: str = TRAINING_SCOPE_UNIVERSAL_RESEARCH_POOL,
) -> pd.DataFrame:
    decision_lookup = {member.symbol.upper(): member for member in decision_members}
    rows = []
    for member in research_members:
        symbol = member.symbol.upper()
        paths = member.paths
        rows.append(
            {
                "symbol": symbol,
                "symbol_group": member.symbol_group,
                "market_region": member.market_region,
                "is_decision_universe": symbol in decision_lookup,
                "decision_scope": decision_scope,
                "training_scope": training_scope,
                "enabled": member.enabled,
                "paper_enabled": member.paper_enabled,
                "daily_enriched_exists": paths.enriched.exists(),
                "signals_exists": paths.signals.exists(),
                "risk_policy_exists": paths.risk_policy.exists(),
                "trade_log_exists": paths.trade_log.exists(),
                "research_pool_status": "AVAILABLE_FOR_RESEARCH"
                if paths.enriched.exists() and paths.signals.exists() and paths.risk_policy.exists()
                else "MISSING_RESEARCH_INPUTS",
            }
        )
    return pd.DataFrame(rows)


def members_to_frame(members: Iterable[UniverseMember]) -> pd.DataFrame:
    return pd.DataFrame([member_to_config_row(member) for member in members])


def find_member(members: Iterable[UniverseMember], symbol: str) -> UniverseMember | None:
    symbol_upper = str(symbol).upper()
    for member in members:
        if member.symbol.upper() == symbol_upper:
            return member
    return None

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider selection and audit helpers for intraday market-data adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable


DEFAULT_PROVIDER_ORDER = ("alpha_vantage", "polygon", "eodhd", "alpaca", "yahoo")
SUPPORTED_PROVIDERS = (*DEFAULT_PROVIDER_ORDER, "auto")
PROVIDER_ENV_KEYS = {
    "alpha_vantage": ("ALPHAVANTAGE_API_KEY",),
    "polygon": ("POLYGON_API_KEY",),
    "eodhd": ("EODHD_API_KEY",),
    "alpaca": ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"),
    "yahoo": tuple(),
}
PROVIDER_DOC_URL = {
    "alpha_vantage": "https://www.alphavantage.co/documentation/",
    "polygon": "https://polygon.io/docs/flat-files/stocks/overview",
    "eodhd": "https://eodhd.com/financial-apis/intraday-historical-data-api/",
    "alpaca": "https://docs.alpaca.markets/us/reference/stockbars",
    "yahoo": "https://query1.finance.yahoo.com/v8/finance/chart/",
}
PROVIDER_CAPABILITY_NOTE = {
    "alpha_vantage": "TIME_SERIES_INTRADAY can provide deep historical intraday data when a premium key is configured.",
    "polygon": "Flat files provide U.S. equity minute aggregates when the account and plan include the requested history.",
    "eodhd": "Intraday historical API requires an API token and plan-specific historical availability.",
    "alpaca": "Historical stock bars endpoint supports minute/hour/day bars with API credentials and plan limits.",
    "yahoo": "Public chart endpoint requires no key but has short intraday history windows.",
}
IMPLEMENTED_PROVIDERS = {"yahoo"}


@dataclass(frozen=True)
class ProviderAttempt:
    provider: str
    interval: str
    auth_status: str
    request_status: str
    selected: bool
    fallback_used: bool
    supports_complete_requested_10y: object
    evidence: str
    source_url: str
    attempted_at_utc: str

    def to_row(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "interval": self.interval,
            "auth_status": self.auth_status,
            "request_status": self.request_status,
            "selected": self.selected,
            "fallback_used": self.fallback_used,
            "supports_complete_requested_10y": self.supports_complete_requested_10y,
            "actual_start": "",
            "actual_end": "",
            "rows": "",
            "evidence": self.evidence,
            "source_url": self.source_url,
            "attempted_at_utc": self.attempted_at_utc,
        }


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_provider_order(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_PROVIDER_ORDER
    if isinstance(value, str):
        raw = [part.strip().lower() for part in value.split(",")]
    else:
        raw = [str(part).strip().lower() for part in value]
    providers = [part for part in raw if part and part in DEFAULT_PROVIDER_ORDER]
    providers = list(dict.fromkeys(providers))
    return tuple(providers) if providers else DEFAULT_PROVIDER_ORDER


def provider_auth_status(provider: str) -> str:
    keys = PROVIDER_ENV_KEYS.get(provider, tuple())
    if not keys:
        return "no_key_required"
    present = [key for key in keys if os.getenv(key)]
    missing = [key for key in keys if not os.getenv(key)]
    if not missing:
        return "key_present"
    if present:
        return f"partial_key_present_missing:{','.join(missing)}"
    return f"missing:{','.join(keys)}"


def provider_has_credentials(provider: str) -> bool:
    keys = PROVIDER_ENV_KEYS.get(provider, tuple())
    return not keys or all(os.getenv(key) for key in keys)


def select_intraday_provider(
    requested_provider: str,
    interval: str,
    provider_order: str | Iterable[str] | None = None,
) -> tuple[str, list[ProviderAttempt]]:
    """Return the provider currently usable by this code path plus audit attempts.

    Paid-provider credentials are detected and recorded, but only Yahoo is
    implemented in this package today. Auto mode therefore selects Yahoo when
    no implemented paid adapter can serve the request.
    """

    requested = (requested_provider or "auto").lower()
    order = parse_provider_order(provider_order)
    candidates = order if requested == "auto" else (requested,)
    attempts: list[ProviderAttempt] = []
    selected = ""
    now = now_utc_iso()
    for provider in candidates:
        if provider not in DEFAULT_PROVIDER_ORDER:
            continue
        has_credentials = provider_has_credentials(provider)
        implemented = provider in IMPLEMENTED_PROVIDERS
        if implemented and has_credentials and not selected:
            selected = provider
            status = "selected"
        elif not has_credentials:
            status = "skipped_missing_credentials"
        elif not implemented:
            status = "skipped_adapter_not_implemented"
        else:
            status = "skipped_not_selected"
        attempts.append(
            ProviderAttempt(
                provider=provider,
                interval=interval,
                auth_status=provider_auth_status(provider),
                request_status=status,
                selected=status == "selected",
                fallback_used=False,
                supports_complete_requested_10y="provider_plan_dependent" if provider != "yahoo" else False,
                evidence=PROVIDER_CAPABILITY_NOTE.get(provider, ""),
                source_url=PROVIDER_DOC_URL.get(provider, ""),
                attempted_at_utc=now,
            )
        )

    if selected:
        return selected, attempts
    if requested != "auto":
        return requested, attempts
    fallback_attempt = ProviderAttempt(
        provider="yahoo",
        interval=interval,
        auth_status=provider_auth_status("yahoo"),
        request_status="selected_fallback",
        selected=True,
        fallback_used=True,
        supports_complete_requested_10y=False,
        evidence=PROVIDER_CAPABILITY_NOTE["yahoo"],
        source_url=PROVIDER_DOC_URL["yahoo"],
        attempted_at_utc=now,
    )
    attempts.append(fallback_attempt)
    return "yahoo", attempts

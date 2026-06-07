#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct-web news and cause matching engine for the semiconductor research pipeline.

This module intentionally avoids paid/free news APIs and LLM APIs. It collects
metadata from public web/RSS pages, keeps only short snippets, maps articles to
trading dates, clusters related headlines, and emits daily cause features for
the rule engine.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urljoin
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - optional fallback
    TfidfVectorizer = None
    cosine_similarity = None


TSMC_PRESS_ARCHIVE_URL = "https://pr.tsmc.com/english/news-archives"
TSMC_PRESS_LATEST_URL = "https://pr.tsmc.com/english"
YAHOO_FINANCE_QUOTE_URLS = {
    "yahoo_finance_tsm": "https://finance.yahoo.com/quote/TSM/",
    "yahoo_finance_samsung": "https://finance.yahoo.com/quote/005930.KS/",
    "yahoo_finance_sk_hynix": "https://finance.yahoo.com/quote/000660.KS/",
}
GOOGLE_NEWS_RSS_URL = (
    "https://news.google.com/rss/search?q="
    + quote_plus('TSMC OR "Taiwan Semiconductor" OR "TSM stock" OR "Samsung Electronics" OR "005930.KS" OR "SK hynix" OR "SK Hynix" OR "000660.KS" semiconductor')
    + "&hl=en-US&gl=US&ceid=US:en"
)
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; Semiconductor-News-Causal-Research/1.0)"
MARKET_CLOSE = dt_time(16, 0)


CAUSE_KEYWORDS: dict[str, list[str]] = {
    "earnings_results": ["earnings", "eps", "profit", "net income", "operating profit", "quarter results", "quarterly results", "preliminary earnings", "margin"],
    "guidance": ["guidance", "outlook", "forecast", "revenue view", "retains", "raises forecast", "cuts forecast", "memory outlook"],
    "monthly_revenue": ["monthly revenue", "revenue report", "november revenue", "december revenue", "january revenue", "february revenue", "march revenue", "april revenue", "may revenue", "june revenue", "july revenue", "august revenue", "september revenue", "october revenue"],
    "capex_fab_expansion": ["fab", "plant", "factory", "capacity", "capex", "expansion", "arizona", "japan", "germany", "new facility"],
    "ai_hpc_demand": ["ai demand", "ai chip", "ai chips", "artificial intelligence demand", "hpc", "hbm", "high bandwidth memory", "dram", "nvidia", "advanced packaging", "chip demand", "accelerator"],
    "customer_supply_chain": ["apple", "amd", "nvidia", "qualcomm", "broadcom", "sony", "samsung", "samsung electronics", "sk hynix", "hynix", "customer", "supply chain", "outsourcing", "intel delay", "foundry", "partnership"],
    "regulation_export_controls": ["export control", "bis", "commerce department", "sanction", "regulation", "chip curbs", "advanced computing"],
    "geopolitics_taiwan": ["china", "geopolitical", "geopolitics", "strait", "military", "tariff", "invasion", "war", "tension", "national security"],
    "operational_disruption": ["earthquake", "disruption", "virus", "incident", "shutdown", "power outage", "evacuation", "production halt"],
    "analyst_rating_target": ["analyst", "rating", "price target", "upgrade", "downgrade", "initiates", "maintains"],
    "dividend_capital_return": ["dividend", "buyback", "cash distribution", "capital return", "board of directors"],
    "peer_sector_move": ["semiconductor stocks", "chip stocks", "soxx", "smh", "sector", "samsung", "samsung electronics", "sk hynix", "hynix", "micron", "asml"],
    "macro_rates_fx": ["rate", "inflation", "fed", "dollar", "fx", "yen", "treasury", "macro"],
    "technical_market_move": [
        "why is",
        "stock moves",
        "shares rise",
        "shares fall",
        "market cap",
        "technical",
        "stocks",
        "valuation",
        "buy",
        "sell",
        "hold",
        "stake",
        "position",
        "trims",
        "boosts",
        "cuts stake",
    ],
}

CAUSE_IMPORTANCE = {
    "earnings_results": 20,
    "guidance": 19,
    "monthly_revenue": 18,
    "regulation_export_controls": 20,
    "operational_disruption": 20,
    "geopolitics_taiwan": 18,
    "ai_hpc_demand": 17,
    "customer_supply_chain": 16,
    "capex_fab_expansion": 14,
    "analyst_rating_target": 12,
    "dividend_capital_return": 10,
    "peer_sector_move": 9,
    "macro_rates_fx": 9,
    "technical_market_move": 6,
    "other": 4,
}

POSITIVE_TERMS = [
    "beat",
    "beats",
    "record",
    "rises",
    "rise",
    "surge",
    "surged",
    "strong",
    "growth",
    "increase",
    "increased",
    "raises",
    "upgrade",
    "optimism",
    "demand",
    "retains",
]
NEGATIVE_TERMS = [
    "fall",
    "falls",
    "fell",
    "drop",
    "drops",
    "slump",
    "slumps",
    "delay",
    "cut",
    "cuts",
    "downgrade",
    "weak",
    "decrease",
    "decreased",
    "disruption",
    "earthquake",
    "virus",
    "export control",
    "sanction",
]

NEWS_DAILY_COLUMNS = [
    "date",
    "news_event_count_1d",
    "news_event_count_3d",
    "news_sentiment_score_1d",
    "news_primary_cause_type",
    "news_primary_cluster_id",
    "news_match_confidence",
    "news_match_confidence_score",
    "news_coverage_status",
    "news_source_count",
    "news_primary_source_url",
    "news_cause_summary",
    "news_penalty_event",
]

NO_HIGH_CONFIDENCE_CAUSE = "NO_HIGH_CONFIDENCE_NEWS"
NO_CLUSTER = "NO_CLUSTER"
LOW_MATERIAL_PRIMARY_CAUSES = {"other", "technical_market_move"}
NEGATIVE_PENALTY_CAUSES = {
    "earnings_results",
    "guidance",
    "monthly_revenue",
    "capex_fab_expansion",
    "customer_supply_chain",
    "regulation_export_controls",
    "geopolitics_taiwan",
    "operational_disruption",
}
SEVERE_NEGATIVE_PENALTY_CAUSES = {"regulation_export_controls", "operational_disruption", "geopolitics_taiwan"}
INVESTMENT_OPINION_TERMS = [
    "better buy",
    "buy, sell, or hold",
    "buy sell or hold",
    "stock will",
    "ai stocks",
    "prediction:",
    "could be",
    "smartest",
    "long-term buy",
    "valuation",
    "shareholder returns",
    "billionaire",
    "loading up",
    "stake",
    "position",
    "trims",
    "boosts",
    "cuts stake",
]
CONCRETE_EVENT_HINTS = [
    "earnings",
    "eps",
    "guidance",
    "monthly revenue",
    "revenue report",
    "revenue growth",
    "demand",
    "orders",
    "hpc",
    "hbm",
    "high bandwidth memory",
    "dram",
    "memory chip",
    "nvidia",
    "samsung",
    "samsung electronics",
    "sk hynix",
    "hynix",
    "advanced packaging",
    "accelerator",
    "partnership",
    "customer",
    "outsourcing",
    "foundry",
    "export control",
    "sanction",
    "earthquake",
    "disruption",
    "dividend",
    "capex",
    "fab",
    "strait",
    "military",
    "tension",
    "invasion",
    "war",
    "national security",
]


@dataclass
class FetchResult:
    source_name: str
    url: str
    text: str
    ok: bool
    status: str
    error: str = ""


class AnchorExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.anchors: list[dict[str, str]] = []
        self._current_href: str | None = None
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            attrs_map = {k.lower(): v or "" for k, v in attrs}
            self._current_href = attrs_map.get("href")
            self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._current_href is None:
            return
        text = normalize_text(" ".join(self._chunks))
        if text:
            self.anchors.append({"href": self._current_href, "text": text})
        self._current_href = None
        self._chunks = []


def normalize_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def short_snippet(value: Any, limit: int = 240) -> str:
    text = normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def stable_id(*parts: Any) -> str:
    payload = "|".join(normalize_text(part).lower() for part in parts if part is not None)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def safe_to_datetime(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce", utc=False)
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def date_only_from_text(text: str) -> tuple[pd.Timestamp | None, str]:
    m = re.search(r"((?:19|20)\d{2})[/-](\d{1,2})[/-](\d{1,2})", text)
    if not m:
        return None, text
    date_text = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    title = normalize_text(text[m.end() :])
    return pd.Timestamp(date_text), title


def fetch_url(url: str, source_name: str, timeout: int = 20) -> FetchResult:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "text/html,application/rss+xml,application/xml,text/xml,*/*"},
            timeout=timeout,
        )
        status = str(resp.status_code)
        if resp.status_code >= 400:
            return FetchResult(source_name, url, resp.text[:500], False, status, f"HTTP {resp.status_code}")
        return FetchResult(source_name, url, resp.text, True, status)
    except Exception as exc:
        return FetchResult(source_name, url, "", False, "ERROR", str(exc))


def source_rank(source: str, url: str = "") -> int:
    lower_url = url.lower()
    if source in {"seed_events", "tsmc_press", "tsmc_investor"}:
        return 10
    if "reuters.com" in lower_url or "bloomberg.com" in lower_url:
        return 9
    if "yahoo.com" in lower_url:
        return 7
    if source == "google_news_rss":
        return 6
    return 5


def article_row(
    *,
    source: str,
    title: str,
    url: str,
    published_at: pd.Timestamp | None,
    date_only: bool,
    publisher: str = "",
    snippet: str = "",
    discovered_at: str = "",
    source_status: str = "OK",
) -> dict[str, Any]:
    title = normalize_text(title)
    url = normalize_text(url)
    article_id = stable_id(source, url or title, published_at.date().isoformat() if published_at is not None else "")
    return {
        "article_id": article_id,
        "source": source,
        "publisher": normalize_text(publisher or source),
        "title": title,
        "source_url": url,
        "published_at_utc": "" if published_at is None else to_utc_iso(published_at, date_only=date_only),
        "published_date_only": bool(date_only),
        "discovered_at_utc": discovered_at,
        "snippet": short_snippet(snippet),
        "source_rank": source_rank(source, url),
        "coverage_status": source_status,
    }


def to_utc_iso(ts: pd.Timestamp, date_only: bool = False) -> str:
    if date_only:
        normalized = ts.normalize()
        if normalized.tzinfo is None:
            return normalized.tz_localize(timezone.utc).isoformat()
        return normalized.tz_convert(timezone.utc).isoformat()
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.isoformat()


def parse_tsmc_archive_html(html_text: str, page_url: str, discovered_at: str) -> list[dict[str, Any]]:
    parser = AnchorExtractor()
    parser.feed(html_text)
    rows = []
    seen: set[str] = set()
    for anchor in parser.anchors:
        href = urljoin(page_url, anchor["href"])
        if "/news/" not in href:
            continue
        event_date, title = date_only_from_text(anchor["text"])
        if event_date is None or not title:
            continue
        key = href.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            article_row(
                source="tsmc_press",
                title=title,
                url=href,
                published_at=event_date,
                date_only=True,
                publisher="TSMC",
                snippet=title,
                discovered_at=discovered_at,
            )
        )
    return rows


def parse_tsmc_latest_html(html_text: str, page_url: str, discovered_at: str) -> list[dict[str, Any]]:
    parser = AnchorExtractor()
    parser.feed(html_text)
    rows = []
    seen: set[str] = set()
    for anchor in parser.anchors:
        href = urljoin(page_url, anchor["href"])
        if "/news/" not in href:
            continue
        text = anchor["text"]
        date_match = re.search(r"\(((?:19|20)\d{2}/\d{1,2}/\d{1,2})\)", text)
        event_date = pd.Timestamp(date_match.group(1).replace("/", "-")) if date_match else None
        title = normalize_text(re.sub(r"\s*\((?:19|20)\d{2}/\d{1,2}/\d{1,2}\)\s*$", "", text))
        if not title or href.lower() in seen:
            continue
        seen.add(href.lower())
        rows.append(
            article_row(
                source="tsmc_press",
                title=title,
                url=href,
                published_at=event_date,
                date_only=True,
                publisher="TSMC",
                snippet=title,
                discovered_at=discovered_at,
            )
        )
    return rows


def parse_google_news_rss(xml_text: str, discovered_at: str) -> list[dict[str, Any]]:
    rows = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return rows
    for item in root.findall(".//item"):
        values: dict[str, str] = {}
        for child in list(item):
            name = child.tag.split("}", 1)[-1].lower()
            values[name] = normalize_text(child.text or "")
        title = values.get("title", "")
        link = values.get("link", "")
        pub = safe_to_datetime(values.get("pubdate"))
        publisher = values.get("source", "Google News")
        if not title or not link:
            continue
        rows.append(
            article_row(
                source="google_news_rss",
                title=title,
                url=link,
                published_at=pub,
                date_only=False if pub is not None else True,
                publisher=publisher,
                snippet=values.get("description", ""),
                discovered_at=discovered_at,
            )
        )
    return rows


def parse_yahoo_finance_html(html_text: str, page_url: str, discovered_at: str) -> list[dict[str, Any]]:
    parser = AnchorExtractor()
    parser.feed(html_text)
    rows = []
    seen: set[str] = set()
    keywords = re.compile(
        r"\b(tsmc|taiwan semiconductor|tsm|samsung electronics|samsung|sk hynix|hynix|005930|000660|semiconductor|chip|ai|hbm|dram|memory)\b",
        re.I,
    )
    article_url = re.compile(r"(/news/|/m/|fool\.com|investors\.com|benzinga\.com|zacks\.com|reuters\.com|bloomberg\.com|cnbc\.com)", re.I)
    generic_titles = {
        "more about taiwan semiconductor manufacturing company limited",
        "tsm taiwan semiconductor manufacturing company limited",
        "taiwan semiconductor manufacturing company limited",
        "005930.ks samsung electronics co., ltd.",
        "000660.ks sk hynix inc.",
    }
    discovered_ts = safe_to_datetime(discovered_at) or pd.Timestamp.now(tz="UTC")
    for anchor in parser.anchors:
        title = normalize_text(anchor["text"])
        title_key = title.lower()
        if len(title) < 18 or title_key in generic_titles or not keywords.search(title):
            continue
        href = urljoin(page_url, anchor["href"])
        if not article_url.search(href):
            continue
        if href.lower() in seen:
            continue
        seen.add(href.lower())
        rows.append(
            article_row(
                source="yahoo_finance",
                title=title,
                url=href,
                published_at=discovered_ts,
                date_only=False,
                publisher="Yahoo Finance",
                snippet=title,
                discovered_at=discovered_at,
            )
        )
    return rows[:40]


def load_seed_events(path: Path, discovered_at: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    seed = pd.read_csv(path)
    if seed.empty:
        return []
    rows = []
    for _, row in seed.iterrows():
        event_date = safe_to_datetime(row.get("event_date"))
        title = normalize_text(row.get("event_name", ""))
        if not title:
            continue
        rows.append(
            article_row(
                source="seed_events",
                title=title,
                url=str(row.get("source_url", "")),
                published_at=event_date,
                date_only=True,
                publisher=str(row.get("event_type", "seed_event")),
                snippet=str(row.get("notes", "")),
                discovered_at=discovered_at,
                source_status="SEED_EVENT",
            )
        )
    return rows


def collect_articles(
    *,
    events_path: Path,
    sources: list[str],
    skip_web: bool,
    politeness_delay_sec: float,
    offline_fixtures: Path | None,
) -> tuple[pd.DataFrame, list[str]]:
    discovered_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rows = load_seed_events(events_path, discovered_at)
    warnings: list[str] = []

    if skip_web:
        return dedupe_articles(pd.DataFrame(rows)), warnings

    fixture_map: dict[str, tuple[str, str, Any]] = {
        "tsmc_press": ("tsmc_archive.html", TSMC_PRESS_ARCHIVE_URL, parse_tsmc_archive_html),
        "tsmc_latest": ("tsmc_latest.html", TSMC_PRESS_LATEST_URL, parse_tsmc_latest_html),
        "google_news_rss": ("google_news_rss.xml", GOOGLE_NEWS_RSS_URL, parse_google_news_rss),
        "yahoo_finance_tsm": ("yahoo_finance.html", YAHOO_FINANCE_QUOTE_URLS["yahoo_finance_tsm"], parse_yahoo_finance_html),
        "yahoo_finance_samsung": ("yahoo_finance_samsung.html", YAHOO_FINANCE_QUOTE_URLS["yahoo_finance_samsung"], parse_yahoo_finance_html),
        "yahoo_finance_sk_hynix": ("yahoo_finance_sk_hynix.html", YAHOO_FINANCE_QUOTE_URLS["yahoo_finance_sk_hynix"], parse_yahoo_finance_html),
    }

    for source in sources:
        source = source.strip()
        if source == "tsmc_investor":
            # TSMC monthly revenue releases are mirrored in the press archive. Keep
            # this source name in config without duplicating sparse table pages.
            continue
        if source == "tsmc_press":
            jobs = ["tsmc_press", "tsmc_latest"]
        elif source == "yahoo_finance":
            jobs = ["yahoo_finance_tsm", "yahoo_finance_samsung", "yahoo_finance_sk_hynix"]
        else:
            jobs = [source]
        for job in jobs:
            if job not in fixture_map:
                continue
            fixture_name, url, parser = fixture_map[job]
            text = ""
            if offline_fixtures is not None:
                fixture_path = offline_fixtures / fixture_name
                if fixture_path.exists():
                    text = fixture_path.read_text(encoding="utf-8")
                else:
                    warnings.append(f"{job}: fixture missing {fixture_path}")
                    continue
            else:
                result = fetch_url(url, job)
                if not result.ok:
                    warnings.append(f"{job}: {result.error or result.status}")
                    continue
                text = result.text
                if politeness_delay_sec > 0:
                    time.sleep(politeness_delay_sec)
            if job == "google_news_rss":
                rows.extend(parser(text, discovered_at))
            else:
                rows.extend(parser(text, url, discovered_at))

    return dedupe_articles(pd.DataFrame(rows)), warnings


def dedupe_articles(articles: pd.DataFrame) -> pd.DataFrame:
    if articles.empty:
        return pd.DataFrame(columns=["article_id", "source", "publisher", "title", "source_url", "published_at_utc", "published_date_only", "discovered_at_utc", "snippet", "source_rank", "coverage_status"])
    out = articles.copy()
    out["dedupe_key"] = np.where(
        out["source_url"].astype(str).str.len() > 0,
        out["source_url"].astype(str).str.lower(),
        out["title"].astype(str).str.lower() + "|" + out["published_at_utc"].astype(str).str[:10],
    )
    out = out.sort_values(["source_rank", "published_at_utc"], ascending=[False, True]).drop_duplicates("dedupe_key", keep="first")
    out = out.drop(columns=["dedupe_key"]).sort_values(["published_at_utc", "title"], na_position="last").reset_index(drop=True)
    return out


def keyword_matches(text: str, keyword: str) -> bool:
    keyword = keyword.lower().strip()
    if not keyword:
        return False
    if re.fullmatch(r"[a-z0-9]+", keyword):
        return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
    return keyword in text


def merge_existing_articles(current: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    if not existing_path.exists():
        return current
    try:
        existing = pd.read_csv(existing_path)
    except Exception:
        return current
    if existing.empty:
        return current
    return dedupe_articles(pd.concat([existing, current], ignore_index=True, sort=False))


def preserve_existing_coverage_for_unchanged_dates(current: pd.DataFrame, existing_path: Path, updated_dates: set[str]) -> pd.DataFrame:
    if current.empty or "date" not in current.columns or "news_coverage_status" not in current.columns or not existing_path.exists():
        return current
    try:
        existing = pd.read_csv(existing_path)
    except Exception:
        return current
    if existing.empty or "date" not in existing.columns or "news_coverage_status" not in existing.columns:
        return current
    out = current.copy()
    old_status = existing[["date", "news_coverage_status"]].dropna(subset=["date"]).drop_duplicates("date", keep="last")
    old_map = dict(zip(old_status["date"].astype(str), old_status["news_coverage_status"].astype(str)))
    date_text = out["date"].astype(str)
    unchanged = ~date_text.isin(updated_dates)
    out.loc[unchanged, "news_coverage_status"] = date_text[unchanged].map(old_map).fillna(out.loc[unchanged, "news_coverage_status"])
    return out


def classify_cause(title: str, snippet: str = "") -> str:
    text = f"{title} {snippet}".lower()
    opinion_like = any(term in text for term in INVESTMENT_OPINION_TERMS)
    concrete_event_like = any(keyword_matches(text, term) for term in CONCRETE_EVENT_HINTS)
    if opinion_like and not concrete_event_like:
        return "technical_market_move"
    priority = [
        "operational_disruption",
        "regulation_export_controls",
        "geopolitics_taiwan",
        "earnings_results",
        "guidance",
        "monthly_revenue",
        "analyst_rating_target",
        "dividend_capital_return",
        "capex_fab_expansion",
        "ai_hpc_demand",
        "customer_supply_chain",
        "peer_sector_move",
        "macro_rates_fx",
        "technical_market_move",
    ]
    for cause in priority:
        keywords = CAUSE_KEYWORDS[cause]
        if any(keyword_matches(text, keyword) for keyword in keywords):
            return cause
    return "other"


def sentiment_score(title: str, snippet: str = "") -> float:
    text = f"{title} {snippet}".lower()
    pos = sum(1 for term in POSITIVE_TERMS if term in text)
    neg = sum(1 for term in NEGATIVE_TERMS if term in text)
    score = (pos - neg) / max(pos + neg, 1)
    return float(np.clip(score, -1.0, 1.0))


def relevance_score(title: str, snippet: str = "") -> int:
    text = f"{title} {snippet}".lower()
    if "tsmc" in text or "taiwan semiconductor" in text:
        return 25
    if re.search(r"\btsm\b", text):
        return 22
    if "samsung electronics" in text or "005930" in text:
        return 24
    if "sk hynix" in text or "hynix" in text or "000660" in text:
        return 24
    if "semiconductor" in text or "chip" in text:
        return 14
    return 6


def normalize_events(articles: pd.DataFrame, trading_dates: pd.Series, market_timezone: str) -> pd.DataFrame:
    if articles.empty:
        return empty_events()
    rows = []
    for _, article in articles.iterrows():
        title = str(article.get("title", ""))
        snippet = str(article.get("snippet", ""))
        published = safe_to_datetime(article.get("published_at_utc"))
        event_date = published.date().isoformat() if published is not None else ""
        rows.append(
            {
                "event_id": stable_id(article.get("article_id"), title),
                "article_id": article.get("article_id"),
                "event_name": title,
                "event_date": event_date,
                "published_at_utc": article.get("published_at_utc", ""),
                "published_date_only": bool(article.get("published_date_only", False)),
                "source": article.get("source", ""),
                "publisher": article.get("publisher", ""),
                "source_url": article.get("source_url", ""),
                "snippet": snippet,
                "cause_type": classify_cause(title, snippet),
                "sentiment_score": sentiment_score(title, snippet),
                "relevance_score": relevance_score(title, snippet),
                "source_rank": int(article.get("source_rank", 0) or 0),
                "coverage_status": article.get("coverage_status", ""),
            }
        )
    events = pd.DataFrame(rows)
    events = assign_available_signal_dates(events, trading_dates, market_timezone)
    return events


def empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "article_id",
            "event_name",
            "event_date",
            "available_for_signal_date",
            "published_at_utc",
            "published_date_only",
            "source",
            "publisher",
            "source_url",
            "snippet",
            "cause_type",
            "sentiment_score",
            "relevance_score",
            "source_rank",
            "coverage_status",
        ]
    )


def assign_available_signal_dates(events: pd.DataFrame, trading_dates: pd.Series, market_timezone: str = "America/New_York") -> pd.DataFrame:
    if events.empty:
        return events
    out = events.copy()
    dates = pd.to_datetime(trading_dates).dt.normalize().dropna().drop_duplicates().sort_values().reset_index(drop=True)
    tz = ZoneInfo(market_timezone)

    def next_on_or_after(day: pd.Timestamp) -> str:
        day = pd.Timestamp(day).normalize()
        future = dates[dates >= day]
        if not future.empty:
            return future.iloc[0].date().isoformat()
        return day.date().isoformat()

    def next_after(day: pd.Timestamp) -> str:
        day = pd.Timestamp(day).normalize()
        future = dates[dates > day]
        if not future.empty:
            return future.iloc[0].date().isoformat()
        return day.date().isoformat()

    assigned = []
    for _, row in out.iterrows():
        event_day = safe_to_datetime(row.get("event_date"))
        published = safe_to_datetime(row.get("published_at_utc"))
        source = str(row.get("source", ""))
        date_only = bool(row.get("published_date_only", False))
        if event_day is None and published is not None:
            event_day = published
        if event_day is None:
            assigned.append("")
            continue
        if source == "seed_events":
            assigned.append(next_on_or_after(event_day))
            continue
        if date_only or published is None:
            assigned.append(next_after(event_day))
            continue
        if published.tzinfo is None:
            published = published.tz_localize(timezone.utc)
        local = published.tz_convert(tz)
        local_day = local.normalize()
        is_trading_day = bool((dates == local_day.tz_localize(None)).any()) if local_day.tzinfo is not None else bool((dates == local_day).any())
        naive_local_day = pd.Timestamp(local.date())
        if is_trading_day and local.time() <= MARKET_CLOSE:
            assigned.append(next_on_or_after(naive_local_day))
        else:
            assigned.append(next_after(naive_local_day))
    out["available_for_signal_date"] = assigned
    return out


def tokenize_title(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 2 and tok not in {"the", "and", "for", "with", "from", "tsmc", "tsm"}}


def jaccard_similarity(a: str, b: str) -> float:
    ta = tokenize_title(a)
    tb = tokenize_title(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def similarity_matrix(titles: list[str]) -> np.ndarray:
    if not titles:
        return np.zeros((0, 0))
    if TfidfVectorizer is not None and cosine_similarity is not None:
        try:
            matrix = TfidfVectorizer(min_df=1, ngram_range=(1, 2)).fit_transform(titles)
            return cosine_similarity(matrix)
        except Exception:
            pass
    out = np.zeros((len(titles), len(titles)))
    for i, title_a in enumerate(titles):
        for j, title_b in enumerate(titles):
            out[i, j] = 1.0 if i == j else jaccard_similarity(title_a, title_b)
    return out


def cluster_events(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        empty_clusters = pd.DataFrame(columns=["cluster_id", "representative_title", "cause_type", "start_date", "end_date", "article_count", "source_count", "source_urls", "coverage_status"])
        return events.assign(cluster_id=pd.Series(dtype=str)), empty_clusters
    out = events.copy().reset_index(drop=True)
    out["available_dt"] = pd.to_datetime(out["available_for_signal_date"], errors="coerce")
    titles = out["event_name"].fillna("").astype(str).tolist()
    sim = similarity_matrix(titles)
    assigned: dict[int, str] = {}
    clusters = []
    cluster_num = 1
    for i, row in out.sort_values(["available_dt", "source_rank"], ascending=[True, False]).iterrows():
        if i in assigned:
            continue
        cluster_id = f"NEWS{cluster_num:05d}"
        cluster_num += 1
        assigned[i] = cluster_id
        members = [i]
        for j, other in out.iterrows():
            if j == i or j in assigned:
                continue
            if str(other.get("cause_type")) != str(row.get("cause_type")):
                continue
            day_gap = abs((pd.Timestamp(other["available_dt"]) - pd.Timestamp(row["available_dt"])).days) if pd.notna(other["available_dt"]) and pd.notna(row["available_dt"]) else 99
            if day_gap > 3:
                continue
            title_sim = float(sim[i, j]) if sim.size else 0.0
            if title_sim >= 0.65 or jaccard_similarity(str(row.get("event_name", "")), str(other.get("event_name", ""))) >= 0.5:
                assigned[j] = cluster_id
                members.append(j)
        member_df = out.loc[members].sort_values(["source_rank", "available_dt"], ascending=[False, True])
        representative = member_df.iloc[0]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "representative_title": representative.get("event_name", ""),
                "cause_type": representative.get("cause_type", ""),
                "start_date": member_df["available_for_signal_date"].min(),
                "end_date": member_df["available_for_signal_date"].max(),
                "article_count": len(member_df),
                "source_count": member_df["publisher"].nunique(dropna=True),
                "source_urls": "|".join(member_df["source_url"].dropna().astype(str).head(5)),
                "coverage_status": "|".join(sorted(set(member_df["coverage_status"].dropna().astype(str)))),
            }
        )
    out["cluster_id"] = [assigned.get(i, "") for i in range(len(out))]
    out = out.drop(columns=["available_dt"])
    return out, pd.DataFrame(clusters)


def confidence_label(score: float) -> str:
    if score >= 70:
        return "HIGH"
    if score >= 45:
        return "MEDIUM"
    return "LOW"


def adjusted_confidence_label(score: float, event: pd.Series) -> str:
    label = confidence_label(score)
    if label == "HIGH" and str(event.get("cause_type", "other")) in LOW_MATERIAL_PRIMARY_CAUSES:
        return "MEDIUM"
    return label


def confidence_score(event: pd.Series, price_row: pd.Series, day_distance: int) -> float:
    time_score = max(0, 35 - max(day_distance, 0) * 12)
    relevance = min(25, int(event.get("relevance_score", 0) or 0))
    importance = CAUSE_IMPORTANCE.get(str(event.get("cause_type", "other")), 4)
    source = min(10, int(event.get("source_rank", 0) or 0))
    price_move = float(price_row.get("close_change_pct", 0) or 0)
    sentiment = float(event.get("sentiment_score", 0) or 0)
    if sentiment == 0 or abs(price_move) < 0.005:
        direction = 5
    elif sentiment > 0 and price_move > 0:
        direction = 10
    elif sentiment < 0 and price_move < 0:
        direction = 10
    else:
        direction = 0
    return float(time_score + relevance + importance + source + direction)


def build_price_news_matches(enriched: pd.DataFrame, events: pd.DataFrame, clusters: pd.DataFrame, overall_coverage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = enriched.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    if events.empty:
        matches = default_daily_news(prices, overall_coverage)
        return matches, matches[NEWS_DAILY_COLUMNS].copy()

    ev = events.copy()
    ev["available_dt"] = pd.to_datetime(ev["available_for_signal_date"], errors="coerce").dt.normalize()
    ev = ev.dropna(subset=["available_dt"])
    date_to_idx = {d: i for i, d in enumerate(prices["date"])}
    ev["date_idx"] = ev["available_dt"].map(date_to_idx)
    ev = ev[pd.notna(ev["date_idx"])].copy()
    ev["date_idx"] = ev["date_idx"].astype(int)

    rows = []
    for idx, price_row in prices.iterrows():
        window = ev[(ev["date_idx"] <= idx) & (ev["date_idx"] >= idx - 2)].copy()
        one_day = ev[ev["date_idx"] == idx].copy()
        base = {
            "date": price_row["date"].date().isoformat(),
            "close_change_pct": price_row.get("close_change_pct", np.nan),
            "open_gap_pct": price_row.get("open_gap_pct", np.nan),
            "volume_ratio_20": price_row.get("volume_ratio_20", np.nan),
            "algo_event_shock_day": bool(price_row.get("algo_event_shock_day", False)),
            "news_event_count_1d": int(len(one_day)),
            "news_event_count_3d": int(len(window)),
            "news_sentiment_score_1d": float(one_day["sentiment_score"].mean()) if not one_day.empty else 0.0,
            "news_source_count": int(window["publisher"].nunique()) if not window.empty else 0,
            "news_coverage_status": overall_coverage,
            "news_penalty_event": 0.0,
        }
        if window.empty:
            rows.append(
                {
                    **base,
                    "news_primary_cause_type": NO_HIGH_CONFIDENCE_CAUSE,
                    "news_primary_cluster_id": NO_CLUSTER,
                    "news_match_confidence": "NO_MATCH",
                    "news_match_confidence_score": 0.0,
                    "news_primary_source_url": "",
                    "news_cause_summary": "",
                    "candidate_event_titles": "",
                }
            )
            continue
        scored = []
        for _, event in window.iterrows():
            day_distance = idx - int(event["date_idx"])
        scored.append((confidence_score(event, price_row, day_distance), event))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_event = scored[0]
        label = adjusted_confidence_label(best_score, best_event)
        primary_cause = str(best_event.get("cause_type", "")) if label == "HIGH" else NO_HIGH_CONFIDENCE_CAUSE
        primary_cluster = str(best_event.get("cluster_id", "")) if label == "HIGH" else NO_CLUSTER
        possible_prefix = "" if label == "HIGH" else "possible: "
        summary = f"{possible_prefix}{best_event.get('event_name', '')}"
        sentiment = float(best_event.get("sentiment_score", 0) or 0)
        best_cause = str(best_event.get("cause_type", ""))
        negative_high = label == "HIGH" and sentiment < -0.15 and best_cause in NEGATIVE_PENALTY_CAUSES
        news_penalty = 6.0 if negative_high else 0.0
        if negative_high and best_cause in SEVERE_NEGATIVE_PENALTY_CAUSES:
            news_penalty += 3.0
        rows.append(
            {
                **base,
                "news_primary_cause_type": primary_cause,
                "news_primary_cluster_id": primary_cluster,
                "news_match_confidence": label,
                "news_match_confidence_score": round(best_score, 2),
                "news_primary_source_url": best_event.get("source_url", ""),
                "news_cause_summary": short_snippet(summary, 180),
                "candidate_event_titles": "|".join(window.sort_values("source_rank", ascending=False)["event_name"].astype(str).head(5)),
                "news_penalty_event": news_penalty,
            }
        )
    matches = pd.DataFrame(rows)
    daily = matches[NEWS_DAILY_COLUMNS].copy()
    return matches, daily


def default_daily_news(prices: pd.DataFrame, coverage: str) -> pd.DataFrame:
    out = prices[["date"]].copy()
    out["date"] = out["date"].dt.date.astype(str)
    for col in NEWS_DAILY_COLUMNS:
        if col == "date":
            continue
        if col in {"news_event_count_1d", "news_event_count_3d", "news_source_count"}:
            out[col] = 0
        elif col in {"news_sentiment_score_1d", "news_match_confidence_score", "news_penalty_event"}:
            out[col] = 0.0
        elif col == "news_coverage_status":
            out[col] = coverage
        elif col == "news_primary_cause_type":
            out[col] = NO_HIGH_CONFIDENCE_CAUSE
        elif col == "news_primary_cluster_id":
            out[col] = NO_CLUSTER
        else:
            out[col] = ""
    return out


def write_report(path: Path, matches: pd.DataFrame, articles: pd.DataFrame, events: pd.DataFrame, clusters: pd.DataFrame, warnings: list[str], coverage: str) -> None:
    lines = [
        "# Semiconductor News Causal Event Report",
        "",
        f"- Coverage status: {coverage}",
        f"- Articles collected: {len(articles)}",
        f"- Normalized events: {len(events)}",
        f"- Event clusters: {len(clusters)}",
        f"- Generated at UTC: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
    ]
    if warnings:
        lines.extend(["", "## Collection Warnings"])
        for warning in warnings[:20]:
            lines.append(f"- {warning}")
    if not matches.empty:
        focus = matches.copy()
        for col in ["close_change_pct", "open_gap_pct", "volume_ratio_20"]:
            if col in focus.columns:
                focus[col] = pd.to_numeric(focus[col], errors="coerce")
        shock = focus[
            (focus["close_change_pct"].abs() >= 0.03)
            | (focus["open_gap_pct"].abs() >= 0.025)
            | (focus["algo_event_shock_day"].astype(str).str.lower().isin(["true", "1"]))
        ].tail(30)
        lines.extend(["", "## Price Move Cause Matches", "", "| Date | Move | Confidence | Cause | Summary |", "|---|---:|---|---|---|"])
        for _, row in shock.iterrows():
            move = float(row.get("close_change_pct", 0) or 0) * 100
            lines.append(
                f"| {row.get('date', '')} | {move:.2f}% | {row.get('news_match_confidence', '')} | {row.get('news_primary_cause_type', '')} | {normalize_text(row.get('news_cause_summary', ''))} |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "- This engine stores metadata and short snippets only; full article bodies are not stored.",
            "- Direct web/RSS collection is best-effort and can miss historical external news.",
            "- HIGH confidence is required before a cause is treated as a primary matched cause.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_sources(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect direct-web news metadata and match likely causes to semiconductor price moves.")
    parser.add_argument("--enriched", default="output/tsm_daily_10y_enriched.csv")
    parser.add_argument("--events", default="tsm_events_seed.csv")
    parser.add_argument("--start", default="2016-05-12")
    parser.add_argument("--end", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument("--mode", choices=["backfill", "daily"], default="daily")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--sources", default="tsmc_press,tsmc_investor,google_news_rss,yahoo_finance")
    parser.add_argument("--outdir", default="output")
    parser.add_argument("--rule-outdir", default="tsm_price_rule_output")
    parser.add_argument("--market-timezone", default="America/New_York")
    parser.add_argument("--politeness-delay-sec", type=float, default=1.0)
    parser.add_argument("--skip-web", action="store_true")
    parser.add_argument("--offline-fixtures", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enriched_path = Path(args.enriched)
    outdir = Path(args.outdir)
    rule_outdir = Path(args.rule_outdir)
    if not enriched_path.exists():
        raise SystemExit(f"missing enriched file: {enriched_path}")
    enriched = pd.read_csv(enriched_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if args.mode == "daily":
        start = pd.Timestamp(args.end) - pd.Timedelta(days=int(args.lookback_days))
    else:
        start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    trading_dates = pd.to_datetime(enriched["date"]).dt.normalize()
    target_enriched = enriched[(trading_dates >= start.normalize()) & (trading_dates <= end.normalize())].copy()
    if target_enriched.empty:
        target_enriched = enriched.tail(1).copy()

    fixtures = Path(args.offline_fixtures) if args.offline_fixtures else None
    articles, warnings = collect_articles(
        events_path=Path(args.events),
        sources=parse_sources(args.sources),
        skip_web=args.skip_web,
        politeness_delay_sec=float(args.politeness_delay_sec),
        offline_fixtures=fixtures,
    )
    if args.mode == "daily":
        articles = merge_existing_articles(articles, outdir / "tsm_news_raw_articles.csv")
    coverage = "DIRECT_WEB_OK" if not warnings else "DIRECT_WEB_PARTIAL"
    events = normalize_events(articles, enriched["date"], args.market_timezone)
    events, clusters = cluster_events(events)
    matches, daily = build_price_news_matches(enriched, events, clusters, coverage)

    if args.mode == "daily":
        keep_dates = set(pd.to_datetime(target_enriched["date"]).dt.date.astype(str))
        daily_out = preserve_existing_coverage_for_unchanged_dates(
            daily.copy(),
            rule_outdir / "tsm_news_integrated_daily.csv",
            keep_dates,
        )
        matches_out = preserve_existing_coverage_for_unchanged_dates(
            matches.copy(),
            outdir / "tsm_price_news_matches.csv",
            keep_dates,
        )
    else:
        daily_out = daily
        matches_out = matches

    if args.dry_run:
        print(
            {
                "articles": len(articles),
                "events": len(events),
                "clusters": len(clusters),
                "matches": len(matches_out),
                "daily_rows": len(daily_out),
                "coverage_status": coverage,
                "warnings": warnings,
            }
        )
        return

    outdir.mkdir(parents=True, exist_ok=True)
    rule_outdir.mkdir(parents=True, exist_ok=True)
    articles.to_csv(outdir / "tsm_news_raw_articles.csv", index=False)
    events.to_csv(outdir / "tsm_news_events_normalized.csv", index=False)
    clusters.to_csv(outdir / "tsm_news_event_clusters.csv", index=False)
    matches_out.to_csv(outdir / "tsm_price_news_matches.csv", index=False)
    daily_out.to_csv(rule_outdir / "tsm_news_integrated_daily.csv", index=False)
    write_report(rule_outdir / "tsm_news_causal_event_report.md", matches_out, articles, events, clusters, warnings, coverage)
    print("completed: news causal outputs =", rule_outdir.resolve())
    print({"articles": len(articles), "events": len(events), "clusters": len(clusters), "matches": len(matches_out), "coverage_status": coverage})


if __name__ == "__main__":
    main()

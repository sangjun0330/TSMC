import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tsm_news_causal_engine import (
    adjusted_confidence_label,
    article_row,
    assign_available_signal_dates,
    classify_cause,
    confidence_label,
    confidence_score,
    dedupe_articles,
    parse_google_news_rss,
    parse_tsmc_archive_html,
    preserve_existing_coverage_for_unchanged_dates,
    short_snippet,
)


class NewsCausalEngineTests(unittest.TestCase):
    def test_parse_tsmc_archive_html(self):
        html = '<a href="/english/news/3294">2026/04/10 TSMC March 2026 Revenue Report</a>'
        rows = parse_tsmc_archive_html(html, "https://pr.tsmc.com/english/news-archives", "2026-05-17T00:00:00+00:00")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "TSMC March 2026 Revenue Report")
        self.assertEqual(rows[0]["source_url"], "https://pr.tsmc.com/english/news/3294")

    def test_google_news_rss_parse_and_dedupe(self):
        rss = """<?xml version="1.0"?><rss><channel>
        <item><title>TSMC shares rise on AI chip demand</title><link>https://example.com/a</link><pubDate>Fri, 10 May 2024 14:00:00 GMT</pubDate><source>Example</source><description>Short summary</description></item>
        <item><title>Duplicate title</title><link>https://example.com/a</link><pubDate>Fri, 10 May 2024 14:05:00 GMT</pubDate><source>Example</source></item>
        </channel></rss>"""
        rows = parse_google_news_rss(rss, "2026-05-17T00:00:00+00:00")
        deduped = dedupe_articles(pd.DataFrame(rows))
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(deduped), 1)

    def test_taxonomy_keyword_classification(self):
        self.assertEqual(classify_cause("TSMC Reports First Quarter EPS of NT$22.08"), "earnings_results")
        self.assertEqual(classify_cause("TSMC March 2026 Revenue Report"), "monthly_revenue")
        self.assertEqual(classify_cause("US BIS export controls hit advanced computing chips"), "regulation_export_controls")
        self.assertEqual(classify_cause("Taiwan earthquake causes temporary chip production disruption"), "operational_disruption")
        self.assertEqual(classify_cause("Analyst upgrades TSM price target"), "analyst_rating_target")
        self.assertEqual(classify_cause("TSMC rises as AI demand fuels HPC growth"), "ai_hpc_demand")
        self.assertEqual(classify_cause("Taiwan Semiconductor Manufacturing rallied on increasing demand for AI chips"), "ai_hpc_demand")
        self.assertEqual(classify_cause("Lone Pine Capital cuts stake in Taiwan Semiconductor"), "technical_market_move")
        self.assertEqual(classify_cause("Broadcom vs Taiwan Semiconductor: Which AI Chip Giant Is the Better Buy Right Now?"), "technical_market_move")
        self.assertEqual(classify_cause("TSMC shares rise as AI chip demand accelerates"), "ai_hpc_demand")
        self.assertEqual(classify_cause("Taiwan Semiconductor CEO hints at the next move in AI stocks"), "technical_market_move")
        self.assertEqual(classify_cause("Billionaire is loading up on Taiwan Semiconductor despite geopolitical risks"), "technical_market_move")
        self.assertEqual(classify_cause("China Taiwan Strait military tension hits chip stocks"), "geopolitics_taiwan")

    def test_after_close_news_maps_to_next_trading_day(self):
        events = pd.DataFrame(
            [
                {
                    "event_id": "a",
                    "event_date": "2024-01-02",
                    "published_at_utc": "2024-01-02T22:30:00+00:00",
                    "published_date_only": False,
                    "source": "google_news_rss",
                }
            ]
        )
        trading_dates = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"]))
        out = assign_available_signal_dates(events, trading_dates, "America/New_York")
        self.assertEqual(out.iloc[0]["available_for_signal_date"], "2024-01-03")

    def test_confidence_labels_and_score(self):
        event = pd.Series({"relevance_score": 25, "cause_type": "earnings_results", "source_rank": 10, "sentiment_score": 1.0})
        price = pd.Series({"close_change_pct": 0.04})
        self.assertEqual(confidence_label(confidence_score(event, price, 0)), "HIGH")
        self.assertEqual(confidence_label(50), "MEDIUM")
        self.assertEqual(confidence_label(20), "LOW")

    def test_low_material_cause_is_not_primary_high_confidence(self):
        event = pd.Series({"cause_type": "technical_market_move"})
        self.assertEqual(adjusted_confidence_label(82, event), "MEDIUM")

    def test_daily_coverage_preserves_unchanged_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.csv"
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "news_coverage_status": ["DIRECT_WEB_OK", "DIRECT_WEB_PARTIAL"],
                }
            ).to_csv(path, index=False)
            current = pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "news_coverage_status": ["DIRECT_WEB_PARTIAL", "DIRECT_WEB_PARTIAL"],
                }
            )
            out = preserve_existing_coverage_for_unchanged_dates(current, path, {"2024-01-03"})
            self.assertEqual(out.loc[0, "news_coverage_status"], "DIRECT_WEB_OK")
            self.assertEqual(out.loc[1, "news_coverage_status"], "DIRECT_WEB_PARTIAL")

    def test_article_body_is_not_stored(self):
        long_text = "TSMC " + ("long body " * 80)
        row = article_row(
            source="google_news_rss",
            title="TSMC article",
            url="https://example.com/article",
            published_at=pd.Timestamp("2024-01-01"),
            date_only=True,
            snippet=long_text,
            discovered_at="2026-05-17T00:00:00+00:00",
        )
        self.assertLessEqual(len(row["snippet"]), 240)
        self.assertEqual(row["snippet"], short_snippet(long_text))

    def test_dry_run_with_offline_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            enriched = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2020-07-23", "2020-07-24", "2020-07-27"]),
                    "close_change_pct": [0.0, 0.096, 0.12],
                    "open_gap_pct": [0.0, 0.05, 0.1],
                    "volume_ratio_20": [1.0, 2.0, 3.0],
                    "algo_event_shock_day": [False, True, True],
                }
            )
            enriched_path = root / "enriched.csv"
            events_path = root / "events.csv"
            fixtures = root / "fixtures"
            fixtures.mkdir()
            enriched.to_csv(enriched_path, index=False)
            pd.DataFrame(
                [
                    {
                        "event_date": "2020-07-24",
                        "event_name": "Intel 7nm delay and foundry outsourcing expectations",
                        "event_type": "industry",
                        "source_url": "https://example.com/intel",
                        "notes": "Intel delay news increased investor focus on TSMC outsourcing.",
                    }
                ]
            ).to_csv(events_path, index=False)
            (fixtures / "tsmc_archive.html").write_text('<a href="/english/news/1">2020/07/24 TSMC Revenue Report</a>', encoding="utf-8")
            cmd = [
                sys.executable,
                "tsm_news_causal_engine.py",
                "--enriched",
                str(enriched_path),
                "--events",
                str(events_path),
                "--offline-fixtures",
                str(fixtures),
                "--dry-run",
                "--outdir",
                str(root / "output"),
                "--rule-outdir",
                str(root / "rule"),
            ]
            result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("articles", result.stdout)


if __name__ == "__main__":
    unittest.main()

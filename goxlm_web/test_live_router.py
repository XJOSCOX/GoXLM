"""Synthetic-only tests for the offline router."""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

from goxlm_web.live_router import (
    Decision, Delivery, Freshness, Intent, Permission, assess_freshness, classify, permission,
)


class OfflineRouterTests(unittest.TestCase):
    def setUp(self):
        for target in ("socket.socket", "socket.create_connection", "socket.getaddrinfo"):
            guard = patch(target, side_effect=AssertionError(f"Unexpected networking: {target}"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_deterministic_classification(self):
        cases = {
            "Hi, thanks for helping": Intent.CONVERSATION,
            "How does a capacitor work?": Intent.TIMELESS,
            "What is a stock quote?": Intent.TIMELESS,
            "How do stock prices work?": Intent.TIMELESS,
            "What is news?": Intent.TIMELESS,
            "What's today's news?": Intent.NEWS,
            "Show me the latest headlines": Intent.NEWS,
            "How is NVIDIA stock doing?": Intent.MARKET_QUOTE,
            "Give me a quote for NYSE:IBM": Intent.MARKET_QUOTE,
            "What is NVDA's stock price?": Intent.MARKET_QUOTE,
            "Is NVDA stock up?": Intent.MARKET_QUOTE,
            "NVDA price?": Intent.MARKET_QUOTE,
            "What's the current market quote for $ZZZZ?": Intent.CLARIFY,
            "What's the live price for Tesla shares?": Intent.CLARIFY,
            "What is the latest news and NVIDIA stock price?": Intent.CLARIFY,
            "Show the current quotes for NVDA and AAPL shares": Intent.CLARIFY,
            "What is the weather right now?": Intent.CLARIFY,
            "": Intent.CLARIFY,
            "a" * 2049: Intent.CLARIFY,
            "\ud800": Intent.CLARIFY,
        }
        for text, expected in cases.items():
            with self.subTest(text=text[:72]):
                self.assertIs(classify(text).intent, expected)
                self.assertEqual(classify(text), classify(text))
        self.assertEqual(classify("How is NVIDIA stock doing?").query, "quote NASDAQ:NVDA")
        self.assertEqual(classify("Quote for NYSE:IBM").instrument, "NYSE:IBM")

    def test_opt_in_and_exact_query_permission(self):
        route = classify("What’s today’s news?")
        other = classify("Give me today's news about science")
        settings = {"browsing_enabled": True}
        self.assertIs(permission(route), Permission.DISABLED)
        self.assertIs(permission(route, reviewed=route, decision=Decision.APPROVE), Permission.DISABLED)
        self.assertIs(permission(route, **settings), Permission.CONFIRM)
        self.assertIs(permission(route, **settings, reviewed=route, decision=Decision.DENY), Permission.DENIED)
        self.assertIs(permission(route, **settings, reviewed=route, decision=Decision.APPROVE), Permission.APPROVED)
        self.assertIs(permission(route, **settings, reviewed=route, decision="approve"), Permission.CONFIRM)
        self.assertIs(permission(other, **settings, reviewed=route, decision=Decision.APPROVE), Permission.CONFIRM)
        self.assertIs(permission(classify("hello"), **settings), Permission.LOCAL)

    def test_news_timestamp_uses_source_time(self):
        now = datetime(2026, 1, 3, 16, tzinfo=timezone.utc)
        base = {"now": now, "retrieved_at": now, "max_age": timedelta(days=1)}
        self.assertEqual(assess_freshness(source_at=now - timedelta(minutes=1), **base).label, "Fresh")
        self.assertEqual(assess_freshness(source_at=now - timedelta(days=2), **base).label, "Stale")
        self.assertEqual(assess_freshness(source_at=None, **base).label, "Freshness unverified")
        self.assertIs(assess_freshness(source_at=now + timedelta(seconds=1), **base).freshness, Freshness.INVALID)
        self.assertIs(assess_freshness(source_at=now.replace(tzinfo=None), **base).freshness, Freshness.INVALID)
        late = {**base, "retrieved_at": now - timedelta(hours=1)}
        self.assertIs(assess_freshness(source_at=now, **late).freshness, Freshness.INVALID)
        offset = timezone(timedelta(hours=-5))
        self.assertEqual(assess_freshness(source_at=now.astimezone(offset), **base).age, timedelta())

    def test_delayed_quote_stays_delayed_and_expires(self):
        now = datetime(2026, 1, 3, 16, tzinfo=timezone.utc)
        base = {"now": now, "retrieved_at": now, "max_age": timedelta(minutes=2)}
        delayed = {**base, "delivery": Delivery.DELAYED, "delay": timedelta(minutes=15)}
        self.assertEqual(assess_freshness(source_at=now - timedelta(minutes=16), **delayed).label, "Delayed quote")
        self.assertEqual(assess_freshness(source_at=now - timedelta(minutes=17), **delayed).label, "Delayed quote")
        self.assertEqual(assess_freshness(source_at=now - timedelta(minutes=18), **delayed).label, "Stale · delayed quote")
        self.assertEqual(assess_freshness(source_at=now, delivery=Delivery.REAL_TIME, **base).label, "Real-time quote")
        self.assertEqual(assess_freshness(source_at=now, delivery=Delivery.END_OF_DAY, **base).label, "End-of-day quote")
        self.assertEqual(assess_freshness(source_at=now, delivery=Delivery.UNVERIFIED, **base).label, "Quote timing unverified")
        self.assertEqual(assess_freshness(source_at=now, delivery=Delivery.DELAYED, **base).label, "Delayed quote · freshness unverified")
        with self.assertRaises(ValueError):
            assess_freshness(source_at=now, delay=timedelta(minutes=15), **base)
        with self.assertRaises(ValueError):
            assess_freshness(source_at=now, **{**base, "max_age": timedelta()})

    def test_clock_offsets_use_absolute_time_across_dst(self):
        zone = ZoneInfo("America/New_York")
        now = datetime(2026, 11, 1, 1, 15, tzinfo=zone, fold=1)
        source = datetime(2026, 11, 1, 1, 45, tzinfo=zone, fold=0)
        state = assess_freshness(source_at=source, retrieved_at=now,
                                 now=now, max_age=timedelta(minutes=25))
        self.assertIs(state.freshness, Freshness.STALE)
        self.assertEqual(state.age, timedelta(minutes=30))

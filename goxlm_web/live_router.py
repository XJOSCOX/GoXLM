"""Pure routing, exact-query permission, and freshness rules for offline fixtures.

There is no transport and this module is independent of inference and UI state.
"""
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

ROUTER_VERSION = "offline-live-v1"
MAX_QUERY_BYTES = 2048
_TICKERS = {
    "nvidia": "NASDAQ:NVDA", "nvda": "NASDAQ:NVDA",
    "apple": "NASDAQ:AAPL", "aapl": "NASDAQ:AAPL",
    "microsoft": "NASDAQ:MSFT", "msft": "NASDAQ:MSFT",
}
_EXCHANGE_SYMBOL = re.compile(r"\b(?:nasdaq|nyse):[a-z][a-z0-9.-]{0,9}\b")
_SYMBOL = re.compile(r"\$([a-z][a-z0-9.-]{0,9})\b")
_CURRENT = re.compile(r"\b(?:today|latest|current(?:ly)?|live|now|right now|this morning|this week|breaking)\b")
_MARKET = re.compile(r"\b(?:stock|stocks|shares?|share price|ticker|equity|quote|trading|market price|market cap)\b")
_QUOTE = re.compile(r"\b(?:stock price|share price|market quote|stock quote|trading at|quote for|price of)\b")
_NEWS = re.compile(r"\b(?:news|headlines|top stories|breaking stories)\b")
_INFORMATION = re.compile(r"^(?:what|who|when|where|why|how|explain|define|describe|tell me about|can you explain)\b")
_CONCEPT = re.compile(r"^(?:(?:what (?:is|are)|explain|define|describe|tell me about) (?:an? |the )?(?:stock quotes?|stock prices?|financial quotes?|stock markets?|news|headlines|stock splits?|market caps?|quote delays?|delayed quotes?)\s*[?.!]*$|(?:how|why) (?:does|do|are) (?:the )?(?:stock quotes?|stock prices?|financial quotes?|stock markets?|news|headlines|stock splits?|market caps?|quote delays?|delayed quotes?)\b)")
_UPDATES = re.compile(r"\b(?:give|show|fetch|find|get|check|look up|what(?:'s| is)|update)\b")


class Intent(str, Enum):
    CONVERSATION = "conversation"
    TIMELESS = "timeless"
    NEWS = "news"
    MARKET_QUOTE = "market_quote"
    CLARIFY = "clarify"


class Decision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"


class Permission(str, Enum):
    LOCAL = "local"
    DISABLED = "disabled"
    CONFIRM = "confirmation_required"
    DENIED = "denied"
    APPROVED = "approved"


class Freshness(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"
    INVALID = "invalid"


class Delivery(str, Enum):
    REAL_TIME = "real_time"
    DELAYED = "delayed"
    END_OF_DAY = "end_of_day"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class Route:
    intent: Intent
    query: str | None = None
    instrument: str | None = None
    question: str | None = None


@dataclass(frozen=True)
class FreshnessResult:
    freshness: Freshness
    delivery: Delivery | None
    source_at: datetime | None
    retrieved_at: datetime
    age: timedelta | None

    @property
    def label(self) -> str:
        if self.freshness is Freshness.INVALID:
            return "Invalid timestamp"
        if self.freshness is Freshness.UNKNOWN:
            return ("Delayed quote · freshness unverified" if self.delivery is Delivery.DELAYED
                    else "Freshness unverified")
        if self.freshness is Freshness.STALE:
            if self.delivery is Delivery.DELAYED:
                return "Stale · delayed quote"
            return "Stale"
        return {
            None: "Fresh",
            Delivery.REAL_TIME: "Real-time quote",
            Delivery.DELAYED: "Delayed quote",
            Delivery.END_OF_DAY: "End-of-day quote",
            Delivery.UNVERIFIED: "Quote timing unverified",
        }[self.delivery]


def _instruments(normalized: str) -> set[str]:
    exchange_explicit = set()
    remaining = normalized
    for match in _EXCHANGE_SYMBOL.finditer(normalized):
        exchange_explicit.add(match.group().upper())
        remaining = remaining.replace(match.group(), " ")
    result = set(exchange_explicit)
    for term in _SYMBOL.findall(remaining):
        result.add(_TICKERS.get(term, term.upper()))
    for word in re.findall(r"\b[a-z]+\b", remaining):
        if word in _TICKERS:
            result.add(_TICKERS[word])
    return result


def classify(message: str) -> Route:
    """Use visible syntax only. No model, network, or browser dependency."""
    try:
        size = len(message.encode("utf-8"))
    except UnicodeEncodeError:
        size = MAX_QUERY_BYTES + 1
    if size > MAX_QUERY_BYTES:
        return Route(Intent.CLARIFY, question="Please use a shorter question.")
    query = " ".join(message.split())
    if not query:
        return Route(Intent.CLARIFY, question="What would you like to know?")
    text = unicodedata.normalize("NFKC", query).casefold().replace("’", "'")
    if _CONCEPT.search(text) and not _CURRENT.search(text):
        return Route(Intent.TIMELESS)
    instruments = _instruments(text)
    news = bool(_NEWS.search(text) and (
        _CURRENT.search(text) or _UPDATES.search(text) or _NEWS.match(text)
    ))
    market = bool((_MARKET.search(text) or _SYMBOL.search(text) or _EXCHANGE_SYMBOL.search(text)
                   or (instruments and re.search(r"\bprice\b", text))) and (
        _QUOTE.search(text) or _CURRENT.search(text) or _UPDATES.search(text)
        or re.search(r"\b(?:doing|performing|price|quote|up|down)\b", text)
    ))
    if news and market:
        return Route(Intent.CLARIFY, question="Do you want news headlines or a market quote?")
    if news:
        return Route(Intent.NEWS, query)
    if market:
        if len(instruments) != 1:
            return Route(Intent.CLARIFY, question="Which single exchange and ticker should I use?")
        instrument = next(iter(instruments))
        if ":" not in instrument:
            return Route(Intent.CLARIFY, question="What exchange is that ticker listed on?")
        return Route(Intent.MARKET_QUOTE, f"quote {instrument}", instrument)
    if _CURRENT.search(text) and (_INFORMATION.match(text) or _UPDATES.search(text)):
        return Route(Intent.CLARIFY, question="Do you want current news or a market quote?")
    if _INFORMATION.match(text) or query.endswith("?"):
        return Route(Intent.TIMELESS)
    return Route(Intent.CONVERSATION)


def permission(route: Route, *, browsing_enabled: bool = False,
               decision: Decision | None = None, reviewed: Route | None = None) -> Permission:
    """Require opt-in and approval of the exact immutable route and payload."""
    if route.intent not in (Intent.NEWS, Intent.MARKET_QUOTE) or not route.query:
        return Permission.LOCAL
    if browsing_enabled is not True:
        return Permission.DISABLED
    if reviewed != route:
        return Permission.CONFIRM
    if decision is Decision.DENY:
        return Permission.DENIED
    if decision is Decision.APPROVE:
        return Permission.APPROVED
    return Permission.CONFIRM


def assess_freshness(*, source_at: datetime | None, retrieved_at: datetime,
                     now: datetime, max_age: timedelta, delivery: Delivery | None = None,
                     delay: timedelta | None = None) -> FreshnessResult:
    """max_age is the provider/session policy allowance beyond any known delay."""
    if max_age <= timedelta(0):
        raise ValueError("max_age must be positive")
    if delay is not None and (delay <= timedelta(0) or delivery is not Delivery.DELAYED):
        raise ValueError("A positive delay belongs only to delayed delivery")
    times = (now, retrieved_at) if source_at is None else (now, retrieved_at, source_at)
    if any(t.tzinfo is None or t.utcoffset() is None for t in times):
        return FreshnessResult(Freshness.INVALID, delivery, source_at, retrieved_at, None)
    now_utc, retrieved_utc = (t.astimezone(timezone.utc) for t in times[:2])
    source_utc = source_at.astimezone(timezone.utc) if source_at is not None else None
    if retrieved_utc > now_utc or (source_utc is not None and source_utc > retrieved_utc):
        return FreshnessResult(Freshness.INVALID, delivery, source_at, retrieved_at, None)
    if source_utc is None:
        return FreshnessResult(Freshness.UNKNOWN, delivery, source_at, retrieved_at, None)
    age = now_utc - source_utc
    if delivery is Delivery.DELAYED and delay is None:
        return FreshnessResult(Freshness.UNKNOWN, delivery, source_at, retrieved_at, age)
    age_limit = max_age + (delay or timedelta())
    state = Freshness.STALE if age > age_limit else Freshness.FRESH
    return FreshnessResult(state, delivery, source_at, retrieved_at, age)

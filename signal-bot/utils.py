"""
utils.py — Shared utility helpers: logging setup, time formatting,
retry decorators, and generic async helpers.
"""

import asyncio
import logging
import os
import functools
from datetime import datetime
from typing import Any, Callable, TypeVar

import pytz

from config import TIMEZONE, LOG_FILE, MAX_RETRIES, WS_RECONNECT_DELAY

# ─── Logging ─────────────────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)

def setup_logger(name: str = "quotex_bot") -> logging.Logger:
    """Return a logger that writes to both console and the log file."""
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # rotating file
    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


logger = setup_logger()

# ─── Time helpers ─────────────────────────────────────────────────────────────

def now_pkt() -> datetime:
    """Return current datetime in Pakistan Standard Time (UTC+5)."""
    tz = pytz.timezone(TIMEZONE)
    return datetime.now(tz)


def fmt_pkt(dt: datetime | None = None) -> str:
    """Format a datetime as 'YYYY-MM-DD HH:MM:SS PKT'."""
    if dt is None:
        dt = now_pkt()
    elif dt.tzinfo is None:
        dt = pytz.utc.localize(dt).astimezone(pytz.timezone(TIMEZONE))
    return dt.strftime("%Y-%m-%d %H:%M:%S PKT")


def fmt_time_only(dt: datetime | None = None) -> str:
    """Return 'HH:MM' in PKT."""
    if dt is None:
        dt = now_pkt()
    return dt.strftime("%H:%M")


# ─── Retry decorator ─────────────────────────────────────────────────────────

F = TypeVar("F", bound=Callable[..., Any])


def async_retry(
    max_retries: int = MAX_RETRIES,
    delay: float = WS_RECONNECT_DELAY,
    exceptions: tuple = (Exception,),
):
    """Decorator: retry an async function up to *max_retries* times."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s",
                        attempt,
                        max_retries,
                        func.__name__,
                        exc,
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(delay * attempt)
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator


# ─── Misc helpers ─────────────────────────────────────────────────────────────

def clean_pair_name(pair: str) -> str:
    """'EURUSD_otc' → 'EURUSD OTC'"""
    return pair.replace("_", " ").upper()


def base_pair(pair: str) -> str:
    """'EURUSD_otc' → 'EURUSD'"""
    return pair.replace("_otc", "").replace("_OTC", "")


def flag_for_pair(pair: str) -> str:
    """Return a rough country-flag emoji for display purposes."""
    p = base_pair(pair).upper()
    flags = {
        "EURUSD": "🇪🇺🇺🇸",
        "GBPUSD": "🇬🇧🇺🇸",
        "AUDUSD": "🇦🇺🇺🇸",
        "USDJPY": "🇺🇸🇯🇵",
        "EURJPY": "🇪🇺🇯🇵",
        "USDCAD": "🇺🇸🇨🇦",
        "AUDCAD": "🇦🇺🇨🇦",
        "EURGBP": "🇪🇺🇬🇧",
        "BTCUSD": "₿🇺🇸",
        "ETHUSD": "Ξ🇺🇸",
        "XAUUSD": "🥇🇺🇸",
    }
    return flags.get(p, "🌐")


def risk_label(strength: float) -> str:
    from config import RISK_LOW_THRESHOLD, RISK_MEDIUM_THRESHOLD
    if strength > RISK_LOW_THRESHOLD:
        return "🟢 LOW"
    if strength > RISK_MEDIUM_THRESHOLD:
        return "🟡 MEDIUM"
    return "🔴 HIGH"


def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

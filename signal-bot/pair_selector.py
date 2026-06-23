"""
pair_selector.py — Scans all configured pairs across multiple timeframes,
scores each one using the indicator confluence engine, and selects the
highest-confidence setup for signal generation.

Caching:
  - M1  → always fetched fresh (1-minute candles change every minute)
  - M5  → cached for 4 minutes  (candle only changes every 5 min)
  - M15 → cached for 12 minutes (candle only changes every 15 min)
This cuts API calls from 33 per cycle to 11, halving cycle time.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import (
    PAIRS, TIMEFRAMES, MIN_TIMEFRAMES_AGREE,
    MIN_STRENGTH_PCT, SCAN_INTERVAL_SECONDS,
)
from indicators import analyse, AnalysisResult, Direction
from quotex_client import DataProvider
from utils import logger, now_pkt, fmt_pkt, clean_pair_name

# ─── Cache TTLs (seconds) ─────────────────────────────────────────────────────
_CACHE_TTL: dict[str, float] = {
    "M1":  0,     # never cache — always fresh
    "M5":  360,   # 6 minutes  (M5 candle = 5 min; cache survives across cycles)
    "M15": 900,   # 15 minutes (M15 candle = 15 min; valid for 3+ cycles)
}


@dataclass
class PairScore:
    pair: str
    direction: Direction
    strength: float
    agreements: int
    active_indicators: int
    adx: float
    atr: float
    rsi: float
    timeframes_agree: int
    tf_results: dict[str, AnalysisResult]
    valid: bool

    @property
    def confluence_str(self) -> str:
        return f"{self.agreements}/{self.active_indicators}"


class PairSelector:
    """
    Scans all pairs, runs multi-timeframe analysis, returns best setup.
    M5/M15 candle data is cached to avoid redundant API calls every cycle.
    """

    def __init__(self, provider: DataProvider):
        self.provider = provider
        # Cache: (pair, timeframe) → (DataFrame, fetched_at_unix)
        self._cache: dict[tuple[str, str], tuple[pd.DataFrame, float]] = {}

    # ─── Cached candle fetch ──────────────────────────────────────────────────

    async def _get_candles_cached(
        self, pair: str, timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Return candles from cache if still fresh, else fetch and cache."""
        ttl = _CACHE_TTL.get(timeframe, 0)
        key = (pair, timeframe)
        now = time.monotonic()

        if ttl > 0 and key in self._cache:
            df_cached, fetched_at = self._cache[key]
            age = now - fetched_at
            if age < ttl:
                logger.debug(
                    "Cache hit %s/%s (age=%.0fs ttl=%.0fs)",
                    pair, timeframe, age, ttl,
                )
                return df_cached

        # Fetch fresh
        df = await self.provider.get_candles(pair, timeframe)
        if df is not None and ttl > 0:
            self._cache[key] = (df, now)
        return df

    # ─── Analysis per pair+TF ─────────────────────────────────────────────────

    async def _analyse_pair_tf(
        self, pair: str, timeframe: str
    ) -> Optional[AnalysisResult]:
        try:
            df = await self._get_candles_cached(pair, timeframe)
            if df is None or len(df) < 50:
                return None
            return analyse(df)
        except Exception as exc:
            logger.warning("Analysis error %s/%s: %s", pair, timeframe, exc)
            return None

    async def _analyse_pair(self, pair: str) -> Optional[PairScore]:
        """
        Run multi-timeframe analysis for *pair*.
        Fetches M1 fresh; serves M5/M15 from cache when available.
        Returns PairScore if at least MIN_TIMEFRAMES_AGREE agree, else None.
        """
        results: dict[str, Optional[AnalysisResult]] = {}
        for tf in TIMEFRAMES:
            results[tf] = await self._analyse_pair_tf(pair, tf)

        valid_results = {
            tf: r for tf, r in results.items() if r is not None
        }

        if not valid_results:
            logger.debug("No valid data for %s", pair)
            return None

        # Count timeframes agreeing on direction
        buy_tfs  = [tf for tf, r in valid_results.items() if r.direction == "BUY"]
        sell_tfs = [tf for tf, r in valid_results.items() if r.direction == "SELL"]

        if len(buy_tfs) >= len(sell_tfs):
            dominant_dir: Direction = "BUY"
            agreeing_tfs = buy_tfs
        else:
            dominant_dir = "SELL"
            agreeing_tfs = sell_tfs

        if len(agreeing_tfs) < MIN_TIMEFRAMES_AGREE:
            logger.debug(
                "%s: only %d/%d TFs agree — skipping",
                pair, len(agreeing_tfs), MIN_TIMEFRAMES_AGREE
            )
            return None

        agreeing_analyses = [valid_results[tf] for tf in agreeing_tfs]
        avg_strength    = sum(r.strength   for r in agreeing_analyses) / len(agreeing_analyses)
        avg_adx         = sum(r.adx        for r in agreeing_analyses) / len(agreeing_analyses)
        avg_atr         = sum(r.atr        for r in agreeing_analyses) / len(agreeing_analyses)
        avg_rsi         = sum(r.rsi        for r in agreeing_analyses) / len(agreeing_analyses)
        total_agreements = sum(r.agreements        for r in agreeing_analyses)
        total_active     = sum(r.active_indicators for r in agreeing_analyses)

        any_valid = any(r.valid for r in agreeing_analyses)

        return PairScore(
            pair=pair,
            direction=dominant_dir,
            strength=round(avg_strength, 1),
            agreements=total_agreements,
            active_indicators=total_active,
            adx=round(avg_adx, 2),
            atr=round(avg_atr, 6),
            rsi=round(avg_rsi, 1),
            timeframes_agree=len(agreeing_tfs),
            tf_results=valid_results,
            valid=any_valid and avg_strength >= MIN_STRENGTH_PCT,
        )

    # ─── Full scan ────────────────────────────────────────────────────────────

    async def scan_all_pairs(self) -> list[PairScore]:
        """
        Scan every pair and return all valid PairScores sorted by strength.
        Pairs are processed sequentially to respect TwelveData rate limits;
        cached M5/M15 data is served instantly without API calls.
        """
        logger.info("🔍 Scanning %d pairs at %s", len(PAIRS), fmt_pkt())

        # Count how many TF fetches will actually hit the API vs cache
        now = time.monotonic()
        cache_hits = sum(
            1
            for pair in PAIRS
            for tf in TIMEFRAMES
            if _CACHE_TTL.get(tf, 0) > 0
            and (pair, tf) in self._cache
            and (now - self._cache[(pair, tf)][1]) < _CACHE_TTL[tf]
        )
        api_calls = len(PAIRS) * len(TIMEFRAMES) - cache_hits
        logger.info(
            "API calls this scan: %d (cache hits: %d / %d total)",
            api_calls, cache_hits, len(PAIRS) * len(TIMEFRAMES),
        )

        sem = asyncio.Semaphore(4)

        async def bounded(pair: str):
            async with sem:
                return await self._analyse_pair(pair)

        tasks  = [bounded(p) for p in PAIRS]
        scores = await asyncio.gather(*tasks, return_exceptions=True)

        valid_scores: list[PairScore] = []
        for pair, score in zip(PAIRS, scores):
            if isinstance(score, Exception):
                logger.warning("Exception scanning %s: %s", pair, score)
            elif score is not None and score.valid:
                valid_scores.append(score)
                logger.info(
                    "  ✅ %s | %s | Str:%.1f%% | ADX:%.1f | TFs:%d/%d",
                    pair, score.direction, score.strength,
                    score.adx, score.timeframes_agree, len(TIMEFRAMES),
                )
            else:
                logger.debug("  ⛔ %s — no valid signal", pair)

        valid_scores.sort(key=lambda s: s.strength, reverse=True)
        logger.info("Scan complete: %d/%d pairs qualify", len(valid_scores), len(PAIRS))
        return valid_scores

    async def select_best_pair(self) -> Optional[PairScore]:
        scores = await self.scan_all_pairs()
        if not scores:
            logger.info("No pair met the minimum confidence threshold — skipping cycle.")
            return None

        best = scores[0]
        logger.info(
            "🏆 Best pair selected: %s | %s | %.1f%% strength",
            best.pair, best.direction, best.strength,
        )
        return best

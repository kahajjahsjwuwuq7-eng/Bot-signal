"""
pair_selector.py — Scans all configured pairs across multiple timeframes,
scores each one using the indicator confluence engine, and selects the
highest-confidence setup for signal generation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from config import (
    PAIRS, TIMEFRAMES, MIN_TIMEFRAMES_AGREE,
    MIN_STRENGTH_PCT, SCAN_INTERVAL_SECONDS,
)
from indicators import analyse, AnalysisResult, Direction
from quotex_client import DataProvider
from utils import logger, now_pkt, fmt_pkt, clean_pair_name


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
    """

    def __init__(self, provider: DataProvider):
        self.provider = provider

    async def _analyse_pair_tf(
        self, pair: str, timeframe: str
    ) -> Optional[AnalysisResult]:
        """Fetch candles and run analysis for a single pair+timeframe."""
        try:
            df = await self.provider.get_candles(pair, timeframe)
            if df is None or len(df) < 50:
                return None
            result = analyse(df)
            return result
        except Exception as exc:
            logger.warning("Analysis error %s/%s: %s", pair, timeframe, exc)
            return None

    async def _analyse_pair(self, pair: str) -> Optional[PairScore]:
        """
        Run multi-timeframe analysis for *pair*.
        Returns PairScore if at least MIN_TIMEFRAMES_AGREE agree, else None.
        """
        # Fetch all timeframes concurrently
        tasks = {tf: self._analyse_pair_tf(pair, tf) for tf in TIMEFRAMES}
        results: dict[str, Optional[AnalysisResult]] = {}
        for tf, coro in tasks.items():
            results[tf] = await coro

        valid_results = {
            tf: r for tf, r in results.items() if r is not None
        }

        if not valid_results:
            logger.debug("No valid data for %s", pair)
            return None

        # Count how many timeframes agree on direction
        buy_tfs = [tf for tf, r in valid_results.items() if r.direction == "BUY"]
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

        # Average strength across agreeing timeframes
        agreeing_analyses = [valid_results[tf] for tf in agreeing_tfs]
        avg_strength = sum(r.strength for r in agreeing_analyses) / len(agreeing_analyses)
        avg_adx = sum(r.adx for r in agreeing_analyses) / len(agreeing_analyses)
        avg_atr = sum(r.atr for r in agreeing_analyses) / len(agreeing_analyses)
        avg_rsi = sum(r.rsi for r in agreeing_analyses) / len(agreeing_analyses)
        total_agreements = sum(r.agreements for r in agreeing_analyses)
        total_active = sum(r.active_indicators for r in agreeing_analyses)

        # Overall signal validity: at least one TF must pass all filters
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

    async def scan_all_pairs(self) -> list[PairScore]:
        """
        Scan every pair concurrently and return all valid PairScores
        sorted by strength descending.
        """
        logger.info("🔍 Scanning %d pairs at %s", len(PAIRS), fmt_pkt())

        # Run pair analyses concurrently (limit concurrency to avoid API rate limits)
        sem = asyncio.Semaphore(4)

        async def bounded(pair):
            async with sem:
                return await self._analyse_pair(pair)

        tasks = [bounded(p) for p in PAIRS]
        scores = await asyncio.gather(*tasks, return_exceptions=True)

        valid_scores = []
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
        logger.info(
            "Scan complete: %d/%d pairs qualify", len(valid_scores), len(PAIRS)
        )
        return valid_scores

    async def select_best_pair(self) -> Optional[PairScore]:
        """
        Return the single best qualifying pair, or None if nothing qualifies.
        """
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

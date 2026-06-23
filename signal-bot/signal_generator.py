"""
signal_generator.py — Orchestrates the full signal lifecycle:
  1. Send signal (15s before candle opens)
  2. Wait 75s (15s prep + 60s M1 expiry)
  3. Determine WIN / LOSS / DRAW
  4. Send result message
  5. Update stats
"""

from __future__ import annotations

import asyncio
from typing import Optional, Callable, Awaitable

from config import (
    PRIMARY_TIMEFRAME, TIMEFRAME_SECONDS, DEFAULT_STAKE,
)
from pair_selector import PairScore
from quotex_client import DataProvider
from tracker import StatsTracker
from utils import logger, now_pkt, fmt_time_only

SendFn = Callable[[str], Awaitable[None]]

# Unicode bold italic header strings (as user requested)
_HEADER   = "𝙌𝙐𝙊𝙏𝙀𝙓 𝘽𝙍𝙊𝙆𝙀𝙍"
_AVOID    = "⚡️  𝘼𝙑𝙊𝙄𝘿 𝘿𝙊𝙅𝙄 𝘾𝘼𝙉𝘿𝙇𝙀𝙎  ⚡️"

# Seconds the user has to place the trade before candle opens
PREP_SECONDS = 15


def _fmt_signal_pair(pair: str) -> str:
    """'EURUSD_otc' → 'EURUSD OTC'"""
    return pair.replace("_otc", " OTC").replace("_OTC", " OTC").upper()


def _fmt_result_pair(pair: str) -> str:
    """'EURUSD_otc' → 'EURUSD'"""
    return pair.replace("_otc", "").replace("_OTC", "").upper()


class SignalGenerator:
    def __init__(
        self,
        provider: DataProvider,
        tracker: StatsTracker,
        send_fn: SendFn,
        timeframe: str = PRIMARY_TIMEFRAME,
        stake: float = DEFAULT_STAKE,
    ):
        self.provider  = provider
        self.tracker   = tracker
        self.send      = send_fn
        self.timeframe = timeframe
        self.stake     = stake

    # ─── Signal message ───────────────────────────────────────────────────────

    def _build_signal_message(self, score: PairScore) -> str:
        time_str     = fmt_time_only(now_pkt())
        pair_display = _fmt_signal_pair(score.pair)

        if score.direction == "BUY":
            direction_line = "💀 GO FOR UP 🔼"
        else:
            direction_line = "💀 GO FOR DOWN 🔽"

        lines = [
            _HEADER,
            "",
            f"📊 {pair_display}",
            "",
            f"✔️ Time  : {time_str} (PKT 🇵🇰)",
            f"⏳ Frame : {self.timeframe}",
            "",
            direction_line,
            "",
            _AVOID,
        ]
        return "\n".join(lines)

    async def send_signal(self, score: PairScore) -> None:
        msg = self._build_signal_message(score)
        await self.send(msg)
        await self.tracker.record_signal()
        logger.info(
            "Signal sent: %s | %s | %.1f%%",
            score.pair, score.direction, score.strength,
        )

    # ─── Wait for expiry ─────────────────────────────────────────────────────

    async def wait_expiry(self) -> None:
        """Wait PREP_SECONDS + candle duration so result is after candle close."""
        candle_secs = TIMEFRAME_SECONDS.get(self.timeframe, 60)
        total = PREP_SECONDS + candle_secs
        logger.info(
            "Waiting %ds (%ds prep + %ds candle) for result…",
            total, PREP_SECONDS, candle_secs,
        )
        await asyncio.sleep(total)

    # ─── Determine result ────────────────────────────────────────────────────

    async def determine_result(
        self, score: PairScore, entry_price: Optional[float]
    ) -> str:
        if entry_price is None:
            logger.warning("No entry price — marking DRAW")
            return "DRAW"

        try:
            exit_price = await self.provider.get_current_price(score.pair)
            if exit_price is None:
                logger.warning("No exit price for %s — marking DRAW", score.pair)
                return "DRAW"

            diff = exit_price - entry_price
            pct  = abs(diff) / entry_price * 100

            if pct < 0.001:
                result = "DRAW"
            elif score.direction == "BUY":
                result = "WIN" if diff > 0 else "LOSS"
            else:
                result = "WIN" if diff < 0 else "LOSS"

            logger.info(
                "Result: %s | entry=%.6f exit=%.6f diff=%.6f (%.4f%%)",
                result, entry_price, exit_price, diff, pct,
            )
            return result

        except Exception as exc:
            logger.error("Error determining result: %s", exc)
            return "DRAW"

    # ─── Send result ─────────────────────────────────────────────────────────

    async def send_result(self, score: PairScore, result: str) -> None:
        snap = await self.tracker.record_result(
            result, score.pair, score.direction, self.stake,
        )
        msg = self.tracker.build_result_message(
            result, score.pair, score.direction, self.stake,
        )
        await self.send(msg)
        logger.info(
            "Result sent: %s | WR:%.1f%% | P&L:$%.2f",
            result, snap["win_rate"], snap["pnl"],
        )

    # ─── Full lifecycle ───────────────────────────────────────────────────────

    async def run_signal_cycle(self, score: PairScore) -> None:
        logger.info("=== Signal cycle start: %s ===", score.pair)

        # 1. Capture entry price from scan result (no extra API call)
        entry_price: Optional[float] = None
        primary_result = score.tf_results.get(self.timeframe) or next(
            iter(score.tf_results.values()), None
        )
        if primary_result and primary_result.close > 0:
            entry_price = primary_result.close
            logger.info("Entry price for %s from scan: %.6f", score.pair, entry_price)
        else:
            for attempt in range(3):
                entry_price = await self.provider.get_current_price(score.pair)
                if entry_price is not None:
                    break
                if attempt < 2:
                    await asyncio.sleep(8)
            logger.info("Entry price for %s (fetched): %s", score.pair, entry_price)

        # 2. Send signal immediately (user has PREP_SECONDS to place trade)
        await self.send_signal(score)

        # 3. Wait prep + candle duration
        await self.wait_expiry()

        # 4. Determine result
        result = await self.determine_result(score, entry_price)

        # 5. Send result
        await self.send_result(score, result)

        logger.info("=== Signal cycle complete: %s → %s ===", score.pair, result)

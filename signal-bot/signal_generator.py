"""
signal_generator.py — Orchestrates the full signal lifecycle:
  1. Announce best pair
  2. Countdown
  3. Send formatted signal
  4. Wait for candle expiry
  5. Determine WIN / LOSS / DRAW
  6. Send result message
  7. Update stats
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
from utils import (
    logger, now_pkt, fmt_time_only, fmt_pkt,
    clean_pair_name, flag_for_pair, risk_label,
)

# Type alias for a "send message" callback
SendFn = Callable[[str], Awaitable[None]]


class SignalGenerator:
    """
    Drives the full signal lifecycle for a given PairScore.
    Calls *send_fn* to push messages to Telegram.
    """

    def __init__(
        self,
        provider: DataProvider,
        tracker: StatsTracker,
        send_fn: SendFn,
        timeframe: str = PRIMARY_TIMEFRAME,
        stake: float = DEFAULT_STAKE,
    ):
        self.provider = provider
        self.tracker = tracker
        self.send = send_fn
        self.timeframe = timeframe
        self.stake = stake

    # ─── Step 1: Announce ────────────────────────────────────────────────────

    async def announce_pair(self, score: PairScore) -> None:
        """Send the 'best pair selected' announcement."""
        pair_display = clean_pair_name(score.pair)
        msg = (
            f"⚠️ *{pair_display}* selected as BEST pair\n"
            f"🔥 Confidence: *{score.strength:.0f}%*\n"
            f"📡 Direction: *{score.direction}*\n"
            f"⏳ Preparing signal..."
        )
        await self.send(msg)
        logger.info("Announced pair: %s | %s | %.1f%%", score.pair, score.direction, score.strength)

    # ─── Step 2: Countdown ───────────────────────────────────────────────────

    async def countdown(self, seconds: int = 10) -> None:
        """Send a single countdown message (10 → 1)."""
        nums = " ".join(str(i) for i in range(seconds, 0, -1))
        await self.send(f"⏱ *Signal in:*\n`{nums}`")
        await asyncio.sleep(seconds)

    # ─── Step 3: Signal message ──────────────────────────────────────────────

    @staticmethod
    def _escape(text: str) -> str:
        """Escape Markdown special chars in dynamic text."""
        for ch in ("_", "*", "[", "]", "`"):
            text = text.replace(ch, f"\\{ch}")
        return text

    def _build_signal_message(self, score: PairScore) -> str:
        now = now_pkt()
        time_str = fmt_time_only(now)
        pair_display = clean_pair_name(score.pair)

        if score.direction == "BUY":
            direction_line = "💀 *GO FOR UP* 🔼"
        else:
            direction_line = "💀 *GO FOR DOWN* 🔽"

        risk = risk_label(score.strength)
        confluence = f"{score.agreements}/{score.active_indicators}"
        structure = list(score.tf_results.values())[0].market_structure

        lines = [
            f"📊 *{pair_display}*",
            "",
            f"✔️ Time  : {time_str} (PKT 🇵🇰)",
            f"⏳ Frame : {self.timeframe}",
            "",
            direction_line,
            "",
            "⚡️ AVOID DOJI CANDLES ⚡️",
            "📺 NON SIGNAL MTG",
            "",
            f"🔥 Strength   : *{score.strength:.0f}%*",
            f"📊 Confluence : {confluence}",
            f"🎯 Risk Level : {risk}",
            f"📈 ADX        : {score.adx:.1f}",
            f"📡 TFs Agree  : {score.timeframes_agree}/{len(score.tf_results)}",
            f"🏗 Structure  : {self._escape(structure)}",
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

    # ─── Step 4: Wait for expiry ─────────────────────────────────────────────

    async def wait_expiry(self) -> None:
        wait_secs = TIMEFRAME_SECONDS.get(self.timeframe, 60)
        logger.info("Waiting %ds for candle expiry (%s)…", wait_secs, self.timeframe)
        await asyncio.sleep(wait_secs)

    # ─── Step 5: Determine result ────────────────────────────────────────────

    async def determine_result(
        self, score: PairScore, entry_price: Optional[float]
    ) -> str:
        """
        Fetch current price after expiry and compare to *entry_price*.
        Returns 'WIN', 'LOSS', or 'DRAW'.
        """
        if entry_price is None:
            logger.warning("No entry price recorded — marking DRAW")
            return "DRAW"

        try:
            exit_price = await self.provider.get_current_price(score.pair)
            if exit_price is None:
                logger.warning("Could not get exit price for %s — marking DRAW", score.pair)
                return "DRAW"

            diff = exit_price - entry_price
            pct = abs(diff) / entry_price * 100

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

    # ─── Step 6: Send result ─────────────────────────────────────────────────

    async def send_result(
        self, score: PairScore, result: str
    ) -> None:
        snap = await self.tracker.record_result(
            result,           # type: ignore[arg-type]
            score.pair,
            score.direction,
            self.stake,
        )
        msg = self.tracker.build_result_message(
            result,           # type: ignore[arg-type]
            score.pair,
            score.direction,
            self.stake,
        )
        await self.send(msg)
        logger.info(
            "Result sent: %s | WR:%.1f%% | P&L:$%.2f",
            result, snap["win_rate"], snap["pnl"],
        )

    # ─── Full lifecycle ──────────────────────────────────────────────────────

    async def run_signal_cycle(self, score: PairScore) -> None:
        """Execute the full signal lifecycle for *score*."""
        logger.info("=== Signal cycle start: %s ===", score.pair)

        # 1. Capture entry price — use the close already in the scan result (no extra API call).
        #    Fall back to a fresh fetch only if the scan close is missing.
        entry_price: Optional[float] = None
        primary_result = score.tf_results.get(self.timeframe) or next(
            iter(score.tf_results.values()), None
        )
        if primary_result and primary_result.close > 0:
            entry_price = primary_result.close
            logger.info("Entry price for %s from scan: %.6f", score.pair, entry_price)
        else:
            # Fallback: fresh API call with a short back-off
            for attempt in range(3):
                entry_price = await self.provider.get_current_price(score.pair)
                if entry_price is not None:
                    break
                if attempt < 2:
                    await asyncio.sleep(8)
            logger.info("Entry price for %s (fetched): %s", score.pair, entry_price)

        # 2. Announce
        await self.announce_pair(score)
        await asyncio.sleep(2)

        # 3. Countdown
        await self.countdown(10)

        # 4. Send signal
        await self.send_signal(score)

        # 5. Wait expiry
        await self.wait_expiry()

        # 6. Determine result
        result = await self.determine_result(score, entry_price)

        # 7. Send result
        await self.send_result(score, result)

        logger.info("=== Signal cycle complete: %s → %s ===", score.pair, result)

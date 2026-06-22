"""
tracker.py — Persistent win/loss/draw statistics tracker.
Results are stored in stats.json and updated after every trade.
"""

import json
import asyncio
import os
from dataclasses import dataclass, field, asdict
from typing import Literal

from config import STATS_FILE, DEFAULT_STAKE
from utils import logger, now_pkt, fmt_pkt

Result = Literal["WIN", "LOSS", "DRAW"]


@dataclass
class Stats:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_signals: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    @property
    def total_trades(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return round(self.wins / self.total_trades * 100, 1)

    @property
    def pnl(self) -> float:
        return round(self.gross_profit - self.gross_loss, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["win_rate"] = self.win_rate
        d["pnl"] = self.pnl
        d["total_trades"] = self.total_trades
        return d


class StatsTracker:
    """Thread-safe async stats tracker backed by a JSON file."""

    def __init__(self, filepath: str = STATS_FILE):
        self.filepath = filepath
        self._stats = Stats()
        self._lock = asyncio.Lock()
        self._load()

    # ─── Persistence ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                self._stats = Stats(
                    wins=data.get("wins", 0),
                    losses=data.get("losses", 0),
                    draws=data.get("draws", 0),
                    total_signals=data.get("total_signals", 0),
                    gross_profit=data.get("gross_profit", 0.0),
                    gross_loss=data.get("gross_loss", 0.0),
                )
                logger.info("Stats loaded: %s", self._stats.to_dict())
            except Exception as exc:
                logger.error("Failed to load stats: %s — starting fresh", exc)
                self._stats = Stats()

    async def _save(self) -> None:
        try:
            with open(self.filepath, "w") as f:
                json.dump(self._stats.to_dict(), f, indent=2)
        except Exception as exc:
            logger.error("Failed to save stats: %s", exc)

    # ─── Public API ──────────────────────────────────────────────────────────

    async def record_signal(self) -> None:
        """Increment total signal counter without recording a trade result."""
        async with self._lock:
            self._stats.total_signals += 1
            await self._save()

    async def record_result(
        self,
        result: Result,
        pair: str,
        direction: str,
        stake: float = DEFAULT_STAKE,
    ) -> dict:
        """
        Record a trade outcome and return a dict with updated stats
        plus a formatted result message string.
        """
        async with self._lock:
            if result == "WIN":
                self._stats.wins += 1
                self._stats.gross_profit += stake
            elif result == "LOSS":
                self._stats.losses += 1
                self._stats.gross_loss += stake
            else:
                self._stats.draws += 1

            await self._save()

            snap = self._stats.to_dict()
            snap["result"] = result
            snap["pair"] = pair
            snap["direction"] = direction
            snap["stake"] = stake
            snap["timestamp"] = fmt_pkt()
            logger.info(
                "Result recorded: %s | %s | %s | W:%d L:%d D:%d WR:%.1f%%",
                result, pair, direction,
                self._stats.wins, self._stats.losses, self._stats.draws,
                self._stats.win_rate,
            )
            return snap

    def get_stats(self) -> dict:
        return self._stats.to_dict()

    # ─── Telegram message builders ───────────────────────────────────────────

    def build_result_message(
        self,
        result: Result,
        pair: str,
        direction: str,
        stake: float = DEFAULT_STAKE,
    ) -> str:
        s = self._stats
        dir_emoji = "🟢 BUY" if direction.upper() == "BUY" else "🔴 SELL"

        if result == "WIN":
            header = f"✅ WIN! +${stake:.2f}"
        elif result == "LOSS":
            header = f"❌ LOSS! -${stake:.2f}"
        else:
            header = "➖ DRAW"

        lines = [
            header,
            f"📌 {pair} | {dir_emoji}",
            f"📊 Win Rate: {s.win_rate}% ({s.wins}W / {s.losses}L)",
            f"💰 Total P&L: ${s.pnl:.2f}",
        ]
        return "\n".join(lines)

    def build_stats_message(self) -> str:
        s = self._stats
        return (
            "📈 *BOT STATISTICS*\n"
            f"───────────────────\n"
            f"📊 Total Signals : {s.total_signals}\n"
            f"✅ Wins          : {s.wins}\n"
            f"❌ Losses        : {s.losses}\n"
            f"➖ Draws         : {s.draws}\n"
            f"🎯 Win Rate      : {s.win_rate}%\n"
            f"💰 Gross Profit  : ${s.gross_profit:.2f}\n"
            f"💸 Gross Loss    : ${s.gross_loss:.2f}\n"
            f"📉 Net P&L       : ${s.pnl:.2f}\n"
            f"───────────────────\n"
            f"🕐 {fmt_pkt()}"
        )

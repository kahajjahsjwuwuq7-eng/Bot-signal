"""
telegram_bot.py — Main entry point.

Wires together:
  • python-telegram-bot 20.x (command handlers, broadcast)
  • PairSelector (multi-timeframe confluence scanner)
  • SignalGenerator (full signal lifecycle)
  • StatsTracker (persistent win/loss/P&L)
  • DataProvider (Quotex WS → TwelveData → AV → Finnhub)

Run: python telegram_bot.py
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time as _time
from typing import Optional

from telegram import Update, Bot
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)
from telegram.error import TelegramError

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_CHANNEL_ID,
    PAIRS,
    PRIMARY_TIMEFRAME,
    SCAN_INTERVAL_SECONDS,
    DEFAULT_STAKE,
    STATS_FILE,
)
from quotex_client import DataProvider
from pair_selector import PairSelector
from signal_generator import SignalGenerator
from tracker import StatsTracker
from utils import logger, now_pkt, fmt_pkt, clean_pair_name, risk_label

# ─── Subscriber list ─────────────────────────────────────────────────────────
# Loaded from / persisted to subscribers.json
SUBSCRIBERS_FILE = "subscribers.json"


def load_subscribers() -> set[str]:
    if os.path.exists(SUBSCRIBERS_FILE):
        try:
            with open(SUBSCRIBERS_FILE) as f:
                return set(json.load(f))
        except Exception:
            pass
    # Default to configured IDs
    subs = set()
    if TELEGRAM_CHAT_ID:
        subs.add(TELEGRAM_CHAT_ID)
    if TELEGRAM_CHANNEL_ID:
        subs.add(TELEGRAM_CHANNEL_ID)
    return subs


def save_subscribers(subs: set[str]) -> None:
    try:
        with open(SUBSCRIBERS_FILE, "w") as f:
            json.dump(list(subs), f, indent=2)
    except Exception as exc:
        logger.error("Failed to save subscribers: %s", exc)


subscribers: set[str] = load_subscribers()

# ─── Global shared state ──────────────────────────────────────────────────────
provider = DataProvider()
tracker = StatsTracker(STATS_FILE)
selector: Optional[PairSelector] = None
generator: Optional[SignalGenerator] = None
bot_app: Optional[Application] = None
scan_task: Optional[asyncio.Task] = None
_running = True


# ─── Broadcast helper ─────────────────────────────────────────────────────────

async def broadcast(text: str, parse_mode: str = ParseMode.MARKDOWN) -> None:
    """Send *text* to all subscribers."""
    if bot_app is None:
        logger.warning("bot_app not ready — skipping broadcast")
        return

    targets = set(subscribers)
    if TELEGRAM_CHAT_ID:
        targets.add(TELEGRAM_CHAT_ID)
    if TELEGRAM_CHANNEL_ID:
        targets.add(TELEGRAM_CHANNEL_ID)

    for chat_id in targets:
        try:
            await bot_app.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
            )
        except TelegramError as exc:
            logger.warning("Telegram send error to %s: %s", chat_id, exc)


# ─── Command handlers ─────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    name = user.first_name if user else "Trader"
    msg = (
        f"👋 *Welcome, {name}!*\n\n"
        f"🤖 *Advanced Quotex Signal Bot*\n"
        f"📡 Multi-API • 20+ Indicators • Auto Pair Selection\n\n"
        f"I scan *{len(PAIRS)} OTC pairs* every minute using:\n"
        f"• RSI, MACD, Stoch, Bollinger, EMA, ADX\n"
        f"• Williams %R, CCI, SAR, ATR, OBV\n"
        f"• Chart patterns, Candlestick analysis\n"
        f"• Multi-timeframe (M1/M5/M15) confirmation\n\n"
        f"📊 Use /help to see all commands\n"
        f"🟢 Bot is running — signals will arrive automatically!"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "📋 *BOT COMMANDS*\n"
        "─────────────────────\n"
        "/start       — Welcome message\n"
        "/help        — This help menu\n"
        "/status      — Bot health & data sources\n"
        "/signal      — Force immediate scan & signal\n"
        "/subscribe   — Add this chat to broadcasts\n"
        "/unsubscribe — Remove this chat from broadcasts\n"
        "/stats       — Win/loss statistics\n"
        "/pairs       — List all monitored pairs\n"
        "/settings    — Current bot configuration\n"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    tf_status = "✅ Running" if scan_task and not scan_task.done() else "⚠️ Stopped"
    s = tracker.get_stats()
    msg = (
        f"🤖 *BOT STATUS*\n"
        f"─────────────────────\n"
        f"🕐 Time      : {fmt_pkt()}\n"
        f"🔄 Scanner   : {tf_status}\n"
        f"📡 Sources   : Quotex → TwelveData → AV → Finnhub\n"
        f"📊 Signals   : {s['total_signals']}\n"
        f"🎯 Win Rate  : {s['win_rate']}%\n"
        f"💰 P&L       : ${s['pnl']:.2f}\n"
        f"👥 Subscribers: {len(subscribers)}\n"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Force an immediate scan and signal generation."""
    await update.message.reply_text(
        "🔍 *Forcing immediate scan...*", parse_mode=ParseMode.MARKDOWN
    )
    try:
        score = await selector.select_best_pair()
        if score is None:
            await update.message.reply_text(
                "⚠️ No qualifying pair found right now. Try again in a minute.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        # Run cycle (will broadcast to all subscribers)
        asyncio.create_task(generator.run_signal_cycle(score))
    except Exception as exc:
        logger.error("Force signal error: %s", exc)
        await update.message.reply_text(f"❌ Error: {exc}")


async def cmd_subscribe(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cid = str(update.effective_chat.id)
    if cid in subscribers:
        await update.message.reply_text("✅ You are already subscribed!")
    else:
        subscribers.add(cid)
        save_subscribers(subscribers)
        await update.message.reply_text(
            "✅ *Subscribed!* You will now receive all signals.",
            parse_mode=ParseMode.MARKDOWN,
        )


async def cmd_unsubscribe(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cid = str(update.effective_chat.id)
    if cid in subscribers:
        subscribers.discard(cid)
        save_subscribers(subscribers)
        await update.message.reply_text("✅ Unsubscribed. You won't receive signals anymore.")
    else:
        await update.message.reply_text("ℹ️ You were not subscribed.")


async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = tracker.build_stats_message()
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_pairs(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    lines = ["📊 *MONITORED PAIRS*", "─────────────────────"]
    for i, p in enumerate(PAIRS, 1):
        lines.append(f"{i:2}. `{p}`")
    lines.append(f"\n_Scanned every {SCAN_INTERVAL_SECONDS}s_")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_settings(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    from config import (
        RSI_PERIOD, MACD_FAST, MACD_SLOW, BB_PERIOD,
        ADX_PERIOD, MIN_STRENGTH_PCT, MIN_INDICATOR_AGREEMENTS,
        MIN_TIMEFRAMES_AGREE, TIMEFRAMES,
    )
    msg = (
        "⚙️ *BOT SETTINGS*\n"
        "─────────────────────\n"
        f"📊 Pairs         : {len(PAIRS)}\n"
        f"⏳ Timeframes    : {', '.join(TIMEFRAMES)}\n"
        f"🕐 Signal TF     : {PRIMARY_TIMEFRAME}\n"
        f"🔄 Scan interval : {SCAN_INTERVAL_SECONDS}s\n"
        f"💰 Stake         : ${DEFAULT_STAKE:.2f}\n"
        f"─────────────────────\n"
        f"📈 RSI Period    : {RSI_PERIOD}\n"
        f"📈 MACD          : {MACD_FAST}/{MACD_SLOW}\n"
        f"📈 BB Period     : {BB_PERIOD}\n"
        f"📈 ADX Period    : {ADX_PERIOD}\n"
        f"─────────────────────\n"
        f"🎯 Min Strength  : {MIN_STRENGTH_PCT}%\n"
        f"🎯 Min Agreements: {MIN_INDICATOR_AGREEMENTS}\n"
        f"🎯 Min TFs Agree : {MIN_TIMEFRAMES_AGREE}\n"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


# ─── Candle-boundary timing ───────────────────────────────────────────────────

SIGNAL_EARLY_SECS = 15   # send signal this many seconds before candle opens

def _secs_until_next_signal() -> float:
    """
    Return seconds to sleep so we wake up exactly SIGNAL_EARLY_SECS before
    the next UTC minute boundary (i.e. at HH:MM:45 UTC for M1 candles).
    """
    now_utc   = _time.time()
    secs_in   = now_utc % 60                        # how far into this minute
    fire_at   = 60 - SIGNAL_EARLY_SECS              # :45 within the minute
    if secs_in < fire_at:
        return fire_at - secs_in                    # still before :45 this min
    else:
        return 60 - secs_in + fire_at               # past :45 — wait for next


def _next_candle_open_pkt() -> str:
    """
    Return the next UTC minute boundary formatted as HH:MM PKT.
    This is the candle open time the user should enter the trade on.
    """
    import math
    from datetime import datetime, timezone
    import pytz
    now_utc       = _time.time()
    secs_in       = now_utc % 60
    next_open_utc = now_utc + (60 - secs_in)        # next :00 UTC
    dt_utc        = datetime.fromtimestamp(next_open_utc, tz=timezone.utc)
    dt_pkt        = dt_utc.astimezone(pytz.timezone("Asia/Karachi"))
    return dt_pkt.strftime("%H:%M")


# ─── Auto-scan loop ───────────────────────────────────────────────────────────

async def scan_loop() -> None:
    """
    Candle-aligned scan loop.
    Wakes at :45 of each UTC minute (15 s before the M1 candle opens),
    scans all pairs, and fires the signal so the user can enter right at
    the candle open (:00 of the next minute).
    """
    global _running
    logger.info("🚀 Scan loop started at %s", fmt_pkt())
    await broadcast(
        "🟢 *Quotex Signal Bot STARTED*\n"
        f"📡 Scanning {len(PAIRS)} OTC pairs — aligned to candle clock\n"
        f"🕐 {fmt_pkt()}"
    )

    while _running:
        # ── Wait until 15 s before next candle open ──────────────────────────
        sleep_secs = _secs_until_next_signal()
        logger.debug("Sleeping %.1fs until next signal window…", sleep_secs)
        await asyncio.sleep(sleep_secs)

        if not _running:
            break

        logger.info("🔍 Scanning 11 pairs at %s", fmt_pkt())

        try:
            score = await selector.select_best_pair()

            if score is not None:
                # Pass the candle open time so the signal message shows it
                await generator.run_signal_cycle(
                    score, candle_open_time=_next_candle_open_pkt()
                )
            else:
                logger.info("No qualifying pair this cycle — waiting…")

        except asyncio.CancelledError:
            logger.info("Scan loop cancelled.")
            break
        except Exception as exc:
            logger.error("Scan loop error: %s", exc, exc_info=True)
            await asyncio.sleep(5)

    logger.info("Scan loop exited.")


# ─── Application bootstrap ────────────────────────────────────────────────────

async def post_init(application: Application) -> None:
    """Called once after the bot is initialised."""
    global bot_app, selector, generator, scan_task

    bot_app = application
    logger.info("Bot initialised. Starting components…")

    selector = PairSelector(provider)
    generator = SignalGenerator(
        provider=provider,
        tracker=tracker,
        send_fn=broadcast,
        timeframe=PRIMARY_TIMEFRAME,
        stake=DEFAULT_STAKE,
    )

    scan_task = asyncio.create_task(scan_loop())
    logger.info("Components ready. Scan loop running.")


async def post_shutdown(application: Application) -> None:
    """Graceful shutdown."""
    global _running, scan_task
    _running = False
    if scan_task:
        scan_task.cancel()
        try:
            await scan_task
        except asyncio.CancelledError:
            pass
    await provider.quotex.disconnect()
    logger.info("Bot shut down cleanly at %s", fmt_pkt())


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN is not set in .env — cannot start.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  Advanced Quotex Signal Bot — starting up")
    logger.info("  Time: %s", fmt_pkt())
    logger.info("  Pairs: %d | TF: %s | Stake: $%.2f",
                len(PAIRS), PRIMARY_TIMEFRAME, DEFAULT_STAKE)
    logger.info("=" * 60)

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    # Register command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("pairs", cmd_pairs))
    app.add_handler(CommandHandler("settings", cmd_settings))

    # Run the bot (blocks until Ctrl+C)
    logger.info("Starting polling…")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()

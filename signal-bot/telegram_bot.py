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


# ─── Auto-scan loop ───────────────────────────────────────────────────────────

async def scan_loop() -> None:
    """
    Main scanning loop — runs every SCAN_INTERVAL_SECONDS.
    Finds best pair, runs full signal lifecycle, repeats.
    """
    global _running
    logger.info("🚀 Scan loop started at %s", fmt_pkt())
    await broadcast(
        "🟢 *Quotex Signal Bot STARTED*\n"
        f"📡 Scanning {len(PAIRS)} OTC pairs every {SCAN_INTERVAL_SECONDS}s\n"
        f"🕐 {fmt_pkt()}"
    )

    while _running:
        cycle_start = asyncio.get_event_loop().time()
        try:
            score = await selector.select_best_pair()

            if score is not None:
                await generator.run_signal_cycle(score)
            else:
                logger.info("No qualifying pair this cycle — waiting…")
                # Small status nudge every 10 skipped cycles
                # (handled via counter in production)

        except asyncio.CancelledError:
            logger.info("Scan loop cancelled.")
            break
        except Exception as exc:
            logger.error("Scan loop error: %s", exc, exc_info=True)
            await asyncio.sleep(10)

        # Sleep until the next minute boundary
        elapsed = asyncio.get_event_loop().time() - cycle_start
        sleep_time = max(0, SCAN_INTERVAL_SECONDS - elapsed)
        logger.debug("Cycle done in %.1fs — sleeping %.1fs", elapsed, sleep_time)
        await asyncio.sleep(sleep_time)

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

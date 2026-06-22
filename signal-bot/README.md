# Advanced Quotex Signal Bot

Production-ready Python 3.11 Telegram bot for OTC binary options signal generation.

## Features

- ✅ Auto pair selection from 11 OTC assets
- ✅ 20+ technical indicators (RSI, MACD, Stoch, BB, EMA, ADX, CCI, SAR, ATR, OBV, Williams %R, Stoch RSI…)
- ✅ Advanced analysis: S/R, Trendlines, Market Structure, Chart & Candlestick Patterns
- ✅ Multi-timeframe confirmation (M1 / M5 / M15)
- ✅ Confluence scoring system (strength %)
- ✅ Fallback data sources: Quotex WS → TwelveData → Alpha Vantage → Finnhub
- ✅ Auto WIN/LOSS/DRAW detection
- ✅ Persistent stats tracker (stats.json)
- ✅ Broadcast to multiple Telegram chats/channels/groups
- ✅ Fully async architecture
- ✅ Production logging with rotation

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 2. Configure credentials
cp .env.example .env
# → Fill in your TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, etc.

# 3. Run
python telegram_bot.py
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | ✅ | From @BotFather |
| `TELEGRAM_CHAT_ID` | ✅ | Your Telegram chat/user ID |
| `TELEGRAM_CHANNEL_ID` | ⬜ | Optional broadcast channel |
| `QUOTEX_EMAIL` | ⬜ | Quotex account (for live WS) |
| `QUOTEX_PASSWORD` | ⬜ | Quotex account password |
| `TWELVE_DATA_API_KEY` | ✅ | Already configured |
| `ALPHA_VANTAGE_API_KEY` | ✅ | Already configured |
| `FINNHUB_API_KEY` | ✅ | Already configured |

## Signal Flow

```
Every 60s:
  1. Scan all 11 pairs (concurrent)
  2. Multi-TF analysis (M1 + M5 + M15)
  3. Confluence scoring (20+ indicators)
  4. Select highest-confidence pair
  5. ⚠️ Announce selected pair
  6. ⏱ Countdown (10→1)
  7. 📊 Send signal message
  8. ⏳ Wait 60s (M1 expiry)
  9. ✅/❌ Send WIN/LOSS/DRAW result
  10. Repeat forever
```

## Telegram Commands

| Command | Description |
|---|---|
| `/start` | Welcome message |
| `/help` | All commands |
| `/status` | Bot health, win rate, P&L |
| `/signal` | Force immediate scan |
| `/subscribe` | Add chat to broadcasts |
| `/unsubscribe` | Remove from broadcasts |
| `/stats` | Win/loss/P&L statistics |
| `/pairs` | All monitored pairs |
| `/settings` | Configuration snapshot |

## File Structure

```
signal-bot/
├── telegram_bot.py      ← Main entry point
├── pair_selector.py     ← Multi-TF pair scanner
├── signal_generator.py  ← Signal lifecycle orchestrator
├── indicators.py        ← 20+ indicators + patterns
├── quotex_client.py     ← Data providers (Quotex/TwelveData/AV/Finnhub)
├── tracker.py           ← Win/loss/P&L persistence
├── config.py            ← All tuning parameters
├── utils.py             ← Logging, PKT time, retry decorator
├── stats.json           ← Auto-created on first run
├── logs/bot.log         ← Rotating log file
└── requirements.txt
```

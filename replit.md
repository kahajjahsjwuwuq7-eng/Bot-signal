# Advanced Quotex Signal Bot

A production-ready Python 3.11 Telegram bot that scans 11 OTC pairs every minute, selects the highest-confidence trade setup using 20+ technical indicators, sends professional signals to Telegram, tracks results, and runs 24/7.

## Run & Operate

- `cd signal-bot && python telegram_bot.py` — run the bot
- `cd signal-bot && pip install -r requirements.txt` — install dependencies
- `playwright install chromium` — install browser (for Quotex WS login)
- Workflow: **"Quotex Signal Bot"** — managed long-running process

## Stack

- Python 3.11, asyncio, aiohttp
- python-telegram-bot 20.x
- pandas + pandas-ta (indicators)
- playwright (Chromium for Quotex WebSocket auth)
- Fallback APIs: TwelveData, Alpha Vantage, Finnhub

## Where things live

- `signal-bot/telegram_bot.py` — main entry point, command handlers, scan loop
- `signal-bot/pair_selector.py` — multi-TF pair scanner + confluence scoring
- `signal-bot/signal_generator.py` — full signal lifecycle (announce → countdown → signal → result)
- `signal-bot/indicators.py` — 20+ indicators, patterns, market structure
- `signal-bot/quotex_client.py` — data provider chain (Quotex WS → TwelveData → AV → Finnhub)
- `signal-bot/tracker.py` — win/loss/P&L persistence to stats.json
- `signal-bot/config.py` — all tuning parameters in one place
- `signal-bot/utils.py` — logging, PKT timezone, retry decorator

## Architecture decisions

- **Fully async** — all IO uses asyncio + aiohttp; no blocking calls
- **Data fallback chain** — Quotex WS first, then TwelveData, AV, Finnhub; skip if all fail
- **Confluence scoring** — strength = (agreeing indicators / total active) × 100
- **Multi-TF gating** — signal only fires if M1 + M5 + M15 agree (min 2/3)
- **All timestamps in PKT** (UTC+5 / Asia/Karachi)

## Product

- Scans 11 OTC pairs every 60s
- Runs M1/M5/M15 analysis in parallel
- Selects highest-confidence pair (min 50% strength)
- Sends announcement → countdown → signal → wait → WIN/LOSS/DRAW result
- Supports multi-chat broadcast (/subscribe command)
- Tracks cumulative win rate and P&L in stats.json

## User preferences

- All timestamps in Pakistan Standard Time (PKT = UTC+5)
- Signal format must match the template exactly (UP/DOWN emojis, Confluence fraction)
- Risk levels: LOW >70%, MEDIUM 60-70%, HIGH 50-60%
- Default stake: $10.00 per signal

## Required Environment Variables

| Variable | Status |
|---|---|
| `TELEGRAM_BOT_TOKEN` | ❗ Must be provided |
| `TELEGRAM_CHAT_ID` | ❗ Must be provided |
| `TELEGRAM_CHANNEL_ID` | Optional |
| `QUOTEX_EMAIL` | Optional (for live WS) |
| `QUOTEX_PASSWORD` | Optional (for live WS) |
| `TWELVE_DATA_API_KEY` | ✅ Configured |
| `ALPHA_VANTAGE_API_KEY` | ✅ Configured |
| `FINNHUB_API_KEY` | ✅ Configured |

## Gotchas

- The bot will NOT start without `TELEGRAM_BOT_TOKEN` set in Secrets
- Without Quotex credentials, it falls back automatically to TwelveData (configured)
- `pandas-ta` must be installed BEFORE `ta` to avoid import conflicts
- Always run `pip install -r requirements.txt` before first start

## Pointers

- See `signal-bot/config.py` for all tunable parameters (periods, thresholds, pairs)
- See `signal-bot/stats.json` for running win/loss statistics
- See `signal-bot/logs/bot.log` for full debug log

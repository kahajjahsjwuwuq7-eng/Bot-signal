"""
config.py — Centralised configuration for the Quotex Signal Bot.
All constants and tuning parameters live here so nothing is magic-numbered
elsewhere in the codebase.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Telegram ────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_CHANNEL_ID: str = os.getenv("TELEGRAM_CHANNEL_ID", "")

# ─── Quotex credentials ──────────────────────────────────────────────────────
QUOTEX_EMAIL: str = os.getenv("QUOTEX_EMAIL", "")
QUOTEX_PASSWORD: str = os.getenv("QUOTEX_PASSWORD", "")
QUOTEX_IS_DEMO: bool = os.getenv("QUOTEX_IS_DEMO", "True").lower() == "true"

# ─── Backup API keys ─────────────────────────────────────────────────────────
TWELVE_DATA_API_KEY: str = os.getenv("TWELVE_DATA_API_KEY", "")
ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")

# ─── Timezone ────────────────────────────────────────────────────────────────
TIMEZONE = "Asia/Karachi"   # PKT = UTC+5

# ─── Asset list ──────────────────────────────────────────────────────────────
PAIRS = [
    "EURUSD_otc",
    "GBPUSD_otc",
    "AUDUSD_otc",
    "USDJPY_otc",
    "EURJPY_otc",
    "USDCAD_otc",
    "AUDCAD_otc",
    "EURGBP_otc",
    "BTCUSD_otc",
    "ETHUSD_otc",
    "XAUUSD_otc",
]

# ─── Timeframes ──────────────────────────────────────────────────────────────
TIMEFRAMES = ["M1", "M5", "M15"]
TIMEFRAME_SECONDS = {"M1": 60, "M5": 300, "M15": 900}
PRIMARY_TIMEFRAME = "M1"          # signal expiry timeframe
CANDLE_COUNT = 250                # candles fetched per analysis window

# ─── Indicator parameters ────────────────────────────────────────────────────
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

STOCH_RSI_PERIOD = 14
STOCH_RSI_K = 3
STOCH_RSI_D = 3
STOCH_RSI_OB = 80
STOCH_RSI_OS = 20

STOCH_K = 14
STOCH_D = 3
STOCH_SMOOTH = 3

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BB_PERIOD = 20
BB_STDDEV = 2

EMA_PERIODS = [9, 21, 50, 200]

ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25

WILLIAMS_R_PERIOD = 14

CCI_PERIOD = 20

SAR_STEP = 0.02
SAR_MAX = 0.2

ATR_PERIOD = 14

# ─── Scoring thresholds ──────────────────────────────────────────────────────
MIN_INDICATOR_AGREEMENTS = 4
MIN_STRENGTH_PCT = 45.0
MIN_ADX = 18.0
MIN_TIMEFRAMES_AGREE = 2

# ─── Risk buckets ────────────────────────────────────────────────────────────
RISK_LOW_THRESHOLD = 70
RISK_MEDIUM_THRESHOLD = 60

# ─── Trade tracking ──────────────────────────────────────────────────────────
DEFAULT_STAKE = 10.0      # USD per signal (for P&L display)
STATS_FILE = "stats.json"
LOG_FILE = "logs/bot.log"

# ─── Retry / timeout settings ────────────────────────────────────────────────
HTTP_TIMEOUT = 10          # seconds
WS_RECONNECT_DELAY = 5     # seconds
MAX_RETRIES = 3

# ─── Scan interval ───────────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 60

# ─── Quotex WebSocket ────────────────────────────────────────────────────────
QUOTEX_WS_URI = "wss://ws2.po.market/socket.io/?EIO=4&transport=websocket"
QUOTEX_WS_ORIGIN = "https://qxbroker.com"

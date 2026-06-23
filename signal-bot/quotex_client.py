"""
quotex_client.py — Data acquisition layer.

Priority fallback chain:
  1. Quotex WebSocket (real-time OTC data)
  2. TwelveData REST API
  3. Alpha Vantage REST API
  4. Finnhub REST API

Returns a standard pandas OHLCV DataFrame for any successful source.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import aiohttp
import pandas as pd
import numpy as np

from config import (
    QUOTEX_EMAIL, QUOTEX_PASSWORD, QUOTEX_IS_DEMO,
    QUOTEX_WS_URI, QUOTEX_WS_ORIGIN,
    TWELVE_DATA_API_KEY, ALPHA_VANTAGE_API_KEY, FINNHUB_API_KEY,
    HTTP_TIMEOUT, WS_RECONNECT_DELAY, MAX_RETRIES, CANDLE_COUNT,
    TIMEFRAME_SECONDS,
)
from utils import logger, async_retry, base_pair

# ─── Symbol mappings ─────────────────────────────────────────────────────────

_TWELVE_DATA_SYMBOLS: dict[str, str] = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "AUDUSD": "AUD/USD",
    "USDJPY": "USD/JPY",
    "EURJPY": "EUR/JPY",
    "USDCAD": "USD/CAD",
    "AUDCAD": "AUD/CAD",
    "EURGBP": "EUR/GBP",
    "BTCUSD": "BTC/USD",
    "ETHUSD": "ETH/USD",
    "XAUUSD": "XAU/USD",
}

_AV_SYMBOLS: dict[str, str] = {
    "EURUSD": "EUR", "GBPUSD": "GBP", "AUDUSD": "AUD",
    "USDJPY": "JPY", "EURJPY": "EUR", "USDCAD": "CAD",
    "AUDCAD": "AUD", "EURGBP": "EUR",
}

_FINNHUB_SYMBOLS: dict[str, str] = {
    "EURUSD": "OANDA:EUR_USD",
    "GBPUSD": "OANDA:GBP_USD",
    "AUDUSD": "OANDA:AUD_USD",
    "USDJPY": "OANDA:USD_JPY",
    "EURJPY": "OANDA:EUR_JPY",
    "USDCAD": "OANDA:USD_CAD",
    "AUDCAD": "OANDA:AUD_CAD",
    "EURGBP": "OANDA:EUR_GBP",
    "BTCUSD": "BINANCE:BTCUSDT",
    "ETHUSD": "BINANCE:ETHUSDT",
    "XAUUSD": "OANDA:XAU_USD",
}

_TF_TO_TWELVE: dict[str, str] = {"M1": "1min", "M5": "5min", "M15": "15min"}
_TF_TO_AV: dict[str, str] = {"M1": "1min", "M5": "5min", "M15": "15min"}
_TF_TO_FINNHUB: dict[str, str] = {"M1": "1", "M5": "5", "M15": "15"}


# ─── DataFrame builder ───────────────────────────────────────────────────────

def _build_df(rows: list[dict]) -> pd.DataFrame:
    """Convert a list of OHLCV dicts to a clean DataFrame."""
    df = pd.DataFrame(rows)
    df.columns = [c.lower() for c in df.columns]
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    if "datetime" in df.columns:
        df.index = pd.to_datetime(df["datetime"])
        df.drop(columns=["datetime"], inplace=True, errors="ignore")
    elif "timestamp" in df.columns:
        df.index = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df.drop(columns=["timestamp"], inplace=True, errors="ignore")
    df.sort_index(inplace=True)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


# ─── Quotex WebSocket client ─────────────────────────────────────────────────

class QuotexWebSocketClient:
    """
    Thin async WebSocket client for Quotex streaming.
    Streams real-time candle data; falls back to REST providers on failure.
    """

    # Track first-failure so we only log once per session
    _logged_failure: bool = False

    def __init__(self):
        self._ws = None
        self._candle_buffer: dict[str, list] = {}
        self._connected = False
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        """Attempt WebSocket connection to Quotex. Returns True on success."""
        try:
            import websockets
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    QUOTEX_WS_URI,
                    additional_headers={
                        "Origin": QUOTEX_WS_ORIGIN,
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                        ),
                    },
                    ping_interval=20,
                    ping_timeout=10,
                ),
                timeout=15,
            )
            self._connected = True
            QuotexWebSocketClient._logged_failure = False
            logger.info("Quotex WebSocket connected.")
            return True
        except Exception as exc:
            if not QuotexWebSocketClient._logged_failure:
                logger.warning(
                    "Quotex WebSocket unavailable (%s) — using backup APIs.", exc
                )
                QuotexWebSocketClient._logged_failure = True
            self._connected = False
            return False

    async def disconnect(self):
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._connected = False
        logger.info("Quotex WebSocket disconnected.")

    async def get_candles(self, pair: str, timeframe: str = "M1",
                          count: int = CANDLE_COUNT) -> Optional[pd.DataFrame]:
        """
        Request historical candles for *pair* via WebSocket.
        Returns None if unavailable.
        """
        if not self._connected:
            if not await self.connect():
                return None

        try:
            tf_sec = TIMEFRAME_SECONDS.get(timeframe, 60)
            end_ts = int(time.time())
            start_ts = end_ts - tf_sec * count

            # Quotex uses Socket.IO-style framing
            payload = json.dumps({
                "action": "history",
                "asset": pair,
                "period": tf_sec,
                "time": end_ts,
                "offset": -count,
            })
            await self._ws.send(f"42{json.dumps(['history', {'asset': pair, 'period': tf_sec, 'time': end_ts, 'offset': -count}])}")

            # Wait for response with timeout
            deadline = asyncio.get_event_loop().time() + 10
            while asyncio.get_event_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(self._ws.recv(), timeout=3)
                    if "candles" in raw or "history" in raw:
                        # Parse Socket.IO message
                        if raw.startswith("42"):
                            data = json.loads(raw[2:])
                            if isinstance(data, list) and len(data) > 1:
                                candle_data = data[1]
                                if isinstance(candle_data, dict):
                                    candles = candle_data.get("candles", candle_data.get("history", []))
                                    if candles:
                                        rows = []
                                        for c in candles:
                                            if isinstance(c, (list, tuple)) and len(c) >= 5:
                                                rows.append({
                                                    "timestamp": c[0],
                                                    "open": c[1],
                                                    "close": c[2],
                                                    "high": c[3],
                                                    "low": c[4],
                                                    "volume": c[5] if len(c) > 5 else 0,
                                                })
                                            elif isinstance(c, dict):
                                                rows.append(c)
                                        if rows:
                                            return _build_df(rows)
                except asyncio.TimeoutError:
                    break
        except Exception as exc:
            logger.warning("Quotex candle fetch error: %s", exc)
            self._connected = False

        return None


# ─── TwelveData REST source ──────────────────────────────────────────────────

class TwelveDataClient:
    BASE = "https://api.twelvedata.com"

    def __init__(self, api_key: str = TWELVE_DATA_API_KEY):
        self.api_key = api_key

    # Rate-limit: free tier = 8 req/min → 1 req every 7.5s
    _call_lock = asyncio.Lock()
    _last_call_ts: float = 0.0
    _MIN_CALL_INTERVAL: float = 7.5

    @async_retry(max_retries=MAX_RETRIES, delay=2.0)
    async def get_candles(self, pair: str, timeframe: str = "M1",
                          count: int = CANDLE_COUNT) -> Optional[pd.DataFrame]:
        if not self.api_key:
            return None

        sym = base_pair(pair)
        symbol = _TWELVE_DATA_SYMBOLS.get(sym)
        if not symbol:
            return None

        # Rate-limit: free tier allows ~8 req/min
        async with TwelveDataClient._call_lock:
            now = asyncio.get_event_loop().time()
            wait = TwelveDataClient._MIN_CALL_INTERVAL - (now - TwelveDataClient._last_call_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            TwelveDataClient._last_call_ts = asyncio.get_event_loop().time()

        interval = _TF_TO_TWELVE.get(timeframe, "1min")
        url = (
            f"{self.BASE}/time_series"
            f"?symbol={symbol}&interval={interval}&outputsize={count}"
            f"&apikey={self.api_key}&format=JSON"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as resp:
                    data = await resp.json()

            if data.get("status") == "error" or "values" not in data:
                logger.debug("TwelveData error: %s", data.get("message", "unknown"))
                return None

            rows = [
                {
                    "datetime": v["datetime"],
                    "open": v["open"],
                    "high": v["high"],
                    "low": v["low"],
                    "close": v["close"],
                    "volume": v.get("volume", 0),
                }
                for v in data["values"]
            ]
            df = _build_df(rows)
            logger.info("TwelveData: got %d candles for %s %s", len(df), pair, timeframe)
            return df

        except Exception as exc:
            logger.warning("TwelveData request failed: %s", exc)
            return None


# ─── Alpha Vantage REST source ────────────────────────────────────────────────

class AlphaVantageClient:
    BASE = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str = ALPHA_VANTAGE_API_KEY):
        self.api_key = api_key

    @async_retry(max_retries=MAX_RETRIES, delay=3.0)
    async def get_candles(self, pair: str, timeframe: str = "M1",
                          count: int = CANDLE_COUNT) -> Optional[pd.DataFrame]:
        if not self.api_key:
            return None

        sym = base_pair(pair)
        # Alpha Vantage Forex Intraday
        from_sym = sym[:3]
        to_sym = sym[3:]
        interval = _TF_TO_AV.get(timeframe, "1min")

        params = {
            "function": "FX_INTRADAY",
            "from_symbol": from_sym,
            "to_symbol": to_sym,
            "interval": interval,
            "outputsize": "full",
            "apikey": self.api_key,
        }

        # Crypto override
        if sym in ("BTCUSD", "ETHUSD"):
            coin = sym[:3]
            params = {
                "function": "CRYPTO_INTRADAY",
                "symbol": coin,
                "market": "USD",
                "interval": interval,
                "outputsize": "full",
                "apikey": self.api_key,
            }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BASE,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT + 5),
                ) as resp:
                    data = await resp.json()

            ts_key = [k for k in data if "Time Series" in k]
            if not ts_key:
                logger.debug("AlphaVantage no time series for %s: %s", pair, list(data.keys()))
                return None

            ts = data[ts_key[0]]
            rows = []
            for dt_str, vals in ts.items():
                rows.append({
                    "datetime": dt_str,
                    "open": vals.get("1. open", vals.get("1a. open (USD)", 0)),
                    "high": vals.get("2. high", vals.get("2a. high (USD)", 0)),
                    "low": vals.get("3. low", vals.get("3a. low (USD)", 0)),
                    "close": vals.get("4. close", vals.get("4a. close (USD)", 0)),
                    "volume": vals.get("5. volume", 0),
                })

            df = _build_df(rows)
            df = df.tail(count)
            logger.info("AlphaVantage: got %d candles for %s %s", len(df), pair, timeframe)
            return df

        except Exception as exc:
            logger.warning("AlphaVantage request failed: %s", exc)
            return None


# ─── Finnhub REST source ─────────────────────────────────────────────────────

class FinnhubClient:
    BASE = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str = FINNHUB_API_KEY):
        self.api_key = api_key

    @async_retry(max_retries=MAX_RETRIES, delay=2.0)
    async def get_candles(self, pair: str, timeframe: str = "M1",
                          count: int = CANDLE_COUNT) -> Optional[pd.DataFrame]:
        if not self.api_key:
            return None

        sym = base_pair(pair)
        symbol = _FINNHUB_SYMBOLS.get(sym)
        if not symbol:
            return None

        resolution = _TF_TO_FINNHUB.get(timeframe, "1")
        tf_sec = TIMEFRAME_SECONDS.get(timeframe, 60)
        to_ts = int(time.time())
        from_ts = to_ts - tf_sec * count * 2   # request extra to compensate weekends

        url = (
            f"{self.BASE}/forex/candle"
            f"?symbol={symbol}&resolution={resolution}"
            f"&from={from_ts}&to={to_ts}&token={self.api_key}"
        )
        # Crypto uses different endpoint
        if sym in ("BTCUSD", "ETHUSD"):
            url = (
                f"{self.BASE}/crypto/candle"
                f"?symbol={symbol}&resolution={resolution}"
                f"&from={from_ts}&to={to_ts}&token={self.api_key}"
            )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as resp:
                    data = await resp.json()

            if data.get("s") != "ok":
                logger.debug("Finnhub status: %s for %s", data.get("s"), pair)
                return None

            rows = [
                {
                    "timestamp": data["t"][i],
                    "open": data["o"][i],
                    "high": data["h"][i],
                    "low": data["l"][i],
                    "close": data["c"][i],
                    "volume": data["v"][i] if "v" in data else 0,
                }
                for i in range(len(data["t"]))
            ]
            df = _build_df(rows)
            df = df.tail(count)
            logger.info("Finnhub: got %d candles for %s %s", len(df), pair, timeframe)
            return df

        except Exception as exc:
            logger.warning("Finnhub request failed: %s", exc)
            return None


# ─── Unified data provider ───────────────────────────────────────────────────

class DataProvider:
    """
    Tries each data source in priority order and returns the first success.
    Priority: Quotex WS → TwelveData → AlphaVantage → Finnhub
    """

    def __init__(self):
        self.quotex = QuotexWebSocketClient()
        self.twelve = TwelveDataClient()
        self.av = AlphaVantageClient()
        self.finnhub = FinnhubClient()

    async def get_candles(
        self,
        pair: str,
        timeframe: str = "M1",
        count: int = CANDLE_COUNT,
    ) -> Optional[pd.DataFrame]:
        sources = [
            ("Quotex WS", self.quotex.get_candles),
            ("TwelveData", self.twelve.get_candles),
            ("AlphaVantage", self.av.get_candles),
            ("Finnhub", self.finnhub.get_candles),
        ]

        min_rows = min(count, 30)   # price-only fetches (count≤10) don't need 30 rows
        for name, fn in sources:
            try:
                df = await fn(pair, timeframe, count)
                if df is not None and len(df) >= min_rows:
                    logger.debug("Data from %s for %s/%s (%d rows)", name, pair, timeframe, len(df))
                    return df
            except Exception as exc:
                logger.warning("%s failed for %s/%s: %s", name, pair, timeframe, exc)

        logger.error("All data sources failed for %s / %s — skipping.", pair, timeframe)
        return None

    async def get_current_price(self, pair: str) -> Optional[float]:
        """Return latest close price for *pair* (M1 candle)."""
        df = await self.get_candles(pair, "M1", 5)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
        return None

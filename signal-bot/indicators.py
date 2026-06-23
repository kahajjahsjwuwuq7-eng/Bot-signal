"""
indicators.py — Full technical-analysis suite.

Computes all 20+ indicators, pattern recognition, market structure,
support/resistance, candlestick analysis, and a confluence score
from a pandas OHLCV DataFrame.

Expected DataFrame columns (case-insensitive normalised to lower):
    open, high, low, close, volume
Index: DatetimeIndex (ascending)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import pandas_ta as ta          # preferred
    _HAS_PANDAS_TA = True
except ImportError:
    _HAS_PANDAS_TA = False

from config import (
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    STOCH_RSI_PERIOD, STOCH_RSI_K, STOCH_RSI_D, STOCH_RSI_OB, STOCH_RSI_OS,
    STOCH_K, STOCH_D, STOCH_SMOOTH,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STDDEV,
    EMA_PERIODS,
    ADX_PERIOD, ADX_TREND_THRESHOLD,
    WILLIAMS_R_PERIOD,
    CCI_PERIOD,
    SAR_STEP, SAR_MAX,
    ATR_PERIOD,
    MIN_INDICATOR_AGREEMENTS, MIN_STRENGTH_PCT, MIN_ADX,
)

Direction = Literal["BUY", "SELL", "NEUTRAL"]


# ─── Low-level maths helpers ────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series,
           k: int = 14, d: int = 3, smooth: int = 3) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    k_line = k_raw.rolling(smooth).mean()
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (volume * direction).cumsum()


def _macd(close: pd.Series, fast=12, slow=26, signal=9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(close: pd.Series, period=20, std=2) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = _sma(close, period)
    std_dev = close.rolling(period).std()
    upper = mid + std * std_dev
    lower = mid - std * std_dev
    return upper, mid, lower


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = _atr(high, low, close, period)
    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean(), plus_di, minus_di


def _parabolic_sar(high: pd.Series, low: pd.Series,
                   step=0.02, max_af=0.2) -> pd.Series:
    """Vectorised Parabolic SAR."""
    highs = high.values
    lows = low.values
    n = len(highs)
    sar = np.zeros(n)
    ep = lows[0]
    af = step
    uptrend = True
    sar[0] = highs[0]

    for i in range(1, n):
        if uptrend:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = min(sar[i], lows[i - 1], lows[max(0, i - 2)])
            if lows[i] < sar[i]:
                uptrend = False
                sar[i] = ep
                ep = lows[i]
                af = step
            else:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + step, max_af)
        else:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = max(sar[i], highs[i - 1], highs[max(0, i - 2)])
            if highs[i] > sar[i]:
                uptrend = True
                sar[i] = ep
                ep = highs[i]
                af = step
            else:
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + step, max_af)

    return pd.Series(sar, index=high.index)


# ─── Market structure helpers ────────────────────────────────────────────────

def _pivots(series: pd.Series, left=5, right=5):
    """Return (highs_idx, lows_idx) of swing pivot indices."""
    highs, lows = [], []
    for i in range(left, len(series) - right):
        window_h = series.iloc[i - left: i + right + 1]
        window_l = series.iloc[i - left: i + right + 1]
        if series.iloc[i] == window_h.max():
            highs.append(i)
        if series.iloc[i] == window_l.min():
            lows.append(i)
    return highs, lows


def detect_market_structure(high: pd.Series, low: pd.Series) -> dict:
    """Detect HH/HL (bullish) or LH/LL (bearish) market structure."""
    result = {"bullish": False, "bearish": False, "structure": "SIDEWAYS"}
    if len(high) < 20:
        return result

    h_pivots, l_pivots = _pivots(high, 3, 3), _pivots(low, 3, 3)

    if len(h_pivots[0]) >= 2 and len(l_pivots[1]) >= 2:
        # Check last 2 swing highs & lows
        sh = [high.iloc[i] for i in h_pivots[0][-2:]]
        sl = [low.iloc[i] for i in l_pivots[1][-2:]]
        hh = sh[-1] > sh[-2]
        hl = sl[-1] > sl[-2]
        lh = sh[-1] < sh[-2]
        ll = sl[-1] < sl[-2]
        if hh and hl:
            result.update({"bullish": True, "structure": "BULLISH (HH/HL)"})
        elif lh and ll:
            result.update({"bearish": True, "structure": "BEARISH (LH/LL)"})
    return result


def detect_support_resistance(close: pd.Series, window=10, tolerance=0.002) -> dict:
    """Simple S/R via pivot clustering."""
    levels = []
    for i in range(window, len(close) - window):
        chunk = close.iloc[i - window: i + window]
        if close.iloc[i] == chunk.max() or close.iloc[i] == chunk.min():
            levels.append(float(close.iloc[i]))

    current = float(close.iloc[-1])
    nearest_support = max((l for l in levels if l < current), default=None)
    nearest_resistance = min((l for l in levels if l > current), default=None)

    near_support = nearest_support is not None and abs(current - nearest_support) / current < tolerance
    near_resistance = nearest_resistance is not None and abs(nearest_resistance - current) / current < tolerance

    return {
        "levels": sorted(set(round(l, 5) for l in levels)),
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "near_support": near_support,
        "near_resistance": near_resistance,
    }


# ─── Candlestick patterns ────────────────────────────────────────────────────

def detect_candlestick_patterns(o: pd.Series, h: pd.Series,
                                 l: pd.Series, c: pd.Series) -> dict:
    """Detect last-candle candlestick patterns."""
    body = abs(c - o)
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
    avg_body = body.rolling(10).mean()

    last = -1   # latest candle index

    is_bull = c.iloc[last] > o.iloc[last]
    is_bear = c.iloc[last] < o.iloc[last]
    body_size = float(body.iloc[last])
    avg = float(avg_body.iloc[last]) if not pd.isna(avg_body.iloc[last]) else body_size

    # Doji: body < 10% of average body
    is_doji = body_size < 0.1 * avg

    # Hammer: small body near top, long lower wick, uptrend signal
    is_hammer = (
        lower_wick.iloc[last] > 2 * body_size and
        upper_wick.iloc[last] < 0.5 * body_size and
        not is_doji
    )

    # Shooting Star: small body near bottom, long upper wick
    is_shooting_star = (
        upper_wick.iloc[last] > 2 * body_size and
        lower_wick.iloc[last] < 0.5 * body_size and
        not is_doji
    )

    # Engulfing (compare last two candles)
    if len(c) >= 2:
        prev_body_size = float(abs(c.iloc[-2] - o.iloc[-2]))
        bullish_engulfing = (
            c.iloc[-2] < o.iloc[-2] and   # prev bearish
            is_bull and
            o.iloc[last] <= c.iloc[-2] and
            c.iloc[last] >= o.iloc[-2] and
            body_size > prev_body_size
        )
        bearish_engulfing = (
            c.iloc[-2] > o.iloc[-2] and   # prev bullish
            is_bear and
            o.iloc[last] >= c.iloc[-2] and
            c.iloc[last] <= o.iloc[-2] and
            body_size > prev_body_size
        )
    else:
        bullish_engulfing = bearish_engulfing = False

    return {
        "is_doji": is_doji,
        "is_hammer": is_hammer,
        "is_shooting_star": is_shooting_star,
        "bullish_engulfing": bullish_engulfing,
        "bearish_engulfing": bearish_engulfing,
    }


# ─── Chart patterns ──────────────────────────────────────────────────────────

def detect_chart_patterns(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    """Detect Double Top, Double Bottom, H&S patterns (simplified)."""
    if len(close) < 50:
        return {}

    h_idx, l_idx = _pivots(high, 5, 5), _pivots(low, 5, 5)
    h_vals = [float(high.iloc[i]) for i in h_idx[0][-6:]]
    l_vals = [float(low.iloc[i]) for i in l_idx[1][-6:]]

    tol = 0.003  # 0.3% price tolerance

    patterns = {"double_top": False, "double_bottom": False,
                "head_shoulders": False, "inv_head_shoulders": False}

    # Double Top: two nearly equal swing highs
    if len(h_vals) >= 2:
        if abs(h_vals[-1] - h_vals[-2]) / (h_vals[-1] + 1e-10) < tol:
            patterns["double_top"] = True

    # Double Bottom: two nearly equal swing lows
    if len(l_vals) >= 2:
        if abs(l_vals[-1] - l_vals[-2]) / (l_vals[-1] + 1e-10) < tol:
            patterns["double_bottom"] = True

    # Head & Shoulders: 3 highs where middle is highest
    if len(h_vals) >= 3:
        left, head, right = h_vals[-3], h_vals[-2], h_vals[-1]
        if head > left and head > right and abs(left - right) / (head + 1e-10) < tol:
            patterns["head_shoulders"] = True

    # Inverse H&S: 3 lows where middle is lowest
    if len(l_vals) >= 3:
        left, head, right = l_vals[-3], l_vals[-2], l_vals[-1]
        if head < left and head < right and abs(left - right) / (max(left, right) + 1e-10) < tol:
            patterns["inv_head_shoulders"] = True

    return patterns


# ─── Hull Moving Average ─────────────────────────────────────────────────────

def _hma(series: pd.Series, period: int = 20) -> pd.Series:
    """Hull Moving Average — half the lag of a WMA."""
    half  = max(1, period // 2)
    sqrt_p = max(1, int(period ** 0.5))
    wma_h = series.ewm(span=half,   adjust=False).mean()
    wma_f = series.ewm(span=period, adjust=False).mean()
    return (2 * wma_h - wma_f).ewm(span=sqrt_p, adjust=False).mean()


# ─── Supertrend ───────────────────────────────────────────────────────────────

def _supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 10, multiplier: float = 3.0,
) -> pd.Series:
    """Supertrend: returns +1 (bull) or -1 (bear) for each bar."""
    atr  = _atr(high, low, close, period)
    hl2  = (high + low) / 2
    ub   = (hl2 + multiplier * atr).values
    lb   = (hl2 - multiplier * atr).values
    c    = close.values
    fu   = ub.copy()
    fl   = lb.copy()
    t    = np.ones(len(c), dtype=np.int8)

    for i in range(1, len(c)):
        fu[i] = ub[i] if ub[i] < fu[i-1] or c[i-1] > fu[i-1] else fu[i-1]
        fl[i] = lb[i] if lb[i] > fl[i-1] or c[i-1] < fl[i-1] else fl[i-1]
        if t[i-1] == -1 and c[i] > fu[i]:
            t[i] = 1
        elif t[i-1] == 1 and c[i] < fl[i]:
            t[i] = -1
        else:
            t[i] = t[i-1]

    return pd.Series(t, index=close.index)


# ─── Ichimoku Cloud ───────────────────────────────────────────────────────────

def _ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    """Return Tenkan, Kijun, Span A, Span B, Chikou."""
    tenkan  = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    kijun   = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a  = ((tenkan + kijun) / 2).shift(26)          # current cloud = .iloc[-27]
    span_b  = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou  = close.shift(-26)
    return {"tenkan": tenkan, "kijun": kijun,
            "span_a": span_a, "span_b": span_b, "chikou": chikou}


# ─── VWAP ────────────────────────────────────────────────────────────────────

def _vwap(high: pd.Series, low: pd.Series, close: pd.Series,
          volume: pd.Series) -> pd.Series:
    tp  = (high + low + close) / 3
    cum = (tp * volume).cumsum()
    vol = volume.cumsum().replace(0, np.nan)
    return cum / vol


# ─── Money Flow Index ─────────────────────────────────────────────────────────

def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, period: int = 14) -> pd.Series:
    tp  = (high + low + close) / 3
    mf  = tp * volume
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    mfr = pos.rolling(period).sum() / neg.rolling(period).sum().replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


# ─── Chaikin Money Flow ───────────────────────────────────────────────────────

def _cmf(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, period: int = 20) -> pd.Series:
    clv = ((close - low) - (high - close)) / (high - low + 1e-10)
    return (clv * volume).rolling(period).sum() / \
           volume.rolling(period).sum().replace(0, np.nan)


# ─── Heikin Ashi trend ────────────────────────────────────────────────────────

def _heikin_ashi_trend(
    open_: pd.Series, high: pd.Series, low: pd.Series,
    close: pd.Series, bars: int = 5,
) -> tuple[int, int]:
    """Return (bull_bars, bear_bars) among the last *bars* Heikin Ashi candles."""
    ha_close = (open_ + high + low + close) / 4
    hac = ha_close.values.copy()
    hao = hac.copy()                     # start HA open = HA close
    for i in range(1, len(hao)):
        hao[i] = (hao[i-1] + hac[i-1]) / 2
    bull = int(np.sum(hac[-bars:] > hao[-bars:]))
    return bull, bars - bull


# ─── RSI Divergence ───────────────────────────────────────────────────────────

def detect_rsi_divergence(
    close: pd.Series, rsi_series: pd.Series, lookback: int = 25,
) -> str:
    """
    Returns 'BULL', 'BEAR', or 'NONE'.
    Bullish: price makes lower low but RSI makes higher low  → BUY.
    Bearish: price makes higher high but RSI makes lower high → SELL.
    """
    if len(close) < lookback + 5:
        return "NONE"
    c = close.iloc[-lookback:].dropna()
    r = rsi_series.iloc[-lookback:].reindex(c.index).ffill().dropna()
    if len(r) < 10 or r.isna().all():
        return "NONE"

    c  = c.reindex(r.index)
    rng = float(c.max() - c.min())
    if rng == 0:
        return "NONE"

    last_c = float(c.iloc[-1])
    last_r = float(r.iloc[-1])

    price_near_low  = (last_c - float(c.min())) / rng < 0.2
    rsi_above_low   = last_r > float(r.min()) + 3          # RSI noticeably higher than its low
    price_near_high = (float(c.max()) - last_c) / rng < 0.2
    rsi_below_high  = last_r < float(r.max()) - 3          # RSI noticeably lower than its high

    if price_near_low and rsi_above_low and last_r < 55:
        return "BULL"
    if price_near_high and rsi_below_high and last_r > 45:
        return "BEAR"
    return "NONE"


# ─── SMC — Break of Structure ─────────────────────────────────────────────────

def detect_smc_bos(
    high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 20
) -> dict:
    """Bullish BOS: close breaks above the last swing high. Bearish: below swing low."""
    if len(close) < lookback + 3:
        return {"bullish_bos": False, "bearish_bos": False}
    swing_high = float(high.iloc[-lookback:-2].max())
    swing_low  = float(low.iloc[-lookback:-2].min())
    prev  = float(close.iloc[-2])
    curr  = float(close.iloc[-1])
    return {
        "bullish_bos": prev <= swing_high < curr,
        "bearish_bos": curr < swing_low  <= prev,
    }


# ─── SMC — Order Blocks ───────────────────────────────────────────────────────

def detect_order_blocks(
    open_: pd.Series, high: pd.Series, low: pd.Series,
    close: pd.Series, lookback: int = 40,
) -> dict:
    """
    Demand OB: last bearish candle before a 2×ATR bullish impulse.
    Supply OB: last bullish candle before a 2×ATR bearish impulse.
    Returns True if current price sits inside a recent OB zone.
    """
    if len(close) < lookback + 5:
        return {"ob_bull": False, "ob_bear": False}

    atr_val = float(_atr(high, low, close, 14).iloc[-1])
    curr    = float(close.iloc[-1])
    o = open_.values[-lookback:]
    h = high.values[-lookback:]
    l = low.values[-lookback:]
    c = close.values[-lookback:]

    demand, supply = [], []
    for i in range(1, len(c) - 3):
        if c[i] < o[i]:                                   # bearish candle
            if max(h[i+1:i+4]) - c[i] > 2 * atr_val:    # strong bullish impulse after
                demand.append((l[i], h[i]))
        if c[i] > o[i]:                                   # bullish candle
            if o[i] - min(l[i+1:i+4]) > 2 * atr_val:    # strong bearish impulse after
                supply.append((l[i], h[i]))

    ob_bull = any(lo <= curr <= hi for lo, hi in demand[-5:])
    ob_bear = any(lo <= curr <= hi for lo, hi in supply[-5:])
    return {"ob_bull": ob_bull, "ob_bear": ob_bear}


# ─── SMC — Fair Value Gaps ────────────────────────────────────────────────────

def detect_fvg(
    high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 30
) -> dict:
    """
    Bullish FVG: candle[i+1].low > candle[i-1].high — gap above, price has momentum.
    Bearish FVG: candle[i+1].high < candle[i-1].low — gap below.
    Only signals when current price is beyond the gap (unfilled impulse).
    """
    if len(close) < lookback + 3:
        return {"fvg_bull": False, "fvg_bear": False}

    h = high.values
    l = low.values
    c = close.values
    curr = c[-1]
    bull_fvg = bear_fvg = False

    for i in range(max(1, len(c) - lookback), len(c) - 2):
        if l[i+1] > h[i-1] and curr > l[i+1]:           # bullish FVG, price above gap
            bull_fvg = True
        if h[i+1] < l[i-1] and curr < h[i+1]:           # bearish FVG, price below gap
            bear_fvg = True

    return {"fvg_bull": bull_fvg, "fvg_bear": bear_fvg}


# ─── Breakout / Liquidity ────────────────────────────────────────────────────

def detect_breakout(close: pd.Series, high: pd.Series, low: pd.Series,
                    period=20) -> dict:
    recent_high = high.iloc[-period - 1: -1].max()
    recent_low = low.iloc[-period - 1: -1].min()
    current = float(close.iloc[-1])
    return {
        "bullish_breakout": current > float(recent_high),
        "bearish_breakout": current < float(recent_low),
        "range_high": float(recent_high),
        "range_low": float(recent_low),
    }


def detect_liquidity_sweep(high: pd.Series, low: pd.Series,
                            close: pd.Series, lookback=10) -> dict:
    """Detect price sweeping beyond recent S/R then reversing (stop hunt)."""
    recent_high = float(high.iloc[-lookback - 1: -2].max())
    recent_low = float(low.iloc[-lookback - 1: -2].min())
    curr_h = float(high.iloc[-1])
    curr_l = float(low.iloc[-1])
    curr_c = float(close.iloc[-1])

    sweep_high = curr_h > recent_high and curr_c < recent_high  # swept highs, closed back
    sweep_low = curr_l < recent_low and curr_c > recent_low     # swept lows, closed back
    return {"sweep_high": sweep_high, "sweep_low": sweep_low}


# ─── Main AnalysisResult dataclass ──────────────────────────────────────────

@dataclass
class AnalysisResult:
    direction: Direction = "NEUTRAL"
    strength: float = 0.0
    agreements: int = 0
    active_indicators: int = 0
    adx: float = 0.0
    atr: float = 0.0
    rsi: float = 50.0
    ema200: float = 0.0
    close: float = 0.0
    volume_confirm: bool = False
    trend_confirm: bool = False
    is_doji: bool = False
    market_structure: str = "SIDEWAYS"
    patterns: dict = field(default_factory=dict)
    candle_patterns: dict = field(default_factory=dict)
    support_resistance: dict = field(default_factory=dict)
    breakout: dict = field(default_factory=dict)
    liquidity: dict = field(default_factory=dict)
    # ── Advanced fields ──────────────────────────────────────────────────────
    supertrend_bull: bool = False
    hma_bull: bool = False
    ichimoku_bull: bool | None = None    # None = inside cloud / indeterminate
    rsi_divergence: str = "NONE"         # BULL / BEAR / NONE
    order_block: dict = field(default_factory=dict)
    fvg: dict = field(default_factory=dict)
    bos: dict = field(default_factory=dict)
    vwap: float = 0.0
    mfi: float = 50.0
    valid: bool = False                  # passes all signal conditions


# ─── Master analysis function ────────────────────────────────────────────────

def analyse(df: pd.DataFrame) -> AnalysisResult:
    """
    Run the full indicator + pattern suite on *df* and return an AnalysisResult.
    *df* must have columns: open, high, low, close, volume (lowercase).
    Needs at least 210 rows for EMA200 to warm up.
    """
    result = AnalysisResult()

    if df is None or len(df) < 50:
        return result

    # Normalise columns
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", pd.Series([0] * len(df))), errors="coerce").fillna(0)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    if len(df) < 30:
        return result

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # ── Indicators ────────────────────────────────────────────────────────────
    rsi_series = _rsi(c, RSI_PERIOD)
    rsi_val = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0

    macd_line, signal_line, histogram = _macd(c, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    macd_val = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
    macd_sig = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0
    macd_hist = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0

    stoch_k, stoch_d = _stoch(h, l, c, STOCH_K, STOCH_D, STOCH_SMOOTH)
    sk = float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else 50.0
    sd = float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else 50.0

    # Stochastic RSI (RSI of RSI then stochasticise)
    rsi_of_rsi = _rsi(rsi_series.dropna(), STOCH_RSI_PERIOD)
    rsi_of_rsi = rsi_of_rsi.reindex(rsi_series.index)
    srsi_k, srsi_d = _stoch(rsi_of_rsi, rsi_of_rsi, rsi_of_rsi,
                             STOCH_RSI_PERIOD, STOCH_RSI_D, STOCH_RSI_K)
    srsi_k_val = float(srsi_k.iloc[-1]) if not pd.isna(srsi_k.iloc[-1]) else 50.0

    bb_upper, bb_mid, bb_lower = _bollinger(c, BB_PERIOD, BB_STDDEV)
    bb_u = float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else float(c.iloc[-1])
    bb_l = float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else float(c.iloc[-1])
    bb_m = float(bb_mid.iloc[-1]) if not pd.isna(bb_mid.iloc[-1]) else float(c.iloc[-1])

    emas = {}
    for p in EMA_PERIODS:
        e = _ema(c, p)
        emas[p] = float(e.iloc[-1]) if not pd.isna(e.iloc[-1]) else float(c.iloc[-1])

    adx_series, plus_di, minus_di = _adx(h, l, c, ADX_PERIOD)
    adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0.0
    pdi = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0
    mdi = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0

    wr_series = _williams_r(h, l, c, WILLIAMS_R_PERIOD)
    wr_val = float(wr_series.iloc[-1]) if not pd.isna(wr_series.iloc[-1]) else -50.0

    cci_series = _cci(h, l, c, CCI_PERIOD)
    cci_val = float(cci_series.iloc[-1]) if not pd.isna(cci_series.iloc[-1]) else 0.0

    sar_series = _parabolic_sar(h, l, SAR_STEP, SAR_MAX)
    sar_val = float(sar_series.iloc[-1]) if not pd.isna(sar_series.iloc[-1]) else float(c.iloc[-1])

    atr_series = _atr(h, l, c, ATR_PERIOD)
    atr_val = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0

    obv_series = _obv(c, v)
    obv_slope = float(obv_series.diff(5).iloc[-1]) if len(obv_series) > 5 else 0.0

    # Current price
    current_price = float(c.iloc[-1])

    # ── Advanced indicators ────────────────────────────────────────────────────
    # Supertrend
    st_series  = _supertrend(h, l, c, period=10, multiplier=3.0)
    st_bull    = int(st_series.iloc[-1]) == 1

    # Hull MA
    hma_series = _hma(c, 20)
    hma_val    = float(hma_series.iloc[-1]) if not pd.isna(hma_series.iloc[-1]) else current_price
    hma_slope  = float(hma_series.diff(3).iloc[-1]) if len(hma_series) > 3 else 0.0

    # Ichimoku
    ichi       = _ichimoku(h, l, c)
    tenkan_val = float(ichi["tenkan"].iloc[-1]) if not pd.isna(ichi["tenkan"].iloc[-1]) else None
    kijun_val  = float(ichi["kijun"].iloc[-1])  if not pd.isna(ichi["kijun"].iloc[-1])  else None
    # Cloud at current bar = span_a/b calculated 26 bars ago and projected forward
    span_a_now = float(ichi["span_a"].iloc[-27]) if len(ichi["span_a"]) > 27 and not pd.isna(ichi["span_a"].iloc[-27]) else None
    span_b_now = float(ichi["span_b"].iloc[-27]) if len(ichi["span_b"]) > 27 and not pd.isna(ichi["span_b"].iloc[-27]) else None

    # VWAP
    has_vol  = bool(float(v.max()) > 0)
    vwap_val = 0.0
    if has_vol:
        vwap_s   = _vwap(h, l, c, v)
        vwap_val = float(vwap_s.iloc[-1]) if not pd.isna(vwap_s.iloc[-1]) else 0.0

    # MFI
    mfi_val = 50.0
    if has_vol:
        mfi_s   = _mfi(h, l, c, v, 14)
        mfi_val = float(mfi_s.iloc[-1]) if not pd.isna(mfi_s.iloc[-1]) else 50.0

    # CMF
    cmf_val = 0.0
    if has_vol:
        cmf_s   = _cmf(h, l, c, v, 20)
        cmf_val = float(cmf_s.iloc[-1]) if not pd.isna(cmf_s.iloc[-1]) else 0.0

    # Heikin Ashi trend
    ha_bull_bars, ha_bear_bars = _heikin_ashi_trend(o, h, l, c, bars=5)

    # RSI Divergence
    rsi_div = detect_rsi_divergence(c, rsi_series)

    # SMC
    bos_data = detect_smc_bos(h, l, c)
    ob_data  = detect_order_blocks(o, h, l, c)
    fvg_data = detect_fvg(h, l, c)

    # ── Candlestick patterns ──────────────────────────────────────────────────
    candle_pats = detect_candlestick_patterns(o, h, l, c)
    is_doji = bool(candle_pats["is_doji"])

    # ── Chart patterns ────────────────────────────────────────────────────────
    chart_pats = detect_chart_patterns(h, l, c)

    # ── Market structure ──────────────────────────────────────────────────────
    mkt_struct = detect_market_structure(h, l)

    # ── S/R ───────────────────────────────────────────────────────────────────
    sr = detect_support_resistance(c)

    # ── Breakout & liquidity ──────────────────────────────────────────────────
    bo = detect_breakout(c, h, l)
    liq = detect_liquidity_sweep(h, l, c)

    # ── Volume analysis ───────────────────────────────────────────────────────
    vol_avg = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else float(v.mean())
    vol_rising = float(v.iloc[-1]) > vol_avg

    # ─── Confluence scoring ───────────────────────────────────────────────────
    # Only indicators that fire for *either* side count toward active.
    # Neutral indicators are skipped entirely so they don't dilute the score.
    buy_score = 0
    sell_score = 0
    active = 0

    def vote(buy_cond: bool, sell_cond: bool, weight: int = 1):
        """Record a vote only when at least one side fires."""
        nonlocal buy_score, sell_score, active
        if buy_cond or sell_cond:   # skip neutral votes — don't dilute score
            active += weight
            if buy_cond:
                buy_score += weight
            if sell_cond:
                sell_score += weight

    # ── Momentum / oscillator indicators ──────────────────────────────────────
    # 1. RSI — use wider bands for trend-following (40/60) in addition to extreme (30/70)
    vote(rsi_val < RSI_OVERSOLD, rsi_val > RSI_OVERBOUGHT, weight=2)      # extreme reversal
    vote(rsi_val < 45, rsi_val > 55, weight=1)                            # directional bias

    # 2. Stoch RSI
    vote(srsi_k_val < STOCH_RSI_OS, srsi_k_val > STOCH_RSI_OB, weight=2)
    vote(srsi_k_val < 45, srsi_k_val > 55, weight=1)                     # directional bias

    # 3. Stochastic — extreme reversal + crossover
    vote(sk < 20, sk > 80, weight=2)
    vote(sk < sd and sk < 50, sk > sd and sk > 50, weight=1)             # cross + position

    # 4. MACD — histogram direction (strong signal)
    vote(macd_hist > 0, macd_hist < 0, weight=2)
    vote(macd_val > macd_sig, macd_val < macd_sig, weight=1)             # line cross

    # 5. Bollinger Bands — price vs bands
    vote(current_price < bb_l, current_price > bb_u, weight=2)           # outside band
    vote(current_price < bb_m, current_price > bb_m, weight=1)           # vs midline

    # ── Trend indicators ──────────────────────────────────────────────────────
    # 6. EMA stack alignment (strong weight)
    ema_bull = emas[9] > emas[21] > emas[50]
    ema_bear = emas[9] < emas[21] < emas[50]
    vote(ema_bull, ema_bear, weight=3)

    # 7. Price vs EMAs (each individually)
    vote(current_price > emas[9], current_price < emas[9], weight=1)
    vote(current_price > emas[21], current_price < emas[21], weight=1)
    vote(current_price > emas[50], current_price < emas[50], weight=1)

    # 8. ADX + DI direction (strong momentum signal)
    vote(pdi > mdi, mdi > pdi, weight=2)                                 # DI cross always counts
    if adx_val > ADX_TREND_THRESHOLD:
        vote(pdi > mdi, mdi > pdi, weight=1)                             # extra weight when trending

    # 9. Parabolic SAR (reliable trend follower)
    vote(sar_val < current_price, sar_val > current_price, weight=2)

    # 10. EMA200 trend bias
    vote(current_price > emas[200], current_price < emas[200], weight=1)

    # ── Mean-reversion / extremes ─────────────────────────────────────────────
    # 11. Williams %R
    vote(wr_val < -80, wr_val > -20, weight=2)
    vote(wr_val < -60, wr_val > -40, weight=1)                           # softer zone

    # 12. CCI
    vote(cci_val < -100, cci_val > 100, weight=2)
    vote(cci_val < -50, cci_val > 50, weight=1)

    # ── Structure / pattern indicators ────────────────────────────────────────
    # 13. Market structure (HH/HL vs LH/LL)
    vote(mkt_struct["bullish"], mkt_struct["bearish"], weight=2)

    # 14. Candlestick patterns
    vote(
        candle_pats["bullish_engulfing"] or candle_pats["is_hammer"],
        candle_pats["bearish_engulfing"] or candle_pats["is_shooting_star"],
        weight=2,
    )

    # 15. Chart patterns
    vote(
        bool(chart_pats.get("double_bottom") or chart_pats.get("inv_head_shoulders")),
        bool(chart_pats.get("double_top") or chart_pats.get("head_shoulders")),
        weight=1,
    )

    # 16. Breakout
    vote(bo["bullish_breakout"], bo["bearish_breakout"], weight=2)

    # 17. Liquidity sweep (sweep low → buy the dip; sweep high → sell the rally)
    vote(liq["sweep_low"], liq["sweep_high"], weight=1)

    # 18. OBV — only include when real volume data exists
    if float(v.max()) > 0:
        vote(obv_slope > 0, obv_slope < 0, weight=1)
        if vol_rising:
            vote(obv_slope > 0, obv_slope < 0, weight=1)   # extra confirmation

    # ── Advanced indicator votes ───────────────────────────────────────────────

    # 19. Supertrend (highly reliable ATR-based trend follower)
    vote(st_bull, not st_bull, weight=3)

    # 20. Hull Moving Average (less lag, tracks trend quickly)
    hma_bull_now = hma_slope > 0 and current_price > hma_val
    hma_bear_now = hma_slope < 0 and current_price < hma_val
    vote(hma_bull_now, hma_bear_now, weight=2)

    # 21. Ichimoku — TK cross (weight 2)
    if tenkan_val is not None and kijun_val is not None:
        vote(tenkan_val > kijun_val, tenkan_val < kijun_val, weight=2)

    # 22. Ichimoku — price vs cloud (weight 2)
    if span_a_now is not None and span_b_now is not None:
        cloud_top = max(span_a_now, span_b_now)
        cloud_bot = min(span_a_now, span_b_now)
        vote(current_price > cloud_top, current_price < cloud_bot, weight=2)

    # 23. Heikin Ashi — 4 or 5 consecutive HA candles (strong trend)
    vote(ha_bull_bars >= 4, ha_bear_bars >= 4, weight=2)
    vote(ha_bull_bars >= 3, ha_bear_bars >= 3, weight=1)    # softer bias

    # 24. RSI Divergence (highest-weight reversal signal)
    vote(rsi_div == "BULL", rsi_div == "BEAR", weight=4)

    # 25. SMC — Break of Structure
    vote(bos_data["bullish_bos"], bos_data["bearish_bos"], weight=3)

    # 26. SMC — Order Blocks (institutional demand/supply zones)
    vote(ob_data["ob_bull"], ob_data["ob_bear"], weight=3)

    # 27. SMC — Fair Value Gaps (momentum imbalance)
    vote(fvg_data["fvg_bull"], fvg_data["fvg_bear"], weight=2)

    # 28. VWAP (only when volume data is real)
    if has_vol and vwap_val > 0:
        vote(current_price > vwap_val, current_price < vwap_val, weight=2)

    # 29. MFI — momentum with volume
    if has_vol:
        vote(mfi_val < 20, mfi_val > 80, weight=2)
        vote(mfi_val < 40, mfi_val > 60, weight=1)

    # 30. CMF — Chaikin money flow
    if has_vol:
        vote(cmf_val > 0.1, cmf_val < -0.1, weight=1)

    # ─── Determine direction ──────────────────────────────────────────────────
    if buy_score >= sell_score:
        direction: Direction = "BUY"
        agreements = buy_score
    else:
        direction = "SELL"
        agreements = sell_score

    strength = (agreements / active * 100) if active > 0 else 0.0

    # ─── EMA200 trend filter (soft — counted in score, not a hard gate) ────────
    trend_confirm = (
        (direction == "BUY" and current_price > emas[200]) or
        (direction == "SELL" and current_price < emas[200])
    )

    # ─── Volume confirmation (soft — OTC feeds often return flat volume) ───────
    # True if volume is non-zero AND rising, or if volume data is unavailable
    vol_nonzero = bool(float(v.iloc[-1]) > 0)
    volume_confirm = (not vol_nonzero) or vol_rising   # no data → pass; has data → must be rising

    # ─── ATR filter (not too low volatility) ─────────────────────────────────
    atr_pct = atr_val / current_price * 100 if current_price > 0 else 0
    atr_ok = atr_pct > 0.005  # at least 0.005% move per bar (relaxed from 0.01%)

    # ─── Signal validity ──────────────────────────────────────────────────────
    # Hard gates: minimum agreements, minimum strength, minimum ADX, volatility
    # Doji is advisory — only blocks when strength is borderline
    # Soft gates: volume and trend — only block when BOTH fail together
    soft_ok = volume_confirm or trend_confirm
    strong_enough = agreements >= MIN_INDICATOR_AGREEMENTS and strength >= MIN_STRENGTH_PCT
    # Allow doji through if signal is strong (strength > 60%)
    doji_ok = (not is_doji) or (strength >= 60.0)
    valid = (
        doji_ok and
        strong_enough and
        adx_val >= MIN_ADX and
        atr_ok and
        soft_ok
    )

    # ── Ichimoku cloud direction summary ──────────────────────────────────────
    ichi_bull: bool | None = None
    if span_a_now is not None and span_b_now is not None:
        cloud_top = max(span_a_now, span_b_now)
        cloud_bot = min(span_a_now, span_b_now)
        if current_price > cloud_top:
            ichi_bull = True
        elif current_price < cloud_bot:
            ichi_bull = False
        # else: inside cloud → None

    result.direction = direction
    result.strength = round(strength, 1)
    result.agreements = agreements
    result.active_indicators = active
    result.adx = round(adx_val, 2)
    result.atr = round(atr_val, 6)
    result.rsi = round(rsi_val, 1)
    result.ema200 = round(emas[200], 6)
    result.close = round(current_price, 6)
    result.volume_confirm = volume_confirm
    result.trend_confirm = trend_confirm
    result.is_doji = is_doji
    result.market_structure = mkt_struct["structure"]
    result.patterns = chart_pats
    result.candle_patterns = candle_pats
    result.support_resistance = sr
    result.breakout = bo
    result.liquidity = liq
    # advanced
    result.supertrend_bull = st_bull
    result.hma_bull        = hma_bull_now
    result.ichimoku_bull   = ichi_bull
    result.rsi_divergence  = rsi_div
    result.order_block     = ob_data
    result.fvg             = fvg_data
    result.bos             = bos_data
    result.vwap            = round(vwap_val, 6)
    result.mfi             = round(mfi_val, 1)
    result.valid = valid
    return result

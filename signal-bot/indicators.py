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
    result.valid = valid
    return result

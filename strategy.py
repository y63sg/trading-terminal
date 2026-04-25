"""
strategy.py
Day Trading + Trend Following + SMC техник анализын модуль.
20 EMA, RSI, Fibonacci, Liquidity Grab, Trendline, Support/Resistance.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    PTA_AVAILABLE = True
except Exception:
    PTA_AVAILABLE = False


# ---------------------------------------------------------------
# 1. INDICATORS
# ---------------------------------------------------------------

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """20 EMA, 50 EMA, RSI(14), ATR(14)-г нэмэх."""
    df = df.copy()
    if PTA_AVAILABLE:
        df["EMA20"] = ta.ema(df["Close"], length=20)
        df["EMA50"] = ta.ema(df["Close"], length=50)
        df["RSI"] = ta.rsi(df["Close"], length=14)
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    else:
        df["EMA20"] = _ema(df["Close"], 20)
        df["EMA50"] = _ema(df["Close"], 50)
        df["RSI"] = _rsi(df["Close"], 14)
        df["ATR"] = _atr(df, 14)
    return df


# ---------------------------------------------------------------
# 2. FIBONACCI
# ---------------------------------------------------------------

def fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Fibonacci retracement түвшнүүд + Golden Zone."""
    diff = high - low
    return {
        "0.0":   low,
        "0.236": low + diff * 0.236,
        "0.382": low + diff * 0.382,
        "0.5":   low + diff * 0.5,    # Golden zone start
        "0.618": low + diff * 0.618,  # Golden zone end (PRIME)
        "0.786": low + diff * 0.786,
        "1.0":   high,
    }


def in_golden_zone(price: float, high: float, low: float, tol: float = 0.0015) -> bool:
    """Ханш 0.5 - 0.618 Fibonacci мужид байгаа эсэх."""
    levels = fibonacci_levels(high, low)
    lo, hi = sorted([levels["0.5"], levels["0.618"]])
    return lo * (1 - tol) <= price <= hi * (1 + tol)


# ---------------------------------------------------------------
# 3. SWING / SMC: Liquidity Grabs (Stop Hunts)
# ---------------------------------------------------------------

def detect_swings(df: pd.DataFrame, window: int = 5) -> Tuple[pd.Series, pd.Series]:
    """Swing high/low илрүүлэх (rolling pivot)."""
    n = window
    highs = df["High"].rolling(2 * n + 1, center=True).apply(
        lambda x: x.iloc[n] if x.iloc[n] == x.max() else np.nan, raw=False
    )
    lows = df["Low"].rolling(2 * n + 1, center=True).apply(
        lambda x: x.iloc[n] if x.iloc[n] == x.min() else np.nan, raw=False
    )
    return highs, lows


def detect_liquidity_grab(df: pd.DataFrame, lookback: int = 30) -> Dict:
    """
    Liquidity Grab (Stop Hunt) илрүүлэх:
    - Ханш сүүлийн swing low-оос доош хатгаж буцаад дээшээ хаалт хийсэн → Bullish grab.
    - Ханш сүүлийн swing high-аас дээш хатгаж буцаад доошоо хаалт хийсэн → Bearish grab.
    """
    if len(df) < lookback + 5:
        return {"bullish_grab": False, "bearish_grab": False, "direction": None}

    recent = df.tail(lookback + 5)
    body = recent.iloc[:-3]            # стоп-н зорилт болсон swing-үүд
    last3 = recent.iloc[-3:]           # сүүлийн 3 лааг grab-ын лаа гэж үзнэ

    swing_high = float(body["High"].max())
    swing_low = float(body["Low"].min())

    last_close = float(last3["Close"].iloc[-1])

    bullish_grab = bool(last3["Low"].min() < swing_low and last_close > swing_low)
    bearish_grab = bool(last3["High"].max() > swing_high and last_close < swing_high)

    return {
        "bullish_grab": bullish_grab,
        "bearish_grab": bearish_grab,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "direction": "BUY" if bullish_grab else ("SELL" if bearish_grab else None),
    }


# ---------------------------------------------------------------
# 4. SUPPORT / RESISTANCE
# ---------------------------------------------------------------

def detect_support_resistance(df: pd.DataFrame, lookback: int = 120,
                              n_levels: int = 3, cluster_pct: float = 0.0015) -> Dict[str, List[float]]:
    """Топ support/resistance түвшин — pivot-ыг кластерлах."""
    recent = df.tail(lookback)
    if recent.empty:
        return {"support": [], "resistance": []}

    raw_highs = sorted(recent["High"].nlargest(n_levels * 4).tolist(), reverse=True)
    raw_lows = sorted(recent["Low"].nsmallest(n_levels * 4).tolist())

    def cluster(levels: List[float]) -> List[float]:
        if not levels:
            return []
        levels = sorted(levels)
        clustered, cur = [], [levels[0]]
        for x in levels[1:]:
            if abs(x - cur[-1]) / cur[-1] < cluster_pct:
                cur.append(x)
            else:
                clustered.append(float(np.mean(cur)))
                cur = [x]
        clustered.append(float(np.mean(cur)))
        return clustered

    return {
        "resistance": cluster(raw_highs)[:n_levels],
        "support": cluster(raw_lows)[:n_levels],
    }


# ---------------------------------------------------------------
# 5. TRENDLINE
# ---------------------------------------------------------------

def trendline_analysis(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """Хялбар трендлайн — lows/highs дээгүүр шугаман регресс."""
    recent = df.tail(lookback).reset_index(drop=True)
    if len(recent) < 10:
        return {"uptrend": False, "downtrend": False}

    x = np.arange(len(recent), dtype=float)
    low_slope, low_intercept = np.polyfit(x, recent["Low"].values, 1)
    high_slope, high_intercept = np.polyfit(x, recent["High"].values, 1)

    cur_low_line = float(low_slope * (len(recent) - 1) + low_intercept)
    cur_high_line = float(high_slope * (len(recent) - 1) + high_intercept)
    last_close = float(recent["Close"].iloc[-1])

    return {
        "uptrend": low_slope > 0 and high_slope > 0,
        "downtrend": low_slope < 0 and high_slope < 0,
        "support_line": cur_low_line,
        "resistance_line": cur_high_line,
        "low_slope": float(low_slope),
        "high_slope": float(high_slope),
        "distance_to_support_pct": (last_close - cur_low_line) / last_close * 100,
        "distance_to_resistance_pct": (cur_high_line - last_close) / last_close * 100,
    }


# ---------------------------------------------------------------
# 6. MARKET STRUCTURE
# ---------------------------------------------------------------

def market_structure(df: pd.DataFrame) -> str:
    """EMA20 vs EMA50 + ханшийн байрлалаар бүтцийг гаргах."""
    last = df.iloc[-1]
    if pd.isna(last.get("EMA20")) or pd.isna(last.get("EMA50")):
        return "UNKNOWN"
    if last["Close"] > last["EMA20"] > last["EMA50"]:
        return "BULLISH"
    if last["Close"] < last["EMA20"] < last["EMA50"]:
        return "BEARISH"
    return "RANGING"


# ---------------------------------------------------------------
# 7. POSITION SIZING (1% эрсдэл)
# ---------------------------------------------------------------

def position_size(account_balance: float, risk_pct: float, entry: float,
                  stop_loss: float, pip_value_per_lot: float = 10.0,
                  pip_size: float = 0.1) -> Dict[str, float]:
    """
    1% эрсдэлийн lot тооцоолол. XAUUSD-н хувьд default: 1 pip = $0.10/oz, 100oz/lot.
    """
    risk_amount = account_balance * (risk_pct / 100.0)
    distance_pips = abs(entry - stop_loss) / pip_size
    if distance_pips <= 0:
        return {"lot_size": 0.0, "risk_usd": 0.0, "stop_pips": 0.0}
    lot_size = risk_amount / (distance_pips * pip_value_per_lot)
    return {
        "lot_size": round(lot_size, 2),
        "risk_usd": round(risk_amount, 2),
        "stop_pips": round(distance_pips, 1),
    }


# ---------------------------------------------------------------
# 8. SUMMARY snapshot for AI
# ---------------------------------------------------------------

def snapshot_for_ai(df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
    """Claude API-д өгөх компакт техник snapshot."""
    if df.empty or len(df) < 30:
        return {}

    last = df.iloc[-1]
    swing_high = float(df["High"].tail(60).max())
    swing_low = float(df["Low"].tail(60).min())
    fib = fibonacci_levels(swing_high, swing_low)
    sr = detect_support_resistance(df)
    tl = trendline_analysis(df)
    grab = detect_liquidity_grab(df)
    structure = market_structure(df)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "price": float(last["Close"]),
        "ema20": float(last["EMA20"]),
        "ema50": float(last["EMA50"]),
        "rsi": float(last["RSI"]),
        "atr": float(last["ATR"]),
        "structure": structure,
        "swing_high_60": swing_high,
        "swing_low_60": swing_low,
        "fib_levels": fib,
        "in_golden_zone": in_golden_zone(float(last["Close"]), swing_high, swing_low),
        "support": sr["support"],
        "resistance": sr["resistance"],
        "trendline": tl,
        "liquidity_grab": grab,
    }

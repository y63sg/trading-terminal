"""
strategy.py — Pure pandas/numpy strategy engine
=================================================
Day-trading + trend-following indicators, SMC primitives,
Fibonacci, S/R clustering, trendline analysis.

No pandas_ta dependency. Compatible with Python 3.10–3.14.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────────────
# CORE INDICATORS (pure pandas/numpy)
# ────────────────────────────────────────────────────────────────────────────
def ema(series: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False, min_periods=1).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().bfill()


# ────────────────────────────────────────────────────────────────────────────
# FIBONACCI
# ────────────────────────────────────────────────────────────────────────────
def fibonacci_levels(df: pd.DataFrame, lookback: int = 60) -> Dict[str, float]:
    """Fibonacci retracement on the most recent swing range."""
    if df.empty or len(df) < 5:
        return {}
    window = df.tail(lookback)
    hi = float(window["high"].max())
    lo = float(window["low"].min())
    if hi <= lo:
        return {}
    diff = hi - lo
    return {
        "high":  hi,
        "low":   lo,
        "0.0":   hi,
        "0.236": hi - 0.236 * diff,
        "0.382": hi - 0.382 * diff,
        "0.5":   hi - 0.5 * diff,
        "0.618": hi - 0.618 * diff,
        "0.786": hi - 0.786 * diff,
        "1.0":   lo,
    }


def in_golden_zone(price: float, fib: Dict[str, float]) -> bool:
    """Check if price is in 0.5–0.618 zone."""
    if not fib:
        return False
    upper = max(fib.get("0.5", 0), fib.get("0.618", 0))
    lower = min(fib.get("0.5", 0), fib.get("0.618", 0))
    return lower <= price <= upper


# ────────────────────────────────────────────────────────────────────────────
# SWINGS / MARKET STRUCTURE
# ────────────────────────────────────────────────────────────────────────────
def detect_swings(df: pd.DataFrame, k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Detect swing highs/lows. k = number of bars on each side."""
    swings: Dict[str, List[Dict[str, Any]]] = {"highs": [], "lows": []}
    if df.empty or len(df) < 2 * k + 1:
        return swings

    highs = df["high"].values
    lows = df["low"].values
    idx = df.index

    for i in range(k, len(df) - k):
        win_h = highs[i - k: i + k + 1]
        win_l = lows[i - k: i + k + 1]
        if highs[i] == win_h.max():
            swings["highs"].append({"idx": idx[i], "price": float(highs[i])})
        if lows[i] == win_l.min():
            swings["lows"].append({"idx": idx[i], "price": float(lows[i])})

    # Keep most recent 8 each
    swings["highs"] = swings["highs"][-8:]
    swings["lows"] = swings["lows"][-8:]
    return swings


def market_structure(df: pd.DataFrame, swings: Dict[str, List[Dict[str, Any]]]) -> str:
    """Classify market structure as UPTREND / DOWNTREND / RANGE based on HH/HL or LH/LL."""
    highs = swings.get("highs", [])
    lows = swings.get("lows", [])
    if len(highs) < 2 or len(lows) < 2:
        return "RANGE"

    last_two_highs = [h["price"] for h in highs[-2:]]
    last_two_lows = [l["price"] for l in lows[-2:]]

    higher_highs = last_two_highs[1] > last_two_highs[0]
    higher_lows = last_two_lows[1] > last_two_lows[0]
    lower_highs = last_two_highs[1] < last_two_highs[0]
    lower_lows = last_two_lows[1] < last_two_lows[0]

    if higher_highs and higher_lows:
        return "UPTREND"
    if lower_highs and lower_lows:
        return "DOWNTREND"
    return "RANGE"


# ────────────────────────────────────────────────────────────────────────────
# SMART MONEY CONCEPTS — Liquidity grabs
# ────────────────────────────────────────────────────────────────────────────
def detect_liquidity_grab(
    df: pd.DataFrame,
    swings: Dict[str, List[Dict[str, Any]]],
    lookback: int = 5,
) -> Dict[str, Any]:
    """
    Detect a liquidity grab / stop hunt:
    - Wick pierces a recent swing high/low
    - But close returns inside the prior range (rejection)
    """
    if df.empty or len(df) < 5:
        return {"detected": False}

    last = df.iloc[-1]
    recent = df.tail(lookback)

    # Last swing high/low candidates
    swing_highs = [h["price"] for h in swings.get("highs", [])[-3:]]
    swing_lows = [l["price"] for l in swings.get("lows", [])[-3:]]

    # Bull liquidity grab: low pierces recent swing low, close above it
    if swing_lows:
        nearest_low = max(swing_lows)  # most relevant nearby support
        for s_low in swing_lows:
            if last["low"] < s_low and last["close"] > s_low:
                return {
                    "detected": True,
                    "type": "BULL",
                    "level": float(s_low),
                    "wick": float(s_low - last["low"]),
                    "comment": "Stop hunt below support, bullish reclaim.",
                }

    # Bear liquidity grab: high pierces recent swing high, close below it
    if swing_highs:
        for s_high in swing_highs:
            if last["high"] > s_high and last["close"] < s_high:
                return {
                    "detected": True,
                    "type": "BEAR",
                    "level": float(s_high),
                    "wick": float(last["high"] - s_high),
                    "comment": "Stop hunt above resistance, bearish rejection.",
                }

    return {"detected": False}


# ────────────────────────────────────────────────────────────────────────────
# SUPPORT / RESISTANCE — pivot clustering
# ────────────────────────────────────────────────────────────────────────────
def detect_support_resistance(
    df: pd.DataFrame,
    bins: int = 30,
    top_n: int = 5,
) -> Dict[str, List[float]]:
    """Cluster pivot prices to find S/R levels."""
    if df.empty or len(df) < 20:
        return {"support": [], "resistance": []}

    price = float(df["close"].iloc[-1])
    highs = df["high"].values
    lows = df["low"].values

    all_prices = np.concatenate([highs, lows])
    if all_prices.size == 0:
        return {"support": [], "resistance": []}

    hist, edges = np.histogram(all_prices, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    # Sort bins by frequency
    sorted_idx = np.argsort(hist)[::-1]

    levels = []
    for i in sorted_idx:
        if hist[i] < 2:
            continue
        levels.append(float(centers[i]))
        if len(levels) >= top_n * 2:
            break

    resistance = sorted([lvl for lvl in levels if lvl > price])[:top_n]
    support = sorted([lvl for lvl in levels if lvl < price], reverse=True)[:top_n]

    return {"support": support, "resistance": resistance}


# ────────────────────────────────────────────────────────────────────────────
# TRENDLINE — linear regression on swing pivots
# ────────────────────────────────────────────────────────────────────────────
def trendline_analysis(df: pd.DataFrame, lookback: int = 60) -> Dict[str, Any]:
    """Linear regression slope on closing prices over lookback bars."""
    if df.empty or len(df) < 10:
        return {"direction": "FLAT", "slope": 0.0, "r2": 0.0}

    window = df["close"].tail(lookback).values
    if len(window) < 5:
        return {"direction": "FLAT", "slope": 0.0, "r2": 0.0}

    x = np.arange(len(window), dtype=float)
    y = window.astype(float)

    # Linear fit
    try:
        coef = np.polyfit(x, y, 1)
        slope = float(coef[0])
        intercept = float(coef[1])
        y_pred = slope * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    except Exception:
        return {"direction": "FLAT", "slope": 0.0, "r2": 0.0}

    avg_price = float(y.mean()) or 1.0
    pct_per_bar = (slope / avg_price) * 100

    if pct_per_bar > 0.02:
        direction = "UP"
    elif pct_per_bar < -0.02:
        direction = "DOWN"
    else:
        direction = "FLAT"

    return {
        "direction": direction,
        "slope": slope,
        "r2": float(r2),
        "current_line": float(slope * (len(window) - 1) + intercept),
    }


# ────────────────────────────────────────────────────────────────────────────
# POSITION SIZING — 1% risk model
# ────────────────────────────────────────────────────────────────────────────
def position_size(
    account_balance: float,
    risk_pct: float,
    entry: float,
    stop_loss: float,
    pip_value: float = 0.01,
    contract_size: float = 100.0,
) -> Dict[str, float]:
    """
    Calculate position size to risk `risk_pct` of `account_balance`.
    Default values target XAUUSD: 1 pip = $0.01 movement, 100oz contract.
    """
    if entry <= 0 or stop_loss <= 0 or entry == stop_loss:
        return {"lots": 0.0, "risk_usd": 0.0, "pips": 0.0}

    risk_usd = account_balance * risk_pct
    distance = abs(entry - stop_loss)
    pips = distance / pip_value

    # USD per pip per 1 lot
    usd_per_pip_per_lot = pip_value * contract_size  # 0.01 * 100 = $1.00 for XAUUSD
    if usd_per_pip_per_lot <= 0:
        return {"lots": 0.0, "risk_usd": risk_usd, "pips": pips}

    lots = risk_usd / (pips * usd_per_pip_per_lot)
    return {
        "lots": round(max(0.0, lots), 2),
        "risk_usd": round(risk_usd, 2),
        "pips": round(pips, 1),
    }


# ────────────────────────────────────────────────────────────────────────────
# AI SNAPSHOT — compact JSON for the model
# ────────────────────────────────────────────────────────────────────────────
def snapshot_for_ai(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    fib: Dict[str, float],
    sr: Dict[str, List[float]],
    swings: Dict[str, List[Dict[str, Any]]],
    trend: Dict[str, Any],
    structure: str,
    liq_grab: Dict[str, Any],
    rsi_value: float,
    ema20: float,
    ema50: float,
    atr_value: float,
) -> Dict[str, Any]:
    """Build a compact dict that summarizes the current market state."""
    if df.empty:
        return {}

    last = df.iloc[-1]
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": str(df.index[-1]),
        "price": float(last["close"]),
        "ohlc": {
            "open":  float(last["open"]),
            "high":  float(last["high"]),
            "low":   float(last["low"]),
            "close": float(last["close"]),
        },
        "indicators": {
            "ema20": float(ema20),
            "ema50": float(ema50),
            "rsi14": float(rsi_value),
            "atr14": float(atr_value),
        },
        "fibonacci": {k: float(v) for k, v in fib.items()} if fib else {},
        "support_resistance": {
            "support":    [float(s) for s in sr.get("support", [])[:3]],
            "resistance": [float(r) for r in sr.get("resistance", [])[:3]],
        },
        "swings": {
            "last_high": float(swings["highs"][-1]["price"]) if swings.get("highs") else None,
            "last_low":  float(swings["lows"][-1]["price"]) if swings.get("lows") else None,
        },
        "trendline": {
            "direction": trend.get("direction", "FLAT"),
            "slope":     float(trend.get("slope", 0.0)),
            "r2":        float(trend.get("r2", 0.0)),
        },
        "structure": structure,
        "liquidity_grab": liq_grab,
        "in_golden_zone": in_golden_zone(float(last["close"]), fib),
    }
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

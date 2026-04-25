"""
signal_engine.py
1) Market Execution — одоо орох NOW: BUY/SELL дохио.
2) Pending Orders   — Limit/Trendline ALERT хэлбэрээр ирээдүйд орох төлөвлөгөө.

Стратеги: 20 EMA + RSI + Price action + Liquidity grab + Fib Golden Zone + S/R + Trendline.
Risk:Reward 1:2 – 1:4.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

import strategy as S


# ---------------------------------------------------------------
# Дотоод туслах
# ---------------------------------------------------------------

@dataclass
class Trade:
    kind: str              # "MARKET" | "LIMIT" | "TRENDLINE"
    side: str              # "BUY" | "SELL"
    label: str             # UI текст
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    rr1: float
    rr2: float
    confidence: int        # 0–100
    reasons: List[str]


def _round(x: float, decimals: int = 2) -> float:
    return float(round(x, decimals))


def _atr_distance(atr: float, mult: float) -> float:
    return atr * mult if atr and not np.isnan(atr) else 0.0


# ---------------------------------------------------------------
# 1) MARKET EXECUTION
# ---------------------------------------------------------------

def market_execution_signal(df: pd.DataFrame,
                            symbol: str = "XAUUSD",
                            news_bias: Optional[str] = None,
                            rr_target: float = 3.0) -> Optional[Trade]:
    """
    Одоо яг орох эсэхийг шалгана. Бүх нөхцөл хангавал NOW: BUY/SELL дохио буцаана.
    Нөхцлүүд (нэгэн зэрэг хангагдвал):
      - 20 EMA-ийн дагуу price action (close > EMA20 buy / < EMA20 sell)
      - RSI зөв муж (40–65 buy / 35–60 sell, хэт overbought/oversold-ыг шүүх)
      - Liquidity grab (stop hunt) илэрсэн
      - Fib Golden Zone эсвэл хүчтэй S/R дээр зөв чиглэлтэй retest
    """
    if len(df) < 60:
        return None

    last = df.iloc[-1]
    price = float(last["Close"])
    ema20 = float(last["EMA20"])
    rsi = float(last["RSI"])
    atr = float(last["ATR"])

    swing_high = float(df["High"].tail(60).max())
    swing_low = float(df["Low"].tail(60).min())
    fib = S.fibonacci_levels(swing_high, swing_low)
    sr = S.detect_support_resistance(df)
    grab = S.detect_liquidity_grab(df)
    structure = S.market_structure(df)

    reasons: List[str] = []
    score_buy = 0
    score_sell = 0

    # EMA condition
    if price > ema20:
        score_buy += 25; reasons.append("Ханш 20 EMA-ийн дээр")
    elif price < ema20:
        score_sell += 25; reasons.append("Ханш 20 EMA-ийн доор")

    # RSI condition (хэт нэг талд биш)
    if 40 <= rsi <= 65 and structure == "BULLISH":
        score_buy += 20; reasons.append(f"RSI bullish дунд муж ({rsi:.1f})")
    if 35 <= rsi <= 60 and structure == "BEARISH":
        score_sell += 20; reasons.append(f"RSI bearish дунд муж ({rsi:.1f})")
    if rsi > 75:
        score_buy -= 25; reasons.append("RSI overbought — long-д сөрөг")
    if rsi < 25:
        score_sell -= 25; reasons.append("RSI oversold — short-д сөрөг")

    # Liquidity grab
    if grab["bullish_grab"]:
        score_buy += 25; reasons.append("Bullish liquidity grab (stop hunt доороос)")
    if grab["bearish_grab"]:
        score_sell += 25; reasons.append("Bearish liquidity grab (stop hunt дээрээс)")

    # Golden Zone retest
    in_gz = S.in_golden_zone(price, swing_high, swing_low)
    if in_gz and structure == "BULLISH":
        score_buy += 15; reasons.append("Fib Golden Zone (0.5–0.618) дээр retest")
    if in_gz and structure == "BEARISH":
        score_sell += 15; reasons.append("Fib Golden Zone (0.5–0.618) дээр retest")

    # Support/Resistance proximity (ойрхон туссан)
    nearest_sup = min(sr["support"], key=lambda x: abs(x - price)) if sr["support"] else None
    nearest_res = min(sr["resistance"], key=lambda x: abs(x - price)) if sr["resistance"] else None
    if nearest_sup and abs(price - nearest_sup) / price < 0.0015:
        score_buy += 15; reasons.append(f"Хүчтэй support {nearest_sup:.2f} дээр reaction")
    if nearest_res and abs(price - nearest_res) / price < 0.0015:
        score_sell += 15; reasons.append(f"Хүчтэй resistance {nearest_res:.2f} дээр reject")

    # News bias туслалт
    if news_bias == "UP":
        score_buy += 10; reasons.append("News bias UP")
    elif news_bias == "DOWN":
        score_sell += 10; reasons.append("News bias DOWN")

    # Шийдвэр
    threshold = 60
    if score_buy >= threshold and score_buy > score_sell:
        side = "BUY"
        entry = price
        sl = min(grab["swing_low"] if grab["swing_low"] else price - _atr_distance(atr, 1.5),
                 price - _atr_distance(atr, 1.5))
        risk = entry - sl
        if risk <= 0:
            return None
        tp1 = entry + risk * 2.0
        tp2 = entry + risk * rr_target
        return Trade("MARKET", "BUY",
                     f"NOW: BUY {symbol}",
                     _round(entry), _round(sl), _round(tp1), _round(tp2),
                     2.0, _round(rr_target),
                     min(score_buy, 100), reasons)

    if score_sell >= threshold and score_sell > score_buy:
        side = "SELL"
        entry = price
        sl = max(grab["swing_high"] if grab["swing_high"] else price + _atr_distance(atr, 1.5),
                 price + _atr_distance(atr, 1.5))
        risk = sl - entry
        if risk <= 0:
            return None
        tp1 = entry - risk * 2.0
        tp2 = entry - risk * rr_target
        return Trade("MARKET", "SELL",
                     f"NOW: SELL {symbol}",
                     _round(entry), _round(sl), _round(tp1), _round(tp2),
                     2.0, _round(rr_target),
                     min(score_sell, 100), reasons)

    return None


# ---------------------------------------------------------------
# 2) PENDING ORDERS
# ---------------------------------------------------------------

def pending_orders(df: pd.DataFrame, symbol: str = "XAUUSD",
                   rr_target: float = 3.0) -> List[Trade]:
    """
    Ханш Golden Zone, S/R, Trendline-д хүрээгүй байгаа үед — ирээдүйн хүлээлт.
    Limit Order болон Trendline Alert-уудыг гаргана.
    """
    if len(df) < 60:
        return []

    last = df.iloc[-1]
    price = float(last["Close"])
    atr = float(last["ATR"])
    ema20 = float(last["EMA20"])
    structure = S.market_structure(df)

    swing_high = float(df["High"].tail(60).max())
    swing_low = float(df["Low"].tail(60).min())
    fib = S.fibonacci_levels(swing_high, swing_low)
    sr = S.detect_support_resistance(df)
    tl = S.trendline_analysis(df)

    pendings: List[Trade] = []

    # ----- Fib Golden Zone-д тулгуурласан LIMIT -----
    gz_mid = (fib["0.5"] + fib["0.618"]) / 2
    in_gz_now = S.in_golden_zone(price, swing_high, swing_low)
    if not in_gz_now:
        if structure == "BULLISH" and price > gz_mid:
            entry = fib["0.618"]                 # доош нь буцаж ирэхэд BUY LIMIT
            sl = swing_low - _atr_distance(atr, 0.8)
            risk = entry - sl
            if risk > 0:
                tp1 = entry + risk * 2
                tp2 = entry + risk * rr_target
                conf = 65 if structure == "BULLISH" else 55
                pendings.append(Trade(
                    "LIMIT", "BUY",
                    f"LIMIT ORDER: BUY LIMIT @ {entry:.2f} (Fib 0.618)",
                    _round(entry), _round(sl), _round(tp1), _round(tp2),
                    2.0, _round(rr_target), conf,
                    [f"Ханш {price:.2f} → Golden Zone {fib['0.5']:.2f}-{fib['0.618']:.2f} руу буцах магадлалтай",
                     f"Bullish структур, retest BUY"]))
        if structure == "BEARISH" and price < gz_mid:
            entry = fib["0.5"]                   # дээш нь буцаж очвол SELL LIMIT
            sl = swing_high + _atr_distance(atr, 0.8)
            risk = sl - entry
            if risk > 0:
                tp1 = entry - risk * 2
                tp2 = entry - risk * rr_target
                pendings.append(Trade(
                    "LIMIT", "SELL",
                    f"LIMIT ORDER: SELL LIMIT @ {entry:.2f} (Fib 0.5)",
                    _round(entry), _round(sl), _round(tp1), _round(tp2),
                    2.0, _round(rr_target), 65,
                    [f"Ханш {price:.2f} → Golden Zone {fib['0.5']:.2f}-{fib['0.618']:.2f} руу буцах магадлалтай",
                     "Bearish структур, retest SELL"]))

    # ----- S/R LIMIT -----
    if sr["support"]:
        nearest_sup = max([s for s in sr["support"] if s < price] or [None])
        if nearest_sup and (price - nearest_sup) / price > 0.002:
            entry = nearest_sup
            sl = entry - _atr_distance(atr, 1.2)
            risk = entry - sl
            if risk > 0:
                tp1 = entry + risk * 2
                tp2 = entry + risk * rr_target
                pendings.append(Trade(
                    "LIMIT", "BUY",
                    f"LIMIT ORDER: BUY LIMIT @ {entry:.2f} (Support)",
                    _round(entry), _round(sl), _round(tp1), _round(tp2),
                    2.0, _round(rr_target), 60,
                    [f"Хүчтэй support {entry:.2f}, retest BUY",
                     "RR 1:2–1:4-ийн зорилго"]))
    if sr["resistance"]:
        nearest_res = min([r for r in sr["resistance"] if r > price] or [None])
        if nearest_res and (nearest_res - price) / price > 0.002:
            entry = nearest_res
            sl = entry + _atr_distance(atr, 1.2)
            risk = sl - entry
            if risk > 0:
                tp1 = entry - risk * 2
                tp2 = entry - risk * rr_target
                pendings.append(Trade(
                    "LIMIT", "SELL",
                    f"LIMIT ORDER: SELL LIMIT @ {entry:.2f} (Resistance)",
                    _round(entry), _round(sl), _round(tp1), _round(tp2),
                    2.0, _round(rr_target), 60,
                    [f"Хүчтэй resistance {entry:.2f}, reject SELL"]))

    # ----- TRENDLINE ALERT -----
    dist_sup = tl.get("distance_to_support_pct", 99)
    dist_res = tl.get("distance_to_resistance_pct", 99)
    if 0 < dist_sup < 0.4 and tl.get("uptrend"):
        line = tl["support_line"]
        entry_break = line - _atr_distance(atr, 0.5)   # breakdown
        entry_bounce = line                            # bounce buy
        sl_b = entry_bounce - _atr_distance(atr, 1.2)
        risk = entry_bounce - sl_b
        if risk > 0:
            pendings.append(Trade(
                "TRENDLINE", "BUY",
                f"TRENDLINE ALERT: ascending TL {line:.2f} ойртож байна — bounce BUY",
                _round(entry_bounce), _round(sl_b),
                _round(entry_bounce + risk * 2), _round(entry_bounce + risk * rr_target),
                2.0, _round(rr_target), 55,
                ["Uptrend трендлайнаас rejection — buy зөвлөж байна",
                 f"Хэрэв {entry_break:.2f}-аас доош хаалт хийвэл breakout SELL руу шилжинэ"]))
    if 0 < dist_res < 0.4 and tl.get("downtrend"):
        line = tl["resistance_line"]
        entry = line
        sl = entry + _atr_distance(atr, 1.2)
        risk = sl - entry
        if risk > 0:
            pendings.append(Trade(
                "TRENDLINE", "SELL",
                f"TRENDLINE ALERT: descending TL {line:.2f} ойртож байна — rejection SELL",
                _round(entry), _round(sl),
                _round(entry - risk * 2), _round(entry - risk * rr_target),
                2.0, _round(rr_target), 55,
                ["Downtrend трендлайн руу retest — sell зөвлөж байна"]))

    return pendings


# ---------------------------------------------------------------
# Бүх дохиог хослуулсан хариу
# ---------------------------------------------------------------

def generate_all_signals(df: pd.DataFrame, symbol: str = "XAUUSD",
                         news_bias: Optional[str] = None,
                         rr_target: float = 3.0) -> Dict:
    market = market_execution_signal(df, symbol, news_bias, rr_target)
    pendings = pending_orders(df, symbol, rr_target)
    return {
        "market": asdict(market) if market else None,
        "pendings": [asdict(t) for t in pendings],
    }

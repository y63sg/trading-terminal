"""
news_analyzer.py
Сүүлийн 10 сарын макро мэдээний (NFP, CPI, Unemployment) Actual vs Forecast зөрүүг
XAUUSD-н 15м/1цаг pip хөдөлгөөнтэй харьцуулж магадлал гаргадаг модуль.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# Загвар: Actual > Forecast байх үеийн Gold-ын урвалын thumb-rule
# NFP/Unemployment (USD-эрх): актив > прогноз → USD хүчтэй → XAU доош
# CPI (инфляц): актив > прогноз → бодит хүү дээш → XAU доош (эхэндээ)
# ---------------------------------------------------------------

DEFAULT_PATH = Path(__file__).parent / "data" / "news_history.json"


def load_news_history(path: Optional[str] = None) -> pd.DataFrame:
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        # хоосон тохиолдолд хоосон df буцаах
        return pd.DataFrame(columns=["date", "event", "actual", "forecast",
                                     "previous", "xauusd_15m", "xauusd_1h"])
    with open(p, "r", encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["deviation"] = df["actual"] - df["forecast"]
    df["abs_deviation"] = df["deviation"].abs()
    df["deviation_pct"] = df["deviation"] / df["forecast"].replace(0, np.nan).abs() * 100
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------
# Үндсэн магадлалын логик
# ---------------------------------------------------------------

def calculate_news_probability(news_df: pd.DataFrame,
                               event_type: str,
                               deviation: float) -> Dict:
    """
    Өгсөн мэдээний deviation-д тулгуурлан XAUUSD ямар чиглэлд хэдэн pip хөдөлдөг
    байсныг түүхэн өгөгдлөөс гаргаж магадлал тооцно.
    """
    relevant = news_df[news_df["event"] == event_type].copy()
    if relevant.empty:
        return {
            "direction": "NEUTRAL",
            "probability": 0.5,
            "sample_size": 0,
            "avg_pips_15m": 0.0,
            "avg_pips_1h": 0.0,
            "max_abs_pips_15m": 0.0,
            "volatility_breakout_recommended": False,
            "comment": f"{event_type}-н түүхэн өгөгдөл байхгүй.",
        }

    # 1) deviation-аар дөхөм sample-ууд
    same_sign = relevant[np.sign(relevant["deviation"]) == np.sign(deviation)] \
        if deviation != 0 else relevant
    if same_sign.empty:
        same_sign = relevant

    # 2) Deviation хэмжээгээр ойролцоо багц авах
    closeness = (same_sign["deviation"] - deviation).abs()
    threshold = max(abs(deviation) * 0.6, same_sign["abs_deviation"].median() * 0.5, 0.05)
    similar = same_sign[closeness <= threshold]
    if len(similar) < 3:                        # хэт цөөн бол бүх same-sign-аар
        similar = same_sign

    avg_15m = float(similar["xauusd_15m"].mean())
    avg_1h = float(similar["xauusd_1h"].mean())
    max_abs = float(similar["xauusd_15m"].abs().max())

    up = int((similar["xauusd_15m"] > 0).sum())
    down = int((similar["xauusd_15m"] < 0).sum())
    total = up + down
    if total == 0:
        prob = 0.5
        direction = "NEUTRAL"
    else:
        if avg_15m > 0:
            direction = "UP"      # XAU дээшээ
            prob = up / total
        elif avg_15m < 0:
            direction = "DOWN"
            prob = down / total
        else:
            direction = "NEUTRAL"
            prob = 0.5

    # Volatility breakout зөвлөмж: deviation нь түүхэн дунджаас илүү бол
    median_dev = float(relevant["abs_deviation"].median()) or 0.0
    vol_breakout = abs(deviation) > median_dev * 1.2

    return {
        "direction": direction,
        "probability": round(prob, 3),
        "sample_size": int(len(similar)),
        "avg_pips_15m": round(avg_15m, 1),
        "avg_pips_1h": round(avg_1h, 1),
        "max_abs_pips_15m": round(max_abs, 1),
        "volatility_breakout_recommended": bool(vol_breakout),
        "comment": _build_comment(event_type, deviation, avg_15m, prob),
    }


def _build_comment(event: str, dev: float, avg_pips: float, prob: float) -> str:
    """Хүний унших товч хувилбар."""
    if abs(dev) < 1e-6:
        return f"{event}: deviation 0 — өчүүхэн савалгаа л хүлээгдэж байна."
    side = "хүчтэй" if abs(avg_pips) > 100 else "дунд"
    direction = "DOWN (XAU доош)" if avg_pips < 0 else "UP (XAU дээш)"
    return (f"{event} deviation={dev:+.2f}: түүхэн дундаар XAUUSD "
            f"15 минутад ~{avg_pips:+.0f} pip {side} {direction}. "
            f"Чиглэлийн магадлал {prob*100:.0f}%.")


# ---------------------------------------------------------------
# 5–15 минутын дараах Institutional move зөвлөмж
# ---------------------------------------------------------------

def institutional_move_advice(news_df: pd.DataFrame, event_type: str,
                              deviation: float) -> Dict:
    """
    Мэдээ гарснаас 5–15 мин дараа эхэлдэг 'Institutional move'-ийн дунджийг гаргана.
    """
    rel = news_df[news_df["event"] == event_type].copy()
    if rel.empty:
        return {"hold_minutes": 0, "expected_pips": 0, "advice": "Өгөгдөл алга"}

    # 1h - 15m зөрүү = 15м дараах хэсгийн нэмэлт хөдөлгөөн
    rel["post_window_pips"] = rel["xauusd_1h"] - rel["xauusd_15m"]
    same_sign = rel[np.sign(rel["deviation"]) == np.sign(deviation)] if deviation != 0 else rel
    if same_sign.empty:
        same_sign = rel

    expected = float(same_sign["post_window_pips"].mean())
    hit_rate = float((np.sign(same_sign["post_window_pips"]) == np.sign(same_sign["xauusd_15m"])).mean())

    direction = "анхны савалгааны чиглэл рүү үргэлжилнэ" if hit_rate > 0.55 \
                else "анхны савалгаа эргэх магадлалтай"
    return {
        "hold_minutes": 15,
        "expected_pips": round(expected, 1),
        "follow_through_rate": round(hit_rate, 2),
        "advice": (f"Мэдээ гарснаас 5–15 мин хүлээ. Дараа нь {direction}. "
                   f"Хүлээгдэж буй нэмэлт хөдөлгөөн ~{expected:+.0f} pip."),
    }


# ---------------------------------------------------------------
# Хураангуй (UI-д хэрэглэх)
# ---------------------------------------------------------------

def summarize_recent(news_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    if news_df.empty:
        return news_df
    cols = ["date", "event", "actual", "forecast", "deviation",
            "xauusd_15m", "xauusd_1h"]
    return news_df.sort_values("date", ascending=False).head(limit)[cols].reset_index(drop=True)

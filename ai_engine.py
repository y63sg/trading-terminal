"""
ai_engine.py
Claude API-тай харьцах wrapper. Snapshot + News + Signal-уудыг нэгтгэн
эцсийн BUY/SELL/HOLD дүгнэлт гаргуулна.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except Exception:
    ANTHROPIC_AVAILABLE = False


SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"

DEFAULT_MODEL = "claude-sonnet-4-6"  # Хурдтай, хямд үнэтэй мэргэжлийн хувилбар


def _load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return "You are a professional trading assistant."


def get_ai_decision(snapshot: Dict,
                    news_analysis: Dict,
                    signals: Dict,
                    api_key: Optional[str] = None,
                    model: str = DEFAULT_MODEL) -> Dict:
    """
    Claude-аас эцсийн шийдвэрийг JSON форматаар гаргуулна.
    Хэрэв API key байхгүй бол rule-based fallback хариу буцаана.
    """
    if not ANTHROPIC_AVAILABLE or not (api_key or os.getenv("ANTHROPIC_API_KEY")):
        return _fallback_decision(snapshot, news_analysis, signals)

    client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    system_prompt = _load_system_prompt()

    user_payload = {
        "technical_snapshot": snapshot,
        "news_analysis": news_analysis,
        "rule_engine_signals": signals,
        "instructions": (
            "Дээрх өгөгдлүүд дээр үндэслээд минийх стратегийн дагуу "
            "BUY / SELL / HOLD шийдвэр гарга. Зөвхөн дараах форматын JSON-оор хариул:\n"
            '{"decision":"BUY|SELL|HOLD","confidence":0-100,"probability_up":0-1,'
            '"entry":number,"stop_loss":number,"take_profit":number,'
            '"rr":number,"timeframe":"15m|1h","reasoning":"товч 2-4 өгүүлбэр",'
            '"warnings":["..."],"market_or_pending":"MARKET|PENDING"}'
        ),
    }

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1200,
            system=system_prompt,
            messages=[{"role": "user", "content": json.dumps(user_payload, default=str)}],
        )
        text = resp.content[0].text.strip()
        # JSON-г олох
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return {"decision": "HOLD", "confidence": 0, "reasoning": text, "warnings": ["Could not parse JSON"]}
    except Exception as e:
        fb = _fallback_decision(snapshot, news_analysis, signals)
        fb["warnings"] = fb.get("warnings", []) + [f"AI алдаа: {e}"]
        return fb


def _fallback_decision(snapshot: Dict, news: Dict, signals: Dict) -> Dict:
    """API-гүй үед: rule engine-ний дохиог нэгтгэж эцсийн дүгнэлт хийнэ."""
    market = signals.get("market")
    if market:
        return {
            "decision": market["side"],
            "confidence": market["confidence"],
            "probability_up": 0.6 if market["side"] == "BUY" else 0.4,
            "entry": market["entry"],
            "stop_loss": market["stop_loss"],
            "take_profit": market["take_profit_2"],
            "rr": market["rr2"],
            "timeframe": snapshot.get("timeframe", "15m"),
            "reasoning": "Rule engine market signal: " + "; ".join(market["reasons"][:3]),
            "warnings": ["AI API хэрэглээгүй (fallback)."],
            "market_or_pending": "MARKET",
        }
    pendings = signals.get("pendings", [])
    if pendings:
        p = pendings[0]
        return {
            "decision": p["side"],
            "confidence": p["confidence"],
            "probability_up": 0.55 if p["side"] == "BUY" else 0.45,
            "entry": p["entry"],
            "stop_loss": p["stop_loss"],
            "take_profit": p["take_profit_2"],
            "rr": p["rr2"],
            "timeframe": snapshot.get("timeframe", "15m"),
            "reasoning": "Rule engine pending: " + "; ".join(p["reasons"][:3]),
            "warnings": ["AI API хэрэглээгүй (fallback). PENDING — limit/alert."],
            "market_or_pending": "PENDING",
        }
    return {
        "decision": "HOLD",
        "confidence": 35,
        "probability_up": 0.5,
        "entry": snapshot.get("price", 0),
        "stop_loss": 0, "take_profit": 0, "rr": 0,
        "timeframe": snapshot.get("timeframe", "15m"),
        "reasoning": "Тодорхой setup байхгүй — confluence хүлээх.",
        "warnings": ["No high-quality setup."],
        "market_or_pending": "HOLD",
    }

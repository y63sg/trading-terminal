"""
app.py — Claude Trading Terminal (Streamlit)
Author: Murun
Strategy: Day Trading + Trend Following + SMC.
Run: streamlit run app.py
"""

from __future__ import annotations
import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# --- optional libs ---------------------------------------------
try:
    import yfinance as yf
    YF = True
except Exception:
    YF = False

try:
    import plotly.graph_objects as go
    PLOTLY = True
except Exception:
    PLOTLY = False

try:
    from streamlit_lightweight_charts import renderLightweightCharts
    LWC = True
except Exception:
    LWC = False

# --- local modules ---------------------------------------------
import strategy as S
from news_analyzer import (load_news_history, calculate_news_probability,
                           institutional_move_advice, summarize_recent)
from signal_engine import generate_all_signals
from ai_engine import get_ai_decision

# ===============================================================
# 1. PAGE & THEME
# ===============================================================
st.set_page_config(
    page_title="Claude Trading Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#00FFA3"   # Mint Green
DANGER = "#FF4B4B"    # Red
ACCENT = "#D97757"    # Claude Terracotta
BG = "#0E1117"
PANEL = "#161B22"
BORDER = "#30363D"
TEXT = "#E6EDF3"
MUTED = "#8B949E"

st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(180deg, {BG} 0%, #0A0D12 100%);
    color: {TEXT};
}}
section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
}}
h1, h2, h3 {{ color: {PRIMARY}; letter-spacing: 0.4px; }}
.block-container {{ padding-top: 1.2rem; }}
.metric-card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}}
.metric-label {{ font-size: 12px; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.6px; }}
.metric-value {{ font-size: 24px; color: {TEXT}; font-weight: 700; }}
.signal-buy {{
    background: linear-gradient(90deg, {PRIMARY}22, {PRIMARY}55);
    border-left: 5px solid {PRIMARY};
    color: {PRIMARY};
    padding: 14px 18px; border-radius: 8px; font-weight: 700;
    font-size: 18px; margin-bottom: 10px;
}}
.signal-sell {{
    background: linear-gradient(90deg, {DANGER}22, {DANGER}55);
    border-left: 5px solid {DANGER};
    color: {DANGER};
    padding: 14px 18px; border-radius: 8px; font-weight: 700;
    font-size: 18px; margin-bottom: 10px;
}}
.signal-pending {{
    background: linear-gradient(90deg, {ACCENT}22, {ACCENT}44);
    border-left: 5px solid {ACCENT};
    color: {ACCENT};
    padding: 12px 16px; border-radius: 8px; font-weight: 600;
    font-size: 15px; margin-bottom: 8px;
}}
.signal-hold {{
    background: {PANEL};
    border-left: 5px solid {MUTED};
    color: {MUTED};
    padding: 14px 18px; border-radius: 8px; font-weight: 600;
}}
.gauge-bar {{
    height: 14px; border-radius: 7px;
    background: linear-gradient(90deg, {DANGER} 0%, #444 50%, {PRIMARY} 100%);
    position: relative;
}}
.gauge-marker {{
    position: absolute; top: -4px; width: 4px; height: 22px;
    background: {TEXT}; border-radius: 2px;
}}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
    color: {PRIMARY} !important;
    border-bottom: 2px solid {PRIMARY} !important;
}}
table {{ color: {TEXT} !important; }}
.small {{ font-size: 12px; color: {MUTED}; }}
</style>
""", unsafe_allow_html=True)


# ===============================================================
# 2. DATA LAYER
# ===============================================================
SYMBOL_MAP = {
    "XAUUSD (Gold)": "GC=F",
    "EURUSD":        "EURUSD=X",
    "GBPUSD":        "GBPUSD=X",
    "USDJPY":        "USDJPY=X",
    "BTCUSD":        "BTC-USD",
    "ETHUSD":        "ETH-USD",
    "NAS100":        "^NDX",
    "SPX500":        "^GSPC",
}

INTERVAL_MAP = {
    "5m":  ("5m",  "5d"),
    "15m": ("15m", "10d"),
    "1h":  ("60m", "1mo"),
    "4h":  ("60m", "3mo"),    # yfinance-д 4h байхгүй → 60m авч resample
    "1d":  ("1d",  "1y"),
}

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlc(yf_ticker: str, interval: str, period: str) -> pd.DataFrame:
    if not YF:
        return pd.DataFrame()
    try:
        df = yf.download(yf_ticker, interval=interval, period=period,
                         progress=False, auto_adjust=True)
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={
            df.reset_index().columns[0]: "Datetime"
        })
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df[["Datetime", "Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        st.warning(f"Үнийн өгөгдөл татахад алдаа гарлаа: {e}")
        return pd.DataFrame()


def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.set_index("Datetime")
    o = d["Open"].resample("4h").first()
    h = d["High"].resample("4h").max()
    l = d["Low"].resample("4h").min()
    c = d["Close"].resample("4h").last()
    v = d["Volume"].resample("4h").sum()
    out = pd.concat([o, h, l, c, v], axis=1).dropna().reset_index()
    out.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    return out


# ===============================================================
# 3. SIDEBAR — Account & Strategy config
# ===============================================================
st.sidebar.markdown(f"## 🟢 <span style='color:{PRIMARY}'>Trading Terminal</span>",
                    unsafe_allow_html=True)
st.sidebar.caption("Murun's Strategy · Day + Trend + SMC")

with st.sidebar.expander("💼 Дансны удирдлага", expanded=True):
    account_balance = st.number_input("Дансны үлдэгдэл ($)", 100.0, 10_000_000.0,
                                       10_000.0, step=100.0)
    risk_pct = st.slider("Эрсдэл/арилжаа (%)", 0.25, 3.0, 1.0, 0.25)
    rr_target = st.select_slider("RR зорилго", options=[2.0, 2.5, 3.0, 3.5, 4.0],
                                  value=3.0)

with st.sidebar.expander("📈 Зах зээл / Хугацаа", expanded=True):
    symbol_label = st.selectbox("Symbol", list(SYMBOL_MAP.keys()), index=0)
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "1h", "4h", "1d"], index=1)
    auto_refresh = st.checkbox("60 сек автомат шинэчлэл", value=False)

with st.sidebar.expander("🤖 AI Engine", expanded=False):
    api_key = st.text_input("Anthropic API Key",
                            value=os.getenv("ANTHROPIC_API_KEY", ""),
                            type="password",
                            help="Хоосон үед rule-engine fallback ажиллана.")
    model = st.selectbox("Model", ["claude-sonnet-4-6", "claude-opus-4-6",
                                    "claude-haiku-4-5-20251001"], index=0)

with st.sidebar.expander("📰 News тохиргоо", expanded=False):
    news_event = st.selectbox("Сүүлийн мэдээний төрөл",
                              ["None", "NFP", "CPI", "Unemployment"], index=0)
    actual_val = st.number_input("Actual", value=0.0, step=0.1, format="%.2f")
    forecast_val = st.number_input("Forecast", value=0.0, step=0.1, format="%.2f")

if auto_refresh:
    st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)


# ===============================================================
# 4. LOAD DATA
# ===============================================================
yf_ticker = SYMBOL_MAP[symbol_label]
yf_interval, yf_period = INTERVAL_MAP[timeframe]
raw = fetch_ohlc(yf_ticker, yf_interval, yf_period)
if timeframe == "4h" and not raw.empty:
    raw = resample_to_4h(raw)

if raw.empty:
    st.error("Үнийн өгөгдөл татаж чадсангүй. Symbol/timeframe-ээ шалгана уу, эсвэл "
             "интернет холболт байгаа эсэхийг нягтална уу.")
    st.stop()

df = S.add_indicators(raw).dropna().reset_index(drop=True)
last = df.iloc[-1]


# ===============================================================
# 5. HEADER METRICS
# ===============================================================
st.markdown(f"# 📊 Claude Trading Terminal — `{symbol_label}` · {timeframe}")
st.caption(f"Сүүлчийн шинэчлэлт: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ·  "
           f"Bars: {len(df)}")

c1, c2, c3, c4, c5 = st.columns(5)
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else float(last["Close"])
chg = float(last["Close"]) - prev_close
chg_pct = chg / prev_close * 100 if prev_close else 0.0
chg_color = PRIMARY if chg >= 0 else DANGER

c1.markdown(f"<div class='metric-card'><div class='metric-label'>Price</div>"
            f"<div class='metric-value'>{last['Close']:.2f}</div>"
            f"<div style='color:{chg_color};font-weight:600'>{chg:+.2f} ({chg_pct:+.2f}%)</div>"
            f"</div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-card'><div class='metric-label'>20 EMA</div>"
            f"<div class='metric-value'>{last['EMA20']:.2f}</div></div>",
            unsafe_allow_html=True)
c3.markdown(f"<div class='metric-card'><div class='metric-label'>RSI(14)</div>"
            f"<div class='metric-value'>{last['RSI']:.1f}</div></div>",
            unsafe_allow_html=True)
c4.markdown(f"<div class='metric-card'><div class='metric-label'>ATR(14)</div>"
            f"<div class='metric-value'>{last['ATR']:.2f}</div></div>",
            unsafe_allow_html=True)
structure = S.market_structure(df)
struct_color = PRIMARY if structure == "BULLISH" else DANGER if structure == "BEARISH" else ACCENT
c5.markdown(f"<div class='metric-card'><div class='metric-label'>Structure</div>"
            f"<div class='metric-value' style='color:{struct_color}'>{structure}</div></div>",
            unsafe_allow_html=True)


# ===============================================================
# 6. CHART
# ===============================================================
def plotly_chart(df: pd.DataFrame, fib: Dict[str, float],
                 sr: Dict[str, List[float]]):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Datetime"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color=PRIMARY, decreasing_line_color=DANGER,
        increasing_fillcolor=PRIMARY, decreasing_fillcolor=DANGER,
    ))
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA20"], name="EMA 20",
                             line=dict(color=PRIMARY, width=1.4)))
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA50"], name="EMA 50",
                             line=dict(color=ACCENT, width=1.2, dash="dot")))

    # Golden zone
    fig.add_hrect(y0=fib["0.5"], y1=fib["0.618"],
                  fillcolor=PRIMARY, opacity=0.07, line_width=0,
                  annotation_text="Golden Zone (0.5–0.618)",
                  annotation_position="top left",
                  annotation_font_color=PRIMARY)

    for lvl, val in fib.items():
        if lvl in ("0.0", "1.0"):
            continue
        fig.add_hline(y=val, line=dict(color="#5e6770", width=0.6, dash="dot"),
                      annotation_text=f"Fib {lvl}", annotation_font_color=MUTED,
                      annotation_position="right")

    for s in sr.get("support", []):
        fig.add_hline(y=s, line=dict(color=PRIMARY, width=1, dash="dash"))
    for r in sr.get("resistance", []):
        fig.add_hline(y=r, line=dict(color=DANGER, width=1, dash="dash"))

    fig.update_layout(
        height=560,
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        font=dict(color=TEXT),
    )
    return fig


def lwc_chart(df: pd.DataFrame, fib: Dict[str, float]):
    """TradingView Lightweight Charts (хэрвээ суулгасан бол)."""
    candles = [{
        "time": int(pd.Timestamp(t).timestamp()),
        "open": float(o), "high": float(h),
        "low": float(l), "close": float(c),
    } for t, o, h, l, c in zip(df["Datetime"], df["Open"], df["High"],
                                df["Low"], df["Close"])]
    ema = [{"time": int(pd.Timestamp(t).timestamp()), "value": float(v)}
           for t, v in zip(df["Datetime"], df["EMA20"]) if not np.isnan(v)]

    chart = [{
        "chart": {
            "height": 540,
            "layout": {"background": {"type": "solid", "color": BG},
                       "textColor": TEXT},
            "grid": {"vertLines": {"color": "#1c2128"},
                     "horzLines": {"color": "#1c2128"}},
            "rightPriceScale": {"borderColor": BORDER},
            "timeScale": {"borderColor": BORDER, "timeVisible": True},
        },
        "series": [
            {"type": "Candlestick", "data": candles,
             "options": {"upColor": PRIMARY, "downColor": DANGER,
                          "borderVisible": False,
                          "wickUpColor": PRIMARY, "wickDownColor": DANGER}},
            {"type": "Line", "data": ema,
             "options": {"color": PRIMARY, "lineWidth": 2}},
        ],
    }]
    renderLightweightCharts(chart, key="main_chart")


swing_high = float(df["High"].tail(60).max())
swing_low = float(df["Low"].tail(60).min())
fib = S.fibonacci_levels(swing_high, swing_low)
sr = S.detect_support_resistance(df)
tl = S.trendline_analysis(df)
grab = S.detect_liquidity_grab(df)

st.markdown("### 🕯️ Лааны график · Fib · S/R")
chart_tabs = st.tabs(["Plotly", "Lightweight Charts"])
with chart_tabs[0]:
    if PLOTLY:
        st.plotly_chart(plotly_chart(df, fib, sr), use_container_width=True)
    else:
        st.info("plotly суулгана уу: `pip install plotly`")
with chart_tabs[1]:
    if LWC:
        lwc_chart(df, fib)
    else:
        st.info("`pip install streamlit-lightweight-charts` хийж дахин ажиллуулна уу. "
                "Үгүй бол Plotly tab-ыг ашиглаарай.")


# ===============================================================
# 7. NEWS ANALYSIS
# ===============================================================
news_df = load_news_history()
news_result, inst_advice, news_bias = None, None, None

if news_event != "None":
    deviation = float(actual_val - forecast_val)
    news_result = calculate_news_probability(news_df, news_event, deviation)
    inst_advice = institutional_move_advice(news_df, news_event, deviation)
    news_bias = news_result["direction"]


# ===============================================================
# 8. SIGNAL GENERATION
# ===============================================================
signals = generate_all_signals(df, symbol=symbol_label, news_bias=news_bias,
                                rr_target=rr_target)
snapshot = S.snapshot_for_ai(df, symbol_label, timeframe)


# ===============================================================
# 9. AI DECISION
# ===============================================================
with st.spinner("🤖 Claude дүгнэлт хийж байна..."):
    ai = get_ai_decision(snapshot, news_result or {}, signals,
                         api_key=api_key or None, model=model)


# ===============================================================
# 10. SIGNAL PANEL
# ===============================================================
st.markdown("### 🎯 Signal Generation")

sig_l, sig_r = st.columns([3, 2])

# --- Market Execution ---
with sig_l:
    st.markdown("#### ⚡ Market Execution (одоо орох)")
    market = signals.get("market")
    if market:
        cls = "signal-buy" if market["side"] == "BUY" else "signal-sell"
        st.markdown(
            f"<div class='{cls}'>{market['label']}<br>"
            f"<span style='font-size:13px;font-weight:500'>"
            f"Entry: {market['entry']} · SL: {market['stop_loss']} · "
            f"TP1: {market['take_profit_1']} (1:2) · "
            f"TP2: {market['take_profit_2']} (1:{market['rr2']:.1f}) · "
            f"Confidence: {market['confidence']}%</span></div>",
            unsafe_allow_html=True)
        with st.expander("Шалтгаан (confluence)"):
            for r in market["reasons"]:
                st.write(f"• {r}")
        size = S.position_size(account_balance, risk_pct,
                               market["entry"], market["stop_loss"])
        st.caption(f"Lot size (1% risk): **{size['lot_size']}** · "
                   f"Эрсдэл: ${size['risk_usd']} · Stop: {size['stop_pips']} pips")
    else:
        st.markdown("<div class='signal-hold'>Одоохондоо confluence бүрдээгүй. "
                    "Pending order-уудаа хар.</div>", unsafe_allow_html=True)

    st.markdown("#### 📅 Pending Orders (төлөвлөх)")
    if signals.get("pendings"):
        for p in signals["pendings"]:
            st.markdown(
                f"<div class='signal-pending'>{p['label']}<br>"
                f"<span style='font-size:13px;font-weight:500;color:{TEXT}'>"
                f"Entry: {p['entry']} · SL: {p['stop_loss']} · "
                f"TP1: {p['take_profit_1']} · TP2: {p['take_profit_2']} · "
                f"RR: 1:{p['rr2']:.1f} · Confidence: {p['confidence']}%</span></div>",
                unsafe_allow_html=True)
            with st.expander(f"Шалтгаан — {p['kind']} {p['side']}"):
                for r in p["reasons"]:
                    st.write(f"• {r}")
    else:
        st.info("Идэвхтэй pending setup байхгүй.")

# --- AI / Probability gauge ---
with sig_r:
    st.markdown("#### 🧠 Market Intelligence")
    decision = ai.get("decision", "HOLD")
    color = PRIMARY if decision == "BUY" else DANGER if decision == "SELL" else MUTED
    st.markdown(f"<div class='metric-card' style='border-left:5px solid {color}'>"
                f"<div class='metric-label'>Final Decision</div>"
                f"<div class='metric-value' style='color:{color}'>{decision}</div>"
                f"<div class='small'>Confidence: {ai.get('confidence', 0)}% · "
                f"Mode: {ai.get('market_or_pending', '—')}</div>"
                f"</div>", unsafe_allow_html=True)

    # Probability gauge
    prob = float(ai.get("probability_up", 0.5))
    pos_pct = max(2, min(98, prob * 100))
    st.markdown("<div class='small'>Probability Gauge (DOWN ← → UP)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='gauge-bar'><div class='gauge-marker' "
                f"style='left:{pos_pct}%'></div></div>", unsafe_allow_html=True)
    st.caption(f"P(up) = {prob:.0%}")

    if ai.get("entry"):
        st.markdown(f"**Entry:** `{ai['entry']}`  ·  **SL:** `{ai.get('stop_loss')}`  "
                    f"·  **TP:** `{ai.get('take_profit')}`  ·  **RR:** `1:{ai.get('rr', 0)}`")

    if ai.get("reasoning"):
        st.markdown("**Reasoning:**")
        st.write(ai["reasoning"])

    if ai.get("warnings"):
        with st.expander("⚠️ Анхааруулга"):
            for w in ai["warnings"]:
                st.write(f"• {w}")


# ===============================================================
# 11. NEWS PANEL
# ===============================================================
st.markdown("---")
st.markdown("### 📰 News Intelligence — сүүлийн 10 сарын макро мэдээ")

n_left, n_right = st.columns([2, 3])

with n_left:
    if news_result:
        prob = news_result["probability"]
        bg = PRIMARY if news_result["direction"] == "UP" else DANGER if news_result["direction"] == "DOWN" else MUTED
        st.markdown(f"<div class='metric-card' style='border-left:5px solid {bg}'>"
                    f"<div class='metric-label'>{news_event} Deviation Probability</div>"
                    f"<div class='metric-value' style='color:{bg}'>{news_result['direction']}  ·  {prob:.0%}</div>"
                    f"<div class='small'>Sample: {news_result['sample_size']} · "
                    f"Avg 15m: {news_result['avg_pips_15m']:+.0f} pips · "
                    f"Avg 1h: {news_result['avg_pips_1h']:+.0f} pips</div>"
                    f"</div>", unsafe_allow_html=True)
        st.write(news_result["comment"])
        if news_result["volatility_breakout_recommended"]:
            st.markdown(f"<div class='signal-pending'>📢 Volatility Breakout зөвлөмжтэй — "
                        f"deviation түүхэн дунджаас илүү. Эхний 5 минутыг хүлээгээд "
                        f"institutional move-руу ор.</div>", unsafe_allow_html=True)
        if inst_advice:
            st.markdown(f"<div class='small'>🕒 <b>5–15 мин дараах хөдөлгөөн:</b> "
                        f"{inst_advice['advice']}  "
                        f"(follow-through {inst_advice['follow_through_rate']:.0%})</div>",
                        unsafe_allow_html=True)
    else:
        st.info("Sidebar-аас сүүлийн мэдээний event-ийг сонгож, Actual/Forecast-оо оруулна уу.")

with n_right:
    st.markdown("**Сүүлийн 10 release-н түүх**")
    recent = summarize_recent(news_df, limit=12)
    if not recent.empty:
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info("data/news_history.json файл алга.")


# ===============================================================
# 12. RAW SNAPSHOT (debug)
# ===============================================================
with st.expander("🔧 Snapshot (AI-д өгсөн өгөгдөл)"):
    st.json({"snapshot": snapshot, "news": news_result, "signals": signals,
             "ai_decision": ai}, expanded=False)

st.markdown("<br><div class='small'>"
            "⚠️ Энэ систем нь хувийн арилжааны туслах зорилготой. "
            "Санхүүгийн зөвлөгөө биш — өөрийн эрсдэлийг өөрөө хариуцна. "
            "© Murun · Claude Trading Terminal"
            "</div>", unsafe_allow_html=True)

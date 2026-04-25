"""
Claude Trading Terminal — Ultra Premium UI v3.0
================================================
Institutional-grade financial terminal for XAUUSD day-trading.
Modern dark glass-morphism with Bloomberg/TradingView-tier polish.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from ai_engine import get_ai_decision
from news_analyzer import (
    calculate_news_probability,
    institutional_move_advice,
    load_news_history,
    summarize_recent,
)
from signal_engine import generate_all_signals
from strategy import (
    detect_liquidity_grab,
    detect_support_resistance,
    detect_swings,
    ema,
    fibonacci_levels,
    market_structure,
    position_size,
    rsi,
    atr,
    snapshot_for_ai,
    trendline_analysis,
)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Claude Trading Terminal",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS — Ultra Premium Dark Theme
# ════════════════════════════════════════════════════════════════════════════
T = {
    "bg_0":         "#04060A",
    "bg_1":         "#080B12",
    "bg_2":         "#0D1119",
    "surface":      "rgba(255,255,255,0.025)",
    "surface_2":    "rgba(255,255,255,0.045)",
    "surface_3":    "rgba(255,255,255,0.07)",
    "border":       "rgba(255,255,255,0.06)",
    "border_2":     "rgba(255,255,255,0.10)",
    "border_3":     "rgba(255,255,255,0.16)",
    "text":         "#F0F2F5",
    "text_2":       "#B6BCC8",
    "text_3":       "#7A8190",
    "text_4":       "#4A5160",
    "primary":      "#00E5A0",
    "primary_2":    "#00C088",
    "primary_glow": "rgba(0,229,160,0.35)",
    "danger":       "#FF4D6A",
    "danger_2":     "#E63E5C",
    "danger_glow":  "rgba(255,77,106,0.35)",
    "warning":      "#FFB547",
    "warning_glow": "rgba(255,181,71,0.30)",
    "accent":       "#D97757",
    "accent_glow":  "rgba(217,119,87,0.35)",
    "info":         "#5B8DEF",
    "info_glow":    "rgba(91,141,239,0.30)",
    "purple":       "#A78BFA",
    "gold":         "#F5C242",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

:root {{
  --bg-0: {T['bg_0']};
  --bg-1: {T['bg_1']};
  --bg-2: {T['bg_2']};
  --surface: {T['surface']};
  --surface-2: {T['surface_2']};
  --surface-3: {T['surface_3']};
  --border: {T['border']};
  --border-2: {T['border_2']};
  --border-3: {T['border_3']};
  --text: {T['text']};
  --text-2: {T['text_2']};
  --text-3: {T['text_3']};
  --text-4: {T['text_4']};
  --primary: {T['primary']};
  --primary-2: {T['primary_2']};
  --primary-glow: {T['primary_glow']};
  --danger: {T['danger']};
  --danger-2: {T['danger_2']};
  --danger-glow: {T['danger_glow']};
  --warning: {T['warning']};
  --warning-glow: {T['warning_glow']};
  --accent: {T['accent']};
  --accent-glow: {T['accent_glow']};
  --info: {T['info']};
  --info-glow: {T['info_glow']};
  --purple: {T['purple']};
  --gold: {T['gold']};
}}

/* ═══ FOUNDATION ═══ */
html, body, [class*="css"], .stApp {{
  background: var(--bg-0) !important;
  color: var(--text) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  font-feature-settings: 'cv11', 'ss01', 'ss03';
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}}

/* Animated mesh backdrop */
.stApp::before {{
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 1400px 700px at 12% -8%,  rgba(0,229,160,0.07) 0%, transparent 55%),
    radial-gradient(ellipse  900px 500px at 88% 18%,  rgba(217,119,87,0.05) 0%, transparent 55%),
    radial-gradient(ellipse 1100px 600px at 50% 105%, rgba(91,141,239,0.06) 0%, transparent 55%),
    radial-gradient(ellipse  700px 400px at 100% 75%, rgba(167,139,250,0.04) 0%, transparent 55%);
  pointer-events: none;
  z-index: 0;
  animation: meshShift 25s ease-in-out infinite alternate;
}}

@keyframes meshShift {{
  0%   {{ transform: translate3d(0,0,0) scale(1); }}
  100% {{ transform: translate3d(2%,-2%,0) scale(1.08); }}
}}

.stApp > div {{ position: relative; z-index: 1; }}

/* Hide chrome */
#MainMenu, footer, .stDeployButton {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent !important; height: 0 !important; }}

.block-container {{
  padding: 1.25rem 2rem 4rem !important;
  max-width: 100% !important;
}}

/* ═══ TYPOGRAPHY ═══ */
h1, h2, h3, h4, h5, h6 {{
  font-family: 'Inter', sans-serif !important;
  letter-spacing: -0.022em !important;
  color: var(--text) !important;
  font-weight: 700 !important;
}}

p, span, div, label {{ color: var(--text); }}

code, pre, .mono {{
  font-family: 'JetBrains Mono', monospace !important;
  font-feature-settings: 'zero', 'ss01';
}}

/* ═══ TICKER BAR (top scrolling status) ═══ */
.ticker-bar {{
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.55rem 1.25rem;
  background: linear-gradient(90deg, rgba(0,229,160,0.04), rgba(255,255,255,0.02), rgba(217,119,87,0.04));
  border: 1px solid var(--border);
  border-radius: 12px;
  margin-bottom: 1rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.74rem;
  font-weight: 500;
  color: var(--text-2);
  overflow: hidden;
  position: relative;
}}

.ticker-bar::after {{
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.04) 50%, transparent 100%);
  animation: shimmer 4s ease-in-out infinite;
  pointer-events: none;
}}

@keyframes shimmer {{
  0%   {{ transform: translateX(-100%); }}
  100% {{ transform: translateX(100%); }}
}}

.ticker-item {{ display: flex; align-items: center; gap: 6px; white-space: nowrap; }}
.ticker-divider {{ color: var(--text-4); }}
.ticker-label {{ color: var(--text-4); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; }}
.ticker-value.up   {{ color: var(--primary); }}
.ticker-value.down {{ color: var(--danger); }}

/* ═══ HERO HEADER ═══ */
.hero {{
  position: relative;
  padding: 1.85rem 2rem;
  margin-bottom: 1.25rem;
  background:
    linear-gradient(135deg, rgba(255,255,255,0.045) 0%, rgba(255,255,255,0.01) 100%);
  border: 1px solid var(--border-2);
  border-radius: 22px;
  backdrop-filter: blur(28px) saturate(180%);
  overflow: hidden;
}}

.hero::before {{
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 700px 240px at  0% 0%,   rgba(0,229,160,0.14)  0%, transparent 65%),
    radial-gradient(ellipse 500px 250px at 100% 100%, rgba(217,119,87,0.10) 0%, transparent 65%);
  pointer-events: none;
}}

.hero::after {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0,229,160,0.5), rgba(217,119,87,0.4), transparent);
}}

.hero-content {{
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 1.25rem;
}}

.hero-title {{ display: flex; align-items: center; gap: 1.1rem; }}

.hero-logo {{
  width: 56px;
  height: 56px;
  border-radius: 16px;
  background:
    radial-gradient(circle at 30% 30%, #00FFB8 0%, var(--primary) 50%, #00B886 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 28px;
  font-weight: 900;
  color: #001a14;
  box-shadow:
    0 14px 40px var(--primary-glow),
    0 0 0 1px rgba(0,229,160,0.25),
    inset 0 1px 0 rgba(255,255,255,0.40);
  position: relative;
}}

.hero-logo::after {{
  content: '';
  position: absolute;
  inset: -2px;
  border-radius: 18px;
  background: conic-gradient(from 0deg, transparent, var(--primary), transparent);
  opacity: 0.6;
  z-index: -1;
  animation: rotate 4s linear infinite;
}}

@keyframes rotate {{
  to {{ transform: rotate(360deg); }}
}}

.hero-text h1 {{
  margin: 0 !important;
  font-size: 1.7rem !important;
  font-weight: 800 !important;
  background: linear-gradient(135deg, #FFFFFF 0%, #B8C0CC 75%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -0.03em !important;
  line-height: 1.1 !important;
}}

.hero-text .subtitle {{
  font-size: 0.75rem;
  color: var(--text-4);
  margin-top: 4px;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
}}

.hero-tags {{ display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }}

.tag {{
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 7px 13px;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.03em;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  color: var(--text-2);
  white-space: nowrap;
  transition: all 0.2s;
}}

.tag:hover {{ background: rgba(255,255,255,0.07); border-color: var(--border-3); }}

.tag-live {{
  background: rgba(0,229,160,0.10);
  border-color: rgba(0,229,160,0.30);
  color: var(--primary);
  font-weight: 700;
}}

.tag-live::before {{
  content: '';
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--primary);
  box-shadow: 0 0 0 0 var(--primary-glow);
  animation: pulse 1.8s infinite;
}}

@keyframes pulse {{
  0%   {{ box-shadow: 0 0 0 0 var(--primary-glow); }}
  70%  {{ box-shadow: 0 0 0 10px rgba(0,229,160,0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(0,229,160,0); }}
}}

.tag-symbol {{
  background: linear-gradient(135deg, rgba(245,194,66,0.12), rgba(245,194,66,0.04));
  border-color: rgba(245,194,66,0.30);
  color: var(--gold);
  font-weight: 700;
}}

.tag-tf {{
  background: rgba(91,141,239,0.10);
  border-color: rgba(91,141,239,0.25);
  color: var(--info);
}}

/* ═══ METRIC CARDS — Premium Grid ═══ */
.metric-row {{
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.85rem;
  margin-bottom: 1.25rem;
}}

.metric {{
  position: relative;
  padding: 1.15rem 1.3rem;
  background:
    linear-gradient(135deg, rgba(255,255,255,0.035) 0%, rgba(255,255,255,0.012) 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  backdrop-filter: blur(16px);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  overflow: hidden;
}}

.metric::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
}}

.metric::after {{
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 16px;
  padding: 1px;
  background: linear-gradient(135deg, transparent 30%, rgba(255,255,255,0.04) 50%, transparent 70%);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.3s;
}}

.metric:hover {{
  border-color: var(--border-3);
  transform: translateY(-3px);
  box-shadow: 0 16px 32px rgba(0,0,0,0.35), 0 0 0 1px rgba(255,255,255,0.04);
}}

.metric:hover::after {{ opacity: 1; }}

.metric-label {{
  font-size: 0.66rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--text-4);
  font-weight: 700;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: 'JetBrains Mono', monospace;
}}

.metric-value {{
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 1.65rem;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.025em;
  line-height: 1.05;
}}

.metric-delta {{
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem;
  font-weight: 700;
  margin-top: 8px;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}}

.metric-delta.up   {{ color: var(--primary); }}
.metric-delta.down {{ color: var(--danger); }}
.metric-delta.flat {{ color: var(--text-4); }}

.metric-pill {{
  display: inline-block;
  padding: 4px 10px;
  border-radius: 8px;
  font-size: 0.7rem;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace !important;
  letter-spacing: 0.06em;
  margin-top: 6px;
}}

.pill-bull {{ background: rgba(0,229,160,0.12);  color: var(--primary); border: 1px solid rgba(0,229,160,0.28); }}
.pill-bear {{ background: rgba(255,77,106,0.12); color: var(--danger);  border: 1px solid rgba(255,77,106,0.28); }}
.pill-flat {{ background: rgba(255,255,255,0.05); color: var(--text-2); border: 1px solid var(--border-2); }}
.pill-warn {{ background: rgba(255,181,71,0.12); color: var(--warning); border: 1px solid rgba(255,181,71,0.30); }}

/* ═══ PANELS ═══ */
.panel {{
  position: relative;
  background:
    linear-gradient(135deg, rgba(255,255,255,0.030) 0%, rgba(255,255,255,0.008) 100%);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1.4rem;
  backdrop-filter: blur(16px);
  margin-bottom: 1rem;
  transition: border-color 0.2s ease;
  overflow: hidden;
}}

.panel:hover {{ border-color: var(--border-2); }}

.panel-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.1rem;
  padding-bottom: 0.85rem;
  border-bottom: 1px solid var(--border);
}}

.panel-title {{
  font-size: 0.92rem;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.01em;
  display: flex;
  align-items: center;
  gap: 10px;
}}

.panel-icon {{
  width: 30px;
  height: 30px;
  border-radius: 9px;
  background: rgba(0,229,160,0.10);
  border: 1px solid rgba(0,229,160,0.22);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  color: var(--primary);
  font-weight: 700;
}}

/* ═══ SIGNAL CARDS — Animated Gradient Borders ═══ */
.signal-card {{
  position: relative;
  padding: 1.35rem;
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.012) 100%);
  border: 1px solid var(--border-2);
  margin-bottom: 0.85rem;
  overflow: hidden;
  transition: all 0.3s ease;
}}

.signal-card::before {{
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 4px;
  height: 100%;
  background: var(--text-4);
  border-radius: 4px 0 0 4px;
}}

.signal-card.buy   {{ border-color: rgba(0,229,160,0.32);   background: linear-gradient(135deg, rgba(0,229,160,0.10) 0%, rgba(0,229,160,0.025) 100%); }}
.signal-card.buy::before  {{ background: linear-gradient(180deg, var(--primary), var(--primary-2)); box-shadow: 0 0 16px var(--primary-glow); }}

.signal-card.sell  {{ border-color: rgba(255,77,106,0.32);  background: linear-gradient(135deg, rgba(255,77,106,0.10) 0%, rgba(255,77,106,0.025) 100%); }}
.signal-card.sell::before {{ background: linear-gradient(180deg, var(--danger), var(--danger-2)); box-shadow: 0 0 16px var(--danger-glow); }}

.signal-card.alert {{ border-color: rgba(255,181,71,0.32);  background: linear-gradient(135deg, rgba(255,181,71,0.10) 0%, rgba(255,181,71,0.025) 100%); }}
.signal-card.alert::before {{ background: var(--warning); box-shadow: 0 0 16px var(--warning-glow); }}

.signal-card.info {{ border-color: rgba(91,141,239,0.32); background: linear-gradient(135deg, rgba(91,141,239,0.08) 0%, rgba(91,141,239,0.02) 100%); }}
.signal-card.info::before {{ background: var(--info); box-shadow: 0 0 16px var(--info-glow); }}

.signal-card:hover {{
  transform: translateX(4px);
  border-color: rgba(255,255,255,0.20);
  box-shadow: 0 12px 28px rgba(0,0,0,0.35);
}}

.signal-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.95rem;
  flex-wrap: wrap;
  gap: 0.5rem;
}}

.signal-badge {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 800;
  padding: 6px 11px;
  border-radius: 7px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}

.signal-badge.buy   {{ background: var(--primary); color: #001a14; box-shadow: 0 4px 14px var(--primary-glow); }}
.signal-badge.sell  {{ background: var(--danger);  color: #ffffff; box-shadow: 0 4px 14px var(--danger-glow); }}
.signal-badge.alert {{ background: var(--warning); color: #1a1000; }}
.signal-badge.limit {{ background: rgba(91,141,239,0.18); color: var(--info); border: 1px solid rgba(91,141,239,0.40); }}
.signal-badge.standby {{ background: rgba(255,255,255,0.05); color: var(--text-3); border: 1px solid var(--border-2); }}

.signal-title {{
  font-size: 0.9rem;
  font-weight: 700;
  color: var(--text);
}}

.signal-subtitle {{
  font-size: 0.75rem;
  color: var(--text-3);
  font-weight: 500;
  margin-top: 2px;
}}

.signal-grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.7rem;
  margin-top: 0.85rem;
}}

.signal-stat {{
  background: rgba(0,0,0,0.25);
  padding: 0.7rem 0.9rem;
  border-radius: 11px;
  border: 1px solid var(--border);
  transition: border-color 0.2s;
}}

.signal-stat:hover {{ border-color: var(--border-2); }}

.signal-stat-label {{
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--text-4);
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
}}

.signal-stat-value {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.96rem;
  font-weight: 800;
  color: var(--text);
  margin-top: 5px;
  letter-spacing: -0.015em;
}}

.signal-rationale {{
  margin-top: 0.95rem;
  padding-top: 0.95rem;
  border-top: 1px dashed var(--border-2);
  font-size: 0.83rem;
  color: var(--text-2);
  line-height: 1.55;
}}

/* ═══ AI VERDICT — Hero Card ═══ */
.ai-verdict {{
  position: relative;
  padding: 1.7rem;
  border-radius: 22px;
  background:
    radial-gradient(ellipse 600px 240px at 50% 0%, rgba(217,119,87,0.14) 0%, transparent 65%),
    linear-gradient(135deg, rgba(255,255,255,0.045) 0%, rgba(255,255,255,0.012) 100%);
  border: 1px solid rgba(217,119,87,0.22);
  margin-bottom: 1rem;
  overflow: hidden;
  backdrop-filter: blur(20px);
}}

.ai-verdict::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent), var(--purple), var(--accent), transparent);
  background-size: 200% 100%;
  animation: gradientShift 4s ease-in-out infinite;
}}

@keyframes gradientShift {{
  0%, 100% {{ background-position: 0% 50%; }}
  50%      {{ background-position: 100% 50%; }}
}}

.ai-header {{
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 1.25rem;
}}

.ai-orb {{
  position: relative;
  width: 44px;
  height: 44px;
  border-radius: 14px;
  background: linear-gradient(135deg, var(--accent) 0%, #B85A3F 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  font-weight: 800;
  box-shadow:
    0 10px 26px rgba(217,119,87,0.35),
    inset 0 1px 0 rgba(255,255,255,0.30);
  color: #fff;
}}

.ai-orb::after {{
  content: '';
  position: absolute;
  inset: -3px;
  border-radius: 17px;
  background: conic-gradient(from 0deg, transparent, var(--accent), var(--purple), transparent);
  opacity: 0.5;
  z-index: -1;
  animation: rotate 6s linear infinite;
}}

.ai-meta {{ display: flex; flex-direction: column; gap: 2px; }}
.ai-name  {{ font-size: 1rem; font-weight: 800; color: var(--text); letter-spacing: -0.01em; }}
.ai-model {{ font-size: 0.7rem; color: var(--text-4); text-transform: uppercase; letter-spacing: 0.10em; font-family: 'JetBrains Mono', monospace; font-weight: 600; }}

.ai-verdict-decision {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 2.4rem;
  font-weight: 900;
  letter-spacing: -0.04em;
  margin: 0.85rem 0;
  line-height: 1;
}}

.ai-verdict-decision.buy  {{ color: var(--primary); text-shadow: 0 0 32px var(--primary-glow); }}
.ai-verdict-decision.sell {{ color: var(--danger);  text-shadow: 0 0 32px var(--danger-glow); }}
.ai-verdict-decision.wait {{ color: var(--text-2); }}

.ai-confidence-bar {{
  margin-top: 1rem;
  padding: 0.85rem;
  background: rgba(0,0,0,0.25);
  border: 1px solid var(--border);
  border-radius: 12px;
}}

.ai-conf-row {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.74rem;
  font-weight: 600;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}}

.ai-conf-track {{
  height: 8px;
  background: rgba(255,255,255,0.06);
  border-radius: 999px;
  overflow: hidden;
  position: relative;
}}

.ai-conf-fill {{
  height: 100%;
  background: linear-gradient(90deg, var(--accent), var(--primary));
  border-radius: 999px;
  position: relative;
  box-shadow: 0 0 14px rgba(0,229,160,0.40);
}}

.ai-conf-fill::after {{
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
  animation: shimmer 2.5s ease-in-out infinite;
}}

.ai-reasoning {{
  margin-top: 1rem;
  padding: 1rem;
  background: rgba(0,0,0,0.20);
  border-left: 3px solid var(--accent);
  border-radius: 0 12px 12px 0;
  font-size: 0.85rem;
  color: var(--text-2);
  line-height: 1.6;
}}

/* ═══ CONIC GAUGE ═══ */
.gauge-conic {{
  position: relative;
  width: 200px;
  height: 200px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: center;
}}

.gauge-conic-ring {{
  position: absolute;
  inset: 0;
  border-radius: 50%;
  background:
    conic-gradient(from 180deg, var(--gauge-color) 0%, var(--gauge-color) calc(var(--gauge-pct) * 1%), rgba(255,255,255,0.05) calc(var(--gauge-pct) * 1%));
  -webkit-mask: radial-gradient(circle, transparent 62%, #000 63%);
  mask: radial-gradient(circle, transparent 62%, #000 63%);
}}

.gauge-conic-glow {{
  position: absolute;
  inset: 5%;
  border-radius: 50%;
  filter: blur(20px);
  background: var(--gauge-color);
  opacity: 0.20;
}}

.gauge-conic-inner {{
  position: relative;
  z-index: 2;
  text-align: center;
}}

.gauge-conic-value {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 2.6rem;
  font-weight: 800;
  letter-spacing: -0.04em;
  color: var(--text);
  line-height: 1;
}}

.gauge-conic-pct {{
  font-size: 1.2rem;
  color: var(--text-3);
  font-weight: 600;
}}

.gauge-conic-label {{
  font-size: 0.7rem;
  color: var(--text-4);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-weight: 700;
  margin-top: 6px;
  font-family: 'JetBrains Mono', monospace;
}}

/* ═══ SIDEBAR ═══ */
section[data-testid="stSidebar"] {{
  background: rgba(6,8,12,0.92) !important;
  border-right: 1px solid var(--border) !important;
  backdrop-filter: blur(24px);
}}

section[data-testid="stSidebar"] > div {{ padding-top: 1rem; }}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stSlider label {{
  font-size: 0.7rem !important;
  font-weight: 700 !important;
  color: var(--text-4) !important;
  text-transform: uppercase;
  letter-spacing: 0.10em;
  font-family: 'JetBrains Mono', monospace !important;
}}

.stSelectbox [data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input,
[data-baseweb="input"] {{
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  transition: all 0.2s ease;
}}

.stSelectbox [data-baseweb="select"] > div:hover,
.stTextInput input:focus,
.stNumberInput input:focus {{
  border-color: rgba(0,229,160,0.45) !important;
  box-shadow: 0 0 0 3px rgba(0,229,160,0.10) !important;
}}

.stButton > button {{
  background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
  color: #001a14 !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 800 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  padding: 0.7rem 1.3rem !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 4px 14px var(--primary-glow) !important;
}}

.stButton > button:hover {{
  transform: translateY(-2px);
  box-shadow: 0 10px 26px var(--primary-glow) !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 4px;
  gap: 4px;
}}

.stTabs [data-baseweb="tab"] {{
  background: transparent !important;
  color: var(--text-3) !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  padding: 0.5rem 1rem !important;
  border: none !important;
  transition: all 0.2s ease;
}}

.stTabs [aria-selected="true"] {{
  background: rgba(0,229,160,0.12) !important;
  color: var(--primary) !important;
}}

.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* Expander */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {{
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  color: var(--text-2) !important;
  font-size: 0.78rem !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-family: 'JetBrains Mono', monospace !important;
}}

[data-testid="stExpander"] {{ border: none !important; background: transparent !important; }}

.stCheckbox label {{ color: var(--text-2) !important; font-family: 'Inter', sans-serif !important; text-transform: none !important; letter-spacing: 0 !important; font-size: 0.85rem !important; }}

.stSlider [data-baseweb="slider"] > div > div > div {{ background: var(--primary) !important; }}

.stAlert {{
  background: var(--surface) !important;
  border: 1px solid var(--border-2) !important;
  border-radius: 12px !important;
  backdrop-filter: blur(12px);
}}

.js-plotly-plot, .plotly-graph-div {{ border-radius: 14px !important; }}

.stCodeBlock {{
  background: rgba(0,0,0,0.45) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}}

::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.10); border-radius: 10px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.20); }}

/* Section dividers */
.section-header {{
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 1.5rem 0 1rem;
}}

.section-header h2, .section-header h3 {{
  margin: 0 !important;
  font-size: 1.05rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.02em !important;
}}

.section-icon {{
  width: 30px;
  height: 30px;
  border-radius: 8px;
  background: linear-gradient(135deg, rgba(0,229,160,0.12), rgba(0,229,160,0.04));
  border: 1px solid rgba(0,229,160,0.22);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--primary);
  font-size: 14px;
  font-weight: 700;
}}

.section-line {{
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--border-2), transparent);
}}

/* News rows */
.news-row {{
  display: grid;
  grid-template-columns: 105px 1fr 95px 95px 110px;
  gap: 1rem;
  padding: 0.85rem 1rem;
  border-radius: 11px;
  background: rgba(255,255,255,0.018);
  border: 1px solid var(--border);
  margin-bottom: 0.45rem;
  align-items: center;
  transition: all 0.2s ease;
}}

.news-row:hover {{
  background: rgba(255,255,255,0.04);
  border-color: var(--border-2);
  transform: translateX(2px);
}}

.news-date  {{ font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--text-4); font-weight: 500; }}
.news-event {{ font-size: 0.85rem; font-weight: 600; color: var(--text); }}
.news-num   {{ font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 700; text-align: right; }}
.news-pip-bar {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  font-weight: 700;
  text-align: right;
  position: relative;
  padding: 4px 8px;
  border-radius: 6px;
}}

/* Chips */
.chip {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 600;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  color: var(--text-3);
  margin-right: 4px;
  font-family: 'JetBrains Mono', monospace;
}}

/* Footer */
.footer {{
  margin-top: 3.5rem;
  padding: 1.5rem 0;
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.5rem;
  font-size: 0.74rem;
  color: var(--text-4);
  font-family: 'JetBrains Mono', monospace;
}}

[data-testid="stMetricLabel"] {{ font-size: 0.7rem !important; color: var(--text-4) !important; }}
[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace !important; }}

@media (max-width: 1100px) {{
  .metric-row  {{ grid-template-columns: repeat(2, 1fr); }}
  .signal-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .news-row    {{ grid-template-columns: 1fr 1fr; }}
}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
SYMBOL_MAP = {
    "XAUUSD (Gold)": "GC=F",
    "EURUSD":        "EURUSD=X",
    "GBPUSD":        "GBPUSD=X",
    "USDJPY":        "USDJPY=X",
    "BTCUSD":        "BTC-USD",
    "ETHUSD":        "ETH-USD",
    "S&P 500":       "^GSPC",
    "NASDAQ":        "^NDX",
}

INTERVAL_MAP = {
    "5m":  ("5m",  "5d"),
    "15m": ("15m", "10d"),
    "30m": ("30m", "30d"),
    "1h":  ("60m", "60d"),
    "4h":  ("60m", "60d"),
    "1d":  ("1d",  "1y"),
}

NEWS_TYPES = ["NFP", "CPI", "Unemployment Rate", "FOMC", "Core PCE"]

# ════════════════════════════════════════════════════════════════════════════
#  DATA LAYER
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlc(yf_symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        yf_symbol, interval=interval, period=period,
        auto_adjust=False, progress=False, prepost=False,
    )
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return df


def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.resample("4h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).dropna()


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Premium Control Deck
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;padding:0.5rem 0 1.5rem;border-bottom:1px solid var(--border);margin-bottom:1.25rem;">
          <div style="width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,var(--primary),var(--primary-2));display:flex;align-items:center;justify-content:center;font-weight:900;color:#001a14;font-size:18px;box-shadow:0 8px 22px var(--primary-glow);">◆</div>
          <div>
            <div style="font-size:1rem;font-weight:800;color:var(--text);letter-spacing:-0.02em;">Trading Console</div>
            <div style="font-size:0.68rem;color:var(--text-4);text-transform:uppercase;letter-spacing:0.10em;font-family:'JetBrains Mono',monospace;font-weight:600;">v3.0 · Configuration</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("◇ MARKET · TIMEFRAME", expanded=True):
        symbol_label = st.selectbox("Symbol", list(SYMBOL_MAP.keys()), index=0)
        timeframe = st.selectbox("Timeframe", list(INTERVAL_MAP.keys()), index=1)
        auto_refresh = st.checkbox("Auto-refresh every 60s", value=False)

    with st.expander("◈ RISK MANAGEMENT", expanded=True):
        account_balance = st.number_input(
            "Account Balance (USD)", min_value=100.0, value=10000.0, step=500.0,
        )
        risk_pct = st.slider("Risk per Trade (%)", 0.25, 3.0, 1.0, 0.25)
        target_rr = st.slider("Target R:R", 1.5, 5.0, 2.5, 0.5)

    with st.expander("◉ AI ENGINE", expanded=False):
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else "",
            help="Optional. Falls back to rule engine if not provided.",
        )
        ai_model = st.selectbox(
            "Model",
            ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
            index=1,
        )

    with st.expander("◎ NEWS DEVIATION", expanded=False):
        news_type = st.selectbox("Indicator", NEWS_TYPES, index=0)
        news_actual = st.number_input("Actual",   value=0.00, step=0.05, format="%.2f")
        news_forecast = st.number_input("Forecast", value=0.00, step=0.05, format="%.2f")

    st.markdown(
        """
        <div style="margin-top:1.5rem;padding:1rem;background:linear-gradient(135deg,rgba(0,229,160,0.06),rgba(0,229,160,0.02));border:1px solid rgba(0,229,160,0.18);border-radius:12px;">
          <div style="font-size:0.66rem;color:var(--primary);text-transform:uppercase;letter-spacing:0.10em;font-weight:800;margin-bottom:6px;font-family:'JetBrains Mono',monospace;">◆ PRO TIP</div>
          <div style="font-size:0.78rem;color:var(--text-2);line-height:1.5;">
            Highest probability XAUUSD setups occur during the <b>London/NY overlap</b> (13:00–16:00 UTC).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
#  DATA FETCH
# ════════════════════════════════════════════════════════════════════════════
yf_symbol = SYMBOL_MAP[symbol_label]
yf_interval, yf_period = INTERVAL_MAP[timeframe]

with st.spinner("Loading market data..."):
    df = fetch_ohlc(yf_symbol, yf_interval, yf_period)
    if timeframe == "4h" and not df.empty:
        df = resample_to_4h(df)

if df.empty or len(df) < 50:
    st.error("Unable to load enough market data. Try another timeframe or symbol.")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
#  INDICATORS / ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
df["ema20"] = ema(df["close"], 20)
df["ema50"] = ema(df["close"], 50)
df["rsi14"] = rsi(df["close"], 14)
df["atr14"] = atr(df["high"], df["low"], df["close"], 14)

last       = df.iloc[-1]
prev       = df.iloc[-2]
price      = float(last["close"])
ema20_v    = float(last["ema20"])
ema50_v    = float(last["ema50"])
rsi_v      = float(last["rsi14"])
atr_v      = float(last["atr14"])
prev_close = float(prev["close"])
delta_abs  = price - prev_close
delta_pct  = (delta_abs / prev_close) * 100 if prev_close else 0.0

# Session range
session_high = float(df["high"].tail(48).max())
session_low  = float(df["low"].tail(48).min())

swings   = detect_swings(df)
fib      = fibonacci_levels(df)
sr       = detect_support_resistance(df)
struct   = market_structure(df, swings)
trend    = trendline_analysis(df)
liq_grab = detect_liquidity_grab(df, swings)

# News
news_history = load_news_history()
news_prob = calculate_news_probability(
    news_type, news_actual, news_forecast, news_history,
)
inst_advice = institutional_move_advice(news_prob)

# Signals
signals = generate_all_signals(
    df=df, fib=fib, sr=sr, swings=swings, trend=trend,
    structure=struct, liq_grab=liq_grab, rsi_value=rsi_v,
    ema20=ema20_v, atr_value=atr_v, news_bias=news_prob,
    account_balance=account_balance, risk_pct=risk_pct / 100,
    target_rr=target_rr,
)

# AI snapshot
snapshot = snapshot_for_ai(
    symbol=symbol_label, timeframe=timeframe, df=df, fib=fib,
    sr=sr, swings=swings, trend=trend, structure=struct,
    liq_grab=liq_grab, rsi_value=rsi_v, ema20=ema20_v,
    ema50=ema50_v, atr_value=atr_v,
)
snapshot["news_signal"]    = news_prob
snapshot["news_advice"]    = inst_advice
snapshot["pending_signals"] = [s.__dict__ for s in signals.get("pending", [])]
snapshot["market_signal"]   = signals.get("market").__dict__ if signals.get("market") else None

ai_decision = get_ai_decision(
    snapshot=snapshot, api_key=anthropic_key or None, model=ai_model,
)

# ════════════════════════════════════════════════════════════════════════════
#  TICKER BAR (top status strip)
# ════════════════════════════════════════════════════════════════════════════
now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
trend_dir = (trend.get("direction", "FLAT") or "FLAT").upper() if isinstance(trend, dict) else "FLAT"
liq_status = "DETECTED" if (liq_grab and liq_grab.get("detected")) else "NONE"

session_in_range = ((price - session_low) / (session_high - session_low) * 100) if session_high > session_low else 50

st.markdown(
    f"""
    <div class="ticker-bar">
      <div class="ticker-item"><span class="ticker-label">SESSION</span><span style="color:var(--text);font-weight:700;">LON/NY</span></div>
      <span class="ticker-divider">│</span>
      <div class="ticker-item"><span class="ticker-label">RANGE</span><span class="ticker-value">{session_low:,.2f} – {session_high:,.2f}</span></div>
      <span class="ticker-divider">│</span>
      <div class="ticker-item"><span class="ticker-label">POS IN RANGE</span><span class="ticker-value">{session_in_range:.0f}%</span></div>
      <span class="ticker-divider">│</span>
      <div class="ticker-item"><span class="ticker-label">TREND</span><span class="ticker-value">{trend_dir}</span></div>
      <span class="ticker-divider">│</span>
      <div class="ticker-item"><span class="ticker-label">LIQUIDITY</span><span class="ticker-value {('up' if liq_status=='DETECTED' else '')}">{liq_status}</span></div>
      <span class="ticker-divider">│</span>
      <div class="ticker-item"><span class="ticker-label">BARS</span><span class="ticker-value">{len(df)}</span></div>
      <span class="ticker-divider" style="margin-left:auto;">│</span>
      <div class="ticker-item"><span class="ticker-label">SYNC</span><span class="ticker-value">{now_str}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════
#  HERO HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-content">
        <div class="hero-title">
          <div class="hero-logo">◆</div>
          <div class="hero-text">
            <h1>Claude Trading Terminal</h1>
            <div class="subtitle">Institutional-grade signal engine · v3.0 Premium</div>
          </div>
        </div>
        <div class="hero-tags">
          <span class="tag tag-live">LIVE</span>
          <span class="tag tag-symbol">⊛ {symbol_label}</span>
          <span class="tag tag-tf">⏱ {timeframe.upper()}</span>
          <span class="tag">⊙ ${account_balance:,.0f}</span>
          <span class="tag">⊘ {risk_pct}% RISK</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════
#  METRIC ROW
# ════════════════════════════════════════════════════════════════════════════
delta_class = "up" if delta_abs > 0 else ("down" if delta_abs < 0 else "flat")
delta_arrow = "▲" if delta_abs > 0 else ("▼" if delta_abs < 0 else "■")

ema_state = "BULLISH" if price > ema20_v else "BEARISH"
ema_class = "pill-bull" if price > ema20_v else "pill-bear"

if rsi_v > 70:
    rsi_label, rsi_class = "OVERBOUGHT", "pill-bear"
elif rsi_v < 30:
    rsi_label, rsi_class = "OVERSOLD", "pill-bull"
else:
    rsi_label, rsi_class = "NEUTRAL", "pill-flat"

if struct == "UPTREND":
    struct_class = "pill-bull"
elif struct == "DOWNTREND":
    struct_class = "pill-bear"
else:
    struct_class = "pill-flat"

st.markdown(
    f"""
    <div class="metric-row">
      <div class="metric">
        <div class="metric-label">⊛ PRICE</div>
        <div class="metric-value">{price:,.2f}</div>
        <div class="metric-delta {delta_class}">{delta_arrow} {delta_abs:+.2f} ({delta_pct:+.2f}%)</div>
      </div>
      <div class="metric">
        <div class="metric-label">∿ EMA(20)</div>
        <div class="metric-value">{ema20_v:,.2f}</div>
        <div class="metric-pill {ema_class}">{ema_state}</div>
      </div>
      <div class="metric">
        <div class="metric-label">⊕ RSI(14)</div>
        <div class="metric-value">{rsi_v:.1f}</div>
        <div class="metric-pill {rsi_class}">{rsi_label}</div>
      </div>
      <div class="metric">
        <div class="metric-label">⊜ ATR(14)</div>
        <div class="metric-value">{atr_v:,.2f}</div>
        <div class="metric-delta flat">VOLATILITY</div>
      </div>
      <div class="metric">
        <div class="metric-label">⌬ STRUCTURE</div>
        <div class="metric-value" style="font-size:1.25rem;">{struct}</div>
        <div class="metric-pill {struct_class}">SMC</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════
#  CHART (premium plotly styling)
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header"><div class="section-icon">◇</div><h2>Price Action · Fibonacci · Liquidity</h2><div class="section-line"></div></div>',
    unsafe_allow_html=True,
)

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    name="Price",
    increasing=dict(line=dict(color=T["primary"], width=1), fillcolor=T["primary"]),
    decreasing=dict(line=dict(color=T["danger"],  width=1), fillcolor=T["danger"]),
))

fig.add_trace(go.Scatter(
    x=df.index, y=df["ema20"],
    line=dict(color=T["primary"], width=1.8),
    name="EMA 20", opacity=0.92,
))
fig.add_trace(go.Scatter(
    x=df.index, y=df["ema50"],
    line=dict(color=T["accent"], width=1.4, dash="dot"),
    name="EMA 50", opacity=0.78,
))

if fib:
    fig.add_hrect(
        y0=fib["0.5"], y1=fib["0.618"],
        fillcolor=T["primary"], opacity=0.13,
        line_width=0,
        annotation_text="◆ GOLDEN ZONE",
        annotation_position="top right",
        annotation=dict(
            font=dict(family="JetBrains Mono", size=10, color=T["primary"]),
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor=T["primary"], borderwidth=1, borderpad=4,
        ),
    )
    for level in ["0.236", "0.382", "0.5", "0.618", "0.786"]:
        fig.add_hline(
            y=fib[level],
            line_dash="dot",
            line_color="rgba(245,194,66,0.30)",
            line_width=1,
            annotation_text=f"FIB {level}",
            annotation_position="right",
            annotation=dict(
                font=dict(family="JetBrains Mono", size=9, color=T["gold"]),
            ),
        )

for r in sr.get("resistance", [])[:3]:
    fig.add_hline(
        y=r,
        line_color=T["danger"], line_dash="dash", line_width=1.2, opacity=0.65,
        annotation_text=f"R · {r:,.2f}",
        annotation_position="left",
        annotation=dict(
            font=dict(family="JetBrains Mono", size=10, color=T["danger"]),
            bgcolor="rgba(255,77,106,0.18)",
            bordercolor=T["danger"], borderwidth=1, borderpad=4,
        ),
    )
for s in sr.get("support", [])[:3]:
    fig.add_hline(
        y=s,
        line_color=T["primary"], line_dash="dash", line_width=1.2, opacity=0.65,
        annotation_text=f"S · {s:,.2f}",
        annotation_position="left",
        annotation=dict(
            font=dict(family="JetBrains Mono", size=10, color=T["primary"]),
            bgcolor="rgba(0,229,160,0.18)",
            bordercolor=T["primary"], borderwidth=1, borderpad=4,
        ),
    )

if liq_grab and liq_grab.get("detected"):
    fig.add_annotation(
        x=df.index[-1], y=liq_grab.get("level", price),
        text="◆ LIQUIDITY GRAB",
        showarrow=True, arrowhead=2,
        arrowcolor=T["warning"],
        font=dict(family="JetBrains Mono", size=10, color=T["warning"]),
        bgcolor="rgba(255,181,71,0.18)",
        bordercolor=T["warning"], borderwidth=1, borderpad=5,
    )

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=600,
    margin=dict(l=10, r=85, t=20, b=20),
    xaxis=dict(
        rangeslider=dict(visible=False),
        gridcolor="rgba(255,255,255,0.04)",
        showgrid=True, zeroline=False,
        showspikes=True,
        spikecolor="rgba(255,255,255,0.25)",
        spikethickness=1, spikedash="dot",
        tickfont=dict(family="JetBrains Mono", size=10, color=T["text_3"]),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        showgrid=True, zeroline=False, side="right",
        tickfont=dict(family="JetBrains Mono", size=10, color=T["text_3"]),
    ),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="left", x=0,
        bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11, color=T["text_2"]),
    ),
    hoverlabel=dict(
        bgcolor="rgba(11,13,18,0.96)",
        bordercolor="rgba(255,255,255,0.18)",
        font=dict(family="JetBrains Mono", size=11, color=T["text"]),
    ),
    dragmode="zoom",
)

st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# ════════════════════════════════════════════════════════════════════════════
#  SIGNALS + AI VERDICT (two columns)
# ════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1.4, 1])

with col_left:
    st.markdown(
        '<div class="section-header"><div class="section-icon">◆</div><h2>Signal Engine</h2><div class="section-line"></div></div>',
        unsafe_allow_html=True,
    )

    market = signals.get("market")
    pending = signals.get("pending", [])

    if market:
        side = market.side.upper()
        css_kind = side.lower()
        st.markdown(
            f"""
            <div class="signal-card {css_kind}">
              <div class="signal-header">
                <div style="display:flex;align-items:center;gap:10px;">
                  <span class="signal-badge {css_kind}">{side} NOW</span>
                  <div>
                    <div class="signal-title">Market Execution</div>
                    <div class="signal-subtitle">{market.confidence}% conviction · {market.timeframe.upper()}</div>
                  </div>
                </div>
                <span class="chip">⚡ EXECUTE</span>
              </div>
              <div class="signal-grid">
                <div class="signal-stat">
                  <div class="signal-stat-label">⊕ ENTRY</div>
                  <div class="signal-stat-value">{market.entry:,.2f}</div>
                </div>
                <div class="signal-stat">
                  <div class="signal-stat-label">⊘ STOP LOSS</div>
                  <div class="signal-stat-value" style="color:var(--danger);">{market.stop_loss:,.2f}</div>
                </div>
                <div class="signal-stat">
                  <div class="signal-stat-label">⊛ TAKE PROFIT</div>
                  <div class="signal-stat-value" style="color:var(--primary);">{market.take_profit:,.2f}</div>
                </div>
                <div class="signal-stat">
                  <div class="signal-stat-label">⊜ R:R · LOTS</div>
                  <div class="signal-stat-value">1:{market.rr:.1f} · {market.lot_size:.2f}</div>
                </div>
              </div>
              <div class="signal-rationale">{market.rationale}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="signal-card info">
              <div class="signal-header">
                <div style="display:flex;align-items:center;gap:10px;">
                  <span class="signal-badge standby">STANDBY</span>
                  <div>
                    <div class="signal-title">Awaiting Confluence</div>
                    <div class="signal-subtitle">EMA · RSI · Liquidity criteria not aligned</div>
                  </div>
                </div>
                <span class="chip">◌ MONITOR</span>
              </div>
              <div class="signal-rationale">
                No market execution signal on the current candle. Watch the pending orders below for upcoming setups at Golden Zone, S/R, or trendline retest.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if pending:
        for sig in pending[:4]:
            kind_cls = "buy" if sig.side.upper() == "BUY" else ("sell" if sig.side.upper() == "SELL" else "alert")
            badge_cls = "limit" if "LIMIT" in sig.kind.upper() else ("alert" if "ALERT" in sig.kind.upper() else kind_cls)
            st.markdown(
                f"""
                <div class="signal-card {kind_cls}">
                  <div class="signal-header">
                    <div style="display:flex;align-items:center;gap:10px;">
                      <span class="signal-badge {badge_cls}">{sig.kind.upper()}</span>
                      <div>
                        <div class="signal-title">{sig.side.upper()} Setup</div>
                        <div class="signal-subtitle">{sig.confidence}% conviction · {sig.timeframe.upper()}</div>
                      </div>
                    </div>
                    <span class="chip">⏱ PENDING</span>
                  </div>
                  <div class="signal-grid">
                    <div class="signal-stat">
                      <div class="signal-stat-label">⊕ TRIGGER</div>
                      <div class="signal-stat-value">{sig.entry:,.2f}</div>
                    </div>
                    <div class="signal-stat">
                      <div class="signal-stat-label">⊘ STOP LOSS</div>
                      <div class="signal-stat-value" style="color:var(--danger);">{sig.stop_loss:,.2f}</div>
                    </div>
                    <div class="signal-stat">
                      <div class="signal-stat-label">⊛ TAKE PROFIT</div>
                      <div class="signal-stat-value" style="color:var(--primary);">{sig.take_profit:,.2f}</div>
                    </div>
                    <div class="signal-stat">
                      <div class="signal-stat-label">⊜ R:R · LOTS</div>
                      <div class="signal-stat-value">1:{sig.rr:.1f} · {sig.lot_size:.2f}</div>
                    </div>
                  </div>
                  <div class="signal-rationale">{sig.rationale}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="padding:1rem;color:var(--text-4);font-size:0.85rem;text-align:center;border:1px dashed var(--border);border-radius:12px;">No pending orders queued.</div>',
            unsafe_allow_html=True,
        )

with col_right:
    st.markdown(
        '<div class="section-header"><div class="section-icon">◈</div><h2>AI Intelligence</h2><div class="section-line"></div></div>',
        unsafe_allow_html=True,
    )

    decision = (ai_decision.get("decision", "WAIT") or "WAIT").upper()
    confidence = ai_decision.get("confidence", 0)
    prob_up = ai_decision.get("probability_up", 50)
    reasoning = ai_decision.get("reasoning", "")
    warnings = ai_decision.get("warnings", [])

    dec_cls = "buy" if decision == "BUY" else ("sell" if decision == "SELL" else "wait")

    st.markdown(
        f"""
        <div class="ai-verdict">
          <div class="ai-header">
            <div class="ai-orb">◈</div>
            <div class="ai-meta">
              <div class="ai-name">Claude Verdict</div>
              <div class="ai-model">{ai_model.upper()}</div>
            </div>
          </div>
          <div class="ai-verdict-decision {dec_cls}">{decision}</div>
          <div class="ai-confidence-bar">
            <div class="ai-conf-row">
              <span>CONFIDENCE</span>
              <span style="color:var(--primary);">{confidence}%</span>
            </div>
            <div class="ai-conf-track">
              <div class="ai-conf-fill" style="width:{confidence}%;"></div>
            </div>
          </div>
          <div class="ai-reasoning">
            {reasoning if reasoning else 'Awaiting reasoning from AI engine. Connect Anthropic API key in sidebar for richer analysis.'}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Conic gauge for probability
    if prob_up > 55:
        gauge_color = T["primary"]
    elif prob_up < 45:
        gauge_color = T["danger"]
    else:
        gauge_color = T["text_3"]

    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title"><div class="panel-icon">↑</div>Direction Probability</div>
          </div>
          <div class="gauge-conic" style="--gauge-pct:{prob_up};--gauge-color:{gauge_color};">
            <div class="gauge-conic-glow"></div>
            <div class="gauge-conic-ring"></div>
            <div class="gauge-conic-inner">
              <div class="gauge-conic-value">{prob_up}<span class="gauge-conic-pct">%</span></div>
              <div class="gauge-conic-label">UPSIDE</div>
            </div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:1rem;font-family:'JetBrains Mono',monospace;font-size:0.74rem;font-weight:700;">
            <span style="color:var(--danger);">▼ {100-prob_up}% DOWN</span>
            <span style="color:var(--primary);">{prob_up}% UP ▲</span>
          </div>
          <div style="height:6px;background:rgba(255,255,255,0.06);border-radius:999px;margin-top:8px;overflow:hidden;">
            <div style="height:100%;width:{prob_up}%;background:linear-gradient(90deg,var(--danger),var(--warning),var(--primary));border-radius:999px;box-shadow:0 0 12px rgba(0,229,160,0.30);"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if warnings:
        warn_html = "".join(f'<li style="margin-bottom:6px;">{w}</li>' for w in warnings)
        st.markdown(
            f"""
            <div class="panel" style="border-color:rgba(255,181,71,0.25);background:linear-gradient(135deg,rgba(255,181,71,0.06),rgba(255,181,71,0.01));">
              <div class="panel-header" style="border-bottom-color:rgba(255,181,71,0.20);">
                <div class="panel-title">
                  <div class="panel-icon" style="background:rgba(255,181,71,0.10);border-color:rgba(255,181,71,0.25);color:var(--warning);">!</div>
                  Risk Warnings
                </div>
              </div>
              <ul style="margin:0;padding-left:1.25rem;color:var(--text-2);font-size:0.83rem;line-height:1.65;">
                {warn_html}
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════════════════
#  NEWS INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header"><div class="section-icon">◉</div><h2>News Intelligence · Macro Database</h2><div class="section-line"></div></div>',
    unsafe_allow_html=True,
)

news_col1, news_col2 = st.columns([1, 1.35])

with news_col1:
    direction = (news_prob.get("direction", "NEUTRAL") or "NEUTRAL").upper()
    prob_pct = news_prob.get("probability", 0)
    avg_pips = news_prob.get("avg_pips", 0)
    breakout = news_prob.get("volatility_breakout", False)

    dir_arrow = "▲" if direction == "UP" else ("▼" if direction == "DOWN" else "■")
    dir_color = T["primary"] if direction == "UP" else (T["danger"] if direction == "DOWN" else T["text_3"])

    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title"><div class="panel-icon">⊙</div>Deviation-Based Forecast</div>
          </div>
          <div style="text-align:center;padding:1.25rem 0 0.5rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:3.6rem;font-weight:900;color:{dir_color};line-height:1;letter-spacing:-0.045em;text-shadow:0 0 30px {dir_color}33;">
              {dir_arrow} {direction}
            </div>
            <div style="font-size:0.75rem;color:var(--text-3);margin-top:10px;text-transform:uppercase;letter-spacing:0.10em;font-family:'JetBrains Mono',monospace;font-weight:600;">
              {prob_pct}% probability · ~{avg_pips:.0f} pips avg move
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.7rem;margin-top:1.25rem;">
            <div class="signal-stat">
              <div class="signal-stat-label">INDICATOR</div>
              <div class="signal-stat-value" style="font-size:0.85rem;">{news_type}</div>
            </div>
            <div class="signal-stat">
              <div class="signal-stat-label">DEVIATION</div>
              <div class="signal-stat-value">{news_actual - news_forecast:+.2f}</div>
            </div>
            <div class="signal-stat">
              <div class="signal-stat-label">BREAKOUT</div>
              <div class="signal-stat-value" style="color:{'var(--warning)' if breakout else 'var(--text-3)'};">
                {'YES' if breakout else 'NO'}
              </div>
            </div>
          </div>
          <div class="signal-rationale">{inst_advice.get('summary', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with news_col2:
    recent = summarize_recent(news_history, limit=8)
    rows_html = ""
    max_pip = max((abs(r.get("xauusd_15m_pips", 0)) for r in recent), default=1) or 1
    for rec in recent:
        ev = rec.get("indicator", "—")
        date = rec.get("date", "—")
        actual = rec.get("actual", 0)
        forecast = rec.get("forecast", 0)
        dev = actual - forecast
        pip = rec.get("xauusd_15m_pips", 0)
        pip_color = T["primary"] if pip > 0 else (T["danger"] if pip < 0 else T["text_3"])
        dev_color = T["primary"] if dev > 0 else (T["danger"] if dev < 0 else T["text_3"])
        bar_pct = abs(pip) / max_pip * 100
        bar_bg = (
            f"linear-gradient(90deg, transparent {100-bar_pct}%, {pip_color}22 {100-bar_pct}%)"
            if pip < 0 else
            f"linear-gradient(90deg, {pip_color}22 {bar_pct}%, transparent {bar_pct}%)"
        )
        rows_html += f"""
        <div class="news-row">
          <div class="news-date">{date}</div>
          <div class="news-event">{ev}</div>
          <div class="news-num" style="color:var(--text-3);">A: {actual}</div>
          <div class="news-num" style="color:{dev_color};">Δ {dev:+.2f}</div>
          <div class="news-pip-bar" style="color:{pip_color};background:{bar_bg};">{pip:+.0f} pips</div>
        </div>
        """
    st.markdown(
        '<div class="panel">'
        '<div class="panel-header">'
        '<div class="panel-title"><div class="panel-icon">≡</div>Recent Macro Releases</div>'
        f'<span class="chip">{len(recent)} events</span>'
        '</div>'
        + rows_html +
        '</div>',
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
#  DEBUG SNAPSHOT
# ════════════════════════════════════════════════════════════════════════════
with st.expander("◇ Raw AI Snapshot (debug)", expanded=False):
    st.json(snapshot, expanded=False)

# ════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
    <div class="footer">
      <div>◆ Claude Trading Terminal · v3.0 Premium</div>
      <div>Last sync: {now_str} · Bars: {len(df)} · {symbol_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if auto_refresh:
    time.sleep(60)
    st.rerun()

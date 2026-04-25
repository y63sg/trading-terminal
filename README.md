# Claude Trading Terminal

Murun-ы хувийн арилжааны туслах вэбсайт. Day Trading + Trend Following + SMC.

## 1. Файлын бүтэц

```
.
├── app.py                # Streamlit вэб app
├── strategy.py           # 20 EMA, RSI, Fib, SMC, S/R, trendline
├── news_analyzer.py      # 10 сарын мэдээний нөлөөллийн логик
├── signal_engine.py      # Market + Pending дохио үүсгэгч
├── ai_engine.py          # Claude API wrapper + fallback
├── system_prompt.md      # AI-д зориулсан Advanced System Prompt
├── data/
│   └── news_history.json # NFP / CPI / Unemployment 10 сарын түүх
├── requirements.txt
└── README.md
```

## 2. Суулгах

```bash
pip install -r requirements.txt
```

`pandas-ta` build алдаа гарвал орхиж болно — `strategy.py` нь fallback EMA/RSI/ATR-тай.

## 3. Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Эсвэл sidebar-н "AI Engine" хэсэгт гараар оруулна. **Хоосон үед rule-engine fallback автоматаар ажиллана**, дохио үргэлжилнэ.

## 4. Ажиллуулах

```bash
streamlit run app.py
```

Браузер дээр `http://localhost:8501` нээгдэнэ.

## 5. Хэрэглэх

1. **Sidebar → 💼 Дансны удирдлага**: үлдэгдэл, эрсдэл (1%), RR зорилго.
2. **Sidebar → 📈 Зах зээл / Хугацаа**: Symbol (XAUUSD default), 15m/1h/4h.
3. **Sidebar → 📰 News тохиргоо**: NFP/CPI/Unemployment, Actual & Forecast оруулах.
4. Үндсэн хэсэгт:
   * 🕯️ Лааны график + Fib Golden Zone + S/R
   * ⚡ **Market Execution** — `NOW: BUY/SELL` бүх confluence бүрдсэн үед.
   * 📅 **Pending Orders** — `LIMIT ORDER` ба `TRENDLINE ALERT` ирээдүйн setup-ууд.
   * 🧠 **Market Intelligence** — Claude-ын эцсийн шийдвэр + Probability Gauge.
   * 📰 **News Intelligence** — deviation-аас гаргасан магадлал, institutional move зөвлөмж.

## 6. Data API талаар анхаарах

* `yfinance` нь real-time биш, ~1–5 минутын саатлаар үнэ өгдөг. Live арилжаанд MT5 эсвэл OANDA API-тай хослуулан өөрчилж болно — `fetch_ohlc` функцийг солих.
* 5m/15m нь зөвхөн 7–10 хоног рүү буцдаг (yfinance limit).
* 4h timeframe нь 60m-аас resample хийсэн.

## 7. Стратегийн дүрэм (хатуу)

* Эрсдэл: 1% / арилжаа.
* RR: 1:2 (TP1) – 1:4 (TP2).
* SL нь structure-ийн ард байх ёстой (swept liquidity, swing, эсвэл 1.2–1.5×ATR).
* RSI > 75 үед long-аас зайлсхий, < 25 үед short-аас зайлсхий.
* Мэдээ гарснаас хойш эхний 5 мин хүлээж, institutional move руу ор.

## 8. Цаашид өргөтгөх

* `fetch_ohlc`-г MT5/OANDA/CCXT-тай солих → live ticks.
* `data/news_history.json`-г Forex Factory / Investing.com-аас өдөр бүр scrape хийх.
* `signal_engine.py`-д Order Block / Fair Value Gap (FVG) илрүүлэгч нэмэх.
* Backtest module — `strategy.py`-г түүхэн өгөгдөл дээр шалгах.
* Telegram/Discord webhook → дохио гарсан үед мессеж явуулах.

## Анхааруулга

Энэ нь **арилжааны туслах багаж** болохоос **арилжаа автоматаар хийдэг bot биш**. Бүх захиалгыг өөрөө гараар орох ёстой. Санхүүгийн зөвлөгөө биш.

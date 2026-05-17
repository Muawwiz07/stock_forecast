# 📈 StockCast — AI Stock Intelligence Platform

> **Intelligent · Shariah-Screened · Predictive**
> XGBoost forecasting · Signal engine · NLP sentiment · Live broker sync · Multi-language

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Supabase](https://img.shields.io/badge/Supabase-Auth%20%2B%20DB-3ECF8E?style=flat-square&logo=supabase)](https://supabase.com)
[![License](https://img.shields.io/badge/License-Proprietary-red?style=flat-square)](./license.py)

---

## 🚀 Live Demo

| Credential | Value |
|---|---|
| Email | `demo@stockcast.com` |
| Password | `demo1234` |

Or click **"Enter Demo Mode"** on the login screen — no account needed.

---

## ✨ Features

### 🤖 AI Forecasting Engine
- **XGBoost** model trained on 30 technical indicators + lag closes
- Bootstrap confidence intervals (95% CI ribbon)
- Configurable horizon (7–30 days), sequence length, hyperparameters
- RMSE / MAE / MAPE / R² metrics on test set

### 📊 Signal Engine
- **BUY / SELL / HOLD** with 0–100% confidence scoring
- 9-factor composite: RSI, MACD, Bollinger %B, MA cross, Volume, Stochastic, Williams %R, ADX, AI outlook
- Trend–sentiment conflict detection
- ATR-scaled take profit / stop loss / risk:reward

### 🧠 NLP Sentiment Analysis
- **FinBERT** (ProsusAI/finbert) — financial-domain transformer, most accurate
- TextBlob fallback — lightweight general NLP
- Keyword scorer — zero-dependency last resort
- Per-headline bar chart with pos/neg/neu breakdown

### 🕌 Shariah Compliance Screening
- AAOIFI Standard No.21 ratio checks: Debt/MarketCap < 30%, Debt/Assets < 33%, Cash/Assets < 33%
- Curated haram ticker list (alcohol, tobacco, gambling, weapons)
- Questionable tier for banks and financial firms requiring ratio review

### 📂 Portfolio Tracker
- Add / remove holdings with qty and avg cost
- Live P&L, P&L%, sector breakdown, Win Rate, Best/Worst performer
- Full transaction history, CSV export, Supabase-persisted per user

### 👁 Watchlist
- Supabase-persisted per user (up to 5 free / 50 pro)
- Daily email digest on first login (via SMTP)
- Signal alert emails with TP/SL/confidence

### 🌍 Multi-language UI
- English · Arabic · Urdu · Hindi · Chinese (Simplified)
- Full UI string dictionary — add new languages in `config.py`

### 🔗 Live Broker Sync
- **Zerodha Kite** (India) — live holdings, P&L, order history
- **Upstox** (India) — portfolio sync, margin data
- **Alpaca** (US) — fractional shares, paper trading, live orders

### 📈 Markets Tab
- Live S&P 500, NASDAQ 100, DOW, VIX
- Sector ETF heatmap (10 sectors)
- Fear & Greed index (CNN → alternative.me → VIX fallback)
- Live ticker tape (12 major symbols)

### 🏦 Startup Hub
- Macro risk scanner, treasury optimizer, competitor radar
- AI investor reports (Claude API)
- Investor-ready CSV/PDF export

---

## 🗂 Project Structure

```
stockcast/
├── app.py              # Entry point — page config, auth gate, tabs, render loop
├── config.py           # Shared imports, constants, logging, Supabase init
├── signals.py          # BUY/SELL/HOLD engine, AI insight generator, signal card UI
├── data.py             # yfinance helpers, market data, Fear & Greed, ticker search
├── database.py         # All Supabase helpers — portfolio, watchlist, usage, plan
├── analytics.py        # Technical indicators, XGBoost, backtest, bootstrap CI, Shariah
├── brokers.py          # Zerodha, Upstox, Alpaca live integrations
├── sentiment.py        # FinBERT → TextBlob → keyword NLP pipeline
├── authgate.py         # Login / signup / demo UI (Supabase Auth)
├── ui_components.py    # Reusable UI components (toast, accordion, bento grid)
├── send_alerts.py      # Email alert scheduler (GitHub Actions / cron)
├── train.py            # Offline XGBoost training script
├── crypto.py           # Crypto market data helpers
├── license.py          # Proprietary license
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python version pin
├── .streamlit/
│   └── secrets.toml    # 🔒 Never commit — see setup below
└── .github/
    └── workflows/
        └── daily_alerts.yml  # GitHub Actions daily digest cron
```

---

## ⚙️ Setup & Deployment

### 1. Clone the repo
```bash
git clone https://github.com/Muawwiz07/stock_forecast.git
cd stock_forecast
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Optional — for FinBERT NLP (recommended):**
> ```bash
> pip install transformers torch
> ```

### 3. Configure secrets

Create `.streamlit/secrets.toml`:

```toml
# ── Supabase (required) ───────────────────────────────────────────
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-public-key"

# ── Email alerts (optional) ───────────────────────────────────────
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "you@gmail.com"
SMTP_PASS = "your-app-password"
SMTP_FROM = "StockCast <you@gmail.com>"

# ── Broker integrations (optional) ───────────────────────────────
ZERODHA_API_KEY      = ""
ZERODHA_ACCESS_TOKEN = ""
UPSTOX_API_KEY       = ""
UPSTOX_ACCESS_TOKEN  = ""
ALPACA_KEY           = ""
ALPACA_SECRET        = ""
ALPACA_BASE_URL      = "https://paper-api.alpaca.markets"

# ── AI reports (optional) ─────────────────────────────────────────
ANTHROPIC_API_KEY = "sk-ant-..."

# ── Analytics (optional) ──────────────────────────────────────────
POSTHOG_API_KEY = ""
```

> ⚠️ **Never commit `secrets.toml` to Git.** It is already in `.gitignore`.

### 4. Run locally
```bash
streamlit run app.py
```

### 5. Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set **Main file path** to `app.py`
4. Under **Advanced settings → Secrets**, paste your `secrets.toml` contents
5. Click **Deploy**

---

## 🗄 Supabase Database Setup

Run once in your [Supabase SQL editor](https://app.supabase.com):

```sql
-- Portfolio holdings
CREATE TABLE IF NOT EXISTS portfolio_holdings (
  id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id       UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  ticker        TEXT NOT NULL,
  name          TEXT,
  sector        TEXT,
  qty           FLOAT NOT NULL,
  avg_cost      FLOAT NOT NULL,
  current_price FLOAT,
  pl            FLOAT,
  pl_pct        FLOAT,
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, ticker)
);

-- Transaction history
CREATE TABLE IF NOT EXISTS portfolio_history (
  id         UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  date       TEXT,
  type       TEXT,
  ticker     TEXT,
  shares     FLOAT,
  price      FLOAT,
  amount     FLOAT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Watchlist
CREATE TABLE IF NOT EXISTS watchlist (
  id           UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id      UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  stock_symbol TEXT NOT NULL,
  created_at   TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, stock_symbol)
);

-- Usage limits + plan
CREATE TABLE IF NOT EXISTS user_usage (
  user_id            UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  usage_count        INT  NOT NULL DEFAULT 0,
  last_used_date     DATE NOT NULL DEFAULT CURRENT_DATE,
  plan               TEXT NOT NULL DEFAULT 'free',
  email_alerts_enabled BOOLEAN DEFAULT FALSE
);
```

---

## 📦 Requirements

```
streamlit>=1.35.0
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.20.0
xgboost>=2.0.0
scikit-learn>=1.4.0
supabase>=2.0.0
requests>=2.31.0
textblob>=0.18.0
transformers>=4.40.0   # optional — for FinBERT
torch>=2.0.0           # optional — for FinBERT
kiteconnect            # optional — Zerodha
upstox-python-sdk      # optional — Upstox
alpaca-trade-api       # optional — Alpaca
posthog                # optional — Analytics
```

---

## 🔑 Plan Tiers

| Feature | Free | Pro |
|---|---|---|
| Daily analyses | 3 | Unlimited |
| Watchlist stocks | 5 | 50 |
| Forecast horizon | 7 days | 30 days |
| Bootstrap CI | ❌ | ✅ |
| Model compare | ❌ | ✅ |
| Multi-stock | ❌ | ✅ |
| Data history | 3 years | 10 years |

---

## ⚠️ Disclaimer

StockCast is a **research and educational tool only**. It does not constitute financial advice. Always consult a licensed financial advisor before making investment decisions. Past model performance does not guarantee future results.

---

## 📄 License

Proprietary — All Rights Reserved © 2026 Stockcast.
Unauthorized copying, distribution, or use is strictly prohibited.
For licensing enquiries: `legal@stockcast.com`

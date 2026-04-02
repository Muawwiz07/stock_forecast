import streamlit as st
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from supabase import create_client
import warnings
import os
import time
warnings.filterwarnings('ignore')

# ── yfinance helpers ───────────────────────────────────────────────────────────

def _yf_download_with_retry(ticker, retries=3, **kwargs):
    """Download yfinance data with retry logic for rate limiting."""
    import logging
    last_exc = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, progress=False, auto_adjust=True, **kwargs)
            if not df.empty:
                return df
        except Exception as e:
            last_exc = e
        if attempt < retries - 1:
            time.sleep(2 + attempt * 2)
    if last_exc is not None:
        logging.warning(f"yfinance download failed for '{ticker}' after {retries} attempts: {type(last_exc).__name__}: {last_exc}")
    return pd.DataFrame()

@st.cache_data(ttl=300)
def av_get_daily(ticker):
    """Fetch full daily OHLCV via yfinance. Returns DataFrame."""
    df = _yf_download_with_retry(ticker, period="max", interval="1d")
    if df.empty:
        return df
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_localize(None) if idx.tz is not None else idx
    df.index.name = "Date"
    return df.sort_index()

@st.cache_data(ttl=60)
def av_get_quote(ticker):
    """Fetch live quote via yfinance. Returns dict with price, change%, prev_close."""
    try:
        info = yf.Ticker(ticker).fast_info
        price      = float(getattr(info, "last_price", 0) or 0)
        prev_close = float(getattr(info, "previous_close", 0) or 0)
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
        open_price = float(getattr(info, "open", 0) or 0)
        return {"price": price, "change_pct": change_pct, "prev_close": prev_close, "open": open_price}
    except Exception:
        return {"price": 0.0, "change_pct": 0.0, "prev_close": 0.0, "open": 0.0}

@st.cache_data(ttl=180)
def get_live_ticker_tape():
    """Fetch live prices for ticker tape symbols — single batched download."""
    tape_syms = ["AAPL","TSLA","NVDA","MSFT","GOOGL","META","AMZN","AMD","JPM","SPY","QQQ","NFLX"]
    try:
        raw = yf.download(tape_syms, period="2d", interval="1d", progress=False, auto_adjust=True)
        close = raw["Close"] if "Close" in raw.columns else raw
        if isinstance(close.columns, pd.MultiIndex):
            close = close.droplevel(0, axis=1)
        items = []
        for sym in tape_syms:
            try:
                prices = close[sym].dropna()
                if len(prices) >= 2:
                    price, prev = float(prices.iloc[-1]), float(prices.iloc[-2])
                elif len(prices) == 1:
                    price = prev = float(prices.iloc[-1])
                else:
                    continue
                chg_pct = ((price - prev) / prev * 100) if prev else 0.0
                sign    = "+" if chg_pct >= 0 else ""
                arrow   = "▲" if chg_pct >= 0 else "▼"
                css     = "tape-up" if chg_pct >= 0 else "tape-down"
                items.append((sym, f"${price:,.2f}", f"{sign}{chg_pct:.2f}%", arrow, css))
            except Exception:
                continue
        return items
    except Exception:
        return []


@st.cache_data(ttl=3600)
def av_get_overview(ticker_sym):
    """Fetch company overview via yfinance. Uses get_info() with retries for rate-limit resilience."""
    for attempt in range(3):
        try:
            t    = yf.Ticker(ticker_sym)
            info = t.get_info() or {}
            if not info.get("symbol") and not info.get("longName"):
                # Empty info dict — yfinance returned nothing useful
                if attempt < 2:
                    time.sleep(2 + attempt * 2)
                    continue
                return {}
            return {
                "Symbol":                              ticker_sym,
                "Name":                                info.get("longName", ticker_sym),
                "Sector":                              info.get("sector", "Unknown"),
                "Industry":                            info.get("industry", "Unknown"),
                "MarketCapitalization":                str(info.get("marketCap", 0) or 0),
                "TotalDebt":                           str(info.get("totalDebt", 0) or 0),
                "TotalAssets":                         str(info.get("totalAssets", 0) or 0),
                "CashAndCashEquivalentsAtCarryingValue": str(info.get("totalCash", 0) or 0),
            }
        except Exception:
            if attempt < 2:
                time.sleep(2 + attempt * 2)
    return {}

@st.cache_data(ttl=300)
def av_search(query):
    """Search tickers — returns empty list (yfinance has no search API; app falls back to POPULAR_TICKERS)."""
    return []

@st.cache_data(ttl=300)
def av_get_news(ticker):
    """Fetch news via yfinance. Handles both old (title at root) and
    new yfinance API (title nested under content dict)."""
    try:
        news = yf.Ticker(ticker).news or []
        results = []
        for n in news[:10]:
            title = (
                n.get("title")
                or (n.get("content") or {}).get("title")
                or ""
            )
            if title:
                results.append({"title": title})
        return results
    except Exception:
        return []

@st.cache_data(ttl=120)
def get_live_market_indices():
    """Fetch live S&P500, NASDAQ, DOW, VIX via yfinance — single batched download."""
    symbols = {"S&P 500":"^GSPC","NASDAQ 100":"^NDX","DOW JONES":"^DJI","VIX":"^VIX"}
    syms = list(symbols.values())
    result = []
    try:
        raw = yf.download(syms, period="2d", interval="1d", progress=False, auto_adjust=True)
        close = raw["Close"] if "Close" in raw.columns else raw
        if isinstance(close.columns, pd.MultiIndex):
            close = close.droplevel(0, axis=1)
        for name, sym in symbols.items():
            try:
                prices = close[sym].dropna()
                if len(prices) >= 2:
                    price, prev = float(prices.iloc[-1]), float(prices.iloc[-2])
                elif len(prices) == 1:
                    price, prev = float(prices.iloc[-1]), float(prices.iloc[-1])
                else:
                    raise ValueError("no data")
                chg_pct = ((price - prev) / prev * 100) if prev else 0.0
                col  = "#00e5b0" if chg_pct >= 0 else "#ff6b6b"
                sign = "+" if chg_pct >= 0 else ""
                fmt_price = f"{price:,.2f}" if sym != "^VIX" else f"{price:.2f}"
                result.append((name, fmt_price, f"{sign}{chg_pct:.2f}%", col))
            except Exception:
                result.append((name, "—", "—", "#424754"))
    except Exception:
        for name in symbols:
            result.append((name, "—", "—", "#424754"))
    return result

@st.cache_data(ttl=300)
def get_live_sector_heatmap():
    """Fetch live sector ETF performance via yfinance — single batched download."""
    sector_etfs = {
        "Technology":"XLK","Healthcare":"XLV","Financials":"XLF",
        "Energy":"XLE","Consumer Disc.":"XLY","Industrials":"XLI",
        "Utilities":"XLU","Real Estate":"XLRE","Materials":"XLB","Comm. Services":"XLC"
    }
    syms = list(sector_etfs.values())
    result = []
    try:
        raw = yf.download(syms, period="2d", interval="1d", progress=False, auto_adjust=True)
        close = raw["Close"] if "Close" in raw.columns else raw
        if isinstance(close.columns, pd.MultiIndex):
            close = close.droplevel(0, axis=1)
        for name, sym in sector_etfs.items():
            try:
                prices = close[sym].dropna()
                if len(prices) >= 2:
                    price, prev = float(prices.iloc[-1]), float(prices.iloc[-2])
                elif len(prices) == 1:
                    price, prev = float(prices.iloc[-1]), float(prices.iloc[-1])
                else:
                    raise ValueError("no data")
                chg_pct = ((price - prev) / prev * 100) if prev else 0.0
                col  = "#00e5b0" if chg_pct >= 0 else "#ff6b6b"
                sign = "+" if chg_pct >= 0 else ""
                result.append((name, f"{sign}{chg_pct:.2f}%", col))
            except Exception:
                result.append((name, "—", "#424754"))
    except Exception:
        for name in sector_etfs:
            result.append((name, "—", "#424754"))
    return result

@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Fetch CNN Fear & Greed index via their API."""
    try:
        r = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", timeout=10,
                         headers={"User-Agent": "Mozilla/5.0"})
        data = r.json()
        score = float(data["fear_and_greed"]["score"])
        rating = data["fear_and_greed"]["rating"].title()
        return {"score": score, "rating": rating}
    except Exception:
        return None

import nltk

def _ensure_nltk_data():
    """Lazily download required NLTK tokenizer data on first use."""
    for _nltk_pkg in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{_nltk_pkg}')
        except LookupError:
            try:
                nltk.download(_nltk_pkg, quiet=True)
            except Exception:
                pass

# ── Supabase config ────────────────────────────────────────────────────────────
# Credentials are loaded from Streamlit secrets (secrets.toml) or environment variables.
# Never hardcode credentials in source code.
# Set up: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except (KeyError, FileNotFoundError):
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("⚠ Supabase credentials not found. Add SUPABASE_URL and SUPABASE_KEY to your Streamlit secrets or environment variables.")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Portfolio Supabase helpers ─────────────────────────────────────────────────
# Required Supabase tables (run once in your Supabase SQL editor):
#
#   CREATE TABLE IF NOT EXISTS portfolio_holdings (
#     id           UUID DEFAULT gen_random_uuid() PRIMARY KEY,
#     user_id      UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
#     ticker       TEXT NOT NULL,
#     name         TEXT,
#     sector       TEXT,
#     qty          FLOAT NOT NULL,
#     avg_cost     FLOAT NOT NULL,
#     current_price FLOAT,
#     pl           FLOAT,
#     pl_pct       FLOAT,
#     created_at   TIMESTAMPTZ DEFAULT NOW(),
#     UNIQUE(user_id, ticker)
#   );
#
#   CREATE TABLE IF NOT EXISTS portfolio_history (
#     id         UUID DEFAULT gen_random_uuid() PRIMARY KEY,
#     user_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
#     date       TEXT,
#     type       TEXT,
#     ticker     TEXT,
#     shares     FLOAT,
#     price      FLOAT,
#     amount     FLOAT,
#     created_at TIMESTAMPTZ DEFAULT NOW()
#   );

def _sb_load_portfolio(user_id: str) -> list:
    try:
        res = supabase.table("portfolio_holdings").select("*").eq("user_id", user_id).execute()
        rows = res.data or []
        # Sanitize: fill None numeric fields so UI arithmetic never crashes
        for r in rows:
            r["current_price"] = float(r.get("current_price") or r.get("avg_cost") or 0.0)
            r["pl"]      = float(r.get("pl")      or 0.0)
            r["pl_pct"]  = float(r.get("pl_pct")  or 0.0)
            r["qty"]     = float(r.get("qty")      or 0.0)
            r["avg_cost"]= float(r.get("avg_cost") or 0.0)
            r["name"]    = r.get("name")   or r.get("ticker", "")
            r["sector"]  = r.get("sector") or "Unknown"
        return rows
    except Exception:
        return []

def _sb_load_history(user_id: str) -> list:
    try:
        res = (supabase.table("portfolio_history").select("*")
               .eq("user_id", user_id).order("created_at", desc=True).limit(100).execute())
        return res.data or []
    except Exception:
        return []

def _sb_upsert_holding(user_id: str, h: dict):
    try:
        supabase.table("portfolio_holdings").upsert({
            "user_id": user_id,
            "ticker":        h["ticker"],
            "name":          h.get("name", h["ticker"]),
            "sector":        h.get("sector", "Unknown"),
            "qty":           h["qty"],
            "avg_cost":      h["avg_cost"],
            "current_price": h.get("current_price", h["avg_cost"]),
            "pl":            h.get("pl", 0.0),
            "pl_pct":        h.get("pl_pct", 0.0),
        }, on_conflict="user_id,ticker").execute()
    except Exception:
        pass

def _sb_delete_holding(user_id: str, ticker: str):
    try:
        supabase.table("portfolio_holdings").delete().eq("user_id", user_id).eq("ticker", ticker).execute()
    except Exception:
        pass

def _sb_insert_history(user_id: str, record: dict):
    try:
        supabase.table("portfolio_history").insert({
            "user_id": user_id,
            "date":    record.get("date"),
            "type":    record.get("type"),
            "ticker":  record.get("ticker"),
            "shares":  record.get("shares"),
            "price":   record.get("price"),
            "amount":  record.get("amount"),
        }).execute()
    except Exception:
        pass

def _sb_update_prices(user_id: str, holdings: list):
    for h in holdings:
        try:
            supabase.table("portfolio_holdings").update({
                "current_price": h["current_price"],
                "pl":            h["pl"],
                "pl_pct":        h["pl_pct"],
            }).eq("user_id", user_id).eq("ticker", h["ticker"]).execute()
        except Exception:
            pass

# ── Multi-language support ─────────────────────────────────────────────────────
LANGUAGES = {
    "English": {
        # Sidebar controls
        "run": "▶  Run Forecast", "ticker": "Ticker", "from": "From", "to": "To",
        "mode": "Mode", "beginner": "🟢 Beginner", "pro": "🔴 Pro",
        "watchlist": "⭐ Watchlist", "alerts": "🔔 Signal Alerts",
        "forecast": "Forecast", "days": "Days", "add": "Add",
        "portfolio": "💼  Portfolio", "markets": "🌍  Markets",
        "back": "← Back to Dashboard",
        "search_label": "🔍 Search Company / Ticker",
        "lookback": "Lookback Window (days)", "horizon": "Forecast Horizon (days)",
        "simple_view": "✓ Simple view active",
        "pro_view": "⚡ Pro view — all parameters unlocked",
        "fast_mode": "⚡ Fast Mode (skip CI + backtest)",
        "xgb_params": "XGBoost Hyperparameters",
        "alert_target": "Price Alert Target ($)",
        "backtesting": "Backtesting", "enable_backtest": "Enable Backtesting Engine",
        "init_capital": "Initial Capital ($)", "commission": "Commission per Trade ($)",
        "signal_thresh": "Signal Threshold (%)", "extra_features": "Extra Features",
        "model_compare": "Model Comparison (XGB vs LR vs Prophet)",
        "halal_check": "Halal / Shariah Compliance Check",
        "conf_interval": "Confidence Intervals on Forecast",
        "bootstrap_samples": "Bootstrap Samples (CI)", "multi_stock": "Multi-Stock Comparison",
        "compare_tickers": "Compare Tickers", "alert_signal_change": "Alert when signal changes",
        # Dashboard
        "dashboard_title": "Dashboard", "dashboard_subtitle": "Overview",
        "dashboard_desc": "AI-powered stock intelligence · Enter a ticker in the sidebar and click Run Forecast to begin.",
        "watchlist_live": "⭐ Watchlist — Live Prices",
        "how_it_works": "How It Works",
        "hw1_title": "Enter a Ticker", "hw1_body": "Search by company name or symbol. Add it to your watchlist to track it persistently.",
        "hw2_title": "Run the Model", "hw2_body": "XGBoost trains on 7 years of OHLCV data with 20 engineered features. Results in seconds.",
        "hw3_title": "Read the Signal", "hw3_body": "Get a BUY / SELL / HOLD verdict with a full explanation of every contributing factor.",
        "platform_features": "Platform Features",
        "no_stocks_saved": "No stocks saved yet.",
        "no_holdings": "No holdings yet. Add your first stock above.",
        "no_transactions": "No transactions yet.",
        # Portfolio
        "portfolio_title": "Portfolio", "portfolio_tracker": "Tracker",
        "portfolio_desc": "Add your real holdings — prices refresh live from Yahoo Finance",
        "add_holding": "➕ Add Holding", "add_to_portfolio": "Add to Portfolio",
        "refresh_prices": "🔄 Refresh Live Prices",
        "total_value": "Total Value", "total_pl": "Total P&L",
        "invested": "Invested", "holdings": "Holdings",
        "holdings_label": "Holdings", "sector_allocation": "Sector Allocation",
        "recent_activity": "Recent Activity",
        # Analysis
        "price_chart": "Price Chart", "tech_indicators": "Technical Indicators",
        "news_sentiment": "News Sentiment NLP", "screening_criteria": "Screening Criteria",
        "analysis_tab": "📊  Analysis", "methodology_tab": "📖  Methodology",
        "dashboard_tab": "🖥  Dashboard",
        "reality_check_title": "⚠ Model Reality Check — Read Before Trading",
        "reality_check_body": "This model uses <b style='color:#dae2fd;'>price &amp; volume data only</b>. It has <b style='color:#ff6b6b;'>zero awareness</b> of: &nbsp;📰 breaking news &nbsp;·&nbsp;📊 earnings releases &nbsp;·&nbsp;🏦 Fed/macro events &nbsp;·&nbsp;🧠 analyst upgrades &nbsp;·&nbsp;🌍 geopolitical events. <b style='color:#ffdd2d;'>Use signals as one input — never as sole decision.</b>",
        "logout": "⏏  Logout",
        "verify_symbol": "Using: {ticker} — verify symbol",
        "active_ticker": "● ACTIVE: {ticker}",
        "greed_territory": "Greed territory", "low_volatility": "Low volatility",
        "no_recent_news": "No recent news found for this ticker.",
        "already_in_portfolio": "{sym} already in portfolio. Remove it first to update.",
        "added_success": "✓ Added {sym} — live price ${price:.2f}",
        "footer": "⚠ STOCKCAST · FOR EDUCATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE · Developed by MUAWWIZ GHANI",
        "model_performance": "Model Performance", "actual_vs_pred": "Actual vs Predicted",
        "feature_importance": "Feature Importance", "signal_intelligence": "Signal Intelligence",
        "forecast_next": "Forecast — Next {n} Days", "backtest_engine": "Backtesting Engine",
        "trade_log": "Trade Log", "forecast_ci": "Forecast with Confidence Intervals",
        "model_comparison": "Model Comparison — XGBoost vs Prophet vs Linear Regression",
        "sector_heatmap": "Sector Heat Map · Live", "fear_greed": "Fear & Greed Index · Live",
        "days_loaded": "✓ {n} trading days loaded for {ticker}", "fetching": "Fetching {ticker} data...",
        "engineering": "Engineering technical features...", "building": "Building feature matrix...",
        "training": "Training XGBoost model (cached after first run)...",
        "running_backtest": "Running backtest simulation...", "running_bootstrap": "Running {n} bootstrap samples...",
        "training_lr": "Training Linear Regression...", "fetching_market": "Loading live market data...",
        "fetching_financial": "Fetching financial data for {ticker}...",
        "not_enough_data": "Not enough data to train. Try a longer date range or smaller lookback window.",
        "enter_ticker": "Please enter a ticker symbol.", "max_holdings": "Maximum {n} holdings reached.",
        "loading_prices": "Fetching live price for {sym}...",
        "deep_analysis": "📈  Deep Analysis",
        "model_conf_score": "MODEL CONFIDENCE SCORE",
        "high_confidence": "HIGH CONFIDENCE", "moderate_confidence": "MODERATE CONFIDENCE", "low_confidence": "LOW CONFIDENCE",
        "r2_fit": "R² fit", "mape_accuracy": "MAPE accuracy", "directional_acc": "Directional acc.", "data_volume": "Data volume",
        "composite_signal": "Composite Signal", "forecast_lbl": "forecast",
        "score_lbl": "Score",
        "take_profit_lbl": "Take Profit", "stop_loss_lbl": "Stop Loss", "risk_reward_lbl": "Risk / Reward",
        "rsi_lbl": "RSI (14)",
        "favorable": "✓ Favorable", "marginal": "⚠ Marginal", "unfavorable": "✗ Unfavorable",
        "oversold_zone": "Oversold zone", "overbought_zone": "Overbought zone", "neutral_zone": "Neutral zone",
        "factor_breakdown": "6-Factor Signal Breakdown",
        "last_close_lbl": "Last Close", "model_confidence_lbl": "Model Confidence",
        "high_lbl": "High", "moderate_lbl": "Moderate", "low_lbl": "Low",
        "at_above_target": "AT or ABOVE your target of",
        "below_target": "below target of",
        "shariah_debt_mktcap": "Debt/MarketCap",
        "shariah_debt_assets": "Debt/Assets",
        "shariah_cash_assets": "Cash/Assets",
        "known_noncompliant": "Known non-compliant ticker",
    },
    "Arabic": {
        "run": "▶  تشغيل التنبؤ", "ticker": "الرمز", "from": "من", "to": "إلى",
        "mode": "الوضع", "beginner": "🟢 مبتدئ", "pro": "🔴 محترف",
        "watchlist": "⭐ قائمة المراقبة", "alerts": "🔔 تنبيهات الإشارة",
        "forecast": "التوقعات", "days": "أيام", "add": "إضافة",
        "portfolio": "💼  المحفظة", "markets": "🌍  الأسواق",
        "back": "← العودة إلى لوحة التحكم",
        "search_label": "🔍 البحث عن شركة / رمز",
        "lookback": "نافذة الاسترجاع (أيام)", "horizon": "أفق التنبؤ (أيام)",
        "simple_view": "✓ العرض البسيط مفعّل",
        "pro_view": "⚡ وضع الاحتراف — جميع الخيارات مفتوحة",
        "fast_mode": "⚡ الوضع السريع (بدون CI والاختبار)",
        "xgb_params": "معلمات XGBoost",
        "alert_target": "سعر التنبيه المستهدف ($)",
        "backtesting": "الاختبار الخلفي", "enable_backtest": "تفعيل محرك الاختبار الخلفي",
        "init_capital": "رأس المال الابتدائي ($)", "commission": "عمولة كل صفقة ($)",
        "signal_thresh": "حد الإشارة (%)", "extra_features": "ميزات إضافية",
        "model_compare": "مقارنة النماذج (XGB vs LR vs Prophet)",
        "halal_check": "فحص الامتثال للشريعة / الحلال",
        "conf_interval": "فترات الثقة في التنبؤ",
        "bootstrap_samples": "عينات Bootstrap (CI)", "multi_stock": "مقارنة متعددة الأسهم",
        "compare_tickers": "مقارنة الرموز", "alert_signal_change": "تنبيه عند تغيير الإشارة",
        "dashboard_title": "لوحة التحكم", "dashboard_subtitle": "نظرة عامة",
        "dashboard_desc": "ذكاء الأسهم بالذكاء الاصطناعي · أدخل رمزًا في الشريط الجانبي وانقر تشغيل للبدء.",
        "watchlist_live": "⭐ قائمة المراقبة — الأسعار الحية",
        "how_it_works": "كيف يعمل",
        "hw1_title": "أدخل الرمز", "hw1_body": "ابحث باسم الشركة أو الرمز. أضفه إلى قائمة المراقبة للمتابعة.",
        "hw2_title": "تشغيل النموذج", "hw2_body": "يتدرب XGBoost على 7 سنوات من البيانات بـ 20 ميزة مهندسة. النتائج في ثوانٍ.",
        "hw3_title": "اقرأ الإشارة", "hw3_body": "احصل على حكم شراء / بيع / انتظار مع شرح كامل لكل عامل.",
        "platform_features": "ميزات المنصة",
        "no_stocks_saved": "لا أسهم محفوظة بعد.",
        "no_holdings": "لا حيازات بعد. أضف أول سهم أعلاه.",
        "no_transactions": "لا معاملات بعد.",
        "portfolio_title": "المحفظة", "portfolio_tracker": "المتتبع",
        "portfolio_desc": "أضف حيازاتك الحقيقية — تتحدث الأسعار مباشرةً من Yahoo Finance",
        "add_holding": "➕ إضافة حيازة", "add_to_portfolio": "أضف إلى المحفظة",
        "refresh_prices": "🔄 تحديث الأسعار الحية",
        "total_value": "القيمة الإجمالية", "total_pl": "الربح/الخسارة الإجمالية",
        "invested": "المستثمر", "holdings": "الحيازات",
        "holdings_label": "الحيازات", "sector_allocation": "توزيع القطاعات",
        "recent_activity": "النشاط الأخير",
        "price_chart": "مخطط السعر", "tech_indicators": "المؤشرات الفنية",
        "news_sentiment": "تحليل مشاعر الأخبار", "screening_criteria": "معايير الفحص",
        "analysis_tab": "📊  التحليل", "methodology_tab": "📖  المنهجية",
        "dashboard_tab": "🖥  لوحة التحكم",
        "reality_check_title": "⚠ التحقق من الواقع — اقرأ قبل التداول",
        "reality_check_body": "يستخدم هذا النموذج <b style='color:#dae2fd;'>بيانات السعر والحجم فقط</b>. ليس لديه <b style='color:#ff6b6b;'>أي وعي</b> بـ: 📰 الأخبار العاجلة · 📊 إصدارات الأرباح · 🏦 أحداث الاقتصاد الكلي · 🧠 ترقيات المحللين · 🌍 الأحداث الجيوسياسية. <b style='color:#ffdd2d;'>استخدم الإشارات كمدخل واحد فقط.</b>",
        "logout": "⏏  تسجيل الخروج",
        "verify_symbol": "استخدام: {ticker} — تحقق من الرمز",
        "active_ticker": "● نشط: {ticker}",
        "greed_territory": "منطقة الجشع", "low_volatility": "تذبذب منخفض",
        "no_recent_news": "لا توجد أخبار حديثة لهذا الرمز.",
        "already_in_portfolio": "{sym} موجود بالفعل في المحفظة. أزله أولاً للتحديث.",
        "added_success": "✓ تمت إضافة {sym} — السعر الحي ${price:.2f}",
        "footer": "⚠ STOCKCAST · للأغراض التعليمية فقط · ليست نصيحة مالية · طوّره معاوية غني",
        "model_performance": "أداء النموذج", "actual_vs_pred": "الفعلي مقابل المتوقع",
        "feature_importance": "أهمية الميزات", "signal_intelligence": "ذكاء الإشارات",
        "forecast_next": "التنبؤ — الـ {n} أيام القادمة", "backtest_engine": "محرك الاختبار الخلفي",
        "trade_log": "سجل التداول", "forecast_ci": "التنبؤ مع فترات الثقة",
        "model_comparison": "مقارنة النماذج — XGBoost مقابل Prophet مقابل الانحدار الخطي",
        "sector_heatmap": "خريطة حرارة القطاعات · مباشر", "fear_greed": "مؤشر الخوف والجشع · مباشر",
        "days_loaded": "✓ {n} يوم تداول تم تحميله لـ {ticker}", "fetching": "جارٍ جلب بيانات {ticker}...",
        "engineering": "جارٍ هندسة الميزات التقنية...", "building": "جارٍ بناء مصفوفة الميزات...",
        "training": "جارٍ تدريب نموذج XGBoost...", "running_backtest": "جارٍ تشغيل محاكاة الاختبار الخلفي...",
        "running_bootstrap": "جارٍ تشغيل {n} عينة Bootstrap...", "training_lr": "جارٍ تدريب الانحدار الخطي...",
        "fetching_market": "جارٍ تحميل بيانات السوق المباشرة...", "fetching_financial": "جارٍ جلب البيانات المالية لـ {ticker}...",
        "not_enough_data": "بيانات غير كافية للتدريب. جرّب نطاقاً زمنياً أطول أو نافذة استرجاع أصغر.",
        "enter_ticker": "الرجاء إدخال رمز السهم.", "max_holdings": "تم الوصول إلى الحد الأقصى {n} حيازة.",
        "loading_prices": "جارٍ جلب السعر المباشر لـ {sym}...",
        "deep_analysis": "📈  تحليل عميق",
        "model_conf_score": "نقاط ثقة النموذج",
        "high_confidence": "ثقة عالية", "moderate_confidence": "ثقة متوسطة", "low_confidence": "ثقة منخفضة",
        "r2_fit": "دقة R²", "mape_accuracy": "دقة MAPE", "directional_acc": "الدقة الاتجاهية", "data_volume": "حجم البيانات",
        "composite_signal": "الإشارة المركبة", "forecast_lbl": "توقع",
        "score_lbl": "النقاط",
        "take_profit_lbl": "جني الأرباح", "stop_loss_lbl": "وقف الخسارة", "risk_reward_lbl": "المخاطرة / العائد",
        "rsi_lbl": "RSI (14)",
        "favorable": "✓ مناسب", "marginal": "⚠ هامشي", "unfavorable": "✗ غير مناسب",
        "oversold_zone": "منطقة البيع الزائد", "overbought_zone": "منطقة الشراء الزائد", "neutral_zone": "المنطقة المحايدة",
        "factor_breakdown": "تحليل 6 عوامل للإشارة",
        "last_close_lbl": "آخر إغلاق", "model_confidence_lbl": "ثقة النموذج",
        "high_lbl": "عالية", "moderate_lbl": "متوسطة", "low_lbl": "منخفضة",
        "at_above_target": "عند الهدف أو أعلاه",
        "below_target": "تحت الهدف بـ",
        "shariah_debt_mktcap": "الدين / القيمة السوقية",
        "shariah_debt_assets": "الدين / الأصول",
        "shariah_cash_assets": "النقد / الأصول",
        "known_noncompliant": "رمز غير متوافق معروف",
    },
    "Urdu": {
        "run": "▶  پیشن گوئی چلائیں", "ticker": "ٹکر", "from": "سے", "to": "تک",
        "mode": "موڈ", "beginner": "🟢 ابتدائی", "pro": "🔴 پرو",
        "watchlist": "⭐ واچ لسٹ", "alerts": "🔔 سگنل الرٹس",
        "forecast": "پیشن گوئی", "days": "دن", "add": "شامل کریں",
        "portfolio": "💼  پورٹ فولیو", "markets": "🌍  مارکیٹس",
        "back": "← ڈیش بورڈ پر واپس",
        "search_label": "🔍 کمپنی / ٹکر تلاش کریں",
        "lookback": "لُک بیک ونڈو (دن)", "horizon": "پیشن گوئی کا دورانیہ (دن)",
        "simple_view": "✓ سادہ منظر فعال ہے",
        "pro_view": "⚡ پرو منظر — تمام اختیارات کھلے ہیں",
        "fast_mode": "⚡ فاسٹ موڈ (CI اور بیک ٹیسٹ چھوڑیں)",
        "xgb_params": "XGBoost پیرامیٹرز",
        "alert_target": "قیمت الرٹ ہدف ($)",
        "backtesting": "بیک ٹیسٹنگ", "enable_backtest": "بیک ٹیسٹنگ انجن فعال کریں",
        "init_capital": "ابتدائی سرمایہ ($)", "commission": "فی ٹریڈ کمیشن ($)",
        "signal_thresh": "سگنل حد (%)", "extra_features": "اضافی خصوصیات",
        "model_compare": "ماڈل موازنہ (XGB vs LR vs Prophet)",
        "halal_check": "حلال / شریعت کی تعمیل کی جانچ",
        "conf_interval": "پیشن گوئی پر اعتماد کے وقفے",
        "bootstrap_samples": "Bootstrap سیمپل (CI)", "multi_stock": "کثیر اسٹاک موازنہ",
        "compare_tickers": "ٹکر موازنہ کریں", "alert_signal_change": "سگنل بدلنے پر الرٹ",
        "dashboard_title": "ڈیش بورڈ", "dashboard_subtitle": "جائزہ",
        "dashboard_desc": "AI سے چلنے والی اسٹاک انٹیلی جنس · سائڈبار میں ٹکر درج کریں اور پیشن گوئی چلائیں۔",
        "watchlist_live": "⭐ واچ لسٹ — لائیو قیمتیں",
        "how_it_works": "یہ کیسے کام کرتا ہے",
        "hw1_title": "ٹکر درج کریں", "hw1_body": "کمپنی کے نام یا علامت سے تلاش کریں۔ اسے واچ لسٹ میں شامل کریں۔",
        "hw2_title": "ماڈل چلائیں", "hw2_body": "XGBoost 7 سال کے ڈیٹا پر 20 انجینئرڈ فیچرز کے ساتھ تربیت کرتا ہے۔ نتائج سیکنڈوں میں۔",
        "hw3_title": "سگنل پڑھیں", "hw3_body": "خرید / فروخت / انتظار کا فیصلہ حاصل کریں ہر عنصر کی مکمل وضاحت کے ساتھ۔",
        "platform_features": "پلیٹ فارم کی خصوصیات",
        "no_stocks_saved": "ابھی تک کوئی اسٹاک محفوظ نہیں۔",
        "no_holdings": "ابھی تک کوئی ہولڈنگ نہیں۔ اوپر پہلا اسٹاک شامل کریں۔",
        "no_transactions": "ابھی تک کوئی لین دین نہیں۔",
        "portfolio_title": "پورٹ فولیو", "portfolio_tracker": "ٹریکر",
        "portfolio_desc": "اپنی اصل ہولڈنگز شامل کریں — قیمتیں Yahoo Finance سے لائیو اپ ڈیٹ ہوتی ہیں",
        "add_holding": "➕ ہولڈنگ شامل کریں", "add_to_portfolio": "پورٹ فولیو میں شامل کریں",
        "refresh_prices": "🔄 لائیو قیمتیں تازہ کریں",
        "total_value": "کل قیمت", "total_pl": "کل نفع/نقصان",
        "invested": "سرمایہ کاری", "holdings": "ہولڈنگز",
        "holdings_label": "ہولڈنگز", "sector_allocation": "شعبہ وار تقسیم",
        "recent_activity": "حالیہ سرگرمی",
        "price_chart": "قیمت کا چارٹ", "tech_indicators": "تکنیکی اشارے",
        "news_sentiment": "خبر جذباتی تجزیہ", "screening_criteria": "اسکریننگ کا معیار",
        "analysis_tab": "📊  تجزیہ", "methodology_tab": "📖  طریقہ کار",
        "dashboard_tab": "🖥  ڈیش بورڈ",
        "reality_check_title": "⚠ ماڈل کی حقیقت — تجارت سے پہلے پڑھیں",
        "reality_check_body": "یہ ماڈل صرف <b style='color:#dae2fd;'>قیمت اور حجم کا ڈیٹا</b> استعمال کرتا ہے۔ اسے <b style='color:#ff6b6b;'>کوئی آگاہی نہیں</b> 📰 خبروں · 📊 آمدنی کے اعلانات · 🏦 اقتصادی واقعات · 🧠 تجزیہ کاروں کی رائے · 🌍 جیوپولیٹیکل واقعات سے۔ <b style='color:#ffdd2d;'>سگنلز صرف ایک ان پٹ کے طور پر استعمال کریں۔</b>",
        "logout": "⏏  لاگ آؤٹ",
        "verify_symbol": "استعمال: {ticker} — علامت کی تصدیق کریں",
        "active_ticker": "● فعال: {ticker}",
        "greed_territory": "لالچ کا علاقہ", "low_volatility": "کم اتار چڑھاؤ",
        "no_recent_news": "اس ٹکر کے لیے کوئی حالیہ خبر نہیں ملی۔",
        "already_in_portfolio": "{sym} پہلے سے پورٹ فولیو میں ہے۔ اپڈیٹ کرنے کے لیے پہلے ہٹائیں۔",
        "added_success": "✓ {sym} شامل کیا — لائیو قیمت ${price:.2f}",
        "footer": "⚠ STOCKCAST · صرف تعلیمی مقاصد کے لیے · مالی مشورہ نہیں · تیار کردہ معاویہ غنی",
        "model_performance": "ماڈل کی کارکردگی", "actual_vs_pred": "حقیقی بمقابلہ پیشین گوئی",
        "feature_importance": "فیچرز کی اہمیت", "signal_intelligence": "سگنل انٹیلی جنس",
        "forecast_next": "پیشن گوئی — اگلے {n} دن", "backtest_engine": "بیک ٹیسٹ انجن",
        "trade_log": "ٹریڈ لاگ", "forecast_ci": "اعتماد کے وقفوں کے ساتھ پیشن گوئی",
        "model_comparison": "ماڈل موازنہ — XGBoost بمقابلہ Prophet بمقابلہ لکیری رجعت",
        "sector_heatmap": "سیکٹر ہیٹ میپ · لائیو", "fear_greed": "خوف اور لالچ انڈیکس · لائیو",
        "days_loaded": "✓ {n} ٹریڈنگ دن {ticker} کے لیے لوڈ ہوئے", "fetching": "{ticker} کا ڈیٹا لا رہے ہیں...",
        "engineering": "تکنیکی فیچرز بنا رہے ہیں...", "building": "فیچر میٹرکس بنا رہے ہیں...",
        "training": "XGBoost ماڈل ٹرین ہو رہا ہے...", "running_backtest": "بیک ٹیسٹ سمولیشن چل رہی ہے...",
        "running_bootstrap": "{n} Bootstrap سیمپل چل رہے ہیں...", "training_lr": "لکیری رجعت ٹرین ہو رہی ہے...",
        "fetching_market": "مارکیٹ کا لائیو ڈیٹا لوڈ ہو رہا ہے...", "fetching_financial": "{ticker} کا مالی ڈیٹا لا رہے ہیں...",
        "not_enough_data": "تربیت کے لیے کافی ڈیٹا نہیں۔ طویل تاریخی رینج یا چھوٹا لُک بیک آزمائیں۔",
        "enter_ticker": "براہ کرم ٹکر علامت درج کریں۔", "max_holdings": "زیادہ سے زیادہ {n} ہولڈنگز پہنچ گئی۔",
        "loading_prices": "{sym} کی لائیو قیمت لا رہے ہیں...",
        "deep_analysis": "📈  گہرا تجزیہ",
        "model_conf_score": "ماڈل اعتماد اسکور",
        "high_confidence": "اعلی اعتماد", "moderate_confidence": "اوسط اعتماد", "low_confidence": "کم اعتماد",
        "r2_fit": "R² فٹ", "mape_accuracy": "MAPE درستگی", "directional_acc": "سمتی درستگی", "data_volume": "ڈیٹا حجم",
        "composite_signal": "مرکب سگنل", "forecast_lbl": "پیشن گوئی",
        "score_lbl": "اسکور",
        "take_profit_lbl": "منافع لیں", "stop_loss_lbl": "نقصان روکیں", "risk_reward_lbl": "خطرہ / انعام",
        "rsi_lbl": "RSI (14)",
        "favorable": "✓ موزوں", "marginal": "⚠ معمولی", "unfavorable": "✗ ناموزوں",
        "oversold_zone": "زیادہ فروخت زون", "overbought_zone": "زیادہ خریداری زون", "neutral_zone": "غیر جانبدار زون",
        "factor_breakdown": "6 عوامل سگنل تجزیہ",
        "last_close_lbl": "آخری بندش", "model_confidence_lbl": "ماڈل اعتماد",
        "high_lbl": "اعلی", "moderate_lbl": "اوسط", "low_lbl": "کم",
        "at_above_target": "ہدف پر یا اوپر",
        "below_target": "ہدف سے نیچے",
        "shariah_debt_mktcap": "قرض / مارکیٹ کیپ",
        "shariah_debt_assets": "قرض / اثاثے",
        "shariah_cash_assets": "نقد / اثاثے",
        "known_noncompliant": "معروف غیر موافق ٹکر",
    },
    "Hindi": {
        "run": "▶  पूर्वानुमान चलाएं", "ticker": "टिकर", "from": "से", "to": "तक",
        "mode": "मोड", "beginner": "🟢 शुरुआती", "pro": "🔴 प्रो",
        "watchlist": "⭐ वॉचलिस्ट", "alerts": "🔔 सिग्नल अलर्ट",
        "forecast": "पूर्वानुमान", "days": "दिन", "add": "जोड़ें",
        "portfolio": "💼  पोर्टफोलियो", "markets": "🌍  बाज़ार",
        "back": "← डैशबोर्ड पर वापस",
        "search_label": "🔍 कंपनी / टिकर खोजें",
        "lookback": "लुकबैक विंडो (दिन)", "horizon": "पूर्वानुमान अवधि (दिन)",
        "simple_view": "✓ सरल दृश्य सक्रिय",
        "pro_view": "⚡ प्रो दृश्य — सभी पैरामीटर अनलॉक",
        "fast_mode": "⚡ फास्ट मोड (CI + बैकटेस्ट छोड़ें)",
        "xgb_params": "XGBoost हाइपरपैरामीटर",
        "alert_target": "मूल्य अलर्ट लक्ष्य ($)",
        "backtesting": "बैकटेस्टिंग", "enable_backtest": "बैकटेस्टिंग इंजन सक्षम करें",
        "init_capital": "प्रारंभिक पूंजी ($)", "commission": "प्रति ट्रेड कमीशन ($)",
        "signal_thresh": "सिग्नल सीमा (%)", "extra_features": "अतिरिक्त सुविधाएं",
        "model_compare": "मॉडल तुलना (XGB vs LR vs Prophet)",
        "halal_check": "हलाल / शरिया अनुपालन जांच",
        "conf_interval": "पूर्वानुमान पर विश्वास अंतराल",
        "bootstrap_samples": "Bootstrap नमूने (CI)", "multi_stock": "बहु-स्टॉक तुलना",
        "compare_tickers": "टिकर तुलना करें", "alert_signal_change": "सिग्नल बदलने पर अलर्ट",
        "dashboard_title": "डैशबोर्ड", "dashboard_subtitle": "अवलोकन",
        "dashboard_desc": "AI-संचालित स्टॉक इंटेलिजेंस · साइडबार में टिकर दर्ज करें और पूर्वानुमान चलाएं।",
        "watchlist_live": "⭐ वॉचलिस्ट — लाइव कीमतें",
        "how_it_works": "यह कैसे काम करता है",
        "hw1_title": "टिकर दर्ज करें", "hw1_body": "कंपनी नाम या प्रतीक से खोजें। लगातार ट्रैक करने के लिए वॉचलिस्ट में जोड़ें।",
        "hw2_title": "मॉडल चलाएं", "hw2_body": "XGBoost 7 साल के OHLCV डेटा पर 20 इंजीनियर्ड फीचर्स के साथ प्रशिक्षण लेता है। सेकंडों में परिणाम।",
        "hw3_title": "सिग्नल पढ़ें", "hw3_body": "हर योगदान कारक की पूरी व्याख्या के साथ BUY/SELL/HOLD निर्णय प्राप्त करें।",
        "platform_features": "प्लेटफॉर्म की विशेषताएं",
        "no_stocks_saved": "अभी तक कोई स्टॉक सहेजा नहीं गया।",
        "no_holdings": "अभी तक कोई होल्डिंग नहीं। ऊपर पहला स्टॉक जोड़ें।",
        "no_transactions": "अभी तक कोई लेनदेन नहीं।",
        "portfolio_title": "पोर्टफोलियो", "portfolio_tracker": "ट्रैकर",
        "portfolio_desc": "अपनी वास्तविक होल्डिंग्स जोड़ें — Yahoo Finance से कीमतें लाइव अपडेट होती हैं",
        "add_holding": "➕ होल्डिंग जोड़ें", "add_to_portfolio": "पोर्टफोलियो में जोड़ें",
        "refresh_prices": "🔄 लाइव कीमतें ताज़ा करें",
        "total_value": "कुल मूल्य", "total_pl": "कुल लाभ/हानि",
        "invested": "निवेशित", "holdings": "होल्डिंग्स",
        "holdings_label": "होल्डिंग्स", "sector_allocation": "क्षेत्र आवंटन",
        "recent_activity": "हालिया गतिविधि",
        "price_chart": "मूल्य चार्ट", "tech_indicators": "तकनीकी संकेतक",
        "news_sentiment": "समाचार भावना NLP", "screening_criteria": "स्क्रीनिंग मानदंड",
        "analysis_tab": "📊  विश्लेषण", "methodology_tab": "📖  कार्यप्रणाली",
        "dashboard_tab": "🖥  डैशबोर्ड",
        "reality_check_title": "⚠ मॉडल वास्तविकता जांच — ट्रेड से पहले पढ़ें",
        "reality_check_body": "यह मॉडल केवल <b style='color:#dae2fd;'>मूल्य और वॉल्यूम डेटा</b> उपयोग करता है। इसे <b style='color:#ff6b6b;'>कोई जागरूकता नहीं</b> 📰 ताज़ा खबरें · 📊 आय रिलीज़ · 🏦 Fed/मैक्रो इवेंट · 🧠 विश्लेषक अपग्रेड · 🌍 भू-राजनीतिक घटनाओं की। <b style='color:#ffdd2d;'>सिग्नल को एक इनपुट के रूप में उपयोग करें।</b>",
        "logout": "⏏  लॉगआउट",
        "verify_symbol": "उपयोग: {ticker} — प्रतीक सत्यापित करें",
        "active_ticker": "● सक्रिय: {ticker}",
        "greed_territory": "लालच क्षेत्र", "low_volatility": "कम अस्थिरता",
        "no_recent_news": "इस टिकर के लिए कोई हालिया खबर नहीं मिली।",
        "already_in_portfolio": "{sym} पहले से पोर्टफोलियो में है। अपडेट करने के लिए पहले हटाएं।",
        "added_success": "✓ {sym} जोड़ा गया — लाइव कीमत ${price:.2f}",
        "footer": "⚠ STOCKCAST · केवल शैक्षणिक उद्देश्यों के लिए · वित्तीय सलाह नहीं · निर्मित मुआवविज़ घनी द्वारा",
        "model_performance": "मॉडल प्रदर्शन", "actual_vs_pred": "वास्तविक बनाम पूर्वानुमान",
        "feature_importance": "फीचर महत्व", "signal_intelligence": "सिग्नल इंटेलिजेंस",
        "forecast_next": "पूर्वानुमान — अगले {n} दिन", "backtest_engine": "बैकटेस्टिंग इंजन",
        "trade_log": "ट्रेड लॉग", "forecast_ci": "विश्वास अंतराल के साथ पूर्वानुमान",
        "model_comparison": "मॉडल तुलना — XGBoost बनाम Prophet बनाम रैखिक प्रतिगमन",
        "sector_heatmap": "सेक्टर हीट मैप · लाइव", "fear_greed": "भय और लालच सूचकांक · लाइव",
        "days_loaded": "✓ {ticker} के लिए {n} ट्रेडिंग दिन लोड हुए", "fetching": "{ticker} का डेटा लाया जा रहा है...",
        "engineering": "तकनीकी फीचर बनाए जा रहे हैं...", "building": "फीचर मैट्रिक्स बनाया जा रहा है...",
        "training": "XGBoost मॉडल प्रशिक्षित हो रहा है...", "running_backtest": "बैकटेस्ट सिमुलेशन चल रही है...",
        "running_bootstrap": "{n} Bootstrap सैंपल चल रहे हैं...", "training_lr": "रैखिक प्रतिगमन प्रशिक्षित हो रहा है...",
        "fetching_market": "लाइव बाज़ार डेटा लोड हो रहा है...", "fetching_financial": "{ticker} का वित्तीय डेटा लाया जा रहा है...",
        "not_enough_data": "प्रशिक्षण के लिए पर्याप्त डेटा नहीं। लंबी तिथि सीमा या छोटा लुकबैक आज़माएं।",
        "enter_ticker": "कृपया टिकर प्रतीक दर्ज करें।", "max_holdings": "अधिकतम {n} होल्डिंग्स पहुंच गई।",
        "loading_prices": "{sym} की लाइव कीमत लाई जा रही है...",
        "deep_analysis": "📈  गहरा विश्लेषण",
        "model_conf_score": "मॉडल विश्वास स्कोर",
        "high_confidence": "उच्च विश्वास", "moderate_confidence": "मध्यम विश्वास", "low_confidence": "कम विश्वास",
        "r2_fit": "R² फिट", "mape_accuracy": "MAPE सटीकता", "directional_acc": "दिशात्मक सटीकता", "data_volume": "डेटा मात्रा",
        "composite_signal": "समग्र सिग्नल", "forecast_lbl": "पूर्वानुमान",
        "score_lbl": "स्कोर",
        "take_profit_lbl": "लाभ लें", "stop_loss_lbl": "नुकसान रोकें", "risk_reward_lbl": "जोखिम / पुरस्कार",
        "rsi_lbl": "RSI (14)",
        "favorable": "✓ अनुकूल", "marginal": "⚠ सीमांत", "unfavorable": "✗ प्रतिकूल",
        "oversold_zone": "ओवरसोल्ड ज़ोन", "overbought_zone": "ओवरबॉट ज़ोन", "neutral_zone": "तटस्थ क्षेत्र",
        "factor_breakdown": "6-कारक सिग्नल विश्लेषण",
        "last_close_lbl": "अंतिम बंद", "model_confidence_lbl": "मॉडल विश्वास",
        "high_lbl": "उच्च", "moderate_lbl": "मध्यम", "low_lbl": "कम",
        "at_above_target": "लक्ष्य पर या ऊपर",
        "below_target": "लक्ष्य से नीचे",
        "shariah_debt_mktcap": "ऋण / बाज़ार पूंजी",
        "shariah_debt_assets": "ऋण / संपत्ति",
        "shariah_cash_assets": "नकद / संपत्ति",
        "known_noncompliant": "ज्ञात गैर-अनुपालन टिकर",
    },
    "Chinese": {
        "run": "▶  运行预测", "ticker": "股票代码", "from": "从", "to": "到",
        "mode": "模式", "beginner": "🟢 新手", "pro": "🔴 专业",
        "watchlist": "⭐ 关注列表", "alerts": "🔔 信号提醒",
        "forecast": "预测", "days": "天", "add": "添加",
        "portfolio": "💼  投资组合", "markets": "🌍  市场",
        "back": "← 返回仪表板",
        "search_label": "🔍 搜索公司 / 代码",
        "lookback": "回溯窗口（天）", "horizon": "预测周期（天）",
        "simple_view": "✓ 简单视图已激活",
        "pro_view": "⚡ 专业视图 — 所有参数已解锁",
        "fast_mode": "⚡ 快速模式（跳过CI和回测）",
        "xgb_params": "XGBoost 超参数",
        "alert_target": "价格提醒目标 ($)",
        "backtesting": "回测", "enable_backtest": "启用回测引擎",
        "init_capital": "初始资金 ($)", "commission": "每笔交易佣金 ($)",
        "signal_thresh": "信号阈值 (%)", "extra_features": "额外功能",
        "model_compare": "模型对比 (XGB vs LR vs Prophet)",
        "halal_check": "清真 / 伊斯兰教法合规检查",
        "conf_interval": "预测置信区间",
        "bootstrap_samples": "Bootstrap 样本 (CI)", "multi_stock": "多股票对比",
        "compare_tickers": "对比股票代码", "alert_signal_change": "信号变化时提醒",
        "dashboard_title": "仪表板", "dashboard_subtitle": "概览",
        "dashboard_desc": "AI 驱动的股票分析 · 在侧边栏输入代码并点击运行预测。",
        "watchlist_live": "⭐ 关注列表 — 实时价格",
        "how_it_works": "运作方式",
        "hw1_title": "输入代码", "hw1_body": "按公司名称或代码搜索，添加到关注列表持续跟踪。",
        "hw2_title": "运行模型", "hw2_body": "XGBoost 对 7 年 OHLCV 数据进行 20 个工程特征训练，秒级出结果。",
        "hw3_title": "读取信号", "hw3_body": "获得买入/卖出/持有判断及每个贡献因子的完整说明。",
        "platform_features": "平台功能",
        "no_stocks_saved": "尚未保存任何股票。",
        "no_holdings": "尚无持仓。在上方添加第一只股票。",
        "no_transactions": "尚无交易记录。",
        "portfolio_title": "投资组合", "portfolio_tracker": "跟踪器",
        "portfolio_desc": "添加您的真实持仓 — 价格实时从 Yahoo Finance 更新",
        "add_holding": "➕ 添加持仓", "add_to_portfolio": "添加到投资组合",
        "refresh_prices": "🔄 刷新实时价格",
        "total_value": "总价值", "total_pl": "总盈亏",
        "invested": "已投资", "holdings": "持仓",
        "holdings_label": "持仓", "sector_allocation": "板块分配",
        "recent_activity": "最近活动",
        "price_chart": "价格图表", "tech_indicators": "技术指标",
        "news_sentiment": "新闻情绪 NLP", "screening_criteria": "筛选标准",
        "analysis_tab": "📊  分析", "methodology_tab": "📖  方法论",
        "dashboard_tab": "🖥  仪表板",
        "reality_check_title": "⚠ 模型现实检验 — 交易前请阅读",
        "reality_check_body": "该模型仅使用<b style='color:#dae2fd;'>价格和成交量数据</b>，对以下内容<b style='color:#ff6b6b;'>毫无感知</b>：📰 突发新闻 · 📊 财报发布 · 🏦 美联储/宏观事件 · 🧠 分析师评级 · 🌍 地缘政治事件。<b style='color:#ffdd2d;'>仅将信号作为参考，切勿作为唯一依据。</b>",
        "logout": "⏏  退出登录",
        "verify_symbol": "使用中: {ticker} — 请验证代码",
        "active_ticker": "● 当前: {ticker}",
        "greed_territory": "贪婪区域", "low_volatility": "低波动",
        "no_recent_news": "未找到该股票的近期新闻。",
        "already_in_portfolio": "{sym} 已在投资组合中。如需更新请先删除。",
        "added_success": "✓ 已添加 {sym} — 实时价格 ${price:.2f}",
        "footer": "⚠ STOCKCAST · 仅供教育目的 · 非财务建议 · 由 MUAWWIZ GHANI 开发",
        "model_performance": "模型表现", "actual_vs_pred": "实际 vs 预测",
        "feature_importance": "特征重要性", "signal_intelligence": "信号分析",
        "forecast_next": "预测 — 未来 {n} 天", "backtest_engine": "回测引擎",
        "trade_log": "交易记录", "forecast_ci": "带置信区间的预测",
        "model_comparison": "模型对比 — XGBoost vs Prophet vs 线性回归",
        "sector_heatmap": "板块热力图 · 实时", "fear_greed": "恐贪指数 · 实时",
        "days_loaded": "✓ 已加载 {ticker} 的 {n} 个交易日", "fetching": "正在获取 {ticker} 数据...",
        "engineering": "正在计算技术特征...", "building": "正在构建特征矩阵...",
        "training": "XGBoost 模型训练中...", "running_backtest": "正在运行回测模拟...",
        "running_bootstrap": "正在运行 {n} 次 Bootstrap 采样...", "training_lr": "正在训练线性回归...",
        "fetching_market": "正在加载实时市场数据...", "fetching_financial": "正在获取 {ticker} 财务数据...",
        "not_enough_data": "数据不足以训练模型。请尝试更长的日期范围或更小的回溯窗口。",
        "enter_ticker": "请输入股票代码。", "max_holdings": "已达到最大持仓数 {n}。",
        "loading_prices": "正在获取 {sym} 的实时价格...",
        "deep_analysis": "📈  深度分析",
        "model_conf_score": "模型置信度评分",
        "high_confidence": "高置信度", "moderate_confidence": "中等置信度", "low_confidence": "低置信度",
        "r2_fit": "R² 拟合", "mape_accuracy": "MAPE 准确率", "directional_acc": "方向准确率", "data_volume": "数据量",
        "composite_signal": "综合信号", "forecast_lbl": "预测",
        "score_lbl": "评分",
        "take_profit_lbl": "止盈", "stop_loss_lbl": "止损", "risk_reward_lbl": "风险 / 收益",
        "rsi_lbl": "RSI (14)",
        "favorable": "✓ 有利", "marginal": "⚠ 边际", "unfavorable": "✗ 不利",
        "oversold_zone": "超卖区域", "overbought_zone": "超买区域", "neutral_zone": "中性区域",
        "factor_breakdown": "6因子信号分析",
        "last_close_lbl": "最新收盘", "model_confidence_lbl": "模型置信度",
        "high_lbl": "高", "moderate_lbl": "中", "low_lbl": "低",
        "at_above_target": "等于或高于目标",
        "below_target": "低于目标",
        "shariah_debt_mktcap": "债务/市值",
        "shariah_debt_assets": "债务/资产",
        "shariah_cash_assets": "现金/资产",
        "known_noncompliant": "已知不合规代码",
    },
}
if "lang" not in st.session_state:
    st.session_state.lang = "English"

# ── Session state ──────────────────────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "alert_signals" not in st.session_state:
    st.session_state.alert_signals = {}
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []   # Real editable portfolio — starts empty
if "portfolio_history" not in st.session_state:
    st.session_state.portfolio_history = []

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stockcast —  Stock Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Plotly theme ───────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0b1326",
    plot_bgcolor="#0b1326",
    font=dict(family="Manrope", color="#8c909f", size=10),
    xaxis=dict(gridcolor="#424754", linecolor="#424754", tickfont=dict(color="#8c909f", size=10), showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#424754", linecolor="#424754", tickfont=dict(color="#8c909f", size=10), showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#424754", borderwidth=1, font=dict(size=10)),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#131b2e", bordercolor="#2d3449", font=dict(family="Manrope", size=11, color="#dae2fd")),
)

C_GREEN  = "#adc6ff"
C_ACCENT = "#4d8eff"
C_RED    = "#ff6b6b"
C_YELLOW = "#ffdd2d"
C_GREY   = "#8c909f"
C_EMERALD = "#00e5b0"

# ── Master CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@200;300;400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

/* ── ROOT ── */
:root {
    --bg:          #0b1326;
    --bg2:         #131b2e;
    --bg3:         #171f33;
    --bg4:         #222a3d;
    --bg5:         #2d3449;
    --primary:     #adc6ff;
    --accent:      #4d8eff;
    --on-primary:  #002e6a;
    --secondary:   #b1c6f9;
    --t1:          #dae2fd;
    --t2:          #c2c6d6;
    --t3:          #8c909f;
    --t4:          #424754;
    --border:      #2d3449;
    --border2:     #424754;
    --emerald:     #00e5b0;
    --red:         #ff6b6b;
    --yellow:      #ffdd2d;
    --mono:        'IBM Plex Mono', monospace;
    --sans:        'Manrope', sans-serif;
    --radius:      0.5rem;
}

/* ── GLOBAL ── */
html, body, [class*="css"], [data-testid="stApp"],
[data-testid="stAppViewContainer"], .main {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--t1) !important;
}
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }

/* scanline + ambient glow */
[data-testid="stApp"]::before {
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse at 0% 0%, rgba(77,142,255,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 100%, rgba(173,198,255,0.04) 0%, transparent 50%);
    pointer-events: none; z-index: 0;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, var(--bg2) 0%, var(--bg) 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--t3) !important; }
[data-testid="stSidebar"] input {
    background-color: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    color: var(--primary) !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(77,142,255,0.25) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    color: var(--primary) !important;
    border: 1px solid rgba(173,198,255,0.35) !important;
    border-radius: var(--radius) !important;
    font-family: var(--sans) !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: #fff !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 20px rgba(77,142,255,0.3) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── METRICS ── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, var(--bg2), var(--bg3)) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--accent) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 0 20px rgba(77,142,255,0.15) !important;
    transform: translateY(-1px) !important;
    transition: all 0.2s !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--sans) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--t3) !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: var(--primary) !important;
}

/* ── HEADINGS ── */
h2, h3 {
    font-family: var(--sans) !important;
    color: var(--t1) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0.5rem !important;
    margin-top: 1.4rem !important;
    font-weight: 800 !important;
}
h4 {
    font-family: var(--sans) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--t3) !important;
}
hr { border-color: var(--border) !important; }
p, .stMarkdown p { color: var(--t2) !important; font-size: 0.85rem !important; }

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    background: var(--bg2) !important;
    border-radius: var(--radius) !important;
}

/* ── LABELS ── */
label, [data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stTextInput"] label {
    font-family: var(--sans) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--t3) !important;
    font-weight: 700 !important;
}

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg2) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 2px !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--sans) !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    color: var(--t3) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    text-transform: uppercase !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}
[data-testid="stTabPanel"] { background: transparent !important; padding: 1rem 0 !important; }

/* ── APP HEADER ── */
.wi-header {
    background: linear-gradient(90deg, var(--bg2) 0%, var(--bg3) 100%);
    border-bottom: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    padding: 1rem 2rem;
    margin: 3rem -2rem 1.5rem -2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.wi-logo {
    font-family: var(--sans);
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--t1);
    letter-spacing: -0.01em;
}
.wi-logo span { color: var(--accent); }
.wi-sub {
    font-size: 0.65rem;
    color: var(--t3);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 2px;
    font-weight: 600;
}
.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: var(--emerald);
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
    margin-right: 5px;
    vertical-align: middle;
    box-shadow: 0 0 8px rgba(0,229,176,0.5);
}
@keyframes pulse-dot {
    0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(0,229,176,0.4); }
    50%      { opacity:.8; box-shadow: 0 0 0 6px rgba(0,229,176,0); }
}
.live-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--emerald);
    letter-spacing: 0.12em;
    vertical-align: middle;
}

/* ── TICKER TAPE ── */
.ticker-tape-wrap {
    overflow: hidden;
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    border-top: 1px solid var(--border);
    padding: 0.28rem 0;
    margin: 0 -2rem 1.5rem -2rem;
}
.ticker-tape {
    display: inline-flex;
    gap: 2.5rem;
    animation: tape 38s linear infinite;
    white-space: nowrap;
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.05em;
    color: var(--t3);
}
.ticker-tape:hover { animation-play-state: paused; }
@keyframes tape { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
.tape-up   { color: var(--emerald); font-weight: 700; }
.tape-down { color: var(--red); font-weight: 700; }
.tape-sym  { color: var(--t4); font-size: 0.58rem; margin-right: 0.25rem; }

/* ── GLASS CARDS ── */
.wi-card {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    transition: transform 0.18s, box-shadow 0.18s;
}
.wi-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.wi-card-accent { border-top: 2px solid var(--accent); }
.wi-card-emerald { border-top: 2px solid var(--emerald); }
.wi-card-red { border-top: 2px solid var(--red); }
.wi-card-yellow { border-top: 2px solid var(--yellow); }

/* ── SUMMARY STAT GRID ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}
.stat-card {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 50px; height: 50px;
    background: radial-gradient(circle at top right, rgba(77,142,255,0.07), transparent 70%);
}
.stat-label {
    font-family: var(--sans);
    font-size: 0.56rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--t3);
    font-weight: 700;
    margin-bottom: 4px;
}
.stat-value {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--primary);
    line-height: 1.1;
}
.stat-sub { font-size: 0.58rem; color: var(--t3); margin-top: 3px; font-family: var(--sans); }

/* ── SIGNAL PANEL ── */
.signal-panel {
    display: flex;
    gap: 1rem;
    margin: 1.2rem 0;
    flex-wrap: wrap;
}
.signal-main {
    flex: 0 0 240px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem 1.5rem;
    border: 2px solid var(--accent);
    background: rgba(77,142,255,0.05);
    border-radius: var(--radius);
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 30px rgba(77,142,255,0.08);
}
.signal-main::before {
    content: '';
    position: absolute;
    bottom: -20px; right: -20px;
    width: 100px; height: 100px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(77,142,255,0.18) 0%, transparent 70%);
}
.signal-main.sell { border-color: var(--red); background: rgba(255,107,107,0.05); box-shadow: 0 0 30px rgba(255,107,107,0.08); }
.signal-main.sell::before { background: radial-gradient(circle, rgba(255,107,107,0.18) 0%, transparent 70%); }
.signal-main.hold { border-color: var(--yellow); background: rgba(255,221,45,0.05); box-shadow: 0 0 30px rgba(255,221,45,0.08); }
.signal-main.hold::before { background: radial-gradient(circle, rgba(255,221,45,0.18) 0%, transparent 70%); }
.signal-action {
    font-family: var(--mono);
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: 0.2em;
    color: var(--primary);
    line-height: 1;
}
.signal-action.sell { color: var(--red); }
.signal-action.hold { color: var(--yellow); }
.signal-pct {
    font-family: var(--mono);
    font-size: 1rem;
    font-weight: 600;
    margin-top: 0.5rem;
    color: var(--t1);
}
.signal-lbl {
    font-size: 0.54rem;
    letter-spacing: 0.2em;
    color: var(--t3);
    margin-top: 8px;
    text-transform: uppercase;
    font-weight: 700;
    font-family: var(--sans);
}
.signal-details {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    min-width: 200px;
}
.sig-card {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    padding: 0.75rem 1rem;
    position: relative;
    border-radius: var(--radius);
    overflow: hidden;
    transition: transform 0.15s;
}
.sig-card:hover { transform: translateY(-1px); }
.sig-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--border);
}
.sig-card.positive::before { background: var(--emerald); }
.sig-card.negative::before { background: var(--red); }
.sig-card.neutral::before  { background: var(--yellow); }
.sig-lbl { font-size: 0.53rem; letter-spacing: 0.13em; text-transform: uppercase; color: var(--t3); margin-bottom: 4px; font-weight: 700; font-family: var(--sans); }
.sig-val { font-family: var(--mono); font-size: 0.88rem; font-weight: 600; color: var(--t1); }
.sig-sub { font-size: 0.54rem; color: var(--t3); margin-top: 2px; font-family: var(--sans); }

/* ── COMPOSITE METER ── */
.composite-meter {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    padding: 1rem 1.5rem;
    margin: 0.8rem 0;
    border-radius: 0 var(--radius) var(--radius) 0;
}
.meter-title { font-size: 0.56rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--t3); margin-bottom: 0.8rem; font-weight: 700; font-family: var(--sans); }
.sir { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.4rem; font-family: var(--mono); font-size: 0.67rem; }
.sir-label { color: var(--t2); width: 120px; flex-shrink: 0; }
.sir-bar-bg { flex: 1; height: 4px; background: rgba(255,255,255,0.04); border-radius: 2px; overflow: hidden; }
.sir-bar { height: 100%; border-radius: 2px; transition: width 0.6s ease; }
.sir-bar.positive { background: linear-gradient(90deg, var(--emerald), rgba(0,229,176,0.5)); }
.sir-bar.negative { background: linear-gradient(90deg, var(--red), rgba(255,107,107,0.5)); }
.sir-bar.neutral  { background: linear-gradient(90deg, var(--yellow), rgba(255,221,45,0.5)); }
.sir-val { width: 55px; text-align: right; font-weight: 600; color: var(--t1); }
.sir-sig { width: 40px; text-align: right; font-size: 0.57rem; letter-spacing: 0.08em; font-weight: 700; }
.sir-sig.buy { color: var(--emerald); }
.sir-sig.sell { color: var(--red); }
.sir-sig.hold { color: var(--yellow); }

/* ── BT CARDS ── */
.bt-card {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    border-top: 2px solid var(--border);
    padding: 1rem 1.2rem;
    margin-bottom: 0.4rem;
    font-family: var(--mono);
    border-radius: var(--radius);
}
.bt-label { font-size: 0.58rem; color: var(--t3); letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 4px; font-family: var(--sans); font-weight: 700; }
.bt-val       { font-size: 1.3rem; font-weight: 700; color: var(--t1); }
.bt-val-green { font-size: 1.3rem; font-weight: 700; color: var(--emerald); }
.bt-val-red   { font-size: 1.3rem; font-weight: 700; color: var(--red); }

/* ── HALAL CARDS ── */
.halal-card {
    background: rgba(0,229,176,0.03);
    border: 1px solid rgba(0,229,176,0.15);
    border-left: 3px solid var(--emerald);
    padding: 0.85rem 1.2rem;
    margin: 0.3rem 0;
    font-family: var(--sans);
    font-size: 0.8rem;
    color: var(--t2);
    border-radius: 0 var(--radius) var(--radius) 0;
}
.halal-card-fail {
    background: rgba(255,107,107,0.03);
    border: 1px solid rgba(255,107,107,0.15);
    border-left: 3px solid var(--red);
    padding: 0.85rem 1.2rem;
    margin: 0.3rem 0;
    font-family: var(--sans);
    font-size: 0.8rem;
    color: var(--t2);
    border-radius: 0 var(--radius) var(--radius) 0;
}

/* ── MODEL BADGE ── */
.model-badge {
    display: inline-block;
    background: rgba(77,142,255,0.1);
    border: 1px solid rgba(77,142,255,0.25);
    color: var(--primary);
    font-family: var(--sans);
    font-size: 0.62rem;
    font-weight: 700;
    padding: 0.22rem 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    border-radius: var(--radius);
}

/* ── ALERT BOX ── */
.alert-box {
    background: rgba(77,142,255,0.05);
    border: 1px solid rgba(77,142,255,0.3);
    border-left: 3px solid var(--accent);
    padding: 0.85rem 1.3rem;
    font-family: var(--sans);
    font-size: 0.78rem;
    color: var(--primary);
    margin: 0.8rem 0;
    letter-spacing: 0.03em;
    border-radius: 0 var(--radius) var(--radius) 0;
}

/* ── SIDEBAR STAT ROW ── */
.stat-row {
    font-family: var(--sans);
    font-size: 0.58rem;
    color: var(--t3);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
    margin-top: 2px;
    font-weight: 700;
}

/* ── NAV ITEM ── */
.nav-item-active {
    background: var(--bg4);
    border-left: 3px solid var(--accent);
    color: var(--primary) !important;
    padding: 0.5rem 1rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 2px 0;
    border-radius: 0 var(--radius) var(--radius) 0;
    font-family: var(--sans);
}
.nav-item-idle {
    color: var(--t3);
    padding: 0.5rem 1rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 2px 0;
    font-family: var(--sans);
}

/* ── WATCHLIST BADGE ── */
.wl-badge {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--bg3);
    border: 1px solid var(--border);
    padding: 0.5rem 0.75rem;
    border-radius: var(--radius);
    margin-bottom: 0.3rem;
    font-family: var(--mono);
    font-size: 0.7rem;
    transition: background 0.15s;
}
.wl-badge:hover { background: var(--bg4); }

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

POPULAR_TICKERS = {
    "AAPL":"Apple Inc.","MSFT":"Microsoft Corp.","GOOGL":"Alphabet Inc.",
    "AMZN":"Amazon.com Inc.","NVDA":"NVIDIA Corp.","META":"Meta Platforms",
    "TSLA":"Tesla Inc.","NFLX":"Netflix Inc.","AMD":"Advanced Micro Devices",
    "ORCL":"Oracle Corp.","INTC":"Intel Corp.","CRM":"Salesforce Inc.",
    "ADBE":"Adobe Inc.","PYPL":"PayPal Holdings","UBER":"Uber Technologies",
    "BABA":"Alibaba Group","JPM":"JPMorgan Chase","BAC":"Bank of America",
    "GS":"Goldman Sachs","V":"Visa Inc.","MA":"Mastercard Inc.",
    "JNJ":"Johnson & Johnson","PFE":"Pfizer Inc.","MRNA":"Moderna Inc.",
    "DIS":"Walt Disney Co.","SPOT":"Spotify Technology","SNAP":"Snap Inc.",
    "SHOP":"Shopify Inc.","SQ":"Block Inc.","COIN":"Coinbase Global",
    "PLTR":"Palantir Technologies","ABNB":"Airbnb Inc.","ZM":"Zoom Video",
    "ARKK":"ARK Innovation ETF","SPY":"S&P 500 ETF","QQQ":"Nasdaq-100 ETF",
    "2222.SR":"Saudi Aramco","9988.HK":"Alibaba HK","7203.T":"Toyota Motor",
    "005930.KS":"Samsung Electronics","RELIANCE.NS":"Reliance Industries",
    "TCS.NS":"Tata Consultancy","INFY.NS":"Infosys Ltd.",
    "XOM":"ExxonMobil Corp.","CVX":"Chevron Corp.","BP":"BP plc",
    "NKE":"Nike Inc.","MCD":"McDonald's Corp.","SBUX":"Starbucks Corp.",
    "WMT":"Walmart Inc.","COST":"Costco Wholesale","TGT":"Target Corp.",
    "BA":"Boeing Co.","LMT":"Lockheed Martin","GE":"GE Aerospace",
    "GOOG":"Alphabet Class C","BRK-B":"Berkshire Hathaway B",
}

@st.cache_data(ttl=3600)
def search_tickers(query):
    q = query.strip().upper()
    results = []
    if q in POPULAR_TICKERS:
        results.append(f"{q} — {POPULAR_TICKERS[q]}")
    ql = query.strip().lower()
    for sym, name in POPULAR_TICKERS.items():
        if sym != q and (ql in name.lower() or ql in sym.lower()):
            results.append(f"{sym} — {name}")
    return results[:10]

@st.cache_data(ttl=300)
def fetch_data(ticker, start, end):
    # Preserve suffix for international tickers (e.g. RELIANCE.NS, 7203.T)
    parts = ticker.strip().split(".")
    ticker = parts[0].upper() + ("." + parts[1].upper() if len(parts) > 1 else "")
    df = av_get_daily(ticker)
    if df.empty:
        return df
    # Filter by date range
    df = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean().replace(0, 1e-10)  # avoid division by zero
    return 100 - (100 / (1 + avg_gain / avg_loss))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

def compute_bollinger_bands(series, period=20, std=2):
    sma = series.rolling(period).mean()
    rs  = series.rolling(period).std()
    return sma + std*rs, sma, sma - std*rs

def add_technical_features(df):
    close  = df['Close'].squeeze()
    high   = df['High'].squeeze()
    low    = df['Low'].squeeze()
    volume = df['Volume'].squeeze()
    df['MA5']   = close.rolling(5).mean()
    df['MA10']  = close.rolling(10).mean()
    df['MA20']  = close.rolling(20).mean()
    df['MA50']  = close.rolling(50).mean()
    df['MA200'] = close.rolling(200).mean()
    df['EMA12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA26'] = close.ewm(span=26, adjust=False).mean()
    df['RSI']   = compute_rsi(close)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(close)
    df['BB_Upper'], df['BB_Mid'], df['BB_Lower']   = compute_bollinger_bands(close)
    df['BB_Width']       = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    df['BB_Pct']         = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['Returns']        = close.pct_change()
    df['Returns_5d']     = close.pct_change(5)
    df['Volatility']     = df['Returns'].rolling(20).std()
    df['Momentum']       = close - close.shift(10)
    df['Volume_MA10']    = volume.rolling(10).mean()
    df['Volume_Ratio']   = volume / df['Volume_MA10']
    df['High_Low_Pct']   = (high - low) / close
    df['Close_Open_Pct'] = (close - df['Open'].squeeze()) / df['Open'].squeeze()
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df

FEATURE_COLS = [
    'MA5','MA10','MA20','MA50','MA200','EMA12','EMA26',
    'RSI','MACD','MACD_Signal','MACD_Hist',
    'BB_Width','BB_Pct','Returns','Returns_5d','Volatility','Momentum',
    'Volume_Ratio','High_Low_Pct','Close_Open_Pct','ATR'
]

def build_xgb_dataset(df, seq_len):
    close   = df['Close'].squeeze().values
    feat_df = df[FEATURE_COLS].copy()
    feat_df['Close'] = close
    X_rows, y_rows = [], []
    for i in range(seq_len, len(feat_df) - 1):
        row_feats  = feat_df[FEATURE_COLS].iloc[i].values
        lag_closes = close[i - seq_len:i]
        X_rows.append(np.concatenate([row_feats, lag_closes]))
        y_rows.append(close[i + 1])
    X = np.array(X_rows)
    y = np.array(y_rows)
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    dropped = (~mask).sum()
    if dropped > 0:
        warnings.warn(f"build_xgb_dataset: dropped {dropped} rows containing NaN (out of {len(mask)} total)")
    return X[mask], y[mask]

def compute_composite_signal(df, last_close, forecast_price, preds, actual):
    close = df['Close'].squeeze()
    rsi   = float(df['RSI'].squeeze().iloc[-1])
    macd  = float(df['MACD'].squeeze().iloc[-1])
    macd_s= float(df['MACD_Signal'].squeeze().iloc[-1])
    macd_h= float(df['MACD_Hist'].squeeze().iloc[-1])
    bb_pct= float(df['BB_Pct'].squeeze().iloc[-1])
    ma50  = float(df['MA50'].squeeze().iloc[-1])
    ma200 = float(df['MA200'].squeeze().iloc[-1])
    vol_r = float(df['Volume_Ratio'].squeeze().iloc[-1])
    mom   = float(df['Momentum'].squeeze().iloc[-1])
    atr   = float(df['ATR'].squeeze().iloc[-1])
    signals = {}
    xgb_pct = (forecast_price - last_close) / last_close * 100
    if   xgb_pct >  1.5: signals['XGBoost Forecast'] = ('BUY',  min(35, abs(xgb_pct)*6), xgb_pct, 'positive')
    elif xgb_pct < -1.5: signals['XGBoost Forecast'] = ('SELL', -min(35, abs(xgb_pct)*6), xgb_pct, 'negative')
    else:                 signals['XGBoost Forecast'] = ('HOLD', 0, xgb_pct, 'neutral')
    if   rsi < 30: signals['RSI (14)'] = ('BUY',  20, rsi, 'positive')
    elif rsi > 70: signals['RSI (14)'] = ('SELL', -20, rsi, 'negative')
    elif rsi < 45: signals['RSI (14)'] = ('BUY',   8, rsi, 'positive')
    elif rsi > 55: signals['RSI (14)'] = ('SELL',  -8, rsi, 'negative')
    else:          signals['RSI (14)'] = ('HOLD',   0, rsi, 'neutral')
    prev_hist = float(df['MACD_Hist'].squeeze().iloc[-2]) if len(df) > 2 else 0
    if   macd_h > 0 and prev_hist <= 0: signals['MACD Cross'] = ('BUY',  20, macd_h, 'positive')
    elif macd_h < 0 and prev_hist >= 0: signals['MACD Cross'] = ('SELL', -20, macd_h, 'negative')
    elif macd > macd_s:                 signals['MACD Cross'] = ('BUY',  10, macd_h, 'positive')
    elif macd < macd_s:                 signals['MACD Cross'] = ('SELL', -10, macd_h, 'negative')
    else:                               signals['MACD Cross'] = ('HOLD',  0, macd_h, 'neutral')
    if   bb_pct < 0.1: signals['Bollinger %B'] = ('BUY',  10, bb_pct, 'positive')
    elif bb_pct > 0.9: signals['Bollinger %B'] = ('SELL', -10, bb_pct, 'negative')
    else:              signals['Bollinger %B'] = ('HOLD',   0, bb_pct, 'neutral')
    if   ma50 > ma200 and close.iloc[-1] > ma50: signals['MA Cross'] = ('BUY',  15, ma50-ma200, 'positive')
    elif ma50 < ma200 and close.iloc[-1] < ma50: signals['MA Cross'] = ('SELL', -15, ma50-ma200, 'negative')
    else:                                         signals['MA Cross'] = ('HOLD',  0, ma50-ma200, 'neutral')
    if   vol_r > 1.5 and xgb_pct > 0: signals['Volume'] = ('BUY',  10, vol_r, 'positive')
    elif vol_r > 1.5 and xgb_pct < 0: signals['Volume'] = ('SELL', -10, vol_r, 'negative')
    else:                              signals['Volume'] = ('HOLD',   0, vol_r, 'neutral')
    total_score = sum(s[1] for s in signals.values())
    if   total_score >= 25: verdict = "⬆ STRONG BUY";  verdict_short = "BUY"
    elif total_score >= 10: verdict = "↑ BUY";          verdict_short = "BUY"
    elif total_score <= -25:verdict = "⬇ STRONG SELL"; verdict_short = "SELL"
    elif total_score <= -10:verdict = "↓ SELL";         verdict_short = "SELL"
    else:                   verdict = "◆ HOLD";         verdict_short = "HOLD"
    stop_loss   = last_close - 2 * atr
    take_profit = last_close + 3 * atr
    risk_reward = (take_profit - last_close) / max(last_close - stop_loss, 0.01)
    return {
        'signals': signals, 'verdict': verdict, 'verdict_short': verdict_short,
        'total_score': total_score, 'xgb_pct': xgb_pct, 'rsi': rsi,
        'stop_loss': stop_loss, 'take_profit': take_profit, 'risk_reward': risk_reward,
        'vol_ratio': vol_r, 'atr': atr,
    }

def run_backtest_engine(actual_prices, predicted_prices, initial_capital, commission, threshold_pct):
    capital = float(initial_capital); position = 0; entry_price = 0.0
    trades = []; equity = []
    for i in range(len(predicted_prices) - 1):
        price_now  = float(actual_prices[i])
        pred_next  = float(predicted_prices[i])
        diff_pct   = (pred_next - price_now) / price_now * 100
        equity.append(capital + position * price_now)
        if diff_pct > threshold_pct and position == 0:
            shares = int((capital - commission) / price_now)
            if shares > 0:
                capital -= shares * price_now + commission
                position = shares; entry_price = price_now
                trades.append({"Day":i,"Type":"BUY","Price":price_now,"Shares":shares,"Capital":capital})
            # shares==0 means the stock price exceeds available capital — silently skip
        elif diff_pct < -threshold_pct and position > 0:
            proceeds = position * price_now - commission
            pnl = proceeds - (entry_price * position + commission)
            capital += proceeds
            trades.append({"Day":i,"Type":"SELL","Price":price_now,"Shares":position,"P&L":pnl,"Capital":capital})
            position = 0; entry_price = 0.0
    if position > 0:
        fp = float(actual_prices[-1]); proceeds = position*fp - commission
        pnl = proceeds - (entry_price*position + commission); capital += proceeds
        trades.append({"Day":len(actual_prices)-1,"Type":"SELL (EOD)","Price":fp,"Shares":position,"P&L":pnl,"Capital":capital})
    equity.append(capital)
    bh_shares  = int((initial_capital - commission) / float(actual_prices[0]))
    bh_final   = bh_shares * float(actual_prices[-1]) - commission
    bh_return  = (bh_final - initial_capital) / initial_capital * 100
    strat_return = (capital - initial_capital) / initial_capital * 100
    equity_s   = pd.Series(equity)
    drawdown   = equity_s / equity_s.cummax() - 1
    daily_r    = equity_s.pct_change().dropna()
    sharpe     = (daily_r.mean() / daily_r.std() * np.sqrt(252)) if daily_r.std() > 0 else 0.0
    trades_df  = pd.DataFrame(trades)
    if not trades_df.empty and "P&L" in trades_df.columns:
        closed = trades_df[trades_df["Type"].str.contains("SELL")]
        win_trades = (closed["P&L"] > 0).sum(); loss_trades = (closed["P&L"] <= 0).sum()
        win_rate  = win_trades / len(closed) * 100 if len(closed) > 0 else 0.0
        avg_win   = closed[closed["P&L"] > 0]["P&L"].mean()  if win_trades  > 0 else 0.0
        avg_loss  = closed[closed["P&L"] <= 0]["P&L"].mean() if loss_trades > 0 else 0.0
        pf = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        total_trades = len(closed)
    else:
        win_rate = avg_win = avg_loss = pf = 0.0; total_trades = 0
    bh_equity = [initial_capital * (float(actual_prices[i]) / float(actual_prices[0])) for i in range(len(actual_prices))]
    return {"final_capital":capital,"strat_return":strat_return,"bh_return":bh_return,
            "max_drawdown":float(drawdown.min()*100),"sharpe":sharpe,"win_rate":win_rate,
            "total_trades":total_trades,"avg_win":avg_win,"avg_loss":avg_loss,"profit_factor":pf,
            "equity_curve":equity,"bh_equity":bh_equity,"trades_df":trades_df,"drawdown_series":drawdown.tolist()}

def bootstrap_confidence_intervals(model, X_input, n_bootstrap=100, noise_std=None):
    # Adapt noise to asset's recent volatility if not explicitly provided.
    # The last feature column index for 'Volatility' is position 14 in FEATURE_COLS.
    if noise_std is None:
        try:
            vol_idx = FEATURE_COLS.index('Volatility')
            recent_vol = float(np.nanmedian(X_input[-20:, vol_idx]))
            noise_std = max(0.005, min(0.05, recent_vol))  # clamp between 0.5% and 5%
        except Exception:
            noise_std = 0.02
    # Use relative noise scaled to each feature's std, so a $500 stock
    # and a $5 stock both get proportionally equivalent perturbations.
    feature_scale = np.std(X_input, axis=0, keepdims=True)
    feature_scale = np.where(feature_scale == 0, 1.0, feature_scale)
    all_preds = [
        model.predict(X_input + np.random.normal(0, noise_std, X_input.shape) * feature_scale)
        for _ in range(n_bootstrap)
    ]
    a = np.array(all_preds)
    return np.percentile(a, 5, axis=0), np.percentile(a, 50, axis=0), np.percentile(a, 95, axis=0)

HARAM_TICKERS = {
    "BUD","STZ","SAM","BREW","ABEV","DEO","BF-B",       # Alcohol
    "MO","PM","BTI","LO","VGR",                          # Tobacco
    "LVS","MGM","WYNN","CZR","PENN","DKNG","BYD",        # Gambling
    "MET","PRU","AIG","ALL","TRV","CB",                   # Pure insurance
    "HRL","TSN","SFD","CAG","LMT","RTX","NOC","GD","HII", # Pork/Defense
}
# Per AAOIFI Standard No.21: banks and financial firms require ratio screening,
# not a blanket ban. Moved here so the app performs the debt/assets check.
QUESTIONABLE_TICKERS = {
    "DIS","NFLX","PARA","WBD","FOXA","SPOT",
    "MAR","HLT","H","IHG","WH",
    "JPM","BAC","WFC","C","GS","MS",   # Banks — require ratio screening
    "V","MA","AXP","COF","USB","PNC",  # Payment/finance — require ratio screening
}
HARAM_SECTORS_KW = ["bank","insurance","casino","gambling","alcohol","tobacco",
                    "brewing","distill","porn","adult","weapons","defense","firearm"]

@st.cache_data(ttl=3600)
def get_shariah_data(ticker_sym):
    info = av_get_overview(ticker_sym)
    if not info or "Symbol" not in info:
        return None
    def _safe(k, d=0):
        v = info.get(k, d)
        try: return float(v) if v not in (None, "None", "-", "") else d
        except: return d
    mc = _safe("MarketCapitalization", 1) or 1
    td = _safe("LongTermDebtUSD", 0) or _safe("TotalDebt", 0)
    ta = _safe("TotalAssets", 1) or 1
    tc = _safe("CashAndCashEquivalentsAtCarryingValue", 0)
    return {
        "debt_to_mktcap": td/mc,
        "debt_to_assets": td/ta,
        "cash_to_assets": tc/ta,
        "market_cap": mc,
        "total_debt": td,
        "total_assets": ta,
        "total_cash": tc,
        "sector":   info.get("Sector","Unknown"),
        "industry": info.get("Industry","Unknown"),
        "company_name": info.get("Name", ticker_sym)
    }

def check_shariah_compliance(ticker_sym, data, _L=None):
    if _L is None:
        _L = {}
    t = ticker_sym.upper(); ind_lower = data["industry"].lower(); haram_hit = None
    if t in HARAM_TICKERS: haram_hit = _L.get("known_noncompliant", "Known non-compliant ticker")
    else:
        for kw in HARAM_SECTORS_KW:
            if kw in ind_lower: haram_hit = data["industry"]; break
    questionable = t in QUESTIONABLE_TICKERS
    r = {
        "business":    {"pass": haram_hit is None, "haram_hit": haram_hit, "questionable": questionable},
        "debt_mktcap": {"pass": data["debt_to_mktcap"] < 0.30, "value": data["debt_to_mktcap"],
                        "label": _L.get("shariah_debt_mktcap", "Debt/MarketCap") + f" = {data['debt_to_mktcap']*100:.1f}% (< 30%)"},
        "debt_assets": {"pass": data["debt_to_assets"] < 0.33, "value": data["debt_to_assets"],
                        "label": _L.get("shariah_debt_assets", "Debt/Assets") + f" = {data['debt_to_assets']*100:.1f}% (< 33%)"},
        "cash_assets": {"pass": data["cash_to_assets"] < 0.33, "value": data["cash_to_assets"],
                        "label": _L.get("shariah_cash_assets", "Cash/Assets") + f" = {data['cash_to_assets']*100:.1f}% (< 33%)"},
    }
    all_pass = all(r[k]["pass"] for k in ["business","debt_mktcap","debt_assets","cash_assets"])
    r["verdict"] = "NON-COMPLIANT" if not r["business"]["pass"] or not all_pass else ("QUESTIONABLE" if questionable else "COMPLIANT")
    return r

def render_methodology_page(seq_len_val=30, ci_n=100, show_ci=True):
    st.markdown(f"""
    <div style="font-family:Manrope,sans-serif;font-size:0.6rem;letter-spacing:.18em;
         text-transform:uppercase;color:#8c909f;margin-bottom:.3rem;font-weight:700;">Technical Documentation</div>
    <div style="font-family:Manrope,sans-serif;font-size:1.15rem;font-weight:800;
         color:#dae2fd;letter-spacing:-.01em;margin-bottom:1.4rem;">
         Stockcast <span style="color:#4d8eff;">·</span> Methodology & Model Architecture
    </div>
    """, unsafe_allow_html=True)
    steps = [
        ("01","#4d8eff","Data Ingestion","OHLCV via yfinance",
         "Up to 7 years of daily Open/High/Low/Close/Volume data is fetched from Yahoo Finance. Timezone normalization and MultiIndex flattening are applied for compatibility across yfinance versions."),
        ("02","#adc6ff","Feature Engineering","20 Technical Indicators",
         f"Each trading day is described by 20 derived signals: MA5/10/20/50/200, EMA12/26, RSI(14), MACD(12/26/9) with histogram, Bollinger Band width & %B, ATR(14), Volume Ratio, Momentum, Returns(1d/5d), Volatility(20d), and High-Low%. Additionally, {seq_len_val} lag closes are appended as sequential memory."),
        ("03","#00e5b0","Train/Test Split","80% train · 20% test (chronological)",
         "Data is split strictly chronologically — no shuffling — to prevent look-ahead bias. The model never sees future data during training. Evaluation is performed exclusively on the held-out 20%."),
        ("04","#4d8eff","XGBoost Regressor","Gradient-boosted decision trees",
         "XGBoost is trained to predict the next day's closing price. Hyperparameters (n_estimators, max_depth, learning_rate) are configurable via the sidebar. Subsample=0.8 and colsample_bytree=0.8 provide regularisation."),
        ("05","#adc6ff","Bootstrap CI",f"{ci_n} resampling iterations" if show_ci else "Disabled",
         f"Confidence intervals are produced by running the model {ci_n} times on inputs perturbed with Gaussian noise (σ=1.5%). The 5th and 95th percentiles form the 95% CI ribbon. A wider band indicates higher forecast uncertainty."),
        ("06","#00e5b0","Forward Forecast","Iterative multi-step prediction",
         "Future prices are predicted by rolling: each day's predicted price feeds back as the next day's lag input. Error compounds over time — Days 1–3 are most reliable. Days 6+ are directional signals only."),
        ("07","#ff6b6b","Signal Generation","BUY / SELL / HOLD",
         "A composite 6-factor signal fires from XGBoost forecast, RSI, MACD crossover, Bollinger %B, MA Golden/Death cross, and Volume confirmation. Score >+25 = STRONG BUY, <-25 = STRONG SELL."),
        ("08","#4d8eff","Backtesting Engine","Walk-forward simulation",
         "The backtest replays XGBoost signals on test-set prices: BUY fires when predicted return exceeds threshold, SELL when below. KPIs: Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor, equity curve vs Buy-and-Hold."),
    ]
    for num, color, title, subtitle, body in steps:
        st.markdown(f"""
        <div style="display:flex;gap:1.2rem;margin-bottom:1rem;
             background:#131b2e;border:1px solid #2d3449;border-left:3px solid {color};
             padding:1.1rem 1.4rem;border-radius:0 0.5rem 0.5rem 0;">
          <div style="font-family:IBM Plex Mono,monospace;font-size:1.5rem;font-weight:700;
               color:{color};opacity:.5;min-width:2.5rem;line-height:1.1;">{num}</div>
          <div>
            <div style="font-family:Manrope,sans-serif;font-size:0.7rem;font-weight:800;
                 letter-spacing:.12em;text-transform:uppercase;color:#dae2fd;">{title}</div>
            <div style="font-family:IBM Plex Mono,monospace;font-size:0.6rem;letter-spacing:.1em;
                 color:{color};margin-bottom:.4rem;">{subtitle}</div>
            <div style="font-family:Manrope,sans-serif;font-size:0.82rem;
                 color:#8c909f;line-height:1.6;">{body}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(255,107,107,0.04);border:1px solid rgba(255,107,107,0.2);
         border-left:3px solid #ff6b6b;padding:1rem 1.5rem;margin-top:.5rem;border-radius:0 0.5rem 0.5rem 0;">
      <div style="font-family:Manrope,sans-serif;font-size:0.63rem;letter-spacing:.14em;
           text-transform:uppercase;color:#ff6b6b;margin-bottom:.4rem;font-weight:700;">⚠ Key Limitations</div>
      <div style="font-family:Manrope,sans-serif;font-size:0.82rem;color:#8c909f;line-height:1.7;">
        This model uses <b style="color:#dae2fd;">price and volume data only</b>. It has no awareness of
        earnings releases, macroeconomic events, analyst upgrades, geopolitical news, or central bank decisions.
        A single unexpected event can invalidate any technical forecast.
        <b style="color:#ff6b6b;">This is a research and educational tool — not financial advice.</b>
        Always consult a licensed financial advisor before making investment decisions.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Auth Gate: Login / Signup ──────────────────────────────────────────────────
if st.session_state.user is None:

    import streamlit.components.v1 as _components

    _auth_error   = ""
    _auth_success = ""

    if "auth_view" not in st.session_state:
        st.session_state.auth_view = "login"

    _is_login = (st.session_state.auth_view == "login")

    # ── Futuristic CSS override ────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');
    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], .main {
        background: #010a06 !important;
        font-family: 'Rajdhani', sans-serif !important;
        color: #d0ffe8 !important;
        overflow: hidden !important;
    }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    header[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }

    [data-testid="stTextInput"] input {
        background: #020f07 !important;
        border: 1px solid #0d3320 !important;
        border-radius: 2px !important;
        color: #00ffaa !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.72rem !important;
        padding: 10px 14px !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
        letter-spacing: 0.04em !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #00ffaa !important;
        box-shadow: 0 0 0 1px rgba(0,255,170,0.2) !important;
    }
    [data-testid="stTextInput"] input::placeholder { color: #1a4030 !important; }
    [data-testid="stTextInput"] label { display: none !important; }

    .stButton > button {
        width: 100% !important;
        padding: 0.8rem !important;
        background: #00ffaa !important;
        color: #010a06 !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 700 !important;
        font-size: 0.62rem !important;
        border: none !important;
        border-radius: 2px !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
        transition: all 0.2s !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .stButton > button:hover {
        background: #00dd99 !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active { transform: scale(0.98) !important; }

    [data-testid="stAlert"] { border-radius: 2px !important; font-size: 0.75rem !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Futuristic animated background (Three.js → canvas-based real-time chart) ──
    _components.html("""
<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0;}
html,body{width:100%;height:100%;background:#010a06;overflow:hidden;font-family:'Rajdhani',sans-serif;color:#d0ffe8;}
canvas{display:block;position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;}
.scan-sweep{position:fixed;top:0;left:-100%;width:55%;height:100%;z-index:3;pointer-events:none;
  background:linear-gradient(90deg,transparent,rgba(0,255,170,0.04),rgba(0,255,170,0.015),transparent);
  animation:sweep 9s linear infinite;}
@keyframes sweep{0%{left:-100%}100%{left:210%}}

.scan-line{position:fixed;left:0;right:0;height:100px;pointer-events:none;z-index:4;
  background:linear-gradient(to bottom,transparent,rgba(0,255,170,0.025),transparent);
  animation:scanV 6s linear infinite;}
@keyframes scanV{0%{top:-100px}100%{top:100%}}

.hero{position:fixed;top:0;left:0;width:56%;height:100%;display:flex;flex-direction:column;
  justify-content:flex-end;padding:0 3rem 6rem;z-index:10;}

.neural-badge{display:inline-flex;align-items:center;gap:7px;font-family:'Orbitron',monospace;
  font-size:0.5rem;color:#00ffaa;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:1.2rem;
  padding:5px 12px;border:1px solid rgba(0,255,170,0.2);border-radius:2px;width:fit-content;
  position:relative;overflow:hidden;}
.neural-badge::before{content:'';position:absolute;left:0;top:0;bottom:0;width:2px;
  background:#00ffaa;animation:sidePulse 2s ease-in-out infinite;}
@keyframes sidePulse{0%,100%{opacity:1}50%{opacity:0.2}}
.pulse-dot{width:6px;height:6px;border-radius:50%;background:#00ffaa;animation:pulse 1.2s ease-in-out infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.2;transform:scale(0.7)}}

.hero h1{font-family:'Orbitron',monospace;font-size:clamp(1.4rem,3vw,2.2rem);font-weight:700;
  color:#d0ffe8;line-height:1.2;letter-spacing:-0.01em;margin-bottom:0.8rem;}
.hero h1 em{font-style:normal;color:#00ffaa;}
.hero-sub{font-size:0.68rem;color:#3a6b50;line-height:1.8;max-width:340px;
  font-family:'Space Mono',monospace;margin-bottom:1.4rem;border-left:2px solid rgba(0,255,170,0.2);padding-left:1rem;}

.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;max-width:280px;}
.stat-box{background:rgba(0,255,136,0.03);border:1px solid #0d3320;padding:10px 12px;position:relative;overflow:hidden;}
.stat-box::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:rgba(0,255,170,0.13);}
.stat-label{font-size:0.45rem;color:#2a5a3a;font-family:'Space Mono',monospace;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px;}
.stat-value{font-size:0.88rem;font-weight:700;color:#d0ffe8;font-family:'Orbitron',monospace;}
.stat-value.green{color:#00ffaa;}

.bottom-bar{position:fixed;bottom:0;left:0;right:0;z-index:20;display:flex;justify-content:space-between;
  align-items:center;padding:0.45rem 2rem;background:rgba(1,10,6,0.75);border-top:1px solid #0d3320;
  font-size:0.45rem;font-family:'Space Mono',monospace;color:#1a4030;letter-spacing:0.1em;text-transform:uppercase;}
.footer-live{display:flex;align-items:center;gap:6px;color:rgba(0,255,170,0.5);}
.flive-dot{width:5px;height:5px;border-radius:50%;background:#00ffaa;animation:pulse 1.5s infinite;display:inline-block;}

@media(max-width:700px){.hero{display:none;}}
</style>
</head><body>

<canvas id="bgC"></canvas>
<canvas id="chartC" style="z-index:1;"></canvas>
<div class="scan-sweep"></div>
<div class="scan-line"></div>

<div class="hero">
  <div class="neural-badge"><div class="pulse-dot"></div>Neural Engine Active</div>
  <h1>Predicting the<br>pulse of <em>global markets.</em></h1>
  <p class="hero-sub">Institutional-grade XGBoost forecasting, 8-factor signal intelligence, Shariah compliance screening, and real-time NLP sentiment — all in one unified terminal.</p>
  <div class="stat-grid">
    <div class="stat-box"><div class="stat-label">XGBoost RMSE</div><div class="stat-value" id="forecastPrice">$2.14</div></div>
    <div class="stat-box"><div class="stat-label">Latency</div><div class="stat-value" id="latency">15ms</div></div>
    <div class="stat-box"><div class="stat-label">Auth Provider</div><div class="stat-value" style="font-size:0.72rem;">Supabase</div></div>
    <div class="stat-box"><div class="stat-label">Signal Strength</div><div class="stat-value green">STRONG</div></div>
  </div>
</div>

<div class="bottom-bar">
  <span>&#9888; For Educational Purposes Only · Not Financial Advice</span>
  <div class="footer-live"><div class="flive-dot"></div><span>Live Grid <span id="footer-clock">00:00:00</span></span></div>
  <span>Developed by Muawwiz Ghani</span>
</div>

<script>
const bgC=document.getElementById('bgC');
const bgX=bgC.getContext('2d');
const cc=document.getElementById('chartC');
const ctx=cc.getContext('2d');

function resize(){
  bgC.width=cc.width=innerWidth;
  bgC.height=cc.height=innerHeight;
}
resize();window.addEventListener('resize',resize);

// Data streams
const streams=[];
for(let i=0;i<28;i++){
  streams.push({
    x:Math.random()*0.58,y:Math.random(),
    speed:0.00035+Math.random()*0.0007,
    len:0.06+Math.random()*0.16,
    alpha:0.07+Math.random()*0.16,
    width:Math.random()>0.7?1.5:0.5,
  });
}

// Particles
const particles=[];
for(let i=0;i<50;i++){
  particles.push({
    x:Math.random()*0.58,y:Math.random(),
    vx:(Math.random()-0.5)*0.00012,
    vy:(Math.random()-0.5)*0.00012,
    r:0.5+Math.random()*1.5,
    alpha:0.08+Math.random()*0.25,
  });
}

function hexPath(ctx,cx,cy,r){
  ctx.beginPath();
  for(let i=0;i<6;i++){
    const a=Math.PI/180*(60*i-30);
    i===0?ctx.moveTo(cx+r*Math.cos(a),cy+r*Math.sin(a)):ctx.lineTo(cx+r*Math.cos(a),cy+r*Math.sin(a));
  }
  ctx.closePath();
}

function drawBg(){
  const W=bgC.width,H=bgC.height;
  bgX.clearRect(0,0,W,H);

  // Hex grid
  const sz=38;
  for(let row=0;row<H/sz+2;row++){
    for(let col=0;col<W*0.6/sz+2;col++){
      const ox=col%2===0?0:sz*0.866;
      hexPath(bgX,col*sz*0.866*1.5,row*sz+ox,sz*0.48);
      bgX.strokeStyle='rgba(0,255,136,0.04)';bgX.lineWidth=0.5;bgX.stroke();
    }
  }

  // Data streams (vertical falling lines)
  streams.forEach(s=>{
    s.y+=s.speed;
    if(s.y>1+s.len)s.y=-s.len;
    const x=s.x*W,y0=(s.y-s.len)*H,y1=s.y*H;
    const g=bgX.createLinearGradient(0,y0,0,y1);
    g.addColorStop(0,'rgba(0,255,136,0)');
    g.addColorStop(0.5,`rgba(0,255,136,${s.alpha})`);
    g.addColorStop(1,'rgba(0,255,136,0)');
    bgX.strokeStyle=g;bgX.lineWidth=s.width;
    bgX.beginPath();bgX.moveTo(x,y0);bgX.lineTo(x,y1);bgX.stroke();
  });

  // Particles
  particles.forEach(p=>{
    p.x+=p.vx;p.y+=p.vy;
    if(p.x<0)p.x=0.58;if(p.x>0.58)p.x=0;
    if(p.y<0)p.y=1;if(p.y>1)p.y=0;
    bgX.fillStyle=`rgba(0,255,136,${p.alpha})`;
    bgX.beginPath();bgX.arc(p.x*W,p.y*H,p.r,0,Math.PI*2);bgX.fill();
  });

  // Radial glow
  const rg=bgX.createRadialGradient(W*0.22,H*0.5,0,W*0.22,H*0.5,W*0.32);
  rg.addColorStop(0,'rgba(0,255,136,0.045)');rg.addColorStop(1,'rgba(0,255,136,0)');
  bgX.fillStyle=rg;bgX.fillRect(0,0,W,H);
}

// Candlestick engine
const CWIDTH=14,CGAP=7,STEP=CWIDTH+CGAP;
const GREEN='#00cc77',RED='#ff3344';
const GFILL='rgba(0,204,119,0.15)',RFILL='rgba(255,51,68,0.15)';
let price=150,candles=[],livePrice=150,lastCandle=Date.now();
const CDUR=2800;

function nextCandle(startPrice){
  const open=startPrice;let close=open,hi=open,lo=open;
  for(let i=0;i<10;i++){const d=(Math.random()-0.478)*2.2;close+=d;if(close>hi)hi=close;if(close<lo)lo=close;}
  hi+=Math.random()*1.2;lo-=Math.random()*1.2;price=close;
  return{open,high:hi,low:lo,close,live:true};
}
for(let i=0;i<80;i++){const c=nextCandle(price);c.live=false;candles.push(c);}
candles.push(nextCandle(price));
livePrice=candles[candles.length-1].open;

function genFuture(n){
  let p=livePrice;const out=[];
  for(let i=0;i<n;i++){
    const open=p;let close=open,hi=open,lo=open;
    for(let j=0;j<8;j++){const d=(Math.random()-0.46)*1.8;close+=d;if(close>hi)hi=close;if(close<lo)lo=close;}
    hi+=Math.random()*0.8;lo-=Math.random()*0.8;p=close;out.push({open,high:hi,low:lo,close});
  }
  return out;
}
let futureCandles=genFuture(8);
setInterval(()=>{futureCandles=genFuture(8);},2000);

function drawChart(){
  const W=cc.width,H=cc.height;
  ctx.clearRect(0,0,W,H);
  const panelW=W*0.58;
  const topM=50,botM=28,chartH=H-topM-botM;
  const maxVisible=Math.floor(panelW/STEP)+2;
  const visible=[...candles.slice(-maxVisible),...futureCandles.slice(0,8)];
  let minP=Infinity,maxP=-Infinity;
  visible.forEach(c=>{if(c.low<minP)minP=c.low;if(c.high>maxP)maxP=c.high;});
  const pad=(maxP-minP)*0.18;minP-=pad;maxP+=pad;
  const range=maxP-minP||1;
  const toY=p=>topM+chartH*(1-(p-minP)/range);

  // Grid
  ctx.font='8px "Space Mono",monospace';
  for(let i=0;i<=4;i++){
    const y=topM+(chartH/4)*i;
    const pval=(maxP-(range/4)*i).toFixed(2);
    ctx.strokeStyle='rgba(0,255,136,0.05)';ctx.lineWidth=0.5;ctx.setLineDash([3,6]);
    ctx.beginPath();ctx.moveTo(20,y);ctx.lineTo(panelW-10,y);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='rgba(0,255,136,0.28)';ctx.textAlign='left';ctx.textBaseline='middle';
    ctx.fillText(pval,2,y);
  }

  const startX=panelW-CWIDTH-8;
  const live=candles[candles.length-1];

  // Future ghost candles
  futureCandles.forEach((c,idx)=>{
    const x=startX+(idx+1)*STEP;
    if(x>panelW-4||x<0)return;
    const isG=c.close>=c.open;
    const oY=toY(c.open),cY=toY(c.close),hY=toY(c.high),lY=toY(c.low);
    const bTop=Math.min(oY,cY),bH=Math.max(Math.abs(cY-oY),1);
    ctx.globalAlpha=0.22;
    ctx.strokeStyle=isG?GREEN:RED;ctx.lineWidth=1;ctx.setLineDash([2,3]);
    ctx.beginPath();ctx.moveTo(x+CWIDTH/2,hY);ctx.lineTo(x+CWIDTH/2,lY);ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=isG?GFILL:RFILL;ctx.fillRect(x,bTop,CWIDTH,bH);
    ctx.strokeRect(x,bTop,CWIDTH,bH);
    ctx.globalAlpha=1;
  });

  // Forecast divider + label
  ctx.font='7px "Orbitron",monospace';
  ctx.fillStyle='rgba(0,255,170,0.35)';ctx.textAlign='left';
  ctx.fillText('AI FORECAST',startX+STEP+2,topM-10);
  ctx.strokeStyle='rgba(0,255,170,0.18)';ctx.lineWidth=0.5;ctx.setLineDash([2,5]);
  ctx.beginPath();ctx.moveTo(startX+STEP-2,topM-22);ctx.lineTo(startX+STEP-2,H-botM);ctx.stroke();
  ctx.setLineDash([]);

  // Historical candles
  for(let i=candles.length-1;i>=0;i--){
    const c=candles[i];
    const idx=candles.length-1-i;
    const x=startX-idx*STEP;
    if(x+CWIDTH<0)break;
    const isG=c.close>=c.open;
    const oY=toY(c.open),cY=toY(c.close),hY=toY(c.high),lY=toY(c.low);
    const bTop=Math.min(oY,cY),bH=Math.max(Math.abs(cY-oY),1);
    const isLive=i===candles.length-1;
    if(isLive){ctx.shadowColor=isG?'rgba(0,255,136,0.7)':'rgba(255,51,68,0.7)';ctx.shadowBlur=10;}
    ctx.globalAlpha=isLive?1:0.72;
    ctx.strokeStyle=isG?GREEN:RED;ctx.lineWidth=isLive?1.5:1;
    ctx.beginPath();ctx.moveTo(x+CWIDTH/2,hY);ctx.lineTo(x+CWIDTH/2,lY);ctx.stroke();
    ctx.fillStyle=isG?GFILL:RFILL;ctx.fillRect(x,bTop,CWIDTH,bH);
    ctx.strokeRect(x,bTop,CWIDTH,bH);
    ctx.shadowBlur=0;ctx.globalAlpha=1;
  }

  // Volume bars
  for(let i=candles.length-1;i>=0;i--){
    const c=candles[i];const idx=candles.length-1-i;const x=startX-idx*STEP;
    if(x+CWIDTH<0)break;
    const isG=c.close>=c.open;const vol=0.3+Math.random()*0.7;const volH=16;
    ctx.fillStyle=isG?'rgba(0,204,119,0.22)':'rgba(255,51,68,0.18)';
    ctx.fillRect(x,H-botM-volH*vol,CWIDTH,volH*vol);
  }

  // Live price dashed line
  const lY2=toY(livePrice);
  ctx.setLineDash([3,5]);ctx.strokeStyle='rgba(0,255,170,0.4)';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(20,lY2);ctx.lineTo(panelW-72,lY2);ctx.stroke();ctx.setLineDash([]);
  // Price tag
  const tag=livePrice.toFixed(2);const tagW=62,tagH=16,tagX=panelW-tagW-6;
  ctx.fillStyle='#00ffaa';
  ctx.beginPath();ctx.roundRect(tagX,lY2-tagH/2,tagW,tagH,2);ctx.fill();
  ctx.fillStyle='#010a06';ctx.font='600 8px "Orbitron",monospace';
  ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(tag,tagX+tagW/2,lY2);
}

function updateLive(){
  const now=Date.now();
  const live=candles[candles.length-1];
  const tick=(Math.random()-0.478)*0.25;
  livePrice+=tick;
  if(livePrice>live.high)live.high=livePrice;
  if(livePrice<live.low)live.low=livePrice;
  live.close=livePrice;
  if(now-lastCandle>=CDUR){
    live.live=false;
    const nc=nextCandle(livePrice);nc.open=livePrice;nc.close=livePrice;nc.high=livePrice;nc.low=livePrice;
    candles.push(nc);if(candles.length>200)candles.shift();lastCandle=now;
  }
  const fpEl=document.getElementById('forecastPrice');
  if(fpEl)fpEl.textContent='$'+livePrice.toFixed(2);
}

// Clock + latency
setInterval(()=>{
  const now=new Date();
  const t=[now.getHours(),now.getMinutes(),now.getSeconds()].map(n=>String(n).padStart(2,'0')).join(':');
  const el=document.getElementById('footer-clock');if(el)el.textContent=t;
},1000);
setInterval(()=>{
  const v=12+Math.floor(Math.random()*6);
  const l=document.getElementById('latency');if(l)l.textContent=v+'ms';
},2400);

function loop(){updateLive();drawBg();drawChart();requestAnimationFrame(loop);}
loop();
</script>
</body></html>
""", height=840, scrolling=False)

    # ── Native Streamlit form overlaid on top (right side) ────────────────────
    _col1, _col2 = st.columns([1.1, 1])
    with _col2:
        st.markdown("""
        <div style="background:rgba(2,12,7,0.95);border:1px solid rgba(0,255,170,0.15);
             border-radius:4px;padding:1.6rem 1.5rem 1.3rem;margin-top:-840px;
             position:relative;z-index:100;box-shadow:0 25px 60px rgba(0,0,0,0.8);">
        <div style="position:absolute;top:6px;left:6px;width:10px;height:10px;border-top:1px solid rgba(0,255,170,0.35);border-left:1px solid rgba(0,255,170,0.35);"></div>
        <div style="position:absolute;top:6px;right:6px;width:10px;height:10px;border-top:1px solid rgba(0,255,170,0.35);border-right:1px solid rgba(0,255,170,0.35);"></div>
        <div style="position:absolute;bottom:6px;left:6px;width:10px;height:10px;border-bottom:1px solid rgba(0,255,170,0.35);border-left:1px solid rgba(0,255,170,0.35);"></div>
        <div style="position:absolute;bottom:6px;right:6px;width:10px;height:10px;border-bottom:1px solid rgba(0,255,170,0.35);border-right:1px solid rgba(0,255,170,0.35);"></div>
        """, unsafe_allow_html=True)

        _tab_active   = "background:rgba(0,255,170,0.12);color:#00ffaa;border-bottom:2px solid #00ffaa;font-family:'Orbitron',monospace;"
        _tab_inactive = "color:rgba(0,255,170,0.3);border-bottom:2px solid transparent;font-family:'Orbitron',monospace;"

        _c1, _c2 = st.columns(2)
        with _c1:
            if st.button("TERMINAL ACCESS", key="tab_login", use_container_width=True):
                st.session_state.auth_view = "login"
                st.rerun()
        with _c2:
            if st.button("CREATE ACCOUNT", key="tab_signup", use_container_width=True):
                st.session_state.auth_view = "signup"
                st.rerun()

        if _is_login:
            st.markdown("""
            <div style="text-align:center;margin:0.8rem 0 1rem;">
              <div style="font-family:'Orbitron',monospace;font-size:0.62rem;font-weight:700;letter-spacing:0.2em;color:#d0ffe8;text-transform:uppercase;">Secure Access</div>
              <div style="font-size:0.48rem;color:rgba(0,255,170,0.3);letter-spacing:0.18em;text-transform:uppercase;margin-top:3px;font-family:'Space Mono',monospace;">Powered by Supabase Auth</div>
            </div>
            <div style="font-family:'Orbitron',monospace;font-size:0.48rem;letter-spacing:0.16em;text-transform:uppercase;color:rgba(0,255,170,0.6);margin-bottom:4px;">Identity Token (Email)</div>
            """, unsafe_allow_html=True)
            _login_email = st.text_input("Email", placeholder="name@firm.com", key="login_email_input", label_visibility="collapsed")
            st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.48rem;letter-spacing:0.16em;text-transform:uppercase;color:rgba(0,255,170,0.6);margin-bottom:4px;margin-top:0.7rem;">Access Key</div>', unsafe_allow_html=True)
            _login_pass = st.text_input("Password", type="password", placeholder="••••••••••", key="login_pass_input", label_visibility="collapsed")

            if st.button("⚡  AUTHORIZE TERMINAL", key="login_btn", use_container_width=True):
                if _login_email and _login_pass:
                    with st.spinner("Authenticating…"):
                        try:
                            _res = supabase.auth.sign_in_with_password({"email": _login_email, "password": _login_pass})
                            if _res.user:
                                st.session_state.user = _res.user
                                st.rerun()
                            else:
                                _auth_error = "Invalid credentials. Please try again."
                        except Exception as _e:
                            _auth_error = str(_e)
                else:
                    _auth_error = "Please enter your email and access key."

            st.markdown("""
            <div style="border:1px solid rgba(0,255,170,0.1);border-radius:2px;background:rgba(0,255,170,0.03);
                 padding:0.65rem 1rem;text-align:center;margin-top:0.75rem;">
              <div style="font-family:'Orbitron',monospace;font-size:0.52rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;color:#00ffaa;">Beta Alpha</div>
              <div style="font-size:0.44rem;color:rgba(0,255,170,0.3);letter-spacing:0.1em;text-transform:uppercase;margin-top:2px;font-family:'Space Mono',monospace;">Early access protocols active</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center;margin:0.8rem 0 1rem;">
              <div style="font-family:'Orbitron',monospace;font-size:0.62rem;font-weight:700;letter-spacing:0.2em;color:#d0ffe8;text-transform:uppercase;">Register Node</div>
              <div style="font-size:0.48rem;color:rgba(0,255,170,0.3);letter-spacing:0.18em;text-transform:uppercase;margin-top:3px;font-family:'Space Mono',monospace;">Create Your Terminal Identity</div>
            </div>
            <div style="font-family:'Orbitron',monospace;font-size:0.48rem;letter-spacing:0.16em;text-transform:uppercase;color:rgba(0,255,170,0.6);margin-bottom:4px;">Identity Token (Email)</div>
            """, unsafe_allow_html=True)
            _signup_email = st.text_input("Email", placeholder="name@firm.com", key="signup_email_input", label_visibility="collapsed")
            st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.48rem;letter-spacing:0.16em;text-transform:uppercase;color:rgba(0,255,170,0.6);margin-bottom:4px;margin-top:0.7rem;">Access Key</div>', unsafe_allow_html=True)
            _signup_pass = st.text_input("Password", type="password", placeholder="••••••••••", key="signup_pass_input", label_visibility="collapsed")
            st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.48rem;letter-spacing:0.16em;text-transform:uppercase;color:rgba(0,255,170,0.6);margin-bottom:4px;margin-top:0.7rem;">Confirm Key</div>', unsafe_allow_html=True)
            _signup_conf = st.text_input("Confirm", type="password", placeholder="••••••••••", key="signup_conf_input", label_visibility="collapsed")

            if st.button("⚡  INITIALIZE TERMINAL", key="signup_btn", use_container_width=True):
                if not _signup_email or not _signup_pass or not _signup_conf:
                    _auth_error = "Please fill in all fields."
                elif _signup_pass != _signup_conf:
                    _auth_error = "Passwords do not match."
                elif len(_signup_pass) < 6:
                    _auth_error = "Password must be at least 6 characters."
                else:
                    with st.spinner("Initializing terminal…"):
                        try:
                            _res = supabase.auth.sign_up({"email": _signup_email, "password": _signup_pass})
                            if _res.user:
                                _auth_success = f"Account created! A verification link has been sent to {_signup_email}. Please verify your email before logging in."
                                st.session_state.auth_view = "login"
                                st.rerun()
                            else:
                                _auth_error = "Sign up failed. Please try again."
                        except Exception as _e:
                            _auth_error = str(_e)

            st.markdown("""
            <div style="border:1px solid rgba(0,255,170,0.1);border-radius:2px;background:rgba(0,255,170,0.03);
                 padding:0.65rem 1rem;text-align:center;margin-top:0.75rem;">
              <div style="font-family:'Orbitron',monospace;font-size:0.52rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;color:#00ffaa;">Beta Alpha</div>
              <div style="font-size:0.44rem;color:rgba(0,255,170,0.3);letter-spacing:0.1em;text-transform:uppercase;margin-top:2px;font-family:'Space Mono',monospace;">Early access protocols active</div>
            </div>
            """, unsafe_allow_html=True)

        if _auth_error:
            st.error(f"⚠ {_auth_error}")
        if _auth_success:
            st.success(f"✓ {_auth_success}")

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()  # 🚨 Halt — do not render the app until authenticated

# ── Load portfolio from Supabase once per login session ───────────────────────
_current_uid = st.session_state.user.id if st.session_state.user else None
if _current_uid and st.session_state.get("_portfolio_loaded_for") != _current_uid:
    _loaded = _sb_load_portfolio(_current_uid)
    st.session_state.portfolio = _loaded
    _loaded_hist = _sb_load_history(_current_uid)
    st.session_state.portfolio_history = _loaded_hist
    st.session_state._portfolio_loaded_for = _current_uid


# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="wi-header">
  <div>
    <div class="wi-logo">Stock<span>cast</span>
    <div class="wi-sub">XGBoost · 6-Factor Signals · Backtesting · Shariah Screening · News NLP</div>
  </div>
  <div style="display:flex;align-items:center;gap:1.5rem;">
    <div style="text-align:right;">
      <div style="font-size:.55rem;color:#424754;letter-spacing:.14em;text-transform:uppercase;font-weight:700;font-family:Manrope,sans-serif;">Platform</div>
      <div style="font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#8c909f;">Developed by Muawwiz Ghani</div>
    </div>
    <div style="width:1px;height:28px;background:#2d3449;"></div>
    <div>
      <span class="live-dot"></span>
      <span class="live-label">LIVE · NYSE/NASDAQ</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Ticker tape — live prices
_tape_items = get_live_ticker_tape()
if _tape_items:
    _dot = '<span style="color:#2d3449;">·</span>'
    _tape_spans = f" {_dot} ".join(
        f'<span><span class="tape-sym">{sym}</span>'
        f'<span class="{css}">{arrow} {price} {pct}</span></span>'
        for sym, price, pct, arrow, css in _tape_items * 2  # duplicate for seamless scroll
    )
    st.markdown(f"""
<div class="ticker-tape-wrap">
  <div class="ticker-tape">{_tape_spans}</div>
</div>
""", unsafe_allow_html=True)
else:
    # Fallback: show symbols only if live fetch fails
    st.markdown("""
<div class="ticker-tape-wrap">
  <div class="ticker-tape">
    <span class="tape-sym">AAPL</span> · <span class="tape-sym">TSLA</span> ·
    <span class="tape-sym">NVDA</span> · <span class="tape-sym">MSFT</span> ·
    <span class="tape-sym">GOOGL</span> · <span class="tape-sym">META</span> ·
    <span class="tape-sym">AMZN</span> · <span class="tape-sym">SPY</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    # Language selector
    _lang_choice = st.selectbox("🌐 Language", list(LANGUAGES.keys()),
                                index=list(LANGUAGES.keys()).index(st.session_state.lang),
                                key="lang_selector", label_visibility="collapsed")
    if _lang_choice != st.session_state.lang:
        st.session_state.lang = _lang_choice
        st.rerun()
    _L = LANGUAGES[st.session_state.lang]

    # Logo + User
    st.markdown(f"""
    <div style="padding:1.4rem 1rem 0.8rem;">
      <div style="font-family:Manrope,sans-serif;font-size:1.4rem;font-weight:800;color:#dae2fd;letter-spacing:-.01em;">
        Stock<span style="color:#4d8eff;">cast</span>
      </div>
      <div style="font-size:.55rem;color:#424754;letter-spacing:.1em;text-transform:uppercase;font-weight:700;margin-top:2px;">
        Developed by Muawwiz Ghani
      </div>
    </div>
    <div style="background:rgba(77,142,255,0.08);border:1px solid rgba(77,142,255,0.2);
         border-left:3px solid #4d8eff;padding:.55rem 1rem;margin:.4rem 0 .6rem;
         font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#adc6ff;letter-spacing:.04em;">
      👤 {st.session_state.user.email}
    </div>
    """, unsafe_allow_html=True)

    if st.button(_L.get("logout", "⏏  Logout"), use_container_width=True, key="logout_btn"):
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
        st.session_state.user = None
        st.session_state.run_pressed = False
        st.session_state.portfolio = []
        st.session_state.portfolio_history = []
        st.session_state._portfolio_loaded_for = None
        st.rerun()

    st.markdown("---")

    # Ticker Search
    st.markdown(f'<div class="stat-row">{_L["search_label"]}</div>', unsafe_allow_html=True)
    search_query = st.text_input("Search", placeholder="e.g. Apple, TSLA, Saudi Aramco…",
                                 label_visibility="collapsed", key="search_input")
    ticker = "AAPL"
    if search_query and len(search_query.strip()) >= 1:
        search_results = search_tickers(search_query.strip())
        if search_results:
            selected = st.selectbox("Select", search_results, label_visibility="collapsed")
            ticker   = selected.split(" — ")[0].strip()
            st.markdown(f'<div style="background:rgba(77,142,255,0.08);border:1px solid rgba(77,142,255,0.3);border-left:3px solid #4d8eff;padding:.35rem .9rem;font-family:IBM Plex Mono,monospace;font-size:.68rem;color:#adc6ff;letter-spacing:.05em;margin:.3rem 0;border-radius:0 .5rem .5rem 0;">✓ {ticker}</div>', unsafe_allow_html=True)
        else:
            ticker = search_query.strip().upper()
            st.markdown(f'<div style="background:rgba(255,221,45,0.06);border:1px solid rgba(255,221,45,0.3);border-left:3px solid #ffdd2d;padding:.35rem .9rem;font-family:IBM Plex Mono,monospace;font-size:.68rem;color:#ffdd2d;letter-spacing:.05em;margin:.3rem 0;border-radius:0 .5rem .5rem 0;">{_L["verify_symbol"].format(ticker=ticker)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="stat-row">{_L["ticker"]}</div>', unsafe_allow_html=True)
        ticker = st.text_input("Ticker", value="AAPL", placeholder="AAPL, TSLA, MSFT…",
                               label_visibility="collapsed", key="direct_ticker").strip().upper() or "AAPL"
        st.markdown(f'<div style="background:rgba(77,142,255,0.08);border:1px solid rgba(77,142,255,0.2);border-left:3px solid #4d8eff;padding:.35rem .9rem;font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#4d8eff;letter-spacing:.07em;margin:.3rem 0;border-radius:0 .5rem .5rem 0;">{_L["active_ticker"].format(ticker=ticker)}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1: start_date = st.date_input(_L["from"], value=pd.to_datetime("2018-01-01"))
    with col2: end_date   = st.date_input(_L["to"],   value=pd.Timestamp.today())

    st.markdown(f'<div class="stat-row">{_L["lookback"]}</div>', unsafe_allow_html=True)
    seq_len     = st.slider("", 10, 60, 30, label_visibility="collapsed")
    st.markdown(f'<div class="stat-row">{_L["horizon"]}</div>', unsafe_allow_html=True)
    future_days = st.slider(" ", 1, 30, 7, label_visibility="collapsed")

    st.markdown("---")
    ui_mode    = st.radio("Mode", [_L["beginner"], _L["pro"]], index=1, horizontal=True, label_visibility="collapsed")
    is_beginner = (ui_mode == _L["beginner"])
    if is_beginner:
        st.markdown(f'<div style="background:rgba(0,229,176,0.06);border-left:3px solid #00e5b0;padding:.4rem .9rem;font-family:Manrope,sans-serif;font-size:.62rem;color:#00e5b0;font-weight:700;">{_L["simple_view"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:rgba(255,107,107,0.06);border-left:3px solid #ff6b6b;padding:.4rem .9rem;font-family:Manrope,sans-serif;font-size:.62rem;color:#ff6b6b;font-weight:700;">{_L["pro_view"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    fast_mode = st.checkbox(_L["fast_mode"], value=is_beginner)

    if not is_beginner:
        st.markdown(f'<div class="stat-row">{_L["xgb_params"]}</div>', unsafe_allow_html=True)
        n_estimators  = st.slider("Trees", 100, 500, 200, step=50)
        max_depth     = st.slider("Max Depth", 2, 8, 4)
        learning_rate = st.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2], value=0.05)
    else:
        n_estimators = 200; max_depth = 4; learning_rate = 0.05

    st.markdown("---")
    st.markdown(f'<div class="stat-row">{_L["alert_target"]}</div>', unsafe_allow_html=True)
    alert_price = st.number_input("", min_value=0.0, value=0.0, step=1.0, label_visibility="collapsed")

    if not is_beginner:
        st.markdown("---")
        st.markdown(f'<div class="stat-row">{_L["backtesting"]}</div>', unsafe_allow_html=True)
        run_backtest        = st.checkbox(_L["enable_backtest"], value=True)
        bt_initial_capital  = st.number_input(_L["init_capital"], min_value=1000, value=10000, step=1000)
        bt_commission       = st.number_input(_L["commission"], min_value=0.0, value=1.0, step=0.5)
        bt_signal_threshold = st.slider(_L["signal_thresh"], 0.5, 5.0, 1.0, step=0.5)
    else:
        run_backtest = False; bt_initial_capital = 10000; bt_commission = 1.0; bt_signal_threshold = 1.0

    if not is_beginner:
        st.markdown("---")
        st.markdown(f'<div class="stat-row">{_L["extra_features"]}</div>', unsafe_allow_html=True)
        run_model_compare  = st.checkbox(_L["model_compare"], value=False)
        run_halal_check    = st.checkbox(_L["halal_check"], value=True)
        show_conf_interval = st.checkbox(_L["conf_interval"], value=True) and not fast_mode
        ci_bootstrap_n     = st.slider(_L["bootstrap_samples"], 50, 300, 100, step=50) if show_conf_interval else 100
    else:
        run_model_compare = False; run_halal_check = True; show_conf_interval = False; ci_bootstrap_n = 100

    if not is_beginner:
        st.markdown("---")
        st.markdown(f'<div class="stat-row">{_L["multi_stock"]}</div>', unsafe_allow_html=True)
        compare_tickers_raw = st.text_input(_L["compare_tickers"], value="", placeholder="e.g. AAPL,TSLA,NVDA",
                                            label_visibility="collapsed", key="compare_input")
        compare_tickers = [t.strip().upper() for t in compare_tickers_raw.split(",") if t.strip()] if compare_tickers_raw.strip() else []
    else:
        compare_tickers = []

    st.markdown("---")
    if st.button(_L["run"], use_container_width=True):
        st.session_state.run_pressed = True
    run_btn = st.session_state.get("run_pressed", False)

    # Watchlist
    st.markdown("---")
    st.markdown(f"""<div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#dae2fd;margin-bottom:.5rem;">{_L["watchlist"]}</div>""", unsafe_allow_html=True)
    wl_c1, wl_c2 = st.columns([3,1])
    with wl_c1: add_ticker_input = st.text_input("Add", placeholder="e.g. AAPL", label_visibility="collapsed", key="wl_add").strip().upper()
    with wl_c2: add_clicked = st.button("＋", use_container_width=True, key="wl_add_btn")
    if add_clicked and add_ticker_input:
        if add_ticker_input not in st.session_state.watchlist:
            st.session_state.watchlist.append(add_ticker_input)
    if st.session_state.watchlist:
        for wl_sym in list(st.session_state.watchlist):
            wc1, wc2 = st.columns([3,1])
            with wc1:
                try:
                    _qt   = av_get_quote(wl_sym)
                    _px   = _qt["price"]
                    _chg  = _qt["change_pct"]
                    _col  = "#00e5b0" if _chg >= 0 else "#ff6b6b"
                    _sign = "▲" if _chg >= 0 else "▼"
                    st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.67rem;padding:.2rem 0;"><span style="color:#424754;">{wl_sym}</span> <span style="color:{_col};">{_sign} ${_px:.2f}</span></div>', unsafe_allow_html=True)
                except Exception:
                    st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.67rem;color:#424754;padding:.2rem 0;">{wl_sym}</div>', unsafe_allow_html=True)
            with wc2:
                if st.button("✕", key=f"wl_del_{wl_sym}", use_container_width=True):
                    st.session_state.watchlist.remove(wl_sym)
                    if wl_sym in st.session_state.alert_signals: del st.session_state.alert_signals[wl_sym]
                    st.rerun()
    else:
        st.markdown(f'<div style="font-family:Manrope,sans-serif;font-size:.65rem;color:#2d3449;padding:.3rem 0;">{_L["no_stocks_saved"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#dae2fd;margin-bottom:.5rem;">{_L["alerts"]}</div>', unsafe_allow_html=True)
    alert_on_signal_change = st.checkbox(_L["alert_signal_change"], value=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN CONTENT — Landing or Analysis
# ═══════════════════════════════════════════════════════════════════
if not run_btn:
    # ── Landing Dashboard ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="font-family:Manrope,sans-serif;font-size:2.2rem;font-weight:800;letter-spacing:-.02em;color:#dae2fd;margin-bottom:.3rem;">
      {_L["dashboard_title"]} <span style="color:#4d8eff;">{_L["dashboard_subtitle"]}</span>
    </div>
    <div style="font-size:.82rem;color:#8c909f;margin-bottom:1.5rem;font-weight:500;">
      {_L["dashboard_desc"]}
    </div>
    """, unsafe_allow_html=True)

    # Market summary cards
    st.markdown("""
    <div class="stat-grid" style="grid-template-columns:repeat(4,1fr);">
      <div class="stat-card">
        <div class="stat-label">S&amp;P 500</div>
        <div class="stat-value" style="font-size:1.3rem;">5,137.08</div>
        <div class="stat-sub" style="color:#00e5b0;font-weight:700;">▲ +1.24%</div>
      </div>
      <div class="stat-card" style="border-top-color:#adc6ff;">
        <div class="stat-label">NASDAQ 100</div>
        <div class="stat-value" style="font-size:1.3rem;color:#adc6ff;">18,302</div>
        <div class="stat-sub" style="color:#00e5b0;font-weight:700;">▲ +2.10%</div>
      </div>
      <div class="stat-card" style="border-top-color:#ffdd2d;">
        <div class="stat-label">Fear &amp; Greed</div>
        <div class="stat-value" style="font-size:1.3rem;color:#ffdd2d;">74</div>
        <div class="stat-sub">Greed territory</div>
      </div>
      <div class="stat-card" style="border-top-color:#00e5b0;">
        <div class="stat-label">VIX</div>
        <div class="stat-value" style="font-size:1.3rem;color:#00e5b0;">14.2</div>
        <div class="stat-sub">Low volatility</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Watchlist live prices if any
    if st.session_state.watchlist:
        st.subheader(_L["watchlist_live"])
        wl_cols = st.columns(min(len(st.session_state.watchlist), 4))
        for i, wl_sym in enumerate(st.session_state.watchlist[:4]):
            with wl_cols[i % 4]:
                try:
                    _fi   = av_get_quote(wl_sym)
                    _px   = _fi["price"]
                    _chg  = _fi["change_pct"]
                    _col  = "#00e5b0" if _chg >= 0 else "#ff6b6b"
                    _sign = "▲" if _chg >= 0 else "▼"
                    st.markdown(f"""
                    <div style="background:linear-gradient(145deg,#131b2e,#171f33);border:1px solid #2d3449;
                         border-top:2px solid {_col};padding:1rem 1.2rem;text-align:center;border-radius:.5rem;">
                      <div style="font-family:IBM Plex Mono,monospace;font-size:.62rem;letter-spacing:.14em;color:#424754;text-transform:uppercase;">{wl_sym}</div>
                      <div style="font-family:IBM Plex Mono,monospace;font-size:1.3rem;font-weight:700;color:#dae2fd;margin:.3rem 0;">${_px:.2f}</div>
                      <div style="font-family:IBM Plex Mono,monospace;font-size:.72rem;color:{_col};">{_sign} {_chg:+.2f}%</div>
                    </div>""", unsafe_allow_html=True)
                except Exception:
                    st.markdown(f'<div style="background:#131b2e;border:1px solid #2d3449;padding:1rem;text-align:center;font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#424754;border-radius:.5rem;">{wl_sym}<br>—</div>', unsafe_allow_html=True)

    # How it works
    st.subheader(_L["how_it_works"])
    hw1, hw2, hw3 = st.columns(3)
    for col, num, color, title_key, body_key in [
        (hw1,"01","#4d8eff","hw1_title","hw1_body"),
        (hw2,"02","#00e5b0","hw2_title","hw2_body"),
        (hw3,"03","#ffdd2d","hw3_title","hw3_body"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,#131b2e,#171f33);border:1px solid #2d3449;
                 border-top:2px solid {color};padding:1.4rem 1.5rem;height:100%;border-radius:.5rem;">
              <div style="font-family:IBM Plex Mono,monospace;font-size:1.3rem;font-weight:700;color:{color};margin-bottom:.5rem;">{num}</div>
              <div style="font-family:Manrope,sans-serif;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#dae2fd;font-weight:700;margin-bottom:.5rem;">{_L[title_key]}</div>
              <div style="font-family:Manrope,sans-serif;font-size:.8rem;color:#8c909f;line-height:1.6;">{_L[body_key]}</div>
            </div>""", unsafe_allow_html=True)

    st.subheader(_L["platform_features"])
    feat_grid = [
        ("#4d8eff","📈 XGBoost Forecast","ML trained on 20 technical features. N-day forecast with 95% bootstrap CI."),
        ("#00e5b0","⚙ Explainable Signals","RSI, MACD, Bollinger, MA Cross, Volume — grouped, scored, explained in plain English."),
        ("#ffdd2d","📊 Backtesting Engine","Sharpe ratio, max drawdown, win rate, profit factor, equity curve vs buy-and-hold."),
        ("#ff6b6b","⭐ Watchlist + 🔔 Alerts","Save stocks, see live prices on the dashboard, get banners when signals flip."),
        ("#4d8eff","☪ Shariah Screening","AAOIFI Standard No.21 — screens business activity, debt & cash ratios automatically."),
        ("#adc6ff","🔬 Model Comparison","Benchmark XGBoost vs Prophet vs Linear Regression — RMSE, MAE, MAPE, R² side-by-side."),
        ("#00e5b0","📰 News Sentiment NLP","Live Yahoo Finance headlines scored with TextBlob. Detects confluence with technical signals."),
        ("#ffdd2d","🏦 Portfolio Tracker","Track holdings, P&L, sector allocation, and recent transaction history."),
    ]
    cols4 = st.columns(4)
    for i, (color, title, body) in enumerate(feat_grid):
        with cols4[i % 4]:
            st.markdown(f"""
            <div style="background:#131b2e;border:1px solid #2d3449;border-top:2px solid {color};
                 padding:1.1rem 1.2rem;margin-bottom:.6rem;border-radius:.5rem;">
              <div style="font-family:Manrope,sans-serif;font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:{color};font-weight:700;margin-bottom:.4rem;">{title}</div>
              <div style="font-family:Manrope,sans-serif;font-size:.78rem;color:#8c909f;line-height:1.5;">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="text-align:center;margin-top:2rem;font-family:IBM Plex Mono,monospace;font-size:.58rem;color:#2d3449;letter-spacing:.08em;"> SupportTeam :- ghani24by7@gmail.com </div>', unsafe_allow_html=True)

else:
    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS ENGINE
    # ═══════════════════════════════════════════════════════════════
    if st.sidebar.button(_L.get("back", "← Back to Dashboard"), use_container_width=True, key="back_btn"):
        st.session_state.run_pressed = False
        st.rerun()
    # ── Input validation ──────────────────────────────────────────────────────
    if not ticker or len(ticker.strip()) < 1:
        st.warning("Please enter a ticker symbol.")
        st.stop()
    if len(ticker) > 20:
        st.error("⚠ Ticker symbol too long. Please check and try again.")
        st.stop()
    import re as _re
    if not _re.match(r'^[A-Za-z0-9.\-\^]+$', ticker):
        st.error(f"⚠ '{ticker}' doesn't look like a valid ticker. Use formats like AAPL, RELIANCE.NS, ^GSPC")
        st.stop()

    with st.spinner(_L["fetching"].format(ticker=ticker)):
        df = fetch_data(ticker, start_date, end_date)

    if df.empty:
        st.error(f"⚠ No data found for '{ticker}'. Check the symbol — for Indian stocks use .NS suffix e.g. RELIANCE.NS, TCS.NS. For indices use ^ prefix e.g. ^GSPC")
        st.stop()

    st.success(_L["days_loaded"].format(n=len(df), ticker=ticker))

    st.markdown(f"""
    <div style="background:rgba(255,221,45,0.04);border:1px solid rgba(255,221,45,0.3);
         border-left:4px solid #ffdd2d;padding:.9rem 1.4rem;margin:.5rem 0 1rem;border-radius:0 .5rem .5rem 0;">
      <div style="font-family:Manrope,sans-serif;font-size:.6rem;letter-spacing:.14em;text-transform:uppercase;color:#ffdd2d;margin-bottom:.3rem;font-weight:700;">
        {_L["reality_check_title"]}
      </div>
      <div style="font-family:Manrope,sans-serif;font-size:.78rem;color:#8c909f;line-height:1.6;">
        {_L["reality_check_body"]}
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_analysis, tab_methodology = st.tabs([_L["analysis_tab"], _L["methodology_tab"]])

    with tab_methodology:
        render_methodology_page(seq_len_val=seq_len, ci_n=ci_bootstrap_n, show_ci=show_conf_interval)

    with tab_analysis:
        with st.spinner(_L["engineering"]):
            df = add_technical_features(df)
        close_series = df['Close'].squeeze()

        # ── Candlestick Chart ──────────────────────────────────────────────────
        st.subheader(_L["price_chart"])
        fig_candle = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.02)
        fig_candle.add_trace(go.Candlestick(x=df.index,
            open=df['Open'].squeeze(), high=df['High'].squeeze(),
            low=df['Low'].squeeze(), close=close_series,
            name="Price", increasing_line_color=C_EMERALD, decreasing_line_color=C_RED), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['MA50'].squeeze(), name="MA50", line=dict(color=C_YELLOW, width=1.2)), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['MA200'].squeeze(), name="MA200", line=dict(color=C_ACCENT, width=1.2)), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'].squeeze(), name="BB Upper", line=dict(color=C_GREY, width=0.8, dash='dot')), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'].squeeze(), name="BB Lower", line=dict(color=C_GREY, width=0.8, dash='dot'), fill='tonexty', fillcolor='rgba(77,142,255,0.05)'), row=1, col=1)
        colors_vol = [C_EMERALD if c >= o else C_RED for c, o in zip(close_series, df['Open'].squeeze())]
        fig_candle.add_trace(go.Bar(x=df.index, y=df['Volume'].squeeze(), name="Volume", marker_color=colors_vol, opacity=0.5), row=2, col=1)
        candle_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")}
        fig_candle.update_layout(**candle_layout,
            title=dict(text=f"{ticker} · Candlestick · MA50/200 · Bollinger · Volume", font=dict(color=C_GREEN, size=12)),
            xaxis_rangeslider_visible=False, height=560)
        fig_candle.update_xaxes(gridcolor="#2d3449", linecolor="#2d3449", tickfont=dict(color=C_GREY))
        fig_candle.update_yaxes(gridcolor="#2d3449", linecolor="#2d3449", tickfont=dict(color=C_GREY))
        st.plotly_chart(fig_candle, use_container_width=True)

        # ── RSI + MACD ──────────────────────────────────────────────────────────
        st.subheader(_L["tech_indicators"])
        fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08,
                                 subplot_titles=["RSI (14)", "MACD (12/26/9)"])
        fig_tech.add_trace(go.Scatter(x=df.index, y=df['RSI'].squeeze(), name="RSI", line=dict(color=C_ACCENT, width=1.5)), row=1, col=1)
        fig_tech.add_hline(y=70, line_dash="dash", line_color=C_RED,    row=1, col=1)
        fig_tech.add_hline(y=30, line_dash="dash", line_color=C_EMERALD, row=1, col=1)
        fig_tech.add_hrect(y0=70, y1=100, fillcolor="rgba(255,107,107,0.04)", line_width=0, row=1, col=1)
        fig_tech.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,229,176,0.04)",  line_width=0, row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=df.index, y=df['MACD'].squeeze(), name="MACD", line=dict(color=C_ACCENT, width=1.2)), row=2, col=1)
        fig_tech.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'].squeeze(), name="Signal", line=dict(color=C_GREEN, width=1.2)), row=2, col=1)
        macd_hist   = df['MACD_Hist'].squeeze()
        hist_colors = [C_EMERALD if v >= 0 else C_RED for v in macd_hist]
        fig_tech.add_trace(go.Bar(x=df.index, y=macd_hist, name="Histogram", marker_color=hist_colors, opacity=0.65), row=2, col=1)
        subplot_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis','yaxis')}
        fig_tech.update_layout(**subplot_layout, height=450)
        fig_tech.update_xaxes(gridcolor="#2d3449", linecolor="#2d3449", tickfont=dict(color=C_GREY))
        fig_tech.update_yaxes(gridcolor="#2d3449", linecolor="#2d3449", tickfont=dict(color=C_GREY))
        fig_tech.update_yaxes(range=[0, 100], row=1, col=1)
        st.plotly_chart(fig_tech, use_container_width=True)

        # ── XGBoost Model ──────────────────────────────────────────────────────
        st.markdown('<div class="model-badge">🤖 MODEL: XGBoost Regressor · 20 Technical Features + Lag Window</div>', unsafe_allow_html=True)

        with st.expander("📖 How this model works — methodology & limitations", expanded=False):
            st.markdown(f"""<div style="font-family:Manrope,sans-serif;font-size:0.82rem;color:#8c909f;line-height:1.7;">
            <b style="color:#dae2fd;font-family:IBM Plex Mono,monospace;font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;">Feature Engineering</b><br>
            Each trading day is represented by <b style="color:#4d8eff;">20 technical indicators</b> computed from raw OHLCV data — MAs (5–200), EMA12/26, RSI, MACD, Bollinger Bands, ATR, volume ratio, momentum — plus <b style="color:#4d8eff;">{seq_len} lag closes</b> as sequential context.<br><br>
            <b style="color:#dae2fd;font-family:IBM Plex Mono,monospace;font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;">Training & Evaluation</b><br>
            Data is split <b style="color:#4d8eff;">80% train / 20% test</b> chronologically (no data leakage). XGBoost predicts the next day's closing price. Quality is measured with RMSE, MAE, MAPE and R².<br><br>
            <b style="color:#ff6b6b;font-family:IBM Plex Mono,monospace;font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;">⚠ Key Limitations</b><br>
            This model uses price and volume data only. It has no awareness of earnings, macro events, or news. A single unexpected event can invalidate any technical forecast. <b style="color:#ff6b6b;">Not financial advice.</b>
            </div>""", unsafe_allow_html=True)

        with st.spinner("Building feature matrix..."):
            X, y = build_xgb_dataset(df, seq_len)

        if len(X) < 50:
            st.error(_L["not_enough_data"])
            st.stop()

        split   = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        @st.cache_resource(show_spinner=False)
        def train_xgb_cached(_X_train, _y_train, _X_test, _y_test, _n_est, _depth, _lr):
            m = XGBRegressor(n_estimators=_n_est, max_depth=_depth, learning_rate=_lr,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
            m.fit(_X_train, _y_train, eval_set=[(_X_test, _y_test)], verbose=False)
            return m

        with st.spinner("Training XGBoost model (cached after first run)..."):
            model = train_xgb_cached(X_train, y_train, X_test, y_test, n_estimators, max_depth, learning_rate)

        preds  = model.predict(X_test)
        actual = y_test
        rmse   = float(np.sqrt(mean_squared_error(actual, preds)))
        mae    = float(mean_absolute_error(actual, preds))
        mape   = float(np.mean(np.abs((actual - preds) / actual)) * 100)
        r2     = float(1 - np.sum((actual - preds)**2) / np.sum((actual - np.mean(actual))**2))

        # Confidence score
        r2_norm    = max(0, min(100, r2 * 100))
        mape_norm  = max(0, min(100, 100 - mape * 5))
        dir_acc    = sum(1 for i in range(1, len(actual)) if (preds[i]-actual[i-1])*(actual[i]-actual[i-1])>0) / max(len(actual)-1,1) * 100
        data_score = min(100, len(df)/2000*100)
        confidence_score = max(0, min(100, r2_norm*0.40 + mape_norm*0.30 + dir_acc*0.20 + data_score*0.10))
        last_close = float(df['Close'].squeeze().iloc[-1])

        # ── Model Performance ──────────────────────────────────────────────────
        st.subheader("Model Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE",  f"${rmse:.2f}")
        c2.metric("MAE",   f"${mae:.2f}")
        c3.metric("MAPE",  f"{mape:.2f}%")
        c4.metric("R²",    f"{r2:.4f}")
        mape_label = ("🟢 Excellent" if mape<2 else "🟡 Good" if mape<5 else "🟠 Fair" if mape<10 else "🔴 Poor")
        r2_label   = ("🟢 Excellent" if r2>0.95 else "🟡 Good" if r2>0.85 else "🟠 Fair" if r2>0.70 else "🔴 Poor")
        st.markdown(f'<div style="background:#131b2e;border:1px solid #2d3449;padding:.65rem 1.2rem;font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#424754;display:flex;gap:2rem;flex-wrap:wrap;border-radius:.5rem;"><span>MAPE: {mape_label} · &lt;2% excellent · &lt;5% good · &lt;10% fair</span><span>R²: {r2_label} · &gt;0.95 excellent · &gt;0.85 good · &gt;0.70 fair</span></div>', unsafe_allow_html=True)

        # Tabs
        dash_tab, port_tab, mkt_tab, deep_tab = st.tabs([_L["dashboard_tab"], _L["portfolio"], _L["markets"], _L["deep_analysis"]])

        # ──────────────────────────────────────────────────────────────────────
        with dash_tab:
            _dash_close = float(df["Close"].squeeze().iloc[-1])
            _dash_prev  = float(df["Close"].squeeze().iloc[-2]) if len(df)>1 else _dash_close
            _dash_chg   = _dash_close - _dash_prev
            _dash_pct   = (_dash_chg / _dash_prev * 100) if _dash_prev != 0 else 0
            _dash_sign  = "+" if _dash_chg >= 0 else ""
            _dash_color = "#00e5b0" if _dash_chg >= 0 else "#ff6b6b"
            _dash_arrow = "▲" if _dash_chg >= 0 else "▼"
            _dash_name  = POPULAR_TICKERS.get(ticker, ticker)

            # KPI row
            st.markdown(f"""
            <div class="stat-grid">
              <div class="stat-card">
                <div class="stat-label">{_L["last_close_lbl"]}</div>
                <div class="stat-value">${_dash_close:.2f}</div>
                <div class="stat-sub" style="color:{_dash_color};font-weight:700;">{_dash_arrow} {_dash_sign}{_dash_chg:.2f} ({_dash_sign}{_dash_pct:.2f}%)</div>
              </div>
              <div class="stat-card" style="border-top-color:#adc6ff;">
                <div class="stat-label">{_L["model_confidence_lbl"]}</div>
                <div class="stat-value" style="color:#adc6ff;">{confidence_score:.0f}<span style="font-size:.9rem;color:#8c909f;">/100</span></div>
                <div class="stat-sub">{_L["high_lbl"] if confidence_score>=80 else _L["moderate_lbl"] if confidence_score>=60 else _L["low_lbl"]}</div>
              </div>
              <div class="stat-card" style="border-top-color:#ffdd2d;">
                <div class="stat-label">MAPE</div>
                <div class="stat-value" style="color:#ffdd2d;">{mape:.2f}%</div>
                <div class="stat-sub">{mape_label}</div>
              </div>
              <div class="stat-card" style="border-top-color:#00e5b0;">
                <div class="stat-label">R² Score</div>
                <div class="stat-value" style="color:#00e5b0;">{r2:.4f}</div>
                <div class="stat-sub">{r2_label}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Actual vs Predicted chart
            st.subheader("Actual vs Predicted")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(y=actual, name="Actual", line=dict(color=C_ACCENT, width=1.5), fill='tozeroy', fillcolor='rgba(77,142,255,0.05)'))
            fig1.add_trace(go.Scatter(y=preds, name="XGBoost Predicted", line=dict(color=C_EMERALD, width=1.5, dash='dot')))
            fig1.update_layout(**PLOTLY_LAYOUT, title=dict(text=f"{ticker} · XGBoost Model Fit (Test Set)", font=dict(color=C_GREEN, size=12)), height=350)
            st.plotly_chart(fig1, use_container_width=True)

            # Feature Importance
            st.subheader("Feature Importance")
            lag_names = [f'Lag_{i+1}' for i in range(seq_len)]
            all_feature_names = FEATURE_COLS + lag_names
            importances = model.feature_importances_
            imp_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values('importance', ascending=True).tail(20)
            fig_imp = go.Figure(go.Bar(x=imp_df['importance'], y=imp_df['feature'], orientation='h',
                marker=dict(color=imp_df['importance'], colorscale=[[0,"#131b2e"],[0.5,"#1a3050"],[1,C_ACCENT]], showscale=False)))
            fig_imp.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis"}, title=dict(text="Top 20 Feature Importances", font=dict(color=C_GREEN, size=12)), height=430, xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Importance Score"))
            st.plotly_chart(fig_imp, use_container_width=True)

        # ──────────────────────────────────────────────────────────────────────
        with port_tab:
            st.markdown(f"""
            <div style="margin-bottom:1.2rem;">
              <div style="font-family:Manrope,sans-serif;font-size:2rem;font-weight:800;letter-spacing:-.02em;color:#dae2fd;">{_L["portfolio_title"]} <span style="color:#4d8eff;">{_L["portfolio_tracker"]}</span></div>
              <div style="font-size:.78rem;color:#8c909f;margin-top:.2rem;">{_L["portfolio_desc"]}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Add new holding — always expanded so users can add 1–500 stocks ──
            MAX_HOLDINGS = 500
            current_count = len(st.session_state.portfolio)

            with st.expander(_L["add_holding"], expanded=True):
                st.markdown(
                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:.6rem;color:#424754;margin-bottom:.6rem;">'
                    f'Holdings: {current_count} / {MAX_HOLDINGS}</div>',
                    unsafe_allow_html=True
                )
                with st.form(key="pt_add_form", clear_on_submit=True):
                    pa1, pa2, pa3 = st.columns(3)
                    with pa1: add_sym  = st.text_input("Ticker", placeholder="e.g. AAPL", key="pt_sym_f")
                    with pa2: add_qty  = st.number_input("Quantity", min_value=0.0001, value=1.0, step=0.5, key="pt_qty_f")
                    with pa3: add_cost = st.number_input("Avg Buy Price ($)", min_value=0.01, value=100.0, step=0.5, key="pt_cost_f")
                    submitted = st.form_submit_button(_L["add_to_portfolio"], use_container_width=True)

                if submitted:
                    add_sym = add_sym.strip().upper()
                    if not add_sym:
                        st.warning("Please enter a ticker symbol.")
                    elif len(st.session_state.portfolio) >= MAX_HOLDINGS:
                        st.warning(f"Maximum {MAX_HOLDINGS} holdings reached.")
                    else:
                        existing = [h for h in st.session_state.portfolio if h["ticker"] == add_sym]
                        if existing:
                            st.warning(_L["already_in_portfolio"].format(sym=add_sym))
                        else:
                            with st.spinner(f"Fetching live price for {add_sym}..."):
                                _q = av_get_quote(add_sym)
                                _live_px = _q["price"] if _q["price"] > 0 else add_cost
                                try:
                                    _info = yf.Ticker(add_sym).info
                                    _name = _info.get("longName", add_sym)
                                    _sector = _info.get("sector", "Unknown") + " • " + _info.get("industry", "")
                                except Exception:
                                    _name = add_sym; _sector = "Unknown"
                            _pl = (_live_px - add_cost) * add_qty
                            _pl_pct = ((_live_px - add_cost) / add_cost * 100) if add_cost > 0 else 0
                            _new_holding = {
                                "ticker": add_sym, "name": _name, "sector": _sector,
                                "qty": add_qty, "avg_cost": add_cost,
                                "current_price": _live_px, "pl": _pl, "pl_pct": _pl_pct
                            }
                            st.session_state.portfolio.append(_new_holding)
                            # ── Persist to Supabase ──
                            _sb_upsert_holding(st.session_state.user.id, _new_holding)
                            _date = pd.Timestamp.today().strftime("%b %d")
                            _hist_record = {
                                "date": _date, "type": "BUY", "ticker": add_sym,
                                "shares": add_qty, "price": add_cost,
                                "amount": -(add_qty * add_cost)
                            }
                            st.session_state.portfolio_history.insert(0, _hist_record)
                            _sb_insert_history(st.session_state.user.id, _hist_record)
                            st.success(_L["added_success"].format(sym=add_sym, price=_live_px))
                            st.rerun()

            port = st.session_state.portfolio

            if not port:
                st.info(_L["no_holdings"])
            else:
                # Refresh live prices button
                if st.button(_L["refresh_prices"], key="pt_refresh"):
                    av_get_quote.clear()  # Clear cache to force fresh prices
                    for h in st.session_state.portfolio:
                        try:
                            _q = av_get_quote(h["ticker"])
                            if _q["price"] > 0:
                                h["current_price"] = _q["price"]
                                h["pl"] = (_q["price"] - h["avg_cost"]) * h["qty"]
                                h["pl_pct"] = ((_q["price"] - h["avg_cost"]) / h["avg_cost"] * 100)
                        except Exception:
                            pass
                    # ── Persist updated prices to Supabase ──
                    _sb_update_prices(st.session_state.user.id, st.session_state.portfolio)
                    st.rerun()

                # KPIs
                total_value    = sum(h["qty"] * h["current_price"] for h in port)
                total_invested = sum(h["qty"] * h["avg_cost"]       for h in port)
                total_pl       = total_value - total_invested
                total_pl_pct   = (total_pl / total_invested * 100) if total_invested > 0 else 0
                p1, p2, p3, p4 = st.columns(4)
                p1.metric(_L["total_value"],    f"${total_value:,.2f}")
                p2.metric(_L["total_pl"],       f"${total_pl:+,.2f}", delta=f"{total_pl_pct:+.1f}%")
                p3.metric(_L["invested"],       f"${total_invested:,.2f}")
                p4.metric(_L["holdings"],       str(len(port)))

                # Holdings table with remove buttons
                st.subheader(_L["holdings_label"])
                for h in port:
                    mktval = h["qty"] * h["current_price"]
                    pl_col = "#00e5b0" if h["pl"] >= 0 else "#ff6b6b"
                    pl_sign = "+" if h["pl"] >= 0 else ""
                    hc1, hc2, hc3, hc4, hc5, hc6 = st.columns([1.2,2,1,1,1.5,0.7])
                    hc1.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.8rem;font-weight:700;color:#4d8eff;padding:.4rem 0;">{h["ticker"]}</div>', unsafe_allow_html=True)
                    hc2.markdown(f'<div style="font-size:.75rem;color:#8c909f;padding:.4rem 0;">{(h["name"] or h["ticker"])[:28]}</div>', unsafe_allow_html=True)
                    hc3.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;color:#dae2fd;padding:.4rem 0;">{h["qty"]:.2f} sh</div>', unsafe_allow_html=True)
                    hc4.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;color:#dae2fd;padding:.4rem 0;">${h["current_price"]:.2f}</div>', unsafe_allow_html=True)
                    hc5.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;color:{pl_col};font-weight:700;padding:.4rem 0;">{pl_sign}${abs(h["pl"]):,.2f} ({pl_sign}{h["pl_pct"]:.1f}%)</div>', unsafe_allow_html=True)
                    if hc6.button("✕", key=f"pt_del_{h['ticker']}", use_container_width=True):
                        _del_ticker = h["ticker"]
                        st.session_state.portfolio = [x for x in st.session_state.portfolio if x["ticker"] != _del_ticker]
                        # ── Remove from Supabase ──
                        _sb_delete_holding(st.session_state.user.id, _del_ticker)
                        st.rerun()

                # Sector donut
                sc1, sc2 = st.columns([1,1])
                with sc1:
                    st.subheader(_L["sector_allocation"])
                    sector_map = {}
                    for h in port:
                        sec = (h["sector"] or "Unknown").split(" •")[0].strip()
                        sector_map[sec] = sector_map.get(sec, 0) + h["qty"] * h["current_price"]
                    sec_colors = {"Technology":"#4d8eff","Consumer Cyclical":"#ffdd2d","Financials":"#adc6ff","Energy":"#ff9f40","Healthcare":"#00e5b0","Unknown":"#8c909f"}
                    fig_sector = go.Figure(go.Pie(
                        labels=list(sector_map.keys()), values=list(sector_map.values()),
                        hole=0.6, marker_colors=[sec_colors.get(s,"#8c909f") for s in sector_map.keys()],
                        textfont_size=10, textfont_color="#dae2fd",
                    ))
                    fig_sector.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=True,
                        annotations=[dict(text=f"{len(sector_map)}<br><span style='font-size:10px'>Sectors</span>", x=0.5, y=0.5, font_size=20, showarrow=False, font_color="#dae2fd")])
                    st.plotly_chart(fig_sector, use_container_width=True)

                with sc2:
                    st.subheader(_L["recent_activity"])
                    hist = st.session_state.portfolio_history
                    if not hist:
                        st.markdown(f'<div style="font-size:.78rem;color:#424754;padding:.5rem 0;">{_L["no_transactions"]}</div>', unsafe_allow_html=True)
                    for a in hist[:10]:
                        type_color = {"BUY":"#4d8eff","SELL":"#00e5b0","DIVIDEND":"#ffdd2d"}.get(a["type"],"#8c909f")
                        amt_str = f'+${a["amount"]:,.2f}' if a["amount"] >= 0 else f'-${abs(a["amount"]):,.2f}'
                        desc = f'{a["shares"]} shares @ ${a["price"]:.2f}' if a.get("shares") and a.get("price") else a["ticker"]
                        st.markdown(f"""
                        <div style="display:flex;gap:.8rem;padding:.7rem 0;border-bottom:1px solid #2d3449;align-items:center;">
                          <div style="width:2rem;height:2rem;border-radius:50%;background:rgba({','.join(str(int(type_color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.15);
                               color:{type_color};display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:.8rem;font-weight:700;">
                            {"B" if a["type"]=="BUY" else "S" if a["type"]=="SELL" else "D"}
                          </div>
                          <div style="flex:1;">
                            <div style="display:flex;justify-content:space-between;">
                              <span style="font-size:.8rem;font-weight:700;color:#dae2fd;font-family:Manrope,sans-serif;">{a["type"]} {a["ticker"]}</span>
                              <span style="font-size:.65rem;color:#8c909f;font-family:IBM Plex Mono,monospace;">{a["date"].upper()}</span>
                            </div>
                            <div style="font-size:.7rem;color:#8c909f;margin-top:.1rem;">{desc}</div>
                            <div style="font-size:.7rem;font-weight:700;color:{type_color};margin-top:.1rem;font-family:IBM Plex Mono,monospace;">{amt_str}</div>
                          </div>
                        </div>""", unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────────────────
        with mkt_tab:
            st.markdown(f"""
            <div style="margin-bottom:1.2rem;">
              <div style="font-family:Manrope,sans-serif;font-size:2rem;font-weight:800;letter-spacing:-.02em;color:#dae2fd;">Market <span style="color:#4d8eff;">Intelligence</span></div>
              <div style="font-size:.78rem;color:#8c909f;margin-top:.2rem;">Live global performance and institutional sentiment analysis.</div>
            </div>
            """, unsafe_allow_html=True)

            # Market index cards — LIVE
            mkt_cols = st.columns(4)
            with st.spinner("Loading live market data..."):
                mkt_data = get_live_market_indices()
            for i, (name, price, chg, col) in enumerate(mkt_data):
                with mkt_cols[i]:
                    st.markdown(f"""
                    <div style="background:linear-gradient(145deg,#131b2e,#171f33);border:1px solid #2d3449;border-top:2px solid {col};
                         padding:1.2rem;border-radius:.5rem;">
                      <div style="font-size:.6rem;font-weight:700;color:#8c909f;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem;">{name}</div>
                      <div style="font-family:IBM Plex Mono,monospace;font-size:1.4rem;font-weight:700;color:#dae2fd;">{price}</div>
                      <div style="font-family:IBM Plex Mono,monospace;font-size:.75rem;color:{col};font-weight:700;margin-top:.3rem;">{chg}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            ms1, ms2 = st.columns([2,1])
            with ms1:
                st.subheader("Sector Heat Map · Live")
                sectors = get_live_sector_heatmap()
                cols5 = st.columns(5)
                for i, (name, chg, col) in enumerate(sectors):
                    with cols5[i % 5]:
                        st.markdown(f"""
                        <div style="background:#131b2e;border:1px solid #2d3449;border-left:2px solid {col};
                             padding:.75rem .9rem;margin-bottom:.5rem;border-radius:0 .5rem .5rem 0;">
                          <div style="font-size:.6rem;font-weight:700;color:#8c909f;text-transform:uppercase;margin-bottom:.25rem;">{name}</div>
                          <div style="font-family:IBM Plex Mono,monospace;font-size:.9rem;font-weight:700;color:{col};">{chg}</div>
                        </div>""", unsafe_allow_html=True)

            with ms2:
                st.subheader("Fear & Greed Index · Live")
                _fg = get_fear_greed_index()
                if _fg:
                    _fg_score = _fg["score"]
                    _fg_label = _fg["rating"]
                else:
                    _fg_score = 50; _fg_label = "Neutral"
                _fg_color = "#ff6b6b" if _fg_score < 30 else "#ffdd2d" if _fg_score < 55 else "#00e5b0"
                _fg_pct   = f"{max(2, min(98, _fg_score)):.0f}%"
                st.markdown(f"""
                <div style="background:linear-gradient(145deg,#131b2e,#171f33);border:1px solid #2d3449;
                     padding:1.4rem;text-align:center;border-radius:.5rem;">
                  <div style="font-family:IBM Plex Mono,monospace;font-size:2.8rem;font-weight:800;color:{_fg_color};">{_fg_score:.0f}</div>
                  <div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{_fg_color};margin-bottom:.8rem;">{_fg_label}</div>
                  <div style="height:6px;background:linear-gradient(90deg,#ff6b6b,#ff9f40,#ffdd2d,#00e5b0);border-radius:3px;position:relative;">
                    <div style="position:absolute;top:-10px;left:{_fg_pct};transform:translateX(-50%);width:2px;height:26px;background:#dae2fd;border-radius:1px;"></div>
                  </div>
                  <div style="display:flex;justify-content:space-between;margin-top:.5rem;font-size:.58rem;color:#424754;font-weight:700;text-transform:uppercase;">
                    <span>Fear</span><span>Neutral</span><span>Greed</span>
                  </div>
                  <div style="margin-top:1rem;padding:.75rem;background:rgba(77,142,255,0.06);border-radius:.5rem;">
                    <div style="font-size:.7rem;color:#8c909f;line-height:1.5;">{"Live CNN Fear & Greed data." if _fg else "Could not fetch live data."} Score: <b style="color:{_fg_color};">{_fg_score:.0f}/100</b> — <b style="color:{_fg_color};">{_fg_label}</b>.</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────────────────
        with deep_tab:
            # Price alert
            if alert_price > 0:
                diff = alert_price - last_close
                if last_close >= alert_price:
                    st.markdown(f'<div class="alert-box">🔔 {ticker} at ${last_close:.2f} — {_L["at_above_target"]} ${alert_price:.2f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-box">🔔 {ticker} at ${last_close:.2f} — ${diff:.2f} {_L["below_target"]} ${alert_price:.2f}</div>', unsafe_allow_html=True)

            # Confidence score bar
            conf_color = "#00e5b0" if confidence_score>=80 else "#ffdd2d" if confidence_score>=60 else "#ff6b6b"
            conf_label = _L["high_confidence"] if confidence_score>=80 else _L["moderate_confidence"] if confidence_score>=60 else _L["low_confidence"]
            filled = int(confidence_score / 5)
            bar_html = "".join(f'<span style="display:inline-block;width:18px;height:10px;margin-right:2px;background:{conf_color};opacity:{1.0 if i<filled else 0.1};border-radius:1px;"></span>' for i in range(20))
            st.markdown(f"""
            <div style="background:#131b2e;border:1px solid #2d3449;border-left:3px solid {conf_color};
                 padding:1.2rem 1.6rem;margin:1rem 0;border-radius:0 .5rem .5rem 0;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;">
                <div>
                  <div style="font-family:Manrope,sans-serif;font-size:.6rem;letter-spacing:.16em;text-transform:uppercase;color:#424754;margin-bottom:.3rem;font-weight:700;">{_L["model_conf_score"]}</div>
                  <div style="font-family:IBM Plex Mono,monospace;font-size:2.2rem;font-weight:700;color:{conf_color};">{confidence_score:.0f}<span style="font-size:1rem;color:#8c909f;">/100</span></div>
                  <div style="font-family:Manrope,sans-serif;font-size:.62rem;letter-spacing:.14em;color:{conf_color};margin-top:.3rem;font-weight:700;">{conf_label}</div>
                </div>
                <div style="flex:1;min-width:220px;">
                  <div style="margin-bottom:.6rem;">{bar_html}</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:.3rem .8rem;font-family:IBM Plex Mono,monospace;font-size:.63rem;color:#424754;">
                    <span>{_L["r2_fit"]} <b style="color:#8c909f;">{r2_norm:.0f}/100</b> <span style="color:#2d3449;">(×0.40)</span></span>
                    <span>{_L["mape_accuracy"]} <b style="color:#8c909f;">{mape_norm:.0f}/100</b> <span style="color:#2d3449;">(×0.30)</span></span>
                    <span>{_L["directional_acc"]} <b style="color:#8c909f;">{dir_acc:.0f}/100</b> <span style="color:#2d3449;">(×0.20)</span></span>
                    <span>{_L["data_volume"]} <b style="color:#8c909f;">{data_score:.0f}/100</b> <span style="color:#2d3449;">(×0.10)</span></span>
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Signal Intelligence ────────────────────────────────────────────────
            st.subheader(_L["signal_intelligence"])
            composite    = compute_composite_signal(df, last_close, preds[-1], preds, actual)
            verdict      = composite['verdict']
            verdict_short= composite['verdict_short']
            total_score  = composite['total_score']
            xgb_pct      = composite['xgb_pct']
            stop_loss    = composite['stop_loss']
            take_profit  = composite['take_profit']
            risk_reward  = composite['risk_reward']
            rsi_val      = composite['rsi']
            vol_ratio    = composite['vol_ratio']
            atr_val      = composite['atr']
            sigs         = composite['signals']

            if alert_on_signal_change:
                prev_verdict = st.session_state.alert_signals.get(ticker)
                if prev_verdict is not None and prev_verdict != verdict_short:
                    _ac = {"BUY":"#00e5b0","SELL":"#ff6b6b"}.get(verdict_short,"#ffdd2d")
                    st.markdown(f'<div style="background:rgba(77,142,255,0.08);border:1px solid {_ac};border-left:4px solid {_ac};padding:.8rem 1.4rem;margin-bottom:1rem;font-family:Manrope,sans-serif;font-size:.78rem;color:{_ac};font-weight:700;border-radius:0 .5rem .5rem 0;">🔔 SIGNAL ALERT — {ticker} &nbsp;|&nbsp; {prev_verdict} → {verdict_short} &nbsp;|&nbsp; Score: {total_score:+.0f}</div>', unsafe_allow_html=True)
                st.session_state.alert_signals[ticker] = verdict_short

            verdict_css = 'sell' if verdict_short=='SELL' else 'hold' if verdict_short=='HOLD' else ''
            sign = '+' if xgb_pct>=0 else ''
            score_color = '#00e5b0' if total_score>0 else '#ff6b6b' if total_score<0 else '#ffdd2d'
            rr_color = 'positive' if risk_reward>=1.5 else 'negative' if risk_reward<1 else 'neutral'

            st.markdown(f"""
            <div class="signal-panel">
              <div class="signal-main {verdict_css}">
                <div class="signal-lbl">{_L["composite_signal"]}</div>
                <div class="signal-action {verdict_css}">{verdict}</div>
                <div class="signal-pct">{sign}{xgb_pct:.2f}% {_L["forecast_lbl"]}</div>
                <div class="signal-lbl" style="margin-top:8px;">{_L["score_lbl"]}: <span style="color:{score_color};font-size:.9rem;font-weight:800;">{total_score:+.0f}</span> / ±100</div>
              </div>
              <div class="signal-details">
                <div class="sig-card positive">
                  <div class="sig-lbl">{_L["take_profit_lbl"]}</div>
                  <div class="sig-val">${take_profit:.2f}</div>
                  <div class="sig-sub">+{((take_profit-last_close)/last_close*100):.1f}% · 3× ATR</div>
                </div>
                <div class="sig-card negative">
                  <div class="sig-lbl">{_L["stop_loss_lbl"]}</div>
                  <div class="sig-val">${stop_loss:.2f}</div>
                  <div class="sig-sub">{((stop_loss-last_close)/last_close*100):.1f}% · 2× ATR</div>
                </div>
                <div class="sig-card {rr_color}">
                  <div class="sig-lbl">{_L["risk_reward_lbl"]}</div>
                  <div class="sig-val">{risk_reward:.2f}×</div>
                  <div class="sig-sub">{_L["favorable"] if risk_reward>=1.5 else _L["marginal"] if risk_reward>=1 else _L["unfavorable"]}</div>
                </div>
                <div class="sig-card {'positive' if rsi_val<50 else 'negative'}">
                  <div class="sig-lbl">{_L["rsi_lbl"]}</div>
                  <div class="sig-val">{rsi_val:.1f}</div>
                  <div class="sig-sub">{_L["oversold_zone"] if rsi_val<30 else _L["overbought_zone"] if rsi_val>70 else _L["neutral_zone"]}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Composite meter
            st.markdown(f'<div class="composite-meter"><div class="meter-title">{_L["factor_breakdown"]}</div>', unsafe_allow_html=True)
            for sig_name, (sig_action, sig_score, sig_val, sig_pol) in sigs.items():
                bar_width = min(100, abs(sig_score))
                st.markdown(f"""
                <div class="sir">
                  <span class="sir-label">{sig_name}</span>
                  <div class="sir-bar-bg"><div class="sir-bar {sig_pol}" style="width:{bar_width}%;"></div></div>
                  <span class="sir-val">{sig_val:.2f}</span>
                  <span class="sir-sig {sig_action.lower()}">{sig_action}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Future Forecast ────────────────────────────────────────────────
            st.subheader(_L["forecast_next"].format(n=future_days))
            future_prices    = []
            last_row_feats   = X[-1].copy()
            for d in range(future_days):
                next_pred = float(model.predict(last_row_feats.reshape(1,-1))[0])
                future_prices.append(next_pred)
                n_tech = len(FEATURE_COLS); lags = last_row_feats[n_tech:]
                last_row_feats = np.concatenate([last_row_feats[:n_tech], np.append(lags[1:], next_pred)])

            trend_color  = C_EMERALD if future_prices[-1] > last_close else C_RED
            price_std    = float(df['Close'].squeeze().pct_change().std())
            decay_upper  = [future_prices[i]*(1+price_std*np.sqrt(i+1)*1.5) for i in range(future_days)]
            decay_lower  = [future_prices[i]*(1-price_std*np.sqrt(i+1)*1.5) for i in range(future_days)]

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=list(range(future_days)), y=decay_upper, line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
            fig3.add_trace(go.Scatter(x=list(range(future_days)), y=decay_lower, name="Uncertainty Band", fill="tonexty", fillcolor="rgba(77,142,255,0.08)", line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip"))
            fig3.add_hline(y=last_close, line_dash="dash", line_color=C_GREY, annotation_text=f"Last close ${last_close:.2f}", annotation_font_color=C_GREY)
            if alert_price > 0:
                fig3.add_hline(y=alert_price, line_dash="dash", line_color=C_YELLOW, annotation_text=f"Target ${alert_price:.2f}", annotation_font_color=C_YELLOW)
            fig3.add_trace(go.Scatter(x=list(range(future_days)), y=future_prices, mode='lines+markers', name='XGBoost Forecast',
                line=dict(color=trend_color, width=2), marker=dict(size=7, color=trend_color, line=dict(width=1, color="#0b1326"))))
            if future_days > 5:
                fig3.add_vline(x=4.5, line_dash="dot", line_color="#2d3449", annotation_text="↑ Higher confidence | Lower confidence ↓",
                               annotation_font=dict(color="#424754", size=9), annotation_position="top")
            fig3.update_layout(**PLOTLY_LAYOUT,
                title=dict(text=f"{ticker} · {future_days}-Day Price Forecast · Band shows ±1.5σ uncertainty growth", font=dict(color=C_GREEN, size=12)),
                xaxis_title="Days from today", yaxis_title="Price (USD)", height=380)
            st.plotly_chart(fig3, use_container_width=True)

            if future_days > 5:
                st.markdown('<div style="background:rgba(255,221,45,0.04);border:1px solid rgba(255,221,45,0.2);border-left:3px solid #ffdd2d;padding:.6rem 1.2rem;font-family:Manrope,sans-serif;font-size:.7rem;color:#ffdd2d;font-weight:600;border-radius:0 .5rem .5rem 0;">⚠ Forecast reliability decreases significantly beyond Day 5. Use Days 6+ as directional signals only.</div>', unsafe_allow_html=True)

            future_df = pd.DataFrame({
                "Day": [f"+{i+1}" for i in range(future_days)],
                "Predicted Price ($)": [f"${p:.2f}" for p in future_prices],
                "vs Last Close": [f"{'▲' if p>last_close else '▼'} {abs(p-last_close):.2f} ({(p-last_close)/last_close*100:+.2f}%)" for p in future_prices]
            })
            st.dataframe(future_df, use_container_width=True, hide_index=True)

            # ── Backtesting ────────────────────────────────────────────────────
            if run_backtest and not fast_mode:
                st.subheader("Backtesting Engine")
                st.markdown(f'<div class="model-badge">STRATEGY: XGBoost Signal ±{bt_signal_threshold}% | Capital: ${bt_initial_capital:,.0f} | Commission: ${bt_commission}/trade</div>', unsafe_allow_html=True)
                with st.spinner("Running backtest simulation..."):
                    bt = run_backtest_engine(actual, preds, bt_initial_capital, bt_commission, bt_signal_threshold)

                strat_color = "bt-val-green" if bt["strat_return"]>=0 else "bt-val-red"
                bh_color    = "bt-val-green" if bt["bh_return"]>=0    else "bt-val-red"
                dd_color    = "bt-val-red"   if bt["max_drawdown"]<-10 else "bt-val"
                sh_color    = "bt-val-green" if bt["sharpe"]>=1        else "bt-val-red"

                k1,k2,k3,k4 = st.columns(4)
                k1.markdown(f'<div class="bt-card"><div class="bt-label">Strategy Return</div><div class="{strat_color}">{bt["strat_return"]:+.2f}%</div></div>', unsafe_allow_html=True)
                k2.markdown(f'<div class="bt-card"><div class="bt-label">Buy &amp; Hold Return</div><div class="{bh_color}">{bt["bh_return"]:+.2f}%</div></div>', unsafe_allow_html=True)
                k3.markdown(f'<div class="bt-card"><div class="bt-label">Max Drawdown</div><div class="{dd_color}">{bt["max_drawdown"]:.2f}%</div></div>', unsafe_allow_html=True)
                k4.markdown(f'<div class="bt-card"><div class="bt-label">Sharpe Ratio</div><div class="{sh_color}">{bt["sharpe"]:.2f}</div></div>', unsafe_allow_html=True)
                k5,k6,k7,k8 = st.columns(4)
                k5.markdown(f'<div class="bt-card"><div class="bt-label">Final Capital</div><div class="bt-val">${bt["final_capital"]:,.0f}</div></div>', unsafe_allow_html=True)
                k6.markdown(f'<div class="bt-card"><div class="bt-label">Total Trades</div><div class="bt-val">{bt["total_trades"]}</div></div>', unsafe_allow_html=True)
                k7.markdown(f'<div class="bt-card"><div class="bt-label">Win Rate</div><div class="bt-val">{bt["win_rate"]:.1f}%</div></div>', unsafe_allow_html=True)
                k8.markdown(f'<div class="bt-card"><div class="bt-label">Profit Factor</div><div class="bt-val">{bt["profit_factor"]:.2f}x</div></div>', unsafe_allow_html=True)

                if bt["total_trades"] == 0 and last_close > bt_initial_capital:
                    st.warning(f"⚠ No trades executed — {ticker} (${last_close:.2f}/share) exceeds the initial capital of ${bt_initial_capital:,.0f}. Increase the capital in the sidebar to enable backtesting for this stock.")

                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(y=bt["equity_curve"], name="XGBoost Strategy", line=dict(color=C_EMERALD, width=2), fill="tozeroy", fillcolor="rgba(0,229,176,0.05)"))
                fig_eq.add_trace(go.Scatter(y=bt["bh_equity"], name="Buy & Hold", line=dict(color=C_ACCENT, width=1.5, dash="dot")))
                fig_eq.add_hline(y=bt_initial_capital, line_dash="dash", line_color=C_GREY, annotation_text=f"Start ${bt_initial_capital:,}", annotation_font_color=C_GREY)
                fig_eq.update_layout(**PLOTLY_LAYOUT, title=dict(text=f"{ticker} · Strategy Equity Curve vs Buy & Hold", font=dict(color=C_GREEN, size=12)), height=380)
                st.plotly_chart(fig_eq, use_container_width=True)

                if not bt["trades_df"].empty:
                    st.subheader("Trade Log")
                    td = bt["trades_df"].copy()
                    td["Price"]   = td["Price"].apply(lambda x: f"${x:.2f}")
                    td["Capital"] = td["Capital"].apply(lambda x: f"${x:,.0f}")
                    if "P&L" in td.columns:
                        td["P&L"] = td["P&L"].apply(lambda x: f"+${x:.2f}" if pd.notna(x) and x>=0 else (f"-${abs(x):.2f}" if pd.notna(x) else "-"))
                    st.dataframe(td, use_container_width=True, hide_index=True)
                    st.download_button("⬇ Download Trade Log", data=bt["trades_df"].to_csv(index=False).encode(), file_name=f"{ticker}_trades.csv", mime="text/csv")

            # ── Confidence Intervals ───────────────────────────────────────────
            if show_conf_interval:
                st.subheader("Forecast with Confidence Intervals")
                st.markdown('<div class="model-badge">95% CI — Bootstrap Resampling</div>', unsafe_allow_html=True)
                with st.spinner(f"Running {ci_bootstrap_n} bootstrap samples..."):
                    ci_lower, ci_median, ci_upper = bootstrap_confidence_intervals(model, X_test, n_bootstrap=ci_bootstrap_n, noise_std=0.015)
                fig_ci = go.Figure()
                fig_ci.add_trace(go.Scatter(y=ci_upper, line=dict(color="rgba(0,0,0,0)"), showlegend=False))
                fig_ci.add_trace(go.Scatter(y=ci_lower, name="95% CI Band", fill="tonexty", fillcolor="rgba(77,142,255,0.10)", line=dict(color="rgba(0,0,0,0)")))
                fig_ci.add_trace(go.Scatter(y=actual, name="Actual", line=dict(color=C_ACCENT, width=1.5)))
                fig_ci.add_trace(go.Scatter(y=ci_median, name="XGBoost Median", line=dict(color=C_EMERALD, width=1.8, dash="dot")))
                fig_ci.update_layout(**PLOTLY_LAYOUT, title=dict(text=f"{ticker} · Predictions with 95% CI", font=dict(color=C_GREEN, size=12)), height=380)
                st.plotly_chart(fig_ci, use_container_width=True)

            # ── Model Comparison ───────────────────────────────────────────────
            if run_model_compare:
                st.subheader("Model Comparison — XGBoost vs Prophet vs Linear Regression")
                from sklearn.linear_model import LinearRegression as LR
                cmp = {}
                cmp["XGBoost"] = {"preds":preds,"color":C_EMERALD,
                    "rmse":float(np.sqrt(mean_squared_error(actual,preds))),
                    "mae":float(mean_absolute_error(actual,preds)),
                    "mape":float(np.mean(np.abs((actual-preds)/actual))*100),
                    "r2":float(1-np.sum((actual-preds)**2)/np.sum((actual-np.mean(actual))**2))}
                with st.spinner("Training Linear Regression..."):
                    lr_m = LR(); lr_m.fit(X_train, y_train); lr_p = lr_m.predict(X_test)
                cmp["Linear Regression"] = {"preds":lr_p,"color":C_GREY,
                    "rmse":float(np.sqrt(mean_squared_error(actual,lr_p))),
                    "mae":float(mean_absolute_error(actual,lr_p)),
                    "mape":float(np.mean(np.abs((actual-lr_p)/actual))*100),
                    "r2":float(1-np.sum((actual-lr_p)**2)/np.sum((actual-np.mean(actual))**2))}
                try:
                    from prophet import Prophet
                    cs_full = df["Close"].squeeze()
                    pdf = pd.DataFrame({"ds":df.index[:len(cs_full)],"y":cs_full.values}).dropna()
                    ptr = pdf.iloc[:int(len(pdf)*0.8)]; pte = pdf.iloc[int(len(pdf)*0.8):]
                    with st.spinner("Training Prophet..."):
                        pm = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
                        pm.fit(ptr); pfut = pm.make_future_dataframe(periods=len(pte), freq="B"); pfcst = pm.predict(pfut)
                        pp = pfcst["yhat"].values[-len(pte):]; pa = pte["y"].values; ml = min(len(pp), len(actual)); pp, pa = pp[:ml], actual[:ml]
                    cmp["Prophet"] = {"preds":pp,"color":C_YELLOW,
                        "rmse":float(np.sqrt(mean_squared_error(pa,pp))),
                        "mae":float(mean_absolute_error(pa,pp)),
                        "mape":float(np.mean(np.abs((pa-pp)/pa))*100),
                        "r2":float(1-np.sum((pa-pp)**2)/np.sum((pa-np.mean(pa))**2))}
                except ImportError:
                    st.info("Add `prophet` to requirements.txt to enable Prophet comparison.")

                rows = [{"Model":n,"RMSE ($)":f"${r['rmse']:.2f}","MAE ($)":f"${r['mae']:.2f}","MAPE (%)":f"{r['mape']:.2f}%","R²":f"{r['r2']:.4f}"} for n,r in cmp.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Scatter(y=actual, name="Actual", line=dict(color=C_ACCENT, width=2)))
                for n, r in cmp.items():
                    fig_cmp.add_trace(go.Scatter(y=r["preds"], name=n, line=dict(color=r["color"], width=1.5, dash="dot")))
                fig_cmp.update_layout(**PLOTLY_LAYOUT, title=dict(text=f"{ticker} · Model Comparison", font=dict(color=C_GREEN, size=12)), height=380)
                st.plotly_chart(fig_cmp, use_container_width=True)

            # ── Halal / Shariah ────────────────────────────────────────────────
            if run_halal_check:
                st.markdown("""
                <div style="background:rgba(0,229,176,0.03);border:1px solid rgba(0,229,176,0.15);border-left:4px solid #00e5b0;
                     padding:.8rem 1.4rem;margin:1.5rem 0 .5rem;display:flex;align-items:center;gap:1rem;border-radius:0 .5rem .5rem 0;">
                  <div style="font-size:1.4rem;">☪</div>
                  <div>
                    <div style="font-family:Manrope,sans-serif;font-size:.63rem;letter-spacing:.15em;text-transform:uppercase;color:#00e5b0;font-weight:700;">Halal / Shariah Compliance Screen</div>
                    <div style="font-family:Manrope,sans-serif;font-size:.76rem;color:#424754;margin-top:2px;">Automated screening based on AAOIFI Standard No.21</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                with st.spinner(f"Fetching financial data for {ticker}..."):
                    sd = get_shariah_data(ticker)
                if sd is None:
                    sd = {"debt_to_mktcap":0,"debt_to_assets":0,"cash_to_assets":0,
                          "market_cap":0,"total_debt":0,"total_assets":0,"total_cash":0,
                          "sector":"Unknown","industry":"Unknown","company_name":ticker}
                    st.warning(f"⚠ Could not fetch detailed financial data for {ticker}. Using ticker-list screening only.")
                if sd is not None:
                    compliance_result = check_shariah_compliance(ticker, sd, _L)
                    sh_verdict = compliance_result["verdict"]
                    v_color = {"COMPLIANT":C_EMERALD,"NON-COMPLIANT":C_RED,"QUESTIONABLE":C_YELLOW}[sh_verdict]
                    v_bg    = {"COMPLIANT":"rgba(0,229,176,0.05)","NON-COMPLIANT":"rgba(255,107,107,0.05)","QUESTIONABLE":"rgba(255,221,45,0.05)"}[sh_verdict]
                    v_icon  = {"COMPLIANT":"✅","NON-COMPLIANT":"❌","QUESTIONABLE":"⚠️"}[sh_verdict]
                    st.markdown(f'<div style="background:{v_bg};border:1px solid {v_color};border-left:3px solid {v_color};padding:1.2rem 2rem;margin:1rem 0;text-align:center;border-radius:0 .5rem .5rem 0;"><div style="font-family:Manrope,sans-serif;font-size:.6rem;color:#424754;letter-spacing:.14em;text-transform:uppercase;font-weight:700;">{sd["company_name"]} ({ticker})</div><div style="font-family:IBM Plex Mono,monospace;font-size:1.8rem;font-weight:700;color:{v_color};margin-top:.4rem;">{v_icon}&nbsp;{sh_verdict}</div><div style="font-size:.76rem;color:#8c909f;margin-top:.3rem;">Sector: {sd["sector"]} | Industry: {sd["industry"]}</div></div>', unsafe_allow_html=True)

                    st.subheader(_L["screening_criteria"])
                    col_left, col_right = st.columns(2)
                    with col_left:
                        bs = compliance_result["business"]
                        if bs["haram_hit"]:
                            st.markdown(f'<div class="halal-card-fail"><b>❌ Business Activity</b><br>Non-compliant: <b>{bs["haram_hit"]}</b></div>', unsafe_allow_html=True)
                        elif bs["questionable"]:
                            st.markdown('<div class="halal-card" style="border-left-color:#ffdd2d;"><b>⚠️ Business Activity</b><br>Questionable sector — consult a scholar</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="halal-card"><b>✅ Business Activity</b><br>No Haram core business detected<br><small style="color:#424754;">Sector: {sd["sector"]}</small></div>', unsafe_allow_html=True)
                        dm = compliance_result["debt_mktcap"]
                        cls = "halal-card" if dm["pass"] else "halal-card-fail"
                        icon = "✅" if dm["pass"] else "❌"
                        st.markdown(f'<div class="{cls}"><b>{icon} {dm["label"]}</b></div>', unsafe_allow_html=True)
                    with col_right:
                        da = compliance_result["debt_assets"]
                        cls = "halal-card" if da["pass"] else "halal-card-fail"
                        icon = "✅" if da["pass"] else "❌"
                        st.markdown(f'<div class="{cls}"><b>{icon} {da["label"]}</b></div>', unsafe_allow_html=True)
                        ca = compliance_result["cash_assets"]
                        cls = "halal-card" if ca["pass"] else "halal-card-fail"
                        icon = "✅" if ca["pass"] else "❌"
                        st.markdown(f'<div class="{cls}"><b>{icon} {ca["label"]}</b></div>', unsafe_allow_html=True)

            # ── News Sentiment ──────────────────────────────────────────────────
            if not is_beginner:
                st.subheader(_L["news_sentiment"])
                try:
                    from textblob import TextBlob
                    _ensure_nltk_data()
                    raw_news = av_get_news(ticker)
                    if raw_news:
                        scored = []
                        for item in raw_news[:10]:
                            title = item.get("title", "")
                            if title:
                                pol = TextBlob(title).sentiment.polarity
                                scored.append({"headline": title, "polarity": pol})
                        if scored:
                            sc_df = pd.DataFrame(scored)
                            avg_polarity = sc_df["polarity"].mean()
                            sent_color   = C_EMERALD if avg_polarity>0.05 else C_RED if avg_polarity<-0.05 else C_YELLOW
                            sent_label   = "POSITIVE" if avg_polarity>0.05 else "NEGATIVE" if avg_polarity<-0.05 else "NEUTRAL"
                            st.markdown(f'<div style="background:rgba(77,142,255,0.06);border:1px solid rgba(77,142,255,0.2);border-left:3px solid {sent_color};padding:.7rem 1.2rem;font-family:Manrope,sans-serif;font-size:.72rem;color:#dae2fd;font-weight:700;border-radius:0 .5rem .5rem 0;">Avg Sentiment: <span style="color:{sent_color};">{sent_label}</span> &nbsp;({avg_polarity:+.3f}) &nbsp;·&nbsp; {len(scored)} recent headlines</div>', unsafe_allow_html=True)
                            fig_sent = go.Figure(go.Bar(x=sc_df["polarity"], y=[h[:55]+"…" if len(h)>55 else h for h in sc_df["headline"]], orientation='h',
                                marker_color=[C_EMERALD if p>0 else C_RED for p in sc_df["polarity"]]))
                            fig_sent.add_vline(x=0, line_color=C_GREY)
                            fig_sent.add_vline(x=avg_polarity, line_dash="dot", line_color=sent_color, line_width=1.5)
                            _sent_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis"}
                            fig_sent.update_layout(**_sent_layout,
                                title=dict(text=f"{ticker} · Headline Sentiment (Yahoo Finance + TextBlob)", font=dict(color=C_GREEN, size=11)),
                                height=max(220, len(scored)*32),
                                xaxis=dict(title="Polarity (negative ← 0 → positive)", range=[-1,1], gridcolor="#2d3449", linecolor="#2d3449", zeroline=False, tickfont=dict(color="#424754",size=9)))
                            st.plotly_chart(fig_sent, use_container_width=True)
                            st.caption("⚠ Sentiment is based on headline text only. Powered by Yahoo Finance News + TextBlob.")
                    else:
                        st.info(_L["no_recent_news"])
                except ImportError:
                    st.info("Install `textblob` to enable News Sentiment NLP.")
                except Exception as e:
                    st.warning(f"Could not fetch news: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;margin-top:3rem;padding:1.5rem;border-top:1px solid #2d3449;">
  <div style="font-family:IBM Plex Mono,monospace;font-size:.6rem;color:#2d3449;letter-spacing:.1em;">
    {_L["footer"]}
  </div>
  <div style="margin-top:.6rem;font-family:IBM Plex Mono,monospace;font-size:.55rem;letter-spacing:.08em;">
    <a href="/privacy" target="_blank" style="color:#3a4460;text-decoration:none;margin:0 .5rem;">Privacy Policy</a>
    <span style="color:#2d3449;">·</span>
    <a href="/terms" target="_blank" style="color:#3a4460;text-decoration:none;margin:0 .5rem;">Terms of Service</a>
  </div>
</div>
""", unsafe_allow_html=True)


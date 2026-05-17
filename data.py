# =============================================================================
# Copyright (c) 2026 Stockcast. All Rights Reserved.
# PROPRIETARY AND CONFIDENTIAL
# data.py — yfinance helpers, market data, Fear & Greed, ticker search
# =============================================================================

import time
import logging
import requests
import streamlit as st
import yfinance as yf
import pandas as pd

from config import logger, POPULAR_TICKERS

# ── Core download helper ──────────────────────────────────────────────────────

def _yf_download_with_retry(ticker, retries=3, **kwargs):
    """Download yfinance data with retry logic for rate limiting."""
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
        logger.warning("yfinance download failed for '%s' after %d attempts: %s: %s",
                       ticker, retries, type(last_exc).__name__, last_exc)
    return pd.DataFrame()


# ── OHLCV & quote ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def av_get_daily(ticker):
    """Fetch full daily OHLCV via yfinance. Returns DataFrame."""
    df = _yf_download_with_retry(ticker, period="max", interval="1d")
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_localize(None) if idx.tz is not None else idx
    df.index.name = "Date"
    return df.sort_index()


@st.cache_data(ttl=300)
def get_ticker_full(ticker: str) -> dict:
    """Single cached yfinance get_info() call shared by quote and overview functions."""
    for attempt in range(3):
        try:
            info = yf.Ticker(ticker).get_info() or {}
            if info.get("symbol") or info.get("longName"):
                return info
            if attempt < 2:
                time.sleep(2 + attempt * 2)
        except Exception as e:
            logger.warning("get_ticker_full attempt %d failed for '%s': %s", attempt + 1, ticker, e)
            if attempt < 2:
                time.sleep(2 + attempt * 2)
    return {}


@st.cache_data(ttl=60)
def av_get_quote(ticker):
    """Fetch live quote — derived from get_ticker_full to avoid duplicate yfinance calls."""
    try:
        info       = get_ticker_full(ticker)
        price      = float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("navPrice") or 0)
        prev_close = float(info.get("previousClose") or info.get("regularMarketPreviousClose") or 0)
        open_price = float(info.get("open") or info.get("regularMarketOpen") or 0)
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
        return {"price": price, "change_pct": change_pct, "prev_close": prev_close, "open": open_price}
    except Exception as e:
        logger.warning("av_get_quote failed for '%s': %s", ticker, e)
        return {"price": 0.0, "change_pct": 0.0, "prev_close": 0.0, "open": 0.0}


@st.cache_data(ttl=3600)
def av_get_overview(ticker_sym):
    """Fetch company overview from cached yfinance info."""
    try:
        info = get_ticker_full(ticker_sym)
        if not info:
            logger.warning("av_get_overview: empty info for '%s'", ticker_sym)
            return {}
        return {
            "Symbol":   ticker_sym,
            "Name":     info.get("longName", ticker_sym),
            "Sector":   info.get("sector", "Unknown"),
            "Industry": info.get("industry", "Unknown"),
            "MarketCapitalization":                  str(info.get("marketCap", 0) or 0),
            "TotalDebt":                             str(info.get("totalDebt", 0) or 0),
            "TotalAssets":                           str(info.get("totalAssets", 0) or 0),
            "CashAndCashEquivalentsAtCarryingValue": str(info.get("totalCash", 0) or 0),
        }
    except Exception as e:
        logger.error("av_get_overview failed for '%s': %s", ticker_sym, e, exc_info=True)
        return {}


@st.cache_data(ttl=3600)
def get_fundamentals_rich(ticker_sym: str) -> dict:
    """Return enriched fundamental data for the valuation panel."""
    try:
        info = get_ticker_full(ticker_sym)
        if not info:
            return {}

        def _f(k, d=None):
            v = info.get(k, d)
            try: return float(v) if v not in (None, "None", "", "N/A") else d
            except: return d

        target_mean   = _f("targetMeanPrice")
        target_high   = _f("targetHighPrice")
        target_low    = _f("targetLowPrice")
        current_price = _f("currentPrice") or _f("regularMarketPrice") or 0
        upside_pct    = ((target_mean - current_price) / current_price * 100) if target_mean and current_price else None

        return {
            "name":            info.get("longName", ticker_sym),
            "sector":          info.get("sector", "Unknown"),
            "industry":        info.get("industry", "Unknown"),
            "description":     (info.get("longBusinessSummary") or "")[:400],
            "current_price":   current_price,
            "target_mean":     target_mean,
            "target_high":     target_high,
            "target_low":      target_low,
            "upside_pct":      upside_pct,
            "num_analysts":    int(info.get("numberOfAnalystOpinions") or 0),
            "recommendation":  info.get("recommendationKey", ""),
            "rev_growth_yoy":  _f("revenueGrowth"),
            "earn_growth_yoy": _f("earningsGrowth"),
            "eps_ttm":         _f("trailingEps"),
            "eps_fwd":         _f("forwardEps"),
            "pe_trailing":     _f("trailingPE"),
            "pe_forward":      _f("forwardPE"),
            "pb":              _f("priceToBook"),
            "ps_ttm":          _f("priceToSalesTrailing12Months"),
            "peg":             _f("pegRatio"),
            "ev_ebitda":       _f("enterpriseToEbitda"),
            "roe":             _f("returnOnEquity"),
            "roa":             _f("returnOnAssets"),
            "profit_margin":   _f("profitMargins"),
            "gross_margin":    _f("grossMargins"),
            "op_margin":       _f("operatingMargins"),
            "div_yield":       _f("dividendYield"),
            "payout_ratio":    _f("payoutRatio"),
            "beta":            _f("beta"),
            "float_shares":    _f("floatShares"),
            "short_ratio":     _f("shortRatio"),
            "short_pct_float": _f("shortPercentOfFloat"),
            "insider_pct":     _f("heldPercentInsiders"),
            "inst_pct":        _f("heldPercentInstitutions"),
        }
    except Exception as e:
        logger.error("get_fundamentals_rich failed for '%s': %s", ticker_sym, e)
        return {}


# ── News ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def av_get_news(ticker):
    """Fetch news headlines via yfinance."""
    try:
        news = yf.Ticker(ticker).news or []
        results = []
        for n in news[:10]:
            title = n.get("title") or (n.get("content") or {}).get("title") or ""
            if title:
                results.append({"title": title})
        return results
    except Exception as e:
        logger.warning("av_get_news failed for '%s': %s", ticker, e)
        return []


# ── Live market data ──────────────────────────────────────────────────────────

@st.cache_data(ttl=180)
def get_live_ticker_tape():
    """Fetch live prices for ticker tape symbols — single batched download."""
    tape_syms = ["AAPL","TSLA","NVDA","MSFT","GOOGL","META","AMZN","AMD","JPM","SPY","QQQ","NFLX"]
    try:
        raw   = yf.download(tape_syms, period="2d", interval="1d", progress=False, auto_adjust=True)
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
            except Exception as e:
                logger.debug("Ticker tape: skipping %s — %s", sym, e)
                continue
        return items  # ✅ FIX: return items on success
    except Exception as e:
        logger.warning("get_live_ticker_tape batch download failed: %s", e)
        return []


@st.cache_data(ttl=120)
def get_live_market_indices():
    """Fetch live S&P 500, NASDAQ, DOW, VIX via yfinance."""
    symbols = {"S&P 500": "^GSPC", "NASDAQ 100": "^NDX", "DOW JONES": "^DJI", "VIX": "^VIX"}
    result  = []
    try:
        raw   = yf.download(list(symbols.values()), period="2d", interval="1d", progress=False, auto_adjust=True)
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
                chg_pct   = ((price - prev) / prev * 100) if prev else 0.0
                col       = "#00e5b0" if chg_pct >= 0 else "#ff5f5f"
                sign      = "+" if chg_pct >= 0 else ""
                fmt_price = f"{price:,.2f}" if sym != "^VIX" else f"{price:.2f}"
                result.append((name, fmt_price, f"{sign}{chg_pct:.2f}%", col))
            except Exception as e:
                logger.debug("Market indices: no data for %s (%s): %s", name, sym, e)
                result.append((name, "—", "—", "#3e4558"))
    except Exception as e:
        logger.warning("get_live_market_indices batch download failed: %s", e)
        for name in symbols:
            result.append((name, "—", "—", "#3e4558"))
    return result


@st.cache_data(ttl=300)
def get_live_sector_heatmap():
    """Fetch live sector ETF performance via yfinance."""
    sector_etfs = {
        "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
        "Energy": "XLE", "Consumer Disc.": "XLY", "Industrials": "XLI",
        "Utilities": "XLU", "Real Estate": "XLRE", "Materials": "XLB", "Comm. Services": "XLC",
    }
    result = []
    try:
        raw   = yf.download(list(sector_etfs.values()), period="2d", interval="1d", progress=False, auto_adjust=True)
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
                col     = "#00e5b0" if chg_pct >= 0 else "#ff5f5f"
                sign    = "+" if chg_pct >= 0 else ""
                result.append((name, f"{sign}{chg_pct:.2f}%", col))
            except Exception as e:
                logger.debug("Sector heatmap: no data for %s (%s): %s", name, sym, e)
                result.append((name, "—", "#3e4558"))
    except Exception as e:
        logger.warning("get_live_sector_heatmap batch download failed: %s", e)
        for name in sector_etfs:
            result.append((name, "—", "#3e4558"))
    return result


@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Fetch Fear & Greed index. Tries CNN → alternative.me → VIX estimate."""
    try:
        r = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            timeout=8, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        )
        if r.status_code == 200 and "application/json" in r.headers.get("Content-Type", ""):
            data   = r.json()
            score  = float(data["fear_and_greed"]["score"])
            rating = data["fear_and_greed"]["rating"].replace("_", " ").title()
            return {"score": score, "rating": rating, "source": "CNN"}
    except Exception as e:
        logger.debug("get_fear_greed_index CNN failed: %s", e)

    try:
        r2 = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        if r2.status_code == 200:
            d2     = r2.json()["data"][0]
            score  = float(d2["value"])
            rating = d2["value_classification"].title()
            return {"score": score, "rating": rating, "source": "Alt.me"}
    except Exception as e:
        logger.debug("get_fear_greed_index alternative.me failed: %s", e)

    try:
        vix_df = _yf_download_with_retry("^VIX", period="2d", interval="1d")
        if not vix_df.empty:
            vix = float(vix_df["Close"].dropna().iloc[-1])
            if   vix < 12: score, rating = 80, "Extreme Greed"
            elif vix < 16: score, rating = 65, "Greed"
            elif vix < 20: score, rating = 52, "Neutral"
            elif vix < 28: score, rating = 35, "Fear"
            else:          score, rating = 18, "Extreme Fear"
            return {"score": score, "rating": rating, "source": "VIX estimate"}
    except Exception as e:
        logger.debug("get_fear_greed_index VIX fallback failed: %s", e)

    return None


# ── Ticker search ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def search_tickers(query: str) -> list:
    """Search POPULAR_TICKERS dict first, then probe yfinance for unknown tickers."""
    q  = query.strip().upper()
    ql = query.strip().lower()
    results = []
    if q in POPULAR_TICKERS:
        results.append(f"{q} — {POPULAR_TICKERS[q]}")
    for sym, name in POPULAR_TICKERS.items():
        if sym != q and (ql in name.lower() or ql in sym.lower()):
            results.append(f"{sym} — {name}")
    if not results and len(q) >= 1:
        try:
            info      = yf.Ticker(q).get_info() or {}
            long_name = info.get("longName") or info.get("shortName")
            if long_name:
                results.append(f"{q} — {long_name}")
        except Exception:
            pass
    return results[:10]


@st.cache_data(ttl=300)
def fetch_data(ticker: str, start, end) -> pd.DataFrame:
    """Fetch and date-filter OHLCV data for a ticker."""
    parts  = ticker.strip().split(".")
    ticker = parts[0].upper() + ("." + parts[1].upper() if len(parts) > 1 else "")
    df     = av_get_daily(ticker)
    if df.empty:
        return df
    _start = pd.to_datetime(start)
    _end   = pd.to_datetime(end)
    _start = _start.tz_localize(None) if _start.tzinfo else _start
    _end   = _end.tz_localize(None)   if _end.tzinfo   else _end
    _idx   = df.index.tz_localize(None) if df.index.tz else df.index
    return df[(_idx >= _start) & (_idx <= _end)]

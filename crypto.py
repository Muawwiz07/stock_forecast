"""
crypto.py  –  Stockcast Crypto Analysis Module
Full cryptocurrency analysis page mirroring the stock analysis engine.
app.py calls: render_crypto_page(supabase_client)

Supports:
- Live crypto prices via yfinance (BTC-USD, ETH-USD, etc.)
- Candlestick + volume charts
- RSI, MACD, Bollinger Bands
- XGBoost price forecast with confidence intervals
- BUY/SELL/HOLD composite signal (crypto-tuned)
- Fear & Greed Index (crypto-specific via alternative.me API)
- On-chain style market dominance chart
- Crypto-specific ticker tape
- News sentiment via yfinance headlines
"""

import streamlit as st
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import warnings
import time
warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────

C_GREEN   = "#adc6ff"
C_ACCENT  = "#4d8eff"
C_RED     = "#ff6b6b"
C_YELLOW  = "#ffdd2d"
C_GREY    = "#8c909f"
C_EMERALD = "#00e5b0"
C_ORANGE  = "#ff9f40"
C_PURPLE  = "#b48eff"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0b1326",
    plot_bgcolor="#0b1326",
    font=dict(family="Manrope", color="#8c909f", size=10),
    xaxis=dict(gridcolor="#424754", linecolor="#424754",
               tickfont=dict(color="#8c909f", size=10), showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#424754", linecolor="#424754",
               tickfont=dict(color="#8c909f", size=10), showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#424754", borderwidth=1, font=dict(size=10)),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#131b2e", bordercolor="#2d3449",
                    font=dict(family="Manrope", size=11, color="#dae2fd")),
)

POPULAR_CRYPTOS = {
    "BTC-USD":  "Bitcoin",
    "ETH-USD":  "Ethereum",
    "BNB-USD":  "BNB",
    "SOL-USD":  "Solana",
    "XRP-USD":  "XRP",
    "ADA-USD":  "Cardano",
    "AVAX-USD": "Avalanche",
    "DOGE-USD": "Dogecoin",
    "DOT-USD":  "Polkadot",
    "MATIC-USD":"Polygon",
    "LINK-USD": "Chainlink",
    "LTC-USD":  "Litecoin",
    "UNI-USD":  "Uniswap",
    "ATOM-USD": "Cosmos",
    "XLM-USD":  "Stellar",
    "ALGO-USD": "Algorand",
    "FIL-USD":  "Filecoin",
    "NEAR-USD": "NEAR Protocol",
    "APT-USD":  "Aptos",
    "ARB-USD":  "Arbitrum",
    "OP-USD":   "Optimism",
    "SUI-USD":  "Sui",
    "TRX-USD":  "TRON",
    "SHIB-USD": "Shiba Inu",
    "PEPE-USD": "Pepe",
    "WLD-USD":  "Worldcoin",
}

TAPE_CRYPTOS = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD",
                "ADA-USD","DOGE-USD","AVAX-USD","MATIC-USD","LINK-USD"]

FEATURE_COLS = [
    'MA5','MA10','MA20','MA50','MA200','EMA12','EMA26',
    'RSI','MACD','MACD_Signal','MACD_Hist',
    'BB_Width','BB_Pct','Returns','Returns_5d','Volatility','Momentum',
    'Volume_Ratio','High_Low_Pct','Close_Open_Pct','ATR'
]

# ── Data helpers ────────────────────────────────────────────────────────────────

def _yf_retry(ticker, retries=3, **kwargs):
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
    return pd.DataFrame()


@st.cache_data(ttl=300)
def crypto_get_daily(symbol):
    """Fetch full daily OHLCV for a crypto symbol via yfinance."""
    df = _yf_retry(symbol, period="max", interval="1d")
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_localize(None) if idx.tz is not None else idx
    df.index.name = "Date"
    return df.sort_index()


@st.cache_data(ttl=60)
def crypto_get_quote(symbol):
    """Live quote for a crypto symbol."""
    try:
        info = yf.Ticker(symbol).fast_info
        price      = float(getattr(info, "last_price", 0) or 0)
        prev_close = float(getattr(info, "previous_close", 0) or 0)
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
        return {"price": price, "change_pct": change_pct, "prev_close": prev_close}
    except Exception:
        return {"price": 0.0, "change_pct": 0.0, "prev_close": 0.0}


@st.cache_data(ttl=180)
def crypto_tape():
    """Fetch live prices for the crypto ticker tape."""
    try:
        raw = yf.download(TAPE_CRYPTOS, period="2d", interval="1d",
                          progress=False, auto_adjust=True)
        close = raw["Close"] if "Close" in raw.columns else raw
        if isinstance(close.columns, pd.MultiIndex):
            close = close.droplevel(0, axis=1)
        items = []
        for sym in TAPE_CRYPTOS:
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
                label   = sym.replace("-USD", "")
                # Format price smartly
                if price >= 1000:
                    fmt = f"${price:,.0f}"
                elif price >= 1:
                    fmt = f"${price:,.2f}"
                else:
                    fmt = f"${price:.6f}"
                items.append((label, fmt, f"{sign}{chg_pct:.2f}%", arrow, css))
            except Exception:
                continue
        return items
    except Exception:
        return []


@st.cache_data(ttl=3600)
def crypto_fear_greed():
    """Fetch Crypto Fear & Greed Index from alternative.me."""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = r.json()
        entry = data["data"][0]
        return {"score": int(entry["value"]), "rating": entry["value_classification"]}
    except Exception:
        return None


@st.cache_data(ttl=300)
def crypto_dominance():
    """Fetch BTC + ETH dominance from CoinGecko global endpoint."""
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        d = r.json()["data"]["market_cap_percentage"]
        btc = round(d.get("btc", 0), 1)
        eth = round(d.get("eth", 0), 1)
        others = round(100 - btc - eth, 1)
        return {"BTC": btc, "ETH": eth, "Others": others}
    except Exception:
        return None


@st.cache_data(ttl=300)
def crypto_get_news(symbol):
    """Fetch news via yfinance for a crypto ticker."""
    try:
        news = yf.Ticker(symbol).news or []
        results = []
        for n in news[:10]:
            title = (n.get("title") or (n.get("content") or {}).get("title") or "")
            if title:
                results.append({"title": title})
        return results
    except Exception:
        return []


# ── Technical analysis helpers ──────────────────────────────────────────────────

def _rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.rolling(period).mean()
    avg_l = loss.rolling(period).mean().replace(0, 1e-10)
    return 100 - (100 / (1 + avg_g / avg_l))


def _macd(series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    m  = ef - es
    s  = m.ewm(span=signal, adjust=False).mean()
    return m, s, m - s


def _bollinger(series, period=20, std=2):
    sma = series.rolling(period).mean()
    rs  = series.rolling(period).std()
    return sma + std * rs, sma, sma - std * rs


def add_features(df):
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()
    df["MA5"]   = close.rolling(5).mean()
    df["MA10"]  = close.rolling(10).mean()
    df["MA20"]  = close.rolling(20).mean()
    df["MA50"]  = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()
    df["RSI"]   = _rsi(close)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = _macd(close)
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"]   = _bollinger(close)
    df["BB_Width"]       = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
    df["BB_Pct"]         = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    df["Returns"]        = close.pct_change()
    df["Returns_5d"]     = close.pct_change(5)
    df["Volatility"]     = df["Returns"].rolling(20).std()
    df["Momentum"]       = close - close.shift(10)
    df["Volume_MA10"]    = volume.rolling(10).mean()
    df["Volume_Ratio"]   = volume / df["Volume_MA10"]
    df["High_Low_Pct"]   = (high - low) / close
    df["Close_Open_Pct"] = (close - df["Open"].squeeze()) / df["Open"].squeeze()
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    return df


def build_dataset(df, seq_len):
    close   = df["Close"].squeeze().values
    feat_df = df[FEATURE_COLS].copy()
    feat_df["Close"] = close
    X_rows, y_rows = [], []
    for i in range(seq_len, len(feat_df) - 1):
        row_feats  = feat_df[FEATURE_COLS].iloc[i].values
        lag_closes = close[i - seq_len:i]
        X_rows.append(np.concatenate([row_feats, lag_closes]))
        y_rows.append(close[i + 1])
    X = np.array(X_rows)
    y = np.array(y_rows)
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return X[mask], y[mask]


def bootstrap_ci(model, X_input, n_bootstrap=100, noise_std=0.015):
    fs = np.std(X_input, axis=0, keepdims=True)
    fs = np.where(fs == 0, 1.0, fs)
    all_preds = [
        model.predict(X_input + np.random.normal(0, noise_std, X_input.shape) * fs)
        for _ in range(n_bootstrap)
    ]
    a = np.array(all_preds)
    return np.percentile(a, 5, axis=0), np.percentile(a, 50, axis=0), np.percentile(a, 95, axis=0)


def crypto_composite_signal(df, last_close, forecast_price):
    """Crypto-tuned composite signal (wider RSI thresholds, higher volatility tolerance)."""
    close  = df["Close"].squeeze()
    rsi    = float(df["RSI"].squeeze().iloc[-1])
    macd   = float(df["MACD"].squeeze().iloc[-1])
    macd_s = float(df["MACD_Signal"].squeeze().iloc[-1])
    macd_h = float(df["MACD_Hist"].squeeze().iloc[-1])
    bb_pct = float(df["BB_Pct"].squeeze().iloc[-1])
    ma50   = float(df["MA50"].squeeze().iloc[-1])
    ma200  = float(df["MA200"].squeeze().iloc[-1])
    vol_r  = float(df["Volume_Ratio"].squeeze().iloc[-1])
    atr    = float(df["ATR"].squeeze().iloc[-1])

    signals = {}
    xgb_pct = (forecast_price - last_close) / last_close * 100

    # XGBoost signal — crypto moves faster so 2% threshold
    if   xgb_pct >  2.0: signals["XGBoost Forecast"] = ("BUY",  min(35, abs(xgb_pct) * 5), xgb_pct, "positive")
    elif xgb_pct < -2.0: signals["XGBoost Forecast"] = ("SELL", -min(35, abs(xgb_pct) * 5), xgb_pct, "negative")
    else:                 signals["XGBoost Forecast"] = ("HOLD", 0, xgb_pct, "neutral")

    # RSI — crypto uses wider oversold/overbought zones
    if   rsi < 25: signals["RSI (14)"] = ("BUY",  25, rsi, "positive")
    elif rsi > 75: signals["RSI (14)"] = ("SELL", -25, rsi, "negative")
    elif rsi < 40: signals["RSI (14)"] = ("BUY",   8, rsi, "positive")
    elif rsi > 60: signals["RSI (14)"] = ("SELL",  -8, rsi, "negative")
    else:          signals["RSI (14)"] = ("HOLD",   0, rsi, "neutral")

    # MACD crossover
    prev_hist = float(df["MACD_Hist"].squeeze().iloc[-2]) if len(df) > 2 else 0
    if   macd_h > 0 and prev_hist <= 0: signals["MACD Cross"] = ("BUY",  20, macd_h, "positive")
    elif macd_h < 0 and prev_hist >= 0: signals["MACD Cross"] = ("SELL", -20, macd_h, "negative")
    elif macd > macd_s:                 signals["MACD Cross"] = ("BUY",  10, macd_h, "positive")
    elif macd < macd_s:                 signals["MACD Cross"] = ("SELL", -10, macd_h, "negative")
    else:                               signals["MACD Cross"] = ("HOLD",   0, macd_h, "neutral")

    # Bollinger %B
    if   bb_pct < 0.05: signals["Bollinger %B"] = ("BUY",  12, bb_pct, "positive")
    elif bb_pct > 0.95: signals["Bollinger %B"] = ("SELL", -12, bb_pct, "negative")
    else:               signals["Bollinger %B"] = ("HOLD",   0, bb_pct, "neutral")

    # MA Golden/Death cross
    if   ma50 > ma200 and close.iloc[-1] > ma50: signals["MA Cross"] = ("BUY",  15, ma50 - ma200, "positive")
    elif ma50 < ma200 and close.iloc[-1] < ma50: signals["MA Cross"] = ("SELL", -15, ma50 - ma200, "negative")
    else:                                         signals["MA Cross"] = ("HOLD",   0, ma50 - ma200, "neutral")

    # Volume confirmation — crypto volume spikes are very significant
    if   vol_r > 2.0 and xgb_pct > 0: signals["Volume"] = ("BUY",  15, vol_r, "positive")
    elif vol_r > 2.0 and xgb_pct < 0: signals["Volume"] = ("SELL", -15, vol_r, "negative")
    elif vol_r > 1.3 and xgb_pct > 0: signals["Volume"] = ("BUY",   8, vol_r, "positive")
    elif vol_r > 1.3 and xgb_pct < 0: signals["Volume"] = ("SELL",  -8, vol_r, "negative")
    else:                              signals["Volume"] = ("HOLD",   0, vol_r, "neutral")

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
        "signals": signals, "verdict": verdict, "verdict_short": verdict_short,
        "total_score": total_score, "xgb_pct": xgb_pct, "rsi": rsi,
        "stop_loss": stop_loss, "take_profit": take_profit, "risk_reward": risk_reward,
        "vol_ratio": vol_r, "atr": atr,
    }


def _fmt_price(price):
    """Smart price formatter for crypto (handles $0.000001 to $100k+)."""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:,.4f}"
    elif price >= 0.01:
        return f"${price:.6f}"
    else:
        return f"${price:.8f}"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def render_crypto_page():
    """
    Full Crypto Analysis page. Call this from app.py inside a tab or
    conditional block. Returns nothing; renders directly into Streamlit.
    """

    # ── Page header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(90deg,#0b1326 0%,#131b2e 100%);
         border-bottom:1px solid #2d3449;border-left:4px solid #f7931a;
         padding:1rem 2rem;margin:0 -2rem 1.5rem -2rem;
         display:flex;align-items:center;justify-content:space-between;
         box-shadow:0 4px 24px rgba(0,0,0,0.3);">
      <div>
        <div style="font-family:Manrope,sans-serif;font-size:1.3rem;font-weight:800;color:#dae2fd;">
          Crypto<span style="color:#f7931a;">cast</span>
        </div>
        <div style="font-size:.6rem;color:#424754;letter-spacing:.1em;text-transform:uppercase;font-weight:700;margin-top:2px;">
          XGBoost · 6-Factor Signals · Crypto Fear &amp; Greed · 24/7 Market
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:1rem;">
        <span style="display:inline-block;width:7px;height:7px;background:#f7931a;border-radius:50%;
              animation:pulse-dot 2s infinite;box-shadow:0 0 8px rgba(247,147,26,0.5);"></span>
        <span style="font-family:IBM Plex Mono,monospace;font-size:.6rem;color:#f7931a;letter-spacing:.1em;">
          24/7 · GLOBAL CRYPTO MARKETS
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Crypto ticker tape ──────────────────────────────────────────────────────
    _tape = crypto_tape()
    if _tape:
        _dot = '<span style="color:#2d3449;">·</span>'
        _spans = f" {_dot} ".join(
            f'<span><span class="tape-sym">{sym}</span>'
            f'<span class="{css}">{arrow} {price} {pct}</span></span>'
            for sym, price, pct, arrow, css in _tape * 2
        )
        st.markdown(f"""
<div class="ticker-tape-wrap">
  <div class="ticker-tape">{_spans}</div>
</div>""", unsafe_allow_html=True)

    # ── Sidebar controls for crypto ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="font-family:Manrope,sans-serif;font-size:.65rem;font-weight:800;
             letter-spacing:.12em;text-transform:uppercase;color:#f7931a;margin:.6rem 0 .4rem;">
          ₿ Crypto Settings
        </div>""", unsafe_allow_html=True)

        # Symbol picker
        crypto_search = st.text_input("🔍 Search Crypto", placeholder="e.g. Bitcoin, BTC, ETH",
                                      key="crypto_search_input").strip()
        crypto_sym = "BTC-USD"
        if crypto_search:
            q = crypto_search.upper()
            # match by symbol or name
            matches = [f"{k} — {v}" for k, v in POPULAR_CRYPTOS.items()
                       if q in k or q in v.upper()]
            if not matches:
                # try appending -USD
                test_sym = q if q.endswith("-USD") else q + "-USD"
                matches = [f"{test_sym} — Custom"]
            sel = st.selectbox("Select", matches, key="crypto_sym_sel",
                               label_visibility="collapsed")
            crypto_sym = sel.split(" — ")[0].strip()
        else:
            sel_name = st.selectbox("Coin", list(POPULAR_CRYPTOS.keys()),
                                    format_func=lambda x: f"{x}  —  {POPULAR_CRYPTOS[x]}",
                                    key="crypto_sym_box")
            crypto_sym = sel_name

        st.markdown(f"""
        <div style="background:rgba(247,147,26,0.08);border:1px solid rgba(247,147,26,0.25);
             border-left:3px solid #f7931a;padding:.35rem .9rem;
             font-family:IBM Plex Mono,monospace;font-size:.68rem;
             color:#f7931a;letter-spacing:.05em;margin:.3rem 0;border-radius:0 .5rem .5rem 0;">
          ● ACTIVE: {crypto_sym}
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1: c_start = st.date_input("From", value=pd.to_datetime("2020-01-01"), key="c_start")
        with c2: c_end   = st.date_input("To",   value=pd.Timestamp.today(), key="c_end")

        st.markdown('<div style="font-family:Manrope,sans-serif;font-size:.58rem;color:#8c909f;letter-spacing:.1em;text-transform:uppercase;font-weight:700;margin-top:.4rem;">Lookback Window (days)</div>', unsafe_allow_html=True)
        c_seq_len     = st.slider("", 10, 60, 30, key="c_seq", label_visibility="collapsed")
        st.markdown('<div style="font-family:Manrope,sans-serif;font-size:.58rem;color:#8c909f;letter-spacing:.1em;text-transform:uppercase;font-weight:700;">Forecast Horizon (days)</div>', unsafe_allow_html=True)
        c_future_days = st.slider(" ", 1, 14, 7, key="c_fut", label_visibility="collapsed")

        st.markdown("---")
        c_mode = st.radio("Mode", ["🟢 Beginner", "🔴 Pro"], index=1,
                          horizontal=True, key="c_mode", label_visibility="collapsed")
        c_is_beginner = (c_mode == "🟢 Beginner")

        if not c_is_beginner:
            st.markdown('<div style="font-family:Manrope,sans-serif;font-size:.58rem;color:#8c909f;letter-spacing:.1em;text-transform:uppercase;font-weight:700;">XGBoost Hyperparameters</div>', unsafe_allow_html=True)
            c_trees = st.slider("Trees", 100, 500, 200, step=50, key="c_trees")
            c_depth = st.slider("Max Depth", 2, 8, 4, key="c_depth")
            c_lr    = st.select_slider("Learning Rate",
                                       options=[0.01, 0.05, 0.1, 0.2], value=0.05, key="c_lr")
            c_show_ci = st.checkbox("Confidence Intervals", value=True, key="c_ci")
            c_ci_n    = st.slider("Bootstrap Samples", 50, 200, 100, step=50, key="c_ci_n") if c_show_ci else 100
        else:
            c_trees, c_depth, c_lr = 200, 4, 0.05
            c_show_ci, c_ci_n = False, 100

        st.markdown("---")
        if st.button("▶  Run Crypto Forecast", use_container_width=True, key="crypto_run_btn"):
            st.session_state.crypto_run = True
        c_run = st.session_state.get("crypto_run", False)

        if c_run:
            if st.button("← Back to Crypto Dashboard", use_container_width=True, key="crypto_back_btn"):
                st.session_state.crypto_run = False
                st.rerun()

    # ── Dashboard (no forecast yet) ────────────────────────────────────────────
    if not c_run:
        _render_crypto_dashboard()
        return

    # ══════════════════════════════════════════════════════════════════════════
    # FORECAST ENGINE
    # ══════════════════════════════════════════════════════════════════════════

    # Fetch data
    with st.spinner(f"Fetching {crypto_sym} data…"):
        df_raw = crypto_get_daily(crypto_sym)

    if df_raw.empty:
        st.error(f"⚠ No data found for '{crypto_sym}'. Check the symbol (e.g. BTC-USD, ETH-USD).")
        return

    # Date filter
    df = df_raw[(df_raw.index >= pd.to_datetime(c_start)) &
                (df_raw.index <= pd.to_datetime(c_end))].copy()

    if df.empty:
        st.error("⚠ No data in the selected date range.")
        return

    st.success(f"✓ {len(df)} trading days loaded for {crypto_sym}  ({POPULAR_CRYPTOS.get(crypto_sym, '')})")

    # Reality check banner
    last_close_raw = float(df["Close"].squeeze().iloc[-1])
    st.markdown(f"""
    <div style="background:rgba(247,147,26,0.04);border:1px solid rgba(247,147,26,0.3);
         border-left:4px solid #f7931a;padding:.9rem 1.4rem;margin:.5rem 0 1rem;border-radius:0 .5rem .5rem 0;">
      <div style="font-family:Manrope,sans-serif;font-size:.6rem;letter-spacing:.14em;
           text-transform:uppercase;color:#f7931a;margin-bottom:.3rem;font-weight:700;">
        ⚠ Crypto Model Reality Check — Read Before Trading
      </div>
      <div style="font-family:Manrope,sans-serif;font-size:.78rem;color:#8c909f;line-height:1.6;">
        Crypto markets are <b style="color:#dae2fd;">open 24/7</b> and subject to extreme volatility.
        This model uses <b style="color:#dae2fd;">price & volume data only</b> — it has
        <b style="color:#ff6b6b;">zero awareness</b> of: &nbsp;📰 protocol news &nbsp;·&nbsp;
        🏦 exchange hacks &nbsp;·&nbsp; 🐋 whale movements &nbsp;·&nbsp;
        📊 on-chain metrics &nbsp;·&nbsp; 🌍 regulatory events.
        <b style="color:#ffdd2d;">Use signals as one input — never as sole decision.</b>
      </div>
    </div>""", unsafe_allow_html=True)

    # Tabs
    tab_analysis, tab_market, tab_methodology = st.tabs([
        "📊  Analysis", "🌍  Crypto Market", "📖  Methodology"
    ])

    # ── Methodology tab ─────────────────────────────────────────────────────────
    with tab_methodology:
        _render_crypto_methodology(c_seq_len)

    # ── Market overview tab ─────────────────────────────────────────────────────
    with tab_market:
        _render_crypto_market_tab()

    # ── Analysis tab ────────────────────────────────────────────────────────────
    with tab_analysis:

        # Add features
        with st.spinner("Engineering crypto features…"):
            df = add_features(df)

        close_series = df["Close"].squeeze()

        # ── 1. Candlestick + Volume ────────────────────────────────────────────
        st.subheader("Price Chart")
        fig_c = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.72, 0.28], vertical_spacing=0.02)
        fig_c.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"].squeeze(), high=df["High"].squeeze(),
            low=df["Low"].squeeze(), close=close_series,
            name="Price", increasing_line_color=C_EMERALD,
            decreasing_line_color=C_RED), row=1, col=1)
        fig_c.add_trace(go.Scatter(x=df.index, y=df["MA50"].squeeze(),
            name="MA50",  line=dict(color=C_YELLOW, width=1.2)), row=1, col=1)
        fig_c.add_trace(go.Scatter(x=df.index, y=df["MA200"].squeeze(),
            name="MA200", line=dict(color=C_ACCENT,  width=1.2)), row=1, col=1)
        fig_c.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"].squeeze(),
            name="BB Upper", line=dict(color=C_GREY, width=0.8, dash="dot")), row=1, col=1)
        fig_c.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"].squeeze(),
            name="BB Lower", line=dict(color=C_GREY, width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(247,147,26,0.05)"), row=1, col=1)
        vol_colors = [C_EMERALD if c >= o else C_RED
                      for c, o in zip(close_series, df["Open"].squeeze())]
        fig_c.add_trace(go.Bar(x=df.index, y=df["Volume"].squeeze(),
            name="Volume", marker_color=vol_colors, opacity=0.5), row=2, col=1)
        cl = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")}
        fig_c.update_layout(**cl,
            title=dict(text=f"{crypto_sym} · Candlestick · MA50/200 · Bollinger · Volume",
                       font=dict(color="#f7931a", size=12)),
            xaxis_rangeslider_visible=False, height=560)
        fig_c.update_xaxes(gridcolor="#2d3449", linecolor="#2d3449", tickfont=dict(color=C_GREY))
        fig_c.update_yaxes(gridcolor="#2d3449", linecolor="#2d3449", tickfont=dict(color=C_GREY))
        st.plotly_chart(fig_c, use_container_width=True)

        # ── 2. RSI + MACD ──────────────────────────────────────────────────────
        st.subheader("Technical Indicators")
        fig_t = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.5, 0.5], vertical_spacing=0.08,
                              subplot_titles=["RSI (14)", "MACD (12/26/9)"])
        fig_t.add_trace(go.Scatter(x=df.index, y=df["RSI"].squeeze(),
            name="RSI", line=dict(color="#f7931a", width=1.5)), row=1, col=1)
        fig_t.add_hline(y=75, line_dash="dash", line_color=C_RED,     row=1, col=1)
        fig_t.add_hline(y=25, line_dash="dash", line_color=C_EMERALD, row=1, col=1)
        fig_t.add_hrect(y0=75, y1=100, fillcolor="rgba(255,107,107,0.04)", line_width=0, row=1, col=1)
        fig_t.add_hrect(y0=0,  y1=25,  fillcolor="rgba(0,229,176,0.04)",   line_width=0, row=1, col=1)
        fig_t.add_trace(go.Scatter(x=df.index, y=df["MACD"].squeeze(),
            name="MACD",   line=dict(color=C_ACCENT,  width=1.2)), row=2, col=1)
        fig_t.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"].squeeze(),
            name="Signal", line=dict(color=C_GREEN,   width=1.2)), row=2, col=1)
        mh = df["MACD_Hist"].squeeze()
        fig_t.add_trace(go.Bar(x=df.index, y=mh, name="Histogram",
            marker_color=[C_EMERALD if v >= 0 else C_RED for v in mh],
            opacity=0.65), row=2, col=1)
        sl = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")}
        fig_t.update_layout(**sl, height=450)
        fig_t.update_xaxes(gridcolor="#2d3449", linecolor="#2d3449", tickfont=dict(color=C_GREY))
        fig_t.update_yaxes(gridcolor="#2d3449", linecolor="#2d3449", tickfont=dict(color=C_GREY))
        fig_t.update_yaxes(range=[0, 100], row=1, col=1)
        st.plotly_chart(fig_t, use_container_width=True)

        # ── 3. XGBoost Model ───────────────────────────────────────────────────
        st.markdown("""
        <div style="background:rgba(247,147,26,0.06);border:1px solid rgba(247,147,26,0.2);
             border-left:3px solid #f7931a;padding:.45rem 1rem;margin-bottom:.8rem;
             font-family:IBM Plex Mono,monospace;font-size:.62rem;color:#f7931a;letter-spacing:.1em;">
          🤖 MODEL: XGBoost Regressor · 20 Technical Features + Lag Window · Crypto-Tuned Signals
        </div>""", unsafe_allow_html=True)

        with st.spinner("Building feature matrix…"):
            X, y = build_dataset(df, c_seq_len)

        if len(X) < 50:
            st.error("Not enough data. Try a longer date range or smaller lookback window.")
            return

        split   = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        @st.cache_resource(show_spinner=False)
        def _train(Xtr, ytr, Xte, yte, ne, md, lr):
            m = XGBRegressor(n_estimators=ne, max_depth=md, learning_rate=lr,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbosity=0)
            m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
            return m

        with st.spinner("Training XGBoost model (cached after first run)…"):
            model = _train(X_train, y_train, X_test, y_test,
                           c_trees, c_depth, c_lr)

        preds  = model.predict(X_test)
        actual = y_test
        rmse   = float(np.sqrt(mean_squared_error(actual, preds)))
        mae    = float(mean_absolute_error(actual, preds))
        mape   = float(np.mean(np.abs((actual - preds) / actual)) * 100)
        r2     = float(1 - np.sum((actual - preds)**2) /
                       np.sum((actual - np.mean(actual))**2))

        # Confidence score
        r2_norm   = max(0, min(100, r2 * 100))
        mape_norm = max(0, min(100, 100 - mape * 5))
        dir_acc   = sum(1 for i in range(1, len(actual))
                        if (preds[i] - actual[i-1]) * (actual[i] - actual[i-1]) > 0) / max(len(actual)-1, 1) * 100
        data_score = min(100, len(df) / 2000 * 100)
        confidence = max(0, min(100, r2_norm * 0.40 + mape_norm * 0.30 +
                                     dir_acc * 0.20 + data_score * 0.10))
        last_close = float(df["Close"].squeeze().iloc[-1])

        # Model KPIs
        st.subheader("Model Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE",  _fmt_price(rmse))
        m2.metric("MAE",   _fmt_price(mae))
        m3.metric("MAPE",  f"{mape:.2f}%")
        m4.metric("R²",    f"{r2:.4f}")

        mape_label = ("🟢 Excellent" if mape < 3 else "🟡 Good" if mape < 7
                      else "🟠 Fair" if mape < 15 else "🔴 Poor")
        r2_label   = ("🟢 Excellent" if r2 > 0.95 else "🟡 Good" if r2 > 0.85
                      else "🟠 Fair" if r2 > 0.70 else "🔴 Poor")
        st.markdown(
            f'<div style="background:#131b2e;border:1px solid #2d3449;padding:.65rem 1.2rem;'
            f'font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#424754;'
            f'display:flex;gap:2rem;flex-wrap:wrap;border-radius:.5rem;">'
            f'<span>MAPE: {mape_label} · &lt;3% excellent · &lt;7% good · &lt;15% fair</span>'
            f'<span>R²: {r2_label} · &gt;0.95 excellent · &gt;0.85 good · &gt;0.70 fair</span>'
            f'</div>', unsafe_allow_html=True)

        # Analysis sub-tabs
        dash_t, signal_t, forecast_t = st.tabs(["🖥  Dashboard", "🧠  Signal Intelligence", "📈  Deep Forecast"])

        # ── Dashboard sub-tab ──────────────────────────────────────────────────
        with dash_t:
            _prev  = float(df["Close"].squeeze().iloc[-2]) if len(df) > 1 else last_close
            _chg   = last_close - _prev
            _pct   = (_chg / _prev * 100) if _prev else 0
            _sign  = "+" if _chg >= 0 else ""
            _col   = C_EMERALD if _chg >= 0 else C_RED
            _arrow = "▲" if _chg >= 0 else "▼"

            st.markdown(f"""
            <div class="stat-grid">
              <div class="stat-card" style="border-top-color:#f7931a;">
                <div class="stat-label">Last Close</div>
                <div class="stat-value" style="color:#f7931a;">{_fmt_price(last_close)}</div>
                <div class="stat-sub" style="color:{_col};font-weight:700;">
                  {_arrow} {_sign}{_fmt_price(abs(_chg))} ({_sign}{_pct:.2f}%)
                </div>
              </div>
              <div class="stat-card" style="border-top-color:#adc6ff;">
                <div class="stat-label">Model Confidence</div>
                <div class="stat-value" style="color:#adc6ff;">
                  {confidence:.0f}<span style="font-size:.9rem;color:#8c909f;">/100</span>
                </div>
                <div class="stat-sub">{"High" if confidence>=80 else "Moderate" if confidence>=60 else "Low"}</div>
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
            </div>""", unsafe_allow_html=True)

            # Actual vs Predicted
            st.subheader("Actual vs Predicted")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(y=actual, name="Actual",
                line=dict(color=C_ACCENT, width=1.5),
                fill="tozeroy", fillcolor="rgba(77,142,255,0.05)"))
            fig1.add_trace(go.Scatter(y=preds, name="XGBoost",
                line=dict(color="#f7931a", width=1.5, dash="dot")))
            fig1.update_layout(**PLOTLY_LAYOUT,
                title=dict(text=f"{crypto_sym} · XGBoost Model Fit (Test Set)",
                           font=dict(color="#f7931a", size=12)),
                height=350)
            st.plotly_chart(fig1, use_container_width=True)

            # Feature Importance
            st.subheader("Feature Importance")
            lag_names = [f"Lag_{i+1}" for i in range(c_seq_len)]
            all_feats = FEATURE_COLS + lag_names
            imp_df = (pd.DataFrame({"feature": all_feats,
                                    "importance": model.feature_importances_})
                      .sort_values("importance", ascending=True).tail(20))
            fig_imp = go.Figure(go.Bar(
                x=imp_df["importance"], y=imp_df["feature"], orientation="h",
                marker=dict(color=imp_df["importance"],
                            colorscale=[[0, "#131b2e"], [0.5, "#2a1800"], [1, "#f7931a"]],
                            showscale=False)))
            fig_imp.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis"},
                title=dict(text="Top 20 Feature Importances",
                           font=dict(color="#f7931a", size=12)),
                height=430,
                xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Importance Score"))
            st.plotly_chart(fig_imp, use_container_width=True)

        # ── Signal Intelligence sub-tab ─────────────────────────────────────────
        with signal_t:
            # Forward forecast for signal
            close_vals = df["Close"].squeeze().values
            feat_vals  = df[FEATURE_COLS].values
            last_seq   = close_vals[-c_seq_len:]
            last_feat  = feat_vals[-1]
            row        = np.concatenate([last_feat, last_seq]).reshape(1, -1)
            forecast_price = float(model.predict(row)[0])

            sig_result = crypto_composite_signal(df, last_close, forecast_price)
            signals    = sig_result["signals"]
            verdict    = sig_result["verdict"]
            score      = sig_result["total_score"]
            xgb_pct    = sig_result["xgb_pct"]

            v_color = (C_EMERALD if "BUY" in verdict else C_RED if "SELL" in verdict else C_YELLOW)
            v_bg    = ("rgba(0,229,176,0.05)" if "BUY" in verdict
                       else "rgba(255,107,107,0.05)" if "SELL" in verdict
                       else "rgba(255,221,45,0.05)")

            st.markdown(f"""
            <div style="background:{v_bg};border:1px solid {v_color};
                 border-left:4px solid {v_color};padding:1.4rem 2rem;
                 margin:1rem 0;text-align:center;border-radius:0 .5rem .5rem 0;">
              <div style="font-family:Manrope,sans-serif;font-size:.6rem;color:#424754;
                   letter-spacing:.14em;text-transform:uppercase;font-weight:700;">
                {crypto_sym} · {POPULAR_CRYPTOS.get(crypto_sym, "")} · Composite Signal
              </div>
              <div style="font-family:IBM Plex Mono,monospace;font-size:2rem;
                   font-weight:700;color:{v_color};margin:.4rem 0;">{verdict}</div>
              <div style="font-size:.76rem;color:#8c909f;margin-top:.3rem;">
                Score: {score:+d} · XGBoost: {xgb_pct:+.2f}% ·
                Next-day target: {_fmt_price(forecast_price)}
              </div>
            </div>""", unsafe_allow_html=True)

            # Stop loss / Take profit
            sl_c1, sl_c2, sl_c3 = st.columns(3)
            sl_c1.metric("Take Profit", _fmt_price(sig_result["take_profit"]))
            sl_c2.metric("Stop Loss",   _fmt_price(sig_result["stop_loss"]))
            sl_c3.metric("Risk/Reward", f"{sig_result['risk_reward']:.2f}x")

            # 6-factor breakdown
            st.subheader("6-Factor Signal Breakdown")
            factor_color = {
                "positive": C_EMERALD,
                "negative": C_RED,
                "neutral":  C_YELLOW,
            }
            for fname, (action, sc, val, sentiment) in signals.items():
                fc = factor_color[sentiment]
                bar_w = min(100, abs(sc) / 35 * 100)
                bar_dir = "left" if sc >= 0 else "right"
                icon = ("▲" if action == "BUY" else "▼" if action == "SELL" else "◆")
                st.markdown(f"""
                <div style="background:#131b2e;border:1px solid #2d3449;
                     border-left:3px solid {fc};padding:.7rem 1.1rem;
                     margin-bottom:.4rem;border-radius:0 .5rem .5rem 0;">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="font-family:Manrope,sans-serif;font-size:.68rem;
                         font-weight:700;color:#dae2fd;letter-spacing:.04em;">{fname}</div>
                    <div style="font-family:IBM Plex Mono,monospace;font-size:.72rem;
                         color:{fc};font-weight:700;">{icon} {action} &nbsp;
                      <span style="color:#424754;font-weight:400;">({sc:+d})</span>
                    </div>
                  </div>
                  <div style="background:#0b1326;height:4px;border-radius:2px;
                       margin-top:.5rem;overflow:hidden;">
                    <div style="height:100%;width:{bar_w}%;background:{fc};
                         border-radius:2px;float:{bar_dir};"></div>
                  </div>
                  <div style="font-size:.62rem;color:#424754;margin-top:.3rem;
                       font-family:IBM Plex Mono,monospace;">
                    Value: {val:.4f}
                  </div>
                </div>""", unsafe_allow_html=True)

            # News Sentiment
            if not c_is_beginner:
                st.subheader("News Sentiment")
                try:
                    from textblob import TextBlob
                    raw_news = crypto_get_news(crypto_sym)
                    if raw_news:
                        scored = []
                        for item in raw_news[:10]:
                            title = item.get("title", "")
                            if title:
                                pol = TextBlob(title).sentiment.polarity
                                scored.append({"headline": title, "polarity": pol})
                        if scored:
                            sc_df      = pd.DataFrame(scored)
                            avg_pol    = sc_df["polarity"].mean()
                            sent_color = (C_EMERALD if avg_pol > 0.05
                                          else C_RED if avg_pol < -0.05 else C_YELLOW)
                            sent_label = ("POSITIVE" if avg_pol > 0.05
                                          else "NEGATIVE" if avg_pol < -0.05 else "NEUTRAL")
                            st.markdown(
                                f'<div style="background:rgba(77,142,255,0.06);border:1px solid '
                                f'rgba(77,142,255,0.2);border-left:3px solid {sent_color};'
                                f'padding:.7rem 1.2rem;font-family:Manrope,sans-serif;font-size:.72rem;'
                                f'color:#dae2fd;font-weight:700;border-radius:0 .5rem .5rem 0;">'
                                f'Avg Sentiment: <span style="color:{sent_color};">{sent_label}</span>'
                                f' &nbsp;({avg_pol:+.3f}) &nbsp;·&nbsp; {len(scored)} headlines</div>',
                                unsafe_allow_html=True)
                            _sl = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis"}
                            fig_s = go.Figure(go.Bar(
                                x=sc_df["polarity"],
                                y=[h[:55] + "…" if len(h) > 55 else h
                                   for h in sc_df["headline"]],
                                orientation="h",
                                marker_color=[C_EMERALD if p > 0 else C_RED
                                              for p in sc_df["polarity"]]))
                            fig_s.add_vline(x=0, line_color=C_GREY)
                            fig_s.update_layout(**_sl,
                                title=dict(text=f"{crypto_sym} · Headline Sentiment",
                                           font=dict(color="#f7931a", size=11)),
                                height=max(220, len(scored) * 32),
                                xaxis=dict(title="Polarity (negative ← 0 → positive)",
                                           range=[-1, 1], gridcolor="#2d3449",
                                           linecolor="#2d3449", zeroline=False,
                                           tickfont=dict(color="#424754", size=9)))
                            st.plotly_chart(fig_s, use_container_width=True)
                    else:
                        st.info("No recent news found for this coin.")
                except ImportError:
                    st.info("Install `textblob` to enable News Sentiment.")
                except Exception as e:
                    st.warning(f"Could not fetch news: {e}")

        # ── Deep Forecast sub-tab ───────────────────────────────────────────────
        with forecast_t:
            st.subheader(f"Forecast — Next {c_future_days} Days")

            # Rolling multi-step forecast
            close_vals = df["Close"].squeeze().values
            feat_vals  = df[FEATURE_COLS].values
            future_preds = []
            seq_buf      = list(close_vals[-c_seq_len:])
            feat_buf     = feat_vals[-1].copy()

            for _ in range(c_future_days):
                row  = np.concatenate([feat_buf, seq_buf[-c_seq_len:]]).reshape(1, -1)
                pred = float(model.predict(row)[0])
                future_preds.append(pred)
                seq_buf.append(pred)

            last_date    = df.index[-1]
            future_dates = pd.bdate_range(last_date, periods=c_future_days + 1,
                                          freq="D")[1:]

            # Confidence intervals via bootstrap
            if c_show_ci and not c_is_beginner:
                with st.spinner(f"Running {c_ci_n} bootstrap samples for CI…"):
                    last_row = np.concatenate([feat_vals[-1], close_vals[-c_seq_len:]]).reshape(1, -1)
                    ci_lo, ci_med, ci_hi = bootstrap_ci(model, last_row, c_ci_n)
                ci_available = True
            else:
                ci_available = False

            # Build forecast figure
            hist_n  = min(120, len(df))
            fig_f   = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=df.index[-hist_n:], y=close_series.values[-hist_n:],
                name="Historical", line=dict(color=C_ACCENT, width=1.5)))
            fig_f.add_trace(go.Scatter(
                x=future_dates, y=future_preds,
                name=f"{c_future_days}-Day Forecast",
                line=dict(color="#f7931a", width=2, dash="dot"),
                mode="lines+markers",
                marker=dict(size=6, color="#f7931a", symbol="circle")))

            if ci_available:
                fig_f.add_trace(go.Scatter(
                    x=[future_dates[0]], y=[ci_hi[0]], name="95% CI Upper",
                    line=dict(color="rgba(247,147,26,0.3)", width=0),
                    showlegend=False))
                fig_f.add_trace(go.Scatter(
                    x=[future_dates[0]], y=[ci_lo[0]], name="95% CI",
                    fill="tonexty", fillcolor="rgba(247,147,26,0.08)",
                    line=dict(color="rgba(247,147,26,0.3)", width=0.8, dash="dot")))

            fig_f.add_vline(x=str(last_date), line_dash="dash",
                            line_color=C_GREY, line_width=1)
            fig_f.update_layout(**PLOTLY_LAYOUT,
                title=dict(text=f"{crypto_sym} · {c_future_days}-Day Price Forecast",
                           font=dict(color="#f7931a", size=12)),
                height=420)
            st.plotly_chart(fig_f, use_container_width=True)

            # Forecast table
            st.markdown('<div style="font-family:Manrope,sans-serif;font-size:.6rem;'
                        'letter-spacing:.14em;text-transform:uppercase;color:#8c909f;'
                        'font-weight:700;margin-bottom:.4rem;">Forecast Table</div>',
                        unsafe_allow_html=True)
            rows = []
            for i, (dt, p) in enumerate(zip(future_dates, future_preds)):
                chg     = p - last_close if i == 0 else p - future_preds[i - 1]
                chg_pct = chg / (last_close if i == 0 else future_preds[i-1]) * 100
                rows.append({
                    "Day":     f"Day {i+1}",
                    "Date":    dt.strftime("%b %d"),
                    "Forecast": _fmt_price(p),
                    "Change":  f"{'+'if chg>=0 else ''}{_fmt_price(abs(chg))}",
                    "% Δ":     f"{'+'if chg_pct>=0 else ''}{chg_pct:.2f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("⚠ Multi-step forecast compounds errors — Days 1–3 most reliable. "
                       "Days 7+ are directional only. Crypto is 24/7 — prices may gap overnight.")


# ══════════════════════════════════════════════════════════════════════════════
# CRYPTO DASHBOARD (no forecast)
# ══════════════════════════════════════════════════════════════════════════════

def _render_crypto_dashboard():
    st.markdown("""
    <div style="font-family:Manrope,sans-serif;font-size:2.2rem;font-weight:800;
         letter-spacing:-.02em;color:#dae2fd;margin-bottom:.3rem;">
      Crypto <span style="color:#f7931a;">Dashboard</span>
    </div>
    <div style="font-size:.82rem;color:#8c909f;margin-bottom:1.5rem;font-weight:500;">
      AI-powered crypto intelligence · Select a coin in the sidebar and click Run Forecast to begin.
    </div>""", unsafe_allow_html=True)

    # Fear & Greed
    fg = crypto_fear_greed()
    dom = crypto_dominance()

    fg_score  = fg["score"] if fg else 50
    fg_label  = fg["rating"] if fg else "Neutral"
    fg_color  = (C_EMERALD if fg_score >= 60 else C_RED if fg_score <= 30 else C_YELLOW)

    dom_btc    = dom["BTC"]    if dom else "—"
    dom_eth    = dom["ETH"]    if dom else "—"
    dom_others = dom["Others"] if dom else "—"

    st.markdown(f"""
    <div class="stat-grid" style="grid-template-columns:repeat(4,1fr);">
      <div class="stat-card" style="border-top-color:#f7931a;">
        <div class="stat-label">BTC Dominance</div>
        <div class="stat-value" style="color:#f7931a;">{dom_btc}%</div>
        <div class="stat-sub">of total crypto mkt cap</div>
      </div>
      <div class="stat-card" style="border-top-color:#adc6ff;">
        <div class="stat-label">ETH Dominance</div>
        <div class="stat-value" style="color:#adc6ff;">{dom_eth}%</div>
        <div class="stat-sub">of total crypto mkt cap</div>
      </div>
      <div class="stat-card" style="border-top-color:{fg_color};">
        <div class="stat-label">Crypto Fear &amp; Greed</div>
        <div class="stat-value" style="color:{fg_color};">{fg_score}</div>
        <div class="stat-sub">{fg_label}</div>
      </div>
      <div class="stat-card" style="border-top-color:#00e5b0;">
        <div class="stat-label">Altcoin Dominance</div>
        <div class="stat-value" style="color:#00e5b0;">{dom_others}%</div>
        <div class="stat-sub">combined alt market share</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Top coin live prices
    st.subheader("⭐ Top Coins — Live Prices")
    top_coins   = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD",
                   "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD"]
    coin_cols   = st.columns(4)
    for i, sym in enumerate(top_coins):
        with coin_cols[i % 4]:
            try:
                q    = crypto_get_quote(sym)
                px   = q["price"]
                chg  = q["change_pct"]
                col  = C_EMERALD if chg >= 0 else C_RED
                sign = "▲" if chg >= 0 else "▼"
                name = POPULAR_CRYPTOS.get(sym, sym.replace("-USD", ""))
                st.markdown(f"""
                <div style="background:linear-gradient(145deg,#131b2e,#171f33);
                     border:1px solid #2d3449;border-top:2px solid {col};
                     padding:1rem 1.2rem;text-align:center;border-radius:.5rem;
                     margin-bottom:.6rem;">
                  <div style="font-family:IBM Plex Mono,monospace;font-size:.62rem;
                       letter-spacing:.14em;color:#424754;text-transform:uppercase;">{name}</div>
                  <div style="font-family:IBM Plex Mono,monospace;font-size:1.1rem;
                       font-weight:700;color:#dae2fd;margin:.3rem 0;">{_fmt_price(px)}</div>
                  <div style="font-family:IBM Plex Mono,monospace;font-size:.72rem;color:{col};">
                    {sign} {chg:+.2f}%
                  </div>
                </div>""", unsafe_allow_html=True)
            except Exception:
                st.markdown(f'<div style="background:#131b2e;border:1px solid #2d3449;'
                            f'padding:1rem;text-align:center;font-family:IBM Plex Mono,monospace;'
                            f'font-size:.7rem;color:#424754;border-radius:.5rem;margin-bottom:.6rem;">'
                            f'{sym.replace("-USD","")}<br>—</div>', unsafe_allow_html=True)

    # How it works for crypto
    st.subheader("How It Works")
    hw1, hw2, hw3 = st.columns(3)
    for col, num, color, title, body in [
        (hw1, "01", "#f7931a",
         "Select a Coin",
         "Choose from 25+ cryptocurrencies — Bitcoin, Ethereum, Solana and more. "
         "Crypto tickers use the -USD suffix (e.g. BTC-USD)."),
        (hw2, "02", C_EMERALD,
         "Run the Model",
         "XGBoost trains on daily OHLCV data with 20 crypto-tuned features. "
         "Signals use wider RSI & volume thresholds to handle crypto's higher volatility."),
        (hw3, "03", C_YELLOW,
         "Read the Signal",
         "Get a BUY / SELL / HOLD verdict with Crypto Fear & Greed context, "
         "stop-loss, take-profit, and a multi-day price forecast."),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,#131b2e,#171f33);border:1px solid #2d3449;
                 border-top:2px solid {color};padding:1.4rem 1.5rem;height:100%;border-radius:.5rem;">
              <div style="font-family:IBM Plex Mono,monospace;font-size:1.3rem;font-weight:700;
                   color:{color};margin-bottom:.5rem;">{num}</div>
              <div style="font-family:Manrope,sans-serif;font-size:.68rem;letter-spacing:.1em;
                   text-transform:uppercase;color:#dae2fd;font-weight:700;margin-bottom:.5rem;">
                {title}
              </div>
              <div style="font-family:Manrope,sans-serif;font-size:.8rem;color:#8c909f;line-height:1.6;">
                {body}
              </div>
            </div>""", unsafe_allow_html=True)

    # Crypto vs Stock differences card
    st.markdown("""
    <div style="background:rgba(247,147,26,0.04);border:1px solid rgba(247,147,26,0.2);
         border-left:4px solid #f7931a;padding:1rem 1.5rem;margin-top:1.5rem;border-radius:0 .5rem .5rem 0;">
      <div style="font-family:Manrope,sans-serif;font-size:.63rem;letter-spacing:.14em;
           text-transform:uppercase;color:#f7931a;margin-bottom:.5rem;font-weight:700;">
        ₿ Crypto vs Stocks — Key Differences in This Model
      </div>
      <div style="font-family:Manrope,sans-serif;font-size:.82rem;color:#8c909f;line-height:1.8;">
        <b style="color:#dae2fd;">24/7 Markets</b> — No overnight gap protection; prices move while you sleep. &nbsp;·&nbsp;
        <b style="color:#dae2fd;">Higher Volatility</b> — RSI thresholds widened to 25/75 (vs 30/70 for stocks). &nbsp;·&nbsp;
        <b style="color:#dae2fd;">Volume Signals</b> — 2× volume threshold for confirmation (crypto pump signals). &nbsp;·&nbsp;
        <b style="color:#dae2fd;">No Shariah Screen</b> — Most major cryptos have no direct interest-bearing activity. &nbsp;·&nbsp;
        <b style="color:#dae2fd;">Crypto F&G Index</b> — Uses alternative.me index, not CNN (crypto-specific sentiment).
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CRYPTO MARKET TAB
# ══════════════════════════════════════════════════════════════════════════════

def _render_crypto_market_tab():
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
      <div style="font-family:Manrope,sans-serif;font-size:2rem;font-weight:800;
           letter-spacing:-.02em;color:#dae2fd;">
        Crypto <span style="color:#f7931a;">Market Intelligence</span>
      </div>
      <div style="font-size:.78rem;color:#8c909f;margin-top:.2rem;">
        Live Crypto Fear &amp; Greed · Market Dominance · Top Coin Performance
      </div>
    </div>""", unsafe_allow_html=True)

    col_fg, col_dom = st.columns([1, 1])

    # Fear & Greed gauge
    with col_fg:
        st.subheader("Crypto Fear & Greed Index · Live")
        fg = crypto_fear_greed()
        if fg:
            score  = fg["score"]
            label  = fg["rating"]
            color  = (C_EMERALD if score >= 60 else C_RED if score <= 30 else C_YELLOW)
            fig_fg = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                title={"text": label, "font": {"color": color, "size": 13,
                                               "family": "Manrope"}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"color": C_GREY, "size": 9}},
                    "bar":  {"color": color, "thickness": 0.25},
                    "bgcolor": "#131b2e",
                    "bordercolor": "#2d3449",
                    "steps": [
                        {"range": [0,  25], "color": "rgba(255,107,107,0.15)"},
                        {"range": [25, 45], "color": "rgba(255,107,107,0.05)"},
                        {"range": [45, 55], "color": "rgba(255,221,45,0.05)"},
                        {"range": [55, 75], "color": "rgba(0,229,176,0.05)"},
                        {"range": [75, 100], "color": "rgba(0,229,176,0.15)"},
                    ],
                    "threshold": {"line": {"color": color, "width": 2},
                                  "thickness": 0.75, "value": score},
                },
                number={"font": {"color": color, "size": 36, "family": "IBM Plex Mono"},
                        "suffix": " / 100"},
            ))
            fig_fg.update_layout(
                paper_bgcolor="#0b1326", plot_bgcolor="#0b1326",
                height=260, margin=dict(l=20, r=20, t=40, b=20),
                font=dict(family="Manrope", color=C_GREY))
            st.plotly_chart(fig_fg, use_container_width=True)
            st.markdown(
                f'<div style="text-align:center;font-family:IBM Plex Mono,monospace;'
                f'font-size:.62rem;color:#424754;margin-top:-.5rem;">'
                f'Source: alternative.me · Updates every hour</div>',
                unsafe_allow_html=True)
        else:
            st.info("Could not fetch Crypto Fear & Greed Index.")

    # Market Dominance donut
    with col_dom:
        st.subheader("Market Dominance · Live")
        dom = crypto_dominance()
        if dom:
            fig_dom = go.Figure(go.Pie(
                labels=list(dom.keys()),
                values=list(dom.values()),
                hole=0.6,
                marker_colors=["#f7931a", "#627eea", C_GREY],
                textfont_size=11, textfont_color="#dae2fd",
            ))
            fig_dom.update_layout(
                **PLOTLY_LAYOUT, height=260,
                showlegend=True,
                annotations=[dict(
                    text=f"BTC<br><span style='font-size:10px'>{dom['BTC']}%</span>",
                    x=0.5, y=0.5, font_size=16, showarrow=False,
                    font_color="#f7931a")])
            st.plotly_chart(fig_dom, use_container_width=True)
            st.markdown(
                '<div style="text-align:center;font-family:IBM Plex Mono,monospace;'
                'font-size:.62rem;color:#424754;margin-top:-.5rem;">'
                'Source: CoinGecko · Updates every 5 min</div>',
                unsafe_allow_html=True)
        else:
            st.info("Could not fetch dominance data.")

    # Top 10 performance heatmap
    st.subheader("Top Coins Performance · 24h")
    heat_syms  = list(POPULAR_CRYPTOS.keys())[:12]
    heat_items = []
    with st.spinner("Loading live coin data…"):
        try:
            raw = yf.download(heat_syms, period="2d", interval="1d",
                              progress=False, auto_adjust=True)
            close = raw["Close"] if "Close" in raw.columns else raw
            if isinstance(close.columns, pd.MultiIndex):
                close = close.droplevel(0, axis=1)
            for sym in heat_syms:
                try:
                    prices = close[sym].dropna()
                    if len(prices) >= 2:
                        p, prev = float(prices.iloc[-1]), float(prices.iloc[-2])
                    elif len(prices) == 1:
                        p = prev = float(prices.iloc[-1])
                    else:
                        continue
                    chg = ((p - prev) / prev * 100) if prev else 0
                    heat_items.append({
                        "symbol": sym.replace("-USD", ""),
                        "name":   POPULAR_CRYPTOS[sym],
                        "price":  p,
                        "chg":    chg,
                    })
                except Exception:
                    continue
        except Exception:
            pass

    if heat_items:
        heat_cols = st.columns(4)
        for i, coin in enumerate(heat_items):
            with heat_cols[i % 4]:
                col  = C_EMERALD if coin["chg"] >= 0 else C_RED
                sign = "▲" if coin["chg"] >= 0 else "▼"
                bg_alpha = min(0.25, abs(coin["chg"]) / 20)
                bg_r, bg_g, bg_b = (0, 229, 176) if coin["chg"] >= 0 else (255, 107, 107)
                st.markdown(f"""
                <div style="background:rgba({bg_r},{bg_g},{bg_b},{bg_alpha});
                     border:1px solid #2d3449;border-left:3px solid {col};
                     padding:.7rem 1rem;margin-bottom:.4rem;border-radius:0 .5rem .5rem 0;">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                      <div style="font-family:IBM Plex Mono,monospace;font-size:.75rem;
                           font-weight:700;color:#dae2fd;">{coin["symbol"]}</div>
                      <div style="font-size:.62rem;color:#424754;margin-top:1px;">{coin["name"][:16]}</div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-family:IBM Plex Mono,monospace;font-size:.72rem;
                           color:#dae2fd;">{_fmt_price(coin["price"])}</div>
                      <div style="font-family:IBM Plex Mono,monospace;font-size:.7rem;
                           color:{col};font-weight:700;">{sign} {coin["chg"]:+.2f}%</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Could not load live coin data.")


# ══════════════════════════════════════════════════════════════════════════════
# METHODOLOGY PAGE (crypto)
# ══════════════════════════════════════════════════════════════════════════════

def _render_crypto_methodology(seq_len_val=30):
    st.markdown("""
    <div style="font-family:Manrope,sans-serif;font-size:0.6rem;letter-spacing:.18em;
         text-transform:uppercase;color:#8c909f;margin-bottom:.3rem;font-weight:700;">
      Technical Documentation
    </div>
    <div style="font-family:Manrope,sans-serif;font-size:1.15rem;font-weight:800;
         color:#dae2fd;letter-spacing:-.01em;margin-bottom:1.4rem;">
      Cryptocast <span style="color:#f7931a;">·</span> Methodology & Crypto Adaptations
    </div>""", unsafe_allow_html=True)

    steps = [
        ("01", "#f7931a", "Data Ingestion", "Daily OHLCV via yfinance",
         "Daily crypto OHLCV data is fetched from Yahoo Finance using the -USD suffix "
         "(e.g. BTC-USD). Unlike stocks, crypto trades 24/7 so there are no weekend gaps; "
         "all calendar days are included. Timezone normalization is applied."),
        ("02", "#adc6ff", "Feature Engineering", "20 Technical Indicators (crypto-tuned)",
         f"Identical 20-feature set to the stock model: MA5/10/20/50/200, EMA12/26, RSI(14), "
         f"MACD, Bollinger Bands, ATR, Volume Ratio, Momentum, Returns, Volatility, High-Low%, "
         f"plus {seq_len_val} lag closes. No modifications needed — these indicators are "
         f"exchange-agnostic."),
        ("03", "#00e5b0", "Train/Test Split", "80% train · 20% test (chronological)",
         "Strict chronological split — no shuffling, no leakage. The model is evaluated "
         "exclusively on the held-out 20% test set."),
        ("04", "#f7931a", "XGBoost Regressor", "Gradient-boosted trees",
         "Same XGBoost architecture as the stock model. Crypto's higher raw volatility "
         "naturally produces higher RMSE values in dollar terms — use MAPE (%) for comparison."),
        ("05", "#adc6ff", "Signal Generation", "Crypto-Tuned BUY / SELL / HOLD",
         "RSI thresholds widened to 25/75 (crypto norm). Volume confirmation raised to 2× "
         "average (crypto pumps require stronger volume). XGBoost threshold raised to ±2% "
         "(vs ±1.5% for stocks). Score >+25 = STRONG BUY, <-25 = STRONG SELL."),
        ("06", "#00e5b0", "Crypto Fear & Greed", "alternative.me index",
         "Unlike the stock CNN F&G, crypto uses the alternative.me API which aggregates "
         "volatility, volume, social media, dominance, and Google Trends. Score 0–25 = "
         "Extreme Fear (often contrarian buy signal). Score 75–100 = Extreme Greed."),
        ("07", "#f7931a", "Forward Forecast", "Iterative multi-step prediction",
         "Identical rolling forecast: each prediction feeds as the next lag input. "
         "Crypto's higher intraday volatility means error compounds faster — "
         "Days 1–3 are most reliable; Days 7+ are directional only."),
        ("08", "#adc6ff", "No Shariah Screen", "N/A for crypto assets",
         "AAOIFI Standard No.21 does not directly apply to cryptocurrencies. "
         "Some scholars consider certain cryptos permissible; consult a qualified "
         "Islamic finance scholar for a ruling on specific assets."),
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
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(255,107,107,0.04);border:1px solid rgba(255,107,107,0.2);
         border-left:3px solid #ff6b6b;padding:1rem 1.5rem;margin-top:.5rem;
         border-radius:0 0.5rem 0.5rem 0;">
      <div style="font-family:Manrope,sans-serif;font-size:0.63rem;letter-spacing:.14em;
           text-transform:uppercase;color:#ff6b6b;margin-bottom:.4rem;font-weight:700;">
        ⚠ Crypto-Specific Limitations
      </div>
      <div style="font-family:Manrope,sans-serif;font-size:0.82rem;color:#8c909f;line-height:1.7;">
        Crypto markets are open <b style="color:#dae2fd;">24 hours, 7 days a week</b> — this model
        uses daily candles only, so intraday swings are not captured.
        Regulatory announcements, exchange hacks, whale movements, and protocol upgrades
        can cause instant -30% to +100% moves that <b style="color:#ff6b6b;">no technical model
        can predict</b>. Always use a stop-loss.
        <b style="color:#ff6b6b;">This is a research and educational tool — not financial advice.</b>
      </div>
    </div>""", unsafe_allow_html=True)

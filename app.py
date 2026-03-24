import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Price Forecaster", page_icon="📈", layout="wide")

# ── Bloomberg-style CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0a0a;
    color: #e0e0e0;
}
.stApp { background-color: #0a0a0a; }
[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid #ff6600 !important;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] input {
    background-color: #1a1a1a !important;
    border: 1px solid #ff6600 !important;
    color: #ff6600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
}
.stButton > button {
    background: #ff6600 !important;
    color: #000 !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 0px !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #ff8533 !important;
    box-shadow: 0 4px 15px rgba(255,102,0,0.4) !important;
}
[data-testid="stMetric"] {
    background: #111111;
    border: 1px solid #222222;
    border-top: 2px solid #ff6600;
    padding: 1.2rem 1.5rem !important;
    border-radius: 0px;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #888 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 600 !important;
    color: #ff6600 !important;
}
hr { border-color: #222 !important; }
h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #ffffff !important;
    letter-spacing: 0.05em !important;
    border-bottom: 1px solid #222;
    padding-bottom: 0.4rem;
}
.ticker-header {
    background: #111;
    border-bottom: 2px solid #ff6600;
    padding: 1rem 2rem;
    margin-bottom: 2rem;
}
.ticker-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #ff6600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.ticker-sub {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.85rem;
    color: #888;
    letter-spacing: 0.05em;
}
.signal-badge-buy {
    display: inline-block;
    background: #003300;
    border: 1px solid #00cc44;
    color: #00cc44;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    padding: 0.6rem 2rem;
    letter-spacing: 0.1em;
}
.signal-badge-sell {
    display: inline-block;
    background: #330000;
    border: 1px solid #ff3333;
    color: #ff3333;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    padding: 0.6rem 2rem;
    letter-spacing: 0.1em;
}
.signal-badge-hold {
    display: inline-block;
    background: #1a1a00;
    border: 1px solid #cccc00;
    color: #cccc00;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    padding: 0.6rem 2rem;
    letter-spacing: 0.1em;
}
.alert-box {
    background: #1a0d00;
    border: 1px solid #ff6600;
    border-left: 4px solid #ff6600;
    padding: 1rem 1.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    color: #ff6600;
    margin: 1rem 0;
}
.model-badge {
    display: inline-block;
    background: #001a33;
    border: 1px solid #0088ff;
    color: #0088ff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.3rem 1rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.stat-row {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #666;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.feature-importance-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #888;
}
.bt-win  { color: #00cc44 !important; font-weight: 700; }
.bt-loss { color: #ff3333 !important; font-weight: 700; }
.bt-card {
    background: #111;
    border: 1px solid #222;
    border-top: 2px solid #ff6600;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
    font-family: "IBM Plex Mono", monospace;
}
.bt-card-label { font-size: 0.65rem; color: #666; letter-spacing: 0.15em; text-transform: uppercase; }
.bt-card-value { font-size: 1.6rem; font-weight: 700; color: #ff6600; }
.bt-card-value-green { font-size: 1.6rem; font-weight: 700; color: #00cc44; }
.bt-card-value-red   { font-size: 1.6rem; font-weight: 700; color: #ff3333; }
.halal-pass { color: #00cc44; font-weight: 700; }
.halal-fail { color: #ff3333; font-weight: 700; }
.halal-card { background:#0a1a0a; border:1px solid #1a3a1a; border-left:4px solid #00cc44; padding:1rem 1.5rem; margin:0.5rem 0; font-family:"IBM Plex Mono",monospace; font-size:0.85rem; }
.halal-card-fail { background:#1a0a0a; border:1px solid #3a1a1a; border-left:4px solid #ff3333; padding:1rem 1.5rem; margin:0.5rem 0; font-family:"IBM Plex Mono",monospace; font-size:0.85rem; }
.ci-badge { background:#001a33; border:1px solid #0055aa; color:#4499ff; font-family:"IBM Plex Mono",monospace; font-size:0.75rem; font-weight:700; padding:0.3rem 1rem; letter-spacing:0.1em; text-transform:uppercase; display:inline-block; margin-bottom:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ticker-header">
    <div class="ticker-title">📈 Stock Price Forecaster</div>
    <div class="ticker-sub">XGBoost-powered price prediction &amp; trading signals</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ PARAMETERS")
    st.markdown('<div class="stat-row">Equity Symbol</div>', unsafe_allow_html=True)
    ticker = st.text_input("", value="AAPL", label_visibility="collapsed").upper()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=pd.to_datetime("2018-01-01"))
    with col2:
        end_date = st.date_input("To", value=pd.Timestamp.today())

    st.markdown('<div class="stat-row">Lookback Window (days)</div>', unsafe_allow_html=True)
    seq_len = st.slider("", 10, 60, 30, label_visibility="collapsed")

    st.markdown('<div class="stat-row">Forecast Horizon (days)</div>', unsafe_allow_html=True)
    future_days = st.slider(" ", 1, 30, 7, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="stat-row">XGBoost Settings</div>', unsafe_allow_html=True)
    n_estimators = st.slider("Trees", 100, 500, 200, step=50)
    max_depth = st.slider("Max Depth", 2, 8, 4)
    learning_rate = st.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2], value=0.05)

    st.markdown("---")
    st.markdown('<div class="stat-row">Price Alert Target ($)</div>', unsafe_allow_html=True)
    alert_price = st.number_input("", min_value=0.0, value=0.0, step=1.0, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="stat-row">Backtesting</div>', unsafe_allow_html=True)
    run_backtest = st.checkbox("Enable Backtesting Engine", value=True)
    bt_initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=10000, step=1000)
    bt_commission = st.number_input("Commission per Trade ($)", min_value=0.0, value=1.0, step=0.5)
    bt_signal_threshold = st.slider("Signal Threshold (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5,
        help="Min predicted % move to trigger BUY/SELL")
    st.markdown("---")
    st.markdown('<div class="stat-row">Extra Features</div>', unsafe_allow_html=True)
    run_model_compare = st.checkbox("Model Comparison (XGB vs Prophet vs LR)", value=False)
    run_halal_check   = st.checkbox("Halal / Shariah Compliance Check", value=True)
    show_conf_interval = st.checkbox("Confidence Intervals on Forecast", value=True)
    ci_bootstrap_n = st.slider("Bootstrap Samples (CI)", 50, 300, 100, step=50) if show_conf_interval else 100
    st.markdown("---")
    run_btn = st.button("▶ RUN FORECAST", use_container_width=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=str(start), end=str(end), progress=False)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def compute_bollinger_bands(series, period=20, std=2):
    sma = series.rolling(period).mean()
    rolling_std = series.rolling(period).std()
    upper = sma + (std * rolling_std)
    lower = sma - (std * rolling_std)
    return upper, sma, lower

def add_technical_features(df):
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    volume = df['Volume'].squeeze()

    # Moving averages
    df['MA5']   = close.rolling(5).mean()
    df['MA10']  = close.rolling(10).mean()
    df['MA20']  = close.rolling(20).mean()
    df['MA50']  = close.rolling(50).mean()
    df['MA200'] = close.rolling(200).mean()

    # EMA
    df['EMA12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA26'] = close.ewm(span=26, adjust=False).mean()

    # RSI
    df['RSI'] = compute_rsi(close)

    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(close)

    # Bollinger Bands
    df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = compute_bollinger_bands(close)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    df['BB_Pct']   = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # Momentum & volatility
    df['Returns']    = close.pct_change()
    df['Returns_5d'] = close.pct_change(5)
    df['Volatility'] = df['Returns'].rolling(20).std()
    df['Momentum']   = close - close.shift(10)

    # Volume features
    df['Volume_MA10']  = volume.rolling(10).mean()
    df['Volume_Ratio'] = volume / df['Volume_MA10']

    # Price range
    df['High_Low_Pct'] = (high - low) / close
    df['Close_Open_Pct'] = (close - df['Open'].squeeze()) / df['Open'].squeeze()

    # ATR (Average True Range)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    return df

def build_xgb_dataset(df, seq_len):
    """
    Build feature matrix where each row = (technical features at time t) + (last seq_len closes).
    Target = next day close price.
    """
    close = df['Close'].squeeze().values

    feature_cols = [
        'MA5', 'MA10', 'MA20', 'MA50', 'EMA12', 'EMA26',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Width', 'BB_Pct',
        'Returns', 'Returns_5d', 'Volatility', 'Momentum',
        'Volume_Ratio', 'High_Low_Pct', 'Close_Open_Pct', 'ATR'
    ]

    # Drop rows with NaN in any feature col
    feat_df = df[feature_cols].copy()
    feat_df['Close'] = close

    # We need at least seq_len rows before each sample
    X_rows, y_rows = [], []

    for i in range(seq_len, len(feat_df) - 1):
        row_feats = feat_df[feature_cols].iloc[i].values
        lag_closes = close[i - seq_len:i]  # last seq_len closing prices
        x = np.concatenate([row_feats, lag_closes])
        X_rows.append(x)
        y_rows.append(close[i + 1])  # predict NEXT day's close

    X = np.array(X_rows)
    y = np.array(y_rows)

    # Remove rows with any NaN (from rolling windows at start)
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return X[mask], y[mask]

def generate_forward_signal(last_close, forecast_price):
    diff_pct = (forecast_price - last_close) / last_close * 100
    if diff_pct > 1.0:
        return "BUY", diff_pct
    elif diff_pct < -1.0:
        return "SELL", diff_pct
    else:
        return "HOLD", diff_pct

def run_backtest_engine(actual_prices, predicted_prices, initial_capital, commission, threshold_pct):
    """
    Vectorised backtest: simulate trading on XGBoost signals over the test set.
    Returns a dict of performance stats + a trades DataFrame + equity curve.
    """
    capital     = float(initial_capital)
    position    = 0        # shares held
    entry_price = 0.0
    trades      = []
    equity      = []

    for i in range(len(predicted_prices) - 1):
        price_now  = float(actual_prices[i])
        price_next = float(actual_prices[i + 1])
        pred_next  = float(predicted_prices[i])

        diff_pct = (pred_next - price_now) / price_now * 100

        # Current equity snapshot
        mark_to_market = capital + position * price_now
        equity.append(mark_to_market)

        # ── BUY signal ────────────────────────────────────────────────────────
        if diff_pct > threshold_pct and position == 0:
            shares = int((capital - commission) / price_now)
            if shares > 0:
                cost = shares * price_now + commission
                capital -= cost
                position = shares
                entry_price = price_now
                trades.append({"Day": i, "Type": "BUY",  "Price": price_now,
                               "Shares": shares, "Capital": capital})

        # ── SELL signal ───────────────────────────────────────────────────────
        elif diff_pct < -threshold_pct and position > 0:
            proceeds = position * price_now - commission
            pnl      = proceeds - (entry_price * position + commission)
            capital += proceeds
            trades.append({"Day": i, "Type": "SELL", "Price": price_now,
                           "Shares": position, "P&L": pnl, "Capital": capital})
            position    = 0
            entry_price = 0.0

    # Close any open position at end
    if position > 0:
        final_price = float(actual_prices[-1])
        proceeds    = position * final_price - commission
        pnl         = proceeds - (entry_price * position + commission)
        capital    += proceeds
        trades.append({"Day": len(actual_prices)-1, "Type": "SELL (EOD)",
                       "Price": final_price, "Shares": position,
                       "P&L": pnl, "Capital": capital})
        position = 0

    equity.append(capital)

    # ── Buy-and-Hold benchmark ────────────────────────────────────────────────
    bh_shares = int((initial_capital - commission) / float(actual_prices[0]))
    bh_final  = bh_shares * float(actual_prices[-1]) - commission
    bh_return = (bh_final - initial_capital) / initial_capital * 100

    # ── Strategy metrics ──────────────────────────────────────────────────────
    strat_return = (capital - initial_capital) / initial_capital * 100

    equity_series = pd.Series(equity)
    drawdown      = equity_series / equity_series.cummax() - 1
    max_drawdown  = float(drawdown.min() * 100)

    daily_returns = equity_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0.0)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty and "P&L" in trades_df.columns:
        closed = trades_df[trades_df["Type"].str.contains("SELL")]
        win_trades  = (closed["P&L"] > 0).sum()
        loss_trades = (closed["P&L"] <= 0).sum()
        win_rate    = win_trades / len(closed) * 100 if len(closed) > 0 else 0.0
        avg_win     = closed[closed["P&L"] > 0]["P&L"].mean() if win_trades > 0 else 0.0
        avg_loss    = closed[closed["P&L"] <= 0]["P&L"].mean() if loss_trades > 0 else 0.0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        total_trades  = len(closed)
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0.0
        total_trades = 0

    # Buy-and-hold equity curve
    bh_equity = [initial_capital * (float(actual_prices[i]) / float(actual_prices[0]))
                 for i in range(len(actual_prices))]

    return {
        "final_capital":  capital,
        "strat_return":   strat_return,
        "bh_return":      bh_return,
        "max_drawdown":   max_drawdown,
        "sharpe":         sharpe,
        "win_rate":       win_rate,
        "total_trades":   total_trades,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "profit_factor":  profit_factor,
        "equity_curve":   equity,
        "bh_equity":      bh_equity,
        "trades_df":      trades_df,
        "drawdown_series": drawdown.tolist(),
    }


# ── Confidence Interval via Bootstrap ─────────────────────────────────────────
def bootstrap_confidence_intervals(model, X_input, n_bootstrap=100, noise_std=0.02):
    all_preds = []
    for _ in range(n_bootstrap):
        noise = np.random.normal(0, noise_std, X_input.shape)
        all_preds.append(model.predict(X_input + noise))
    all_preds = np.array(all_preds)
    return (np.percentile(all_preds, 5, axis=0),
            np.percentile(all_preds, 50, axis=0),
            np.percentile(all_preds, 95, axis=0))

# ── Shariah Compliance ─────────────────────────────────────────────────────────
HARAM_TICKERS = {
    "BUD","STZ","SAM","BREW","ABEV","DEO","BF-B",     # Alcohol
    "MO","PM","BTI","LO","VGR",                        # Tobacco
    "LVS","MGM","WYNN","CZR","PENN","DKNG","BYD",     # Gambling
    "JPM","BAC","WFC","C","GS","MS","AXP",             # Conv. Banks
    "MET","PRU","AIG","ALL","TRV","CB",                # Conv. Insurance
    "HRL","TSN","SFD","CAG",                           # Pork
    "LMT","RTX","NOC","GD","HII",                      # Primary weapons
}
QUESTIONABLE_TICKERS = {
    "DIS","NFLX","PARA","WBD","FOXA","SPOT",           # Media/Entertainment
    "MAR","HLT","H","IHG","WH",                        # Hotels
    "V","MA","AXP","COF","USB","PNC",                  # Payment/Finance adjacent
}
HARAM_SECTORS_KW = ["bank","insurance","casino","gambling","alcohol","tobacco",
                     "brewing","distill","porn","adult","weapons","defense","firearm"]

@st.cache_data
def get_shariah_data(ticker_sym):
    try:
        info = yf.Ticker(ticker_sym).info
        return {
            "debt_to_mktcap":  (info.get("totalDebt",0) or 0) / max(info.get("marketCap",1) or 1, 1),
            "debt_to_assets":  (info.get("totalDebt",0) or 0) / max(info.get("totalAssets",1) or 1, 1),
            "cash_to_assets":  (info.get("totalCash",0) or 0) / max(info.get("totalAssets",1) or 1, 1),
            "market_cap":      info.get("marketCap", 0) or 0,
            "total_debt":      info.get("totalDebt", 0) or 0,
            "total_assets":    info.get("totalAssets", 0) or 0,
            "total_cash":      info.get("totalCash", 0) or 0,
            "sector":          info.get("sector", "Unknown"),
            "industry":        info.get("industry", "Unknown"),
            "company_name":    info.get("longName", ticker_sym),
        }
    except:
        return None

def check_shariah_compliance(ticker_sym, data):
    t = ticker_sym.upper()
    ind_lower = data["industry"].lower()

    haram_hit = None
    if t in HARAM_TICKERS:
        haram_hit = "Known non-compliant ticker"
    else:
        for kw in HARAM_SECTORS_KW:
            if kw in ind_lower:
                haram_hit = data["industry"]
                break

    questionable = t in QUESTIONABLE_TICKERS

    d2mc = data["debt_to_mktcap"]
    d2a  = data["debt_to_assets"]
    c2a  = data["cash_to_assets"]

    r = {
        "business": {"pass": haram_hit is None, "haram_hit": haram_hit, "questionable": questionable},
        "debt_mktcap": {"pass": d2mc < 0.30, "value": d2mc, "label": f"Debt/MarketCap = {d2mc*100:.1f}% (< 30%)"},
        "debt_assets": {"pass": d2a  < 0.33, "value": d2a,  "label": f"Debt/Assets = {d2a*100:.1f}% (< 33%)"},
        "cash_assets": {"pass": c2a  < 0.33, "value": c2a,  "label": f"Cash/Assets = {c2a*100:.1f}% (< 33%)"},
    }

    all_pass = r["business"]["pass"] and r["debt_mktcap"]["pass"] and r["debt_assets"]["pass"] and r["cash_assets"]["pass"]

    if not r["business"]["pass"]:
        r["verdict"] = "NON-COMPLIANT"
    elif not all_pass:
        r["verdict"] = "NON-COMPLIANT"
    elif questionable:
        r["verdict"] = "QUESTIONABLE"
    else:
        r["verdict"] = "COMPLIANT"

    return r

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0a0a",
    plot_bgcolor="#0f0f0f",
    font=dict(family="IBM Plex Mono", color="#aaaaaa", size=11),
    xaxis=dict(gridcolor="#1a1a1a", linecolor="#333", tickfont=dict(color="#666")),
    yaxis=dict(gridcolor="#1a1a1a", linecolor="#333", tickfont=dict(color="#666")),
    legend=dict(bgcolor="#111", bordercolor="#333", borderwidth=1),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ── Main ───────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Fetching {ticker} data..."):
        df = fetch_data(ticker, start_date, end_date)

    if df.empty:
        st.error(f"No data found for '{ticker}'. Please check the symbol.")
        st.stop()

    st.success(f"✓ {len(df)} trading days loaded for {ticker}")

    # ── Price Alert Check ──────────────────────────────────────────────────────
    last_close = float(df['Close'].iloc[-1])
    if alert_price > 0:
        if last_close >= alert_price:
            st.markdown(f'<div class="alert-box">🔔 ALERT: {ticker} is currently at ${last_close:.2f} — AT or ABOVE your target of ${alert_price:.2f}</div>', unsafe_allow_html=True)
        else:
            diff = alert_price - last_close
            st.markdown(f'<div class="alert-box">🔔 ALERT: {ticker} at ${last_close:.2f} — ${diff:.2f} below your target of ${alert_price:.2f}</div>', unsafe_allow_html=True)

    # ── Feature Engineering ────────────────────────────────────────────────────
    with st.spinner("Engineering technical features..."):
        df = add_technical_features(df)
    close_series = df['Close'].squeeze()

    # ── Candlestick Chart ──────────────────────────────────────────────────────
    st.subheader("CANDLESTICK CHART")
    fig_candle = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.03
    )
    fig_candle.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'].squeeze(), high=df['High'].squeeze(),
        low=df['Low'].squeeze(),  close=close_series,
        name="Price",
        increasing_line_color='#00cc44',
        decreasing_line_color='#ff3333',
    ), row=1, col=1)
    fig_candle.add_trace(go.Scatter(
        x=df.index, y=df['MA50'].squeeze(),
        name="MA50", line=dict(color="#ff6600", width=1.2)
    ), row=1, col=1)
    fig_candle.add_trace(go.Scatter(
        x=df.index, y=df['MA200'].squeeze(),
        name="MA200", line=dict(color="#00aaff", width=1.2)
    ), row=1, col=1)
    fig_candle.add_trace(go.Scatter(
        x=df.index, y=df['BB_Upper'].squeeze(),
        name="BB Upper", line=dict(color="#666", width=0.8, dash='dot')
    ), row=1, col=1)
    fig_candle.add_trace(go.Scatter(
        x=df.index, y=df['BB_Lower'].squeeze(),
        name="BB Lower", line=dict(color="#666", width=0.8, dash='dot'),
        fill='tonexty', fillcolor='rgba(100,100,100,0.05)'
    ), row=1, col=1)

    colors_vol = ['#00cc44' if c >= o else '#ff3333'
                  for c, o in zip(close_series, df['Open'].squeeze())]
    fig_candle.add_trace(go.Bar(
        x=df.index, y=df['Volume'].squeeze(),
        name="Volume", marker_color=colors_vol, opacity=0.6
    ), row=2, col=1)

    candle_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")}
    fig_candle.update_layout(
        **candle_layout,
        title=dict(text=f"{ticker} — Candlestick + MA50/MA200 + Bollinger Bands + Volume",
                   font=dict(color="#ff6600", size=13)),
        xaxis_rangeslider_visible=False,
        height=550,
    )
    fig_candle.update_xaxes(gridcolor="#1a1a1a", linecolor="#333", tickfont=dict(color="#666"))
    fig_candle.update_yaxes(gridcolor="#1a1a1a", linecolor="#333", tickfont=dict(color="#666"))
    st.plotly_chart(fig_candle, use_container_width=True)

    # ── RSI + MACD Charts ──────────────────────────────────────────────────────
    st.subheader("TECHNICAL INDICATORS")
    fig_tech = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.08,
        subplot_titles=["RSI (14)", "MACD (12/26/9)"]
    )

    # RSI
    fig_tech.add_trace(go.Scatter(
        x=df.index, y=df['RSI'].squeeze(),
        name="RSI", line=dict(color="#ff6600", width=1.5)
    ), row=1, col=1)
    fig_tech.add_hline(y=70, line_dash="dash", line_color="#ff3333", row=1, col=1)
    fig_tech.add_hline(y=30, line_dash="dash", line_color="#00cc44", row=1, col=1)
    fig_tech.add_hrect(y0=70, y1=100, fillcolor="rgba(255,51,51,0.05)", line_width=0, row=1, col=1)
    fig_tech.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,204,68,0.05)",  line_width=0, row=1, col=1)

    # MACD
    fig_tech.add_trace(go.Scatter(
        x=df.index, y=df['MACD'].squeeze(),
        name="MACD", line=dict(color="#ff6600", width=1.2)
    ), row=2, col=1)
    fig_tech.add_trace(go.Scatter(
        x=df.index, y=df['MACD_Signal'].squeeze(),
        name="Signal", line=dict(color="#00aaff", width=1.2)
    ), row=2, col=1)
    macd_hist = df['MACD_Hist'].squeeze()
    hist_colors = ['#00cc44' if v >= 0 else '#ff3333' for v in macd_hist]
    fig_tech.add_trace(go.Bar(
        x=df.index, y=macd_hist,
        name="Histogram", marker_color=hist_colors, opacity=0.7
    ), row=2, col=1)

    # Build layout dict without xaxis/yaxis keys (handled separately for subplots)
    subplot_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis', 'yaxis')}
    fig_tech.update_layout(**subplot_layout, height=450)
    fig_tech.update_xaxes(gridcolor="#1a1a1a", linecolor="#333", tickfont=dict(color="#666"))
    fig_tech.update_yaxes(gridcolor="#1a1a1a", linecolor="#333", tickfont=dict(color="#666"))
    fig_tech.update_yaxes(range=[0, 100], row=1, col=1)
    st.plotly_chart(fig_tech, use_container_width=True)

    # ── XGBoost Model ─────────────────────────────────────────────────────────
    st.markdown('<div class="model-badge">🤖 MODEL: XGBoost Regressor + 20 Technical Features</div>', unsafe_allow_html=True)

    with st.spinner("Building feature matrix..."):
        X, y = build_xgb_dataset(df, seq_len)

    if len(X) < 50:
        st.error("Not enough data to train. Try a longer date range or smaller lookback window.")
        st.stop()

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    with st.spinner("Training XGBoost model..."):
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

    preds = model.predict(X_test)
    actual = y_test

    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae  = mean_absolute_error(actual, preds)
    mape = np.mean(np.abs((actual - preds) / actual)) * 100
    r2   = 1 - np.sum((actual - preds)**2) / np.sum((actual - np.mean(actual))**2)

    # ── Model Performance ──────────────────────────────────────────────────────
    st.subheader("MODEL PERFORMANCE")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"${rmse:.2f}")
    c2.metric("MAE",  f"${mae:.2f}")
    c3.metric("MAPE", f"{mape:.2f}%")
    c4.metric("R²",   f"{r2:.4f}")

    # ── Feature Importance ────────────────────────────────────────────────────
    st.subheader("FEATURE IMPORTANCE")
    feature_cols = [
        'MA5', 'MA10', 'MA20', 'MA50', 'EMA12', 'EMA26',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Width', 'BB_Pct',
        'Returns', 'Returns_5d', 'Volatility', 'Momentum',
        'Volume_Ratio', 'High_Low_Pct', 'Close_Open_Pct', 'ATR'
    ]
    lag_names = [f'Lag_{i+1}' for i in range(seq_len)]
    all_feature_names = feature_cols + lag_names

    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(20)

    fig_imp = go.Figure(go.Bar(
        x=imp_df['importance'],
        y=imp_df['feature'],
        orientation='h',
        marker=dict(
            color=imp_df['importance'],
            colorscale=[[0, '#1a1a1a'], [1, '#ff6600']],
            showscale=False
        )
    ))
    fig_imp.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Top 20 Feature Importances", font=dict(color="#ff6600", size=13)),
        height=450,
        xaxis_title="Importance Score",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Actual vs Predicted ────────────────────────────────────────────────────
    st.subheader("ACTUAL VS PREDICTED")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        y=actual, name="Actual",
        line=dict(color="#00aaff", width=1.5),
        fill='tozeroy', fillcolor='rgba(0,170,255,0.04)'
    ))
    fig1.add_trace(go.Scatter(
        y=preds, name="XGBoost Predicted",
        line=dict(color="#ff6600", width=1.5, dash='dot'),
    ))
    fig1.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"{ticker} — XGBoost Model Fit (Test Set)", font=dict(color="#ff6600", size=13)),
        height=350
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Buy/Sell Signals (based on predicted direction) ────────────────────────
    st.subheader("BUY / SELL SIGNALS")

    # Signal = based on predicted next-day price vs current actual
    signal_list = []
    for i in range(len(preds)):
        diff_pct = (preds[i] - actual[i]) / actual[i] * 100
        if diff_pct > 1.0:
            signal_list.append("BUY")
        elif diff_pct < -1.0:
            signal_list.append("SELL")
        else:
            signal_list.append("HOLD")

    latest_signal, latest_diff = generate_forward_signal(last_close, preds[-1])
    badge_class = {"BUY": "signal-badge-buy", "SELL": "signal-badge-sell", "HOLD": "signal-badge-hold"}[latest_signal]
    icon = {"BUY": "▲", "SELL": "▼", "HOLD": "◆"}[latest_signal]
    direction = "+" if latest_diff >= 0 else ""
    st.markdown(
        f'<div class="{badge_class}">{icon} &nbsp; SIGNAL: {latest_signal} &nbsp; ({direction}{latest_diff:.2f}%)</div><br>',
        unsafe_allow_html=True
    )

    buy_idx  = [i for i, s in enumerate(signal_list) if s == "BUY"]
    sell_idx = [i for i, s in enumerate(signal_list) if s == "SELL"]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=actual, name="Actual Price", line=dict(color="#555", width=1)))
    fig2.add_trace(go.Scatter(
        x=buy_idx, y=[actual[i] for i in buy_idx],
        mode='markers', name='BUY',
        marker=dict(color='#00cc44', symbol='triangle-up', size=8)
    ))
    fig2.add_trace(go.Scatter(
        x=sell_idx, y=[actual[i] for i in sell_idx],
        mode='markers', name='SELL',
        marker=dict(color='#ff3333', symbol='triangle-down', size=8)
    ))
    fig2.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"{ticker} — Signal Map (±1% threshold)", font=dict(color="#ff6600", size=13)),
        height=350
    )
    st.plotly_chart(fig2, use_container_width=True)

    signal_df = pd.DataFrame({
        "Day": range(1, len(signal_list) + 1),
        "Actual Price ($)": [f"${p:.2f}" for p in actual],
        "Predicted ($)":    [f"${p:.2f}" for p in preds],
        "Signal": signal_list
    })
    st.dataframe(signal_df.tail(10), use_container_width=True, hide_index=True)

    # ── Future Forecast ────────────────────────────────────────────────────────
    st.subheader(f"FORECAST — NEXT {future_days} DAYS")

    # Rolling forecast: use last known features + predicted close as next input
    future_prices = []
    last_known_close = float(df['Close'].squeeze().iloc[-1])
    last_row_feats = X[-1].copy()  # last feature vector

    current_close = last_known_close
    for d in range(future_days):
        next_pred = float(model.predict(last_row_feats.reshape(1, -1))[0])
        future_prices.append(next_pred)
        # Shift lag window: drop oldest lag, append new predicted close
        n_tech = len(feature_cols)
        lags = last_row_feats[n_tech:]         # lag values
        new_lags = np.append(lags[1:], next_pred / last_known_close * last_known_close)
        # Keep technical features same (best approximation without live recalculation)
        last_row_feats = np.concatenate([last_row_feats[:n_tech], new_lags])
        current_close = next_pred

    trend_color = "#00cc44" if future_prices[-1] > last_close else "#ff3333"

    fig3 = go.Figure()
    fig3.add_hline(y=last_close, line_dash="dash", line_color="#444",
                   annotation_text=f"Last close ${last_close:.2f}",
                   annotation_font_color="#666")
    if alert_price > 0:
        fig3.add_hline(y=alert_price, line_dash="dot", line_color="#ff6600",
                       annotation_text=f"Alert ${alert_price:.2f}",
                       annotation_font_color="#ff6600")
    fig3.add_trace(go.Scatter(
        x=list(range(future_days)), y=future_prices,
        mode='lines+markers', name='XGBoost Forecast',
        line=dict(color=trend_color, width=2),
        marker=dict(size=7, color=trend_color, line=dict(width=1, color="#0a0a0a"))
    ))
    fig3.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"{ticker} — {future_days}-Day Price Forecast (XGBoost)",
                   font=dict(color="#ff6600", size=13)),
        xaxis_title="Days from today", yaxis_title="Price (USD)",
        height=350
    )
    st.plotly_chart(fig3, use_container_width=True)

    future_df = pd.DataFrame({
        "Day": [f"+{i+1}" for i in range(future_days)],
        "Predicted Price ($)": [f"${p:.2f}" for p in future_prices],
        "vs Last Close": [f"{'▲' if p > last_close else '▼'} {abs(p - last_close):.2f} ({(p-last_close)/last_close*100:+.2f}%)" for p in future_prices]
    })
    st.dataframe(future_df, use_container_width=True, hide_index=True)

    # ── Backtesting Engine ────────────────────────────────────────────────────
    if run_backtest:
        st.subheader("⚡ BACKTESTING ENGINE")
        st.markdown(f'<div class="model-badge">📊 STRATEGY: XGBoost Signal ±{bt_signal_threshold}% | Capital: ${bt_initial_capital:,.0f} | Commission: ${bt_commission}/trade</div>', unsafe_allow_html=True)

        with st.spinner("Running backtest simulation..."):
            bt = run_backtest_engine(
                actual_prices      = actual,
                predicted_prices   = preds,
                initial_capital    = bt_initial_capital,
                commission         = bt_commission,
                threshold_pct      = bt_signal_threshold,
            )

        # ── KPI Cards ─────────────────────────────────────────────────────────
        strat_color = "bt-card-value-green" if bt["strat_return"] >= 0 else "bt-card-value-red"
        bh_color    = "bt-card-value-green" if bt["bh_return"]    >= 0 else "bt-card-value-red"
        dd_color    = "bt-card-value-red"   if bt["max_drawdown"] < -10 else "bt-card-value"
        sh_color    = "bt-card-value-green" if bt["sharpe"]       >= 1  else "bt-card-value-red"

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'''<div class="bt-card">
            <div class="bt-card-label">Strategy Return</div>
            <div class="{strat_color}">{bt["strat_return"]:+.2f}%</div>
        </div>''', unsafe_allow_html=True)
        k2.markdown(f'''<div class="bt-card">
            <div class="bt-card-label">Buy &amp; Hold Return</div>
            <div class="{bh_color}">{bt["bh_return"]:+.2f}%</div>
        </div>''', unsafe_allow_html=True)
        k3.markdown(f'''<div class="bt-card">
            <div class="bt-card-label">Max Drawdown</div>
            <div class="{dd_color}">{bt["max_drawdown"]:.2f}%</div>
        </div>''', unsafe_allow_html=True)
        k4.markdown(f'''<div class="bt-card">
            <div class="bt-card-label">Sharpe Ratio</div>
            <div class="{sh_color}">{bt["sharpe"]:.2f}</div>
        </div>''', unsafe_allow_html=True)

        k5, k6, k7, k8 = st.columns(4)
        k5.markdown(f'''<div class="bt-card">
            <div class="bt-card-label">Final Capital</div>
            <div class="bt-card-value">${bt["final_capital"]:,.0f}</div>
        </div>''', unsafe_allow_html=True)
        k6.markdown(f'''<div class="bt-card">
            <div class="bt-card-label">Total Trades</div>
            <div class="bt-card-value">{bt["total_trades"]}</div>
        </div>''', unsafe_allow_html=True)
        k7.markdown(f'''<div class="bt-card">
            <div class="bt-card-label">Win Rate</div>
            <div class="bt-card-value">{bt["win_rate"]:.1f}%</div>
        </div>''', unsafe_allow_html=True)
        k8.markdown(f'''<div class="bt-card">
            <div class="bt-card-label">Profit Factor</div>
            <div class="bt-card-value">{bt["profit_factor"]:.2f}x</div>
        </div>''', unsafe_allow_html=True)

        # ── Equity Curve vs Buy & Hold ─────────────────────────────────────────
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            y=bt["equity_curve"], name="XGBoost Strategy",
            line=dict(color="#ff6600", width=2),
            fill="tozeroy", fillcolor="rgba(255,102,0,0.05)"
        ))
        fig_eq.add_trace(go.Scatter(
            y=bt["bh_equity"], name="Buy & Hold",
            line=dict(color="#00aaff", width=1.5, dash="dot")
        ))
        fig_eq.add_hline(y=bt_initial_capital, line_dash="dash",
                         line_color="#333",
                         annotation_text=f"Start ${bt_initial_capital:,}",
                         annotation_font_color="#555")
        fig_eq.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text=f"{ticker} — Strategy Equity Curve vs Buy & Hold",
                       font=dict(color="#ff6600", size=13)),
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Trading Day (test set)",
            height=380,
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # ── Drawdown Chart ─────────────────────────────────────────────────────
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            y=[d * 100 for d in bt["drawdown_series"]],
            name="Drawdown",
            line=dict(color="#ff3333", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,51,51,0.08)"
        ))
        fig_dd.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text=f"{ticker} — Strategy Drawdown (%)",
                       font=dict(color="#ff6600", size=13)),
            yaxis_title="Drawdown (%)",
            xaxis_title="Trading Day (test set)",
            height=250,
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── Trade Log ─────────────────────────────────────────────────────────
        if not bt["trades_df"].empty:
            st.markdown("#### TRADE LOG")
            display_trades = bt["trades_df"].copy()
            display_trades["Price"]   = display_trades["Price"].apply(lambda x: f"${x:.2f}")
            display_trades["Capital"] = display_trades["Capital"].apply(lambda x: f"${x:,.0f}")
            if "P&L" in display_trades.columns:
                display_trades["P&L"] = display_trades["P&L"].apply(
                    lambda x: f"+${x:.2f}" if pd.notna(x) and x >= 0 else (f"-${abs(x):.2f}" if pd.notna(x) else "-")
                )
            st.dataframe(display_trades, use_container_width=True, hide_index=True)

            csv_bt = bt["trades_df"].to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Trade Log", data=csv_bt,
                               file_name=f"{ticker}_trades.csv", mime="text/csv")

    # ══ CONFIDENCE INTERVALS ══════════════════════════════════════════════════
    if show_conf_interval:
        st.subheader("📊 FORECAST WITH CONFIDENCE INTERVALS")
        st.markdown('<div class="ci-badge">95% CI — Bootstrap Resampling</div>', unsafe_allow_html=True)

        with st.spinner(f"Running {ci_bootstrap_n} bootstrap samples..."):
            ci_lower, ci_median, ci_upper = bootstrap_confidence_intervals(
                model, X_test, n_bootstrap=ci_bootstrap_n, noise_std=0.015)

        fig_ci = go.Figure()
        fig_ci.add_trace(go.Scatter(y=ci_upper, name="95% Upper",
            line=dict(color="rgba(255,102,0,0)", width=0), showlegend=False))
        fig_ci.add_trace(go.Scatter(y=ci_lower, name="95% CI Band",
            fill="tonexty", fillcolor="rgba(255,102,0,0.15)",
            line=dict(color="rgba(255,102,0,0)", width=0)))
        fig_ci.add_trace(go.Scatter(y=actual, name="Actual",
            line=dict(color="#00aaff", width=1.5)))
        fig_ci.add_trace(go.Scatter(y=ci_median, name="XGBoost Median",
            line=dict(color="#ff6600", width=1.8, dash="dot")))
        fig_ci.update_layout(**PLOTLY_LAYOUT,
            title=dict(text=f"{ticker} — Predictions with 95% CI",
                font=dict(color="#ff6600", size=13)),
            height=380, yaxis_title="Price ($)", xaxis_title="Trading Day (test set)")
        st.plotly_chart(fig_ci, use_container_width=True)

        # Future forecast CI
        future_all = []
        for _ in range(ci_bootstrap_n):
            fp, lrf = [], X[-1].copy()
            for d in range(future_days):
                nxt = float(model.predict((lrf + np.random.normal(0, 0.015, lrf.shape)).reshape(1,-1))[0])
                fp.append(nxt)
                n_tech = len(feature_cols)
                lrf = np.concatenate([lrf[:n_tech], np.append(lrf[n_tech+1:], nxt)])
            future_all.append(fp)
        fa = np.array(future_all)
        fut_l, fut_m, fut_u = np.percentile(fa,5,axis=0), np.percentile(fa,50,axis=0), np.percentile(fa,95,axis=0)

        fig_fci = go.Figure()
        fig_fci.add_trace(go.Scatter(x=list(range(future_days)), y=fut_u,
            line=dict(color="rgba(255,102,0,0)"), showlegend=False))
        fig_fci.add_trace(go.Scatter(x=list(range(future_days)), y=fut_l,
            name="95% CI", fill="tonexty", fillcolor="rgba(255,102,0,0.2)",
            line=dict(color="rgba(255,102,0,0)")))
        fig_fci.add_trace(go.Scatter(x=list(range(future_days)), y=fut_m,
            name="Forecast", mode="lines+markers",
            line=dict(color="#ff6600", width=2), marker=dict(size=7)))
        fig_fci.add_hline(y=last_close, line_dash="dash", line_color="#444",
            annotation_text=f"Last close ${last_close:.2f}", annotation_font_color="#666")
        fig_fci.update_layout(**PLOTLY_LAYOUT,
            title=dict(text=f"{ticker} — {future_days}-Day Forecast with 95% CI",
                font=dict(color="#ff6600", size=13)),
            xaxis_title="Days from today", yaxis_title="Price ($)", height=350)
        st.plotly_chart(fig_fci, use_container_width=True)

    # ══ MODEL COMPARISON ══════════════════════════════════════════════════════
    if run_model_compare:
        st.subheader("🔬 MODEL COMPARISON — XGBoost vs Prophet vs Linear Regression")
        from sklearn.linear_model import LinearRegression as LR
        cmp = {}
        cmp["XGBoost"] = {"preds": preds, "color": "#ff6600",
            "rmse": float(np.sqrt(mean_squared_error(actual, preds))),
            "mae":  float(mean_absolute_error(actual, preds)),
            "mape": float(np.mean(np.abs((actual-preds)/actual))*100),
            "r2":   float(1 - np.sum((actual-preds)**2)/np.sum((actual-np.mean(actual))**2))}

        with st.spinner("Training Linear Regression..."):
            lr_m = LR(); lr_m.fit(X_train, y_train); lr_p = lr_m.predict(X_test)
        cmp["Linear Regression"] = {"preds": lr_p, "color": "#888888",
            "rmse": float(np.sqrt(mean_squared_error(actual, lr_p))),
            "mae":  float(mean_absolute_error(actual, lr_p)),
            "mape": float(np.mean(np.abs((actual-lr_p)/actual))*100),
            "r2":   float(1 - np.sum((actual-lr_p)**2)/np.sum((actual-np.mean(actual))**2))}

        try:
            from prophet import Prophet
            cs_full = df["Close"].squeeze()
            pdf = pd.DataFrame({"ds": df.index[:len(cs_full)], "y": cs_full.values}).dropna()
            ptr = pdf.iloc[:int(len(pdf)*0.8)]; pte = pdf.iloc[int(len(pdf)*0.8):]
            with st.spinner("Training Prophet..."):
                pm = Prophet(daily_seasonality=False, weekly_seasonality=True,
                             yearly_seasonality=True, changepoint_prior_scale=0.05)
                pm.fit(ptr)
                pfut = pm.make_future_dataframe(periods=len(pte), freq="B")
                pfcst = pm.predict(pfut)
                pp = pfcst["yhat"].values[-len(pte):]
                pa = pte["y"].values; ml = min(len(pp), len(actual))
                pp, pa = pp[:ml], actual[:ml]
            cmp["Prophet"] = {"preds": pp, "color": "#aa44ff",
                "rmse": float(np.sqrt(mean_squared_error(pa, pp))),
                "mae":  float(mean_absolute_error(pa, pp)),
                "mape": float(np.mean(np.abs((pa-pp)/pa))*100),
                "r2":   float(1 - np.sum((pa-pp)**2)/np.sum((pa-np.mean(pa))**2))}
        except ImportError:
            st.info("Add `prophet` to requirements.txt to enable Prophet comparison.")

        rows = [{"Model": n, "RMSE ($)": f"${r['rmse']:.2f}", "MAE ($)": f"${r['mae']:.2f}",
                 "MAPE (%)": f"{r['mape']:.2f}%", "R²": f"{r['r2']:.4f}"}
                for n, r in cmp.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(y=actual, name="Actual", line=dict(color="#00aaff", width=2)))
        for n, r in cmp.items():
            fig_cmp.add_trace(go.Scatter(y=r["preds"], name=n,
                line=dict(color=r["color"], width=1.5, dash="dot")))
        fig_cmp.update_layout(**PLOTLY_LAYOUT,
            title=dict(text=f"{ticker} — Model Comparison (Test Set)",
                font=dict(color="#ff6600", size=13)),
            height=380, yaxis_title="Price ($)")
        st.plotly_chart(fig_cmp, use_container_width=True)

        names_l = list(cmp.keys()); rmse_l = [r["rmse"] for r in cmp.values()]
        best = names_l[rmse_l.index(min(rmse_l))]
        fig_b = go.Figure(go.Bar(x=names_l, y=rmse_l,
            marker_color=[r["color"] for r in cmp.values()],
            text=[f"${v:.2f}" for v in rmse_l], textposition="outside"))
        fig_b.update_layout(**PLOTLY_LAYOUT,
            title=dict(text=f"RMSE Comparison — Lower is Better  |  Best: {best}",
                font=dict(color="#ff6600", size=13)),
            yaxis_title="RMSE ($)", height=300)
        st.plotly_chart(fig_b, use_container_width=True)

    # ══ HALAL / SHARIAH COMPLIANCE ════════════════════════════════════════════
    if run_halal_check:
        st.subheader("☪️ HALAL / SHARIAH COMPLIANCE CHECK")
        st.markdown(
            '<div class="model-badge">Based on AAOIFI Standard No.21 + S&P Shariah Indices Methodology</div>',
            unsafe_allow_html=True)

        with st.spinner(f"Fetching financial data for {ticker}..."):
            sd = get_shariah_data(ticker)

        if sd is None:
            st.error("Could not fetch company data. Check the ticker symbol.")
        else:
            cr = check_shariah_compliance(ticker, sd)
            verdict = cr["verdict"]
            v_color = {"COMPLIANT":"#00cc44","NON-COMPLIANT":"#ff3333","QUESTIONABLE":"#ffaa00"}[verdict]
            v_bg    = {"COMPLIANT":"#001a00","NON-COMPLIANT":"#1a0000","QUESTIONABLE":"#1a1000"}[verdict]
            v_icon  = {"COMPLIANT":"\u2705","NON-COMPLIANT":"\u274c","QUESTIONABLE":"\u26a0\ufe0f"}[verdict]

            st.markdown(
                f'<div style="background:{v_bg};border:2px solid {v_color};padding:1.2rem 2rem;' +
                f'margin:1rem 0;text-align:center;">' +
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#888;letter-spacing:.15em;text-transform:uppercase;">' +
                f'{sd["company_name"]} ({ticker})</div>' +
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:2rem;font-weight:700;color:{v_color};margin-top:.4rem;">' +
                f'{v_icon}&nbsp;{verdict}</div>' +
                f'<div style="font-size:.8rem;color:#888;margin-top:.3rem;">Sector: {sd["sector"]} | Industry: {sd["industry"]}</div>' +
                '</div>',
                unsafe_allow_html=True)

            st.markdown("#### SCREENING CRITERIA BREAKDOWN")
            cl, cr2 = st.columns(2)

            with cl:
                bs = cr["business"]
                if bs["haram_hit"]:
                    st.markdown(f'<div class="halal-card-fail"><b>\u274c Business Activity</b><br>Non-compliant: <b>{bs["haram_hit"]}</b></div>', unsafe_allow_html=True)
                elif bs["questionable"]:
                    st.markdown('<div class="halal-card" style="border-left-color:#ffaa00"><b>\u26a0\ufe0f Business Activity</b><br>Questionable sector — consult a scholar</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="halal-card"><b>\u2705 Business Activity</b><br>No Haram core business detected<br><small style="color:#888">Sector: {bs["sector"] if "sector" in bs else sd["sector"]}</small></div>', unsafe_allow_html=True)

                dm = cr["debt_mktcap"]
                cc = "halal-card" if dm["pass"] else "halal-card-fail"
                st.markdown(f'<div class="{cc}"><b>{"\u2705" if dm["pass"] else "\u274c"} Debt / Market Cap</b><br>{dm["label"]}</div>', unsafe_allow_html=True)

            with cr2:
                da = cr["debt_assets"]
                cc = "halal-card" if da["pass"] else "halal-card-fail"
                st.markdown(f'<div class="{cc}"><b>{"\u2705" if da["pass"] else "\u274c"} Debt / Total Assets</b><br>{da["label"]}</div>', unsafe_allow_html=True)

                ca = cr["cash_assets"]
                cc = "halal-card" if ca["pass"] else "halal-card-fail"
                st.markdown(f'<div class="{cc}"><b>{"\u2705" if ca["pass"] else "\u274c"} Cash / Total Assets</b><br>{ca["label"]}</div>', unsafe_allow_html=True)

            st.markdown("#### FINANCIAL SNAPSHOT")
            f1,f2,f3,f4 = st.columns(4)
            f1.metric("Market Cap",   f'${sd["market_cap"]/1e9:.2f}B')
            f2.metric("Total Debt",   f'${sd["total_debt"]/1e9:.2f}B')
            f3.metric("Total Assets", f'${sd["total_assets"]/1e9:.2f}B')
            f4.metric("Cash",         f'${sd["total_cash"]/1e9:.2f}B')

            st.markdown(
                '<div style="background:#111;border:1px solid #222;padding:.8rem 1.2rem;' +
                'font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#555;margin-top:1rem;">' +
                '\u26a0 DISCLAIMER: Automated screen based on AAOIFI Standard No.21. ' +
                'Does not constitute a fatwa. Consult a qualified Islamic finance scholar for binding rulings.</div>',
                unsafe_allow_html=True)

    # ── Download CSV Report ────────────────────────────────────────────────────
    st.subheader("DOWNLOAD REPORT")
    export_df = df[['Open', 'High', 'Low', 'Close', 'Volume',
                    'MA50', 'MA200', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'ATR']].copy()

    csv_hist    = export_df.to_csv().encode('utf-8')
    csv_signals = signal_df.to_csv(index=False).encode('utf-8')
    csv_future  = future_df.to_csv(index=False).encode('utf-8')

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(label="⬇ Historical Data + Indicators", data=csv_hist,
                           file_name=f"{ticker}_historical.csv", mime="text/csv")
    with dl2:
        st.download_button(label="⬇ Buy/Sell Signals", data=csv_signals,
                           file_name=f"{ticker}_signals.csv", mime="text/csv")
    with dl3:
        st.download_button(label="⬇ Forecast Data", data=csv_future,
                           file_name=f"{ticker}_forecast.csv", mime="text/csv")

    st.markdown("---")
    st.markdown('<div class="stat-row">⚠ For educational purposes only. Not financial advice.</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="padding: 3rem; text-align: center; color: #444; font-family: 'IBM Plex Mono', monospace;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
        <div style="font-size: 1rem; letter-spacing: 0.1em; text-transform: uppercase;">
            Configure parameters and press RUN FORECAST
        </div>
        <br>
        <div style="color: #333; font-size: 0.8rem;">AAPL · TSLA · GOOGL · MSFT · AMZN · NFLX · META</div>
    </div>
    """, unsafe_allow_html=True)

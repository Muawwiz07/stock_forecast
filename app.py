import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
.stat-row {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #666;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ticker-header">
    <div class="ticker-title">📈 Stock Price Forecaster</div>
    <div class="ticker-sub">ML-powered price prediction &amp; trading signals</div>
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
        end_date = st.date_input("To", value=pd.to_datetime("2024-01-01"))
    st.markdown('<div class="stat-row">Lookback Window (days)</div>', unsafe_allow_html=True)
    seq_len = st.slider("", 10, 60, 30, label_visibility="collapsed")
    st.markdown('<div class="stat-row">Forecast Horizon (days)</div>', unsafe_allow_html=True)
    future_days = st.slider(" ", 1, 30, 7, label_visibility="collapsed")
    st.markdown("---")
    run_btn = st.button("▶  RUN FORECAST", use_container_width=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=str(start), end=str(end), progress=False)
    return df

def generate_signals(actual, predicted):
    signals = []
    for i in range(len(predicted)):
        if predicted[i] > actual[i]:
            signals.append("BUY")
        elif predicted[i] < actual[i]:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    return signals

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

    st.success(f"✓  {len(df)} trading days loaded for {ticker}")

    close = df[['Close']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    with st.spinner("Training model..."):
        model = LinearRegression()
        model.fit(X_train, y_train)

    preds_scaled = model.predict(X_test).reshape(-1, 1)
    preds  = scaler.inverse_transform(preds_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae  = mean_absolute_error(actual, preds)
    mape = np.mean(np.abs((actual - preds) / actual)) * 100

    # ── Model Performance ──────────────────────────────────────────────────────
    st.subheader("MODEL PERFORMANCE")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"${rmse:.2f}")
    c2.metric("MAE",  f"${mae:.2f}")
    c3.metric("MAPE", f"{mape:.2f}%")

    # ── Actual vs Predicted ────────────────────────────────────────────────────
    st.subheader("ACTUAL VS PREDICTED")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        y=actual.flatten(), name="Actual",
        line=dict(color="#00aaff", width=1.5),
        fill='tozeroy', fillcolor='rgba(0,170,255,0.04)'
    ))
    fig1.add_trace(go.Scatter(
        y=preds.flatten(), name="Predicted",
        line=dict(color="#ff6600", width=1.5, dash='dot'),
    ))
    fig1.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=f"{ticker} — Model Fit", font=dict(color="#ff6600", size=13)))
    st.plotly_chart(fig1, use_container_width=True)

    # ── Buy/Sell Signals ───────────────────────────────────────────────────────
    st.subheader("BUY / SELL SIGNALS")
    signals = generate_signals(actual.flatten(), preds.flatten())
    latest = signals[-1]

    badge_class = {"BUY": "signal-badge-buy", "SELL": "signal-badge-sell", "HOLD": "signal-badge-hold"}[latest]
    icon = {"BUY": "▲", "SELL": "▼", "HOLD": "◆"}[latest]
    st.markdown(f'<div class="{badge_class}">{icon} &nbsp; SIGNAL: {latest}</div><br>', unsafe_allow_html=True)

    buy_idx  = [i for i, s in enumerate(signals) if s == "BUY"]
    sell_idx = [i for i, s in enumerate(signals) if s == "SELL"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=actual.flatten(), name="Actual Price",
        line=dict(color="#555", width=1),
    ))
    fig2.add_trace(go.Scatter(
        x=buy_idx, y=[actual.flatten()[i] for i in buy_idx],
        mode='markers', name='BUY',
        marker=dict(color='#00cc44', symbol='triangle-up', size=8)
    ))
    fig2.add_trace(go.Scatter(
        x=sell_idx, y=[actual.flatten()[i] for i in sell_idx],
        mode='markers', name='SELL',
        marker=dict(color='#ff3333', symbol='triangle-down', size=8)
    ))
    fig2.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=f"{ticker} — Signal Map", font=dict(color="#ff6600", size=13)))
    st.plotly_chart(fig2, use_container_width=True)

    signal_df = pd.DataFrame({
        "Day":              range(1, len(signals) + 1),
        "Actual Price ($)": [f"${p:.2f}" for p in actual.flatten()],
        "Predicted ($)":    [f"${p:.2f}" for p in preds.flatten()],
        "Signal":           signals
    })
    st.dataframe(signal_df.tail(10), use_container_width=True, hide_index=True)

    # ── Future Forecast ────────────────────────────────────────────────────────
    st.subheader(f"FORECAST — NEXT {future_days} DAYS")
    last_seq = scaled[-seq_len:].flatten()
    future_preds = []
    for _ in range(future_days):
        next_val = model.predict(last_seq.reshape(1, -1))[0]
        future_preds.append(next_val)
        last_seq = np.append(last_seq[1:], next_val)

    future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    last_price = float(close[-1][0])
    trend_color = "#00cc44" if future_prices[-1] > last_price else "#ff3333"

    fig3 = go.Figure()
    fig3.add_hline(y=last_price, line_dash="dash", line_color="#444",
                   annotation_text=f"Last close ${last_price:.2f}",
                   annotation_font_color="#666")
    fig3.add_trace(go.Scatter(
        x=list(range(future_days)),
        y=future_prices,
        mode='lines+markers',
        name='Forecast',
        line=dict(color=trend_color, width=2),
        marker=dict(size=7, color=trend_color, line=dict(width=1, color="#0a0a0a"))
    ))
    fig3.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=f"{ticker} — {future_days}-Day Price Forecast", font=dict(color="#ff6600", size=13)),
        xaxis_title="Days from today",
        yaxis_title="Price (USD)"
    )
    st.plotly_chart(fig3, use_container_width=True)

    future_df = pd.DataFrame({
        "Day":                [f"+{i+1}" for i in range(future_days)],
        "Predicted Price ($)": [f"${p:.2f}" for p in future_prices],
        "vs Last Close":       [f"{'▲' if p > last_price else '▼'} {abs(p-last_price):.2f}" for p in future_prices]
    })
    st.dataframe(future_df, use_container_width=True, hide_index=True)

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

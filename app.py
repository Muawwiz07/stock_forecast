import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="📈",
    layout="wide"
)

# ─────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background-color: #1a1a2e !important;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
[data-testid="stSidebar"] input {
    background-color: #2a2a3e !important;
    color: #00ff88 !important;
    -webkit-text-fill-color: #00ff88 !important;
    border: 1px solid #00ff88 !important;
    border-radius: 6px !important;
    font-size: 1rem !important;
}
[data-testid="stSidebar"] input::placeholder {
    color: #888888 !important;
    -webkit-text-fill-color: #888888 !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stTextInput label p {
    color: #cccccc !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background-color: #2a2a3e !important;
    color: #ffffff !important;
    border: 1px solid #00ff88 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #00ff88 !important;
    color: #000000 !important;
    font-weight: bold !important;
    border: none !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #00cc66 !important;
}
[data-testid="collapsedControl"] {
    background-color: #1a1a2e !important;
    color: #ffffff !important;
}
@media (max-width: 768px) {
    [data-testid="stSidebar"] { min-height: 100vh !important; }
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Price Forecaster")
st.markdown("Predict future stock prices using an **XGBoost** machine learning model.")

# ─────────────────────────────────────────
#  SIDEBAR CONTROLS
# ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    ticker = st.text_input("Stock Ticker", value="AAPL").upper()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))

    seq_len       = st.slider("Lookback Window (days)", 10, 60, 30)
    n_estimators  = st.slider("XGBoost Trees", 50, 500, 100)
    forecast_days = st.slider("Forecast Days Ahead", 1, 30, 7)

    run_btn = st.button("🚀 Run Forecast", use_container_width=True)

# ─────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────
@st.cache_data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=str(start), end=str(end), progress=False)
    return df

def make_features(series, seq_len):
    X, y = [], []
    for i in range(seq_len, len(series)):
        X.append(series[i - seq_len:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# ─────────────────────────────────────────
#  MAIN EXECUTION
# ─────────────────────────────────────────
if run_btn:
    with st.spinner(f"Fetching {ticker} data..."):
        df = fetch_data(ticker, start_date, end_date)

    if df.empty:
        st.error(f"No data found for ticker '{ticker}'. Please check the symbol.")
        st.stop()

    st.success(f"✅ Loaded {len(df)} trading days for **{ticker}**")

    # ── Preprocess ──
    close_prices = df[['Close']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices).flatten()

    X, y = make_features(scaled, seq_len)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ── Train ──
    with st.spinner("Training XGBoost model..."):
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)

    # ── Evaluate ──
    predictions_scaled = model.predict(X_test).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions_scaled)
    actual      = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae  = mean_absolute_error(actual, predictions)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100

    # ── Metrics ──
    st.subheader("📊 Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"${rmse:.2f}", help="Root Mean Squared Error — lower is better")
    c2.metric("MAE",  f"${mae:.2f}",  help="Mean Absolute Error — lower is better")
    c3.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")

    # ── Charts ──
    st.subheader("📉 Actual vs Predicted Prices")
    chart_df = pd.DataFrame({
        "Actual":    actual.flatten(),
        "Predicted": predictions.flatten()
    })
    st.line_chart(chart_df)

    # ── Multi-day Forecast ──
    st.subheader(f"🔮 Next {forecast_days}-Day Forecast")

    future_input = scaled[-seq_len:].tolist()
    future_preds = []

    for _ in range(forecast_days):
        x_input = np.array(future_input[-seq_len:]).reshape(1, -1)
        pred = model.predict(x_input)[0]
        future_preds.append(pred)
        future_input.append(pred)

    future_prices = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)
    ).flatten()

    last_price = float(close_prices[-1][0])

    future_dates = pd.bdate_range(
        start=pd.to_datetime(end_date) + pd.Timedelta(days=1),
        periods=forecast_days
    )
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast Price": future_prices.round(2)
    }).set_index("Date")

    st.line_chart(forecast_df)
    st.dataframe(forecast_df, use_container_width=True)

    # ── Summary metrics ──
    final_price = future_prices[-1]
    change      = final_price - last_price
    pct_change  = (change / last_price) * 100

    n1, n2, n3 = st.columns(3)
    n1.metric("Last Close",                    f"${last_price:.2f}")
    n2.metric(f"Day {forecast_days} Forecast", f"${final_price:.2f}")
    n3.metric("Expected Change",               f"{change:+.2f}", delta=f"{pct_change:+.2f}%")

    st.info("⚠️ This is for educational purposes only. Do not use for real trading decisions.")

else:
    st.info("Configure settings in the sidebar and click **Run Forecast** to begin.")
    st.markdown("""
    ### How it works
    1. Fetches historical stock data from Yahoo Finance
    2. Creates lag features from a rolling lookback window
    3. Trains an **XGBoost** gradient boosting model
    4. Predicts and visualizes future prices

    ### Popular tickers to try
    `AAPL` · `TSLA` · `GOOGL` · `MSFT` · `AMZN` · `NFLX` · `META`
    """)

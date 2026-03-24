import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Price Forecaster", page_icon="📈", layout="wide")

st.title("📈 Stock Price Forecaster")
st.markdown("Predict future stock prices using Machine Learning.")

with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))
    seq_len = st.slider("Lookback Window (days)", 10, 60, 30)
    future_days = st.slider("Forecast Days Ahead", 1, 30, 7)
    run_btn = st.button("🚀 Run Forecast", use_container_width=True)

@st.cache_data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=str(start), end=str(end), progress=False)
    return df

if run_btn:
    with st.spinner(f"Fetching {ticker} data..."):
        df = fetch_data(ticker, start_date, end_date)

    if df.empty:
        st.error(f"No data found for '{ticker}'. Please check the symbol.")
        st.stop()

    st.success(f"Loaded {len(df)} trading days for {ticker}")

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

    st.subheader("📊 Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"${rmse:.2f}")
    c2.metric("MAE",  f"${mae:.2f}")
    c3.metric("MAPE", f"{mape:.2f}%")

    st.subheader("📉 Actual vs Predicted")
    chart_df = pd.DataFrame({"Actual": actual.flatten(), "Predicted": preds.flatten()})
    st.line_chart(chart_df)

    # Future forecast
    st.subheader(f"🔮 Next {future_days} Days Forecast")
    last_seq = scaled[-seq_len:].flatten()
    future_preds = []
    for _ in range(future_days):
        next_val = model.predict(last_seq.reshape(1, -1))[0]
        future_preds.append(next_val)
        last_seq = np.append(last_seq[1:], next_val)

    future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    last_price = float(close[-1][0])

    future_df = pd.DataFrame({
        "Day": [f"Day +{i+1}" for i in range(future_days)],
        "Predicted Price ($)": [f"${p:.2f}" for p in future_prices]
    })
    st.dataframe(future_df, use_container_width=True)

    # Plot future
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(future_days), future_prices, marker='o', color='#FF5722', linewidth=2)
    ax.axhline(y=last_price, color='#2196F3', linestyle='--', label=f'Last close: ${last_price:.2f}')
    ax.set_title(f'{ticker} — {future_days}-Day Price Forecast')
    ax.set_xlabel('Days from today')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.info("⚠️ For educational purposes only. Do not use for real trading decisions.")

else:
    st.info("Configure settings in the sidebar and click **Run Forecast** to begin.")
    st.markdown("""
    ### How it works
    1. Fetches real stock data from Yahoo Finance
    2. Creates sliding window sequences from price history
    3. Trains a regression model to predict price trends
    4. Forecasts prices for the next N days

    ### Popular tickers to try
    `AAPL` · `TSLA` · `GOOGL` · `MSFT` · `AMZN` · `NFLX` · `META`
    """)

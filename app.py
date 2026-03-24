import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
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

st.title("📈 Stock Price Forecaster")
st.markdown("Predict future stock prices using a deep learning LSTM model.")

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

    seq_len    = st.slider("Lookback Window (days)", 30, 120, 60)
    epochs     = st.slider("Training Epochs", 5, 50, 20)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

    run_btn = st.button("🚀 Run Forecast", use_container_width=True)

# ─────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────
@st.cache_data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=str(start), end=str(end), progress=False)
    return df

def build_model(seq_len):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ─────────────────────────────────────────
#  MAIN EXECUTION
# ─────────────────────────────────────────
if run_btn:
    with st.spinner(f"Fetching {ticker} data..."):
        df = fetch_data(ticker, start_date, end_date)

    if df.empty:
        st.error(f"No data found for ticker '{ticker}'. Please check the symbol.")
        st.stop()

    st.success(f"Loaded {len(df)} trading days for {ticker}")

    # ── Preprocess ──
    close_prices = df[['Close']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ── Train ──
    model = build_model(seq_len)
    callbacks = [EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)]

    progress_bar = st.progress(0, text="Training model...")
    history_placeholder = st.empty()

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=0
    )
    progress_bar.progress(100, text="Training complete!")

    # ── Evaluate ──
    predictions_scaled = model.predict(X_test, verbose=0)
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

    st.subheader("📉 Training Loss")
    loss_df = pd.DataFrame({
        "Train Loss": history.history['loss'],
        "Val Loss":   history.history['val_loss']
    })
    st.line_chart(loss_df)

    # ── Next Day Prediction ──
    st.subheader("🔮 Next Day Prediction")
    last_seq        = close_prices[-seq_len:]
    last_seq_scaled = scaler.transform(last_seq)
    X_next          = last_seq_scaled.reshape(1, seq_len, 1)
    next_price      = scaler.inverse_transform(model.predict(X_next, verbose=0))[0][0]

    last_price = float(close_prices[-1][0])
    change     = next_price - last_price
    pct_change = (change / last_price) * 100

    n1, n2, n3 = st.columns(3)
    n1.metric("Last Close",       f"${last_price:.2f}")
    n2.metric("Predicted Next",   f"${next_price:.2f}")
    n3.metric("Expected Change",  f"{change:+.2f}", delta=f"{pct_change:+.2f}%")

    st.info("⚠️ This is for educational purposes only. Do not use for real trading decisions.")

else:
    st.info("Configure settings in the sidebar and click **Run Forecast** to begin.")
    st.markdown("""
    ### How it works
    1. Fetches historical stock data from Yahoo Finance
    2. Normalizes prices and creates 60-day sequences
    3. Trains a stacked LSTM neural network
    4. Predicts and visualizes future prices

    ### Popular tickers to try
    `AAPL` · `TSLA` · `GOOGL` · `MSFT` · `AMZN` · `NFLX` · `META`
    """)

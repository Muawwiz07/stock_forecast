import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
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
/* ── Sidebar background ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background-color: #1a1a2e !important;
}
/* ── All sidebar text ── */
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
/* ── Text input ── */
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
/* ── Labels ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stTextInput label p {
    color: #cccccc !important;
    font-size: 0.85rem !important;
}
/* ── Selectbox ── */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background-color: #2a2a3e !important;
    color: #ffffff !important;
    border: 1px solid #00ff88 !important;
}
/* ── Button ── */
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
/* ── Mobile toggle ── */
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
st.markdown("Predict future stock prices using **LSTM** (deep learning) or **XGBoost** (gradient boosting).")

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

    seq_len = st.slider("Lookback Window (days)", 10, 120, 30)

    st.markdown("---")
    st.subheader("🤖 LSTM Settings")
    epochs     = st.slider("Training Epochs", 5, 50, 20)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

    st.markdown("---")
    st.subheader("⚡ XGBoost Settings")
    n_estimators  = st.slider("XGBoost Trees", 50, 500, 100)
    forecast_days = st.slider("Forecast Days Ahead", 1, 30, 7)

    st.markdown("---")
    run_btn = st.button("🚀 Run Forecast", use_container_width=True)

# ─────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────
@st.cache_data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=str(start), end=str(end), progress=False)
    return df

def make_features_xgb(series, seq_len):
    X, y = [], []
    for i in range(seq_len, len(series)):
        X.append(series[i - seq_len:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def make_features_lstm(scaled, seq_len):
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y

def build_lstm_model(seq_len):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
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
    except ImportError:
        return None

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

    close_prices = df[['Close']].values
    scaler = MinMaxScaler()
    scaled_2d = scaler.fit_transform(close_prices)
    scaled_1d = scaled_2d.flatten()

    # ── Split into two tabs ──
    tab1, tab2 = st.tabs(["⚡ XGBoost Model", "🧠 LSTM Model"])

    # ════════════════════════════════════════
    #  TAB 1 — XGBoost
    # ════════════════════════════════════════
    with tab1:
        st.subheader("⚡ XGBoost Forecast")

        X_xgb, y_xgb = make_features_xgb(scaled_1d, seq_len)
        split_idx = int(len(X_xgb) * 0.8)
        X_train, X_test = X_xgb[:split_idx], X_xgb[split_idx:]
        y_train, y_test = y_xgb[:split_idx], y_xgb[split_idx:]

        with st.spinner("Training XGBoost model..."):
            xgb_model = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42,
                verbosity=0
            )
            xgb_model.fit(X_train, y_train)

        # Evaluate
        preds_scaled = xgb_model.predict(X_test).reshape(-1, 1)
        predictions  = scaler.inverse_transform(preds_scaled)
        actual       = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae  = mean_absolute_error(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100

        # Metrics
        st.subheader("📊 Model Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"${rmse:.2f}", help="Root Mean Squared Error")
        c2.metric("MAE",  f"${mae:.2f}",  help="Mean Absolute Error")
        c3.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")

        # Actual vs Predicted
        st.subheader("📉 Actual vs Predicted Prices")
        chart_df = pd.DataFrame({
            "Actual":    actual.flatten(),
            "Predicted": predictions.flatten()
        })
        st.line_chart(chart_df)

        # Multi-day forecast
        st.subheader(f"🔮 Next {forecast_days}-Day Forecast")
        future_input = scaled_1d[-seq_len:].tolist()
        future_preds = []
        for _ in range(forecast_days):
            x_in = np.array(future_input[-seq_len:]).reshape(1, -1)
            pred = xgb_model.predict(x_in)[0]
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
            "Forecast Price": future_prices.round(2)
        }, index=future_dates)
        forecast_df.index.name = "Date"

        st.line_chart(forecast_df)
        st.dataframe(forecast_df, use_container_width=True)

        final_price = future_prices[-1]
        change      = final_price - last_price
        pct_change  = (change / last_price) * 100

        n1, n2, n3 = st.columns(3)
        n1.metric("Last Close",                    f"${last_price:.2f}")
        n2.metric(f"Day {forecast_days} Forecast", f"${final_price:.2f}")
        n3.metric("Expected Change",               f"{change:+.2f}", delta=f"{pct_change:+.2f}%")

    # ════════════════════════════════════════
    #  TAB 2 — LSTM
    # ════════════════════════════════════════
    with tab2:
        st.subheader("🧠 LSTM Deep Learning Forecast")

        lstm_model = build_lstm_model(seq_len)

        if lstm_model is None:
            st.error("❌ TensorFlow is not installed in this environment. LSTM model is unavailable. Please use the XGBoost tab.")
        else:
            from tensorflow.keras.callbacks import EarlyStopping

            X_lstm, y_lstm = make_features_lstm(scaled_2d, seq_len)
            split_idx = int(len(X_lstm) * 0.8)
            X_train_l, X_test_l = X_lstm[:split_idx], X_lstm[split_idx:]
            y_train_l, y_test_l = y_lstm[:split_idx], y_lstm[split_idx:]

            callbacks = [EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)]
            progress_bar = st.progress(0, text="Training LSTM model...")

            history = lstm_model.fit(
                X_train_l, y_train_l,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=callbacks,
                verbose=0
            )
            progress_bar.progress(100, text="Training complete!")

            # Evaluate
            preds_scaled_l = lstm_model.predict(X_test_l, verbose=0)
            predictions_l  = scaler.inverse_transform(preds_scaled_l)
            actual_l       = scaler.inverse_transform(y_test_l.reshape(-1, 1))

            rmse_l = np.sqrt(mean_squared_error(actual_l, predictions_l))
            mae_l  = mean_absolute_error(actual_l, predictions_l)
            mape_l = np.mean(np.abs((actual_l - predictions_l) / actual_l)) * 100

            # Metrics
            st.subheader("📊 Model Performance")
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"${rmse_l:.2f}", help="Root Mean Squared Error")
            c2.metric("MAE",  f"${mae_l:.2f}",  help="Mean Absolute Error")
            c3.metric("MAPE", f"{mape_l:.2f}%", help="Mean Absolute Percentage Error")

            # Actual vs Predicted
            st.subheader("📉 Actual vs Predicted Prices")
            chart_df_l = pd.DataFrame({
                "Actual":    actual_l.flatten(),
                "Predicted": predictions_l.flatten()
            })
            st.line_chart(chart_df_l)

            # Training loss
            st.subheader("📉 Training Loss")
            loss_df = pd.DataFrame({
                "Train Loss": history.history['loss'],
                "Val Loss":   history.history['val_loss']
            })
            st.line_chart(loss_df)

            # Next day prediction
            st.subheader("🔮 Next Day Prediction")
            last_seq        = close_prices[-seq_len:]
            last_seq_scaled = scaler.transform(last_seq)
            X_next          = last_seq_scaled.reshape(1, seq_len, 1)
            next_price      = scaler.inverse_transform(lstm_model.predict(X_next, verbose=0))[0][0]

            last_price_l = float(close_prices[-1][0])
            change_l     = next_price - last_price_l
            pct_change_l = (change_l / last_price_l) * 100

            n1, n2, n3 = st.columns(3)
            n1.metric("Last Close",      f"${last_price_l:.2f}")
            n2.metric("Predicted Next",  f"${next_price:.2f}")
            n3.metric("Expected Change", f"{change_l:+.2f}", delta=f"{pct_change_l:+.2f}%")

    st.info("⚠️ This is for educational purposes only. Do not use for real trading decisions.")

else:
    st.info("Configure settings in the sidebar and click **Run Forecast** to begin.")
    st.markdown("""
    ### How it works
    Choose between two powerful models:

    **⚡ XGBoost** — Fast gradient boosting, works on Streamlit Cloud, supports multi-day forecasting

    **🧠 LSTM** — Deep learning with stacked recurrent layers, requires TensorFlow, great for sequential patterns

    Both models:
    1. Fetch historical stock data from Yahoo Finance
    2. Normalize prices using MinMaxScaler
    3. Build lag features from a rolling lookback window
    4. Evaluate with RMSE, MAE, and MAPE metrics

    ### Popular tickers to try
    `AAPL` · `TSLA` · `GOOGL` · `MSFT` · `AMZN` · `NFLX` · `META`
    """)

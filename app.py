import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="StockCast · Market Intelligence",
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
[data-testid="stSidebar"] * { color: #ffffff !important; }
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
[data-testid="stSidebar"] .stButton > button:hover { background-color: #00cc66 !important; }
[data-testid="collapsedControl"] { background-color: #1a1a2e !important; color: #ffffff !important; }
@media (max-width: 768px) { [data-testid="stSidebar"] { min-height: 100vh !important; } }

/* Halal badge styling */
.halal-pass {
    background: linear-gradient(135deg, #00ff88, #00cc66);
    color: #000 !important;
    padding: 18px 24px;
    border-radius: 12px;
    font-size: 1.3rem;
    font-weight: bold;
    text-align: center;
    margin: 10px 0;
}
.halal-fail {
    background: linear-gradient(135deg, #ff4444, #cc0000);
    color: #fff !important;
    padding: 18px 24px;
    border-radius: 12px;
    font-size: 1.3rem;
    font-weight: bold;
    text-align: center;
    margin: 10px 0;
}
.halal-warn {
    background: linear-gradient(135deg, #ffaa00, #cc8800);
    color: #000 !important;
    padding: 18px 24px;
    border-radius: 12px;
    font-size: 1.3rem;
    font-weight: bold;
    text-align: center;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 StockCast · Market Intelligence")
st.markdown("Forecast prices with **XGBoost · LSTM · Prophet** — with confidence intervals, model comparison & Shariah screening.")

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Stock Ticker", value="AAPL").upper()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))

    seq_len       = st.slider("Lookback Window (days)", 10, 120, 30)
    forecast_days = st.slider("Forecast Days Ahead",    1,  30,   7)
    ci_pct        = st.slider("Confidence Interval %",  80, 99,  95)
    n_bootstrap   = st.slider("Bootstrap Samples",      50, 300, 100)

    st.markdown("---")
    st.subheader("🤖 LSTM Settings")
    epochs     = st.slider("Training Epochs", 5, 50, 20)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

    st.markdown("---")
    st.subheader("⚡ XGBoost Settings")
    n_estimators = st.slider("XGBoost Trees", 50, 500, 100)

    st.markdown("---")
    run_btn = st.button("🚀 Run Forecast", use_container_width=True)

# ─────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────
@st.cache_data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=str(start), end=str(end), progress=False)
    return df

@st.cache_data
def fetch_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info
    except:
        return {}

def make_features(series, seq_len):
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
    return X.reshape(X.shape[0], X.shape[1], 1), y

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

def bootstrap_confidence_intervals(model_fn, X_future, n_bootstrap, ci_pct, scaler):
    """Generate CI by adding noise to future predictions via bootstrapping."""
    alpha = (100 - ci_pct) / 2
    all_preds = []
    for _ in range(n_bootstrap):
        noise = np.random.normal(0, 0.01, size=X_future.shape)
        preds = model_fn(X_future + noise).reshape(-1, 1)
        preds_inv = scaler.inverse_transform(preds).flatten()
        all_preds.append(preds_inv)
    all_preds = np.array(all_preds)
    lower = np.percentile(all_preds, alpha, axis=0)
    upper = np.percentile(all_preds, 100 - alpha, axis=0)
    mean  = np.mean(all_preds, axis=0)
    return mean, lower, upper

def plot_forecast_with_ci(dates, mean, lower, upper, title, color="#00ff88"):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.plot(dates, mean,  color=color, linewidth=2, label="Forecast")
    ax.fill_between(dates, lower, upper, alpha=0.25, color=color, label=f"{ci_pct}% CI")
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.set_title(title, color='white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig

def metrics_row(actual, predictions):
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae  = mean_absolute_error(actual, predictions)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    return rmse, mae, mape

# ─────────────────────────────────────────
#  SHARIAH SCREENING LOGIC
# ─────────────────────────────────────────
# Based on AAOIFI Shariah Standard No. 21
HARAM_SECTORS = [
    "alcohol", "tobacco", "weapons", "defense", "gambling",
    "banking", "financial services", "insurance", "entertainment",
    "pork", "adult entertainment", "cannabis"
]

def shariah_screen(info):
    results = {}
    verdict = "PASS"
    warnings_list = []

    # 1. Business activity check
    sector   = (info.get("sector", "") or "").lower()
    industry = (info.get("industry", "") or "").lower()
    business = sector + " " + industry
    haram_found = [h for h in HARAM_SECTORS if h in business]
    if haram_found:
        results["Business Activity"] = ("❌ FAIL", f"Involves: {', '.join(haram_found)}")
        verdict = "FAIL"
    else:
        results["Business Activity"] = ("✅ PASS", f"Sector: {info.get('sector','N/A')} | Industry: {info.get('industry','N/A')}")

    # 2. Debt ratio (Total Debt / Market Cap < 33%)
    total_debt  = info.get("totalDebt", 0) or 0
    market_cap  = info.get("marketCap", 1) or 1
    debt_ratio  = total_debt / market_cap * 100
    if debt_ratio > 33:
        results["Debt Ratio"] = ("❌ FAIL", f"{debt_ratio:.1f}% (limit: 33%)")
        verdict = "FAIL"
    elif debt_ratio > 25:
        results["Debt Ratio"] = ("⚠️ CAUTION", f"{debt_ratio:.1f}% (approaching 33% limit)")
        if verdict == "PASS":
            verdict = "CAUTION"
    else:
        results["Debt Ratio"] = ("✅ PASS", f"{debt_ratio:.1f}% (limit: 33%)")

    # 3. Cash & interest-bearing securities < 33% of market cap
    total_cash = info.get("totalCash", 0) or 0
    cash_ratio = total_cash / market_cap * 100
    if cash_ratio > 33:
        results["Cash Ratio"] = ("⚠️ CAUTION", f"{cash_ratio:.1f}% (limit: 33%)")
        if verdict == "PASS":
            verdict = "CAUTION"
    else:
        results["Cash Ratio"] = ("✅ PASS", f"{cash_ratio:.1f}% (limit: 33%)")

    # 4. Receivables < 49% of market cap
    net_receivables = info.get("netReceivables", 0) or 0
    rec_ratio = net_receivables / market_cap * 100
    if rec_ratio > 49:
        results["Receivables Ratio"] = ("❌ FAIL", f"{rec_ratio:.1f}% (limit: 49%)")
        verdict = "FAIL"
    else:
        results["Receivables Ratio"] = ("✅ PASS", f"{rec_ratio:.1f}% (limit: 49%)")

    return verdict, results

# ─────────────────────────────────────────
#  PROPHET HELPER
# ─────────────────────────────────────────
def run_prophet(df, forecast_days):
    try:
        from prophet import Prophet
        prophet_df = df[['Close']].reset_index()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)
        prophet_df['y']  = prophet_df['y'].astype(float)
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(prophet_df)
        future   = m.make_future_dataframe(periods=forecast_days, freq='B')
        forecast = m.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days), None
    except ImportError:
        return None, "prophet not installed"
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
if run_btn:
    with st.spinner(f"Fetching {ticker} data..."):
        df   = fetch_data(ticker, start_date, end_date)
        info = fetch_info(ticker)

    if df.empty:
        st.error(f"No data found for '{ticker}'. Please check the symbol.")
        st.stop()

    st.success(f"✅ Loaded {len(df)} trading days for **{ticker}** — {info.get('longName', ticker)}")

    close_prices = df[['Close']].values
    scaler    = MinMaxScaler()
    scaled_2d = scaler.fit_transform(close_prices)
    scaled_1d = scaled_2d.flatten()

    future_dates = pd.bdate_range(
        start=pd.to_datetime(end_date) + pd.Timedelta(days=1),
        periods=forecast_days
    )

    # ════════════════════════════════════
    #  TABS
    # ════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚡ XGBoost",
        "🧠 LSTM",
        "🔵 Prophet",
        "📊 Model Comparison",
        "☪️ Shariah Screen"
    ])

    # ── Store metrics for comparison ──
    comparison = {}

    # ════════════════════════════════════
    #  TAB 1 — XGBoost + CI
    # ════════════════════════════════════
    with tab1:
        st.subheader("⚡ XGBoost Forecast with Confidence Intervals")

        X_xgb, y_xgb = make_features(scaled_1d, seq_len)
        split_idx     = int(len(X_xgb) * 0.8)
        X_tr, X_te    = X_xgb[:split_idx], X_xgb[split_idx:]
        y_tr, y_te    = y_xgb[:split_idx], y_xgb[split_idx:]

        with st.spinner("Training XGBoost..."):
            xgb = XGBRegressor(
                n_estimators=n_estimators, learning_rate=0.05,
                max_depth=5, subsample=0.8, random_state=42, verbosity=0
            )
            xgb.fit(X_tr, y_tr)

        preds_sc  = xgb.predict(X_te).reshape(-1, 1)
        preds_inv = scaler.inverse_transform(preds_sc)
        actual    = scaler.inverse_transform(y_te.reshape(-1, 1))
        rmse, mae, mape = metrics_row(actual, preds_inv)
        comparison["XGBoost"] = {"RMSE": rmse, "MAE": mae, "MAPE": mape}

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"${rmse:.2f}")
        c2.metric("MAE",  f"${mae:.2f}")
        c3.metric("MAPE", f"{mape:.2f}%")

        st.subheader("📉 Actual vs Predicted")
        st.line_chart(pd.DataFrame({"Actual": actual.flatten(), "Predicted": preds_inv.flatten()}))

        # Multi-day forecast + CI via bootstrap
        st.subheader(f"🔮 {forecast_days}-Day Forecast + {ci_pct}% Confidence Interval")
        future_input = scaled_1d[-seq_len:].tolist()
        fut_X = []
        temp_input = future_input.copy()
        for _ in range(forecast_days):
            x_in = np.array(temp_input[-seq_len:]).reshape(1, -1)
            pred = xgb.predict(x_in)[0]
            fut_X.append(temp_input[-seq_len:].copy())
            temp_input.append(pred)

        fut_X = np.array(fut_X)
        mean_f, lower_f, upper_f = bootstrap_confidence_intervals(
            lambda X: np.array([xgb.predict(X[i:i+1])[0] for i in range(len(X))]),
            fut_X, n_bootstrap, ci_pct, scaler
        )
        fig = plot_forecast_with_ci(future_dates, mean_f, lower_f, upper_f,
                                    f"{ticker} XGBoost {forecast_days}-Day Forecast", "#00ff88")
        st.pyplot(fig)

        forecast_df = pd.DataFrame({
            "Forecast":    mean_f.round(2),
            f"Lower {ci_pct}%": lower_f.round(2),
            f"Upper {ci_pct}%": upper_f.round(2)
        }, index=future_dates)
        forecast_df.index.name = "Date"
        st.dataframe(forecast_df, use_container_width=True)

        last_price = float(close_prices[-1][0])
        n1, n2, n3 = st.columns(3)
        n1.metric("Last Close",                    f"${last_price:.2f}")
        n2.metric(f"Day {forecast_days} Forecast", f"${mean_f[-1]:.2f}")
        n3.metric("CI Range (last day)",           f"${lower_f[-1]:.2f} – ${upper_f[-1]:.2f}")

    # ════════════════════════════════════
    #  TAB 2 — LSTM + CI
    # ════════════════════════════════════
    with tab2:
        st.subheader("🧠 LSTM Deep Learning Forecast with Confidence Intervals")
        lstm_model = build_lstm_model(seq_len)

        if lstm_model is None:
            st.error("❌ TensorFlow not installed. LSTM unavailable — use XGBoost or Prophet tabs.")
        else:
            from tensorflow.keras.callbacks import EarlyStopping

            X_lstm, y_lstm = make_features_lstm(scaled_2d, seq_len)
            split_idx      = int(len(X_lstm) * 0.8)
            X_tr_l, X_te_l = X_lstm[:split_idx], X_lstm[split_idx:]
            y_tr_l, y_te_l = y_lstm[:split_idx], y_lstm[split_idx:]

            cb = [EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)]
            pb = st.progress(0, text="Training LSTM...")
            history = lstm_model.fit(
                X_tr_l, y_tr_l, epochs=epochs, batch_size=batch_size,
                validation_split=0.1, callbacks=cb, verbose=0
            )
            pb.progress(100, text="Training complete!")

            preds_sc_l  = lstm_model.predict(X_te_l, verbose=0)
            preds_inv_l = scaler.inverse_transform(preds_sc_l)
            actual_l    = scaler.inverse_transform(y_te_l.reshape(-1, 1))
            rmse_l, mae_l, mape_l = metrics_row(actual_l, preds_inv_l)
            comparison["LSTM"] = {"RMSE": rmse_l, "MAE": mae_l, "MAPE": mape_l}

            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"${rmse_l:.2f}")
            c2.metric("MAE",  f"${mae_l:.2f}")
            c3.metric("MAPE", f"{mape_l:.2f}%")

            st.subheader("📉 Actual vs Predicted")
            st.line_chart(pd.DataFrame({"Actual": actual_l.flatten(), "Predicted": preds_inv_l.flatten()}))

            st.subheader("📉 Training Loss")
            st.line_chart(pd.DataFrame({"Train Loss": history.history['loss'], "Val Loss": history.history['val_loss']}))

            # Next day with CI
            st.subheader(f"🔮 Next Day Prediction + {ci_pct}% CI")
            last_seq_scaled = scaler.transform(close_prices[-seq_len:])
            X_next = last_seq_scaled.reshape(1, seq_len, 1)

            boot_preds = []
            for _ in range(n_bootstrap):
                noise = np.random.normal(0, 0.005, X_next.shape)
                p = lstm_model.predict(X_next + noise, verbose=0)
                boot_preds.append(scaler.inverse_transform(p)[0][0])
            boot_preds = np.array(boot_preds)
            alpha = (100 - ci_pct) / 2
            next_mean  = float(np.mean(boot_preds))
            next_lower = float(np.percentile(boot_preds, alpha))
            next_upper = float(np.percentile(boot_preds, 100 - alpha))

            last_price_l = float(close_prices[-1][0])
            n1, n2, n3 = st.columns(3)
            n1.metric("Last Close",      f"${last_price_l:.2f}")
            n2.metric("Predicted Next",  f"${next_mean:.2f}")
            n3.metric(f"{ci_pct}% CI",   f"${next_lower:.2f} – ${next_upper:.2f}")

    # ════════════════════════════════════
    #  TAB 3 — Prophet
    # ════════════════════════════════════
    with tab3:
        st.subheader("🔵 Prophet Forecast with Confidence Intervals")
        with st.spinner("Running Prophet..."):
            prophet_result, prophet_err = run_prophet(df, forecast_days)

        if prophet_err:
            st.error(f"❌ Prophet unavailable: {prophet_err}. Add `prophet` to requirements.txt.")
        else:
            # Evaluate on historical fit
            prophet_df_full = df[['Close']].reset_index()
            prophet_df_full.columns = ['ds', 'y']
            prophet_df_full['ds'] = pd.to_datetime(prophet_df_full['ds']).dt.tz_localize(None)
            prophet_df_full['y']  = prophet_df_full['y'].astype(float)

            from prophet import Prophet
            m2 = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            m2.fit(prophet_df_full)
            hist_forecast = m2.predict(prophet_df_full)

            actual_p = prophet_df_full['y'].values
            pred_p   = hist_forecast['yhat'].values
            rmse_p, mae_p, mape_p = metrics_row(
                actual_p.reshape(-1,1), pred_p.reshape(-1,1)
            )
            comparison["Prophet"] = {"RMSE": rmse_p, "MAE": mae_p, "MAPE": mape_p}

            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"${rmse_p:.2f}")
            c2.metric("MAE",  f"${mae_p:.2f}")
            c3.metric("MAPE", f"{mape_p:.2f}%")

            st.subheader(f"🔮 {forecast_days}-Day Forecast + {ci_pct}% CI")
            fig2 = plot_forecast_with_ci(
                prophet_result['ds'],
                prophet_result['yhat'].values,
                prophet_result['yhat_lower'].values,
                prophet_result['yhat_upper'].values,
                f"{ticker} Prophet {forecast_days}-Day Forecast",
                "#4da6ff"
            )
            st.pyplot(fig2)

            pr_out = prophet_result.rename(columns={
                'ds': 'Date', 'yhat': 'Forecast',
                'yhat_lower': f'Lower {ci_pct}%', 'yhat_upper': f'Upper {ci_pct}%'
            }).set_index('Date')
            pr_out = pr_out.round(2)
            st.dataframe(pr_out, use_container_width=True)

    # ════════════════════════════════════
    #  TAB 4 — Model Comparison
    # ════════════════════════════════════
    with tab4:
        st.subheader("📊 Side-by-Side Model Comparison")

        if not comparison:
            st.info("Run the individual model tabs first to populate comparison data.")
        else:
            comp_df = pd.DataFrame(comparison).T
            comp_df = comp_df.round(4)

            # Best model
            best_model = comp_df['RMSE'].idxmin()
            st.success(f"🏆 Best model by RMSE: **{best_model}**")

            st.dataframe(comp_df.style.highlight_min(axis=0, color="#00ff4422"), use_container_width=True)

            # Bar chart comparison
            fig3, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig3.patch.set_facecolor('#0d0d0d')
            colors = ['#00ff88', '#ff6b6b', '#4da6ff']
            models = list(comparison.keys())

            for ax, metric, color_list in zip(axes, ['RMSE', 'MAE', 'MAPE'],
                                               [colors[:len(models)]]*3):
                vals = [comparison[m][metric] for m in models]
                bars = ax.bar(models, vals, color=colors[:len(models)])
                ax.set_facecolor('#0d0d0d')
                ax.set_title(metric, color='white')
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#333333')
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.95,
                            f'{val:.2f}', ha='center', va='top', color='black', fontsize=9, fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig3)

            # Winner badges
            st.markdown("### 🥇 Winners by Metric")
            w1, w2, w3 = st.columns(3)
            w1.metric("Best RMSE", comp_df['RMSE'].idxmin(), f"${comp_df['RMSE'].min():.2f}")
            w2.metric("Best MAE",  comp_df['MAE'].idxmin(),  f"${comp_df['MAE'].min():.2f}")
            w3.metric("Best MAPE", comp_df['MAPE'].idxmin(), f"{comp_df['MAPE'].min():.2f}%")

    # ════════════════════════════════════
    #  TAB 5 — Shariah Screening
    # ════════════════════════════════════
    with tab5:
        st.subheader(f"☪️ Shariah Compliance Screen — {ticker}")
        st.caption("Based on AAOIFI Shariah Standard No. 21")

        with st.spinner("Running Shariah screening..."):
            verdict, results = shariah_screen(info)

        # Overall verdict badge
        if verdict == "PASS":
            st.markdown(f'<div class="halal-pass">✅ {ticker} — HALAL COMPLIANT</div>', unsafe_allow_html=True)
        elif verdict == "CAUTION":
            st.markdown(f'<div class="halal-warn">⚠️ {ticker} — PROCEED WITH CAUTION</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="halal-fail">❌ {ticker} — NOT SHARIAH COMPLIANT</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📋 Screening Details")

        for check, (status, detail) in results.items():
            col_a, col_b = st.columns([1, 3])
            col_a.markdown(f"**{check}**")
            col_b.markdown(f"{status} &nbsp; — &nbsp; {detail}")

        st.markdown("---")
        st.subheader("ℹ️ Company Info")
        i1, i2, i3 = st.columns(3)
        i1.metric("Sector",    info.get("sector", "N/A"))
        i2.metric("Industry",  info.get("industry", "N/A"))
        i3.metric("Country",   info.get("country", "N/A"))

        market_cap = info.get("marketCap", 0) or 0
        total_debt = info.get("totalDebt", 0) or 0
        total_cash = info.get("totalCash", 0) or 0
        i4, i5, i6 = st.columns(3)
        i4.metric("Market Cap",  f"${market_cap/1e9:.1f}B" if market_cap else "N/A")
        i5.metric("Total Debt",  f"${total_debt/1e9:.1f}B" if total_debt else "N/A")
        i6.metric("Total Cash",  f"${total_cash/1e9:.1f}B" if total_cash else "N/A")

        st.info("⚠️ This screen is automated and for reference only. Always consult a qualified Shariah scholar for investment decisions.")

    st.info("⚠️ For educational purposes only. Not financial advice.")

else:
    st.info("Configure settings in the sidebar and click **🚀 Run Forecast** to begin.")
    st.markdown("""
    ### Features
    | Tab | What you get |
    |-----|-------------|
    | ⚡ XGBoost | Fast ML forecast + bootstrap confidence intervals |
    | 🧠 LSTM | Deep learning forecast + next-day CI |
    | 🔵 Prophet | Facebook Prophet trend + seasonality forecast |
    | 📊 Model Comparison | RMSE / MAE / MAPE side-by-side bar charts + winner |
    | ☪️ Shariah Screen | AAOIFI Standard No.21 — business, debt, cash, receivables checks |

    ### Popular tickers to try
    `AAPL` · `TSLA` · `GOOGL` · `MSFT` · `AMZN` · `NFLX` · `META` · `2222.SR` · `BABA`
    """)

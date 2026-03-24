import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
TICKER      = "AAPL"       # change to any stock e.g. "TSLA", "GOOGL"
START_DATE  = "2018-01-01"
END_DATE    = "2024-01-01"
SEQ_LEN     = 60           # lookback window (days)
EPOCHS      = 25
BATCH_SIZE  = 32
TEST_SPLIT  = 0.2
MODEL_PATH  = "saved_model.h5"


# ─────────────────────────────────────────
#  1. FETCH DATA
# ─────────────────────────────────────────
def fetch_data(ticker, start, end):
    print(f"\n📥 Fetching {ticker} data from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")
    print(f"   ✓ {len(df)} trading days loaded")
    return df


# ─────────────────────────────────────────
#  2. PREPROCESS
# ─────────────────────────────────────────
def preprocess(df, seq_len=60, test_split=0.2):
    close_prices = df[['Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n📊 Data split:")
    print(f"   Train samples : {len(X_train)}")
    print(f"   Test  samples : {len(X_test)}")

    return X_train, X_test, y_train, y_test, scaler, close_prices


# ─────────────────────────────────────────
#  3. BUILD MODEL
# ─────────────────────────────────────────
def build_model(seq_len):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),

        LSTM(64, return_sequences=True),
        Dropout(0.2),

        LSTM(32, return_sequences=False),
        Dropout(0.2),

        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    print("\n🧠 Model architecture:")
    model.summary()
    return model


# ─────────────────────────────────────────
#  4. TRAIN
# ─────────────────────────────────────────
def train_model(model, X_train, y_train):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', save_format='h5')
    ]

    print(f"\n🚀 Training for up to {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    print(f"   ✓ Model saved to {MODEL_PATH}")
    return history


# ─────────────────────────────────────────
#  5. EVALUATE
# ─────────────────────────────────────────
def evaluate(model, X_test, y_test, scaler):
    predictions_scaled = model.predict(X_test, verbose=0)

    predictions = scaler.inverse_transform(predictions_scaled)
    actual      = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae  = mean_absolute_error(actual, predictions)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100

    print(f"\n📈 Evaluation Results:")
    print(f"   RMSE : ${rmse:.2f}")
    print(f"   MAE  : ${mae:.2f}")
    print(f"   MAPE : {mape:.2f}%")

    return predictions, actual


# ─────────────────────────────────────────
#  6. PLOT
# ─────────────────────────────────────────
def plot_results(actual, predictions, history, ticker, close_prices, seq_len, test_split):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{ticker} Stock Price Forecasting — LSTM Model', fontsize=16, fontweight='bold')

    # Plot 1: Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.plot(actual, label='Actual Price', color='#2196F3', linewidth=1.5)
    ax1.plot(predictions, label='Predicted Price', color='#FF5722', linewidth=1.5, linestyle='--')
    ax1.set_title('Actual vs Predicted (Test Set)')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training Loss
    ax2 = axes[0, 1]
    ax2.plot(history.history['loss'], label='Training Loss', color='#4CAF50')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='#FF9800')
    ax2.set_title('Model Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Full price history with prediction zone
    ax3 = axes[1, 0]
    full_actual = close_prices.flatten()
    split_idx = int(len(full_actual) * (1 - test_split))
    train_len = len(full_actual) - len(actual) - seq_len
    ax3.plot(full_actual[:train_len + seq_len],
         label='Training Data', color='#9E9E9E', linewidth=1)
    ax3.plot(range(train_len + seq_len, train_len + seq_len + len(actual)),
         actual.flatten(), label='Actual (Test)', color='#2196F3', linewidth=1.5)
    ax3.plot(range(train_len + seq_len, train_len + seq_len + len(predictions)),
         predictions.flatten(), label='Predicted (Test)', color='#FF5722',
         linewidth=1.5, linestyle='--')
    ax3.axvline(x=train_len + seq_len, color='black', linestyle=':', alpha=0.5, label='Train/Test split')
    ax3.set_title('Full Price History with Forecast')
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Price (USD)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Prediction error
    ax4 = axes[1, 1]
    errors = actual.flatten() - predictions.flatten()
    ax4.bar(range(len(errors)), errors, color=['#4CAF50' if e >= 0 else '#F44336' for e in errors],
            alpha=0.6, width=1)
    ax4.axhline(y=0, color='black', linewidth=0.8)
    ax4.set_title('Prediction Error (Actual − Predicted)')
    ax4.set_xlabel('Days')
    ax4.set_ylabel('Error (USD)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('forecast_results.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Plot saved as forecast_results.png")
    plt.show()


# ─────────────────────────────────────────
#  7. PREDICT NEXT DAY
# ─────────────────────────────────────────
def predict_next_day(model, df, scaler, seq_len):
    last_seq = df[['Close']].values[-seq_len:]
    last_seq_scaled = scaler.transform(last_seq)
    X_input = last_seq_scaled.reshape(1, seq_len, 1)

    next_scaled = model.predict(X_input, verbose=0)
    next_price  = scaler.inverse_transform(next_scaled)[0][0]

    last_price = df['Close'].values[-1]
    change     = next_price - last_price
    pct_change = (change / last_price) * 100

    print(f"\n🔮 Next Day Prediction:")
    print(f"   Last close  : ${float(last_price):.2f}")
    print(f"   Predicted   : ${float(next_price):.2f}")
    print(f"   Change      : {float(change):+.2f} ({float(pct_change):+.2f}%)")
    return next_price


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    # 1. Fetch
    df = fetch_data(TICKER, START_DATE, END_DATE)

    # 2. Preprocess
    X_train, X_test, y_train, y_test, scaler, close_prices = preprocess(
        df, seq_len=SEQ_LEN, test_split=TEST_SPLIT
    )

    # 3. Build
    model = build_model(SEQ_LEN)

    # 4. Train
    history = train_model(model, X_train, y_train)

    # 5. Evaluate
    predictions, actual = evaluate(model, X_test, y_test, scaler)

    # 6. Plot
    plot_results(actual, predictions, history, TICKER, close_prices, SEQ_LEN, TEST_SPLIT)

    # 7. Predict next day
    predict_next_day(model, df, scaler, SEQ_LEN)

# =============================================================================
# Copyright (c) 2026 Stockcast. All Rights Reserved.
# PROPRIETARY AND CONFIDENTIAL
# analytics.py — Technical indicators, XGBoost dataset builder, backtest engine,
#                bootstrap CI, Shariah compliance screening
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import streamlit as st

from config import logger
from data import av_get_overview

# ── Technical indicators ──────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean().replace(0, 1e-10)
    return 100 - (100 / (1 + avg_gain / avg_loss))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def compute_bollinger_bands(series, period=20, std=2):
    sma = series.rolling(period).mean()
    rs  = series.rolling(period).std()
    return sma + std * rs, sma, sma - std * rs


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df["RSI"]   = compute_rsi(close)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(close)
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"]   = compute_bollinger_bands(close)
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
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Stochastic %K / %D
    low14, high14 = low.rolling(14).min(), high.rolling(14).max()
    df["Stoch_K"]  = (close - low14) / (high14 - low14 + 1e-10) * 100
    df["Stoch_D"]  = df["Stoch_K"].rolling(3).mean()

    # OBV
    obv         = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df["OBV"]     = obv
    df["OBV_EMA"] = obv.ewm(span=20, adjust=False).mean()

    # Williams %R
    df["Williams_R"] = (high14 - close) / (high14 - low14 + 1e-10) * -100

    # CCI
    tp      = (high + low + close) / 3
    df["CCI"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std().replace(0, 1e-10))

    # ADX
    plus_dm  = (high.diff().clip(lower=0)).where(high.diff() > (-low.diff()), 0)
    minus_dm = (-low.diff().clip(upper=0)).where((-low.diff()) > high.diff(), 0)
    atr14    = tr.rolling(14).mean().replace(0, 1e-10)
    plus_di  = 100 * plus_dm.rolling(14).mean() / atr14
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14
    dx       = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    df["ADX"]      = dx.rolling(14).mean()
    df["Plus_DI"]  = plus_di
    df["Minus_DI"] = minus_di

    return df


FEATURE_COLS = [
    "MA5", "MA10", "MA20", "MA50", "MA200", "EMA12", "EMA26",
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Width", "BB_Pct", "Returns", "Returns_5d", "Volatility", "Momentum",
    "Volume_Ratio", "High_Low_Pct", "Close_Open_Pct", "ATR",
    "Stoch_K", "Stoch_D", "OBV", "OBV_EMA", "Williams_R", "CCI",
    "ADX", "Plus_DI", "Minus_DI",
]


# ── XGBoost dataset ───────────────────────────────────────────────────────────

def build_xgb_dataset(df: pd.DataFrame, seq_len: int):
    close   = df["Close"].squeeze().values
    feat_df = df[FEATURE_COLS].copy()
    feat_df["Close"] = close
    X_rows, y_rows = [], []
    for i in range(seq_len, len(feat_df) - 1):
        row_feats  = feat_df[FEATURE_COLS].iloc[i].values
        lag_closes = close[i - seq_len:i]
        X_rows.append(np.concatenate([row_feats, lag_closes]))
        y_rows.append(close[i + 1])
    X    = np.array(X_rows)
    y    = np.array(y_rows)
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    dropped = (~mask).sum()
    if dropped > 0:
        warnings.warn(f"build_xgb_dataset: dropped {dropped} rows with NaN (out of {len(mask)} total)")
    return X[mask], y[mask]


# ── Composite signal ──────────────────────────────────────────────────────────

def compute_composite_signal(df, last_close, forecast_price, preds, actual):
    close    = df["Close"].squeeze()
    rsi      = float(df["RSI"].squeeze().iloc[-1])
    macd     = float(df["MACD"].squeeze().iloc[-1])
    macd_s   = float(df["MACD_Signal"].squeeze().iloc[-1])
    macd_h   = float(df["MACD_Hist"].squeeze().iloc[-1])
    bb_pct   = float(df["BB_Pct"].squeeze().iloc[-1])
    ma50     = float(df["MA50"].squeeze().iloc[-1])
    ma200    = float(df["MA200"].squeeze().iloc[-1])
    vol_r    = float(df["Volume_Ratio"].squeeze().iloc[-1])
    atr      = float(df["ATR"].squeeze().iloc[-1])
    stoch_k  = float(df["Stoch_K"].squeeze().iloc[-1])  if "Stoch_K"    in df.columns else 50.0
    stoch_d  = float(df["Stoch_D"].squeeze().iloc[-1])  if "Stoch_D"    in df.columns else 50.0
    will_r   = float(df["Williams_R"].squeeze().iloc[-1]) if "Williams_R" in df.columns else -50.0
    adx      = float(df["ADX"].squeeze().iloc[-1])       if "ADX"        in df.columns else 25.0
    plus_di  = float(df["Plus_DI"].squeeze().iloc[-1])   if "Plus_DI"    in df.columns else 25.0
    minus_di = float(df["Minus_DI"].squeeze().iloc[-1])  if "Minus_DI"   in df.columns else 25.0

    signals  = {}
    xgb_pct  = (forecast_price - last_close) / last_close * 100

    if   xgb_pct >  1.5: signals["AI Outlook"]   = ("BUY",  min(35, abs(xgb_pct) * 6), xgb_pct, "positive")
    elif xgb_pct < -1.5: signals["AI Outlook"]   = ("SELL", -min(35, abs(xgb_pct) * 6), xgb_pct, "negative")
    else:                 signals["AI Outlook"]   = ("HOLD", 0, xgb_pct, "neutral")

    if   rsi < 30: signals["RSI (14)"]   = ("BUY",  20, rsi, "positive")
    elif rsi > 70: signals["RSI (14)"]   = ("SELL", -20, rsi, "negative")
    elif rsi < 45: signals["RSI (14)"]   = ("BUY",   8, rsi, "positive")
    elif rsi > 55: signals["RSI (14)"]   = ("SELL",  -8, rsi, "negative")
    else:          signals["RSI (14)"]   = ("HOLD",   0, rsi, "neutral")

    prev_hist = float(df["MACD_Hist"].squeeze().iloc[-2]) if len(df) > 2 else 0
    if   macd_h > 0 and prev_hist <= 0: signals["MACD Cross"] = ("BUY",  20, macd_h, "positive")
    elif macd_h < 0 and prev_hist >= 0: signals["MACD Cross"] = ("SELL", -20, macd_h, "negative")
    elif macd > macd_s:                 signals["MACD Cross"] = ("BUY",  10, macd_h, "positive")
    elif macd < macd_s:                 signals["MACD Cross"] = ("SELL", -10, macd_h, "negative")
    else:                               signals["MACD Cross"] = ("HOLD",   0, macd_h, "neutral")

    if   bb_pct < 0.1: signals["Bollinger %B"] = ("BUY",  10, bb_pct, "positive")
    elif bb_pct > 0.9: signals["Bollinger %B"] = ("SELL", -10, bb_pct, "negative")
    else:              signals["Bollinger %B"] = ("HOLD",   0, bb_pct, "neutral")

    if   ma50 > ma200 and close.iloc[-1] > ma50: signals["MA Cross"] = ("BUY",  15, ma50 - ma200, "positive")
    elif ma50 < ma200 and close.iloc[-1] < ma50: signals["MA Cross"] = ("SELL", -15, ma50 - ma200, "negative")
    else:                                         signals["MA Cross"] = ("HOLD",   0, ma50 - ma200, "neutral")

    if   vol_r > 1.5 and xgb_pct > 0: signals["Volume"] = ("BUY",  10, vol_r, "positive")
    elif vol_r > 1.5 and xgb_pct < 0: signals["Volume"] = ("SELL", -10, vol_r, "negative")
    else:                              signals["Volume"] = ("HOLD",   0, vol_r, "neutral")

    if   stoch_k < 20 and stoch_k > stoch_d: signals["Stochastic"] = ("BUY",  12, stoch_k, "positive")
    elif stoch_k > 80 and stoch_k < stoch_d: signals["Stochastic"] = ("SELL", -12, stoch_k, "negative")
    elif stoch_k < 30:                        signals["Stochastic"] = ("BUY",   6, stoch_k, "positive")
    elif stoch_k > 70:                        signals["Stochastic"] = ("SELL",  -6, stoch_k, "negative")
    else:                                     signals["Stochastic"] = ("HOLD",   0, stoch_k, "neutral")

    if   will_r < -80: signals["Williams %R"] = ("BUY",  10, will_r, "positive")
    elif will_r > -20: signals["Williams %R"] = ("SELL", -10, will_r, "negative")
    else:              signals["Williams %R"] = ("HOLD",   0, will_r, "neutral")

    if adx > 25:
        if plus_di > minus_di: signals["ADX Trend"] = ("BUY",  8, adx, "positive")
        else:                  signals["ADX Trend"] = ("SELL", -8, adx, "negative")
    else:
        signals["ADX Trend"] = ("HOLD", 0, adx, "neutral")

    total_score = sum(s[1] for s in signals.values())
    if   total_score >= 25:  verdict, verdict_short = "⬆ STRONG BUY",  "BUY"
    elif total_score >= 10:  verdict, verdict_short = "↑ BUY",          "BUY"
    elif total_score <= -25: verdict, verdict_short = "⬇ STRONG SELL",  "SELL"
    elif total_score <= -10: verdict, verdict_short = "↓ SELL",         "SELL"
    else:                    verdict, verdict_short = "◆ HOLD",         "HOLD"

    volatility_mult = 1.0 + min(0.5, float(df["Volatility"].squeeze().dropna().iloc[-1]) * 10)
    stop_loss       = last_close - 1.5 * atr * volatility_mult
    take_profit     = last_close + 2.5 * atr * volatility_mult
    risk_reward     = (take_profit - last_close) / max(last_close - stop_loss, 0.01)

    return {
        "signals": signals, "verdict": verdict, "verdict_short": verdict_short,
        "total_score": total_score, "xgb_pct": xgb_pct, "rsi": rsi,
        "stop_loss": stop_loss, "take_profit": take_profit, "risk_reward": risk_reward,
        "vol_ratio": vol_r, "atr": atr,
        "stoch_k": stoch_k, "stoch_d": stoch_d,
        "williams_r": will_r, "adx": adx,
    }


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest_engine(actual_prices, predicted_prices, initial_capital, commission, threshold_pct):
    capital = float(initial_capital)
    position, entry_price = 0, 0.0
    trades, equity = [], []

    for i in range(len(predicted_prices) - 1):
        price_now = float(actual_prices[i])
        pred_next = float(predicted_prices[i])
        diff_pct  = (pred_next - price_now) / price_now * 100
        equity.append(capital + position * price_now)

        if diff_pct > threshold_pct and position == 0:
            shares = int((capital - commission) / price_now)
            if shares > 0:
                capital  -= shares * price_now + commission
                position  = shares
                entry_price = price_now
                trades.append({"Day": i, "Type": "BUY", "Price": price_now, "Shares": shares, "Capital": capital})
        elif diff_pct < -threshold_pct and position > 0:
            proceeds = position * price_now - commission
            pnl      = proceeds - (entry_price * position + commission)
            capital += proceeds
            trades.append({"Day": i, "Type": "SELL", "Price": price_now, "Shares": position, "P&L": pnl, "Capital": capital})
            position, entry_price = 0, 0.0

    if position > 0:
        fp       = float(actual_prices[-1])
        proceeds = position * fp - commission
        pnl      = proceeds - (entry_price * position + commission)
        capital += proceeds
        trades.append({"Day": len(actual_prices) - 1, "Type": "SELL (EOD)", "Price": fp,
                       "Shares": position, "P&L": pnl, "Capital": capital})

    equity.append(capital)
    equity_s     = pd.Series(equity)
    drawdown     = equity_s / equity_s.cummax() - 1
    daily_r      = equity_s.pct_change().dropna()
    sharpe       = (daily_r.mean() / daily_r.std() * np.sqrt(252)) if daily_r.std() > 0 else 0.0
    strat_return = (capital - initial_capital) / initial_capital * 100

    bh_shares = int((initial_capital - commission) / float(actual_prices[0]))
    bh_final  = bh_shares * float(actual_prices[-1]) - commission
    bh_return = (bh_final - initial_capital) / initial_capital * 100

    trades_df = pd.DataFrame(trades)
    win_rate = avg_win = avg_loss = pf = 0.0
    total_trades = 0
    if not trades_df.empty and "P&L" in trades_df.columns:
        closed       = trades_df[trades_df["Type"].str.contains("SELL")]
        win_trades   = (closed["P&L"] > 0).sum()
        loss_trades  = (closed["P&L"] <= 0).sum()
        win_rate     = win_trades / len(closed) * 100 if len(closed) > 0 else 0.0
        avg_win      = closed[closed["P&L"] > 0]["P&L"].mean() if win_trades > 0 else 0.0
        avg_loss     = closed[closed["P&L"] <= 0]["P&L"].mean() if loss_trades > 0 else 0.0
        pf           = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        total_trades = len(closed)

    bh_equity = [initial_capital * (float(actual_prices[i]) / float(actual_prices[0])) for i in range(len(actual_prices))]

    return {
        "final_capital": capital, "strat_return": strat_return, "bh_return": bh_return,
        "max_drawdown": float(drawdown.min() * 100), "sharpe": sharpe,
        "win_rate": win_rate, "total_trades": total_trades,
        "avg_win": avg_win, "avg_loss": avg_loss, "profit_factor": pf,
        "equity_curve": equity, "bh_equity": bh_equity,
        "trades_df": trades_df, "drawdown_series": drawdown.tolist(),
    }


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_confidence_intervals(model, X_input, n_bootstrap=100, noise_std=None):
    if noise_std is None:
        try:
            vol_idx    = FEATURE_COLS.index("Volatility")
            recent_vol = float(np.nanmedian(X_input[-20:, vol_idx]))
            noise_std  = max(0.005, min(0.05, recent_vol))
        except Exception as e:
            logger.warning("bootstrap_confidence_intervals: using default noise_std=0.02: %s", e)
            noise_std = 0.02
    feature_scale = np.std(X_input, axis=0, keepdims=True)
    feature_scale = np.where(feature_scale == 0, 1.0, feature_scale)
    all_preds = [
        model.predict(X_input + np.random.normal(0, noise_std, X_input.shape) * feature_scale)
        for _ in range(n_bootstrap)
    ]
    a = np.array(all_preds)
    return np.percentile(a, 5, axis=0), np.percentile(a, 50, axis=0), np.percentile(a, 95, axis=0)


# ── Shariah compliance ────────────────────────────────────────────────────────

HARAM_TICKERS = {
    "BUD", "STZ", "SAM", "BREW", "ABEV", "DEO", "BF-B",
    "MO", "PM", "BTI", "LO", "VGR",
    "LVS", "MGM", "WYNN", "CZR", "PENN", "DKNG", "BYD",
    "MET", "PRU", "AIG", "ALL", "TRV", "CB",
    "HRL", "TSN", "SFD", "CAG", "LMT", "RTX", "NOC", "GD", "HII",
}
QUESTIONABLE_TICKERS = {
    "DIS", "NFLX", "PARA", "WBD", "FOXA", "SPOT",
    "MAR", "HLT", "H", "IHG", "WH",
    "JPM", "BAC", "WFC", "C", "GS", "MS",
    "V", "MA", "AXP", "COF", "USB", "PNC",
}
HARAM_SECTORS_KW = [
    "bank", "insurance", "casino", "gambling", "alcohol", "tobacco",
    "brewing", "distill", "porn", "adult", "weapons", "defense", "firearm",
]


@st.cache_data(ttl=3600)
def get_shariah_data(ticker_sym: str):
    info = av_get_overview(ticker_sym)
    if not info or "Symbol" not in info:
        return None

    def _safe(k, d=0):
        v = info.get(k, d)
        try: return float(v) if v not in (None, "None", "-", "") else d
        except: return d

    mc = _safe("MarketCapitalization", 1) or 1
    td = _safe("LongTermDebtUSD", 0) or _safe("TotalDebt", 0)
    ta = _safe("TotalAssets", 1) or 1
    tc = _safe("CashAndCashEquivalentsAtCarryingValue", 0)
    return {
        "debt_to_mktcap": td / mc,
        "debt_to_assets": td / ta,
        "cash_to_assets": tc / ta,
        "market_cap":     mc,
        "total_debt":     td,
        "total_assets":   ta,
        "total_cash":     tc,
        "sector":         info.get("Sector", "Unknown"),
        "industry":       info.get("Industry", "Unknown"),
        "company_name":   info.get("Name", ticker_sym),
    }


def check_shariah_compliance(ticker_sym: str, data: dict, _L=None) -> dict:
    if _L is None:
        _L = {}
    t         = ticker_sym.upper()
    ind_lower = data["industry"].lower()
    haram_hit = None

    if t in HARAM_TICKERS:
        haram_hit = _L.get("known_noncompliant", "Known non-compliant ticker")
    else:
        for kw in HARAM_SECTORS_KW:
            if kw in ind_lower:
                haram_hit = data["industry"]
                break

    questionable = t in QUESTIONABLE_TICKERS
    r = {
        "business":    {"pass": haram_hit is None, "haram_hit": haram_hit, "questionable": questionable},
        "debt_mktcap": {"pass": data["debt_to_mktcap"] < 0.30, "value": data["debt_to_mktcap"],
                        "label": _L.get("shariah_debt_mktcap", "Debt/MarketCap") + f" = {data['debt_to_mktcap']*100:.1f}% (< 30%)"},
        "debt_assets": {"pass": data["debt_to_assets"] < 0.33, "value": data["debt_to_assets"],
                        "label": _L.get("shariah_debt_assets", "Debt/Assets") + f" = {data['debt_to_assets']*100:.1f}% (< 33%)"},
        "cash_assets": {"pass": data["cash_to_assets"] < 0.33, "value": data["cash_to_assets"],
                        "label": _L.get("shariah_cash_assets", "Cash/Assets") + f" = {data['cash_to_assets']*100:.1f}% (< 33%)"},
    }
    all_pass  = all(r[k]["pass"] for k in ["business", "debt_mktcap", "debt_assets", "cash_assets"])
    r["verdict"] = ("NON-COMPLIANT" if not r["business"]["pass"] or not all_pass
                    else ("QUESTIONABLE" if questionable else "COMPLIANT"))
    return r

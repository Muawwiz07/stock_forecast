# =============================================================================
# Copyright (c) 2026 Stockcast. All Rights Reserved.
#
# PROPRIETARY AND CONFIDENTIAL
#
# This software and its source code are the exclusive property of Stockcast.
# Unauthorized copying, reproduction, modification, distribution, or use of
# this software, in whole or in part, via any medium, is strictly prohibited
# without the prior written permission of Stockcast.
#
# This software is provided "as is", without warranty of any kind, express or
# implied. Stockcast shall not be liable for any damages arising from the use
# of this software.
#
# For licensing inquiries, contact: legal@stockcast.com
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from supabase import create_client
import warnings
import smtplib
import html as _html_mod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import time
import logging
warnings.filterwarnings('ignore')

# ── threading (used for cache locks) ──────────────────────────────────────────
import threading
from typing import List
import json
import io
import csv
import math
from typing import List, Optional, Dict, Any

# ── Default False; overwritten to True if library is available at runtime ──────
_KITE_OK    = False
_UPSTOX_OK  = False
_ALPACA_OK  = False
_POSTHOG_OK = False

try:
    from kiteconnect import KiteConnect
    _KITE_OK = True
except ImportError:
    pass

try:
    import upstox_client
    _UPSTOX_OK = True
except ImportError:
    pass

try:
    import alpaca_trade_api as tradeapi
    _ALPACA_OK = True
except ImportError:
    pass

try:
    import posthog as _posthog_lib
    _posthog_lib.project_api_key = os.environ.get('POSTHOG_API_KEY', '')
    _posthog_lib.host = 'https://app.posthog.com'
    _POSTHOG_OK = bool(_posthog_lib.project_api_key)
except Exception:
    pass


# =============================================================================
# SIGNAL ENGINE — inlined (no separate file needed on Streamlit Cloud)
# BUY / SELL / HOLD system with confidence scoring and AI insights.
# =============================================================================

def generate_signal(df: pd.DataFrame, sentiment_score: float) -> dict:
    """
    Generate a BUY / SELL / HOLD signal with confidence score.

    Parameters
    ----------
    df : pd.DataFrame
        Stock dataframe with at least a 'Close' column.
        MA20 is computed internally if not already present.
    sentiment_score : float
        News/sentiment score in the range [-1.0, +1.0].

    Returns
    -------
    dict with keys:
        signal      : str   — "BUY", "SELL", or "HOLD"
        confidence  : float — 0 to 100
        trend       : str   — "Uptrend" | "Downtrend" | "Sideways"
        sentiment   : str   — "Positive" | "Negative" | "Neutral"
        conflict    : bool  — True when trend and sentiment disagree
        volatility  : str   — "High" | "Normal" | "Low"
        details     : dict  — raw intermediate values for transparency
    """
    close = df["Close"].squeeze()
    sentiment_score = max(-1.0, min(1.0, float(sentiment_score)))

    # ── 1. Short-term price momentum (5-day vs prior 5-day) ──────────────────
    if len(close) < 10:
        momentum_score = 0.0
        trend_pct      = 0.0
        trend_label    = "Sideways"
    else:
        recent_avg = float(close.iloc[-5:].mean())
        prior_avg  = float(close.iloc[-10:-5].mean())
        trend_pct  = (recent_avg - prior_avg) / prior_avg * 100 if prior_avg != 0 else 0.0

        # Normalise into [-1, +1] using a ±5% scale (tanh-like soft clamp)
        momentum_score = float(np.tanh(trend_pct / 3.0))

        if   trend_pct >  1.0: trend_label = "Uptrend"
        elif trend_pct < -1.0: trend_label = "Downtrend"
        else:                  trend_label = "Sideways"

    # ── 2. MA20 position (price vs 20-day moving average) ────────────────────
    if "MA20" in df.columns:
        ma20 = float(df["MA20"].squeeze().iloc[-1])
    else:
        ma20 = (
            float(close.rolling(20).mean().iloc[-1])
            if len(close) >= 20
            else float(close.mean())
        )

    last_price = float(close.iloc[-1])
    ma_gap_pct = (last_price - ma20) / ma20 * 100 if ma20 != 0 else 0.0

    # Normalise into [-1, +1] using a ±8% scale
    ma_score = float(np.tanh(ma_gap_pct / 5.0))

    # ── 3. Volatility regime (20-day rolling std of returns) ─────────────────
    if len(close) >= 20:
        daily_returns = close.pct_change().dropna()
        vol_20d = float(daily_returns.rolling(20).std().iloc[-1]) * 100  # in %
    else:
        vol_20d = 1.5  # fallback: assume ~normal

    # Classify volatility — thresholds tuned for typical equities
    if   vol_20d > 3.0: volatility_label = "High"
    elif vol_20d < 1.0: volatility_label = "Low"
    else:               volatility_label = "Normal"

    # ── 4. Sentiment label ────────────────────────────────────────────────────
    if   sentiment_score >  0.20: sentiment_label = "Positive"
    elif sentiment_score < -0.20: sentiment_label = "Negative"
    else:                         sentiment_label = "Neutral"

    # ── 5. Conflict detection (trend and sentiment pulling opposite ways) ─────
    conflict = (
        momentum_score > 0.15 and sentiment_score < -0.20
        or momentum_score < -0.15 and sentiment_score > 0.20
    )

    # ── 6. Signal decision ────────────────────────────────────────────────────
    # Require both momentum AND MA position to agree for a clean BUY/SELL.
    # Sentiment alone can't flip the signal — it modulates confidence.
    technical_score = 0.55 * momentum_score + 0.45 * ma_score

    if   technical_score >  0.15 and sentiment_score >= -0.20: signal = "BUY"
    elif technical_score < -0.15 and sentiment_score <=  0.20: signal = "SELL"
    else:                                                       signal = "HOLD"

    # ── 7. Confidence — built from four independent sub-scores ───────────────
    #
    # (a) SIGNAL STRENGTH — how decisively the technicals point somewhere
    #     Ranges 0→1. Near 0 = borderline, near 1 = strong clear direction.
    strength = min(abs(technical_score) / 0.6, 1.0)

    # (b) AGREEMENT — do trend, MA, and sentiment all point the same way?
    #     Score each pairwise agreement on a soft 0→1 scale.
    trend_vs_ma   = 1.0 - abs(momentum_score - ma_score) / 2.0
    trend_vs_sent = 1.0 - abs(momentum_score - sentiment_score) / 2.0
    ma_vs_sent    = 1.0 - abs(ma_score - sentiment_score) / 2.0
    agreement     = (trend_vs_ma * 0.4 + trend_vs_sent * 0.35 + ma_vs_sent * 0.25)

    # (c) SENTIMENT WEIGHT — strong clear sentiment adds conviction,
    #     wishy-washy neutral sentiment doesn't
    sentiment_weight = min(abs(sentiment_score) / 0.6, 1.0)

    # (d) VOLATILITY PENALTY — high volatility = less predictable = lower conf
    vol_penalty = {"High": 0.75, "Normal": 1.0, "Low": 1.05}.get(volatility_label, 1.0)

    # (e) CONFLICT PENALTY — opposing trend and sentiment kills confidence
    conflict_penalty = 0.65 if conflict else 1.0

    # Weighted blend → [0, 1]
    raw_conf = (
        strength          * 0.40
        + agreement       * 0.30
        + sentiment_weight * 0.30
    )

    # Apply regime and conflict multipliers
    raw_conf *= vol_penalty * conflict_penalty

    # HOLD signals are inherently uncertain — cap them lower
    if signal == "HOLD":
        raw_conf = min(raw_conf, 0.52)

    confidence = round(min(100.0, max(5.0, raw_conf * 100)), 1)

    return {
        "signal":     signal,
        "confidence": confidence,
        "trend":      trend_label,
        "sentiment":  sentiment_label,
        "conflict":   conflict,
        "volatility": volatility_label,
        "details": {
            "momentum_score":    round(momentum_score, 3),
            "ma_score":          round(ma_score, 3),
            "technical_score":   round(technical_score, 3),
            "sentiment_score":   round(sentiment_score, 3),
            "agreement":         round(agreement, 3),
            "strength":          round(strength, 3),
            "vol_20d":           round(vol_20d, 2),
            "trend_pct":         round(trend_pct, 2),
            "last_price":        round(last_price, 2),
            "ma20":              round(ma20, 2),
            "ma_gap_pct":        round(ma_gap_pct, 2),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 · AI INSIGHT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_insight(
    df: pd.DataFrame,
    sentiment_score: float,
    signal: dict,
) -> str:
    """
    Generate 2–3 sentences of plain-English AI insight that reads naturally,
    not like a template-filled form. Language adapts to magnitude, conflict,
    and volatility — not just the signal label.

    Parameters
    ----------
    df              : pd.DataFrame  — stock dataframe with 'Close'
    sentiment_score : float         — [-1, +1]
    signal          : dict          — output from generate_signal()

    Returns
    -------
    str — 2–3 sentence human-readable insight
    """
    sig        = signal["signal"]
    conf       = signal["confidence"]
    trend      = signal["trend"]
    sentiment  = signal["sentiment"]
    conflict   = signal["conflict"]
    volatility = signal["volatility"]
    details    = signal["details"]

    ma20      = details["ma20"]
    ma_gap    = details["ma_gap_pct"]
    trend_pct = details["trend_pct"]
    vol_20d   = details["vol_20d"]
    last_price = details["last_price"]

    sent_raw  = details["sentiment_score"]

    # ── Helpers: readable magnitude language ──────────────────────────────────
    def _gap_word(pct: float) -> str:
        a = abs(pct)
        if   a < 1.0: return "just barely"
        elif a < 3.0: return "modestly"
        elif a < 6.0: return "clearly"
        else:          return "comfortably"

    def _trend_word(pct: float) -> str:
        a = abs(pct)
        if   a < 1.5: return "drifting"
        elif a < 3.0: return "moving"
        elif a < 6.0: return "climbing" if pct > 0 else "sliding"
        else:          return "surging" if pct > 0 else "falling sharply"

    def _sent_word(s: float) -> str:
        a = abs(s)
        if   a < 0.25: return "mildly"
        elif a < 0.55: return "noticeably"
        else:           return "strongly"

    direction  = "above" if ma_gap >= 0 else "below"
    trend_verb = _trend_word(trend_pct)
    gap_adv    = _gap_word(ma_gap)

    # ── Sentence 1: Technical picture ─────────────────────────────────────────
    if trend == "Uptrend":
        s1 = (
            f"The stock has been {trend_verb} higher over the past week and sits "
            f"{gap_adv} {direction} its 20-day average (${ma20:.2f}) — "
            f"the short-term tape is constructive."
        )
    elif trend == "Downtrend":
        s1 = (
            f"The stock has been {trend_verb} over the past week, trading "
            f"{gap_adv} {direction} its 20-day average (${ma20:.2f}) — "
            f"sellers have been in control."
        )
    else:
        # Sideways — use the MA gap to add colour
        if abs(ma_gap) < 1.0:
            s1 = (
                f"Price is essentially flat, hugging its 20-day average (${ma20:.2f}) "
                f"without committing to a direction — the market is waiting for a catalyst."
            )
        elif ma_gap > 0:
            s1 = (
                f"The stock is drifting sideways but still holding {gap_adv} above "
                f"its 20-day average (${ma20:.2f}), suggesting underlying support."
            )
        else:
            s1 = (
                f"Price is stuck in a sideways range, sitting {gap_adv} below "
                f"its 20-day average (${ma20:.2f}) — buyers haven't stepped in yet."
            )

    # ── Sentence 2: Sentiment, with conflict handling ─────────────────────────
    sent_adv = _sent_word(sent_raw)

    if conflict:
        # The interesting case — technically one direction, sentiment the other
        if trend == "Uptrend":
            s2 = (
                f"What makes this tricky: despite the price action, news flow is "
                f"{sent_adv} negative — that kind of disagreement often signals "
                f"a setup worth watching carefully rather than acting on immediately."
            )
        else:
            s2 = (
                f"Interestingly, news sentiment has turned {sent_adv} positive "
                f"even as the price has been weak — worth watching whether buyers "
                f"step in to close that gap, or the negativity drags sentiment down."
            )
    elif sentiment == "Positive":
        if sent_raw > 0.55:
            s2 = (
                f"News flow is {sent_adv} bullish right now, which tends to attract "
                f"momentum buyers and reinforces the technical picture."
            )
        else:
            s2 = (
                f"The news backdrop is {sent_adv} positive — not a screaming headline "
                f"moment, but it adds a gentle tailwind to the setup."
            )
    elif sentiment == "Negative":
        if sent_raw < -0.55:
            s2 = (
                f"Sentiment is {sent_adv} negative at the moment — that kind of "
                f"headline pressure can weigh on price action even when technicals "
                f"look reasonable."
            )
        else:
            s2 = (
                f"The news backdrop carries a {sent_adv} negative tilt, which adds "
                f"some friction to any recovery attempt."
            )
    else:
        s2 = (
            f"News sentiment is broadly neutral — no major catalyst in either "
            f"direction, so the price action is doing most of the talking."
        )

    # ── Sentence 3: Expectation, calibrated to confidence + volatility ────────
    if volatility == "High":
        vol_note = f" Keep in mind this stock is moving roughly {vol_20d:.1f}% a day on average — position sizing matters."
    elif volatility == "Low":
        vol_note = f" Low volatility ({vol_20d:.1f}% daily) means moves could be smaller than usual."
    else:
        vol_note = ""

    if sig == "BUY":
        if conf >= 75:
            s3 = (
                f"The trend, MA position, and sentiment all point in the same direction — "
                f"that alignment is what drives the {conf:.0f}% confidence here."
                f"{vol_note}"
            )
        elif conf >= 50:
            s3 = (
                f"There's a reasonable case for upside, but it's not a slam dunk at {conf:.0f}% confidence. "
                f"A partial position or a tighter stop makes sense here.{vol_note}"
            )
        else:
            s3 = (
                f"This reads as a tentative buy at best — confidence is only {conf:.0f}%, "
                f"reflecting the mixed picture. Wait for the setup to sharpen before adding size.{vol_note}"
            )
    elif sig == "SELL":
        if conf >= 75:
            s3 = (
                f"Downside pressure looks real — both the technicals and sentiment are aligned, "
                f"giving this a {conf:.0f}% confidence reading.{vol_note}"
            )
        elif conf >= 50:
            s3 = (
                f"The bear case is present but not overwhelming ({conf:.0f}% confidence). "
                f"Consider reducing exposure rather than pressing short aggressively.{vol_note}"
            )
        else:
            s3 = (
                f"Sell signals exist, but confidence sits at just {conf:.0f}%. "
                f"Better to protect existing positions than to act aggressively on this read.{vol_note}"
            )
    else:  # HOLD
        if conflict:
            s3 = (
                f"With trend and sentiment pointing in opposite directions, sitting on "
                f"the sidelines is the most honest call. The next few sessions should "
                f"resolve which force wins out.{vol_note}"
            )
        elif conf < 30:
            s3 = (
                f"Nothing is clear enough to act on right now. Low confidence ({conf:.0f}%) "
                f"means the risk of being wrong is high in either direction.{vol_note}"
            )
        else:
            s3 = (
                f"No strong edge in either direction at this stage — patience is the trade.{vol_note}"
            )

    return f"{s1} {s2} {s3}"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 · STREAMLIT UI DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

# Color palette matching your existing Stockcast theme
_COLORS = {
    "BUY":  {"border": "#00e5b0", "bg": "rgba(0,229,176,0.08)",  "text": "#00e5b0", "emoji": "🟢"},
    "SELL": {"border": "#ff5f5f", "bg": "rgba(255,95,95,0.08)",  "text": "#ff5f5f", "emoji": "🔴"},
    "HOLD": {"border": "#ffd426", "bg": "rgba(255,212,38,0.08)", "text": "#ffd426", "emoji": "🟡"},
}


def render_signal_card(
    signal_result: dict,
    insight_text: str,
    ticker: str = "",
) -> None:
    """
    Render the full signal UI:
      1. Signal Card   — large, colour-coded BUY / SELL / HOLD + graded confidence bar
      2. AI Insight    — natural-language explanation box
      3. Reason rows   — Trend · Sentiment · Volatility · Conflict flag
    """
    sig        = signal_result["signal"]
    conf       = signal_result["confidence"]
    trend      = signal_result["trend"]
    sent       = signal_result["sentiment"]
    conflict   = signal_result.get("conflict", False)
    volatility = signal_result.get("volatility", "Normal")
    c          = _COLORS[sig]

    conf_int   = int(conf)
    bar_filled = int(conf / 5)  # 20 segments total

    # Confidence label
    if   conf >= 70: conf_label = "HIGH CONFIDENCE"
    elif conf >= 45: conf_label = "MODERATE"
    else:            conf_label = "LOW CONFIDENCE"

    # Confidence bar
    bar_html = "".join(
        f'<span style="display:inline-block;width:20px;height:9px;margin-right:2px;' f'border-radius:2px;background:{c["border"]};' f'opacity:{1.0 if i < bar_filled else 0.12};"></span>'
        for i in range(20)
    )

    trend_color = (
        "#00e5b0" if trend == "Uptrend"
        else "#ff5f5f" if trend == "Downtrend"
        else "#ffd426"
    )
    sent_color = (
        "#00e5b0" if sent == "Positive"
        else "#ff5f5f" if sent == "Negative"
        else "#8a8fa0"
    )
    vol_color = (
        "#ff5f5f" if volatility == "High"
        else "#8a8fa0" if volatility == "Low"
        else "#4d8eff"
    )
    header_label = f"SIGNAL · {ticker}" if ticker else "SIGNAL"

    conflict_html = (
        '<div style="display:inline-flex;align-items:center;gap:.35rem;' 'background:rgba(255,212,38,0.1);border:1px solid rgba(255,212,38,0.35);' 'border-radius:2rem;padding:.2rem .7rem;margin-top:.7rem;">'
        '<span style="font-size:.65rem;color:#ffd426;">⚠</span>'
        '<span style="font-family:Manrope,sans-serif;font-size:.65rem;font-weight:700;' 'color:#ffd426;letter-spacing:.06em;">TREND–SENTIMENT CONFLICT</span>'
        '</div>'
        if conflict else ""
    )

    # ── 1. Signal Card ────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background:{c['bg']};border:1.5px solid {c['border']}; border-left:5px solid {c['border']};border-radius:0 .75rem .75rem 0; padding:1.4rem 1.8rem;margin-bottom:1rem;">
            <div style="font-family:Manrope,sans-serif;font-size:.6rem;font-weight:800; letter-spacing:.18em;text-transform:uppercase;color:#3d4760; margin-bottom:.6rem;">{header_label}</div>
            <div style="display:flex;align-items:center;justify-content:space-between; flex-wrap:wrap;gap:1rem;margin-bottom:1rem;">
                <div style="font-family:IBM Plex Mono,monospace;font-size:2.4rem; font-weight:800;color:{c['text']};letter-spacing:.08em;line-height:1;">
                    {c['emoji']}&nbsp;{sig}
                </div>
                <div style="text-align:right;">
                    <div style="font-family:IBM Plex Mono,monospace;font-size:2rem; font-weight:700;color:{c['text']};">{conf_int}%</div>
                    <div style="font-family:Manrope,sans-serif;font-size:.6rem; letter-spacing:.14em;text-transform:uppercase; color:{c['text']};font-weight:700;margin-top:.1rem;">{conf_label}</div>
                </div>
            </div>
            <div style="margin-bottom:1.1rem;">{bar_html}</div>
            <div style="display:flex;gap:1.8rem;flex-wrap:wrap;">
                <div>
                    <div style="font-family:Manrope,sans-serif;font-size:.58rem;letter-spacing:.14em; text-transform:uppercase;color:#3d4760;font-weight:700;">Trend</div>
                    <div style="font-family:IBM Plex Mono,monospace;font-size:.78rem; color:{trend_color};font-weight:700;margin-top:.2rem;">{trend}</div>
                </div>
                <div>
                    <div style="font-family:Manrope,sans-serif;font-size:.58rem;letter-spacing:.14em; text-transform:uppercase;color:#3d4760;font-weight:700;">Sentiment</div>
                    <div style="font-family:IBM Plex Mono,monospace;font-size:.78rem; color:{sent_color};font-weight:700;margin-top:.2rem;">{sent}</div>
                </div>
                <div>
                    <div style="font-family:Manrope,sans-serif;font-size:.58rem;letter-spacing:.14em; text-transform:uppercase;color:#3d4760;font-weight:700;">Volatility</div>
                    <div style="font-family:IBM Plex Mono,monospace;font-size:.78rem; color:{vol_color};font-weight:700;margin-top:.2rem;">{volatility}</div>
                </div>
            </div>
            {conflict_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 2. AI Insight Box ─────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background:rgba(77,142,255,0.05);border:1px solid rgba(77,142,255,0.2); border-left:4px solid #4d8eff;border-radius:0 .75rem .75rem 0; padding:1.1rem 1.5rem;margin-bottom:.8rem;">
            <div style="font-family:Manrope,sans-serif;font-size:.6rem;font-weight:800; letter-spacing:.16em;text-transform:uppercase;color:#4d8eff; margin-bottom:.5rem;">💡 AI Insight</div>
            <div style="font-family:Manrope,sans-serif;font-size:.84rem;color:#b8c4d8; line-height:1.7;">{insight_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 3. Reason rows ────────────────────────────────────────────────────────
    _reason_rows = [
        ("Trend Direction", trend,                                      trend_color),
        ("Sentiment",       sent,                                       sent_color),
        ("Volatility",      volatility,                                 vol_color),
        ("Signal Conflict", "Yes ⚠" if conflict else "None detected",  "#ffd426" if conflict else "#3e4558"),
    ]
    rows_html = "".join(
        f'<div style="display:flex;justify-content:space-between;align-items:center;' f'padding:.38rem 0;border-bottom:1px solid #1e2740;' f'font-family:IBM Plex Mono,monospace;font-size:.72rem;">'
        f'<span style="color:#3e4558;">{label}</span>'
        f'<span style="color:{color};font-weight:700;">{value}</span></div>'
        for label, value, color in _reason_rows
    )

    st.markdown(
        f"""
        <div style="background:#0f1727;border:1px solid #1e2d45; border-radius:.6rem;padding:.9rem 1.2rem;">
            <div style="font-family:Manrope,sans-serif;font-size:.6rem;font-weight:800; letter-spacing:.16em;text-transform:uppercase;color:#3d4760; margin-bottom:.5rem;">Reason</div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: single function to call all three phases at once
# ─────────────────────────────────────────────────────────────────────────────

def run_signal_ui(df: pd.DataFrame, sentiment_score: float, ticker: str = "") -> dict:
    """
    One-call convenience wrapper — generates signal, insight, and renders the UI.
    Returns the signal dict for downstream use (email alerts, investor report, etc.)

    Example
    -------
    signal = run_signal_ui(df, sentiment_score=0.35, ticker="AAPL")
    # signal == {"signal": "BUY", "confidence": 72.0, "conflict": False, ...}
    """
    signal  = generate_signal(df, sentiment_score)
    insight = generate_insight(df, sentiment_score, signal)
    render_signal_card(signal, insight, ticker=ticker)
    return signal



# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stockcast")

# ── Suppress Streamlit's "missing ScriptRunContext" warnings ──────────────────
import warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

class _SuppressScriptRunContext(logging.Filter):
    _keywords = ("ScriptRunContext", "No runtime found", "MemoryCacheStorageManager")
    def filter(self, record):
        msg = record.getMessage()
        return not any(kw in msg for kw in self._keywords)

_scrc_filter = _SuppressScriptRunContext()
logging.root.addFilter(_scrc_filter)
for _nl in ("streamlit", "streamlit.runtime", "streamlit.runtime.scriptrunner",
            "streamlit.runtime.caching", "tornado"):
    logging.getLogger(_nl).addFilter(_scrc_filter)

# ══════════════════════════════════════════════════════════════════════════════
# ── yfinance helpers ───────────────────────────────────────────────────────────

def _yf_download_with_retry(ticker, retries=3, **kwargs):
    """Download yfinance data with retry logic for rate limiting."""
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
    if last_exc is not None:
        logging.warning(f"yfinance download failed for '{ticker}' after {retries} attempts: {type(last_exc).__name__}: {last_exc}")
    return pd.DataFrame()

@st.cache_data(ttl=300)
def av_get_daily(ticker):
    """Fetch full daily OHLCV via yfinance. Returns DataFrame."""
    df = _yf_download_with_retry(ticker, period="max", interval="1d")
    if df.empty:
        return df
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_localize(None) if idx.tz is not None else idx
    df.index.name = "Date"
    return df.sort_index()

@st.cache_data(ttl=300)
def get_ticker_full(ticker: str) -> dict:
    """Single cached yfinance get_info() call — used by both av_get_quote and av_get_overview.
    Halves the number of yfinance round-trips when both are called for the same ticker."""
    for attempt in range(3):
        try:
            info = yf.Ticker(ticker).get_info() or {}
            if info.get("symbol") or info.get("longName"):
                return info
            if attempt < 2:
                time.sleep(2 + attempt * 2)
        except Exception as e:
            logger.warning("get_ticker_full attempt %d failed for '%s': %s", attempt + 1, ticker, e)
            if attempt < 2:
                time.sleep(2 + attempt * 2)
    return {}

@st.cache_data(ttl=60)
def av_get_quote(ticker):
    """Fetch live quote — derived from get_ticker_full to avoid duplicate yfinance calls."""
    try:
        info       = get_ticker_full(ticker)
        price      = float(info.get("currentPrice") or info.get("regularMarketPrice") or
                           info.get("navPrice") or 0)
        prev_close = float(info.get("previousClose") or info.get("regularMarketPreviousClose") or 0)
        open_price = float(info.get("open") or info.get("regularMarketOpen") or 0)
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
        return {"price": price, "change_pct": change_pct, "prev_close": prev_close, "open": open_price}
    except Exception as e:
        logger.warning("av_get_quote failed for '%s': %s", ticker, e)
        return {"price": 0.0, "change_pct": 0.0, "prev_close": 0.0, "open": 0.0}

@st.cache_data(ttl=180)
def get_live_ticker_tape():
    """Fetch live prices for ticker tape symbols — single batched download."""
    tape_syms = ["AAPL","TSLA","NVDA","MSFT","GOOGL","META","AMZN","AMD","JPM","SPY","QQQ","NFLX"]
    try:
        raw = yf.download(tape_syms, period="2d", interval="1d", progress=False, auto_adjust=True)
        close = raw["Close"] if "Close" in raw.columns else raw
        if isinstance(close.columns, pd.MultiIndex):
            close = close.droplevel(0, axis=1)
        items = []
        for sym in tape_syms:
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
                items.append((sym, f"${price:,.2f}", f"{sign}{chg_pct:.2f}%", arrow, css))
            except Exception as e:
                logger.debug("Ticker tape: skipping %s — %s", sym, e)
                continue
        return items  # ✅ FIX: was missing — function previously returned None on success
    except Exception as e:
        logger.warning("get_live_ticker_tape batch download failed: %s", e)
        return []


@st.cache_data(ttl=3600)
def av_get_overview(ticker_sym):
    """Fetch company overview — delegates to get_ticker_full to avoid duplicate yfinance calls."""
    try:
        info = get_ticker_full(ticker_sym)
        if not info:
            logger.warning("av_get_overview: empty info returned for '%s'", ticker_sym)
            return {}
        return {
            "Symbol":                              ticker_sym,
            "Name":                                info.get("longName", ticker_sym),
            "Sector":                              info.get("sector", "Unknown"),
            "Industry":                            info.get("industry", "Unknown"),
            "MarketCapitalization":                str(info.get("marketCap", 0) or 0),
            "TotalDebt":                           str(info.get("totalDebt", 0) or 0),
            "TotalAssets":                         str(info.get("totalAssets", 0) or 0),
            "CashAndCashEquivalentsAtCarryingValue": str(info.get("totalCash", 0) or 0),
        }
    except Exception as e:
        logger.error("av_get_overview failed for '%s': %s", ticker_sym, e, exc_info=True)
        return {}

@st.cache_data(ttl=3600)
def get_fundamentals_rich(ticker_sym: str) -> dict:
    """Return enriched fundamental data for the valuation panel.

    Pulls from a single get_ticker_full() call so no extra yfinance round-trips.
    Returns an empty dict if data unavailable — callers must guard for missing keys.
    """
    try:
        info = get_ticker_full(ticker_sym)
        if not info:
            return {}

        def _f(k, d=None):
            v = info.get(k, d)
            try: return float(v) if v not in (None, "None", "", "N/A") else d
            except: return d

        # Analyst price targets
        target_mean   = _f("targetMeanPrice")
        target_high   = _f("targetHighPrice")
        target_low    = _f("targetLowPrice")
        current_price = _f("currentPrice") or _f("regularMarketPrice") or 0

        upside_pct = (
            ((target_mean - current_price) / current_price * 100)
            if target_mean and current_price else None
        )

        # Revenue / earnings growth
        rev_growth_yoy = _f("revenueGrowth")   # trailing 12m YoY
        earn_growth_yoy = _f("earningsGrowth")  # trailing 12m YoY

        # Per-share metrics
        eps_ttm     = _f("trailingEps")
        eps_fwd     = _f("forwardEps")
        pe_trailing = _f("trailingPE")
        pe_forward  = _f("forwardPE")
        pb          = _f("priceToBook")
        ps_ttm      = _f("priceToSalesTrailing12Months")
        peg         = _f("pegRatio")
        ev_ebitda   = _f("enterpriseToEbitda")
        roe         = _f("returnOnEquity")
        roa         = _f("returnOnAssets")
        profit_margin = _f("profitMargins")
        gross_margin  = _f("grossMargins")
        op_margin     = _f("operatingMargins")
        div_yield     = _f("dividendYield")
        payout_ratio  = _f("payoutRatio")
        beta          = _f("beta")
        float_shares  = _f("floatShares")
        short_ratio   = _f("shortRatio")      # days to cover
        short_pct_float = _f("shortPercentOfFloat")
        insider_pct   = _f("heldPercentInsiders")
        inst_pct      = _f("heldPercentInstitutions")
        num_analysts  = int(info.get("numberOfAnalystOpinions") or 0)
        rec           = info.get("recommendationKey", "")  # strong_buy / buy / hold / sell

        return {
            "name":           info.get("longName", ticker_sym),
            "sector":         info.get("sector", "Unknown"),
            "industry":       info.get("industry", "Unknown"),
            "description":    (info.get("longBusinessSummary") or "")[:400],
            "current_price":  current_price,
            "target_mean":    target_mean,
            "target_high":    target_high,
            "target_low":     target_low,
            "upside_pct":     upside_pct,
            "num_analysts":   num_analysts,
            "recommendation": rec,
            "rev_growth_yoy": rev_growth_yoy,
            "earn_growth_yoy":earn_growth_yoy,
            "eps_ttm":        eps_ttm,
            "eps_fwd":        eps_fwd,
            "pe_trailing":    pe_trailing,
            "pe_forward":     pe_forward,
            "pb":             pb,
            "ps_ttm":         ps_ttm,
            "peg":            peg,
            "ev_ebitda":      ev_ebitda,
            "roe":            roe,
            "roa":            roa,
            "profit_margin":  profit_margin,
            "gross_margin":   gross_margin,
            "op_margin":      op_margin,
            "div_yield":      div_yield,
            "payout_ratio":   payout_ratio,
            "beta":           beta,
            "float_shares":   float_shares,
            "short_ratio":    short_ratio,
            "short_pct_float":short_pct_float,
            "insider_pct":    insider_pct,
            "inst_pct":       inst_pct,
        }
    except Exception as e:
        logger.error("get_fundamentals_rich failed for '%s': %s", ticker_sym, e)
        return {}

@st.cache_data(ttl=300)
def av_get_news(ticker):
    """Fetch news via yfinance. Handles both old (title at root) and
    new yfinance API (title nested under content dict)."""
    try:
        news = yf.Ticker(ticker).news or []
        results = []
        for n in news[:10]:
            title = (
                n.get("title")
                or (n.get("content") or {}).get("title")
                or ""
            )
            if title:
                results.append({"title": title})
        return results
    except Exception as e:
        logger.warning("av_get_news failed for '%s': %s", ticker, e)
        return []

@st.cache_data(ttl=120)
def get_live_market_indices():
    """Fetch live S&P500, NASDAQ, DOW, VIX via yfinance — single batched download."""
    symbols = {"S&P 500":"^GSPC","NASDAQ 100":"^NDX","DOW JONES":"^DJI","VIX":"^VIX"}
    syms = list(symbols.values())
    result = []
    try:
        raw = yf.download(syms, period="2d", interval="1d", progress=False, auto_adjust=True)
        close = raw["Close"] if "Close" in raw.columns else raw
        if isinstance(close.columns, pd.MultiIndex):
            close = close.droplevel(0, axis=1)
        for name, sym in symbols.items():
            try:
                prices = close[sym].dropna()
                if len(prices) >= 2:
                    price, prev = float(prices.iloc[-1]), float(prices.iloc[-2])
                elif len(prices) == 1:
                    price, prev = float(prices.iloc[-1]), float(prices.iloc[-1])
                else:
                    raise ValueError("no data")
                chg_pct = ((price - prev) / prev * 100) if prev else 0.0
                col  = "#00e5b0" if chg_pct >= 0 else "#ff5f5f"
                sign = "+" if chg_pct >= 0 else ""
                fmt_price = f"{price:,.2f}" if sym != "^VIX" else f"{price:.2f}"
                result.append((name, fmt_price, f"{sign}{chg_pct:.2f}%", col))
            except Exception as e:
                logger.debug("Market indices: no data for %s (%s): %s", name, sym, e)
                result.append((name, "—", "—", "#3e4558"))
    except Exception as e:
        logger.warning("get_live_market_indices batch download failed: %s", e)
        for name in symbols:
            result.append((name, "—", "—", "#3e4558"))
    return result

@st.cache_data(ttl=300)
def get_live_sector_heatmap():
    """Fetch live sector ETF performance via yfinance — single batched download."""
    sector_etfs = {
        "Technology":"XLK","Healthcare":"XLV","Financials":"XLF",
        "Energy":"XLE","Consumer Disc.":"XLY","Industrials":"XLI",
        "Utilities":"XLU","Real Estate":"XLRE","Materials":"XLB","Comm. Services":"XLC"
    }
    syms = list(sector_etfs.values())
    result = []
    try:
        raw = yf.download(syms, period="2d", interval="1d", progress=False, auto_adjust=True)
        close = raw["Close"] if "Close" in raw.columns else raw
        if isinstance(close.columns, pd.MultiIndex):
            close = close.droplevel(0, axis=1)
        for name, sym in sector_etfs.items():
            try:
                prices = close[sym].dropna()
                if len(prices) >= 2:
                    price, prev = float(prices.iloc[-1]), float(prices.iloc[-2])
                elif len(prices) == 1:
                    price, prev = float(prices.iloc[-1]), float(prices.iloc[-1])
                else:
                    raise ValueError("no data")
                chg_pct = ((price - prev) / prev * 100) if prev else 0.0
                col  = "#00e5b0" if chg_pct >= 0 else "#ff5f5f"
                sign = "+" if chg_pct >= 0 else ""
                result.append((name, f"{sign}{chg_pct:.2f}%", col))
            except Exception as e:
                logger.debug("Sector heatmap: no data for %s (%s): %s", name, sym, e)
                result.append((name, "—", "#3e4558"))
    except Exception as e:
        logger.warning("get_live_sector_heatmap batch download failed: %s", e)
        for name in sector_etfs:
            result.append((name, "—", "#3e4558"))
    return result

@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Fetch Fear & Greed index. Tries CNN first, then alternative.me, then VIX-based estimate."""
    # Source 1: CNN
    try:
        r = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            timeout=8, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        )
        if r.status_code == 200 and "application/json" in r.headers.get("Content-Type", ""):
            data = r.json()
            score  = float(data["fear_and_greed"]["score"])
            rating = data["fear_and_greed"]["rating"].replace("_", " ").title()
            return {"score": score, "rating": rating, "source": "CNN"}
    except Exception as e:
        logger.debug("get_fear_greed_index CNN failed: %s", e)

    # Source 2: alternative.me (crypto F&G — decent proxy)
    try:
        r2 = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        if r2.status_code == 200:
            d2    = r2.json()["data"][0]
            score = float(d2["value"])
            rating = d2["value_classification"].title()
            return {"score": score, "rating": rating, "source": "Alt.me"}
    except Exception as e:
        logger.debug("get_fear_greed_index alternative.me failed: %s", e)

    # Source 3: VIX-based estimate
    try:
        vix_df = _yf_download_with_retry("^VIX", period="2d", interval="1d")
        if not vix_df.empty:
            vix = float(vix_df["Close"].dropna().iloc[-1])
            if   vix < 12: score, rating = 80, "Extreme Greed"
            elif vix < 16: score, rating = 65, "Greed"
            elif vix < 20: score, rating = 52, "Neutral"
            elif vix < 28: score, rating = 35, "Fear"
            else:          score, rating = 18, "Extreme Fear"
            return {"score": score, "rating": rating, "source": "VIX estimate"}
    except Exception as e:
        logger.debug("get_fear_greed_index VIX fallback failed: %s", e)

    return None


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP HUB  ·  helper functions
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def get_macro_risk_score():
    """Composite macro climate score 0-100: VIX + S&P momentum + F&G + TLT."""
    factors, score = {}, 50
    try:
        v = _yf_download_with_retry("^VIX", period="5d", interval="1d")
        if not v.empty:
            vv = float(v["Close"].dropna().iloc[-1])
            c = 20 if vv<15 else 10 if vv<20 else 0 if vv<25 else -15 if vv<35 else -25
            lbl = f"VIX {vv:.1f} — {'Very low' if vv<15 else 'Low' if vv<20 else 'Moderate' if vv<25 else 'Elevated' if vv<35 else 'High'} vol"
            factors["Volatility (VIX)"] = (c, lbl); score += c
    except Exception: factors["Volatility (VIX)"] = (0, "VIX — unavailable")
    try:
        s = _yf_download_with_retry("^GSPC", period="30d", interval="1d")
        if not s.empty:
            sc = s["Close"].dropna()
            if len(sc) >= 20:
                pct = (float(sc.iloc[-1]) - float(sc.iloc[-20])) / float(sc.iloc[-20]) * 100
                c = 20 if pct>4 else 10 if pct>1 else 0 if pct>-2 else -10 if pct>-6 else -20
                factors["S&P Momentum"] = (c, f"S&P {pct:+.1f}% 20d"); score += c
    except Exception: factors["S&P Momentum"] = (0, "S&P — unavailable")
    try:
        fg = get_fear_greed_index()
        if fg:
            fgs = fg["score"]
            c = 15 if fgs>=65 else 5 if fgs>=45 else -10 if fgs>=30 else -20
            factors["Fear & Greed"] = (c, f"F&G {fgs:.0f} — {'Greed' if fgs>=65 else 'Neutral' if fgs>=45 else 'Fear' if fgs>=30 else 'Extreme Fear'}"); score += c
    except Exception: factors["Fear & Greed"] = (0, "F&G — unavailable")
    try:
        t = _yf_download_with_retry("TLT", period="30d", interval="1d")
        if not t.empty:
            tc = t["Close"].dropna()
            if len(tc) >= 10:
                pct = (float(tc.iloc[-1]) - float(tc.iloc[-10])) / float(tc.iloc[-10]) * 100
                c = -10 if pct>2 else 5 if pct>-1 else 10
                factors["Bond Market (TLT)"] = (c, f"TLT {pct:+.1f}% 10d"); score += c
    except Exception: factors["Bond Market (TLT)"] = (0, "TLT — unavailable")
    score = max(0, min(100, score))
    if score >= 70:   label, color, verdict = "RAISE NOW", "#00e5b0", "Macro conditions are favorable — low volatility, positive momentum, risk-on sentiment. Strong window to fundraise or deploy capital."
    elif score >= 52: label, color, verdict = "PROCEED",   "#4d8eff", "Broadly positive but mixed conditions. Manageable risk for fundraising with solid diligence."
    elif score >= 35: label, color, verdict = "CAUTION",   "#ffd426", "Uncertain environment — elevated volatility or fear signals. Consider waiting before major capital deployment."
    else:             label, color, verdict = "WAIT",      "#ff5f5f", "Hostile macro environment — high volatility, negative momentum, extreme fear. Preserve runway and delay fundraising."
    return {"score": score, "label": label, "color": color, "verdict": verdict, "factors": factors}


TREASURY_PROFILES = {
    "Ultra-safe (T-Bills)": {"tickers":["BIL","SGOV","SHV"],    "desc":"3-month T-bill ETFs — near-zero risk, highest liquidity. Ideal for <6-month runway parking."},
    "Short-term Bonds":     {"tickers":["SHY","VGSH","BSV"],    "desc":"1–3yr Treasury ETFs — modest yield bump over T-bills, still low duration risk."},
    "Intermediate Bonds":   {"tickers":["IEF","VGIT","BND"],    "desc":"3–10yr Treasuries — meaningful yield, some interest rate sensitivity. 12–24 month horizon."},
    "Dividend / Income":    {"tickers":["VYM","SCHD","HDV"],    "desc":"Dividend equity ETFs — higher yield with equity risk. For surplus capital beyond 18-month runway."},
    "Balanced (60/40)":     {"tickers":["AOA","AOR","AOM"],     "desc":"All-in-one allocation ETFs — diversified, auto-rebalanced. For non-critical reserves."},
}

SECTOR_COMPS = {
    "SaaS / Cloud":          ["CRM","NOW","SNOW","DDOG","ZS","MDB","HUBS"],
    "Fintech / Payments":    ["V","MA","PYPL","SQ","AFRM","SOFI","NU"],
    "AI / Semiconductors":   ["NVDA","AMD","INTC","AVGO","QCOM","ARM","SMCI"],
    "E-Commerce":            ["AMZN","SHOP","ETSY","WMT","TGT","EBAY"],
    "Healthcare / Biotech":  ["JNJ","UNH","ABBV","PFE","MRNA","AMGN","GILD"],
    "Mobility / EV":         ["TSLA","GM","F","RIVN","LCID","NIO","XPEV"],
    "Cybersecurity":         ["CRWD","PANW","FTNT","ZS","OKTA","S"],
    "Consumer / Brands":     ["AAPL","NKE","MCD","SBUX","LULU","CMG"],
    "Media / Entertainment": ["NFLX","DIS","SPOT","META","SNAP","RDDT"],
    "Energy / CleanTech":    ["ENPH","FSLR","NEE","PLUG","BE","RUN"],
}

@st.cache_data(ttl=180)
def get_comp_snapshot(tickers: list) -> list:
    results = []
    for sym in tickers:
        try:
            info = get_ticker_full(sym)
            q    = av_get_quote(sym)
            w52h = info.get("fiftyTwoWeekHigh", 0) or 0
            mktcap = info.get("marketCap", 0) or 0
            pe     = info.get("trailingPE") or info.get("forwardPE") or 0
            pfh    = ((q["price"] - w52h) / w52h * 100) if w52h > 0 else 0
            results.append({"ticker":sym, "name":(info.get("longName") or sym)[:28],
                            "price":q["price"], "change_pct":q["change_pct"],
                            "mktcap":mktcap, "pe":pe, "w52h":w52h, "pfh":pfh})
        except Exception:
            results.append({"ticker":sym,"name":sym,"price":0,"change_pct":0,"mktcap":0,"pe":0,"w52h":0,"pfh":0})
    return results

@st.cache_data(ttl=300)
def get_treasury_data(tickers: list) -> list:
    results = []
    for sym in tickers:
        try:
            info = get_ticker_full(sym)
            q    = av_get_quote(sym)
            hist = _yf_download_with_retry(sym, period="1y", interval="1d")
            ret1y = 0.0
            if not hist.empty:
                c = hist["Close"].dropna()
                if len(c) >= 2:
                    ret1y = (float(c.iloc[-1]) - float(c.iloc[0])) / float(c.iloc[0]) * 100
            div_yield = (info.get("dividendYield") or 0) * 100
            aum       = info.get("totalAssets", 0) or 0
            results.append({"ticker":sym, "name":(info.get("longName") or info.get("shortName") or sym)[:32],
                            "price":q["price"], "change_pct":q["change_pct"],
                            "ret1y":ret1y, "div_yield":div_yield, "aum":aum})
        except Exception:
            results.append({"ticker":sym,"name":sym,"price":0,"change_pct":0,"ret1y":0,"div_yield":0,"aum":0})
    return results

def _build_signal_email_html(user_email, ticker, signal, price, score, tp, sl, rr, xp):
    sc = {"BUY":"#00e5b0","SELL":"#ff5f5f"}.get(signal,"#ffd426")
    dt = pd.Timestamp.now().strftime("%B %d, %Y · %H:%M UTC")
    return f"""<!DOCTYPE html><html><body style="margin:0;padding:0;background:#080e1c;font-family:Manrope,sans-serif;">
<div style="max-width:520px;margin:32px auto;background:#0f1727;border-radius:12px;border:1px solid #252f47;overflow:hidden;">
  <div style="background:#141d30;padding:24px 28px;border-bottom:3px solid {sc};">
    <div style="font-family:monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#4d8eff;margin-bottom:4px;">STOCKCAST · SIGNAL ALERT</div>
    <div style="font-size:22px;font-weight:800;color:#e4eafd;">🔔 {ticker} — {signal}</div>
    <div style="font-size:12px;color:#8a8fa0;margin-top:3px;">{dt}</div>
  </div>
  <div style="padding:22px 28px;">
    <table style="width:100%;border-collapse:collapse;">
      {"".join(f'<tr><td style="padding:7px 0;font-family:monospace;font-size:12px;color:#8a8fa0;border-bottom:1px solid #1e2740;">{k}</td><td style="padding:7px 0;font-family:monospace;font-size:14px;font-weight:700;color:{vc};text-align:right;border-bottom:1px solid #1e2740;">{v}</td></tr>' for k,v,vc in [
        ("Signal",signal,sc),("Score",f"{score:+.0f} / ±100","#adc6ff"),
        ("Last Close",f"${price:.2f}","#e4eafd"),("AI Forecast",f"{xp:+.2f}%","#adc6ff"),
        ("Take Profit",f"${tp:.2f}","#00e5b0"),("Stop Loss",f"${sl:.2f}","#ff5f5f"),
        ("Risk/Reward",f"{rr:.2f}×","#ffd426")])}
    </table>
    <div style="margin-top:16px;padding:12px;background:rgba(255,212,38,0.05);border:1px solid rgba(255,212,38,0.2);border-radius:6px;font-size:10px;color:#8a8fa0;line-height:1.6;">
      ⚠ AI-generated signal from technical analysis only. Not financial advice.
    </div>
  </div>
  <div style="padding:14px 28px;border-top:1px solid #252f47;text-align:center;font-size:10px;color:#3e4558;">
    Sent to {user_email} · <a href="https://muawwizghani-stock-forecast.streamlit.app" style="color:#4d8eff;">Open Stockcast</a>
  </div>
</div></body></html>"""

def _build_investor_csv(ticker, df, preds, actual, rmse, mae, mape, r2, comp, se_signal=None):
    import io, csv
    buf = io.StringIO(); w = csv.writer(buf)
    lc  = float(df["Close"].squeeze().iloc[-1])
    h52 = float(df["Close"].squeeze().max())
    l52 = float(df["Close"].squeeze().min())
    sig = comp.get("verdict_short","—") if comp else "—"
    sco = comp.get("total_score",0) if comp else 0
    tp  = comp.get("take_profit",0) if comp else 0
    sl  = comp.get("stop_loss",0) if comp else 0
    rr  = comp.get("risk_reward",0) if comp else 0
    xp  = comp.get("xgb_pct",0) if comp else 0
    w.writerows([
        ["STOCKCAST — Investor Intelligence Report"],
        [f"Ticker: {ticker}",f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}"],
        ["Not financial advice. For research purposes only."],[],
        ["── PRICE SUMMARY ──"],
        ["Last Close ($)",f"{lc:.2f}"],["52w High ($)",f"{h52:.2f}"],["52w Low ($)",f"{l52:.2f}"],
        ["% from 52w High",f"{((lc-h52)/h52*100):+.1f}%"],["Days Analysed",str(len(df))],[],
        ["── AI MODEL QUALITY ──"],
        ["RMSE ($)",f"{rmse:.2f}"],["MAE ($)",f"{mae:.2f}"],["MAPE (%)",f"{mape:.2f}"],["R²",f"{r2:.4f}"],[],
        ["── SIGNAL INTELLIGENCE ──"],
        ["Signal",sig],["Score (±100)",f"{sco:+.0f}"],["XGBoost Forecast %",f"{xp:+.2f}%"],
        ["Take Profit ($)",f"{tp:.2f}"],["Stop Loss ($)",f"{sl:.2f}"],["Risk/Reward",f"{rr:.2f}×"],[],
        ["── AI SIGNAL ENGINE ──"],
        ["SE Signal",(se_signal or {}).get("signal","—")],
        ["SE Confidence (%)",f'{(se_signal or {}).get("confidence",0):.1f}'],
        ["SE Trend",(se_signal or {}).get("trend","—")],
        ["SE Sentiment",(se_signal or {}).get("sentiment","—")],
        ["SE Volatility",(se_signal or {}).get("volatility","—")],
        ["SE Conflict","Yes" if (se_signal or {}).get("conflict",False) else "No"],[],
        ["── DISCLAIMER ──"],
        ["For educational and research purposes only. Not financial advice."],
    ])
    return buf.getvalue()


# =============================================================================
# V3 · BACKTESTING ENGINE
# =============================================================================

def run_advanced_backtest(prices, strategy="momentum", initial_capital=10000.0):
    """CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Equity Curve."""
    if len(prices) < 20:
        return {}
    rets  = prices.pct_change().dropna()
    years = max(len(prices) / 252.0, 0.01)
    total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    cagr  = (pow(1 + total_return / 100, 1 / years) - 1) * 100
    ann_vol = float(rets.std() * (252 ** 0.5) * 100)
    rf = 4.5
    sharpe  = (cagr - rf) / ann_vol if ann_vol > 0 else 0.0
    neg     = rets[rets < 0]
    d_vol   = float(neg.std() * (252 ** 0.5) * 100) if len(neg) > 1 else ann_vol
    sortino = (cagr - rf) / d_vol if d_vol > 0 else 0.0
    rolling_max = prices.cummax()
    dd_series   = (prices - rolling_max) / rolling_max * 100
    max_dd  = float(dd_series.min())
    calmar  = cagr / abs(max_dd) if max_dd != 0 else 0.0
    sigs    = (prices.pct_change(5).shift(-1) > 0).astype(int).values
    trade_r = rets.values * sigs[:len(rets)]
    win_rate = float((trade_r > 0).mean() * 100) if len(trade_r) else 50.0
    equity   = (initial_capital * prices / prices.iloc[0]).tolist()
    monthly  = prices.resample("ME").last().pct_change().dropna() * 100
    monthly_r = {d.strftime("%b %Y"): round(float(v), 2) for d, v in monthly.items()}
    return {
        "total_return": round(total_return, 2), "cagr": round(cagr, 2),
        "sharpe": round(sharpe, 2), "sortino": round(sortino, 2),
        "calmar": round(calmar, 2), "max_drawdown": round(max_dd, 2),
        "win_rate": round(win_rate, 1), "volatility": round(ann_vol, 2),
        "trades": int(len(prices) / 10), "equity_curve": equity,
        "drawdown_series": dd_series.tolist(), "monthly_returns": monthly_r,
    }


def render_backtest_dashboard(ticker, df, preds, actual):
    """Full institutional backtesting UI inside Streamlit."""
    st.markdown("""<div style="display:flex;align-items:center;gap:.75rem;margin:1.5rem 0 .6rem;">
      <div style="font-family:Manrope,sans-serif;font-size:1rem;font-weight:800;color:#e4eafd;">
        \U0001f4c8 Strategy Backtester</div>
      <span style="background:rgba(255,212,38,.1);border:1px solid rgba(255,212,38,.3);
        color:#ffd426;font-family:IBM Plex Mono,monospace;font-size:.58rem;font-weight:700;
        padding:.15rem .55rem;border-radius:.25rem;letter-spacing:.1em;">INVESTOR-GRADE</span>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        strategy = st.selectbox("Strategy",
            ["Momentum", "Mean Reversion", "MACD Crossover", "RSI Bands", "Buy & Hold"],
            key="bt_strat")
    with c2:
        benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM", "DIA"], key="bt_bench")
    with c3:
        capital = st.number_input("Initial Capital ($)", min_value=1000,
            max_value=10_000_000, value=10000, step=1000, key="bt_cap")

    prices = df["Close"].squeeze().dropna()
    if len(prices) < 30:
        st.warning("Need 30+ days of price history for backtesting.")
        return

    bt = run_advanced_backtest(prices, strategy=strategy, initial_capital=float(capital))
    if not bt:
        st.warning("Backtest could not be computed.")
        return

    kpis = [
        ("Total Return",  f"{bt['total_return']:+.2f}%", "#00e5b0" if bt["total_return"] >= 0 else "#ff5f5f"),
        ("CAGR",          f"{bt['cagr']:.2f}%",          "#4d8eff"),
        ("Sharpe Ratio",  f"{bt['sharpe']:.2f}",         "#00e5b0" if bt["sharpe"] > 1 else "#ffd426"),
        ("Max Drawdown",  f"{bt['max_drawdown']:.2f}%",  "#ff5f5f"),
        ("Win Rate",      f"{bt['win_rate']:.1f}%",      "#00e5b0"),
        ("Volatility",    f"{bt['volatility']:.2f}%",    "#ffd426"),
        ("Sortino",       f"{bt['sortino']:.2f}",        "#00e5b0" if bt["sortino"] > 1 else "#ffd426"),
        ("Calmar",        f"{bt['calmar']:.2f}",         "#adc6ff"),
    ]
    cols = st.columns(len(kpis))
    for col, (label, value, color) in zip(cols, kpis):
        col.markdown(
            f'<div style="background:#0f1727;border:1px solid #252f47;border-top:2px solid {color};'
            f'border-radius:.6rem;padding:.85rem 1rem;">'
            f'<div style="font-family:Manrope,sans-serif;font-size:.58rem;letter-spacing:.12em;'
            f'text-transform:uppercase;color:#3e4558;margin-bottom:.35rem;">{label}</div>'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.1rem;font-weight:700;'
            f'color:{color};">{value}</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    eq   = bt["equity_curve"]
    bh   = [float(capital) * (prices.iloc[min(i, len(prices)-1)] / prices.iloc[0]) for i in range(len(eq))]
    labs = [prices.index[min(i, len(prices)-1)].strftime("%b %Y") for i in range(len(eq))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labs, y=eq, name=strategy,
        line=dict(color="#4d8eff", width=2), fill="tozeroy",
        fillcolor="rgba(77,142,255,0.06)"))
    fig.add_trace(go.Scatter(x=labs, y=bh, name="Buy & Hold",
        line=dict(color="#3e4558", width=1.5, dash="dot")))
    fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()},
        title=dict(text=f"{ticker} \xb7 Equity Curve", font=dict(color="#00e5b0", size=12)),
        height=260, yaxis_tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        months = list(bt["monthly_returns"].keys())[-12:]
        vals   = [bt["monthly_returns"][m] for m in months]
        fig2   = go.Figure(go.Bar(
            x=months, y=vals,
            text=[f"{v:+.1f}%" for v in vals],
            textfont=dict(size=8), textposition="outside",
            marker_color=["rgba(0,229,176,.7)" if v >= 0 else "rgba(255,95,95,.7)" for v in vals]))
        fig2.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()},
            title=dict(text="Monthly Returns", font=dict(color="#00e5b0", size=12)),
            height=230, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    with cb:
        dd  = bt["drawdown_series"]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=labs[:len(dd)], y=dd, name="Drawdown",
            line=dict(color="#ff5f5f", width=1.5), fill="tozeroy",
            fillcolor="rgba(255,95,95,.1)"))
        fig3.add_hline(y=0, line_color="#3e4558", line_width=1)
        fig3.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()},
            title=dict(text="Drawdown (%)", font=dict(color="#ff5f5f", size=12)),
            height=230, yaxis_ticksuffix="%")
        st.plotly_chart(fig3, use_container_width=True)

    rows = [
        ("Total Return (%)",     f"{bt['total_return']:+.2f}", "---"),
        ("CAGR (%)",             f"{bt['cagr']:.2f}",         "---"),
        ("Sharpe Ratio",         f"{bt['sharpe']:.2f}",        "---"),
        ("Sortino Ratio",        f"{bt['sortino']:.2f}",       "---"),
        ("Calmar Ratio",         f"{bt['calmar']:.2f}",        "---"),
        ("Max Drawdown (%)",     f"{bt['max_drawdown']:.2f}",  "---"),
        ("Volatility Ann. (%)",  f"{bt['volatility']:.2f}",    "---"),
        ("Win Rate (%)",         f"{bt['win_rate']:.1f}",      "~50"),
        ("No. of Trades",        str(bt["trades"]),            "N/A"),
    ]
    header = (
        '<table style="width:100%;border-collapse:collapse;font-family:IBM Plex Mono,monospace;font-size:.72rem;">'
        '<thead><tr style="border-bottom:1px solid #252f47;">'
        '<th style="padding:.5rem .8rem;text-align:left;color:#3e4558;font-size:.6rem;">Metric</th>'
        f'<th style="padding:.5rem .8rem;text-align:right;color:#4d8eff;font-size:.6rem;">{strategy}</th>'
        f'<th style="padding:.5rem .8rem;text-align:right;color:#3e4558;font-size:.6rem;">{benchmark}</th>'
        '</tr></thead><tbody>'
    )
    body = "".join(
        f'<tr style="border-bottom:1px solid #1e2740;">'
        f'<td style="padding:.45rem .8rem;color:#8a8fa0;">{lbl}</td>'
        f'<td style="padding:.45rem .8rem;text-align:right;color:#e4eafd;">{sv}</td>'
        f'<td style="padding:.45rem .8rem;text-align:right;color:#3e4558;">{bv}</td></tr>'
        for lbl, sv, bv in rows
    )
    st.markdown(
        f'<div style="background:#0a1120;border:1px solid #1e2740;border-radius:.6rem;overflow:hidden;">'
        f'{header}{body}</tbody></table></div>',
        unsafe_allow_html=True)
    st.caption("\u26a0 Backtesting uses historical data. Past results do not guarantee future returns.")


# =============================================================================
# V3 · BROKERAGE INTEGRATIONS
# =============================================================================

def get_zerodha_holdings(api_key, access_token):
    """Live Zerodha holdings via Kite Connect. pip install kiteconnect"""
    if not _KITE_OK:
        return []
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        return [{"symbol": h["tradingsymbol"], "qty": h["quantity"],
                 "avg_price": h["average_price"], "last_price": h["last_price"],
                 "pnl": h["pnl"],
                 "pnl_pct": (h["pnl"] / max(h["average_price"] * h["quantity"], 0.01)) * 100,
                 "broker": "Zerodha"} for h in kite.holdings()]
    except Exception as e:
        logging.warning("Zerodha: %s", e)
        return []


def get_upstox_holdings(api_key, access_token):
    """Live Upstox holdings. pip install upstox-python-sdk"""
    if not _UPSTOX_OK:
        return []
    try:
        cfg = upstox_client.Configuration(host="https://api.upstox.com/v2")
        cfg.access_token = access_token
        api = upstox_client.PortfolioApi(upstox_client.ApiClient(cfg))
        return [{"symbol": h.tradingsymbol, "qty": h.quantity,
                 "avg_price": h.average_price, "last_price": h.last_price,
                 "pnl": (h.last_price - h.average_price) * h.quantity,
                 "pnl_pct": ((h.last_price / max(h.average_price, 0.01)) - 1) * 100,
                 "broker": "Upstox"} for h in (api.get_holdings().data or [])]
    except Exception as e:
        logging.warning("Upstox: %s", e)
        return []


def get_alpaca_holdings(api_key, api_secret, base_url="https://paper-api.alpaca.markets"):
    """Live Alpaca positions. pip install alpaca-trade-api"""
    if not _ALPACA_OK:
        return []
    try:
        api = tradeapi.REST(api_key, api_secret, base_url, api_version="v2")
        return [{"symbol": p.symbol, "qty": float(p.qty),
                 "avg_price": float(p.avg_entry_price), "last_price": float(p.current_price),
                 "pnl": float(p.unrealized_pl), "pnl_pct": float(p.unrealized_plpc) * 100,
                 "broker": "Alpaca"} for p in api.list_positions()]
    except Exception as e:
        logging.warning("Alpaca: %s", e)
        return []


def render_brokerage_panel():
    """Brokerage integration UI panel."""
    st.markdown("""<div style="font-family:Manrope,sans-serif;font-size:1rem;font-weight:800;
        color:#e4eafd;margin-bottom:.4rem;">\U0001f517 Live Portfolio Sync</div>
      <div style="font-size:.82rem;color:#8a8fa0;line-height:1.6;margin-bottom:1.2rem;">
        Connect your broker to sync live holdings, P&amp;L and order history.</div>""",
        unsafe_allow_html=True)

    brokers = [
        ("Zerodha Kite", "\U0001f7e0", "#ff6b35", ["Live Holdings", "Order Insights", "Historical Trades"], "IN"),
        ("Upstox",       "\U0001f535", "#3b82f6", ["Portfolio Sync", "Market Depth", "Margin Data"],        "IN"),
        ("Alpaca",       "\U0001f999", "#f59e0b", ["Fractional Shares", "Paper Trading", "Live Orders"],    "US"),
        ("IBKR",         "\U0001f310", "#6b7280", ["Global Markets", "Options", "Futures"],                 "Global \u2014 coming soon"),
        ("Angel One",    "\u2b50",     "#6b7280", ["Smart API", "Mutual Funds"],                            "IN \u2014 coming soon"),
    ]
    cols = st.columns(len(brokers))
    for col, (name, logo, color, feats, region) in zip(cols, brokers):
        coming = "coming soon" in region
        feats_html = "".join(
            f'<div style="font-size:.68rem;color:#8a8fa0;padding:.1rem 0;">'
            f'<span style="color:#00e5b0;font-size:.6rem;">\u2713</span> {f}</div>'
            for f in feats)
        with col:
            st.markdown(
                f'<div style="background:#0f1727;border:1px solid {"#252f47" if coming else color+"44"};'
                f'border-top:2px solid {color};border-radius:.75rem;padding:1.1rem;'
                f'opacity:{"0.5" if coming else "1"};">'
                f'<div style="font-size:1.5rem;margin-bottom:.4rem;">{logo}</div>'
                f'<div style="font-family:Manrope,sans-serif;font-size:.8rem;font-weight:700;color:#e4eafd;">{name}</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:.6rem;color:#3e4558;margin-bottom:.6rem;">{region}</div>'
                f'{feats_html}</div>', unsafe_allow_html=True)
            if not coming:
                if st.button(f"Connect {name}", key=f"br_{name}", use_container_width=True):
                    st.info(f"Add {name} credentials to .streamlit/secrets.toml")

    with st.expander("\U0001f511 Setup Guide \u2014 secrets.toml"):
        st.code("""[secrets]
ZERODHA_API_KEY      = "your_key"
ZERODHA_ACCESS_TOKEN = "your_token"
UPSTOX_API_KEY       = "your_key"
UPSTOX_ACCESS_TOKEN  = "your_token"
ALPACA_KEY           = "your_key_id"
ALPACA_SECRET        = "your_secret"
ALPACA_BASE_URL      = "https://paper-api.alpaca.markets"
ANTHROPIC_API_KEY    = "sk-ant-..."
POSTHOG_API_KEY      = "phc_...""", language="toml")
        st.code("pip install kiteconnect upstox-python-sdk alpaca-trade-api posthog", language="bash")


# =============================================================================
# V3 · AI INVESTOR REPORTS
# =============================================================================

def generate_ai_investor_report(ticker, report_type, signal_data=None, fundamentals=None):
    """Institutional investor report via Claude claude-sonnet-4-20250514."""
    api_key = ""
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    sig_ctx = ""
    if signal_data:
        sig_ctx = (
            f"AI signal: {signal_data.get('verdict_short', '---')} "
            f"(score {signal_data.get('total_score', 0):+.0f}/100), "
            f"TP ${signal_data.get('take_profit', 0):.2f}, "
            f"SL ${signal_data.get('stop_loss', 0):.2f}. "
        )
    fund_ctx = ""
    if fundamentals:
        fund_ctx = (
            f"P/E {fundamentals.get('pe_trailing', '---')}, "
            f"Forward P/E {fundamentals.get('pe_forward', '---')}, "
            f"EV/EBITDA {fundamentals.get('ev_ebitda', '---')}. "
        )

    prompts = {
        "portfolio": (
            f"Write a 3-paragraph institutional portfolio summary for {ticker}. "
            f"{sig_ctx}{fund_ctx}"
            "P1: Technical positioning and signal. "
            "P2: Key risk factors and macro. "
            "P3: Strategic recommendation with price targets. Under 220 words."
        ),
        "risk": (
            f"Write a 2-paragraph institutional risk analysis for {ticker}. "
            f"{sig_ctx}"
            "P1: Quantitative risk metrics (vol, drawdown, VaR). "
            "P2: Qualitative risks and mitigation. Under 180 words."
        ),
        "weekly": (
            f"Write a 2-paragraph weekly outlook for {ticker}'s sector. "
            f"{sig_ctx}"
            "P1: Technical levels and catalyst calendar. "
            "P2: Macro backdrop and 5-day trade thesis. Under 160 words."
        ),
        "stock": (
            f"Write a 3-paragraph stock deep dive on {ticker}. "
            f"{fund_ctx}"
            "P1: Business moat. P2: Valuation vs peers. "
            "P3: Catalysts and risks. Under 230 words."
        ),
    }
    prompt = prompts.get(report_type, prompts["portfolio"])

    if not api_key:
        return (
            f"[AI Report --- {ticker}]\n\n"
            "Add ANTHROPIC_API_KEY to .streamlit/secrets.toml to enable "
            "AI-generated investor reports powered by Claude claude-sonnet-4-20250514."
        )
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        ).strip()
    except Exception as e:
        logging.error("AI report failed: %s", e)
        return f"Report generation failed: {e}"


def render_ai_report_panel(ticker, signal_data=None, fundamentals=None):
    """AI Investor Report UI tab."""
    report_types = {
        "\U0001f4ca Portfolio Summary": "portfolio",
        "\u26a0\ufe0f Risk Analysis":   "risk",
        "\U0001f4c5 Weekly Outlook":    "weekly",
        "\U0001f50d Stock Deep Dive":   "stock",
    }
    st.markdown("""<div style="font-family:Manrope,sans-serif;font-size:1rem;font-weight:800;
        color:#e4eafd;margin-bottom:.4rem;">\U0001f916 AI Investor Reports</div>
      <div style="font-size:.82rem;color:#8a8fa0;margin-bottom:1rem;">
        Powered by Claude claude-sonnet-4-20250514 \xb7 Institutional-grade \xb7 PDF-ready</div>""",
        unsafe_allow_html=True)

    sel   = st.radio("Report type", list(report_types.keys()), horizontal=True, key="air_type")
    rtype = report_types[sel]

    if st.button(f"\U0001f916 Generate {sel}", key="gen_air", use_container_width=True, type="primary"):
        with st.spinner(f"Generating {sel} for {ticker}\u2026"):
            text = generate_ai_investor_report(ticker, rtype, signal_data, fundamentals)
        st.session_state["_air_text"] = text
        st.session_state["_air_meta"] = f"{ticker} \xb7 {sel}"

    if "_air_text" in st.session_state:
        escaped = _html_mod.escape(st.session_state["_air_text"])
        meta    = st.session_state.get("_air_meta", "")
        st.markdown(
            f'<div style="background:#0f1727;border:1px solid #252f47;border-left:3px solid #9d7ff5;'
            f'border-radius:0 .75rem .75rem 0;padding:1.3rem 1.6rem;margin-top:.8rem;">'
            f'<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.85rem;">'
            f'<span style="background:rgba(157,127,245,.1);border:1px solid rgba(157,127,245,.3);'
            f'color:#9d7ff5;font-family:IBM Plex Mono,monospace;font-size:.58rem;font-weight:700;'
            f'padding:.15rem .55rem;border-radius:.25rem;">Claude claude-sonnet-4-20250514</span>'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:.62rem;color:#3e4558;">{meta}</span>'
            f'</div>'
            f'<div style="font-family:Georgia,serif;font-size:.84rem;color:#c8cedd;'
            f'line-height:1.85;white-space:pre-wrap;">{escaped}</div></div>',
            unsafe_allow_html=True)

        ts = pd.Timestamp.now().strftime("%Y%m%d")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "\U0001f4c4 Download TXT",
                data=st.session_state["_air_text"],
                file_name=f"stockcast_{ticker}_{ts}.txt",
                mime="text/plain", use_container_width=True)
        with c2:
            report_csv = (
                f"Stockcast AI Report,{ticker}\n"
                f"Type,{sel}\n"
                f"Generated,{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                f"Report\n"
            ) + st.session_state["_air_text"]
            st.download_button(
                "\U0001f4ca Download CSV",
                data=report_csv,
                file_name=f"stockcast_report_{ticker}_{ts}.csv",
                mime="text/csv", use_container_width=True)

    st.caption("\u26a0 For educational purposes only. Not financial advice.")


# =============================================================================
# V3 · PRODUCT ANALYTICS (PostHog)
# =============================================================================

def track_event(event, properties=None, user_id=None):
    """Track event via PostHog. Silent no-op if not configured."""
    if not _POSTHOG_OK:
        return
    try:
        uid = user_id or st.session_state.get("user_id", "anonymous")
        _posthog_lib.capture(uid, event, properties or {})
    except Exception:
        pass


# =============================================================================
# V3 · BRAND & SOCIAL SHARING
# =============================================================================

def render_share_card(ticker, signal, price, confidence, target, stop):
    """Social-ready AI signal share card with download."""
    sig_color = {"BUY": "#00e5b0", "SELL": "#ff5f5f"}.get(signal, "#ffd426")
    sig_icon  = {"BUY": "\u2191", "SELL": "\u2193"}.get(signal, "\u2192")
    share_text = (
        f"\U0001f916 Stockcast AI Signal \u2014 {ticker}\n"
        f"{sig_icon} {signal} @ ${price:.2f}\n"
        f"Confidence: {confidence}% | Target: ${target:.2f} | Stop: ${stop:.2f}\n"
        f"#AI #Stocks #{ticker} #Stockcast\nstockcast.io"
    )
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0f1727,#141d30);'
        f'border:1px solid #252f47;border-top:2px solid {sig_color};'
        f'border-radius:.75rem;padding:1.2rem 1.4rem;">'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:.58rem;color:#3e4558;'
        f'letter-spacing:.14em;text-transform:uppercase;margin-bottom:.3rem;">STOCKCAST \xb7 SIGNAL</div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.8rem;font-weight:700;'
        f'color:{sig_color};">{sig_icon} {signal}</div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:.8rem;color:#e4eafd;'
        f'margin:.2rem 0 .4rem;">{ticker} \xb7 ${price:.2f}</div>'
        f'<div style="font-size:.72rem;color:#8a8fa0;">'
        f'Confidence: {confidence}% \xb7 Target: ${target:.2f} \xb7 Stop: ${stop:.2f}</div></div>',
        unsafe_allow_html=True)
    st.download_button(
        "\U0001f517 Copy Share Text", data=share_text,
        file_name=f"stockcast_signal_{ticker}.txt", mime="text/plain",
        key=f"share_{ticker}_{signal}")


def render_referral_panel(user_email=""):
    """Referral code and share UI."""
    suffix  = "".join(c for c in user_email.upper().replace("@", "").replace(".", "") if c.isalpha())[:4]
    ref_num = str(abs(hash(user_email)) % 1000)
    ref_code = f"STOCKCAST-{suffix}{ref_num}"
    st.markdown(
        f'<div style="background:rgba(255,212,38,.05);border:1px solid rgba(255,212,38,.25);'
        f'border-radius:.75rem;padding:1.1rem 1.4rem;margin-bottom:1rem;">'
        f'<div style="font-size:.62rem;font-weight:800;letter-spacing:.14em;'
        f'text-transform:uppercase;color:#ffd426;margin-bottom:.5rem;">Your Referral Code</div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.1rem;color:#ffd426;'
        f'letter-spacing:.12em;margin-bottom:.5rem;">{ref_code}</div>'
        f'<div style="font-size:.75rem;color:#8a8fa0;">Earn 1 free Pro month per successful referral.</div>'
        f'</div>', unsafe_allow_html=True)
    st.download_button(
        "\U0001f517 Copy Referral Link",
        data=f"https://stockcast.io?ref={ref_code}",
        file_name="stockcast_referral.txt", mime="text/plain", key="ref_copy")


# =============================================================================
# V3 · UPSELL GATES
# =============================================================================

PLAN_FEATURES = {
    "free":          {"backtest": False, "ai_reports": False, "brokers": False},
    "pro":           {"backtest": True,  "ai_reports": True,  "brokers": True},
    "institutional": {"backtest": True,  "ai_reports": True,  "brokers": True},
}


def render_upsell_gate(feature, plan="free", source="unknown"):
    """Show upsell prompt if gated. Returns True if user has access."""
    if PLAN_FEATURES.get(plan, PLAN_FEATURES["free"]).get(feature, False):
        return True
    names = {
        "backtest":   "Strategy Backtesting",
        "ai_reports": "AI Investor Reports",
        "brokers":    "Brokerage Integration",
    }
    fname = names.get(feature, feature.replace("_", " ").title())
    track_event("upgrade_intent", {"feature": feature, "source": source})
    st.markdown(
        f'<div style="background:rgba(77,142,255,0.05);border:1px solid rgba(77,142,255,0.2);'
        f'border-left:4px solid #4d8eff;border-radius:0 .75rem .75rem 0;'
        f'padding:1.1rem 1.5rem;margin:.5rem 0;">'
        f'<div style="font-family:Manrope,sans-serif;font-size:.62rem;font-weight:800;'
        f'letter-spacing:.14em;text-transform:uppercase;color:#4d8eff;margin-bottom:.4rem;">'
        f'\U0001f512 Pro Feature \u2014 {fname}</div>'
        f'<div style="font-size:.82rem;color:#8a8fa0;line-height:1.6;margin-bottom:.7rem;">'
        f'Upgrade to <b style="color:#e4eafd;">Pro ($19/mo)</b> to unlock {fname}.</div></div>',
        unsafe_allow_html=True)
    st.button(
        "\u2b06 Upgrade to Pro \u2014 $19/mo",
        key=f"up_{feature}_{source}", type="primary", use_container_width=True)
    return False


import nltk

def _ensure_nltk_data():
    """Lazily download required NLTK tokenizer data on first use."""
    for _nltk_pkg in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{_nltk_pkg}')
        except LookupError:
            try:
                nltk.download(_nltk_pkg, quiet=True)
            except Exception as e:
                logger.warning("Failed to download NLTK package '%s': %s", _nltk_pkg, e)

# ── Supabase config ────────────────────────────────────────────────────────────
# Credentials are loaded from Streamlit secrets (secrets.toml) or environment variables.
# Never hardcode credentials in source code.
# Set up: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
SUPABASE_URL: str = ""
SUPABASE_KEY: str = ""
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except (KeyError, FileNotFoundError):
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

_SUPABASE_MISSING = not SUPABASE_URL or not SUPABASE_KEY
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if not _SUPABASE_MISSING else None

# ── Portfolio Supabase helpers ─────────────────────────────────────────────────
# Required Supabase tables (run once in your Supabase SQL editor):
#
#   CREATE TABLE IF NOT EXISTS portfolio_holdings (
#     id           UUID DEFAULT gen_random_uuid() PRIMARY KEY,
#     user_id      UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
#     ticker       TEXT NOT NULL,
#     name         TEXT,
#     sector       TEXT,
#     qty          FLOAT NOT NULL,
#     avg_cost     FLOAT NOT NULL,
#     current_price FLOAT,
#     pl           FLOAT,
#     pl_pct       FLOAT,
#     created_at   TIMESTAMPTZ DEFAULT NOW(),
#     UNIQUE(user_id, ticker)
#   );
#
#   CREATE TABLE IF NOT EXISTS portfolio_history (
#     id         UUID DEFAULT gen_random_uuid() PRIMARY KEY,
#     user_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
#     date       TEXT,
#     type       TEXT,
#     ticker     TEXT,
#     shares     FLOAT,
#     price      FLOAT,
#     amount     FLOAT,
#     created_at TIMESTAMPTZ DEFAULT NOW()
#   );

def _sb_load_portfolio(user_id: str) -> list:
    try:
        res = supabase.table("portfolio_holdings").select("*").eq("user_id", user_id).execute()
        rows = res.data or []
        # Sanitize: fill None numeric fields so UI arithmetic never crashes
        for r in rows:
            r["current_price"] = float(r.get("current_price") or r.get("avg_cost") or 0.0)
            r["pl"]      = float(r.get("pl")      or 0.0)
            r["pl_pct"]  = float(r.get("pl_pct")  or 0.0)
            r["qty"]     = float(r.get("qty")      or 0.0)
            r["avg_cost"]= float(r.get("avg_cost") or 0.0)
            r["name"]    = r.get("name")   or r.get("ticker", "")
            r["sector"]  = r.get("sector") or "Unknown"
        return rows
    except Exception as e:
        logger.error("_sb_load_portfolio failed for user '%s': %s", user_id, e, exc_info=True)
        return []

def _sb_load_history(user_id: str) -> list:
    try:
        res = (supabase.table("portfolio_history").select("*")
               .eq("user_id", user_id).order("created_at", desc=True).limit(100).execute())
        return res.data or []
    except Exception as e:
        logger.error("_sb_load_history failed for user '%s': %s", user_id, e, exc_info=True)
        return []

def _sb_upsert_holding(user_id: str, h: dict):
    try:
        supabase.table("portfolio_holdings").upsert({
            "user_id": user_id,
            "ticker":        h["ticker"],
            "name":          h.get("name", h["ticker"]),
            "sector":        h.get("sector", "Unknown"),
            "qty":           h["qty"],
            "avg_cost":      h["avg_cost"],
            "current_price": h.get("current_price", h["avg_cost"]),
            "pl":            h.get("pl", 0.0),
            "pl_pct":        h.get("pl_pct", 0.0),
        }, on_conflict="user_id,ticker").execute()
    except Exception as e:
        logger.error("_sb_upsert_holding failed for user '%s', ticker '%s': %s",
                     user_id, h.get("ticker"), e, exc_info=True)
        st.warning(f"⚠ Could not save holding {h.get('ticker', '')} — please retry.")

def _sb_delete_holding(user_id: str, ticker: str):
    try:
        supabase.table("portfolio_holdings").delete().eq("user_id", user_id).eq("ticker", ticker).execute()
    except Exception as e:
        logger.error("_sb_delete_holding failed for user '%s', ticker '%s': %s",
                     user_id, ticker, e, exc_info=True)
        st.warning(f"⚠ Could not remove {ticker} — please retry.")

def _sb_insert_history(user_id: str, record: dict):
    try:
        supabase.table("portfolio_history").insert({
            "user_id": user_id,
            "date":    record.get("date"),
            "type":    record.get("type"),
            "ticker":  record.get("ticker"),
            "shares":  record.get("shares"),
            "price":   record.get("price"),
            "amount":  record.get("amount"),
        }).execute()
    except Exception as e:
        logger.error("_sb_insert_history failed for user '%s', ticker '%s': %s",
                     user_id, record.get("ticker"), e, exc_info=True)

def _sb_update_prices(user_id: str, holdings: list):
    """Batch-update current_price / pl / pl_pct for all holdings in ONE round-trip."""
    if not holdings:
        return
    rows = [
        {
            "user_id":       user_id,
            "ticker":        h["ticker"],
            "current_price": h["current_price"],
            "pl":            h["pl"],
            "pl_pct":        h["pl_pct"],
        }
        for h in holdings
    ]
    try:
        supabase.table("portfolio_holdings") \
            .upsert(rows, on_conflict="user_id,ticker").execute()
        logger.info("_sb_update_prices: batch-updated %d holdings for user '%s'", len(rows), user_id)
    except Exception as e:
        logger.error("_sb_update_prices batch upsert failed for user '%s': %s",
                     user_id, e, exc_info=True)
        st.warning("⚠ Could not refresh portfolio prices — data may be stale.")

# ── Watchlist Supabase helpers ────────────────────────────────────────────────
# Required table (run once in Supabase SQL editor):
#
#   CREATE TABLE IF NOT EXISTS watchlist (
#     id           UUID DEFAULT gen_random_uuid() PRIMARY KEY,
#     user_id      UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
#     stock_symbol TEXT NOT NULL,
#     created_at   TIMESTAMPTZ DEFAULT NOW(),
#     UNIQUE(user_id, stock_symbol)
#   );

FREE_PLAN_WATCHLIST_LIMIT = 5

def _sb_load_watchlist(user_id: str) -> list:
    try:
        res = (supabase.table("watchlist").select("stock_symbol")
               .eq("user_id", user_id).order("created_at").execute())
        return [r["stock_symbol"] for r in (res.data or [])]
    except Exception as e:
        logger.error("_sb_load_watchlist failed for user '%s': %s", user_id, e)
        return []

def _sb_add_watchlist(user_id: str, symbol: str) -> bool:
    try:
        supabase.table("watchlist").insert(
            {"user_id": user_id, "stock_symbol": symbol.upper()}
        ).execute()
        return True
    except Exception as e:
        logger.error("_sb_add_watchlist failed for user '%s', symbol '%s': %s", user_id, symbol, e)
        return False

def _sb_remove_watchlist(user_id: str, symbol: str):
    try:
        supabase.table("watchlist").delete().eq("user_id", user_id).eq("stock_symbol", symbol).execute()
    except Exception as e:
        logger.error("_sb_remove_watchlist failed for user '%s', symbol '%s': %s", user_id, symbol, e)

# ── Usage limit Supabase helpers ──────────────────────────────────────────────
# Required table (run once in Supabase SQL editor):
#
#   CREATE TABLE IF NOT EXISTS user_usage (
#     user_id       UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
#     usage_count   INT NOT NULL DEFAULT 0,
#     last_used_date DATE NOT NULL DEFAULT CURRENT_DATE
#   );

FREE_PLAN_DAILY_LIMIT = 3

def _sb_get_usage(user_id: str) -> dict:
    """Returns {usage_count, last_used_date, plan}. Resets count if date has changed."""
    today = pd.Timestamp.now().date().isoformat()
    try:
        res = supabase.table("user_usage").select("*").eq("user_id", user_id).execute()
        row = (res.data or [None])[0]
        if not row:
            supabase.table("user_usage").insert(
                {"user_id": user_id, "usage_count": 0, "last_used_date": today, "plan": "free"}
            ).execute()
            return {"usage_count": 0, "last_used_date": today, "plan": "free"}
        if row.get("last_used_date") != today:
            supabase.table("user_usage").update(
                {"usage_count": 0, "last_used_date": today}
            ).eq("user_id", user_id).execute()
            row["usage_count"] = 0
            row["last_used_date"] = today
        row.setdefault("plan", "free")
        return row
    except Exception as e:
        logger.error("_sb_get_usage failed for user '%s': %s", user_id, e)
        return {"usage_count": 0, "last_used_date": today, "plan": "free"}

def _sb_increment_usage(user_id: str):
    today = pd.Timestamp.now().date().isoformat()
    try:
        supabase.table("user_usage").upsert(
            {"user_id": user_id, "usage_count": st.session_state.get("usage_count", 0) + 1,
             "last_used_date": today},
            on_conflict="user_id"
        ).execute()
    except Exception as e:
        logger.error("_sb_increment_usage failed for user '%s': %s", user_id, e)

# ── Plan definitions ──────────────────────────────────────────────────────────
# Plan limits — change here to update everywhere in the app
PLAN_LIMITS = {
    "free": {
        "daily_analyses":  3,
        "watchlist_stocks": 5,
        "forecast_horizon": 7,       # max days free users can forecast
        "model_compare":   False,    # Prophet comparison locked
        "conf_interval":   False,    # Bootstrap CI locked
        "multi_stock":     False,    # Multi-ticker comparison locked
        "data_years":      3,        # years of history
    },
    "pro": {
        "daily_analyses":  999,      # effectively unlimited
        "watchlist_stocks": 50,
        "forecast_horizon": 30,
        "model_compare":   True,
        "conf_interval":   True,
        "multi_stock":     True,
        "data_years":      10,
    },
}

def _get_limit(key: str) -> object:
    """Return the limit value for the current user's plan."""
    plan = st.session_state.get("user_plan", "free")
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["free"]).get(key)

def _is_pro() -> bool:
    return st.session_state.get("user_plan", "free") == "pro"

# ── Plan Supabase helpers ─────────────────────────────────────────────────────
# Required column on user_usage table (add if not present):
#   ALTER TABLE user_usage ADD COLUMN IF NOT EXISTS plan TEXT NOT NULL DEFAULT 'free';
#
# Or create fresh:
#   CREATE TABLE IF NOT EXISTS user_usage (
#     user_id       UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
#     usage_count   INT  NOT NULL DEFAULT 0,
#     last_used_date DATE NOT NULL DEFAULT CURRENT_DATE,
#     plan          TEXT NOT NULL DEFAULT 'free'
#   );

def _sb_get_plan(user_id: str) -> str:
    """Returns 'free' or 'pro' for the given user."""
    try:
        res = supabase.table("user_usage").select("plan").eq("user_id", user_id).execute()
        row = (res.data or [None])[0]
        return (row or {}).get("plan", "free")
    except Exception as e:
        logger.error("_sb_get_plan failed for user '%s': %s", user_id, e)
        return "free"

def _sb_set_plan(user_id: str, plan: str):
    """Set user plan to 'free' or 'pro'. Used for manual upgrade / admin."""
    try:
        supabase.table("user_usage").upsert(
            {"user_id": user_id, "plan": plan},
            on_conflict="user_id"
        ).execute()
        st.session_state.user_plan = plan
        logger.info("Plan updated to '%s' for user '%s'", plan, user_id)
    except Exception as e:
        logger.error("_sb_set_plan failed for user '%s': %s", user_id, e)

def _sb_get_email_alerts(user_id: str) -> bool:
    """Returns True if user has email alerts enabled."""
    try:
        res = supabase.table("user_usage").select("email_alerts_enabled") \
                  .eq("user_id", user_id).execute()
        row = (res.data or [None])[0]
        return bool((row or {}).get("email_alerts_enabled", False))
    except Exception as e:
        logger.error("_sb_get_email_alerts failed for user '%s': %s", user_id, e)
        return False

def _sb_set_email_alerts(user_id: str, enabled: bool):
    """Toggle email alerts on or off for the user. Gracefully handles missing column."""
    try:
        supabase.table("user_usage").upsert(
            {"user_id": user_id, "email_alerts_enabled": enabled},
            on_conflict="user_id"
        ).execute()
        st.session_state.email_alerts_enabled = enabled
        logger.info("Email alerts set to %s for user '%s'", enabled, user_id)
        return True
    except Exception as e:
        logger.error("_sb_set_email_alerts failed for user '%s': %s", user_id, e)
        # Still update session state so UI reflects the toggle
        st.session_state.email_alerts_enabled = enabled
        return False


def _send_email(to_address: str, subject: str, html_body: str) -> bool:
    """Send an email via SMTP using credentials from Streamlit secrets.

    Required secrets (add to .streamlit/secrets.toml):
        SMTP_HOST  = "smtp.gmail.com"        # or your provider
        SMTP_PORT  = 587
        SMTP_USER  = "you@yourdomain.com"
        SMTP_PASS  = "your-app-password"
        SMTP_FROM  = "Stockcast <you@yourdomain.com>"   # optional display name
    """
    try:
        smtp_host = st.secrets.get("SMTP_HOST", os.environ.get("SMTP_HOST", ""))
        smtp_port = int(st.secrets.get("SMTP_PORT", os.environ.get("SMTP_PORT", 587)))
        smtp_user = st.secrets.get("SMTP_USER", os.environ.get("SMTP_USER", ""))
        smtp_pass = st.secrets.get("SMTP_PASS", os.environ.get("SMTP_PASS", ""))
        smtp_from = st.secrets.get("SMTP_FROM", os.environ.get("SMTP_FROM", smtp_user))

        if not smtp_host or not smtp_user or not smtp_pass:
            logger.warning("_send_email: SMTP credentials not configured — email skipped.")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = smtp_from
        msg["To"]      = to_address
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_address, msg.as_string())

        logger.info("_send_email: sent '%s' to %s", subject, to_address)
        return True
    except Exception as e:
        logger.error("_send_email failed for %s: %s", to_address, e)
        return False


def _build_digest_html(user_email: str, watchlist: list) -> str:
    """Build the HTML body for the daily watchlist digest email."""
    rows_html = ""
    for sym in watchlist:
        try:
            q = av_get_quote(sym)
            price  = q["price"]
            chg    = q["change_pct"]
            color  = "#00e5b0" if chg >= 0 else "#ff5f5f"
            arrow  = "▲" if chg >= 0 else "▼"
            sign   = "+" if chg >= 0 else ""
            rows_html += (
                f'<tr>'
                f'<td style="padding:10px 14px;font-family:monospace;font-weight:700;color:#4d8eff;">{_html_mod.escape(sym)}</td>'
                f'<td style="padding:10px 14px;font-family:monospace;color:#e4eafd;">${price:,.2f}</td>'
                f'<td style="padding:10px 14px;font-family:monospace;color:{color};font-weight:700;">{arrow} {sign}{chg:.2f}%</td>'
                f'</tr>'
            )
        except Exception:
            rows_html += (
                f'<tr>'
                f'<td style="padding:10px 14px;font-family:monospace;color:#4d8eff;">{_html_mod.escape(sym)}</td>'
                f'<td colspan="2" style="padding:10px 14px;color:#8a8fa0;">Price unavailable</td>'
                f'</tr>'
            )

    date_str = pd.Timestamp.now().strftime("%B %d, %Y")
    return f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0;padding:0;background:#080e1c;font-family:Manrope,sans-serif;">
      <div style="max-width:560px;margin:32px auto;background:#0f1727;border-radius:12px; border:1px solid #252f47;overflow:hidden;">
        <div style="background:linear-gradient(135deg,#0f1727,#141d30);padding:28px 32px; border-bottom:2px solid #4d8eff;">
          <div style="font-family:monospace;font-size:11px;letter-spacing:2px; text-transform:uppercase;color:#4d8eff;margin-bottom:6px;">Stockcast · Daily Digest</div>
          <div style="font-size:22px;font-weight:800;color:#e4eafd;">Good morning ☀</div>
          <div style="font-size:13px;color:#8a8fa0;margin-top:4px;">{date_str} · Your watchlist summary</div>
        </div>
        <div style="padding:24px 32px;">
          <table style="width:100%;border-collapse:collapse;background:#080e1c;
                        border-radius:8px;overflow:hidden;">
            <thead>
              <tr style="background:#141d30;">
                <th style="padding:10px 14px;text-align:left;font-size:11px;letter-spacing:1.5px;
                           text-transform:uppercase;color:#3e4558;">Ticker</th>
                <th style="padding:10px 14px;text-align:left;font-size:11px;letter-spacing:1.5px;
                           text-transform:uppercase;color:#3e4558;">Price</th>
                <th style="padding:10px 14px;text-align:left;font-size:11px;letter-spacing:1.5px;
                           text-transform:uppercase;color:#3e4558;">Change</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
          <div style="margin-top:20px;padding:14px 18px;background:rgba(255,212,38,0.05); border:1px solid rgba(255,212,38,0.2);border-radius:8px; font-size:11px;color:#8a8fa0;line-height:1.6;">
            ⚠ This digest is for educational and research purposes only. Not financial advice.
          </div>
        </div>
        <div style="padding:18px 32px;border-top:1px solid #252f47;text-align:center;">
          <div style="font-size:11px;color:#3e4558;">
            Sent to {_html_mod.escape(user_email)} · 
            <a href="https://muawwizghani-stock-forecast.streamlit.app" 
               style="color:#4d8eff;text-decoration:none;">Open Stockcast</a>
          </div>
        </div>
      </div>
    </body>
    </html>
    """


# ── Multi-language support ─────────────────────────────────────────────────────
LANGUAGES = {
    "English": {
        # Sidebar controls
        "run": "▶  Run Analysis", "ticker": "Ticker", "from": "From", "to": "To",
        "mode": "Mode", "beginner": "🟢 Beginner", "pro": "🔴 Pro",
        "watchlist": "⭐ Watchlist", "alerts": "🔔 Signal Alerts",
        "forecast": "Outlook", "days": "Days", "add": "Add",
        "portfolio": "💼  Portfolio", "markets": "🌍  Markets",
        "back": "← Back to Dashboard",
        "search_label": "🔍 Search Company / Ticker",
        "lookback": "Lookback Window (days)", "horizon": "Outlook Horizon (days)",
        "simple_view": "✓ Simple view active",
        "pro_view": "⚡ Pro view — all parameters unlocked",
        "fast_mode": "⚡ Fast Mode (skip CI + simulator)",
        "xgb_params": "XGBoost Hyperparameters",
        "alert_target": "Price Alert Target ($)",
        "backtesting": "Strategy Simulator", "enable_backtest": "Enable Strategy Simulator",
        "init_capital": "Initial Capital ($)", "commission": "Commission per Trade ($)",
        "signal_thresh": "Signal Threshold (%)", "extra_features": "Extra Features",
        "model_compare": "Model Comparison (XGB vs LR vs Prophet)",
        "halal_check": "Halal / Shariah Compliance Check",
        "conf_interval": "Confidence Intervals on Outlook",
        "bootstrap_samples": "Bootstrap Samples (CI)", "multi_stock": "Multi-Stock Comparison",
        "compare_tickers": "Compare Tickers", "alert_signal_change": "Alert when signal changes",
        # Dashboard
        "dashboard_title": "Dashboard", "dashboard_subtitle": "Overview",
        "dashboard_desc": "Your AI stock research assistant · Enter a ticker in the sidebar and click Run Analysis to begin.",
        "watchlist_live": "⭐ Watchlist — Live Prices",
        "how_it_works": "How It Works",
        "hw1_title": "Enter a Ticker", "hw1_body": "Search by company name or symbol. Add it to your watchlist to track it persistently.",
        "hw2_title": "Run the Assistant", "hw2_body": "Your AI assistant analyses 7 years of market data across 20 technical signals. Results in seconds.",
        "hw3_title": "Read the Signal", "hw3_body": "Get a BUY / SELL / HOLD research signal with a full breakdown of every contributing factor.",
        "platform_features": "Platform Features",
        "no_stocks_saved": "No stocks saved yet.",
        "no_holdings": "No holdings yet. Add your first stock above.",
        "no_transactions": "No transactions yet.",
        # Portfolio
        "portfolio_title": "Portfolio", "portfolio_tracker": "Tracker",
        "portfolio_desc": "Add your real holdings — prices refresh live from Yahoo Finance",
        "add_holding": "➕ Add Holding", "add_to_portfolio": "Add to Portfolio",
        "refresh_prices": "🔄 Refresh Live Prices",
        "total_value": "Total Value", "total_pl": "Total P&L",
        "invested": "Invested", "holdings": "Holdings",
        "holdings_label": "Holdings", "sector_allocation": "Sector Allocation",
        "recent_activity": "Recent Activity",
        # Analysis
        "price_chart": "Price Chart", "tech_indicators": "Technical Indicators",
        "news_sentiment": "News Sentiment NLP", "screening_criteria": "Screening Criteria",
        "analysis_tab": "📊  Analysis", "methodology_tab": "📖  Methodology",
        "dashboard_tab": "🖥  Dashboard",
        "reality_check_title": "💡 How to Use This Assistant",
        "reality_check_body": "Like any research assistant, Stockcast works best when <b style='color:#e4eafd;'>combined with your own judgment</b> and current market context. The assistant analyses <b style='color:#4d8eff;'>price &amp; volume patterns</b> — complement it with: &nbsp;📰 recent news &nbsp;·&nbsp;📊 earnings releases &nbsp;·&nbsp;🏦 macro events &nbsp;·&nbsp;🧠 analyst reports &nbsp;·&nbsp;🌍 sector context. <b style='color:#ffd426;'>Use signals as a research input — never as the sole basis for a trade.</b>",
        "logout": "⏏  Logout",
        "verify_symbol": "Using: {ticker} — verify symbol",
        "active_ticker": "● ACTIVE: {ticker}",
        "greed_territory": "Greed territory", "low_volatility": "Low volatility",
        "no_recent_news": "No recent news found for this ticker.",
        "already_in_portfolio": "{sym} already in portfolio. Remove it first to update.",
        "added_success": "✓ Added {sym} — live price ${price:.2f}",
        "model_performance": "Analysis Quality", "actual_vs_pred": "Actual vs Predicted",
        "feature_importance": "Signal Drivers", "signal_intelligence": "Signal Intelligence",
        "forecast_next": "Price Outlook — Next {n} Days", "backtest_engine": "Strategy Simulator",
        "trade_log": "Trade Log", "forecast_ci": "Outlook with Confidence Intervals",
        "model_comparison": "Model Comparison — XGBoost vs Prophet vs Linear Regression",
        "sector_heatmap": "Sector Heat Map · Live", "fear_greed": "Fear & Greed Index · Live",
        "days_loaded": "✓ {n} trading days loaded for {ticker}", "fetching": "Loading {ticker} data...",
        "engineering": "Computing technical signals...", "building": "Building signal matrix...",
        "training": "Training AI model (cached after first run)...",
        "running_backtest": "Running strategy simulation...", "running_bootstrap": "Running {n} bootstrap samples...",
        "training_lr": "Training Linear Regression...", "fetching_market": "Loading live market data...",
        "fetching_financial": "Fetching financial data for {ticker}...",
        "not_enough_data": "Not enough data to analyse. Try a longer date range or smaller lookback window.",
        "enter_ticker": "Please enter a ticker symbol.", "max_holdings": "Maximum {n} holdings reached.",
        "loading_prices": "Fetching live price for {sym}...",
        "deep_analysis": "📈  Deep Analysis",
        "model_conf_score": "ASSISTANT CONFIDENCE SCORE",
        "high_confidence": "HIGH CONFIDENCE", "moderate_confidence": "MODERATE CONFIDENCE", "low_confidence": "LOW CONFIDENCE",
        "r2_fit": "R² fit", "mape_accuracy": "MAPE accuracy", "directional_acc": "Directional acc.", "data_volume": "Data volume",
        "composite_signal": "Research Signal", "forecast_lbl": "outlook",
        "score_lbl": "Score",
        "take_profit_lbl": "Take Profit", "stop_loss_lbl": "Stop Loss", "risk_reward_lbl": "Risk / Reward",
        "rsi_lbl": "RSI (14)",
        "favorable": "✓ Favorable", "marginal": "⚠ Marginal", "unfavorable": "✗ Unfavorable",
        "oversold_zone": "Oversold zone", "overbought_zone": "Overbought zone", "neutral_zone": "Neutral zone",
        "factor_breakdown": "6-Factor Signal Breakdown",
        "last_close_lbl": "Last Close", "model_confidence_lbl": "Assistant Confidence",
        "high_lbl": "High", "moderate_lbl": "Moderate", "low_lbl": "Low",
        "at_above_target": "AT or ABOVE your target of",
        "below_target": "below target of",
        "shariah_debt_mktcap": "Debt/MarketCap",
        "shariah_debt_assets": "Debt/Assets",
        "shariah_cash_assets": "Cash/Assets",
        "known_noncompliant": "Known non-compliant ticker",
    },
    "Arabic": {
        "run": "▶  تشغيل التحليل", "ticker": "الرمز", "from": "من", "to": "إلى",
        "mode": "الوضع", "beginner": "🟢 مبتدئ", "pro": "🔴 محترف",
        "watchlist": "⭐ قائمة المراقبة", "alerts": "🔔 تنبيهات الإشارة",
        "forecast": "التوقعات", "days": "أيام", "add": "إضافة",
        "portfolio": "💼  المحفظة", "markets": "🌍  الأسواق",
        "back": "← العودة إلى لوحة التحكم",
        "search_label": "🔍 البحث عن شركة / رمز",
        "lookback": "نافذة الاسترجاع (أيام)", "horizon": "أفق التنبؤ (أيام)",
        "simple_view": "✓ العرض البسيط مفعّل",
        "pro_view": "⚡ وضع الاحتراف — جميع الخيارات مفتوحة",
        "fast_mode": "⚡ الوضع السريع (بدون CI والاختبار)",
        "xgb_params": "معلمات XGBoost",
        "alert_target": "سعر التنبيه المستهدف ($)",
        "backtesting": "محاكاة الاستراتيجية", "enable_backtest": "تفعيل محاكي الاستراتيجية",
        "init_capital": "رأس المال الابتدائي ($)", "commission": "عمولة كل صفقة ($)",
        "signal_thresh": "حد الإشارة (%)", "extra_features": "ميزات إضافية",
        "model_compare": "مقارنة النماذج (XGB vs LR vs Prophet)",
        "halal_check": "فحص الامتثال للشريعة / الحلال",
        "conf_interval": "فترات الثقة في التنبؤ",
        "bootstrap_samples": "عينات Bootstrap (CI)", "multi_stock": "مقارنة متعددة الأسهم",
        "compare_tickers": "مقارنة الرموز", "alert_signal_change": "تنبيه عند تغيير الإشارة",
        "dashboard_title": "لوحة التحكم", "dashboard_subtitle": "نظرة عامة",
        "dashboard_desc": "ذكاء الأسهم بالذكاء الاصطناعي · أدخل رمزًا في الشريط الجانبي وانقر تشغيل للبدء.",
        "watchlist_live": "⭐ قائمة المراقبة — الأسعار الحية",
        "how_it_works": "كيف يعمل",
        "hw1_title": "أدخل الرمز", "hw1_body": "ابحث باسم الشركة أو الرمز. أضفه إلى قائمة المراقبة للمتابعة.",
        "hw2_title": "تشغيل النموذج", "hw2_body": "يتدرب XGBoost على 7 سنوات من البيانات بـ 20 ميزة مهندسة. النتائج في ثوانٍ.",
        "hw3_title": "اقرأ الإشارة", "hw3_body": "احصل على حكم شراء / بيع / انتظار مع شرح كامل لكل عامل.",
        "platform_features": "ميزات المنصة",
        "no_stocks_saved": "لا أسهم محفوظة بعد.",
        "no_holdings": "لا حيازات بعد. أضف أول سهم أعلاه.",
        "no_transactions": "لا معاملات بعد.",
        "portfolio_title": "المحفظة", "portfolio_tracker": "المتتبع",
        "portfolio_desc": "أضف حيازاتك الحقيقية — تتحدث الأسعار مباشرةً من Yahoo Finance",
        "add_holding": "➕ إضافة حيازة", "add_to_portfolio": "أضف إلى المحفظة",
        "refresh_prices": "🔄 تحديث الأسعار الحية",
        "total_value": "القيمة الإجمالية", "total_pl": "الربح/الخسارة الإجمالية",
        "invested": "المستثمر", "holdings": "الحيازات",
        "holdings_label": "الحيازات", "sector_allocation": "توزيع القطاعات",
        "recent_activity": "النشاط الأخير",
        "price_chart": "مخطط السعر", "tech_indicators": "المؤشرات الفنية",
        "news_sentiment": "تحليل مشاعر الأخبار", "screening_criteria": "معايير الفحص",
        "analysis_tab": "📊  التحليل", "methodology_tab": "📖  المنهجية",
        "dashboard_tab": "🖥  لوحة التحكم",
        "reality_check_title": "💡 كيفية استخدام هذا المساعد",
        "reality_check_body": "مثل أي مساعد بحثي، يعمل Stockcast بشكل أفضل عند <b style='color:#e4eafd;'>دمجه مع حكمك الخاص</b> والسياق الحالي للسوق. يحلل المساعد <b style='color:#4d8eff;'>أنماط السعر والحجم</b> — أضف إليها: 📰 الأخبار الحديثة · 📊 إصدارات الأرباح · 🏦 الأحداث الاقتصادية · 🧠 تقارير المحللين · 🌍 السياق القطاعي. <b style='color:#ffd426;'>استخدم الإشارات كمدخل بحثي — وليس كأساس وحيد للتداول.</b>",
        "logout": "⏏  تسجيل الخروج",
        "verify_symbol": "استخدام: {ticker} — تحقق من الرمز",
        "active_ticker": "● نشط: {ticker}",
        "greed_territory": "منطقة الجشع", "low_volatility": "تذبذب منخفض",
        "no_recent_news": "لا توجد أخبار حديثة لهذا الرمز.",
        "already_in_portfolio": "{sym} موجود بالفعل في المحفظة. أزله أولاً للتحديث.",
        "added_success": "✓ تمت إضافة {sym} — السعر الحي ${price:.2f}",
        "footer": "⚠ STOCKCAST · للأغراض التعليمية فقط · ليست نصيحة مالية · طوّره معاوية غني",
        "model_performance": "أداء النموذج", "actual_vs_pred": "الفعلي مقابل المتوقع",
        "feature_importance": "أهمية الميزات", "signal_intelligence": "ذكاء الإشارات",
        "forecast_next": "توقعات السعر — الـ {n} أيام القادمة", "backtest_engine": "محاكي الاستراتيجية",
        "trade_log": "سجل التداول", "forecast_ci": "التنبؤ مع فترات الثقة",
        "model_comparison": "مقارنة النماذج — XGBoost مقابل Prophet مقابل الانحدار الخطي",
        "sector_heatmap": "خريطة حرارة القطاعات · مباشر", "fear_greed": "مؤشر الخوف والجشع · مباشر",
        "days_loaded": "✓ {n} يوم تداول تم تحميله لـ {ticker}", "fetching": "جارٍ جلب بيانات {ticker}...",
        "engineering": "جارٍ هندسة الميزات التقنية...", "building": "جارٍ بناء مصفوفة الميزات...",
        "training": "جارٍ تدريب نموذج XGBoost...", "running_backtest": "جارٍ تشغيل محاكاة الاختبار الخلفي...",
        "running_bootstrap": "جارٍ تشغيل {n} عينة Bootstrap...", "training_lr": "جارٍ تدريب الانحدار الخطي...",
        "fetching_market": "جارٍ تحميل بيانات السوق المباشرة...", "fetching_financial": "جارٍ جلب البيانات المالية لـ {ticker}...",
        "not_enough_data": "بيانات غير كافية للتدريب. جرّب نطاقاً زمنياً أطول أو نافذة استرجاع أصغر.",
        "enter_ticker": "الرجاء إدخال رمز السهم.", "max_holdings": "تم الوصول إلى الحد الأقصى {n} حيازة.",
        "loading_prices": "جارٍ جلب السعر المباشر لـ {sym}...",
        "deep_analysis": "📈  تحليل عميق",
        "model_conf_score": "نقاط ثقة المساعد",
        "high_confidence": "ثقة عالية", "moderate_confidence": "ثقة متوسطة", "low_confidence": "ثقة منخفضة",
        "r2_fit": "دقة R²", "mape_accuracy": "دقة MAPE", "directional_acc": "الدقة الاتجاهية", "data_volume": "حجم البيانات",
        "composite_signal": "إشارة البحث", "forecast_lbl": "توقع",
        "score_lbl": "النقاط",
        "take_profit_lbl": "جني الأرباح", "stop_loss_lbl": "وقف الخسارة", "risk_reward_lbl": "المخاطرة / العائد",
        "rsi_lbl": "RSI (14)",
        "favorable": "✓ مناسب", "marginal": "⚠ هامشي", "unfavorable": "✗ غير مناسب",
        "oversold_zone": "منطقة البيع الزائد", "overbought_zone": "منطقة الشراء الزائد", "neutral_zone": "المنطقة المحايدة",
        "factor_breakdown": "تحليل 6 عوامل للإشارة",
        "last_close_lbl": "آخر إغلاق", "model_confidence_lbl": "ثقة المساعد",
        "high_lbl": "عالية", "moderate_lbl": "متوسطة", "low_lbl": "منخفضة",
        "at_above_target": "عند الهدف أو أعلاه",
        "below_target": "تحت الهدف بـ",
        "shariah_debt_mktcap": "الدين / القيمة السوقية",
        "shariah_debt_assets": "الدين / الأصول",
        "shariah_cash_assets": "النقد / الأصول",
        "known_noncompliant": "رمز غير متوافق معروف",
    },
    "Urdu": {
        "run": "▶  تجزیہ چلائیں", "ticker": "ٹکر", "from": "سے", "to": "تک",
        "mode": "موڈ", "beginner": "🟢 ابتدائی", "pro": "🔴 پرو",
        "watchlist": "⭐ واچ لسٹ", "alerts": "🔔 سگنل الرٹس",
        "forecast": "پیشن گوئی", "days": "دن", "add": "شامل کریں",
        "portfolio": "💼  پورٹ فولیو", "markets": "🌍  مارکیٹس",
        "back": "← ڈیش بورڈ پر واپس",
        "search_label": "🔍 کمپنی / ٹکر تلاش کریں",
        "lookback": "لُک بیک ونڈو (دن)", "horizon": "پیشن گوئی کا دورانیہ (دن)",
        "simple_view": "✓ سادہ منظر فعال ہے",
        "pro_view": "⚡ پرو منظر — تمام اختیارات کھلے ہیں",
        "fast_mode": "⚡ فاسٹ موڈ (CI اور بیک ٹیسٹ چھوڑیں)",
        "xgb_params": "XGBoost پیرامیٹرز",
        "alert_target": "قیمت الرٹ ہدف ($)",
        "backtesting": "اسٹریٹجی سمیولیٹر", "enable_backtest": "اسٹریٹجی سمیولیٹر فعال کریں",
        "init_capital": "ابتدائی سرمایہ ($)", "commission": "فی ٹریڈ کمیشن ($)",
        "signal_thresh": "سگنل حد (%)", "extra_features": "اضافی خصوصیات",
        "model_compare": "ماڈل موازنہ (XGB vs LR vs Prophet)",
        "halal_check": "حلال / شریعت کی تعمیل کی جانچ",
        "conf_interval": "پیشن گوئی پر اعتماد کے وقفے",
        "bootstrap_samples": "Bootstrap سیمپل (CI)", "multi_stock": "کثیر اسٹاک موازنہ",
        "compare_tickers": "ٹکر موازنہ کریں", "alert_signal_change": "سگنل بدلنے پر الرٹ",
        "dashboard_title": "ڈیش بورڈ", "dashboard_subtitle": "جائزہ",
        "dashboard_desc": "AI سے چلنے والی اسٹاک انٹیلی جنس · سائڈبار میں ٹکر درج کریں اور پیشن گوئی چلائیں۔",
        "watchlist_live": "⭐ واچ لسٹ — لائیو قیمتیں",
        "how_it_works": "یہ کیسے کام کرتا ہے",
        "hw1_title": "ٹکر درج کریں", "hw1_body": "کمپنی کے نام یا علامت سے تلاش کریں۔ اسے واچ لسٹ میں شامل کریں۔",
        "hw2_title": "ماڈل چلائیں", "hw2_body": "XGBoost 7 سال کے ڈیٹا پر 20 انجینئرڈ فیچرز کے ساتھ تربیت کرتا ہے۔ نتائج سیکنڈوں میں۔",
        "hw3_title": "سگنل پڑھیں", "hw3_body": "خرید / فروخت / انتظار کا فیصلہ حاصل کریں ہر عنصر کی مکمل وضاحت کے ساتھ۔",
        "platform_features": "پلیٹ فارم کی خصوصیات",
        "no_stocks_saved": "ابھی تک کوئی اسٹاک محفوظ نہیں۔",
        "no_holdings": "ابھی تک کوئی ہولڈنگ نہیں۔ اوپر پہلا اسٹاک شامل کریں۔",
        "no_transactions": "ابھی تک کوئی لین دین نہیں۔",
        "portfolio_title": "پورٹ فولیو", "portfolio_tracker": "ٹریکر",
        "portfolio_desc": "اپنی اصل ہولڈنگز شامل کریں — قیمتیں Yahoo Finance سے لائیو اپ ڈیٹ ہوتی ہیں",
        "add_holding": "➕ ہولڈنگ شامل کریں", "add_to_portfolio": "پورٹ فولیو میں شامل کریں",
        "refresh_prices": "🔄 لائیو قیمتیں تازہ کریں",
        "total_value": "کل قیمت", "total_pl": "کل نفع/نقصان",
        "invested": "سرمایہ کاری", "holdings": "ہولڈنگز",
        "holdings_label": "ہولڈنگز", "sector_allocation": "شعبہ وار تقسیم",
        "recent_activity": "حالیہ سرگرمی",
        "price_chart": "قیمت کا چارٹ", "tech_indicators": "تکنیکی اشارے",
        "news_sentiment": "خبر جذباتی تجزیہ", "screening_criteria": "اسکریننگ کا معیار",
        "analysis_tab": "📊  تجزیہ", "methodology_tab": "📖  طریقہ کار",
        "dashboard_tab": "🖥  ڈیش بورڈ",
        "reality_check_title": "💡 اس مساعد کو کیسے استعمال کریں",
        "reality_check_body": "کسی بھی تحقیقی معاون کی طرح، Stockcast اس وقت بہترین کام کرتا ہے جب <b style='color:#e4eafd;'>آپ کے اپنے فیصلے</b> اور موجودہ مارکیٹ کے سیاق کے ساتھ استعمال ہو۔ مساعد <b style='color:#4d8eff;'>قیمت اور حجم کے نمونوں</b> کا تجزیہ کرتا ہے — اسے شامل کریں: 📰 حالیہ خبریں · 📊 آمدنی کے اعلانات · 🏦 معاشی واقعات · 🧠 تجزیہ کاروں کی رپورٹس · 🌍 شعبے کا سیاق. <b style='color:#ffd426;'>سگنلز کو تحقیقی ان پٹ کے طور پر استعمال کریں — تجارت کی واحد بنیاد نہیں۔</b>",
        "logout": "⏏  لاگ آؤٹ",
        "verify_symbol": "استعمال: {ticker} — علامت کی تصدیق کریں",
        "active_ticker": "● فعال: {ticker}",
        "greed_territory": "لالچ کا علاقہ", "low_volatility": "کم اتار چڑھاؤ",
        "no_recent_news": "اس ٹکر کے لیے کوئی حالیہ خبر نہیں ملی۔",
        "already_in_portfolio": "{sym} پہلے سے پورٹ فولیو میں ہے۔ اپڈیٹ کرنے کے لیے پہلے ہٹائیں۔",
        "added_success": "✓ {sym} شامل کیا — لائیو قیمت ${price:.2f}",
        "footer": "⚠ STOCKCAST · صرف تعلیمی مقاصد کے لیے · مالی مشورہ نہیں · تیار کردہ معاویہ غنی",
        "model_performance": "ماڈل کی کارکردگی", "actual_vs_pred": "حقیقی بمقابلہ پیشین گوئی",
        "feature_importance": "فیچرز کی اہمیت", "signal_intelligence": "سگنل انٹیلی جنس",
        "forecast_next": "قیمت کا رجحان — اگلے {n} دن", "backtest_engine": "اسٹریٹجی سمیولیٹر",
        "trade_log": "ٹریڈ لاگ", "forecast_ci": "اعتماد کے وقفوں کے ساتھ پیشن گوئی",
        "model_comparison": "ماڈل موازنہ — XGBoost بمقابلہ Prophet بمقابلہ لکیری رجعت",
        "sector_heatmap": "سیکٹر ہیٹ میپ · لائیو", "fear_greed": "خوف اور لالچ انڈیکس · لائیو",
        "days_loaded": "✓ {n} ٹریڈنگ دن {ticker} کے لیے لوڈ ہوئے", "fetching": "{ticker} کا ڈیٹا لا رہے ہیں...",
        "engineering": "تکنیکی فیچرز بنا رہے ہیں...", "building": "فیچر میٹرکس بنا رہے ہیں...",
        "training": "XGBoost ماڈل ٹرین ہو رہا ہے...", "running_backtest": "بیک ٹیسٹ سمولیشن چل رہی ہے...",
        "running_bootstrap": "{n} Bootstrap سیمپل چل رہے ہیں...", "training_lr": "لکیری رجعت ٹرین ہو رہی ہے...",
        "fetching_market": "مارکیٹ کا لائیو ڈیٹا لوڈ ہو رہا ہے...", "fetching_financial": "{ticker} کا مالی ڈیٹا لا رہے ہیں...",
        "not_enough_data": "تربیت کے لیے کافی ڈیٹا نہیں۔ طویل تاریخی رینج یا چھوٹا لُک بیک آزمائیں۔",
        "enter_ticker": "براہ کرم ٹکر علامت درج کریں۔", "max_holdings": "زیادہ سے زیادہ {n} ہولڈنگز پہنچ گئی۔",
        "loading_prices": "{sym} کی لائیو قیمت لا رہے ہیں...",
        "deep_analysis": "📈  گہرا تجزیہ",
        "model_conf_score": "مساعد اعتماد اسکور",
        "high_confidence": "اعلی اعتماد", "moderate_confidence": "اوسط اعتماد", "low_confidence": "کم اعتماد",
        "r2_fit": "R² فٹ", "mape_accuracy": "MAPE درستگی", "directional_acc": "سمتی درستگی", "data_volume": "ڈیٹا حجم",
        "composite_signal": "تحقیقی سگنل", "forecast_lbl": "رجحان",
        "score_lbl": "اسکور",
        "take_profit_lbl": "منافع لیں", "stop_loss_lbl": "نقصان روکیں", "risk_reward_lbl": "خطرہ / انعام",
        "rsi_lbl": "RSI (14)",
        "favorable": "✓ موزوں", "marginal": "⚠ معمولی", "unfavorable": "✗ ناموزوں",
        "oversold_zone": "زیادہ فروخت زون", "overbought_zone": "زیادہ خریداری زون", "neutral_zone": "غیر جانبدار زون",
        "factor_breakdown": "6 عوامل سگنل تجزیہ",
        "last_close_lbl": "آخری بندش", "model_confidence_lbl": "مساعد اعتماد",
        "high_lbl": "اعلی", "moderate_lbl": "اوسط", "low_lbl": "کم",
        "at_above_target": "ہدف پر یا اوپر",
        "below_target": "ہدف سے نیچے",
        "shariah_debt_mktcap": "قرض / مارکیٹ کیپ",
        "shariah_debt_assets": "قرض / اثاثے",
        "shariah_cash_assets": "نقد / اثاثے",
        "known_noncompliant": "معروف غیر موافق ٹکر",
    },
    "Hindi": {
        "run": "▶  विश्लेषण चलाएं", "ticker": "टिकर", "from": "से", "to": "तक",
        "mode": "मोड", "beginner": "🟢 शुरुआती", "pro": "🔴 प्रो",
        "watchlist": "⭐ वॉचलिस्ट", "alerts": "🔔 सिग्नल अलर्ट",
        "forecast": "पूर्वानुमान", "days": "दिन", "add": "जोड़ें",
        "portfolio": "💼  पोर्टफोलियो", "markets": "🌍  बाज़ार",
        "back": "← डैशबोर्ड पर वापस",
        "search_label": "🔍 कंपनी / टिकर खोजें",
        "lookback": "लुकबैक विंडो (दिन)", "horizon": "पूर्वानुमान अवधि (दिन)",
        "simple_view": "✓ सरल दृश्य सक्रिय",
        "pro_view": "⚡ प्रो दृश्य — सभी पैरामीटर अनलॉक",
        "fast_mode": "⚡ फास्ट मोड (CI + बैकटेस्ट छोड़ें)",
        "xgb_params": "XGBoost हाइपरपैरामीटर",
        "alert_target": "मूल्य अलर्ट लक्ष्य ($)",
        "backtesting": "स्ट्रैटेजी सिम्युलेटर", "enable_backtest": "स्ट्रैटेजी सिम्युलेटर सक्षम करें",
        "init_capital": "प्रारंभिक पूंजी ($)", "commission": "प्रति ट्रेड कमीशन ($)",
        "signal_thresh": "सिग्नल सीमा (%)", "extra_features": "अतिरिक्त सुविधाएं",
        "model_compare": "मॉडल तुलना (XGB vs LR vs Prophet)",
        "halal_check": "हलाल / शरिया अनुपालन जांच",
        "conf_interval": "पूर्वानुमान पर विश्वास अंतराल",
        "bootstrap_samples": "Bootstrap नमूने (CI)", "multi_stock": "बहु-स्टॉक तुलना",
        "compare_tickers": "टिकर तुलना करें", "alert_signal_change": "सिग्नल बदलने पर अलर्ट",
        "dashboard_title": "डैशबोर्ड", "dashboard_subtitle": "अवलोकन",
        "dashboard_desc": "AI-संचालित स्टॉक इंटेलिजेंस · साइडबार में टिकर दर्ज करें और पूर्वानुमान चलाएं।",
        "watchlist_live": "⭐ वॉचलिस्ट — लाइव कीमतें",
        "how_it_works": "यह कैसे काम करता है",
        "hw1_title": "टिकर दर्ज करें", "hw1_body": "कंपनी नाम या प्रतीक से खोजें। लगातार ट्रैक करने के लिए वॉचलिस्ट में जोड़ें।",
        "hw2_title": "मॉडल चलाएं", "hw2_body": "XGBoost 7 साल के OHLCV डेटा पर 20 इंजीनियर्ड फीचर्स के साथ प्रशिक्षण लेता है। सेकंडों में परिणाम।",
        "hw3_title": "सिग्नल पढ़ें", "hw3_body": "हर योगदान कारक की पूरी व्याख्या के साथ BUY/SELL/HOLD निर्णय प्राप्त करें।",
        "platform_features": "प्लेटफॉर्म की विशेषताएं",
        "no_stocks_saved": "अभी तक कोई स्टॉक सहेजा नहीं गया।",
        "no_holdings": "अभी तक कोई होल्डिंग नहीं। ऊपर पहला स्टॉक जोड़ें।",
        "no_transactions": "अभी तक कोई लेनदेन नहीं।",
        "portfolio_title": "पोर्टफोलियो", "portfolio_tracker": "ट्रैकर",
        "portfolio_desc": "अपनी वास्तविक होल्डिंग्स जोड़ें — Yahoo Finance से कीमतें लाइव अपडेट होती हैं",
        "add_holding": "➕ होल्डिंग जोड़ें", "add_to_portfolio": "पोर्टफोलियो में जोड़ें",
        "refresh_prices": "🔄 लाइव कीमतें ताज़ा करें",
        "total_value": "कुल मूल्य", "total_pl": "कुल लाभ/हानि",
        "invested": "निवेशित", "holdings": "होल्डिंग्स",
        "holdings_label": "होल्डिंग्स", "sector_allocation": "क्षेत्र आवंटन",
        "recent_activity": "हालिया गतिविधि",
        "price_chart": "मूल्य चार्ट", "tech_indicators": "तकनीकी संकेतक",
        "news_sentiment": "समाचार भावना NLP", "screening_criteria": "स्क्रीनिंग मानदंड",
        "analysis_tab": "📊  विश्लेषण", "methodology_tab": "📖  कार्यप्रणाली",
        "dashboard_tab": "🖥  डैशबोर्ड",
        "reality_check_title": "💡 इस असिस्टेंट का उपयोग कैसे करें",
        "reality_check_body": "किसी भी शोध सहायक की तरह, Stockcast तब सबसे अच्छा काम करता है जब <b style='color:#e4eafd;'>आपके अपने निर्णय</b> और वर्तमान बाज़ार संदर्भ के साथ मिलाया जाए। असिस्टेंट <b style='color:#4d8eff;'>मूल्य और वॉल्यूम पैटर्न</b> का विश्लेषण करता है — इसे पूरक बनाएं: 📰 हालिया खबरें · 📊 आय रिलीज़ · 🏦 मैक्रो इवेंट · 🧠 विश्लेषक रिपोर्ट · 🌍 सेक्टर संदर्भ। <b style='color:#ffd426;'>सिग्नल को शोध इनपुट के रूप में उपयोग करें — ट्रेड का एकमात्र आधार नहीं।</b>",
        "logout": "⏏  लॉगआउट",
        "verify_symbol": "उपयोग: {ticker} — प्रतीक सत्यापित करें",
        "active_ticker": "● सक्रिय: {ticker}",
        "greed_territory": "लालच क्षेत्र", "low_volatility": "कम अस्थिरता",
        "no_recent_news": "इस टिकर के लिए कोई हालिया खबर नहीं मिली।",
        "already_in_portfolio": "{sym} पहले से पोर्टफोलियो में है। अपडेट करने के लिए पहले हटाएं।",
        "added_success": "✓ {sym} जोड़ा गया — लाइव कीमत ${price:.2f}",
        "footer": "⚠ STOCKCAST · केवल शैक्षणिक उद्देश्यों के लिए · वित्तीय सलाह नहीं · निर्मित मुआवविज़ घनी द्वारा",
        "model_performance": "मॉडल प्रदर्शन", "actual_vs_pred": "वास्तविक बनाम पूर्वानुमान",
        "feature_importance": "फीचर महत्व", "signal_intelligence": "सिग्नल इंटेलिजेंस",
        "forecast_next": "मूल्य रुझान — अगले {n} दिन", "backtest_engine": "स्ट्रैटेजी सिम्युलेटर",
        "trade_log": "ट्रेड लॉग", "forecast_ci": "विश्वास अंतराल के साथ पूर्वानुमान",
        "model_comparison": "मॉडल तुलना — XGBoost बनाम Prophet बनाम रैखिक प्रतिगमन",
        "sector_heatmap": "सेक्टर हीट मैप · लाइव", "fear_greed": "भय और लालच सूचकांक · लाइव",
        "days_loaded": "✓ {ticker} के लिए {n} ट्रेडिंग दिन लोड हुए", "fetching": "{ticker} का डेटा लाया जा रहा है...",
        "engineering": "तकनीकी फीचर बनाए जा रहे हैं...", "building": "फीचर मैट्रिक्स बनाया जा रहा है...",
        "training": "XGBoost मॉडल प्रशिक्षित हो रहा है...", "running_backtest": "बैकटेस्ट सिमुलेशन चल रही है...",
        "running_bootstrap": "{n} Bootstrap सैंपल चल रहे हैं...", "training_lr": "रैखिक प्रतिगमन प्रशिक्षित हो रहा है...",
        "fetching_market": "लाइव बाज़ार डेटा लोड हो रहा है...", "fetching_financial": "{ticker} का वित्तीय डेटा लाया जा रहा है...",
        "not_enough_data": "प्रशिक्षण के लिए पर्याप्त डेटा नहीं। लंबी तिथि सीमा या छोटा लुकबैक आज़माएं।",
        "enter_ticker": "कृपया टिकर प्रतीक दर्ज करें।", "max_holdings": "अधिकतम {n} होल्डिंग्स पहुंच गई।",
        "loading_prices": "{sym} की लाइव कीमत लाई जा रही है...",
        "deep_analysis": "📈  गहरा विश्लेषण",
        "model_conf_score": "असिस्टेंट विश्वास स्कोर",
        "high_confidence": "उच्च विश्वास", "moderate_confidence": "मध्यम विश्वास", "low_confidence": "कम विश्वास",
        "r2_fit": "R² फिट", "mape_accuracy": "MAPE सटीकता", "directional_acc": "दिशात्मक सटीकता", "data_volume": "डेटा मात्रा",
        "composite_signal": "शोध संकेत", "forecast_lbl": "रुझान",
        "score_lbl": "स्कोर",
        "take_profit_lbl": "लाभ लें", "stop_loss_lbl": "नुकसान रोकें", "risk_reward_lbl": "जोखिम / पुरस्कार",
        "rsi_lbl": "RSI (14)",
        "favorable": "✓ अनुकूल", "marginal": "⚠ सीमांत", "unfavorable": "✗ प्रतिकूल",
        "oversold_zone": "ओवरसोल्ड ज़ोन", "overbought_zone": "ओवरबॉट ज़ोन", "neutral_zone": "तटस्थ क्षेत्र",
        "factor_breakdown": "6-कारक सिग्नल विश्लेषण",
        "last_close_lbl": "अंतिम बंद", "model_confidence_lbl": "असिस्टेंट विश्वास",
        "high_lbl": "उच्च", "moderate_lbl": "मध्यम", "low_lbl": "कम",
        "at_above_target": "लक्ष्य पर या ऊपर",
        "below_target": "लक्ष्य से नीचे",
        "shariah_debt_mktcap": "ऋण / बाज़ार पूंजी",
        "shariah_debt_assets": "ऋण / संपत्ति",
        "shariah_cash_assets": "नकद / संपत्ति",
        "known_noncompliant": "ज्ञात गैर-अनुपालन टिकर",
    },
    "Chinese": {
        "run": "▶  运行分析", "ticker": "股票代码", "from": "从", "to": "到",
        "mode": "模式", "beginner": "🟢 新手", "pro": "🔴 专业",
        "watchlist": "⭐ 关注列表", "alerts": "🔔 信号提醒",
        "forecast": "预测", "days": "天", "add": "添加",
        "portfolio": "💼  投资组合", "markets": "🌍  市场",
        "back": "← 返回仪表板",
        "search_label": "🔍 搜索公司 / 代码",
        "lookback": "回溯窗口（天）", "horizon": "预测周期（天）",
        "simple_view": "✓ 简单视图已激活",
        "pro_view": "⚡ 专业视图 — 所有参数已解锁",
        "fast_mode": "⚡ 快速模式（跳过CI和回测）",
        "xgb_params": "XGBoost 超参数",
        "alert_target": "价格提醒目标 ($)",
        "backtesting": "策略模拟器", "enable_backtest": "启用策略模拟器",
        "init_capital": "初始资金 ($)", "commission": "每笔交易佣金 ($)",
        "signal_thresh": "信号阈值 (%)", "extra_features": "额外功能",
        "model_compare": "模型对比 (XGB vs LR vs Prophet)",
        "halal_check": "清真 / 伊斯兰教法合规检查",
        "conf_interval": "预测置信区间",
        "bootstrap_samples": "Bootstrap 样本 (CI)", "multi_stock": "多股票对比",
        "compare_tickers": "对比股票代码", "alert_signal_change": "信号变化时提醒",
        "dashboard_title": "仪表板", "dashboard_subtitle": "概览",
        "dashboard_desc": "AI 驱动的股票分析 · 在侧边栏输入代码并点击运行预测。",
        "watchlist_live": "⭐ 关注列表 — 实时价格",
        "how_it_works": "运作方式",
        "hw1_title": "输入代码", "hw1_body": "按公司名称或代码搜索，添加到关注列表持续跟踪。",
        "hw2_title": "运行模型", "hw2_body": "XGBoost 对 7 年 OHLCV 数据进行 20 个工程特征训练，秒级出结果。",
        "hw3_title": "读取信号", "hw3_body": "获得买入/卖出/持有判断及每个贡献因子的完整说明。",
        "platform_features": "平台功能",
        "no_stocks_saved": "尚未保存任何股票。",
        "no_holdings": "尚无持仓。在上方添加第一只股票。",
        "no_transactions": "尚无交易记录。",
        "portfolio_title": "投资组合", "portfolio_tracker": "跟踪器",
        "portfolio_desc": "添加您的真实持仓 — 价格实时从 Yahoo Finance 更新",
        "add_holding": "➕ 添加持仓", "add_to_portfolio": "添加到投资组合",
        "refresh_prices": "🔄 刷新实时价格",
        "total_value": "总价值", "total_pl": "总盈亏",
        "invested": "已投资", "holdings": "持仓",
        "holdings_label": "持仓", "sector_allocation": "板块分配",
        "recent_activity": "最近活动",
        "price_chart": "价格图表", "tech_indicators": "技术指标",
        "news_sentiment": "新闻情绪 NLP", "screening_criteria": "筛选标准",
        "analysis_tab": "📊  分析", "methodology_tab": "📖  方法论",
        "dashboard_tab": "🖥  仪表板",
        "reality_check_title": "💡 如何使用此智能助手",
        "reality_check_body": "像任何研究助手一样，Stockcast 与<b style='color:#e4eafd;'>您自己的判断</b>和当前市场背景结合使用时效果最佳。助手分析<b style='color:#4d8eff;'>价格与成交量模式</b>——请结合参考：📰 近期新闻 · 📊 财报发布 · 🏦 宏观事件 · 🧠 分析师报告 · 🌍 行业背景。<b style='color:#ffd426;'>将信号作为研究参考——而非交易的唯一依据。</b>",
        "logout": "⏏  退出登录",
        "verify_symbol": "使用中: {ticker} — 请验证代码",
        "active_ticker": "● 当前: {ticker}",
        "greed_territory": "贪婪区域", "low_volatility": "低波动",
        "no_recent_news": "未找到该股票的近期新闻。",
        "already_in_portfolio": "{sym} 已在投资组合中。如需更新请先删除。",
        "added_success": "✓ 已添加 {sym} — 实时价格 ${price:.2f}",
        "footer": "⚠ STOCKCAST · 仅供教育目的 · 非财务建议 · 由 MUAWWIZ GHANI 开发",
        "model_performance": "模型表现", "actual_vs_pred": "实际 vs 预测",
        "feature_importance": "特征重要性", "signal_intelligence": "信号分析",
        "forecast_next": "价格走势 — 未来 {n} 天", "backtest_engine": "策略模拟器",
        "trade_log": "交易记录", "forecast_ci": "带置信区间的预测",
        "model_comparison": "模型对比 — XGBoost vs Prophet vs 线性回归",
        "sector_heatmap": "板块热力图 · 实时", "fear_greed": "恐贪指数 · 实时",
        "days_loaded": "✓ 已加载 {ticker} 的 {n} 个交易日", "fetching": "正在获取 {ticker} 数据...",
        "engineering": "正在计算技术特征...", "building": "正在构建特征矩阵...",
        "training": "XGBoost 模型训练中...", "running_backtest": "正在运行回测模拟...",
        "running_bootstrap": "正在运行 {n} 次 Bootstrap 采样...", "training_lr": "正在训练线性回归...",
        "fetching_market": "正在加载实时市场数据...", "fetching_financial": "正在获取 {ticker} 财务数据...",
        "not_enough_data": "数据不足以训练模型。请尝试更长的日期范围或更小的回溯窗口。",
        "enter_ticker": "请输入股票代码。", "max_holdings": "已达到最大持仓数 {n}。",
        "loading_prices": "正在获取 {sym} 的实时价格...",
        "deep_analysis": "📈  深度分析",
        "model_conf_score": "助手置信度评分",
        "high_confidence": "高置信度", "moderate_confidence": "中等置信度", "low_confidence": "低置信度",
        "r2_fit": "R² 拟合", "mape_accuracy": "MAPE 准确率", "directional_acc": "方向准确率", "data_volume": "数据量",
        "composite_signal": "研究信号", "forecast_lbl": "走势",
        "score_lbl": "评分",
        "take_profit_lbl": "止盈", "stop_loss_lbl": "止损", "risk_reward_lbl": "风险 / 收益",
        "rsi_lbl": "RSI (14)",
        "favorable": "✓ 有利", "marginal": "⚠ 边际", "unfavorable": "✗ 不利",
        "oversold_zone": "超卖区域", "overbought_zone": "超买区域", "neutral_zone": "中性区域",
        "factor_breakdown": "6因子信号分析",
        "last_close_lbl": "最新收盘", "model_confidence_lbl": "助手置信度",
        "high_lbl": "高", "moderate_lbl": "中", "low_lbl": "低",
        "at_above_target": "等于或高于目标",
        "below_target": "低于目标",
        "shariah_debt_mktcap": "债务/市值",
        "shariah_debt_assets": "债务/资产",
        "shariah_cash_assets": "现金/资产",
        "known_noncompliant": "已知不合规代码",
    },
}
# ── Page config — MUST be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Stockcast · Your AI Stock Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

if _SUPABASE_MISSING:
    st.error("⚠ Supabase credentials not found. Add SUPABASE_URL and SUPABASE_KEY to your Streamlit secrets or environment variables.")
    st.stop()

# ── Session state ──────────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "English"
if "user" not in st.session_state:
    st.session_state.user = None
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "alert_signals" not in st.session_state:
    st.session_state.alert_signals = {}
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []
if "portfolio_history" not in st.session_state:
    st.session_state.portfolio_history = []
if "run_pressed" not in st.session_state:
    st.session_state.run_pressed = False
if "usage_count" not in st.session_state:
    st.session_state.usage_count = 0
if "analyses_today" not in st.session_state:
    st.session_state.analyses_today = 0
if "user_plan" not in st.session_state:
    st.session_state.user_plan = "free"
if "show_onboarding" not in st.session_state:
    st.session_state.show_onboarding = True
if "load_ticker_from_watchlist" not in st.session_state:
    st.session_state.load_ticker_from_watchlist = None
if "show_upgrade_modal" not in st.session_state:
    st.session_state.show_upgrade_modal = False
if "email_alerts_enabled" not in st.session_state:
    st.session_state.email_alerts_enabled = False


# ── Plotly theme ───────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#080e1c",
    plot_bgcolor="#080e1c",
    font=dict(family="Manrope", color="#8a8fa0", size=11),
    xaxis=dict(gridcolor="#1e2740", linecolor="#1e2740", tickfont=dict(color="#8a8fa0", size=10),
               showgrid=True, zeroline=False, showspikes=True, spikethickness=1,
               spikecolor="#4d8eff", spikedash="dot"),
    yaxis=dict(gridcolor="#1e2740", linecolor="#1e2740", tickfont=dict(color="#8a8fa0", size=10),
               showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(8,14,28,0.8)", bordercolor="#252f47", borderwidth=1,
                font=dict(size=11, family="Manrope"), itemsizing="constant"),
    margin=dict(l=12, r=12, t=44, b=12),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#0f1727", bordercolor="#252f47",
                    font=dict(family="IBM Plex Mono", size=11, color="#e4eafd")),
    dragmode="pan",
    selectdirection="h",
)

C_GREEN   = "#adc6ff"
C_ACCENT  = "#4d8eff"
C_RED     = "#ff5f5f"
C_YELLOW  = "#ffd426"
C_GREY    = "#8a8fa0"
C_EMERALD = "#00e5b0"

# ── Master CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── ROOT ── */
:root {
    --bg:          #070d1a;
    --bg2:         #0d1524;
    --bg3:         #131e30;
    --bg4:         #1a2640;
    --bg5:         #243050;
    --primary:     #c2d4ff;
    --accent:      #4d8eff;
    --accent2:     #7ab0ff;
    --on-primary:  #002e6a;
    --secondary:   #b1c6f9;
    --t1:          #eaefff;
    --t2:          #b8c4d8;
    --t3:          #7a8299;
    --t4:          #3d4760;
    --border:      #1e2d45;
    --border2:     #2d3f5a;
    --emerald:     #00d9a6;
    --red:         #ff5757;
    --yellow:      #ffcb2b;
    --mono:        'IBM Plex Mono', monospace;
    --sans:        'Manrope', sans-serif;
    --radius:      0.5rem;
    --radius-lg:   0.875rem;
    --shadow-sm:   0 1px 6px rgba(0,0,0,0.35);
    --shadow-md:   0 4px 20px rgba(0,0,0,0.5);
    --shadow-lg:   0 10px 40px rgba(0,0,0,0.6);
    /* Accessible font sizes — nothing below 0.7rem */
    --text-2xs:    0.7rem;
    --text-xs:     0.75rem;
    --text-sm:     0.82rem;
    --text-base:   0.9rem;
    --text-md:     1rem;
}

/* ── GLOBAL ── */
html, body, [class*="css"], [data-testid="stApp"],
[data-testid="stAppViewContainer"], .main {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--t1) !important;
    -webkit-font-smoothing: antialiased !important;
}
.block-container {
    padding: 1.25rem 2rem 4rem 2rem !important;
    max-width: 1300px !important;
    margin: 0 auto !important;
}

/* Subtle ambient glow */
[data-testid="stApp"]::before {
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse 70% 40% at 8% 0%, rgba(77,142,255,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 50% 35% at 92% 100%, rgba(0,217,166,0.03) 0%, transparent 55%);
    pointer-events: none; z-index: 0;
}

/* ── GLOBAL FONT FLOOR — nothing illegibly small ── */
.stat-label, .bento-label, .sig-lbl, .bt-label, .sir-sig,
.signal-lbl, .meter-title, .section-title, .plan-badge-label,
.accordion-num, .accordion-badge, .chip, .live-label,
.trust-item, .nav-item-active, .nav-item-idle,
.data-ts, .sk-line, .tab-dot {
    font-size: var(--text-2xs) !important;
}
.stat-sub, .bento-sub, .sig-sub, .sir-label, .wl-badge,
.plan-badge-value, .freshness-badge, .disclaimer-pill,
[data-testid="stMetricLabel"], .sc-toast-msg {
    font-size: var(--text-xs) !important;
}
.bento-desc, .accordion-trigger, .accordion-body-inner div,
.halal-card, .halal-card-fail, p, .stMarkdown p,
.sc-toast-title, [data-testid="stTabs"] [role="tab"] {
    font-size: var(--text-sm) !important;
}
h2, h3 { font-size: 0.82rem !important; }
h4     { font-size: 0.76rem !important; }
.stat-row { font-size: var(--text-xs) !important; }
[data-testid="stMetricLabel"] {
    font-size: var(--text-2xs) !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stSidebar"] input { font-size: var(--text-sm) !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(175deg, #0d1524 0%, #07111f 100%) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] * { color: var(--t3) !important; }
[data-testid="stSidebar"] input {
    background-color: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    color: var(--primary) !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 0.85rem !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(77,142,255,0.2) !important;
    outline: none !important;
}
/* Sidebar selectbox */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background-color: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    font-size: 0.82rem !important;
}
/* Sidebar labels */
[data-testid="stSidebar"] label {
    font-size: var(--text-xs) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    color: var(--t4) !important;
}
/* Sidebar dividers */
[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    margin: 0.8rem 0 !important;
}
/* Sidebar checkbox */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    font-size: 0.8rem !important;
    letter-spacing: 0.01em !important;
    text-transform: none !important;
    color: var(--t2) !important;
    font-weight: 500 !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #3a76ed 0%, #5294ff 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--sans) !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.18s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: 0 2px 10px rgba(77,142,255,0.28) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4d8eff 0%, #6aabff 100%) !important;
    box-shadow: 0 4px 18px rgba(77,142,255,0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 1px 5px rgba(77,142,255,0.3) !important;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    padding: 0.7rem 1.4rem !important;
    font-size: 0.76rem !important;
    border-radius: var(--radius) !important;
}

/* ── METRICS ── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, var(--bg2), #080f1c) !important;
    border: 1px solid rgba(255,255,255,0.04) !important;
    border-top: 2px solid var(--accent) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.1rem 1.3rem !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(77,142,255,0.3) !important;
    border-top-color: var(--accent) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md) !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--sans) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--t3) !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: var(--primary) !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricDelta"] {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
}

/* ── HEADINGS ── */
h2, h3 {
    font-family: var(--sans) !important;
    color: var(--t2) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0.5rem !important;
    margin-top: 1.6rem !important;
    margin-bottom: 0.9rem !important;
    font-weight: 800 !important;
}
h4 {
    font-family: var(--sans) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    color: var(--t3) !important;
    margin-top: 1rem !important;
}
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }
p, .stMarkdown p {
    color: var(--t2) !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    background: var(--bg2) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* ── FORM INPUTS ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background-color: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    color: var(--t1) !important;
    font-family: var(--mono) !important;
    font-size: 0.84rem !important;
    padding: 0.5rem 0.8rem !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(77,142,255,0.18) !important;
    outline: none !important;
}
[data-testid="stSelectbox"] > div > div {
    background-color: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    color: var(--t1) !important;
    font-size: 0.84rem !important;
}

/* ── LABELS ── */
label, [data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label {
    font-family: var(--sans) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    color: var(--t3) !important;
    font-weight: 700 !important;
    margin-bottom: 0.25rem !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(77,142,255,0.22) !important;
}

/* ── CHECKBOX ── */
[data-testid="stCheckbox"] label {
    font-family: var(--sans) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.01em !important;
    text-transform: none !important;
    color: var(--t2) !important;
    font-weight: 500 !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--bg2) !important;
    margin-bottom: 0.5rem !important;
    transition: border-color 0.18s !important;
}
[data-testid="stExpander"]:hover {
    border-color: var(--border2) !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--sans) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--t2) !important;
    padding: 0.65rem 1rem !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] p {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    color: var(--accent) !important;
    letter-spacing: 0.06em !important;
}

/* ── ALERTS ── */
[data-testid="stSuccess"] {
    background: rgba(0,217,166,0.06) !important;
    border: 1px solid rgba(0,217,166,0.2) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stInfo"] {
    background: rgba(77,142,255,0.06) !important;
    border: 1px solid rgba(77,142,255,0.18) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stWarning"] {
    background: rgba(255,203,43,0.06) !important;
    border: 1px solid rgba(255,203,43,0.18) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stError"] {
    background: rgba(255,87,87,0.06) !important;
    border: 1px solid rgba(255,87,87,0.18) !important;
    border-radius: var(--radius) !important;
}

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] {
    background: linear-gradient(90deg, var(--bg2) 0%, var(--bg3) 100%) !important;
    border: 1px solid var(--border) !important;
    border-bottom: none !important;
    border-radius: var(--radius-lg) var(--radius-lg) 0 0 !important;
    gap: 2px !important;
    padding: 0.3rem 0.45rem !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--sans) !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    color: var(--t4) !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius) !important;
    text-transform: uppercase !important;
    padding: 0.48rem 1rem !important;
    transition: color 0.15s, background 0.15s, border-color 0.15s !important;
    white-space: nowrap !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: var(--t2) !important;
    background: rgba(255,255,255,0.03) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--t1) !important;
    background: linear-gradient(135deg, rgba(77,142,255,0.14) 0%, rgba(77,142,255,0.05) 100%) !important;
    border-color: rgba(77,142,255,0.28) !important;
    box-shadow: 0 0 12px rgba(77,142,255,0.12), inset 0 1px 0 rgba(77,142,255,0.15) !important;
}
[data-testid="stTabPanel"] {
    background: linear-gradient(180deg, rgba(77,142,255,0.015) 0%, transparent 50px) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-lg) var(--radius-lg) !important;
    padding: 1.2rem 0.2rem 0.3rem !important;
}
.tab-dot {
    display: inline-block; width: 6px; height: 6px;
    background: var(--red); border-radius: 50%;
    margin-left: 5px; vertical-align: top; margin-top: 1px;
    box-shadow: 0 0 5px rgba(255,87,87,0.65);
    animation: tab-dot-pulse 2s ease-in-out infinite;
}
.tab-dot.green  { background: var(--emerald); box-shadow: 0 0 5px rgba(0,217,166,0.65); }
.tab-dot.yellow { background: var(--yellow);  box-shadow: 0 0 5px rgba(255,203,43,0.65); }
@keyframes tab-dot-pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.5; transform: scale(0.8); }
}

/* ── APP HEADER ── */
.wi-header {
    background: linear-gradient(90deg, var(--bg2) 0%, var(--bg3) 60%, var(--bg2) 100%);
    border-bottom: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    padding: 1.1rem 2rem;
    margin: 1.5rem -2rem 1.5rem -2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 24px rgba(0,0,0,0.4);
}
.wi-logo {
    font-family: var(--sans);
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--t1);
    letter-spacing: -0.02em;
}
.wi-logo span { color: var(--accent); }
.wi-sub {
    font-size: 0.7rem;
    color: var(--t3);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.2rem;
    font-weight: 600;
    line-height: 1.4;
}
.live-dot {
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--emerald);
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
    margin-right: 5px;
    vertical-align: middle;
    box-shadow: 0 0 8px rgba(0,217,166,0.55);
}
@keyframes pulse-dot {
    0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(0,217,166,0.45); }
    50%      { opacity:.8; box-shadow: 0 0 0 6px rgba(0,217,166,0); }
}
.live-label {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--emerald);
    letter-spacing: 0.1em;
    vertical-align: middle;
}

/* ── TICKER TAPE ── */
.ticker-tape-wrap {
    overflow: hidden;
    background: linear-gradient(90deg, var(--bg2), var(--bg3), var(--bg2));
    border-bottom: 1px solid var(--border);
    border-top: 1px solid var(--border);
    padding: 0.28rem 0;
    margin: 0 -2rem 1.8rem -2rem;
}
.ticker-tape {
    display: inline-flex;
    gap: 2.5rem;
    animation: tape 40s linear infinite;
    white-space: nowrap;
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.03em;
    color: var(--t3);
}
.ticker-tape:hover { animation-play-state: paused; }
@keyframes tape { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
.tape-up   { color: var(--emerald); font-weight: 700; }
.tape-down { color: var(--red); font-weight: 700; }
.tape-sym  { color: var(--t4); font-size: 0.62rem; margin-right: 0.25rem; }

/* ── DATA FRESHNESS BADGE ── */
.freshness-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: rgba(0,217,166,0.05);
    border: 1px solid rgba(0,217,166,0.16);
    border-radius: 2rem; padding: 0.25rem 0.7rem;
    font-family: var(--mono); font-size: 0.68rem;
    letter-spacing: 0.07em; color: var(--emerald);
}

/* ── GLASS CARDS ── */
.wi-card {
    background: linear-gradient(145deg, #0d1524, #07101e);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: var(--radius-lg);
    padding: 1.3rem 1.5rem;
    transition: all 0.22s ease;
    box-shadow: var(--shadow-sm);
}
.wi-card:hover {
    border-color: rgba(77,142,255,0.28);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
.wi-card-accent  { border-top: 2px solid var(--accent); }
.wi-card-emerald { border-top: 2px solid var(--emerald); }
.wi-card-red     { border-top: 2px solid var(--red); }
.wi-card-yellow  { border-top: 2px solid var(--yellow); }

/* ── STAT GRID ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}
@media (max-width: 768px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr) !important; gap: .55rem !important; }
    .block-container { padding: .7rem .7rem 3rem .7rem !important; }
    .wi-header { padding: .75rem 1rem !important; margin: 0 -.7rem 1rem -.7rem !important; flex-wrap: wrap; gap: .4rem; }
    .wi-sub { display: none !important; }
    .trust-row { display: none !important; }
    .ticker-tape-wrap { margin: 0 -.7rem 1rem -.7rem !important; }
    .signal-panel { flex-direction: column !important; }
    .signal-main { flex: none !important; width: 100% !important; }
    .signal-details { grid-template-columns: 1fr 1fr !important; width: 100% !important; }
    .stat-value { font-size: 1.25rem !important; }
    .wi-logo { font-size: 1.2rem !important; }
    h2, h3 { margin-top: 1rem !important; }
}
@media (max-width: 480px) {
    .stat-grid { grid-template-columns: 1fr 1fr !important; }
    .stat-value { font-size: 1.05rem !important; }
}
.stat-card {
    background: linear-gradient(145deg, var(--bg2), #080d1c);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    border-radius: var(--radius-lg);
    padding: 1rem 1.25rem;
    position: relative; overflow: hidden;
    transition: transform 0.18s, box-shadow 0.18s;
    box-shadow: var(--shadow-sm);
}
.stat-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
.stat-card::after {
    content: ''; position: absolute; top: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient(circle at top right, rgba(77,142,255,0.07), transparent 70%);
}
.stat-label {
    font-family: var(--sans);
    font-size: 0.68rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--t3);
    font-weight: 700;
    margin-bottom: 5px;
}
.stat-value {
    font-family: var(--mono);
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--primary);
    line-height: 1.1;
}
.stat-sub { font-size: 0.68rem; color: var(--t3); margin-top: 4px; font-family: var(--sans); font-weight: 600; }

/* ── SIGNAL PANEL ── */
.signal-panel { display: flex; gap: 1rem; margin: 1.2rem 0; flex-wrap: wrap; }
.signal-main {
    flex: 0 0 250px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 2rem 1.6rem;
    border: 1.5px solid var(--accent);
    background: rgba(77,142,255,0.04);
    border-radius: var(--radius-lg);
    position: relative; overflow: hidden;
    box-shadow: 0 0 30px rgba(77,142,255,0.08), var(--shadow-sm);
    transition: box-shadow 0.2s;
}
.signal-main::before {
    content: ''; position: absolute; bottom: -20px; right: -20px;
    width: 100px; height: 100px; border-radius: 50%;
    background: radial-gradient(circle, rgba(77,142,255,0.15) 0%, transparent 70%);
}
.signal-main.sell { border-color: var(--red); background: rgba(255,87,87,0.04); box-shadow: 0 0 30px rgba(255,87,87,0.08), var(--shadow-sm); }
.signal-main.sell::before { background: radial-gradient(circle, rgba(255,87,87,0.15) 0%, transparent 70%); }
.signal-main.hold { border-color: var(--yellow); background: rgba(255,203,43,0.04); box-shadow: 0 0 30px rgba(255,203,43,0.08), var(--shadow-sm); }
.signal-main.hold::before { background: radial-gradient(circle, rgba(255,203,43,0.15) 0%, transparent 70%); }
.signal-action {
    font-family: var(--mono); font-size: 2.2rem; font-weight: 800;
    letter-spacing: 0.14em; color: var(--primary); line-height: 1;
}
.signal-action.sell { color: var(--red); }
.signal-action.hold { color: var(--yellow); }
.signal-pct { font-family: var(--mono); font-size: 1rem; font-weight: 600; margin-top: 0.5rem; color: var(--t1); }
.signal-lbl { font-size: 0.68rem; letter-spacing: 0.16em; color: var(--t3); margin-top: 8px; text-transform: uppercase; font-weight: 700; font-family: var(--sans); }
.signal-details { flex: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; min-width: 200px; }
@media (max-width: 600px) {
    .signal-panel { flex-direction: column; }
    .signal-main  { flex: none; width: 100%; }
    .signal-details { min-width: 0; width: 100%; }
}
.sig-card {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    padding: 0.8rem 1rem; position: relative;
    border-radius: var(--radius); overflow: hidden;
    transition: transform 0.15s, border-color 0.15s;
}
.sig-card:hover { transform: translateY(-1px); border-color: var(--border2); }
.sig-card::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 3px; height: 100%; background: var(--border); border-radius: 2px 0 0 2px;
}
.sig-card.positive::before { background: var(--emerald); }
.sig-card.negative::before { background: var(--red); }
.sig-card.neutral::before  { background: var(--yellow); }
.sig-lbl { font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--t3); margin-bottom: 4px; font-weight: 700; font-family: var(--sans); }
.sig-val { font-family: var(--mono); font-size: 0.9rem; font-weight: 700; color: var(--t1); }
.sig-sub { font-size: 0.68rem; color: var(--t3); margin-top: 2px; font-family: var(--sans); }

/* ── COMPOSITE METER ── */
.composite-meter {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    padding: 1.1rem 1.5rem;
    margin: 0.8rem 0;
    border-radius: 0 var(--radius-lg) var(--radius-lg) 0;
}
.meter-title { font-size: 0.68rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--t3); margin-bottom: 0.8rem; font-weight: 700; font-family: var(--sans); }
.sir { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.45rem; font-family: var(--mono); font-size: 0.72rem; }
.sir-label { color: var(--t2); width: 128px; flex-shrink: 0; }
.sir-bar-bg { flex: 1; height: 4px; background: rgba(255,255,255,0.04); border-radius: 2px; overflow: hidden; }
.sir-bar { height: 100%; border-radius: 2px; transition: width 0.7s cubic-bezier(0.4,0,0.2,1); }
.sir-bar.positive { background: linear-gradient(90deg, var(--emerald), rgba(0,217,166,0.4)); }
.sir-bar.negative { background: linear-gradient(90deg, var(--red), rgba(255,87,87,0.4)); }
.sir-bar.neutral  { background: linear-gradient(90deg, var(--yellow), rgba(255,203,43,0.4)); }
.sir-val { width: 55px; text-align: right; font-weight: 600; color: var(--t1); }
.sir-sig { width: 40px; text-align: right; font-size: 0.65rem; letter-spacing: 0.07em; font-weight: 700; }
.sir-sig.buy { color: var(--emerald); }
.sir-sig.sell { color: var(--red); }
.sir-sig.hold { color: var(--yellow); }

/* ── CHIPS ── */
.chip {
    display: inline-flex; align-items: center; gap: 4px;
    font-family: var(--mono); font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase;
    padding: 0.27rem 0.7rem; border-radius: 100px;
    border: 1px solid var(--border2); background: var(--bg3);
    color: var(--t3); white-space: nowrap; user-select: none;
    transition: all 0.15s ease;
}
.chip:hover { border-color: var(--accent); color: var(--t2); background: rgba(77,142,255,0.07); }
.chip.buy    { background: rgba(0,217,166,0.09);  border-color: rgba(0,217,166,0.35);  color: var(--emerald); }
.chip.sell   { background: rgba(255,87,87,0.09);  border-color: rgba(255,87,87,0.35);  color: var(--red); }
.chip.hold   { background: rgba(255,203,43,0.09); border-color: rgba(255,203,43,0.35); color: var(--yellow); }
.chip.live   { background: rgba(0,217,166,0.07);  border-color: rgba(0,217,166,0.22);  color: var(--emerald); animation: chip-live-pulse 2.5s ease-in-out infinite; }
.chip.info   { background: rgba(77,142,255,0.09); border-color: rgba(77,142,255,0.32); color: var(--accent); }
.chip.pro    { background: rgba(255,203,43,0.09); border-color: rgba(255,203,43,0.32); color: var(--yellow); }
.chip.ai     { background: rgba(194,212,255,0.08);border-color: rgba(194,212,255,0.25);color: var(--primary); }
.chip.dot::before { content: ''; width: 5px; height: 5px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
@keyframes chip-live-pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(0,217,166,0.28); } 50% { box-shadow: 0 0 0 3px rgba(0,217,166,0); } }
.chip-group { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; }

/* ── BT CARDS ── */
.bt-card {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border); border-top: 2px solid var(--border2);
    padding: 1rem 1.2rem; margin-bottom: 0.45rem;
    font-family: var(--mono); border-radius: var(--radius);
    transition: transform 0.15s;
}
.bt-card:hover { transform: translateY(-1px); }
.bt-label { font-size: 0.68rem; color: var(--t3); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 4px; font-family: var(--sans); font-weight: 700; }
.bt-val       { font-size: 1.3rem; font-weight: 700; color: var(--t1); }
.bt-val-green { font-size: 1.3rem; font-weight: 700; color: var(--emerald); }
.bt-val-red   { font-size: 1.3rem; font-weight: 700; color: var(--red); }

/* ── HALAL CARDS ── */
.halal-card {
    background: rgba(0,217,166,0.03); border: 1px solid rgba(0,217,166,0.13);
    border-left: 3px solid var(--emerald); padding: 0.85rem 1.2rem; margin: 0.35rem 0;
    font-family: var(--sans); font-size: 0.82rem; color: var(--t2); line-height: 1.5;
    border-radius: 0 var(--radius) var(--radius) 0;
}
.halal-card-fail {
    background: rgba(255,87,87,0.03); border: 1px solid rgba(255,87,87,0.13);
    border-left: 3px solid var(--red); padding: 0.85rem 1.2rem; margin: 0.35rem 0;
    font-family: var(--sans); font-size: 0.82rem; color: var(--t2); line-height: 1.5;
    border-radius: 0 var(--radius) var(--radius) 0;
}

/* ── MODEL BADGE ── */
.model-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: rgba(77,142,255,0.08); border: 1px solid rgba(77,142,255,0.2);
    color: var(--primary); font-family: var(--sans); font-size: 0.68rem;
    font-weight: 700; padding: 0.22rem 0.85rem; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 0.8rem; border-radius: 2rem;
}

/* ── ALERT BOX ── */
.alert-box {
    background: rgba(77,142,255,0.05); border: 1px solid rgba(77,142,255,0.22);
    border-left: 3px solid var(--accent); padding: 0.85rem 1.3rem;
    font-family: var(--sans); font-size: 0.82rem; color: var(--primary);
    margin: 0.8rem 0; letter-spacing: 0.02em;
    border-radius: 0 var(--radius) var(--radius) 0; line-height: 1.55;
}

/* ── SIDEBAR STAT ROW ── */
.stat-row {
    font-family: var(--sans); font-size: 0.7rem; color: var(--t3);
    letter-spacing: 0.09em; text-transform: uppercase;
    margin-bottom: 4px; margin-top: 2px; font-weight: 700;
}

/* ── PLAN BADGE ── */
.plan-badge {
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(77,142,255,0.05); border: 1px solid rgba(77,142,255,0.15);
    border-radius: var(--radius); padding: 0.5rem 0.8rem; margin: 0.45rem 0;
}
.plan-badge-label {
    font-family: var(--sans); font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--t3);
}
.plan-badge-value {
    font-family: var(--mono); font-size: 0.7rem; font-weight: 600; color: var(--accent);
}
.usage-bar-bg { width: 100%; height: 3px; background: rgba(255,255,255,0.05); border-radius: 2px; margin-top: 0.35rem; overflow: hidden; }
.usage-bar-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--accent), var(--accent2)); transition: width 0.6s ease; }

/* ── NAV ITEMS ── */
.nav-item-active {
    background: var(--bg4); border-left: 3px solid var(--accent);
    color: var(--primary) !important; padding: 0.5rem 1rem;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase; margin: 2px 0;
    border-radius: 0 var(--radius) var(--radius) 0; font-family: var(--sans);
}
.nav-item-idle {
    color: var(--t3); padding: 0.5rem 1rem; font-size: 0.7rem;
    font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
    margin: 2px 0; font-family: var(--sans);
}

/* ── WATCHLIST BADGE ── */
.wl-badge {
    display: flex; justify-content: space-between; align-items: center;
    background: var(--bg3); border: 1px solid var(--border);
    padding: 0.5rem 0.75rem; border-radius: var(--radius);
    margin-bottom: 0.28rem; font-family: var(--mono);
    font-size: 0.72rem; transition: background 0.15s, border-color 0.15s;
}
.wl-badge:hover { background: var(--bg4); border-color: var(--border2); }

/* ── METRIC CARD (feature grid) ── */
.metric-card {
    background: linear-gradient(145deg, #0d1524, #07101e);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: var(--radius-lg); padding: 1.2rem 1.3rem;
    transition: all 0.22s ease; height: 100%; box-shadow: var(--shadow-sm);
}
.metric-card:hover { border-color: rgba(77,142,255,0.28); transform: translateY(-2px); box-shadow: var(--shadow-md); }
.section-title {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.11em;
    color: var(--t3); margin-bottom: 0.5rem; font-weight: 700;
    font-family: var(--sans); line-height: 1.4;
}

/* ── TRUST ELEMENTS ── */
.trust-row { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; margin: 0.6rem 0 0; }
.trust-item {
    display: flex; align-items: center; gap: 0.28rem;
    font-family: var(--sans); font-size: 0.68rem; font-weight: 600;
    color: var(--t4); letter-spacing: 0.04em; text-transform: uppercase;
}
.trust-item-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--emerald); flex-shrink: 0; }

/* ── DISCLAIMER PILL ── */
.disclaimer-pill {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: rgba(255,87,87,0.06); border: 1px solid rgba(255,87,87,0.16);
    border-radius: 2rem; padding: 0.28rem 0.8rem;
    font-family: var(--mono); font-size: 0.68rem;
    letter-spacing: 0.06em; color: rgba(255,87,87,0.65);
}

/* ── TIMESTAMP ── */
.data-ts {
    font-family: var(--mono); font-size: 0.65rem;
    color: var(--t4); letter-spacing: 0.06em; margin-top: 0.25rem;
}

/* ── SKELETON LOADING ── */
.skeleton {
    background: linear-gradient(90deg, var(--bg3) 25%, rgba(255,255,255,0.03) 50%, var(--bg3) 75%);
    background-size: 400% 100%; animation: skeleton-shimmer 1.6s ease-in-out infinite;
    border-radius: var(--radius);
}
@keyframes skeleton-shimmer { 0% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
.skeleton-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 1rem 1.25rem;
    overflow: hidden; position: relative;
}
.skeleton-card::after {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(105deg, transparent 40%, rgba(255,255,255,0.025) 50%, transparent 60%);
    background-size: 200% 100%; animation: sk-overlay 1.6s ease-in-out infinite;
}
@keyframes sk-overlay { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
.sk-line       { height: 9px;  margin-bottom: 7px; border-radius: 3px; }
.sk-line.w-100 { width: 100%; } .sk-line.w-80 { width: 80%; }
.sk-line.w-60  { width: 60%; }  .sk-line.w-40 { width: 40%; }
.sk-line.title { height: 18px; width: 50%; margin-bottom: 9px; }
.sk-line.value { height: 26px; width: 70%; margin-bottom: 5px; }
.sk-circle     { width: 38px; height: 38px; border-radius: 50%; flex-shrink: 0; }

/* ── ACCORDION ── */
.accordion { border: 1px solid var(--border); border-radius: var(--radius-lg); overflow: hidden; margin: 0.45rem 0; }
.accordion-item { border-bottom: 1px solid var(--border); }
.accordion-item:last-child { border-bottom: none; }
.accordion-trigger {
    width: 100%; display: flex; align-items: center; justify-content: space-between;
    padding: 0.85rem 1.2rem; background: var(--bg2); border: none; cursor: pointer;
    font-family: var(--sans); font-size: 0.78rem; font-weight: 700;
    color: var(--t2); text-align: left; gap: 0.75rem;
    transition: background 0.15s, color 0.15s;
}
.accordion-trigger:hover  { background: var(--bg3); color: var(--t1); }
.accordion-trigger.active { color: var(--accent); background: rgba(77,142,255,0.04); }
.accordion-icon {
    width: 17px; height: 17px; display: flex; align-items: center; justify-content: center;
    border-radius: 50%; border: 1px solid var(--border2); flex-shrink: 0;
    font-size: 0.62rem; color: var(--t4);
    transition: transform 0.25s cubic-bezier(0.4,0,0.2,1), border-color 0.15s;
}
.accordion-trigger.active .accordion-icon { transform: rotate(180deg); border-color: var(--accent); color: var(--accent); }
.accordion-num   { font-family: var(--mono); font-size: 0.65rem; color: var(--t4); flex-shrink: 0; }
.accordion-label { flex: 1; }
.accordion-badge { font-family: var(--mono); font-size: 0.62rem; color: var(--t4); flex-shrink: 0; }
.accordion-body  { max-height: 0; overflow: hidden; transition: max-height 0.35s cubic-bezier(0.4,0,0.2,1); background: var(--bg); }
.accordion-body.open { max-height: 900px; }
.accordion-body-inner { padding: 0.9rem 1.2rem 1.1rem; }

/* ── TOAST ── */
#sc-toasts { position: fixed; bottom: 1.5rem; right: 1.5rem; z-index: 99999; display: flex; flex-direction: column-reverse; gap: 0.55rem; pointer-events: none; }
.sc-toast {
    display: flex; align-items: flex-start; gap: 0.7rem;
    min-width: 270px; max-width: 360px; padding: 0.8rem 0.95rem;
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: var(--radius-lg); position: relative;
    box-shadow: 0 8px 28px rgba(0,0,0,0.55); pointer-events: all;
    animation: sc-toast-in 0.3s cubic-bezier(0.34,1.56,0.64,1) forwards;
}
.sc-toast.hiding { animation: sc-toast-out 0.22s ease-in forwards; }
@keyframes sc-toast-in  { from{opacity:0;transform:translateX(16px) scale(0.94)} to{opacity:1;transform:translateX(0) scale(1)} }
@keyframes sc-toast-out { from{opacity:1;transform:translateX(0) scale(1)} to{opacity:0;transform:translateX(16px) scale(0.92)} }
.sc-toast::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; border-radius: var(--radius-lg) 0 0 var(--radius-lg); }
.sc-toast.t-success::before { background: var(--emerald); }
.sc-toast.t-error::before   { background: var(--red); }
.sc-toast.t-warn::before    { background: var(--yellow); }
.sc-toast.t-info::before    { background: var(--accent); }
.sc-toast-icon  { font-size: 0.95rem; line-height: 1; flex-shrink: 0; margin-top: 1px; }
.sc-toast-body  { flex: 1; min-width: 0; }
.sc-toast-title { font-family: var(--sans); font-size: 0.74rem; font-weight: 700; color: var(--t1); margin-bottom: 2px; }
.sc-toast-msg   { font-family: var(--sans); font-size: 0.69rem; color: var(--t3); line-height: 1.45; }
.sc-toast-close { background: none; border: none; color: var(--t4); cursor: pointer; font-size: 0.72rem; padding: 0; flex-shrink: 0; transition: color 0.12s; pointer-events: all; }
.sc-toast-close:hover { color: var(--t2); }
.sc-toast-bar   { position: absolute; bottom: 0; left: 0; height: 2px; width: 100%; overflow: hidden; border-radius: 0 0 var(--radius-lg) var(--radius-lg); background: rgba(255,255,255,0.05); }


</style>

<div id="sc-toasts"></div>

<script>
(function() {
  /* ---- TOAST ENGINE ---- */
  var ICONS = { success:'&#10003;', error:'&#9888;', warn:'&#9889;', info:'&#9672;' };
  window.SCToast = function(title, message, type, duration) {
    type = type || 'info';
    duration = duration === undefined ? 4000 : duration;
    var container = document.getElementById('sc-toasts');
    if (!container) return;
    var t = document.createElement('div');
    t.className = 'sc-toast t-' + type;
    t.innerHTML =
      '<span class="sc-toast-icon">' + (ICONS[type] || '&#9672;') + '</span>' +
      '<div class="sc-toast-body">' +
        '<div class="sc-toast-title">' + title + '</div>' +
        (message ? '<div class="sc-toast-msg">' + message + '</div>' : '') +
      '</div>' +
      '<button class="sc-toast-close" onclick="this.closest('.sc-toast').remove()">&#x2715;</button>' +
      '<div class="sc-toast-bar"><div class="sc-toast-bar-fill" style="animation-duration:' + duration + 'ms"></div></div>';
    container.appendChild(t);
    setTimeout(function() {
      t.classList.add('hiding');
      setTimeout(function() { if(t.parentNode) t.parentNode.removeChild(t); }, 280);
    }, duration);
  };

  /* ---- ACCORDION ENGINE ---- */
  function initAccordions() {
    var triggers = document.querySelectorAll('.accordion-trigger');
    for (var i = 0; i < triggers.length; i++) {
      (function(trigger) {
        if (trigger._sc) return;
        trigger._sc = true;
        trigger.addEventListener('click', function() {
          var body = trigger.nextElementSibling;
          var isOpen = body.classList.contains('open');
          var acc = trigger.closest('.accordion');
          if (acc) {
            var open = acc.querySelectorAll('.accordion-body.open');
            for (var j = 0; j < open.length; j++) {
              open[j].classList.remove('open');
              if (open[j].previousElementSibling) open[j].previousElementSibling.classList.remove('active');
            }
          }
          if (!isOpen) { body.classList.add('open'); trigger.classList.add('active'); }
        });
      })(triggers[i]);
    }
  }
  var _obs = new MutationObserver(function() { initAccordions(); });
  _obs.observe(document.body, { childList: true, subtree: true });
  initAccordions();

  /* ---- NOTIFY HELPERS ---- */
  window.SCNotify = {
    saved:   function(s) { SCToast('Watchlist Updated', s + ' added', 'success'); },
    removed: function(s) { SCToast('Watchlist Updated', s + ' removed', 'info'); },
    signal:  function(s, v) { SCToast('Signal: ' + v, s + ' flipped to ' + v, v==='BUY'?'success':v==='SELL'?'error':'warn'); },
    email:   function()  { SCToast('Email Sent', 'Signal digest delivered', 'success'); },
    error:   function(m) { SCToast('Error', m, 'error'); },
    copy:    function()  { SCToast('Copied', 'Copied to clipboard', 'info'); },
    macro:   function(l, sc) { SCToast('Macro Climate: ' + l, 'Score ' + sc + '/100', l==='RAISE NOW'?'success':l==='WAIT'?'error':'warn'); }
  };

  /* ---- WELCOME TOAST (once per session) ---- */
  try {
    if (!sessionStorage.getItem('sc-v')) {
      setTimeout(function() { SCToast('Stockcast Ready', 'AI Stock Assistant &#183; Live Data Connected', 'success', 5000); }, 1400);
      sessionStorage.setItem('sc-v', '1');
    }
  } catch(e) {}
})();
</script>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

POPULAR_TICKERS = {
    "AAPL":"Apple Inc.","MSFT":"Microsoft Corp.","GOOGL":"Alphabet Inc.",
    "AMZN":"Amazon.com Inc.","NVDA":"NVIDIA Corp.","META":"Meta Platforms",
    "TSLA":"Tesla Inc.","NFLX":"Netflix Inc.","AMD":"Advanced Micro Devices",
    "ORCL":"Oracle Corp.","INTC":"Intel Corp.","CRM":"Salesforce Inc.",
    "ADBE":"Adobe Inc.","PYPL":"PayPal Holdings","UBER":"Uber Technologies",
    "BABA":"Alibaba Group","JPM":"JPMorgan Chase","BAC":"Bank of America",
    "GS":"Goldman Sachs","V":"Visa Inc.","MA":"Mastercard Inc.",
    "JNJ":"Johnson & Johnson","PFE":"Pfizer Inc.","MRNA":"Moderna Inc.",
    "DIS":"Walt Disney Co.","SPOT":"Spotify Technology","SNAP":"Snap Inc.",
    "SHOP":"Shopify Inc.","SQ":"Block Inc.","COIN":"Coinbase Global",
    "PLTR":"Palantir Technologies","ABNB":"Airbnb Inc.","ZM":"Zoom Video",
    "ARKK":"ARK Innovation ETF","SPY":"S&P 500 ETF","QQQ":"Nasdaq-100 ETF",
    "2222.SR":"Saudi Aramco","9988.HK":"Alibaba HK","7203.T":"Toyota Motor",
    "005930.KS":"Samsung Electronics","RELIANCE.NS":"Reliance Industries",
    "TCS.NS":"Tata Consultancy","INFY.NS":"Infosys Ltd.",
    "XOM":"ExxonMobil Corp.","CVX":"Chevron Corp.","BP":"BP plc",
    "NKE":"Nike Inc.","MCD":"McDonald's Corp.","SBUX":"Starbucks Corp.",
    "WMT":"Walmart Inc.","COST":"Costco Wholesale","TGT":"Target Corp.",
    "BA":"Boeing Co.","LMT":"Lockheed Martin","GE":"GE Aerospace",
    "GOOG":"Alphabet Class C","BRK-B":"Berkshire Hathaway B",
}

@st.cache_data(ttl=3600)
def search_tickers(query):
    """Search local POPULAR_TICKERS dict first, then fall back to yfinance Ticker info."""
    q = query.strip().upper()
    results = []
    # 1. Exact match first
    if q in POPULAR_TICKERS:
        results.append(f"{q} — {POPULAR_TICKERS[q]}")
    # 2. Partial match in local dict
    ql = query.strip().lower()
    for sym, name in POPULAR_TICKERS.items():
        if sym != q and (ql in name.lower() or ql in sym.lower()):
            results.append(f"{sym} — {name}")
    # 3. If nothing found locally AND query looks like a ticker, probe yfinance
    if not results and len(q) >= 1:
        try:
            info = yf.Ticker(q).get_info() or {}
            long_name = info.get("longName") or info.get("shortName")
            if long_name:
                results.append(f"{q} — {long_name}")
        except Exception:
            pass
    return results[:10]

@st.cache_data(ttl=300)
def fetch_data(ticker, start, end):
    # Preserve suffix for international tickers (e.g. RELIANCE.NS, 7203.T)
    parts = ticker.strip().split(".")
    ticker = parts[0].upper() + ("." + parts[1].upper() if len(parts) > 1 else "")
    df = av_get_daily(ticker)
    if df.empty:
        return df
    # Filter by date range
    _start_ts = pd.to_datetime(start).tz_localize(None) if pd.to_datetime(start).tzinfo is not None else pd.to_datetime(start)
    _end_ts   = pd.to_datetime(end).tz_localize(None)   if pd.to_datetime(end).tzinfo   is not None else pd.to_datetime(end)
    _idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df = df[(_idx >= _start_ts) & (_idx <= _end_ts)]
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean().replace(0, 1e-10)  # avoid division by zero
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
    return sma + std*rs, sma, sma - std*rs

def add_technical_features(df):
    close  = df['Close'].squeeze()
    high   = df['High'].squeeze()
    low    = df['Low'].squeeze()
    volume = df['Volume'].squeeze()
    df['MA5']   = close.rolling(5).mean()
    df['MA10']  = close.rolling(10).mean()
    df['MA20']  = close.rolling(20).mean()
    df['MA50']  = close.rolling(50).mean()
    df['MA200'] = close.rolling(200).mean()
    df['EMA12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA26'] = close.ewm(span=26, adjust=False).mean()
    df['RSI']   = compute_rsi(close)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(close)
    df['BB_Upper'], df['BB_Mid'], df['BB_Lower']   = compute_bollinger_bands(close)
    df['BB_Width']       = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    df['BB_Pct']         = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['Returns']        = close.pct_change()
    df['Returns_5d']     = close.pct_change(5)
    df['Volatility']     = df['Returns'].rolling(20).std()
    df['Momentum']       = close - close.shift(10)
    df['Volume_MA10']    = volume.rolling(10).mean()
    df['Volume_Ratio']   = volume / df['Volume_MA10']
    df['High_Low_Pct']   = (high - low) / close
    df['Close_Open_Pct'] = (close - df['Open'].squeeze()) / df['Open'].squeeze()
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # ── Extra indicators ──────────────────────────────────────────────────────
    # Stochastic %K / %D (14,3)
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df['Stoch_K'] = (close - low14) / (high14 - low14 + 1e-10) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # On-Balance Volume (OBV)
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df['OBV'] = obv
    df['OBV_EMA'] = obv.ewm(span=20, adjust=False).mean()

    # Williams %R (14)
    df['Williams_R'] = (high14 - close) / (high14 - low14 + 1e-10) * -100

    # Commodity Channel Index (CCI 20)
    tp = (high + low + close) / 3
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std().replace(0, 1e-10))

    # Average Directional Index (ADX 14) — simplified
    plus_dm  = (high.diff().clip(lower=0)).where(high.diff() > (-low.diff()), 0)
    minus_dm = (-low.diff().clip(upper=0)).where((-low.diff()) > high.diff(), 0)
    atr14    = tr.rolling(14).mean().replace(0, 1e-10)
    plus_di  = 100 * plus_dm.rolling(14).mean()  / atr14
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14
    dx       = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    df['ADX']      = dx.rolling(14).mean()
    df['Plus_DI']  = plus_di
    df['Minus_DI'] = minus_di

    return df

FEATURE_COLS = [
    'MA5','MA10','MA20','MA50','MA200','EMA12','EMA26',
    'RSI','MACD','MACD_Signal','MACD_Hist',
    'BB_Width','BB_Pct','Returns','Returns_5d','Volatility','Momentum',
    'Volume_Ratio','High_Low_Pct','Close_Open_Pct','ATR',
    # ── New indicators ──────────────────────────────────
    'Stoch_K','Stoch_D','OBV','OBV_EMA','Williams_R','CCI','ADX','Plus_DI','Minus_DI',
]

def build_xgb_dataset(df, seq_len):
    close   = df['Close'].squeeze().values
    feat_df = df[FEATURE_COLS].copy()
    feat_df['Close'] = close
    X_rows, y_rows = [], []
    for i in range(seq_len, len(feat_df) - 1):
        row_feats  = feat_df[FEATURE_COLS].iloc[i].values
        lag_closes = close[i - seq_len:i]
        X_rows.append(np.concatenate([row_feats, lag_closes]))
        y_rows.append(close[i + 1])
    X = np.array(X_rows)
    y = np.array(y_rows)
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    dropped = (~mask).sum()
    if dropped > 0:
        warnings.warn(f"build_xgb_dataset: dropped {dropped} rows containing NaN (out of {len(mask)} total)")
    return X[mask], y[mask]

def compute_composite_signal(df, last_close, forecast_price, preds, actual):
    close = df['Close'].squeeze()
    rsi   = float(df['RSI'].squeeze().iloc[-1])
    macd  = float(df['MACD'].squeeze().iloc[-1])
    macd_s= float(df['MACD_Signal'].squeeze().iloc[-1])
    macd_h= float(df['MACD_Hist'].squeeze().iloc[-1])
    bb_pct= float(df['BB_Pct'].squeeze().iloc[-1])
    ma50  = float(df['MA50'].squeeze().iloc[-1])
    ma200 = float(df['MA200'].squeeze().iloc[-1])
    vol_r = float(df['Volume_Ratio'].squeeze().iloc[-1])
    mom   = float(df['Momentum'].squeeze().iloc[-1])
    atr   = float(df['ATR'].squeeze().iloc[-1])

    # New indicators (safe access — may be absent on short histories)
    stoch_k  = float(df['Stoch_K'].squeeze().iloc[-1])  if 'Stoch_K'   in df.columns else 50.0
    stoch_d  = float(df['Stoch_D'].squeeze().iloc[-1])  if 'Stoch_D'   in df.columns else 50.0
    will_r   = float(df['Williams_R'].squeeze().iloc[-1]) if 'Williams_R' in df.columns else -50.0
    adx      = float(df['ADX'].squeeze().iloc[-1])       if 'ADX'       in df.columns else 25.0
    plus_di  = float(df['Plus_DI'].squeeze().iloc[-1])   if 'Plus_DI'   in df.columns else 25.0
    minus_di = float(df['Minus_DI'].squeeze().iloc[-1])  if 'Minus_DI'  in df.columns else 25.0

    signals = {}
    xgb_pct = (forecast_price - last_close) / last_close * 100

    # 1. AI Outlook
    if   xgb_pct >  1.5: signals['AI Outlook']    = ('BUY',  min(35, abs(xgb_pct)*6), xgb_pct, 'positive')
    elif xgb_pct < -1.5: signals['AI Outlook']    = ('SELL', -min(35, abs(xgb_pct)*6), xgb_pct, 'negative')
    else:                 signals['AI Outlook']    = ('HOLD', 0, xgb_pct, 'neutral')

    # 2. RSI
    if   rsi < 30: signals['RSI (14)']   = ('BUY',  20, rsi, 'positive')
    elif rsi > 70: signals['RSI (14)']   = ('SELL', -20, rsi, 'negative')
    elif rsi < 45: signals['RSI (14)']   = ('BUY',   8, rsi, 'positive')
    elif rsi > 55: signals['RSI (14)']   = ('SELL',  -8, rsi, 'negative')
    else:          signals['RSI (14)']   = ('HOLD',   0, rsi, 'neutral')

    # 3. MACD Cross
    prev_hist = float(df['MACD_Hist'].squeeze().iloc[-2]) if len(df) > 2 else 0
    if   macd_h > 0 and prev_hist <= 0: signals['MACD Cross'] = ('BUY',  20, macd_h, 'positive')
    elif macd_h < 0 and prev_hist >= 0: signals['MACD Cross'] = ('SELL', -20, macd_h, 'negative')
    elif macd > macd_s:                 signals['MACD Cross'] = ('BUY',  10, macd_h, 'positive')
    elif macd < macd_s:                 signals['MACD Cross'] = ('SELL', -10, macd_h, 'negative')
    else:                               signals['MACD Cross'] = ('HOLD',  0, macd_h, 'neutral')

    # 4. Bollinger %B
    if   bb_pct < 0.1: signals['Bollinger %B'] = ('BUY',  10, bb_pct, 'positive')
    elif bb_pct > 0.9: signals['Bollinger %B'] = ('SELL', -10, bb_pct, 'negative')
    else:              signals['Bollinger %B'] = ('HOLD',   0, bb_pct, 'neutral')

    # 5. MA Cross (Golden/Death)
    if   ma50 > ma200 and close.iloc[-1] > ma50: signals['MA Cross'] = ('BUY',  15, ma50-ma200, 'positive')
    elif ma50 < ma200 and close.iloc[-1] < ma50: signals['MA Cross'] = ('SELL', -15, ma50-ma200, 'negative')
    else:                                         signals['MA Cross'] = ('HOLD',  0, ma50-ma200, 'neutral')

    # 6. Volume Confirmation
    if   vol_r > 1.5 and xgb_pct > 0: signals['Volume'] = ('BUY',  10, vol_r, 'positive')
    elif vol_r > 1.5 and xgb_pct < 0: signals['Volume'] = ('SELL', -10, vol_r, 'negative')
    else:                              signals['Volume'] = ('HOLD',   0, vol_r, 'neutral')

    # 7. Stochastic %K/%D  ← NEW
    if   stoch_k < 20 and stoch_k > stoch_d: signals['Stochastic'] = ('BUY',  12, stoch_k, 'positive')
    elif stoch_k > 80 and stoch_k < stoch_d: signals['Stochastic'] = ('SELL', -12, stoch_k, 'negative')
    elif stoch_k < 30:                        signals['Stochastic'] = ('BUY',   6, stoch_k, 'positive')
    elif stoch_k > 70:                        signals['Stochastic'] = ('SELL',  -6, stoch_k, 'negative')
    else:                                     signals['Stochastic'] = ('HOLD',   0, stoch_k, 'neutral')

    # 8. Williams %R  ← NEW
    if   will_r < -80: signals['Williams %R'] = ('BUY',  10, will_r, 'positive')
    elif will_r > -20: signals['Williams %R'] = ('SELL', -10, will_r, 'negative')
    else:              signals['Williams %R'] = ('HOLD',   0, will_r, 'neutral')

    # 9. ADX trend strength  ← NEW
    if adx > 25:
        if plus_di > minus_di: signals['ADX Trend'] = ('BUY',  8, adx, 'positive')
        else:                  signals['ADX Trend'] = ('SELL', -8, adx, 'negative')
    else:
        signals['ADX Trend'] = ('HOLD', 0, adx, 'neutral')

    total_score = sum(s[1] for s in signals.values())
    if   total_score >= 25: verdict = "⬆ STRONG BUY";   verdict_short = "BUY"
    elif total_score >= 10: verdict = "↑ BUY";           verdict_short = "BUY"
    elif total_score <= -25:verdict = "⬇ STRONG SELL";  verdict_short = "SELL"
    elif total_score <= -10:verdict = "↓ SELL";          verdict_short = "SELL"
    else:                   verdict = "◆ HOLD";          verdict_short = "HOLD"

    # ATR-scaled TP/SL — 1.5× ATR stop, 2.5× ATR target for better R:R discipline
    volatility_mult = 1.0 + min(0.5, float(df['Volatility'].squeeze().dropna().iloc[-1]) * 10)
    stop_loss   = last_close - 1.5 * atr * volatility_mult
    take_profit = last_close + 2.5 * atr * volatility_mult
    risk_reward = (take_profit - last_close) / max(last_close - stop_loss, 0.01)

    return {
        'signals': signals, 'verdict': verdict, 'verdict_short': verdict_short,
        'total_score': total_score, 'xgb_pct': xgb_pct, 'rsi': rsi,
        'stop_loss': stop_loss, 'take_profit': take_profit, 'risk_reward': risk_reward,
        'vol_ratio': vol_r, 'atr': atr,
        'stoch_k': stoch_k, 'stoch_d': stoch_d,
        'williams_r': will_r, 'adx': adx,
    }

def run_backtest_engine(actual_prices, predicted_prices, initial_capital, commission, threshold_pct):
    capital = float(initial_capital); position = 0; entry_price = 0.0
    trades = []; equity = []
    for i in range(len(predicted_prices) - 1):
        price_now  = float(actual_prices[i])
        pred_next  = float(predicted_prices[i])
        diff_pct   = (pred_next - price_now) / price_now * 100
        equity.append(capital + position * price_now)
        if diff_pct > threshold_pct and position == 0:
            shares = int((capital - commission) / price_now)
            if shares > 0:
                capital -= shares * price_now + commission
                position = shares; entry_price = price_now
                trades.append({"Day":i,"Type":"BUY","Price":price_now,"Shares":shares,"Capital":capital})
            # shares==0 means the stock price exceeds available capital — silently skip
        elif diff_pct < -threshold_pct and position > 0:
            proceeds = position * price_now - commission
            pnl = proceeds - (entry_price * position + commission)
            capital += proceeds
            trades.append({"Day":i,"Type":"SELL","Price":price_now,"Shares":position,"P&L":pnl,"Capital":capital})
            position = 0; entry_price = 0.0
    if position > 0:
        fp = float(actual_prices[-1]); proceeds = position*fp - commission
        pnl = proceeds - (entry_price*position + commission); capital += proceeds
        trades.append({"Day":len(actual_prices)-1,"Type":"SELL (EOD)","Price":fp,"Shares":position,"P&L":pnl,"Capital":capital})
    equity.append(capital)
    bh_shares  = int((initial_capital - commission) / float(actual_prices[0]))
    bh_final   = bh_shares * float(actual_prices[-1]) - commission
    bh_return  = (bh_final - initial_capital) / initial_capital * 100
    strat_return = (capital - initial_capital) / initial_capital * 100
    equity_s   = pd.Series(equity)
    drawdown   = equity_s / equity_s.cummax() - 1
    daily_r    = equity_s.pct_change().dropna()
    sharpe     = (daily_r.mean() / daily_r.std() * np.sqrt(252)) if daily_r.std() > 0 else 0.0
    trades_df  = pd.DataFrame(trades)
    if not trades_df.empty and "P&L" in trades_df.columns:
        closed = trades_df[trades_df["Type"].str.contains("SELL")]
        win_trades = (closed["P&L"] > 0).sum(); loss_trades = (closed["P&L"] <= 0).sum()
        win_rate  = win_trades / len(closed) * 100 if len(closed) > 0 else 0.0
        avg_win   = closed[closed["P&L"] > 0]["P&L"].mean()  if win_trades  > 0 else 0.0
        avg_loss  = closed[closed["P&L"] <= 0]["P&L"].mean() if loss_trades > 0 else 0.0
        pf = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        total_trades = len(closed)
    else:
        win_rate = avg_win = avg_loss = pf = 0.0; total_trades = 0
    bh_equity = [initial_capital * (float(actual_prices[i]) / float(actual_prices[0])) for i in range(len(actual_prices))]
    return {"final_capital":capital,"strat_return":strat_return,"bh_return":bh_return,
            "max_drawdown":float(drawdown.min()*100),"sharpe":sharpe,"win_rate":win_rate,
            "total_trades":total_trades,"avg_win":avg_win,"avg_loss":avg_loss,"profit_factor":pf,
            "equity_curve":equity,"bh_equity":bh_equity,"trades_df":trades_df,"drawdown_series":drawdown.tolist()}

def bootstrap_confidence_intervals(model, X_input, n_bootstrap=100, noise_std=None):
    # Adapt noise to asset's recent volatility if not explicitly provided.
    # The last feature column index for 'Volatility' is position 14 in FEATURE_COLS.
    if noise_std is None:
        try:
            vol_idx = FEATURE_COLS.index('Volatility')
            recent_vol = float(np.nanmedian(X_input[-20:, vol_idx]))
            noise_std = max(0.005, min(0.05, recent_vol))  # clamp between 0.5% and 5%
        except Exception as e:
            logger.warning("bootstrap_confidence_intervals: could not auto-detect volatility, using default noise_std=0.02: %s", e)
            noise_std = 0.02
    # Use relative noise scaled to each feature's std, so a $500 stock
    # and a $5 stock both get proportionally equivalent perturbations.
    feature_scale = np.std(X_input, axis=0, keepdims=True)
    feature_scale = np.where(feature_scale == 0, 1.0, feature_scale)
    all_preds = [
        model.predict(X_input + np.random.normal(0, noise_std, X_input.shape) * feature_scale)
        for _ in range(n_bootstrap)
    ]
    a = np.array(all_preds)
    return np.percentile(a, 5, axis=0), np.percentile(a, 50, axis=0), np.percentile(a, 95, axis=0)

HARAM_TICKERS = {
    "BUD","STZ","SAM","BREW","ABEV","DEO","BF-B",       # Alcohol
    "MO","PM","BTI","LO","VGR",                          # Tobacco
    "LVS","MGM","WYNN","CZR","PENN","DKNG","BYD",        # Gambling
    "MET","PRU","AIG","ALL","TRV","CB",                   # Pure insurance
    "HRL","TSN","SFD","CAG","LMT","RTX","NOC","GD","HII", # Pork/Defense
}
# Per AAOIFI Standard No.21: banks and financial firms require ratio screening,
# not a blanket ban. Moved here so the app performs the debt/assets check.
QUESTIONABLE_TICKERS = {
    "DIS","NFLX","PARA","WBD","FOXA","SPOT",
    "MAR","HLT","H","IHG","WH",
    "JPM","BAC","WFC","C","GS","MS",   # Banks — require ratio screening
    "V","MA","AXP","COF","USB","PNC",  # Payment/finance — require ratio screening
}
HARAM_SECTORS_KW = ["bank","insurance","casino","gambling","alcohol","tobacco",
                    "brewing","distill","porn","adult","weapons","defense","firearm"]

@st.cache_data(ttl=3600)
def get_shariah_data(ticker_sym):
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
        "debt_to_mktcap": td/mc,
        "debt_to_assets": td/ta,
        "cash_to_assets": tc/ta,
        "market_cap": mc,
        "total_debt": td,
        "total_assets": ta,
        "total_cash": tc,
        "sector":   info.get("Sector","Unknown"),
        "industry": info.get("Industry","Unknown"),
        "company_name": info.get("Name", ticker_sym)
    }

def check_shariah_compliance(ticker_sym, data, _L=None):
    if _L is None:
        _L = {}
    t = ticker_sym.upper(); ind_lower = data["industry"].lower(); haram_hit = None
    if t in HARAM_TICKERS: haram_hit = _L.get("known_noncompliant", "Known non-compliant ticker")
    else:
        for kw in HARAM_SECTORS_KW:
            if kw in ind_lower: haram_hit = data["industry"]; break
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
    all_pass = all(r[k]["pass"] for k in ["business","debt_mktcap","debt_assets","cash_assets"])
    r["verdict"] = "NON-COMPLIANT" if not r["business"]["pass"] or not all_pass else ("QUESTIONABLE" if questionable else "COMPLIANT")
    return r

def render_methodology_page(seq_len_val=30, ci_n=100, show_ci=True):
    st.markdown(f"""
    <div style="font-family:Manrope,sans-serif;font-size:0.6rem;letter-spacing:.18em; text-transform:uppercase;color:#8a8fa0;margin-bottom:.3rem;font-weight:700;">Technical Documentation</div>
    <div style="font-family:Manrope,sans-serif;font-size:1.15rem;font-weight:800; color:#e4eafd;letter-spacing:-.01em;margin-bottom:1.4rem;">
         Stockcast <span style="color:#4d8eff;">·</span> How the AI Assistant Works
    </div>
    """, unsafe_allow_html=True)
    _steps_data = [
        ("01","#4d8eff","Data Ingestion","OHLCV via yfinance","Data",
         "Up to 7 years of daily OHLCV data fetched from Yahoo Finance. Timezone normalization and MultiIndex flattening applied for cross-version compatibility.",
         [("chip info","yfinance"),("chip info","OHLCV"),("chip info","7Y History")]),
        ("02","#adc6ff","Feature Engineering","20 Technical Indicators","Features",
         f"MA5/10/20/50/200, EMA12/26, RSI(14), MACD(12/26/9) with histogram, Bollinger Band width &amp; %B, ATR(14), Volume Ratio, Momentum, Returns(1d/5d), Volatility(20d), High-Low%. Plus {seq_len_val} lag closes as sequential memory.",
         [("chip ai","RSI"),("chip ai","MACD"),("chip ai","Bollinger"),("chip info","20 Signals")]),
        ("03","#00e5b0","Train / Test Split","80% train · 20% test (chronological)","Split",
         "Strictly chronological split — no shuffling — to prevent look-ahead bias. The model never sees future data during training.",
         [("chip buy","80% Train"),("chip warn","20% Test"),("chip info","No Leakage")]),
        ("04","#4d8eff","XGBoost Engine","Gradient-boosted decision trees","Model",
         "XGBoost trained to project next day's closing price. Hyperparameters (n_estimators, max_depth, learning_rate) configurable via sidebar. Subsample=0.8, colsample_bytree=0.8.",
         [("chip ai","XGBoost"),("chip info","Configurable HP"),("chip ai","Regularised")]),
        ("05","#adc6ff","Bootstrap CI",f"{ci_n} resampling iterations" if show_ci else "Disabled","Uncertainty",
         f"Model run {ci_n}× on inputs perturbed with Gaussian noise (σ=1.5%). 5th–95th percentile forms 95% CI ribbon. Wider band = higher uncertainty.",
         [("chip ai",f"{ci_n} Samples"),("chip info","95% CI"),("chip warn","σ=1.5%")]),
        ("06","#00e5b0","Price Outlook","Iterative multi-step projection","Forecast",
         "Each projected price feeds back as next day's lag input. Uncertainty compounds — Days 1–3 most reliable, Days 6+ directional guidance only.",
         [("chip buy","Day 1-3 Reliable"),("chip warn","Day 6+ Directional")]),
        ("07","#ff5f5f","Signal Generation","BUY / SELL / HOLD research signal","Signal",
         "Composite 6-factor signal from AI outlook, RSI, MACD crossover, Bollinger %B, MA Golden/Death cross, and Volume confirmation. Score &gt;+25 = STRONG BUY, &lt;-25 = STRONG SELL.",
         [("chip buy","BUY"),("chip sell","SELL"),("chip hold","HOLD"),("chip info","6 Factors")]),
        ("08","#4d8eff","Strategy Simulator","Walk-forward simulation","Backtest",
         "Replays AI signals on test-set prices. KPIs: Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor, equity curve vs Buy-and-Hold.",
         [("chip warn","Sharpe"),("chip warn","Drawdown"),("chip info","Win Rate")]),
    ]
    _acc_html = '<div class="accordion">'
    for _sn, _sc, _st, _ss, _sb, _body, _chips in _steps_data:
        _chips_html = " ".join(f'<span class="{cc}">{cl}</span>' for cc, cl in _chips)
        _acc_html += (
            f'<div class="accordion-item">' +
            f'<button class="accordion-trigger">' +
            f'<span class="accordion-num">{_sn}</span>' +
            f'<span class="accordion-label">{_st} ' +
            f'<span style="font-family:var(--mono);font-size:.7rem;color:{_sc};margin-left:.5rem;">{_ss}</span></span>' +
            f'<span class="accordion-badge">{_sb}</span>' +
            f'<span class="accordion-icon">&#9662;</span></button>' +
            f'<div class="accordion-body"><div class="accordion-body-inner">' +
            f'<div style="font-family:Manrope,sans-serif;font-size:.83rem;color:#8a8fa0;line-height:1.65;margin-bottom:.7rem;">{_body}</div>' +
            f'<div class="chip-group">{_chips_html}</div>' +
            f'</div></div></div>'
        )
    _acc_html += "</div>"
    st.markdown(_acc_html, unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(255,107,107,0.04);border:1px solid rgba(255,107,107,0.2); border-left:3px solid #ff5f5f;padding:1rem 1.5rem;margin-top:.5rem;border-radius:0 0.5rem 0.5rem 0;">
      <div style="font-family:Manrope,sans-serif;font-size:0.63rem;letter-spacing:.14em; text-transform:uppercase;color:#ff5f5f;margin-bottom:.4rem;font-weight:700;">⚠ Key Limitations</div>
      <div style="font-family:Manrope,sans-serif;font-size:0.82rem;color:#8a8fa0;line-height:1.7;">
        This assistant analyses price and volume patterns. It works best when combined with your own
        market knowledge, recent news, and broader context.
        A single unexpected event can shift any technical outlook.
        <b style="color:#ff5f5f;">This is a research and educational tool — not financial advice.</b>
        Always consult a licensed financial advisor before making investment decisions.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Auth Gate ─────────────────────────────────────────────────────────────────
# UI lives in authgate.py — edit that file to change the login/signup design.
from authgate import render_auth_gate
render_auth_gate(supabase)

# v3: session analytics
if not st.session_state.get('_sess_tracked'):
    track_event('session_start', {'plan': st.session_state.get('plan', 'free')})
    st.session_state['_sess_tracked'] = True

# Everything below only runs once the user is authenticated.
if st.session_state.user is None:  # fallback guard (render_auth_gate calls st.stop())
    st.stop()

# ── Demo mode notice ──────────────────────────────────────────────────────────
if st.session_state.get("_is_demo"):
    st.markdown("""
    <div style="background:rgba(212,168,83,0.07);border:1px solid rgba(212,168,83,0.22);
      border-left:3px solid #d4a853;border-radius:0 .5rem .5rem 0;
      padding:.55rem 1.1rem;margin-bottom:.6rem;
      font-family:Manrope,sans-serif;font-size:.74rem;color:#d4a853;
      display:flex;align-items:center;gap:.6rem;">
        👁 <b>Demo Mode</b> — you're exploring a read-only preview.
        Portfolio &amp; watchlist changes won't be saved.
        <a href="?demo=0" style="color:#4d8eff;margin-left:auto;text-decoration:none;font-size:.7rem;">
        Sign up for full access →</a>
    </div>
    """, unsafe_allow_html=True)


# ── Safe user attribute helpers ───────────────────────────────────────────────
# supabase-py ≥2.0 may return a Session object (user at .user.email)
# or a User object (.email directly). These handle both safely.
def _user_email() -> str:
    u = st.session_state.user
    if u is None: return ""
    if hasattr(u, "user") and u.user is not None:
        return getattr(u.user, "email", "") or ""
    return getattr(u, "email", "") or ""

def _user_id() -> str:
    u = st.session_state.user
    if u is None: return ""
    if hasattr(u, "user") and u.user is not None:
        return str(getattr(u.user, "id", "")) or ""
    return str(getattr(u, "id", "")) or ""


# ── Load user data from Supabase once per login session ──────────────────────
_current_uid = _user_id() if st.session_state.user else None
if _current_uid and st.session_state.get("_portfolio_loaded_for") != _current_uid:
    _loaded = _sb_load_portfolio(_current_uid)
    st.session_state.portfolio = _loaded
    _loaded_hist = _sb_load_history(_current_uid)
    st.session_state.portfolio_history = _loaded_hist
    # Load watchlist from Supabase — deduplicate just in case
    _wl_raw = _sb_load_watchlist(_current_uid)
    st.session_state.watchlist = list(dict.fromkeys(_wl_raw))  # preserves order, removes dupes
    # Load usage count + plan
    _usage = _sb_get_usage(_current_uid)
    st.session_state.usage_count = _usage.get("usage_count", 0)
    st.session_state.analyses_today = _usage.get("usage_count", 0)
    st.session_state.user_plan = _usage.get("plan", "free")
    st.session_state.email_alerts_enabled = bool(_usage.get("email_alerts_enabled", False))
    _wl = st.session_state.watchlist
    # Show onboarding only for brand-new users (empty watchlist + no usage)
    st.session_state.show_onboarding = (len(_wl) == 0 and st.session_state.usage_count == 0)
    st.session_state._portfolio_loaded_for = _current_uid

    # ── Daily email digest — send once per calendar day on first login ─────────
    _today_str = pd.Timestamp.now().date().isoformat()
    _digest_sent_key = f"_digest_sent_{_current_uid}_{_today_str}"
    if (
        st.session_state.get("email_alerts_enabled", False)
        and not st.session_state.get(_digest_sent_key, False)
        and st.session_state.watchlist
    ):
        try:
            _d_html = _build_digest_html(_user_email(), st.session_state.watchlist)
            _d_sent = _send_email(
                _user_email(),
                f"📈 Stockcast Daily Digest — {pd.Timestamp.now().strftime('%b %d, %Y')}",
                _d_html,
            )
            if _d_sent:
                logger.info("Daily digest sent to %s", _user_email())
        except Exception as _de:
            logger.error("Daily digest send failed: %s", _de)
        finally:
            st.session_state[_digest_sent_key] = True  # mark as attempted regardless


# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
_pro_badge_html = (
    "<span style='font-family:Manrope,sans-serif;font-size:.6rem;font-weight:800;"
    "background:linear-gradient(90deg,#ffd426,#ffb300);-webkit-background-clip:text;"
    "-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:.08em;"
    "margin-left:.5rem;vertical-align:middle;'>PRO</span>"
) if _is_pro() else ""

st.markdown(f"""
<div class="wi-header">
  <div style="min-width:0;flex:1;">
    <div class="wi-logo">Stock<span>cast</span>{_pro_badge_html}</div>
    <div class="wi-sub">AI Stock Intelligence · Signals · Forecast · Shariah · NLP</div>
    <div class="trust-row">
      <span class="trust-item"><span class="trust-item-dot"></span>Yahoo Finance</span>
      <span class="trust-item"><span class="trust-item-dot" style="background:#4d8eff;"></span>Supabase Auth</span>
      <span class="trust-item"><span class="trust-item-dot" style="background:#ffcb2b;"></span>Educational Only</span>
    </div>
  </div>
  <div style="display:flex;flex-direction:column;align-items:flex-end;gap:5px;flex-shrink:0;margin-left:1rem;">
    <div style="display:flex;align-items:center;gap:5px;">
      <span class="live-dot"></span>
      <span class="live-label">LIVE</span>
    </div>
    <span class="disclaimer-pill">⚠ Not Financial Advice</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Upgrade Page (replaces broken overlay modal) ──────────────────────────────
if st.session_state.get("show_upgrade_modal"):
    # Full-page upgrade screen — stop rendering rest of app while shown
    st.markdown("""
    <style>
    /* Hide sidebar while on upgrade page */
    [data-testid="stSidebar"] { display: none !important; }
    .block-container { max-width: 680px !important; }
    </style>
    """, unsafe_allow_html=True)

    # Back button
    if st.button("← Back", key="upgrade_back"):
        st.session_state.show_upgrade_modal = False
        st.rerun()

    # Header
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1.5rem;">
      <div style="font-family:IBM Plex Mono,monospace;font-size:.62rem;letter-spacing:.2em; text-transform:uppercase;color:#ffd426;margin-bottom:.6rem;">Stockcast Pro</div>
      <div style="font-family:Manrope,sans-serif;font-size:2rem;font-weight:800; color:#e4eafd;letter-spacing:-.03em;line-height:1.2;margin-bottom:.5rem;">
        Unlock the full<br>AI Stock Assistant
      </div>
      <div style="font-family:Manrope,sans-serif;font-size:.9rem;color:#8a8fa0;line-height:1.6;">
        Everything in Free, plus the features serious investors need.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Free vs Pro comparison columns
    col_free, col_pro = st.columns(2)

    with col_free:
        st.markdown("""
        <div style="background:#0f1727;border:1px solid #252f47;border-radius:1rem; padding:1.4rem 1.5rem;height:100%;">
          <div style="font-family:Manrope,sans-serif;font-size:.6rem;font-weight:800; letter-spacing:.14em;text-transform:uppercase;color:#8a8fa0;margin-bottom:1rem;">
            Free — $0
          </div>
          <div style="font-family:Manrope,sans-serif;font-size:.82rem;color:#8a8fa0;line-height:2;">
            ✓ &nbsp;3 analyses / day<br>
            ✓ &nbsp;5 watchlist stocks<br>
            ✓ &nbsp;7-day price outlook<br>
            ✓ &nbsp;6-factor signals<br>
            ✓ &nbsp;Shariah screening<br>
            ✗ &nbsp;Prophet comparison<br>
            ✗ &nbsp;Confidence intervals<br>
            ✗ &nbsp;Multi-stock compare<br>
            ✗ &nbsp;30-day outlook
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_pro:
        st.markdown("""
        <div style="background:linear-gradient(145deg,rgba(255,212,38,0.08),#0a0f1e); border:2px solid rgba(255,212,38,0.4);border-radius:1rem; padding:1.4rem 1.5rem;height:100%;position:relative;">
          <div style="position:absolute;top:-1px;right:1.2rem;background:#ffd426; color:#080e1c;font-family:Manrope,sans-serif;font-size:.7rem; font-weight:800;letter-spacing:.08em;text-transform:uppercase; padding:.22rem .75rem;border-radius:0 0 .5rem .5rem;">BEST VALUE</div>
          <div style="font-family:Manrope,sans-serif;font-size:.6rem;font-weight:800; letter-spacing:.14em;text-transform:uppercase;color:#ffd426;margin-bottom:1rem;">
            Pro — $9/mo
          </div>
          <div style="font-family:Manrope,sans-serif;font-size:.82rem;color:#e4eafd;line-height:2;">
            ✦ &nbsp;Unlimited analyses<br>
            ✦ &nbsp;50 watchlist stocks<br>
            ✦ &nbsp;30-day price outlook<br>
            ✦ &nbsp;All signal types<br>
            ✦ &nbsp;Shariah screening<br>
            ✦ &nbsp;Prophet + XGBoost<br>
            ✦ &nbsp;Bootstrap CI bands<br>
            ✦ &nbsp;Multi-stock compare<br>
            ✦ &nbsp;Priority speed
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Price + CTA
    st.markdown("""
    <div style="text-align:center;margin-bottom:1rem;">
      <div style="font-family:IBM Plex Mono,monospace;font-size:2.8rem;font-weight:700;color:#ffd426;">
        $9<span style="font-size:1.1rem;color:#8a8fa0;font-weight:400;">/month</span>
      </div>
      <div style="font-family:Manrope,sans-serif;font-size:.75rem;color:#3e4558;margin-top:.3rem;">
        Cancel anytime · No contracts · Payments via Razorpay (coming soon)
      </div>
    </div>
    """, unsafe_allow_html=True)

    btn_l, btn_c, btn_r = st.columns([1, 2, 1])
    with btn_c:
        if st.button("✦ Activate Pro — $9/mo", use_container_width=True, key="upgrade_confirm"):
            # ── Payment gate ─────────────────────────────────────────────────
            # TODO: Replace this block with your Razorpay/Stripe payment flow.
            # Once payment is confirmed by the provider webhook, call:
            #   _sb_set_plan(user_id, "pro")
            # For now, grant access immediately (testing only — remove before launch).
            _sb_set_plan(_user_id(), "pro")
            st.session_state.show_upgrade_modal = False
            st.session_state._portfolio_loaded_for = None
            st.rerun()

    st.markdown("""
    <div style="text-align:center;margin-top:1rem;font-family:Manrope,sans-serif; font-size:.65rem;color:#3e4558;line-height:1.7;">
      ⚠ Payments not yet live — Razorpay / Stripe integration coming soon.<br>
      Upgrade grants immediate access for testing purposes only.
    </div>
    """, unsafe_allow_html=True)

    # FAQ
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Frequently asked questions"):
        st.markdown("""
        **What happens when I upgrade?**
        Your plan switches to Pro immediately. All limits are removed and Pro features unlock without restarting.

        **Is payment active?**
        Not yet — this is a simulated upgrade for testing. Real payments via Razorpay/Stripe are coming soon.

        **Can I go back to Free?**
        Yes — contact support and we will process your cancellation. Your data (watchlist, portfolio, history) is always preserved.

        **Does my data carry over?**
        Yes. Watchlist, portfolio, and history are all stored in Supabase and persist across plan changes.
        """)

    st.stop()  # Don't render the rest of the app while upgrade page is shown

# Ticker tape — live prices
_tape_items = get_live_ticker_tape()
if _tape_items:
    _dot = '<span style="color:#252f47;">·</span>'
    _tape_spans = f" {_dot} ".join(
        f'<span><span class="tape-sym">{sym}</span>'
        f'<span class="{css}">{arrow} {price} {pct}</span></span>'
        for sym, price, pct, arrow, css in _tape_items * 2  # duplicate for seamless scroll
    )
    st.markdown(f"""
<div class="ticker-tape-wrap">
  <div class="ticker-tape">{_tape_spans}</div>
</div>
""", unsafe_allow_html=True)
else:
    # Fallback: show symbols only if live fetch fails
    st.markdown("""
<div class="ticker-tape-wrap">
  <div class="ticker-tape">
    <span class="tape-sym">AAPL</span> · <span class="tape-sym">TSLA</span> ·
    <span class="tape-sym">NVDA</span> · <span class="tape-sym">MSFT</span> ·
    <span class="tape-sym">GOOGL</span> · <span class="tape-sym">META</span> ·
    <span class="tape-sym">AMZN</span> · <span class="tape-sym">SPY</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    # Language selector
    _lang_choice = st.selectbox("🌐 Language", list(LANGUAGES.keys()),
                                index=list(LANGUAGES.keys()).index(st.session_state.lang),
                                key="lang_selector", label_visibility="collapsed")
    if _lang_choice != st.session_state.lang:
        st.session_state.lang = _lang_choice
        st.rerun()
    _L = LANGUAGES[st.session_state.lang]

    # Logo + User
    _usage_count  = st.session_state.get("usage_count", 0)
    _is_pro_user  = _is_pro()
    _daily_limit  = _get_limit("daily_analyses")
    _wl_limit     = _get_limit("watchlist_stocks")
    _usage_pct    = 0 if _is_pro_user else min(100, int(_usage_count / _daily_limit * 100))
    _usage_color  = "#ffd426" if _is_pro_user else (
        "#ff5f5f" if _usage_count >= _daily_limit else
        "#4d8eff" if _usage_count >= 2 else "#00e5b0"
    )

    st.markdown(f"""
    <div style="padding:1.3rem 0.85rem 0.7rem;">
      <div style="font-family:Manrope,sans-serif;font-size:1.3rem;font-weight:800; color:#eaefff;letter-spacing:-.025em;line-height:1;">
        Stock<span style="color:#4d8eff;">cast</span>
      </div>
      <div style="font-size:.7rem;color:#3d4760;letter-spacing:.08em;text-transform:uppercase; font-weight:700;margin-top:4px;">AI Stock Intelligence</div>
    </div>
    <div style="background:rgba(77,142,255,0.05);border:1px solid rgba(77,142,255,0.13); border-left:2px solid #4d8eff;padding:.42rem .85rem;margin:.15rem 0 .4rem; font-family:IBM Plex Mono,monospace;font-size:0.74rem;color:#7ab0ff; letter-spacing:.025em;border-radius:0 .45rem .45rem 0; display:flex;align-items:center;gap:.4rem;overflow:hidden;">
      <span style="color:#3d4760;flex-shrink:0;font-size:.8rem;">👤</span>
      <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#7ab0ff;">
        {_user_email()}
      </span>
    </div>
    """, unsafe_allow_html=True)

    if _is_pro_user:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(255,203,43,0.1),rgba(255,160,0,0.06)); border:1px solid rgba(255,203,43,0.28);border-radius:.5rem;padding:.6rem .9rem; margin:.2rem 0 .45rem;display:flex;align-items:center;justify-content:space-between;">
          <div>
            <div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:800; letter-spacing:.1em;text-transform:uppercase;color:#ffcb2b;">✦ Pro Plan</div>
            <div style="font-family:Manrope,sans-serif;font-size:.75rem;color:#7a8299;margin-top:2px;">
              Unlimited · {_wl_limit} watchlist stocks
            </div>
          </div>
          <div style="font-family:IBM Plex Mono,monospace;font-size:1.1rem;color:#ffcb2b;opacity:.8;">∞</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        _seg_html = ""
        for _si in range(_daily_limit):
            _seg_color = _usage_color if _si < _usage_count else "#1a2640"
            _seg_html += f'<div style="flex:1;height:5px;background:{_seg_color};border-radius:2px;transition:background .3s;"></div>'

        _limit_banner = ""
        if _usage_count >= _daily_limit:
            _limit_banner = f"""
            <div style="background:rgba(255,87,87,0.06);border:1px solid rgba(255,87,87,0.18); border-left:3px solid #ff5757;padding:.5rem .85rem;margin:.3rem 0; border-radius:0 .45rem .45rem 0;">
              <div style="font-family:Manrope,sans-serif;font-size:.76rem;font-weight:800; color:#ff5757;margin-bottom:.15rem;">🔒 Daily limit reached</div>
              <div style="font-family:Manrope,sans-serif;font-size:.73rem;color:#7a8299;line-height:1.4;">
                Resets at midnight UTC · Upgrade for unlimited.
              </div>
            </div>"""
        elif _usage_count > 0:
            _rem = _daily_limit - _usage_count
            _limit_banner = f'<div style="font-family:Manrope,sans-serif;font-size:.72rem;color:#7a8299;margin-top:.25rem;">{_rem} {"analysis" if _rem==1 else "analyses"} remaining today</div>'

        st.markdown(f"""
        <div class="plan-badge">
          <div style="flex:1;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
              <div class="plan-badge-label">Free Plan · Daily Usage</div>
              <div style="font-family:IBM Plex Mono,monospace;font-size:.72rem;color:{_usage_color};font-weight:700;">
                {_usage_count}/{_daily_limit}
              </div>
            </div>
            <div style="display:flex;gap:3px;margin-bottom:.25rem;">{_seg_html}</div>
          </div>
        </div>
        {_limit_banner}
        """, unsafe_allow_html=True)

    if st.button(_L.get("logout", "⏏  Logout"), use_container_width=True, key="logout_btn"):
        try:
            supabase.auth.sign_out()
        except Exception as e:
            logger.warning("Supabase sign_out failed (session already expired?): %s", e)
        st.session_state.user = None
        st.session_state.run_pressed = False
        st.session_state.portfolio = []
        st.session_state.portfolio_history = []
        st.session_state._portfolio_loaded_for = None
        st.rerun()

    st.markdown("---")

    # Ticker Search — respects watchlist click override
    _wl_override = st.session_state.get("load_ticker_from_watchlist")
    st.markdown(f'<div class="stat-row">{_L["search_label"]}</div>', unsafe_allow_html=True)
    search_query = st.text_input("Search", placeholder="e.g. Apple, TSLA, Saudi Aramco…",
                                 label_visibility="collapsed", key="search_input",
                                 value=_wl_override or "")
    # Clear watchlist override after consuming it
    if _wl_override:
        st.session_state.load_ticker_from_watchlist = None

    ticker = "AAPL"
    if search_query and len(search_query.strip()) >= 1:
        search_results = search_tickers(search_query.strip())
        if search_results:
            selected = st.selectbox("Select", search_results, label_visibility="collapsed")
            ticker   = selected.split(" — ")[0].strip()
            # FIX 4: Confirmed ticker — green validation badge
            st.markdown(f'<div style="background:rgba(0,217,166,0.07);border:1px solid rgba(0,217,166,0.25);border-left:3px solid #00d9a6;padding:.35rem .85rem;font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#00d9a6;letter-spacing:.04em;margin:.25rem 0;border-radius:0 .45rem .45rem 0;">✓ {ticker} — symbol verified</div>', unsafe_allow_html=True)
        else:
            ticker = search_query.strip().upper()
            # FIX 4: Unknown ticker — amber warning with hint
            st.markdown(f'<div style="background:rgba(255,203,43,0.06);border:1px solid rgba(255,203,43,0.25);border-left:3px solid #ffcb2b;padding:.38rem .85rem;font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#ffcb2b;letter-spacing:.04em;margin:.25rem 0;border-radius:0 .45rem .45rem 0;">⚠ {_L["verify_symbol"].format(ticker=ticker)}</div>', unsafe_allow_html=True)
            # Probe yfinance lightly to confirm symbol exists
            @st.cache_data(ttl=60)
            def _quick_validate(sym):
                try:
                    info = yf.Ticker(sym).fast_info
                    return getattr(info, "last_price", None) is not None or getattr(info, "quote_type", None) is not None
                except Exception:
                    return False
            _valid = _quick_validate(ticker)
            if not _valid and len(ticker) >= 1:
                st.markdown(f'<div style="font-family:Manrope,sans-serif;font-size:0.78rem;color:#ff5f5f;padding:.2rem 0 .3rem;">Symbol not found — check spelling. Indian stocks need .NS e.g. RELIANCE.NS</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="stat-row">{_L["ticker"]}</div>', unsafe_allow_html=True)
        ticker = st.text_input("Ticker", value=_wl_override or "AAPL", placeholder="AAPL, TSLA, MSFT…",
                               label_visibility="collapsed", key="direct_ticker").strip().upper() or "AAPL"
        st.markdown(f'<div style="background:rgba(77,142,255,0.08);border:1px solid rgba(77,142,255,0.2);border-left:3px solid #4d8eff;padding:.35rem .9rem;font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#4d8eff;letter-spacing:.07em;margin:.3rem 0;border-radius:0 .5rem .5rem 0;">{_L["active_ticker"].format(ticker=ticker)}</div>', unsafe_allow_html=True)

    # ⭐ Add to Watchlist inline button
    _in_wl = ticker in st.session_state.watchlist
    _wl_full = len(st.session_state.watchlist) >= _get_limit("watchlist_stocks")
    if _in_wl:
        st.markdown(f'<div style="font-family:Manrope,sans-serif;font-size:.63rem;color:#00e5b0;padding:.2rem 0;">⭐ {ticker} is in your watchlist</div>', unsafe_allow_html=True)
    elif _wl_full:
        st.markdown(f'<div style="font-family:Manrope,sans-serif;font-size:.73rem;color:#ff5757;padding:.2rem 0;">Watchlist full ({FREE_PLAN_WATCHLIST_LIMIT}/{FREE_PLAN_WATCHLIST_LIMIT})</div>', unsafe_allow_html=True)
    else:
        if st.button(f"⭐ Add {ticker} to Watchlist", use_container_width=True, key="sidebar_wl_add"):
            if ticker not in st.session_state.watchlist:
                if _sb_add_watchlist(_user_id(), ticker):
                    st.session_state.watchlist.append(ticker)
                    st.session_state["_wl_toast"] = ticker
                    st.rerun()

    col1, col2 = st.columns(2)
    with col1: start_date = st.date_input(_L["from"], value=pd.to_datetime("2018-01-01"))
    with col2: end_date   = st.date_input(_L["to"],   value=pd.Timestamp.today())

    st.markdown(f'<div class="stat-row">{_L["lookback"]}</div>', unsafe_allow_html=True)
    seq_len     = st.slider("Lookback window", 10, 60, 30, label_visibility="collapsed")
    st.markdown(f'<div class="stat-row">{_L["horizon"]}</div>', unsafe_allow_html=True)
    _max_horizon = _get_limit("forecast_horizon")
    future_days  = st.slider("Outlook horizon", 1, _max_horizon, min(7, _max_horizon), label_visibility="collapsed")
    if not _is_pro() and _max_horizon < 30:
        st.markdown(f'<div style="font-family:Manrope,sans-serif;font-size:.72rem;color:#3d4760;">🔒 Pro unlocks 30-day outlook</div>', unsafe_allow_html=True)

    st.markdown("---")
    ui_mode    = st.radio("Mode", [_L["beginner"], _L["pro"]], index=1, horizontal=True, label_visibility="collapsed")
    is_beginner = (ui_mode == _L["beginner"])
    if is_beginner:
        st.markdown(f'<div style="background:rgba(0,217,166,0.05);border-left:3px solid #00d9a6;padding:.38rem .85rem;font-family:Manrope,sans-serif;font-size:.73rem;color:#00d9a6;font-weight:700;">{_L["simple_view"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:rgba(255,87,87,0.05);border-left:3px solid #ff5757;padding:.38rem .85rem;font-family:Manrope,sans-serif;font-size:.73rem;color:#ff5757;font-weight:700;">{_L["pro_view"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    fast_mode = st.checkbox(_L["fast_mode"], value=is_beginner)

    if not is_beginner:
        st.markdown(f'<div class="stat-row">{_L["xgb_params"]}</div>', unsafe_allow_html=True)
        n_estimators  = st.slider("Trees", 100, 500, 200, step=50)
        max_depth     = st.slider("Max Depth", 2, 8, 4)
        learning_rate = st.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2], value=0.05)
    else:
        n_estimators = 200; max_depth = 4; learning_rate = 0.05

    st.markdown("---")
    st.markdown(f'<div class="stat-row">{_L["alert_target"]}</div>', unsafe_allow_html=True)
    # Restore per-ticker saved alert price
    _ap_key = f"alert_price_{ticker}"
    _ap_saved = st.session_state.get(_ap_key, 0.0)
    alert_price = st.number_input("Alert price", min_value=0.0, value=float(_ap_saved), step=1.0, label_visibility="collapsed", key="alert_price_input")
    # Save back so it persists when user switches tickers and returns
    if alert_price != _ap_saved:
        st.session_state[_ap_key] = alert_price
    if alert_price > 0:
        st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.73rem;color:#ffcb2b;margin-top:.2rem;">🔔 Target set: ${alert_price:.2f}</div>', unsafe_allow_html=True)

    if not is_beginner:
        st.markdown("---")
        st.markdown(f'<div class="stat-row">{_L["backtesting"]}</div>', unsafe_allow_html=True)
        run_backtest        = st.checkbox(_L["enable_backtest"], value=True)
        bt_initial_capital  = st.number_input(_L["init_capital"], min_value=1000, value=10000, step=1000)
        bt_commission       = st.number_input(_L["commission"], min_value=0.0, value=1.0, step=0.5)
        bt_signal_threshold = st.slider(_L["signal_thresh"], 0.5, 5.0, 1.0, step=0.5)
    else:
        run_backtest = False; bt_initial_capital = 10000; bt_commission = 1.0; bt_signal_threshold = 1.0

    if not is_beginner:
        st.markdown("---")
        st.markdown(f'<div class="stat-row">{_L["extra_features"]}</div>', unsafe_allow_html=True)
        if _is_pro():
            run_model_compare  = st.checkbox(_L["model_compare"], value=False)
            show_conf_interval = st.checkbox(_L["conf_interval"], value=True) and not fast_mode
            ci_bootstrap_n     = st.slider(_L["bootstrap_samples"], 50, 300, 100, step=50) if show_conf_interval else 100
        else:
            run_model_compare  = False
            show_conf_interval = False
            ci_bootstrap_n     = 100
            # FIX 3: Visible upgrade CTA instead of silent gating
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(255,203,43,0.06),rgba(77,142,255,0.03)); border:1px solid rgba(255,203,43,0.2);border-radius:.5rem; padding:.85rem 1rem;margin:.25rem 0;">
              <div style="font-family:Manrope,sans-serif;font-size:0.8rem;font-weight:800; color:#ffcb2b;margin-bottom:.4rem;">🔒 Pro Features</div>
              <div style="font-family:Manrope,sans-serif;font-size:0.78rem;color:#7a8299; line-height:1.65;margin-bottom:.45rem;">
                <span style="color:#eaefff;">Model Comparison</span> — XGB vs Prophet vs LR<br>
                <span style="color:#eaefff;">Bootstrap CI</span> — 95% confidence ribbon<br>
                <span style="color:#eaefff;">30-day Outlook</span> — extended forecast
              </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("✦ Upgrade to Pro — Unlock Everything", use_container_width=True, key="upgrade_cta_features"):
                st.session_state.show_upgrade_modal = True
                st.rerun()
        run_halal_check = st.checkbox(_L["halal_check"], value=True)
    else:
        run_model_compare = False; run_halal_check = True; show_conf_interval = False; ci_bootstrap_n = 100

    if not is_beginner:
        st.markdown("---")
        st.markdown(f'<div class="stat-row">{_L["multi_stock"]}</div>', unsafe_allow_html=True)
        if _is_pro():
            compare_tickers_raw = st.text_input(_L["compare_tickers"], value="", placeholder="e.g. AAPL,TSLA,NVDA",
                                                label_visibility="collapsed", key="compare_input")
            compare_tickers = [t.strip().upper() for t in compare_tickers_raw.split(",") if t.strip()] if compare_tickers_raw.strip() else []
        else:
            compare_tickers = []
            # FIX 3: Explicit locked feature card
            st.markdown(f"""
            <div style="background:rgba(77,142,255,0.04);border:1px solid rgba(77,142,255,0.16); border-radius:.5rem;padding:.7rem .85rem;margin:.15rem 0;">
              <div style="font-family:Manrope,sans-serif;font-size:0.78rem;color:#7a8299;line-height:1.55;">
                🔒 <span style="color:#c2d4ff;font-weight:700;">Multi-stock compare</span> — 
                run up to 5 tickers side-by-side with Pro.
              </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("✦ Unlock Multi-Stock", use_container_width=True, key="upgrade_cta_multi"):
                st.session_state.show_upgrade_modal = True
                st.rerun()
    else:
        compare_tickers = []

    st.markdown("---")
    _at_limit = (not _is_pro_user) and (st.session_state.get("usage_count", 0) >= _get_limit("daily_analyses"))
    if _at_limit:
        st.markdown(f"""
        <div style="background:rgba(255,87,87,0.05);border:1px solid rgba(255,87,87,0.18); padding:.7rem .9rem;border-radius:.45rem;text-align:center;margin-bottom:.35rem;">
          <div style="font-family:Manrope,sans-serif;font-size:.76rem;color:#ff5757; font-weight:700;letter-spacing:.06em;text-transform:uppercase;">
            🔒 Daily limit reached
          </div>
          <div style="font-family:Manrope,sans-serif;font-size:.74rem;color:#7a8299;margin-top:.25rem;">
            {_get_limit("daily_analyses")}/{_get_limit("daily_analyses")} analyses used today
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("✦ Upgrade to Pro", use_container_width=True, key="upgrade_cta_run"):
            st.session_state.show_upgrade_modal = True
            st.rerun()
    else:
        if st.button(_L["run"], use_container_width=True):
            st.session_state.run_pressed = True
            if not _is_pro_user:
                new_count = st.session_state.get("usage_count", 0) + 1
                st.session_state.usage_count = new_count
                st.session_state.analyses_today = new_count
                _sb_increment_usage(_user_id())
    run_btn = st.session_state.get("run_pressed", False) and not _at_limit

    # Watchlist
    st.markdown("---")
    _wl_count    = len(st.session_state.watchlist)
    _wl_plan_lim = _get_limit("watchlist_stocks")
    _wl_at_limit = _wl_count >= _wl_plan_lim
    _wl_ct_color = '#ffd426' if _is_pro_user else ('#ff5f5f' if _wl_at_limit else '#3e4558')
    st.markdown(f'''
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem;">
      <div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#e4eafd;">{_L["watchlist"]}</div>
      <div style="font-family:IBM Plex Mono,monospace;font-size:.7rem;color:{_wl_ct_color};">{_wl_count}/{_wl_plan_lim}</div>
    </div>
    ''', unsafe_allow_html=True)

    if not _wl_at_limit:
        wl_c1, wl_c2 = st.columns([3, 1])
        with wl_c1:
            add_ticker_input = st.text_input("Add", placeholder="e.g. AAPL", label_visibility="collapsed", key="wl_add").strip().upper()
        with wl_c2:
            add_clicked = st.button("＋", use_container_width=True, key="wl_add_btn")
        if add_clicked and add_ticker_input:
            if add_ticker_input not in st.session_state.watchlist:
                # Re-check limit at add time (prevents race condition)
                if len(st.session_state.watchlist) < _wl_plan_lim:
                    if _sb_add_watchlist(_user_id(), add_ticker_input):
                        st.session_state.watchlist.append(add_ticker_input)
                        st.rerun()
                else:
                    st.warning(f"Watchlist limit reached ({_wl_plan_lim} stocks on your plan).")
    else:
        if _is_pro_user:
            st.markdown(f'<div style="font-family:Manrope,sans-serif;font-size:.63rem;color:#8a8fa0;margin-bottom:.4rem;">Pro limit: {_wl_plan_lim} stocks</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-family:Manrope,sans-serif;font-size:.63rem;color:#ff5f5f;margin-bottom:.4rem;">Free limit reached · <span style="color:#4d8eff;">Upgrade for {PLAN_LIMITS["pro"]["watchlist_stocks"]} stocks</span></div>', unsafe_allow_html=True)

    if st.session_state.get("_wl_toast"):
        _wt = st.session_state.pop("_wl_toast")
        st.markdown(f'<script>if(window.SCNotify)SCNotify.saved("{_wt}");</script>', unsafe_allow_html=True)

    if st.session_state.watchlist:
        for wl_sym in list(st.session_state.watchlist):
            wc1, wc2, wc3 = st.columns([2, 2, 1])
            with wc1:
                # Clickable — loads that ticker into the analysis
                if st.button(wl_sym, key=f"wl_load_{wl_sym}", use_container_width=True):
                    st.session_state.load_ticker_from_watchlist = wl_sym
                    st.session_state.run_pressed = False
                    st.rerun()
            with wc2:
                try:
                    _qt   = av_get_quote(wl_sym)
                    _px   = _qt["price"]
                    _chg  = _qt["change_pct"]
                    _col  = "#00e5b0" if _chg >= 0 else "#ff5f5f"
                    _sign = "▲" if _chg >= 0 else "▼"
                    st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.65rem;color:{_col};padding:.4rem 0;text-align:right;">{_sign} ${_px:.2f}</div>', unsafe_allow_html=True)
                except Exception:
                    st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#3e4558;padding:.4rem 0;text-align:right;">—</div>', unsafe_allow_html=True)
            with wc3:
                if st.button("✕", key=f"wl_del_{wl_sym}", use_container_width=True):
                    _sb_remove_watchlist(_user_id(), wl_sym)
                    st.session_state.watchlist.remove(wl_sym)
                    if wl_sym in st.session_state.alert_signals:
                        del st.session_state.alert_signals[wl_sym]
                    st.rerun()
    else:
        st.markdown(f'<div style="font-family:Manrope,sans-serif;font-size:.65rem;color:#252f47;padding:.3rem 0;">{_L["no_stocks_saved"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    # ── Alerts section ──────────────────────────────────────────────────────
    st.markdown('<div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#e4eafd;margin-bottom:.6rem;">🔔 Alerts</div>', unsafe_allow_html=True)

    # In-session signal change alert
    alert_on_signal_change = st.checkbox(_L["alert_signal_change"], value=True, key="signal_alert_chk")

    # Daily email digest toggle — pre-compute ALL variables before markdown
    _email_on     = st.session_state.get("email_alerts_enabled", False)
    _email_status = "ON" if _email_on else "OFF"
    _email_color  = "#00e5b0" if _email_on else "#3e4558"
    _email_bg     = "rgba(0,229,176,0.06)" if _email_on else "rgba(77,142,255,0.05)"
    _email_border = "rgba(0,229,176,0.2)" if _email_on else "rgba(77,142,255,0.15)"
    _pro_badge    = "✦ Pro · delivered 6AM daily" if _is_pro() else "Free · weekdays 7AM UTC"

    st.markdown(
        f'<div style="background:{_email_bg};border:1px solid {_email_border};' f'border-radius:.6rem;padding:.8rem 1rem;margin:.4rem 0;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;">'
        f'<div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:700;color:#e4eafd;">📧 Daily Email Digest</div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:.62rem;color:{_email_color};font-weight:700;">{_email_status}</div>'
        f'</div>'
        f'<div style="font-family:Manrope,sans-serif;font-size:.62rem;color:#8a8fa0;margin-top:.3rem;line-height:1.5;">'
        f'Get watchlist signals in your inbox every morning.<br>'
        f'<span style="color:#3e4558;">{_pro_badge} · {_user_email()}</span>'
        f'</div></div>',
        unsafe_allow_html=True
    )

    _toggle_label = "🔕 Turn Off Email Alerts" if _email_on else "📧 Enable Daily Email Digest"
    if st.button(_toggle_label, use_container_width=True, key="email_alert_toggle"):
        new_val = not _email_on
        _ok = _sb_set_email_alerts(_user_id(), new_val)
        if new_val:
            _current_email = _user_email()
            if _ok:
                st.success(f"✓ Daily digest enabled — emails will be sent to {_current_email} on weekdays.")
            else:
                st.warning(
                    f"⚠ Preference saved locally for this session. "
                    "Run the Supabase SQL setup script to persist across logins. "
                    "(ALTER TABLE user_usage ADD COLUMN IF NOT EXISTS email_alerts_enabled BOOLEAN NOT NULL DEFAULT FALSE;)"
                )
        else:
            st.info("Email alerts turned off.")
        st.rerun()

    # Plan management
    st.markdown("---")
    if _is_pro():
        st.markdown(
            '<div style="background:linear-gradient(90deg,rgba(255,212,38,0.08),rgba(77,142,255,0.05));' 'border:1px solid rgba(255,212,38,0.25);border-radius:.5rem;padding:.55rem .9rem;' 'font-family:Manrope,sans-serif;font-size:.63rem;font-weight:700;color:#ffd426;' 'text-align:center;letter-spacing:.04em;">✦ Pro Plan Active</div>',
            unsafe_allow_html=True
        )
    else:
        if st.button("✦ Upgrade to Pro", use_container_width=True, key="upgrade_cta_sidebar_bottom"):
            st.session_state.show_upgrade_modal = True
            st.rerun()


# ═══════════════════════════════════════════════════════════════════
# MAIN CONTENT — Landing or Analysis
# ═══════════════════════════════════════════════════════════════════
if not run_btn:
    # ── Landing Dashboard ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;padding-top:.5rem;">
      <div style="font-family:var(--sans);font-size:1.65rem;font-weight:800;color:#eaefff; letter-spacing:-.025em;line-height:1.2;">
        {_L["dashboard_title"]} <span style="color:#4d8eff;">{_L["dashboard_subtitle"]}</span>
      </div>
      <div style="font-size:.88rem;color:#7a8299;margin-top:.5rem;font-weight:500;line-height:1.65; max-width:580px;">{_L["dashboard_desc"]}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Onboarding card — only for new users ──────────────────────────────────
    if st.session_state.get("show_onboarding", False):
        _user_first = _user_email().split("@")[0].title()
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(77,142,255,0.1) 0%,rgba(0,229,176,0.06) 100%); border:1px solid rgba(77,142,255,0.25);border-radius:1rem;padding:1.8rem 2rem; margin-bottom:1.5rem;position:relative;overflow:hidden;">
          <div style="position:absolute;top:-20px;right:-20px;width:120px;height:120px; border-radius:50%;background:radial-gradient(circle,rgba(77,142,255,0.15),transparent 70%); pointer-events:none;"></div>
          <div style="font-family:IBM Plex Mono,monospace;font-size:.7rem;letter-spacing:.12em; text-transform:uppercase;color:#4d8eff;margin-bottom:.4rem;font-weight:700;">
            Welcome to Stockcast
          </div>
          <div style="font-family:Manrope,sans-serif;font-size:1.2rem;font-weight:800; color:#eaefff;letter-spacing:-.02em;margin-bottom:.4rem;">
            Hi {_user_first} 👋 — Your AI stock research assistant is ready.
          </div>
          <div style="font-family:Manrope,sans-serif;font-size:.86rem;color:#7a8299; margin-bottom:1.2rem;line-height:1.65;max-width:540px;">
            AI-powered stock insights for smarter decisions. Here's how to get started:
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin-bottom:1.2rem;">
            <div style="background:rgba(8,14,28,0.6);border:1px solid rgba(77,142,255,0.2); border-radius:.75rem;padding:1rem 1.1rem;">
              <div style="font-family:IBM Plex Mono,monospace;font-size:1.3rem;font-weight:700; color:#4d8eff;margin-bottom:.4rem;">01</div>
              <div style="font-family:Manrope,sans-serif;font-size:.68rem;font-weight:800; letter-spacing:.08em;text-transform:uppercase;color:#e4eafd;margin-bottom:.3rem;">
                Search a Stock
              </div>
              <div style="font-family:Manrope,sans-serif;font-size:.75rem;color:#8a8fa0;line-height:1.5;">
                Type a company name or ticker in the sidebar search.
              </div>
            </div>
            <div style="background:rgba(8,14,28,0.6);border:1px solid rgba(0,229,176,0.2); border-radius:.75rem;padding:1rem 1.1rem;">
              <div style="font-family:IBM Plex Mono,monospace;font-size:1.3rem;font-weight:700; color:#00e5b0;margin-bottom:.4rem;">02</div>
              <div style="font-family:Manrope,sans-serif;font-size:.68rem;font-weight:800; letter-spacing:.08em;text-transform:uppercase;color:#e4eafd;margin-bottom:.3rem;">
                Run Analysis
              </div>
              <div style="font-family:Manrope,sans-serif;font-size:.75rem;color:#8a8fa0;line-height:1.5;">
                Click "Run Analysis" — your AI assistant will analyse 20 signals in seconds.
              </div>
            </div>
            <div style="background:rgba(8,14,28,0.6);border:1px solid rgba(255,212,38,0.2); border-radius:.75rem;padding:1rem 1.1rem;">
              <div style="font-family:IBM Plex Mono,monospace;font-size:1.3rem;font-weight:700; color:#ffd426;margin-bottom:.4rem;">03</div>
              <div style="font-family:Manrope,sans-serif;font-size:.68rem;font-weight:800; letter-spacing:.08em;text-transform:uppercase;color:#e4eafd;margin-bottom:.3rem;">
                Save to Watchlist
              </div>
              <div style="font-family:Manrope,sans-serif;font-size:.75rem;color:#8a8fa0;line-height:1.5;">
                Add up to {FREE_PLAN_WATCHLIST_LIMIT} stocks to your watchlist for quick access.
              </div>
            </div>
          </div>
          <div style="margin-top:1rem;font-family:Manrope,sans-serif;font-size:.65rem; color:#3e4558;letter-spacing:.04em;">
            Free Plan · {PLAN_LIMITS["free"]["daily_analyses"]} analyses per day · {PLAN_LIMITS["free"]["watchlist_stocks"]} watchlist stocks ·
            <span style="color:#4d8eff;cursor:pointer;" onclick="">Upgrade to Pro for unlimited access</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Got it — dismiss", key="dismiss_onboarding"):
            st.session_state.show_onboarding = False
            st.rerun()

    # Market summary cards — live data
    live_indices = get_live_market_indices()
    _idx_map = {name: (price, pct, col) for name, price, pct, col in live_indices}
    _sp  = _idx_map.get("S&P 500",    ("—","—","#3e4558"))
    _nd  = _idx_map.get("NASDAQ 100", ("—","—","#adc6ff"))
    _vix = _idx_map.get("VIX",        ("—","—","#00e5b0"))
    _fg_data = get_fear_greed_index()
    if _fg_data and _fg_data.get("score"):
        _fg_score = _fg_data["score"]
        _fg_val = f"{_fg_score:.0f}"
        _fg_sub = _fg_data["rating"]
        _fg_color = "#00e5b0" if _fg_score >= 55 else ("#ff5f5f" if _fg_score <= 45 else "#ffd426")
    else:
        # Fallback: estimate from VIX — VIX < 15 → greed, > 25 → fear
        try:
            _vix_val = float(_vix[0].replace(",","")) if _vix[0] != "—" else None
        except Exception:
            _vix_val = None
        if _vix_val is not None:
            if _vix_val < 15:
                _fg_val, _fg_sub, _fg_color = "Greed", "Low VIX → Risk-on", "#00e5b0"
            elif _vix_val > 25:
                _fg_val, _fg_sub, _fg_color = "Fear", "High VIX → Risk-off", "#ff5f5f"
            else:
                _fg_val, _fg_sub, _fg_color = "Neutral", "Moderate VIX", "#ffd426"
        else:
            _fg_val, _fg_sub, _fg_color = "N/A", "Data unavailable", "#3e4558"

    def _sk_val(v, color=None):
        if v == "—":
            return '<div class="skeleton sk-line value" style="width:65%;margin:.3rem auto;"></div>'
        s = f"color:{color};" if color else ""
        return f'<div class="stat-value" style="{s}">{v}</div>'

    st.markdown(f"""
    <div class="stat-grid" style="margin-bottom:.5rem;">
      <div class="stat-card">
        <div class="stat-label">S&amp;P 500 · Market Pulse
          <span class="chip live dot" style="font-size:.7rem;padding:.15rem .45rem;margin-left:5px;">Live</span>
        </div>
        {_sk_val(_sp[0])}
        <div class="stat-sub" style="color:{_sp[2]};font-weight:700;font-size:.7rem;">{_sp[1]}</div>
      </div>
      <div class="stat-card" style="border-top-color:#adc6ff;">
        <div class="stat-label">NASDAQ 100 · Tech Momentum
          <span class="chip info dot" style="font-size:.7rem;padding:.15rem .45rem;margin-left:5px;">Tech</span>
        </div>
        {_sk_val(_nd[0], "#adc6ff")}
        <div class="stat-sub" style="color:{_nd[2]};font-weight:700;font-size:.7rem;">{_nd[1]}</div>
      </div>
      <div class="stat-card" style="border-top-color:{_fg_color};">
        <div class="stat-label">Fear &amp; Greed · Sentiment</div>
        <div class="stat-value" style="color:{_fg_color};">{_fg_val}</div>
        <div class="stat-sub" style="color:{_fg_color};font-size:.7rem;">{_fg_sub}</div>
      </div>
      <div class="stat-card" style="border-top-color:#00e5b0;">
        <div class="stat-label">VIX · Volatility Index</div>
        {_sk_val(_vix[0], "#00e5b0")}
        <div class="stat-sub" style="color:{_vix[2]};font-size:.7rem;">{_vix[1]} · {_L.get("low_volatility","Low volatility")}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Watchlist live prices if any — enhanced cards with signal badges
    if st.session_state.watchlist:
        st.markdown("<hr style='margin:.8rem 0;'>", unsafe_allow_html=True)

        # Header row with count
        _wl_total = len(st.session_state.watchlist)
        _wl_plan_max = _get_limit("watchlist_stocks")
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.8rem;">
          <div style="font-family:Manrope,sans-serif;font-size:.68rem;font-weight:800; letter-spacing:.12em;text-transform:uppercase;color:#e4eafd;">
            ⭐ {_L["watchlist_live"]}
          </div>
          <div style="display:flex;align-items:center;gap:.5rem;">
            <span class="chip live dot" style="font-size:.5rem;padding:.18rem .55rem;">Live</span>
            <span style="font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#3e4558;">
              {_wl_total}/{_wl_plan_max}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Show all watchlist items in rows of 4
        _wl_display = st.session_state.watchlist
        _n_cols = min(len(_wl_display), 4)
        wl_cols = st.columns(_n_cols)
        for i, wl_sym in enumerate(_wl_display):
            with wl_cols[i % _n_cols]:
                try:
                    _fi   = av_get_quote(wl_sym)
                    _px   = _fi["price"]
                    _chg  = _fi["change_pct"]
                    _col  = "#00e5b0" if _chg >= 0 else "#ff5f5f"
                    _sign = "▲" if _chg >= 0 else "▼"
                    _wl_chip = "buy" if _chg >= 0 else "sell"
                    # Compute quick signal for badge
                    try:
                        _wl_hist = _yf_download_with_retry(wl_sym, period="3mo", interval="1d")
                        _wl_sig  = _api_compute_signal(_wl_hist)
                        _sig_txt  = _wl_sig["signal"]
                        _sig_conf = _wl_sig["conf"]
                        _sig_chip = "buy" if _sig_txt == "BUY" else "sell" if _sig_txt == "SELL" else "hold"
                        _sig_icon = "▲" if _sig_txt == "BUY" else "▼" if _sig_txt == "SELL" else "◆"
                    except Exception:
                        _sig_txt, _sig_conf, _sig_chip, _sig_icon = "—", 0, "hold", "◆"
                    st.markdown(f"""
                    <div style="background:linear-gradient(145deg,#0f1727,#141d30);border:1px solid #252f47; border-top:2px solid {_col};padding:1.1rem 1.2rem;border-radius:.75rem; transition:border-color .2s;">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem;">
                        <div style="font-family:IBM Plex Mono,monospace;font-size:.62rem;letter-spacing:.14em; color:#3e4558;text-transform:uppercase;font-weight:700;">{wl_sym}</div>
                        <span class="chip {_sig_chip}" style="font-size:.5rem;padding:.18rem .5rem;">
                          {_sig_icon} {_sig_txt}
                        </span>
                      </div>
                      <div style="font-family:IBM Plex Mono,monospace;font-size:1.25rem;font-weight:700; color:#e4eafd;margin:.25rem 0 .5rem;">${_px:.2f}</div>
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span class="chip {_wl_chip} dot" style="font-size:.7rem;padding:.18rem .5rem;">
                          {_sign} {abs(_chg):.2f}%
                        </span>
                        <span style="font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#3e4558;">
                          conf {_sig_conf}%
                        </span>
                      </div>
                    </div>""", unsafe_allow_html=True)
                    # Small gap between rows
                    if i < len(_wl_display) - 1:
                        st.markdown('<div style="margin-bottom:.5rem;"></div>', unsafe_allow_html=True)
                except Exception as e:
                    logger.debug("Dashboard watchlist: could not load quote for '%s': %s", wl_sym, e)
                    st.markdown(f'<div style="background:#0f1727;border:1px solid #252f47;padding:1rem;text-align:center;font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#3e4558;border-radius:.75rem;">{wl_sym}<br><span style="color:#252f47;">—</span></div>', unsafe_allow_html=True)

        # Watchlist signal summary table (all tickers, full row)
        if len(st.session_state.watchlist) > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            _wl_rows_html = ""
            for _ws in st.session_state.watchlist:
                try:
                    _wq   = av_get_quote(_ws)
                    _wpx  = _wq["price"]
                    _wchg = _wq["change_pct"]
                    _wcol = "#00e5b0" if _wchg >= 0 else "#ff5f5f"
                    _warr = "▲" if _wchg >= 0 else "▼"
                    # Quick signal
                    try:
                        _wh  = _yf_download_with_retry(_ws, period="3mo", interval="1d")
                        _wsig = _api_compute_signal(_wh)
                        _wst  = _wsig["signal"]
                        _wsc  = _wsig["conf"]
                        _wsr  = _wsig.get("reason", "")[:55]
                    except Exception:
                        _wst, _wsc, _wsr = "—", 0, "—"
                    _wsig_col  = "#00e5b0" if _wst == "BUY" else "#ff5f5f" if _wst == "SELL" else "#ffd426"
                    _wsig_bg   = "rgba(0,229,176,0.08)" if _wst == "BUY" else "rgba(255,95,95,0.08)" if _wst == "SELL" else "rgba(255,212,38,0.08)"
                    _conf_bar  = f'<div style="display:inline-block;width:{_wsc}px;max-width:80px;height:3px;background:{_wsig_col};border-radius:2px;vertical-align:middle;margin-left:5px;opacity:.6;"></div>'
                    _wl_rows_html += f"""
                    <tr style="border-bottom:1px solid #1e2740;">
                      <td style="padding:.6rem .9rem;font-family:IBM Plex Mono,monospace;font-size:.72rem;font-weight:700;color:#adc6ff;letter-spacing:.06em;">{_ws}</td>
                      <td style="padding:.6rem .9rem;font-family:IBM Plex Mono,monospace;font-size:.75rem;font-weight:700;color:#e4eafd;">${_wpx:,.2f}</td>
                      <td style="padding:.6rem .9rem;font-family:IBM Plex Mono,monospace;font-size:.72rem;color:{_wcol};font-weight:700;">{_warr} {abs(_wchg):.2f}%</td>
                      <td style="padding:.6rem .9rem;">
                        <span style="background:{_wsig_bg};border:1px solid {_wsig_col}44;border-radius:100px; padding:.2rem .65rem;font-family:IBM Plex Mono,monospace;font-size:.62rem; font-weight:800;color:{_wsig_col};letter-spacing:.06em;">{_wst}</span>
                      </td>
                      <td style="padding:.6rem .9rem;font-family:IBM Plex Mono,monospace;font-size:.62rem;color:#3e4558;">
                        {_wsc}%{_conf_bar}
                      </td>
                      <td style="padding:.6rem .9rem;font-family:Manrope,sans-serif;font-size:.7rem;color:#8a8fa0;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{_wsr}</td>
                    </tr>"""
                except Exception:
                    _wl_rows_html += f"""
                    <tr style="border-bottom:1px solid #1e2740;">
                      <td style="padding:.6rem .9rem;font-family:IBM Plex Mono,monospace;font-size:.72rem;font-weight:700;color:#adc6ff;">{_ws}</td>
                      <td colspan="5" style="padding:.6rem .9rem;font-family:Manrope,sans-serif;font-size:.7rem;color:#3e4558;">Unable to fetch data</td>
                    </tr>"""
            st.markdown(f"""
            <div style="background:#0a1120;border:1px solid #1e2740;border-radius:.75rem;overflow:hidden;margin-bottom:1rem;">
              <div style="padding:.75rem 1.1rem;background:#0f1727;border-bottom:1px solid #1e2740; display:flex;align-items:center;justify-content:space-between;">
                <div style="font-family:Manrope,sans-serif;font-size:.62rem;font-weight:800;letter-spacing:.12em; text-transform:uppercase;color:#4d8eff;">Signal Overview</div>
                <span class="chip live dot" style="font-size:.5rem;padding:.18rem .55rem;">Live</span>
              </div>
              <table style="width:100%;border-collapse:collapse;">
                <thead>
                  <tr style="border-bottom:1px solid #252f47;background:#090e1c;">
                    <th style="padding:.5rem .9rem;text-align:left;font-family:Manrope,sans-serif;font-size:.6rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#3e4558;">Ticker</th>
                    <th style="padding:.5rem .9rem;text-align:left;font-family:Manrope,sans-serif;font-size:.6rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#3e4558;">Price</th>
                    <th style="padding:.5rem .9rem;text-align:left;font-family:Manrope,sans-serif;font-size:.6rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#3e4558;">Change</th>
                    <th style="padding:.5rem .9rem;text-align:left;font-family:Manrope,sans-serif;font-size:.6rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#3e4558;">Signal</th>
                    <th style="padding:.5rem .9rem;text-align:left;font-family:Manrope,sans-serif;font-size:.6rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#3e4558;">Conf.</th>
                    <th style="padding:.5rem .9rem;text-align:left;font-family:Manrope,sans-serif;font-size:.6rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#3e4558;">Reason</th>
                  </tr>
                </thead>
                <tbody>{_wl_rows_html}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

    # How it works
    st.markdown("<hr style='margin:.8rem 0;'>", unsafe_allow_html=True)
    st.subheader(_L["how_it_works"])
    hw1, hw2, hw3 = st.columns(3)
    for _col, _num, _color, _tk, _bk in [
        (hw1, "01", "#4d8eff", "hw1_title", "hw1_body"),
        (hw2, "02", "#00e5b0", "hw2_title", "hw2_body"),
        (hw3, "03", "#ffd426", "hw3_title", "hw3_body"),
    ]:
        _t = _L[_tk].replace("'", "&#39;")
        _b = _L[_bk].replace("'", "&#39;")
        with _col:
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#0f1727,#141d30);' f'border:1px solid #252f47;border-top:2px solid {_color};' f'padding:1.3rem 1.4rem;border-radius:.6rem;min-height:140px;">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.2rem;' f'font-weight:700;color:{_color};margin-bottom:.5rem;">{_num}</div>'
                f'<div style="font-family:Manrope,sans-serif;font-size:.68rem;' f'letter-spacing:.1em;text-transform:uppercase;color:#e4eafd;' f'font-weight:700;margin-bottom:.4rem;">{_t}</div>'
                f'<div style="font-family:Manrope,sans-serif;font-size:.78rem;' f'color:#8a8fa0;line-height:1.6;">{_b}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<hr style='margin:.8rem 0;'>", unsafe_allow_html=True)
    st.subheader(_L["platform_features"])
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin:1rem 0 1.5rem;">

      <!-- XGBoost Engine — full width top -->
      <div style="grid-column:1/-1;background:linear-gradient(135deg,rgba(77,142,255,0.08) 0%,rgba(77,142,255,0.03) 100%); border:1px solid rgba(77,142,255,0.2);border-top:2px solid #4d8eff;border-radius:.75rem; padding:1.3rem 1.5rem;display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;">
        <div style="font-size:2rem;line-height:1;flex-shrink:0;">📈</div>
        <div style="flex:1;min-width:200px;">
          <div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:700;letter-spacing:.1em; text-transform:uppercase;color:#4d8eff;margin-bottom:.3rem;">AI Price Outlook</div>
          <div style="font-family:Manrope,sans-serif;font-size:1.05rem;font-weight:800;color:#eaefff; margin-bottom:.35rem;">XGBoost Engine</div>
          <div style="font-family:Manrope,sans-serif;font-size:.82rem;color:#7a8299;line-height:1.6;">
            Projects next-day price across 30 technical signals — RSI, MACD, Stochastic, ADX, Williams %R, OBV — with ATR-scaled take-profit and stop-loss levels.
          </div>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:.35rem;flex-shrink:0;">
          <span style="background:rgba(194,212,255,0.08);border:1px solid rgba(194,212,255,0.2); color:#c2d4ff;font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700; padding:.22rem .65rem;border-radius:100px;letter-spacing:.05em;">● XGBoost</span>
          <span style="background:rgba(77,142,255,0.08);border:1px solid rgba(77,142,255,0.25); color:#4d8eff;font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700; padding:.22rem .65rem;border-radius:100px;letter-spacing:.05em;">● 30 Signals</span>
          <span style="background:rgba(77,142,255,0.08);border:1px solid rgba(77,142,255,0.25); color:#4d8eff;font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700; padding:.22rem .65rem;border-radius:100px;letter-spacing:.05em;">● ATR TP/SL</span>
        </div>
      </div>

      <!-- Watchlist + Alerts -->
      <div style="background:rgba(0,217,166,0.04);border:1px solid rgba(0,217,166,0.15); border-top:2px solid #00d9a6;border-radius:.75rem;padding:1.1rem 1.25rem;">
        <div style="font-size:1.4rem;margin-bottom:.6rem;">⭐</div>
        <div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:700;letter-spacing:.09em; text-transform:uppercase;color:#00d9a6;margin-bottom:.3rem;">Watchlist + Alerts</div>
        <div style="font-family:Manrope,sans-serif;font-size:.82rem;color:#7a8299;line-height:1.6;margin-bottom:.7rem;">
          Save stocks, track live prices, get instant alerts when signals flip BUY ↔ SELL.
        </div>
        <span style="background:rgba(0,217,166,0.08);border:1px solid rgba(0,217,166,0.22); color:#00d9a6;font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700; padding:.22rem .65rem;border-radius:100px;">● Live Prices</span>
      </div>

      <!-- Explainable Signals -->
      <div style="background:rgba(0,217,166,0.04);border:1px solid rgba(0,217,166,0.15); border-top:2px solid #00d9a6;border-radius:.75rem;padding:1.1rem 1.25rem;">
        <div style="font-size:1.4rem;margin-bottom:.6rem;">⚡</div>
        <div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:700;letter-spacing:.09em; text-transform:uppercase;color:#00d9a6;margin-bottom:.3rem;">9 Explainable Signals</div>
        <div style="font-family:Manrope,sans-serif;font-size:.82rem;color:#7a8299;line-height:1.6;margin-bottom:.7rem;">
          RSI · MACD · Bollinger · Stochastic · Williams %R · ADX · OBV — each scored and explained.
        </div>
        <div style="display:flex;gap:.3rem;flex-wrap:wrap;">
          <span style="background:rgba(0,217,166,0.09);border:1px solid rgba(0,217,166,0.3);color:#00d9a6; font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700;padding:.22rem .65rem;border-radius:100px;">● BUY</span>
          <span style="background:rgba(255,87,87,0.09);border:1px solid rgba(255,87,87,0.3);color:#ff5757; font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700;padding:.22rem .65rem;border-radius:100px;">● SELL</span>
          <span style="background:rgba(255,203,43,0.09);border:1px solid rgba(255,203,43,0.3);color:#ffcb2b; font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700;padding:.22rem .65rem;border-radius:100px;">● HOLD</span>
        </div>
      </div>

      <!-- Shariah Screen -->
      <div style="background:rgba(194,212,255,0.03);border:1px solid rgba(194,212,255,0.12); border-top:2px solid #c2d4ff;border-radius:.75rem;padding:1.1rem 1.25rem;">
        <div style="font-size:1.4rem;margin-bottom:.6rem;">☪</div>
        <div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:700;letter-spacing:.09em; text-transform:uppercase;color:#c2d4ff;margin-bottom:.3rem;">Shariah Screen</div>
        <div style="font-family:Manrope,sans-serif;font-size:.82rem;color:#7a8299;line-height:1.6;margin-bottom:.3rem;">
          AAOIFI Standard No.21 — business activity, debt, and cash ratios auto-screened.
        </div>
        <div style="font-family:IBM Plex Mono,monospace;font-size:.72rem;color:#c2d4ff;font-weight:600;">AAOIFI · Standard No.21</div>
      </div>

      <!-- Strategy Simulator -->
      <div style="background:rgba(255,203,43,0.03);border:1px solid rgba(255,203,43,0.15); border-top:2px solid #ffcb2b;border-radius:.75rem;padding:1.1rem 1.25rem;">
        <div style="font-size:1.4rem;margin-bottom:.6rem;">📊</div>
        <div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:700;letter-spacing:.09em; text-transform:uppercase;color:#ffcb2b;margin-bottom:.3rem;">Strategy Simulator</div>
        <div style="font-family:Manrope,sans-serif;font-size:.82rem;color:#7a8299;line-height:1.6;margin-bottom:.7rem;">
          Walk-forward backtesting — Sharpe ratio, max drawdown, win rate, profit factor, equity curve.
        </div>
        <div style="display:flex;gap:.3rem;flex-wrap:wrap;">
          <span style="background:rgba(255,203,43,0.08);border:1px solid rgba(255,203,43,0.25);color:#ffcb2b; font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700;padding:.22rem .65rem;border-radius:100px;">● Sharpe</span>
          <span style="background:rgba(255,203,43,0.08);border:1px solid rgba(255,203,43,0.25);color:#ffcb2b; font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700;padding:.22rem .65rem;border-radius:100px;">● Drawdown</span>
        </div>
      </div>

      <!-- News Sentiment NLP -->
      <div style="background:rgba(0,217,166,0.04);border:1px solid rgba(0,217,166,0.15); border-top:2px solid #00d9a6;border-radius:.75rem;padding:1.1rem 1.25rem;">
        <div style="font-size:1.4rem;margin-bottom:.6rem;">📰</div>
        <div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:700;letter-spacing:.09em; text-transform:uppercase;color:#00d9a6;margin-bottom:.3rem;">News Sentiment NLP</div>
        <div style="font-family:Manrope,sans-serif;font-size:.82rem;color:#7a8299;line-height:1.6;margin-bottom:.7rem;">
          Live headlines analysed with TextBlob. Polarity scored and overlaid on signal context.
        </div>
        <div style="display:flex;gap:.3rem;flex-wrap:wrap;">
          <span style="background:rgba(194,212,255,0.08);border:1px solid rgba(194,212,255,0.2);color:#c2d4ff; font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700;padding:.22rem .65rem;border-radius:100px;">● TextBlob</span>
          <span style="background:rgba(0,217,166,0.08);border:1px solid rgba(0,217,166,0.22);color:#00d9a6; font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700;padding:.22rem .65rem;border-radius:100px;">● Live News</span>
        </div>
      </div>

      <!-- Model Comparison + Portfolio — bottom row -->
      <div style="background:rgba(194,212,255,0.03);border:1px solid rgba(194,212,255,0.12); border-top:2px solid #c2d4ff;border-radius:.75rem;padding:1.1rem 1.25rem;">
        <div style="font-size:1.4rem;margin-bottom:.6rem;">🔬</div>
        <div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:700;letter-spacing:.09em; text-transform:uppercase;color:#c2d4ff;margin-bottom:.3rem;">Model Comparison</div>
        <div style="font-family:Manrope,sans-serif;font-size:.82rem;color:#7a8299;line-height:1.6;margin-bottom:.7rem;">
          XGBoost vs Prophet vs Linear Regression — RMSE, MAE, MAPE, R² side-by-side.
        </div>
        <div style="display:flex;gap:.3rem;flex-wrap:wrap;">
          <span style="background:rgba(255,203,43,0.08);border:1px solid rgba(255,203,43,0.25);color:#ffcb2b; font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700;padding:.22rem .65rem;border-radius:100px;">● Pro</span>
          <span style="background:rgba(194,212,255,0.08);border:1px solid rgba(194,212,255,0.2);color:#c2d4ff; font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700;padding:.22rem .65rem;border-radius:100px;">● 3 Models</span>
        </div>
      </div>

      <!-- Fundamentals Panel -->
      <div style="background:rgba(77,142,255,0.04);border:1px solid rgba(77,142,255,0.15); border-top:2px solid #4d8eff;border-radius:.75rem;padding:1.1rem 1.25rem;">
        <div style="font-size:1.4rem;margin-bottom:.6rem;">📋</div>
        <div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:700;letter-spacing:.09em; text-transform:uppercase;color:#4d8eff;margin-bottom:.3rem;">Fundamentals Panel</div>
        <div style="font-family:Manrope,sans-serif;font-size:.82rem;color:#7a8299;line-height:1.6;margin-bottom:.7rem;">
          P/E, P/B, EPS growth, margins, ROE, analyst targets, short float — all in one panel.
        </div>
        <div style="display:flex;gap:.3rem;flex-wrap:wrap;">
          <span style="background:rgba(77,142,255,0.08);border:1px solid rgba(77,142,255,0.25);color:#4d8eff; font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700;padding:.22rem .65rem;border-radius:100px;">● Valuation</span>
          <span style="background:rgba(77,142,255,0.08);border:1px solid rgba(77,142,255,0.25);color:#4d8eff; font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700;padding:.22rem .65rem;border-radius:100px;">● Analyst Targets</span>
        </div>
      </div>

      <!-- Portfolio Tracker -->
      <div style="background:rgba(255,203,43,0.03);border:1px solid rgba(255,203,43,0.15); border-top:2px solid #ffcb2b;border-radius:.75rem;padding:1.1rem 1.25rem;">
        <div style="font-size:1.4rem;margin-bottom:.6rem;">🏦</div>
        <div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:700;letter-spacing:.09em; text-transform:uppercase;color:#ffcb2b;margin-bottom:.3rem;">Portfolio Tracker</div>
        <div style="font-family:Manrope,sans-serif;font-size:.82rem;color:#7a8299;line-height:1.6;margin-bottom:.7rem;">
          Track holdings, unrealised P&amp;L, sector allocation, cost basis, and transaction history.
        </div>
        <div style="display:flex;gap:.3rem;flex-wrap:wrap;">
          <span style="background:rgba(77,142,255,0.08);border:1px solid rgba(77,142,255,0.25);color:#4d8eff; font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700;padding:.22rem .65rem;border-radius:100px;">● P&amp;L</span>
          <span style="background:rgba(77,142,255,0.08);border:1px solid rgba(77,142,255,0.25);color:#4d8eff; font-family:IBM Plex Mono,monospace;font-size:.68rem;font-weight:700;padding:.22rem .65rem;border-radius:100px;">● Sectors</span>
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="text-align:center;margin-top:2rem;font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#252f47;letter-spacing:.08em;"> </div>', unsafe_allow_html=True)

    # Upgrade CTA — only for free users
    if not _is_pro():
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(255,203,43,0.06) 0%,rgba(77,142,255,0.04) 100%); border:1px solid rgba(255,203,43,0.18);border-radius:.875rem; padding:1.8rem 2rem;margin-top:1rem;text-align:center;">
          <div style="font-family:IBM Plex Mono,monospace;font-size:.7rem;letter-spacing:.14em; text-transform:uppercase;color:#ffcb2b;margin-bottom:.5rem;font-weight:700;">Stockcast Pro</div>
          <div style="font-family:Manrope,sans-serif;font-size:1.1rem;font-weight:800; color:#eaefff;letter-spacing:-.015em;margin-bottom:.4rem;">
            Unlock unlimited analyses &amp; advanced signals
          </div>
          <div style="font-family:Manrope,sans-serif;font-size:.84rem;color:#7a8299; margin-bottom:1.3rem;line-height:1.6;">
            Prophet + XGBoost combo · Bootstrap CI · Multi-stock · 50 watchlist stocks
          </div>
          <div style="font-family:IBM Plex Mono,monospace;font-size:1.5rem;font-weight:800; color:#ffcb2b;margin-bottom:1.2rem;">
            $9<span style="font-size:.82rem;color:#7a8299;font-weight:400;margin-left:3px;">/month</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("✦ Upgrade to Pro — Unlock Everything", use_container_width=True, key="upgrade_cta_dashboard"):
            st.session_state.show_upgrade_modal = True
            st.rerun()

else:
    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS ENGINE
    # ═══════════════════════════════════════════════════════════════
    if st.sidebar.button(_L.get("back", "← Back to Dashboard"), use_container_width=True, key="back_btn"):
        st.session_state.run_pressed = False
        st.rerun()
    # ── Breadcrumb bar ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="breadcrumb-bar">
      <nav class="breadcrumb">
        <span class="bc-item"><span class="bc-dot"></span>Stockcast</span>
        <span class="bc-sep">&#xbb;</span>
        <span class="bc-item"><span class="bc-dot"></span>Analysis</span>
        <span class="bc-sep">&#xbb;</span>
        <span class="bc-item active"><span class="bc-dot"></span>{ticker}</span>
      </nav>
      <div class="bc-context" style="display:flex;align-items:center;gap:.5rem;">
        <span class="chip live dot" style="font-size:.5rem;padding:.18rem .5rem;">Live</span>
        <span>{pd.Timestamp.now().strftime('%b %d · %H:%M')}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input validation ──────────────────────────────────────────────────────
    if not ticker or len(ticker.strip()) < 1:
        st.warning("Please enter a ticker symbol.")
        st.stop()
    if len(ticker) > 20:
        st.error("⚠ Ticker symbol too long. Please check and try again.")
        st.stop()
    import re as _re
    if not _re.match(r'^[A-Za-z0-9.\-\^]+$', ticker):
        st.error(f"⚠ '{ticker}' doesn't look like a valid ticker. Use formats like AAPL, RELIANCE.NS, ^GSPC")
        st.stop()

    # ── FIX 2: Multi-step progress feedback ─────────────────────────────────────
    _progress_placeholder = st.empty()

    def _show_progress(step: int, total: int, label: str):
        """Render an inline progress bar with step label."""
        pct = int(step / total * 100)
        _progress_placeholder.markdown(f"""
        <div style="background:var(--bg2);border:1px solid var(--border);border-radius:.75rem; padding:1.1rem 1.4rem;margin:.5rem 0;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.6rem;">
            <span style="font-family:var(--mono);font-size:0.78rem;color:var(--t1);font-weight:600;">
              {label}
            </span>
            <span style="font-family:var(--mono);font-size:0.72rem;color:var(--accent);">
              Step {step} / {total}
            </span>
          </div>
          <div style="height:4px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;">
            <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,#4d8eff,#00e5b0); border-radius:3px;transition:width .4s ease;"></div>
          </div>
          <div style="display:flex;gap:1.2rem;margin-top:.8rem;flex-wrap:wrap;">
            {"".join(
              f'<span style="font-family:var(--mono);font-size:0.72rem;' f'color:{"var(--emerald)" if i < step else "var(--accent)" if i == step else "var(--t4)"};">'
              f'{"✓" if i < step else "●" if i == step else "○"} {lbl}</span>'
              for i, lbl in enumerate([
                "Fetching data", "Computing signals", "Building matrix",
                "Training model", "Generating forecast"
              ], 1)
            )}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── PHASE 1: DATA QUALITY LAYER ──────────────────────────────────────
    _show_progress(1, 5, f"📡 Fetching {ticker} price history...")

    def _get_data_quality(ticker_sym, start, end):
        """Safe fetch with quality metadata — source, timestamp, missing%, validation."""
        import datetime as _dt
        meta = {
            "source": "Yahoo Finance",
            "fetched_at": pd.Timestamp.utcnow(),
            "trading_days": 0,
            "missing_pct": 0.0,
            "last_updated_str": "just now",
            "quality_ok": False,
            "error": None,
        }
        try:
            _raw = fetch_data(ticker_sym, start, end)
            if _raw.empty:
                meta["error"] = f"No data returned for '{ticker_sym}'."
                return pd.DataFrame(), meta
            # Quality checks
            total_cells = _raw.size
            missing     = _raw.isnull().sum().sum()
            missing_pct = round(missing / total_cells * 100, 2) if total_cells > 0 else 0
            _raw = _raw.dropna(subset=["Close"]).ffill().bfill()
            # Human-readable last-update label
            last_ts = _raw.index[-1] if not _raw.empty else None
            if last_ts is not None:
                _last_ts_naive = pd.Timestamp(last_ts).tz_localize(None) if pd.Timestamp(last_ts).tzinfo is not None else pd.Timestamp(last_ts)
                delta_days = (pd.Timestamp.utcnow().normalize().tz_localize(None) -
                              _last_ts_naive.normalize()).days
                if   delta_days == 0: label = "today"
                elif delta_days == 1: label = "yesterday"
                else:                 label = f"{delta_days} days ago"
            else:
                label = "unknown"
            meta.update({
                "trading_days": len(_raw),
                "missing_pct":  missing_pct,
                "last_updated_str": label,
                "quality_ok": len(_raw) >= 50 and missing_pct < 5,
            })
            return _raw, meta
        except Exception as _exc:
            meta["error"] = str(_exc)
            return pd.DataFrame(), meta

    df, _data_meta = _get_data_quality(ticker, start_date, end_date)

    if df.empty:
        _progress_placeholder.empty()
        _err = _data_meta.get("error", "")
        st.markdown(f"""
        <div style="background:rgba(255,87,87,0.06);border:1px solid rgba(255,87,87,0.25); border-left:4px solid #ff5f5f;border-radius:0 .75rem .75rem 0; padding:1.2rem 1.6rem;margin:.8rem 0;">
          <div style="font-family:Manrope,sans-serif;font-size:.62rem;letter-spacing:.16em; text-transform:uppercase;color:#ff5f5f;margin-bottom:.5rem;font-weight:700;">
            ⚠ Data unavailable — {ticker}
          </div>
          <div style="font-family:Manrope,sans-serif;font-size:.82rem;color:#8a8fa0;line-height:1.65;">
            {"Could not fetch market data. " + _err if _err else "No data returned from Yahoo Finance."}<br><br>
            <b style="color:#e4eafd;">Possible fixes:</b><br>
            · Indian stocks: <code>RELIANCE.NS</code> · Indices: <code>^GSPC</code><br>
            · Try a longer date range — ticker may be newly listed<br>
            · Yahoo Finance may be temporarily unavailable — try again shortly
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    _progress_placeholder.empty()

    # ── Enhanced data quality strip (replaces old freshness bar) ──────────
    _dq_color   = "#00e5b0" if _data_meta["quality_ok"] else "#ffd426"
    _dq_warn    = (f'<span style="color:#ffd426;margin-left:.6rem;">⚠ {_data_meta["missing_pct"]:.1f}% missing (forward-filled)</span>'
                   if _data_meta["missing_pct"] > 0 else "")
    _dq_ts      = _data_meta["fetched_at"].strftime("%b %d, %Y · %H:%M UTC")
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.5rem;margin:.2rem 0 .8rem;padding:.55rem 1.1rem;background:rgba(0,229,176,0.04);border:1px solid rgba(0,229,176,0.14);border-radius:.5rem;"><div style="display:flex;align-items:center;gap:.5rem;"><span style="width:7px;height:7px;border-radius:50%;background:{_dq_color};display:inline-block;box-shadow:0 0 5px {_dq_color};"></span><span style="font-family:IBM Plex Mono,monospace;font-size:.68rem;color:{_dq_color};letter-spacing:.05em;">Data source: <b>Yahoo Finance</b> &nbsp;·&nbsp; Updated: <b>{_data_meta["last_updated_str"]}</b> &nbsp;·&nbsp; {_data_meta["trading_days"]:,} trading days loaded {_dq_warn}</span></div><div style="display:flex;align-items:center;gap:.7rem;"><span style="font-family:IBM Plex Mono,monospace;font-size:.62rem;color:#3e4558;">{_dq_ts}</span><span style="display:inline-flex;align-items:center;gap:.4rem;background:rgba(255,87,87,0.07);border:1px solid rgba(255,87,87,0.2);border-radius:2rem;padding:.22rem .75rem;font-family:IBM Plex Mono,monospace;font-size:.63rem;color:rgba(255,87,87,0.7);">⚠ Not Financial Advice</span></div></div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(255,221,45,0.04);border:1px solid rgba(255,221,45,0.3); border-left:4px solid #ffd426;padding:.9rem 1.4rem;margin:.5rem 0 1rem;border-radius:0 .5rem .5rem 0;">
      <div style="font-family:Manrope,sans-serif;font-size:.6rem;letter-spacing:.14em;text-transform:uppercase;color:#ffd426;margin-bottom:.3rem;font-weight:700;">
        {_L["reality_check_title"]}
      </div>
      <div style="font-family:Manrope,sans-serif;font-size:.78rem;color:#8a8fa0;line-height:1.6;">
        {_L["reality_check_body"]}
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_analysis, tab_methodology = st.tabs([_L["analysis_tab"], _L["methodology_tab"]])

    with tab_methodology:
        render_methodology_page(seq_len_val=seq_len, ci_n=ci_bootstrap_n, show_ci=show_conf_interval)

    with tab_analysis:
        # ⭐ Add to Watchlist — prominent action at top of analysis tab
        _in_wl_main = ticker in st.session_state.watchlist
        _wl_full_main = len(st.session_state.watchlist) >= _get_limit("watchlist_stocks")
        _, _wl_col2 = st.columns([3, 1])
        with _wl_col2:
            if _in_wl_main:
                st.markdown(f'<div style="background:rgba(0,229,176,0.08);border:1px solid rgba(0,229,176,0.25);border-radius:.5rem;padding:.5rem .8rem;font-family:Manrope,sans-serif;font-size:.65rem;color:#00e5b0;font-weight:700;text-align:center;">⭐ In Watchlist</div>', unsafe_allow_html=True)
            elif _wl_full_main:
                st.markdown(f'<div style="background:rgba(255,95,95,0.06);border:1px solid rgba(255,95,95,0.2);border-radius:.5rem;padding:.5rem .8rem;font-family:Manrope,sans-serif;font-size:.63rem;color:#ff5f5f;text-align:center;">Watchlist full</div>', unsafe_allow_html=True)
            else:
                if st.button(f"⭐ Add to Watchlist", key="forecast_wl_add", use_container_width=True):
                    if _sb_add_watchlist(_user_id(), ticker):
                        st.session_state.watchlist.append(ticker)
                        st.rerun()
        with _progress_placeholder:
            _show_progress(2, 5, "⚙️ Computing 20 technical signals...")
        df = add_technical_features(df)
        _progress_placeholder.empty()
        close_series = df['Close'].squeeze()

        # ── PHASE 2: ENHANCED PRICE CHART (timeframe selector + MA toggle) ──────
        st.subheader(_L["price_chart"])

        # Timeframe selector buttons
        _tf_options = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252, "All": None}
        _tf_key = f"chart_tf_{ticker}"
        if _tf_key not in st.session_state:
            st.session_state[_tf_key] = "3M"

        _tf_cols = st.columns(len(_tf_options))
        for _tfi, (_tfl, _) in enumerate(_tf_options.items()):
            _is_active = (st.session_state[_tf_key] == _tfl)
            _btn_style = ("background:rgba(77,142,255,0.18);border:1px solid #4d8eff;color:#adc6ff;"
                          if _is_active else "background:#0f1727;border:1px solid #252f47;color:#3e4558;")
            if _tf_cols[_tfi].button(_tfl, key=f"tf_{_tfl}_{ticker}", use_container_width=True):
                st.session_state[_tf_key] = _tfl
                st.rerun()

        _tf_bars = _tf_options[st.session_state[_tf_key]]
        _df_view = df.iloc[-_tf_bars:] if _tf_bars else df
        _cv_close = _df_view["Close"].squeeze()
        _cv_open  = _df_view["Open"].squeeze()

        _show_ma20 = st.checkbox("Show MA 20", value=True, key=f"ma20_{ticker}")

        fig_candle = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.73, 0.27], vertical_spacing=0.02)

        fig_candle.add_trace(go.Candlestick(
            x=_df_view.index,
            open=_cv_open, high=_df_view["High"].squeeze(),
            low=_df_view["Low"].squeeze(), close=_cv_close,
            name="Price",
            increasing_line_color=C_EMERALD, decreasing_line_color=C_RED,
            increasing_fillcolor=C_EMERALD,  decreasing_fillcolor=C_RED,
            hoverlabel=dict(bgcolor="#0f1727", font_size=11),
        ), row=1, col=1)

        # MA overlays — MA20 (toggleable), MA50, MA200
        if _show_ma20 and len(_df_view) >= 20:
            _ma20 = _cv_close.rolling(20).mean()
            fig_candle.add_trace(go.Scatter(
                x=_df_view.index, y=_ma20, name="MA 20",
                line=dict(color="#ff9f40", width=1.3, dash="dot"),
                hovertemplate="MA20: $%{y:.2f}<extra></extra>",
            ), row=1, col=1)

        if "MA50" in _df_view.columns:
            fig_candle.add_trace(go.Scatter(
                x=_df_view.index, y=_df_view["MA50"].squeeze(), name="MA 50",
                line=dict(color=C_YELLOW, width=1.2),
                hovertemplate="MA50: $%{y:.2f}<extra></extra>",
            ), row=1, col=1)
        if "MA200" in _df_view.columns:
            fig_candle.add_trace(go.Scatter(
                x=_df_view.index, y=_df_view["MA200"].squeeze(), name="MA 200",
                line=dict(color=C_ACCENT, width=1.2),
                hovertemplate="MA200: $%{y:.2f}<extra></extra>",
            ), row=1, col=1)

        # Bollinger bands
        if "BB_Upper" in _df_view.columns:
            fig_candle.add_trace(go.Scatter(
                x=_df_view.index, y=_df_view["BB_Upper"].squeeze(), name="BB Upper",
                line=dict(color=C_GREY, width=0.8, dash="dot"),
                hovertemplate="BB Upper: $%{y:.2f}<extra></extra>",
            ), row=1, col=1)
            fig_candle.add_trace(go.Scatter(
                x=_df_view.index, y=_df_view["BB_Lower"].squeeze(), name="BB Lower",
                line=dict(color=C_GREY, width=0.8, dash="dot"),
                fill="tonexty", fillcolor="rgba(77,142,255,0.05)",
                hovertemplate="BB Lower: $%{y:.2f}<extra></extra>",
            ), row=1, col=1)

        # Volume
        colors_vol = [C_EMERALD if c >= o else C_RED for c, o in zip(_cv_close, _cv_open)]
        fig_candle.add_trace(go.Bar(
            x=_df_view.index, y=_df_view["Volume"].squeeze(), name="Volume",
            marker_color=colors_vol, opacity=0.45,
            hovertemplate="%{x}<br>Vol: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)

        candle_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")}
        fig_candle.update_layout(
            **candle_layout,
            title=dict(
                text=f"{ticker} · {st.session_state[_tf_key]} · Candlestick · BB · Volume",
                font=dict(color=C_GREEN, size=13),
            ),
            xaxis_rangeslider_visible=False,
            height=600,
        )
        fig_candle.update_xaxes(
            gridcolor="#252f47", linecolor="#252f47", tickfont=dict(color=C_GREY),
            showspikes=True, spikethickness=1, spikecolor=C_ACCENT, spikedash="dot",
        )
        fig_candle.update_yaxes(gridcolor="#252f47", linecolor="#252f47", tickfont=dict(color=C_GREY))
        st.plotly_chart(fig_candle, use_container_width=True)

        # ── Technical Indicators — 4-panel ──────────────────────────────────────
        st.subheader(_L["tech_indicators"])
        fig_tech = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.28, 0.26, 0.24, 0.22],
            vertical_spacing=0.04,
            subplot_titles=["RSI (14)", "MACD (12/26/9)", "Stochastic %K/%D (14,3)", "Williams %R (14)"]
        )
        # Row 1: RSI
        fig_tech.add_trace(go.Scatter(x=df.index, y=df['RSI'].squeeze(), name="RSI", line=dict(color=C_ACCENT, width=1.5)), row=1, col=1)
        fig_tech.add_hline(y=70, line_dash="dash", line_color=C_RED,    row=1, col=1)
        fig_tech.add_hline(y=30, line_dash="dash", line_color=C_EMERALD, row=1, col=1)
        fig_tech.add_hrect(y0=70, y1=100, fillcolor="rgba(255,107,107,0.04)", line_width=0, row=1, col=1)
        fig_tech.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,229,176,0.04)",  line_width=0, row=1, col=1)
        # Row 2: MACD
        fig_tech.add_trace(go.Scatter(x=df.index, y=df['MACD'].squeeze(), name="MACD", line=dict(color=C_ACCENT, width=1.2)), row=2, col=1)
        fig_tech.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'].squeeze(), name="Signal", line=dict(color=C_GREEN, width=1.2)), row=2, col=1)
        macd_hist   = df['MACD_Hist'].squeeze()
        hist_colors = [C_EMERALD if v >= 0 else C_RED for v in macd_hist]
        fig_tech.add_trace(go.Bar(x=df.index, y=macd_hist, name="Histogram", marker_color=hist_colors, opacity=0.65), row=2, col=1)
        # Row 3: Stochastic
        if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
            fig_tech.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'].squeeze(), name="%K", line=dict(color=C_ACCENT, width=1.2)), row=3, col=1)
            fig_tech.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'].squeeze(), name="%D", line=dict(color=C_YELLOW, width=1.0, dash='dot')), row=3, col=1)
            fig_tech.add_hline(y=80, line_dash="dash", line_color=C_RED,    row=3, col=1)
            fig_tech.add_hline(y=20, line_dash="dash", line_color=C_EMERALD, row=3, col=1)
            fig_tech.add_hrect(y0=80, y1=100, fillcolor="rgba(255,107,107,0.04)", line_width=0, row=3, col=1)
            fig_tech.add_hrect(y0=0,  y1=20,  fillcolor="rgba(0,229,176,0.04)",  line_width=0, row=3, col=1)
        # Row 4: Williams %R
        if 'Williams_R' in df.columns:
            fig_tech.add_trace(go.Scatter(x=df.index, y=df['Williams_R'].squeeze(), name="Williams %R", line=dict(color="#adc6ff", width=1.2)), row=4, col=1)
            fig_tech.add_hline(y=-20, line_dash="dash", line_color=C_RED,    row=4, col=1)
            fig_tech.add_hline(y=-80, line_dash="dash", line_color=C_EMERALD, row=4, col=1)
            fig_tech.add_hrect(y0=-20, y1=0,    fillcolor="rgba(255,107,107,0.04)", line_width=0, row=4, col=1)
            fig_tech.add_hrect(y0=-100,y1=-80,  fillcolor="rgba(0,229,176,0.04)",  line_width=0, row=4, col=1)
        subplot_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis','yaxis')}
        fig_tech.update_layout(**subplot_layout, height=680)
        fig_tech.update_xaxes(gridcolor="#252f47", linecolor="#252f47", tickfont=dict(color=C_GREY))
        fig_tech.update_yaxes(gridcolor="#252f47", linecolor="#252f47", tickfont=dict(color=C_GREY))
        fig_tech.update_yaxes(range=[0, 100],   row=1, col=1)
        fig_tech.update_yaxes(range=[0, 100],   row=3, col=1)
        fig_tech.update_yaxes(range=[-100, 0],  row=4, col=1)
        st.plotly_chart(fig_tech, use_container_width=True)

        # ── XGBoost Model ──────────────────────────────────────────────────────
        st.markdown('<div class="model-badge">🤖 Powered by XGBoost · 28 Technical Signals (RSI · MACD · Stochastic · Williams %R · ADX · OBV · CCI + Lag Window)</div>', unsafe_allow_html=True)

        with st.expander("📖 How this assistant works — methodology & limitations", expanded=False):
            st.markdown(f"""<div style="font-family:Manrope,sans-serif;font-size:0.82rem;color:#8a8fa0;line-height:1.7;">
            <b style="color:#e4eafd;font-family:IBM Plex Mono,monospace;font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;">Feature Engineering</b><br>
            Each trading day is represented by <b style="color:#4d8eff;">20 technical indicators</b> computed from raw OHLCV data — MAs (5–200), EMA12/26, RSI, MACD, Bollinger Bands, ATR, volume ratio, momentum — plus <b style="color:#4d8eff;">{seq_len} lag closes</b> as sequential context.<br><br>
            <b style="color:#e4eafd;font-family:IBM Plex Mono,monospace;font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;">Training & Evaluation</b><br>
            Data is split <b style="color:#4d8eff;">80% train / 20% test</b> chronologically (no data leakage). XGBoost projects the next day's closing price. Quality is measured with RMSE, MAE, MAPE and R².<br><br>
            <b style="color:#ff5f5f;font-family:IBM Plex Mono,monospace;font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;">⚠ Key Limitations</b><br>
            This assistant analyses price and volume data only. It works best when combined with current news, earnings context, and your own market judgment. A single unexpected event can shift any technical outlook. <b style="color:#ff5f5f;">Not financial advice.</b>
            </div>""", unsafe_allow_html=True)

        _show_progress(3, 5, "🔢 Building feature matrix...")
        X, y = build_xgb_dataset(df, seq_len)
        _progress_placeholder.empty()

        if len(X) < 50:
            st.error(_L["not_enough_data"])
            st.stop()

        split   = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        @st.cache_resource(show_spinner=False)
        def train_xgb_cached(_X_train, _y_train, _X_test, _y_test, _n_est, _depth, _lr):
            m = XGBRegressor(n_estimators=_n_est, max_depth=_depth, learning_rate=_lr,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
            m.fit(_X_train, _y_train, eval_set=[(_X_test, _y_test)], verbose=False)
            return m

        _show_progress(4, 5, "🤖 Training XGBoost model (cached after first run)...")
        model = train_xgb_cached(X_train, y_train, X_test, y_test, n_estimators, max_depth, learning_rate)
        _progress_placeholder.empty()

        preds  = model.predict(X_test)
        actual = y_test
        rmse   = float(np.sqrt(mean_squared_error(actual, preds)))
        mae    = float(mean_absolute_error(actual, preds))
        mape   = float(np.mean(np.abs((actual - preds) / actual)) * 100)
        r2     = float(1 - np.sum((actual - preds)**2) / np.sum((actual - np.mean(actual))**2))

        # Confidence score
        r2_norm    = max(0, min(100, r2 * 100))
        mape_norm  = max(0, min(100, 100 - mape * 5))
        dir_acc    = sum(1 for i in range(1, len(actual)) if (preds[i]-actual[i-1])*(actual[i]-actual[i-1])>0) / max(len(actual)-1,1) * 100
        data_score = min(100, len(df)/2000*100)
        confidence_score = max(0, min(100, r2_norm*0.40 + mape_norm*0.30 + dir_acc*0.20 + data_score*0.10))
        last_close = float(df['Close'].squeeze().iloc[-1])

        _show_progress(5, 5, "📈 Generating signal intelligence & forecast...")
        _progress_placeholder.empty()  # clear once all steps done

        # ── Model Performance ──────────────────────────────────────────────────
        mape_label = ("🟢 Excellent" if mape<2 else "🟡 Good" if mape<5 else "🟠 Fair" if mape<10 else "🔴 Poor")
        r2_label   = ("🟢 Excellent" if r2>0.95 else "🟡 Good" if r2>0.85 else "🟠 Fair" if r2>0.70 else "🔴 Poor")
        dir_acc_label = ("🟢 Strong" if dir_acc>=60 else "🟡 Moderate" if dir_acc>=50 else "🔴 Weak")
        st.markdown(f"""
        <div class="stat-grid" style="margin-bottom:.5rem;">
          <div class="stat-card">
            <div class="stat-label">RMSE</div>
            <div class="stat-value" style="color:#adc6ff;">${rmse:.2f}</div>
            <div class="stat-sub">Root mean sq error</div>
          </div>
          <div class="stat-card" style="border-top-color:#ffd426;">
            <div class="stat-label">MAPE</div>
            <div class="stat-value" style="color:#ffd426;">{mape:.2f}%</div>
            <div class="stat-sub">{mape_label}</div>
          </div>
          <div class="stat-card" style="border-top-color:#00e5b0;">
            <div class="stat-label">R² Score</div>
            <div class="stat-value" style="color:#00e5b0;">{r2:.4f}</div>
            <div class="stat-sub">{r2_label}</div>
          </div>
          <div class="stat-card" style="border-top-color:#4d8eff;">
            <div class="stat-label">Directional Accuracy</div>
            <div class="stat-value" style="color:#4d8eff;">{dir_acc:.1f}%</div>
            <div class="stat-sub">{dir_acc_label}</div>
          </div>
        </div>
        <div style="background:#0f1727;border:1px solid #252f47;padding:.55rem 1.2rem;font-family:IBM Plex Mono,monospace;font-size:.63rem;color:#3e4558;display:flex;gap:2rem;flex-wrap:wrap;border-radius:.5rem;margin-bottom:.5rem;">
          <span>MAPE: {mape_label} · &lt;2% excellent · &lt;5% good · &lt;10% fair</span>
          <span>R²: {r2_label} · &gt;0.95 excellent · &gt;0.85 good · &gt;0.70 fair</span>
          <span>MAE: ${mae:.2f} · Dir Acc: {dir_acc:.1f}%</span>
        </div>""", unsafe_allow_html=True)

        # FIX 5: Tab orientation cue — shown once after analysis runs
        st.markdown(f"""
        <div style="background:rgba(0,229,176,0.05);border:1px solid rgba(0,229,176,0.15); border-radius:.6rem;padding:.65rem 1.2rem;margin-bottom:.5rem; display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;">
          <span style="font-family:IBM Plex Mono,monospace;font-size:0.78rem; color:#00e5b0;font-weight:700;">✓ Analysis complete — {ticker}</span>
          <span style="font-family:Manrope,sans-serif;font-size:0.78rem;color:#8a8fa0;">
            <b style="color:#e4eafd;">Dashboard</b> → signals &amp; charts ·
            <b style="color:#e4eafd;">Deep Analysis</b> → forecast &amp; backtest ·
            <b style="color:#e4eafd;">Startup Hub</b> → macro, treasury &amp; reports
          </span>
        </div>
        """, unsafe_allow_html=True)

        # Tabs
        dash_tab, port_tab, mkt_tab, deep_tab, startup_tab = st.tabs([_L["dashboard_tab"], _L["portfolio"], _L["markets"], _L["deep_analysis"], "🚀  Startup Hub"])

        # ──────────────────────────────────────────────────────────────────────
        with dash_tab:
            _dash_close = float(df["Close"].squeeze().iloc[-1])
            _dash_prev  = float(df["Close"].squeeze().iloc[-2]) if len(df)>1 else _dash_close
            _dash_chg   = _dash_close - _dash_prev
            _dash_pct   = (_dash_chg / _dash_prev * 100) if _dash_prev != 0 else 0
            _dash_sign  = "+" if _dash_chg >= 0 else ""
            _dash_color = "#00e5b0" if _dash_chg >= 0 else "#ff5f5f"
            _dash_arrow = "▲" if _dash_chg >= 0 else "▼"
            _dash_name  = POPULAR_TICKERS.get(ticker, ticker)

            # KPI row
            st.markdown(f"""
            <div class="stat-grid">
              <div class="stat-card">
                <div class="stat-label">{_L["last_close_lbl"]}</div>
                <div class="stat-value">${_dash_close:.2f}</div>
                <div class="stat-sub" style="color:{_dash_color};font-weight:700;">{_dash_arrow} {_dash_sign}{_dash_chg:.2f} ({_dash_sign}{_dash_pct:.2f}%)</div>
              </div>
              <div class="stat-card" style="border-top-color:#adc6ff;">
                <div class="stat-label">{_L["model_confidence_lbl"]}</div>
                <div class="stat-value" style="color:#adc6ff;">{confidence_score:.0f}<span style="font-size:.9rem;color:#8a8fa0;">/100</span></div>
                <div class="stat-sub">{_L["high_lbl"] if confidence_score>=80 else _L["moderate_lbl"] if confidence_score>=60 else _L["low_lbl"]}</div>
              </div>
              <div class="stat-card" style="border-top-color:#ffd426;">
                <div class="stat-label">MAPE</div>
                <div class="stat-value" style="color:#ffd426;">{mape:.2f}%</div>
                <div class="stat-sub">{mape_label}</div>
              </div>
              <div class="stat-card" style="border-top-color:#00e5b0;">
                <div class="stat-label">R² Score</div>
                <div class="stat-value" style="color:#00e5b0;">{r2:.4f}</div>
                <div class="stat-sub">{r2_label}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Actual vs Predicted chart
            st.subheader("Actual vs Projected")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(y=actual, name="Actual", line=dict(color=C_ACCENT, width=1.5), fill='tozeroy', fillcolor='rgba(77,142,255,0.05)'))
            fig1.add_trace(go.Scatter(y=preds, name="AI Projection", line=dict(color=C_EMERALD, width=1.5, dash='dot')))
            fig1.update_layout(**PLOTLY_LAYOUT, title=dict(text=f"{ticker} · AI Model Fit (Test Set)", font=dict(color=C_GREEN, size=13)), height=400)
            st.plotly_chart(fig1, use_container_width=True)

            # Feature Importance
            st.subheader("Signal Drivers")
            lag_names = [f'Lag_{i+1}' for i in range(seq_len)]
            all_feature_names = FEATURE_COLS + lag_names
            importances = model.feature_importances_
            imp_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values('importance', ascending=True).tail(20)
            fig_imp = go.Figure(go.Bar(x=imp_df['importance'], y=imp_df['feature'], orientation='h',
                marker=dict(color=imp_df['importance'], colorscale=[[0,"#0f1727"],[0.5,"#1a3050"],[1,C_ACCENT]], showscale=False)))
            fig_imp.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis"}, title=dict(text="Top 20 Signal Drivers", font=dict(color=C_GREEN, size=13)), height=480, xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Importance Score"))
            st.plotly_chart(fig_imp, use_container_width=True)

        # ──────────────────────────────────────────────────────────────────────
        with port_tab:
            st.markdown("""
            <div class="wi-header" style="margin-top:0;margin-bottom:1.2rem;">
              <div>
                <div class="wi-logo">Portfolio <span>Command Center</span></div>
                <div class="wi-sub">Monitor performance, track exposure, and manage positions in real time</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Add new holding ───────────────────────────────────────────────
            MAX_HOLDINGS = 500
            current_count = len(st.session_state.portfolio)

            with st.expander(_L["add_holding"], expanded=True):
                st.markdown(
                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:.6rem;color:#3e4558;margin-bottom:.6rem;">'
                    f'Holdings: {current_count} / {MAX_HOLDINGS}</div>',
                    unsafe_allow_html=True
                )
                with st.form(key="pt_add_form", clear_on_submit=True):
                    pa1, pa2, pa3 = st.columns(3)
                    with pa1: add_sym  = st.text_input("Ticker", placeholder="e.g. AAPL", key="pt_sym_f")
                    with pa2: add_qty  = st.number_input("Quantity", min_value=0.0001, value=1.0, step=0.5, key="pt_qty_f")
                    with pa3: add_cost = st.number_input("Avg Buy Price ($)", min_value=0.01, value=100.0, step=0.5, key="pt_cost_f")
                    submitted = st.form_submit_button(_L["add_to_portfolio"], use_container_width=True)

                if submitted:
                    add_sym = add_sym.strip().upper()
                    if not add_sym:
                        st.warning("Please enter a ticker symbol.")
                    elif len(st.session_state.portfolio) >= MAX_HOLDINGS:
                        st.warning(f"Maximum {MAX_HOLDINGS} holdings reached.")
                    else:
                        existing = [h for h in st.session_state.portfolio if h["ticker"] == add_sym]
                        if existing:
                            st.warning(_L["already_in_portfolio"].format(sym=add_sym))
                        else:
                            with st.spinner(f"Fetching live price for {add_sym}..."):
                                _q = av_get_quote(add_sym)
                                _live_px = _q["price"] if _q["price"] > 0 else add_cost
                                try:
                                    _info = get_ticker_full(add_sym)
                                    _name = _info.get("longName", add_sym)
                                    _sector = _info.get("sector", "Unknown") + " • " + _info.get("industry", "")
                                except Exception as e:
                                    logger.warning("Could not fetch ticker info for '%s': %s", add_sym, e)
                                    _name = add_sym; _sector = "Unknown"
                            _pl = (_live_px - add_cost) * add_qty
                            _pl_pct = ((_live_px - add_cost) / add_cost * 100) if add_cost > 0 else 0
                            _new_holding = {
                                "ticker": add_sym, "name": _name, "sector": _sector,
                                "qty": add_qty, "avg_cost": add_cost,
                                "current_price": _live_px, "pl": _pl, "pl_pct": _pl_pct
                            }
                            st.session_state.portfolio.append(_new_holding)
                            _sb_upsert_holding(_user_id(), _new_holding)
                            _date = pd.Timestamp.today().strftime("%b %d")
                            _hist_record = {
                                "date": _date, "type": "BUY", "ticker": add_sym,
                                "shares": add_qty, "price": add_cost,
                                "amount": -(add_qty * add_cost)
                            }
                            st.session_state.portfolio_history.insert(0, _hist_record)
                            _sb_insert_history(_user_id(), _hist_record)
                            st.success(_L["added_success"].format(sym=add_sym, price=_live_px))
                            st.rerun()

            port = st.session_state.portfolio

            if not port:
                st.info(_L["no_holdings"])
            else:
                # ── Action bar: Refresh + CSV Export ─────────────────────────
                import io as _io
                _ab1, _ab2, _ab3 = st.columns([2, 2, 4])

                with _ab1:
                    if st.button(_L["refresh_prices"], key="pt_refresh", use_container_width=True):
                        av_get_quote.clear()
                        get_ticker_full.clear()
                        for h in st.session_state.portfolio:
                            try:
                                _q = av_get_quote(h["ticker"])
                                if _q["price"] > 0:
                                    h["current_price"] = _q["price"]
                                    h["pl"]     = (_q["price"] - h["avg_cost"]) * h["qty"]
                                    h["pl_pct"] = ((_q["price"] - h["avg_cost"]) / h["avg_cost"] * 100)
                            except Exception as e:
                                logger.warning("refresh_prices: failed to update '%s': %s", h.get("ticker"), e)
                        _sb_update_prices(_user_id(), st.session_state.portfolio)
                        st.rerun()

                with _ab2:
                    # Holdings CSV
                    _csv_rows = []
                    for h in port:
                        _mv = h["qty"] * h["current_price"]
                        _iv = h["qty"] * h["avg_cost"]
                        _csv_rows.append({
                            "Ticker":            h["ticker"],
                            "Name":              h.get("name", h["ticker"]),
                            "Sector":            (h.get("sector") or "Unknown").split(" •")[0].strip(),
                            "Quantity":          h["qty"],
                            "Avg Cost ($)":      h["avg_cost"],
                            "Current Price ($)": h["current_price"],
                            "Market Value ($)":  round(_mv, 2),
                            "Invested ($)":      round(_iv, 2),
                            "P&L ($)":           round(h["pl"], 2),
                            "P&L (%)":           round(h["pl_pct"], 2),
                        })
                    _csv_buf = _io.StringIO()
                    pd.DataFrame(_csv_rows).to_csv(_csv_buf, index=False)
                    st.download_button(
                        label="⬇ Export Holdings CSV",
                        data=_csv_buf.getvalue(),
                        file_name=f"stockcast_portfolio_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="pt_csv_holdings"
                    )

                with _ab3:
                    # Transaction history CSV
                    _hist_all = st.session_state.portfolio_history
                    if _hist_all:
                        _hist_buf = _io.StringIO()
                        pd.DataFrame(_hist_all).to_csv(_hist_buf, index=False)
                        st.download_button(
                            label="⬇ Export Transaction History CSV",
                            data=_hist_buf.getvalue(),
                            file_name=f"stockcast_transactions_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="pt_csv_hist"
                        )

                # ── KPI row (now 5 metrics including Win Rate) ────────────────
                total_value    = sum(h["qty"] * h["current_price"] for h in port)
                total_invested = sum(h["qty"] * h["avg_cost"]       for h in port)
                total_pl       = total_value - total_invested
                total_pl_pct   = (total_pl / total_invested * 100) if total_invested > 0 else 0
                winners        = [h for h in port if h["pl"] >= 0]
                win_rate       = len(winners) / len(port) * 100

                p1, p2, p3, p4, p5 = st.columns(5)
                p1.metric(_L["total_value"],  f"${total_value:,.2f}")
                p2.metric(_L["total_pl"],     f"${total_pl:+,.2f}", delta=f"{total_pl_pct:+.1f}%")
                p3.metric(_L["invested"],     f"${total_invested:,.2f}")
                p4.metric(_L["holdings"],     str(len(port)))
                p5.metric("Win Rate",         f"{win_rate:.0f}%",
                          delta=f"{len(winners)}/{len(port)} profitable")

                # ── Best / Worst performer ────────────────────────────────────
                _sorted_pct = sorted(port, key=lambda h: h["pl_pct"], reverse=True)
                _best  = _sorted_pct[0]
                _worst = _sorted_pct[-1]
                _bw1, _bw2 = st.columns(2)
                with _bw1:
                    st.markdown(f"""
                    <div style="background:rgba(0,229,176,0.05);border:1px solid rgba(0,229,176,0.2); border-left:4px solid #00e5b0;padding:.8rem 1.2rem;border-radius:0 .5rem .5rem 0;">
                      <div style="font-family:Manrope,sans-serif;font-size:.7rem;letter-spacing:.14em; text-transform:uppercase;color:#00e5b0;font-weight:700;margin-bottom:.3rem;">
                        🏆 Best Performer
                      </div>
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                          <div style="font-family:IBM Plex Mono,monospace;font-size:.9rem;font-weight:700;color:#e4eafd;">{_best["ticker"]}</div>
                          <div style="font-size:.7rem;color:#8a8fa0;">{(_best.get("name") or _best["ticker"])[:30]}</div>
                        </div>
                        <div style="text-align:right;">
                          <div style="font-family:IBM Plex Mono,monospace;font-size:.9rem;font-weight:700;color:#00e5b0;">+{_best["pl_pct"]:.2f}%</div>
                          <div style="font-size:.7rem;color:#00e5b0;">+${_best["pl"]:,.2f}</div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                with _bw2:
                    _ws = "+" if _worst["pl"] >= 0 else ""
                    st.markdown(f"""
                    <div style="background:rgba(255,107,107,0.05);border:1px solid rgba(255,107,107,0.2); border-left:4px solid #ff5f5f;padding:.8rem 1.2rem;border-radius:0 .5rem .5rem 0;">
                      <div style="font-family:Manrope,sans-serif;font-size:.7rem;letter-spacing:.14em; text-transform:uppercase;color:#ff5f5f;font-weight:700;margin-bottom:.3rem;">
                        📉 Worst Performer
                      </div>
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                          <div style="font-family:IBM Plex Mono,monospace;font-size:.9rem;font-weight:700;color:#e4eafd;">{_worst["ticker"]}</div>
                          <div style="font-size:.7rem;color:#8a8fa0;">{(_worst.get("name") or _worst["ticker"])[:30]}</div>
                        </div>
                        <div style="text-align:right;">
                          <div style="font-family:IBM Plex Mono,monospace;font-size:.9rem;font-weight:700;color:#ff5f5f;">{_worst["pl_pct"]:+.2f}%</div>
                          <div style="font-size:.7rem;color:#ff5f5f;">{_ws}${abs(_worst["pl"]):,.2f}</div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                # ── P&L Over Time chart ───────────────────────────────────────
                st.subheader("📈 Portfolio P&L Over Time")
                with st.spinner("Building P&L history..."):
                    try:
                        _syms     = [h["ticker"] for h in port]
                        _qtys     = {h["ticker"]: h["qty"]      for h in port}
                        _costs    = {h["ticker"]: h["avg_cost"] for h in port}
                        _raw_hist = yf.download(_syms, period="1y", interval="1d",
                                                progress=False, auto_adjust=True)
                        _close_h  = _raw_hist["Close"] if "Close" in _raw_hist.columns else _raw_hist
                        if isinstance(_close_h.columns, pd.MultiIndex):
                            _close_h = _close_h.droplevel(0, axis=1)
                        if isinstance(_close_h, pd.Series):
                            _close_h = _close_h.to_frame(name=_syms[0])

                        _port_val  = pd.Series(0.0, index=_close_h.index)
                        _port_cost = 0.0
                        for sym in _syms:
                            if sym in _close_h.columns:
                                _port_val  += _close_h[sym].ffill() * _qtys[sym]
                                _port_cost += _costs[sym] * _qtys[sym]

                        _port_pl = _port_val - _port_cost

                        # P&L chart
                        fig_pl = go.Figure()
                        fig_pl.add_trace(go.Scatter(
                            x=_port_pl.index, y=_port_pl.values,
                            name="Unrealised P&L ($)",
                            line=dict(color="#4d8eff", width=1.8),
                            fill="tozeroy",
                            fillcolor="rgba(77,142,255,0.08)",
                        ))
                        fig_pl.add_hline(y=0, line_dash="dash",
                                         line_color="#3e4558", line_width=1)
                        _pl_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"}
                        fig_pl.update_layout(
                            **_pl_layout,
                            title=dict(text="Portfolio Unrealised P&L — Last 12 Months",
                                       font=dict(color=C_GREEN, size=12)),
                            height=300,
                            yaxis=dict(**PLOTLY_LAYOUT["yaxis"],
                                       title="P&L ($)", tickprefix="$"),
                        )
                        st.plotly_chart(fig_pl, use_container_width=True)

                        # Value vs cost basis chart
                        fig_val = go.Figure()
                        fig_val.add_trace(go.Scatter(
                            x=_port_val.index, y=_port_val.values,
                            name="Market Value",
                            line=dict(color=C_EMERALD, width=1.8),
                            fill="tozeroy", fillcolor="rgba(0,229,176,0.05)",
                        ))
                        fig_val.add_hline(
                            y=_port_cost, line_dash="dot",
                            line_color=C_YELLOW, line_width=1.2,
                            annotation_text=f"Cost Basis ${_port_cost:,.0f}",
                            annotation_font_color=C_YELLOW,
                            annotation_font_size=9,
                        )
                        _val_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"}
                        fig_val.update_layout(
                            **_val_layout,
                            title=dict(text="Portfolio Value vs Cost Basis — Last 12 Months",
                                       font=dict(color=C_GREEN, size=12)),
                            height=280,
                            yaxis=dict(**PLOTLY_LAYOUT["yaxis"],
                                       title="Value ($)", tickprefix="$"),
                        )
                        st.plotly_chart(fig_val, use_container_width=True)

                    except Exception as _e:
                        st.info(f"P&L chart unavailable: {_e}")

                # ── Holdings table ────────────────────────────────────────────
                st.subheader(_L["holdings_label"])
                for h in port:
                    _pl_col  = "#00e5b0" if h["pl"] >= 0 else "#ff5f5f"
                    _pl_sign = "+" if h["pl"] >= 0 else ""
                    hc1, hc2, hc3, hc4, hc5, hc6 = st.columns([1.2, 2, 1, 1, 1.5, 0.7])
                    hc1.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.8rem;font-weight:700;color:#4d8eff;padding:.4rem 0;">{h["ticker"]}</div>', unsafe_allow_html=True)
                    hc2.markdown(f'<div style="font-size:.75rem;color:#8a8fa0;padding:.4rem 0;">{(h["name"] or h["ticker"])[:28]}</div>', unsafe_allow_html=True)
                    hc3.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;color:#e4eafd;padding:.4rem 0;">{h["qty"]:.2f} sh</div>', unsafe_allow_html=True)
                    hc4.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;color:#e4eafd;padding:.4rem 0;">${h["current_price"]:.2f}</div>', unsafe_allow_html=True)
                    hc5.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;color:{_pl_col};font-weight:700;padding:.4rem 0;">{_pl_sign}${abs(h["pl"]):,.2f} ({_pl_sign}{h["pl_pct"]:.1f}%)</div>', unsafe_allow_html=True)
                    if hc6.button("✕", key=f"pt_del_{h['ticker']}", use_container_width=True):
                        _del_ticker = h["ticker"]
                        st.session_state.portfolio = [x for x in st.session_state.portfolio if x["ticker"] != _del_ticker]
                        _sb_delete_holding(_user_id(), _del_ticker)
                        st.rerun()

                # ── Market value bar chart ────────────────────────────────────
                st.subheader("Holdings Breakdown")
                _bar_cols = [C_EMERALD if h["pl"] >= 0 else C_RED for h in port]
                fig_bar = go.Figure(go.Bar(
                    x=[h["ticker"] for h in port],
                    y=[h["qty"] * h["current_price"] for h in port],
                    marker_color=_bar_cols,
                    text=[f'${h["qty"]*h["current_price"]:,.0f}' for h in port],
                    textposition="outside",
                    textfont=dict(color="#8a8fa0", size=9, family="IBM Plex Mono"),
                ))
                _bar_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"}
                fig_bar.update_layout(
                    **_bar_layout,
                    title=dict(text="Market Value per Holding",
                               font=dict(color=C_GREEN, size=12)),
                    height=260,
                    yaxis=dict(**PLOTLY_LAYOUT["yaxis"],
                               title="Market Value ($)", tickprefix="$"),
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # ── Sector donut + Recent Activity ────────────────────────────
                sc1, sc2 = st.columns([1, 1])
                with sc1:
                    st.subheader(_L["sector_allocation"])
                    sector_map = {}
                    for h in port:
                        sec = (h["sector"] or "Unknown").split(" •")[0].strip()
                        sector_map[sec] = sector_map.get(sec, 0) + h["qty"] * h["current_price"]
                    sec_colors = {
                        "Technology": "#4d8eff", "Consumer Cyclical": "#ffd426",
                        "Financials": "#adc6ff", "Energy": "#ff9f40",
                        "Healthcare": "#00e5b0", "Unknown": "#8a8fa0",
                    }
                    fig_sector = go.Figure(go.Pie(
                        labels=list(sector_map.keys()),
                        values=list(sector_map.values()),
                        hole=0.6,
                        marker_colors=[sec_colors.get(s, "#8a8fa0") for s in sector_map.keys()],
                        textfont_size=10, textfont_color="#e4eafd",
                    ))
                    fig_sector.update_layout(
                        **PLOTLY_LAYOUT, height=260, showlegend=True,
                        annotations=[dict(
                            text=f"{len(sector_map)}<br><span style='font-size:10px'>Sectors</span>",
                            x=0.5, y=0.5, font_size=20, showarrow=False,
                            font_color="#e4eafd")])
                    st.plotly_chart(fig_sector, use_container_width=True)

                with sc2:
                    st.subheader(_L["recent_activity"])
                    hist = st.session_state.portfolio_history
                    if not hist:
                        st.markdown(f'<div style="font-size:.78rem;color:#3e4558;padding:.5rem 0;">{_L["no_transactions"]}</div>', unsafe_allow_html=True)
                    for a in hist[:10]:
                        type_color = {"BUY": "#4d8eff", "SELL": "#00e5b0", "DIVIDEND": "#ffd426"}.get(a["type"], "#8a8fa0")
                        amt_str    = f'+${a["amount"]:,.2f}' if a["amount"] >= 0 else f'-${abs(a["amount"]):,.2f}'
                        desc       = f'{a["shares"]} shares @ ${a["price"]:.2f}' if a.get("shares") and a.get("price") else a["ticker"]
                        st.markdown(f"""
                        <div style="display:flex;gap:.8rem;padding:.7rem 0;border-bottom:1px solid #252f47;align-items:center;">
                          <div style="width:2rem;height:2rem;border-radius:50%;background:rgba({','.join(str(int(type_color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.15); color:{type_color};display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:.8rem;font-weight:700;">
                            {"B" if a["type"]=="BUY" else "S" if a["type"]=="SELL" else "D"}
                          </div>
                          <div style="flex:1;">
                            <div style="display:flex;justify-content:space-between;">
                              <span style="font-size:.8rem;font-weight:700;color:#e4eafd;font-family:Manrope,sans-serif;">{a["type"]} {a["ticker"]}</span>
                              <span style="font-size:.65rem;color:#8a8fa0;font-family:IBM Plex Mono,monospace;">{a["date"].upper()}</span>
                            </div>
                            <div style="font-size:.7rem;color:#8a8fa0;margin-top:.1rem;">{desc}</div>
                            <div style="font-size:.7rem;font-weight:700;color:{type_color};margin-top:.1rem;font-family:IBM Plex Mono,monospace;">{amt_str}</div>
                          </div>
                        </div>""", unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────────────────
        with mkt_tab:
            st.markdown(f"""
            <div style="margin-bottom:1.2rem;">
              <div style="font-family:Manrope,sans-serif;font-size:2rem;font-weight:800;letter-spacing:-.02em;color:#e4eafd;">Market <span style="color:#4d8eff;">Intelligence</span></div>
              <div style="font-size:.82rem;color:#8a8fa0;margin-top:.3rem;line-height:1.6;">Live global performance, sector heatmap, and institutional sentiment — updated every 2 minutes.</div>
            </div>
            """, unsafe_allow_html=True)

            # Market index cards — st.columns (2×2 reliable layout)
            with st.spinner("Loading live market data..."):
                mkt_data = get_live_market_indices()
            _mc1, _mc2 = st.columns(2)
            _mc3, _mc4 = st.columns(2)
            for _mcol, (name, price, chg, col) in zip([_mc1, _mc2, _mc3, _mc4], mkt_data):
                with _mcol:
                    st.markdown(
                        f'<div style="background:linear-gradient(145deg,#0f1727,#141d30);' f'border:1px solid #252f47;border-top:2px solid {col};' f'padding:1rem 1.1rem;border-radius:.6rem;margin-bottom:.5rem;">'
                        f'<div style="font-size:.7rem;font-weight:700;color:#8a8fa0;' f'letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem;">{name}</div>'
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.2rem;' f'font-weight:700;color:#e4eafd;line-height:1.1;">{price}</div>'
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:.7rem;' f'color:{col};font-weight:700;margin-top:.25rem;">{chg}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            ms1, ms2 = st.columns([2, 1])
            with ms1:
                st.subheader("Sector Heat Map · Live")
                sectors = get_live_sector_heatmap()
                # Render sectors as 2-column pairs
                for i in range(0, len(sectors), 2):
                    _sc1, _sc2 = st.columns(2)
                    for _scol, _si in [(_sc1, i), (_sc2, i+1)]:
                        if _si < len(sectors):
                            _sname, _schg, _scol_color = sectors[_si]
                            with _scol:
                                st.markdown(
                                    f'<div style="background:#0f1727;border:1px solid #252f47;' f'border-left:2px solid {_scol_color};padding:.6rem .8rem;' f'border-radius:0 .5rem .5rem 0;margin-bottom:.4rem;">'
                                    f'<div style="font-size:.7rem;font-weight:700;color:#8a8fa0;' f'text-transform:uppercase;margin-bottom:.2rem;">{_sname}</div>'
                                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:.85rem;' f'font-weight:700;color:{_scol_color};">{_schg}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

            with ms2:
                st.subheader("Fear & Greed Index · Live")
                _fg = get_fear_greed_index()
                if _fg:
                    _fg_score = _fg["score"]
                    _fg_label = _fg["rating"]
                else:
                    _fg_score = 50; _fg_label = "Neutral"
                _fg_color = "#ff5f5f" if _fg_score < 30 else "#ffd426" if _fg_score < 55 else "#00e5b0"
                _fg_pct   = f"{max(2, min(98, _fg_score)):.0f}%"
                st.markdown(f"""
                <div style="background:linear-gradient(145deg,#0f1727,#141d30);border:1px solid #252f47; padding:1.4rem;text-align:center;border-radius:.5rem;">
                  <div style="font-family:IBM Plex Mono,monospace;font-size:2.8rem;font-weight:800;color:{_fg_color};">{_fg_score:.0f}</div>
                  <div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{_fg_color};margin-bottom:.8rem;">{_fg_label}</div>
                  <div style="height:6px;background:linear-gradient(90deg,#ff5f5f,#ff9f40,#ffd426,#00e5b0);border-radius:3px;position:relative;">
                    <div style="position:absolute;top:-10px;left:{_fg_pct};transform:translateX(-50%);width:2px;height:26px;background:#e4eafd;border-radius:1px;"></div>
                  </div>
                  <div style="display:flex;justify-content:space-between;margin-top:.5rem;font-size:.7rem;color:#3e4558;font-weight:700;text-transform:uppercase;">
                    <span>Fear</span><span>Neutral</span><span>Greed</span>
                  </div>
                  <div style="margin-top:1rem;padding:.75rem;background:rgba(77,142,255,0.06);border-radius:.5rem;">
                    <div style="font-size:.7rem;color:#8a8fa0;line-height:1.5;">{"Live CNN Fear & Greed data." if _fg else "Could not fetch live data."} Score: <b style="color:{_fg_color};">{_fg_score:.0f}/100</b> — <b style="color:{_fg_color};">{_fg_label}</b>.</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────────────────
        with deep_tab:
            # Price alert
            if alert_price > 0:
                diff = alert_price - last_close
                if last_close >= alert_price:
                    st.markdown(f'<div class="alert-box">🔔 {ticker} at ${last_close:.2f} — {_L["at_above_target"]} ${alert_price:.2f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-box">🔔 {ticker} at ${last_close:.2f} — ${diff:.2f} {_L["below_target"]} ${alert_price:.2f}</div>', unsafe_allow_html=True)

            # Confidence score bar
            conf_color = "#00e5b0" if confidence_score>=80 else "#ffd426" if confidence_score>=60 else "#ff5f5f"
            conf_label = _L["high_confidence"] if confidence_score>=80 else _L["moderate_confidence"] if confidence_score>=60 else _L["low_confidence"]
            filled = int(confidence_score / 5)
            bar_html = "".join(f'<span style="display:inline-block;width:18px;height:10px;margin-right:2px;background:{conf_color};opacity:{1.0 if i<filled else 0.1};border-radius:1px;"></span>' for i in range(20))
            st.markdown(f"""
            <div style="background:#0f1727;border:1px solid #252f47;border-left:3px solid {conf_color}; padding:1.2rem 1.6rem;margin:1rem 0;border-radius:0 .5rem .5rem 0;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;">
                <div>
                  <div style="font-family:Manrope,sans-serif;font-size:.6rem;letter-spacing:.16em;text-transform:uppercase;color:#3e4558;margin-bottom:.3rem;font-weight:700;">{_L["model_conf_score"]}</div>
                  <div style="font-family:IBM Plex Mono,monospace;font-size:2.2rem;font-weight:700;color:{conf_color};">{confidence_score:.0f}<span style="font-size:1rem;color:#8a8fa0;">/100</span></div>
                  <div style="font-family:Manrope,sans-serif;font-size:.62rem;letter-spacing:.14em;color:{conf_color};margin-top:.3rem;font-weight:700;">{conf_label}</div>
                </div>
                <div style="flex:1;min-width:220px;">
                  <div style="margin-bottom:.6rem;">{bar_html}</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:.3rem .8rem;font-family:IBM Plex Mono,monospace;font-size:.63rem;color:#3e4558;">
                    <span>{_L["r2_fit"]} <b style="color:#8a8fa0;">{r2_norm:.0f}/100</b> <span style="color:#252f47;">(×0.40)</span></span>
                    <span>{_L["mape_accuracy"]} <b style="color:#8a8fa0;">{mape_norm:.0f}/100</b> <span style="color:#252f47;">(×0.30)</span></span>
                    <span>{_L["directional_acc"]} <b style="color:#8a8fa0;">{dir_acc:.0f}/100</b> <span style="color:#252f47;">(×0.20)</span></span>
                    <span>{_L["data_volume"]} <b style="color:#8a8fa0;">{data_score:.0f}/100</b> <span style="color:#252f47;">(×0.10)</span></span>
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── PHASE 3 + 4: SIGNAL INTELLIGENCE · TRUST SYSTEM · ALERTS ────────
            st.subheader(_L["signal_intelligence"])
            composite    = compute_composite_signal(df, last_close, preds[-1], preds, actual)
            verdict      = composite['verdict']
            verdict_short= composite['verdict_short']
            total_score  = composite['total_score']
            xgb_pct      = composite['xgb_pct']
            stop_loss    = composite['stop_loss']
            take_profit  = composite['take_profit']
            risk_reward  = composite['risk_reward']
            rsi_val      = composite['rsi']
            vol_ratio    = composite['vol_ratio']
            atr_val      = composite['atr']
            sigs         = composite['signals']
            _stoch_k_sig = composite.get('stoch_k', 50.0)
            _adx_sig     = composite.get('adx', 25.0)

            # ── PHASE 3: Confidence score ────────────────────────────────────
            # Normalise |score| to 0-100 (max theoretical ≈ ±135)
            _sig_conf = min(100, int(abs(total_score) / 135 * 100))
            # Penalise when AI direction and composite score disagree
            if (total_score > 0 and xgb_pct < 0) or (total_score < 0 and xgb_pct > 0):
                _sig_conf = int(_sig_conf * 0.7)
            _conf_color = "#00e5b0" if _sig_conf >= 70 else "#ffd426" if _sig_conf >= 45 else "#ff5f5f"
            _conf_label = "HIGH CONFIDENCE" if _sig_conf >= 70 else "MODERATE CONFIDENCE" if _sig_conf >= 45 else "LOW CONFIDENCE"

            # ── PHASE 3: Signal reasons (2–3 plain-English bullets) ──────────
            _direction = "upward" if xgb_pct >= 0 else "downward"
            _sig_reasons = [
                f"AI model projects {_direction} move of {abs(xgb_pct):.2f}% (XGBoost next-day forecast)"
            ]
            if rsi_val < 30:
                _sig_reasons.append(f"RSI at {rsi_val:.1f} — deeply oversold, potential reversal zone")
            elif rsi_val > 70:
                _sig_reasons.append(f"RSI at {rsi_val:.1f} — overbought, elevated reversal risk")
            else:
                _mom = "positive" if total_score > 0 else "negative"
                _sig_reasons.append(f"RSI at {rsi_val:.1f} (neutral zone) with {_mom} price momentum")
            if _adx_sig > 25:
                _trend_word = "bullish" if total_score > 0 else "bearish"
                _sig_reasons.append(f"ADX at {_adx_sig:.1f} confirms strong {_trend_word} trend (ADX > 25 = trending market)")
            elif _stoch_k_sig < 20:
                _sig_reasons.append(f"Stochastic %K at {_stoch_k_sig:.1f} — oversold, watch for bullish crossover")
            elif _stoch_k_sig > 80:
                _sig_reasons.append(f"Stochastic %K at {_stoch_k_sig:.1f} — overbought, momentum may stall")
            else:
                _macd_cross = sigs.get("MACD Cross", ("HOLD", 0, 0, "neutral"))
                _sig_reasons.append(f"MACD histogram shows {_macd_cross[0].lower()} crossover signal")

            # ── PHASE 3: Trust & Credibility panel ──────────────────────────
            _vs_color = {"BUY": "#00e5b0", "SELL": "#ff5f5f", "HOLD": "#ffd426"}.get(verdict_short, "#8a8fa0")
            _vs_rgb   = ",".join(str(int(_vs_color.lstrip("#")[i:i+2], 16)) for i in (0, 2, 4))
            _conf_segs = "".join(
                f'<span style="display:inline-block;width:16px;height:8px;margin-right:2px;' f'border-radius:1px;background:{_conf_color};' f'opacity:{1.0 if i < int(_sig_conf / 5) else 0.1};"></span>'
                for i in range(20)
            )
            _reason_html = "".join(
                f'<div style="display:flex;align-items:flex-start;gap:.5rem;margin-bottom:.35rem;">'
                f'<span style="color:{_vs_color};flex-shrink:0;margin-top:.1rem;">▸</span>'
                f'<span style="font-family:Manrope,sans-serif;font-size:.79rem;' f'color:#b8c4d8;line-height:1.55;">{_r}</span></div>'
                for _r in _sig_reasons
            )
            _trust_ts = pd.Timestamp.utcnow().strftime("%b %d, %Y · %H:%M UTC")
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,#0d1524,#07101e); border:1px solid #1e2d45;border-left:4px solid {_vs_color}; border-radius:0 .75rem .75rem 0;padding:1.3rem 1.6rem;margin:.6rem 0 1rem;">
              <div style="display:flex;align-items:center;justify-content:space-between; flex-wrap:wrap;gap:.7rem;margin-bottom:.9rem;">
                <div>
                  <div style="font-family:Manrope,sans-serif;font-size:.6rem;letter-spacing:.18em; text-transform:uppercase;color:#3d4760;margin-bottom:.3rem;font-weight:700;">
                    {_L["composite_signal"]} · {ticker}
                  </div>
                  <span style="background:rgba({_vs_rgb},0.14);border:1.5px solid {_vs_color}; color:{_vs_color};font-family:IBM Plex Mono,monospace; font-size:1.4rem;font-weight:800;padding:.35rem 1.1rem; border-radius:.45rem;letter-spacing:.12em;">{verdict}</span>
                </div>
                <div style="text-align:right;">
                  <div style="font-family:IBM Plex Mono,monospace;font-size:2rem; font-weight:700;color:{_conf_color};">{_sig_conf}%</div>
                  <div style="font-family:Manrope,sans-serif;font-size:.62rem;letter-spacing:.13em; text-transform:uppercase;color:{_conf_color};font-weight:700;">{_conf_label}</div>
                </div>
              </div>
              <div style="margin-bottom:1rem;">{_conf_segs}</div>
              <div style="font-family:Manrope,sans-serif;font-size:.6rem;letter-spacing:.14em; text-transform:uppercase;color:#3d4760;margin-bottom:.5rem;font-weight:700;">
                Signal Explanation
              </div>
              {_reason_html}
              <div style="display:flex;align-items:center;justify-content:space-between; flex-wrap:wrap;gap:.5rem;margin-top:1rem;padding-top:.8rem; border-top:1px solid #1e2d45;">
                <span style="font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#3d4760;letter-spacing:.06em;">
                  Data source: <b style="color:#8a8fa0;">Yahoo Finance</b>
                  &nbsp;·&nbsp; Fetched: <b style="color:#8a8fa0;">{_trust_ts}</b>
                </span>
                <span style="display:inline-flex;align-items:center;gap:.4rem; background:rgba(255,87,87,0.07);border:1px solid rgba(255,87,87,0.2); border-radius:2rem;padding:.22rem .75rem;font-family:IBM Plex Mono,monospace; font-size:.63rem;color:rgba(255,87,87,0.7);">⚠ Not financial advice</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── PHASE 4: ALERTS SYSTEM ───────────────────────────────────────
            # Signal-change detection (stored in session_state)
            if "alert_signals" not in st.session_state:
                st.session_state.alert_signals = {}
            _prev_sig = st.session_state.alert_signals.get(ticker)

            if alert_on_signal_change and _prev_sig is not None and _prev_sig != verdict_short:
                _ac      = {"BUY": "#00e5b0", "SELL": "#ff5f5f"}.get(verdict_short, "#ffd426")
                _ac_rgb  = ",".join(str(int(_ac.lstrip("#")[i:i+2], 16)) for i in (0, 2, 4))
                st.markdown(f"""
                <div style="background:rgba({_ac_rgb},0.08);border:1px solid {_ac}; border-left:5px solid {_ac};border-radius:0 .6rem .6rem 0; padding:.85rem 1.4rem;margin:.4rem 0 .8rem; animation:flash-in .4s ease;">
                  <div style="font-family:IBM Plex Mono,monospace;font-size:.9rem; font-weight:800;color:{_ac};">
                    🔔&nbsp; ALERT — {ticker} signal changed:
                    &nbsp;<span style="opacity:.55">{_prev_sig}</span>
                    &nbsp;→&nbsp;<b>{verdict_short}</b>
                  </div>
                  <div style="font-family:Manrope,sans-serif;font-size:.78rem; color:#8a8fa0;margin-top:.3rem;line-height:1.5;">
                    Confidence: <b style="color:{_ac};">{_sig_conf}%</b>
                    &nbsp;·&nbsp; AI Forecast: <b style="color:{_ac};">{xgb_pct:+.2f}%</b>
                    &nbsp;·&nbsp; Score: <b style="color:{_ac};">{total_score:+.0f} / ±135</b>
                    &nbsp;·&nbsp; Price: <b style="color:#e4eafd;">${last_close:.2f}</b>
                  </div>
                </div>
                <style>
                  @keyframes flash-in {{
                    from {{ opacity:0; transform:translateX(-8px); }}
                    to   {{ opacity:1; transform:translateX(0); }}
                  }}
                </style>
                <script>if(window.SCNotify)SCNotify.signal("{ticker}","{verdict_short}");</script>
                """, unsafe_allow_html=True)

            # Price-target alert
            if alert_price > 0 and last_close > 0:
                _diff = last_close - alert_price
                if last_close >= alert_price:
                    st.markdown(f"""
                    <div style="background:rgba(0,229,176,0.07);border:1px solid #00e5b0; border-left:4px solid #00e5b0;padding:.7rem 1.2rem;margin:.3rem 0; border-radius:0 .5rem .5rem 0;">
                      <span style="font-family:IBM Plex Mono,monospace;font-size:.8rem; color:#00e5b0;font-weight:700;">
                        🎯 {ticker} at ${last_close:.2f} — AT or ABOVE your target of ${alert_price:.2f}
                      </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:rgba(77,142,255,0.06);border:1px solid #4d8eff; border-left:4px solid #4d8eff;padding:.7rem 1.2rem;margin:.3rem 0; border-radius:0 .5rem .5rem 0;">
                      <span style="font-family:IBM Plex Mono,monospace;font-size:.78rem;color:#adc6ff;">
                        🔔 {ticker} at ${last_close:.2f} — ${abs(_diff):.2f} below target of ${alert_price:.2f}
                      </span>
                    </div>
                    """, unsafe_allow_html=True)

            # Persist current signal for next run
            st.session_state.alert_signals[ticker] = verdict_short

            verdict_css = 'sell' if verdict_short=='SELL' else 'hold' if verdict_short=='HOLD' else ''
            sign = '+' if xgb_pct>=0 else ''
            score_color = '#00e5b0' if total_score>0 else '#ff5f5f' if total_score<0 else '#ffd426'
            rr_color = 'positive' if risk_reward>=1.5 else 'negative' if risk_reward<1 else 'neutral'
            _chs  = "buy" if verdict_short=="BUY" else "sell" if verdict_short=="SELL" else "hold"
            _rsic = "sell" if rsi_val>70 else "buy" if rsi_val<30 else "hold"
            _rsil = "Overbought" if rsi_val>70 else "Oversold" if rsi_val<30 else "Neutral RSI"
            _trc  = "buy" if xgb_pct>=0 else "sell"
            _trl  = f"{'Up' if xgb_pct>=0 else 'Down'} {abs(xgb_pct):.1f}%"

            _stoch_k = composite.get('stoch_k', 50.0)
            _will_r  = composite.get('williams_r', -50.0)
            _adx_val = composite.get('adx', 25.0)
            _stoch_c = "sell" if _stoch_k > 80 else "buy" if _stoch_k < 20 else "hold"
            _will_c  = "sell" if _will_r > -20 else "buy" if _will_r < -80 else "hold"
            _adx_lbl = "Strong Trend" if _adx_val > 25 else "Weak/Ranging"
            _adx_c   = "buy" if (_adx_val > 25 and xgb_pct > 0) else "sell" if (_adx_val > 25 and xgb_pct < 0) else "hold"

            st.markdown(f"""
            <div class="chip-group" style="margin-bottom:.75rem;flex-wrap:wrap;gap:.35rem;">
              <span class="chip {_chs} dot" style="font-size:.68rem;padding:.35rem .9rem;">{verdict_short}</span>
              <span class="chip {_rsic} dot">RSI {rsi_val:.0f} · {_rsil}</span>
              <span class="chip {_trc} dot">{_trl} Outlook</span>
              <span class="chip {_stoch_c} dot">Stoch {_stoch_k:.0f}</span>
              <span class="chip {_will_c} dot">W%R {_will_r:.0f}</span>
              <span class="chip {_adx_c} dot">ADX {_adx_val:.0f} · {_adx_lbl}</span>
              <span class="chip live dot">Live</span>
              <span class="chip ai dot">XGBoost</span>
            </div>
            <div class="signal-panel">
              <div class="signal-main {verdict_css}">
                <div class="signal-lbl">{_L["composite_signal"]}</div>
                <div class="signal-action {verdict_css}">{verdict}</div>
                <div class="signal-pct">{sign}{xgb_pct:.2f}% {_L["forecast_lbl"]}</div>
                <div class="signal-lbl" style="margin-top:8px;">{_L["score_lbl"]}: <span style="color:{score_color};font-size:.9rem;font-weight:800;">{total_score:+.0f}</span> / ±135</div>
              </div>
              <div class="signal-details">
                <div class="sig-card positive">
                  <div class="sig-lbl">{_L["take_profit_lbl"]}</div>
                  <div class="sig-val">${take_profit:.2f}</div>
                  <div class="sig-sub">+{((take_profit-last_close)/last_close*100):.1f}% · 2.5× ATR</div>
                </div>
                <div class="sig-card negative">
                  <div class="sig-lbl">{_L["stop_loss_lbl"]}</div>
                  <div class="sig-val">${stop_loss:.2f}</div>
                  <div class="sig-sub">{((stop_loss-last_close)/last_close*100):.1f}% · 1.5× ATR</div>
                </div>
                <div class="sig-card {rr_color}">
                  <div class="sig-lbl">{_L["risk_reward_lbl"]}</div>
                  <div class="sig-val">{risk_reward:.2f}×</div>
                  <div class="sig-sub">{_L["favorable"] if risk_reward>=1.5 else _L["marginal"] if risk_reward>=1 else _L["unfavorable"]}</div>
                </div>
                <div class="sig-card {'positive' if rsi_val<50 else 'negative'}">
                  <div class="sig-lbl">{_L["rsi_lbl"]}</div>
                  <div class="sig-val">{rsi_val:.1f}</div>
                  <div class="sig-sub">{_L["oversold_zone"] if rsi_val<30 else _L["overbought_zone"] if rsi_val>70 else _L["neutral_zone"]}</div>
                </div>
                <div class="sig-card {'positive' if _stoch_k < 50 else 'negative'}">
                  <div class="sig-lbl">Stochastic %K</div>
                  <div class="sig-val">{_stoch_k:.1f}</div>
                  <div class="sig-sub">{"Oversold Zone" if _stoch_k<20 else "Overbought Zone" if _stoch_k>80 else "Neutral"}</div>
                </div>
                <div class="sig-card {'positive' if _will_r < -50 else 'negative'}">
                  <div class="sig-lbl">Williams %R</div>
                  <div class="sig-val">{_will_r:.1f}</div>
                  <div class="sig-sub">{"Oversold Zone" if _will_r<-80 else "Overbought Zone" if _will_r>-20 else "Neutral"}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Composite meter
            st.markdown(f'<div class="composite-meter"><div class="meter-title">{_L["factor_breakdown"]}</div>', unsafe_allow_html=True)
            for sig_name, (sig_action, sig_score, sig_val, sig_pol) in sigs.items():
                bar_width = min(100, abs(sig_score))
                st.markdown(f"""
                <div class="sir">
                  <span class="sir-label">{sig_name}</span>
                  <div class="sir-bar-bg"><div class="sir-bar {sig_pol}" style="width:{bar_width}%;"></div></div>
                  <span class="sir-val">{sig_val:.2f}</span>
                  <span class="sir-sig {sig_action.lower()}">{sig_action}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Trust & credibility strip ─────────────────────────────────────
            st.markdown(f"""
            <div style="display:flex;flex-wrap:wrap;gap:.5rem;align-items:center; padding:.55rem 1rem;background:#080e1c;border:1px solid #1a2236; border-radius:.5rem;margin:.6rem 0 1.2rem;">
              <span style="font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#3e4558;font-weight:700;text-transform:uppercase;letter-spacing:.1em;">Data sources</span>
              <span style="font-size:.6rem;color:#3e4558;">·</span>
              <span style="font-family:IBM Plex Mono,monospace;font-size:.6rem;color:#8a8fa0;">Yahoo Finance (OHLCV)</span>
              <span style="font-size:.6rem;color:#3e4558;">·</span>
              <span style="font-family:IBM Plex Mono,monospace;font-size:.6rem;color:#8a8fa0;">yfinance API (Fundamentals)</span>
              <span style="font-size:.6rem;color:#3e4558;">·</span>
              <span style="font-family:IBM Plex Mono,monospace;font-size:.6rem;color:#8a8fa0;">CNN Fear &amp; Greed (Sentiment)</span>
              <span style="font-size:.6rem;color:#3e4558;">·</span>
              <span style="font-family:IBM Plex Mono,monospace;font-size:.6rem;color:#ffd426;">⚠ Not financial advice</span>
              <span style="margin-left:auto;font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#252f47;">9 signals · XGBoost · ATR-scaled TP/SL</span>
            </div>
            """, unsafe_allow_html=True)

            # ── SIGNAL ENGINE ──────────────────────────────────────────
            _se_sentiment = float(np.clip(total_score / 135.0, -1.0, 1.0))
            st.markdown(
                "<div style='margin:.5rem 0 .2rem;font-family:Manrope,sans-serif;"
                "font-size:.6rem;font-weight:800;letter-spacing:.18em;"
                "text-transform:uppercase;color:#3d4760;'>▸ AI SIGNAL ENGINE</div>",
                unsafe_allow_html=True,
            )
            try:
                _se_signal_result = run_signal_ui(df, _se_sentiment, ticker=ticker)
                st.session_state[f"se_signal_{ticker}"] = _se_signal_result
            except Exception as _se_err:
                st.warning(f"Signal Engine unavailable: {_se_err}")
            # ───────────────────────────────────────────────────────────

            with st.expander("📊 Fundamentals & Valuation — " + ticker, expanded=False):
                with st.spinner("Loading fundamental data..."):
                    _fund = get_fundamentals_rich(ticker)
                if not _fund:
                    st.info("Fundamental data unavailable for this ticker.")
                else:
                    _f1, _f2, _f3 = st.columns(3)

                    def _fval(v, fmt="{:.2f}", suffix="", prefix="", fallback="—"):
                        if v is None: return fallback
                        try: return prefix + fmt.format(v) + suffix
                        except: return fallback

                    def _fpct(v, fallback="—"):
                        if v is None: return fallback
                        return f"{v*100:+.1f}%"

                    def _fcolor(v, good_positive=True):
                        if v is None: return "#8a8fa0"
                        return ("#00e5b0" if v > 0 else "#ff5f5f") if good_positive else ("#ff5f5f" if v > 0 else "#00e5b0")

                    # Column 1: Valuation multiples
                    with _f1:
                        st.markdown(f"""
                        <div style="background:#0f1727;border:1px solid #252f47;border-top:2px solid #4d8eff;border-radius:.75rem;padding:1.2rem 1.4rem;">
                          <div style="font-size:.7rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;color:#4d8eff;margin-bottom:.9rem;">Valuation Multiples</div>
                          {"".join(f'<div style="display:flex;justify-content:space-between;padding:.3rem 0;border-bottom:1px solid #1e2740;font-family:IBM Plex Mono,monospace;font-size:.7rem;"><span style="color:#3e4558;">{lbl}</span><span style="color:{vc};">{val}</span></div>' for lbl,val,vc in [
                            ("P/E (TTM)",  _fval(_fund.get("pe_trailing"), "{:.1f}×"),
                             "#ffd426" if _fund.get("pe_trailing") and _fund["pe_trailing"]<25 else "#ff5f5f" if _fund.get("pe_trailing") else "#8a8fa0"),
                            ("P/E (Fwd)",  _fval(_fund.get("pe_forward"), "{:.1f}×"), "#e4eafd"),
                            ("P/B",        _fval(_fund.get("pb"),         "{:.2f}×"), "#e4eafd"),
                            ("P/S (TTM)",  _fval(_fund.get("ps_ttm"),     "{:.2f}×"), "#e4eafd"),
                            ("PEG Ratio",  _fval(_fund.get("peg"),        "{:.2f}"),
                             "#00e5b0" if _fund.get("peg") and _fund["peg"]<1.5 else "#ffd426" if _fund.get("peg") else "#8a8fa0"),
                            ("EV/EBITDA",  _fval(_fund.get("ev_ebitda"),  "{:.1f}×"), "#e4eafd"),
                          ])}
                        </div>""", unsafe_allow_html=True)

                    # Column 2: Growth & Margins
                    with _f2:
                        st.markdown(f"""
                        <div style="background:#0f1727;border:1px solid #252f47;border-top:2px solid #00e5b0;border-radius:.75rem;padding:1.2rem 1.4rem;">
                          <div style="font-size:.7rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;color:#00e5b0;margin-bottom:.9rem;">Growth &amp; Margins</div>
                          {"".join(f'<div style="display:flex;justify-content:space-between;padding:.3rem 0;border-bottom:1px solid #1e2740;font-family:IBM Plex Mono,monospace;font-size:.7rem;"><span style="color:#3e4558;">{lbl}</span><span style="color:{vc};">{val}</span></div>' for lbl,val,vc in [
                            ("Rev Growth YoY",  _fpct(_fund.get("rev_growth_yoy")),  _fcolor(_fund.get("rev_growth_yoy"))),
                            ("EPS Growth YoY",  _fpct(_fund.get("earn_growth_yoy")), _fcolor(_fund.get("earn_growth_yoy"))),
                            ("EPS (TTM)",       _fval(_fund.get("eps_ttm"), "${:.2f}"), "#e4eafd"),
                            ("EPS (Fwd)",       _fval(_fund.get("eps_fwd"), "${:.2f}"), "#adc6ff"),
                            ("Gross Margin",    _fpct(_fund.get("gross_margin")),     "#e4eafd"),
                            ("Op Margin",       _fpct(_fund.get("op_margin")),        _fcolor(_fund.get("op_margin"))),
                            ("Net Margin",      _fpct(_fund.get("profit_margin")),    _fcolor(_fund.get("profit_margin"))),
                            ("ROE",             _fpct(_fund.get("roe")),              _fcolor(_fund.get("roe"))),
                          ])}
                        </div>""", unsafe_allow_html=True)

                    # Column 3: Analyst targets + sentiment
                    with _f3:
                        _rec = (_fund.get("recommendation") or "").replace("_", " ").title()
                        _rec_color = "#00e5b0" if "buy" in _rec.lower() else "#ff5f5f" if "sell" in _rec.lower() else "#ffd426"
                        _upside = _fund.get("upside_pct")
                        _up_color = "#00e5b0" if (_upside or 0) > 0 else "#ff5f5f"
                        st.markdown(f"""
                        <div style="background:#0f1727;border:1px solid #252f47;border-top:2px solid #ffd426;border-radius:.75rem;padding:1.2rem 1.4rem;">
                          <div style="font-size:.7rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;color:#ffd426;margin-bottom:.9rem;">Analyst Consensus · {_fund.get("num_analysts",0)} analysts</div>
                          {"".join(f'<div style="display:flex;justify-content:space-between;padding:.3rem 0;border-bottom:1px solid #1e2740;font-family:IBM Plex Mono,monospace;font-size:.7rem;"><span style="color:#3e4558;">{lbl}</span><span style="color:{vc};">{val}</span></div>' for lbl,val,vc in [
                            ("Consensus",    _rec or "—",                                              _rec_color),
                            ("Price Target", _fval(_fund.get("target_mean"),"${:.2f}"),               "#e4eafd"),
                            ("Target High",  _fval(_fund.get("target_high"),"${:.2f}"),               "#00e5b0"),
                            ("Target Low",   _fval(_fund.get("target_low"), "${:.2f}"),               "#ff5f5f"),
                            ("Upside/Down",  (f"{_upside:+.1f}%" if _upside is not None else "—"),   _up_color),
                            ("Beta",         _fval(_fund.get("beta"),       "{:.2f}"),                "#e4eafd"),
                            ("Short %Float", _fpct(_fund.get("short_pct_float")),                     "#ffd426" if (_fund.get("short_pct_float") or 0) > 0.1 else "#e4eafd"),
                            ("Insider %",    _fpct(_fund.get("insider_pct")),                         "#e4eafd"),
                          ])}
                        </div>""", unsafe_allow_html=True)

                    # Company description
                    if _fund.get("description"):
                        st.markdown(f"""
                        <div style="margin-top:.8rem;background:rgba(77,142,255,0.04);border:1px solid rgba(77,142,255,0.12); border-radius:.5rem;padding:.9rem 1.2rem;">
                          <div style="font-size:.7rem;font-weight:700;color:#4d8eff;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.4rem;">
                            About · {_fund.get("name",ticker)}
                          </div>
                          <div style="font-family:Manrope,sans-serif;font-size:.78rem;color:#8a8fa0;line-height:1.65;">
                            {_fund["description"]}{"..." if len(_fund.get("description",""))>=400 else ""}
                          </div>
                          <div style="font-size:.7rem;color:#3e4558;margin-top:.4rem;">{_fund.get("sector","")}{" · " + _fund.get("industry","") if _fund.get("industry") else ""}</div>
                        </div>""", unsafe_allow_html=True)

            # ── OBV Chart ─────────────────────────────────────────────────────
            if 'OBV' in df.columns:
                with st.expander("📈 On-Balance Volume (OBV) — Smart Money Flow", expanded=False):
                    fig_obv = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.06)
                    fig_obv.add_trace(go.Scatter(x=df.index, y=close_series, name="Price", line=dict(color=C_ACCENT, width=1.5)), row=1, col=1)
                    fig_obv.add_trace(go.Scatter(x=df.index, y=df['OBV'].squeeze(), name="OBV", line=dict(color=C_EMERALD, width=1.5), fill='tozeroy', fillcolor='rgba(0,229,176,0.05)'), row=2, col=1)
                    if 'OBV_EMA' in df.columns:
                        fig_obv.add_trace(go.Scatter(x=df.index, y=df['OBV_EMA'].squeeze(), name="OBV EMA(20)", line=dict(color=C_YELLOW, width=1.0, dash='dot')), row=2, col=1)
                    _obv_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis','yaxis')}
                    fig_obv.update_layout(**_obv_layout, height=420,
                        title=dict(text=f"{ticker} · Price vs On-Balance Volume · Divergence = early signal", font=dict(color=C_GREEN, size=13)))
                    fig_obv.update_xaxes(gridcolor="#252f47", linecolor="#252f47", tickfont=dict(color=C_GREY))
                    fig_obv.update_yaxes(gridcolor="#252f47", linecolor="#252f47", tickfont=dict(color=C_GREY))
                    st.plotly_chart(fig_obv, use_container_width=True)
                    st.markdown('<div style="background:rgba(0,229,176,0.04);border:1px solid rgba(0,229,176,0.15);border-left:3px solid #00e5b0;padding:.6rem 1.2rem;font-family:Manrope,sans-serif;font-size:.72rem;color:#8a8fa0;border-radius:0 .5rem .5rem 0;">💡 OBV rising with flat/falling price = accumulation (bullish divergence). OBV falling with rising price = distribution (bearish divergence).</div>', unsafe_allow_html=True)

            # ── Future Forecast ────────────────────────────────────────────────
            st.subheader(_L["forecast_next"].format(n=future_days))
            future_prices    = []
            last_row_feats   = X[-1].copy()
            for d in range(future_days):
                next_pred = float(model.predict(last_row_feats.reshape(1,-1))[0])
                future_prices.append(next_pred)
                n_tech = len(FEATURE_COLS); lags = last_row_feats[n_tech:]
                last_row_feats = np.concatenate([last_row_feats[:n_tech], np.append(lags[1:], next_pred)])

            trend_color  = C_EMERALD if future_prices[-1] > last_close else C_RED
            price_std    = float(df['Close'].squeeze().pct_change().std())
            decay_upper  = [future_prices[i]*(1+price_std*np.sqrt(i+1)*1.5) for i in range(future_days)]
            decay_lower  = [future_prices[i]*(1-price_std*np.sqrt(i+1)*1.5) for i in range(future_days)]

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=list(range(future_days)), y=decay_upper, line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
            fig3.add_trace(go.Scatter(x=list(range(future_days)), y=decay_lower, name="Uncertainty Band", fill="tonexty", fillcolor="rgba(77,142,255,0.08)", line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip"))
            fig3.add_hline(y=last_close, line_dash="dash", line_color=C_GREY, annotation_text=f"Last close ${last_close:.2f}", annotation_font_color=C_GREY)
            if alert_price > 0:
                fig3.add_hline(y=alert_price, line_dash="dash", line_color=C_YELLOW, annotation_text=f"Target ${alert_price:.2f}", annotation_font_color=C_YELLOW)
            fig3.add_trace(go.Scatter(x=list(range(future_days)), y=future_prices, mode='lines+markers', name='AI Price Outlook',
                line=dict(color=trend_color, width=2), marker=dict(size=7, color=trend_color, line=dict(width=1, color="#080e1c"))))
            if future_days > 5:
                fig3.add_vline(x=4.5, line_dash="dot", line_color="#252f47", annotation_text="↑ Higher confidence | Lower confidence ↓",
                               annotation_font=dict(color="#3e4558", size=9), annotation_position="top")
            fig3.update_layout(**PLOTLY_LAYOUT,
                title=dict(text=f"{ticker} · {future_days}-Day Price Outlook · Band shows ±1.5σ uncertainty growth", font=dict(color=C_GREEN, size=13)),
                xaxis_title="Days from today", yaxis_title="Price (USD)", height=420)
            st.plotly_chart(fig3, use_container_width=True)

            if future_days > 5:
                st.markdown('<div style="background:rgba(255,212,38,0.04);border:1px solid rgba(255,212,38,0.2);border-left:3px solid #ffd426;padding:.6rem 1.2rem;font-family:Manrope,sans-serif;font-size:.7rem;color:#ffd426;font-weight:600;border-radius:0 .5rem .5rem 0;">💡 Outlook confidence is highest for Days 1–3. Days 6+ should be read as directional guidance only.</div>', unsafe_allow_html=True)

            future_df = pd.DataFrame({
                "Day": [f"+{i+1}" for i in range(future_days)],
                "AI Outlook ($)": [f"${p:.2f}" for p in future_prices],
                "vs Last Close": [f"{'▲' if p>last_close else '▼'} {abs(p-last_close):.2f} ({(p-last_close)/last_close*100:+.2f}%)" for p in future_prices]
            })
            st.dataframe(future_df, use_container_width=True, hide_index=True)

            # ── Backtesting ────────────────────────────────────────────────────
            if run_backtest and not fast_mode:
                st.subheader("Strategy Simulator")
                st.markdown(f'<div class="model-badge">STRATEGY: AI Signal ±{bt_signal_threshold}% | Capital: ${bt_initial_capital:,.0f} | Commission: ${bt_commission}/trade</div>', unsafe_allow_html=True)
                with st.spinner("Running backtest simulation..."):
                    bt = run_backtest_engine(actual, preds, bt_initial_capital, bt_commission, bt_signal_threshold)

                strat_color = "bt-val-green" if bt["strat_return"]>=0 else "bt-val-red"
                bh_color    = "bt-val-green" if bt["bh_return"]>=0    else "bt-val-red"
                dd_color    = "bt-val-red"   if bt["max_drawdown"]<-10 else "bt-val"
                sh_color    = "bt-val-green" if bt["sharpe"]>=1        else "bt-val-red"

                k1,k2,k3,k4 = st.columns(4)
                k1.markdown(f'<div class="bt-card"><div class="bt-label">Strategy Return</div><div class="{strat_color}">{bt["strat_return"]:+.2f}%</div></div>', unsafe_allow_html=True)
                k2.markdown(f'<div class="bt-card"><div class="bt-label">Buy &amp; Hold Return</div><div class="{bh_color}">{bt["bh_return"]:+.2f}%</div></div>', unsafe_allow_html=True)
                k3.markdown(f'<div class="bt-card"><div class="bt-label">Max Drawdown</div><div class="{dd_color}">{bt["max_drawdown"]:.2f}%</div></div>', unsafe_allow_html=True)
                k4.markdown(f'<div class="bt-card"><div class="bt-label">Sharpe Ratio</div><div class="{sh_color}">{bt["sharpe"]:.2f}</div></div>', unsafe_allow_html=True)
                k5,k6,k7,k8 = st.columns(4)
                k5.markdown(f'<div class="bt-card"><div class="bt-label">Final Capital</div><div class="bt-val">${bt["final_capital"]:,.0f}</div></div>', unsafe_allow_html=True)
                k6.markdown(f'<div class="bt-card"><div class="bt-label">Total Trades</div><div class="bt-val">{bt["total_trades"]}</div></div>', unsafe_allow_html=True)
                k7.markdown(f'<div class="bt-card"><div class="bt-label">Win Rate</div><div class="bt-val">{bt["win_rate"]:.1f}%</div></div>', unsafe_allow_html=True)
                k8.markdown(f'<div class="bt-card"><div class="bt-label">Profit Factor</div><div class="bt-val">{bt["profit_factor"]:.2f}x</div></div>', unsafe_allow_html=True)

                if bt["total_trades"] == 0 and last_close > bt_initial_capital:
                    st.warning(f"⚠ No trades executed — {ticker} (${last_close:.2f}/share) exceeds the initial capital of ${bt_initial_capital:,.0f}. Increase the capital in the sidebar to enable backtesting for this stock.")

                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(y=bt["equity_curve"], name="AI Strategy", line=dict(color=C_EMERALD, width=2), fill="tozeroy", fillcolor="rgba(0,229,176,0.05)"))
                fig_eq.add_trace(go.Scatter(y=bt["bh_equity"], name="Buy & Hold", line=dict(color=C_ACCENT, width=1.5, dash="dot")))
                fig_eq.add_hline(y=bt_initial_capital, line_dash="dash", line_color=C_GREY, annotation_text=f"Start ${bt_initial_capital:,}", annotation_font_color=C_GREY)
                fig_eq.update_layout(**PLOTLY_LAYOUT, title=dict(text=f"{ticker} · AI Strategy Equity Curve vs Buy & Hold", font=dict(color=C_GREEN, size=13)), height=420)
                st.plotly_chart(fig_eq, use_container_width=True)

                if not bt["trades_df"].empty:
                    st.subheader("Trade Log")
                    td = bt["trades_df"].copy()
                    td["Price"]   = td["Price"].apply(lambda x: f"${x:.2f}")
                    td["Capital"] = td["Capital"].apply(lambda x: f"${x:,.0f}")
                    if "P&L" in td.columns:
                        td["P&L"] = td["P&L"].apply(lambda x: f"+${x:.2f}" if pd.notna(x) and x>=0 else (f"-${abs(x):.2f}" if pd.notna(x) else "-"))
                    st.dataframe(td, use_container_width=True, hide_index=True)
                    st.download_button("⬇ Download Trade Log", data=bt["trades_df"].to_csv(index=False).encode(), file_name=f"{ticker}_trades.csv", mime="text/csv")

            # ── Confidence Intervals ───────────────────────────────────────────
            if show_conf_interval:
                st.subheader("Outlook with Confidence Intervals")
                st.markdown('<div class="model-badge">95% CI — Bootstrap Resampling</div>', unsafe_allow_html=True)
                with st.spinner(f"Running {ci_bootstrap_n} bootstrap samples..."):
                    ci_lower, ci_median, ci_upper = bootstrap_confidence_intervals(model, X_test, n_bootstrap=ci_bootstrap_n, noise_std=0.015)
                fig_ci = go.Figure()
                fig_ci.add_trace(go.Scatter(y=ci_upper, line=dict(color="rgba(0,0,0,0)"), showlegend=False))
                fig_ci.add_trace(go.Scatter(y=ci_lower, name="95% CI Band", fill="tonexty", fillcolor="rgba(77,142,255,0.10)", line=dict(color="rgba(0,0,0,0)")))
                fig_ci.add_trace(go.Scatter(y=actual, name="Actual", line=dict(color=C_ACCENT, width=1.5)))
                fig_ci.add_trace(go.Scatter(y=ci_median, name="AI Projection Median", line=dict(color=C_EMERALD, width=1.8, dash="dot")))
                fig_ci.update_layout(**PLOTLY_LAYOUT, title=dict(text=f"{ticker} · AI Projection with 95% CI", font=dict(color=C_GREEN, size=13)), height=380)
                st.plotly_chart(fig_ci, use_container_width=True)

            # ── Model Comparison ───────────────────────────────────────────────
            if run_model_compare:
                st.subheader("Model Comparison — XGBoost vs Prophet vs Linear Regression")
                from sklearn.linear_model import LinearRegression as LR
                cmp = {}
                cmp["XGBoost"] = {"preds":preds,"color":C_EMERALD,
                    "rmse":float(np.sqrt(mean_squared_error(actual,preds))),
                    "mae":float(mean_absolute_error(actual,preds)),
                    "mape":float(np.mean(np.abs((actual-preds)/actual))*100),
                    "r2":float(1-np.sum((actual-preds)**2)/np.sum((actual-np.mean(actual))**2))}
                with st.spinner("Training Linear Regression..."):
                    lr_m = LR(); lr_m.fit(X_train, y_train); lr_p = lr_m.predict(X_test)
                cmp["Linear Regression"] = {"preds":lr_p,"color":C_GREY,
                    "rmse":float(np.sqrt(mean_squared_error(actual,lr_p))),
                    "mae":float(mean_absolute_error(actual,lr_p)),
                    "mape":float(np.mean(np.abs((actual-lr_p)/actual))*100),
                    "r2":float(1-np.sum((actual-lr_p)**2)/np.sum((actual-np.mean(actual))**2))}
                try:
                    from prophet import Prophet
                    cs_full = df["Close"].squeeze()
                    pdf = pd.DataFrame({"ds":df.index[:len(cs_full)],"y":cs_full.values}).dropna()
                    ptr = pdf.iloc[:int(len(pdf)*0.8)]; pte = pdf.iloc[int(len(pdf)*0.8):]
                    with st.spinner("Training Prophet..."):
                        pm = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
                        pm.fit(ptr); pfut = pm.make_future_dataframe(periods=len(pte), freq="B"); pfcst = pm.predict(pfut)
                        pp = pfcst["yhat"].values[-len(pte):]; pa = pte["y"].values; ml = min(len(pp), len(actual)); pp, pa = pp[:ml], actual[:ml]
                    cmp["Prophet"] = {"preds":pp,"color":C_YELLOW,
                        "rmse":float(np.sqrt(mean_squared_error(pa,pp))),
                        "mae":float(mean_absolute_error(pa,pp)),
                        "mape":float(np.mean(np.abs((pa-pp)/pa))*100),
                        "r2":float(1-np.sum((pa-pp)**2)/np.sum((pa-np.mean(pa))**2))}
                except ImportError:
                    st.info("Add `prophet` to requirements.txt to enable Prophet comparison.")

                rows = [{"Model":n,"RMSE ($)":f"${r['rmse']:.2f}","MAE ($)":f"${r['mae']:.2f}","MAPE (%)":f"{r['mape']:.2f}%","R²":f"{r['r2']:.4f}"} for n,r in cmp.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Scatter(y=actual, name="Actual", line=dict(color=C_ACCENT, width=2)))
                for n, r in cmp.items():
                    fig_cmp.add_trace(go.Scatter(y=r["preds"], name=n, line=dict(color=r["color"], width=1.5, dash="dot")))
                fig_cmp.update_layout(**PLOTLY_LAYOUT, title=dict(text=f"{ticker} · Model Comparison", font=dict(color=C_GREEN, size=13)), height=420)
                st.plotly_chart(fig_cmp, use_container_width=True)

            # ── Halal / Shariah ────────────────────────────────────────────────
            if run_halal_check:
                st.markdown("""
                <div style="background:rgba(0,229,176,0.03);border:1px solid rgba(0,229,176,0.15);border-left:4px solid #00e5b0; padding:.8rem 1.4rem;margin:1.5rem 0 .5rem;display:flex;align-items:center;gap:1rem;border-radius:0 .5rem .5rem 0;">
                  <div style="font-size:1.4rem;">☪</div>
                  <div>
                    <div style="font-family:Manrope,sans-serif;font-size:.63rem;letter-spacing:.15em;text-transform:uppercase;color:#00e5b0;font-weight:700;">Halal / Shariah Compliance Screen</div>
                    <div style="font-family:Manrope,sans-serif;font-size:.76rem;color:#3e4558;margin-top:2px;">Automated screening based on AAOIFI Standard No.21</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                with st.spinner(f"Fetching financial data for {ticker}..."):
                    sd = get_shariah_data(ticker)
                if sd is None:
                    sd = {"debt_to_mktcap":0,"debt_to_assets":0,"cash_to_assets":0,
                          "market_cap":0,"total_debt":0,"total_assets":0,"total_cash":0,
                          "sector":"Unknown","industry":"Unknown","company_name":ticker}
                    st.warning(f"⚠ Could not fetch detailed financial data for {ticker}. Using ticker-list screening only.")
                if sd is not None:
                    compliance_result = check_shariah_compliance(ticker, sd, _L)
                    sh_verdict = compliance_result["verdict"]
                    v_color = {"COMPLIANT":C_EMERALD,"NON-COMPLIANT":C_RED,"QUESTIONABLE":C_YELLOW}[sh_verdict]
                    v_bg    = {"COMPLIANT":"rgba(0,229,176,0.05)","NON-COMPLIANT":"rgba(255,107,107,0.05)","QUESTIONABLE":"rgba(255,221,45,0.05)"}[sh_verdict]
                    v_icon  = {"COMPLIANT":"✅","NON-COMPLIANT":"❌","QUESTIONABLE":"⚠️"}[sh_verdict]
                    st.markdown(f'<div style="background:{v_bg};border:1px solid {v_color};border-left:3px solid {v_color};padding:1.2rem 2rem;margin:1rem 0;text-align:center;border-radius:0 .5rem .5rem 0;"><div style="font-family:Manrope,sans-serif;font-size:.6rem;color:#3e4558;letter-spacing:.14em;text-transform:uppercase;font-weight:700;">{sd["company_name"]} ({ticker})</div><div style="font-family:IBM Plex Mono,monospace;font-size:1.8rem;font-weight:700;color:{v_color};margin-top:.4rem;">{v_icon}&nbsp;{sh_verdict}</div><div style="font-size:.76rem;color:#8a8fa0;margin-top:.3rem;">Sector: {sd["sector"]} | Industry: {sd["industry"]}</div></div>', unsafe_allow_html=True)

                    st.subheader(_L["screening_criteria"])
                    col_left, col_right = st.columns(2)
                    with col_left:
                        bs = compliance_result["business"]
                        if bs["haram_hit"]:
                            st.markdown(f'<div class="halal-card-fail"><b>❌ Business Activity</b><br>Non-compliant: <b>{bs["haram_hit"]}</b></div>', unsafe_allow_html=True)
                        elif bs["questionable"]:
                            st.markdown('<div class="halal-card" style="border-left-color:#ffd426;"><b>⚠️ Business Activity</b><br>Questionable sector — consult a scholar</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="halal-card"><b>✅ Business Activity</b><br>No Haram core business detected<br><small style="color:#3e4558;">Sector: {sd["sector"]}</small></div>', unsafe_allow_html=True)
                        dm = compliance_result["debt_mktcap"]
                        cls = "halal-card" if dm["pass"] else "halal-card-fail"
                        icon = "✅" if dm["pass"] else "❌"
                        st.markdown(f'<div class="{cls}"><b>{icon} {dm["label"]}</b></div>', unsafe_allow_html=True)
                    with col_right:
                        da = compliance_result["debt_assets"]
                        cls = "halal-card" if da["pass"] else "halal-card-fail"
                        icon = "✅" if da["pass"] else "❌"
                        st.markdown(f'<div class="{cls}"><b>{icon} {da["label"]}</b></div>', unsafe_allow_html=True)
                        ca = compliance_result["cash_assets"]
                        cls = "halal-card" if ca["pass"] else "halal-card-fail"
                        icon = "✅" if ca["pass"] else "❌"
                        st.markdown(f'<div class="{cls}"><b>{icon} {ca["label"]}</b></div>', unsafe_allow_html=True)

            # ── News Sentiment ──────────────────────────────────────────────────
            if not is_beginner:
                st.subheader(_L["news_sentiment"])
                try:
                    from sentiment import render_sentiment_ui
                    raw_news    = av_get_news(ticker)
                    avg_polarity = render_sentiment_ui(ticker, raw_news, _L=_L)
                    # Store for downstream use in signal engine
                    st.session_state["last_sentiment_score"] = avg_polarity
                except Exception as e:
                    logger.warning("Sentiment UI failed: %s", e)
                    st.warning(f"Could not load sentiment analysis: {e}")


        # ══════════════════════════════════════════════════════════════════════
        # 🚀  STARTUP HUB
        # ══════════════════════════════════════════════════════════════════════
        with startup_tab:
            st.markdown("""
            <div style="margin-bottom:1.4rem;">
              <div style="font-family:Manrope,sans-serif;font-size:1.9rem;font-weight:800; letter-spacing:-.03em;color:#e4eafd;">Startup <span style="color:#4d8eff;">Hub</span></div>
              <div style="font-size:.82rem;color:#8a8fa0;margin-top:.35rem;line-height:1.65;max-width:680px;">
                Six intelligence tools built for founders — macro risk scanner, treasury optimizer,
                competitor radar, sector benchmarks, signal alerts, and investor-ready reports.
              </div>
            </div>
            """, unsafe_allow_html=True)

            _sh_comp = None
            try:
                _sh_comp = compute_composite_signal(df, last_close, preds[-1], preds, actual)
            except Exception:
                pass

            # FIX 5: Tab orientation — numbered labels + "Start here" cue
            st.markdown(f"""
            <div style="background:rgba(77,142,255,0.05);border:1px solid rgba(77,142,255,0.15); border-left:3px solid #4d8eff;border-radius:0 .6rem .6rem 0; padding:.65rem 1.2rem;margin-bottom:.75rem; display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;">
              <span style="font-family:IBM Plex Mono,monospace;font-size:0.78rem; color:#4d8eff;font-weight:700;">📍 Start here</span>
              <span style="font-family:Manrope,sans-serif;font-size:0.78rem;color:#8a8fa0;">
                Begin with <b style="color:#e4eafd;">01 Macro Risk</b> for market context,
                then work across. <b style="color:#e4eafd;">05 Signal Alert</b> and
                <b style="color:#e4eafd;">06 Report</b> need analysis to run first.
              </span>
            </div>
            """, unsafe_allow_html=True)

            hub1, hub2, hub3, hub4, hub5, hub6, hub7, hub8, hub9 = st.tabs([
                "01 · 🌡 Macro", "02 · 🏦 Treasury", "03 · 🔭 Competitors",
                "04 · 📊 Benchmarks", "05 · 🔔 Signal Alert", "06 · 📑 Report",
                "07 · 📈 Backtest", "08 · 🔗 Brokers", "09 · 🤖 AI Report"
            ])

            # ── HUB 1: MACRO RISK SCANNER ─────────────────────────────────────────
            with hub1:
                st.markdown('''<div style="background:rgba(77,142,255,0.06);border:1px solid rgba(77,142,255,0.2); border-left:4px solid #4d8eff;padding:.85rem 1.3rem;margin-bottom:1.2rem;border-radius:0 .6rem .6rem 0;">
                  <div style="font-size:.6rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#4d8eff;margin-bottom:.25rem;">Macroeconomic Risk Scanner</div>
                  <div style="font-size:.8rem;color:#8a8fa0;line-height:1.6;">Live composite score from VIX, S&amp;P momentum, Fear &amp; Greed, and bond market.
                    Verdict: <b style="color:#e4eafd;">RAISE NOW · PROCEED · CAUTION · WAIT</b></div>
                </div>''', unsafe_allow_html=True)

                with st.spinner("Computing macro climate..."):
                    _mac = get_macro_risk_score()
                _ms = _mac["score"]; _ml = _mac["label"]; _mc = _mac["color"]
                _mv = _mac["verdict"]; _mf = _mac["factors"]
                _mp = f"{max(2,min(98,_ms)):.0f}%"

                _h1a, _h1b = st.columns([1, 2])
                with _h1a:
                    st.markdown(f'''<div style="background:linear-gradient(145deg,#0f1727,#080e1c); border:2px solid {_mc};border-radius:1.2rem;padding:2rem 1.5rem;text-align:center;">
                      <div style="font-size:.7rem;font-weight:800;letter-spacing:.18em;text-transform:uppercase;color:#3e4558;margin-bottom:.5rem;">Market Climate</div>
                      <div style="font-family:IBM Plex Mono,monospace;font-size:4rem;font-weight:700;color:{_mc};line-height:1;">{_ms:.0f}</div>
                      <div style="font-size:.68rem;color:#3e4558;margin-bottom:.9rem;">/100</div>
                      <div style="background:{_mc};color:#080e1c;font-size:.72rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;padding:.5rem 1.2rem;border-radius:.4rem;display:inline-block;">{_ml}</div>
                      <div style="height:4px;background:linear-gradient(90deg,#ff5f5f,#ffd426,#00e5b0);border-radius:2px;margin-top:1.3rem;position:relative;">
                        <div style="position:absolute;top:-7px;left:{_mp};transform:translateX(-50%);width:3px;height:18px;background:#e4eafd;border-radius:1px;"></div>
                      </div>
                      <div style="display:flex;justify-content:space-between;margin-top:.35rem;font-size:.5rem;color:#3e4558;text-transform:uppercase;font-weight:700;">
                        <span>Wait</span><span>Proceed</span><span>Raise</span>
                      </div>
                    </div>''', unsafe_allow_html=True)

                with _h1b:
                    st.markdown(
                        f'<div style="background:rgba(0,0,0,0.2);border:1px solid {_mc};border-left:4px solid {_mc};padding:1rem 1.4rem;border-radius:0 .6rem .6rem 0;margin-bottom:.8rem;">'
                        f'<div style="font-size:.7rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:{_mc};margin-bottom:.35rem;">Founder Verdict</div>'
                        f'<div style="font-size:.84rem;color:#c8cedd;line-height:1.65;">{_mv}</div>'
                        f'</div>',
                        unsafe_allow_html=True)
                    st.markdown('<div style="font-size:.7rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#3e4558;margin-bottom:.4rem;">Factor Breakdown</div>', unsafe_allow_html=True)
                    for _fn, (_fc, _fl) in _mf.items():
                        _fcc = "#00e5b0" if _fc > 0 else "#ff5f5f" if _fc < 0 else "#ffd426"
                        _fcb = "#00e5b0" if _fc > 0 else "#ff5f5f" if _fc < 0 else "#1e2740"
                        _fcs = "+" if _fc > 0 else ""
                        st.markdown(
                            f'<div style="display:flex;align-items:center;justify-content:space-between;' f'padding:.5rem .9rem;background:#0f1727;border-left:2px solid {_fcb};' f'border-radius:0 .35rem .35rem 0;margin-bottom:.25rem;">'
                            f'<div style="font-size:.72rem;color:#8a8fa0;">{_fl}</div>'
                            f'<div style="font-family:IBM Plex Mono,monospace;font-size:.72rem;font-weight:700;color:{_fcc};">{_fcs}{_fc}</div>'
                            f'</div>',
                            unsafe_allow_html=True)
                try:
                    _sp90 = _yf_download_with_retry("^GSPC", period="90d", interval="1d")
                    if not _sp90.empty:
                        _sp90c = _sp90["Close"].dropna()
                        fig_mc = go.Figure()
                        fig_mc.add_trace(go.Scatter(x=_sp90c.index, y=_sp90c.values, name="S&P 500",
                            line=dict(color="#4d8eff",width=1.8), fill="tozeroy", fillcolor="rgba(77,142,255,0.07)"))
                        fig_mc.update_layout(**PLOTLY_LAYOUT,
                            title=dict(text="S&P 500 · 90-Day Macro Context",font=dict(color=C_GREEN,size=12)), height=200)
                        st.plotly_chart(fig_mc, use_container_width=True)
                except Exception:
                    pass
                st.caption("⚠ Macro Climate is a composite heuristic. Not financial advice.")

            # ── HUB 2: TREASURY OPTIMIZER ─────────────────────────────────────────
            with hub2:
                st.markdown('''<div style="background:rgba(0,229,176,0.05);border:1px solid rgba(0,229,176,0.18); border-left:4px solid #00e5b0;padding:.85rem 1.3rem;margin-bottom:1.2rem;border-radius:0 .6rem .6rem 0;">
                  <div style="font-size:.6rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#00e5b0;margin-bottom:.25rem;">Treasury / Cash Reserve Optimizer</div>
                  <div style="font-size:.8rem;color:#8a8fa0;line-height:1.6;">Don't let idle startup cash earn nothing. Compare low-risk ETF options optimized for your runway horizon.</div>
                </div>''', unsafe_allow_html=True)

                _tb1, _tb2, _tb3 = st.columns(3)
                with _tb1: _cash = st.number_input("Cash to deploy ($)", min_value=10000, max_value=50000000, value=500000, step=25000, key="hub_cash")
                with _tb2: _horiz = st.selectbox("Time horizon", ["3 months","6 months","12 months","18 months","24 months+"], index=1, key="hub_horiz")
                with _tb3: _riskp = st.selectbox("Risk tolerance", ["Capital preservation","Balanced growth","Income-focused"], index=0, key="hub_risk")

                _hmap = {"3 months":"Ultra-safe (T-Bills)","6 months":"Ultra-safe (T-Bills)",
                         "12 months":"Short-term Bonds","18 months":"Intermediate Bonds","24 months+":"Balanced (60/40)"}
                _auto = "Dividend / Income" if _riskp=="Income-focused" else _hmap.get(_horiz,"Ultra-safe (T-Bills)") if _riskp=="Capital preservation" else _hmap.get(_horiz,"Short-term Bonds")
                _tp   = TREASURY_PROFILES[_auto]

                st.markdown(f'''<div style="background:linear-gradient(135deg,rgba(0,229,176,0.08),rgba(77,142,255,0.04)); border:1px solid rgba(0,229,176,0.25);border-radius:.75rem;padding:1rem 1.4rem;margin:.6rem 0 1rem;">
                  <div style="font-size:.7rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#00e5b0;margin-bottom:.3rem;">✦ Recommended Strategy</div>
                  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem;">
                    <div>
                      <div style="font-size:1rem;font-weight:800;color:#e4eafd;">{_auto}</div>
                      <div style="font-size:.78rem;color:#8a8fa0;margin-top:.25rem;max-width:440px;line-height:1.55;">{_tp["desc"]}</div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-size:.7rem;color:#3e4558;font-family:IBM Plex Mono,monospace;">Suggested tickers</div>
                      <div style="font-family:IBM Plex Mono,monospace;font-size:.9rem;font-weight:700;color:#4d8eff;">{" · ".join(_tp["tickers"])}</div>
                    </div>
                  </div>
                </div>''', unsafe_allow_html=True)

                with st.spinner("Loading ETF data..."):
                    _tdata = get_treasury_data(_tp["tickers"])

                _th = st.columns([1,2.5,1.1,1.1,1.2,1.2,1.4])
                for _thc, _tht in zip(_th,["Ticker","Name","Price","Today","1Y Return","Div Yield","AUM"]):
                    _thc.markdown(f'<div style="font-size:.7rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#3e4558;">{_tht}</div>', unsafe_allow_html=True)

                for _te in _tdata:
                    _tc = "#00e5b0" if _te["change_pct"]>=0 else "#ff5f5f"
                    _ts = "+" if _te["change_pct"]>=0 else ""
                    _tr1c = "#00e5b0" if _te["ret1y"]>=0 else "#ff5f5f"
                    _taum = f"${_te['aum']/1e9:.1f}B" if _te["aum"]>=1e9 else f"${_te['aum']/1e6:.0f}M" if _te["aum"]>0 else "—"
                    _tr = st.columns([1,2.5,1.1,1.1,1.2,1.2,1.4])
                    _tr[0].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.8rem;font-weight:700;color:#4d8eff;padding:.4rem 0;">{_te["ticker"]}</div>', unsafe_allow_html=True)
                    _tr[1].markdown(f'<div style="font-size:.73rem;color:#8a8fa0;padding:.4rem 0;">{_te["name"]}</div>', unsafe_allow_html=True)
                    _tr[2].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;color:#e4eafd;padding:.4rem 0;">${_te["price"]:.2f}</div>', unsafe_allow_html=True)
                    _tr[3].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;font-weight:700;color:{_tc};padding:.4rem 0;">{_ts}{_te["change_pct"]:.2f}%</div>', unsafe_allow_html=True)
                    _tr[4].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;color:{_tr1c};font-weight:700;padding:.4rem 0;">{"+" if _te["ret1y"]>=0 else ""}{_te["ret1y"]:.1f}%</div>', unsafe_allow_html=True)
                    _tr[5].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;color:#adc6ff;padding:.4rem 0;">{_te["div_yield"]:.2f}%</div>', unsafe_allow_html=True)
                    _tr[6].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.73rem;color:#8a8fa0;padding:.4rem 0;">{_taum}</div>', unsafe_allow_html=True)

                _vr = [e["ret1y"] for e in _tdata if e["ret1y"] != 0]
                if _vr:
                    _ar = sum(_vr)/len(_vr)
                    _pg = _cash * (_ar/100)
                    st.markdown(f'''<div style="display:flex;gap:1rem;flex-wrap:wrap;margin-top:.9rem;">
                      <div style="background:#0f1727;border:1px solid #252f47;border-top:2px solid #00e5b0;border-radius:.6rem;padding:.9rem 1.3rem;flex:1;min-width:140px;">
                        <div style="font-size:.7rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#3e4558;margin-bottom:.3rem;">Deployed Capital</div>
                        <div style="font-family:IBM Plex Mono,monospace;font-size:1.3rem;font-weight:700;color:#e4eafd;">${_cash:,.0f}</div>
                      </div>
                      <div style="background:#0f1727;border:1px solid #252f47;border-top:2px solid #4d8eff;border-radius:.6rem;padding:.9rem 1.3rem;flex:1;min-width:140px;">
                        <div style="font-size:.7rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#3e4558;margin-bottom:.3rem;">Avg 1Y Return</div>
                        <div style="font-family:IBM Plex Mono,monospace;font-size:1.3rem;font-weight:700;color:#4d8eff;">{"+" if _ar>=0 else ""}{_ar:.2f}%</div>
                      </div>
                      <div style="background:#0f1727;border:1px solid #252f47;border-top:2px solid #ffd426;border-radius:.6rem;padding:.9rem 1.3rem;flex:1;min-width:140px;">
                        <div style="font-size:.7rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#3e4558;margin-bottom:.3rem;">Projected Gain (1Y)</div>
                        <div style="font-family:IBM Plex Mono,monospace;font-size:1.3rem;font-weight:700;color:#ffd426;">{"+" if _pg>=0 else ""}${_pg:,.0f}</div>
                      </div>
                      <div style="background:#0f1727;border:1px solid #252f47;border-top:2px solid #00e5b0;border-radius:.6rem;padding:.9rem 1.3rem;flex:1;min-width:140px;">
                        <div style="font-size:.7rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#3e4558;margin-bottom:.3rem;">Final Value</div>
                        <div style="font-family:IBM Plex Mono,monospace;font-size:1.3rem;font-weight:700;color:#00e5b0;">${_cash+_pg:,.0f}</div>
                      </div>
                    </div>''', unsafe_allow_html=True)
                st.caption("⚠ Past ETF returns do not guarantee future results. Not financial advice.")

            # ── HUB 3: COMPETITOR TRACKER ─────────────────────────────────────────
            with hub3:
                st.markdown('''<div style="background:rgba(255,212,38,0.05);border:1px solid rgba(255,212,38,0.18); border-left:4px solid #ffd426;padding:.85rem 1.3rem;margin-bottom:1.2rem;border-radius:0 .6rem .6rem 0;">
                  <div style="font-size:.6rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#ffd426;margin-bottom:.25rem;">Competitor Stock Tracker · B2B Intel</div>
                  <div style="font-size:.8rem;color:#8a8fa0;line-height:1.6;">Monitor public competitors in real time — price, market cap, P/E, and 52-week positioning. Your analysis ticker is always included.</div>
                </div>''', unsafe_allow_html=True)

                if "comp_tickers" not in st.session_state: st.session_state.comp_tickers = []
                _ci1, _ci2 = st.columns([5,1])
                with _ci1: _cr = st.text_input("Add tickers (comma-separated)", placeholder="e.g. MSFT, AAPL, SHOP", label_visibility="collapsed", key="comp_input_hub")
                with _ci2:
                    if st.button("➕ Add", key="comp_add_hub", use_container_width=True):
                        for _s in [x.strip().upper() for x in _cr.split(",") if x.strip()]:
                            if _s and _s not in st.session_state.comp_tickers and len(st.session_state.comp_tickers) < 9:
                                st.session_state.comp_tickers.append(_s)
                        st.rerun()

                if st.session_state.comp_tickers:
                    _rcols = st.columns(min(len(st.session_state.comp_tickers),5))
                    for _ri, _rs in enumerate(list(st.session_state.comp_tickers)):
                        with _rcols[_ri%5]:
                            if st.button(f"✕ {_rs}", key=f"cremove_{_rs}_{_ri}", use_container_width=True):
                                st.session_state.comp_tickers.remove(_rs); st.rerun()

                _allc = list(dict.fromkeys([ticker] + st.session_state.comp_tickers))
                with st.spinner("Fetching competitor data..."):
                    _crows = get_comp_snapshot(_allc)

                _chdr = st.columns([1.2,2.4,1.2,1.2,1.5,1.2,1.5])
                for _cc,_ct in zip(_chdr,["Ticker","Company","Price","Δ Today","Mkt Cap","P/E","vs 52w High"]):
                    _cc.markdown(f'<div style="font-size:.7rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#3e4558;">{_ct}</div>', unsafe_allow_html=True)

                for _cr2 in _crows:
                    _isc   = _cr2["ticker"] == ticker
                    _cc2   = "#00e5b0" if _cr2["change_pct"]>=0 else "#ff5f5f"
                    _cs2   = "+" if _cr2["change_pct"]>=0 else ""
                    _cmc   = f"${_cr2['mktcap']/1e12:.2f}T" if _cr2["mktcap"]>=1e12 else f"${_cr2['mktcap']/1e9:.1f}B" if _cr2["mktcap"]>=1e9 else "—"
                    _cpe   = f"{_cr2['pe']:.1f}×" if _cr2["pe"]>0 else "—"
                    _cpfhc = "#ff5f5f" if _cr2["pfh"]<-20 else "#ffd426" if _cr2["pfh"]<-10 else "#00e5b0"
                    _cpfhs = f"{_cr2['pfh']:+.1f}%" if _cr2["w52h"]>0 else "—"
                    _crow  = st.columns([1.2,2.4,1.2,1.2,1.5,1.2,1.5])
                    _crow[0].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.8rem;font-weight:700;color:#4d8eff;padding:.45rem 0;">{_cr2["ticker"]}{"  ◄" if _isc else ""}</div>', unsafe_allow_html=True)
                    _crow[1].markdown(f'<div style="font-size:.73rem;color:#8a8fa0;padding:.45rem 0;">{_cr2["name"]}</div>', unsafe_allow_html=True)
                    _crow[2].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;color:#e4eafd;padding:.45rem 0;">${_cr2["price"]:,.2f}</div>', unsafe_allow_html=True)
                    _crow[3].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;font-weight:700;color:{_cc2};padding:.45rem 0;">{_cs2}{_cr2["change_pct"]:.2f}%</div>', unsafe_allow_html=True)
                    _crow[4].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.73rem;color:#8a8fa0;padding:.45rem 0;">{_cmc}</div>', unsafe_allow_html=True)
                    _crow[5].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.73rem;color:#adc6ff;padding:.45rem 0;">{_cpe}</div>', unsafe_allow_html=True)
                    _crow[6].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;font-weight:700;color:{_cpfhc};padding:.45rem 0;">{_cpfhs}</div>', unsafe_allow_html=True)

                if len(_allc) >= 2:
                    try:
                        import yfinance as yf
                        _ch = yf.download(_allc[:6], period="90d", interval="1d", progress=False, auto_adjust=True)
                        _cc3 = _ch["Close"] if "Close" in _ch.columns else _ch
                        if isinstance(_cc3.columns, pd.MultiIndex): _cc3 = _cc3.droplevel(0, axis=1)
                        if isinstance(_cc3, pd.Series): _cc3 = _cc3.to_frame(name=_allc[0])
                        fig_cmp = go.Figure()
                        _pal = ["#4d8eff","#00e5b0","#ffd426","#ff5f5f","#adc6ff","#ff9f40"]
                        for _ci3, _cs3 in enumerate(_allc[:6]):
                            if _cs3 in _cc3.columns:
                                _sv = _cc3[_cs3].dropna()
                                if not _sv.empty:
                                    _rv = (_sv / _sv.iloc[0]) * 100
                                    fig_cmp.add_trace(go.Scatter(x=_rv.index, y=_rv.values, name=_cs3,
                                        line=dict(color=_pal[_ci3%6], width=2.2 if _cs3==ticker else 1.2,
                                                  dash="solid" if _cs3==ticker else "dot")))
                        fig_cmp.update_layout(**PLOTLY_LAYOUT,
                            title=dict(text="Relative Performance · 90d Rebased to 100",font=dict(color=C_GREEN,size=12)), height=320)
                        st.plotly_chart(fig_cmp, use_container_width=True)
                    except Exception as _e:
                        st.info(f"Chart unavailable: {_e}")

            # ── HUB 4: SECTOR BENCHMARK ───────────────────────────────────────────
            with hub4:
                st.markdown('''<div style="background:rgba(173,198,255,0.06);border:1px solid rgba(173,198,255,0.2); border-left:4px solid #adc6ff;padding:.85rem 1.3rem;margin-bottom:1.2rem;border-radius:0 .6rem .6rem 0;">
                  <div style="font-size:.6rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#adc6ff;margin-bottom:.25rem;">Pre-IPO · Sector Benchmark Mode</div>
                  <div style="font-size:.8rem;color:#8a8fa0;line-height:1.6;">Generate a public comps table for your sector — the comparable companies table VCs ask for in every pitch.</div>
                </div>''', unsafe_allow_html=True)

                _secs = st.selectbox("Your sector", list(SECTOR_COMPS.keys()), label_visibility="collapsed", key="sec_bench_hub")
                with st.spinner(f"Loading {_secs} benchmarks..."):
                    _bench = get_comp_snapshot(SECTOR_COMPS[_secs])

                _vpe = [b["pe"] for b in _bench if b["pe"]>0]
                _vch = [b["change_pct"] for b in _bench]
                _vmc = [b["mktcap"] for b in _bench if b["mktcap"]>0]
                _mpe = sorted(_vpe)[len(_vpe)//2] if _vpe else 0
                _ach = sum(_vch)/len(_vch) if _vch else 0
                _tmc = sum(_vmc)

                _s1,_s2,_s3 = st.columns(3)
                _s1.markdown(f'<div style="background:#0f1727;border:1px solid #252f47;border-top:2px solid #adc6ff;border-radius:.6rem;padding:.9rem 1.2rem;"><div style="font-size:.7rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#3e4558;margin-bottom:.3rem;">Median P/E</div><div style="font-family:IBM Plex Mono,monospace;font-size:1.5rem;font-weight:700;color:#adc6ff;">{_mpe:.1f}×</div></div>', unsafe_allow_html=True)
                _s2.markdown(f'<div style="background:#0f1727;border:1px solid #252f47;border-top:2px solid {"#00e5b0" if _ach>=0 else "#ff5f5f"};border-radius:.6rem;padding:.9rem 1.2rem;"><div style="font-size:.7rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#3e4558;margin-bottom:.3rem;">Avg Daily Change</div><div style="font-family:IBM Plex Mono,monospace;font-size:1.5rem;font-weight:700;color:{"#00e5b0" if _ach>=0 else "#ff5f5f"};">{_ach:+.2f}%</div></div>', unsafe_allow_html=True)
                _s3.markdown(f'<div style="background:#0f1727;border:1px solid #252f47;border-top:2px solid #ffd426;border-radius:.6rem;padding:.9rem 1.2rem;"><div style="font-size:.7rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#3e4558;margin-bottom:.3rem;">Combined Mkt Cap</div><div style="font-family:IBM Plex Mono,monospace;font-size:1.5rem;font-weight:700;color:#ffd426;">${_tmc/1e12:.2f}T</div></div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                _bh = st.columns([1.2,2.4,1.2,1.2,1.5,1.2,1.5])
                for _bhc,_bht in zip(_bh,["Ticker","Company","Price","Δ Today","Mkt Cap","P/E","vs 52w High"]):
                    _bhc.markdown(f'<div style="font-size:.7rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#3e4558;">{_bht}</div>', unsafe_allow_html=True)
                for _bd in _bench:
                    _bdc = "#00e5b0" if _bd["change_pct"]>=0 else "#ff5f5f"
                    _bds = "+" if _bd["change_pct"]>=0 else ""
                    _bdm = f"${_bd['mktcap']/1e12:.2f}T" if _bd["mktcap"]>=1e12 else f"${_bd['mktcap']/1e9:.1f}B" if _bd["mktcap"]>=1e9 else "—"
                    _bdp = f"{_bd['pe']:.1f}×" if _bd["pe"]>0 else "—"
                    _bdfhc = "#ff5f5f" if _bd["pfh"]<-20 else "#ffd426" if _bd["pfh"]<-10 else "#00e5b0"
                    _bdfhs = f"{_bd['pfh']:+.1f}%" if _bd["w52h"]>0 else "—"
                    _br = st.columns([1.2,2.4,1.2,1.2,1.5,1.2,1.5])
                    _br[0].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.8rem;font-weight:700;color:#4d8eff;padding:.4rem 0;">{_bd["ticker"]}</div>', unsafe_allow_html=True)
                    _br[1].markdown(f'<div style="font-size:.73rem;color:#8a8fa0;padding:.4rem 0;">{_bd["name"]}</div>', unsafe_allow_html=True)
                    _br[2].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;color:#e4eafd;padding:.4rem 0;">${_bd["price"]:,.2f}</div>', unsafe_allow_html=True)
                    _br[3].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;font-weight:700;color:{_bdc};padding:.4rem 0;">{_bds}{_bd["change_pct"]:.2f}%</div>', unsafe_allow_html=True)
                    _br[4].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.73rem;color:#8a8fa0;padding:.4rem 0;">{_bdm}</div>', unsafe_allow_html=True)
                    _br[5].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.73rem;color:#adc6ff;padding:.4rem 0;">{_bdp}</div>', unsafe_allow_html=True)
                    _br[6].markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;font-weight:700;color:{_bdfhc};padding:.4rem 0;">{_bdfhs}</div>', unsafe_allow_html=True)
                st.caption("⚠ Benchmarks are for research purposes only. Not financial advice.")

            # ── HUB 5: SIGNAL ALERT EMAIL ─────────────────────────────────────────
            with hub5:
                # FIX 5: Context guard — show dependency message if no signal data yet
                if _sh_comp is None:
                    st.markdown("""
                    <div style="background:rgba(255,212,38,0.05);border:1px solid rgba(255,212,38,0.2); border-left:3px solid #ffd426;border-radius:0 .6rem .6rem 0; padding:.85rem 1.3rem;margin-bottom:1rem;">
                      <div style="font-family:Manrope,sans-serif;font-size:0.84rem;font-weight:700; color:#ffd426;margin-bottom:.3rem;">⚠ Analysis required</div>
                      <div style="font-family:Manrope,sans-serif;font-size:0.82rem;color:#8a8fa0;line-height:1.5;">
                        Signal data isn't available yet. Run a full analysis from the sidebar first,
                        then return here to send an email alert.
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('''<div style="background:rgba(255,95,95,0.05);border:1px solid rgba(255,95,95,0.18); border-left:4px solid #ff5f5f;padding:.85rem 1.3rem;margin-bottom:1.2rem;border-radius:0 .6rem .6rem 0;">
                  <div style="font-size:.6rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#ff5f5f;margin-bottom:.25rem;">Signal Alert · Email Automation</div>
                  <div style="font-size:.8rem;color:#8a8fa0;line-height:1.6;">Send yourself an instant AI signal email. Requires SMTP credentials in Streamlit secrets.</div>
                </div>''', unsafe_allow_html=True)

                if _sh_comp:
                    _sa_sig = _sh_comp.get("verdict_short","—")
                    _sa_sco = _sh_comp.get("total_score",0)
                    _sa_xp  = _sh_comp.get("xgb_pct",0)
                    _sa_tp  = _sh_comp.get("take_profit",0)
                    _sa_sl  = _sh_comp.get("stop_loss",0)
                    _sa_rr  = _sh_comp.get("risk_reward",0)
                    _sa_c   = {"BUY":"#00e5b0","SELL":"#ff5f5f"}.get(_sa_sig,"#ffd426")

                    _sp1, _sp2 = st.columns([1,2])
                    with _sp1:
                        st.markdown(f'''<div style="background:linear-gradient(145deg,#0f1727,#141d30); border:2px solid {_sa_c};border-radius:.9rem;padding:1.5rem;text-align:center;">
                          <div style="font-size:.7rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;color:#3e4558;margin-bottom:.4rem;">Current Signal</div>
                          <div style="font-family:IBM Plex Mono,monospace;font-size:2rem;font-weight:800;color:{_sa_c};letter-spacing:.04em;">{_sa_sig}</div>
                          <div style="font-family:IBM Plex Mono,monospace;font-size:.82rem;color:#8a8fa0;margin-top:.3rem;">{_sa_sco:+.0f} / ±100</div>
                          <div style="font-family:IBM Plex Mono,monospace;font-size:.75rem;color:#e4eafd;margin-top:.4rem;">${last_close:.2f}</div>
                        </div>''', unsafe_allow_html=True)

                    with _sp2:
                        st.markdown(f'''<div style="background:#0f1727;border:1px solid #252f47;border-radius:.75rem;padding:1.2rem 1.5rem;">
                          <div style="font-size:.7rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#3e4558;margin-bottom:.6rem;">Email Preview · {ticker}</div>
                          {"".join(f'<div style="display:flex;justify-content:space-between;padding:.32rem 0;border-bottom:1px solid #1e2740;font-family:IBM Plex Mono,monospace;font-size:.72rem;"><span style="color:#3e4558;">{k}</span><span style="color:{vc};">{v}</span></div>' for k,v,vc in [
                            ("Signal",_sa_sig,_sa_c),("Score",f"{_sa_sco:+.0f}","#adc6ff"),
                            ("AI Forecast",f"{_sa_xp:+.2f}%","#adc6ff"),
                            ("Take Profit",f"${_sa_tp:.2f}","#00e5b0"),("Stop Loss",f"${_sa_sl:.2f}","#ff5f5f"),
                            ("Risk/Reward",f"{_sa_rr:.2f}×","#ffd426")])}
                          <div style="margin-top:.7rem;font-family:IBM Plex Mono,monospace;font-size:.62rem;color:#3e4558;">📬 {_user_email()}</div>
                        </div>''', unsafe_allow_html=True)

                    if st.button(f"📧 Send Signal Alert — {ticker}", key="hub_send_email", use_container_width=True):
                        _html_e = _build_signal_email_html(
                            _user_email(), ticker,
                            _sa_sig, last_close, _sa_sco, _sa_tp, _sa_sl, _sa_rr, _sa_xp)
                        _ok = _send_email(_user_email(),
                            f"Stockcast Signal · {ticker} — {_sa_sig} · {pd.Timestamp.now().strftime('%b %d')}",
                            _html_e)
                        if _ok:
                            st.success(f"✓ Signal alert sent to {_user_email()}")
                        else:
                            st.warning("⚠ Email not sent — configure SMTP_HOST, SMTP_USER, SMTP_PASS in Streamlit secrets.")
                else:
                    st.info("Run analysis first to generate signal data for email alerts.")

                # ── Signal Engine card in hub5 ──────────────────────────
                _se_key = f"se_signal_{ticker}"
                if _sh_comp and _se_key in st.session_state:
                    try:
                        _se_res = st.session_state[_se_key]
                        st.markdown(
                            "<div style='margin:.9rem 0 .3rem;font-family:Manrope,sans-serif;"
                            "font-size:.6rem;font-weight:800;letter-spacing:.18em;"
                            "text-transform:uppercase;color:#3d4760;'>▸ AI SIGNAL ENGINE SUMMARY</div>",
                            unsafe_allow_html=True,
                        )
                        _se_insight2 = generate_insight(df, float(np.clip(
                            _sh_comp.get("total_score", 0) / 135.0, -1.0, 1.0)), _se_res)
                        render_signal_card(_se_res, _se_insight2, ticker=ticker)
                    except Exception as _se_hub5_err:
                        st.warning(f"Signal Engine card unavailable: {_se_hub5_err}")
                # ────────────────────────────────────────────────────────

                # SMTP setup note removed from UI — configure via Streamlit secrets.toml (server-side only)

            # ── HUB 6: INVESTOR REPORT ────────────────────────────────────────────
            with hub6:
                # FIX 5: Context guard for report tab
                if _sh_comp is None:
                    st.markdown("""
                    <div style="background:rgba(255,212,38,0.05);border:1px solid rgba(255,212,38,0.2); border-left:3px solid #ffd426;border-radius:0 .6rem .6rem 0; padding:.85rem 1.3rem;margin-bottom:1rem;">
                      <div style="font-family:Manrope,sans-serif;font-size:0.84rem;font-weight:700; color:#ffd426;margin-bottom:.3rem;">⚠ Analysis required</div>
                      <div style="font-family:Manrope,sans-serif;font-size:0.82rem;color:#8a8fa0;line-height:1.5;">
                        Signal and model data isn't ready. Run a full analysis first,
                        then return here to download the investor report CSV.
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('''<div style="background:rgba(0,229,176,0.05);border:1px solid rgba(0,229,176,0.18); border-left:4px solid #00e5b0;padding:.85rem 1.3rem;margin-bottom:1.2rem;border-radius:0 .6rem .6rem 0;">
                  <div style="font-size:.6rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#00e5b0;margin-bottom:.25rem;">Investor-Ready Report Generator</div>
                  <div style="font-size:.8rem;color:#8a8fa0;line-height:1.6;">One-click CSV export — price data, model quality, signal intelligence, and risk metrics. Ready for pitch decks or treasury memos.</div>
                </div>''', unsafe_allow_html=True)

                _ir1, _ir2 = st.columns(2)
                _lc2  = float(df["Close"].squeeze().iloc[-1])
                _h52  = float(df["Close"].squeeze().max())
                _l52  = float(df["Close"].squeeze().min())

                with _ir1:
                    st.markdown(f'''<div style="background:linear-gradient(145deg,#0f1727,#141d30);border:1px solid #252f47;border-top:2px solid #00e5b0;border-radius:.75rem;padding:1.3rem 1.5rem;">
                      <div style="font-size:.7rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#00e5b0;margin-bottom:.8rem;">Price Summary · {ticker}</div>
                      {"".join(f'<div style="display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid #1e2740;font-family:IBM Plex Mono,monospace;font-size:.72rem;"><span style="color:#3e4558;">{lbl}</span><span style="color:{vc};">{val}</span></div>' for lbl,val,vc in [
                        ("Last Close ($)",f"{_lc2:.2f}","#e4eafd"),
                        ("52-Week High ($)",f"{_h52:.2f}","#00e5b0"),
                        ("52-Week Low ($)",f"{_l52:.2f}","#ff5f5f"),
                        ("% from 52w High",f"{((_lc2-_h52)/_h52*100):+.1f}%","#ffd426"),
                        ("Days Analysed",str(len(df)),"#8a8fa0")])}
                    </div>''', unsafe_allow_html=True)

                with _ir2:
                    _isig = _sh_comp.get("verdict_short","—") if _sh_comp else "—"
                    _isigc = {"BUY":"#00e5b0","SELL":"#ff5f5f"}.get(_isig,"#ffd426")
                    st.markdown(f'''<div style="background:linear-gradient(145deg,#0f1727,#141d30);border:1px solid #252f47;border-top:2px solid #4d8eff;border-radius:.75rem;padding:1.3rem 1.5rem;">
                      <div style="font-size:.7rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#4d8eff;margin-bottom:.8rem;">AI Model &amp; Signals</div>
                      {"".join(f'<div style="display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid #1e2740;font-family:IBM Plex Mono,monospace;font-size:.72rem;"><span style="color:#3e4558;">{lbl}</span><span style="color:{vc};">{val}</span></div>' for lbl,val,vc in [
                        ("RMSE ($)",f"{rmse:.2f}","#e4eafd"),("MAE ($)",f"{mae:.2f}","#e4eafd"),
                        ("MAPE (%)",f"{mape:.2f}","#ffd426"),("R²",f"{r2:.4f}","#00e5b0"),
                        ("Signal",_isig,_isigc),
                        ("Score (±100)",f"{_sh_comp.get('total_score',0):+.0f}" if _sh_comp else "—","#adc6ff"),
                        ("XGBoost %",f"{_sh_comp.get('xgb_pct',0):+.2f}%" if _sh_comp else "—","#adc6ff"),
                        ("Take Profit",f"${_sh_comp.get('take_profit',0):.2f}" if _sh_comp else "—","#00e5b0"),
                        ("Stop Loss",f"${_sh_comp.get('stop_loss',0):.2f}" if _sh_comp else "—","#ff5f5f"),
                        ("Risk/Reward",f"{_sh_comp.get('risk_reward',0):.2f}×" if _sh_comp else "—","#ffd426")])}
                    </div>''', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                _csv_data = _build_investor_csv(ticker, df, preds, actual, rmse, mae, mape, r2, _sh_comp, se_signal=st.session_state.get(f"se_signal_{ticker}"))
                st.download_button(
                    label=f"⬇  Download Investor Report — {ticker} · {pd.Timestamp.now().strftime('%Y-%m-%d')}.csv",
                    data=_csv_data,
                    file_name=f"stockcast_{ticker}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv", use_container_width=True, key="hub_investor_dl")
                st.caption("⚠ For educational and research purposes only. Not financial advice.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;margin-top:4rem;padding:2rem 1rem;border-top:1px solid #1e2740;">
  <div style="display:flex;align-items:center;justify-content:center;gap:1.2rem;flex-wrap:wrap;margin-bottom:1rem;">
    <span class="trust-item" style="font-size:.7rem;"><span class="trust-item-dot"></span>Data via Yahoo Finance</span>
    <span style="color:#1e2740;">·</span>
    <span class="trust-item" style="font-size:.7rem;"><span class="trust-item-dot" style="background:#4d8eff;"></span>Auth by Supabase</span>
    <span style="color:#1e2740;">·</span>
    <span class="trust-item" style="font-size:.7rem;"><span class="trust-item-dot" style="background:#ffd426;"></span>AI-Powered Analysis</span>
    <span style="color:#1e2740;">·</span>
    <span class="trust-item" style="font-size:.7rem;"><span class="trust-item-dot" style="background:#00e5b0;"></span>Shariah Screening</span>
  </div>
  <div style="margin-bottom:.7rem;">
    <a href="/privacy" target="_blank" style="color:#3e4558;text-decoration:none;font-family:IBM Plex Mono,monospace;font-size:.7rem;letter-spacing:.08em;margin:0 .6rem;">Privacy Policy</a>
    <span style="color:#1e2740;">·</span>
    <a href="/terms" target="_blank" style="color:#3e4558;text-decoration:none;font-family:IBM Plex Mono,monospace;font-size:.7rem;letter-spacing:.08em;margin:0 .6rem;">Terms of Service</a>
  </div>
  <div style="margin-bottom:.5rem;">
    <span class="disclaimer-pill">⚠ Stockcast is for educational and research purposes only. Not financial advice. Past performance does not guarantee future results. Always consult a licensed financial advisor.</span>
  </div>
  <div style="font-family:IBM Plex Mono,monospace;font-size:.5rem;color:#1e2740;letter-spacing:.08em;margin-top:.6rem;">
    © 2026 Stockcast · Built by Muawwiz Ghani · v3.0
  </div>
</div>
""", unsafe_allow_html=True)

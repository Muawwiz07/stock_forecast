# =============================================================================
# Copyright (c) 2026 Stockcast. All Rights Reserved.
# PROPRIETARY AND CONFIDENTIAL
# signals.py — Signal engine: generate_signal, generate_insight, render_signal_card
# =============================================================================

import numpy as np
import pandas as pd
import streamlit as st


def generate_signal(df: pd.DataFrame, sentiment_score: float) -> dict:
    """
    Generate a BUY / SELL / HOLD signal with confidence score.

    Parameters
    ----------
    df : pd.DataFrame       — stock dataframe with at least a 'Close' column.
    sentiment_score : float — news/sentiment score in range [-1.0, +1.0].

    Returns
    -------
    dict with keys: signal, confidence, trend, sentiment, conflict, volatility, details
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
        momentum_score = float(np.tanh(trend_pct / 3.0))
        if   trend_pct >  1.0: trend_label = "Uptrend"
        elif trend_pct < -1.0: trend_label = "Downtrend"
        else:                  trend_label = "Sideways"

    # ── 2. MA20 position ──────────────────────────────────────────────────────
    if "MA20" in df.columns:
        ma20 = float(df["MA20"].squeeze().iloc[-1])
    else:
        ma20 = (float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else float(close.mean()))

    last_price = float(close.iloc[-1])
    ma_gap_pct = (last_price - ma20) / ma20 * 100 if ma20 != 0 else 0.0
    ma_score   = float(np.tanh(ma_gap_pct / 5.0))

    # ── 3. Volatility regime ──────────────────────────────────────────────────
    if len(close) >= 20:
        vol_20d = float(close.pct_change().dropna().rolling(20).std().iloc[-1]) * 100
    else:
        vol_20d = 1.5
    if   vol_20d > 3.0: volatility_label = "High"
    elif vol_20d < 1.0: volatility_label = "Low"
    else:               volatility_label = "Normal"

    # ── 4. Sentiment label ────────────────────────────────────────────────────
    if   sentiment_score >  0.20: sentiment_label = "Positive"
    elif sentiment_score < -0.20: sentiment_label = "Negative"
    else:                         sentiment_label = "Neutral"

    # ── 5. Conflict detection ─────────────────────────────────────────────────
    conflict = (
        momentum_score > 0.15 and sentiment_score < -0.20
        or momentum_score < -0.15 and sentiment_score > 0.20
    )

    # ── 6. Signal decision ────────────────────────────────────────────────────
    technical_score = 0.55 * momentum_score + 0.45 * ma_score
    if   technical_score >  0.15 and sentiment_score >= -0.20: signal = "BUY"
    elif technical_score < -0.15 and sentiment_score <=  0.20: signal = "SELL"
    else:                                                       signal = "HOLD"

    # ── 7. Confidence ─────────────────────────────────────────────────────────
    strength         = min(abs(technical_score) / 0.6, 1.0)
    trend_vs_ma      = 1.0 - abs(momentum_score - ma_score) / 2.0
    trend_vs_sent    = 1.0 - abs(momentum_score - sentiment_score) / 2.0
    ma_vs_sent       = 1.0 - abs(ma_score - sentiment_score) / 2.0
    agreement        = trend_vs_ma * 0.4 + trend_vs_sent * 0.35 + ma_vs_sent * 0.25
    sentiment_weight = min(abs(sentiment_score) / 0.6, 1.0)
    vol_penalty      = {"High": 0.75, "Normal": 1.0, "Low": 1.05}.get(volatility_label, 1.0)
    conflict_penalty = 0.65 if conflict else 1.0

    raw_conf = (strength * 0.40 + agreement * 0.30 + sentiment_weight * 0.30)
    raw_conf *= vol_penalty * conflict_penalty
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
            "momentum_score":  round(momentum_score, 3),
            "ma_score":        round(ma_score, 3),
            "technical_score": round(technical_score, 3),
            "sentiment_score": round(sentiment_score, 3),
            "agreement":       round(agreement, 3),
            "strength":        round(strength, 3),
            "vol_20d":         round(vol_20d, 2),
            "trend_pct":       round(trend_pct, 2),
            "last_price":      round(last_price, 2),
            "ma20":            round(ma20, 2),
            "ma_gap_pct":      round(ma_gap_pct, 2),
        },
    }


def generate_insight(df: pd.DataFrame, sentiment_score: float, signal: dict) -> str:
    """
    Generate 2–3 sentences of plain-English AI insight.
    Language adapts to magnitude, conflict, and volatility.
    """
    sig        = signal["signal"]
    conf       = signal["confidence"]
    trend      = signal["trend"]
    sentiment  = signal["sentiment"]
    conflict   = signal["conflict"]
    volatility = signal["volatility"]
    details    = signal["details"]
    ma20       = details["ma20"]
    ma_gap     = details["ma_gap_pct"]
    trend_pct  = details["trend_pct"]
    vol_20d    = details["vol_20d"]
    sent_raw   = details["sentiment_score"]

    def _gap_word(pct):
        a = abs(pct)
        if a < 1.0: return "just barely"
        if a < 3.0: return "modestly"
        if a < 6.0: return "clearly"
        return "comfortably"

    def _trend_word(pct):
        a = abs(pct)
        if a < 1.5: return "drifting"
        if a < 3.0: return "moving"
        if a < 6.0: return "climbing" if pct > 0 else "sliding"
        return "surging" if pct > 0 else "falling sharply"

    def _sent_word(s):
        a = abs(s)
        if a < 0.25: return "mildly"
        if a < 0.55: return "noticeably"
        return "strongly"

    direction  = "above" if ma_gap >= 0 else "below"
    trend_verb = _trend_word(trend_pct)
    gap_adv    = _gap_word(ma_gap)

    # Sentence 1: Technical picture
    if trend == "Uptrend":
        s1 = f"The stock has been {trend_verb} higher over the past week and sits {gap_adv} {direction} its 20-day average (${ma20:.2f}) — the short-term tape is constructive."
    elif trend == "Downtrend":
        s1 = f"The stock has been {trend_verb} over the past week, trading {gap_adv} {direction} its 20-day average (${ma20:.2f}) — sellers have been in control."
    else:
        if abs(ma_gap) < 1.0:
            s1 = f"Price is essentially flat, hugging its 20-day average (${ma20:.2f}) without committing to a direction — the market is waiting for a catalyst."
        elif ma_gap > 0:
            s1 = f"The stock is drifting sideways but still holding {gap_adv} above its 20-day average (${ma20:.2f}), suggesting underlying support."
        else:
            s1 = f"Price is stuck in a sideways range, sitting {gap_adv} below its 20-day average (${ma20:.2f}) — buyers haven't stepped in yet."

    # Sentence 2: Sentiment
    sent_adv = _sent_word(sent_raw)
    if conflict:
        if trend == "Uptrend":
            s2 = f"What makes this tricky: despite the price action, news flow is {sent_adv} negative — that kind of disagreement often signals a setup worth watching carefully rather than acting on immediately."
        else:
            s2 = f"Interestingly, news sentiment has turned {sent_adv} positive even as the price has been weak — worth watching whether buyers step in to close that gap, or the negativity drags sentiment down."
    elif sentiment == "Positive":
        s2 = f"News flow is {sent_adv} bullish right now, which tends to attract momentum buyers and reinforces the technical picture." if sent_raw > 0.55 else f"The news backdrop is {sent_adv} positive — not a screaming headline moment, but it adds a gentle tailwind to the setup."
    elif sentiment == "Negative":
        s2 = f"Sentiment is {sent_adv} negative at the moment — that kind of headline pressure can weigh on price action even when technicals look reasonable." if sent_raw < -0.55 else f"The news backdrop carries a {sent_adv} negative tilt, which adds some friction to any recovery attempt."
    else:
        s2 = "News sentiment is broadly neutral — no major catalyst in either direction, so the price action is doing most of the talking."

    # Sentence 3: Expectation
    vol_note = f" Keep in mind this stock is moving roughly {vol_20d:.1f}% a day on average — position sizing matters." if volatility == "High" else (f" Low volatility ({vol_20d:.1f}% daily) means moves could be smaller than usual." if volatility == "Low" else "")

    if sig == "BUY":
        if conf >= 75:
            s3 = f"The trend, MA position, and sentiment all point in the same direction — that alignment is what drives the {conf:.0f}% confidence here.{vol_note}"
        elif conf >= 50:
            s3 = f"There's a reasonable case for upside, but it's not a slam dunk at {conf:.0f}% confidence. A partial position or a tighter stop makes sense here.{vol_note}"
        else:
            s3 = f"This reads as a tentative buy at best — confidence is only {conf:.0f}%, reflecting the mixed picture. Wait for the setup to sharpen before adding size.{vol_note}"
    elif sig == "SELL":
        if conf >= 75:
            s3 = f"Downside pressure looks real — both the technicals and sentiment are aligned, giving this a {conf:.0f}% confidence reading.{vol_note}"
        elif conf >= 50:
            s3 = f"The bear case is present but not overwhelming ({conf:.0f}% confidence). Consider reducing exposure rather than pressing short aggressively.{vol_note}"
        else:
            s3 = f"Sell signals exist, but confidence sits at just {conf:.0f}%. Better to protect existing positions than to act aggressively on this read.{vol_note}"
    else:
        if conflict:
            s3 = f"With trend and sentiment pointing in opposite directions, sitting on the sidelines is the most honest call. The next few sessions should resolve which force wins out.{vol_note}"
        elif conf < 30:
            s3 = f"Nothing is clear enough to act on right now. Low confidence ({conf:.0f}%) means the risk of being wrong is high in either direction.{vol_note}"
        else:
            s3 = f"No strong edge in either direction at this stage — patience is the trade.{vol_note}"

    return f"{s1} {s2} {s3}"


# ── Signal card color palette ─────────────────────────────────────────────────
_COLORS = {
    "BUY":  {"border": "#00e5b0", "bg": "rgba(0,229,176,0.08)",  "text": "#00e5b0", "emoji": "🟢"},
    "SELL": {"border": "#ff5f5f", "bg": "rgba(255,95,95,0.08)",  "text": "#ff5f5f", "emoji": "🔴"},
    "HOLD": {"border": "#ffd426", "bg": "rgba(255,212,38,0.08)", "text": "#ffd426", "emoji": "🟡"},
}


def render_signal_card(signal_result: dict, insight_text: str, ticker: str = "") -> None:
    """Render the full signal UI: card, AI insight box, reason rows."""
    sig        = signal_result["signal"]
    conf       = signal_result["confidence"]
    trend      = signal_result["trend"]
    sent       = signal_result["sentiment"]
    conflict   = signal_result.get("conflict", False)
    volatility = signal_result.get("volatility", "Normal")
    c          = _COLORS[sig]
    conf_int   = int(conf)
    bar_filled = int(conf / 5)

    if   conf >= 70: conf_label = "HIGH CONFIDENCE"
    elif conf >= 45: conf_label = "MODERATE"
    else:            conf_label = "LOW CONFIDENCE"

    bar_html = "".join(
        f'<span style="display:inline-block;width:20px;height:9px;margin-right:2px;'
        f'border-radius:2px;background:{c["border"]};'
        f'opacity:{1.0 if i < bar_filled else 0.12};"></span>'
        for i in range(20)
    )
    trend_color = "#00e5b0" if trend == "Uptrend" else ("#ff5f5f" if trend == "Downtrend" else "#ffd426")
    sent_color  = "#00e5b0" if sent == "Positive" else ("#ff5f5f" if sent == "Negative" else "#8a8fa0")
    vol_color   = "#ff5f5f" if volatility == "High" else ("#8a8fa0" if volatility == "Low" else "#4d8eff")
    header_label = f"SIGNAL · {ticker}" if ticker else "SIGNAL"

    conflict_html = (
        '<div style="display:inline-flex;align-items:center;gap:.35rem;'
        'background:rgba(255,212,38,0.1);border:1px solid rgba(255,212,38,0.35);'
        'border-radius:2rem;padding:.2rem .7rem;margin-top:.7rem;">'
        '<span style="font-size:.65rem;color:#ffd426;">⚠</span>'
        '<span style="font-family:Manrope,sans-serif;font-size:.65rem;font-weight:700;'
        'color:#ffd426;letter-spacing:.06em;">TREND–SENTIMENT CONFLICT</span>'
        '</div>' if conflict else ""
    )

    st.markdown(f"""
        <div style="background:{c['bg']};border:1.5px solid {c['border']};
          border-left:5px solid {c['border']};border-radius:0 .75rem .75rem 0;
          padding:1.4rem 1.8rem;margin-bottom:1rem;">
            <div style="font-family:Manrope,sans-serif;font-size:.6rem;font-weight:800;
              letter-spacing:.18em;text-transform:uppercase;color:#3d4760;margin-bottom:.6rem;">{header_label}</div>
            <div style="display:flex;align-items:center;justify-content:space-between;
              flex-wrap:wrap;gap:1rem;margin-bottom:1rem;">
                <div style="font-family:IBM Plex Mono,monospace;font-size:2.4rem;
                  font-weight:800;color:{c['text']};letter-spacing:.08em;line-height:1;">
                    {c['emoji']}&nbsp;{sig}
                </div>
                <div style="text-align:right;">
                    <div style="font-family:IBM Plex Mono,monospace;font-size:2rem;
                      font-weight:700;color:{c['text']};">{conf_int}%</div>
                    <div style="font-family:Manrope,sans-serif;font-size:.6rem;
                      letter-spacing:.14em;text-transform:uppercase;
                      color:{c['text']};font-weight:700;margin-top:.1rem;">{conf_label}</div>
                </div>
            </div>
            <div style="margin-bottom:1.1rem;">{bar_html}</div>
            <div style="display:flex;gap:1.8rem;flex-wrap:wrap;">
                <div>
                    <div style="font-family:Manrope,sans-serif;font-size:.58rem;letter-spacing:.14em;
                      text-transform:uppercase;color:#3d4760;font-weight:700;">Trend</div>
                    <div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;
                      color:{trend_color};font-weight:700;margin-top:.2rem;">{trend}</div>
                </div>
                <div>
                    <div style="font-family:Manrope,sans-serif;font-size:.58rem;letter-spacing:.14em;
                      text-transform:uppercase;color:#3d4760;font-weight:700;">Sentiment</div>
                    <div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;
                      color:{sent_color};font-weight:700;margin-top:.2rem;">{sent}</div>
                </div>
                <div>
                    <div style="font-family:Manrope,sans-serif;font-size:.58rem;letter-spacing:.14em;
                      text-transform:uppercase;color:#3d4760;font-weight:700;">Volatility</div>
                    <div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;
                      color:{vol_color};font-weight:700;margin-top:.2rem;">{volatility}</div>
                </div>
            </div>
            {conflict_html}
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background:rgba(77,142,255,0.05);border:1px solid rgba(77,142,255,0.2);
          border-left:4px solid #4d8eff;border-radius:0 .75rem .75rem 0;
          padding:1.1rem 1.5rem;margin-bottom:.8rem;">
            <div style="font-family:Manrope,sans-serif;font-size:.6rem;font-weight:800;
              letter-spacing:.16em;text-transform:uppercase;color:#4d8eff;margin-bottom:.5rem;">💡 AI Insight</div>
            <div style="font-family:Manrope,sans-serif;font-size:.84rem;color:#b8c4d8;
              line-height:1.7;">{insight_text}</div>
        </div>
    """, unsafe_allow_html=True)

    rows_html = "".join(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:.38rem 0;border-bottom:1px solid #1e2740;'
        f'font-family:IBM Plex Mono,monospace;font-size:.72rem;">'
        f'<span style="color:#3e4558;">{label}</span>'
        f'<span style="color:{color};font-weight:700;">{value}</span></div>'
        for label, value, color in [
            ("Trend Direction", trend,                                     trend_color),
            ("Sentiment",       sent,                                      sent_color),
            ("Volatility",      volatility,                                vol_color),
            ("Signal Conflict", "Yes ⚠" if conflict else "None detected", "#ffd426" if conflict else "#3e4558"),
        ]
    )
    st.markdown(f"""
        <div style="background:#0f1727;border:1px solid #1e2d45;
          border-radius:.6rem;padding:.9rem 1.2rem;">
            <div style="font-family:Manrope,sans-serif;font-size:.6rem;font-weight:800;
              letter-spacing:.16em;text-transform:uppercase;color:#3d4760;margin-bottom:.5rem;">Reason</div>
            {rows_html}
        </div>
    """, unsafe_allow_html=True)


def run_signal_ui(df: pd.DataFrame, sentiment_score: float, ticker: str = "") -> dict:
    """One-call wrapper: generate signal + insight + render card. Returns signal dict."""
    signal  = generate_signal(df, sentiment_score)
    insight = generate_insight(df, sentiment_score, signal)
    render_signal_card(signal, insight, ticker=ticker)
    return signal

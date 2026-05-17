# =============================================================================
# Copyright (c) 2026 Stockcast. All Rights Reserved.
# PROPRIETARY AND CONFIDENTIAL
# sentiment.py — News sentiment NLP pipeline
#
# Priority:
#   1. FinBERT (ProsusAI/finbert) — financial-domain BERT, most accurate
#   2. TextBlob                   — lightweight general NLP, good fallback
#   3. Keyword scorer             — zero-dependency last resort
#
# Usage:
#   from sentiment import analyze_headlines, render_sentiment_ui
#   score, results, method = analyze_headlines(headlines)
# =============================================================================

import re
import logging
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger("stockcast")

# ── Tier 1: FinBERT ───────────────────────────────────────────────────────────

_FINBERT_PIPE = None   # lazy-loaded on first use
_FINBERT_OK   = False

def _load_finbert():
    """Attempt to load FinBERT pipeline (lazy, cached in module scope)."""
    global _FINBERT_PIPE, _FINBERT_OK
    if _FINBERT_OK:
        return True
    try:
        from transformers import pipeline as hf_pipeline
        _FINBERT_PIPE = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,           # return all 3 label scores
            truncation=True,
            max_length=512,
        )
        _FINBERT_OK = True
        logger.info("sentiment: FinBERT loaded successfully")
        return True
    except Exception as e:
        logger.info("sentiment: FinBERT unavailable (%s) — falling back", type(e).__name__)
        return False


def _score_finbert(headlines: list) -> list:
    """
    Run FinBERT on a list of headline strings.
    Returns list of dicts: {headline, polarity, label, method}
    polarity ∈ [-1, +1]: positive=+score, negative=-score, neutral=0
    """
    results = []
    for text in headlines:
        try:
            raw = _FINBERT_PIPE(text[:512])[0]  # list of {label, score}
            scores = {r["label"].lower(): r["score"] for r in raw}
            polarity = scores.get("positive", 0) - scores.get("negative", 0)
            label = max(scores, key=scores.get)
            results.append({
                "headline": text,
                "polarity": round(polarity, 4),
                "label":    label.upper(),
                "method":   "FinBERT",
                "pos":      round(scores.get("positive", 0), 3),
                "neg":      round(scores.get("negative", 0), 3),
                "neu":      round(scores.get("neutral",  0), 3),
            })
        except Exception as e:
            logger.debug("FinBERT single headline failed: %s", e)
            continue
    return results


# ── Tier 2: TextBlob ──────────────────────────────────────────────────────────

def _score_textblob(headlines: list) -> list:
    """Run TextBlob on headlines. Falls back to keyword if import fails."""
    try:
        from textblob import TextBlob
        results = []
        for text in headlines:
            pol = TextBlob(text).sentiment.polarity
            if   pol >  0.05: label = "POSITIVE"
            elif pol < -0.05: label = "NEGATIVE"
            else:             label = "NEUTRAL"
            results.append({
                "headline": text,
                "polarity": round(pol, 4),
                "label":    label,
                "method":   "TextBlob",
                "pos": max(pol, 0),
                "neg": max(-pol, 0),
                "neu": 1 - abs(pol),
            })
        return results
    except ImportError:
        logger.info("sentiment: TextBlob unavailable — using keyword scorer")
        return _score_keywords(headlines)


# ── Tier 3: Keyword scorer ────────────────────────────────────────────────────

_BULLISH_KW = [
    "surge", "soar", "rally", "gain", "rise", "jump", "climb", "beat",
    "record", "profit", "growth", "strong", "upgrade", "buy", "bull",
    "outperform", "exceed", "positive", "recover", "rebound", "boost",
    "upside", "breakout", "expansion", "dividend", "buyback",
]
_BEARISH_KW = [
    "fall", "drop", "crash", "plunge", "decline", "loss", "miss", "cut",
    "downgrade", "sell", "bear", "weak", "negative", "risk", "concern",
    "layoff", "recall", "lawsuit", "fine", "penalty", "default", "debt",
    "warning", "fraud", "probe", "investigation", "bankruptcy",
]

def _score_keywords(headlines: list) -> list:
    results = []
    for text in headlines:
        words  = re.findall(r'\b\w+\b', text.lower())
        bulls  = sum(1 for w in words if w in _BULLISH_KW)
        bears  = sum(1 for w in words if w in _BEARISH_KW)
        total  = bulls + bears or 1
        pol    = (bulls - bears) / total
        if   pol >  0.1: label = "POSITIVE"
        elif pol < -0.1: label = "NEGATIVE"
        else:            label = "NEUTRAL"
        results.append({
            "headline": text,
            "polarity": round(pol, 4),
            "label":    label,
            "method":   "Keywords",
            "pos": bulls / total,
            "neg": bears / total,
            "neu": 1 - abs(pol),
        })
    return results


# ── Public API ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def analyze_headlines(headlines: tuple) -> tuple:
    """
    Score a tuple of headline strings using the best available NLP method.
    (tuple instead of list so Streamlit can hash it for caching)

    Returns
    -------
    (avg_polarity: float, results: list[dict], method: str)
    """
    if not headlines:
        return 0.0, [], "none"

    texts = list(headlines)

    # Try FinBERT first
    if _load_finbert():
        results = _score_finbert(texts)
        if results:
            avg = sum(r["polarity"] for r in results) / len(results)
            return round(avg, 4), results, "FinBERT"

    # TextBlob / keyword fallback
    results = _score_textblob(texts)
    method  = results[0]["method"] if results else "Keywords"
    avg     = sum(r["polarity"] for r in results) / len(results) if results else 0.0
    return round(avg, 4), results, method


def render_sentiment_ui(ticker: str, news_items: list, _L: dict = None) -> float:
    """
    Fetch headlines, run NLP, render the full sentiment panel.
    Returns the avg_polarity float for downstream use in signal engine.

    Parameters
    ----------
    ticker     : str   — stock symbol (for display)
    news_items : list  — from av_get_news(ticker) — list of {title: str}
    _L         : dict  — language dict (optional)

    Returns
    -------
    float — avg sentiment polarity ∈ [-1, +1]
    """
    if _L is None:
        _L = {}

    C_GREEN  = "#00e5b0"
    C_RED    = "#ff5f5f"
    C_YELLOW = "#ffd426"
    C_GREY   = "#3e4558"

    if not news_items:
        st.info(_L.get("no_recent_news", "No recent news found for this ticker."))
        return 0.0

    headlines = tuple(
        item.get("title", "")
        for item in news_items[:12]
        if item.get("title", "").strip()
    )

    if not headlines:
        st.info(_L.get("no_recent_news", "No recent news found."))
        return 0.0

    with st.spinner("Analysing headlines…"):
        avg_polarity, results, method = analyze_headlines(headlines)

    if not results:
        st.warning("Could not score headlines.")
        return 0.0

    # ── Method badge ──────────────────────────────────────────────────────────
    method_color = {
        "FinBERT":  "#4d8eff",
        "TextBlob": "#00e5b0",
        "Keywords": "#ffd426",
    }.get(method, "#8a8fa0")

    method_desc = {
        "FinBERT":  "Financial BERT — domain-trained transformer model",
        "TextBlob": "TextBlob — general NLP polarity scoring",
        "Keywords": "Keyword scorer — lightweight fallback",
    }.get(method, method)

    sent_color = C_GREEN if avg_polarity > 0.05 else (C_RED if avg_polarity < -0.05 else C_YELLOW)
    sent_label = "POSITIVE" if avg_polarity > 0.05 else ("NEGATIVE" if avg_polarity < -0.05 else "NEUTRAL")

    st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;
          background:rgba(77,142,255,0.06);border:1px solid rgba(77,142,255,0.2);
          border-left:3px solid {sent_color};padding:.75rem 1.2rem;
          border-radius:0 .5rem .5rem 0;margin-bottom:.75rem;gap:.5rem;">
            <div>
                <span style="font-family:Manrope,sans-serif;font-size:.72rem;
                  color:#8a8fa0;font-weight:600;">Avg Sentiment&nbsp;</span>
                <span style="font-family:IBM Plex Mono,monospace;font-size:.82rem;
                  color:{sent_color};font-weight:800;">{sent_label}&nbsp;({avg_polarity:+.3f})</span>
                <span style="font-family:Manrope,sans-serif;font-size:.68rem;
                  color:#3e4558;">&nbsp;·&nbsp;{len(results)} headlines</span>
            </div>
            <div style="background:rgba(0,0,0,0.3);border:1px solid {method_color}33;
              border-radius:2rem;padding:.2rem .75rem;">
                <span style="font-family:IBM Plex Mono,monospace;font-size:.6rem;
                  color:{method_color};font-weight:700;">⚡ {method}</span>
                <span style="font-family:Manrope,sans-serif;font-size:.58rem;
                  color:#3e4558;">&nbsp;— {method_desc}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    sc_df  = pd.DataFrame(results)
    labels = [h[:55] + "…" if len(h) > 55 else h for h in sc_df["headline"]]
    colors = [C_GREEN if p > 0.05 else (C_RED if p < -0.05 else C_YELLOW) for p in sc_df["polarity"]]

    fig = go.Figure(go.Bar(
        x=sc_df["polarity"],
        y=labels,
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Polarity: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0,            line_color=C_GREY,   line_width=1)
    fig.add_vline(x=avg_polarity, line_color=sent_color, line_width=1.5, line_dash="dot")

    fig.update_layout(
        paper_bgcolor="#080f1e",
        plot_bgcolor="#080f1e",
        font=dict(family="IBM Plex Mono, monospace", color="#8a8fa0", size=10),
        margin=dict(l=10, r=20, t=40, b=30),
        height=max(220, len(results) * 34),
        title=dict(
            text=f"{ticker} · News Sentiment — {method}",
            font=dict(color=C_GREEN, size=12),
        ),
        xaxis=dict(
            title="Polarity  (negative ← 0 → positive)",
            range=[-1, 1],
            gridcolor="#1e2740",
            zeroline=False,
            tickfont=dict(color="#8a8fa0", size=9),
        ),
        yaxis=dict(gridcolor="#1e2740", tickfont=dict(size=9)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-headline detail table (FinBERT only — shows pos/neg/neu scores) ──
    if method == "FinBERT":
        with st.expander("📊 Full FinBERT score breakdown"):
            tbl = sc_df[["headline", "pos", "neg", "neu", "label"]].copy()
            tbl.columns = ["Headline", "Positive", "Negative", "Neutral", "Label"]
            tbl["Headline"] = tbl["Headline"].str[:70]
            st.dataframe(tbl.style.format({"Positive": "{:.3f}", "Negative": "{:.3f}", "Neutral": "{:.3f}"}),
                         use_container_width=True, hide_index=True)

    st.caption(
        f"💡 Sentiment scored via {method} on {len(results)} recent Yahoo Finance headlines. "
        "Combine with your own reading of the news for best results."
    )

    return avg_polarity

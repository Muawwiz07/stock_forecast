import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from supabase import create_client
import warnings
warnings.filterwarnings('ignore')

# ── Supabase config ────────────────────────────────────────────────────────────
SUPABASE_URL = "https://tpaqlfjszguinigygxcq.supabase.co"
SUPABASE_KEY = "sb_publishable_k3YRnuw5-wzS3VzpCpzQdg_0B-XldpV"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Session state: Auth ────────────────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="StockCast — Market Intelligence", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ── Session state: Watchlist & Alerts ─────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []           # list of ticker strings
if "alert_signals" not in st.session_state:
    st.session_state.alert_signals = {}       # {ticker: last_known_verdict}
if "signal_alerts_fired" not in st.session_state:
    st.session_state.signal_alerts_fired = [] # banners to show this run

# ── Ultra Pro CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

/* ── ROOT VARS ── */
:root {
    --bg:         #080b0f;
    --bg2:        #0d1117;
    --bg3:        #111820;
    --bg4:        #151f2a;
    --green:      #00e5b0;
    --green-dim:  rgba(0,229,176,0.10);
    --green-glow: rgba(0,229,176,0.20);
    --red:        #ff3d57;
    --red-dim:    rgba(255,61,87,0.10);
    --blue:       #4da6ff;
    --blue-dim:   rgba(77,166,255,0.10);
    --yellow:     #ffdd2d;
    --yellow-dim: rgba(255,221,45,0.10);
    --purple:     #a78bfa;
    --t1:         #eef2f7;
    --t2:         #7a8fa8;
    --t3:         #3d5068;
    --border:     #182030;
    --border2:    #243346;
    --mono:       'IBM Plex Mono', monospace;
    --sans:       'Inter', sans-serif;
    --radius:     4px;
}

/* ── GLOBAL ── */
html, body, [class*="css"], [data-testid="stApp"],
[data-testid="stAppViewContainer"], .main {
    font-family: var(--sans);
    background-color: var(--bg) !important;
    color: var(--t1) !important;
}
.block-container { padding: 1.2rem 2rem !important; max-width: 100% !important; }

/* subtle noise/scanline texture */
[data-testid="stApp"]::before {
    content: '';
    position: fixed; inset: 0;
    background:
        repeating-linear-gradient(0deg, transparent, transparent 2px,
            rgba(0,0,0,0.025) 2px, rgba(0,0,0,0.025) 4px),
        radial-gradient(ellipse at 0% 0%, rgba(0,229,176,0.03) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 100%, rgba(77,166,255,0.03) 0%, transparent 50%);
    pointer-events: none; z-index: 9999;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #0d1117 0%, #080b0f 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--t2) !important; }
[data-testid="stSidebar"] input {
    background-color: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    color: var(--green) !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 2px var(--green-glow), 0 0 16px rgba(0,229,176,0.08) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    color: var(--green) !important;
    border: 1px solid var(--green) !important;
    border-radius: var(--radius) !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.18s cubic-bezier(0.4,0,0.2,1) !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::before {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(135deg, var(--green-dim), transparent) !important;
    opacity: 0 !important;
    transition: opacity 0.18s !important;
}
.stButton > button:hover {
    background: var(--green) !important;
    color: #000 !important;
    box-shadow: 0 0 24px var(--green-glow), 0 4px 16px rgba(0,0,0,0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:hover::before { opacity: 1 !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── METRICS ── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, var(--bg2), var(--bg3)) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--green) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
    transition: box-shadow 0.2s, transform 0.2s !important;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 0 20px var(--green-dim), 0 4px 20px rgba(0,0,0,0.3) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: var(--t3) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: var(--green) !important;
}

/* ── HEADINGS ── */
h2, h3 {
    font-family: var(--mono) !important;
    color: var(--t1) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0.5rem !important;
    margin-top: 1.4rem !important;
}
h4 {
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
    color: var(--t3) !important;
}

hr { border-color: var(--border) !important; }

p, .stMarkdown p {
    color: var(--t2) !important;
    font-size: 0.85rem !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    background: var(--bg2) !important;
    border-radius: var(--radius) !important;
}

/* ── SELECTBOX / SLIDER LABELS ── */
label, [data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stTextInput"] label {
    font-family: var(--mono) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
    color: var(--t3) !important;
}

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg2) !important;
    border-bottom: 1px solid var(--border) !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
    gap: 2px !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.09em !important;
    color: var(--t3) !important;
    border-radius: 0 !important;
    border: none !important;
    padding: 8px 18px !important;
    transition: color 0.15s !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: var(--t2) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--green) !important;
    border-bottom: 2px solid var(--green) !important;
    background: transparent !important;
    text-shadow: 0 0 12px var(--green-glow) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--t3); }

/* ── HIDE STREAMLIT CHROME ── */
footer { visibility: hidden !important; }

/* ── CUSTOM COMPONENTS ── */

/* Top header banner */
.app-header {
    background: linear-gradient(90deg, var(--bg2) 0%, var(--bg3) 100%);
    border-bottom: 1px solid var(--border);
    border-left: 3px solid var(--green);
    padding: 0.9rem 1.8rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 0 var(--radius) var(--radius) 0;
    box-shadow: 0 2px 20px rgba(0,0,0,0.4), inset 0 1px 0 rgba(0,229,176,0.04);
}
.app-header-left {}
.app-title {
    font-family: var(--mono);
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--t1);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.app-title span { color: var(--green); text-shadow: 0 0 16px var(--green-glow); }
.app-sub {
    font-family: var(--sans);
    font-size: 0.72rem;
    color: var(--t3);
    letter-spacing: 0.03em;
    margin-top: 3px;
}
.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: var(--green);
    border-radius: 50%;
    animation: pulse 2s infinite;
    margin-right: 6px;
    vertical-align: middle;
    box-shadow: 0 0 8px var(--green-glow);
}
@keyframes pulse {
    0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(0,229,176,0.5); }
    50%      { opacity:.8; box-shadow: 0 0 0 6px rgba(0,229,176,0); }
}
.live-label {
    font-family: var(--mono);
    font-size: 0.63rem;
    color: var(--green);
    letter-spacing: 0.13em;
    vertical-align: middle;
}

/* Sidebar stat row label */
.stat-row {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--t3);
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 4px;
    margin-top: 2px;
}

/* Model badge */
.model-badge {
    display: inline-block;
    background: var(--blue-dim);
    border: 1px solid rgba(77,166,255,0.25);
    color: var(--blue);
    font-family: var(--mono);
    font-size: 0.66rem;
    font-weight: 600;
    padding: 0.22rem 0.85rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    border-radius: var(--radius);
}

/* CI badge */
.ci-badge {
    display: inline-block;
    background: var(--blue-dim);
    border: 1px solid rgba(77,166,255,0.25);
    color: var(--blue);
    font-family: var(--mono);
    font-size: 0.66rem;
    font-weight: 600;
    padding: 0.22rem 0.85rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    border-radius: var(--radius);
}

/* Alert box */
.alert-box {
    background: rgba(0,229,176,0.04);
    border: 1px solid rgba(0,229,176,0.25);
    border-left: 3px solid var(--green);
    padding: 0.85rem 1.3rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--green);
    margin: 0.8rem 0;
    letter-spacing: 0.04em;
    border-radius: 0 var(--radius) var(--radius) 0;
}

/* Signal badges */
.signal-badge-buy {
    display: inline-block;
    background: rgba(0,229,176,0.07);
    border: 1px solid var(--green);
    color: var(--green);
    font-family: var(--mono);
    font-size: 1.2rem;
    font-weight: 700;
    padding: 0.5rem 2rem;
    letter-spacing: 0.12em;
    border-radius: var(--radius);
    box-shadow: 0 0 20px var(--green-dim);
}
.signal-badge-sell {
    display: inline-block;
    background: var(--red-dim);
    border: 1px solid var(--red);
    color: var(--red);
    font-family: var(--mono);
    font-size: 1.2rem;
    font-weight: 700;
    padding: 0.5rem 2rem;
    letter-spacing: 0.12em;
    border-radius: var(--radius);
    box-shadow: 0 0 20px var(--red-dim);
}
.signal-badge-hold {
    display: inline-block;
    background: var(--yellow-dim);
    border: 1px solid var(--yellow);
    color: var(--yellow);
    font-family: var(--mono);
    font-size: 1.2rem;
    font-weight: 700;
    padding: 0.5rem 2rem;
    letter-spacing: 0.12em;
    border-radius: var(--radius);
    box-shadow: 0 0 20px var(--yellow-dim);
}

/* Backtest KPI cards */
.bt-card {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    border-top: 2px solid var(--border2);
    padding: 1rem 1.2rem;
    margin-bottom: 0.4rem;
    font-family: var(--mono);
    border-radius: var(--radius);
    transition: transform 0.15s, box-shadow 0.15s;
}
.bt-card:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(0,0,0,0.3); }
.bt-card-label {
    font-size: 0.6rem;
    color: var(--t3);
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.bt-card-value       { font-size: 1.35rem; font-weight: 700; color: var(--t1); }
.bt-card-value-green { font-size: 1.35rem; font-weight: 700; color: var(--green); text-shadow: 0 0 10px var(--green-dim); }
.bt-card-value-red   { font-size: 1.35rem; font-weight: 700; color: var(--red); }
.bt-win              { color: var(--green) !important; font-weight: 700; }
.bt-loss             { color: var(--red) !important; font-weight: 700; }

/* Halal cards */
.halal-pass { color: var(--green); font-weight: 700; }
.halal-fail { color: var(--red); font-weight: 700; }
.halal-card {
    background: rgba(0,229,176,0.03);
    border: 1px solid rgba(0,229,176,0.12);
    border-left: 3px solid var(--green);
    padding: 0.85rem 1.2rem;
    margin: 0.35rem 0;
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--t2);
    border-radius: 0 var(--radius) var(--radius) 0;
}
.halal-card-fail {
    background: rgba(255,61,87,0.03);
    border: 1px solid rgba(255,61,87,0.12);
    border-left: 3px solid var(--red);
    padding: 0.85rem 1.2rem;
    margin: 0.35rem 0;
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--t2);
    border-radius: 0 var(--radius) var(--radius) 0;
}

/* Feature importance label */
.feature-importance-label {
    font-family: var(--mono);
    font-size: 0.63rem;
    color: var(--t3);
}

/* ── TICKER TAPE ── */
.ticker-tape-wrap {
    overflow: hidden;
    background: linear-gradient(90deg, var(--bg2), var(--bg3), var(--bg2));
    border-bottom: 1px solid var(--border);
    border-top: 1px solid var(--border);
    padding: 0.3rem 0;
    margin-bottom: 0;
}
.ticker-tape {
    display: inline-flex;
    gap: 2.8rem;
    animation: tape 38s linear infinite;
    white-space: nowrap;
    font-family: var(--mono);
    font-size: 0.67rem;
    letter-spacing: 0.07em;
    color: var(--t2);
}
.ticker-tape:hover { animation-play-state: paused; }
@keyframes tape { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
.tape-up   { color: var(--green); font-weight: 700; }
.tape-down { color: var(--red);   font-weight: 700; }
.tape-sym  { color: var(--t3); font-size: 0.6rem; margin-right: 0.3rem; }

/* ── SIGNAL PANEL v2 ── */
.signal-panel-v2 {
    display: flex;
    gap: 1rem;
    align-items: stretch;
    margin: 1.2rem 0;
    flex-wrap: wrap;
}
.signal-main-v2 {
    flex: 0 0 240px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.8rem 1.5rem;
    border: 2px solid var(--green);
    background: rgba(0,229,176,0.03);
    position: relative;
    overflow: hidden;
    border-radius: var(--radius);
    box-shadow: 0 0 30px rgba(0,229,176,0.06), inset 0 1px 0 rgba(0,229,176,0.08);
}
.signal-main-v2::before {
    content: '';
    position: absolute;
    bottom: -20px; right: -20px;
    width: 100px; height: 100px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,229,176,0.14) 0%, transparent 70%);
}
.signal-main-v2.sell { border-color: var(--red);    background: rgba(255,61,87,0.03); box-shadow: 0 0 30px rgba(255,61,87,0.06), inset 0 1px 0 rgba(255,61,87,0.06); }
.signal-main-v2.sell::before { background: radial-gradient(circle, rgba(255,61,87,0.14) 0%, transparent 70%); }
.signal-main-v2.hold { border-color: var(--yellow); background: rgba(255,221,45,0.03); box-shadow: 0 0 30px rgba(255,221,45,0.06), inset 0 1px 0 rgba(255,221,45,0.06); }
.signal-main-v2.hold::before { background: radial-gradient(circle, rgba(255,221,45,0.14) 0%, transparent 70%); }
.signal-action-v2 {
    font-family: var(--mono);
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: 0.2em;
    color: var(--green);
    line-height: 1;
    text-shadow: 0 0 20px var(--green-glow);
}
.signal-action-v2.sell { color: var(--red); text-shadow: 0 0 20px rgba(255,61,87,0.4); }
.signal-action-v2.hold { color: var(--yellow); text-shadow: 0 0 20px rgba(255,221,45,0.4); }
.signal-pct-v2 {
    font-family: var(--mono);
    font-size: 1.05rem;
    font-weight: 600;
    margin-top: 0.5rem;
    color: var(--t1);
}
.signal-label-v2 {
    font-family: var(--mono);
    font-size: 0.56rem;
    letter-spacing: 0.24em;
    color: var(--t3);
    margin-top: 8px;
    text-transform: uppercase;
}
.signal-details-v2 {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    min-width: 200px;
}
.sig-detail-card-v2 {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    padding: 0.75rem 1rem;
    position: relative;
    overflow: hidden;
    border-radius: var(--radius);
    transition: transform 0.15s, box-shadow 0.15s;
}
.sig-detail-card-v2:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
.sig-detail-card-v2::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--border2);
    border-radius: 2px 0 0 2px;
}
.sig-detail-card-v2.positive::after { background: var(--green); box-shadow: 0 0 8px var(--green-dim); }
.sig-detail-card-v2.negative::after { background: var(--red); }
.sig-detail-card-v2.neutral::after  { background: var(--yellow); }
.sig-detail-label-v2 {
    font-family: var(--mono);
    font-size: 0.55rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--t3);
    margin-bottom: 4px;
}
.sig-detail-val-v2 {
    font-family: var(--mono);
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--t1);
}
.sig-detail-sub-v2 {
    font-family: var(--mono);
    font-size: 0.56rem;
    color: var(--t3);
    margin-top: 2px;
}

/* ── COMPOSITE SIGNAL METER ── */
.composite-meter {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    border-left: 3px solid var(--blue);
    padding: 1.1rem 1.5rem;
    margin: 0.8rem 0;
    border-radius: 0 var(--radius) var(--radius) 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2);
}
.meter-title {
    font-family: var(--mono);
    font-size: 0.58rem;
    letter-spacing: 0.19em;
    text-transform: uppercase;
    color: var(--t3);
    margin-bottom: 0.8rem;
}
.signal-indicator-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.45rem;
    font-family: var(--mono);
    font-size: 0.68rem;
}
.sir-label { color: var(--t2); width: 110px; flex-shrink: 0; }
.sir-bar-bg {
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.04);
    border-radius: 2px;
    overflow: hidden;
}
.sir-bar { height: 100%; border-radius: 2px; transition: width 0.6s ease; }
.sir-bar.positive { background: linear-gradient(90deg, var(--green), rgba(0,229,176,0.6)); }
.sir-bar.negative { background: linear-gradient(90deg, var(--red), rgba(255,61,87,0.6)); }
.sir-bar.neutral  { background: linear-gradient(90deg, var(--yellow), rgba(255,221,45,0.6)); }
.sir-val { width: 55px; text-align: right; font-weight: 600; color: var(--t1); }
.sir-signal { width: 40px; text-align: right; font-size: 0.58rem; letter-spacing: 0.1em; font-weight: 700; }
.sir-signal.buy  { color: var(--green); }
.sir-signal.sell { color: var(--red); }
.sir-signal.hold { color: var(--yellow); }

/* ── GLASS SUMMARY CARDS ── */
.glass-summary-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.7rem;
    margin: 0.8rem 0;
}
.glass-summary-card {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    border-top: 2px solid var(--green);
    padding: 1rem 1.1rem;
    position: relative;
    overflow: hidden;
    border-radius: var(--radius);
    transition: transform 0.15s, box-shadow 0.15s;
}
.glass-summary-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.35); }
.glass-summary-card::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient(circle at top right, rgba(0,229,176,0.06), transparent 70%);
}
.glass-summary-card.red   { border-top-color: var(--red); }
.glass-summary-card.red::after { background: radial-gradient(circle at top right, rgba(255,61,87,0.06), transparent 70%); }
.glass-summary-card.blue  { border-top-color: var(--blue); }
.glass-summary-card.blue::after { background: radial-gradient(circle at top right, rgba(77,166,255,0.06), transparent 70%); }
.glass-summary-card.yellow{ border-top-color: var(--yellow); }
.glass-summary-card.yellow::after { background: radial-gradient(circle at top right, rgba(255,221,45,0.06), transparent 70%); }
.gsc-label {
    font-family: var(--mono);
    font-size: 0.57rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--t3);
    margin-bottom: 4px;
}
.gsc-value {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--green);
    line-height: 1.1;
    text-shadow: 0 0 10px var(--green-dim);
}
.gsc-value.red    { color: var(--red); text-shadow: none; }
.gsc-value.blue   { color: var(--blue); text-shadow: none; }
.gsc-value.yellow { color: var(--yellow); text-shadow: none; }
.gsc-value.white  { color: var(--t1); text-shadow: none; }
.gsc-sub {
    font-family: var(--mono);
    font-size: 0.58rem;
    color: var(--t3);
    margin-top: 3px;
}

</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-header-left">
        <div class="app-title">STOCK<span>CAST</span> &nbsp;·&nbsp; Market Intelligence</div>
        <div class="app-sub">XGBoost · Explainable Signals · Backtesting · Shariah Screening · News NLP</div>
    </div>
    <div style="display:flex;align-items:center;gap:1.2rem;">
        <div style="text-align:right;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.58rem;letter-spacing:.14em;color:#3d5068;text-transform:uppercase;">Platform</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:#7a8fa8;letter-spacing:.04em;">v2.0 Ultra</div>
        </div>
        <div style="width:1px;height:32px;background:#182030;"></div>
        <div>
            <span class="live-dot"></span>
            <span class="live-label">LIVE · NYSE/NASDAQ</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Animated Ticker Tape ───────────────────────────────────────────────────────
st.markdown("""
<div class="ticker-tape-wrap">
  <div class="ticker-tape">
    <span><span class="tape-sym">AAPL</span> <span class="tape-up">▲ $189.42 +1.2%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">TSLA</span> <span class="tape-down">▼ $248.11 -0.8%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">NVDA</span> <span class="tape-up">▲ $875.33 +2.1%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">MSFT</span> <span class="tape-up">▲ $421.05 +0.5%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">GOOGL</span> <span class="tape-down">▼ $168.22 -0.3%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">META</span> <span class="tape-up">▲ $512.88 +1.7%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">AMZN</span> <span class="tape-up">▲ $186.44 +0.9%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">AMD</span> <span class="tape-up">▲ $167.55 +3.2%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">JPM</span> <span class="tape-down">▼ $198.30 -0.4%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">SPY</span> <span class="tape-up">▲ $521.67 +0.6%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">QQQ</span> <span class="tape-up">▲ $448.90 +0.8%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">AAPL</span> <span class="tape-up">▲ $189.42 +1.2%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">TSLA</span> <span class="tape-down">▼ $248.11 -0.8%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">NVDA</span> <span class="tape-up">▲ $875.33 +2.1%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">MSFT</span> <span class="tape-up">▲ $421.05 +0.5%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">GOOGL</span> <span class="tape-down">▼ $168.22 -0.3%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">META</span> <span class="tape-up">▲ $512.88 +1.7%</span></span>
    <span style="color:#182030;">·</span>
    <span><span class="tape-sym">AMZN</span> <span class="tape-up">▲ $186.44 +0.9%</span></span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Ticker Search Helper ───────────────────────────────────────────────────────
POPULAR_TICKERS = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.", "NVDA": "NVIDIA Corp.", "META": "Meta Platforms",
    "TSLA": "Tesla Inc.", "NFLX": "Netflix Inc.", "AMD": "Advanced Micro Devices",
    "ORCL": "Oracle Corp.", "INTC": "Intel Corp.", "CRM": "Salesforce Inc.",
    "ADBE": "Adobe Inc.", "PYPL": "PayPal Holdings", "UBER": "Uber Technologies",
    "BABA": "Alibaba Group", "JPM": "JPMorgan Chase", "BAC": "Bank of America",
    "GS": "Goldman Sachs", "V": "Visa Inc.", "MA": "Mastercard Inc.",
    "JNJ": "Johnson & Johnson", "PFE": "Pfizer Inc.", "MRNA": "Moderna Inc.",
    "DIS": "Walt Disney Co.", "SPOT": "Spotify Technology", "SNAP": "Snap Inc.",
    "TWTR": "Twitter / X", "SHOP": "Shopify Inc.", "SQ": "Block Inc.",
    "COIN": "Coinbase Global", "HOOD": "Robinhood Markets", "PLTR": "Palantir Technologies",
    "RBLX": "Roblox Corp.", "ABNB": "Airbnb Inc.", "LYFT": "Lyft Inc.",
    "ZM": "Zoom Video", "DOCU": "DocuSign Inc.", "ROKU": "Roku Inc.",
    "ARKK": "ARK Innovation ETF", "SPY": "S&P 500 ETF", "QQQ": "Nasdaq-100 ETF",
    "2222.SR": "Saudi Aramco", "9988.HK": "Alibaba HK", "7203.T": "Toyota Motor",
    "005930.KS": "Samsung Electronics", "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy", "INFY.NS": "Infosys Ltd.",
    "XOM": "ExxonMobil Corp.", "CVX": "Chevron Corp.", "BP": "BP plc",
    "NKE": "Nike Inc.", "MCD": "McDonald's Corp.", "SBUX": "Starbucks Corp.",
    "WMT": "Walmart Inc.", "COST": "Costco Wholesale", "TGT": "Target Corp.",
    "BA": "Boeing Co.", "LMT": "Lockheed Martin", "GE": "GE Aerospace",
    "GOOG": "Alphabet Class C", "BRK-B": "Berkshire Hathaway B",
}

@st.cache_data(ttl=3600)
def search_tickers(query):
    q = query.strip().upper()
    results = []
    # Exact symbol match first
    if q in POPULAR_TICKERS:
        results.append(f"{q} — {POPULAR_TICKERS[q]}")
    # Name contains match
    ql = query.strip().lower()
    for sym, name in POPULAR_TICKERS.items():
        if sym != q and (ql in name.lower() or ql in sym.lower()):
            results.append(f"{sym} — {name}")
    # Also try yf.Search if available
    try:
        res = yf.Search(query, max_results=6)
        for r in res.quotes:
            sym  = r.get("symbol", "")
            name = r.get("longname") or r.get("shortname") or sym
            exch = r.get("exchange", "")
            qt   = r.get("quoteType", "")
            entry = f"{sym} — {name} ({exch})"
            if sym and qt in ("EQUITY", "ETF", "INDEX") and entry not in results:
                results.append(entry)
    except Exception:
        pass
    return results[:10]

@st.cache_data(ttl=300)
def validate_ticker(sym):
    try:
        info = yf.Ticker(sym).fast_info
        return float(info.last_price) > 0
    except Exception:
        return False

# ── Auth Gate: Login / Signup ──────────────────────────────────────────────────
if st.session_state.user is None:

    # ── Full-page auth styles ──────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

    /* Full page override for auth screen */
    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], .main {
        background:
            radial-gradient(ellipse at 15% 25%, rgba(0,229,176,0.10), transparent 45%),
            radial-gradient(ellipse at 85% 75%, rgba(77,166,255,0.09), transparent 45%),
            radial-gradient(ellipse at 55% 5%,  rgba(167,139,250,0.06), transparent 40%),
            #080b0f !important;
        font-family: 'Inter', sans-serif !important;
    }
    .block-container { padding: 0 !important; max-width: 100% !important; }

    /* Hide streamlit chrome */
    header[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }

    /* Animated background blobs */
    .auth-bg {
        position: fixed; inset: 0; z-index: 0;
        background: #080b0f;
        overflow: hidden;
    }
    .auth-bg::before {
        content: '';
        position: absolute;
        top: -15%; left: -8%;
        width: 700px; height: 700px;
        background: radial-gradient(circle, rgba(0,229,176,0.07) 0%, transparent 60%);
        animation: float1 14s ease-in-out infinite;
    }
    .auth-bg::after {
        content: '';
        position: absolute;
        bottom: -8%; right: -5%;
        width: 600px; height: 600px;
        background: radial-gradient(circle, rgba(77,166,255,0.06) 0%, transparent 60%);
        animation: float2 18s ease-in-out infinite;
    }
    @keyframes float1 {
        0%,100% { transform: translate(0,0) scale(1); }
        33%      { transform: translate(50px, 35px) scale(1.08); }
        66%      { transform: translate(20px, -20px) scale(0.96); }
    }
    @keyframes float2 {
        0%,100% { transform: translate(0,0) scale(1); }
        33%      { transform: translate(-35px, -25px) scale(1.06); }
        66%      { transform: translate(15px, 10px) scale(0.97); }
    }

    /* Card */
    .auth-card {
        position: relative; z-index: 10;
        background: rgba(13, 17, 23, 0.80);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        padding: 2.5rem 2.2rem 2rem;
        max-width: 420px;
        margin: 0 auto;
        box-shadow:
            0 20px 60px rgba(0,0,0,0.7),
            inset 0 1px 0 rgba(255,255,255,0.06),
            0 0 0 1px rgba(255,255,255,0.02),
            0 0 80px rgba(0,229,176,0.04);
        animation: cardIn 0.55s cubic-bezier(0.16,1,0.3,1) both;
    }
    @keyframes cardIn {
        from { opacity: 0; transform: translateY(28px) scale(0.96); }
        to   { opacity: 1; transform: translateY(0)    scale(1); }
    }

    /* Logo */
    .auth-logo {
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .auth-logo-icon {
        display: inline-flex; align-items: center; justify-content: center;
        width: 54px; height: 54px;
        background: linear-gradient(135deg, rgba(0,229,176,0.18), rgba(0,229,176,0.04));
        border: 1px solid rgba(0,229,176,0.28);
        border-radius: 14px;
        font-size: 1.5rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 0 28px rgba(0,229,176,0.15), inset 0 1px 0 rgba(0,229,176,0.1);
    }
    .auth-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.75rem;
        font-weight: 700;
        color: #eef2f7;
        letter-spacing: 0.04em;
        line-height: 1;
    }
    .auth-title span { color: #00e5b0; text-shadow: 0 0 20px rgba(0,229,176,0.4); }
    .auth-subtitle {
        font-size: 0.84rem;
        color: #3d5068;
        margin-top: 0.45rem;
        font-weight: 400;
        letter-spacing: 0.02em;
    }

    /* Tab bar */
    .auth-tabs {
        display: flex;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 4px;
        margin: 1.6rem 0 1.4rem;
        gap: 4px;
    }
    .auth-tab {
        flex: 1; text-align: center;
        padding: 0.5rem;
        border-radius: 7px;
        font-size: 0.82rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        color: #64748b;
        border: none; background: transparent;
    }
    .auth-tab.active {
        background: rgba(0,229,176,0.12);
        color: #00e5b0;
        border: 1px solid rgba(0,229,176,0.25);
        box-shadow: 0 2px 8px rgba(0,229,176,0.08);
    }

    /* Input group */
    .auth-input-group {
        margin-bottom: 1rem;
    }
    .auth-label {
        display: block;
        font-size: 0.78rem;
        font-weight: 500;
        color: #94a3b8;
        margin-bottom: 0.4rem;
        letter-spacing: 0.03em;
    }
    .auth-input-wrap {
        position: relative;
    }
    .auth-input-icon {
        position: absolute; left: 14px; top: 50%;
        transform: translateY(-50%);
        color: #475569; font-size: 0.9rem;
        pointer-events: none; z-index: 1;
    }

    /* Override Streamlit inputs — Upgrade 1 */
    [data-testid="stTextInput"] input {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid #1f2937 !important;
        border-radius: 10px !important;
        color: #e5e7eb !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
        padding: 12px 14px 12px 36px !important;
        transition: all 0.25s ease !important;
        outline: none !important;
    }
    [data-testid="stTextInput"] input:focus {
        border: 1px solid #00e5b0 !important;
        box-shadow: 0 0 0 2px rgba(0,229,176,0.2) !important;
        background: rgba(15, 23, 42, 0.9) !important;
    }
    [data-testid="stTextInput"] input::placeholder { color: #64748b !important; }
    [data-testid="stTextInput"] label { display: none !important; }

    /* Upgrade 3: Smooth fade-up animation for each input */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stTextInput"] {
        animation: fadeUp 0.4s ease forwards;
    }
    [data-testid="stTextInput"]:nth-child(1) { animation-delay: 0.05s; }
    [data-testid="stTextInput"]:nth-child(2) { animation-delay: 0.12s; }
    [data-testid="stTextInput"]:nth-child(3) { animation-delay: 0.19s; }

    /* Upgrade 2: Premium button */
    div.stButton > button {
        background: linear-gradient(135deg, #00e5b0, #00b88d) !important;
        color: #050d0a !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        height: 48px !important;
        border: none !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        width: 100% !important;
        margin-top: 0.4rem !important;
        transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
        box-shadow: 0 4px 24px rgba(0,229,176,0.3), inset 0 1px 0 rgba(255,255,255,0.15) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    div.stButton > button::after {
        content: '';
        position: absolute; inset: 0;
        background: linear-gradient(180deg, rgba(255,255,255,0.12), transparent);
        pointer-events: none;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(0,229,176,0.4) !important;
        background: linear-gradient(135deg, #1aebbb, #00e5b0) !important;
    }
    div.stButton > button:active {
        transform: scale(0.98) !important;
        box-shadow: 0 2px 12px rgba(0,229,176,0.25) !important;
    }

    /* Divider */
    .auth-divider {
        display: flex; align-items: center; gap: 0.8rem;
        margin: 1.2rem 0;
        color: #334155; font-size: 0.75rem;
    }
    .auth-divider::before, .auth-divider::after {
        content: ''; flex: 1;
        height: 1px; background: rgba(255,255,255,0.06);
    }

    /* Forgot password */
    .auth-forgot {
        text-align: right;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }
    .auth-forgot a {
        font-size: 0.75rem;
        color: #00e5b0;
        text-decoration: none;
        opacity: 0.8;
        transition: opacity 0.15s;
    }
    .auth-forgot a:hover { opacity: 1; }

    /* Trust badges */
    .auth-trust {
        display: flex; justify-content: center; gap: 1.2rem;
        margin-top: 1.4rem;
        flex-wrap: wrap;
    }
    .auth-trust-item {
        font-size: 0.7rem; color: #334155;
        display: flex; align-items: center; gap: 0.3rem;
    }
    .auth-trust-item span { color: #00e5b0; }

    /* Streamlit alert overrides */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        font-size: 0.82rem !important;
        font-family: 'Inter', sans-serif !important;
    }
    </style>

    <!-- Animated background -->
    <div class="auth-bg"></div>
    """, unsafe_allow_html=True)

    # ── Centered layout ────────────────────────────────────────────────────────
    # Extra CSS to style the middle column as the card
    st.markdown("""
    <style>
    /* Style the auth middle column as a glassy card */
    [data-testid="stApp"] [data-testid="stHorizontalBlock"] > div:nth-child(2) > div {
        background: rgba(13,17,23,0.85) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-top: 1px solid rgba(0,229,176,0.18) !important;
        border-radius: 20px !important;
        padding: 2rem 1.6rem 1.8rem !important;
        box-shadow:
            0 24px 64px rgba(0,0,0,0.8),
            inset 0 1px 0 rgba(0,229,176,0.07),
            0 0 60px rgba(0,229,176,0.05) !important;
        margin-top: 3rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _l, _mid, _r = st.columns([1, 1.4, 1])
    with _mid:

        # ── Logo + Title ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding:0.5rem 0 1.2rem;animation:cardIn 0.4s ease both;">
            <div style="display:inline-flex;align-items:center;justify-content:center;
                 width:58px;height:58px;
                 background:linear-gradient(135deg,rgba(0,229,176,0.20),rgba(0,229,176,0.05));
                 border:1px solid rgba(0,229,176,0.35);border-radius:16px;
                 font-size:1.6rem;margin-bottom:1rem;
                 box-shadow:0 0 32px rgba(0,229,176,0.20),inset 0 1px 0 rgba(0,229,176,0.15);">📈</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.9rem;font-weight:700;
                 color:#eef2f7;letter-spacing:.05em;line-height:1;">
                STOCK<span style="color:#00e5b0;text-shadow:0 0 24px rgba(0,229,176,0.5);">CAST</span>
            </div>
            <div style="font-size:.82rem;color:#3d5068;margin-top:.5rem;font-family:'Inter',sans-serif;letter-spacing:.03em;">
                AI-powered stock intelligence platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        auth_tab1, auth_tab2 = st.tabs(["🔑  Login", "✨  Sign Up"])

        # ── LOGIN TAB ──────────────────────────────────────────────────────────
        with auth_tab1:
            st.markdown('<div style="position:relative;">'
                        '<span style="position:absolute;left:12px;top:50%;transform:translateY(-8px);'
                        'color:#475569;font-size:.9rem;z-index:999;pointer-events:none;">✉</span>'
                        '</div>', unsafe_allow_html=True)
            login_email = st.text_input("Email", key="login_email",
                                        placeholder="  you@example.com",
                                        label_visibility="collapsed")

            st.markdown('<div style="position:relative;">'
                        '<span style="position:absolute;left:12px;top:50%;transform:translateY(-8px);'
                        'color:#475569;font-size:.9rem;z-index:999;pointer-events:none;">🔒</span>'
                        '</div>', unsafe_allow_html=True)
            login_password = st.text_input("Password", key="login_password",
                                           type="password",
                                           placeholder="  ••••••••",
                                           label_visibility="collapsed")

            st.markdown('<div style="text-align:right;margin-top:-.4rem;margin-bottom:.8rem;">'
                        '<a href="#" style="font-size:.75rem;color:#00e5b0;text-decoration:none;opacity:.8;">'
                        'Forgot password?</a></div>', unsafe_allow_html=True)

            if st.button("Login →", use_container_width=True, key="login_btn"):
                if login_email and login_password:
                    with st.spinner("🔐 Securing session..."):
                        try:
                            res = supabase.auth.sign_in_with_password({
                                "email":    login_email,
                                "password": login_password,
                            })
                            if res.user:
                                st.session_state.user = res.user
                                st.success("✓ Welcome back! Loading your dashboard...")
                                st.rerun()
                            else:
                                st.error("⚠ Invalid credentials. Please try again.")
                        except Exception as e:
                            st.error(f"⚠ Login failed: {e}")
                else:
                    st.warning("Please enter your email and password.")

            st.markdown("""
            <div style="display:flex;justify-content:center;gap:1.4rem;margin-top:1.2rem;flex-wrap:wrap;">
              <span style="font-size:.7rem;color:#334155;display:flex;align-items:center;gap:.3rem;">
                <span style="color:#00e5b0;">✓</span> 256-bit encryption</span>
              <span style="font-size:.7rem;color:#334155;display:flex;align-items:center;gap:.3rem;">
                <span style="color:#00e5b0;">✓</span> No data stored</span>
              <span style="font-size:.7rem;color:#334155;display:flex;align-items:center;gap:.3rem;">
                <span style="color:#00e5b0;">✓</span> Free to use</span>
            </div>
            """, unsafe_allow_html=True)

        # ── SIGNUP TAB ─────────────────────────────────────────────────────────
        with auth_tab2:
            signup_email = st.text_input("Email", key="signup_email",
                                         placeholder="  you@example.com",
                                         label_visibility="collapsed")
            signup_password = st.text_input("Password", key="signup_password",
                                            type="password",
                                            placeholder="  Min. 6 characters",
                                            label_visibility="collapsed")
            signup_confirm = st.text_input("Confirm Password", key="signup_confirm",
                                           type="password",
                                           placeholder="  Re-enter password",
                                           label_visibility="collapsed")

            if st.button("Create Account →", use_container_width=True, key="signup_btn"):
                if signup_email and signup_password and signup_confirm:
                    if signup_password != signup_confirm:
                        st.error("⚠ Passwords do not match.")
                    elif len(signup_password) < 6:
                        st.error("⚠ Password must be at least 6 characters.")
                    else:
                        with st.spinner("✨ Setting up your account..."):
                            try:
                                res = supabase.auth.sign_up({
                                    "email":    signup_email,
                                    "password": signup_password,
                                })
                                if res.user:
                                    st.success("✓ Account created! Check your email to confirm, then log in.")
                                else:
                                    st.error("⚠ Signup failed. Please try again.")
                            except Exception as e:
                                st.error(f"⚠ Signup failed: {e}")
                else:
                    st.warning("Please fill in all fields.")

            st.markdown("""
            <div style="text-align:center;margin-top:1rem;font-size:.75rem;color:#334155;font-family:'Inter',sans-serif;">
              By signing up you agree to our
              <a href="#" style="color:#00e5b0;text-decoration:none;">Terms</a> &
              <a href="#" style="color:#00e5b0;text-decoration:none;">Privacy Policy</a>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center;margin-top:1.6rem;font-family:'IBM Plex Mono',monospace;
             font-size:.6rem;color:#182030;letter-spacing:.08em;">
            ⚠ FOR EDUCATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE
        </div>
        """, unsafe_allow_html=True)

    st.stop()  # 🚨 Halt — do not render the app until authenticated

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── User info + Logout ─────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:rgba(0,229,176,0.06);border:1px solid rgba(0,229,176,0.2);'
        f'border-left:3px solid #00e5b0;padding:.6rem 1rem;font-family:IBM Plex Mono,monospace;'
        f'font-size:.68rem;color:#00e5b0;letter-spacing:.06em;margin-bottom:.6rem;">'
        f'👤 {st.session_state.user.email}</div>',
        unsafe_allow_html=True
    )
    if st.button("⏏  Logout", use_container_width=True, key="logout_btn"):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.rerun()
    st.markdown("---")

    # ── Ticker Search ──────────────────────────────────────────────────────────
    st.markdown('<div class="stat-row">🔍 Search Company / Ticker</div>', unsafe_allow_html=True)
    search_query = st.text_input("Search", placeholder="e.g. Apple, TSLA, Saudi Aramco…",
                                 label_visibility="collapsed", key="search_input")

    ticker = "AAPL"   # safe default

    if search_query and len(search_query.strip()) >= 1:
        search_results = search_tickers(search_query.strip())
        if search_results:
            selected = st.selectbox("Select", search_results, label_visibility="collapsed")
            ticker = selected.split(" — ")[0].strip()
            st.markdown(
                f'<div style="background:rgba(0,229,176,0.07);border:1px solid rgba(0,229,176,0.3);'
                f'border-left:3px solid #00e5b0;padding:.4rem 1rem;font-family:IBM Plex Mono,monospace;'
                f'font-size:.72rem;color:#00e5b0;letter-spacing:.06em;margin:.3rem 0;">'
                f'✓ {ticker}</div>',
                unsafe_allow_html=True)
        else:
            # Treat the query itself as a direct ticker entry
            ticker = search_query.strip().upper()
            st.markdown(
                f'<div style="background:rgba(255,221,45,0.07);border:1px solid rgba(255,221,45,0.3);'
                f'border-left:3px solid #ffdd2d;padding:.4rem 1rem;font-family:IBM Plex Mono,monospace;'
                f'font-size:.72rem;color:#ffdd2d;letter-spacing:.06em;margin:.3rem 0;">'
                f'Using: {ticker} — verify symbol is correct</div>',
                unsafe_allow_html=True)
    else:
        st.markdown('<div class="stat-row">Or type ticker directly below</div>', unsafe_allow_html=True)
        ticker = st.text_input("Ticker Symbol", value="AAPL",
                               placeholder="AAPL, TSLA, MSFT…",
                               label_visibility="collapsed", key="direct_ticker").strip().upper() or "AAPL"
        st.markdown(
            f'<div style="background:rgba(0,229,176,0.07);border:1px solid rgba(0,229,176,0.3);'
            f'border-left:3px solid #00e5b0;padding:.4rem 1rem;font-family:IBM Plex Mono,monospace;'
            f'font-size:.75rem;color:#00e5b0;letter-spacing:.08em;margin:.3rem 0;">'
            f'● ACTIVE: {ticker}</div>',
            unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=pd.to_datetime("2018-01-01"))
    with col2:
        end_date = st.date_input("To", value=pd.Timestamp.today())

    st.markdown('<div class="stat-row">Lookback Window (days)</div>', unsafe_allow_html=True)
    seq_len = st.slider("", 10, 60, 30, label_visibility="collapsed")

    st.markdown('<div class="stat-row">Forecast Horizon (days)</div>', unsafe_allow_html=True)
    future_days = st.slider(" ", 1, 30, 7, label_visibility="collapsed")

    st.markdown("---")
    # ── Mode Switch ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:rgba(0,229,176,0.04);border:1px solid rgba(0,229,176,0.2);
         padding:.65rem 1rem;font-family:'IBM Plex Mono',monospace;font-size:.6rem;
         letter-spacing:.14em;text-transform:uppercase;color:#3d5068;margin-bottom:.3rem;">
      Experience Level
    </div>""", unsafe_allow_html=True)

    ui_mode = st.radio(
        "Mode",
        ["🟢 Beginner", "🔴 Pro"],
        index=1,
        horizontal=True,
        label_visibility="collapsed",
        help="Beginner: simple clean view. Pro: full control + advanced features."
    )
    is_beginner = (ui_mode == "🟢 Beginner")

    if is_beginner:
        st.markdown("""<div style="background:rgba(0,229,176,0.06);border-left:3px solid #00e5b0;
             padding:.45rem .9rem;font-family:'IBM Plex Mono',monospace;font-size:.63rem;
             color:#00e5b0;letter-spacing:.05em;margin-bottom:.2rem;">
        ✓ Simple view active — just pick a ticker and run</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background:rgba(255,61,87,0.06);border-left:3px solid #ff3d57;
             padding:.45rem .9rem;font-family:'IBM Plex Mono',monospace;font-size:.63rem;
             color:#ff3d57;letter-spacing:.05em;margin-bottom:.2rem;">
        ⚡ Pro view — all parameters unlocked</div>""", unsafe_allow_html=True)

    st.markdown("---")
    # ── Fast Mode (always visible) ─────────────────────────────────────────────
    fast_mode = st.checkbox("⚡ Fast Mode (skip CI + backtest)", value=is_beginner,
        help="Disables bootstrap CI and backtesting for much faster results")

    # ── Pro-only: XGBoost tuning ───────────────────────────────────────────────
    if not is_beginner:
        st.markdown('<div class="stat-row">XGBoost Settings</div>', unsafe_allow_html=True)
        n_estimators  = st.slider("Trees", 100, 500, 200, step=50)
        max_depth     = st.slider("Max Depth", 2, 8, 4)
        learning_rate = st.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2], value=0.05)
    else:
        n_estimators  = 200
        max_depth     = 4
        learning_rate = 0.05

    st.markdown("---")
    st.markdown('<div class="stat-row">Price Alert Target ($)</div>', unsafe_allow_html=True)
    alert_price = st.number_input("", min_value=0.0, value=0.0, step=1.0, label_visibility="collapsed")

    # ── Pro-only: Backtesting ──────────────────────────────────────────────────
    if not is_beginner:
        st.markdown("---")
        st.markdown('<div class="stat-row">Backtesting</div>', unsafe_allow_html=True)
        run_backtest        = st.checkbox("Enable Backtesting Engine", value=True)
        bt_initial_capital  = st.number_input("Initial Capital ($)", min_value=1000, value=10000, step=1000)
        bt_commission       = st.number_input("Commission per Trade ($)", min_value=0.0, value=1.0, step=0.5)
        bt_signal_threshold = st.slider("Signal Threshold (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5,
            help="Min predicted % move to trigger BUY/SELL")
    else:
        run_backtest        = False
        bt_initial_capital  = 10000
        bt_commission       = 1.0
        bt_signal_threshold = 1.0

    # ── Pro-only: Extra features ───────────────────────────────────────────────
    if not is_beginner:
        st.markdown("---")
        st.markdown('<div class="stat-row">Extra Features</div>', unsafe_allow_html=True)
        run_model_compare  = st.checkbox("Model Comparison (XGB vs Prophet vs LR)", value=False)
        run_halal_check    = st.checkbox("Halal / Shariah Compliance Check", value=True)
        show_conf_interval = st.checkbox("Confidence Intervals on Forecast", value=True) and not fast_mode
        ci_bootstrap_n     = st.slider("Bootstrap Samples (CI)", 50, 300, 100, step=50) if show_conf_interval else 100
    else:
        run_model_compare  = False
        run_halal_check    = True
        show_conf_interval = False
        ci_bootstrap_n     = 100

    # ── Pro only: Multi-Stock Comparison ─────────────────────────────────────
    if not is_beginner:
        st.markdown("---")
        st.markdown('<div class="stat-row">🔥 Multi-Stock Comparison</div>', unsafe_allow_html=True)
        compare_tickers_raw = st.text_input(
            "Compare Tickers (comma-separated)",
            value="",
            placeholder="e.g. AAPL,TSLA,NVDA",
            label_visibility="collapsed",
            key="compare_input",
            help="Leave blank to skip. Fetches live data and ranks by composite signal score."
        )
        compare_tickers = [t.strip().upper() for t in compare_tickers_raw.split(",")
                           if t.strip()] if compare_tickers_raw.strip() else []
    else:
        compare_tickers = []

    st.markdown("---")
    run_btn = st.button("▶  RUN FORECAST", use_container_width=True)

    # ── Watchlist ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⭐ WATCHLIST")

    wl_col1, wl_col2 = st.columns([3, 1])
    with wl_col1:
        add_ticker_input = st.text_input(
            "Add ticker", placeholder="e.g. AAPL",
            label_visibility="collapsed", key="wl_add_input"
        ).strip().upper()
    with wl_col2:
        add_clicked = st.button("＋ Add", use_container_width=True, key="wl_add_btn")

    if add_clicked and add_ticker_input:
        if add_ticker_input not in st.session_state.watchlist:
            st.session_state.watchlist.append(add_ticker_input)

    if st.session_state.watchlist:
        for wl_sym in list(st.session_state.watchlist):
            wc1, wc2 = st.columns([3, 1])
            with wc1:
                # Quick live price badge
                try:
                    _qt = yf.Ticker(wl_sym).fast_info
                    _px = _qt.get("last_price") or _qt.get("regularMarketPrice") or 0
                    _chg = _qt.get("regularMarketChangePercent") or 0
                    _color = "#00e5b0" if _chg >= 0 else "#ff3d57"
                    _sign = "▲" if _chg >= 0 else "▼"
                    st.markdown(
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:.7rem;'
                        f'color:#eef2f7;padding:.25rem 0;">'
                        f'<span style="color:#3d5068;">{wl_sym}</span>  '
                        f'<span style="color:{_color};">{_sign} ${_px:.2f} ({_chg:+.1f}%)</span></div>',
                        unsafe_allow_html=True
                    )
                except Exception:
                    st.markdown(
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:.7rem;'
                        f'color:#3d5068;padding:.25rem 0;">{wl_sym}</div>',
                        unsafe_allow_html=True
                    )
            with wc2:
                if st.button("✕", key=f"wl_del_{wl_sym}", use_container_width=True):
                    st.session_state.watchlist.remove(wl_sym)
                    if wl_sym in st.session_state.alert_signals:
                        del st.session_state.alert_signals[wl_sym]
                    st.rerun()
    else:
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#243346;'
            'padding:.4rem 0;">No stocks saved yet.</div>',
            unsafe_allow_html=True
        )

    # ── Signal Alert Toggle ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔔 SIGNAL ALERTS")
    alert_on_signal_change = st.checkbox(
        "Alert when signal changes (BUY/SELL/HOLD)",
        value=True,
        help="Shows a banner at the top of the analysis if the signal has changed since last run"
    )
    st.caption("Run the forecast to detect signal changes.")

# ── Plotly theme ───────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono", color="#7a8fa8", size=10),
    xaxis=dict(gridcolor="#182030", linecolor="#182030", tickfont=dict(color="#3d5068", size=10), showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#182030", linecolor="#182030", tickfont=dict(color="#3d5068", size=10), showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#182030", borderwidth=1, font=dict(size=10)),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#111820", bordercolor="#243346", font=dict(family="IBM Plex Mono", size=11, color="#eef2f7")),
)

# Colour constants
C_GREEN  = "#00e5b0"
C_RED    = "#ff3d57"
C_BLUE   = "#4da6ff"
C_YELLOW = "#ffdd2d"
C_GREY   = "#3d5068"

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=str(start), end=str(end), progress=False, auto_adjust=True)
    # Flatten MultiIndex columns produced by newer yfinance versions (e.g. ('Close', 'AAPL') → 'Close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Strip timezone from index (yfinance may return tz-aware index which breaks Prophet & Plotly)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast, adjust=False).mean()
    ema_slow    = series.ewm(span=slow, adjust=False).mean()
    macd        = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram   = macd - signal_line
    return macd, signal_line, histogram

def compute_bollinger_bands(series, period=20, std=2):
    sma         = series.rolling(period).mean()
    rolling_std = series.rolling(period).std()
    upper       = sma + (std * rolling_std)
    lower       = sma - (std * rolling_std)
    return upper, sma, lower

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
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df

def build_xgb_dataset(df, seq_len):
    close = df['Close'].squeeze().values
    feature_cols = [
        'MA5','MA10','MA20','MA50','EMA12','EMA26',
        'RSI','MACD','MACD_Signal','MACD_Hist',
        'BB_Width','BB_Pct',
        'Returns','Returns_5d','Volatility','Momentum',
        'Volume_Ratio','High_Low_Pct','Close_Open_Pct','ATR'
    ]
    feat_df         = df[feature_cols].copy()
    feat_df['Close'] = close
    X_rows, y_rows  = [], []
    for i in range(seq_len, len(feat_df) - 1):
        row_feats  = feat_df[feature_cols].iloc[i].values
        lag_closes = close[i - seq_len:i]
        X_rows.append(np.concatenate([row_feats, lag_closes]))
        y_rows.append(close[i + 1])
    X    = np.array(X_rows)
    y    = np.array(y_rows)
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return X[mask], y[mask]

def generate_forward_signal(last_close, forecast_price):
    diff_pct = (forecast_price - last_close) / last_close * 100
    if diff_pct > 1.0:
        return "BUY", diff_pct
    elif diff_pct < -1.0:
        return "SELL", diff_pct
    else:
        return "HOLD", diff_pct


def compute_composite_signal(df, last_close, forecast_price, preds, actual):
    """
    Multi-factor composite Buy/Sell/Hold signal engine.
    Combines: XGBoost forecast, RSI, MACD crossover, Bollinger Band position,
    MA crossover (Golden/Death Cross), Volume confirmation, Momentum.
    Returns a dict with per-indicator signals and a weighted composite score.
    Score > 0 = bullish, < 0 = bearish, range approx -100 to +100.
    """
    close = df['Close'].squeeze()
    rsi   = float(df['RSI'].squeeze().iloc[-1])
    macd  = float(df['MACD'].squeeze().iloc[-1])
    macd_s= float(df['MACD_Signal'].squeeze().iloc[-1])
    macd_h= float(df['MACD_Hist'].squeeze().iloc[-1])
    bb_pct= float(df['BB_Pct'].squeeze().iloc[-1])      # 0=lower band, 1=upper band
    ma50  = float(df['MA50'].squeeze().iloc[-1])
    ma200 = float(df['MA200'].squeeze().iloc[-1])
    vol_r = float(df['Volume_Ratio'].squeeze().iloc[-1])
    mom   = float(df['Momentum'].squeeze().iloc[-1])
    atr   = float(df['ATR'].squeeze().iloc[-1])

    signals = {}

    # 1. XGBoost forecast direction (weight: 35)
    xgb_pct = (forecast_price - last_close) / last_close * 100
    if xgb_pct > 1.5:
        signals['XGBoost Forecast'] = ('BUY',  min(35, abs(xgb_pct) * 6),  xgb_pct, 'positive')
    elif xgb_pct < -1.5:
        signals['XGBoost Forecast'] = ('SELL', -min(35, abs(xgb_pct) * 6), xgb_pct, 'negative')
    else:
        signals['XGBoost Forecast'] = ('HOLD', 0, xgb_pct, 'neutral')

    # 2. RSI (weight: 20)
    if rsi < 30:
        signals['RSI (14)'] = ('BUY',  20, rsi, 'positive')
    elif rsi > 70:
        signals['RSI (14)'] = ('SELL', -20, rsi, 'negative')
    elif rsi < 45:
        signals['RSI (14)'] = ('BUY',  8, rsi, 'positive')
    elif rsi > 55:
        signals['RSI (14)'] = ('SELL', -8, rsi, 'negative')
    else:
        signals['RSI (14)'] = ('HOLD', 0, rsi, 'neutral')

    # 3. MACD crossover (weight: 20)
    prev_hist = float(df['MACD_Hist'].squeeze().iloc[-2]) if len(df) > 2 else 0
    if macd_h > 0 and prev_hist <= 0:
        signals['MACD Cross'] = ('BUY',  20, macd_h, 'positive')   # fresh bullish cross
    elif macd_h < 0 and prev_hist >= 0:
        signals['MACD Cross'] = ('SELL', -20, macd_h, 'negative')  # fresh bearish cross
    elif macd > macd_s:
        signals['MACD Cross'] = ('BUY',  10, macd_h, 'positive')
    elif macd < macd_s:
        signals['MACD Cross'] = ('SELL', -10, macd_h, 'negative')
    else:
        signals['MACD Cross'] = ('HOLD', 0, macd_h, 'neutral')

    # 4. Bollinger Band position (weight: 10)
    if bb_pct < 0.1:
        signals['Bollinger %B'] = ('BUY',  10, bb_pct, 'positive')
    elif bb_pct > 0.9:
        signals['Bollinger %B'] = ('SELL', -10, bb_pct, 'negative')
    else:
        signals['Bollinger %B'] = ('HOLD', 0, bb_pct, 'neutral')

    # 5. MA50 vs MA200 Golden/Death Cross (weight: 10)
    if not (ma50 == 0 or ma200 == 0):
        if ma50 > ma200:
            signals['MA Cross'] = ('BUY',  8, (ma50 - ma200) / ma200 * 100, 'positive')
        else:
            signals['MA Cross'] = ('SELL', -8, (ma50 - ma200) / ma200 * 100, 'negative')
    else:
        signals['MA Cross'] = ('HOLD', 0, 0, 'neutral')

    # 6. Volume confirmation (weight: 5)
    if vol_r > 1.5:
        # High volume confirms direction from XGBoost
        direction = signals['XGBoost Forecast'][0]
        score     = 5 if direction == 'BUY' else -5 if direction == 'SELL' else 0
        signals['Volume'] = (direction, score, vol_r, 'positive' if score > 0 else 'negative' if score < 0 else 'neutral')
    else:
        signals['Volume'] = ('HOLD', 0, vol_r, 'neutral')

    # Composite score
    total_score = sum(v[1] for v in signals.values())

    # Determine verdict
    if total_score >= 20:
        verdict = 'STRONG BUY'
        verdict_short = 'BUY'
    elif total_score >= 8:
        verdict = 'BUY'
        verdict_short = 'BUY'
    elif total_score <= -20:
        verdict = 'STRONG SELL'
        verdict_short = 'SELL'
    elif total_score <= -8:
        verdict = 'SELL'
        verdict_short = 'SELL'
    else:
        verdict = 'HOLD'
        verdict_short = 'HOLD'

    # Stop-loss / Take-profit levels (ATR-based)
    if atr > 0:
        stop_loss    = last_close - 2.0 * atr
        take_profit  = last_close + 3.0 * atr
        risk_reward  = (take_profit - last_close) / (last_close - stop_loss) if (last_close - stop_loss) > 0 else 0
    else:
        stop_loss = last_close * 0.97
        take_profit = last_close * 1.05
        risk_reward = 1.5

    return {
        'signals':       signals,
        'total_score':   total_score,
        'verdict':       verdict,
        'verdict_short': verdict_short,
        'xgb_pct':       xgb_pct,
        'rsi':           rsi,
        'stop_loss':     stop_loss,
        'take_profit':   take_profit,
        'risk_reward':   risk_reward,
        'vol_ratio':     vol_r,
        'atr':           atr,
    }

def run_backtest_engine(actual_prices, predicted_prices, initial_capital, commission, threshold_pct):
    capital     = float(initial_capital)
    position    = 0
    entry_price = 0.0
    trades      = []
    equity      = []

    for i in range(len(predicted_prices) - 1):
        price_now  = float(actual_prices[i])
        price_next = float(actual_prices[i + 1])
        pred_next  = float(predicted_prices[i])
        diff_pct   = (pred_next - price_now) / price_now * 100
        equity.append(capital + position * price_now)

        if diff_pct > threshold_pct and position == 0:
            shares = int((capital - commission) / price_now)
            if shares > 0:
                capital    -= shares * price_now + commission
                position    = shares
                entry_price = price_now
                trades.append({"Day": i, "Type": "BUY",  "Price": price_now,
                               "Shares": shares, "Capital": capital})
        elif diff_pct < -threshold_pct and position > 0:
            proceeds = position * price_now - commission
            pnl      = proceeds - (entry_price * position + commission)
            capital += proceeds
            trades.append({"Day": i, "Type": "SELL", "Price": price_now,
                           "Shares": position, "P&L": pnl, "Capital": capital})
            position    = 0
            entry_price = 0.0

    if position > 0:
        final_price = float(actual_prices[-1])
        proceeds    = position * final_price - commission
        pnl         = proceeds - (entry_price * position + commission)
        capital    += proceeds
        trades.append({"Day": len(actual_prices)-1, "Type": "SELL (EOD)",
                       "Price": final_price, "Shares": position,
                       "P&L": pnl, "Capital": capital})
        position = 0

    equity.append(capital)

    bh_shares  = int((initial_capital - commission) / float(actual_prices[0]))
    bh_final   = bh_shares * float(actual_prices[-1]) - commission
    bh_return  = (bh_final - initial_capital) / initial_capital * 100
    strat_return = (capital - initial_capital) / initial_capital * 100

    equity_series = pd.Series(equity)
    drawdown      = equity_series / equity_series.cummax() - 1
    max_drawdown  = float(drawdown.min() * 100)
    daily_returns = equity_series.pct_change().dropna()
    sharpe        = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                     if daily_returns.std() > 0 else 0.0)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty and "P&L" in trades_df.columns:
        closed        = trades_df[trades_df["Type"].str.contains("SELL")]
        win_trades    = (closed["P&L"] > 0).sum()
        loss_trades   = (closed["P&L"] <= 0).sum()
        win_rate      = win_trades / len(closed) * 100 if len(closed) > 0 else 0.0
        avg_win       = closed[closed["P&L"] > 0]["P&L"].mean()  if win_trades  > 0 else 0.0
        avg_loss      = closed[closed["P&L"] <= 0]["P&L"].mean() if loss_trades > 0 else 0.0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        total_trades  = len(closed)
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0.0
        total_trades = 0

    bh_equity = [initial_capital * (float(actual_prices[i]) / float(actual_prices[0]))
                 for i in range(len(actual_prices))]

    return {
        "final_capital":   capital,
        "strat_return":    strat_return,
        "bh_return":       bh_return,
        "max_drawdown":    max_drawdown,
        "sharpe":          sharpe,
        "win_rate":        win_rate,
        "total_trades":    total_trades,
        "avg_win":         avg_win,
        "avg_loss":        avg_loss,
        "profit_factor":   profit_factor,
        "equity_curve":    equity,
        "bh_equity":       bh_equity,
        "trades_df":       trades_df,
        "drawdown_series": drawdown.tolist(),
    }

def bootstrap_confidence_intervals(model, X_input, n_bootstrap=100, noise_std=0.02):
    all_preds = []
    for _ in range(n_bootstrap):
        noise = np.random.normal(0, noise_std, X_input.shape)
        all_preds.append(model.predict(X_input + noise))
    all_preds = np.array(all_preds)
    return (np.percentile(all_preds,  5, axis=0),
            np.percentile(all_preds, 50, axis=0),
            np.percentile(all_preds, 95, axis=0))

# ── Shariah Compliance ─────────────────────────────────────────────────────────
HARAM_TICKERS = {
    "BUD","STZ","SAM","BREW","ABEV","DEO","BF-B",
    "MO","PM","BTI","LO","VGR",
    "LVS","MGM","WYNN","CZR","PENN","DKNG","BYD",
    "JPM","BAC","WFC","C","GS","MS","AXP",
    "MET","PRU","AIG","ALL","TRV","CB",
    "HRL","TSN","SFD","CAG",
    "LMT","RTX","NOC","GD","HII",
}
QUESTIONABLE_TICKERS = {
    "DIS","NFLX","PARA","WBD","FOXA","SPOT",
    "MAR","HLT","H","IHG","WH",
    "V","MA","AXP","COF","USB","PNC",
}
HARAM_SECTORS_KW = ["bank","insurance","casino","gambling","alcohol","tobacco",
                     "brewing","distill","porn","adult","weapons","defense","firearm"]

@st.cache_data
def get_shariah_data(ticker_sym):
    """Fetch financial data for Shariah screening with multiple fallback strategies."""
    t = yf.Ticker(ticker_sym)
    info = {}

    # Strategy 1: try .info (may fail or return empty in newer yfinance)
    try:
        raw = t.info
        if raw and len(raw) > 5:  # valid response has many keys
            info = raw
    except Exception:
        pass

    # Strategy 2: fall back to fast_info for market cap at least
    if not info:
        try:
            fi = t.fast_info
            info = {
                "marketCap":   getattr(fi, "market_cap", 0) or 0,
                "sector":      "Unknown",
                "industry":    "Unknown",
                "longName":    ticker_sym,
                "totalDebt":   0,
                "totalAssets": 0,
                "totalCash":   0,
            }
        except Exception:
            pass

    # Strategy 3: try get_info() method (some yfinance versions)
    if not info:
        try:
            info = t.get_info()
        except Exception:
            pass

    if not info:
        return None

    def _safe(key, default=0):
        v = info.get(key, default)
        return v if v is not None else default

    market_cap   = _safe("marketCap",   1) or 1
    total_debt   = _safe("totalDebt",   0)
    total_assets = _safe("totalAssets", 1) or 1
    total_cash   = _safe("totalCash",   0)

    return {
        "debt_to_mktcap":  total_debt   / market_cap,
        "debt_to_assets":  total_debt   / total_assets,
        "cash_to_assets":  total_cash   / total_assets,
        "market_cap":      market_cap,
        "total_debt":      total_debt,
        "total_assets":    total_assets,
        "total_cash":      total_cash,
        "sector":          _safe("sector",   "Unknown"),
        "industry":        _safe("industry", "Unknown"),
        "company_name":    _safe("longName", ticker_sym),
    }

def check_shariah_compliance(ticker_sym, data):
    t         = ticker_sym.upper()
    ind_lower = data["industry"].lower()
    haram_hit = None
    if t in HARAM_TICKERS:
        haram_hit = "Known non-compliant ticker"
    else:
        for kw in HARAM_SECTORS_KW:
            if kw in ind_lower:
                haram_hit = data["industry"]
                break
    questionable = t in QUESTIONABLE_TICKERS
    d2mc = data["debt_to_mktcap"]
    d2a  = data["debt_to_assets"]
    c2a  = data["cash_to_assets"]
    r    = {
        "business":    {"pass": haram_hit is None, "haram_hit": haram_hit, "questionable": questionable},
        "debt_mktcap": {"pass": d2mc < 0.30, "value": d2mc, "label": f"Debt/MarketCap = {d2mc*100:.1f}% (< 30%)"},
        "debt_assets": {"pass": d2a  < 0.33, "value": d2a,  "label": f"Debt/Assets = {d2a*100:.1f}% (< 33%)"},
        "cash_assets": {"pass": c2a  < 0.33, "value": c2a,  "label": f"Cash/Assets = {c2a*100:.1f}% (< 33%)"},
    }
    all_pass = r["business"]["pass"] and r["debt_mktcap"]["pass"] and r["debt_assets"]["pass"] and r["cash_assets"]["pass"]
    if not r["business"]["pass"]:
        r["verdict"] = "NON-COMPLIANT"
    elif not all_pass:
        r["verdict"] = "NON-COMPLIANT"
    elif questionable:
        r["verdict"] = "QUESTIONABLE"
    else:
        r["verdict"] = "COMPLIANT"
    return r

# ── Methodology tab content helper ─────────────────────────────────────────────
def render_methodology_page(seq_len_val=30, ci_n=100, show_ci=True):
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;letter-spacing:.22em;
         text-transform:uppercase;color:#3d5068;margin-bottom:.4rem;">Technical Documentation</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:700;
         color:#eef2f7;letter-spacing:.08em;margin-bottom:1.4rem;">
         STOCK<span style="color:#00e5b0;">CAST</span> · Methodology & Model Architecture
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "#00e5b0", "Data Ingestion", "OHLCV via yfinance",
         "Up to 7 years of daily Open/High/Low/Close/Volume data is fetched from Yahoo Finance. "
         "Timezone normalization and MultiIndex flattening are applied for compatibility across yfinance versions."),
        ("02", "#4da6ff", "Feature Engineering", "20 Technical Indicators",
         f"Each trading day is described by 20 derived signals: MA5/10/20/50/200, EMA12/26, RSI(14), "
         f"MACD(12/26/9) with histogram, Bollinger Band width & %B, ATR(14), Volume Ratio, Momentum, "
         f"Returns(1d/5d), Volatility(20d), and High-Low%. Additionally, {seq_len_val} lag closes are appended "
         "as sequential memory, giving the model temporal context."),
        ("03", "#ffdd2d", "Train / Test Split", "80% train · 20% test (chronological)",
         "Data is split strictly chronologically — no shuffling — to prevent look-ahead bias. "
         "The model never sees future data during training. Evaluation is performed exclusively on the held-out 20%."),
        ("04", "#00e5b0", "XGBoost Regressor", "Gradient-boosted decision trees",
         "XGBoost is trained to predict the next day's closing price. Hyperparameters (n_estimators, "
         "max_depth, learning_rate) are configurable via the sidebar. Subsample=0.8 and colsample_bytree=0.8 "
         "provide regularisation. Early stopping uses the test set as an eval set."),
        ("05", "#4da6ff", "Bootstrap CI", f"{ci_n} resampling iterations" if show_ci else "Disabled",
         f"Confidence intervals are produced by running the model {ci_n} times on inputs perturbed with "
         "Gaussian noise (σ=1.5%). The 5th and 95th percentiles form the 95% CI ribbon. "
         "A wider band indicates higher forecast uncertainty."),
        ("06", "#ffdd2d", "Forward Forecast", "Iterative multi-step prediction",
         "Future prices are predicted by rolling: each day's predicted price feeds back as the next day's lag input. "
         "Error compounds over time — Days 1–3 are most reliable. Days 6+ are directional signals only. "
         "An expanding ±1.5σ uncertainty band visualises this degradation."),
        ("07", "#ff3d57", "Signal Generation", "BUY / SELL / HOLD",
         "A BUY signal fires when the predicted next-day return exceeds +1% (configurable). "
         "SELL fires below −1%. HOLD covers the neutral zone. The latest signal reflects the model's "
         "view on tomorrow's likely price movement."),
        ("08", "#00e5b0", "Backtesting Engine", "Walk-forward simulation",
         "The backtest replays XGBoost signals on the test-set prices: when a BUY fires, it buys "
         "as many shares as capital allows (minus commission). When SELL fires, it liquidates. "
         "KPIs: Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor, and equity curve vs Buy-and-Hold."),
    ]

    for num, color, title, subtitle, body in steps:
        st.markdown(f"""
        <div style="display:flex;gap:1.2rem;margin-bottom:1rem;
             background:#0d1117;border:1px solid #182030;border-left:3px solid {color};
             padding:1.1rem 1.4rem;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:1.6rem;font-weight:700;
               color:{color};opacity:.6;min-width:2.5rem;line-height:1.1;">{num}</div>
          <div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;font-weight:700;
                 letter-spacing:.14em;text-transform:uppercase;color:#eef2f7;">{title}</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;letter-spacing:.1em;
                 color:{color};margin-bottom:.4rem;">{subtitle}</div>
            <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.82rem;
                 color:#7a8fa8;line-height:1.6;">{body}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(255,61,87,0.04);border:1px solid rgba(255,61,87,0.2);
         border-left:3px solid #ff3d57;padding:1rem 1.5rem;margin-top:.5rem;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;letter-spacing:.15em;
           text-transform:uppercase;color:#ff3d57;margin-bottom:.4rem;">⚠ Key Limitations</div>
      <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.82rem;color:#7a8fa8;line-height:1.7;">
        This model uses <b style="color:#eef2f7;">price and volume data only</b>. It has no awareness of
        earnings releases, macroeconomic events, analyst upgrades, geopolitical news, or central bank decisions.
        A single unexpected event can invalidate any technical forecast.
        Multi-day forecasts degrade rapidly due to compounding prediction error.
        <b style="color:#ff3d57;">This is a research and educational tool — not financial advice.</b>
        Always consult a licensed financial advisor before making investment decisions.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Fetching {ticker} data..."):
        df = fetch_data(ticker, start_date, end_date)

    if df.empty:
        st.error(f"No data found for '{ticker}'. Please check the symbol.")
        st.stop()

    st.success(f"✓ {len(df)} trading days loaded for {ticker}")

    # ── Realism Warning Banner ─────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:rgba(255,221,45,0.05);border:1px solid rgba(255,221,45,0.35);
         border-left:4px solid #ffdd2d;padding:1rem 1.5rem;margin:.5rem 0 1rem 0;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.62rem;letter-spacing:.16em;
           text-transform:uppercase;color:#ffdd2d;margin-bottom:.4rem;">
        ⚠ Model Reality Check — Read Before Trading
      </div>
      <div style="font-family:'IBM Plex Sans',sans-serif;font-size:.8rem;color:#7a8fa8;line-height:1.6;">
        This model uses <b style="color:#eef2f7;">price &amp; volume data only.</b>
        It has <b style="color:#ff3d57;">zero awareness</b> of:
        &nbsp;📰 breaking news &nbsp;·&nbsp;
        📊 earnings releases &nbsp;·&nbsp;
        🏦 Fed/macro events &nbsp;·&nbsp;
        🧠 analyst upgrades &nbsp;·&nbsp;
        🌍 geopolitical events.
        A single unexpected headline can invalidate any technical forecast instantly.
        <b style="color:#ffdd2d;">Use signals as one input — never as sole decision.</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top-level tabs ─────────────────────────────────────────────────────────
    tab_analysis, tab_methodology = st.tabs(["📊  Analysis", "📖  Methodology"])

    with tab_methodology:
        render_methodology_page(seq_len_val=seq_len, ci_n=ci_bootstrap_n, show_ci=show_conf_interval)

    with tab_analysis:

        # ── Price Alert ────────────────────────────────────────────────────────────
        last_close = float(df['Close'].squeeze().iloc[-1])
        if alert_price > 0:
            if last_close >= alert_price:
                st.markdown(f'<div class="alert-box">🔔 {ticker} is at ${last_close:.2f} — AT or ABOVE your target of ${alert_price:.2f}</div>',
                            unsafe_allow_html=True)
            else:
                diff = alert_price - last_close
                st.markdown(f'<div class="alert-box">🔔 {ticker} at ${last_close:.2f} — ${diff:.2f} below your target of ${alert_price:.2f}</div>',
                            unsafe_allow_html=True)

        # ── Feature Engineering ────────────────────────────────────────────────────
        with st.spinner("Engineering technical features..."):
            df = add_technical_features(df)
        close_series = df['Close'].squeeze()

        # ── Candlestick Chart ──────────────────────────────────────────────────────
        st.subheader("Candlestick Chart")
        fig_candle = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.72, 0.28], vertical_spacing=0.02)
        fig_candle.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'].squeeze(), high=df['High'].squeeze(),
            low=df['Low'].squeeze(),   close=close_series,
            name="Price",
            increasing_line_color=C_GREEN,
            decreasing_line_color=C_RED,
        ), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['MA50'].squeeze(),
            name="MA50",  line=dict(color=C_YELLOW, width=1.2)), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['MA200'].squeeze(),
            name="MA200", line=dict(color=C_BLUE, width=1.2)), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'].squeeze(),
            name="BB Upper", line=dict(color=C_GREY, width=0.8, dash='dot')), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'].squeeze(),
            name="BB Lower", line=dict(color=C_GREY, width=0.8, dash='dot'),
            fill='tonexty', fillcolor='rgba(77,166,255,0.04)'), row=1, col=1)
        colors_vol = [C_GREEN if c >= o else C_RED
                      for c, o in zip(close_series, df['Open'].squeeze())]
        fig_candle.add_trace(go.Bar(x=df.index, y=df['Volume'].squeeze(),
            name="Volume", marker_color=colors_vol, opacity=0.5), row=2, col=1)
        candle_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")}
        fig_candle.update_layout(
            **candle_layout,
            title=dict(text=f"{ticker} · Candlestick · MA50/200 · Bollinger · Volume",
                       font=dict(color=C_GREEN, size=12)),
            xaxis_rangeslider_visible=False, height=560,
        )
        fig_candle.update_xaxes(gridcolor="#182030", linecolor="#182030", tickfont=dict(color=C_GREY))
        fig_candle.update_yaxes(gridcolor="#182030", linecolor="#182030", tickfont=dict(color=C_GREY))
        st.plotly_chart(fig_candle, use_container_width=True)

        # ── RSI + MACD ─────────────────────────────────────────────────────────────
        st.subheader("Technical Indicators")
        fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 row_heights=[0.5, 0.5], vertical_spacing=0.08,
                                 subplot_titles=["RSI (14)", "MACD (12/26/9)"])
        fig_tech.add_trace(go.Scatter(x=df.index, y=df['RSI'].squeeze(),
            name="RSI", line=dict(color=C_GREEN, width=1.5)), row=1, col=1)
        fig_tech.add_hline(y=70, line_dash="dash", line_color=C_RED,  row=1, col=1)
        fig_tech.add_hline(y=30, line_dash="dash", line_color=C_GREEN, row=1, col=1)
        fig_tech.add_hrect(y0=70, y1=100, fillcolor="rgba(255,61,87,0.04)",   line_width=0, row=1, col=1)
        fig_tech.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,229,176,0.04)",   line_width=0, row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=df.index, y=df['MACD'].squeeze(),
            name="MACD",   line=dict(color=C_GREEN, width=1.2)), row=2, col=1)
        fig_tech.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'].squeeze(),
            name="Signal", line=dict(color=C_BLUE, width=1.2)), row=2, col=1)
        macd_hist  = df['MACD_Hist'].squeeze()
        hist_colors = [C_GREEN if v >= 0 else C_RED for v in macd_hist]
        fig_tech.add_trace(go.Bar(x=df.index, y=macd_hist,
            name="Histogram", marker_color=hist_colors, opacity=0.65), row=2, col=1)
        subplot_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis', 'yaxis')}
        fig_tech.update_layout(**subplot_layout, height=450)
        fig_tech.update_xaxes(gridcolor="#182030", linecolor="#182030", tickfont=dict(color=C_GREY))
        fig_tech.update_yaxes(gridcolor="#182030", linecolor="#182030", tickfont=dict(color=C_GREY))
        fig_tech.update_yaxes(range=[0, 100], row=1, col=1)
        st.plotly_chart(fig_tech, use_container_width=True)

        # ── XGBoost ────────────────────────────────────────────────────────────────
        st.markdown('<div class="model-badge">🤖 MODEL: XGBoost Regressor · 20 Technical Features + Lag Window</div>',
                    unsafe_allow_html=True)

        with st.expander("📖  How this model works — methodology & limitations", expanded=False):
            st.markdown(f"""
            <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.84rem;color:#7a8fa8;line-height:1.7;">

            <b style="color:#eef2f7;font-family:'IBM Plex Mono',monospace;font-size:0.7rem;letter-spacing:.12em;text-transform:uppercase;">
            Feature Engineering</b><br>
            Each trading day is represented by <b style="color:#00e5b0;">20 technical indicators</b> computed from raw OHLCV data —
            moving averages (MA5–200, EMA12/26), RSI, MACD with histogram, Bollinger Bands width &amp; position,
            ATR, volume ratio, and momentum — plus <b style="color:#00e5b0;">{seq_len} lag closes</b> as sequential context.
            This gives the model both market-state awareness and short-term price memory.
            <br><br>

            <b style="color:#eef2f7;font-family:'IBM Plex Mono',monospace;font-size:0.7rem;letter-spacing:.12em;text-transform:uppercase;">
            Training &amp; Evaluation</b><br>
            Data is split <b style="color:#00e5b0;">80% train / 20% test</b> chronologically (no data leakage).
            XGBoost is trained to predict the <em>next day's closing price</em> given today's features.
            Model quality is measured on the held-out test set using RMSE, MAE, MAPE and R².
            <br><br>

            <b style="color:#eef2f7;font-family:'IBM Plex Mono',monospace;font-size:0.7rem;letter-spacing:.12em;text-transform:uppercase;">
            Confidence Intervals</b><br>
            The 95% CI ribbon is produced via <b style="color:#00e5b0;">bootstrap resampling</b>: the model is run
            {ci_bootstrap_n if show_conf_interval else 100}x on inputs with small Gaussian noise added (sigma=1.5%), and the 5th-95th percentile
            range across all runs forms the band. A <em>wider band = higher uncertainty</em>.
            <br><br>

            <b style="color:#eef2f7;font-family:'IBM Plex Mono',monospace;font-size:0.7rem;letter-spacing:.12em;text-transform:uppercase;">
            Forward Forecast Reliability</b><br>
            Multi-day forecasts roll predictions iteratively — each day's output feeds the next day's lag input.
            <b style="color:#ffdd2d;">Prediction errors compound.</b> Day 1–3 forecasts are most reliable.
            Days 7+ should be treated as directional trend signals only, not price targets.
            <br><br>

            <b style="color:#eef2f7;font-family:'IBM Plex Mono',monospace;font-size:0.7rem;letter-spacing:.12em;text-transform:uppercase;">
            Key Limitations</b><br>
            This model uses <em>only price and volume data</em>. It has no awareness of earnings releases,
            macroeconomic events, analyst upgrades, or breaking news. A single unexpected event
            can invalidate any technical forecast. <b style="color:#ff3d57;">This is a research tool, not financial advice.</b>

            </div>
            """, unsafe_allow_html=True)

        with st.spinner("Building feature matrix..."):
            X, y = build_xgb_dataset(df, seq_len)

        if len(X) < 50:
            st.error("Not enough data to train. Try a longer date range or smaller lookback window.")
            st.stop()

        split   = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        @st.cache_resource(show_spinner=False)
        def train_xgb_cached(_X_train, _y_train, _X_test, _y_test,
                              _n_est, _depth, _lr):
            m = XGBRegressor(
                n_estimators=_n_est, max_depth=_depth,
                learning_rate=_lr, subsample=0.8,
                colsample_bytree=0.8, random_state=42, verbosity=0
            )
            m.fit(_X_train, _y_train, eval_set=[(_X_test, _y_test)], verbose=False)
            return m

        with st.spinner("Training XGBoost model (cached after first run)..."):
            model = train_xgb_cached(X_train, y_train, X_test, y_test,
                                     n_estimators, max_depth, learning_rate)

        preds  = model.predict(X_test)
        actual = y_test
        rmse   = np.sqrt(mean_squared_error(actual, preds))
        mae    = mean_absolute_error(actual, preds)
        mape   = np.mean(np.abs((actual - preds) / actual)) * 100
        r2     = 1 - np.sum((actual - preds)**2) / np.sum((actual - np.mean(actual))**2)

        # ── Model Performance ──────────────────────────────────────────────────────
        st.subheader("Model Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", f"${rmse:.2f}")
        c2.metric("MAE",  f"${mae:.2f}")
        c3.metric("MAPE", f"{mape:.2f}%")
        c4.metric("R²",   f"{r2:.4f}")

        # Metric interpretation
        mape_label  = ("🟢 Excellent" if mape < 2 else "🟡 Good" if mape < 5 else "🟠 Fair" if mape < 10 else "🔴 Poor")
        r2_label    = ("🟢 Excellent" if r2 > 0.95 else "🟡 Good" if r2 > 0.85 else "🟠 Fair" if r2 > 0.70 else "🔴 Poor")
        st.markdown(
            f'<div style="background:#0d1117;border:1px solid #182030;padding:.7rem 1.3rem;'
            f'font-family:IBM Plex Mono,monospace;font-size:0.68rem;color:#3d5068;'
            f'display:flex;gap:2rem;flex-wrap:wrap;margin-top:-.3rem;">'
            f'<span>MAPE: {mape_label} &nbsp;·&nbsp; &lt;2% excellent · &lt;5% good · &lt;10% fair</span>'
            f'<span>R²: {r2_label} &nbsp;·&nbsp; &gt;0.95 excellent · &gt;0.85 good · &gt;0.70 fair</span>'
            f'<span style="color:#243346;">RMSE/MAE are in $ — lower is better</span>'
            f'</div>', unsafe_allow_html=True)

        # ── Confidence Score ──────────────────────────────────────────────────────
        # Composite score: 40% R², 30% MAPE, 20% directional accuracy, 10% data volume
        directional_correct = sum(
            1 for a, p in zip(actual[1:], preds[1:])
            if (p - actual[:-1][list(preds[1:]).index(p)]) * (a - actual[:-1][list(preds[1:]).index(p)]) > 0
        ) / max(len(actual) - 1, 1) * 100 if len(actual) > 1 else 50.0
        # Simpler directional accuracy
        dir_acc = sum(
            1 for i in range(1, len(actual))
            if (preds[i] - actual[i-1]) * (actual[i] - actual[i-1]) > 0
        ) / max(len(actual) - 1, 1) * 100

        r2_score_norm   = max(0, min(100, r2 * 100))
        mape_score_norm = max(0, min(100, 100 - mape * 5))
        data_score      = min(100, len(df) / 2000 * 100)
        confidence_score = (r2_score_norm * 0.40 + mape_score_norm * 0.30 +
                            dir_acc * 0.20 + data_score * 0.10)
        confidence_score = max(0, min(100, confidence_score))

        if confidence_score >= 80:
            conf_color = "#00e5b0"; conf_label = "HIGH CONFIDENCE"; conf_bar_bg = "rgba(0,229,176,0.15)"
        elif confidence_score >= 60:
            conf_color = "#ffdd2d"; conf_label = "MODERATE CONFIDENCE"; conf_bar_bg = "rgba(255,221,45,0.15)"
        else:
            conf_color = "#ff3d57"; conf_label = "LOW CONFIDENCE"; conf_bar_bg = "rgba(255,61,87,0.15)"

        filled_blocks = int(confidence_score / 5)   # 20 blocks total
        bar_html = "".join(
            f'<span style="display:inline-block;width:18px;height:10px;margin-right:2px;'
            f'background:{conf_color};opacity:{1.0 if i < filled_blocks else 0.12};border-radius:1px;"></span>'
            for i in range(20)
        )

        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #182030;border-left:3px solid {conf_color};
             padding:1.2rem 1.6rem;margin:1rem 0;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;">
            <div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;letter-spacing:.18em;
                   text-transform:uppercase;color:#3d5068;margin-bottom:.3rem;">MODEL CONFIDENCE SCORE</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:2.2rem;font-weight:700;
                   color:{conf_color};line-height:1;">{confidence_score:.0f}<span style="font-size:1rem;color:#3d5068;">/100</span></div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;letter-spacing:.14em;
                   color:{conf_color};margin-top:.3rem;">{conf_label}</div>
            </div>
            <div style="flex:1;min-width:220px;">
              <div style="margin-bottom:.6rem;">{bar_html}</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:.3rem .8rem;
                   font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#3d5068;">
                <span>R² fit &nbsp;<b style="color:#7a8fa8;">{r2_score_norm:.0f}/100</b> <span style="color:#243346;">(×0.40)</span></span>
                <span>MAPE accuracy &nbsp;<b style="color:#7a8fa8;">{mape_score_norm:.0f}/100</b> <span style="color:#243346;">(×0.30)</span></span>
                <span>Directional accuracy &nbsp;<b style="color:#7a8fa8;">{dir_acc:.0f}/100</b> <span style="color:#243346;">(×0.20)</span></span>
                <span>Data volume &nbsp;<b style="color:#7a8fa8;">{data_score:.0f}/100</b> <span style="color:#243346;">(×0.10)</span></span>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Feature Importance ─────────────────────────────────────────────────────
        st.subheader("Feature Importance")
        feature_cols = [
            'MA5','MA10','MA20','MA50','EMA12','EMA26',
            'RSI','MACD','MACD_Signal','MACD_Hist',
            'BB_Width','BB_Pct',
            'Returns','Returns_5d','Volatility','Momentum',
            'Volume_Ratio','High_Low_Pct','Close_Open_Pct','ATR'
        ]
        lag_names        = [f'Lag_{i+1}' for i in range(seq_len)]
        all_feature_names = feature_cols + lag_names
        importances      = model.feature_importances_
        imp_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})\
                   .sort_values('importance', ascending=True).tail(20)

        fig_imp = go.Figure(go.Bar(
            x=imp_df['importance'], y=imp_df['feature'], orientation='h',
            marker=dict(color=imp_df['importance'],
                        colorscale=[[0, "#0d1117"], [0.5, "#0d3d2e"], [1, C_GREEN]],
                        showscale=False)
        ))
        fig_imp.update_layout(**PLOTLY_LAYOUT,
            title=dict(text="Top 20 Feature Importances", font=dict(color=C_GREEN, size=12)),
            height=450, xaxis_title="Importance Score")
        st.plotly_chart(fig_imp, use_container_width=True)

        # ── Actual vs Predicted ────────────────────────────────────────────────────
        st.subheader("Actual vs Predicted")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(y=actual, name="Actual",
            line=dict(color=C_BLUE, width=1.5),
            fill='tozeroy', fillcolor='rgba(77,166,255,0.04)'))
        fig1.add_trace(go.Scatter(y=preds, name="XGBoost Predicted",
            line=dict(color=C_GREEN, width=1.5, dash='dot')))
        fig1.update_layout(**PLOTLY_LAYOUT,
            title=dict(text=f"{ticker} · XGBoost Model Fit (Test Set)", font=dict(color=C_GREEN, size=12)),
            height=350)
        st.plotly_chart(fig1, use_container_width=True)

        # ── Multi-Factor Buy/Sell Signal Engine ──────────────────────────────────
        st.subheader("Signal Intelligence")

        # Build composite signal
        composite = compute_composite_signal(df, last_close, preds[-1], preds, actual)
        verdict        = composite['verdict']
        verdict_short  = composite['verdict_short']
        total_score    = composite['total_score']
        xgb_pct        = composite['xgb_pct']
        stop_loss      = composite['stop_loss']
        take_profit    = composite['take_profit']
        risk_reward    = composite['risk_reward']
        rsi_val        = composite['rsi']
        vol_ratio      = composite['vol_ratio']
        atr_val        = composite['atr']
        sigs           = composite['signals']

        # ── 🔔 Signal Alert Detection ──────────────────────────────────────────
        if alert_on_signal_change:
            prev_verdict = st.session_state.alert_signals.get(ticker)
            if prev_verdict is not None and prev_verdict != verdict_short:
                _alert_color = {"BUY": "#00e5b0", "SELL": "#ff3d57"}.get(verdict_short, "#ffdd2d")
                _alert_icon  = {"BUY": "⬆", "SELL": "⬇"}.get(verdict_short, "◆")
                st.markdown(
                    f'<div style="background:rgba({",".join(str(int(_alert_color.lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.12);'
                    f'border:1px solid {_alert_color};border-left:4px solid {_alert_color};'
                    f'padding:.8rem 1.4rem;margin-bottom:1rem;font-family:IBM Plex Mono,monospace;'
                    f'font-size:.8rem;color:{_alert_color};letter-spacing:.04em;">'
                    f'🔔 <b>SIGNAL ALERT — {ticker}</b> &nbsp;|&nbsp; '
                    f'{prev_verdict} → {_alert_icon} {verdict_short} &nbsp;|&nbsp; Score: {total_score:+.0f}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            # Always update stored verdict after this run
            st.session_state.alert_signals[ticker] = verdict_short

        verdict_css = 'sell' if verdict_short == 'SELL' else 'hold' if verdict_short == 'HOLD' else ''
        sign        = '+' if xgb_pct >= 0 else ''

        # ── Main Signal Panel ──────────────────────────────────────────────────
        rr_color    = 'positive' if risk_reward >= 1.5 else 'negative' if risk_reward < 1 else 'neutral'
        sl_color    = 'negative'
        tp_color    = 'positive'
        score_color = '#00e5b0' if total_score > 0 else '#ff3d57' if total_score < 0 else '#ffdd2d'

        st.markdown(f"""
        <div class="signal-panel-v2">
            <div class="signal-main-v2 {verdict_css}">
                <div class="signal-label-v2">Composite Signal</div>
                <div class="signal-action-v2 {verdict_css}">{verdict}</div>
                <div class="signal-pct-v2">{sign}{xgb_pct:.2f}% forecast</div>
                <div class="signal-label-v2" style="margin-top:8px;">Score: <span style="color:{score_color};font-size:0.9rem;font-weight:800;">{total_score:+.0f}</span> / ±100</div>
            </div>
            <div class="signal-details-v2">
                <div class="sig-detail-card-v2 {tp_color}">
                    <div class="sig-detail-label-v2">Take Profit</div>
                    <div class="sig-detail-val-v2">${take_profit:.2f}</div>
                    <div class="sig-detail-sub-v2">+{((take_profit-last_close)/last_close*100):.1f}% · 3× ATR</div>
                </div>
                <div class="sig-detail-card-v2 {sl_color}">
                    <div class="sig-detail-label-v2">Stop Loss</div>
                    <div class="sig-detail-val-v2">${stop_loss:.2f}</div>
                    <div class="sig-detail-sub-v2">{((stop_loss-last_close)/last_close*100):.1f}% · 2× ATR</div>
                </div>
                <div class="sig-detail-card-v2 {rr_color}">
                    <div class="sig-detail-label-v2">Risk / Reward</div>
                    <div class="sig-detail-val-v2">{risk_reward:.2f}×</div>
                    <div class="sig-detail-sub-v2">{"✓ Favorable" if risk_reward >= 1.5 else "⚠ Marginal" if risk_reward >= 1 else "✗ Unfavorable"}</div>
                </div>
                <div class="sig-detail-card-v2 {'positive' if rsi_val < 50 else 'negative'}">
                    <div class="sig-detail-label-v2">RSI (14)</div>
                    <div class="sig-detail-val-v2">{rsi_val:.1f}</div>
                    <div class="sig-detail-sub-v2">{"Oversold" if rsi_val < 30 else "Overbought" if rsi_val > 70 else "Neutral zone"}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Per-Indicator Breakdown ────────────────────────────────────────────
        st.markdown('<div class="composite-meter"><div class="meter-title">⚙ Per-Indicator Signal Breakdown</div>', unsafe_allow_html=True)

        indicator_rows_html = ""
        for ind_name, (ind_sig, ind_score, ind_val, ind_class) in sigs.items():
            bar_width = min(100, abs(ind_score) / 35 * 100)
            sig_class = ind_sig.lower() if ind_sig in ('BUY','SELL') else 'hold'
            indicator_rows_html += f"""
            <div class="signal-indicator-row">
                <span class="sir-label">{ind_name}</span>
                <div class="sir-bar-bg"><div class="sir-bar {ind_class}" style="width:{bar_width:.0f}%;"></div></div>
                <span class="sir-val">{ind_val:.2f}</span>
                <span class="sir-signal {sig_class}">{ind_sig}</span>
            </div>"""

        st.markdown(indicator_rows_html + '</div>', unsafe_allow_html=True)

        # ── Why This Signal? Human-readable explanation ────────────────────────
        reason_lines = []
        emoji_map = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}
        for ind_name, (ind_sig, ind_score, ind_val, ind_class) in sigs.items():
            if ind_sig == 'BUY':
                if ind_name == 'XGBoost Forecast':
                    reason_lines.append(f"🟢 XGBoost model predicts <b style='color:#00e5b0;'>{xgb_pct:+.2f}%</b> move tomorrow")
                elif ind_name == 'RSI (14)':
                    reason_lines.append(f"🟢 RSI is <b style='color:#00e5b0;'>{ind_val:.1f}</b> — {'oversold territory' if ind_val < 30 else 'leaning oversold'}")
                elif ind_name == 'MACD Cross':
                    reason_lines.append(f"🟢 MACD shows {'fresh bullish crossover' if abs(ind_score) >= 20 else 'bullish momentum'}")
                elif ind_name == 'Bollinger %B':
                    reason_lines.append(f"🟢 Price near lower Bollinger Band (%B = {ind_val:.2f}) — potential bounce")
                elif ind_name == 'MA Cross':
                    reason_lines.append(f"🟢 MA50 above MA200 — Golden Cross (uptrend)")
                elif ind_name == 'Volume':
                    reason_lines.append(f"🟢 Volume {ind_val:.1f}× average confirms buying pressure")
            elif ind_sig == 'SELL':
                if ind_name == 'XGBoost Forecast':
                    reason_lines.append(f"🔴 XGBoost model predicts <b style='color:#ff3d57;'>{xgb_pct:+.2f}%</b> move tomorrow")
                elif ind_name == 'RSI (14)':
                    reason_lines.append(f"🔴 RSI is <b style='color:#ff3d57;'>{ind_val:.1f}</b> — {'overbought territory' if ind_val > 70 else 'leaning overbought'}")
                elif ind_name == 'MACD Cross':
                    reason_lines.append(f"🔴 MACD shows {'fresh bearish crossover' if abs(ind_score) >= 20 else 'bearish momentum'}")
                elif ind_name == 'Bollinger %B':
                    reason_lines.append(f"🔴 Price near upper Bollinger Band (%B = {ind_val:.2f}) — potential reversal")
                elif ind_name == 'MA Cross':
                    reason_lines.append(f"🔴 MA50 below MA200 — Death Cross (downtrend)")
                elif ind_name == 'Volume':
                    reason_lines.append(f"🔴 Volume {ind_val:.1f}× average confirms selling pressure")
            else:
                if ind_name == 'XGBoost Forecast':
                    reason_lines.append(f"🟡 XGBoost predicts only {xgb_pct:+.2f}% — not enough directional conviction")

        border_color = '#00e5b0' if verdict_short == 'BUY' else '#ff3d57' if verdict_short == 'SELL' else '#ffdd2d'
        reasons_html = "".join(
            f'<div style="margin-bottom:.4rem;font-size:.8rem;color:#7a8fa8;">{r}</div>'
            for r in reason_lines
        )
        st.markdown(f"""
        <div style="background:var(--bg2);border:1px solid var(--border);border-left:4px solid {border_color};
             padding:1.1rem 1.5rem;margin:.8rem 0;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:.6rem;letter-spacing:.18em;
               text-transform:uppercase;color:#3d5068;margin-bottom:.7rem;">
            💡 Why This Signal? — <span style="color:{border_color};">{verdict}</span>
          </div>
          <div style="font-family:'IBM Plex Sans',sans-serif;line-height:1.7;">
            {reasons_html if reasons_html else '<span style="color:#3d5068;font-size:.8rem;">All indicators neutral — no strong directional edge detected.</span>'}
          </div>
          <div style="margin-top:.8rem;padding-top:.7rem;border-top:1px solid var(--border);
               font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:#3d5068;">
            ⚠ Technical signals use price & volume only. See News Sentiment section below for headline analysis.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Historical signal list ─────────────────────────────────────────────
        signal_list = []
        for i in range(len(preds)):
            diff_pct = (preds[i] - actual[i]) / actual[i] * 100
            if diff_pct > 1.0:
                signal_list.append("BUY")
            elif diff_pct < -1.0:
                signal_list.append("SELL")
            else:
                signal_list.append("HOLD")

        buy_idx  = [i for i, s in enumerate(signal_list) if s == "BUY"]
        sell_idx = [i for i, s in enumerate(signal_list) if s == "SELL"]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=actual, name="Actual Price",
            line=dict(color=C_GREY, width=1)))
        fig2.add_trace(go.Scatter(x=buy_idx,  y=[actual[i] for i in buy_idx],
            mode='markers', name='BUY',
            marker=dict(color=C_GREEN, symbol='triangle-up', size=10,
                        line=dict(width=1, color='#080b0f'))))
        fig2.add_trace(go.Scatter(x=sell_idx, y=[actual[i] for i in sell_idx],
            mode='markers', name='SELL',
            marker=dict(color=C_RED, symbol='triangle-down', size=10,
                        line=dict(width=1, color='#080b0f'))))
        # Add stop-loss and take-profit reference lines
        fig2.add_hline(y=stop_loss,   line_dash="dot", line_color=C_RED,    line_width=1,
                       annotation_text=f"Stop ${stop_loss:.2f}", annotation_font=dict(color=C_RED,   size=9))
        fig2.add_hline(y=take_profit, line_dash="dot", line_color=C_GREEN,  line_width=1,
                       annotation_text=f"Target ${take_profit:.2f}", annotation_font=dict(color=C_GREEN, size=9))
        fig2.update_layout(**PLOTLY_LAYOUT,
            title=dict(text=f"{ticker} · Signal Map — BUY ▲ / SELL ▼ (XGBoost ±1% threshold) · SL/TP lines shown",
                       font=dict(color=C_GREEN, size=12)),
            height=380)
        st.plotly_chart(fig2, use_container_width=True)

        signal_df = pd.DataFrame({
            "Day":               range(1, len(signal_list) + 1),
            "Actual Price ($)":  [f"${p:.2f}" for p in actual],
            "Predicted ($)":     [f"${p:.2f}" for p in preds],
            "Signal":            signal_list,
            "Δ%":                [f"{(preds[i]-actual[i])/actual[i]*100:+.2f}%" for i in range(len(preds))],
        })
        st.dataframe(signal_df.tail(10), use_container_width=True, hide_index=True)

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # ── 📰 NEWS SENTIMENT ANALYSIS ────────────────────────────────────────
        # ══════════════════════════════════════════════════════════════════════
        st.subheader("📰 News Sentiment")
        st.caption("Live headlines fetched from Yahoo Finance · Sentiment scored via TextBlob NLP")

        @st.cache_data(ttl=900)   # cache 15 min so reruns don't re-fetch
        def fetch_news_headlines(sym):
            """Fetch recent news headlines for a ticker via yfinance (no API key needed)."""
            try:
                tk   = yf.Ticker(sym)
                news = tk.news or []
                out  = []
                for item in news[:10]:
                    title     = item.get("title", "")
                    publisher = item.get("publisher", "")
                    link      = item.get("link", "")
                    ts        = item.get("providerPublishTime", 0)
                    age_h     = int((pd.Timestamp.now().timestamp() - ts) / 3600) if ts else 0
                    if title:
                        out.append({"title": title, "publisher": publisher,
                                    "link": link, "age_h": age_h})
                return out
            except Exception:
                return []

        def analyze_sentiment_textblob(headlines):
            """Score each headline with TextBlob polarity (-1 → +1)."""
            try:
                from textblob import TextBlob
            except ImportError:
                return None, []          # TextBlob not installed — handled below

            scored = []
            for h in headlines:
                pol = TextBlob(h["title"]).sentiment.polarity
                sub = TextBlob(h["title"]).sentiment.subjectivity
                scored.append({**h, "polarity": pol, "subjectivity": sub})
            return scored

        def sentiment_label(avg_pol):
            if avg_pol >= 0.15:
                return "VERY POSITIVE", "#00e5b0", "🟢"
            elif avg_pol >= 0.03:
                return "POSITIVE",      "#00e5b0", "🟢"
            elif avg_pol <= -0.15:
                return "VERY NEGATIVE", "#ff3d57", "🔴"
            elif avg_pol <= -0.03:
                return "NEGATIVE",      "#ff3d57", "🔴"
            else:
                return "NEUTRAL",       "#ffdd2d", "⚪"

        with st.spinner("Fetching latest news…"):
            headlines = fetch_news_headlines(ticker)

        if not headlines:
            st.info("No recent news found for this ticker. Try a major index stock like AAPL, TSLA, MSFT.")
        else:
            scored = analyze_sentiment_textblob(headlines)

            if scored is None:
                # TextBlob missing — show headlines only, no scores
                st.warning("Install `textblob` in requirements.txt for sentiment scoring. Showing headlines only.")
                for h in headlines:
                    st.markdown(f"🔹 [{h['title']}]({h['link']})  \n"
                                f"<span style='font-size:.72rem;color:#3d5068;'>"
                                f"{h['publisher']} · {h['age_h']}h ago</span>",
                                unsafe_allow_html=True)
            else:
                polarities  = [s["polarity"] for s in scored]
                avg_polarity = sum(polarities) / len(polarities) if polarities else 0
                label, sent_color, sent_icon = sentiment_label(avg_polarity)

                # ── Overall Sentiment Banner ───────────────────────────────────
                bullish_count  = sum(1 for p in polarities if p >  0.03)
                bearish_count  = sum(1 for p in polarities if p < -0.03)
                neutral_count  = len(polarities) - bullish_count - bearish_count

                ns1, ns2, ns3, ns4 = st.columns(4)
                ns1.metric("📰 News Sentiment", f"{sent_icon} {label}")
                ns2.metric("Avg Polarity Score", f"{avg_polarity:+.3f}",
                           delta="Bullish lean" if avg_polarity > 0.03
                                 else "Bearish lean" if avg_polarity < -0.03 else "Neutral",
                           delta_color="normal" if avg_polarity > 0.03
                                       else "inverse" if avg_polarity < -0.03 else "off")
                ns3.metric("🟢 Positive Headlines", bullish_count)
                ns4.metric("🔴 Negative Headlines", bearish_count)

                # ── Combined Signal Banner (AI + News) ────────────────────────
                _tech_bull = verdict_short == "BUY"
                _tech_bear = verdict_short == "SELL"
                _news_bull = avg_polarity > 0.03
                _news_bear = avg_polarity < -0.03

                if _tech_bull and _news_bull:
                    combo_label = "⬆ STRONG CONFLUENCE — Technical BUY + Positive News"
                    combo_color = "#00e5b0"
                elif _tech_bear and _news_bear:
                    combo_label = "⬇ STRONG CONFLUENCE — Technical SELL + Negative News"
                    combo_color = "#ff3d57"
                elif _tech_bull and _news_bear:
                    combo_label = "⚠ DIVERGENCE — Technical BUY but Negative News sentiment"
                    combo_color = "#ffdd2d"
                elif _tech_bear and _news_bull:
                    combo_label = "⚠ DIVERGENCE — Technical SELL but Positive News sentiment"
                    combo_color = "#ffdd2d"
                else:
                    combo_label = "◆ MIXED — No strong confluence between technicals and news"
                    combo_color = "#3d5068"

                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.02);border:1px solid {combo_color};'
                    f'border-left:4px solid {combo_color};padding:.8rem 1.4rem;margin:.6rem 0 1rem 0;'
                    f'font-family:IBM Plex Mono,monospace;font-size:.78rem;color:{combo_color};'
                    f'letter-spacing:.04em;">🧠 Combined Intelligence: {combo_label}</div>',
                    unsafe_allow_html=True
                )

                # ── Headlines with per-article sentiment bars ──────────────────
                with st.expander(f"📋 {len(scored)} Recent Headlines — click to expand", expanded=True):
                    for h in scored:
                        pol   = h["polarity"]
                        _hcol = "#00e5b0" if pol > 0.03 else "#ff3d57" if pol < -0.03 else "#ffdd2d"
                        _hico = "🟢" if pol > 0.03 else "🔴" if pol < -0.03 else "⚪"
                        _bar_pct = min(100, int(abs(pol) * 300))   # scale to visible width

                        st.markdown(
                            f'<div style="padding:.55rem 0;border-bottom:1px solid #182030;">'
                            f'  <div style="display:flex;align-items:flex-start;gap:.6rem;">'
                            f'    <span style="font-size:.9rem;margin-top:.05rem;">{_hico}</span>'
                            f'    <div style="flex:1;">'
                            f'      <a href="{h["link"]}" target="_blank" style="color:#eef2f7;'
                            f'         font-family:IBM Plex Sans,sans-serif;font-size:.82rem;'
                            f'         text-decoration:none;line-height:1.4;">{h["title"]}</a>'
                            f'      <div style="display:flex;align-items:center;gap:.8rem;margin-top:.3rem;">'
                            f'        <span style="font-family:IBM Plex Mono,monospace;font-size:.62rem;'
                            f'               color:#3d5068;">{h["publisher"]} · {h["age_h"]}h ago</span>'
                            f'        <div style="flex:1;max-width:120px;height:3px;'
                            f'             background:#182030;border-radius:2px;">'
                            f'          <div style="width:{_bar_pct}%;height:100%;'
                            f'               background:{_hcol};border-radius:2px;"></div></div>'
                            f'        <span style="font-family:IBM Plex Mono,monospace;font-size:.62rem;'
                            f'               color:{_hcol};">{pol:+.3f}</span>'
                            f'      </div>'
                            f'    </div>'
                            f'  </div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                # ── Polarity distribution mini chart ──────────────────────────
                if len(scored) >= 3:
                    fig_sent = go.Figure()
                    titles_short = [s["title"][:40] + "…" if len(s["title"]) > 40
                                    else s["title"] for s in scored]
                    bar_colors   = [C_GREEN if p > 0.03 else C_RED if p < -0.03 else C_YELLOW
                                    for p in polarities]
                    fig_sent.add_trace(go.Bar(
                        x=polarities, y=titles_short,
                        orientation="h",
                        marker_color=bar_colors,
                        text=[f"{p:+.3f}" for p in polarities],
                        textposition="outside",
                        textfont=dict(family="IBM Plex Mono", size=9, color="#7a8fa8"),
                    ))
                    fig_sent.add_vline(x=0, line_color="#243346", line_width=1)
                    fig_sent.add_vline(x=avg_polarity, line_color=sent_color,
                                       line_dash="dot", line_width=1.5,
                                       annotation_text=f"avg {avg_polarity:+.3f}",
                                       annotation_font=dict(color=sent_color, size=9))
                    _sent_layout = {**PLOTLY_LAYOUT, "margin": dict(l=10, r=80, t=30, b=10)}
                    fig_sent.update_layout(
                        **_sent_layout,
                        title=dict(text=f"{ticker} · Headline Sentiment Polarity (TextBlob)",
                                   font=dict(color=C_GREEN, size=11)),
                        height=max(220, len(scored) * 32),
                        xaxis_title="Polarity (negative ← 0 → positive)",
                        xaxis=dict(range=[-1, 1], gridcolor="#182030",
                                   linecolor="#182030", zeroline=False,
                                   tickfont=dict(color="#3d5068", size=9)),
                    )
                    st.plotly_chart(fig_sent, use_container_width=True)

                st.caption("⚠ Sentiment is based on headline text only — not article content. "
                           "Use as a supplementary signal, not a sole decision factor.")

        st.divider()

        # ── Future Forecast ────────────────────────────────────────────────────────
        st.subheader(f"Forecast — Next {future_days} Days")
        future_prices    = []
        last_known_close = float(df['Close'].squeeze().iloc[-1])
        last_row_feats   = X[-1].copy()
        current_close    = last_known_close

        for d in range(future_days):
            next_pred = float(model.predict(last_row_feats.reshape(1, -1))[0])
            future_prices.append(next_pred)
            n_tech         = len(feature_cols)
            lags           = last_row_feats[n_tech:]
            new_lags       = np.append(lags[1:], next_pred)
            last_row_feats = np.concatenate([last_row_feats[:n_tech], new_lags])
            current_close  = next_pred

        trend_color = C_GREEN if future_prices[-1] > last_close else C_RED

        # Build expanding uncertainty band even without full bootstrap (lightweight ±σ proxy)
        price_std    = float(df['Close'].squeeze().pct_change().std())  # daily vol
        decay_upper  = [future_prices[i] * (1 + price_std * np.sqrt(i + 1) * 1.5) for i in range(future_days)]
        decay_lower  = [future_prices[i] * (1 - price_std * np.sqrt(i + 1) * 1.5) for i in range(future_days)]

        fig3 = go.Figure()
        # Expanding uncertainty shading
        fig3.add_trace(go.Scatter(
            x=list(range(future_days)), y=decay_upper,
            line=dict(color="rgba(0,229,176,0)"), showlegend=False, hoverinfo="skip"))
        fig3.add_trace(go.Scatter(
            x=list(range(future_days)), y=decay_lower,
            name="Uncertainty band", fill="tonexty",
            fillcolor="rgba(0,229,176,0.07)", line=dict(color="rgba(0,229,176,0)"), hoverinfo="skip"))

        fig3.add_hline(y=last_close, line_dash="dash", line_color=C_GREY,
                       annotation_text=f"Last close ${last_close:.2f}",
                       annotation_font_color=C_GREY)
        if alert_price > 0:
            fig3.add_hline(y=alert_price, line_dash="dash", line_color=C_YELLOW,
                           annotation_text=f"Target ${alert_price:.2f}",
                           annotation_font_color=C_YELLOW)
        fig3.add_trace(go.Scatter(
            x=list(range(future_days)), y=future_prices,
            mode='lines+markers', name='XGBoost Forecast',
            line=dict(color=trend_color, width=2),
            marker=dict(size=7, color=trend_color, line=dict(width=1, color="#080b0f"))
        ))
        # Vertical reliability boundary at day 5
        if future_days > 5:
            fig3.add_vline(x=4.5, line_dash="dot", line_color="#243346",
                           annotation_text="↑ Higher confidence  |  Lower confidence ↓",
                           annotation_font=dict(color="#3d5068", size=9),
                           annotation_position="top")

        fig3.update_layout(**PLOTLY_LAYOUT,
            title=dict(text=f"{ticker} · {future_days}-Day Price Forecast (XGBoost) · Band shows ±1.5σ uncertainty growth",
                       font=dict(color=C_GREEN, size=12)),
            xaxis_title="Days from today", yaxis_title="Price (USD)", height=380)
        st.plotly_chart(fig3, use_container_width=True)

        if future_days > 5:
            st.markdown(
                '<div style="background:rgba(255,221,45,0.04);border:1px solid rgba(255,221,45,0.2);'
                'border-left:3px solid #ffdd2d;padding:.6rem 1.2rem;font-family:IBM Plex Mono,monospace;'
                'font-size:0.7rem;color:#ffdd2d;letter-spacing:.05em;margin-top:-.5rem;">'
                '⚠ Forecast reliability decreases significantly beyond Day 5. '
                'Errors compound in iterative multi-step prediction. Use Days 6+ as directional signals only.'
                '</div>', unsafe_allow_html=True)

        future_df = pd.DataFrame({
            "Day":                 [f"+{i+1}" for i in range(future_days)],
            "Predicted Price ($)": [f"${p:.2f}" for p in future_prices],
            "vs Last Close":       [f"{'▲' if p > last_close else '▼'} {abs(p - last_close):.2f}"
                                    f" ({(p-last_close)/last_close*100:+.2f}%)" for p in future_prices]
        })
        st.dataframe(future_df, use_container_width=True, hide_index=True)

        # ── Backtesting ────────────────────────────────────────────────────────────
        if run_backtest and not fast_mode:
            st.subheader("Backtesting Engine")
            st.markdown(
                f'<div class="model-badge">STRATEGY: XGBoost Signal ±{bt_signal_threshold}% '
                f'| Capital: ${bt_initial_capital:,.0f} | Commission: ${bt_commission}/trade</div>',
                unsafe_allow_html=True)

            with st.spinner("Running backtest simulation..."):
                bt = run_backtest_engine(
                    actual_prices=actual, predicted_prices=preds,
                    initial_capital=bt_initial_capital,
                    commission=bt_commission, threshold_pct=bt_signal_threshold,
                )

            strat_color = "bt-card-value-green" if bt["strat_return"] >= 0 else "bt-card-value-red"
            bh_color    = "bt-card-value-green" if bt["bh_return"]    >= 0 else "bt-card-value-red"
            dd_color    = "bt-card-value-red"   if bt["max_drawdown"] < -10 else "bt-card-value"
            sh_color    = "bt-card-value-green" if bt["sharpe"]       >= 1  else "bt-card-value-red"

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f'<div class="bt-card"><div class="bt-card-label">Strategy Return</div>'
                        f'<div class="{strat_color}">{bt["strat_return"]:+.2f}%</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="bt-card"><div class="bt-card-label">Buy &amp; Hold Return</div>'
                        f'<div class="{bh_color}">{bt["bh_return"]:+.2f}%</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="bt-card"><div class="bt-card-label">Max Drawdown</div>'
                        f'<div class="{dd_color}">{bt["max_drawdown"]:.2f}%</div></div>', unsafe_allow_html=True)
            k4.markdown(f'<div class="bt-card"><div class="bt-card-label">Sharpe Ratio</div>'
                        f'<div class="{sh_color}">{bt["sharpe"]:.2f}</div></div>', unsafe_allow_html=True)

            k5, k6, k7, k8 = st.columns(4)
            k5.markdown(f'<div class="bt-card"><div class="bt-card-label">Final Capital</div>'
                        f'<div class="bt-card-value">${bt["final_capital"]:,.0f}</div></div>', unsafe_allow_html=True)
            k6.markdown(f'<div class="bt-card"><div class="bt-card-label">Total Trades</div>'
                        f'<div class="bt-card-value">{bt["total_trades"]}</div></div>', unsafe_allow_html=True)
            k7.markdown(f'<div class="bt-card"><div class="bt-card-label">Win Rate</div>'
                        f'<div class="bt-card-value">{bt["win_rate"]:.1f}%</div></div>', unsafe_allow_html=True)
            k8.markdown(f'<div class="bt-card"><div class="bt-card-label">Profit Factor</div>'
                        f'<div class="bt-card-value">{bt["profit_factor"]:.2f}x</div></div>', unsafe_allow_html=True)

            # Equity Curve
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(y=bt["equity_curve"], name="XGBoost Strategy",
                line=dict(color=C_GREEN, width=2),
                fill="tozeroy", fillcolor="rgba(0,229,176,0.05)"))
            fig_eq.add_trace(go.Scatter(y=bt["bh_equity"], name="Buy & Hold",
                line=dict(color=C_BLUE, width=1.5, dash="dot")))
            fig_eq.add_hline(y=bt_initial_capital, line_dash="dash", line_color=C_GREY,
                             annotation_text=f"Start ${bt_initial_capital:,}",
                             annotation_font_color=C_GREY)
            fig_eq.update_layout(**PLOTLY_LAYOUT,
                title=dict(text=f"{ticker} · Strategy Equity Curve vs Buy & Hold",
                           font=dict(color=C_GREEN, size=12)),
                yaxis_title="Portfolio Value ($)", xaxis_title="Trading Day (test set)", height=380)
            st.plotly_chart(fig_eq, use_container_width=True)

            # Drawdown
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(y=[d * 100 for d in bt["drawdown_series"]],
                name="Drawdown", line=dict(color=C_RED, width=1.5),
                fill="tozeroy", fillcolor="rgba(255,61,87,0.07)"))
            fig_dd.update_layout(**PLOTLY_LAYOUT,
                title=dict(text=f"{ticker} · Strategy Drawdown (%)", font=dict(color=C_RED, size=12)),
                yaxis_title="Drawdown (%)", xaxis_title="Trading Day (test set)", height=250)
            st.plotly_chart(fig_dd, use_container_width=True)

            # Trade log
            if not bt["trades_df"].empty:
                st.markdown("#### Trade Log")
                display_trades = bt["trades_df"].copy()
                display_trades["Price"]   = display_trades["Price"].apply(lambda x: f"${x:.2f}")
                display_trades["Capital"] = display_trades["Capital"].apply(lambda x: f"${x:,.0f}")
                if "P&L" in display_trades.columns:
                    display_trades["P&L"] = display_trades["P&L"].apply(
                        lambda x: f"+${x:.2f}" if pd.notna(x) and x >= 0 else (f"-${abs(x):.2f}" if pd.notna(x) else "-"))
                st.dataframe(display_trades, use_container_width=True, hide_index=True)
                csv_bt = bt["trades_df"].to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download Trade Log", data=csv_bt,
                                   file_name=f"{ticker}_trades.csv", mime="text/csv")

        # ── Confidence Intervals ───────────────────────────────────────────────────
        if show_conf_interval:
            st.subheader("Forecast with Confidence Intervals")
            st.markdown('<div class="ci-badge">95% CI — Bootstrap Resampling</div>', unsafe_allow_html=True)

            with st.spinner(f"Running {ci_bootstrap_n} bootstrap samples..."):
                ci_lower, ci_median, ci_upper = bootstrap_confidence_intervals(
                    model, X_test, n_bootstrap=ci_bootstrap_n, noise_std=0.015)

            fig_ci = go.Figure()
            fig_ci.add_trace(go.Scatter(y=ci_upper, line=dict(color="rgba(0,229,176,0)"),
                showlegend=False))
            fig_ci.add_trace(go.Scatter(y=ci_lower, name="95% CI Band",
                fill="tonexty", fillcolor="rgba(0,229,176,0.10)",
                line=dict(color="rgba(0,229,176,0)")))
            fig_ci.add_trace(go.Scatter(y=actual, name="Actual",
                line=dict(color=C_BLUE, width=1.5)))
            fig_ci.add_trace(go.Scatter(y=ci_median, name="XGBoost Median",
                line=dict(color=C_GREEN, width=1.8, dash="dot")))
            fig_ci.update_layout(**PLOTLY_LAYOUT,
                title=dict(text=f"{ticker} · Predictions with 95% CI", font=dict(color=C_GREEN, size=12)),
                height=380, yaxis_title="Price ($)", xaxis_title="Trading Day (test set)")
            st.plotly_chart(fig_ci, use_container_width=True)

            # Future CI
            future_all = []
            for _ in range(ci_bootstrap_n):
                fp, lrf = [], X[-1].copy()
                for d in range(future_days):
                    nxt = float(model.predict((lrf + np.random.normal(0, 0.015, lrf.shape)).reshape(1,-1))[0])
                    fp.append(nxt)
                    n_tech = len(feature_cols)
                    lrf    = np.concatenate([lrf[:n_tech], np.append(lrf[n_tech+1:], nxt)])
                future_all.append(fp)
            fa = np.array(future_all)
            fut_l, fut_m, fut_u = (np.percentile(fa, 5,  axis=0),
                                   np.percentile(fa, 50, axis=0),
                                   np.percentile(fa, 95, axis=0))

            fig_fci = go.Figure()
            fig_fci.add_trace(go.Scatter(x=list(range(future_days)), y=fut_u,
                line=dict(color="rgba(0,229,176,0)"), showlegend=False))
            fig_fci.add_trace(go.Scatter(x=list(range(future_days)), y=fut_l,
                name="95% CI", fill="tonexty", fillcolor="rgba(0,229,176,0.12)",
                line=dict(color="rgba(0,229,176,0)")))
            fig_fci.add_trace(go.Scatter(x=list(range(future_days)), y=fut_m,
                name="Forecast", mode="lines+markers",
                line=dict(color=C_GREEN, width=2), marker=dict(size=7)))
            fig_fci.add_hline(y=last_close, line_dash="dash", line_color=C_GREY,
                annotation_text=f"Last close ${last_close:.2f}", annotation_font_color=C_GREY)
            fig_fci.update_layout(**PLOTLY_LAYOUT,
                title=dict(text=f"{ticker} · {future_days}-Day Forecast with 95% CI",
                    font=dict(color=C_GREEN, size=12)),
                xaxis_title="Days from today", yaxis_title="Price ($)", height=350)
            st.plotly_chart(fig_fci, use_container_width=True)

        # ── Model Comparison ───────────────────────────────────────────────────────
        if run_model_compare:
            st.subheader("Model Comparison — XGBoost vs Prophet vs Linear Regression")
            from sklearn.linear_model import LinearRegression as LR
            cmp = {}
            cmp["XGBoost"] = {"preds": preds, "color": C_GREEN,
                "rmse": float(np.sqrt(mean_squared_error(actual, preds))),
                "mae":  float(mean_absolute_error(actual, preds)),
                "mape": float(np.mean(np.abs((actual-preds)/actual))*100),
                "r2":   float(1 - np.sum((actual-preds)**2)/np.sum((actual-np.mean(actual))**2))}

            with st.spinner("Training Linear Regression..."):
                lr_m = LR(); lr_m.fit(X_train, y_train); lr_p = lr_m.predict(X_test)
            cmp["Linear Regression"] = {"preds": lr_p, "color": C_GREY,
                "rmse": float(np.sqrt(mean_squared_error(actual, lr_p))),
                "mae":  float(mean_absolute_error(actual, lr_p)),
                "mape": float(np.mean(np.abs((actual-lr_p)/actual))*100),
                "r2":   float(1 - np.sum((actual-lr_p)**2)/np.sum((actual-np.mean(actual))**2))}

            try:
                from prophet import Prophet
                cs_full = df["Close"].squeeze()
                pdf = pd.DataFrame({"ds": df.index[:len(cs_full)], "y": cs_full.values}).dropna()
                ptr = pdf.iloc[:int(len(pdf)*0.8)]; pte = pdf.iloc[int(len(pdf)*0.8):]
                with st.spinner("Training Prophet..."):
                    pm = Prophet(daily_seasonality=False, weekly_seasonality=True,
                                 yearly_seasonality=True, changepoint_prior_scale=0.05)
                    pm.fit(ptr)
                    pfut  = pm.make_future_dataframe(periods=len(pte), freq="B")
                    pfcst = pm.predict(pfut)
                    pp    = pfcst["yhat"].values[-len(pte):]
                    pa    = pte["y"].values; ml = min(len(pp), len(actual))
                    pp, pa = pp[:ml], actual[:ml]
                cmp["Prophet"] = {"preds": pp, "color": C_YELLOW,
                    "rmse": float(np.sqrt(mean_squared_error(pa, pp))),
                    "mae":  float(mean_absolute_error(pa, pp)),
                    "mape": float(np.mean(np.abs((pa-pp)/pa))*100),
                    "r2":   float(1 - np.sum((pa-pp)**2)/np.sum((pa-np.mean(pa))**2))}
            except ImportError:
                st.info("Add `prophet` to requirements.txt to enable Prophet comparison.")

            rows = [{"Model": n, "RMSE ($)": f"${r['rmse']:.2f}", "MAE ($)": f"${r['mae']:.2f}",
                     "MAPE (%)": f"{r['mape']:.2f}%", "R²": f"{r['r2']:.4f}"}
                    for n, r in cmp.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(y=actual, name="Actual", line=dict(color=C_BLUE, width=2)))
            for n, r in cmp.items():
                fig_cmp.add_trace(go.Scatter(y=r["preds"], name=n,
                    line=dict(color=r["color"], width=1.5, dash="dot")))
            fig_cmp.update_layout(**PLOTLY_LAYOUT,
                title=dict(text=f"{ticker} · Model Comparison (Test Set)", font=dict(color=C_GREEN, size=12)),
                height=380, yaxis_title="Price ($)")
            st.plotly_chart(fig_cmp, use_container_width=True)

            names_l = list(cmp.keys()); rmse_l = [r["rmse"] for r in cmp.values()]
            best    = names_l[rmse_l.index(min(rmse_l))]
            fig_b   = go.Figure(go.Bar(x=names_l, y=rmse_l,
                marker_color=[r["color"] for r in cmp.values()],
                text=[f"${v:.2f}" for v in rmse_l], textposition="outside"))
            fig_b.update_layout(**PLOTLY_LAYOUT,
                title=dict(text=f"RMSE Comparison · Lower is Better · Best: {best}",
                    font=dict(color=C_GREEN, size=12)),
                yaxis_title="RMSE ($)", height=300)
            st.plotly_chart(fig_b, use_container_width=True)

        # ── Halal / Shariah ────────────────────────────────────────────────────────
        if run_halal_check:
            st.markdown("""
            <div style="background:rgba(0,229,176,0.03);border:1px solid rgba(0,229,176,0.15);
                 border-left:4px solid #00e5b0;padding:.8rem 1.4rem;margin:1.5rem 0 .5rem 0;
                 display:flex;align-items:center;gap:1rem;">
                <div style="font-size:1.4rem;">☪</div>
                <div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;letter-spacing:.18em;
                         text-transform:uppercase;color:#00e5b0;">Halal / Shariah Compliance Screen</div>
                    <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.78rem;color:#3d5068;margin-top:2px;">
                        Automated screening based on AAOIFI Standard No.21 — a unique feature not found in mainstream platforms
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(
                '<div class="model-badge">Based on AAOIFI Standard No.21 + S&P Shariah Indices Methodology</div>',
                unsafe_allow_html=True)

            with st.spinner(f"Fetching financial data for {ticker}..."):
                sd = get_shariah_data(ticker)

            if sd is None:
                st.warning(
                    f"⚠ Could not fetch detailed financial data for **{ticker}** from Yahoo Finance. "
                    "This may be a temporary API issue. The ticker symbol screening will use "
                    "the known-ticker list only.")
                # Build minimal sd from ticker list screening only
                sd = {
                    "debt_to_mktcap": 0, "debt_to_assets": 0, "cash_to_assets": 0,
                    "market_cap": 0, "total_debt": 0, "total_assets": 0, "total_cash": 0,
                    "sector": "Unknown", "industry": "Unknown", "company_name": ticker,
                }
            if sd is not None:
                cr      = check_shariah_compliance(ticker, sd)
                verdict = cr["verdict"]
                v_color = {"COMPLIANT": C_GREEN, "NON-COMPLIANT": C_RED, "QUESTIONABLE": C_YELLOW}[verdict]
                v_bg    = {"COMPLIANT": "rgba(0,229,176,0.05)", "NON-COMPLIANT": "rgba(255,61,87,0.05)",
                           "QUESTIONABLE": "rgba(255,221,45,0.05)"}[verdict]
                v_icon  = {"COMPLIANT": "✅", "NON-COMPLIANT": "❌", "QUESTIONABLE": "⚠️"}[verdict]

                st.markdown(
                    f'<div style="background:{v_bg};border:1px solid {v_color};border-left:3px solid {v_color};'
                    f'padding:1.2rem 2rem;margin:1rem 0;text-align:center;">'
                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#3d5068;'
                    f'letter-spacing:.15em;text-transform:uppercase;">{sd["company_name"]} ({ticker})</div>'
                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.8rem;font-weight:700;'
                    f'color:{v_color};margin-top:.4rem;">{v_icon}&nbsp;{verdict}</div>'
                    f'<div style="font-size:.78rem;color:#7a8fa8;margin-top:.3rem;">'
                    f'Sector: {sd["sector"]} | Industry: {sd["industry"]}</div>'
                    f'</div>', unsafe_allow_html=True)

                st.markdown("#### Screening Criteria Breakdown")
                cl, cr2 = st.columns(2)
                with cl:
                    bs = cr["business"]
                    if bs["haram_hit"]:
                        st.markdown(f'<div class="halal-card-fail"><b>❌ Business Activity</b><br>'
                                    f'Non-compliant: <b>{bs["haram_hit"]}</b></div>', unsafe_allow_html=True)
                    elif bs["questionable"]:
                        st.markdown('<div class="halal-card" style="border-left-color:#ffdd2d">'
                                    '<b>⚠️ Business Activity</b><br>Questionable sector — consult a scholar</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="halal-card"><b>✅ Business Activity</b><br>'
                                    f'No Haram core business detected<br>'
                                    f'<small style="color:#3d5068">Sector: {sd["sector"]}</small></div>',
                                    unsafe_allow_html=True)
                    dm = cr["debt_mktcap"]
                    cc = "halal-card" if dm["pass"] else "halal-card-fail"
                    st.markdown(f'<div class="{cc}"><b>{"✅" if dm["pass"] else "❌"} Debt / Market Cap</b>'
                                f'<br>{dm["label"]}</div>', unsafe_allow_html=True)
                with cr2:
                    da = cr["debt_assets"]
                    cc = "halal-card" if da["pass"] else "halal-card-fail"
                    st.markdown(f'<div class="{cc}"><b>{"✅" if da["pass"] else "❌"} Debt / Total Assets</b>'
                                f'<br>{da["label"]}</div>', unsafe_allow_html=True)
                    ca = cr["cash_assets"]
                    cc = "halal-card" if ca["pass"] else "halal-card-fail"
                    st.markdown(f'<div class="{cc}"><b>{"✅" if ca["pass"] else "❌"} Cash / Total Assets</b>'
                                f'<br>{ca["label"]}</div>', unsafe_allow_html=True)

                st.markdown("#### Financial Snapshot")
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Market Cap",   f'${sd["market_cap"]/1e9:.2f}B')
                f2.metric("Total Debt",   f'${sd["total_debt"]/1e9:.2f}B')
                f3.metric("Total Assets", f'${sd["total_assets"]/1e9:.2f}B')
                f4.metric("Cash",         f'${sd["total_cash"]/1e9:.2f}B')

                st.markdown(
                    '<div style="background:#0d1117;border:1px solid #182030;padding:.7rem 1.2rem;'
                    'font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#3d5068;margin-top:1rem;">'
                    '⚠ Automated screen based on AAOIFI Standard No.21. '
                    'Does not constitute a fatwa. Consult a qualified Islamic finance scholar for binding rulings.</div>',
                    unsafe_allow_html=True)

        # ── Download ───────────────────────────────────────────────────────────────
        st.subheader("Download Report")
        export_df = df[['Open','High','Low','Close','Volume',
                        'MA50','MA200','RSI','MACD','BB_Upper','BB_Lower','ATR']].copy()
        csv_hist    = export_df.to_csv().encode('utf-8')
        csv_signals = signal_df.to_csv(index=False).encode('utf-8')
        csv_future  = future_df.to_csv(index=False).encode('utf-8')

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button("⬇ Historical Data + Indicators", data=csv_hist,
                               file_name=f"{ticker}_historical.csv", mime="text/csv")
        with dl2:
            st.download_button("⬇ Buy/Sell Signals", data=csv_signals,
                               file_name=f"{ticker}_signals.csv", mime="text/csv")
        with dl3:
            st.download_button("⬇ Forecast Data", data=csv_future,
                               file_name=f"{ticker}_forecast.csv", mime="text/csv")

        # ── Multi-Stock Comparison ─────────────────────────────────────────────
        if compare_tickers:
            st.subheader(f"Multi-Stock Comparison vs {ticker}")
            st.markdown(
                f'<div class="model-badge">🔥 COMPARING: {ticker} + {", ".join(compare_tickers[:4])}'
                f' — Live Composite Signal Ranking</div>',
                unsafe_allow_html=True)

            compare_results = []
            for ct in [ticker] + compare_tickers[:4]:
                try:
                    with st.spinner(f"Fetching {ct}..."):
                        cdf = fetch_data(ct, start_date, end_date)
                    if cdf.empty or len(cdf) < 60:
                        continue
                    cdf = add_technical_features(cdf)
                    cX, cy = build_xgb_dataset(cdf, seq_len)
                    if len(cX) < 30:
                        continue
                    csplit = int(len(cX) * 0.8)
                    cmodel = XGBRegressor(n_estimators=150, max_depth=4,
                                          learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, random_state=42, verbosity=0)
                    cmodel.fit(cX[:csplit], cy[:csplit], verbose=False)
                    cpred_last = float(cmodel.predict(cX[-1].reshape(1, -1))[0])
                    c_last_close = float(cdf['Close'].squeeze().iloc[-1])
                    ccomp = compute_composite_signal(cdf, c_last_close, cpred_last, cmodel.predict(cX), cy)
                    crmse = float(np.sqrt(np.mean((cmodel.predict(cX[csplit:]) - cy[csplit:])**2)))
                    crsi  = float(cdf['RSI'].squeeze().iloc[-1])
                    compare_results.append({
                        'Ticker':        ct,
                        'Last Price':    f"${c_last_close:.2f}",
                        'Signal':        ccomp['verdict'],
                        'Score':         ccomp['total_score'],
                        'Forecast Δ%':   f"{ccomp['xgb_pct']:+.2f}%",
                        'RSI':           f"{crsi:.1f}",
                        'Stop Loss':     f"${ccomp['stop_loss']:.2f}",
                        'Take Profit':   f"${ccomp['take_profit']:.2f}",
                        'Risk:Reward':   f"{ccomp['risk_reward']:.2f}×",
                        'RMSE ($)':      f"${crmse:.2f}",
                    })
                except Exception as e:
                    st.warning(f"Could not process {ct}: {e}")
                    continue

            if compare_results:
                comp_df = pd.DataFrame(compare_results).sort_values('Score', ascending=False)

                # Colour-coded signal column
                def signal_color_html(row):
                    sc = {'STRONG BUY': '#00e5b0', 'BUY': '#00e5b0',
                          'STRONG SELL': '#ff3d57', 'SELL': '#ff3d57'}.get(row['Signal'], '#ffdd2d')
                    return (f'<span style="color:{sc};font-weight:700;font-family:IBM Plex Mono,'
                            f'monospace;font-size:.75rem;">{row["Signal"]}</span>')

                rows_html = ""
                for _, row in comp_df.iterrows():
                    sc = '#00e5b0' if row['Score'] > 0 else '#ff3d57' if row['Score'] < 0 else '#ffdd2d'
                    sig_html = {'STRONG BUY': '<span style="color:#00e5b0;font-weight:800;">⬆ STRONG BUY</span>',
                                'BUY':        '<span style="color:#00e5b0;font-weight:700;">▲ BUY</span>',
                                'STRONG SELL':'<span style="color:#ff3d57;font-weight:800;">⬇ STRONG SELL</span>',
                                'SELL':       '<span style="color:#ff3d57;font-weight:700;">▼ SELL</span>',
                                'HOLD':       '<span style="color:#ffdd2d;font-weight:700;">◆ HOLD</span>'}.get(row['Signal'], row['Signal'])
                    rows_html += f"""
                    <tr style="border-bottom:1px solid #182030;">
                      <td style="padding:.55rem .8rem;font-family:IBM Plex Mono,monospace;font-weight:700;
                          color:#eef2f7;font-size:.85rem;">{row['Ticker']}</td>
                      <td style="padding:.55rem .8rem;font-family:IBM Plex Mono,monospace;color:#7a8fa8;">{row['Last Price']}</td>
                      <td style="padding:.55rem .8rem;">{sig_html}</td>
                      <td style="padding:.55rem .8rem;font-family:IBM Plex Mono,monospace;
                          color:{sc};font-weight:700;">{row['Score']:+.0f}</td>
                      <td style="padding:.55rem .8rem;font-family:IBM Plex Mono,monospace;
                          color:{'#00e5b0' if '+' in row['Forecast Δ%'] else '#ff3d57'};">{row['Forecast Δ%']}</td>
                      <td style="padding:.55rem .8rem;font-family:IBM Plex Mono,monospace;color:#7a8fa8;">{row['RSI']}</td>
                      <td style="padding:.55rem .8rem;font-family:IBM Plex Mono,monospace;color:#ff3d57;">{row['Stop Loss']}</td>
                      <td style="padding:.55rem .8rem;font-family:IBM Plex Mono,monospace;color:#00e5b0;">{row['Take Profit']}</td>
                      <td style="padding:.55rem .8rem;font-family:IBM Plex Mono,monospace;color:#7a8fa8;">{row['Risk:Reward']}</td>
                    </tr>"""

                st.markdown(f"""
                <div style="overflow-x:auto;">
                <table style="width:100%;border-collapse:collapse;background:#0d1117;
                       border:1px solid #182030;font-size:.8rem;">
                  <thead>
                    <tr style="border-bottom:2px solid #243346;">
                      <th style="padding:.5rem .8rem;font-family:IBM Plex Mono,monospace;font-size:.6rem;
                          letter-spacing:.14em;text-transform:uppercase;color:#3d5068;text-align:left;">Ticker</th>
                      <th style="padding:.5rem .8rem;font-family:IBM Plex Mono,monospace;font-size:.6rem;
                          letter-spacing:.14em;text-transform:uppercase;color:#3d5068;text-align:left;">Price</th>
                      <th style="padding:.5rem .8rem;font-family:IBM Plex Mono,monospace;font-size:.6rem;
                          letter-spacing:.14em;text-transform:uppercase;color:#3d5068;text-align:left;">Signal</th>
                      <th style="padding:.5rem .8rem;font-family:IBM Plex Mono,monospace;font-size:.6rem;
                          letter-spacing:.14em;text-transform:uppercase;color:#3d5068;text-align:left;">Score</th>
                      <th style="padding:.5rem .8rem;font-family:IBM Plex Mono,monospace;font-size:.6rem;
                          letter-spacing:.14em;text-transform:uppercase;color:#3d5068;text-align:left;">Forecast Δ</th>
                      <th style="padding:.5rem .8rem;font-family:IBM Plex Mono,monospace;font-size:.6rem;
                          letter-spacing:.14em;text-transform:uppercase;color:#3d5068;text-align:left;">RSI</th>
                      <th style="padding:.5rem .8rem;font-family:IBM Plex Mono,monospace;font-size:.6rem;
                          letter-spacing:.14em;text-transform:uppercase;color:#3d5068;text-align:left;">Stop Loss</th>
                      <th style="padding:.5rem .8rem;font-family:IBM Plex Mono,monospace;font-size:.6rem;
                          letter-spacing:.14em;text-transform:uppercase;color:#3d5068;text-align:left;">Take Profit</th>
                      <th style="padding:.5rem .8rem;font-family:IBM Plex Mono,monospace;font-size:.6rem;
                          letter-spacing:.14em;text-transform:uppercase;color:#3d5068;text-align:left;">R:R</th>
                    </tr>
                  </thead>
                  <tbody>{rows_html}</tbody>
                </table>
                </div>
                """, unsafe_allow_html=True)

                csv_compare = comp_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download Comparison", data=csv_compare,
                                   file_name=f"comparison_{ticker}.csv", mime="text/csv")

        st.markdown("---")
        st.markdown('<div class="stat-row">⚠ For educational purposes only. Not financial advice.</div>',
                    unsafe_allow_html=True)

else:
    land_tab_overview, land_tab_methodology = st.tabs(["🏠  Overview", "📖  Methodology"])

    with land_tab_overview:

        # ── Hero ──────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem 1.5rem 1rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;letter-spacing:.3em;
                 text-transform:uppercase;color:#3d5068;margin-bottom:.7rem;">
                Free · Open-source · No login required
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:2.8rem;font-weight:700;
                 color:#eef2f7;letter-spacing:.05em;line-height:1.15;margin-bottom:.8rem;">
                STOCK<span style="color:#00e5b0;">CAST</span>
            </div>
            <div style="font-family:'IBM Plex Sans',sans-serif;font-size:1.1rem;color:#7a8fa8;
                 max-width:580px;margin:0 auto 1.5rem auto;line-height:1.7;">
                AI-powered stock intelligence with <span style="color:#eef2f7;">explainable signals</span> —
                not just a prediction, but <span style="color:#00e5b0;">why</span> the model thinks it.
            </div>
            <div style="display:flex;justify-content:center;gap:.8rem;flex-wrap:wrap;margin-bottom:2.5rem;">
                <span style="background:rgba(0,229,176,0.1);border:1px solid rgba(0,229,176,0.3);
                      color:#00e5b0;font-family:IBM Plex Mono,monospace;font-size:.65rem;
                      letter-spacing:.1em;padding:.3rem .9rem;">✓ XGBoost ML</span>
                <span style="background:rgba(77,166,255,0.1);border:1px solid rgba(77,166,255,0.3);
                      color:#4da6ff;font-family:IBM Plex Mono,monospace;font-size:.65rem;
                      letter-spacing:.1em;padding:.3rem .9rem;">✓ 6-Factor Signals</span>
                <span style="background:rgba(255,221,45,0.1);border:1px solid rgba(255,221,45,0.3);
                      color:#ffdd2d;font-family:IBM Plex Mono,monospace;font-size:.65rem;
                      letter-spacing:.1em;padding:.3rem .9rem;">✓ Backtesting Engine</span>
                <span style="background:rgba(0,229,176,0.1);border:1px solid rgba(0,229,176,0.3);
                      color:#00e5b0;font-family:IBM Plex Mono,monospace;font-size:.65rem;
                      letter-spacing:.1em;padding:.3rem .9rem;">✓ Watchlist + Alerts</span>
                <span style="background:rgba(255,61,87,0.1);border:1px solid rgba(255,61,87,0.3);
                      color:#ff3d57;font-family:IBM Plex Mono,monospace;font-size:.65rem;
                      letter-spacing:.1em;padding:.3rem .9rem;">✓ Shariah Screening</span>
                <span style="background:rgba(255,221,45,0.1);border:1px solid rgba(255,221,45,0.3);
                      color:#ffdd2d;font-family:IBM Plex Mono,monospace;font-size:.65rem;
                      letter-spacing:.1em;padding:.3rem .9rem;">✓ News Sentiment NLP</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Live Watchlist Preview on landing ────────────────────────────────
        if st.session_state.watchlist:
            st.markdown("#### ⭐ Your Watchlist — Live Prices")
            wl_cols = st.columns(min(len(st.session_state.watchlist), 4))
            for i, wl_sym in enumerate(st.session_state.watchlist[:4]):
                with wl_cols[i % 4]:
                    try:
                        _fi   = yf.Ticker(wl_sym).fast_info
                        _px   = _fi.get("last_price") or _fi.get("regularMarketPrice") or 0
                        _chg  = _fi.get("regularMarketChangePercent") or 0
                        _col  = "#00e5b0" if _chg >= 0 else "#ff3d57"
                        _sign = "▲" if _chg >= 0 else "▼"
                        st.markdown(
                            f'<div style="background:#0d1117;border:1px solid #182030;'
                            f'border-top:2px solid {_col};padding:1rem 1.2rem;text-align:center;">'
                            f'<div style="font-family:IBM Plex Mono,monospace;font-size:.65rem;'
                            f'letter-spacing:.15em;color:#3d5068;text-transform:uppercase;">{wl_sym}</div>'
                            f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.3rem;'
                            f'font-weight:700;color:#eef2f7;margin:.3rem 0;">${_px:.2f}</div>'
                            f'<div style="font-family:IBM Plex Mono,monospace;font-size:.75rem;'
                            f'color:{_col};">{_sign} {_chg:+.2f}%</div></div>',
                            unsafe_allow_html=True
                        )
                    except Exception:
                        st.markdown(
                            f'<div style="background:#0d1117;border:1px solid #182030;'
                            f'padding:1rem 1.2rem;text-align:center;font-family:IBM Plex Mono,monospace;'
                            f'font-size:.7rem;color:#3d5068;">{wl_sym}<br>—</div>',
                            unsafe_allow_html=True
                        )
            st.markdown("---")

        # ── How it works (3 steps) ───────────────────────────────────────────
        st.markdown("#### How It Works")
        hw1, hw2, hw3 = st.columns(3)
        with hw1:
            st.markdown("""
            <div style="background:#0d1117;border:1px solid #182030;border-top:2px solid #00e5b0;
                 padding:1.4rem 1.5rem;height:100%;">
                <div style="font-family:IBM Plex Mono,monospace;font-size:1.4rem;font-weight:700;
                     color:#00e5b0;margin-bottom:.5rem;">01</div>
                <div style="font-family:IBM Plex Mono,monospace;font-size:.65rem;letter-spacing:.15em;
                     text-transform:uppercase;color:#eef2f7;margin-bottom:.5rem;">Enter a Ticker</div>
                <div style="font-family:IBM Plex Sans,sans-serif;font-size:.82rem;color:#7a8fa8;line-height:1.6;">
                    Search by company name or symbol. Add it to your watchlist to track it persistently.
                </div>
            </div>""", unsafe_allow_html=True)
        with hw2:
            st.markdown("""
            <div style="background:#0d1117;border:1px solid #182030;border-top:2px solid #4da6ff;
                 padding:1.4rem 1.5rem;height:100%;">
                <div style="font-family:IBM Plex Mono,monospace;font-size:1.4rem;font-weight:700;
                     color:#4da6ff;margin-bottom:.5rem;">02</div>
                <div style="font-family:IBM Plex Mono,monospace;font-size:.65rem;letter-spacing:.15em;
                     text-transform:uppercase;color:#eef2f7;margin-bottom:.5rem;">Run the Model</div>
                <div style="font-family:IBM Plex Sans,sans-serif;font-size:.82rem;color:#7a8fa8;line-height:1.6;">
                    XGBoost trains on 7 years of OHLCV data with 20 engineered features. Results in seconds.
                </div>
            </div>""", unsafe_allow_html=True)
        with hw3:
            st.markdown("""
            <div style="background:#0d1117;border:1px solid #182030;border-top:2px solid #ffdd2d;
                 padding:1.4rem 1.5rem;height:100%;">
                <div style="font-family:IBM Plex Mono,monospace;font-size:1.4rem;font-weight:700;
                     color:#ffdd2d;margin-bottom:.5rem;">03</div>
                <div style="font-family:IBM Plex Mono,monospace;font-size:.65rem;letter-spacing:.15em;
                     text-transform:uppercase;color:#eef2f7;margin-bottom:.5rem;">Read the Signal</div>
                <div style="font-family:IBM Plex Sans,sans-serif;font-size:.82rem;color:#7a8fa8;line-height:1.6;">
                    Get a BUY / SELL / HOLD verdict with a full explanation of <em>every</em> contributing factor.
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Feature grid ────────────────────────────────────────────────────
        st.markdown("#### What's Inside")
        import streamlit.components.v1 as components
        components.html("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500&display=swap');
        * { box-sizing:border-box; margin:0; padding:0; }
        body { background:transparent; font-family:'IBM Plex Sans',sans-serif; }
        .grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.9rem; }
        .card { background:#0d1117; border:1px solid #182030; padding:1.2rem 1.3rem; }
        .card-title { font-family:'IBM Plex Mono',monospace; font-size:.62rem; letter-spacing:.18em;
                      text-transform:uppercase; margin-bottom:.4rem; }
        .card-body  { font-family:'IBM Plex Sans',sans-serif; font-size:.8rem; color:#7a8fa8; line-height:1.5; }
        </style>
        <div class="grid">
            <div class="card" style="border-top:2px solid #00e5b0;">
                <div class="card-title" style="color:#00e5b0;">📈 XGBoost Forecast</div>
                <div class="card-body">ML trained on 20 technical features. N-day forecast with 95% bootstrap CI.</div>
            </div>
            <div class="card" style="border-top:2px solid #4da6ff;">
                <div class="card-title" style="color:#4da6ff;">⚙ Explainable Signals</div>
                <div class="card-body">RSI, MACD, Bollinger, MA Cross, Volume — grouped, scored, explained in plain English.</div>
            </div>
            <div class="card" style="border-top:2px solid #ffdd2d;">
                <div class="card-title" style="color:#ffdd2d;">📊 Backtesting Engine</div>
                <div class="card-body">Sharpe ratio, max drawdown, win rate, profit factor, equity curve vs buy-and-hold.</div>
            </div>
            <div class="card" style="border-top:2px solid #ff3d57;">
                <div class="card-title" style="color:#ff3d57;">⭐ Watchlist + 🔔 Alerts</div>
                <div class="card-body">Save stocks, see live prices on the dashboard, get banners when signals flip.</div>
            </div>
            <div class="card" style="border-top:2px solid #00e5b0;">
                <div class="card-title" style="color:#00e5b0;">☪ Shariah Screening</div>
                <div class="card-body">AAOIFI Standard No.21 — screens business activity, debt & cash ratios automatically.</div>
            </div>
            <div class="card" style="border-top:2px solid #4da6ff;">
                <div class="card-title" style="color:#4da6ff;">🔬 Model Comparison</div>
                <div class="card-body">Benchmark XGBoost vs Prophet vs Linear Regression — RMSE, MAE, MAPE, R² side-by-side.</div>
            </div>
            <div class="card" style="border-top:2px solid #ffdd2d;">
                <div class="card-title" style="color:#ffdd2d;">📰 News Sentiment NLP</div>
                <div class="card-body">Live Yahoo Finance headlines scored with TextBlob. Detects confluence or divergence with technical signals.</div>
            </div>
        </div>
        <div style="text-align:center;margin-top:1.6rem;font-family:'IBM Plex Mono',monospace;
             font-size:.6rem;color:#182030;letter-spacing:.08em;">
            ⚠ For educational purposes only. Not financial advice.
        </div>
        """, height=310, scrolling=False)

    with land_tab_methodology:
        render_methodology_page()

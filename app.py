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
import nltk
nltk.download('punkt')

# ── Supabase config ────────────────────────────────────────────────────────────
SUPABASE_URL = "https://tpaqlfjszguinigygxcq.supabase.co"
SUPABASE_KEY = "sb_publishable_k3YRnuw5-wzS3VzpCpzQdg_0B-XldpV"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Session state ──────────────────────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "alert_signals" not in st.session_state:
    st.session_state.alert_signals = {}
if "portfolio" not in st.session_state:
    st.session_state.portfolio = [
        {"ticker":"AAPL","name":"Apple Inc.",  "sector":"Technology • Consumer Electronics","qty":142.5,"avg_cost":162.01,"current_price":189.43,"pl":4210.40, "pl_pct":12.4},
        {"ticker":"NVDA","name":"NVIDIA Corp", "sector":"Technology • Semiconductors",      "qty":85.0, "avg_cost":343.65,"current_price":485.12,"pl":12055.20,"pl_pct":42.1},
        {"ticker":"MSFT","name":"Microsoft",   "sector":"Technology • Software",             "qty":62.0, "avg_cost":346.65,"current_price":328.79,"pl":-1104.50,"pl_pct":-4.2},
        {"ticker":"TSLA","name":"Tesla, Inc.", "sector":"Consumer Cyclical • Auto",          "qty":45.0, "avg_cost":179.69,"current_price":242.68,"pl":2840.12, "pl_pct":18.5},
    ]
if "portfolio_history" not in st.session_state:
    st.session_state.portfolio_history = [
        {"date":"Today",     "type":"BUY",     "ticker":"NVDA","shares":12.5,"price":482.10,"amount":-6026.25},
        {"date":"Yesterday", "type":"DIVIDEND","ticker":"AAPL","shares":None,"price":None,  "amount":142.24},
        {"date":"Aug 25",    "type":"SELL",    "ticker":"AMZN","shares":5.0, "price":138.45,"amount":692.25},
    ]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stockcast —  Stock Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Plotly theme ───────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0b1326",
    plot_bgcolor="#0b1326",
    font=dict(family="Manrope", color="#8c909f", size=10),
    xaxis=dict(gridcolor="#424754", linecolor="#424754", tickfont=dict(color="#8c909f", size=10), showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#424754", linecolor="#424754", tickfont=dict(color="#8c909f", size=10), showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#424754", borderwidth=1, font=dict(size=10)),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#131b2e", bordercolor="#2d3449", font=dict(family="Manrope", size=11, color="#dae2fd")),
)

C_GREEN  = "#adc6ff"
C_ACCENT = "#4d8eff"
C_RED    = "#ff6b6b"
C_YELLOW = "#ffdd2d"
C_GREY   = "#8c909f"
C_EMERALD = "#00e5b0"

# ── Master CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@200;300;400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

/* ── ROOT ── */
:root {
    --bg:          #0b1326;
    --bg2:         #131b2e;
    --bg3:         #171f33;
    --bg4:         #222a3d;
    --bg5:         #2d3449;
    --primary:     #adc6ff;
    --accent:      #4d8eff;
    --on-primary:  #002e6a;
    --secondary:   #b1c6f9;
    --t1:          #dae2fd;
    --t2:          #c2c6d6;
    --t3:          #8c909f;
    --t4:          #424754;
    --border:      #2d3449;
    --border2:     #424754;
    --emerald:     #00e5b0;
    --red:         #ff6b6b;
    --yellow:      #ffdd2d;
    --mono:        'IBM Plex Mono', monospace;
    --sans:        'Manrope', sans-serif;
    --radius:      0.5rem;
}

/* ── GLOBAL ── */
html, body, [class*="css"], [data-testid="stApp"],
[data-testid="stAppViewContainer"], .main {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--t1) !important;
}
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }

/* scanline + ambient glow */
[data-testid="stApp"]::before {
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse at 0% 0%, rgba(77,142,255,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 100%, rgba(173,198,255,0.04) 0%, transparent 50%);
    pointer-events: none; z-index: 0;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, var(--bg2) 0%, var(--bg) 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--t3) !important; }
[data-testid="stSidebar"] input {
    background-color: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    color: var(--primary) !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(77,142,255,0.25) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    color: var(--primary) !important;
    border: 1px solid rgba(173,198,255,0.35) !important;
    border-radius: var(--radius) !important;
    font-family: var(--sans) !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: #fff !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 20px rgba(77,142,255,0.3) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── METRICS ── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, var(--bg2), var(--bg3)) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--accent) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 0 20px rgba(77,142,255,0.15) !important;
    transform: translateY(-1px) !important;
    transition: all 0.2s !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--sans) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--t3) !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: var(--primary) !important;
}

/* ── HEADINGS ── */
h2, h3 {
    font-family: var(--sans) !important;
    color: var(--t1) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0.5rem !important;
    margin-top: 1.4rem !important;
    font-weight: 800 !important;
}
h4 {
    font-family: var(--sans) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--t3) !important;
}
hr { border-color: var(--border) !important; }
p, .stMarkdown p { color: var(--t2) !important; font-size: 0.85rem !important; }

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    background: var(--bg2) !important;
    border-radius: var(--radius) !important;
}

/* ── LABELS ── */
label, [data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stTextInput"] label {
    font-family: var(--sans) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--t3) !important;
    font-weight: 700 !important;
}

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg2) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 2px !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--sans) !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    color: var(--t3) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    text-transform: uppercase !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}
[data-testid="stTabPanel"] { background: transparent !important; padding: 1rem 0 !important; }

/* ── APP HEADER ── */
.wi-header {
    background: linear-gradient(90deg, var(--bg2) 0%, var(--bg3) 100%);
    border-bottom: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    padding: 1rem 2rem;
    margin: 3rem -2rem 1.5rem -2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.wi-logo {
    font-family: var(--sans);
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--t1);
    letter-spacing: -0.01em;
}
.wi-logo span { color: var(--accent); }
.wi-sub {
    font-size: 0.65rem;
    color: var(--t3);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 2px;
    font-weight: 600;
}
.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: var(--emerald);
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
    margin-right: 5px;
    vertical-align: middle;
    box-shadow: 0 0 8px rgba(0,229,176,0.5);
}
@keyframes pulse-dot {
    0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(0,229,176,0.4); }
    50%      { opacity:.8; box-shadow: 0 0 0 6px rgba(0,229,176,0); }
}
.live-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--emerald);
    letter-spacing: 0.12em;
    vertical-align: middle;
}

/* ── TICKER TAPE ── */
.ticker-tape-wrap {
    overflow: hidden;
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    border-top: 1px solid var(--border);
    padding: 0.28rem 0;
    margin: 0 -2rem 1.5rem -2rem;
}
.ticker-tape {
    display: inline-flex;
    gap: 2.5rem;
    animation: tape 38s linear infinite;
    white-space: nowrap;
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.05em;
    color: var(--t3);
}
.ticker-tape:hover { animation-play-state: paused; }
@keyframes tape { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
.tape-up   { color: var(--emerald); font-weight: 700; }
.tape-down { color: var(--red); font-weight: 700; }
.tape-sym  { color: var(--t4); font-size: 0.58rem; margin-right: 0.25rem; }

/* ── GLASS CARDS ── */
.wi-card {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    transition: transform 0.18s, box-shadow 0.18s;
}
.wi-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.wi-card-accent { border-top: 2px solid var(--accent); }
.wi-card-emerald { border-top: 2px solid var(--emerald); }
.wi-card-red { border-top: 2px solid var(--red); }
.wi-card-yellow { border-top: 2px solid var(--yellow); }

/* ── SUMMARY STAT GRID ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}
.stat-card {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 50px; height: 50px;
    background: radial-gradient(circle at top right, rgba(77,142,255,0.07), transparent 70%);
}
.stat-label {
    font-family: var(--sans);
    font-size: 0.56rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--t3);
    font-weight: 700;
    margin-bottom: 4px;
}
.stat-value {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--primary);
    line-height: 1.1;
}
.stat-sub { font-size: 0.58rem; color: var(--t3); margin-top: 3px; font-family: var(--sans); }

/* ── SIGNAL PANEL ── */
.signal-panel {
    display: flex;
    gap: 1rem;
    margin: 1.2rem 0;
    flex-wrap: wrap;
}
.signal-main {
    flex: 0 0 240px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem 1.5rem;
    border: 2px solid var(--accent);
    background: rgba(77,142,255,0.05);
    border-radius: var(--radius);
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 30px rgba(77,142,255,0.08);
}
.signal-main::before {
    content: '';
    position: absolute;
    bottom: -20px; right: -20px;
    width: 100px; height: 100px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(77,142,255,0.18) 0%, transparent 70%);
}
.signal-main.sell { border-color: var(--red); background: rgba(255,107,107,0.05); box-shadow: 0 0 30px rgba(255,107,107,0.08); }
.signal-main.sell::before { background: radial-gradient(circle, rgba(255,107,107,0.18) 0%, transparent 70%); }
.signal-main.hold { border-color: var(--yellow); background: rgba(255,221,45,0.05); box-shadow: 0 0 30px rgba(255,221,45,0.08); }
.signal-main.hold::before { background: radial-gradient(circle, rgba(255,221,45,0.18) 0%, transparent 70%); }
.signal-action {
    font-family: var(--mono);
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: 0.2em;
    color: var(--primary);
    line-height: 1;
}
.signal-action.sell { color: var(--red); }
.signal-action.hold { color: var(--yellow); }
.signal-pct {
    font-family: var(--mono);
    font-size: 1rem;
    font-weight: 600;
    margin-top: 0.5rem;
    color: var(--t1);
}
.signal-lbl {
    font-size: 0.54rem;
    letter-spacing: 0.2em;
    color: var(--t3);
    margin-top: 8px;
    text-transform: uppercase;
    font-weight: 700;
    font-family: var(--sans);
}
.signal-details {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    min-width: 200px;
}
.sig-card {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    padding: 0.75rem 1rem;
    position: relative;
    border-radius: var(--radius);
    overflow: hidden;
    transition: transform 0.15s;
}
.sig-card:hover { transform: translateY(-1px); }
.sig-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--border);
}
.sig-card.positive::before { background: var(--emerald); }
.sig-card.negative::before { background: var(--red); }
.sig-card.neutral::before  { background: var(--yellow); }
.sig-lbl { font-size: 0.53rem; letter-spacing: 0.13em; text-transform: uppercase; color: var(--t3); margin-bottom: 4px; font-weight: 700; font-family: var(--sans); }
.sig-val { font-family: var(--mono); font-size: 0.88rem; font-weight: 600; color: var(--t1); }
.sig-sub { font-size: 0.54rem; color: var(--t3); margin-top: 2px; font-family: var(--sans); }

/* ── COMPOSITE METER ── */
.composite-meter {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    padding: 1rem 1.5rem;
    margin: 0.8rem 0;
    border-radius: 0 var(--radius) var(--radius) 0;
}
.meter-title { font-size: 0.56rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--t3); margin-bottom: 0.8rem; font-weight: 700; font-family: var(--sans); }
.sir { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.4rem; font-family: var(--mono); font-size: 0.67rem; }
.sir-label { color: var(--t2); width: 120px; flex-shrink: 0; }
.sir-bar-bg { flex: 1; height: 4px; background: rgba(255,255,255,0.04); border-radius: 2px; overflow: hidden; }
.sir-bar { height: 100%; border-radius: 2px; transition: width 0.6s ease; }
.sir-bar.positive { background: linear-gradient(90deg, var(--emerald), rgba(0,229,176,0.5)); }
.sir-bar.negative { background: linear-gradient(90deg, var(--red), rgba(255,107,107,0.5)); }
.sir-bar.neutral  { background: linear-gradient(90deg, var(--yellow), rgba(255,221,45,0.5)); }
.sir-val { width: 55px; text-align: right; font-weight: 600; color: var(--t1); }
.sir-sig { width: 40px; text-align: right; font-size: 0.57rem; letter-spacing: 0.08em; font-weight: 700; }
.sir-sig.buy { color: var(--emerald); }
.sir-sig.sell { color: var(--red); }
.sir-sig.hold { color: var(--yellow); }

/* ── BT CARDS ── */
.bt-card {
    background: linear-gradient(145deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    border-top: 2px solid var(--border);
    padding: 1rem 1.2rem;
    margin-bottom: 0.4rem;
    font-family: var(--mono);
    border-radius: var(--radius);
}
.bt-label { font-size: 0.58rem; color: var(--t3); letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 4px; font-family: var(--sans); font-weight: 700; }
.bt-val       { font-size: 1.3rem; font-weight: 700; color: var(--t1); }
.bt-val-green { font-size: 1.3rem; font-weight: 700; color: var(--emerald); }
.bt-val-red   { font-size: 1.3rem; font-weight: 700; color: var(--red); }

/* ── HALAL CARDS ── */
.halal-card {
    background: rgba(0,229,176,0.03);
    border: 1px solid rgba(0,229,176,0.15);
    border-left: 3px solid var(--emerald);
    padding: 0.85rem 1.2rem;
    margin: 0.3rem 0;
    font-family: var(--sans);
    font-size: 0.8rem;
    color: var(--t2);
    border-radius: 0 var(--radius) var(--radius) 0;
}
.halal-card-fail {
    background: rgba(255,107,107,0.03);
    border: 1px solid rgba(255,107,107,0.15);
    border-left: 3px solid var(--red);
    padding: 0.85rem 1.2rem;
    margin: 0.3rem 0;
    font-family: var(--sans);
    font-size: 0.8rem;
    color: var(--t2);
    border-radius: 0 var(--radius) var(--radius) 0;
}

/* ── MODEL BADGE ── */
.model-badge {
    display: inline-block;
    background: rgba(77,142,255,0.1);
    border: 1px solid rgba(77,142,255,0.25);
    color: var(--primary);
    font-family: var(--sans);
    font-size: 0.62rem;
    font-weight: 700;
    padding: 0.22rem 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    border-radius: var(--radius);
}

/* ── ALERT BOX ── */
.alert-box {
    background: rgba(77,142,255,0.05);
    border: 1px solid rgba(77,142,255,0.3);
    border-left: 3px solid var(--accent);
    padding: 0.85rem 1.3rem;
    font-family: var(--sans);
    font-size: 0.78rem;
    color: var(--primary);
    margin: 0.8rem 0;
    letter-spacing: 0.03em;
    border-radius: 0 var(--radius) var(--radius) 0;
}

/* ── SIDEBAR STAT ROW ── */
.stat-row {
    font-family: var(--sans);
    font-size: 0.58rem;
    color: var(--t3);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
    margin-top: 2px;
    font-weight: 700;
}

/* ── NAV ITEM ── */
.nav-item-active {
    background: var(--bg4);
    border-left: 3px solid var(--accent);
    color: var(--primary) !important;
    padding: 0.5rem 1rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 2px 0;
    border-radius: 0 var(--radius) var(--radius) 0;
    font-family: var(--sans);
}
.nav-item-idle {
    color: var(--t3);
    padding: 0.5rem 1rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 2px 0;
    font-family: var(--sans);
}

/* ── WATCHLIST BADGE ── */
.wl-badge {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--bg3);
    border: 1px solid var(--border);
    padding: 0.5rem 0.75rem;
    border-radius: var(--radius);
    margin-bottom: 0.3rem;
    font-family: var(--mono);
    font-size: 0.7rem;
    transition: background 0.15s;
}
.wl-badge:hover { background: var(--bg4); }

</style>
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
    q = query.strip().upper()
    results = []
    if q in POPULAR_TICKERS:
        results.append(f"{q} — {POPULAR_TICKERS[q]}")
    ql = query.strip().lower()
    for sym, name in POPULAR_TICKERS.items():
        if sym != q and (ql in name.lower() or ql in sym.lower()):
            results.append(f"{sym} — {name}")
    try:
        res = yf.Search(query, max_results=6)
        for r in res.quotes:
            sym  = r.get("symbol","")
            name = r.get("longname") or r.get("shortname") or sym
            exch = r.get("exchange","")
            qt   = r.get("quoteType","")
            entry = f"{sym} — {name} ({exch})"
            if sym and qt in ("EQUITY","ETF","INDEX") and entry not in results:
                results.append(entry)
    except Exception:
        pass
    return results[:10]

@st.cache_data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=str(start), end=str(end), progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    return 100 - (100 / (1 + gain.rolling(period).mean() / loss.rolling(period).mean()))

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
    return df

FEATURE_COLS = [
    'MA5','MA10','MA20','MA50','EMA12','EMA26',
    'RSI','MACD','MACD_Signal','MACD_Hist',
    'BB_Width','BB_Pct','Returns','Returns_5d','Volatility','Momentum',
    'Volume_Ratio','High_Low_Pct','Close_Open_Pct','ATR'
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
    signals = {}
    xgb_pct = (forecast_price - last_close) / last_close * 100
    if   xgb_pct >  1.5: signals['XGBoost Forecast'] = ('BUY',  min(35, abs(xgb_pct)*6), xgb_pct, 'positive')
    elif xgb_pct < -1.5: signals['XGBoost Forecast'] = ('SELL', -min(35, abs(xgb_pct)*6), xgb_pct, 'negative')
    else:                 signals['XGBoost Forecast'] = ('HOLD', 0, xgb_pct, 'neutral')
    if   rsi < 30: signals['RSI (14)'] = ('BUY',  20, rsi, 'positive')
    elif rsi > 70: signals['RSI (14)'] = ('SELL', -20, rsi, 'negative')
    elif rsi < 45: signals['RSI (14)'] = ('BUY',   8, rsi, 'positive')
    elif rsi > 55: signals['RSI (14)'] = ('SELL',  -8, rsi, 'negative')
    else:          signals['RSI (14)'] = ('HOLD',   0, rsi, 'neutral')
    prev_hist = float(df['MACD_Hist'].squeeze().iloc[-2]) if len(df) > 2 else 0
    if   macd_h > 0 and prev_hist <= 0: signals['MACD Cross'] = ('BUY',  20, macd_h, 'positive')
    elif macd_h < 0 and prev_hist >= 0: signals['MACD Cross'] = ('SELL', -20, macd_h, 'negative')
    elif macd > macd_s:                 signals['MACD Cross'] = ('BUY',  10, macd_h, 'positive')
    elif macd < macd_s:                 signals['MACD Cross'] = ('SELL', -10, macd_h, 'negative')
    else:                               signals['MACD Cross'] = ('HOLD',  0, macd_h, 'neutral')
    if   bb_pct < 0.1: signals['Bollinger %B'] = ('BUY',  10, bb_pct, 'positive')
    elif bb_pct > 0.9: signals['Bollinger %B'] = ('SELL', -10, bb_pct, 'negative')
    else:              signals['Bollinger %B'] = ('HOLD',   0, bb_pct, 'neutral')
    if   ma50 > ma200 and close.iloc[-1] > ma50: signals['MA Cross'] = ('BUY',  15, ma50-ma200, 'positive')
    elif ma50 < ma200 and close.iloc[-1] < ma50: signals['MA Cross'] = ('SELL', -15, ma50-ma200, 'negative')
    else:                                         signals['MA Cross'] = ('HOLD',  0, ma50-ma200, 'neutral')
    if   vol_r > 1.5 and xgb_pct > 0: signals['Volume'] = ('BUY',  10, vol_r, 'positive')
    elif vol_r > 1.5 and xgb_pct < 0: signals['Volume'] = ('SELL', -10, vol_r, 'negative')
    else:                              signals['Volume'] = ('HOLD',   0, vol_r, 'neutral')
    total_score = sum(s[1] for s in signals.values())
    if   total_score >= 25: verdict = "⬆ STRONG BUY";  verdict_short = "BUY"
    elif total_score >= 10: verdict = "↑ BUY";          verdict_short = "BUY"
    elif total_score <= -25:verdict = "⬇ STRONG SELL"; verdict_short = "SELL"
    elif total_score <= -10:verdict = "↓ SELL";         verdict_short = "SELL"
    else:                   verdict = "◆ HOLD";         verdict_short = "HOLD"
    stop_loss   = last_close - 2 * atr
    take_profit = last_close + 3 * atr
    risk_reward = (take_profit - last_close) / max(last_close - stop_loss, 0.01)
    return {
        'signals': signals, 'verdict': verdict, 'verdict_short': verdict_short,
        'total_score': total_score, 'xgb_pct': xgb_pct, 'rsi': rsi,
        'stop_loss': stop_loss, 'take_profit': take_profit, 'risk_reward': risk_reward,
        'vol_ratio': vol_r, 'atr': atr,
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

def bootstrap_confidence_intervals(model, X_input, n_bootstrap=100, noise_std=0.02):
    all_preds = [model.predict(X_input + np.random.normal(0, noise_std, X_input.shape)) for _ in range(n_bootstrap)]
    a = np.array(all_preds)
    return np.percentile(a, 5, axis=0), np.percentile(a, 50, axis=0), np.percentile(a, 95, axis=0)

HARAM_TICKERS = {
    "BUD","STZ","SAM","BREW","ABEV","DEO","BF-B",
    "MO","PM","BTI","LO","VGR",
    "LVS","MGM","WYNN","CZR","PENN","DKNG","BYD",
    "JPM","BAC","WFC","C","GS","MS","AXP",
    "MET","PRU","AIG","ALL","TRV","CB",
    "HRL","TSN","SFD","CAG","LMT","RTX","NOC","GD","HII",
}
QUESTIONABLE_TICKERS = {
    "DIS","NFLX","PARA","WBD","FOXA","SPOT",
    "MAR","HLT","H","IHG","WH","V","MA","AXP","COF","USB","PNC",
}
HARAM_SECTORS_KW = ["bank","insurance","casino","gambling","alcohol","tobacco",
                    "brewing","distill","porn","adult","weapons","defense","firearm"]

@st.cache_data
def get_shariah_data(ticker_sym):
    t = yf.Ticker(ticker_sym); info = {}
    try:
        raw = t.info
        if raw and len(raw) > 5: info = raw
    except Exception: pass
    if not info:
        try:
            fi = t.fast_info
            info = {"marketCap": getattr(fi,"market_cap",0) or 0, "sector":"Unknown",
                    "industry":"Unknown","longName":ticker_sym,"totalDebt":0,"totalAssets":0,"totalCash":0}
        except Exception: pass
    if not info: return None
    def _safe(k, d=0):
        v = info.get(k, d); return v if v is not None else d
    mc = _safe("marketCap", 1) or 1; td = _safe("totalDebt", 0)
    ta = _safe("totalAssets", 1) or 1; tc = _safe("totalCash", 0)
    return {"debt_to_mktcap":td/mc,"debt_to_assets":td/ta,"cash_to_assets":tc/ta,
            "market_cap":mc,"total_debt":td,"total_assets":ta,"total_cash":tc,
            "sector":_safe("sector","Unknown"),"industry":_safe("industry","Unknown"),
            "company_name":_safe("longName",ticker_sym)}

def check_shariah_compliance(ticker_sym, data):
    t = ticker_sym.upper(); ind_lower = data["industry"].lower(); haram_hit = None
    if t in HARAM_TICKERS: haram_hit = "Known non-compliant ticker"
    else:
        for kw in HARAM_SECTORS_KW:
            if kw in ind_lower: haram_hit = data["industry"]; break
    questionable = t in QUESTIONABLE_TICKERS
    r = {
        "business":    {"pass": haram_hit is None, "haram_hit": haram_hit, "questionable": questionable},
        "debt_mktcap": {"pass": data["debt_to_mktcap"] < 0.30, "value": data["debt_to_mktcap"], "label": f"Debt/MarketCap = {data['debt_to_mktcap']*100:.1f}% (< 30%)"},
        "debt_assets": {"pass": data["debt_to_assets"] < 0.33, "value": data["debt_to_assets"], "label": f"Debt/Assets = {data['debt_to_assets']*100:.1f}% (< 33%)"},
        "cash_assets": {"pass": data["cash_to_assets"] < 0.33, "value": data["cash_to_assets"], "label": f"Cash/Assets = {data['cash_to_assets']*100:.1f}% (< 33%)"},
    }
    all_pass = all(r[k]["pass"] for k in ["business","debt_mktcap","debt_assets","cash_assets"])
    r["verdict"] = "NON-COMPLIANT" if not r["business"]["pass"] or not all_pass else ("QUESTIONABLE" if questionable else "COMPLIANT")
    return r

def render_methodology_page(seq_len_val=30, ci_n=100, show_ci=True):
    st.markdown(f"""
    <div style="font-family:Manrope,sans-serif;font-size:0.6rem;letter-spacing:.18em;
         text-transform:uppercase;color:#8c909f;margin-bottom:.3rem;font-weight:700;">Technical Documentation</div>
    <div style="font-family:Manrope,sans-serif;font-size:1.15rem;font-weight:800;
         color:#dae2fd;letter-spacing:-.01em;margin-bottom:1.4rem;">
         WealthIntel <span style="color:#4d8eff;">·</span> Methodology & Model Architecture
    </div>
    """, unsafe_allow_html=True)
    steps = [
        ("01","#4d8eff","Data Ingestion","OHLCV via yfinance",
         "Up to 7 years of daily Open/High/Low/Close/Volume data is fetched from Yahoo Finance. Timezone normalization and MultiIndex flattening are applied for compatibility across yfinance versions."),
        ("02","#adc6ff","Feature Engineering","20 Technical Indicators",
         f"Each trading day is described by 20 derived signals: MA5/10/20/50/200, EMA12/26, RSI(14), MACD(12/26/9) with histogram, Bollinger Band width & %B, ATR(14), Volume Ratio, Momentum, Returns(1d/5d), Volatility(20d), and High-Low%. Additionally, {seq_len_val} lag closes are appended as sequential memory."),
        ("03","#00e5b0","Train/Test Split","80% train · 20% test (chronological)",
         "Data is split strictly chronologically — no shuffling — to prevent look-ahead bias. The model never sees future data during training. Evaluation is performed exclusively on the held-out 20%."),
        ("04","#4d8eff","XGBoost Regressor","Gradient-boosted decision trees",
         "XGBoost is trained to predict the next day's closing price. Hyperparameters (n_estimators, max_depth, learning_rate) are configurable via the sidebar. Subsample=0.8 and colsample_bytree=0.8 provide regularisation."),
        ("05","#adc6ff","Bootstrap CI",f"{ci_n} resampling iterations" if show_ci else "Disabled",
         f"Confidence intervals are produced by running the model {ci_n} times on inputs perturbed with Gaussian noise (σ=1.5%). The 5th and 95th percentiles form the 95% CI ribbon. A wider band indicates higher forecast uncertainty."),
        ("06","#00e5b0","Forward Forecast","Iterative multi-step prediction",
         "Future prices are predicted by rolling: each day's predicted price feeds back as the next day's lag input. Error compounds over time — Days 1–3 are most reliable. Days 6+ are directional signals only."),
        ("07","#ff6b6b","Signal Generation","BUY / SELL / HOLD",
         "A composite 6-factor signal fires from XGBoost forecast, RSI, MACD crossover, Bollinger %B, MA Golden/Death cross, and Volume confirmation. Score >+25 = STRONG BUY, <-25 = STRONG SELL."),
        ("08","#4d8eff","Backtesting Engine","Walk-forward simulation",
         "The backtest replays XGBoost signals on test-set prices: BUY fires when predicted return exceeds threshold, SELL when below. KPIs: Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor, equity curve vs Buy-and-Hold."),
    ]
    for num, color, title, subtitle, body in steps:
        st.markdown(f"""
        <div style="display:flex;gap:1.2rem;margin-bottom:1rem;
             background:#131b2e;border:1px solid #2d3449;border-left:3px solid {color};
             padding:1.1rem 1.4rem;border-radius:0 0.5rem 0.5rem 0;">
          <div style="font-family:IBM Plex Mono,monospace;font-size:1.5rem;font-weight:700;
               color:{color};opacity:.5;min-width:2.5rem;line-height:1.1;">{num}</div>
          <div>
            <div style="font-family:Manrope,sans-serif;font-size:0.7rem;font-weight:800;
                 letter-spacing:.12em;text-transform:uppercase;color:#dae2fd;">{title}</div>
            <div style="font-family:IBM Plex Mono,monospace;font-size:0.6rem;letter-spacing:.1em;
                 color:{color};margin-bottom:.4rem;">{subtitle}</div>
            <div style="font-family:Manrope,sans-serif;font-size:0.82rem;
                 color:#8c909f;line-height:1.6;">{body}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(255,107,107,0.04);border:1px solid rgba(255,107,107,0.2);
         border-left:3px solid #ff6b6b;padding:1rem 1.5rem;margin-top:.5rem;border-radius:0 0.5rem 0.5rem 0;">
      <div style="font-family:Manrope,sans-serif;font-size:0.63rem;letter-spacing:.14em;
           text-transform:uppercase;color:#ff6b6b;margin-bottom:.4rem;font-weight:700;">⚠ Key Limitations</div>
      <div style="font-family:Manrope,sans-serif;font-size:0.82rem;color:#8c909f;line-height:1.7;">
        This model uses <b style="color:#dae2fd;">price and volume data only</b>. It has no awareness of
        earnings releases, macroeconomic events, analyst upgrades, geopolitical news, or central bank decisions.
        A single unexpected event can invalidate any technical forecast.
        <b style="color:#ff6b6b;">This is a research and educational tool — not financial advice.</b>
        Always consult a licensed financial advisor before making investment decisions.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Auth Gate: Login / Signup ──────────────────────────────────────────────────
if st.session_state.user is None:

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], .main {
        background: #0d0e0f !important;
        font-family: 'Inter', sans-serif !important;
        color: #ffffff !important;
    }
    .block-container { padding: 0 2rem !important; max-width: 100% !important; }
    header[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }

    [data-testid="stTextInput"] input {
        background: #000000 !important;
        border: none !important;
        border-radius: 0.125rem !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
        padding: 14px 16px !important;
        transition: box-shadow 0.2s !important;
    }
    [data-testid="stTextInput"] input:focus {
        box-shadow: 0 0 0 1px #3fff8b !important;
        outline: none !important;
    }
    [data-testid="stTextInput"] input::placeholder { color: rgba(117,117,118,0.6) !important; }
    [data-testid="stTextInput"] label { display: none !important; }

    .stButton > button {
        width: 100% !important;
        padding: 1rem !important;
        background: linear-gradient(135deg, #3fff8b 0%, #13ea79 100%) !important;
        color: #005d2c !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        border: none !important;
        border-radius: 0.75rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        box-shadow: 0 0 20px rgba(63,255,139,0.15) !important;
        transition: all 0.18s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 0 28px rgba(63,255,139,0.3) !important;
    }
    .stButton > button:active { transform: scale(0.98) !important; }
    [data-testid="stAlert"] { border-radius: 0.25rem !important; font-size: 0.82rem !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Determine which view: login or signup ─────────────────────────────────
    if "auth_view" not in st.session_state:
        st.session_state.auth_view = "login"

    import streamlit.components.v1 as _ac

    if st.session_state.auth_view == "login":
        # ── LOGIN: Full two-panel layout ──────────────────────────────────────
        left_col, right_col = st.columns([7, 5], gap="small")

        with left_col:
            _ac.html("""
<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"/>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #121315; font-family: 'Inter', sans-serif; color: #fff; }
  .material-symbols-outlined { font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24; }
  .forecast-pulse-texture {
    background-image: radial-gradient(circle at 2px 2px, rgba(63,255,139,0.05) 1px, transparent 0);
    background-size: 24px 24px;
  }
  @keyframes sc-pulse { 0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(63,255,139,0.5);} 50%{opacity:.7;box-shadow:0 0 0 7px rgba(63,255,139,0);} }
  @keyframes bar-rise { from { transform: scaleY(0); } to { transform: scaleY(1); } }
  .chart-bar { transform-origin: bottom; animation: bar-rise 1s ease-out forwards; }
</style>
</head>
<body>
<div class="forecast-pulse-texture" style="min-height:100vh;padding:3rem 3rem;display:flex;flex-direction:column;justify-content:space-between;position:relative;overflow:hidden;">
  <!-- Ambient glows -->
  <div style="position:absolute;top:25%;left:-5rem;width:26rem;height:26rem;background:rgba(63,255,139,0.07);filter:blur(110px);border-radius:50%;pointer-events:none;z-index:0;"></div>
  <div style="position:absolute;bottom:20%;right:-5rem;width:20rem;height:20rem;background:rgba(69,254,201,0.05);filter:blur(90px);border-radius:50%;pointer-events:none;z-index:0;"></div>
  <!-- Decorative trading room background -->
  <div style="position:absolute;inset:0;opacity:0.07;pointer-events:none;z-index:0;overflow:hidden;">
    <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuAtVciMobXdrEkXbJw-yFvYglj9rDCT9oAWQqlRhg8V6vcDVerwYGDMD4bsGnEKE6UMHAIuBQBI7vlO0PCpPr10eRDcYmTNwsE51dklXEdET0kmGMrEKfq1ljaIxlI0HS42CVXPLM41OcX0NyRppI_C9jSxIG5wioUfq8wd5ZIKwm1fSSnQm83J0F66f6upJWmr5-71EkJpQU6b7ORqCGbv49p5NtF67oIgLSYcDZ0ZvR3LNBPxmIU-PnAX2zhC9m8PDqviCj2Od_E"
         alt="" style="width:100%;height:100%;object-fit:cover;" />
  </div>

  <!-- Branding -->
  <nav style="position:relative;z-index:1;display:flex;align-items:center;gap:0.6rem;">
    <span class="material-symbols-outlined" style="color:#3fff8b;font-size:2rem;">insights</span>
    <div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;letter-spacing:-0.02em;color:#3fff8b;line-height:1.1;">Stockcast</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:0.52rem;letter-spacing:0.16em;color:rgba(63,255,139,0.55);text-transform:uppercase;">Developed by Muawwiz Ghani</div>
    </div>
  </nav>

  <!-- Hero copy -->
  <div style="position:relative;z-index:1;margin:auto 0;padding:2rem 0;">
    <h2 style="font-family:'Space Grotesk',sans-serif;font-size:2.6rem;font-weight:500;line-height:1.18;letter-spacing:-0.02em;color:#fff;margin-bottom:1rem;">
      Predicting the pulse<br>of <span style="background:linear-gradient(135deg,#3fff8b,#13ea79);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">global markets.</span>
    </h2>
    <p style="font-family:'Inter',sans-serif;font-size:0.95rem;color:#ababac;line-height:1.75;max-width:400px;margin-bottom:2rem;">
      Access high-frequency forecasting models and neural-driven market analysis in one unified institutional interface.
    </p>
    <!-- Feature badges -->
    <div style="display:flex;flex-direction:column;gap:0.85rem;">
      <div style="display:flex;align-items:center;gap:0.85rem;">
        <div style="width:2.6rem;height:2.6rem;border-radius:4px;background:#1e2021;border:1px solid rgba(71,72,73,0.2);display:flex;align-items:center;justify-content:center;flex-shrink:0;">
          <span class="material-symbols-outlined" style="color:#3fff8b;font-size:1.1rem;">query_stats</span>
        </div>
        <div>
          <p style="font-family:'Space Grotesk',sans-serif;font-size:0.82rem;font-weight:600;color:#fff;margin-bottom:1px;">XGBoost Forecasting Engine</p>
          <p style="font-family:'Inter',sans-serif;font-size:0.68rem;color:#ababac;">6-factor signal · backtesting · bootstrap CI</p>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:0.85rem;">
        <div style="width:2.6rem;height:2.6rem;border-radius:4px;background:#1e2021;border:1px solid rgba(71,72,73,0.2);display:flex;align-items:center;justify-content:center;flex-shrink:0;">
          <span class="material-symbols-outlined" style="color:#3fff8b;font-size:1.1rem;">mosque</span>
        </div>
        <div>
          <p style="font-family:'Space Grotesk',sans-serif;font-size:0.82rem;font-weight:600;color:#fff;margin-bottom:1px;">Shariah Compliance Screening</p>
          <p style="font-family:'Inter',sans-serif;font-size:0.68rem;color:#ababac;">AAOIFI Standard No.21 automated checks</p>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:0.85rem;">
        <div style="width:2.6rem;height:2.6rem;border-radius:4px;background:#1e2021;border:1px solid rgba(71,72,73,0.2);display:flex;align-items:center;justify-content:center;flex-shrink:0;">
          <span class="material-symbols-outlined" style="color:#3fff8b;font-size:1.1rem;">security</span>
        </div>
        <div>
          <p style="font-family:'Space Grotesk',sans-serif;font-size:0.82rem;font-weight:600;color:#fff;margin-bottom:1px;">Institutional Security</p>
          <p style="font-family:'Inter',sans-serif;font-size:0.68rem;color:#ababac;">Supabase Auth · 256-bit encryption</p>
        </div>
      </div>
    </div>
  </div>

  <!-- Live Alpha Stream card (from Stockcast Terminal) -->
  <div style="position:relative;z-index:1;background:rgba(25,26,27,0.75);backdrop-filter:blur(20px);border:1px solid rgba(71,72,73,0.2);border-radius:8px;padding:1.2rem 1.4rem;margin-bottom:1.5rem;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.85rem;">
      <span style="font-family:'Space Grotesk',sans-serif;font-size:0.58rem;letter-spacing:0.18em;text-transform:uppercase;color:#ababac;">Live Alpha Stream</span>
      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#3fff8b;animation:sc-pulse 2s infinite;"></span>
    </div>
    <!-- Abstract bar chart -->
    <div style="height:72px;background:#000;border-radius:4px;overflow:hidden;display:flex;align-items:flex-end;padding:0 8px 6px;gap:5px;">
      <div class="chart-bar" style="flex:1;background:rgba(63,255,139,0.18);border-radius:2px 2px 0 0;height:40%;animation-delay:0.0s;"></div>
      <div class="chart-bar" style="flex:1;background:rgba(63,255,139,0.36);border-radius:2px 2px 0 0;height:58%;animation-delay:0.1s;"></div>
      <div class="chart-bar" style="flex:1;background:rgba(63,255,139,0.54);border-radius:2px 2px 0 0;height:52%;animation-delay:0.2s;"></div>
      <div class="chart-bar" style="flex:1;background:rgba(63,255,139,0.72);border-radius:2px 2px 0 0;height:80%;animation-delay:0.3s;"></div>
      <div class="chart-bar" style="flex:1;background:#3fff8b;border-radius:2px 2px 0 0;height:100%;animation-delay:0.4s;box-shadow:0 0 14px rgba(63,255,139,0.35);"></div>
    </div>
    <div style="display:flex;justify-content:space-between;margin-top:0.75rem;">
      <span style="font-family:'Space Grotesk',sans-serif;font-size:0.65rem;color:#ababac;">Model Confidence</span>
      <span style="font-family:'Space Grotesk',sans-serif;font-size:0.65rem;font-weight:700;color:#3fff8b;">98.4%</span>
    </div>
  </div>

  <!-- Footer links -->
  <footer style="position:relative;z-index:1;display:flex;gap:1.5rem;flex-wrap:wrap;border-top:1px solid rgba(71,72,73,0.15);padding-top:1rem;">
    <a href="#" style="font-family:'Inter',sans-serif;font-size:0.62rem;color:#757576;text-decoration:none;">Terms of Service</a>
    <a href="#" style="font-family:'Inter',sans-serif;font-size:0.62rem;color:#757576;text-decoration:none;">Privacy Policy</a>
    <a href="#" style="font-family:'Inter',sans-serif;font-size:0.62rem;color:#757576;text-decoration:none;">API Documentation</a>
    <span style="font-family:'Inter',sans-serif;font-size:0.62rem;color:#474849;margin-left:auto;">© 2025 Stockcast Terminal</span>
  </footer>
</div>
</body></html>
""", height=780, scrolling=False)

        with right_col:
            # ── Segmented toggle header (decorative — switching handled by button below)
            st.markdown("""
            <div style="background:#191a1b;padding:2.5rem 2.5rem 0;border-left:1px solid rgba(71,72,73,0.1);">
              <!-- Segmented toggle -->
              <div style="display:flex;padding:4px;background:#000;border-radius:12px;border:1px solid rgba(71,72,73,0.12);margin-bottom:2rem;">
                <div style="flex:1;padding:0.7rem 1rem;border-radius:8px;background:linear-gradient(135deg,#3fff8b,#13ea79);text-align:center;
                     font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:0.78rem;color:#005d2c;
                     box-shadow:0 4px 14px rgba(63,255,139,0.2);">Terminal Access</div>
                <div style="flex:1;padding:0.7rem 1rem;text-align:center;font-family:'Space Grotesk',sans-serif;font-weight:700;
                     font-size:0.78rem;color:rgba(171,171,172,0.6);">Create Account</div>
              </div>
              <!-- Welcome copy -->
              <h3 style="font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:700;color:#fff;margin-bottom:0.3rem;">Welcome Back</h3>
              <p style="font-family:'Inter',sans-serif;font-size:0.82rem;color:#ababac;margin-bottom:1.5rem;">Synchronize with the terminal to resume analysis.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<p style="font-family:Space Grotesk,sans-serif;font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;color:#ababac;margin-bottom:4px;">Identity Token (Email)</p>', unsafe_allow_html=True)
            login_email = st.text_input("Email", key="login_email", placeholder="name@firm.com")

            st.markdown('<div style="display:flex;justify-content:space-between;align-items:center;margin-top:1rem;margin-bottom:4px;"><p style="font-family:Space Grotesk,sans-serif;font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;color:#ababac;">Access Key</p><a href="#" style="font-family:Space Grotesk,sans-serif;font-size:0.6rem;color:rgba(63,255,139,0.7);text-decoration:none;text-transform:uppercase;letter-spacing:0.08em;">Forgot Key?</a></div>', unsafe_allow_html=True)
            login_password = st.text_input("Password", key="login_password", type="password", placeholder="••••••••")

            if st.button("AUTHORIZE ACCESS  →", key="login_btn"):
                if login_email and login_password:
                    with st.spinner("Securing session..."):
                        try:
                            res = supabase.auth.sign_in_with_password({"email": login_email, "password": login_password})
                            if res.user:
                                st.session_state.user = res.user
                                st.success("✓ Access granted. Loading dashboard...")
                                st.rerun()
                            else:
                                st.error("⚠ Invalid credentials.")
                        except Exception as e:
                            st.error(f"⚠ Authorization failed: {e}")
                else:
                    st.warning("Please enter your identity token and access key.")

            st.markdown("""
            <div style="position:relative;padding:1.2rem 0;display:flex;align-items:center;gap:1rem;">
              <div style="flex:1;height:1px;background:rgba(71,72,73,0.12);"></div>
              <span style="font-family:'Space Grotesk',sans-serif;font-size:0.55rem;letter-spacing:0.2em;text-transform:uppercase;color:#757576;white-space:nowrap;">Alternative Protocols</span>
              <div style="flex:1;height:1px;background:rgba(71,72,73,0.12);"></div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;margin-bottom:1.5rem;">
              <button style="display:flex;align-items:center;justify-content:center;gap:0.5rem;padding:0.75rem;background:#1e2021;border:1px solid rgba(71,72,73,0.12);border-radius:8px;color:#fff;font-family:'Space Grotesk',sans-serif;font-size:0.72rem;cursor:pointer;">
                <svg width="14" height="14" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
                Google
              </button>
              <button style="display:flex;align-items:center;justify-content:center;gap:0.5rem;padding:0.75rem;background:#1e2021;border:1px solid rgba(71,72,73,0.12);border-radius:8px;color:#fff;font-family:'Space Grotesk',sans-serif;font-size:0.72rem;cursor:pointer;">
                ⌨ SSO
              </button>
            </div>
            <!-- Limited Offer trial banner (from Stockcast Terminal) -->
            <div style="padding:1rem 1.2rem;background:rgba(63,255,139,0.04);border-radius:8px;border:1px solid rgba(63,255,139,0.12);display:flex;align-items:flex-start;gap:0.75rem;margin-bottom:1.5rem;">
              <span class="material-symbols-outlined" style="color:#3fff8b;font-size:1.3rem;flex-shrink:0;font-variation-settings:'FILL' 1;">verified</span>
              <div>
                <p style="font-family:'Space Grotesk',sans-serif;font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#3fff8b;margin-bottom:3px;">Limited Offer</p>
                <p style="font-family:'Inter',sans-serif;font-size:0.72rem;color:#ababac;line-height:1.5;">Start your 14-day trial of Alpha-Tier models. No credit card required for initial calibration.</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Request Alpha Access →", key="goto_signup"):
                st.session_state.auth_view = "signup"
                st.rerun()

            st.markdown('<div style="text-align:center;margin-top:1rem;font-family:Space Grotesk,sans-serif;font-size:0.55rem;color:#474849;letter-spacing:0.1em;text-transform:uppercase;">⚠ For Educational Purposes Only · Not Financial Advice</div>', unsafe_allow_html=True)

    else:
        # ── SIGNUP: Full two-panel layout ─────────────────────────────────────
        left_col, right_col = st.columns([3, 2], gap="small")

        with left_col:
            _ac.html("""
<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"/>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<style>
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:#0d0e0f; font-family:'Inter',sans-serif; color:#fff; }
  .material-symbols-outlined { font-variation-settings:'FILL' 1; vertical-align:middle; }
  .primary-gradient { background:linear-gradient(135deg,#3fff8b 0%,#13ea79 100%); }
  .forecast-pulse-bg { background-image:radial-gradient(circle at 2px 2px,rgba(63,255,139,0.05) 1px,transparent 0); background-size:24px 24px; }
</style>
</head>
<body>
<div class="forecast-pulse-bg" style="min-height:100vh;padding:3.5rem 3rem;display:flex;flex-direction:column;justify-content:center;position:relative;overflow:hidden;">
  <div style="position:absolute;top:25%;left:25%;width:26rem;height:26rem;background:rgba(63,255,139,0.08);filter:blur(110px);border-radius:50%;pointer-events:none;z-index:0;"></div>
  <div style="position:absolute;bottom:25%;right:25%;width:18rem;height:18rem;background:rgba(69,254,201,0.07);filter:blur(90px);border-radius:50%;pointer-events:none;z-index:0;"></div>

  <div style="position:relative;z-index:1;max-width:560px;">
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:3rem;">
      <span style="font-family:'Space Grotesk',sans-serif;font-size:1.6rem;font-weight:700;letter-spacing:-0.02em;color:#3fff8b;">Stockcast</span>
      <span style="font-size:0.58rem;padding:0.2rem 0.55rem;background:#252627;color:#3fff8b;font-family:'Space Grotesk',sans-serif;letter-spacing:0.1em;font-weight:700;text-transform:uppercase;border-radius:2px;">Developed by Muawwiz Ghani</span>
    </div>

    <h1 style="font-family:'Space Grotesk',sans-serif;font-size:3.5rem;font-weight:500;line-height:1.1;letter-spacing:-0.02em;color:#fff;margin-bottom:1.2rem;">
      Democratizing<br><span style="color:#3fff8b;font-style:italic;">Market Intelligence</span>
    </h1>
    <p style="font-family:'Inter',sans-serif;font-size:1rem;color:#ababac;line-height:1.75;max-width:460px;margin-bottom:2.5rem;">
      Access institutional-grade XGBoost forecasting models, real-time signal analysis, and Shariah compliance screening. Join a community of professional traders.
    </p>

    <!-- Forecast Pulse Card -->
    <div style="background:rgba(37,38,39,0.6);backdrop-filter:blur(20px);padding:1.5rem;border-radius:8px;border:1px solid rgba(71,72,73,0.15);max-width:380px;">
      <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:1.2rem;">
        <div>
          <p style="font-family:'Space Grotesk',sans-serif;font-size:0.62rem;letter-spacing:0.12em;text-transform:uppercase;color:#ababac;margin-bottom:0.3rem;">Projected Growth</p>
          <h3 style="font-family:'Space Grotesk',sans-serif;font-size:1.8rem;font-weight:700;color:#3fff8b;">+24.8%</h3>
        </div>
        <span class="material-symbols-outlined" style="color:#3fff8b;font-size:2rem;">trending_up</span>
      </div>
      <div style="height:6rem;display:flex;align-items:flex-end;gap:3px;">
        <div style="flex:1;background:rgba(63,255,139,0.18);height:30%;border-radius:2px 2px 0 0;"></div>
        <div style="flex:1;background:rgba(63,255,139,0.28);height:45%;border-radius:2px 2px 0 0;"></div>
        <div style="flex:1;background:rgba(63,255,139,0.38);height:35%;border-radius:2px 2px 0 0;"></div>
        <div style="flex:1;background:rgba(63,255,139,0.55);height:60%;border-radius:2px 2px 0 0;"></div>
        <div style="flex:1;background:rgba(63,255,139,0.7);height:50%;border-radius:2px 2px 0 0;"></div>
        <div style="flex:1;background:rgba(63,255,139,0.85);height:85%;border-radius:2px 2px 0 0;"></div>
        <div class="primary-gradient" style="flex:1;height:100%;border-radius:2px 2px 0 0;box-shadow:0 0 12px rgba(63,255,139,0.4);"></div>
      </div>
    </div>
  </div>
</div>
</body></html>
""", height=780, scrolling=False)

        with right_col:
            st.markdown("""
            <div style="background:#121315;padding:3rem 2rem 1.5rem;border-left:1px solid rgba(71,72,73,0.1);">
              <div style="margin-bottom:1.8rem;">
                <h2 style="font-family:'Space Grotesk',sans-serif;font-size:1.6rem;font-weight:500;color:#fff;margin-bottom:0.3rem;">Create Account</h2>
                <p style="font-family:'Inter',sans-serif;font-size:0.85rem;color:#ababac;">Start your 14-day premium trial today.</p>
              </div>
              <!-- Social buttons (decorative) -->
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;margin-bottom:1.5rem;">
                <button style="display:flex;align-items:center;justify-content:center;gap:0.5rem;padding:0.75rem;background:#252627;border:none;border-radius:4px;color:#fff;font-family:Space Grotesk,sans-serif;font-size:0.78rem;font-weight:500;cursor:pointer;">
                  <svg width="16" height="16" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
                  Google
                </button>
                <button style="display:flex;align-items:center;justify-content:center;gap:0.5rem;padding:0.75rem;background:#252627;border:none;border-radius:4px;color:#fff;font-family:Space Grotesk,sans-serif;font-size:0.78rem;font-weight:500;cursor:pointer;">
                  ⌨ Terminal ID
                </button>
              </div>
              <div style="position:relative;display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;">
                <div style="flex:1;height:1px;background:rgba(71,72,73,0.2);"></div>
                <span style="font-family:Space Grotesk,sans-serif;font-size:0.58rem;letter-spacing:0.14em;text-transform:uppercase;color:#ababac;white-space:nowrap;">or continue with email</span>
                <div style="flex:1;height:1px;background:rgba(71,72,73,0.2);"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<p style="font-family:Space Grotesk,sans-serif;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#ababac;margin-bottom:4px;">Full Name</p>', unsafe_allow_html=True)
            signup_name = st.text_input("Name", key="signup_name", placeholder="Enter your legal name")

            st.markdown('<p style="font-family:Space Grotesk,sans-serif;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#ababac;margin-top:0.75rem;margin-bottom:4px;">Work Email</p>', unsafe_allow_html=True)
            signup_email = st.text_input("Email", key="signup_email", placeholder="name@company.com")

            st.markdown('<p style="font-family:Space Grotesk,sans-serif;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#ababac;margin-top:0.75rem;margin-bottom:4px;">Access Password</p>', unsafe_allow_html=True)
            signup_password = st.text_input("Password", key="signup_password", type="password", placeholder="••••••••••••")

            st.markdown('<p style="font-family:Space Grotesk,sans-serif;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#ababac;margin-top:0.75rem;margin-bottom:4px;">Confirm Password</p>', unsafe_allow_html=True)
            signup_confirm = st.text_input("Confirm", key="signup_confirm", type="password", placeholder="••••••••••••")

            if st.button("INITIALIZE TERMINAL", key="signup_btn"):
                if signup_email and signup_password and signup_confirm:
                    if signup_password != signup_confirm:
                        st.error("⚠ Passwords do not match.")
                    elif len(signup_password) < 6:
                        st.error("⚠ Password must be at least 6 characters.")
                    else:
                        with st.spinner("Initializing your terminal..."):
                            try:
                                res = supabase.auth.sign_up({"email": signup_email, "password": signup_password})
                                if res.user:
                                    st.success("✓ Account created! Check your email to confirm, then sign in.")
                                else:
                                    st.error("⚠ Sign-up failed. Please try again.")
                            except Exception as e:
                                st.error(f"⚠ Failed: {e}")
                else:
                    st.warning("Please fill in all fields.")

            if st.button("Already an active node? Sign in →", key="goto_login"):
                st.session_state.auth_view = "login"
                st.rerun()

            st.markdown("""
            <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:1.2rem;opacity:0.4;margin-top:2rem;padding-top:1.5rem;border-top:1px solid rgba(71,72,73,0.2);">
              <span style="font-family:Space Grotesk,sans-serif;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#ababac;">© 2024 Stockcast</span>
              <a href="#" style="font-family:Space Grotesk,sans-serif;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#ababac;text-decoration:none;">Privacy</a>
              <a href="#" style="font-family:Space Grotesk,sans-serif;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#ababac;text-decoration:none;">SLA</a>
              <a href="#" style="font-family:Space Grotesk,sans-serif;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#ababac;text-decoration:none;">Support</a>
            </div>
            """, unsafe_allow_html=True)

    st.stop()  # 🚨 Halt — do not render the app until authenticated


# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="wi-header">
  <div>
    Stock<span>cast</span> 
    <div class="wi-sub">XGBoost · 6-Factor Signals · Backtesting · Shariah Screening · News NLP</div>
  </div>
  <div style="display:flex;align-items:center;gap:1.5rem;">
    <div style="text-align:right;">
      <div style="font-size:.55rem;color:#424754;letter-spacing:.14em;text-transform:uppercase;font-weight:700;font-family:Manrope,sans-serif;">Platform</div>
      <div style="font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#8c909f;">Developed by Muawwiz Ghani</div>
    </div>
    <div style="width:1px;height:28px;background:#2d3449;"></div>
    <div>
      <span class="live-dot"></span>
      <span class="live-label">LIVE · NYSE/NASDAQ</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Ticker tape
st.markdown("""
<div class="ticker-tape-wrap">
  <div class="ticker-tape">
    <span><span class="tape-sym">AAPL</span><span class="tape-up">▲ $189.42 +1.2%</span></span>
    <span style="color:#2d3449;">·</span>
    <span><span class="tape-sym">TSLA</span><span class="tape-down">▼ $248.11 -0.8%</span></span>
    <span style="color:#2d3449;">·</span>
    <span><span class="tape-sym">NVDA</span><span class="tape-up">▲ $875.33 +2.1%</span></span>
    <span style="color:#2d3449;">·</span>
    <span><span class="tape-sym">MSFT</span><span class="tape-up">▲ $421.05 +0.5%</span></span>
    <span style="color:#2d3449;">·</span>
    <span><span class="tape-sym">GOOGL</span><span class="tape-down">▼ $168.22 -0.3%</span></span>
    <span style="color:#2d3449;">·</span>
    <span><span class="tape-sym">META</span><span class="tape-up">▲ $512.88 +1.7%</span></span>
    <span style="color:#2d3449;">·</span>
    <span><span class="tape-sym">AMZN</span><span class="tape-up">▲ $186.44 +0.9%</span></span>
    <span style="color:#2d3449;">·</span>
    <span><span class="tape-sym">AMD</span><span class="tape-up">▲ $167.55 +3.2%</span></span>
    <span style="color:#2d3449;">·</span>
    <span><span class="tape-sym">JPM</span><span class="tape-down">▼ $198.30 -0.4%</span></span>
    <span style="color:#2d3449;">·</span>
    <span><span class="tape-sym">SPY</span><span class="tape-up">▲ $521.67 +0.6%</span></span>
    <span style="color:#2d3449;">·</span>
    <span><span class="tape-sym">QQQ</span><span class="tape-up">▲ $448.90 +0.8%</span></span>
    <span style="color:#2d3449;">·</span>
    <span><span class="tape-sym">AAPL</span><span class="tape-up">▲ $189.42 +1.2%</span></span>
    <span style="color:#2d3449;">·</span>
    <span><span class="tape-sym">TSLA</span><span class="tape-down">▼ $248.11 -0.8%</span></span>
    <span style="color:#2d3449;">·</span>
    <span><span class="tape-sym">NVDA</span><span class="tape-up">▲ $875.33 +2.1%</span></span>
    <span style="color:#2d3449;">·</span>
    <span><span class="tape-sym">MSFT</span><span class="tape-up">▲ $421.05 +0.5%</span></span>
    <span style="color:#2d3449;">·</span>
    <span><span class="tape-sym">META</span><span class="tape-up">▲ $512.88 +1.7%</span></span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo + User
    st.markdown(f"""
    <div style="padding:1.4rem 1rem 0.8rem;">
      <div style="font-family:Manrope,sans-serif;font-size:1.4rem;font-weight:800;color:#dae2fd;letter-spacing:-.01em;">
        Stock<span style="color:#4d8eff;">cast</span>
      </div>
      <div style="font-size:.55rem;color:#424754;letter-spacing:.1em;text-transform:uppercase;font-weight:700;margin-top:2px;">
        Developed by Muawwiz Ghani
      </div>
    </div>
    <div style="background:rgba(77,142,255,0.08);border:1px solid rgba(77,142,255,0.2);
         border-left:3px solid #4d8eff;padding:.55rem 1rem;margin:.4rem 0 .6rem;
         font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#adc6ff;letter-spacing:.04em;">
      👤 {st.session_state.user.email}
    </div>
    """, unsafe_allow_html=True)

    if st.button("⏏  Logout", use_container_width=True, key="logout_btn"):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.rerun()

    st.markdown("---")

    # Ticker Search
    st.markdown('<div class="stat-row">🔍 Search Company / Ticker</div>', unsafe_allow_html=True)
    search_query = st.text_input("Search", placeholder="e.g. Apple, TSLA, Saudi Aramco…",
                                 label_visibility="collapsed", key="search_input")
    ticker = "AAPL"
    if search_query and len(search_query.strip()) >= 1:
        search_results = search_tickers(search_query.strip())
        if search_results:
            selected = st.selectbox("Select", search_results, label_visibility="collapsed")
            ticker   = selected.split(" — ")[0].strip()
            st.markdown(f'<div style="background:rgba(77,142,255,0.08);border:1px solid rgba(77,142,255,0.3);border-left:3px solid #4d8eff;padding:.35rem .9rem;font-family:IBM Plex Mono,monospace;font-size:.68rem;color:#adc6ff;letter-spacing:.05em;margin:.3rem 0;border-radius:0 .5rem .5rem 0;">✓ {ticker}</div>', unsafe_allow_html=True)
        else:
            ticker = search_query.strip().upper()
            st.markdown(f'<div style="background:rgba(255,221,45,0.06);border:1px solid rgba(255,221,45,0.3);border-left:3px solid #ffdd2d;padding:.35rem .9rem;font-family:IBM Plex Mono,monospace;font-size:.68rem;color:#ffdd2d;letter-spacing:.05em;margin:.3rem 0;border-radius:0 .5rem .5rem 0;">Using: {ticker} — verify symbol</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stat-row">Or enter ticker directly</div>', unsafe_allow_html=True)
        ticker = st.text_input("Ticker", value="AAPL", placeholder="AAPL, TSLA, MSFT…",
                               label_visibility="collapsed", key="direct_ticker").strip().upper() or "AAPL"
        st.markdown(f'<div style="background:rgba(77,142,255,0.08);border:1px solid rgba(77,142,255,0.2);border-left:3px solid #4d8eff;padding:.35rem .9rem;font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#4d8eff;letter-spacing:.07em;margin:.3rem 0;border-radius:0 .5rem .5rem 0;">● ACTIVE: {ticker}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1: start_date = st.date_input("From", value=pd.to_datetime("2018-01-01"))
    with col2: end_date   = st.date_input("To",   value=pd.Timestamp.today())

    st.markdown('<div class="stat-row">Lookback Window (days)</div>', unsafe_allow_html=True)
    seq_len     = st.slider("", 10, 60, 30, label_visibility="collapsed")
    st.markdown('<div class="stat-row">Forecast Horizon (days)</div>', unsafe_allow_html=True)
    future_days = st.slider(" ", 1, 30, 7, label_visibility="collapsed")

    st.markdown("---")
    ui_mode    = st.radio("Mode", ["🟢 Beginner","🔴 Pro"], index=1, horizontal=True, label_visibility="collapsed")
    is_beginner = (ui_mode == "🟢 Beginner")
    if is_beginner:
        st.markdown('<div style="background:rgba(0,229,176,0.06);border-left:3px solid #00e5b0;padding:.4rem .9rem;font-family:Manrope,sans-serif;font-size:.62rem;color:#00e5b0;font-weight:700;">✓ Simple view active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:rgba(255,107,107,0.06);border-left:3px solid #ff6b6b;padding:.4rem .9rem;font-family:Manrope,sans-serif;font-size:.62rem;color:#ff6b6b;font-weight:700;">⚡ Pro view — all parameters unlocked</div>', unsafe_allow_html=True)

    st.markdown("---")
    fast_mode = st.checkbox("⚡ Fast Mode (skip CI + backtest)", value=is_beginner)

    if not is_beginner:
        st.markdown('<div class="stat-row">XGBoost Hyperparameters</div>', unsafe_allow_html=True)
        n_estimators  = st.slider("Trees", 100, 500, 200, step=50)
        max_depth     = st.slider("Max Depth", 2, 8, 4)
        learning_rate = st.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2], value=0.05)
    else:
        n_estimators = 200; max_depth = 4; learning_rate = 0.05

    st.markdown("---")
    st.markdown('<div class="stat-row">Price Alert Target ($)</div>', unsafe_allow_html=True)
    alert_price = st.number_input("", min_value=0.0, value=0.0, step=1.0, label_visibility="collapsed")

    if not is_beginner:
        st.markdown("---")
        st.markdown('<div class="stat-row">Backtesting</div>', unsafe_allow_html=True)
        run_backtest        = st.checkbox("Enable Backtesting Engine", value=True)
        bt_initial_capital  = st.number_input("Initial Capital ($)", min_value=1000, value=10000, step=1000)
        bt_commission       = st.number_input("Commission per Trade ($)", min_value=0.0, value=1.0, step=0.5)
        bt_signal_threshold = st.slider("Signal Threshold (%)", 0.5, 5.0, 1.0, step=0.5)
    else:
        run_backtest = False; bt_initial_capital = 10000; bt_commission = 1.0; bt_signal_threshold = 1.0

    if not is_beginner:
        st.markdown("---")
        st.markdown('<div class="stat-row">Extra Features</div>', unsafe_allow_html=True)
        run_model_compare  = st.checkbox("Model Comparison (XGB vs LR vs Prophet)", value=False)
        run_halal_check    = st.checkbox("Halal / Shariah Compliance Check", value=True)
        show_conf_interval = st.checkbox("Confidence Intervals on Forecast", value=True) and not fast_mode
        ci_bootstrap_n     = st.slider("Bootstrap Samples (CI)", 50, 300, 100, step=50) if show_conf_interval else 100
    else:
        run_model_compare = False; run_halal_check = True; show_conf_interval = False; ci_bootstrap_n = 100

    if not is_beginner:
        st.markdown("---")
        st.markdown('<div class="stat-row">Multi-Stock Comparison</div>', unsafe_allow_html=True)
        compare_tickers_raw = st.text_input("Compare Tickers", value="", placeholder="e.g. AAPL,TSLA,NVDA",
                                            label_visibility="collapsed", key="compare_input")
        compare_tickers = [t.strip().upper() for t in compare_tickers_raw.split(",") if t.strip()] if compare_tickers_raw.strip() else []
    else:
        compare_tickers = []

    st.markdown("---")
    run_btn = st.button("▶  Run Forecast", use_container_width=True)

    # Watchlist
    st.markdown("---")
    st.markdown("""<div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#dae2fd;margin-bottom:.5rem;">⭐ Watchlist</div>""", unsafe_allow_html=True)
    wl_c1, wl_c2 = st.columns([3,1])
    with wl_c1: add_ticker_input = st.text_input("Add", placeholder="e.g. AAPL", label_visibility="collapsed", key="wl_add").strip().upper()
    with wl_c2: add_clicked = st.button("＋", use_container_width=True, key="wl_add_btn")
    if add_clicked and add_ticker_input:
        if add_ticker_input not in st.session_state.watchlist:
            st.session_state.watchlist.append(add_ticker_input)
    if st.session_state.watchlist:
        for wl_sym in list(st.session_state.watchlist):
            wc1, wc2 = st.columns([3,1])
            with wc1:
                try:
                    _qt   = yf.Ticker(wl_sym).fast_info
                    _px   = _qt.get("last_price") or _qt.get("regularMarketPrice") or 0
                    _chg  = _qt.get("regularMarketChangePercent") or 0
                    _col  = "#00e5b0" if _chg >= 0 else "#ff6b6b"
                    _sign = "▲" if _chg >= 0 else "▼"
                    st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.67rem;padding:.2rem 0;"><span style="color:#424754;">{wl_sym}</span> <span style="color:{_col};">{_sign} ${_px:.2f}</span></div>', unsafe_allow_html=True)
                except Exception:
                    st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.67rem;color:#424754;padding:.2rem 0;">{wl_sym}</div>', unsafe_allow_html=True)
            with wc2:
                if st.button("✕", key=f"wl_del_{wl_sym}", use_container_width=True):
                    st.session_state.watchlist.remove(wl_sym)
                    if wl_sym in st.session_state.alert_signals: del st.session_state.alert_signals[wl_sym]
                    st.rerun()
    else:
        st.markdown('<div style="font-family:Manrope,sans-serif;font-size:.65rem;color:#2d3449;padding:.3rem 0;">No stocks saved yet.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#dae2fd;margin-bottom:.5rem;">🔔 Signal Alerts</div>', unsafe_allow_html=True)
    alert_on_signal_change = st.checkbox("Alert when signal changes", value=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN CONTENT — Landing or Analysis
# ═══════════════════════════════════════════════════════════════════
if not run_btn:
    # ── Landing Dashboard ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-family:Manrope,sans-serif;font-size:2.2rem;font-weight:800;letter-spacing:-.02em;color:#dae2fd;margin-bottom:.3rem;">
      Dashboard <span style="color:#4d8eff;">Overview</span>
    </div>
    <div style="font-size:.82rem;color:#8c909f;margin-bottom:1.5rem;font-weight:500;">
      AI-powered stock intelligence · Enter a ticker in the sidebar and click Run Forecast to begin.
    </div>
    """, unsafe_allow_html=True)

    # Market summary cards
    st.markdown("""
    <div class="stat-grid" style="grid-template-columns:repeat(4,1fr);">
      <div class="stat-card">
        <div class="stat-label">S&amp;P 500</div>
        <div class="stat-value" style="font-size:1.3rem;">5,137.08</div>
        <div class="stat-sub" style="color:#00e5b0;font-weight:700;">▲ +1.24%</div>
      </div>
      <div class="stat-card" style="border-top-color:#adc6ff;">
        <div class="stat-label">NASDAQ 100</div>
        <div class="stat-value" style="font-size:1.3rem;color:#adc6ff;">18,302</div>
        <div class="stat-sub" style="color:#00e5b0;font-weight:700;">▲ +2.10%</div>
      </div>
      <div class="stat-card" style="border-top-color:#ffdd2d;">
        <div class="stat-label">Fear &amp; Greed</div>
        <div class="stat-value" style="font-size:1.3rem;color:#ffdd2d;">74</div>
        <div class="stat-sub">Greed territory</div>
      </div>
      <div class="stat-card" style="border-top-color:#00e5b0;">
        <div class="stat-label">VIX</div>
        <div class="stat-value" style="font-size:1.3rem;color:#00e5b0;">14.2</div>
        <div class="stat-sub">Low volatility</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Watchlist live prices if any
    if st.session_state.watchlist:
        st.subheader("⭐ Watchlist — Live Prices")
        wl_cols = st.columns(min(len(st.session_state.watchlist), 4))
        for i, wl_sym in enumerate(st.session_state.watchlist[:4]):
            with wl_cols[i % 4]:
                try:
                    _fi   = yf.Ticker(wl_sym).fast_info
                    _px   = _fi.get("last_price") or _fi.get("regularMarketPrice") or 0
                    _chg  = _fi.get("regularMarketChangePercent") or 0
                    _col  = "#00e5b0" if _chg >= 0 else "#ff6b6b"
                    _sign = "▲" if _chg >= 0 else "▼"
                    st.markdown(f"""
                    <div style="background:linear-gradient(145deg,#131b2e,#171f33);border:1px solid #2d3449;
                         border-top:2px solid {_col};padding:1rem 1.2rem;text-align:center;border-radius:.5rem;">
                      <div style="font-family:IBM Plex Mono,monospace;font-size:.62rem;letter-spacing:.14em;color:#424754;text-transform:uppercase;">{wl_sym}</div>
                      <div style="font-family:IBM Plex Mono,monospace;font-size:1.3rem;font-weight:700;color:#dae2fd;margin:.3rem 0;">${_px:.2f}</div>
                      <div style="font-family:IBM Plex Mono,monospace;font-size:.72rem;color:{_col};">{_sign} {_chg:+.2f}%</div>
                    </div>""", unsafe_allow_html=True)
                except Exception:
                    st.markdown(f'<div style="background:#131b2e;border:1px solid #2d3449;padding:1rem;text-align:center;font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#424754;border-radius:.5rem;">{wl_sym}<br>—</div>', unsafe_allow_html=True)

    # How it works
    st.subheader("How It Works")
    hw1, hw2, hw3 = st.columns(3)
    for col, num, color, title, body in [
        (hw1,"01","#4d8eff","Enter a Ticker","Search by company name or symbol. Add it to your watchlist to track it persistently."),
        (hw2,"02","#00e5b0","Run the Model","XGBoost trains on 7 years of OHLCV data with 20 engineered features. Results in seconds."),
        (hw3,"03","#ffdd2d","Read the Signal","Get a BUY / SELL / HOLD verdict with a full explanation of every contributing factor."),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,#131b2e,#171f33);border:1px solid #2d3449;
                 border-top:2px solid {color};padding:1.4rem 1.5rem;height:100%;border-radius:.5rem;">
              <div style="font-family:IBM Plex Mono,monospace;font-size:1.3rem;font-weight:700;color:{color};margin-bottom:.5rem;">{num}</div>
              <div style="font-family:Manrope,sans-serif;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#dae2fd;font-weight:700;margin-bottom:.5rem;">{title}</div>
              <div style="font-family:Manrope,sans-serif;font-size:.8rem;color:#8c909f;line-height:1.6;">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.subheader("Platform Features")
    feat_grid = [
        ("#4d8eff","📈 XGBoost Forecast","ML trained on 20 technical features. N-day forecast with 95% bootstrap CI."),
        ("#00e5b0","⚙ Explainable Signals","RSI, MACD, Bollinger, MA Cross, Volume — grouped, scored, explained in plain English."),
        ("#ffdd2d","📊 Backtesting Engine","Sharpe ratio, max drawdown, win rate, profit factor, equity curve vs buy-and-hold."),
        ("#ff6b6b","⭐ Watchlist + 🔔 Alerts","Save stocks, see live prices on the dashboard, get banners when signals flip."),
        ("#4d8eff","☪ Shariah Screening","AAOIFI Standard No.21 — screens business activity, debt & cash ratios automatically."),
        ("#adc6ff","🔬 Model Comparison","Benchmark XGBoost vs Prophet vs Linear Regression — RMSE, MAE, MAPE, R² side-by-side."),
        ("#00e5b0","📰 News Sentiment NLP","Live Yahoo Finance headlines scored with TextBlob. Detects confluence with technical signals."),
        ("#ffdd2d","🏦 Portfolio Tracker","Track holdings, P&L, sector allocation, and recent transaction history."),
    ]
    cols4 = st.columns(4)
    for i, (color, title, body) in enumerate(feat_grid):
        with cols4[i % 4]:
            st.markdown(f"""
            <div style="background:#131b2e;border:1px solid #2d3449;border-top:2px solid {color};
                 padding:1.1rem 1.2rem;margin-bottom:.6rem;border-radius:.5rem;">
              <div style="font-family:Manrope,sans-serif;font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:{color};font-weight:700;margin-bottom:.4rem;">{title}</div>
              <div style="font-family:Manrope,sans-serif;font-size:.78rem;color:#8c909f;line-height:1.5;">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="text-align:center;margin-top:2rem;font-family:IBM Plex Mono,monospace;font-size:.58rem;color:#2d3449;letter-spacing:.08em;"> SupportTeam :- ghani24by7@gmail.com </div>', unsafe_allow_html=True)

else:
    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS ENGINE
    # ═══════════════════════════════════════════════════════════════
    with st.spinner(f"Fetching {ticker} data..."):
        df = fetch_data(ticker, start_date, end_date)

    if df.empty:
        st.error(f"No data found for '{ticker}'. Please check the symbol.")
        st.stop()

    st.success(f"✓ {len(df)} trading days loaded for {ticker}")

    st.markdown(f"""
    <div style="background:rgba(255,221,45,0.04);border:1px solid rgba(255,221,45,0.3);
         border-left:4px solid #ffdd2d;padding:.9rem 1.4rem;margin:.5rem 0 1rem;border-radius:0 .5rem .5rem 0;">
      <div style="font-family:Manrope,sans-serif;font-size:.6rem;letter-spacing:.14em;text-transform:uppercase;color:#ffdd2d;margin-bottom:.3rem;font-weight:700;">
        ⚠ Model Reality Check — Read Before Trading
      </div>
      <div style="font-family:Manrope,sans-serif;font-size:.78rem;color:#8c909f;line-height:1.6;">
        This model uses <b style="color:#dae2fd;">price &amp; volume data only</b>.
        It has <b style="color:#ff6b6b;">zero awareness</b> of:
        &nbsp;📰 breaking news &nbsp;·&nbsp;📊 earnings releases &nbsp;·&nbsp;🏦 Fed/macro events &nbsp;·&nbsp;
        🧠 analyst upgrades &nbsp;·&nbsp;🌍 geopolitical events.
        <b style="color:#ffdd2d;">Use signals as one input — never as sole decision.</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_analysis, tab_methodology = st.tabs(["📊  Analysis", "📖  Methodology"])

    with tab_methodology:
        render_methodology_page(seq_len_val=seq_len, ci_n=ci_bootstrap_n, show_ci=show_conf_interval)

    with tab_analysis:
        with st.spinner("Engineering technical features..."):
            df = add_technical_features(df)
        close_series = df['Close'].squeeze()

        # ── Candlestick Chart ──────────────────────────────────────────────────
        st.subheader("Price Chart")
        fig_candle = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.02)
        fig_candle.add_trace(go.Candlestick(x=df.index,
            open=df['Open'].squeeze(), high=df['High'].squeeze(),
            low=df['Low'].squeeze(), close=close_series,
            name="Price", increasing_line_color=C_EMERALD, decreasing_line_color=C_RED), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['MA50'].squeeze(), name="MA50", line=dict(color=C_YELLOW, width=1.2)), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['MA200'].squeeze(), name="MA200", line=dict(color=C_ACCENT, width=1.2)), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'].squeeze(), name="BB Upper", line=dict(color=C_GREY, width=0.8, dash='dot')), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'].squeeze(), name="BB Lower", line=dict(color=C_GREY, width=0.8, dash='dot'), fill='tonexty', fillcolor='rgba(77,142,255,0.05)'), row=1, col=1)
        colors_vol = [C_EMERALD if c >= o else C_RED for c, o in zip(close_series, df['Open'].squeeze())]
        fig_candle.add_trace(go.Bar(x=df.index, y=df['Volume'].squeeze(), name="Volume", marker_color=colors_vol, opacity=0.5), row=2, col=1)
        candle_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")}
        fig_candle.update_layout(**candle_layout,
            title=dict(text=f"{ticker} · Candlestick · MA50/200 · Bollinger · Volume", font=dict(color=C_GREEN, size=12)),
            xaxis_rangeslider_visible=False, height=560)
        fig_candle.update_xaxes(gridcolor="#2d3449", linecolor="#2d3449", tickfont=dict(color=C_GREY))
        fig_candle.update_yaxes(gridcolor="#2d3449", linecolor="#2d3449", tickfont=dict(color=C_GREY))
        st.plotly_chart(fig_candle, use_container_width=True)

        # ── RSI + MACD ──────────────────────────────────────────────────────────
        st.subheader("Technical Indicators")
        fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08,
                                 subplot_titles=["RSI (14)", "MACD (12/26/9)"])
        fig_tech.add_trace(go.Scatter(x=df.index, y=df['RSI'].squeeze(), name="RSI", line=dict(color=C_ACCENT, width=1.5)), row=1, col=1)
        fig_tech.add_hline(y=70, line_dash="dash", line_color=C_RED,    row=1, col=1)
        fig_tech.add_hline(y=30, line_dash="dash", line_color=C_EMERALD, row=1, col=1)
        fig_tech.add_hrect(y0=70, y1=100, fillcolor="rgba(255,107,107,0.04)", line_width=0, row=1, col=1)
        fig_tech.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,229,176,0.04)",  line_width=0, row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=df.index, y=df['MACD'].squeeze(), name="MACD", line=dict(color=C_ACCENT, width=1.2)), row=2, col=1)
        fig_tech.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'].squeeze(), name="Signal", line=dict(color=C_GREEN, width=1.2)), row=2, col=1)
        macd_hist   = df['MACD_Hist'].squeeze()
        hist_colors = [C_EMERALD if v >= 0 else C_RED for v in macd_hist]
        fig_tech.add_trace(go.Bar(x=df.index, y=macd_hist, name="Histogram", marker_color=hist_colors, opacity=0.65), row=2, col=1)
        subplot_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis','yaxis')}
        fig_tech.update_layout(**subplot_layout, height=450)
        fig_tech.update_xaxes(gridcolor="#2d3449", linecolor="#2d3449", tickfont=dict(color=C_GREY))
        fig_tech.update_yaxes(gridcolor="#2d3449", linecolor="#2d3449", tickfont=dict(color=C_GREY))
        fig_tech.update_yaxes(range=[0, 100], row=1, col=1)
        st.plotly_chart(fig_tech, use_container_width=True)

        # ── XGBoost Model ──────────────────────────────────────────────────────
        st.markdown('<div class="model-badge">🤖 MODEL: XGBoost Regressor · 20 Technical Features + Lag Window</div>', unsafe_allow_html=True)

        with st.expander("📖 How this model works — methodology & limitations", expanded=False):
            st.markdown(f"""<div style="font-family:Manrope,sans-serif;font-size:0.82rem;color:#8c909f;line-height:1.7;">
            <b style="color:#dae2fd;font-family:IBM Plex Mono,monospace;font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;">Feature Engineering</b><br>
            Each trading day is represented by <b style="color:#4d8eff;">20 technical indicators</b> computed from raw OHLCV data — MAs (5–200), EMA12/26, RSI, MACD, Bollinger Bands, ATR, volume ratio, momentum — plus <b style="color:#4d8eff;">{seq_len} lag closes</b> as sequential context.<br><br>
            <b style="color:#dae2fd;font-family:IBM Plex Mono,monospace;font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;">Training & Evaluation</b><br>
            Data is split <b style="color:#4d8eff;">80% train / 20% test</b> chronologically (no data leakage). XGBoost predicts the next day's closing price. Quality is measured with RMSE, MAE, MAPE and R².<br><br>
            <b style="color:#ff6b6b;font-family:IBM Plex Mono,monospace;font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;">⚠ Key Limitations</b><br>
            This model uses price and volume data only. It has no awareness of earnings, macro events, or news. A single unexpected event can invalidate any technical forecast. <b style="color:#ff6b6b;">Not financial advice.</b>
            </div>""", unsafe_allow_html=True)

        with st.spinner("Building feature matrix..."):
            X, y = build_xgb_dataset(df, seq_len)

        if len(X) < 50:
            st.error("Not enough data to train. Try a longer date range or smaller lookback window.")
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

        with st.spinner("Training XGBoost model (cached after first run)..."):
            model = train_xgb_cached(X_train, y_train, X_test, y_test, n_estimators, max_depth, learning_rate)

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

        # ── Model Performance ──────────────────────────────────────────────────
        st.subheader("Model Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE",  f"${rmse:.2f}")
        c2.metric("MAE",   f"${mae:.2f}")
        c3.metric("MAPE",  f"{mape:.2f}%")
        c4.metric("R²",    f"{r2:.4f}")
        mape_label = ("🟢 Excellent" if mape<2 else "🟡 Good" if mape<5 else "🟠 Fair" if mape<10 else "🔴 Poor")
        r2_label   = ("🟢 Excellent" if r2>0.95 else "🟡 Good" if r2>0.85 else "🟠 Fair" if r2>0.70 else "🔴 Poor")
        st.markdown(f'<div style="background:#131b2e;border:1px solid #2d3449;padding:.65rem 1.2rem;font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#424754;display:flex;gap:2rem;flex-wrap:wrap;border-radius:.5rem;"><span>MAPE: {mape_label} · &lt;2% excellent · &lt;5% good · &lt;10% fair</span><span>R²: {r2_label} · &gt;0.95 excellent · &gt;0.85 good · &gt;0.70 fair</span></div>', unsafe_allow_html=True)

        # Tabs
        dash_tab, port_tab, mkt_tab, deep_tab = st.tabs(["🖥  Dashboard", "💼  Portfolio", "🌍  Markets", "📈  Deep Analysis"])

        # ──────────────────────────────────────────────────────────────────────
        with dash_tab:
            _dash_close = float(df["Close"].squeeze().iloc[-1])
            _dash_prev  = float(df["Close"].squeeze().iloc[-2]) if len(df)>1 else _dash_close
            _dash_chg   = _dash_close - _dash_prev
            _dash_pct   = (_dash_chg / _dash_prev * 100) if _dash_prev != 0 else 0
            _dash_sign  = "+" if _dash_chg >= 0 else ""
            _dash_color = "#00e5b0" if _dash_chg >= 0 else "#ff6b6b"
            _dash_arrow = "▲" if _dash_chg >= 0 else "▼"
            _dash_name  = POPULAR_TICKERS.get(ticker, ticker)

            # KPI row
            st.markdown(f"""
            <div class="stat-grid">
              <div class="stat-card">
                <div class="stat-label">Last Close</div>
                <div class="stat-value">${_dash_close:.2f}</div>
                <div class="stat-sub" style="color:{_dash_color};font-weight:700;">{_dash_arrow} {_dash_sign}{_dash_chg:.2f} ({_dash_sign}{_dash_pct:.2f}%)</div>
              </div>
              <div class="stat-card" style="border-top-color:#adc6ff;">
                <div class="stat-label">Model Confidence</div>
                <div class="stat-value" style="color:#adc6ff;">{confidence_score:.0f}<span style="font-size:.9rem;color:#8c909f;">/100</span></div>
                <div class="stat-sub">{"High" if confidence_score>=80 else "Moderate" if confidence_score>=60 else "Low"}</div>
              </div>
              <div class="stat-card" style="border-top-color:#ffdd2d;">
                <div class="stat-label">MAPE</div>
                <div class="stat-value" style="color:#ffdd2d;">{mape:.2f}%</div>
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
            st.subheader("Actual vs Predicted")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(y=actual, name="Actual", line=dict(color=C_ACCENT, width=1.5), fill='tozeroy', fillcolor='rgba(77,142,255,0.05)'))
            fig1.add_trace(go.Scatter(y=preds, name="XGBoost Predicted", line=dict(color=C_EMERALD, width=1.5, dash='dot')))
            fig1.update_layout(**PLOTLY_LAYOUT, title=dict(text=f"{ticker} · XGBoost Model Fit (Test Set)", font=dict(color=C_GREEN, size=12)), height=350)
            st.plotly_chart(fig1, use_container_width=True)

            # Feature Importance
            st.subheader("Feature Importance")
            lag_names = [f'Lag_{i+1}' for i in range(seq_len)]
            all_feature_names = FEATURE_COLS + lag_names
            importances = model.feature_importances_
            imp_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values('importance', ascending=True).tail(20)
            fig_imp = go.Figure(go.Bar(x=imp_df['importance'], y=imp_df['feature'], orientation='h',
                marker=dict(color=imp_df['importance'], colorscale=[[0,"#131b2e"],[0.5,"#1a3050"],[1,C_ACCENT]], showscale=False)))
            fig_imp.update_layout(**PLOTLY_LAYOUT, title=dict(text="Top 20 Feature Importances", font=dict(color=C_GREEN, size=12)), height=430, xaxis_title="Importance Score")
            st.plotly_chart(fig_imp, use_container_width=True)

        # ──────────────────────────────────────────────────────────────────────
        with port_tab:
            port = st.session_state.portfolio
            hist = st.session_state.portfolio_history
            total_value    = sum(h["qty"] * h["current_price"] for h in port)
            total_invested = sum(h["qty"] * h["avg_cost"]       for h in port)
            total_pl       = total_value - total_invested
            total_pl_pct   = (total_pl / total_invested * 100) if total_invested > 0 else 0
            day_pl         = sum(h["pl"] for h in port if h["pl"] > 0) * 0.1

            st.markdown(f"""
            <div style="margin-bottom:1.2rem;">
              <div style="font-family:Manrope,sans-serif;font-size:2rem;font-weight:800;letter-spacing:-.02em;color:#dae2fd;">Portfolio <span style="color:#4d8eff;">Overview</span></div>
              <div style="font-size:.78rem;color:#8c909f;margin-top:.2rem;">Performance summary · Managed Assets</div>
            </div>
            """, unsafe_allow_html=True)

            # Portfolio KPIs
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Total Value",    f"${total_value:,.2f}")
            p2.metric("Total P&L",      f"${total_pl:+,.2f}", delta=f"{total_pl_pct:+.1f}%")
            p3.metric("Day's Gain",     f"${day_pl:+,.2f}")
            p4.metric("Holdings",       str(len(port)))

            # Holdings table
            st.subheader("Holdings")
            holdings_data = []
            for h in port:
                mktval = h["qty"] * h["current_price"]
                pl_sign = "+" if h["pl"] >= 0 else ""
                holdings_data.append({
                    "Ticker": h["ticker"], "Name": h["name"], "Qty": f"{h['qty']:.1f}",
                    "Avg Cost": f"${h['avg_cost']:.2f}", "Current": f"${h['current_price']:.2f}",
                    "Mkt Value": f"${mktval:,.0f}", "P&L": f"{pl_sign}${abs(h['pl']):,.2f}",
                    "P&L %": f"{pl_sign}{abs(h['pl_pct']):.1f}%"
                })
            st.dataframe(pd.DataFrame(holdings_data), use_container_width=True, hide_index=True)

            # Sector donut + recent activity
            sc1, sc2 = st.columns([1, 1])
            with sc1:
                st.subheader("Sector Allocation")
                sector_map = {}
                for h in port:
                    sec = h["sector"].split(" •")[0].strip()
                    sector_map[sec] = sector_map.get(sec, 0) + h["qty"] * h["current_price"]
                sec_colors = {"Technology":"#4d8eff","Consumer Cyclical":"#ffdd2d","Finance":"#adc6ff","Others":"#8c909f"}
                fig_sector = go.Figure(go.Pie(
                    labels=list(sector_map.keys()), values=list(sector_map.values()),
                    hole=0.6, marker_colors=[sec_colors.get(s,"#8c909f") for s in sector_map.keys()],
                    textfont_size=10, textfont_color="#dae2fd",
                ))
                fig_sector.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=True,
                    annotations=[dict(text=f"{len(sector_map)}<br><span style='font-size:10px'>Sectors</span>", x=0.5, y=0.5, font_size=20, showarrow=False, font_color="#dae2fd")])
                st.plotly_chart(fig_sector, use_container_width=True)

            with sc2:
                st.subheader("Recent Activity")
                for a in hist:
                    type_color = {"BUY":"#4d8eff","SELL":"#00e5b0","DIVIDEND":"#ffdd2d"}.get(a["type"],"#8c909f")
                    amt_str = f'+${a["amount"]:,.2f}' if a["amount"] >= 0 else f'-${abs(a["amount"]):,.2f}'
                    desc = f'{a["shares"]} shares @ ${a["price"]:.2f}' if a.get("shares") and a.get("price") else a["ticker"]
                    st.markdown(f"""
                    <div style="display:flex;gap:.8rem;padding:.7rem 0;border-bottom:1px solid #2d3449;align-items:center;">
                      <div style="width:2rem;height:2rem;border-radius:50%;background:rgba({','.join(str(int(type_color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.15);
                           color:{type_color};display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:.8rem;font-weight:700;">
                        {"B" if a["type"]=="BUY" else "S" if a["type"]=="SELL" else "D"}
                      </div>
                      <div style="flex:1;">
                        <div style="display:flex;justify-content:space-between;">
                          <span style="font-size:.8rem;font-weight:700;color:#dae2fd;font-family:Manrope,sans-serif;">{a["type"]} {a["ticker"]}</span>
                          <span style="font-size:.65rem;color:#8c909f;font-family:IBM Plex Mono,monospace;">{a["date"].upper()}</span>
                        </div>
                        <div style="font-size:.7rem;color:#8c909f;margin-top:.1rem;">{desc}</div>
                        <div style="font-size:.7rem;font-weight:700;color:{type_color};margin-top:.1rem;font-family:IBM Plex Mono,monospace;">{amt_str}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────────────────
        with mkt_tab:
            st.markdown(f"""
            <div style="margin-bottom:1.2rem;">
              <div style="font-family:Manrope,sans-serif;font-size:2rem;font-weight:800;letter-spacing:-.02em;color:#dae2fd;">Market <span style="color:#4d8eff;">Intelligence</span></div>
              <div style="font-size:.78rem;color:#8c909f;margin-top:.2rem;">Live global performance and institutional sentiment analysis.</div>
            </div>
            """, unsafe_allow_html=True)

            # Market index cards
            mkt_cols = st.columns(4)
            mkt_data = [
                ("S&P 500","5,137.08","+1.24%","#00e5b0"),
                ("NASDAQ 100","18,302.91","+2.10%","#00e5b0"),
                ("DOW JONES","38,989.83","+0.68%","#00e5b0"),
                ("VIX","14.23","-5.2%","#00e5b0"),
            ]
            for i, (name, price, chg, col) in enumerate(mkt_data):
                with mkt_cols[i]:
                    st.markdown(f"""
                    <div style="background:linear-gradient(145deg,#131b2e,#171f33);border:1px solid #2d3449;border-top:2px solid {col};
                         padding:1.2rem;border-radius:.5rem;">
                      <div style="font-size:.6rem;font-weight:700;color:#8c909f;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem;">{name}</div>
                      <div style="font-family:IBM Plex Mono,monospace;font-size:1.4rem;font-weight:700;color:#dae2fd;">{price}</div>
                      <div style="font-family:IBM Plex Mono,monospace;font-size:.75rem;color:{col};font-weight:700;margin-top:.3rem;">{chg}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            ms1, ms2 = st.columns([2,1])
            with ms1:
                st.subheader("Sector Heat Map")
                sectors = [
                    ("Technology","+3.2%","#00e5b0"),("Healthcare","+1.1%","#00e5b0"),
                    ("Financials","-0.4%","#ff6b6b"),("Energy","+0.8%","#00e5b0"),
                    ("Consumer Disc.","+1.9%","#00e5b0"),("Industrials","-0.2%","#ff6b6b"),
                    ("Utilities","+0.3%","#00e5b0"),("Real Estate","-1.2%","#ff6b6b"),
                    ("Materials","+0.6%","#00e5b0"),("Comm. Services","+2.4%","#00e5b0"),
                ]
                cols5 = st.columns(5)
                for i, (name, chg, col) in enumerate(sectors):
                    with cols5[i % 5]:
                        st.markdown(f"""
                        <div style="background:#131b2e;border:1px solid #2d3449;border-left:2px solid {col};
                             padding:.75rem .9rem;margin-bottom:.5rem;border-radius:0 .5rem .5rem 0;">
                          <div style="font-size:.6rem;font-weight:700;color:#8c909f;text-transform:uppercase;margin-bottom:.25rem;">{name}</div>
                          <div style="font-family:IBM Plex Mono,monospace;font-size:.9rem;font-weight:700;color:{col};">{chg}</div>
                        </div>""", unsafe_allow_html=True)

            with ms2:
                st.subheader("Fear & Greed Index")
                st.markdown("""
                <div style="background:linear-gradient(145deg,#131b2e,#171f33);border:1px solid #2d3449;
                     padding:1.4rem;text-align:center;border-radius:.5rem;">
                  <div style="font-family:IBM Plex Mono,monospace;font-size:2.8rem;font-weight:800;color:#ffdd2d;">74</div>
                  <div style="font-family:Manrope,sans-serif;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#ffdd2d;margin-bottom:.8rem;">Greed</div>
                  <div style="height:6px;background:linear-gradient(90deg,#ff6b6b,#ff9f40,#ffdd2d,#00e5b0);border-radius:3px;position:relative;">
                    <div style="position:absolute;top:-10px;left:72%;transform:translateX(-50%);width:2px;height:26px;background:#dae2fd;border-radius:1px;"></div>
                  </div>
                  <div style="display:flex;justify-content:space-between;margin-top:.5rem;font-size:.58rem;color:#424754;font-weight:700;text-transform:uppercase;">
                    <span>Fear</span><span>Neutral</span><span>Greed</span>
                  </div>
                  <div style="margin-top:1rem;padding:.75rem;background:rgba(77,142,255,0.06);border-radius:.5rem;">
                    <div style="font-size:.7rem;color:#8c909f;line-height:1.5;">The market is in <b style="color:#ffdd2d;">Greed</b> territory, driven by tech earnings. Watch for potential pullbacks.</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────────────────
        with deep_tab:
            # Price alert
            if alert_price > 0:
                diff = alert_price - last_close
                if last_close >= alert_price:
                    st.markdown(f'<div class="alert-box">🔔 {ticker} at ${last_close:.2f} — AT or ABOVE your target of ${alert_price:.2f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-box">🔔 {ticker} at ${last_close:.2f} — ${diff:.2f} below target of ${alert_price:.2f}</div>', unsafe_allow_html=True)

            # Confidence score bar
            conf_color = "#00e5b0" if confidence_score>=80 else "#ffdd2d" if confidence_score>=60 else "#ff6b6b"
            conf_label = "HIGH CONFIDENCE" if confidence_score>=80 else "MODERATE CONFIDENCE" if confidence_score>=60 else "LOW CONFIDENCE"
            filled = int(confidence_score / 5)
            bar_html = "".join(f'<span style="display:inline-block;width:18px;height:10px;margin-right:2px;background:{conf_color};opacity:{1.0 if i<filled else 0.1};border-radius:1px;"></span>' for i in range(20))
            st.markdown(f"""
            <div style="background:#131b2e;border:1px solid #2d3449;border-left:3px solid {conf_color};
                 padding:1.2rem 1.6rem;margin:1rem 0;border-radius:0 .5rem .5rem 0;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;">
                <div>
                  <div style="font-family:Manrope,sans-serif;font-size:.6rem;letter-spacing:.16em;text-transform:uppercase;color:#424754;margin-bottom:.3rem;font-weight:700;">MODEL CONFIDENCE SCORE</div>
                  <div style="font-family:IBM Plex Mono,monospace;font-size:2.2rem;font-weight:700;color:{conf_color};">{confidence_score:.0f}<span style="font-size:1rem;color:#8c909f;">/100</span></div>
                  <div style="font-family:Manrope,sans-serif;font-size:.62rem;letter-spacing:.14em;color:{conf_color};margin-top:.3rem;font-weight:700;">{conf_label}</div>
                </div>
                <div style="flex:1;min-width:220px;">
                  <div style="margin-bottom:.6rem;">{bar_html}</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:.3rem .8rem;font-family:IBM Plex Mono,monospace;font-size:.63rem;color:#424754;">
                    <span>R² fit <b style="color:#8c909f;">{r2_norm:.0f}/100</b> <span style="color:#2d3449;">(×0.40)</span></span>
                    <span>MAPE accuracy <b style="color:#8c909f;">{mape_norm:.0f}/100</b> <span style="color:#2d3449;">(×0.30)</span></span>
                    <span>Directional acc. <b style="color:#8c909f;">{dir_acc:.0f}/100</b> <span style="color:#2d3449;">(×0.20)</span></span>
                    <span>Data volume <b style="color:#8c909f;">{data_score:.0f}/100</b> <span style="color:#2d3449;">(×0.10)</span></span>
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Signal Intelligence ────────────────────────────────────────────────
            st.subheader("Signal Intelligence")
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

            if alert_on_signal_change:
                prev_verdict = st.session_state.alert_signals.get(ticker)
                if prev_verdict is not None and prev_verdict != verdict_short:
                    _ac = {"BUY":"#00e5b0","SELL":"#ff6b6b"}.get(verdict_short,"#ffdd2d")
                    st.markdown(f'<div style="background:rgba(77,142,255,0.08);border:1px solid {_ac};border-left:4px solid {_ac};padding:.8rem 1.4rem;margin-bottom:1rem;font-family:Manrope,sans-serif;font-size:.78rem;color:{_ac};font-weight:700;border-radius:0 .5rem .5rem 0;">🔔 SIGNAL ALERT — {ticker} &nbsp;|&nbsp; {prev_verdict} → {verdict_short} &nbsp;|&nbsp; Score: {total_score:+.0f}</div>', unsafe_allow_html=True)
                st.session_state.alert_signals[ticker] = verdict_short

            verdict_css = 'sell' if verdict_short=='SELL' else 'hold' if verdict_short=='HOLD' else ''
            sign = '+' if xgb_pct>=0 else ''
            score_color = '#00e5b0' if total_score>0 else '#ff6b6b' if total_score<0 else '#ffdd2d'
            rr_color = 'positive' if risk_reward>=1.5 else 'negative' if risk_reward<1 else 'neutral'

            st.markdown(f"""
            <div class="signal-panel">
              <div class="signal-main {verdict_css}">
                <div class="signal-lbl">Composite Signal</div>
                <div class="signal-action {verdict_css}">{verdict}</div>
                <div class="signal-pct">{sign}{xgb_pct:.2f}% forecast</div>
                <div class="signal-lbl" style="margin-top:8px;">Score: <span style="color:{score_color};font-size:.9rem;font-weight:800;">{total_score:+.0f}</span> / ±100</div>
              </div>
              <div class="signal-details">
                <div class="sig-card positive">
                  <div class="sig-lbl">Take Profit</div>
                  <div class="sig-val">${take_profit:.2f}</div>
                  <div class="sig-sub">+{((take_profit-last_close)/last_close*100):.1f}% · 3× ATR</div>
                </div>
                <div class="sig-card negative">
                  <div class="sig-lbl">Stop Loss</div>
                  <div class="sig-val">${stop_loss:.2f}</div>
                  <div class="sig-sub">{((stop_loss-last_close)/last_close*100):.1f}% · 2× ATR</div>
                </div>
                <div class="sig-card {rr_color}">
                  <div class="sig-lbl">Risk / Reward</div>
                  <div class="sig-val">{risk_reward:.2f}×</div>
                  <div class="sig-sub">{"✓ Favorable" if risk_reward>=1.5 else "⚠ Marginal" if risk_reward>=1 else "✗ Unfavorable"}</div>
                </div>
                <div class="sig-card {'positive' if rsi_val<50 else 'negative'}">
                  <div class="sig-lbl">RSI (14)</div>
                  <div class="sig-val">{rsi_val:.1f}</div>
                  <div class="sig-sub">{'Oversold zone' if rsi_val<30 else 'Overbought zone' if rsi_val>70 else 'Neutral zone'}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Composite meter
            st.markdown('<div class="composite-meter"><div class="meter-title">6-Factor Signal Breakdown</div>', unsafe_allow_html=True)
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

            # ── Future Forecast ────────────────────────────────────────────────
            st.subheader(f"Forecast — Next {future_days} Days")
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
            fig3.add_trace(go.Scatter(x=list(range(future_days)), y=future_prices, mode='lines+markers', name='XGBoost Forecast',
                line=dict(color=trend_color, width=2), marker=dict(size=7, color=trend_color, line=dict(width=1, color="#0b1326"))))
            if future_days > 5:
                fig3.add_vline(x=4.5, line_dash="dot", line_color="#2d3449", annotation_text="↑ Higher confidence | Lower confidence ↓",
                               annotation_font=dict(color="#424754", size=9), annotation_position="top")
            fig3.update_layout(**PLOTLY_LAYOUT,
                title=dict(text=f"{ticker} · {future_days}-Day Price Forecast · Band shows ±1.5σ uncertainty growth", font=dict(color=C_GREEN, size=12)),
                xaxis_title="Days from today", yaxis_title="Price (USD)", height=380)
            st.plotly_chart(fig3, use_container_width=True)

            if future_days > 5:
                st.markdown('<div style="background:rgba(255,221,45,0.04);border:1px solid rgba(255,221,45,0.2);border-left:3px solid #ffdd2d;padding:.6rem 1.2rem;font-family:Manrope,sans-serif;font-size:.7rem;color:#ffdd2d;font-weight:600;border-radius:0 .5rem .5rem 0;">⚠ Forecast reliability decreases significantly beyond Day 5. Use Days 6+ as directional signals only.</div>', unsafe_allow_html=True)

            future_df = pd.DataFrame({
                "Day": [f"+{i+1}" for i in range(future_days)],
                "Predicted Price ($)": [f"${p:.2f}" for p in future_prices],
                "vs Last Close": [f"{'▲' if p>last_close else '▼'} {abs(p-last_close):.2f} ({(p-last_close)/last_close*100:+.2f}%)" for p in future_prices]
            })
            st.dataframe(future_df, use_container_width=True, hide_index=True)

            # ── Backtesting ────────────────────────────────────────────────────
            if run_backtest and not fast_mode:
                st.subheader("Backtesting Engine")
                st.markdown(f'<div class="model-badge">STRATEGY: XGBoost Signal ±{bt_signal_threshold}% | Capital: ${bt_initial_capital:,.0f} | Commission: ${bt_commission}/trade</div>', unsafe_allow_html=True)
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

                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(y=bt["equity_curve"], name="XGBoost Strategy", line=dict(color=C_EMERALD, width=2), fill="tozeroy", fillcolor="rgba(0,229,176,0.05)"))
                fig_eq.add_trace(go.Scatter(y=bt["bh_equity"], name="Buy & Hold", line=dict(color=C_ACCENT, width=1.5, dash="dot")))
                fig_eq.add_hline(y=bt_initial_capital, line_dash="dash", line_color=C_GREY, annotation_text=f"Start ${bt_initial_capital:,}", annotation_font_color=C_GREY)
                fig_eq.update_layout(**PLOTLY_LAYOUT, title=dict(text=f"{ticker} · Strategy Equity Curve vs Buy & Hold", font=dict(color=C_GREEN, size=12)), height=380)
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
                st.subheader("Forecast with Confidence Intervals")
                st.markdown('<div class="model-badge">95% CI — Bootstrap Resampling</div>', unsafe_allow_html=True)
                with st.spinner(f"Running {ci_bootstrap_n} bootstrap samples..."):
                    ci_lower, ci_median, ci_upper = bootstrap_confidence_intervals(model, X_test, n_bootstrap=ci_bootstrap_n, noise_std=0.015)
                fig_ci = go.Figure()
                fig_ci.add_trace(go.Scatter(y=ci_upper, line=dict(color="rgba(0,0,0,0)"), showlegend=False))
                fig_ci.add_trace(go.Scatter(y=ci_lower, name="95% CI Band", fill="tonexty", fillcolor="rgba(77,142,255,0.10)", line=dict(color="rgba(0,0,0,0)")))
                fig_ci.add_trace(go.Scatter(y=actual, name="Actual", line=dict(color=C_ACCENT, width=1.5)))
                fig_ci.add_trace(go.Scatter(y=ci_median, name="XGBoost Median", line=dict(color=C_EMERALD, width=1.8, dash="dot")))
                fig_ci.update_layout(**PLOTLY_LAYOUT, title=dict(text=f"{ticker} · Predictions with 95% CI", font=dict(color=C_GREEN, size=12)), height=380)
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
                fig_cmp.update_layout(**PLOTLY_LAYOUT, title=dict(text=f"{ticker} · Model Comparison", font=dict(color=C_GREEN, size=12)), height=380)
                st.plotly_chart(fig_cmp, use_container_width=True)

            # ── Halal / Shariah ────────────────────────────────────────────────
            if run_halal_check:
                st.markdown("""
                <div style="background:rgba(0,229,176,0.03);border:1px solid rgba(0,229,176,0.15);border-left:4px solid #00e5b0;
                     padding:.8rem 1.4rem;margin:1.5rem 0 .5rem;display:flex;align-items:center;gap:1rem;border-radius:0 .5rem .5rem 0;">
                  <div style="font-size:1.4rem;">☪</div>
                  <div>
                    <div style="font-family:Manrope,sans-serif;font-size:.63rem;letter-spacing:.15em;text-transform:uppercase;color:#00e5b0;font-weight:700;">Halal / Shariah Compliance Screen</div>
                    <div style="font-family:Manrope,sans-serif;font-size:.76rem;color:#424754;margin-top:2px;">Automated screening based on AAOIFI Standard No.21</div>
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
                    cr = check_shariah_compliance(ticker, sd)
                    verdict = cr["verdict"]
                    v_color = {"COMPLIANT":C_EMERALD,"NON-COMPLIANT":C_RED,"QUESTIONABLE":C_YELLOW}[verdict]
                    v_bg    = {"COMPLIANT":"rgba(0,229,176,0.05)","NON-COMPLIANT":"rgba(255,107,107,0.05)","QUESTIONABLE":"rgba(255,221,45,0.05)"}[verdict]
                    v_icon  = {"COMPLIANT":"✅","NON-COMPLIANT":"❌","QUESTIONABLE":"⚠️"}[verdict]
                    st.markdown(f'<div style="background:{v_bg};border:1px solid {v_color};border-left:3px solid {v_color};padding:1.2rem 2rem;margin:1rem 0;text-align:center;border-radius:0 .5rem .5rem 0;"><div style="font-family:Manrope,sans-serif;font-size:.6rem;color:#424754;letter-spacing:.14em;text-transform:uppercase;font-weight:700;">{sd["company_name"]} ({ticker})</div><div style="font-family:IBM Plex Mono,monospace;font-size:1.8rem;font-weight:700;color:{v_color};margin-top:.4rem;">{v_icon}&nbsp;{verdict}</div><div style="font-size:.76rem;color:#8c909f;margin-top:.3rem;">Sector: {sd["sector"]} | Industry: {sd["industry"]}</div></div>', unsafe_allow_html=True)

                    st.subheader("Screening Criteria")
                    cl, cr2 = st.columns(2)
                    with cl:
                        bs = cr["business"]
                        if bs["haram_hit"]:
                            st.markdown(f'<div class="halal-card-fail"><b>❌ Business Activity</b><br>Non-compliant: <b>{bs["haram_hit"]}</b></div>', unsafe_allow_html=True)
                        elif bs["questionable"]:
                            st.markdown('<div class="halal-card" style="border-left-color:#ffdd2d;"><b>⚠️ Business Activity</b><br>Questionable sector — consult a scholar</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="halal-card"><b>✅ Business Activity</b><br>No Haram core business detected<br><small style="color:#424754;">Sector: {sd["sector"]}</small></div>', unsafe_allow_html=True)
                        dm = cr["debt_mktcap"]
                        cls = "halal-card" if dm["pass"] else "halal-card-fail"
                        icon = "✅" if dm["pass"] else "❌"
                        st.markdown(f'<div class="{cls}"><b>{icon} {dm["label"]}</b></div>', unsafe_allow_html=True)
                    with cr2:
                        da = cr["debt_assets"]
                        cls = "halal-card" if da["pass"] else "halal-card-fail"
                        icon = "✅" if da["pass"] else "❌"
                        st.markdown(f'<div class="{cls}"><b>{icon} {da["label"]}</b></div>', unsafe_allow_html=True)
                        ca = cr["cash_assets"]
                        cls = "halal-card" if ca["pass"] else "halal-card-fail"
                        icon = "✅" if ca["pass"] else "❌"
                        st.markdown(f'<div class="{cls}"><b>{icon} {ca["label"]}</b></div>', unsafe_allow_html=True)

            # ── News Sentiment ──────────────────────────────────────────────────
            if not is_beginner:
                st.subheader("News Sentiment NLP")
                try:
                    from textblob import TextBlob
                    t_obj = yf.Ticker(ticker)
                    news  = t_obj.news
                    if news:
                        scored = []
                        for item in news[:10]:
                            title = item.get("title","")
                            if title:
                                pol = TextBlob(title).sentiment.polarity
                                scored.append({"headline":title,"polarity":pol})
                        if scored:
                            sc_df = pd.DataFrame(scored)
                            avg_polarity = sc_df["polarity"].mean()
                            sent_color   = C_EMERALD if avg_polarity>0.05 else C_RED if avg_polarity<-0.05 else C_YELLOW
                            sent_label   = "POSITIVE" if avg_polarity>0.05 else "NEGATIVE" if avg_polarity<-0.05 else "NEUTRAL"
                            st.markdown(f'<div style="background:rgba(77,142,255,0.06);border:1px solid rgba(77,142,255,0.2);border-left:3px solid {sent_color};padding:.7rem 1.2rem;font-family:Manrope,sans-serif;font-size:.72rem;color:#dae2fd;font-weight:700;border-radius:0 .5rem .5rem 0;">Avg Sentiment: <span style="color:{sent_color};">{sent_label}</span> &nbsp;({avg_polarity:+.3f}) &nbsp;·&nbsp; {len(scored)} recent headlines</div>', unsafe_allow_html=True)
                            fig_sent = go.Figure(go.Bar(x=sc_df["polarity"], y=[h[:55]+"…" if len(h)>55 else h for h in sc_df["headline"]], orientation='h',
                                marker_color=[C_EMERALD if p>0 else C_RED for p in sc_df["polarity"]]))
                            fig_sent.add_vline(x=0, line_color=C_GREY)
                            fig_sent.add_vline(x=avg_polarity, line_dash="dot", line_color=sent_color, line_width=1.5)
                            fig_sent.update_layout(**PLOTLY_LAYOUT,
                                title=dict(text=f"{ticker} · Headline Sentiment (TextBlob)", font=dict(color=C_GREEN, size=11)),
                                height=max(220, len(scored)*32), xaxis_title="Polarity (negative ← 0 → positive)",
                                xaxis=dict(range=[-1,1], gridcolor="#2d3449", linecolor="#2d3449", zeroline=False, tickfont=dict(color="#424754",size=9)))
                            st.plotly_chart(fig_sent, use_container_width=True)
                            st.caption("⚠ Sentiment is based on headline text only — not article content. Use as supplementary signal.")
                    else:
                        st.info("No recent news found for this ticker.")
                except ImportError:
                    st.info("Install `textblob` to enable News Sentiment NLP.")
                except Exception as e:
                    st.warning(f"Could not fetch news: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding:1.5rem;border-top:1px solid #2d3449;">
  <div style="font-family:IBM Plex Mono,monospace;font-size:.6rem;color:#2d3449;letter-spacing:.1em;">
    ⚠ STOCKCAST · FOR EDUCATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE · Developed by MUAWWIZ GHANI
  </div>
</div>
""", unsafe_allow_html=True)


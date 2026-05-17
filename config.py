# =============================================================================
# Copyright (c) 2026 Stockcast. All Rights Reserved.
# PROPRIETARY AND CONFIDENTIAL
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import os
import io
import csv
import json
import math
import time
import logging
import smtplib
import threading
import warnings
from typing import List, Optional, Dict, Any

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from supabase import create_client
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import html as _html_mod

warnings.filterwarnings("ignore")

# ── Optional broker integrations ──────────────────────────────────────────────
# Default to False; overwritten to True if the library is available at runtime.
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
    _posthog_lib.project_api_key = os.environ.get("POSTHOG_API_KEY", "")
    _posthog_lib.host = "https://app.posthog.com"
    _POSTHOG_OK = bool(_posthog_lib.project_api_key)
except Exception:
    pass

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stockcast")

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

# ── Supabase config ───────────────────────────────────────────────────────────
# Credentials loaded from Streamlit secrets (secrets.toml) or environment variables.
# Never hardcode credentials in source code.
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

# ── Plan definitions ──────────────────────────────────────────────────────────
PLAN_LIMITS = {
    "free": {
        "daily_analyses":   3,
        "watchlist_stocks": 5,
        "forecast_horizon": 7,
        "model_compare":    False,
        "conf_interval":    False,
        "multi_stock":      False,
        "data_years":       3,
    },
    "pro": {
        "daily_analyses":   999,
        "watchlist_stocks": 50,
        "forecast_horizon": 30,
        "model_compare":    True,
        "conf_interval":    True,
        "multi_stock":      True,
        "data_years":       10,
    },
}

# ── Popular tickers dictionary ────────────────────────────────────────────────
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

# ── Plotly dark theme ─────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#080f1e",
    plot_bgcolor="#080f1e",
    font=dict(family="IBM Plex Mono, monospace", color="#8a8fa0", size=11),
    xaxis=dict(gridcolor="#1e2740", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e2740", showgrid=True, zeroline=False),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2740", borderwidth=1),
)

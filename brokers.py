# =============================================================================
# Copyright (c) 2026 Stockcast. All Rights Reserved.
# PROPRIETARY AND CONFIDENTIAL
# brokers.py — Live brokerage integrations: Zerodha, Upstox, Alpaca
# =============================================================================

import logging
import streamlit as st

from config import _KITE_OK, _UPSTOX_OK, _ALPACA_OK

try:
    from kiteconnect import KiteConnect
except ImportError:
    pass

try:
    import upstox_client
except ImportError:
    pass

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    pass


# ── Holdings fetchers ─────────────────────────────────────────────────────────

def get_zerodha_holdings(api_key: str, access_token: str) -> list:
    """Live Zerodha holdings via Kite Connect. Requires: pip install kiteconnect"""
    if not _KITE_OK:
        return []
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        return [
            {"symbol": h["tradingsymbol"], "qty": h["quantity"],
             "avg_price": h["average_price"], "last_price": h["last_price"],
             "pnl": h["pnl"],
             "pnl_pct": (h["pnl"] / max(h["average_price"] * h["quantity"], 0.01)) * 100,
             "broker": "Zerodha"}
            for h in kite.holdings()
        ]
    except Exception as e:
        logging.warning("Zerodha: %s", e)
        return []


def get_upstox_holdings(api_key: str, access_token: str) -> list:
    """Live Upstox holdings. Requires: pip install upstox-python-sdk"""
    if not _UPSTOX_OK:
        return []
    try:
        cfg = upstox_client.Configuration(host="https://api.upstox.com/v2")
        cfg.access_token = access_token
        api = upstox_client.PortfolioApi(upstox_client.ApiClient(cfg))
        return [
            {"symbol": h.tradingsymbol, "qty": h.quantity,
             "avg_price": h.average_price, "last_price": h.last_price,
             "pnl": (h.last_price - h.average_price) * h.quantity,
             "pnl_pct": ((h.last_price / max(h.average_price, 0.01)) - 1) * 100,
             "broker": "Upstox"}
            for h in (api.get_holdings().data or [])
        ]
    except Exception as e:
        logging.warning("Upstox: %s", e)
        return []


def get_alpaca_holdings(api_key: str, api_secret: str,
                        base_url: str = "https://paper-api.alpaca.markets") -> list:
    """Live Alpaca positions. Requires: pip install alpaca-trade-api"""
    if not _ALPACA_OK:
        return []
    try:
        api = tradeapi.REST(api_key, api_secret, base_url, api_version="v2")
        return [
            {"symbol": p.symbol, "qty": float(p.qty),
             "avg_price": float(p.avg_entry_price), "last_price": float(p.current_price),
             "pnl": float(p.unrealized_pl), "pnl_pct": float(p.unrealized_plpc) * 100,
             "broker": "Alpaca"}
            for p in api.list_positions()
        ]
    except Exception as e:
        logging.warning("Alpaca: %s", e)
        return []


# ── Brokerage panel UI ────────────────────────────────────────────────────────

def render_brokerage_panel():
    """Render the brokerage integration UI panel."""
    st.markdown("""
        <div style="font-family:Manrope,sans-serif;font-size:1rem;font-weight:800;
          color:#e4eafd;margin-bottom:.4rem;">🔗 Live Portfolio Sync</div>
        <div style="font-size:.82rem;color:#8a8fa0;line-height:1.6;margin-bottom:1.2rem;">
          Connect your broker to sync live holdings, P&L and order history.</div>
    """, unsafe_allow_html=True)

    brokers = [
        ("Zerodha Kite", "🟠", "#ff6b35", ["Live Holdings", "Order Insights", "Historical Trades"], "IN"),
        ("Upstox",       "🔵", "#3b82f6", ["Portfolio Sync", "Market Depth", "Margin Data"],        "IN"),
        ("Alpaca",       "🦙", "#f59e0b", ["Fractional Shares", "Paper Trading", "Live Orders"],    "US"),
        ("IBKR",         "🌐", "#6b7280", ["Global Markets", "Options", "Futures"],                 "Global — coming soon"),
        ("Angel One",    "⭐", "#6b7280", ["Smart API", "Mutual Funds"],                            "IN — coming soon"),
    ]

    cols = st.columns(len(brokers))
    for col, (name, logo, color, feats, region) in zip(cols, brokers):
        coming     = "coming soon" in region
        feats_html = "".join(
            f'<div style="font-size:.68rem;color:#8a8fa0;padding:.1rem 0;">'
            f'<span style="color:#00e5b0;font-size:.6rem;">✓</span> {f}</div>'
            for f in feats
        )
        with col:
            st.markdown(
                f'<div style="background:#0f1727;border:1px solid {"#252f47" if coming else color+"44"};'
                f'border-top:2px solid {color};border-radius:.75rem;padding:1.1rem;'
                f'opacity:{"0.5" if coming else "1"};">'
                f'<div style="font-size:1.5rem;margin-bottom:.4rem;">{logo}</div>'
                f'<div style="font-family:Manrope,sans-serif;font-size:.8rem;font-weight:700;color:#e4eafd;">{name}</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:.6rem;color:#3e4558;margin-bottom:.6rem;">{region}</div>'
                f'{feats_html}</div>',
                unsafe_allow_html=True
            )
            if not coming:
                if st.button(f"Connect {name}", key=f"br_{name}", use_container_width=True):
                    st.info(f"Add {name} credentials to .streamlit/secrets.toml")

    with st.expander("🔑 Setup Guide — secrets.toml"):
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

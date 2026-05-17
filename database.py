# =============================================================================
# Copyright (c) 2026 Stockcast. All Rights Reserved.
# PROPRIETARY AND CONFIDENTIAL
# database.py — All Supabase helpers: portfolio, watchlist, usage, plan, email alerts
# =============================================================================

import os
import smtplib
import logging
import streamlit as st
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import logger, supabase, PLAN_LIMITS

# ── Plan helpers ──────────────────────────────────────────────────────────────

def _get_limit(key: str):
    """Return the limit value for the current user's plan."""
    plan = st.session_state.get("user_plan", "free")
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["free"]).get(key)

def _is_pro() -> bool:
    return st.session_state.get("user_plan", "free") == "pro"

# ── Portfolio helpers ─────────────────────────────────────────────────────────

def _sb_load_portfolio(user_id: str) -> list:
    try:
        res  = supabase.table("portfolio_holdings").select("*").eq("user_id", user_id).execute()
        rows = res.data or []
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
            "user_id":       user_id,
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
        logger.error("_sb_delete_holding failed for user '%s', ticker '%s': %s", user_id, ticker, e, exc_info=True)
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
    """Batch-update current_price / pl / pl_pct for all holdings in one round-trip."""
    if not holdings:
        return
    rows = [{"user_id": user_id, "ticker": h["ticker"],
             "current_price": h["current_price"], "pl": h["pl"], "pl_pct": h["pl_pct"]}
            for h in holdings]
    try:
        supabase.table("portfolio_holdings").upsert(rows, on_conflict="user_id,ticker").execute()
        logger.info("_sb_update_prices: updated %d holdings for user '%s'", len(rows), user_id)
    except Exception as e:
        logger.error("_sb_update_prices failed for user '%s': %s", user_id, e, exc_info=True)
        st.warning("⚠ Could not refresh portfolio prices — data may be stale.")

# ── Watchlist helpers ─────────────────────────────────────────────────────────

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

# ── Usage limit helpers ───────────────────────────────────────────────────────

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

# ── Plan Supabase helpers ─────────────────────────────────────────────────────

def _sb_get_plan(user_id: str) -> str:
    try:
        res = supabase.table("user_usage").select("plan").eq("user_id", user_id).execute()
        row = (res.data or [None])[0]
        return (row or {}).get("plan", "free")
    except Exception as e:
        logger.error("_sb_get_plan failed for user '%s': %s", user_id, e)
        return "free"

def _sb_set_plan(user_id: str, plan: str):
    try:
        supabase.table("user_usage").upsert(
            {"user_id": user_id, "plan": plan}, on_conflict="user_id"
        ).execute()
        st.session_state.user_plan = plan
        logger.info("Plan updated to '%s' for user '%s'", plan, user_id)
    except Exception as e:
        logger.error("_sb_set_plan failed for user '%s': %s", user_id, e)

def _sb_get_email_alerts(user_id: str) -> bool:
    try:
        res = supabase.table("user_usage").select("email_alerts_enabled").eq("user_id", user_id).execute()
        row = (res.data or [None])[0]
        return bool((row or {}).get("email_alerts_enabled", False))
    except Exception as e:
        logger.error("_sb_get_email_alerts failed for user '%s': %s", user_id, e)
        return False

def _sb_set_email_alerts(user_id: str, enabled: bool):
    try:
        supabase.table("user_usage").upsert(
            {"user_id": user_id, "email_alerts_enabled": enabled}, on_conflict="user_id"
        ).execute()
        st.session_state.email_alerts_enabled = enabled
        logger.info("Email alerts set to %s for user '%s'", enabled, user_id)
        return True
    except Exception as e:
        logger.error("_sb_set_email_alerts failed for user '%s': %s", user_id, e)
        st.session_state.email_alerts_enabled = enabled
        return False

# ── Email sending ─────────────────────────────────────────────────────────────

def _send_email(to_address: str, subject: str, html_body: str) -> bool:
    """Send an email via SMTP using credentials from Streamlit secrets.

    Required secrets (add to .streamlit/secrets.toml):
        SMTP_HOST = "smtp.gmail.com"
        SMTP_PORT = 587
        SMTP_USER = "you@yourdomain.com"
        SMTP_PASS = "your-app-password"
        SMTP_FROM = "Stockcast <you@yourdomain.com>"  # optional
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

        msg            = MIMEMultipart("alternative")
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

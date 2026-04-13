# =============================================================================
# Stockcast — Daily Email Alerts
# =============================================================================
# Sends a daily stock insight digest to all users who have enabled email alerts.
#
# Run manually:          python send_alerts.py
# Run via cron (daily):  0 7 * * 1-5 /path/to/venv/bin/python /path/to/send_alerts.py
# Run via GitHub Actions: see .github/workflows/daily_alerts.yml
#
# Required environment variables (set in .env or Streamlit secrets):
#   SUPABASE_URL        — your Supabase project URL
#   SUPABASE_KEY        — your Supabase service role key (NOT anon key)
#   RESEND_API_KEY      — your Resend.com API key
#   ALERT_FROM_EMAIL    — verified sender address (e.g. alerts@yourdomain.com)
#   ALERT_FROM_NAME     — sender display name (e.g. "Stockcast")
# =============================================================================

import os
import sys
import time
import logging
import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from supabase import create_client
from datetime import date
warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stockcast.alerts")

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SUPABASE_URL     = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY     = os.environ.get("SUPABASE_KEY", "")   # use service_role key here
RESEND_API_KEY   = os.environ.get("RESEND_API_KEY", "")
FROM_EMAIL       = os.environ.get("ALERT_FROM_EMAIL", "alerts@stockcast.app")
FROM_NAME        = os.environ.get("ALERT_FROM_NAME", "Stockcast")
APP_URL          = os.environ.get("APP_URL", "https://muawwizghani-stock-forecast.streamlit.app")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("SUPABASE_URL and SUPABASE_KEY must be set.")
    sys.exit(1)

if not RESEND_API_KEY:
    logger.error("RESEND_API_KEY must be set.")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Quick Technical Signal ────────────────────────────────────────────────────

def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta    = series.diff()
    gain     = delta.clip(lower=0).rolling(period).mean()
    loss     = (-delta.clip(upper=0)).rolling(period).mean().replace(0, 1e-10)
    rs       = gain / loss
    rsi_vals = (100 - 100 / (1 + rs)).dropna()
    return float(rsi_vals.iloc[-1]) if len(rsi_vals) else 50.0

def _quick_signal(ticker: str) -> dict:
    """
    Returns a dict with:
        price, change_pct, signal (BUY/SELL/HOLD),
        confidence (0-100), direction (📈/📉/➡️),
        trend_label, rsi, ma_trend
    """
    default = {
        "price": 0.0, "change_pct": 0.0, "signal": "HOLD",
        "confidence": 0, "direction": "➡️", "trend_label": "Neutral",
        "rsi": 50.0, "ma_trend": "—", "error": None,
    }
    try:
        df = yf.download(ticker, period="6mo", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            default["error"] = "Insufficient data"
            return default

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close  = df["Close"].squeeze().dropna()
        price  = float(close.iloc[-1])
        prev   = float(close.iloc[-2])
        chg    = (price - prev) / prev * 100 if prev else 0.0

        ma20   = float(close.rolling(20).mean().iloc[-1])
        ma50   = float(close.rolling(50).mean().iloc[-1])
        rsi    = _compute_rsi(close)
        ema12  = float(close.ewm(span=12, adjust=False).mean().iloc[-1])
        ema26  = float(close.ewm(span=26, adjust=False).mean().iloc[-1])
        macd   = ema12 - ema26

        # Composite quick score (simplified 4-factor)
        score = 0
        if rsi < 35:              score += 25
        elif rsi > 65:            score -= 25
        if price > ma20 > ma50:   score += 20
        elif price < ma20 < ma50: score -= 20
        if macd > 0:              score += 15
        else:                     score -= 15
        if chg > 0.5:             score += 10
        elif chg < -0.5:          score -= 10

        if   score >= 25:  signal, direction, label = "BUY",  "📈", "Uptrend"
        elif score <= -25: signal, direction, label = "SELL", "📉", "Downtrend"
        else:              signal, direction, label = "HOLD", "➡️", "Neutral"

        confidence = min(95, max(30, 50 + abs(score)))
        ma_trend   = "Above MA50" if price > ma50 else "Below MA50"

        return {
            "price": price, "change_pct": chg, "signal": signal,
            "confidence": int(confidence), "direction": direction,
            "trend_label": label, "rsi": round(rsi, 1),
            "ma_trend": ma_trend, "error": None,
        }
    except Exception as e:
        logger.warning("_quick_signal failed for %s: %s", ticker, e)
        default["error"] = str(e)
        return default

# ── Email Template ────────────────────────────────────────────────────────────

def _build_email_html(user_email: str, stocks: list[dict], today_str: str) -> str:
    """Build the HTML email body."""

    def _signal_color(signal: str) -> str:
        return {"BUY": "#00e5b0", "SELL": "#ff5f5f", "HOLD": "#ffd426"}.get(signal, "#8a8fa0")

    def _confidence_bar(pct: int, color: str) -> str:
        return f"""
        <div style="width:100%;height:4px;background:#1e2740;border-radius:2px;margin-top:4px;">
          <div style="width:{pct}%;height:100%;background:{color};border-radius:2px;"></div>
        </div>"""

    rows_html = ""
    for s in stocks:
        if s.get("error"):
            rows_html += f"""
            <tr>
              <td style="padding:14px 0;border-bottom:1px solid #1e2740;">
                <strong style="font-family:'IBM Plex Mono',monospace;color:#e4eafd;
                  font-size:15px;">{s["ticker"]}</strong>
                <span style="font-family:Manrope,sans-serif;color:#3e4558;
                  font-size:12px;margin-left:8px;">Data unavailable</span>
              </td>
            </tr>"""
            continue

        col  = _signal_color(s["signal"])
        sign = "+" if s["change_pct"] >= 0 else ""
        rows_html += f"""
        <tr>
          <td style="padding:16px 0;border-bottom:1px solid #1e2740;">
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td width="40%">
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;
                    font-weight:700;color:#e4eafd;">{s["ticker"]}</div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;
                    color:#8a8fa0;margin-top:2px;">${s["price"]:.2f}
                    <span style="color:{'#00e5b0' if s['change_pct']>=0 else '#ff5f5f'};">
                      {sign}{s["change_pct"]:.2f}%
                    </span>
                  </div>
                </td>
                <td width="35%">
                  <div style="font-family:Manrope,sans-serif;font-size:11px;
                    font-weight:700;letter-spacing:.08em;text-transform:uppercase;
                    color:#8a8fa0;margin-bottom:3px;">Signal</div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;
                    font-weight:700;color:{col};">
                    {s["direction"]} {s["signal"]}
                  </div>
                  <div style="font-family:Manrope,sans-serif;font-size:11px;
                    color:#8a8fa0;">{s["trend_label"]}</div>
                  {_confidence_bar(s["confidence"], col)}
                </td>
                <td width="25%" style="text-align:right;">
                  <div style="font-family:Manrope,sans-serif;font-size:11px;
                    color:#3e4558;margin-bottom:2px;">RSI · {s["rsi"]}</div>
                  <div style="font-family:Manrope,sans-serif;font-size:11px;
                    color:#3e4558;">{s["ma_trend"]}</div>
                  <div style="display:inline-block;margin-top:4px;
                    background:rgba({','.join(str(int(col.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.12);
                    border:1px solid {col};border-radius:3px;
                    padding:2px 7px;font-family:'IBM Plex Mono',monospace;
                    font-size:10px;font-weight:700;color:{col};">
                    {s["confidence"]}%
                  </div>
                </td>
              </tr>
            </table>
          </td>
        </tr>"""

    buy_count  = sum(1 for s in stocks if s.get("signal") == "BUY")
    sell_count = sum(1 for s in stocks if s.get("signal") == "SELL")
    hold_count = sum(1 for s in stocks if s.get("signal") == "HOLD")

    if buy_count > sell_count:
        market_tone = "📈 Mostly bullish signals across your watchlist today."
        tone_color  = "#00e5b0"
    elif sell_count > buy_count:
        market_tone = "📉 Mostly bearish signals across your watchlist today."
        tone_color  = "#ff5f5f"
    else:
        market_tone = "➡️ Mixed signals — a cautious day across your watchlist."
        tone_color  = "#ffd426"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Your Daily Stock Insights — {today_str}</title>
</head>
<body style="margin:0;padding:0;background:#050a14;font-family:Manrope,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0"
       style="background:#050a14;min-height:100vh;">
  <tr><td align="center" style="padding:32px 16px;">

    <table width="100%" style="max-width:600px;background:#080e1c;
      border-radius:16px;border:1px solid #1e2740;overflow:hidden;">

      <!-- Header -->
      <tr>
        <td style="background:linear-gradient(135deg,#0f1727,#080e1c);
            border-bottom:1px solid #1e2740;padding:28px 32px;">
          <table width="100%" cellpadding="0" cellspacing="0">
            <tr>
              <td>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:20px;
                  font-weight:700;color:#e4eafd;letter-spacing:-.01em;">
                  Stock<span style="color:#4d8eff;">cast</span>
                </div>
                <div style="font-family:Manrope,sans-serif;font-size:11px;
                  color:#3e4558;letter-spacing:.1em;text-transform:uppercase;
                  margin-top:3px;">AI Stock Assistant</div>
              </td>
              <td align="right">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                  color:#3e4558;">{today_str}</div>
                <div style="display:inline-block;margin-top:4px;background:rgba(0,229,176,0.1);
                  border:1px solid rgba(0,229,176,0.3);border-radius:4px;
                  padding:3px 10px;font-family:'IBM Plex Mono',monospace;
                  font-size:10px;color:#00e5b0;letter-spacing:.06em;">
                  DAILY DIGEST
                </div>
              </td>
            </tr>
          </table>
        </td>
      </tr>

      <!-- Tone summary bar -->
      <tr>
        <td style="background:rgba({','.join(str(int(tone_color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.07);
            border-bottom:1px solid rgba({','.join(str(int(tone_color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.2);
            padding:12px 32px;">
          <div style="font-family:Manrope,sans-serif;font-size:13px;
            color:{tone_color};font-weight:600;">{market_tone}</div>
          <div style="font-family:Manrope,sans-serif;font-size:11px;color:#3e4558;margin-top:2px;">
            {buy_count} BUY · {hold_count} HOLD · {sell_count} SELL across {len(stocks)} stocks
          </div>
        </td>
      </tr>

      <!-- Stock rows -->
      <tr>
        <td style="padding:0 32px;">
          <table width="100%" cellpadding="0" cellspacing="0">
            {rows_html}
          </table>
        </td>
      </tr>

      <!-- CTA -->
      <tr>
        <td style="padding:24px 32px 12px;text-align:center;">
          <a href="{APP_URL}"
             style="display:inline-block;background:linear-gradient(135deg,#3d7bf5,#5a9aff);
               color:#fff;font-family:Manrope,sans-serif;font-size:13px;font-weight:700;
               letter-spacing:.06em;text-transform:uppercase;text-decoration:none;
               padding:12px 32px;border-radius:8px;">
            Open Stockcast →
          </a>
        </td>
      </tr>

      <!-- Disclaimer -->
      <tr>
        <td style="padding:16px 32px 28px;border-top:1px solid #1e2740;margin-top:12px;">
          <div style="font-family:Manrope,sans-serif;font-size:11px;color:#252f47;
            text-align:center;line-height:1.6;">
            ⚠ These signals are for educational use only — not financial advice.<br>
            Always combine AI signals with your own research and market context.<br>
            <a href="{APP_URL}" style="color:#3e4558;text-decoration:none;">
              Manage alerts
            </a>
            &nbsp;·&nbsp;
            <a href="{APP_URL}" style="color:#3e4558;text-decoration:none;">
              Unsubscribe
            </a>
          </div>
        </td>
      </tr>

    </table>
  </td></tr>
</table>
</body>
</html>"""

def _build_email_text(user_email: str, stocks: list[dict], today_str: str) -> str:
    """Plain-text fallback."""
    lines = [
        f"STOCKCAST — Daily Stock Insights",
        f"{today_str}",
        "=" * 42,
        "",
    ]
    for s in stocks:
        if s.get("error"):
            lines.append(f"{s['ticker']}: Data unavailable")
        else:
            sign = "+" if s["change_pct"] >= 0 else ""
            lines.append(
                f"{s['ticker']:8s}  {s['direction']} {s['signal']:4s}  "
                f"${s['price']:.2f} ({sign}{s['change_pct']:.2f}%)  "
                f"Confidence: {s['confidence']}%  RSI: {s['rsi']}"
            )
    lines += [
        "",
        "─" * 42,
        "Open Stockcast: " + APP_URL,
        "",
        "⚠ For educational use only — not financial advice.",
        "Manage alerts or unsubscribe: " + APP_URL,
    ]
    return "\n".join(lines)

# ── Send via Resend ───────────────────────────────────────────────────────────

def _send_via_resend(to_email: str, subject: str, html: str, text: str) -> bool:
    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from":    f"{FROM_NAME} <{FROM_EMAIL}>",
                "to":      [to_email],
                "subject": subject,
                "html":    html,
                "text":    text,
            },
            timeout=15,
        )
        if resp.status_code in (200, 201):
            logger.info("✓ Sent to %s (id: %s)", to_email, resp.json().get("id"))
            return True
        else:
            logger.error("✗ Resend error for %s: %s — %s",
                         to_email, resp.status_code, resp.text[:200])
            return False
    except Exception as e:
        logger.error("✗ Request failed for %s: %s", to_email, e)
        return False

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    today_str = date.today().strftime("%B %d, %Y")
    subject   = f"📊 Your Daily Stock Insights — {today_str}"

    logger.info("=== Stockcast Daily Alerts — %s ===", today_str)

    # 1. Fetch all users who have alerts enabled
    try:
        res = supabase.table("user_usage").select("user_id,email_alerts_enabled").execute()
        users = [r for r in (res.data or []) if r.get("email_alerts_enabled")]
    except Exception as e:
        logger.error("Could not fetch alert subscribers: %s", e)
        return

    logger.info("Found %d subscribers with alerts enabled", len(users))
    if not users:
        logger.info("No subscribers — nothing to send.")
        return

    # 2. Fetch each user's email from Supabase auth
    sent = skipped = failed = 0

    for user_row in users:
        uid = user_row["user_id"]

        # Get email from Supabase auth.users via admin API
        try:
            auth_res = supabase.auth.admin.get_user_by_id(uid)
            email = auth_res.user.email if auth_res and auth_res.user else None
        except Exception as e:
            logger.warning("Could not fetch email for uid %s: %s", uid, e)
            email = None

        if not email:
            logger.warning("Skipping uid %s — no email found", uid)
            skipped += 1
            continue

        # 3. Fetch their watchlist
        try:
            wl_res = supabase.table("watchlist").select("stock_symbol") \
                         .eq("user_id", uid).order("created_at").execute()
            symbols = [r["stock_symbol"] for r in (wl_res.data or [])]
        except Exception as e:
            logger.warning("Could not fetch watchlist for %s: %s", uid, e)
            symbols = []

        if not symbols:
            logger.info("Skipping %s — empty watchlist", email)
            skipped += 1
            continue

        # 4. Compute quick signal for each symbol
        stocks = []
        for sym in symbols[:10]:   # cap at 10 per email
            logger.debug("  Computing signal for %s...", sym)
            result = _quick_signal(sym)
            result["ticker"] = sym
            stocks.append(result)
            time.sleep(0.3)   # be gentle with yfinance rate limits

        if not stocks:
            skipped += 1
            continue

        # 5. Build and send email
        html = _build_email_html(email, stocks, today_str)
        text = _build_email_text(email, stocks, today_str)

        if _send_via_resend(email, subject, html, text):
            sent += 1
        else:
            failed += 1

        time.sleep(0.5)  # rate limit between sends

    logger.info("=== Done: %d sent · %d skipped · %d failed ===",
                sent, skipped, failed)

if __name__ == "__main__":
    run()

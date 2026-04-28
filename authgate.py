"""
authgate.py — StockCast Auth Gate
Handles login, signup, and password reset UI using Supabase Auth.
Called from app.py as: from authgate import render_auth_gate; render_auth_gate(supabase)
Supabase client is passed in — never initialised here (avoids secrets KeyError on import).

Architecture:
- Left panel: components.html iframe, pinned position:fixed via CSS on its Streamlit wrapper
- Right panel: pure Streamlit widgets with block-container offset to the right half
"""

import streamlit as st
import streamlit.components.v1 as components


def render_auth_gate(supabase):
    # ── Session state ─────────────────────────────────────────────────────────
    if "user" not in st.session_state:
        st.session_state.user = None
    if "_auth_mode" not in st.session_state:
        st.session_state._auth_mode = "login"

    if st.session_state.user is not None:
        return

    mode = st.session_state._auth_mode

    # ── Alpha card bar data ───────────────────────────────────────────────────
    _bar_heights = [20, 35, 45, 25, 60, 80, 70, 90, 100, 85, 65, 40, 20]
    _bars_html = ""
    for h in _bar_heights:
        cls = "active" if h == 100 else ("mid" if h >= 60 else "")
        _bars_html += f'<div class="{cls}" style="height:{h}%"></div>'

    _mini_html = "".join(
        f'<div class="mini-bar" style="height:{int(h*44/100)}px;opacity:{0.2+0.8*(h/100):.2f}"></div>'
        for h in [30, 60, 45, 80, 100, 70, 40]
    )

    # ── Ticker data ───────────────────────────────────────────────────────────
    _tape_items = []
    try:
        import yfinance as _yf
        _syms = ["AAPL", "MSFT", "TSLA", "GOOG", "NVDA", "META", "JPM"]
        _td = _yf.download(_syms, period="2d", interval="1d", progress=False, auto_adjust=True)
        for _s in _syms:
            try:
                _row = _td["Close"][_s].dropna()
                if len(_row) >= 2:
                    _px = float(_row.iloc[-1])
                    _chg = (_px - float(_row.iloc[-2])) / float(_row.iloc[-2]) * 100
                    _col = "#8eff71" if _chg >= 0 else "#ff7351"
                    _arr = "▲" if _chg >= 0 else "▼"
                    _tape_items.append(
                        f'<span class="ti"><b>{_s}</b> {_px:.2f} '
                        f'<span style="color:{_col}">{_arr}{abs(_chg):.2f}%</span></span>'
                    )
            except Exception:
                pass
    except Exception:
        pass
    if not _tape_items:
        _tape_items = [
            f'<span class="ti"><b>{s}</b> <span style="color:#8eff71">–</span></span>'
            for s in ["AAPL", "MSFT", "TSLA", "NVDA", "META", "GOOG", "JPM"]
        ]
    _tape = "".join(_tape_items * 4)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — Global CSS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg:          #0e0e0e;
        --surface:     #1a1919;
        --surf-mid:    #262626;
        --surf-hi:     #201f1f;
        --border:      rgba(90,88,87,0.28);
        --primary:     #8eff71;
        --pglow:       rgba(142,255,113,0.30);
        --pdim:        rgba(142,255,113,0.08);
        --text:        #ffffff;
        --muted:       #adaaaa;
        --fhead:       'Space Grotesk', sans-serif;
        --fbody:       'Inter', sans-serif;
        --fmono:       'JetBrains Mono', monospace;
    }

    /* ── Page shell ── */
    html, body, [class*="css"] {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: var(--fbody) !important;
    }
    #MainMenu, footer, header,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    .stDeployButton { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }

    /* ── Right-panel layout — shove content into right 50% ── */
    .main .block-container {
        max-width:  100% !important;
        padding:    0 !important;
        margin:     0 !important;
    }
    /* Every direct child of the main block lives in the right half */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:first-child {
        margin-left: 50vw !important;
        padding: 0 5vw !important;
        min-height: 100vh !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }

    /* ── Pin the components.html iframe as a fixed left panel ── */
    [data-testid="stCustomComponentV1"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 50vw !important;
        height: 100vh !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        z-index: 100 !important;
    }
    [data-testid="stCustomComponentV1"] iframe {
        width:  100% !important;
        height: 100% !important;
        border: none !important;
        display: block !important;
    }

    /* ── Tab bar ── */
    .sc-tabs {
        display: flex;
        gap: 1.8rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0;
    }
    .sc-tab {
        padding-bottom: 0.85rem;
        font-family: var(--fhead);
        font-size: 0.58rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-weight: 700;
        color: var(--muted);
        margin-bottom: -1px;
        user-select: none;
    }
    .sc-tab.on {
        color: var(--primary);
        border-bottom: 2px solid var(--primary);
    }

    /* ── Input labels ── */
    .sc-lbl {
        font-family: var(--fhead);
        font-size: 0.58rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 700;
        color: var(--muted);
        margin: 0.9rem 0 0.3rem;
        display: block;
    }

    /* ── Text inputs ── */
    .stTextInput label { display: none !important; }
    .stTextInput > div > div > input {
        background:  var(--surf-mid) !important;
        border:      1px solid var(--border) !important;
        border-radius: 4px !important;
        color:       var(--text) !important;
        font-family: var(--fbody) !important;
        font-size:   0.88rem !important;
        padding:     0.8rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--primary) !important;
        box-shadow:   0 0 0 3px rgba(142,255,113,.10) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder { color: var(--muted) !important; opacity: .5 !important; }

    /* ── Buttons ── */
    .stButton > button {
        width: 100% !important;
        border-radius: 4px !important;
        font-family: var(--fhead) !important;
        font-size:   0.68rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        padding: 0.82rem !important;
        transition: all .18s !important;
    }
    .stButton > button[kind="primary"] {
        background:  var(--primary) !important;
        border:      none !important;
        color:       #053800 !important;
        box-shadow:  0 0 18px var(--pglow) !important;
        margin-top:  0.5rem !important;
    }
    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.1) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 0 30px var(--pglow) !important;
    }
    .stButton > button[kind="secondary"] {
        background: var(--surf-hi) !important;
        border:     1px solid var(--border) !important;
        color:      var(--muted) !important;
        font-size:  0.58rem !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: var(--surf-mid) !important;
        color:      var(--text) !important;
    }

    /* ── Info cards ── */
    .sc-trial {
        margin-top: 1.4rem;
        padding: .9rem 1rem;
        background: var(--pdim);
        border: 1px solid rgba(142,255,113,.18);
        border-radius: 4px;
        display: flex; gap: .75rem; align-items: flex-start;
    }
    .sc-trial-icon { color: var(--primary); flex-shrink: 0; margin-top: 1px; }
    .sc-trial-title { font-size: .78rem; font-weight: 600; margin-bottom: .2rem; }
    .sc-trial-body  { font-size: .7rem; color: var(--muted); line-height: 1.55; }
    .sc-shariah {
        display: flex; align-items: center; gap: .5rem;
        background: rgba(212,168,83,.07);
        border: 1px solid rgba(212,168,83,.22);
        border-radius: 4px; padding: .65rem .9rem;
        margin: 1rem 0 .5rem;
        font-size: .72rem; color: #d4a853; font-weight: 500;
    }

    /* ── System log ── */
    .sc-log {
        position: fixed; bottom: 1.5rem; right: 1.5rem;
        width: 215px;
        background: rgba(30,30,30,.75);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border);
        border-radius: 6px; padding: .65rem .85rem;
        z-index: 9999;
    }
    @media (max-width: 1100px) { .sc-log { display: none; } }
    .sc-log-row { display: flex; justify-content: space-between; margin-bottom: .35rem; }
    .sc-log-lbl { font-family: var(--fhead); font-size: .48rem; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); }
    .sc-log-live { font-family: var(--fhead); font-size: .48rem; text-transform: uppercase; letter-spacing: .1em; color: var(--primary); }
    .sc-log-line { font-family: var(--fmono); font-size: .46rem; color: #666; margin-bottom: 2px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
    .sc-log-line.ok { color: rgba(142,255,113,.55); }

    /* ── Alerts ── */
    .stAlert { border-radius: 4px !important; font-size: .8rem !important; }

    /* Animation */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — Left panel via components.html (self-contained iframe)
    # CSS on [data-testid="stCustomComponentV1"] pins it position:fixed left
    # ─────────────────────────────────────────────────────────────────────────
    left_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --primary:#8eff71;
  --pglow:rgba(142,255,113,.3);
  --surface:#1a1919;
  --border:rgba(90,88,87,.28);
  --text:#fff;
  --muted:#adaaaa;
  --fh:'Space Grotesk',sans-serif;
  --fm:'JetBrains Mono',monospace;
}}
html,body{{width:100%;height:100%;background:var(--surface);overflow:hidden;}}
.panel{{
  width:100%;height:100%;
  position:relative;
  display:flex;flex-direction:column;justify-content:center;
  padding:3rem 4.5rem;
  overflow:hidden;
  animation:slideIn .5s ease both;
}}
@keyframes slideIn{{from{{opacity:0;transform:translateX(-16px)}}to{{opacity:1;transform:translateX(0)}}}}
.dots{{
  position:absolute;inset:0;pointer-events:none;opacity:.14;
  background-image:radial-gradient(circle at 2px 2px,#555 1px,transparent 0);
  background-size:24px 24px;
}}
/* Gradient overlay edges */
.panel::before{{
  content:'';position:absolute;top:0;right:0;width:60px;height:100%;
  background:linear-gradient(to right,transparent,rgba(14,14,14,.6));
  pointer-events:none;z-index:5;
}}
/* Brand */
.brand{{position:absolute;top:2.2rem;left:4.5rem;}}
.wordmark{{font-family:var(--fh);font-size:1.15rem;font-weight:800;letter-spacing:-.03em;color:var(--primary);}}
.inst{{font-family:var(--fh);font-size:.52rem;text-transform:uppercase;letter-spacing:.18em;color:var(--muted);margin-top:.1rem;}}
/* Hero */
.hero{{position:relative;z-index:2;}}
.hero h2{{font-family:var(--fh);font-size:clamp(1.7rem,3.2vw,2.8rem);font-weight:800;line-height:1.08;letter-spacing:-.04em;color:var(--text);margin-bottom:.9rem;}}
.accent{{color:var(--primary);}}
.hero p{{font-family:var(--fh);color:var(--muted);font-size:.85rem;max-width:320px;line-height:1.65;margin-bottom:1.6rem;}}
/* Features */
.feats{{list-style:none;margin-bottom:1.8rem;}}
.feats li{{display:flex;align-items:center;gap:.55rem;font-family:var(--fh);font-size:.55rem;text-transform:uppercase;letter-spacing:.13em;font-weight:700;color:var(--muted);margin-bottom:.7rem;}}
.dot{{width:6px;height:6px;border-radius:50%;background:var(--primary);box-shadow:0 0 8px var(--pglow);flex-shrink:0;}}
/* Alpha card */
.alpha{{background:rgba(10,10,10,.72);border:1px solid var(--border);border-radius:8px;padding:1.1rem 1.3rem;max-width:340px;backdrop-filter:blur(20px);}}
.a-head{{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:.7rem;}}
.a-lbl{{font-family:var(--fh);font-size:.52rem;text-transform:uppercase;letter-spacing:.13em;color:var(--muted);margin-bottom:.18rem;}}
.a-val{{font-family:var(--fh);font-size:1.45rem;font-weight:700;color:var(--primary);line-height:1;}}
.a-val span{{font-size:.5rem;font-weight:400;color:var(--muted);margin-left:.28rem;}}
.mini-bars{{display:flex;gap:2px;height:44px;align-items:flex-end;}}
.mini-bar{{width:4px;background:var(--primary);border-radius:1px 1px 0 0;}}
.bar-chart{{display:flex;gap:2px;height:64px;align-items:flex-end;margin-top:.4rem;}}
.bar-chart div{{flex:1;background:var(--primary);border-radius:1px 1px 0 0;opacity:.14;}}
.bar-chart div.active{{opacity:1;box-shadow:0 0 8px var(--pglow);}}
.bar-chart div.mid{{opacity:.5;}}
/* Ticker */
.ticker-wrap{{position:absolute;bottom:4rem;left:0;right:0;height:1.5rem;overflow:hidden;border-top:1px solid var(--border);border-bottom:1px solid var(--border);background:rgba(10,10,10,.45);}}
.ticker{{display:flex;align-items:center;height:100%;white-space:nowrap;animation:tick 28s linear infinite;}}
.ti{{margin-right:2.2rem;font-family:var(--fm);font-size:.56rem;color:var(--muted);}}
.ti b{{color:var(--text);font-weight:600;margin-right:.25rem;}}
@keyframes tick{{0%{{transform:translateX(0)}}100%{{transform:translateX(-50%)}}}}
/* Status */
.status{{position:absolute;bottom:1.6rem;left:4.5rem;display:flex;gap:1.4rem;}}
.s-item{{display:flex;align-items:center;gap:.38rem;font-family:var(--fh);font-size:.52rem;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);}}
.pulse{{width:6px;height:6px;border-radius:50%;background:var(--primary);box-shadow:0 0 8px var(--pglow);animation:pulse 2s infinite;}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
</style>
</head>
<body>
<div class="panel">
  <div class="dots"></div>
  <div class="brand">
    <div class="wordmark">STOCKCAST</div>
    <div class="inst">Institutional Access</div>
  </div>
  <div class="hero">
    <h2>Predicting the<br><span class="accent">pulse of global</span><br>markets</h2>
    <p>High-frequency predictive intelligence for professional traders and institutional desk managers.</p>
    <ul class="feats">
      <li><span class="dot"></span>XGBoost Forecasting Engine</li>
      <li><span class="dot"></span>Shariah Compliance Screening</li>
      <li><span class="dot"></span>Institutional-Grade Security</li>
    </ul>
    <div class="alpha">
      <div class="a-head">
        <div>
          <div class="a-lbl">Live Alpha Stream</div>
          <div class="a-val">0.942<span>CONFIDENCE</span></div>
        </div>
        <div class="mini-bars">{_mini_html}</div>
      </div>
      <div class="bar-chart">{_bars_html}</div>
    </div>
  </div>
  <div class="ticker-wrap"><div class="ticker">{_tape}</div></div>
  <div class="status">
    <div class="s-item"><span class="pulse"></span>Engine Online</div>
    <div class="s-item">Uptime 99.99%</div>
  </div>
</div>
</body>
</html>"""

    # Inject left panel — height must be >0 for iframe to render,
    # but CSS pins it to position:fixed covering the full left half
    components.html(left_html, height=1, scrolling=False)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Right panel: tab bar + form widgets
    # ─────────────────────────────────────────────────────────────────────────
    tl = "sc-tab on" if mode == "login"  else "sc-tab"
    ts = "sc-tab on" if mode == "signup" else "sc-tab"
    tr = "sc-tab on" if mode == "reset"  else "sc-tab"

    st.markdown(f"""
    <div class="sc-tabs" style="margin-top:2rem;">
      <span class="{tl}">Terminal Access</span>
      <span class="{ts}">Create Account</span>
      <span class="{tr}">Forgot Key</span>
    </div>
    """, unsafe_allow_html=True)

    # ── LOGIN ────────────────────────────────────────────────────────────────
    if mode == "login":
        st.markdown('<span class="sc-lbl">Identity Token (Email)</span>', unsafe_allow_html=True)
        email = st.text_input("_e", placeholder="operator@stockcast.ai",
                              key="ag_login_email", label_visibility="collapsed")
        st.markdown('<span class="sc-lbl">Access Key</span>', unsafe_allow_html=True)
        password = st.text_input("_p", type="password", placeholder="••••••••••••",
                                 key="ag_login_pw", label_visibility="collapsed")

        if st.button("▶  Authorize Access", type="primary", key="login_btn"):
            if not email or not password:
                st.warning("Please enter your email and password.")
            else:
                try:
                    res = supabase.auth.sign_in_with_password(
                        {"email": email.strip(), "password": password}
                    )
                    if res.user:
                        st.session_state.user = res.user
                        st.success("Authenticated ✓")
                        st.rerun()
                    else:
                        st.error("Login failed — check your credentials.")
                except Exception as e:
                    err = str(e)
                    if "Invalid login credentials" in err or "invalid_credentials" in err:
                        st.error("❌ Invalid email or password.")
                    elif "Email not confirmed" in err:
                        st.warning("📧 Please confirm your email before logging in.")
                    else:
                        st.error(f"Login error: {err}")

        st.markdown("""
        <div class="sc-trial">
          <div class="sc-trial-icon">ℹ</div>
          <div>
            <div class="sc-trial-title">Alpha Stream — Limited Offer</div>
            <div class="sc-trial-body">Initialize your account today and receive a 14-day
            institutional-grade trial with full prediction logs.</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── SIGN UP ──────────────────────────────────────────────────────────────
    elif mode == "signup":
        st.markdown('<span class="sc-lbl">Identity Token (Email)</span>', unsafe_allow_html=True)
        email = st.text_input("_e2", placeholder="operator@stockcast.ai",
                              key="ag_signup_email", label_visibility="collapsed")
        st.markdown('<span class="sc-lbl">Access Key (min 6 chars)</span>', unsafe_allow_html=True)
        password = st.text_input("_p2", type="password", placeholder="••••••••••••",
                                 key="ag_signup_pw", label_visibility="collapsed")
        st.markdown('<span class="sc-lbl">Confirm Access Key</span>', unsafe_allow_html=True)
        password2 = st.text_input("_p3", type="password", placeholder="••••••••••••",
                                  key="ag_signup_pw2", label_visibility="collapsed")
        st.markdown('<div class="sc-shariah">☽&nbsp; Portfolio will be screened for Shariah compliance automatically.</div>',
                    unsafe_allow_html=True)

        if st.button("🚀  Initialize Account", type="primary", key="signup_btn"):
            if not email or not password:
                st.warning("Please fill in all fields.")
            elif len(password) < 6:
                st.warning("Password must be at least 6 characters.")
            elif password != password2:
                st.error("Passwords do not match.")
            else:
                try:
                    res = supabase.auth.sign_up({"email": email.strip(), "password": password})
                    if res.user:
                        st.success("✅ Account created! Check your email to confirm, then log in.")
                        st.session_state._auth_mode = "login"
                        st.rerun()
                    else:
                        st.error("Sign-up failed — please try again.")
                except Exception as e:
                    err = str(e)
                    if "already registered" in err or "already been registered" in err:
                        st.error("❌ This email is already registered. Try logging in instead.")
                    else:
                        st.error(f"Sign-up error: {err}")

    # ── PASSWORD RESET ────────────────────────────────────────────────────────
    elif mode == "reset":
        st.caption("Enter your account email — we'll dispatch a reset link.")
        st.markdown('<span class="sc-lbl">Identity Token (Email)</span>', unsafe_allow_html=True)
        email = st.text_input("_e3", placeholder="operator@stockcast.ai",
                              key="ag_reset_email", label_visibility="collapsed")

        if st.button("📡  Dispatch Reset Link", type="primary", key="reset_btn"):
            if not email:
                st.warning("Please enter your email address.")
            else:
                try:
                    supabase.auth.reset_password_email(email.strip())
                    st.success("Reset link dispatched — check your inbox.")
                except Exception as e:
                    st.error(f"Could not send reset email: {e}")

    # ── Mode switch buttons ───────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.8rem;'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🖥 Terminal Access", key="sw_login", use_container_width=True):
            st.session_state._auth_mode = "login"; st.rerun()
    with c2:
        if st.button("📝 Create Account", key="sw_signup", use_container_width=True):
            st.session_state._auth_mode = "signup"; st.rerun()
    with c3:
        if st.button("🔑 Forgot Key", key="sw_reset", use_container_width=True):
            st.session_state._auth_mode = "reset"; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── System log ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sc-log">
      <div class="sc-log-row">
        <span class="sc-log-lbl">System Logs</span>
        <span class="sc-log-live">● Live</span>
      </div>
      <div class="sc-log-line">Requesting handshake...</div>
      <div class="sc-log-line ok">Access token validated ✓</div>
      <div class="sc-log-line">Encrypted tunnel active</div>
    </div>""", unsafe_allow_html=True)

    st.stop()

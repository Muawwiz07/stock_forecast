"""
authgate.py — StockCast Auth Gate
Handles login, signup, and password reset UI using Supabase Auth.
Called from app.py as: from authgate import render_auth_gate; render_auth_gate(supabase)
Supabase client is passed in — never initialised here (avoids secrets KeyError on import).

Design: Two-column institutional terminal aesthetic (Space Grotesk + Inter, neon-green on near-black).
Left panel: branding + live alpha stream visual.
Right panel: login / signup / reset card.
"""

import streamlit as st


def render_auth_gate(supabase):
    """
    Renders the two-column institutional terminal login / signup / reset UI.
    Supabase client is injected from app.py — no st.secrets access at module level.
    Calls st.stop() if the user is not yet authenticated.
    """

    # ── Session state defaults ─────────────────────────────────────────────────
    if "user" not in st.session_state:
        st.session_state.user = None
    if "_auth_mode" not in st.session_state:
        st.session_state._auth_mode = "login"   # "login" | "signup" | "reset"

    # ── Already authenticated — let app.py continue ────────────────────────────
    if st.session_state.user is not None:
        return

    # ── CSS — two-column institutional terminal theme ──────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg:           #0e0e0e;
        --surface:      #1a1919;
        --surface-high: #201f1f;
        --surface-mid:  #262626;
        --border:       rgba(73,72,71,0.20);
        --primary:      #8eff71;
        --primary-dim:  rgba(142,255,113,0.08);
        --primary-glow: rgba(142,255,113,0.30);
        --text:         #ffffff;
        --muted:        #adaaaa;
        --outline:      #777575;
        --error:        #ff7351;
        --font-head:    'Space Grotesk', sans-serif;
        --font-body:    'Inter', sans-serif;
        --font-mono:    'JetBrains Mono', monospace;
    }

    html, body, [class*="css"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: var(--font-body) !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stSidebar"] { display: none !important; }

    /* ── Full-width two-column layout ── */
    .main .block-container {
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* ── Outer shell ── */
    .sc-shell {
        display: flex;
        min-height: 100vh;
        width: 100%;
        font-family: var(--font-body);
    }

    /* ── Left panel ── */
    .sc-left {
        position: relative;
        width: 50%;
        background: var(--surface);
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 3rem 5rem;
        overflow: hidden;
    }
    @media (max-width: 768px) { .sc-left { display: none; } }

    .sc-dot-grid {
        position: absolute;
        inset: 0;
        opacity: 0.18;
        background-image: radial-gradient(circle at 2px 2px, #494847 1px, transparent 0);
        background-size: 24px 24px;
        pointer-events: none;
    }

    .sc-brand-top {
        position: absolute;
        top: 2rem;
        left: 5rem;
    }
    .sc-wordmark {
        font-family: var(--font-head);
        font-size: 1.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--primary);
    }
    .sc-inst {
        font-family: var(--font-head);
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: var(--muted);
        margin-top: 0.1rem;
    }

    .sc-hero {
        position: relative;
        z-index: 2;
    }
    .sc-hero h2 {
        font-family: var(--font-head);
        font-size: clamp(2.2rem, 4vw, 3.5rem);
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.04em;
        margin: 0 0 1rem;
    }
    .sc-hero h2 .accent { color: var(--primary); }
    .sc-hero p {
        color: var(--muted);
        font-size: 1rem;
        max-width: 360px;
        line-height: 1.6;
        margin-bottom: 2rem;
    }

    .sc-features {
        list-style: none;
        padding: 0;
        margin: 0 0 2.5rem;
        display: flex;
        flex-direction: column;
        gap: 0.9rem;
    }
    .sc-features li {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-family: var(--font-head);
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 600;
    }
    .sc-feat-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--primary);
        box-shadow: 0 0 8px var(--primary-glow);
        flex-shrink: 0;
    }

    /* Alpha stream card */
    .sc-alpha {
        background: rgba(14,14,14,0.7);
        border: 1px solid var(--border);
        border-radius: 0.5rem;
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(20px);
        max-width: 380px;
    }
    .sc-alpha-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        margin-bottom: 1rem;
    }
    .sc-alpha-label {
        font-family: var(--font-head);
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--muted);
        margin-bottom: 0.25rem;
    }
    .sc-alpha-val {
        font-family: var(--font-head);
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--primary);
        line-height: 1;
    }
    .sc-alpha-val span {
        font-size: 0.6rem;
        font-weight: 400;
        color: var(--muted);
        margin-left: 0.35rem;
    }
    .sc-mini-bars {
        display: flex;
        gap: 2px;
        height: 48px;
        align-items: flex-end;
    }
    .sc-mini-bar {
        width: 4px;
        background: var(--primary);
        border-radius: 1px 1px 0 0;
    }
    .sc-bar-chart {
        display: flex;
        gap: 2px;
        height: 80px;
        align-items: flex-end;
        margin-top: 0.5rem;
    }
    .sc-bar-chart div {
        flex: 1;
        background: var(--primary);
        border-radius: 1px 1px 0 0;
        opacity: 0.15;
        transition: opacity 0.3s;
    }
    .sc-bar-chart div.active { opacity: 1; box-shadow: 0 0 8px var(--primary-glow); }
    .sc-bar-chart div.mid    { opacity: 0.55; }

    /* Status bar */
    .sc-status {
        position: absolute;
        bottom: 2rem;
        left: 5rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    .sc-status-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-family: var(--font-head);
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--muted);
    }
    .sc-pulse {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--primary);
        box-shadow: 0 0 8px var(--primary-glow);
        animation: pulse 2s infinite;
    }

    /* ── Right panel ── */
    .sc-right {
        width: 50%;
        background: var(--bg);
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 3rem 6rem;
    }
    @media (max-width: 768px) {
        .sc-right { width: 100%; padding: 2rem 1.5rem; }
    }

    .sc-right-inner { max-width: 400px; width: 100%; margin: 0 auto; }

    /* Mode tabs */
    .sc-tabs {
        display: flex;
        gap: 2rem;
        margin-bottom: 2.5rem;
        border-bottom: 1px solid var(--border);
    }
    .sc-tab {
        padding-bottom: 1rem;
        font-family: var(--font-head);
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-weight: 600;
        color: var(--muted);
        cursor: pointer;
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
        transition: color 0.2s, border-color 0.2s;
        background: none;
        border-top: none;
        border-left: none;
        border-right: none;
    }
    .sc-tab.active {
        color: var(--primary);
        border-bottom-color: var(--primary);
    }
    .sc-tab:hover { color: var(--text); }

    /* Inputs */
    .stTextInput label {
        font-family: var(--font-head) !important;
        font-size: 0.6rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
    }
    .stTextInput > div > div > input {
        background: var(--surface-mid) !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
        color: var(--text) !important;
        font-family: var(--font-body) !important;
        font-size: 0.9rem !important;
        padding: 1rem !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(142,255,113,0.10) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: var(--outline) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 4px !important;
        font-family: var(--font-head) !important;
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        padding: 0.85rem !important;
        transition: filter 0.2s, transform 0.15s, box-shadow 0.2s !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--primary) !important;
        border: none !important;
        color: #064200 !important;
        box-shadow: 0 0 18px var(--primary-glow) !important;
        width: 100% !important;
        margin-top: 0.4rem !important;
    }
    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.12) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 0 28px var(--primary-glow) !important;
    }
    .stButton > button[kind="primary"]:active {
        transform: scale(0.98) !important;
    }
    .stButton > button[kind="secondary"] {
        background: var(--surface-high) !important;
        border: 1px solid var(--border) !important;
        color: var(--muted) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: var(--surface-mid) !important;
        color: var(--text) !important;
    }

    /* Divider */
    .sc-divider {
        position: relative;
        margin: 2rem 0;
        display: flex;
        align-items: center;
    }
    .sc-divider::before {
        content: '';
        flex: 1;
        border-top: 1px solid var(--border);
    }
    .sc-divider::after {
        content: '';
        flex: 1;
        border-top: 1px solid var(--border);
    }
    .sc-divider span {
        padding: 0 1rem;
        font-family: var(--font-head);
        font-size: 0.55rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--outline);
    }

    /* Shariah badge */
    .sc-shariah {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        background: rgba(212,168,83,0.07);
        border: 1px solid rgba(212,168,83,0.20);
        border-radius: 4px;
        padding: 0.7rem 0.9rem;
        margin: 1rem 0 0.5rem;
        font-family: var(--font-body);
        font-size: 0.75rem;
        color: #d4a853;
        font-weight: 500;
    }

    /* Trial banner */
    .sc-trial {
        margin-top: 2rem;
        padding: 1rem 1.1rem;
        background: var(--primary-dim);
        border: 1px solid rgba(142,255,113,0.18);
        border-radius: 4px;
        display: flex;
        gap: 0.8rem;
        align-items: flex-start;
    }
    .sc-trial-icon { color: var(--primary); font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
    .sc-trial-title { font-size: 0.8rem; font-weight: 600; color: var(--text); margin-bottom: 0.2rem; }
    .sc-trial-body  { font-size: 0.72rem; color: var(--muted); line-height: 1.5; }

    /* System log widget */
    .sc-log {
        position: fixed;
        bottom: 1.5rem;
        right: 1.5rem;
        width: 240px;
        background: rgba(38,38,38,0.6);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.75rem 0.9rem;
    }
    @media (max-width: 1024px) { .sc-log { display: none; } }
    .sc-log-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .sc-log-label { font-family: var(--font-head); font-size: 0.5rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); }
    .sc-log-live  { font-family: var(--font-head); font-size: 0.5rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--primary); }
    .sc-log-line  { font-family: var(--font-mono); font-size: 0.5rem; color: var(--outline); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 2px; }
    .sc-log-line.ok { color: rgba(142,255,113,0.55); }

    /* Alerts */
    .stAlert { border-radius: 4px !important; font-size: 0.82rem !important; }

    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--primary-glow); }
        50%       { opacity: 0.4; box-shadow: none; }
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .sc-shell { animation: fadeIn 0.45s ease both; }
    </style>
    """, unsafe_allow_html=True)

    mode = st.session_state._auth_mode

    # ── Ticker tape — live via yfinance, graceful fallback ─────────────────────
    _tape_syms = ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN", "NVDA", "META", "JPM", "V", "BRK-B"]
    _tape_items = []
    try:
        import yfinance as _yf
        _td = _yf.download(_tape_syms, period="2d", interval="1d", progress=False, auto_adjust=True)
        _closes = _td["Close"]
        for _s in _tape_syms:
            try:
                _row = _closes[_s].dropna()
                if len(_row) >= 2:
                    _px   = float(_row.iloc[-1])
                    _prev = float(_row.iloc[-2])
                    _chg  = (_px - _prev) / _prev * 100
                    _col  = "#8eff71" if _chg >= 0 else "#ff7351"
                    _arr  = "▲" if _chg >= 0 else "▼"
                    _tape_items.append(
                        f'<span style="margin-right:2rem; font-family:var(--font-mono); font-size:0.62rem;">'
                        f'<span style="color:var(--text);font-weight:600">{_s}</span> '
                        f'{_px:.2f} <span style="color:{_col}">{_arr} {abs(_chg):.2f}%</span></span>'
                    )
            except Exception:
                pass
    except Exception:
        pass

    if not _tape_items:
        for _s in ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN", "NVDA", "META"]:
            _tape_items.append(
                f'<span style="margin-right:2rem; font-family:var(--font-mono); font-size:0.62rem;">'
                f'<span style="color:var(--text);font-weight:600">{_s}</span> '
                f'<span style="color:#8eff71">Loading…</span></span>'
            )

    _tape_html = "".join(_tape_items * 2)

    # ── Build alpha stream bar heights ─────────────────────────────────────────
    _bar_heights = [20, 35, 45, 25, 60, 80, 70, 90, 100, 85, 65, 40, 20]
    _bars_html = ""
    for i, h in enumerate(_bar_heights):
        _cls = "active" if h == 100 else ("mid" if h >= 60 else "")
        _bars_html += f'<div class="{_cls}" style="height:{h}%"></div>'

    _mini_bar_data = [30, 60, 45, 80, 100, 70, 40]
    _mini_html = "".join(
        f'<div class="sc-mini-bar" style="height:{int(h*48/100)}px; opacity:{0.2 + 0.8*(h/100):.2f}"></div>'
        for h in _mini_bar_data
    )

    # ── Tab highlight helper ───────────────────────────────────────────────────
    def _tab_cls(t): return "sc-tab active" if mode == t else "sc-tab"

    # ── Full HTML shell ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sc-shell">

      <!-- ── LEFT PANEL ── -->
      <div class="sc-left">
        <div class="sc-dot-grid"></div>

        <div class="sc-brand-top">
          <div class="sc-wordmark">STOCKCAST</div>
          <div class="sc-inst">Institutional Access</div>
        </div>

        <div class="sc-hero">
          <h2>Predicting the<br><span class="accent">pulse of global</span><br>markets</h2>
          <p>High-frequency predictive intelligence for professional traders and institutional desk managers.</p>

          <ul class="sc-features">
            <li><span class="sc-feat-dot"></span>XGBoost Forecasting Engine</li>
            <li><span class="sc-feat-dot"></span>Shariah Compliance Screening</li>
            <li><span class="sc-feat-dot"></span>Institutional-Grade Security</li>
          </ul>

          <!-- Alpha stream card -->
          <div class="sc-alpha">
            <div class="sc-alpha-head">
              <div>
                <div class="sc-alpha-label">Live Alpha Stream</div>
                <div class="sc-alpha-val">0.942<span>CONFIDENCE</span></div>
              </div>
              <div class="sc-mini-bars">{_mini_html}</div>
            </div>
            <div class="sc-bar-chart">{_bars_html}</div>
          </div>
        </div>

        <div class="sc-status">
          <div class="sc-status-item"><span class="sc-pulse"></span> Engine Online</div>
          <div class="sc-status-item">Uptime 99.99%</div>
        </div>
      </div>

      <!-- ── RIGHT PANEL ── -->
      <div class="sc-right">
        <div class="sc-right-inner">

          <!-- Tabs — visual only; real switching done by hidden Streamlit buttons below -->
          <div class="sc-tabs">
            <div class="{_tab_cls('login')}"  onclick="(function(){{var btns=Array.from(document.querySelectorAll('button')).filter(function(b){{return b.innerText.trim()==='Terminal Access'}}); if(btns.length)btns[btns.length-1].click();}})()">Terminal Access</div>
            <div class="{_tab_cls('signup')}" onclick="(function(){{var btns=Array.from(document.querySelectorAll('button')).filter(function(b){{return b.innerText.trim()==='Create Account'}}); if(btns.length)btns[btns.length-1].click();}})()">Create Account</div>
            <div class="{_tab_cls('reset')}"  onclick="(function(){{var btns=Array.from(document.querySelectorAll('button')).filter(function(b){{return b.innerText.trim()==='Forgot Key'}}); if(btns.length)btns[btns.length-1].click();}})()">Forgot Key</div>
          </div>

    """, unsafe_allow_html=True)

    # ── LOGIN ──────────────────────────────────────────────────────────────────
    if mode == "login":
        st.markdown("""
        <div style="margin-bottom:0.2rem">
          <div style="font-family:var(--font-head);font-size:0.6rem;text-transform:uppercase;
                      letter-spacing:0.14em;color:var(--muted);margin-bottom:0.4rem;">
            Identity Token (Email)
          </div>
        </div>
        """, unsafe_allow_html=True)
        email    = st.text_input("Email address", placeholder="operator@stockcast.ai", key="ag_login_email",
                                 label_visibility="collapsed")
        st.markdown("""
        <div style="margin-bottom:0.2rem;margin-top:0.8rem">
          <div style="font-family:var(--font-head);font-size:0.6rem;text-transform:uppercase;
                      letter-spacing:0.14em;color:var(--muted);margin-bottom:0.4rem;">
            Access Key
          </div>
        </div>
        """, unsafe_allow_html=True)
        password = st.text_input("Password", type="password", placeholder="••••••••••••",
                                 key="ag_login_pw", label_visibility="collapsed")

        if st.button("▶  Authorize Access", use_container_width=True, type="primary", key="login_btn"):
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

        # Trial banner
        st.markdown("""
        <div class="sc-trial">
          <div class="sc-trial-icon">ℹ</div>
          <div>
            <div class="sc-trial-title">Alpha Stream Limited Offer</div>
            <div class="sc-trial-body">Initialize your account today and receive a 14-day
            institutional-grade trial with full prediction logs.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── SIGN UP ────────────────────────────────────────────────────────────────
    elif mode == "signup":
        st.markdown("""
        <div style="margin-bottom:0.2rem">
          <div style="font-family:var(--font-head);font-size:0.6rem;text-transform:uppercase;
                      letter-spacing:0.14em;color:var(--muted);margin-bottom:0.4rem;">
            Identity Token (Email)
          </div>
        </div>
        """, unsafe_allow_html=True)
        email = st.text_input("Email", placeholder="operator@stockcast.ai",
                              key="ag_signup_email", label_visibility="collapsed")
        st.markdown("""<div style="height:0.6rem"></div>""", unsafe_allow_html=True)
        password = st.text_input("Password (min 6 chars)", type="password",
                                 placeholder="••••••••••••", key="ag_signup_pw")
        password2 = st.text_input("Confirm password", type="password",
                                  placeholder="••••••••••••", key="ag_signup_pw2")

        st.markdown("""
        <div class="sc-shariah">
            ☽&nbsp; Your portfolio will be screened for Shariah compliance automatically.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀  Initialize Account", use_container_width=True, type="primary", key="signup_btn"):
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

    # ── PASSWORD RESET ─────────────────────────────────────────────────────────
    elif mode == "reset":
        st.caption("Enter your account email and we'll dispatch a reset link.")
        st.markdown("""
        <div style="margin-bottom:0.2rem">
          <div style="font-family:var(--font-head);font-size:0.6rem;text-transform:uppercase;
                      letter-spacing:0.14em;color:var(--muted);margin-bottom:0.4rem;">
            Identity Token (Email)
          </div>
        </div>
        """, unsafe_allow_html=True)
        email = st.text_input("Email", placeholder="operator@stockcast.ai",
                              key="ag_reset_email", label_visibility="collapsed")

        if st.button("📡  Dispatch Reset Link", use_container_width=True, type="primary", key="reset_btn"):
            if not email:
                st.warning("Please enter your email address.")
            else:
                try:
                    supabase.auth.reset_password_email(email.strip())
                    st.success("Reset link dispatched — check your inbox.")
                except Exception as e:
                    st.error(f"Could not send reset email: {e}")

    # ── Close right panel shell ────────────────────────────────────────────────
    st.markdown("""
        </div><!-- sc-right-inner -->
      </div><!-- sc-right -->
    </div><!-- sc-shell -->

    <!-- System log widget -->
    <div class="sc-log">
      <div class="sc-log-head">
        <span class="sc-log-label">System Logs</span>
        <span class="sc-log-live">Live</span>
      </div>
      <div class="sc-log-line">09:44:21 — Requesting handshake...</div>
      <div class="sc-log-line ok">09:44:22 — Access token validated</div>
      <div class="sc-log-line">09:44:22 — Encrypted tunnel active</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Mode switch buttons (visually hidden; triggered by JS from the HTML tabs) ──
    # Hidden Streamlit buttons — positioned off-screen, triggered by JS tab clicks
    # Using visibility:hidden (not display:none / height:0) so JS .click() still works
    st.markdown("""
    <style>
    .sc-real-tabs {
        position: fixed;
        top: -9999px;
        left: -9999px;
        visibility: hidden;
        width: 0;
        height: 0;
        overflow: hidden;
    }
    </style>
    <div class="sc-real-tabs" id="sc-real-tabs">
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Terminal Access", key="tab_login"):
            st.session_state._auth_mode = "login"; st.rerun()
    with c2:
        if st.button("Create Account", key="tab_signup"):
            st.session_state._auth_mode = "signup"; st.rerun()
    with c3:
        if st.button("Forgot Key", key="tab_reset"):
            st.session_state._auth_mode = "reset"; st.rerun()

    # ── Block app.py from rendering until authenticated ────────────────────────
    st.stop()

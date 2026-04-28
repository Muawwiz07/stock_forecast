"""
authgate.py — StockCast Auth Gate
Handles login, signup, and password reset UI using Supabase Auth.
Called from app.py as: from authgate import render_auth_gate; render_auth_gate(supabase)
Supabase client is passed in — never initialised here (avoids secrets KeyError on import).

Architecture fix: Left panel injected as a position:fixed overlay via st.components.v1.html().
Right panel is normal Streamlit flow with CSS styling.
This avoids the "split HTML across st.markdown blocks" DOM corruption bug.
"""

import streamlit as st
import streamlit.components.v1 as components


def render_auth_gate(supabase):
    """
    Renders the two-column institutional terminal login / signup / reset UI.
    Supabase client is injected from app.py — no st.secrets access at module level.
    Calls st.stop() if the user is not yet authenticated.
    """

    # ── Session state defaults ────────────────────────────────────────────────
    if "user" not in st.session_state:
        st.session_state.user = None
    if "_auth_mode" not in st.session_state:
        st.session_state._auth_mode = "login"

    # ── Already authenticated — let app.py continue ───────────────────────────
    if st.session_state.user is not None:
        return

    mode = st.session_state._auth_mode

    # ── Build alpha stream bar HTML ───────────────────────────────────────────
    _bar_heights = [20, 35, 45, 25, 60, 80, 70, 90, 100, 85, 65, 40, 20]
    _bars_html = ""
    for h in _bar_heights:
        _cls = "active" if h == 100 else ("mid" if h >= 60 else "")
        _bars_html += f'<div class="{_cls}" style="height:{h}%"></div>'

    _mini_bar_data = [30, 60, 45, 80, 100, 70, 40]
    _mini_html = "".join(
        f'<div class="sc-mini-bar" style="height:{int(h * 48 / 100)}px; opacity:{0.2 + 0.8 * (h / 100):.2f}"></div>'
        for h in _mini_bar_data
    )

    # ── Fetch ticker data with fallback ──────────────────────────────────────
    _tape_items = []
    try:
        import yfinance as _yf
        _syms = ["AAPL", "MSFT", "TSLA", "GOOG", "NVDA", "META", "JPM"]
        _td = _yf.download(_syms, period="2d", interval="1d", progress=False, auto_adjust=True)
        _closes = _td["Close"]
        for _s in _syms:
            try:
                _row = _closes[_s].dropna()
                if len(_row) >= 2:
                    _px   = float(_row.iloc[-1])
                    _prev = float(_row.iloc[-2])
                    _chg  = (_px - _prev) / _prev * 100
                    _col  = "#8eff71" if _chg >= 0 else "#ff7351"
                    _arr  = "▲" if _chg >= 0 else "▼"
                    _tape_items.append(
                        f'<span class="tick-item"><span class="tick-sym">{_s}</span>'
                        f' {_px:.2f} <span style="color:{_col}">{_arr} {abs(_chg):.2f}%</span></span>'
                    )
            except Exception:
                pass
    except Exception:
        pass

    if not _tape_items:
        for _s in ["AAPL", "MSFT", "TSLA", "GOOG", "NVDA", "META", "JPM"]:
            _tape_items.append(
                f'<span class="tick-item"><span class="tick-sym">{_s}</span>'
                f' <span style="color:#8eff71">Loading…</span></span>'
            )

    _tape_html = "".join(_tape_items * 3)

    # ── GLOBAL CSS (right panel + full page) ─────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg:           #0e0e0e;
        --surface:      #1a1919;
        --surface-high: #201f1f;
        --surface-mid:  #262626;
        --border:       rgba(73,72,71,0.30);
        --primary:      #8eff71;
        --primary-dim:  rgba(142,255,113,0.08);
        --primary-glow: rgba(142,255,113,0.30);
        --text:         #ffffff;
        --muted:        #adaaaa;
        --outline:      #777575;
        --font-head:    'Space Grotesk', sans-serif;
        --font-body:    'Inter', sans-serif;
        --font-mono:    'JetBrains Mono', monospace;
    }

    html, body, [class*="css"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: var(--font-body) !important;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header, [data-testid="stToolbar"],
    [data-testid="stDecoration"], .stDeployButton { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }

    /* Page padding — push right panel content away from left overlay */
    .main .block-container {
        max-width: 700px !important;
        margin-left: 50% !important;
        padding: 0 4rem !important;
        padding-top: 0 !important;
    }

    /* Right panel full height centering */
    .sc-right-flow {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 3rem 0;
    }

    /* Tab bar */
    .sc-tabs {
        display: flex;
        gap: 2rem;
        margin-bottom: 2.5rem;
        border-bottom: 1px solid var(--border);
    }
    .sc-tab-item {
        padding-bottom: 1rem;
        font-family: var(--font-head);
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-weight: 600;
        color: var(--muted);
        margin-bottom: -1px;
    }
    .sc-tab-item.active {
        color: var(--primary);
        border-bottom: 2px solid var(--primary);
    }

    /* Input labels */
    .sc-label {
        font-family: var(--font-head);
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--muted);
        font-weight: 600;
        margin-bottom: 0.35rem;
        margin-top: 0.8rem;
        display: block;
    }

    /* Override Streamlit inputs */
    .stTextInput label { display: none !important; }
    .stTextInput > div > div > input {
        background: var(--surface-mid) !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
        color: var(--text) !important;
        font-family: var(--font-body) !important;
        font-size: 0.9rem !important;
        padding: 0.85rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(142,255,113,0.10) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder { color: var(--outline) !important; }

    /* Buttons */
    .stButton > button {
        border-radius: 4px !important;
        font-family: var(--font-head) !important;
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        padding: 0.85rem !important;
        transition: all 0.2s !important;
        width: 100% !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--primary) !important;
        border: none !important;
        color: #064200 !important;
        box-shadow: 0 0 18px var(--primary-glow) !important;
        margin-top: 0.5rem !important;
    }
    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.12) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 0 28px var(--primary-glow) !important;
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
        font-size: 0.75rem;
        color: #d4a853;
        font-weight: 500;
    }

    /* Trial banner */
    .sc-trial {
        margin-top: 1.5rem;
        padding: 1rem 1.1rem;
        background: var(--primary-dim);
        border: 1px solid rgba(142,255,113,0.18);
        border-radius: 4px;
        display: flex;
        gap: 0.8rem;
        align-items: flex-start;
    }
    .sc-trial-icon { color: var(--primary); font-size: 1rem; flex-shrink: 0; }
    .sc-trial-title { font-size: 0.8rem; font-weight: 600; color: var(--text); margin-bottom: 0.25rem; }
    .sc-trial-body  { font-size: 0.72rem; color: var(--muted); line-height: 1.5; }

    /* Hidden tab switcher buttons */
    .sc-real-tabs {
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        overflow: hidden !important;
        clip: rect(0,0,0,0) !important;
        white-space: nowrap !important;
    }

    /* Alerts */
    .stAlert { border-radius: 4px !important; font-size: 0.82rem !important; }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .sc-right-flow { animation: fadeIn 0.4s ease both; }
    </style>
    """, unsafe_allow_html=True)

    # ── LEFT PANEL — injected as fixed overlay via components.html ───────────
    # This avoids the "split HTML across st.markdown wrappers" DOM corruption bug.
    left_panel_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
        --bg:           #0e0e0e;
        --surface:      #1a1919;
        --border:       rgba(73,72,71,0.25);
        --primary:      #8eff71;
        --primary-glow: rgba(142,255,113,0.30);
        --text:         #ffffff;
        --muted:        #adaaaa;
        --font-head:    'Space Grotesk', sans-serif;
        --font-mono:    'JetBrains Mono', monospace;
    }}
    html, body {{
        height: 100%;
        background: transparent;
        overflow: hidden;
    }}
    .panel {{
        position: fixed;
        top: 0; left: 0;
        width: 50vw;
        height: 100vh;
        background: var(--surface);
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 3rem 5rem;
        overflow: hidden;
        animation: fadeIn 0.5s ease both;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateX(-20px); }}
        to   {{ opacity: 1; transform: translateX(0); }}
    }}
    .dot-grid {{
        position: absolute;
        inset: 0;
        opacity: 0.15;
        background-image: radial-gradient(circle at 2px 2px, #494847 1px, transparent 0);
        background-size: 24px 24px;
        pointer-events: none;
    }}
    .brand {{
        position: absolute;
        top: 2.5rem;
        left: 5rem;
    }}
    .wordmark {{
        font-family: var(--font-head);
        font-size: 1.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--primary);
    }}
    .inst {{
        font-size: 0.55rem;
        font-family: var(--font-head);
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: var(--muted);
        margin-top: 0.1rem;
    }}
    .hero {{ position: relative; z-index: 2; }}
    .hero h2 {{
        font-family: var(--font-head);
        font-size: clamp(1.8rem, 3.5vw, 3rem);
        font-weight: 800;
        line-height: 1.08;
        letter-spacing: -0.04em;
        color: var(--text);
        margin-bottom: 1rem;
    }}
    .hero h2 .accent {{ color: var(--primary); }}
    .hero p {{
        color: var(--muted);
        font-size: 0.9rem;
        max-width: 340px;
        line-height: 1.65;
        margin-bottom: 1.8rem;
        font-family: var(--font-head);
    }}
    .features {{ list-style: none; margin-bottom: 2rem; }}
    .features li {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-family: var(--font-head);
        font-size: 0.58rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 600;
        color: var(--muted);
        margin-bottom: 0.75rem;
    }}
    .feat-dot {{
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--primary);
        box-shadow: 0 0 8px var(--primary-glow);
        flex-shrink: 0;
    }}
    /* Alpha card */
    .alpha {{
        background: rgba(14,14,14,0.75);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        max-width: 360px;
        backdrop-filter: blur(20px);
    }}
    .alpha-head {{
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        margin-bottom: 0.75rem;
    }}
    .alpha-label {{
        font-family: var(--font-head);
        font-size: 0.55rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--muted);
        margin-bottom: 0.2rem;
    }}
    .alpha-val {{
        font-family: var(--font-head);
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary);
        line-height: 1;
    }}
    .alpha-val span {{
        font-size: 0.55rem;
        font-weight: 400;
        color: var(--muted);
        margin-left: 0.3rem;
    }}
    .mini-bars {{
        display: flex;
        gap: 2px;
        height: 44px;
        align-items: flex-end;
    }}
    .mini-bar {{
        width: 4px;
        background: var(--primary);
        border-radius: 1px 1px 0 0;
    }}
    .bar-chart {{
        display: flex;
        gap: 2px;
        height: 70px;
        align-items: flex-end;
    }}
    .bar-chart div {{
        flex: 1;
        background: var(--primary);
        border-radius: 1px 1px 0 0;
        opacity: 0.15;
    }}
    .bar-chart div.active {{ opacity: 1; box-shadow: 0 0 8px var(--primary-glow); }}
    .bar-chart div.mid    {{ opacity: 0.5; }}
    /* Status */
    .status {{
        position: absolute;
        bottom: 2rem; left: 5rem;
        display: flex;
        gap: 1.5rem;
    }}
    .status-item {{
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-family: var(--font-head);
        font-size: 0.55rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--muted);
    }}
    .pulse {{
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--primary);
        box-shadow: 0 0 8px var(--primary-glow);
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50%       {{ opacity: 0.35; }}
    }}
    /* Ticker tape */
    .ticker-wrap {{
        position: absolute;
        bottom: 4.5rem; left: 0; right: 0;
        overflow: hidden;
        height: 1.6rem;
        border-top: 1px solid var(--border);
        border-bottom: 1px solid var(--border);
        background: rgba(14,14,14,0.5);
    }}
    .ticker {{
        display: flex;
        align-items: center;
        height: 100%;
        white-space: nowrap;
        animation: ticker 30s linear infinite;
    }}
    .tick-item {{
        margin-right: 2.5rem;
        font-family: var(--font-mono);
        font-size: 0.58rem;
        color: var(--muted);
    }}
    .tick-sym {{ color: var(--text); font-weight: 600; margin-right: 0.3rem; }}
    @keyframes ticker {{
        0%   {{ transform: translateX(0); }}
        100% {{ transform: translateX(-50%); }}
    }}
    </style>
    </head>
    <body>
    <div class="panel">
      <div class="dot-grid"></div>
      <div class="brand">
        <div class="wordmark">STOCKCAST</div>
        <div class="inst">Institutional Access</div>
      </div>
      <div class="hero">
        <h2>Predicting the<br><span class="accent">pulse of global</span><br>markets</h2>
        <p>High-frequency predictive intelligence for professional traders and institutional desk managers.</p>
        <ul class="features">
          <li><span class="feat-dot"></span>XGBoost Forecasting Engine</li>
          <li><span class="feat-dot"></span>Shariah Compliance Screening</li>
          <li><span class="feat-dot"></span>Institutional-Grade Security</li>
        </ul>
        <div class="alpha">
          <div class="alpha-head">
            <div>
              <div class="alpha-label">Live Alpha Stream</div>
              <div class="alpha-val">0.942<span>CONFIDENCE</span></div>
            </div>
            <div class="mini-bars">{_mini_html}</div>
          </div>
          <div class="bar-chart">{_bars_html}</div>
        </div>
      </div>
      <div class="ticker-wrap">
        <div class="ticker">{_tape_html}</div>
      </div>
      <div class="status">
        <div class="status-item"><span class="pulse"></span> Engine Online</div>
        <div class="status-item">Uptime 99.99%</div>
      </div>
    </div>
    </body>
    </html>
    """

    # Inject left panel as a fixed overlay — height=0 so it takes no Streamlit space
    components.html(left_panel_html, height=0, scrolling=False)

    # ── RIGHT PANEL — pure Streamlit widgets ──────────────────────────────────
    # Tab bar (visual)
    tab_login  = "sc-tab-item active" if mode == "login"  else "sc-tab-item"
    tab_signup = "sc-tab-item active" if mode == "signup" else "sc-tab-item"
    tab_reset  = "sc-tab-item active" if mode == "reset"  else "sc-tab-item"

    st.markdown(f"""
    <div class="sc-right-flow">
      <div class="sc-tabs">
        <span class="{tab_login}">Terminal Access</span>
        <span class="{tab_signup}">Create Account</span>
        <span class="{tab_reset}">Forgot Key</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tab switcher (hidden, triggered by visible buttons below form) ────────
    st.markdown('<div class="sc-real-tabs">', unsafe_allow_html=True)
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        if st.button("Terminal Access", key="tab_login", use_container_width=True):
            st.session_state._auth_mode = "login"; st.rerun()
    with tc2:
        if st.button("Create Account", key="tab_signup", use_container_width=True):
            st.session_state._auth_mode = "signup"; st.rerun()
    with tc3:
        if st.button("Forgot Key", key="tab_reset", use_container_width=True):
            st.session_state._auth_mode = "reset"; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── LOGIN ─────────────────────────────────────────────────────────────────
    if mode == "login":
        st.markdown('<span class="sc-label">Identity Token (Email)</span>', unsafe_allow_html=True)
        email = st.text_input("email_login", placeholder="operator@stockcast.ai",
                              key="ag_login_email", label_visibility="collapsed")
        st.markdown('<span class="sc-label">Access Key</span>', unsafe_allow_html=True)
        password = st.text_input("pw_login", type="password", placeholder="••••••••••••",
                                 key="ag_login_pw", label_visibility="collapsed")

        if st.button("▶  Authorize Access", use_container_width=True,
                     type="primary", key="login_btn"):
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
            <div class="sc-trial-title">Alpha Stream Limited Offer</div>
            <div class="sc-trial-body">Initialize your account today and receive a 14-day
            institutional-grade trial with full prediction logs.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── SIGN UP ───────────────────────────────────────────────────────────────
    elif mode == "signup":
        st.markdown('<span class="sc-label">Identity Token (Email)</span>', unsafe_allow_html=True)
        email = st.text_input("email_signup", placeholder="operator@stockcast.ai",
                              key="ag_signup_email", label_visibility="collapsed")
        st.markdown('<span class="sc-label">Access Key (min 6 chars)</span>', unsafe_allow_html=True)
        password = st.text_input("pw_signup", type="password", placeholder="••••••••••••",
                                 key="ag_signup_pw", label_visibility="collapsed")
        st.markdown('<span class="sc-label">Confirm Access Key</span>', unsafe_allow_html=True)
        password2 = st.text_input("pw_signup2", type="password", placeholder="••••••••••••",
                                  key="ag_signup_pw2", label_visibility="collapsed")

        st.markdown("""
        <div class="sc-shariah">
            ☽&nbsp; Your portfolio will be screened for Shariah compliance automatically.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀  Initialize Account", use_container_width=True,
                     type="primary", key="signup_btn"):
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
        st.caption("Enter your account email and we'll dispatch a reset link.")
        st.markdown('<span class="sc-label">Identity Token (Email)</span>', unsafe_allow_html=True)
        email = st.text_input("email_reset", placeholder="operator@stockcast.ai",
                              key="ag_reset_email", label_visibility="collapsed")

        if st.button("📡  Dispatch Reset Link", use_container_width=True,
                     type="primary", key="reset_btn"):
            if not email:
                st.warning("Please enter your email address.")
            else:
                try:
                    supabase.auth.reset_password_email(email.strip())
                    st.success("Reset link dispatched — check your inbox.")
                except Exception as e:
                    st.error(f"Could not send reset email: {e}")

    # ── Visible tab-switch buttons (below form, always accessible) ────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Switch mode:")
    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        if st.button("🖥 Login", key="vis_login", use_container_width=True):
            st.session_state._auth_mode = "login"; st.rerun()
    with vc2:
        if st.button("📝 Sign Up", key="vis_signup", use_container_width=True):
            st.session_state._auth_mode = "signup"; st.rerun()
    with vc3:
        if st.button("🔑 Reset", key="vis_reset", use_container_width=True):
            st.session_state._auth_mode = "reset"; st.rerun()

    # System log widget
    st.markdown("""
    <style>
    .sc-log {
        position: fixed;
        bottom: 1.5rem; right: 1.5rem;
        width: 220px;
        background: rgba(38,38,38,0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(73,72,71,0.30);
        border-radius: 6px;
        padding: 0.7rem 0.9rem;
        z-index: 9999;
    }
    @media (max-width: 1100px) { .sc-log { display: none; } }
    .sc-log-head {
        display: flex; justify-content: space-between;
        align-items: center; margin-bottom: 0.4rem;
    }
    .sc-log-label { font-family:'Space Grotesk',sans-serif; font-size:0.48rem; text-transform:uppercase; letter-spacing:0.12em; color:#adaaaa; }
    .sc-log-live  { font-family:'Space Grotesk',sans-serif; font-size:0.48rem; text-transform:uppercase; letter-spacing:0.12em; color:#8eff71; }
    .sc-log-line  { font-family:'JetBrains Mono',monospace; font-size:0.48rem; color:#777575; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-bottom:2px; }
    .sc-log-line.ok { color:rgba(142,255,113,0.6); }
    </style>
    <div class="sc-log">
      <div class="sc-log-head">
        <span class="sc-log-label">System Logs</span>
        <span class="sc-log-live">● Live</span>
      </div>
      <div class="sc-log-line">Requesting handshake...</div>
      <div class="sc-log-line ok">Access token validated ✓</div>
      <div class="sc-log-line">Encrypted tunnel active</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Block app.py from rendering ───────────────────────────────────────────
    st.stop()

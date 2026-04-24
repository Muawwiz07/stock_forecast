"""
authgate.py — StockCast Auth Gate
Handles login, signup, and password reset UI using Supabase Auth.
Called from app.py as: from authgate import render_auth_gate; render_auth_gate(supabase)
Supabase client is passed in — never initialised here (avoids secrets KeyError on import).
"""

import streamlit as st


def render_auth_gate(supabase):
    """
    Renders the portrait login / signup / reset UI.
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

    # ── CSS — portrait card, dark fintech theme ────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg:          #0b0f1a;
        --surface:     #111827;
        --border:      #1e2d45;
        --accent:      #3b82f6;
        --accent-light:#60a5fa;
        --accent-dim:  rgba(59,130,246,0.12);
        --gold:        #d4a853;
        --gold-dim:    rgba(212,168,83,0.10);
        --text:        #e8edf5;
        --muted:       #64748b;
        --danger:      #ef4444;
        --success:     #22c55e;
        --font-serif:  'DM Serif Display', Georgia, serif;
        --font-sans:   'DM Sans', sans-serif;
        --font-mono:   'JetBrains Mono', monospace;
    }

    html, body, [class*="css"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: var(--font-sans) !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stSidebar"] { display: none !important; }

    /* Portrait card constraint */
    .main .block-container {
        max-width: 420px !important;
        padding: 2rem 1.5rem 3rem !important;
        margin: 0 auto !important;
    }

    /* Logo */
    .sc-logo-wrap {
        text-align: center;
        margin-bottom: 1.6rem;
        animation: fadeDown 0.55s ease both;
    }
    .sc-logo-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 52px; height: 52px;
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%);
        border-radius: 14px;
        margin-bottom: 0.8rem;
        box-shadow: 0 0 28px rgba(59,130,246,0.35);
        font-size: 1.5rem;
    }
    .sc-wordmark {
        font-family: var(--font-serif);
        font-size: 1.9rem;
        letter-spacing: -0.02em;
        color: var(--text);
        display: block;
        line-height: 1;
    }
    .sc-wordmark span { color: var(--accent-light); }
    .sc-tagline {
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--muted);
        margin-top: 0.3rem;
        display: block;
        font-weight: 500;
    }

    /* Ticker tape */
    .sc-ticker-wrap {
        overflow: hidden;
        margin-bottom: 1.4rem;
        border-top: 1px solid var(--border);
        border-bottom: 1px solid var(--border);
        padding: 0.4rem 0;
        animation: fadeUp 0.5s ease both;
        animation-delay: 0.05s;
    }
    .sc-ticker {
        display: flex;
        gap: 2rem;
        animation: tickerScroll 20s linear infinite;
        white-space: nowrap;
        font-family: var(--font-mono);
        font-size: 0.7rem;
        color: var(--muted);
    }
    .sc-ticker .up   { color: var(--success); }
    .sc-ticker .down { color: var(--danger); }
    .sc-ticker .sym  { color: var(--text); font-weight: 500; }

    /* Card */
    .sc-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.8rem 1.6rem;
        box-shadow: 0 20px 56px rgba(0,0,0,0.45),
                    0 0 0 1px rgba(255,255,255,0.03) inset;
        animation: fadeUp 0.5s ease both;
        animation-delay: 0.1s;
    }

    /* Inputs */
    .stTextInput label {
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
    }
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid var(--border) !important;
        border-radius: 9px !important;
        color: var(--text) !important;
        font-family: var(--font-sans) !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 0.8rem !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-dim) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: var(--muted) !important;
        opacity: 0.55 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px !important;
        font-family: var(--font-sans) !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        padding: 0.6rem !important;
        transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%) !important;
        border: none !important;
        color: #fff !important;
        box-shadow: 0 4px 14px rgba(59,130,246,0.35) !important;
        width: 100% !important;
        margin-top: 0.3rem !important;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(59,130,246,0.45) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid var(--border) !important;
        color: var(--muted) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: var(--accent) !important;
        color: var(--accent-light) !important;
    }

    /* Shariah badge */
    .sc-shariah {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        background: var(--gold-dim);
        border: 1px solid rgba(212,168,83,0.22);
        border-radius: 9px;
        padding: 0.55rem 0.8rem;
        margin: 1.2rem 0 0.4rem;
        font-size: 0.76rem;
        color: var(--gold);
        font-weight: 500;
    }

    /* Footer */
    .sc-footer {
        text-align: center;
        font-size: 0.7rem;
        color: var(--muted);
        margin-top: 1.6rem;
        line-height: 1.7;
        animation: fadeUp 0.6s ease both;
        animation-delay: 0.2s;
    }
    .sc-footer .mono {
        font-family: var(--font-mono);
        font-size: 0.65rem;
        opacity: 0.5;
    }

    /* Alerts */
    .stAlert { border-radius: 9px !important; font-size: 0.83rem !important; }

    /* Animations */
    @keyframes fadeDown {
        from { opacity:0; transform:translateY(-12px); }
        to   { opacity:1; transform:translateY(0); }
    }
    @keyframes fadeUp {
        from { opacity:0; transform:translateY(12px); }
        to   { opacity:1; transform:translateY(0); }
    }
    @keyframes tickerScroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Logo ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sc-logo-wrap">
        <div class="sc-logo-mark">📈</div>
        <span class="sc-wordmark">Stock<span>Cast</span></span>
        <span class="sc-tagline">Intelligent · Shariah-Screened · Predictive</span>
    </div>
    """, unsafe_allow_html=True)

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
                    _px  = float(_row.iloc[-1])
                    _prev= float(_row.iloc[-2])
                    _chg = (_px - _prev) / _prev * 100
                    _cls = "up" if _chg >= 0 else "down"
                    _arr = "▲" if _chg >= 0 else "▼"
                    _tape_items.append(
                        f'<span><span class="sym">{_s}</span> {_px:.2f} '
                        f'<span class="{_cls}">{_arr} {abs(_chg):.2f}%</span></span>'
                    )
            except Exception:
                pass
    except Exception:
        pass

    if not _tape_items:  # fallback — static labels, no prices
        _tape_items = [
            f'<span><span class="sym">{s}</span> <span class="up">Market data loading…</span></span>'
            for s in ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN", "NVDA", "META"]
        ]

    _tape_html = "".join(_tape_items * 2)  # duplicate for seamless CSS scroll loop
    st.markdown(
        f'<div class="sc-ticker-wrap"><div class="sc-ticker">{_tape_html}</div></div>',
        unsafe_allow_html=True
    )

    # ── Card open ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sc-card">', unsafe_allow_html=True)

    mode = st.session_state._auth_mode

    # ── Mode tab buttons (original 3-col pattern preserved) ───────────────────
    col_login, col_signup, col_reset = st.columns(3)
    with col_login:
        if st.button("Sign In", use_container_width=True,
                     type="primary" if mode == "login" else "secondary",
                     key="tab_login"):
            st.session_state._auth_mode = "login"; st.rerun()
    with col_signup:
        if st.button("Sign Up", use_container_width=True,
                     type="primary" if mode == "signup" else "secondary",
                     key="tab_signup"):
            st.session_state._auth_mode = "signup"; st.rerun()
    with col_reset:
        if st.button("Reset PW", use_container_width=True,
                     type="primary" if mode == "reset" else "secondary",
                     key="tab_reset"):
            st.session_state._auth_mode = "reset"; st.rerun()

    st.markdown("---")

    # ── LOGIN ──────────────────────────────────────────────────────────────────
    if mode == "login":
        st.subheader("Welcome back")
        email    = st.text_input("Email address", placeholder="you@example.com", key="ag_login_email")
        password = st.text_input("Password", type="password", placeholder="••••••••", key="ag_login_pw")

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

    # ── SIGN UP ────────────────────────────────────────────────────────────────
    elif mode == "signup":
        st.subheader("Create account")
        email     = st.text_input("Email address", placeholder="you@example.com", key="ag_signup_email")
        password  = st.text_input("Password (min 6 chars)", type="password",
                                  placeholder="••••••••", key="ag_signup_pw")
        password2 = st.text_input("Confirm password", type="password",
                                  placeholder="••••••••", key="ag_signup_pw2")

        st.markdown("""
        <div class="sc-shariah">
            ☽&nbsp; Your portfolio will be screened for Shariah compliance automatically.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀  Create Account", use_container_width=True, type="primary", key="signup_btn"):
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
        st.subheader("Reset password")
        st.caption("Enter your account email and we'll send a reset link.")
        email = st.text_input("Email address", placeholder="you@example.com", key="ag_reset_email")

        if st.button("📧  Send Reset Link", use_container_width=True, type="primary", key="reset_btn"):
            if not email:
                st.warning("Please enter your email address.")
            else:
                try:
                    supabase.auth.reset_password_email(email.strip())
                    st.success("Reset link sent — check your inbox.")
                except Exception as e:
                    st.error(f"Could not send reset email: {e}")

    # ── Card close ─────────────────────────────────────────────────────────────
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sc-footer">
        Secured with Supabase · Alpha Vantage &amp; Yahoo Finance<br>
        <span class="mono">StockCast · © 2025 All rights reserved</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Block app.py from rendering until authenticated ────────────────────────
    st.stop()

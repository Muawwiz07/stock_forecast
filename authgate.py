"""
authgate.py — Stockcast Auth Gate
Handles login, signup, and password reset UI using Supabase Auth.
Called from app.py as: from authgate import render_auth_gate; render_auth_gate(supabase)
"""

import streamlit as st


def render_auth_gate(supabase):
    """
    Renders the glassmorphic login / signup / reset UI and manages st.session_state.user.
    Calls st.stop() if the user is not yet authenticated.
    """

    # ── Initialise session state ───────────────────────────────────────────────
    if "user" not in st.session_state:
        st.session_state.user = None
    if "_auth_mode" not in st.session_state:
        st.session_state._auth_mode = "login"

    # ── Already authenticated ──────────────────────────────────────────────────
    if st.session_state.user is not None:
        return

    # ── Glassmorphic CSS ───────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

    /* ── Hide Streamlit chrome ── */
    [data-testid="stSidebar"]  { display: none !important; }
    [data-testid="stToolbar"]  { display: none !important; }
    footer                     { display: none !important; }
    #MainMenu                  { display: none !important; }

    /* ── Page background ── */
    .stApp {
        background:
            radial-gradient(ellipse 70% 55% at 50% 60%, rgba(0,180,90,0.13) 0%, transparent 70%),
            radial-gradient(ellipse 90% 70% at 50% 100%, rgba(0,100,50,0.18) 0%, transparent 65%),
            linear-gradient(180deg, #04100c 0%, #020b08 100%) !important;
        font-family: 'Rajdhani', sans-serif !important;
    }

    /* Grid lines overlay */
    .stApp::before {
        content: '';
        position: fixed; inset: 0;
        background-image:
            linear-gradient(rgba(0,255,136,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,255,136,0.025) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
    }

    /* Floating glow orb top-left */
    .stApp::after {
        content: '';
        position: fixed;
        width: 380px; height: 380px;
        background: radial-gradient(circle, rgba(0,255,120,0.08) 0%, transparent 70%);
        top: -100px; left: -100px;
        border-radius: 50%;
        filter: blur(40px);
        pointer-events: none;
        z-index: 0;
        animation: orbDrift 8s ease-in-out infinite alternate;
    }
    @keyframes orbDrift {
        from { transform: translate(0, 0); }
        to   { transform: translate(25px, 20px); }
    }

    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 2rem !important;
        position: relative;
        z-index: 1;
    }

    /* ── Auth card wrapper ── */
    .auth-card {
        max-width: 430px;
        margin: 0 auto;
        background: rgba(8, 28, 22, 0.75);
        border: 1px solid rgba(0,255,136,0.20);
        border-radius: 16px;
        padding: 2.4rem 2.2rem 2rem;
        backdrop-filter: blur(22px) saturate(1.4);
        -webkit-backdrop-filter: blur(22px) saturate(1.4);
        box-shadow:
            0 0 0 1px rgba(0,255,136,0.05),
            0 30px 90px rgba(0,0,0,0.75),
            inset 0 1px 0 rgba(0,255,136,0.14),
            inset 0 -1px 0 rgba(0,255,136,0.04);
        position: relative;
        overflow: hidden;
    }

    /* Glowing top edge */
    .auth-card::before {
        content: '';
        position: absolute;
        top: 0; left: 15%; right: 15%; height: 1px;
        background: linear-gradient(90deg, transparent, #00ff88, transparent);
    }

    /* Scan line sweep */
    .auth-card::after {
        content: '';
        position: absolute; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0,255,136,0.12), transparent);
        animation: scan 4s linear infinite;
        pointer-events: none;
    }
    @keyframes scan {
        0%   { top: 0%;   opacity: 1; }
        95%  { top: 100%; opacity: .4; }
        100% { top: 100%; opacity: 0; }
    }

    /* ── Logo ── */
    .auth-logo {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.45rem;
        color: #00ff88;
        letter-spacing: .08em;
        text-align: center;
        margin-bottom: .2rem;
        text-shadow: 0 0 22px rgba(0,255,136,0.50), 0 0 60px rgba(0,255,136,0.15);
    }
    .auth-logo .blue  { color: #4d9fff; }
    .auth-logo .arrow { color: rgba(0,255,136,0.35); }

    .auth-sub {
        font-family: 'Share Tech Mono', monospace;
        font-size: .55rem;
        color: #2a5040;
        letter-spacing: .25em;
        text-align: center;
        text-transform: uppercase;
        margin-bottom: 1.6rem;
    }

    /* ── Tab buttons ── */
    div[data-testid="column"] .stButton > button {
        font-family: 'Share Tech Mono', monospace !important;
        font-size: .60rem !important;
        letter-spacing: .14em !important;
        text-transform: uppercase !important;
        border-radius: 6px !important;
        padding: .42rem .3rem !important;
        transition: all .18s !important;
    }
    div[data-testid="column"] .stButton > button[kind="primary"] {
        background: #00ff88 !important;
        color: #020d08 !important;
        border: 1px solid #00ff88 !important;
        box-shadow: 0 0 18px rgba(0,255,136,0.45) !important;
    }
    div[data-testid="column"] .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: #2a5040 !important;
        border: 1px solid rgba(0,255,136,0.12) !important;
    }
    div[data-testid="column"] .stButton > button[kind="secondary"]:hover {
        color: #00ff88 !important;
        border-color: rgba(0,255,136,0.30) !important;
        background: rgba(0,255,136,0.05) !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(0,255,136,0.08) !important;
        margin: .7rem 0 1.1rem !important;
    }

    /* ── Input labels ── */
    .stTextInput > label {
        font-family: 'Share Tech Mono', monospace !important;
        font-size: .62rem !important;
        color: #2d6647 !important;
        letter-spacing: .20em !important;
        text-transform: uppercase !important;
        margin-bottom: .3rem !important;
    }

    /* ── Input fields ── */
    .stTextInput > div > div > input {
        font-family: 'Share Tech Mono', monospace !important;
        background: rgba(0,0,0,0.45) !important;
        border: 1px solid rgba(0,255,136,0.18) !important;
        border-radius: 8px !important;
        color: #c8ffe8 !important;
        font-size: .88rem !important;
        padding: .65rem 1rem !important;
        caret-color: #00ff88 !important;
        transition: border-color .2s, box-shadow .2s !important;
    }
    .stTextInput > div > div > input::placeholder { color: #1a3528 !important; }
    .stTextInput > div > div > input:focus {
        border-color: rgba(0,255,136,0.50) !important;
        box-shadow: 0 0 0 3px rgba(0,255,136,0.08), 0 0 18px rgba(0,255,136,0.10) !important;
        outline: none !important;
    }

    /* ── Primary CTA button ── */
    .stButton > button[kind="primary"] {
        font-family: 'Share Tech Mono', monospace !important;
        font-size: .78rem !important;
        letter-spacing: .18em !important;
        text-transform: uppercase !important;
        color: #020d08 !important;
        background: linear-gradient(90deg, #00d472, #00ff88, #00d472) !important;
        background-size: 200% 100% !important;
        border: none !important;
        border-radius: 8px !important;
        padding: .72rem 1rem !important;
        margin-top: .5rem !important;
        box-shadow: 0 4px 24px rgba(0,255,136,0.30) !important;
        transition: all .2s !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 32px rgba(0,255,136,0.50) !important;
        letter-spacing: .22em !important;
    }
    .stButton > button[kind="primary"]:active { transform: scale(.985) !important; }

    /* ── Secondary button (Request Alpha) ── */
    .stButton > button[kind="secondary"] {
        font-family: 'Share Tech Mono', monospace !important;
        font-size: .72rem !important;
        letter-spacing: .16em !important;
        text-transform: uppercase !important;
        color: #020d08 !important;
        background: linear-gradient(90deg, #00c46a, #00ff88, #00c46a) !important;
        border: none !important;
        border-radius: 7px !important;
        padding: .60rem 1rem !important;
        box-shadow: 0 3px 18px rgba(0,255,136,0.25) !important;
        transition: all .2s !important;
    }
    .stButton > button[kind="secondary"]:hover {
        box-shadow: 0 5px 26px rgba(0,255,136,0.42) !important;
    }

    /* ── Alerts ── */
    .stAlert {
        font-family: 'Share Tech Mono', monospace !important;
        font-size: .72rem !important;
        border-radius: 8px !important;
        margin-top: .8rem !important;
    }

    /* ── Offer banner ── */
    .offer-banner {
        background: rgba(0,0,0,0.40);
        border: 1px solid rgba(0,255,136,0.18);
        border-radius: 10px;
        padding: 1rem 1.2rem .6rem;
        text-align: center;
        margin-top: 1.2rem;
        position: relative; overflow: hidden;
    }
    .offer-banner::before {
        content: '';
        position: absolute; top:0; left:0; right:0; height:1px;
        background: linear-gradient(90deg, transparent, #00ff88, transparent);
    }
    .offer-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: .90rem;
        color: #00ff88;
        letter-spacing: .18em;
        text-transform: uppercase;
        text-shadow: 0 0 16px rgba(0,255,136,0.45);
        margin-bottom: .2rem;
    }
    .offer-sub {
        font-family: 'Rajdhani', sans-serif;
        font-size: .80rem;
        color: #2d5040;
        margin-bottom: .55rem;
    }

    /* ── Footer ── */
    .auth-footer {
        font-family: 'Share Tech Mono', monospace;
        font-size: .50rem;
        color: #1a2e24;
        text-align: center;
        letter-spacing: .18em;
        text-transform: uppercase;
        margin-top: 1.4rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Logo ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="auth-card">
        <div class="auth-logo">
            <span class="arrow">&gt; </span>Stock<span class="blue">cast</span>
        </div>
        <div class="auth-sub">AI-Powered Stock Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.session_state._auth_mode

    # ── Tabs ──────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⬡  Login", use_container_width=True,
                     type="primary" if mode == "login" else "secondary"):
            st.session_state._auth_mode = "login"; st.rerun()
    with c2:
        if st.button("⬡  Sign Up", use_container_width=True,
                     type="primary" if mode == "signup" else "secondary"):
            st.session_state._auth_mode = "signup"; st.rerun()
    with c3:
        if st.button("⬡  Reset PW", use_container_width=True,
                     type="primary" if mode == "reset" else "secondary"):
            st.session_state._auth_mode = "reset"; st.rerun()

    st.markdown("---")

    # ── LOGIN ─────────────────────────────────────────────────────────────────
    if mode == "login":
        email    = st.text_input("Identity Token (Email)", placeholder="name@firm.com",
                                 key="ag_login_email")
        password = st.text_input("Access Key", type="password",
                                 placeholder="••••••••", key="ag_login_pw")

        if st.button("▶  Authorize Access", use_container_width=True, type="primary"):
            if not email or not password:
                st.warning("Please enter your email and access key.")
            else:
                try:
                    res = supabase.auth.sign_in_with_password(
                        {"email": email.strip(), "password": password}
                    )
                    if res.user:
                        st.session_state.user = res.user
                        st.success("✓  Access Granted")
                        st.rerun()
                    else:
                        st.error("Login failed — check your credentials.")
                except Exception as e:
                    err = str(e)
                    if "Invalid login credentials" in err or "invalid_credentials" in err:
                        st.error("❌ Invalid identity token or access key.")
                    elif "Email not confirmed" in err:
                        st.warning("📧 Please confirm your email before logging in.")
                    else:
                        st.error(f"Login error: {err}")

        # Offer banner
        st.markdown("""
        <div class="offer-banner">
            <div class="offer-title">Limited Offer</div>
            <div class="offer-sub">Glassmorphic alpha — exclusive early access</div>
        </div>
        """, unsafe_allow_html=True)
        st.button("🚀  Request Alpha Access", use_container_width=True)

    # ── SIGN UP ───────────────────────────────────────────────────────────────
    elif mode == "signup":
        email     = st.text_input("Identity Token (Email)", placeholder="name@firm.com",
                                  key="ag_signup_email")
        password  = st.text_input("Access Key (min 6 chars)", type="password",
                                  placeholder="••••••••", key="ag_signup_pw")
        password2 = st.text_input("Confirm Access Key", type="password",
                                  placeholder="••••••••", key="ag_signup_pw2")

        if st.button("🚀  Create Account", use_container_width=True, type="primary"):
            if not email or not password:
                st.warning("Please fill in all fields.")
            elif len(password) < 6:
                st.warning("Access key must be at least 6 characters.")
            elif password != password2:
                st.error("Access keys do not match.")
            else:
                try:
                    res = supabase.auth.sign_up(
                        {"email": email.strip(), "password": password}
                    )
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
        email = st.text_input("Identity Token (Email)", placeholder="name@firm.com",
                               key="ag_reset_email")

        if st.button("📧  Send Reset Link", use_container_width=True, type="primary"):
            if not email:
                st.warning("Please enter your email address.")
            else:
                try:
                    supabase.auth.reset_password_email(email.strip())
                    st.success("Reset link sent — check your inbox.")
                except Exception as e:
                    st.error(f"Could not send reset email: {e}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="auth-footer">Stockcast © 2025 &nbsp;·&nbsp; Secured via Supabase Auth</div>',
        unsafe_allow_html=True,
    )

    st.stop()

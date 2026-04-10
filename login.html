"""
authgate.py — Stockcast Auth Gate
Handles login, signup, and password reset UI using Supabase Auth.
Called from app.py as: from authgate import render_auth_gate; render_auth_gate(supabase)
"""

import streamlit as st


def render_auth_gate(supabase):
    """
    Renders the login / signup / reset UI and manages st.session_state.user.
    Calls st.stop() if the user is not yet authenticated.
    """

    # ── Initialise session state ───────────────────────────────────────────────
    if "user" not in st.session_state:
        st.session_state.user = None
    if "_auth_mode" not in st.session_state:
        st.session_state._auth_mode = "login"  # "login" | "signup" | "reset"

    # ── Already authenticated ──────────────────────────────────────────────────
    if st.session_state.user is not None:
        return  # nothing to do — let app.py continue

    # ── CSS ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

    /* ── Page & sidebar ── */
    [data-testid="stSidebar"]          { display: none !important; }
    [data-testid="stToolbar"]          { display: none !important; }
    .stApp                             { background: #04080f; }
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 2rem !important;
    }

    /* ── Animated grid background ── */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(0,229,176,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,229,176,0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
    }

    /* ── Card ── */
    .auth-card {
        max-width: 440px;
        margin: 0 auto;
        background: rgba(6, 14, 30, 0.90);
        border: 1px solid rgba(0, 229, 176, 0.20);
        border-radius: 10px;
        padding: 2.6rem 2.6rem 2.2rem;
        box-shadow:
            0 0 0 1px rgba(0,229,176,0.05),
            0 24px 80px rgba(0,0,0,0.7),
            inset 0 1px 0 rgba(0,229,176,0.12);
        position: relative;
        z-index: 1;
    }

    /* Glowing top bar */
    .auth-card::before {
        content: '';
        position: absolute;
        top: -1px; left: 20%; right: 20%;
        height: 1px;
        background: linear-gradient(90deg, transparent, #00e5b0, transparent);
        border-radius: 50%;
    }

    /* ── Logo ── */
    .auth-logo {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #00e5b0;
        letter-spacing: .06em;
        text-align: center;
        margin-bottom: .25rem;
        text-shadow: 0 0 24px rgba(0,229,176,0.40);
    }
    .auth-logo .blue  { color: #4d8eff; }
    .auth-logo .arrow { color: rgba(0,229,176,0.45); font-weight: 400; }

    .auth-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: .62rem;
        color: #3a4255;
        letter-spacing: .22em;
        text-align: center;
        margin-bottom: 2rem;
        text-transform: uppercase;
    }

    /* ── Mode indicator dots ── */
    .mode-dots {
        display: flex;
        justify-content: center;
        gap: .55rem;
        margin-bottom: 1.5rem;
    }
    .mode-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: rgba(0,229,176,0.15);
        border: 1px solid rgba(0,229,176,0.20);
    }
    .mode-dot.active {
        background: #00e5b0;
        box-shadow: 0 0 8px rgba(0,229,176,0.60);
    }

    /* ── Section header ── */
    .auth-section-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: .72rem;
        font-weight: 600;
        color: #00e5b0;
        letter-spacing: .18em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
        opacity: .85;
    }

    /* ── Inputs ── */
    .stTextInput > label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: .70rem !important;
        color: #4a5568 !important;
        letter-spacing: .12em !important;
        text-transform: uppercase !important;
        margin-bottom: .2rem !important;
    }
    .stTextInput > div > div > input {
        font-family: 'IBM Plex Mono', monospace !important;
        background: rgba(0,0,0,0.50) !important;
        border: 1px solid rgba(0,229,176,0.15) !important;
        border-radius: 5px !important;
        color: #c8d6ef !important;
        font-size: .88rem !important;
        padding: .55rem .85rem !important;
        transition: border-color .2s, box-shadow .2s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: rgba(0,229,176,0.50) !important;
        box-shadow: 0 0 0 2px rgba(0,229,176,0.10) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder { color: #2a3245 !important; }

    /* ── Primary button ── */
    .stButton > button[kind="primary"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: .78rem !important;
        font-weight: 600 !important;
        letter-spacing: .12em !important;
        text-transform: uppercase !important;
        background: linear-gradient(135deg, #00c49a 0%, #00e5b0 100%) !important;
        color: #030a14 !important;
        border: none !important;
        border-radius: 5px !important;
        padding: .65rem 1rem !important;
        margin-top: .6rem !important;
        box-shadow: 0 4px 20px rgba(0,229,176,0.25) !important;
        transition: all .2s !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #00dba9 0%, #33ecc0 100%) !important;
        box-shadow: 0 6px 28px rgba(0,229,176,0.40) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="primary"]:active { transform: translateY(0) !important; }

    /* ── Secondary / tab buttons ── */
    .stButton > button[kind="secondary"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: .68rem !important;
        letter-spacing: .10em !important;
        text-transform: uppercase !important;
        background: transparent !important;
        color: #3a4a60 !important;
        border: 1px solid rgba(0,229,176,0.10) !important;
        border-radius: 5px !important;
        transition: all .18s !important;
    }
    .stButton > button[kind="secondary"]:hover {
        color: #00e5b0 !important;
        border-color: rgba(0,229,176,0.30) !important;
        background: rgba(0,229,176,0.05) !important;
    }

    /* ── Divider ── */
    hr { border-color: rgba(0,229,176,0.08) !important; margin: 1rem 0 1.4rem !important; }

    /* ── Alerts ── */
    .stAlert {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: .75rem !important;
        border-radius: 5px !important;
        margin-top: .8rem !important;
    }

    /* ── Footer watermark ── */
    .auth-footer {
        font-family: 'IBM Plex Mono', monospace;
        font-size: .52rem;
        color: #1e2535;
        text-align: center;
        letter-spacing: .20em;
        margin-top: 1.6rem;
        text-transform: uppercase;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Determine active mode ──────────────────────────────────────────────────
    mode = st.session_state._auth_mode

    # ── Mode dot indicator ─────────────────────────────────────────────────────
    dots = {
        "login":  [True,  False, False],
        "signup": [False, True,  False],
        "reset":  [False, False, True],
    }
    d = dots.get(mode, [True, False, False])
    dot_html = "".join(
        f'<div class="mode-dot {"active" if active else ""}"></div>'
        for active in d
    )

    # ── Card open + logo ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="auth-card">
        <div class="auth-logo">
            <span class="arrow">&gt; </span>Stock<span class="blue">cast</span>
        </div>
        <div class="auth-sub">AI-Powered Stock Intelligence</div>
        <div class="mode-dots">{dot_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Spacer so Streamlit widgets render inside the visual card area ─────────
    # (Streamlit can't truly nest widgets inside custom HTML, so we replicate the
    #  card look as a background layer and let widgets flow naturally beneath it.)

    # ── Tab row ───────────────────────────────────────────────────────────────
    col_login, col_signup, col_reset = st.columns(3)
    with col_login:
        if st.button("⬡  Login", use_container_width=True,
                     type="primary" if mode == "login" else "secondary"):
            st.session_state._auth_mode = "login"
            st.rerun()
    with col_signup:
        if st.button("⬡  Sign Up", use_container_width=True,
                     type="primary" if mode == "signup" else "secondary"):
            st.session_state._auth_mode = "signup"
            st.rerun()
    with col_reset:
        if st.button("⬡  Reset PW", use_container_width=True,
                     type="primary" if mode == "reset" else "secondary"):
            st.session_state._auth_mode = "reset"
            st.rerun()

    st.markdown("---")

    # ── Login ──────────────────────────────────────────────────────────────────
    if mode == "login":
        st.markdown('<div class="auth-section-title">// Authorize Access</div>',
                    unsafe_allow_html=True)
        email    = st.text_input("Email", placeholder="analyst@firm.com",
                                 key="ag_login_email")
        password = st.text_input("Password", type="password",
                                 placeholder="••••••••", key="ag_login_pw")

        if st.button("▶  Authorize Access", use_container_width=True, type="primary"):
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

    # ── Sign Up ────────────────────────────────────────────────────────────────
    elif mode == "signup":
        st.markdown('<div class="auth-section-title">// Create Account</div>',
                    unsafe_allow_html=True)
        email     = st.text_input("Email", placeholder="analyst@firm.com",
                                  key="ag_signup_email")
        password  = st.text_input("Password (min 6 chars)", type="password",
                                  placeholder="••••••••", key="ag_signup_pw")
        password2 = st.text_input("Confirm password", type="password",
                                  placeholder="••••••••", key="ag_signup_pw2")

        if st.button("🚀  Create Account", use_container_width=True, type="primary"):
            if not email or not password:
                st.warning("Please fill in all fields.")
            elif len(password) < 6:
                st.warning("Password must be at least 6 characters.")
            elif password != password2:
                st.error("Passwords do not match.")
            else:
                try:
                    res = supabase.auth.sign_up(
                        {"email": email.strip(), "password": password}
                    )
                    if res.user:
                        st.success(
                            "✅ Account created! Check your email to confirm, then log in."
                        )
                        st.session_state._auth_mode = "login"
                        st.rerun()
                    else:
                        st.error("Sign-up failed — please try again.")
                except Exception as e:
                    err = str(e)
                    if "already registered" in err or "already been registered" in err:
                        st.error(
                            "❌ This email is already registered. Try logging in instead."
                        )
                    else:
                        st.error(f"Sign-up error: {err}")

    # ── Password Reset ─────────────────────────────────────────────────────────
    elif mode == "reset":
        st.markdown('<div class="auth-section-title">// Reset Password</div>',
                    unsafe_allow_html=True)
        email = st.text_input("Email", placeholder="analyst@firm.com",
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

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="auth-footer">Stockcast © 2025 &nbsp;·&nbsp; Secured via Supabase Auth</div>',
        unsafe_allow_html=True,
    )

    # Block the rest of app.py from running
    st.stop()

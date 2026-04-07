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
        st.session_state._auth_mode = "login"   # "login" | "signup" | "reset"

    # ── Already authenticated ──────────────────────────────────────────────────
    if st.session_state.user is not None:
        return  # nothing to do — let app.py continue

    # ── CSS (terminal / dark theme matching app.py) ────────────────────────────
    st.markdown("""
    <style>
    /* Hide default Streamlit chrome on the auth page */
    [data-testid="stSidebar"] { display: none !important; }
    .auth-card {
        max-width: 420px;
        margin: 4rem auto 0;
        background: rgba(6,14,30,0.82);
        border: 1px solid rgba(0,229,176,0.18);
        border-radius: 8px;
        padding: 2.2rem 2.4rem 2rem;
        box-shadow: 0 8px 60px rgba(0,0,0,0.6), inset 0 1px 0 rgba(0,229,176,0.1);
    }
    .auth-logo {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.35rem;
        font-weight: 700;
        color: #00e5b0;
        letter-spacing: .04em;
        text-align: center;
        margin-bottom: .3rem;
    }
    .auth-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: .58rem;
        color: #424754;
        letter-spacing: .15em;
        text-align: center;
        margin-bottom: 1.8rem;
        text-transform: uppercase;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Auth card shell ────────────────────────────────────────────────────────
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="auth-logo">📈 Stockcast</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-sub">AI-Powered Stock Intelligence</div>', unsafe_allow_html=True)

    mode = st.session_state._auth_mode

    # ── Mode tabs ──────────────────────────────────────────────────────────────
    col_login, col_signup, col_reset = st.columns(3)
    with col_login:
        if st.button("Login", use_container_width=True,
                     type="primary" if mode == "login" else "secondary"):
            st.session_state._auth_mode = "login"
            st.rerun()
    with col_signup:
        if st.button("Sign Up", use_container_width=True,
                     type="primary" if mode == "signup" else "secondary"):
            st.session_state._auth_mode = "signup"
            st.rerun()
    with col_reset:
        if st.button("Reset PW", use_container_width=True,
                     type="primary" if mode == "reset" else "secondary"):
            st.session_state._auth_mode = "reset"
            st.rerun()

    st.markdown("---")

    # ── Login ──────────────────────────────────────────────────────────────────
    if mode == "login":
        st.subheader("Welcome back")
        email    = st.text_input("Email", placeholder="name@firm.com", key="ag_login_email")
        password = st.text_input("Password", type="password", placeholder="••••••••", key="ag_login_pw")

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
        st.subheader("Create account")
        email    = st.text_input("Email", placeholder="name@firm.com", key="ag_signup_email")
        password = st.text_input("Password (min 6 chars)", type="password",
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

    # ── Password Reset ─────────────────────────────────────────────────────────
    elif mode == "reset":
        st.subheader("Reset password")
        email = st.text_input("Email", placeholder="name@firm.com", key="ag_reset_email")

        if st.button("📧  Send Reset Link", use_container_width=True, type="primary"):
            if not email:
                st.warning("Please enter your email address.")
            else:
                try:
                    supabase.auth.reset_password_email(email.strip())
                    st.success("Reset link sent — check your inbox.")
                except Exception as e:
                    st.error(f"Could not send reset email: {e}")

    st.markdown('</div>', unsafe_allow_html=True)  # close .auth-card

    # Block the rest of app.py from running
    st.stop()

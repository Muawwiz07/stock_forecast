"""
authgate.py — StockCast Authentication Gate
Portrait-oriented login & signup page with Supabase auth integration.
"""

import streamlit as st
from supabase import create_client, Client
import re
import time

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="StockCast · Sign In",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Supabase client (reads from st.secrets) ───────────────────────────────────
@st.cache_resource
def get_supabase() -> Client:
    url  = st.secrets["supabase"]["url"]
    key  = st.secrets["supabase"]["anon_key"]
    return create_client(url, key)

supabase = get_supabase()

# ── CSS — portrait card, dark fintech theme ───────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ──────────────────────────────── */
:root {
    --bg:          #0b0f1a;
    --surface:     #111827;
    --border:      #1e2d45;
    --border-glow: #1a4a7a;
    --accent:      #3b82f6;
    --accent-light:#60a5fa;
    --accent-dim:  rgba(59,130,246,0.12);
    --gold:        #d4a853;
    --gold-dim:    rgba(212,168,83,0.12);
    --text:        #e8edf5;
    --muted:       #64748b;
    --danger:      #ef4444;
    --success:     #22c55e;
    --font-serif:  'DM Serif Display', Georgia, serif;
    --font-sans:   'DM Sans', sans-serif;
    --font-mono:   'JetBrains Mono', monospace;
}

/* ── Global reset ────────────────────────────────── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-sans) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stSidebar"] { display: none; }

/* ── Constrain to portrait card ─────────────────── */
.main .block-container {
    max-width: 420px !important;
    padding: 2.5rem 1.5rem 3rem !important;
    margin: 0 auto !important;
}

/* ── Logo / Header ───────────────────────────────── */
.sc-logo-wrap {
    text-align: center;
    margin-bottom: 2.4rem;
    animation: fadeDown 0.6s ease both;
}
.sc-logo-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 56px; height: 56px;
    background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%);
    border-radius: 14px;
    margin-bottom: 1rem;
    box-shadow: 0 0 32px rgba(59,130,246,0.35);
    font-size: 1.6rem;
}
.sc-wordmark {
    font-family: var(--font-serif);
    font-size: 2rem;
    letter-spacing: -0.02em;
    color: var(--text);
    display: block;
    line-height: 1;
}
.sc-wordmark span {
    color: var(--accent-light);
}
.sc-tagline {
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.35rem;
    font-weight: 500;
    display: block;
}

/* ── Card ────────────────────────────────────────── */
.sc-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2rem 1.75rem;
    box-shadow: 0 24px 64px rgba(0,0,0,0.45),
                0 0 0 1px rgba(255,255,255,0.03) inset;
    animation: fadeUp 0.55s ease both;
    animation-delay: 0.1s;
}

/* ── Tab switcher ────────────────────────────────── */
.sc-tabs {
    display: flex;
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 4px;
    margin-bottom: 1.6rem;
    gap: 4px;
}
.sc-tab {
    flex: 1;
    text-align: center;
    padding: 0.5rem;
    border-radius: 7px;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    color: var(--muted);
    user-select: none;
}
.sc-tab.active {
    background: var(--accent);
    color: #fff;
    box-shadow: 0 2px 8px rgba(59,130,246,0.4);
}

/* ── Form labels ─────────────────────────────────── */
.stTextInput label, .stSelectbox label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 4px !important;
}

/* ── Inputs ──────────────────────────────────────── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 0.85rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--muted) !important;
    opacity: 0.6 !important;
}

/* ── Primary button ──────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: var(--font-sans) !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    padding: 0.7rem !important;
    letter-spacing: 0.03em !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.35) !important;
    margin-top: 0.4rem !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(59,130,246,0.45) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Divider ─────────────────────────────────────── */
.sc-divider {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.2rem 0;
    color: var(--muted);
    font-size: 0.75rem;
}
.sc-divider::before, .sc-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Shariah badge ───────────────────────────────── */
.sc-shariah-badge {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: var(--gold-dim);
    border: 1px solid rgba(212,168,83,0.25);
    border-radius: 10px;
    padding: 0.65rem 0.9rem;
    margin: 1.4rem 0 0.2rem;
    font-size: 0.78rem;
    color: var(--gold);
    font-weight: 500;
}
.sc-shariah-badge .icon { font-size: 1rem; }

/* ── Forgot password link ────────────────────────── */
.sc-forgot {
    text-align: right;
    margin-top: -0.3rem;
    margin-bottom: 0.8rem;
}
.sc-forgot a {
    font-size: 0.76rem;
    color: var(--accent-light);
    text-decoration: none;
    opacity: 0.8;
}
.sc-forgot a:hover { opacity: 1; text-decoration: underline; }

/* ── Footer note ─────────────────────────────────── */
.sc-footer-note {
    text-align: center;
    font-size: 0.73rem;
    color: var(--muted);
    margin-top: 1.8rem;
    line-height: 1.6;
    animation: fadeUp 0.7s ease both;
    animation-delay: 0.25s;
}
.sc-footer-note .mono {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    opacity: 0.6;
}

/* ── Ticker tape ─────────────────────────────────── */
.sc-ticker-wrap {
    overflow: hidden;
    margin-bottom: 1.6rem;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    padding: 0.45rem 0;
    animation: fadeUp 0.6s ease both;
    animation-delay: 0.05s;
}
.sc-ticker {
    display: flex;
    gap: 2rem;
    animation: tickerScroll 18s linear infinite;
    white-space: nowrap;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted);
}
.sc-ticker .up   { color: var(--success); }
.sc-ticker .down { color: var(--danger); }
.sc-ticker .sym  { color: var(--text); font-weight: 500; }

/* ── Alerts ──────────────────────────────────────── */
.stAlert { border-radius: 10px !important; font-size: 0.85rem !important; }

/* ── Animations ──────────────────────────────────── */
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes tickerScroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
</style>
""", unsafe_allow_html=True)


# ── Helper: validate email ─────────────────────────────────────────────────────
def _valid_email(email: str) -> bool:
    return bool(re.match(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$", email))


# ── Helper: validate password strength ────────────────────────────────────────
def _strong_password(pw: str) -> tuple[bool, str]:
    if len(pw) < 8:
        return False, "At least 8 characters required."
    if not re.search(r"[A-Z]", pw):
        return False, "Include at least one uppercase letter."
    if not re.search(r"\d", pw):
        return False, "Include at least one number."
    return True, ""


# ── Session state defaults ────────────────────────────────────────────────────
if "auth_tab" not in st.session_state:
    st.session_state.auth_tab = "login"   # "login" | "signup" | "reset"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None


# ── If already authenticated, skip gate ──────────────────────────────────────
def is_authenticated() -> bool:
    """Call this from app.py to guard pages."""
    return st.session_state.get("authenticated", False)


def logout():
    """Call from app.py sidebar."""
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()


# ── Auth actions ──────────────────────────────────────────────────────────────
def _do_login(email: str, password: str):
    try:
        resp = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if resp.user:
            st.session_state.authenticated = True
            st.session_state.user = resp.user
            st.rerun()
        else:
            st.error("Login failed. Please check your credentials.")
    except Exception as e:
        msg = str(e)
        if "Invalid login" in msg or "invalid" in msg.lower():
            st.error("Invalid email or password.")
        elif "Email not confirmed" in msg:
            st.warning("Please confirm your email before signing in.")
        else:
            st.error(f"Login error: {msg}")


def _do_signup(email: str, password: str, full_name: str):
    try:
        resp = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"full_name": full_name}},
        })
        if resp.user:
            st.success("Account created! Check your email to confirm, then sign in.")
            time.sleep(1.5)
            st.session_state.auth_tab = "login"
            st.rerun()
        else:
            st.error("Signup failed. Please try again.")
    except Exception as e:
        msg = str(e)
        if "already registered" in msg.lower() or "already exists" in msg.lower():
            st.error("This email is already registered. Please sign in.")
        else:
            st.error(f"Signup error: {msg}")


def _do_reset(email: str):
    try:
        supabase.auth.reset_password_email(email)
        st.success("Password reset link sent! Check your inbox.")
        time.sleep(1.5)
        st.session_state.auth_tab = "login"
        st.rerun()
    except Exception as e:
        st.error(f"Reset error: {e}")


# ── Ticker tape (static mock — replace with live API if desired) ──────────────
TICKER_HTML = """
<div class="sc-ticker-wrap">
  <div class="sc-ticker">
    <span><span class="sym">AAPL</span> 189.42 <span class="up">▲ 0.8%</span></span>
    <span><span class="sym">MSFT</span> 415.60 <span class="up">▲ 1.2%</span></span>
    <span><span class="sym">TSLA</span> 172.11 <span class="down">▼ 0.5%</span></span>
    <span><span class="sym">GOOG</span> 174.95 <span class="up">▲ 0.3%</span></span>
    <span><span class="sym">AMZN</span> 185.22 <span class="up">▲ 0.9%</span></span>
    <span><span class="sym">NVDA</span> 873.50 <span class="down">▼ 1.1%</span></span>
    <span><span class="sym">META</span> 512.33 <span class="up">▲ 0.6%</span></span>
    <!-- duplicate for seamless loop -->
    <span><span class="sym">AAPL</span> 189.42 <span class="up">▲ 0.8%</span></span>
    <span><span class="sym">MSFT</span> 415.60 <span class="up">▲ 1.2%</span></span>
    <span><span class="sym">TSLA</span> 172.11 <span class="down">▼ 0.5%</span></span>
    <span><span class="sym">GOOG</span> 174.95 <span class="up">▲ 0.3%</span></span>
    <span><span class="sym">AMZN</span> 185.22 <span class="up">▲ 0.9%</span></span>
    <span><span class="sym">NVDA</span> 873.50 <span class="down">▼ 1.1%</span></span>
    <span><span class="sym">META</span> 512.33 <span class="up">▲ 0.6%</span></span>
  </div>
</div>
"""


# ── Main render ───────────────────────────────────────────────────────────────
def render_auth_gate():
    """
    Renders the full auth gate UI.
    Returns True if authenticated (caller can then render the main app).
    """
    if st.session_state.authenticated:
        return True

    # ── Logo
    st.markdown("""
    <div class="sc-logo-wrap">
        <div class="sc-logo-mark">📈</div>
        <span class="sc-wordmark">Stock<span>Cast</span></span>
        <span class="sc-tagline">Intelligent · Shariah-Screened · Predictive</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Ticker tape
    st.markdown(TICKER_HTML, unsafe_allow_html=True)

    # ── Tab switcher (HTML visual only; logic via buttons below)
    tab = st.session_state.auth_tab

    if tab != "reset":
        login_active  = "active" if tab == "login"  else ""
        signup_active = "active" if tab == "signup" else ""

        st.markdown(f"""
        <div class="sc-tabs">
            <div class="sc-tab {login_active}"  id="_tab_login">Sign In</div>
            <div class="sc-tab {signup_active}" id="_tab_signup">Create Account</div>
        </div>
        """, unsafe_allow_html=True)

        # invisible real buttons to switch tabs
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sign In",        key="switch_login",  use_container_width=True):
                st.session_state.auth_tab = "login";  st.rerun()
        with col2:
            if st.button("Create Account", key="switch_signup", use_container_width=True):
                st.session_state.auth_tab = "signup"; st.rerun()

    # ── Card open
    st.markdown('<div class="sc-card">', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # LOGIN TAB
    # ──────────────────────────────────────────────────────────────────────────
    if tab == "login":
        email    = st.text_input("Email address", placeholder="you@example.com", key="li_email")
        password = st.text_input("Password",      placeholder="••••••••",         type="password", key="li_pw")

        st.markdown('<div class="sc-forgot"><a href="#">Forgot password?</a></div>', unsafe_allow_html=True)

        # Override forgot-password via a button
        if st.button("Forgot password?", key="goto_reset"):
            st.session_state.auth_tab = "reset"; st.rerun()

        if st.button("Sign In →", key="login_btn", type="primary", use_container_width=True):
            if not email or not password:
                st.warning("Please enter both email and password.")
            elif not _valid_email(email):
                st.error("Please enter a valid email address.")
            else:
                with st.spinner("Signing in…"):
                    _do_login(email.strip().lower(), password)

    # ──────────────────────────────────────────────────────────────────────────
    # SIGNUP TAB
    # ──────────────────────────────────────────────────────────────────────────
    elif tab == "signup":
        full_name = st.text_input("Full name",     placeholder="Ahmad bin Yusuf",  key="su_name")
        email     = st.text_input("Email address", placeholder="you@example.com",  key="su_email")
        password  = st.text_input("Password",      placeholder="Min. 8 chars, 1 upper, 1 number",
                                  type="password", key="su_pw")
        confirm   = st.text_input("Confirm password", placeholder="••••••••",
                                  type="password", key="su_confirm")

        st.markdown("""
        <div class="sc-shariah-badge">
            <span class="icon">☽</span>
            Your portfolio will be screened for Shariah compliance automatically.
        </div>
        """, unsafe_allow_html=True)

        if st.button("Create Account →", key="signup_btn", type="primary", use_container_width=True):
            if not full_name or not email or not password or not confirm:
                st.warning("All fields are required.")
            elif not _valid_email(email):
                st.error("Please enter a valid email address.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, msg = _strong_password(password)
                if not ok:
                    st.error(msg)
                else:
                    with st.spinner("Creating your account…"):
                        _do_signup(email.strip().lower(), password, full_name.strip())

    # ──────────────────────────────────────────────────────────────────────────
    # RESET TAB
    # ──────────────────────────────────────────────────────────────────────────
    elif tab == "reset":
        st.markdown("#### Reset Password")
        st.caption("Enter your account email and we'll send a reset link.")
        email = st.text_input("Email address", placeholder="you@example.com", key="rs_email")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("← Back", key="reset_back"):
                st.session_state.auth_tab = "login"; st.rerun()
        with col_b:
            if st.button("Send Link →", key="reset_btn", type="primary", use_container_width=True):
                if not email or not _valid_email(email):
                    st.error("Enter a valid email address.")
                else:
                    with st.spinner("Sending reset link…"):
                        _do_reset(email.strip().lower())

    # ── Card close
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Footer
    st.markdown("""
    <div class="sc-footer-note">
        Secured with Supabase · Data from Alpha Vantage &amp; Yahoo Finance<br>
        <span class="mono">StockCast v1.0 · © 2024 All rights reserved</span>
    </div>
    """, unsafe_allow_html=True)

    return False


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    authenticated = render_auth_gate()
    if authenticated:
        st.success(f"Welcome back, {st.session_state.user.email}!")
        st.info("(Replace this block with your main app content or call your app router.)")
        if st.button("Sign Out"):
            logout()

"""
authgate.py — STOCKCAST Command Center Login / Signup
──────────────────────────────────────────────────────
Self-contained auth module. Import and call render_auth_gate(supabase)
from app.py. Handles login, signup, session state, and st.stop().
"""

import streamlit as st


def render_auth_gate(supabase):
    """
    Render the Command Center auth screen.

    Parameters
    ----------
    supabase : supabase.Client
        Initialised Supabase client from app.py.

    Behaviour
    ---------
    - If the user is already logged in (st.session_state.user is not None),
      returns immediately so app.py continues normally.
    - Otherwise renders the full login/signup UI and calls st.stop().
    """

    if st.session_state.get("user") is not None:
        return  # already authenticated — let app.py continue

    _auth_error   = ""
    _auth_success = ""
    if "auth_view" not in st.session_state:
        st.session_state.auth_view = "login"
    _is_login = (st.session_state.auth_view == "login")

    # ── Command Center CSS ────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;700;800&family=Space+Grotesk:wght@300;500;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

    html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"],[data-testid="stMain"],.main,.stMainBlockContainer {
        background: #111318 !important; padding: 0 !important; margin: 0 !important;
        overflow-x: hidden !important; font-family: 'Manrope', sans-serif !important;
    }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stMain"] > div:first-child { padding-top: 0 !important; }

    /* ── Background layers ── */
    .auth-bg {
        position: fixed; inset: 0; z-index: 0; background: #111318; overflow: hidden;
    }
    .auth-bg-grid {
        position: absolute; inset: 0; opacity: 0.2;
        background-image: linear-gradient(to right, rgba(0,229,255,0.05) 1px, transparent 1px),
                          linear-gradient(to bottom, rgba(0,229,255,0.05) 1px, transparent 1px);
        background-size: 40px 40px;
    }
    .auth-bg-glow-tr {
        position: absolute; top: -10%; right: -10%; width: 50%; height: 50%;
        background: rgba(195,245,255,0.05); border-radius: 50%; filter: blur(120px);
    }
    .auth-bg-glow-bl {
        position: absolute; bottom: -5%; left: -5%; width: 40%; height: 40%;
        background: rgba(255,187,243,0.05); border-radius: 50%; filter: blur(100px);
    }

    /* ── Central card / form container ── */
    .auth-canvas {
        position: fixed; inset: 0; z-index: 10;
        display: flex; align-items: center; justify-content: center; padding: 1.5rem;
    }
    .auth-terminal {
        position: relative; width: 100%; max-width: 480px;
    }
    .auth-bracket-tl {
        position: absolute; top: -1rem; left: -1rem; width: 3rem; height: 3rem;
        border-top: 2px solid rgba(195,245,255,0.4); border-left: 2px solid rgba(195,245,255,0.4);
    }
    .auth-bracket-br {
        position: absolute; bottom: -1rem; right: -1rem; width: 3rem; height: 3rem;
        border-bottom: 2px solid rgba(195,245,255,0.4); border-right: 2px solid rgba(195,245,255,0.4);
    }
    .glass-panel {
        background: rgba(17,19,24,0.75);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(195,245,255,0.1);
        border-radius: 0.5rem;
        padding: 2.5rem 2.8rem;
        box-shadow: 0 0 50px rgba(0,229,255,0.05);
        position: relative; overflow: hidden;
    }
    /* scanning line */
    .scanning-line {
        position: absolute; left: 0; right: 0; top: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00e5ff, transparent);
        box-shadow: 0 0 15px #00e5ff;
        animation: scanner 4s linear infinite;
        pointer-events: none; z-index: 20;
    }
    @keyframes scanner {
        0%   { transform: translateY(-100%); opacity: 0; }
        50%  { opacity: 0.5; }
        100% { transform: translateY(22rem); opacity: 0; }
    }

    /* ── Header ── */
    .auth-header { text-align: center; margin-bottom: 2.2rem; }
    .auth-icon {
        font-family: 'Material Symbols Outlined'; font-size: 3.2rem; font-weight: 200;
        color: #c3f5ff; margin-bottom: 0.8rem; display: block;
        font-variation-settings: 'wght' 200;
    }
    .auth-title {
        font-family: 'Manrope', sans-serif; font-size: 2.4rem; font-weight: 900;
        color: #c3f5ff; letter-spacing: -0.03em; text-transform: uppercase; margin-bottom: 0.3rem;
    }
    .auth-subtitle {
        font-family: 'Space Grotesk', sans-serif; font-size: 0.62rem;
        letter-spacing: 0.3em; color: #bac9cc; text-transform: uppercase;
    }

    /* ── Tab buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, rgba(195,245,255,0.8), rgba(0,229,255,1)) !important;
        color: #00363d !important;
        font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important;
        font-size: 0.68rem !important; letter-spacing: 0.2em !important;
        text-transform: uppercase !important; border: none !important;
        border-radius: 0.25rem !important; padding: 0.9rem 1rem !important;
        width: 100% !important;
        box-shadow: 0 0 20px rgba(0,229,255,0.3) !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 35px rgba(0,229,255,0.5) !important;
        transform: scale(0.99) !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background: rgba(195,245,255,0.04) !important;
        color: #849396 !important;
        border: 1px solid rgba(59,73,76,0.6) !important;
        box-shadow: none !important;
        font-size: 0.6rem !important; padding: 0.6rem !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        color: #c3f5ff !important;
        border-color: rgba(195,245,255,0.35) !important;
        background: rgba(195,245,255,0.08) !important;
    }

    /* ── Inputs ── */
    [data-testid="stTextInput"] label { display: none !important; }
    [data-testid="stTextInput"] > div > div > input {
        background: transparent !important;
        border: none !important; border-bottom: 1px solid #3b494c !important;
        border-radius: 0 !important;
        color: #e2e2e8 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.82rem !important; letter-spacing: 0.08em !important;
        padding: 0.75rem 0.5rem 0.75rem 1.8rem !important;
        transition: border-color 0.3s !important;
    }
    [data-testid="stTextInput"] > div > div > input:focus {
        border-bottom-color: #c3f5ff !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-testid="stTextInput"] > div > div > input::placeholder {
        color: rgba(59,73,76,0.5) !important; text-transform: uppercase; letter-spacing: 0.1em;
    }

    /* ── Field labels ── */
    .auth-field-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.6rem; letter-spacing: 0.25em;
        color: rgba(195,245,255,0.6); text-transform: uppercase;
        margin-bottom: 0.3rem; display: block; margin-top: 1.4rem;
    }

    /* ── Footer badge ── */
    .auth-footer-badge {
        margin-top: 1.8rem; display: flex; justify-content: center;
    }
    .auth-footer-inner {
        display: flex; align-items: center; gap: 1rem;
        background: rgba(51,53,57,0.3); backdrop-filter: blur(12px);
        padding: 0.4rem 1.2rem; border-radius: 9999px;
        border: 1px solid rgba(59,73,76,0.1);
    }
    .auth-footer-warn {
        font-family: 'Space Grotesk', sans-serif; font-size: 0.55rem;
        color: #ffabf3; letter-spacing: 0.15em; text-transform: uppercase;
        display: flex; align-items: center; gap: 0.4rem;
    }
    .auth-footer-secure {
        font-family: 'Space Grotesk', sans-serif; font-size: 0.55rem;
        color: #c3f5ff; letter-spacing: 0.15em; text-transform: uppercase;
        display: flex; align-items: center; gap: 0.4rem;
    }
    .auth-divider { width: 1px; height: 0.75rem; background: rgba(59,73,76,0.3); }

    /* ── Left sidebar telemetry (decorative) ── */
    .auth-left-panel {
        position: fixed; left: 3rem; top: 50%; transform: translateY(-50%);
        z-index: 5; pointer-events: none;
        display: flex; flex-direction: column; gap: 2rem;
        width: 16rem;
    }
    .auth-telemetry-label {
        font-family: 'Space Grotesk', sans-serif; font-size: 0.62rem;
        letter-spacing: 0.2em; color: rgba(195,245,255,0.6); text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .auth-market-item {
        display: flex; justify-content: space-between; align-items: flex-end;
        border-bottom: 1px solid rgba(59,73,76,0.2); padding-bottom: 0.5rem; margin-bottom: 0.25rem;
    }
    .auth-market-sym { font-family: 'Space Grotesk', sans-serif; font-size: 0.72rem; color: #bac9cc; }
    .auth-market-val {
        font-family: 'Space Grotesk', sans-serif; font-size: 0.72rem; color: #c3f5ff;
        text-shadow: 0 0 10px rgba(0,229,255,0.5);
    }
    .auth-node-box {
        background: rgba(12,14,18,0.5); border: 1px solid rgba(195,245,255,0.1); padding: 0.75rem 1rem;
    }
    .auth-node-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.3rem; }
    .auth-node-dot {
        width: 0.4rem; height: 0.4rem; border-radius: 50%; background: #c3f5ff;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.2} }
    .auth-node-title { font-family: 'Space Grotesk', sans-serif; font-size: 0.6rem; color: #c3f5ff; font-weight: 700; letter-spacing: 0.1em; }
    .auth-node-detail { font-family: 'Space Grotesk', sans-serif; font-size: 0.55rem; color: #bac9cc; line-height: 1.6; }

    /* ── Right sparkline (decorative) ── */
    .auth-right-panel {
        position: fixed; right: 3rem; bottom: 3.5rem;
        z-index: 5; pointer-events: none; text-align: right;
    }
    .auth-sparkline {
        width: 12rem; height: 3rem; display: flex; align-items: flex-end; gap: 2px;
        padding: 0.25rem 0.5rem; background: rgba(12,14,18,0.5);
        border-right: 2px solid #00e5ff; margin-bottom: 0.5rem; margin-left: auto;
    }
    .auth-spark-bar { flex: 1; background: rgba(195,245,255,0.3); border-radius: 1px; }
    .auth-right-meta { font-family: 'Space Grotesk', sans-serif; font-size: 0.55rem; color: #bac9cc; line-height: 1.8; }

    /* ── OS footer bar ── */
    .auth-os-bar {
        position: fixed; bottom: 0; left: 0; right: 0; height: 2.5rem;
        background: rgba(12,14,18,0.8); backdrop-filter: blur(16px);
        border-top: 1px solid rgba(59,73,76,0.1);
        display: flex; align-items: center; justify-content: space-between;
        padding: 0 2.5rem; z-index: 50; pointer-events: none;
    }
    .auth-os-left { display: flex; align-items: center; gap: 1.5rem; }
    .auth-os-ver { font-family: 'Space Grotesk', sans-serif; font-size: 0.55rem; letter-spacing: 0.3em; color: #c3f5ff; font-weight: 700; }
    .auth-os-status { font-family: 'Space Grotesk', sans-serif; font-size: 0.55rem; letter-spacing: 0.2em; color: #bac9cc; }
    .auth-os-right { display: flex; align-items: center; gap: 2rem; }
    .auth-os-region { font-family: 'Space Grotesk', sans-serif; font-size: 0.55rem; letter-spacing: 0.2em; }
    .auth-os-region span:first-child { color: #bac9cc; }
    .auth-os-region span:last-child { color: #c3f5ff; }
    .auth-os-ts { font-family: 'Space Grotesk', sans-serif; font-size: 0.55rem; letter-spacing: 0.2em; color: #bac9cc; }

    @media(max-width:1100px){.auth-left-panel,.auth-right-panel{display:none!important;}}
    @media(max-width:600px){.auth-canvas{padding:0.75rem;} .glass-panel{padding:1.8rem 1.4rem;}}

    [data-testid="stAlert"] { border-radius: 0.25rem !important; font-size: 0.78rem !important; font-family: 'Space Grotesk', sans-serif !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Decorative background + side panels ──────────────────────────
    st.markdown("""
    <div class="auth-bg">
      <div class="auth-bg-grid"></div>
      <div class="auth-bg-glow-tr"></div>
      <div class="auth-bg-glow-bl"></div>
    </div>

    <!-- Left telemetry panel -->
    <div class="auth-left-panel">
      <div>
        <div class="auth-telemetry-label">System_State</div>
        <div style="display:flex;align-items:center;gap:0.75rem;background:rgba(26,28,32,0.5);padding:0.75rem;border-left:2px solid #00e5ff;">
          <span class="material-symbols-outlined" style="color:#c3f5ff;font-size:1rem;font-variation-settings:'wght' 400;">terminal</span>
          <span style="font-family:'Space Grotesk',sans-serif;font-size:0.7rem;font-weight:700;letter-spacing:0.05em;color:#e2e2e8;">QUANTUM_ENCRYPTION_ACTIVE</span>
        </div>
      </div>
      <div>
        <div class="auth-telemetry-label">Market_Pulse</div>
        <div class="auth-market-item"><span class="auth-market-sym">BTC/USD</span><span class="auth-market-val">64,210.42</span></div>
        <div class="auth-market-item"><span class="auth-market-sym">NASDAQ_100</span><span class="auth-market-val">18,124.50</span></div>
        <div class="auth-market-item"><span class="auth-market-sym">GOLD_SPOT</span><span class="auth-market-val" style="color:#ffabf3;text-shadow:0 0 8px rgba(255,187,243,0.6);">2,382.10</span></div>
      </div>
      <div class="auth-node-box">
        <div class="auth-node-header"><div class="auth-node-dot"></div><span class="auth-node-title">NODE_042 CONNECTED</span></div>
        <div class="auth-node-detail">Latency: 2ms | Region: Orbital_01 | Status: Stable</div>
      </div>
    </div>

    <!-- Right sparkline panel -->
    <div class="auth-right-panel">
      <div class="auth-telemetry-label" style="margin-bottom:0.3rem;">Data_Stream_042</div>
      <div class="auth-sparkline">
        <div class="auth-spark-bar" style="height:30%;"></div>
        <div class="auth-spark-bar" style="height:60%;background:rgba(195,245,255,0.45);"></div>
        <div class="auth-spark-bar" style="height:45%;"></div>
        <div class="auth-spark-bar" style="height:80%;background:rgba(195,245,255,0.6);"></div>
        <div class="auth-spark-bar" style="height:20%;"></div>
        <div class="auth-spark-bar" style="height:50%;background:rgba(195,245,255,0.5);"></div>
        <div class="auth-spark-bar" style="height:95%;background:rgba(195,245,255,0.8);"></div>
      </div>
      <div class="auth-right-meta">PACKET_RECEIVE: 2.4 TB/S<br>COMPRESSION: 14.2:1<br>NEURAL_LOAD: 0.12%</div>
    </div>

    <!-- OS footer bar -->
    <div class="auth-os-bar">
      <div class="auth-os-left">
        <span class="auth-os-ver">STOCKCAST_OS_v4.2.0</span>
        <span class="auth-os-status">STATUS: NOMINAL</span>
      </div>
      <div class="auth-os-right">
        <div class="auth-os-region">
          <span>Region: </span><span>ORBITAL_STATION_SIGMA</span>
        </div>
        <div class="auth-os-ts">FOR EDUCATIONAL PURPOSES ONLY · MUAWWIZ GHANI</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Central glass card (decorative shell) ────────────────────────
    st.markdown("""
    <div class="auth-canvas">
      <div class="auth-terminal">
        <div class="auth-bracket-tl"></div>
        <div class="auth-bracket-br"></div>
        <div class="glass-panel">
          <div class="scanning-line"></div>
          <div class="auth-header">
            <span class="material-symbols-outlined auth-icon">rocket_launch</span>
            <div class="auth-title">STOCKCAST</div>
            <div class="auth-subtitle">Command Center Protocol</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Position Streamlit widgets over the glass card
    st.markdown("""
    <style>
    [data-testid="stMainBlockContainer"] > div > div > div:nth-child(2) {
        position: fixed !important; top: 50% !important; left: 50% !important;
        transform: translate(-50%, -50%) !important;
        width: min(420px, 90vw) !important; z-index: 30 !important;
        padding: 0 2.8rem 2rem !important;
        margin-top: 10rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Tab switcher ─────────────────────────────────────────────────
    _tc1, _tc2 = st.columns(2)
    with _tc1:
        if st.button("INITIALIZE SESSION", key="tab_login", use_container_width=True):
            st.session_state.auth_view = "login"
            st.rerun()
    with _tc2:
        if st.button("REQUEST ACCESS", key="tab_signup", use_container_width=True):
            st.session_state.auth_view = "signup"
            st.rerun()

    if _is_login:
        # ── Login form ───────────────────────────────────────────────
        st.markdown('<span class="auth-field-label">Operator_ID (Email)</span>', unsafe_allow_html=True)
        _login_email = st.text_input("e", placeholder="EMAIL@STOCKCAST.QUANTUM", key="login_email", label_visibility="collapsed")
        st.markdown('<span class="auth-field-label">Security_Cipher</span>', unsafe_allow_html=True)
        _login_pass = st.text_input("p", type="password", placeholder="••••••••••••", key="login_pass", label_visibility="collapsed")
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        if st.button("⚡  INITIALIZE SESSION", key="login_btn", use_container_width=True):
            if not _login_email or not _login_pass:
                _auth_error = "Please enter your Operator ID and Security Cipher."
            else:
                with st.spinner("Authenticating…"):
                    try:
                        _res = supabase.auth.sign_in_with_password({
                            "email": _login_email.strip(),
                            "password": _login_pass,
                        })
                        if _res.user:
                            st.session_state.user = _res.user
                            st.rerun()
                        else:
                            _auth_error = "Invalid credentials. Please try again."
                    except Exception as _e:
                        _auth_error = str(_e)
    else:
        # ── Signup form ──────────────────────────────────────────────
        st.markdown('<span class="auth-field-label">Operator_ID (Email)</span>', unsafe_allow_html=True)
        _signup_email = st.text_input("e2", placeholder="EMAIL@STOCKCAST.QUANTUM", key="signup_email", label_visibility="collapsed")
        st.markdown('<span class="auth-field-label">Security_Cipher</span>', unsafe_allow_html=True)
        _signup_pass = st.text_input("p2", type="password", placeholder="••••••••••••", key="signup_pass", label_visibility="collapsed")
        st.markdown('<span class="auth-field-label">Confirm_Cipher</span>', unsafe_allow_html=True)
        _signup_conf = st.text_input("c2", type="password", placeholder="••••••••••••", key="signup_conf", label_visibility="collapsed")
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        if st.button("⚡  REGISTER OPERATOR NODE", key="signup_btn", use_container_width=True):
            if not _signup_email or not _signup_pass or not _signup_conf:
                _auth_error = "Please fill in all fields."
            elif _signup_pass != _signup_conf:
                _auth_error = "Ciphers do not match."
            elif len(_signup_pass) < 6:
                _auth_error = "Cipher must be at least 6 characters."
            else:
                with st.spinner("Registering operator node…"):
                    try:
                        _res = supabase.auth.sign_up({
                            "email": _signup_email.strip(),
                            "password": _signup_pass,
                        })
                        if _res.user:
                            _auth_success = (
                                f"Access granted! Verification dispatched to "
                                f"{_signup_email.strip()}. Verify before initializing session."
                            )
                            st.session_state.auth_view = "login"
                            st.rerun()
                        else:
                            _auth_error = "Registration failed. Please try again."
                    except Exception as _e:
                        _auth_error = str(_e)

    if _auth_error:
        st.error(f"⚠ {_auth_error}")
    if _auth_success:
        st.success(f"✓ {_auth_success}")

    # ── Security badge ────────────────────────────────────────────────
    st.markdown("""
    <div class="auth-footer-badge">
      <div class="auth-footer-inner">
        <div class="auth-footer-warn">
          <span class="material-symbols-outlined" style="font-size:0.8rem;font-variation-settings:'FILL' 1;">warning</span>
          HFT_VOLATILITY: HIGH
        </div>
        <div class="auth-divider"></div>
        <div class="auth-footer-secure">
          <span class="material-symbols-outlined" style="font-size:0.8rem;font-variation-settings:'FILL' 1;">verified_user</span>
          AES-512 SECURE
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()  # 🚨 Halt — do not render app until authenticated

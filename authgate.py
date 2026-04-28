"""
authgate.py — StockCast Auth Gate
Handles login, signup, and password reset via Supabase Auth.
Called from app.py as: from authgate import render_auth_gate; render_auth_gate(supabase)

Architecture (no components, no iframes, Streamlit 1.56+ compatible):
  - Left panel: single self-contained position:fixed div in one st.markdown call
  - Right panel: native Streamlit widgets offset with margin-left:50vw via CSS
  - No split HTML tags across st.markdown calls
"""

import streamlit as st


def render_auth_gate(supabase):
    # ── Session state ─────────────────────────────────────────────────────────
    if "user" not in st.session_state:
        st.session_state.user = None
    if "_auth_mode" not in st.session_state:
        st.session_state._auth_mode = "login"

    if st.session_state.user is not None:
        return

    mode = st.session_state._auth_mode

    # ── Build bar chart HTML ──────────────────────────────────────────────────
    _bar_heights = [20, 35, 45, 25, 60, 80, 70, 90, 100, 85, 65, 40, 20]
    _bars = ""
    for h in _bar_heights:
        cls = "ac" if h == 100 else ("md" if h >= 60 else "")
        _bars += f'<div class="b {cls}" style="height:{h}%"></div>'

    _mini = "".join(
        f'<div class="mb" style="height:{int(h*44/100)}px;opacity:{0.2+0.8*(h/100):.2f}"></div>'
        for h in [30, 60, 45, 80, 100, 70, 40]
    )

    # ── Ticker data ───────────────────────────────────────────────────────────
    _ticks = []
    try:
        import yfinance as _yf
        _syms = ["AAPL", "MSFT", "TSLA", "GOOG", "NVDA", "META", "JPM"]
        _td = _yf.download(_syms, period="2d", interval="1d", progress=False, auto_adjust=True)
        for s in _syms:
            try:
                row = _td["Close"][s].dropna()
                if len(row) >= 2:
                    px  = float(row.iloc[-1])
                    chg = (px - float(row.iloc[-2])) / float(row.iloc[-2]) * 100
                    col = "#8eff71" if chg >= 0 else "#ff7351"
                    arr = "▲" if chg >= 0 else "▼"
                    _ticks.append(
                        f'<span class="ti"><b>{s}</b> {px:.2f} '
                        f'<span style="color:{col}">{arr}{abs(chg):.2f}%</span></span>'
                    )
            except Exception:
                pass
    except Exception:
        pass
    if not _ticks:
        _ticks = [f'<span class="ti"><b>{s}</b> <span style="color:#8eff71">–</span></span>'
                  for s in ["AAPL","MSFT","TSLA","NVDA","META","GOOG","JPM"]]
    _tape = "".join(_ticks * 4)

    # ─────────────────────────────────────────────────────────────────────────
    # GLOBAL CSS + LEFT PANEL (one self-contained fixed div — no split tags)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{{
  --bg:#0e0e0e;--surf:#1a1919;--mid:#262626;--hi:#201f1f;
  --bdr:rgba(90,88,87,.28);--p:#8eff71;--pg:rgba(142,255,113,.3);
  --pd:rgba(142,255,113,.08);--txt:#fff;--mt:#adaaaa;
  --fh:'Space Grotesk',sans-serif;--fb:'Inter',sans-serif;
  --fm:'JetBrains Mono',monospace;
}}
html,body,[class*="css"]{{background:var(--bg)!important;color:var(--txt)!important;font-family:var(--fb)!important;}}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],.stDeployButton{{display:none!important;}}
[data-testid="stSidebar"]{{display:none!important;}}

/* Right panel — offset to right half */
.main .block-container{{max-width:100%!important;padding:0!important;margin:0!important;}}
[data-testid="stMainBlockContainer"]{{padding:0!important;}}
[data-testid="stVerticalBlock"]{{padding:0 4vw!important;margin-left:50vw!important;min-height:100vh!important;display:flex!important;flex-direction:column!important;justify-content:center!important;}}

/* Tab bar */
.sc-tabs{{display:flex;gap:1.8rem;margin-bottom:2rem;border-bottom:1px solid var(--bdr);}}
.sc-tab{{padding-bottom:.85rem;font-family:var(--fh);font-size:.58rem;text-transform:uppercase;letter-spacing:.16em;font-weight:700;color:var(--mt);margin-bottom:-1px;}}
.sc-tab.on{{color:var(--p);border-bottom:2px solid var(--p);}}
/* Input labels */
.sc-lbl{{font-family:var(--fh);font-size:.58rem;text-transform:uppercase;letter-spacing:.14em;font-weight:700;color:var(--mt);margin:.9rem 0 .3rem;display:block;}}
/* Streamlit inputs */
.stTextInput label{{display:none!important;}}
.stTextInput>div>div>input{{background:var(--mid)!important;border:1px solid var(--bdr)!important;border-radius:4px!important;color:var(--txt)!important;font-family:var(--fb)!important;font-size:.88rem!important;padding:.8rem 1rem!important;}}
.stTextInput>div>div>input:focus{{border-color:var(--p)!important;box-shadow:0 0 0 3px rgba(142,255,113,.10)!important;outline:none!important;}}
.stTextInput>div>div>input::placeholder{{color:var(--mt)!important;opacity:.5!important;}}
/* Buttons */
.stButton>button{{width:100%!important;border-radius:4px!important;font-family:var(--fh)!important;font-size:.68rem!important;font-weight:700!important;letter-spacing:.12em!important;text-transform:uppercase!important;padding:.82rem!important;transition:all .18s!important;}}
.stButton>button[kind="primary"]{{background:var(--p)!important;border:none!important;color:#053800!important;box-shadow:0 0 18px var(--pg)!important;margin-top:.5rem!important;}}
.stButton>button[kind="primary"]:hover{{filter:brightness(1.1)!important;transform:translateY(-1px)!important;box-shadow:0 0 30px var(--pg)!important;}}
.stButton>button[kind="secondary"]{{background:var(--hi)!important;border:1px solid var(--bdr)!important;color:var(--mt)!important;font-size:.58rem!important;}}
.stButton>button[kind="secondary"]:hover{{background:var(--mid)!important;color:var(--txt)!important;}}
/* Info cards */
.sc-trial{{margin-top:1.4rem;padding:.9rem 1rem;background:var(--pd);border:1px solid rgba(142,255,113,.18);border-radius:4px;display:flex;gap:.75rem;align-items:flex-start;}}
.sc-trial-i{{color:var(--p);flex-shrink:0;margin-top:1px;}}
.sc-trial-t{{font-size:.78rem;font-weight:600;margin-bottom:.2rem;}}
.sc-trial-b{{font-size:.7rem;color:var(--mt);line-height:1.55;}}
.sc-shariah{{display:flex;align-items:center;gap:.5rem;background:rgba(212,168,83,.07);border:1px solid rgba(212,168,83,.22);border-radius:4px;padding:.65rem .9rem;margin:1rem 0 .5rem;font-size:.72rem;color:#d4a853;font-weight:500;}}
/* Log widget */
.sc-log{{position:fixed;bottom:1.5rem;right:1.5rem;width:215px;background:rgba(30,30,30,.8);backdrop-filter:blur(16px);border:1px solid var(--bdr);border-radius:6px;padding:.65rem .85rem;z-index:9999;}}
@media(max-width:1100px){{.sc-log{{display:none;}}}}
.sc-log-row{{display:flex;justify-content:space-between;margin-bottom:.35rem;}}
.sc-log-lbl{{font-family:var(--fh);font-size:.48rem;text-transform:uppercase;letter-spacing:.12em;color:var(--mt);}}
.sc-log-live{{font-family:var(--fh);font-size:.48rem;text-transform:uppercase;letter-spacing:.1em;color:var(--p);}}
.sc-log-line{{font-family:var(--fm);font-size:.46rem;color:#666;margin-bottom:2px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;}}
.sc-log-line.ok{{color:rgba(142,255,113,.55);}}
/* Alerts */
.stAlert{{border-radius:4px!important;font-size:.8rem!important;}}
/* Left panel internal styles */
@keyframes slideIn{{from{{opacity:0;transform:translateX(-16px)}}to{{opacity:1;transform:translateX(0)}}}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
@keyframes tick{{0%{{transform:translateX(0)}}100%{{transform:translateX(-50%)}}}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}
.sc-right-anim{{animation:fadeUp .4s ease both;}}
</style>

<!-- LEFT PANEL — self-contained position:fixed div, no split tags -->
<div style="position:fixed;top:0;left:0;width:50vw;height:100vh;background:#1a1919;display:flex;flex-direction:column;justify-content:center;padding:3rem 4.5rem;overflow:hidden;animation:slideIn .5s ease both;z-index:50;">

  <!-- Dot grid -->
  <div style="position:absolute;inset:0;opacity:.14;background-image:radial-gradient(circle at 2px 2px,#555 1px,transparent 0);background-size:24px 24px;pointer-events:none;"></div>
  <!-- Right edge fade -->
  <div style="position:absolute;top:0;right:0;width:60px;height:100%;background:linear-gradient(to right,transparent,rgba(10,10,10,.55));pointer-events:none;z-index:2;"></div>

  <!-- Brand -->
  <div style="position:absolute;top:2.2rem;left:4.5rem;z-index:3;">
    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.15rem;font-weight:800;letter-spacing:-.03em;color:#8eff71;">STOCKCAST</div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:.5rem;text-transform:uppercase;letter-spacing:.18em;color:#adaaaa;margin-top:.1rem;">Institutional Access</div>
  </div>

  <!-- Hero -->
  <div style="position:relative;z-index:3;">
    <h2 style="font-family:'Space Grotesk',sans-serif;font-size:clamp(1.7rem,3.2vw,2.8rem);font-weight:800;line-height:1.08;letter-spacing:-.04em;color:#fff;margin:0 0 .9rem;">
      Predicting the<br><span style="color:#8eff71;">pulse of global</span><br>markets
    </h2>
    <p style="font-family:'Space Grotesk',sans-serif;color:#adaaaa;font-size:.85rem;max-width:320px;line-height:1.65;margin-bottom:1.6rem;">
      High-frequency predictive intelligence for professional traders and institutional desk managers.
    </p>
    <ul style="list-style:none;padding:0;margin:0 0 1.8rem;">
      <li style="display:flex;align-items:center;gap:.55rem;font-family:'Space Grotesk',sans-serif;font-size:.55rem;text-transform:uppercase;letter-spacing:.13em;font-weight:700;color:#adaaaa;margin-bottom:.7rem;">
        <span style="width:6px;height:6px;border-radius:50%;background:#8eff71;box-shadow:0 0 8px rgba(142,255,113,.3);flex-shrink:0;"></span>XGBoost Forecasting Engine
      </li>
      <li style="display:flex;align-items:center;gap:.55rem;font-family:'Space Grotesk',sans-serif;font-size:.55rem;text-transform:uppercase;letter-spacing:.13em;font-weight:700;color:#adaaaa;margin-bottom:.7rem;">
        <span style="width:6px;height:6px;border-radius:50%;background:#8eff71;box-shadow:0 0 8px rgba(142,255,113,.3);flex-shrink:0;"></span>Shariah Compliance Screening
      </li>
      <li style="display:flex;align-items:center;gap:.55rem;font-family:'Space Grotesk',sans-serif;font-size:.55rem;text-transform:uppercase;letter-spacing:.13em;font-weight:700;color:#adaaaa;margin-bottom:.7rem;">
        <span style="width:6px;height:6px;border-radius:50%;background:#8eff71;box-shadow:0 0 8px rgba(142,255,113,.3);flex-shrink:0;"></span>Institutional-Grade Security
      </li>
    </ul>
    <!-- Alpha card -->
    <div style="background:rgba(10,10,10,.72);border:1px solid rgba(90,88,87,.28);border-radius:8px;padding:1.1rem 1.3rem;max-width:340px;backdrop-filter:blur(20px);">
      <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:.7rem;">
        <div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:.52rem;text-transform:uppercase;letter-spacing:.13em;color:#adaaaa;margin-bottom:.18rem;">Live Alpha Stream</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.45rem;font-weight:700;color:#8eff71;line-height:1;">0.942<span style="font-size:.5rem;font-weight:400;color:#adaaaa;margin-left:.28rem;">CONFIDENCE</span></div>
        </div>
        <div style="display:flex;gap:2px;height:44px;align-items:flex-end;">{_mini}</div>
      </div>
      <div style="display:flex;gap:2px;height:64px;align-items:flex-end;">{_bars}</div>
    </div>
  </div>

  <!-- Ticker tape -->
  <div style="position:absolute;bottom:4rem;left:0;right:0;height:1.5rem;overflow:hidden;border-top:1px solid rgba(90,88,87,.28);border-bottom:1px solid rgba(90,88,87,.28);background:rgba(10,10,10,.45);">
    <div style="display:flex;align-items:center;height:100%;white-space:nowrap;animation:tick 28s linear infinite;">
      {_tape}
    </div>
  </div>

  <!-- Status bar -->
  <div style="position:absolute;bottom:1.6rem;left:4.5rem;display:flex;gap:1.4rem;z-index:3;">
    <div style="display:flex;align-items:center;gap:.38rem;font-family:'Space Grotesk',sans-serif;font-size:.52rem;text-transform:uppercase;letter-spacing:.12em;color:#adaaaa;">
      <span style="width:6px;height:6px;border-radius:50%;background:#8eff71;box-shadow:0 0 8px rgba(142,255,113,.3);animation:pulse 2s infinite;"></span>Engine Online
    </div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:.52rem;text-transform:uppercase;letter-spacing:.12em;color:#adaaaa;">Uptime 99.99%</div>
  </div>

</div>
<!-- END LEFT PANEL -->

<!-- System log widget -->
<div class="sc-log">
  <div class="sc-log-row"><span class="sc-log-lbl">System Logs</span><span class="sc-log-live">● Live</span></div>
  <div class="sc-log-line">Requesting handshake...</div>
  <div class="sc-log-line ok">Access token validated ✓</div>
  <div class="sc-log-line">Encrypted tunnel active</div>
</div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # RIGHT PANEL — Streamlit widgets
    # ─────────────────────────────────────────────────────────────────────────
    tl = "sc-tab on" if mode == "login"  else "sc-tab"
    ts = "sc-tab on" if mode == "signup" else "sc-tab"
    tr = "sc-tab on" if mode == "reset"  else "sc-tab"

    st.markdown(f"""
<div class="sc-right-anim" style="margin-top:2rem;">
  <div class="sc-tabs">
    <span class="{tl}">Terminal Access</span>
    <span class="{ts}">Create Account</span>
    <span class="{tr}">Forgot Key</span>
  </div>
</div>
    """, unsafe_allow_html=True)

    # ── LOGIN ─────────────────────────────────────────────────────────────────
    if mode == "login":
        st.markdown('<span class="sc-lbl">Identity Token (Email)</span>', unsafe_allow_html=True)
        email = st.text_input("_l_e", placeholder="operator@stockcast.ai",
                              key="ag_login_email", label_visibility="collapsed")
        st.markdown('<span class="sc-lbl">Access Key</span>', unsafe_allow_html=True)
        password = st.text_input("_l_p", type="password", placeholder="••••••••••••",
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
  <div class="sc-trial-i">ℹ</div>
  <div>
    <div class="sc-trial-t">Alpha Stream — Limited Offer</div>
    <div class="sc-trial-b">Initialize your account today and receive a 14-day institutional-grade trial with full prediction logs.</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── SIGN UP ───────────────────────────────────────────────────────────────
    elif mode == "signup":
        st.markdown('<span class="sc-lbl">Identity Token (Email)</span>', unsafe_allow_html=True)
        email = st.text_input("_s_e", placeholder="operator@stockcast.ai",
                              key="ag_signup_email", label_visibility="collapsed")
        st.markdown('<span class="sc-lbl">Access Key (min 6 chars)</span>', unsafe_allow_html=True)
        password = st.text_input("_s_p", type="password", placeholder="••••••••••••",
                                 key="ag_signup_pw", label_visibility="collapsed")
        st.markdown('<span class="sc-lbl">Confirm Access Key</span>', unsafe_allow_html=True)
        password2 = st.text_input("_s_p2", type="password", placeholder="••••••••••••",
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
        email = st.text_input("_r_e", placeholder="operator@stockcast.ai",
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

    # ── Mode switcher buttons ─────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.8rem;'></div>", unsafe_allow_html=True)
    st.caption("Switch mode ↓")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🖥 Terminal Access", key="sw_login", use_container_width=True):
            st.session_state._auth_mode = "login"; st.rerun()
    with c2:
        if st.button("📝 Create Account",  key="sw_signup", use_container_width=True):
            st.session_state._auth_mode = "signup"; st.rerun()
    with c3:
        if st.button("🔑 Forgot Key",      key="sw_reset", use_container_width=True):
            st.session_state._auth_mode = "reset"; st.rerun()

    st.stop()

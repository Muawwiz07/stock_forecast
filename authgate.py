"""
authgate.py  –  Stockcast authentication gate
Single-page login/signup: injects CSS into Streamlit's own DOM so there is
only ONE set of inputs and ONE submit button.  No iframe, no postMessage.
app.py calls:  render_auth_gate(supabase)
"""

import streamlit as st

# ── Page-wide CSS + animated canvas background ────────────────────────────────
_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
*, *::before, *::after { box-sizing: border-box; }
:root {
  --green:#00ff88; --green-dim:#00cc66;
  --green-glow:rgba(0,255,136,0.35); --green-faint:rgba(0,255,136,0.08);
  --navy:#020d1a; --panel:rgba(4,20,40,0.94);
  --border:rgba(0,255,136,0.18); --text:#c8e8d8;
  --muted:rgba(180,220,200,0.45);
}

/* Full-page dark background */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--navy) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none !important; }
footer { display: none !important; }
[data-testid="stMainBlockContainer"] {
  padding-top: 0 !important;
  max-width: 100% !important;
}

/* Animated canvas background */
#sc-bg-canvas {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
}
.sc-grid-overlay {
  position: fixed; inset: 0; z-index: 1; pointer-events: none;
  background-image:
    linear-gradient(rgba(0,255,136,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,255,136,0.04) 1px, transparent 1px);
  background-size: 48px 48px;
  mask-image: radial-gradient(ellipse 80% 60% at 50% 100%, black 30%, transparent 80%);
}
.sc-scanline {
  position: fixed; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--green), transparent);
  opacity: 0.4; z-index: 2; animation: sc-scan 6s linear infinite;
}
@keyframes sc-scan { from { top: -2px; } to { top: 100%; } }

/* Particles */
.sc-particles { position: fixed; inset: 0; z-index: 1; pointer-events: none; }
.sc-particle {
  position: absolute; border-radius: 50%;
  background: var(--green); box-shadow: 0 0 4px var(--green);
  animation: sc-floatUp linear infinite; opacity: 0;
}
@keyframes sc-floatUp {
  0%   { opacity:0; transform: translateY(0) scale(0.5); }
  10%  { opacity:0.8; }
  90%  { opacity:0.4; }
  100% { opacity:0; transform: translateY(-120px) scale(0.2); }
}

/* Page layout */
.sc-page {
  position: relative; z-index: 10;
  min-height: 100vh;
  display: grid;
  grid-template-columns: 1fr 440px;
  grid-template-rows: auto 1fr auto;
  padding: 0 48px;
  font-family: 'Rajdhani', sans-serif;
  color: var(--text);
}
@media (max-width: 800px) {
  .sc-page { grid-template-columns: 1fr; padding: 0 18px; }
  .sc-hero { display: none !important; }
}

/* Header */
.sc-header {
  grid-column: 1 / -1;
  display: flex; align-items: center; justify-content: space-between;
  padding: 22px 0 18px;
  border-bottom: 1px solid var(--border);
}
.sc-brand-name {
  font-family: 'Orbitron', monospace; font-size: 18px; font-weight: 700;
  color: var(--green); letter-spacing: 3px;
  text-shadow: 0 0 18px var(--green-glow);
}
.sc-brand-sub {
  font-family: 'Share Tech Mono', monospace; font-size: 9px;
  color: var(--muted); letter-spacing: 4px;
}
.sc-clock-wrap {
  font-family: 'Share Tech Mono', monospace; font-size: 11px;
  color: var(--muted); letter-spacing: 2px;
}
#sc-clock { color: var(--green); }

/* Hero */
.sc-hero {
  display: flex; flex-direction: column; justify-content: center;
  padding-right: 60px; padding-bottom: 40px;
  animation: sc-fadeUp 1s ease both;
}
@keyframes sc-fadeUp { from { opacity:0; transform:translateY(24px); } to { opacity:1; transform:none; } }
.sc-hero-tag {
  font-family: 'Share Tech Mono', monospace; font-size: 10px; letter-spacing: 5px;
  color: var(--green); margin-bottom: 20px;
  display: flex; align-items: center; gap: 10px;
}
.sc-hero-tag::before {
  content: ''; display: block; width: 24px; height: 1px;
  background: var(--green); box-shadow: 0 0 6px var(--green);
}
.sc-live-badge {
  display: inline-flex; align-items: center; gap: 6px; margin-bottom: 28px;
  font-family: 'Share Tech Mono', monospace; font-size: 10px;
  letter-spacing: 3px; color: var(--green);
}
.sc-live-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--green); box-shadow: 0 0 8px var(--green);
  animation: sc-pulse 1.4s ease-in-out infinite;
}
@keyframes sc-pulse {
  0%,100% { opacity:1; transform:scale(1); }
  50% { opacity:0.4; transform:scale(0.6); }
}
.sc-hero h1 {
  font-family: 'Orbitron', monospace;
  font-size: clamp(28px, 3.5vw, 52px); font-weight: 900; line-height: 1.12;
  color: #fff; text-shadow: 0 0 40px rgba(0,255,136,0.12); margin-bottom: 10px;
}
.sc-hero h1 span { color: var(--green); }
.sc-hero-desc {
  font-size: 13px; color: var(--muted); line-height: 1.7;
  max-width: 360px; margin-top: 16px; margin-bottom: 40px;
  font-weight: 300; letter-spacing: 0.5px;
}
.sc-stats { display: flex; gap: 16px; flex-wrap: wrap; }
.sc-stat-pill {
  background: var(--green-faint); border: 1px solid var(--border);
  border-radius: 6px; padding: 10px 18px;
  display: flex; flex-direction: column; gap: 2px; backdrop-filter: blur(8px);
}
.sc-stat-val {
  font-family: 'Orbitron', monospace; font-size: 20px; font-weight: 700;
  color: var(--green); text-shadow: 0 0 10px var(--green-glow);
}
.sc-stat-lbl {
  font-family: 'Share Tech Mono', monospace; font-size: 9px;
  letter-spacing: 3px; color: var(--muted);
}

/* Right panel column */
.sc-panel-col {
  display: flex; flex-direction: column; justify-content: center;
  padding-bottom: 40px; padding-top: 20px;
  animation: sc-fadeUp 1s 0.2s ease both;
}

/* Footer */
.sc-footer {
  grid-column: 1 / -1;
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 0; border-top: 1px solid var(--border);
  font-family: 'Share Tech Mono', monospace; font-size: 10px;
  letter-spacing: 2px; color: var(--muted);
}
.sc-footer a { color: var(--muted); text-decoration: none; margin-right: 20px; }
.sc-footer a:hover { color: var(--green); }

/* ── Streamlit widget overrides ── */

/* Text inputs */
.stTextInput input {
  background: rgba(0,255,136,0.04) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 14px !important;
  letter-spacing: 1px !important;
}
.stTextInput input:focus {
  border-color: var(--green-dim) !important;
  box-shadow: 0 0 0 3px rgba(0,255,136,0.1) !important;
}
.stTextInput input::placeholder { color: rgba(180,220,200,0.25) !important; }
.stTextInput label {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 9px !important; letter-spacing: 4px !important;
  color: var(--muted) !important;
}

/* Submit / action buttons */
.stFormSubmitButton > button, .stButton > button {
  background: linear-gradient(135deg, #00cc66, #00ff88, #00cc66) !important;
  background-size: 200% 200% !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 11px !important; font-weight: 700 !important;
  letter-spacing: 4px !important;
  color: #020d1a !important;
  width: 100% !important;
  padding: 14px !important;
  box-shadow: 0 0 24px rgba(0,255,136,0.35), 0 4px 16px rgba(0,0,0,0.4) !important;
  animation: sc-shimmer 3s linear infinite !important;
}
@keyframes sc-shimmer {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.stFormSubmitButton > button:hover, .stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 0 40px rgba(0,255,136,0.55), 0 8px 24px rgba(0,0,0,0.5) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
  gap: 0 !important;
  margin-bottom: 20px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 9px !important; letter-spacing: 3px !important;
  color: var(--muted) !important;
  border: none !important;
  flex: 1 !important; justify-content: center !important;
}
.stTabs [aria-selected="true"] {
  background: var(--green-faint) !important;
  color: var(--green) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* Glass card around the form area */
.sc-glass {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 32px 28px;
  backdrop-filter: blur(24px);
  box-shadow:
    0 0 0 1px rgba(0,255,136,0.06),
    0 32px 80px rgba(0,0,0,0.6),
    inset 0 1px 0 rgba(0,255,136,0.1);
  position: relative;
}
.sc-glass::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--green), transparent);
  opacity: 0.5;
}
.sc-panel-title {
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px; letter-spacing: 5px;
  color: var(--muted); text-align: center;
  margin-bottom: 20px;
}
</style>

<!-- Background layers -->
<canvas id="sc-bg-canvas"></canvas>
<div class="sc-grid-overlay"></div>
<div class="sc-scanline"></div>
<div class="sc-particles" id="sc-particles"></div>

<script>
(function(){
  // Clock
  function tick() {
    const el = document.getElementById('sc-clock');
    if (el) el.textContent = [new Date().getHours(), new Date().getMinutes(), new Date().getSeconds()]
      .map(n=>String(n).padStart(2,'0')).join(':');
  }
  tick(); setInterval(tick, 1000);

  // Particles
  const pc = document.getElementById('sc-particles');
  if (pc) for (let i=0;i<30;i++){
    const p=document.createElement('div'); p.className='sc-particle';
    p.style.left=Math.random()*50+'%'; p.style.bottom=Math.random()*40+'vh';
    p.style.animationDuration=(6+Math.random()*10)+'s';
    p.style.animationDelay=(Math.random()*12)+'s';
    const s=(1+Math.random()*2)+'px'; p.style.width=p.style.height=s;
    pc.appendChild(p);
  }

  // Canvas
  const cv=document.getElementById('sc-bg-canvas');
  if(!cv) return;
  const ctx=cv.getContext('2d');
  function rsz(){cv.width=window.innerWidth;cv.height=window.innerHeight;}
  rsz(); window.addEventListener('resize',rsz);
  let t=0;
  function draw(){
    ctx.clearRect(0,0,cv.width,cv.height);
    const cx=cv.width*.38,cy=cv.height*.65;
    const g=ctx.createRadialGradient(cx,cy,0,cx,cy,340);
    g.addColorStop(0,'rgba(0,255,136,0.14)');
    g.addColorStop(.4,'rgba(0,100,255,0.06)');
    g.addColorStop(1,'transparent');
    ctx.fillStyle=g; ctx.fillRect(0,0,cv.width,cv.height);
    for(let i=0;i<22;i++){
      const x=80+i*28,ph=i*.4+t;
      const mid=cy-80-Math.sin(ph)*55-i*4;
      const op=mid+Math.sin(ph*1.3)*20,cl=mid+Math.cos(ph*.9)*25;
      const hi=Math.min(op,cl)-Math.abs(Math.sin(ph*2.1))*18-6;
      const lo=Math.max(op,cl)+Math.abs(Math.cos(ph*1.7))*14+5;
      const bull=cl<op,a=.45+.55*(i/22);
      ctx.strokeStyle=bull?`rgba(0,255,136,${a*.7})`:`rgba(255,80,120,${a*.7})`;
      ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x+9,hi);ctx.lineTo(x+9,lo);ctx.stroke();
      ctx.fillStyle=bull?`rgba(0,255,136,${a*.85})`:`rgba(255,60,100,${a*.7})`;
      ctx.shadowColor=bull?'rgba(0,255,136,0.6)':'rgba(255,60,100,0.4)';ctx.shadowBlur=8;
      ctx.fillRect(x,Math.min(op,cl),18,Math.abs(op-cl)||2);ctx.shadowBlur=0;
    }
    t+=.003;requestAnimationFrame(draw);
  }
  draw();
})();
</script>
"""

_HEADER_HTML = """
<div class="sc-page">
  <div class="sc-header">
    <div>
      <div class="sc-brand-name">STOCKCAST</div>
      <div class="sc-brand-sub">DEVELOPED BY MUAWWIZ GHANI</div>
    </div>
    <div class="sc-clock-wrap">TIME &nbsp;<span id="sc-clock">00:00:00</span></div>
  </div>

  <div class="sc-hero">
    <div class="sc-hero-tag">LIVE ALPHA STREAM</div>
    <div class="sc-live-badge"><div class="sc-live-dot"></div>MARKETS ACTIVE</div>
    <h1>Predicting the<br>pulse of <span>global</span><br>markets.</h1>
    <p class="sc-hero-desc">
      Stockcast is a futuristic terminal for global markets — it analyzes,
      monitors in real-time, and predicts the pulse of equities with
      machine learning precision.
    </p>
    <div class="sc-stats">
      <div class="sc-stat-pill">
        <div class="sc-stat-val">98.4%</div><div class="sc-stat-lbl">CONFIDENCE</div>
      </div>
      <div class="sc-stat-pill">
        <div class="sc-stat-val">12ms</div><div class="sc-stat-lbl">LATENCY</div>
      </div>
      <div class="sc-stat-pill">
        <div class="sc-stat-val">1,240+</div><div class="sc-stat-lbl">TICKERS</div>
      </div>
    </div>
  </div>

  <div class="sc-panel-col">
    <div class="sc-glass">
      <div class="sc-panel-title">SECURE TERMINAL ACCESS</div>
"""

_FOOTER_HTML = """
    </div><!-- /.sc-glass -->
  </div><!-- /.sc-panel-col -->

  <div class="sc-footer">
    <div><a href="#">PRIVACY</a><a href="#">TERMS</a><a href="#">CONTACT</a></div>
    <div>STOCKCAST &copy; 2026</div>
  </div>
</div><!-- /.sc-page -->
"""


def render_auth_gate(supabase_client):
    """
    Render the full-screen Stockcast login page.
    Sets st.session_state.user on success, calls st.stop() otherwise.
    """
    if "user" not in st.session_state:
        st.session_state.user = None
    if "_auth_msg" not in st.session_state:
        st.session_state._auth_msg = ("", "")

    if st.session_state.user is not None:
        return  # Already authenticated — let app.py continue

    # Inject CSS + background animations
    st.markdown(_CSS, unsafe_allow_html=True)

    # Inject header + hero + open the glass panel div
    st.markdown(_HEADER_HTML, unsafe_allow_html=True)

    # Show error/success from previous attempt
    msg_text, msg_type = st.session_state._auth_msg
    if msg_text:
        if msg_type == "error":
            st.error(msg_text)
        else:
            st.success(msg_text)

    # ── Real Streamlit tabs + forms ──
    tab_login, tab_signup = st.tabs(["LOGIN", "SIGN UP"])

    with tab_login:
        with st.form("_sc_login", clear_on_submit=False):
            email_l = st.text_input("IDENTITY TOKEN (EMAIL)", placeholder="name@firm.com",
                                    key="_sc_l_email")
            pass_l  = st.text_input("ACCESS KEY", type="password",
                                    placeholder="••••••••••••", key="_sc_l_pass")
            submitted_l = st.form_submit_button("AUTHORIZE ACCESS", use_container_width=True)

        if submitted_l:
            if not email_l or not pass_l:
                st.session_state._auth_msg = ("⚠ All fields are required.", "error")
                st.rerun()
            else:
                try:
                    res = supabase_client.auth.sign_in_with_password(
                        {"email": email_l, "password": pass_l}
                    )
                    st.session_state.user = res.user
                    st.session_state._auth_msg = ("", "")
                    st.rerun()
                except Exception as exc:
                    st.session_state._auth_msg = (f"Login failed: {exc}", "error")
                    st.rerun()

    with tab_signup:
        with st.form("_sc_signup", clear_on_submit=False):
            email_s = st.text_input("IDENTITY TOKEN (EMAIL)", placeholder="name@firm.com",
                                    key="_sc_s_email")
            pass_s  = st.text_input("ACCESS KEY", type="password",
                                    placeholder="Min 6 characters", key="_sc_s_pass")
            pass_s2 = st.text_input("CONFIRM ACCESS KEY", type="password",
                                    placeholder="Repeat password", key="_sc_s_pass2")
            submitted_s = st.form_submit_button("CREATE ACCOUNT", use_container_width=True)

        if submitted_s:
            if not email_s or not pass_s or not pass_s2:
                st.session_state._auth_msg = ("⚠ All fields are required.", "error")
                st.rerun()
            elif pass_s != pass_s2:
                st.session_state._auth_msg = ("⚠ Passwords do not match.", "error")
                st.rerun()
            elif len(pass_s) < 6:
                st.session_state._auth_msg = ("⚠ Password must be at least 6 characters.", "error")
                st.rerun()
            else:
                try:
                    res = supabase_client.auth.sign_up(
                        {"email": email_s, "password": pass_s}
                    )
                    if res.user:
                        st.session_state._auth_msg = (
                            "✓ Account created! Check your email to confirm, then log in.",
                            "success"
                        )
                    else:
                        st.session_state._auth_msg = ("Sign-up failed. Please try again.", "error")
                    st.rerun()
                except Exception as exc:
                    st.session_state._auth_msg = (f"Sign-up failed: {exc}", "error")
                    st.rerun()

    # Close the glass panel + footer
    st.markdown(_FOOTER_HTML, unsafe_allow_html=True)

    st.stop()

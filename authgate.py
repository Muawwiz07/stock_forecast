"""
authgate.py  –  Stockcast authentication gate
Renders the login/signup UI and handles Supabase auth.
app.py calls:  render_auth_gate(supabase)
"""

import streamlit as st
import streamlit.components.v1 as components


# ── HTML / CSS / JS login shell ───────────────────────────────────────────────
_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stockcast · Login</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --green: #00ff88; --green-dim: #00cc66;
    --green-glow: rgba(0,255,136,0.35); --green-faint: rgba(0,255,136,0.08);
    --navy: #020d1a; --panel: rgba(4,20,40,0.92);
    --border: rgba(0,255,136,0.18); --text: #c8e8d8;
    --muted: rgba(180,220,200,0.45);
  }
  html, body { height: 100%; background: var(--navy); font-family: 'Rajdhani', sans-serif; color: var(--text); overflow-x: hidden; }
  #bg { position: fixed; inset: 0; z-index: 0; }
  .grid-overlay {
    position: fixed; inset: 0; z-index: 1; pointer-events: none;
    background-image: linear-gradient(rgba(0,255,136,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,136,0.04) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(ellipse 80% 60% at 50% 100%, black 30%, transparent 80%);
  }
  .scanline { position: fixed; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--green), transparent); opacity: 0.4; z-index: 2; animation: scan 6s linear infinite; }
  @keyframes scan { from { top: -2px; } to { top: 100%; } }
  .page { position: relative; z-index: 10; min-height: 100vh; display: grid; grid-template-columns: 1fr 420px; grid-template-rows: auto 1fr auto; padding: 0 48px; }
  header { grid-column: 1 / -1; display: flex; align-items: center; justify-content: space-between; padding: 22px 0 18px; border-bottom: 1px solid var(--border); }
  .brand { display: flex; flex-direction: column; gap: 2px; }
  .brand-name { font-family: 'Orbitron', monospace; font-size: 18px; font-weight: 700; color: var(--green); letter-spacing: 3px; text-shadow: 0 0 18px var(--green-glow); }
  .brand-sub { font-family: 'Share Tech Mono', monospace; font-size: 9px; color: var(--muted); letter-spacing: 4px; }
  .header-right { font-family: 'Share Tech Mono', monospace; font-size: 11px; color: var(--muted); letter-spacing: 2px; }
  #clock { color: var(--green); }
  .hero { display: flex; flex-direction: column; justify-content: center; padding-right: 60px; padding-bottom: 40px; animation: fadeUp 1s ease both; }
  @keyframes fadeUp { from { opacity:0; transform:translateY(24px); } to { opacity:1; transform:none; } }
  .hero-tag { font-family: 'Share Tech Mono', monospace; font-size: 10px; letter-spacing: 5px; color: var(--green); margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
  .hero-tag::before { content: ''; display: block; width: 24px; height: 1px; background: var(--green); box-shadow: 0 0 6px var(--green); }
  .hero h1 { font-family: 'Orbitron', monospace; font-size: clamp(32px, 4vw, 52px); font-weight: 900; line-height: 1.12; color: #fff; text-shadow: 0 0 40px rgba(0,255,136,0.12); margin-bottom: 10px; }
  .hero h1 span { color: var(--green); }
  .hero-desc { font-size: 13px; color: var(--muted); line-height: 1.7; max-width: 360px; margin-top: 16px; margin-bottom: 40px; font-weight: 300; letter-spacing: 0.5px; }
  .stats { display: flex; gap: 20px; flex-wrap: wrap; }
  .stat-pill { background: var(--green-faint); border: 1px solid var(--border); border-radius: 6px; padding: 10px 18px; display: flex; flex-direction: column; gap: 2px; backdrop-filter: blur(8px); }
  .stat-val { font-family: 'Orbitron', monospace; font-size: 20px; font-weight: 700; color: var(--green); text-shadow: 0 0 10px var(--green-glow); }
  .stat-lbl { font-family: 'Share Tech Mono', monospace; font-size: 9px; letter-spacing: 3px; color: var(--muted); }
  .live-badge { display: inline-flex; align-items: center; gap: 6px; margin-bottom: 28px; font-family: 'Share Tech Mono', monospace; font-size: 10px; letter-spacing: 3px; color: var(--green); }
  .live-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse 1.4s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity:1; transform: scale(1); } 50% { opacity:0.4; transform: scale(0.6); } }
  .panel-col { display: flex; flex-direction: column; justify-content: center; padding-bottom: 40px; animation: fadeUp 1s 0.2s ease both; }
  .glass-panel { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 36px 32px; backdrop-filter: blur(24px); box-shadow: 0 0 0 1px rgba(0,255,136,0.06), 0 32px 80px rgba(0,0,0,0.6), inset 0 1px 0 rgba(0,255,136,0.1); position: relative; overflow: hidden; }
  .glass-panel::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--green), transparent); opacity: 0.5; }
  .panel-title { font-family: 'Share Tech Mono', monospace; font-size: 10px; letter-spacing: 5px; color: var(--muted); margin-bottom: 28px; text-align: center; }
  .tab-row { display: flex; gap: 0; margin-bottom: 24px; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .tab-btn { flex: 1; padding: 9px; background: transparent; border: none; font-family: 'Share Tech Mono', monospace; font-size: 9px; letter-spacing: 3px; color: var(--muted); cursor: pointer; transition: all 0.2s; }
  .tab-btn.active { background: var(--green-faint); color: var(--green); }
  .field { margin-bottom: 18px; }
  .field label { display: block; font-family: 'Share Tech Mono', monospace; font-size: 9px; letter-spacing: 4px; color: var(--muted); margin-bottom: 8px; }
  .field input { width: 100%; background: rgba(0,255,136,0.04); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; font-family: 'Rajdhani', sans-serif; font-size: 14px; color: var(--text); outline: none; transition: border-color 0.2s, box-shadow 0.2s; letter-spacing: 1px; }
  .field input::placeholder { color: rgba(180,220,200,0.25); }
  .field input:focus { border-color: var(--green-dim); box-shadow: 0 0 0 3px rgba(0,255,136,0.1), 0 0 16px rgba(0,255,136,0.08); }
  .btn-auth { width: 100%; padding: 14px; background: linear-gradient(135deg, #00cc66, #00ff88, #00cc66); background-size: 200% 200%; border: none; border-radius: 8px; font-family: 'Orbitron', monospace; font-size: 11px; font-weight: 700; letter-spacing: 4px; color: #020d1a; cursor: pointer; transition: all 0.3s; box-shadow: 0 0 24px rgba(0,255,136,0.35), 0 4px 16px rgba(0,0,0,0.4); animation: shimmer 3s linear infinite; margin-bottom: 22px; position: relative; overflow: hidden; }
  @keyframes shimmer { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
  .btn-auth:hover { transform: translateY(-1px); box-shadow: 0 0 40px rgba(0,255,136,0.55), 0 8px 24px rgba(0,0,0,0.5); }
  .btn-auth:active { transform: translateY(0); }
  .msg { font-family: 'Share Tech Mono', monospace; font-size: 10px; letter-spacing: 2px; padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; display: none; }
  .msg.error { background: rgba(255,60,100,0.12); border: 1px solid rgba(255,60,100,0.3); color: #ff6b8a; }
  .msg.success { background: rgba(0,255,136,0.08); border: 1px solid var(--border); color: var(--green); }
  footer { grid-column: 1 / -1; display: flex; align-items: center; justify-content: space-between; padding: 14px 0; border-top: 1px solid var(--border); }
  .footer-links { display: flex; gap: 24px; font-family: 'Share Tech Mono', monospace; font-size: 10px; letter-spacing: 2px; color: var(--muted); }
  .footer-links a { color: inherit; text-decoration: none; transition: color 0.2s; }
  .footer-links a:hover { color: var(--green); }
  .footer-right { font-family: 'Share Tech Mono', monospace; font-size: 10px; letter-spacing: 2px; color: var(--muted); }
  .particles { position: fixed; inset: 0; z-index: 1; pointer-events: none; }
  .particle { position: absolute; width: 2px; height: 2px; border-radius: 50%; background: var(--green); box-shadow: 0 0 4px var(--green); animation: floatUp linear infinite; opacity: 0; }
  @keyframes floatUp { 0% { opacity:0; transform: translateY(0) scale(0.5); } 10% { opacity:0.8; } 90% { opacity:0.4; } 100% { opacity:0; transform: translateY(-120px) scale(0.2); } }
  @media (max-width: 768px) { .page { grid-template-columns: 1fr; padding: 0 20px; } .hero { display: none; } }
</style>
</head>
<body>
<canvas id="bg"></canvas>
<div class="grid-overlay"></div>
<div class="scanline"></div>
<div class="particles" id="particles"></div>
<div class="page">
  <header>
    <div class="brand">
      <div class="brand-name">STOCKCAST</div>
      <div class="brand-sub">DEVELOPED BY MUAWWIZ GHANI</div>
    </div>
    <div class="header-right">TIME &nbsp;<span id="clock">00:00:00</span></div>
  </header>
  <main class="hero">
    <div class="hero-tag">LIVE ALPHA STREAM</div>
    <div class="live-badge"><div class="live-dot"></div>MARKETS ACTIVE</div>
    <h1>Predicting the<br>pulse of <span>global</span><br>markets.</h1>
    <p class="hero-desc">Stockcast is a futuristic terminal for global markets — it analyzes, monitors in real-time, and predicts the pulse of equities with machine learning precision.</p>
    <div class="stats">
      <div class="stat-pill"><div class="stat-val">98.4%</div><div class="stat-lbl">CONFIDENCE</div></div>
      <div class="stat-pill"><div class="stat-val">12ms</div><div class="stat-lbl">LATENCY</div></div>
      <div class="stat-pill"><div class="stat-val">1,240+</div><div class="stat-lbl">TICKERS</div></div>
    </div>
  </main>
  <div class="panel-col">
    <div class="glass-panel">
      <div class="panel-title">SECURE TERMINAL ACCESS</div>
      <div class="tab-row">
        <button class="tab-btn active" id="tab-login" onclick="switchTab('login')">LOGIN</button>
        <button class="tab-btn" id="tab-signup" onclick="switchTab('signup')">SIGN UP</button>
      </div>
      <div id="msg" class="msg"></div>
      <div class="field"><label>IDENTITY TOKEN (EMAIL)</label><input type="email" id="email" placeholder="name@firm.com" autocomplete="email"></div>
      <div class="field"><label>ACCESS KEY</label><input type="password" id="password" placeholder="••••••••••••" autocomplete="current-password"></div>
      <button class="btn-auth" id="submit-btn" onclick="handleSubmit()">AUTHORIZE ACCESS</button>
    </div>
  </div>
  <footer>
    <nav class="footer-links">
      <a href="#">PRIVACY</a><a href="#">TERMS</a><a href="#">CONTACT</a>
    </nav>
    <div class="footer-right">STOCKCAST © 2026</div>
  </footer>
</div>
<script>
// Clock
function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent =
    [now.getHours(), now.getMinutes(), now.getSeconds()]
      .map(n => String(n).padStart(2,'0')).join(':');
}
updateClock(); setInterval(updateClock, 1000);

// Particles
const container = document.getElementById('particles');
for (let i = 0; i < 30; i++) {
  const p = document.createElement('div'); p.className = 'particle';
  p.style.left = Math.random() * 50 + '%';
  p.style.bottom = Math.random() * 40 + 'vh';
  p.style.animationDuration = (6 + Math.random() * 10) + 's';
  p.style.animationDelay = (Math.random() * 12) + 's';
  p.style.width = p.style.height = (1 + Math.random() * 2) + 'px';
  container.appendChild(p);
}

// Canvas
const canvas = document.getElementById('bg');
const ctx = canvas.getContext('2d');
function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
resize(); window.addEventListener('resize', resize);
let time = 0;
function drawBg() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const cx = canvas.width * 0.42, cy = canvas.height * 0.72;
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 340);
  grad.addColorStop(0,'rgba(0,255,136,0.18)'); grad.addColorStop(0.4,'rgba(0,100,255,0.08)'); grad.addColorStop(1,'transparent');
  ctx.fillStyle = grad; ctx.fillRect(0,0,canvas.width,canvas.height);
  const bars=22, barW=18, spacing=28, startX=100;
  for (let i=0; i<bars; i++) {
    const x=startX+i*spacing, phase=i*0.4+time;
    const mid=cy-100-Math.sin(phase)*60-i*4.5;
    const open=mid+(Math.sin(phase*1.3)*20), close=mid+(Math.cos(phase*0.9)*25);
    const high=Math.min(open,close)-Math.abs(Math.sin(phase*2.1))*20-8;
    const low=Math.max(open,close)+Math.abs(Math.cos(phase*1.7))*15+6;
    const bull=close<open, alpha=0.5+0.5*(i/bars);
    ctx.strokeStyle=bull?`rgba(0,255,136,${alpha*0.7})`:`rgba(255,80,120,${alpha*0.7})`;
    ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(x+barW/2,high); ctx.lineTo(x+barW/2,low); ctx.stroke();
    ctx.fillStyle=bull?`rgba(0,255,136,${alpha*0.85})`:`rgba(255,60,100,${alpha*0.7})`;
    ctx.shadowColor=bull?'rgba(0,255,136,0.6)':'rgba(255,60,100,0.4)'; ctx.shadowBlur=8;
    ctx.fillRect(x,Math.min(open,close),barW,Math.abs(open-close)||2); ctx.shadowBlur=0;
  }
  time+=0.003; requestAnimationFrame(drawBg);
}
drawBg();

// Tab switching
let currentTab = 'login';
function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tab-login').classList.toggle('active', tab==='login');
  document.getElementById('tab-signup').classList.toggle('active', tab==='signup');
  document.getElementById('submit-btn').textContent = tab==='login' ? 'AUTHORIZE ACCESS' : 'CREATE ACCOUNT';
  showMsg('','');
}

function showMsg(text, type) {
  const el = document.getElementById('msg');
  el.textContent = text; el.className = 'msg ' + type;
  el.style.display = text ? 'block' : 'none';
}

function handleSubmit() {
  const email = document.getElementById('email').value.trim();
  const password = document.getElementById('password').value;
  if (!email || !password) { showMsg('⚠ ALL FIELDS REQUIRED', 'error'); return; }
  // Send credentials to Streamlit via query params trick
  const payload = JSON.stringify({ tab: currentTab, email, password });
  window.parent.postMessage({ type: 'streamlit:setComponentValue', value: payload }, '*');
}

// Allow Enter key
document.addEventListener('keydown', e => { if (e.key === 'Enter') handleSubmit(); });
</script>
</body>
</html>"""


def render_auth_gate(supabase_client):
    """
    Render the Stockcast login/signup page.
    Sets st.session_state.user on success and calls st.stop() if not authenticated.
    Call this near the top of app.py:  render_auth_gate(supabase)
    """
    # Initialise session state
    if "user" not in st.session_state:
        st.session_state.user = None
    if "_auth_msg" not in st.session_state:
        st.session_state._auth_msg = ("", "")   # (text, type)

    # Already authenticated – return immediately
    if st.session_state.user is not None:
        return

    # ── Streamlit native form (hidden behind the HTML shell) ──────────────────
    # We render the beautiful HTML component for visuals, then use a native
    # Streamlit form so the server actually receives the credentials.
    # The HTML's AUTHORIZE button posts a message; we also show native inputs
    # as a fallback (they're styled to be minimal / unobtrusive below the HTML).

    # Render the animated HTML page (takes most of the viewport height)
    components.html(_LOGIN_HTML, height=700, scrolling=False)

    # Error / success message from previous attempt
    msg_text, msg_type = st.session_state._auth_msg
    if msg_text:
        if msg_type == "error":
            st.error(msg_text)
        else:
            st.success(msg_text)

    # Compact native Streamlit form (acts as the real auth handler)
    with st.container():
        tab_login, tab_signup = st.tabs(["🔑  Login", "✨  Sign Up"])

        with tab_login:
            with st.form("_login_form", clear_on_submit=False):
                email_l = st.text_input("Email", placeholder="name@firm.com", key="_login_email")
                pass_l  = st.text_input("Password", type="password", placeholder="••••••••", key="_login_pass")
                submitted_l = st.form_submit_button("Authorize Access", use_container_width=True)

            if submitted_l:
                if not email_l or not pass_l:
                    st.session_state._auth_msg = ("All fields are required.", "error")
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
            with st.form("_signup_form", clear_on_submit=False):
                email_s = st.text_input("Email", placeholder="name@firm.com", key="_signup_email")
                pass_s  = st.text_input("Password", type="password", placeholder="Min 6 characters", key="_signup_pass")
                pass_s2 = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="_signup_pass2")
                submitted_s = st.form_submit_button("Create Account", use_container_width=True)

            if submitted_s:
                if not email_s or not pass_s or not pass_s2:
                    st.session_state._auth_msg = ("All fields are required.", "error")
                    st.rerun()
                elif pass_s != pass_s2:
                    st.session_state._auth_msg = ("Passwords do not match.", "error")
                    st.rerun()
                elif len(pass_s) < 6:
                    st.session_state._auth_msg = ("Password must be at least 6 characters.", "error")
                    st.rerun()
                else:
                    try:
                        res = supabase_client.auth.sign_up(
                            {"email": email_s, "password": pass_s}
                        )
                        if res.user:
                            st.session_state._auth_msg = (
                                "Account created! Check your email to confirm, then log in.", "success"
                            )
                        else:
                            st.session_state._auth_msg = ("Sign-up failed. Please try again.", "error")
                        st.rerun()
                    except Exception as exc:
                        st.session_state._auth_msg = (f"Sign-up failed: {exc}", "error")
                        st.rerun()

    # Block the rest of app.py from running
    st.stop()

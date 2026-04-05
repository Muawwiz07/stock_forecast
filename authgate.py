<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stockcast · Login</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --green: #00ff88;
    --green-dim: #00cc66;
    --green-glow: rgba(0,255,136,0.35);
    --green-faint: rgba(0,255,136,0.08);
    --navy: #020d1a;
    --panel: rgba(4,20,40,0.82);
    --border: rgba(0,255,136,0.18);
    --text: #c8e8d8;
    --muted: rgba(180,220,200,0.45);
  }

  html, body {
    height: 100%;
    background: var(--navy);
    font-family: 'Rajdhani', sans-serif;
    color: var(--text);
    overflow: hidden;
  }

  /* ── CANVAS BACKGROUND ── */
  #bg {
    position: fixed; inset: 0; z-index: 0;
  }

  /* ── GRID OVERLAY ── */
  .grid-overlay {
    position: fixed; inset: 0; z-index: 1; pointer-events: none;
    background-image:
      linear-gradient(rgba(0,255,136,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,255,136,0.04) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(ellipse 80% 60% at 50% 100%, black 30%, transparent 80%);
  }

  /* ── SCAN LINE ── */
  .scanline {
    position: fixed; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--green), transparent);
    opacity: 0.4; z-index: 2; animation: scan 6s linear infinite;
  }
  @keyframes scan { from { top: -2px; } to { top: 100%; } }

  /* ── LAYOUT ── */
  .page {
    position: relative; z-index: 10;
    height: 100vh;
    display: grid;
    grid-template-columns: 1fr 420px;
    grid-template-rows: auto 1fr auto;
    padding: 0 48px;
  }

  /* ── HEADER ── */
  header {
    grid-column: 1 / -1;
    display: flex; align-items: center; justify-content: space-between;
    padding: 22px 0 18px;
    border-bottom: 1px solid var(--border);
  }
  .brand { display: flex; flex-direction: column; gap: 2px; }
  .brand-name {
    font-family: 'Orbitron', monospace;
    font-size: 18px; font-weight: 700;
    color: var(--green);
    letter-spacing: 3px;
    text-shadow: 0 0 18px var(--green-glow);
  }
  .brand-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px; color: var(--muted); letter-spacing: 4px;
  }
  .header-right {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px; color: var(--muted); letter-spacing: 2px;
  }
  #clock { color: var(--green); }

  /* ── LEFT HERO ── */
  .hero {
    display: flex; flex-direction: column; justify-content: center;
    padding-right: 60px; padding-bottom: 40px;
    animation: fadeUp 1s ease both;
  }
  @keyframes fadeUp { from { opacity:0; transform:translateY(24px); } to { opacity:1; transform:none; } }

  .hero-tag {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px; letter-spacing: 5px;
    color: var(--green); margin-bottom: 20px;
    display: flex; align-items: center; gap: 10px;
  }
  .hero-tag::before {
    content: ''; display: block; width: 24px; height: 1px;
    background: var(--green); box-shadow: 0 0 6px var(--green);
  }

  .hero h1 {
    font-family: 'Orbitron', monospace;
    font-size: clamp(32px, 4vw, 52px);
    font-weight: 900; line-height: 1.12;
    color: #fff;
    text-shadow: 0 0 40px rgba(0,255,136,0.12);
    margin-bottom: 10px;
  }
  .hero h1 span { color: var(--green); }

  .hero-desc {
    font-size: 13px; color: var(--muted);
    line-height: 1.7; max-width: 360px;
    margin-top: 16px; margin-bottom: 40px;
    font-weight: 300; letter-spacing: 0.5px;
  }

  /* ── STAT PILLS ── */
  .stats { display: flex; gap: 20px; flex-wrap: wrap; }
  .stat-pill {
    background: var(--green-faint);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 18px;
    display: flex; flex-direction: column; gap: 2px;
    backdrop-filter: blur(8px);
  }
  .stat-val {
    font-family: 'Orbitron', monospace;
    font-size: 20px; font-weight: 700; color: var(--green);
    text-shadow: 0 0 10px var(--green-glow);
  }
  .stat-lbl {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px; letter-spacing: 3px; color: var(--muted);
  }

  /* ── LIVE BADGE ── */
  .live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    margin-bottom: 28px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px; letter-spacing: 3px; color: var(--green);
  }
  .live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { opacity:1; transform: scale(1); }
    50% { opacity:0.4; transform: scale(0.6); }
  }

  /* ── RIGHT PANEL ── */
  .panel-col {
    display: flex; flex-direction: column; justify-content: center;
    padding-bottom: 40px;
    animation: fadeUp 1s 0.2s ease both;
  }

  .glass-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 36px 32px;
    backdrop-filter: blur(24px);
    box-shadow:
      0 0 0 1px rgba(0,255,136,0.06),
      0 32px 80px rgba(0,0,0,0.6),
      inset 0 1px 0 rgba(0,255,136,0.1);
    position: relative; overflow: hidden;
  }
  .glass-panel::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--green), transparent);
    opacity: 0.5;
  }

  .panel-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px; letter-spacing: 5px;
    color: var(--muted); margin-bottom: 28px; text-align: center;
  }

  /* ── FORM ── */
  .field { margin-bottom: 18px; }
  .field label {
    display: block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px; letter-spacing: 4px;
    color: var(--muted); margin-bottom: 8px;
  }
  .field input {
    width: 100%;
    background: rgba(0,255,136,0.04);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 14px; color: var(--text);
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
    letter-spacing: 1px;
  }
  .field input::placeholder { color: rgba(180,220,200,0.25); }
  .field input:focus {
    border-color: var(--green-dim);
    box-shadow: 0 0 0 3px rgba(0,255,136,0.1), 0 0 16px rgba(0,255,136,0.08);
  }

  /* ── AUTHORIZE BUTTON ── */
  .btn-auth {
    width: 100%; padding: 14px;
    background: linear-gradient(135deg, #00cc66, #00ff88, #00cc66);
    background-size: 200% 200%;
    border: none; border-radius: 8px;
    font-family: 'Orbitron', monospace;
    font-size: 11px; font-weight: 700; letter-spacing: 4px;
    color: #020d1a; cursor: pointer;
    transition: all 0.3s;
    box-shadow: 0 0 24px rgba(0,255,136,0.35), 0 4px 16px rgba(0,0,0,0.4);
    animation: shimmer 3s linear infinite;
    margin-bottom: 22px;
    position: relative; overflow: hidden;
  }
  @keyframes shimmer {
    0%  { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100%{ background-position: 0% 50%; }
  }
  .btn-auth:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 40px rgba(0,255,136,0.55), 0 8px 24px rgba(0,0,0,0.5);
  }
  .btn-auth:active { transform: translateY(0); }

  /* ── DIVIDER ── */
  .divider {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 18px;
  }
  .divider-line { flex: 1; height: 1px; background: var(--border); }
  .divider-text {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px; letter-spacing: 3px; color: var(--muted);
  }

  /* ── OAUTH BUTTONS ── */
  .oauth-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 24px; }
  .btn-oauth {
    padding: 11px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 13px; font-weight: 500;
    color: var(--text); cursor: pointer;
    display: flex; align-items: center; justify-content: center; gap: 8px;
    transition: all 0.2s; letter-spacing: 1px;
  }
  .btn-oauth:hover {
    background: rgba(0,255,136,0.06);
    border-color: rgba(0,255,136,0.3);
  }
  .btn-oauth svg { width: 14px; height: 14px; flex-shrink: 0; }

  /* ── ALPHA ACCESS ── */
  .alpha-box {
    background: rgba(0,255,136,0.05);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
  }
  .alpha-title {
    font-family: 'Orbitron', monospace;
    font-size: 11px; font-weight: 700;
    color: var(--green); letter-spacing: 3px;
    text-shadow: 0 0 10px var(--green-glow);
    margin-bottom: 4px;
  }
  .alpha-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px; letter-spacing: 2px;
    color: var(--muted); margin-bottom: 12px;
  }
  .btn-alpha {
    width: 100%; padding: 10px;
    background: transparent;
    border: 1px solid var(--green-dim);
    border-radius: 6px;
    font-family: 'Orbitron', monospace;
    font-size: 9px; font-weight: 600; letter-spacing: 3px;
    color: var(--green); cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 0 12px rgba(0,255,136,0.1);
  }
  .btn-alpha:hover {
    background: rgba(0,255,136,0.1);
    box-shadow: 0 0 20px rgba(0,255,136,0.25);
  }

  /* ── FOOTER ── */
  footer {
    grid-column: 1 / -1;
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 0;
    border-top: 1px solid var(--border);
  }
  .footer-links {
    display: flex; gap: 24px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px; letter-spacing: 2px; color: var(--muted);
  }
  .footer-links a {
    color: inherit; text-decoration: none;
    transition: color 0.2s;
  }
  .footer-links a:hover { color: var(--green); }
  .footer-right {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px; letter-spacing: 2px; color: var(--muted);
  }

  /* ── FLOATING PARTICLES ── */
  .particles { position: fixed; inset: 0; z-index: 1; pointer-events: none; }
  .particle {
    position: absolute;
    width: 2px; height: 2px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 4px var(--green);
    animation: floatUp linear infinite;
    opacity: 0;
  }
  @keyframes floatUp {
    0%   { opacity:0; transform: translateY(0) scale(0.5); }
    10%  { opacity:0.8; }
    90%  { opacity:0.4; }
    100% { opacity:0; transform: translateY(-120px) scale(0.2); }
  }
</style>
</head>
<body>

<canvas id="bg"></canvas>
<div class="grid-overlay"></div>
<div class="scanline"></div>
<div class="particles" id="particles"></div>

<div class="page">

  <!-- HEADER -->
  <header>
    <div class="brand">
      <div class="brand-name">STOCKCAST</div>
      <div class="brand-sub">DEVELOPED BY MUAWWIZ GHANI</div>
    </div>
    <div class="header-right">
      TIME &nbsp;<span id="clock">00:00:00</span>
    </div>
  </header>

  <!-- HERO LEFT -->
  <main class="hero">
    <div class="hero-tag">LIVE ALPHA STREAM</div>

    <div class="live-badge">
      <div class="live-dot"></div>
      MARKETS ACTIVE
    </div>

    <h1>Predicting the<br>pulse of <span>global</span><br>markets.</h1>

    <p class="hero-desc">
      Stockcast is a futuristic terminal for global markets — it analyzes, monitors in real-time, and predicts the pulse of equities with machine learning precision.
    </p>

    <div class="stats">
      <div class="stat-pill">
        <div class="stat-val">98.4%</div>
        <div class="stat-lbl">CONFIDENCE</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val">12ms</div>
        <div class="stat-lbl">LATENCY</div>
      </div>
      <div class="stat-pill">
        <div class="stat-val">1,240+</div>
        <div class="stat-lbl">TICKERS</div>
      </div>
    </div>
  </main>

  <!-- RIGHT PANEL -->
  <div class="panel-col">
    <div class="glass-panel">
      <div class="panel-title">SECURE TERMINAL ACCESS</div>

      <div class="field">
        <label>IDENTITY TOKEN (EMAIL)</label>
        <input type="email" placeholder="name@firm.com" autocomplete="email">
      </div>

      <div class="field">
        <label>ACCESS KEY</label>
        <input type="password" placeholder="••••••••••••" autocomplete="current-password">
      </div>

      <button class="btn-auth">AUTHORIZE ACCESS</button>

      <div class="divider">
        <div class="divider-line"></div>
        <div class="divider-text">ALTERNATIVE PROTOCOLS</div>
        <div class="divider-line"></div>
      </div>

      <div class="oauth-row">
        <button class="btn-oauth">
          <svg viewBox="0 0 24 24" fill="none">
            <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
            <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
            <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/>
            <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
          </svg>
          Google
        </button>
        <button class="btn-oauth">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"/>
          </svg>
          SSO
        </button>
      </div>

      <div class="alpha-box">
        <div class="alpha-title">⬡ LIMITED OFFER</div>
        <div class="alpha-sub">EARLY ACCESS · GLASSMORPHIC TERMINAL</div>
        <button class="btn-alpha">REQUEST ALPHA ACCESS</button>
      </div>
    </div>
  </div>

  <!-- FOOTER -->
  <footer>
    <nav class="footer-links">
      <a href="#">FOOTER</a>
      <a href="#">LINK</a>
      <a href="#">PRIVACY</a>
      <a href="#">TERMS</a>
      <a href="#">CONTACT</a>
    </nav>
    <div class="footer-right">STOCKCAST © 2026</div>
  </footer>

</div>

<script>
// ── CLOCK ──
function updateClock() {
  const now = new Date();
  const h = String(now.getHours()).padStart(2,'0');
  const m = String(now.getMinutes()).padStart(2,'0');
  const s = String(now.getSeconds()).padStart(2,'0');
  document.getElementById('clock').textContent = `${h}:${m}:${s}`;
}
updateClock();
setInterval(updateClock, 1000);

// ── PARTICLES ──
const container = document.getElementById('particles');
for (let i = 0; i < 30; i++) {
  const p = document.createElement('div');
  p.className = 'particle';
  p.style.left = Math.random() * 50 + '%'; // left half only
  p.style.bottom = Math.random() * 40 + 'vh';
  p.style.animationDuration = (6 + Math.random() * 10) + 's';
  p.style.animationDelay = (Math.random() * 12) + 's';
  p.style.width = p.style.height = (1 + Math.random() * 2) + 'px';
  container.appendChild(p);
}

// ── CANVAS CHART ──
const canvas = document.getElementById('bg');
const ctx = canvas.getContext('2d');

function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
resize();
window.addEventListener('resize', resize);

// Draw glowing chart
let time = 0;
function drawBg() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // radial glow center
  const cx = canvas.width * 0.42;
  const cy = canvas.height * 0.72;

  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 340);
  grad.addColorStop(0, 'rgba(0,255,136,0.18)');
  grad.addColorStop(0.4, 'rgba(0,100,255,0.08)');
  grad.addColorStop(1, 'transparent');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // horizon glow
  const hg = ctx.createLinearGradient(0, cy - 60, 0, cy + 40);
  hg.addColorStop(0, 'rgba(0,255,136,0.0)');
  hg.addColorStop(0.5, 'rgba(0,200,255,0.12)');
  hg.addColorStop(1, 'rgba(0,255,136,0.0)');
  ctx.fillStyle = hg;
  ctx.fillRect(0, cy - 60, canvas.width * 0.7, 100);

  // animated candlesticks
  const bars = 22;
  const barW = 18;
  const spacing = 28;
  const startX = 100;

  for (let i = 0; i < bars; i++) {
    const x = startX + i * spacing;
    const phase = i * 0.4 + time;
    const mid = cy - 100 - Math.sin(phase) * 60 - i * 4.5;
    const open = mid + Math.random() * 0 + (Math.sin(phase * 1.3) * 20);
    const close = mid + (Math.cos(phase * 0.9) * 25);
    const high = Math.min(open, close) - Math.abs(Math.sin(phase * 2.1)) * 20 - 8;
    const low  = Math.max(open, close) + Math.abs(Math.cos(phase * 1.7)) * 15 + 6;

    const bull = close < open;
    const alpha = 0.5 + 0.5 * (i / bars);

    // wick
    ctx.strokeStyle = bull
      ? `rgba(0,255,136,${alpha * 0.7})`
      : `rgba(255,80,120,${alpha * 0.7})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x + barW/2, high);
    ctx.lineTo(x + barW/2, low);
    ctx.stroke();

    // body
    ctx.fillStyle = bull
      ? `rgba(0,255,136,${alpha * 0.85})`
      : `rgba(255,60,100,${alpha * 0.7})`;
    ctx.shadowColor = bull ? 'rgba(0,255,136,0.6)' : 'rgba(255,60,100,0.4)';
    ctx.shadowBlur = 8;
    ctx.fillRect(x, Math.min(open,close), barW, Math.abs(open - close) || 2);
    ctx.shadowBlur = 0;
  }

  time += 0.003;
  requestAnimationFrame(drawBg);
}
drawBg();
</script>
</body>
</html>

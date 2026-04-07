<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stockcast – Terminal Access</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;600;700&display=swap" rel="stylesheet">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }

:root {
  --green: #00ff88;
  --green-dim: #00cc6a;
  --green-glow: rgba(0,255,136,0.4);
  --green-subtle: rgba(0,255,136,0.08);
  --bg: #050d12;
  --bg2: #070f18;
  --panel-bg: rgba(6,20,30,0.82);
  --border: rgba(0,255,136,0.2);
  --text: #c8e8d8;
  --text-dim: #5a8a70;
  --mono: 'Share Tech Mono', monospace;
  --ui: 'Rajdhani', sans-serif;
  --display: 'Orbitron', sans-serif;
}

body {
  font-family: var(--ui);
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  overflow: hidden;
  position: relative;
}

/* ── BACKGROUND LAYERS ── */
.bg-layer {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
}

/* Grid floor */
.grid-floor {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 55%;
  background:
    linear-gradient(to bottom, transparent 0%, rgba(0,255,136,0.03) 100%),
    repeating-linear-gradient(90deg, rgba(0,255,136,0.07) 0px, transparent 1px, transparent 60px, rgba(0,255,136,0.07) 61px),
    repeating-linear-gradient(0deg, rgba(0,255,136,0.07) 0px, transparent 1px, transparent 40px, rgba(0,255,136,0.07) 41px);
  transform: perspective(600px) rotateX(55deg);
  transform-origin: top center;
}

/* Radial glow from center */
.center-glow {
  position: absolute;
  left: 50%; top: 38%;
  transform: translate(-50%, -50%);
  width: 700px; height: 400px;
  background: radial-gradient(ellipse, rgba(0,200,100,0.18) 0%, rgba(0,100,50,0.08) 40%, transparent 70%);
  filter: blur(20px);
}

/* Deep space gradient */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 60% at 50% 30%, rgba(0,30,20,0.9) 0%, #050d12 70%),
    linear-gradient(180deg, #020810 0%, #050d12 50%, #030a0f 100%);
  z-index: 0;
}

/* ── CANDLESTICK CHART (SVG Canvas) ── */
#chart-canvas {
  position: fixed;
  left: 50%; top: 50%;
  transform: translate(-50%, -50%);
  width: 100%; height: 100%;
  z-index: 1;
  opacity: 0.9;
}

/* ── LAYOUT ── */
.page {
  position: relative;
  z-index: 10;
  display: grid;
  grid-template-rows: auto 1fr auto;
  min-height: 100vh;
  padding: 0 3rem;
}

/* ── HEADER ── */
header {
  display: flex;
  align-items: flex-start;
  padding: 1.4rem 0 0;
}

.brand-name {
  font-family: var(--display);
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--green);
  letter-spacing: 0.05em;
  text-shadow: 0 0 20px var(--green-glow);
}

.brand-sub {
  font-family: var(--mono);
  font-size: 0.58rem;
  color: var(--text-dim);
  letter-spacing: 0.15em;
  margin-top: 0.15rem;
}

/* ── MAIN ── */
main {
  display: grid;
  grid-template-columns: 1fr 420px;
  align-items: center;
  gap: 3rem;
  padding: 1rem 0;
}

/* ── LEFT: HERO ── */
.hero h1 {
  font-family: var(--ui);
  font-size: 3.6rem;
  font-weight: 700;
  line-height: 1.1;
  color: #fff;
  margin-bottom: 1.2rem;
  text-shadow: 0 2px 30px rgba(0,0,0,0.8);
}

.hero h1 .accent {
  color: var(--green);
  text-shadow: 0 0 30px var(--green-glow), 0 0 60px rgba(0,255,136,0.2);
}

.hero-desc {
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--text-dim);
  line-height: 1.8;
  max-width: 380px;
  margin-bottom: 2.5rem;
}

/* Live Alpha Stream */
.stream-label {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--green-dim);
  letter-spacing: 0.2em;
  margin-bottom: 1.2rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.stream-label::before {
  content: '';
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 8px var(--green);
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.8); }
}

.widgets {
  display: flex;
  gap: 2rem;
  align-items: center;
}

/* Ring widget */
.ring-widget {
  width: 90px; height: 90px;
  position: relative;
}

.ring-widget svg {
  width: 100%; height: 100%;
  transform: rotate(-90deg);
}

.ring-track { fill: none; stroke: rgba(0,255,136,0.1); stroke-width: 4; }
.ring-fill {
  fill: none;
  stroke: var(--green);
  stroke-width: 3;
  stroke-linecap: round;
  filter: drop-shadow(0 0 6px var(--green));
  stroke-dasharray: 220;
  stroke-dashoffset: 220;
  animation: ringFill 2s 0.5s ease-out forwards;
}
@keyframes ringFill {
  to { stroke-dashoffset: 30; }
}

.ring-inner {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.ring-val {
  font-family: var(--display);
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--green);
  text-shadow: 0 0 10px var(--green-glow);
  line-height: 1;
}

.ring-lbl {
  font-family: var(--mono);
  font-size: 0.5rem;
  color: var(--text-dim);
  margin-top: 0.2rem;
}

/* Plasma orb */
.plasma-orb {
  width: 90px; height: 90px;
  border-radius: 50%;
  background: conic-gradient(
    from 0deg,
    transparent 0deg,
    rgba(0,255,136,0.6) 60deg,
    rgba(0,180,80,0.8) 120deg,
    transparent 180deg,
    rgba(0,255,136,0.3) 240deg,
    transparent 360deg
  );
  box-shadow:
    inset 0 0 20px rgba(0,255,136,0.2),
    0 0 25px rgba(0,255,136,0.3),
    0 0 50px rgba(0,255,136,0.1);
  animation: spinOrb 4s linear infinite;
  position: relative;
}

.plasma-orb::before {
  content: '';
  position: absolute;
  inset: 8px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(0,80,40,0.9) 0%, rgba(0,20,10,0.95) 70%);
}

@keyframes spinOrb {
  to { transform: rotate(360deg); }
}

/* ── RIGHT: LOGIN PANEL ── */
.login-panel {
  background: var(--panel-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1.8rem;
  backdrop-filter: blur(24px) saturate(1.4);
  box-shadow:
    0 0 0 1px rgba(0,255,136,0.05),
    0 8px 60px rgba(0,0,0,0.6),
    inset 0 1px 0 rgba(0,255,136,0.1),
    0 0 80px rgba(0,255,136,0.04);
  animation: slideIn 0.7s 0.2s cubic-bezier(0.16, 1, 0.3, 1) both;
}

@keyframes slideIn {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

.field-label {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--text-dim);
  letter-spacing: 0.18em;
  margin-bottom: 0.5rem;
}

.field-group {
  margin-bottom: 1.1rem;
}

input[type="email"],
input[type="password"] {
  width: 100%;
  background: rgba(0,20,12,0.6);
  border: 1px solid rgba(0,255,136,0.18);
  border-radius: 4px;
  padding: 0.7rem 1rem;
  font-family: var(--mono);
  font-size: 0.78rem;
  color: var(--text);
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
}

input[type="email"]::placeholder,
input[type="password"]::placeholder {
  color: rgba(90,138,112,0.5);
}

input[type="email"]:focus,
input[type="password"]:focus {
  border-color: rgba(0,255,136,0.5);
  box-shadow: 0 0 0 3px rgba(0,255,136,0.06), inset 0 0 10px rgba(0,255,136,0.04);
}

/* Neon button */
.btn-primary {
  width: 100%;
  padding: 0.75rem;
  background: linear-gradient(135deg, rgba(0,200,100,0.15), rgba(0,150,70,0.1));
  border: 1px solid var(--green-dim);
  border-radius: 4px;
  color: var(--green);
  font-family: var(--display);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.2em;
  cursor: pointer;
  transition: all 0.25s;
  position: relative;
  overflow: hidden;
  box-shadow: 0 0 20px rgba(0,255,136,0.15), inset 0 0 20px rgba(0,255,136,0.05);
  margin-bottom: 1.1rem;
}

.btn-primary::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent, rgba(0,255,136,0.15), transparent);
  transform: translateX(-100%);
  transition: transform 0.5s;
}

.btn-primary:hover::before { transform: translateX(100%); }
.btn-primary:hover {
  background: linear-gradient(135deg, rgba(0,220,110,0.2), rgba(0,180,80,0.15));
  box-shadow: 0 0 30px rgba(0,255,136,0.3), inset 0 0 20px rgba(0,255,136,0.08);
  transform: translateY(-1px);
}

.btn-primary:active { transform: translateY(0); }

/* Divider */
.alt-label {
  font-family: var(--mono);
  font-size: 0.58rem;
  color: var(--text-dim);
  letter-spacing: 0.18em;
  text-align: center;
  margin-bottom: 0.8rem;
}

.alt-buttons {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.6rem;
  margin-bottom: 1.2rem;
}

.btn-alt {
  padding: 0.6rem;
  background: rgba(0,20,12,0.5);
  border: 1px solid rgba(0,255,136,0.12);
  border-radius: 4px;
  color: var(--text);
  font-family: var(--ui);
  font-size: 0.75rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.4rem;
  letter-spacing: 0.05em;
}

.btn-alt:hover {
  border-color: rgba(0,255,136,0.3);
  background: rgba(0,255,136,0.05);
  color: var(--green);
}

.btn-alt svg { flex-shrink: 0; }

/* CTA panel */
.cta-panel {
  background: rgba(0,255,136,0.04);
  border: 1px solid rgba(0,255,136,0.15);
  border-radius: 4px;
  padding: 1rem;
}

.cta-title {
  font-family: var(--display);
  font-size: 0.7rem;
  font-weight: 700;
  color: var(--green);
  letter-spacing: 0.15em;
  text-align: center;
  margin-bottom: 0.3rem;
  text-shadow: 0 0 15px var(--green-glow);
}

.cta-desc {
  font-family: var(--mono);
  font-size: 0.58rem;
  color: var(--text-dim);
  text-align: center;
  margin-bottom: 0.8rem;
}

.btn-cta {
  width: 100%;
  padding: 0.65rem;
  background: linear-gradient(90deg, rgba(0,200,100,0.12), rgba(0,180,80,0.08));
  border: 1px solid rgba(0,255,136,0.25);
  border-radius: 4px;
  color: var(--green-dim);
  font-family: var(--display);
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.15em;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-cta:hover {
  border-color: rgba(0,255,136,0.5);
  color: var(--green);
  box-shadow: 0 0 20px rgba(0,255,136,0.15);
}

/* ── FOOTER ── */
footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 0 1.2rem;
  border-top: 1px solid rgba(0,255,136,0.06);
}

.footer-links {
  display: flex;
  gap: 1.5rem;
}

.footer-links a {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--text-dim);
  text-decoration: none;
  letter-spacing: 0.05em;
  transition: color 0.2s;
}

.footer-links a:hover { color: var(--green); }

.clock {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--text-dim);
  letter-spacing: 0.1em;
}

/* ── SCANLINES ── */
body::after {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.03) 2px,
    rgba(0,0,0,0.03) 4px
  );
  pointer-events: none;
  z-index: 999;
}

/* ── RESPONSIVE ── */
@media (max-width: 900px) {
  body { overflow-y: auto; }
  main { grid-template-columns: 1fr; gap: 2rem; }
  .hero h1 { font-size: 2.5rem; }
  .page { padding: 0 1.5rem; }
}
</style>
</head>
<body>

<!-- Background -->
<div class="bg-layer">
  <div class="grid-floor"></div>
  <div class="center-glow"></div>
</div>

<!-- Candlestick Chart SVG -->
<svg id="chart-canvas" viewBox="0 0 1400 800" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="glow-strong">
      <feGaussianBlur stdDeviation="6" result="coloredBlur"/>
      <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <linearGradient id="candleUp" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#00ff88"/>
      <stop offset="100%" stop-color="#00aa55"/>
    </linearGradient>
    <linearGradient id="candleDown" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#ff4466"/>
      <stop offset="100%" stop-color="#cc2244"/>
    </linearGradient>
    <radialGradient id="baseGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#00ff88" stop-opacity="0.5"/>
      <stop offset="100%" stop-color="#00ff88" stop-opacity="0"/>
    </radialGradient>
  </defs>

  <!-- Glow base under candles -->
  <ellipse cx="700" cy="700" rx="400" ry="120" fill="url(#baseGlow)" opacity="0.4"/>
  <ellipse cx="700" cy="700" rx="250" ry="80" fill="url(#baseGlow)" opacity="0.3"/>

  <!-- Candles: (x, wickTop, bodyTop, bodyH, isUp) -->
  <g filter="url(#glow)" opacity="0.85">
    <!-- Generated candle data -->
    <g id="candles"></g>
  </g>

  <!-- Horizontal light beams from tallest candles -->
  <line x1="0" y1="200" x2="1400" y2="200" stroke="#00ff88" stroke-width="0.5" opacity="0.06"/>
  <line x1="0" y1="300" x2="1400" y2="300" stroke="#00ff88" stroke-width="0.5" opacity="0.04"/>
  <line x1="0" y1="400" x2="1400" y2="400" stroke="#00ff88" stroke-width="0.5" opacity="0.03"/>

  <!-- Light cone from center bottom -->
  <path d="M700,800 L400,100 L1000,100 Z" fill="url(#baseGlow)" opacity="0.06"/>
  <path d="M700,800 L500,200 L900,200 Z" fill="url(#baseGlow)" opacity="0.08"/>
</svg>

<div class="page">
  <!-- Header -->
  <header>
    <div>
      <div class="brand-name">Stockcast</div>
      <div class="brand-sub">DEVELOPED BY NUAWIZ GHANI</div>
    </div>
  </header>

  <!-- Main -->
  <main>
    <!-- Left: Hero -->
    <div class="hero">
      <h1>Predicting the<br>pulse of <span class="accent">global<br>markets.</span></h1>
      <p class="hero-desc">
        Stockcast futuristic terminal global markets,<br>
        analyzes, and minates give now, predicting the<br>
        pulse of us in the alophe.
      </p>

      <div class="stream-label">LIVE ALPHA STREAM</div>
      <div class="widgets">
        <div class="plasma-orb"></div>
        <div class="ring-widget">
          <svg viewBox="0 0 80 80">
            <circle class="ring-track" cx="40" cy="40" r="34"/>
            <circle class="ring-fill" cx="40" cy="40" r="34"/>
          </svg>
          <div class="ring-inner">
            <div class="ring-val">98.4%</div>
            <div class="ring-lbl">confidence</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Right: Login Panel -->
    <div class="login-panel">
      <div class="field-group">
        <div class="field-label">IDENTITY TOKEN (EMAIL)</div>
        <input type="email" placeholder="name@firm.com" autocomplete="email"/>
      </div>

      <div class="field-group">
        <div class="field-label">ACCESS KEY</div>
        <input type="password" placeholder="••••••••••" autocomplete="current-password"/>
      </div>

      <button class="btn-primary">AUTHORIZE ACCESS</button>

      <div class="alt-label">ALTERNATIVE PROTOCOLS</div>
      <div class="alt-buttons">
        <button class="btn-alt">
          <svg width="14" height="14" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
          Google
        </button>
        <button class="btn-alt">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
          SSO
        </button>
      </div>

      <div class="cta-panel">
        <div class="cta-title">LIMITED OFFER</div>
        <div class="cta-desc">Glassmorphic withns offer</div>
        <button class="btn-cta">REQUEST ALPHA ACCESS</button>
      </div>
    </div>
  </main>

  <!-- Footer -->
  <footer>
    <nav class="footer-links">
      <a href="#">Footer</a>
      <a href="#">Link</a>
      <a href="#">Privacy</a>
      <a href="#">Terms</a>
      <a href="#">Contact</a>
    </nav>
    <div class="clock" id="clock">Time at 00:00:00</div>
  </footer>
</div>

<script>
// ── Clock ──
function updateClock() {
  const now = new Date();
  const hh = String(now.getHours()).padStart(2,'0');
  const mm = String(now.getMinutes()).padStart(2,'0');
  const ss = String(now.getSeconds()).padStart(2,'0');
  document.getElementById('clock').textContent = `Time at ${hh}:${mm}:${ss}`;
}
setInterval(updateClock, 1000);
updateClock();

// ── Generate Candlestick Chart ──
(function() {
  const svg = document.getElementById('candles');
  const W = 1400, H = 800;
  const numCandles = 28;
  const candleW = 22;
  const spacing = W / numCandles;
  const yBase = 720;
  const maxH = 520;

  // Price simulation
  let price = 400;
  const candles = [];
  for (let i = 0; i < numCandles; i++) {
    const open = price;
    const change = (Math.random() - 0.42) * 60;
    price += change;
    const close = price;
    const high = Math.max(open, close) + Math.random() * 30;
    const low = Math.min(open, close) - Math.random() * 20;
    candles.push({ open, close, high, low });
  }

  // Normalize to view
  const allVals = candles.flatMap(c => [c.high, c.low]);
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const range = maxVal - minVal;

  function toY(v) {
    return yBase - ((v - minVal) / range) * maxH;
  }

  candles.forEach((c, i) => {
    const x = spacing * i + spacing / 2;
    const isUp = c.close >= c.open;
    const color = isUp ? '#00ff88' : '#ff4466';
    const bodyTop = toY(Math.max(c.open, c.close));
    const bodyBot = toY(Math.min(c.open, c.close));
    const bodyH = Math.max(bodyBot - bodyTop, 2);
    const wickTop = toY(c.high);
    const wickBot = toY(c.low);
    const opacity = 0.5 + (i / numCandles) * 0.5;
    const glowSize = isUp ? (bodyH > 60 ? 8 : 4) : 2;

    const g = document.createElementNS('http://www.w3.org/2000/svg','g');

    // Wick
    const wick = document.createElementNS('http://www.w3.org/2000/svg','line');
    wick.setAttribute('x1', x); wick.setAttribute('x2', x);
    wick.setAttribute('y1', wickTop); wick.setAttribute('y2', wickBot);
    wick.setAttribute('stroke', color);
    wick.setAttribute('stroke-width', '1.5');
    wick.setAttribute('opacity', opacity);

    // Body
    const body = document.createElementNS('http://www.w3.org/2000/svg','rect');
    body.setAttribute('x', x - candleW/2);
    body.setAttribute('y', bodyTop);
    body.setAttribute('width', candleW);
    body.setAttribute('height', bodyH);
    body.setAttribute('fill', color);
    body.setAttribute('opacity', opacity * 0.85);
    body.setAttribute('rx', '1');

    // Glow for tall bullish candles
    if (isUp && bodyH > 40) {
      const glow = document.createElementNS('http://www.w3.org/2000/svg','rect');
      glow.setAttribute('x', x - candleW/2 - glowSize);
      glow.setAttribute('y', bodyTop - glowSize);
      glow.setAttribute('width', candleW + glowSize*2);
      glow.setAttribute('height', bodyH + glowSize*2);
      glow.setAttribute('fill', 'none');
      glow.setAttribute('stroke', '#00ff88');
      glow.setAttribute('stroke-width', '1');
      glow.setAttribute('opacity', '0.3');
      glow.setAttribute('rx', '2');
      glow.setAttribute('filter', 'url(#glow)');
      g.appendChild(glow);

      // vertical light beam
      const beam = document.createElementNS('http://www.w3.org/2000/svg','line');
      beam.setAttribute('x1', x); beam.setAttribute('x2', x);
      beam.setAttribute('y1', '0'); beam.setAttribute('y2', wickTop);
      beam.setAttribute('stroke', '#00ff88');
      beam.setAttribute('stroke-width', '1');
      beam.setAttribute('opacity', '0.12');
      g.appendChild(beam);
    }

    g.appendChild(wick);
    g.appendChild(body);
    svg.appendChild(g);
  });
})();
</script>
</body>
</html>

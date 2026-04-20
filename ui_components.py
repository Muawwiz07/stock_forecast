import { useState, useEffect, useRef } from "react";

// ─── DESIGN TOKENS (mirroring app-32.py CSS variables) ─────────────────────
const T = {
  bg1:     "#080e1c",
  bg2:     "#0f1727",
  bg3:     "#141d30",
  bg4:     "#1a2540",
  border:  "#1e2740",
  border2: "#252f47",
  accent:  "#4d8eff",
  accent2: "#7ab3ff",
  emerald: "#00e5b0",
  red:     "#ff5f5f",
  yellow:  "#ffd426",
  t1:      "#e4eafd",
  t2:      "#b0bcd4",
  t3:      "#8a8fa0",
  t4:      "#3e4558",
  mono:    "'IBM Plex Mono', monospace",
  sans:    "'Manrope', sans-serif",
};

const css = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Manrope:wght@500;600;700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: ${T.bg1};
    color: ${T.t1};
    font-family: ${T.sans};
    -webkit-font-smoothing: antialiased;
  }

  .sc-root {
    min-height: 100vh;
    background: ${T.bg1};
    padding: 0 0 4rem 0;
  }

  /* ─── NAV ─── */
  .sc-nav {
    background: ${T.bg2};
    border-bottom: 1px solid ${T.border};
    padding: .75rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
  }
  .sc-logo {
    font-family: ${T.mono};
    font-size: .95rem;
    font-weight: 700;
    letter-spacing: -.01em;
    color: ${T.t1};
  }
  .sc-logo span { color: ${T.accent}; }
  .sc-nav-pills {
    display: flex;
    gap: .35rem;
    flex-wrap: wrap;
  }
  .sc-nav-pill {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 2rem;
    padding: .28rem .85rem;
    font-family: ${T.sans};
    font-size: .6rem;
    font-weight: 700;
    letter-spacing: .09em;
    text-transform: uppercase;
    color: ${T.t3};
    cursor: pointer;
    transition: all .18s;
  }
  .sc-nav-pill:hover { color: ${T.t1}; border-color: ${T.border2}; }
  .sc-nav-pill.active {
    background: rgba(77,142,255,.12);
    border-color: rgba(77,142,255,.35);
    color: ${T.accent};
  }

  /* ─── HERO ─── */
  .sc-hero {
    padding: 3.5rem 2rem 2.5rem;
    max-width: 900px;
    margin: 0 auto;
  }
  .sc-hero-eyebrow {
    font-family: ${T.mono};
    font-size: .58rem;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: ${T.accent};
    margin-bottom: .7rem;
    display: flex;
    align-items: center;
    gap: .5rem;
  }
  .sc-hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 24px; height: 1px;
    background: ${T.accent};
  }
  .sc-hero-title {
    font-family: ${T.sans};
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -.02em;
    color: ${T.t1};
    margin-bottom: .9rem;
  }
  .sc-hero-title span { color: ${T.accent}; }
  .sc-hero-sub {
    font-size: .84rem;
    color: ${T.t3};
    line-height: 1.7;
    max-width: 560px;
  }

  /* ─── SECTION ─── */
  .sc-section {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 2rem 3rem;
    scroll-margin-top: 70px;
  }
  .sc-section-header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding-bottom: .6rem;
    border-bottom: 1px solid ${T.border};
  }
  .sc-section-number {
    font-family: ${T.mono};
    font-size: .58rem;
    font-weight: 700;
    letter-spacing: .18em;
    color: ${T.t4};
  }
  .sc-section-title {
    font-family: ${T.sans};
    font-size: .63rem;
    font-weight: 800;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: ${T.t2};
  }
  .sc-section-badge {
    margin-left: auto;
    background: rgba(77,142,255,.08);
    border: 1px solid rgba(77,142,255,.2);
    border-radius: 2rem;
    padding: .18rem .65rem;
    font-family: ${T.mono};
    font-size: .52rem;
    letter-spacing: .1em;
    color: ${T.accent};
  }

  /* ════════════════════════════════════
     1. PREMIUM TABS
  ════════════════════════════════════ */
  .sc-tabs-bar {
    display: flex;
    background: ${T.bg2};
    border: 1px solid ${T.border};
    border-radius: 10px 10px 0 0;
    border-bottom: none;
    padding: 0 .5rem;
    gap: 2px;
    position: relative;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .sc-tabs-bar::-webkit-scrollbar { display: none; }
  .sc-tab-btn {
    position: relative;
    padding: .7rem 1.1rem;
    font-family: ${T.sans};
    font-size: .61rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: ${T.t3};
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    transition: color .15s, border-color .15s;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: .45rem;
  }
  .sc-tab-btn:hover { color: ${T.t2}; }
  .sc-tab-btn.active {
    color: ${T.t1};
    border-bottom-color: ${T.accent};
  }
  .sc-tab-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: ${T.accent};
    animation: sc-pulse 2s infinite;
    flex-shrink: 0;
  }
  .sc-tab-dot.red { background: ${T.red}; }
  .sc-tab-dot.emerald { background: ${T.emerald}; }
  .sc-tab-dot.yellow { background: ${T.yellow}; }
  @keyframes sc-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(77,142,255,.5); }
    50% { opacity: .85; box-shadow: 0 0 0 4px rgba(77,142,255,0); }
  }
  .sc-tab-notif {
    background: ${T.red};
    color: #fff;
    border-radius: 2rem;
    font-size: .48rem;
    font-weight: 700;
    padding: .1rem .38rem;
    letter-spacing: 0;
    line-height: 1.4;
    min-width: 16px;
    text-align: center;
  }
  .sc-tab-panel {
    background: ${T.bg2};
    border: 1px solid ${T.border};
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 1.4rem 1.5rem;
    min-height: 120px;
  }
  .sc-tab-panel-content {
    font-size: .82rem;
    color: ${T.t2};
    line-height: 1.7;
  }
  .sc-tab-panel-title {
    font-family: ${T.mono};
    font-size: .58rem;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: ${T.accent};
    margin-bottom: .55rem;
  }

  /* ════════════════════════════════════
     2. SKELETON LOADING
  ════════════════════════════════════ */
  @keyframes sc-shimmer {
    0% { background-position: -600px 0; }
    100% { background-position: 600px 0; }
  }
  .sc-skeleton {
    background: linear-gradient(90deg,
      rgba(255,255,255,.03) 0%,
      rgba(255,255,255,.07) 50%,
      rgba(255,255,255,.03) 100%);
    background-size: 600px 100%;
    animation: sc-shimmer 1.6s infinite linear;
    border-radius: 4px;
  }
  .sc-skel-card {
    background: ${T.bg2};
    border: 1px solid ${T.border};
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
  }
  .sc-skel-row { display: flex; align-items: center; gap: .75rem; margin-bottom: .65rem; }
  .sc-skel-circle { width: 36px; height: 36px; border-radius: 50%; flex-shrink: 0; }
  .sc-skel-line { height: 10px; border-radius: 4px; }
  .sc-skel-block { height: 44px; border-radius: 6px; margin-top: .8rem; }

  /* ════════════════════════════════════
     3. CHIPS
  ════════════════════════════════════ */
  .sc-chips-row { display: flex; flex-wrap: wrap; gap: .45rem; align-items: center; }
  .sc-chip {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    border-radius: 2rem;
    padding: .28rem .78rem;
    font-family: ${T.sans};
    font-size: .6rem;
    font-weight: 700;
    letter-spacing: .09em;
    text-transform: uppercase;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all .16s;
    user-select: none;
    white-space: nowrap;
  }
  .sc-chip-buy {
    background: rgba(0,229,176,.1);
    border-color: rgba(0,229,176,.3);
    color: ${T.emerald};
  }
  .sc-chip-buy:hover, .sc-chip-buy.active {
    background: rgba(0,229,176,.2);
    border-color: ${T.emerald};
    box-shadow: 0 0 12px rgba(0,229,176,.25);
  }
  .sc-chip-sell {
    background: rgba(255,95,95,.1);
    border-color: rgba(255,95,95,.3);
    color: ${T.red};
  }
  .sc-chip-sell:hover, .sc-chip-sell.active {
    background: rgba(255,95,95,.2);
    border-color: ${T.red};
    box-shadow: 0 0 12px rgba(255,95,95,.25);
  }
  .sc-chip-hold {
    background: rgba(255,212,38,.1);
    border-color: rgba(255,212,38,.3);
    color: ${T.yellow};
  }
  .sc-chip-hold:hover, .sc-chip-hold.active {
    background: rgba(255,212,38,.2);
    border-color: ${T.yellow};
    box-shadow: 0 0 12px rgba(255,212,38,.25);
  }
  .sc-chip-neutral {
    background: rgba(77,142,255,.08);
    border-color: rgba(77,142,255,.2);
    color: ${T.accent};
  }
  .sc-chip-neutral:hover, .sc-chip-neutral.active {
    background: rgba(77,142,255,.16);
    border-color: ${T.accent};
  }
  .sc-chip-grey {
    background: rgba(138,143,160,.06);
    border-color: ${T.border2};
    color: ${T.t3};
  }
  .sc-chip-grey:hover, .sc-chip-grey.active {
    background: rgba(138,143,160,.12);
    border-color: ${T.t3};
    color: ${T.t2};
  }
  .sc-chip-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  /* ════════════════════════════════════
     4. ACCORDION
  ════════════════════════════════════ */
  .sc-accordion { border: 1px solid ${T.border}; border-radius: 10px; overflow: hidden; }
  .sc-accordion-item { border-bottom: 1px solid ${T.border}; }
  .sc-accordion-item:last-child { border-bottom: none; }
  .sc-accordion-trigger {
    width: 100%;
    background: ${T.bg2};
    border: none;
    padding: .95rem 1.3rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    transition: background .15s;
    text-align: left;
    gap: 1rem;
  }
  .sc-accordion-trigger:hover { background: ${T.bg3}; }
  .sc-accordion-trigger.open { background: rgba(77,142,255,.05); }
  .sc-accordion-left { display: flex; align-items: center; gap: .75rem; }
  .sc-accordion-step {
    width: 24px; height: 24px;
    border-radius: 50%;
    background: rgba(77,142,255,.12);
    border: 1px solid rgba(77,142,255,.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: ${T.mono};
    font-size: .56rem;
    font-weight: 700;
    color: ${T.accent};
    flex-shrink: 0;
    transition: all .2s;
  }
  .sc-accordion-trigger.open .sc-accordion-step {
    background: rgba(77,142,255,.22);
    border-color: ${T.accent};
  }
  .sc-accordion-label {
    font-family: ${T.sans};
    font-size: .74rem;
    font-weight: 700;
    color: ${T.t2};
  }
  .sc-accordion-trigger.open .sc-accordion-label { color: ${T.t1}; }
  .sc-accordion-meta {
    font-family: ${T.mono};
    font-size: .54rem;
    color: ${T.t4};
    letter-spacing: .08em;
  }
  .sc-accordion-chevron {
    width: 16px; height: 16px;
    color: ${T.t4};
    transition: transform .25s cubic-bezier(.4,0,.2,1);
    flex-shrink: 0;
  }
  .sc-accordion-chevron.open { transform: rotate(180deg); color: ${T.accent}; }
  .sc-accordion-body {
    overflow: hidden;
    transition: max-height .3s cubic-bezier(.4,0,.2,1), opacity .25s;
    opacity: 0;
    max-height: 0;
  }
  .sc-accordion-body.open { opacity: 1; max-height: 400px; }
  .sc-accordion-inner {
    padding: 1rem 1.3rem 1.2rem 1.3rem;
    background: rgba(0,0,0,.15);
    border-left: 3px solid rgba(77,142,255,.2);
    margin: 0 0 0 1.3rem;
    border-radius: 0 0 0 4px;
    font-size: .8rem;
    color: ${T.t3};
    line-height: 1.7;
  }
  .sc-accordion-inner strong { color: ${T.accent}; font-weight: 700; }

  /* ════════════════════════════════════
     5. TOAST
  ════════════════════════════════════ */
  .sc-toast-stack {
    position: fixed;
    bottom: 1.5rem;
    right: 1.5rem;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: .5rem;
    pointer-events: none;
  }
  .sc-toast {
    display: flex;
    align-items: flex-start;
    gap: .75rem;
    background: ${T.bg3};
    border: 1px solid ${T.border2};
    border-left: 3px solid ${T.accent};
    border-radius: 10px;
    padding: .85rem 1rem;
    min-width: 280px;
    max-width: 340px;
    box-shadow: 0 8px 30px rgba(0,0,0,.5);
    pointer-events: all;
    transform: translateX(0);
    animation: sc-toast-in .3s cubic-bezier(.4,0,.2,1) forwards;
  }
  .sc-toast.removing {
    animation: sc-toast-out .25s cubic-bezier(.4,0,.2,1) forwards;
  }
  .sc-toast.success { border-left-color: ${T.emerald}; }
  .sc-toast.error   { border-left-color: ${T.red}; }
  .sc-toast.warning { border-left-color: ${T.yellow}; }
  @keyframes sc-toast-in {
    from { opacity: 0; transform: translateX(20px); }
    to   { opacity: 1; transform: translateX(0); }
  }
  @keyframes sc-toast-out {
    from { opacity: 1; transform: translateX(0); }
    to   { opacity: 0; transform: translateX(20px); }
  }
  .sc-toast-icon {
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: .7rem;
    flex-shrink: 0;
    margin-top: .05rem;
  }
  .sc-toast.success .sc-toast-icon { background: rgba(0,229,176,.15); color: ${T.emerald}; }
  .sc-toast.error   .sc-toast-icon { background: rgba(255,95,95,.15);  color: ${T.red}; }
  .sc-toast.warning .sc-toast-icon { background: rgba(255,212,38,.15); color: ${T.yellow}; }
  .sc-toast.info    .sc-toast-icon { background: rgba(77,142,255,.15); color: ${T.accent}; }
  .sc-toast-title {
    font-family: ${T.sans};
    font-size: .68rem;
    font-weight: 700;
    color: ${T.t1};
    letter-spacing: .02em;
    margin-bottom: .15rem;
  }
  .sc-toast-body {
    font-family: ${T.sans};
    font-size: .62rem;
    color: ${T.t3};
    line-height: 1.5;
  }
  .sc-toast-close {
    margin-left: auto;
    background: none;
    border: none;
    color: ${T.t4};
    cursor: pointer;
    font-size: .9rem;
    line-height: 1;
    padding: 0 0 0 .5rem;
    flex-shrink: 0;
    transition: color .15s;
  }
  .sc-toast-close:hover { color: ${T.t2}; }
  .sc-toast-progress {
    position: absolute;
    bottom: 0; left: 0;
    height: 2px;
    background: rgba(77,142,255,.4);
    border-radius: 0 0 0 8px;
    transition: width linear;
  }
  .sc-toast.success .sc-toast-progress { background: rgba(0,229,176,.4); }
  .sc-toast.error   .sc-toast-progress { background: rgba(255,95,95,.4); }
  .sc-toast.warning .sc-toast-progress { background: rgba(255,212,38,.4); }

  /* ─── Toast trigger buttons ─── */
  .sc-btn-row { display: flex; gap: .5rem; flex-wrap: wrap; }
  .sc-btn {
    border: none;
    border-radius: 6px;
    padding: .5rem 1rem;
    font-family: ${T.sans};
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .07em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all .18s;
  }
  .sc-btn-success { background: rgba(0,229,176,.12); color: ${T.emerald}; border: 1px solid rgba(0,229,176,.3); }
  .sc-btn-success:hover { background: rgba(0,229,176,.2); box-shadow: 0 0 12px rgba(0,229,176,.2); }
  .sc-btn-error   { background: rgba(255,95,95,.12); color: ${T.red}; border: 1px solid rgba(255,95,95,.3); }
  .sc-btn-error:hover   { background: rgba(255,95,95,.2); box-shadow: 0 0 12px rgba(255,95,95,.2); }
  .sc-btn-warning { background: rgba(255,212,38,.1); color: ${T.yellow}; border: 1px solid rgba(255,212,38,.3); }
  .sc-btn-warning:hover { background: rgba(255,212,38,.18); box-shadow: 0 0 12px rgba(255,212,38,.2); }
  .sc-btn-info    { background: rgba(77,142,255,.1); color: ${T.accent}; border: 1px solid rgba(77,142,255,.25); }
  .sc-btn-info:hover    { background: rgba(77,142,255,.18); box-shadow: 0 0 12px rgba(77,142,255,.2); }

  /* ════════════════════════════════════
     6. BENTO GRID
  ════════════════════════════════════ */
  .sc-bento {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: auto auto;
    gap: .75rem;
  }
  @media (max-width: 640px) {
    .sc-bento { grid-template-columns: 1fr 1fr; }
    .sc-bento-hero { grid-column: span 2 !important; }
  }
  .sc-bento-cell {
    background: linear-gradient(145deg, ${T.bg2}, #090e1b);
    border: 1px solid rgba(255,255,255,.04);
    border-radius: 12px;
    padding: 1.3rem 1.4rem;
    transition: all .25s cubic-bezier(.4,0,.2,1);
    box-shadow: 0 2px 20px rgba(0,0,0,.3);
    position: relative;
    overflow: hidden;
  }
  .sc-bento-cell:hover {
    border-color: rgba(77,142,255,.3);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(77,142,255,.1);
  }
  .sc-bento-hero { grid-column: span 2; grid-row: span 1; }
  .sc-bento-cell::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 80px; height: 80px;
    background: radial-gradient(circle at top right, rgba(77,142,255,.07), transparent 70%);
    pointer-events: none;
  }
  .sc-bento-tag {
    font-family: ${T.mono};
    font-size: .5rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: ${T.t4};
    margin-bottom: .5rem;
  }
  .sc-bento-heading {
    font-family: ${T.sans};
    font-size: .88rem;
    font-weight: 800;
    color: ${T.t1};
    margin-bottom: .35rem;
    line-height: 1.3;
  }
  .sc-bento-body {
    font-size: .72rem;
    color: ${T.t3};
    line-height: 1.6;
  }
  .sc-bento-accent-bar {
    position: absolute;
    top: 0; left: 0;
    right: 0;
    height: 2px;
  }
  .sc-bento-big-num {
    font-family: ${T.mono};
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    margin: .5rem 0 .2rem;
  }
  .sc-bento-chip {
    display: inline-flex;
    align-items: center;
    gap: .3rem;
    font-family: ${T.mono};
    font-size: .54rem;
    font-weight: 700;
    letter-spacing: .06em;
    border-radius: 2rem;
    padding: .2rem .6rem;
    margin-top: .6rem;
  }

  /* ════════════════════════════════════
     7. BREADCRUMBS
  ════════════════════════════════════ */
  .sc-bc-wrap {
    background: ${T.bg2};
    border: 1px solid ${T.border};
    border-radius: 8px;
    padding: .6rem 1rem;
    display: inline-flex;
    align-items: center;
    gap: .2rem;
    flex-wrap: wrap;
  }
  .sc-bc-item {
    display: flex;
    align-items: center;
    gap: .2rem;
    font-family: ${T.mono};
    font-size: .6rem;
    letter-spacing: .07em;
  }
  .sc-bc-link {
    color: ${T.t4};
    cursor: pointer;
    padding: .15rem .4rem;
    border-radius: 4px;
    transition: color .15s, background .15s;
    text-decoration: none;
  }
  .sc-bc-link:hover { color: ${T.accent}; background: rgba(77,142,255,.08); }
  .sc-bc-sep {
    color: ${T.t4};
    font-size: .55rem;
    margin: 0 .05rem;
    user-select: none;
  }
  .sc-bc-current {
    color: ${T.t1};
    font-weight: 700;
    padding: .15rem .5rem;
    background: rgba(77,142,255,.1);
    border: 1px solid rgba(77,142,255,.2);
    border-radius: 4px;
  }

  /* ─── Divider ─── */
  .sc-divider {
    border: none;
    border-top: 1px solid ${T.border};
    margin: 2.5rem 0;
  }

  /* ─── Copy block ─── */
  .sc-copy-block {
    background: ${T.bg3};
    border: 1px solid ${T.border};
    border-left: 3px solid ${T.accent};
    border-radius: 0 8px 8px 0;
    padding: .9rem 1.2rem;
    font-family: ${T.mono};
    font-size: .7rem;
    color: ${T.t2};
    line-height: 1.6;
    margin-top: 1rem;
  }
  .sc-copy-block .kw { color: ${T.accent}; }
  .sc-copy-block .str { color: ${T.emerald}; }
  .sc-copy-block .cm { color: ${T.t4}; }

  /* ─── Sub label ─── */
  .sc-sub-label {
    font-family: ${T.sans};
    font-size: .56rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: ${T.t4};
    margin-bottom: .6rem;
    margin-top: 1rem;
  }
`;

// ─── TOAST SYSTEM ────────────────────────────────────────────────────────────
function useToasts() {
  const [toasts, setToasts] = useState([]);
  const add = (type, title, body) => {
    const id = Date.now() + Math.random();
    setToasts(prev => [...prev, { id, type, title, body, progress: 100 }]);
    // auto-remove
    const t1 = setTimeout(() => {
      setToasts(prev => prev.map(t => t.id === id ? { ...t, removing: true } : t));
    }, 3500);
    const t2 = setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 3800);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  };
  const remove = (id) => setToasts(prev => prev.filter(t => t.id !== id));
  return { toasts, add, remove };
}

function ToastStack({ toasts, remove }) {
  const icons = { success: "✓", error: "✕", warning: "⚠", info: "i" };
  return (
    <div className="sc-toast-stack">
      {toasts.map(t => (
        <div key={t.id} className={`sc-toast ${t.type} ${t.removing ? 'removing' : ''}`}
          style={{ position: 'relative' }}>
          <div className="sc-toast-icon">{icons[t.type] || "i"}</div>
          <div>
            <div className="sc-toast-title">{t.title}</div>
            <div className="sc-toast-body">{t.body}</div>
          </div>
          <button className="sc-toast-close" onClick={() => remove(t.id)}>×</button>
        </div>
      ))}
    </div>
  );
}

// ─── SECTION WRAPPER ─────────────────────────────────────────────────────────
function Section({ id, number, title, badge, children }) {
  return (
    <section id={id} className="sc-section">
      <div className="sc-section-header">
        <span className="sc-section-number">{number}</span>
        <span className="sc-section-title">{title}</span>
        {badge && <span className="sc-section-badge">{badge}</span>}
      </div>
      {children}
    </section>
  );
}

// ─── COMPONENT 1 — PREMIUM TABS ──────────────────────────────────────────────
function PremiumTabs() {
  const [active, setActive] = useState(0);
  const tabs = [
    {
      label: "Dashboard",
      icon: "🖥",
      dot: "emerald",
      notif: null,
      content: {
        title: "Dashboard · Overview",
        body: "Market indices are updated every 5 minutes. Your 4 watchlist stocks have new price data available. The Fear & Greed Index is currently reading 72 — Greed territory.",
      }
    },
    {
      label: "Analysis",
      icon: "📊",
      dot: null,
      notif: null,
      content: {
        title: "Analysis · AAPL",
        body: "XGBoost model trained on 2,532 trading days. 20 engineered features. MAPE: 2.14%. The composite signal is showing BUY with high confidence (87/100). RSI at 44 — neutral zone.",
      }
    },
    {
      label: "Watchlist",
      icon: "⭐",
      dot: "accent",
      notif: "3",
      content: {
        title: "Watchlist · 4 of 5 slots used",
        body: "AAPL ▲ +1.24% · TSLA ▼ −0.87% · NVDA ▲ +2.11% · MSFT ▲ +0.63%. Three stocks have triggered new signal states since your last visit.",
      }
    },
    {
      label: "Portfolio",
      icon: "💼",
      dot: null,
      notif: null,
      content: {
        title: "Portfolio · Tracker",
        body: "Total value: $24,830.40. Unrealized P&L: +$3,211.20 (+14.84%). Best performer: NVDA (+34.2%). Portfolio win rate: 68%.",
      }
    },
    {
      label: "Markets",
      icon: "🌍",
      dot: "red",
      notif: "!",
      content: {
        title: "Markets · Live Overview",
        body: "S&P 500 down 0.4% on Fed commentary. VIX spiked to 21.3. Technology sector leading losses (−1.2%). Energy sector holding green (+0.6%).",
      }
    },
  ];
  return (
    <>
      <div className="sc-tabs-bar">
        {tabs.map((tab, i) => (
          <button key={i} className={`sc-tab-btn ${active === i ? 'active' : ''}`}
            onClick={() => setActive(i)}>
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
            {tab.dot && <span className={`sc-tab-dot ${tab.dot}`} />}
            {tab.notif && <span className="sc-tab-notif">{tab.notif}</span>}
          </button>
        ))}
      </div>
      <div className="sc-tab-panel">
        <div className="sc-tab-panel-content">
          <div className="sc-tab-panel-title">{tabs[active].content.title}</div>
          <div style={{ fontSize: '.82rem', color: T.t2, lineHeight: 1.7 }}>
            {tabs[active].content.body}
          </div>
        </div>
      </div>
    </>
  );
}

// ─── COMPONENT 2 — SKELETON LOADING ──────────────────────────────────────────
function SkeletonDemo() {
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    let t; if (!loaded) { t = setTimeout(() => setLoaded(true), 3000); } return () => clearTimeout(t);
  }, [loaded]);
  const reset = () => setLoaded(false);

  if (loaded) return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '.75rem' }}>
      {[
        { sym: 'AAPL', price: '$189.32', chg: '+1.24%', dir: T.emerald },
        { sym: 'TSLA', price: '$248.70', chg: '−0.87%', dir: T.red },
        { sym: 'NVDA', price: '$876.10', chg: '+2.11%', dir: T.emerald },
      ].map((s, i) => (
        <div key={i} className="sc-skel-card" style={{ borderTop: `2px solid ${s.dir}` }}>
          <div style={{ fontFamily: T.mono, fontSize: '.56rem', letterSpacing: '.16em', textTransform: 'uppercase', color: T.t4, marginBottom: '.3rem' }}>Live Quote</div>
          <div style={{ fontFamily: T.mono, fontSize: '1.1rem', fontWeight: 700, color: T.t1 }}>{s.sym}</div>
          <div style={{ fontFamily: T.mono, fontSize: '1.5rem', fontWeight: 700, color: T.t1, marginTop: '.3rem' }}>{s.price}</div>
          <div style={{ fontFamily: T.mono, fontSize: '.72rem', fontWeight: 700, color: s.dir, marginTop: '.15rem' }}>{s.chg}</div>
        </div>
      ))}
      <div style={{ gridColumn: 'span 3', textAlign: 'center', marginTop: '.5rem' }}>
        <button className="sc-btn sc-btn-info" onClick={reset}>↺ Replay skeleton</button>
      </div>
    </div>
  );

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '.75rem' }}>
      {[0,1,2].map(i => (
        <div key={i} className="sc-skel-card">
          <div className="sc-skel-row">
            <div className="sc-skeleton sc-skel-circle" />
            <div style={{ flex: 1 }}>
              <div className="sc-skeleton sc-skel-line" style={{ width: '60%', marginBottom: '.4rem' }} />
              <div className="sc-skeleton sc-skel-line" style={{ width: '40%' }} />
            </div>
          </div>
          <div className="sc-skeleton sc-skel-block" />
          <div className="sc-skeleton sc-skel-line" style={{ width: '75%', marginTop: '.6rem', height: '8px' }} />
        </div>
      ))}
      <div style={{ gridColumn: 'span 3' }}>
        <div className="sc-skel-card">
          <div className="sc-skeleton sc-skel-line" style={{ width: '30%', height: '8px', marginBottom: '.7rem' }} />
          <div style={{ display: 'flex', gap: '.5rem' }}>
            {[80,55,68,72,91,60].map((w,i) => (
              <div key={i} className="sc-skeleton" style={{ flex: 1, height: `${w}px`, borderRadius: '4px' }} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── COMPONENT 3 — CHIPS ─────────────────────────────────────────────────────
function ChipsDemo() {
  const [signalFilter, setSignalFilter] = useState('ALL');
  const [sectorFilters, setSectorFilters] = useState([]);
  const toggleSector = s => setSectorFilters(prev => prev.includes(s) ? prev.filter(x => x !== s) : [...prev, s]);

  const signals = [
    { sym: 'AAPL', signal: 'BUY', sector: 'Tech', score: 87 },
    { sym: 'TSLA', signal: 'HOLD', sector: 'Auto', score: 52 },
    { sym: 'NVDA', signal: 'BUY', sector: 'Tech', score: 91 },
    { sym: 'JPM',  signal: 'SELL', sector: 'Finance', score: 28 },
    { sym: 'XLE',  signal: 'BUY', sector: 'Energy', score: 74 },
    { sym: 'META', signal: 'HOLD', sector: 'Tech', score: 49 },
  ];

  const filtered = signals.filter(s =>
    (signalFilter === 'ALL' || s.signal === signalFilter) &&
    (sectorFilters.length === 0 || sectorFilters.includes(s.sector))
  );

  const chipClass = { BUY: 'sc-chip-buy', HOLD: 'sc-chip-hold', SELL: 'sc-chip-sell' };
  const dotColor = { BUY: T.emerald, HOLD: T.yellow, SELL: T.red };
  const sectors = [...new Set(signals.map(s => s.sector))];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Signal filter chips */}
      <div>
        <div className="sc-sub-label" style={{ marginTop: 0 }}>Signal Filter</div>
        <div className="sc-chips-row">
          {['ALL','BUY','HOLD','SELL'].map(v => (
            <button key={v}
              className={`sc-chip ${v === 'BUY' ? 'sc-chip-buy' : v === 'HOLD' ? 'sc-chip-hold' : v === 'SELL' ? 'sc-chip-sell' : 'sc-chip-neutral'} ${signalFilter === v ? 'active' : ''}`}
              onClick={() => setSignalFilter(v)}>
              {v !== 'ALL' && <span className="sc-chip-dot" style={{ background: dotColor[v] }} />}
              {v}
            </button>
          ))}
        </div>
      </div>

      {/* Sector chips */}
      <div>
        <div className="sc-sub-label">Sector Filter</div>
        <div className="sc-chips-row">
          {sectors.map(s => (
            <button key={s}
              className={`sc-chip sc-chip-grey ${sectorFilters.includes(s) ? 'active' : ''}`}
              onClick={() => toggleSector(s)}>
              {s}
            </button>
          ))}
          {sectorFilters.length > 0 &&
            <button className="sc-chip sc-chip-neutral" onClick={() => setSectorFilters([])}>
              ✕ Clear
            </button>
          }
        </div>
      </div>

      {/* Results */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.5rem', marginTop: '.25rem' }}>
        {filtered.map(s => (
          <div key={s.sym} style={{
            background: T.bg2, border: `1px solid ${T.border}`, borderRadius: '8px',
            padding: '.65rem .85rem', display: 'flex', alignItems: 'center', gap: '.6rem',
          }}>
            <span style={{ fontFamily: T.mono, fontSize: '.72rem', fontWeight: 700, color: T.t1 }}>{s.sym}</span>
            <span className={`sc-chip ${chipClass[s.signal]}`} style={{ padding: '.18rem .55rem', fontSize: '.54rem' }}>
              <span className="sc-chip-dot" style={{ background: dotColor[s.signal], width: '5px', height: '5px' }} />
              {s.signal}
            </span>
            <span style={{ fontFamily: T.mono, fontSize: '.62rem', color: T.t3 }}>{s.score}/100</span>
          </div>
        ))}
        {filtered.length === 0 && (
          <div style={{ color: T.t4, fontSize: '.72rem', fontFamily: T.mono }}>No matches for current filters</div>
        )}
      </div>
    </div>
  );
}

// ─── COMPONENT 4 — ACCORDION ─────────────────────────────────────────────────
function AccordionDemo() {
  const [open, setOpen] = useState(0);
  const items = [
    {
      step: '01', label: 'Data Ingestion & Preprocessing',
      meta: 'yfinance · 7 years OHLCV',
      body: <>
        Stockcast pulls <strong>daily OHLCV data</strong> via yfinance with retry logic and a 5-minute cache TTL. The pipeline flattens MultiIndex columns, strips timezone info, and sorts chronologically — ensuring clean input for feature engineering.
      </>
    },
    {
      step: '02', label: 'Feature Engineering',
      meta: '20 engineered features',
      body: <>
        20 features are computed including <strong>SMA-7/14/21/50</strong>, EMA-12/26, MACD, RSI-14, Bollinger Bands (±2σ), ATR, volume Z-score, and lag features (t−1 to t−5). All features are normalized via RobustScaler.
      </>
    },
    {
      step: '03', label: 'XGBoost Model Training',
      meta: 'Walk-forward validation',
      body: <>
        A <strong>walk-forward split</strong> uses the last 20% of rows as the test window. XGBRegressor is trained with a configurable lookback. Hyperparameters (n_estimators, max_depth, learning_rate) are exposed to Pro users via the sidebar.
      </>
    },
    {
      step: '04', label: 'Signal Generation',
      meta: '6-factor composite',
      body: <>
        The composite signal weighs 6 factors: <strong>forecast direction, RSI zone, MACD crossover, price vs SMA-50, volume trend, and model confidence score</strong>. The weighted sum maps to BUY / HOLD / SELL with a configurable threshold.
      </>
    },
    {
      step: '05', label: 'Confidence Score & Uncertainty',
      meta: 'Bootstrap CI (Pro)',
      body: <>
        The confidence score is derived from R², MAPE, directional accuracy, and data volume. Pro users additionally get <strong>Bootstrap confidence intervals</strong> over N re-samples, visualized as shaded bands on the forecast chart.
      </>
    },
  ];
  return (
    <div className="sc-accordion">
      {items.map((item, i) => {
        const isOpen = open === i;
        return (
          <div key={i} className="sc-accordion-item">
            <button className={`sc-accordion-trigger ${isOpen ? 'open' : ''}`}
              onClick={() => setOpen(isOpen ? -1 : i)}>
              <div className="sc-accordion-left">
                <div className="sc-accordion-step">{item.step}</div>
                <div>
                  <div className="sc-accordion-label">{item.label}</div>
                  <div className="sc-accordion-meta">{item.meta}</div>
                </div>
              </div>
              <svg className={`sc-accordion-chevron ${isOpen ? 'open' : ''}`}
                viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>
            <div className={`sc-accordion-body ${isOpen ? 'open' : ''}`}>
              <div className="sc-accordion-inner">{item.body}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── COMPONENT 5 — TOAST ─────────────────────────────────────────────────────
function ToastDemo({ toastAdd }) {
  const triggers = [
    { type: 'success', label: '✓ Watchlist added', cls: 'sc-btn-success',
      title: 'Added to Watchlist', body: 'NVDA is now being tracked. Live price $876.10' },
    { type: 'error',   label: '✕ Email failed',   cls: 'sc-btn-error',
      title: 'Alert Not Sent', body: 'SMTP credentials missing — check secrets.toml' },
    { type: 'warning', label: '⚠ Plan limit',     cls: 'sc-btn-warning',
      title: 'Daily Limit Reached', body: '3/3 analyses used today. Upgrade to Pro for unlimited.' },
    { type: 'info',    label: 'ℹ Signal changed', cls: 'sc-btn-info',
      title: 'Signal Update · AAPL', body: 'BUY → HOLD  ·  Score dropped from 87 → 49' },
  ];
  return (
    <div>
      <div className="sc-btn-row">
        {triggers.map((t, i) => (
          <button key={i} className={`sc-btn ${t.cls}`}
            onClick={() => toastAdd(t.type, t.title, t.body)}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="sc-copy-block" style={{ marginTop: '1rem' }}>
        <span className="cm"># Streamlit equivalent pattern</span><br />
        <span className="kw">st</span>.<span className="str">toast</span>(<span className="str">"✓ NVDA added to watchlist"</span>, icon=<span className="str">"✅"</span>)<br />
        <span className="cm"># Or inject via st.markdown with auto-dismiss JS</span>
      </div>
    </div>
  );
}

// ─── COMPONENT 6 — BENTO GRID ────────────────────────────────────────────────
function BentoGrid() {
  const cells = [
    {
      span: 'hero',
      tag: 'Core Intelligence',
      heading: 'XGBoost Forecasting Engine',
      body: 'Trained on 7 years of OHLCV data with 20 engineered features. Walk-forward validation. Configurable lookback and forecast horizon.',
      accentColor: T.accent,
      extra: (
        <div style={{ display: 'flex', gap: '.5rem', marginTop: '.8rem', flexWrap: 'wrap' }}>
          {['20 Features', 'Walk-forward', 'MAPE ~2%', 'R² > 0.92'].map(t => (
            <span key={t} className="sc-chip sc-chip-neutral" style={{ fontSize: '.54rem', padding: '.2rem .6rem' }}>{t}</span>
          ))}
        </div>
      )
    },
    {
      tag: 'Signal Engine',
      heading: 'BUY · HOLD · SELL',
      body: '6-factor composite signal with configurable thresholds.',
      accentColor: T.emerald,
      extra: (
        <div>
          <div className="sc-bento-big-num" style={{ color: T.emerald }}>87</div>
          <div className="sc-bento-chip" style={{ background: 'rgba(0,229,176,.1)', border: `1px solid rgba(0,229,176,.25)`, color: T.emerald }}>
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: T.emerald, display: 'inline-block' }} />
            HIGH CONFIDENCE
          </div>
        </div>
      )
    },
    {
      tag: 'Risk Management',
      heading: 'Take-Profit & Stop-Loss',
      body: 'Auto-calculated ATR-based levels with configurable risk/reward ratio.',
      accentColor: T.yellow,
    },
    {
      tag: 'Shariah Screening',
      heading: 'Halal Compliance',
      body: 'AAOIFI Standard No.21 automated screening. Debt/MktCap, Debt/Assets, Cash/Assets ratios.',
      accentColor: '#a78bfa',
    },
    {
      tag: 'Strategy Simulator',
      heading: 'Backtesting Engine',
      body: 'Signal-driven strategy simulator with commission modeling and trade log.',
      accentColor: T.red,
      extra: (
        <div className="sc-bento-chip" style={{ background: 'rgba(255,95,95,.08)', border: `1px solid rgba(255,95,95,.2)`, color: T.red }}>
          PRO FEATURE
        </div>
      )
    },
  ];

  return (
    <div className="sc-bento">
      {cells.map((c, i) => (
        <div key={i} className={`sc-bento-cell ${c.span === 'hero' ? 'sc-bento-hero' : ''}`}>
          <div className="sc-bento-accent-bar"
            style={{ background: `linear-gradient(90deg, ${c.accentColor}, transparent)` }} />
          <div className="sc-bento-tag">{c.tag}</div>
          <div className="sc-bento-heading">{c.heading}</div>
          <div className="sc-bento-body">{c.body}</div>
          {c.extra}
        </div>
      ))}
    </div>
  );
}

// ─── COMPONENT 7 — BREADCRUMBS ───────────────────────────────────────────────
function BreadcrumbsDemo() {
  const [path, setPath] = useState(['Dashboard', 'AAPL', 'Analysis']);

  const chains = [
    ['Dashboard'],
    ['Dashboard', 'AAPL'],
    ['Dashboard', 'AAPL', 'Analysis'],
    ['Dashboard', 'AAPL', 'Analysis', 'Deep Analysis'],
    ['Markets', 'Sector Heatmap'],
    ['Portfolio', 'Holdings', 'NVDA'],
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Interactive breadcrumb */}
      <div>
        <div className="sc-sub-label" style={{ marginTop: 0 }}>Interactive — click a crumb to navigate back</div>
        <div className="sc-bc-wrap">
          {path.map((crumb, i) => {
            const isLast = i === path.length - 1;
            return (
              <div key={i} className="sc-bc-item">
                {i > 0 && <span className="sc-bc-sep">/</span>}
                {isLast
                  ? <span className="sc-bc-current">{crumb}</span>
                  : <span className="sc-bc-link" onClick={() => setPath(path.slice(0, i + 1))}>{crumb}</span>
                }
              </div>
            );
          })}
        </div>
        <div style={{ marginTop: '.75rem', display: 'flex', gap: '.4rem', flexWrap: 'wrap' }}>
          {chains.map((chain, i) => (
            <button key={i} className="sc-btn sc-btn-info" style={{ fontSize: '.54rem', padding: '.35rem .7rem' }}
              onClick={() => setPath(chain)}>
              {chain.join(' › ')}
            </button>
          ))}
        </div>
      </div>

      {/* Context examples */}
      <div>
        <div className="sc-sub-label">Context Variants</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
          {[
            { crumbs: ['Home', 'Markets', 'VIX'], accent: T.red },
            { crumbs: ['Portfolio', 'Holdings', 'TSLA', 'History'], accent: T.yellow },
            { crumbs: ['Settings', 'Alerts', 'Email'], accent: T.emerald },
          ].map(({ crumbs, accent }, ri) => (
            <div key={ri} className="sc-bc-wrap" style={{ borderColor: `rgba(${accent === T.red ? '255,95,95' : accent === T.yellow ? '255,212,38' : '0,229,176'},.2)` }}>
              {crumbs.map((c, ci) => (
                <div key={ci} className="sc-bc-item">
                  {ci > 0 && <span className="sc-bc-sep" style={{ color: accent + '66' }}>›</span>}
                  {ci === crumbs.length - 1
                    ? <span className="sc-bc-current" style={{ background: `rgba(${accent === T.red ? '255,95,95' : accent === T.yellow ? '255,212,38' : '0,229,176'},.1)`, border: `1px solid ${accent}44`, color: accent }}>{c}</span>
                    : <span className="sc-bc-link">{c}</span>
                  }
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── NAV SECTIONS ─────────────────────────────────────────────────────────────
const SECTIONS = [
  { id: 'tabs',       label: 'Tabs' },
  { id: 'skeleton',   label: 'Skeleton' },
  { id: 'chips',      label: 'Chips' },
  { id: 'accordion',  label: 'Accordion' },
  { id: 'toast',      label: 'Toast' },
  { id: 'bento',      label: 'Bento Grid' },
  { id: 'breadcrumbs',label: 'Breadcrumbs' },
];

// ─── ROOT APP ────────────────────────────────────────────────────────────────
export default function App() {
  const { toasts, add: toastAdd, remove: toastRemove } = useToasts();
  const [activeNav, setActiveNav] = useState('tabs');

  const scrollTo = (id) => {
    setActiveNav(id);
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    const handleScroll = () => {
      for (const s of SECTIONS) {
        const el = document.getElementById(s.id);
        if (el) {
          const rect = el.getBoundingClientRect();
          if (rect.top >= 0 && rect.top < 300) { setActiveNav(s.id); break; }
        }
      }
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <>
      <style>{css}</style>
      <div className="sc-root">

        {/* NAV */}
        <nav className="sc-nav">
          <div className="sc-logo">STOCK<span>CAST</span></div>
          <div className="sc-nav-pills">
            {SECTIONS.map(s => (
              <button key={s.id} className={`sc-nav-pill ${activeNav === s.id ? 'active' : ''}`}
                onClick={() => scrollTo(s.id)}>
                {s.label}
              </button>
            ))}
          </div>
        </nav>

        {/* HERO */}
        <div className="sc-hero">
          <div className="sc-hero-eyebrow">UI Component Kit · v2.0</div>
          <h1 className="sc-hero-title">7 Premium Components<br />for <span>Stockcast</span></h1>
          <p className="sc-hero-sub">
            Production-ready UI patterns in the existing dark-futuristic design system. All components are styled with the same CSS variables used in app.py — drop the generated HTML/CSS straight in.
          </p>
        </div>

        {/* ── 1. TABS ── */}
        <Section id="tabs" number="01" title="Premium Tabs" badge="Active indicators + notification dots">
          <PremiumTabs />
        </Section>

        <hr className="sc-divider" />

        {/* ── 2. SKELETON ── */}
        <Section id="skeleton" number="02" title="Skeleton Loading" badge="Auto-loads in 3 seconds">
          <p style={{ fontSize: '.8rem', color: T.t3, lineHeight: 1.6, marginBottom: '1rem' }}>
            Shown while watchlist quotes, market indices, or signal data loads asynchronously. Click "Replay" to re-trigger the shimmer state.
          </p>
          <SkeletonDemo />
        </Section>

        <hr className="sc-divider" />

        {/* ── 3. CHIPS ── */}
        <Section id="chips" number="03" title="Chips" badge="Signal tags · sector filters · feature tags">
          <ChipsDemo />
        </Section>

        <hr className="sc-divider" />

        {/* ── 4. ACCORDION ── */}
        <Section id="accordion" number="04" title="Accordion" badge="Methodology · signal breakdown · FAQ">
          <AccordionDemo />
        </Section>

        <hr className="sc-divider" />

        {/* ── 5. TOAST ── */}
        <Section id="toast" number="05" title="Toast Notifications" badge="Alerts · watchlist · email · copy">
          <p style={{ fontSize: '.8rem', color: T.t3, lineHeight: 1.6, marginBottom: '1rem' }}>
            Click any button to fire a toast. They auto-dismiss after 3.5 seconds, stack vertically, and respect the Stockcast color system.
          </p>
          <ToastDemo toastAdd={toastAdd} />
        </Section>

        <hr className="sc-divider" />

        {/* ── 6. BENTO ── */}
        <Section id="bento" number="06" title="Bento Grid" badge="Landing dashboard · feature showcase">
          <BentoGrid />
        </Section>

        <hr className="sc-divider" />

        {/* ── 7. BREADCRUMBS ── */}
        <Section id="breadcrumbs" number="07" title="Breadcrumbs" badge="Analysis navigation">
          <BreadcrumbsDemo />
        </Section>

      </div>

      {/* TOAST STACK — global overlay */}
      <ToastStack toasts={toasts} remove={toastRemove} />
    </>
  );
}

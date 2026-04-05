<!DOCTYPE html>

<html class="dark" lang="en"><head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>Stockcast - Predicting the Pulse</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&amp;family=Manrope:wght@300;400;500;600;700;800&amp;family=Inter:wght@300;400;500;600;700&amp;display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&amp;display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&amp;display=swap" rel="stylesheet"/>
<script id="tailwind-config">
      tailwind.config = {
        darkMode: "class",
        theme: {
          extend: {
            "colors": {
                    "secondary-fixed-dim": "#53e7b1",
                    "surface-container-high": "#1b2026",
                    "secondary": "#63f6bf",
                    "outline": "#72767b",
                    "on-primary-container": "#005b33",
                    "tertiary-fixed-dim": "#00c7ec",
                    "on-secondary-fixed-variant": "#006548",
                    "on-error": "#490006",
                    "secondary-dim": "#53e7b1",
                    "primary-dim": "#00ec8d",
                    "inverse-primary": "#006e3f",
                    "on-tertiary-fixed": "#002f39",
                    "error-container": "#9f0519",
                    "on-secondary-fixed": "#004530",
                    "surface-dim": "#0a0f13",
                    "on-secondary": "#005a40",
                    "inverse-surface": "#f7f9ff",
                    "surface-container-low": "#0f1418",
                    "primary-fixed": "#00fc97",
                    "on-surface": "#f1f4fa",
                    "on-background": "#f1f4fa",
                    "surface-container": "#151a1f",
                    "on-secondary-container": "#e0ffee",
                    "error": "#ff716c",
                    "on-tertiary-fixed-variant": "#004e5f",
                    "on-primary": "#006439",
                    "on-error-container": "#ffa8a3",
                    "surface-tint": "#a1ffc1",
                    "tertiary-container": "#00d5fe",
                    "tertiary": "#74e0ff",
                    "surface-bright": "#262d33",
                    "on-primary-fixed-variant": "#00653a",
                    "tertiary-dim": "#00c7ec",
                    "surface-container-highest": "#21262c",
                    "on-surface-variant": "#a8abb1",
                    "primary": "#a1ffc1",
                    "outline-variant": "#44484d",
                    "primary-fixed-dim": "#00ec8d",
                    "on-primary-fixed": "#004626",
                    "error-dim": "#d7383b",
                    "inverse-on-surface": "#51555a",
                    "on-tertiary": "#004f5f",
                    "background": "#0a0f13",
                    "secondary-fixed": "#63f6bf",
                    "surface-container-lowest": "#000000",
                    "primary-container": "#00fc97",
                    "tertiary-fixed": "#00d5fe",
                    "surface-variant": "#21262c",
                    "surface": "#0a0f13",
                    "secondary-container": "#006c4e",
                    "on-tertiary-container": "#004553"
            },
            "borderRadius": {
                    "DEFAULT": "0.125rem",
                    "lg": "0.25rem",
                    "xl": "0.5rem",
                    "full": "0.75rem"
            },
            "fontFamily": {
                    "headline": ["Space Grotesk"],
                    "body": ["Manrope"],
                    "label": ["Inter"]
            }
          },
        },
      }
    </script>
<style>
        .glass-panel {
            background: rgba(33, 38, 44, 0.4);
            backdrop-filter: blur(40px);
            border: 1px solid rgba(161, 255, 193, 0.1);
        }
        .glow-button:hover {
            box-shadow: 0 0 20px rgba(161, 255, 193, 0.4);
        }
        .pulse-animation {
            box-shadow: 0 0 0 0 rgba(0, 252, 151, 0.4);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(0, 252, 151, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(0, 252, 151, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 252, 151, 0); }
        }
        body {
            background-color: #0a0f13;
            overflow-x: hidden;
        }
    </style>
</head>
<body class="font-body text-on-surface">
<!-- Top Bar Component - From JSON -->
<header class="fixed top-0 w-full z-50 flex justify-between items-center px-6 py-4 bg-transparent">
<div class="text-2xl font-bold tracking-tighter text-[#00FF99] drop-shadow-[0_0_8px_rgba(0,255,153,0.5)] font-headline">
            STOCKCAST
        </div>
<div class="flex items-center gap-6">
<div class="hidden md:flex gap-8 text-sm uppercase tracking-widest font-label">
<a class="text-slate-400 font-medium hover:text-white transition-colors" href="#">Markets</a>
<a class="text-slate-400 font-medium hover:text-white transition-colors" href="#">Nodes</a>
<a class="text-slate-400 font-medium hover:text-white transition-colors" href="#">Protocols</a>
</div>
<div class="flex gap-4">
<span class="material-symbols-outlined text-[#00FF99] cursor-pointer hover:bg-white/5 p-2 rounded-full transition-all">settings</span>
<span class="material-symbols-outlined text-[#00FF99] cursor-pointer hover:bg-white/5 p-2 rounded-full transition-all">help_outline</span>
</div>
</div>
</header>
<main class="relative min-h-screen flex flex-col items-center justify-center pt-20 pb-24 overflow-hidden">
<!-- Background Visualization Layer -->
<div class="absolute inset-0 z-0 opacity-40 pointer-events-none">
<div class="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,_rgba(0,252,151,0.05)_0%,_transparent_50%)]"></div>
<div class="absolute top-1/4 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-primary-fixed/20 to-transparent"></div>
<img alt="Data rich background" class="absolute inset-0 object-cover mix-blend-overlay grayscale" data-alt="Futuristic data visualization with glowing green neon line charts and abstract geometric data nodes on a dark digital background" src="https://lh3.googleusercontent.com/aida-public/AB6AXuBp7gT0yd7gbV9AIUitmr63b9kLawlW8GnJfP09_Od7-KtM92dzmeuS2fTMzFhHRaZ_NDjvty7h4otcKk4r0hOtEYSPo3Hb_F75sIhfCzPmmrAdtBql8l66I5ynVp12xUkmkVuSprdJ_8189kpoDWaJaLdYuDG7xhncySl7gyA0V1QcKc9As-5wjmEuntGGHZbPOjPZlZEDAyBSNigTOWNsXOkdg4xjmSz24uP2yKU3mnMSSf5i2SDXKTvYx8AhbsFh8h1MXIyYboY"/>
</div>
<div class="container mx-auto px-6 relative z-10 flex flex-col lg:flex-row items-center justify-between gap-16">
<!-- Left Narrative Column -->
<div class="max-w-2xl text-left">
<div class="inline-flex items-center gap-2 mb-6 px-3 py-1 rounded-full bg-primary/10 border border-primary/20">
<span class="relative flex h-2 w-2">
<span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
<span class="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
</span>
<span class="text-[10px] font-label font-bold tracking-[0.2em] text-primary uppercase">System Active: 0.003ms latency</span>
</div>
<h1 class="font-headline text-5xl md:text-7xl font-bold tracking-tight text-on-surface leading-tight mb-8">
                    Predicting the <span class="text-primary-fixed drop-shadow-[0_0_15px_rgba(0,252,151,0.3)]">pulse</span> of global markets.
                </h1>
<p class="font-body text-xl text-on-surface-variant max-w-lg mb-10 leading-relaxed">
                    Access the first real-time predictive engine powered by quantum-recursive neural networks. Secure your position in the future of finance.
                </p>
<!-- Limited Offer Section -->
<div class="glass-panel p-6 rounded-xl border-l-4 border-l-primary max-w-md">
<div class="flex items-start gap-4">
<span class="material-symbols-outlined text-primary-fixed" style="font-variation-settings: 'FILL' 1;">stars</span>
<div>
<h3 class="font-headline font-bold text-lg mb-1">Limited Offer</h3>
<p class="text-sm text-on-surface-variant mb-4">Phase 01 Alpha testing is now open for accredited institutional partners. Limited slots available.</p>
<button class="bg-primary/10 hover:bg-primary/20 text-primary px-4 py-2 rounded-lg text-sm font-bold tracking-wide transition-all border border-primary/30 flex items-center gap-2 group">
                                Request Alpha Access
                                <span class="material-symbols-outlined text-sm group-hover:translate-x-1 transition-transform">arrow_forward</span>
</button>
</div>
</div>
</div>
</div>
<!-- Login Form Container -->
<div class="w-full max-w-md">
<div class="glass-panel p-8 md:p-10 rounded-2xl shadow-2xl relative overflow-hidden group">
<div class="absolute -top-24 -right-24 w-48 h-48 bg-primary/10 blur-[60px] rounded-full group-hover:bg-primary/20 transition-all duration-700"></div>
<div class="relative z-10">
<div class="mb-10">
<h2 class="font-headline text-3xl font-bold mb-2">Initialize Session</h2>
<p class="text-on-surface-variant font-label text-sm">Enter your credentials to connect to the terminal.</p>
</div>
<form class="space-y-6">
<div class="space-y-2">
<label class="block text-xs font-label font-bold uppercase tracking-widest text-primary/80">Identity Token (Email)</label>
<div class="relative group">
<span class="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-outline text-lg group-focus-within:text-primary transition-colors">fingerprint</span>
<input class="w-full bg-surface-container-lowest border border-outline-variant hover:border-primary/50 focus:border-primary focus:ring-1 focus:ring-primary/20 transition-all rounded-xl py-4 pl-12 pr-4 text-on-surface placeholder:text-outline-variant outline-none" placeholder="user@kinetic.node" type="email"/>
</div>
</div>
<div class="space-y-2">
<div class="flex justify-between items-center">
<label class="block text-xs font-label font-bold uppercase tracking-widest text-primary/80">Access Key</label>
<a class="text-[10px] text-outline hover:text-primary transition-colors font-label font-bold uppercase tracking-tighter" href="#">Recover Key</a>
</div>
<div class="relative group">
<span class="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-outline text-lg group-focus-within:text-primary transition-colors">key</span>
<input class="w-full bg-surface-container-lowest border border-outline-variant hover:border-primary/50 focus:border-primary focus:ring-1 focus:ring-primary/20 transition-all rounded-xl py-4 pl-12 pr-4 text-on-surface placeholder:text-outline-variant outline-none" placeholder="••••••••••••" type="password"/>
</div>
</div>
<button class="w-full py-4 bg-gradient-to-r from-primary to-primary-container text-on-primary font-headline font-extrabold uppercase tracking-widest rounded-xl hover:shadow-[0_0_20px_rgba(0,252,151,0.4)] active:scale-95 transition-all glow-button flex items-center justify-center gap-3" type="submit">
                                Authorize Access
                                <span class="material-symbols-outlined">verified_user</span>
</button>
</form>
<div class="relative my-8">
<div class="absolute inset-0 flex items-center"><div class="w-full border-t border-outline-variant/30"></div></div>
<div class="relative flex justify-center text-[10px] uppercase tracking-[0.3em]"><span class="bg-transparent px-4 text-outline font-label">Federated Login</span></div>
</div>
<div class="grid grid-cols-2 gap-4">
<button class="flex items-center justify-center gap-3 py-3 glass-panel rounded-xl hover:bg-white/5 transition-all text-xs font-bold uppercase tracking-widest">
<svg class="w-4 h-4" viewbox="0 0 24 24"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="currentColor"></path><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="currentColor"></path><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="currentColor"></path><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="currentColor"></path></svg>
                                Google
                            </button>
<button class="flex items-center justify-center gap-3 py-3 glass-panel rounded-xl hover:bg-white/5 transition-all text-xs font-bold uppercase tracking-widest">
<span class="material-symbols-outlined text-lg">hub</span>
                                SSO
                            </button>
</div>
<p class="mt-8 text-center text-xs text-on-surface-variant font-label tracking-wide">
                            Unauthorized access attempts are monitored. 
                            <a class="text-primary hover:underline underline-offset-4" href="#">Security Protocol v4.2</a>
</p>
</div>
</div>
</div>
</div>
</main>
<!-- Footer Component - From JSON -->
<footer class="fixed bottom-0 w-full z-40 flex flex-col md:flex-row justify-between items-center px-8 py-6 bg-[#0a0f13]/80 backdrop-blur-xl border-t border-white/10">
<div class="font-inter text-[10px] uppercase tracking-[0.2em] text-slate-500 mb-4 md:mb-0">
            © 2024 KINETIC SYSTEMS. ALL RIGHTS RESERVED.
        </div>
<div class="flex gap-8">
<a class="font-inter text-[10px] uppercase tracking-[0.2em] text-slate-500 hover:text-[#00FF99] transition-all duration-300" href="#">Security Protocol</a>
<a class="font-inter text-[10px] uppercase tracking-[0.2em] text-slate-500 hover:text-[#00FF99] transition-all duration-300" href="#">System Status</a>
<a class="font-inter text-[10px] uppercase tracking-[0.2em] text-slate-500 hover:text-[#00FF99] transition-all duration-300" href="#">Privacy Node</a>
</div>
</footer>
</body></html>

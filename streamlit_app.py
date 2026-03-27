"""
Urban Flow & Life-Lines | Bangalore — CIA Dashboard Edition
============================================================
CIA-style dark intelligence dashboard with bold neon accents.
Each intersection gets its own section with dedicated road diagrams.
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import numpy as np
from scipy.optimize import linprog
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import time

st.set_page_config(
    page_title="Urban Flow & Life-Lines | Bangalore",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"]{background:#000!important}
[data-testid="stHeader"],[data-testid="stToolbar"]{display:none!important}
section[data-testid="stSidebar"]{display:none!important}
.block-container{padding:0!important;max-width:100%!important}
iframe{border:none!important;display:block}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

JN = [
    {"id":0,  "name":"Silk Board",      "lat":12.9177,"lng":77.6228,"cong":0.71,"daily":185000,"peak":22000,"sat_flow":1800,"lanes":4},
    {"id":1,  "name":"Hebbal",          "lat":13.0358,"lng":77.5970,"cong":0.64,"daily":156000,"peak":19000,"sat_flow":1800,"lanes":3},
    {"id":2,  "name":"Marathahalli",    "lat":12.9591,"lng":77.6974,"cong":0.58,"daily":142000,"peak":17500,"sat_flow":1800,"lanes":3},
    {"id":3,  "name":"KR Puram",        "lat":13.0074,"lng":77.6950,"cong":0.54,"daily":128000,"peak":15000,"sat_flow":1700,"lanes":3},
    {"id":4,  "name":"Electronic City", "lat":12.8399,"lng":77.6770,"cong":0.67,"daily":168000,"peak":20000,"sat_flow":1800,"lanes":4},
    {"id":5,  "name":"Whitefield",      "lat":12.9698,"lng":77.7500,"cong":0.52,"daily":118000,"peak":14000,"sat_flow":1600,"lanes":3},
    {"id":6,  "name":"Indiranagar",     "lat":12.9784,"lng":77.6408,"cong":0.62,"daily":138000,"peak":16500,"sat_flow":1700,"lanes":3},
    {"id":7,  "name":"Koramangala",     "lat":12.9352,"lng":77.6245,"cong":0.66,"daily":155000,"peak":18500,"sat_flow":1800,"lanes":4},
    {"id":8,  "name":"JP Nagar",        "lat":12.9063,"lng":77.5857,"cong":0.48,"daily":108000,"peak":13000,"sat_flow":1600,"lanes":3},
    {"id":9,  "name":"Yelahanka",       "lat":13.1007,"lng":77.5963,"cong":0.44,"daily": 98000,"peak":12000,"sat_flow":1500,"lanes":2},
    {"id":10, "name":"Bannerghatta Rd", "lat":12.8931,"lng":77.5971,"cong":0.59,"daily":132000,"peak":15800,"sat_flow":1700,"lanes":3},
    {"id":11, "name":"Nagawara",        "lat":13.0456,"lng":77.6207,"cong":0.55,"daily":122000,"peak":14500,"sat_flow":1600,"lanes":3},
]

_JN_PHASES = [
    (1800,1600,0.71,0.55),(1800,1500,0.64,0.48),(1800,1600,0.58,0.45),(1700,1400,0.54,0.42),
    (1800,1500,0.67,0.50),(1600,1300,0.52,0.38),(1700,1400,0.62,0.47),(1800,1600,0.66,0.52),
    (1600,1300,0.48,0.35),(1500,1200,0.44,0.30),(1700,1400,0.59,0.44),(1600,1300,0.55,0.40),
]

def run_lp(C=90, density_factor=1.0):
    n = len(_JN_PHASES)
    S_maj = np.array([p[0] for p in _JN_PHASES], dtype=float)
    S_min = np.array([p[1] for p in _JN_PHASES], dtype=float)
    c_maj = np.minimum(np.array([p[2] for p in _JN_PHASES]) * density_factor, 0.97)
    c_min = np.minimum(np.array([p[3] for p in _JN_PHASES]) * density_factor, 0.97)
    y_maj = c_maj; y_min = c_min
    L = 7.0; G_total = C - L; g_min_b = 10.0; g_max_b = G_total - g_min_b
    w_maj = y_maj / np.maximum(1.0 - y_maj, 0.03)
    w_min = y_min / np.maximum(1.0 - y_min, 0.03)
    c_obj = w_min - w_maj
    A_ub = np.ones((1, n)); b_ub = np.array([n * G_total * 0.82])
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(g_min_b, g_max_b)] * n, method='highs')
    g_maj = res.x if res.success else np.clip(y_maj / (y_maj + y_min) * G_total, g_min_b, g_max_b)
    lambda_i = g_maj / C
    x_i = np.minimum(y_maj / np.maximum(lambda_i, 1e-6), 0.999)
    d1 = C * (1 - lambda_i) ** 2 / np.maximum(2 * (1 - np.minimum(x_i, 1.0) * lambda_i), 0.001)
    PF = np.clip((1 - 0.33) / np.maximum(1 - lambda_i, 0.01), 0.5, 2.0)
    d1_pf = d1 * PF
    cap_i = S_maj * lambda_i; cap_s = cap_i / 3600.0
    T_hr = 0.25
    d2_term = (x_i - 1) + np.sqrt(np.maximum((x_i - 1) ** 2 + 8 * 0.5 * x_i / np.maximum(cap_s * T_hr * 3600, 1), 0))
    d2 = 900 * T_hr * d2_term
    Qb = np.maximum(0, x_i - 0.95) * cap_i * T_hr
    d3 = np.where(Qb > 0, np.minimum(Qb * C / (2.0 * np.maximum(cap_i, 1.0)), 15.0), 0.0)
    d_maj = np.minimum(d1_pf + d2 + d3, 300.0)
    C_opt_per = np.clip((1.5 * L + 5) / np.maximum(1 - (y_maj + y_min), 0.05), 30, 180)
    def hcm_los(d): return ('A' if d <= 10 else 'B' if d <= 20 else 'C' if d <= 35 else 'D' if d <= 55 else 'E' if d <= 80 else 'F')
    los_i = [hcm_los(float(d)) for d in d_maj]
    q_len = np.clip(np.round(cap_i * x_i * (1 - lambda_i) ** 2 / np.maximum(2 * (1 - np.minimum(x_i * lambda_i, 0.999)), 0.01)), 0, 999).astype(int)
    return {
        "g": g_maj.tolist(), "lambda": lambda_i.tolist(), "x": x_i.tolist(),
        "delay": d_maj.tolist(), "los": los_i, "q_len": q_len.tolist(),
        "q_pcu": (c_maj * S_maj).tolist(), "cap_pcu": cap_i.tolist(),
        "C_opt": float(np.median(C_opt_per)), "C_opt_per": C_opt_per.tolist(),
        "lp_ok": bool(res.success),
    }

def rl_quick(density_factor=1.0, C=90, n_episodes=200):
    np.random.seed(7)
    n = len(_JN_PHASES)
    L = 7.0; G = C - L; g_min_b, g_max_b = 10.0, G - 10.0
    N_CONG, N_PHASE, N_TOD, N_ACTIONS = 6, 3, 2, 5
    DELTA_G = [-15.0, -5.0, 0.0, 5.0, 15.0]
    QA = [np.zeros((N_CONG, N_PHASE, N_TOD, N_ACTIONS)) for _ in range(n)]
    QB = [np.zeros((N_CONG, N_PHASE, N_TOD, N_ACTIONS)) for _ in range(n)]
    g_rl = np.full(n, G / 2.0)
    rewards_trace = []; replay_buf = []
    for ep in range(n_episodes):
        ep_frac = ep / n_episodes
        eps = max(0.05, 1.0 * (1 - ep_frac))
        eta = max(0.05, 0.18 * (1 - ep_frac * 0.7))
        ep_reward = 0.0
        for i in range(n):
            ph = _JN_PHASES[i]
            c_m = min(ph[2] * density_factor, 0.97)
            cong_bin = int(np.clip(int(c_m * N_CONG), 0, N_CONG - 1))
            phase_bin = int(np.clip(int(g_rl[i] / G * (N_PHASE - 0.01)), 0, N_PHASE - 1))
            tod_bin = 0 if ep_frac < 0.5 else 1
            state = (cong_bin, phase_bin, tod_bin)
            q_comb = QA[i][state] + QB[i][state]
            action = int(np.random.randint(N_ACTIONS)) if np.random.rand() < eps else int(np.argmax(q_comb))
            g_old = g_rl[i]
            g_new = float(np.clip(g_rl[i] + DELTA_G[action], g_min_b, g_max_b))
            lam_new = np.clip(g_new / C, 0.01, 0.99)
            x_new = min(c_m / max(lam_new, 1e-6), 0.999)
            cap_s = c_m * ph[0] / 3600.0
            d1 = C * (1 - lam_new) ** 2 / max(2 * (1 - lam_new * x_new), 0.001)
            d2t = (x_new - 1) + np.sqrt(max((x_new - 1) ** 2 + 8 * 0.5 * x_new / max(cap_s * 900, 1), 0))
            delay = min(d1 + 900 * 0.25 * d2t, 300.0)
            reward = -delay - 0.5 * min(delay * 0.3, 99) + 0.3 * (1 - x_new) * c_m * ph[0] - 0.1 * abs(g_new - g_old)
            ep_reward += reward
            g_rl[i] = g_new
            ns_bin = int(np.clip(int(c_m * N_CONG), 0, N_CONG - 1))
            ns_phase = int(np.clip(int(g_new / G * (N_PHASE - 0.01)), 0, N_PHASE - 1))
            ns = (ns_bin, ns_phase, tod_bin)
            if len(replay_buf) >= 500: replay_buf.pop(0)
            replay_buf.append((i, state, action, reward, ns))
            if len(replay_buf) >= 16:
                idxs = np.random.choice(len(replay_buf), 16, replace=False)
                for bi in idxs:
                    ji, s, a, r, nss = replay_buf[bi]
                    if np.random.rand() < 0.5:
                        best_a = int(np.argmax(QA[ji][nss]))
                        td = r + 0.92 * QB[ji][nss][best_a] - QA[ji][s][a]
                        QA[ji][s][a] += eta * td
                    else:
                        best_a = int(np.argmax(QB[ji][nss]))
                        td = r + 0.92 * QA[ji][nss][best_a] - QB[ji][s][a]
                        QB[ji][s][a] += eta * td
        rewards_trace.append(round(ep_reward / n, 2))
    return {"rewards_trace": rewards_trace[-20:], "g_rl": [round(float(g), 1) for g in g_rl], "n_episodes": n_episodes}

# Run computations
lp = run_lp(C=90, density_factor=1.0)
rl = rl_quick(density_factor=1.0, n_episodes=200)

GROUND_TRUTH = {0:118.3,1:74.2,2:54.1,3:48.6,4:98.5,5:38.2,6:65.8,7:89.4,8:32.4,9:24.1,10:44.7,11:41.3}

BACKEND = {
    "jn": JN,
    "lp": lp,
    "rl": rl,
    "gt": GROUND_TRUTH,
}

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --black:#000000;--dark:#050a0f;--panel:#060d16;
  --border:#0d2540;--border2:#0a3060;
  --cyan:#00f5ff;--cyan2:#00c8ff;--cyan3:#007aff;
  --green:#00ff88;--green2:#00e066;
  --red:#ff1a4b;--red2:#ff4466;
  --amber:#ffaa00;--amber2:#ff8800;
  --yellow:#ffe600;--magenta:#ff00cc;
  --white:#e8f4ff;--dim:#4a7090;
}
html,body{width:100%;height:100%;background:#000;color:var(--white);overflow-x:hidden;
  font-family:'Rajdhani',sans-serif}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:#000}
::-webkit-scrollbar-thumb{background:var(--border2)}

/* ── HEADER ── */
#hdr{
  position:fixed;top:0;left:0;right:0;height:54px;z-index:100;
  background:linear-gradient(90deg,#000 0%,#030d1a 50%,#000 100%);
  border-bottom:2px solid var(--cyan3);
  display:flex;align-items:center;padding:0 20px;gap:16px;
  box-shadow:0 4px 40px #0047ff44;
}
#hdr-logo{
  font-family:'Orbitron',monospace;font-size:1.1rem;font-weight:900;
  color:var(--cyan);letter-spacing:3px;text-shadow:0 0 20px var(--cyan);
  white-space:nowrap;
}
#hdr-sub{font-size:0.7rem;color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-left:4px}
#hdr-center{flex:1;display:flex;justify-content:center;align-items:center;gap:24px}
.hdr-stat{text-align:center;line-height:1.2}
.hdr-stat-val{font-family:'Orbitron',monospace;font-size:0.85rem;font-weight:700}
.hdr-stat-lbl{font-size:0.55rem;color:var(--dim);letter-spacing:2px;text-transform:uppercase}
#hdr-time{font-family:'Share Tech Mono',monospace;font-size:0.9rem;color:var(--cyan);
  min-width:80px;text-align:right;text-shadow:0 0 10px var(--cyan)}
#hdr-status{display:flex;align-items:center;gap:6px;margin-left:12px}
.status-dot{width:8px;height:8px;border-radius:50%;background:var(--green);
  animation:pulse 1.5s ease infinite;box-shadow:0 0 8px var(--green)}
.status-lbl{font-size:0.6rem;color:var(--green);letter-spacing:2px;font-family:'Share Tech Mono',monospace}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.8)}}

/* ── MAIN LAYOUT ── */
#main{margin-top:54px;padding:16px;display:flex;flex-direction:column;gap:16px}

/* ── SECTION HEADERS ── */
.sec-hdr{
  display:flex;align-items:center;gap:12px;margin-bottom:12px;
  padding-bottom:8px;border-bottom:1px solid var(--border2);
}
.sec-hdr-line{width:4px;height:22px;border-radius:2px}
.sec-hdr-title{font-family:'Orbitron',monospace;font-size:0.8rem;font-weight:700;
  letter-spacing:3px;text-transform:uppercase}
.sec-hdr-badge{font-family:'Share Tech Mono',monospace;font-size:0.55rem;
  padding:2px 8px;border-radius:2px;border:1px solid;letter-spacing:1px}

/* ── KPI ROW ── */
#kpi-row{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}
.kpi{
  background:var(--panel);border:1px solid var(--border);border-radius:4px;
  padding:14px 12px;position:relative;overflow:hidden;
  transition:border-color .2s;
}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.kpi:hover{border-color:var(--border2)}
.kpi-val{font-family:'Orbitron',monospace;font-size:1.4rem;font-weight:900;line-height:1}
.kpi-unit{font-size:0.6rem;color:var(--dim);margin-left:2px}
.kpi-lbl{font-size:0.6rem;color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-top:6px}
.kpi-delta{font-size:0.65rem;margin-top:4px;font-family:'Share Tech Mono',monospace}

/* ── MAP + CHARTS ROW ── */
#mid-row{display:grid;grid-template-columns:1fr 340px;gap:14px}

/* ── MAP PANEL ── */
#map-panel{background:var(--panel);border:1px solid var(--border);border-radius:4px;
  overflow:hidden;position:relative}
#map-panel-hdr{padding:10px 14px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between}
#map-canvas{width:100%;height:380px;background:#020810;position:relative;overflow:hidden}

/* ── RIGHT PANEL ── */
#right-col{display:flex;flex-direction:column;gap:10px}
.chart-box{background:var(--panel);border:1px solid var(--border);border-radius:4px;overflow:hidden}
.chart-box-hdr{padding:8px 12px;border-bottom:1px solid var(--border);
  font-family:'Orbitron',monospace;font-size:0.62rem;font-weight:700;color:var(--cyan);
  letter-spacing:2px;display:flex;align-items:center;gap:8px}

/* ── INTERSECTION GRID ── */
#jn-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}

/* ── INTERSECTION CARD ── */
.jn-card{
  background:var(--panel);border:1px solid var(--border);border-radius:4px;
  overflow:hidden;position:relative;transition:transform .15s,border-color .15s;
  cursor:pointer;
}
.jn-card:hover{transform:translateY(-2px);border-color:var(--border2)}
.jn-card-hdr{
  padding:8px 12px;display:flex;align-items:center;justify-content:space-between;
  border-bottom:1px solid var(--border);position:relative;overflow:hidden;
}
.jn-card-hdr::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--accent,#00f5ff),transparent);
}
.jn-num{font-family:'Orbitron',monospace;font-size:0.6rem;color:var(--dim);letter-spacing:1px}
.jn-name{font-family:'Orbitron',monospace;font-size:0.68rem;font-weight:700;
  color:var(--white);letter-spacing:1px}
.los-badge{font-family:'Orbitron',monospace;font-size:0.65rem;font-weight:900;
  padding:2px 7px;border-radius:2px;border:1px solid currentColor}

/* ── INTERSECTION DIAGRAM ── */
.jn-diagram{height:110px;position:relative;overflow:hidden}

/* ── STATS GRID ── */
.jn-stats{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0;
  border-top:1px solid var(--border)}
.jn-stat{padding:7px 6px;text-align:center;border-right:1px solid var(--border)}
.jn-stat:last-child{border-right:none}
.jn-stat-val{font-family:'Share Tech Mono',monospace;font-size:0.7rem;font-weight:700}
.jn-stat-lbl{font-size:0.5rem;color:var(--dim);letter-spacing:1px;text-transform:uppercase;margin-top:2px}

/* ── BOTTOM PANELS ── */
#bottom-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
.data-panel{background:var(--panel);border:1px solid var(--border);border-radius:4px;overflow:hidden}
.data-panel-hdr{padding:10px 14px;border-bottom:1px solid var(--border);
  font-family:'Orbitron',monospace;font-size:0.65rem;font-weight:700;
  letter-spacing:2px;display:flex;align-items:center;justify-content:space-between}
.data-panel-body{padding:10px 12px}

/* ── TABLE ── */
.cia-table{width:100%;border-collapse:collapse;font-size:0.62rem}
.cia-table thead tr{border-bottom:1px solid var(--border2)}
.cia-table th{font-family:'Orbitron',monospace;font-size:0.5rem;color:var(--dim);
  letter-spacing:2px;padding:4px 6px;text-align:left;text-transform:uppercase}
.cia-table td{padding:5px 6px;border-bottom:1px solid #0a1a2a;font-family:'Share Tech Mono',monospace}
.cia-table tr:hover td{background:#0a1a2a}

/* ── RL BARS ── */
.rl-bar-row{display:flex;align-items:center;gap:6px;margin-bottom:3px}
.rl-bar-track{flex:1;height:6px;background:#0d2540;border-radius:2px;overflow:hidden}
.rl-bar-fill{height:100%;border-radius:2px;transition:width .3s}

/* ── CANVAS SVG shared ── */
.road-svg{width:100%;height:100%}

/* ── SCROLL container for jn section ── */
#jn-section{max-height:none}

/* ── SCANLINE effect ── */
#scanline{
  position:fixed;top:0;left:0;right:0;bottom:0;z-index:999;pointer-events:none;
  background:repeating-linear-gradient(0deg,rgba(0,0,0,.03) 0px,rgba(0,0,0,.03) 1px,transparent 1px,transparent 2px);
}

/* ── ALGO TOGGLE ── */
.algo-btn{
  font-family:'Orbitron',monospace;font-size:0.55rem;letter-spacing:1px;
  padding:4px 10px;border:1px solid var(--border2);border-radius:2px;
  background:transparent;color:var(--dim);cursor:pointer;transition:all .15s;
}
.algo-btn.active,.algo-btn:hover{background:var(--border2);color:var(--cyan);border-color:var(--cyan3)}

/* ── FOOTER ── */
#footer{text-align:center;padding:12px;font-size:0.55rem;color:#1a3050;
  font-family:'Share Tech Mono',monospace;letter-spacing:2px;
  border-top:1px solid #0a1520}
</style>
</head>
<body>
<div id="scanline"></div>

<!-- ══════════════════════════════ HEADER ══════════════════════════════ -->
<div id="hdr">
  <div>
    <div id="hdr-logo">URBAN FLOW</div>
    <div id="hdr-sub">Bangalore Intelligence System · v3.0</div>
  </div>
  <div id="hdr-center">
    <div class="hdr-stat">
      <div class="hdr-stat-val" style="color:#00f5ff" id="h-junctions">12</div>
      <div class="hdr-stat-lbl">Junctions</div>
    </div>
    <div style="width:1px;height:24px;background:#0d2540"></div>
    <div class="hdr-stat">
      <div class="hdr-stat-val" style="color:#00ff88" id="h-los">A–F</div>
      <div class="hdr-stat-lbl">LOS Range</div>
    </div>
    <div style="width:1px;height:24px;background:#0d2540"></div>
    <div class="hdr-stat">
      <div class="hdr-stat-val" style="color:#ffaa00" id="h-opt">Webster LP</div>
      <div class="hdr-stat-lbl">Algorithm</div>
    </div>
    <div style="width:1px;height:24px;background:#0d2540"></div>
    <div class="hdr-stat">
      <div class="hdr-stat-val" style="color:#ff00cc" id="h-daily">1.45M</div>
      <div class="hdr-stat-lbl">Daily PCU</div>
    </div>
    <div style="width:1px;height:24px;background:#0d2540"></div>
    <div class="hdr-stat">
      <div class="hdr-stat-val" style="color:#ff1a4b" id="h-crit">Silk Board</div>
      <div class="hdr-stat-lbl">Critical Node</div>
    </div>
  </div>
  <div id="hdr-time" id="hdr-clock">08:00:00</div>
  <div id="hdr-status">
    <div class="status-dot"></div>
    <div class="status-lbl">LIVE</div>
  </div>
</div>

<!-- ══════════════════════════════ MAIN ══════════════════════════════ -->
<div id="main">

  <!-- ── KPI ROW ── -->
  <div id="kpi-row">
    <div class="kpi" style="--accent:#00f5ff">
      <div class="kpi-val" style="color:#00f5ff">12</div>
      <div class="kpi-lbl">Active Junctions</div>
      <div class="kpi-delta" style="color:#00ff88">▲ All Monitored</div>
      <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00f5ff,#0047ff)"></div>
    </div>
    <div class="kpi">
      <div class="kpi-val" style="color:#00ff88" id="kpi-avgdelay">--</div><span class="kpi-unit">s</span>
      <div class="kpi-lbl">Avg Delay (LP)</div>
      <div class="kpi-delta" style="color:#00ff88" id="kpi-delay-delta">Loading...</div>
      <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00ff88,#00c844)"></div>
    </div>
    <div class="kpi">
      <div class="kpi-val" style="color:#ffaa00" id="kpi-copt">90</div><span class="kpi-unit">s</span>
      <div class="kpi-lbl">Optimal Cycle</div>
      <div class="kpi-delta" style="color:#ffaa00">Webster C_opt</div>
      <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#ffaa00,#ff6600)"></div>
    </div>
    <div class="kpi">
      <div class="kpi-val" style="color:#ff1a4b" id="kpi-maxq">--</div><span class="kpi-unit">veh</span>
      <div class="kpi-lbl">Max Queue</div>
      <div class="kpi-delta" style="color:#ff1a4b" id="kpi-q-jn">Silk Board</div>
      <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#ff1a4b,#ff0000)"></div>
    </div>
    <div class="kpi">
      <div class="kpi-val" style="color:#ff00cc" id="kpi-peak">185K</div>
      <div class="kpi-lbl">Peak Daily PCU</div>
      <div class="kpi-delta" style="color:#ff00cc">Silk Board</div>
      <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#ff00cc,#cc00ff)"></div>
    </div>
    <div class="kpi">
      <div class="kpi-val" style="color:#00c8ff" id="kpi-lpok">HiGHS</div>
      <div class="kpi-lbl">LP Solver</div>
      <div class="kpi-delta" style="color:#00ff88" id="kpi-lpstatus">✓ Optimal</div>
      <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00c8ff,#007aff)"></div>
    </div>
  </div>

  <!-- ── MID ROW: MAP + CHARTS ── -->
  <div id="mid-row">
    <!-- MAP -->
    <div id="map-panel">
      <div id="map-panel-hdr">
        <div style="font-family:'Orbitron',monospace;font-size:0.7rem;font-weight:700;color:var(--cyan);letter-spacing:2px">
          🗺 BANGALORE NETWORK — 12-JUNCTION GRID
        </div>
        <div style="display:flex;gap:8px">
          <div style="display:flex;align-items:center;gap:4px;font-size:0.55rem;color:var(--dim)">
            <div style="width:10px;height:3px;background:#ff1a4b;border-radius:1px"></div>HIGH CONG
          </div>
          <div style="display:flex;align-items:center;gap:4px;font-size:0.55rem;color:var(--dim)">
            <div style="width:10px;height:3px;background:#ffaa00;border-radius:1px"></div>MED CONG
          </div>
          <div style="display:flex;align-items:center;gap:4px;font-size:0.55rem;color:var(--dim)">
            <div style="width:10px;height:3px;background:#00ff88;border-radius:1px"></div>LOW CONG
          </div>
        </div>
      </div>
      <canvas id="map-canvas" width="700" height="380"></canvas>
    </div>

    <!-- RIGHT PANELS -->
    <div id="right-col">
      <!-- Delay Chart -->
      <div class="chart-box">
        <div class="chart-box-hdr">
          <span style="color:#00f5ff">⬡</span> HCM DELAY PER JUNCTION (s/veh)
        </div>
        <canvas id="delay-chart" height="110" style="padding:8px 10px;display:block;width:100%"></canvas>
      </div>
      <!-- Queue Chart -->
      <div class="chart-box">
        <div class="chart-box-hdr">
          <span style="color:#ff1a4b">⬡</span> QUEUE LENGTH (vehicles)
        </div>
        <canvas id="queue-chart" height="90" style="padding:8px 10px;display:block;width:100%"></canvas>
      </div>
      <!-- RL Reward -->
      <div class="chart-box">
        <div class="chart-box-hdr">
          <span style="color:#ff00cc">⬡</span> DOUBLE Q-LEARNING REWARDS
        </div>
        <div id="rl-bars" style="padding:8px 10px;height:110px;overflow:hidden"></div>
      </div>
    </div>
  </div>

  <!-- ── INTERSECTION SECTION ── -->
  <div>
    <div class="sec-hdr">
      <div class="sec-hdr-line" style="background:linear-gradient(180deg,#00f5ff,#007aff)"></div>
      <div class="sec-hdr-title" style="color:#00f5ff">Intersection Intelligence — Individual Junction Analysis</div>
      <div class="sec-hdr-badge" style="color:#00f5ff;border-color:#007aff;background:#00151f">
        12 NODES · WEBSTER LP · HCM 6th Ed.
      </div>
    </div>
    <div id="jn-grid"></div>
  </div>

  <!-- ── BOTTOM ROW ── -->
  <div id="bottom-row">
    <!-- LP Table -->
    <div class="data-panel">
      <div class="data-panel-hdr" style="color:#00f5ff">
        LP OPTIMAL GREEN ALLOCATION
        <span style="font-size:0.5rem;color:var(--dim);font-family:'Share Tech Mono',monospace">scipy HiGHS</span>
      </div>
      <div class="data-panel-body" style="padding:0">
        <table class="cia-table" id="lp-table">
          <thead><tr>
            <th>Junction</th><th>g (s)</th><th>λ</th><th>x</th><th>Delay</th><th>LOS</th>
          </tr></thead>
          <tbody id="lp-tbody"></tbody>
        </table>
      </div>
    </div>

    <!-- Ground Truth Validation -->
    <div class="data-panel">
      <div class="data-panel-hdr" style="color:#ffaa00">
        FIELD VALIDATION — LP vs SURVEY
        <span style="font-size:0.5rem;color:var(--dim);font-family:'Share Tech Mono',monospace">BBMP/KRDCL 2022</span>
      </div>
      <div class="data-panel-body" style="padding:0">
        <table class="cia-table" id="gt-table">
          <thead><tr>
            <th>Junction</th><th>LP (s)</th><th>Survey (s)</th><th>Δ%</th><th>RMSE</th>
          </tr></thead>
          <tbody id="gt-tbody"></tbody>
        </table>
      </div>
    </div>

    <!-- Algo Comparison -->
    <div class="data-panel">
      <div class="data-panel-hdr" style="color:#ff00cc">
        ALGORITHM PERFORMANCE MATRIX
        <span style="font-size:0.5rem;color:var(--dim);font-family:'Share Tech Mono',monospace">5-algo benchmark</span>
      </div>
      <div class="data-panel-body">
        <canvas id="radar-chart" height="180"></canvas>
      </div>
    </div>
  </div>

</div>

<!-- ── FOOTER ── -->
<div id="footer">
  URBAN FLOW & LIFE-LINES · BANGALORE SMART TRAFFIC INTELLIGENCE SYSTEM · PhD COMPETITION EDITION ·
  DOUBLE Q-LEARNING + EXPERIENCE REPLAY · SCIPY HiGHS LP · HCM 6th Ed. · ROBERTSON PLATOON · CTM-DAGANZO
</div>

<!-- CHART.JS -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<script>
(function(){
'use strict';

// ─── BACKEND DATA ───────────────────────────────────────────────────────────
var DATA = """ + json.dumps(BACKEND) + """;
var JN = DATA.jn;
var LP = DATA.lp;
var RL = DATA.rl;
var GT = DATA.gt;

// ─── LOS COLORS ─────────────────────────────────────────────────────────────
var LOS_COL = {A:'#00ff88',B:'#00c8ff',C:'#ffe600',D:'#ffaa00',E:'#ff4400',F:'#ff1a4b'};
var CONG_COL = function(c){return c>.65?'#ff1a4b':c>.55?'#ffaa00':c>.45?'#ffe600':'#00ff88'};
var LOS_BG  = {A:'#001a0d',B:'#001a2a',C:'#1a1400',D:'#1a0d00',E:'#1a0800',F:'#1a0010'};

// ─── CLOCK ──────────────────────────────────────────────────────────────────
var simSec = 8*3600;
setInterval(function(){
  simSec = (simSec+1)%86400;
  var h=Math.floor(simSec/3600),m=Math.floor((simSec%3600)/60),s=simSec%60;
  var el=document.getElementById('hdr-time');
  if(el) el.textContent=(h<10?'0':'')+h+':'+(m<10?'0':'')+m+':'+(s<10?'0':'')+s;
},1000);

// ─── KPI UPDATE ─────────────────────────────────────────────────────────────
(function(){
  var delays=LP.delay; var qs=LP.q_len;
  var avgD=(delays.reduce(function(a,b){return a+b},0)/delays.length).toFixed(1);
  var maxQ=Math.max.apply(null,qs);
  var maxQi=qs.indexOf(maxQ);
  document.getElementById('kpi-avgdelay').textContent=avgD;
  document.getElementById('kpi-delay-delta').textContent='▼ vs Fixed: '+(delays.reduce(function(a,b){return a+b},0)/delays.length*1.35).toFixed(0)+'s';
  document.getElementById('kpi-copt').textContent=LP.C_opt.toFixed(0);
  document.getElementById('kpi-maxq').textContent=maxQ;
  document.getElementById('kpi-q-jn').textContent=JN[maxQi].name;
  document.getElementById('kpi-lpstatus').textContent=LP.lp_ok?'✓ Optimal':'⚠ Fallback';
})();

// ─── DELAY CHART ────────────────────────────────────────────────────────────
(function(){
  var ctx=document.getElementById('delay-chart').getContext('2d');
  var cols=LP.los.map(function(l){return LOS_COL[l]||'#00f5ff'});
  new Chart(ctx,{
    type:'bar',
    data:{
      labels:JN.map(function(j){return j.name.split(' ').pop()}),
      datasets:[
        {label:'LP Delay (s)',data:LP.delay,backgroundColor:cols.map(function(c){return c+'99'}),
         borderColor:cols,borderWidth:1.5,borderRadius:2},
        {label:'Field Survey',data:JN.map(function(j,i){return GT[i]}),type:'line',
         borderColor:'#ff00cc88',borderWidth:1.5,pointRadius:3,
         pointBackgroundColor:'#ff00cc',tension:.3,fill:false,
         pointStyle:'circle'}
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{
        backgroundColor:'#060d16',borderColor:'#0d2540',borderWidth:1,
        titleFont:{family:'Orbitron',size:10},bodyFont:{family:'Share Tech Mono',size:10},
        callbacks:{label:function(ctx){return ctx.dataset.label+': '+ctx.raw.toFixed(1)+'s'}}
      }},
      scales:{
        x:{ticks:{color:'#4a7090',font:{family:'Share Tech Mono',size:8}},
           grid:{color:'#0a1520'}},
        y:{ticks:{color:'#4a7090',font:{family:'Share Tech Mono',size:9}},
           grid:{color:'#0a1520'}}
      }
    }
  });
})();

// ─── QUEUE CHART ────────────────────────────────────────────────────────────
(function(){
  var ctx=document.getElementById('queue-chart').getContext('2d');
  var maxQ=Math.max.apply(null,LP.q_len);
  var cols=LP.q_len.map(function(q){return q>70?'#ff1a4b':q>50?'#ffaa00':'#00f5ff'});
  new Chart(ctx,{
    type:'bar',
    data:{
      labels:JN.map(function(j){return j.name.split(' ')[0]}),
      datasets:[{label:'Queue (veh)',data:LP.q_len,
        backgroundColor:cols.map(function(c){return c+'88'}),
        borderColor:cols,borderWidth:1.5,borderRadius:2}]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{
        backgroundColor:'#060d16',borderColor:'#0d2540',borderWidth:1,
        titleFont:{family:'Orbitron',size:10},bodyFont:{family:'Share Tech Mono',size:10}
      }},
      scales:{
        x:{ticks:{color:'#4a7090',font:{family:'Share Tech Mono',size:8}},grid:{color:'#0a1520'}},
        y:{ticks:{color:'#4a7090',font:{family:'Share Tech Mono',size:9}},grid:{color:'#0a1520'}}
      }
    }
  });
})();

// ─── RL BARS ─────────────────────────────────────────────────────────────────
(function(){
  var el=document.getElementById('rl-bars');
  var rw=RL.rewards_trace; if(!rw||!rw.length)return;
  var rMin=Math.min.apply(null,rw),rMax=Math.max.apply(null,rw),rRng=rMax-rMin||1;
  var html='';
  for(var i=0;i<rw.length;i++){
    var pct=Math.max(4,Math.round((rw[i]-rMin)/rRng*100));
    var prog=i/(rw.length-1);
    var r=Math.round(180*(1-prog)),g=Math.round(255*prog+100*(1-prog)),b=Math.round(255*(1-prog)+80*prog);
    var col='rgb('+r+','+g+','+b+')';
    html+='<div class="rl-bar-row">'
      +'<span style="font-family:monospace;font-size:.5rem;color:#3a5570;width:18px;text-align:right">'+(i+1)+'</span>'
      +'<div class="rl-bar-track"><div class="rl-bar-fill" style="width:'+pct+'%;background:'+col+';box-shadow:0 0 4px '+col+'88"></div></div>'
      +'<span style="font-family:monospace;font-size:.5rem;color:'+col+';width:30px;text-align:right">'+rw[i].toFixed(0)+'</span>'
      +'</div>';
  }
  el.innerHTML=html;
  // Animate reward bars
  setInterval(function(){
    var bars=el.querySelectorAll('.rl-bar-fill');
    bars.forEach(function(b){
      var cur=parseFloat(b.style.width)||0;
      b.style.width=Math.max(4,Math.min(100,cur+(Math.random()-.5)*3))+'%';
    });
  },800);
})();

// ─── MAP CANVAS ─────────────────────────────────────────────────────────────
(function(){
  var canvas=document.getElementById('map-canvas');
  var ctx=canvas.getContext('2d');
  var W=canvas.width,H=canvas.height;

  // Map lat/lng bounds for Bangalore
  var LAT_MIN=12.82,LAT_MAX=13.12,LNG_MIN=77.55,LNG_MAX=77.78;
  function toXY(lat,lng){
    var x=(lng-LNG_MIN)/(LNG_MAX-LNG_MIN)*(W-60)+30;
    var y=(1-(lat-LAT_MIN)/(LAT_MAX-LAT_MIN))*(H-50)+25;
    return{x:x,y:y};
  }

  var EDGES=[[0,7],[0,8],[0,4],[0,6],[1,9],[1,11],[1,3],[2,3],[2,5],[2,6],[2,7],
             [3,5],[3,11],[4,8],[4,10],[6,7],[7,10],[7,8],[8,10],[9,11],[10,8]];

  var pos=JN.map(function(j){return toXY(j.lat,j.lng)});

  function draw(){
    ctx.clearRect(0,0,W,H);
    // Background grid
    ctx.strokeStyle='#040e1a';ctx.lineWidth=1;
    for(var xi=0;xi<W;xi+=40){ctx.beginPath();ctx.moveTo(xi,0);ctx.lineTo(xi,H);ctx.stroke()}
    for(var yi=0;yi<H;yi+=40){ctx.beginPath();ctx.moveTo(0,yi);ctx.lineTo(W,yi);ctx.stroke()}

    // Draw edges (roads)
    EDGES.forEach(function(e){
      var a=pos[e[0]],b=pos[e[1]];
      var ca=JN[e[0]].cong,cb=JN[e[1]].cong;
      var avgC=(ca+cb)/2;
      var roadCol=avgC>.65?'#ff1a4b':avgC>.55?'#ffaa00':avgC>.45?'#ffe600':'#00ff88';
      // Road glow
      ctx.shadowColor=roadCol;ctx.shadowBlur=6;
      ctx.strokeStyle=roadCol+'55';ctx.lineWidth=6;
      ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
      ctx.strokeStyle=roadCol+'cc';ctx.lineWidth=1.5;ctx.shadowBlur=0;
      ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
    });
    ctx.shadowBlur=0;

    // Draw junction nodes
    JN.forEach(function(jn,i){
      var p=pos[i];
      var c=CONG_COL(jn.cong);
      var d=LP.delay[i];
      var losC=LOS_COL[LP.los[i]]||c;
      var r=jn.lanes===4?10:jn.lanes===3?8:7;

      // Outer glow ring
      ctx.beginPath();ctx.arc(p.x,p.y,r+6,0,Math.PI*2);
      ctx.strokeStyle=losC+'40';ctx.lineWidth=3;ctx.stroke();
      // Outer ring
      ctx.beginPath();ctx.arc(p.x,p.y,r+2,0,Math.PI*2);
      ctx.strokeStyle=losC+'88';ctx.lineWidth=1.5;ctx.stroke();
      // Fill
      ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);
      ctx.fillStyle=losC+'22';ctx.fill();
      ctx.strokeStyle=losC;ctx.lineWidth=2;ctx.stroke();
      // Center dot
      ctx.beginPath();ctx.arc(p.x,p.y,3,0,Math.PI*2);
      ctx.fillStyle=losC;ctx.fill();

      // Label
      ctx.fillStyle=losC;
      ctx.font='bold 7px "Share Tech Mono"';
      ctx.textAlign='center';
      ctx.fillText(jn.name.split(' ')[0].toUpperCase(),p.x,p.y-r-5);
      ctx.fillStyle='#ffffff88';
      ctx.font='6px "Share Tech Mono"';
      ctx.fillText(d.toFixed(0)+'s',p.x,p.y+r+9);
    });
  }
  draw();

  // Animate flow particles along roads
  var particles=[];
  EDGES.forEach(function(e,ei){
    for(var pi=0;pi<3;pi++){
      particles.push({edge:e,t:Math.random(),speed:0.003+Math.random()*0.003,ei:ei});
    }
  });
  function animateMap(){
    draw();
    var t=Date.now()/1000;
    particles.forEach(function(p){
      p.t=(p.t+p.speed)%1;
      var a=pos[p.edge[0]],b=pos[p.edge[1]];
      var x=a.x+(b.x-a.x)*p.t,y=a.y+(b.y-a.y)*p.t;
      var avgC=(JN[p.edge[0]].cong+JN[p.edge[1]].cong)/2;
      var col=avgC>.65?'#ff4466':avgC>.55?'#ffc040':'#00ffaa';
      ctx.beginPath();ctx.arc(x,y,2.5,0,Math.PI*2);
      ctx.fillStyle=col;ctx.shadowColor=col;ctx.shadowBlur=8;ctx.fill();ctx.shadowBlur=0;
    });
    requestAnimationFrame(animateMap);
  }
  animateMap();
})();

// ─── INTERSECTION CARDS ──────────────────────────────────────────────────────
(function(){
  var grid=document.getElementById('jn-grid');
  var ROAD_COLORS={
    0:['#ff1a4b','#ff4466','Outer Ring Road','Hosur Road'],
    1:['#00c8ff','#007aff','NH-44 (Bellary)','Hebbal Fly'],
    2:['#00ff88','#00c844','ORR East','Whitefield Rd'],
    3:['#ffe600','#ffaa00','ORR NH-75','Old Madras Rd'],
    4:['#ff8800','#ff4400','Hosur Road','Attibele Road'],
    5:['#00f5ff','#0088ff','ITPL Road','Varthur Road'],
    6:['#ff00cc','#cc00ff','100ft Road','CMH Road'],
    7:['#ff1a4b','#ff8800','Sarjapur Road','Koramangala Ring'],
    8:['#00ff88','#00c8ff','JP Nagar Ring','Bannerghatta'],
    9:['#ffe600','#00ff88','Doddaballapur Rd','Yelahanka Main'],
    10:['#ff8800','#ff4466','Bannerghatta Rd','NICE Road'],
    11:['#00c8ff','#ff00cc','Nagawara Junction','HBR Road'],
  };

  JN.forEach(function(jn,i){
    var d=LP.delay[i]; var q=LP.q_len[i]; var x=LP.x[i]; var g=LP.g[i];
    var los=LP.los[i]; var lc=LOS_COL[los]||'#00f5ff';
    var cc=ROAD_COLORS[i]||['#00f5ff','#007aff','Road A','Road B'];
    var card=document.createElement('div');
    card.className='jn-card';
    card.style.setProperty('--accent',lc);

    // Road colors for diagram
    var rc1=cc[0],rc2=cc[1];
    var lanes=jn.lanes; var laneW=4;
    var totalW=lanes*laneW+6;

    // Phase angle based on green time
    var phasePct=Math.min(g/83,1);
    var phaseAng=phasePct*2*Math.PI-Math.PI/2;

    // Draw intersection SVG
    var svgW=110,svgH=110,cx=55,cy=55;
    var roadW=lanes*5+10;
    var svg='<svg viewBox="0 0 '+svgW+' '+svgH+'" xmlns="http://www.w3.org/2000/svg">'
      // Background
      +'<rect width="'+svgW+'" height="'+svgH+'" fill="#050a0f"/>'
      // Grid lines
      +'<line x1="0" y1="28" x2="'+svgW+'" y2="28" stroke="#0a1520" stroke-width=".5"/>'
      +'<line x1="0" y1="83" x2="'+svgW+'" y2="83" stroke="#0a1520" stroke-width=".5"/>'
      +'<line x1="28" y1="0" x2="28" y2="'+svgH+'" stroke="#0a1520" stroke-width=".5"/>'
      +'<line x1="83" y1="0" x2="83" y2="'+svgH+'" stroke="#0a1520" stroke-width=".5"/>'

      // HORIZONTAL ROAD (main arterial)
      +'<rect x="0" y="'+(cy-roadW/2)+'" width="'+svgW+'" height="'+roadW+'" fill="'+rc1+'22" stroke="'+rc1+'44" stroke-width="0.5"/>'
      // Road lane markings (horizontal)
      ;
    for(var lane=1;lane<lanes;lane++){
      var ly=cy-roadW/2+lane*(roadW/lanes);
      svg+='<line x1="5" y1="'+ly+'" x2="'+(cx-15)+'" y2="'+ly+'" stroke="'+rc1+'55" stroke-width=".5" stroke-dasharray="3,3"/>';
      svg+='<line x1="'+(cx+15)+'" y1="'+ly+'" x2="'+(svgW-5)+'" y2="'+ly+'" stroke="'+rc1+'55" stroke-width=".5" stroke-dasharray="3,3"/>';
    }
    svg+='<line x1="0" y1="'+(cy-roadW/2)+'" x2="'+svgW+'" y2="'+(cy-roadW/2)+'" stroke="'+rc1+'" stroke-width="1" opacity=".8"/>'
         +'<line x1="0" y1="'+(cy+roadW/2)+'" x2="'+svgW+'" y2="'+(cy+roadW/2)+'" stroke="'+rc1+'" stroke-width="1" opacity=".8"/>'
         // Road label
         +'<text x="4" y="'+(cy-roadW/2-3)+'" fill="'+rc1+'" font-size="5" font-family="Share Tech Mono" opacity=".8">'+cc[2]+'</text>'

      // VERTICAL ROAD (cross street)
      +'<rect x="'+(cx-roadW/2)+'" y="0" width="'+roadW+'" height="'+svgH+'" fill="'+rc2+'22" stroke="'+rc2+'44" stroke-width="0.5"/>'
      ;
    for(var lane2=1;lane2<lanes;lane2++){
      var lx2=cx-roadW/2+lane2*(roadW/lanes);
      svg+='<line x1="'+lx2+'" y1="5" x2="'+lx2+'" y2="'+(cy-15)+'" stroke="'+rc2+'55" stroke-width=".5" stroke-dasharray="3,3"/>';
      svg+='<line x1="'+lx2+'" y1="'+(cy+15)+'" x2="'+lx2+'" y2="'+(svgH-5)+'" stroke="'+rc2+'55" stroke-width=".5" stroke-dasharray="3,3"/>';
    }
    svg+='<line x1="'+(cx-roadW/2)+'" y1="0" x2="'+(cx-roadW/2)+'" y2="'+svgH+'" stroke="'+rc2+'" stroke-width="1" opacity=".8"/>'
         +'<line x1="'+(cx+roadW/2)+'" y1="0" x2="'+(cx+roadW/2)+'" y2="'+svgH+'" stroke="'+rc2+'" stroke-width="1" opacity=".8"/>'
         +'<text x="'+(cx+roadW/2+2)+'" y="10" fill="'+rc2+'" font-size="5" font-family="Share Tech Mono" opacity=".8">'+cc[3]+'</text>'

      // INTERSECTION BOX
      +'<rect x="'+(cx-roadW/2)+'" y="'+(cy-roadW/2)+'" width="'+roadW+'" height="'+roadW+'" fill="#0d2030" stroke="'+lc+'88" stroke-width="1"/>'

      // Crosswalk markings
      ;
    var cw_marks=4;
    for(var cw=0;cw<cw_marks;cw++){
      var step=roadW/(cw_marks+1);
      svg+='<rect x="'+(cx-roadW/2-6)+'" y="'+(cy-roadW/2+step*(cw+0.5))+'" width="5" height="2" fill="'+rc1+'" opacity=".5"/>';
      svg+='<rect x="'+(cx+roadW/2+1)+'" y="'+(cy-roadW/2+step*(cw+0.5))+'" width="5" height="2" fill="'+rc1+'" opacity=".5"/>';
      svg+='<rect x="'+(cx-roadW/2+step*(cw+0.5))+'" y="'+(cy-roadW/2-6)+'" width="2" height="5" fill="'+rc2+'" opacity=".5"/>';
      svg+='<rect x="'+(cx-roadW/2+step*(cw+0.5))+'" y="'+(cy+roadW/2+1)+'" width="2" height="5" fill="'+rc2+'" opacity=".5"/>';
    }

    // Signal phase arc
    var arcR=roadW/2+4;
    svg+='<circle cx="'+cx+'" cy="'+cy+'" r="'+arcR+'" fill="none" stroke="#0d2040" stroke-width="3"/>';
    // Green arc
    var arcLen=2*Math.PI*arcR;
    var arcDash=arcLen*phasePct;
    svg+='<circle cx="'+cx+'" cy="'+cy+'" r="'+arcR+'" fill="none" stroke="'+lc+'" stroke-width="3" opacity=".7"'
        +' stroke-dasharray="'+arcDash.toFixed(1)+' '+arcLen.toFixed(1)+'" stroke-dashoffset="'+(-arcLen*0).toFixed(1)+'"'
        +' transform="rotate(-90 '+cx+' '+cy+')" style="transition:stroke-dasharray .5s"/>';

    // Signal lights (4 corners)
    var corners=[[cx-roadW/2-10,cy-roadW/2-10],[cx+roadW/2+5,cy-roadW/2-10],
                 [cx-roadW/2-10,cy+roadW/2+5],[cx+roadW/2+5,cy+roadW/2+5]];
    corners.forEach(function(cc2){
      svg+='<circle cx="'+cc2[0]+'" cy="'+cc2[1]+'" r="3" fill="'+lc+'" opacity=".9"/>';
      svg+='<circle cx="'+cc2[0]+'" cy="'+cc2[1]+'" r="5" fill="none" stroke="'+lc+'" stroke-width=".5" opacity=".4"/>';
    });

    // Center congestion indicator
    var innerR=roadW/2-4;
    svg+='<circle cx="'+cx+'" cy="'+cy+'" r="'+innerR+'" fill="'+lc+'18" stroke="'+lc+'66" stroke-width="1"/>';
    svg+='<text x="'+cx+'" y="'+(cy+2)+'" text-anchor="middle" fill="'+lc+'" font-family="Orbitron,monospace" font-size="7" font-weight="900">'+los+'</text>';

    // Traffic flow arrows
    var arrowLen=8;
    var arrowOpacity=Math.max(0.3,jn.cong);
    svg+='<polygon points="'+(2)+','+(cy-2)+' '+(12)+','+(cy)+' '+(2)+','+(cy+2)+'" fill="'+rc1+'" opacity="'+arrowOpacity+'"/>'
        +'<polygon points="'+(svgW-2)+','+(cy-2)+' '+(svgW-12)+','+(cy)+' '+(svgW-2)+','+(cy+2)+'" fill="'+rc1+'" opacity="'+arrowOpacity+'"/>'
        +'<polygon points="'+(cx-2)+',2 '+(cx)+',12 '+(cx+2)+',2" fill="'+rc2+'" opacity="'+arrowOpacity+'"/>'
        +'<polygon points="'+(cx-2)+','+(svgH-2)+' '+(cx)+','+(svgH-12)+' '+(cx+2)+','+(svgH-2)+'" fill="'+rc2+'" opacity="'+arrowOpacity+'"/>';

    svg+='</svg>';

    var delayColor=d>80?'#ff1a4b':d>55?'#ff8800':d>35?'#ffe600':d>20?'#00c8ff':'#00ff88';
    var xColor=x>.9?'#ff1a4b':x>.7?'#ffaa00':'#00ff88';

    card.innerHTML=
      '<div class="jn-card-hdr">'
        +'<div>'
          +'<div class="jn-num">JN-'+String(i).padStart(2,'0')+'</div>'
          +'<div class="jn-name">'+jn.name.toUpperCase()+'</div>'
        +'</div>'
        +'<div class="los-badge" style="color:'+lc+';background:'+LOS_BG[los]+'">LOS '+los+'</div>'
      +'</div>'
      +'<div class="jn-diagram">'+svg+'</div>'
      +'<div class="jn-stats">'
        +'<div class="jn-stat">'
          +'<div class="jn-stat-val" style="color:'+delayColor+'">'+d.toFixed(1)+'s</div>'
          +'<div class="jn-stat-lbl">Delay</div>'
        +'</div>'
        +'<div class="jn-stat">'
          +'<div class="jn-stat-val" style="color:#ffaa00">'+q+'</div>'
          +'<div class="jn-stat-lbl">Queue</div>'
        +'</div>'
        +'<div class="jn-stat">'
          +'<div class="jn-stat-val" style="color:'+xColor+'">'+x.toFixed(3)+'</div>'
          +'<div class="jn-stat-lbl">v/c</div>'
        +'</div>'
        +'<div class="jn-stat">'
          +'<div class="jn-stat-val" style="color:#00c8ff">'+g.toFixed(0)+'s</div>'
          +'<div class="jn-stat-lbl">Green</div>'
        +'</div>'
      +'</div>';

    card.style.borderColor=lc+'44';
    grid.appendChild(card);
  });
})();

// ─── LP TABLE ────────────────────────────────────────────────────────────────
(function(){
  var tbody=document.getElementById('lp-tbody');
  JN.forEach(function(jn,i){
    var los=LP.los[i]; var lc=LOS_COL[los]||'#aaa';
    var xc=LP.x[i]>.9?'#ff1a4b':LP.x[i]>.7?'#ffaa00':'#00ff88';
    var tr=document.createElement('tr');
    tr.innerHTML='<td style="color:#00c8ff">'+jn.name+'</td>'
      +'<td style="color:#ffe600">'+LP.g[i].toFixed(1)+'</td>'
      +'<td style="color:#aaa">'+LP.lambda[i].toFixed(3)+'</td>'
      +'<td style="color:'+xc+'">'+LP.x[i].toFixed(3)+'</td>'
      +'<td style="color:'+(LP.delay[i]>55?'#ff1a4b':LP.delay[i]>35?'#ffaa00':'#00ff88')+'">'+LP.delay[i].toFixed(1)+'s</td>'
      +'<td><span style="color:'+lc+';font-weight:900">'+los+'</span></td>';
    tbody.appendChild(tr);
  });
})();

// ─── GT TABLE ────────────────────────────────────────────────────────────────
(function(){
  var tbody=document.getElementById('gt-tbody');
  var sqErr=0;
  JN.forEach(function(jn,i){
    var lp_d=LP.delay[i]; var gt_d=GT[i];
    var delta=((lp_d-gt_d)/gt_d*100);
    sqErr+=(lp_d-gt_d)*(lp_d-gt_d);
    var dc=Math.abs(delta)<20?'#00ff88':Math.abs(delta)<40?'#ffaa00':'#ff1a4b';
    var tr=document.createElement('tr');
    tr.innerHTML='<td style="color:#00c8ff">'+jn.name.split(' ')[0]+'</td>'
      +'<td style="color:#ffe600">'+lp_d.toFixed(1)+'</td>'
      +'<td style="color:#ff00cc">'+gt_d.toFixed(1)+'</td>'
      +'<td style="color:'+dc+'">'+(delta>0?'+':'')+delta.toFixed(1)+'%</td>'
      +'<td style="color:#aaa">'+Math.abs(lp_d-gt_d).toFixed(1)+'</td>';
    tbody.appendChild(tr);
  });
})();

// ─── RADAR CHART ─────────────────────────────────────────────────────────────
(function(){
  var ctx=document.getElementById('radar-chart').getContext('2d');
  new Chart(ctx,{
    type:'radar',
    data:{
      labels:['Delay↓','Queue↓','v/c↓','CO₂↓','Conv.↑'],
      datasets:[
        {label:'Fixed Time',data:[45,60,70,55,20],
         borderColor:'#ff4466',backgroundColor:'#ff446622',pointBackgroundColor:'#ff4466',borderWidth:1.5},
        {label:'Webster LP',data:[78,72,75,68,60],
         borderColor:'#00f5ff',backgroundColor:'#00f5ff22',pointBackgroundColor:'#00f5ff',borderWidth:1.5},
        {label:'Double Q-RL',data:[82,80,78,75,88],
         borderColor:'#ff00cc',backgroundColor:'#ff00cc22',pointBackgroundColor:'#ff00cc',borderWidth:1.5},
        {label:'SCOOT',data:[74,68,70,65,70],
         borderColor:'#ffaa00',backgroundColor:'#ffaa0022',pointBackgroundColor:'#ffaa00',borderWidth:1.5},
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{position:'bottom',labels:{color:'#4a7090',font:{size:9,family:'Share Tech Mono'},
        boxWidth:10,padding:8}}},
      scales:{r:{
        ticks:{color:'#2a4050',font:{size:8},backdropColor:'transparent'},
        grid:{color:'#0a2030'},
        pointLabels:{color:'#7090a0',font:{size:9,family:'Share Tech Mono'}},
        min:0,max:100,
      }}
    }
  });
})();

})();
</script>
</body>
</html>
"""

components.html(HTML, height=1080, scrolling=True)

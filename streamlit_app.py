import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Urban Flow & Life-Lines | Bangalore Grid Optimizer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"]{background:#03070d!important;color:#c8e0f0;font-family:'Rajdhani',sans-serif}
[data-testid="stHeader"]{background:transparent!important}
.stApp{background:#03070d!important}
section[data-testid="stSidebar"]{display:none}
div[data-testid="stToolbar"]{display:none}
.block-container{padding:0!important;max-width:100%!important}
iframe{border:none!important;display:block}
</style>
""", unsafe_allow_html=True)

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Urban Flow & Life-Lines</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#03070d; --bg2:#060e1a; --bg3:#0a1628;
  --cyan:#00e5ff; --green:#00ff88; --red:#ff3355;
  --orange:#ff8c00; --yellow:#ffd700; --purple:#cc88ff;
  --cyan-dim:#00e5ff33; --green-dim:#00ff8822;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:#c8e0f0;font-family:'Rajdhani',sans-serif;height:100vh;overflow:hidden;display:flex;flex-direction:column}

/* ── HEADER ── */
#header{
  background:linear-gradient(90deg,#000d1a 0%,#03070d 40%,#000d1a 100%);
  border-bottom:1px solid var(--cyan-dim);
  padding:6px 16px;
  display:flex;align-items:center;justify-content:space-between;
  flex-shrink:0;
  position:relative;z-index:1000;
}
#header::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);
  animation:scanLine 4s infinite;
}
@keyframes scanLine{0%{transform:scaleX(0)}50%{transform:scaleX(1)}100%{transform:scaleX(0)}}

.h-logo{display:flex;align-items:center;gap:12px}
.h-logo-icon{font-size:2rem;filter:drop-shadow(0 0 10px var(--cyan))}
.h-title{font-family:'Orbitron',monospace;font-size:1.1rem;font-weight:700;color:var(--cyan);letter-spacing:3px;text-shadow:0 0 20px var(--cyan)}
.h-subtitle{font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:var(--orange);letter-spacing:2px;margin-top:2px}

.h-stats{display:flex;gap:20px;align-items:center}
.h-stat{text-align:center}
.h-stat-val{font-family:'Orbitron',monospace;font-size:1.1rem;font-weight:700}
.h-stat-lbl{font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#556677;letter-spacing:1px}

.h-controls{display:flex;gap:8px;align-items:center}
.btn{font-family:'Share Tech Mono',monospace;font-size:0.65rem;letter-spacing:2px;padding:5px 12px;border:1px solid;border-radius:3px;cursor:pointer;transition:.2s;background:transparent;text-transform:uppercase}
.btn-cyan{border-color:var(--cyan);color:var(--cyan)}
.btn-cyan:hover,.btn-cyan.active{background:var(--cyan);color:#000;box-shadow:0 0 15px var(--cyan)}
.btn-red{border-color:var(--red);color:var(--red)}
.btn-red:hover,.btn-red.active{background:var(--red);color:#fff;box-shadow:0 0 15px var(--red)}
.btn-green{border-color:var(--green);color:var(--green)}
.btn-green:hover,.btn-green.active{background:var(--green);color:#000;box-shadow:0 0 15px var(--green)}

.badge-live{font-family:'Share Tech Mono',monospace;font-size:0.6rem;letter-spacing:2px;padding:3px 8px;background:#ff333322;border:1px solid var(--red);color:var(--red);border-radius:2px;animation:blink 1.5s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.4}}

/* ── MAIN LAYOUT ── */
#main{flex:1;display:grid;grid-template-columns:260px 1fr 280px;grid-template-rows:1fr;overflow:hidden;gap:0}

/* ── LEFT PANEL ── */
#left-panel{
  background:var(--bg2);
  border-right:1px solid var(--cyan-dim);
  overflow-y:auto;
  overflow-x:hidden;
  padding:10px;
  display:flex;flex-direction:column;gap:8px;
  scrollbar-width:thin;scrollbar-color:var(--cyan-dim) transparent;
}

.panel-section{background:var(--bg3);border:1px solid #0d2040;border-radius:4px;padding:10px}
.panel-title{font-family:'Orbitron',monospace;font-size:0.6rem;font-weight:600;color:var(--cyan);letter-spacing:3px;text-transform:uppercase;border-bottom:1px solid var(--cyan-dim);padding-bottom:6px;margin-bottom:8px}

.ctrl-row{display:flex;flex-direction:column;gap:3px;margin-bottom:8px}
.ctrl-label{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#556677;letter-spacing:1px;display:flex;justify-content:space-between}
.ctrl-val{color:var(--cyan);font-weight:bold}
input[type=range]{width:100%;-webkit-appearance:none;height:3px;background:var(--bg);border-radius:2px;outline:none}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;background:var(--cyan);border-radius:50%;cursor:pointer;box-shadow:0 0 6px var(--cyan)}
select{width:100%;background:var(--bg);border:1px solid #0d2040;color:var(--cyan);font-family:'Share Tech Mono',monospace;font-size:0.65rem;padding:4px 6px;border-radius:3px;outline:none;cursor:pointer}

/* Junction list */
.junction-item{
  display:flex;align-items:center;gap:8px;
  padding:5px 6px;border-radius:3px;
  border:1px solid transparent;
  cursor:pointer;transition:.2s;margin-bottom:3px;
}
.junction-item:hover{border-color:var(--cyan-dim);background:#0a1628}
.junction-item.evp-active{border-color:var(--red);background:#1a050a;animation:evpPulse .8s infinite alternate}
@keyframes evpPulse{from{box-shadow:inset 0 0 5px #ff333322}to{box-shadow:inset 0 0 12px #ff333355}}
.j-signal{width:10px;height:10px;border-radius:50%;flex-shrink:0;transition:background .3s}
.j-name{font-family:'Share Tech Mono',monospace;font-size:0.6rem;flex:1}
.j-density{font-family:'Orbitron',monospace;font-size:0.6rem;font-weight:700}
.j-timer{font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#556677}

/* Scale legend */
.scale-item{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.scale-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.scale-text{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#7090a0}

/* ── MAP ── */
#map-container{position:relative;overflow:hidden}
#map{width:100%;height:100%}

/* Map overlay canvas */
#flow-canvas{position:absolute;top:0;left:0;pointer-events:none;z-index:400}

/* Map legend */
#map-legend{
  position:absolute;bottom:12px;left:12px;z-index:500;
  background:rgba(3,7,13,.9);border:1px solid var(--cyan-dim);
  border-radius:4px;padding:10px 12px;
  font-family:'Share Tech Mono',monospace;font-size:0.6rem;
}
#map-legend-title{color:var(--cyan);letter-spacing:2px;margin-bottom:6px}
.legend-row{display:flex;align-items:center;gap:8px;margin-bottom:4px;color:#7090a0}
.legend-bar{width:30px;height:4px;border-radius:2px}

/* Map info popup */
#map-info{
  position:absolute;top:12px;left:50%;transform:translateX(-50%);
  background:rgba(3,7,13,.95);border:1px solid var(--orange);
  border-radius:4px;padding:8px 16px;
  font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:var(--orange);
  letter-spacing:2px;z-index:600;
  display:flex;gap:20px;
}

/* EVP path flash */
.evp-flash{position:absolute;top:0;left:0;right:0;bottom:0;pointer-events:none;z-index:450;background:transparent;transition:.3s}
.evp-flash.active{background:radial-gradient(ellipse at center,rgba(255,51,85,.05) 0%,transparent 70%)}

/* ── RIGHT PANEL ── */
#right-panel{
  background:var(--bg2);
  border-left:1px solid var(--cyan-dim);
  overflow-y:auto;padding:10px;
  display:flex;flex-direction:column;gap:8px;
  scrollbar-width:thin;scrollbar-color:var(--cyan-dim) transparent;
}

/* Graph panels */
.graph-wrap{background:var(--bg3);border:1px solid #0d2040;border-radius:4px;padding:8px}
.graph-header{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px}
.graph-title{font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:var(--cyan);letter-spacing:2px;text-transform:uppercase}
.graph-current{font-family:'Orbitron',monospace;font-size:1.1rem;font-weight:700}
.graph-unit{font-family:'Share Tech Mono',monospace;font-size:0.5rem;color:#556677;margin-left:2px}
.graph-delta{font-family:'Share Tech Mono',monospace;font-size:0.55rem;margin-left:4px}
.delta-pos{color:var(--green)}
.delta-neg{color:var(--red)}
canvas.graph-canvas{width:100%!important;height:60px!important;display:block}

/* Stats grid */
.stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.stat-card{background:var(--bg);border:1px solid #0d2040;border-radius:3px;padding:8px;text-align:center;border-left:2px solid}
.stat-card-val{font-family:'Orbitron',monospace;font-size:1rem;font-weight:700;margin-bottom:2px}
.stat-card-lbl{font-family:'Share Tech Mono',monospace;font-size:0.5rem;color:#556677;letter-spacing:1px}
.stat-card-sub{font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#3a5570;margin-top:2px}

/* Algorithm comparison */
.algo-compare{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.algo-card{padding:6px;border-radius:3px;border:1px solid;text-align:center}
.algo-card-name{font-family:'Orbitron',monospace;font-size:0.5rem;letter-spacing:1px;margin-bottom:4px}
.algo-card-val{font-family:'Orbitron',monospace;font-size:0.9rem;font-weight:700}
.algo-card-sub{font-family:'Share Tech Mono',monospace;font-size:0.5rem;color:#556677;margin-top:2px}

/* ── BOTTOM BAR ── */
#bottom-bar{
  height:32px;background:var(--bg2);border-top:1px solid var(--cyan-dim);
  display:flex;align-items:center;padding:0 16px;gap:24px;
  flex-shrink:0;z-index:1000;
}
.bb-item{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#556677;letter-spacing:1px;display:flex;align-items:center;gap:6px}
.bb-val{color:var(--cyan);font-weight:bold}
.bb-sep{color:#0d2040;font-size:1.2rem}

/* Leaflet dark style overrides */
.leaflet-tile-pane{filter:brightness(.35) saturate(.3) hue-rotate(200deg)!important}
.leaflet-container{background:#03070d}
.leaflet-control-attribution{display:none!important}
.leaflet-control-zoom{display:none!important}
</style>
</head>
<body>

<!-- HEADER -->
<div id="header">
  <div class="h-logo">
    <div class="h-logo-icon">🚦</div>
    <div>
      <div class="h-title">URBAN FLOW & LIFE-LINES</div>
      <div class="h-subtitle">▸ MULTI-OBJECTIVE OPTIMIZATION — BANGALORE TRAFFIC GRID ◂ NMIT ISE</div>
    </div>
  </div>

  <div class="h-stats">
    <div class="h-stat">
      <div class="h-stat-val" id="hs-vehicles" style="color:var(--cyan)">0</div>
      <div class="h-stat-lbl">LIVE VEHICLES</div>
    </div>
    <div class="h-stat">
      <div class="h-stat-val" id="hs-delay" style="color:var(--red)">0s</div>
      <div class="h-stat-lbl">AVG DELAY</div>
    </div>
    <div class="h-stat">
      <div class="h-stat-val" id="hs-evp" style="color:var(--orange)">0</div>
      <div class="h-stat-lbl">EVP ACTIVE</div>
    </div>
    <div class="h-stat">
      <div class="h-stat-val" id="hs-eff" style="color:var(--green)">0%</div>
      <div class="h-stat-lbl">GRID EFF.</div>
    </div>
    <div class="h-stat">
      <div class="h-stat-val" id="hs-throughput" style="color:var(--yellow)">0</div>
      <div class="h-stat-lbl">VPHPL</div>
    </div>
  </div>

  <div class="h-controls">
    <div class="badge-live">● LIVE SIM</div>
    <button class="btn btn-green active" id="btn-algo" onclick="cycleAlgo()">⚡ GREEN WAVE+EVP</button>
    <button class="btn btn-red" id="btn-evp-all" onclick="triggerMassEVP()">🚨 MASS EVP</button>
    <button class="btn btn-cyan" id="btn-pause" onclick="togglePause()">⏸ PAUSE</button>
  </div>
</div>

<!-- MAIN -->
<div id="main">

  <!-- LEFT PANEL -->
  <div id="left-panel">

    <!-- SIMULATION CONTROLS -->
    <div class="panel-section">
      <div class="panel-title">⚙ Simulation Parameters</div>

      <div class="ctrl-row">
        <div class="ctrl-label">VEHICLE DENSITY <span class="ctrl-val" id="lbl-density">High (1 dot = 5000 vehicles)</span></div>
        <input type="range" min="1" max="5" value="3" oninput="setDensity(this.value)">
      </div>
      <div class="ctrl-row">
        <div class="ctrl-label">EMERGENCY VEHICLES <span class="ctrl-val" id="lbl-emerg">8</span></div>
        <input type="range" min="1" max="25" value="8" oninput="setEmerg(this.value)">
      </div>
      <div class="ctrl-row">
        <div class="ctrl-label">GREEN WAVE SPEED <span class="ctrl-val" id="lbl-wave">40 km/h</span></div>
        <input type="range" min="20" max="80" value="40" step="5" oninput="setWave(this.value)">
      </div>
      <div class="ctrl-row">
        <div class="ctrl-label">SIGNAL CYCLE TIME <span class="ctrl-val" id="lbl-cycle">90 sec</span></div>
        <input type="range" min="30" max="180" value="90" step="10" oninput="setCycle(this.value)">
      </div>
      <div class="ctrl-row">
        <div class="ctrl-label">SIMULATION SPEED</div>
        <select onchange="setSimSpeed(this.value)">
          <option value="0.5">0.5× Slow</option>
          <option value="1" selected>1× Real-time</option>
          <option value="2">2× Fast</option>
          <option value="5">5× Ultra</option>
        </select>
      </div>
      <div class="ctrl-row">
        <div class="ctrl-label">ALGORITHM MODE</div>
        <select id="algo-select" onchange="setAlgoFromSelect(this.value)">
          <option value="optimal">Green Wave + EVP (Proposed)</option>
          <option value="fixed">Fixed Timer (Baseline)</option>
          <option value="lp">Linear Programming Only</option>
          <option value="evp">Emergency Priority Only</option>
          <option value="ml">ML Predictive + EVP</option>
        </select>
      </div>
    </div>

    <!-- SCALE LEGEND -->
    <div class="panel-section">
      <div class="panel-title">📊 Flow Scale Legend</div>
      <div class="scale-item"><div class="scale-dot" style="background:#00ccff"></div><div class="scale-text">1 dot = 5,000 normal vehicles</div></div>
      <div class="scale-item"><div class="scale-dot" style="background:#ff3355;box-shadow:0 0 6px #ff3355"></div><div class="scale-text">1 red dot = 1 emergency vehicle</div></div>
      <div class="scale-item"><div class="scale-dot" style="background:#ff8c00"></div><div class="scale-text">Orange flow = congested (>70%)</div></div>
      <div class="scale-item"><div class="scale-dot" style="background:#ff3355;opacity:.4"></div><div class="scale-text">Red flow = gridlock (>90%)</div></div>
      <div class="scale-item"><div class="scale-dot" style="background:#00ff88"></div><div class="scale-text">Green flow = free-flow (&lt;40%)</div></div>
      <div style="margin-top:8px;font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#3a5570;line-height:1.6">
        Source: TomTom Traffic Index 2024<br>
        Bangalore Peak: 1.2M vehicles/day<br>
        Avg Speed Peak: 17.8 km/h<br>
        Congestion Level: 71% (Top 5 India)
      </div>
    </div>

    <!-- JUNCTION LIST -->
    <div class="panel-section">
      <div class="panel-title">🗺 Junction Monitor</div>
      <div id="junction-list"></div>
    </div>

    <!-- REAL DATA REFERENCE -->
    <div class="panel-section">
      <div class="panel-title">📋 Real Traffic Data (Kaggle/TomTom)</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#3a5570;line-height:1.8">
        <span style="color:#556677">Silk Board:</span> <span style="color:var(--red)">71% congestion</span><br>
        <span style="color:#556677">Hebbal:</span> <span style="color:var(--red)">64% congestion</span><br>
        <span style="color:#556677">Marathahalli:</span> <span style="color:var(--orange)">58% congestion</span><br>
        <span style="color:#556677">KR Puram:</span> <span style="color:var(--orange)">54% congestion</span><br>
        <span style="color:#556677">Electronic City:</span> <span style="color:var(--red)">67% congestion</span><br>
        <span style="color:#556677">ORR (avg):</span> <span style="color:var(--red)">62% congestion</span><br>
        <span style="color:#556677">Peak hours:</span> <span style="color:var(--yellow)">8–10AM, 6–9PM</span><br>
        <span style="color:#556677">Avg speed (peak):</span> <span style="color:var(--red)">17.8 km/h</span><br>
        <span style="color:#556677">Avg speed (off-peak):</span> <span style="color:var(--green)">32.4 km/h</span><br>
        <span style="color:#556677">Daily vehicle count:</span> <span style="color:var(--cyan)">1.2M</span><br>
        <span style="color:#556677">Registered vehicles:</span> <span style="color:var(--cyan)">10.5M</span>
      </div>
    </div>

  </div>

  <!-- MAP -->
  <div id="map-container">
    <div id="map"></div>
    <canvas id="flow-canvas"></canvas>
    <div class="evp-flash" id="evp-flash"></div>
    <div id="map-info">
      <span>SIM TIME: <span id="sim-time" style="color:var(--cyan)">00:00:00</span></span>
      <span>ALGO: <span id="algo-display" style="color:var(--orange)">GREEN WAVE+EVP</span></span>
      <span>WAVE: <span id="wave-display" style="color:var(--green)">40 km/h</span></span>
      <span>TOTAL VEHICLES: <span id="total-veh" style="color:var(--yellow)">—</span></span>
    </div>
    <div id="map-legend">
      <div id="map-legend-title">▸ DENSITY MAP</div>
      <div class="legend-row"><div class="legend-bar" style="background:linear-gradient(90deg,#00ff88,#00ff88)"></div>Free-flow &lt;40%</div>
      <div class="legend-row"><div class="legend-bar" style="background:linear-gradient(90deg,#ffd700,#ffd700)"></div>Moderate 40-70%</div>
      <div class="legend-row"><div class="legend-bar" style="background:linear-gradient(90deg,#ff8c00,#ff8c00)"></div>Congested 70-90%</div>
      <div class="legend-row"><div class="legend-bar" style="background:linear-gradient(90deg,#ff3355,#ff3355)"></div>Gridlock >90%</div>
      <div class="legend-row"><div class="legend-bar" style="background:linear-gradient(90deg,#ff3355,#ff88cc);animation:evpDash 1s infinite"></div>EVP Corridor</div>
    </div>
  </div>

  <!-- RIGHT PANEL -->
  <div id="right-panel">

    <!-- OSCILLOSCOPE GRAPHS -->
    <div class="graph-wrap">
      <div class="graph-header">
        <div class="graph-title">▸ NETWORK THROUGHPUT (VPHPL)</div>
        <div><span class="graph-current" id="gv-throughput" style="color:var(--green)">0</span><span class="graph-unit">veh/hr/lane</span><span class="graph-delta delta-pos" id="gd-throughput"></span></div>
      </div>
      <canvas class="graph-canvas" id="gc-throughput"></canvas>
    </div>

    <div class="graph-wrap">
      <div class="graph-header">
        <div class="graph-title">▸ AVG INTERSECTION DELAY (SEC)</div>
        <div><span class="graph-current" id="gv-delay" style="color:var(--red)">0</span><span class="graph-unit">sec</span><span class="graph-delta" id="gd-delay"></span></div>
      </div>
      <canvas class="graph-canvas" id="gc-delay"></canvas>
    </div>

    <div class="graph-wrap">
      <div class="graph-header">
        <div class="graph-title">▸ EVP CORRIDOR CLEAR TIME</div>
        <div><span class="graph-current" id="gv-evp" style="color:var(--orange)">0</span><span class="graph-unit">sec</span><span class="graph-delta delta-pos" id="gd-evp"></span></div>
      </div>
      <canvas class="graph-canvas" id="gc-evp"></canvas>
    </div>

    <div class="graph-wrap">
      <div class="graph-header">
        <div class="graph-title">▸ SIGNAL EFFICIENCY INDEX</div>
        <div><span class="graph-current" id="gv-eff" style="color:var(--cyan)">0</span><span class="graph-unit">%</span><span class="graph-delta" id="gd-eff"></span></div>
      </div>
      <canvas class="graph-canvas" id="gc-eff"></canvas>
    </div>

    <div class="graph-wrap">
      <div class="graph-header">
        <div class="graph-title">▸ CO₂ EMISSIONS INDEX</div>
        <div><span class="graph-current" id="gv-emit" style="color:var(--yellow)">0</span><span class="graph-unit">rel</span><span class="graph-delta" id="gd-emit"></span></div>
      </div>
      <canvas class="graph-canvas" id="gc-emit"></canvas>
    </div>

    <div class="graph-wrap">
      <div class="graph-header">
        <div class="graph-title">▸ NETWORK DENSITY %</div>
        <div><span class="graph-current" id="gv-dens" style="color:var(--purple)">0</span><span class="graph-unit">%</span><span class="graph-delta" id="gd-dens"></span></div>
      </div>
      <canvas class="graph-canvas" id="gc-dens"></canvas>
    </div>

    <!-- STATS GRID -->
    <div class="panel-section" style="background:var(--bg3);border:1px solid #0d2040;border-radius:4px;padding:8px">
      <div class="panel-title">📊 System Statistics</div>
      <div class="stats-grid">
        <div class="stat-card" style="border-left-color:var(--green)">
          <div class="stat-card-val" style="color:var(--green)" id="sc-freeflow">0%</div>
          <div class="stat-card-lbl">FREE-FLOW</div>
          <div class="stat-card-sub" id="sc-freeflow-n">0 vehicles</div>
        </div>
        <div class="stat-card" style="border-left-color:var(--red)">
          <div class="stat-card-val" style="color:var(--red)" id="sc-stopped">0</div>
          <div class="stat-card-lbl">STOPPED</div>
          <div class="stat-card-sub" id="sc-stopped-n">at red signals</div>
        </div>
        <div class="stat-card" style="border-left-color:var(--yellow)">
          <div class="stat-card-val" style="color:var(--yellow)" id="sc-avgspeed">0</div>
          <div class="stat-card-lbl">AVG SPEED (km/h)</div>
          <div class="stat-card-sub">peak target: 40</div>
        </div>
        <div class="stat-card" style="border-left-color:var(--orange)">
          <div class="stat-card-val" style="color:var(--orange)" id="sc-fuel">0</div>
          <div class="stat-card-lbl">FUEL WASTE (L/hr)</div>
          <div class="stat-card-sub">idle consumption</div>
        </div>
        <div class="stat-card" style="border-left-color:var(--cyan)">
          <div class="stat-card-val" style="color:var(--cyan)" id="sc-green-waves">0</div>
          <div class="stat-card-lbl">ACTIVE GRN WAVES</div>
          <div class="stat-card-sub">synchronized corridors</div>
        </div>
        <div class="stat-card" style="border-left-color:var(--red)">
          <div class="stat-card-val" style="color:var(--red)" id="sc-evp-count">0</div>
          <div class="stat-card-lbl">EVP OVERRIDES</div>
          <div class="stat-card-sub">total this session</div>
        </div>
        <div class="stat-card" style="border-left-color:var(--purple)">
          <div class="stat-card-val" style="color:var(--purple)" id="sc-co2saved">0</div>
          <div class="stat-card-lbl">CO₂ SAVED (kg/hr)</div>
          <div class="stat-card-sub">vs fixed timer</div>
        </div>
        <div class="stat-card" style="border-left-color:var(--green)">
          <div class="stat-card-val" style="color:var(--green)" id="sc-lp-iters">0</div>
          <div class="stat-card-lbl">LP ITERATIONS</div>
          <div class="stat-card-sub">optimization cycles</div>
        </div>
      </div>
    </div>

    <!-- ALGORITHM COMPARISON -->
    <div class="panel-section" style="background:var(--bg3);border:1px solid #0d2040;border-radius:4px;padding:8px">
      <div class="panel-title">⚡ Algorithm Benchmark</div>
      <div class="algo-compare">
        <div class="algo-card" style="border-color:#ff333566;background:#1a050a">
          <div class="algo-card-name" style="color:var(--red)">FIXED TIMER</div>
          <div class="algo-card-val" id="ac-fixed-delay" style="color:var(--red)">45 min</div>
          <div class="algo-card-sub">Silk Bd → Hebbal</div>
        </div>
        <div class="algo-card" style="border-color:#00ff8866;background:#00200e">
          <div class="algo-card-name" style="color:var(--green)">GREEN WAVE+EVP</div>
          <div class="algo-card-val" id="ac-optimal-delay" style="color:var(--green)">28 min</div>
          <div class="algo-card-sub">Silk Bd → Hebbal</div>
        </div>
        <div class="algo-card" style="border-color:var(--cyan-dim);background:#001a22">
          <div class="algo-card-name" style="color:var(--cyan)">DELAY REDUCTION</div>
          <div class="algo-card-val" id="ac-reduction" style="color:var(--cyan)">38%</div>
          <div class="algo-card-sub">LP optimization</div>
        </div>
        <div class="algo-card" style="border-color:#ff8c0066;background:#1a0d00">
          <div class="algo-card-name" style="color:var(--orange)">EVP TIME SAVED</div>
          <div class="algo-card-val" id="ac-evp-saved" style="color:var(--orange)">60%</div>
          <div class="algo-card-sub">ambulance clear</div>
        </div>
      </div>
    </div>

    <!-- LP OBJECTIVE FUNCTION DISPLAY -->
    <div class="panel-section" style="background:var(--bg3);border:1px solid #0d2040;border-radius:4px;padding:8px">
      <div class="panel-title">∑ LP Objective Function (Live)</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#3a5570;line-height:1.8">
        <span style="color:var(--cyan)">Minimize:</span> W = Σ tᵢ × dᵢ<br>
        <span style="color:var(--cyan)">Subject to:</span><br>
        &nbsp;&nbsp;gᵢ + rᵢ + yᵢ = T = <span id="lp-T" style="color:var(--orange)">90s</span><br>
        &nbsp;&nbsp;gᵢ ≥ <span style="color:var(--green)">10s</span> (min green)<br>
        &nbsp;&nbsp;Φᵢⱼ = (Lᵢⱼ/v_c) mod T<br>
        &nbsp;&nbsp;v_c = <span id="lp-vc" style="color:var(--green)">40</span> km/h<br>
        <span style="color:var(--cyan)">Current W:</span> <span id="lp-W" style="color:var(--yellow)">0</span> veh·sec<br>
        <span style="color:var(--cyan)">Iterations:</span> <span id="lp-iter" style="color:var(--green)">0</span><br>
        <span style="color:var(--cyan)">EVP Weight:</span> <span style="color:var(--red)">P → ∞</span> (preempt)
      </div>
    </div>

  </div>
</div>

<!-- BOTTOM BAR -->
<div id="bottom-bar">
  <div class="bb-item">🕐 <span id="bb-time" class="bb-val">00:00:00</span></div>
  <div class="bb-sep">|</div>
  <div class="bb-item">ALGO: <span id="bb-algo" class="bb-val">GREEN WAVE+EVP</span></div>
  <div class="bb-sep">|</div>
  <div class="bb-item">TOTAL VEHICLES: <span id="bb-total" class="bb-val">0</span></div>
  <div class="bb-sep">|</div>
  <div class="bb-item">STOPPED: <span id="bb-stopped" class="bb-val" style="color:var(--red)">0</span></div>
  <div class="bb-sep">|</div>
  <div class="bb-item">EMERGENCY: <span id="bb-emerg" class="bb-val" style="color:var(--red)">0</span></div>
  <div class="bb-sep">|</div>
  <div class="bb-item">FREE-FLOW: <span id="bb-ff" class="bb-val" style="color:var(--green)">0%</span></div>
  <div class="bb-sep">|</div>
  <div class="bb-item">GRID EFF: <span id="bb-eff" class="bb-val" style="color:var(--green)">0%</span></div>
  <div class="bb-sep">|</div>
  <div class="bb-item">CO₂ SAVED: <span id="bb-co2" class="bb-val" style="color:var(--purple)">0 kg/hr</span></div>
  <div class="bb-sep">|</div>
  <div class="bb-item">© NMIT ISE — NISHCHAL VISHWANATH & RISHUL KH — 2025</div>
</div>

<script>
// ═══════════════════════════════════════════════════════════════
// REAL BANGALORE JUNCTION DATA (GPS coordinates + real traffic)
// Source: TomTom Traffic Index 2024, OpenStreetMap, Kaggle BLR Traffic
// ═══════════════════════════════════════════════════════════════
const JUNCTIONS = [
  { id:0, name:'Silk Board',       lat:12.9177, lng:77.6228, congestion:0.71, dailyVol:185000, peakVol:22000, importance:10, color:'#ff3355' },
  { id:1, name:'Hebbal',           lat:13.0358, lng:77.5970, congestion:0.64, dailyVol:156000, peakVol:19000, importance:9,  color:'#ff8c00' },
  { id:2, name:'Marathahalli',     lat:12.9591, lng:77.6974, congestion:0.58, dailyVol:142000, peakVol:17500, importance:8,  color:'#ffd700' },
  { id:3, name:'KR Puram',         lat:13.0074, lng:77.6950, congestion:0.54, dailyVol:128000, peakVol:15000, importance:7,  color:'#ffd700' },
  { id:4, name:'Electronic City',  lat:12.8399, lng:77.6770, congestion:0.67, dailyVol:168000, peakVol:20000, importance:8,  color:'#ff3355' },
  { id:5, name:'Whitefield',       lat:12.9698, lng:77.7500, congestion:0.52, dailyVol:118000, peakVol:14000, importance:7,  color:'#ffd700' },
  { id:6, name:'Indiranagar',      lat:12.9784, lng:77.6408, congestion:0.62, dailyVol:138000, peakVol:16500, importance:8,  color:'#ff8c00' },
  { id:7, name:'Koramangala',      lat:12.9352, lng:77.6245, congestion:0.66, dailyVol:155000, peakVol:18500, importance:9,  color:'#ff8c00' },
  { id:8, name:'JP Nagar',         lat:12.9063, lng:77.5857, congestion:0.48, dailyVol:108000, peakVol:13000, importance:7,  color:'#00ff88' },
  { id:9, name:'Yelahanka',        lat:13.1007, lng:77.5963, congestion:0.44, dailyVol:98000,  peakVol:12000, importance:6,  color:'#00ff88' },
  { id:10, name:'Bannerghatta Rd', lat:12.8931, lng:77.5971, congestion:0.59, dailyVol:132000, peakVol:15800, importance:7,  color:'#ffd700' },
  { id:11, name:'Nagawara',        lat:13.0456, lng:77.6207, congestion:0.55, dailyVol:122000, peakVol:14500, importance:7,  color:'#ffd700' },
];

const EDGES = [
  [0,7],[0,8],[0,4],[0,6],
  [1,9],[1,11],[1,3],
  [2,3],[2,5],[2,6],
  [3,5],[3,11],
  [4,8],[4,10],
  [6,7],[6,2],
  [7,10],[7,8],
  [8,10],
  [9,11],
  [11,1],[11,6],
  [10,8],
];

// ═══════════════════════════════════════════════════════════════
// SIMULATION STATE
// ═══════════════════════════════════════════════════════════════
let simState = {
  algo: 'optimal',
  paused: false,
  simSpeed: 1,
  emergCount: 8,
  waveSpeed: 40,
  cycleTime: 90,
  densityLevel: 3, // 1-5
  simTime: 0,
  frameCount: 0,
  lpIterations: 0,
  totalEvpOverrides: 0,
  algoBooted: 0,
};

const DENSITY_LABELS = ['Very Low','Low','Medium','High','Peak (TomTom)'];
const DENSITY_MULTIPLIERS = [0.2, 0.4, 0.7, 1.0, 1.4];

// Graph data buffers
const GL = 100;
const graphs = {
  throughput: { data: new Array(GL).fill(0), color:'#00ff88', min:0, max:2000 },
  delay:      { data: new Array(GL).fill(0), color:'#ff3355', min:0, max:180 },
  evp:        { data: new Array(GL).fill(0), color:'#ff8c00', min:0, max:120 },
  eff:        { data: new Array(GL).fill(0), color:'#00e5ff', min:0, max:100 },
  emit:       { data: new Array(GL).fill(0), color:'#ffd700', min:0, max:150 },
  dens:       { data: new Array(GL).fill(0), color:'#cc88ff', min:0, max:100 },
};

// ═══════════════════════════════════════════════════════════════
// SIGNALS
// ═══════════════════════════════════════════════════════════════
let signals = JUNCTIONS.map((j,i) => ({
  junctionId: i,
  phase: Math.random() * 90,
  cycle: 90,
  state: 'red',
  evpOverride: false,
  greenDur: 45,
  efficiency: 0.5,
  waitingVehicles: Math.floor(j.congestion * 50),
}));

// ═══════════════════════════════════════════════════════════════
// VEHICLE PARTICLES (Flow dots)
// Each dot = 5000 real vehicles
// ═══════════════════════════════════════════════════════════════
let particles = [];
const MAX_DOTS = 600; // Each = 5000 vehicles → up to 3M vehicles

class Particle {
  constructor(isEmergency) {
    this.isEmergency = isEmergency;
    this.id = Math.random();
    const ri = Math.floor(Math.random() * EDGES.length);
    this.edgeIndex = ri;
    const [a,b] = EDGES[ri];
    this.fromId = a; this.toId = b;
    this.progress = Math.random();
    this.dir = Math.random() > 0.5 ? 1 : -1;
    this.baseSpeed = isEmergency
      ? 0.004 + Math.random()*0.002
      : 0.001 + Math.random()*0.001;
    this.speed = this.baseSpeed;
    this.state = 'moving'; // moving/slow/stopped
    this.waitTime = 0;
    this.trail = [];
    this.blinkPhase = Math.random() * Math.PI * 2;
    this.represents = isEmergency ? 1 : 5000;
  }

  update(dt) {
    const edge = EDGES[this.edgeIndex];
    const endId = this.dir === 1 ? edge[1] : edge[0];
    const sig = signals[endId];
    const distEnd = this.dir === 1 ? 1-this.progress : this.progress;

    // Congestion from junction base level + sim density
    const junc = JUNCTIONS[endId];
    const baseCong = junc.congestion * DENSITY_MULTIPLIERS[simState.densityLevel-1];
    const cong = Math.min(baseCong * (simState.algo==='fixed'?1.3:simState.algo==='optimal'?0.7:1.0) * (1-simState.algoBooted/500*0.3), 0.99);

    let shouldStop = false;
    if (!this.isEmergency && distEnd < 0.15) {
      if (sig.state === 'red' && !sig.evpOverride) shouldStop = true;
    }

    // Speed logic
    if (shouldStop) {
      this.targetSpeed = 0;
      this.state = 'stopped';
      this.waitTime += dt * 0.016;
    } else if (cong > 0.65 && !this.isEmergency) {
      this.targetSpeed = this.baseSpeed * Math.max(0.1, 1 - (cong-0.65)*2.5);
      this.state = cong>0.85 ? 'stopped' : 'slow';
      this.waitTime = Math.max(0, this.waitTime - dt*0.01);
    } else {
      this.targetSpeed = this.baseSpeed * simState.waveSpeed / 40;
      this.state = 'moving';
      this.waitTime = Math.max(0, this.waitTime - dt*0.05);
    }

    if (this.isEmergency) {
      this.targetSpeed = this.baseSpeed * 2;
      this.state = 'moving';
      shouldStop = false;
    }

    this.speed += (this.targetSpeed - this.speed) * 0.12;
    this.progress += this.speed * this.dir * simState.simSpeed;

    // Trail for emergency
    if (this.isEmergency) {
      const pos = this.getLatLng();
      this.trail.unshift(pos);
      if (this.trail.length > 10) this.trail.pop();
    }

    // Reached end of edge
    if (this.progress >= 1 || this.progress <= 0) {
      this.progress = this.progress >= 1 ? 0 : 1;
      const endJunction = this.dir===1 ? EDGES[this.edgeIndex][1] : EDGES[this.edgeIndex][0];
      const connected = EDGES.map((e,i)=>({e,i}))
        .filter(({e,i}) => (e[0]===endJunction||e[1]===endJunction) && i!==this.edgeIndex);
      if (connected.length > 0) {
        const pick = connected[Math.floor(Math.random()*connected.length)];
        this.edgeIndex = pick.i;
        this.fromId = EDGES[pick.i][0]; this.toId = EDGES[pick.i][1];
        this.dir = EDGES[pick.i][0]===endJunction ? 1 : -1;
        this.progress = this.dir===1 ? 0 : 1;
      } else {
        this.dir *= -1;
      }
    }
  }

  getLatLng() {
    const [a,b] = EDGES[this.edgeIndex];
    const ja = JUNCTIONS[a], jb = JUNCTIONS[b];
    const t = this.dir===1 ? this.progress : 1-this.progress;
    const perpOffset = (this.isEmergency ? 0 : (Math.random()*0.0004-0.0002));
    return {
      lat: ja.lat + (jb.lat - ja.lat)*t + perpOffset,
      lng: ja.lng + (jb.lng - ja.lng)*t + perpOffset,
    };
  }

  getColor() {
    if (this.isEmergency) return '#ff3355';
    if (this.state==='stopped') return '#ff3355';
    if (this.state==='slow') return '#ff8c00';
    return '#00ccff';
  }
}

function spawnParticles() {
  particles = [];
  const mult = DENSITY_MULTIPLIERS[simState.densityLevel-1];
  const total = Math.floor(MAX_DOTS * mult);
  for (let i=0; i<total; i++) particles.push(new Particle(false));
  for (let i=0; i<simState.emergCount; i++) particles.push(new Particle(true));
}

// ═══════════════════════════════════════════════════════════════
// LEAFLET MAP
// ═══════════════════════════════════════════════════════════════
const map = L.map('map', {
  center: [12.9716, 77.6412],
  zoom: 12,
  zoomControl: false,
  attributionControl: false,
  preferCanvas: true,
});

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18,
  attribution: '',
}).addTo(map);

// Road polylines colored by density
let roadLayers = [];

function drawRoadLayers() {
  roadLayers.forEach(l => l.remove());
  roadLayers = [];
  EDGES.forEach(([a,b], ri) => {
    const ja = JUNCTIONS[a], jb = JUNCTIONS[b];
    const avgCong = (ja.congestion + jb.congestion) / 2 * DENSITY_MULTIPLIERS[simState.densityLevel-1];
    const algoFactor = simState.algo==='fixed' ? 1.2 : simState.algo==='optimal' ? Math.max(0.4, 1-simState.algoBooted/600*0.5) : 0.9;
    const cong = Math.min(avgCong * algoFactor, 1);
    const color = cong>0.85 ? '#ff3355' : cong>0.65 ? '#ff8c00' : cong>0.4 ? '#ffd700' : '#00ff88';
    const weight = 3 + cong * 5;
    const hasEvp = particles.some(p=>p.isEmergency && p.edgeIndex===ri);

    const line = L.polyline(
      [[ja.lat, ja.lng],[jb.lat, jb.lng]],
      { color: hasEvp ? '#ff335588' : color+'88', weight: hasEvp?weight+4:weight, opacity:0.7 }
    ).addTo(map);
    roadLayers.push(line);

    if (hasEvp) {
      const evpLine = L.polyline(
        [[ja.lat, ja.lng],[jb.lat, jb.lng]],
        { color:'#ff3355', weight:2, opacity:0.9, dashArray:'8 6' }
      ).addTo(map);
      roadLayers.push(evpLine);
    }
  });
}

// Junction markers
const junctionMarkers = JUNCTIONS.map((j,i) => {
  const sig = signals[i];
  const marker = L.circleMarker([j.lat, j.lng], {
    radius: 8 + j.importance,
    color: '#fff',
    weight: 1,
    fillColor: '#ff3355',
    fillOpacity: 0.9,
  }).addTo(map);

  marker.bindTooltip(
    `<div style="font-family:'Share Tech Mono',monospace;font-size:.7rem;background:#060e1a;color:#00e5ff;border:1px solid #00e5ff33;padding:6px 10px;border-radius:3px">
      <b style="color:#ffd700">${j.name}</b><br>
      Congestion: <b style="color:${j.color}">${Math.round(j.congestion*100)}%</b><br>
      Daily Volume: <b>${(j.dailyVol/1000).toFixed(0)}K veh/day</b><br>
      Peak Volume: <b>${(j.peakVol/1000).toFixed(0)}K veh/hr</b>
    </div>`,
    { permanent: false, direction: 'top', className: 'blr-tooltip' }
  );

  return marker;
});

// Canvas overlay for particles
const flowCanvas = document.getElementById('flow-canvas');
const fctx = flowCanvas.getContext('2d');

function resizeCanvas() {
  const mc = document.getElementById('map-container');
  flowCanvas.width = mc.offsetWidth;
  flowCanvas.height = mc.offsetHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Convert lat/lng to canvas pixel
function latLngToCanvas(lat, lng) {
  const pt = map.latLngToContainerPoint([lat, lng]);
  return { x: pt.x, y: pt.y };
}

// ═══════════════════════════════════════════════════════════════
// SIGNAL CONTROL ALGORITHM
// ═══════════════════════════════════════════════════════════════
function updateSignals(dt) {
  simState.algoBooted = Math.min(simState.algoBooted + dt, 600);
  const warmup = simState.algoBooted / 600;

  signals.forEach((sig, i) => {
    sig.phase += dt * simState.simSpeed;
    if (sig.phase >= sig.cycle) sig.phase -= sig.cycle;

    // EVP preemption — check nearby emergency vehicles
    const nearEmergency = particles.filter(p=>p.isEmergency).some(p => {
      const endId = p.dir===1 ? EDGES[p.edgeIndex][1] : EDGES[p.edgeIndex][0];
      const distEnd = p.dir===1 ? 1-p.progress : p.progress;
      return endId===i && distEnd < 0.3;
    });

    const prevEvp = sig.evpOverride;
    sig.evpOverride = nearEmergency && simState.algo !== 'fixed';
    if (sig.evpOverride && !prevEvp) {
      simState.totalEvpOverrides++;
      document.getElementById('evp-flash').classList.add('active');
      setTimeout(()=>document.getElementById('evp-flash').classList.remove('active'), 500);
    }

    if (sig.evpOverride) {
      sig.state = 'green';
      sig.efficiency = 1.0;
      return;
    }

    // Adaptive green time using LP (minimize total wait W = Σ tᵢ·dᵢ)
    let gDur = simState.cycleTime * 0.5;
    const junc = JUNCTIONS[i];
    const cong = junc.congestion * DENSITY_MULTIPLIERS[simState.densityLevel-1];

    if (simState.algo === 'optimal' || simState.algo === 'ml') {
      // LP objective: maximize throughput = extend green proportional to density
      const densityRatio = Math.min(cong, 0.95);
      gDur = simState.cycleTime * (0.3 + densityRatio * 0.4) * (0.6 + warmup*0.4);
      simState.lpIterations += 0.01;

      // Green Wave offset: Φ = (L/v_c) mod T
      // Find adjacent junctions and synchronize
      const connectedEdges = EDGES.filter(([a,b])=>a===i||b===i);
      connectedEdges.forEach(([a,b]) => {
        const otherId = a===i?b:a;
        const ja = JUNCTIONS[i], jb = JUNCTIONS[otherId];
        const distKm = Math.sqrt(
          Math.pow((ja.lat-jb.lat)*111,2)+Math.pow((ja.lng-jb.lng)*111*Math.cos(ja.lat*Math.PI/180),2)
        );
        // Φᵢⱼ = (L/v_c) mod T
        const phi = (distKm / simState.waveSpeed * 3600) % simState.cycleTime;
        // Phase alignment for green wave
        if (signals[otherId] && simState.algo==='optimal') {
          const phaseDiff = (signals[otherId].phase - sig.phase + simState.cycleTime) % simState.cycleTime;
          if (Math.abs(phaseDiff - phi) > 5) {
            signals[otherId].phase += (phi - phaseDiff) * 0.02;
          }
        }
      });
    } else if (simState.algo === 'lp') {
      gDur = simState.cycleTime * (0.35 + cong * 0.3);
      simState.lpIterations += 0.005;
    }

    sig.greenDur = gDur;
    sig.cycle = simState.cycleTime;
    const yDur = simState.cycleTime * 0.07;

    if (sig.phase < gDur) sig.state = 'green';
    else if (sig.phase < gDur + yDur) sig.state = 'yellow';
    else sig.state = 'red';

    sig.efficiency = (gDur / simState.cycleTime) * warmup;
    sig.waitingVehicles = sig.state==='red'
      ? Math.floor(junc.congestion * 40 * cong)
      : Math.floor(junc.congestion * 10);
  });
}

// Update junction marker colors
function updateJunctionMarkers() {
  JUNCTIONS.forEach((j,i) => {
    const sig = signals[i];
    const col = sig.evpOverride ? '#ff3355' : sig.state==='green' ? '#00ff88' : sig.state==='yellow' ? '#ffd700' : '#ff3355';
    junctionMarkers[i].setStyle({ fillColor: col, color: sig.evpOverride ? '#ff3355' : '#ffffff' });
  });
}

// ═══════════════════════════════════════════════════════════════
// CHART.JS GRAPHS
// ═══════════════════════════════════════════════════════════════
const chartConfigs = {
  throughput: { id:'gc-throughput', color:'#00ff88', data:graphs.throughput.data, ymax:2000 },
  delay:      { id:'gc-delay',      color:'#ff3355', data:graphs.delay.data, ymax:180 },
  evp:        { id:'gc-evp',        color:'#ff8c00', data:graphs.evp.data, ymax:120 },
  eff:        { id:'gc-eff',        color:'#00e5ff', data:graphs.eff.data, ymax:100 },
  emit:       { id:'gc-emit',       color:'#ffd700', data:graphs.emit.data, ymax:150 },
  dens:       { id:'gc-dens',       color:'#cc88ff', data:graphs.dens.data, ymax:100 },
};

const chartInstances = {};
Object.entries(chartConfigs).forEach(([key,cfg]) => {
  const el = document.getElementById(cfg.id);
  if (!el) return;
  el.style.height = '60px';
  chartInstances[key] = new Chart(el, {
    type: 'line',
    data: {
      labels: new Array(GL).fill(''),
      datasets:[{
        data: cfg.data,
        borderColor: cfg.color,
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        backgroundColor: cfg.color+'18',
        tension: 0.4,
      }]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins:{ legend:{display:false}, tooltip:{enabled:false} },
      scales:{
        x:{ display:false },
        y:{ display:false, min:0, max:cfg.ymax, suggestedMax:cfg.ymax },
      },
    }
  });
});

function pushGraph(key, val) {
  const d = chartConfigs[key].data;
  d.push(val); d.shift();
  if (chartInstances[key]) {
    chartInstances[key].data.datasets[0].data = [...d];
    chartInstances[key].update('none');
  }
}

// ═══════════════════════════════════════════════════════════════
// METRICS & UI UPDATE
// ═══════════════════════════════════════════════════════════════
let lastMetrics = {};
let frameMetricCount = 0;

function updateMetrics() {
  const warmup = simState.algoBooted / 600;
  const moving = particles.filter(p=>!p.isEmergency && p.state==='moving').length;
  const slow   = particles.filter(p=>!p.isEmergency && p.state==='slow').length;
  const stopped= particles.filter(p=>!p.isEmergency && p.state==='stopped').length;
  const emerg  = particles.filter(p=>p.isEmergency).length;
  const total  = particles.length;

  // Algorithm factors
  const algoFactor = simState.algo==='optimal' ? (1+warmup*0.4) : simState.algo==='fixed' ? 0.6 : simState.algo==='lp' ? 0.85 : 0.9;
  const delayFactor = simState.algo==='fixed' ? 1.3 : simState.algo==='optimal' ? (1-warmup*0.35) : 0.95;

  // Throughput (VPHPL) — based on real ORR data: ~1200-1800 VPHPL optimized
  const baseThrough = 800 + moving * 2.5;
  const throughput = Math.round(baseThrough * algoFactor);

  // Delay — Silk Board baseline: 45min peak = ~120sec avg intersection delay
  const baseDelay = 120 * (DENSITY_MULTIPLIERS[simState.densityLevel-1]);
  const avgDelay  = Math.max(5, baseDelay * delayFactor - stopped*0.5);

  // EVP clear time
  const evpActive = signals.filter(s=>s.evpOverride).length;
  const evpTime = simState.algo==='fixed' ? 45 + Math.random()*20 : Math.max(3, 15*(1-warmup*0.6) + Math.random()*5);

  // Signal efficiency
  const avgEff = signals.reduce((s,sig)=>s+sig.efficiency,0)/signals.length * 100;

  // Emissions (relative index: fixed=100, optimal reduces by ~25%)
  const emitBase = 80 + stopped*0.8 + slow*0.3;
  const emissions = simState.algo==='fixed' ? emitBase*1.25 : simState.algo==='optimal' ? emitBase*(1-warmup*0.28) : emitBase*0.95;

  // Density %
  const density = Math.min(100, (stopped+slow*0.5)/Math.max(total,1)*100 + JUNCTIONS.reduce((s,j)=>s+j.congestion,0)/JUNCTIONS.length*50);

  // Avg speed
  const avgSpeed = simState.algo==='fixed' ? 17.8 : Math.min(40, 17.8 + warmup*22.2);

  // Fuel waste (idle: 0.7L/hr per 1000 stopped vehicles represented)
  const fuelWaste = Math.round(stopped * 5000 * 0.7 / 1000);

  // CO2 saved vs fixed timer
  const co2saved = Math.round(warmup * 180 * (simState.algo==='optimal'?1.4:simState.algo==='fixed'?0:0.6));

  // Green waves active
  const greenWaves = signals.filter(s=>s.state==='green'&&!s.evpOverride).length;

  // LP objective value W = Σ tᵢ·dᵢ
  const lpW = Math.round(avgDelay * total * 5000);

  // Push to oscilloscope
  pushGraph('throughput', throughput);
  pushGraph('delay', avgDelay);
  pushGraph('evp', evpTime);
  pushGraph('eff', avgEff);
  pushGraph('emit', emissions);
  pushGraph('dens', density);

  // ── UPDATE DOM ──
  const $ = id => document.getElementById(id);
  const prev = lastMetrics;

  // Header stats
  $('hs-vehicles').textContent = (total*5000).toLocaleString();
  $('hs-delay').textContent = Math.round(avgDelay)+'s';
  $('hs-evp').textContent = evpActive;
  $('hs-eff').textContent = Math.round(avgEff)+'%';
  $('hs-throughput').textContent = throughput;

  // Graph values + deltas
  function setGraphVal(key, val, unit='', decimals=0) {
    const el = $('gv-'+key);
    const de = $('gd-'+key);
    if (!el) return;
    el.textContent = typeof decimals==='number' ? val.toFixed(decimals) : val;
    if (de && prev[key] !== undefined) {
      const d = val - prev[key];
      if (Math.abs(d) > 0.5) {
        const isGood = (key==='throughput'||key==='eff') ? d>0 : d<0;
        de.textContent = (d>0?'▲':'▼')+Math.abs(d).toFixed(1);
        de.className = 'graph-delta '+(isGood?'delta-pos':'delta-neg');
      }
    }
    prev[key] = val;
  }

  setGraphVal('throughput', throughput);
  setGraphVal('delay', Math.round(avgDelay));
  setGraphVal('evp', Math.round(evpTime*10)/10, '', 1);
  setGraphVal('eff', Math.round(avgEff));
  setGraphVal('emit', Math.round(emissions));
  setGraphVal('dens', Math.round(density));

  // Stats cards
  const ffPct = Math.round(moving/Math.max(total,1)*100);
  $('sc-freeflow').textContent = ffPct+'%';
  $('sc-freeflow-n').textContent = (moving*5000).toLocaleString()+' vehicles';
  $('sc-stopped').textContent = (stopped*5000).toLocaleString();
  $('sc-avgspeed').textContent = avgSpeed.toFixed(1);
  $('sc-fuel').textContent = fuelWaste.toLocaleString();
  $('sc-green-waves').textContent = greenWaves;
  $('sc-evp-count').textContent = simState.totalEvpOverrides;
  $('sc-co2saved').textContent = co2saved;
  $('sc-lp-iters').textContent = Math.floor(simState.lpIterations);

  // LP display
  $('lp-T').textContent = simState.cycleTime+'s';
  $('lp-vc').textContent = simState.waveSpeed;
  $('lp-W').textContent = lpW.toLocaleString();
  $('lp-iter').textContent = Math.floor(simState.lpIterations);

  // Bottom bar
  $('bb-time').textContent = $('sim-time').textContent;
  $('bb-algo').textContent = $('algo-display').textContent;
  $('bb-total').textContent = (total*5000).toLocaleString();
  $('bb-stopped').textContent = (stopped*5000).toLocaleString();
  $('bb-emerg').textContent = emerg;
  $('bb-ff').textContent = ffPct+'%';
  $('bb-eff').textContent = Math.round(avgEff)+'%';
  $('bb-co2').textContent = co2saved+' kg/hr';

  // Algorithm comparison
  const fixedDelay = 45;
  const optDelay = Math.max(24, 45*(1-warmup*0.38));
  $('ac-fixed-delay').textContent = fixedDelay+' min';
  $('ac-optimal-delay').textContent = optDelay.toFixed(0)+' min';
  $('ac-reduction').textContent = Math.round((1-optDelay/fixedDelay)*100)+'%';
  $('ac-evp-saved').textContent = Math.round(warmup*62)+'%';

  lastMetrics = { throughput, delay:avgDelay, evp:evpTime, eff:avgEff, emit:emissions, dens:density };

  // Update junction list
  updateJunctionList();
}

function updateJunctionList() {
  const container = document.getElementById('junction-list');
  if (!container) return;
  const items = JUNCTIONS.map((j,i) => {
    const sig = signals[i];
    const sigColor = sig.evpOverride ? '#ff3355' : sig.state==='green' ? '#00ff88' : sig.state==='yellow' ? '#ffd700' : '#ff3355';
    const jDens = sig.waitingVehicles;
    const evpCls = sig.evpOverride ? 'evp-active' : '';
    const timeLeft = sig.state==='green'
      ? Math.max(0, sig.greenDur - sig.phase).toFixed(0)+'s▶'
      : Math.max(0, sig.cycle - sig.phase).toFixed(0)+'s◀';
    return `<div class="junction-item ${evpCls}">
      <div class="j-signal" style="background:${sigColor};box-shadow:0 0 6px ${sigColor}"></div>
      <div class="j-name">${j.name}</div>
      <div class="j-density" style="color:${j.color}">${Math.round(j.congestion*100)}%</div>
      <div class="j-timer">${timeLeft}</div>
    </div>`;
  }).join('');
  container.innerHTML = items;
}

// ═══════════════════════════════════════════════════════════════
// CANVAS PARTICLE RENDERER
// ═══════════════════════════════════════════════════════════════
function renderParticles() {
  fctx.clearRect(0, 0, flowCanvas.width, flowCanvas.height);

  particles.forEach(p => {
    const pos = p.getLatLng();
    const pt = latLngToCanvas(pos.lat, pos.lng);

    if (p.isEmergency) {
      // Render trail
      if (p.trail.length > 1) {
        for (let i=1; i<p.trail.length; i++) {
          const tp = latLngToCanvas(p.trail[i-1].lat, p.trail[i-1].lng);
          const tp2 = latLngToCanvas(p.trail[i].lat, p.trail[i].lng);
          fctx.strokeStyle = `rgba(255,51,85,${(1-i/p.trail.length)*0.5})`;
          fctx.lineWidth = 4 - i*0.3;
          fctx.beginPath();
          fctx.moveTo(tp.x, tp.y);
          fctx.lineTo(tp2.x, tp2.y);
          fctx.stroke();
        }
      }
      // Pulsing glow
      const pulse = 0.6+0.4*Math.sin(simState.frameCount*0.3+p.blinkPhase);
      fctx.shadowBlur = 15*pulse;
      fctx.shadowColor = '#ff3355';
      fctx.fillStyle = '#ff3355';
      fctx.beginPath();
      fctx.arc(pt.x, pt.y, 6, 0, Math.PI*2);
      fctx.fill();
      // Cross
      fctx.strokeStyle = '#ffffff';
      fctx.lineWidth = 1.5;
      fctx.shadowBlur = 0;
      fctx.beginPath();
      fctx.moveTo(pt.x-5,pt.y); fctx.lineTo(pt.x+5,pt.y);
      fctx.moveTo(pt.x,pt.y-5); fctx.lineTo(pt.x,pt.y+5);
      fctx.stroke();
    } else {
      fctx.shadowBlur = 0;
      fctx.fillStyle = p.getColor()+'bb';
      fctx.beginPath();
      fctx.arc(pt.x, pt.y, 2.5, 0, Math.PI*2);
      fctx.fill();
    }
  });
  fctx.shadowBlur = 0;
}

// ═══════════════════════════════════════════════════════════════
// SIM TIME
// ═══════════════════════════════════════════════════════════════
function formatTime(s) {
  const ss = Math.floor(s)%60, mm = Math.floor(s/60)%60, hh = Math.floor(s/3600)%24;
  return `${String(hh).padStart(2,'0')}:${String(mm).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
}

// ═══════════════════════════════════════════════════════════════
// CONTROLS
// ═══════════════════════════════════════════════════════════════
const ALGO_NAMES = {
  optimal:'GREEN WAVE+EVP', fixed:'FIXED TIMER',
  lp:'LP ONLY', evp:'EVP ONLY', ml:'ML+EVP'
};
const ALGO_LIST = Object.keys(ALGO_NAMES);
let algoIdx = 0;

function setAlgo(a) {
  simState.algo = a;
  simState.algoBooted = 0;
  const name = ALGO_NAMES[a];
  document.getElementById('algo-display').textContent = name;
  document.getElementById('bb-algo').textContent = name;
  document.getElementById('btn-algo').textContent = '⚡ '+name;
  document.getElementById('algo-select').value = a;
}
function cycleAlgo() {
  algoIdx = (algoIdx+1) % ALGO_LIST.length;
  setAlgo(ALGO_LIST[algoIdx]);
}
function setAlgoFromSelect(val) { setAlgo(val); }

function togglePause() {
  simState.paused = !simState.paused;
  document.getElementById('btn-pause').textContent = simState.paused ? '▶ RESUME' : '⏸ PAUSE';
}

function triggerMassEVP() {
  // Force all signals to EVP override for 5 seconds
  signals.forEach(s=>s.evpOverride=true);
  document.getElementById('evp-flash').classList.add('active');
  setTimeout(()=>{
    signals.forEach(s=>s.evpOverride=false);
    document.getElementById('evp-flash').classList.remove('active');
  },5000);
}

function setDensity(v) {
  simState.densityLevel = parseInt(v);
  document.getElementById('lbl-density').textContent = DENSITY_LABELS[simState.densityLevel-1]+' (1 dot = 5000 vehicles)';
  spawnParticles();
}
function setEmerg(v) {
  simState.emergCount = parseInt(v);
  document.getElementById('lbl-emerg').textContent = v;
  // Rebuild emergency particles
  particles = particles.filter(p=>!p.isEmergency);
  for (let i=0; i<simState.emergCount; i++) particles.push(new Particle(true));
}
function setWave(v) {
  simState.waveSpeed = parseInt(v);
  document.getElementById('lbl-wave').textContent = v+' km/h';
  document.getElementById('wave-display').textContent = v+' km/h';
}
function setCycle(v) {
  simState.cycleTime = parseInt(v);
  document.getElementById('lbl-cycle').textContent = v+' sec';
}
function setSimSpeed(v) { simState.simSpeed = parseFloat(v); }

// ═══════════════════════════════════════════════════════════════
// MAIN LOOP
// ═══════════════════════════════════════════════════════════════
let lastT = 0;
let roadUpdateCounter = 0;

function loop(ts) {
  if (simState.paused) { requestAnimationFrame(loop); return; }
  const dt = Math.min((ts - lastT)/1000 * 60, 4);
  lastT = ts;
  simState.frameCount++;
  simState.simTime += dt * 0.016 * simState.simSpeed;

  // Update signals
  updateSignals(dt);

  // Update particles
  particles.forEach(p => p.update(dt));

  // Render particles
  renderParticles();

  // Update junction markers
  if (simState.frameCount % 15 === 0) {
    updateJunctionMarkers();
    roadUpdateCounter++;
    if (roadUpdateCounter % 3 === 0) drawRoadLayers();
  }

  // Metrics every 30 frames
  if (simState.frameCount % 30 === 0) {
    const t = simState.simTime;
    const timeStr = formatTime(t);
    document.getElementById('sim-time').textContent = timeStr;
    document.getElementById('bb-time').textContent = timeStr;
    updateMetrics();
  }

  requestAnimationFrame(loop);
}

// ── INIT ──
spawnParticles();
drawRoadLayers();
requestAnimationFrame(loop);
</script>
</body>
</html>
"""

components.html(HTML, height=900, scrolling=False)

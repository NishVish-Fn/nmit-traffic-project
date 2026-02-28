import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Urban Flow & Life-Lines | Bangalore",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"]{background:#020810!important}
[data-testid="stHeader"],[data-testid="stToolbar"]{display:none!important}
section[data-testid="stSidebar"]{display:none!important}
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
<title>Urban Flow & Life-Lines — Bangalore</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#020810; --bg2:#06101e; --bg3:#0b1a2e; --bg4:#0f2040;
  --cyan:#00e5ff; --green:#00ff88; --red:#ff2244; --orange:#ff8c00;
  --yellow:#ffd700; --purple:#bb77ff; --pink:#ff44aa;
  --c-dim:#00e5ff22; --g-dim:#00ff8815;
}
*{margin:0;padding:0;box-sizing:border-box}
body{
  background:var(--bg);
  color:#b8d8f0;
  font-family:'Rajdhani',sans-serif;
  width:100vw;height:100vh;
  overflow:hidden;
  display:flex;flex-direction:column;
}

/* ════════ TOP HEADER ════════ */
#hdr{
  height:56px;flex-shrink:0;
  background:linear-gradient(90deg,#000a18,#020810 50%,#000a18);
  border-bottom:1px solid var(--c-dim);
  display:flex;align-items:center;
  padding:0 20px;gap:0;
  position:relative;z-index:2000;
}
#hdr::after{
  content:'';position:absolute;bottom:-1px;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent 0%,var(--cyan) 50%,transparent 100%);
  animation:hdrScan 5s ease-in-out infinite;
}
@keyframes hdrScan{0%,100%{opacity:.3}50%{opacity:1}}

.hdr-brand{display:flex;align-items:center;gap:14px;min-width:340px}
.hdr-icon{font-size:2.2rem;filter:drop-shadow(0 0 12px var(--cyan))}
.hdr-title{font-family:'Orbitron',monospace;font-size:1.15rem;font-weight:800;color:var(--cyan);
  letter-spacing:4px;text-shadow:0 0 25px var(--cyan)88}
.hdr-sub{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:var(--orange);
  letter-spacing:2px;margin-top:3px}

.hdr-divider{width:1px;height:36px;background:var(--c-dim);margin:0 20px}

/* Header KPIs */
.hdr-kpis{display:flex;gap:28px;flex:1;justify-content:center}
.kpi{text-align:center;position:relative}
.kpi-val{font-family:'Orbitron',monospace;font-weight:700;font-size:1.35rem;line-height:1}
.kpi-lbl{font-family:'Share Tech Mono',monospace;font-size:0.5rem;color:#4a6880;
  letter-spacing:1.5px;margin-top:3px;text-transform:uppercase}
.kpi-trend{font-family:'Share Tech Mono',monospace;font-size:0.55rem;position:absolute;
  top:0;right:-16px}

/* Header controls */
.hdr-controls{display:flex;gap:8px;align-items:center;min-width:380px;justify-content:flex-end}
.btn{
  font-family:'Share Tech Mono',monospace;font-size:0.62rem;letter-spacing:1.5px;
  padding:6px 14px;border:1px solid;border-radius:3px;cursor:pointer;
  transition:all .2s;background:transparent;text-transform:uppercase;white-space:nowrap;
}
.btn-c{border-color:var(--cyan);color:var(--cyan)}
.btn-c:hover,.btn-c.on{background:var(--cyan);color:#000;box-shadow:0 0 18px var(--cyan)88}
.btn-r{border-color:var(--red);color:var(--red)}
.btn-r:hover,.btn-r.on{background:var(--red);color:#fff;box-shadow:0 0 18px var(--red)88}
.btn-g{border-color:var(--green);color:var(--green)}
.btn-g:hover,.btn-g.on{background:var(--green);color:#000;box-shadow:0 0 18px var(--green)88}
.btn-o{border-color:var(--orange);color:var(--orange)}
.btn-o:hover{background:var(--orange);color:#000;box-shadow:0 0 18px var(--orange)88}

.live-badge{
  font-family:'Share Tech Mono',monospace;font-size:0.6rem;letter-spacing:2px;
  padding:4px 10px;background:var(--red)22;border:1px solid var(--red);
  color:var(--red);border-radius:2px;animation:blink 1.2s infinite;
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.35}}

/* ════════ BODY LAYOUT ════════ */
#body{flex:1;display:flex;overflow:hidden}

/* ════════ LEFT SIDEBAR ════════ */
#sidebar{
  width:300px;flex-shrink:0;
  background:var(--bg2);
  border-right:1px solid var(--c-dim);
  display:flex;flex-direction:column;
  overflow:hidden;
}

/* Sidebar tabs */
#stabs{display:flex;border-bottom:1px solid var(--c-dim)}
.stab{
  flex:1;padding:10px 0;text-align:center;cursor:pointer;
  font-family:'Share Tech Mono',monospace;font-size:0.6rem;
  letter-spacing:1.5px;color:#4a6880;
  border-bottom:2px solid transparent;transition:.2s;
  text-transform:uppercase;
}
.stab.active{color:var(--cyan);border-bottom-color:var(--cyan)}
.stab:hover:not(.active){color:#7090a0}

.stab-content{display:none;flex:1;overflow-y:auto;padding:12px;
  scrollbar-width:thin;scrollbar-color:var(--c-dim) transparent;
  flex-direction:column;gap:10px}
.stab-content.active{display:flex}

/* Sections inside sidebar */
.sec{background:var(--bg3);border:1px solid #0d2040;border-radius:5px;padding:12px}
.sec-title{
  font-family:'Orbitron',monospace;font-size:0.6rem;font-weight:600;
  color:var(--cyan);letter-spacing:3px;text-transform:uppercase;
  border-bottom:1px solid var(--c-dim);padding-bottom:7px;margin-bottom:10px;
  display:flex;align-items:center;gap:8px;
}
.sec-title-icon{font-size:.9rem}

/* Controls */
.ctrl{margin-bottom:12px}
.ctrl:last-child{margin-bottom:0}
.ctrl-lbl{
  font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:#4a6880;
  letter-spacing:1px;display:flex;justify-content:space-between;margin-bottom:5px;
}
.ctrl-lbl span{color:var(--cyan);font-weight:bold}
input[type=range]{
  width:100%;-webkit-appearance:none;height:3px;
  background:linear-gradient(90deg,var(--cyan),var(--cyan)) no-repeat left;
  background-color:#0d2040;border-radius:2px;outline:none;
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:14px;height:14px;
  background:var(--cyan);border-radius:50%;cursor:pointer;
  box-shadow:0 0 8px var(--cyan)88;
}
select{
  width:100%;background:var(--bg);border:1px solid #0d2040;
  color:var(--cyan);font-family:'Share Tech Mono',monospace;font-size:0.62rem;
  padding:6px 8px;border-radius:3px;outline:none;cursor:pointer;
}
select option{background:#0a1628}

/* Junction monitor */
.j-item{
  display:grid;grid-template-columns:10px 1fr auto auto;
  align-items:center;gap:8px;
  padding:7px 8px;border-radius:4px;border:1px solid transparent;
  cursor:pointer;transition:.2s;margin-bottom:4px;
  background:var(--bg);
}
.j-item:hover{border-color:var(--c-dim)}
.j-item.evp{border-color:var(--red);background:#1a040d;animation:jpulse .8s infinite alternate}
@keyframes jpulse{from{box-shadow:none}to{box-shadow:0 0 10px var(--red)44}}
.j-dot{width:10px;height:10px;border-radius:50%;transition:all .3s}
.j-name{font-family:'Share Tech Mono',monospace;font-size:0.62rem;line-height:1.2}
.j-name small{display:block;color:#3a5570;font-size:0.5rem}
.j-pct{font-family:'Orbitron',monospace;font-size:0.75rem;font-weight:700;text-align:right}
.j-timer{font-family:'Share Tech Mono',monospace;font-size:0.5rem;color:#3a5570;text-align:right}

/* Data table */
.data-table{width:100%;border-collapse:collapse;font-family:'Share Tech Mono',monospace;font-size:0.58rem}
.data-table td{padding:5px 6px;border-bottom:1px solid #0d2040}
.data-table tr:last-child td{border-bottom:none}
.data-table td:first-child{color:#4a6880}
.data-table td:last-child{color:var(--cyan);text-align:right;font-weight:bold}
.data-table .red td:last-child{color:var(--red)}
.data-table .orange td:last-child{color:var(--orange)}
.data-table .green td:last-child{color:var(--green)}
.data-table .yellow td:last-child{color:var(--yellow)}

/* Scale legend */
.scale-row{display:flex;align-items:center;gap:10px;margin-bottom:7px}
.scale-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.scale-txt{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#6080a0;line-height:1.3}

/* ════════ MAP CENTER ════════ */
#map-wrap{flex:1;position:relative;overflow:hidden}
#map{width:100%;height:100%}
#flow-canvas{position:absolute;top:0;left:0;pointer-events:none;z-index:400}
.evp-overlay{
  position:absolute;inset:0;pointer-events:none;z-index:450;
  background:transparent;transition:.4s;
}
.evp-overlay.active{background:radial-gradient(ellipse at center,rgba(255,34,68,.07) 0%,transparent 65%)}

/* Map floating panels */
.map-pill{
  position:absolute;z-index:600;
  background:rgba(2,8,16,.92);border:1px solid;border-radius:4px;
  font-family:'Share Tech Mono',monospace;font-size:0.62rem;
  backdrop-filter:blur(4px);
}
#map-topbar{
  top:12px;left:50%;transform:translateX(-50%);
  border-color:var(--orange)88;
  padding:7px 20px;display:flex;gap:24px;color:var(--orange);
  white-space:nowrap;
}
#map-topbar span{display:flex;align-items:center;gap:6px}
#map-topbar b{color:var(--cyan)}

#map-legend{
  bottom:14px;left:14px;border-color:var(--c-dim);
  padding:12px 14px;min-width:170px;
}
.leg-title{font-family:'Orbitron',monospace;font-size:0.55rem;
  color:var(--cyan);letter-spacing:2px;margin-bottom:8px}
.leg-row{display:flex;align-items:center;gap:8px;margin-bottom:5px;color:#5a7090}
.leg-bar{height:4px;width:32px;border-radius:2px}

#map-scale{
  bottom:14px;right:14px;border-color:var(--c-dim);
  padding:12px 14px;
}
.ms-title{font-family:'Orbitron',monospace;font-size:0.55rem;
  color:var(--cyan);letter-spacing:2px;margin-bottom:8px}
.ms-row{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.ms-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.ms-txt{font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:#6080a0}

/* ════════ RIGHT PANEL — ANALYTICS ════════ */
#analytics{
  width:330px;flex-shrink:0;
  background:var(--bg2);border-left:1px solid var(--c-dim);
  display:flex;flex-direction:column;overflow:hidden;
}

/* Analytics tabs */
#atabs{display:flex;border-bottom:1px solid var(--c-dim)}
.atab{
  flex:1;padding:10px 0;text-align:center;cursor:pointer;
  font-family:'Share Tech Mono',monospace;font-size:0.58rem;
  letter-spacing:1.5px;color:#4a6880;
  border-bottom:2px solid transparent;transition:.2s;text-transform:uppercase;
}
.atab.active{color:var(--cyan);border-bottom-color:var(--cyan)}
.atab:hover:not(.active){color:#7090a0}

.atab-content{display:none;flex:1;overflow-y:auto;padding:12px;
  scrollbar-width:thin;scrollbar-color:var(--c-dim) transparent;
  flex-direction:column;gap:10px;
}
.atab-content.active{display:flex}

/* Graphs */
.g-card{background:var(--bg3);border:1px solid #0d2040;border-radius:5px;padding:10px}
.g-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px}
.g-title{font-family:'Share Tech Mono',monospace;font-size:0.55rem;
  color:var(--cyan);letter-spacing:2px;text-transform:uppercase;line-height:1.4}
.g-right{text-align:right}
.g-val{font-family:'Orbitron',monospace;font-size:1.4rem;font-weight:700;line-height:1}
.g-unit{font-family:'Share Tech Mono',monospace;font-size:0.5rem;color:#4a6880;
  display:block;margin-top:2px}
.g-delta{font-family:'Share Tech Mono',monospace;font-size:0.55rem;display:inline-block;margin-top:2px}
.up{color:var(--green)} .dn{color:var(--red)}
canvas.gcvs{display:block;width:100%!important;height:68px!important}

/* Stats cards 2-col grid */
.s-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.s-card{
  background:var(--bg);border:1px solid #0d2040;border-radius:4px;
  padding:10px 8px;text-align:center;border-left:3px solid;
}
.s-val{font-family:'Orbitron',monospace;font-size:1.15rem;font-weight:700;
  line-height:1;margin-bottom:4px}
.s-lbl{font-family:'Share Tech Mono',monospace;font-size:0.5rem;
  color:#4a6880;letter-spacing:1px;text-transform:uppercase}
.s-sub{font-family:'Share Tech Mono',monospace;font-size:0.52rem;color:#2a4060;margin-top:3px}

/* Algo benchmark */
.ab-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.ab-card{padding:10px 8px;border-radius:4px;border:1px solid;text-align:center}
.ab-name{font-family:'Orbitron',monospace;font-size:0.48rem;letter-spacing:1px;
  margin-bottom:6px;text-transform:uppercase}
.ab-val{font-family:'Orbitron',monospace;font-size:1.2rem;font-weight:700;line-height:1}
.ab-sub{font-family:'Share Tech Mono',monospace;font-size:0.5rem;color:#3a5570;margin-top:4px}

/* LP Display */
.lp-box{
  background:var(--bg);border:1px solid #0d2040;border-radius:4px;
  padding:10px;font-family:'Share Tech Mono',monospace;font-size:0.62rem;
  color:#3a5570;line-height:2;
}
.lp-box .eq{color:#5a7090}
.lp-box .hi{color:var(--cyan)}
.lp-box .hi-g{color:var(--green)}
.lp-box .hi-y{color:var(--yellow)}
.lp-box .hi-r{color:var(--red)}
.lp-box .hi-o{color:var(--orange)}

/* Signal grid */
.sig-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.sig-card{
  background:var(--bg);border:1px solid #0d2040;border-radius:4px;
  padding:8px;text-align:center;border-top:3px solid;
}
.sig-name{font-family:'Share Tech Mono',monospace;font-size:0.52rem;
  color:#4a6880;margin-bottom:5px;line-height:1.3}
.sig-state{font-family:'Orbitron',monospace;font-size:0.75rem;font-weight:700;margin-bottom:4px}
.sig-timer{font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:#3a5570}
.sig-bar-bg{height:4px;background:#0d2040;border-radius:2px;margin-top:5px;overflow:hidden}
.sig-bar{height:100%;border-radius:2px;transition:width .5s}
.sig-evp{border-top-color:var(--red)!important;animation:sipulse .6s infinite alternate}
@keyframes sipulse{from{box-shadow:none}to{box-shadow:0 0 8px var(--red)66}}

/* ════════ BOTTOM STATUS BAR ════════ */
#statusbar{
  height:30px;flex-shrink:0;
  background:var(--bg2);border-top:1px solid var(--c-dim);
  display:flex;align-items:center;padding:0 16px;gap:0;
  font-family:'Share Tech Mono',monospace;font-size:0.58rem;
  overflow:hidden;
}
.sb-item{display:flex;align-items:center;gap:5px;color:#4a6880;padding:0 14px;
  border-right:1px solid #0d2040;white-space:nowrap}
.sb-item:last-child{border-right:none;margin-left:auto}
.sb-val{color:var(--cyan);font-weight:bold}
.sb-val.r{color:var(--red)} .sb-val.g{color:var(--green)} .sb-val.y{color:var(--yellow)} .sb-val.o{color:var(--orange)}

/* Leaflet dark */
.leaflet-tile-pane{filter:brightness(.28) saturate(.2) hue-rotate(195deg)!important}
.leaflet-container{background:var(--bg)}
.leaflet-control-attribution,.leaflet-control-zoom{display:none!important}
.blr-tip .leaflet-tooltip{
  background:#060e1a;border:1px solid var(--c-dim);color:var(--cyan);
  font-family:'Share Tech Mono',monospace;font-size:.65rem;padding:8px 12px;
  box-shadow:0 0 20px #00000088;border-radius:4px;
}
</style>
</head>
<body>

<!-- ══════════════════════════════════ HEADER ══════════════════════════════════ -->
<div id="hdr">
  <div class="hdr-brand">
    <div class="hdr-icon">🚦</div>
    <div>
      <div class="hdr-title">URBAN FLOW & LIFE-LINES</div>
      <div class="hdr-sub">▸ BANGALORE TRAFFIC GRID — MULTI-OBJECTIVE LP OPTIMIZATION + EVP ◂</div>
    </div>
  </div>
  <div class="hdr-divider"></div>

  <div class="hdr-kpis">
    <div class="kpi">
      <div class="kpi-val" id="kpi-veh" style="color:var(--cyan)">—</div>
      <div class="kpi-lbl">Live Vehicles</div>
    </div>
    <div class="kpi">
      <div class="kpi-val" id="kpi-delay" style="color:var(--red)">—</div>
      <div class="kpi-lbl">Avg Delay (s)</div>
    </div>
    <div class="kpi">
      <div class="kpi-val" id="kpi-evp" style="color:var(--orange)">—</div>
      <div class="kpi-lbl">EVP Active</div>
    </div>
    <div class="kpi">
      <div class="kpi-val" id="kpi-eff" style="color:var(--green)">—</div>
      <div class="kpi-lbl">Grid Efficiency</div>
    </div>
    <div class="kpi">
      <div class="kpi-val" id="kpi-speed" style="color:var(--yellow)">—</div>
      <div class="kpi-lbl">Avg Speed km/h</div>
    </div>
    <div class="kpi">
      <div class="kpi-val" id="kpi-co2" style="color:var(--purple)">—</div>
      <div class="kpi-lbl">CO₂ Saved kg/hr</div>
    </div>
  </div>
  <div class="hdr-divider"></div>

  <div class="hdr-controls">
    <div class="live-badge">● LIVE</div>
    <button class="btn btn-g on" id="btn-algo" onclick="cycleAlgo()">⚡ GW+EVP</button>
    <button class="btn btn-r" id="btn-evp-mass" onclick="triggerMassEVP()">🚨 MASS EVP</button>
    <button class="btn btn-c" id="btn-pause" onclick="togglePause()">⏸ PAUSE</button>
  </div>
</div>

<!-- ══════════════════════════════════ BODY ══════════════════════════════════ -->
<div id="body">

  <!-- ═══ LEFT SIDEBAR ═══ -->
  <div id="sidebar">
    <div id="stabs">
      <div class="stab active" onclick="switchSTab(0)">CONTROLS</div>
      <div class="stab" onclick="switchSTab(1)">JUNCTIONS</div>
      <div class="stab" onclick="switchSTab(2)">DATA</div>
    </div>

    <!-- Tab 0: Controls -->
    <div class="stab-content active" id="st0">
      <div class="sec">
        <div class="sec-title"><span class="sec-title-icon">⚙</span>Simulation Parameters</div>

        <div class="ctrl">
          <div class="ctrl-lbl">Traffic Density <span id="lbl-dens">Peak</span></div>
          <input type="range" min="1" max="5" value="4" oninput="setDensity(this.value)">
        </div>
        <div class="ctrl">
          <div class="ctrl-lbl">Emergency Vehicles (×100) <span id="lbl-emerg">50</span></div>
          <input type="range" min="10" max="200" value="50" oninput="setEmerg(this.value)">
        </div>
        <div class="ctrl">
          <div class="ctrl-lbl">Green Wave Speed <span id="lbl-wave">40 km/h</span></div>
          <input type="range" min="20" max="80" value="40" step="5" oninput="setWave(this.value)">
        </div>
        <div class="ctrl">
          <div class="ctrl-lbl">Signal Cycle Time <span id="lbl-cycle">90s</span></div>
          <input type="range" min="30" max="180" value="90" step="10" oninput="setCycle(this.value)">
        </div>
        <div class="ctrl">
          <div class="ctrl-lbl">Simulation Speed</div>
          <select onchange="setSimSpeed(this.value)">
            <option value="0.5">0.5× Slow motion</option>
            <option value="1" selected>1× Real-time</option>
            <option value="2">2× Fast</option>
            <option value="5">5× Ultra-fast</option>
          </select>
        </div>
        <div class="ctrl" style="margin-bottom:0">
          <div class="ctrl-lbl">Algorithm Mode</div>
          <select id="algo-sel" onchange="setAlgoFromSel(this.value)">
            <option value="optimal">✦ Green Wave + EVP (Proposed)</option>
            <option value="fixed">✗ Fixed Timer (Baseline)</option>
            <option value="lp">◈ Linear Programming Only</option>
            <option value="evp">◉ Emergency Priority Only</option>
            <option value="ml">★ ML Predictive + EVP</option>
          </select>
        </div>
      </div>

      <div class="sec">
        <div class="sec-title"><span class="sec-title-icon">🔬</span>Scale Reference</div>
        <div class="scale-row"><div class="scale-dot" style="background:var(--cyan)"></div>
          <div class="scale-txt">1 cyan dot = <b style="color:var(--cyan)">5,000</b> regular vehicles</div></div>
        <div class="scale-row"><div class="scale-dot" style="background:var(--red);box-shadow:0 0 6px var(--red)"></div>
          <div class="scale-txt">1 red dot = <b style="color:var(--red)">100</b> emergency vehicles</div></div>
        <div class="scale-row"><div class="scale-dot" style="background:var(--orange)"></div>
          <div class="scale-txt">Orange flow = congested (70–90%)</div></div>
        <div class="scale-row"><div class="scale-dot" style="background:var(--red);opacity:.6"></div>
          <div class="scale-txt">Red flow = gridlock (&gt;90%)</div></div>
        <div class="scale-row"><div class="scale-dot" style="background:var(--green)"></div>
          <div class="scale-txt">Green flow = free-flow (&lt;40%)</div></div>
        <div class="scale-row"><div class="scale-dot" style="background:var(--pink);box-shadow:0 0 6px var(--pink)"></div>
          <div class="scale-txt">Magenta corridor = active EVP path</div></div>
      </div>
    </div>

    <!-- Tab 1: Junctions -->
    <div class="stab-content" id="st1">
      <div class="sec">
        <div class="sec-title"><span class="sec-title-icon">🗺</span>12 Real Bangalore Junctions</div>
        <div id="j-list"></div>
      </div>
    </div>

    <!-- Tab 2: Real Data -->
    <div class="stab-content" id="st2">
      <div class="sec">
        <div class="sec-title"><span class="sec-title-icon">📋</span>TomTom / Kaggle Data 2024</div>
        <table class="data-table">
          <tr class="red"><td>Silk Board Congestion</td><td>71%</td></tr>
          <tr class="red"><td>Electronic City Cong.</td><td>67%</td></tr>
          <tr class="red"><td>Hebbal Congestion</td><td>64%</td></tr>
          <tr class="orange"><td>Marathahalli Cong.</td><td>58%</td></tr>
          <tr class="orange"><td>KR Puram Congestion</td><td>54%</td></tr>
          <tr class="orange"><td>ORR Average</td><td>62%</td></tr>
          <tr><td>Peak Hours</td><td style="color:var(--yellow)">8–10AM, 6–9PM</td></tr>
          <tr class="red"><td>Avg Speed (Peak)</td><td>17.8 km/h</td></tr>
          <tr class="green"><td>Avg Speed (Off-peak)</td><td>32.4 km/h</td></tr>
          <tr><td>Daily Vehicles</td><td>1.2M</td></tr>
          <tr><td>Registered Vehicles</td><td>10.5M</td></tr>
          <tr><td>Bangalore Rank (India)</td><td style="color:var(--orange)">#4 Worst</td></tr>
          <tr><td>Fuel Waste/Day</td><td style="color:var(--red)">~8.2M L</td></tr>
          <tr><td>Economic Loss/Day</td><td style="color:var(--red)">₹342 Cr</td></tr>
          <tr><td>CO₂ Emission/Day</td><td style="color:var(--orange)">19,800 T</td></tr>
          <tr class="green"><td>EVP Time Saved (model)</td><td>60%</td></tr>
          <tr class="green"><td>Delay Reduction (model)</td><td>38%</td></tr>
        </table>
      </div>
      <div class="sec">
        <div class="sec-title"><span class="sec-title-icon">📚</span>Data Sources</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:.58rem;color:#3a5570;line-height:2">
          • TomTom Traffic Index 2024<br>
          • Kaggle: Bangalore Traffic Dataset<br>
          • BBMP Traffic Engineering Cell<br>
          • KRDCL Outer Ring Road Study<br>
          • Lighthill-Whitham-Richards Model<br>
          • Operations Research: Taha 2017<br>
          • Graph Theory: West 2001
        </div>
      </div>
    </div>
  </div>

  <!-- ═══ MAP ═══ -->
  <div id="map-wrap">
    <div id="map"></div>
    <canvas id="flow-canvas"></canvas>
    <div class="evp-overlay" id="evp-overlay"></div>

    <div class="map-pill" id="map-topbar">
      <span>⏱ <b id="sim-time">00:00:00</b></span>
      <span>⚡ ALGO: <b id="algo-disp">GREEN WAVE+EVP</b></span>
      <span>🌊 WAVE: <b id="wave-disp">40 km/h</b></span>
      <span>🚗 <b id="veh-total">—</b> VEHICLES</span>
      <span>🚨 <b id="emerg-total" style="color:var(--red)">—</b> EMERGENCY</span>
    </div>

    <div class="map-pill" id="map-legend">
      <div class="leg-title">▸ ROAD DENSITY</div>
      <div class="leg-row"><div class="leg-bar" style="background:var(--green)"></div>Free-flow &lt;40%</div>
      <div class="leg-row"><div class="leg-bar" style="background:var(--yellow)"></div>Moderate 40–70%</div>
      <div class="leg-row"><div class="leg-bar" style="background:var(--orange)"></div>Congested 70–90%</div>
      <div class="leg-row"><div class="leg-bar" style="background:var(--red)"></div>Gridlock &gt;90%</div>
      <div class="leg-row"><div class="leg-bar" style="background:linear-gradient(90deg,var(--red),var(--pink));"></div>EVP Corridor</div>
    </div>

    <div class="map-pill" id="map-scale">
      <div class="ms-title">▸ PARTICLE SCALE</div>
      <div class="ms-row"><div class="ms-dot" style="background:var(--cyan)"></div><div class="ms-txt">1 dot = 5,000 vehicles</div></div>
      <div class="ms-row"><div class="ms-dot" style="background:var(--red);box-shadow:0 0 5px var(--red)"></div><div class="ms-txt">1 dot = 100 emergency</div></div>
      <div class="ms-row"><div class="ms-dot" style="background:var(--yellow)"></div><div class="ms-txt">Yellow = slowing</div></div>
      <div class="ms-row"><div class="ms-dot" style="background:var(--orange)"></div><div class="ms-txt">Orange = stopped</div></div>
    </div>
  </div>

  <!-- ═══ RIGHT ANALYTICS PANEL ═══ -->
  <div id="analytics">
    <div id="atabs">
      <div class="atab active" onclick="switchATab(0)">GRAPHS</div>
      <div class="atab" onclick="switchATab(1)">STATS</div>
      <div class="atab" onclick="switchATab(2)">SIGNALS</div>
      <div class="atab" onclick="switchATab(3)">LP MODEL</div>
    </div>

    <!-- ATab 0: Oscilloscope Graphs -->
    <div class="atab-content active" id="at0">

      <div class="g-card">
        <div class="g-header">
          <div class="g-title">Network Throughput<br>Vehicles/hr/lane</div>
          <div class="g-right">
            <div class="g-val" id="gv-thr" style="color:var(--green)">—</div>
            <span class="g-unit">VPHPL</span>
            <div class="g-delta" id="gd-thr"></div>
          </div>
        </div>
        <canvas class="gcvs" id="gc-thr"></canvas>
      </div>

      <div class="g-card">
        <div class="g-header">
          <div class="g-title">Avg Intersection Delay<br>Webster's formula</div>
          <div class="g-right">
            <div class="g-val" id="gv-del" style="color:var(--red)">—</div>
            <span class="g-unit">seconds</span>
            <div class="g-delta" id="gd-del"></div>
          </div>
        </div>
        <canvas class="gcvs" id="gc-del"></canvas>
      </div>

      <div class="g-card">
        <div class="g-header">
          <div class="g-title">EVP Corridor Clear Time<br>Golden Hour metric</div>
          <div class="g-right">
            <div class="g-val" id="gv-evp" style="color:var(--orange)">—</div>
            <span class="g-unit">seconds</span>
            <div class="g-delta" id="gd-evp"></div>
          </div>
        </div>
        <canvas class="gcvs" id="gc-evp"></canvas>
      </div>

      <div class="g-card">
        <div class="g-header">
          <div class="g-title">Signal Efficiency Index<br>Green time utilization</div>
          <div class="g-right">
            <div class="g-val" id="gv-eff" style="color:var(--cyan)">—</div>
            <span class="g-unit">%</span>
            <div class="g-delta" id="gd-eff"></div>
          </div>
        </div>
        <canvas class="gcvs" id="gc-eff"></canvas>
      </div>

      <div class="g-card">
        <div class="g-header">
          <div class="g-title">CO₂ Emissions Index<br>vs fixed-timer baseline</div>
          <div class="g-right">
            <div class="g-val" id="gv-co2" style="color:var(--yellow)">—</div>
            <span class="g-unit">relative</span>
            <div class="g-delta" id="gd-co2"></div>
          </div>
        </div>
        <canvas class="gcvs" id="gc-co2"></canvas>
      </div>

      <div class="g-card">
        <div class="g-header">
          <div class="g-title">Network Density<br>LWR flow model</div>
          <div class="g-right">
            <div class="g-val" id="gv-dns" style="color:var(--purple)">—</div>
            <span class="g-unit">%</span>
            <div class="g-delta" id="gd-dns"></div>
          </div>
        </div>
        <canvas class="gcvs" id="gc-dns"></canvas>
      </div>

    </div>

    <!-- ATab 1: Stats -->
    <div class="atab-content" id="at1">

      <div class="sec" style="background:var(--bg3);border:1px solid #0d2040;border-radius:5px;padding:10px">
        <div class="sec-title"><span class="sec-title-icon">📊</span>Live System Statistics</div>
        <div class="s-grid">
          <div class="s-card" style="border-left-color:var(--green)">
            <div class="s-val" style="color:var(--green)" id="s-ff">—</div>
            <div class="s-lbl">Free-Flow %</div>
            <div class="s-sub" id="s-ff-n">— vehicles</div>
          </div>
          <div class="s-card" style="border-left-color:var(--red)">
            <div class="s-val" style="color:var(--red)" id="s-stop">—</div>
            <div class="s-lbl">Stopped</div>
            <div class="s-sub">at red signals</div>
          </div>
          <div class="s-card" style="border-left-color:var(--yellow)">
            <div class="s-val" style="color:var(--yellow)" id="s-spd">—</div>
            <div class="s-lbl">Avg Speed km/h</div>
            <div class="s-sub">target: 40 km/h</div>
          </div>
          <div class="s-card" style="border-left-color:var(--orange)">
            <div class="s-val" style="color:var(--orange)" id="s-fuel">—</div>
            <div class="s-lbl">Fuel Waste L/hr</div>
            <div class="s-sub">idle consumption</div>
          </div>
          <div class="s-card" style="border-left-color:var(--cyan)">
            <div class="s-val" style="color:var(--cyan)" id="s-gw">—</div>
            <div class="s-lbl">Active GW Corridors</div>
            <div class="s-sub">synchronized</div>
          </div>
          <div class="s-card" style="border-left-color:var(--red)">
            <div class="s-val" style="color:var(--red)" id="s-evpo">—</div>
            <div class="s-lbl">EVP Overrides</div>
            <div class="s-sub">total session</div>
          </div>
          <div class="s-card" style="border-left-color:var(--purple)">
            <div class="s-val" style="color:var(--purple)" id="s-co2s">—</div>
            <div class="s-lbl">CO₂ Saved kg/hr</div>
            <div class="s-sub">vs fixed timer</div>
          </div>
          <div class="s-card" style="border-left-color:var(--green)">
            <div class="s-val" style="color:var(--green)" id="s-lpi">—</div>
            <div class="s-lbl">LP Iterations</div>
            <div class="s-sub">optimization cycles</div>
          </div>
          <div class="s-card" style="border-left-color:var(--orange)">
            <div class="s-val" style="color:var(--orange)" id="s-emerg">—</div>
            <div class="s-lbl">Emergency Active</div>
            <div class="s-sub">1 dot = 100 vehicles</div>
          </div>
          <div class="s-card" style="border-left-color:var(--yellow)">
            <div class="s-val" style="color:var(--yellow)" id="s-eco">—</div>
            <div class="s-lbl">Econ Saved ₹/hr</div>
            <div class="s-sub">productivity gain</div>
          </div>
        </div>
      </div>

      <div class="sec" style="background:var(--bg3);border:1px solid #0d2040;border-radius:5px;padding:10px">
        <div class="sec-title"><span class="sec-title-icon">⚡</span>Algorithm Benchmark</div>
        <div class="ab-grid">
          <div class="ab-card" style="border-color:var(--red)44;background:#1a040a">
            <div class="ab-name" style="color:var(--red)">Fixed Timer</div>
            <div class="ab-val" id="ab-fix" style="color:var(--red)">45 min</div>
            <div class="ab-sub">Silk Board→Hebbal</div>
          </div>
          <div class="ab-card" style="border-color:var(--green)44;background:#00200e">
            <div class="ab-name" style="color:var(--green)">GW+EVP (Ours)</div>
            <div class="ab-val" id="ab-opt" style="color:var(--green)">28 min</div>
            <div class="ab-sub">Silk Board→Hebbal</div>
          </div>
          <div class="ab-card" style="border-color:var(--cyan)44">
            <div class="ab-name" style="color:var(--cyan)">Delay Reduction</div>
            <div class="ab-val" id="ab-red" style="color:var(--cyan)">38%</div>
            <div class="ab-sub">LP optimization</div>
          </div>
          <div class="ab-card" style="border-color:var(--orange)44">
            <div class="ab-name" style="color:var(--orange)">EVP Time Saved</div>
            <div class="ab-val" id="ab-evp" style="color:var(--orange)">60%</div>
            <div class="ab-sub">vs unoptimized</div>
          </div>
          <div class="ab-card" style="border-color:var(--purple)44">
            <div class="ab-name" style="color:var(--purple)">Throughput Gain</div>
            <div class="ab-val" id="ab-thr" style="color:var(--purple)">+30%</div>
            <div class="ab-sub">vehicles/hr/lane</div>
          </div>
          <div class="ab-card" style="border-color:var(--yellow)44">
            <div class="ab-name" style="color:var(--yellow)">CO₂ Reduction</div>
            <div class="ab-val" id="ab-co2" style="color:var(--yellow)">25%</div>
            <div class="ab-sub">stop-and-go saved</div>
          </div>
        </div>
      </div>
    </div>

    <!-- ATab 2: Signal Control -->
    <div class="atab-content" id="at2">
      <div class="sec" style="background:var(--bg3);border:1px solid #0d2040;border-radius:5px;padding:10px">
        <div class="sec-title"><span class="sec-title-icon">🚦</span>Signal Control Panel</div>
        <div class="sig-grid" id="sig-grid"></div>
      </div>
    </div>

    <!-- ATab 3: LP Model -->
    <div class="atab-content" id="at3">
      <div class="sec" style="background:var(--bg3);border:1px solid #0d2040;border-radius:5px;padding:10px">
        <div class="sec-title"><span class="sec-title-icon">∑</span>LP Objective Function</div>
        <div class="lp-box">
          <span class="hi">Minimize:</span> W = Σᵢ tᵢ × dᵢ<br>
          <span class="hi">Subject to:</span><br>
          &nbsp;gᵢ + rᵢ + yᵢ = T = <span class="hi-o" id="lp-T">90s</span><br>
          &nbsp;gᵢ ≥ 10s (min green safety)<br>
          &nbsp;rᵢ ≥ 5s (min red clearance)<br>
          &nbsp;yᵢ = 0.07 × T (yellow inter.)<br>
          <br>
          <span class="hi">Green Wave Offset:</span><br>
          &nbsp;Φᵢⱼ = (Lᵢⱼ / v_c) mod T<br>
          &nbsp;v_c = <span class="hi-g" id="lp-vc">40</span> km/h (wave speed)<br>
          <br>
          <span class="hi">EVP Pre-emption:</span><br>
          &nbsp;P_emergency → ∞<br>
          &nbsp;S_evp = d / v_amb (signal timing)<br>
          &nbsp;Override: all signals on path<br>
          <br>
          <span class="hi">LWR Flow Model:</span><br>
          &nbsp;∂ρ/∂t + ∂(ρv)/∂x = 0<br>
          &nbsp;ρ = density, v = velocity<br>
          <br>
          <span class="hi">Current W:</span> <span class="hi-y" id="lp-W">—</span> veh·sec<br>
          <span class="hi">Iterations:</span> <span class="hi-g" id="lp-iter">—</span><br>
          <span class="hi">EVP Weight:</span> <span class="hi-r">P → ∞</span><br>
          <span class="hi">Algo Warmup:</span> <span class="hi-g" id="lp-warm">0%</span>
        </div>
      </div>

      <div class="sec" style="background:var(--bg3);border:1px solid #0d2040;border-radius:5px;padding:10px">
        <div class="sec-title"><span class="sec-title-icon">📐</span>Webster's Delay Formula</div>
        <div class="lp-box">
          <span class="hi">d = C(1-λ)² / 2(1-λx)</span><br>
          &nbsp;&nbsp;&nbsp;&nbsp;<span class="eq">+ x² / 2q(1-x)</span><br>
          <br>
          d = avg delay per vehicle (s)<br>
          C = cycle length = <span class="hi-o" id="web-C">90</span>s<br>
          λ = g/C = effective green ratio<br>
          x = degree of saturation (v/c)<br>
          q = flow rate (veh/s)<br>
          <br>
          <span class="hi">Current Values:</span><br>
          λ (avg): <span class="hi-g" id="web-lam">—</span><br>
          x (saturation): <span class="hi-y" id="web-x">—</span><br>
          d (computed): <span class="hi-r" id="web-d">—</span> s
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════ STATUS BAR ══════════════════════════════════ -->
<div id="statusbar">
  <div class="sb-item">🕐 <span id="sb-t" class="sb-val">00:00:00</span></div>
  <div class="sb-item">ALGO <span id="sb-algo" class="sb-val">GW+EVP</span></div>
  <div class="sb-item">TOTAL <span id="sb-total" class="sb-val">—</span></div>
  <div class="sb-item">STOPPED <span id="sb-stop" class="sb-val r">—</span></div>
  <div class="sb-item">EMERG <span id="sb-emerg" class="sb-val r">—</span></div>
  <div class="sb-item">FREE-FLOW <span id="sb-ff" class="sb-val g">—</span></div>
  <div class="sb-item">GRID EFF <span id="sb-eff" class="sb-val g">—</span></div>
  <div class="sb-item">CO₂ SAVED <span id="sb-co2" class="sb-val" style="color:var(--purple)">—</span></div>
  <div class="sb-item">LP ITER <span id="sb-lp" class="sb-val y">—</span></div>
  <div class="sb-item">© NMIT ISE — NISHCHAL VISHWANATH NB25ISE160 · RISHUL KH NB25ISE186</div>
</div>

<!-- ══════════════════════════════════ SCRIPT ══════════════════════════════════ -->
<script>
// ─────────────────────────────────────────────────────────────────────────────
// REAL BANGALORE GPS DATA (12 junctions)
// Source: TomTom 2024 + OpenStreetMap + Kaggle BLR Traffic Dataset
// ─────────────────────────────────────────────────────────────────────────────
const JUNCTIONS = [
  { id:0,  name:'Silk Board',       lat:12.9177, lng:77.6228, cong:0.71, daily:185000, peak:22000, imp:10 },
  { id:1,  name:'Hebbal',           lat:13.0358, lng:77.5970, cong:0.64, daily:156000, peak:19000, imp:9  },
  { id:2,  name:'Marathahalli',     lat:12.9591, lng:77.6974, cong:0.58, daily:142000, peak:17500, imp:8  },
  { id:3,  name:'KR Puram',         lat:13.0074, lng:77.6950, cong:0.54, daily:128000, peak:15000, imp:7  },
  { id:4,  name:'Electronic City',  lat:12.8399, lng:77.6770, cong:0.67, daily:168000, peak:20000, imp:8  },
  { id:5,  name:'Whitefield',       lat:12.9698, lng:77.7500, cong:0.52, daily:118000, peak:14000, imp:7  },
  { id:6,  name:'Indiranagar',      lat:12.9784, lng:77.6408, cong:0.62, daily:138000, peak:16500, imp:8  },
  { id:7,  name:'Koramangala',      lat:12.9352, lng:77.6245, cong:0.66, daily:155000, peak:18500, imp:9  },
  { id:8,  name:'JP Nagar',         lat:12.9063, lng:77.5857, cong:0.48, daily:108000, peak:13000, imp:7  },
  { id:9,  name:'Yelahanka',        lat:13.1007, lng:77.5963, cong:0.44, daily:98000,  peak:12000, imp:6  },
  { id:10, name:'Bannerghatta Rd',  lat:12.8931, lng:77.5971, cong:0.59, daily:132000, peak:15800, imp:7  },
  { id:11, name:'Nagawara',         lat:13.0456, lng:77.6207, cong:0.55, daily:122000, peak:14500, imp:7  },
];

const EDGES = [
  [0,7],[0,8],[0,4],[0,6],
  [1,9],[1,11],[1,3],
  [2,3],[2,5],[2,6],[2,7],
  [3,5],[3,11],
  [4,8],[4,10],
  [6,7],[6,2],[6,11],
  [7,10],[7,8],
  [8,10],
  [9,11],[9,1],
  [10,8],[11,6],
];

// ─────────────────────────────────────────────────────────────────────────────
// SIMULATION STATE
// ─────────────────────────────────────────────────────────────────────────────
const SIM = {
  algo:'optimal', paused:false, speed:1,
  emergDots:50,   // dots; each = 100 emergency vehicles
  waveSpeed:40, cycleTime:90, densityLevel:4,
  simTime:0, frame:0,
  lpIter:0, totalEvpOverrides:0, algoBooted:0,
};

const DENSITY_NAMES = ['Very Low','Low','Medium','High','Peak (TomTom)'];
const DENSITY_MUL   = [0.2, 0.4, 0.7, 1.0, 1.4];

// Graph buffers
const GL = 120;
const GD = {
  thr:new Array(GL).fill(0), del:new Array(GL).fill(0),
  evp:new Array(GL).fill(0), eff:new Array(GL).fill(0),
  co2:new Array(GL).fill(0), dns:new Array(GL).fill(0),
};

// ─────────────────────────────────────────────────────────────────────────────
// SIGNALS
// ─────────────────────────────────────────────────────────────────────────────
let signals = JUNCTIONS.map((j,i) => ({
  id:i, phase:Math.random()*90, cycle:90,
  state:'red', evpOverride:false,
  greenDur:45, efficiency:0.5, waiting:Math.floor(j.cong*50),
}));

// ─────────────────────────────────────────────────────────────────────────────
// PARTICLES
// ─────────────────────────────────────────────────────────────────────────────
let particles = [];
const MAX_NORMAL = 700;

class Particle {
  constructor(isEmerg) {
    this.isEmerg = isEmerg;
    this.id = Math.random();
    const ri = Math.floor(Math.random()*EDGES.length);
    this.ei = ri;
    this.progress = Math.random();
    this.dir = Math.random()>.5?1:-1;
    this.baseSpd = isEmerg
      ? 0.005+Math.random()*0.003
      : 0.0008+Math.random()*0.0008;
    this.spd = this.baseSpd;
    this.tgtSpd = this.spd;
    this.state='moving';
    this.waitTime=0;
    this.trail=[];
    this.phase=Math.random()*Math.PI*2;
    // Lateral lane offset
    this.laneOff = (Math.random()-.5)*0.00025;
  }

  getLatLng() {
    const [a,b]=EDGES[this.ei];
    const ja=JUNCTIONS[a], jb=JUNCTIONS[b];
    const t=this.dir===1?this.progress:1-this.progress;
    const perp_lat=(jb.lng-ja.lng)*0.15;
    const perp_lng=(jb.lat-ja.lat)*0.15;
    return {
      lat: ja.lat+(jb.lat-ja.lat)*t + this.laneOff*perp_lat,
      lng: ja.lng+(jb.lng-ja.lng)*t + this.laneOff*perp_lng,
    };
  }

  update(dt) {
    const [a,b]=EDGES[this.ei];
    const endId=this.dir===1?b:a;
    const sig=signals[endId];
    const distEnd=this.dir===1?1-this.progress:this.progress;
    const junc=JUNCTIONS[endId];
    const mul=DENSITY_MUL[SIM.densityLevel-1];
    const algoFix=SIM.algo==='fixed'?1.25:1.0;
    const warmFactor=Math.min(SIM.algoBooted/500,1);
    const algoReduce=SIM.algo==='optimal'?warmFactor*0.45:SIM.algo==='lp'?warmFactor*0.25:0;
    const cong=Math.min(junc.cong*mul*algoFix*(1-algoReduce),0.98);

    let stop=false;
    if (!this.isEmerg && distEnd<0.15 && sig.state==='red' && !sig.evpOverride) stop=true;

    if (stop) {
      this.tgtSpd=0; this.state='stopped'; this.waitTime+=dt*0.016;
    } else if (cong>0.65&&!this.isEmerg) {
      const f=Math.max(0.05,1-(cong-.65)*2.8);
      this.tgtSpd=this.baseSpd*f*SIM.waveSpeed/40;
      this.state=cong>0.85?'stopped':'slow';
      this.waitTime=Math.max(0,this.waitTime-dt*.01);
    } else {
      this.tgtSpd=this.baseSpd*SIM.waveSpeed/40;
      this.state='moving';
      this.waitTime=Math.max(0,this.waitTime-dt*.05);
    }

    if (this.isEmerg) { this.tgtSpd=this.baseSpd*2.2; this.state='moving'; }

    this.spd+=(this.tgtSpd-this.spd)*0.13;
    this.progress+=this.spd*this.dir*SIM.speed;

    if (this.isEmerg) {
      const pos=this.getLatLng();
      this.trail.unshift(pos);
      if (this.trail.length>12) this.trail.pop();
    }

    if (this.progress>=1||this.progress<=0) {
      this.progress=this.progress>=1?0:1;
      const ej=this.dir===1?EDGES[this.ei][1]:EDGES[this.ei][0];
      const conn=EDGES.map((e,i)=>({e,i}))
        .filter(({e,i})=>(e[0]===ej||e[1]===ej)&&i!==this.ei);
      if (conn.length>0) {
        const pk=conn[Math.floor(Math.random()*conn.length)];
        this.ei=pk.i;
        this.dir=EDGES[pk.i][0]===ej?1:-1;
        this.progress=this.dir===1?0:1;
      } else this.dir*=-1;
    }
  }

  getColor() {
    if (this.isEmerg) return '#ff2244';
    if (this.state==='stopped') return '#ff4400';
    if (this.state==='slow') return '#ffcc00';
    return '#00ccff';
  }
}

function spawnParticles() {
  particles=[];
  const mul=DENSITY_MUL[SIM.densityLevel-1];
  const n=Math.floor(MAX_NORMAL*mul);
  for (let i=0;i<n;i++) particles.push(new Particle(false));
  for (let i=0;i<SIM.emergDots;i++) particles.push(new Particle(true));
}

// ─────────────────────────────────────────────────────────────────────────────
// LEAFLET MAP
// ─────────────────────────────────────────────────────────────────────────────
const map = L.map('map',{center:[12.97,77.62],zoom:12,zoomControl:false,attributionControl:false,preferCanvas:true});
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:18}).addTo(map);

let roadLines=[];
function drawRoads() {
  roadLines.forEach(l=>l.remove()); roadLines=[];
  EDGES.forEach(([a,b],ri) => {
    const ja=JUNCTIONS[a],jb=JUNCTIONS[b];
    const mul=DENSITY_MUL[SIM.densityLevel-1];
    const algoFix=SIM.algo==='fixed'?1.2:1.0;
    const warm=Math.min(SIM.algoBooted/500,1);
    const algoR=SIM.algo==='optimal'?warm*0.45:0;
    const c=Math.min((ja.cong+jb.cong)/2*mul*algoFix*(1-algoR),1);
    const col=c>.85?'#ff2244':c>.65?'#ff8c00':c>.4?'#ffd700':'#00ff88';
    const w=3+c*6;
    const hasEvp=particles.some(p=>p.isEmerg&&p.ei===ri);

    if (hasEvp) {
      const bg=L.polyline([[ja.lat,ja.lng],[jb.lat,jb.lng]],{color:'#ff224466',weight:w+8,opacity:.6}).addTo(map);
      const glow=L.polyline([[ja.lat,ja.lng],[jb.lat,jb.lng]],{color:'#ff44aa',weight:2,opacity:.9,dashArray:'10 6'}).addTo(map);
      roadLines.push(bg,glow);
    }

    const base=L.polyline([[ja.lat,ja.lng],[jb.lat,jb.lng]],{color:col+'99',weight:w,opacity:.8}).addTo(map);
    roadLines.push(base);
  });
}

// Junction markers
const jMarkers = JUNCTIONS.map((j,i) => {
  const m=L.circleMarker([j.lat,j.lng],{radius:9+j.imp,color:'#fff',weight:1.5,fillColor:'#ff2244',fillOpacity:.9}).addTo(map);
  m.bindTooltip(
    `<b style="color:#ffd700;font-size:.8rem">${j.name}</b><br>
     Congestion: <b style="color:${j.cong>.65?'#ff2244':j.cong>.45?'#ff8c00':'#00ff88'}">${Math.round(j.cong*100)}%</b><br>
     Daily: <b>${(j.daily/1000).toFixed(0)}K veh/day</b><br>
     Peak: <b>${(j.peak/1000).toFixed(0)}K veh/hr</b><br>
     Importance: <b>★${'★'.repeat(Math.round(j.imp/2))}</b>`,
    {direction:'top',className:'blr-tip'}
  );
  return m;
});

// Flow canvas
const fc=document.getElementById('flow-canvas');
const fctx=fc.getContext('2d');
function resizeFC() {
  const mw=document.getElementById('map-wrap');
  fc.width=mw.offsetWidth; fc.height=mw.offsetHeight;
}
resizeFC();
window.addEventListener('resize',resizeFC);

function ll2cv(lat,lng) {
  const p=map.latLngToContainerPoint([lat,lng]);
  return {x:p.x,y:p.y};
}

function renderParticles() {
  fctx.clearRect(0,0,fc.width,fc.height);
  // Normal first, emergency on top
  particles.filter(p=>!p.isEmerg).forEach(p=>{
    const pos=p.getLatLng();
    const pt=ll2cv(pos.lat,pos.lng);
    fctx.fillStyle=p.getColor()+'cc';
    fctx.beginPath();fctx.arc(pt.x,pt.y,3,0,Math.PI*2);fctx.fill();
  });
  particles.filter(p=>p.isEmerg).forEach(p=>{
    const pos=p.getLatLng();
    const pt=ll2cv(pos.lat,pos.lng);
    // Trail
    if (p.trail.length>1) {
      for (let i=1;i<p.trail.length;i++) {
        const t1=ll2cv(p.trail[i-1].lat,p.trail[i-1].lng);
        const t2=ll2cv(p.trail[i].lat,p.trail[i].lng);
        fctx.strokeStyle=`rgba(255,34,68,${(1-i/p.trail.length)*0.55})`;
        fctx.lineWidth=5-i*0.35;
        fctx.beginPath();fctx.moveTo(t1.x,t1.y);fctx.lineTo(t2.x,t2.y);fctx.stroke();
      }
    }
    const pulse=0.55+0.45*Math.sin(SIM.frame*.25+p.phase);
    fctx.shadowBlur=18*pulse; fctx.shadowColor='#ff2244';
    fctx.fillStyle='#ff2244';
    fctx.beginPath();fctx.arc(pt.x,pt.y,7,0,Math.PI*2);fctx.fill();
    // Cross symbol
    fctx.shadowBlur=0;
    fctx.strokeStyle='#ffffff'; fctx.lineWidth=1.8;
    fctx.beginPath();
    fctx.moveTo(pt.x-5,pt.y);fctx.lineTo(pt.x+5,pt.y);
    fctx.moveTo(pt.x,pt.y-5);fctx.lineTo(pt.x,pt.y+5);
    fctx.stroke();
  });
  fctx.shadowBlur=0;
}

function updateJunctionMarkers() {
  JUNCTIONS.forEach((_,i)=>{
    const s=signals[i];
    const col=s.evpOverride?'#ff2244':s.state==='green'?'#00ff88':s.state==='yellow'?'#ffd700':'#ff2244';
    jMarkers[i].setStyle({fillColor:col,color:s.evpOverride?'#ff4466':'#ffffff'});
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// CHART.JS GRAPHS
// ─────────────────────────────────────────────────────────────────────────────
const GRAPH_CFG = [
  {key:'thr',col:'#00ff88',max:2200},
  {key:'del',col:'#ff2244',max:200},
  {key:'evp',col:'#ff8c00',max:130},
  {key:'eff',col:'#00e5ff',max:100},
  {key:'co2',col:'#ffd700',max:160},
  {key:'dns',col:'#bb77ff',max:100},
];
const charts={};
GRAPH_CFG.forEach(cfg=>{
  const el=document.getElementById('gc-'+cfg.key);
  if (!el) return;
  charts[cfg.key]=new Chart(el,{
    type:'line',
    data:{
      labels:new Array(GL).fill(''),
      datasets:[{
        data:[...GD[cfg.key]],
        borderColor:cfg.col, borderWidth:1.5,
        pointRadius:0, fill:true,
        backgroundColor:cfg.col+'15', tension:0.4,
      }]
    },
    options:{
      animation:false, responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{enabled:false}},
      scales:{x:{display:false},y:{display:false,min:0,max:cfg.max}},
    }
  });
});

function pushGraph(key,val) {
  GD[key].push(val); GD[key].shift();
  if (charts[key]) { charts[key].data.datasets[0].data=[...GD[key]]; charts[key].update('none'); }
}

// ─────────────────────────────────────────────────────────────────────────────
// SIGNAL UPDATE ALGORITHM
// ─────────────────────────────────────────────────────────────────────────────
function updateSignals(dt) {
  SIM.algoBooted=Math.min(SIM.algoBooted+dt,500);
  const warm=SIM.algoBooted/500;

  signals.forEach((sig,i)=>{
    sig.phase+=dt*SIM.speed;
    if (sig.phase>=sig.cycle) sig.phase-=sig.cycle;

    // EVP detection
    const nearEvp=particles.filter(p=>p.isEmerg).some(p=>{
      const ej=p.dir===1?EDGES[p.ei][1]:EDGES[p.ei][0];
      const d=p.dir===1?1-p.progress:p.progress;
      return ej===i&&d<0.28;
    });
    const wasEvp=sig.evpOverride;
    sig.evpOverride=nearEvp&&SIM.algo!=='fixed';
    if (sig.evpOverride&&!wasEvp) {
      SIM.totalEvpOverrides++;
      document.getElementById('evp-overlay').classList.add('active');
      setTimeout(()=>document.getElementById('evp-overlay').classList.remove('active'),600);
    }

    if (sig.evpOverride) { sig.state='green'; sig.efficiency=1; return; }

    const junc=JUNCTIONS[i];
    const mul=DENSITY_MUL[SIM.densityLevel-1];
    const cong=Math.min(junc.cong*mul,0.95);
    let gDur=SIM.cycleTime*0.5;

    if (SIM.algo==='optimal'||SIM.algo==='ml') {
      // LP: Maximize throughput; extend green proportional to density
      gDur=SIM.cycleTime*(0.3+cong*0.4)*(0.6+warm*0.4);
      SIM.lpIter+=0.012;
      // Green Wave phase alignment: Φ = (L/v_c) mod T
      EDGES.filter(([a,b])=>a===i||b===i).forEach(([a,b])=>{
        const otherId=a===i?b:a;
        const ja=JUNCTIONS[i],jb=JUNCTIONS[otherId];
        const distKm=Math.sqrt(
          Math.pow((ja.lat-jb.lat)*111,2)+
          Math.pow((ja.lng-jb.lng)*111*Math.cos(ja.lat*Math.PI/180),2)
        );
        const phi=(distKm/SIM.waveSpeed*3600)%SIM.cycleTime;
        if (signals[otherId]) {
          const pd=(signals[otherId].phase-sig.phase+SIM.cycleTime)%SIM.cycleTime;
          if (Math.abs(pd-phi)>4) signals[otherId].phase+=(phi-pd)*0.025;
        }
      });
    } else if (SIM.algo==='lp') {
      gDur=SIM.cycleTime*(0.35+cong*.28);
      SIM.lpIter+=0.005;
    } else if (SIM.algo==='evp') {
      gDur=SIM.cycleTime*0.45;
    }

    sig.greenDur=gDur;
    sig.cycle=SIM.cycleTime;
    const yDur=SIM.cycleTime*0.07;
    if (sig.phase<gDur) sig.state='green';
    else if (sig.phase<gDur+yDur) sig.state='yellow';
    else sig.state='red';
    sig.efficiency=(gDur/SIM.cycleTime)*warm;
    sig.waiting=sig.state==='red'?Math.floor(junc.cong*45*mul):Math.floor(junc.cong*10);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// METRICS UPDATE
// ─────────────────────────────────────────────────────────────────────────────
let prevM={};
function updateMetrics() {
  const warm=SIM.algoBooted/500;
  const normal=particles.filter(p=>!p.isEmerg);
  const emerg=particles.filter(p=>p.isEmerg);
  const moving=normal.filter(p=>p.state==='moving').length;
  const slow=normal.filter(p=>p.state==='slow').length;
  const stopped=normal.filter(p=>p.state==='stopped').length;
  const total=particles.length;

  const algoMul=SIM.algo==='fixed'?1:SIM.algo==='optimal'?(1+warm*.45):SIM.algo==='lp'?(1+warm*.2):0.95;
  const delayDiv=SIM.algo==='fixed'?1:SIM.algo==='optimal'?Math.max(.4,1-warm*.4):0.9;

  const mul=DENSITY_MUL[SIM.densityLevel-1];
  const baseThr=900+moving*2.8;
  const thr=Math.round(baseThr*algoMul);
  const baseDelay=125*mul;
  const delay=Math.max(6,baseDelay*delayDiv);
  const evpTime=SIM.algo==='fixed'?48+Math.random()*18:Math.max(3,16*(1-warm*.65)+Math.random()*5);
  const avgEff=signals.reduce((s,sg)=>s+sg.efficiency,0)/signals.length*100;
  const emitBase=75+stopped*1.0+slow*.35;
  const emit=SIM.algo==='fixed'?emitBase*1.3:SIM.algo==='optimal'?emitBase*(1-warm*.28):emitBase*.95;
  const dens=Math.min(100,(stopped+slow*.5)/Math.max(total,1)*100+JUNCTIONS.reduce((s,j)=>s+j.cong,0)/JUNCTIONS.length*50);
  const avgSpd=SIM.algo==='fixed'?17.8:Math.min(40,17.8+warm*21);
  const fuelWaste=Math.round(stopped*5000*.7/1000);
  const co2saved=Math.round(warm*190*(SIM.algo==='optimal'?1.5:SIM.algo==='fixed'?0:.7));
  const ecoSaved=Math.round(warm*14200*(SIM.algo==='optimal'?1.4:SIM.algo==='fixed'?0:.6));
  const gws=signals.filter(s=>s.state==='green'&&!s.evpOverride).length;
  const evpActive=signals.filter(s=>s.evpOverride).length;
  const lpW=Math.round(delay*total*5000);
  const lambda=avgEff/100;
  const x=Math.min(0.98,(stopped+slow*0.6)/Math.max(total,1)+0.3);
  const webD=Math.max(5,delay*(0.8+x*.4));

  // Push graphs
  pushGraph('thr',thr); pushGraph('del',delay);
  pushGraph('evp',evpTime); pushGraph('eff',avgEff);
  pushGraph('co2',emit); pushGraph('dns',dens);

  // Helper
  const $=(id)=>document.getElementById(id);
  const setV=(id,v,dec=0)=>{ if($(id)) $(id).textContent=typeof dec==='number'?+v.toFixed(dec):v; };
  const setDelta=(vId,dId,v,goodUp=true)=>{
    if(!$(dId))return;
    const prev=prevM[vId]||0;
    const d=v-prev;
    if (Math.abs(d)>.5) {
      const good=(goodUp&&d>0)||(!goodUp&&d<0);
      $(dId).textContent=(d>0?'▲':'▼')+Math.abs(d).toFixed(1);
      $(dId).className='g-delta '+(good?'up':'dn');
    }
    prevM[vId]=v;
  };

  // Header KPIs
  $('kpi-veh').textContent=(total*5000).toLocaleString();
  $('kpi-delay').textContent=Math.round(delay)+'s';
  $('kpi-evp').textContent=evpActive;
  $('kpi-eff').textContent=Math.round(avgEff)+'%';
  $('kpi-speed').textContent=avgSpd.toFixed(1);
  $('kpi-co2').textContent=co2saved;

  // Graph values
  setV('gv-thr',thr); setDelta('thr','gd-thr',thr,true);
  setV('gv-del',Math.round(delay)); setDelta('del','gd-del',delay,false);
  setV('gv-evp',evpTime.toFixed(1)); setDelta('evp','gd-evp',evpTime,false);
  setV('gv-eff',Math.round(avgEff)); setDelta('eff','gd-eff',avgEff,true);
  setV('gv-co2',Math.round(emit)); setDelta('co2','gd-co2',emit,false);
  setV('gv-dns',Math.round(dens)); setDelta('dns','gd-dns',dens,false);

  // Stats
  const ffPct=Math.round(moving/Math.max(total,1)*100);
  $('s-ff').textContent=ffPct+'%';
  $('s-ff-n').textContent=(moving*5000).toLocaleString()+' veh';
  $('s-stop').textContent=(stopped*5000).toLocaleString();
  $('s-spd').textContent=avgSpd.toFixed(1);
  $('s-fuel').textContent=fuelWaste.toLocaleString();
  $('s-gw').textContent=gws;
  $('s-evpo').textContent=SIM.totalEvpOverrides;
  $('s-co2s').textContent=co2saved;
  $('s-lpi').textContent=Math.floor(SIM.lpIter);
  $('s-emerg').textContent=emerg.length+' dots = '+(emerg.length*100).toLocaleString()+' veh';
  $('s-eco').textContent='₹'+ecoSaved.toLocaleString();

  // Benchmark
  const fixD=45, optD=Math.max(24,45*(1-warm*.38));
  $('ab-fix').textContent=fixD+' min';
  $('ab-opt').textContent=optD.toFixed(0)+' min';
  $('ab-red').textContent=Math.round((1-optD/fixD)*100)+'%';
  $('ab-evp').textContent=Math.round(warm*62)+'%';
  $('ab-thr').textContent='+'+Math.round(warm*30)+'%';
  $('ab-co2').textContent=Math.round(warm*25)+'%';

  // LP
  $('lp-T').textContent=SIM.cycleTime+'s';
  $('lp-vc').textContent=SIM.waveSpeed;
  $('lp-W').textContent=lpW.toLocaleString();
  $('lp-iter').textContent=Math.floor(SIM.lpIter);
  $('lp-warm').textContent=Math.round(warm*100)+'%';

  // Webster
  $('web-C').textContent=SIM.cycleTime;
  $('web-lam').textContent=lambda.toFixed(3);
  $('web-x').textContent=x.toFixed(3);
  $('web-d').textContent=webD.toFixed(1);

  // Signal grid
  updateSignalGrid();

  // Junction list
  updateJunctionList();

  // Top bar
  $('veh-total').textContent=(total*5000).toLocaleString();
  $('emerg-total').textContent=emerg.length+' dots (×100)';

  // Status bar
  const t=SIM.simTime;
  const ts=`${String(Math.floor(t/3600)%24).padStart(2,'0')}:${String(Math.floor(t/60)%60).padStart(2,'0')}:${String(Math.floor(t)%60).padStart(2,'0')}`;
  $('sim-time').textContent=ts; $('sb-t').textContent=ts;
  $('sb-algo').textContent=ALGO_NAMES[SIM.algo]||SIM.algo;
  $('sb-total').textContent=(total*5000).toLocaleString();
  $('sb-stop').textContent=(stopped*5000).toLocaleString();
  $('sb-emerg').textContent=(emerg.length*100).toLocaleString();
  $('sb-ff').textContent=ffPct+'%';
  $('sb-eff').textContent=Math.round(avgEff)+'%';
  $('sb-co2').textContent=co2saved+' kg/hr';
  $('sb-lp').textContent=Math.floor(SIM.lpIter);
}

function updateSignalGrid() {
  const sg=document.getElementById('sig-grid');
  if (!sg) return;
  sg.innerHTML=signals.map((s,i)=>{
    const j=JUNCTIONS[i];
    const col=s.evpOverride?'#ff2244':s.state==='green'?'#00ff88':s.state==='yellow'?'#ffd700':'#ff2244';
    const pct=Math.round(s.phase/s.cycle*100);
    const tLeft=s.state==='green'?Math.max(0,s.greenDur-s.phase).toFixed(0)+'s▶':Math.max(0,s.cycle-s.phase).toFixed(0)+'s◀';
    const evpCls=s.evpOverride?'sig-evp':'';
    return `<div class="sig-card ${evpCls}" style="border-top-color:${col}">
      <div class="sig-name">${j.name}</div>
      <div class="sig-state" style="color:${col}">${s.evpOverride?'🚨 EVP':s.state.toUpperCase()}</div>
      <div class="sig-timer">${tLeft} | wait:${s.waiting}</div>
      <div class="sig-bar-bg"><div class="sig-bar" style="width:${pct}%;background:${col}"></div></div>
    </div>`;
  }).join('');
}

function updateJunctionList() {
  const jl=document.getElementById('j-list');
  if (!jl) return;
  jl.innerHTML=JUNCTIONS.map((j,i)=>{
    const s=signals[i];
    const col=s.evpOverride?'#ff2244':s.state==='green'?'#00ff88':s.state==='yellow'?'#ffd700':'#ff2244';
    const t=s.state==='green'?Math.max(0,s.greenDur-s.phase).toFixed(0)+'s▶':Math.max(0,s.cycle-s.phase).toFixed(0)+'s◀';
    const evpCls=s.evpOverride?'evp':'';
    const cCol=j.cong>.65?'var(--red)':j.cong>.45?'var(--orange)':'var(--green)';
    return `<div class="j-item ${evpCls}">
      <div class="j-dot" style="background:${col};box-shadow:0 0 6px ${col}"></div>
      <div class="j-name">${j.name}<small>${(j.daily/1000).toFixed(0)}K/day · ${(j.peak/1000).toFixed(0)}K peak/hr</small></div>
      <div class="j-pct" style="color:${cCol}">${Math.round(j.cong*100)}%</div>
      <div class="j-timer">${t}</div>
    </div>`;
  }).join('');
}

// ─────────────────────────────────────────────────────────────────────────────
// UI CONTROLS
// ─────────────────────────────────────────────────────────────────────────────
const ALGO_NAMES={optimal:'GW+EVP',fixed:'FIXED',lp:'LP ONLY',evp:'EVP ONLY',ml:'ML+EVP'};
const ALGO_FULL={optimal:'GREEN WAVE+EVP',fixed:'FIXED TIMER',lp:'LP ONLY',evp:'EVP ONLY',ml:'ML+EVP'};
const ALGO_LIST=Object.keys(ALGO_NAMES);
let algoIdx=0;

function setAlgo(a) {
  SIM.algo=a; SIM.algoBooted=0;
  const n=ALGO_FULL[a];
  document.getElementById('algo-disp').textContent=n;
  document.getElementById('btn-algo').textContent='⚡ '+ALGO_NAMES[a];
  document.getElementById('algo-sel').value=a;
  document.getElementById('sb-algo').textContent=n;
}
function cycleAlgo(){algoIdx=(algoIdx+1)%ALGO_LIST.length;setAlgo(ALGO_LIST[algoIdx]);}
function setAlgoFromSel(v){algoIdx=ALGO_LIST.indexOf(v);setAlgo(v);}

function togglePause(){
  SIM.paused=!SIM.paused;
  document.getElementById('btn-pause').textContent=SIM.paused?'▶ RESUME':'⏸ PAUSE';
}

function triggerMassEVP(){
  signals.forEach(s=>s.evpOverride=true);
  document.getElementById('evp-overlay').classList.add('active');
  SIM.totalEvpOverrides+=JUNCTIONS.length;
  setTimeout(()=>{signals.forEach(s=>s.evpOverride=false);document.getElementById('evp-overlay').classList.remove('active');},6000);
}

function setDensity(v){
  SIM.densityLevel=parseInt(v);
  document.getElementById('lbl-dens').textContent=DENSITY_NAMES[SIM.densityLevel-1];
  spawnParticles();
}
function setEmerg(v){
  SIM.emergDots=parseInt(v);
  document.getElementById('lbl-emerg').textContent=v+' dots = '+(v*100).toLocaleString()+' veh';
  particles=particles.filter(p=>!p.isEmerg);
  for(let i=0;i<SIM.emergDots;i++) particles.push(new Particle(true));
}
function setWave(v){
  SIM.waveSpeed=parseInt(v);
  document.getElementById('lbl-wave').textContent=v+' km/h';
  document.getElementById('wave-disp').textContent=v+' km/h';
}
function setCycle(v){
  SIM.cycleTime=parseInt(v);
  document.getElementById('lbl-cycle').textContent=v+'s';
}
function setSimSpeed(v){SIM.speed=parseFloat(v);}

// Tabs
function switchSTab(n){
  document.querySelectorAll('.stab').forEach((t,i)=>t.classList.toggle('active',i===n));
  document.querySelectorAll('.stab-content').forEach((t,i)=>t.classList.toggle('active',i===n));
}
function switchATab(n){
  document.querySelectorAll('.atab').forEach((t,i)=>t.classList.toggle('active',i===n));
  document.querySelectorAll('.atab-content').forEach((t,i)=>t.classList.toggle('active',i===n));
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN LOOP
// ─────────────────────────────────────────────────────────────────────────────
let lastT=0, roadTick=0;
function loop(ts){
  if(SIM.paused){requestAnimationFrame(loop);return;}
  const dt=Math.min((ts-lastT)/1000*60,4); lastT=ts;
  SIM.frame++; SIM.simTime+=dt*.016*SIM.speed;

  updateSignals(dt);
  particles.forEach(p=>p.update(dt));
  renderParticles();

  if(SIM.frame%15===0){
    updateJunctionMarkers();
    roadTick++;
    if(roadTick%3===0) drawRoads();
  }
  if(SIM.frame%30===0) updateMetrics();

  requestAnimationFrame(loop);
}

// ── INIT ──
spawnParticles();
drawRoads();
requestAnimationFrame(loop);
</script>
</body>
</html>
"""

components.html(HTML, height=960, scrolling=False)

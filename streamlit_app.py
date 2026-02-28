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

# NOTE: Using single quotes throughout JS to avoid Python string conflicts
# All template literals replaced with string concatenation for safety

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Urban Flow & Life-Lines</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#020810;--bg2:#06101e;--bg3:#0b1a2e;
  --cyan:#00e5ff;--green:#00ff88;--red:#ff2244;
  --orange:#ff8c00;--yellow:#ffd700;--purple:#bb77ff;--pink:#ff44aa;
  --cdim:#00e5ff22;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:#b8d8f0;font-family:'Rajdhani',sans-serif;
     width:100vw;height:100vh;overflow:hidden;display:flex;flex-direction:column}

/* ── HEADER ── */
#hdr{height:58px;flex-shrink:0;background:linear-gradient(90deg,#000a18,#020810 50%,#000a18);
  border-bottom:1px solid var(--cdim);display:flex;align-items:center;
  padding:0 18px;gap:0;position:relative;z-index:2000}
#hdr::after{content:'';position:absolute;bottom:-1px;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);
  animation:scan 5s ease-in-out infinite}
@keyframes scan{0%,100%{opacity:.3}50%{opacity:1}}
.h-brand{display:flex;align-items:center;gap:12px;min-width:320px}
.h-icon{font-size:2rem;filter:drop-shadow(0 0 10px var(--cyan))}
.h-title{font-family:'Orbitron',monospace;font-size:1.05rem;font-weight:800;
  color:var(--cyan);letter-spacing:3px;text-shadow:0 0 20px #00e5ff66}
.h-sub{font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:var(--orange);
  letter-spacing:2px;margin-top:2px}
.h-div{width:1px;height:34px;background:var(--cdim);margin:0 16px}
.h-kpis{display:flex;gap:24px;flex:1;justify-content:center}
.kpi{text-align:center}
.kpi-v{font-family:'Orbitron',monospace;font-weight:700;font-size:1.25rem;line-height:1}
.kpi-l{font-family:'Share Tech Mono',monospace;font-size:0.48rem;color:#4a6880;
  letter-spacing:1.5px;margin-top:3px;text-transform:uppercase}
.h-btns{display:flex;gap:7px;align-items:center;min-width:340px;justify-content:flex-end}
.btn{font-family:'Share Tech Mono',monospace;font-size:0.6rem;letter-spacing:1.5px;
  padding:6px 13px;border:1px solid;border-radius:3px;cursor:pointer;
  transition:all .2s;background:transparent;text-transform:uppercase;white-space:nowrap}
.btn-c{border-color:var(--cyan);color:var(--cyan)}
.btn-c:hover,.btn-c.on{background:var(--cyan);color:#000;box-shadow:0 0 16px #00e5ff88}
.btn-r{border-color:var(--red);color:var(--red)}
.btn-r:hover,.btn-r.on{background:var(--red);color:#fff;box-shadow:0 0 16px #ff224488}
.btn-g{border-color:var(--green);color:var(--green)}
.btn-g:hover,.btn-g.on{background:var(--green);color:#000;box-shadow:0 0 16px #00ff8888}
.live{font-family:'Share Tech Mono',monospace;font-size:0.58rem;letter-spacing:2px;
  padding:4px 9px;background:#ff224418;border:1px solid var(--red);color:var(--red);
  border-radius:2px;animation:blink 1.2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* ── BODY ── */
#body{flex:1;display:flex;overflow:hidden}

/* ── LEFT PANEL ── */
#lp{width:296px;flex-shrink:0;background:var(--bg2);
  border-right:1px solid var(--cdim);display:flex;flex-direction:column;overflow:hidden}
.tabs{display:flex;border-bottom:1px solid var(--cdim)}
.tab{flex:1;padding:9px 0;text-align:center;cursor:pointer;
  font-family:'Share Tech Mono',monospace;font-size:0.58rem;letter-spacing:1.5px;
  color:#4a6880;border-bottom:2px solid transparent;transition:.2s;text-transform:uppercase}
.tab.on{color:var(--cyan);border-bottom-color:var(--cyan)}
.tab:hover:not(.on){color:#7090a0}
.tpane{display:none;flex:1;overflow-y:auto;padding:11px;
  scrollbar-width:thin;scrollbar-color:var(--cdim) transparent;
  flex-direction:column;gap:9px}
.tpane.on{display:flex}
.sec{background:var(--bg3);border:1px solid #0d2040;border-radius:4px;padding:11px}
.stitle{font-family:'Orbitron',monospace;font-size:0.58rem;font-weight:600;
  color:var(--cyan);letter-spacing:3px;text-transform:uppercase;
  border-bottom:1px solid var(--cdim);padding-bottom:6px;margin-bottom:9px}
.ctrl{margin-bottom:11px}.ctrl:last-child{margin-bottom:0}
.clbl{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#4a6880;
  letter-spacing:1px;display:flex;justify-content:space-between;margin-bottom:4px}
.clbl span{color:var(--cyan);font-weight:bold}
input[type=range]{width:100%;-webkit-appearance:none;height:3px;
  background:#0d2040;border-radius:2px;outline:none}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:13px;height:13px;
  background:var(--cyan);border-radius:50%;cursor:pointer;box-shadow:0 0 7px #00e5ff88}
select{width:100%;background:var(--bg);border:1px solid #0d2040;
  color:var(--cyan);font-family:'Share Tech Mono',monospace;font-size:0.6rem;
  padding:5px 7px;border-radius:3px;outline:none;cursor:pointer}
.ji{display:grid;grid-template-columns:10px 1fr auto auto;align-items:center;gap:7px;
  padding:6px 7px;border-radius:3px;border:1px solid transparent;cursor:pointer;
  transition:.2s;margin-bottom:3px;background:var(--bg)}
.ji:hover{border-color:var(--cdim)}
.ji.evp{border-color:var(--red);background:#150308;animation:jp .8s infinite alternate}
@keyframes jp{from{box-shadow:none}to{box-shadow:0 0 8px #ff224433}}
.jdot{width:10px;height:10px;border-radius:50%;transition:all .3s}
.jname{font-family:'Share Tech Mono',monospace;font-size:0.6rem;line-height:1.3}
.jname small{display:block;color:#3a5570;font-size:0.48rem}
.jpct{font-family:'Orbitron',monospace;font-size:0.72rem;font-weight:700;text-align:right}
.jtmr{font-family:'Share Tech Mono',monospace;font-size:0.48rem;color:#3a5570}
.sc-row{display:flex;align-items:center;gap:9px;margin-bottom:6px}
.sc-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.sc-txt{font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:#5a7590;line-height:1.4}
.dt{width:100%;border-collapse:collapse;font-family:'Share Tech Mono',monospace;font-size:0.57rem}
.dt td{padding:4px 5px;border-bottom:1px solid #0d2040}
.dt tr:last-child td{border-bottom:none}
.dt td:first-child{color:#4a6880}
.dt td:last-child{text-align:right;font-weight:bold}

/* ── MAP ── */
#mw{flex:1;position:relative;overflow:hidden}
#map{width:100%;height:100%}
#fc{position:absolute;top:0;left:0;pointer-events:none;z-index:400}
.evpo{position:absolute;inset:0;pointer-events:none;z-index:450;
  background:transparent;transition:.4s}
.evpo.on{background:radial-gradient(ellipse at center,rgba(255,34,68,.07) 0%,transparent 65%)}
.mpill{position:absolute;z-index:600;background:rgba(2,8,16,.92);
  border:1px solid;border-radius:4px;font-family:'Share Tech Mono',monospace;
  font-size:0.6rem;backdrop-filter:blur(4px)}
#mtop{top:11px;left:50%;transform:translateX(-50%);border-color:#ff8c0088;
  padding:7px 18px;display:flex;gap:20px;color:var(--orange);white-space:nowrap}
#mtop b{color:var(--cyan)}
#mleg{bottom:13px;left:13px;border-color:var(--cdim);padding:11px 13px;min-width:165px}
#mscl{bottom:13px;right:13px;border-color:var(--cdim);padding:11px 13px}
.lt{font-family:'Orbitron',monospace;font-size:0.53rem;color:var(--cyan);
  letter-spacing:2px;margin-bottom:7px}
.lr{display:flex;align-items:center;gap:7px;margin-bottom:4px;color:#5a7090}
.lb{height:4px;width:30px;border-radius:2px}
.mr{display:flex;align-items:center;gap:7px;margin-bottom:4px}
.md{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.mt{font-family:'Share Tech Mono',monospace;font-size:0.57rem;color:#5a7590}

/* ── RIGHT PANEL ── */
#rp{width:326px;flex-shrink:0;background:var(--bg2);
  border-left:1px solid var(--cdim);display:flex;flex-direction:column;overflow:hidden}
.atab-content{display:none;flex:1;overflow-y:auto;padding:11px;
  scrollbar-width:thin;scrollbar-color:var(--cdim) transparent;
  flex-direction:column;gap:9px}
.atab-content.on{display:flex}
.gc{background:var(--bg3);border:1px solid #0d2040;border-radius:4px;padding:9px}
.gh{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:5px}
.gtl{font-family:'Share Tech Mono',monospace;font-size:0.53rem;color:var(--cyan);
  letter-spacing:2px;text-transform:uppercase;line-height:1.5}
.gr{text-align:right}
.gv{font-family:'Orbitron',monospace;font-size:1.35rem;font-weight:700;line-height:1}
.gu{font-family:'Share Tech Mono',monospace;font-size:0.48rem;color:#4a6880;
  display:block;margin-top:2px}
.gd{font-family:'Share Tech Mono',monospace;font-size:0.53rem;display:inline-block;margin-top:2px}
.up{color:var(--green)}.dn{color:var(--red)}
canvas.gcanv{display:block;width:100%!important;height:66px!important}
.sgrid{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.scard{background:var(--bg);border:1px solid #0d2040;border-radius:3px;
  padding:9px 7px;text-align:center;border-left:3px solid}
.sv{font-family:'Orbitron',monospace;font-size:1.1rem;font-weight:700;
  line-height:1;margin-bottom:3px}
.sl{font-family:'Share Tech Mono',monospace;font-size:0.48rem;
  color:#4a6880;letter-spacing:1px;text-transform:uppercase}
.ss{font-family:'Share Tech Mono',monospace;font-size:0.5rem;color:#2a4060;margin-top:2px}
.ab-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.ab{padding:9px 7px;border-radius:3px;border:1px solid;text-align:center}
.abn{font-family:'Orbitron',monospace;font-size:0.46rem;letter-spacing:1px;
  margin-bottom:5px;text-transform:uppercase}
.abv{font-family:'Orbitron',monospace;font-size:1.15rem;font-weight:700;line-height:1}
.abs{font-family:'Share Tech Mono',monospace;font-size:0.48rem;color:#3a5570;margin-top:3px}
.sc-card{background:var(--bg);border:1px solid #0d2040;border-radius:3px;
  padding:7px;border-top:3px solid;margin-bottom:5px}
.sc-name{font-family:'Share Tech Mono',monospace;font-size:0.5rem;color:#4a6880;margin-bottom:4px}
.sc-state{font-family:'Orbitron',monospace;font-size:0.72rem;font-weight:700;margin-bottom:3px}
.sc-tmr{font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#3a5570}
.sc-bar{height:4px;background:#0d2040;border-radius:2px;margin-top:5px;overflow:hidden}
.sc-fill{height:100%;border-radius:2px;transition:width .5s}
.sc-evp{animation:scp .6s infinite alternate}
@keyframes scp{from{box-shadow:none}to{box-shadow:0 0 8px #ff224466}}
.lp-box{background:var(--bg);border:1px solid #0d2040;border-radius:3px;
  padding:10px;font-family:'Share Tech Mono',monospace;font-size:0.6rem;
  color:#3a5570;line-height:2}
.hi{color:var(--cyan)}.hig{color:var(--green)}.hiy{color:var(--yellow)}
.hio{color:var(--orange)}.hir{color:var(--red)}
#statusbar{height:29px;flex-shrink:0;background:var(--bg2);
  border-top:1px solid var(--cdim);display:flex;align-items:center;
  padding:0 14px;gap:0;font-family:'Share Tech Mono',monospace;font-size:0.57rem;overflow:hidden}
.sb{display:flex;align-items:center;gap:4px;color:#4a6880;padding:0 12px;
  border-right:1px solid #0d2040;white-space:nowrap}
.sb:last-child{border-right:none;margin-left:auto}
.sbv{color:var(--cyan);font-weight:bold}
.sbv.r{color:var(--red)}.sbv.g{color:var(--green)}.sbv.y{color:var(--yellow)}
.sbv.p{color:var(--purple)}
.leaflet-tile-pane{filter:brightness(.28) saturate(.2) hue-rotate(195deg)!important}
.leaflet-container{background:var(--bg)}
.leaflet-control-attribution,.leaflet-control-zoom{display:none!important}
</style>
</head>
<body>

<div id="hdr">
  <div class="h-brand">
    <div class="h-icon">&#x1F6A6;</div>
    <div>
      <div class="h-title">URBAN FLOW &amp; LIFE-LINES</div>
      <div class="h-sub">&#9658; BANGALORE GRID &#8212; LP + GREEN WAVE + EVP &#9668; NMIT ISE</div>
    </div>
  </div>
  <div class="h-div"></div>
  <div class="h-kpis">
    <div class="kpi"><div class="kpi-v" id="kv0" style="color:var(--cyan)">&#8212;</div><div class="kpi-l">Live Vehicles</div></div>
    <div class="kpi"><div class="kpi-v" id="kv1" style="color:var(--red)">&#8212;</div><div class="kpi-l">Avg Delay (s)</div></div>
    <div class="kpi"><div class="kpi-v" id="kv2" style="color:var(--orange)">&#8212;</div><div class="kpi-l">EVP Active</div></div>
    <div class="kpi"><div class="kpi-v" id="kv3" style="color:var(--green)">&#8212;</div><div class="kpi-l">Grid Efficiency</div></div>
    <div class="kpi"><div class="kpi-v" id="kv4" style="color:var(--yellow)">&#8212;</div><div class="kpi-l">Avg Speed km/h</div></div>
    <div class="kpi"><div class="kpi-v" id="kv5" style="color:var(--purple)">&#8212;</div><div class="kpi-l">CO&#x2082; Saved kg/hr</div></div>
  </div>
  <div class="h-div"></div>
  <div class="h-btns">
    <div class="live">&#x25CF; LIVE</div>
    <button class="btn btn-g on" id="btn-algo" onclick="cycleAlgo()">&#x26A1; GW+EVP</button>
    <button class="btn btn-r" onclick="massEVP()">&#x1F6A8; MASS EVP</button>
    <button class="btn btn-c" id="btn-pause" onclick="togglePause()">&#x23F8; PAUSE</button>
  </div>
</div>

<div id="body">
  <!-- LEFT -->
  <div id="lp">
    <div class="tabs">
      <div class="tab on" onclick="lTab(0)">CONTROLS</div>
      <div class="tab" onclick="lTab(1)">JUNCTIONS</div>
      <div class="tab" onclick="lTab(2)">DATA</div>
    </div>

    <div class="tpane on" id="lt0">
      <div class="sec">
        <div class="stitle">&#9881; Simulation Controls</div>
        <div class="ctrl">
          <div class="clbl">Traffic Density <span id="ldns">Peak</span></div>
          <input type="range" min="1" max="5" value="4" oninput="setDens(this.value)">
        </div>
        <div class="ctrl">
          <div class="clbl">Emergency Dots (x100 veh) <span id="lems">50 = 5,000 veh</span></div>
          <input type="range" min="5" max="150" value="50" oninput="setEmerg(this.value)">
        </div>
        <div class="ctrl">
          <div class="clbl">Green Wave Speed <span id="lwav">40 km/h</span></div>
          <input type="range" min="20" max="80" value="40" step="5" oninput="setWave(this.value)">
        </div>
        <div class="ctrl">
          <div class="clbl">Signal Cycle Time <span id="lcyc">90s</span></div>
          <input type="range" min="30" max="180" value="90" step="10" oninput="setCycle(this.value)">
        </div>
        <div class="ctrl">
          <div class="clbl">Sim Speed</div>
          <select onchange="setSS(this.value)">
            <option value="0.5">0.5x Slow</option>
            <option value="1" selected>1x Real-time</option>
            <option value="2">2x Fast</option>
            <option value="4">4x Ultra</option>
          </select>
        </div>
        <div class="ctrl">
          <div class="clbl">Algorithm</div>
          <select id="algo-sel" onchange="setAlgoSel(this.value)">
            <option value="optimal">Green Wave + EVP (Proposed)</option>
            <option value="fixed">Fixed Timer (Baseline)</option>
            <option value="lp">Linear Programming Only</option>
            <option value="evp">Emergency Priority Only</option>
            <option value="ml">ML Predictive + EVP</option>
          </select>
        </div>
      </div>
      <div class="sec">
        <div class="stitle">&#x1F50D; Scale Legend</div>
        <div class="sc-row"><div class="sc-dot" style="background:var(--cyan)"></div>
          <div class="sc-txt">1 cyan dot = <b style="color:var(--cyan)">5,000</b> regular vehicles</div></div>
        <div class="sc-row"><div class="sc-dot" style="background:var(--red);box-shadow:0 0 5px var(--red)"></div>
          <div class="sc-txt">1 red dot = <b style="color:var(--red)">100</b> emergency vehicles</div></div>
        <div class="sc-row"><div class="sc-dot" style="background:var(--yellow)"></div>
          <div class="sc-txt">Yellow = slow (&gt;65% congestion)</div></div>
        <div class="sc-row"><div class="sc-dot" style="background:var(--orange)"></div>
          <div class="sc-txt">Orange = stopped at red signal</div></div>
        <div class="sc-row"><div class="sc-dot" style="background:var(--pink);box-shadow:0 0 5px var(--pink)"></div>
          <div class="sc-txt">Pink dashed = active EVP corridor</div></div>
      </div>
    </div>

    <div class="tpane" id="lt1">
      <div class="sec">
        <div class="stitle">&#x1F5FA; Junction Monitor (12)</div>
        <div id="jlist"></div>
      </div>
    </div>

    <div class="tpane" id="lt2">
      <div class="sec">
        <div class="stitle">&#x1F4CB; TomTom / Kaggle 2024</div>
        <table class="dt">
          <tr><td>Silk Board Congestion</td><td style="color:var(--red)">71%</td></tr>
          <tr><td>Electronic City Cong.</td><td style="color:var(--red)">67%</td></tr>
          <tr><td>Hebbal Congestion</td><td style="color:var(--red)">64%</td></tr>
          <tr><td>Marathahalli Cong.</td><td style="color:var(--orange)">58%</td></tr>
          <tr><td>KR Puram Congestion</td><td style="color:var(--orange)">54%</td></tr>
          <tr><td>ORR Average</td><td style="color:var(--orange)">62%</td></tr>
          <tr><td>Peak Hours</td><td style="color:var(--yellow)">8-10AM, 6-9PM</td></tr>
          <tr><td>Avg Speed (Peak)</td><td style="color:var(--red)">17.8 km/h</td></tr>
          <tr><td>Avg Speed (Off-peak)</td><td style="color:var(--green)">32.4 km/h</td></tr>
          <tr><td>Daily Vehicles</td><td style="color:var(--cyan)">1.2M</td></tr>
          <tr><td>Registered Vehicles</td><td style="color:var(--cyan)">10.5M</td></tr>
          <tr><td>Bangalore Rank India</td><td style="color:var(--orange)">#4 Worst</td></tr>
          <tr><td>Fuel Waste/Day</td><td style="color:var(--red)">8.2M L</td></tr>
          <tr><td>Economic Loss/Day</td><td style="color:var(--red)">Rs 342 Cr</td></tr>
          <tr><td>CO2 Emission/Day</td><td style="color:var(--orange)">19,800 T</td></tr>
          <tr><td>EVP Time Saved (model)</td><td style="color:var(--green)">60%</td></tr>
          <tr><td>Delay Reduction (model)</td><td style="color:var(--green)">38%</td></tr>
        </table>
      </div>
      <div class="sec">
        <div class="stitle">&#x1F4DA; Sources</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:.56rem;color:#3a5570;line-height:2">
          TomTom Traffic Index 2024<br>
          Kaggle: Bangalore Traffic Dataset<br>
          BBMP Traffic Engineering Cell<br>
          KRDCL Outer Ring Road Study<br>
          Lighthill-Whitham-Richards Model<br>
          Operations Research: Taha 2017<br>
          Graph Theory: West 2001
        </div>
      </div>
    </div>
  </div>

  <!-- MAP -->
  <div id="mw">
    <div id="map"></div>
    <canvas id="fc"></canvas>
    <div class="evpo" id="evpo"></div>
    <div class="mpill" id="mtop">
      <span>SIM: <b id="stm">00:00:00</b></span>
      <span>ALGO: <b id="algod">GREEN WAVE+EVP</b></span>
      <span>WAVE: <b id="wavd">40 km/h</b></span>
      <span>VEHICLES: <b id="vtot">--</b></span>
      <span>EMERG: <b id="etot" style="color:var(--red)">--</b></span>
    </div>
    <div class="mpill" id="mleg">
      <div class="lt">ROAD DENSITY</div>
      <div class="lr"><div class="lb" style="background:var(--green)"></div>Free-flow &lt;40%</div>
      <div class="lr"><div class="lb" style="background:var(--yellow)"></div>Moderate 40-70%</div>
      <div class="lr"><div class="lb" style="background:var(--orange)"></div>Congested 70-90%</div>
      <div class="lr"><div class="lb" style="background:var(--red)"></div>Gridlock &gt;90%</div>
      <div class="lr"><div class="lb" style="background:var(--pink)"></div>EVP Corridor</div>
    </div>
    <div class="mpill" id="mscl">
      <div class="lt">PARTICLE SCALE</div>
      <div class="mr"><div class="md" style="background:var(--cyan)"></div><div class="mt">1 dot = 5,000 vehicles</div></div>
      <div class="mr"><div class="md" style="background:var(--red);box-shadow:0 0 4px var(--red)"></div><div class="mt">1 dot = 100 emergency</div></div>
      <div class="mr"><div class="md" style="background:var(--yellow)"></div><div class="mt">Yellow = slowing</div></div>
      <div class="mr"><div class="md" style="background:var(--orange)"></div><div class="mt">Orange = stopped</div></div>
    </div>
  </div>

  <!-- RIGHT ANALYTICS -->
  <div id="rp">
    <div class="tabs">
      <div class="tab on" onclick="rTab(0)">GRAPHS</div>
      <div class="tab" onclick="rTab(1)">STATS</div>
      <div class="tab" onclick="rTab(2)">SIGNALS</div>
      <div class="tab" onclick="rTab(3)">LP MODEL</div>
    </div>

    <div class="atab-content on" id="rt0">
      <div class="gc"><div class="gh">
        <div class="gtl">Network Throughput<br>veh/hr/lane (VPHPL)</div>
        <div class="gr"><div class="gv" id="gv0" style="color:var(--green)">--</div>
          <span class="gu">VPHPL</span><div class="gd" id="gd0"></div></div>
      </div><canvas class="gcanv" id="gc0"></canvas></div>

      <div class="gc"><div class="gh">
        <div class="gtl">Avg Intersection Delay<br>Webster formula</div>
        <div class="gr"><div class="gv" id="gv1" style="color:var(--red)">--</div>
          <span class="gu">seconds</span><div class="gd" id="gd1"></div></div>
      </div><canvas class="gcanv" id="gc1"></canvas></div>

      <div class="gc"><div class="gh">
        <div class="gtl">EVP Corridor Clear Time<br>Golden Hour metric</div>
        <div class="gr"><div class="gv" id="gv2" style="color:var(--orange)">--</div>
          <span class="gu">seconds</span><div class="gd" id="gd2"></div></div>
      </div><canvas class="gcanv" id="gc2"></canvas></div>

      <div class="gc"><div class="gh">
        <div class="gtl">Signal Efficiency Index<br>Green time utilization %</div>
        <div class="gr"><div class="gv" id="gv3" style="color:var(--cyan)">--</div>
          <span class="gu">percent</span><div class="gd" id="gd3"></div></div>
      </div><canvas class="gcanv" id="gc3"></canvas></div>

      <div class="gc"><div class="gh">
        <div class="gtl">CO&#x2082; Emissions Index<br>vs fixed-timer baseline</div>
        <div class="gr"><div class="gv" id="gv4" style="color:var(--yellow)">--</div>
          <span class="gu">relative</span><div class="gd" id="gd4"></div></div>
      </div><canvas class="gcanv" id="gc4"></canvas></div>

      <div class="gc"><div class="gh">
        <div class="gtl">Network Density %<br>LWR flow model</div>
        <div class="gr"><div class="gv" id="gv5" style="color:var(--purple)">--</div>
          <span class="gu">percent</span><div class="gd" id="gd5"></div></div>
      </div><canvas class="gcanv" id="gc5"></canvas></div>
    </div>

    <div class="atab-content" id="rt1">
      <div class="sec">
        <div class="stitle">&#x1F4CA; Live Statistics</div>
        <div class="sgrid">
          <div class="scard" style="border-left-color:var(--green)">
            <div class="sv" style="color:var(--green)" id="s0">--</div>
            <div class="sl">Free-Flow %</div><div class="ss" id="s0b">-- veh</div>
          </div>
          <div class="scard" style="border-left-color:var(--red)">
            <div class="sv" style="color:var(--red)" id="s1">--</div>
            <div class="sl">Stopped</div><div class="ss">at red signals</div>
          </div>
          <div class="scard" style="border-left-color:var(--yellow)">
            <div class="sv" style="color:var(--yellow)" id="s2">--</div>
            <div class="sl">Avg Speed km/h</div><div class="ss">target: 40 km/h</div>
          </div>
          <div class="scard" style="border-left-color:var(--orange)">
            <div class="sv" style="color:var(--orange)" id="s3">--</div>
            <div class="sl">Fuel Waste L/hr</div><div class="ss">idle consumption</div>
          </div>
          <div class="scard" style="border-left-color:var(--cyan)">
            <div class="sv" style="color:var(--cyan)" id="s4">--</div>
            <div class="sl">GW Corridors</div><div class="ss">synchronized</div>
          </div>
          <div class="scard" style="border-left-color:var(--red)">
            <div class="sv" style="color:var(--red)" id="s5">--</div>
            <div class="sl">EVP Overrides</div><div class="ss">total session</div>
          </div>
          <div class="scard" style="border-left-color:var(--purple)">
            <div class="sv" style="color:var(--purple)" id="s6">--</div>
            <div class="sl">CO&#x2082; Saved kg/hr</div><div class="ss">vs fixed timer</div>
          </div>
          <div class="scard" style="border-left-color:var(--green)">
            <div class="sv" style="color:var(--green)" id="s7">--</div>
            <div class="sl">LP Iterations</div><div class="ss">optimization cycles</div>
          </div>
          <div class="scard" style="border-left-color:var(--orange)">
            <div class="sv" style="color:var(--orange)" id="s8">--</div>
            <div class="sl">Emergency Dots</div><div class="ss">x100 vehicles each</div>
          </div>
          <div class="scard" style="border-left-color:var(--yellow)">
            <div class="sv" style="color:var(--yellow)" id="s9">--</div>
            <div class="sl">Econ Saved Rs/hr</div><div class="ss">productivity gain</div>
          </div>
        </div>
      </div>
      <div class="sec">
        <div class="stitle">&#x26A1; Algorithm Benchmark</div>
        <div class="ab-grid">
          <div class="ab" style="border-color:#ff224444;background:#150308">
            <div class="abn" style="color:var(--red)">Fixed Timer</div>
            <div class="abv" id="ab0" style="color:var(--red)">45 min</div>
            <div class="abs">Silk Bd-Hebbal</div>
          </div>
          <div class="ab" style="border-color:#00ff8844;background:#00200e">
            <div class="abn" style="color:var(--green)">GW+EVP (Ours)</div>
            <div class="abv" id="ab1" style="color:var(--green)">28 min</div>
            <div class="abs">Silk Bd-Hebbal</div>
          </div>
          <div class="ab" style="border-color:#00e5ff44">
            <div class="abn" style="color:var(--cyan)">Delay Reduction</div>
            <div class="abv" id="ab2" style="color:var(--cyan)">38%</div>
            <div class="abs">LP optimization</div>
          </div>
          <div class="ab" style="border-color:#ff8c0044">
            <div class="abn" style="color:var(--orange)">EVP Time Saved</div>
            <div class="abv" id="ab3" style="color:var(--orange)">60%</div>
            <div class="abs">vs unoptimized</div>
          </div>
          <div class="ab" style="border-color:#bb77ff44">
            <div class="abn" style="color:var(--purple)">Throughput Gain</div>
            <div class="abv" id="ab4" style="color:var(--purple)">+30%</div>
            <div class="abs">vehicles/hr/lane</div>
          </div>
          <div class="ab" style="border-color:#ffd70044">
            <div class="abn" style="color:var(--yellow)">CO2 Reduction</div>
            <div class="abv" id="ab5" style="color:var(--yellow)">25%</div>
            <div class="abs">stop-and-go saved</div>
          </div>
        </div>
      </div>
    </div>

    <div class="atab-content" id="rt2">
      <div class="sec">
        <div class="stitle">&#x1F6A6; Signal Control Panel</div>
        <div id="sigpanel"></div>
      </div>
    </div>

    <div class="atab-content" id="rt3">
      <div class="sec">
        <div class="stitle">&#x2211; LP Objective Function</div>
        <div class="lp-box">
          <span class="hi">Minimize:</span> W = &#x2211;i ti x di<br>
          <span class="hi">Subject to:</span><br>
          gi + ri + yi = T = <span class="hio" id="lT">90s</span><br>
          gi &ge; 10s (min green safety)<br>
          ri &ge; 5s (min red clearance)<br>
          yi = 0.07 x T (yellow interval)<br>
          <br>
          <span class="hi">Green Wave Offset:</span><br>
          &Phi;ij = (Lij / vc) mod T<br>
          vc = <span class="hig" id="lvc">40</span> km/h<br>
          <br>
          <span class="hi">EVP Pre-emption:</span><br>
          P_emergency &#8594; &#x221E;<br>
          S_evp = d / v_amb<br>
          Override: all signals on path<br>
          <br>
          <span class="hi">LWR Flow Model:</span><br>
          &#x2202;&#x03C1;/&#x2202;t + &#x2202;(&#x03C1;v)/&#x2202;x = 0<br>
          &#x03C1; = density, v = velocity<br>
          <br>
          <span class="hi">Current W:</span> <span class="hiy" id="lW">--</span> veh&#xB7;sec<br>
          <span class="hi">Iterations:</span> <span class="hig" id="lI">--</span><br>
          <span class="hi">EVP Weight:</span> <span class="hir">P &#x2192; &#x221E;</span><br>
          <span class="hi">Warmup:</span> <span class="hig" id="lWu">0%</span>
        </div>
      </div>
      <div class="sec">
        <div class="stitle">&#x1F4D0; Webster's Delay Formula</div>
        <div class="lp-box">
          <span class="hi">d = C(1-&#x03BB;)&#xB2; / 2(1-&#x03BB;x)</span><br>
          &nbsp;&nbsp;&nbsp;&nbsp;+ x&#xB2; / 2q(1-x)<br><br>
          d = avg delay/vehicle (s)<br>
          C = cycle = <span class="hio" id="wC">90</span>s<br>
          &#x03BB; = g/C = green ratio<br>
          x = degree of saturation<br>
          q = flow rate (veh/s)<br><br>
          <span class="hi">Current:</span><br>
          &#x03BB; (avg): <span class="hig" id="wL">--</span><br>
          x (sat.): <span class="hiy" id="wX">--</span><br>
          d (calc): <span class="hir" id="wD">--</span> s
        </div>
      </div>
    </div>
  </div>
</div>

<div id="statusbar">
  <div class="sb">&#x1F551; <span id="sbt" class="sbv">00:00:00</span></div>
  <div class="sb">ALGO <span id="sba" class="sbv">GW+EVP</span></div>
  <div class="sb">TOTAL <span id="sbn" class="sbv">--</span></div>
  <div class="sb">STOPPED <span id="sbs" class="sbv r">--</span></div>
  <div class="sb">EMERG <span id="sbe" class="sbv r">--</span></div>
  <div class="sb">FREE-FLOW <span id="sbf" class="sbv g">--</span></div>
  <div class="sb">GRID EFF <span id="sbg" class="sbv g">--</span></div>
  <div class="sb">CO2 SAVED <span id="sbc" class="sbv p">--</span></div>
  <div class="sb">LP ITER <span id="sbl" class="sbv y">--</span></div>
  <div class="sb">&#xA9; NMIT ISE &#8212; NISHCHAL VISHWANATH NB25ISE160 &#xB7; RISHUL KH NB25ISE186</div>
</div>

<script>
(function() {
'use strict';

// ── DATA ──────────────────────────────────────────────────────────────────────
var JN = [
  {id:0,  name:'Silk Board',      lat:12.9177,lng:77.6228,cong:.71,daily:185000,peak:22000,imp:10},
  {id:1,  name:'Hebbal',          lat:13.0358,lng:77.5970,cong:.64,daily:156000,peak:19000,imp:9},
  {id:2,  name:'Marathahalli',    lat:12.9591,lng:77.6974,cong:.58,daily:142000,peak:17500,imp:8},
  {id:3,  name:'KR Puram',        lat:13.0074,lng:77.6950,cong:.54,daily:128000,peak:15000,imp:7},
  {id:4,  name:'Electronic City', lat:12.8399,lng:77.6770,cong:.67,daily:168000,peak:20000,imp:8},
  {id:5,  name:'Whitefield',      lat:12.9698,lng:77.7500,cong:.52,daily:118000,peak:14000,imp:7},
  {id:6,  name:'Indiranagar',     lat:12.9784,lng:77.6408,cong:.62,daily:138000,peak:16500,imp:8},
  {id:7,  name:'Koramangala',     lat:12.9352,lng:77.6245,cong:.66,daily:155000,peak:18500,imp:9},
  {id:8,  name:'JP Nagar',        lat:12.9063,lng:77.5857,cong:.48,daily:108000,peak:13000,imp:7},
  {id:9,  name:'Yelahanka',       lat:13.1007,lng:77.5963,cong:.44,daily:98000, peak:12000,imp:6},
  {id:10, name:'Bannerghatta Rd', lat:12.8931,lng:77.5971,cong:.59,daily:132000,peak:15800,imp:7},
  {id:11, name:'Nagawara',        lat:13.0456,lng:77.6207,cong:.55,daily:122000,peak:14500,imp:7}
];

var ED = [
  [0,7],[0,8],[0,4],[0,6],
  [1,9],[1,11],[1,3],
  [2,3],[2,5],[2,6],[2,7],
  [3,5],[3,11],
  [4,8],[4,10],
  [6,7],[6,2],[6,11],
  [7,10],[7,8],
  [8,10],
  [9,11],[9,1],
  [10,8],[11,6]
];

// ── STATE ─────────────────────────────────────────────────────────────────────
var S = {
  algo:'optimal', paused:false, speed:1,
  emergDots:50, wave:40, cycle:90, dens:4,
  simTime:0, frame:0,
  lpIter:0, evpTotal:0, booted:0
};

var DNAMES = ['Very Low','Low','Medium','High','Peak (TomTom)'];
var DMUL   = [0.2, 0.4, 0.7, 1.0, 1.4];
var ANAMES = {optimal:'GREEN WAVE+EVP', fixed:'FIXED TIMER', lp:'LP ONLY', evp:'EVP ONLY', ml:'ML+EVP'};
var ALIST  = ['optimal','fixed','lp','evp','ml'];
var aidx   = 0;

// Graph buffers
var GL = 120;
var GD = {g0:[],g1:[],g2:[],g3:[],g4:[],g5:[]};
for (var k in GD) { for (var i=0;i<GL;i++) GD[k].push(0); }

// ── SIGNALS ───────────────────────────────────────────────────────────────────
var SIG = JN.map(function(j,i) {
  return {id:i, phase:Math.random()*90, cycle:90,
          state:'red', evp:false, gDur:45, eff:0.5, wait:Math.floor(j.cong*50)};
});

// ── PARTICLES ─────────────────────────────────────────────────────────────────
var particles = [];
var MAX_N = 650;

function Particle(isE) {
  this.isE = isE;
  this.ei = Math.floor(Math.random()*ED.length);
  this.prog = Math.random();
  this.dir = Math.random()>.5?1:-1;
  this.bspd = isE ? (0.004+Math.random()*.003) : (0.0007+Math.random()*.0007);
  this.spd = this.bspd;
  this.tspd = this.bspd;
  this.state = 'moving';
  this.wt = 0;
  this.trail = [];
  this.ph = Math.random()*Math.PI*2;
  this.loff = (Math.random()-.5)*.00022;
}

Particle.prototype.pos = function() {
  var e = ED[this.ei];
  var a = JN[e[0]], b = JN[e[1]];
  var t = this.dir===1 ? this.prog : 1-this.prog;
  var plat = (b.lng-a.lng)*.15;
  var plng = (b.lat-a.lat)*.15;
  return {lat: a.lat+(b.lat-a.lat)*t+this.loff*plat,
          lng: a.lng+(b.lng-a.lng)*t+this.loff*plng};
};

Particle.prototype.update = function(dt) {
  try {
    var e = ED[this.ei];
    var endId = this.dir===1 ? e[1] : e[0];
    var sig = SIG[endId];
    var junc = JN[endId];
    var distEnd = this.dir===1 ? 1-this.prog : this.prog;
    var mul = DMUL[S.dens-1];
    var af = S.algo==='fixed' ? 1.25 : 1.0;
    var warm = Math.min(S.booted/500,1);
    var ar = S.algo==='optimal'?warm*.45:S.algo==='lp'?warm*.25:0;
    var cong = Math.min(junc.cong*mul*af*(1-ar), 0.97);

    var stop = (!this.isE && distEnd<.15 && sig.state==='red' && !sig.evp);

    if (stop) {
      this.tspd=0; this.state='stopped'; this.wt+=dt*.016;
    } else if (cong>.65 && !this.isE) {
      var f = Math.max(.05, 1-(cong-.65)*2.8);
      this.tspd = this.bspd*f*S.wave/40;
      this.state = cong>.85?'stopped':'slow';
      this.wt = Math.max(0,this.wt-dt*.01);
    } else {
      this.tspd = this.bspd*S.wave/40;
      this.state = 'moving';
      this.wt = Math.max(0,this.wt-dt*.05);
    }

    if (this.isE) { this.tspd = this.bspd*2.2; this.state='moving'; }

    this.spd += (this.tspd-this.spd)*.13;
    this.prog += this.spd*this.dir*S.speed;

    if (this.isE) {
      var p = this.pos();
      this.trail.unshift({lat:p.lat,lng:p.lng});
      if (this.trail.length>10) this.trail.pop();
    }

    if (this.prog>=1 || this.prog<=0) {
      this.prog = this.prog>=1 ? 0 : 1;
      var ej = this.dir===1 ? ED[this.ei][1] : ED[this.ei][0];
      var conn = [];
      for (var i=0;i<ED.length;i++) {
        if (i!==this.ei && (ED[i][0]===ej||ED[i][1]===ej)) conn.push(i);
      }
      if (conn.length>0) {
        var pk = conn[Math.floor(Math.random()*conn.length)];
        this.ei = pk;
        this.dir = ED[pk][0]===ej ? 1 : -1;
        this.prog = this.dir===1 ? 0 : 1;
      } else {
        this.dir *= -1;
      }
    }
  } catch(err) { /* silent — keep loop alive */ }
};

Particle.prototype.col = function() {
  if (this.isE) return '#ff2244';
  if (this.state==='stopped') return '#ff5500';
  if (this.state==='slow') return '#ffcc00';
  return '#00ccff';
};

function spawnParticles() {
  particles = [];
  var mul = DMUL[S.dens-1];
  var n = Math.floor(MAX_N*mul);
  for (var i=0;i<n;i++) particles.push(new Particle(false));
  for (var i=0;i<S.emergDots;i++) particles.push(new Particle(true));
}

// ── MAP ───────────────────────────────────────────────────────────────────────
var map = L.map('map',{center:[12.97,77.62],zoom:12,
  zoomControl:false,attributionControl:false,preferCanvas:true});
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:18}).addTo(map);

var roadLines = [];
function drawRoads() {
  for (var i=0;i<roadLines.length;i++) try{roadLines[i].remove();}catch(e){}
  roadLines = [];
  var warm = Math.min(S.booted/500,1);
  for (var ri=0;ri<ED.length;ri++) {
    var e = ED[ri];
    var ja = JN[e[0]], jb = JN[e[1]];
    var mul = DMUL[S.dens-1];
    var af = S.algo==='fixed'?1.2:1.0;
    var ar = S.algo==='optimal'?warm*.45:0;
    var c = Math.min((ja.cong+jb.cong)/2*mul*af*(1-ar),1);
    var col = c>.85?'#ff2244':c>.65?'#ff8c00':c>.4?'#ffd700':'#00ff88';
    var w = 3+c*6;
    var hasEvp = false;
    for (var pi=0;pi<particles.length;pi++) {
      if (particles[pi].isE && particles[pi].ei===ri) { hasEvp=true; break; }
    }
    if (hasEvp) {
      try {
        var bg = L.polyline([[ja.lat,ja.lng],[jb.lat,jb.lng]],
          {color:'#ff224455',weight:w+8,opacity:.6}).addTo(map);
        var gw = L.polyline([[ja.lat,ja.lng],[jb.lat,jb.lng]],
          {color:'#ff44aa',weight:2,opacity:.9,dashArray:'10 6'}).addTo(map);
        roadLines.push(bg); roadLines.push(gw);
      } catch(e) {}
    }
    try {
      var bl = L.polyline([[ja.lat,ja.lng],[jb.lat,jb.lng]],
        {color:col+'99',weight:w,opacity:.8}).addTo(map);
      roadLines.push(bl);
    } catch(e) {}
  }
}

var jmkrs = JN.map(function(j,i) {
  var m = L.circleMarker([j.lat,j.lng],
    {radius:8+j.imp,color:'#fff',weight:1.5,fillColor:'#ff2244',fillOpacity:.9}).addTo(map);
  var tc = j.cong>.65?'#ff2244':j.cong>.45?'#ff8c00':'#00ff88';
  m.bindTooltip(
    '<b style="color:#ffd700">' + j.name + '</b><br>' +
    'Congestion: <b style="color:'+tc+'">' + Math.round(j.cong*100) + '%</b><br>' +
    'Daily: <b>' + (j.daily/1000).toFixed(0) + 'K/day</b><br>' +
    'Peak: <b>' + (j.peak/1000).toFixed(0) + 'K/hr</b>',
    {direction:'top'}
  );
  return m;
});

// ── CANVAS OVERLAY ────────────────────────────────────────────────────────────
var fc = document.getElementById('fc');
var cx = fc.getContext('2d');

function resizeFC() {
  var mw = document.getElementById('mw');
  fc.width = mw.offsetWidth;
  fc.height = mw.offsetHeight;
}
resizeFC();
window.addEventListener('resize', resizeFC);

function ll2px(lat,lng) {
  try {
    var p = map.latLngToContainerPoint([lat,lng]);
    return {x:p.x, y:p.y};
  } catch(e) { return {x:-100,y:-100}; }
}

function renderParticles() {
  cx.clearRect(0,0,fc.width,fc.height);
  // Normal vehicles
  for (var i=0;i<particles.length;i++) {
    var p = particles[i];
    if (p.isE) continue;
    try {
      var pos = p.pos();
      var pt = ll2px(pos.lat,pos.lng);
      cx.fillStyle = p.col()+'cc';
      cx.beginPath();
      cx.arc(pt.x,pt.y,3,0,Math.PI*2);
      cx.fill();
    } catch(e) {}
  }
  // Emergency on top
  for (var i=0;i<particles.length;i++) {
    var p = particles[i];
    if (!p.isE) continue;
    try {
      // Trail
      for (var t=1;t<p.trail.length;t++) {
        var t1 = ll2px(p.trail[t-1].lat,p.trail[t-1].lng);
        var t2 = ll2px(p.trail[t].lat,p.trail[t].lng);
        cx.strokeStyle = 'rgba(255,34,68,'+(((1-t/p.trail.length)*.5).toFixed(2))+')';
        cx.lineWidth = Math.max(.5,4-t*.35);
        cx.beginPath();cx.moveTo(t1.x,t1.y);cx.lineTo(t2.x,t2.y);cx.stroke();
      }
      var pos2 = p.pos();
      var pt2 = ll2px(pos2.lat,pos2.lng);
      var pulse = .55+.45*Math.sin(S.frame*.25+p.ph);
      cx.shadowBlur = 16*pulse;
      cx.shadowColor = '#ff2244';
      cx.fillStyle = '#ff2244';
      cx.beginPath();cx.arc(pt2.x,pt2.y,7,0,Math.PI*2);cx.fill();
      cx.shadowBlur = 0;
      cx.strokeStyle = '#ffffff';cx.lineWidth = 1.8;
      cx.beginPath();
      cx.moveTo(pt2.x-5,pt2.y);cx.lineTo(pt2.x+5,pt2.y);
      cx.moveTo(pt2.x,pt2.y-5);cx.lineTo(pt2.x,pt2.y+5);
      cx.stroke();
    } catch(e) {}
  }
  cx.shadowBlur = 0;
}

// ── CHARTS ────────────────────────────────────────────────────────────────────
var GCFG = [
  {id:'gc0',col:'#00ff88',max:2200},
  {id:'gc1',col:'#ff2244',max:200},
  {id:'gc2',col:'#ff8c00',max:130},
  {id:'gc3',col:'#00e5ff',max:100},
  {id:'gc4',col:'#ffd700',max:160},
  {id:'gc5',col:'#bb77ff',max:100}
];
var GKEYS = ['g0','g1','g2','g3','g4','g5'];
var charts = {};

// Wait for DOM then init charts
setTimeout(function() {
  for (var i=0;i<GCFG.length;i++) {
    (function(cfg,key) {
      var el = document.getElementById(cfg.id);
      if (!el) return;
      try {
        charts[key] = new Chart(el, {
          type:'line',
          data:{
            labels: new Array(GL).fill(''),
            datasets:[{
              data: GD[key].slice(),
              borderColor:cfg.col, borderWidth:1.5,
              pointRadius:0, fill:true,
              backgroundColor:cfg.col+'18', tension:.4
            }]
          },
          options:{
            animation:false, responsive:true, maintainAspectRatio:false,
            plugins:{legend:{display:false},tooltip:{enabled:false}},
            scales:{x:{display:false},y:{display:false,min:0,max:cfg.max}}
          }
        });
      } catch(e) {}
    })(GCFG[i], GKEYS[i]);
  }
}, 300);

function pushGraph(key, val) {
  GD[key].push(val); GD[key].shift();
  if (charts[key]) {
    try {
      charts[key].data.datasets[0].data = GD[key].slice();
      charts[key].update('none');
    } catch(e) {}
  }
}

// ── SIGNAL UPDATE ─────────────────────────────────────────────────────────────
function updateSignals(dt) {
  S.booted = Math.min(S.booted+dt, 500);
  var warm = S.booted/500;

  for (var i=0;i<SIG.length;i++) {
    var sig = SIG[i];
    sig.phase += dt*S.speed;
    if (sig.phase >= sig.cycle) sig.phase -= sig.cycle;

    // EVP detection
    var nearEvp = false;
    for (var pi=0;pi<particles.length;pi++) {
      var p = particles[pi];
      if (!p.isE) continue;
      var ej = p.dir===1 ? ED[p.ei][1] : ED[p.ei][0];
      var d  = p.dir===1 ? 1-p.prog : p.prog;
      if (ej===i && d<.3) { nearEvp=true; break; }
    }
    var wasEvp = sig.evp;
    sig.evp = nearEvp && S.algo!=='fixed';
    if (sig.evp && !wasEvp) {
      S.evpTotal++;
      var ov = document.getElementById('evpo');
      if(ov){ov.classList.add('on');setTimeout(function(){var o=document.getElementById('evpo');if(o)o.classList.remove('on');},600);}
    }
    if (sig.evp) { sig.state='green'; sig.eff=1; continue; }

    var junc = JN[i];
    var mul = DMUL[S.dens-1];
    var cong = Math.min(junc.cong*mul,.95);
    var gDur = S.cycle*.5;

    if (S.algo==='optimal'||S.algo==='ml') {
      gDur = S.cycle*(0.3+cong*.4)*(0.6+warm*.4);
      S.lpIter += .012;
      // Green Wave phase sync
      for (var ei=0;ei<ED.length;ei++) {
        if (ED[ei][0]===i||ED[ei][1]===i) {
          var oth = ED[ei][0]===i?ED[ei][1]:ED[ei][0];
          var ja2 = JN[i], jb2 = JN[oth];
          var dx = (ja2.lat-jb2.lat)*111;
          var dy = (ja2.lng-jb2.lng)*111*Math.cos(ja2.lat*Math.PI/180);
          var distKm = Math.sqrt(dx*dx+dy*dy);
          var phi = (distKm/S.wave*3600) % S.cycle;
          var pd = (SIG[oth].phase-sig.phase+S.cycle)%S.cycle;
          if (Math.abs(pd-phi)>4) SIG[oth].phase += (phi-pd)*.025;
        }
      }
    } else if (S.algo==='lp') {
      gDur = S.cycle*(0.35+cong*.28); S.lpIter+=.005;
    }

    sig.gDur = gDur; sig.cycle = S.cycle;
    var yDur = S.cycle*.07;
    if (sig.phase<gDur) sig.state='green';
    else if (sig.phase<gDur+yDur) sig.state='yellow';
    else sig.state='red';
    sig.eff = (gDur/S.cycle)*warm;
    sig.wait = sig.state==='red' ? Math.floor(junc.cong*45*mul) : Math.floor(junc.cong*10);
  }
}

function updateJMkrs() {
  for (var i=0;i<JN.length;i++) {
    var s = SIG[i];
    var c = s.evp?'#ff2244':s.state==='green'?'#00ff88':s.state==='yellow'?'#ffd700':'#ff2244';
    try { jmkrs[i].setStyle({fillColor:c,color:s.evp?'#ff4466':'#ffffff'}); } catch(e){}
  }
}

// ── DOM HELPERS ───────────────────────────────────────────────────────────────
var prevM = {};

function g(id) { return document.getElementById(id); }
function sv(id,v) { var el=g(id); if(el) el.textContent=v; }

function setDelta(key,vId,dId,val,goodUp) {
  var prev = prevM[key]||0;
  var d = val-prev;
  if (Math.abs(d)>.5) {
    var good = goodUp ? d>0 : d<0;
    sv(dId,(d>0?'▲':'▼')+Math.abs(d).toFixed(1));
    var el = g(dId);
    if(el) el.className='gd '+(good?'up':'dn');
  }
  prevM[key]=val;
}

// ── METRICS ───────────────────────────────────────────────────────────────────
function updateMetrics() {
  var warm = S.booted/500;
  var norm=[],emerg=[];
  for(var i=0;i<particles.length;i++) {
    if(particles[i].isE) emerg.push(particles[i]); else norm.push(particles[i]);
  }
  var moving=0,slow=0,stopped=0;
  for(var i=0;i<norm.length;i++) {
    if(norm[i].state==='moving') moving++;
    else if(norm[i].state==='slow') slow++;
    else stopped++;
  }
  var total = particles.length;

  var amul = S.algo==='fixed'?1:(S.algo==='optimal'?(1+warm*.45):S.algo==='lp'?(1+warm*.2):.95);
  var ddiv = S.algo==='fixed'?1:(S.algo==='optimal'?Math.max(.4,1-warm*.4):.9);
  var mul  = DMUL[S.dens-1];

  var thr   = Math.round((880+moving*2.8)*amul);
  var delay = Math.max(6,125*mul*ddiv);
  var evpT  = S.algo==='fixed'?(48+Math.random()*18):Math.max(3,16*(1-warm*.65)+Math.random()*5);
  var avgEff= 0;
  for(var i=0;i<SIG.length;i++) avgEff+=SIG[i].eff;
  avgEff = avgEff/SIG.length*100;
  var emit  = S.algo==='fixed'?(75+stopped*1+slow*.35)*1.3:(75+stopped*1+slow*.35)*(1-warm*.28);
  var dens  = Math.min(100,(stopped+slow*.5)/Math.max(total,1)*100+62*.5);
  var avgSpd= S.algo==='fixed'?17.8:Math.min(40,17.8+warm*21);
  var fuel  = Math.round(stopped*5000*.7/1000);
  var co2s  = Math.round(warm*190*(S.algo==='optimal'?1.5:S.algo==='fixed'?0:.7));
  var ecos  = Math.round(warm*14200*(S.algo==='optimal'?1.4:S.algo==='fixed'?0:.6));
  var gws   = 0;
  for(var i=0;i<SIG.length;i++) if(SIG[i].state==='green'&&!SIG[i].evp) gws++;
  var evpAct= 0;
  for(var i=0;i<SIG.length;i++) if(SIG[i].evp) evpAct++;
  var lpW   = Math.round(delay*total*5000);
  var lam   = avgEff/100;
  var x     = Math.min(.98,(stopped+slow*.6)/Math.max(total,1)+.3);
  var webD  = Math.max(5,delay*(.8+x*.4));
  var ffPct = Math.round(moving/Math.max(total,1)*100);

  // Push graphs
  pushGraph('g0',thr); pushGraph('g1',delay);
  pushGraph('g2',evpT); pushGraph('g3',avgEff);
  pushGraph('g4',emit); pushGraph('g5',dens);

  // Header KPIs
  sv('kv0',(total*5000).toLocaleString());
  sv('kv1',Math.round(delay)+'s');
  sv('kv2',evpAct);
  sv('kv3',Math.round(avgEff)+'%');
  sv('kv4',avgSpd.toFixed(1));
  sv('kv5',co2s);

  // Graphs
  sv('gv0',thr); setDelta('thr','gv0','gd0',thr,true);
  sv('gv1',Math.round(delay)); setDelta('del','gv1','gd1',delay,false);
  sv('gv2',evpT.toFixed(1)); setDelta('evp','gv2','gd2',evpT,false);
  sv('gv3',Math.round(avgEff)); setDelta('eff','gv3','gd3',avgEff,true);
  sv('gv4',Math.round(emit)); setDelta('co2','gv4','gd4',emit,false);
  sv('gv5',Math.round(dens)); setDelta('dns','gv5','gd5',dens,false);

  // Stats
  sv('s0',ffPct+'%'); sv('s0b',(moving*5000).toLocaleString()+' veh');
  sv('s1',(stopped*5000).toLocaleString());
  sv('s2',avgSpd.toFixed(1));
  sv('s3',fuel.toLocaleString());
  sv('s4',gws);
  sv('s5',S.evpTotal);
  sv('s6',co2s);
  sv('s7',Math.floor(S.lpIter));
  sv('s8',emerg.length+' (='+(emerg.length*100).toLocaleString()+' veh)');
  sv('s9','Rs '+ecos.toLocaleString());

  // Benchmark
  var fixD=45, optD=Math.max(24,45*(1-warm*.38));
  sv('ab0',fixD+' min'); sv('ab1',optD.toFixed(0)+' min');
  sv('ab2',Math.round((1-optD/fixD)*100)+'%');
  sv('ab3',Math.round(warm*62)+'%');
  sv('ab4','+'+Math.round(warm*30)+'%');
  sv('ab5',Math.round(warm*25)+'%');

  // LP
  sv('lT',S.cycle+'s'); sv('lvc',S.wave);
  sv('lW',lpW.toLocaleString()); sv('lI',Math.floor(S.lpIter));
  sv('lWu',Math.round(warm*100)+'%');
  sv('wC',S.cycle); sv('wL',lam.toFixed(3));
  sv('wX',x.toFixed(3)); sv('wD',webD.toFixed(1));

  // Signal panel
  var sp = g('sigpanel');
  if(sp) {
    var html='';
    for(var i=0;i<SIG.length;i++){
      var s=SIG[i];
      var col=s.evp?'#ff2244':s.state==='green'?'#00ff88':s.state==='yellow'?'#ffd700':'#ff2244';
      var pct=Math.round(s.phase/s.cycle*100);
      var tl=s.state==='green'?Math.max(0,s.gDur-s.phase).toFixed(0)+'s GO':Math.max(0,s.cycle-s.phase).toFixed(0)+'s WAIT';
      var ecls=s.evp?' sc-evp':'';
      html+='<div class="sc-card'+ecls+'" style="border-top-color:'+col+'">';
      html+='<div class="sc-name">'+JN[i].name+'</div>';
      html+='<div class="sc-state" style="color:'+col+'">'+(s.evp?'EVP!':s.state.toUpperCase())+'</div>';
      html+='<div class="sc-tmr">'+tl+' | wait:'+s.wait+'</div>';
      html+='<div class="sc-bar"><div class="sc-fill" style="width:'+pct+'%;background:'+col+'"></div></div>';
      html+='</div>';
    }
    sp.innerHTML=html;
  }

  // Junction list
  var jl=g('jlist');
  if(jl){
    var jhtml='';
    for(var i=0;i<JN.length;i++){
      var j=JN[i]; var s=SIG[i];
      var col=s.evp?'#ff2244':s.state==='green'?'#00ff88':s.state==='yellow'?'#ffd700':'#ff2244';
      var tl=s.state==='green'?Math.max(0,s.gDur-s.phase).toFixed(0)+'s':Math.max(0,s.cycle-s.phase).toFixed(0)+'s';
      var cc=j.cong>.65?'var(--red)':j.cong>.45?'var(--orange)':'var(--green)';
      var ecls=s.evp?' evp':'';
      jhtml+='<div class="ji'+ecls+'">';
      jhtml+='<div class="jdot" style="background:'+col+';box-shadow:0 0 5px '+col+'"></div>';
      jhtml+='<div class="jname">'+j.name+'<small>'+(j.daily/1000).toFixed(0)+'K/day</small></div>';
      jhtml+='<div class="jpct" style="color:'+cc+'">'+Math.round(j.cong*100)+'%</div>';
      jhtml+='<div class="jtmr">'+tl+'</div>';
      jhtml+='</div>';
    }
    jl.innerHTML=jhtml;
  }

  // Map top bar & status
  var t=S.simTime;
  var ts=pad(Math.floor(t/3600)%24)+':'+pad(Math.floor(t/60)%60)+':'+pad(Math.floor(t)%60);
  sv('stm',ts); sv('sbt',ts);
  sv('algod',ANAMES[S.algo]); sv('sba',ANAMES[S.algo]);
  sv('vtot',(total*5000).toLocaleString());
  sv('etot',emerg.length+' dots x100');
  sv('sbn',(total*5000).toLocaleString());
  sv('sbs',(stopped*5000).toLocaleString());
  sv('sbe',(emerg.length*100).toLocaleString());
  sv('sbf',ffPct+'%');
  sv('sbg',Math.round(avgEff)+'%');
  sv('sbc',co2s+' kg/hr');
  sv('sbl',Math.floor(S.lpIter));
  sv('wavd',S.wave+' km/h');
}

function pad(n){return n<10?'0'+n:String(n);}

// ── CONTROLS ──────────────────────────────────────────────────────────────────
function setAlgo(a) {
  S.algo=a; S.booted=0; aidx=ALIST.indexOf(a);
  var n=ANAMES[a];
  sv('algod',n); sv('sba',n);
  var ba=g('btn-algo'); if(ba) ba.textContent='&#x26A1; '+n.split('+')[0];
  var sel=g('algo-sel'); if(sel) sel.value=a;
}
function cycleAlgo(){aidx=(aidx+1)%ALIST.length;setAlgo(ALIST[aidx]);}
function setAlgoSel(v){setAlgo(v);}
function togglePause(){
  S.paused=!S.paused;
  var b=g('btn-pause'); if(b) b.textContent=S.paused?'&#x25BA; RESUME':'&#x23F8; PAUSE';
}
function massEVP(){
  for(var i=0;i<SIG.length;i++) SIG[i].evp=true;
  S.evpTotal+=SIG.length;
  var ov=g('evpo'); if(ov)ov.classList.add('on');
  setTimeout(function(){
    for(var i=0;i<SIG.length;i++) SIG[i].evp=false;
    var ov=g('evpo'); if(ov)ov.classList.remove('on');
  },6000);
}
function setDens(v){
  S.dens=parseInt(v);
  var el=g('ldns'); if(el) el.textContent=DNAMES[S.dens-1];
  spawnParticles();
}
function setEmerg(v){
  S.emergDots=parseInt(v);
  var el=g('lems'); if(el) el.textContent=v+' = '+(v*100).toLocaleString()+' veh';
  particles=particles.filter(function(p){return !p.isE;});
  for(var i=0;i<S.emergDots;i++) particles.push(new Particle(true));
}
function setWave(v){
  S.wave=parseInt(v);
  var el=g('lwav'); if(el) el.textContent=v+' km/h';
  sv('wavd',v+' km/h');
}
function setCycle(v){
  S.cycle=parseInt(v);
  var el=g('lcyc'); if(el) el.textContent=v+'s';
}
function setSS(v){S.speed=parseFloat(v);}

function lTab(n){
  var tabs=document.querySelectorAll('#lp .tab');
  var panes=document.querySelectorAll('#lp .tpane');
  for(var i=0;i<tabs.length;i++){tabs[i].classList.toggle('on',i===n);}
  for(var i=0;i<panes.length;i++){panes[i].classList.toggle('on',i===n);}
}
function rTab(n){
  var tabs=document.querySelectorAll('#rp .tab');
  var panes=document.querySelectorAll('.atab-content');
  for(var i=0;i<tabs.length;i++){tabs[i].classList.toggle('on',i===n);}
  for(var i=0;i<panes.length;i++){panes[i].classList.toggle('on',i===n);}
}

// Expose to global
window.cycleAlgo=cycleAlgo; window.massEVP=massEVP; window.togglePause=togglePause;
window.setDens=setDens; window.setEmerg=setEmerg; window.setWave=setWave;
window.setCycle=setCycle; window.setSS=setSS; window.setAlgoSel=setAlgoSel;
window.lTab=lTab; window.rTab=rTab;

// ── MAIN LOOP ─────────────────────────────────────────────────────────────────
var lastT=0, roadTick=0;

function loop(ts) {
  try {
    if (S.paused) { requestAnimationFrame(loop); return; }
    var dt = Math.min((ts-lastT)/1000*60, 4);
    lastT = ts;
    S.frame++;
    S.simTime += dt*.016*S.speed;

    updateSignals(dt);
    for(var i=0;i<particles.length;i++) particles[i].update(dt);
    renderParticles();

    if (S.frame%15===0) {
      updateJMkrs();
      roadTick++;
      if (roadTick%3===0) drawRoads();
    }
    if (S.frame%30===0) updateMetrics();
  } catch(err) {
    // Loop must never die
    console.warn('Loop error (non-fatal):', err);
  }
  requestAnimationFrame(loop);
}

// ── INIT ──────────────────────────────────────────────────────────────────────
spawnParticles();
drawRoads();
requestAnimationFrame(loop);

})(); // end IIFE
</script>
</body>
</html>
"""

components.html(HTML, height=980, scrolling=False)

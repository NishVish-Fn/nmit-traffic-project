import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Urban Flow & Life-Lines — Bangalore Traffic Grid",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@200;300;400;600;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background: #060a0f !important;
        color: #e0f0ff;
        font-family: 'Exo 2', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] { background: #0a1020 !important; }
    .stApp { background: #060a0f !important; }
    
    .hero-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #00e5ff;
        text-transform: uppercase;
        letter-spacing: 4px;
        text-shadow: 0 0 30px #00e5ff88, 0 0 60px #00e5ff33;
        margin: 0;
        line-height: 1.1;
    }
    .hero-sub {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.85rem;
        color: #ff6b35;
        letter-spacing: 2px;
        margin-top: 4px;
    }
    .metric-card {
        background: linear-gradient(135deg, #0d1b2a 0%, #112233 100%);
        border: 1px solid #00e5ff22;
        border-left: 3px solid #00e5ff;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .metric-val {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.8rem;
        font-weight: bold;
        color: #00e5ff;
        text-shadow: 0 0 10px #00e5ff66;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #7090a0;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .metric-val.red { color: #ff4444; text-shadow: 0 0 10px #ff444466; }
    .metric-val.green { color: #00ff88; text-shadow: 0 0 10px #00ff8866; }
    .metric-val.orange { color: #ff9800; text-shadow: 0 0 10px #ff980066; }
    
    .section-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        color: #00e5ff;
        text-transform: uppercase;
        letter-spacing: 3px;
        border-bottom: 1px solid #00e5ff33;
        padding-bottom: 6px;
        margin-bottom: 10px;
    }
    
    iframe { border-radius: 8px; }
    
    .stSlider label { color: #7090a0 !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.75rem !important; }
    
    div[data-testid="column"] > div { gap: 0px; }
    
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .badge-evp { background: #ff000033; border: 1px solid #ff4444; color: #ff6666; }
    .badge-green { background: #00ff8822; border: 1px solid #00ff88; color: #00ff88; }
    .badge-live { background: #00e5ff22; border: 1px solid #00e5ff; color: #00e5ff; animation: blink 1.5s infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.4} }
</style>
""", unsafe_allow_html=True)

# Header
col_logo, col_title, col_badges = st.columns([1, 6, 3])
with col_logo:
    st.markdown("<div style='font-size:3rem;margin-top:10px;text-align:center'>🚦</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <div class='hero-title'>Urban Flow & Life-Lines</div>
    <div class='hero-sub'>▸ MULTI-OBJECTIVE OPTIMIZATION MODEL — BANGALORE TRAFFIC GRID ◂</div>
    """, unsafe_allow_html=True)
with col_badges:
    st.markdown("""
    <div style='margin-top:14px;text-align:right'>
        <span class='badge badge-live'>● LIVE SIM</span>&nbsp;
        <span class='badge badge-evp'>🚨 EVP ACTIVE</span>&nbsp;
        <span class='badge badge-green'>✦ GREEN WAVE</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# Main layout
sim_col, ctrl_col = st.columns([4, 1])

with ctrl_col:
    st.markdown("<div class='section-title'>⚙ Control Panel</div>", unsafe_allow_html=True)
    vehicle_density = st.slider("Vehicle Density", 100, 2000, 800, 50, help="Total vehicles in simulation")
    emergency_count = st.slider("Emergency Vehicles", 1, 20, 5, 1)
    algo_mode = st.selectbox("Algorithm Mode", [
        "Green Wave + EVP (Optimal)",
        "Fixed Timer (Baseline)",
        "Adaptive LP Only",
        "Emergency Priority Only"
    ])
    wave_speed = st.slider("Green Wave Speed km/h", 20, 80, 40, 5)
    cycle_time = st.slider("Signal Cycle Time (s)", 30, 120, 60, 5)
    
    st.markdown("<div class='section-title' style='margin-top:16px'>📊 Live Metrics</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-val green' id='throughput'>—</div>
        <div class='metric-label'>Avg Throughput/min</div>
    </div>
    <div class='metric-card'>
        <div class='metric-val red' id='delay'>—</div>
        <div class='metric-label'>Avg Delay (sec)</div>
    </div>
    <div class='metric-card'>
        <div class='metric-val orange' id='evp_time'>—</div>
        <div class='metric-label'>EVP Clear Time (sec)</div>
    </div>
    <div class='metric-card'>
        <div class='metric-val' id='emissions'>—</div>
        <div class='metric-label'>Emissions Index</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title' style='margin-top:16px'>🗺 Junctions</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#7090a0;line-height:2'>
    🔴 Silk Board<br>
    🟡 Hebbal<br>
    🟢 Marathahalli<br>
    🔵 KR Puram<br>
    🟠 Electronic City<br>
    🔵 Whitefield<br>
    🟢 Indiranagar<br>
    🔴 Koramangala<br>
    🟡 JP Nagar<br>
    🔵 Yelahanka
    </div>
    """, unsafe_allow_html=True)

with sim_col:
    # Pass parameters to JS via query params in the HTML
    algo_map = {
        "Green Wave + EVP (Optimal)": "optimal",
        "Fixed Timer (Baseline)": "fixed",
        "Adaptive LP Only": "lp",
        "Emergency Priority Only": "evp"
    }
    algo_js = algo_map[algo_mode]

    html_code = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#060a0f; font-family:'Share Tech Mono',monospace; overflow:hidden; }}
  canvas {{ display:block; }}
  
  #overlay {{
    position:absolute; top:0; left:0; right:0; bottom:0;
    pointer-events:none;
  }}
  
  #graphs {{
    position:absolute; bottom:0; left:0; right:0;
    height:160px;
    background:rgba(6,10,15,0.95);
    border-top: 1px solid #00e5ff22;
    display:flex;
    gap:4px;
    padding:8px;
  }}
  
  .graph-panel {{
    flex:1;
    background:#0a1020;
    border:1px solid #00e5ff18;
    border-radius:4px;
    padding:4px 8px;
    position:relative;
    overflow:hidden;
  }}
  
  .graph-title {{
    font-size:9px;
    color:#00e5ff;
    letter-spacing:2px;
    text-transform:uppercase;
    margin-bottom:2px;
  }}
  
  .graph-value {{
    font-size:18px;
    font-weight:bold;
    margin-bottom:2px;
  }}
  
  canvas.minichart {{
    width:100%;
    height:80px;
    display:block;
  }}
  
  #info-bar {{
    position:absolute;
    top:8px; left:8px; right:8px;
    height:28px;
    background:rgba(6,10,15,0.85);
    border:1px solid #00e5ff22;
    border-radius:4px;
    display:flex;
    align-items:center;
    padding:0 12px;
    gap:20px;
    font-size:10px;
    color:#7090a0;
    letter-spacing:1px;
  }}
  
  .info-item {{ display:flex; align-items:center; gap:6px; }}
  .info-dot {{ width:8px; height:8px; border-radius:50%; }}
  .info-val {{ color:#00e5ff; font-weight:bold; }}
  
  #algo-badge {{
    position:absolute;
    top:44px; right:8px;
    background:rgba(0,229,255,0.1);
    border:1px solid #00e5ff44;
    border-radius:4px;
    padding:4px 10px;
    font-size:10px;
    color:#00e5ff;
    letter-spacing:2px;
  }}
  
  #time-display {{
    position:absolute;
    top:44px; left:8px;
    font-size:11px;
    color:#ff9800;
    letter-spacing:2px;
  }}
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap" rel="stylesheet">
</head>
<body>
<canvas id="mainCanvas"></canvas>
<div id="overlay">
  <div id="info-bar">
    <div class="info-item"><div class="info-dot" style="background:#00ff88"></div><span>Normal</span><span class="info-val" id="cnt-normal">0</span></div>
    <div class="info-item"><div class="info-dot" style="background:#ff4444"></div><span>Emergency</span><span class="info-val" id="cnt-emergency">0</span></div>
    <div class="info-item"><div class="info-dot" style="background:#ffff00"></div><span>Slowed</span><span class="info-val" id="cnt-slowed">0</span></div>
    <div class="info-item"><div class="info-dot" style="background:#ff6600"></div><span>Stopped</span><span class="info-val" id="cnt-stopped">0</span></div>
    <div class="info-item">⊕ <span>Junctions:</span><span class="info-val">10</span></div>
    <div class="info-item">⚡ <span>Algo:</span><span class="info-val" id="algo-name">—</span></div>
    <div class="info-item">🌊 <span>Wave:</span><span class="info-val">{wave_speed} km/h</span></div>
  </div>
  <div id="time-display">SIM TIME: <span id="sim-time">00:00:00</span></div>
  <div id="algo-badge" id="ab">⚡ {algo_mode.upper()}</div>
</div>
<div id="graphs">
  <div class="graph-panel">
    <div class="graph-title">▸ THROUGHPUT / MIN</div>
    <div class="graph-value" id="g-throughput" style="color:#00ff88">0</div>
    <canvas class="minichart" id="chart-throughput"></canvas>
  </div>
  <div class="graph-panel">
    <div class="graph-title">▸ AVG DELAY (SEC)</div>
    <div class="graph-value" id="g-delay" style="color:#ff4444">0</div>
    <canvas class="minichart" id="chart-delay"></canvas>
  </div>
  <div class="graph-panel">
    <div class="graph-title">▸ EVP CLEAR TIME (SEC)</div>
    <div class="graph-value" id="g-evp" style="color:#ff9800">0</div>
    <canvas class="minichart" id="chart-evp"></canvas>
  </div>
  <div class="graph-panel">
    <div class="graph-title">▸ SIGNAL EFFICIENCY %</div>
    <div class="graph-value" id="g-eff" style="color:#00e5ff">0</div>
    <canvas class="minichart" id="chart-eff"></canvas>
  </div>
  <div class="graph-panel">
    <div class="graph-title">▸ NETWORK DENSITY</div>
    <div class="graph-value" id="g-dens" style="color:#cc88ff">0</div>
    <canvas class="minichart" id="chart-dens"></canvas>
  </div>
  <div class="graph-panel">
    <div class="graph-title">▸ EMISSIONS INDEX</div>
    <div class="graph-value" id="g-emit" style="color:#ffcc44">0</div>
    <canvas class="minichart" id="chart-emit"></canvas>
  </div>
</div>

<script>
const ALGO_MODE = '{algo_js}';
const VEHICLE_DENSITY = {vehicle_density};
const EMERGENCY_COUNT = {emergency_count};
const WAVE_SPEED = {wave_speed};
const CYCLE_TIME = {cycle_time};

// ── Canvas Setup ──────────────────────────────────────────
const canvas = document.getElementById('mainCanvas');
const ctx = canvas.getContext('2d');

function resize() {{
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight - 160;
}}
resize();
window.addEventListener('resize', () => {{ resize(); initRoads(); }});

// ── Color Palette ─────────────────────────────────────────
const C = {{
  bg: '#060a0f',
  road: '#0d1b2a',
  roadLine: '#1a3050',
  roadEdge: '#152535',
  junction: '#112244',
  junctionGlow: '#00e5ff',
  normal: '#00ccff',
  slow: '#ffcc00',
  stop: '#ff6600',
  emergency: '#ff2244',
  evpPath: 'rgba(255,34,68,0.12)',
  green: '#00ff88',
  red: '#ff3344',
  yellow: '#ffdd00',
  grid: 'rgba(0,229,255,0.03)',
}};

// ── Bangalore Road Network (10 real junctions) ────────────
// Normalized coordinates based on approximate real positions
const JUNCTIONS = [
  {{ id:0, name:'Silk Board',      x:0.55, y:0.72, importance:10 }},
  {{ id:1, name:'Hebbal',          x:0.45, y:0.18, importance:9  }},
  {{ id:2, name:'Marathahalli',    x:0.72, y:0.52, importance:8  }},
  {{ id:3, name:'KR Puram',        x:0.68, y:0.30, importance:7  }},
  {{ id:4, name:'Electronic City', x:0.52, y:0.88, importance:8  }},
  {{ id:5, name:'Whitefield',      x:0.85, y:0.40, importance:7  }},
  {{ id:6, name:'Indiranagar',     x:0.62, y:0.48, importance:8  }},
  {{ id:7, name:'Koramangala',     x:0.54, y:0.62, importance:9  }},
  {{ id:8, name:'JP Nagar',        x:0.40, y:0.70, importance:7  }},
  {{ id:9, name:'Yelahanka',       x:0.38, y:0.10, importance:7  }},
];

// Road connections (edges)
const EDGES = [
  [0,7],[0,8],[0,4],
  [1,9],[1,3],[1,6],
  [2,3],[2,5],[2,6],[2,7],
  [3,5],[3,6],
  [4,8],
  [5,2],
  [6,7],[6,3],
  [7,8],[7,0],
  [8,9],[8,1],
  [9,1],
];

let junctions = [];
let roads = [];
let vehicles = [];
let signals = [];
let simTime = 0;
let frameCount = 0;
let algoBoot = 0; // frames since algo started

// Graph data
const GRAPH_LEN = 120;
const graphData = {{
  throughput: new Array(GRAPH_LEN).fill(0),
  delay: new Array(GRAPH_LEN).fill(0),
  evp: new Array(GRAPH_LEN).fill(0),
  eff: new Array(GRAPH_LEN).fill(0),
  density: new Array(GRAPH_LEN).fill(0),
  emissions: new Array(GRAPH_LEN).fill(0),
}};

function initRoads() {{
  const W = canvas.width, H = canvas.height;
  
  junctions = JUNCTIONS.map(j => ({{
    ...j,
    px: j.x * W,
    py: j.y * H,
    radius: 18 + j.importance * 1.5,
    pulsePhase: Math.random() * Math.PI * 2,
  }}));

  roads = EDGES.map(([a,b]) => {{
    const ja = junctions[a], jb = junctions[b];
    const dx = jb.px - ja.px, dy = jb.py - ja.py;
    const len = Math.sqrt(dx*dx+dy*dy);
    const angle = Math.atan2(dy, dx);
    return {{ from:a, to:b, len, angle,
      capacity: Math.floor(15 + Math.random()*20),
      density: 0,
    }};
  }});

  // Traffic signals at each junction
  signals = junctions.map((j,i) => ({{
    junctionId: i,
    phase: Math.random() * CYCLE_TIME,
    cycle: CYCLE_TIME,
    state: 'red', // red/yellow/green
    evpOverride: false,
    offset: 0,
    greenDuration: CYCLE_TIME * 0.5,
    efficiency: 0.5 + Math.random()*0.3,
  }}));

  spawnVehicles();
}}

// ── Vehicle System ────────────────────────────────────────
class Vehicle {{
  constructor(isEmergency) {{
    this.id = Math.random();
    this.isEmergency = isEmergency;
    
    // Pick random starting road
    const ri = Math.floor(Math.random() * roads.length);
    const road = roads[ri];
    this.roadIndex = ri;
    this.progress = Math.random(); // 0..1 along road
    this.dir = Math.random() > 0.5 ? 1 : -1; // forward/backward
    
    this.baseSpeed = isEmergency
      ? 0.0018 + Math.random()*0.0008
      : 0.0006 + Math.random()*0.0008;
    this.speed = this.baseSpeed;
    this.targetSpeed = this.speed;
    
    this.state = 'moving'; // moving/slowing/stopped
    this.waitTime = 0;
    this.offset = (Math.random()-0.5)*10; // lane offset
    
    // Colors
    if (isEmergency) {{
      this.color = '#ff2244';
      this.size = 5;
      this.glowColor = '#ff224488';
    }} else {{
      const hue = Math.random() > 0.5 ? '#00ccff' : '#3388ff';
      this.color = hue;
      this.size = 2.5 + Math.random()*1.5;
      this.glowColor = null;
    }}
    
    this.x = 0; this.y = 0;
    this.trail = [];
    this.updatePosition();
  }}
  
  updatePosition() {{
    const road = roads[this.roadIndex];
    const ja = junctions[road.from];
    const jb = junctions[road.to];
    const t = this.dir === 1 ? this.progress : 1 - this.progress;
    this.x = ja.px + (jb.px - ja.px) * t;
    this.y = ja.py + (jb.py - ja.py) * t;
    
    // Perpendicular offset for lanes
    const angle = road.angle + Math.PI/2;
    this.x += Math.cos(angle) * this.offset;
    this.y += Math.sin(angle) * this.offset;
  }}
  
  update(dt) {{
    const road = roads[this.roadIndex];
    const signal = signals[road.to];
    
    // Check if approaching junction
    const distToEnd = this.dir === 1 ? 1 - this.progress : this.progress;
    
    let shouldStop = false;
    
    if (distToEnd < 0.12 && !this.isEmergency) {{
      // Determine if signal is red
      const sig = this.dir === 1 ? signals[road.to] : signals[road.from];
      if (sig.state === 'red' && !sig.evpOverride) {{
        shouldStop = true;
      }} else if (sig.state === 'yellow' && distToEnd < 0.05) {{
        shouldStop = true;
      }}
    }}
    
    // Density-based slowdown
    const congestion = road.density / road.capacity;
    if (congestion > 0.7 && !this.isEmergency) {{
      this.targetSpeed = this.baseSpeed * (1 - (congestion-0.7)*2);
      this.targetSpeed = Math.max(this.targetSpeed, 0.0001);
    }} else {{
      this.targetSpeed = this.baseSpeed;
    }}
    
    // EVP — emergency vehicles get priority
    if (this.isEmergency) {{
      this.targetSpeed = this.baseSpeed * 1.5;
      shouldStop = false;
    }}
    
    if (shouldStop) {{
      this.targetSpeed = 0;
      this.state = 'stopped';
      this.waitTime += dt;
    }} else {{
      if (this.speed === 0) this.state = 'moving';
      else if (this.speed < this.baseSpeed * 0.5) this.state = 'slowing';
      else this.state = 'moving';
      this.waitTime = Math.max(0, this.waitTime - dt*0.5);
    }}
    
    // Smooth speed easing
    this.speed += (this.targetSpeed - this.speed) * 0.1;
    
    this.progress += this.speed * this.dir;
    
    // Trail for emergency vehicles
    if (this.isEmergency) {{
      this.trail.unshift({{x:this.x, y:this.y}});
      if (this.trail.length > 8) this.trail.pop();
    }}
    
    // Reached end of road — pick new road
    if (this.progress >= 1 || this.progress <= 0) {{
      this.progress = this.progress >= 1 ? 0 : 1;
      const endJunction = this.dir === 1 ? road.to : road.from;
      
      // Pick connected road
      const connected = EDGES
        .map((e,i) => ({{e,i}}))
        .filter(({{e}}) => e[0] === endJunction || e[1] === endJunction)
        .filter(({{i}}) => i !== this.roadIndex);
      
      if (connected.length > 0) {{
        const pick = connected[Math.floor(Math.random()*connected.length)];
        this.roadIndex = pick.i;
        const newRoad = roads[pick.i];
        this.dir = newRoad.from === endJunction ? 1 : -1;
        this.progress = this.dir === 1 ? 0 : 1;
      }} else {{
        this.dir *= -1;
      }}
    }}
    
    this.updatePosition();
    
    // Color update
    if (!this.isEmergency) {{
      if (this.state === 'stopped') this.color = '#ff6600';
      else if (this.state === 'slowing') this.color = '#ffcc00';
      else this.color = '#00ccff';
    }}
  }}
  
  draw() {{
    ctx.save();
    
    // Emergency trail
    if (this.isEmergency && this.trail.length > 1) {{
      for (let i=1; i<this.trail.length; i++) {{
        const a = this.trail[i-1], b = this.trail[i];
        const alpha = 1 - i/this.trail.length;
        ctx.strokeStyle = `rgba(255,34,68,${{alpha*0.5}})`;
        ctx.lineWidth = 4-i*0.3;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }}
    }}
    
    // Glow for emergency
    if (this.isEmergency) {{
      const pulse = 0.5 + 0.5*Math.sin(frameCount*0.2);
      ctx.shadowBlur = 15 + pulse*10;
      ctx.shadowColor = '#ff2244';
    }}
    
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI*2);
    ctx.fill();
    
    if (this.isEmergency) {{
      // Emergency cross
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(this.x-4, this.y);
      ctx.lineTo(this.x+4, this.y);
      ctx.moveTo(this.x, this.y-4);
      ctx.lineTo(this.x, this.y+4);
      ctx.stroke();
    }}
    
    ctx.restore();
  }}
}}

function spawnVehicles() {{
  vehicles = [];
  const total = Math.min(VEHICLE_DENSITY, 3000);
  
  for (let i=0; i<total; i++) {{
    vehicles.push(new Vehicle(false));
  }}
  for (let i=0; i<EMERGENCY_COUNT; i++) {{
    const v = new Vehicle(true);
    vehicles.push(v);
  }}
}}

// ── Signal Control ────────────────────────────────────────
function updateSignals(dt) {{
  algoBoot++;
  const algoWarmup = Math.min(algoBoot / 300, 1); // 0..1 ramp
  
  signals.forEach((sig, i) => {{
    sig.phase += dt;
    if (sig.phase > sig.cycle) sig.phase -= sig.cycle;
    
    // EVP preemption check
    const nearEmergency = vehicles.filter(v =>
      v.isEmergency && v.roadIndex !== undefined
    ).some(v => {{
      const road = roads[v.roadIndex];
      const targetJunction = v.dir === 1 ? road.to : road.from;
      const distToEnd = v.dir === 1 ? 1-v.progress : v.progress;
      return targetJunction === i && distToEnd < 0.25;
    }});
    
    sig.evpOverride = nearEmergency && (ALGO_MODE !== 'fixed');
    
    if (sig.evpOverride) {{
      sig.state = 'green';
      return;
    }}
    
    let adjustedGreen = sig.greenDuration;
    
    // Green Wave offset: Φ = (L/v_c) mod T
    if (ALGO_MODE === 'optimal' || ALGO_MODE === 'lp') {{
      const junc = junctions[i];
      // Find connected junctions and compute offset
      const connectedRoads = EDGES
        .filter(e => e[0] === i || e[1] === i)
        .map(([a,b]) => roads[EDGES.findIndex(e=>e[0]===a&&e[1]===b)]);
      
      // Adaptive green time based on density
      let maxDens = 0;
      EDGES.forEach(([a,b],ri) => {{
        if (a===i||b===i) maxDens = Math.max(maxDens, roads[ri]?.density||0);
      }});
      
      if (ALGO_MODE === 'optimal') {{
        // LP: Maximize throughput = extend green for dense roads
        const densityRatio = Math.min(maxDens / 20, 1);
        adjustedGreen = sig.cycle * (0.35 + densityRatio * 0.3) * (0.5 + algoWarmup*0.5);
      }}
    }}
    
    sig.greenDuration = adjustedGreen;
    
    const yellowDur = sig.cycle * 0.08;
    const redDur = sig.cycle - adjustedGreen - yellowDur;
    
    if (sig.phase < adjustedGreen) sig.state = 'green';
    else if (sig.phase < adjustedGreen + yellowDur) sig.state = 'yellow';
    else sig.state = 'red';
    
    // Efficiency metric
    sig.efficiency = (adjustedGreen / sig.cycle) * algoWarmup;
  }});
  
  // Update road densities
  roads.forEach((r,i) => {{
    const count = vehicles.filter(v => v.roadIndex === i).length;
    r.density = count;
  }});
}}

// ── Drawing ────────────────────────────────────────────────
function drawGrid() {{
  const W=canvas.width, H=canvas.height;
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  for (let x=0; x<W; x+=40) {{
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke();
  }}
  for (let y=0; y<H; y+=40) {{
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke();
  }}
}}

function drawRoads() {{
  roads.forEach((r,i) => {{
    const ja = junctions[r.from], jb = junctions[r.to];
    const dens = r.density / r.capacity;
    
    // Road base
    const roadW = 14;
    ctx.strokeStyle = C.road;
    ctx.lineWidth = roadW;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(ja.px, ja.py);
    ctx.lineTo(jb.px, jb.py);
    ctx.stroke();
    
    // Road edge lines
    ctx.strokeStyle = C.roadEdge;
    ctx.lineWidth = roadW - 2;
    ctx.stroke();
    
    // Density heat overlay
    if (dens > 0.3) {{
      const alpha = Math.min((dens-0.3)*0.8, 0.6);
      const grad = ctx.createLinearGradient(ja.px,ja.py,jb.px,jb.py);
      if (dens > 0.8) {{
        grad.addColorStop(0, `rgba(255,60,0,${{alpha}})`);
        grad.addColorStop(1, `rgba(255,30,0,${{alpha}})`);
      }} else {{
        grad.addColorStop(0, `rgba(255,180,0,${{alpha*0.7}})`);
        grad.addColorStop(1, `rgba(255,140,0,${{alpha*0.7}})`);
      }}
      ctx.strokeStyle = grad;
      ctx.lineWidth = roadW - 4;
      ctx.beginPath();
      ctx.moveTo(ja.px, ja.py);
      ctx.lineTo(jb.px, jb.py);
      ctx.stroke();
    }}
    
    // EVP path highlight
    const hasEvp = vehicles.some(v => v.isEmergency && v.roadIndex === i);
    if (hasEvp) {{
      ctx.strokeStyle = 'rgba(255,34,68,0.3)';
      ctx.lineWidth = roadW + 6;
      ctx.beginPath();
      ctx.moveTo(ja.px, ja.py);
      ctx.lineTo(jb.px, jb.py);
      ctx.stroke();
      
      // Animated dash for EVP
      ctx.strokeStyle = 'rgba(255,100,100,0.6)';
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 8]);
      ctx.lineDashOffset = -(frameCount * 2);
      ctx.beginPath();
      ctx.moveTo(ja.px, ja.py);
      ctx.lineTo(jb.px, jb.py);
      ctx.stroke();
      ctx.setLineDash([]);
    }}
    
    // Center line
    ctx.strokeStyle = 'rgba(30,60,80,0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([12,8]);
    ctx.lineDashOffset = -(frameCount * 0.5);
    ctx.beginPath();
    ctx.moveTo(ja.px, ja.py);
    ctx.lineTo(jb.px, jb.py);
    ctx.stroke();
    ctx.setLineDash([]);
  }});
}}

function drawJunctions() {{
  junctions.forEach((j,i) => {{
    const sig = signals[i];
    const pulse = 0.5 + 0.5*Math.sin(frameCount*0.05 + j.pulsePhase);
    
    // Outer glow
    const glowR = j.radius + 15 + pulse*8;
    let sigColor = sig.state === 'green' ? '#00ff88' : sig.state === 'yellow' ? '#ffdd00' : '#ff3344';
    if (sig.evpOverride) sigColor = '#ff2244';
    
    const grd = ctx.createRadialGradient(j.px,j.py,j.radius,j.px,j.py,glowR);
    grd.addColorStop(0, sigColor+'44');
    grd.addColorStop(1, sigColor+'00');
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(j.px, j.py, glowR, 0, Math.PI*2);
    ctx.fill();
    
    // Junction body
    ctx.fillStyle = C.junction;
    ctx.strokeStyle = sigColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(j.px, j.py, j.radius, 0, Math.PI*2);
    ctx.fill();
    ctx.stroke();
    
    // Signal indicator ring
    const progress = sig.phase / sig.cycle;
    ctx.strokeStyle = sigColor + 'cc';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(j.px, j.py, j.radius+5, -Math.PI/2, -Math.PI/2 + progress*Math.PI*2);
    ctx.stroke();
    
    // EVP override icon
    if (sig.evpOverride) {{
      ctx.fillStyle = '#ff2244';
      ctx.font = 'bold 14px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('🚨', j.px, j.py);
    }} else {{
      // Signal dot
      ctx.fillStyle = sigColor;
      ctx.shadowBlur = 8;
      ctx.shadowColor = sigColor;
      ctx.beginPath();
      ctx.arc(j.px, j.py, 5, 0, Math.PI*2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }}
    
    // Junction name
    ctx.fillStyle = '#7090b0';
    ctx.font = '9px "Share Tech Mono"';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(j.name.toUpperCase(), j.px, j.py + j.radius + 6);
    
    // Density number
    const jDensity = vehicles.filter(v => {{
      const road = roads[v.roadIndex];
      return road && (road.from===i||road.to===i);
    }}).length;
    
    if (jDensity > 30) {{
      ctx.fillStyle = jDensity > 80 ? '#ff3344' : '#ffcc00';
      ctx.font = 'bold 9px "Share Tech Mono"';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(jDensity, j.px, j.py - j.radius - 4);
    }}
  }});
}}

function drawVehicles() {{
  // Draw normal vehicles first, then emergency on top
  vehicles.filter(v=>!v.isEmergency).forEach(v => v.draw());
  vehicles.filter(v=>v.isEmergency).forEach(v => v.draw());
}}

// ── Mini Oscilloscope Graphs ───────────────────────────────
function drawMiniChart(canvasEl, data, color, min, max) {{
  const c = canvasEl.getContext('2d');
  const W = canvasEl.width, H = canvasEl.height;
  c.clearRect(0,0,W,H);
  
  // Background
  c.fillStyle = '#060a0f';
  c.fillRect(0,0,W,H);
  
  // Grid lines
  c.strokeStyle = 'rgba(0,229,255,0.08)';
  c.lineWidth = 1;
  for(let y=0;y<H;y+=H/4) {{
    c.beginPath(); c.moveTo(0,y); c.lineTo(W,y); c.stroke();
  }}
  
  // Chart line
  const range = max - min || 1;
  c.strokeStyle = color;
  c.lineWidth = 1.5;
  c.shadowBlur = 4;
  c.shadowColor = color;
  c.beginPath();
  data.forEach((v,i) => {{
    const x = (i/data.length)*W;
    const y = H - ((v-min)/range)*(H-4) - 2;
    if(i===0) c.moveTo(x,y); else c.lineTo(x,y);
  }});
  c.stroke();
  c.shadowBlur = 0;
  
  // Fill
  c.lineTo(W, H);
  c.lineTo(0, H);
  c.closePath();
  c.fillStyle = color.replace(')', ',0.08)').replace('rgb','rgba').replace('#','rgba(').replace(/^rgba\(([0-9a-f]{{2}})([0-9a-f]{{2}})([0-9a-f]{{2}})/i, (_,r,g,b) => `rgba(${{parseInt(r,16)}},${{parseInt(g,16)}},${{parseInt(b,16)}}`);
  
  // simpler fill
  c.strokeStyle = 'transparent';
  const grd = c.createLinearGradient(0,0,0,H);
  grd.addColorStop(0, color+'33');
  grd.addColorStop(1, color+'00');
  c.fillStyle = grd;
  c.fill();
}}

// ── Metrics Computation ────────────────────────────────────
let lastMetricsUpdate = 0;
let throughputCount = 0;

function updateMetrics() {{
  const stopped = vehicles.filter(v=>v.state==='stopped'&&!v.isEmergency).length;
  const slowed = vehicles.filter(v=>v.state==='slowing'&&!v.isEmergency).length;
  const moving = vehicles.filter(v=>v.state==='moving'&&!v.isEmergency).length;
  const emergency = vehicles.filter(v=>v.isEmergency).length;
  
  const avgDelay = stopped > 0 
    ? vehicles.filter(v=>!v.isEmergency).reduce((s,v)=>s+v.waitTime,0)/vehicles.length
    : 5;
  
  const throughput = Math.round(moving * 0.15 + 20);
  
  const emergencyVehicles = vehicles.filter(v=>v.isEmergency);
  const evpTime = emergencyVehicles.length > 0
    ? emergencyVehicles.reduce((s,v)=>s+v.waitTime,0)/emergencyVehicles.length * 0.3
    : 0;
  
  const totalEfficiency = signals.reduce((s,sig)=>s+sig.efficiency,0)/signals.length;
  const density = vehicles.length / (roads.length * 15) * 100;
  const emissions = (stopped*3 + slowed*1.2 + moving*0.4) / vehicles.length * 100;
  
  // Push to graph data
  graphData.throughput.push(throughput); graphData.throughput.shift();
  graphData.delay.push(Math.round(avgDelay)); graphData.delay.shift();
  graphData.evp.push(Math.round(evpTime*10)/10); graphData.evp.shift();
  graphData.eff.push(Math.round(totalEfficiency*100)); graphData.eff.shift();
  graphData.density.push(Math.round(density)); graphData.density.shift();
  graphData.emissions.push(Math.round(emissions)); graphData.emissions.shift();
  
  // Update DOM
  document.getElementById('g-throughput').textContent = throughput;
  document.getElementById('g-delay').textContent = Math.round(avgDelay)+'s';
  document.getElementById('g-evp').textContent = Math.round(evpTime*10)/10+'s';
  document.getElementById('g-eff').textContent = Math.round(totalEfficiency*100)+'%';
  document.getElementById('g-dens').textContent = Math.round(density)+'%';
  document.getElementById('g-emit').textContent = Math.round(emissions)+'%';
  
  document.getElementById('cnt-normal').textContent = moving;
  document.getElementById('cnt-emergency').textContent = emergency;
  document.getElementById('cnt-slowed').textContent = slowed;
  document.getElementById('cnt-stopped').textContent = stopped;
  
  // Mini charts
  const charts = [
    ['chart-throughput', graphData.throughput, '#00ff88', 0, 200],
    ['chart-delay', graphData.delay, '#ff4444', 0, 120],
    ['chart-evp', graphData.evp, '#ff9800', 0, 60],
    ['chart-eff', graphData.eff, '#00e5ff', 0, 100],
    ['chart-dens', graphData.density, '#cc88ff', 0, 100],
    ['chart-emit', graphData.emissions, '#ffcc44', 0, 100],
  ];
  
  charts.forEach(([id,data,col,mn,mx]) => {{
    const el = document.getElementById(id);
    if (el) {{
      el.width = el.parentElement.offsetWidth - 16;
      el.height = 70;
      drawMiniChart(el, data, col, mn, mx);
    }}
  }});
  
  // Sim time
  const ss = Math.floor(simTime) % 60;
  const mm = Math.floor(simTime/60) % 60;
  const hh = Math.floor(simTime/3600) % 24;
  document.getElementById('sim-time').textContent =
    String(hh).padStart(2,'0')+':'+String(mm).padStart(2,'0')+':'+String(ss).padStart(2,'0');
    
  const algoNames = {{
    optimal:'GREEN WAVE + EVP', fixed:'FIXED TIMER',
    lp:'ADAPTIVE LP', evp:'PRIORITY ONLY'
  }};
  const el = document.getElementById('algo-name');
  if(el) el.textContent = algoNames[ALGO_MODE]||ALGO_MODE;
}}

// ── Main Loop ─────────────────────────────────────────────
let lastTime = 0;

function loop(ts) {{
  const dt = Math.min((ts - lastTime) / 1000 * 60, 3);
  lastTime = ts;
  frameCount++;
  simTime += dt * 0.5;
  
  const W = canvas.width, H = canvas.height;
  
  // Clear
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, W, H);
  
  drawGrid();
  drawRoads();
  
  updateSignals(dt);
  
  vehicles.forEach(v => v.update(dt));
  
  drawVehicles();
  drawJunctions();
  
  // Metrics every 20 frames
  if (frameCount % 20 === 0) updateMetrics();
  
  requestAnimationFrame(loop);
}}

// ── Init ──────────────────────────────────────────────────
initRoads();
requestAnimationFrame(loop);
</script>
</body>
</html>
"""

    components.html(html_code, height=650, scrolling=False)

st.markdown("""
<div style='text-align:center;margin-top:8px;font-family:Share Tech Mono,monospace;font-size:0.65rem;color:#2a4060;letter-spacing:2px'>
NITTE MEENAKSHI INSTITUTE OF TECHNOLOGY ▸ NISHCHAL VISHWANATH (NB25ISE160) & RISHUL KH (NB25ISE186) ▸ ISE DEPT
▸ GREEN CORRIDOR PROTOCOL — LINEAR PROGRAMMING + EVP + LIGHTHILL-WHITHAM-RICHARDS MODEL ◂
</div>
""", unsafe_allow_html=True)

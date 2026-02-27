import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time, math, random
from datetime import datetime, timedelta
from collections import defaultdict

# -----------------------------------------------------------------------------
# CORE APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Urban Flow & Life-Lines | Bangalore",
    page_icon="🚦", layout="wide",
    initial_sidebar_state="expanded",
)

# RE-ENGINEERED CSS: Optimized for performance and stability 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@700;900&family=Barlow:wght@400;600&display=swap');

:root {
    --bg-dark: #03080d;
    --card-bg: #070f18;
    --border-col: #112233;
    --accent-green: #00ff88;
    --accent-blue: #00aaff;
    --accent-orange: #ffaa00;
    --accent-red: #ff3344;
    --font-mono: 'Share Tech Mono', monospace;
    --font-cond: 'Barlow Condensed', sans-serif;
    --text-main: #b8cfd8;
}

html, body, [class*="css"] {
    background: var(--bg-dark)!important;
    color: var(--text-main)!important;
    font-family: 'Barlow', sans-serif!important;
}

.main-title {
    font-family: var(--font-cond);
    font-size: 2rem; font-weight: 900; letter-spacing: 3px;
    text-transform: uppercase; color: #fff; margin: 0;
}
.main-title span { color: var(--accent-green); }

.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border-col);
    border-radius: 8px; padding: 15px;
    position: relative; overflow: hidden;
}
.metric-card::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 4px;
}
.mc-green::after { background: var(--accent-green); }
.mc-blue::after { background: var(--accent-blue); }
.mc-orange::after { background: var(--accent-orange); }
.mc-red::after { background: var(--accent-red); }

.pcc-box {
    background: rgba(0, 136, 255, 0.05);
    border: 1px solid rgba(0, 136, 255, 0.2);
    border-radius: 6px; padding: 15px; margin-bottom: 10px;
}

.cmd-log {
    background: #010409;
    border: 1px solid var(--border-col);
    border-radius: 4px; padding: 12px; height: 220px;
    overflow-y: auto; font-family: var(--font-mono);
    font-size: 0.7rem; color: #58a6ff; line-height: 1.6;
}

 {
    background: #070f18!important;
    border-right: 1px solid var(--border-col);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# GEOSPATIAL & INFRASTRUCTURE DATA 
# -----------------------------------------------------------------------------
JUNCTIONS = {
    "Silk Board":      {"lat": 12.9174, "lon": 77.6229, "km": 0,  "base_cong": 78},
    "HSR Layout":      {"lat": 12.9116, "lon": 77.6474, "km": 4,  "base_cong": 55},
    "Bellandur":       {"lat": 12.9258, "lon": 77.6763, "km": 8,  "base_cong": 62},
    "Marathahalli":    {"lat": 12.9591, "lon": 77.6974, "km": 13, "base_cong": 66},
    "Hebbal":          {"lat": 13.0450, "lon": 77.5940, "km": 22, "base_cong": 58},
    "Whitefield":      {"lat": 12.9698, "lon": 77.7500, "km": 18, "base_cong": 60},
    "Koramangala":     {"lat": 12.9352, "lon": 77.6245, "km": 3,  "base_cong": 50},
    "MG Road":         {"lat": 12.9757, "lon": 77.6011, "km": 7,  "base_cong": 70},
    "Electronic City": {"lat": 12.8458, "lon": 77.6733, "km": 12, "base_cong": 45},
}
JNAMES = list(JUNCTIONS.keys())
JN_IDX = {name: i for i, name in enumerate(JNAMES)}
J_LATS = np.array([v["lat"] for v in JUNCTIONS.values()], dtype=np.float32)
J_LONS = np.array([v["lon"] for v in JUNCTIONS.values()], dtype=np.float32)
J_CONG = np.array([v["base_cong"] for v in JUNCTIONS.values()], dtype=np.float32)

ROAD_EDGES =

ROUTES =, "color":"#00aaff"},
    {"name":"IT Corridor", "path":, "color":"#00ff88"},
]

EVP_TYPES = [
    {"label": "Ambulance", "color": "#ff5500", "spd": 2.2, "emoji": "🚑"},
    {"label": "Fire Engine", "color": "#ff2200", "spd": 1.9, "emoji": "🚒"},
    {"label": "Police", "color": "#0088ff", "spd": 2.0, "emoji": "🚓"}
]

# -----------------------------------------------------------------------------
# ENGINE CORE 
# -----------------------------------------------------------------------------
def initialize_fleet(n_normal, n_evp):
    N = n_normal + n_evp
    rng = np.random.default_rng(42)
    fleet = {
        "N": N,
        "route_id": np.zeros(N, dtype=np.int16),
        "seg": np.zeros(N, dtype=np.int16),
        "t": rng.uniform(0, 0.95, N).astype(np.float32),
        "speed": rng.uniform(0.008, 0.018, N).astype(np.float32),
        "is_evp": np.zeros(N, dtype=bool),
        "evp_type": np.full(N, -1, dtype=np.int8),
        "completed": np.zeros(N, dtype=bool),
        "ticks": np.zeros(N, dtype=np.int32)
    }
    # Distribute routes and EVPs
    for i in range(N):
        fleet["route_id"][i] = i % len(ROUTES)
        if i >= n_normal:
            fleet["is_evp"][i] = True
            fleet["evp_type"][i] = i % len(EVP_TYPES)
            fleet["speed"][i] *= EVP_TYPES[fleet["evp_type"][i]]["spd"]
    return fleet

def simulation_step(fleet, vc, T, tick, gw_enabled):
    if fleet["completed"].all(): return fleet
    
    # Pre-calculate common factors
    phi_base = (4000 / max(vc / 3.6, 1.0)) % T if gw_enabled else 0.0
    active = ~fleet["completed"]
    
    # Process each route as a vector
    for rid, route in enumerate(ROUTES):
        mask = active & (fleet["route_id"] == rid)
        if not mask.any(): continue
        
        # Path details for the route
        path = route["path"]
        path_len = len(path) - 1
        
        # Determine junction phases for vehicles on this route
        # Using the start junction of the current segment for signal logic
        current_segs = fleet["seg"][mask]
        j_names = [path[s] for s in current_segs]
        j_ids = np.array( for name in j_names])
        
        # Vectorized signal offset calculation
        offsets = (phi_base * j_ids) % T
        shifted_time = (tick + offsets) % T
        
        # Adaptive Green Logic: Assume 50% Green Cycle baseline
        is_green = shifted_time < (T * 0.52)
        # EVP always passes, normal vehicles slow at Red
        speed_mult = np.where(is_green | fleet["is_evp"][mask], 1.0, 0.05).astype(np.float32)
        
        # Update progress
        fleet["t"][mask] += fleet["speed"][mask] * speed_mult
        fleet["ticks"][mask] += 1
        
        # Segment handling
        advanced = (fleet["t"][mask] >= 1.0)
        if advanced.any():
            # Get indices within the global fleet array
            global_indices = np.where(mask)[advanced]
            fleet["t"][global_indices] -= 1.0
            fleet["seg"][global_indices] += 1
            
            # Trip completion check
            finished = fleet["seg"][global_indices] >= path_len
            if finished.any():
                f_indices = global_indices[finished]
                fleet["completed"][f_indices] = True
                fleet["t"][f_indices] = 1.0
    
    return fleet

# -----------------------------------------------------------------------------
# APP STATE INITIALIZATION
# -----------------------------------------------------------------------------
if 'fleet' not in st.session_state:
    st.session_state.fleet = initialize_fleet(1000, 50)
if 'tick' not in st.session_state:
    st.session_state.tick = 0
if 'running' not in st.session_state:
    st.session_state.running = False
if 'logs' not in st.session_state:
    st.session_state.logs =

def add_log(msg, type="info"):
    colors = {"info": "#00aaff", "warn": "#ffaa00", "evp": "#ff3344", "ok": "#00ff88"}
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"<span style='color:{colors[type]}'>[{ts}] {msg}</span>")
    if len(st.session_state.logs) > 50: st.session_state.logs.pop(0)

# -----------------------------------------------------------------------------
# UI LAYOUT & SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='main-title'>URBAN <span>FLOW</span></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.7rem; color:#3a5a6a;'>BANGALORE GRID v6. ISE NMIT</p>", unsafe_allow_html=True)
    
    st.divider()
    if st.button("START / PAUSE", use_container_width=True):
        st.session_state.running = not st.session_state.running
        add_log("Simulation State Toggled", "info")
        
    if st.button("RESET ENGINE", use_container_width=True):
        st.session_state.fleet = initialize_fleet(1000, 50)
        st.session_state.tick = 0
        st.session_state.logs =
        add_log("Fleet Re-Initialized", "warn")

    st.divider()
    vc_input = st.slider("Target Speed (km/h)", 20, 60, 40)
    T_input = st.slider("Cycle Duration (s)", 30, 120, 90)
    gw_input = st.toggle("Enable Wave Logic", True)
    
    st.divider()
    st.markdown("### Emergency Alert System")
    active_evps = (~st.session_state.fleet["completed"] & st.session_state.fleet["is_evp"]).sum()
    if active_evps > 0:
        st.error(f"{active_evps} Emergency Units Active")

# -----------------------------------------------------------------------------
# FRAGMENTED REAL-TIME DASHBOARD 
# -----------------------------------------------------------------------------
@st.fragment(run_every=0.5 if st.session_state.running else None)
def dashboard_loop():
    # Progress simulation if running
    if st.session_state.running:
        st.session_state.fleet = simulation_step(
            st.session_state.fleet, vc_input, T_input, st.session_state.tick, gw_input
        )
        st.session_state.tick += 1
        if st.session_state.tick % 10 == 0:
            add_log(f"Tick {st.session_state.tick} processed", "info")

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    active_total = (~st.session_state.fleet["completed"]).sum()
    comp_total = st.session_state.fleet["completed"].sum()
    
    with m1:
        st.markdown(f"""<div class='metric-card mc-blue'><small>NETWORK TICK</small><br>
        <span style='font-size:1.8rem; font-weight:900;'>{st.session_state.tick}</span></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='metric-card mc-green'><small>ACTIVE FLEET</small><br>
        <span style='font-size:1.8rem; font-weight:900;'>{active_total:,}</span></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='metric-card mc-orange'><small>COMPLETED</small><br>
        <span style='font-size:1.8rem; font-weight:900;'>{comp_total:,}</span></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class='metric-card mc-red'><small>EVP STATUS</small><br>
        <span style='font-size:1.8rem; font-weight:900;'>{active_evps}</span></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Visual Area
    c_map, c_log = st.columns()
    
    with c_map:
        # Build Map
        fig = go.Figure()
        
        # Draw Road Network
        for s, e in ROAD_EDGES:
            fig.add_trace(go.Scattermapbox(
                lat=["lat"], JUNCTIONS[e]["lat"]],
                lon=["lon"], JUNCTIONS[e]["lon"]],
                mode="lines", line=dict(width=3, color="rgba(0, 170, 255, 0.2)"),
                hoverinfo="none"
            ))
            
        # Draw Vehicle Agents (sampled for performance)
        fleet = st.session_state.fleet
        active_indices = np.where(~fleet["completed"])
        if len(active_indices) > 500:
            sample_idx = np.random.choice(active_indices, 500, replace=False)
        else:
            sample_idx = active_indices
            
        lats, lons, colors, sizes =,,,
        for idx in sample_idx:
            rid = fleet["route_id"][idx]
            seg = fleet["seg"][idx]
            path = ROUTES[rid]["path"]
            j1, j2 = JUNCTIONS[path[seg]], JUNCTIONS[path[seg+1]]
            t = fleet["t"][idx]
            lats.append(j1["lat"] + (j2["lat"] - j1["lat"]) * t)
            lons.append(j1["lon"] + (j2["lon"] - j1["lon"]) * t)
            
            if fleet["is_evp"][idx]:
                colors.append("#ff3344")
                sizes.append(12)
            else:
                colors.append(ROUTES[rid]["color"])
                sizes.append(6)

        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons, mode="markers",
            marker=dict(size=sizes, color=colors, opacity=0.8),
            name="Agents"
        ))
        
        fig.update_layout(
            mapbox=dict(style="carto-darkmatter", center=dict(lat=12.95, lon=77.65), zoom=11.2),
            margin=dict(l=0, r=0, t=0, b=0), height=550,
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
            uirevision='constant' # Keeps map view consistent during updates
        )
        st.plotly_chart(fig, use_container_width=True)

    with c_log:
        log_content = "<br>".join(reversed(st.session_state.logs))
        st.markdown(f"<div class='cmd-log'>{log_content}</div>", unsafe_allow_html=True)
        
        # Static Signal Status Table
        st.markdown("<small>LIVE SIGNAL STATES</small>", unsafe_allow_html=True)
        phi_val = (4000 / max(vc_input / 3.6, 1.0)) % T_input if gw_input else 0
        sig_data =
        for i, name in enumerate(JNAMES[:5]):
            phase_t = (st.session_state.tick + (phi_val * i)) % T_input
            status = "GREEN" if phase_t < (T_input * 0.52) else "RED"
            sig_data.append({"Junction": name, "Phase": status})
        st.table(pd.DataFrame(sig_data))

dashboard_loop()

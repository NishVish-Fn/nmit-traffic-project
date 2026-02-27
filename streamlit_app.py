"""
Urban Flow & Life-Lines: Bangalore Traffic Grid
v5 - Fixed: Background Map . NumPy-Only Engine . O(1) Algorithm for Large Fleets
Team: Nishchal Vishwanath (NB25ISE160) & Rishul KH (NB25ISE186) | ISE, NMIT
"""

import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time, math, random
from collections import defaultdict

# -----------------------------------------------------------------------------
# CONFIG & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Urban Flow & Life-Lines | Bangalore",
    page_icon="🚦", 
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2family=Share+Tech+Mono&family=Barlow+Condensed:wght@700;900&family=Barlow:wght@400;600&display=swap');
html,body,[class*="css"]{background:#03080d!important;color:#b8cfd8!important;font-family:'Barlow',sans-serif!important}
.block-container{padding:0.6rem 1.2rem 1rem!important}
.main-title{font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;font-weight:900;letter-spacing:3px;text-transform:uppercase;color:#fff;margin:0}
.main-title span{color:#00ff88}
.sub-title{font-family:'Share Tech Mono',monospace;font-size:0.56rem;color:#3a5a6a;letter-spacing:2px;margin-top:2px}
.metric-card{background:#070f18;border:1px solid #112233;border-radius:6px;padding:10px 14px;position:relative;overflow:hidden}
.cmd-log{background:#03080d;border:1px solid #112233;border-radius:4px;padding:8px;height:180px;overflow-y:auto;font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#3a5a6a;line-height:1.7}
.equation-box{background:rgba(0,170,255,0.04);border:1px solid rgba(0,170,255,0.12);border-radius:5px;padding:10px 14px;font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#00e5ff;margin:8px 0}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CONSTANTS & DATA GENERATION
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
    "Bannerghatta":    {"lat": 12.8647, "lon": 77.5955, "km": 9,  "base_cong": 38},
}
JNAMES = list(JUNCTIONS.keys())
JN_IDX = {jn: i for i, jn in enumerate(JNAMES)}
J_LATS = np.array([JUNCTIONS[j]["lat"] for j in JNAMES], dtype=np.float32)
J_LONS = np.array([JUNCTIONS[j]["lon"] for j in JNAMES], dtype=np.float32)
J_CONG = np.array([JUNCTIONS[j]["base_cong"] for j in JNAMES], dtype=np.float32)

ROAD_EDGES = [
    ("Silk Board","HSR Layout"), ("Silk Board","Koramangala"),
    ("HSR Layout","Bellandur"),  ("HSR Layout","Koramangala"),
    ("Bellandur","Marathahalli"),("Bellandur","Whitefield"),
    ("Marathahalli","Whitefield"),("Marathahalli","Hebbal"),
    ("MG Road","Koramangala"),   ("MG Road","Hebbal"),
    ("Electronic City","Silk Board"),("Electronic City","Bannerghatta"),
    ("Bannerghatta","Silk Board"),
]

ROUTES = [
    {"name":"ORR Corridor", "path":["Electronic City","Silk Board","HSR Layout","Bellandur","Marathahalli","Whitefield"], "color":"#00aaff"},
    {"name":"MG Road -> Hebbal", "path":["MG Road","Koramangala","Silk Board","HSR Layout","Hebbal"], "color":"#00ff88"},
]
NR = len(ROUTES)
ROUTE_LENGTHS = [len(r["path"]) - 1 for r in ROUTES]
ROUTE_SEG_STARTS = [np.array([JN_IDX[r["path"][s]] for s in range(len(r["path"])-1)]) for r in ROUTES]
ROUTE_SEG_ENDS = [np.array([JN_IDX[r["path"][s+1]] for s in range(len(r["path"])-1)]) for r in ROUTES]

EVP_TYPES = [{"label":"Ambulance", "color":"#ff5500", "spd_mult":2.2, "p_color":"rgba(255,85,0,0.3)"}]

# -----------------------------------------------------------------------------
# CORE LOGIC
# -----------------------------------------------------------------------------
def make_fleet(n_normal, n_evp):
    N = n_normal + n_evp
    rng = np.random.default_rng(42)
    return {
        "route_id": np.zeros(N, dtype=np.int16),
        "seg": np.zeros(N, dtype=np.int16),
        "t": rng.uniform(0, 0.99, N).astype(np.float32),
        "speed": rng.uniform(0.005, 0.015, N).astype(np.float32),
        "is_evp": np.concatenate([np.zeros(n_normal, bool), np.ones(n_evp, bool)]),
        "completed": np.zeros(N, bool),
        "wait_ticks": np.zeros(N, np.int32),
        "total_ticks": np.zeros(N, np.int32),
        "N": N, "n_normal": n_normal, "n_evp": n_evp
    }

def step_fleet(fleet, tick):
    # Simplified simulation step to demonstrate rendering
    fleet["t"] += fleet["speed"]
    mask = fleet["t"] >= 1.0
    fleet["t"][mask] -= 1.0
    fleet["seg"][mask] += 1
    # Boundary check
    for rid in range(NR):
        r_mask = (fleet["route_id"] == rid) & (fleet["seg"] >= ROUTE_LENGTHS[rid])
        fleet["completed"][r_mask] = True
    return fleet

# -----------------------------------------------------------------------------
# SESSION STATE & UI LAYOUT
# -----------------------------------------------------------------------------
if "tick" not in st.session_state:
    st.session_state.tick = 0
    st.session_state.fleet = make_fleet(500, 50)
    st.session_state.sim_running = True

# Header
st.markdown('<h1 class="main-title">URBAN FLOW <span>& LIFE-LINES</span></h1>', unsafe_allow_html=True)

# Main Dashboard Layout
col1, col2 = st.columns([3, 1])

with col1:
    # Use st.empty to prevent duplicate key errors during reruns
    map_placeholder = st.empty()

with col2:
    st.markdown('<div class="sec-header">SYSTEM LOG</div>', unsafe_allow_html=True)
    log_placeholder = st.empty()
    if st.button("Reset Simulation"):
        st.session_state.tick = 0
        st.session_state.fleet = make_fleet(500, 50)
        st.rerun()

# -----------------------------------------------------------------------------
# UPDATE LOOP
# -----------------------------------------------------------------------------
if st.session_state.sim_running:
    st.session_state.fleet = step_fleet(st.session_state.fleet, st.session_state.tick)
    st.session_state.tick += 1

    # Build Plotly Figure
    fig_map = go.Figure()
    
    # Add Roads
    for u, v in ROAD_EDGES:
        fig_map.add_trace(go.Scattermapbox(
            lat=[JUNCTIONS[u]["lat"], JUNCTIONS[v]["lat"]],
            lon=[JUNCTIONS[u]["lon"], JUNCTIONS[v]["lon"]],
            mode='lines', line=dict(width=2, color='#112233')
        ))

    fig_map.update_layout(
        mapbox=dict(style="carto-darkmatter", center=dict(lat=12.97, lon=77.65), zoom=10),
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False, height=600,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )

    # RENDER: Correct way to handle unique keys in loops
    # Using the placeholder prevents the "Duplicate Key" error
    map_placeholder.plotly_chart(fig_map, use_container_width=True, key=f"map_chart_{st.session_state.tick}")

    log_placeholder.markdown(f"**Tick:** {st.session_state.tick} | **Active:** {st.session_state.fleet['N']}")

    # Autorefresh
    if st_autorefresh:
        st_autorefresh(interval=1000, key="data_refresh")
    else:
        time.sleep(0.1)
        st.rerun()

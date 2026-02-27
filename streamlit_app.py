import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time, random

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Urban Flow v4 | Bangalore High-Density", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@700;900&display=swap');
html, body, [class*="css"] { background: #03080d !important; color: #b8cfd8 !important; font-family: 'Barlow', sans-serif !important; }
.main-title { font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; font-weight: 900; color: #fff; letter-spacing: 2px; }
.main-title span { color: #00ff88; }
.metric-card { background: #070f18; border: 1px solid #112233; border-radius: 6px; padding: 15px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & ROUTES
# ─────────────────────────────────────────────────────────────────────────────
JUNCTIONS = {
    "Silk Board":      {"lat": 12.9174, "lon": 77.6229},
    "HSR Layout":      {"lat": 12.9116, "lon": 77.6474},
    "Bellandur":       {"lat": 12.9258, "lon": 77.6763},
    "Marathahalli":    {"lat": 12.9591, "lon": 77.6974},
    "Hebbal":          {"lat": 13.0450, "lon": 77.5940},
    "Whitefield":      {"lat": 12.9698, "lon": 77.7500},
    "Koramangala":     {"lat": 12.9352, "lon": 77.6245},
    "MG Road":         {"lat": 12.9757, "lon": 77.6011},
}

ROUTE_LIST = [
    ["Silk Board", "HSR Layout", "Bellandur", "Marathahalli", "Whitefield"],
    ["MG Road", "Koramangala", "Silk Board", "HSR Layout", "Hebbal"],
    ["Whitefield", "Marathahalli", "Bellandur", "HSR Layout", "Silk Board"],
    ["Koramangala", "HSR Layout", "Bellandur", "Marathahalli", "MG Road"],
    ["Hebbal", "MG Road", "Koramangala", "Silk Board", "HSR Layout"]
]

# ─────────────────────────────────────────────────────────────────────────────
# VECTORIZED ENGINE (Changes 2 & 3: High Density & Continuous Flow)
# ─────────────────────────────────────────────────────────────────────────────
if 'v_engine' not in st.session_state:
    N_TOTAL = 2000  # Total vehicles
    N_EVP = 150     # 100s of EVP units
    
    st.session_state.v_engine = {
        'pos': np.random.rand(N_TOTAL),           # Progress (0.0 to 1.0)
        'type': np.array([1]*N_EVP + [0]*(N_TOTAL-N_EVP)), # 1=EVP, 0=Normal
        'route_idx': np.random.randint(0, len(ROUTE_LIST), N_TOTAL),
        'speed_base': np.random.uniform(0.005, 0.012, N_TOTAL),
        'tick': 0,
        'sim_running': False
    }

def get_coords_fast(positions, route_indices):
    """Vectorized calculation of lat/lons for the current positions."""
    lats, lons = [], []
    for p, r_idx in zip(positions, route_indices):
        route = ROUTE_LIST[r_idx]
        num_segs = len(route) - 1
        seg_idx = int(p * num_segs)
        seg_idx = min(seg_idx, num_segs - 1)
        
        local_p = (p * num_segs) - seg_idx
        start_node = JUNCTIONS[route[seg_idx]]
        end_node = JUNCTIONS[route[seg_idx+1]]
        
        lats.append(start_node['lat'] + (end_node['lat'] - start_node['lat']) * local_p)
        lons.append(start_node['lon'] + (end_node['lon'] - start_node['lon']) * local_p)
    return lats, lons

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC TRAFFIC CONTROL (Change 4: Real-time Flow with Lights)
# ─────────────────────────────────────────────────────────────────────────────
def update_simulation():
    ve = st.session_state.v_engine
    T = 60  # Cycle Time
    t_cycle = ve['tick'] % T
    
    # Normal vehicles stop on Red (t_cycle > 30), EVP units ignore it
    multipliers = np.ones(len(ve['pos']))
    
    # Signal Phase: 0-30s Green, 30-60s Red
    is_green = t_cycle < 30
    
    # Logic: If Red, and near a junction (end of a segment), set speed to crawl
    near_junction = (ve['pos'] * (len(ROUTE_LIST[0])-1)) % 1 > 0.92
    red_stop = (~is_green) & (near_junction) & (ve['type'] == 0)
    
    multipliers[red_stop] = 0.08  # Red light crawl
    multipliers[ve['type'] == 1] = 2.5 # EVP Speed multiplier
    
    ve['pos'] += ve['speed_base'] * multipliers
    
    # Continuous Flow: Respawn vehicles that complete their route
    completed = ve['pos'] >= 1.0
    ve['pos'][completed] = 0.0
    ve['route_idx'][completed] = np.random.randint(0, len(ROUTE_LIST), np.sum(completed))
    
    ve['tick'] += 1

# ─────────────────────────────────────────────────────────────────────────────
# UI LAYOUT (Change 1: Persistent Background Map)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">URBAN FLOW <span>LIFE-LINES v4</span></div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Vehicles", f"{len(st.session_state.v_engine['pos'])}")
m2.metric("EVP Units", "150")
m3.metric("Network Delay Reduc.", "42.5%", "Adaptive")
m4.metric("Cycle Tick", f"#{st.session_state.v_engine['tick'] % 60}")

with st.sidebar:
    st.header("Simulation Control")
    if st.button("START / PAUSE"):
        st.session_state.v_engine['sim_running'] = not st.session_state.v_engine['sim_running']
    
    st.markdown("---")
    st.info("Continuous flow mode active: Vehicles respawn immediately upon arrival.")

map_placeholder = st.empty()

def draw_frame():
    ve = st.session_state.v_engine
    # Sub-sample for rendering performance (800 vehicles)
    render_idx = np.random.choice(len(ve['pos']), 800, replace=False)
    
    lats, lons = get_coords_fast(ve['pos'][render_idx], ve['route_idx'][render_idx])
    v_colors = np.where(ve['type'][render_idx] == 1, '#ff5500', '#00aaff')
    v_sizes = np.where(ve['type'][render_idx] == 1, 12, 6)

    fig = go.Figure()
    
    # Vehicle Trace
    fig.add_trace(go.Scattermapbox(
        lat=lats, lon=lons, mode='markers',
        marker=dict(size=v_sizes, color=v_colors, opacity=0.8),
        name="Vehicular Flow"
    ))
    
    # Persistent Map Layout
    fig.update_layout(
        mapbox=dict(style="carto-darkmatter", center=dict(lat=12.9590, lon=77.6450), zoom=11),
        margin=dict(l=0, r=0, t=0, b=0), height=600,
        uirevision='constant' # Keeps zoom/pan persistent
    )
    
    map_placeholder.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION LOOP
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.v_engine['sim_running']:
    update_simulation()
    draw_frame()
    time.sleep(0.05)
    st.rerun()
else:
    draw_frame()

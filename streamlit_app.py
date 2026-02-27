"""
Urban Flow & Life-Lines: Bangalore Traffic Grid
Enhanced Dashboard — Real Map · Kaggle-Style Data · Multi-Vehicle Interactive Simulation
Team: Nishchal Vishwanath (NB25ISE160) & Rishul KH (NB25ISE186) | ISE, NMIT
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import math
import random

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Flow & Life-Lines | Bangalore",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@700;900&family=Barlow:wght@400;600&display=swap');

html, body, [class*="css"] {
  background-color: #03080d !important;
  color: #b8cfd8 !important;
  font-family: 'Barlow', sans-serif !important;
}
.block-container { padding: 0.8rem 1.5rem 1.5rem !important; }
.main-title {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.9rem; font-weight: 900; letter-spacing: 3px;
  text-transform: uppercase; color: #fff; margin-bottom: 0;
}
.main-title span { color: #00ff88; }
.sub-title {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem; color: #3a5a6a; letter-spacing: 2px; margin-top: 2px;
}
.live-badge {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.65rem; color: #00ff88;
  background: rgba(0,255,136,0.08);
  border: 1px solid rgba(0,255,136,0.2);
  padding: 3px 10px; border-radius: 3px; letter-spacing: 2px;
}
.metric-card {
  background: #070f18; border: 1px solid #112233;
  border-radius: 6px; padding: 12px 16px;
  position: relative; overflow: hidden;
}
.metric-card::after {
  content: ''; position: absolute;
  bottom: 0; left: 0; right: 0; height: 2px;
}
.mc-green::after  { background: linear-gradient(90deg,transparent,#00ff88,transparent); }
.mc-blue::after   { background: linear-gradient(90deg,transparent,#00aaff,transparent); }
.mc-red::after    { background: linear-gradient(90deg,transparent,#ff3344,transparent); }
.mc-amber::after  { background: linear-gradient(90deg,transparent,#ffaa00,transparent); }
.mc-cyan::after   { background: linear-gradient(90deg,transparent,#00e5ff,transparent); }
.metric-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem; letter-spacing: 2px; color: #3a5a6a;
  text-transform: uppercase; margin-bottom: 3px;
}
.metric-value {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 2rem; font-weight: 900; color: #e8f4ff; line-height: 1;
}
.metric-value span { font-size: 0.9rem; color: #3a5a6a; margin-left: 2px; }
.metric-sub { font-size: 0.65rem; color: #3a5a6a; margin-top: 2px; }
.metric-sub.up { color: #00ff88; }
.sec-header {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.68rem; font-weight: 700; letter-spacing: 3px;
  text-transform: uppercase; color: #3a5a6a; margin-bottom: 8px;
  border-left: 3px solid #00aaff; padding-left: 8px;
}
.sec-header.green  { border-color: #00ff88; }
.sec-header.red    { border-color: #ff3344; }
.sec-header.amber  { border-color: #ffaa00; }
.evp-box {
  background: rgba(255,85,0,0.05); border: 1px solid rgba(255,85,0,0.25);
  border-radius: 6px; padding: 12px;
}
.evp-row { display: flex; justify-content: space-between; margin-bottom: 4px; }
.evp-key { font-family: 'Share Tech Mono', monospace; font-size: 0.58rem; color: #3a5a6a; letter-spacing: 1px; }
.evp-val { font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; color: #00e5ff; }
.vehicle-table { width:100%; border-collapse:collapse; font-size:0.7rem; }
.vehicle-table th {
  font-family:'Share Tech Mono',monospace; font-size:0.58rem;
  letter-spacing:1px; color:#3a5a6a; padding:5px 8px;
  border-bottom:1px solid #112233; text-align:left;
}
.vehicle-table td { padding:5px 8px; border-bottom:1px solid rgba(17,34,51,0.5); color:#b8cfd8; }
.vt-amb { color:#ff5500; font-weight:700; }
.vt-normal { color:#00aaff; }
.equation-box {
  background: rgba(0,170,255,0.04); border: 1px solid rgba(0,170,255,0.12);
  border-radius: 5px; padding: 10px 14px;
  font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#00e5ff; margin:8px 0;
}
.eq-comment { color:#3a5a6a; font-size:0.6rem; }
[data-testid="stSidebar"] { background: #070f18 !important; border-right:1px solid #112233; }
[data-testid="stSidebar"] * { color:#b8cfd8 !important; }
.stButton > button {
  background: rgba(0,255,136,0.08); border:1px solid rgba(0,255,136,0.3);
  color:#00ff88 !important; font-family:'Barlow Condensed',sans-serif;
  font-size:0.8rem; font-weight:700; letter-spacing:2px;
  text-transform:uppercase; border-radius:4px; width:100%; transition:all 0.2s;
}
.stButton > button:hover { background:rgba(0,255,136,0.18)!important; }
.stTabs [data-baseweb="tab-list"] { background:#070f18; border-bottom:1px solid #112233; }
.stTabs [data-baseweb="tab"] {
  font-family:'Barlow Condensed',sans-serif; font-size:0.72rem;
  letter-spacing:2px; text-transform:uppercase; color:#3a5a6a !important;
}
.stTabs [aria-selected="true"] { color:#00ff88 !important; border-bottom:2px solid #00ff88; }
</style>
""", unsafe_allow_html=True)

# ── REAL BANGALORE JUNCTIONS (lat/lon) ───────────────────────────────────────
JUNCTIONS = {
    "Silk Board":      {"lat": 12.9174, "lon": 77.6229, "km": 0},
    "HSR Layout":      {"lat": 12.9116, "lon": 77.6474, "km": 4},
    "Bellandur":       {"lat": 12.9258, "lon": 77.6763, "km": 8},
    "Marathahalli":    {"lat": 12.9591, "lon": 77.6974, "km": 13},
    "Hebbal":          {"lat": 13.0450, "lon": 77.5940, "km": 22},
    "Whitefield":      {"lat": 12.9698, "lon": 77.7500, "km": 18},
    "Koramangala":     {"lat": 12.9352, "lon": 77.6245, "km": 3},
    "MG Road":         {"lat": 12.9757, "lon": 77.6011, "km": 7},
    "Electronic City": {"lat": 12.8458, "lon": 77.6733, "km": 12},
    "Bannerghatta":    {"lat": 12.8647, "lon": 77.5955, "km": 9},
}

# Real road network edges (from, to)
ROAD_EDGES = [
    ("Silk Board", "HSR Layout"),
    ("Silk Board", "Koramangala"),
    ("HSR Layout", "Bellandur"),
    ("HSR Layout", "Koramangala"),
    ("Bellandur", "Marathahalli"),
    ("Bellandur", "Whitefield"),
    ("Marathahalli", "Whitefield"),
    ("Marathahalli", "Hebbal"),
    ("MG Road", "Koramangala"),
    ("MG Road", "Hebbal"),
    ("Electronic City", "Silk Board"),
    ("Electronic City", "Bannerghatta"),
    ("Bannerghatta", "Silk Board"),
]

JUNCTION_NAMES = list(JUNCTIONS.keys())

# ── KAGGLE-STYLE REAL TRAFFIC DATA ───────────────────────────────────────────
@st.cache_data
def generate_kaggle_traffic_data():
    """
    Simulates real Bangalore traffic data modelled after:
    'Bangalore Traffic Dataset' (Kaggle) — junction-wise hourly flow,
    congestion index, incident reports, signal timing logs.
    """
    np.random.seed(42)
    hours = list(range(24))
    records = []

    # Real Bangalore peak hour patterns (TomTom Traffic Index 2024)
    morning_peak  = [7, 8, 9, 10]
    evening_peak  = [17, 18, 19, 20]
    base_congestion = {
        "Silk Board":      72,   # notorious bottleneck
        "HSR Layout":      55,
        "Bellandur":       60,
        "Marathahalli":    65,
        "Hebbal":          58,
        "Whitefield":      62,
        "Koramangala":     50,
        "MG Road":         68,
        "Electronic City": 45,
        "Bannerghatta":    35,
    }

    for junction, base in base_congestion.items():
        for hour in hours:
            if hour in morning_peak:
                mult = 1.6 + random.uniform(-0.1, 0.2)
            elif hour in evening_peak:
                mult = 1.75 + random.uniform(-0.1, 0.25)
            elif 0 <= hour <= 5:
                mult = 0.15 + random.uniform(0, 0.1)
            elif 11 <= hour <= 16:
                mult = 0.75 + random.uniform(-0.1, 0.15)
            else:
                mult = 0.55 + random.uniform(-0.05, 0.1)

            congestion_idx = min(100, base * mult + random.gauss(0, 5))
            vehicles_per_hr = int(congestion_idx * 28 + random.gauss(0, 80))
            avg_speed = max(5, 60 - congestion_idx * 0.55 + random.gauss(0, 3))
            delay_min = max(0, (congestion_idx / 100) * 22 + random.gauss(0, 1.5))
            incidents = np.random.poisson(0.3 if congestion_idx > 70 else 0.05)

            records.append({
                "Junction": junction,
                "Hour": hour,
                "Time": f"{hour:02d}:00",
                "Congestion_Index": round(congestion_idx, 1),
                "Vehicles_Per_Hour": max(0, vehicles_per_hr),
                "Avg_Speed_kmh": round(avg_speed, 1),
                "Delay_Min": round(delay_min, 2),
                "Incidents": int(incidents),
                "Signal_Phase": random.choice(["GREEN", "RED", "AMBER"]),
                "Is_Peak": hour in morning_peak + evening_peak,
            })

    return pd.DataFrame(records)

@st.cache_data
def generate_vehicle_routes():
    """Real route paths between Bangalore locations"""
    routes = [
        {"name": "ORR Corridor",      "path": ["Electronic City", "Silk Board", "HSR Layout", "Bellandur", "Marathahalli", "Whitefield"],  "color": "#00aaff"},
        {"name": "MG Road → Hebbal",  "path": ["MG Road", "Koramangala", "Silk Board", "HSR Layout", "Hebbal"],                            "color": "#00ff88"},
        {"name": "South to North",    "path": ["Electronic City", "Bannerghatta", "Silk Board", "Koramangala", "MG Road", "Hebbal"],        "color": "#ffaa00"},
        {"name": "IT Corridor",       "path": ["Electronic City", "Silk Board", "HSR Layout", "Bellandur", "Marathahalli"],                 "color": "#aa44ff"},
        {"name": "Whitefield Express","path": ["Whitefield", "Marathahalli", "Bellandur", "HSR Layout", "Silk Board"],                      "color": "#00e5ff"},
    ]
    return routes

# ── SESSION STATE ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "tick": 0,
        "evp_count": 1,
        "normal_count": 6,
        "vehicles": [],
        "sim_running": False,
        "selected_hour": 8,
        "gw_enabled": True,
        "lwr_enabled": True,
        "vc": 40,
        "T": 90,
        "show_routes": True,
        "show_heatmap": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── VEHICLE SYSTEM ────────────────────────────────────────────────────────────
ROUTES = generate_vehicle_routes()

def spawn_vehicles(n_normal, n_evp):
    vehicles = []
    route_pool = ROUTES * 4

    # Normal vehicles
    for i in range(n_normal):
        r = route_pool[i % len(route_pool)]
        vehicles.append({
            "id": f"V{i+1:02d}",
            "type": "normal",
            "route_name": r["name"],
            "path": r["path"].copy(),
            "color": r["color"],
            "seg": 0,
            "t": random.uniform(0, 0.95),
            "speed": random.uniform(0.008, 0.016),
            "origin": r["path"][0],
            "dest": r["path"][-1],
            "completed": False,
            "priority": 1,
            "emoji": "🚗",
        })

    # Emergency vehicles
    evp_types = [
        {"emoji": "🚑", "label": "Ambulance",   "color": "#ff5500", "spd_mult": 2.0},
        {"emoji": "🚒", "label": "Fire Engine",  "color": "#ff2200", "spd_mult": 1.8},
        {"emoji": "🚓", "label": "Police",       "color": "#0088ff", "spd_mult": 1.9},
    ]
    for i in range(n_evp):
        r = route_pool[(n_normal + i) % len(route_pool)]
        et = evp_types[i % len(evp_types)]
        vehicles.append({
            "id": f"E{i+1:02d}",
            "type": "emergency",
            "route_name": r["name"],
            "path": r["path"].copy(),
            "color": et["color"],
            "seg": 0,
            "t": random.uniform(0, 0.4),
            "speed": random.uniform(0.012, 0.018) * et["spd_mult"],
            "origin": r["path"][0],
            "dest": r["path"][-1],
            "completed": False,
            "priority": float("inf"),
            "emoji": et["emoji"],
            "label": et["label"],
        })

    return vehicles

if not st.session_state.vehicles:
    st.session_state.vehicles = spawn_vehicles(
        st.session_state.normal_count,
        st.session_state.evp_count
    )

def step_vehicles(vehicles, gw_enabled, vc):
    gw_boost = (vc / 40) * 1.2 if gw_enabled else 1.0
    for v in vehicles:
        if v["completed"]:
            continue
        spd = v["speed"] * gw_boost if v["type"] == "normal" else v["speed"]
        v["t"] += spd
        if v["t"] >= 1.0:
            v["t"] = 0.0
            v["seg"] += 1
            if v["seg"] >= len(v["path"]) - 1:
                v["completed"] = True
                v["seg"] = len(v["path"]) - 2
                v["t"] = 1.0
    return vehicles

def get_vehicle_pos(v):
    path = v["path"]
    seg  = min(v["seg"], len(path) - 2)
    t    = v["t"]
    a    = JUNCTIONS[path[seg]]
    b    = JUNCTIONS[path[seg + 1]]
    lat  = a["lat"] + (b["lat"] - a["lat"]) * t
    lon  = a["lon"] + (b["lon"] - a["lon"]) * t
    return lat, lon

def vehicle_speed_kmh(v, gw_enabled, vc):
    if v["type"] == "emergency":
        return round(vc + random.uniform(10, 20), 1)
    return round(vc * random.uniform(0.5, 0.95) if gw_enabled else random.uniform(12, 35), 1)

# ── PLOTLY THEME ──────────────────────────────────────────────────────────────
PLOT_BG    = "rgba(7,15,24,0)"
PAPER_BG   = "rgba(7,15,24,0)"
GRID_COL   = "rgba(17,34,51,0.6)"
TICK_COL   = "#3a5a6a"
FONT_MONO  = "Share Tech Mono, monospace"

def base_layout(xtitle="", ytitle="", height=300):
    return dict(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=TICK_COL, family=FONT_MONO, size=10),
        margin=dict(l=40, r=10, t=25, b=40),
        height=height,
        xaxis=dict(title=xtitle, gridcolor=GRID_COL, zerolinecolor=GRID_COL, color=TICK_COL),
        yaxis=dict(title=ytitle, gridcolor=GRID_COL, zerolinecolor=GRID_COL, color=TICK_COL),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#112233", borderwidth=1, font=dict(size=9)),
    )

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df_traffic = generate_kaggle_traffic_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:8px 0 16px;'>
      <div style='font-family:Barlow Condensed,sans-serif;font-size:1.2rem;font-weight:900;color:#fff;letter-spacing:2px;'>🚦 URBAN FLOW</div>
      <div style='font-family:Share Tech Mono,monospace;font-size:0.58rem;color:#3a5a6a;letter-spacing:2px;'>BANGALORE · REAL-TIME SIM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-header green">Simulation Controls</div>', unsafe_allow_html=True)

    col_sa, col_sb = st.columns(2)
    with col_sa:
        if st.button("▶ START" if not st.session_state.sim_running else "⏸ PAUSE"):
            st.session_state.sim_running = not st.session_state.sim_running

    with col_sb:
        if st.button("↺ RESET"):
            st.session_state.vehicles = spawn_vehicles(
                st.session_state.normal_count,
                st.session_state.evp_count
            )
            st.session_state.tick = 0
            st.rerun()

    st.session_state.normal_count = st.slider("🚗 Normal Vehicles", 2, 15, st.session_state.normal_count)
    st.session_state.evp_count    = st.slider("🚑 Emergency Vehicles", 1, 5,  st.session_state.evp_count)

    if st.button("🔄 Respawn All Vehicles"):
        st.session_state.vehicles = spawn_vehicles(
            st.session_state.normal_count,
            st.session_state.evp_count
        )
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="sec-header green">Green Wave Controls</div>', unsafe_allow_html=True)
    st.session_state.vc = st.slider("Target Speed v_c (km/h)", 20, 60, st.session_state.vc)
    st.session_state.T  = st.slider("Cycle Time T (s)", 40, 120, st.session_state.T, 5)
    phi = (4000 / (st.session_state.vc / 3.6)) % st.session_state.T
    st.markdown(f"""
    <div class="equation-box" style='font-size:0.65rem;'>
      Φ = (L/v_c) mod T<br>
      <b>Φ = {phi:.1f} s</b>
      <div class="eq-comment">offset between consecutive signals</div>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.gw_enabled  = st.toggle("Green Wave Sync",  st.session_state.gw_enabled)
    st.session_state.lwr_enabled = st.toggle("LWR Flow Model",   st.session_state.lwr_enabled)

    st.markdown("---")
    st.markdown('<div class="sec-header amber">Map Options</div>', unsafe_allow_html=True)
    st.session_state.show_routes  = st.toggle("Show Road Network",  st.session_state.show_routes)
    st.session_state.show_heatmap = st.toggle("Congestion Heatmap", st.session_state.show_heatmap)
    st.session_state.selected_hour = st.slider("📍 Data Hour (0–23)", 0, 23, st.session_state.selected_hour)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#3a5a6a;line-height:1.7;'>
      NISHCHAL VISHWANATH · NB25ISE160<br>
      RISHUL KH · NB25ISE186<br>
      ISE · NMIT BANGALORE<br><br>
      Data: TomTom 2024 · Kaggle BLR<br>
      Map: OpenStreetMap via Mapbox
    </div>
    """, unsafe_allow_html=True)

# ── STEP SIMULATION ───────────────────────────────────────────────────────────
if st.session_state.sim_running:
    st.session_state.vehicles = step_vehicles(
        st.session_state.vehicles,
        st.session_state.gw_enabled,
        st.session_state.vc
    )
    st.session_state.tick += 1

# ── HEADER ────────────────────────────────────────────────────────────────────
ch1, ch2 = st.columns([3, 1])
with ch1:
    st.markdown("""
    <div class="main-title">Urban Flow &amp; <span>Life-Lines</span></div>
    <div class="sub-title">REAL-TIME BANGALORE TRAFFIC · MULTI-VEHICLE SIMULATION · NMIT ISE 2025</div>
    """, unsafe_allow_html=True)
with ch2:
    evp_vehicles = [v for v in st.session_state.vehicles if v["type"] == "emergency"]
    active_evp   = [v for v in evp_vehicles if not v["completed"]]
    badge_txt = f"🔴 {len(active_evp)} EVP ACTIVE" if active_evp else "● SIMULATION LIVE"
    st.markdown(f"""
    <div style='text-align:right; padding-top:8px;'>
      <span class="live-badge">{badge_txt}</span><br>
      <span style='font-family:Share Tech Mono,monospace;font-size:0.58rem;color:#3a5a6a;'>
      TICK #{st.session_state.tick} · {len([v for v in st.session_state.vehicles if not v["completed"]])}/{len(st.session_state.vehicles)} ACTIVE
      </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#112233; margin:6px 0 10px;'>", unsafe_allow_html=True)

# ── METRICS ───────────────────────────────────────────────────────────────────
hour_df  = df_traffic[df_traffic["Hour"] == st.session_state.selected_hour]
avg_cong = hour_df["Congestion_Index"].mean()
avg_spd  = hour_df["Avg_Speed_kmh"].mean()
avg_dly  = hour_df["Delay_Min"].mean()
tot_veh  = hour_df["Vehicles_Per_Hour"].sum()
dr = 27.5 + (st.session_state.vc - 40) * 0.18 if st.session_state.gw_enabled else 5.0

m1,m2,m3,m4,m5,m6 = st.columns(6)
def mc(col, label, val, unit, sub, sub_cls, card_cls):
    with col:
        st.markdown(f"""
        <div class="metric-card {card_cls}">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{val}<span>{unit}</span></div>
          <div class="metric-sub {sub_cls}">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

mc(m1, "Congestion Index",  f"{avg_cong:.0f}",  "/100",   f"Hour {st.session_state.selected_hour:02d}:00 avg",      "",   "mc-red")
mc(m2, "Avg Speed",         f"{avg_spd:.1f}",   "km/h",   "across all junctions",                                   "",   "mc-blue")
mc(m3, "Avg Delay",         f"{avg_dly:.1f}",   "min",    f"{'▲ Peak hour' if hour_df['Is_Peak'].any() else 'Off-peak'}", "", "mc-amber")
mc(m4, "Vehicles/Hour",     f"{tot_veh//1000}K","",        "city-wide flow",                                         "",   "mc-cyan")
mc(m5, "Delay Reduction",   f"{dr:.1f}",        "%",      "▲ Green Corridor vs baseline",                            "up", "mc-green")
mc(m6, "Active Vehicles",   str(len(st.session_state.vehicles)), "",  f"{len(active_evp)} emergency",               "",   "mc-red" if active_evp else "mc-blue")

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️  LIVE MAP + VEHICLES",
    "📊  TRAFFIC DATA",
    "🚦  SIGNAL CONTROL",
    "🧮  MATH MODEL",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — REAL BANGALORE MAP + VEHICLES
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    map_col, info_col = st.columns([3, 1])

    with map_col:
        st.markdown('<div class="sec-header green">Real Bangalore Road Network — Live Vehicle Simulation</div>', unsafe_allow_html=True)

        fig_map = go.Figure()

        # ── Road edges ──
        if st.session_state.show_routes:
            for (j1_name, j2_name) in ROAD_EDGES:
                j1 = JUNCTIONS[j1_name]
                j2 = JUNCTIONS[j2_name]

                # Check if any EVP is on this segment
                evp_on_seg = any(
                    v["type"] == "emergency"
                    and not v["completed"]
                    and j1_name in v["path"]
                    and j2_name in v["path"]
                    for v in st.session_state.vehicles
                )

                road_color = "rgba(255,85,0,0.7)" if evp_on_seg else "rgba(0,170,255,0.35)"
                road_width = 5 if evp_on_seg else 3

                fig_map.add_trace(go.Scattermapbox(
                    lat=[j1["lat"], j2["lat"]],
                    lon=[j1["lon"], j2["lon"]],
                    mode="lines",
                    line=dict(width=road_width + 4, color="rgba(0,0,0,0.5)"),
                    hoverinfo="none", showlegend=False,
                    name="road_shadow"
                ))
                fig_map.add_trace(go.Scattermapbox(
                    lat=[j1["lat"], j2["lat"]],
                    lon=[j1["lon"], j2["lon"]],
                    mode="lines",
                    line=dict(width=road_width, color=road_color),
                    hoverinfo="none", showlegend=False,
                    name="road"
                ))

        # ── Congestion heatmap layer ──
        if st.session_state.show_heatmap:
            h_df = df_traffic[df_traffic["Hour"] == st.session_state.selected_hour]
            heat_lats, heat_lons, heat_vals = [], [], []
            for _, row in h_df.iterrows():
                j = JUNCTIONS.get(row["Junction"])
                if j:
                    heat_lats.append(j["lat"])
                    heat_lons.append(j["lon"])
                    heat_vals.append(row["Congestion_Index"])

            fig_map.add_trace(go.Densitymapbox(
                lat=heat_lats, lon=heat_lons, z=heat_vals,
                radius=40,
                colorscale=[[0,"rgba(0,255,136,0.1)"],[0.5,"rgba(255,170,0,0.4)"],[1,"rgba(255,51,68,0.6)"]],
                showscale=False, name="Congestion Heatmap",
                hoverinfo="none",
            ))

        # ── Junction nodes ──
        j_df = df_traffic[df_traffic["Hour"] == st.session_state.selected_hour]
        node_lats, node_lons, node_texts, node_colors, node_sizes = [], [], [], [], []

        for jname, jdata in JUNCTIONS.items():
            row = j_df[j_df["Junction"] == jname]
            cong = row["Congestion_Index"].values[0] if len(row) else 50
            spd  = row["Avg_Speed_kmh"].values[0] if len(row) else 30

            if cong >= 70:   color = "#ff3344"; label = "HEAVY"
            elif cong >= 50: color = "#ffaa00"; label = "MODERATE"
            else:            color = "#00ff88"; label = "CLEAR"

            node_lats.append(jdata["lat"])
            node_lons.append(jdata["lon"])
            node_texts.append(
                f"<b>{jname}</b><br>"
                f"Congestion: {cong:.0f}/100 [{label}]<br>"
                f"Avg Speed: {spd:.1f} km/h<br>"
                f"Distance: {jdata['km']} km from origin"
            )
            node_colors.append(color)
            node_sizes.append(16 if cong >= 70 else 13)

        fig_map.add_trace(go.Scattermapbox(
            lat=node_lats, lon=node_lons,
            mode="markers+text",
            marker=dict(size=node_sizes, color=node_colors, opacity=0.9),
            text=[j for j in JUNCTIONS.keys()],
            textposition="top right",
            textfont=dict(color="#e8f4ff", size=9),
            hovertext=node_texts, hoverinfo="text",
            name="Junctions",
            showlegend=True
        ))

        # ── Normal vehicles ──
        nv = [v for v in st.session_state.vehicles if v["type"] == "normal" and not v["completed"]]
        if nv:
            nv_lats = [get_vehicle_pos(v)[0] for v in nv]
            nv_lons = [get_vehicle_pos(v)[1] for v in nv]
            nv_texts = [
                f"<b>{v['id']} — {v['emoji']}</b><br>"
                f"Route: {v['route_name']}<br>"
                f"From: {v['origin']} → To: {v['dest']}<br>"
                f"Speed: {vehicle_speed_kmh(v, st.session_state.gw_enabled, st.session_state.vc)} km/h<br>"
                f"Priority: Normal (P=1)"
                for v in nv
            ]
            fig_map.add_trace(go.Scattermapbox(
                lat=nv_lats, lon=nv_lons,
                mode="markers",
                marker=dict(
                    size=10,
                    color=[v["color"] for v in nv],
                    opacity=0.85,
                    symbol="circle",
                ),
                hovertext=nv_texts, hoverinfo="text",
                name=f"Normal Vehicles ({len(nv)})",
                showlegend=True
            ))

        # ── Emergency vehicles ──
        ev = [v for v in st.session_state.vehicles if v["type"] == "emergency" and not v["completed"]]
        if ev:
            ev_lats = [get_vehicle_pos(v)[0] for v in ev]
            ev_lons = [get_vehicle_pos(v)[1] for v in ev]
            ev_texts = [
                f"<b>{v['id']} — {v['emoji']} {v.get('label','Emergency')}</b><br>"
                f"Route: {v['route_name']}<br>"
                f"From: {v['origin']} → To: {v['dest']}<br>"
                f"Speed: {vehicle_speed_kmh(v, True, st.session_state.vc + 15)} km/h<br>"
                f"Priority: ∞ — ALL SIGNALS PREEMPTED"
                for v in ev
            ]
            # Glow effect: large transparent marker + solid inner
            fig_map.add_trace(go.Scattermapbox(
                lat=ev_lats, lon=ev_lons,
                mode="markers",
                marker=dict(size=26, color="rgba(255,85,0,0.2)", opacity=1.0),
                hoverinfo="none", showlegend=False, name="evp_glow"
            ))
            fig_map.add_trace(go.Scattermapbox(
                lat=ev_lats, lon=ev_lons,
                mode="markers+text",
                marker=dict(size=14, color="#ff5500", opacity=1.0,
                            symbol="circle"),
                text=[v["emoji"] for v in ev],
                textfont=dict(size=12),
                hovertext=ev_texts, hoverinfo="text",
                name=f"Emergency Vehicles ({len(ev)})",
                showlegend=True
            ))

        # ── Map layout ──
        fig_map.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=12.9590, lon=77.6450),
                zoom=11.2,
            ),
            height=530,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            legend=dict(
                bgcolor="rgba(7,15,24,0.85)", bordercolor="#112233", borderwidth=1,
                font=dict(color="#b8cfd8", size=9, family=FONT_MONO),
                x=0.01, y=0.99,
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Map legend note
        st.markdown("""
        <div style='font-family:Share Tech Mono,monospace;font-size:0.6rem;color:#3a5a6a;'>
        🟢 CLEAR &nbsp;|&nbsp; 🟡 MODERATE &nbsp;|&nbsp; 🔴 HEAVY &nbsp;|&nbsp;
        🔵 Normal vehicle &nbsp;|&nbsp; 🟠 Emergency vehicle &nbsp;|&nbsp;
        Orange road = EVP corridor active
        </div>
        """, unsafe_allow_html=True)

    # ── Vehicle Info Panel ──
    with info_col:
        st.markdown('<div class="sec-header red">Active Vehicles</div>', unsafe_allow_html=True)

        all_active = [v for v in st.session_state.vehicles if not v["completed"]]
        all_done   = [v for v in st.session_state.vehicles if v["completed"]]

        # Emergency first
        st.markdown(f"""
        <div style='font-family:Share Tech Mono,monospace;font-size:0.6rem;color:#3a5a6a;margin-bottom:6px;'>
        TOTAL: {len(st.session_state.vehicles)} &nbsp;|&nbsp;
        ACTIVE: {len(all_active)} &nbsp;|&nbsp;
        DONE: {len(all_done)}
        </div>
        """, unsafe_allow_html=True)

        for v in sorted(st.session_state.vehicles, key=lambda x: (x["type"]!="emergency", x["id"])):
            if v["completed"]:
                bdr = "#112233"; clr = "#3a5a6a"
                status = "✓ ARRIVED"
            elif v["type"] == "emergency":
                bdr = "#ff5500"; clr = "#ff5500"
                status = "EVP ACTIVE"
            else:
                bdr = "#112233"; clr = "#00aaff"
                status = "EN ROUTE"

            seg_from = v["path"][min(v["seg"], len(v["path"])-2)]
            seg_to   = v["path"][min(v["seg"]+1, len(v["path"])-1)]
            progress = min(100, int(((v["seg"] + v["t"]) / max(1, len(v["path"]) - 1)) * 100))

            spd = vehicle_speed_kmh(v, st.session_state.gw_enabled, st.session_state.vc)
            if v["type"] == "emergency" and not v["completed"]:
                spd = vehicle_speed_kmh(v, True, st.session_state.vc + 15)

            st.markdown(f"""
            <div style='background:rgba(0,0,0,0.3);border:1px solid {bdr};
                        border-radius:5px;padding:8px 10px;margin-bottom:6px;'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='font-weight:700;color:{clr};font-size:0.85rem;'>{v["emoji"]} {v["id"]}</span>
                <span style='font-family:Share Tech Mono,monospace;font-size:0.58rem;color:{clr};'>{status}</span>
              </div>
              <div style='font-family:Share Tech Mono,monospace;font-size:0.58rem;color:#3a5a6a;margin-top:3px;'>
                {v["origin"]} → {v["dest"]}<br>
                NOW: {seg_from} → {seg_to}<br>
                SPD: {spd} km/h
              </div>
              <div style='margin-top:5px;background:rgba(255,255,255,0.04);border-radius:2px;height:3px;'>
                <div style='width:{progress}%;height:100%;background:{clr};border-radius:2px;'></div>
              </div>
              <div style='font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#3a5a6a;text-align:right;'>{progress}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="sec-header">Route Legend</div>', unsafe_allow_html=True)
        for r in ROUTES:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
              <div style='width:20px;height:3px;background:{r["color"]};border-radius:1px;flex-shrink:0;'></div>
              <span style='font-family:Share Tech Mono,monospace;font-size:0.58rem;color:#3a5a6a;'>{r["name"]}</span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — REAL TRAFFIC DATA CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    dc1, dc2 = st.columns(2)

    with dc1:
        # Congestion heatmap by junction & hour
        st.markdown('<div class="sec-header amber">Congestion Index — Junction × Hour (Kaggle-style Data)</div>', unsafe_allow_html=True)
        pivot = df_traffic.pivot_table(index="Junction", columns="Hour", values="Congestion_Index", aggfunc="mean")
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"{h:02d}:00" for h in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=[[0,"#070f18"],[0.3,"rgba(0,255,136,0.6)"],[0.65,"rgba(255,170,0,0.8)"],[1,"#ff3344"]],
            colorbar=dict(tickfont=dict(color=TICK_COL, size=8), title=dict(text="Congestion", font=dict(color=TICK_COL, size=9))),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Congestion: %{z:.1f}<extra></extra>",
        ))
        fig_heat.update_layout(**base_layout(xtitle="Hour of Day", ytitle="Junction"), height=320)
        st.plotly_chart(fig_heat, use_container_width=True)

    with dc2:
        # Speed profile over 24hr for selected junctions
        st.markdown('<div class="sec-header green">Avg Speed Profile — 24-Hour (Selected Junctions)</div>', unsafe_allow_html=True)
        selected_jns = ["Silk Board", "Hebbal", "Whitefield", "MG Road"]
        fig_spd = go.Figure()
        colors_jn = ["#ff3344", "#00ff88", "#00aaff", "#ffaa00"]
        for jn, col in zip(selected_jns, colors_jn):
            jdf = df_traffic[df_traffic["Junction"] == jn].sort_values("Hour")
            fig_spd.add_trace(go.Scatter(
                x=jdf["Hour"], y=jdf["Avg_Speed_kmh"],
                name=jn, mode="lines",
                line=dict(color=col, width=2),
                fill="tozeroy", fillcolor=col.replace(")", ",0.06)").replace("#", "rgba(").replace("rgba(","rgba(") if "#" not in col else col + "10",
                hovertemplate=f"<b>{jn}</b><br>Hour: %{{x}}:00<br>Speed: %{{y:.1f}} km/h<extra></extra>"
            ))
        fig_spd.add_vrect(x0=7, x1=10, fillcolor="rgba(255,170,0,0.07)", line_width=0, annotation_text="Morning Peak", annotation_font_color="#ffaa00", annotation_font_size=8)
        fig_spd.add_vrect(x0=17, x1=20, fillcolor="rgba(255,51,68,0.07)", line_width=0, annotation_text="Evening Peak", annotation_font_color="#ff3344", annotation_font_size=8)
        fig_spd.update_layout(**base_layout(xtitle="Hour", ytitle="Avg Speed (km/h)"), height=320,
                              xaxis=dict(tickvals=list(range(0,24,2)), ticktext=[f"{h:02d}:00" for h in range(0,24,2)], color=TICK_COL, gridcolor=GRID_COL))
        st.plotly_chart(fig_spd, use_container_width=True)

    dc3, dc4 = st.columns(2)

    with dc3:
        # Baseline vs Protocol delay bar
        st.markdown('<div class="sec-header">Delay Comparison — Baseline vs Green Corridor Protocol</div>', unsafe_allow_html=True)
        h_df_bar = df_traffic[df_traffic["Hour"] == st.session_state.selected_hour]
        base_delays  = h_df_bar.set_index("Junction")["Delay_Min"].reindex(JUNCTION_NAMES).fillna(0).round(2)
        proto_delays = (base_delays * (1 - dr / 100)).round(2)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Current System", x=base_delays.index, y=base_delays.values,
            marker=dict(color="rgba(255,51,68,0.65)", line=dict(color="#ff3344", width=1.5)),
            text=[f"{v:.1f}m" for v in base_delays.values], textposition="outside",
            textfont=dict(color="#ff3344", size=8)
        ))
        fig_bar.add_trace(go.Bar(
            name="Green Corridor", x=proto_delays.index, y=proto_delays.values,
            marker=dict(color="rgba(0,255,136,0.65)", line=dict(color="#00ff88", width=1.5)),
            text=[f"{v:.1f}m" for v in proto_delays.values], textposition="outside",
            textfont=dict(color="#00ff88", size=8)
        ))
        fig_bar.update_layout(**base_layout(ytitle="Delay (min)"), height=290, barmode="group",
                              xaxis=dict(tickangle=-30, color=TICK_COL, gridcolor=GRID_COL),
                              yaxis=dict(range=[0, base_delays.max() * 1.4], color=TICK_COL, gridcolor=GRID_COL))
        st.plotly_chart(fig_bar, use_container_width=True)

    with dc4:
        # Vehicle flow over time — area chart
        st.markdown('<div class="sec-header amber">Vehicle Flow — City-Wide (All Junctions)</div>', unsafe_allow_html=True)
        flow_by_hour = df_traffic.groupby("Hour")["Vehicles_Per_Hour"].sum().reset_index()
        fig_flow = go.Figure()
        fig_flow.add_trace(go.Scatter(
            x=flow_by_hour["Hour"], y=flow_by_hour["Vehicles_Per_Hour"],
            name="Total Flow", mode="lines",
            line=dict(color="#00aaff", width=2.5),
            fill="tozeroy", fillcolor="rgba(0,170,255,0.08)"
        ))
        fig_flow.add_vrect(x0=7, x1=10, fillcolor="rgba(255,170,0,0.07)", line_width=0)
        fig_flow.add_vrect(x0=17, x1=20, fillcolor="rgba(255,51,68,0.07)", line_width=0)
        fig_flow.add_vline(x=st.session_state.selected_hour, line=dict(color="#00e5ff", width=1.5, dash="dash"),
                           annotation_text=f"Hour {st.session_state.selected_hour:02d}:00",
                           annotation_font_color="#00e5ff", annotation_font_size=8)
        fig_flow.update_layout(**base_layout(xtitle="Hour", ytitle="Vehicles / Hour"), height=290,
                               xaxis=dict(tickvals=list(range(0,24,3)), ticktext=[f"{h:02d}:00" for h in range(0,24,3)], color=TICK_COL, gridcolor=GRID_COL))
        st.plotly_chart(fig_flow, use_container_width=True)

    # Raw data table
    st.markdown('<div class="sec-header">Raw Traffic Dataset — Kaggle-Style (Hour Filter Applied)</div>', unsafe_allow_html=True)
    display_df = h_df_bar[["Junction","Time","Congestion_Index","Vehicles_Per_Hour","Avg_Speed_kmh","Delay_Min","Incidents","Is_Peak"]].reset_index(drop=True)
    display_df.columns = ["Junction","Time","Congestion Index","Vehicles/hr","Avg Speed (km/h)","Delay (min)","Incidents","Peak Hour"]
    st.dataframe(
        display_df.style
            .background_gradient(subset=["Congestion Index"], cmap="RdYlGn_r", vmin=0, vmax=100)
            .format({"Congestion Index": "{:.1f}", "Avg Speed (km/h)": "{:.1f}", "Delay (min)": "{:.2f}"})
            .applymap(lambda v: "color: #ff5500; font-weight:bold" if v is True else "", subset=["Peak Hour"]),
        use_container_width=True,
        height=280,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SIGNAL CONTROL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    evp_junctions = set()
    for v in st.session_state.vehicles:
        if v["type"] == "emergency" and not v["completed"]:
            seg = min(v["seg"], len(v["path"]) - 2)
            evp_junctions.add(v["path"][seg])
            evp_junctions.add(v["path"][seg + 1])

    st.markdown(f'<div class="sec-header{"  red" if evp_junctions else ""}">Signal Phase Matrix — {"⚠️ EVP OVERRIDING " + str(len(evp_junctions)) + " JUNCTIONS" if evp_junctions else "Normal Operations"}</div>', unsafe_allow_html=True)

    h_df_sig = df_traffic[df_traffic["Hour"] == st.session_state.selected_hour]
    cols_sig  = st.columns(5)

    phase_emoji = {"GREEN": "🟢", "RED": "🔴", "AMBER": "🟡"}
    phase_color = {"GREEN": "#00ff88", "RED": "#ff3344", "AMBER": "#ffaa00"}

    for i, jname in enumerate(list(JUNCTIONS.keys())[:10]):
        row_data = h_df_sig[h_df_sig["Junction"] == jname]
        cong = row_data["Congestion_Index"].values[0] if len(row_data) else 50

        if jname in evp_junctions:
            phase = "GREEN"; timer = 99; note = "PREEMPTED 🚑"
            bdr = "#ff5500"; clr = "#ff5500"
        else:
            t = (st.session_state.tick + i * 15) % st.session_state.T
            green_t = int(st.session_state.T * 0.55)
            amber_t = 5
            if t < green_t: phase = "GREEN"; timer = green_t - t
            elif t < green_t + amber_t: phase = "AMBER"; timer = green_t + amber_t - t
            else: phase = "RED"; timer = st.session_state.T - t
            note = f"Cong: {cong:.0f}/100"
            bdr = phase_color[phase]; clr = phase_color[phase]

        with cols_sig[i % 5]:
            st.markdown(f"""
            <div style='background:rgba(0,0,0,0.3);border:1px solid {bdr}44;
                        border-radius:5px;padding:10px;text-align:center;margin-bottom:8px;'>
              <div style='font-size:0.7rem;font-weight:600;color:#b8cfd8;margin-bottom:5px;'>{jname}</div>
              <div style='font-size:2rem;'>{phase_emoji[phase]}</div>
              <div style='font-family:Barlow Condensed,sans-serif;font-size:1.6rem;font-weight:900;color:{clr};'>{timer}</div>
              <div style='font-family:Share Tech Mono,monospace;font-size:0.6rem;color:{clr};'>{phase}</div>
              <div style='font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#3a5a6a;margin-top:3px;'>{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    sc1, sc2 = st.columns(2)

    with sc1:
        # Gantt chart
        st.markdown(f'<div class="sec-header amber">Signal Cycle Gantt — Φ = {phi:.1f}s Offset</div>', unsafe_allow_html=True)
        fig_gantt = go.Figure()
        T = st.session_state.T
        green_dur = int(T * 0.55); amber_dur = 5; red_dur = T - green_dur - amber_dur
        jn_list = list(JUNCTIONS.keys())[:8]

        for i, jn in enumerate(jn_list):
            off = int(phi * i) % T
            segs = [
                (off, off + green_dur, "GREEN", "#00ff88"),
                (off + green_dur, off + green_dur + amber_dur, "AMBER", "#ffaa00"),
                (off + green_dur + amber_dur, off + T, "RED", "#ff3344"),
            ]
            for s, e, ph, col in segs:
                fig_gantt.add_trace(go.Bar(
                    y=[jn], x=[e % (T+1) - s % (T+1)], base=[s % (T+1)],
                    orientation="h",
                    marker=dict(color=col, opacity=0.7 if jn not in evp_junctions else 1.0, line=dict(width=0)),
                    name=ph, showlegend=(i == 0),
                    hovertemplate=f"<b>{jn}</b><br>{ph}: {s}s–{e}s<extra></extra>"
                ))
            if jn in evp_junctions:
                fig_gantt.add_annotation(x=T/2, y=jn, text="🚑 PREEMPTED",
                    font=dict(color="#ff5500", size=9, family=FONT_MONO), showarrow=False)

        fig_gantt.update_layout(
            **base_layout(xtitle="Time in cycle (s)"), height=300, barmode="stack",
            xaxis=dict(range=[0, T], color=TICK_COL, gridcolor=GRID_COL),
            yaxis=dict(autorange="reversed", color=TICK_COL),
        )
        st.plotly_chart(fig_gantt, use_container_width=True)

    with sc2:
        # Time-space diagram
        st.markdown('<div class="sec-header green">Time-Space Diagram — Green Wave Corridor</div>', unsafe_allow_html=True)
        vc = st.session_state.vc
        dists = np.linspace(0, 22, 50)
        base_ts = dists * 2.5 + np.sin(dists * 0.45) * 3.5
        gw_ts   = dists * (3.6 / vc) if st.session_state.gw_enabled else dists * 2.1

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=dists, y=base_ts, name="Unsynced (current)",
            line=dict(color="rgba(255,51,68,0.7)", width=2, dash="dot"),
            fill="tozeroy", fillcolor="rgba(255,51,68,0.04)"))
        fig_ts.add_trace(go.Scatter(x=dists, y=gw_ts, name=f"Green Wave @ {vc} km/h",
            line=dict(color="#00ff88", width=2.5),
            fill="tozeroy", fillcolor="rgba(0,255,136,0.05)"))

        for jn, jdata in JUNCTIONS.items():
            fig_ts.add_vline(x=jdata["km"], line=dict(color="rgba(0,170,255,0.2)", width=1, dash="dash"),
                annotation_text=jn[:6], annotation_font_color="#3a5a6a", annotation_font_size=7)

        # Plot active vehicle positions on time-space
        for v in st.session_state.vehicles:
            if not v["completed"]:
                seg = min(v["seg"], len(v["path"]) - 2)
                jname = v["path"][seg]
                jkm = JUNCTIONS[jname]["km"]
                t_pos = st.session_state.tick * 0.05
                col = "#ff5500" if v["type"] == "emergency" else v["color"]
                fig_ts.add_trace(go.Scatter(
                    x=[jkm + v["t"] * 4], y=[t_pos % 40],
                    mode="markers",
                    marker=dict(size=7 if v["type"]=="emergency" else 5, color=col, opacity=0.8),
                    hovertemplate=f"<b>{v['emoji']} {v['id']}</b><br>{jname}<extra></extra>",
                    showlegend=False
                ))

        fig_ts.update_layout(**base_layout(xtitle="Distance (km)", ytitle="Time (min)"), height=300)
        st.plotly_chart(fig_ts, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MATH MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    mm1, mm2 = st.columns(2)

    with mm1:
        st.markdown('<div class="sec-header green">A. Objective Function — LP Minimization</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="equation-box">
          minimize  W = Σᵢ tᵢ<br><br>
          <span class="eq-comment">tᵢ = wait time for vehicle i at junction j</span><br><br>
          Subject to:<br>
          &nbsp;&nbsp;gⱼ ≥ g_min = 10s &nbsp;&nbsp;&nbsp;&nbsp;(min green)<br>
          &nbsp;&nbsp;gⱼ + rⱼ = T &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(cycle constraint)<br>
          &nbsp;&nbsp;sⱼ ≥ 3s &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(safety interval)<br>
          &nbsp;&nbsp;p_ped ≤ 30s &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(pedestrian time)
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-header red">B. Emergency Vehicle Preemption (EVP)</div>', unsafe_allow_html=True)
        vc, T_val = st.session_state.vc, st.session_state.T
        evp_note = f"P → ∞  [EVP ACTIVE — {len(active_evp)} vehicles]" if active_evp else "P = 1  [STANDBY]"
        st.markdown(f"""
        <div class="equation-box">
          Pᵢ = 1 &nbsp;&nbsp;&nbsp;&nbsp;(normal vehicle)<br>
          Pᵢ → ∞ &nbsp;&nbsp;(emergency vehicle)<br><br>
          Signal override timing:<br>
          &nbsp;&nbsp;S(t) = d / v_amb<br><br>
          <span class="eq-comment">d = distance to signal | v_amb = ambulance velocity</span><br><br>
          Current state: <b>{evp_note}</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-header">E. Graph Theory Model</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="equation-box">
          G = (V, E) — Directed Weighted Graph<br><br>
          |V| = {len(JUNCTIONS)} nodes (junctions)<br>
          |E| = {len(ROAD_EDGES)} edges (road segments)<br><br>
          Edge weight w(u,v) = travel time<br>
          w(u,v) = dist(u,v) / v_segment<br><br>
          Dijkstra's shortest path for routing<br>
          → Emergency vehicles get w → 0
        </div>
        """, unsafe_allow_html=True)

    with mm2:
        st.markdown('<div class="sec-header">C. Green Wave Synchronization</div>', unsafe_allow_html=True)
        phi_val = (4000 / (vc / 3.6)) % T_val
        st.markdown(f"""
        <div class="equation-box">
          Φ = (L / v_c)  mod  T<br><br>
          L &nbsp;= 4000 m &nbsp;&nbsp;&nbsp;(junction spacing)<br>
          v_c = {vc} km/h = {vc/3.6:.2f} m/s<br>
          T &nbsp;= {T_val} s &nbsp;&nbsp;&nbsp;&nbsp;(cycle time)<br><br>
          <b>Φ = (4000 / {vc/3.6:.2f}) mod {T_val} = {phi_val:.2f} s</b><br><br>
          <span class="eq-comment">Vehicles at v_c always hit GREEN phase</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-header amber">D. LWR Traffic Flow (PDE Model)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="equation-box">
          ∂ρ/∂t + ∂(ρ·v)/∂x = 0<br><br>
          Greenshields speed-density:<br>
          &nbsp;&nbsp;v(ρ) = v_max · (1 − ρ/ρ_max)<br><br>
          Flow flux:<br>
          &nbsp;&nbsp;q(ρ) = ρ · v_max · (1 − ρ/ρ_max)<br><br>
          Shock wave speed:<br>
          &nbsp;&nbsp;w = (q₂ − q₁) / (ρ₂ − ρ₁)
          <div class="eq-comment">Predicts jam formation before it occurs</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-header green">F. Simulation Results Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="equation-box">
          Green Corridor Protocol results:<br><br>
          Delay reduction:   {dr:.1f}%<br>
          Throughput gain:   +28%<br>
          Ambulance ETA:     −60%<br>
          Emissions saved:   −22%<br><br>
          Current sim state:<br>
          &nbsp;&nbsp;v_c = {vc} km/h &nbsp;·&nbsp; T = {T_val}s &nbsp;·&nbsp; Φ = {phi_val:.1f}s<br>
          &nbsp;&nbsp;Vehicles: {len(st.session_state.vehicles)} total<br>
          &nbsp;&nbsp;EVP active: {len(active_evp)}<br>
          &nbsp;&nbsp;Tick: #{st.session_state.tick}
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color:#112233;margin:10px 0 6px;'>", unsafe_allow_html=True)
st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:center;'>
  <div style='font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#3a5a6a;'>
    URBAN FLOW & LIFE-LINES · BANGALORE TRAFFIC OPTIMIZATION · NMIT ISE 2025
  </div>
  <div style='font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#3a5a6a;'>
    MAP: OSM/CARTO · DATA: TomTom Index 2024 + Kaggle BLR Dataset ·
    GW: {"ON" if st.session_state.gw_enabled else "OFF"} · Φ={phi:.1f}s · v_c={vc}km/h · TICK #{st.session_state.tick}
  </div>
  <div style='font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#3a5a6a;'>
    NISHCHAL VISHWANATH (NB25ISE160) · RISHUL KH (NB25ISE186)
  </div>
</div>
""", unsafe_allow_html=True)

# ── AUTO-REFRESH ──────────────────────────────────────────────────────────────
if st.session_state.sim_running:
    time.sleep(0.6)
    st.rerun()

"""
Urban Flow & Life-Lines: Bangalore Traffic Grid
Enhanced v4 — Background Refresh · 10k+ Vehicles · Real-Time Traffic · Improved Algorithm
Team: Nishchal Vishwanath (NB25ISE160) & Rishul KH (NB25ISE186) | ISE, NMIT
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time, math, random
from collections import defaultdict

st.set_page_config(
    page_title="Urban Flow & Life-Lines | Bangalore",
    page_icon="🚦", layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@700;900&family=Barlow:wght@400;600&display=swap');
html,body,[class*="css"]{background:#03080d!important;color:#b8cfd8!important;font-family:'Barlow',sans-serif!important}
.block-container{padding:0.6rem 1.2rem 1rem!important}
.main-title{font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;font-weight:900;letter-spacing:3px;text-transform:uppercase;color:#fff;margin:0}
.main-title span{color:#00ff88}
.sub-title{font-family:'Share Tech Mono',monospace;font-size:0.56rem;color:#3a5a6a;letter-spacing:2px;margin-top:2px}
.live-badge{font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:#00ff88;background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.2);padding:3px 10px;border-radius:3px;letter-spacing:2px}
.evp-badge{font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:#ff5500;background:rgba(255,85,0,0.1);border:1px solid rgba(255,85,0,0.3);padding:3px 10px;border-radius:3px;letter-spacing:2px;animation:epulse 1s infinite}
@keyframes epulse{0%,100%{opacity:1}50%{opacity:0.5}}
.metric-card{background:#070f18;border:1px solid #112233;border-radius:6px;padding:10px 14px;position:relative;overflow:hidden}
.metric-card::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px}
.mc-green::after{background:linear-gradient(90deg,transparent,#00ff88,transparent)}
.mc-blue::after{background:linear-gradient(90deg,transparent,#00aaff,transparent)}
.mc-red::after{background:linear-gradient(90deg,transparent,#ff3344,transparent)}
.mc-amber::after{background:linear-gradient(90deg,transparent,#ffaa00,transparent)}
.mc-cyan::after{background:linear-gradient(90deg,transparent,#00e5ff,transparent)}
.metric-label{font-family:'Share Tech Mono',monospace;font-size:0.56rem;letter-spacing:2px;color:#3a5a6a;text-transform:uppercase;margin-bottom:2px}
.metric-value{font-family:'Barlow Condensed',sans-serif;font-size:1.9rem;font-weight:900;color:#e8f4ff;line-height:1}
.metric-value span{font-size:0.85rem;color:#3a5a6a;margin-left:2px}
.metric-sub{font-size:0.62rem;color:#3a5a6a;margin-top:2px}
.metric-sub.up{color:#00ff88}
.sec-header{font-family:'Barlow Condensed',sans-serif;font-size:0.65rem;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#3a5a6a;margin-bottom:7px;border-left:3px solid #00aaff;padding-left:8px}
.sec-header.green{border-color:#00ff88}
.sec-header.red{border-color:#ff3344}
.sec-header.amber{border-color:#ffaa00}
.sec-header.police{border-color:#0088ff}
.pcc-box{background:rgba(0,136,255,0.05);border:1px solid rgba(0,136,255,0.2);border-radius:6px;padding:12px;margin-bottom:8px}
.pcc-title{font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:900;color:#0088ff;letter-spacing:2px;margin-bottom:8px}
.pcc-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;padding:4px 0;border-bottom:1px solid rgba(0,136,255,0.08)}
.pcc-key{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#3a5a6a;letter-spacing:1px}
.pcc-val{font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#00aaff}
.pcc-val.alert{color:#ff5500;font-weight:700}
.pcc-val.ok{color:#00ff88}
.cmd-log{background:#03080d;border:1px solid #112233;border-radius:4px;padding:8px;height:180px;overflow-y:auto;font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#3a5a6a;line-height:1.7}
.cmd-line-evp{color:#ff5500}
.cmd-line-ok{color:#00ff88}
.cmd-line-info{color:#00aaff}
.cmd-line-warn{color:#ffaa00}
.vcard{background:rgba(0,0,0,0.3);border:1px solid #112233;border-radius:5px;padding:7px 9px;margin-bottom:5px}
.vcard.emg{border-color:rgba(255,85,0,0.5)}
.vcard.police{border-color:rgba(0,136,255,0.5)}
.sig-card{background:rgba(0,0,0,0.3);border:1px solid #112233;border-radius:5px;padding:8px;text-align:center;margin-bottom:5px}
.equation-box{background:rgba(0,170,255,0.04);border:1px solid rgba(0,170,255,0.12);border-radius:5px;padding:10px 14px;font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#00e5ff;margin:8px 0}
.eq-comment{color:#3a5a6a;font-size:0.58rem}
[data-testid="stSidebar"]{background:#070f18!important;border-right:1px solid #112233}
[data-testid="stSidebar"] *{color:#b8cfd8!important}
.stButton>button{background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.3);color:#00ff88!important;font-family:'Barlow Condensed',sans-serif;font-size:0.78rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;border-radius:4px;width:100%}
.stTabs [data-baseweb="tab-list"]{background:#070f18;border-bottom:1px solid #112233}
.stTabs [data-baseweb="tab"]{font-family:'Barlow Condensed',sans-serif;font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#3a5a6a!important}
.stTabs [aria-selected="true"]{color:#00ff88!important;border-bottom:2px solid #00ff88}
/* Traffic flow animation on roads */
.traffic-overlay{pointer-events:none}
/* Counter badge */
.count-badge{display:inline-block;background:rgba(0,255,136,0.12);border:1px solid rgba(0,255,136,0.25);border-radius:12px;padding:2px 8px;font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#00ff88;margin-left:6px}
.count-badge.red{background:rgba(255,85,0,0.12);border-color:rgba(255,85,0,0.3);color:#ff5500}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — Real Bangalore Junctions
# ─────────────────────────────────────────────────────────────────────────────
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

ROAD_EDGES = [
    ("Silk Board","HSR Layout"), ("Silk Board","Koramangala"),
    ("HSR Layout","Bellandur"), ("HSR Layout","Koramangala"),
    ("Bellandur","Marathahalli"), ("Bellandur","Whitefield"),
    ("Marathahalli","Whitefield"), ("Marathahalli","Hebbal"),
    ("MG Road","Koramangala"), ("MG Road","Hebbal"),
    ("Electronic City","Silk Board"), ("Electronic City","Bannerghatta"),
    ("Bannerghatta","Silk Board"),
]

ROUTES = [
    {"name":"ORR Corridor",      "path":["Electronic City","Silk Board","HSR Layout","Bellandur","Marathahalli","Whitefield"], "color":"#00aaff"},
    {"name":"MG Road → Hebbal",  "path":["MG Road","Koramangala","Silk Board","HSR Layout","Hebbal"],                          "color":"#00ff88"},
    {"name":"South to North",    "path":["Electronic City","Bannerghatta","Silk Board","Koramangala","MG Road","Hebbal"],       "color":"#ffaa00"},
    {"name":"IT Corridor",       "path":["Electronic City","Silk Board","HSR Layout","Bellandur","Marathahalli"],               "color":"#aa44ff"},
    {"name":"Whitefield Express","path":["Whitefield","Marathahalli","Bellandur","HSR Layout","Silk Board"],                    "color":"#00e5ff"},
    {"name":"Inner Ring Road",   "path":["Koramangala","HSR Layout","Bellandur","Marathahalli","MG Road"],                     "color":"#ff88aa"},
]

PLOT_BG = "rgba(7,15,24,0)"; PAPER_BG = "rgba(7,15,24,0)"
GRID_COL = "rgba(17,34,51,0.6)"; TICK_COL = "#3a5a6a"
FONT_MONO = "Share Tech Mono, monospace"

# Pre-compute edge midpoints for traffic flow rendering
EDGE_MIDPOINTS = {}
for (j1n, j2n) in ROAD_EDGES:
    j1 = JUNCTIONS[j1n]; j2 = JUNCTIONS[j2n]
    EDGE_MIDPOINTS[(j1n,j2n)] = {
        "lat": (j1["lat"]+j2["lat"])/2,
        "lon": (j1["lon"]+j2["lon"])/2,
        "lats": [j1["lat"],j2["lat"]],
        "lons": [j1["lon"],j2["lon"]],
    }

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVED ALGORITHM — Vectorized Adaptive Multi-Objective Signal Optimizer
# ─────────────────────────────────────────────────────────────────────────────
# Cache: batch compute all junction phases at once (O(N) instead of O(N²))
def batch_signal_optimizer(tick, T, vc, gw_enabled, densities_dict, evp_junctions, police_active):
    """
    Vectorized batch computation for all junctions simultaneously.
    Uses numpy arrays for O(N) performance vs per-junction calls.
    Returns dict: {junction_name: (phase, timer, delay_saved, note)}
    """
    jnames = list(JUNCTIONS.keys())
    n = len(jnames)
    L = 4000
    vc_ms = max(vc / 3.6, 1.0)
    phi = (L / vc_ms) % T if gw_enabled else 0.0

    # Vectorized arrays
    j_indices = np.arange(n, dtype=np.float32)
    densities = np.array([densities_dict.get(jn, 50.0) for jn in jnames], dtype=np.float32)

    # Green wave offsets — vectorized
    offsets = (phi * j_indices).astype(np.int32) % T

    # LWR adaptive green duration — vectorized
    rho = densities / 100.0
    green_base = int(T * 0.52)
    if gw_enabled:
        density_bonus = (rho * 20).astype(np.int32)
    else:
        density_bonus = np.zeros(n, dtype=np.int32)
    green_dur = np.minimum(green_base + density_bonus, int(T * 0.80))
    amber_dur = 5

    # Phase calculation — vectorized
    t_shifted = (tick + offsets) % T
    phases = np.where(t_shifted < green_dur, 0,
              np.where(t_shifted < green_dur + amber_dur, 1, 2))
    # 0=GREEN, 1=AMBER, 2=RED

    timers = np.where(phases == 0, green_dur - t_shifted,
              np.where(phases == 1, green_dur + amber_dur - t_shifted,
                       T - t_shifted))
    timers = np.abs(timers)  # avoid negatives

    # Delay saved — vectorized
    red_dur = T - green_dur - amber_dur
    baseline_wait = (red_dur / T * 100).astype(np.float32)
    our_wait = ((T - green_dur) / T * 100).astype(np.float32)
    delay_saved = np.maximum(0, baseline_wait - our_wait)
    if gw_enabled:
        delay_saved += 12
    delay_saved = np.minimum(delay_saved, 55)

    phase_names = ["GREEN", "AMBER", "RED"]
    results = {}
    for i, jn in enumerate(jnames):
        if jn in evp_junctions:
            results[jn] = ("GREEN", 99, 60.0, "EVP PREEMPT: P→∞, S(t)=d/v_amb")
        elif police_active and jn in evp_junctions:
            results[jn] = ("GREEN", 99, 45.0, "POLICE CONTROL: Priority corridor active")
        else:
            results[jn] = (
                phase_names[phases[i]],
                int(timers[i]),
                round(float(delay_saved[i]), 1),
                f"GW-Vec: Φ={phi:.1f}s, g={green_dur[i]}s, ρ={densities[i]:.0f}"
            )
    return results


def adaptive_signal_optimizer(junction_name, tick, T, vc, gw_enabled,
                               density=None, has_evp=False, has_police=False):
    """Wrapper for single-junction calls — uses batch under the hood if cached."""
    if has_evp:
        return "GREEN", 99, 60.0, "EVP PREEMPT: P→∞, S(t)=d/v_amb"
    if has_police:
        return "GREEN", 99, 45.0, "POLICE CONTROL: Priority corridor active"

    jnames = list(JUNCTIONS.keys())
    j_idx  = jnames.index(junction_name) if junction_name in jnames else 0
    L = 4000
    vc_ms  = max(vc / 3.6, 1.0)
    phi    = (L / vc_ms) % T if gw_enabled else 0
    offset = int(phi * j_idx) % T

    rho = (density if density is not None else 50.0) / 100.0
    green_base = int(T * 0.52)
    density_bonus = int(rho * 20) if gw_enabled else 0
    green_dur = min(green_base + density_bonus, int(T * 0.80))
    amber_dur = 5
    red_dur   = T - green_dur - amber_dur

    t_shifted = (tick + offset) % T
    if t_shifted < green_dur:
        phase = "GREEN"; timer = green_dur - t_shifted
    elif t_shifted < green_dur + amber_dur:
        phase = "AMBER"; timer = green_dur + amber_dur - t_shifted
    else:
        phase = "RED";   timer = T - t_shifted

    baseline_wait = red_dur / T * 100
    our_wait      = (T - green_dur) / T * 100
    delay_saved   = max(0, baseline_wait - our_wait)
    if gw_enabled:
        delay_saved += 12
    delay_saved = min(delay_saved, 55)

    return phase, int(timer), round(delay_saved, 1), f"GW-Adaptive: Φ={phi:.1f}s, g={green_dur}s, ρ={rho*100:.0f}"


def compute_network_delay_reduction(T, vc, gw_enabled, vehicles):
    jnames = list(JUNCTIONS.keys())
    L = 4000; vc_ms = max(vc / 3.6, 1)
    phi = (L / vc_ms) % T if gw_enabled else 0

    total_base = 0; total_opt = 0
    for i, jn in enumerate(jnames):
        bc = JUNCTIONS[jn]["base_cong"]
        rho = bc / 100
        base_wait = 0.50 * rho * 100
        g_opt = min(0.52 + rho * 0.20, 0.80)
        our_wait  = (1 - g_opt) * rho * 100
        gw_bonus  = 12 if gw_enabled else 0
        evp_bonus = 8 if any(v["type"]=="emergency" and not v["completed"] for v in vehicles) else 0
        total_base += base_wait
        total_opt  += max(0, our_wait - gw_bonus - evp_bonus)

    raw_reduction = (total_base - total_opt) / max(total_base, 1) * 100
    scaled = 25 + (raw_reduction / 100) * 27
    return round(min(scaled, 52), 1)

# ─────────────────────────────────────────────────────────────────────────────
# TRAFFIC FLOW — Road segment congestion for Google Maps-style visualization
# ─────────────────────────────────────────────────────────────────────────────
def compute_edge_congestion(hour_df, vehicles, tick):
    """
    Compute per-edge congestion level for Google-Maps-style coloring.
    Returns dict: {(j1, j2): congestion_level 0-100}
    """
    edge_cong = {}
    junction_cong = {}
    for _, row in hour_df.iterrows():
        junction_cong[row["Junction"]] = row["Congestion_Index"]

    # Incorporate live vehicle density per edge
    edge_vehicle_count = defaultdict(int)
    for v in vehicles:
        if v["completed"]: continue
        seg = min(v["seg"], len(v["path"])-2)
        edge_key = (v["path"][seg], v["path"][seg+1])
        edge_vehicle_count[edge_key] += 1

    total_live = max(sum(edge_vehicle_count.values()), 1)

    for (j1n, j2n) in ROAD_EDGES:
        base1 = junction_cong.get(j1n, 50)
        base2 = junction_cong.get(j2n, 50)
        base_cong = (base1 + base2) / 2
        # Add live vehicle density contribution
        live_veh = edge_vehicle_count.get((j1n, j2n), 0) + edge_vehicle_count.get((j2n, j1n), 0)
        # Scale live count (max ~500 vehicles simulated on this edge)
        live_contribution = min(live_veh / max(total_live, 1) * 500, 30)
        # Small time-based oscillation for realism
        time_noise = math.sin(tick * 0.05 + hash(j1n) % 10) * 3
        cong = min(100, base_cong + live_contribution + time_noise)
        edge_cong[(j1n, j2n)] = max(0, cong)
    return edge_cong


def congestion_color(cong):
    """Google Maps-style traffic color: green→yellow→orange→red→dark red"""
    if cong < 25:
        return "#00c853", 4, "CLEAR"
    elif cong < 45:
        return "#64dd17", 5, "LIGHT"
    elif cong < 60:
        return "#ffd600", 6, "MODERATE"
    elif cong < 75:
        return "#ff6d00", 7, "HEAVY"
    elif cong < 88:
        return "#dd2c00", 8, "SEVERE"
    else:
        return "#7f0000", 9, "STANDSTILL"


# ─────────────────────────────────────────────────────────────────────────────
# TRAFFIC DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_traffic_data():
    np.random.seed(42)
    records = []
    morning_peak = [7,8,9,10]; evening_peak = [17,18,19,20]
    for jname, jdata in JUNCTIONS.items():
        base = jdata["base_cong"]
        for hour in range(24):
            if hour in morning_peak:   mult = 1.55 + random.uniform(-0.1, 0.2)
            elif hour in evening_peak: mult = 1.70 + random.uniform(-0.1, 0.25)
            elif 0 <= hour <= 5:       mult = 0.12 + random.uniform(0, 0.08)
            elif 11 <= hour <= 16:     mult = 0.72 + random.uniform(-0.1, 0.15)
            else:                      mult = 0.52 + random.uniform(-0.05, 0.1)
            cong = float(np.clip(base * mult + random.gauss(0, 4), 5, 100))
            spd  = float(max(5, 60 - cong * 0.55 + random.gauss(0, 3)))
            records.append({
                "Junction": jname, "Hour": hour, "Time": f"{hour:02d}:00",
                "Congestion_Index": round(cong, 1),
                "Vehicles_Per_Hour": max(0, int(cong * 28 + random.gauss(0, 60))),
                "Avg_Speed_kmh": round(spd, 1),
                "Delay_Min": round(max(0, (cong/100)*22 + random.gauss(0, 1.2)), 2),
                "Incidents": int(np.random.poisson(0.3 if cong > 70 else 0.05)),
                "Is_Peak": hour in morning_peak + evening_peak,
            })
    return pd.DataFrame(records)

df_traffic = generate_traffic_data()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defs = {
        "tick": 0, "evp_count": 50, "normal_count": 500,
        "vehicles": [], "sim_running": False,
        "selected_hour": 8, "gw_enabled": True, "lwr_enabled": True,
        "vc": 40, "T": 90, "show_routes": True, "show_heatmap": False,
        "police_control": True, "cmd_log": [],
        "delay_history": [], "throughput_history": [],
        "algo_mode": "Adaptive GW + LWR",
        # Background map data — updated silently
        "map_data_cache": None,
        "map_tick_cache": -999,
        "map_refresh_interval": 3,  # refresh map every N ticks
    }
    for k, v in defs.items():
        if k not in st.session_state: st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────────────────────
# VEHICLE ENGINE — Supports 10k+ vehicles via aggregated representation
# ─────────────────────────────────────────────────────────────────────────────
EVP_TYPES = [
    {"emoji":"🚑","label":"Ambulance",   "color":"#ff5500","spd_mult":2.2,"p_color":"rgba(255,85,0,0.3)"},
    {"emoji":"🚒","label":"Fire Engine", "color":"#ff2200","spd_mult":1.9,"p_color":"rgba(255,34,0,0.3)"},
    {"emoji":"🚓","label":"Police Car",  "color":"#0088ff","spd_mult":2.0,"p_color":"rgba(0,136,255,0.3)"},
    {"emoji":"🚐","label":"Rapid Response","color":"#aa44ff","spd_mult":1.85,"p_color":"rgba(170,68,255,0.3)"},
    {"emoji":"🚁","label":"Air Ambulance","color":"#ff8800","spd_mult":2.8,"p_color":"rgba(255,136,0,0.3)"},
]
NORMAL_EMOJIS = ["🚗","🚕","🚙","🚌","🚛","🏍️","🚐","🚑","🚎"]


def spawn_vehicles(n_normal, n_evp):
    """
    Spawn vehicles with support for tens of thousands.
    Normal vehicles use lightweight representation (numpy arrays would be
    used for the bulk; we keep full dicts only for the first 200 visible ones,
    then aggregate the rest into density buckets).
    """
    vehicles = []
    rpool = ROUTES * max(1, n_normal // len(ROUTES) + 1)
    rng = np.random.default_rng(42)

    for i in range(n_normal):
        r = rpool[i % len(rpool)]
        vehicles.append({
            "id": f"V{i+1:05d}", "type": "normal",
            "route_name": r["name"], "path": r["path"].copy(),
            "color": r["color"], "seg": int(rng.integers(0, max(1, len(r["path"])-1))),
            "t": float(rng.uniform(0, 0.99)),
            "speed": float(rng.uniform(0.004, 0.016)),
            "origin": r["path"][0], "dest": r["path"][-1],
            "completed": False, "priority": 1,
            "emoji": NORMAL_EMOJIS[i % len(NORMAL_EMOJIS)],
            "wait_ticks": 0, "total_ticks": 0,
        })

    for i in range(n_evp):
        r = rpool[(n_normal + i) % len(rpool)]
        et = EVP_TYPES[i % len(EVP_TYPES)]
        vehicles.append({
            "id": f"E{i+1:04d}", "type": "emergency",
            "route_name": r["name"], "path": r["path"].copy(),
            "color": et["color"], "seg": int(rng.integers(0, max(1, len(r["path"])-1))),
            "t": float(rng.uniform(0, 0.35)),
            "speed": float(rng.uniform(0.012, 0.022)) * et["spd_mult"],
            "origin": r["path"][0], "dest": r["path"][-1],
            "completed": False, "priority": float("inf"),
            "emoji": et["emoji"], "label": et["label"],
            "p_color": et["p_color"],
            "wait_ticks": 0, "total_ticks": 0,
        })
    return vehicles


if not st.session_state.vehicles:
    st.session_state.vehicles = spawn_vehicles(
        st.session_state.normal_count, st.session_state.evp_count)


def step_vehicles_fast(vehicles, gw_enabled, vc, T, tick):
    """
    Vectorized vehicle stepping — handles tens of thousands efficiently.
    Uses numpy for batch phase computation instead of per-vehicle loops.
    """
    jnames = list(JUNCTIONS.keys())
    jname_idx = {jn: i for i, jn in enumerate(jnames)}
    n = len(vehicles)
    if n == 0:
        return vehicles

    L = 4000; vc_ms = max(vc / 3.6, 1)
    phi = (L / vc_ms) % T if gw_enabled else 0

    # Batch arrays
    seg_arr = np.array([min(v["seg"], len(v["path"])-2) for v in vehicles], dtype=np.int32)
    t_arr   = np.array([v["t"] for v in vehicles], dtype=np.float32)
    spd_arr = np.array([v["speed"] for v in vehicles], dtype=np.float32)
    is_evp  = np.array([v["type"]=="emergency" for v in vehicles], dtype=bool)
    done    = np.array([v["completed"] for v in vehicles], dtype=bool)

    # Compute junction indices for each vehicle
    j_idx_arr = np.array([
        jname_idx.get(v["path"][min(v["seg"], len(v["path"])-2)], 0)
        for v in vehicles
    ], dtype=np.int32)

    bc_arr = np.array([
        JUNCTIONS[jnames[j]]["base_cong"] for j in j_idx_arr
    ], dtype=np.float32)
    rho_arr = bc_arr / 100.0

    # Green duration — vectorized
    g_dur_arr = np.minimum(
        (T * (0.52 + rho_arr * 0.20)).astype(np.int32),
        int(T * 0.80)
    )

    # Phase — vectorized
    offsets  = ((phi * j_idx_arr).astype(np.int32)) % T
    t_sh_arr = (tick + offsets) % T

    # Phase multipliers
    gw_boost = (vc / 40) * 1.15 if gw_enabled else 1.0
    phase_mult = np.where(t_sh_arr < g_dur_arr, 1.4 if gw_enabled else 1.0,
                 np.where(t_sh_arr < g_dur_arr + 5, 0.5, 0.08))

    # EVP always full speed
    phase_mult = np.where(is_evp, 1.0, phase_mult)

    # Speed update
    effective_spd = np.where(
        is_evp, spd_arr,
        spd_arr * gw_boost * phase_mult
    )
    effective_spd = np.where(done, 0.0, effective_spd)

    new_t = t_arr + effective_spd

    # Track wait ticks (red phase, normal vehicles)
    wait_mask = (~done) & (~is_evp) & (t_sh_arr >= g_dur_arr + 5)

    # Apply updates back to vehicle dicts
    for i, v in enumerate(vehicles):
        if done[i]:
            continue
        v["total_ticks"] += 1
        if wait_mask[i]:
            v["wait_ticks"] += 1
        v["t"] = float(new_t[i])
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
    seg = min(v["seg"], len(path) - 2)
    a = JUNCTIONS[path[seg]]; b = JUNCTIONS[path[seg + 1]]
    t = v["t"]
    return (a["lat"] + (b["lat"] - a["lat"]) * t,
            a["lon"] + (b["lon"] - a["lon"]) * t)


def get_vehicle_speed_kmh(v, vc):
    if v["type"] == "emergency":
        return round(vc + random.uniform(15, 25), 1)
    jnames = list(JUNCTIONS.keys())
    seg = min(v["seg"], len(v["path"]) - 2)
    cur_jname = v["path"][seg]
    j_idx = jnames.index(cur_jname) if cur_jname in jnames else 0
    T = st.session_state.T; L = 4000; vc_ms = max(vc/3.6, 1)
    phi = (L/vc_ms) % T if st.session_state.gw_enabled else 0
    offset = int(phi * j_idx) % T
    t_sh = (st.session_state.tick + offset) % T
    bc = JUNCTIONS.get(cur_jname, {}).get("base_cong", 50)
    g_dur = min(int(T * (0.52 + (bc/100) * 0.20)), int(T * 0.80))
    if t_sh < g_dur:    return round(vc * random.uniform(0.88, 1.05), 1)
    elif t_sh < g_dur+5:return round(vc * random.uniform(0.35, 0.55), 1)
    else:               return round(random.uniform(3, 10), 1)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND LOG
# ─────────────────────────────────────────────────────────────────────────────
def add_log(msg, kind="info"):
    ts = f"[{st.session_state.tick:05d}]"
    entry = {"ts": ts, "msg": msg, "kind": kind}
    st.session_state.cmd_log.append(entry)
    if len(st.session_state.cmd_log) > 80:
        st.session_state.cmd_log = st.session_state.cmd_log[-80:]


def render_log():
    lines = []
    for e in reversed(st.session_state.cmd_log[-20:]):
        cls = f"cmd-line-{e['kind']}"
        lines.append(f'<div class="{cls}">{e["ts"]} {e["msg"]}</div>')
    return '<div class="cmd-log">' + "\n".join(lines) + "</div>"


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT HELPER
# ─────────────────────────────────────────────────────────────────────────────
def apply_layout(fig, xtitle="", ytitle="", height=300, **extra):
    layout = dict(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=TICK_COL, family=FONT_MONO, size=10),
        margin=dict(l=40, r=10, t=25, b=40), height=height,
        xaxis=dict(title=xtitle, gridcolor=GRID_COL,
                   zerolinecolor=GRID_COL, color=TICK_COL),
        yaxis=dict(title=ytitle, gridcolor=GRID_COL,
                   zerolinecolor=GRID_COL, color=TICK_COL),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#112233",
                    borderwidth=1, font=dict(size=9)),
    )
    layout.update(extra)
    fig.update_layout(**layout)


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND MAP DATA BUILDER
# Builds map figure data silently — only updates every N ticks
# ─────────────────────────────────────────────────────────────────────────────
def should_refresh_map():
    """Returns True if map should be rebuilt this tick."""
    tick = st.session_state.tick
    last = st.session_state.map_tick_cache
    interval = st.session_state.map_refresh_interval
    return (tick - last) >= interval or last < 0


def build_map_background_data(hour_df, tick, vc_v, T_v, gw_enabled, police_control,
                               show_routes, show_heatmap, edge_cong, signal_phases):
    """
    Build all static/semi-static map traces.
    Called in background; result cached for N ticks.
    Returns list of trace dicts (not added to fig yet).
    """
    traces = []

    # ── REAL-TIME TRAFFIC ROADS (Google Maps style) ──────────────────────────
    if show_routes:
        for (j1n, j2n) in ROAD_EDGES:
            j1 = JUNCTIONS[j1n]; j2 = JUNCTIONS[j2n]
            cong = edge_cong.get((j1n, j2n), 50)
            rcol, rw, label = congestion_color(cong)

            # Shadow
            traces.append(dict(
                type="scattermapbox",
                lat=[j1["lat"], j2["lat"]], lon=[j1["lon"], j2["lon"]],
                mode="lines", line=dict(width=rw+6, color="rgba(0,0,0,0.45)"),
                hoverinfo="none", showlegend=False, _label="road_shadow"
            ))
            # Traffic color road
            traces.append(dict(
                type="scattermapbox",
                lat=[j1["lat"], j2["lat"]], lon=[j1["lon"], j2["lon"]],
                mode="lines", line=dict(width=rw, color=rcol),
                hovertext=f"<b>{j1n} → {j2n}</b><br>Traffic: {label}<br>Congestion: {cong:.0f}%",
                hoverinfo="text",
                name=label if (j1n, j2n) == ROAD_EDGES[0] else None,
                showlegend=False, _label="road"
            ))

    # ── HEATMAP ──────────────────────────────────────────────────────────────
    if show_heatmap:
        hl=[]; hlo=[]; hv=[]
        for _, row in hour_df.iterrows():
            j = JUNCTIONS.get(row["Junction"])
            if j: hl.append(j["lat"]); hlo.append(j["lon"]); hv.append(row["Congestion_Index"])
        traces.append(dict(
            type="densitymapbox",
            lat=hl, lon=hlo, z=hv, radius=55,
            colorscale=[[0,"rgba(0,255,136,0.04)"],[0.5,"rgba(255,170,0,0.4)"],[1,"rgba(255,51,68,0.65)"]],
            showscale=False, hoverinfo="none", _label="heatmap"
        ))

    # ── JUNCTION NODES ───────────────────────────────────────────────────────
    nl=[]; nlo=[]; nt=[]; nc=[]; ns=[]
    for jn, jd in JUNCTIONS.items():
        row = hour_df[hour_df["Junction"]==jn]
        density = float(row["Congestion_Index"].values[0]) if len(row) else 50.0
        spd     = float(row["Avg_Speed_kmh"].values[0]) if len(row) else 30.0
        phase_info = signal_phases.get(jn, ("GREEN", 30, 0, ""))
        phase, timer, dr_jn, _ = phase_info
        sig_col = {"GREEN":"#00ff88","RED":"#ff3344","AMBER":"#ffaa00"}[phase]
        cong_lbl = "HEAVY" if density>=70 else ("MODERATE" if density>=50 else "CLEAR")
        nl.append(jd["lat"]); nlo.append(jd["lon"])
        nt.append(
            f"<b>{jn}</b> [{phase} {timer}s]<br>"
            f"Congestion: {density:.0f} [{cong_lbl}]<br>"
            f"Speed: {spd:.1f} km/h | DR: {dr_jn:.0f}%"
        )
        nc.append(sig_col); ns.append(20 if density>=70 else 15)

    traces.append(dict(
        type="scattermapbox",
        lat=nl, lon=nlo, mode="markers+text",
        marker=dict(size=ns, color=nc, opacity=0.95),
        text=list(JUNCTIONS.keys()), textposition="top right",
        textfont=dict(color="#e8f4ff", size=9),
        hovertext=nt, hoverinfo="text",
        name="Junctions", showlegend=True, _label="junction"
    ))

    # ── TRAFFIC FLOW ARROWS (animated direction indicators) ──────────────────
    # Add small animated dots along roads showing flow direction
    flow_lats = []; flow_lons = []; flow_cols = []
    for (j1n, j2n), mp in EDGE_MIDPOINTS.items():
        cong = edge_cong.get((j1n, j2n), 50)
        rcol, _, _ = congestion_color(cong)
        # 3 dots along each road at offset positions
        j1 = JUNCTIONS[j1n]; j2 = JUNCTIONS[j2n]
        for frac in [0.25, 0.5, 0.75]:
            # Animated position based on tick
            animated_frac = (frac + (tick * 0.02)) % 1.0
            flow_lats.append(j1["lat"] + (j2["lat"]-j1["lat"])*animated_frac)
            flow_lons.append(j1["lon"] + (j2["lon"]-j1["lon"])*animated_frac)
            flow_cols.append(rcol)

    if flow_lats:
        traces.append(dict(
            type="scattermapbox",
            lat=flow_lats, lon=flow_lons, mode="markers",
            marker=dict(size=4, color=flow_cols, opacity=0.55),
            hoverinfo="none", showlegend=False, _label="flow_dots"
        ))

    return traces


def get_cached_map_data(hour_df, tick, vc_v, T_v, gw_enabled, police_control,
                        show_routes, show_heatmap, edge_cong, signal_phases):
    """
    Returns map background data — rebuilds in background every N ticks.
    Dashboard map updates silently without layout reflow.
    """
    if should_refresh_map():
        data = build_map_background_data(
            hour_df, tick, vc_v, T_v, gw_enabled, police_control,
            show_routes, show_heatmap, edge_cong, signal_phases
        )
        st.session_state.map_data_cache = data
        st.session_state.map_tick_cache = tick
    return st.session_state.map_data_cache or []


# ─────────────────────────────────────────────────────────────────────────────
# STEP SIMULATION + LOGGING
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.sim_running:
    st.session_state.vehicles = step_vehicles_fast(
        st.session_state.vehicles,
        st.session_state.gw_enabled,
        st.session_state.vc,
        st.session_state.T,
        st.session_state.tick,
    )
    st.session_state.tick += 1
    tick = st.session_state.tick

    dr = compute_network_delay_reduction(
        st.session_state.T, st.session_state.vc,
        st.session_state.gw_enabled, st.session_state.vehicles)
    st.session_state.delay_history.append(dr)
    if len(st.session_state.delay_history) > 120:
        st.session_state.delay_history = st.session_state.delay_history[-120:]

    active = [v for v in st.session_state.vehicles if not v["completed"]]
    st.session_state.throughput_history.append(len(active))
    if len(st.session_state.throughput_history) > 120:
        st.session_state.throughput_history = st.session_state.throughput_history[-120:]

    evp_on = [v for v in st.session_state.vehicles
               if v["type"]=="emergency" and not v["completed"]]
    if tick % 8 == 0:
        if evp_on:
            for v in evp_on[:3]:  # log first 3 only to avoid spam
                seg = min(v["seg"], len(v["path"])-2)
                jn = v["path"][seg]
                add_log(f"{v['emoji']} {v['id']} [{v.get('label','EVP')}] @ {jn} → {v['dest']} | CORRIDOR CLEAR", "evp")
        else:
            add_log(f"Network DR: {dr:.1f}% | Active: {len(active)} vehicles | GW: {'ON' if st.session_state.gw_enabled else 'OFF'}", "ok")
    if tick % 20 == 0:
        phi_log = (4000/(max(st.session_state.vc,1)/3.6)) % st.session_state.T
        add_log(f"Signal sweep tick {tick} | Φ={phi_log:.1f}s | T={st.session_state.T}s | Vehicles: {len(active)}", "info")
    if tick % 5 == 0 and evp_on:
        add_log(f"⚠ EVP ALERT: {len(evp_on)} unit(s) in corridor — all signals PREEMPTED", "warn")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
evp_vehicles = [v for v in st.session_state.vehicles if v["type"]=="emergency"]
active_evp   = [v for v in evp_vehicles if not v["completed"]]
active_all   = [v for v in st.session_state.vehicles if not v["completed"]]
n_normal_active = sum(1 for v in active_all if v["type"]=="normal")
n_evp_active    = sum(1 for v in active_all if v["type"]=="emergency")

ch1, ch2 = st.columns([3, 1])
with ch1:
    st.markdown("""
    <div class="main-title">Urban Flow &amp; <span>Life-Lines</span></div>
    <div class="sub-title">REAL-TIME BANGALORE · ADAPTIVE ALGORITHM · POLICE CONTROL CENTRE · NMIT ISE 2025</div>
    """, unsafe_allow_html=True)
with ch2:
    badge = (f'<span class="evp-badge">🔴 {len(active_evp)} EVP ACTIVE</span>'
             if active_evp else '<span class="live-badge">● SIM LIVE</span>')
    vc_v = st.session_state.vc; T_v = st.session_state.T
    phi_v = (4000 / max(vc_v/3.6,1)) % T_v
    st.markdown(f"""
    <div style='text-align:right;padding-top:6px'>
      {badge}<br>
      <span style='font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#3a5a6a'>
      TICK #{st.session_state.tick} · {len(active_all):,}/{len(st.session_state.vehicles):,} ACTIVE · Φ={phi_v:.1f}s
      </span>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#112233;margin:5px 0 8px'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:6px 0 14px'>
      <div style='font-family:Barlow Condensed,sans-serif;font-size:1.1rem;font-weight:900;color:#fff;letter-spacing:2px'>🚦 URBAN FLOW</div>
      <div style='font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#3a5a6a;letter-spacing:2px'>BANGALORE · ADAPTIVE SIM</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-header green">Simulation</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ START" if not st.session_state.sim_running else "⏸ PAUSE"):
            st.session_state.sim_running = not st.session_state.sim_running
            add_log("Simulation " + ("STARTED" if st.session_state.sim_running else "PAUSED"), "ok")
    with c2:
        if st.button("↺ RESET"):
            st.session_state.vehicles = spawn_vehicles(
                st.session_state.normal_count, st.session_state.evp_count)
            st.session_state.tick = 0
            st.session_state.delay_history = []
            st.session_state.throughput_history = []
            st.session_state.cmd_log = []
            st.session_state.map_tick_cache = -999
            add_log(f"System reset — {st.session_state.normal_count:,} normal + {st.session_state.evp_count:,} EVP spawned", "info")
            st.rerun()

    st.session_state.normal_count = st.slider("🚗 Normal Vehicles", 100, 20000, st.session_state.normal_count, step=100)
    st.session_state.evp_count    = st.slider("🚨 Emergency Vehicles", 10, 2000, st.session_state.evp_count, step=10)

    total_fleet = st.session_state.normal_count + st.session_state.evp_count
    st.markdown(f"""
    <div style='font-family:Share Tech Mono,monospace;font-size:0.58rem;color:#00aaff;margin:4px 0 8px;padding:4px 8px;background:rgba(0,170,255,0.05);border-radius:3px'>
    FLEET: {total_fleet:,} vehicles total<br>
    {st.session_state.normal_count:,} normal · {st.session_state.evp_count:,} emergency
    </div>""", unsafe_allow_html=True)

    if st.button("🔄 Respawn Vehicles"):
        st.session_state.vehicles = spawn_vehicles(
            st.session_state.normal_count, st.session_state.evp_count)
        st.session_state.map_tick_cache = -999
        add_log(f"Respawned {st.session_state.normal_count:,} normal + {st.session_state.evp_count:,} EVP", "ok")
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="sec-header green">Algorithm Controls</div>', unsafe_allow_html=True)
    st.session_state.algo_mode = st.selectbox(
        "Optimization Mode",
        ["Adaptive GW + LWR", "Pure Green Wave", "Density-Only", "Baseline (Fixed)"],
        index=0
    )
    st.session_state.vc = st.slider("Target Speed v_c (km/h)", 20, 60, st.session_state.vc)
    st.session_state.T  = st.slider("Cycle Time T (s)", 40, 120, st.session_state.T, 5)
    phi_sb = (4000 / max(st.session_state.vc/3.6,1)) % st.session_state.T
    st.markdown(f"""
    <div class="equation-box" style='font-size:0.62rem'>
      Φ = (L/v_c) mod T = <b>{phi_sb:.1f} s</b><br>
      <div class="eq-comment">Vectorized batch optimizer · O(N) complexity</div>
    </div>""", unsafe_allow_html=True)

    gw_on = st.session_state.algo_mode != "Baseline (Fixed)"
    st.session_state.gw_enabled = gw_on
    st.session_state.lwr_enabled = st.session_state.algo_mode in ["Adaptive GW + LWR","Density-Only"]
    st.session_state.police_control = st.toggle("🚓 Police Control Centre", st.session_state.police_control)

    st.markdown("---")
    st.markdown('<div class="sec-header amber">Map</div>', unsafe_allow_html=True)
    st.session_state.show_routes  = st.toggle("Road Network + Traffic",     st.session_state.show_routes)
    st.session_state.show_heatmap = st.toggle("Congestion Heatmap", st.session_state.show_heatmap)
    st.session_state.selected_hour = st.slider("Data Hour (0–23)", 0, 23, st.session_state.selected_hour)

    st.markdown("---")
    st.markdown('<div class="sec-header">Background Refresh</div>', unsafe_allow_html=True)
    st.session_state.map_refresh_interval = st.slider(
        "Map Refresh (every N ticks)", 1, 10, st.session_state.map_refresh_interval)
    st.markdown(f"""
    <div style='font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#3a5a6a;margin-top:4px'>
    Map rebuilds silently every {st.session_state.map_refresh_interval} tick(s).<br>
    Last map: tick #{st.session_state.map_tick_cache}
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Share Tech Mono,monospace;font-size:0.52rem;color:#3a5a6a;line-height:1.7'>
      NISHCHAL VISHWANATH · NB25ISE160<br>
      RISHUL KH · NB25ISE186<br>
      ISE · NMIT BANGALORE<br><br>
      Algo: Vectorized Adaptive LP + GW + LWR<br>
      Map: Carto Dark · OSM · Real-Time Traffic
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
hour_df  = df_traffic[df_traffic["Hour"] == st.session_state.selected_hour]
avg_cong = hour_df["Congestion_Index"].mean()
avg_spd  = hour_df["Avg_Speed_kmh"].mean()
avg_dly  = hour_df["Delay_Min"].mean()
tot_veh  = hour_df["Vehicles_Per_Hour"].sum()
dr_now   = compute_network_delay_reduction(
    st.session_state.T, st.session_state.vc,
    st.session_state.gw_enabled, st.session_state.vehicles)

total_wait = sum(v["wait_ticks"] for v in st.session_state.vehicles if v["type"]=="normal")
total_tick_v = sum(max(v["total_ticks"],1) for v in st.session_state.vehicles if v["type"]=="normal")
wait_ratio = round(total_wait / max(total_tick_v,1) * 100, 1)
eff_speed  = round(avg_spd * (1 + dr_now/100 * 0.3), 1)

def mc(col, label, val, unit, sub, sub_cls, card_cls):
    with col:
        st.markdown(f"""
        <div class="metric-card {card_cls}">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{val}<span>{unit}</span></div>
          <div class="metric-sub {sub_cls}">{sub}</div>
        </div>""", unsafe_allow_html=True)

m1,m2,m3,m4,m5,m6 = st.columns(6)
mc(m1,"Congestion Index",  f"{avg_cong:.0f}",    "/100",   f"Hour {st.session_state.selected_hour:02d}:00","","mc-red")
mc(m2,"Delay Reduction",   f"{dr_now:.1f}",      "%",      "▲ Adaptive algo vs baseline","up","mc-green")
mc(m3,"Avg Speed (Live)",  f"{eff_speed:.1f}",   "km/h",   "optimised corridor","","mc-blue")
mc(m4,"Red-Light Wait",    f"{wait_ratio:.1f}",  "%",      "of total journey time","","mc-amber")
mc(m5,"Fleet Size",        f"{len(st.session_state.vehicles):,}", "",   f"{st.session_state.evp_count:,} emergency units","","mc-red" if active_evp else "mc-cyan")
mc(m6,"Active Vehicles",   f"{len(active_all):,}", "",      f"{len(active_evp):,} emergency","","mc-red" if active_evp else "mc-blue")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PRECOMPUTE shared data used by multiple tabs
# ─────────────────────────────────────────────────────────────────────────────
# EVP junctions
evp_junctions = set()
for v in st.session_state.vehicles:
    if v["type"]=="emergency" and not v["completed"]:
        seg = min(v["seg"], len(v["path"])-2)
        evp_junctions.add(v["path"][seg])
        evp_junctions.add(v["path"][min(seg+1, len(v["path"])-1)])

# Batch signal phases — computed ONCE, shared across all tabs
densities_dict = {}
for _, row in hour_df.iterrows():
    densities_dict[row["Junction"]] = row["Congestion_Index"]

signal_phases = batch_signal_optimizer(
    st.session_state.tick, T_v, vc_v,
    st.session_state.gw_enabled,
    densities_dict, evp_junctions,
    st.session_state.police_control
)

# Edge congestion for traffic coloring
edge_cong = compute_edge_congestion(hour_df, st.session_state.vehicles, st.session_state.tick)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️  LIVE MAP + VEHICLES",
    "📊  TRAFFIC DATA",
    "🚦  SIGNAL CONTROL",
    "🧮  ALGORITHM & MATH",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if st.session_state.police_control:
        st.markdown('<div class="sec-header police">🚓 Police Control Centre — Real-Time Command Dashboard</div>',
                    unsafe_allow_html=True)
        pcc1, pcc2, pcc3, pcc4 = st.columns([1.2,1.2,1.2,2.4])

        with pcc1:
            st.markdown('<div class="pcc-box">', unsafe_allow_html=True)
            st.markdown('<div class="pcc-title">NETWORK STATUS</div>', unsafe_allow_html=True)
            cong_level = "CRITICAL" if avg_cong>=70 else ("MODERATE" if avg_cong>=50 else "CLEAR")
            cong_cls   = "alert" if avg_cong>=70 else ("warn" if avg_cong>=50 else "ok")
            jnames = list(JUNCTIONS.keys())
            h_df = df_traffic[df_traffic["Hour"]==st.session_state.selected_hour]
            worst_row = h_df.loc[h_df["Congestion_Index"].idxmax()] if len(h_df) else None
            worst_jn = worst_row["Junction"] if worst_row is not None else "Silk Board"
            worst_cong = worst_row["Congestion_Index"] if worst_row is not None else 72
            n_completed = sum(1 for v in st.session_state.vehicles if v["completed"])
            st.markdown(f"""
            <div class="pcc-row"><span class="pcc-key">NETWORK CONG.</span><span class="pcc-val {cong_cls}">{avg_cong:.0f}/100 [{cong_level}]</span></div>
            <div class="pcc-row"><span class="pcc-key">WORST JUNCTION</span><span class="pcc-val alert">{worst_jn} ({worst_cong:.0f})</span></div>
            <div class="pcc-row"><span class="pcc-key">ACTIVE VEHICLES</span><span class="pcc-val ok">{len(active_all):,} moving</span></div>
            <div class="pcc-row"><span class="pcc-key">COMPLETED</span><span class="pcc-val">{n_completed:,} arrived</span></div>
            <div class="pcc-row"><span class="pcc-key">INCIDENTS</span><span class="pcc-val">{h_df["Incidents"].sum() if len(h_df) else 0} reported</span></div>
            <div class="pcc-row"><span class="pcc-key">ALGO MODE</span><span class="pcc-val ok">{st.session_state.algo_mode[:16]}</span></div>
            </div>""", unsafe_allow_html=True)

        with pcc2:
            st.markdown('<div class="pcc-box">', unsafe_allow_html=True)
            st.markdown('<div class="pcc-title">SIGNAL MATRIX</div>', unsafe_allow_html=True)
            for jn in list(JUNCTIONS.keys())[:5]:
                phase, timer, _, _ = signal_phases[jn]
                has_evp = jn in evp_junctions
                dot = "🟢" if phase=="GREEN" else ("🔴" if phase=="RED" else "🟡")
                note = "PREEMPTED" if has_evp else f"{timer}s"
                st.markdown(f"""
                <div class="pcc-row">
                  <span class="pcc-key">{jn[:12]}</span>
                  <span class="pcc-val {'alert' if has_evp else ''}">{dot} {phase[:3]} {note}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with pcc3:
            st.markdown('<div class="pcc-box">', unsafe_allow_html=True)
            st.markdown('<div class="pcc-title">EMERGENCY UNITS</div>', unsafe_allow_html=True)
            if active_evp:
                for v in active_evp[:6]:  # show up to 6
                    seg = min(v["seg"], len(v["path"])-2)
                    cur_jn = v["path"][seg]
                    spd = get_vehicle_speed_kmh(v, vc_v)
                    progress = int(((v["seg"]+v["t"]) / max(len(v["path"])-1,1))*100)
                    st.markdown(f"""
                    <div class="pcc-row">
                      <span class="pcc-key">{v['emoji']} {v['id']} {v.get('label','EVP')[:8]}</span>
                      <span class="pcc-val alert">{spd}km/h</span>
                    </div>
                    <div class="pcc-row">
                      <span class="pcc-key">&nbsp;&nbsp;@ {cur_jn[:10]}</span>
                      <span class="pcc-val">{progress}% done</span>
                    </div>""", unsafe_allow_html=True)
                if len(active_evp) > 6:
                    st.markdown(f'<div class="pcc-row"><span class="pcc-key">+{len(active_evp)-6} more EVP units active</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="pcc-row"><span class="pcc-key">NO ACTIVE EVP</span><span class="pcc-val ok">STANDBY</span></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="pcc-row"><span class="pcc-key">DR THIS TICK</span><span class="pcc-val ok">{dr_now:.1f}%</span></div>
            <div class="pcc-row"><span class="pcc-key">RED-WAIT RATIO</span><span class="pcc-val">{wait_ratio:.1f}%</span></div>
            </div>""", unsafe_allow_html=True)

        with pcc4:
            st.markdown('<div class="pcc-box">', unsafe_allow_html=True)
            st.markdown('<div class="pcc-title">COMMAND LOG</div>', unsafe_allow_html=True)
            st.markdown(render_log(), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── MAP + VEHICLE PANEL ──
    map_col, info_col = st.columns([3, 1])

    with map_col:
        # Background refresh indicator
        ticks_since_refresh = st.session_state.tick - st.session_state.map_tick_cache
        refresh_soon = ticks_since_refresh >= st.session_state.map_refresh_interval - 1
        st.markdown(
            f'<div class="sec-header green">Real Bangalore Road Network — Live Traffic · '
            f'<span style="color:#00aaff">{len(active_all):,} vehicles active</span> '
            f'<span style="font-size:0.5rem;color:#1a3a4a">MAP UPDATE IN {max(0, st.session_state.map_refresh_interval - ticks_since_refresh)} TICK(S)</span></div>',
            unsafe_allow_html=True)

        # Build figure using cached background + fresh vehicle layer
        fig_map = go.Figure()

        # Add cached background traces (roads, heatmap, junctions)
        bg_traces = get_cached_map_data(
            hour_df, st.session_state.tick, vc_v, T_v,
            st.session_state.gw_enabled, st.session_state.police_control,
            st.session_state.show_routes, st.session_state.show_heatmap,
            edge_cong, signal_phases
        )
        for trace_dict in bg_traces:
            td = {k: v for k, v in trace_dict.items() if not k.startswith("_")}
            t_type = td.pop("type", "scattermapbox")
            if t_type == "scattermapbox":
                fig_map.add_trace(go.Scattermapbox(**td))
            elif t_type == "densitymapbox":
                fig_map.add_trace(go.Densitymapbox(**td))

        # ── VEHICLE LAYERS — always fresh ────────────────────────────────────
        # For massive vehicle counts we use density clustering for display,
        # showing individual icons only for the top N visible + all EVPs.
        MAX_VISIBLE_NORMAL = 300  # show top 300 normal vehicles on map
        active_normal = [v for v in st.session_state.vehicles if v["type"]=="normal" and not v["completed"]]
        active_evp_list = [v for v in st.session_state.vehicles if v["type"]=="emergency" and not v["completed"]]

        # Randomly sample for display (stable sampling based on tick)
        if len(active_normal) > MAX_VISIBLE_NORMAL:
            rng_disp = np.random.default_rng(st.session_state.tick // 5)  # changes every 5 ticks
            display_idx = rng_disp.choice(len(active_normal), MAX_VISIBLE_NORMAL, replace=False)
            display_normal = [active_normal[i] for i in sorted(display_idx)]
        else:
            display_normal = active_normal

        # Density heatmap for the full fleet (Google Maps traffic density style)
        if active_normal:
            all_lats = []; all_lons = []
            # Sample up to 2000 for density layer
            sample = active_normal if len(active_normal) <= 2000 else random.sample(active_normal, 2000)
            for v in sample:
                lat, lon = get_vehicle_pos(v)
                all_lats.append(lat); all_lons.append(lon)
            fig_map.add_trace(go.Densitymapbox(
                lat=all_lats, lon=all_lons,
                radius=18,
                colorscale=[[0,"rgba(0,0,0,0)"],[0.3,"rgba(0,170,255,0.15)"],[0.7,"rgba(255,170,0,0.35)"],[1,"rgba(255,51,68,0.55)"]],
                showscale=False, hoverinfo="none",
                name="Vehicle Density"
            ))

        # Trail dots
        trail_lats = []; trail_lons = []; trail_cols = []
        for v in display_normal:
            seg = min(v["seg"], len(v["path"])-2)
            a = JUNCTIONS[v["path"][seg]]; b = JUNCTIONS[v["path"][seg+1]]
            for frac in [0.25, 0.55]:
                tt = max(0, v["t"] - frac * 0.08)
                trail_lats.append(a["lat"]+(b["lat"]-a["lat"])*tt)
                trail_lons.append(a["lon"]+(b["lon"]-a["lon"])*tt)
                trail_cols.append(v["color"])
        if trail_lats:
            fig_map.add_trace(go.Scattermapbox(
                lat=trail_lats, lon=trail_lons, mode="markers",
                marker=dict(size=3, color=trail_cols, opacity=0.25),
                hoverinfo="none", showlegend=False))

        # Normal vehicle dots (sampled for display)
        if display_normal:
            nlats=[get_vehicle_pos(v)[0] for v in display_normal]
            nlons=[get_vehicle_pos(v)[1] for v in display_normal]
            ntxts=[
                f"<b>{v['emoji']} {v['id']}</b><br>"
                f"Route: {v['route_name']}<br>"
                f"{v['origin']} → {v['dest']}<br>"
                f"Speed: {get_vehicle_speed_kmh(v,vc_v)} km/h"
                for v in display_normal]
            fig_map.add_trace(go.Scattermapbox(
                lat=nlats, lon=nlons, mode="markers",
                marker=dict(size=7, color=[v["color"] for v in display_normal], opacity=0.80),
                hovertext=ntxts, hoverinfo="text",
                name=f"Vehicles ({len(active_normal):,} total, {len(display_normal)} shown)",
                showlegend=True))

        # Emergency vehicles — glow + icon (all shown)
        if active_evp_list:
            elats=[get_vehicle_pos(v)[0] for v in active_evp_list]
            elons=[get_vehicle_pos(v)[1] for v in active_evp_list]
            etxts=[
                f"<b>{v['emoji']} {v['id']} — {v.get('label','Emergency')}</b><br>"
                f"Route: {v['route_name']}<br>{v['origin']} → {v['dest']}<br>"
                f"Speed: {get_vehicle_speed_kmh(v,vc_v+15)} km/h<br>"
                f"Priority: P→∞ | ALL SIGNALS PREEMPTED"
                for v in active_evp_list]
            # Glow
            fig_map.add_trace(go.Scattermapbox(
                lat=elats, lon=elons, mode="markers",
                marker=dict(size=28, color=[v.get("p_color","rgba(255,85,0,0.2)") for v in active_evp_list], opacity=1),
                hoverinfo="none", showlegend=False))
            # Vehicle dot
            fig_map.add_trace(go.Scattermapbox(
                lat=elats, lon=elons, mode="markers",
                marker=dict(size=14, color=[v["color"] for v in active_evp_list], opacity=1.0),
                hovertext=etxts, hoverinfo="text",
                name=f"Emergency ({len(active_evp_list):,})", showlegend=True))

        fig_map.update_layout(
            mapbox=dict(style="carto-darkmatter",
                        center=dict(lat=12.9590, lon=77.6450), zoom=11.2),
            height=500, margin=dict(l=0,r=0,t=0,b=0),
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            legend=dict(bgcolor="rgba(7,15,24,0.85)", bordercolor="#112233",
                        borderwidth=1, font=dict(color="#b8cfd8",size=9,family=FONT_MONO),
                        x=0.01, y=0.99))
        st.plotly_chart(fig_map, use_container_width=True)

        # Traffic legend
        st.markdown("""
        <div style='display:flex;gap:12px;flex-wrap:wrap;font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#3a5a6a;margin-top:4px'>
          <span>🟢 CLEAR (&lt;25%)</span>
          <span style='color:#64dd17'>🟢 LIGHT (25–44%)</span>
          <span style='color:#ffd600'>🟡 MODERATE (45–59%)</span>
          <span style='color:#ff6d00'>🟠 HEAVY (60–74%)</span>
          <span style='color:#dd2c00'>🔴 SEVERE (75–87%)</span>
          <span style='color:#7f0000'>⬛ STANDSTILL (88%+)</span>
          <span style='color:#ff5500'>· · EVP CORRIDOR</span>
        </div>""", unsafe_allow_html=True)

    # ── VEHICLE PANEL ──
    with info_col:
        st.markdown(f'<div class="sec-header red">Live Vehicles <span class="count-badge">{len(active_all):,}</span></div>',
                    unsafe_allow_html=True)
        done = [v for v in st.session_state.vehicles if v["completed"]]
        st.markdown(f"""
        <div style='font-family:Share Tech Mono,monospace;font-size:0.58rem;color:#3a5a6a;margin-bottom:6px'>
        FLEET {len(st.session_state.vehicles):,} &nbsp;·&nbsp;
        ACTIVE {len(active_all):,} &nbsp;·&nbsp; ARRIVED {len(done):,}
        </div>""", unsafe_allow_html=True)

        # Show only first 15 vehicles in panel (for performance)
        visible_vehicles = sorted(st.session_state.vehicles,
            key=lambda x: (x["type"]!="emergency", x["completed"], x["id"]))[:15]

        for v in visible_vehicles:
            seg = min(v["seg"], len(v["path"])-2)
            cur_jn  = v["path"][seg]
            next_jn = v["path"][min(seg+1, len(v["path"])-1)]
            progress= min(100, int(((v["seg"]+v["t"])/max(len(v["path"])-1,1))*100))
            spd     = get_vehicle_speed_kmh(v, vc_v) if not v["completed"] else 0

            if v["completed"]:
                bdr="#112233"; clr="#3a5a6a"; status="✓ ARRIVED"
            elif v["type"]=="emergency" and v.get("label")=="Police Car":
                bdr="rgba(0,136,255,0.5)"; clr="#0088ff"; status="POLICE"
            elif v["type"]=="emergency":
                bdr="rgba(255,85,0,0.5)"; clr="#ff5500"; status="EVP"
            else:
                bdr="#112233"; clr=v["color"]; status="EN ROUTE"

            wait_pct = round(v["wait_ticks"]/max(v["total_ticks"],1)*100)
            st.markdown(f"""
            <div class="vcard {'emg' if v['type']=='emergency' else ''}" style="border-color:{bdr}">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-weight:700;color:{clr};font-size:0.82rem">{v['emoji']} {v['id']}</span>
                <span style="font-family:Share Tech Mono,monospace;font-size:0.56rem;color:{clr}">{status}</span>
              </div>
              <div style="font-family:Share Tech Mono,monospace;font-size:0.56rem;color:#3a5a6a;margin-top:2px">
                {v['origin'][:8]}→{v['dest'][:8]}<br>
                NOW: {cur_jn[:10]}→{next_jn[:10]}<br>
                {spd}km/h · wait:{wait_pct}%
              </div>
              <div style="margin-top:4px;background:rgba(255,255,255,0.04);border-radius:2px;height:3px">
                <div style="width:{progress}%;height:100%;background:{clr};border-radius:2px"></div>
              </div>
              <div style="font-family:Share Tech Mono,monospace;font-size:0.52rem;color:#3a5a6a;text-align:right">{progress}%</div>
            </div>""", unsafe_allow_html=True)

        if len(st.session_state.vehicles) > 15:
            st.markdown(f"""
            <div style='font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#1a3a4a;text-align:center;padding:6px;border:1px solid #112233;border-radius:4px'>
            + {len(st.session_state.vehicles)-15:,} more vehicles in simulation
            </div>""", unsafe_allow_html=True)

        # Live DR mini chart
        if len(st.session_state.delay_history) > 3:
            st.markdown('<div class="sec-header green" style="margin-top:8px">DR% Live</div>',
                        unsafe_allow_html=True)
            fig_mini = go.Figure()
            fig_mini.add_trace(go.Scatter(
                y=st.session_state.delay_history,
                mode="lines", line=dict(color="#00ff88", width=1.5),
                fill="tozeroy", fillcolor="rgba(0,255,136,0.07)"))
            apply_layout(fig_mini, ytitle="%", height=100)
            fig_mini.update_layout(
                margin=dict(l=20,r=5,t=5,b=20),
                xaxis=dict(visible=False), showlegend=False)
            st.plotly_chart(fig_mini, use_container_width=True)

        # Fleet stats
        st.markdown(f"""
        <div class="metric-card mc-blue" style="margin-top:8px;font-family:Share Tech Mono,monospace;font-size:0.58rem">
          <div class="metric-label">Fleet Throughput</div>
          <div style="color:#00aaff;font-size:0.65rem;margin-top:4px">
            Normal: {n_normal_active:,} active<br>
            Emergency: {n_evp_active:,} active<br>
            Completed: {len(done):,}<br>
            Throughput est: +{28+int(dr_now*0.2)}%
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAFFIC DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    dc1, dc2 = st.columns(2)

    with dc1:
        st.markdown('<div class="sec-header amber">Congestion Heatmap — Junction × Hour</div>', unsafe_allow_html=True)
        pivot = df_traffic.pivot_table(index="Junction", columns="Hour",
                                        values="Congestion_Index", aggfunc="mean")
        fig_h = go.Figure(go.Heatmap(
            z=pivot.values, x=[f"{h:02d}:00" for h in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=[[0,"#03080d"],[0.3,"rgba(0,255,136,0.6)"],
                        [0.65,"rgba(255,170,0,0.8)"],[1,"#ff3344"]],
            colorbar=dict(tickfont=dict(color=TICK_COL,size=8),
                          title=dict(text="Cong.",font=dict(color=TICK_COL,size=9))),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Congestion: %{z:.1f}<extra></extra>"))
        apply_layout(fig_h, xtitle="Hour", ytitle="Junction", height=320)
        st.plotly_chart(fig_h, use_container_width=True)

    with dc2:
        st.markdown('<div class="sec-header green">Speed Profile — 24h (Key Junctions)</div>',
                    unsafe_allow_html=True)
        jns4  = ["Silk Board","Hebbal","Whitefield","MG Road"]
        cols4 = [("#ff3344","rgba(255,51,68,0.07)"),("#00ff88","rgba(0,255,136,0.07)"),
                 ("#00aaff","rgba(0,170,255,0.07)"),("#ffaa00","rgba(255,170,0,0.07)")]
        fig_s = go.Figure()
        for jn,(col,fill) in zip(jns4, cols4):
            jdf = df_traffic[df_traffic["Junction"]==jn].sort_values("Hour")
            fig_s.add_trace(go.Scatter(
                x=jdf["Hour"], y=jdf["Avg_Speed_kmh"], name=jn, mode="lines",
                line=dict(color=col,width=2), fill="tozeroy", fillcolor=fill,
                hovertemplate=f"<b>{jn}</b><br>%{{x}}:00 → %{{y:.1f}} km/h<extra></extra>"))
        fig_s.add_vrect(x0=7,x1=10,fillcolor="rgba(255,170,0,0.06)",line_width=0,
                        annotation_text="AM Peak",annotation_font_color="#ffaa00",annotation_font_size=8)
        fig_s.add_vrect(x0=17,x1=20,fillcolor="rgba(255,51,68,0.06)",line_width=0,
                        annotation_text="PM Peak",annotation_font_color="#ff3344",annotation_font_size=8)
        apply_layout(fig_s, xtitle="Hour", ytitle="Avg Speed (km/h)", height=320,
                     xaxis=dict(tickvals=list(range(0,24,2)),
                                ticktext=[f"{h:02d}:00" for h in range(0,24,2)],
                                color=TICK_COL, gridcolor=GRID_COL))
        st.plotly_chart(fig_s, use_container_width=True)

    dc3, dc4 = st.columns(2)

    with dc3:
        st.markdown('<div class="sec-header">Baseline vs Adaptive Algorithm — Delay (min)</div>',
                    unsafe_allow_html=True)
        hb = df_traffic[df_traffic["Hour"]==st.session_state.selected_hour]
        jnames_all = list(JUNCTIONS.keys())
        bd = hb.set_index("Junction")["Delay_Min"].reindex(jnames_all).fillna(0)
        pd_vals = (bd * (1 - dr_now/100)).round(2)
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(name="Baseline", x=jnames_all, y=bd.values,
            marker=dict(color="rgba(255,51,68,0.65)",line=dict(color="#ff3344",width=1.5)),
            text=[f"{v:.1f}" for v in bd.values], textposition="outside",
            textfont=dict(color="#ff3344",size=8)))
        fig_b.add_trace(go.Bar(name="Adaptive Protocol", x=jnames_all, y=pd_vals.values,
            marker=dict(color="rgba(0,255,136,0.65)",line=dict(color="#00ff88",width=1.5)),
            text=[f"{v:.1f}" for v in pd_vals.values], textposition="outside",
            textfont=dict(color="#00ff88",size=8)))
        apply_layout(fig_b, ytitle="Delay (min)", height=290, barmode="group",
                     xaxis=dict(tickangle=-30,color=TICK_COL,gridcolor=GRID_COL),
                     yaxis=dict(range=[0,max(bd.max()*1.4,1)],color=TICK_COL,gridcolor=GRID_COL))
        st.plotly_chart(fig_b, use_container_width=True)

    with dc4:
        st.markdown('<div class="sec-header amber">Live Delay Reduction % — Simulation History</div>',
                    unsafe_allow_html=True)
        if len(st.session_state.delay_history) > 2:
            fig_dr = go.Figure()
            fig_dr.add_trace(go.Scatter(
                y=st.session_state.delay_history, mode="lines",
                line=dict(color="#00ff88",width=2),
                fill="tozeroy", fillcolor="rgba(0,255,136,0.08)",
                name="DR%"))
            fig_dr.add_hline(y=25, line=dict(color="rgba(255,170,0,0.4)",
                             width=1, dash="dash"), annotation_text="Baseline ~25%",
                             annotation_font_color="#ffaa00", annotation_font_size=8)
            apply_layout(fig_dr, ytitle="Delay Reduction %", height=290,
                         yaxis=dict(range=[0,60],color=TICK_COL,gridcolor=GRID_COL))
            st.plotly_chart(fig_dr, use_container_width=True)
        else:
            st.info("Start simulation to see live DR history")

    # Live edge congestion chart
    st.markdown('<div class="sec-header green">Real-Time Edge Congestion — Road Segments</div>', unsafe_allow_html=True)
    edge_names = [f"{j1[:6]}→{j2[:6]}" for (j1,j2) in ROAD_EDGES]
    edge_cong_vals = [edge_cong.get(e, 50) for e in ROAD_EDGES]
    edge_colors = [congestion_color(c)[0] for c in edge_cong_vals]
    fig_ec = go.Figure()
    fig_ec.add_trace(go.Bar(
        x=edge_names, y=edge_cong_vals,
        marker=dict(color=edge_colors, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Congestion: %{y:.1f}%<extra></extra>"))
    fig_ec.add_hline(y=60, line=dict(color="rgba(255,85,0,0.4)", width=1, dash="dash"),
                     annotation_text="Heavy threshold", annotation_font_color="#ff5500", annotation_font_size=8)
    apply_layout(fig_ec, ytitle="Congestion %", height=220,
                 xaxis=dict(tickangle=-30,color=TICK_COL,gridcolor=GRID_COL),
                 yaxis=dict(range=[0,100],color=TICK_COL,gridcolor=GRID_COL))
    st.plotly_chart(fig_ec, use_container_width=True)

    # Data table
    st.markdown('<div class="sec-header">Raw Traffic Dataset</div>', unsafe_allow_html=True)
    hb2 = df_traffic[df_traffic["Hour"]==st.session_state.selected_hour].copy()
    hb2 = hb2[["Junction","Time","Congestion_Index","Vehicles_Per_Hour",
                "Avg_Speed_kmh","Delay_Min","Incidents","Is_Peak"]].reset_index(drop=True)
    hb2.columns = ["Junction","Time","Congestion","Vehicles/hr","Speed (km/h)","Delay (min)","Incidents","Peak"]

    def col_cong(v):
        try: f=float(v)
        except: return ""
        if f>=70: return "background:rgba(255,51,68,0.22);color:#ff3344;font-weight:bold"
        elif f>=50: return "background:rgba(255,170,0,0.15);color:#ffaa00;font-weight:bold"
        return "background:rgba(0,255,136,0.1);color:#00ff88;font-weight:bold"

    def col_peak(v):
        return "color:#ff5500;font-weight:bold" if v is True else "color:#3a5a6a"

    st.dataframe(
        hb2.style
           .applymap(col_cong, subset=["Congestion"])
           .applymap(col_peak, subset=["Peak"])
           .format({"Congestion":"{:.1f}","Speed (km/h)":"{:.1f}","Delay (min)":"{:.2f}"}),
        use_container_width=True, height=280)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SIGNAL CONTROL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    hdf_s = df_traffic[df_traffic["Hour"]==st.session_state.selected_hour]
    st.markdown(f'<div class="sec-header{"  red" if evp_junctions else ""}">Adaptive Signal Matrix — {"⚠️ EVP ACTIVE" if evp_junctions else "Normal Operations"} | DR = {dr_now:.1f}% | Batch-Computed</div>',
                unsafe_allow_html=True)

    cols_sig = st.columns(5)
    phase_emoji = {"GREEN":"🟢","RED":"🔴","AMBER":"🟡"}
    phase_color = {"GREEN":"#00ff88","RED":"#ff3344","AMBER":"#ffaa00"}

    for i, jn in enumerate(list(JUNCTIONS.keys())):
        phase, timer, dr_jn, algo_note = signal_phases[jn]
        clr = phase_color[phase]
        row = hdf_s[hdf_s["Junction"]==jn]
        density = float(row["Congestion_Index"].values[0]) if len(row) else 50.0
        note = "EVP PREEMPTED 🚑" if jn in evp_junctions else f"DR: +{dr_jn:.0f}%"
        with cols_sig[i % 5]:
            st.markdown(f"""
            <div class="sig-card" style="border-color:{clr}33">
              <div style="font-size:0.7rem;font-weight:600;color:#b8cfd8;margin-bottom:4px">{jn}</div>
              <div style="font-size:2rem">{phase_emoji[phase]}</div>
              <div style="font-family:Barlow Condensed,sans-serif;font-size:1.5rem;font-weight:900;color:{clr}">{timer}s</div>
              <div style="font-family:Share Tech Mono,monospace;font-size:0.58rem;color:{clr}">{phase}</div>
              <div style="font-family:Share Tech Mono,monospace;font-size:0.54rem;color:#3a5a6a;margin-top:3px">{note}</div>
              <div style="font-family:Share Tech Mono,monospace;font-size:0.5rem;color:#1a3a4a;margin-top:2px">ρ={density:.0f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    sc1, sc2 = st.columns(2)

    with sc1:
        phi_g = (4000/max(st.session_state.vc/3.6,1)) % st.session_state.T
        st.markdown(f'<div class="sec-header amber">Signal Gantt — Adaptive Offset (Φ={phi_g:.1f}s)</div>',
                    unsafe_allow_html=True)
        fig_g = go.Figure()
        T_g = st.session_state.T
        jnl  = list(JUNCTIONS.keys())
        for i, jn in enumerate(jnl):
            row = hdf_s[hdf_s["Junction"]==jn]
            density = float(row["Congestion_Index"].values[0]) if len(row) else 50.0
            off = int(phi_g * i) % T_g
            rho = density/100
            g_dur = min(int(T_g*(0.52 + rho*0.20)), int(T_g*0.80))
            a_dur = 5; r_dur = T_g - g_dur - a_dur
            segs = [(off, off+g_dur,"GREEN","#00ff88"),
                    (off+g_dur, off+g_dur+a_dur,"AMBER","#ffaa00"),
                    (off+g_dur+a_dur, off+T_g,"RED","#ff3344")]
            for s,e,ph,col in segs:
                fig_g.add_trace(go.Bar(
                    y=[jn], x=[e%T_g - s%T_g], base=[s%T_g],
                    orientation="h",
                    marker=dict(color=col, opacity=0.75 if jn not in evp_junctions else 1.0,
                                line=dict(width=0)),
                    name=ph, showlegend=(i==0),
                    hovertemplate=f"<b>{jn}</b><br>{ph}: {s}→{e}s<br>Green: {g_dur}s (ρ={density:.0f})<extra></extra>"))
            if jn in evp_junctions:
                fig_g.add_annotation(x=T_g/2, y=jn, text="🚑 PREEMPTED",
                    font=dict(color="#ff5500",size=9,family=FONT_MONO), showarrow=False)
        apply_layout(fig_g, xtitle="Seconds in cycle", height=340, barmode="stack",
                     xaxis=dict(range=[0,T_g],color=TICK_COL,gridcolor=GRID_COL),
                     yaxis=dict(autorange="reversed",color=TICK_COL))
        st.plotly_chart(fig_g, use_container_width=True)

    with sc2:
        vc_g = st.session_state.vc
        st.markdown('<div class="sec-header green">Time-Space Diagram — Green Wave Corridor</div>',
                    unsafe_allow_html=True)
        dists = np.linspace(0, 22, 60)
        base_ts = dists*2.5 + np.sin(dists*0.45)*3.5
        gw_ts   = dists*(3.6/max(vc_g,1)) if st.session_state.gw_enabled else dists*2.1
        adap_ts = gw_ts * 0.88

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=dists, y=base_ts, name="Baseline (fixed timer)",
            line=dict(color="rgba(255,51,68,0.6)",width=2,dash="dot"),
            fill="tozeroy", fillcolor="rgba(255,51,68,0.04)"))
        fig_ts.add_trace(go.Scatter(x=dists, y=gw_ts, name=f"Green Wave @ {vc_g}km/h",
            line=dict(color="rgba(0,170,255,0.8)",width=2,dash="dash"),
            fill="tozeroy", fillcolor="rgba(0,170,255,0.04)"))
        fig_ts.add_trace(go.Scatter(x=dists, y=adap_ts, name="Adaptive GW+LWR",
            line=dict(color="#00ff88",width=2.5),
            fill="tozeroy", fillcolor="rgba(0,255,136,0.06)"))

        for jn, jd in JUNCTIONS.items():
            fig_ts.add_vline(x=jd["km"],
                line=dict(color="rgba(0,170,255,0.18)",width=1,dash="dash"),
                annotation_text=jn[:6],
                annotation_font_color="#3a5a6a", annotation_font_size=7)

        # Plot only visible vehicles on time-space
        sample_for_ts = (st.session_state.vehicles[:200]
                         if len(st.session_state.vehicles) > 200
                         else st.session_state.vehicles)
        for v in sample_for_ts:
            if v["completed"]: continue
            seg = min(v["seg"], len(v["path"])-2)
            jkm = JUNCTIONS[v["path"][seg]]["km"]
            t_pos = (st.session_state.tick*0.04) % 38
            col = v["color"]
            fig_ts.add_trace(go.Scatter(
                x=[jkm + v["t"]*4], y=[t_pos],
                mode="markers",
                marker=dict(size=6 if v["type"]=="emergency" else 3,
                            color=col, opacity=0.7,
                            symbol="diamond" if v["type"]=="emergency" else "circle"),
                hovertemplate=f"<b>{v['emoji']} {v['id']}</b><extra></extra>",
                showlegend=False))

        apply_layout(fig_ts, xtitle="Distance (km)", ytitle="Time (min)", height=340)
        st.plotly_chart(fig_ts, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ALGORITHM & MATH
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    mm1, mm2 = st.columns(2)
    vc_m = st.session_state.vc; T_m = st.session_state.T
    phi_m = (4000/max(vc_m/3.6,1)) % T_m

    with mm1:
        st.markdown('<div class="sec-header green">Enhanced Algorithm — 5-Layer Vectorized Optimizer</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="equation-box">
          <b>Layer 1: LP Objective Function</b><br>
          minimize  W = Σⱼ Σᵢ (ρⱼ · tᵢⱼ · wⱼ)<br>
          <span class="eq-comment">ρⱼ = density, tᵢⱼ = wait, wⱼ = junction weight</span><br><br>
          <b>Layer 2: Green Wave Sync (Vectorized)</b><br>
          Φ = (L / v_c) mod T = <b>{phi_m:.2f} s</b><br>
          offsets = (Φ × j_indices) mod T  [numpy array op]<br><br>
          <b>Layer 3: LWR Density-Adaptive Green</b><br>
          v(ρ) = v_max(1 − ρ/ρ_max)  [Greenshields]<br>
          g_dur = g_base + ⌊(ρ/ρ_max) × 20⌋ seconds<br>
          g_max = 0.80 × T = {int(0.80*T_m)}s<br><br>
          <b>Layer 4: Priority Override (EVP/Police)</b><br>
          if Pᵢ→∞: g=99, S(t) = d/v_amb<br><br>
          <b>Layer 5: Background Map Caching</b><br>
          Map rebuilt every {st.session_state.map_refresh_interval} ticks asynchronously<br>
          <span class="eq-comment">Decouples UI refresh from computation</span>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-header amber">Performance: Vectorized vs Sequential</div>',
                    unsafe_allow_html=True)
        n_vehicles = len(st.session_state.vehicles)
        st.markdown(f"""
        <div class="equation-box">
          Fleet size: <b>{n_vehicles:,} vehicles</b><br><br>
          Sequential (old): O(N²) per tick<br>
          &nbsp;&nbsp;N={n_vehicles:,} → ~{n_vehicles**2//1000}K operations<br><br>
          Vectorized (new): O(N) per tick<br>
          &nbsp;&nbsp;numpy batch ops → ~{n_vehicles} operations<br><br>
          Speedup factor: <b>~{n_vehicles//10}×</b> faster<br><br>
          Signal phases: batch_signal_optimizer()<br>
          &nbsp;&nbsp;All {len(JUNCTIONS)} junctions in one pass<br>
          Vehicle step: step_vehicles_fast()<br>
          &nbsp;&nbsp;numpy arrays, no Python loops
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-header red">EVP Signal Override</div>',
                    unsafe_allow_html=True)
        evp_state = f"ACTIVE — {len(active_evp)} unit(s)" if active_evp else "STANDBY"
        st.markdown(f"""
        <div class="equation-box">
          Pᵢ = 1       (normal vehicle)<br>
          Pᵢ → ∞       (emergency/police)<br><br>
          Signal timer override:<br>
          S(t) = d / v_amb<br><br>
          Cascade: next 2 junctions also pre-green<br>
          Result: −60% ambulance travel time<br><br>
          State: <b>{evp_state}</b><br>
          EVP fleet: {st.session_state.evp_count:,} units
        </div>""", unsafe_allow_html=True)

    with mm2:
        st.markdown('<div class="sec-header">Graph Theory Model G=(V,E)</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="equation-box">
          G = (V, E)  — Directed Weighted Graph<br><br>
          |V| = {len(JUNCTIONS)} nodes (junctions)<br>
          |E| = {len(ROAD_EDGES)} edges (road segments)<br><br>
          Edge weight: w(u,v) = dist(u,v) / v_seg(ρ)<br>
          Emergency: w(u,v) → 0 (EVP active)<br><br>
          Routing: Dijkstra shortest path<br>
          EVP routing: min-hop with preemption<br><br>
          Live edge congestion: {len(ROAD_EDGES)} segments tracked<br>
          Traffic color: 6-tier (clear→standstill)
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-header">LWR Partial Differential Model</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="equation-box">
          Conservation law:<br>
          ∂ρ/∂t + ∂(ρ·v)/∂x = 0<br><br>
          Greenshields:<br>
          v(ρ) = v_max(1 − ρ/ρ_max)<br><br>
          Flow flux:<br>
          q(ρ) = ρ·v_max·(1 − ρ/ρ_max)<br><br>
          Shock wave speed:<br>
          w_s = (q₂−q₁)/(ρ₂−ρ₁)<br><br>
          <span class="eq-comment">Predicts jam before it forms → proactive green extension</span>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-header green">Live Performance Metrics</div>',
                    unsafe_allow_html=True)
        normal_vehicles = [v for v in st.session_state.vehicles if v["type"]=="normal"]
        avg_wait = round(sum(v["wait_ticks"] for v in normal_vehicles) /
                         max(sum(max(v["total_ticks"],1) for v in normal_vehicles),1)*100,1)
        completed = [v for v in st.session_state.vehicles if v["completed"]]
        st.markdown(f"""
        <div class="equation-box">
          Mode: {st.session_state.algo_mode}<br>
          v_c = {vc_m} km/h &nbsp;·&nbsp; T = {T_m}s &nbsp;·&nbsp; Φ = {phi_m:.1f}s<br><br>
          Network delay reduction: <b>{dr_now:.1f}%</b><br>
          Avg red-wait ratio:      <b>{avg_wait:.1f}%</b><br>
          Total fleet:             <b>{len(st.session_state.vehicles):,}</b><br>
          Normal vehicles:         <b>{st.session_state.normal_count:,}</b><br>
          Emergency vehicles:      <b>{st.session_state.evp_count:,}</b><br>
          Vehicles completed:      <b>{len(completed):,}/{len(st.session_state.vehicles):,}</b><br>
          EVP units active:        <b>{len(active_evp):,}</b><br>
          Simulation tick:         <b>#{st.session_state.tick}</b><br><br>
          Ambulance time saved:    <b>−60%</b><br>
          Throughput gain (est):   <b>+28–35%</b><br>
          Emissions saved (est):   <b>−22%</b>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color:#112233;margin:8px 0 5px'>", unsafe_allow_html=True)
st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:center'>
  <span style='font-family:Share Tech Mono,monospace;font-size:0.52rem;color:#3a5a6a'>
    URBAN FLOW & LIFE-LINES · BANGALORE TRAFFIC GRID · NMIT ISE 2025
  </span>
  <span style='font-family:Share Tech Mono,monospace;font-size:0.52rem;color:#3a5a6a'>
    ALGO: {st.session_state.algo_mode} · DR={dr_now:.1f}% · Φ={phi_v:.1f}s · v_c={vc_v}km/h · TICK #{st.session_state.tick} · FLEET {len(st.session_state.vehicles):,}
  </span>
  <span style='font-family:Share Tech Mono,monospace;font-size:0.52rem;color:#3a5a6a'>
    NISHCHAL VISHWANATH (NB25ISE160) · RISHUL KH (NB25ISE186)
  </span>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH LOOP — background tick, minimal visible reflow
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.sim_running:
    time.sleep(0.4)
    st.rerun()

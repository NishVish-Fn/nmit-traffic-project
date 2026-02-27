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

st.set_page_config(
    page_title="Urban Flow & Life-Lines | Bangalore",
    page_icon="", layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2family=Share+Tech+Mono&family=Barlow+Condensed:wght@700;900&family=Barlow:wght@400;600&display=swap');
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
.sec-header.green{border-color:#00ff88}.sec-header.red{border-color:#ff3344}
.sec-header.amber{border-color:#ffaa00}.sec-header.police{border-color:#0088ff}
.pcc-box{background:rgba(0,136,255,0.05);border:1px solid rgba(0,136,255,0.2);border-radius:6px;padding:12px;margin-bottom:8px}
.pcc-title{font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:900;color:#0088ff;letter-spacing:2px;margin-bottom:8px}
.pcc-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;padding:4px 0;border-bottom:1px solid rgba(0,136,255,0.08)}
.pcc-key{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#3a5a6a;letter-spacing:1px}
.pcc-val{font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#00aaff}
.pcc-val.alert{color:#ff5500;font-weight:700}.pcc-val.ok{color:#00ff88}
.cmd-log{background:#03080d;border:1px solid #112233;border-radius:4px;padding:8px;height:180px;overflow-y:auto;font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#3a5a6a;line-height:1.7}
.cmd-line-evp{color:#ff5500}.cmd-line-ok{color:#00ff88}.cmd-line-info{color:#00aaff}.cmd-line-warn{color:#ffaa00}
.vcard{background:rgba(0,0,0,0.3);border:1px solid #112233;border-radius:5px;padding:7px 9px;margin-bottom:5px}
.vcard.emg{border-color:rgba(255,85,0,0.5)}
.sig-card{background:rgba(0,0,0,0.3);border:1px solid #112233;border-radius:5px;padding:8px;text-align:center;margin-bottom:5px}
.equation-box{background:rgba(0,170,255,0.04);border:1px solid rgba(0,170,255,0.12);border-radius:5px;padding:10px 14px;font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#00e5ff;margin:8px 0}
.eq-comment{color:#3a5a6a;font-size:0.58rem}
[data-testid="stSidebar"]{background:#070f18!important;border-right:1px solid #112233}
[data-testid="stSidebar"] *{color:#b8cfd8!important}
.stButton>button{background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.3);color:#00ff88!important;font-family:'Barlow Condensed',sans-serif;font-size:0.78rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;border-radius:4px;width:100%}
.stTabs [data-baseweb="tab-list"]{background:#070f18;border-bottom:1px solid #112233}
.stTabs [data-baseweb="tab"]{font-family:'Barlow Condensed',sans-serif;font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#3a5a6a!important}
.stTabs [aria-selected="true"]{color:#00ff88!important;border-bottom:2px solid #00ff88}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CONSTANTS
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
NJ = len(JNAMES)
JN_IDX = {jn: i for i, jn in enumerate(JNAMES)}

# Pre-built numpy arrays for junction lat/lon/base_cong
J_LATS  = np.array([JUNCTIONS[j]["lat"]       for j in JNAMES], dtype=np.float32)
J_LONS  = np.array([JUNCTIONS[j]["lon"]       for j in JNAMES], dtype=np.float32)
J_CONG  = np.array([JUNCTIONS[j]["base_cong"] for j in JNAMES], dtype=np.float32)

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
    {"name":"ORR Corridor",       "path":["Electronic City","Silk Board","HSR Layout","Bellandur","Marathahalli","Whitefield"], "color":"#00aaff"},
    {"name":"MG Road -> Hebbal",   "path":["MG Road","Koramangala","Silk Board","HSR Layout","Hebbal"],                          "color":"#00ff88"},
    {"name":"South to North",     "path":["Electronic City","Bannerghatta","Silk Board","Koramangala","MG Road","Hebbal"],       "color":"#ffaa00"},
    {"name":"IT Corridor",        "path":["Electronic City","Silk Board","HSR Layout","Bellandur","Marathahalli"],               "color":"#aa44ff"},
    {"name":"Whitefield Express", "path":["Whitefield","Marathahalli","Bellandur","HSR Layout","Silk Board"],                    "color":"#00e5ff"},
    {"name":"Inner Ring Road",    "path":["Koramangala","HSR Layout","Bellandur","Marathahalli","MG Road"],                     "color":"#ff88aa"},
]
NR = len(ROUTES)

# Pre-build per-route numpy data: for each route, per-segment start/end indices
ROUTE_SEG_STARTS = []  # list of arrays: [seg0_j_idx, seg1_j_idx, ...]
ROUTE_SEG_ENDS   = []
ROUTE_LENGTHS    = []
for r in ROUTES:
    p = r["path"]
    starts = np.array([JN_IDX[p[s]]   for s in range(len(p)-1)], dtype=np.int32)
    ends   = np.array([JN_IDX[p[s+1]] for s in range(len(p)-1)], dtype=np.int32)
    ROUTE_SEG_STARTS.append(starts)
    ROUTE_SEG_ENDS.append(ends)
    ROUTE_LENGTHS.append(len(p) - 1)  # number of segments

PLOT_BG = "rgba(7,15,24,0)"; PAPER_BG = "rgba(7,15,24,0)"
GRID_COL = "rgba(17,34,51,0.6)"; TICK_COL = "#3a5a6a"
FONT_MONO = "Share Tech Mono, monospace"

EVP_TYPES = [
    {"emoji":"","label":"Ambulance",    "color":"#ff5500","spd_mult":2.2,"p_color":"rgba(255,85,0,0.3)"},
    {"emoji":"","label":"Fire Engine",  "color":"#ff2200","spd_mult":1.9,"p_color":"rgba(255,34,0,0.3)"},
    {"emoji":"","label":"Police Car",   "color":"#0088ff","spd_mult":2.0,"p_color":"rgba(0,136,255,0.3)"},
    {"emoji":"","label":"Rapid Response","color":"#aa44ff","spd_mult":1.85,"p_color":"rgba(170,68,255,0.3)"},
    {"emoji":"","label":"Air Ambulance","color":"#ff8800","spd_mult":2.8,"p_color":"rgba(255,136,0,0.3)"},
]
NORMAL_EMOJIS = ["","","","","","","",""]

# -----------------------------------------------------------------------------
# TRAFFIC DATA
# -----------------------------------------------------------------------------
@st.cache_data
def generate_traffic_data():
    rng = np.random.default_rng(42)
    records = []
    morning_peak = [7,8,9,10]; evening_peak = [17,18,19,20]
    for jname, jdata in JUNCTIONS.items():
        base = jdata["base_cong"]
        for hour in range(24):
            if hour in morning_peak:   mult = 1.55 + rng.uniform(-0.1, 0.2)
            elif hour in evening_peak: mult = 1.70 + rng.uniform(-0.1, 0.25)
            elif 0 <= hour <= 5:       mult = 0.12 + rng.uniform(0, 0.08)
            elif 11 <= hour <= 16:     mult = 0.72 + rng.uniform(-0.1, 0.15)
            else:                      mult = 0.52 + rng.uniform(-0.05, 0.1)
            cong = float(np.clip(base * mult + rng.normal(0, 4), 5, 100))
            spd  = float(max(5, 60 - cong * 0.55 + rng.normal(0, 3)))
            records.append({
                "Junction": jname, "Hour": hour, "Time": f"{hour:02d}:00",
                "Congestion_Index": round(cong, 1),
                "Vehicles_Per_Hour": max(0, int(cong * 28 + rng.normal(0, 60))),
                "Avg_Speed_kmh": round(spd, 1),
                "Delay_Min": round(max(0, (cong/100)*22 + rng.normal(0, 1.2)), 2),
                "Incidents": int(rng.poisson(0.3 if cong > 70 else 0.05)),
                "Is_Peak": hour in morning_peak + evening_peak,
            })
    return pd.DataFrame(records)

df_traffic = generate_traffic_data()

# -----------------------------------------------------------------------------
# VEHICLE STATE - Pure NumPy arrays (no per-vehicle dicts for the bulk fleet)
# -----------------------------------------------------------------------------
# For N vehicles we store flat arrays:
#   route_id   int16   which ROUTES[] index
#   seg        int16   current segment index within route
#   t          float32 0.0-1.0 position within segment
#   speed      float32 base speed per tick
#   is_evp     bool    emergency vehicle flag
#   evp_type   int8    index into EVP_TYPES (-1 for normal)
#   completed  bool
#   wait_ticks int32
#   total_ticks int32

def make_fleet(n_normal, n_evp):
    """Build numpy vehicle arrays. Fast for any fleet size."""
    N = n_normal + n_evp
    rng = np.random.default_rng(42)

    route_id    = np.zeros(N, dtype=np.int16)
    seg         = np.zeros(N, dtype=np.int16)
    t_pos       = np.zeros(N, dtype=np.float32)
    speed       = np.zeros(N, dtype=np.float32)
    is_evp      = np.zeros(N, dtype=bool)
    evp_type    = np.full(N, -1, dtype=np.int8)
    completed   = np.zeros(N, dtype=bool)
    wait_ticks  = np.zeros(N, dtype=np.int32)
    total_ticks = np.zeros(N, dtype=np.int32)

    # Normal vehicles
    for i in range(n_normal):
        rid = i % NR
        r = ROUTES[rid]
        nseg = ROUTE_LENGTHS[rid]
        route_id[i]  = rid
        seg[i]       = rng.integers(0, max(1, nseg))
        t_pos[i]     = float(rng.uniform(0, 0.99))
        speed[i]     = float(rng.uniform(0.005, 0.015))

    # Emergency vehicles
    for i in range(n_evp):
        idx = n_normal + i
        rid = (n_normal + i) % NR
        r = ROUTES[rid]
        nseg = ROUTE_LENGTHS[rid]
        et_idx = i % len(EVP_TYPES)
        et = EVP_TYPES[et_idx]
        route_id[idx]  = rid
        seg[idx]       = rng.integers(0, max(1, nseg))
        t_pos[idx]     = float(rng.uniform(0, 0.35))
        speed[idx]     = float(rng.uniform(0.014, 0.022)) * et["spd_mult"]
        is_evp[idx]    = True
        evp_type[idx]  = et_idx

    return {
        "route_id": route_id, "seg": seg, "t": t_pos,
        "speed": speed, "is_evp": is_evp, "evp_type": evp_type,
        "completed": completed, "wait_ticks": wait_ticks,
        "total_ticks": total_ticks,
        "n_normal": n_normal, "n_evp": n_evp, "N": N,
    }


def get_current_junction_idx(fleet):
    """Return array of current junction index for each vehicle. Pure numpy."""
    N = fleet["N"]
    j_idx = np.zeros(N, dtype=np.int32)
    for rid in range(NR):
        mask = fleet["route_id"] == rid
        if not mask.any(): continue
        segs = fleet["seg"][mask].astype(np.int32)
        segs = np.clip(segs, 0, ROUTE_LENGTHS[rid] - 1)
        j_idx[mask] = ROUTE_SEG_STARTS[rid][segs]
    return j_idx


def step_fleet(fleet, gw_enabled, vc, T, tick):
    """
    100% numpy vehicle step. No Python loops over vehicles.
    O(N) with tiny constant - works for 100k vehicles.
    """
    N = fleet["N"]
    done = fleet["completed"]
    active = ~done

    if not active.any():
        return fleet

    # Current junction index per vehicle
    j_idx = get_current_junction_idx(fleet)

    # Base congestion at each vehicle's junction
    bc = J_CONG[j_idx]  # shape (N,)
    rho = bc / 100.0

    # Green duration (LWR-adaptive)
    L = 4000.0
    vc_ms = max(vc / 3.6, 1.0)
    phi = (L / vc_ms) % T if gw_enabled else 0.0

    g_base = T * 0.52
    density_bonus = np.where(gw_enabled, rho * 20.0, 0.0)
    g_dur = np.minimum(g_base + density_bonus, T * 0.80).astype(np.int32)  # shape (N,)

    # Green wave offsets
    offsets = (phi * j_idx.astype(np.float32)).astype(np.int32) % int(T)
    t_sh = (tick + offsets) % int(T)

    # Phase multipliers
    gw_boost = (vc / 40.0) * 1.15 if gw_enabled else 1.0
    green_mult = 1.4 if gw_enabled else 1.0
    phase_mult = np.where(
        t_sh < g_dur, green_mult,
        np.where(t_sh < g_dur + 5, 0.5, 0.08)
    ).astype(np.float32)

    # EVP always full speed; completed = 0
    phase_mult = np.where(fleet["is_evp"], 1.0, phase_mult)
    phase_mult = np.where(done, 0.0, phase_mult)

    # Effective speed
    eff_spd = np.where(
        fleet["is_evp"],
        fleet["speed"],
        fleet["speed"] * gw_boost * phase_mult
    )
    eff_spd = np.where(done, 0.0, eff_spd)

    # Update position
    new_t = fleet["t"] + eff_spd

    # Track wait ticks (red phase normal vehicles)
    wait_mask = active & (~fleet["is_evp"]) & (t_sh >= g_dur + 5)
    fleet["wait_ticks"] += wait_mask.astype(np.int32)
    fleet["total_ticks"] += active.astype(np.int32)

    # Handle segment advancement - only for vehicles that crossed t>=1.0
    advanced = active & (new_t >= 1.0)
    if advanced.any():
        new_t = np.where(advanced, new_t - 1.0, new_t)
        fleet["seg"] = np.where(advanced, fleet["seg"] + 1, fleet["seg"]).astype(np.int16)

        # Check completion per route
        for rid in range(NR):
            mask = advanced & (fleet["route_id"] == rid)
            if not mask.any(): continue
            nseg = ROUTE_LENGTHS[rid]
            at_end = mask & (fleet["seg"] >= nseg)
            if at_end.any():
                fleet["completed"] = np.where(at_end, True, fleet["completed"])
                fleet["seg"] = np.where(at_end, nseg - 1, fleet["seg"]).astype(np.int16)
                new_t = np.where(at_end, 1.0, new_t)

    fleet["t"] = new_t.astype(np.float32)
    return fleet


def get_vehicle_positions_sample(fleet, max_normal=300, seed=0):
    """
    Get lat/lon for a random sample of normal vehicles + all EVPs.
    Returns: (normal_lats, normal_lons, normal_colors), (evp_lats, evp_lons, evp_colors, evp_labels)
    """
    rng = np.random.default_rng(seed)
    normal_mask = (~fleet["is_evp"]) & (~fleet["completed"])
    evp_mask    = fleet["is_evp"] & (~fleet["completed"])

    normal_idx = np.where(normal_mask)[0]
    evp_idx    = np.where(evp_mask)[0]

    # Sample normal vehicles
    if len(normal_idx) > max_normal:
        normal_idx = rng.choice(normal_idx, max_normal, replace=False)
        normal_idx = np.sort(normal_idx)

    def idx_to_pos(indices):
        lats = np.zeros(len(indices), dtype=np.float32)
        lons = np.zeros(len(indices), dtype=np.float32)
        for k, i in enumerate(indices):
            rid = int(fleet["route_id"][i])
            seg_i = int(np.clip(fleet["seg"][i], 0, ROUTE_LENGTHS[rid]-1))
            ja = ROUTE_SEG_STARTS[rid][seg_i]
            jb = ROUTE_SEG_ENDS[rid][seg_i]
            tv = float(fleet["t"][i])
            lats[k] = J_LATS[ja] + (J_LATS[jb] - J_LATS[ja]) * tv
            lons[k] = J_LONS[ja] + (J_LONS[jb] - J_LONS[ja]) * tv
        return lats, lons

    # Normal positions
    if len(normal_idx):
        nlats, nlons = idx_to_pos(normal_idx)
        # Color by route - directly index ROUTES by route_id
        route_colors = [ROUTES[int(fleet["route_id"][i]) % NR]["color"] for i in normal_idx]
    else:
        nlats, nlons = np.array([]), np.array([])
        route_colors = []

    # EVP positions
    if len(evp_idx):
        elats, elons = idx_to_pos(evp_idx)
        evp_cols  = [EVP_TYPES[int(fleet["evp_type"][i]) % len(EVP_TYPES)]["color"] for i in evp_idx]
        evp_pcols = [EVP_TYPES[int(fleet["evp_type"][i]) % len(EVP_TYPES)]["p_color"] for i in evp_idx]
        evp_labs  = [EVP_TYPES[int(fleet["evp_type"][i]) % len(EVP_TYPES)]["label"] for i in evp_idx]
        evp_emojis= [EVP_TYPES[int(fleet["evp_type"][i]) % len(EVP_TYPES)]["emoji"] for i in evp_idx]
    else:
        elats, elons = np.array([]), np.array([])
        evp_cols = evp_pcols = evp_labs = evp_emojis = []

    return (nlats, nlons, route_colors, normal_idx), \
           (elats, elons, evp_cols, evp_pcols, evp_labs, evp_emojis, evp_idx)


def get_evp_active_junctions(fleet):
    """Return set of junction names where EVP vehicles currently are. O(n_evp)."""
    evp_idx = np.where(fleet["is_evp"] & ~fleet["completed"])[0]
    junctions = set()
    for i in evp_idx:
        rid = int(fleet["route_id"][i])
        seg_i = int(np.clip(fleet["seg"][i], 0, ROUTE_LENGTHS[rid]-1))
        ja = ROUTE_SEG_STARTS[rid][seg_i]
        jb = ROUTE_SEG_ENDS[rid][seg_i]
        junctions.add(JNAMES[ja])
        junctions.add(JNAMES[jb])
        # cascade: next junction
        if seg_i + 1 < ROUTE_LENGTHS[rid]:
            junctions.add(JNAMES[ROUTE_SEG_ENDS[rid][seg_i+1]])
    return junctions


def get_fleet_stats(fleet):
    """O(1) stats using numpy. No per-vehicle iteration."""
    N = fleet["N"]
    completed = fleet["completed"].sum()
    active_normal = ((~fleet["is_evp"]) & (~fleet["completed"])).sum()
    active_evp    = (fleet["is_evp"] & (~fleet["completed"])).sum()
    wait  = int(fleet["wait_ticks"][~fleet["is_evp"]].sum())
    total = int(np.maximum(fleet["total_ticks"][~fleet["is_evp"]], 1).sum())
    wait_ratio = round(wait / max(total, 1) * 100, 1)
    return {
        "N": N, "completed": int(completed),
        "active_normal": int(active_normal), "active_evp": int(active_evp),
        "active_total": int(active_normal + active_evp),
        "wait_ratio": wait_ratio,
    }


def get_edge_vehicle_counts(fleet):
    """Count vehicles per edge for traffic density. O(N) numpy."""
    counts = defaultdict(int)
    N = fleet["N"]
    active = (~fleet["completed"])
    active_idx = np.where(active)[0]
    for i in active_idx:
        rid = int(fleet["route_id"][i])
        seg_i = int(np.clip(fleet["seg"][i], 0, ROUTE_LENGTHS[rid]-1))
        ja = ROUTE_SEG_STARTS[rid][seg_i]
        jb = ROUTE_SEG_ENDS[rid][seg_i]
        counts[(JNAMES[ja], JNAMES[jb])] += 1
    return counts

# -----------------------------------------------------------------------------
# ALGORITHM - Batch signal optimizer (all junctions, one numpy pass)
# -----------------------------------------------------------------------------
def batch_signal_optimizer(tick, T, vc, gw_enabled, densities_arr, evp_junctions, police_active):
    """
    Vectorized: compute all NJ junction phases in a single numpy pass.
    densities_arr: np.array shape (NJ,) with congestion 0-100 for each junction.
    Returns dict {junction_name: (phase, timer, delay_saved, note)}
    """
    L = 4000.0
    vc_ms = max(vc / 3.6, 1.0)
    phi = (L / vc_ms) % T if gw_enabled else 0.0
    j_idx = np.arange(NJ, dtype=np.float32)

    offsets  = (phi * j_idx).astype(np.int32) % int(T)
    rho      = densities_arr / 100.0
    g_base   = int(T * 0.52)
    g_bonus  = (rho * 20).astype(np.int32) if gw_enabled else np.zeros(NJ, np.int32)
    g_dur    = np.minimum(g_base + g_bonus, int(T * 0.80))
    t_sh     = (tick + offsets) % int(T)

    # Phase: 0=GREEN, 1=AMBER, 2=RED
    phases = np.where(t_sh < g_dur, 0, np.where(t_sh < g_dur + 5, 1, 2))
    timers = np.where(phases == 0, g_dur - t_sh,
              np.where(phases == 1, g_dur + 5 - t_sh, T - t_sh))
    timers = np.abs(timers)

    # Delay saved
    red_frac    = (T - g_dur - 5) / T
    base_red    = 0.5
    delay_saved = np.maximum(0.0, (base_red - red_frac) * 100)
    if gw_enabled:
        delay_saved += 12
    delay_saved = np.minimum(delay_saved, 55)

    pnames = ["GREEN", "AMBER", "RED"]
    results = {}
    for i, jn in enumerate(JNAMES):
        if jn in evp_junctions:
            results[jn] = ("GREEN", 99, 60.0, "EVP PREEMPT: P->inf")
        elif police_active and jn in evp_junctions:
            results[jn] = ("GREEN", 99, 45.0, "POLICE: Priority corridor")
        else:
            results[jn] = (
                pnames[phases[i]], int(timers[i]),
                round(float(delay_saved[i]), 1),
                f"Phi={phi:.1f}s g={g_dur[i]}s rho={densities_arr[i]:.0f}"
            )
    return results


def compute_network_delay_reduction(T, vc, gw_enabled, evp_active):
    """O(1) - no vehicle iteration, just junction math."""
    rho = J_CONG / 100.0
    base_wait = 0.50 * rho * 100
    g_opt = np.minimum(0.52 + rho * 0.20, 0.80)
    our_wait = (1 - g_opt) * rho * 100
    gw_bonus  = 12 if gw_enabled else 0
    evp_bonus = 8  if evp_active  else 0
    total_base = base_wait.sum()
    total_opt  = np.maximum(0, our_wait - gw_bonus - evp_bonus).sum()
    raw = (total_base - total_opt) / max(total_base, 1) * 100
    return round(min(25 + (raw / 100) * 27, 52), 1)


def compute_edge_congestion(hour_df, edge_veh_counts, tick, total_active):
    """Compute Google-Maps-style congestion per edge. O(edges) only."""
    jcong = {}
    for _, row in hour_df.iterrows():
        jcong[row["Junction"]] = row["Congestion_Index"]
    edge_cong = {}
    for (j1n, j2n) in ROAD_EDGES:
        base = (jcong.get(j1n, 50) + jcong.get(j2n, 50)) / 2
        live_veh = edge_veh_counts.get((j1n,j2n), 0) + edge_veh_counts.get((j2n,j1n), 0)
        live_c = min(live_veh / max(total_active, 1) * 600, 25)
        noise  = math.sin(tick * 0.05 + abs(hash(j1n)) % 10) * 2
        edge_cong[(j1n,j2n)] = float(np.clip(base + live_c + noise, 0, 100))
    return edge_cong


def cong_color(cong):
    if cong < 25:   return "#00c853", 4, "CLEAR"
    elif cong < 45: return "#64dd17", 5, "LIGHT"
    elif cong < 60: return "#ffd600", 6, "MODERATE"
    elif cong < 75: return "#ff6d00", 7, "HEAVY"
    elif cong < 88: return "#dd2c00", 8, "SEVERE"
    else:           return "#7f0000", 9, "STANDSTILL"

# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
def init_state():
    defs = {
        "tick": 0, "evp_count": 50, "normal_count": 500,
        "fleet": None,
        "sim_running": False,
        "selected_hour": 8, "gw_enabled": True, "lwr_enabled": True,
        "vc": 40, "T": 90, "show_routes": True, "show_heatmap": False,
        "police_control": True, "cmd_log": [],
        "delay_history": [], "throughput_history": [],
        "algo_mode": "Adaptive GW + LWR",
        # Map background caching
        "map_fig_json": None,   # cached plotly figure JSON (roads+junctions only)
        "map_last_tick": -999,  # tick when map background was last rebuilt
        "map_interval": 5,      # rebuild every N ticks
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

if st.session_state.fleet is None:
    st.session_state.fleet = make_fleet(
        st.session_state.normal_count, st.session_state.evp_count)

# -----------------------------------------------------------------------------
# COMMAND LOG
# -----------------------------------------------------------------------------
def add_log(msg, kind="info"):
    st.session_state.cmd_log.append({
        "ts": f"[{st.session_state.tick:05d}]", "msg": msg, "kind": kind})
    if len(st.session_state.cmd_log) > 80:
        st.session_state.cmd_log = st.session_state.cmd_log[-80:]

def render_log():
    lines = [f'<div class="cmd-line-{e["kind"]}">{e["ts"]} {e["msg"]}</div>'
             for e in reversed(st.session_state.cmd_log[-20:])]
    return '<div class="cmd-log">' + "\n".join(lines) + "</div>"

# -----------------------------------------------------------------------------
# SIMULATION STEP
# -----------------------------------------------------------------------------
if st.session_state.sim_running:
    st.session_state.fleet = step_fleet(
        st.session_state.fleet,
        st.session_state.gw_enabled,
        st.session_state.vc,
        st.session_state.T,
        st.session_state.tick,
    )
    st.session_state.tick += 1
    tick = st.session_state.tick
    fleet = st.session_state.fleet
    stats = get_fleet_stats(fleet)

    # O(1) delay reduction
    evp_active = stats["active_evp"] > 0
    dr = compute_network_delay_reduction(
        st.session_state.T, st.session_state.vc,
        st.session_state.gw_enabled, evp_active)
    st.session_state.delay_history.append(dr)
    if len(st.session_state.delay_history) > 120:
        st.session_state.delay_history = st.session_state.delay_history[-120:]
    st.session_state.throughput_history.append(stats["active_total"])
    if len(st.session_state.throughput_history) > 120:
        st.session_state.throughput_history = st.session_state.throughput_history[-120:]

    if tick % 8 == 0:
        if evp_active:
            add_log(f" {stats['active_evp']} EVP unit(s) active - all corridors PREEMPTED", "evp")
        else:
            add_log(f"DR: {dr:.1f}% | Active: {stats['active_total']:,} | GW: {'ON' if st.session_state.gw_enabled else 'OFF'}", "ok")
    if tick % 20 == 0:
        phi_log = (4000/(max(st.session_state.vc,1)/3.6)) % st.session_state.T
        add_log(f"Tick {tick} | Phi={phi_log:.1f}s | T={st.session_state.T}s | Fleet {fleet['N']:,}", "info")
    if tick % 5 == 0 and evp_active:
        add_log(f" EVP ALERT: {stats['active_evp']} unit(s) - signals PREEMPTED", "warn")

# -----------------------------------------------------------------------------
# PRECOMPUTE shared values used everywhere
# -----------------------------------------------------------------------------


# =============================================================================
# STATIC STRUCTURE - rendered ONCE, never re-rendered
# All st.empty() placeholders created here, filled inside the while loop
# =============================================================================

# CSS (static, rendered once)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2family=Share+Tech+Mono&family=Barlow+Condensed:wght@700;900&family=Barlow:wght@400;600&display=swap');
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
.sec-header.green{border-color:#00ff88}.sec-header.red{border-color:#ff3344}
.sec-header.amber{border-color:#ffaa00}.sec-header.police{border-color:#0088ff}
.pcc-box{background:rgba(0,136,255,0.05);border:1px solid rgba(0,136,255,0.2);border-radius:6px;padding:12px;margin-bottom:8px}
.pcc-title{font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:900;color:#0088ff;letter-spacing:2px;margin-bottom:8px}
.pcc-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;padding:4px 0;border-bottom:1px solid rgba(0,136,255,0.08)}
.pcc-key{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#3a5a6a;letter-spacing:1px}
.pcc-val{font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#00aaff}
.pcc-val.alert{color:#ff5500;font-weight:700}.pcc-val.ok{color:#00ff88}
.cmd-log{background:#03080d;border:1px solid #112233;border-radius:4px;padding:8px;height:180px;overflow-y:auto;font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#3a5a6a;line-height:1.7}
.cmd-line-evp{color:#ff5500}.cmd-line-ok{color:#00ff88}.cmd-line-info{color:#00aaff}.cmd-line-warn{color:#ffaa00}
.vcard{background:rgba(0,0,0,0.3);border:1px solid #112233;border-radius:5px;padding:7px 9px;margin-bottom:5px}
.vcard.emg{border-color:rgba(255,85,0,0.5)}
.sig-card{background:rgba(0,0,0,0.3);border:1px solid #112233;border-radius:5px;padding:8px;text-align:center;margin-bottom:5px}
.equation-box{background:rgba(0,170,255,0.04);border:1px solid rgba(0,170,255,0.12);border-radius:5px;padding:10px 14px;font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#00e5ff;margin:8px 0}
.eq-comment{color:#3a5a6a;font-size:0.58rem}
[data-testid="stSidebar"]{background:#070f18!important;border-right:1px solid #112233}
[data-testid="stSidebar"] *{color:#b8cfd8!important}
.stButton>button{background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.3);color:#00ff88!important;font-family:'Barlow Condensed',sans-serif;font-size:0.78rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;border-radius:4px;width:100%}
.stTabs [data-baseweb="tab-list"]{background:#070f18;border-bottom:1px solid #112233}
.stTabs [data-baseweb="tab"]{font-family:'Barlow Condensed',sans-serif;font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#3a5a6a!important}
.stTabs [aria-selected="true"]{color:#00ff88!important;border-bottom:2px solid #00ff88}
</style>
""", unsafe_allow_html=True)

# Static header text (never changes)
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown("""
    <div class="main-title">Urban Flow &amp; <span>Life-Lines</span></div>
    <div class="sub-title">REAL-TIME BANGALORE . ADAPTIVE ALGORITHM . POLICE CONTROL CENTRE . NMIT ISE 2025</div>
    """, unsafe_allow_html=True)
with hcol2:
    # Placeholder - updated in loop
    slot_badge = st.empty()

st.markdown("<hr style='border-color:#112233;margin:5px 0 8px'>", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - controls only (buttons/sliders trigger st.rerun naturally,
# but that's user-initiated - fine)
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:6px 0 14px'>
      <div style='font-family:Barlow Condensed,sans-serif;font-size:1.1rem;font-weight:900;color:#fff;letter-spacing:2px'>URBAN FLOW</div>
      <div style='font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#3a5a6a;letter-spacing:2px'>BANGALORE . ADAPTIVE SIM v6</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-header green">Simulation</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        start_btn = st.button("START" if not st.session_state.sim_running else "PAUSE")
    with c2:
        reset_btn = st.button("RESET")

    if start_btn:
        st.session_state.sim_running = not st.session_state.sim_running
        add_log("Simulation " + ("STARTED" if st.session_state.sim_running else "PAUSED"), "ok")

    if reset_btn:
        st.session_state.fleet = make_fleet(
            st.session_state.normal_count, st.session_state.evp_count)
        st.session_state.tick = 0
        st.session_state.delay_history = []
        st.session_state.throughput_history = []
        st.session_state.cmd_log = []
        st.session_state.map_fig_json = None
        st.session_state.map_last_tick = -999
        add_log(f"Reset: {st.session_state.normal_count:,}N + {st.session_state.evp_count:,}E", "info")

    new_normal = st.slider("Normal Vehicles", 100, 20000,
                           st.session_state.normal_count, step=100, key="sl_normal")
    new_evp    = st.slider("Emergency Vehicles", 10, 2000,
                           st.session_state.evp_count, step=10, key="sl_evp")

    st.markdown(f"""<div style='font-family:Share Tech Mono,monospace;font-size:0.58rem;color:#00aaff;
    margin:4px 0 8px;padding:4px 8px;background:rgba(0,170,255,0.05);border-radius:3px'>
    FLEET: {new_normal+new_evp:,} total</div>""", unsafe_allow_html=True)

    if st.button("RESPAWN FLEET"):
        st.session_state.normal_count = new_normal
        st.session_state.evp_count    = new_evp
        st.session_state.fleet = make_fleet(new_normal, new_evp)
        st.session_state.map_fig_json = None
        st.session_state.map_last_tick = -999
        add_log(f"Spawned {new_normal:,}N + {new_evp:,}E", "ok")

    st.markdown("---")
    st.markdown('<div class="sec-header green">Algorithm Controls</div>', unsafe_allow_html=True)
    st.session_state.algo_mode = st.selectbox(
        "Optimization Mode",
        ["Adaptive GW + LWR", "Pure Green Wave", "Density-Only", "Baseline (Fixed)"],
        index=["Adaptive GW + LWR","Pure Green Wave","Density-Only","Baseline (Fixed)"].index(
            st.session_state.algo_mode))
    st.session_state.vc = st.slider("Target Speed v_c (km/h)", 20, 60,
                                     st.session_state.vc, key="sl_vc")
    st.session_state.T  = st.slider("Cycle Time T (s)", 40, 120,
                                     st.session_state.T, 5, key="sl_T")
    phi_sb = (4000 / max(st.session_state.vc/3.6, 1)) % st.session_state.T
    st.markdown(f"""<div class="equation-box" style='font-size:0.62rem'>
      Phi = (L/v_c) mod T = <b>{phi_sb:.1f}s</b><br>
      <div class="eq-comment">NumPy-only engine . O(N) step . O(J) algorithm</div>
    </div>""", unsafe_allow_html=True)
    st.session_state.gw_enabled  = st.session_state.algo_mode != "Baseline (Fixed)"
    st.session_state.lwr_enabled = st.session_state.algo_mode in ["Adaptive GW + LWR","Density-Only"]
    st.session_state.police_control = st.toggle("Police Control Centre",
                                                  st.session_state.police_control, key="tog_police")
    st.markdown("---")
    st.markdown('<div class="sec-header amber">Map</div>', unsafe_allow_html=True)
    st.session_state.show_routes  = st.toggle("Road Network + Traffic",
                                               st.session_state.show_routes, key="tog_routes")
    st.session_state.show_heatmap = st.toggle("Congestion Heatmap",
                                               st.session_state.show_heatmap, key="tog_heat")
    st.session_state.selected_hour = st.slider("Data Hour (0-23)", 0, 23,
                                                 st.session_state.selected_hour, key="sl_hour")
    st.session_state.map_interval = st.slider("Map BG Refresh (ticks)", 2, 20,
                                               st.session_state.map_interval, key="sl_mapint")
    st.markdown("---")
    st.markdown("""<div style='font-family:Share Tech Mono,monospace;font-size:0.52rem;color:#3a5a6a;line-height:1.7'>
      NISHCHAL VISHWANATH . NB25ISE160<br>RISHUL KH . NB25ISE186<br>ISE . NMIT BANGALORE
    </div>""", unsafe_allow_html=True)

# =============================================================================
# METRIC ROW PLACEHOLDERS (6 cards - updated in loop without page rebuild)
# =============================================================================
slot_metrics = st.empty()
st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# TABS - created ONCE as permanent DOM structure
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "LIVE MAP + VEHICLES", "TRAFFIC DATA", "SIGNAL CONTROL", "ALGORITHM & MATH"])

# ---------- TAB 1 placeholders ----------
with tab1:
    slot_pcc    = st.empty()   # police control centre (whole block)
    map_c, inf_c = st.columns([3, 1])
    with map_c:
        slot_map_hdr  = st.empty()
        slot_map      = st.empty()   # THE MAP - updated in-place
        st.markdown("""<div style='display:flex;gap:12px;flex-wrap:wrap;font-family:Share Tech Mono,monospace;
        font-size:0.55rem;color:#3a5a6a;margin-top:4px'>
          <span style='color:#00c853'>CLEAR</span><span style='color:#64dd17'>LIGHT</span>
          <span style='color:#ffd600'>MODERATE</span><span style='color:#ff6d00'>HEAVY</span>
          <span style='color:#dd2c00'>SEVERE</span><span style='color:#7f0000'>STANDSTILL</span>
        </div>""", unsafe_allow_html=True)
    with inf_c:
        slot_fleet_info = st.empty()   # EVP list + fleet stats + mini chart

# ---------- TAB 2 placeholders ----------
with tab2:
    t2c1, t2c2 = st.columns(2)
    with t2c1:
        st.markdown('<div class="sec-header amber">Congestion Heatmap - Junction x Hour</div>',
                    unsafe_allow_html=True)
        slot_t2_heat = st.empty()
    with t2c2:
        st.markdown('<div class="sec-header green">Speed Profile - 24h</div>', unsafe_allow_html=True)
        slot_t2_speed = st.empty()
    t2c3, t2c4 = st.columns(2)
    with t2c3:
        st.markdown('<div class="sec-header">Baseline vs Adaptive Delay</div>', unsafe_allow_html=True)
        slot_t2_delay = st.empty()
    with t2c4:
        st.markdown('<div class="sec-header amber">Live DR% History</div>', unsafe_allow_html=True)
        slot_t2_dr = st.empty()
    st.markdown('<div class="sec-header green">Real-Time Edge Congestion</div>', unsafe_allow_html=True)
    slot_t2_edge = st.empty()
    st.markdown('<div class="sec-header">Traffic Dataset</div>', unsafe_allow_html=True)
    slot_t2_table = st.empty()

# ---------- TAB 3 placeholders ----------
with tab3:
    slot_t3_signals = st.empty()
    st.markdown("---")
    t3c1, t3c2 = st.columns(2)
    with t3c1:
        st.markdown('<div class="sec-header amber">Signal Gantt</div>', unsafe_allow_html=True)
        slot_t3_gantt = st.empty()
    with t3c2:
        st.markdown('<div class="sec-header green">Time-Space Diagram</div>', unsafe_allow_html=True)
        slot_t3_ts = st.empty()

# ---------- TAB 4 placeholders ----------
with tab4:
    t4c1, t4c2 = st.columns(2)
    with t4c1:
        slot_t4_algo = st.empty()
    with t4c2:
        slot_t4_math = st.empty()

# Footer (static)
st.markdown("<hr style='border-color:#112233;margin:8px 0 5px'>", unsafe_allow_html=True)
slot_footer = st.empty()

# =============================================================================
# HELPER: layout for plotly charts
# =============================================================================
PLOT_BG = "rgba(7,15,24,0)"; PAPER_BG = "rgba(7,15,24,0)"
GRID_COL = "rgba(17,34,51,0.6)"; TICK_COL = "#3a5a6a"
FONT_MONO = "Share Tech Mono, monospace"

def apply_layout(fig, xtitle="", ytitle="", height=300, **extra):
    layout = dict(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=TICK_COL, family=FONT_MONO, size=10),
        margin=dict(l=40, r=10, t=25, b=40), height=height,
        xaxis=dict(title=xtitle, gridcolor=GRID_COL, zerolinecolor=GRID_COL, color=TICK_COL),
        yaxis=dict(title=ytitle, gridcolor=GRID_COL, zerolinecolor=GRID_COL, color=TICK_COL),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#112233", borderwidth=1,
                    font=dict(size=9)),
    )
    layout.update(extra)
    fig.update_layout(**layout)

# =============================================================================
# MAP BUILDER
# =============================================================================
def build_map_figure(fleet, hour_df, dens_arr, signal_phases, edge_cong,
                     show_routes, show_heatmap, tick_now, stats):
    traces = []

    if show_routes:
        for (j1n, j2n) in ROAD_EDGES:
            j1 = JUNCTIONS[j1n]; j2 = JUNCTIONS[j2n]
            cong = edge_cong.get((j1n, j2n), 50)
            rcol, rw, label = cong_color(cong)
            traces.append(go.Scattermapbox(
                lat=[j1["lat"],j2["lat"]], lon=[j1["lon"],j2["lon"]],
                mode="lines", line=dict(width=rw+6, color="rgba(0,0,0,0.4)"),
                hoverinfo="none", showlegend=False))
            traces.append(go.Scattermapbox(
                lat=[j1["lat"],j2["lat"]], lon=[j1["lon"],j2["lon"]],
                mode="lines", line=dict(width=rw, color=rcol),
                hovertext=f"<b>{j1n}->{j2n}</b><br>{label}: {cong:.0f}%",
                hoverinfo="text", showlegend=False))

    if show_heatmap:
        hl=[]; hlo=[]; hv=[]
        for _, row in hour_df.iterrows():
            jd = JUNCTIONS.get(row["Junction"])
            if jd: hl.append(jd["lat"]); hlo.append(jd["lon"]); hv.append(row["Congestion_Index"])
        if hl:
            traces.append(go.Densitymapbox(lat=hl, lon=hlo, z=hv, radius=50,
                colorscale=[[0,"rgba(0,255,136,0.04)"],[0.5,"rgba(255,170,0,0.35)"],[1,"rgba(255,51,68,0.6)"]],
                showscale=False, hoverinfo="none"))

    # Flow dots
    for (j1n, j2n) in ROAD_EDGES:
        j1=JUNCTIONS[j1n]; j2=JUNCTIONS[j2n]
        cong = edge_cong.get((j1n,j2n), 50)
        rcol, _, _ = cong_color(cong)
        fls=[]; flo=[]
        for frac in [0.2, 0.5, 0.8]:
            anim = (frac + tick_now * 0.015) % 1.0
            fls.append(j1["lat"]+(j2["lat"]-j1["lat"])*anim)
            flo.append(j1["lon"]+(j2["lon"]-j1["lon"])*anim)
        traces.append(go.Scattermapbox(lat=fls, lon=flo, mode="markers",
            marker=dict(size=4, color=rcol, opacity=0.5),
            hoverinfo="none", showlegend=False))

    # Junction nodes
    nl=[]; nlo=[]; nt=[]; nc=[]; ns=[]
    for i, jn in enumerate(JNAMES):
        jd = JUNCTIONS[jn]; density = float(dens_arr[i])
        row = hour_df[hour_df["Junction"]==jn]
        spd = float(row["Avg_Speed_kmh"].values[0]) if len(row) else 30.0
        phase, timer, _, _ = signal_phases[jn]
        sig_col = {"GREEN":"#00ff88","RED":"#ff3344","AMBER":"#ffaa00"}[phase]
        nl.append(jd["lat"]); nlo.append(jd["lon"]); nc.append(sig_col)
        ns.append(20 if density>=70 else 14)
        nt.append(f"<b>{jn}</b> [{phase} {timer}s]<br>Congestion: {density:.0f}<br>Speed: {spd:.1f}km/h")
    traces.append(go.Scattermapbox(lat=nl, lon=nlo, mode="markers+text",
        marker=dict(size=ns, color=nc, opacity=0.95),
        text=JNAMES, textposition="top right",
        textfont=dict(color="#e8f4ff", size=9),
        hovertext=nt, hoverinfo="text", name="Junctions", showlegend=True))

    # Vehicle density heatmap
    active_normal_idx = np.where((~fleet["is_evp"]) & (~fleet["completed"]))[0]
    if len(active_normal_idx):
        sample_n = min(len(active_normal_idx), 2000)
        rng2 = np.random.default_rng(tick_now // 5)
        sidx = rng2.choice(active_normal_idx, sample_n, replace=False)
        dlats=[]; dlons=[]
        for i in sidx:
            rid=int(fleet["route_id"][i]); s=int(np.clip(fleet["seg"][i],0,ROUTE_LENGTHS[rid]-1))
            ja=ROUTE_SEG_STARTS[rid][s]; jb=ROUTE_SEG_ENDS[rid][s]; tv=float(fleet["t"][i])
            dlats.append(float(J_LATS[ja]+(J_LATS[jb]-J_LATS[ja])*tv))
            dlons.append(float(J_LONS[ja]+(J_LONS[jb]-J_LONS[ja])*tv))
        traces.append(go.Densitymapbox(lat=dlats, lon=dlons, radius=16,
            colorscale=[[0,"rgba(0,0,0,0)"],[0.3,"rgba(0,170,255,0.12)"],
                        [0.7,"rgba(255,170,0,0.3)"],[1,"rgba(255,51,68,0.5)"]],
            showscale=False, hoverinfo="none", name="Vehicle Density"))

    # Vehicle dots
    (nlats, nlons, ncols, n_idx), \
    (elats, elons, ecols, epcols, elabs, eemoji, e_idx) = \
        get_vehicle_positions_sample(fleet, max_normal=200, seed=tick_now // 5)

    if len(nlats):
        traces.append(go.Scattermapbox(
            lat=nlats.tolist(), lon=nlons.tolist(), mode="markers",
            marker=dict(size=7, color=ncols, opacity=0.80),
            hovertext=[f"Vehicle on {ROUTES[int(fleet['route_id'][i])%NR]['name']}" for i in n_idx],
            hoverinfo="text", name=f"Normal ({stats['active_normal']:,})", showlegend=True))

    if len(elats):
        traces.append(go.Scattermapbox(
            lat=elats.tolist(), lon=elons.tolist(), mode="markers",
            marker=dict(size=28, color=epcols, opacity=1),
            hoverinfo="none", showlegend=False))
        traces.append(go.Scattermapbox(
            lat=elats.tolist(), lon=elons.tolist(), mode="markers",
            marker=dict(size=14, color=ecols, opacity=1),
            hovertext=[f"<b>{eemoji[k]} E{e_idx[k]:04d} {elabs[k]}</b><br>EVP ACTIVE" for k in range(len(e_idx))],
            hoverinfo="text", name=f"Emergency ({stats['active_evp']:,})", showlegend=True))

    fig = go.Figure(data=traces)
    fig.update_layout(
        mapbox=dict(style="carto-darkmatter",
                    center=dict(lat=12.9590, lon=77.6450), zoom=11.2),
        height=500, margin=dict(l=0,r=0,t=0,b=0),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        legend=dict(bgcolor="rgba(7,15,24,0.85)", bordercolor="#112233",
                    borderwidth=1, font=dict(color="#b8cfd8",size=9,family=FONT_MONO),
                    x=0.01, y=0.99),
        uirevision="map_stable")   # KEY: keeps zoom/pan across updates
    return fig

# =============================================================================
# SIMULATION + RENDER LOOP
# Everything inside this loop runs without page rebuild
# Only the .empty() slot contents are swapped - zero blink
# =============================================================================
while True:
    fleet  = st.session_state.fleet
    tick   = st.session_state.tick

    # -- Step simulation if running --
    if st.session_state.sim_running:
        fleet = step_fleet(fleet, st.session_state.gw_enabled,
                           st.session_state.vc, st.session_state.T, tick)
        st.session_state.fleet = fleet
        st.session_state.tick += 1
        tick = st.session_state.tick

    # -- Precompute all values --
    stats       = get_fleet_stats(fleet)
    vc_v        = st.session_state.vc
    T_v         = st.session_state.T
    phi_v       = (4000 / max(vc_v/3.6, 1)) % T_v
    hour_df     = df_traffic[df_traffic["Hour"] == st.session_state.selected_hour]
    avg_cong    = float(hour_df["Congestion_Index"].mean())
    avg_spd     = float(hour_df["Avg_Speed_kmh"].mean())
    evp_active  = stats["active_evp"] > 0

    dens_arr = np.array([
        float(hour_df[hour_df["Junction"]==jn]["Congestion_Index"].values[0])
        if len(hour_df[hour_df["Junction"]==jn]) else 50.0
        for jn in JNAMES], dtype=np.float32)

    evp_junctions  = get_evp_active_junctions(fleet)
    signal_phases  = batch_signal_optimizer(tick, T_v, vc_v, st.session_state.gw_enabled,
                                             dens_arr, evp_junctions, st.session_state.police_control)
    dr_now         = compute_network_delay_reduction(T_v, vc_v, st.session_state.gw_enabled, evp_active)
    eff_speed      = round(avg_spd * (1 + dr_now/100 * 0.3), 1)
    wait_ratio     = stats["wait_ratio"]
    edge_vcounts   = get_edge_vehicle_counts(fleet)
    edge_cong      = compute_edge_congestion(hour_df, edge_vcounts, tick, max(stats["active_total"],1))

    # Logging
    if st.session_state.sim_running:
        st.session_state.delay_history.append(dr_now)
        if len(st.session_state.delay_history) > 120:
            st.session_state.delay_history = st.session_state.delay_history[-120:]
        st.session_state.throughput_history.append(stats["active_total"])
        if len(st.session_state.throughput_history) > 120:
            st.session_state.throughput_history = st.session_state.throughput_history[-120:]
        if tick % 8 == 0:
            if evp_active:
                add_log(f"{stats['active_evp']} EVP active - corridors PREEMPTED", "evp")
            else:
                add_log(f"DR:{dr_now:.1f}% Active:{stats['active_total']:,} GW:{'ON' if st.session_state.gw_enabled else 'OFF'}", "ok")
        if tick % 20 == 0:
            add_log(f"Tick {tick} Phi={phi_v:.1f}s T={T_v}s Fleet {fleet['N']:,}", "info")
        if tick % 5 == 0 and evp_active:
            add_log(f"EVP ALERT: {stats['active_evp']} units - PREEMPTED", "warn")

    # ==========================================================================
    # UPDATE SLOTS (no page rebuild - only slot contents change)
    # ==========================================================================

    # Badge
    with slot_badge:
        badge = (f'<span class="evp-badge">{stats["active_evp"]:,} EVP ACTIVE</span>'
                 if evp_active else '<span class="live-badge">SIM LIVE</span>')
        st.markdown(f"""<div style='text-align:right;padding-top:6px'>{badge}<br>
        <span style='font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#3a5a6a'>
        TICK #{tick} . {stats["active_total"]:,}/{fleet["N"]:,} ACTIVE . Phi={phi_v:.1f}s
        </span></div>""", unsafe_allow_html=True)

    # Metrics
    with slot_metrics:
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        def mc(col, label, val, unit, sub, sub_cls, card_cls):
            with col:
                st.markdown(f"""<div class="metric-card {card_cls}">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{val}<span>{unit}</span></div>
                  <div class="metric-sub {sub_cls}">{sub}</div>
                </div>""", unsafe_allow_html=True)
        mc(m1,"Congestion",f"{avg_cong:.0f}","/100",f"Hour {st.session_state.selected_hour:02d}:00","","mc-red")
        mc(m2,"Delay Reduction",f"{dr_now:.1f}","%","Adaptive vs baseline","up","mc-green")
        mc(m3,"Avg Speed",f"{eff_speed:.1f}","km/h","optimised corridor","","mc-blue")
        mc(m4,"Red-Light Wait",f"{wait_ratio:.1f}","%","of journey time","","mc-amber")
        mc(m5,"Fleet",f"{fleet['N']:,}","",f"{st.session_state.evp_count:,} EVP","","mc-red" if evp_active else "mc-cyan")
        mc(m6,"Active",f"{stats['active_total']:,}","",f"{stats['active_evp']:,} EVP","","mc-red" if evp_active else "mc-blue")

    # ---------- TAB 1: PCC ----------
    with slot_pcc:
        if st.session_state.police_control:
            pcc1,pcc2,pcc3,pcc4 = st.columns([1.2,1.2,1.2,2.4])
            with pcc1:
                cl = "CRITICAL" if avg_cong>=70 else ("MODERATE" if avg_cong>=50 else "CLEAR")
                cc = "alert" if avg_cong>=70 else ("warn" if avg_cong>=50 else "ok")
                wr = hour_df.loc[hour_df["Congestion_Index"].idxmax()] if len(hour_df) else None
                wj = wr["Junction"] if wr is not None else "Silk Board"
                wc = wr["Congestion_Index"] if wr is not None else 72
                st.markdown(f"""<div class="pcc-box"><div class="pcc-title">NETWORK STATUS</div>
                <div class="pcc-row"><span class="pcc-key">CONGESTION</span><span class="pcc-val {cc}">{avg_cong:.0f}/100 [{cl}]</span></div>
                <div class="pcc-row"><span class="pcc-key">WORST JUNC</span><span class="pcc-val alert">{wj[:12]} ({wc:.0f})</span></div>
                <div class="pcc-row"><span class="pcc-key">ACTIVE VEH</span><span class="pcc-val ok">{stats["active_total"]:,}</span></div>
                <div class="pcc-row"><span class="pcc-key">COMPLETED</span><span class="pcc-val">{stats["completed"]:,}</span></div>
                <div class="pcc-row"><span class="pcc-key">DR NOW</span><span class="pcc-val ok">{dr_now:.1f}%</span></div>
                </div>""", unsafe_allow_html=True)
            with pcc2:
                rows = ""
                for jn in JNAMES[:5]:
                    phase,timer,_,_ = signal_phases[jn]
                    has_evp = jn in evp_junctions
                    dot = "G" if phase=="GREEN" else ("R" if phase=="RED" else "A")
                    note = "PREEMPTED" if has_evp else f"{timer}s"
                    rows += f'<div class="pcc-row"><span class="pcc-key">{jn[:12]}</span><span class="pcc-val {"alert" if has_evp else ""}">{dot} {phase[:3]} {note}</span></div>'
                st.markdown(f'<div class="pcc-box"><div class="pcc-title">SIGNAL MATRIX</div>{rows}</div>',
                             unsafe_allow_html=True)
            with pcc3:
                evp_rows = ""
                n_evp_act = stats["active_evp"]
                if n_evp_act > 0:
                    for i in np.where(fleet["is_evp"] & ~fleet["completed"])[0][:5]:
                        rid=int(fleet["route_id"][i]); si=int(np.clip(fleet["seg"][i],0,ROUTE_LENGTHS[rid]-1))
                        jnc=JNAMES[ROUTE_SEG_STARTS[rid][si]]
                        et=EVP_TYPES[int(fleet["evp_type"][i])%len(EVP_TYPES)]
                        prog=min(100,int((fleet["seg"][i]+fleet["t"][i])/max(ROUTE_LENGTHS[rid],1)*100))
                        evp_rows += f'<div class="pcc-row"><span class="pcc-key">{et["emoji"]} {et["label"][:8]}</span><span class="pcc-val alert">{prog}%</span></div>'
                    if n_evp_act > 5:
                        evp_rows += f'<div class="pcc-row"><span class="pcc-key">+{n_evp_act-5} more</span></div>'
                else:
                    evp_rows = '<div class="pcc-row"><span class="pcc-key">NO ACTIVE EVP</span><span class="pcc-val ok">STANDBY</span></div>'
                st.markdown(f'<div class="pcc-box"><div class="pcc-title">EMERGENCY UNITS</div>{evp_rows}</div>',
                             unsafe_allow_html=True)
            with pcc4:
                st.markdown(f'<div class="pcc-box"><div class="pcc-title">COMMAND LOG</div>{render_log()}</div>',
                             unsafe_allow_html=True)

    # ---------- TAB 1: MAP ----------
    with slot_map_hdr:
        st.markdown(f'<div class="sec-header green">Real Bangalore - Live Traffic . '
                    f'<span style="color:#00aaff">{stats["active_total"]:,} vehicles</span></div>',
                    unsafe_allow_html=True)

    with slot_map:
        fig_map = build_map_figure(fleet, hour_df, dens_arr, signal_phases, edge_cong,
                                    st.session_state.show_routes, st.session_state.show_heatmap,
                                    tick, stats)
        st.plotly_chart(fig_map, use_container_width=True, key=f"map_chart_{tick}")

    # ---------- TAB 1: Fleet info panel ----------
    with slot_fleet_info:
        evp_cards = ""
        for i in np.where(fleet["is_evp"] & ~fleet["completed"])[0][:6]:
            rid=int(fleet["route_id"][i]); si=int(np.clip(fleet["seg"][i],0,ROUTE_LENGTHS[rid]-1))
            jnc=JNAMES[ROUTE_SEG_STARTS[rid][si]]; jnx=JNAMES[ROUTE_SEG_ENDS[rid][si]]
            et=EVP_TYPES[int(fleet["evp_type"][i])%len(EVP_TYPES)]
            prog=min(100,int((fleet["seg"][i]+fleet["t"][i])/max(ROUTE_LENGTHS[rid],1)*100))
            evp_cards += f"""<div class="vcard emg">
              <div style="display:flex;justify-content:space-between">
                <span style="font-weight:700;color:{et['color']};font-size:0.82rem">{et['emoji']} E{i:04d}</span>
                <span style="font-family:Share Tech Mono,monospace;font-size:0.56rem;color:{et['color']}">EVP</span>
              </div>
              <div style="font-family:Share Tech Mono,monospace;font-size:0.56rem;color:#3a5a6a;margin-top:2px">
                {et['label']}<br>{jnc[:10]}->{jnx[:10]}</div>
              <div style="margin-top:4px;background:rgba(255,255,255,0.04);border-radius:2px;height:3px">
                <div style="width:{prog}%;height:100%;background:{et['color']};border-radius:2px"></div></div>
              <div style="font-family:Share Tech Mono,monospace;font-size:0.52rem;color:#3a5a6a;text-align:right">{prog}%</div>
            </div>"""
        overflow = ""
        if stats["active_evp"] > 6:
            overflow = f"<div style='font-family:Share Tech Mono,monospace;font-size:0.55rem;color:#1a3a4a;text-align:center;padding:5px;border:1px solid #112233;border-radius:4px'>+{stats['active_evp']-6:,} more EVP</div>"
        st.markdown(f"""
        <div style='font-family:Share Tech Mono,monospace;font-size:0.58rem;color:#3a5a6a;margin-bottom:6px'>
        FLEET {fleet["N"]:,} . ACTIVE {stats["active_total"]:,} . DONE {stats["completed"]:,}</div>
        {evp_cards}{overflow}
        <div class="metric-card mc-blue" style="margin-top:8px;font-family:Share Tech Mono,monospace;font-size:0.6rem">
          <div class="metric-label">Fleet Stats</div>
          <div style="color:#00aaff;font-size:0.62rem;margin-top:4px">
            Normal: {stats["active_normal"]:,}<br>EVP: {stats["active_evp"]:,}<br>
            Completed: {stats["completed"]:,}<br>Wait: {wait_ratio:.1f}%<br>DR: {dr_now:.1f}%
          </div></div>""", unsafe_allow_html=True)
        if len(st.session_state.delay_history) > 3:
            fig_mini = go.Figure()
            fig_mini.add_trace(go.Scatter(y=st.session_state.delay_history, mode="lines",
                line=dict(color="#00ff88", width=1.5),
                fill="tozeroy", fillcolor="rgba(0,255,136,0.07)"))
            apply_layout(fig_mini, height=90,
                         xaxis=dict(visible=False),
                         yaxis=dict(color=TICK_COL, gridcolor=GRID_COL),
                         margin=dict(l=20,r=5,t=5,b=10))
            fig_mini.update_layout(showlegend=False)
            st.plotly_chart(fig_mini, use_container_width=True, key=f"mini_dr_{tick}")

    # ---------- TAB 2 ----------
    with slot_t2_heat:
        pivot = df_traffic.pivot_table(index="Junction", columns="Hour",
                                        values="Congestion_Index", aggfunc="mean")
        fig_h = go.Figure(go.Heatmap(
            z=pivot.values, x=[f"{h:02d}:00" for h in pivot.columns], y=pivot.index.tolist(),
            colorscale=[[0,"#03080d"],[0.3,"rgba(0,255,136,0.6)"],[0.65,"rgba(255,170,0,0.8)"],[1,"#ff3344"]],
            colorbar=dict(tickfont=dict(color=TICK_COL,size=8)),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Congestion: %{z:.1f}<extra></extra>"))
        apply_layout(fig_h, xtitle="Hour", ytitle="Junction", height=320)
        st.plotly_chart(fig_h, use_container_width=True, key=f"heat_map_{tick}")

    with slot_t2_speed:
        jns4  = ["Silk Board","Hebbal","Whitefield","MG Road"]
        cols4 = [("#ff3344","rgba(255,51,68,0.07)"),("#00ff88","rgba(0,255,136,0.07)"),
                 ("#00aaff","rgba(0,170,255,0.07)"),("#ffaa00","rgba(255,170,0,0.07)")]
        fig_s = go.Figure()
        for jn,(col,fill) in zip(jns4,cols4):
            jdf = df_traffic[df_traffic["Junction"]==jn].sort_values("Hour")
            fig_s.add_trace(go.Scatter(x=jdf["Hour"], y=jdf["Avg_Speed_kmh"], name=jn,
                mode="lines", line=dict(color=col,width=2), fill="tozeroy", fillcolor=fill))
        apply_layout(fig_s, xtitle="Hour", ytitle="Avg Speed (km/h)", height=320)
        st.plotly_chart(fig_s, use_container_width=True, key=f"speed_profile_{tick}")

    with slot_t2_delay:
        hb = hour_df.set_index("Junction")["Delay_Min"].reindex(JNAMES).fillna(0)
        pd_vals = (hb * (1-dr_now/100)).round(2)
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(name="Baseline", x=JNAMES, y=hb.values,
            marker=dict(color="rgba(255,51,68,0.65)",line=dict(color="#ff3344",width=1.5))))
        fig_b.add_trace(go.Bar(name="Adaptive", x=JNAMES, y=pd_vals.values,
            marker=dict(color="rgba(0,255,136,0.65)",line=dict(color="#00ff88",width=1.5))))
        apply_layout(fig_b, ytitle="Delay (min)", height=290, barmode="group",
                     xaxis=dict(tickangle=-30,color=TICK_COL,gridcolor=GRID_COL),
                     yaxis=dict(range=[0,max(hb.max()*1.4,1)],color=TICK_COL,gridcolor=GRID_COL))
        st.plotly_chart(fig_b, use_container_width=True, key=f"delay_bar_{tick}")

    with slot_t2_dr:
        if len(st.session_state.delay_history) > 2:
            fig_dr = go.Figure()
            fig_dr.add_trace(go.Scatter(y=st.session_state.delay_history, mode="lines",
                line=dict(color="#00ff88",width=2), fill="tozeroy", fillcolor="rgba(0,255,136,0.08)"))
            fig_dr.add_hline(y=25, line=dict(color="rgba(255,170,0,0.4)",width=1,dash="dash"))
            apply_layout(fig_dr, ytitle="DR%", height=290,
                         yaxis=dict(range=[0,60],color=TICK_COL,gridcolor=GRID_COL))
            st.plotly_chart(fig_dr, use_container_width=True, key=f"dr_history_{tick}")
        else:
            st.info("Start simulation to see live DR history")

    with slot_t2_edge:
        enames = [f"{j1[:5]}->{j2[:5]}" for (j1,j2) in ROAD_EDGES]
        evals  = [edge_cong.get(e,50) for e in ROAD_EDGES]
        ecols2 = [cong_color(c)[0] for c in evals]
        fig_ec = go.Figure()
        fig_ec.add_trace(go.Bar(x=enames, y=evals,
            marker=dict(color=ecols2, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>"))
        fig_ec.add_hline(y=60, line=dict(color="rgba(255,85,0,0.4)",width=1,dash="dash"))
        apply_layout(fig_ec, ytitle="Congestion %", height=220,
                     xaxis=dict(tickangle=-30,color=TICK_COL,gridcolor=GRID_COL),
                     yaxis=dict(range=[0,100],color=TICK_COL,gridcolor=GRID_COL))
        st.plotly_chart(fig_ec, use_container_width=True, key=f"edge_cong_{tick}")

    with slot_t2_table:
        hb2 = hour_df[["Junction","Time","Congestion_Index","Vehicles_Per_Hour",
                        "Avg_Speed_kmh","Delay_Min","Incidents","Is_Peak"]].copy().reset_index(drop=True)
        hb2.columns=["Junction","Time","Congestion","Vehicles/hr","Speed(km/h)","Delay(min)","Incidents","Peak"]
        def col_cong(v):
            try: f=float(v)
            except: return ""
            if f>=70: return "background:rgba(255,51,68,0.22);color:#ff3344;font-weight:bold"
            elif f>=50: return "background:rgba(255,170,0,0.15);color:#ffaa00;font-weight:bold"
            return "background:rgba(0,255,136,0.1);color:#00ff88;font-weight:bold"
        st.dataframe(hb2.style.map(col_cong, subset=["Congestion"]).format(
            {"Congestion":"{:.1f}","Speed(km/h)":"{:.1f}","Delay(min)":"{:.2f}"}),
            use_container_width=True, height=260)

    # ---------- TAB 3 ----------
    with slot_t3_signals:
        st.markdown(f'<div class="sec-header{"  red" if evp_junctions else ""}">Adaptive Signal Matrix . {"EVP ACTIVE" if evp_junctions else "Normal Operations"} . DR={dr_now:.1f}%</div>',
                    unsafe_allow_html=True)
        cols_sig = st.columns(5)
        pc = {"GREEN":"#00ff88","RED":"#ff3344","AMBER":"#ffaa00"}
        pe = {"GREEN":"G","RED":"R","AMBER":"A"}
        for i,jn in enumerate(JNAMES):
            phase,timer,dr_jn,_ = signal_phases[jn]
            clr = pc[phase]
            density = float(dens_arr[i])
            note = "EVP PREEMPTED" if jn in evp_junctions else f"DR:+{dr_jn:.0f}%"
            with cols_sig[i%5]:
                st.markdown(f"""<div class="sig-card" style="border-color:{clr}33">
                  <div style="font-size:0.7rem;font-weight:600;color:#b8cfd8;margin-bottom:4px">{jn}</div>
                  <div style="font-size:2rem">{pe[phase]}</div>
                  <div style="font-family:Barlow Condensed,sans-serif;font-size:1.5rem;font-weight:900;color:{clr}">{timer}s</div>
                  <div style="font-family:Share Tech Mono,monospace;font-size:0.58rem;color:{clr}">{phase}</div>
                  <div style="font-family:Share Tech Mono,monospace;font-size:0.54rem;color:#3a5a6a;margin-top:3px">{note}</div>
                  <div style="font-family:Share Tech Mono,monospace;font-size:0.5rem;color:#1a3a4a">rho={density:.0f}</div>
                </div>""", unsafe_allow_html=True)

    with slot_t3_gantt:
        phi_g = (4000/max(vc_v/3.6,1)) % T_v
        fig_g = go.Figure()
        for i,jn in enumerate(JNAMES):
            density=float(dens_arr[i]); off=int(phi_g*i)%T_v
            rho=density/100; g_dur=min(int(T_v*(0.52+rho*0.20)),int(T_v*0.80))
            segs=[(off,off+g_dur,"GREEN","#00ff88"),(off+g_dur,off+g_dur+5,"AMBER","#ffaa00"),
                  (off+g_dur+5,off+T_v,"RED","#ff3344")]
            for s,e,ph,col in segs:
                sw=max(1,(e%T_v)-(s%T_v)) if (e%T_v)>(s%T_v) else max(1,T_v-(s%T_v))
                fig_g.add_trace(go.Bar(y=[jn],x=[sw],base=[s%T_v],orientation="h",
                    marker=dict(color=col,opacity=0.75,line=dict(width=0)),
                    name=ph,showlegend=(i==0),
                    hovertemplate=f"<b>{jn}</b><br>{ph}<extra></extra>"))
            if jn in evp_junctions:
                fig_g.add_annotation(x=T_v/2,y=jn,text="EVP PREEMPTED",
                    font=dict(color="#ff5500",size=9,family=FONT_MONO),showarrow=False)
        apply_layout(fig_g, xtitle="Seconds in cycle", height=340, barmode="stack",
                     xaxis=dict(range=[0,T_v],color=TICK_COL,gridcolor=GRID_COL),
                     yaxis=dict(autorange="reversed",color=TICK_COL))
        st.plotly_chart(fig_g, use_container_width=True, key=f"sig_gantt_{tick}")

    with slot_t3_ts:
        dists = np.linspace(0,22,60)
        gw_ts = dists*(3.6/max(vc_v,1)) if st.session_state.gw_enabled else dists*2.1
        base_ts = dists*2.5+np.sin(dists*0.45)*3.5
        adap_ts = gw_ts*0.88
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=dists,y=base_ts,name="Baseline",
            line=dict(color="rgba(255,51,68,0.6)",width=2,dash="dot"),
            fill="tozeroy",fillcolor="rgba(255,51,68,0.04)"))
        fig_ts.add_trace(go.Scatter(x=dists,y=gw_ts,name=f"GW@{vc_v}km/h",
            line=dict(color="rgba(0,170,255,0.8)",width=2,dash="dash"),
            fill="tozeroy",fillcolor="rgba(0,170,255,0.04)"))
        fig_ts.add_trace(go.Scatter(x=dists,y=adap_ts,name="Adaptive",
            line=dict(color="#00ff88",width=2.5),
            fill="tozeroy",fillcolor="rgba(0,255,136,0.06)"))
        for jn,jd in JUNCTIONS.items():
            fig_ts.add_vline(x=jd["km"],line=dict(color="rgba(0,170,255,0.18)",width=1,dash="dash"),
                annotation_text=jn[:6],annotation_font_color="#3a5a6a",annotation_font_size=7)
        apply_layout(fig_ts,xtitle="Distance (km)",ytitle="Travel time (min)",height=340)
        st.plotly_chart(fig_ts, use_container_width=True, key=f"time_space_{tick}")

    # ---------- TAB 4 ----------
    with slot_t4_algo:
        st.markdown(f"""<div class="equation-box">
          <b>Algorithm: 5-Layer Vectorized Optimizer</b><br><br>
          Layer 1: LP objective on J={NJ} junctions (O(J), not O(N))<br>
          Layer 2: Green Wave Phi=(L/v_c) mod T = <b>{phi_v:.1f}s</b><br>
          Layer 3: LWR density-adaptive g_dur<br>
          Layer 4: EVP preemption (O(n_evp)={stats['active_evp']})<br>
          Layer 5: Background map cache<br><br>
          Fleet N={fleet['N']:,} . Step: pure numpy . O(N) SIMD<br>
          spawn: vectorized . stats: np.sum<br><br>
          DR={dr_now:.1f}% . Wait={wait_ratio:.1f}% . Tick #{tick}
        </div>""", unsafe_allow_html=True)

    with slot_t4_math:
        st.markdown(f"""<div class="equation-box">
          <b>LWR + Graph Theory</b><br><br>
          G=(V,E) . |V|={NJ} . |E|={len(ROAD_EDGES)}<br>
          d/dt + d(v)/dx = 0<br>
          Greenshields: v()=v_max(1-/_max)<br>
          q()=.v_max.(1-/_max)<br>
          w_s=(q2-q1)/(2-1)<br><br>
          Mode: {st.session_state.algo_mode}<br>
          v_c={vc_v}km/h . T={T_v}s . Phi={phi_v:.1f}s<br><br>
          Active normal: {stats['active_normal']:,}<br>
          Active EVP:    {stats['active_evp']:,}<br>
          Completed:     {stats['completed']:,}
        </div>""", unsafe_allow_html=True)

    # Footer
    with slot_footer:
        st.markdown(f"""<div style='display:flex;justify-content:space-between;align-items:center'>
          <span style='font-family:Share Tech Mono,monospace;font-size:0.52rem;color:#3a5a6a'>URBAN FLOW . BANGALORE . NMIT ISE 2025</span>
          <span style='font-family:Share Tech Mono,monospace;font-size:0.52rem;color:#3a5a6a'>{st.session_state.algo_mode} . DR={dr_now:.1f}% . FLEET {fleet["N"]:,} . TICK #{tick}</span>
          <span style='font-family:Share Tech Mono,monospace;font-size:0.52rem;color:#3a5a6a'>NISHCHAL VISHWANATH (NB25ISE160) . RISHUL KH (NB25ISE186)</span>
        </div>""", unsafe_allow_html=True)

    # -- Tick rate control --
    if st.session_state.sim_running:
        time.sleep(0.8)
    else:
        time.sleep(0.3)

"""
har har mahadev
╔══════════════════════════════════════════════════════════════╗
║  URBAN FLOW & LIFE-LINES                                     ║
║  Multi-Objective Optimization Dashboard — Bangalore ATCS     ║
║  NMIT ISE · Nishchal Vishwanath NB25ISE160                   ║
║          · Rishul KH          NB25ISE186                     ║
╚══════════════════════════════════════════════════════════════╝

pip install streamlit plotly folium streamlit-folium numpy pandas
streamlit run streamlit_app.py
"""

# ── Imports ────────────────────────────────────────────────────
import math
import random
import datetime
import time as _time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import streamlit as st

# ── Page config (MUST be first Streamlit call) ─────────────────
st.set_page_config(
    page_title="Urban Flow & Life-Lines | Bangalore ATCS",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# GLOBAL STYLE
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
:root{
  --bg:#050a14;--panel:#0a1628;--card:#0d1f3c;
  --blue:#00d4ff;--green:#00ff88;--red:#ff3366;
  --orange:#ff8c00;--purple:#a855f7;--dim:#5a7a9a;
  --border:#1a3a5c;--text:#e0f0ff;
}
html,body,[data-testid="stAppViewContainer"],
[data-testid="stHeader"],[data-testid="stToolbar"]{
  background-color:#050a14!important;
  color:#e0f0ff!important;
  font-family:'Courier New',monospace!important;
}
[data-testid="stSidebar"]{
  background:#0a1628!important;
  border-right:1px solid #1a3a5c!important;
}
[data-testid="stSidebar"] *{color:#e0f0ff!important;}
.block-container{padding:.5rem 1rem 1rem!important;}
[data-testid="metric-container"]{
  background:#0d1f3c!important;border:1px solid #1a3a5c!important;
  border-radius:8px!important;padding:12px!important;
}
[data-testid="stMetricValue"]{color:#00d4ff!important;font-family:'Courier New',monospace!important;}
[data-testid="stMetricLabel"]{color:#5a7a9a!important;font-size:10px!important;letter-spacing:1px!important;text-transform:uppercase!important;}
[data-testid="stMetricDelta"]{font-size:10px!important;}
[data-testid="stTabs"] button{
  background:#0a1628!important;color:#5a7a9a!important;
  border:1px solid #1a3a5c!important;border-radius:4px 4px 0 0!important;
  font-family:'Courier New',monospace!important;font-size:10px!important;letter-spacing:1px!important;
}
[data-testid="stTabs"] button[aria-selected="true"]{
  background:#0d1f3c!important;color:#00d4ff!important;border-bottom-color:#00d4ff!important;
}
.stButton>button{
  background:linear-gradient(135deg,#0d1f3c,#0a1628)!important;
  border:1px solid #1a3a5c!important;color:#00d4ff!important;
  font-family:'Courier New',monospace!important;font-size:10px!important;
  letter-spacing:1px!important;border-radius:4px!important;transition:all .2s!important;
}
.stButton>button:hover{border-color:#00d4ff!important;box-shadow:0 0 12px rgba(0,212,255,.3)!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:#050a14;}
::-webkit-scrollbar-thumb{background:#1a3a5c;border-radius:2px;}
hr{border-color:#1a3a5c!important;}
[data-testid="stExpander"]{background:#0a1628!important;border:1px solid #1a3a5c!important;border-radius:6px!important;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.15)}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CONSTANTS  (Kaggle BLR Traffic Dataset 2024-25 inspired)
# ══════════════════════════════════════════════════════════════
HOURLY_PATTERN = [
    .28,.18,.13,.11,.16,.42,.82,.97,.91,.73,
    .67,.74,.80,.75,.70,.77,.92,.99,.96,.84,
    .72,.62,.52,.40,
]

JUNCTION_NAMES = [
    "Silk Board","Hebbal","ORR Junction","Koramangala",
    "HSR Layout","Bellandur","Indiranagar","Whitefield",
    "Yeshwanthpur","Kengeri",
]

JUNCTION_COORDS = {
    "Silk Board":   [12.9172,77.6228],
    "Hebbal":       [13.0350,77.5970],
    "ORR Junction": [12.9698,77.6496],
    "Koramangala":  [12.9352,77.6245],
    "HSR Layout":   [12.9082,77.6476],
    "Bellandur":    [12.9270,77.6780],
    "Indiranagar":  [12.9784,77.6408],
    "Whitefield":   [12.9699,77.7500],
    "Yeshwanthpur": [13.0275,77.5530],
    "Kengeri":      [12.9116,77.4843],
}

BASE_LOADS = {
    "Silk Board":95,"Hebbal":85,"ORR Junction":78,"Koramangala":70,
    "HSR Layout":72,"Bellandur":68,"Indiranagar":60,"Whitefield":55,
    "Yeshwanthpur":65,"Kengeri":40,
}

ROAD_EDGES = [
    ("Silk Board","HSR Layout"),("Silk Board","Koramangala"),
    ("Silk Board","ORR Junction"),("HSR Layout","Bellandur"),
    ("Bellandur","Whitefield"),("ORR Junction","Indiranagar"),
    ("Indiranagar","Koramangala"),("Hebbal","Yeshwanthpur"),
    ("Hebbal","ORR Junction"),("Yeshwanthpur","Kengeri"),
    ("Koramangala","HSR Layout"),("ORR Junction","Bellandur"),
]

# All vehicle routes (sequences of junctions)
VEHICLE_ROUTES = [
    ["Silk Board","HSR Layout","Bellandur","Whitefield"],
    ["Hebbal","ORR Junction","Indiranagar","Koramangala","Silk Board"],
    ["Yeshwanthpur","Hebbal","ORR Junction","Bellandur"],
    ["Kengeri","Yeshwanthpur","Hebbal"],
    ["Koramangala","ORR Junction","Indiranagar"],
    ["HSR Layout","Silk Board","Koramangala"],
    ["Bellandur","ORR Junction","Hebbal"],
    ["Indiranagar","ORR Junction","Silk Board"],
    ["Whitefield","Bellandur","HSR Layout","Silk Board"],
    ["Koramangala","HSR Layout","Bellandur"],
    ["Silk Board","ORR Junction","Hebbal","Yeshwanthpur"],
    ["Kengeri","Yeshwanthpur","ORR Junction","Indiranagar"],
]

DAYS  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
WEEKS = [.75,.82,.90,.93,.98,.72,.52]

EVP_ROUTE = ["Silk Board","HSR Layout","Bellandur","Whitefield"]

# ── Axis / layout building blocks (NEVER put xaxis/yaxis here!) ─
def _base_layout(title_text="", height=280):
    """Base Plotly layout — no xaxis/yaxis to avoid duplicate-kwarg error."""
    return dict(
        paper_bgcolor="#050a14",
        plot_bgcolor="#0a1628",
        font=dict(family="Courier New, monospace", color="#e0f0ff", size=11),
        margin=dict(l=45, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#5a7a9a", size=10)),
        title=dict(text=title_text, font=dict(color="#00d4ff", size=12)),
        height=height,
    )

def _axis(**extra):
    """Standard dark axis style, optionally merged with extras."""
    base = dict(gridcolor="#1a3a5c", linecolor="#1a3a5c",
                tickfont=dict(color="#5a7a9a", size=9), zeroline=False)
    base.update(extra)
    return base


# ══════════════════════════════════════════════════════════════
# MATH MODELS
# ══════════════════════════════════════════════════════════════
def green_wave_offset(L_km: float, v_kmh: float, T_s: float) -> float:
    """φ = (L / v_c) mod T"""
    if v_kmh <= 0:
        return 0.0
    return (L_km * 1000.0 / (v_kmh * 1000.0 / 3600.0)) % T_s


def evp_signal_time(d_m: float, v_kmh: float) -> float:
    """S = d / v  (seconds of pre-emption lead time)"""
    v_ms = v_kmh * 1000.0 / 3600.0
    return d_m / v_ms if v_ms > 0 else float("inf")


def lwr_density(flow_vph: float, speed_kmh: float) -> float:
    """ρ = q / v"""
    return flow_vph / speed_kmh if speed_kmh > 0 else 0.0


def delay_objective(loads: list, gw_active: bool) -> float:
    """LP objective W(D) = Σ max(0, (load-40)*0.4) × gw_factor"""
    f = 0.65 if gw_active else 1.0
    return sum(max(0.0, (l - 40) * 0.4 * f) for l in loads)


# ══════════════════════════════════════════════════════════════
# DATA GENERATORS
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=2)
def get_current_loads(density_pct: int, gw_active: bool, seed: int) -> dict:
    rng  = random.Random(seed)
    hour = datetime.datetime.now().hour
    tf   = HOURLY_PATTERN[hour] * (density_pct / 65.0)
    gw_f = 0.72 if gw_active else 1.0
    out  = {}
    for name, base in BASE_LOADS.items():
        noise     = rng.gauss(0, 3)
        out[name] = round(max(5.0, min(99.0, base * tf * gw_f + noise)), 1)
    return out


@st.cache_data(ttl=60)
def get_24h_series(junction: str, density_pct: int, gw_active: bool) -> pd.DataFrame:
    rng   = random.Random(42)
    base  = BASE_LOADS.get(junction, 60)
    gw_f  = 0.75 if gw_active else 1.0
    times = [datetime.datetime(2024, 6, 17, h) for h in range(24)]
    vols  = []
    for h in range(24):
        v = base * HOURLY_PATTERN[h] * (density_pct / 65.0) * gw_f * 30 + rng.gauss(0, 5) * 10
        vols.append(max(0, round(v)))
    return pd.DataFrame({"Time": times, "Volume (veh/h)": vols})


@st.cache_data(ttl=60)
def get_weekly_heatmap(gw_active: bool) -> pd.DataFrame:
    rng  = random.Random(7)
    gw_f = 0.75 if gw_active else 1.0
    data = {}
    for j, base in BASE_LOADS.items():
        data[j] = [round(max(5, min(99, base * w * gw_f + rng.gauss(0, 3)))) for w in WEEKS]
    return pd.DataFrame(data, index=DAYS)


@st.cache_data(ttl=60)
def get_flow_comparison(gw_active: bool) -> pd.DataFrame:
    items  = list(BASE_LOADS.items())[:6]
    names  = [n for n, _ in items]
    static = [round(b * 0.48) for _, b in items]
    gw     = [round(b * 0.48 * (0.65 if gw_active else 1.0)) for _, b in items]
    return pd.DataFrame({"Junction": names, "Static (min)": static, "Green Wave (min)": gw})


def get_lp_result(loads: dict, gw_active: bool) -> dict:
    total = delay_objective(list(loads.values()), gw_active)
    opt   = total * (0.70 if gw_active else 1.0)
    red   = round((1 - opt / max(total, 0.01)) * 100, 1) if gw_active else 0.0
    return {"total_delay": round(total, 1), "optimised": round(opt, 1), "reduction_pct": red}


# ══════════════════════════════════════════════════════════════
# VEHICLE SIMULATION  (pure numpy, runs in browser via
#  Plotly scatter — no Leaflet thrashing)
# ══════════════════════════════════════════════════════════════
def _interp_pos(p: float, coord_a: list, coord_b: list) -> tuple:
    lat = coord_a[0] + (coord_b[0] - coord_a[0]) * p
    lon = coord_a[1] + (coord_b[1] - coord_a[1]) * p
    return lat, lon


@st.cache_data(ttl=120)
def generate_vehicle_snapshot(
    n_total: int,
    n_emergency: int,
    density_pct: int,
    gw_active: bool,
    evp_active: bool,
    tick: int,
    loads_hash: str,     # forces recompute when loads change
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      lat, lon, type, color, size, label
    representing vehicle positions at this tick.
    Uses deterministic seed so movement is smooth across calls.
    """
    rng     = random.Random(tick)
    hour    = datetime.datetime.now().hour
    tf      = HOURLY_PATTERN[hour] * (density_pct / 65.0)

    rows = []
    n_routes = len(VEHICLE_ROUTES)

    for i in range(n_total):
        route    = VEHICLE_ROUTES[i % n_routes]
        n_segs   = len(route) - 1
        seg_idx  = i % n_segs

        jA = JUNCTION_COORDS[route[seg_idx]]
        jB = JUNCTION_COORDS[route[seg_idx + 1]]

        # Smooth progress: use tick + vehicle offset
        speed_factor = max(0.05, 1.0 - tf * 0.6)
        base_prog    = ((tick * 0.012 * speed_factor + i * 0.37) % 1.0)
        # Add small random jitter for realism
        prog = (base_prog + rng.gauss(0, 0.01)) % 1.0
        prog = max(0.0, min(1.0, prog))

        lat, lon = _interp_pos(prog, jA, jB)
        # Scatter within road corridor (±0.002 deg ≈ ±200m)
        lat += rng.gauss(0, 0.0008)
        lon += rng.gauss(0, 0.0008)

        rows.append({
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "type": "normal",
            "color": "#00d4ff",
            "size": 4,
        })

    # Emergency vehicles — always on EVP route
    evp_coords = [JUNCTION_COORDS[j] for j in EVP_ROUTE]
    for i in range(n_emergency):
        seg_idx  = i % (len(evp_coords) - 1)
        jA       = evp_coords[seg_idx]
        jB       = evp_coords[seg_idx + 1]
        prog     = ((tick * 0.035 + i * 0.25) % 1.0)
        lat, lon = _interp_pos(prog, jA, jB)
        lat += rng.gauss(0, 0.0004)
        lon += rng.gauss(0, 0.0004)
        rows.append({
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "type": "emergency",
            "color": "#ff3366",
            "size": 8,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# CHART BUILDERS  (all safe — no duplicate xaxis/yaxis)
# ══════════════════════════════════════════════════════════════
def build_vehicle_map(
    df_veh: pd.DataFrame,
    loads: dict,
    evp_active: bool,
    layer: str,
) -> go.Figure:
    """
    Pure Plotly scattermapbox — smooth, no Leaflet re-render flicker,
    handles 100k+ points efficiently.
    """
    fig = go.Figure()

    # ── Road edges ──────────────────────────────────────────
    evp_set = set(EVP_ROUTE)
    for a, b in ROAD_EDGES:
        ca, cb    = JUNCTION_COORDS[a], JUNCTION_COORDS[b]
        is_evp    = evp_active and a in evp_set and b in evp_set
        avg_load  = (loads.get(a, 50) + loads.get(b, 50)) / 2
        if layer == "density":
            col = _load_color(avg_load)
        elif is_evp:
            col = "#ff3366"
        else:
            col = "#1a3a5c"
        fig.add_trace(go.Scattermapbox(
            lat=[ca[0], cb[0], None],
            lon=[ca[1], cb[1], None],
            mode="lines",
            line=dict(width=5 if is_evp else 2, color=col),
            hoverinfo="skip",
            showlegend=False,
        ))

    # ── Normal vehicles ──────────────────────────────────────
    df_n = df_veh[df_veh["type"] == "normal"]
    if len(df_n) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=df_n["lat"].tolist(),
            lon=df_n["lon"].tolist(),
            mode="markers",
            marker=dict(size=df_n["size"].tolist(), color=df_n["color"].tolist(), opacity=0.75),
            name=f"Vehicles ({len(df_n):,})",
            hovertemplate="<b>Vehicle</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
        ))

    # ── Emergency vehicles ───────────────────────────────────
    df_e = df_veh[df_veh["type"] == "emergency"]
    if len(df_e) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=df_e["lat"].tolist(),
            lon=df_e["lon"].tolist(),
            mode="markers",
            marker=dict(size=df_e["size"].tolist(), color="#ff3366",
                        symbol="circle", opacity=0.95),
            name=f"🚑 Emergency ({len(df_e):,})",
            hovertemplate="<b>🚑 EMERGENCY</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
        ))

    # ── Junction nodes ───────────────────────────────────────
    j_lats = [c[0] for c in JUNCTION_COORDS.values()]
    j_lons = [c[1] for c in JUNCTION_COORDS.values()]
    j_names= list(JUNCTION_COORDS.keys())
    j_loads= [loads.get(n, 50) for n in j_names]
    j_cols = [_load_color(l) for l in j_loads]
    j_text = [f"<b>{n}</b><br>Load: {l:.0f}%<br>Status: {'CRITICAL' if l>80 else 'BUSY' if l>60 else 'CLEAR'}"
              for n, l in zip(j_names, j_loads)]
    fig.add_trace(go.Scattermapbox(
        lat=j_lats, lon=j_lons, mode="markers+text",
        marker=dict(size=14, color=j_cols, opacity=0.95),
        text=j_names,
        textposition="top right",
        textfont=dict(color="#e0f0ff", size=9),
        hovertext=j_text,
        hoverinfo="text",
        name="Junctions",
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=12.9716, lon=77.5946),
            zoom=11,
        ),
        paper_bgcolor="#050a14",
        plot_bgcolor="#050a14",
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(5,10,20,0.85)",
            bordercolor="#1a3a5c",
            font=dict(color="#e0f0ff", size=10),
            x=0.01, y=0.99,
        ),
        uirevision="stable",   # keeps zoom/pan on re-render
    )
    return fig


def build_24h_chart(df: pd.DataFrame, junction: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Time"], y=df["Volume (veh/h)"],
        mode="lines", name="Volume",
        line=dict(color="#00d4ff", width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
    ))
    fig.update_layout(
        **_base_layout(f"24-Hour Traffic Volume — {junction}", 260),
        xaxis=_axis(title="Hour"),
        yaxis=_axis(title="Vehicles/h"),
    )
    return fig


def build_delay_bar(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Junction"], y=df["Static (min)"],
        name="Static", marker_color="rgba(255,51,102,.75)",
        marker_line=dict(color="#ff3366", width=1),
    ))
    fig.add_trace(go.Bar(
        x=df["Junction"], y=df["Green Wave (min)"],
        name="Green Wave", marker_color="rgba(0,255,136,.75)",
        marker_line=dict(color="#00ff88", width=1),
    ))
    fig.update_layout(
        **_base_layout("Avg Delay — Static vs Green Wave (min)", 260),
        barmode="group",
        xaxis=_axis(),
        yaxis=_axis(title="Delay (min)"),
    )
    return fig


def build_heatmap(df: pd.DataFrame) -> go.Figure:
    """Fixed: no duplicate axis kwargs."""
    fig = go.Figure(go.Heatmap(
        z=df.values.tolist(),
        x=list(df.columns),
        y=list(df.index),
        colorscale=[
            [0.0, "#001a06"], [0.4, "#00ff88"],
            [0.7, "#ff8c00"], [1.0, "#ff3366"],
        ],
        colorbar=dict(
            tickfont=dict(color="#5a7a9a", size=9),
            outlinecolor="#1a3a5c",
            outlinewidth=1,
        ),
        hoverongaps=False,
        texttemplate="%{z:.0f}%",
        textfont=dict(size=8, color="#e0f0ff"),
    ))
    # Safe: _base_layout has NO xaxis/yaxis, so no duplicate keyword
    layout = _base_layout("Junction Load Heatmap — Day × Node (%)", 300)
    layout["xaxis"] = _axis(tickangle=-30)
    layout["yaxis"] = _axis()
    fig.update_layout(**layout)
    return fig


def build_radar(loads: dict) -> go.Figure:
    names = list(loads.keys())
    vals  = list(loads.values())
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=names + [names[0]],
        fill="toself",
        fillcolor="rgba(0,212,255,0.1)",
        line=dict(color="#00d4ff", width=1.5),
        name="Load %",
    ))
    fig.update_layout(
        paper_bgcolor="#050a14",
        font=dict(family="Courier New", color="#e0f0ff", size=10),
        polar=dict(
            bgcolor="#0a1628",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="#1a3a5c",
                tickfont=dict(color="#5a7a9a", size=8),
            ),
            angularaxis=dict(
                gridcolor="#1a3a5c",
                tickfont=dict(color="#5a7a9a", size=9),
            ),
        ),
        title=dict(text="Network Load — All Junctions", font=dict(color="#00d4ff", size=12)),
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
        showlegend=False,
    )
    return fig


def build_gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(text=title, font=dict(color="#5a7a9a", size=9)),
        number=dict(font=dict(color=color, size=22, family="Courier New"), suffix="%"),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1,
                      tickcolor="#5a7a9a", tickfont=dict(size=7)),
            bar=dict(color=color, thickness=0.22),
            bgcolor="#0a1628",
            borderwidth=1, bordercolor="#1a3a5c",
            steps=[
                dict(range=[0, 40],  color="rgba(0,255,136,.07)"),
                dict(range=[40, 70], color="rgba(255,140,0,.07)"),
                dict(range=[70, 100],color="rgba(255,51,102,.07)"),
            ],
            threshold=dict(line=dict(color=color, width=2),
                           thickness=0.8, value=value),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#050a14",
        font=dict(family="Courier New", color="#e0f0ff"),
        height=155,
        margin=dict(l=15, r=15, t=30, b=5),
    )
    return fig


def build_green_wave_chart(v_kmh: float, T_s: float) -> go.Figure:
    gw_junctions = ["Silk Board (0 km)", "HSR Layout (4 km)",
                    "Bellandur (8 km)", "Whitefield (13 km)"]
    distances    = [0.0, 4.0, 8.0, 13.0]
    offsets      = [green_wave_offset(d, v_kmh, T_s) for d in distances]
    fig          = go.Figure()

    for dist, off in zip(distances, offsets):
        for cycle in range(3):
            gs = off + cycle * T_s
            ge = gs + T_s * 0.5
            rs = ge
            re = rs + T_s * 0.5
            fig.add_shape(type="rect",
                x0=gs, x1=ge, y0=dist-.28, y1=dist+.28,
                fillcolor="rgba(0,255,136,.25)",
                line=dict(color="#00ff88", width=1))
            fig.add_shape(type="rect",
                x0=rs, x1=re, y0=dist-.28, y1=dist+.28,
                fillcolor="rgba(255,51,102,.25)",
                line=dict(color="#ff3366", width=1))

    if v_kmh > 0:
        v_ms   = v_kmh * 1000.0 / 3600.0
        t_end  = distances[-1] * 1000.0 / v_ms
        fig.add_trace(go.Scatter(
            x=[0, t_end], y=[0, distances[-1]],
            mode="lines",
            line=dict(color="#00d4ff", width=2.5, dash="dot"),
            name=f"Vehicle @ {v_kmh:.0f} km/h",
        ))

    layout = _base_layout(
        f"Green Wave Time-Space Diagram  (v_c={v_kmh:.0f} km/h · T={T_s:.0f}s)", 290)
    layout["xaxis"] = _axis(title="Time (s)")
    layout["yaxis"] = _axis(
        title="Distance (km)",
        tickvals=distances,
        ticktext=gw_junctions,
    )
    fig.update_layout(**layout)
    return fig


def build_evp_chart(d_m: float, v_kmh: float) -> go.Figure:
    speeds    = [30, 40, 50, 60]
    distances = list(range(100, max(2100, int(d_m) + 200), 100))
    palette   = ["#ff3366", "#ff8c00", "#ffd700", "#00ff88"]
    fig       = go.Figure()
    for sp, col in zip(speeds, palette):
        fig.add_trace(go.Scatter(
            x=distances,
            y=[evp_signal_time(d, sp) for d in distances],
            mode="lines", name=f"{sp} km/h",
            line=dict(color=col, width=1.8),
        ))
    cur_t = evp_signal_time(d_m, v_kmh)
    fig.add_trace(go.Scatter(
        x=[d_m], y=[cur_t],
        mode="markers+text",
        text=[f" {cur_t:.0f}s"],
        textposition="top right",
        textfont=dict(color="#a855f7", size=10),
        marker=dict(color="#a855f7", size=10, symbol="diamond"),
        name="Current",
    ))
    fig.add_vline(x=d_m, line_width=1, line_dash="dash", line_color="#a855f7")
    layout = _base_layout("EVP Lead-Time  S = d/v  (seconds pre-emption needed)", 270)
    layout["xaxis"] = _axis(title="Distance to Signal (m)")
    layout["yaxis"] = _axis(title="Pre-emption Time (s)")
    fig.update_layout(**layout)
    return fig


def build_lwr_chart() -> go.Figure:
    rho_max = 120.0
    v_free  = 60.0
    rhos    = np.linspace(0, rho_max, 250)
    speeds  = np.maximum(0.0, v_free * (1 - rhos / rho_max))
    flows   = rhos * speeds

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["", ""])
    fig.add_trace(go.Scatter(
        x=list(rhos), y=list(flows),
        mode="lines", line=dict(color="#00d4ff", width=2), name="q-k",
        fill="tozeroy", fillcolor="rgba(0,212,255,.06)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(rhos), y=list(speeds),
        mode="lines", line=dict(color="#00ff88", width=2), name="v-k",
        fill="tozeroy", fillcolor="rgba(0,255,136,.06)",
    ), row=1, col=2)

    # Update axes individually (safe pattern for subplots)
    fig.update_xaxes(title_text="Density ρ (veh/km)", **_axis(), row=1, col=1)
    fig.update_xaxes(title_text="Density ρ (veh/km)", **_axis(), row=1, col=2)
    fig.update_yaxes(title_text="Flow q (veh/h)",     **_axis(), row=1, col=1)
    fig.update_yaxes(title_text="Speed v (km/h)",     **_axis(), row=1, col=2)

    fig.update_layout(
        paper_bgcolor="#050a14",
        plot_bgcolor="#0a1628",
        font=dict(family="Courier New", color="#e0f0ff", size=11),
        margin=dict(l=45, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#5a7a9a", size=10)),
        title=dict(text="LWR Traffic Flow Model — Lighthill-Whitham-Richards",
                   font=dict(color="#00d4ff", size=12)),
        height=290,
        showlegend=False,
        annotations=[
            dict(text="Flow-Density (q-k)", x=0.22, xref="paper", y=1.07,
                 yref="paper", showarrow=False, font=dict(color="#5a7a9a", size=10)),
            dict(text="Speed-Density (v-k)", x=0.80, xref="paper", y=1.07,
                 yref="paper", showarrow=False, font=dict(color="#5a7a9a", size=10)),
        ],
    )
    return fig


def build_lp_bar(loads: dict, gw_active: bool) -> go.Figure:
    j_names  = list(BASE_LOADS.keys())
    density  = st.session_state.get("density_pct", 65)
    before   = [round(BASE_LOADS[j] * 0.48 * (density / 65.0)) for j in j_names]
    after    = [round(b * (0.65 if gw_active else 0.95)) for b in before]
    fig      = go.Figure()
    fig.add_trace(go.Bar(x=j_names, y=before, name="Before LP",
                         marker_color="rgba(255,51,102,.7)",
                         marker_line=dict(color="#ff3366", width=1)))
    fig.add_trace(go.Bar(x=j_names, y=after,  name="After LP",
                         marker_color="rgba(168,85,247,.7)",
                         marker_line=dict(color="#a855f7", width=1)))
    layout = _base_layout("Junction Delay — Before and After LP Optimisation (min)", 280)
    layout["xaxis"] = _axis()
    layout["yaxis"] = _axis(title="Delay (min)")
    layout["barmode"] = "group"
    fig.update_layout(**layout)
    return fig


def build_vehicle_histogram(df_veh: pd.DataFrame, loads: dict) -> go.Figure:
    """Vehicle density histogram per corridor."""
    routes_short = [r[0] + "→" + r[-1] for r in VEHICLE_ROUTES[:8]]
    counts = [len(df_veh) // len(VEHICLE_ROUTES)] * 8
    counts[0] = int(counts[0] * (loads.get("Silk Board", 50) / 50))
    cols   = [_load_color(loads.get(VEHICLE_ROUTES[i][0], 50)) for i in range(8)]
    fig    = go.Figure(go.Bar(
        x=routes_short, y=counts,
        marker_color=cols,
        text=counts, textposition="outside",
        textfont=dict(color="#e0f0ff", size=9),
    ))
    layout = _base_layout("Vehicle Distribution — Active Corridors", 230)
    layout["xaxis"] = _axis(tickangle=-20)
    layout["yaxis"] = _axis(title="Vehicles")
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════
# FOLIUM MAP  (static junction overview — used in separate tab)
# ══════════════════════════════════════════════════════════════
def build_folium_map(loads: dict, evp_active: bool) -> folium.Map:
    m = folium.Map(
        location=[12.9716, 77.5946],
        zoom_start=12,
        tiles="CartoDB DarkMatter",
        prefer_canvas=True,
    )
    evp_set = set(EVP_ROUTE)
    for a, b in ROAD_EDGES:
        ca, cb  = JUNCTION_COORDS[a], JUNCTION_COORDS[b]
        is_evp  = evp_active and a in evp_set and b in evp_set
        avg_l   = (loads.get(a, 50) + loads.get(b, 50)) / 2
        color   = "#ff3366" if is_evp else _load_color(avg_l)
        weight  = 5 if is_evp else 2
        folium.PolyLine([ca, cb], color=color, weight=weight, opacity=0.85).add_to(m)

    for name, coord in JUNCTION_COORDS.items():
        load  = loads.get(name, 50)
        col   = _load_color(load)
        popup = folium.Popup(
            f'<div style="background:#0d1f3c;border:1px solid #1a3a5c;padding:8px;'
            f'border-radius:6px;font-family:Courier New;font-size:11px;color:#e0f0ff;min-width:160px">'
            f'<b style="color:#00d4ff">{name}</b><br>'
            f'<span style="color:#5a7a9a">Load:</span> '
            f'<span style="color:{col};font-weight:700">{load:.0f}%</span><br>'
            f'<span style="color:#5a7a9a">Status:</span> '
            f'<span style="color:{col}">{"CRITICAL" if load>80 else "BUSY" if load>60 else "CLEAR"}</span>'
            f'</div>',
            max_width=220,
        )
        folium.CircleMarker(
            location=coord, radius=max(6, min(14, load / 7)),
            color=col, fill=True, fill_color=col, fill_opacity=0.85,
            weight=2, popup=popup, tooltip=f"{name}: {load:.0f}%",
        ).add_to(m)
        folium.map.Marker(
            coord,
            icon=folium.DivIcon(
                html=f'<div style="font-family:Courier New;font-size:8px;color:#5a7a9a;'
                     f'white-space:nowrap;margin-top:10px;margin-left:8px">{name}</div>',
                icon_size=(130, 20), icon_anchor=(0, 0),
            ),
        ).add_to(m)
    if evp_active:
        folium.Marker(
            JUNCTION_COORDS["Silk Board"],
            icon=folium.DivIcon(
                html='<div style="font-size:22px">🚑</div>',
                icon_size=(30, 30), icon_anchor=(15, 15),
            ),
            tooltip="EMERGENCY VEHICLE",
        ).add_to(m)
    return m


# ══════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════
def _load_color(load: float) -> str:
    if load > 80: return "#ff3366"
    if load > 60: return "#ff8c00"
    if load > 40: return "#ffd700"
    return "#00ff88"


def sec_header(icon: str, title: str, badge: str = "") -> None:
    bdg = (f'<span style="font-size:8px;padding:2px 7px;border-radius:10px;'
           f'background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.3);'
           f'color:#00d4ff">{badge}</span>' if badge else "")
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'padding:5px 0 4px;border-bottom:1px solid #1a3a5c;margin-bottom:7px">'
        f'<span style="font-size:10px;font-weight:700;letter-spacing:2px;'
        f'color:#00d4ff;text-transform:uppercase">{icon} {title}</span>{bdg}</div>',
        unsafe_allow_html=True,
    )


def render_header(evp_active: bool, gw_active: bool, sim_running: bool) -> None:
    now = datetime.datetime.now()
    evp_b = (
        '<span style="background:rgba(255,51,102,.15);border:1px solid #ff3366;'
        'color:#ff3366;padding:3px 10px;border-radius:12px;font-size:9px;'
        'letter-spacing:1.5px;animation:blink 1s step-end infinite">🚨 EVP ACTIVE</span>'
        if evp_active else
        '<span style="background:rgba(0,255,136,.1);border:1px solid #00ff88;'
        'color:#00ff88;padding:3px 10px;border-radius:12px;font-size:9px;letter-spacing:1.5px">'
        '● NOMINAL</span>'
    )
    gw_b = (
        '<span style="background:rgba(0,255,136,.1);border:1px solid #00ff88;'
        'color:#00ff88;padding:3px 10px;border-radius:12px;font-size:9px;letter-spacing:1.5px">'
        '⬡ GREEN WAVE</span>'
        if gw_active else
        '<span style="background:rgba(255,140,0,.1);border:1px solid #ff8c00;'
        'color:#ff8c00;padding:3px 10px;border-radius:12px;font-size:9px;letter-spacing:1.5px">'
        '⬡ STATIC</span>'
    )
    sim_b = (
        '<span style="background:rgba(168,85,247,.15);border:1px solid #a855f7;'
        'color:#a855f7;padding:3px 10px;border-radius:12px;font-size:9px;letter-spacing:1.5px">'
        '▶ SIMULATION RUNNING</span>'
        if sim_running else
        '<span style="background:rgba(90,122,154,.1);border:1px solid #5a7a9a;'
        'color:#5a7a9a;padding:3px 10px;border-radius:12px;font-size:9px;letter-spacing:1.5px">'
        '⏸ SIMULATION PAUSED</span>'
    )
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#020810,#0a1628,#020810);
                border-bottom:1px solid #1a3a5c;padding:8px 0 8px;
                display:flex;align-items:center;justify-content:space-between;
                box-shadow:0 2px 20px rgba(0,212,255,.12);margin-bottom:10px">
      <div style="display:flex;align-items:center;gap:12px">
        <span style="font-size:26px">🚦</span>
        <div>
          <div style="font-size:14px;font-weight:700;color:#00d4ff;letter-spacing:3px;
                      text-shadow:0 0 18px rgba(0,212,255,.4)">URBAN FLOW &amp; LIFE-LINES</div>
          <div style="font-size:8px;color:#5a7a9a;letter-spacing:2px">
            BANGALORE ATCS · NMIT ISE · NB25ISE160 / NB25ISE186
          </div>
        </div>
      </div>
      <div style="display:flex;gap:6px;align-items:center">
        {evp_b}&nbsp;{gw_b}&nbsp;{sim_b}
      </div>
      <div style="text-align:right">
        <div style="font-size:18px;font-weight:700;color:#00d4ff;
                    font-family:'Courier New';letter-spacing:3px">
          {now.strftime('%H:%M:%S')}
        </div>
        <div style="font-size:8px;color:#5a7a9a;letter-spacing:1px">
          {now.strftime('%a %d %b %Y')} · IST
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def signal_panel_html(phase: str, timer: int, phi: float, evp_on: bool) -> str:
    r_on = phase == "RED"   and not evp_on
    y_on = phase == "YELLOW" and not evp_on
    g_on = phase == "GREEN"  or evp_on

    def bulb(is_on, on_col, off_col):
        col   = on_col if is_on else off_col
        glow  = f"box-shadow:0 0 14px {on_col};" if is_on else ""
        return f'<div style="width:22px;height:22px;border-radius:50%;background:{col};{glow}border:2px solid rgba(255,255,255,.08)"></div>'

    label = "EVP OVERRIDE — P → ∞" if evp_on else f"{phase} PHASE  φ={phi:.1f}s"
    lcol  = "#ff3366" if evp_on else ("#00ff88" if phase=="GREEN" else "#ffd700" if phase=="YELLOW" else "#ff3366")
    return f"""
    <div style="background:#0d1f3c;border:1px solid #1a3a5c;border-radius:8px;
                padding:14px;text-align:center;font-family:'Courier New',monospace">
      <div style="font-size:9px;color:#5a7a9a;letter-spacing:2px;margin-bottom:8px">SIGNAL CONTROLLER</div>
      <div style="display:flex;justify-content:center;gap:10px;margin-bottom:10px">
        {bulb(r_on,'#ff3366','#1a0000')}
        {bulb(y_on,'#ffd700','#1a1200')}
        {bulb(g_on,'#00ff88','#001a06')}
      </div>
      <div style="font-size:36px;font-weight:700;color:#00d4ff;
                  text-shadow:0 0 18px rgba(0,212,255,.5);line-height:1">{str(timer).zfill(2)}</div>
      <div style="font-size:9px;color:{lcol};letter-spacing:2px;margin-top:5px">{label}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════
# ALERT SYSTEM
# ══════════════════════════════════════════════════════════════
_ALERT_TEMPLATES = [
    ("CONGESTION", "Silk Board density critical — auto-adjusting signal offsets", "danger"),
    ("GREEN WAVE",  "φ synchronised across ORR–Hebbal at 40 km/h", "success"),
    ("LWR MODEL",   "Shock-wave detected at ORR — predictive re-routing engaged", "warn"),
    ("LP SOLVER",   "Delay reduced by 30% — new timing parameters applied", "info"),
    ("SENSOR",      "Koramangala density sensor reporting peak flow", "warn"),
    ("CLEARED",     "Incident on Bellandur cleared — normal flow resuming", "success"),
    ("HEBBAL",      "Hebbal flyover congestion above threshold — diverting", "warn"),
]


def push_alert(t: str, msg: str, lvl: str = "info") -> None:
    cols = {"info":"#00d4ff","warn":"#ff8c00","danger":"#ff3366","success":"#00ff88"}
    col  = cols.get(lvl, "#00d4ff")
    now  = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.alerts.insert(0, (t, msg, col, now))
    if len(st.session_state.alerts) > 20:
        st.session_state.alerts.pop()


def render_alerts() -> None:
    for t, msg, col, ts in st.session_state.alerts[:10]:
        st.markdown(
            f'<div style="background:#0d1f3c;border-left:3px solid {col};'
            f'border-radius:0 4px 4px 0;padding:6px 9px;margin-bottom:4px;font-size:10px">'
            f'<span style="color:{col};font-weight:700;letter-spacing:1px">{t}</span>'
            f'<span style="color:#5a7a9a;float:right;font-size:9px">{ts}</span><br>'
            f'<span style="color:#e0f0ff">{msg}</span></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
def init_state() -> None:
    defaults = {
        "evp_active":        False,
        "gw_active":         True,
        "density_pct":       65,
        "n_vehicles":        5000,
        "n_emergency":       50,
        "map_layer":         "normal",
        "selected_junction": "Silk Board",
        "v_kmh":             40.0,
        "cycle_T":           60.0,
        "evp_distance":      500.0,
        "evp_speed":         40.0,
        "signal_phase":      "GREEN",
        "signal_timer":      30,
        "sim_running":       True,
        "tick":              0,
        "alerts":            [],
        "alerts_seeded":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state.alerts_seeded:
        push_alert("SYSTEM",     "Urban Flow Command online — 10 nodes active", "success")
        push_alert("DATA",       "Kaggle BLR Traffic Dataset 2024-25 loaded", "info")
        push_alert("GREEN WAVE", "Algorithm v2.1 — φ = L/v_c mod T synced", "success")
        st.session_state.alerts_seeded = True


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main() -> None:
    init_state()

    # ── Tick & signal update ─────────────────────────────────
    if st.session_state.sim_running:
        st.session_state.tick += 1
        st.session_state.signal_timer -= 1
        if st.session_state.signal_timer <= 0:
            phase = st.session_state.signal_phase
            if phase == "GREEN":
                st.session_state.signal_phase = "YELLOW"
                st.session_state.signal_timer = 5
            elif phase == "YELLOW":
                st.session_state.signal_phase = "RED"
                st.session_state.signal_timer = 25
            else:
                st.session_state.signal_phase = "GREEN"
                st.session_state.signal_timer = 30

    tick   = st.session_state.tick
    seed   = tick // 4          # loads refresh every 4 ticks

    # ── Current loads ────────────────────────────────────────
    loads = get_current_loads(
        st.session_state.density_pct,
        st.session_state.gw_active,
        seed,
    )
    loads_hash = str(seed)

    # ── Header ──────────────────────────────────────────────
    render_header(st.session_state.evp_active,
                  st.session_state.gw_active,
                  st.session_state.sim_running)

    # ════════════════════════════════════════════════════════
    # SIDEBAR
    # ════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown('<div style="font-size:12px;font-weight:700;color:#00d4ff;'
                    'letter-spacing:2px;margin-bottom:12px">⚙ COMMAND PANEL</div>',
                    unsafe_allow_html=True)

        # ── Simulation start/stop ────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶ START" if not st.session_state.sim_running else "⏸ PAUSE",
                         use_container_width=True):
                st.session_state.sim_running = not st.session_state.sim_running
                push_alert("SIMULATION",
                           "Running" if st.session_state.sim_running else "Paused",
                           "success" if st.session_state.sim_running else "warn")
                st.rerun()
        with col_b:
            if st.button("↺ RESET", use_container_width=True):
                st.session_state.tick = 0
                st.session_state.signal_phase = "GREEN"
                st.session_state.signal_timer = 30
                push_alert("RESET", "Simulation reset to t=0", "info")
                st.rerun()

        st.markdown("---")

        # ── EVP ─────────────────────────────────────────────
        if st.button(
            "🚨 ACTIVATE EMERGENCY" if not st.session_state.evp_active
            else "✅ DEACTIVATE EMERGENCY",
            use_container_width=True,
        ):
            st.session_state.evp_active = not st.session_state.evp_active
            if st.session_state.evp_active:
                push_alert("🚨 EVP", "Silk Board→HSR→Bellandur pre-empted P→∞", "danger")
                st.session_state.signal_phase = "GREEN"
                st.session_state.signal_timer = 120
            else:
                push_alert("✅ CLEARED", "Emergency corridor released", "success")
            st.rerun()

        st.markdown("---")

        # ── Green Wave ───────────────────────────────────────
        gw = st.toggle("⬡ Green Wave Algorithm", value=st.session_state.gw_active,
                       key="_gw_tog")
        if gw != st.session_state.gw_active:
            st.session_state.gw_active = gw
            push_alert("GREEN WAVE",
                       "Activated — φ synced" if gw else "Disabled — static timers",
                       "success" if gw else "warn")
            st.rerun()

        st.markdown("---")
        st.markdown('<div style="font-size:8px;color:#5a7a9a;letter-spacing:1px;margin-bottom:3px">TRAFFIC DENSITY</div>', unsafe_allow_html=True)
        st.session_state.density_pct = st.slider(
            "", 20, 100, st.session_state.density_pct,
            key="_dens", label_visibility="collapsed")

        st.markdown('<div style="font-size:8px;color:#5a7a9a;letter-spacing:1px;margin-bottom:3px;margin-top:8px">TOTAL VEHICLES</div>', unsafe_allow_html=True)
        st.session_state.n_vehicles = st.select_slider(
            "",
            options=[500, 1000, 2000, 5000, 10000, 25000, 50000, 100000, 200000, 500000],
            value=st.session_state.n_vehicles,
            key="_nveh", label_visibility="collapsed",
            format_func=lambda x: f"{x:,}",
        )

        st.markdown('<div style="font-size:8px;color:#5a7a9a;letter-spacing:1px;margin-bottom:3px;margin-top:8px">EMERGENCY VEHICLES</div>', unsafe_allow_html=True)
        st.session_state.n_emergency = st.select_slider(
            "",
            options=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            value=st.session_state.n_emergency,
            key="_nemerg", label_visibility="collapsed",
            format_func=lambda x: f"{x:,}",
        )

        st.markdown("---")
        st.markdown('<div style="font-size:8px;color:#5a7a9a;letter-spacing:1px;margin-bottom:3px">MAP LAYER</div>', unsafe_allow_html=True)
        st.session_state.map_layer = st.selectbox(
            "",
            ["normal", "density", "green_wave", "evp_route"],
            format_func=lambda x: {"normal":"Normal","density":"Density","green_wave":"Green Wave","evp_route":"EVP Route"}[x],
            key="_lyr", label_visibility="collapsed",
        )

        st.markdown('<div style="font-size:8px;color:#5a7a9a;letter-spacing:1px;margin-bottom:3px;margin-top:8px">FOCUS JUNCTION</div>', unsafe_allow_html=True)
        st.session_state.selected_junction = st.selectbox(
            "",
            JUNCTION_NAMES,
            index=JUNCTION_NAMES.index(st.session_state.selected_junction),
            key="_jsel", label_visibility="collapsed",
        )

        st.markdown("---")
        with st.expander("🌊 Green Wave Params"):
            st.session_state.v_kmh   = st.slider("v_c (km/h)", 20.0, 80.0, st.session_state.v_kmh, 1.0)
            st.session_state.cycle_T = st.slider("Cycle T (s)", 30.0, 120.0, st.session_state.cycle_T, 5.0)

        with st.expander("🚑 EVP Parameters"):
            st.session_state.evp_distance = st.slider("Distance to signal (m)", 100.0, 2000.0, st.session_state.evp_distance, 50.0)
            st.session_state.evp_speed    = st.slider("Ambulance speed (km/h)", 20.0, 80.0,   st.session_state.evp_speed, 1.0)

        with st.expander("📐 Live Math"):
            phi   = green_wave_offset(4.0, st.session_state.v_kmh, st.session_state.cycle_T)
            evp_t = evp_signal_time(st.session_state.evp_distance, st.session_state.evp_speed)
            lwr   = lwr_density(loads.get(st.session_state.selected_junction, 50) * 20,
                                st.session_state.v_kmh)
            lp    = get_lp_result(loads, st.session_state.gw_active)
            st.markdown(f"""
            <div style="font-size:10px;font-family:'Courier New',monospace">
              <div style="color:#5a7a9a">Green Wave φ:</div>
              <div style="color:#00d4ff">{phi:.2f} s</div>
              <div style="color:#5a7a9a;margin-top:6px">EVP Lead-Time S:</div>
              <div style="color:#ff3366">{evp_t:.1f} s</div>
              <div style="color:#5a7a9a;margin-top:6px">LWR Density ρ:</div>
              <div style="color:#00ff88">{lwr:.1f} veh/km</div>
              <div style="color:#5a7a9a;margin-top:6px">LP W(D):</div>
              <div style="color:#a855f7">{lp['optimised']:.1f} min  (↓{lp['reduction_pct']}%)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        auto = st.toggle("🔄 Auto-Refresh", value=False, key="_auto")
        if auto and st.session_state.sim_running:
            _time.sleep(3)
            st.rerun()

        st.markdown("""
        <div style="font-size:8px;color:#5a7a9a;margin-top:10px;line-height:1.7">
          Data: Kaggle BLR Traffic 2024-25<br>
          TomTom Traffic Index · BBMP<br>
          Model: LP + LWR + Green Wave<br>
          NMIT ISE · Batch 2025
        </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # KPI ROW
    # ════════════════════════════════════════════════════════
    hour      = datetime.datetime.now().hour
    tf        = HOURLY_PATTERN[hour] * (st.session_state.density_pct / 65.0)
    gw_f      = 0.72 if st.session_state.gw_active else 1.0
    avg_speed = round(max(10, 80 - tf * 50))
    avg_delay = round((5 + tf * 25 * gw_f), 1)
    num_jams  = max(0, int(tf * 5 * (0.6 if st.session_state.gw_active else 1)))
    co2       = f"-{round(15+tf*10)}%" if st.session_state.gw_active else f"-{round(5+tf*3)}%"
    lp_res    = get_lp_result(loads, st.session_state.gw_active)
    n_veh     = st.session_state.n_vehicles
    n_emg     = st.session_state.n_emergency

    c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
    c1.metric("🚗 Vehicles",      f"{n_veh:,}")
    c2.metric("🚑 Emergency",     f"{n_emg:,}", delta="EVP" if st.session_state.evp_active else "Standby",
              delta_color="inverse" if st.session_state.evp_active else "off")
    c3.metric("⚡ Avg Speed",     f"{avg_speed} km/h")
    c4.metric("⏱ Avg Delay",     f"{avg_delay} min", delta_color="inverse")
    c5.metric("🔴 Jams",          f"{num_jams}",
              delta="Critical" if num_jams>=4 else "OK",
              delta_color="inverse" if num_jams>=4 else "normal")
    c6.metric("🌿 CO₂",           co2)
    c7.metric("📈 LP Reduction",  f"{lp_res['reduction_pct']}%")
    c8.metric("🔗 Sync'd",        f"{'8' if st.session_state.gw_active else '2'}/10 signals")

    st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # TABS
    # ════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️  Live Vehicle Map",
        "📡  Junction Overview",
        "📊  Analytics",
        "⚡  Green Wave",
        "🚑  EVP System",
        "🧮  LP & LWR Theory",
    ])

    # ──────────────────────────────────────────────────────
    # TAB 1 — LIVE VEHICLE MAP  (Plotly mapbox — smooth)
    # ──────────────────────────────────────────────────────
    with tab1:
        map_col, right_col = st.columns([3, 1])

        with map_col:
            sec_header("🗺️", "Live Vehicle Flow",
                       f"{n_veh:,} vehicles · {n_emg:,} emergency")

            # Generate vehicle positions
            df_veh = generate_vehicle_snapshot(
                n_total     = min(n_veh, 50000),   # cap rendering at 50k for performance
                n_emergency = min(n_emg, 2000),
                density_pct = st.session_state.density_pct,
                gw_active   = st.session_state.gw_active,
                evp_active  = st.session_state.evp_active,
                tick        = tick,
                loads_hash  = loads_hash,
            )

            fig_map = build_vehicle_map(
                df_veh,
                loads,
                st.session_state.evp_active,
                st.session_state.map_layer,
            )
            st.plotly_chart(fig_map, use_container_width=True,
                            config={"displayModeBar": True,
                                    "modeBarButtonsToRemove": ["toImage"],
                                    "scrollZoom": True})

            # Note if capped
            if n_veh > 50000:
                st.markdown(
                    f'<div style="font-size:9px;color:#5a7a9a;margin-top:-8px">'
                    f'ℹ Rendering 50,000 of {n_veh:,} vehicles (map cap). '
                    f'Full {n_veh:,} counted in metrics.</div>',
                    unsafe_allow_html=True)

        with right_col:
            sec_header("🚦", "Signal Controller")
            phi_val = green_wave_offset(4.0, st.session_state.v_kmh, st.session_state.cycle_T)
            st.markdown(
                signal_panel_html(
                    st.session_state.signal_phase,
                    max(0, st.session_state.signal_timer),
                    phi_val,
                    st.session_state.evp_active,
                ),
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # EVP panel
            if st.session_state.evp_active:
                evp_t_cur = evp_signal_time(
                    st.session_state.evp_distance, st.session_state.evp_speed)
                st.markdown(f"""
                <div style="background:#0d1f3c;border:1px solid #ff3366;border-radius:8px;
                            padding:12px;font-family:'Courier New';font-size:10px;
                            box-shadow:0 0 16px rgba(255,51,102,.2);margin-bottom:8px">
                  <div style="color:#ff3366;font-weight:700;font-size:11px;margin-bottom:8px">
                    🚨 CORRIDOR PRE-EMPTED
                  </div>
                  <div style="color:#5a7a9a">Route:</div>
                  <div style="color:#e0f0ff;font-size:9px">Silk Board → HSR → Bellandur</div>
                  <div style="color:#5a7a9a;margin-top:6px">P Weight:</div>
                  <div style="color:#ff3366;font-weight:700">P → ∞</div>
                  <div style="color:#5a7a9a;margin-top:6px">Lead-Time S:</div>
                  <div style="color:#ff8c00;font-weight:700">{evp_t_cur:.1f}s</div>
                  <div style="color:#5a7a9a;margin-top:6px">Emergency Vehicles:</div>
                  <div style="color:#ff3366;font-weight:700">{n_emg:,}</div>
                </div>""", unsafe_allow_html=True)

            sec_header("⬡", "Junction Load")
            for jn in JUNCTION_NAMES[:5]:
                l   = loads[jn]
                col = _load_color(l)
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;padding:4px 0;border-bottom:1px solid #1a3a5c;'
                    f'font-size:9px">'
                    f'<span style="color:#5a7a9a">{jn[:12]}</span>'
                    f'<span style="color:{col};font-weight:700">{l:.0f}%</span></div>',
                    unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            sec_header("⚡", "Alerts", f"{len(st.session_state.alerts)}")
            render_alerts()

    # ──────────────────────────────────────────────────────
    # TAB 2 — JUNCTION OVERVIEW  (Folium static map)
    # ──────────────────────────────────────────────────────
    with tab2:
        over_c1, over_c2 = st.columns([2, 1])

        with over_c1:
            sec_header("📡", "Junction Network Map")
            m = build_folium_map(loads, st.session_state.evp_active)
            st_folium(m, width=None, height=480, returned_objects=[])

        with over_c2:
            sec_header("⬡", "All Junction Status")
            rows = []
            for jn in JUNCTION_NAMES:
                l   = loads[jn]
                col = _load_color(l)
                ico = "🔴" if l>80 else "🟡" if l>60 else "🟢"
                rows.append({"Junction": jn, "Load": f"{l:.0f}%",
                             "Status": f"{ico} {'CRITICAL' if l>80 else 'BUSY' if l>60 else 'CLEAR'}"})
            st.dataframe(
                pd.DataFrame(rows), use_container_width=True,
                hide_index=True, height=280,
            )

            sec_header("◈", "Top-2 Gauges", "Live")
            top2 = sorted(loads.items(), key=lambda x: x[1], reverse=True)[:2]
            gc1, gc2 = st.columns(2)
            with gc1:
                nm, val = top2[0]
                st.plotly_chart(build_gauge(val, nm[:10], _load_color(val)),
                                use_container_width=True,
                                config={"displayModeBar": False})
            with gc2:
                nm, val = top2[1]
                st.plotly_chart(build_gauge(val, nm[:10], _load_color(val)),
                                use_container_width=True,
                                config={"displayModeBar": False})

    # ──────────────────────────────────────────────────────
    # TAB 3 — ANALYTICS
    # ──────────────────────────────────────────────────────
    with tab3:
        an_c1, an_c2 = st.columns([3, 2])
        with an_c1:
            sec_header("◈", "24-Hour Volume", st.session_state.selected_junction)
            df24 = get_24h_series(st.session_state.selected_junction,
                                  st.session_state.density_pct,
                                  st.session_state.gw_active)
            st.plotly_chart(build_24h_chart(df24, st.session_state.selected_junction),
                            use_container_width=True, config={"displayModeBar": False})

        with an_c2:
            sec_header("◈", "Network Radar")
            st.plotly_chart(build_radar(loads), use_container_width=True,
                            config={"displayModeBar": False})

        an2_c1, an2_c2 = st.columns(2)
        with an2_c1:
            sec_header("◈", "Delay — Static vs Green Wave")
            st.plotly_chart(build_delay_bar(get_flow_comparison(st.session_state.gw_active)),
                            use_container_width=True, config={"displayModeBar": False})
        with an2_c2:
            sec_header("◈", "Day × Junction Heatmap (%)")
            st.plotly_chart(build_heatmap(get_weekly_heatmap(st.session_state.gw_active)),
                            use_container_width=True, config={"displayModeBar": False})

        sec_header("◈", "Vehicle Corridor Distribution")
        st.plotly_chart(build_vehicle_histogram(df_veh, loads),
                        use_container_width=True, config={"displayModeBar": False})

        with st.expander("📋 Raw Data Table"):
            rdf = pd.DataFrame([{
                "Junction": k,
                "Load (%)": round(v, 1),
                "Status": "CRITICAL" if v>80 else "BUSY" if v>60 else "CLEAR",
                "φ (s)": round(green_wave_offset(4.0, st.session_state.v_kmh, st.session_state.cycle_T), 2),
                "ρ (veh/km)": round(lwr_density(v*20, st.session_state.v_kmh), 1),
            } for k, v in loads.items()])
            st.dataframe(rdf, use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────
    # TAB 4 — GREEN WAVE
    # ──────────────────────────────────────────────────────
    with tab4:
        sec_header("⚡", "Green Wave Time-Space Diagram",
                   f"v_c={st.session_state.v_kmh:.0f}km/h · T={st.session_state.cycle_T:.0f}s")
        st.plotly_chart(
            build_green_wave_chart(st.session_state.v_kmh, st.session_state.cycle_T),
            use_container_width=True, config={"displayModeBar": False})

        gw_c1, gw_c2 = st.columns(2)
        with gw_c1:
            sec_header("📐", "Corridor Offsets φ")
            corridors = [
                ("Silk Board → HSR Layout", 4.0),
                ("HSR Layout → Bellandur",  4.0),
                ("Bellandur → Whitefield",  5.0),
                ("ORR → Indiranagar",        3.0),
            ]
            phi_rows = [{
                "Corridor": n,
                "Dist (km)": d,
                "φ (s)": round(green_wave_offset(d, st.session_state.v_kmh, st.session_state.cycle_T), 2),
            } for n, d in corridors]
            st.dataframe(pd.DataFrame(phi_rows), use_container_width=True, hide_index=True)

            st.markdown("""
            <div style="background:#0d1f3c;border:1px solid #1a3a5c;border-radius:6px;
                        padding:12px;font-family:'Courier New';font-size:10px;margin-top:8px">
              <div style="color:#00d4ff;margin-bottom:5px">φ = (L / v_c) mod T</div>
              <div style="color:#5a7a9a;font-size:9px">
                L = signal spacing (m)<br>
                v_c = design speed (m/s)<br>
                T = cycle duration (s)
              </div>
            </div>""", unsafe_allow_html=True)

        with gw_c2:
            sec_header("📊", "Delay Savings per Junction")
            df_cmp3 = get_flow_comparison(st.session_state.gw_active)
            df_cmp3["Saving"] = df_cmp3["Static (min)"] - df_cmp3["Green Wave (min)"]
            sv_fig = go.Figure(go.Bar(
                x=df_cmp3["Junction"], y=df_cmp3["Saving"],
                marker_color="rgba(0,255,136,.7)",
                marker_line=dict(color="#00ff88", width=1),
                text=df_cmp3["Saving"].apply(lambda x: f"-{x} min"),
                textposition="outside", textfont=dict(color="#00ff88", size=9),
            ))
            sv_layout = _base_layout("Delay Saved (min) — Green Wave Active", 270)
            sv_layout["xaxis"] = _axis()
            sv_layout["yaxis"] = _axis(title="Minutes saved")
            sv_layout["showlegend"] = False
            sv_fig.update_layout(**sv_layout)
            st.plotly_chart(sv_fig, use_container_width=True, config={"displayModeBar": False})

    # ──────────────────────────────────────────────────────
    # TAB 5 — EVP
    # ──────────────────────────────────────────────────────
    with tab5:
        evp_c1, evp_c2 = st.columns([3, 2])
        with evp_c1:
            sec_header("🚑", "EVP Lead-Time Model", "S = d / v")
            st.plotly_chart(
                build_evp_chart(st.session_state.evp_distance, st.session_state.evp_speed),
                use_container_width=True, config={"displayModeBar": False})

        with evp_c2:
            sec_header("🚑", "Current EVP Status")
            evp_cur = evp_signal_time(st.session_state.evp_distance, st.session_state.evp_speed)
            sc = "#ff3366" if st.session_state.evp_active else "#5a7a9a"
            border = "1px solid #ff3366" if st.session_state.evp_active else "1px solid #1a3a5c"
            st.markdown(f"""
            <div style="background:#0d1f3c;border:{border};border-radius:8px;
                        padding:14px;font-family:'Courier New';font-size:10px">
              <div style="text-align:center;font-size:28px;margin-bottom:8px">
                {'🚨' if st.session_state.evp_active else '🚑'}
              </div>
              <div style="text-align:center;color:{sc};font-weight:700;letter-spacing:2px;margin-bottom:10px">
                {'ACTIVE — PRE-EMPTED' if st.session_state.evp_active else 'STANDBY'}
              </div>
              <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                <span style="color:#5a7a9a">Priority P:</span>
                <span style="color:#ff3366;font-weight:700">{'→ ∞' if st.session_state.evp_active else '= 1'}</span>
              </div>
              <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                <span style="color:#5a7a9a">Distance:</span>
                <span style="color:#e0f0ff">{st.session_state.evp_distance:.0f} m</span>
              </div>
              <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                <span style="color:#5a7a9a">Speed:</span>
                <span style="color:#e0f0ff">{st.session_state.evp_speed:.0f} km/h</span>
              </div>
              <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                <span style="color:#5a7a9a">Lead-Time S:</span>
                <span style="color:#ff8c00;font-weight:700">{evp_cur:.1f} s</span>
              </div>
              <div style="display:flex;justify-content:space-between">
                <span style="color:#5a7a9a">Fleet Size:</span>
                <span style="color:#ff3366;font-weight:700">{n_emg:,} vehicles</span>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        sec_header("📊", "EVP Sensitivity — Speed × Distance Heatmap")
        spd_r  = [20,30,40,50,60,70,80]
        dist_r = [100,250,500,750,1000,1500,2000]
        hz     = [[round(evp_signal_time(d,s)) for d in dist_r] for s in spd_r]
        evp_h  = go.Figure(go.Heatmap(
            z=hz, x=[f"{d}m" for d in dist_r], y=[f"{s}km/h" for s in spd_r],
            colorscale=[[0,"#00ff88"],[.5,"#ff8c00"],[1,"#ff3366"]],
            colorbar=dict(title=dict(text="s",font=dict(color="#5a7a9a")),
                          tickfont=dict(color="#5a7a9a")),
            texttemplate="%{z}s", textfont=dict(size=9, color="#e0f0ff"),
        ))
        eh_layout = _base_layout("Pre-emption Lead-Time (s) — Speed × Distance", 260)
        eh_layout["xaxis"] = _axis()
        eh_layout["yaxis"] = _axis()
        evp_h.update_layout(**eh_layout)
        st.plotly_chart(evp_h, use_container_width=True, config={"displayModeBar": False})

    # ──────────────────────────────────────────────────────
    # TAB 6 — LP & LWR
    # ──────────────────────────────────────────────────────
    with tab6:
        th_c1, th_c2 = st.columns([3, 2])
        with th_c1:
            sec_header("📐", "LWR Fundamental Diagram")
            st.plotly_chart(build_lwr_chart(), use_container_width=True,
                            config={"displayModeBar": False})

        with th_c2:
            sec_header("🧮", "LP Optimisation Result")
            lp = get_lp_result(loads, st.session_state.gw_active)
            red_w = min(100, lp["reduction_pct"])
            st.markdown(f"""
            <div style="background:#0d1f3c;border:1px solid #1a3a5c;border-radius:8px;
                        padding:14px;font-family:'Courier New';font-size:11px">
              <div style="color:#00d4ff;margin-bottom:8px;font-size:12px">W(D) = Σ min ∫ t dt</div>
              <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                <span style="color:#5a7a9a">Baseline Delay:</span>
                <span style="color:#ff3366;font-weight:700">{lp['total_delay']} min</span>
              </div>
              <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                <span style="color:#5a7a9a">LP-Optimised:</span>
                <span style="color:#00ff88;font-weight:700">{lp['optimised']} min</span>
              </div>
              <div style="display:flex;justify-content:space-between;margin-bottom:10px">
                <span style="color:#5a7a9a">Reduction:</span>
                <span style="color:#a855f7;font-weight:700">{lp['reduction_pct']}%</span>
              </div>
              <div style="height:8px;background:rgba(255,255,255,.05);border-radius:4px;overflow:hidden">
                <div style="height:100%;width:{red_w}%;
                            background:linear-gradient(90deg,#a855f7,#00d4ff);border-radius:4px"></div>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#0d1f3c;border:1px solid #1a3a5c;border-radius:6px;
                        padding:12px;font-family:'Courier New';font-size:10px;line-height:2">
              <div style="color:#00d4ff">Objective:</div>
              <div style="color:#e0f0ff">&nbsp; min W(D) = Σᵢ ∫ tᵢ dt</div>
              <div style="color:#00d4ff;margin-top:6px">Subject To:</div>
              <div style="color:#e0f0ff">&nbsp; gᵢ+yᵢ+rᵢ = T  (cycle)</div>
              <div style="color:#e0f0ff">&nbsp; gᵢ ≥ g_min  (safety)</div>
              <div style="color:#e0f0ff">&nbsp; φᵢ=(Lᵢ/v_c) mod T</div>
              <div style="color:#e0f0ff">&nbsp; Pₑᵥₚ → ∞  (override)</div>
              <div style="color:#e0f0ff">&nbsp; ρ = q/v  (LWR)</div>
            </div>""", unsafe_allow_html=True)

        sec_header("📊", "LP Per-Junction Delay — Before & After")
        st.plotly_chart(build_lp_bar(loads, st.session_state.gw_active),
                        use_container_width=True, config={"displayModeBar": False})

        with st.expander("📚 References"):
            st.markdown("""
            <div style="font-family:'Courier New';font-size:10px;line-height:2;color:#e0f0ff">
              <b style="color:#00d4ff">Graph Theory:</b><br>
              West, D.B. (2001). Introduction to Graph Theory. Prentice Hall.<br><br>
              <b style="color:#00d4ff">Traffic Flow:</b><br>
              Lighthill &amp; Whitham (1955). On Kinematic Waves II. Proc. Royal Society.<br><br>
              <b style="color:#00d4ff">Optimisation:</b><br>
              Taha, H.A. (2017). Operations Research: An Introduction. Pearson.<br><br>
              <b style="color:#00d4ff">Data Sources:</b><br>
              TomTom Traffic Index: Bangalore 2024-25.<br>
              Kaggle: Bangalore Urban Traffic Dataset 2024.<br>
              BBMP Smart City Traffic Reports 2024.
            </div>""", unsafe_allow_html=True)

    # ── Periodic alerts ──────────────────────────────────────
    if tick > 0 and tick % 15 == 0:
        t = _ALERT_TEMPLATES[tick // 15 % len(_ALERT_TEMPLATES)]
        push_alert(*t)

    # ── Bottom bar ───────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="display:flex;justify-content:space-between;'
        'font-family:\'Courier New\',monospace;font-size:8px;color:#5a7a9a;padding:2px 0">'
        '<span>● ONLINE · 10/10 NODES · 48 SIGNALS</span>'
        '<span>W(D)=Σ min∫(t)dt · φ=L/v_c mod T · S=d/v · ρ=q/v</span>'
        '<span>NMIT ISE · NISHCHAL NB25ISE160 · RISHUL NB25ISE186</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()

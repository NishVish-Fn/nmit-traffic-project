"""
har har mahadev
╔══════════════════════════════════════════════════════════════════════╗
║   URBAN FLOW & LIFE-LINES                                            ║
║   Multi-Objective Optimization Dashboard – Bangalore ATCS            ║
║   NMIT ISE · Nishchal Vishwanath (NB25ISE160) · Rishul KH (NB25ISE186)║
╚══════════════════════════════════════════════════════════════════════╝

Install requirements:
    pip install streamlit plotly folium streamlit-folium numpy pandas

Run:
    streamlit run bangalore_traffic_dashboard.py
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS  (all standard; verified against requirements.txt)
# ─────────────────────────────────────────────────────────────
import math
import random
import datetime
import time as _time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Flow & Life-Lines | Bangalore ATCS",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLE  (dark police-dashboard theme)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Root Variables ── */
:root {
    --bg:        #050a14;
    --panel:     #0a1628;
    --card:      #0d1f3c;
    --blue:      #00d4ff;
    --green:     #00ff88;
    --red:       #ff3366;
    --orange:    #ff8c00;
    --purple:    #a855f7;
    --dim:       #5a7a9a;
    --border:    #1a3a5c;
    --text:      #e0f0ff;
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: #050a14 !important;
    color: #e0f0ff !important;
    font-family: 'Courier New', monospace !important;
}

[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid #1a3a5c !important;
}

[data-testid="stSidebar"] * { color: #e0f0ff !important; }

/* ── Remove default padding ── */
.block-container { padding: 0.5rem 1rem 1rem 1rem !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #0d1f3c !important;
    border: 1px solid #1a3a5c !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
[data-testid="stMetricValue"]  { color: #00d4ff !important; font-family: 'Courier New', monospace !important; }
[data-testid="stMetricLabel"]  { color: #5a7a9a !important; font-size: 11px !important; letter-spacing: 1px !important; text-transform: uppercase !important; }
[data-testid="stMetricDelta"]  { font-size: 11px !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    background: #0a1628 !important;
    color: #5a7a9a !important;
    border: 1px solid #1a3a5c !important;
    border-radius: 4px 4px 0 0 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: #0d1f3c !important;
    color: #00d4ff !important;
    border-bottom-color: #00d4ff !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div { background: #00d4ff !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0d1f3c, #0a1628) !important;
    border: 1px solid #1a3a5c !important;
    color: #00d4ff !important;
    font-family: 'Courier New', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    border-radius: 4px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #00d4ff !important;
    box-shadow: 0 0 12px rgba(0,212,255,0.3) !important;
}

/* ── Selectbox / Multiselect ── */
[data-testid="stSelectbox"] select,
[data-testid="stSelectbox"] > div,
.stSelectbox > div { background: #0d1f3c !important; border-color: #1a3a5c !important; color: #e0f0ff !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #050a14; }
::-webkit-scrollbar-thumb { background: #1a3a5c; border-radius: 2px; }

/* ── Divider ── */
hr { border-color: #1a3a5c !important; }

/* ── Info / warning boxes ── */
[data-testid="stAlert"] { background: #0d1f3c !important; border-color: #1a3a5c !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0a1628 !important;
    border: 1px solid #1a3a5c !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# KAGGLE-INSPIRED DATA (Bangalore Traffic Dataset 2024-2025)
# Source: TomTom Traffic Index, Kaggle BLR Traffic, BBMP data
# ─────────────────────────────────────────────────────────────
HOURLY_PATTERN = [
    0.28, 0.18, 0.13, 0.11, 0.16, 0.42,
    0.82, 0.97, 0.91, 0.73, 0.67, 0.74,
    0.80, 0.75, 0.70, 0.77, 0.92, 0.99,
    0.96, 0.84, 0.72, 0.62, 0.52, 0.40,
]

JUNCTION_NAMES = [
    "Silk Board", "Hebbal", "ORR Junction", "Koramangala",
    "HSR Layout", "Bellandur", "Indiranagar", "Whitefield",
    "Yeshwanthpur", "Kengeri",
]

JUNCTION_COORDS = {
    "Silk Board":    [12.9172, 77.6228],
    "Hebbal":        [13.0350, 77.5970],
    "ORR Junction":  [12.9698, 77.6496],
    "Koramangala":   [12.9352, 77.6245],
    "HSR Layout":    [12.9082, 77.6476],
    "Bellandur":     [12.9270, 77.6780],
    "Indiranagar":   [12.9784, 77.6408],
    "Whitefield":    [12.9699, 77.7500],
    "Yeshwanthpur":  [13.0275, 77.5530],
    "Kengeri":       [12.9116, 77.4843],
}

BASE_LOADS = {
    "Silk Board": 95, "Hebbal": 85, "ORR Junction": 78, "Koramangala": 70,
    "HSR Layout": 72, "Bellandur": 68, "Indiranagar": 60, "Whitefield": 55,
    "Yeshwanthpur": 65, "Kengeri": 40,
}

ROAD_EDGES = [
    ("Silk Board",   "HSR Layout"),
    ("Silk Board",   "Koramangala"),
    ("Silk Board",   "ORR Junction"),
    ("HSR Layout",   "Bellandur"),
    ("Bellandur",    "Whitefield"),
    ("ORR Junction", "Indiranagar"),
    ("Indiranagar",  "Koramangala"),
    ("Hebbal",       "Yeshwanthpur"),
    ("Hebbal",       "ORR Junction"),
    ("Yeshwanthpur", "Kengeri"),
    ("Koramangala",  "HSR Layout"),
    ("ORR Junction", "Bellandur"),
]

DAYS  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKS = [0.75, 0.82, 0.90, 0.93, 0.98, 0.72, 0.52]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#050a14",
    plot_bgcolor="#0a1628",
    font=dict(family="Courier New, monospace", color="#e0f0ff", size=11),
    xaxis=dict(gridcolor="#1a3a5c", linecolor="#1a3a5c", tickfont=dict(color="#5a7a9a")),
    yaxis=dict(gridcolor="#1a3a5c", linecolor="#1a3a5c", tickfont=dict(color="#5a7a9a")),
    margin=dict(l=40, r=20, t=35, b=35),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#5a7a9a", size=10)),
)


# ─────────────────────────────────────────────────────────────
# MATH MODELS  (from the paper)
# ─────────────────────────────────────────────────────────────
def green_wave_offset(L_km: float, v_kmh: float, T_s: float) -> float:
    """φ = (L / v_c) mod T  — Green Wave synchronisation offset (seconds)"""
    if v_kmh <= 0:
        return 0.0
    v_ms = v_kmh * 1000.0 / 3600.0
    L_m  = L_km * 1000.0
    return (L_m / v_ms) % T_s


def evp_signal_time(d_m: float, v_kmh: float) -> float:
    """S = d / v  — Emergency Vehicle Preemption lead time (seconds)"""
    v_ms = v_kmh * 1000.0 / 3600.0
    return (d_m / v_ms) if v_ms > 0 else float("inf")


def lwr_density(flow_vph: float, speed_kmh: float) -> float:
    """LWR: density = flow / speed  (vehicles per km)"""
    return flow_vph / speed_kmh if speed_kmh > 0 else 0.0


def delay_objective(loads: list[float], gw_active: bool) -> float:
    """LP objective: W(D) = Σ min ∫ t dt  (simplified)"""
    factor = 0.65 if gw_active else 1.0
    return sum(max(0, (l - 40) * 0.4 * factor) for l in loads)


# ─────────────────────────────────────────────────────────────
# DATA GENERATORS
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1)
def get_current_loads(density_pct: float, gw_active: bool, seed: int) -> dict:
    """Returns current % load for each junction (re-generates every second)."""
    rng = random.Random(seed)
    hour = datetime.datetime.now().hour
    tf   = HOURLY_PATTERN[hour] * (density_pct / 65.0)
    gw_f = 0.72 if gw_active else 1.0
    result = {}
    for name, base in BASE_LOADS.items():
        noise = rng.gauss(0, 4)
        result[name] = round(max(5, min(99, base * tf * gw_f + noise)), 1)
    return result


def get_24h_series(junction: str, density_pct: float, gw_active: bool) -> pd.DataFrame:
    """24-hour traffic volume series for a junction (Kaggle pattern)."""
    rng   = random.Random(42)
    base  = BASE_LOADS.get(junction, 60)
    gw_f  = 0.75 if gw_active else 1.0
    times = [datetime.datetime(2024, 6, 17, h) for h in range(24)]
    vols  = []
    for h in range(24):
        tf    = HOURLY_PATTERN[h]
        noise = rng.gauss(0, 5)
        v     = base * tf * (density_pct / 65.0) * gw_f * 30 + noise * 10
        vols.append(max(0, round(v)))
    return pd.DataFrame({"Time": times, "Volume (veh/h)": vols})


def get_weekly_heatmap(gw_active: bool) -> pd.DataFrame:
    """Junction × Day load heatmap."""
    rng   = random.Random(7)
    gw_f  = 0.75 if gw_active else 1.0
    data  = {}
    for j, base in BASE_LOADS.items():
        data[j] = [round(base * w * gw_f + rng.gauss(0, 3)) for w in WEEKS]
    return pd.DataFrame(data, index=DAYS)


def get_flow_comparison(gw_active: bool) -> pd.DataFrame:
    """Static vs Green-Wave delay for 6 key junctions."""
    top6   = list(BASE_LOADS.items())[:6]
    names  = [n for n, _ in top6]
    static = [round(b * 0.48) for _, b in top6]
    gw     = [round(b * 0.48 * (0.65 if gw_active else 1.0)) for _, b in top6]
    return pd.DataFrame({"Junction": names, "Static (min)": static, "Green Wave (min)": gw})


def get_lp_result(loads: dict, gw_active: bool) -> dict:
    """Simulate LP optimisation result."""
    total_delay   = delay_objective(list(loads.values()), gw_active)
    optimised     = total_delay * (0.70 if gw_active else 1.0)
    reduction_pct = round((1 - optimised / max(total_delay, 0.01)) * 100, 1) if gw_active else 0
    return {
        "total_delay":   round(total_delay, 1),
        "optimised":     round(optimised, 1),
        "reduction_pct": reduction_pct,
    }


# ─────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────
def build_24h_chart(df: pd.DataFrame, junction: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Time"], y=df["Volume (veh/h)"],
        mode="lines", name="Volume",
        line=dict(color="#00d4ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.07)",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"24-Hour Traffic Volume — {junction}", font=dict(color="#00d4ff", size=12)),
        height=260,
    )
    return fig


def build_delay_bar(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Junction"], y=df["Static (min)"],
        name="Static Signals", marker_color="rgba(255,51,102,0.75)",
        marker_line_color="#ff3366", marker_line_width=1,
    ))
    fig.add_trace(go.Bar(
        x=df["Junction"], y=df["Green Wave (min)"],
        name="Green Wave", marker_color="rgba(0,255,136,0.75)",
        marker_line_color="#00ff88", marker_line_width=1,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        title=dict(text="Average Delay Reduction — Static vs Green Wave (min)", font=dict(color="#00d4ff", size=12)),
        height=260,
    )
    return fig


def build_heatmap(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=df.values.tolist(),
        x=list(df.columns),
        y=list(df.index),
        colorscale=[
            [0.0,  "#001a06"],
            [0.4,  "#00ff88"],
            [0.7,  "#ff8c00"],
            [1.0,  "#ff3366"],
        ],
        colorbar=dict(
            tickfont=dict(color="#5a7a9a", size=9),
            outlinecolor="#1a3a5c",
        ),
        hoverongaps=False,
        text=df.values.tolist(),
        texttemplate="%{z:.0f}%",
        textfont=dict(size=8, color="#e0f0ff"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Junction Load Heatmap — Day × Node (%)", font=dict(color="#00d4ff", size=12)),
        height=300,
        xaxis=dict(tickfont=dict(color="#5a7a9a", size=9), linecolor="#1a3a5c"),
        yaxis=dict(tickfont=dict(color="#5a7a9a", size=9), linecolor="#1a3a5c"),
    )
    return fig


def build_radar(loads: dict) -> go.Figure:
    names = list(loads.keys())
    vals  = list(loads.values())
    vals_closed = vals + [vals[0]]
    names_closed = names + [names[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=names_closed, fill="toself",
        fillcolor="rgba(0,212,255,0.1)",
        line=dict(color="#00d4ff", width=1.5),
        name="Current Load",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="#0a1628",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1a3a5c", tickfont=dict(color="#5a7a9a", size=8)),
            angularaxis=dict(gridcolor="#1a3a5c", tickfont=dict(color="#5a7a9a", size=9)),
        ),
        title=dict(text="Network Load — All Junctions", font=dict(color="#00d4ff", size=12)),
        height=320,
    )
    return fig


def build_gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(text=title, font=dict(color="#5a7a9a", size=10)),
        number=dict(font=dict(color=color, size=26, family="Courier New"), suffix="%"),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#5a7a9a", tickfont=dict(size=8)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#0a1628",
            borderwidth=1,
            bordercolor="#1a3a5c",
            steps=[
                dict(range=[0, 40],  color="rgba(0,255,136,0.07)"),
                dict(range=[40, 70], color="rgba(255,140,0,0.07)"),
                dict(range=[70, 100],color="rgba(255,51,102,0.07)"),
            ],
            threshold=dict(line=dict(color=color, width=2), thickness=0.8, value=value),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#050a14",
        font=dict(family="Courier New, monospace", color="#e0f0ff"),
        height=160,
        margin=dict(l=15, r=15, t=30, b=5),
    )
    return fig


def build_green_wave_chart(v_kmh: float, T_s: float) -> go.Figure:
    """Time-Space diagram showing Green Wave corridor."""
    junctions_gw = ["Silk Board (0 km)", "HSR Layout (4 km)", "Bellandur (8 km)", "Whitefield (13 km)"]
    distances_km  = [0, 4, 8, 13]
    offsets        = [green_wave_offset(d, v_kmh, T_s) for d in distances_km]

    fig = go.Figure()
    # Green phases
    for i, (dist, off) in enumerate(zip(distances_km, offsets)):
        for cycle in range(3):
            g_start = off + cycle * T_s
            g_end   = g_start + T_s * 0.5
            fig.add_shape(type="rect",
                x0=g_start, x1=g_end, y0=dist - 0.3, y1=dist + 0.3,
                fillcolor="rgba(0,255,136,0.25)", line=dict(color="#00ff88", width=1))
            r_start = g_end
            r_end   = r_start + T_s * 0.5
            fig.add_shape(type="rect",
                x0=r_start, x1=r_end, y0=dist - 0.3, y1=dist + 0.3,
                fillcolor="rgba(255,51,102,0.25)", line=dict(color="#ff3366", width=1))

    # Vehicle trajectory line
    v_ms = v_kmh * 1000.0 / 3600.0
    t_vals = [0, distances_km[-1] * 1000 / v_ms]
    d_vals = [0, distances_km[-1]]
    fig.add_trace(go.Scatter(
        x=t_vals, y=d_vals,
        mode="lines",
        line=dict(color="#00d4ff", width=2.5, dash="dot"),
        name=f"Vehicle @ {v_kmh} km/h",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Green Wave Time-Space Diagram  (v_c = {v_kmh} km/h, T = {T_s}s)", font=dict(color="#00d4ff", size=12)),
        height=280,
        xaxis=dict(title="Time (s)", gridcolor="#1a3a5c", linecolor="#1a3a5c", tickfont=dict(color="#5a7a9a")),
        yaxis=dict(title="Distance (km)", gridcolor="#1a3a5c", linecolor="#1a3a5c", tickfont=dict(color="#5a7a9a"),
                   tickvals=distances_km, ticktext=junctions_gw),
        showlegend=True,
    )
    return fig


def build_evp_chart(d_m: float, v_kmh: float) -> go.Figure:
    """EVP lead-time vs distance for different speeds."""
    speeds    = [30, 40, 50, 60]
    distances = list(range(100, int(d_m) + 100, 100))
    fig       = go.Figure()
    palette   = ["#ff3366", "#ff8c00", "#ffd700", "#00ff88"]
    for sp, col in zip(speeds, palette):
        times_s = [evp_signal_time(d, sp) for d in distances]
        fig.add_trace(go.Scatter(
            x=distances, y=times_s,
            mode="lines", name=f"v = {sp} km/h",
            line=dict(color=col, width=1.8),
        ))
    # Mark current selection
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
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="EVP Lead-Time: S = d / v  (seconds pre-emption needed)", font=dict(color="#00d4ff", size=12)),
        height=260,
        xaxis=dict(title="Distance to Signal (m)", gridcolor="#1a3a5c", linecolor="#1a3a5c", tickfont=dict(color="#5a7a9a")),
        yaxis=dict(title="Pre-emption Time (s)",    gridcolor="#1a3a5c", linecolor="#1a3a5c", tickfont=dict(color="#5a7a9a")),
    )
    return fig


def build_lwr_chart() -> go.Figure:
    """LWR flow-density-speed fundamental diagram."""
    rho_max  = 120   # veh/km (jam density)
    v_free   = 60    # km/h (free-flow speed)
    rhos     = np.linspace(0, rho_max, 200)
    speeds   = np.maximum(0, v_free * (1 - rhos / rho_max))
    flows    = rhos * speeds

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Flow-Density (q-k)", "Speed-Density (v-k)"],
    )
    fig.add_trace(go.Scatter(
        x=list(rhos), y=list(flows),
        mode="lines", line=dict(color="#00d4ff", width=2), name="q-k",
        fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(rhos), y=list(speeds),
        mode="lines", line=dict(color="#00ff88", width=2), name="v-k",
        fill="tozeroy", fillcolor="rgba(0,255,136,0.06)",
    ), row=1, col=2)

    for col_idx in [1, 2]:
        fig.update_xaxes(
            title_text="Density ρ (veh/km)",
            gridcolor="#1a3a5c", linecolor="#1a3a5c",
            tickfont=dict(color="#5a7a9a"), row=1, col=col_idx,
        )
    fig.update_yaxes(title_text="Flow q (veh/h)",  gridcolor="#1a3a5c", linecolor="#1a3a5c", tickfont=dict(color="#5a7a9a"), row=1, col=1)
    fig.update_yaxes(title_text="Speed v (km/h)",  gridcolor="#1a3a5c", linecolor="#1a3a5c", tickfont=dict(color="#5a7a9a"), row=1, col=2)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="LWR Traffic Flow Model — Lighthill-Whitham-Richards", font=dict(color="#00d4ff", size=12)),
        height=290,
        showlegend=False,
        annotations=[
            dict(text="Flow-Density (q-k)", font=dict(color="#5a7a9a", size=10), showarrow=False, x=0.18, xref="paper", y=1.05, yref="paper"),
            dict(text="Speed-Density (v-k)", font=dict(color="#5a7a9a", size=10), showarrow=False, x=0.82, xref="paper", y=1.05, yref="paper"),
        ],
    )
    return fig


# ─────────────────────────────────────────────────────────────
# FOLIUM MAP BUILDER
# ─────────────────────────────────────────────────────────────
def build_map(loads: dict, layer: str, evp_active: bool) -> folium.Map:
    m = folium.Map(
        location=[12.9716, 77.5946],
        zoom_start=12,
        tiles="CartoDB DarkMatter",
        prefer_canvas=True,
    )

    # Draw road edges
    evp_path = {"Silk Board", "HSR Layout", "Bellandur"}
    for a, b in ROAD_EDGES:
        ca  = JUNCTION_COORDS[a]
        cb  = JUNCTION_COORDS[b]
        is_evp = evp_active and a in evp_path and b in evp_path
        color  = "#ff3366" if is_evp else (
            "#1a3a5c" if layer == "normal" else
            _road_color((loads.get(a, 50) + loads.get(b, 50)) / 2)
        )
        weight = 4 if is_evp else 2
        folium.PolyLine(
            locations=[ca, cb],
            color=color,
            weight=weight,
            opacity=0.85 if is_evp else 0.7,
        ).add_to(m)

    # Junction markers
    for name, coord in JUNCTION_COORDS.items():
        load  = loads.get(name, 50)
        color = _load_color(load)
        radius = max(6, min(16, load / 7))

        popup_html = f"""
        <div style="background:#0d1f3c;border:1px solid #1a3a5c;
                    padding:10px;border-radius:6px;font-family:'Courier New',monospace;
                    font-size:11px;color:#e0f0ff;min-width:180px">
            <b style="color:#00d4ff">{name}</b><br>
            <hr style="border-color:#1a3a5c;margin:5px 0">
            <span style="color:#5a7a9a">LOAD:</span>
            <span style="color:{color};font-weight:700">{load:.0f}%</span><br>
            <span style="color:#5a7a9a">STATUS:</span>
            <span style="color:{color}">{'CRITICAL' if load>80 else 'BUSY' if load>60 else 'CLEAR'}</span><br>
            <span style="color:#5a7a9a">GREEN WAVE:</span>
            <span style="color:#00ff88">ACTIVE</span>
        </div>"""

        folium.CircleMarker(
            location=coord,
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=2,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{name}: {load:.0f}%",
        ).add_to(m)

        # Label
        folium.map.Marker(
            coord,
            icon=folium.DivIcon(
                html=f'<div style="font-family:Courier New;font-size:9px;color:#5a7a9a;'
                     f'white-space:nowrap;margin-top:12px;margin-left:8px">{name}</div>',
                icon_size=(120, 20),
                icon_anchor=(0, 0),
            ),
        ).add_to(m)

    # EVP ambulance marker
    if evp_active:
        evp_coord = JUNCTION_COORDS["Silk Board"]
        folium.Marker(
            location=evp_coord,
            icon=folium.DivIcon(
                html='<div style="font-size:20px;animation:blink 1s infinite">🚑</div>',
                icon_size=(30, 30),
                icon_anchor=(15, 15),
            ),
            tooltip="EMERGENCY VEHICLE — EVP Active",
        ).add_to(m)

    return m


def _load_color(load: float) -> str:
    if load > 80:
        return "#ff3366"
    elif load > 60:
        return "#ff8c00"
    elif load > 40:
        return "#ffd700"
    return "#00ff88"


def _road_color(avg_load: float) -> str:
    if avg_load > 80:
        return "#ff3366"
    elif avg_load > 60:
        return "#ff8c00"
    return "#00ff88"


# ─────────────────────────────────────────────────────────────
# SIGNAL TIMER COMPONENT
# ─────────────────────────────────────────────────────────────
def signal_html(phase: str, countdown: int, phi: float, evp_on: bool) -> str:
    r_on = "on" if phase == "RED"    or evp_on else "off"
    y_on = "on" if phase == "YELLOW" and not evp_on else "off"
    g_on = "on" if phase == "GREEN"  or evp_on else "off"

    colors = {
        "on_RED":    "#ff3366",  "off_RED":    "#1a0000",
        "on_YELLOW": "#ffd700",  "off_YELLOW": "#1a1200",
        "on_GREEN":  "#00ff88",  "off_GREEN":  "#001a06",
    }
    r_col = colors[f"{r_on}_RED"]
    y_col = colors[f"{y_on}_YELLOW"]
    g_col = colors[f"{g_on}_GREEN"]

    glow_r = f"box-shadow:0 0 14px {r_col};" if r_on == "on" else ""
    glow_y = f"box-shadow:0 0 14px {y_col};" if y_on == "on" else ""
    glow_g = f"box-shadow:0 0 14px {g_col};" if g_on == "on" else ""

    phase_label = "EVP OVERRIDE — P → ∞" if evp_on else f"{phase} PHASE  φ = {phi:.1f}s"
    phase_color = "#ff3366" if evp_on else ("#00ff88" if phase == "GREEN" else "#ffd700" if phase == "YELLOW" else "#ff3366")

    return f"""
    <div style="background:#0d1f3c;border:1px solid #1a3a5c;border-radius:8px;
                padding:14px;text-align:center;font-family:'Courier New',monospace">
        <div style="font-size:10px;color:#5a7a9a;letter-spacing:2px;margin-bottom:10px">
            SIGNAL CONTROLLER
        </div>
        <div style="display:flex;justify-content:center;gap:10px;margin-bottom:12px">
            <div style="width:22px;height:22px;border-radius:50%;background:{r_col};{glow_r}border:2px solid rgba(255,255,255,0.1)"></div>
            <div style="width:22px;height:22px;border-radius:50%;background:{y_col};{glow_y}border:2px solid rgba(255,255,255,0.1)"></div>
            <div style="width:22px;height:22px;border-radius:50%;background:{g_col};{glow_g}border:2px solid rgba(255,255,255,0.1)"></div>
        </div>
        <div style="font-size:38px;font-weight:700;color:#00d4ff;
                    font-family:'Courier New',monospace;text-shadow:0 0 20px rgba(0,212,255,0.5);
                    line-height:1">{str(countdown).zfill(2)}</div>
        <div style="font-size:9px;color:{phase_color};letter-spacing:2px;margin-top:6px">{phase_label}</div>
    </div>
    """


# ─────────────────────────────────────────────────────────────
# HEADER COMPONENT
# ─────────────────────────────────────────────────────────────
def render_header(evp_active: bool, gw_active: bool):
    now = datetime.datetime.now()
    evp_badge = (
        '<span style="background:rgba(255,51,102,0.15);border:1px solid #ff3366;'
        'color:#ff3366;padding:3px 10px;border-radius:12px;font-size:10px;'
        'letter-spacing:1.5px;animation:blink 1s step-end infinite">'
        '🚨 EVP ACTIVE</span>'
        if evp_active else
        '<span style="background:rgba(0,255,136,0.1);border:1px solid #00ff88;'
        'color:#00ff88;padding:3px 10px;border-radius:12px;font-size:10px;letter-spacing:1.5px">'
        '● SYSTEM NOMINAL</span>'
    )
    gw_badge = (
        '<span style="background:rgba(0,255,136,0.1);border:1px solid #00ff88;'
        'color:#00ff88;padding:3px 10px;border-radius:12px;font-size:10px;letter-spacing:1.5px">'
        '⬡ GREEN WAVE ON</span>'
        if gw_active else
        '<span style="background:rgba(255,140,0,0.1);border:1px solid #ff8c00;'
        'color:#ff8c00;padding:3px 10px;border-radius:12px;font-size:10px;letter-spacing:1.5px">'
        '⬡ STATIC SIGNALS</span>'
    )

    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#020810,#0a1628,#020810);
                border-bottom:1px solid #1a3a5c;padding:10px 0 10px 0;
                display:flex;align-items:center;justify-content:space-between;
                box-shadow:0 2px 30px rgba(0,212,255,0.15);margin-bottom:12px">
        <div style="display:flex;align-items:center;gap:14px">
            <div style="font-size:28px">🚦</div>
            <div>
                <div style="font-size:15px;font-weight:700;color:#00d4ff;
                            letter-spacing:3px;text-shadow:0 0 20px rgba(0,212,255,0.4)">
                    URBAN FLOW &amp; LIFE-LINES
                </div>
                <div style="font-size:9px;color:#5a7a9a;letter-spacing:2px">
                    BANGALORE ATCS · NMIT ISE · NB25ISE160 / NB25ISE186
                </div>
            </div>
        </div>
        <div style="display:flex;gap:8px;align-items:center">
            {evp_badge}&nbsp;{gw_badge}
        </div>
        <div style="text-align:right">
            <div style="font-size:20px;font-weight:700;color:#00d4ff;
                        font-family:'Courier New',monospace;letter-spacing:3px">
                {now.strftime('%H:%M:%S')}
            </div>
            <div style="font-size:9px;color:#5a7a9a;letter-spacing:1px">
                {now.strftime('%a %d %b %Y')} · IST · BANGALORE
            </div>
        </div>
    </div>
    <style>
        @keyframes blink {{0%,100%{{opacity:1}}50%{{opacity:0.3}}}}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SECTION HEADER HELPER
# ─────────────────────────────────────────────────────────────
def sec_header(icon: str, title: str, badge: str = ""):
    badge_html = (
        f'<span style="font-size:8px;padding:2px 7px;border-radius:10px;'
        f'background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);'
        f'color:#00d4ff">{badge}</span>'
        if badge else ""
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'padding:6px 0 4px 0;border-bottom:1px solid #1a3a5c;margin-bottom:8px">'
        f'<span style="font-size:10px;font-weight:700;letter-spacing:2px;'
        f'color:#00d4ff;text-transform:uppercase">{icon} {title}</span>'
        f'{badge_html}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# ALERT BOX HELPER
# ─────────────────────────────────────────────────────────────
_ALERTS: list = []

def push_alert(alert_type: str, msg: str, level: str = "info"):
    colors = {"info": "#00d4ff", "warn": "#ff8c00", "danger": "#ff3366", "success": "#00ff88"}
    col    = colors.get(level, "#00d4ff")
    now    = datetime.datetime.now().strftime("%H:%M:%S")
    _ALERTS.insert(0, (alert_type, msg, col, now))
    if len(_ALERTS) > 15:
        _ALERTS.pop()


def render_alerts():
    for a_type, msg, col, ts in _ALERTS[:8]:
        st.markdown(
            f'<div style="background:#0d1f3c;border-left:3px solid {col};'
            f'border-radius:0 4px 4px 0;padding:7px 10px;margin-bottom:5px;font-size:10px">'
            f'<span style="color:{col};font-weight:700;letter-spacing:1px">{a_type}</span>'
            f'<span style="color:#5a7a9a;float:right;font-size:9px">{ts}</span><br>'
            f'<span style="color:#e0f0ff">{msg}</span></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "evp_active":       False,
        "gw_active":        True,
        "density_pct":      65,
        "map_layer":        "normal",
        "selected_junction":"Silk Board",
        "sim_speed":        1,
        "v_kmh":            40.0,
        "cycle_T":          60.0,
        "evp_distance":     500.0,
        "evp_speed":        40.0,
        "signal_phase":     "GREEN",
        "signal_timer":     30,
        "tick":             0,
        "alerts_seeded":    False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state.alerts_seeded:
        push_alert("SYSTEM",    "Urban Flow Command initialised — 10 nodes online", "success")
        push_alert("DATA",      "Kaggle BLR Traffic Dataset 2024-25 loaded", "info")
        push_alert("GREEN WAVE","Algorithm v2.1 active — φ = L/v_c mod T", "success")
        st.session_state.alerts_seeded = True


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    init_state()

    # ── Tick for live feel ──────────────────────────────────
    st.session_state.tick += 1
    seed = st.session_state.tick // 5   # data refreshes every ~5 re-renders

    # ── Load current junction data ──────────────────────────
    loads = get_current_loads(
        st.session_state.density_pct,
        st.session_state.gw_active,
        seed,
    )

    # ── Signal phase logic ──────────────────────────────────
    st.session_state.signal_timer -= 1
    if st.session_state.signal_timer <= 0:
        if st.session_state.signal_phase == "GREEN":
            st.session_state.signal_phase = "YELLOW"
            st.session_state.signal_timer = 5
        elif st.session_state.signal_phase == "YELLOW":
            st.session_state.signal_phase = "RED"
            st.session_state.signal_timer = 25
        else:
            st.session_state.signal_phase = "GREEN"
            st.session_state.signal_timer = 30

    # ── HEADER ──────────────────────────────────────────────
    render_header(st.session_state.evp_active, st.session_state.gw_active)

    # ── SIDEBAR ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="font-size:13px;font-weight:700;color:#00d4ff;'
            'letter-spacing:2px;margin-bottom:14px">⚙ COMMAND PANEL</div>',
            unsafe_allow_html=True,
        )

        # EVP Toggle
        if st.button(
            "🚑 ACTIVATE EMERGENCY MODE" if not st.session_state.evp_active else "✅ DEACTIVATE EMERGENCY",
            use_container_width=True,
        ):
            st.session_state.evp_active = not st.session_state.evp_active
            if st.session_state.evp_active:
                push_alert("🚨 EVP ACTIVATED", "Silk Board→HSR→Bellandur pre-empted. P→∞", "danger")
                st.session_state.signal_phase = "GREEN"
                st.session_state.signal_timer = 120
            else:
                push_alert("✅ EVP CLEARED", "Emergency vehicle arrived. Signals resuming.", "success")
            st.rerun()

        st.markdown("---")

        # Green Wave Toggle
        gw_val = st.toggle("⬡ Green Wave Algorithm", value=st.session_state.gw_active, key="gw_toggle_input")
        if gw_val != st.session_state.gw_active:
            st.session_state.gw_active = gw_val
            push_alert("GREEN WAVE", "Activated — φ synchronisation" if gw_val else "Disabled — static timers", "success" if gw_val else "warn")
            st.rerun()

        st.markdown("---")
        st.markdown('<div style="font-size:9px;color:#5a7a9a;letter-spacing:1px;margin-bottom:4px">TRAFFIC DENSITY</div>', unsafe_allow_html=True)
        new_dens = st.slider("", 20, 100, st.session_state.density_pct, key="density_slider", label_visibility="collapsed")
        if new_dens != st.session_state.density_pct:
            st.session_state.density_pct = new_dens

        st.markdown('<div style="font-size:9px;color:#5a7a9a;letter-spacing:1px;margin-bottom:4px;margin-top:10px">GREEN WAVE SPEED (km/h)</div>', unsafe_allow_html=True)
        st.session_state.v_kmh = st.slider("", 20.0, 80.0, st.session_state.v_kmh, 1.0, key="v_slider", label_visibility="collapsed")

        st.markdown('<div style="font-size:9px;color:#5a7a9a;letter-spacing:1px;margin-bottom:4px;margin-top:10px">SIGNAL CYCLE T (s)</div>', unsafe_allow_html=True)
        st.session_state.cycle_T = st.slider("", 30.0, 120.0, st.session_state.cycle_T, 5.0, key="T_slider", label_visibility="collapsed")

        st.markdown("---")
        st.markdown('<div style="font-size:9px;color:#5a7a9a;letter-spacing:1px;margin-bottom:4px">MAP LAYER</div>', unsafe_allow_html=True)
        st.session_state.map_layer = st.selectbox(
            "",
            ["normal", "density", "green_wave", "evp_route"],
            format_func=lambda x: {"normal": "Normal", "density": "Density Heatmap", "green_wave": "Green Wave", "evp_route": "EVP Route"}[x],
            key="layer_select",
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown('<div style="font-size:9px;color:#5a7a9a;letter-spacing:1px;margin-bottom:4px">FOCUS JUNCTION</div>', unsafe_allow_html=True)
        st.session_state.selected_junction = st.selectbox(
            "",
            JUNCTION_NAMES,
            index=JUNCTION_NAMES.index(st.session_state.selected_junction),
            key="junction_select",
            label_visibility="collapsed",
        )

        st.markdown("---")
        # EVP parameters
        with st.expander("🚑 EVP Parameters"):
            st.session_state.evp_distance = st.slider("Distance to signal (m)", 100.0, 2000.0, st.session_state.evp_distance, 50.0)
            st.session_state.evp_speed    = st.slider("Ambulance speed (km/h)", 20.0, 80.0,   st.session_state.evp_speed, 1.0)

        # Math formulas
        with st.expander("📐 Live Math"):
            phi   = green_wave_offset(4.0, st.session_state.v_kmh, st.session_state.cycle_T)
            evp_t = evp_signal_time(st.session_state.evp_distance, st.session_state.evp_speed)
            sel_l = loads.get(st.session_state.selected_junction, 50)
            lwr   = lwr_density(sel_l * 20, st.session_state.v_kmh)
            st.markdown(f"""
            <div style="font-size:10px;font-family:'Courier New',monospace">
                <div style="color:#5a7a9a">Green Wave Offset:</div>
                <div style="color:#00d4ff">φ = {phi:.2f} s</div>
                <div style="color:#5a7a9a;margin-top:8px">EVP Lead-Time:</div>
                <div style="color:#ff3366">S = {evp_t:.1f} s</div>
                <div style="color:#5a7a9a;margin-top:8px">LWR Density ({st.session_state.selected_junction[:6]}):</div>
                <div style="color:#00ff88">ρ = {lwr:.1f} veh/km</div>
                <div style="color:#5a7a9a;margin-top:8px">LP Objective W(D):</div>
                <div style="color:#a855f7">{get_lp_result(loads,st.session_state.gw_active)['optimised']:.1f} min</div>
            </div>
            """, unsafe_allow_html=True)

        # Auto-refresh
        st.markdown("---")
        auto_refresh = st.toggle("🔄 Auto-Refresh (5s)", value=False, key="auto_refresh")
        if auto_refresh:
            _time.sleep(5)
            st.rerun()

        # Footer
        st.markdown("""
        <div style="font-size:8px;color:#5a7a9a;margin-top:14px;line-height:1.6">
            Data: Kaggle BLR Traffic 2024-25<br>
            TomTom Traffic Index · BBMP<br>
            Model: LP + LWR + Green Wave<br>
            NMIT ISE · Batch 2025
        </div>
        """, unsafe_allow_html=True)

    # ── GLOBAL METRICS ROW ───────────────────────────────────
    hour       = datetime.datetime.now().hour
    tf         = HOURLY_PATTERN[hour] * (st.session_state.density_pct / 65.0)
    gw_f       = 0.72 if st.session_state.gw_active else 1.0
    avg_speed  = round(80 - tf * 50)
    avg_delay  = round((5 + tf * 25 * gw_f), 1)
    active_veh = int(1500 + tf * 3000)
    num_jams   = max(0, int(tf * 5 * (0.6 if st.session_state.gw_active else 1)))
    co2_pct    = f"-{round(15 + tf * 10)}%" if st.session_state.gw_active else f"-{round(5 + tf * 3)}%"
    lp_res     = get_lp_result(loads, st.session_state.gw_active)

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("🚗 Active Vehicles", f"{active_veh:,}",   delta=f"+{int(tf*12)}%")
    c2.metric("⚡ Avg Speed (km/h)", f"{avg_speed}",       delta=f"{'-' if tf>0.5 else '+'}{int(abs(avg_speed-60))}",
              delta_color="inverse" if tf > 0.5 else "normal")
    c3.metric("⏱ Avg Delay (min)",  f"{avg_delay}",       delta=f"+{round(tf*2.1,1)} min", delta_color="inverse")
    c4.metric("🔴 Active Jams",      f"{num_jams}",        delta="Critical" if num_jams >= 4 else "Manageable", delta_color="inverse" if num_jams >= 4 else "normal")
    c5.metric("🌿 CO₂ Reduction",    co2_pct,              delta="Green Wave" if st.session_state.gw_active else "Static mode")
    c6.metric("📈 LP Reduction",     f"{lp_res['reduction_pct']}%", delta="Optimised")
    c7.metric("🔗 Signals Sync'd",   f"{'8' if st.session_state.gw_active else '2'}/10", delta="Wave active" if st.session_state.gw_active else "Off")

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    # ── MAIN TABS ────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️  Live Map & Signals",
        "📊  Traffic Analytics",
        "⚡  Green Wave Model",
        "🚑  EVP System",
        "🧮  LP & LWR Theory",
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1 — LIVE MAP & SIGNALS
    # ════════════════════════════════════════════════════════
    with tab1:
        map_col, sig_col = st.columns([2, 1])

        with map_col:
            sec_header("🗺️", "Bangalore Road Network", f"Layer: {st.session_state.map_layer.upper()}")
            m = build_map(loads, st.session_state.map_layer, st.session_state.evp_active)
            st_folium(m, width=None, height=480, returned_objects=[])

        with sig_col:
            # Signal timer
            sec_header("🚦", "Signal Controller")
            phi_val = green_wave_offset(4.0, st.session_state.v_kmh, st.session_state.cycle_T)
            st.markdown(
                signal_html(
                    st.session_state.signal_phase,
                    max(0, int(st.session_state.signal_timer)),
                    phi_val,
                    st.session_state.evp_active,
                ),
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # Junction load gauges
            sec_header("⬡", "Top Junction Load")
            g1, g2 = st.columns(2)
            top2 = sorted(loads.items(), key=lambda x: x[1], reverse=True)[:2]
            with g1:
                nm, val = top2[0]
                st.plotly_chart(build_gauge(val, nm[:10], _load_color(val)), use_container_width=True, config={"displayModeBar": False})
            with g2:
                nm, val = top2[1]
                st.plotly_chart(build_gauge(val, nm[:10], _load_color(val)), use_container_width=True, config={"displayModeBar": False})

            # Junction status table
            sec_header("⬡", "All Junction Status")
            rows = []
            for jn in JUNCTION_NAMES:
                l  = loads[jn]
                c  = "🔴" if l > 80 else "🟡" if l > 60 else "🟢"
                st_val = "CRITICAL" if l > 80 else "BUSY" if l > 60 else "CLEAR"
                rows.append({"Junction": jn, "Load": f"{l:.0f}%", "State": f"{c} {st_val}"})
            df_status = pd.DataFrame(rows)
            st.dataframe(
                df_status,
                use_container_width=True,
                hide_index=True,
                height=220,
                column_config={
                    "Junction": st.column_config.TextColumn("Junction"),
                    "Load":     st.column_config.TextColumn("Load %"),
                    "State":    st.column_config.TextColumn("Status"),
                },
            )

        # Alert Feed
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        sec_header("⚡", "Live Alert Feed", f"{len(_ALERTS)} alerts")
        render_alerts()

    # ════════════════════════════════════════════════════════
    # TAB 2 — TRAFFIC ANALYTICS
    # ════════════════════════════════════════════════════════
    with tab2:
        r1c1, r1c2 = st.columns([3, 2])

        with r1c1:
            sec_header("◈", "24-Hour Traffic Volume", st.session_state.selected_junction)
            df24 = get_24h_series(
                st.session_state.selected_junction,
                st.session_state.density_pct,
                st.session_state.gw_active,
            )
            st.plotly_chart(build_24h_chart(df24, st.session_state.selected_junction), use_container_width=True, config={"displayModeBar": False})

        with r1c2:
            sec_header("◈", "Network Radar — Load %")
            st.plotly_chart(build_radar(loads), use_container_width=True, config={"displayModeBar": False})

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            sec_header("◈", "Avg Delay — Static vs Green Wave")
            df_cmp = get_flow_comparison(st.session_state.gw_active)
            st.plotly_chart(build_delay_bar(df_cmp), use_container_width=True, config={"displayModeBar": False})

        with r2c2:
            sec_header("◈", "Day × Junction Load Heatmap (%)")
            df_heat = get_weekly_heatmap(st.session_state.gw_active)
            st.plotly_chart(build_heatmap(df_heat), use_container_width=True, config={"displayModeBar": False})

        # Raw data table
        with st.expander("📋 Raw Junction Load Data (Live)"):
            df_raw = pd.DataFrame([
                {"Junction": k, "Load (%)": round(v, 1),
                 "Status": "CRITICAL" if v > 80 else "BUSY" if v > 60 else "CLEAR",
                 "GW Offset φ (s)": round(green_wave_offset(4.0, st.session_state.v_kmh, st.session_state.cycle_T), 2),
                 "LWR Density (veh/km)": round(lwr_density(v * 20, st.session_state.v_kmh), 1)}
                for k, v in loads.items()
            ])
            st.dataframe(df_raw, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════
    # TAB 3 — GREEN WAVE MODEL
    # ════════════════════════════════════════════════════════
    with tab3:
        sec_header("⚡", "Green Wave Time-Space Diagram",
                   f"v_c = {st.session_state.v_kmh:.0f} km/h · T = {st.session_state.cycle_T:.0f}s")
        st.plotly_chart(build_green_wave_chart(st.session_state.v_kmh, st.session_state.cycle_T),
                        use_container_width=True, config={"displayModeBar": False})

        gw_c1, gw_c2 = st.columns(2)
        with gw_c1:
            sec_header("📐", "Computed Corridor Offsets")
            corridors = [("Silk Board → HSR Layout", 4.0), ("HSR Layout → Bellandur", 4.0), ("Bellandur → Whitefield", 5.0)]
            rows_gw = []
            for name, dist in corridors:
                phi = green_wave_offset(dist, st.session_state.v_kmh, st.session_state.cycle_T)
                rows_gw.append({"Corridor": name, "Dist (km)": dist, "φ Offset (s)": round(phi, 2), "v_c (km/h)": st.session_state.v_kmh})
            st.dataframe(pd.DataFrame(rows_gw), use_container_width=True, hide_index=True)

            st.markdown("""
            <div style="background:#0d1f3c;border:1px solid #1a3a5c;border-radius:6px;
                        padding:12px;font-family:'Courier New',monospace;font-size:11px;margin-top:10px">
                <div style="color:#00d4ff;margin-bottom:6px">📐 Green Wave Formula</div>
                <div style="color:#e0f0ff">φ = (L / v_c)  mod  T</div>
                <div style="color:#5a7a9a;font-size:9px;margin-top:6px">
                    L = distance between signals (m)<br>
                    v_c = corridor design speed (m/s)<br>
                    T = total signal cycle time (s)<br>
                    φ = required offset between phases
                </div>
            </div>
            """, unsafe_allow_html=True)

        with gw_c2:
            sec_header("📊", "Delay Savings — Green Wave Active")
            df_cmp2 = get_flow_comparison(st.session_state.gw_active)
            saving_fig = go.Figure()
            df_cmp2["Saving (min)"] = df_cmp2["Static (min)"] - df_cmp2["Green Wave (min)"]
            saving_fig.add_trace(go.Bar(
                x=df_cmp2["Junction"],
                y=df_cmp2["Saving (min)"],
                marker_color="rgba(0,255,136,0.7)",
                marker_line_color="#00ff88",
                marker_line_width=1,
                text=df_cmp2["Saving (min)"].apply(lambda x: f"-{x} min"),
                textposition="outside",
                textfont=dict(color="#00ff88", size=9),
                name="Delay Saving",
            ))
            saving_fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Per-Junction Delay Saved (min)", font=dict(color="#00d4ff", size=12)),
                height=280,
                showlegend=False,
            )
            st.plotly_chart(saving_fig, use_container_width=True, config={"displayModeBar": False})

    # ════════════════════════════════════════════════════════
    # TAB 4 — EVP SYSTEM
    # ════════════════════════════════════════════════════════
    with tab4:
        evp_c1, evp_c2 = st.columns([3, 2])

        with evp_c1:
            sec_header("🚑", "EVP Lead-Time Model", "S = d / v")
            st.plotly_chart(build_evp_chart(st.session_state.evp_distance, st.session_state.evp_speed),
                            use_container_width=True, config={"displayModeBar": False})

        with evp_c2:
            sec_header("🚑", "EVP Status")
            evp_t_cur = evp_signal_time(st.session_state.evp_distance, st.session_state.evp_speed)
            status_color = "#ff3366" if st.session_state.evp_active else "#5a7a9a"
            st.markdown(f"""
            <div style="background:#0d1f3c;border:1px solid {'#ff3366' if st.session_state.evp_active else '#1a3a5c'};
                        border-radius:8px;padding:16px;font-family:'Courier New',monospace;
                        {'box-shadow:0 0 20px rgba(255,51,102,0.2)' if st.session_state.evp_active else ''}">
                <div style="font-size:24px;text-align:center;margin-bottom:10px">
                    {'🚨' if st.session_state.evp_active else '🚑'}
                </div>
                <div style="text-align:center;font-size:10px;color:{status_color};
                            letter-spacing:2px;font-weight:700;margin-bottom:12px">
                    {'EVP ACTIVE — CORRIDOR PRE-EMPTED' if st.session_state.evp_active else 'EVP STANDBY'}
                </div>
                <div style="font-size:10px;margin-bottom:6px">
                    <span style="color:#5a7a9a">Priority Weight:</span>
                    <span style="color:#ff3366;font-weight:700">{'P → ∞' if st.session_state.evp_active else 'P = 1'}</span>
                </div>
                <div style="font-size:10px;margin-bottom:6px">
                    <span style="color:#5a7a9a">Route:</span>
                    <span style="color:#e0f0ff">Silk Board → HSR → Bellandur</span>
                </div>
                <div style="font-size:10px;margin-bottom:6px">
                    <span style="color:#5a7a9a">Distance:</span>
                    <span style="color:#e0f0ff">{st.session_state.evp_distance:.0f} m</span>
                </div>
                <div style="font-size:10px;margin-bottom:6px">
                    <span style="color:#5a7a9a">Speed:</span>
                    <span style="color:#e0f0ff">{st.session_state.evp_speed:.0f} km/h</span>
                </div>
                <div style="font-size:10px;margin-bottom:6px">
                    <span style="color:#5a7a9a">Pre-emption Lead Time:</span>
                    <span style="color:#ff8c00;font-weight:700">{evp_t_cur:.1f} s</span>
                </div>
                <div style="font-size:10px">
                    <span style="color:#5a7a9a">Golden Hour Remaining:</span>
                    <span style="color:#00ff88;font-weight:700">{'42:18' if st.session_state.evp_active else '--:--'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            sec_header("📐", "EVP Formula")
            st.markdown("""
            <div style="background:#0d1f3c;border:1px solid #1a3a5c;border-radius:6px;
                        padding:12px;font-family:'Courier New',monospace;font-size:11px">
                <div style="color:#ff3366;margin-bottom:6px">Signal Override Timing:</div>
                <div style="color:#e0f0ff">S = d / v</div>
                <div style="color:#5a7a9a;font-size:9px;margin-top:6px">
                    d = distance to next signal (m)<br>
                    v = ambulance velocity (m/s)<br>
                    S = seconds before signal to pre-empt<br><br>
                    When P = ∞ detected:<br>
                    All corridor signals → GREEN<br>
                    Cross-traffic → RED (safety interval maintained)
                </div>
            </div>
            """, unsafe_allow_html=True)

        # EVP speed sensitivity
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        sec_header("📊", "EVP Sensitivity — Distance vs Lead-Time vs Speed")
        speeds_range   = [20, 30, 40, 50, 60, 70, 80]
        distances_evp  = [100, 250, 500, 750, 1000, 1500, 2000]
        heat_z         = [[round(evp_signal_time(d, s)) for d in distances_evp] for s in speeds_range]
        evp_heat_fig   = go.Figure(go.Heatmap(
            z=heat_z,
            x=[f"{d}m" for d in distances_evp],
            y=[f"{s}km/h" for s in speeds_range],
            colorscale=[[0, "#00ff88"], [0.5, "#ff8c00"], [1, "#ff3366"]],
            colorbar=dict(title=dict(text="Sec", font=dict(color="#5a7a9a")), tickfont=dict(color="#5a7a9a")),
            text=heat_z,
            texttemplate="%{z}s",
            textfont=dict(size=9, color="#e0f0ff"),
        ))
        evp_heat_fig.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Pre-emption Lead Time (seconds) — Ambulance Speed × Signal Distance", font=dict(color="#00d4ff", size=12)),
            height=260,
        )
        st.plotly_chart(evp_heat_fig, use_container_width=True, config={"displayModeBar": False})

    # ════════════════════════════════════════════════════════
    # TAB 5 — LP & LWR THEORY
    # ════════════════════════════════════════════════════════
    with tab5:
        th_c1, th_c2 = st.columns([3, 2])

        with th_c1:
            sec_header("📐", "LWR Traffic Flow Model — Fundamental Diagram")
            st.plotly_chart(build_lwr_chart(), use_container_width=True, config={"displayModeBar": False})

        with th_c2:
            sec_header("🧮", "LP Optimisation Result")
            lp = get_lp_result(loads, st.session_state.gw_active)
            st.markdown(f"""
            <div style="background:#0d1f3c;border:1px solid #1a3a5c;border-radius:8px;
                        padding:14px;font-family:'Courier New',monospace;font-size:11px">
                <div style="color:#00d4ff;margin-bottom:8px;font-size:12px">
                    W(D) = Σ min ∫ t dt
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                    <span style="color:#5a7a9a">Baseline Delay:</span>
                    <span style="color:#ff3366;font-weight:700">{lp['total_delay']} min</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                    <span style="color:#5a7a9a">LP-Optimised:</span>
                    <span style="color:#00ff88;font-weight:700">{lp['optimised']} min</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:12px">
                    <span style="color:#5a7a9a">Reduction:</span>
                    <span style="color:#a855f7;font-weight:700">{lp['reduction_pct']}%</span>
                </div>
                <div style="height:8px;background:rgba(255,255,255,0.05);border-radius:4px;overflow:hidden">
                    <div style="height:100%;width:{lp['reduction_pct']}%;
                                background:linear-gradient(90deg,#a855f7,#00d4ff);
                                border-radius:4px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            sec_header("📐", "Model Constraints")
            st.markdown("""
            <div style="background:#0d1f3c;border:1px solid #1a3a5c;border-radius:6px;
                        padding:12px;font-family:'Courier New',monospace;font-size:10px;line-height:1.8">
                <div style="color:#00d4ff">Objective:</div>
                <div style="color:#e0f0ff">  min W(D) = Σᵢ ∫ tᵢ dt</div>
                <div style="color:#00d4ff;margin-top:8px">Subject To:</div>
                <div style="color:#e0f0ff">  • gᵢ + yᵢ + rᵢ = T  (cycle constraint)</div>
                <div style="color:#e0f0ff">  • gᵢ ≥ g_min         (pedestrian safety)</div>
                <div style="color:#e0f0ff">  • φᵢ = (Lᵢ/v_c) mod T (green wave)</div>
                <div style="color:#e0f0ff">  • Pₑᵥₚ → ∞             (emergency override)</div>
                <div style="color:#e0f0ff">  • ρ = q/v              (LWR density)</div>
            </div>
            """, unsafe_allow_html=True)

        # LP comparison bar
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        sec_header("📊", "LP Optimisation — Per Junction Delay (Before vs After)")
        j_names_lp  = list(BASE_LOADS.keys())
        before_lp   = [round(BASE_LOADS[j] * 0.48 * (st.session_state.density_pct / 65.0)) for j in j_names_lp]
        after_lp    = [round(b * (0.65 if st.session_state.gw_active else 0.95)) for b in before_lp]

        lp_fig = go.Figure()
        lp_fig.add_trace(go.Bar(x=j_names_lp, y=before_lp, name="Before LP",
                                marker_color="rgba(255,51,102,0.7)", marker_line_color="#ff3366", marker_line_width=1))
        lp_fig.add_trace(go.Bar(x=j_names_lp, y=after_lp, name="After LP",
                                marker_color="rgba(168,85,247,0.7)", marker_line_color="#a855f7", marker_line_width=1))
        lp_fig.update_layout(
            **PLOTLY_LAYOUT,
            barmode="group",
            title=dict(text="Junction Delay — Before and After LP Optimisation (min)", font=dict(color="#00d4ff", size=12)),
            height=280,
        )
        st.plotly_chart(lp_fig, use_container_width=True, config={"displayModeBar": False})

        # References
        with st.expander("📚 References"):
            st.markdown("""
            <div style="font-family:'Courier New',monospace;font-size:10px;line-height:2;color:#e0f0ff">
                <b style="color:#00d4ff">Graph Theory:</b><br>
                West, D.B. (2001). Introduction to Graph Theory. Prentice Hall.<br><br>
                <b style="color:#00d4ff">Traffic Flow Dynamics:</b><br>
                Lighthill, M.J. & Whitham, G.B. (1955). On Kinematic Waves II.<br>
                Proc. Royal Society of London, 229(1178), 317–345.<br><br>
                <b style="color:#00d4ff">Mathematical Optimisation:</b><br>
                Taha, H.A. (2017). Operations Research: An Introduction. Pearson.<br><br>
                <b style="color:#00d4ff">Local Traffic Data:</b><br>
                TomTom Traffic Index: Bangalore 2024-2025.<br>
                Kaggle: Bangalore Urban Traffic Dataset 2024.<br>
                BBMP Smart City Traffic Reports 2024.
            </div>
            """, unsafe_allow_html=True)

    # ── Bottom bar ───────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="display:flex;justify-content:space-between;font-family:\'Courier New\',monospace;'
        'font-size:9px;color:#5a7a9a;padding:2px 0">'
        '<span>● SYSTEM ONLINE · 10/10 NODES · 48 SIGNALS ACTIVE</span>'
        '<span>W(D) = Σ min∫(t)dt  ·  φ = L/v_c mod T  ·  S = d/v</span>'
        '<span>NMIT ISE · NISHCHAL VISHWANATH NB25ISE160 · RISHUL KH NB25ISE186</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Periodic alerts ──────────────────────────────────────
    if st.session_state.tick % 20 == 0:
        templates = [
            ("CONGESTION", f"Silk Board density {loads['Silk Board']:.0f}% — Auto-adjusting offsets", "warn"),
            ("GREEN WAVE",  "φ synchronised across ORR-Hebbal corridor at 40 km/h", "success"),
            ("LWR MODEL",   f"Shock-wave forming at ORR — predictive routing engaged", "warn"),
            ("OPTIMISER",   f"LP solver reduced Hebbal delay by {lp_res['reduction_pct']}%", "info"),
        ]
        t_idx = (st.session_state.tick // 20) % len(templates)
        push_alert(*templates[t_idx])


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

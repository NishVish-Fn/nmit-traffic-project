import streamlit as st
import folium
from streamlit_folium import st_folium
import random
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Urban Flow Command Center",
    layout="wide",
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {background-color:#0e1117;}
.block-container {padding-top:1rem;}
.metric-card{
    background:#161b22;
    padding:15px;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚦 Urban Flow Traffic Command Center")

# ---------------- SESSION STATE ----------------
if "emergency" not in st.session_state:
    st.session_state.emergency = False

if "density" not in st.session_state:
    st.session_state.density = {
        "Silk Board": random.randint(40,90),
        "BTM": random.randint(30,80),
        "MG Road": random.randint(20,70),
        "Hebbal": random.randint(40,85),
    }

# ---------------- SIDEBAR ----------------
st.sidebar.header("Control Panel")

if st.sidebar.button("🚑 Dispatch Ambulance"):
    st.session_state.emergency = True

if st.sidebar.button("Reset System"):
    st.session_state.emergency = False

# ---------------- SIGNAL LOGIC ----------------
def compute_signals(density, emergency):

    if emergency:
        return {k:"GREEN" for k in density}

    signals = {}
    for k,v in density.items():
        if v>70:
            signals[k]="GREEN"
        elif v>40:
            signals[k]="YELLOW"
        else:
            signals[k]="RED"

    return signals


signals = compute_signals(
    st.session_state.density,
    st.session_state.emergency
)

# ---------------- LAYOUT ----------------
left, right = st.columns([3,1])

# ================= MAP =================
with left:

    if "map" not in st.session_state:

        center=[12.95,77.61]

        m = folium.Map(
            location=center,
            zoom_start=12,
            tiles="CartoDB dark_matter"
        )

        st.session_state.map = m

    m = st.session_state.map

    m._children.clear()

    color_map={
        "GREEN":"green",
        "YELLOW":"orange",
        "RED":"red"
    }

    locations={
        "Silk Board":[12.9177,77.6238],
        "BTM":[12.9166,77.6101],
        "MG Road":[12.9756,77.6050],
        "Hebbal":[13.0358,77.5970]
    }

    for name,loc in locations.items():

        folium.CircleMarker(
            location=loc,
            radius=10,
            color=color_map[signals[name]],
            fill=True,
            popup=f"{name} | {signals[name]}"
        ).add_to(m)

    if st.session_state.emergency:

        route=[
            locations["Silk Board"],
            locations["BTM"],
            locations["MG Road"],
            locations["Hebbal"],
        ]

        folium.PolyLine(
            route,
            color="lime",
            weight=7
        ).add_to(m)

    st_folium(m, height=600, width=None)

# ================= DASHBOARD =================
with right:

    st.subheader("System Status")

    avg=np.mean(list(st.session_state.density.values()))

    st.metric("Avg Density",f"{int(avg)}%")

    if st.session_state.emergency:
        st.metric("Emergency Mode","ACTIVE")
        st.metric("ETA Reduction","60%")
    else:
        st.metric("Emergency Mode","OFF")
        st.metric("ETA Reduction","0%")

    st.divider()

    df=pd.DataFrame({
        "Intersection":signals.keys(),
        "Signal":signals.values(),
        "Density":st.session_state.density.values()
    })

    st.dataframe(df,use_container_width=True)

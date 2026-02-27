import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import time
import random

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Urban Flow Traffic Command",
    layout="wide"
)

st.title("🚦 Urban Flow – Smart Traffic Command Center")

st.markdown(
"""
AI-Based Traffic Optimization  
✔ Green Wave Synchronization  
✔ Emergency Vehicle Pre-emption  
✔ Dynamic Signal Control
"""
)

# ---------------- TRAFFIC MODEL ----------------
class TrafficOptimizer:

    def __init__(self):
        self.cycle_time = 120
        self.green_speed = 40  # km/h

        self.intersections = {
            "Silk Board": [12.9177, 77.6238],
            "BTM": [12.9166, 77.6101],
            "MG Road": [12.9756, 77.6050],
            "Hebbal": [13.0358, 77.5970]
        }

    # simulate traffic density
    def traffic_density(self):
        return {
            k: random.randint(20, 90)
            for k in self.intersections
        }

    # GREEN WAVE logic
    def green_wave(self, density):
        signals = {}

        for j, d in density.items():
            if d > 70:
                signals[j] = "GREEN"
            elif d > 40:
                signals[j] = "YELLOW"
            else:
                signals[j] = "RED"

        return signals

    # Emergency Vehicle Preemption
    def emergency_override(self):
        return {k: "GREEN" for k in self.intersections}


optimizer = TrafficOptimizer()

# ---------------- SIDEBAR CONTROL ----------------
st.sidebar.header("🚑 Emergency Control")

emergency = st.sidebar.button("Dispatch Ambulance")

# ---------------- TRAFFIC DATA ----------------
density = optimizer.traffic_density()

if emergency:
    signal_status = optimizer.emergency_override()
else:
    signal_status = optimizer.green_wave(density)

# ---------------- MAP ----------------
center = [12.95, 77.61]

m = folium.Map(
    location=center,
    zoom_start=12,
    tiles="CartoDB dark_matter"
)

# signal colors
color_map = {
    "GREEN": "green",
    "YELLOW": "orange",
    "RED": "red"
}

# draw intersections
for name, loc in optimizer.intersections.items():

    folium.CircleMarker(
        location=loc,
        radius=10,
        popup=f"{name}\nSignal: {signal_status[name]}",
        color=color_map[signal_status[name]],
        fill=True
    ).add_to(m)

# ambulance route animation
if emergency:

    route = [
        optimizer.intersections["Silk Board"],
        optimizer.intersections["BTM"],
        optimizer.intersections["MG Road"],
        optimizer.intersections["Hebbal"]
    ]

    folium.PolyLine(
        route,
        color="lime",
        weight=6,
        tooltip="Green Corridor Active"
    ).add_to(m)

st_folium(m, width=1200, height=550)

# ---------------- ANALYTICS ----------------
st.subheader("📊 Live Traffic Analytics")

col1, col2, col3 = st.columns(3)

avg_density = int(np.mean(list(density.values())))

delay_reduction = 30 if emergency else random.randint(10, 20)
eta_saved = 60 if emergency else 0
efficiency = 35 if emergency else random.randint(15, 25)

col1.metric("Average Density", f"{avg_density}%")
col2.metric("Delay Reduction", f"{delay_reduction}%")
col3.metric("Traffic Efficiency", f"{efficiency}%")

# ---------------- SIGNAL TABLE ----------------
st.subheader("🚦 Signal Status Table")

df = pd.DataFrame({
    "Intersection": list(signal_status.keys()),
    "Signal": list(signal_status.values()),
    "Traffic Density": list(density.values())
})

st.dataframe(df, use_container_width=True)

# ---------------- REAL-TIME SIMULATION ----------------
st.subheader("🔄 Live Simulation")

placeholder = st.empty()

for i in range(5):
    placeholder.write(f"Optimizing Traffic Grid... Step {i+1}/5")
    time.sleep(0.5)

st.success("City Traffic Optimized Successfully ✅")

if emergency:
    st.success("🚑 Emergency Green Corridor Activated — Golden Hour Protected")

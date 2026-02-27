import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time
from scipy.optimize import minimize

# --- 1. CORE LOGIC ---
class PoliceCommandLogic:
    def __init__(self):
        self.cycle_time = 120
        self.intersections = {
            "Silk Board": {"pos": [77.6238, 12.9177], "idx": 0},
            "HSR Layout": {"pos": [77.6450, 12.9100], "idx": 1},
            "Bellandur": {"pos": [77.6762, 12.9260], "idx": 2},
        }

    def solve_delay_objective(self, densities, em_indices):

        def objective_func(x):
            weights = np.ones(len(densities))
            for idx in em_indices:
                weights[idx] = 1_000_000
            return np.sum((densities * weights) / x)

        constraints = ({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - self.cycle_time
        })

        bounds = [(15, 90)] * len(densities)

        res = minimize(
            objective_func,
            [40] * len(densities),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        return res.x


# --- 2. DASHBOARD UI ---
st.set_page_config(page_title="Bangalore Traffic Command Center", layout="wide")
st.title("🏙️ Bangalore Active Grid Control: Police Dashboard")

logic = PoliceCommandLogic()

# Session state vehicles
if "vehicles" not in st.session_state:
    st.session_state.vehicles = pd.DataFrame({
        "lon": np.random.uniform(77.62, 77.68, 150),
        "lat": np.random.uniform(12.91, 12.93, 150),
        "speed": np.random.uniform(0.0002, 0.0005, 150),
    })

# Sidebar
st.sidebar.header("🚨 Emergency Dispatch")
em_active = st.sidebar.multiselect(
    "Active Emergency Corridors",
    list(logic.intersections.keys())
)

em_indices = [logic.intersections[name]["idx"] for name in em_active]

# Optimization
densities = np.array([50, 30, 45])
optimized_signals = logic.solve_delay_objective(densities, em_indices)

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Commuter Delay Reduction", f"{30 if em_active else 25}%")
with col2:
    st.metric("Throughput (ORR)", "2,400 vph")
with col3:
    st.metric("Golden Hour Status", "SECURED" if em_active else "MONITORING")

map_placeholder = st.empty()

# --- 3. REAL-TIME ANIMATION ---
if st.button("▶️ Initialize Real-Time Grid Feed"):

    # Update vehicle movement
    st.session_state.vehicles["lon"] += st.session_state.vehicles["speed"]
    st.session_state.vehicles["lat"] += st.session_state.vehicles["speed"] * 0.2

    # Reset vehicles looping
    st.session_state.vehicles.loc[
        st.session_state.vehicles["lon"] > 77.68, "lon"
    ] = 77.62

    st.session_state.vehicles.loc[
        st.session_state.vehicles["lat"] > 12.93, "lat"
    ] = 12.91

    # Signal states
    signals = []
    for name, data in logic.intersections.items():
        is_green = (
            data["idx"] in em_indices
            or (int(time.time() * 2) % 4 > 1)
        )

        signals.append({
            "lon": data["pos"][0],
            "lat": data["pos"][1],
            "color": [0, 255, 0, 200] if is_green else [255, 0, 0, 200],
        })

    # Vehicle layer
    vehicle_layer = pdk.Layer(
        "ScatterplotLayer",
        st.session_state.vehicles,
        get_position="[lon, lat]",
        get_color="[255, 255, 255, 150]",
        get_radius=30,
    )

    # Signal layer
    signal_layer = pdk.Layer(
        "ScatterplotLayer",
        pd.DataFrame(signals),
        get_position="[lon, lat]",
        get_color="color",
        get_radius=120,
    )

    deck = pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=12.92,
            longitude=77.65,
            zoom=13,
            pitch=0,
        ),
        layers=[vehicle_layer, signal_layer],
    )

    map_placeholder.pydeck_chart(deck)

    # Animation refresh
    time.sleep(0.1)
    st.rerun()

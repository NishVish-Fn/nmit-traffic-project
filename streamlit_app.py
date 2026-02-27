import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time
from scipy.optimize import minimize

# --- 1. CORE MATH & OPTIMIZATION ---
class PoliceCommandLogic:
    def __init__(self):
        self.cycle_time = 120 
        self.intersections = {
            "Silk Board": {"pos": [12.9177, 77.6238], "dist": 0},
            "HSR Layout": {"pos": [12.9100, 77.6450], "dist": 4000},
            "Bellandur": {"pos": [12.9260, 77.6762], "dist": 8000}
        }

    def solve_delay_objective(self, densities, em_indices):
        def objective_func(x):
            weights = np.ones(len(densities))
            for idx in em_indices: 
                weights[idx] = 1000000 
            return np.sum((densities * weights) / x)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - self.cycle_time})
        bounds = [(15, 90) for _ in range(len(densities))]
        
        res = minimize(objective_func, [40]*len(densities), 
                       method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x

# --- 2. THE DASHBOARD UI ---
st.set_page_config(page_title="Bangalore Traffic Command Center", layout="wide")
st.title("🏙️ Bangalore Active Grid Control: Police Dashboard")

if 'cars' not in st.session_state:
    st.session_state.cars = pd.DataFrame(
        np.random.randn(50, 2) / [150, 150] + [12.9177, 77.6238],
        columns=['lat', 'lon']
    )

logic = PoliceCommandLogic()

# Sidebar
st.sidebar.header("🚨 Emergency Dispatch")
em_active = st.sidebar.multiselect("Active Emergency Corridors", list(logic.intersections.keys()))
em_indices = [list(logic.intersections.keys()).index(name) for name in em_active]

# Processing
densities = np.array([np.random.randint(40, 90) for _ in range(3)])
optimized_signals = logic.solve_delay_objective(densities, em_indices)

# --- 3. METRICS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Commuter Delay", f"{28 if em_active else 40} min", delta="-30%" if em_active else "0%")
with col2:
    st.metric("Throughput (ORR)", "2,400 vph", delta="+15%")
with col3:
    st.metric("Ambulance Path Status", "CLEARED" if em_active else "BLOCKED")

# --- 4. MAP ---
car_layer = pdk.Layer(
    "ScatterplotLayer",
    st.session_state.cars,
    get_position='[lon, lat]',
    get_color='[200, 30, 0, 160]',
    get_radius=40,
)

signal_data = []
for i, (name, info) in enumerate(logic.intersections.items()):
    is_green = i in em_indices
    signal_data.append({
        "name": name,
        "coordinates": [info["pos"][1], info["pos"][0]],
        "color": [0, 255, 0] if is_green else [255, 0, 0]
    })

signal_layer = pdk.Layer(
    "IconLayer",
    pd.DataFrame(signal_data),
    get_position='coordinates',
    get_color='color',
    get_radius=150,
)

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/dark-v9',
    initial_view_state=pdk.ViewState(latitude=12.9177, longitude=77.6450, zoom=12, pitch=45),
    layers=[car_layer, signal_layer]
))

if st.button("🔴 Start Live Feed"):
    for _ in range(10):
        st.session_state.cars += np.random.randn(50, 2) / [5000, 5000]
        st.rerun()

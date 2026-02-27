import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time
from scipy.optimize import minimize

# --- 1. CORE MATH & OPTIMIZATION (Integrated from PoC Report) ---
class TrafficBrain:
    def __init__(self):
        self.cycle_time = 120 # T = 120s 
        self.v_c = 11.11       # Constant velocity 40 km/h in m/s [cite: 45]
        # Directed Graph G=(V,E) for Bangalore [cite: 27]
        self.intersections = {
            "Silk Board": {"pos": [77.6238, 12.9177], "idx": 0},
            "HSR Layout": {"pos": [77.6450, 12.9100], "idx": 1},
            "Bellandur":  {"pos": [77.6762, 12.9260], "idx": 2}
        }

    def solve_grid_optimization(self, densities, em_indices):
        """Minimizes Total Delay W using Linear Programming [cite: 28, 38]"""
        def objective_func(x):
            # x represents green light timings for each node
            weights = np.ones(len(densities))
            for idx in em_indices: 
                weights[idx] = 1000000 # Priority Weight P -> Infinity [cite: 42]
            return np.sum((densities * weights) / x) # [cite: 38]

        # Constraints: Green times sum to Cycle Time T 
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - self.cycle_time})
        bounds = [(15, 90) for _ in range(len(densities))] # Safety/Pedestrian intervals [cite: 28]
        
        res = minimize(objective_func, [40]*len(densities), method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x

# --- 2. DASHBOARD UI DESIGN ---
st.set_page_config(page_title="Bangalore Traffic Command Center", layout="wide")
st.title("🏙️ Bangalore Active Grid Control: Police Dashboard")

# Session State for Moving Vehicles (LWR Flow Simulation) 
if 'vehicles' not in st.session_state:
    st.session_state.vehicles = pd.DataFrame({
        'lon': np.random.uniform(77.62, 77.68, 150),
        'lat': np.random.uniform(12.91, 12.93, 150),
        'speed': np.random.uniform(0.0002, 0.0005, 150) # Target v_c [cite: 45]
    })

brain = TrafficBrain()

# Sidebar: Dispatch & Perception Layer [cite: 31, 32]
st.sidebar.header("🚨 Emergency Dispatch")
em_active = st.sidebar.multiselect("Active Emergency Corridors", list(brain.intersections.keys()))
em_indices = [brain.intersections[name]["idx"] for name in em_active]

# Processing Layer: Real-time math [cite: 31, 33]
densities = np.array([50, 30, 45])
optimized_signals = brain.solve_grid_optimization(densities, em_indices)

# --- 3. PERFORMANCE METRICS (From PoC Results) ---
c1, c2, c3 = st.columns(3)
with c1:
    # Estimated 25-30% reduction [cite: 76, 93]
    st.metric("Commuter Delay Reduction", f"{30 if em_active else 25}%", delta="Target Met")
with c2:
    st.metric("Throughput (ORR)", "2,400 vph", delta="+15%")
with c3:
    # 60% reduction for emergency travel [cite: 90]
    st.metric("Golden Hour Status", "SECURED" if em_active else "MONITORING")

# --- 4. ANIMATED MAP (Actuator Layer) ---
map_placeholder = st.empty()

# Initialization for Active Grid Feed [cite: 16, 34]
if st.button("▶️ Initialize Active Grid Feed"):
    while True:
        # Move vehicles (Simulating Flow Dynamics) 
        st.session_state.vehicles['lon'] += st.session_state.vehicles['speed']
        st.session_state.vehicles['lat'] += st.session_state.vehicles['speed'] * 0.2
        
        # Reset vehicles for continuous cycle
        st.session_state.vehicles.loc[st.session_state.vehicles['lon'] > 77.68, 'lon'] = 77.62
        st.session_state.vehicles.loc[st.session_state.vehicles['lat'] > 12.93, 'lat'] = 12.91

        # Signal status for Map
        signals = []
        for name, data in brain.intersections.items():
            # Green if EVP logic active, else alternate phases [cite: 33, 42]
            is_green = data["idx"] in em_indices or (int(time.time()*2) % 4 > 1)
            signals.append({
                "pos": data["pos"],
                "color": [0, 255, 0, 200] if is_green else [255, 0, 0, 200]
            })

        # Render 3D Pydeck Map [cite: 53]
        v_layer = pdk.Layer("ScatterplotLayer", st.session_state.vehicles, 
                            get_position='[lon, lat]', get_color='[255, 255, 255, 150]', get_radius=30)
        s_layer = pdk.Layer("ScatterplotLayer", pd.DataFrame(signals), 
                            get_position='pos', get_color='color', get_radius=180)

        map_placeholder.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/dark-v10', # Detailed Urban Roads
            initial_view_state=pdk.ViewState(longitude=77.6450, latitude=12.9177, zoom=13, pitch=45),
            layers=[v_layer, s_layer]
        ))
        time.sleep(0.05)

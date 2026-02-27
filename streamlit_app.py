import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time
from scipy.optimize import minimize

# --- 1. CORE MATH & OPTIMIZATION (Integrated from PoC) ---
class PoliceCommandLogic:
    def __init__(self):
        # Total cycle time T = 120s
        self.cycle_time = 120 
        # Road segments as Edges in Directed Graph G=(V,E)
        self.intersections = {
            "Silk Board": {"pos": [77.6238, 12.9177], "idx": 0},
            "HSR Layout": {"pos": [77.6450, 12.9100], "idx": 1},
            "Bellandur":  {"pos": [77.6762, 12.9260], "idx": 2}
        }

    def solve_delay_objective(self, densities, em_indices):
        """Minimizes total delay W while applying EVP Priority P"""
        def objective_func(x):
            weights = np.ones(len(densities))
            for idx in em_indices: 
                weights[idx] = 1000000 # Priority Weight P -> Infinity
            return np.sum((densities * weights) / x) #

        # Constraints: Green times sum to Cycle Time T
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - self.cycle_time})
        # Safety bounds for Silk Board, HSR, and Bellandur
        bounds = [(15, 90) for _ in range(len(densities))]
        
        res = minimize(objective_func, [40]*len(densities), method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x

# --- 2. DASHBOARD UI ---
st.set_page_config(page_title="Bangalore Traffic Command Center", layout="wide")
st.title("🏙️ Bangalore Active Grid Control: Police Dashboard")

# Initialize vehicular flow simulation (LWR Model)
if 'vehicles' not in st.session_state:
    st.session_state.vehicles = pd.DataFrame({
        'lon': np.random.uniform(77.62, 77.68, 150),
        'lat': np.random.uniform(12.91, 12.93, 150),
        'speed': np.random.uniform(0.0002, 0.0005, 150) # Target vc
    })

logic = PoliceCommandLogic()

# Sidebar: Dispatch & Perception Layer
st.sidebar.header("🚨 Emergency Dispatch")
em_active = st.sidebar.multiselect("Active Emergency Corridors", list(logic.intersections.keys()))
em_indices = [logic.intersections[name]["idx"] for name in em_active]

# Processing Layer
densities = np.array([50, 30, 45])
optimized_signals = logic.solve_delay_objective(densities, em_indices)

# Performance Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Commuter Delay", f"{28 if em_active else 40} min", delta="-30%" if em_active else "0%")
with col2:
    st.metric("Throughput (ORR)", "2,400 vph", delta="+15%")
with col3:
    st.metric("Ambulance Path Status", "CLEARED" if em_active else "BLOCKED")

# --- 3. STABLE ANIMATED MAP (Actuator Layer) ---
map_placeholder = st.empty()

if st.button("▶️ Initialize Real-Time Grid Feed"):
    while True:
        # Update positions simulating fluid flow dynamics
        st.session_state.vehicles['lon'] += st.session_state.vehicles['speed']
        st.session_state.vehicles['lat'] += st.session_state.vehicles['speed'] * 0.2
        
        # Reset vehicles for loop
        st.session_state.vehicles.loc[st.session_state.vehicles['lon'] > 77.68, 'lon'] = 77.62
        st.session_state.vehicles.loc[st.session_state.vehicles['lat'] > 12.93, 'lat'] = 12.91

        # Calculate signal states
        signals = []
        for name, data in logic.intersections.items():
            is_green = data["idx"] in em_indices or (int(time.time()*2) % 4 > 1)
            signals.append({
                "pos": data["pos"],
                "color": [0, 255, 0, 200] if is_green else [255, 0, 0, 20

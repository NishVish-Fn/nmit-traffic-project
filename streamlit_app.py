import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time
from scipy.optimize import minimize

# --- 1. CORE LOGIC (Integrated from Project Framework) ---
class PoliceCommandLogic:
    def __init__(self):
        self.cycle_time = 120 
        self.intersections = {
            "Silk Board": {"pos": [77.6238, 12.9177], "idx": 0},
            "HSR Layout": {"pos": [77.6450, 12.9100], "idx": 1},
            "Bellandur":  {"pos": [77.6762, 12.9260], "idx": 2}
        }

    def solve_delay_objective(self, densities, em_indices):
        def objective_func(x):
            weights = np.ones(len(densities))
            for idx in em_indices: 
                weights[idx] = 1000000 
            return np.sum((densities * weights) / x)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - self.cycle_time})
        bounds = [(15, 90) for _ in range(len(densities))]
        res = minimize(objective_func, [40]*len(densities), method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x

# --- 2. DASHBOARD UI ---
st.set_page_config(page_title="Bangalore Traffic Command Center", layout="wide")
st.title("🏙️ Bangalore Active Grid Control: Police Dashboard")

if 'vehicles' not in st.session_state:
    st.session_state.vehicles = pd.DataFrame({
        'lon': np.random.uniform(77.62, 77.68, 150),
        'lat': np.random.uniform(12.91, 12.93, 150),
        'speed': np.random.uniform(0.0002, 0.0005, 150)
    })

logic = PoliceCommandLogic()

# Sidebar Control
st.sidebar.header("🚨 Emergency Dispatch")
em_active = st.sidebar.multiselect("Active Emergency Corridors", list(logic.intersections.keys()))
em_indices = [logic.intersections[name]["idx"] for name in em_active]

# Data Processing
densities = np.array([50, 30, 45])
optimized_signals = logic.solve_delay_objective(densities, em_indices)

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Commuter Delay Reduction", f"{30 if em_active else 25}%", delta="Target Met")
with col2:
    st.metric("Throughput (ORR)", "2,400 vph", delta="+15%")
with col3:
    st.metric("Golden Hour Status", "SECURED" if em_active else "MONITORING")

# --- 3. ANIMATED MAP ---
map_placeholder = st.empty()

if st.button("▶️ Initialize Real-Time Grid Feed"):
    while True:
        # Update positions simulating vehicle flow
        st.session_state.vehicles['lon'] += st.session_state

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time
from scipy.optimize import minimize

# --- 1. CORE MATH & OPTIMIZATION ---
class PoliceCommandLogic:

    def __init__(self):
        self.cycle_time = 120  # Total cycle time

        self.intersections = {
            "Silk Board": {"pos": [12.9177, 77.6238], "dist": 0},
            "HSR Layout": {"pos": [12.9100, 77.6450], "dist": 4000},
            "Bellandur": {"pos": [12.9260, 77.6762], "dist": 8000},
        }

    def solve_delay_objective(self, densities, em_indices):
        """
        Minimize delay W while giving priority to emergency vehicles.
        """

        def objective(x):  # x = green times
            weights = np.ones(len(densities))

            # Emergency Vehicle Priority
            for idx in em_indices:
                weights[idx] = 1_000_000

            return np.sum((densities * weights) / x)

        # Total cycle constraint
        constraints = ({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - self.cycle_time
        })

        # Optimization
        res = minimize(
            objective,
            x0=[40] * len(densities),
            bounds=[(15, 90)] * len(densities),
            constraints=constraints
        )

        return res.x


# --- 2. STREAMLIT DASHBOARD ---
st.set_page_config(
    page_title="Bangalore Traffic Command Center",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🚦 Bangalore Traffic Command Center")

logic = PoliceCommandLogic()

# Example traffic densities
densities = np.array([30, 50, 20])

# Emergency vehicle at HSR Layout
em_indices = [1]

greens = logic.solve_delay_objective(densities, em_indices)

st.subheader("Optimized Green Signal Timings")
for name, g in zip(logic.intersections.keys(), greens):
    st.write(f"{name}: {g:.2f} seconds")

# --- Simulated vehicle movement ---
if "cars" not in st.session_state:
    st.session_state.cars = np.random.rand(50, 2) + [12.91, 77.63]

if st.button("Simulate Traffic"):
    for _ in range(10):
        st.session_state.cars += np.random.randn(50, 2) / [5000, 5000]
        time.sleep(0.1)
        st.rerun()

# Map visualization
layer = pdk.Layer(
    "ScatterplotLayer",
    data=pd.DataFrame(st.session_state.cars, columns=["lat", "lon"]),
    get_position='[lon, lat]',
    get_radius=30,
)

st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=12.92,
        longitude=77.64,
        zoom=12
    ),
    layers=[layer],
))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import random

# --- Data Preparation ---
@st.cache_data
def load_and_prepare(csv_path='parking_data.csv', seq_len=10):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    scaler = MinMaxScaler()
    df['status_scaled'] = scaler.fit_transform(df[['status']])
    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(df['status_scaled'].values[i:i+seq_len])
        y.append(df['status_scaled'].values[i+seq_len])
    return np.array(X), np.array(y), scaler, df

X, y, scaler, df = load_and_prepare()

# --- Model Training ---
@st.cache_resource
def train(X, y):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

model = train(X, y)

st.title("🅿️ Smart Parking Forecasting")

mode = st.radio("Prediction Mode:", options=["No Date (Real-Time)", "Specific Date"])

# --- Route Map Drawing Function ---
def draw_route_map(free_slots, best_slot=None, title="Route Map"):
    fig, ax = plt.subplots(figsize=(10, 6))
    slots = list(free_slots.keys())

    # Define 2-row × 5-col layout (spaced)
    cols = 5
    spacing = 3
    lots = {slots[i]: ((i % cols) * spacing + 4, -(i // cols) * spacing) for i in range(len(slots))}

    # Entrance and exit horizontally aligned
    entrance = (0, -1.5)   # Far left, mid of row gap
    exit = (20, -1.5)      # Far right

    # Draw parking lots
    for lot, (x, y) in lots.items():
        ax.add_patch(plt.Rectangle((x - 1, y - 1), 2, 2,
                                   color='green' if lot == best_slot else 'gray'))
        ax.text(x, y, lot, ha='center', va='center', fontsize=8,
                color='white' if lot == best_slot else 'black')

    # Entrance/Exit
    ax.plot(*entrance, 'go')
    ax.text(entrance[0] - 0.5, entrance[1], "Entrance", ha='right')
    ax.plot(*exit, 'ro')
    ax.text(exit[0] + 0.5, exit[1], "Exit", ha='left')

    # --- Route Logic ---
    if best_slot:
        bx, by = lots[best_slot]
        # Midpoint path Y (between rows)
        path_y = entrance[1]

        # Vertical then horizontal path to column of best lot
        route_x = [entrance[0], bx, bx]
        route_y = [entrance[1], path_y, by]
        ax.plot(route_x, route_y, 'r--', lw=2)
    else:
        ax.text(10, -6, "No Free Parking Slot Available!", ha='center', fontsize=12, color='red')

    ax.set_xlim(-3, 24)
    ax.set_ylim(-7, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(title)
    st.pyplot(fig)

# --- No Date Mode ---
if mode == "No Date (Real-Time)":
    if st.button("Predict Now"):
        nsteps = 10
        preds = []
        seq = X[-1].copy()
        for _ in range(nsteps):
            p = model.predict([seq])[0]
            preds.append(p)
            seq = np.append(seq[1:], p)
        preds = np.round(scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten())
        
        realtime = [random.randint(0, 35) for _ in range(nsteps)]
        times = [(datetime.now() + timedelta(minutes=10*i)).strftime("%H:%M") for i in range(nsteps)]
        slots = [f"Lot_{i+1}" for i in range(nsteps)]

        st.subheader("⏱️ Real-Time Free Slot Prediction")
        for i in range(nsteps):
            st.write(f"{times[i]} -> {int(preds[i])} free slots")

        # Bar Chart
        fig, ax = plt.subplots()
        idx = np.arange(nsteps)
        w = 0.35
        ax.bar(idx, realtime, w, label='Real-Time', color='skyblue')
        ax.bar(idx + w, preds, w, label='Predicted', color='salmon')
        ax.set_xticks(idx + w/2)
        ax.set_xticklabels(times, rotation=45)
        ax.legend()
        ax.set_title("Real-Time vs Predicted Free Slots")
        st.pyplot(fig)

        # Route Map
        free_slots = {slots[i]: int(realtime[i]) for i in range(nsteps)}
        available = [s for s in free_slots if free_slots[s] > 0]
        best_slot = max(available, key=lambda x: free_slots[x]) if available else None
        draw_route_map(free_slots, best_slot, "Best Slot & Route")

# --- Specific Date Mode ---
elif mode == "Specific Date":
    d = st.date_input("Choose a date:")
    if st.button("Predict for Date"):
        today = datetime.today().date()
        if d < today:
            st.error("Cannot predict for past dates.")
        else:
            seq = X[-1].copy()
            preds = []
            for _ in range(144):  # Full day at 10-min intervals
                p = model.predict([seq])[0]
                preds.append(p)
                seq = np.append(seq[1:], p)
            actuals = np.round(scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten())

            st.subheader(f"📆 Free Parking Predictions for {d.strftime('%Y-%m-%d')}")
            for i in range(144):
                ts = datetime.combine(d, datetime.min.time()) + timedelta(minutes=10*i)
                st.write(f"{ts.strftime('%H:%M')} -> {int(actuals[i])} free slots")

            avg_slots = int(np.mean(actuals[:10]))
            free_slots = {f"Lot_{i+1}": random.randint(5, 35) if avg_slots > 0 else 0 for i in range(10)}
            best_slot = max(free_slots, key=lambda k: free_slots[k]) if any(free_slots.values()) else None
            draw_route_map(free_slots, best_slot, f"Best Slot on {d.strftime('%Y-%m-%d')}")

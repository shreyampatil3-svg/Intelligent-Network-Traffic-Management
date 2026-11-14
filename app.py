"""Chunk 5: Streamlit dashboard.
Needs these files in the same folder: model.keras, scaler.pkl, traffic.csv, rules.py
(results.csv is optional and shows the model comparison table).
Run: streamlit run app.py      (needs: pip install streamlit tensorflow pandas joblib)
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

from rules import decide

WINDOW = 30
FEATURES = ["packets_per_sec", "bandwidth_mbps", "latency_ms", "packet_loss_pct"]
COLORS = {"NORMAL": "success", "ELEVATED": "info", "HIGH": "warning", "CRITICAL": "error"}

st.set_page_config(page_title="Congestion Predictor", layout="wide")
st.title("AI Network Congestion Predictor")
st.caption("1D-CNN predicts congestion 10 seconds ahead, and a rule engine suggests traffic-management actions.")


@st.cache_resource
def load_assets():
    return load_model("model.keras"), joblib.load("scaler.pkl")


@st.cache_data
def load_traffic():
    df = pd.read_csv("traffic.csv")
    return df.iloc[int(len(df) * 0.85):].reset_index(drop=True)  # unseen test period


model, scaler = load_assets()
df = load_traffic()


def predict(t):
    window = df[FEATURES].iloc[t - WINDOW:t].values
    x = scaler.transform(window)[None].astype("float32")
    return float(model.predict(x, verbose=0)[0, 0])


# ---------------- Sidebar controls ----------------
st.sidebar.header("Simulation controls")
start = st.sidebar.slider("Start second", WINDOW, len(df) - 1, WINDOW)
steps = st.sidebar.slider("Seconds to simulate", 20, 400, 150)
delay = st.sidebar.slider("Delay per second (s)", 0.0, 1.0, 0.1)
apply_actions = st.sidebar.checkbox("Apply management actions", value=True)
run = st.sidebar.button("Start live simulation")

# ---------------- Model comparison table ----------------
if os.path.exists("results.csv"):
    with st.expander("Model comparison (test set)", expanded=not run):
        st.dataframe(pd.read_csv("results.csv"), hide_index=True)

if not run:
    st.info("Press **Start live simulation** in the sidebar. Tip: pick a start second near a traffic burst for a better demo.")
    st.stop()

# ---------------- Live loop ----------------
holder = st.empty()
history = []
for t in range(start, min(start + steps, len(df))):
    prob = predict(t)
    now = df.iloc[t - 1]
    result = decide(prob, now.latency_ms, now.packet_loss_pct)

    cut = result["load_reduction_pct"] if apply_actions else 0
    history.append({
        "Without action (Mbps)": now.bandwidth_mbps,
        "With action (Mbps)": now.bandwidth_mbps * (1 - cut / 100),
        "Congestion probability": prob,
    })
    hist = pd.DataFrame(history).tail(120)

    with holder.container():
        banner = getattr(st, COLORS[result["state"]])
        banner(f"**State: {result['state']}**  |  {result['action']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Congestion probability", f"{prob * 100:.0f}%")
        c2.metric("Bandwidth", f"{now.bandwidth_mbps:.1f} Mbps")
        c3.metric("Latency", f"{now.latency_ms:.0f} ms")
        c4.metric("Packet loss", f"{now.packet_loss_pct:.1f} %")

        left, right = st.columns(2)
        left.subheader("Traffic load: before vs after action")
        left.line_chart(hist[["Without action (Mbps)", "With action (Mbps)"]])
        right.subheader("Predicted congestion probability")
        right.line_chart(hist[["Congestion probability"]])

    time.sleep(delay)

st.success("Simulation finished. Change the start second in the sidebar and run again.")

"""Chunk 1: Network traffic simulator.
Creates traffic.csv with one row per second:
packets_per_sec, bandwidth_mbps, latency_ms, packet_loss_pct, congested (0/1)
Run: python simulator.py
"""
import numpy as np
import pandas as pd

SECONDS = 30000          # length of the simulation
CAPACITY = 100.0         # link capacity in Mbps
rng = np.random.default_rng(42)

t = np.arange(SECONDS)

# 1) Base load: slow "daily" wave + random noise
base = 45 + 20 * np.sin(2 * np.pi * t / 6000) + rng.normal(0, 4, SECONDS)

# 2) Random bursts (sudden spikes that last 20-120 seconds)
burst = np.zeros(SECONDS)
for _ in range(60):
    start = rng.integers(0, SECONDS - 150)
    length = rng.integers(20, 120)
    burst[start:start + length] += rng.uniform(25, 50)

demand = np.clip(base + burst, 5, None)          # offered load in Mbps
bandwidth = np.minimum(demand, CAPACITY * 0.98)  # link cannot exceed capacity
util = demand / CAPACITY                         # utilisation (can exceed 1)

# 3) Queueing effect: latency and loss rise sharply near full capacity
latency = 10 + 90 * np.clip(util, 0, 1.2) ** 4 + rng.normal(0, 3, SECONDS)
loss = np.clip((util - 0.85) * 25, 0, None) + np.abs(rng.normal(0, 0.3, SECONDS))
packets = bandwidth * 125 + rng.normal(0, 60, SECONDS)  # ~packets/sec

df = pd.DataFrame({
    "packets_per_sec": packets.round(0),
    "bandwidth_mbps": bandwidth.round(2),
    "latency_ms": np.clip(latency, 5, None).round(2),
    "packet_loss_pct": loss.round(2),
})

# 4) Label: congested when latency and loss are both high (noisy, not a pure rule on one column)
df["congested"] = ((df.latency_ms > 60) | (df.packet_loss_pct > 2)).astype(int)

df.to_csv("traffic.csv", index=False)
print(df.head())
print("\nRows:", len(df), "| Congested %:", round(df.congested.mean() * 100, 1))

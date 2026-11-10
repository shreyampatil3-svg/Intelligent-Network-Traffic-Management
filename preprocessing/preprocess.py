"""Chunk 2: Preprocess traffic.csv into CNN-ready windows.
Input : traffic.csv (from simulator.py)
Output: data.npz  (X_train, y_train, X_val, y_val, X_test, y_test) and scaler.pkl
Run: python preprocess.py     (needs: pip install scikit-learn joblib)
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

WINDOW = 30    # model sees the last 30 seconds
HORIZON = 10   # and predicts congestion in the NEXT 10 seconds

df = pd.read_csv("traffic.csv")
features = ["packets_per_sec", "bandwidth_mbps", "latency_ms", "packet_loss_pct"]
values = df[features].values
labels = df["congested"].values

# 1) Split by TIME (not random) so the test set is truly "the future"
n = len(df)
i_train, i_val = int(n * 0.70), int(n * 0.85)

# 2) Scale using the training part only (prevents data leakage)
scaler = MinMaxScaler().fit(values[:i_train])
values = scaler.transform(values)

# 3) Sliding windows: X = 30 seconds of traffic, y = 1 if congestion occurs in next 10 s
X, y = [], []
for s in range(n - WINDOW - HORIZON):
    X.append(values[s:s + WINDOW])
    y.append(int(labels[s + WINDOW:s + WINDOW + HORIZON].max()))
X, y = np.array(X, dtype="float32"), np.array(y, dtype="int64")

# 4) Chronological split of the windows
a, b = i_train - WINDOW, i_val - WINDOW
data = dict(X_train=X[:a], y_train=y[:a], X_val=X[a:b], y_val=y[a:b],
            X_test=X[b:], y_test=y[b:])

np.savez("data.npz", **data)
joblib.dump(scaler, "scaler.pkl")

for k in ("train", "val", "test"):
    print(f"{k:5s} X{data['X_' + k].shape}  congested-soon: {data['y_' + k].mean() * 100:.1f}%")

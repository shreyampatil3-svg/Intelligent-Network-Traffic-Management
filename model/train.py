"""Chunk 3: Train a 1D-CNN and compare it with two baselines.
Input : data.npz (from preprocess.py)
Output: model.keras, results.csv
Run: python train_model.py     (needs: pip install tensorflow scikit-learn pandas)
Tip: this runs fine on Google Colab too (upload data.npz first).
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

tf.random.set_seed(42)
d = np.load("data.npz")
Xtr, ytr, Xv, yv, Xte, yte = (d["X_train"], d["y_train"], d["X_val"],
                              d["y_val"], d["X_test"], d["y_test"])

# ---------- 1D-CNN ----------
cnn = models.Sequential([
    layers.Input(shape=Xtr.shape[1:]),            # (30 seconds, 4 features)
    layers.Conv1D(32, 3, activation="relu"),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation="relu"),
    layers.GlobalAveragePooling1D(),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),        # probability of congestion soon
])
cnn.compile(optimizer="adam", loss="binary_crossentropy")

# Congestion is rare, so give the rare class more weight
pos = ytr.mean()
class_weight = {0: 1.0, 1: (1 - pos) / pos}

cnn.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=30, batch_size=64,
        class_weight=class_weight, verbose=2,
        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
cnn.save("model.keras")
# Find the best classification threshold using validation data
val_prob = cnn.predict(Xv, verbose=0).ravel()

best_t = max(
    np.arange(0.1, 0.95, 0.05),
    key=lambda t: f1_score(yv, val_prob > t)
)

print("Best threshold:", round(best_t, 2))

# ---------- Baselines (flatten each window into one long row) ----------
flat = lambda X: X.reshape(len(X), -1)
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                            n_jobs=-1, random_state=42).fit(flat(Xtr), ytr)
lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(flat(Xtr), ytr)

# ---------- Compare on the TEST set ----------
preds = {
    "1D-CNN": (cnn.predict(Xte, verbose=0).ravel() > best_t).astype(int),
    "Random Forest": rf.predict(flat(Xte)),
    "Logistic Regression": lr.predict(flat(Xte)),
}
rows = []
for name, p in preds.items():
    rows.append({"Model": name,
                 "Accuracy": (p == yte).mean(),
                 "Precision": precision_score(yte, p, zero_division=0),
                 "Recall": recall_score(yte, p, zero_division=0),
                 "F1": f1_score(yte, p, zero_division=0)})
res = pd.DataFrame(rows).round(3)
res.to_csv("results.csv", index=False)
print("\n", res.to_string(index=False))
print("\nCNN confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(yte, preds["1D-CNN"]))

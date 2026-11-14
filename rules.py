"""Chunk 4: Traffic-management rule engine.
Turns the CNN's congestion probability (plus current latency) into a state and an action.
Used by the dashboard in Chunk 5.  Test it: python rules.py
"""

# (min_probability, state, action, estimated load reduction in %)
LEVELS = [
    (0.85, "CRITICAL", "Reroute heavy flows to backup link + throttle low-priority traffic", 35),
    (0.60, "HIGH",     "Throttle low-priority traffic (video, downloads)",                  20),
    (0.30, "ELEVATED", "Raise alert and prioritise real-time traffic (calls, gaming)",      10),
    (0.00, "NORMAL",   "No action needed",                                                   0),
]
STATE_ORDER = ["NORMAL", "ELEVATED", "HIGH", "CRITICAL"]


def decide(prob, latency_ms=0.0, packet_loss_pct=0.0):
    """Return a dict with state, action and expected load reduction."""
    state, action, cut = next((s, a, c) for p, s, a, c in LEVELS if prob >= p)

    # Safety override: if the network is ALREADY suffering, escalate one level
    if (latency_ms > 60 or packet_loss_pct > 2) and state != "CRITICAL":
        state = STATE_ORDER[STATE_ORDER.index(state) + 1]
        state, action, cut = next((s, a, c) for _, s, a, c in LEVELS if s == state)
        action += "  (escalated: congestion already happening)"

    return {"state": state, "action": action, "load_reduction_pct": cut}


if __name__ == "__main__":
    for p, lat, loss in [(0.05, 20, 0), (0.45, 30, 0.2), (0.70, 40, 1), (0.95, 80, 5), (0.20, 75, 3)]:
        print(f"prob={p:.2f} latency={lat:>3} loss={loss}  ->", decide(p, lat, loss))

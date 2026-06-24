"""Real-Time Anomaly Detection Dashboard — Streamlit app."""

import time
import collections
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from detector.stream import MetricStream
from detector.engine import AnomalyEngine, FEATURES

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_DISPLAY = 120        # points shown on each chart
REFRESH_MS = 800         # how often the dashboard redraws (milliseconds)

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔴 Anomaly Detector")
    st.caption("Real-time ML monitoring · Built by Naia Lorente")
    st.divider()

    speed = st.slider("Ticks per refresh", 1, 8, 3,
                      help="How many data points are generated per UI refresh.")
    show_ground_truth = st.toggle("Show injected anomalies", value=True,
                                  help="Highlights ground-truth anomaly regions (simulated).")
    st.divider()
    st.markdown("**Detectors**")
    st.markdown("- **Isolation Forest** — multivariate, unsupervised")
    st.markdown("- **Z-score** — rolling window per feature")
    st.markdown("- **Ensemble** — flags if either method agrees")
    st.divider()
    reset = st.button("Reset stream")

# ── Session state ──────────────────────────────────────────────────────────────
def _init():
    st.session_state.stream = MetricStream(seed=int(time.time()))
    st.session_state.engine = AnomalyEngine()
    st.session_state.history = {f: collections.deque(maxlen=MAX_DISPLAY) for f in FEATURES}
    st.session_state.history["timestamp"] = collections.deque(maxlen=MAX_DISPLAY)
    st.session_state.history["score_if"] = collections.deque(maxlen=MAX_DISPLAY)
    st.session_state.history["is_anomaly"] = collections.deque(maxlen=MAX_DISPLAY)
    st.session_state.history["is_injected"] = collections.deque(maxlen=MAX_DISPLAY)
    st.session_state.history["severity"] = collections.deque(maxlen=MAX_DISPLAY)
    st.session_state.total = 0
    st.session_state.detected = 0
    st.session_state.false_positives = 0

if "stream" not in st.session_state or reset:
    _init()

stream: MetricStream = st.session_state.stream
engine: AnomalyEngine = st.session_state.engine
hist = st.session_state.history

# ── Generate new data points ───────────────────────────────────────────────────
for _ in range(speed):
    m = stream.next()
    d = engine.ingest(m)
    hist["timestamp"].append(m.timestamp)
    for f in FEATURES:
        hist[f].append(getattr(m, f))
    hist["score_if"].append(d.score_if)
    hist["is_anomaly"].append(d.is_anomaly)
    hist["is_injected"].append(m.is_injected)
    hist["severity"].append(d.severity)
    st.session_state.total += 1
    if d.is_anomaly:
        st.session_state.detected += 1
        if not m.is_injected:
            st.session_state.false_positives += 1

df = pd.DataFrame(hist)

# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ticks", f"{st.session_state.total:,}")
k2.metric("Anomalies flagged", st.session_state.detected)
k3.metric("False positives", st.session_state.false_positives)
warmup_pct = min(100, int(engine.tick / 50 * 100))
k4.metric("Model status", "Live ✅" if engine.is_warmed_up else f"Warming up {warmup_pct}%")
high = (df["severity"] == "HIGH").sum()
k5.metric("HIGH severity", int(high))

st.divider()

# ── Anomaly score chart ────────────────────────────────────────────────────────
col_a, col_b = st.columns([3, 1])

with col_a:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(df["timestamp"]), y=list(df["score_if"]),
        mode="lines", name="IF score", line=dict(color="#4f8ef7", width=1.5),
    ))
    # Mark detected anomalies
    anom_df = df[df["is_anomaly"]]
    fig.add_trace(go.Scatter(
        x=list(anom_df["timestamp"]), y=list(anom_df["score_if"]),
        mode="markers", name="Detected anomaly",
        marker=dict(color="red", size=7, symbol="x"),
    ))
    if show_ground_truth:
        inj_df = df[df["is_injected"]]
        fig.add_trace(go.Scatter(
            x=list(inj_df["timestamp"]), y=list(inj_df["score_if"]),
            mode="markers", name="Injected (ground truth)",
            marker=dict(color="orange", size=5, symbol="circle-open"),
        ))
    fig.update_layout(
        title="Isolation Forest Anomaly Score (lower = more anomalous)",
        height=280, margin=dict(t=40, b=20, l=0, r=0),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    counts = df["severity"].value_counts().reindex(["HIGH", "MEDIUM", "OK"], fill_value=0)
    fig2 = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=["#e74c3c", "#f39c12", "#2ecc71"],
    ))
    fig2.update_layout(title="Severity breakdown", height=280,
                       margin=dict(t=40, b=20, l=0, r=0), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ── Feature charts ─────────────────────────────────────────────────────────────
COLORS = {"cpu": "#4f8ef7", "memory": "#9b59b6", "latency": "#e67e22", "error_rate": "#e74c3c"}
UNITS = {"cpu": "%", "memory": "%", "latency": "ms", "error_rate": ""}

fig3 = make_subplots(rows=2, cols=2, subplot_titles=[f.replace("_", " ").title() for f in FEATURES])

anom_mask = df["is_anomaly"].tolist()
ts = list(df["timestamp"])

for i, feat in enumerate(FEATURES):
    r, c = divmod(i, 2)
    vals = list(df[feat])
    fig3.add_trace(go.Scatter(
        x=ts, y=vals, mode="lines", name=feat,
        line=dict(color=COLORS[feat], width=1.5), showlegend=False,
    ), row=r + 1, col=c + 1)
    # Highlight anomaly points
    anom_x = [ts[j] for j, a in enumerate(anom_mask) if a]
    anom_y = [vals[j] for j, a in enumerate(anom_mask) if a]
    fig3.add_trace(go.Scatter(
        x=anom_x, y=anom_y, mode="markers", showlegend=False,
        marker=dict(color="red", size=5),
    ), row=r + 1, col=c + 1)

fig3.update_layout(height=420, margin=dict(t=40, b=10, l=0, r=0))
st.plotly_chart(fig3, use_container_width=True)

# ── Recent anomaly log ─────────────────────────────────────────────────────────
st.subheader("Recent anomalies")
log = df[df["is_anomaly"]].tail(10)[["timestamp", "cpu", "memory", "latency", "error_rate", "severity"]]
if log.empty:
    st.info("No anomalies detected yet — model is warming up or data is clean.")
else:
    st.dataframe(log.style.map(
        lambda v: "background-color:#f8d7da" if v == "HIGH" else
                  "background-color:#fff3cd" if v == "MEDIUM" else "",
        subset=["severity"],
    ), use_container_width=True, hide_index=True)

# ── Auto-refresh ───────────────────────────────────────────────────────────────
time.sleep(REFRESH_MS / 1000)
st.rerun()

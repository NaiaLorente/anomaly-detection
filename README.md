# 🔴 Real-Time Anomaly Detection Dashboard

A live ML monitoring system that detects anomalies in streaming telemetry data using an **ensemble of Isolation Forest and Z-score detection** — updating in real time with no manual input required.

---

## Demo

The dashboard simulates a continuous stream of server metrics (CPU, memory, latency, error rate) and flags anomalies as they happen:

- **Green** baseline — normal operation
- **Orange** MEDIUM — one detector fired
- **Red** HIGH — both detectors agree (high confidence)

---

## Features

- **Isolation Forest** — scikit-learn unsupervised model, detects multivariate anomalies even without labels
- **Z-score detector** — rolling-window per-feature spike detection, catches sudden jumps instantly
- **Ensemble vote** — combines both signals; severity is HIGH when both agree
- **Adaptive refit** — the Isolation Forest refits every 50 ticks to track concept drift
- **Diurnal simulation** — the synthetic stream follows a realistic load pattern with random injected incidents
- **Live Plotly dashboard** — auto-refreshes every ~800 ms with no page reload
- **Anomaly log** — colour-coded table of recent detections with full metric context
- **Ground-truth overlay** — toggle to compare detections vs. injected anomalies

---

## Architecture

```
MetricStream  →  AnomalyEngine  →  Streamlit Dashboard
   (stream.py)      (engine.py)         (app.py)
       │                │
  Synthetic data    IsolationForest
  diurnal pattern   Z-score (rolling)
  injected spikes   Ensemble vote
```

---

## Quick start

```bash
git clone https://github.com/NaiaLorente/anomaly-detection.git
cd anomaly-detection

pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) — the stream starts immediately, no API key or data upload needed.

---

## Running tests

```bash
pip install pytest ruff
pytest tests/ -v
ruff check .
```

---

## Tech stack

| Layer | Library |
|---|---|
| ML | scikit-learn (`IsolationForest`) |
| Numerics | NumPy, pandas |
| Visualisation | Plotly, Streamlit |
| CI | GitHub Actions |

---

## Project structure

```
anomaly-detection/
├── app.py                    # Live Streamlit dashboard
├── detector/
│   ├── stream.py             # Synthetic telemetry stream generator
│   └── engine.py             # Isolation Forest + Z-score ensemble
├── tests/
│   └── test_engine.py        # Unit tests (no live stream required)
├── .github/
│   └── workflows/ci.yml      # Lint + test on every push
└── requirements.txt
```

---

## Roadmap

- [ ] Plug in a real data source (Prometheus, CloudWatch, CSV replay)
- [ ] Add LSTM-based detector for temporal pattern anomalies
- [ ] Alert webhook (Slack / PagerDuty) on HIGH severity
- [ ] Persist detections to SQLite for historical review

---

## Author

**Naia Lorente** · [github.com/NaiaLorente](https://github.com/NaiaLorente)

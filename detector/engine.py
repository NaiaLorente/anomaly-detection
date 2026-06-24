"""
Anomaly detection engine combining:
  1. Isolation Forest  — unsupervised, catches multivariate anomalies
  2. Z-score           — fast univariate spike detection on a rolling window
  3. Ensemble vote     — flag if either method agrees
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from detector.stream import Metric

FEATURES = ["cpu", "memory", "latency", "error_rate"]
WARMUP_TICKS = 50       # collect this many points before IF predictions
WINDOW_SIZE = 60        # rolling window for Z-score
Z_THRESHOLD = 2.8       # standard deviations to flag as anomaly
IF_CONTAMINATION = 0.05  # expected anomaly fraction for Isolation Forest


@dataclass
class Detection:
    metric: Metric
    score_if: float        # Isolation Forest anomaly score (lower = more anomalous)
    z_scores: dict         # per-feature Z-score
    is_anomaly_if: bool
    is_anomaly_z: bool

    @property
    def is_anomaly(self) -> bool:
        return self.is_anomaly_if or self.is_anomaly_z

    @property
    def severity(self) -> str:
        both = self.is_anomaly_if and self.is_anomaly_z
        return "HIGH" if both else ("MEDIUM" if self.is_anomaly else "OK")


class AnomalyEngine:
    def __init__(self):
        self._buffer: deque[list[float]] = deque(maxlen=WINDOW_SIZE * 10)
        self._window: deque[list[float]] = deque(maxlen=WINDOW_SIZE)
        self._scaler = StandardScaler()
        self._model = IsolationForest(
            contamination=IF_CONTAMINATION,
            n_estimators=100,
            random_state=42,
        )
        self._trained = False
        self._tick = 0

    # ── public ────────────────────────────────────────────────────────────────

    def ingest(self, metric: Metric) -> Detection:
        self._tick += 1
        row = self._to_row(metric)
        self._buffer.append(row)
        self._window.append(row)

        if self._tick == WARMUP_TICKS:
            self._fit()
        elif self._tick > WARMUP_TICKS and self._tick % 50 == 0:
            self._fit()  # periodic refit to adapt to drift

        score_if, flag_if = self._isolation_forest_score(row)
        z_scores, flag_z = self._zscore_flag(row)

        return Detection(
            metric=metric,
            score_if=score_if,
            z_scores=z_scores,
            is_anomaly_if=flag_if,
            is_anomaly_z=flag_z,
        )

    @property
    def is_warmed_up(self) -> bool:
        return self._trained

    @property
    def tick(self) -> int:
        return self._tick

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _to_row(m: Metric) -> list[float]:
        return [m.cpu, m.memory, m.latency, m.error_rate]

    def _fit(self):
        data = np.array(self._buffer)
        self._scaler.fit(data)
        self._model.fit(self._scaler.transform(data))
        self._trained = True

    def _isolation_forest_score(self, row: list[float]) -> tuple[float, bool]:
        if not self._trained:
            return 0.0, False
        x = self._scaler.transform([row])
        score = float(self._model.score_samples(x)[0])
        flag = self._model.predict(x)[0] == -1
        return score, flag

    def _zscore_flag(self, row: list[float]) -> tuple[dict, bool]:
        if len(self._window) < 10:
            return {f: 0.0 for f in FEATURES}, False
        arr = np.array(self._window)
        mean, std = arr.mean(axis=0), arr.std(axis=0) + 1e-9
        zs = np.abs((np.array(row) - mean) / std)
        z_dict = {f: round(float(z), 2) for f, z in zip(FEATURES, zs)}
        return z_dict, bool(zs.max() > Z_THRESHOLD)

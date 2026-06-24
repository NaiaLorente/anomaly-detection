"""Synthetic real-time data stream simulating server/sensor telemetry."""

import time
import random
import math
import numpy as np
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Metric:
    timestamp: datetime
    cpu: float       # 0–100 %
    memory: float    # 0–100 %
    latency: float   # ms
    error_rate: float  # 0–1
    is_injected: bool = False  # ground-truth label for evaluation


class MetricStream:
    """
    Generates a realistic telemetry stream with occasional injected anomalies.

    Normal behaviour follows a diurnal pattern (busier during "business hours")
    plus gaussian noise. Anomalies are sudden spikes or sustained high values.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._t = 0                    # ticks since start
        self._anomaly_until = 0        # tick when current anomaly ends

    # ── public ────────────────────────────────────────────────────────────────

    def next(self) -> Metric:
        self._t += 1
        injected = self._maybe_inject_anomaly()
        return Metric(
            timestamp=datetime.now(),
            cpu=self._cpu(injected),
            memory=self._memory(injected),
            latency=self._latency(injected),
            error_rate=self._error_rate(injected),
            is_injected=injected,
        )

    def stream(self, interval: float = 1.0):
        """Yield metrics indefinitely, sleeping `interval` seconds between each."""
        while True:
            yield self.next()
            time.sleep(interval)

    # ── internals ─────────────────────────────────────────────────────────────

    def _diurnal_load(self) -> float:
        """Sinusoidal base load: peaks every ~288 ticks (simulated 8-hour day)."""
        return 0.5 + 0.3 * math.sin(2 * math.pi * self._t / 288)

    def _maybe_inject_anomaly(self) -> bool:
        if self._t < self._anomaly_until:
            return True
        # 2 % chance of starting a new anomaly lasting 5-20 ticks
        if self._rng.random() < 0.02:
            self._anomaly_until = self._t + self._rng.randint(5, 20)
            return True
        return False

    def _cpu(self, anomaly: bool) -> float:
        base = self._diurnal_load() * 60 + self._np_rng.normal(0, 5)
        spike = self._np_rng.uniform(30, 45) if anomaly else 0
        return float(np.clip(base + spike, 0, 100))

    def _memory(self, anomaly: bool) -> float:
        base = 40 + self._diurnal_load() * 30 + self._np_rng.normal(0, 3)
        spike = self._np_rng.uniform(20, 40) if anomaly else 0
        return float(np.clip(base + spike, 0, 100))

    def _latency(self, anomaly: bool) -> float:
        base = 50 + self._diurnal_load() * 100 + self._np_rng.exponential(10)
        spike = self._np_rng.uniform(200, 600) if anomaly else 0
        return float(np.clip(base + spike, 0, 2000))

    def _error_rate(self, anomaly: bool) -> float:
        base = self._np_rng.beta(1, 40)   # usually near 0
        spike = self._np_rng.uniform(0.1, 0.4) if anomaly else 0
        return float(np.clip(base + spike, 0, 1))

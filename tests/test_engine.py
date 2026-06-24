"""Unit tests for the anomaly detection engine — no live stream needed."""

import pytest
import numpy as np
from detector.stream import MetricStream
from detector.engine import AnomalyEngine, WARMUP_TICKS


@pytest.fixture
def warmed_engine():
    stream = MetricStream(seed=0)
    engine = AnomalyEngine()
    for _ in range(WARMUP_TICKS + 5):
        engine.ingest(stream.next())
    return engine, stream


def test_not_trained_before_warmup():
    engine = AnomalyEngine()
    stream = MetricStream(seed=1)
    for _ in range(WARMUP_TICKS - 1):
        engine.ingest(stream.next())
    assert not engine.is_warmed_up


def test_trained_after_warmup(warmed_engine):
    engine, _ = warmed_engine
    assert engine.is_warmed_up


def test_detection_returns_valid_severity(warmed_engine):
    engine, stream = warmed_engine
    d = engine.ingest(stream.next())
    assert d.severity in ("OK", "MEDIUM", "HIGH")


def test_zscore_keys(warmed_engine):
    engine, stream = warmed_engine
    d = engine.ingest(stream.next())
    assert set(d.z_scores.keys()) == {"cpu", "memory", "latency", "error_rate"}


def test_spike_triggers_anomaly(warmed_engine):
    from detector.stream import Metric
    from datetime import datetime
    engine, _ = warmed_engine
    # Inject an extreme metric manually
    extreme = Metric(
        timestamp=datetime.now(),
        cpu=99.9, memory=99.9, latency=1999.0, error_rate=0.99,
    )
    d = engine.ingest(extreme)
    # At least one detector should fire on such an extreme value
    assert d.is_anomaly_z or d.is_anomaly_if


def test_normal_data_mostly_ok(warmed_engine):
    engine, stream = warmed_engine
    flags = [engine.ingest(stream.next()).is_anomaly for _ in range(100)]
    # Contamination is set to 5%, allow up to 15% false positives in short run
    assert sum(flags) / len(flags) < 0.15

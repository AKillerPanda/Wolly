import numpy as np

from affect_pi.trend import GaussianTrendLayer, GaussianTrendState


def test_first_update_initializes_state():
    trend = GaussianTrendLayer(alpha=0.5)
    s = trend.update(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert s.count == 1
    assert np.allclose(s.mean, [1.0, 2.0, 3.0])
    assert np.allclose(s.z_score, 0.0)
    assert s.anomaly_score == 0.0


def test_counts_increment_and_condensed_matrix_shape():
    trend = GaussianTrendLayer(alpha=0.5)
    s1 = trend.update(np.array([1.0, 2.0], dtype=np.float32))
    s2 = trend.update(np.array([2.0, 4.0], dtype=np.float32))
    assert (s1.count, s2.count) == (1, 2)
    assert s2.condensed_matrix().shape == (2, 3)


def test_empty_vector_returns_zeroed_state():
    trend = GaussianTrendLayer()
    s = trend.update(np.array([], dtype=np.float32))
    assert isinstance(s, GaussianTrendState)
    assert s.count == 0
    assert s.anomaly_score == 0.0


def test_dimension_change_reinitializes():
    trend = GaussianTrendLayer(alpha=0.3)
    trend.update(np.zeros(2, dtype=np.float32))
    s = trend.update(np.zeros(5, dtype=np.float32))   # different size -> reset
    assert s.count == 1
    assert s.mean.size == 5


def test_variance_floor_and_anomaly_nonnegative():
    trend = GaussianTrendLayer(alpha=0.5, min_variance=1e-4)
    trend.update(np.array([0.0, 0.0], dtype=np.float32))
    s = trend.update(np.array([10.0, -10.0], dtype=np.float32))
    assert np.all(s.variance >= 1e-4)
    assert s.anomaly_score >= 0.0
    assert np.all(np.isfinite(s.z_score))

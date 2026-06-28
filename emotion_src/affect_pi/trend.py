from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GaussianTrendState:
    count: int
    mean: np.ndarray
    variance: np.ndarray
    z_score: np.ndarray
    anomaly_score: float

    def condensed_matrix(self) -> np.ndarray:
        """Return [mean, variance, current_z] as a compact trend matrix."""
        if self.mean.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack([self.mean, self.variance, self.z_score], axis=1).astype(np.float32)


@dataclass
class GaussianTrendLayer:
    """Online Gaussian baseline layer using exponential moving statistics.

    This is a practical base for the 'deep Bayesian network / Gaussian distribution'
    layer you described. It keeps a personalized trend state and returns z-scores
    against that state. You can later replace this with a true DBN, Kalman filter,
    HMM, or variational model without changing the pipeline interface.
    """

    alpha: float = 0.03
    min_variance: float = 1e-5
    _count: int = 0
    _mean: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    _variance: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))

    def update(self, vector: np.ndarray) -> GaussianTrendState:
        x = np.asarray(vector, dtype=np.float32).ravel()
        if x.size == 0:
            return GaussianTrendState(0, x, x, x, 0.0)

        if self._mean.size != x.size:
            self._count = 1
            self._mean = x.copy()
            self._variance = np.ones_like(x, dtype=np.float32) * self.min_variance
            z = np.zeros_like(x)
            return GaussianTrendState(self._count, self._mean.copy(), self._variance.copy(), z, 0.0)

        self._count += 1
        prev_mean = self._mean.copy()
        delta = x - prev_mean
        self._mean = (1.0 - self.alpha) * self._mean + self.alpha * x
        # EMA variance update; not an unbiased sample variance, but robust for live streams.
        self._variance = (1.0 - self.alpha) * self._variance + self.alpha * (delta ** 2)
        self._variance = np.maximum(self._variance, self.min_variance)

        z = (x - self._mean) / np.sqrt(self._variance)
        anomaly_score = float(np.mean(np.abs(z)))
        return GaussianTrendState(
            count=self._count,
            mean=self._mean.copy(),
            variance=self._variance.copy(),
            z_score=z.astype(np.float32),
            anomaly_score=anomaly_score,
        )

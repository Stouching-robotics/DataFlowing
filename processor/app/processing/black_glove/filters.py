"""Minimal local One-Euro filter used by black-glove keypoint smoothing."""

from __future__ import annotations

import math
from typing import Optional


class OneEuroFilter:
    def __init__(self, freq_min=1.0, beta=0.007, dcutoff=1.0):
        self.freq_min = freq_min
        self.beta = beta
        self.dcutoff = dcutoff
        self.reset()

    def reset(self):
        self._prev_x: Optional[float] = None
        self._prev_dx: Optional[float] = None
        self._prev_ts: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff) if cutoff > 1e-9 else 0.0
        return dt / (dt + tau) if tau > 0 else 1.0

    def __call__(self, x: float, ts_ms: float) -> float:
        if self._prev_x is None or self._prev_ts is None:
            self._prev_x = x
            self._prev_dx = 0.0
            self._prev_ts = ts_ms
            return x
        dt = (ts_ms - self._prev_ts) / 1000.0
        if dt <= 1e-9:
            return self._prev_x
        dx = (x - self._prev_x) / dt
        alpha_d = self._alpha(self.dcutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._prev_dx
        fc = self.freq_min + self.beta * abs(dx_hat)
        alpha = self._alpha(fc, dt)
        x_hat = alpha * x + (1.0 - alpha) * self._prev_x
        self._prev_x = x_hat
        self._prev_dx = dx_hat
        self._prev_ts = ts_ms
        return x_hat


class OneEuroFilter2D:
    """Independent One-Euro filters for an (x, y) keypoint."""

    def __init__(self, freq_min=1.0, beta=0.007, dcutoff=1.0):
        self._fx = OneEuroFilter(freq_min, beta, dcutoff)
        self._fy = OneEuroFilter(freq_min, beta, dcutoff)

    def reset(self):
        self._fx.reset()
        self._fy.reset()

    def __call__(self, x: float, y: float, ts_ms: float) -> tuple[float, float]:
        return self._fx(x, ts_ms), self._fy(y, ts_ms)

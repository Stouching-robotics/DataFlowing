"""Small detection contract shared by the black-glove pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectedHand:
    """One hand in image pixels, using the MediaPipe 21-point topology."""

    landmarks: np.ndarray
    label: str = "Hand"
    score: float = 0.0
    index: int = 0
    conf: np.ndarray | None = None

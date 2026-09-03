"""Self-contained D435 3D tracking helpers for the black-glove workflow.

This file deliberately contains no import from the standalone v1.0 project.
It keeps only the numerical parts needed after the existing YOLO/RTMPose
detector has produced 2D hands: depth outlier gating and two-slot alpha-beta
tracking.  Predicted points are always marked as ``propagated`` by the caller.
"""

from __future__ import annotations

import numpy as np


BAND_HALF_M = 0.12
BAND_MIN_VALID = 4
GATE_M = 0.15


def gate_observations(points: np.ndarray, prediction: np.ndarray | None,
                      gate: float = GATE_M,
                      wholesale_frac: float = 0.6) -> tuple[np.ndarray, bool]:
    """Reject implausible 3D joints against the slot prediction.

    Individual jumps become NaN and are left for the alpha-beta tracker to
    predict.  If most joints jump together, return ``wholesale=True`` so the
    caller can wait for a consistent observation before re-seeding the slot.
    """
    points = np.asarray(points, np.float64).reshape(21, 3).copy()
    if prediction is None:
        return points, False
    prediction = np.asarray(prediction, np.float64).reshape(21, 3)
    finite = np.isfinite(points).all(axis=1)
    pred_finite = np.isfinite(prediction).all(axis=1)
    comparable = finite & pred_finite
    if int(comparable.sum()) < BAND_MIN_VALID:
        return points, False
    distance = np.linalg.norm(points - prediction, axis=1)
    suspect = comparable & (distance > float(gate))
    if int(suspect.sum()) >= wholesale_frac * int(comparable.sum()):
        return points, True
    points[suspect] = np.nan
    return points, False


class AlphaBetaHandSlots:
    """Two fixed hand slots with constant-velocity alpha-beta prediction."""

    def __init__(self, max_lost: int = 15, alpha: float = 0.5,
                 beta: float = 0.1):
        self.max_lost = max(1, int(max_lost))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.slots = [self._new_slot(), self._new_slot()]

    @staticmethod
    def _new_slot() -> dict:
        return {"label": "", "x": None, "v": None,
                "last_t": None, "lost": 0}

    def reset(self, slot: int, label: str = "") -> None:
        self.slots[int(slot)] = self._new_slot()
        self.slots[int(slot)]["label"] = str(label or "")

    def slot_label(self, slot: int) -> str:
        return str(self.slots[int(slot)].get("label") or "")

    def observe(self, slot: int, label: str, points: np.ndarray,
                frame_index: int) -> None:
        slot = int(slot)
        state = self.slots[slot]
        label = str(label or "")
        measured = np.asarray(points, np.float64).reshape(21, 3)
        if state["label"] and label and state["label"] != label:
            self.reset(slot, label)
            state = self.slots[slot]
        state["label"] = label or state["label"]
        if state["x"] is None:
            state["x"] = measured.copy()
            state["v"] = np.zeros_like(measured)
            state["last_t"] = int(frame_index)
            state["lost"] = 0
            return
        dt = max(float(frame_index - state["last_t"]), 1.0e-3)
        if dt > self.max_lost:
            state["x"] = measured.copy()
            state["v"] = np.zeros_like(measured)
            state["last_t"] = int(frame_index)
            state["lost"] = 0
            return
        previous = state["x"]
        velocity = state["v"]
        prediction = previous + velocity * dt
        finite = np.isfinite(measured).all(axis=1)
        updated = np.where(finite[:, None],
                           self.alpha * measured + (1.0 - self.alpha) * prediction,
                           prediction)
        updated_velocity = np.where(
            finite[:, None],
            self.beta * (updated - previous) / dt + (1.0 - self.beta) * velocity,
            velocity,
        )
        state["x"] = updated
        state["v"] = updated_velocity
        state["last_t"] = int(frame_index)
        state["lost"] = 0

    def mark_lost(self, slot: int) -> None:
        self.slots[int(slot)]["lost"] += 1

    def predict(self, slot: int, frame_index: int) -> np.ndarray | None:
        state = self.slots[int(slot)]
        if state["x"] is None or int(state["lost"]) > self.max_lost:
            return None
        dt = max(float(frame_index - state["last_t"]), 0.0)
        return np.asarray(state["x"] + state["v"] * dt, np.float64)

    def close(self) -> None:
        self.slots = [self._new_slot(), self._new_slot()]

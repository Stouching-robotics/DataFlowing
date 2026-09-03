"""Small shared helpers for stable two-hand slot assignment.

MediaPipe does not promise that the order of ``hand_landmarks`` is stable
between frames.  Keeping that order as ``hand_0``/``hand_1`` lets the
per-slot filters consume the other hand's history when the hands cross or
one hand is briefly lost.  This module assigns detections to persistent
slots using handedness when available and image-space proximity as a
fallback.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Any

import numpy as np


def clip_normalized_xy(points: Any) -> np.ndarray:
    """Return keypoints with normalized image x/y constrained to [0, 1].

    The detector may legitimately return a point just outside the image
    after smoothing or tracker translation.  Such a point cannot project
    onto the source image and makes the overlay appear detached from the
    hand, so only x/y are clipped; the model's z value is preserved.
    """

    array = np.asarray(points, dtype=np.float32).copy()
    if array.ndim == 2 and array.shape[1] >= 2:
        array[:, :2] = np.clip(array[:, :2], 0.0, 1.0)
    return array


class StableHandSlotAssigner:
    """Assign up to ``max_slots`` detections to persistent hand slots.

    Candidates are dictionaries with ``label``, ``center`` and ``payload``
    keys.  The payload is returned untouched, so each caller can keep its
    own landmark representation.  Previous slots are deliberately retained
    briefly when a hand is missing; this lets the next visible frame match
    by identity without inventing a keypoint row for the missing hand.
    """

    def __init__(self, max_slots: int = 2):
        self.max_slots = max(1, int(max_slots))
        self.labels: list[str | None] = [None] * self.max_slots
        self.centers: list[np.ndarray | None] = [None] * self.max_slots
        self.velocities: list[np.ndarray | None] = [None] * self.max_slots
        self._last_seen: list[int | None] = [None] * self.max_slots
        self._tick = 0

    @staticmethod
    def _distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return float("inf")
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    def assign(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        self._tick += 1
        candidates = list(candidates[:self.max_slots])
        # MediaPipe handedness is a per-detection estimate, not a guaranteed
        # physical identity.  In a non-mirrored overhead RGB view it can
        # occasionally report both hands as the same side (for example
        # [Right, Right]).  On the first two-hand frame, use the separated
        # image-space positions to recover the missing semantic distinction.
        # This keeps both the 3D preview and exported hand_left/right fields
        # correct; subsequent frames remain slot-latched.
        if (not any(self.labels) and len(candidates) == 2
                and all(str(c.get("label") or "").lower() in
                        {"", "left", "right"} for c in candidates)):
            centers = [c.get("center") for c in candidates]
            try:
                x_values = [float(np.asarray(center, dtype=np.float32)[0])
                            for center in centers]
                if all(np.isfinite(x) for x in x_values) \
                        and abs(x_values[0] - x_values[1]) > 0.05:
                    order = np.argsort(np.asarray(x_values))
                    normalized = list(candidates)
                    for rank, candidate_index in enumerate(order):
                        normalized[int(candidate_index)] = {
                            **candidates[int(candidate_index)],
                            "label": "Left" if rank == 0 else "Right",
                        }
                    candidates = normalized
            except (TypeError, ValueError, IndexError):
                pass
        result: list[dict[str, Any] | None] = [None] * self.max_slots
        used: set[int] = set()

        def predicted_center(slot: int) -> np.ndarray | None:
            center = self.centers[slot]
            if center is None:
                return None
            velocity = self.velocities[slot]
            last_seen = self._last_seen[slot]
            if velocity is None or last_seen is None:
                return center
            gap = max(1, self._tick - last_seen)
            # A short motion prediction is important when two hands cross:
            # the nearest point after the crossing is often the other hand's
            # old position. Do not extrapolate indefinitely over long gaps.
            return center + velocity * min(gap, 3)

        def assignment_cost(slot: int, index: int) -> float:
            predicted = predicted_center(slot)
            distance = self._distance(
                predicted, candidates[index].get("center"))
            if not np.isfinite(distance):
                return 1e6
            # Handedness is only a weak tie-breaker after a track exists. A
            # hard label preference makes slots swap exactly when MediaPipe's
            # label briefly flips during a crossing or occlusion.
            label = str(candidates[index].get("label") or "").lower()
            stable = str(self.labels[slot] or "").lower()
            if label in {"left", "right"} and stable in {"left", "right"} \
                    and label != stable:
                distance += 0.01
            return distance

        # Existing tracks are matched by predicted image position. Enumerate
        # the small (at most two-hand) assignment space so the two hands are
        # assigned jointly rather than greedily. This preserves identity when
        # they cross in the image.
        active_slots = [slot for slot in range(self.max_slots)
                        if self.centers[slot] is not None]
        if candidates and active_slots:
            count = min(len(active_slots), len(candidates))
            best = None
            for slot_subset in combinations(active_slots, count):
                for candidate_order in permutations(range(len(candidates)), count):
                    cost = sum(
                        assignment_cost(slot, index)
                        for slot, index in zip(slot_subset, candidate_order)
                    )
                    key = (cost, tuple(slot_subset), tuple(candidate_order))
                    if best is None or key < best[0]:
                        best = (key, slot_subset, candidate_order)
            if best is not None:
                _, slot_subset, candidate_order = best
                for slot, index in zip(slot_subset, candidate_order):
                    result[slot] = candidates[index]
                    used.add(index)

        # Fill never-seen/empty slots deterministically, preferring the
        # semantic label only during initialization (not during tracking).
        for slot in range(self.max_slots):
            if result[slot] is not None:
                continue
            remaining = [i for i in range(len(candidates)) if i not in used]
            if not remaining:
                continue
            expected = "left" if slot == 0 else "right"
            index = next(
                (i for i in remaining
                 if str(candidates[i].get("label") or "").lower() == expected),
                remaining[0],
            )
            result[slot] = candidates[index]
            used.add(index)

        for slot, candidate in enumerate(result):
            if candidate is None:
                continue
            label = str(candidate.get("label") or "")
            center = candidate.get("center")
            new_center = (
                np.asarray(center, dtype=np.float32).copy()
                if center is not None else None
            )
            if self.labels[slot] is None and label:
                self.labels[slot] = label
            if new_center is not None:
                old_center = self.centers[slot]
                last_seen = self._last_seen[slot]
                if old_center is not None and last_seen is not None:
                    gap = max(1, self._tick - last_seen)
                    self.velocities[slot] = (new_center - old_center) / gap
                else:
                    self.velocities[slot] = None
                self.centers[slot] = new_center
                self._last_seen[slot] = self._tick
        return result

    def stable_label(self, slot: int, fallback: str | None = None) -> str:
        """Return the label latched when a slot was initialized.

        The detector's per-frame handedness value is useful for initialization
        but must not be allowed to rename an already tracked slot.
        """
        index = max(0, min(int(slot), self.max_slots - 1))
        if self.labels[index] is None and fallback:
            value = str(fallback).strip()
            if value:
                self.labels[index] = value
        return str(self.labels[index] or fallback or "")

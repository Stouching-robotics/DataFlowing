"""RGB-only hand-to-camera tracking used by the Hand Skeleton module.

This module is deliberately self-contained.  It does not import the legacy
``Python`` project or any code outside ``Data Acquisition``.

MediaPipe gives us two useful pieces of information from an RGB frame:

* normalized image landmarks (the 2D observation), and
* a hand-local 3D landmark model (the shape used as the PnP object model).

The PnP solution gives a stable camera-relative placement when the model and
the image points are good enough.  A scale-from-image fallback keeps the hand
visible when PnP is temporarily ill-conditioned.  The result is intentionally
named ``rgb_estimated_meters``: it is a camera-relative estimate, not a depth
sensor measurement and must not be presented as metric ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# MediaPipe image landmarks expose ``z`` as an optional image-space hint.
# Some RGB/black-glove detections provide a valid 21x3 array but set every
# image-space z to zero.  Feeding that planar model to PnP produces a hand
# whose joints have identical depth, so keep a small anatomical depth prior
# as a visual/pose-estimation fallback.  This is deliberately relative and
# must not be interpreted as measured depth.
_RGB_FALLBACK_Z_PROFILE = np.array([
    0.000,                         # wrist
    0.004,  0.009,  0.014,  0.018,  # thumb
    0.000,  0.006,  0.012,  0.017,  # index
    0.000,  0.007,  0.014,  0.020,  # middle
    0.000,  0.006,  0.012,  0.017,  # ring
    0.000,  0.005,  0.010,  0.014,  # pinky
], dtype=np.float64)


def _cohere_thumb_depth_direction(points: np.ndarray) -> np.ndarray:
    """Keep the RGB thumb on the same relative depth side as the fingers.

    RGB image landmarks do not provide reliable metric Z.  Depending on the
    detector/model, the thumb's image-space Z hint can have the opposite sign
    from the four finger chains.  That produces a visually mirrored thumb in
    the 3D preview even though the 2D keypoints are correct.  Only the thumb
    chain's relative Z is corrected; its X/Y image geometry and all measured
    depth paths remain untouched.
    """
    out = np.asarray(points, dtype=np.float64).copy()
    finger_z = out[[8, 12, 16, 20], 2]
    thumb_z = out[[1, 2, 3, 4], 2]
    finger_median = float(np.median(finger_z))
    thumb_median = float(np.median(thumb_z))
    if (abs(finger_median) > 1e-5 and abs(thumb_median) > 1e-5
            and finger_median * thumb_median < 0.0):
        out[1:5, 2] *= -1.0
    return out


def camera_matrix(width: int, height: int, focal_scale: float = 0.9) -> np.ndarray:
    """Return a useful pinhole approximation when no camera calibration exists."""
    w = max(1, int(width))
    h = max(1, int(height))
    focal = max(1.0, float(focal_scale) * max(w, h))
    return np.array([[focal, 0.0, w / 2.0],
                     [0.0, focal, h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


class _OneEuro:
    """Small vector One-Euro filter for camera position."""

    def __init__(self, freq_min: float = 5.0, beta: float = 0.05,
                 dcutoff: float = 1.0):
        self.freq_min = max(0.1, float(freq_min))
        self.beta = max(0.0, float(beta))
        self.dcutoff = max(0.1, float(dcutoff))
        self.prev_x: np.ndarray | None = None
        self.prev_dx = np.zeros(3, dtype=np.float64)
        self.prev_ts: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(1e-9, cutoff))
        return float(dt / (dt + tau))

    def __call__(self, value: np.ndarray, timestamp_s: float) -> np.ndarray:
        x = np.asarray(value, dtype=np.float64)
        if self.prev_x is None or self.prev_ts is None:
            self.prev_x = x.copy()
            self.prev_ts = float(timestamp_s)
            self.prev_dx.fill(0.0)
            return x.copy()
        dt = max(1e-5, float(timestamp_s) - self.prev_ts)
        dx = (x - self.prev_x) / dt
        alpha_d = self._alpha(self.dcutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.prev_dx
        cutoff = self.freq_min + self.beta * np.abs(dx_hat)
        alpha = np.array([self._alpha(float(c), dt) for c in cutoff])
        filtered = alpha * x + (1.0 - alpha) * self.prev_x
        self.prev_x = filtered
        self.prev_dx = dx_hat
        self.prev_ts = float(timestamp_s)
        return filtered.copy()


@dataclass
class _Track:
    position: np.ndarray | None = None
    timestamp_s: float | None = None
    velocity: np.ndarray = None  # type: ignore[assignment]
    position_filter: _OneEuro | None = None

    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(3, dtype=np.float64)


class RGBWorldTracker:
    """Estimate hand camera-relative 3D from one RGB stream."""

    def __init__(self, width: int, height: int, *, camera_matrix_=None,
                 distortion=None, focal_scale: float = 0.9,
                 freq_min: float = 5.0, beta: float = 0.05,
                 max_depth: float = 2.5, min_depth: float = 0.20):
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.K = (np.asarray(camera_matrix_, dtype=np.float64).reshape(3, 3)
                  if camera_matrix_ is not None else
                  camera_matrix(self.width, self.height, focal_scale))
        self.dist = (np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
                     if distortion is not None else np.zeros((5, 1), np.float64))
        self.freq_min = float(freq_min)
        self.beta = float(beta)
        self.max_depth = float(max_depth)
        self.min_depth = float(min_depth)
        self._tracks: dict[str, _Track] = {}
        # RGB handedness can flip briefly during fast motion or self-occlusion.
        # Keep a small per-track vote state so the preview root anchor does
        # not jump from the left hand slot to the right hand slot.
        self._hand_labels: dict[str, dict[str, object]] = {}

    def stabilize_handedness(self, key: str, label: str | None,
                             votes_required: int = 3) -> str:
        """Return a short-term latched left/right label for one RGB track."""
        value = str(label or "").strip().lower()
        if value not in {"left", "right"}:
            state = self._hand_labels.get(str(key))
            return str(state["stable"]).title() if state else ""
        track_key = str(key)
        state = self._hand_labels.setdefault(
            track_key, {"stable": value, "candidate": value, "count": 0})
        stable = str(state["stable"])
        if value == stable:
            state["candidate"] = value
            state["count"] = 0
        else:
            if state.get("candidate") == value:
                state["count"] = int(state.get("count", 0)) + 1
            else:
                state["candidate"] = value
                state["count"] = 1
            if int(state["count"]) >= max(1, int(votes_required)):
                state["stable"] = value
                state["candidate"] = value
                state["count"] = 0
        return str(state["stable"]).title()

    @staticmethod
    def normalize_local_world(points) -> np.ndarray | None:
        """Center MediaPipe world landmarks on the wrist.

        The Y/Z flips put the points in the display convention used by the
        project: X right, Y up, Z forward.  The returned model remains in the
        original hand scale (approximately metres), which is useful for PnP.
        """
        arr = np.asarray(points, dtype=np.float64)
        if arr.shape != (21, 3) or not np.isfinite(arr).all():
            return None
        out = arr - arr[0]
        # MediaPipe can return a correctly shaped all-zero world model when
        # the optional world-landmark estimate is unavailable.  It is not a
        # valid PnP object model; returning None lets the caller construct the
        # independent RGB/image-scale model instead.
        if (float(np.max(np.abs(out))) <= 1e-6
                or float(np.ptp(out[:, 2])) <= 1e-5):
            return None
        out[:, 1] *= -1.0
        out[:, 2] *= -1.0
        # MediaPipe world landmarks are still an RGB-relative estimate here,
        # not a depth-sensor measurement.  Keep its thumb depth direction
        # consistent with the four finger chains for the shared preview.
        return _cohere_thumb_depth_direction(out)

    @staticmethod
    def orient_palm_facing(points, handedness: str | None) -> np.ndarray:
        """Orient an RGB-relative model toward its semantic palm side.

        RGB landmarks do not contain a reliable absolute palm/back signal.
        For the shared estimated-3D convention, keep the four finger chains
        on one forward depth side of the wrist.  This prevents an RGB/PnP
        mirror solution from making the fingers point away from the hand
        while preserving every landmark's X/Y image geometry.  This is a
        display/estimated-3D convention, not a depth-sensor measurement.
        """
        out = np.asarray(points, dtype=np.float64).copy()
        if out.shape != (21, 3) or not np.isfinite(out).all():
            return out
        # Keep the handedness argument in this shared API because callers
        # already have it and future calibration can use it.  It does not
        # change the relative-Z sign: left and right hands use the same
        # camera-relative depth convention.
        _ = str(handedness or "").strip().lower()

        # Select one stable RGB convention: fingertips lie on the positive
        # relative-Z side of the wrist/palm.  This is the direction used by
        # the fallback anatomical prior and removes the mirrored PnP choice.
        finger_depth = float(np.median(out[[8, 12, 16, 20], 2]) - out[0, 2])
        if finger_depth < -1e-6:
            out[:, 2] *= -1.0
        return _cohere_thumb_depth_direction(out)

    @staticmethod
    def local_from_image(points_2d) -> np.ndarray | None:
        """Build a non-planar relative hand model when World Landmarks fail.

        The 2D detector's ``p.z`` is not guaranteed to be populated.  When
        it is all zero/constant, using it as depth creates a flat skeleton.
        Preserve a real image-space z signal when available; otherwise use a
        small topology-based prior so PnP and the right-side preview retain a
        relative third dimension.  This remains RGB-estimated 3D, not depth
        ground truth.
        """
        arr = np.asarray(points_2d, dtype=np.float64)
        if arr.shape != (21, 3) or not np.isfinite(arr[:, :2]).all():
            return None
        out = np.zeros((21, 3), dtype=np.float64)
        out[:, 0] = (arr[:, 0] - arr[0, 0]) * 0.22
        out[:, 1] = -(arr[:, 1] - arr[0, 1]) * 0.22
        image_z = (-(arr[:, 2] - arr[0, 2]) * 0.08
                   if np.isfinite(arr[:, 2]).all() else None)
        if image_z is not None and float(np.ptp(image_z)) > 1e-5:
            out[:, 2] = image_z
        else:
            image_span = max(
                float(np.ptp(arr[:, 0])), float(np.ptp(arr[:, 1])), 1e-3)
            # Scale the prior with the observed hand size, while keeping it
            # close to the x/y model scale and bounded for distant hands.
            prior_scale = float(np.clip(image_span / 0.14, 0.55, 1.45))
            out[:, 2] = _RGB_FALLBACK_Z_PROFILE * prior_scale
        return _cohere_thumb_depth_direction(out)

    @staticmethod
    def black_glove_reference_local(points_2d) -> np.ndarray | None:
        """Build the same RGB local model used by Black Glove RGB3D.

        The bare-hand MediaPipe path also exposes ``p.z`` and world
        landmarks, but those values use a different local orientation from
        the YOLO/RTMPose black-glove path. Feeding that optional z hint made
        the two RGB workflows render different palm directions. The
        black-glove reference derives depth from the shared anatomical prior
        and observed 2D geometry only.
        """
        arr = np.asarray(points_2d, dtype=np.float64).copy()
        if arr.shape != (21, 3):
            return None
        arr[:, 2] = 0.0
        return RGBWorldTracker.local_from_image(arr)

    def _pixels(self, points_2d) -> np.ndarray | None:
        arr = np.asarray(points_2d, dtype=np.float64)
        if arr.shape != (21, 3) or not np.isfinite(arr[:, :2]).all():
            return None
        return arr[:, :2] * np.array([self.width, self.height], dtype=np.float64)

    def _image_position(self, pixels: np.ndarray,
                        local: np.ndarray) -> np.ndarray | None:
        # Estimate hand depth from the median ratio of corresponding 3D and
        # image bone lengths.  This remains useful when PnP loses an inlier.
        model_pairs = ((0, 5), (0, 9), (0, 13), (0, 17),
                       (5, 9), (9, 13), (13, 17),
                       (5, 8), (9, 12), (13, 16), (17, 20))
        ratios = []
        for a, b in model_pairs:
            model_len = float(np.linalg.norm(local[a] - local[b]))
            image_len = float(np.linalg.norm(pixels[a] - pixels[b]))
            if model_len > 1e-5 and image_len > 1.0:
                ratios.append(model_len / image_len)
        if len(ratios) < 3:
            return None
        depth = float(np.median(ratios) * self.K[0, 0])
        depth = float(np.clip(depth, self.min_depth, self.max_depth))
        u, v = pixels[0]
        x = (u - self.K[0, 2]) * depth / self.K[0, 0]
        y_down = (v - self.K[1, 2]) * depth / self.K[1, 1]
        return np.array([x, -y_down, depth], dtype=np.float64)

    def _pnp(self, pixels: np.ndarray, local: np.ndarray):
        valid = np.isfinite(local).all(axis=1) & np.isfinite(pixels).all(axis=1)
        if int(valid.sum()) < 6:
            return None
        object_points = np.ascontiguousarray(local[valid], dtype=np.float64)
        image_points = np.ascontiguousarray(pixels[valid], dtype=np.float64)
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points, image_points, self.K, self.dist,
                iterationsCount=80, reprojectionError=8.0,
                confidence=0.98,
                flags=cv2.SOLVEPNP_EPNP)
        except Exception:
            ok = False
            rvec = None
            tvec = None
            inliers = None
        if not ok:
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points, image_points, self.K, self.dist,
                    flags=cv2.SOLVEPNP_SQPNP)
                inliers = None
            except Exception:
                return None
        if not ok or rvec is None or tvec is None:
            return None
        try:
            rotation, _ = cv2.Rodrigues(rvec)
            projected, _ = cv2.projectPoints(object_points, rvec, tvec,
                                              self.K, self.dist)
            error = float(np.mean(np.linalg.norm(
                projected.reshape(-1, 2) - image_points, axis=1)))
        except Exception:
            return None
        if not np.isfinite(error):
            return None
        return rotation, np.asarray(tvec, dtype=np.float64).reshape(3), error, inliers

    def update(self, key: str, points_2d, local_points,
               timestamp_s: float, *, preserve_model_geometry: bool = True) -> dict | None:
        """Estimate camera-relative hand points.

        RGB-only 3D is intentionally model-geometry-preserving by default.
        PnP is retained for the camera-relative wrist position/depth estimate,
        but its frame-by-frame rotation is not applied to the hand joints.
        This avoids mirrored or rapidly changing PnP rotations becoming
        visible after the frontend fixes the wrist root.
        """
        pixels = self._pixels(points_2d)
        local = np.asarray(local_points, dtype=np.float64)
        if pixels is None or local.shape != (21, 3) or not np.isfinite(local).all():
            return None
        track = self._tracks.setdefault(
            str(key), _Track(position_filter=_OneEuro(self.freq_min, self.beta)))

        pnp = self._pnp(pixels, local)
        source = "rgb_image_scale"
        reprojection_error = None
        rotation = np.eye(3, dtype=np.float64)
        position = None
        if pnp is not None:
            rotation, tvec, reprojection_error, _inliers = pnp
            if reprojection_error <= 18.0 and self.min_depth <= tvec[2] <= self.max_depth:
                position = np.array([tvec[0], -tvec[1], tvec[2]], dtype=np.float64)
                source = "rgb_pose_pnp"

        if position is None:
            position = self._image_position(pixels, local)
            if position is None and track.position is not None:
                previous_timestamp = (track.timestamp_s
                                      if track.timestamp_s is not None
                                      else float(timestamp_s))
                dt = max(0.0, float(timestamp_s) - previous_timestamp)
                position = track.position + track.velocity * min(dt, 0.10)
                source = "rgb_prediction"
            if position is None:
                return None

        if track.position_filter is None:
            track.position_filter = _OneEuro(self.freq_min, self.beta)
        filtered = track.position_filter(position, float(timestamp_s))
        if track.position is not None and track.timestamp_s is not None:
            dt = max(1e-5, float(timestamp_s) - track.timestamp_s)
            measured_velocity = (filtered - track.position) / dt
            track.velocity = 0.7 * track.velocity + 0.3 * measured_velocity
        track.position = filtered.copy()
        track.timestamp_s = float(timestamp_s)

        # PnP tvec is in the OpenCV camera convention (Y down).  Rotate the
        # local model into that camera frame, then convert every point to the
        # display convention (Y up) so position and landmarks share a frame.
        camera_points = (rotation @ local.T).T + np.array(
            [filtered[0], -filtered[1], filtered[2]], dtype=np.float64)
        landmarks = camera_points.copy()
        landmarks[:, 1] *= -1.0
        if source != "rgb_pose_pnp" or preserve_model_geometry:
            landmarks = local + filtered

        return {
            "landmarks_3d": landmarks.astype(np.float32),
            "landmarks_3d_local": local.astype(np.float32),
            "position": filtered.astype(np.float32),
            "tracking_source": source,
            "coordinate_frame": "camera_relative",
            "unit": "rgb_estimated_meters",
            "metric_3d_available": False,
            "reprojection_error": (float(reprojection_error)
                                    if reprojection_error is not None else None),
        }

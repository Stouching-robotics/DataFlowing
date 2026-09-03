"""Depth-image hand 3D pipeline — internal helper for Hand Skeleton.

Combines an RGB video, a depth PNG sequence (16-bit millimetres), and
calibration data to lift MediaPipe 2D hand landmarks into camera-coordinate
3D points in metres. The imported demo retains its historical filename.

No standalone card is registered. Hand Skeleton calls ``run_depth_hand_3d``
when mode=auto finds compatible depth data; otherwise it falls back to the
standalone RGB camera-relative 3D path.

The parquet stores only smoothed measurements. Display-only anchors are not
fed back into tracking and never contaminate exported coordinates.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from app.processing import JobContext, ArtifactRef
from app.lerobot_v21 import DepthVideoReader, is_depth_source, iter_video_streams

_BLACK_GLOVE_ROOT = Path(__file__).resolve().parents[1] / "black_glove"
_DEPTH_HAND_DEMO_FILE = _BLACK_GLOVE_ROOT / "d435_hands_demo.py"


def _resolve_hand_landmarker_model() -> Path:
    """Resolve the MediaPipe model from the application or demo package."""
    candidates = (
        Path(__file__).resolve().parents[3] / "models" / "hand_landmarker.task",
        _BLACK_GLOVE_ROOT / "hand_landmarker.task",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"MediaPipe hand model not found; searched: {searched}")


_MODEL = _resolve_hand_landmarker_model()


def _load_demo():
    """importlib 加载_hands_demo.py(自包含、有 __main__ 保护,零副作用)。"""
    try:
        spec = importlib.util.spec_from_file_location(
            "depth_hand_demo_loaded", str(_DEPTH_HAND_DEMO_FILE))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        # @dataclass 在 exec 期间要查 sys.modules[__module__].__dict__,
        # 必须先注册进 sys.modules
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        print(f"[depth_hand_3d] demo load failed: {exc}")
        return None


def _finger_gesture_from_pts(demo, pts3d: np.ndarray) -> tuple[str, int]:
    """(21,3) 相机系米制 → (gesture 标签, fingers bitmask)。

    与 demo HandResult.extended / stereo _finger_gesture_from_world 同一套
    角度判据(Thumb: MCP>145° 且 IP>150°;其余: PIP>150° 且 DIP>140°)。
    """
    pts = np.asarray(pts3d, np.float64).reshape(21, 3)
    angles = {(f, j): demo._angle_between(pts[p], pts[v], pts[n])
              for f, j, v, p, n, _c in demo.JOINT_SPECS}
    extended: list[str] = []
    mask = 0
    bits = {"Thumb": 1, "Index": 2, "Middle": 4, "Ring": 8, "Pinky": 16}
    for finger in demo.FINGERS:
        if finger == "Thumb":
            ok = (angles.get((finger, "MCP"), 0) > 145
                  and angles.get((finger, "IP"), 0) > 150)
        else:
            ok = (angles.get((finger, "PIP"), 0) > 150
                  and angles.get((finger, "DIP"), 0) > 140)
        if ok:
            extended.append(finger.lower())
            mask |= bits[finger]
    label = "fist" if not extended else "open:" + ",".join(extended)
    return label, mask


class DepthHand3DDriver:
    """demo main() 逐帧状态机的可驱动封装(demo L1928-2140 数据路径)。

    每帧输入 (rgb_bgr, depth_mm, frame_idx) → 输出双槽位平滑 3D
    (彩色相机系米制,NaN=无效)。只做数据路径:渲染/伪彩/视频输出
    全部剔除;centroid_anchor/view_anchor 展示路径不回流(不进 parquet)。
    """

    def __init__(self, demo, model_path: str, det_conf: float = 0.4,
                 track_conf: float = 0.4, fill: int = 1,
                 propagate_max: int = 15,
                 color_intr: dict | None = None,
                 depth_to_color: dict | None = None,
                 depth_intr: dict | None = None):
        calib = None
        if color_intr is None or depth_to_color is None:
            calib = json.loads(json.dumps(demo._EMBEDDED_CALIB))
            color_intr = calib["color_intrinsics"]
            depth_to_color = calib["depth_to_color"]
        if depth_intr is None and calib:
            depth_intr = calib["depth_intrinsics"]
        self._aligner = demo.LiveAligner(color_intr, depth_to_color,
                                         depth_intr, fill_passes=fill)
        self._color_intr = (self._aligner.fx_c, self._aligner.fy_c,
                            self._aligner.cx_c, self._aligner.cy_c)
        self._det = demo.MediaPipeDetector(
            model_path=model_path, num_hands=2,
            det_conf=det_conf, track_conf=track_conf)
        self._voter = demo.HandednessVoter()
        self._tracker = demo.HandSlotTracker(max_lost=propagate_max)
        self._smoother = demo.Hand3DSmoother()
        self._soft = demo._SoftSmoother(self._smoother)
        self._lost_counts = [0, 0]
        self._zc_slot: list[float | None] = [None, None]
        self._ws_prev: list[np.ndarray | None] = [None, None]
        self._ws_streak = [0, 0]
        self._gate_streak = [np.zeros(21, np.int64), np.zeros(21, np.int64)]
        # 2D display identity is independent from the metric-3D tracker.
        # This prevents a bad depth sample from hiding a valid RGB detection.
        self._last_kp2d: list[np.ndarray | None] = [None, None]
        self.stats = {"real": [0, 0], "propagated": [0, 0], "absent": [0, 0]}

    def process_frame(self, demo, rgb_bgr, depth_mm, frame_idx: int) -> dict:
        """一帧状态机(镜像 demo L1949-2093)。返回:
        {"slots_3d": (2,21,3) float32 米(平滑后), "labels": [str,str],
         "states": [real|propagated|absent], "presents": [bool,bool],
         "dets": [DetectedHand|None, ...]}"""
        aligner = self._aligner
        if depth_mm is None or depth_mm.shape[:2] != (aligner.dh, aligner.dw):
            aligned = np.zeros((aligner.ch, aligner.cw), np.float32)
        else:
            aligned = aligner.align_depth_to_color(depth_mm)

        hands = self._det.detect(rgb_bgr)
        # 空帧不喂 voter:空帧会清空轨迹(scene reset 语义),短暂漏检
        # 会清票仓 → 重建期原始 label 闪烁 → 两手同 label(demo 注释)。
        if hands:
            self._voter.update(hands, frame_w=rgb_bgr.shape[1],
                               frame_h=rgb_bgr.shape[0], frame=frame_idx,
                               cam="depth_camera")

        pairs = [demo.lift_hand(hd, aligner, aligned) for hd in hands]
        same_lab = (len(pairs) == 2 and pairs[0].left_label
                    and pairs[0].left_label == pairs[1].left_label)
        if same_lab:
            for p in pairs:
                p.left_label = ""
        out = demo.assign_mono_slots(pairs, self._tracker, frame_idx,
                                     self._color_intr,
                                     lost_counts=tuple(self._lost_counts))
        if same_lab:
            for s in range(2):
                if out[s] is not None:
                    sl = self._tracker.slot_label(s)
                    if sl:
                        out[s].left_label = sl

        slot_pairs, slot_dets, states = [], [], []
        for s in range(2):
            if out[s] is not None:
                p = out[s]
                # M5:补点深度锚定到槽级稳定 zc(换手/首帧取实测)
                meas = getattr(p, "measured", None)
                if meas is not None and meas.any():
                    pts3d = np.asarray(p.result.points_3d, np.float64) \
                        .reshape(21, 3)
                    zf = float(np.median(pts3d[meas, 2]))
                    if self._tracker.slot_label(s) != p.left_label \
                            or self._zc_slot[s] is None:
                        self._zc_slot[s] = zf
                    else:
                        self._zc_slot[s] = 0.5 * self._zc_slot[s] + 0.5 * zf
                    demo.apply_slot_zc(p, self._zc_slot[s], aligner)
                # 时序一致性门:与槽预测差 >150mm 的点判可疑置 NaN
                gated, wholesale = demo.gate_observations(
                    p.result.points_3d, self._tracker.predict(s, frame_idx))
                # M6:门控锁死豁免(连续被门控 ≥_GATE_FORGIVE 且观测恢复
                # 有限时采信观测);换手帧不豁免。
                if not wholesale:
                    if self._tracker.slot_label(s) != p.left_label:
                        self._gate_streak[s][:] = 0
                    meas3d = np.asarray(p.result.points_3d, np.float64) \
                        .reshape(21, 3)
                    g_fin = np.isfinite(
                        np.asarray(gated, np.float64).reshape(-1, 3)).all(axis=1)
                    m_fin = np.isfinite(meas3d).all(axis=1)
                    gs = self._gate_streak[s]
                    latched = ~g_fin & m_fin
                    gs[latched] += 1
                    gs[~latched] = 0
                    forgive = (gs >= demo._GATE_FORGIVE) & m_fin
                    if forgive.any():
                        gated[forgive] = meas3d[forgive]
                if wholesale:
                    # M3②:整手级不匹配先两帧确认,连续两帧观测互相一致
                    # 才判槽状态过时 → 借 label 翻转触发槽位重置。
                    if demo._ws_agree(self._ws_prev[s], gated) \
                            or self._ws_streak[s] >= 3:
                        self._tracker.observe_slot(
                            s, "\x00reset", np.full((21, 3), np.nan), frame_idx)
                        self._tracker.observe_slot(s, p.left_label, gated,
                                                   frame_idx)
                        p.result.points_3d = gated
                        self._lost_counts[s] = 0
                        self._ws_prev[s] = None
                        self._ws_streak[s] = 0
                        self._gate_streak[s][:] = 0
                        slot_pairs.append(p)
                        slot_dets.append(p.det)
                        states.append("real")
                    else:
                        self._ws_prev[s] = gated
                        self._ws_streak[s] += 1
                        pred_now = self._tracker.predict(s, frame_idx)
                        if pred_now is not None:
                            slot_pairs.append(demo._pred_pair(
                                pred_now, self._tracker.slot_label(s)))
                            slot_dets.append(None)
                            states.append("propagated")
                        else:
                            slot_pairs.append(demo._nan_pair(
                                self._tracker.slot_label(s)))
                            slot_dets.append(None)
                            states.append("absent")
                else:
                    self._tracker.observe_slot(s, p.left_label, gated,
                                               frame_idx)
                    p.result.points_3d = gated
                    self._lost_counts[s] = 0
                    self._ws_prev[s] = None
                    self._ws_streak[s] = 0
                    slot_pairs.append(p)
                    slot_dets.append(p.det)
                    states.append("real")
            else:
                pred = self._tracker.predict(s, frame_idx)
                self._tracker.mark_lost(s, frame_idx)
                self._lost_counts[s] += 1
                self._ws_prev[s] = None
                self._ws_streak[s] = 0
                if pred is not None:
                    slot_pairs.append(demo._pred_pair(
                        pred, self._tracker.slot_label(s)))
                    slot_dets.append(None)
                    states.append("propagated")
                else:
                    slot_pairs.append(demo._nan_pair(
                        self._tracker.slot_label(s)))
                    slot_dets.append(None)
                    states.append("absent")
        for s, st in enumerate(states):
            self.stats[st][s] += 1

        presents = [st != "absent" for st in states]
        labels = [slot_pairs[s].left_label if out[s] is not None
                  else self._tracker.slot_label(s) for s in range(2)]

        # The metric tracker may reject a 3D observation while the RGB
        # detector still has a perfectly usable 2D hand.  Assign those
        # detections to display slots using handedness first, then the last
        # 2D position, and finally detector order.  This is intentionally a
        # separate association path: 3D validity must not control 2D review.
        display_dets = [slot_dets[s] for s in range(2)]
        used = {id(det) for det in display_dets if det is not None}
        remaining = [p for p in pairs if p.det is not None and id(p.det) not in used]
        for s in range(2):
            if display_dets[s] is not None:
                continue
            slot_label = self._tracker.slot_label(s)
            labeled = [p for p in remaining
                       if slot_label and p.left_label == slot_label]
            if labeled:
                chosen = labeled[0]
            elif remaining and self._last_kp2d[s] is not None:
                prev = self._last_kp2d[s]
                chosen = min(
                    remaining,
                    key=lambda p: float(np.linalg.norm(
                        np.asarray(p.hand2d, np.float64).reshape(-1, 2).mean(0)
                        - prev
                    )),
                )
            elif remaining:
                chosen = remaining[0]
            else:
                chosen = None
            if chosen is not None:
                display_dets[s] = chosen.det
                remaining = [p for p in remaining if p is not chosen]
        for s, det in enumerate(display_dets):
            if det is not None:
                pts2d = np.asarray(det.landmarks, np.float32).reshape(-1, 2)
                self._last_kp2d[s] = pts2d.mean(axis=0)

        # (2,21,3) 槽位 3D(tracker αβ 已平滑)→ OneEuro 再平滑压静止抖动
        h3 = np.stack([np.asarray(p.result.points_3d, np.float64)
                       .reshape(21, 3) for p in slot_pairs])
        valids = [int(np.isfinite(h3[s]).all(axis=1).sum()) for s in range(2)]
        smoothed = self._soft.update(h3, labels, valids)
        return {"slots_3d": smoothed, "labels": labels, "states": states,
                "presents": presents, "dets": slot_dets,
                "display_dets": display_dets}

    def close(self) -> None:
        try:
            self._det.close()
        except Exception:
            pass


def _row(demo, frame_index: int, res: dict, w: int, h: int) -> dict:
    """一帧 → stereo 同构行(hand_0/1_* 列,63 位扁平,NaN 填充)。"""
    row: dict[str, Any] = {"frame_index": int(frame_index)}
    for si in (0, 1):
        pts = np.asarray(res["slots_3d"][si], np.float64)
        present = bool(res["presents"][si])
        det = (res.get("display_dets") or res["dets"])[si]
        label = str(res["labels"][si] or "")
        if det is not None:
            kp2 = np.asarray(det.landmarks, np.float32).reshape(-1, 2)
            kp = np.zeros((21, 3), np.float32)
            kp[:, 0] = kp2[:, 0] / max(1, w)
            kp[:, 1] = kp2[:, 1] / max(1, h)
            kp[:, :2] = np.clip(kp[:, :2], 0.0, 1.0)
            kp = kp.reshape(63).tolist()
        else:
            kp = np.full(63, np.nan, np.float32).tolist()
        if present:
            gesture, fingers = _finger_gesture_from_pts(demo, pts)
        else:
            gesture, fingers = "", -1
        row.update({
            f"hand_{si}_present": present,
            f"hand_{si}_2d_present": bool(det is not None),
            f"hand_{si}_landmarks_3d": (pts.reshape(63).tolist() if present
                                        else np.full(63, np.nan,
                                                     np.float32).tolist()),
            f"hand_{si}_keypoints": kp,
            f"hand_{si}_label": label if present else "",
            f"hand_{si}_reprojection_error": float("nan"),
            f"hand_{si}_gesture": gesture if present else "",
            f"hand_{si}_fingers": int(fingers) if present else -1,
            f"hand_{si}_confidence": float(det.score)
            if det is not None else 0.0,
        })
    return row




def _batch_root_of(ctx: JobContext, depth_cam: str = "") -> tuple[Path, str] | None:
    """定位批次根目录。worker 输入包以 `<批次名>/...` 归档,解压后
    input_root 下多一层批次目录;旧布局(直接平铺)时 input_root 即批次根。"""
    roots = [ctx.input_root]
    roots.extend(sub for sub in sorted(ctx.input_root.iterdir()) if sub.is_dir())
    for root in roots:
        depth_root = root / "depth"
        if not depth_root.is_dir():
            depth_root = root / "meta" / "depth"
        if depth_root.is_dir():
            candidates = ([depth_root / depth_cam] if depth_cam else
                          [p for p in sorted(depth_root.iterdir()) if p.is_dir()])
            for candidate in candidates:
                if candidate.is_dir() and any(candidate.glob("*.png")):
                    return root, candidate.name
        try:
            for source, _path in iter_video_streams(root / "videos"):
                if is_depth_source(source) and (not depth_cam or source == depth_cam):
                    return root, source
        except Exception:
            pass
    return None


def _safe_source_key(value: str | None) -> str:
    """Return a stable filename/feature suffix for a device stream."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return value.strip("._") or "device"


def _session_roots(input_root: Path) -> list[Path]:
    """Candidate unpacked episode roots, including video-only depth batches."""
    roots = [Path(input_root)]
    roots.extend(p for p in sorted(Path(input_root).iterdir()) if p.is_dir())
    result = []
    for root in roots:
        if (root / "depth").is_dir() or (root / "meta" / "depth").is_dir():
            result.append(root)
            continue
        try:
            if any(is_depth_source(source)
                   for source, _path in iter_video_streams(root / "videos")):
                result.append(root)
        except (OSError, TypeError, ValueError):
            continue
    return result


def _device_records(batch_root: Path,
                    embedded_metadata: dict | None = None) -> list[dict]:
    """Read physical-device/slot declarations from the uploaded metadata.

    Older uploads may not contain ``devices``; callers then use the source-name
    fallback below.  The declaration is deliberately kept local to the worker
    so a multi-device episode cannot accidentally use another episode's slots.
    """
    documents = []
    if isinstance(embedded_metadata, dict):
        documents.append(embedded_metadata)
    for name in ("metadata.json", "meta/info.json"):
        path = batch_root / name
        if not path.is_file():
            continue
        try:
            documents.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        raw = doc.get("devices") or []
        if isinstance(raw, dict):
            raw = [dict(value or {}, key=key) for key, value in raw.items()]
        result = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            slots = [str(v) for v in (item.get("slots") or []) if str(v).strip()]
            result.append({
                "key": str(item.get("key") or item.get("id") or f"device_{index}"),
                "name": str(item.get("name") or item.get("device_name") or ""),
                "slots": slots,
                "calibration": str(item.get("calibration") or ""),
                "serial": str(item.get("serial") or item.get("serial_number") or ""),
            })
        if result:
            return result
    return []


def _device_token(value: str | None) -> str:
    """Normalize D405_depth_rgb and D405_depth to the same device token."""
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    for suffix in ("_depth_rgb", "_depth_color", "_color", "_rgb", "_depth"):
        if text.endswith(suffix):
            text = text[:-len(suffix)].rstrip("_")
            break
    return text


def _find_device_pairs(ctx: JobContext, video_refs: list) -> list[dict]:
    """Pair every RGB artifact with its physical device's depth stream.

    Matching order is metadata slots → normalized device token → single-depth
    legacy fallback.  A pair is returned even when its depth is unavailable so
    the caller can report that device explicitly instead of silently assigning
    another camera's depth image.
    """
    roots = _session_roots(ctx.input_root)
    if not roots:
        return []
    root = roots[0]
    depth_root = root / "depth"
    if not depth_root.is_dir():
        depth_root = root / "meta" / "depth"
    depth_dirs = {
        p.name: p for p in sorted(depth_root.iterdir())
        if depth_root.is_dir() and p.is_dir() and any(p.glob("*.png"))
    } if depth_root.is_dir() else {}
    episode_index = None
    episode_row = None
    try:
        from app.project_dataset import project_episode_rows
        rows = project_episode_rows(root)
        if len(rows) == 1:
            episode_row = rows[0]
            episode_index = int(episode_row.get("episode_index"))
    except (OSError, ValueError, TypeError):
        episode_index = None
    depth_videos = {
        source: path for source, path in iter_video_streams(root / "videos")
        if is_depth_source(source)
        and (episode_index is None
             or path.stem == f"episode_{episode_index:06d}")
    }
    depth_sources = set(depth_dirs) | set(depth_videos)
    embedded_collector = {}
    if isinstance(episode_row, dict):
        collector = episode_row.get("collector")
        if isinstance(collector, dict):
            candidate = collector.get("metadata.json")
            if isinstance(candidate, dict):
                embedded_collector = candidate
    records = _device_records(root, embedded_collector)
    pairs = []
    for ref in video_refs:
        source = str(ref.source_key or Path(str(ref.path or "video")).stem)
        source_lower = source.lower()
        if ("depth" in source_lower and
                not any(token in source_lower for token in ("rgb", "color", "video"))):
            continue
        record = next((d for d in records if source in d["slots"]), None)
        if record is None:
            # Workflow input cards often use the physical device token
            # (``D405``) while uploaded metadata uses a slot name
            # (``D405_depth_rgb``).  Resolve that alias before falling back to
            # a synthetic record, so serial/calibration declarations are not
            # lost and another same-resolution device cannot win inference.
            token = _device_token(source)
            record = next(
                (d for d in records
                 if token and any(_device_token(slot) == token
                                  for slot in d["slots"])),
                None,
            )
        depth_name = ""
        if record:
            depth_name = next((slot for slot in record["slots"]
                               if slot in depth_sources and slot != source), "")
        if not depth_name:
            token = _device_token(source)
            depth_name = next((name for name in depth_sources
                               if _device_token(name) == token), "")
        if not depth_name and len(depth_sources) == 1:
            depth_name = next(iter(depth_sources))
        pairs.append({
            "rgb_ref": ref,
            "rgb_source": source,
            "depth_source": depth_name or None,
            "depth_dir": depth_dirs.get(depth_name),
            "depth_video": depth_videos.get(depth_name),
            "device": record or {
                "key": _device_token(source) or "device_0",
                "name": _device_token(source) or source,
                "slots": [source, depth_name] if depth_name else [source],
                "calibration": "",
                "serial": "",
            },
            "batch_root": root,
            "episode_calibration": (
                episode_row.get("calibration") if isinstance(episode_row, dict)
                else {}
            ),
        })
    return pairs


def _intrinsics(values, width: int, height: int) -> dict | None:
    try:
        fx, fy, cx, cy = [float(v) for v in values[:4]]
    except (TypeError, ValueError, IndexError):
        return None
    if min(fx, fy) <= 0:
        return None
    return {"width": int(width), "height": int(height),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "model": "distortion.none", "coeffs": [0.0] * 5}


def _approx_intrinsics(width: int, height: int) -> dict:
    """Create a deterministic projection for a resolution-mapped depth pair.

    This fallback is only used when the uploaded batch explicitly identifies
    one physical D435 and has no calibration document.  The focal lengths are
    scaled with the image dimensions so an identity transform maps the two
    image grids by normalized coordinates instead of incorrectly treating
    848x480 pixels as 1280x720 pixels.  It never changes the stored depth
    codes; the manifest records that XY is approximate.
    """
    width = int(width)
    height = int(height)
    # A stable 16:9 pinhole approximation.  Keeping fx/fy proportional to
    # the corresponding dimensions makes the same normalized ray land at the
    # same normalized pixel in the other resolution.
    fx = float(width) * 0.9
    fy = float(height) * 0.9
    return {"width": int(width), "height": int(height),
            "fx": fx, "fy": fy,
            "cx": (float(width) - 1.0) / 2.0,
            "cy": (float(height) - 1.0) / 2.0,
            "model": "approximate_pinhole", "coeffs": [0.0] * 5}


def _is_d435_pair(pair: dict) -> bool:
    """Return whether metadata identifies the pair as one physical D435."""
    device = pair.get("device") or {}
    values = (
        device.get("kind"), device.get("name"), device.get("key"),
        pair.get("rgb_source"), pair.get("depth_source"),
    )
    text = " ".join(str(value or "").lower() for value in values)
    return "d435" in text


def _calibration_resolution(raw: dict) -> tuple[int, int] | None:
    """Return a calibration file's declared (width, height), if available."""
    if not isinstance(raw, dict):
        return None
    values = []
    depth_intrinsics = raw.get("depth_intrinsics")
    if isinstance(depth_intrinsics, dict):
        values.append((depth_intrinsics.get("width"),
                       depth_intrinsics.get("height")))
    values.append(raw.get("resolution"))
    for value in values:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                width, height = int(value[0]), int(value[1])
            except (TypeError, ValueError):
                continue
            if width > 0 and height > 0:
                return width, height
    return None


def _calibration_token_score(path: Path, raw: dict, pair: dict) -> int:
    """Prefer calibration names that identify the same physical device."""
    text = " ".join([
        path.stem,
        str(raw.get("name") or ""),
        str(pair.get("rgb_source") or ""),
        str(pair.get("depth_source") or ""),
        str((pair.get("device") or {}).get("name") or ""),
    ]).lower()
    tokens = set()
    for value in (pair.get("rgb_source"), pair.get("depth_source"),
                  (pair.get("device") or {}).get("name")):
        token = _device_token(value)
        if token and len(token) > 1:
            tokens.add(token)
    return sum(1 for token in tokens if token in text)


def _infer_calibration_path(pair: dict, depth_shape: tuple[int, int]) -> Path | None:
    """Find a calibration for this device without hard-coding D405/D435.

    Upload metadata from older sessions does not carry calibration paths.  A
    calibration is safe to infer only when its declared resolution equals the
    actual depth PNG resolution.  This prevents a D435 calibration from being
    silently applied to a different camera just because it is the first JSON
    file in the directory.
    """
    root = Path(pair["batch_root"])
    calibration_root = root / "calibration"
    if not calibration_root.is_dir():
        calibration_root = root / "meta" / "calibration"
    if not calibration_root.is_dir():
        return None
    depth_width, depth_height = int(depth_shape[1]), int(depth_shape[0])
    matches: list[tuple[int, Path]] = []
    for path in sorted(calibration_root.rglob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(raw, dict):
            continue
        if _calibration_resolution(raw) != (depth_width, depth_height):
            continue
        matches.append((_calibration_token_score(path, raw, pair), path))
    if not matches:
        return None
    matches.sort(key=lambda item: (-item[0], str(item[1])))
    return matches[0][1]


def _pair_calibration(pair: dict, demo, rgb_shape: tuple[int, int],
                      depth_shape: tuple[int, int], config: dict | None = None
                      ) -> tuple[dict, str]:
    """Build the calibration arguments expected by ``DepthHand3DDriver``.

    New calibration files may contain the full color/depth/transform schema.
    Legacy files without an explicit RGB/depth registration are not enough to
    produce camera-coordinate 3D.  The canonical S80C ``stereo_depth`` stream
    is an exception: the collector contract says it is already in the RGB
    pixel grid, so this direct-depth path uses identity registration and never
    uses the stereo baseline. A bundled calibration profile may only be used
    when its declared serial matches the physical device.
    """
    import copy
    root = pair["batch_root"]
    device = pair["device"]
    raw = None
    raw_path = ""
    calibration = str(device.get("calibration") or "")
    candidate = root / calibration if calibration else None
    if candidate is None or not candidate.is_file():
        candidate = _infer_calibration_path(pair, depth_shape)
    if candidate is not None and candidate.is_file():
        raw_path = str(candidate.relative_to(root))
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            raw = None
    raw = raw if isinstance(raw, dict) else {}
    if not raw:
        embedded = pair.get("episode_calibration")
        if isinstance(embedded, dict):
            # The migration stores calibration JSON inside the episode row as
            # {"head_stereo.json": {...}}.  Select a profile matching the
            # decoded depth resolution, then use the same validation below.
            candidates = [value for value in embedded.values()
                          if isinstance(value, dict)]
            dw, dh = int(depth_shape[1]), int(depth_shape[0])
            for profile in candidates:
                if _calibration_resolution(profile) in {(dw, dh), None}:
                    raw = copy.deepcopy(profile)
                    raw_path = "episode.calibration"
                    break
    rw, rh = int(rgb_shape[1]), int(rgb_shape[0])
    dw, dh = int(depth_shape[1]), int(depth_shape[0])
    if all(isinstance(raw.get(key), dict) for key in
           ("color_intrinsics", "depth_intrinsics", "depth_to_color")):
        color = copy.deepcopy(raw["color_intrinsics"])
        depth = copy.deepcopy(raw["depth_intrinsics"])
        transform = copy.deepcopy(raw["depth_to_color"])
        color["width"], color["height"] = rw, rh
        depth["width"], depth["height"] = dw, dh
        return {"color_intrinsics": color, "depth_intrinsics": depth,
                "depth_to_color": transform}, raw_path or "full_calibration"

    rgb_source = str(pair.get("rgb_source") or "").lower()
    color_camera = (
        raw.get("right_camera") if "right" in rgb_source
        else raw.get("left_camera")
    ) or raw.get("color_camera") or raw.get("left_camera") or {}
    depth_camera = raw.get("depth_camera") or {}
    depth_values = depth_camera.get("intrinsic") or color_camera.get("intrinsic")
    color_values = color_camera.get("intrinsic") or depth_values
    depth = _intrinsics(depth_values, dw, dh)
    color = _intrinsics(color_values, rw, rh)
    aligned_flag = (
        raw.get("depth_aligned_to_color") is True
        or raw.get("aligned_depth_to_color") is True
        or raw.get("depth_registered") is True
        or str(raw.get("alignment") or "").lower()
        in {"aligned", "registered", "color", "color_aligned"}
    )
    configured_alignment = str((config or {}).get("depth_alignment") or "").lower()
    direct_depth_flag = configured_alignment in {
        "direct", "aligned", "same_pixel", "same-pixel", "identity"
    }
    # S80C writes one canonical depth slot beside the two RGB slots. The
    # collector's depth image is the authoritative aligned measurement for
    # both views; do not attempt left/right stereo triangulation here.
    canonical_direct_depth = (
        str(pair.get("depth_source") or "").lower() == "stereo_depth"
        and (rw, rh) == (dw, dh)
    )
    if canonical_direct_depth:
        # This collector contract is stronger than the legacy calibration
        # requirement: stereo_depth is a registered depth image, not a second
        # stereo colour camera.  Use calibration intrinsics when present and
        # a stable size-derived projection when the upload omitted them.
        color = color or _approx_intrinsics(rw, rh)
        depth = depth or copy.deepcopy(color)
        return {"color_intrinsics": color, "depth_intrinsics": depth,
                "depth_to_color": {"rotation": np.eye(3).tolist(),
                                    "translation": [0.0, 0.0, 0.0]}}, \
               (raw_path or "direct_depth_same_pixel") + \
               (":direct_depth_same_pixel" if not aligned_flag
                else ":declared_identity_alignment")
    if depth is not None and color is not None and (rw, rh) == (dw, dh) \
            and (aligned_flag or direct_depth_flag or canonical_direct_depth):
        if canonical_direct_depth and not aligned_flag:
            depth = copy.deepcopy(color)
        return {"color_intrinsics": color, "depth_intrinsics": depth,
                "depth_to_color": {"rotation": np.eye(3).tolist(),
                                    "translation": [0.0, 0.0, 0.0]}}, \
               (raw_path or "direct_depth_same_pixel") + \
               (":direct_depth_same_pixel" if canonical_direct_depth and not aligned_flag
                else ":declared_identity_alignment")

    # Uploaded D435 batches commonly contain the canonical color stream at
    # 1280x720 and the metric depth stream at 848x480, but omit the optional
    # calibration directory.  The batch metadata still gives us a strong
    # device pairing (D435_rgb + D435_depth).  Use a normalized-resolution
    # mapping so the workflow produces a usable depth-lifted preview instead
    # of silently dropping the entire 3D node.  This is deliberately limited
    # to D435 metadata and is marked approximate in the output manifest;
    # precise RGB/depth registration still takes precedence when supplied.
    if _is_d435_pair(pair) and depth is None and color is None:
        return {
            "color_intrinsics": _approx_intrinsics(rw, rh),
            "depth_intrinsics": _approx_intrinsics(dw, dh),
            "depth_to_color": {
                "rotation": np.eye(3).tolist(),
                "translation": [0.0, 0.0, 0.0],
            },
        }, "approximate_d435_resolution_mapping"

    embedded = copy.deepcopy(demo._EMBEDDED_CALIB)
    declared_serial = str((pair.get("device") or {}).get("serial") or "")
    embedded_serial = str(embedded.get("serial") or "")
    if not declared_serial or not embedded_serial \
            or declared_serial != embedded_serial:
        raise ValueError(
            "calibration has no explicit color/depth alignment for "
            f"{pair.get('rgb_source') or 'device'}; "
            "same resolution is not sufficient"
        )
    if depth is not None:
        embedded["depth_intrinsics"] = depth
    embedded_depth = embedded.get("depth_intrinsics") or {}
    embedded_shape = (int(embedded_depth.get("height") or 0),
                      int(embedded_depth.get("width") or 0))
    if embedded_shape != (dh, dw):
        raise ValueError(
            f"no calibration matches depth resolution {dw}x{dh}"
            f" for {pair.get('rgb_source') or 'device'}")
    embedded["color_intrinsics"]["width"] = rw
    embedded["color_intrinsics"]["height"] = rh
    return embedded, raw_path or "embedded_d435_fallback"


def _is_depth_video_source(source_key: str | None, depth_camera: str) -> bool:
    """Distinguish a pure depth slot from an RGB slot sharing its device name."""
    key = str(source_key or "").lower().strip()
    depth_name = str(depth_camera or "").lower().strip()
    if not key:
        return False
    if key == depth_name:
        return True
    if "depth" not in key:
        return False
    # D435_depth_rgb / D435_depth_color are RGB slots, not depth videos.
    return not any(token in key for token in ("rgb", "color", "video"))


def _depth_frame_path(depth_dir: Path, depth_pngs: list[Path],
                      frame_index: int) -> Path | None:
    """Resolve a depth frame for either 000000- or 000001-based recordings."""
    offset = 0 if (depth_dir / "000000.png").exists() else 1
    named = depth_dir / f"{frame_index + offset:06d}.png"
    if named.exists():
        return named
    if 0 <= frame_index < len(depth_pngs):
        candidate = depth_pngs[frame_index]
        expected = frame_index + offset
        if not candidate.stem.isdigit() or int(candidate.stem) == expected:
            return candidate
    return None


def _run_2d_only_pair(demo, pair: dict, video_path: Path, config: dict,
                      ctx: JobContext, out_dir: Path,
                      reason: str) -> ArtifactRef | None:
    """Write a safe per-device 2D artifact when metric 3D is unavailable.

    A missing/ambiguous RGB-depth registration must never be replaced with a
    guessed camera transform.  The RGB detector is still useful for review,
    so keep its normalized keypoints in the same hand_3d namespace while
    explicitly marking all metric 3D points invalid in the manifest.
    """
    import cv2
    import pandas as pd

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    detector = demo.MediaPipeDetector(
        model_path=str(_MODEL), num_hands=2,
        det_conf=float(config.get("det_conf", 0.4)),
        track_conf=float(config.get("track_conf", 0.4)),
    )
    rows: list[dict] = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            hands = detector.detect(frame)
            row: dict[str, Any] = {"frame_index": int(frame_index)}
            for si in (0, 1):
                det = hands[si] if si < len(hands) else None
                if det is None:
                    kp = np.full(63, np.nan, np.float32).tolist()
                    label = ""
                    confidence = 0.0
                    present_2d = False
                else:
                    pts = np.asarray(det.landmarks, np.float32).reshape(-1, 2)
                    kp = np.zeros((21, 3), np.float32)
                    kp[:, :2] = pts / [max(1, width), max(1, height)]
                    kp[:, :2] = np.clip(kp[:, :2], 0.0, 1.0)
                    kp = kp.reshape(63).tolist()
                    label = str(det.label or "")
                    confidence = float(det.score)
                    present_2d = True
                row.update({
                    f"hand_{si}_present": False,
                    f"hand_{si}_2d_present": present_2d,
                    f"hand_{si}_landmarks_3d": np.full(
                        63, np.nan, np.float32).tolist(),
                    f"hand_{si}_keypoints": kp,
                    f"hand_{si}_label": label,
                    f"hand_{si}_reprojection_error": float("nan"),
                    f"hand_{si}_gesture": "",
                    f"hand_{si}_fingers": -1,
                    f"hand_{si}_confidence": confidence,
                })
            rows.append(row)
            frame_index += 1
            if total:
                ctx.progress(min(0.99, frame_index / max(1, total)))
    finally:
        cap.release()
        detector.close()
    if not rows:
        return None

    source_key = str(pair["rgb_source"])
    safe = _safe_source_key(source_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    device = pair.get("device") or {}
    manifest = {
        "frames": len(rows),
        "source_key": source_key,
        "rgb_source": source_key,
        "depth_camera": pair.get("depth_source"),
        "device_key": device.get("key"),
        "device_name": device.get("name"),
        "serial": device.get("serial"),
        "detector": "hand_landmarker",
        "mode": "2d_only",
        "unit": "image_normalized",
        "coordinate_frame": "image_normalized",
        "metric_3d_available": False,
        "processing_warnings": [reason],
        "render_video": "",
    }
    (out_dir / f"{safe}.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return ctx.ref("hand_3d", out_path, source_key=source_key,
                   metadata=manifest)


def _run_rgb_estimated_pair(pair: dict, video_path: Path, config: dict,
                            ctx: JobContext, out_dir: Path,
                            reason: str) -> ArtifactRef | None:
    """Process one depth-failed RGB source with the internal RGB 3D path."""
    import pandas as pd

    # Imported lazily to keep the depth helper independent during normal
    # module discovery.  Both modules are inside Data Acquisition; this is
    # not a dependency on the legacy Python project.
    from app.processing.modules.stereo_triangulate import (
        _detect_hands, _rgb_frame_rows,
    )

    video_path = Path(video_path)
    rgb_cfg = {
        "max_hands": int(config.get("max_hands", 2)),
        "min_detection_conf": float(config.get("min_detection_conf", 0.1)),
        "min_presence_conf": float(config.get("min_presence_conf", 0.1)),
        "min_tracking_conf": float(config.get("min_tracking_conf", 0.5)),
        "smooth": bool(config.get("smooth", True)),
        "freq_min": float(config.get("freq_min", 5.0)),
        "beta": float(config.get("beta", 0.05)),
        "device": str(config.get("device", "auto")),
    }
    det_df = _detect_hands(video_path, rgb_cfg, ctx.progress)
    rows = _rgb_frame_rows(det_df)
    if not rows:
        return None

    source_key = str(pair["rgb_source"])
    safe = _safe_source_key(source_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    device = pair.get("device") or {}
    manifest = {
        "frames": len(rows),
        "source_key": source_key,
        "rgb_source": source_key,
        "depth_camera": pair.get("depth_source"),
        "device_key": device.get("key"),
        "device_name": device.get("name"),
        "serial": device.get("serial"),
        "detector": "hand_landmarker",
        "mode": "rgb_estimated_3d",
        "unit": "rgb_estimated_meters",
        "coordinate_frame": "camera_relative",
        "metric_3d_available": False,
        "method": "MediaPipe RGB landmarks + hand-model PnP + image-scale fallback",
        "camera_model": "approximate_pinhole",
        "processing_warnings": [reason],
        "render_video": "",
    }
    (out_dir / f"{safe}.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return ctx.ref("hand_3d", out_path, source_key=source_key,
                   metadata=manifest)


def run_depth_hand_3d(ctx: JobContext, video_refs: list, config: dict,
                      *, allow_rgb_fallback: bool = True
                      ) -> tuple[dict[str, ArtifactRef] | None, str | None]:
    """Run depth-image hand 3D once per physical RGB/depth device pair.

    The first returned handle is ``hand_3d`` for old workflows; additional
    devices use ``hand_3d#2``, ``hand_3d#3`` and so on.  The worker runner
    already forwards ``#`` siblings when an edge targets ``hand_3d``, so old
    export graphs continue to work while new exporters can keep each source
    in its own namespace.
    """
    import cv2
    import pandas as pd

    demo = _load_demo()
    if demo is None:
        return None, "depth hand demo not found"
    if not video_refs:
        return None, "no video input"
    configured_depth = str(config.get("depth_camera") or "").strip()
    pairs = _find_device_pairs(ctx, video_refs)
    if configured_depth:
        pairs = [pair for pair in pairs
                 if pair.get("depth_source") == configured_depth]
    if not pairs:
        detail = f"depth/{configured_depth}/" if configured_depth else "depth/*/"
        return None, f"no RGB/depth device pairs at {detail}"

    out_dir = ctx.output_root / "hand_3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, ArtifactRef] = {}
    failures: list[str] = []
    failed_pairs: list[tuple[dict, str]] = []
    successful = 0
    for pair in pairs:
        rgb_ref = pair["rgb_ref"]
        depth_cam = pair.get("depth_source")
        depth_dir = pair.get("depth_dir")
        depth_video = pair.get("depth_video")
        video_path = ctx.resolve(rgb_ref)
        if not video_path or not video_path.exists():
            failures.append(f"{pair['rgb_source']}: video artifact missing")
            continue
        if (depth_dir is None and depth_video is None) or not depth_cam:
            reason = f"{pair['rgb_source']}: metric 3D unavailable (depth pair missing)"
            # Do not terminate the Hand Skeleton workflow with a 2D-only
            # artifact here.  The caller must continue into its standalone
            # RGB camera-relative 3D fallback.
            failures.append(reason)
            failed_pairs.append((pair, reason))
            continue
        depth_pngs = sorted(depth_dir.glob("*.png")) if depth_dir else []
        depth_reader = None
        if depth_video is not None:
            try:
                depth_reader = DepthVideoReader(depth_video)
                first_depth = depth_reader.read()
            except (OSError, RuntimeError, ValueError) as exc:
                if depth_reader is not None:
                    depth_reader.close()
                reason = f"{pair['rgb_source']}: metric 3D unavailable (depth video unreadable: {exc})"
                failures.append(reason)
                failed_pairs.append((pair, reason))
                continue
        else:
            first_depth = cv2.imread(str(depth_pngs[0]), cv2.IMREAD_UNCHANGED) \
                if depth_pngs else None
        if first_depth is None:
            if depth_reader is not None:
                depth_reader.close()
            reason = f"{pair['rgb_source']}: metric 3D unavailable (depth frame unreadable)"
            # Let stereo_triangulate run the RGB-only estimator instead of
            # writing a misleading artifact with no 3D points.
            failures.append(reason)
            failed_pairs.append((pair, reason))
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            if depth_reader is not None:
                depth_reader.close()
            failures.append(f"{pair['rgb_source']}: cannot open video")
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(depth_pngs)
        try:
            calib, calib_source = _pair_calibration(
                pair, demo, (height, width), first_depth.shape[:2], config)
            driver = DepthHand3DDriver(
                demo, str(_MODEL),
                det_conf=float(config.get("det_conf", 0.4)),
                track_conf=float(config.get("track_conf", 0.4)),
                fill=int(config.get("fill", 1)),
                propagate_max=int(config.get("propagate_max", 15)),
                color_intr=calib["color_intrinsics"],
                depth_to_color=calib["depth_to_color"],
                depth_intr=calib["depth_intrinsics"])
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            cap.release()
            if depth_reader is not None:
                depth_reader.close()
            reason = f"{pair['rgb_source']}: metric 3D unavailable ({exc})"
            # Calibration failure is also an RGB fallback case.
            failures.append(reason)
            failed_pairs.append((pair, reason))
            continue
        rows: list[dict] = []
        try:
            fi = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if depth_video is not None:
                    depth = first_depth if fi == 0 else depth_reader.read()
                else:
                    dpath = _depth_frame_path(depth_dir, depth_pngs, fi)
                    depth = (cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
                             if dpath is not None else None)
                res = driver.process_frame(demo, frame, depth, fi)
                rows.append(_row(demo, fi, res, frame.shape[1], frame.shape[0]))
                fi += 1
                if total > 0 and fi % 30 == 0:
                    ctx.progress(min(1.0, (successful + fi / total) /
                                     max(1, len(pairs))))
        finally:
            cap.release()
            if depth_reader is not None:
                depth_reader.close()
            stats = dict(driver.stats)
            driver.close()
        if not rows:
            reason = f"{pair['rgb_source']}: no video frames processed"
            failures.append(reason)
            failed_pairs.append((pair, reason))
            continue

        source_key = str(pair["rgb_source"])
        safe = _safe_source_key(source_key)
        out_path = out_dir / f"{safe}.parquet"
        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
        manifest = {
            "frames": len(rows),
            "source_key": source_key,
            "rgb_source": source_key,
            "depth_camera": depth_cam,
            "device_key": pair["device"].get("key"),
            "device_name": pair["device"].get("name"),
            "serial": pair["device"].get("serial"),
            "detector": "hand_landmarker",
            "mode": "depth_hand_3d",
            "unit": "camera_meters",
            "coordinate_frame": f"{source_key}_color_camera",
            "frame_offset": 1,
            "method": ("mediapipe 2D + direct aligned depth sampling + "
                       "slot tracking + One-Euro (per-camera intrinsics, meters)"),
            "calib_source": calib_source,
            "depth_alignment": (
                "resolution_scaled_same_device"
                if calib_source == "approximate_d435_resolution_mapping"
                else "direct_same_pixel"
            ),
            "stereo_triangulation": False,
            "calibration_quality": (
                "approximate"
                if calib_source == "approximate_d435_resolution_mapping"
                else "measured_or_declared"
            ),
            "processing_warnings": (
                ["RGB/Depth calibration file absent; XY uses normalized "
                 "D435 resolution mapping. Z comes from metric depth."]
                if calib_source == "approximate_d435_resolution_mapping"
                else []
            ),
            "smoothing": {"enabled": True, "freq_min": 3.0, "beta": 0.3},
            "slot_stats": stats,
            "render_video": "",
        }
        if failures:
            manifest["processing_warnings"] = list(failures)
        (out_dir / f"{safe}.manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        handle = "hand_3d" if successful == 0 else f"hand_3d#{successful + 1}"
        outputs[handle] = ctx.ref("hand_3d", out_path,
                                  source_key=source_key,
                                  metadata=manifest)
        successful += 1
        print(f"[depth_hand_3d] wrote {len(rows)} frames for {source_key} "
              f"-> {out_path} (stats: {stats})")

    # RGB-only estimated 3D is retained only for legacy callers that explicitly
    # allow it.  The canonical RGB-D_3D workflow passes False: a missing or
    # unreadable depth stream must skip that module instead of creating a
    # misleading metric-looking 3D artifact.
    if not allow_rgb_fallback:
        if not outputs:
            return None, "; ".join(failures) or "no metric depth output"
        reason = "; ".join(failures) if failures else None
        return outputs, reason

    # Preserve successful metric outputs while also covering every RGB source
    # whose depth stream/calibration failed for legacy callers. Previously any
    # successful pair made this function return early at the caller, silently
    # dropping those failed devices from the batch.
    for pair, reason in failed_pairs:
        video_path = ctx.resolve(pair["rgb_ref"])
        if not video_path or not video_path.exists():
            continue
        try:
            fallback = _run_rgb_estimated_pair(
                pair, video_path, config, ctx, out_dir, reason)
        except Exception as exc:
            failures.append(f"{pair['rgb_source']}: RGB fallback failed ({exc})")
            continue
        if fallback is not None:
            handle = ("hand_3d" if successful == 0
                      else f"hand_3d#{successful + 1}")
            outputs[handle] = fallback
            successful += 1

    if not outputs:
        return None, "; ".join(failures) or "no device pair processed"
    reason = "; ".join(failures) if failures else None
    return outputs, reason

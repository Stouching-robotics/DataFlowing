"""手部骨骼 — 手部 3D 骨骼识别(一体化模块,v1.3 world 方案)。

v1.3 起**不再做双目三角化**(米制 3D 效果不佳:鱼眼近距畸变 + 标定兜底 +
2D 抖动经三角化放大)。改为与 stereo_hands_demo_v1.0 同方案:

每路(左/右)独立 HandLandmarker 检测 → 2D 关键点 + 手部模型相对 3D
+ handedness;One-Euro 平滑(默认 5.0/0.05,槽位 + handedness 身份自愈);
再由本模块内置的 RGBWorldTracker 用 PnP/图像尺度估计相机相对位置，
并输出共享相机空间的 3D 骨架。关节角度基于手部模型坐标，渲染为左右
目并排的 2D 五指分色骨架 + 角度数字(demo 风格,原画面无矫正)。

数据先输出到 worker 临时目录的 hand_3d/hand_world.parquet，完成阶段合并到
对应 episode parquet，列结构与统一协议
3D 产物兼容(hand_0/1_present/label/landmarks_3d/confidence/
gesture/fingers);landmarks_3d 语义 = camera_relative。无深度时 unit
为 rgb_estimated_meters，明确表示 RGB 估计而非深度真值。

无标定依赖;标定文件与三角化代码全部移除(历史可查 git /
Temp/code_backup_20260814/)。

参考:
- stereo_hands_demo_v1.0:检测 + 平滑 + world 角度 + 渲染方案
- 21 关键点(MediaPipe 标准):Wrist(0)、拇指(1-4)、食指(5-8)、
  中指(9-12)、无名指(13-16)、小指(17-20)
"""

import json
from pathlib import Path

import numpy as np

from app.processing import ProcessingModule, JobContext, ArtifactRef, field
from app.processing.hand_render import FINGERS, draw_demo_style
from app.processing.hand_slot_assignment import (
    StableHandSlotAssigner,
    clip_normalized_xy,
)
from app.processing.registry import register
from app.processing.theme import HAND3D_COLOR

_BLACK_GLOVE_ROOT = Path(__file__).resolve().parents[1] / "black_glove"


def _resolve_hand_landmarker_model() -> Path:
    """Locate the bundled MediaPipe model across the supported layouts.

    The worker ships the model with its internal black-glove/D435 support
    package so the platform does not depend on sibling demo repositories.
    """
    candidates = (
        Path(__file__).resolve().parents[3] / "models" / "hand_landmarker.task",
        _BLACK_GLOVE_ROOT / "hand_landmarker.task",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"MediaPipe hand model not found; searched: {searched}")


HAND_LANDMARKER_MODEL = _resolve_hand_landmarker_model()

def _joint_specs():
    specs = []
    for finger, (ids, _c) in FINGERS.items():
        a, b, c, d = ids
        names = ("CMC", "MCP", "IP") if finger == "Thumb" else ("MCP", "PIP", "DIP")
        specs.append((finger, names[0], a, 0, b))
        specs.append((finger, names[1], b, a, c))
        specs.append((finger, names[2], c, b, d))
    return specs


JOINT_SPECS = _joint_specs()


def _angle_between(p_prev, p_vertex, p_next):
    """顶点处两条骨段的夹角(度)。180° = 伸直。"""
    v1 = np.asarray(p_prev, dtype=np.float64) - np.asarray(p_vertex, dtype=np.float64)
    v2 = np.asarray(p_next, dtype=np.float64) - np.asarray(p_vertex, dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def _finger_gesture_from_world(world_pts) -> tuple[str, int, dict]:
    """world 3D 21 点 → (label, bitmask, 关节角度表)。角度基于 world 坐标
    (视角无关,与 stereo_hands_demo 一致)。"""
    angles = {(f, j): _angle_between(world_pts[p], world_pts[v], world_pts[n])
              for f, j, v, p, n in JOINT_SPECS}
    extended: list[str] = []
    mask = 0
    bits = {"Thumb": 1, "Index": 2, "Middle": 4, "Ring": 8, "Pinky": 16}
    for finger in FINGERS:
        if finger == "Thumb":
            ok = angles.get((finger, "MCP"), 0) > 145 and angles.get((finger, "IP"), 0) > 150
        else:
            ok = angles.get((finger, "PIP"), 0) > 150 and angles.get((finger, "DIP"), 0) > 140
        if ok:
            extended.append(finger.lower())
            mask |= bits[finger]
    label = "fist" if not extended else "open:" + ",".join(extended)
    return label, mask, angles


class _OneEuroFilter:
    """单值 One-Euro 自适应低通滤波器(与 demo 同款公式)。"""

    def __init__(self, freq_min=5.0, beta=0.05, dcutoff=1.0):
        self.freq_min = freq_min
        self.beta = beta
        self.dcutoff = dcutoff
        self._prev_x = None
        self._prev_dx = 0.0
        self._prev_ts = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff) if cutoff > 1e-9 else 0.0
        return dt / (dt + tau) if tau > 0 else 1.0

    def __call__(self, x, ts_ms):
        if self._prev_x is None or self._prev_ts is None:
            self._prev_x, self._prev_dx, self._prev_ts = x, 0.0, ts_ms
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
        self._prev_x, self._prev_dx, self._prev_ts = x_hat, dx_hat, ts_ms
        return x_hat


class _HandSmoother:
    """双手关键点平滑(2D 归一化 x/y + world x/y/z),槽位身份自愈。

    槽位手消失重出现、或 handedness 标签变化(双手换槽)时重置该槽位
    全部滤波 —— 修复按 (手序号, 关节) 存状态导致双手换位时滤波历史
    串扰、又抖又飘的问题。
    """

    def __init__(self, freq_min=5.0, beta=0.05):
        self._freq_min = freq_min
        self._beta = beta
        self._filters: dict = {}       # (slot, j, axis, kind) -> _OneEuroFilter
        self._prev_labels: dict = {}
        self._prev_present: dict = {}

    def smooth_frame(self, hands: list, ts_ms: float) -> None:
        """hands: [(label, kp2d [[x,y,z]...21], world [[x,y,z]...21]), ...]"""
        for slot, item in enumerate(hands):
            if item is None:
                self._prev_present[slot] = False
                continue
            label, kp2d, world = item
            if not self._prev_present.get(slot, False) \
                    or self._prev_labels.get(slot) != label:
                self._filters = {k: v for k, v in self._filters.items()
                                 if k[0] != slot}
                self._prev_labels[slot] = label
            self._prev_present[slot] = True
            for j, p in enumerate(kp2d):
                for ax in (0, 1):
                    key = (slot, j, ax, "pix")
                    f = self._filters.get(key)
                    if f is None:
                        f = self._filters[key] = _OneEuroFilter(self._freq_min,
                                                                self._beta)
                    p[ax] = f(p[ax], ts_ms)
            for j, p in enumerate(world):
                for ax in (0, 1, 2):
                    key = (slot, j, ax, "w")
                    f = self._filters.get(key)
                    if f is None:
                        f = self._filters[key] = _OneEuroFilter(self._freq_min,
                                                                self._beta)
                    p[ax] = f(p[ax], ts_ms)
        for slot in list(self._prev_present):
            if slot >= len(hands):
                self._prev_present[slot] = False


def _detect_hands(video_path: Path, cfg: dict, progress_callback=None):
    """单路视频检测(HandLandmarker + RGB 相机相对 3D + 手势)→ DataFrame。

    列:frame_index / hand_0/1_keypoints(归一化 2D [21,3])/
    hand_0/1_world_landmarks [21,3] / handedness / confidence /
    hand_0/1_gesture / hand_0/1_fingers。
    """
    import cv2
    import mediapipe as mp
    import pandas as pd
    from app.processing.modules.rgb_world_tracking import RGBWorldTracker
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision as mp_vision

    max_hands = max(1, min(2, int(cfg.get("max_hands", 2))))
    min_detection = float(cfg.get("min_detection_conf", 0.1))
    min_presence = float(cfg.get("min_presence_conf", 0.1))
    min_tracking = float(cfg.get("min_tracking_conf", 0.5))
    smooth = bool(cfg.get("smooth", True))
    freq_min = float(cfg.get("freq_min", 5.0))
    beta = float(cfg.get("beta", 0.05))

    # device 配置映射到 MediaPipe delegate:auto/cuda:0 → 先试官方 GPU
    # delegate(实测 2× 加速、检出零损失、关键点中位差 ~1.2px),任何失败
    # 回退 CPU —— 工作流永不因 GPU 问题挂掉。
    requested_device = str(cfg.get("device", "auto"))
    landmarker = None
    if requested_device in ("auto", "cuda:0"):
        try:
            landmarker = mp_vision.HandLandmarker.create_from_options(
                mp_vision.HandLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(HAND_LANDMARKER_MODEL),
                        delegate=BaseOptions.Delegate.GPU),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=max_hands,
                    min_hand_detection_confidence=min_detection,
                    min_hand_presence_confidence=min_presence,
                    min_tracking_confidence=min_tracking,
                ))
        except Exception as exc:
            print(f"[stereo_hand] GPU delegate unavailable, fallback CPU: {exc}")
    if landmarker is None:
        landmarker = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=min_detection,
                min_hand_presence_confidence=min_presence,
                min_tracking_confidence=min_tracking,
            ))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rgb_tracker = RGBWorldTracker(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1,
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1,
        freq_min=float(cfg.get("position_freq_min", 5.0)),
        beta=float(cfg.get("position_beta", 0.05)),
    )

    smoother = _HandSmoother(freq_min, beta) if smooth else None
    slot_assigner = StableHandSlotAssigner(2)
    rows: list[dict] = []
    frame_index = 0
    ts_ms = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, ts_ms)
        ts_ms += int(1000.0 / max(1.0, fps))

        candidates = []
        for i, lms in enumerate(result.hand_landmarks[:2]):
            kp = [[float(p.x), float(p.y), float(p.z)] for p in lms[:21]]
            wld = ([[float(p.x), float(p.y), float(p.z)]
                    for p in result.hand_world_landmarks[i][:21]]
                   if i < len(result.hand_world_landmarks) else None)
            raw_label = str(result.handedness[i][0].category_name) \
                if i < len(result.handedness) else ""
            score = float(result.handedness[i][0].score) \
                if i < len(result.handedness) else 0.0
            center = np.mean(np.asarray(kp, dtype=np.float32)[:, :2], axis=0)
            candidates.append({
                "label": raw_label,
                "center": center,
                "payload": (kp, wld, raw_label, score),
            })
        ordered = slot_assigner.assign(candidates)
        slots: list = []
        hands_to_smooth: list = []
        for candidate in ordered:
            if candidate is None:
                slots.append(None)
                hands_to_smooth.append(None)
                continue
            kp, wld, raw_label, score = candidate["payload"]
            # Slot identity is already stabilized by image-space tracking.
            # Do not let a transient MediaPipe handedness flip rename the
            # slot and move the whole hand to the opposite display side.
            label = slot_assigner.stable_label(
                len(slots), str(candidate.get("label") or raw_label))
            slots.append((kp, wld, label, score))
            hands_to_smooth.append(
                (label, kp, wld or [[0.0, 0.0, 0.0]] * 21))
        if smoother is not None:
            smoother.smooth_frame(hands_to_smooth, ts_ms)

        row: dict = {"frame_index": frame_index}
        for si in range(2):
            if si < len(slots) and slots[si] is not None:
                kp, wld, label, score = slots[si]
                kp = clip_normalized_xy(kp).tolist()
                # Keep Hand3D's RGB fallback identical to Black Hand RGB3D:
                # derive the local estimated model from 2D image geometry and
                # the shared anatomical depth prior. MediaPipe's optional world
                # landmark orientation uses a different convention and made the
                # bare-hand and black-glove previews face differently.
                local = RGBWorldTracker.black_glove_reference_local(kp)
                local = (RGBWorldTracker.orient_palm_facing(local, label)
                         if local is not None else None)
                # Keep the RGB hand model's local geometry stable. PnP still
                # estimates the wrist's camera-relative position, but its
                # frame-by-frame rotation is not applied to the joints.
                preserve_model_geometry = True
                tracking = (rgb_tracker.update(
                    f"slot_{si}", kp, local,
                    frame_index / max(1.0, fps),
                    preserve_model_geometry=preserve_model_geometry)
                            if local is not None else None)
                gesture_points = local if local is not None else np.zeros((21, 3))
                label_g, mask, angles = _finger_gesture_from_world(gesture_points)
                row.update({
                    f"hand_{si}_keypoints": kp,
                    # Keep this legacy column name so downstream consumers
                    # remain compatible; its value is now a camera-relative
                    # RGB estimate rather than hand-local MediaPipe points.
                    f"hand_{si}_world_landmarks": (
                        tracking["landmarks_3d"].tolist()
                        if tracking is not None else None),
                    f"hand_{si}_landmarks_3d_local": (
                        tracking["landmarks_3d_local"].tolist()
                        if tracking is not None else (
                            local.tolist() if local is not None else None)),
                    f"hand_{si}_world_position": (
                        tracking["position"].tolist()
                        if tracking is not None else None),
                    f"hand_{si}_coordinate_frame": (
                        tracking["coordinate_frame"] if tracking else None),
                    f"hand_{si}_tracking_source": (
                        tracking["tracking_source"] if tracking else None),
                    f"hand_{si}_depth_source": "rgb_estimate" if tracking else None,
                    f"hand_{si}_reprojection_error": (
                        tracking["reprojection_error"] if tracking else None),
                    f"hand_{si}_handedness": label,
                    f"hand_{si}_confidence": score,
                    f"hand_{si}_gesture": label_g,
                    f"hand_{si}_fingers": mask,
                })
            else:
                row.update({
                    f"hand_{si}_keypoints": None,
                    f"hand_{si}_world_landmarks": None,
                    f"hand_{si}_landmarks_3d_local": None,
                    f"hand_{si}_world_position": None,
                    f"hand_{si}_coordinate_frame": None,
                    f"hand_{si}_tracking_source": None,
                    f"hand_{si}_depth_source": None,
                    f"hand_{si}_reprojection_error": None,
                    f"hand_{si}_handedness": None,
                    f"hand_{si}_confidence": None,
                    f"hand_{si}_gesture": "",
                    f"hand_{si}_fingers": -1,
                })
        rows.append(row)
        frame_index += 1
        if progress_callback and total:
            progress_callback(min(0.99, frame_index / total))
    cap.release()
    output = pd.DataFrame(rows)
    output.attrs["rgb_tracking"] = {
        "mode": "rgb_estimated_3d",
        "unit": "rgb_estimated_meters",
        "coordinate_frame": "camera_relative",
        "metric_3d_available": False,
        "method": ("MediaPipe RGB landmarks + hand-model PnP/image-scale "
                   "wrist placement; stable local hand geometry"),
        "orientation_mode": "stable_local_model_wrist_pnp_only",
        "camera_model": "approximate_pinhole",
    }
    return output


def _rgb_frame_rows(det_df) -> list[dict]:
    """Convert RGB detector rows into the canonical Hand 3D row schema."""
    def _flat(value, size: int):
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        return arr.tolist() if arr.size == size else None

    rows: list[dict] = []
    for _, r in det_df.iterrows():
        row = {"frame_index": int(r["frame_index"])}
        for si in ("hand_0", "hand_1"):
            wld = _flat(r.get(f"{si}_world_landmarks"), 63)
            local = _flat(r.get(f"{si}_landmarks_3d_local"), 63)
            position = _flat(r.get(f"{si}_world_position"), 3)
            kp = _flat(r.get(f"{si}_keypoints"), 63)
            if kp is not None:
                kp = clip_normalized_xy(np.asarray(kp, dtype=np.float32)
                                        .reshape(21, 3)).reshape(-1).tolist()
            present = wld is not None
            present_2d = kp is not None
            row.update({
                f"{si}_present": present,
                f"{si}_2d_present": present_2d,
                f"{si}_landmarks_3d": wld if present else np.full(
                    63, np.nan, dtype=np.float32).tolist(),
                f"{si}_landmarks_3d_local": local if local is not None
                else np.full(63, np.nan, dtype=np.float32).tolist(),
                f"{si}_world_position": position if position is not None
                else np.full(3, np.nan, dtype=np.float32).tolist(),
                f"{si}_coordinate_frame": str(
                    r.get(f"{si}_coordinate_frame") or ""),
                f"{si}_tracking_source": str(
                    r.get(f"{si}_tracking_source") or ""),
                f"{si}_depth_source": str(
                    r.get(f"{si}_depth_source") or ""),
                f"{si}_keypoints": kp if present_2d else np.full(
                    63, np.nan, dtype=np.float32).tolist(),
                f"{si}_label": str(r.get(f"{si}_handedness") or "")
                if present_2d else "",
                f"{si}_reprojection_error": float(
                    r.get(f"{si}_reprojection_error")
                    if r.get(f"{si}_reprojection_error") is not None
                    else float("nan")),
                f"{si}_gesture": str(r.get(f"{si}_gesture") or "")
                if present_2d else "",
                f"{si}_fingers": int(r.get(f"{si}_fingers") or 0)
                if present_2d else -1,
                f"{si}_confidence": float(r.get(f"{si}_confidence") or float("nan"))
                if present_2d else float("nan"),
            })
        rows.append(row)
    return rows


def _transcode_to_h264(src: Path, dst: Path) -> bool:
    """mp4v 临时文件 → H.264(ffmpeg 缺失时保留原文件)。

    短 GOP(-g 25,每秒 1 个关键帧)+ 关闭场景切换插帧:浏览器逐帧
    步进/多路同步纠偏时,seek 只需解码 ≤24 帧即到目标帧 —— 默认
    GOP(250 ≈ 10s)下 2560×800 大画面每次 seek 要补解最多 249 帧,
    是骨骼视频"卡顿/与原图不同步"的根因。
    """
    import shutil
    import subprocess

    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    try:
        subprocess.run(
            [exe, "-y", "-loglevel", "error", "-i", str(src),
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-g", "25", "-keyint_min", "25", "-sc_threshold", "0",
             "-movflags", "+faststart", "-an", str(dst)],
            check=True, capture_output=True, timeout=3600,
        )
        return dst.exists() and dst.stat().st_size > 0
    except Exception:
        return False


_draw_demo_style = draw_demo_style


@register
class StereoTriangulateModule(ProcessingModule):
    slug = "rgbd_to_3d_bare_hand"
    version = "1.8"
    category = "process"
    label = "RGB-D_3D_BareHand"
    icon = "ant-design:deployment-unit-outlined"
    color = HAND3D_COLOR
    inputs = ({"key": "video", "label": "RGB Video"},
              {"key": "depth", "label": "Depth"})
    # One canonical output for downstream nodes. Stereo right-view data is
    # kept as an internal auxiliary file, not a second connection port.
    outputs = ({"key": "hand_3d", "label": "Hand 3D"},)
    default_config = {"mode": "auto",
                      "max_hands": 2, "min_detection_conf": 0.1,
                      "min_presence_conf": 0.1,
                      "min_tracking_conf": 0.5,
                      "smooth": True, "freq_min": 5.0, "beta": 0.05,
                      "depth_camera": "",
                      "det_conf": 0.4, "track_conf": 0.4,
                      "fill": 1, "propagate_max": 15}
    config_schema = (
        # relative 保留为旧 JSON 的兼容值，但不再作为前端选项。这个
        # RGB-D 模块没有深度输入时只跳过，不生成估计 3D。
        field("mode", "select", "Mode", "auto",
              options=["auto", "depth"]),
        field("max_hands", "number", "Max hands", 2, min=1, max=2),
        field("min_detection_conf", "number", "Detection confidence", 0.1,
              min=0, max=1, step=0.05),
        field("min_presence_conf", "number", "Presence confidence", 0.1,
              min=0, max=1, step=0.05),
        field("min_tracking_conf", "number", "Tracking confidence", 0.5,
              min=0, max=1, step=0.05),
        field("smooth", "boolean", "One-Euro smoothing", True),
        field("freq_min", "number", "Smoothing cutoff Hz", 5.0,
              min=1, max=60, step=1),
        field("beta", "number", "Smoothing speed coefficient", 0.05,
              min=0, max=2, step=0.05),
        field("depth_camera", "string", "Depth source override (optional)", ""),
        field("det_conf", "number", "Depth-path detection confidence", 0.4,
              min=0.3, max=1, step=0.05),
        field("track_conf", "number", "Depth-path tracking confidence", 0.4,
              min=0.3, max=1, step=0.05),
        field("fill", "number", "Depth hole-fill passes", 1, min=1, max=3),
        field("propagate_max", "number", "Depth propagation cap (frames)", 15,
              min=5, max=120),
    )
    execution_target = "worker"
    capabilities = ("hand_3d", "stereo", "world_landmarks",
                    "d435_depth", "camera_meters")

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        video_refs = [r for key, r in ctx.incoming.items()
                      if r.kind == "video"
                      and (key == "video" or key.startswith("video#"))]
        if not video_refs:
            ctx.skip("RGB Video is not connected — skipped")

        depth_refs = [r for key, r in ctx.incoming.items()
                      if r.kind == "depth"
                      and (key == "depth" or key.startswith("depth#"))]
        if not depth_refs:
            ctx.skip("Depth is not connected — skipped")
        depth_ref = depth_refs[0]
        depth_path = ctx.resolve(depth_ref)
        if not depth_path or not depth_path.exists():
            ctx.skip("Connected Depth is unavailable — skipped")

        # The connected depth artifact is authoritative.  Passing its source
        # key prevents the helper's historical directory scan from selecting a
        # different camera's depth stream when several cameras are uploaded.
        metric_config = dict(ctx.config)
        depth_metadata = getattr(depth_ref, "metadata", None) or {}
        connected_depth_key = str(
            depth_ref.source_key or depth_metadata.get("depth_source") or ""
        ).strip()
        if connected_depth_key:
            metric_config["depth_camera"] = connected_depth_key

        from app.processing.modules.depth_hand_3d import run_depth_hand_3d
        out, reason = run_depth_hand_3d(
            ctx, video_refs, metric_config, allow_rgb_fallback=False)
        if out is not None:
            return out
        ctx.skip(f"Metric depth processing unavailable ({reason}) — skipped")

        video_refs.sort(key=lambda r: (r.source_key or "") or str(r.path))
        left_ref = video_refs[0]
        left_path = ctx.resolve(left_ref)
        if not left_path:
            ctx.skip("Video artifact missing — skipped")

        import pandas as pd

        cfg = {
            "max_hands": int(ctx.config.get("max_hands", 2)),
            "min_detection_conf": float(ctx.config.get("min_detection_conf", 0.1)),
            "min_presence_conf": float(ctx.config.get("min_presence_conf", 0.1)),
            "min_tracking_conf": float(ctx.config.get("min_tracking_conf", 0.5)),
            "smooth": bool(ctx.config.get("smooth", True)),
            "freq_min": float(ctx.config.get("freq_min", 5.0)),
            "beta": float(ctx.config.get("beta", 0.05)),
        }

        # 数据集规范:hand_0/1 = 两只实体手,左目为规范视图(主数据);
        # 右目检测结果单独保存(辅助视角,同样进 LeRobot 导出,_rcam 列)。
        def _frame_rows(det_df) -> list[dict]:
            return _rgb_frame_rows(det_df)

        if len(video_refs) >= 2:
            right_ref = video_refs[1]
            right_path = ctx.resolve(right_ref)
            if not right_path:
                ctx.skip("Video artifact missing — skipped")
            left_df = _detect_hands(left_path, cfg, ctx.progress)
            right_df = _detect_hands(right_path, cfg, ctx.progress)
        else:
            # 单目(无深度或深度不可用):单路 RGB 相机相对 3D,
            # 不产右目产物与并排渲染。
            right_ref = None
            right_path = None
            left_df = _detect_hands(left_path, cfg, ctx.progress)
            right_df = None

        rows = _frame_rows(left_df)
        right_rows = _frame_rows(right_df) if right_df is not None else []
        if not rows:
            ctx.skip("No frames — skipped")
        out_dir = ctx.output_root / "hand_3d"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "hand_world.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        # 右目单独一份:hand_3d_right/ 子目录,不干扰审核页/AI 标注对
        # hand_3d/*.parquet 的通配读取;渲染视频两路都画(与数据解耦)。
        right_out_dir = ctx.output_root / "hand_3d_right"
        right_out_path = right_out_dir / "hand_world_right.parquet"
        if right_df is not None:
            right_out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(right_rows).to_parquet(right_out_path, index=False)

        # ── 渲染:demo 风格左右目并排(原始画面 + 2D 骨架 + 角度数字)──
        render_video = None
        try:
            if right_path is None:
                raise ValueError("mono view: no stereo side-by-side render")
            import cv2

            def kp_px(df_row, slot, w, h):
                kp = df_row.get(f"{slot}_keypoints")
                if not kp:
                    return None
                return [[float(x) * w, float(y) * h] for x, y, _z in kp]

            cap_l = cv2.VideoCapture(str(left_path))
            cap_r = cv2.VideoCapture(str(right_path))
            W = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            H = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 800
            fps = cap_l.get(cv2.CAP_PROP_FPS) or 25.0
            raw_out = out_dir / "stereo_hands.mp4"
            writer = cv2.VideoWriter(str(raw_out),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (W * 2, H))
            i = 0
            while True:
                ok_l, fl = cap_l.read()
                ok_r, fr = cap_r.read()
                if not ok_l or not ok_r:
                    break
                lrow = left_df.iloc[i] if i < len(left_df) else None
                rrow = right_df.iloc[i] if i < len(right_df) else None
                for im, row in ((fl, lrow), (fr, rrow)):
                    if row is not None:
                        for slot in ("hand_0", "hand_1"):
                            px = kp_px(row, slot, W, H)
                            if not px:
                                continue
                            _draw_demo_style(im, px)
                writer.write(cv2.hconcat([fl, fr]))
                i += 1
            writer.release()
            cap_l.release()
            cap_r.release()
            if raw_out.exists() and raw_out.stat().st_size > 0:
                h264 = out_dir / "stereo_hands_h264.mp4"
                if _transcode_to_h264(raw_out, h264):
                    raw_out.unlink(missing_ok=True)
                    render_video = h264
        except Exception as exc:
            print(f"[stereo_hand] render skipped: {exc}")

        tracking_meta = dict((left_df.attrs.get("rgb_tracking") or {}))
        manifest = {
            "frames": len(rows),
            "left_source": left_ref.source_key,
            "right_source": right_ref.source_key if right_ref else None,
            "right_parquet": str(right_out_path) if right_df is not None else None,
            "views": len(video_refs),
            "detector": "hand_landmarker",
            "mode": tracking_meta.get("mode", "rgb_estimated_3d"),
            # RGB-only placement is shared by hands in the same camera frame,
            # but it is still an estimate because no depth sensor was used.
            "unit": tracking_meta.get("unit", "rgb_estimated_meters"),
            "coordinate_frame": tracking_meta.get("coordinate_frame",
                                                    "camera_relative"),
            "metric_3d_available": bool(
                tracking_meta.get("metric_3d_available", False)),
            "method": tracking_meta.get("method", "RGB hand pose estimate"),
            "camera_model": tracking_meta.get("camera_model",
                                                "approximate_pinhole"),
            "smoothing": {
                "enabled": cfg["smooth"],
                "freq_min": cfg["freq_min"],
                "beta": cfg["beta"],
            },
            "render_video": str(render_video) if render_video else "",
        }
        if depth_fallback_reason:
            manifest["depth_fallback_reason"] = depth_fallback_reason
            # 旧审核页/导出器兼容历史字段。
            manifest["d435_fallback_reason"] = depth_fallback_reason
        (out_dir / "hand_world.manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        if right_df is not None and right_ref is not None:
            # Keep the per-view artifact self-describing.  The parent manifest
            # describes the stereo pair and therefore mentions both sources;
            # it must not be the only source declaration used to resolve the
            # right parquet in review APIs.
            right_manifest = {
                **manifest,
                "source_key": right_ref.source_key,
                "camera": right_ref.source_key,
                "view": "right",
                "artifact": "hand_world_right",
            }
            (right_out_dir / "hand_world_right.manifest.json").write_text(
                json.dumps(right_manifest, indent=2), encoding="utf-8")

        source_key = str(left_ref.source_key or "stereo")
        out: dict[str, ArtifactRef] = {
            "hand_3d": ctx.ref("hand_3d", out_path, source_key=source_key,
                               metadata=manifest),
        }
        # The right-view parquet remains available inside the processed run
        # for diagnostics, but is intentionally not exposed as a workflow
        # output. Downstream modules consume one canonical Hand 3D stream.
        if render_video:
            out["render_video"] = ctx.ref(
                "video", render_video, source_key=source_key,
                metadata={"skeleton": True, "stereo_side_by_side": True})
        return out

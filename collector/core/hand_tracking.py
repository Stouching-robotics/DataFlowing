"""
手部关键点后处理独立模块 —— 零 Qt 依赖，纯函数式 API。

支持两种追踪模式:
  - "glove": YOLO 检测黑色手套 + RTMPose 关键点 (仅 2D)
  - "bare":  MediaPipe 裸手追踪 (2D + 3D world_landmarks)

bare 模式下额外保存 3D 世界坐标到 keypoints/hand_pose_3d/，
供下游 3D 可视化与数据集导出使用。

用法:
    from core.hand_tracking import process_session, load_hand_kpts, draw_kpts_overlay

    result = process_session("/path/to/session", mode="bare")
    kpts = load_hand_kpts("/path/to/session")
"""

from __future__ import annotations
import os
import sys
import json
import time
from typing import Optional, Callable

import numpy as np
import cv2
import pyarrow as pa
import pyarrow.parquet as pq

from config import settings
from core.helpers import (
    hand_kpts_parquet_path, hand_3d_parquet_path, auto_labels_parquet_path,
    data_parquet_path, video_mp4_path, detect_session_format,
    egodata_video_path, egodata_metadata_path,
    _session_kpts_hand_kpts_path, _session_kpts_hand_3d_path,
    _session_kpts_auto_labels_path,
    _legacy_hand_kpts_path, _legacy_hand_3d_path, _legacy_auto_labels_path,
)

# ── 将本地 tools/hand_detection 目录加入 sys.path ────────────
_HAND_DET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "tools", "hand_detection"))
_HAND_DET_DIR = os.path.abspath(_HAND_DET_DIR)
if _HAND_DET_DIR not in sys.path and os.path.isdir(_HAND_DET_DIR):
    sys.path.insert(0, _HAND_DET_DIR)

# ── 常量 ────────────────────────────────────────────────

PER_HAND_DIM = 21 * 2 + 4   # 46: 21点×2坐标 + 4框坐标
MAX_HANDS = 2
TOTAL_DIM = PER_HAND_DIM * MAX_HANDS  # 92

PER_HAND_3D_DIM = 21 * 3   # 63: xyz per landmark

# ── Parquet Schema ──────────────────────────────────────

_HAND_KPTS_SCHEMA = pa.schema([
    ("frame_index", pa.int32()),
    ("num_hands",   pa.int32()),
    ("hand_data",   pa.list_(pa.float32(), TOTAL_DIM)),
    ("track_ids",   pa.list_(pa.int32(), MAX_HANDS)),
])

_HAND_3D_SCHEMA = pa.schema([
    ("frame_index",          pa.int32()),
    ("hand_0_present",       pa.bool_()),
    ("hand_0_landmarks_3d",  pa.list_(pa.float32(), PER_HAND_3D_DIM)),
    ("hand_0_label",         pa.string()),
    ("hand_1_present",       pa.bool_()),
    ("hand_1_landmarks_3d",  pa.list_(pa.float32(), PER_HAND_3D_DIM)),
    ("hand_1_label",         pa.string()),
])

_AUTO_LABELS_SCHEMA = pa.schema([
    ("frame_index",             pa.int32()),
    ("hand_0_gesture",          pa.string()),
    ("hand_0_extended",         pa.list_(pa.string())),
    ("hand_0_extended_count",   pa.int32()),
    ("hand_0_fist",             pa.bool_()),
    ("hand_0_pinch",            pa.bool_()),
    ("hand_0_center_x",         pa.float32()),
    ("hand_0_center_y",         pa.float32()),
    ("hand_0_motion",           pa.float32()),
    ("hand_1_gesture",          pa.string()),
    ("hand_1_extended",         pa.list_(pa.string())),
    ("hand_1_extended_count",   pa.int32()),
    ("hand_1_fist",             pa.bool_()),
    ("hand_1_pinch",            pa.bool_()),
    ("hand_1_motion",           pa.float32()),
    ("two_hand_distance",       pa.float32()),
    ("contact",                 pa.string()),
])

# ── 延迟导入（重型依赖仅处理时加载） ───────────────────

_HandPipeline = None
_hand_common = None


def _lazy_import_pipeline():
    """导入 HandPipeline（含 ultralytics/torch/CUDA）。"""
    global _HandPipeline
    if _HandPipeline is None:
        try:
            from hand_pipeline import HandPipeline as _HP
            _HandPipeline = _HP
        except ImportError as e:
            raise ImportError(
                f"无法导入 glove 模式依赖 (hand_pipeline): {e}\n"
                "请确保已安装: pip install ultralytics torch rtmlib"
            ) from e
    return _HandPipeline


def _lazy_import_common():
    """导入 hand_common（纯 numpy/cv2）。"""
    global _hand_common
    if _hand_common is None:
        try:
            from hand_common import draw_hand, compute_joint_angles, count_extended_fingers, FINGERS
            _hand_common = {
                "draw_hand": draw_hand,
                "compute_joint_angles": compute_joint_angles,
                "count_extended_fingers": count_extended_fingers,
                "FINGERS": FINGERS,
            }
        except ImportError as e:
            raise ImportError(
                f"无法导入 hand_common: {e}\n"
                "请确保 tools/hand_detection/ 目录存在且包含 hand_common.py"
            ) from e
    return _hand_common


# ═══════════════════════════════════════════════════════════
#  数据打包
# ═══════════════════════════════════════════════════════════

def _pack_hand_data(boxes, kpts, track_ids) -> np.ndarray:
    """将一帧手部结果打包为固定长度 float32 数组 (TOTAL_DIM,)。"""
    arr = np.zeros(TOTAL_DIM, dtype=np.float32)
    num = min(len(boxes), MAX_HANDS)
    for h in range(num):
        off = h * PER_HAND_DIM
        pts = kpts[h] if kpts is not None and h < len(kpts) else np.zeros((21, 2))
        for j in range(min(21, len(pts))):
            arr[off + j * 2] = float(pts[j][0])
            arr[off + j * 2 + 1] = float(pts[j][1])
        box = boxes[h] if h < len(boxes) else [0, 0, 0, 0]
        for j in range(4):
            arr[off + 42 + j] = float(box[j])
    return arr


def _unpack_hand_data(arr: np.ndarray) -> tuple:
    """从打包数组恢复 (boxes, kpts, num_hands)。"""
    num = 0; boxes = []; kpts = []
    for h in range(MAX_HANDS):
        off = h * PER_HAND_DIM
        box = [float(arr[off + 42 + j]) for j in range(4)]
        if box[2] > 1 and box[3] > 1:
            pts = np.array([[float(arr[off + j * 2]),
                             float(arr[off + j * 2 + 1])] for j in range(21)],
                           dtype=np.float32)
            boxes.append(box); kpts.append(pts); num += 1
    return boxes, kpts, num


def _pack_hand_3d_data(world_landmarks_list) -> np.ndarray:
    """将单帧 3D 数据打包为 (PER_HAND_3D_DIM,) float32。"""
    arr = np.zeros(PER_HAND_3D_DIM, dtype=np.float32)
    wl = world_landmarks_list[0] if world_landmarks_list else None
    if wl is not None:
        flat = np.array(wl, dtype=np.float32).ravel()[:PER_HAND_3D_DIM]
        arr[:len(flat)] = flat
    return arr


# ═══════════════════════════════════════════════════════════
#  关键点叠加绘制
# ═══════════════════════════════════════════════════════════

def draw_kpts_overlay(frame: np.ndarray, data: np.ndarray,
                      track_ids: list = None) -> np.ndarray:
    """在 BGR 帧上叠加手部关键点。"""
    hc = _lazy_import_common()
    overlay = frame.copy()
    boxes, kpts, num = _unpack_hand_data(data)
    if num == 0:
        return overlay
    for i, (box, pts) in enumerate(zip(boxes, kpts)):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 220), 2, cv2.LINE_AA)
        tid = track_ids[i] if track_ids and i < len(track_ids) else i
        label = f"Hand #{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 200, 200), -1)
        cv2.putText(overlay, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        hc["draw_hand"](overlay, pts, angles=None, show_angles=False,
                        kpt_scores=None, thr=0.0)
        try:
            angles = hc["compute_joint_angles"](pts)
            extended = hc["count_extended_fingers"](angles)
            ext_text = "open: " + ",".join(extended) if extended else "fist"
            cv2.putText(overlay, ext_text, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 120), 1, cv2.LINE_AA)
        except Exception:
            pass
    return overlay


# ═══════════════════════════════════════════════════════════
#  手势分类
# ═══════════════════════════════════════════════════════════

PINCH_THRESH = 30.0
TWO_HAND_TOUCH_THRESH = 80.0


def classify_gesture(extended: list, thumb_tip_pt: np.ndarray,
                     index_tip_pt: np.ndarray) -> dict:
    """根据伸直手指 + 指尖距离分类手势。"""
    n = len(extended)
    pinch = (np.linalg.norm(thumb_tip_pt - index_tip_pt) < PINCH_THRESH)
    if n == 0:
        gesture, fist = "fist", True
    elif n >= 4:
        gesture, fist = ("open" if not pinch else "ok"), False
    elif n == 1:
        if "Index" in extended:   gesture, fist = "point", False
        elif "Thumb" in extended: gesture, fist = "hook", False
        else:                     gesture, fist = "unknown", False
    elif pinch and n == 3:
        gesture, fist = "ok", False
    elif pinch:
        gesture, fist = "pinch", False
    elif n in (2, 3):
        gesture, fist = "grab", False
    else:
        gesture, fist = "unknown", False
    return {"gesture": gesture, "pinch": bool(pinch), "fist": fist,
            "extended_count": n}


def _thumb_tip(kpts: np.ndarray) -> np.ndarray:
    return kpts[4] if len(kpts) > 4 else np.zeros(2)


def _index_tip(kpts: np.ndarray) -> np.ndarray:
    return kpts[8] if len(kpts) > 8 else np.zeros(2)


def _hand_center(kpts: np.ndarray) -> np.ndarray:
    wrist = kpts[0] if len(kpts) > 0 else np.zeros(2)
    middle_mcp = kpts[9] if len(kpts) > 9 else np.zeros(2)
    return (wrist + middle_mcp) / 2.0


def _compute_contact(motion_0: float, motion_1: float,
                     sensors: Optional[dict]) -> str:
    max_p = 0.0
    if sensors:
        for arr in sensors.values():
            if arr is not None and len(arr) > 0:
                max_p = max(max_p, float(arr.max()))
    motion = max(motion_0, motion_1) if motion_0 >= 0 else motion_0
    if max_p > 500 and motion > 15:
        return "grasping"
    elif max_p > 500:
        return "holding"
    elif motion > 10:
        return "reaching"
    elif max_p > 0:
        return "resting"
    return "none"


# ═══════════════════════════════════════════════════════════
#  主入口: 批量处理一个会话
# ═══════════════════════════════════════════════════════════

def process_session(session_path: str,
                    mode: str = "",
                    detector: str = "",
                    det_device: str = "cuda",
                    pose_device: str = "cuda",
                    progress_cb: Optional[Callable[[int, int], None]] = None,
                    status_cb: Optional[Callable[[str], None]] = None,
                    cancel_check: Optional[Callable[[], bool]] = None,
                    ) -> dict:
    """对已录制会话逐帧提取手部关键点，写入 parquet。

    bare 模式额外保存 3D world_landmarks 到独立 parquet。

    Returns:
        {"success": True, "frames": N, "elapsed": s, "fps": float,
         "mode": str, "session_path": str}
        或 {"success": False, "error": msg}
    """
    if not os.path.isdir(session_path):
        return {"success": False, "error": f"会话目录不存在: {session_path}"}

    mode = mode or settings.HAND_TRACK_MODE
    det_path = detector or settings.HAND_DET_MODEL
    det_device = det_device or settings.HAND_DET_DEVICE
    pose_device = pose_device or settings.HAND_POSE_DEVICE

    out_path = hand_kpts_parquet_path(session_path)
    out_3d_path = hand_3d_parquet_path(session_path)
    t_start = time.perf_counter()

    try:
        # ── 加载管线 ──────────────────────────────────
        _is_mediapipe = False
        if mode == "bare":
            if status_cb: status_cb("loading (MediaPipe bare hand)")
            _mp_model_src = settings.HAND_MEDIAPIPE_MODEL
            _mp_cache = os.path.join(os.path.expanduser("~"), ".cache",
                                     "hand_landmarker.task")
            if not os.path.isfile(_mp_cache) or \
               os.path.getsize(_mp_cache) != os.path.getsize(_mp_model_src):
                os.makedirs(os.path.dirname(_mp_cache), exist_ok=True)
                import shutil
                shutil.copy2(_mp_model_src, _mp_cache)
            from hand_pipeline_mediapipe import MediaPipeHandPipeline
            pipeline = MediaPipeHandPipeline(
                model_path=_mp_cache, num_hands=MAX_HANDS, mirror=False)
            _is_mediapipe = True
        else:
            _lazy_import_pipeline()
            try:
                import torch
                if not torch.cuda.is_available():
                    if det_device == "cuda": det_device = "cpu"
                    if pose_device == "cuda": pose_device = "cpu"
            except Exception:
                det_device = "cpu"; pose_device = "cpu"
            pipeline = _HandPipeline(
                detector=det_path, det_device=det_device,
                pose_device=pose_device, max_hands=MAX_HANDS)
            _is_mediapipe = False

        if status_cb: status_cb("processing")

        # ── 找视频 ────────────────────────────────────
        fmt = detect_session_format(session_path)
        cameras = {}
        if fmt == "egodata":
            meta_path = egodata_metadata_path(session_path)
        else:
            meta_path = os.path.join(session_path, "meta", "info.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                cameras = json.load(f).get("cameras", {})

        # 为 EgoData 格式构造视频路径（cameras 元数据不含 "video" 键）
        if fmt == "egodata" and cameras:
            for cam_name in list(cameras.keys()):
                # 优先 EgoData 标准路径, 回退 flat videos/<cam>.mp4
                for vp in [egodata_video_path(session_path, cam_name),
                           os.path.join(session_path, "videos", f"{cam_name}.mp4")]:
                    if os.path.isfile(vp):
                        cameras[cam_name]["video"] = vp
                        break

        if not cameras:
            vdir = os.path.join(session_path, "videos")
            if os.path.isdir(vdir):
                for cam_name in sorted(os.listdir(vdir)):
                    if fmt == "egodata":
                        mp4 = egodata_video_path(session_path, cam_name)
                        flat = os.path.join(session_path, "videos", f"{cam_name}.mp4")
                        if os.path.isfile(mp4):
                            cameras[cam_name] = {"video": mp4}
                        elif cam_name.endswith(".mp4") and os.path.isfile(flat):
                            base = cam_name[:-4]
                            cameras[base] = {"video": flat}
                    else:
                        mp4 = video_mp4_path(session_path, cam_name)
                        if os.path.isfile(mp4):
                            cameras[cam_name] = {"video": mp4}
                        else:
                            old = os.path.join(vdir, cam_name, "chunk_000000.mp4")
                            if os.path.isfile(old):
                                cameras[cam_name] = {"video": old}

        if not cameras:
            return {"success": False, "error": "未找到视频文件"}

        # 只保留有视频文件的 RGB 摄像机（排除 depth）
        _rgb_cameras = {}
        for cam_id, cam_info in cameras.items():
            if isinstance(cam_info, dict) and cam_info.get("type") == "depth":
                continue
            if cam_id.lower().endswith("_depth"):
                continue
            vp = cam_info.get("video", "")
            if os.path.isfile(vp):
                _rgb_cameras[cam_id] = cam_info
        cameras = _rgb_cameras

        if not cameras:
            return {"success": False, "error": "未找到可播放的视频文件"}

        first_cam = list(cameras.keys())[0]
        video_path = cameras[first_cam].get("video", "")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "error": f"无法打开视频: {video_path}"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        # ── 逐帧处理 ──────────────────────────────────
        rows = []
        rows_3d = []
        frame_idx = 0

        while True:
            if cancel_check and cancel_check():
                cap.release()
                return {"success": False, "error": "cancelled"}

            ok, frame = cap.read()
            if not ok:
                break

            if _is_mediapipe:
                result = pipeline.process(frame)
                boxes, kpts_list, tids_raw = [], [], []
                world_3d_per_hand = []
                handedness = []
                for i, hand in enumerate(result.hands):
                    lms = hand.landmarks  # (21, 2)
                    x1, y1 = lms.min(axis=0); x2, y2 = lms.max(axis=0)
                    boxes.append([float(x1), float(y1), float(x2), float(y2)])
                    kpts_list.append(lms)
                    tids_raw.append(i)
                    wl = getattr(hand, "world_landmarks", None)
                    world_3d_per_hand.append(wl)
                    hlabel = getattr(hand, "label", None) or getattr(hand, "handedness", "Unknown")
                    handedness.append(str(hlabel))
                kpts = np.array(kpts_list, dtype=np.float32) if kpts_list \
                    else np.zeros((0, 21, 2), dtype=np.float32)
                track_ids = tids_raw
            else:
                boxes, kpts, scores, track_ids = pipeline.process(frame)
                if kpts is None: kpts = np.zeros((0, 21, 2), dtype=np.float32)
                world_3d_per_hand = [None] * len(boxes)
                handedness = ["Unknown"] * len(boxes)

            # 2D 打包
            packed = _pack_hand_data(boxes, kpts, track_ids)
            num = min(len(boxes), MAX_HANDS)
            tids = list(track_ids[:MAX_HANDS]) if track_ids else []
            tids += [0] * (MAX_HANDS - len(tids))
            rows.append({
                "frame_index": frame_idx,
                "num_hands": num,
                "hand_data": packed.tolist(),
                "track_ids": tids,
            })

            # 3D 打包 (bare 模式)
            if _is_mediapipe:
                hl = handedness if handedness else []
                wl = world_3d_per_hand if world_3d_per_hand else []
                row_3d = {
                    "frame_index": frame_idx,
                    "hand_0_present": len(boxes) >= 1,
                    "hand_0_landmarks_3d": _pack_hand_3d_data(wl[0:1]).tolist(),
                    "hand_0_label": hl[0] if hl else "Unknown",
                    "hand_1_present": len(boxes) >= 2,
                    "hand_1_landmarks_3d": _pack_hand_3d_data(wl[1:2]).tolist(),
                    "hand_1_label": hl[1] if len(hl) >= 2 else "Unknown",
                }
                rows_3d.append(row_3d)

            frame_idx += 1
            if progress_cb and frame_idx % 10 == 0:
                progress_cb(frame_idx, total_frames)

        cap.release()

        if cancel_check and cancel_check():
            return {"success": False, "error": "cancelled"}

        # ── 写入 2D parquet ───────────────────────────
        if rows:
            table = pa.table({
                "frame_index": pa.array([r["frame_index"] for r in rows], pa.int32()),
                "num_hands":   pa.array([r["num_hands"] for r in rows], pa.int32()),
                "hand_data":   pa.array([r["hand_data"] for r in rows],
                                        pa.list_(pa.float32(), TOTAL_DIM)),
                "track_ids":   pa.array([r["track_ids"] for r in rows],
                                        pa.list_(pa.int32(), MAX_HANDS)),
            }, schema=_HAND_KPTS_SCHEMA)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pq.write_table(table, out_path, compression="zstd")

        # ── 写入 3D parquet (bare 模式) ───────────────
        if rows_3d:
            table_3d = pa.table({
                "frame_index":          pa.array([r["frame_index"] for r in rows_3d], pa.int32()),
                "hand_0_present":       pa.array([r["hand_0_present"] for r in rows_3d], pa.bool_()),
                "hand_0_landmarks_3d":  pa.array([r["hand_0_landmarks_3d"] for r in rows_3d],
                                                  pa.list_(pa.float32(), PER_HAND_3D_DIM)),
                "hand_0_label":         pa.array([r["hand_0_label"] for r in rows_3d], pa.string()),
                "hand_1_present":       pa.array([r["hand_1_present"] for r in rows_3d], pa.bool_()),
                "hand_1_landmarks_3d":  pa.array([r["hand_1_landmarks_3d"] for r in rows_3d],
                                                  pa.list_(pa.float32(), PER_HAND_3D_DIM)),
                "hand_1_label":         pa.array([r["hand_1_label"] for r in rows_3d], pa.string()),
            }, schema=_HAND_3D_SCHEMA)
            os.makedirs(os.path.dirname(out_3d_path), exist_ok=True)
            pq.write_table(table_3d, out_3d_path, compression="zstd")

        elapsed = time.perf_counter() - t_start
        result = {
            "success": True, "frames": frame_idx, "elapsed": elapsed,
            "fps": frame_idx / elapsed if elapsed > 0 else 0,
            "mode": mode, "session_path": session_path,
        }
        if status_cb:
            status_cb(f"done ({frame_idx} frames in {elapsed:.1f}s, "
                      f"{frame_idx/elapsed:.1f} fps)")
        if progress_cb:
            progress_cb(frame_idx, total_frames)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════
#  自动标注
# ═══════════════════════════════════════════════════════════

MIN_SEGMENT_FRAMES = 10


def label_session(session_path: str,
                  progress_cb: Optional[Callable[[int, int], None]] = None,
                  cancel_check: Optional[Callable[[], bool]] = None,
                  ) -> dict:
    """对手部关键点数据逐帧计算手势标签，写入 auto_labels.parquet + auto_actions.jsonl。

    Returns:
        {"success": True/False, "frames": N, "error": ""}
    """
    t_start = time.perf_counter()

    try:
        # 1. 加载手部关键点
        hand_kpts = load_hand_kpts(session_path)
        if not hand_kpts:
            return {"success": False, "error": "手部关键点数据不存在，请先提取关键点"}

        # 2. 加载传感器数据
        sensor_data = None
        sensor_path = data_parquet_path(session_path)
        if os.path.isfile(sensor_path):
            try:
                sensor_rows = pq.read_table(sensor_path).to_pylist()
                sensor_data = {}
                for sn in settings.SENSOR_NAMES:
                    sensor_data[sn] = {}
                for r in sensor_rows:
                    fi = r["frame_index"]
                    for sn in settings.SENSOR_NAMES:
                        col = f"observation.{sn}"
                        val = r.get(col)
                        if val is not None:
                            sensor_data[sn][fi] = np.array(val, dtype=np.float32)
            except Exception:
                sensor_data = None

        # 3. 逐帧标注
        _lazy_import_common()
        frame_indices = sorted(hand_kpts.keys())
        total = len(frame_indices)
        rows = []
        prev_centers = [None, None]

        for idx, fi in enumerate(frame_indices):
            if cancel_check and cancel_check():
                return {"success": False, "error": "cancelled"}

            kpt = hand_kpts[fi]
            boxes, kpts_list, num = _unpack_hand_data(kpt["hand_data"])
            row = {"frame_index": fi}

            for h in range(2):
                prefix = f"hand_{h}"
                if h < num and h < len(kpts_list):
                    pts = kpts_list[h]
                    tt = _thumb_tip(pts)
                    it = _index_tip(pts)
                    hc_ = _lazy_import_common()
                    angles = hc_["compute_joint_angles"](pts)
                    extended = hc_["count_extended_fingers"](angles)
                    result = classify_gesture(extended, tt, it)

                    center = _hand_center(pts)
                    motion = 0.0
                    if prev_centers[h] is not None:
                        motion = float(np.linalg.norm(center - prev_centers[h]))
                    prev_centers[h] = center

                    row[f"{prefix}_gesture"]        = result["gesture"]
                    row[f"{prefix}_extended"]       = extended
                    row[f"{prefix}_extended_count"] = result["extended_count"]
                    row[f"{prefix}_fist"]           = result["fist"]
                    row[f"{prefix}_pinch"]          = result["pinch"]
                    row[f"{prefix}_center_x"]       = float(center[0])
                    row[f"{prefix}_center_y"]       = float(center[1])
                    row[f"{prefix}_motion"]         = motion
                else:
                    for key in ["gesture", "extended", "extended_count",
                                "fist", "pinch", "center_x", "center_y", "motion"]:
                        row[f"{prefix}_{key}"] = {
                            "gesture": "none", "extended": [], "extended_count": 0,
                            "fist": False, "pinch": False,
                            "center_x": -1.0, "center_y": -1.0, "motion": -1.0,
                        }[key]

            # 两手距离
            if num >= 2:
                c0 = _hand_center(kpts_list[0])
                c1 = _hand_center(kpts_list[1])
                row["two_hand_distance"] = float(np.linalg.norm(c0 - c1))
            else:
                row["two_hand_distance"] = -1.0

            # 接触状态
            cur_sensors = None
            if sensor_data is not None:
                cur_sensors = {}
                for sn in settings.SENSOR_NAMES:
                    arr = sensor_data[sn].get(fi)
                    if arr is not None:
                        cur_sensors[sn] = arr
            row["contact"] = _compute_contact(
                row["hand_0_motion"], row["hand_1_motion"], cur_sensors)

            rows.append(row)

            if progress_cb and idx % 100 == 0:
                progress_cb(idx, total)

        if cancel_check and cancel_check():
            return {"success": False, "error": "cancelled"}

        # 4. 写入 parquet
        if rows:
            table = pa.Table.from_pylist(rows, schema=_AUTO_LABELS_SCHEMA)
            out_path = auto_labels_parquet_path(session_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pq.write_table(table, out_path, compression="zstd")

        # 5. 时序分割 → jsonl
        from core.helpers import _keypoints_session_dir
        actions_path = os.path.join(_keypoints_session_dir(session_path),
                                    "auto_actions.jsonl")
        os.makedirs(os.path.dirname(actions_path), exist_ok=True)
        with open(actions_path, "w", encoding="utf-8") as f:
            for action in _segment_actions(rows):
                f.write(json.dumps(action, ensure_ascii=False) + "\n")

        elapsed = time.perf_counter() - t_start
        if progress_cb:
            progress_cb(total, total)
        return {"success": True, "frames": total, "elapsed": elapsed}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def _segment_actions(rows: list) -> list:
    """帧级标签 → 动作片段列表。"""
    if not rows:
        return []
    segments = []
    cur_label = rows[0]["hand_0_gesture"]
    cur_start = 0
    for i in range(1, len(rows)):
        label = rows[i]["hand_0_gesture"]
        if label != cur_label:
            segments.append({"start": cur_start, "end": i - 1, "label": cur_label})
            cur_label = label
            cur_start = i
    segments.append({"start": cur_start, "end": len(rows) - 1, "label": cur_label})

    for seg in segments:
        seg["length"] = seg["end"] - seg["start"] + 1

    merged = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg["length"] < MIN_SEGMENT_FRAMES and len(merged) > 0:
            merged[-1]["end"] = seg["end"]
            merged[-1]["length"] = merged[-1]["end"] - merged[-1]["start"] + 1
        else:
            merged.append(dict(seg))
        i += 1

    actions = []
    for seg in merged:
        seg_rows = rows[seg["start"]:seg["end"] + 1]
        known = sum(1 for r in seg_rows
                    if r.get("hand_0_gesture", "unknown") != "unknown")
        conf = known / max(len(seg_rows), 1)
        actions.append({
            "start_frame": seg["start"],
            "end_frame":   seg["end"],
            "label":       seg["label"],
            "dominant_hand": 0,
            "confidence":  round(conf, 3),
        })
    return actions


# ═══════════════════════════════════════════════════════════
#  加载函数
# ═══════════════════════════════════════════════════════════

def load_hand_kpts(session_path: str) -> Optional[dict]:
    """加载 2D 手部关键点，返回 {frame_index: {hand_data, num_hands, track_ids}}。

    读取顺序: keypoints_output/ → session/keypoints/ → session/annotations/
    """
    for path_fn in [hand_kpts_parquet_path, _session_kpts_hand_kpts_path,
                    _legacy_hand_kpts_path]:
        kpts_path = path_fn(session_path)
        if os.path.isfile(kpts_path):
            break
    else:
        return None
    table = pq.read_table(kpts_path)
    rows = table.to_pylist()
    return {r["frame_index"]: {
        "hand_data": np.array(r["hand_data"], dtype=np.float32),
        "num_hands": r["num_hands"],
        "track_ids": r["track_ids"],
    } for r in rows}


def load_hand_kpts_pooled(task_dir: str, episode_index: int) -> Optional[dict]:
    """池化布局加载 2D 手部关键点（v1.1.0）。

    镜像目录 keypoints_output/<task>/episode_{N:06d}/hand_pose/chunk-000.parquet；
    返回结构与 load_hand_kpts 相同：{frame_index: {hand_data, num_hands,
    track_ids}}；无数据返回 None。
    """
    from core.helpers import episode_hand_kpts_parquet_path
    kpts_path = episode_hand_kpts_parquet_path(task_dir, episode_index)
    if not os.path.isfile(kpts_path):
        return None
    table = pq.read_table(kpts_path)
    rows = table.to_pylist()
    return {r["frame_index"]: {
        "hand_data": np.array(r["hand_data"], dtype=np.float32),
        "num_hands": r["num_hands"],
        "track_ids": r["track_ids"],
    } for r in rows}


def load_hand_3d(session_path: str) -> Optional[dict]:
    """加载 3D 手部 landmarks，返回 {frame_index: {...}}。

    读取顺序: keypoints_output/ → session/keypoints/ → session/annotations/
    """
    for path_fn in [hand_3d_parquet_path, _session_kpts_hand_3d_path,
                    _legacy_hand_3d_path]:
        path = path_fn(session_path)
        if os.path.isfile(path):
            break
    else:
        return None
    table = pq.read_table(path)
    rows = table.to_pylist()
    return {r["frame_index"]: r for r in rows}


def load_auto_labels(session_path: str) -> Optional[dict]:
    """加载自动标注，返回 {frame_index: row}。

    读取顺序: keypoints_output/ → session/keypoints/ → session/annotations/
    """
    for path_fn in [auto_labels_parquet_path, _session_kpts_auto_labels_path,
                    _legacy_auto_labels_path]:
        path = path_fn(session_path)
        if os.path.isfile(path):
            break
    else:
        return None
    table = pq.read_table(path)
    rows = table.to_pylist()
    return {r["frame_index"]: r for r in rows}

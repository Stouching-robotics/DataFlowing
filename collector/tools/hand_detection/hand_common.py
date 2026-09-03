"""RTMPose 方案共用的手部关键点定义与绘制。

关键点约定与 MediaPipe 完全一致（RTMPose 的 hand5 模型就是这 21 点同序），
所以这里的定义和上级目录 hand_demo.py 是一套东西。刻意不 import 父目录，
让 rtmpose/ 保持自包含——这一套不需要装 mediapipe 也能跑。
"""

import os

_HAND_COMMON_DIR = os.path.dirname(os.path.abspath(__file__))

import math

import cv2
import numpy as np

# 21 点顺序: 0 腕, 1-4 拇指, 5-8 食指, 9-12 中指, 13-16 无名指, 17-20 小指
FINGERS = {
    "Thumb":  ([1, 2, 3, 4],     (255, 128, 0)),
    "Index":  ([5, 6, 7, 8],     (0, 255, 0)),
    "Middle": ([9, 10, 11, 12],  (0, 255, 255)),
    "Ring":   ([13, 14, 15, 16], (255, 0, 255)),
    "Pinky":  ([17, 18, 19, 20], (0, 128, 255)),
}

PALM_CONNECTIONS = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]


def joint_specs():
    specs = []
    for finger, (ids, color) in FINGERS.items():
        a, b, c, d = ids
        names = ("CMC", "MCP", "IP") if finger == "Thumb" else ("MCP", "PIP", "DIP")
        specs.append((finger, names[0], a, 0, b, color))
        specs.append((finger, names[1], b, a, c, color))
        specs.append((finger, names[2], c, b, d, color))
    return specs


JOINT_SPECS = joint_specs()


def angle_between(p_prev, p_vertex, p_next):
    """顶点处两条骨段的夹角（度）。180° = 伸直。"""
    v1 = np.asarray(p_prev, dtype=np.float64) - np.asarray(p_vertex, dtype=np.float64)
    v2 = np.asarray(p_next, dtype=np.float64) - np.asarray(p_vertex, dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def compute_joint_angles(pts):
    """RTMPose 只输出 2D 关键点，所以角度是在图像平面上算的。

    这一点和 hand_demo.py 不同——那边用 MediaPipe 的 3D world landmarks，
    手绕轴转动时角度更稳。这里手指朝向摄像头（透视缩短）时角度会偏小，
    看数值时要心里有数。
    """
    return {(f, j): angle_between(pts[p], pts[v], pts[n])
            for f, j, v, p, n, _ in JOINT_SPECS}


def count_extended_fingers(angles):
    out = []
    for finger in FINGERS:
        if finger == "Thumb":
            ok = angles[(finger, "MCP")] > 145 and angles[(finger, "IP")] > 150
        else:
            ok = angles[(finger, "PIP")] > 150 and angles[(finger, "DIP")] > 140
        if ok:
            out.append(finger)
    return out


def draw_hand(frame, pts, angles=None, show_angles=True, kpt_scores=None, thr=0.3):
    """pts: (21,2) 像素坐标; kpt_scores: (21,) 每点置信度, 低于 thr 的画空心。"""
    p = [(int(x), int(y)) for x, y in pts[:21]]

    for a, b in PALM_CONNECTIONS:
        cv2.line(frame, p[a], p[b], (200, 200, 200), 2, cv2.LINE_AA)

    for finger, (ids, color) in FINGERS.items():
        chain = ids if finger == "Thumb" else [0] + ids
        for i in range(len(chain) - 1):
            cv2.line(frame, p[chain[i]], p[chain[i + 1]], color, 3, cv2.LINE_AA)
        for idx in ids:
            r = 7 if idx == ids[-1] else 5
            weak = kpt_scores is not None and kpt_scores[idx] < thr
            cv2.circle(frame, p[idx], r, color, 1 if weak else -1, cv2.LINE_AA)
            if not weak:
                cv2.circle(frame, p[idx], r, (30, 30, 30), 1, cv2.LINE_AA)

    cv2.circle(frame, p[0], 9, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, p[0], 9, (40, 40, 40), 2, cv2.LINE_AA)

    if show_angles and angles:
        for finger, joint, vertex, prev_id, _n, color in JOINT_SPECS:
            x, y = p[vertex]
            dx, dy = x - p[prev_id][0], y - p[prev_id][1]
            norm = math.hypot(dx, dy) or 1.0
            tx, ty = int(x + dx / norm * 16) + 4, int(y + dy / norm * 16)
            txt = f"{angles[(finger, joint)]:.0f}"
            cv2.putText(frame, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        color, 1, cv2.LINE_AA)


def draw_panel(frame, x, y, lines, width=340):
    h = 22 * len(lines) + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + h), (90, 90, 90), 1)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (x + 10, y + 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════
# 关键点模型 & 设备检测（infer.py / hand_demo_mmpose.py 共用）
# ═══════════════════════════════════════════════════════════════════

def build_pose(device="cpu"):
    """RTMPose hand5: 21 关键点，与 MediaPipe 同序。首次运行自动下载 ~56MB。"""
    from rtmlib import RTMPose
    url = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
           "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip")
    return RTMPose(url, model_input_size=(256, 256),
                   backend="onnxruntime", device=device)


def auto_device():
    """自动检测最佳推理设备：CUDA > MPS > CPU"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def pose_is_glove(kpts, scores, box, mean_thr=0.45, n_ok_thr=15, ok_thr=0.3,
                  span_thr=0.3):
    """误检抑制：判定一帧关键点是否属于「真手套」。

    RTMPose 在背景/手套状物体上解不出成体系的关键点。实测（2026-08-20，
    D435 无手套画面 vs 000003 视频）：误检框 kpt 均值 0.20~0.31、
    ≥0.3 高置信点 ≤10 个、点团 span ≤0.23×框对角线；真手套 0.64~0.81、
    21/21 高置信、span 0.55~0.79。三条全过才算手套，否则不画框不解算。
    """
    p = np.asarray(kpts, dtype=np.float64)[:21]
    s = np.asarray(scores, dtype=np.float64)[:21]
    if np.isnan(p).any() or np.isnan(s).any():
        return False
    if float(s.mean()) < mean_thr:
        return False
    if int((s >= ok_thr).sum()) < n_ok_thr:
        return False
    diag = np.hypot(box[2] - box[0], box[3] - box[1])
    if diag > 0:
        span = float(np.hypot(*(p.max(0) - p.min(0))))
        if span < span_thr * diag:
            return False
    return True

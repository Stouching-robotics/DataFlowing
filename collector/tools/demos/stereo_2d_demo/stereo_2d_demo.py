#!/usr/bin/env python3
"""stereo_2d_demo.py —— 双目(或单目)视频 2D 手部关键点检测 + 渲染演示。

**自包含单文件**：不依赖仓库里任何其他模块。MediaPipe 检测、One-Euro 平滑、
五指分色绘制、手势识别、视频合成与 H.264 转码全部逻辑都在这一个文件里。
模型文件 hand_landmarker.task 需与本文件放在同一目录（默认路径为
本文件所在目录下的 hand_landmarker.task，可用 --model 指定）。

用法::

    # 双目（左右目并排输出，标注 2D 关键点）
    python stereo_2d_demo.py left.mp4 right.mp4 -o out.mp4

    # 单目
    python stereo_2d_demo.py single.mp4 -o out.mp4

    # 更跟手/更平滑调节
    python stereo_2d_demo.py left.mp4 right.mp4 --freq-min 15 --beta 0.6
    python stereo_2d_demo.py left.mp4 right.mp4 --no-smooth   # 完全关平滑

依赖（pip install mediapipe opencv-python numpy）::

    pip install mediapipe opencv-python numpy

输出: 带关键点叠加的视频（H.264 或 mp4v 兜底）。
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. 手部关键点定义（MediaPipe 21 点）
#    0 腕, 1-4 拇指, 5-8 食指, 9-12 中指, 13-16 无名指, 17-20 小指
# ─────────────────────────────────────────────────────────────

FINGERS = {
    "Thumb":  ([1, 2, 3, 4],     (255, 128, 0)),
    "Index":  ([5, 6, 7, 8],     (0, 255, 0)),
    "Middle": ([9, 10, 11, 12],  (0, 255, 255)),
    "Ring":   ([13, 14, 15, 16], (255, 0, 255)),
    "Pinky":  ([17, 18, 19, 20], (0, 128, 255)),
}
PALM_CONNECTIONS = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]


def _joint_specs():
    specs = []
    for finger, (ids, color) in FINGERS.items():
        a, b, c, d = ids
        names = ("CMC", "MCP", "IP") if finger == "Thumb" else ("MCP", "PIP", "DIP")
        specs.append((finger, names[0], a, 0, b, color))
        specs.append((finger, names[1], b, a, c, color))
        specs.append((finger, names[2], c, b, d, color))
    return specs


JOINT_SPECS = _joint_specs()

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

# ─────────────────────────────────────────────────────────────
# 2. 关节角度 / 手势判定
# ─────────────────────────────────────────────────────────────

def _angle_between(p_prev, p_vertex, p_next):
    """顶点处两条骨段的夹角（度）。180° = 伸直。"""
    v1 = np.asarray(p_prev, dtype=np.float64) - np.asarray(p_vertex, dtype=np.float64)
    v2 = np.asarray(p_next, dtype=np.float64) - np.asarray(p_vertex, dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _compute_joint_angles(pts):
    """21 点 (21,2) → 每关节角度 dict {(finger, joint): 度}。"""
    return {(f, j): _angle_between(pts[p], pts[v], pts[n])
            for f, j, v, p, n, _ in JOINT_SPECS}


def _count_extended_fingers(angles):
    """角度 → 伸直手指名列表（空 = 握拳）。"""
    out = []
    for finger in FINGERS:
        if finger == "Thumb":
            ok = angles.get((finger, "MCP"), 0) > 145 and angles.get((finger, "IP"), 0) > 150
        else:
            ok = angles.get((finger, "PIP"), 0) > 150 and angles.get((finger, "DIP"), 0) > 140
        if ok:
            out.append(finger)
    return out


# ─────────────────────────────────────────────────────────────
# 3. One-Euro 自适应平滑（关键点跟手性）
#    freq_min 越大越跟手（越抖）；beta 越大快速运动响应越快
# ─────────────────────────────────────────────────────────────

class _OneEuroFilter:
    """单值 One-Euro 自适应低通滤波器。"""

    def __init__(self, freq_min=5.0, beta=0.05, dcutoff=1.0):
        self.freq_min = freq_min
        self.beta = beta
        self.dcutoff = dcutoff
        self._prev_x = None
        self._prev_dx = 0.0
        self._prev_ts = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff) if cutoff > 1e-9 else 0.0
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


class _OneEuroFilter2D:
    """对 (x, y) 分量独立滤波。"""

    def __init__(self, freq_min=5.0, beta=0.05, dcutoff=1.0):
        self._fx = _OneEuroFilter(freq_min, beta, dcutoff)
        self._fy = _OneEuroFilter(freq_min, beta, dcutoff)

    def __call__(self, x, y, ts_ms):
        return self._fx(x, ts_ms), self._fy(y, ts_ms)


# ─────────────────────────────────────────────────────────────
# 4. MediaPipe 手部检测（Tasks API，VIDEO 模式）
# ─────────────────────────────────────────────────────────────

class HandDetector:
    """每帧返回手部关键点列表：[[(x,y),...21点], ...]。"""

    def __init__(self, model_path=MODEL_PATH, num_hands=2, mirror=False,
                 smooth=True, freq_min=15.0, beta=0.6, dcutoff=1.0):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"模型不存在: {model_path}\n"
                "请下载 hand_landmarker.task 并与本脚本放同一目录:\n"
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task")

        self.mirror = mirror
        self.smooth = smooth
        self._mp = mp
        self._freq_min, self._beta, self._dcutoff = freq_min, beta, dcutoff
        self._filters: dict = {}
        self._t0 = time.perf_counter()

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr):
        """BGR 帧 → [[(x, y), ×21], ...] 像素关键点列表（每只手一个）。"""
        h, w = frame_bgr.shape[:2]
        frame = cv2.flip(frame_bgr, 1) if self.mirror else frame_bgr
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts_ms = int((time.perf_counter() - self._t0) * 1000)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        hands = []
        for lms in result.hand_landmarks:
            if self.smooth:
                pts = []
                for j, lm in enumerate(lms):
                    key = (len(hands), j)
                    if key not in self._filters:
                        self._filters[key] = _OneEuroFilter2D(
                            self._freq_min, self._beta, self._dcutoff)
                    fx, fy = self._filters[key](lm.x, lm.y, ts_ms)
                    pts.append((int(fx * w), int(fy * h)))
                hands.append(pts)
            else:
                hands.append([(int(lm.x * w), int(lm.y * h)) for lm in lms])
        return hands

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# 5. 叠加绘制（demo 风格：五指分色 + 边框标签 + 手势文本）
# ─────────────────────────────────────────────────────────────

def _draw_hand(frame, pts):
    """按五指分色画骨架：掌心灰连接、腕白圆、指尖 7px/关节 5px + 深色描边。"""
    p = [(int(x), int(y)) for x, y in pts[:21]]
    for a, b in PALM_CONNECTIONS:
        cv2.line(frame, p[a], p[b], (200, 200, 200), 2, cv2.LINE_AA)
    for finger, (ids, color) in FINGERS.items():
        chain = ids if finger == "Thumb" else [0] + ids
        for i in range(len(chain) - 1):
            cv2.line(frame, p[chain[i]], p[chain[i + 1]], color, 3, cv2.LINE_AA)
        for idx in ids:
            r = 7 if idx == ids[-1] else 5
            cv2.circle(frame, p[idx], r, color, -1, cv2.LINE_AA)
            cv2.circle(frame, p[idx], r, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.circle(frame, p[0], 9, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, p[0], 9, (40, 40, 40), 2, cv2.LINE_AA)


def _draw_kpts_overlay(frame, hands, cam_label="", frame_idx=0, total=0):
    """边框 + 底衬标签 + 分色骨架 + 手势文本；hands: [[(x,y)...21], ...]。"""
    for i, pts in enumerate(hands[:2]):
        pts = np.asarray(pts, np.float32).reshape(21, 2)
        xs, ys = pts[:, 0], pts[:, 1]
        pad = 12
        x1, y1 = int(xs.min()) - pad, int(ys.min()) - pad
        x2, y2 = int(xs.max()) + pad, int(ys.max()) + pad
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 220), 2, cv2.LINE_AA)
        label = f"Hand #{i}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 200, 200), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2, cv2.LINE_AA)
        _draw_hand(frame, pts)
        try:
            angles = _compute_joint_angles(pts)
            extended = _count_extended_fingers(angles)
            text = "open: " + ",".join(extended) if extended else "fist"
            cv2.putText(frame, text, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 120), 1, cv2.LINE_AA)
        except Exception:
            pass

    info = cam_label
    if total:
        info += f"  frame {frame_idx}/{total}  hands={len(hands)}"
    cv2.putText(frame, info, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    cv2.putText(frame, "COLORED: 2D MediaPipe hand keypoints",
                (10, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────────────────────
# 6. 视频输出（H.264 转码，无 ffmpeg 时保留 mp4v 兜底）
# ─────────────────────────────────────────────────────────────

def _create_writer(out_path, fps, width, height):
    tmp = os.path.splitext(out_path)[0] + "_tmp.avi"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写器: {tmp}")
    return writer, tmp


def _find_working_ffmpeg():
    """探测可用的 ffmpeg（存在的候选可能不可运行，如损坏的 conda 版本）。"""
    candidates = [shutil.which("ffmpeg"), "/usr/bin/ffmpeg",
                  "/usr/local/bin/ffmpeg",
                  os.path.expanduser("~/miniconda3/envs/lerobot/bin/ffmpeg"),
                  os.path.expanduser("~/anaconda3/envs/lerobot/bin/ffmpeg")]
    for ff in dict.fromkeys(c for c in candidates if c):
        try:
            r = subprocess.run([ff, "-version"], capture_output=True, timeout=10)
            if r.returncode == 0:
                return ff
        except (OSError, subprocess.TimeoutExpired):
            continue
    return None


def _finalize(writer, tmp_path, out_path):
    writer.release()
    ffmpeg = _find_working_ffmpeg()
    if ffmpeg:
        try:
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", tmp_path,
                            "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                            out_path], check=True, capture_output=True)
            os.remove(tmp_path)
            return out_path
        except (OSError, subprocess.CalledProcessError):
            pass   # 转换失败则退回 mp4v 原样交付
    os.replace(tmp_path, out_path)   # 无 ffmpeg: mp4v 直接改名交付
    return out_path


# ─────────────────────────────────────────────────────────────
# 7. 主流程
# ─────────────────────────────────────────────────────────────

def process_video(streams, output, detector, cam_labels):
    """streams: [cv2.VideoCapture, ...]；输出并排叠加视频。"""
    n_total = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in streams)
    fps = streams[0].get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(streams[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(streams[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = w * len(streams)
    writer, tmp = _create_writer(output, fps, out_w, h)

    t0 = time.time()
    for i in range(n_total):
        frames, hands = [], []
        ok_all = True
        for c in streams:
            ok, fr = c.read()
            if not ok:
                ok_all = False
                break
            frames.append(fr)
            hands.append(detector.process(fr))
        if not ok_all:
            break
        canvases = []
        for fr, hs, lab in zip(frames, hands, cam_labels):
            canvases.append(_draw_kpts_overlay(fr, hs, lab, i, n_total))
        writer.write(cv2.hconcat(canvases) if len(canvases) > 1 else canvases[0])
        if (i + 1) % 25 == 0 or i + 1 == n_total:
            el = time.time() - t0
            print(f"  {i + 1}/{n_total} 帧  ({(i + 1) / el:.1f} fps 检测+渲染)")

    for c in streams:
        c.release()
    return _finalize(writer, tmp, output)


def main():
    ap = argparse.ArgumentParser(
        description="双目/单目视频 2D 手部关键点检测 + 渲染 demo（自包含单文件）")
    ap.add_argument("videos", nargs="+", help="视频路径（1 个=单目, 2 个=左右目并排）")
    ap.add_argument("-o", "--output", default="stereo_2d_output.mp4",
                    help="输出视频路径（默认 stereo_2d_output.mp4）")
    ap.add_argument("--model", default=MODEL_PATH, help="hand_landmarker.task 路径")
    ap.add_argument("--freq-min", type=float, default=15.0,
                    help="One-Euro 位置滤波截止频率 Hz（默认 15；越大越跟手越抖）")
    ap.add_argument("--beta", type=float, default=0.6,
                    help="One-Euro 速度自适应系数（默认 0.6；快速运动滞后就调大）")
    ap.add_argument("--no-smooth", action="store_true",
                    help="关闭 One-Euro 平滑（最跟手，原始抖动）")
    args = ap.parse_args()

    if len(args.videos) > 2:
        ap.error("最多支持 2 个视频（单目或左右目）")

    streams = []
    try:
        for v in args.videos:
            if not os.path.isfile(v):
                raise FileNotFoundError(f"视频不存在: {v}")
            c = cv2.VideoCapture(v)
            if not c.isOpened():
                raise RuntimeError(f"无法打开视频: {v}")
            streams.append(c)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"加载视频: {args.videos}  ({int(streams[0].get(cv2.CAP_PROP_FRAME_COUNT))} 帧, "
          f"{streams[0].get(cv2.CAP_PROP_FPS):.1f}fps)")
    print(f"模型: {args.model}")
    if args.no_smooth:
        print("平滑: 关闭")
    else:
        print(f"平滑: One-Euro freq_min={args.freq_min} beta={args.beta}")

    detector = HandDetector(model_path=args.model, num_hands=2,
                            smooth=not args.no_smooth,
                            freq_min=args.freq_min, beta=args.beta)
    labels = ["Left", "Right"] if len(args.videos) == 2 else ["Camera"]

    try:
        out = process_video(streams, args.output, detector, labels)
        print(f"✓ 完成: {out}")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    finally:
        detector.close()


if __name__ == "__main__":
    main()

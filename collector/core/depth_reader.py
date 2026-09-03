"""深度数据读取（v1.1.2 12-bit 灰度 MP4 + 旧 MKV/png16 回退）。

新格式 depth/<slot>/<slot>.mp4：单视频轨 hevc (Rext) gray12le ——
12-bit 对数深度码（core.depth_codec 量化域），ffmpeg CLI 解
gray12le 后反量化回 uint16 毫米。
旧格式回退：
  depth/<slot>/<slot>.mkv   v1.0.14 双流（流0 热力图 h264 + 流1
                            FFV1 gray16le 无损 uint16 毫米）；
                            迁移合成件 = 单流 FFV1（流0）。
  depth/<slot>/*.png        v1.0.13 及以前 PNG16 序列。

cv2 的 CAP_PROP_STREAM_INDEX 对 ffmpeg 后端不可靠（实测不生效），
数据轨统一走 ffmpeg CLI 解 rawvideo。
"""

from __future__ import annotations
import glob
import os
import re
import subprocess
from typing import Iterator, Optional

import cv2
import numpy as np

from core.depth_codec import codes_to_heatmap_bgr, dequantize_depth
from core.encoder_probe import list_working_ffmpegs
from core.helpers import (egodata_depth_path, egodata_depth_video_path,
                          episode_video_files)


def _get_ffmpeg() -> Optional[str]:
    bins = list_working_ffmpegs()
    return bins[0] if bins else None


def has_depth_mkv(episode_dir: str, depth_name: str) -> bool:
    """该槽是否有视频深度（12-bit 灰 MP4 / 双流 MKV 任一）。"""
    return os.path.isfile(egodata_depth_video_path(episode_dir, depth_name))


def iter_depth_frames(episode_dir: str, depth_name: str) -> Iterator[np.ndarray]:
    """顺序产出 uint16 毫米深度帧（(H, W) numpy 数组）。

    新格式：12-bit 灰度 MP4（gray12le 解码 + 反量化）；
    旧格式：双流/单流 FFV1 MKV（uint16 毫米直出）；
    更旧：depth/<slot>/000001.png 序列（v1.0.13 及以前）。
    """
    path = egodata_depth_video_path(episode_dir, depth_name)
    if os.path.isfile(path):
        if path.lower().endswith(".mp4"):
            yield from _iter_mp4_depth_frames_path(path)
        else:
            yield from _iter_mkv_depth_frames_path(path)
        return
    yield from _iter_png16_frames(episode_dir, depth_name)


def _ffv1_stream_index(ffmpeg: str, mkv_path: str) -> Optional[int]:
    """探测 MKV 里 FFV1 无损深度轨的流号（stderr 解析，无则 None）。

    v1.0.14 双流 = 流0 热力图 h264 + 流1 FFV1；迁移合成件 = 单流 FFV1
    （流0）。解码前探测一次，两种布局都命中。
    """
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-i", mkv_path],
                           capture_output=True, timeout=30)
    except Exception:
        return None
    text = (r.stderr or b"").decode("utf-8", "ignore")
    for line in text.splitlines():
        m = re.match(r"\s*Stream #0:(\d+)(?:\([^)]*\))?:\s*Video:\s*ffv1", line)
        if m:
            return int(m.group(1))
    return None


def _iter_mkv_depth_frames_path(mkv_path: str) -> Iterator[np.ndarray]:
    """ffmpeg 流式解 MKV 的 FFV1 无损轨（gray16le → uint16 毫米帧）。"""
    ffmpeg = _get_ffmpeg()
    if not ffmpeg:
        return
    cap = cv2.VideoCapture(mkv_path)
    if not cap.isOpened():
        cap.release()
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        return
    # FFV1 轨流号探测：双流件（v1.0.14）= 1，迁移单流件 = 0；探测失败
    # 回退 0:1（v1.0.14 双流默认）。
    idx = _ffv1_stream_index(ffmpeg, mkv_path)
    stream = f"0:{idx}" if idx is not None else "0:1"
    proc = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-i", mkv_path,
         "-map", stream, "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = w * h * 2
    buf = b""
    while proc.poll() is None:
        chunk = proc.stdout.read(65536)
        if not chunk:
            break
        buf += chunk
        while len(buf) >= frame_bytes:
            yield np.frombuffer(buf[:frame_bytes], "<u2").reshape(h, w).copy()
            buf = buf[frame_bytes:]
    proc.stdout.close()
    proc.wait()


def _iter_mp4_depth_frames_path(mp4_path: str) -> Iterator[np.ndarray]:
    """ffmpeg 流式解 12-bit 灰度 MP4（gray12le → 反量化 → uint16 毫米）。

    探测容器内流是否为 gray12le（v1.0.14 合封失败回退产生的 8-bit
    热力图 mp4 无数据轨，产出空帧序列）；其余 HEVC 位深不回退（
    写端只有 12-bit 一条路）。
    """
    if not _mp4_is_gray12(mp4_path):
        return
    ffmpeg = _get_ffmpeg()
    if not ffmpeg:
        return
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        cap.release()
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        return
    proc = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-i", mp4_path,
         "-f", "rawvideo", "-pix_fmt", "gray12le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = w * h * 2
    buf = b""
    while proc.poll() is None:
        chunk = proc.stdout.read(65536)
        if not chunk:
            break
        buf += chunk
        while len(buf) >= frame_bytes:
            codes = np.frombuffer(buf[:frame_bytes], "<u2").reshape(h, w)
            buf = buf[frame_bytes:]
            yield dequantize_depth(codes)
    proc.stdout.close()
    proc.wait()


class Gray12DepthVideo:
    """12-bit 灰度 MP4 随机访问读取（回放用）。

    cv2 直读 gray12le 的 8-bit 转换不可靠（实测与规范码值映射
    mean 差 169/255），像素一律走 ffmpeg CLI 解 gray12le →
    codes_to_heatmap_bgr（规范 JET，与实时显示同构）。
    小步顺序读续用当前流；随机大跳用 -ss 输入快进（从最近关键帧
    解到目标，keyint~250 最坏数百帧，亚秒级）。
    """

    def __init__(self, path: str, width: int, height: int,
                 fps: float, total: int):
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps > 0 else 30.0
        self.total = max(1, int(total))
        self._proc: Optional[subprocess.Popen] = None
        self._buf = b""
        self._next_idx = 0
        self._frame_bytes = self.width * self.height * 2

    @classmethod
    def from_path(cls, mp4_path: str) -> Optional["Gray12DepthVideo"]:
        """gray12le 且可探测尺寸/帧率的返回实例，否则 None。"""
        if not _mp4_is_gray12(mp4_path):
            return None
        w = h = 0
        fps = 0.0
        total = 0
        try:
            cap = cv2.VideoCapture(mp4_path)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        except Exception:
            pass
        if w > 0 and h > 0:
            return cls(mp4_path, w, h, fps, total)
        # cv2 探不开（灰度 12-bit 后端缺省）→ ffmpeg -i stderr 兜底
        ffmpeg = _get_ffmpeg()
        if not ffmpeg:
            return None
        try:
            r = subprocess.run([ffmpeg, "-hide_banner", "-i", mp4_path],
                               capture_output=True, timeout=30)
        except Exception:
            return None
        text = (r.stderr or b"").decode("utf-8", "ignore")
        m = re.search(r"gray12le[^\n]*?(\d{3,5})x(\d{3,5})[^\n]*?([\d.]+)\s+fps",
                      text)
        if not m:
            return None
        w, h = int(m.group(1)), int(m.group(2))
        fps = float(m.group(3))
        dm = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", text)
        if dm and fps > 0:
            total = int((int(dm.group(1)) * 3600 + int(dm.group(2)) * 60
                         + float(dm.group(3))) * fps)
        return cls(mp4_path, w, h, fps, total)

    def read(self, idx: int) -> Optional[np.ndarray]:
        """读取帧 idx（0-based）→ BGR 热力图；越界/EOF 返回 None。"""
        if idx >= self.total:
            return None
        if self._proc is None or self._proc.poll() is not None:
            self._open(idx)
        while self._next_idx < idx:
            if self._pull() is None:
                return None
        return self._pull()

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=3)
            except Exception:
                pass
            self._proc = None
        self._buf = b""

    def _open(self, idx: int) -> None:
        self.close()
        ffmpeg = _get_ffmpeg()
        if not ffmpeg:
            return
        cmd = [ffmpeg, "-hide_banner", "-nostdin"]
        if idx > 0:
            # -ss 输入快进：从 ss 前最近关键帧解码并丢弃；-0.5/fps 余量
            # 防 PTS 舍入把 idx-1 带出来（均匀 PTS 下首输出即帧 idx）
            ss = max(0.0, (idx - 0.5) / self.fps)
            cmd += ["-ss", f"{ss:.6f}"]
        cmd += ["-i", self.path, "-f", "rawvideo",
                "-pix_fmt", "gray12le", "-"]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception:
            self._proc = None
            return
        self._next_idx = idx

    def _pull(self) -> Optional[np.ndarray]:
        if self._proc is None:
            return None
        try:
            while len(self._buf) < self._frame_bytes:
                chunk = self._proc.stdout.read(65536)
                if not chunk:
                    return None  # EOF（流结束/解码失败）
                self._buf += chunk
            raw = self._buf[:self._frame_bytes]
            self._buf = self._buf[self._frame_bytes:]
            self._next_idx += 1
        except Exception:
            return None
        codes = np.frombuffer(raw, "<u2").reshape(self.height, self.width)
        return codes_to_heatmap_bgr(codes)


def _mp4_is_gray12(mp4_path: str) -> bool:
    """容器内视频流是否为 gray12le（stderr 解析；探测失败保守 False）。"""
    ffmpeg = _get_ffmpeg()
    if not ffmpeg:
        return False
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-i", mp4_path],
                           capture_output=True, timeout=30)
    except Exception:
        return False
    text = (r.stderr or b"").decode("utf-8", "ignore")
    return bool(re.search(r"Video:\s*[^\n]*gray12le", text))


def _iter_png16_frames(episode_dir: str, depth_name: str) -> Iterator[np.ndarray]:
    pattern = os.path.join(episode_dir, "depth", depth_name, "*.png")
    for path in sorted(glob.glob(pattern)):
        yield cv2.imread(path, cv2.IMREAD_UNCHANGED)


def read_depth_frame(episode_dir: str, depth_name: str,
                     frame_index: int) -> Optional[np.ndarray]:
    """读取单帧（1-based 序号，与 RGB 帧号对齐）；越界/缺失返回 None。

    新格式下为 O(N) 顺序解到目标帧（FFV1 无随机访问捷径）；
    批量遍历请直接用 iter_depth_frames。
    """
    for i, frame in enumerate(iter_depth_frames(episode_dir, depth_name)):
        if i == frame_index - 1:
            return frame
    return None


# ═══════════════════════════════════════════════════════
#  池化布局（v1.1.0）：(task_dir, depth_key, episode_index) 键控
# ═══════════════════════════════════════════════════════

def pooled_depth_mkv_path(task_dir: str, depth_key: str,
                          episode_index: int) -> Optional[str]:
    """episode N 的深度文件路径（videos/chunk-{c}/<depth_key>/episode-{f}.{ext}）。

    新录制 = 12-bit 灰度 mp4（v1.1.2 起）；历史 = mkv（v1.0.14 双流
    FFV1 / 迁移合成单流 FFV1）。无视频深度返回 None。
    """
    path = episode_video_files(task_dir, episode_index).get(depth_key)
    if path and os.path.splitext(path)[1].lower() in (".mp4", ".mkv"):
        return path
    return None


def iter_pooled_depth_frames(task_dir: str, depth_key: str,
                             episode_index: int) -> Iterator[np.ndarray]:
    """顺序产出 episode N 该深度槽的 uint16 毫米帧；无文件产出空。"""
    path = pooled_depth_mkv_path(task_dir, depth_key, episode_index)
    if not path:
        return
    if path.lower().endswith(".mp4"):
        yield from _iter_mp4_depth_frames_path(path)
    else:
        yield from _iter_mkv_depth_frames_path(path)


def read_pooled_depth_frame(task_dir: str, depth_key: str, episode_index: int,
                            frame_index: int) -> Optional[np.ndarray]:
    """池化单帧读取（1-based 序号，与 RGB 帧号对齐）；越界/缺失返回 None。

    O(N) 顺序解到目标帧；批量遍历请直接用 iter_pooled_depth_frames。
    """
    for i, frame in enumerate(iter_pooled_depth_frames(
            task_dir, depth_key, episode_index)):
        if i == frame_index - 1:
            return frame
    return None

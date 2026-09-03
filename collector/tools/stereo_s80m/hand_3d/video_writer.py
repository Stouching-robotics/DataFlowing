#!/usr/bin/env python3
"""
管道视频写器 —— 渲染段提速。

旧路径（render_stereo.create_video_writer/finalize_video）是两段式：
cv2 mp4v 写临时 avi（CPU 编码 ~15ms/帧）+ 事后 libx264 转码（再花一遍全片时间）。
本模块用 ffmpeg 管道单段直出：rawvideo bgr24 → stdin → H.264 → mp4，
编码移出主线程（ffmpeg 子进程），主循环只剩帧拷贝。

编码器逐级回退：nvenc（lerobot ffmpeg 带 h264_nvenc，RTX 5090 硬编）
→ libx264（管道）→ 旧两段式 mp4v（render_stereo 原路径）。

BrokenPipe 处理：ffmpeg 中途死掉时 stdin.write 抛 BrokenPipeError →
latch 失败状态放弃该视频（parquet 数据不受影响，视频只是演示产物）。

写器接口（与旧 create_video_writer/finalize_video 平替）：
    sink = create_video_sink(path, fps, w, h, encoder="auto")
    sink.write(frame_bgr)          # 失败静默（已 latch），返回 bool
    final_path = sink.close()      # None = 该视频放弃
"""

from __future__ import annotations

import os
import shutil
import subprocess

import cv2

_LEROBOT_FFMPEG = os.environ.get(
    "FFMPEG_BIN", os.path.expanduser("~/miniconda3/envs/lerobot/bin/ffmpeg"))

_ffmpeg_cache = None          # find_ffmpeg 缓存（None=未探测）
_nvenc_cache = {}             # ffmpeg 路径 → 是否带 h264_nvenc


def find_ffmpeg() -> str | None:
    """找一个能跑起来的 ffmpeg。lerobot env 优先（带 nvenc 且避开
    conda base 的 openvino/tbb 符号错误），PATH / /usr/bin 兜底。
    用 `-version` 退出码探测，结果缓存。"""
    global _ffmpeg_cache
    if _ffmpeg_cache is not None:
        return _ffmpeg_cache
    candidates = [_LEROBOT_FFMPEG, shutil.which("ffmpeg"), "/usr/bin/ffmpeg"]
    for ff in dict.fromkeys(c for c in candidates if c):
        try:
            subprocess.run([ff, "-version"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True, timeout=10)
            _ffmpeg_cache = ff
            return ff
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    _ffmpeg_cache = ""
    return None


def has_nvenc(ffmpeg: str | None = None) -> bool:
    """该 ffmpeg 是否带 h264_nvenc 编码器（结果缓存）。"""
    ff = ffmpeg or find_ffmpeg()
    if not ff:
        return False
    if ff not in _nvenc_cache:
        try:
            out = subprocess.run([ff, "-hide_banner", "-encoders"],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 check=True, timeout=30).stdout.decode()
            _nvenc_cache[ff] = "h264_nvenc" in out
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            _nvenc_cache[ff] = False
    return _nvenc_cache[ff]


class PipeVideoWriter:
    """ffmpeg 管道写器：rawvideo bgr24 → stdin → 单段 H.264 mp4。"""

    def __init__(self, out_path: str, fps: float, width: int, height: int,
                 codec: str, codec_args: list):
        self.out_path = out_path
        self.failed = False          # BrokenPipe latch：一旦失败放弃该视频
        cmd = [find_ffmpeg(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{int(width)}x{int(height)}", "-r", str(fps), "-i", "-",
               "-an", "-c:v", codec, *codec_args,
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
        # stderr 走 DEVNULL：-loglevel error 时正常静默；若想调试
        # BrokenPipe 根因，把 DEVNULL 换成日志文件即可。
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)

    def write(self, frame_bgr) -> bool:
        """写一帧 BGR。ffmpeg 已死 → latch 失败并返回 False（调用方忽略即可）。"""
        if self.failed:
            return False
        try:
            self._proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError:
            self.failed = True
            return False
        return True

    def close(self) -> str | None:
        """收尾：关 stdin、等 ffmpeg 退出。成功返回 out_path，失败 None。"""
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                self.failed = True
        try:
            rc = self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self.failed = True
        if self.failed or rc != 0:
            if os.path.isfile(self.out_path):
                try:
                    os.remove(self.out_path)     # 半成品 mp4 不留
                except OSError:
                    pass
            return None
        return self.out_path


class Mp4vWriter:
    """旧两段式路径包装（mp4v avi + 事后 libx264 转码），统一 .write/.close 接口。"""

    def __init__(self, out_path: str, fps: float, width: int, height: int):
        # 惰性 import：render_stereo 是共享模块，只读复用其函数
        from stereo_s80m.render_stereo import create_video_writer, finalize_video
        self.out_path = out_path
        self._finalize = finalize_video
        self._writer, self._tmp = create_video_writer(
            out_path, fps, int(width), int(height))

    def write(self, frame_bgr) -> bool:
        self._writer.write(frame_bgr)
        return True

    def close(self) -> str | None:
        return self._finalize(self._writer, self._tmp, self.out_path)


def create_video_sink(out_path: str, fps: float, width: int, height: int,
                      encoder: str = "auto"):
    """按 encoder 逐级回退创建写器。

    auto: nvenc → libx264（管道）→ 旧两段式 mp4v
    nvenc/libx264: 强制管道（无可用 ffmpeg 时回退 mp4v 并警告）
    mp4v: 旧两段式（零回归基线用）
    返回对象有 .write(frame_bgr) 与 .close()→最终路径(None=失败)。
    """
    w, h = int(width), int(height)
    if encoder in ("auto", "nvenc", "libx264"):
        ff = find_ffmpeg()
        if ff:
            want_nvenc = encoder in ("auto", "nvenc") and has_nvenc(ff)
            if want_nvenc:
                return PipeVideoWriter(out_path, fps, w, h, "h264_nvenc",
                                       ["-rc", "vbr", "-cq", "23", "-b:v", "0"])
            if encoder in ("auto", "libx264"):
                return PipeVideoWriter(out_path, fps, w, h, "libx264",
                                       ["-crf", "23", "-preset", "veryfast"])
            print(f"  [警告] 无 h264_nvenc，{encoder} 回退旧两段式 mp4v")
        elif encoder != "auto":
            print(f"  [警告] 无可用 ffmpeg，{encoder} 回退旧两段式 mp4v")
    return Mp4vWriter(out_path, fps, w, h)

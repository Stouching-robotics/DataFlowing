"""
编码器探针 —— 录制编码器自动选择（v1.0.9：录制直出 HEVC）。

纯函数模块（无 Qt 依赖），供 EgoDataWriter 在录制启动前调用（发生在
pipeline 的 _start_async 后台线程内，写盘线程在其返回后才启动，探针
不卡 UI、不影响写入节奏）：
  - 候选 ffmpeg 二进制枚举（imageio 静态 → PATH，-version 自检）
  - 编码器能力探测（-encoders）与速度探针（null muxer 微型编码）
  - 编码器选择（auto: hevc_nvenc → libx265 速度达标 → libx264 回退）
    与参数映射

选择结果进程内缓存（同一进程内机器不变不重探；负结果也缓存）。探针
同步执行、最坏 ~10-20s；所有探针异常按「该编码器不可用」降级处理。
"""
from __future__ import annotations

import dataclasses
import shutil
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional

from config import settings

# ── 进程内缓存 ────────────────────────────────────────
_FFMPEG_CACHE: Optional[List[str]] = None
_ENCODER_CACHE: Dict[tuple, bool] = {}
_SELECT_CACHE: Dict[tuple, Optional["EncoderChoice"]] = {}


@dataclasses.dataclass
class EncoderChoice:
    """一次编码器选择的结果。"""
    encoder: str        # "hevc_nvenc" | "libx265" | "libx264"
    codec: str          # "HEVC" | "H.264"
    crf: int            # 本选择实际用的档位（x264 用回退档）
    args: list          # ffmpeg 编码参数（不含 -c:v）
    ffmpeg: str         # 采用的 ffmpeg 二进制
    label: str          # UI 显示名，如 "HEVC (hevc_nvenc)"
    probe: dict         # 探针明细（写 metadata.json）
    selected_by: str    # settings.RECORD_VIDEO_ENCODER 原值


# ═══════════════════════════════════════════════════════
#  ffmpeg 二进制枚举 / 编码器能力
# ═══════════════════════════════════════════════════════

def list_working_ffmpegs() -> List[str]:
    """候选 ffmpeg 二进制（imageio 静态 → PATH），经 -version 自检。

    全部自检失败时，仍把 imageio 静态二进制作为最后兜底返回（与旧
    _get_ffmpeg 口径一致：它已在录制段被证明可用）。
    """
    global _FFMPEG_CACHE
    if _FFMPEG_CACHE is not None:
        return list(_FFMPEG_CACHE)
    candidates: List[str] = []
    imageio_exe = None
    try:
        import imageio_ffmpeg
        imageio_exe = imageio_ffmpeg.get_ffmpeg_exe()
        candidates.append(imageio_exe)
    except Exception:
        pass
    for name in ("ffmpeg", "ffmpeg.exe"):
        p = shutil.which(name)
        if p and p not in candidates:
            candidates.append(p)
    working = []
    for exe in candidates:
        try:
            r = subprocess.run([exe, "-version"], capture_output=True,
                               timeout=10)
            if r.returncode == 0:
                working.append(exe)
        except Exception:
            continue
    if not working and imageio_exe:
        working.append(imageio_exe)  # 兜底：录段历史证明可用
    _FFMPEG_CACHE = working
    return list(working)


def ffmpeg_has_encoder(ffmpeg: str, encoder: str) -> bool:
    """-encoders 输出里是否含该编码器（结果缓存）。"""
    key = (ffmpeg, encoder)
    if key in _ENCODER_CACHE:
        return _ENCODER_CACHE[key]
    found = False
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                           capture_output=True, timeout=30)
        text = ((r.stdout or b"") + (r.stderr or b"")).decode("utf-8", "ignore")
        found = encoder in text.split()
    except Exception:
        found = False
    _ENCODER_CACHE[key] = found
    return found


# ═══════════════════════════════════════════════════════
#  参数映射 / 微型探针
# ═══════════════════════════════════════════════════════

def encoder_args(encoder: str, crf: Optional[int] = None) -> List[str]:
    """编码器参数映射（HEVC 一律带 hvc1 容器标记）。"""
    if crf is None:
        crf = settings.RECORD_VIDEO_CRF
    table = {
        "hevc_nvenc": ["-rc", "vbr", "-cq", str(crf), "-preset", "p4",
                       "-tag:v", "hvc1"],
        "libx265": ["-crf", str(crf), "-preset", "veryfast", "-tag:v", "hvc1"],
        "libx264": ["-preset", "ultrafast",
                    "-crf", str(settings.RECORD_VIDEO_X264_CRF)],
    }
    return list(table[encoder])


def _kill(p: subprocess.Popen) -> None:
    try:
        p.kill()
    except Exception:
        pass
    try:
        p.wait(timeout=3)
    except Exception:
        pass


def _feed_raw(proc: subprocess.Popen, frame_bytes: bytes, frames: int) -> None:
    try:
        for _ in range(frames):
            proc.stdin.write(frame_bytes)
        proc.stdin.flush()
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass


def probe_nvenc(ffmpeg: str, width: int, height: int, fps: float,
                crf: Optional[int] = None, timeout: int = 10) -> bool:
    """NVENC 微型探针：~1s 编码（null muxer）确认 GPU 可用。"""
    frames = max(3, int(fps))
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
           "-frames:v", str(frames),
           "-c:v", "hevc_nvenc", *encoder_args("hevc_nvenc", crf),
           "-f", "null", "-"]
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        _feed_raw(p, bytes(width * height * 3), frames)
        return p.wait(timeout=timeout) == 0
    except Exception:
        return False


def probe_x265(ffmpeg: str, width: int, height: int, fps: float,
               n_streams: int, frames: Optional[int] = None,
               timeout: Optional[int] = None) -> dict:
    """x265 速度探针：并行 n 流各编 frames 帧 → null，返回每流实测 fps。

    n_streams = 本会话视频流数（封顶 ENCODER_PROBE_MAX_STREAMS）。全部
    流并行跑满同一墙钟，每流 fps = frames / 总耗时 —— 与写盘线程每 tick
    串行写全部槽的背压口径近似一致。合成帧为全零帧（编码最快），1.5×
    门槛余量即为此乐观偏差所设。
    """
    if frames is None:
        frames = settings.ENCODER_PROBE_FRAME_COUNT
    if timeout is None:
        timeout = settings.ENCODER_PROBE_TIMEOUT_S
    n = max(1, min(int(n_streams), settings.ENCODER_PROBE_MAX_STREAMS))
    frame_bytes = bytes(width * height * 3)
    cmd_base = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                "-frames:v", str(frames),
                "-c:v", "libx265", *encoder_args("libx265"),
                "-f", "null", "-"]
    procs: List[subprocess.Popen] = []
    start = time.monotonic()
    ok = True
    try:
        for _ in range(n):
            procs.append(subprocess.Popen(
                cmd_base, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL))
        feeders = [threading.Thread(target=_feed_raw, args=(p, frame_bytes, frames),
                                    daemon=True) for p in procs]
        for t in feeders:
            t.start()
        for p in procs:
            p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        ok = False
    for p in procs:
        if p.poll() is None:
            _kill(p)
            ok = False
        elif p.returncode != 0:
            ok = False
    elapsed = time.monotonic() - start
    if not ok or elapsed <= 0:
        return {"per_stream_fps": [], "min_fps": 0.0, "passed": False,
                "streams": n, "frames": frames, "elapsed_s": round(elapsed, 2)}
    per = [frames / elapsed] * n
    min_fps = min(per)
    threshold = fps * settings.ENCODER_X265_MIN_FPS_RATIO
    return {"per_stream_fps": [round(x, 1) for x in per],
            "min_fps": round(min_fps, 1), "passed": min_fps >= threshold,
            "threshold_fps": round(threshold, 1),
            "streams": n, "frames": frames, "elapsed_s": round(elapsed, 2)}


def probe_depth_12bit(ffmpeg: str, timeout: int = 8) -> bool:
    """深度 12-bit 灰度编码能力探针：gray12le → libx265 → null。

    x265 版本/构建差异下 gray12le（4:0:0，hevc Rext）未必可用；
    单帧微型编码（32×32 → null）确认能力，结果进程内缓存。
    失败时调用方回落 FFV1 gray16le MKV（无损 16-bit 旧格式）。
    """
    key = (ffmpeg, "depth12")
    if key in _ENCODER_CACHE:
        return _ENCODER_CACHE[key]
    ok = False
    try:
        p = subprocess.Popen(
            [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "gray12le",
             "-s", "32x32", "-r", "30", "-i", "-",
             "-frames:v", "1", "-c:v", "libx265",
             "-x265-params", "qp=6:range=full", "-f", "null", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        _feed_raw(p, bytes(32 * 32 * 2), 1)
        ok = p.wait(timeout=timeout) == 0
    except Exception:
        ok = False
    _ENCODER_CACHE[key] = ok
    return ok


def find_depth_12bit_ffmpeg() -> Optional[str]:
    """候选二进制中第一个支持 gray12le 12-bit 编码的；全不支持返回 None。"""
    return next((b for b in list_working_ffmpegs() if probe_depth_12bit(b)),
                None)


# ═══════════════════════════════════════════════════════
#  选择入口
# ═══════════════════════════════════════════════════════

def select_encoder(prefer: str, width: int, height: int, fps: float,
                   n_streams: int, log: Optional[Callable[[str], None]] = None
                   ) -> Optional[EncoderChoice]:
    """选择录制编码器。返回 EncoderChoice，全链失败返回 None。

    prefer: settings.RECORD_VIDEO_ENCODER（"auto"|"nvenc"|"x265"|"x264"）
    auto 链: hevc_nvenc(探针可用) → libx265(速度达标) → libx264(兜底)
    显式指定: 跳过速度门槛强制用该编码器；二进制不支持/探针失败时按
    auto 链回退并播报原因。
    """
    if log is None:
        log = lambda msg: None
    cache_key = (prefer, width, height, fps,
                 min(int(n_streams), settings.ENCODER_PROBE_MAX_STREAMS))
    if cache_key in _SELECT_CACHE:
        return _SELECT_CACHE[cache_key]
    bins = list_working_ffmpegs()
    if not bins:
        log("[编码器] 未找到可用 ffmpeg 二进制")
        _SELECT_CACHE[cache_key] = None
        return None

    def find_with(enc: str) -> Optional[str]:
        return next((b for b in bins if ffmpeg_has_encoder(b, enc)), None)

    def make(enc: str, ffmpeg: str, probe: Optional[dict] = None) -> EncoderChoice:
        is_hevc = enc != "libx264"
        crf = settings.RECORD_VIDEO_CRF if is_hevc else settings.RECORD_VIDEO_X264_CRF
        return EncoderChoice(
            encoder=enc,
            codec="HEVC" if is_hevc else "H.264",
            crf=crf,
            args=encoder_args(enc),
            ffmpeg=ffmpeg,
            label=("HEVC (%s)" % enc) if is_hevc else "H.264 (libx264)",
            probe=probe or {},
            selected_by=prefer)

    def auto_chain(why: Optional[str] = None) -> Optional[EncoderChoice]:
        if why:
            log(f"[编码器] {why}，按 auto 链回退")
        exe = find_with("hevc_nvenc")
        if exe and probe_nvenc(exe, width, height, fps,
                               settings.RECORD_VIDEO_CRF):
            log("[编码器] hevc_nvenc 探针通过 → HEVC 硬件编码")
            return make("hevc_nvenc", exe, {"kind": "nvenc", "ok": True})
        if exe:
            log("[编码器] hevc_nvenc 探针失败（GPU 不可用），跳过")
        exe = find_with("libx265")
        if exe:
            probe = probe_x265(exe, width, height, fps, n_streams)
            if probe["passed"]:
                log(f"[编码器] libx265 实测 {probe['min_fps']:.0f} fps/流 "
                    f"(门槛 {probe['threshold_fps']:.0f}) → HEVC 软编")
                return make("libx265", exe, probe)
            log(f"[编码器] libx265 实测 {probe['min_fps']:.0f} fps/流 "
                f"(门槛 {probe['threshold_fps']:.0f}) → 不达标")
        exe = find_with("libx264")
        if exe:
            log(f"[编码器] 回退 H.264 (libx264 ultrafast, "
                f"CRF{settings.RECORD_VIDEO_X264_CRF})")
            return make("libx264", exe)
        return None

    if prefer == "nvenc":
        exe = find_with("hevc_nvenc")
        if exe and probe_nvenc(exe, width, height, fps,
                               settings.RECORD_VIDEO_CRF):
            log("[编码器] 显式 nvenc：hevc_nvenc 探针通过")
            choice = make("hevc_nvenc", exe, {"kind": "nvenc", "ok": True})
        else:
            choice = auto_chain("显式 nvenc 但 hevc_nvenc 探针失败")
    elif prefer == "x265":
        exe = find_with("libx265")
        if exe:
            log("[编码器] 显式 x265：跳过速度门槛，强制 HEVC 软编")
            choice = make("libx265", exe)
        else:
            choice = auto_chain("显式 x265 但无二进制支持 libx265")
    elif prefer == "x264":
        exe = find_with("libx264")
        if exe:
            log("[编码器] 显式 x264：跳过探针，直接 H.264")
            choice = make("libx264", exe)
        else:
            choice = auto_chain("显式 x264 但无二进制支持 libx264")
    else:  # auto
        choice = auto_chain()

    _SELECT_CACHE[cache_key] = choice
    return choice

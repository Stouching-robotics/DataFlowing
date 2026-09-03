#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rec001 离线深度处理：已上色双目视频 → SDK 深度引擎（CPU SGBM）→ 12-bit 灰度深度 MP4。

与主程序实时深度同款引擎（libfayssense_aikit_depth.so，自包含
StereoDepth：initFromYaml + initCameraFromCalibYaml + FeedStereoImage +
GetDepthImage）。**不触碰相机**——无 vikit 句柄、无 BindViKit：标定走
合成 kalibr yaml（工厂标定内参 ×2 缩放到 1280×800，与 rec001
info.json 记录值逐位一致，2026-09-03 核对）。

输入：s80c_arm_convert/{left,right}_episode-000.mp4（回色成品，
debayer 相位已按用户终裁 左 bggr/右 rggb 钉死；左右目逐帧
hardware_ns 全同 → 帧号 i 对 i 即立体对）。
输出：12-bit gray HEVC MP4（gray12le 对数码 qp=6 range=full hvc1，
主程序 v1.1.2 存储口径，core/depth_codec 同款量化）+ JET 预览
h264（VLC 放不了 gray12le，预览片肉眼验收用）。

用法：
  python offline_depth.py --frames 30          # 冒烟（30 帧）
  python offline_depth.py                      # 全程
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import tempfile

# ── 路径 ───────────────────────────────────────────────────
_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_TOOL_DIR))
_HAND3D = os.path.join(_REPO, "tools", "hand_3d_s80c")
_SDK_LIB = os.path.join(_HAND3D, "third_party", "lib")
_OPENCV406 = os.path.join(_HAND3D, "third_party", "opencv4.2", "lib406")
_DEPTH_CFG = os.path.join(_HAND3D, "third_party", "config", "stereo_depth.yaml")
_MODEL = os.path.join(_HAND3D, "third_party", "config", "models",
                      "rk3588", "stereo_s_general.rknn")
_FACTORY_CALIB = os.path.join(
    _REPO, "config", "FS-VI80-S80C_3500000262190088_camera_calibration.yaml")
_FFMPEG = "/home/stouch/miniconda3/envs/lerobot/bin/ffmpeg"

_ap = argparse.ArgumentParser(description="rec001 离线双目深度（SDK 引擎，无相机）")
_ap.add_argument("--left", default=os.path.join(
    _TOOL_DIR, "left_episode-000.mp4"))
_ap.add_argument("--right", default=os.path.join(
    _TOOL_DIR, "right_episode-000.mp4"))
_ap.add_argument("--out", default=os.path.join(
    _TOOL_DIR, "depth_episode-000.mp4"),
    help="12-bit 灰度深度 MP4 输出（主程序存储口径）")
_ap.add_argument("--preview", default=os.path.join(
    _TOOL_DIR, "depth_episode-000_preview.mp4"),
    help="JET 热力图预览 h264（VLC 放不了 gray12le，预览片肉眼验收）")
_ap.add_argument("--frames", type=int, default=0, metavar="N",
                 help="只处理前 N 帧（0=全程）")
_ap.add_argument("--depth-config", default=_DEPTH_CFG)
_ap.add_argument("--calib-yaml", default=_FACTORY_CALIB,
                 help="工厂 kalibr 标定（内参 640×400，自动 ×2）")
_args = _ap.parse_args()


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# ── LD_LIBRARY_PATH 自愈（与 worker 同款：lib406 依赖链靠动态
# 链接器搜索路径解析，进程内改不了 → re-exec）────────────────
if os.path.isdir(_OPENCV406) and \
        _OPENCV406 not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    os.environ["LD_LIBRARY_PATH"] = _OPENCV406 + ":" + \
        os.environ.get("LD_LIBRARY_PATH", "")
    _log(f"[Env] 预置 LD_LIBRARY_PATH: {_OPENCV406}，re-exec …")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ── OpenCV 4.2 预加载（必须早于 import cv2 与 SDK 库加载；
# 深度引擎按 SDK 自带 OpenCV 4.2.0 编译）─────────────────────
from ctypes import (  # noqa: E402
    c_void_p, c_char_p, c_int, c_short, c_float, c_ubyte, c_uint,
    c_ulonglong, POINTER, Structure, byref, cast,
)


def _preload_opencv_dir(libdir):
    for dep in ["libwebp.so.6", "libtiff.so.5", "libtbb.so.2",
                "libdc1394.so.22", "libavcodec.so.58", "libavformat.so.58",
                "libavutil.so.56", "libswscale.so.5"]:
        p = os.path.join(libdir, dep)
        if os.path.exists(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    for mod in ["libopencv_core", "libopencv_imgproc", "libopencv_imgcodecs",
                "libopencv_highgui", "libopencv_calib3d", "libopencv_ximgproc",
                "libopencv_features2d", "libopencv_flann", "libopencv_video",
                "libopencv_videoio", "libopencv_photo", "libopencv_objdetect",
                "libopencv_ml"]:
        for suffix in [".so.406", ".so.4.2"]:
            p = os.path.join(libdir, mod + suffix)
            if os.path.exists(p):
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                    break
                except OSError:
                    pass


if os.path.isdir(_OPENCV406):
    _preload_opencv_dir(_OPENCV406)
else:
    _log(f"[ERROR] lib406 目录不存在: {_OPENCV406}")
    sys.exit(1)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# 主程序深度码值口径（量化 LUT / JET / 编码参数）
sys.path.insert(0, _REPO)
from core.depth_codec import (  # noqa: E402
    quantize_depth, codes_to_heatmap_bgr, depth_video_encoder_args)

# ── SDK C 结构体（fays_atrak_types.h，与 worker 同款）──────


class AtrakImage(Structure):
    _fields_ = [
        ("device_id", c_int), ("seq", c_int), ("timestamp", c_ulonglong),
        ("width", c_int), ("height", c_int), ("channel", c_int),
        ("encoding", c_short), ("step", c_int), ("bytes", c_uint),
        ("data", POINTER(c_ubyte)),
    ]


class AtrakDepthImage(Structure):
    """深度输出: data 指向 float 缓冲（单位: 米）"""
    _fields_ = [
        ("device_id", c_int), ("seq", c_int), ("timestamp", c_ulonglong),
        ("width", c_int), ("height", c_int), ("channel", c_int),
        ("encoding", c_short), ("step", c_int), ("bytes", c_uint),
        ("data", POINTER(c_float)),
    ]


# ── 加载库（vikit 必须 RTLD_GLOBAL：深度库要解析 FAYS_VIK_*）──
_ft602 = os.path.join(_SDK_LIB, "libft602.so")
if os.path.isfile(_ft602):
    try:
        ctypes.CDLL(_ft602, mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        _log(f"[WARN] libft602 预载失败: {e}")
_vikit = ctypes.CDLL(os.path.join(_SDK_LIB, "libfays_vikit.so"),
                     mode=ctypes.RTLD_GLOBAL)
_depth_lib = ctypes.CDLL(os.path.join(
    _SDK_LIB, "libfayssense_aikit_depth.so"))

_d_create = _depth_lib.FAYS_ATRAK_D_CreateHandleWithConfig
_d_destroy = _depth_lib.FAYS_ATRAK_D_DestroyHandle
_d_feed = _depth_lib.FAYS_ATRAK_D_FeedStereoImage
_d_get = _depth_lib.FAYS_ATRAK_D_GetDepthImage

_d_create.argtypes = [POINTER(c_void_p), c_char_p]
_d_create.restype = c_int
_d_destroy.argtypes = [c_void_p]
_d_destroy.restype = c_int
_d_feed.argtypes = [c_void_p, POINTER(AtrakImage)]
_d_feed.restype = c_int
_d_get.argtypes = [c_void_p, POINTER(AtrakDepthImage),
                   POINTER(AtrakDepthImage), POINTER(AtrakImage)]
_d_get.restype = c_int

# ── 引擎标定合成：工厂 yaml 内参 ×2（640×400 → 1280×800），
# 其余（T_cn_cnm1/T_cam_imu/畸变）原样。合成值逐位核对过
# info.json 记录标定（fx 457.22/457.03 等）。──────────────────
_TMP_FILES = []


def _synth_engine_calib(factory: str) -> str:
    lines = []
    with open(factory, encoding="utf-8") as f:
        for line in f:
            if "intrinsics:" in line:
                v = [float(x) for x in
                     line.split("intrinsics:", 1)[1].strip()
                     .strip("[]").split(",")]
                line = (f"  intrinsics: [{v[0] * 2}, {v[1] * 2}, "
                        f"{v[2] * 2}, {v[3] * 2}]\n")
            elif "resolution:" in line:
                line = "  resolution: [1280, 800]\n"
            lines.append(line)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_engine_calib.yaml", delete=False)
    tmp.write("".join(lines))
    tmp.close()
    _TMP_FILES.append(tmp.name)
    return tmp.name


_engine_calib = _synth_engine_calib(_args.calib_yaml)

# ── 深度配置副本：calib_path/model_path 改写 + async 关（同步喂帧）──
with open(_args.depth_config, encoding="utf-8") as _src:
    _cfg_lines = []
    for _line in _src:
        if _line.startswith("calib_path"):
            _line = f"calib_path: {_engine_calib}\n"
        elif _line.startswith("model_path"):
            _line = f"model_path: {_MODEL}\n"
        elif _line.startswith("async_mode"):
            _line = "async_mode: 0\n"   # 0=feed 阻塞至结果就绪（离线最简）
        _cfg_lines.append(_line)
_tmp_cfg = tempfile.NamedTemporaryFile(
    mode="w", suffix="_offline_depth.yaml", delete=False)
_tmp_cfg.write("".join(_cfg_lines))
_tmp_cfg.close()
_TMP_FILES.append(_tmp_cfg.name)
_log(f"[Cfg] 引擎配置 {_tmp_cfg.name}（async 同步，calib→合成标定）")

# ── 创建引擎（无 vikit 句柄、无 BindViKit——离线自包含）────
depth_handle = c_void_p()
_ret = _d_create(byref(depth_handle), _tmp_cfg.name.encode())
if _ret != 0:
    _log(f"[ERROR] 深度引擎创建失败, ret={_ret}")
    sys.exit(1)
_log("深度引擎创建成功（CPU SGBM，无相机句柄）")

# ── 缓冲区（与 worker 同款上限，防越界写堆）────────────────
MAX_STEREO_BYTES = 1280 * 800 * 2 * 3
MAX_DEPTH_FLOATS = 3840 * 2160 * 3 * 4 // 4

_feed_bufs = ((c_ubyte * MAX_STEREO_BYTES)(),
              (c_ubyte * MAX_STEREO_BYTES)())
_feed_alt = False
_depth_buf = (c_float * MAX_DEPTH_FLOATS)()

_feed_img = AtrakImage()
_feed_img.data = cast(_feed_bufs[0], POINTER(c_ubyte))
_depth_out = AtrakDepthImage()
_depth_out.data = cast(_depth_buf, POINTER(c_float))
_disp_out = AtrakDepthImage()
_disp_out.data = cast((c_float * MAX_DEPTH_FLOATS)(), POINTER(c_float))
_paired_out = AtrakImage()
_paired_out.data = cast((c_ubyte * MAX_STEREO_BYTES)(), POINTER(c_ubyte))

STREAM_W, STREAM_H = 1280, 800
_AIE_BGR8 = 0


def _feed_pair(i: int, left_bgr: np.ndarray, right_bgr: np.ndarray):
    """上下拼接（上=左/下=右）喂引擎。"""
    global _feed_alt
    stacked = np.ascontiguousarray(np.vstack([left_bgr, right_bgr]))
    _feed_alt = not _feed_alt
    _feed_img.data = cast(_feed_bufs[_feed_alt], POINTER(c_ubyte))
    ctypes.memmove(_feed_img.data, stacked.ctypes.data, stacked.nbytes)
    _feed_img.width = STREAM_W
    _feed_img.height = STREAM_H * 2
    _feed_img.channel = 3
    _feed_img.encoding = _AIE_BGR8
    _feed_img.step = STREAM_W * 3
    _feed_img.bytes = stacked.nbytes
    _feed_img.timestamp = i
    _feed_img.seq = i
    _feed_img.device_id = 0
    return _d_feed(depth_handle, byref(_feed_img))


def _get_depth():
    for _out in (_depth_out, _disp_out, _paired_out):
        _out.width = _out.height = _out.channel = 0
        _out.step = _out.bytes = 0
    if _d_get(depth_handle, byref(_depth_out), byref(_disp_out),
              byref(_paired_out)) != 0 or _depth_out.width <= 0:
        return None
    n = _depth_out.bytes // 4
    arr = np.ctypeslib.as_array(_depth_out.data, shape=(n,)).copy()
    return arr.reshape(_depth_out.height, _depth_out.width)


def main():
    cap_l = cv2.VideoCapture(_args.left)
    cap_r = cv2.VideoCapture(_args.right)
    if not (cap_l.isOpened() and cap_r.isOpened()):
        _log("[ERROR] 打不开输入视频（左右目）")
        return 1
    total = min(int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    if _args.frames:
        total = min(total, _args.frames)
    _log(f"[In] 处理 {total} 帧（左 {_args.left} / 右 {_args.right}）")

    proc_d, proc_p = None, None
    written = 0
    got = 0
    depth_size = None
    _stat_valid = []
    _stat_med = []

    for i in range(total):
        ok_l, f_l = cap_l.read()
        ok_r, f_r = cap_r.read()
        if not (ok_l and ok_r):
            _log(f"[WARN] 帧 {i} 读取失败（左 {ok_l}/右 {ok_r}），停")
            break
        if f_l.shape[:2] != (STREAM_H, STREAM_W) or \
                f_r.shape[:2] != (STREAM_H, STREAM_W):
            _log(f"[ERROR] 帧 {i} 尺寸不对: {f_l.shape} / {f_r.shape}")
            break
        if _feed_pair(i, f_l, f_r) != 0:
            _log(f"[ERROR] FeedStereoImage 帧 {i} 失败（无 Bind 不支持？）")
            break
        depth_m = _get_depth()
        if depth_m is None:
            if i < 10:
                continue    # 预热：引擎未产出
            _log(f"[WARN] 帧 {i} 无深度输出，跳过")
            continue
        got += 1
        if proc_d is None:
            depth_size = (depth_m.shape[1], depth_m.shape[0])
            _log(f"[Depth] 深度尺寸 {depth_size[0]}×{depth_size[1]}，"
                 f"启动编码器")
            proc_d = subprocess.Popen(
                [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "gray12le",
                 "-s", f"{depth_size[0]}x{depth_size[1]}", "-r", "30",
                 "-i", "pipe:0",
                 "-c:v", "libx265", *depth_video_encoder_args(),
                 _args.out],
                stdin=subprocess.PIPE)
            proc_p = subprocess.Popen(
                [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{depth_size[0]}x{depth_size[1]}", "-r", "30",
                 "-i", "pipe:0",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                 _args.preview],
                stdin=subprocess.PIPE)
        # 米 → 毫米 → 主程序口径对数码
        mm = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        mm = (mm * 1000.0).astype(np.float32)
        valid = np.isfinite(mm) & (mm >= 50.0)
        mm = np.clip(mm, 0, 65535).astype(np.uint16)
        codes = quantize_depth(mm)
        _stat_valid.append(float(valid.mean()))
        if valid.any():
            _stat_med.append(float(np.median(mm[valid])))
        proc_d.stdin.write(codes.tobytes())
        proc_p.stdin.write(codes_to_heatmap_bgr(codes).tobytes())
        written += 1
        if (i + 1) % 120 == 0:
            _log(f"[stat] {i + 1}/{total} 深度 {got} 帧 | 有效 "
                 f"{np.mean(_stat_valid[-120:]) * 100:.0f}% | 中位 "
                 f"{np.median(_stat_med[-120:]):.0f}mm" if _stat_med
                 else f"[stat] {i + 1}/{total} 深度 {got} 帧 | 无有效深度")

    cap_l.release()
    cap_r.release()
    for _p in (proc_d, proc_p):
        if _p is not None:
            _p.stdin.close()
            _p.wait()
    _d_destroy(depth_handle)
    for _t in _TMP_FILES:
        try:
            os.unlink(_t)
        except OSError:
            pass
    _log(f"[Done] 深度帧 {written}/{total}（引擎产出 {got}）| 全程有效 "
         f"{np.mean(_stat_valid) * 100:.1f}% | 中位 "
         f"{np.median(_stat_med):.0f}mm | 输出 {_args.out} + {_args.preview}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _d_destroy(depth_handle)
        sys.exit(130)

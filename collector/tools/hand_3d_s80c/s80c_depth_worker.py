#!/usr/bin/env python3
"""S80C 深度 worker 子进程 —— 供 live_demo_s80c.py 拉起的 SDK 桥接进程。

职责（本进程独占 SDK ctypes + 预载 OpenCV 4.2，隔离于 demo 进程的
venv cv2/mediapipe）：
  1. FAYS_VIK_CreateHandleWithConfig 打开 S80C 双目相机（端口按
     sysfs 自动解析，见 read_stereo_rgb.py 教训：yaml 写死 /dev/videoN
     会被 USB 插拔挤位）
  2. FAYS_ATRAK_D_CreateHandleWithConfig 创建深度引擎（CPU SGBM，
     async_mode:1 预热 ≥5 帧）并 BindViKit
  3. 每帧：取双目帧（上=左/下=右，swap_lr:1 已在传感器层解决）→
     RGB→BGR → FeedStereoImage → GetDepthImage（米制 float32 ×1000
     → 毫米）→ 鱼眼矫正左目图（cv2.fisheye.initUndistortRectifyMap
     with R0/P0，2D 检测/显示视图，与深度同 P0 空间）
  4. stdout 二进制管道输出（fd1 重定向到 stderr，拦截 SDK printf）：

     [1B type][4B seq][8B ts_ns][4B w][4B h][4B len][payload]  大端
       type=0  JSON 握手（P0 内参 + 分辨率 + 基线，demo 建 align_calib 用）
       type=1  JPEG 编码矫正左目 BGR（2D 检测/显示帧）
       type=2  raw float32 毫米深度（w×h×4B；引擎更新间隔内重复最近
               一张发送，3D 链帧间不丢观测 → tracker 平滑衔接）

  5. 退出清理：销毁深度/相机句柄 + 删临时 yaml（FT602 设备不留坏状态）

库选择：默认 3.9.0（仓库内 third_party/lib 自包含副本，RGB 失败
段错误史的 3.9.1 不用）；OpenCV 4.2 预载自 third_party/opencv4.2/lib406
（SDK .so 无 DT_NEEDED，靠 RTLD_GLOBAL 全局符号表解析）。
"""

import argparse
import ctypes
import json
import os
import re
import signal
import struct
import sys
import tempfile
import time
from collections import deque
from ctypes import (
    c_void_p, c_char_p, c_int, c_short, c_float, c_size_t,
    c_ubyte, c_uint, c_uint8, c_uint32, c_ulonglong, c_double,
    POINTER, Structure, byref, cast,
)

_WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_WORKER_DIR))


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# ═══════════════════════════════════════════════════════════════
#  路径（--* 可覆盖，默认值本机已核实的组合）
# ═══════════════════════════════════════════════════════════════

_ap = argparse.ArgumentParser(
    description="S80C 深度 worker（由 live_demo_s80c.py 拉起，勿手跑）")
_THIRD_PARTY = os.path.join(_WORKER_DIR, "third_party")
_ap.add_argument("--sdk-dir",
                 default=os.path.join(_THIRD_PARTY, "lib"),
                 help="3.9.0 库目录（libfays_vikit.so + libfayssense_aikit_depth.so；"
                      "默认仓库内 third_party/lib 自包含副本，无外部依赖）")
_ap.add_argument("--vikit-config",
                 default=os.path.join(_REPO_ROOT, "tools", "stereo_s80m",
                                      "config", "fays_vikit_50fps.yaml"),
                 help="相机配置模板（默认 50fps 副本，与主程序同款；端口"
                      "自动解析会重写临时副本）")
_ap.add_argument("--pipe-format", choices=("raw", "jpeg"), default="raw",
                 help="管道帧格式：raw=裸 BGR（默认，零压缩零伪影，"
                      "MediaPipe 最稳；50fps 下 ~310MB/s 管道带宽）；"
                      "jpeg=JPEG95（旧格式，调试对照用）")
_ap.add_argument("--depth-config",
                 default=os.path.join(_THIRD_PARTY, "config",
                                      "stereo_depth.yaml"),
                 help="深度引擎配置（depth_mode:1 CPU SGBM；默认仓库内副本，"
                      "calib_path/model_path 由 worker 运行时重写为仓库内绝对路径）")
_ap.add_argument("--opencv-dir",
                 default=os.path.join(_THIRD_PARTY, "opencv4.2", "lib406"),
                 help="SDK 自带 OpenCV 4.2 预载目录（lib406；默认仓库内"
                      "third_party/opencv4.2/lib406 自包含副本）")
_ap.add_argument("--calib-yaml",
                 default=os.path.join(_REPO_ROOT, "tools", "fayssense_depth_sdk",
                                      "calib", "calib.yaml"),
                 help="回退标定（SDK GetCalibrationParam 失败时："
                      "引擎 calib_path 与 K1/D1 同用 cam0）")
_ap.add_argument("--rect-mode", choices=("remap", "sdk"), default="remap",
                 help="2D 视图来源：remap=自身鱼眼矫正（默认，与深度同"
                      "P0 空间）；sdk=引擎 GetRectifiedImage 左半（验证"
                      "对照用，尺寸可能非 1280×800 → 3D 对齐失效）")
_ap.add_argument("--stereo-view", action="store_true",
                 help="管道额外发右目矫正帧（type=3）：demo 把 win1 显示为"
                      "左右并排（右目仅显示，检测/深度/3D 仍只用左目）")
_ap.add_argument("--dump-rect-info", metavar="PATH",
                 help="诊断：GetRectifyInfo/标定完成后把 R0/P0/R1/P1/"
                      "K1/D1/K2/D2 写 JSON 并退出（不进入取帧循环）——"
                      "离线复现 remap 用")
_ap.add_argument("--raw-dump", metavar="DIR",
                 help="撕裂诊断：退出（demo 侧 q/SIGTERM）时把最近若干张 "
                      "pre-remap 原始帧（raw_<ts_ns>.jpg）与 post-remap 发送帧"
                      "（sentL_/sentR_<ts_ns>.jpg，即 demo 收到的字节）导出 "
                      "JPEG 到 DIR；编码只在退出做，不占实时预算——与 demo "
                      "--tear-probe 同轮捕获：raw 有缝=SDK/相机侧；raw 净而 "
                      "sent 有缝=remap/引擎链；sent 也净=缝在 demo 合成/显示侧")
_ap.add_argument("--raw-ring", type=int, default=32, metavar="N",
                 help="撕裂诊断：raw 环帧数（默认 32≈0.64s@50fps）。demo "
                      "tear 环 96 帧≈2.4-3.2s，要覆盖标记帧时段需 N≥160"
                      "（存引用零实时成本，内存 ~6MB×N）")
_ap.add_argument("--raw-full", action="store_true",
                 help="撕裂诊断：raw 导出全尺寸（默认半尺寸 0.5×——S80C "
                      "缝比 S80M 细，半尺寸机器检测可能漏；全尺寸单帧 "
                      "~300-500KB JPEG，160 帧 ~60MB）")
_ap.add_argument("--race-probe", action="store_true",
                 help="撕裂诊断（实验，默认关）：每次取帧拷贝后立即二次"
                      "memcpy 逐字节比对，统计『拷贝窗口内 SDK 改写了缓冲』"
                      "的帧数并按秒打日志——客观坐实/排除写缓冲与拷贝竞态")
_ap.add_argument("--double-buffer", action="store_true",
                 help="撕裂诊断（实验，默认关）：每次取帧前交替把 "
                      "stereo_img.data 指向 A/B 两块缓冲。风险：SDK 若缓存 "
                      "data 指针会把写入引向旧缓冲更糟——仅在 --race-probe "
                      "出正信号后试，验证无效即关")
_ap.add_argument("--settle-poll", action="store_true",
                 help="撕裂诊断/修复候选（实验，默认关）：GetStereoFrames "
                      "返回后反复快照比对直到缓冲连续两次一致（最多 ~2-3ms）"
                      "才拷贝使用——若 SDK 返回时其写入线程尚未写完缓冲"
                      "（返回早于写完，拷贝与之按行交错=水平缝且缝行漂移），"
                      "该等待消除撕裂；缓冲本就稳定时零额外等待。"
                      "2026-08-31 实机已证无效（带率 21.0%%→18.2%%），"
                      "仅留作诊断")
_ap.add_argument("--cb-bridge", action="store_true", default=True,
                 help="回调取帧（默认开=主程序同款撕裂修复）：改用 SDK "
                      "RegisterStereoImageCallback 回调取帧（官方 "
                      "stereo_depth_gui 同款路径——SDK 装配线程写完帧才"
                      "回调，绕过 GetStereoFrames 内部拷贝与装配的竞态，"
                      "该竞态疑为水平缝根因：缝行随机漂移、两目共模、"
                      "settle-poll 证明返回后缓冲已稳定=缝在 SDK 内拷贝"
                      "时冻结）。经 third_party/cb_bridge/ 桥接 "
                      "std::function→C ABI（ctypes 可调）")
_ap.add_argument("--no-cb-bridge", action="store_false", dest="cb_bridge",
                 help="回退轮询 GetStereoFrames 取帧（诊断对照用；回调注册"
                      "失败时也自动回退轮询）")
_args, _ = _ap.parse_known_args()

VIKIT_LIB = os.path.join(_args.sdk_dir, "libfays_vikit.so")
DEPTH_LIB = os.path.join(_args.sdk_dir, "libfayssense_aikit_depth.so")
VIKIT_CONFIG = _args.vikit_config
DEPTH_CONFIG = _args.depth_config
OPENCV_LIB406 = _args.opencv_dir
CALIB_YAML = _args.calib_yaml
RECT_MODE = _args.rect_mode
STEREO_VIEW = _args.stereo_view
PIPE_FORMAT = _args.pipe_format
RAW_DUMP_DIR = _args.raw_dump
RAW_RING = _args.raw_ring
RAW_FULL = _args.raw_full
RACE_PROBE = _args.race_probe
DOUBLE_BUFFER = _args.double_buffer
SETTLE_POLL = _args.settle_poll
CB_BRIDGE = _args.cb_bridge
# 桥接锚定仓库内 third_party/cb_bridge/（不随 --sdk-dir 覆盖漂移——
# --sdk-dir 可指向外部 SDK 目录做版本对照，桥接只此一份）
CB_BRIDGE_LIB = os.path.join(_THIRD_PARTY, "cb_bridge",
                             "libfays_cb_bridge.so")

for _p, _n in ((VIKIT_LIB, "libfays_vikit.so"),
               (DEPTH_LIB, "libfayssense_aikit_depth.so"),
               (VIKIT_CONFIG, "相机配置"),
               (DEPTH_CONFIG, "深度引擎配置")):
    if not os.path.isfile(_p):
        _log(f"[ERROR] 找不到 {_n}: {_p}")
        sys.exit(1)

# ── LD_LIBRARY_PATH 自愈 ───────────────────────────────────────
# lib406 的 libopencv_*.so.406 依赖链用裸 SONAME（libopencv_*.so.4.2、
# libwebp.so.6…），必须靠 lib406 在动态链接器搜索路径上解析（同
# tools/demos/run_stereo_depth_demo.sh 做法）；LD_LIBRARY_PATH 只在进程
# exec 时被读取、进程内改不了 → 缺失时 re-exec 自身（stdout 管道 fd
# 跨 exec 保留，二进制协议不受影响）。这样 worker 无论被 demo spawn
# 还是手动单跑都自洽。
if OPENCV_LIB406 and os.path.isdir(OPENCV_LIB406) and \
        OPENCV_LIB406 not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    _log(f"[Env] 预置 LD_LIBRARY_PATH: {OPENCV_LIB406}，re-exec …")
    os.environ["LD_LIBRARY_PATH"] = OPENCV_LIB406 + ":" + \
        os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ── 输出管道（必须在任何 SDK/cv2 活动前接管 stdout）────────────
# SDK 相机线程在 create 后**异步**向 stdout printf（[Camera Chain]
# 标定 dump + 每秒 IMU FPS/Stereo FPS 行）。若重定向晚于 create，
# 这些文本先入管道污染二进制头（demo 解析失步 → 握手超时，实测
# 教训：重定向放后面时 [Camera Chain] dump 竞态时有时无）。因此一
# 进进程就 dup：fd1→stderr 拦截全部 SDK printf，保存的原始 stdout
# 专走二进制帧协议。re-exec 不丢 fd（跨 exec 保留）。
_pipe_fd = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr
PIPE_FD = os.fdopen(_pipe_fd, "wb", buffering=0)
_HEADER = struct.Struct(">BIQIII")
_META_TYPE, _RGB_TYPE, _DEPTH_TYPE, _RIGHT_TYPE = 0, 1, 2, 3
_RAW_RGB_TYPE, _RAW_RIGHT_TYPE = 4, 5    # raw BGR 帧（--pipe-format raw）


def _send(typ: int, seq: int, ts: int, w: int, h: int, payload: bytes):
    PIPE_FD.write(_HEADER.pack(typ, seq, ts, w, h, len(payload)))
    PIPE_FD.write(payload)


# ── OpenCV 预加载（必须早于 import cv2 与 SDK 库加载）──────────
# 深度引擎按 SDK 自带 OpenCV 4.2.0 编译（cv::stereoRectify 传
# MatExpr/STD_ARRAY，系统 4.6+ 的 getMat_ 不支持会崩）。找不到回退
# 系统 .so.406（引擎在其下可能崩，仅限无自带 4.2 的环境）。


def _preload_opencv_dir(libdir):
    for dep in ["libwebp.so.6", "libtiff.so.5", "libtbb.so.2", "libdc1394.so.22",
                "libavcodec.so.58", "libavformat.so.58", "libavutil.so.56",
                "libswscale.so.5"]:
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


if os.path.isdir(OPENCV_LIB406):
    _preload_opencv_dir(OPENCV_LIB406)
else:
    _log(f"[WARN] lib406 目录不存在: {OPENCV_LIB406}，回退系统 OpenCV")
    for _lib in ["libopencv_core", "libopencv_imgproc", "libopencv_imgcodecs",
                 "libopencv_highgui", "libopencv_calib3d"]:
        for _suffix in [".so.406", ".so.4.2"]:
            try:
                ctypes.CDLL(_lib + _suffix, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue

# cv2 在预载后导入（venv cv2 走自身 rpath 绑自家 OpenCV，与 SDK 的
# 4.2 全局符号互不干扰）
import numpy as np
import cv2

_OV_PLUGINS_DIR = os.path.join(os.path.dirname(cv2.__file__), "qt", "plugins")
_OV_PLATFORMS = os.path.join(_OV_PLUGINS_DIR, "platforms.bak")
if not os.path.isdir(_OV_PLATFORMS):
    _OV_PLATFORMS = os.path.join(_OV_PLUGINS_DIR, "platforms")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _OV_PLATFORMS
os.environ["QT_QPA_PLATFORM"] = "xcb"

# ═══════════════════════════════════════════════════════════════
#  设备节点自动解析（read_stereo_rgb.py 同款：yaml 写死端口会随
#  USB 插拔漂移；FTDI 接口 1.0=双目、1.2=IMU；rgb 一律 NULL）
# ═══════════════════════════════════════════════════════════════

_PORTS_TMP = None
try:
    _ports_nodes = {}
    for _name in sorted(os.listdir("/sys/class/video4linux")):
        _vp = os.path.join("/sys/class/video4linux", _name)
        try:
            _dname = open(os.path.join(_vp, "name"), encoding="utf-8") \
                .read().strip()
        except OSError:
            continue
        if "FTDI Superspeed Video Bridge" not in _dname:
            continue
        try:
            _iface = os.path.basename(os.readlink(os.path.join(_vp, "device")))
        except OSError:
            continue
        _m = re.match(r".+:(\d+\.\d+)$", _iface)
        if not _m:
            continue
        _ports_nodes.setdefault(_m.group(1), []).append(f"/dev/{_name}")
    _stereo_port = next(iter(_ports_nodes.get("1.0", [])), None)
    _imu_port = next(iter(_ports_nodes.get("1.2", [])), None)
    if _stereo_port:
        _tf = tempfile.NamedTemporaryFile(
            mode="w", suffix="_s80c_ports.yaml", delete=False)
        with open(VIKIT_CONFIG, encoding="utf-8") as _src:
            for _line in _src:
                if _line.startswith("rgb_dev_port"):
                    _line = "rgb_dev_port: NULL\n"
                elif _line.startswith("stereo_dev_port"):
                    _line = f"stereo_dev_port: {_stereo_port}\n"
                elif _line.startswith("imu_dev_port") and _imu_port:
                    _line = f"imu_dev_port: {_imu_port}\n"
                _tf.write(_line)
        _tf.close()
        _PORTS_TMP = _tf.name
        VIKIT_CONFIG = _PORTS_TMP
        _log(f"[Ports] 自动解析: stereo={_stereo_port} "
             f"imu={_imu_port or '?'} rgb=NULL")
    else:
        _log("[Ports] 未找到 FTDI 双目接口 1.0，沿用 yaml 端口"
             "（相机未插？）")
except Exception as _e:
    _log(f"[Ports] 解析失败，沿用 yaml 端口: {_e}")

# ═══════════════════════════════════════════════════════════════
#  深度配置运行时重写（calib_path/model_path → 仓库内绝对路径）
#  已移到「设备标定读取」之后执行：calib_path 指向运行时合成的
#  设备标定 yaml（见 _synthesize_engine_calib），机型自动适配。
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  SDK C 结构体（fays_atrak_types.h）
# ═══════════════════════════════════════════════════════════════

FAYS_ATRAK_MAX_CAMERAS = 3


class AtrakImage(Structure):
    _fields_ = [
        ("device_id",  c_int),
        ("seq",        c_int),
        ("timestamp",  c_ulonglong),
        ("width",      c_int),
        ("height",     c_int),
        ("channel",    c_int),
        ("encoding",   c_short),
        ("step",       c_int),
        ("bytes",      c_uint),
        ("data",       POINTER(c_ubyte)),
    ]


class AtrakDepthImage(Structure):
    """深度/视差输出: data 指向 float 缓冲区 (深度单位: 米)"""
    _fields_ = [
        ("device_id",  c_int),
        ("seq",        c_int),
        ("timestamp",  c_ulonglong),
        ("width",      c_int),
        ("height",     c_int),
        ("channel",    c_int),
        ("encoding",   c_short),
        ("step",       c_int),
        ("bytes",      c_uint),
        ("data",       POINTER(c_float)),
    ]


class AtrakRectifyInfo(Structure):
    _fields_ = [
        ("cam_pair", c_uint8 * 2),
        ("R0",       c_float * 9),
        ("R1",       c_float * 9),
        ("P0",       c_float * 12),
        ("P1",       c_float * 12),
    ]


class AtrakIntrinsics(Structure):
    _fields_ = [
        ("cam_model",          c_uint8),
        ("width",              c_uint32),
        ("height",             c_uint32),
        ("fx",                 c_float),
        ("fy",                 c_float),
        ("cx",                 c_float),
        ("cy",                 c_float),
        ("intrinsic_extra",    c_float * 4),
        ("distortion_model",   c_uint8),
        ("distortion",         c_float * 8),
    ]


class AtrakExtrinsics(Structure):
    _fields_ = [
        ("rot",   c_float * 9),
        ("trans", c_float * 3),
    ]


class AtrakCamParam(Structure):
    _fields_ = [
        ("cam_id",           c_uint8),
        ("available_mask",   c_uint8),
        ("intrinsics",       AtrakIntrinsics),
        ("T_cn_cnm1",        AtrakExtrinsics),
        ("T_cn_imu",         AtrakExtrinsics),
        ("timeshift_cam_imu", c_double),
    ]


class AtrakCamChainParam(Structure):
    _fields_ = [
        ("num_of_cams",    c_uint32),
        ("downsize_ratio", c_uint32),
        ("cameras",        AtrakCamParam * FAYS_ATRAK_MAX_CAMERAS),
    ]


class AtrakImuParam(Structure):
    _fields_ = [
        ("accelerometer_noise_density", c_double),
        ("accelerometer_random_walk",   c_double),
        ("gyroscope_noise_density",     c_double),
        ("gyroscope_random_walk",       c_double),
        ("update_rate",                 c_float),
    ]


class AtrakCalibrationParam(Structure):
    _fields_ = [
        ("cameras", AtrakCamChainParam),
        ("imu",     AtrakImuParam),
    ]


# ═══════════════════════════════════════════════════════════════
#  加载库（vikit 必须 RTLD_GLOBAL：深度库 dlopen 时要解析 FAYS_VIK_*）
# ═══════════════════════════════════════════════════════════════

# libft602（FT602 USB 桥驱动库）是 vikit 的 DT_NEEDED，官方安装器装进
# /lib——git clone 独立跑时靠仓库副本预载（与 read_stereo_rgb.py 同口径）。
_ft602 = os.path.join(_args.sdk_dir, "libft602.so")
if os.path.isfile(_ft602):
    try:
        ctypes.CDLL(_ft602, mode=ctypes.RTLD_GLOBAL)
    except OSError as _e:
        print(f"[WARN] libft602 预载失败: {_e}", file=sys.stderr)

_vikit = ctypes.CDLL(VIKIT_LIB, mode=ctypes.RTLD_GLOBAL)
try:
    _depth = ctypes.CDLL(DEPTH_LIB)
except OSError as e:
    _log(f"[ERROR] 深度引擎加载失败: {e}")
    sys.exit(1)

_vikit_create = getattr(_vikit, "_Z31FAYS_VIK_CreateHandleWithConfigPPvPKc")
_vikit_destroy = getattr(_vikit, "_Z22FAYS_VIK_DestroyHandlePv")
_vikit_stereo = getattr(_vikit, "_Z24FAYS_VIK_GetStereoFramesPvP10AtrakImage")
_vikit_version = getattr(_vikit, "_Z19FAYS_VIK_GetVersionPv")
try:
    _vikit_calib = getattr(_vikit,
                           "_Z28FAYS_VIK_GetCalibrationParamPvP21AtrakCalibrationParam")
except AttributeError:
    _vikit_calib = None

_vikit_create.argtypes = [POINTER(c_void_p), c_char_p]
_vikit_create.restype = c_int
_vikit_destroy.argtypes = [c_void_p]
_vikit_destroy.restype = c_int
_vikit_stereo.argtypes = [c_void_p, POINTER(AtrakImage)]
_vikit_stereo.restype = c_int
_vikit_version.argtypes = [c_void_p]
_vikit_version.restype = c_char_p
if _vikit_calib is not None:
    _vikit_calib.argtypes = [c_void_p, POINTER(AtrakCalibrationParam)]
    _vikit_calib.restype = c_int

_d_create = _depth.FAYS_ATRAK_D_CreateHandleWithConfig
_d_destroy = _depth.FAYS_ATRAK_D_DestroyHandle
_d_bind = _depth.FAYS_ATRAK_D_BindViKit
_d_feed = _depth.FAYS_ATRAK_D_FeedStereoImage
_d_get = _depth.FAYS_ATRAK_D_GetDepthImage
_d_rect = _depth.FAYS_ATRAK_D_GetRectifiedImage
_d_rectinfo = _depth.FAYS_ATRAK_D_GetRectifyInfo

_d_create.argtypes = [POINTER(c_void_p), c_char_p]
_d_create.restype = c_int
_d_destroy.argtypes = [c_void_p]
_d_destroy.restype = c_int
_d_bind.argtypes = [c_void_p, c_void_p]
_d_bind.restype = c_int
_d_feed.argtypes = [c_void_p, POINTER(AtrakImage)]
_d_feed.restype = c_int
_d_get.argtypes = [c_void_p, POINTER(AtrakDepthImage),
                   POINTER(AtrakDepthImage), POINTER(AtrakImage)]
_d_get.restype = c_int
_d_rect.argtypes = [c_void_p, POINTER(AtrakImage)]
_d_rect.restype = c_int
_d_rectinfo.argtypes = [c_void_p, POINTER(AtrakRectifyInfo)]
_d_rectinfo.restype = c_int

# SIGTERM/SIGINT 优雅退出（demo close() 先 terminate 再 kill；SDK 句柄
# 必须销毁，否则 FT602 设备留坏状态、后续进程初始化崩溃）
def _graceful_exit(sig, frame):
    raise SystemExit(0)


signal.signal(signal.SIGTERM, _graceful_exit)
signal.signal(signal.SIGINT, _graceful_exit)

# ═══════════════════════════════════════════════════════════════
#  初始化: 相机 → 深度引擎 → 绑定 → 矫正矩阵/内参
# ═══════════════════════════════════════════════════════════════

vikit_handle = c_void_p()
ret = _vikit_create(byref(vikit_handle), VIKIT_CONFIG.encode())
if ret != 0:
    _log(f"[ERROR] ViKit 初始化失败, ret={ret} "
         f"(相机是否连接? 端口: {VIKIT_CONFIG})")
    sys.exit(1)
ver = _vikit_version(vikit_handle)
_log(f"ViKit 版本: {ver.decode() if ver else 'unknown'}（lib: {VIKIT_LIB}）")

# ── 回调取帧桥接（--cb-bridge，默认开=主程序同款撕裂修复）──
# 官方 GUI 用 RegisterStereoImageCallback（SDK 装配完成才回调）无撕裂，
# 而轮询 GetStereoFrames 的缝疑在其内部拷贝与装配竞态（见 --cb-bridge
# help）。注册须在 handle 创建后立即做（GUI 同款时序）。
_cb_lib = None
if CB_BRIDGE and not os.path.isfile(CB_BRIDGE_LIB):
    _log(f"[WARN] 找不到回调桥接: {CB_BRIDGE_LIB}（third_party/"
         f"cb_bridge/build.sh 编译），回退轮询取帧")
    CB_BRIDGE = False
if CB_BRIDGE:
    _cb_lib = ctypes.CDLL(CB_BRIDGE_LIB)
    _cb_lib.fays_cb_register.argtypes = [c_void_p, c_size_t]
    _cb_lib.fays_cb_register.restype = c_int
    # 槽容量 = MAX_STEREO_BYTES（下方缓冲节定义；此处注册须紧跟
    # create，故按同一算式内联：1280×800×2×3 双目三通道上限）
    _cb_ret = _cb_lib.fays_cb_register(vikit_handle, 1280 * 800 * 2 * 3)
    if _cb_ret != 0:
        _log(f"[WARN] 回调注册失败 ret={_cb_ret}（SDK 不支持？），"
             f"回退轮询取帧")
        CB_BRIDGE = False
    else:
        _log("已注册双目帧回调（--cb-bridge：官方 GUI 同款取帧路径，"
             "SDK 装配完成后才回调）")

# ── 相机原始内参 K1/D1（SDK 标定优先，yaml 回退）──
# SDK 标定分辨率 ≠ 流分辨率（本机标定 640×400 而流 1280×800，fx 228.6
# ×2 = 457.2 与 calib.yaml 的 1280×800 值完全吻合）→ 内参等比缩放到
# 流分辨率；畸变系数是归一化量不缩放。必须在引擎创建前读取——
# 引擎 calib_path 指向由此合成的 yaml（机型自动适配）。
STREAM_W, STREAM_H = 1280, 800
K1 = None
D1 = None
K2 = None    # 右目内参（--stereo-view 并排显示用）
D2 = None
_cam1_tx = None     # cam1 T_cn_cnm1 平移 x（基线回退用）
_kd_src = None
_dev_calib = None   # 设备标定原始数据（合成引擎 yaml 用）
if _vikit_calib is not None:
    _calib = AtrakCalibrationParam()
    if _vikit_calib(vikit_handle, byref(_calib)) == 0:
        _ci = _calib.cameras.cameras[0].intrinsics
        _fx0, _fy0 = float(_ci.fx), float(_ci.fy)
        _cx0, _cy0 = float(_ci.cx), float(_ci.cy)
        _w0, _h0 = int(_ci.width), int(_ci.height)
        _cam1_tx = float(_calib.cameras.cameras[1].T_cn_cnm1.trans[0])
        # 归一化内参兜底（fx<10 视为归一化值 ×宽高）
        if 0 < _fx0 < 10 and _w0:
            _fx0, _fy0 = _fx0 * _w0, _fy0 * _h0
            _cx0, _cy0 = _cx0 * _w0, _cy0 * _h0
        elif _w0 and _w0 != STREAM_W:
            _sx = STREAM_W / _w0
            _sy = STREAM_H / _h0 if _h0 else _sx
            _fx0, _fy0 = _fx0 * _sx, _fy0 * _sy
            _cx0, _cy0 = _cx0 * _sx, _cy0 * _sy
        if 100 < _fx0 < 800:
            K1 = np.array([[_fx0, 0, _cx0],
                           [0, _fy0, _cy0],
                           [0, 0, 1]], np.float64)
            _dist = list(_ci.distortion)
            D1 = np.asarray(_dist[:4], np.float64)
            _kd_src = f"SDK标定(fx={_fx0:.1f} 缩放自{_w0}x{_h0}, " \
                      f"dist_model={_ci.distortion_model} " \
                      f"k1..k4={[f'{v:.5f}' for v in D1]})"
        # 右目：与左目同款归一化/等比缩放（--stereo-view 用）
        _ci2 = _calib.cameras.cameras[1].intrinsics
        _fx2, _fy2 = float(_ci2.fx), float(_ci2.fy)
        _cx2, _cy2 = float(_ci2.cx), float(_ci2.cy)
        _w2, _h2 = int(_ci2.width), int(_ci2.height)
        if 0 < _fx2 < 10 and _w2:
            _fx2, _fy2 = _fx2 * _w2, _fy2 * _h2
            _cx2, _cy2 = _cx2 * _w2, _cy2 * _h2
        elif _w2 and _w2 != STREAM_W:
            _sx = STREAM_W / _w2
            _sy = STREAM_H / _h2 if _h2 else _sx
            _fx2, _fy2 = _fx2 * _sx, _fy2 * _sy
            _cx2, _cy2 = _cx2 * _sx, _cy2 * _sy
        if 100 < _fx2 < 800:
            K2 = np.array([[_fx2, 0, _cx2],
                           [0, _fy2, _cy2],
                           [0, 0, 1]], np.float64)
            D2 = np.asarray(list(_ci2.distortion)[:4], np.float64)
        if K1 is not None and K2 is not None:
            # 原始设备数据（合成引擎 kalibr yaml 用；引用保活 _calib）
            _dev_calib = {
                "cam0": (_calib.cameras.cameras[0], K1, D1),
                "cam1": (_calib.cameras.cameras[1], K2, D2),
            }
if K1 is None:
    # yaml 回退（kalibr 格式 cam0/cam1；cam1 给 --stereo-view 用）
    try:
        _cam = None
        _in_tcn = False
        with open(CALIB_YAML, encoding="utf-8") as _f:
            for _line in _f:
                if _line.startswith("cam0:"):
                    _cam = 0
                elif _line.startswith("cam1:"):
                    _cam = 1
                elif _cam == 0 and "intrinsics:" in _line:
                    _v = json.loads(_line.split("intrinsics:", 1)[1].strip())
                    if len(_v) == 4:  # kalibr camchain: [fx, fy, cx, cy]
                        K1 = np.array([[_v[0], 0.0, _v[2]],
                                       [0.0, _v[1], _v[3]],
                                       [0.0, 0.0, 1.0]], np.float64)
                    else:             # 3×3 矩阵形式（vendor dump）
                        K1 = np.array(_v, np.float64).reshape(3, 3)
                elif _cam == 0 and "distortion_coeffs:" in _line:
                    D1 = np.asarray(json.loads(
                        _line.split("distortion_coeffs:", 1)[1].strip()),
                        np.float64)
                elif _cam == 1 and "intrinsics:" in _line:
                    _v = json.loads(_line.split("intrinsics:", 1)[1].strip())
                    if len(_v) == 4:  # kalibr camchain: [fx, fy, cx, cy]
                        K2 = np.array([[_v[0], 0.0, _v[2]],
                                       [0.0, _v[1], _v[3]],
                                       [0.0, 0.0, 1.0]], np.float64)
                    else:             # 3×3 矩阵形式（vendor dump）
                        K2 = np.array(_v, np.float64).reshape(3, 3)
                elif _cam == 1 and "distortion_coeffs:" in _line:
                    D2 = np.asarray(json.loads(
                        _line.split("distortion_coeffs:", 1)[1].strip()),
                        np.float64)
                elif _cam == 1 and "T_cn_cnm1:" in _line:
                    _in_tcn = True
                elif _in_tcn and _cam1_tx is None and _line.strip().startswith(
                        "- ["):
                    _cam1_tx = float(json.loads(_line.strip()[2:])[3])
                    _in_tcn = False
        if K1 is not None:
            _kd_src = f"yaml回退({CALIB_YAML} cam0)"
    except (OSError, ValueError):
        pass
if K1 is None or D1 is None or D1.size < 4:
    _log("[ERROR] 拿不到相机内参 K1/D1（SDK 标定与 yaml 回退均失败）")
    _vikit_destroy(vikit_handle)
    sys.exit(1)
_log(f"鱼眼内参: {_kd_src}")

# ═══════════════════════════════════════════════════════════════
#  引擎标定合成 + 深度配置运行时重写
#  设备标定可用 → 用设备数据（K/D + T_cn_cnm1/T_cn_imu，同 vendor
#  dump 格式）合成 kalibr yaml 指给引擎 → 任意机型（S80C/S80M）自动
#  适配，无需按机型准备标定文件；否则沿用 --calib-yaml/仓库静态 yaml。
#  深度配置副本改写 calib_path/model_path → 仓库内绝对路径，原文件
#  不动（与端口重写同款临时副本做法）。
# ═══════════════════════════════════════════════════════════════

ENGINE_CALIB = CALIB_YAML
if _dev_calib is not None:
    def _extr_rows(_ext):
        _m, _t = list(_ext.rot), list(_ext.trans)
        return [[_m[0], _m[1], _m[2], _t[0]],
                [_m[3], _m[4], _m[5], _t[1]],
                [_m[6], _m[7], _m[8], _t[2]],
                [0.0, 0.0, 0.0, 1.0]]

    def _write_cam(_f, _cam, _K, _D, _with_tcnm1):
        _f.write("  T_cam_imu:\n")
        for _row in _extr_rows(_cam.T_cn_imu):
            _f.write(f"  - {[float(_v) for _v in _row]}\n")
        if _with_tcnm1:
            _f.write("  T_cn_cnm1:\n")
            for _row in _extr_rows(_cam.T_cn_cnm1):
                _f.write(f"  - {[float(_v) for _v in _row]}\n")
        _f.write("  camera_model: pinhole\n")
        _f.write(f"  distortion_coeffs: {[float(_v) for _v in _D]}\n")
        # 本机两款相机（S80C/S80M）均为 equidistant（vendor dump 同款）
        _f.write("  distortion_model: equidistant\n")
        _f.write(f"  intrinsics: [{_K[0, 0]}, {_K[1, 1]}, "
                 f"{_K[0, 2]}, {_K[1, 2]}]\n")
        _f.write(f"  resolution: [{STREAM_W}, {STREAM_H}]\n")
        _f.write(f"  timeshift_cam_imu: {float(_cam.timeshift_cam_imu)}\n")

    try:
        _cf = tempfile.NamedTemporaryFile(
            mode="w", suffix="_engine_calib.yaml", delete=False)
        with _cf as _f:
            _f.write("cam0:\n")
            _write_cam(_f, _dev_calib["cam0"][0],
                       _dev_calib["cam0"][1], _dev_calib["cam0"][2], False)
            _f.write("cam1:\n")
            _write_cam(_f, _dev_calib["cam1"][0],
                       _dev_calib["cam1"][1], _dev_calib["cam1"][2], True)
        ENGINE_CALIB = _cf.name
        _fx_d = float(K1[0, 0])
        _model = ("S80C" if abs(_fx_d - 457.2) < 4.0 else
                  "S80M" if abs(_fx_d - 464.8) < 4.0 else
                  f"未知(fx={_fx_d:.1f})")
        _log(f"[Calib] 设备标定合成引擎 yaml（机型 {_model}）→ {ENGINE_CALIB}")
    except Exception as _e:
        _log(f"[Calib] 合成失败，引擎沿用静态标定 {CALIB_YAML}: {_e}")

_DEPTH_TMP = None
try:
    _tf = tempfile.NamedTemporaryFile(
        mode="w", suffix="_s80c_depth.yaml", delete=False)
    _MODEL_YAML = os.path.join(
        _THIRD_PARTY, "config", "models", "rk3588", "stereo_s_general.rknn")
    with open(DEPTH_CONFIG, encoding="utf-8") as _src:
        for _line in _src:
            if _line.startswith("calib_path"):
                _line = f"calib_path: {ENGINE_CALIB}\n"
            elif _line.startswith("model_path"):
                _line = f"model_path: {_MODEL_YAML}\n"
            _tf.write(_line)
    _tf.close()
    _DEPTH_TMP = _tf.name
    DEPTH_CONFIG = _DEPTH_TMP
    _log(f"[DepthCfg] 重写 calib_path → {ENGINE_CALIB}（仓库内）")
except Exception as _e:
    _log(f"[DepthCfg] 重写失败，沿用原配置: {_e}")

depth_handle = c_void_p()
ret = _d_create(byref(depth_handle), DEPTH_CONFIG.encode())
if ret != 0:
    _log(f"[ERROR] 深度引擎创建失败, ret={ret}（config: {DEPTH_CONFIG}）")
    _vikit_destroy(vikit_handle)
    sys.exit(1)
_log(f"深度引擎创建成功（CPU SGBM；本机实测深度更新 ~20fps，SDK 文档"
     f"标称 0.7s/帧为保守量级）")

ret = _d_bind(depth_handle, vikit_handle)
if ret != 0:
    _log(f"[ERROR] BindViKit 失败, ret={ret}")
    _d_destroy(depth_handle)
    _vikit_destroy(vikit_handle)
    sys.exit(1)

# ── 矫正矩阵 R0/P0（引擎返回；深度/矫正图空间由 P0 定义）──
rect_info = AtrakRectifyInfo()
if _d_rectinfo(depth_handle, byref(rect_info)) != 0:
    _log("[ERROR] GetRectifyInfo 失败")
    _d_destroy(depth_handle)
    _vikit_destroy(vikit_handle)
    sys.exit(1)
R0 = np.asarray(rect_info.R0, np.float64).reshape(3, 3)
P0 = np.asarray(rect_info.P0, np.float64).reshape(3, 4)
R1 = np.asarray(rect_info.R1, np.float64).reshape(3, 3)
P1 = np.asarray(rect_info.P1, np.float64).reshape(3, 4)
fx = float(P0[0, 0])

# 基线：P1[3] = ±fx'·Tx；实测引擎返回 P1[3]=0 → 回退标定外参 trans x
_p13 = float(P0[1, 3])
if abs(_p13) > 1e-6:
    baseline_mm = abs(_p13) / fx * 1000.0 if fx else 0.0
else:
    baseline_mm = abs(_cam1_tx) * 1000.0 if _cam1_tx else 0.0
_log(f"矫正矩阵: fx'={fx:.1f} cx={P0[0, 2]:.1f} cy={P0[1, 2]:.1f} "
     f"P1[3]={_p13:.3f} 基线'={baseline_mm:.1f}mm")

# ── 鱼眼矫正映射：raw 左目 → P0 矫正空间（2D 视图与深度同坐标系）──
_map1, _map2 = cv2.fisheye.initUndistortRectifyMap(
    K1, D1, R0, P0, (1280, 800), cv2.CV_32FC1)
_log(f"鱼眼矫正映射已建（remap raw→P0 空间，rect_mode={RECT_MODE}）")

# ── 右目矫正映射（--stereo-view 并排显示；R1/P1 为引擎右目矫正矩阵。
# P1[3]=0 的本机引擎 quirk 只影响视差偏移量，不影响显示）──
_map1r = _map2r = None
if STEREO_VIEW:
    if K2 is not None:
        _map1r, _map2r = cv2.fisheye.initUndistortRectifyMap(
            K2, D2, R1, P1, (1280, 800), cv2.CV_32FC1)
        _log("右目矫正映射已建（--stereo-view 并排显示）")
    else:
        _log("[WARN] --stereo-view 需要右目内参 K2/D2"
             "（SDK 标定 cameras[1] 或 yaml cam1），本次只发左目")

# ── 诊断导出：矫正矩阵+内参落盘后退出（离线复现 remap 用）──
if _args.dump_rect_info:
    import json as _json
    _rect_dump = {
        "R0": R0.tolist(), "P0": P0.tolist(),
        "R1": R1.tolist(), "P1": P1.tolist(),
        "K1": K1.tolist() if K1 is not None else None,
        "D1": D1.tolist() if D1 is not None else None,
        "K2": K2.tolist() if K2 is not None else None,
        "D2": D2.tolist() if D2 is not None else None,
        "fx": float(fx), "baseline_mm": float(baseline_mm),
        "stereo_view": bool(STEREO_VIEW),
    }
    try:
        with open(_args.dump_rect_info, "w") as _fo:
            _json.dump(_rect_dump, _fo, indent=1)
        _log(f"[RectDump] 矫正矩阵已写 {_args.dump_rect_info}")
    except OSError as _e:
        _log(f"[RectDump] 写盘失败: {_e}")
    # 与正常路径同款清理（防相机句柄残留）
    if CB_BRIDGE and _cb_lib is not None:
        _cb_lib.fays_cb_stop()
    _d_destroy(depth_handle)
    _vikit_destroy(vikit_handle)
    sys.exit(0)

# ═══════════════════════════════════════════════════════════════
#  预分配缓冲区（按 SDK 头文件宏上限：偏小会在特定时刻越界写堆）
# ═══════════════════════════════════════════════════════════════

FAYS_ATRAK_MONO_MAX_BYTES = 1280 * 800 * 2
FAYS_ATRAK_RGB_MAX_BYTES = 3840 * 2160 * 3
FAYS_ATRAK_IMG_MAX_BYTES = max(FAYS_ATRAK_MONO_MAX_BYTES,
                               FAYS_ATRAK_RGB_MAX_BYTES)
FAYS_ATRAK_DEPTH_IMG_MAX_BYTES = FAYS_ATRAK_IMG_MAX_BYTES * 4
MAX_STEREO_BYTES = FAYS_ATRAK_MONO_MAX_BYTES * 3
MAX_DEPTH_FLOATS = FAYS_ATRAK_DEPTH_IMG_MAX_BYTES // 4

_stereo_buf = (c_ubyte * MAX_STEREO_BYTES)()
_stereo_buf_b = (c_ubyte * MAX_STEREO_BYTES)() if DOUBLE_BUFFER else None
_race_buf = (c_ubyte * MAX_STEREO_BYTES)() if RACE_PROBE else None
if SETTLE_POLL:
    _settle_a = (c_ubyte * MAX_STEREO_BYTES)()
    _settle_b = (c_ubyte * MAX_STEREO_BYTES)()
else:
    _settle_a = _settle_b = None
# 深度 feed A/B 双缓冲：引擎异步读时单缓冲 memmove 覆盖嫌疑
# （主程序 read_stereo_rgb.py 同款；喂入指针交替，引擎读到的总是完整帧）
_feed_bufs = ((c_ubyte * MAX_STEREO_BYTES)(),
              (c_ubyte * MAX_STEREO_BYTES)())
_feed_alt = False
_depth_buf = (c_float * MAX_DEPTH_FLOATS)()
_rect_buf = (c_ubyte * MAX_STEREO_BYTES)()

_stereo_img = AtrakImage()
_stereo_img.data = cast(_stereo_buf, POINTER(c_ubyte))
_feed_img = AtrakImage()
_feed_img.data = cast(_feed_bufs[0], POINTER(c_ubyte))
_depth_out = AtrakDepthImage()
_depth_out.data = cast(_depth_buf, POINTER(c_float))
_rect_out = AtrakImage()
_rect_out.data = cast(_rect_buf, POINTER(c_ubyte))

_DISP_BUF = (c_float * MAX_DEPTH_FLOATS)()
_disp_out = AtrakDepthImage()
_disp_out.data = cast(_DISP_BUF, POINTER(c_float))
_PAIRED_BUF = (c_ubyte * FAYS_ATRAK_IMG_MAX_BYTES)()
_paired_out = AtrakImage()
_paired_out.data = cast(_PAIRED_BUF, POINTER(c_ubyte))

SWAP_RGB = True    # S80 固件实际输出 RGB 通道序（声称 BGR），统一转 BGR

# ── 握手：P0 内参 JSON（demo 据此建 align_calib）──
_meta = {
    "cam": "S80C",
    "P0": [float(v) for v in rect_info.P0],
    "width": 1280, "height": 800,
    "fx": float(P0[0, 0]), "fy": float(P0[1, 1]),
    "cx": float(P0[0, 2]), "cy": float(P0[1, 2]),
    "baseline_mm": round(baseline_mm, 2),
    "rect_mode": RECT_MODE,
    "stereo_view": bool(STEREO_VIEW and _map1r is not None),
}
_send(_META_TYPE, 0, 0, 0, 0, json.dumps(_meta).encode("utf-8"))
_log(f"握手已发: P0 fx={_meta['fx']:.2f} fy={_meta['fy']:.2f} "
     f"cx={_meta['cx']:.2f} cy={_meta['cy']:.2f}")

# ═══════════════════════════════════════════════════════════════
#  主循环：取流 → 喂引擎 → 取深度 → 矫正左目 → 管道输出
# ═══════════════════════════════════════════════════════════════

_latest_depth_mm = None    # 最近一张深度（引擎更新间隔内沿用）
_last_depth_ts = 0         # 新鲜深度判定（引擎 async 期间 GetDepthImage
                           # 会重复返回同一张结果）
_depth_send_pending = False   # 新深度等待发送（仅新帧发送，demo 侧自留）
_rgb_seq = 0
_depth_seq = 0
_frame_info_shown = False
_warmup_frames = 0
_stat_t0 = time.monotonic()
_stat_frames = 0
_stat_fresh = 0
_last_frame_t = time.monotonic()
# 撕裂诊断状态（flag 全关时零开销）
_raw_ring = deque(maxlen=RAW_RING) if RAW_DUMP_DIR else None
_sent_ring = deque(maxlen=RAW_RING) if RAW_DUMP_DIR else None   # post-remap 发送帧环
_dbuf_alt = False                    # 双缓冲 A/B 相位
_race_total = 0                      # 竞态探测：拷贝窗口统计（按秒清零）
_race_mismatch = 0
_cur_buf = _stereo_buf               # 当前取帧缓冲（双缓冲开时交替）

# 回调取帧（--cb-bridge）：shim poll 元数据出参（循环外分配零每帧开销）
_cb_ts = c_ulonglong(0)
_cb_w = c_int(0)
_cb_h = c_int(0)
_cb_ch = c_int(0)
_cb_enc = c_short(0)
_cb_step = c_int(0)
_cb_nbytes = c_uint(0)
_cb_seq = c_int(0)
if CB_BRIDGE:
    _cb_lib.fays_cb_poll.argtypes = [c_void_p, c_size_t, POINTER(c_ulonglong),
                                     POINTER(c_int), POINTER(c_int),
                                     POINTER(c_int), POINTER(c_short),
                                     POINTER(c_int), POINTER(c_uint),
                                     POINTER(c_int)]
    _cb_lib.fays_cb_poll.restype = c_int
    _cb_lib.fays_cb_stop.restype = None

_log("预热深度引擎（async 需连续喂 ≥5 帧后 GetDepthImage 才有数据）…")

def _export_raw_ring(subdir, label):
    """raw 环一次性 JPEG 导出（编码只在触发时做，不占实时预算）。
    默认半尺寸（0.5× INTER_AREA）+Q90；--raw-full 全尺寸（S80C 缝
    比 S80M 细，半尺寸机器检测可能漏）。subdir="" 写 RAW_DUMP_DIR
    根目录（退出导出），否则写子目录（t 键手动导出）。"""
    if _raw_ring is None or not RAW_DUMP_DIR:
        return
    try:
        _tdir = (os.path.join(RAW_DUMP_DIR, subdir) if subdir
                 else RAW_DUMP_DIR)
        os.makedirs(_tdir, exist_ok=True)
        _wrote = 0
        for _ts, _f in _raw_ring:
            if RAW_FULL:
                _img = _f
            else:
                _img = cv2.resize(_f, None, fx=0.5, fy=0.5,
                                  interpolation=cv2.INTER_AREA)
            _ok, _buf = cv2.imencode(
                ".jpg", _img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if _ok:
                with open(os.path.join(_tdir,
                                       f"raw_{_ts}.jpg"), "wb") as _fo:
                    _fo.write(_buf.tobytes())
                _wrote += 1
        _log(f"[raw-dump] {label} {_wrote}/{len(_raw_ring)} 帧 → {_tdir}")
    except Exception as _e:
        _log(f"[raw-dump] {label} 导出失败: {_e}")


def _export_sent_ring(subdir, label):
    """post-remap 发送帧环一次性 JPEG 导出（与 raw 环同尺寸同质量规则）。
    环存 rgb_view/right_view 引用——remap 每帧返回新数组、sdk 模式 .copy()，
    均无改写风险，零实时成本。raw 净而 sent 有缝=remap/引擎链；sent 也净=
    缝在 demo 合成/显示侧（raw 环参照为同一时刻同一帧）。"""
    if _sent_ring is None or not RAW_DUMP_DIR:
        return
    try:
        _tdir = (os.path.join(RAW_DUMP_DIR, subdir) if subdir
                 else RAW_DUMP_DIR)
        os.makedirs(_tdir, exist_ok=True)
        _wrote = 0
        for _ts, _L, _R in _sent_ring:
            for _tag, _f in (("sentL", _L), ("sentR", _R)):
                if _f is None:
                    continue
                if RAW_FULL:
                    _img = _f
                else:
                    _img = cv2.resize(_f, None, fx=0.5, fy=0.5,
                                      interpolation=cv2.INTER_AREA)
                _ok, _buf = cv2.imencode(
                    ".jpg", _img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if _ok:
                    with open(os.path.join(
                            _tdir, f"{_tag}_{_ts}.jpg"), "wb") as _fo:
                        _fo.write(_buf.tobytes())
                    _wrote += 1
        _log(f"[sent-dump] {label} {_wrote} 帧 → {_tdir}")
    except Exception as _e:
        _log(f"[sent-dump] {label} 导出失败: {_e}")


if RAW_DUMP_DIR:
    # SIGTERM 默认动作不跑 finally——装 handler 抛 SystemExit 保证退出时
    # raw 环落盘（demo 侧 close() 发 SIGTERM 并等 3s；SIGINT 自带
    # KeyboardInterrupt 走 finally 无需处理）
    def _sigterm_exit(_sig, _frm):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _sigterm_exit)

    _raw_dump_n = 0

    def _sigusr1_dump(_sig, _frm):
        # demo 侧 t 键导出 tear 环时同步触发：见撕立即按 t，把 raw 环
        # 与 tear 环按时刻配对导出（S80C 缝细+突发稀疏，靠退出窗口
        # 采样撞大运不可靠）。导出 ~2-3s 阻塞主循环，期间帧丢失可接受
        # （手动触发点，之后自动恢复最新帧）。
        global _raw_dump_n
        _raw_dump_n += 1
        _export_raw_ring(f"t_{_raw_dump_n:03d}", f"手动导出 #{_raw_dump_n}")
        _export_sent_ring(f"t_{_raw_dump_n:03d}", f"手动导出 #{_raw_dump_n}")
    signal.signal(signal.SIGUSR1, _sigusr1_dump)

try:
    while True:
        if CB_BRIDGE:
            # 回调取帧：shim 从环槽拷出 SDK 装配完成的帧（seqlock），
            # 帧由回调线程在装配完成后写入环——不经过 GetStereoFrames
            # 的内部拷贝，绕过其与装配的竞态（疑为水平缝根因）
            _got = _cb_lib.fays_cb_poll(
                _cur_buf, MAX_STEREO_BYTES,
                byref(_cb_ts), byref(_cb_w), byref(_cb_h), byref(_cb_ch),
                byref(_cb_enc), byref(_cb_step), byref(_cb_nbytes),
                byref(_cb_seq))
            if _got != 1:
                if time.monotonic() - _last_frame_t > 5.0:
                    _log("[ERROR] 5s 无相机帧，退出（相机掉线/端口被占？）")
                    sys.exit(2)
                time.sleep(0.002)
                continue
            _last_frame_t = time.monotonic()
            if _cb_ts.value <= _stereo_img.timestamp:
                # 迟到帧（USB 重传批，120ms 回跳整 3 帧）：跳过保时间戳
                # 单调（与主程序 read_stereo_rgb.py 回调路径同款）
                continue
            _stereo_img.width = _cb_w.value
            _stereo_img.height = _cb_h.value
            _stereo_img.channel = _cb_ch.value
            _stereo_img.encoding = _cb_enc.value
            _stereo_img.step = _cb_step.value
            _stereo_img.bytes = _cb_nbytes.value
            _stereo_img.timestamp = _cb_ts.value
            _stereo_img.seq = _cb_seq.value
            _stereo_img.device_id = 0
        else:
            if DOUBLE_BUFFER:
                # 交替 data 指针 A/B（实验：仅 --race-probe 出正信号后试。
                # SDK 黑盒风险——若其缓存 data 指针，写入会引向旧缓冲更糟）
                _cur_buf = _stereo_buf_b if _dbuf_alt else _stereo_buf
                _dbuf_alt = not _dbuf_alt
                _stereo_img.data = cast(_cur_buf, POINTER(c_ubyte))
            ret = _vikit_stereo(vikit_handle, byref(_stereo_img))
            if ret != 0:                 # -2 = VIKIT_RET_NO_NEW_FRAME
                if time.monotonic() - _last_frame_t > 5.0:
                    _log("[ERROR] 5s 无相机帧，退出（相机掉线/端口被占？）")
                    sys.exit(2)
                time.sleep(0.002)
                continue
            _last_frame_t = time.monotonic()

        w, h, ch = _stereo_img.width, _stereo_img.height, _stereo_img.channel
        mono_h = h // 2
        if not _frame_info_shown:
            _log(f"双目帧: {w}x{h} ch={ch} encoding={_stereo_img.encoding} "
                 f"bytes={_stereo_img.bytes}（上=左/下=右）")
            _frame_info_shown = True

        n = _stereo_img.bytes
        if SETTLE_POLL and not CB_BRIDGE:
            # 缓冲稳定轮询（修复候选，实验）：假设 SDK 在 GetStereoFrames
            # 返回后其写入线程可能尚未写完 _cur_buf——拷贝与写入按行交错
            # 会产出水平缝且缝行随相对时序漂移（与观察吻合）。连续两次
            # 快照一致才拷贝使用；缓冲本就稳定时仅多一次比对+~0.5ms。
            ctypes.memmove(_settle_a, _cur_buf, n)
            for _si in range(4):          # 最多 ~2-3ms 后放弃（不比
                time.sleep(0.0005)        # 现状更差，但不再空等）
                ctypes.memmove(_settle_b, _cur_buf, n)
                _va = np.ctypeslib.as_array(
                    (c_ubyte * n).from_address(ctypes.addressof(_settle_a)))
                _vb = np.ctypeslib.as_array(
                    (c_ubyte * n).from_address(ctypes.addressof(_settle_b)))
                if np.array_equal(_va, _vb):
                    break
                ctypes.memmove(_settle_a, _cur_buf, n)   # 未稳：更新基准
        full = np.ctypeslib.as_array(
            (c_ubyte * n).from_address(ctypes.addressof(_cur_buf)))
        if ch == 1:
            full = full.reshape(h, w)
        else:
            full = full.reshape(h, w, ch)
        frame_np = full.copy()
        if RACE_PROBE and not CB_BRIDGE:
            # 竞态探测：拷贝后立即二次 memcpy 比对——缓冲在两次拷贝
            # 窗口内被 SDK 改写则两拷贝不等（拷贝窗口内改动无论落在
            # 哪次拷贝都会造成差异，探测是保守且充分的）
            _race_total += 1
            _race_view = np.ctypeslib.as_array(
                (c_ubyte * n).from_address(ctypes.addressof(_race_buf)))
            ctypes.memmove(_race_buf, _cur_buf, n)
            if not np.array_equal(full, _race_view):
                _race_mismatch += 1
        if SWAP_RGB and ch == 3:
            frame_np = np.ascontiguousarray(frame_np[:, :, ::-1])
        if _raw_ring is not None:
            # pre-remap 原始帧环（存 frame_np 引用，零实时成本，~6MB×N；
            # JPEG 编码只在退出做——与 demo --tear-probe 的矫正帧同轮
            # 对比，判"缝在 raw 还是 remap 后"）
            _raw_ring.append((_stereo_img.timestamp, frame_np))

        # ── 送引擎（拷入 feed 缓冲；引擎按 BGR 约定处理输入）──
        if frame_np.nbytes <= MAX_STEREO_BYTES:
            _feed_alt = not _feed_alt
            _feed_img.data = cast(_feed_bufs[_feed_alt], POINTER(c_ubyte))
            ctypes.memmove(_feed_img.data, frame_np.ctypes.data,
                           frame_np.nbytes)
            _feed_img.width = _stereo_img.width
            _feed_img.height = _stereo_img.height
            _feed_img.channel = _stereo_img.channel
            _feed_img.encoding = _stereo_img.encoding
            _feed_img.step = _stereo_img.step
            _feed_img.bytes = _stereo_img.bytes
            _feed_img.timestamp = _stereo_img.timestamp
            _feed_img.seq = _stereo_img.seq
            _feed_img.device_id = _stereo_img.device_id
            _d_feed(depth_handle, byref(_feed_img))
            _warmup_frames += 1

        # ── 取深度/视差（失败=引擎还在处理/预热，沿用上一张）──
        for _out in (_depth_out, _disp_out, _paired_out):
            _out.width = _out.height = _out.channel = 0
            _out.step = _out.bytes = 0
        if _d_get(depth_handle, byref(_depth_out), byref(_disp_out),
                  byref(_paired_out)) == 0 and _depth_out.width > 0:
            _dn = _depth_out.bytes // 4
            _darr = np.ctypeslib.as_array(
                _depth_out.data, shape=(_dn,)).copy()
            depth_m = _darr.reshape(_depth_out.height, _depth_out.width)
            _latest_depth_mm = (depth_m * 1000.0).astype(np.float32)
            if _depth_out.timestamp != _last_depth_ts:
                _depth_seq += 1
                _last_depth_ts = _depth_out.timestamp
                _stat_fresh += 1
                _depth_send_pending = True   # 仅新帧发送；demo 侧自留最近深度

        # ── 2D 视图：raw 左目 remap 到 P0 矫正空间 ──
        if ch == 1:
            _left = frame_np[:mono_h]
            _left = cv2.cvtColor(_left, cv2.COLOR_GRAY2BGR)
        else:
            _left = frame_np[:mono_h]
        right_view = None
        if RECT_MODE == "remap":
            rgb_view = cv2.remap(_left, _map1, _map2,
                                 cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if _map1r is not None:
                if ch == 1:
                    _right = cv2.cvtColor(frame_np[mono_h:],
                                          cv2.COLOR_GRAY2BGR)
                else:
                    _right = frame_np[mono_h:]
                right_view = cv2.remap(_right, _map1r, _map2r,
                                       cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)
        else:   # sdk: 引擎矫正图左半（对照验证；尺寸可能非 1280×800）
            rgb_view = None
            _rect_out.width = _rect_out.height = _rect_out.channel = 0
            _rect_out.step = _rect_out.bytes = 0
            if _d_rect(depth_handle, byref(_rect_out)) == 0 \
                    and _rect_out.width > 0:
                _rn = _rect_out.bytes
                _rarr = np.ctypeslib.as_array(_rect_out.data,
                                              shape=(_rn,)).copy()
                _rch = _rect_out.channel if _rect_out.channel > 0 else 1
                _rect_np = _rarr.reshape(_rect_out.height, _rect_out.width,
                                         _rch)
                rgb_view = _rect_np[:_rect_np.shape[0] // 2].copy()
                if STEREO_VIEW:    # 引擎矫正图上下拼接，下半即右目
                    right_view = _rect_np[_rect_np.shape[0] // 2:].copy()
            if rgb_view is None:
                time.sleep(0.002)
                continue

        # ── 管道输出 ──
        # 默认 raw BGR（type=4/5，零压缩零伪影，MediaPipe 抖动最小）；
        # --pipe-format jpeg 时退 JPEG95（type=1/3，调试对照）。
        # 深度仅新帧发送（type=2）——demo 侧自留最近一张，50fps 下
        # 每帧重发 4MB 纯浪费带宽。
        _rgb_seq += 1
        _vh, _vw = rgb_view.shape[:2]
        try:
            if PIPE_FORMAT == "raw":
                if right_view is not None:   # 右目先发，demo 侧配对当前帧
                    _rh, _rw = right_view.shape[:2]
                    _send(_RAW_RIGHT_TYPE, _rgb_seq, _stereo_img.timestamp,
                          _rw, _rh, np.ascontiguousarray(right_view).tobytes())
                _send(_RAW_RGB_TYPE, _rgb_seq, _stereo_img.timestamp,
                      _vw, _vh, np.ascontiguousarray(rgb_view).tobytes())
            else:
                _ok, _jpg = cv2.imencode(".jpg", rgb_view,
                                         [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not _ok:
                    continue
                if right_view is not None:
                    _rh, _rw = right_view.shape[:2]
                    _, _rjpg = cv2.imencode(".jpg", right_view,
                                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                    _send(_RIGHT_TYPE, _rgb_seq, _stereo_img.timestamp,
                          _rw, _rh, _rjpg.tobytes())
                _send(_RGB_TYPE, _rgb_seq, _stereo_img.timestamp,
                      _vw, _vh, _jpg.tobytes())
            if _sent_ring is not None:
                # 发送帧环：与 raw 环同轮同刻（ts 相同），切开 raw→发送 之间
                _sent_ring.append(
                    (_stereo_img.timestamp, rgb_view, right_view))
            if _depth_send_pending and _latest_depth_mm is not None:
                _send(_DEPTH_TYPE, _depth_seq, _depth_out.timestamp,
                      _latest_depth_mm.shape[1], _latest_depth_mm.shape[0],
                      _latest_depth_mm.tobytes())
                _depth_send_pending = False
            # 无深度期间（预热）不发 type=2，demo 侧 2D-only
        except BrokenPipeError:
            _log("[Pipe] demo 侧已关闭，退出")
            break

        _stat_frames += 1
        _now = time.monotonic()
        if _now - _stat_t0 >= 1.0:
            _el = _now - _stat_t0
            _d = _latest_depth_mm
            if RACE_PROBE:
                _rline = (f" | 竞态 {_race_mismatch}/{_race_total} "
                          f"({_race_mismatch / max(_race_total, 1) * 100:.1f}%)")
                _race_mismatch = _race_total = 0
            else:
                _rline = ""
            if _d is not None:
                _valid = np.isfinite(_d) & (_d > 50.0)
                _vr = _valid.sum() / max(_valid.size, 1) * 100.0
                if _valid.any():
                    _med = float(np.median(_d[_valid]))
                    _log(f"[stat] 相机 {_stat_frames / _el:.1f}fps | 深度新 "
                         f"{_stat_fresh} 帧({_stat_fresh / _el:.2f}fps) | "
                         f"有效 {_vr:.0f}% 中位 {_med:.0f}mm 范围 "
                         f"{float(_d[_valid].min()):.0f}-"
                         f"{float(_d[_valid].max()):.0f}mm{_rline}")
                else:
                    _log(f"[stat] 相机 {_stat_frames / _el:.1f}fps | 深度新 "
                         f"{_stat_fresh} 帧 | 无有效深度{_rline}")
            else:
                _log(f"[stat] 相机 {_stat_frames / _el:.1f}fps | "
                     f"预热中({_warmup_frames} 帧)…{_rline}")
            _stat_t0 = _now
            _stat_frames = 0
            _stat_fresh = 0

except SystemExit:
    raise
except Exception as _e:
    _log(f"[ERROR] worker 异常: {type(_e).__name__}: {_e}")
    raise
finally:
    # 退出导出走同一函数（SIGTERM 时 demo close() 等 3s；160 帧半尺寸
    # ~1.5s 在其内，--raw-full 全尺寸会超时——用全尺寸时以 t 键导出为主，
    # 退出导出的尾部截断可接受）
    _export_raw_ring("", "退出导出")
    _export_sent_ring("", "退出导出")
    try:
        PIPE_FD.close()
    except Exception:
        pass
    if _PORTS_TMP and os.path.isfile(_PORTS_TMP):
        try:
            os.remove(_PORTS_TMP)
        except OSError:
            pass
    # 先深度后相机（依赖反序销毁）
    try:
        _d_destroy(depth_handle)
    except Exception:
        pass
    if CB_BRIDGE and _cb_lib is not None:
        # 回调桥接先停写（销毁竞态期 SDK 残留回调仍会触发，stop 后
        # 回调空转不写环，防悬垂写入）
        try:
            _cb_lib.fays_cb_stop()
        except Exception:
            pass
    try:
        _vikit_destroy(vikit_handle)
    except Exception:
        pass
    _log("worker 已退出, SDK 句柄已释放")

#!/usr/bin/env python3
"""
FaysSense S80M 双目 RGB 相机读取脚本
使用 FaysSense VI Kit SDK (C++ 库) 通过 ctypes 调用，实时显示左右目画面。

--depth-sdk-dir <dir>（可选）：额外运行 SDK 深度引擎（CPU SGBM，移植自
tools/hand_3d_s80c/s80c_depth_worker.py），管道协议尾部追加深度块。
"""

import ctypes
import os
import sys
import struct
import time
from ctypes import (
    c_void_p, c_char_p, c_int, c_float, c_double, c_size_t,
    c_ubyte, c_short, c_uint, c_uint8, c_uint32, c_ulonglong,
    POINTER, Structure, byref
)
# numpy/cv2 延迟到 OpenCV 预加载之后导入（深度模式：引擎绑定 4.2，见下）

# ── SDK 路径 ──────────────────────────────────────────────────
SDK_DIR = os.path.dirname(os.path.abspath(__file__))
# 架构自适应：x86_64 与 aarch64 包共用同一脚本（官方 SDK 两种架构
# 同名导出符号、同一结构体布局，ctypes 层完全可移植）
import platform
_LIB_ARCH = {"x86_64": "x86_64", "AMD64": "x86_64",
             "aarch64": "aarch64", "arm64": "aarch64",
             "armv7l": "armv7l"}.get(platform.machine(), platform.machine())
LIB_PATH = os.path.join(SDK_DIR, "lib", "fays_atrak", _LIB_ARCH,
                        "Release", "libfays_vikit.so")
CONFIG_PATH = os.path.join(SDK_DIR, "config", "fays_vikit.yaml")

# --config <path>：覆盖默认配置（主程序用 50fps 副本，25fps 原文件不动）
if "--config" in sys.argv:
    _cfg_idx = sys.argv.index("--config")
    if _cfg_idx + 1 < len(sys.argv):
        CONFIG_PATH = os.path.abspath(sys.argv[_cfg_idx + 1])

# --depth-sdk-dir <dir>：深度模式（主程序传 tools/hand_3d_s80c/third_party）。
# 开启后本进程额外运行 SDK 深度引擎（CPU SGBM，移植自
# s80c_depth_worker.py），并在管道协议尾部追加深度块（每帧必写，
# 引擎失败期恒写 0 长度——父进程解析确定性）；未给该参数时行为与
# 旧版逐字节一致（协议无深度块、零深度开销）。
DEPTH_SDK_DIR = None
if "--depth-sdk-dir" in sys.argv:
    _dd_idx = sys.argv.index("--depth-sdk-dir")
    if _dd_idx + 1 < len(sys.argv):
        DEPTH_SDK_DIR = os.path.abspath(sys.argv[_dd_idx + 1])
DEPTH_MODE = DEPTH_SDK_DIR is not None

# --cb-bridge：回调取帧（官方 GUI 同款消费方案，主程序默认开）。
# 轮询 GetStereoFrames 与深度引擎绑定叠加会在交付帧内留下水平错位带
#（8/31 主程序录制回归：25fps 档降缝率但不根治）；官方 GUI 用
# RegisterStereoImageCallback——SDK 装配完成后才回调，帧必完整且无撕裂。
# 注册 API 形参是 C++ std::function，ctypes 无法直调 → 经
# tools/hand_3d_s80c/third_party/cb_bridge/ 桥接（C++ shim 自持 4 槽环
# + seqlock，本脚本只 poll）。--cb-bridge-lib 覆盖默认路径；
# shim 缺失/加载/注册失败自动回退轮询（协议不变）。
CB_BRIDGE = "--cb-bridge" in sys.argv
CB_BRIDGE_LIB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hand_3d_s80c", "third_party", "cb_bridge", "libfays_cb_bridge.so")
if "--cb-bridge-lib" in sys.argv:
    _cb_idx = sys.argv.index("--cb-bridge-lib")
    if _cb_idx + 1 < len(sys.argv):
        CB_BRIDGE_LIB = os.path.abspath(sys.argv[_cb_idx + 1])

# ── LD_LIBRARY_PATH 自愈（深度模式）─────────────────────────────
# 深度引擎绑 OpenCV 4.2（third_party/opencv4.2/lib406），依赖链用裸
# SONAME（libopencv_*.so.4.2、libwebp.so.6…）必须靠 lib406 在动态链接
# 器搜索路径上解析；LD_LIBRARY_PATH 只在 exec 时读取 → 缺失时 re-exec
# 自身（fd 跨 exec 保留，二进制协议不受影响；主程序 spawn 已注入时
# 该段跳过）。必须在端口解析/预加载之前。
if DEPTH_MODE:
    _OV406 = os.path.join(DEPTH_SDK_DIR, "opencv4.2", "lib406")
    if os.path.isdir(_OV406) and \
            _OV406 not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        print(f"[Env] 预置 LD_LIBRARY_PATH: {_OV406}，re-exec …",
              file=sys.stderr)
        os.environ["LD_LIBRARY_PATH"] = _OV406 + ":" + \
            os.environ.get("LD_LIBRARY_PATH", "")
        os.execv(sys.executable, [sys.executable] + sys.argv)

# ── 模式选择 / stdout 重定向（必须在一切输出之前）───────────────
# --pipe - 模式下 fd 1 只允许出现二进制帧字节：任何 Python print 或
# C++ SDK printf 进入管道都会让父进程解析错位。旧代码把重定向放在
# SDK 初始化之后，靠「Python/C stdio 缓冲未满、dup2 后再 flush 到
# stderr」的运气保持管道干净——深度引擎初始化打印标定 dump（>8KB）
# 撑爆缓冲，会在重定向前 flush 进管道，父进程读到 "Fays" 帧头
# （0x46617973）后流错位、imdecode 抛异常（实测 0 帧断连）。
# 修复：重定向提前到最早位置（任何库加载/打印/SDK 调用之前），
# Python print 走 stderr 对象、C++ printf 走 fd 1=stderr。
# 注意：必须在 re-exec 之后执行——execv 保留 fd，若先 dup2 再 exec，
# 新进程 dup(1) 拿到的是 stderr 副本，帧会写进 stderr。
PIPE_MODE = "--pipe" in sys.argv
if PIPE_MODE:
    pipe_idx = sys.argv.index("--pipe")
    PIPE_PATH = sys.argv[pipe_idx + 1]
    if PIPE_PATH == "-":
        # OS 级重定向：fd 1 → stderr（拦截 C++ SDK 的 printf/cout）
        # 保存原始 stdout 用于二进制帧传输
        _pipe_fd = os.dup(1)
        os.dup2(2, 1)               # fd 1 现在指向 stderr
        sys.stdout = sys.stderr     # Python print 也走 stderr
        PIPE_FD = os.fdopen(_pipe_fd, 'wb', buffering=0)  # 无缓冲二进制写
    else:
        PIPE_FD = open(PIPE_PATH, "wb")
else:
    PIPE_FD = None

# ── 设备节点自动解析 ──────────────────────────────────────────
# yaml 里写死的 /dev/videoN 会随 USB 插拔顺序漂移：DECXIN/RealSense 先
# 枚举时，S80C 的 FTDI 桥被挤到 video2+，SDK 会拿别的相机当双目
# （现象：RgbInit failed + 主程序永远等不到帧）。
# 启动时按设备名（FTDI Superspeed Video Bridge）与 USB 接口号重写临时
# yaml（只替换三行端口，其余内容原样保留，SDK 解析不受影响）：
#   接口 1.0 = 双目对（取第一个节点）
#   接口 1.2 = IMU / 中置 RGB（取第一个节点）
#   rgb_dev_port 一律置 NULL 禁用（主程序不用中置 RGB，且 RGB 初始化
#   失败有崩溃史——3.9.1 会段错误）
# 解析失败 → 保持原 yaml 不动（旧行为）。
_PORTS_TMP = None
try:
    import re as _re
    import tempfile as _tempfile
    _ports_nodes = {}
    for _name in sorted(os.listdir("/sys/class/video4linux")):
        _vp = os.path.join("/sys/class/video4linux", _name)
        try:
            _dname = open(os.path.join(_vp, "name"), encoding="utf-8").read().strip()
        except OSError:
            continue
        if "FTDI Superspeed Video Bridge" not in _dname:
            continue
        try:
            _iface = os.path.basename(os.readlink(os.path.join(_vp, "device")))
        except OSError:
            continue
        _m = _re.match(r".+:(\d+\.\d+)$", _iface)
        if not _m:
            continue
        _ports_nodes.setdefault(_m.group(1), []).append(f"/dev/{_name}")
    _stereo_port = next(iter(_ports_nodes.get("1.0", [])), None)
    _imu_port = next(iter(_ports_nodes.get("1.2", [])), None)
    if _stereo_port:
        _tf = _tempfile.NamedTemporaryFile(
            mode="w", suffix="_s80m_ports.yaml", delete=False)
        with open(CONFIG_PATH, encoding="utf-8") as _src:
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
        CONFIG_PATH = _PORTS_TMP
        print(f"[Ports] 自动解析: stereo={_stereo_port} "
              f"imu={_imu_port or '?'} rgb=NULL", file=sys.stderr)
except Exception as _e:
    print(f"[Ports] 解析失败，沿用 yaml 端口: {_e}", file=sys.stderr)

# 预加载 OpenCV：3.9.0 的 libfays_vikit.so 依赖 cv::fastFree / cv::cvtColor
# 等符号；SDK ≥3.9.1（如官方 aarch64 包）已清理该依赖，搜索路径里没有
# OpenCV 4.2 时跳过即可，属正常（不能因缺库整体崩掉）
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

# 仓库自带 lib406（third_party）——两种模式共用，git clone 新机无系统
# OpenCV 也能满足 vikit 的 cv::* 符号依赖
_REPO406 = os.path.join(os.path.dirname(os.path.dirname(SDK_DIR)),
                        "hand_3d_s80c", "third_party", "opencv4.2", "lib406")
if DEPTH_MODE:
    # 深度引擎按 SDK 自带 OpenCV 4.2.0 编译（cv::stereoRectify 传
    # MatExpr/STD_ARRAY，系统 4.6+ 的 getMat_ 不支持会崩）→ 必须先把
    # lib406 的依赖 shims 与 13 个 libopencv_*.so 全量 RTLD_GLOBAL 预载，
    # 且早于 import cv2（venv cv2 走自身 rpath 绑自家 OpenCV，互不干扰）
    _OV406 = os.path.join(DEPTH_SDK_DIR, "opencv4.2", "lib406")
    if os.path.isdir(_OV406):
        _preload_opencv_dir(_OV406)
    else:
        print(f"[WARN] lib406 目录不存在: {_OV406}，深度引擎可能崩溃",
              file=sys.stderr)
elif os.path.isdir(_REPO406):
    # 纯 RGB 模式同样先预载仓库 lib406（3.9.0 vikit 依赖 cv::fastFree /
    # cv::cvtColor 等符号，新机器无系统 OpenCV 时裸名预载失败 → vikit
    # 加载必崩）；仓库缺失才回退搜索路径裸名（如本机 apt OpenCV 4.6）
    _preload_opencv_dir(_REPO406)
else:
    for _lib in ["libopencv_core", "libopencv_imgproc", "libopencv_imgcodecs", "libopencv_highgui"]:
        try:
            ctypes.CDLL(_lib + ".so.406", mode=ctypes.RTLD_GLOBAL)
        except OSError:
            try:
                ctypes.CDLL(_lib + ".so.4.2", mode=ctypes.RTLD_GLOBAL)
            except OSError:
                print(f"[Preload] {_lib} 不在搜索路径，跳过"
                      f"（SDK ≥3.9.1 无 OpenCV 依赖，属正常）", file=sys.stderr)

import numpy as np
import cv2

# cv2 5.0.0 wheel 自带 Qt5，必须用其自带插件（与系统 Qt5 ABI 不匹配）
# 注意: platforms 目录曾被改名 platforms.bak 以隔离插件冲突
_OV_PLUGINS_DIR = os.path.join(os.path.dirname(cv2.__file__), "qt", "plugins")
_OV_PLATFORMS = os.path.join(_OV_PLUGINS_DIR, "platforms.bak")
if not os.path.isdir(_OV_PLATFORMS):
    _OV_PLATFORMS = os.path.join(_OV_PLUGINS_DIR, "platforms")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _OV_PLATFORMS
os.environ["QT_QPA_PLATFORM"] = "xcb"

if not os.path.exists(LIB_PATH):
    print(f"[ERROR] 找不到 SDK 库: {LIB_PATH}")
    sys.exit(1)
if not os.path.exists(CONFIG_PATH):
    print(f"[ERROR] 找不到配置文件: {CONFIG_PATH}")
    sys.exit(1)

# ── 结构体定义 (对应 fays_atrak_types.h) ──────────────────────
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

class AtrakIMU(Structure):
    """IMU 样本: 纳秒时间戳 + 三轴陀螺仪 + 三轴加速度 (fays_atrak_types.h)"""
    _fields_ = [
        ("timestamp",  c_ulonglong),
        ("gyro",       c_double * 3),
        ("acc",        c_double * 3),
    ]

# ── 深度引擎结构体（DEPTH_MODE；fays_atrak_types.h，同
#    s80c_depth_worker.py 布局） ──────────────────────────────
if DEPTH_MODE:
    FAYS_ATRAK_MAX_CAMERAS = 3

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

# ── 加载动态库 ─────────────────────────────────────────────────
# libft602.so（FT602 USB 桥驱动库）是 vikit 的 DT_NEEDED；官方安装器把它
# 装进 /lib，新机器 git clone 后没有它则 vikit 加载必失败。仓库随带副本，
# 加载 vikit 前按绝对路径预载（RTLD_GLOBAL）——vikit 解析 NEEDED 时命中
# 已载副本（SONAME 未版本化），不再依赖系统目录。
def _preload_ft602(_vikit_path):
    _ft = os.path.join(os.path.dirname(_vikit_path), "libft602.so")
    if os.path.isfile(_ft):
        try:
            ctypes.CDLL(_ft, mode=ctypes.RTLD_GLOBAL)
            return True
        except OSError as _e:
            print(f"[WARN] libft602 预载失败: {_e}", file=sys.stderr)
    return False

# 深度模式优先加载 third_party 自带 vikit 且 RTLD_GLOBAL（深度库 dlopen
# 时要解析 FAYS_VIK_* 符号；该库同样导出 SetStereoExposure 与
# GetCalibrationParam，曝光/标定能力不丢）；失败回退本地库（纯 RGB）。
lib = None
if DEPTH_MODE:
    _d_vikit = os.path.join(DEPTH_SDK_DIR, "lib", "libfays_vikit.so")
    if os.path.isfile(_d_vikit):
        _preload_ft602(_d_vikit)
        try:
            lib = ctypes.CDLL(_d_vikit, mode=ctypes.RTLD_GLOBAL)
            print(f"[Depth] vikit 采用 third_party 副本: {_d_vikit}",
                  file=sys.stderr)
        except OSError as _e:
            print(f"[Depth] third_party vikit 加载失败: {_e}", file=sys.stderr)
if lib is None:
    _preload_ft602(LIB_PATH)
    lib = ctypes.CDLL(LIB_PATH)

# C++ mangled 函数符号 (对应 fays_vikit.h 中的 extern "C" 接口)
_fn_create  = getattr(lib, "_Z31FAYS_VIK_CreateHandleWithConfigPPvPKc")
_fn_destroy = getattr(lib, "_Z22FAYS_VIK_DestroyHandlePv")
_fn_stereo  = getattr(lib, "_Z24FAYS_VIK_GetStereoFramesPvP10AtrakImage")
_fn_imu     = getattr(lib, "_Z19FAYS_VIK_GetImuDataPvP8AtrakIMU")
_fn_version = getattr(lib, "_Z19FAYS_VIK_GetVersionPv")

# SIGTERM/SIGINT 优雅退出: 走 finally → _fn_destroy 释放 SDK,
# 否则被 UI terminate() 强杀时 FT602 设备留在坏状态, 后续进程初始化崩溃
import signal
def _graceful_exit(sig, frame):
    raise SystemExit(0)
signal.signal(signal.SIGTERM, _graceful_exit)
signal.signal(signal.SIGINT, _graceful_exit)

_fn_create.argtypes  = [POINTER(c_void_p), c_char_p]
_fn_create.restype   = c_int
_fn_destroy.argtypes = [c_void_p]
_fn_destroy.restype  = c_int
_fn_stereo.argtypes  = [c_void_p, POINTER(AtrakImage)]
_fn_stereo.restype   = c_int
_fn_imu.argtypes     = [c_void_p, POINTER(AtrakIMU)]
_fn_imu.restype      = c_int
_fn_version.argtypes = [c_void_p]
_fn_version.restype  = c_char_p

# 标定参数（深度模式：合成引擎 kalibr yaml 用；符号缺失回退静态 yaml）
_fn_calib = None
if DEPTH_MODE:
    try:
        _fn_calib = getattr(
            lib, "_Z28FAYS_VIK_GetCalibrationParamPvP21AtrakCalibrationParam")
        _fn_calib.argtypes = [c_void_p, POINTER(AtrakCalibrationParam)]
        _fn_calib.restype = c_int
    except AttributeError:
        _fn_calib = None

# ── 初始化 SDK ─────────────────────────────────────────────────
handle = c_void_p()
ret = _fn_create(byref(handle), CONFIG_PATH.encode())
if ret != 0:
    print(f"[ERROR] SDK 初始化失败, 返回码: {ret}")
    sys.exit(1)

version = _fn_version(handle)
print(f"Fays VIKit Version: {version.decode() if version else 'unknown'}")

# ── 回调取帧桥接（--cb-bridge）─────────────────────────────────
# 注册须紧跟 create、在深度引擎 bind 之前（官方 GUI 同款时序，
# main_window.cpp 同位置）。回退轮询时协议与旧版逐字节一致。
_cb_lib = None
if CB_BRIDGE:
    if not os.path.isfile(CB_BRIDGE_LIB):
        print(f"[cb-bridge] 未找到 {CB_BRIDGE_LIB}，回退轮询取帧",
              file=sys.stderr)
        CB_BRIDGE = False
    else:
        try:
            _cb_lib = ctypes.CDLL(CB_BRIDGE_LIB)
            _cb_lib.fays_cb_register.argtypes = [c_void_p, c_size_t]
            _cb_lib.fays_cb_register.restype = c_int
            # 槽容量 = 双目拼接帧上限（1280×800×2×3，与下方 img_buf 同算式）
            _cb_ret = _cb_lib.fays_cb_register(handle, 1280 * 800 * 2 * 3)
        except OSError as _e:
            _cb_ret = -3
            print(f"[cb-bridge] 加载失败: {_e}，回退轮询取帧",
                  file=sys.stderr)
        if _cb_ret != 0:
            print(f"[cb-bridge] 注册失败 ret={_cb_ret}，回退轮询取帧",
                  file=sys.stderr)
            CB_BRIDGE = False
            _cb_lib = None
        else:
            print("[cb-bridge] 已注册双目帧回调（官方 GUI 同款：SDK 装配"
                  "完成后才回调，帧必完整）", file=sys.stderr)

# ── SDK 深度引擎初始化（DEPTH_MODE；任一环节失败 → 打 stderr 停用，
#    继续纯 RGB；深度块仍每帧写 0 长度保证父进程解析确定性）──────
# 移植自 tools/hand_3d_s80c/s80c_depth_worker.py（已实机验证）：
# 结构体布局/标定合成/缓冲区尺寸必须与 worker 保持一致。
_depth_active = False      # 引擎可用（每帧 feed/get）
_depth_broken = False      # 运行期异常 → 停用
_depth_handle = None
_d_feed = _d_get = _d_destroy = None
_feed_buf = _feed_img = None
_depth_out = _disp_out = _paired_out = None
_MAX_STEREO_BYTES = 1280 * 800 * 2 * 3
_last_depth_ts = 0
_depth_payload = b""
_depth_w = _depth_h = 0
_depth_ts = 0
_ENGINE_CALIB_TMP = None
_DEPTH_TMP = None
if DEPTH_MODE:
    import json
    import tempfile
    try:
        _DEPTH_LIB = os.path.join(DEPTH_SDK_DIR, "lib",
                                  "libfayssense_aikit_depth.so")
        _DEPTH_CFG_SRC = os.path.join(DEPTH_SDK_DIR, "config",
                                      "stereo_depth.yaml")
        _MODEL_YAML = os.path.join(DEPTH_SDK_DIR, "config", "models",
                                   "rk3588", "stereo_s_general.rknn")
        _CALIB_YAML = os.path.join(
            os.path.dirname(os.path.dirname(SDK_DIR)),
            "tools", "fayssense_depth_sdk", "calib", "calib.yaml")
        if not (os.path.isfile(_DEPTH_LIB) and os.path.isfile(_DEPTH_CFG_SRC)
                and os.path.isfile(_MODEL_YAML)):
            raise RuntimeError("third_party 深度库/配置/model 缺失")

        _depth = ctypes.CDLL(_DEPTH_LIB)
        _d_create = _depth.FAYS_ATRAK_D_CreateHandleWithConfig
        _d_destroy = _depth.FAYS_ATRAK_D_DestroyHandle
        _d_bind = _depth.FAYS_ATRAK_D_BindViKit
        _d_feed = _depth.FAYS_ATRAK_D_FeedStereoImage
        _d_get = _depth.FAYS_ATRAK_D_GetDepthImage
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

        # ── 相机原始内参 K1/D1（SDK 标定优先，yaml 回退）──
        # SDK 标定分辨率 ≠ 流分辨率（本机标定 640×400 而流 1280×800）
        # → 内参等比缩放到流分辨率；畸变系数是归一化量不缩放。必须在
        # 引擎创建前读取——引擎 calib_path 指向由此合成的 yaml。
        STREAM_W, STREAM_H = 1280, 800
        K1 = K2 = None
        D1 = D2 = None
        _dev_calib = None   # 设备标定原始数据（合成引擎 yaml 用）
        if _fn_calib is not None:
            _calib = AtrakCalibrationParam()
            if _fn_calib(handle, byref(_calib)) == 0:
                def _k_from_ci(_ci):
                    _fx, _fy = float(_ci.fx), float(_ci.fy)
                    _cx, _cy = float(_ci.cx), float(_ci.cy)
                    _w, _h = int(_ci.width), int(_ci.height)
                    # 归一化内参兜底（fx<10 视为归一化值 ×宽高）
                    if 0 < _fx < 10 and _w:
                        _fx, _fy = _fx * _w, _fy * _h
                        _cx, _cy = _cx * _w, _cy * _h
                    elif _w and _w != STREAM_W:
                        _sx = STREAM_W / _w
                        _sy = STREAM_H / _h if _h else _sx
                        _fx, _fy = _fx * _sx, _fy * _sy
                        _cx, _cy = _cx * _sx, _cy * _sy
                    if not 100 < _fx < 800:
                        return None, None
                    _K = np.array([[_fx, 0, _cx],
                                   [0, _fy, _cy],
                                   [0, 0, 1]], np.float64)
                    _D = np.asarray(list(_ci.distortion)[:4], np.float64)
                    return _K, _D
                K1, D1 = _k_from_ci(_calib.cameras.cameras[0].intrinsics)
                K2, D2 = _k_from_ci(_calib.cameras.cameras[1].intrinsics)
                if K1 is not None and K2 is not None:
                    _dev_calib = {
                        "cam0": (_calib.cameras.cameras[0], K1, D1),
                        "cam1": (_calib.cameras.cameras[1], K2, D2),
                    }
        if K1 is None:
            # yaml 回退（kalibr 格式 cam0/cam1；len==4 为 [fx,fy,cx,cy]，
            # 否则 3×3 矩阵形式——reshape(3,3) 对 4 元素永远 ValueError）
            try:
                _cam = None
                with open(_CALIB_YAML, encoding="utf-8") as _f:
                    for _line in _f:
                        if _line.startswith("cam0:"):
                            _cam = 0
                        elif _line.startswith("cam1:"):
                            _cam = 1
                        elif _cam == 0 and "intrinsics:" in _line:
                            _v = json.loads(_line.split("intrinsics:", 1)[1]
                                            .strip())
                            if len(_v) == 4:
                                K1 = np.array([[_v[0], 0.0, _v[2]],
                                               [0.0, _v[1], _v[3]],
                                               [0.0, 0.0, 1.0]], np.float64)
                            else:
                                K1 = np.array(_v, np.float64).reshape(3, 3)
                        elif _cam == 0 and "distortion_coeffs:" in _line:
                            D1 = np.asarray(json.loads(
                                _line.split("distortion_coeffs:", 1)[1]
                                .strip()), np.float64)
                        elif _cam == 1 and "intrinsics:" in _line:
                            _v = json.loads(_line.split("intrinsics:", 1)[1]
                                            .strip())
                            if len(_v) == 4:
                                K2 = np.array([[_v[0], 0.0, _v[2]],
                                               [0.0, _v[1], _v[3]],
                                               [0.0, 0.0, 1.0]], np.float64)
                            else:
                                K2 = np.array(_v, np.float64).reshape(3, 3)
                        elif _cam == 1 and "distortion_coeffs:" in _line:
                            D2 = np.asarray(json.loads(
                                _line.split("distortion_coeffs:", 1)[1]
                                .strip()), np.float64)
            except (OSError, ValueError):
                pass
        if K1 is None or D1 is None or D1.size < 4:
            raise RuntimeError("拿不到相机内参 K1/D1（SDK 标定与 yaml 回退均失败）")

        # ── 引擎标定合成（设备标定 → vendor dump 同款 kalibr yaml；
        #     任意机型自动适配）──
        ENGINE_CALIB = _CALIB_YAML
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
                # 本机两款相机（S80C/S80M）均为 equidistant
                _f.write("  distortion_model: equidistant\n")
                _f.write(f"  intrinsics: [{_K[0, 0]}, {_K[1, 1]}, "
                         f"{_K[0, 2]}, {_K[1, 2]}]\n")
                _f.write(f"  resolution: [{STREAM_W}, {STREAM_H}]\n")
                _f.write(f"  timeshift_cam_imu: "
                         f"{float(_cam.timeshift_cam_imu)}\n")

            try:
                _cf = tempfile.NamedTemporaryFile(
                    mode="w", suffix="_engine_calib.yaml", delete=False)
                with _cf as _f:
                    _f.write("cam0:\n")
                    _write_cam(_f, _dev_calib["cam0"][0],
                               _dev_calib["cam0"][1],
                               _dev_calib["cam0"][2], False)
                    _f.write("cam1:\n")
                    _write_cam(_f, _dev_calib["cam1"][0],
                               _dev_calib["cam1"][1],
                               _dev_calib["cam1"][2], True)
                ENGINE_CALIB = _cf.name
                _ENGINE_CALIB_TMP = _cf.name
                _fx_d = float(K1[0, 0])
                _model = ("S80C" if abs(_fx_d - 457.2) < 4.0 else
                          "S80M" if abs(_fx_d - 464.8) < 4.0 else
                          f"未知(fx={_fx_d:.1f})")
                print(f"[Depth] 设备标定合成引擎 yaml（机型 {_model}）",
                      file=sys.stderr)
            except Exception as _e:
                print(f"[Depth] 合成失败，引擎沿用静态标定: {_e}",
                      file=sys.stderr)

        # ── 深度配置副本改写 calib_path/model_path（原文件不动）──
        _tf = tempfile.NamedTemporaryFile(
            mode="w", suffix="_s80c_depth.yaml", delete=False)
        with open(_DEPTH_CFG_SRC, encoding="utf-8") as _src:
            for _line in _src:
                if _line.startswith("calib_path"):
                    _line = f"calib_path: {ENGINE_CALIB}\n"
                elif _line.startswith("model_path"):
                    _line = f"model_path: {_MODEL_YAML}\n"
                _tf.write(_line)
        _tf.close()
        _DEPTH_TMP = _tf.name
        _depth_cfg = _DEPTH_TMP

        depth_handle = c_void_p()
        ret = _d_create(byref(depth_handle), _depth_cfg.encode())
        if ret != 0:
            raise RuntimeError(f"深度引擎创建失败, ret={ret}")
        _depth_handle = depth_handle
        ret = _d_bind(_depth_handle, handle)
        if ret != 0:
            _d_destroy(_depth_handle)
            _depth_handle = None
            raise RuntimeError(f"BindViKit 失败, ret={ret}")

        # ── 预分配缓冲区（按 SDK 头文件宏上限：偏小会在特定时刻
        #    越界写堆；feed 6MB / depth 95MB / disp 95MB / paired 24MB）──
        _IMG_MAX = 3840 * 2160 * 3
        _DEPTH_MAX_BYTES = _IMG_MAX * 4
        _MAX_DEPTH_FLOATS = _DEPTH_MAX_BYTES // 4
        # 双缓冲交替喂引擎：引擎内部异步处理时，单缓冲每帧 memmove 会
        # 覆盖其仍在读的上一帧输入（8/31 深度热力图撕裂远重于 RGB 的
        # 嫌疑之一）——A/B 交替给足一整帧窗口（官方 GUI C++ 侧 feed 每次
        # 内部拷贝为同款安全性）。
        _feed_bufs = ((c_ubyte * _MAX_STEREO_BYTES)(),
                      (c_ubyte * _MAX_STEREO_BYTES)())
        _feed_alt = False
        # 喂入节拍：隔帧喂（≈25fps，见主循环深度块注释）
        _feed_tick = False
        _feed_img = AtrakImage()
        _feed_img.data = ctypes.cast(_feed_bufs[0], POINTER(c_ubyte))
        _depth_buf = (c_float * _MAX_DEPTH_FLOATS)()
        _depth_out = AtrakDepthImage()
        _depth_out.data = ctypes.cast(_depth_buf, POINTER(c_float))
        _DISP_BUF = (c_float * _MAX_DEPTH_FLOATS)()
        _disp_out = AtrakDepthImage()
        _disp_out.data = ctypes.cast(_DISP_BUF, POINTER(c_float))
        _PAIRED_BUF = (c_ubyte * _IMG_MAX)()
        _paired_out = AtrakImage()
        _paired_out.data = ctypes.cast(_PAIRED_BUF, POINTER(c_ubyte))

        # 深度保持引擎原生 P0 矫正空间输出（1280×800，与 s80c_depth_worker
        # 的深度窗口同几何）。不做对齐到原始鱼眼 RGB 的 remap：原始画面
        # 约 45% 像素在矫正视场外（鱼眼 158° vs 矫正 107°），对齐会留一圈
        # 永不更新的死区，用户已确认改回原生矫正几何（满幅实时，demo 观感）。

        _depth_active = True
        print("[Depth] 深度引擎就绪（CPU SGBM；预热 ≥5 帧后出深度，"
              "实测深度更新 ~20fps）", file=sys.stderr)
    except Exception as _e:
        print(f"[Depth] 深度引擎初始化失败，本进程继续纯 RGB: {_e}",
              file=sys.stderr)

# ── 曝光控制通道（stdin 行协议）───────────────────────────────
# 主程序通过 stdin 下发 "SET_EXPOSURE <float>"：
#   -1.0 = 自动曝光；1.0~885.0 = 手动曝光值（与 yaml stereo_init_exposure
#   同单位同量程，SDK 运行时生效，无需重启）。stdin 关闭/EOF 时线程退出。
# 符号缺失（旧库）时静默降级为无曝光控制。
try:
    _fn_set_exposure = getattr(lib, "_Z26FAYS_VIK_SetStereoExposurePvd")
    _fn_set_exposure.argtypes = [c_void_p, c_double]
    _fn_set_exposure.restype = c_int
except AttributeError:
    _fn_set_exposure = None

def _stdin_control():
    import threading
    if _fn_set_exposure is None:
        return
    try:
        for line in sys.stdin.buffer:
            line = line.decode("utf-8", "ignore").strip()
            if line.startswith("SET_EXPOSURE "):
                try:
                    _fn_set_exposure(handle, float(line.split(None, 1)[1]))
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass

import threading
threading.Thread(target=_stdin_control, daemon=True).start()

# ── 预分配图像缓冲区 ───────────────────────────────────────────
# 双目拼接图最大: 1280*800*2 (mono) 或 1280*800*2*3 (RGB color)
MAX_BYTES = 1280 * 800 * 2 * 3
img_buf = (c_ubyte * MAX_BYTES)()
img = AtrakImage()
img.data = ctypes.cast(img_buf, POINTER(c_ubyte))

# 回调取帧出参（循环外一次性分配，零每帧开销；shim 元数据出参）
if CB_BRIDGE and _cb_lib is not None:
    _cb_lib.fays_cb_poll.argtypes = [c_void_p, c_size_t, POINTER(c_ulonglong),
                                     POINTER(c_int), POINTER(c_int),
                                     POINTER(c_int), POINTER(c_short),
                                     POINTER(c_int), POINTER(c_uint),
                                     POINTER(c_int)]
    _cb_lib.fays_cb_poll.restype = c_int
    _cb_lib.fays_cb_stop.restype = None
    _cb_ts = c_ulonglong(0)
    _cb_w = c_int(0)
    _cb_h = c_int(0)
    _cb_ch = c_int(0)
    _cb_enc = c_short(0)
    _cb_step = c_int(0)
    _cb_nbytes = c_uint(0)
    _cb_seq = c_int(0)
    _cb_last_frame = time.monotonic()

rotate_180 = False

# ── 并行 JPEG 编码（PIPE 模式）────────────────────────────────
# ARM 弱核（A53 级）上 2MP JPEG 编码是帧率墙：cv2.imencode 会释放
# GIL，左右目提交到 2 线程池可真并行到两个核，每帧编码时间≈单目
# （x86 上编码远快于帧周期，此改动零副作用）。JPEG_Q 环境变量调
# 质量（默认 85，与历史帧流一致）。
_JPEG_Q = int(os.environ.get("JPEG_Q", "85"))
from concurrent.futures import ThreadPoolExecutor
_ENC_POOL = ThreadPoolExecutor(max_workers=2)

# ── IMU 采集线程 ────────────────────────────────────────────────
# GetImuData 每次调用只返回最新一条样本 → 必须高频轮询才能收齐
# （官方 fays_vikit_example.cpp 的 ImuOnlineCapture 即此模式）。
# 采集线程把样本放入缓冲，帧循环每帧取走该帧周期内积累的全部样本。
#
# 采样率: SLAM 算法只需 100-200Hz IMU（VINS 100-200Hz, ORB-SLAM3 200Hz），
# 无需 1000Hz 全速。IMU_HZ 控制目标采样率（默认 200, 0=关闭采集）。
# 轮询率 = 2.5× 采样率（GetImuData 只返回最新 → 轮询必须快于采样）。
# ★ 实测教训: 2kHz 轮询（sleep 0.5ms）每秒 ~2000 次 GIL 抢占,
#   把 Stereo FPS 从 25 拖到 6。降采样后 GIL 压力降 4 倍。
IMU_HZ = float(os.environ.get("IMU_HZ", "200"))
_IMU_POLL_S = 1.0 / (2.5 * IMU_HZ) if IMU_HZ > 0 else 0.0
import threading
_imu_buffer = []
_imu_lock = threading.Lock()

def _imu_collector():
    _imu = AtrakIMU()
    while True:
        if _fn_imu(handle, byref(_imu)) == 0:
            with _imu_lock:
                _imu_buffer.append((
                    _imu.timestamp,
                    _imu.gyro[0], _imu.gyro[1], _imu.gyro[2],
                    _imu.acc[0],  _imu.acc[1],  _imu.acc[2],
                ))
        time.sleep(_IMU_POLL_S)

def _drain_imu():
    """取走缓冲中积累的全部 IMU 样本（线程安全）。"""
    with _imu_lock:
        batch = _imu_buffer[:]
        _imu_buffer.clear()
    return batch

if PIPE_MODE and IMU_HZ > 0:
    threading.Thread(target=_imu_collector, daemon=True).start()

if not PIPE_MODE:
    print("=" * 55)
    print("  S80M 双目 RGB 实时预览")
    print("  按键: Q/Esc=退出  S=截图  R=切换旋转")
    print("=" * 55)
    cv2.namedWindow("Left (cam0)",  cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right (cam1)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Left (cam0)",  640, 400)
    cv2.resizeWindow("Right (cam1)", 640, 400)

try:
    while True:
        if CB_BRIDGE:
            # 回调取帧：shim 从环槽拷出 SDK 装配完成的帧（seqlock）——
            # 不经过 GetStereoFrames 的内部拷贝，绕过其与装配的竞态
            #（8/31 主程序录制回归：轮询+深度绑定=交付帧水平错位带）
            _got = _cb_lib.fays_cb_poll(
                img_buf, MAX_BYTES,
                byref(_cb_ts), byref(_cb_w), byref(_cb_h), byref(_cb_ch),
                byref(_cb_enc), byref(_cb_step), byref(_cb_nbytes),
                byref(_cb_seq))
            if _got != 1:
                if time.monotonic() - _cb_last_frame > 5.0:
                    print("[cb-bridge] 5s 无相机帧，退出（掉线/端口被占？）",
                          file=sys.stderr)
                    break
                time.sleep(0.002)
                continue
            _cb_last_frame = time.monotonic()
            if _cb_ts.value <= img.timestamp:
                # 迟到帧（USB 重传批：实测 311 帧中 4 次 120ms 回跳=整 3 帧，
                # 轮询路径的桶抽帧掩盖了它）——跳过，保证交付/录制流
                # 时间戳单调（父进程 1/30s 桶抽帧假设单调）
                continue
            img.width = _cb_w.value
            img.height = _cb_h.value
            img.channel = _cb_ch.value
            img.encoding = _cb_enc.value
            img.step = _cb_step.value
            img.bytes = _cb_nbytes.value
            img.timestamp = _cb_ts.value
            img.seq = _cb_seq.value
            img.device_id = 0
        else:
            ret = _fn_stereo(handle, byref(img))
            if ret != 0:
                if not PIPE_MODE: cv2.waitKey(1)
                continue

        w, h, ch = img.width, img.height, img.channel
        # SDK 输出为上下拼接图: 上=左目, 下=右目 (见 depth_video_demo.cpp)
        mono_h = h // 2

        if ch == 1:
            full = np.ctypeslib.as_array(
                (c_ubyte * (w * h)).from_address(ctypes.addressof(img_buf))
            ).reshape((h, w))
        else:
            full = np.ctypeslib.as_array(
                (c_ubyte * (w * h * 3)).from_address(ctypes.addressof(img_buf))
            ).reshape((h, w, 3))

        left  = full[:mono_h, :].copy()
        right = full[mono_h:, :].copy()

        # ★ S80M SDK 帧实际是 RGB 通道布局（encoding 声称 BGR8 但通道序为 RGB）
        # ★ 统一转成 OpenCV BGR，保证 imshow / imencode 颜色正确
        left  = cv2.cvtColor(left,  cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

        if rotate_180:
            left  = cv2.rotate(left,  cv2.ROTATE_180)
            right = cv2.rotate(right, cv2.ROTATE_180)

        # ── 深度引擎（隔帧喂 ≈25fps；仅新鲜深度更新待发负载）──
        # 2026-09-03 投递空洞修复：引擎实测 ~27fps 产出上限，50fps 每帧
        # 喂入 = 每 20ms 预算内多花 3MB 通道翻转拷贝 + 3MB memmove + 引擎
        # 输入处理，而多喂的帧引擎根本算不完——纯挤占投递循环预算
        # （有深度录制空桶 8-15% vs 无深度 ~2%，60-180ms 投递空洞全在
        # worker 侧）。隔帧喂后输入 25fps ≈ 产出上限：喂入开销减半、
        # 产出率不掉；get 保持每帧轮询，结果一产出即随下一帧上管道
        # （深度到达延迟不变）
        _depth_send_len = 0
        if _depth_active and not _depth_broken:
            _feed_tick = not _feed_tick
            try:
                if _feed_tick:
                    _feed_src = (np.ascontiguousarray(full[:, :, ::-1])
                                 if ch == 3 else full)
                    if _feed_src.nbytes <= _MAX_STEREO_BYTES:
                        _feed_alt = not _feed_alt
                        _feed_img.data = ctypes.cast(
                            _feed_bufs[_feed_alt], POINTER(c_ubyte))
                        ctypes.memmove(_feed_img.data, _feed_src.ctypes.data,
                                       _feed_src.nbytes)
                        _feed_img.width = w
                        _feed_img.height = h
                        _feed_img.channel = ch
                        _feed_img.encoding = img.encoding
                        _feed_img.step = img.step
                        _feed_img.bytes = img.bytes
                        _feed_img.timestamp = img.timestamp
                        _feed_img.seq = img.seq
                        _feed_img.device_id = img.device_id
                        _d_feed(_depth_handle, byref(_feed_img))
                for _out in (_depth_out, _disp_out, _paired_out):
                    _out.width = _out.height = _out.channel = 0
                    _out.step = _out.bytes = 0
                if _d_get(_depth_handle, byref(_depth_out), byref(_disp_out),
                          byref(_paired_out)) == 0 and _depth_out.width > 0:
                    if _depth_out.timestamp != _last_depth_ts:
                        # 新深度：float32 米 ×1000 → uint16 毫米（管道带宽减半；
                        # NaN/Inf→0=无效，>65535 截断）
                        _last_depth_ts = _depth_out.timestamp
                        _dn = _depth_out.bytes // 4
                        _darr = np.ctypeslib.as_array(
                            _depth_out.data, shape=(_dn,)).copy()
                        _depth_m = _darr.reshape(_depth_out.height,
                                                 _depth_out.width)
                        _u16 = np.nan_to_num(_depth_m * 1000.0, nan=0.0,
                                             posinf=0.0, neginf=0.0)
                        _u16 = _u16.clip(0, 65535).astype(np.uint16)
                        _depth_payload = _u16.tobytes()
                        _depth_w, _depth_h = (_depth_out.width,
                                              _depth_out.height)
                        _depth_ts = _depth_out.timestamp
            except Exception as _e:
                _depth_broken = True
                print(f"[Depth] 深度引擎运行期异常，停用: {_e}",
                      file=sys.stderr)
        if _depth_payload:
            _depth_send_len = len(_depth_payload)

        if PIPE_MODE:
            # 取走本帧周期内采集线程积累的 IMU 样本
            imu_samples = _drain_imu()

            # JPEG 编码后发送（左右目并行到两核，见 _ENC_POOL 说明）。
            # 协议: [4B left_len][8B left_ts_ns][left_jpg]
            #       [4B right_len][8B right_ts_ns][right_jpg]
            #       [4B imu_count][imu_count × (8B ts + 8B×3 gyro + 8B×3 acc)]
            #       [4B depth_len][(depth_len>0: 8B depth_ts + 4B w + 4B h
            #        + w*h*2 字节 uint16 毫米)]   ← 仅 --depth-sdk-dir 模式；
            #        深度为引擎原生 P0 矫正空间（与原始 RGB 非同几何）
            # 时间戳均为 SDK 硬件纳秒时钟（帧/IMU 同源，SLAM 对齐用）
            _lf = _ENC_POOL.submit(cv2.imencode, ".jpg", left,
                                   [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])
            _rf = _ENC_POOL.submit(cv2.imencode, ".jpg", right,
                                   [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])
            _, lj = _lf.result()
            _, rj = _rf.result()
            try:
                PIPE_FD.write(struct.pack(">IQ", len(lj), img.timestamp))
                PIPE_FD.write(lj.tobytes())
                PIPE_FD.write(struct.pack(">IQ", len(rj), img.timestamp))
                PIPE_FD.write(rj.tobytes())
                PIPE_FD.write(struct.pack(">I", len(imu_samples)))
                for s in imu_samples:
                    PIPE_FD.write(struct.pack(">Q6d", *s))
                if DEPTH_MODE:
                    # 深度块（给过 --depth-sdk-dir 时每帧必写，解析确定性）：
                    #   [>I depth_len]  0=本帧无新深度
                    #   >0: [>Q depth_ts][>I w][>I h][w*h*2 字节 uint16 毫米 LE]
                    PIPE_FD.write(struct.pack(">I", _depth_send_len))
                    if _depth_send_len:
                        PIPE_FD.write(struct.pack(">QII", _depth_ts,
                                                  _depth_w, _depth_h))
                        PIPE_FD.write(_depth_payload)
                PIPE_FD.flush()
            except BrokenPipeError:
                break
        else:
            cv2.imshow("Left (cam0)",  left)
            cv2.imshow("Right (cam1)", right)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                ts = img.timestamp
                cv2.imwrite(f"s80m_left_{ts}.png",  left)
                cv2.imwrite(f"s80m_right_{ts}.png", right)
                print(f"[截图] 已保存 s80m_left_{ts}.png / s80m_right_{ts}.png")
            elif key == ord('r'):
                rotate_180 = not rotate_180
                print(f"[旋转] 180° 旋转: {'ON' if rotate_180 else 'OFF'}")

finally:
    if not PIPE_MODE:
        cv2.destroyAllWindows()
    if PIPE_FD:
        PIPE_FD.close()
    # 先清临时 yaml 再销毁句柄（_fn_destroy 可能抛异常，别让它跳过清理）
    for _tmp in (_PORTS_TMP, _ENGINE_CALIB_TMP, _DEPTH_TMP):
        if _tmp and os.path.isfile(_tmp):
            try:
                os.remove(_tmp)
            except OSError:
                pass
    # 先深度后相机（依赖反序销毁，s80c_depth_worker.py 同款口径）
    if _d_destroy is not None and _depth_handle is not None:
        try:
            _d_destroy(_depth_handle)
        except Exception:
            pass
    # 回调桥接先于相机句柄销毁（shim 只碰自身环缓冲，序无关但保持反序）
    if CB_BRIDGE and _cb_lib is not None:
        try:
            _cb_lib.fays_cb_stop()
        except Exception:
            pass
    _fn_destroy(handle)
    if not PIPE_MODE:
        print("SDK 已释放, 退出.")

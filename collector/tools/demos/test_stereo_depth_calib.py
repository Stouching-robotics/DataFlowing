#!/usr/bin/env python3
"""
S80M 双目深度 Demo —— 使用 FaysSense VI Kit SDK 官方深度引擎

管线 (与 SDK 文档 stereo_depth.md 9.2 节一致):
  1. FAYS_VIK_CreateHandleWithConfig       打开相机 (fays_vikit.yaml)
  2. FAYS_ATRAK_D_CreateHandleWithConfig   创建深度引擎 (stereo_depth.yaml, depth_mode=1 CPU)
  3. FAYS_ATRAK_D_BindViKit                绑定相机到深度引擎 (在线模式, 标定由相机提供)
  4. 循环:
       FAYS_VIK_GetStereoFrames             -> 上下拼接双目帧 (上半左目 / 下半右目)
       FAYS_ATRAK_D_FeedStereoImage         -> 送入深度引擎
       FAYS_ATRAK_D_GetDepthImage           -> 深度图 (float32, 米) + 视差图
       FAYS_ATRAK_D_GetRectifiedImage       -> 矫正后双目图 (BGR8, 上下拼接)
  5. 'p' 键: FAYS_ATRAK_D_GetPointcloud     -> 导出点云 PLY

注: S80M 固件实际拼接为 上=右目/下=左目, 与 SDK 文档假设 (上=左) 相反,
    左右顺序已由官方配置 stereo_swap_lr: 1 (fays_vikit.yaml) 在传感器层解决,
    脚本不再手动交换 (SWAP_LR=False); 可按 L 切换验证.

用法:
    ./run_stereo_depth_demo.sh      # 必须用启动器 (深度引擎需 SDK 自带 OpenCV 4.2)
按键:
    Q / Esc 退出 | S 截图 | P 导出点云 PLY | L 切换左右 | 1-4 调整深度范围
"""

import ctypes
import os
import sys
import time
import struct
from ctypes import (
    c_void_p, c_char_p, c_int, c_short, c_float,
    c_ubyte, c_uint, c_uint8, c_uint32, c_ulonglong,
    POINTER, Structure, byref, cast,
)

# ═══════════════════════════════════════════════════════════════
#  路径
# ═══════════════════════════════════════════════════════════════

SDK_DIR = os.environ.get("FAYSSENSE_SDK_DIR", "")
if not SDK_DIR:
    raise SystemExit(
        "[ERROR] 请先设置 FAYSSENSE_SDK_DIR=<FaysSense VI Kit Release 目录>")
VIKIT_LIB   = os.path.join(SDK_DIR, "lib", "fays_atrak", "x86_64", "Release",
                           "libfays_vikit.so")
DEPTH_LIB   = os.path.join(SDK_DIR, "lib", "fays_atrak", "x86_64", "Release",
                           "libfayssense_aikit_depth.so")
VIKIT_CONFIG = os.path.join(SDK_DIR, "config", "fays_vikit.yaml")
DEPTH_CONFIG = os.path.join(SDK_DIR, "config", "perception",
                            "stereo_depth", "stereo_depth.yaml")

# ── OpenCV 预加载 (libfays_vikit.so / aikit_depth.so 依赖其符号) ──
# 必须放在 import cv2 之前: SDK 库无 DT_NEEDED, 靠全局符号表解析.
#
# 关键: 深度引擎 libfayssense_aikit_depth.so 按 SDK 自带 OpenCV 4.2.0 编译,
# 它向 cv::stereoRectify 传 MatExpr/STD_ARRAY 数组, 而系统 OpenCV 4.6/4.13
# 的 getMat_ 不支持这两类 (抛 "Unknown/unsupported array type").
# 因此优先预载 SDK 自带 4.2 (lib406 目录), 找不到才回退系统 4.6.
OPENCV_LIB406 = os.path.join(SDK_DIR, "thirdparty", "opencv-4.2.0-linux-x86_64",
                             "lib406")

def _preload_opencv_dir(libdir):
    """从目录预加载 OpenCV 模块 + 外部依赖 (RTLD_GLOBAL)."""
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
    # 回退: 系统 OpenCV 4.6 (引擎在其下会崩, 仅限无自带 4.2 的环境)
    for _lib in ["libopencv_core", "libopencv_imgproc", "libopencv_imgcodecs",
                 "libopencv_highgui", "libopencv_calib3d", "libopencv_ximgproc",
                 "libopencv_features2d", "libopencv_flann", "libopencv_video"]:
        for _suffix in [".so.406", ".so.4.2"]:
            try:
                ctypes.CDLL(_lib + _suffix, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue

# cv2 在预载后导入: python 的 cv2 走自身 rpath 绑定 4.13, 与 SDK 的 4.2 互不干扰
import numpy as np
import cv2

# ── Qt 平台插件 ──────────────────────────────────────────────
_OV_PLUGINS_DIR = os.path.join(os.path.dirname(cv2.__file__), "qt", "plugins")
_OV_PLATFORMS = os.path.join(_OV_PLUGINS_DIR, "platforms.bak")
if not os.path.isdir(_OV_PLATFORMS):
    _OV_PLATFORMS = os.path.join(_OV_PLUGINS_DIR, "platforms")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _OV_PLATFORMS

# ═══════════════════════════════════════════════════════════════
#  SDK C 结构体 (对应 fays_atrak_types.h)
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
    """矫正矩阵 (fays_atrak_types.h)"""
    _fields_ = [
        ("cam_pair", c_uint8 * 2),
        ("R0",       c_float * 9),    # 左目 旋转
        ("R1",       c_float * 9),    # 右目 旋转
        ("P0",       c_float * 12),   # 左目 投影矩阵
        ("P1",       c_float * 12),   # 右目 投影矩阵
    ]

class AtrakMap(Structure):
    """点云: pos = N×3 float, rgb = N×1 ARGB8 (0xRRGGBBAA 位序: 高字节 R)"""
    _fields_ = [
        ("device_id",  c_int),
        ("timestamp",  c_ulonglong),
        ("point_num",  c_uint),
        ("pos",        POINTER(c_float)),
        ("rgb",        POINTER(c_uint32)),
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
        ("timeshift_cam_imu", ctypes.c_double),
    ]

class AtrakCamChainParam(Structure):
    _fields_ = [
        ("num_of_cams",    c_uint32),
        ("downsize_ratio", c_uint32),
        ("cameras",        AtrakCamParam * FAYS_ATRAK_MAX_CAMERAS),
    ]

class AtrakImuParam(Structure):
    _fields_ = [
        ("accelerometer_noise_density", ctypes.c_double),
        ("accelerometer_random_walk",   ctypes.c_double),
        ("gyroscope_noise_density",     ctypes.c_double),
        ("gyroscope_random_walk",       ctypes.c_double),
        ("update_rate",                 c_float),
    ]

class AtrakCalibrationParam(Structure):
    _fields_ = [
        ("cameras", AtrakCamChainParam),
        ("imu",     AtrakImuParam),
    ]

# ═══════════════════════════════════════════════════════════════
#  加载库
#   libfays_vikit.so        -> C++ 混淆符号 (extern C++ 接口)
#   libfayssense_aikit_depth.so -> C 符号 (extern "C", 直接可用)
# ═══════════════════════════════════════════════════════════════

# vikit 必须 RTLD_GLOBAL 加载: 深度库 dlopen 时需要解析其中的 FAYS_VIK_* 符号
_vikit = ctypes.CDLL(VIKIT_LIB, mode=ctypes.RTLD_GLOBAL)
try:
    _depth = ctypes.CDLL(DEPTH_LIB)
except OSError as e:
    print(f"[ERROR] 深度引擎加载失败: {e}")
    print("  请用启动器运行 (它会将 SDK 自带 OpenCV 4.2 加入 LD_LIBRARY_PATH):")
    print("      ./run_stereo_depth_demo.sh")
    sys.exit(1)

# ── ViKit (混淆符号) ────────────────────────────────────────
_vikit_create  = getattr(_vikit, "_Z31FAYS_VIK_CreateHandleWithConfigPPvPKc")
_vikit_destroy = getattr(_vikit, "_Z22FAYS_VIK_DestroyHandlePv")
_vikit_stereo  = getattr(_vikit, "_Z24FAYS_VIK_GetStereoFramesPvP10AtrakImage")
_vikit_version = getattr(_vikit, "_Z19FAYS_VIK_GetVersionPv")
_vikit_calib   = getattr(_vikit, "_Z28FAYS_VIK_GetCalibrationParamPvP21AtrakCalibrationParam")

_vikit_create.argtypes  = [POINTER(c_void_p), c_char_p]
_vikit_create.restype   = c_int
_vikit_destroy.argtypes = [c_void_p]
_vikit_destroy.restype  = c_int
_vikit_stereo.argtypes  = [c_void_p, POINTER(AtrakImage)]
_vikit_stereo.restype   = c_int
_vikit_version.argtypes = [c_void_p]
_vikit_version.restype  = c_char_p
_vikit_calib.argtypes   = [c_void_p, POINTER(AtrakCalibrationParam)]
_vikit_calib.restype    = c_int

# ── 深度引擎 (C 符号) ────────────────────────────────────────
_d_create  = _depth.FAYS_ATRAK_D_CreateHandleWithConfig
_d_destroy = _depth.FAYS_ATRAK_D_DestroyHandle
_d_bind    = _depth.FAYS_ATRAK_D_BindViKit
_d_feed    = _depth.FAYS_ATRAK_D_FeedStereoImage
_d_get     = _depth.FAYS_ATRAK_D_GetDepthImage
_d_rect    = _depth.FAYS_ATRAK_D_GetRectifiedImage
_d_rectinfo = _depth.FAYS_ATRAK_D_GetRectifyInfo
_d_cloud   = _depth.FAYS_ATRAK_D_GetPointcloud

_d_create.argtypes  = [POINTER(c_void_p), c_char_p]
_d_create.restype   = c_int
_d_destroy.argtypes = [c_void_p]
_d_destroy.restype  = c_int
_d_bind.argtypes    = [c_void_p, c_void_p]
_d_bind.restype     = c_int
_d_feed.argtypes    = [c_void_p, POINTER(AtrakImage)]
_d_feed.restype     = c_int
_d_get.argtypes     = [c_void_p, POINTER(AtrakDepthImage),
                       POINTER(AtrakDepthImage), POINTER(AtrakImage)]
_d_get.restype      = c_int
_d_rect.argtypes    = [c_void_p, POINTER(AtrakImage)]
_d_rect.restype     = c_int
_d_rectinfo.argtypes = [c_void_p, POINTER(AtrakRectifyInfo)]
_d_rectinfo.restype  = c_int
_d_cloud.argtypes   = [c_void_p, POINTER(AtrakMap), c_int]
_d_cloud.restype    = c_int

# ═══════════════════════════════════════════════════════════════
#  初始化: 相机 + 深度引擎 + 绑定
# ═══════════════════════════════════════════════════════════════

# ── 1. 打开相机 ──
vikit_handle = c_void_p()
ret = _vikit_create(byref(vikit_handle), VIKIT_CONFIG.encode())
if ret != 0:
    print(f"[ERROR] ViKit 初始化失败, ret={ret} (相机是否连接? /dev/video* 端口配置?)")
    sys.exit(1)

ver = _vikit_version(vikit_handle)
print(f"ViKit 版本: {ver.decode() if ver else 'unknown'}")

# ── 2. 创建深度引擎 ──
depth_handle = c_void_p()
ret = _d_create(byref(depth_handle), DEPTH_CONFIG.encode())
if ret != 0:
    print(f"[ERROR] 深度引擎创建失败, ret={ret}")
    print(f"  请检查 {DEPTH_CONFIG}:")
    print("    - depth_mode: 1 (CPU, 无需 NPU 模型)")
    print("    - input_stereo_width/height 与 fays_vikit.yaml 的 stereo_single_cam_* 一致")
    _vikit_destroy(vikit_handle)
    sys.exit(1)
print(f"深度引擎创建成功 (config: {DEPTH_CONFIG})")

# ── 3. 绑定相机 → 深度引擎 (在线模式, 标定由相机提供) ──
ret = _d_bind(depth_handle, vikit_handle)
if ret != 0:
    print(f"[ERROR] BindViKit 失败, ret={ret}")
    _d_destroy(depth_handle)
    _vikit_destroy(vikit_handle)
    sys.exit(1)
print("深度引擎已绑定 ViKit ✓\n")

# ── 打印标定摘要 (仅信息展示, 校正由引擎内部完成) ──
calib = AtrakCalibrationParam()
if _vikit_calib(vikit_handle, byref(calib)) == 0:
    cam0 = calib.cameras.cameras[0]
    cam1 = calib.cameras.cameras[1]
    baseline = np.linalg.norm(list(cam1.T_cn_cnm1.trans))
    print("相机标定 (信息):")
    print(f"  Cam0 fx={cam0.intrinsics.fx:.2f} fy={cam0.intrinsics.fy:.2f}  "
          f"{cam0.intrinsics.width}x{cam0.intrinsics.height}")
    print(f"  Cam1 fx={cam1.intrinsics.fx:.2f} fy={cam1.intrinsics.fy:.2f}  "
          f"{cam1.intrinsics.width}x{cam1.intrinsics.height}")
    print(f"  Baseline = {baseline*1000:.2f} mm")
    print()

# ── 矫正信息 (fx / 基线, 由引擎返回) ──
rect_info = AtrakRectifyInfo()
if _d_rectinfo(depth_handle, byref(rect_info)) == 0:
    fx = rect_info.P0[0]
    baseline = abs(rect_info.P1[3]) / fx  # P1[3] = -fx·Tx
    print(f"矫正矩阵: fx'={fx:.1f}  baseline'={baseline*1000:.1f} mm")
    print()

# ═══════════════════════════════════════════════════════════════
#  预分配输出缓冲区 (调用方分配, SDK 写入)
# ═══════════════════════════════════════════════════════════════

# SDK 头文件宏 (fays_atrak_types.h)
FAYS_ATRAK_MONO_MAX_BYTES = 1280 * 800 * 2
FAYS_ATRAK_RGB_MAX_BYTES  = 3840 * 2160 * 3
FAYS_ATRAK_IMG_MAX_BYTES  = max(FAYS_ATRAK_MONO_MAX_BYTES, FAYS_ATRAK_RGB_MAX_BYTES)
FAYS_ATRAK_DEPTH_IMG_MAX_BYTES = FAYS_ATRAK_IMG_MAX_BYTES * 4  # * sizeof(float)
FAYS_ATRAK_MAX_DENSE_MAP_POINTS_NUM = 262144

# 输出缓冲必须按 SDK 头文件宏分配:
# 引擎按宏上限写入, 缓冲偏小会在特定时刻越界写堆 -> 随机段错误.
# 与 C++ 版 (stereo_depth_demo/main.cpp) 保持一致.
MAX_STEREO_BYTES = FAYS_ATRAK_MONO_MAX_BYTES * 3    # 双目拼接帧上限 (1280*800*2*3)
MAX_IMG_BYTES    = FAYS_ATRAK_IMG_MAX_BYTES         # rect/对齐图像上限 (3840*2160*3)
MAX_DEPTH_FLOATS = FAYS_ATRAK_DEPTH_IMG_MAX_BYTES // 4  # 深度/视差: float
MAX_CLOUD_POINTS = FAYS_ATRAK_MAX_DENSE_MAP_POINTS_NUM

stereo_buf = (c_ubyte * MAX_STEREO_BYTES)()
stereo_buf2 = (c_ubyte * MAX_STEREO_BYTES)()   # 左右交换用缓冲区
depth_buf  = (c_float * MAX_DEPTH_FLOATS)()
disp_buf   = (c_float * MAX_DEPTH_FLOATS)()
rect_buf   = (c_ubyte * MAX_STEREO_BYTES)()
paired_buf = (c_ubyte * MAX_IMG_BYTES)()
cloud_pos  = (c_float * (MAX_CLOUD_POINTS * 3))()
cloud_rgb  = (c_uint32 * MAX_CLOUD_POINTS)()

stereo_img = AtrakImage()
stereo_img.data = cast(stereo_buf, POINTER(c_ubyte))

# 左右交换用图像结构 (指向 stereo_buf2)
feed_img = AtrakImage()
feed_img.data = cast(stereo_buf2, POINTER(c_ubyte))

depth_out = AtrakDepthImage()
depth_out.data = cast(depth_buf, POINTER(c_float))
disp_out  = AtrakDepthImage()
disp_out.data  = cast(disp_buf, POINTER(c_float))
rect_out  = AtrakImage()
rect_out.data  = cast(rect_buf, POINTER(c_ubyte))
paired_out = AtrakImage()
paired_out.data = cast(paired_buf, POINTER(c_ubyte))

cloud_out = AtrakMap()
cloud_out.pos = cast(cloud_pos, POINTER(c_float))
cloud_out.rgb = cast(cloud_rgb, POINTER(c_uint32))


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════

def img_to_numpy(img: AtrakImage) -> np.ndarray:
    """AtrakImage (uint8) -> numpy, 按 step 处理行填充, 返回副本."""
    ch = img.channel if img.channel > 0 else 1
    n = img.bytes
    arr = np.ctypeslib.as_array(img.data, shape=(n,)).copy()
    row_bytes = img.step if img.step > 0 else img.width * ch
    if row_bytes == img.width * ch:
        return arr.reshape(img.height, img.width, ch)
    # 带行填充: 用 as_strided 按 step 切行
    from numpy.lib.stride_tricks import as_strided
    return as_strided(arr, shape=(img.height, img.width, ch),
                      strides=(row_bytes, ch, 1)).copy()

def depth_to_numpy(img: AtrakDepthImage) -> np.ndarray:
    """AtrakDepthImage (float32) -> numpy 2D."""
    n = img.bytes // 4
    arr = np.ctypeslib.as_array(img.data, shape=(n,)).copy()
    return arr.reshape(img.height, img.width)


def save_pointcloud_ply(path: str, pos: np.ndarray, rgb: np.ndarray):
    """pos: (N,3) float32, rgb: (N,) uint32 ARGB8 -> 二进制 PLY."""
    n = pos.shape[0]
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(b"end_header\n")
        for i in range(n):
            x, y, z = pos[i]
            c = int(rgb[i])
            f.write(struct.pack("<fffBBB", x, y, z,
                                (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF))


# ═══════════════════════════════════════════════════════════════
#  实时循环
# ═══════════════════════════════════════════════════════════════

cv2.namedWindow("Depth Heatmap", cv2.WINDOW_NORMAL)
cv2.namedWindow("Stereo (raw/rect)", cv2.WINDOW_NORMAL)
cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Depth Heatmap", 640, 400)
cv2.resizeWindow("Stereo (raw/rect)", 640, 480)
cv2.resizeWindow("Disparity", 640, 400)

D_NEAR, D_FAR = 0.3, 4.0
# 左右顺序已由官方配置 stereo_swap_lr: 1 (fays_vikit.yaml) 在传感器层解决,
# 脚本不再手动交换; 可按 L 键切换以验证配置是否生效.
SWAP_LR = False
# S80M 固件实际输出 RGB 通道序 (与头文件声称的 BGR 相反), 需转 BGR 再显示/送引擎
SWAP_RGB = True
frame_idx = 0
fps = 0.0
_frame_info_shown = False
_t0 = time.monotonic()

print(f"\n按 Q/Esc 退出 | S 截图 | P 导出点云 PLY | 1-4 调整深度范围")
print(f"深度显示范围: {D_NEAR}m - {D_FAR}m\n")

try:
    while True:
        # ── 1. 获取双目帧 (上下拼接) ──
        ret = _vikit_stereo(vikit_handle, byref(stereo_img))
        if ret != 0:                     # -2 = VIKIT_RET_NO_NEW_FRAME
            cv2.waitKey(1)
            continue

        # 首次检查帧格式
        if not _frame_info_shown:
            enc = stereo_img.encoding  # AIE_BGR8=0, AIE_RAW8=2
            print(f"双目帧: {stereo_img.width}x{stereo_img.height} "
                  f"ch={stereo_img.channel} encoding={enc} bytes={stereo_img.bytes}")
            print(f"  左右顺序: {'已交换 (上=右/下=左)' if SWAP_LR else '默认 (上=左/下=右)'} "
                  f"—— 按 L 切换")
            _frame_info_shown = True

        # ── 2. 通道顺序: S80M 固件实际输出 RGB (与头文件声称的 BGR 相反,
        # 同类固件怪癖, 与左右顺序反了一样). 统一转 BGR 再显示/送引擎. ──
        frame_np = img_to_numpy(stereo_img)          # 副本 (H, W, ch)
        if SWAP_RGB:
            frame_np = np.ascontiguousarray(frame_np[:, :, ::-1])  # RGB -> BGR

        # ── 3. 送引擎 (转换后的帧统一拷入 feed 缓冲) ──
        # 左右顺序已由官方配置 stereo_swap_lr: 1 在传感器层解决, 默认不交换;
        # 按 L 可切换验证 (不能用 flipud 整帧翻转, 会倒置每只眼).
        # 注意: 引擎按 BGR 约定处理输入 (与 C++ 版一致) —— 直接送原始 RGB
        # 会导致 rect 输出窗口通道反色 (用户实测确认).
        _hh = frame_np.shape[0] // 2
        if SWAP_LR:
            frame_np = np.vstack([frame_np[_hh:], frame_np[:_hh]])  # 连续数组
        feed = byref(feed_img)
        if frame_np.nbytes <= MAX_STEREO_BYTES:
            ctypes.memmove(feed_img.data, frame_np.ctypes.data, frame_np.nbytes)
            feed_img.width     = stereo_img.width
            feed_img.height    = stereo_img.height
            feed_img.channel   = stereo_img.channel
            feed_img.encoding  = stereo_img.encoding
            feed_img.step      = stereo_img.step
            feed_img.bytes     = stereo_img.bytes
            feed_img.timestamp = stereo_img.timestamp
            feed_img.seq       = stereo_img.seq
            feed_img.device_id = stereo_img.device_id

        # ── 3. 送入深度引擎 ──
        if _d_feed(depth_handle, feed) != 0:
            continue

        # ── 3. 取深度 / 视差 (调用前清零字段, 引擎回填) ──
        for out in (depth_out, disp_out, paired_out):
            out.width = out.height = out.channel = 0
            out.step = out.bytes = 0
        if _d_get(depth_handle, byref(depth_out), byref(disp_out),
                  byref(paired_out)) != 0 or depth_out.width == 0:
            cv2.waitKey(1)
            continue

        depth = depth_to_numpy(depth_out)          # float32, 米
        disp  = depth_to_numpy(disp_out)           # float32 视差
        disp  = np.maximum(disp, 0.0)

        # ── 4. 矫正图 (上左目 / 下右目) ──
        rect_out.width = rect_out.height = rect_out.channel = 0
        rect_out.step = rect_out.bytes = 0
        rect_np = None
        if _d_rect(depth_handle, byref(rect_out)) == 0 and rect_out.width > 0:
            rect_np = img_to_numpy(rect_out)

        # ── 5. 深度热力图 ──
        finite = np.isfinite(depth)
        visible = finite & (depth > D_NEAR) & (depth < D_FAR)
        normed = np.zeros_like(depth, dtype=np.uint8)
        if visible.any():
            vmin, vmax = depth[visible].min(), depth[visible].max()
            if vmax > vmin:
                normed[visible] = ((depth[visible] - vmin) /
                                   (vmax - vmin) * 254 + 1).astype(np.uint8)
            else:
                normed[visible] = 128
        heatmap = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
        heatmap[~visible] = (0, 0, 0)

        h, w = depth.shape
        d_center = depth[h // 2, w // 2]
        cv2.putText(heatmap,
            f"Frame {frame_idx} | {fps:.1f} fps | Center: "
            f"{d_center:.2f} m" if np.isfinite(d_center) and d_center > 0 else
            f"Frame {frame_idx} | {fps:.1f} fps | Center: --",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(heatmap,
            f"Depth {D_NEAR:.1f}-{D_FAR:.1f} m | SDK depth "
            f"{w}x{h} (float32, m)",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # ── 6. 原始 + 矫正 并排显示 ──
        stereo_np = frame_np                 # 已按 SWAP_LR 交换过
        half_h = stereo_np.shape[0] // 2
        left_raw, right_raw = stereo_np[:half_h], stereo_np[half_h:]

        def _small(im):
            s = im.shape
            return cv2.resize(im, (s[1] // 4, s[0] // 4))

        if rect_np is not None:
            rh = rect_np.shape[0] // 2
            left_rect, right_rect = rect_np[:rh], rect_np[rh:]
            top = np.hstack([_small(left_raw), _small(right_raw)])
            bot = np.hstack([_small(left_rect), _small(right_rect)])
            cv2.putText(top, 'Left raw / Right raw', (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            cv2.putText(bot, 'Left rect / Right rect', (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            stereo_view = np.vstack([top, bot])
        else:
            stereo_view = np.hstack([_small(left_raw), _small(right_raw)])
        cv2.imshow("Stereo (raw/rect)", stereo_view)

        # ── 7. 视差图 ──
        if disp.max() > 0:
            disp_norm = ((disp / disp.max()) * 255).astype(np.uint8)
        else:
            disp_norm = np.zeros_like(disp, dtype=np.uint8)
        disp_heat = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)
        cv2.imshow("Disparity", disp_heat)
        cv2.imshow("Depth Heatmap", heatmap)

        # ── 8. 按键处理 ──
        key = cv2.waitKeyEx(1)
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            ts = stereo_img.timestamp
            cv2.imwrite(f'depth_heat_{ts}.png', heatmap)
            cv2.imwrite(f'disparity_{ts}.png', disp_heat)
            if rect_np is not None:
                cv2.imwrite(f'stereo_rectified_{ts}.png', rect_np)
            with open(f'depth_{ts}.raw', 'wb') as f:
                f.write(depth.astype(np.float32).tobytes())
            print(f'[截图] depth_heat_{ts}.png / disparity_{ts}.png / '
                  f'depth_{ts}.raw (float32 meters)')
        elif key == ord('p'):
            cloud_out.point_num = 0
            if _d_cloud(depth_handle, byref(cloud_out), MAX_CLOUD_POINTS) == 0 \
                    and cloud_out.point_num > 0:
                n = int(cloud_out.point_num)
                pos = np.ctypeslib.as_array(cloud_pos, shape=(n * 3,)).copy()
                rgb = np.ctypeslib.as_array(cloud_rgb, shape=(n,)).copy()
                ply = f'pointcloud_{stereo_img.timestamp}.ply'
                save_pointcloud_ply(ply, pos.reshape(n, 3), rgb)
                print(f'[点云] {ply} ({n} 点)')
            else:
                print('[点云] 获取失败 (深度无效区域过多?)')
        elif key in (ord('l'), ord('L')):
            SWAP_LR = not SWAP_LR
            print(f"[左右] 已切换为 {'交换 (上=右/下=左)' if SWAP_LR else '默认 (上=左/下=右)'}")
        elif key == ord('1'):
            D_NEAR = max(0.05, D_NEAR - 0.1)
            print(f"深度显示: [{D_NEAR:.2f}, {D_FAR:.2f}] m")
        elif key == ord('2'):
            D_NEAR = min(D_FAR - 0.1, D_NEAR + 0.1)
            print(f"深度显示: [{D_NEAR:.2f}, {D_FAR:.2f}] m")
        elif key == ord('3'):
            D_FAR = max(D_NEAR + 0.1, D_FAR - 0.5)
            print(f"深度显示: [{D_NEAR:.2f}, {D_FAR:.2f}] m")
        elif key == ord('4'):
            D_FAR = min(10.0, D_FAR + 0.5)
            print(f"深度显示: [{D_NEAR:.2f}, {D_FAR:.2f}] m")
        elif key > 0 and key not in (ord('q'), 27, ord('s'), ord('p'),
                ord('l'), ord('L'),
                ord('1'), ord('2'), ord('3'), ord('4')):
            print(f'[按键] code={key} hex=0x{key:04X}')

        frame_idx += 1
        now = time.monotonic()
        if now - _t0 >= 1.0:
            fps = frame_idx / (now - _t0)
            frame_idx = 0
            _t0 = now
            # 每秒打印一次深度统计, 方便验证深度数据真实性
            _valid = np.isfinite(depth) & (depth > 0.05)
            if _valid.any():
                print(f"  [深度] 有效 {_valid.sum()/max(_valid.size,1)*100:.0f}%  "
                      f"中位 {np.median(depth[_valid]):.2f}m  "
                      f"范围 {depth[_valid].min():.2f}-{depth[_valid].max():.2f}m")
            else:
                print("  [深度] 当前帧无有效深度数据")

finally:
    cv2.destroyAllWindows()
    _d_destroy(depth_handle)
    _vikit_destroy(vikit_handle)
    print("\n已退出, 深度引擎与相机句柄已释放")

#!/usr/bin/env python3
"""
诊断脚本：验证 S80M 帧布局 + 标定数据
"""
import ctypes
import numpy as np
import cv2
import os, sys
from ctypes import (
    c_void_p, c_char_p, c_int, c_short, c_float, c_double,
    c_ubyte, c_uint, c_uint8, c_uint32, c_ulonglong,
    POINTER, Structure, byref, addressof,
)

SDK_DIR = os.environ.get("FAYSSENSE_SDK_DIR", "")
if not SDK_DIR:
    print("[ERROR] 请设置 FAYSSENSE_SDK_DIR=<FaysSense VI Kit Release 目录>", file=sys.stderr)
    sys.exit(1)
VIKIT_LIB = os.path.join(SDK_DIR, "lib", "fays_atrak", "x86_64", "Release", "libfays_vikit.so")
VIKIT_CONFIG = os.path.join(SDK_DIR, "config", "fays_vikit.yaml")

for _lib in ["libopencv_core", "libopencv_imgproc", "libopencv_imgcodecs", "libopencv_highgui"]:
    for _suffix in [".so.406", ".so.4.2"]:
        try:
            ctypes.CDLL(_lib + _suffix, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue

_OV_PLUGINS_DIR = os.path.join(os.path.dirname(cv2.__file__), "qt", "plugins")
_OV_PLATFORMS = os.path.join(_OV_PLUGINS_DIR, "platforms.bak")
if not os.path.isdir(_OV_PLATFORMS):
    _OV_PLATFORMS = os.path.join(_OV_PLUGINS_DIR, "platforms")
if os.path.isdir(_OV_PLATFORMS):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _OV_PLATFORMS

# ── Struct definitions ──
class AtrakImage(Structure):
    _fields_ = [
        ("device_id", c_int), ("seq", c_int), ("timestamp", c_ulonglong),
        ("width", c_int), ("height", c_int), ("channel", c_int),
        ("encoding", c_short), ("step", c_int), ("bytes", c_uint),
        ("data", POINTER(c_ubyte)),
    ]

class AtrakIntrinsics(Structure):
    _fields_ = [
        ("cam_model", c_uint8),
        ("width", c_uint32), ("height", c_uint32),
        ("fx", c_float), ("fy", c_float), ("cx", c_float), ("cy", c_float),
        ("intrinsic_extra", c_float * 4),
        ("distortion_model", c_uint8),
        ("distortion", c_float * 8),
    ]

class AtrakExtrinsics(Structure):
    _fields_ = [("rot", c_float * 9), ("trans", c_float * 3)]

class AtrakCamParam(Structure):
    _fields_ = [
        ("cam_id", c_uint8), ("available_mask", c_uint8),
        ("intrinsics", AtrakIntrinsics),
        ("T_cn_cnm1", AtrakExtrinsics),
        ("T_cn_imu", AtrakExtrinsics),
        ("timeshift_cam_imu", c_double),
    ]

class AtrakCamChainParam(Structure):
    _fields_ = [("num_of_cams", c_uint32), ("downsize_ratio", c_uint32),
                ("cameras", AtrakCamParam * 3)]

class AtrakImuParam(Structure):
    _fields_ = [
        ("accelerometer_noise_density", c_double),
        ("accelerometer_random_walk", c_double),
        ("gyroscope_noise_density", c_double),
        ("gyroscope_random_walk", c_double),
        ("update_rate", c_float),
    ]

class AtrakCalibrationParam(Structure):
    _fields_ = [("cameras", AtrakCamChainParam), ("imu", AtrakImuParam)]

# ── Load SDK ──
lib = ctypes.CDLL(VIKIT_LIB)
_fn_create = getattr(lib, "_Z31FAYS_VIK_CreateHandleWithConfigPPvPKc")
_fn_destroy = getattr(lib, "_Z22FAYS_VIK_DestroyHandlePv")
_fn_stereo = getattr(lib, "_Z24FAYS_VIK_GetStereoFramesPvP10AtrakImage")
_fn_version = getattr(lib, "_Z19FAYS_VIK_GetVersionPv")
_fn_calib = getattr(lib, "_Z28FAYS_VIK_GetCalibrationParamPvP21AtrakCalibrationParam")

_fn_create.argtypes = [POINTER(c_void_p), c_char_p]
_fn_create.restype = c_int
_fn_destroy.argtypes = [c_void_p]
_fn_destroy.restype = c_int
_fn_stereo.argtypes = [c_void_p, POINTER(AtrakImage)]
_fn_stereo.restype = c_int
_fn_version.argtypes = [c_void_p]
_fn_version.restype = c_char_p
_fn_calib.argtypes = [c_void_p, POINTER(AtrakCalibrationParam)]
_fn_calib.restype = c_int

# ── Init ──
handle = c_void_p()
ret = _fn_create(byref(handle), VIKIT_CONFIG.encode())
if ret != 0:
    print(f"[ERROR] ViKit init failed, ret={ret}")
    sys.exit(1)

ver = _fn_version(handle)
print(f"ViKit version: {ver.decode() if ver else 'unknown'}")

# ── Read calibration ──
calib = AtrakCalibrationParam()
ret = _fn_calib(handle, byref(calib))
if ret != 0:
    print(f"[ERROR] Read calibration failed, ret={ret}")
    sys.exit(1)

ncams = calib.cameras.num_of_cams
print(f"\n{'='*70}")
print(f"  Cameras: {ncams}  |  Downsize ratio: {calib.cameras.downsize_ratio}")
print(f"{'='*70}")

for i in range(ncams):
    cam = calib.cameras.cameras[i]
    intr = cam.intrinsics
    ext = cam.T_cn_cnm1
    print(f"\n  ── Camera {i} (cam_id={cam.cam_id}, mask=0x{cam.available_mask:02X}) ──")
    print(f"  Intrinsics:  {intr.width}×{intr.height}")
    print(f"    fx={intr.fx:.3f}  fy={intr.fy:.3f}  cx={intr.cx:.3f}  cy={intr.cy:.3f}")
    print(f"    cam_model={intr.cam_model}  dist_model={intr.distortion_model}")
    print(f"    distortion: k1={intr.distortion[0]:.6f} k2={intr.distortion[1]:.6f} "
          f"k3={intr.distortion[2]:.6f} k4={intr.distortion[3]:.6f}")
    print(f"  T_cn_cnm1 (cam{i-1}→cam{i}):")
    print(f"    rot: [{ext.rot[0]:.4f} {ext.rot[1]:.4f} {ext.rot[2]:.4f}")
    print(f"          {ext.rot[3]:.4f} {ext.rot[4]:.4f} {ext.rot[5]:.4f}")
    print(f"          {ext.rot[6]:.4f} {ext.rot[7]:.4f} {ext.rot[8]:.4f}]")
    print(f"    trans: [{ext.trans[0]:.4f} {ext.trans[1]:.4f} {ext.trans[2]:.4f}]")
    baseline = np.sqrt(sum(t*t for t in ext.trans))
    print(f"    baseline = {baseline*1000:.2f} mm")

# ── Grab one frame ──
print(f"\n{'='*70}")
print(f"  Frame Layout Diagnosis")
print(f"{'='*70}")

MAX_BYTES = 1280 * 1600 * 3
img_buf = (c_ubyte * MAX_BYTES)()
img = AtrakImage()
img.data = ctypes.cast(img_buf, POINTER(c_ubyte))

for attempt in range(10):
    ret = _fn_stereo(handle, byref(img))
    if ret == 0:
        break
    cv2.waitKey(1)
else:
    print("[ERROR] Could not grab frame")
    sys.exit(1)

w, h, ch = img.width, img.height, img.channel
print(f"  Raw frame: {w}×{h}  channels={ch}")
print(f"  mono_w = {w//2}")

full = np.ctypeslib.as_array(
    (c_ubyte * (w * h * ch)).from_address(addressof(img_buf))
).reshape((h, w, ch))

SUB_H = 400
mono_w = w // 2

# ── Extract all 8 sub-regions (2 cols × 4 rows) ──
print(f"\n  ── Column 0 (left half of frame) ──")
for ri in range(4):
    r0, r1 = ri * SUB_H, (ri + 1) * SUB_H
    s = full[r0:r1, :mono_w]
    print(f"    Row {ri} [{r0}:{r1}, 0:{mono_w}]: "
          f"mean={s.mean():.1f} std={s.std():.1f}")

print(f"\n  ── Column 1 (right half of frame) ──")
for ri in range(4):
    r0, r1 = ri * SUB_H, (ri + 1) * SUB_H
    s = full[r0:r1, mono_w:w]
    print(f"    Row {ri} [{r0}:{r1}, {mono_w}:{w}]: "
          f"mean={s.mean():.1f} std={s.std():.1f}")

# ── Cross-correlation to check if cols are same lens ──
print(f"\n  ── Cross-correlation between col0 and col1 (same row) ──")
for ri in range(4):
    r0, r1 = ri * SUB_H, (ri + 1) * SUB_H
    c0 = cv2.cvtColor(full[r0:r1, :mono_w], cv2.COLOR_BGR2GRAY).astype(np.float32)
    c1 = cv2.cvtColor(full[r0:r1, mono_w:w], cv2.COLOR_BGR2GRAY).astype(np.float32)
    corr = cv2.matchTemplate(c0, c1[200:300, 200:300], cv2.TM_CCOEFF_NORMED)[0,0]
    print(f"    Row {ri}: col0 vs col1 correlation = {corr:.4f}")

# ── Save diagnostic images ──
# Full frame (resized to fit screen)
full_small = cv2.resize(full, (640, 800))
cv2.imwrite("/tmp/diag_full_frame.png", full_small)

# 2×4 grid of all sub-regions
grid = np.zeros((SUB_H * 4, mono_w * 2, 3), dtype=np.uint8)
for ri in range(4):
    r0, r1 = ri * SUB_H, (ri + 1) * SUB_H
    grid[ri*SUB_H:(ri+1)*SUB_H, 0:mono_w] = full[r0:r1, :mono_w]
    grid[ri*SUB_H:(ri+1)*SUB_H, mono_w:mono_w*2] = full[r0:r1, mono_w:w]
cv2.imwrite("/tmp/diag_grid.png", grid)

print(f"\n  Diagnostic images saved:")
print(f"    /tmp/diag_full_frame.png  (full frame, resized)")
print(f"    /tmp/diag_grid.png        (2-col × 4-row grid)")

# ── Check: does col0 look different from col1? ──
col0_all = full[:, :mono_w]
col1_all = full[:, mono_w:]
col0_gray = cv2.cvtColor(col0_all, cv2.COLOR_BGR2GRAY).astype(np.float32)
col1_gray = cv2.cvtColor(col1_all, cv2.COLOR_BGR2GRAY).astype(np.float32)
diff = np.abs(col0_gray - col1_gray)
print(f"\n  col0 vs col1 (all rows): mean_diff={diff.mean():.1f} max_diff={diff.max():.0f}")

# ── Check: are rows 0,1 same lens vs rows 2,3? ──
r01 = cv2.cvtColor(full[:SUB_H*2, :mono_w], cv2.COLOR_BGR2GRAY).astype(np.float32)
r23 = cv2.cvtColor(full[SUB_H*2:, :mono_w], cv2.COLOR_BGR2GRAY).astype(np.float32)
diff_rows = np.abs(r01 - r23)
print(f"  col0: rows0-1 vs rows2-3: mean_diff={diff_rows.mean():.1f} max_diff={diff_rows.max():.0f}")

_fn_destroy(handle)
print("\nDone.")

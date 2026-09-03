#!/usr/bin/env python3
"""
S80M 标定导出 —— 从设备 ROM 读取出厂标定，导出到平台会话/设备级标定文件。

后端反馈确认（2026-08-13，二进制符号走查）：
    libfays_vikit.so 内置完整标定系统，出厂标定烧在设备 ROM：
      FAYS_VIK_GetCalibrationParam(handle, AtrakCalibrationParam*)  ← 读标定
      FAYS_VIK_DumpCalib(handle, path)                              ← 导出 YAML
    但 SDK 示例 read_stereo_rgb.py 只调 GetStereoFrames/GetImuData，
    没导出标定 —— 这正是会话 calibration/head_stereo.json 内参全零的原因。

本脚本补齐该链路（采集端职责）：
    连相机 → GetCalibrationParam（+DumpCalib 原始 YAML 备查）
          → 写 <session>/calibration/head_stereo.json（平台格式，平台 worker 可直接消费）
          → 写 config/s80m_stereo_calibration.json（设备级默认，三角化回退链用）

只调用 Create/GetCalibrationParam/DumpCalib/Destroy，绝不碰 GetStereoFrames
（SDK 3.9.1 在 RGB 失败后调用它有段错误风险；本模块保持 3.9.0）。

用法:
    python stereo_s80m/export_calibration.py                 # 设备级 → config/
    python stereo_s80m/export_calibration.py --session data/recordings/222/222_000002
        # 会话级 → calibration/head_stereo.json + calibration/s80m_dump_calib.yaml
    python stereo_s80m/export_calibration.py --output /tmp/calib.json
"""

import argparse
import ctypes
import json
import os
import sys
import time
from ctypes import (POINTER, Structure, byref, c_char_p, c_double, c_float,
                    c_int, c_uint8, c_uint32, c_void_p)

SDK_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SDK_DIR, "lib", "fays_atrak", "x86_64", "Release", "libfays_vikit.so")
CONFIG_PATH = os.path.join(SDK_DIR, "config", "fays_vikit.yaml")
TARGET_RESOLUTION = [1280, 800]        # 录制分辨率 (stereo_left/right.mp4)
DEFAULT_OUTPUT = os.path.join(os.path.dirname(SDK_DIR), "config", "s80m_stereo_calibration.json")

# 畸变模型枚举 (fays_atrak_types.h ATRAK_DISTORTION_MODEL) → 三角化模块字符串
DISTORTION_MODEL_NAMES = {
    0: "none",             # ADM_NONE
    1: "equidistant",      # ADM_KB4 — Kannala-Brandt k1..k4 (OpenCV fisheye 同款)
    2: "radtan",           # ADM_RADTAN — k1,k2,p1,p2
    3: "brown_conrady",    # ADM_BROWN_CONRADY
    4: "cvbasic",          # ADM_CVBASIC — OpenCV 8 参数
}


# ── SDK 结构体 (对应 fays_atrak_types.h) ────────────────────────
class AtrakIntrinsics(Structure):
    _fields_ = [
        ("cam_model", c_uint8), ("width", c_uint32), ("height", c_uint32),
        ("fx", c_float), ("fy", c_float), ("cx", c_float), ("cy", c_float),
        ("intrinsic_extra", c_float * 4), ("distortion_model", c_uint8),
        ("distortion", c_float * 8),
    ]


class AtrakExtrinsics(Structure):
    _fields_ = [("rot", c_float * 9), ("trans", c_float * 3)]


class AtrakCamParam(Structure):
    _fields_ = [
        ("cam_id", c_uint8), ("available_mask", c_uint8),
        ("intrinsics", AtrakIntrinsics), ("T_cn_cnm1", AtrakExtrinsics),
        ("T_cn_imu", AtrakExtrinsics), ("timeshift_cam_imu", c_double),
    ]


FAYS_ATRAK_MAX_CAMERAS = 3


class AtrakCamChainParam(Structure):
    _fields_ = [("num_of_cams", c_uint32), ("downsize_ratio", c_uint32),
                ("cameras", AtrakCamParam * FAYS_ATRAK_MAX_CAMERAS)]


class AtrakImuParam(Structure):
    _fields_ = [("accelerometer_noise_density", c_double),
                ("accelerometer_random_walk", c_double),
                ("gyroscope_noise_density", c_double),
                ("gyroscope_random_walk", c_double), ("update_rate", c_float)]


class AtrakCalibrationParam(Structure):
    _fields_ = [("cameras", AtrakCamChainParam), ("imu", AtrakImuParam)]


# ── 加载 SDK ───────────────────────────────────────────────────
def _load_sdk():
    if not os.path.isfile(LIB_PATH):
        raise FileNotFoundError(f"找不到 SDK 库: {LIB_PATH}")
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(f"找不到配置文件: {CONFIG_PATH}")
    # 预加载 OpenCV（libfays_vikit.so 依赖 cv:: 符号；此处仅为加载安全，不取帧）
    for _lib in ["libopencv_core", "libopencv_imgproc", "libopencv_imgcodecs", "libopencv_highgui"]:
        try:
            ctypes.CDLL(_lib + ".so.406", mode=ctypes.RTLD_GLOBAL)
        except OSError:
            try:
                ctypes.CDLL(_lib + ".so.4.2", mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    lib = ctypes.CDLL(LIB_PATH)
    fn_create = getattr(lib, "_Z31FAYS_VIK_CreateHandleWithConfigPPvPKc")
    fn_destroy = getattr(lib, "_Z22FAYS_VIK_DestroyHandlePv")
    fn_calib = getattr(lib, "_Z28FAYS_VIK_GetCalibrationParamPvP21AtrakCalibrationParam")
    fn_dump = getattr(lib, "_Z18FAYS_VIK_DumpCalibPvPKc")
    fn_create.argtypes = [POINTER(c_void_p), c_char_p]
    fn_create.restype = c_int
    fn_destroy.argtypes = [c_void_p]
    fn_destroy.restype = c_int
    fn_calib.argtypes = [c_void_p, POINTER(AtrakCalibrationParam)]
    fn_calib.restype = c_int
    fn_dump.argtypes = [c_void_p, c_char_p]
    fn_dump.restype = c_int
    return lib, fn_create, fn_destroy, fn_calib, fn_dump


# ── 序列化 → 平台格式 JSON ─────────────────────────────────────
def calibration_to_dict(cal: AtrakCalibrationParam) -> dict:
    """把 SDK AtrakCalibrationParam 转成平台 head_stereo.json 格式（含三角化扩展字段）。

    内参按 640x400 → 1280x800 等比缩放（SDK 上报原生分辨率，录像是 2x）。
    """
    cams = cal.cameras.cameras
    used = [cams[i] for i in range(cal.cameras.num_of_cams)]
    if len(used) < 2:
        raise RuntimeError(f"相机链不足 2 目: num_of_cams={cal.cameras.num_of_cams}")

    cam0, cam1 = used[0], used[1]
    native_w = int(cam0.intrinsics.width) or 640
    scale = TARGET_RESOLUTION[0] / native_w
    if abs(scale - TARGET_RESOLUTION[1] / max(int(cam0.intrinsics.height), 1)) > 1e-3:
        print(f"[WARN] SDK 上报分辨率 {cam0.intrinsics.width}x{cam0.intrinsics.height} "
              f"与目标 {TARGET_RESOLUTION} 宽高比不一致，取宽度缩放 {scale:.3f}")

    def _cam(cam: AtrakCamParam) -> dict:
        intr = cam.intrinsics
        return {
            "intrinsic": [float(intr.fx * scale), float(intr.fy * scale),
                          float(intr.cx * scale), float(intr.cy * scale)],
            "distortion": [float(x) for x in intr.distortion],
            "distortion_model": DISTORTION_MODEL_NAMES.get(int(intr.distortion_model), "unknown"),
            "resolution": [int(intr.width), int(intr.height)],
        }

    t = [float(x) for x in cam1.T_cn_cnm1.trans]
    baseline = (sum(x * x for x in t)) ** 0.5

    calib = {
        "type": "stereo_rgbd_camera",
        "name": "head_stereo",
        "resolution": list(TARGET_RESOLUTION),
        "fps": 25.0,
        "baseline": baseline,
        "left_camera": _cam(cam0),
        "right_camera": _cam(cam1),
        "depth_scale": 0.001,
        "cam_imu_timeshift": float(cam0.timeshift_cam_imu),
        # 双目三角化扩展字段（行主序 rot[9]，SDK 头文件 AtrakExtrinsics）
        "rotation": [float(x) for x in cam1.T_cn_cnm1.rot],
        "translation": t,
        # IMU 噪声参数（供 VIO/后端使用）
        "imu": {
            "accelerometer_noise_density": float(cal.imu.accelerometer_noise_density),
            "accelerometer_random_walk": float(cal.imu.accelerometer_random_walk),
            "gyroscope_noise_density": float(cal.imu.gyroscope_noise_density),
            "gyroscope_random_walk": float(cal.imu.gyroscope_random_walk),
            "update_rate": float(cal.imu.update_rate),
        },
        "source": "fays_sdk_device_rom",
        "captured_at": time.time(),
    }
    return calib


def is_usable(calib: dict) -> bool:
    """与 stereo_triangulate.is_usable 同逻辑（包内自包含，不依赖 cv2）。"""
    try:
        for side in ("left_camera", "right_camera"):
            fx, fy = calib[side]["intrinsic"][0], calib[side]["intrinsic"][1]
            if not (fx > 0 and fy > 0):
                return False
        R = calib.get("rotation") or []
        t = calib.get("translation") or []
        if len(R) != 9 or len(t) != 3:
            return False
        if all(isinstance(v, (int, float)) for v in t) and sum(v * v for v in t) < 1e-4:
            return False
        return True
    except (KeyError, TypeError, IndexError):
        return False


# ── 主流程 ─────────────────────────────────────────────────────
def export(calib_path: str, yaml_path: str = None, device_output: str = None) -> dict:
    lib, fn_create, fn_destroy, fn_calib, fn_dump = _load_sdk()
    handle = c_void_p()
    ret = fn_create(byref(handle), CONFIG_PATH.encode())
    if ret != 0:
        raise RuntimeError(f"SDK 初始化失败 ret={ret}（设备节点是否匹配 config/fays_vikit.yaml？）")
    try:
        cal = AtrakCalibrationParam()
        if fn_calib(handle, byref(cal)) != 0:
            raise RuntimeError("FAYS_VIK_GetCalibrationParam 失败")
        calib = calibration_to_dict(cal)

        os.makedirs(os.path.dirname(calib_path), exist_ok=True)
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2, ensure_ascii=False)

        if yaml_path:                      # DumpCalib 原始 YAML（备查/备份）
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            ret = fn_dump(handle, yaml_path.encode())
            if ret == 0 and os.path.isfile(yaml_path) and os.path.getsize(yaml_path) > 0:
                pass
            else:
                print(f"[WARN] FAYS_VIK_DumpCalib 失败 ret={ret}（跳过 YAML 备份，JSON 已含全部参数）")
                yaml_path = None

        if device_output:                  # 设备级副本（三角化回退链）
            with open(device_output, "w", encoding="utf-8") as f:
                json.dump(calib, f, indent=2, ensure_ascii=False)
        return calib, yaml_path
    finally:
        fn_destroy(handle)


def main():
    ap = argparse.ArgumentParser(description="S80M 标定导出（设备 ROM → 平台 JSON/YAML）")
    ap.add_argument("--session", default=None, help="会话目录: 写 <session>/calibration/head_stereo.json (+s80m_dump_calib.yaml)")
    ap.add_argument("--output", default=None, help="设备级 JSON 路径（默认 <repo>/config/s80m_stereo_calibration.json）")
    args = ap.parse_args()

    if not args.session and not args.output:
        args.output = DEFAULT_OUTPUT

    calib_path = None
    yaml_path = None
    if args.session:
        calib_dir = os.path.join(args.session, "calibration")
        calib_path = os.path.join(calib_dir, "head_stereo.json")
        yaml_path = os.path.join(calib_dir, "s80m_dump_calib.yaml")

    try:
        calib, dumped = export(calib_path, yaml_path, args.output)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    l, r = calib["left_camera"], calib["right_camera"]
    print("S80M 设备 ROM 标定摘要:")
    print(f"  来源: FAYS_VIK_GetCalibrationParam（设备 ROM, SDK {os.path.basename(LIB_PATH)}）")
    print(f"  SDK 原生分辨率: {l['resolution'][0]}x{l['resolution'][1]} → 录制 {calib['resolution'][0]}x{calib['resolution'][1]} (x{calib['resolution'][0]/max(l['resolution'][0],1):.0f})")
    print(f"  畸变模型: {l['distortion_model']} / {r['distortion_model']}")
    print(f"  左目 fx={l['intrinsic'][0]:.1f} fy={l['intrinsic'][1]:.1f} cx={l['intrinsic'][2]:.1f} cy={l['intrinsic'][3]:.1f}")
    print(f"  右目 fx={r['intrinsic'][0]:.1f} fy={r['intrinsic'][1]:.1f} cx={r['intrinsic'][2]:.1f} cy={r['intrinsic'][3]:.1f}")
    print(f"  基线 = {calib['baseline']*1000:.2f} mm   cam_imu_timeshift = {calib['cam_imu_timeshift']}")
    imu = calib.get("imu", {})
    if imu:
        print(f"  IMU: gyro_noise={imu['gyroscope_noise_density']:.5f} acc_noise={imu['accelerometer_noise_density']:.5f} rate={imu['update_rate']}Hz")
    usable = is_usable(calib)
    print(f"  三角化可用: {'✓ 是（双目三角化将自动使用此标定）' if usable else '✗ 否（内参或 R/t 缺失）'}")

    for p in filter(None, [calib_path, dumped, args.output]):
        print(f"  → {p}")


if __name__ == "__main__":
    main()

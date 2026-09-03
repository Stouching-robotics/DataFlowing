#!/usr/bin/env python3
"""
S80M 出厂标定捕获 —— 生成 config/s80m_stereo_calibration.json（设备级标定，供双目三角化独立模块使用）。

主路径（离线，无需连相机）：
    解析 FaysSense VI Kit SDK 发布目录下的出厂标定静态文件 config/calib/calib.yaml，
    提取左右目内参 / 畸变系数 / 外参 T_cn_cnm1，缩放到录制分辨率 (1280x800)，
    写入 config/s80m_stereo_calibration.json。

校验路径（可选，--live，需连相机）：
    通过 ctypes 加载 libfays_vikit.so，只调用 CreateHandleWithConfig /
    GetCalibrationParam / DestroyHandle（绝不调用 GetStereoFrames，
    规避 3.9.1 版本 RGB 失败后段错误的已知问题），与静态文件对比基线/内参。

JSON 扩展字段（本模块 + stereo_s80m/hand_triangulate.py 使用，不动 storage/calibration.py）:
    rotation[9]      — R 左→右, 3×3 行主序 (SDK T_cn_cnm1.rot)
    translation[3]   — t 左→右 (米, SDK T_cn_cnm1.trans)
    distortion_model — "equidistant" / "radtan" 等（决定三角化走 fisheye 还是普通路径）

用法:
    python stereo_s80m/capture_calibration.py
    python stereo_s80m/capture_calibration.py --live
    python stereo_s80m/capture_calibration.py --calib-yaml <path> --output <path>
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# ── 默认路径 ────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FAYS_SDK = os.environ.get("FAYSSENSE_SDK_DIR", "")
DEFAULT_CALIB_YAML = (
    os.path.join(_FAYS_SDK, "config", "calib", "calib.yaml") if _FAYS_SDK
    else "")
DEFAULT_OUTPUT = os.path.join(_REPO_ROOT, "config", "s80m_stereo_calibration.json")
TARGET_RESOLUTION = [1280, 800]   # 录制分辨率 (stereo_left/right.mp4)
DEFAULT_FPS = 25.0


def load_calib_yaml(path: str) -> dict:
    """解析 SDK 出厂标定 YAML（PyYAML，文件无 %YAML 头，可直接解析）。"""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "cam0" not in data or "cam1" not in data:
        raise ValueError(f"{path} 缺少 cam0/cam1 标定条目")
    return data


def _intrinsics_from_cam(cam: dict) -> tuple:
    """返回 (K 3x3, dist 数组, distortion_model, resolution [w,h])。"""
    K = np.array([[cam["intrinsics"][0], 0, cam["intrinsics"][2]],
                  [0, cam["intrinsics"][1], cam["intrinsics"][3]],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array(cam["distortion_coeffs"], dtype=np.float64)
    model = str(cam.get("distortion_model", "")).lower()
    res = [int(cam["resolution"][0]), int(cam["resolution"][1])]
    return K, dist, model, res


def _T_cn_cnm1_from_cam(cam: dict) -> tuple:
    """从 cam1 的 T_cn_cnm1 (4x4) 提取 (R 3x3, t 3x1)。"""
    m = np.array(cam["T_cn_cnm1"], dtype=np.float64)   # 4x4 齐次矩阵
    return m[:3, :3], m[:3, 3]


def build_calibration(yaml_path: str, target_resolution: tuple = None) -> dict:
    """从静态 calib.yaml 构建设备级标定 dict，内参缩放到 target_resolution。"""
    data = load_calib_yaml(yaml_path)
    target_resolution = target_resolution or TARGET_RESOLUTION
    tw, th = target_resolution

    K_l, dist_l, model_l, res_l = _intrinsics_from_cam(data["cam0"])
    K_r, dist_r, model_r, res_r = _intrinsics_from_cam(data["cam1"])

    # 内参按分辨率等比缩放（calib.yaml 是 640x400，录像是 1280x800）
    scale = tw / res_l[0]
    if abs(scale - th / res_l[1]) > 1e-3:
        print(f"[WARN] 源标定宽高比与目标不一致: {res_l} -> {target_resolution}, 取宽度缩放 {scale:.3f}")
    K_l = K_l * scale
    K_l[2, 2] = 1.0
    K_r = K_r * scale
    K_r[2, 2] = 1.0

    R, t = _T_cn_cnm1_from_cam(data["cam1"])
    baseline = float(np.linalg.norm(t))
    model = model_l if model_l else model_r   # 左右目通常同模型

    if model not in ("equidistant", "radtan", "brown_conrady"):
        print(f"[WARN] 未识别的畸变模型 '{model}'，三角化可能不准确")

    calib = {
        "type": "stereo_rgbd_camera",
        "name": "head_stereo",
        "resolution": list(target_resolution),
        "fps": DEFAULT_FPS,
        "baseline": baseline,
        "left_camera": {
            "intrinsic": [float(K_l[0, 0]), float(K_l[1, 1]),
                          float(K_l[0, 2]), float(K_l[1, 2])],
            "distortion": [float(x) for x in dist_l],
            "distortion_model": model,
        },
        "right_camera": {
            "intrinsic": [float(K_r[0, 0]), float(K_r[1, 1]),
                          float(K_r[0, 2]), float(K_r[1, 2])],
            "distortion": [float(x) for x in dist_r],
            "distortion_model": model,
        },
        "depth_scale": 0.001,
        "cam_imu_timeshift": float(data.get("cam0", {}).get("timeshift_cam_imu", -0.0019)),
        # 双目三角化扩展字段（行主序 rot[9]，SDK 头文件 AtrakExtrinsics）
        "rotation": [float(x) for x in R.flatten()],
        "translation": [float(x) for x in t.flatten()],
        "source": "fays_sdk_static_calib",
        "captured_at": time.time(),
    }
    return calib


def verify_with_sdk(calib: dict, vikit_lib: str, config_path: str) -> None:
    """连相机时从 SDK 读出厂标定对比（只调 Create/GetCalibrationParam/Destroy）。

    绝不调用 GetStereoFrames —— 3.9.1 版本在 RGB 失败后有段错误风险。
    """
    import ctypes
    from ctypes import c_void_p, c_char_p, c_float, c_double, c_uint8, c_uint32, POINTER, Structure, byref

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

    if not os.path.isfile(vikit_lib):
        print(f"[WARN] --live 跳过: 找不到 SDK 库 {vikit_lib}")
        return

    lib = ctypes.CDLL(vikit_lib)
    fn_create = getattr(lib, "_Z31FAYS_VIK_CreateHandleWithConfigPPvPKc")
    fn_destroy = getattr(lib, "_Z22FAYS_VIK_DestroyHandlePv")
    fn_calib = getattr(lib, "_Z28FAYS_VIK_GetCalibrationParamPvP21AtrakCalibrationParam")
    fn_create.argtypes = [POINTER(c_void_p), c_char_p]
    fn_create.restype = c_int
    fn_destroy.argtypes = [c_void_p]
    fn_destroy.restype = c_int
    fn_calib.argtypes = [c_void_p, POINTER(AtrakCalibrationParam)]
    fn_calib.restype = c_int

    handle = c_void_p()
    ret = fn_create(byref(handle), config_path.encode())
    if ret != 0:
        print(f"[WARN] --live 跳过: SDK 初始化失败 ret={ret}")
        return
    try:
        cal = AtrakCalibrationParam()
        if fn_calib(handle, byref(cal)) != 0:
            print("[WARN] --live 跳过: GetCalibrationParam 失败")
            return
        cams = cal.cameras.cameras
        fx0, fx1 = cams[0].intrinsics.fx, cams[1].intrinsics.fx
        t = np.array(list(cams[1].T_cn_cnm1.trans), dtype=np.float64)
        live_baseline = float(np.linalg.norm(t))
        print(f"  [SDK live] fx0={fx0:.2f} fx1={fx1:.2f} baseline={live_baseline*1000:.2f}mm")
        # SDK 输出分辨率与静态文件同为 640x400, 录像 1280x800 → ×2 对比
        sdk_scale = 2.0
        d_baseline = abs(live_baseline - calib["baseline"])
        d_fx = abs(fx0 * sdk_scale - calib["left_camera"]["intrinsic"][0]) / calib["left_camera"]["intrinsic"][0]
        if d_baseline > 0.002:
            print(f"  [WARN] 基线偏差 {d_baseline*1000:.1f}mm (>2mm)，以 SDK 实测为准可重跑")
        else:
            print(f"  [OK] 基线一致 ({d_baseline*1000:.1f}mm)")
        if d_fx > 0.01:
            print(f"  [WARN] fx 偏差 {d_fx*100:.1f}% (>1%)")
        else:
            print(f"  [OK] fx 一致 ({d_fx*100:.2f}%)")
    finally:
        fn_destroy(handle)


def main():
    ap = argparse.ArgumentParser(description="S80M 出厂标定捕获（静态 calib.yaml → config/s80m_stereo_calibration.json）")
    ap.add_argument("--calib-yaml", default=os.environ.get("FAYS_CALIB_YAML", DEFAULT_CALIB_YAML))
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--live", action="store_true", help="连相机时从 SDK 校验标定（需 SDK 库与配置）")
    args = ap.parse_args()

    if not os.path.isfile(args.calib_yaml):
        print(f"[ERROR] 找不到出厂标定文件: {args.calib_yaml}")
        print("  请确认 FaysSense_VI_Kit_Release 已安装，或 --calib-yaml 指定路径")
        sys.exit(1)

    calib = build_calibration(args.calib_yaml)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)

    l, r = calib["left_camera"], calib["right_camera"]
    print("S80M 标定摘要:")
    print(f"  来源: {args.calib_yaml}")
    print(f"  分辨率: {calib['resolution']}  畸变模型: {l['distortion_model']}")
    print(f"  左目 fx={l['intrinsic'][0]:.1f} fy={l['intrinsic'][1]:.1f} "
          f"cx={l['intrinsic'][2]:.1f} cy={l['intrinsic'][3]:.1f}")
    print(f"  右目 fx={r['intrinsic'][0]:.1f} fy={r['intrinsic'][1]:.1f} "
          f"cx={r['intrinsic'][2]:.1f} cy={r['intrinsic'][3]:.1f}")
    print(f"  基线 = {calib['baseline']*1000:.2f} mm")
    R = np.array(calib["rotation"], dtype=np.float64).reshape(3, 3)
    det = np.linalg.det(R)
    print(f"  R 正交性: det(R)={det:.6f} ({'OK' if abs(abs(det)-1) < 0.01 else 'WARN'})")
    print(f"  → 已写入 {args.output}")

    if args.live:
        print("\n── SDK 在线校验 ──")
        module_dir = os.path.dirname(os.path.abspath(__file__))
        vikit_lib = os.path.join(module_dir, "lib", "fays_atrak", "x86_64", "Release", "libfays_vikit.so")
        config_path = os.path.join(module_dir, "config", "fays_vikit.yaml")
        verify_with_sdk(calib, vikit_lib, config_path)


if __name__ == "__main__":
    main()

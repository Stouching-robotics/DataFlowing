#!/usr/bin/env python3
"""
S80C/S80M 实测导出：从设备 ROM 读 IMU 标定 5 参数 + 相机标定 YAML + 设备信息。

生成 v3 episode 容器（路径 B）metadata_json 需要的三个原料文件：
    FS-VI80-<model>_<serial>_imu.yaml                  ← 客户 update_fays_calibration.py 同款格式
    FS-VI80-<model>_<serial>_camera_calibration.yaml   ← SDK FAYS_VIK_DumpCalib 原始输出（640×400）
    fays_device.json                                    ← fays_device 字段（model/firmware/serial/...）

为什么需要本工具：FAYS_VIK_GetCalibrationParam 的 IMU 块（噪声密度/随机游走/更新率）
只存在设备 ROM，任何导出文件里都没有（SDK DumpCalib 只写相机链）。

只调用 Create/GetDeviceInfo/GetCalibrationParam/DumpCalib/GetVersion/Destroy，
绝不碰 GetStereoFrames（SDK 3.9.1 在 RGB 失败后调用它有段错误风险）。

用法:
    python tools/stereo_s80m/dump_fays_imu_calibration.py                # 设备级 → config/
    python tools/stereo_s80m/dump_fays_imu_calibration.py --output /tmp  # 指定输出目录
    python tools/stereo_s80m/dump_fays_imu_calibration.py --expect-serial 3500000262190088
        # 序列号不符时报错退出（防拿错相机）
"""

import argparse
import ctypes
import datetime as dt
import json
import os
import re
import sys
import tempfile
from ctypes import (POINTER, Structure, byref, c_char, c_char_p, c_double,
                    c_float, c_int, c_uint8, c_uint32, c_void_p)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SDK_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SDK_DIR, "lib", "fays_atrak", "x86_64", "Release", "libfays_vikit.so")
CONFIG_PATH = os.path.join(SDK_DIR, "config", "fays_vikit.yaml")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "config")


# ── 设备节点自动解析（与 read_stereo_rgb.py 同口径） ────────────
# yaml 写死的 /dev/videoN 会随 USB 插拔漂移；按设备名（FTDI Superspeed
# Video Bridge）+ USB 接口号重写临时 yaml：接口 1.0 = 双目、1.2 = IMU。
def _resolve_ports(config_path):
    ports = {}
    usb_path = None
    usb_speed = None
    for name in sorted(os.listdir("/sys/class/video4linux")):
        vp = os.path.join("/sys/class/video4linux", name)
        try:
            dname = open(os.path.join(vp, "name"), encoding="utf-8").read().strip()
        except OSError:
            continue
        if "FTDI Superspeed Video Bridge" not in dname:
            continue
        try:
            iface = os.path.basename(os.readlink(os.path.join(vp, "device")))
        except OSError:
            continue
        m = re.match(r".+:(\d+\.\d+)$", iface)
        if not m:
            continue
        ports.setdefault(m.group(1), []).append(f"/dev/{name}")
        if usb_path is None:
            usb_path = iface.rsplit(":", 1)[0]
            try:
                spd = open(f"/sys/bus/usb/devices/{usb_path}/speed",
                           encoding="utf-8").read().strip()
                usb_speed = int(spd)
            except (OSError, ValueError):
                usb_speed = None
    stereo = next(iter(ports.get("1.0", [])), None)
    imu = next(iter(ports.get("1.2", [])), None)
    if not stereo:
        raise RuntimeError(
            "未找到 FaysSense 双目设备（FTDI Superspeed Video Bridge 接口 1.0），"
            "请确认 S80C 已连接。")
    tf = tempfile.NamedTemporaryFile(mode="w", suffix="_s80m_ports.yaml", delete=False)
    with open(config_path, encoding="utf-8") as src:
        for line in src:
            if line.startswith("rgb_dev_port"):
                line = "rgb_dev_port: NULL\n"
            elif line.startswith("stereo_dev_port"):
                line = f"stereo_dev_port: {stereo}\n"
            elif line.startswith("imu_dev_port") and imu:
                line = f"imu_dev_port: {imu}\n"
            tf.write(line)
    tf.close()
    return tf.name, {"stereo": stereo, "imu": imu, "usb_path": usb_path,
                     "usb_speed": usb_speed}


# ── SDK 结构体 (对应 fays_atrak_types.h) ──────────────────────
class ViKitDeviceInfo(Structure):
    _fields_ = [
        ("firmware_version", c_char * 16),
        ("device_model", c_char * 16),
        ("serial_number", c_char * 32),
        ("camera_nums", c_uint32),
        ("imu_nums", c_uint32),
        ("reserved", c_uint32 * 32),
    ]


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


class AtrakCamChainParam(Structure):
    _fields_ = [("num_of_cams", c_uint32), ("downsize_ratio", c_uint32),
                ("cameras", AtrakCamParam * 3)]


class AtrakImuParam(Structure):
    _fields_ = [("accelerometer_noise_density", c_double),
                ("accelerometer_random_walk", c_double),
                ("gyroscope_noise_density", c_double),
                ("gyroscope_random_walk", c_double), ("update_rate", c_float)]


class AtrakCalibrationParam(Structure):
    _fields_ = [("cameras", AtrakCamChainParam), ("imu", AtrakImuParam)]


def _load_sdk():
    if not os.path.isfile(LIB_PATH):
        raise FileNotFoundError(f"找不到 SDK 库: {LIB_PATH}")
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(f"找不到配置文件: {CONFIG_PATH}")
    # 预加载 OpenCV（libfays_vikit.so 依赖 cv:: 符号；仅加载安全，不取帧）
    for _lib in ["libopencv_core", "libopencv_imgproc", "libopencv_imgcodecs",
                 "libopencv_highgui"]:
        try:
            ctypes.CDLL(_lib + ".so.406", mode=ctypes.RTLD_GLOBAL)
        except OSError:
            try:
                ctypes.CDLL(_lib + ".so.4.2", mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    lib = ctypes.CDLL(LIB_PATH)
    create = getattr(lib, "_Z31FAYS_VIK_CreateHandleWithConfigPPvPKc")
    get_info = getattr(lib, "_Z22FAYS_VIK_GetDeviceInfoPvP15ViKitDeviceInfo")
    get_calib = getattr(lib, "_Z28FAYS_VIK_GetCalibrationParamPvP21AtrakCalibrationParam")
    dump_calib = getattr(lib, "_Z18FAYS_VIK_DumpCalibPvPKc")
    get_version = getattr(lib, "_Z19FAYS_VIK_GetVersionPv")
    destroy = getattr(lib, "_Z22FAYS_VIK_DestroyHandlePv")
    create.argtypes = [POINTER(c_void_p), c_char_p]
    create.restype = c_int
    get_info.argtypes = [c_void_p, POINTER(ViKitDeviceInfo)]
    get_info.restype = c_int
    get_calib.argtypes = [c_void_p, POINTER(AtrakCalibrationParam)]
    get_calib.restype = c_int
    dump_calib.argtypes = [c_void_p, c_char_p]
    dump_calib.restype = c_int
    get_version.argtypes = [c_void_p]
    get_version.restype = c_char_p
    destroy.argtypes = [c_void_p]
    destroy.restype = c_int
    return lib, create, get_info, get_calib, dump_calib, get_version, destroy


def _decode(b: bytes) -> str:
    return b.split(b"\x00", 1)[0].decode("utf-8", "replace")


def _imu_yaml_text(device, source, imu, serial) -> bytes:
    """客户 update_fays_calibration.py 同款 imu yaml 格式（逐字段对齐）。"""
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    return (
        "%YAML:1.0\n"
        f"# Generated by dump_fays_imu_calibration.py at {now}\n"
        "device:\n"
        f"  model: {device['model']}\n"
        f"  serial_number: {serial}\n"
        f"  camera_count: {device['camera_count']}\n"
        "source:\n"
        "  api: FAYS_VIK_GetCalibrationParam\n"
        f"  physical_usb_path: {source['usb_path'] or 'null'}\n"
        "  ft_index: null\n"
        f"  stereo_dev_port: {source['stereo']}\n"
        f"  imu_dev_port: {source['imu'] or 'null'}\n"
        f"  usb_speed_mbps: {source['usb_speed'] if source['usb_speed'] is not None else 'null'}\n"
        "imu:\n"
        f"  accelerometer_noise_density: {imu['accelerometer_noise_density']:.15g}\n"
        f"  accelerometer_random_walk: {imu['accelerometer_random_walk']:.15g}\n"
        f"  gyroscope_noise_density: {imu['gyroscope_noise_density']:.15g}\n"
        f"  gyroscope_random_walk: {imu['gyroscope_random_walk']:.15g}\n"
        f"  update_rate_hz: {imu['update_rate_hz']:.15g}\n"
    ).encode("utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help=f"输出目录（默认 {DEFAULT_OUTPUT}）")
    ap.add_argument("--expect-serial", default=None, help="序列号不符时报错退出")
    ap.add_argument("--config", default=CONFIG_PATH, help="fays_vikit.yaml 路径（默认工具包内副本）")
    args = ap.parse_args()

    tmp_config, ports = _resolve_ports(args.config)
    print(f"[Ports] 自动解析: stereo={ports['stereo']} imu={ports['imu'] or '?'} "
          f"rgb=NULL", file=sys.stderr)

    lib, create, get_info, get_calib, dump_calib, get_version, destroy = _load_sdk()

    handle = c_void_p()
    status = int(create(byref(handle), os.fsencode(tmp_config)))
    if status != 0 or not handle.value:
        print(f"[ERROR] FAYS_VIK_CreateHandleWithConfig 失败: status={status}", file=sys.stderr)
        sys.exit(1)
    try:
        info = ViKitDeviceInfo()
        st = int(get_info(handle, byref(info)))
        if st != 0:
            print(f"[ERROR] FAYS_VIK_GetDeviceInfo 失败: status={st}", file=sys.stderr)
            sys.exit(1)
        serial = _decode(bytes(info.serial_number))
        model = _decode(bytes(info.device_model))
        firmware = _decode(bytes(info.firmware_version))
        camera_count = int(info.camera_nums)
        imu_count = int(info.imu_nums)
        print(f"[Device] model={model} serial={serial} firmware={firmware} "
              f"cameras={camera_count} imus={imu_count}", file=sys.stderr)
        if args.expect_serial and serial != args.expect_serial:
            print(f"[ERROR] 序列号不符: expected={args.expect_serial} actual={serial}",
                  file=sys.stderr)
            sys.exit(1)

        calib = AtrakCalibrationParam()
        st = int(get_calib(handle, byref(calib)))
        if st != 0:
            print(f"[ERROR] FAYS_VIK_GetCalibrationParam 失败: status={st}", file=sys.stderr)
            sys.exit(1)
        imu = {
            "accelerometer_noise_density": float(calib.imu.accelerometer_noise_density),
            "accelerometer_random_walk": float(calib.imu.accelerometer_random_walk),
            "gyroscope_noise_density": float(calib.imu.gyroscope_noise_density),
            "gyroscope_random_walk": float(calib.imu.gyroscope_random_walk),
            "update_rate_hz": float(calib.imu.update_rate),
        }
        # 防御：全是 0 = SDK 未填充（3.9.x 只填相机链）
        if all(v == 0.0 for v in imu.values()):
            print("[ERROR] IMU 5 参数全为 0（SDK 未从 ROM 填充），无实测值可用",
                  file=sys.stderr)
            sys.exit(1)
        version = get_version(handle)
        sdk_version = _decode(version) if version else ""

        os.makedirs(args.output, exist_ok=True)
        prefix = f"{model}_{serial}"

        imu_path = os.path.join(args.output, prefix + "_imu.yaml")
        with open(imu_path, "wb") as f:
            f.write(_imu_yaml_text(
                {"model": model, "camera_count": camera_count}, ports, imu, serial))

        dev_path = os.path.join(args.output, "fays_device.json")
        with open(dev_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": model, "serial": serial, "firmware": firmware,
                "camera_count": camera_count, "imu_count": imu_count,
                "sdk_version": sdk_version,
                "stereo_dev_port": ports["stereo"], "imu_dev_port": ports["imu"],
                "physical_usb_path": ports["usb_path"],
                "usb_speed_mbps": ports["usb_speed"],
                "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }, f, ensure_ascii=False, indent=2)

        # FAYS_VIK_DumpCalib 接的是**目录**（SDK 在目录内生成
        # FS-VI80-<model>_<serial>_dump_calib.yaml），不是文件路径
        cam_path = os.path.join(args.output, prefix + "_camera_calibration.yaml")
        before = set(os.listdir(args.output))
        st = int(dump_calib(handle, os.fsencode(args.output)))
        dumped = None
        for name in os.listdir(args.output):
            if name.endswith("_dump_calib.yaml") and name not in before:
                dumped = os.path.join(args.output, name)
        if st != 0 or not dumped:
            print(f"[WARN] FAYS_VIK_DumpCalib 失败: status={st}（相机 YAML 未生成，"
                  f"可用既有 dump 经 --camera-yaml 传入导出器）", file=sys.stderr)
        else:
            if os.path.abspath(dumped) != os.path.abspath(cam_path):
                os.replace(dumped, cam_path)
            print(f"[OK] 相机标定: {cam_path}", file=sys.stderr)

        print(f"[OK] IMU 标定: {imu_path}", file=sys.stderr)
        print(f"[OK] 设备信息: {dev_path}", file=sys.stderr)
        print(json.dumps(imu, ensure_ascii=False), file=sys.stderr)
    finally:
        destroy(handle)
        os.unlink(tmp_config)


if __name__ == "__main__":
    main()

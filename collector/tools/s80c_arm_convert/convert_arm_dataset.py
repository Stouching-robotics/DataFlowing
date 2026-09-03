#!/usr/bin/env python3
"""ARM 端 S80C 数据集 → 主程序会话格式 转换器（demo）。

输入（ARM 端，采集侧约定，见 README）:
    dataset/
    ├── meta/info.json             # 自己定义的 schema/fps（尽力解析，字段对不上
    │                              #   时用 --fps 覆盖）
    ├── meta/episodes/*.parquet    # episode 索引（有 episode_index 列就用）
    ├── data/*.parquet             # 逐帧表：frame_index + 时间戳(秒或纳秒) + IMU
    │                              #   列名按常见名回退识别
    └── videos/*.mp4               # 每路相机一个 MP4 shard，内容为灰度
                                   #   = Bayer 阵列当 8-bit 灰度存储的原始流
        （或单个 stacked 上下叠 MP4，自动检测）
    + IMU 二进制: dataset 下任意 *.bin，raw struct 流，每条样本
        8B 时间戳(ns) + 6×8B double(gyro×3 + acc×3)，与 --pipe 协议同款
        （默认大端；parquet 里已有 imu 列时优先用 parquet）

输出（主程序会话，字段级对齐 egodata_writer）:
    <task>_<task>_<ep>/（timestamps.json / data/imu parquet / meta/* / videos/*）

关键转换:
    1. 灰度帧 = Bayer 原始阵列 → cv2 去马赛克 + 白平衡增益 + gamma → 彩色 BGR
       → ffmpeg 重编码 MP4（默认 libx264 ultrafast crf23，与录制同款）
    2. IMU bin → 逐帧窗口挂靠（样本归属第一个 hw_ns ≥ 样本 ts 的帧）
    3. 全部元数据文件按主程序 writer 的 schema 重建

依赖: numpy / opencv-python / pyarrow / imageio-ffmpeg（或 DAQ_FFMPEG 环境变量
      指定 ffmpeg）
"""

import argparse
import glob
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

# ── 采集侧假设（与 S80C 软 ISP 参数对齐，可用命令行覆盖）───────────
# S80C 无硬件 ISP：灰度流就是 Bayer 直出。isp_param.ini 默认 base_bayer=0(RGGB)、
# wb r1.2/g1.0/b1.5、gamma 2.2。若采集端已做过 demosaic 再存灰（色彩不可恢复），
# 本工具无法还原颜色，见 README。
def find_ffmpeg():
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.isfile(exe):
            return exe
    except Exception:
        pass
    for env_name in ("DAQ_FFMPEG",):
        p = os.environ.get(env_name)
        if p and os.path.isfile(p):
            return p
    # 本机已知可用 ffmpeg（conda base 的会因 openvino/tbb 崩）
    local = os.path.expanduser("~/miniconda3/envs/lerobot/bin/ffmpeg")
    if os.path.isfile(local):
        return local
    from shutil import which
    p = which("ffmpeg")
    if p:
        return p
    raise RuntimeError("找不到可用的 ffmpeg（pip install imageio-ffmpeg）")


# ── 输入发现 ─────────────────────────────────────────────────

def find_frame_parquet(dataset_dir):
    """返回 data/ 下含 frame_index 列的 parquet 路径，没有则 None。"""
    import pyarrow.parquet as pq
    for p in sorted(glob.glob(os.path.join(dataset_dir, "data", "**",
                                           "*.parquet"), recursive=True)):
        try:
            if "frame_index" in pq.read_schema(p).names:
                return p
        except Exception:
            continue
    return None


def find_imu_bin(dataset_dir):
    bins = sorted(glob.glob(os.path.join(dataset_dir, "**", "*.bin"),
                            recursive=True))
    return bins[0] if bins else None


def parse_arm_fps(dataset_dir, fallback):
    """尽力从 meta/info.json 解析 fps。"""
    for p in ("meta/info.json", "info.json"):
        fp = os.path.join(dataset_dir, p)
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                info = json.load(f)
            for key in ("fps", "frame_rate", "framerate"):
                if key in info:
                    return float(info[key])
            cams = info.get("cameras") or {}
            vals = [c.get("fps") for c in cams.values()
                    if isinstance(c, dict) and c.get("fps")]
            if vals:
                return float(vals[0])
        except Exception:
            continue
    return fallback


# ── 逐帧时间戳提取（列名回退）────────────────────────────────

def extract_frame_meta(pq_path):
    """从 ARM 逐帧 parquet 提取 frame_index / 帧时间戳 / hardware_ns。

    时间戳单位自动判定: 列名含 ns 或数值 >1e8 视为纳秒 → 转秒。
    返回 dict: {"ts_s": np.float32数组, "hw_ns": uint64数组, "n": 帧数}
    """
    import pyarrow.parquet as pq
    tbl = pq.read_table(pq_path)
    cols = tbl.column_names
    fcol = None
    for c in cols:
        if c.lower().replace("_", "") in ("frameindex", "frameno", "index", "idx"):
            fcol = c
            break
    tcol = None
    for c in cols:
        lc = c.lower()
        if lc in ("timestamp", "ts", "time", "timestamp_s", "timestamp_ns"):
            tcol = c
            break
    hcol = None
    for c in cols:
        lc = c.lower().replace("_", "")
        if lc in ("hardwarens", "hwns", "hwts", "hardwaretimestamp"):
            hcol = c
            break
    n = tbl.num_rows
    if tcol is not None:
        t = tbl[tcol].to_numpy()
        if t.dtype.kind in "fiu":
            t = t.astype(np.float64)
            if np.nanmax(np.abs(t)) > 1e8:      # 纳秒
                t /= 1e9
            ts_s = t.astype(np.float32)
        else:
            ts_s = np.zeros(n, np.float32)
    else:
        ts_s = np.zeros(n, np.float32)
    if hcol is not None:
        hw = tbl[hcol].to_numpy()
        hw_ns = hw.astype(np.uint64) if hw.dtype.kind in "iu" \
            else np.zeros(n, np.uint64)
    else:
        hw_ns = np.zeros(n, np.uint64)
    # 无时间戳列时按 fps 网格回退（在调用方填充）
    return {"ts_s": ts_s, "hw_ns": hw_ns, "n": n}


def extract_parquet_imu(pq_path):
    """若 ARM parquet 已含 IMU 列（imu_ts_ns + observation.imu），直接取出。

    返回 (sample_ts_ns数组, samples数组 N×6 float64)；不完整则 (None, None)。
    """
    import pyarrow.parquet as pq
    tbl = pq.read_table(pq_path)
    cols = tbl.column_names
    if "imu_ts_ns" not in cols or "observation.imu" not in cols:
        return None, None
    ts_all, gyro_all, acc_all = [], [], []
    try:
        for i in range(tbl.num_rows):
            ts = tbl["imu_ts_ns"][i].as_py() or []
            obs = tbl["observation.imu"][i].as_py() or []
            for k, v in enumerate(obs):
                if k < len(ts):
                    ts_all.append(ts[k])
                    gyro_all.append(v[0:3])
                    acc_all.append(v[3:6])
    except Exception:
        return None, None
    if not ts_all:
        return None, None
    return (np.array(ts_all, np.uint64),
            np.hstack([np.array(gyro_all), np.array(acc_all)]))


def parse_imu_bin(path, endian=">"):
    """解析 raw struct IMU 流: 每条 56B = 8B ts(ns) + 6×8B double。

    返回 (ts_ns uint64数组, samples N×6 float64)。
    """
    data = open(path, "rb").read()
    rec = struct.Struct(endian + "Q6d")
    n = len(data) // rec.size
    if n == 0:
        return None, None
    out = np.zeros(n, dtype=[("ts", np.uint64), ("v", np.float64, (6,))])
    for i in range(n):
        vals = rec.unpack_from(data, i * rec.size)
        out[i]["ts"] = vals[0]
        out[i]["v"] = vals[1:]
    return out["ts"].astype(np.uint64), out["v"].astype(np.float64)


# ── 视频: 灰度 Bayer → 彩色 ─────────────────────────────────

def _normalize_eye(name):
    """视频名/目录名 → 输出槽位名；不认识的返回 None。"""
    n = str(name).lower()
    if "left" in n or n.endswith("l") or "cam0" in n or n == "0":
        return "stereo_left"
    if "right" in n or n.endswith("r") or "cam1" in n or n == "1":
        return "stereo_right"
    return None


def load_arm_videos(dataset_dir):
    """发现视频并映射到槽位。

    优先级: 1) meta/info.json 的 cameras.*.video_path（真实 ARM 布局:
    videos/<eye>/chunk-XXX/file-XXX.mp4，video_path 相对 dataset 根）；
    2) videos/ 下首层目录/文件名（left/right/cam0/cam1）；
    3) 无名单文件 → stacked（上下叠），两个无名文件按序当左右目。
    返回 {slot: path}。
    """
    for p in ("meta/info.json", "info.json"):
        fp = os.path.join(dataset_dir, p)
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                info = json.load(f)
            out = {}
            for key, cam in (info.get("cameras") or {}).items():
                if not isinstance(cam, dict):
                    continue
                vp = cam.get("video_path") or f"videos/{key}.mp4"
                full = os.path.join(dataset_dir, vp)
                if not os.path.isfile(full):
                    full = os.path.join(dataset_dir, os.path.basename(vp))
                if os.path.isfile(full):
                    out.setdefault(_normalize_eye(key) or "stacked", full)
            if out:
                return out
        except Exception:
            pass
    out = {}
    for p in sorted(glob.glob(os.path.join(dataset_dir, "videos", "**",
                                           "*.mp4"), recursive=True)):
        rel = os.path.relpath(p, os.path.join(dataset_dir, "videos"))
        slot = _normalize_eye(rel.split(os.sep)[0])
        if slot:
            out.setdefault(slot, p)
        elif "stacked" not in out:
            out["stacked"] = p
        else:      # 第二个无名文件：按序拆成左右目
            out.setdefault("stereo_left", out.pop("stacked"))
            out.setdefault("stereo_right", p)
    return out


def load_arm_timestamps(dataset_dir):
    """data/timestamps.json（ARM 端逐帧时间戳）→ hw_ns + 相对秒。

    真实布局: {"frame_ts_ns": [...], "fps": ..., "frame_count": ...}。
    返回 (hw_ns uint64数组, ts_s float32数组) 或 (None, None)。
    """
    for p in ("data/timestamps.json", "timestamps.json"):
        fp = os.path.join(dataset_dir, p)
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                tj = json.load(f)
            ts = np.array(tj["frame_ts_ns"], dtype=np.uint64)
            if len(ts) == 0:
                return None, None
            ts_s = ((ts - ts[0]) / 1e9).astype(np.float32)
            return ts, ts_s
        except Exception:
            continue
    return None, None


# OpenCV Bayer 常量名 → 真实阵列名（OpenCV 按「第二行」命名，见 demosaic_gray）
_CV_BAYER_TO_TRUE = {
    "bg2bgr": "rggb", "gb2bgr": "grbg", "gr2bgr": "gbrg", "rg2bgr": "bggr",
}

# Bayer 相位翻转（R/B 位点互换）：rggb↔bggr、grbg↔gbrg
_PHASE_FLIP = {"rggb": "bggr", "bggr": "rggb", "grbg": "gbrg", "gbrg": "grbg"}


def resolve_bayer_from_info(dataset_dir):
    """info.json 的 device.bayer_pattern（或 cameras.*.bayer）→ 真实阵列名。

    两种写法都认: 真实阵列名（rggb/grbg/gbrg/bggr）或 OpenCV 常量名
    （BG2BGR 等）。返回 None 表示读不到。
    """
    for p in ("meta/info.json", "info.json"):
        fp = os.path.join(dataset_dir, p)
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                info = json.load(f)
            val = None
            dev = info.get("device") or {}
            if isinstance(dev, dict):
                val = dev.get("bayer_pattern") or dev.get("bayer")
            if not val:
                for cam in (info.get("cameras") or {}).values():
                    if isinstance(cam, dict) and cam.get("bayer"):
                        val = cam["bayer"]
                        break
            if val:
                v = (str(val).lower().replace("color_", "")
                     .replace("color", "").strip())
                if v in ("rggb", "grbg", "gbrg", "bggr"):
                    return v
                if v in _CV_BAYER_TO_TRUE:
                    return _CV_BAYER_TO_TRUE[v]
        except Exception:
            continue
    return None


def demosaic_gray(gray, bayer, wb, gamma, demosaiced):
    """Bayer 灰度 → BGR 彩色（白平衡+gamma）。demosaiced=True 表示
    采集端已经解过 Bayer（纯灰度，只能当灰度用）。"""
    import cv2
    if demosaiced:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # ⚠ OpenCV 的 Bayer 常量按「第二行」命名（历史坑）：真实阵列（顶左像素）
    # rggb 需用 COLOR_BayerBG2BGR 才不红蓝互换。下表已用 8×8 合成阵列实测。
    code = {
        "rggb": cv2.COLOR_BayerBG2BGR,
        "grbg": cv2.COLOR_BayerGB2BGR,
        "gbrg": cv2.COLOR_BayerGR2BGR,
        "bggr": cv2.COLOR_BayerRG2BGR,
    }[bayer]
    bgr = cv2.cvtColor(gray, code)
    # 白平衡：按增益比缩放 R/B 通道（G 保持，与 isp_param.ini r1.2/g1.0/b1.5 同口径）
    b, g, r = cv2.split(bgr.astype(np.float32))
    r *= wb[0] / wb[1]
    b *= wb[2] / wb[1]
    bgr = cv2.merge([b, g, r])
    bgr = np.clip(bgr, 0, 255)
    if gamma != 1.0:
        bgr = 255.0 * np.power(bgr / 255.0, 1.0 / gamma)
    return bgr.astype(np.uint8)


# ── SDK 离线 ISP（FAYS_VIK_Offline_*）──────────────────────────────
# S80C 无硬件 ISP，彩色化由 SDK 软 ISP 完成（BLC→WB/AWB→demosaic→CCM→
# gamma）。libfays_vikit.so 3.9.0 导出离线接口，用厂商 vikit yaml 的
# stereo_awb + stereo_R/G/B_gain 上色——与采集端 SDK RGB 模式
# （stereo_color_mode=1）同一份参数源，比纯 OpenCV 猜参数准。
# 实测（2026-08-28）: 改 isp_param.ini 的 wb 输出不变（离线不看 ini），
# 改 yaml 的 awb/增益输出立即变；离线 rotate_180 = Bayer 相位开关
# （rotate=1 按 rggb 相位读该目、rotate=0 按 bggr 相位读，无几何旋转），
# swap_lr 离线惰性（逐字节相同）。录制文件相位 左 bggr/右 rggr → 默认
# yaml rotate 0/1/0（2026-08-31 用户 A/B 裁决钉死）。
# 输入 = 左右目 Bayer 灰度上下堆叠（ATRAK_DEV_STEREO，height=2×单目高），
# 输出 = 同布局 BGR。官方示例: fays_release/tools/offline_process.cpp。

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_VIKIT_LIB = os.path.normpath(os.path.join(
    _TOOL_DIR, "..", "hand_3d_s80c", "third_party", "lib", "libfays_vikit.so"))
_DEFAULT_ISP_YAML = os.path.join(_TOOL_DIR, "config",
                                 "fays_vikit_stereo_rgb.yaml")

# vikit 的 cv:: 符号靠预载 OpenCV 4.2 解析（否则 undefined symbol:
# cv::fastFree）；依赖顺序照抄 s80c_depth_worker（字母序会因 calib3d
# 先于 core 失败）。
_PRELOAD_DEPS = ["libwebp.so.6", "libtiff.so.5", "libtbb.so.2",
                 "libdc1394.so.22", "libavcodec.so.58", "libavformat.so.58",
                 "libavutil.so.56", "libswscale.so.5"]
_PRELOAD_MODS = ["libopencv_core", "libopencv_imgproc", "libopencv_imgcodecs",
                 "libopencv_highgui", "libopencv_calib3d", "libopencv_ximgproc",
                 "libopencv_features2d", "libopencv_flann", "libopencv_video",
                 "libopencv_videoio", "libopencv_photo", "libopencv_objdetect",
                 "libopencv_ml"]


def _preload_opencv406(lib406):
    import ctypes
    for dep in _PRELOAD_DEPS:
        p = os.path.join(lib406, dep)
        if os.path.exists(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    for mod in _PRELOAD_MODS:
        for suffix in (".so.406", ".so.4.2"):
            p = os.path.join(lib406, mod + suffix)
            if os.path.exists(p):
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                    break
                except OSError:
                    pass


class SdkIsp:
    """ctypes 封装 SDK 离线彩色化（每帧对左右目堆叠输入，出同布局 BGR）。"""

    def __init__(self, lib_path, yaml_path, awb=None, gains=None):
        import ctypes
        from ctypes import (POINTER, Structure, c_int, c_short, c_uint,
                            c_ulonglong, c_ubyte, c_void_p, c_char_p)

        class AtrakImage(Structure):
            _fields_ = [
                ("device_id", c_int),
                ("seq", c_int),
                ("timestamp", c_ulonglong),
                ("width", c_int),
                ("height", c_int),
                ("channel", c_int),
                ("encoding", c_short),
                ("step", c_int),
                ("bytes", c_uint),
                ("data", POINTER(c_ubyte)),
            ]

        if not os.path.isfile(lib_path):
            raise FileNotFoundError(f"SDK 库不存在: {lib_path}")
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"ISP yaml 不存在: {yaml_path}")
        # 运行时 yaml：awb/增益可覆盖 → 写临时文件（SDK 接口只收文件路径）
        with open(yaml_path, encoding="utf-8") as f:
            text = f.read()
        if awb is not None:
            text = re.sub(r"stereo_awb:\s*\d", f"stereo_awb: {awb}", text)
        if gains is not None:
            text = re.sub(r"stereo_R_gain:\s*[0-9.]+",
                          f"stereo_R_gain: {gains[0]}", text)
            text = re.sub(r"stereo_G_gain:\s*[0-9.]+",
                          f"stereo_G_gain: {gains[1]}", text)
            text = re.sub(r"stereo_B_gain:\s*[0-9.]+",
                          f"stereo_B_gain: {gains[2]}", text)
        fd, self._tmp_yaml = tempfile.mkstemp(suffix=".yaml", prefix="sdk_isp_")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        lib406 = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(lib_path)), "..",
            "opencv4.2", "lib406"))
        if os.path.isdir(lib406):
            _preload_opencv406(lib406)
        self._vikit = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        self._AtrakImage = AtrakImage
        self._create = getattr(
            self._vikit, "_Z39FAYS_VIK_Offline_CreateHandleWithConfigPPvPKc")
        self._create.restype = c_int
        self._create.argtypes = [POINTER(c_void_p), c_char_p]
        self._proc = getattr(
            self._vikit, "_Z29FAYS_VIK_Offline_ProcessFramePvPK10AtrakImagePS0_i")
        self._proc.restype = c_int
        self._proc.argtypes = [c_void_p, POINTER(AtrakImage),
                               POINTER(AtrakImage), c_int]
        self._destroy = getattr(self._vikit, "_Z30FAYS_VIK_Offline_DestroyHandlePv")
        self._destroy.restype = c_int
        self._destroy.argtypes = [c_void_p]
        self._handle = c_void_p()
        rc = self._create(ctypes.byref(self._handle), self._tmp_yaml.encode())
        if rc != 0:
            raise RuntimeError(
                f"FAYS_VIK_Offline_CreateHandleWithConfig 失败 rc={rc}")
        self._size = None
        self._in = self._out = None
        self._inp = self._outp = None

    def colorize_pair(self, gray_l, gray_r, ts=0):
        """左右目 Bayer 灰度（同尺寸）→ (bgr_l, bgr_r) 视图。

        返回的数组是复用缓冲的视图，下次调用前有效（写入方必须即时消费）。
        """
        import ctypes
        h, w = gray_l.shape
        if (w, h) != self._size:
            self._size = (w, h)
            self._in = np.zeros(w * h * 2, np.uint8)
            self._out = np.zeros(3840 * 2160 * 3, np.uint8)
            self._inp = self._AtrakImage()
            self._inp.device_id = 0          # ATRAK_DEV_STEREO
            self._inp.width, self._inp.height, self._inp.channel = w, h * 2, 1
            self._inp.encoding, self._inp.step = 2, w   # AIE_RAW8
            self._inp.bytes = w * h * 2
            self._inp.data = self._in.ctypes.data_as(
                ctypes.POINTER(ctypes.c_ubyte))
            self._outp = self._AtrakImage()
            self._outp.data = self._out.ctypes.data_as(
                ctypes.POINTER(ctypes.c_ubyte))
        self._in[: w * h] = gray_l.ravel()
        self._in[w * h:] = gray_r.ravel()
        self._inp.timestamp = int(ts)
        rc = self._proc(self._handle, ctypes.byref(self._inp),
                        ctypes.byref(self._outp), 1000)
        if rc != 0:
            raise RuntimeError(f"FAYS_VIK_Offline_ProcessFrame 失败 rc={rc}")
        if self._outp.width == 0 or self._outp.height == 0:
            # SDK 对无纹理/全平场会 rc=0 但返回空输出（软拒），防静默 NaN
            raise RuntimeError("SDK 返回空输出（帧被离线管线拒绝）")
        out = self._out[: self._outp.width * self._outp.height * 3].reshape(
            self._outp.height, self._outp.width, 3)
        # ★ 字节序按输出 struct 的 encoding 字段裁决（2026-08-28 用户口径，
        #   2026-08-31 用户 A/B 裁决 B=红蓝互换为对后按色度口径复核）：
        #   输出 AtrakImage.encoding 声称 AIE_BGR8（枚举 0，头文件注释
        #   "BGR-packed if 3-channels"，实测运行时常量 0）→ 字节直出即 BGR，
        #   与写入端 rawvideo bgr24 一致。若某版 SDK 声称 AIE_RGB8 则换通道
        #   成 BGR；其余值报错。
        #   ⚠ 「颜色反了」的根因不在字节序而在**输入相位**：ARM 录制文件
        #   左目 bggr / 右目 rggr（旧 RGGB 三证只 pin 住 G 相位，R/B 盲区），
        #   驱动逐字 1/0/1 会把双眼都读反 → 输出整体红蓝互换（用户裁决 B）。
        #   修法在 yaml：录制文件用 rotate 0/1/0（默认 yaml 已改）；原始
        #   传感器流才用 1/0/1（fays_vikit_raw_sensor.yaml）。此处字节序
        #   一律以 encoding 字段为准直出，不硬翻。
        enc = self._outp.encoding
        if enc == 0:                      # AIE_BGR8 → 直出
            return out[:h], out[h:]
        if enc == 1:                      # AIE_RGB8 → 换通道成 BGR
            bgr = out[:, :, ::-1]
            return bgr[:h], bgr[h:]
        raise RuntimeError(
            f"SDK 输出 encoding 未知: {enc}（0=AIE_BGR8 / 1=AIE_RGB8）")

    def close(self):
        if getattr(self, "_handle", None):
            self._destroy(self._handle)
            self._handle = None
        if getattr(self, "_tmp_yaml", None):
            try:
                os.unlink(self._tmp_yaml)
            except OSError:
                pass


def _apply_wb_gain(rgb, b_gain, r_gain):
    """SDK 输出后对 B/R 通道施加恒定增益（灰世界精修，去残余偏色）。"""
    if b_gain == 1.0 and r_gain == 1.0:
        return rgb
    out = rgb.astype(np.float32)
    out[..., 0] *= b_gain
    out[..., 2] *= r_gain
    return np.clip(out, 0, 255).astype(np.uint8)


def measure_grayworld_gains(sdk_isp, vids, n_frames, hw_ns, k=10):
    """采样 K 帧过 SDK 上色，按灰世界假设算全局 B/R 增益（目标 G/B=G/R=1）。

    SDK 离线 AWB 对真实场景收敛不彻底（实测 G/R≈1.11-1.13 恒定），残余偏绿
    用全局灰世界增益消除；增益钳制 [0.6, 1.8] 防场景色过矫正。
    ⚠ ts 必须与写帧同口径传真实 hw（SDK 会拒首帧 ts=0，实测）。
    """
    import cv2
    if n_frames <= 2 * k + 10:
        idx = list(range(n_frames))
    else:
        step = (n_frames - 10) / k
        idx = sorted({min(int(5 + i * step), n_frames - 1)
                      for i in range(k)})
    acc = np.zeros(3)
    n = 0
    for i in idx:
        caps = [cv2.VideoCapture(vids[slot]) for slot in
                ("stereo_left", "stereo_right")]
        for c in caps:
            c.set(cv2.CAP_PROP_POS_FRAMES, i)
        frs = [c.read()[1] for c in caps]
        for c in caps:
            c.release()
        if any(f is None for f in frs):
            continue
        ts = int(hw_ns[i]) if hw_ns.any() else 1000 + i   # 首帧 ts≠0
        try:
            l, r = sdk_isp.colorize_pair(
                cv2.cvtColor(frs[0], cv2.COLOR_BGR2GRAY)
                if frs[0].ndim == 3 else frs[0],
                cv2.cvtColor(frs[1], cv2.COLOR_BGR2GRAY)
                if frs[1].ndim == 3 else frs[1],
                ts=ts)
        except RuntimeError:
            continue   # 个别帧被离线管线拒绝 → 跳过不参与统计
        acc += np.concatenate([l, r]).astype(np.float64).mean((0, 1))
        n += 1
    if n == 0:
        return 1.0, 1.0
    b, g, r = acc / n
    b_gain = float(np.clip(g / max(b, 1e-6), 0.6, 1.8))
    r_gain = float(np.clip(g / max(r, 1e-6), 0.6, 1.8))
    return b_gain, r_gain


def write_color_mp4(ffmpeg, frames_iter, size, fps, dst):
    """rawvideo 管道 → ffmpeg 编码 MP4（与录制同款 x264 参数）。"""
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{size[0]}x{size[1]}", "-framerate", str(int(round(fps))),
           "-i", "-", "-fps_mode", "passthrough", "-an",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
           "-pix_fmt", "yuv420p", dst]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n = 0
    for fr in frames_iter:
        proc.stdin.write(fr.astype(np.uint8).tobytes())
        n += 1
    proc.stdin.close()
    proc.wait(timeout=300)
    return n


def write_color_mp4_pair(ffmpeg, pair_iter, size, fps, dst_l, dst_r):
    """成对写左右目 MP4（SDK 离线 ISP 成对处理，两路 ffmpeg 同步喂帧，
    避免 tee 缓冲单边跑飞）。"""
    def _open(dst):
        return subprocess.Popen(
            [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{size[0]}x{size[1]}",
             "-framerate", str(int(round(fps))), "-i", "-",
             "-fps_mode", "passthrough", "-an", "-c:v", "libx264",
             "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
             dst],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)

    pl, pr = _open(dst_l), _open(dst_r)
    n = 0
    try:
        for fr_l, fr_r in pair_iter:
            pl.stdin.write(fr_l.astype(np.uint8).tobytes())
            pr.stdin.write(fr_r.astype(np.uint8).tobytes())
            n += 1
    finally:
        pl.stdin.close()
        pr.stdin.close()
    pl.wait(timeout=300)
    pr.wait(timeout=300)
    return n


# ── 输出写出（字段级对齐 egodata_writer）──────────────────────

def write_outputs(out_dir, task_name, episode_index, fps, n_frames,
                  ts_s, hw_ns, imu_ts, imu_vals, cam_meta,
                  calib_src=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    created = time.time()

    # timestamps.json: 每帧左右目两条，hw 相同 → 稳定排序保证成对相邻
    entries = []
    for i in range(n_frames):
        for _ in range(2):
            e = {"frame_index": int(i),
                 "timestamp": float(ts_s[i]),
                 "wall_time": created + float(ts_s[i])}
            if hw_ns[i]:
                e["hardware_ns"] = int(hw_ns[i])
            entries.append(e)
    if hw_ns.any():
        entries.sort(key=lambda e: (e.get("hardware_ns") is None,
                                    e.get("hardware_ns", 0)))
    with open(os.path.join(out_dir, "timestamps.json"), "w",
              encoding="utf-8") as f:
        json.dump({"timestamps": entries, "total_frames": n_frames},
                  f, indent=2, ensure_ascii=False)

    # IMU 样本 → 逐帧窗口。与主程序 writer 同口径：样本随帧送达，
    # 归属 ts ∈ (hw_{i-1}, hw_i] 的帧 i（SDK 实测样本 ts 比帧 hw 早 21-43ms）
    imu_rows = []
    if imu_ts is not None and len(imu_ts):
        dropped = int((imu_ts > hw_ns[n_frames - 1]).sum())
        if dropped:
            print(f"[WARN] {dropped} 个 IMU 样本晚于末帧 hw（无承载帧，与主程序"
                  f" writer 同口径丢弃）", file=sys.stderr)
        for i in range(n_frames):
            if not hw_ns[i]:
                continue
            if n_frames == 1:
                sel = imu_ts <= hw_ns[i]
            elif i == 0:
                sel = imu_ts <= hw_ns[i]   # 含启动期样本（ts < 首帧 hw）
            elif i == n_frames - 1:
                sel = (imu_ts > hw_ns[i - 1]) & (imu_ts <= hw_ns[i])
            else:
                sel = (imu_ts > hw_ns[i - 1]) & (imu_ts <= hw_ns[i])
            idxs = np.nonzero(sel)[0]
            if not len(idxs):
                continue
            imu_rows.append({
                "episode_index": episode_index,
                "frame_index": int(i),
                "timestamp": float(ts_s[i]),
                "task_index": 0,
                "hardware_ns": int(hw_ns[i]),
                "imu_ts_ns": [int(v) for v in imu_ts[idxs]],
                "observation.imu": [
                    [float(x) for x in imu_vals[j]] for j in idxs],
            })
    if imu_rows:
        imu_schema = pa.schema([
            ("episode_index", pa.int64()),
            ("frame_index", pa.int64()),
            ("timestamp", pa.float32()),
            ("task_index", pa.int64()),
            ("hardware_ns", pa.int64()),
            ("imu_ts_ns", pa.list_(pa.int64())),
            ("observation.imu", pa.list_(pa.list_(pa.float32(), 6))),
        ])
        table = pa.table({
            "episode_index": pa.array([r["episode_index"] for r in imu_rows],
                                      pa.int64()),
            "frame_index": pa.array([r["frame_index"] for r in imu_rows],
                                    pa.int64()),
            "timestamp": pa.array([r["timestamp"] for r in imu_rows],
                                  pa.float32()),
            "task_index": pa.array([r["task_index"] for r in imu_rows],
                                   pa.int64()),
            "hardware_ns": pa.array([r["hardware_ns"] for r in imu_rows],
                                    pa.int64()),
            "imu_ts_ns": pa.array([r["imu_ts_ns"] for r in imu_rows],
                                  pa.list_(pa.int64())),
            "observation.imu": pa.array(
                [r["observation.imu"] for r in imu_rows],
                pa.list_(pa.list_(pa.float32(), 6))),
        }, schema=imu_schema)
        imu_dir = os.path.join(out_dir, "data", "imu", "chunk-0000")
        os.makedirs(imu_dir, exist_ok=True)
        pq.write_table(table, os.path.join(imu_dir, "chunk_000000.parquet"),
                       compression="zstd")

    # meta/episodes 两个 parquet（LeRobot v3 兼容）
    ep_schema = pa.schema([
        ("episode_index", pa.int64()),
        ("task_index", pa.int64()),
        ("start_frame_index", pa.int64()),
        ("end_frame_index", pa.int64()),
        ("length", pa.int64()),
    ])
    ep_table = pa.table({
        "episode_index": pa.array([episode_index], pa.int64()),
        "task_index": pa.array([0], pa.int64()),
        "start_frame_index": pa.array([0], pa.int64()),
        "end_frame_index": pa.array([max(0, n_frames - 1)], pa.int64()),
        "length": pa.array([n_frames], pa.int64()),
    }, schema=ep_schema)
    ep_dir = os.path.join(out_dir, "meta", "episodes")
    os.makedirs(ep_dir, exist_ok=True)
    pq.write_table(ep_table, os.path.join(ep_dir, "chunk_000000.parquet"),
                   compression="zstd")
    chunk_d = os.path.join(ep_dir, "chunk-000")
    os.makedirs(chunk_d, exist_ok=True)
    pq.write_table(ep_table, os.path.join(chunk_d, "file-000.parquet"),
                   compression="zstd")

    # info.json
    cameras_dict = {name: {"height": h, "width": w}
                    for name, (h, w) in cam_meta.items()}
    info = {
        "codebase_version": "v3.0",
        "fps": float(fps),
        "video": bool(cam_meta),
        "task_name": task_name,
        "features": {
            "observation.imu": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [1]},
        },
        "cameras": cameras_dict,
        "devices": [{"key": "s80m:ftdi", "kind": "s80m",
                     "name": "FaysSense S80M",
                     "slots": list(cam_meta.keys())}],
        "device_names": {n: "FaysSense S80M" for n in cam_meta},
        "sensors": [],
        "sensor_dim": 256,
        "created_at": created,
    }
    meta_dir = os.path.join(out_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # stats.json（IMU 真实统计；无 IMU 时零）
    if imu_vals is not None and len(imu_vals):
        im = np.asarray(imu_vals, np.float64)
        imu_stats = {"mean": im.mean(0).tolist(), "std": im.std(0).tolist(),
                     "min": im.min(0).tolist(), "max": im.max(0).tolist()}
    else:
        z = [0.0] * 6
        imu_stats = {"mean": z, "std": z, "min": z, "max": z}
    stats = {"observation.imu": imu_stats,
             "action": {"mean": [0.0], "std": [1.0],
                        "min": [0.0], "max": [0.0]}}
    with open(os.path.join(meta_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # tasks.jsonl
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": task_name},
                           ensure_ascii=False) + "\n")

    # metadata.json
    meta = {
        "format": "egodata",
        "format_version": "1.0",
        "episode_index": episode_index,
        "fps": float(fps),
        "task_name": task_name,
        "cameras": {name: {"height": h, "width": w, "type": "rgb",
                           "device_key": "s80m:ftdi",
                           "device": "FaysSense S80M"}
                    for name, (h, w) in cam_meta.items()},
        "devices": [{
            "key": "s80m:ftdi", "kind": "s80m", "name": "FaysSense S80M",
            "slots": list(cam_meta.keys()),
            "resolution": {n: [w, h] for n, (h, w) in cam_meta.items()},
            "fps": {n: fps for n in cam_meta},
            "calibration": ("calibration/head_stereo.json"
                            if calib_src else None),
        }],
        "sensors": [],
        "sensor_dim": 256,
        "created_at": created,
        "codebase_version": "1.0.5",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # calibration（可选）
    if calib_src:
        cal_dir = os.path.join(out_dir, "calibration")
        os.makedirs(cal_dir, exist_ok=True)
        with open(calib_src, encoding="utf-8") as f:
            calib = json.load(f)
        with open(os.path.join(cal_dir, "head_stereo.json"), "w",
                  encoding="utf-8") as f:
            json.dump(calib, f, indent=2, ensure_ascii=False)


# ── 主流程 ─────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="ARM 端 S80C 数据集 → 主程序会话格式（灰度→彩色 + IMU 转换）")
    ap.add_argument("--input", required=True, help="ARM 端 dataset 目录")
    ap.add_argument("--output", required=True, help="输出会话目录")
    ap.add_argument("--task-name", default="s80c_arm",
                    help="任务名（默认 s80c_arm）")
    ap.add_argument("--episode-index", type=int, default=None,
                    help="episode 序号（默认从输入 meta 读取，读不到为 0）")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="输出帧率（默认 30，输入 info.json 有 fps 时优先）")
    ap.add_argument("--bayer", default=None,
                    choices=["rggb", "grbg", "gbrg", "bggr"],
                    help="Bayer 阵列顺序（默认读 info.json 的 bayer_pattern，"
                         "读不到用 rggb）。仅 OpenCV 回退路径用；录制文件口径"
                         "下此值=右目相位，左目默认取相位翻转（--bayer-left "
                         "覆盖）")
    ap.add_argument("--wb", default="1.2,1.0,1.5",
                    help="白平衡增益 r,g,b（默认 isp_param.ini 值）")
    ap.add_argument("--gamma", type=float, default=2.2,
                    help="gamma（默认 2.2，1.0 关闭）")
    ap.add_argument("--no-sdk-isp", action="store_true",
                    help="禁用 SDK 离线 ISP（FAYS_VIK_Offline_*），回退 "
                         "OpenCV demosaic+--wb/--gamma")
    ap.add_argument("--isp-yaml", default=_DEFAULT_ISP_YAML,
                    help=f"SDK ISP 配置 yaml（默认随工具附带的 "
                         f"fays_vikit_stereo_rgb.yaml；采集端有自己那份更准）")
    ap.add_argument("--sdk-awb", type=int, default=None, choices=[0, 1],
                    help="覆盖 yaml 的 stereo_awb（默认不动 yaml）")
    ap.add_argument("--sdk-gains", default=None,
                    help="覆盖 yaml 的 stereo_R/G/B_gain，如 1.0,0.6,1.3"
                         "（默认不动 yaml）")
    ap.add_argument("--sdk-wb-auto", action="store_true",
                    help="SDK 上色后按灰世界自动精修：采样帧统计全局 B/R 增益"
                         "（目标 G/B=G/R=1，钳制 0.6-1.8），去 SDK 离线 AWB 的"
                         "残余偏绿")
    ap.add_argument("--sdk-wb-gain", default=None,
                    help="SDK 输出后 B,R 通道手动增益（与 --sdk-wb-auto 互斥），"
                         "如 1.08,1.10")
    ap.add_argument("--stacked", action="store_true",
                    help="灰度 MP4 是上下叠双眼（上左/下右）；默认自动检测")
    ap.add_argument("--already-demosaiced", action="store_true",
                    help="采集端已解 Bayer（纯灰度，只能转灰度 BGR）")
    ap.add_argument("--resize", default="1280x800",
                    help="去马赛克后统一缩放 WxH（默认 1280x800，0 不缩放）")
    ap.add_argument("--rotate-left", type=int, default=0, choices=[0, 180],
                    help="左目额外旋转（度）")
    ap.add_argument("--rotate-right", type=int, default=0, choices=[0, 180],
                    help="右目额外旋转（度）")
    ap.add_argument("--bayer-left", default=None,
                    choices=["rggb", "grbg", "gbrg", "bggr"],
                    help="左目 Bayer 阵列顺序（仅 OpenCV 回退路径；默认=右目"
                         "阵列的相位翻转，即录制文件口径左 bggr/右 rggb；"
                         "原始传感器流数据集用 --bayer-left rggb "
                         "--bayer-right bggr）")
    ap.add_argument("--bayer-right", default=None,
                    choices=["rggb", "grbg", "gbrg", "bggr"],
                    help="右目 Bayer 阵列顺序（仅 OpenCV 回退路径；默认=info/"
                         "--bayer 声明的阵列，即录制文件口径的右目相位）")
    ap.add_argument("--swap", action="store_true", help="交换左右目")
    ap.add_argument("--imu-endian", default=">", choices=[">", "<"],
                    help="IMU bin 字节序（默认大端 >，与 --pipe 协议一致）")
    ap.add_argument("--calib", default=None,
                    help="标定 JSON 路径（写入 calibration/head_stereo.json）")
    args = ap.parse_args()

    import cv2
    dataset = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.output)
    if not os.path.isdir(dataset):
        print(f"输入目录不存在: {dataset}", file=sys.stderr)
        sys.exit(1)

    fps = parse_arm_fps(dataset, args.fps)
    wb = [float(x) for x in args.wb.split(",")]
    ffmpeg = find_ffmpeg()
    # Bayer 阵列：--bayer 显式 > info.json（真实样本写 "BG2BGR" 这类 OpenCV 名）> rggb
    bayer = args.bayer or resolve_bayer_from_info(dataset) or "rggb"
    # ★ 分目相位（2026-08-31 用户 A/B 裁决钉死，仅 OpenCV 回退路径使用；
    #   SDK 路径的相位在 yaml rotate 0/1/0 里）：ARM 录制文件的存储相位为
    #   左目 bggr / 右目 rggr——左目=右目相位的翻转（采集端把左目物理倒装
    #   几何烘焙成 180° 旋转，Bayer 相位随之 rggb→bggr；旧「RGGB 三证」
    #   只 pin 住 G 相位，R/B 是盲区，被用户肉眼裁决推翻）。
    #   info/--bayer 声明的阵列 = 右目存储相位；左目默认取翻转，可用
    #   --bayer-left/--bayer-right 显式覆盖（原始传感器流数据集：
    #   --bayer-left rggb --bayer-right bggr + --isp-yaml 原始流 yaml）。
    bayer_r = args.bayer_right or bayer
    bayer_l = args.bayer_left or _PHASE_FLIP[bayer_r]
    print(f"ffmpeg: {ffmpeg}\nfps: {fps}\nbayer: {bayer} "
          f"(左目 {bayer_l} / 右目 {bayer_r})")

    # ── 帧元数据（优先 parquet → data/timestamps.json → fps 网格）──
    imu_ts = imu_vals = None
    pq_path = find_frame_parquet(dataset)
    if pq_path is not None:
        fm = extract_frame_meta(pq_path)
        print(f"帧 parquet: {pq_path} ({fm['n']} 行)")
        imu_ts, imu_vals = extract_parquet_imu(pq_path)
        if imu_ts is not None:
            print(f"IMU 取自 parquet: {len(imu_ts)} 样本")
    else:
        fm = None
        hw_ns, ts_s = load_arm_timestamps(dataset)
        if hw_ns is not None:
            fm = {"ts_s": ts_s, "hw_ns": hw_ns, "n": len(hw_ns)}
            print(f"时间戳取自 data/timestamps.json: {len(hw_ns)} 帧")
        else:
            print("[WARN] data/ 下没找到逐帧时间戳（parquet/timestamps.json），"
                  "按 fps 网格回退", file=sys.stderr)
    if imu_ts is None:
        binp = find_imu_bin(dataset)
        if binp:
            imu_ts, imu_vals = parse_imu_bin(binp, args.imu_endian)
            print(f"IMU 取自 bin: {binp} ({len(imu_ts) if imu_ts is not None else 0} 样本)")
        else:
            print("[WARN] 未找到 IMU 数据", file=sys.stderr)

    # ── 视频发现与转换 ──
    vids = load_arm_videos(dataset)
    if not vids:
        print(f"[ERROR] videos/ 下没有 MP4: {dataset}", file=sys.stderr)
        sys.exit(1)
    print(f"输入视频: {vids}")

    cam_meta = {}
    # stacked 判定：videos/ 下存在既非 left/right 也非 cam0/cam1 命名的单文件
    # （真实 S80C stacked 是 1280×802，高宽比 0.63，几何启发式不可靠）
    stacked = args.stacked or "stacked" in vids
    resize = None
    if args.resize != "0":
        try:
            resize = tuple(int(x) for x in args.resize.lower().split("x"))
        except ValueError:
            resize = None

    # 逐路探测帧数与尺寸；n_frames 取视频/时间戳/parquet 三者的最小值
    counts, sizes = {}, {}
    for slot, path in vids.items():
        probe = cv2.VideoCapture(path)
        counts[slot] = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        sizes[slot] = (int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        probe.release()
    w_vid, h_vid = sizes[next(iter(vids))]
    n_vid = min(counts.values())
    n_frames = min([n_vid] + ([fm["n"]] if fm is not None else []))
    if n_vid != n_frames:
        print(f"[WARN] 视频帧数 {counts} 与时间戳 {fm['n'] if fm else '-'} "
              f"不一致，取 {n_frames}", file=sys.stderr)
    # 没有 parquet 时用 fps 网格生成时间戳
    if fm is None:
        ts_s = (np.arange(n_frames, dtype=np.float64) / fps
                ).astype(np.float32)
        hw_ns = np.zeros(n_frames, np.uint64)
    else:
        ts_s = fm["ts_s"][:n_frames]
        hw_ns = fm["hw_ns"][:n_frames]
        if not ts_s.any():
            ts_s = (np.arange(n_frames, dtype=np.float64) / fps
                    ).astype(np.float32)
    # IMU 无 hw 时钟时无法按帧窗口挂靠 → 按时间网格（首样本对齐首帧）
    if imu_ts is not None and not hw_ns.any():
        base = imu_ts[0] - 0
        for i in range(n_frames):
            hw_ns[i] = base + int(ts_s[i] * 1e9)
        print("[WARN] 无硬件时间戳，IMU 按首样本+帧时间网格挂靠", file=sys.stderr)

    # SDK 离线 ISP（--no-sdk-isp 或库缺失时回退 OpenCV demosaic）
    sdk_isp = None
    if not args.no_sdk_isp:
        lib_path = os.environ.get("DAQ_VIKIT_LIB") or _DEFAULT_VIKIT_LIB
        gains = None
        if args.sdk_gains:
            gains = [float(x) for x in args.sdk_gains.split(",")]
            if len(gains) != 3:
                print("[ERROR] --sdk-gains 需 R,G,B 三个数", file=sys.stderr)
                sys.exit(1)
        try:
            sdk_isp = SdkIsp(lib_path, args.isp_yaml,
                             awb=args.sdk_awb, gains=gains)
            print(f"SDK 离线 ISP: {lib_path}\n  配置: {args.isp_yaml}")
        except Exception as e:
            print(f"[WARN] SDK 离线 ISP 不可用，回退 OpenCV demosaic: {e}",
                  file=sys.stderr)
            sdk_isp = None

    # 逐路转换（stacked 拆两路；separate 各转各的）
    def frame_gen(cap, eye_h, slot):
        rot = args.rotate_left if slot == "stereo_left" else args.rotate_right
        eye_bayer = bayer_l if slot == "stereo_left" else bayer_r
        i = 0
        while i < n_frames:
            ok, fr = cap.read()
            if not ok:
                break
            if stacked:
                fr = fr[:eye_h] if slot == "stereo_left" else fr[eye_h:]
            rgb = demosaic_gray(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                                if fr.ndim == 3 else fr,
                                eye_bayer, wb, args.gamma,
                                args.already_demosaiced)
            if rot == 180:
                rgb = cv2.rotate(rgb, cv2.ROTATE_180)
            if resize:
                rgb = cv2.resize(rgb, resize)
            i += 1
            yield rgb

    if "stacked" in vids:
        out_vids = {}
        if args.swap:
            out_vids["stereo_left"] = "stereo_right"
            out_vids["stereo_right"] = "stereo_left"
        else:
            out_vids["stereo_left"] = "stereo_left"
            out_vids["stereo_right"] = "stereo_right"
        cap = cv2.VideoCapture(vids["stacked"])
        eye_h = h_vid // 2
        size = resize or (w_vid, eye_h)
        for slot in ("stereo_left", "stereo_right"):
            dst_slot = out_vids[slot]
            dst = os.path.join(out_dir, "videos", dst_slot, "chunk-0000",
                               f"{dst_slot}.mp4")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            k = write_color_mp4(ffmpeg,
                                frame_gen(cap, eye_h, slot),
                                size, fps, dst)
            cam_meta[dst_slot] = (size[1], size[0])
            print(f"  {slot} → {dst} ({k} 帧)")
        cap.release()
    else:
        dst_slots = {"stereo_left": "stereo_right" if args.swap else "stereo_left",
                     "stereo_right": "stereo_left" if args.swap else "stereo_right"}
        # SDK 离线 ISP：要求左右目分文件且 1280×800（SDK 双目彩色只支持该档）
        use_sdk = (sdk_isp is not None
                   and sizes.get("stereo_left") == (1280, 800)
                   and sizes.get("stereo_right") == (1280, 800))
        wb_b = wb_r = 1.0
        if args.sdk_wb_auto and args.sdk_wb_gain:
            print("[ERROR] --sdk-wb-auto 与 --sdk-wb-gain 互斥", file=sys.stderr)
            sys.exit(1)
        if args.sdk_wb_auto:
            if use_sdk:
                # 独立句柄测量（避免采样后写帧时间戳回跳触发 SDK 拒绝）
                m_sdk = SdkIsp(lib_path, args.isp_yaml,
                               awb=args.sdk_awb, gains=gains)
                try:
                    wb_b, wb_r = measure_grayworld_gains(m_sdk, vids,
                                                         n_frames, hw_ns)
                finally:
                    m_sdk.close()
                print(f"灰世界精修: B×{wb_b:.3f} R×{wb_r:.3f} "
                      f"（采样帧通道均值 → G/B=G/R=1）")
            else:
                print("[WARN] --sdk-wb-auto 仅在 SDK 离线 ISP 生效",
                      file=sys.stderr)
        elif args.sdk_wb_gain:
            try:
                wb_b, wb_r = [float(x) for x in args.sdk_wb_gain.split(",")]
            except ValueError:
                print("[ERROR] --sdk-wb-gain 需 B,R 两个数", file=sys.stderr)
                sys.exit(1)
        if use_sdk:
            size = resize or (1280, 800)

            def pair_gen():
                cap_l = cv2.VideoCapture(vids["stereo_left"])
                cap_r = cv2.VideoCapture(vids["stereo_right"])
                try:
                    for i in range(n_frames):
                        ok_l, fr_l = cap_l.read()
                        ok_r, fr_r = cap_r.read()
                        if not (ok_l and ok_r):
                            break
                        try:
                            rgb_l, rgb_r = sdk_isp.colorize_pair(
                                cv2.cvtColor(fr_l, cv2.COLOR_BGR2GRAY)
                                if fr_l.ndim == 3 else fr_l,
                                cv2.cvtColor(fr_r, cv2.COLOR_BGR2GRAY)
                                if fr_r.ndim == 3 else fr_r,
                                ts=int(hw_ns[i]) if hw_ns.any()
                                else 1000 + i)
                        except RuntimeError as e:
                            # 个别帧被离线管线拒绝 → 回退 OpenCV demosaic
                            # 保 A/V 同步（比静默重复上一帧安全）
                            print(f"[WARN] 帧 {i} 被 SDK 拒绝，"
                                  f"回退 OpenCV demosaic: {e}", file=sys.stderr)
                            rgb_l = demosaic_gray(
                                cv2.cvtColor(fr_l, cv2.COLOR_BGR2GRAY)
                                if fr_l.ndim == 3 else fr_l,
                                bayer_l, wb, args.gamma, args.already_demosaiced)
                            rgb_r = demosaic_gray(
                                cv2.cvtColor(fr_r, cv2.COLOR_BGR2GRAY)
                                if fr_r.ndim == 3 else fr_r,
                                bayer_r, wb, args.gamma, args.already_demosaiced)
                        if wb_b != 1.0 or wb_r != 1.0:
                            rgb_l = _apply_wb_gain(rgb_l, wb_b, wb_r)
                            rgb_r = _apply_wb_gain(rgb_r, wb_b, wb_r)
                        if args.rotate_left == 180:
                            rgb_l = cv2.rotate(rgb_l, cv2.ROTATE_180)
                        if args.rotate_right == 180:
                            rgb_r = cv2.rotate(rgb_r, cv2.ROTATE_180)
                        if resize:
                            rgb_l = cv2.resize(rgb_l, resize)
                            rgb_r = cv2.resize(rgb_r, resize)
                        yield rgb_l, rgb_r
                finally:
                    cap_l.release()
                    cap_r.release()

            dst_l = os.path.join(out_dir, "videos", dst_slots["stereo_left"],
                                 "chunk-0000", f"{dst_slots['stereo_left']}.mp4")
            dst_r = os.path.join(out_dir, "videos", dst_slots["stereo_right"],
                                 "chunk-0000", f"{dst_slots['stereo_right']}.mp4")
            os.makedirs(os.path.dirname(dst_l), exist_ok=True)
            os.makedirs(os.path.dirname(dst_r), exist_ok=True)
            k = write_color_mp4_pair(ffmpeg, pair_gen(), size, fps, dst_l, dst_r)
            cam_meta[dst_slots["stereo_left"]] = (size[1], size[0])
            cam_meta[dst_slots["stereo_right"]] = (size[1], size[0])
            print(f"  SDK 离线 ISP → {dst_l} / {dst_r} ({k} 帧)")
        else:
            if sdk_isp is not None:
                print("[WARN] 双目分辨率非 1280×800（或非分文件），"
                      "SDK 离线 ISP 不支持 → 回退 OpenCV demosaic",
                      file=sys.stderr)
            for slot in ("stereo_left", "stereo_right"):
                if slot not in vids:
                    continue
                cap = cv2.VideoCapture(vids[slot])
                dst_slot = dst_slots[slot]
                size = resize or sizes[slot]
                dst = os.path.join(out_dir, "videos", dst_slot, "chunk-0000",
                                   f"{dst_slot}.mp4")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                k = write_color_mp4(ffmpeg, frame_gen(cap, None, slot),
                                    size, fps, dst)
                cam_meta[dst_slot] = (size[1], size[0])
                print(f"  {slot} → {dst} ({k} 帧)")
                cap.release()

    # 自动找标定
    calib_src = args.calib
    if not calib_src:
        for p in sorted(glob.glob(os.path.join(dataset, "**", "*calib*.json"),
                                  recursive=True)):
            calib_src = p
            break

    # episode 序号：输入 meta/episodes parquet → 参数 → 0
    episode_index = args.episode_index
    if episode_index is None:
        ep_files = sorted(glob.glob(os.path.join(
            dataset, "meta", "episodes", "**", "*.parquet"), recursive=True))
        for ep in ep_files:
            try:
                import pyarrow.parquet as pq
                t = pq.read_table(ep)
                if "episode_index" in t.column_names and t.num_rows:
                    episode_index = int(t["episode_index"][0].as_py())
                    break
            except Exception:
                continue
    if episode_index is None:
        episode_index = 0

    if sdk_isp is not None:
        sdk_isp.close()

    write_outputs(out_dir, args.task_name, episode_index, fps, n_frames,
                  ts_s, hw_ns, imu_ts, imu_vals, cam_meta,
                  calib_src=calib_src)
    print(f"\n转换完成 → {out_dir}")
    print(f"  帧数 {n_frames} @ {fps}fps | 视频 {list(cam_meta)} | "
          f"IMU 样本 {len(imu_ts) if imu_ts is not None else 0}")


if __name__ == "__main__":
    main()

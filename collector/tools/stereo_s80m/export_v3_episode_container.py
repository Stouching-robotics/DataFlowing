#!/usr/bin/env python3
"""
v3 episode 容器导出（路径 B）—— 把 EgoData 录制会话重打包为
GUI「后处理/离线SLAM」接受的 v3 episode 容器（路径 A 的封装）。

产出（extract_v3_dataset 实际读取的最小必需文件）:
    episode_0000/
    ├── meta/info.json                          # status=="complete" + metadata_json
    │     └─ metadata_json: {fays_device, fays_calibration_serial,
    │          camera_calibration_yaml(全文), imu_yaml(全文), fays_sdk_yaml(全文)}
    ├── data/frame_index.parquet                # stream,sequence,sensor_timestamp_ns,
    │                                           # host_monotonic_ns,host_realtime_ns,
    │                                           # width,height,channels,mp4_index
    ├── data/fays_imu.parquet                   # v3 标准 9 列: sensor_ts + host_mono/rt + ax..gz
    └── videos/observation.images.fays_stereo/chunk-000/file-000.mp4
                                                # H.264 上下堆叠灰度（上左/下右），
                                                # 每目默认 1280×800（--eye-res 可调，
                                                # 客户 SLAM 已适配全分辨率）

关键口径（与客户 extract_v3_dataset 对齐）:
  - frame_index 的 sensor_timestamp_ns = 原始相机硬件时钟（回卷修正后），
    **不加 timeshift**——extract 会自己加标定里的 timeshift_cam_imu（cam0）。
  - 视频帧序与 frame_index 中 stream=="fays_stereo" 的行 1:1（帧数必须相等）。
  - IMU 时间戳严格递增，且首样本早于第一帧、末样本晚于最后一帧
    （按「加完 timeshift 之后」的帧时间校验，与 extract 的口径一致）。
  - 标定分辨率默认 1280×800/目 → 拼接视频 1280×1600（frame_index 的
    width/height 同步）；内嵌相机 YAML 由 SDK 原生 640×400 dump 按比例缩放
    （fx,cx 按宽、fy,cy 按高），与录制会话 head_stereo.json 的全分辨率内参一致。

前置：相机/IMU 标定 YAML + 设备信息需先用 dump_fays_imu_calibration.py
实测导出（S80C 必须连接）。默认从 config/FS-VI80-S80C_<serial>_*.yaml 读取。

用法:
    python tools/stereo_s80m/export_v3_episode_container.py <会话目录>
    python tools/stereo_s80m/export_v3_episode_container.py <会话目录> --serial 3500000262190088
    python tools/stereo_s80m/export_v3_episode_container.py <会话目录> --output /tmp/v3
"""

import argparse
import bisect
import json
import os
import re
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from export_offline_slam_dataset import (  # noqa: E402
    load_session, parse_calib_yaml, unwrap_camera_clock)

_DEFAULT_SERIAL = "3500000262190088"          # 本机 S80C（会话 head_stereo.json 验证）
_CONFIG_DIR = os.path.join(_REPO_ROOT, "config")
_SDK_CONFIG = os.path.join(_HERE, "config", "fays_vikit.yaml")
_LEROBOT_FFMPEG = os.path.expanduser("~/miniconda3/envs/lerobot/bin/ffmpeg")

_FRAME_INDEX_COLUMNS = ["stream", "sequence", "sensor_timestamp_ns",
                        "host_monotonic_ns", "host_realtime_ns",
                        "width", "height", "channels", "mp4_index"]
# v3 标准 IMU 列（lerobot_v3.py IMU_COLUMNS）: Viewer 的 view_data.py 无条件
# 读取 host_monotonic_ns 做回放横轴，缺列直接 KeyError → 预览打不开
_IMU_COLUMNS = ["sensor_timestamp_ns", "host_monotonic_ns", "host_realtime_ns",
                "ax", "ay", "az", "gx", "gy", "gz"]
_CAM_YAML_REQUIRED = ["intrinsics", "distortion_coeffs", "resolution",
                      "T_cam_imu", "timeshift_cam_imu"]
_IMU_PARAM_KEYS = ["accelerometer_noise_density", "accelerometer_random_walk",
                   "gyroscope_noise_density", "gyroscope_random_walk",
                   "update_rate_hz"]


# ── ffmpeg 解析（自检版本号，跳过启动即崩的 conda 版） ────────────
def _find_ffmpeg() -> str:
    candidates = [os.environ.get("FFMPEG_BIN", ""), _LEROBOT_FFMPEG,
                  "/usr/bin/ffmpeg", "ffmpeg"]
    for cand in candidates:
        if not cand:
            continue
        try:
            r = subprocess.run([cand, "-version"], capture_output=True,
                               timeout=15, text=True)
            if r.returncode == 0 and "ffmpeg version" in (r.stdout + r.stderr):
                return cand
        except (OSError, subprocess.TimeoutExpired):
            continue
    raise FileNotFoundError("找不到可用的 ffmpeg（候选: FFMPEG_BIN/lerobot/system）")


# ── 简易 YAML 段落读取（imu yaml 自检用，避免依赖 PyYAML） ─────────
def _read_imu_yaml_fields(path: str) -> dict:
    out = {}
    section = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not line.startswith(" ") and line.endswith(":"):
                section = line[:-1].strip()
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                key, val = key.strip(), val.strip()
                out[f"{section}.{key}"] = val
    return out


def _check_inputs(args, serial: str) -> dict:
    """定位三个原料文件，返回 {camera_yaml, imu_yaml, fays_device}。"""
    prefix = f"FS-VI80-S80C_{serial}"
    cam = args.camera_yaml or os.path.join(_CONFIG_DIR, prefix + "_camera_calibration.yaml")
    imu = args.imu_yaml or os.path.join(_CONFIG_DIR, prefix + "_imu.yaml")
    dev = args.fays_device or os.path.join(_CONFIG_DIR, "fays_device.json")
    if not os.path.isfile(cam):
        raise FileNotFoundError(
            f"找不到相机标定 YAML: {cam}\n"
            f"  请先连接 S80C 并运行 dump_fays_imu_calibration.py 实测导出。")
    if not os.path.isfile(imu):
        raise FileNotFoundError(
            f"找不到 IMU 标定 YAML: {imu}\n"
            f"  请先连接 S80C 并运行 dump_fays_imu_calibration.py 实测导出。")
    device = {}
    if os.path.isfile(dev):
        with open(dev, encoding="utf-8") as f:
            device = json.load(f)
        if device.get("serial") != serial:
            raise SystemExit(
                f"fays_device.json 序列号 {device.get('serial')} 与 --serial {serial} 不符")
    return {"camera_yaml": cam, "imu_yaml": imu, "fays_device": device}


def _build_frame_index(frames_kept: list, drop_n: int, eye_w: int,
                       eye_h: int) -> pa.Table:
    """frames_kept: [{hardware_ns, timestamp, wall_time}]（已裁剪），sensor_ts 为原始相机时钟。"""
    n = len(frames_kept)
    rows = {
        "stream": pa.array(["fays_stereo"] * n, pa.string()),
        "sequence": pa.array([i + 1 for i in range(n)], pa.int64()),
        "sensor_timestamp_ns": pa.array(
            [int(f["hardware_ns"]) for f in frames_kept], pa.int64()),
        "host_monotonic_ns": pa.array(
            [int(round(f["timestamp"] * 1e9)) for f in frames_kept], pa.int64()),
        "host_realtime_ns": pa.array(
            [int(round(f["wall_time"] * 1e9)) for f in frames_kept], pa.int64()),
        "width": pa.array([eye_w] * n, pa.int32()),
        "height": pa.array([2 * eye_h] * n, pa.int32()),
        "channels": pa.array([1] * n, pa.int32()),
        "mp4_index": pa.array([i for i in range(n)], pa.int64()),
    }
    return pa.table(rows)


def _make_host_map(frames_kept: list):
    """帧锚点 (sensor_ts → host) 分段线性映射，返回 callable(sensor_ns) -> (mono_ns, rt_ns)。

    客户 v3 格式要求 fays_imu.parquet 带 host_monotonic_ns/host_realtime_ns 列
    （Viewer 的 view_data.py 无条件读 host_monotonic_ns 做回放横轴，缺列直接
    KeyError → "Open failed"），而录制会话的 IMU parquet 只存设备时钟时间戳，
    宿主到达时间在录制时未逐样本保存。设备时钟读数是 40ms 量化 + 偶发陈旧读
    （实测全局线性拟合残差高达 126ms，不可用），宿主时钟（timestamp/wall_time）
    平滑单调，因此用帧锚点分段线性插值：帧处精确、帧间线性——IMU 曲线在
    Viewer 回放轴上与视频严格同步。区间外按边缘段斜率外推（首尾各 ~5-25ms）。
    锚点序列先行单调化（sensor 等值/宿主倒退均取前值+1ns，见下方注释），保证
    插值输出严格递增。
    """
    s = [int(f["hardware_ns"]) for f in frames_kept]      # 回卷修正后，严格递增
    mono = [int(round(f["timestamp"] * 1e9)) for f in frames_kept]
    rt = [int(round(f["wall_time"] * 1e9)) for f in frames_kept]
    if len(s) < 2:
        raise SystemExit("帧数不足 2，无法建立宿主时基映射")

    # 三列锚点单调化（保证插值输出严格递增）：
    # - s：设备时钟 40ms 量化 → 相邻帧 hardware_ns 可能等值，等值锚点使分段插值
    #   退化为平台段（x1==x0 返回常值）→ 等值处 +1ns 严格化；
    # - mono/rt：PAVA 单调回归（isotonic regression，最小 L2 距离的非递减拟合）。
    #   宿主时钟偶发微倒退（000008 帧 434 实测 mono -1.01ms / rt -1.48ms）产生
    #   负斜率段破坏严格递增；池化平均把倒退摊到邻近锚点（亚 ms 级失真）。
    #   不用简单钳位：钳位在 s 的 40ms 量化步长下会连锁放大（60ms 步长钳位过头
    #   → 下一锚点相对它又"倒退"，实测连锁 11 处、最大虚增 34ms）。PAVA 输出
    #   的平台段等值由 _build_imu_table 的 +1ns 后处理转为严格递增。权重取本段
    #   s 步长（对 x 轴保真）
    n_eq = 0
    for i in range(1, len(s)):
        if s[i] <= s[i - 1]:
            s[i] = s[i - 1] + 1
            n_eq += 1
    if n_eq:
        print(f"[WARN] sensor 设备时钟有 {n_eq} 处相邻等值（40ms 量化），已 +1ns 严格化",
              file=sys.stderr)
    for vals, name in ((mono, "host_monotonic"), (rt, "host_realtime")):
        base = vals[0]
        w = [s[1] - s[0]] + [s[i] - s[i - 1] for i in range(1, len(s))]
        blocks = []  # [wsum, vwsum, start, end]
        for i in range(len(vals)):
            blocks.append([w[i], (vals[i] - base) * w[i], i, i])
            while len(blocks) >= 2 and \
                    blocks[-2][1] / blocks[-2][0] > blocks[-1][1] / blocks[-1][0]:
                w0, s0, b0, _ = blocks[-2]
                w1, s1, _, e1 = blocks[-1]
                blocks[-2] = [w0 + w1, s0 + s1, b0, e1]
                blocks.pop()
        n_chg, max_dev = 0, 0
        for wsum, vwsum, b0, e0 in blocks:
            avg = int(round(base + vwsum / wsum))
            for i in range(b0, e0 + 1):
                dev = abs(avg - vals[i])
                if dev:
                    n_chg += 1
                    max_dev = max(max_dev, dev)
                vals[i] = avg
        if n_chg:
            print(f"[WARN] {name} 宿主时钟有非单调毛刺，PAVA 修正 {n_chg} 处锚点"
                  f"（最大偏差 {max_dev / 1e6:.2f}ms）", file=sys.stderr)

    def _at(vals, x):
        i = bisect.bisect_right(s, x) - 1
        i = min(max(i, 0), len(s) - 2)
        x0, x1 = s[i], s[i + 1]
        if x1 == x0:
            return vals[i]
        w = (x - x0) / (x1 - x0)
        return vals[i] + w * (vals[i + 1] - vals[i])

    def to_host(x):
        return int(round(_at(mono, x))), int(round(_at(rt, x)))

    return to_host


def _build_imu_table(imu_rows: list, to_host) -> pa.Table:
    """imu_rows: [(ts_ns, ax, ay, az, gx, gy, gz)] 逐样本 → 9 列标准 v3 表。"""
    sensor = [int(r[0]) for r in imu_rows]
    host = [to_host(s) for s in sensor]
    m = [h[0] for h in host]
    r = [h[1] for h in host]
    # 严格递增后处理：插值/浮点取整理论上已保证，此兜底防等值（Viewer 要求严格）
    for col in (m, r):
        for i in range(1, len(col)):
            if col[i] <= col[i - 1]:
                col[i] = col[i - 1] + 1
    return pa.table({
        "sensor_timestamp_ns": pa.array(sensor, pa.int64()),
        "host_monotonic_ns": pa.array(m, pa.int64()),
        "host_realtime_ns": pa.array(r, pa.int64()),
        "ax": pa.array([float(r[1]) for r in imu_rows], pa.float64()),
        "ay": pa.array([float(r[2]) for r in imu_rows], pa.float64()),
        "az": pa.array([float(r[3]) for r in imu_rows], pa.float64()),
        "gx": pa.array([float(r[4]) for r in imu_rows], pa.float64()),
        "gy": pa.array([float(r[5]) for r in imu_rows], pa.float64()),
        "gz": pa.array([float(r[6]) for r in imu_rows], pa.float64()),
    })


def _encode_stereo_video(session_dir: str, out_path: str, drop_n: int,
                         eye_w: int, eye_h: int, ffmpeg: str) -> None:
    """左右目源视频 → 灰度 eye_w×eye_h → vstack（上左/下右），去前 drop_n 帧。"""
    left = os.path.join(session_dir, "videos", "stereo_left", "chunk-0000",
                        "stereo_left.mp4")
    right = os.path.join(session_dir, "videos", "stereo_right", "chunk-0000",
                         "stereo_right.mp4")
    for p in (left, right):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到源视频: {p}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fc = (f"[0:v]trim=start_frame={drop_n},setpts=PTS-STARTPTS,"
          f"scale={eye_w}:{eye_h},format=gray[l];"
          f"[1:v]trim=start_frame={drop_n},setpts=PTS-STARTPTS,"
          f"scale={eye_w}:{eye_h},format=gray[r];"
          f"[l][r]vstack,format=yuv420p[v]")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-i", left, "-i", right,
           "-filter_complex", fc, "-map", "[v]",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-vsync", "vfr", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg vstack 失败: {r.stderr.strip()[:800]}")


def _scale_calib_yaml(text: str, cal: dict, eye_w: int, eye_h: int) -> str:
    """把 SDK 原生标定 YAML 缩放到 --eye-res 分辨率（fx,cx 按宽比、fy,cy 按高比）。

    S80C SDK dump 原生 640×400；1280×800 时全部 ×2，与录制会话 head_stereo.json
    的 1280×800 内参逐位一致。畸变系数/外参/timeshift 不变。
    """
    nw, nh = cal["cam0"]["resolution"]
    sx, sy = eye_w / nw, eye_h / nh
    if sx == 1.0 and sy == 1.0:
        return text

    def _intr(m):
        v = [float(x) for x in m.group(1).split(",")]
        return (f"intrinsics: [{v[0] * sx!r}, {v[1] * sy!r}, "
                f"{v[2] * sx!r}, {v[3] * sy!r}]")

    out = re.sub(r"intrinsics: \[([^\]]+)\]", _intr, text)
    out = re.sub(r"resolution: \[\d+, \d+\]",
                 f"resolution: [{eye_w}, {eye_h}]", out)
    return out


def _probe_frames(path: str, ffprobe: str) -> int:
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_frames",
                        "-of", "default=noprint_wrappers=1:nokey=1", path],
                       capture_output=True, text=True, timeout=120)
    try:
        return int(r.stdout.strip())
    except ValueError:
        raise RuntimeError(f"无法探测帧数: {path} ({r.stderr.strip()[:200]})")


def _build_info_json(metadata: dict, frames_kept: list, imu_rows: list,
                     camera_yaml_text: str, imu_yaml_text: str,
                     sdk_yaml_text: str, drop_n: int, fays_device: dict,
                     serial: str) -> dict:
    n_frames = len(frames_kept)
    dur_s = (imu_rows[-1][0] - imu_rows[0][0]) / 1e9
    imu_hz = (len(imu_rows) - 1) / dur_s if dur_s > 0 else 0.0
    aligned = [f["hardware_ns"] + f["_timeshift_ns"] for f in frames_kept]
    span_s = (aligned[-1] - aligned[0]) / 1e9
    stereo_hz = (n_frames - 1) / span_s if span_s > 0 else 0.0

    device = {
        "model": fays_device.get("model") or "FS-VI80-S80C",
        "serial": serial,
        "camera_count": fays_device.get("camera_count", 2),
        "imu_count": fays_device.get("imu_count", 1),
    }
    if fays_device.get("firmware"):
        device["firmware"] = fays_device["firmware"]
    if fays_device.get("sdk_version"):
        device["sdk_version"] = fays_device["sdk_version"]

    metadata_json = json.dumps({
        "fays_device": device,
        "fays_calibration_serial": serial,
        "camera_calibration_yaml": camera_yaml_text,
        "imu_yaml": imu_yaml_text,
        "fays_sdk_yaml": sdk_yaml_text,
    }, ensure_ascii=False)

    summary = json.dumps({
        "duration_s": round(dur_s, 6),
        "fays_imu_samples": len(imu_rows),
        "fays_imu_average_hz": round(imu_hz, 6),
        "fays_stereo_samples": n_frames,
        "fays_stereo_average_hz": round(stereo_hz, 6),
        "frames_dropped_front": drop_n,
        "failure": "",
    })

    return {
        "codebase_version": "v3.0",
        "robot_type": "s80c",
        "total_episodes": 1,
        "total_frames": n_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 100,
        "fps": float(metadata.get("fps") or 30.0),
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "extensions": {"calibration_root": "calibration/",
                       "annotation_file": "meta/annotations.jsonl"},
        "features": {
            "observation.images.fays_stereo": {
                "dtype": "video",
                "shape": [800, 640, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(metadata.get("fps") or 30.0),
                    "video.is_depth_map": False,
                    "has_audio": False,
                    "source_episode": str(metadata.get("episode_index", 1)),
                },
            },
        },
        "status": "complete",
        "failure": "",
        "episode_frames": n_frames,
        "summary_json": summary,
        "created_unix_s": time.time(),
        "metadata_json": metadata_json,
    }


def _self_check(info: dict, frames_kept: list, imu_rows: list,
                frame_index: pa.Table, imu_table: pa.Table,
                video_frames: int, cal: dict, imu_fields: dict,
                eye_w: int, eye_h: int) -> list:
    """按客户 extract_v3_dataset 的硬检查逐项自检，返回问题列表。"""
    problems = []
    if info.get("status") != "complete":
        problems.append("info.json status != complete")
    mj = json.loads(info["metadata_json"])
    if not mj.get("fays_device", {}).get("serial"):
        problems.append("metadata_json.fays_device.serial 缺失")
    for key in ("camera_calibration_yaml", "imu_yaml"):
        if not mj.get(key):
            problems.append(f"metadata_json.{key} 缺失")
    # 相机 YAML 结构：cam0/cam1 必备键 + cam1.T_cn_cnm1
    for cam in ("cam0", "cam1"):
        for k in _CAM_YAML_REQUIRED:
            if k not in cal.get(cam, {}):
                problems.append(f"camera_calibration_yaml.{cam}.{k} 缺失")
    if "T_cn_cnm1" not in cal.get("cam1", {}):
        problems.append("camera_calibration_yaml.cam1.T_cn_cnm1 缺失")
    if cal["cam0"]["resolution"] != [640, 400]:
        problems.append(f"标定分辨率 {cal['cam0']['resolution']} 不是 [640, 400]")
    # IMU YAML：device.serial_number + 5 参数
    if imu_fields.get("device.serial_number") != str(mj["fays_device"]["serial"]):
        problems.append("imu_yaml device.serial_number 与 fays_device.serial 不符")
    for k in _IMU_PARAM_KEYS:
        if f"imu.{k}" not in imu_fields:
            problems.append(f"imu_yaml imu.{k} 缺失")
    # frame_index：列 + 行数 = 视频帧数 + 严格递增
    if list(frame_index.column_names) != _FRAME_INDEX_COLUMNS:
        problems.append(f"frame_index 列名不符: {list(frame_index.column_names)}")
    rows = frame_index.to_pydict()
    if len(rows["stream"]) != video_frames:
        problems.append(f"fays_stereo 索引行数 {len(rows['stream'])} != 视频帧数 {video_frames}")
    ts = rows["sensor_timestamp_ns"]
    if not all(a < b for a, b in zip(ts, ts[1:])):
        problems.append("sensor_timestamp_ns 非严格递增")
    if any(w != eye_w or h != 2 * eye_h or c != 1 for w, h, c in
           zip(rows["width"], rows["height"], rows["channels"])):
        problems.append(f"frame_index width/height/channels 不是 {eye_w}/{2 * eye_h}/1")
    # IMU：严格递增 + 覆盖（按加完 timeshift 的帧时间）
    imu_ts = [r[0] for r in imu_rows]
    if not all(a < b for a, b in zip(imu_ts, imu_ts[1:])):
        problems.append("IMU 时间戳非严格递增")
    aligned_first = frames_kept[0]["hardware_ns"] + frames_kept[0]["_timeshift_ns"]
    aligned_last = frames_kept[-1]["hardware_ns"] + frames_kept[-1]["_timeshift_ns"]
    if imu_ts[0] > aligned_first:
        problems.append("IMU 首样本晚于第一帧（timeshift 后）")
    if imu_ts[-1] < aligned_last:
        problems.append("IMU 末样本早于最后一帧（timeshift 后）")
    # IMU 宿主列（v3 标准 9 列；Viewer 无条件读 host_monotonic_ns，缺列即 KeyError）
    if list(imu_table.column_names) != _IMU_COLUMNS:
        problems.append(f"fays_imu 列名不符: {list(imu_table.column_names)}")
    for col in ("host_monotonic_ns", "host_realtime_ns"):
        vals = imu_table.column(col).to_pylist()
        if not all(a < b for a, b in zip(vals, vals[1:])):
            problems.append(f"IMU {col} 非严格递增")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", help="录制会话目录（含 timestamps.json）")
    ap.add_argument("--serial", default=_DEFAULT_SERIAL)
    ap.add_argument("--output", default=None,
                    help="输出目录（默认 tools/stereo_s80m/offline_slam_output/v3_<会话名>/）")
    ap.add_argument("--camera-yaml", default=None)
    ap.add_argument("--imu-yaml", default=None)
    ap.add_argument("--fays-device", default=None, help="fays_device.json 路径")
    ap.add_argument("--eye-res", default="1280x800",
                    help="每目灰度分辨率 WxH（默认 1280x800；标定 YAML 内参按比例缩放，"
                         "vstack 后视频总高 = 2×H）。旧口径为 640x400")
    ap.add_argument("--keep-dir", action="store_true", help="保留解包目录（默认也打包 zip）")
    args = ap.parse_args()
    _m = re.fullmatch(r"(\d+)x(\d+)", args.eye_res)
    if not _m:
        raise SystemExit(f"--eye-res 格式应为 WxH，收到: {args.eye_res!r}")
    eye_w, eye_h = int(_m.group(1)), int(_m.group(2))

    session_dir = os.path.abspath(args.session_dir)
    if not os.path.isfile(os.path.join(session_dir, "timestamps.json")):
        raise SystemExit(f"不是录制会话目录（缺 timestamps.json）: {session_dir}")

    inputs = _check_inputs(args, args.serial)
    with open(inputs["camera_yaml"], encoding="utf-8") as f:
        camera_yaml_text = f.read()
    with open(inputs["imu_yaml"], encoding="utf-8") as f:
        imu_yaml_text = f.read()
    with open(_SDK_CONFIG, encoding="utf-8") as f:
        sdk_yaml_text = f.read()
    cal = parse_calib_yaml(inputs["camera_yaml"])
    imu_fields = _read_imu_yaml_fields(inputs["imu_yaml"])
    timeshift_ns = int(round(cal["cam0"]["timeshift_cam_imu"] * 1e9))
    print(f"timeshift_cam_imu(cam0) = {cal['cam0']['timeshift_cam_imu']} "
          f"({timeshift_ns} ns，extract 侧会再加一次)", file=sys.stderr)

    data = load_session(session_dir)
    frames = data["frames"]
    imu_rows = sorted(data["imu_rows"], key=lambda r: r[0])
    metadata = data["metadata"]
    print(f"会话: 帧数={len(frames)}  IMU 样本数={len(imu_rows)}  "
          f"IMU 范围=[{imu_rows[0][0]}, {imu_rows[-1][0]}]ns", file=sys.stderr)

    # 相机时钟回卷修正 → 绝对原始时钟（与路径 A 同口径）
    cam_abs = unwrap_camera_clock(frames, imu_rows[0][0])
    for f, ts in zip(frames, cam_abs):
        f["hardware_ns"] = int(ts)
        f["_timeshift_ns"] = timeshift_ns

    # IMU 覆盖裁剪：extract 会把 timeshift 加到 sensor_timestamp_ns，
    # 因此按「加完 timeshift」的时间校验覆盖（首样本须早于第一帧）
    first_imu, last_imu = imu_rows[0][0], imu_rows[-1][0]
    drop_n = sum(1 for f in frames if f["hardware_ns"] + timeshift_ns < first_imu)
    frames_kept = frames[drop_n:]
    if not frames_kept:
        raise SystemExit("全部帧都被裁剪（IMU 覆盖为空）")
    aligned_last = frames_kept[-1]["hardware_ns"] + timeshift_ns
    if aligned_last > last_imu:
        raise SystemExit(
            f"IMU 末样本 {last_imu} 早于最后一帧（timeshift 后）{aligned_last}，"
            f"数据不满足覆盖约束")
    print(f"裁剪前 {drop_n} 帧（早于 IMU 首样本）→ 保留 {len(frames_kept)} 帧", file=sys.stderr)
    to_host = _make_host_map(frames_kept)
    print("IMU 宿主时基映射: 帧锚点分段线性插值（设备时钟 40ms 量化 + 陈旧读，"
          "全局线性拟合不可行）", file=sys.stderr)

    # 输出目录
    out_root = args.output or os.path.join(
        _HERE, "offline_slam_output",
        "v3_" + os.path.basename(session_dir.rstrip("/")))
    ep_dir = os.path.join(out_root, "episode_0000")
    for _sub in ("data", "meta"):
        os.makedirs(os.path.join(ep_dir, _sub), exist_ok=True)

    # 1) 视频（先去 drop_n 帧再拼接）
    ffmpeg = _find_ffmpeg()
    ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe")
    if not os.path.isfile(ffprobe):
        ffprobe = "ffprobe"
    video_path = os.path.join(ep_dir, "videos", "observation.images.fays_stereo",
                              "chunk-000", "file-000.mp4")
    print(f"[1/4] 拼接双目视频（每目 {eye_w}×{eye_h} 灰度，ffmpeg={ffmpeg}）...",
          file=sys.stderr)
    _encode_stereo_video(session_dir, video_path, drop_n, eye_w, eye_h, ffmpeg)
    video_frames = _probe_frames(video_path, ffprobe)
    print(f"      视频帧数={video_frames}", file=sys.stderr)

    # 2) frame_index.parquet
    print("[2/4] 写 frame_index.parquet ...", file=sys.stderr)
    frame_index = _build_frame_index(frames_kept, drop_n, eye_w, eye_h)
    pq.write_table(frame_index, os.path.join(ep_dir, "data", "frame_index.parquet"))

    # 3) fays_imu.parquet
    print("[3/4] 写 fays_imu.parquet ...", file=sys.stderr)
    imu_table = _build_imu_table(imu_rows, to_host)
    pq.write_table(imu_table, os.path.join(ep_dir, "data", "fays_imu.parquet"))

    # 4) meta/info.json（相机 YAML 缩放到 --eye-res 分辨率）
    print("[4/4] 写 meta/info.json ...", file=sys.stderr)
    scaled_yaml = _scale_calib_yaml(camera_yaml_text, cal, eye_w, eye_h)
    info = _build_info_json(metadata, frames_kept, imu_rows, scaled_yaml,
                            imu_yaml_text, sdk_yaml_text, drop_n,
                            inputs["fays_device"], args.serial)
    with open(os.path.join(ep_dir, "meta", "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # 自检 + 打包
    problems = _self_check(info, frames_kept, imu_rows, frame_index, imu_table,
                           video_frames, cal, imu_fields, eye_w, eye_h)
    if problems:
        print("[FAIL] 自检发现问题:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(1)
    print("[OK] 全部硬约束自检通过", file=sys.stderr)

    # zip_path 必须绝对化：cwd=out_root 下 zip 会把相对输出路径再嵌套一层
    zip_path = os.path.abspath(os.path.join(out_root, "episode_0000.zip"))
    r = subprocess.run(["zip", "-qr", zip_path, "episode_0000"],
                       cwd=out_root, capture_output=True, text=True)
    if r.returncode != 0:
        # zip 的报错走 stdout（非 stderr），两路都打印
        msg = (r.stdout + r.stderr).strip().replace("\n", " ")[:200]
        print(f"[WARN] zip 打包失败: {msg}", file=sys.stderr)
    else:
        print(f"[OK] 容器目录: {ep_dir}", file=sys.stderr)
        print(f"[OK] 容器 zip : {zip_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

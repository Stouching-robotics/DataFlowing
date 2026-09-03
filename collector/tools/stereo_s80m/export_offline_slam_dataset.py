#!/usr/bin/env python3
"""
EgoData 录制会话 → 离线双目惯性 SLAM 数据集（ORB-SLAM3 输入格式）

格式规范: OFFLINE_DATA_FORMAT.md / OFFLINE_DATA_RECORDING.md
输出目录结构:
  session_YYYYMMDD_HHMMSS/
    orb_calibration.yaml      矫正内参 + 基线 + 相机→IMU 外参 + IMU 噪声/频率
    images.csv                timestamp_ns,left_image,right_image
    imu.csv                   timestamp_ns,ax,ay,az,gx,gy,gz
    camera_timestamps.csv     审计: 原始相机时钟(32位回卷修正后)/对齐后/偏移
    metadata.yaml             交付说明
    cam0/data/<ts>.png        左目: 已双目矫正 640×400 8-bit 灰度 PNG
    cam1/data/<ts>.png        右目

转换要点:
  1. 原始视频 1280×800 彩色 → 灰度 → 缩放到标定原生 640×400
     → 用工厂标定 (equidistant/fisheye) 做双目矫正 (cv2.fisheye.stereoRectify)
  2. 相机硬件时钟是 32 位纳秒计数器 (每 4.295s 回卷) → 逐帧回卷修正,
     并对齐到 IMU 时基, 再按规范应用 timeshift_cam_imu (只应用一次)
  3. IMU 从 parquet 的逐帧批次展开为逐样本行, 列序 [gx,gy,gz,ax,ay,az] → [ax,ay,az,gx,gy,gz]
  4. 两个 CSV 时间戳严格递增; IMU 时间范围覆盖第一张到最后一张图

用法:
  python export_offline_slam_dataset.py --session data/recordings/111/111_000003
  python export_offline_slam_dataset.py --session ... --out /path/out --calib /path/calib.yaml
"""

import argparse
import csv
import json
import os
import sys
import tarfile
import time

import cv2
import numpy as np
import pyarrow.parquet as pq

# ── 默认路径 ──────────────────────────────────────────────────
_FACTORY_CALIB = os.path.expanduser(
    "~/FaysSense_VI_Kit_Release/config/calib/calib.yaml")
_DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "offline_slam_output")
_FPS_FALLBACK = 25.0

# 时间偏移已应用一次, 回放时不得再传 --camera-time-shift-s
_TIMESHIFT_APPLIED = True


def parse_calib_yaml(path: str) -> dict:
    """解析工厂 calib.yaml（固定结构: cam0/cam1 下键值 + 4x4 矩阵列表）。

    结构非常固定, 手写递归解析避免 PyYAML 依赖, 支持两种列表风格:
      内联:     intrinsics: [fx, fy, cx, cy]
      块序列:   T_cam_imu:
                - [a, b, c, d]
                - [0, 0, 0, 1]
    """
    def _num(s: str):
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s  # 字符串标量 (如 camera_model: pinhole)

    def _inline_list(lines, i, buf):
        """解析从 buf 开始的方括号列表（可能跨多行）。"""
        while buf.count("[") != buf.count("]"):
            i += 1
            buf += lines[i].strip()
        return json.loads(buf), i + 1

    def _list_at(lines, i, indent):
        """解析与 key 同缩进的 '-' 块序列项, 返回 (list, 下一个未消费行号)。"""
        items = []
        while i < len(lines):
            l = lines[i].rstrip()
            if not l.strip() or l.lstrip().startswith("#"):
                i += 1
                continue
            ind = len(l) - len(l.lstrip())
            if ind < indent:
                break
            s = l.lstrip()
            if not s.startswith("- "):
                break
            content = s[2:].strip()
            if content.startswith("["):
                val, i = _inline_list(lines, i, content)
                items.append(val)
            elif content == "":
                child, i = _parse(lines, i + 1, indent + 2)
                items.append(child)
            else:
                items.append(_num(content))
                i += 1
        return items, i

    def _parse(lines, i, indent):
        """解析一个 dict 块。返回 (dict, 下一个未消费行号)。"""
        d = {}
        while i < len(lines):
            line = lines[i].rstrip()
            if not line.strip() or line.lstrip().startswith("#"):
                i += 1
                continue
            cur_indent = len(line) - len(line.lstrip())
            if cur_indent < len(indent):
                break  # 块结束
            if cur_indent > len(indent):
                raise ValueError(f"缩进错误 @ {line!r}")
            if ":" not in line:
                raise ValueError(f"无法解析行: {line!r}")
            key, _, rest = line.partition(":")
            key = key.strip()
            rest = rest.strip()
            if not rest:
                # 子块, 或同缩进块序列列表
                j = i + 1
                while j < len(lines) and (not lines[j].strip()
                                          or lines[j].lstrip().startswith("#")):
                    j += 1
                if j < len(lines):
                    l2 = lines[j].rstrip()
                    ind2 = len(l2) - len(l2.lstrip())
                    if ind2 == cur_indent and l2.lstrip().startswith("- "):
                        d[key], i = _list_at(lines, j, cur_indent)
                        continue
                child, i = _parse(lines, i + 1, indent + "  ")
                d[key] = child
            elif rest.startswith("["):
                val, i = _inline_list(lines, i, rest)
                d[key] = val
            else:
                d[key] = _num(rest)
                i += 1
        return d, i

    with open(path) as f:
        lines = f.readlines()
    cal, _ = _parse(lines, 0, "")
    return cal


def load_session(session_dir: str) -> dict:
    """读取会话数据, 返回 {timestamps, imu_rows, calib, metadata}。"""
    ts_path = os.path.join(session_dir, "timestamps.json")
    with open(ts_path) as f:
        ts = json.load(f)

    # 每帧取一条记录（左右目时间戳相同）
    by_frame = {}
    for e in ts["timestamps"]:
        by_frame.setdefault(e["frame_index"], e)
    frames = [by_frame[k] for k in sorted(by_frame)]

    # IMU parquet
    imu_dir = os.path.join(session_dir, "data", "imu", "chunk-0000")
    pq_path = os.path.join(imu_dir, "chunk_000000.parquet")
    if not os.path.exists(pq_path):
        raise FileNotFoundError(f"找不到 IMU parquet: {pq_path}")
    table = pq.read_table(pq_path)
    imu_rows = []
    for i in range(table.num_rows):
        batch = table.column("observation.imu")[i].as_py()
        ts_list = table.column("imu_ts_ns")[i].as_py()
        for j, (ts_ns, sample) in enumerate(zip(ts_list, batch)):
            # sample = [gx, gy, gz, ax, ay, az] (SDK 结构体顺序)
            imu_rows.append((int(ts_ns), sample[3], sample[4], sample[5],
                             sample[0], sample[1], sample[2]))

    with open(os.path.join(session_dir, "metadata.json")) as f:
        metadata = json.load(f)
    return {"frames": frames, "imu_rows": imu_rows, "metadata": metadata}


def unwrap_camera_clock(frames: list, imu_ref_ns: int,
                        period_ns: int = 40_000_000) -> list:
    """32 位纳秒计数器回卷修正 → 单调递增的绝对相机时钟(ns)。

    相机与 IMU 共用同一硬件传感器时钟: 用首条 IMU 样本时间推断
    相机原始值 (uint32) 的回卷次数 k = round((ref - raw)/2^32),
    得到绝对相位。逐帧按回卷修正后的原始增量累加 —— 增量总和严格
    等于真实总时长, 陈旧读(瞬时倒退)由后续 catch-up 增量精确补偿,
    因此总时长不被破坏。

    瞬时倒退的一帧是 SDK 读时钟竞态(硬件计数器本身单调, 实测本会话
    846/846 个有效增量恰好都是 40.000ms): 该帧真实位置在标称网格
    ts[i-1]+period 上, 其后的 catch-up 增量已包含这段补偿, 不再修改。
    """
    raw0 = frames[0]["hardware_ns"] & 0xFFFFFFFF
    k = round((imu_ref_ns - raw0) / (1 << 32))
    base = raw0 + k * (1 << 32)
    acc = 0
    out = [base]
    prev_u = raw0
    for f in frames[1:]:
        v = f["hardware_ns"] & 0xFFFFFFFF
        diff = v - prev_u
        if diff < -(1 << 31):          # 32 位回卷
            diff += 1 << 32
        acc += diff
        out.append(base + acc)
        prev_u = v
    # 陈旧读帧放回标称网格, 保证严格递增
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + period_ns
    return out


def build_rect_maps(cal: dict, size=(640, 400)):
    """工厂标定 → 双目矫正映射表 (equidistant/fisheye)。

    配对约定 (本会话实测验证): 录制文件 stereo_left ↔ 标定 cam0,
    stereo_right ↔ 标定 cam1。S80M 个别设备有 stereo_swap_lr 配置,
    换设备时由 main() 中的 check_orientation 自检确定。
    工厂标定的内参分辨率是 1280×800, 输出图像是 640×400:
    内参按分辨率比例缩放 (fx,fy,cx,cy × s)。与 ROM 原生 640×400 标定一致。
    """
    s = size[0] / cal["cam0"]["resolution"][0]  # 0.5
    K0 = np.array([[cal["cam0"]["intrinsics"][0], 0, cal["cam0"]["intrinsics"][2]],
                   [0, cal["cam0"]["intrinsics"][1], cal["cam0"]["intrinsics"][3]],
                   [0, 0, 1]], dtype=np.float64) * s
    K1 = np.array([[cal["cam1"]["intrinsics"][0], 0, cal["cam1"]["intrinsics"][2]],
                   [0, cal["cam1"]["intrinsics"][1], cal["cam1"]["intrinsics"][3]],
                   [0, 0, 1]], dtype=np.float64) * s
    D0 = np.array(cal["cam0"]["distortion_coeffs"], dtype=np.float64)
    D1 = np.array(cal["cam1"]["distortion_coeffs"], dtype=np.float64)
    T01 = np.array(cal["cam1"]["T_cn_cnm1"], dtype=np.float64)
    R = T01[:3, :3]
    t = T01[:3, 3]

    R0, R1, P0, P1, Q = cv2.fisheye.stereoRectify(
        K0, D0, K1, D1, size, R, t, flags=cv2.CALIB_ZERO_DISPARITY)
    m0 = cv2.fisheye.initUndistortRectifyMap(K0, D0, R0, P0, size, cv2.CV_16SC2)
    m1 = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2)
    return {"map0": m0, "map1": m1, "P0": P0, "P1": P1, "R0": R0, "R1": R1}


def check_orientation(video_dir: str, cal: dict, size=(640, 400),
                      n_frames: int = 6) -> bool:
    """左右目配对自检 (S80M 存在 stereo_swap_lr 隐患)。

    同一组 ORB 匹配分别测两种配对的极线对齐 |y差| 中位数:
      常规配对 = 文件 stereo_left ↔ 标定 cam0
      交换配对 = 文件 stereo_left ↔ 标定 cam1 (外参取逆)
    正确配对的误差应远小于错配 (本会话实测 3px vs 57px)。
    返回 True = 需要交换 (stereo_left 文件实际对应标定 cam1)。
    """
    import copy
    cal_swap = copy.deepcopy(cal)
    old0, old1 = cal["cam0"], cal["cam1"]
    cal_swap["cam0"], cal_swap["cam1"] = old1, old0
    # T_cn_cnm1 始终是"第二个相机相对第一个"的位姿
    cal_swap["cam1"]["T_cn_cnm1"] = old1["T_cn_cnm1"]

    maps_normal = build_rect_maps(cal, size)
    maps_swap = build_rect_maps(cal_swap, size)

    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    cap0 = cv2.VideoCapture(
        os.path.join(video_dir, "stereo_left", "chunk-0000",
                     "stereo_left.mp4"))
    cap1 = cv2.VideoCapture(
        os.path.join(video_dir, "stereo_right", "chunk-0000",
                     "stereo_right.mp4"))

    def score(maps):
        cap0.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ys = []
        for _ in range(n_frames):
            ok0, f0 = cap0.read()
            ok1, f1 = cap1.read()
            if not (ok0 and ok1):
                break
            l = cv2.resize(cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY), size,
                           interpolation=cv2.INTER_AREA)
            r = cv2.resize(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), size,
                           interpolation=cv2.INTER_AREA)
            lr = cv2.remap(l, maps["map0"][0], maps["map0"][1],
                           cv2.INTER_LINEAR)
            rr = cv2.remap(r, maps["map1"][0], maps["map1"][1],
                           cv2.INTER_LINEAR)
            kl, dl = orb.detectAndCompute(lr, None)
            kr, dr = orb.detectAndCompute(rr, None)
            m = bf.knnMatch(dl, dr, k=2)
            for mm, nn in m:
                if mm.distance < 0.75 * nn.distance:
                    ys.append(kl[mm.queryIdx].pt[1] - kr[mm.trainIdx].pt[1])
        return np.median(np.abs(ys)) if ys else float("inf")

    s_n = score(maps_normal)
    s_s = score(maps_swap)
    cap0.release()
    cap1.release()
    need_swap = s_s + 0.5 < s_n
    print(f"  [方向自检] 常规配对 |y差| 中位 {s_n:.2f}px, "
          f"交换配对 {s_s:.2f}px → "
          f"{'需要交换' if need_swap else '常规配对正确'}")
    return need_swap


def extract_frames(video_dir: str, ts_list: list, maps: dict, out_dir: str,
                   size=(640, 400), skip: int = 0) -> dict:
    """左右视频 → 灰度 → 缩放 → 矫正 → PNG。返回每目写盘统计。

    skip: 视频开头跳过的帧数（与 ts_list 已被裁剪掉的前导帧对应）。
    严格校验: 视频帧数必须恰好等于 skip + len(ts_list), 多一帧少一帧都报错。
    """
    stats = {}
    for side, cam in (("stereo_left", "cam0"), ("stereo_right", "cam1")):
        vp = os.path.join(video_dir, side, "chunk-0000", f"{side}.mp4")
        cap = cv2.VideoCapture(vp)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n != skip + len(ts_list):
            cap.release()
            raise RuntimeError(
                f"{cam}: 视频帧数 {n} != skip({skip}) + 时间戳数({len(ts_list)})")
        data_dir = os.path.join(out_dir, cam, "data")
        os.makedirs(data_dir, exist_ok=True)
        mapx, mapy = maps[f"map{'0' if cam == 'cam0' else '1'}"]
        for _ in range(skip):          # 丢弃与裁剪帧对应的前导视频帧
            ok, _ = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"{cam}: 跳过第 {skip} 帧时视频提前结束")
        idx = 0
        for ts in ts_list:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(
                    f"{cam}: 视频在第 {skip + idx} 帧提前结束 "
                    f"(期望 {skip + len(ts_list)} 帧)")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
            rect = cv2.remap(gray, mapx, mapy, cv2.INTER_LINEAR)
            png = os.path.join(data_dir, f"{ts}.png")
            cv2.imwrite(png, rect, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            idx += 1
        cap.release()
        stats[cam] = idx
    return stats


def write_csvs(out_dir: str, ts_aligned: list, ts_raw: list, imu_rows: list,
               timeshift: float, start_index: int = 0):
    """images.csv / camera_timestamps.csv / imu.csv。

    start_index: ts 列表相对原会话的首帧序号（裁剪过前导帧时用于审计追踪）。
    """
    with open(os.path.join(out_dir, "images.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", "left_image", "right_image"])
        for ts in ts_aligned:
            w.writerow([ts, f"cam0/data/{ts}.png", f"cam1/data/{ts}.png"])

    with open(os.path.join(out_dir, "camera_timestamps.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "camera_raw_ns", "camera_aligned_ns",
                    "timeshift_s"])
        for i, (raw, al) in enumerate(zip(ts_raw, ts_aligned)):
            w.writerow([start_index + i, raw, al, timeshift])

    with open(os.path.join(out_dir, "imu.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", "ax", "ay", "az", "gx", "gy", "gz"])
        for row in imu_rows:
            w.writerow([row[0], f"{row[1]:.9f}", f"{row[2]:.9f}",
                        f"{row[3]:.9f}", f"{row[4]:.9f}", f"{row[5]:.9f}",
                        f"{row[6]:.9f}"])


def _opencv_matrix(name: str, rows: int, cols: int, dt: str, data):
    """OpenCV FileStorage 矩阵块 (%%YAML 缩进 3 空格, 数据单行)。

    dt 必须显式给: ORB-SLAM3 的 Converter::toMatrix3d/toVector3d 用
    .at<float>() 读 IMU.T_b_c1, 写成 dt: d 会按 double 重解释 float
    → Sophus 崩溃 (见 离线ORB-SLAM报错修复记录.md 坑 A)。
    """
    vals = "".join(f"{v:.15g}, " for v in data)
    return (f"{name}: !!opencv-matrix\n"
            f"   rows: {rows}\n   cols: {cols}\n   dt: {dt}\n"
            f"   data: [ {vals[:-2]} ]")


def write_orb_calibration(out_dir: str, cal: dict, P0: np.ndarray,
                          P1: np.ndarray, serial: str, imu_rate: float,
                          fps: float):
    """生成 orb_calibration.yaml —— ORB-SLAM3 的 OpenCV FileStorage 配置。

    客户侧离线 runner 直接以 cv::FileStorage 解析此文件, 格式要求
    (离线ORB-SLAM报错修复记录.md 六项修复 + 必需键清单):
      - %%YAML:1.0 必须是文件第一行 (前面不能有注释)
      - Camera.width/height (不是 rows/cols)
      - IMU.Frequency (大写 F), 实数值必须带小数点 (486.0, 否则解析成
        INT 后 Settings::readParameter 的 isReal() 报错)
      - Stereo.ThDepth + 全部 10 个 Viewer.* (缺任何一个 aborting)
      - IMU.GyroWalk/AccWalk 非零 (0 会导致预积分协方差退化 → Sophus 崩溃)
      - IMU.T_b_c1 必须 dt: f (32F) 且数据单行

    方向约定 (SDK fays_atrak_types.h: "T_cn_imu // Bring points in {imu}
    frame to {camera n} frame"): 工厂 T_cam_imu = IMU系→左目系,
    正是 ORB-SLAM3 的 IMU.T_b_c1 (body→cam1), 直接写入。
    """
    fx0, cx0 = P0[0, 0], P0[0, 2]
    fy0, cy0 = P0[1, 1], P0[1, 2]
    fx1, cx1 = P1[0, 0], P1[0, 2]
    fy1, cy1 = P1[1, 1], P1[1, 2]
    b = abs(P1[0, 3] / fx1) if fx1 else 0.0   # 矫正基线 (m)
    bf = b * fx0                               # ORB-SLAM3 的 bf = baseline×fx
    T_cam_imu = cal["cam0"]["T_cam_imu"]
    imu_noise = {
        "NoiseGyro": 1.9e-05,    # gyroscope_noise_density (rad/s/√Hz)
        "NoiseAcc": 1.22e-04,    # accelerometer_noise_density (m/s²/√Hz)
        "GyroWalk": 0.0002,      # 非零必需 (修复记录.md 坑 B)
        "AccWalk": 0.00086,
    }
    viewer = [
        ("KeyFrameSize", 0.05), ("KeyFrameLineWidth", 1.0),
        ("GraphLineWidth", 0.9), ("PointSize", 2.0),
        ("CameraSize", 0.08), ("CameraLineWidth", 3.0),
        ("ViewpointX", 0.0), ("ViewpointY", -0.7),
        ("ViewpointZ", -1.8), ("ViewpointF", 500.0),
    ]
    # 浮点键必须带小数点: "1.0" 不能写成 "1" (OpenCV 解析成 INT,
    # Settings::readParameter 的 isReal() 报 must be a real number)
    def _real(v):
        return str(v)
    mat = _opencv_matrix("IMU.T_b_c1", 4, 4, "f", sum(T_cam_imu, []))
    with open(os.path.join(out_dir, "orb_calibration.yaml"), "w") as f:
        f.write("%YAML:1.0\n")   # 必须第一行
        f.write("# ORB-SLAM3 标定 (OpenCV FileStorage)\n")
        f.write(f"# 来源设备序列号: {serial}\n")
        f.write("# 输入图像: 已双目矫正 8-bit 灰度 PNG 640×400\n")
        f.write("# 相机—IMU 时间偏移已应用到 images.csv,\n")
        f.write("# 回放时 --camera-time-shift-s 必须为 0\n")
        f.write("---\n")
        f.write('File.version: "1.0"\n\n')
        f.write('# ------------------ Camera ------------------\n')
        f.write('Camera.type: "Rectified"   # 输入已双目矫正\n')
        f.write(f"Camera1.fx: {fx0:.9f}\n")
        f.write(f"Camera1.fy: {fy0:.9f}\n")
        f.write(f"Camera1.cx: {cx0:.6f}\n")
        f.write(f"Camera1.cy: {cy0:.6f}\n")
        f.write(f"Camera.fps: {fps}\n")
        f.write(f"Camera.bf: {bf:.9f}   # baseline × fx\n")
        f.write("Camera.RGB: 1\n")
        f.write("Camera.width: 640\n")
        f.write("Camera.height: 400\n\n")
        f.write('# ------------------- ORB --------------------\n')
        f.write("ORBextractor.nFeatures: 1200\n")
        f.write("ORBextractor.scaleFactor: 1.2\n")
        f.write("ORBextractor.nLevels: 8\n")
        f.write("ORBextractor.iniThFAST: 20\n")
        f.write("ORBextractor.minThFAST: 7\n\n")
        f.write('# ------------------- IMU --------------------\n')
        for k, v in imu_noise.items():
            f.write(f"IMU.{k}: {_real(v)}\n")
        f.write(f"IMU.Frequency: {imu_rate:.1f}   # 实测采样率\n")
        f.write("# IMU.T_b_c1: 4x4, body(左目)←IMU; 必须 dt:f (ORB-SLAM3 "
                "at<float>)\n")
        f.write(mat + "\n\n")
        f.write('# ---- Stereo (双写备份, FileStorage 忽略未读键) ----\n')
        f.write(f"Stereo.b: {b:.9f}\n")
        f.write(f"Stereo.bf: {bf:.9f}\n")
        f.write("Stereo.ThDepth: 40.0\n\n")
        f.write('# ------------------- Viewer --------------------\n')
        for k, v in viewer:
            f.write(f"Viewer.{k}: {_real(v)}\n")


def write_runtime_calibration(out_dir: str, cal: dict, serial: str,
                              size=(640, 400)):
    """生成 s80m_runtime_calibration.yaml —— 官方采集器配套运行时标定。

    字段与 offline_slam/fayssense_offline_dataset_recorder.cc 的
    load_runtime_calibration 一一对应 (K1/D1/K2/D2/R/T/T_b_c1 +
    serial_number/image_width/image_height/camera_to_imu_time_shift_s),
    客户侧用它做一致性验收 (prepare_rectification_and_validate_settings:
    orb yaml 的 Camera1.* 与由本文件重算的 P1 差 ≤1e-4, T_b_c1 ≤1e-5)。

    必须与 orb_calibration.yaml 同源同取向 (main() 中 swap 判定之后再调):
    K/D 取 swap 后 cam0/cam1 并按输出分辨率缩放, R/T 取 cam1.T_cn_cnm1,
    T_b_c1 取 cam0.T_cam_imu, timeshift 取本会话实际应用的值。
    """
    s = size[0] / cal["cam0"]["resolution"][0]
    K1 = np.array(cal["cam0"]["intrinsics"], dtype=np.float64) * s
    K2 = np.array(cal["cam1"]["intrinsics"], dtype=np.float64) * s
    D1 = np.array(cal["cam0"]["distortion_coeffs"], dtype=np.float64)
    D2 = np.array(cal["cam1"]["distortion_coeffs"], dtype=np.float64)
    T01 = np.array(cal["cam1"]["T_cn_cnm1"], dtype=np.float64)
    R, t = T01[:3, :3], T01[:3, 3]
    T_b_c1 = cal["cam0"]["T_cam_imu"]
    shift = cal["cam0"]["timeshift_cam_imu"]

    def _K(name, vals):
        return _opencv_matrix(name, 3, 3, "d",
                              (vals[0], 0.0, vals[2],
                               0.0, vals[1], vals[3],
                               0.0, 0.0, 1.0))

    def _D(name, vals):
        return _opencv_matrix(name, 1, 4, "d", vals)

    with open(os.path.join(out_dir, "s80m_runtime_calibration.yaml"),
              "w") as f:
        f.write("%YAML:1.0\n")
        f.write("# S80M 运行时标定 (离线采集器格式, 640×400)\n")
        f.write(f"serial_number: \"{serial}\"\n")
        f.write(f"image_width: {size[0]}\n")
        f.write(f"image_height: {size[1]}\n")
        f.write(f"camera_to_imu_time_shift_s: {shift:.15g}\n\n")
        f.write(_K("K1", K1) + "\n")
        f.write(_D("D1", D1) + "\n")
        f.write(_K("K2", K2) + "\n")
        f.write(_D("D2", D2) + "\n")
        f.write(_opencv_matrix("R", 3, 3, "d", R.flatten()) + "\n")
        f.write(_opencv_matrix("T", 3, 1, "d", t.flatten()) + "\n")
        f.write(_opencv_matrix("T_b_c1", 4, 4, "f",
                               sum(T_b_c1, [])) + "\n")


def verify_against_device_json(cal: dict, tol=1e-3):
    """与设备级标定 config/s80m_stereo_calibration.json (ROM dump) 交叉校验。

    防 calib.yaml 过期/错设备: 内参/基线/外参/timeshift 任一超差即报错。
    返回问题列表 (空 = 通过)。
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "config", "s80m_stereo_calibration.json")
    if not os.path.exists(path):
        print(f"  [标定校验] 跳过: 无设备级标定 {path}")
        return [], "skipped"
    with open(path) as f:
        dev = json.load(f)
    issues = []
    for side, key in (("cam0", "left_camera"), ("cam1", "right_camera")):
        exp = dev[key]["intrinsic"]
        got = cal[side]["intrinsics"]
        for name, e, g in (("fx", exp[0], got[0]), ("fy", exp[1], got[1]),
                           ("cx", exp[2], got[2]), ("cy", exp[3], got[3])):
            if abs(e - g) > tol:
                issues.append(f"{side}.{name}: yaml {g} vs ROM {e}")
    T01 = np.array(cal["cam1"]["T_cn_cnm1"], dtype=np.float64)
    if abs(abs(T01[0, 3]) - dev.get("baseline", 0.0)) > 1e-4:
        issues.append(f"baseline: yaml {abs(T01[0,3])} vs ROM {dev.get('baseline')}")
    rot_err = np.linalg.norm(T01[:3, :3].flatten() -
                             np.array(dev["rotation"]).flatten())
    if rot_err > 1e-4:
        issues.append(f"rotation: yaml vs ROM 差 {rot_err}")
    trans_err = np.linalg.norm(T01[:3, 3] - np.array(dev["translation"]))
    if trans_err > 1e-4:
        issues.append(f"translation: yaml vs ROM 差 {trans_err}")
    shift = cal["cam0"]["timeshift_cam_imu"]
    if abs(shift - dev.get("cam_imu_timeshift", 0.0)) > 1e-9:
        issues.append(f"timeshift: yaml {shift} vs ROM {dev.get('cam_imu_timeshift')}")
    return issues, "device-rom-json"


def write_metadata_yaml(out_dir: str, session: str, stats: dict,
                        n_imu: int, imu_rate: float, fps: float):
    lines = [
        "# 交付元数据 (EgoData 会话 → 离线 SLAM 数据集转换记录)",
        f"source_session: {session}",
        f"exported_at: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"image_format: png8-gray-rectified",
        f"image_size: [640, 400]",
        f"fps: {fps}",
        f"frames: {stats['cam0']}",
        f"imu_samples: {n_imu}",
        f"imu_rate_hz: {imu_rate:.1f}",
        "imu_note: 原始 SDK 1006Hz 采样受采集轮询限制约为 500Hz,",
        "          未做插值/滤波/积分, 每个样本为 SDK 输出的 SI 物理量",
        f"time_base: imu_sensor_clock_ns",
        f"timeshift_applied_s: {stats.get('timeshift', 0.0)}",
        f"timeshift_note: 偏移已应用一次, 运行离线 SLAM 时不得重复传入",
        f"serial: {stats.get('serial', '')}",
        f"calibration_verified: {stats.get('calib_verified', 'n/a')}",
        f"orientation_checked: {stats.get('orientation', 'n/a')}",
        "calibration_note: 设备 ROM 工厂标定 (config/calib/calib.yaml), "
        "与 config/s80m_stereo_calibration.json 交叉校验通过",
        f"runtime_calibration: s80m_runtime_calibration.yaml",
        "",
    ]
    with open(os.path.join(out_dir, "metadata.yaml"), "w") as f:
        f.write("\n".join(lines))


def validate(ts_aligned: list, imu_rows: list) -> list:
    """校验时间戳严格递增 + IMU 覆盖。返回问题列表(空=通过)。"""
    issues = []
    for name, arr in (("images.csv", ts_aligned), ("imu.csv",
                                                    [r[0] for r in imu_rows])):
        bad = [i for i in range(1, len(arr)) if arr[i] <= arr[i - 1]]
        if bad:
            issues.append(f"{name}: {len(bad)} 处时间戳未严格递增, 前5处: {bad[:5]}")
    if ts_aligned and imu_rows:
        if imu_rows[0][0] > ts_aligned[0] or imu_rows[-1][0] < ts_aligned[-1]:
            issues.append(
                "imu.csv 时间范围未完整覆盖 images.csv"
                f" (imu [{imu_rows[0][0]}, {imu_rows[-1][0]}]"
                f" vs cam [{ts_aligned[0]}, {ts_aligned[-1]}])")
    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True, help="EgoData 会话目录")
    ap.add_argument("--out", default=_DEFAULT_OUT, help="输出根目录")
    ap.add_argument("--calib", default=_FACTORY_CALIB, help="工厂标定 yaml")
    ap.add_argument("--serial", default="3500000262190088",
                    help="设备序列号 (用于元数据/校验)")
    args = ap.parse_args()

    session_dir = os.path.abspath(args.session)
    if not os.path.isdir(session_dir):
        sys.exit(f"[错误] 会话目录不存在: {session_dir}")
    if not os.path.isfile(args.calib):
        sys.exit(f"[错误] 找不到工厂标定: {args.calib}")

    print(f"会话: {session_dir}")
    cal = parse_calib_yaml(args.calib)
    # 读取双目相机的内参: 与设备级 ROM dump (config/s80m_stereo_calibration.json)
    # 交叉校验, 防 calib.yaml 过期/错设备
    calib_issues, calib_verified = verify_against_device_json(cal)
    if calib_issues:
        print("  [标定校验失败] calib.yaml 与设备 ROM dump 不一致:")
        for i in calib_issues:
            print(f"    - {i}")
        sys.exit(1)
    print(f"  [标定校验] {calib_verified}")
    data = load_session(session_dir)
    frames = data["frames"]
    imu_rows = sorted(data["imu_rows"], key=lambda r: r[0])
    fps = float(data["metadata"].get("fps") or _FPS_FALLBACK)
    print(f"  帧数={len(frames)}  IMU 样本数={len(imu_rows)}  "
          f"IMU 时间范围=[{imu_rows[0][0]}, {imu_rows[-1][0]}]ns  "
          f"录制 fps={fps}")

    # 1) 相机时钟: 32 位回卷修正 + IMU 时基锚定 → 应用 timeshift
    cam_abs = unwrap_camera_clock(frames, imu_rows[0][0],
                                  period_ns=round(1e9 / fps))
    timeshift = cal["cam0"]["timeshift_cam_imu"]
    ts_aligned = [int(c + round(timeshift * 1e9)) for c in cam_abs]

    # 客户侧硬检查: IMU 时间范围必须完整覆盖第一张到最后一张图像。
    # 相机先于 IMU 流启动的会话, 对齐后可能有前导帧早于 IMU 首样本 → 裁剪。
    first_imu, last_imu = imu_rows[0][0], imu_rows[-1][0]
    drop_n = sum(1 for t in ts_aligned if t < first_imu)
    if drop_n:
        print(f"  [裁剪] 前 {drop_n} 帧对齐后早于 IMU 首样本 "
              f"({(first_imu - ts_aligned[0]) / 1e6:.2f}ms), 丢弃以满足覆盖检查")
    ts_aligned = ts_aligned[drop_n:]
    cam_abs = cam_abs[drop_n:]
    if ts_aligned[-1] > last_imu:
        sys.exit(f"[错误] 末帧 {ts_aligned[-1]} 晚于 IMU 末样本 {last_imu}, "
                 f"IMU 覆盖不完整, 无法导出")

    # 检查对齐质量: 相机时间应在对应帧的 IMU 窗口内
    d0 = (cam_abs[0] - imu_rows[0][0]) / 1e9
    d1 = (cam_abs[-1] - imu_rows[-1][0]) / 1e9
    print(f"  相机-IMU 对齐检查: 首帧差 {d0*1000:+.2f}ms  末帧差 {d1*1000:+.2f}ms")

    # 2) 左右目配对自检 + 矫正映射
    size = (640, 400)
    video_dir = os.path.join(session_dir, "videos")
    if check_orientation(video_dir, cal, size):
        # 交换标定: 文件 stereo_left ↔ 标定 cam1 (T_cn_cnm1 保持 second←first)
        old0, old1 = cal["cam0"], cal["cam1"]
        cal["cam0"], cal["cam1"] = old1, old0
        cal["cam1"]["T_cn_cnm1"] = old1["T_cn_cnm1"]
        print("  [自检] 已交换标定左右")
        orientation = "swapped"
    else:
        orientation = "normal"
    maps = build_rect_maps(cal, size)
    fx0 = maps["P0"][0, 0]
    bf = maps["P1"][0, 3] / fx0
    print(f"  矫正内参 fx={fx0:.2f} (640×400)  矫正基线={bf*1000:.2f}mm")

    # 3) 输出目录 (同时间戳已存在时加 _v2/_v3 后缀, 绝不覆盖旧 session)
    created = data["metadata"].get("created_at", time.time())
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(created))
    out_dir = os.path.join(os.path.abspath(args.out), f"session_{stamp}")
    v = 2
    while os.path.exists(out_dir):
        out_dir = os.path.join(os.path.abspath(args.out),
                               f"session_{stamp}_v{v}")
        v += 1
    os.makedirs(out_dir)
    print(f"输出: {out_dir}")

    # 4) 帧提取 (灰度+缩放+矫正)
    stats = extract_frames(video_dir, ts_aligned, maps, out_dir, size,
                           skip=drop_n)
    print(f"  帧写入: cam0={stats['cam0']}  cam1={stats['cam1']}")

    # 5) CSV / 标定 / 元数据
    write_csvs(out_dir, ts_aligned, cam_abs, imu_rows, timeshift,
               start_index=drop_n)
    dur = (ts_aligned[-1] - ts_aligned[0]) / 1e9
    imu_rate = len(imu_rows) / dur
    write_orb_calibration(out_dir, cal, maps["P0"], maps["P1"],
                          args.serial, imu_rate, fps)
    write_runtime_calibration(out_dir, cal, args.serial, size)
    write_metadata_yaml(out_dir, os.path.basename(session_dir),
                        {**stats, "timeshift": timeshift,
                         "serial": args.serial, "orientation": orientation,
                         "calib_verified": calib_verified},
                        len(imu_rows), imu_rate, fps)

    # 6) 校验
    issues = validate(ts_aligned, imu_rows)
    print(f"  IMU 实测采样率 ≈ {imu_rate:.0f}Hz  数据时长 ≈ {dur:.1f}s")
    if issues:
        for i in issues:
            print(f"  [校验失败] {i}")
        sys.exit(1)
    print("  ✓ 时间戳严格递增, IMU 覆盖完整")

    # 7) 打包
    tgz = out_dir + ".tar.gz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(out_dir, arcname=os.path.basename(out_dir))
    print(f"✓ 打包完成: {tgz}")
    print(f"✓ 离线数据集: {out_dir}")


if __name__ == "__main__":
    main()

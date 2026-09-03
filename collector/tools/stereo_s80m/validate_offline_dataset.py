#!/usr/bin/env python3
"""离线 SLAM 数据集本地验收 —— 复刻客户侧两份检查:

1. cv::FileStorage 格式检查 (离线ORB-SLAM报错修复记录.md 六项修复 + 必需键):
   - %YAML:1.0 必须是首行
   - 必需键齐全 (Camera.*/Camera1.*/Stereo.*/IMU.*/Viewer.*)
   - 实数值键必须是 REAL 节点 (不能是 INT)
   - IMU.T_b_c1 必须是 32F (dt: f)

2. 官方采集器 prepare_rectification_and_validate_settings 复刻
   (offline_slam/fayssense_offline_dataset_recorder.cc):
   用 s80m_runtime_calibration.yaml 的 K/D/R/T 在 640×400 重算
   cv::fisheye::stereoRectify → P1/P2, 校验:
     Camera.type == "Rectified"
     Camera.width/height == image_width/height
     Camera1.fx/fy/cx/cy 与 P1 差 ≤ 1e-4
     Stereo.b 与基线 (P2[0,3]/P2[0,0]) 差 ≤ 1e-4
     IMU.T_b_c1 与 runtime T_b_c1 差 (Inf-norm) ≤ 1e-5
     Camera.fps / IMU.Frequency > 0

3. 数据完整性: 时间戳严格递增, IMU 覆盖, 帧数, 图像属性。

用法:
  python validate_offline_dataset.py <session_dir>
"""

import csv
import os
import sys

import cv2
import numpy as np

REQUIRED_KEYS = [
    "Camera.type", "Camera.width", "Camera.height", "Camera.fps", "Camera.RGB",
    "Camera1.fx", "Camera1.fy", "Camera1.cx", "Camera1.cy",
    "Stereo.b", "Stereo.ThDepth",
    "IMU.NoiseGyro", "IMU.NoiseAcc", "IMU.GyroWalk", "IMU.AccWalk",
    "IMU.Frequency", "IMU.T_b_c1",
    "Viewer.KeyFrameSize", "Viewer.KeyFrameLineWidth", "Viewer.GraphLineWidth",
    "Viewer.PointSize", "Viewer.CameraSize", "Viewer.CameraLineWidth",
    "Viewer.ViewpointX", "Viewer.ViewpointY", "Viewer.ViewpointZ",
    "Viewer.ViewpointF",
]
REAL_KEYS = [
    "Camera.fps", "Camera1.fx", "Camera1.fy", "Camera1.cx", "Camera1.cy",
    "Stereo.b", "Stereo.ThDepth",
    "IMU.NoiseGyro", "IMU.NoiseAcc", "IMU.GyroWalk", "IMU.AccWalk",
    "IMU.Frequency",
    "Viewer.KeyFrameSize", "Viewer.KeyFrameLineWidth", "Viewer.GraphLineWidth",
    "Viewer.PointSize", "Viewer.CameraSize", "Viewer.CameraLineWidth",
    "Viewer.ViewpointX", "Viewer.ViewpointY", "Viewer.ViewpointZ",
    "Viewer.ViewpointF",
]
_REAL = getattr(cv2, "FileNode_REAL", None) or 2  # cv2.FileNode.REAL


def check_orb_yaml(session: str) -> list:
    """检查 orb_calibration.yaml 格式。返回问题列表(空=通过)。"""
    issues = []
    path = os.path.join(session, "orb_calibration.yaml")
    with open(path) as f:
        first = f.readline()
    if not first.startswith("%YAML:1.0"):
        issues.append(f"orb_calibration.yaml 首行不是 %YAML:1.0: {first!r}")

    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return [f"cv::FileStorage 无法打开 {path}"]
    for key in REQUIRED_KEYS:
        node = fs.getNode(key)
        if node.empty():
            issues.append(f"缺少必需键 {key}")
    for key in REAL_KEYS:
        node = fs.getNode(key)
        if not node.empty() and node.type() != _REAL:
            issues.append(f"{key} 不是 REAL 节点 (type={node.type()}, "
                          f"会被 isReal() 拒绝)")
    tbc1 = fs.getNode("IMU.T_b_c1")
    if not tbc1.empty():
        m = tbc1.mat()
        if m.dtype != np.float32:
            issues.append(f"IMU.T_b_c1 必须是 32F (dt:f), 实际 {m.dtype}")
        if m.shape != (4, 4):
            issues.append(f"IMU.T_b_c1 形状 {m.shape}, 应为 4x4")
    fs.release()
    return issues


def check_runtime_consistency(session: str, size=(640, 400)) -> list:
    """复刻官方 recorder 的 prepare_rectification_and_validate_settings。"""
    issues = []
    rt_path = os.path.join(session, "s80m_runtime_calibration.yaml")
    if not os.path.exists(rt_path):
        return [f"缺少 {rt_path} (官方采集器产物)"]
    fs = cv2.FileStorage(rt_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return [f"cv::FileStorage 无法打开 {rt_path}"]

    def mat(name):
        node = fs.getNode(name)
        return node.mat() if not node.empty() else None

    K1, D1, K2, D2 = (mat("K1"), mat("D1"), mat("K2"), mat("D2"))
    R, T = mat("R"), mat("T")
    rt_tbc1 = mat("T_b_c1")
    width = int(fs.getNode("image_width").real() or 0)
    height = int(fs.getNode("image_height").real() or 0)
    shift = fs.getNode("camera_to_imu_time_shift_s").real()
    fs.release()
    for name, m in (("K1", K1), ("D1", D1), ("K2", K2), ("D2", D2),
                    ("R", R), ("T", T), ("T_b_c1", rt_tbc1)):
        if m is None:
            issues.append(f"runtime yaml 缺少矩阵 {name}")
    if issues:
        return issues
    if (width, height) != size:
        issues.append(f"runtime image size {(width, height)} != {size}")

    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY)

    orb = cv2.FileStorage(os.path.join(session, "orb_calibration.yaml"),
                          cv2.FILE_STORAGE_READ)
    ctype = orb.getNode("Camera.type").string()
    fx = orb.getNode("Camera1.fx").real()
    fy = orb.getNode("Camera1.fy").real()
    cx = orb.getNode("Camera1.cx").real()
    cy = orb.getNode("Camera1.cy").real()
    baseline = orb.getNode("Stereo.b").real()
    fps = orb.getNode("Camera.fps").real()
    freq = orb.getNode("IMU.Frequency").real()
    orb_tbc1 = orb.getNode("IMU.T_b_c1").mat().astype(np.float64)
    orb.release()

    if ctype != "Rectified":
        issues.append(f"Camera.type = {ctype!r}, 应为 'Rectified'")
    if fps <= 0:
        issues.append(f"Camera.fps = {fps}")
    if freq <= 0:
        issues.append(f"IMU.Frequency = {freq}")
    expected_baseline = abs(P2[0, 3] / P2[0, 0])
    for name, got, exp in (("fx", fx, P1[0, 0]), ("fy", fy, P1[1, 1]),
                           ("cx", cx, P1[0, 2]), ("cy", cy, P1[1, 2]),
                           ("baseline", baseline, expected_baseline)):
        if abs(got - exp) > 1e-4:
            issues.append(f"Camera1.{name}/Stereo.b {got:.9f} vs 矫正P1 "
                          f"{exp:.9f} (差 {abs(got-exp):.3e} > 1e-4)")
    imu_err = np.max(np.abs(orb_tbc1 - rt_tbc1.astype(np.float64)))
    if imu_err > 1e-5:
        issues.append(f"IMU.T_b_c1 与 runtime 差 {imu_err:.3e} > 1e-5")
    return issues


def check_data(session: str) -> list:
    """时间戳/IMU/帧数/图像属性。返回 (issues, info)。"""
    issues, info = [], []
    with open(os.path.join(session, "images.csv")) as f:
        rows = list(csv.DictReader(f))
    ts = [int(r["timestamp_ns"]) for r in rows]
    if any(ts[i] <= ts[i - 1] for i in range(1, len(ts))):
        issues.append("images.csv 时间戳未严格递增")
    for r in rows[:3] + rows[-3:]:
        for side in ("left_image", "right_image"):
            p = os.path.join(session, r[side])
            if not os.path.exists(p):
                issues.append(f"图像缺失: {p}")

    with open(os.path.join(session, "imu.csv")) as f:
        imu = list(csv.DictReader(f))
    its = [int(r["timestamp_ns"]) for r in imu]
    if any(its[i] <= its[i - 1] for i in range(1, len(its))):
        issues.append("imu.csv 时间戳未严格递增")
    if its[0] > ts[0] or its[-1] < ts[-1]:
        issues.append("IMU 未覆盖相机时间范围")
    info.append(f"frames={len(ts)} imu={len(imu)} "
                f"rate={len(imu)/((ts[-1]-ts[0])/1e9):.1f}Hz")

    # 图像属性 (抽 1 张)
    p = os.path.join(session, rows[0]["left_image"])
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        issues.append(f"图像无法解码: {p}")
    elif img.shape != (400, 640) or img.dtype != np.uint8:
        issues.append(f"图像 {img.shape} {img.dtype}, 应为 640×400 8bit")
    else:
        info.append(f"image={img.shape} black_ratio={(img < 8).mean():.3f}")
    return issues, info


def main():
    session = os.path.abspath(sys.argv[1])
    failed = False
    for name, fn in (("格式(FileStorage+必需键)", check_orb_yaml),
                     ("官方验收(rectify一致性)", check_runtime_consistency),
                     ("数据完整性", check_data)):
        print(f"== {name} ==")
        result = fn(session)
        issues, info = (result if name == "数据完整性"
                        else (result, []))
        if not issues:
            print("  ✓ 通过")
        for i in issues:
            failed = True
            print(f"  [FAIL] {i}")
        for i in info:
            print(f"  info: {i}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

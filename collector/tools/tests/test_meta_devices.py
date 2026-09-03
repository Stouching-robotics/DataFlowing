"""录制 meta 测试（无真机）—— v1.1.0 任务池化布局：info.json 任务级
（devices 紧凑段/device_names/标定 dict/video_extensions）、data parquet
稀疏列（wall_time/hardware_ns/IMU/手部占位）、episodes 每段单行文件、任务级
stats/tasks、旧单值标定路径、S80M 静态标定注入、手套列名绑定。

用法:
    QT_QPA_PLATFORM=offscreen venv/bin/python tools/tests/test_meta_devices.py

覆盖:
  1. 多设备 start_episode: 池化视频布局（videos/chunk-NNN/<key>/episode-NNN.mp4，
     深度槽不建目录；无 metadata.json / calibration/）
  2. data parquet: 键列 + wall_time/hardware_ns + 稀疏观测列（present 传感器
     /IMU 才写）+ 手部占位列 + status 列
  3. episodes 每段文件: episode_index/length/calibration JSON
  4. info.json 任务级: devices 紧凑段 + device_names + calibration dict
     （head_stereo=首台 + {slot前缀}_calibration）+ video_extensions +
     features 只声明存在的列；stats（自含 count）/tasks 生成
  5. 旧单值路径（无 devices）→ info.json calibration head_stereo 行为不变
  6. S80M 静态标定文件可解析为非零内参（注入链路源）
  7. assign_glove_sensor_role: MAC 绑定 + 空余名分配 + 重连保持
"""
import os
import sys
import json
import glob
import time
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pyarrow.parquet as pq

from config import settings
from core.egodata_writer import EgoDataWriter
from core.calibration import StereoCalibration, CameraIntrinsics
from core.helpers import (pooled_info_path, pooled_stats_path,
                          pooled_tasks_jsonl_path, pooled_episodes_path)

OUT_ROOT = "/tmp/meta_devices_test"
FAILS = []


def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAILS.append(msg)


def mk_calib(fx):
    return StereoCalibration(
        left_camera=CameraIntrinsics(intrinsic=[fx, fx, 320, 240]),
        right_camera=CameraIntrinsics(intrinsic=[fx + 1, fx + 1, 321, 241]),
        baseline=0.05, depth_scale=0.001)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    shutil.rmtree(OUT_ROOT, ignore_errors=True)
    os.makedirs(OUT_ROOT, exist_ok=True)

    print("── 1-4. 池化布局：视频 / data parquet / episodes / info.json ──")
    w = EgoDataWriter()
    cameras = {
        "d435_rgb": (720, 1280),
        "d435_depth": (480, 848),
        "d435_rgb_2": (720, 1280),
        "d435_depth_2": (480, 848),
        "stereo_left": (800, 1280),
        "stereo_right": (800, 1280),
        "head_left_rgb": (480, 640),
    }
    devices = [
        {"key": "d435:111", "kind": "d435", "name": "顶部深度相机",
         "serial": "111", "slots": ["d435_rgb", "d435_depth"]},
        {"key": "d435:222", "kind": "d435", "name": "第二台",
         "serial": "222", "slots": ["d435_rgb_2", "d435_depth_2"]},
        {"key": "s80m:ftdi", "kind": "s80m", "name": "双目",
         "slots": ["stereo_left", "stereo_right"]},
        {"key": "uvc:fake", "kind": "uvc", "name": "桌面摄像头",
         "slots": ["head_left_rgb"]},
    ]
    calibrations = {
        "d435:111": mk_calib(600.0),
        "d435:222": mk_calib(700.0),
        "s80m:ftdi": mk_calib(800.0),
    }
    ok = w.start_episode(
        OUT_ROOT, cameras, 30.0,
        sensors=["right_glove"],
        device_ids=["uvc:fake"],
        depth_slots=["d435_depth", "d435_depth_2"],
        devices=devices, calibrations=calibrations)
    check(ok, "start_episode（多设备）成功")
    check(w.episode_index == 1, f"首个 episode 序号=1: {w.episode_index}")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hw_base = 1_000_000_000
    for i in range(3):
        for slot in ["head_left_rgb"]:
            w.write_video_frame(slot, frame)
        for slot in ["d435_rgb", "d435_rgb_2"]:
            w.write_video_frame(slot, np.zeros((720, 1280, 3), np.uint8))
        for slot in ["stereo_left", "stereo_right"]:
            w.write_video_frame(slot, np.zeros((800, 1280, 3), np.uint8))
        # 稀疏观测：仅第 0 帧有传感器数据与 IMU 样本
        sensor_data = {"right_glove": np.ones(256, dtype=np.float32)} \
            if i == 0 else None
        imu = [(hw_base + i * 33_333_333, 0.0, 0.0, 0.0, 0.0, 0.0, 9.8)] \
            if i == 0 else None
        w.write_frame_row(i, i / 30.0, sensors=sensor_data,
                          hardware_ns=hw_base + i * 33_333_333,
                          imu_samples=imu)
    w.end_episode()
    task_dir = w.task_dir

    # 1. 池化视频布局：RGB episode-NNN.mp4，深度槽不建目录，无旧格式残留
    check(not os.path.isdir(os.path.join(task_dir, "videos", "chunk-000",
                                         "d435_depth"))
          and not os.path.isdir(os.path.join(task_dir, "videos", "chunk-000",
                                             "d435_depth_2")),
          "深度槽不建视频目录")
    check(os.path.isfile(os.path.join(task_dir, "videos", "chunk-000",
                                      "head_left_rgb", "episode-000.mp4")),
          "UVC 视频 episode-000.mp4 落盘")
    check(not os.path.exists(os.path.join(task_dir, "metadata.json"))
          and not os.path.isdir(os.path.join(task_dir, "calibration"))
          and not os.path.exists(os.path.join(task_dir, "timestamps.json")),
          "无 metadata.json / calibration/ / timestamps.json（池化）")

    # 2. data parquet：键列 + wall_time/hardware_ns + 稀疏观测列 + 占位列
    t = pq.read_table(os.path.join(task_dir, "data", "chunk-000",
                                   "episode-000.parquet"))
    cols = set(t.column_names)
    check({"episode_index", "frame_index", "timestamp", "task_index",
           "wall_time", "hardware_ns", "action",
           "observation.right_glove", "imu_ts_ns", "observation.imu",
           "observation.left_hand_pose", "observation.right_hand_pose",
           "status.uvc:fake"} <= cols,
          f"data parquet 列齐全: {sorted(cols)}")
    check(t.num_rows == 3, f"data parquet 3 行: {t.num_rows}")
    check(t.column("wall_time")[0].as_py() > 1.7e9,
          "wall_time 为绝对 Unix 秒")
    check(t.column("hardware_ns")[2].as_py() == hw_base + 2 * 33_333_333,
          "hardware_ns 逐帧正确")
    check(len(t.column("observation.imu")[0].as_py()) == 1
          and len(t.column("observation.imu")[1].as_py()) == 0,
          "IMU 稀疏：有样本帧 1 个/无样本帧空列表")
    check(t.column("observation.right_hand_pose")[0].as_py() ==
          [0.0] * settings.HAND_POSE_DIM,
          "手部关键点占位零")

    # 3. episodes 每段一文件（含 calibration JSON）
    ep_t = pq.read_table(pooled_episodes_path(task_dir, 1))
    rows = ep_t.to_pylist()
    check(len(rows) == 1 and rows[0]["episode_index"] == 1
          and rows[0]["length"] == 3 and rows[0]["duration_sec"] > 0,
          f"episodes 每段文件 1 行: {rows}")
    calib_row = json.loads(rows[0]["calibration"])
    check(calib_row["head_stereo"]["left_camera"]["intrinsic"][0] == 600.0,
          "episodes 行 calibration head_stereo fx=600")

    # 4. info.json 任务级 + stats/tasks
    info = read_json(pooled_info_path(task_dir))
    check(info.get("format") == "pooled_episodes_v1"
          and info.get("chunks_size") == 1000
          and info.get("total_episodes") == 1,
          f"info.json format/chunks_size/total_episodes: "
          f"{info.get('format')}, {info.get('chunks_size')}, "
          f"{info.get('total_episodes')}")
    check(info["calibration"]["head_stereo"]["left_camera"]["intrinsic"][0]
          == 600.0,
          "info.json calibration head_stereo = 首台 (fx=600)")
    check(info["calibration"]["d435_rgb_calibration"]
          ["left_camera"]["intrinsic"][0] == 700.0,
          "d435_rgb_calibration = 第二台 (fx=700, 前缀去消歧编号)")
    check(info["calibration"]["stereo_left_calibration"]
          ["left_camera"]["intrinsic"][0] == 800.0,
          "stereo_left_calibration = S80M (fx=800)")
    idevs = {d["key"]: d for d in info.get("devices", [])}
    check(idevs.get("d435:111", {}).get("name") == "顶部深度相机"
          and idevs.get("d435:111", {}).get("slots") == ["d435_rgb",
                                                         "d435_depth"],
          "info.json devices 紧凑段")
    check(info.get("device_names", {}).get("d435_rgb") == "顶部深度相机"
          and info["device_names"].get("stereo_left") == "双目",
          f"info.json device_names: {info.get('device_names')}")
    check("d435_depth" not in info["cameras"]
          and "d435_rgb" in info["cameras"],
          "info.json cameras 深度槽由注册驱动过滤")
    check(info["video_extensions"].get("d435_rgb") == "mp4"
          and info["video_extensions"].get("d435_depth") in ("mp4", "mkv"),
          f"video_extensions: {info.get('video_extensions')}")
    feats = info.get("features", {})
    check("observation.right_glove" in feats and "observation.imu" in feats
          and "action" in feats,
          f"features 只声明存在的列: {sorted(feats)}")
    check(os.path.isfile(pooled_stats_path(task_dir))
          and os.path.isfile(pooled_tasks_jsonl_path(task_dir)),
          "stats/tasks.jsonl 任务级生成")
    stats = read_json(pooled_stats_path(task_dir))
    check("observation.right_glove" in stats and "observation.imu" in stats
          and stats.get("action") == {"count": 0, "mean": [0.0],
                                      "std": [1.0], "min": [0.0],
                                      "max": [0.0]},
          f"stats.json 只统计存在的列: {sorted(stats)}")
    check(stats["observation.right_glove"]["count"] == 3
          and stats["observation.imu"]["count"] > 0,
          f"stats.json 自含 count 累加器: "
          f"{stats['observation.right_glove']['count']}, "
          f"{stats['observation.imu']['count']}")

    print("── 5. 旧单值路径兼容 ──")
    shutil.rmtree(OUT_ROOT, ignore_errors=True)
    w2 = EgoDataWriter()
    ok = w2.start_episode(OUT_ROOT, {"head_left_rgb": (480, 640)}, 30.0,
                          calibration=mk_calib(555.0))
    check(ok, "旧单值 start_episode 成功")
    w2.end_episode()
    info2 = read_json(pooled_info_path(w2.task_dir))
    check(info2["calibration"]["head_stereo"]["left_camera"]["intrinsic"][0]
          == 555.0,
          "旧路径仍写 head_stereo 标定（fx=555, 存 info.json）")

    print("── 6. S80M 静态标定注入源 ──")
    s80m_calib_file = os.path.join(settings.BASE_DIR, "config",
                                   "s80m_stereo_calibration.json")
    if os.path.isfile(s80m_calib_file):
        c = StereoCalibration.load(s80m_calib_file)
        fx = c.left_camera.intrinsic[0]
        check(fx > 100, f"S80M 静态标定解析非零内参 (fx={fx:.1f})")
    else:
        check(False, "config/s80m_stereo_calibration.json 缺失")

    print("── 7. 手套列名绑定 ──")
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp_path = tmp.name
    tmp.close()
    _orig = settings.DEVICE_NAMES_FILE
    settings.DEVICE_NAMES_FILE = tmp_path
    try:
        r1 = settings.assign_glove_sensor_role("ble:AA:11:22:33:44:55")
        r2 = settings.assign_glove_sensor_role("ble:BB:22:33:44:55:66")
        r1b = settings.assign_glove_sensor_role("ble:AA:11:22:33:44:55")
        check(r1 == "right_glove" and r2 == "left_glove",
              f"空余名顺序分配: {r1}, {r2}")
        check(r1b == r1, f"重连保持原列名: {r1b}")
        r3 = settings.assign_glove_sensor_role("ble:CC:33:44:55:66:77")
        check(r3 in settings.SENSOR_NAMES, f"无空余名兜底: {r3}")
    finally:
        settings.DEVICE_NAMES_FILE = _orig
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项未通过")
        return 1
    print("PASS: 录制 meta 测试全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""回放多帧率测试 —— 主时钟 + 逐路独立 seek + 用户命名 + 动态传感器面板。

用法:
    QT_QPA_PLATFORM=offscreen venv/bin/python tools/tests/test_playback_multifps.py

合成会话：cam_a 30fps×60 帧 + cam_b 25fps×50 帧（同 2s 时长）+
right_glove 传感器 parquet（30fps×60 行）+ 蓝牙耳机设备（无画面）。

覆盖:
  1. _get_effective_fps 主时钟 = max（旧数据回退链）
  2. 加载：_cam_fps/_cam_total/_total_frames 主时钟模型
  3. 用户命名叠加 + 统一网格（摄像机格 + 传感器格同入 CameraGrid）
  4. 逐路独立 seek（低帧率路重复帧保持 / 末帧截断 / 时间标签 / 帧号）
  5. 多帧率混合 → 传感器按时间二分；无画面 BLE 计入信息栏
  6. SensorTimeline.nearest_for_column_time 单调二分 / 非单调 argmin
  7. 幽灵传感器过滤（无 BLE 会话的 imu 等特征不建传感器格）
"""
import os
import sys
import json
import time
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
from PyQt5.QtWidgets import QApplication

from config.i18n import tr
from core.session_timeline import SensorTimeline
from ui.playback_dialog import PlaybackDialog, _get_effective_fps

FAILS = []


def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAILS.append(msg)


def _make_video(path, fps, n):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), i % 255, np.uint8))
    w.release()


def _make_session(root: str) -> str:
    d = os.path.join(root, "20260818_120000_synth")
    os.makedirs(os.path.join(d, "meta"))
    for slot, fps, n in (("cam_a", 30, 60), ("cam_b", 25, 50)):
        vdir = os.path.join(d, "videos", slot)
        os.makedirs(vdir)
        _make_video(os.path.join(vdir, "chunk_000000.mp4"), fps, n)

    # 传感器 parquet（per-sensor 新结构，30fps×60 行）
    pdir = os.path.join(d, "data", "right_glove", "chunk-0000")
    os.makedirs(pdir)
    n = 60
    ts = np.arange(n, dtype=np.float64) / 30.0
    mats = [np.ravel(np.random.RandomState(i).rand(16, 16).astype(np.float32) * 2000)
            for i in range(n)]
    tbl = pa.table({
        "episode_index": pa.array(np.zeros(n, np.int64)),
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
        "timestamp": pa.array(ts),
        "observation.right_glove": pa.array(mats, type=pa.list_(pa.float32(), 256)),
    })
    pq.write_table(tbl, os.path.join(pdir, "chunk_000000.parquet"))

    info = {
        "task_name": "synth",
        "fps": 30,
        "cameras": {"cam_a": {"fps": 30}, "cam_b": {"fps": 25}},
        "sensors": ["right_glove"],
        "devices": [
            {"key": "uvc:a", "kind": "uvc", "name": "桌面摄像头",
             "slots": ["cam_a"]},
            {"key": "uvc:b", "kind": "uvc", "name": "侧方摄像头",
             "slots": ["cam_b"]},
            {"key": "ble:AA:11:22:33:44:55", "kind": "data_ble",
             "name": "右手手套", "slots": [], "sensor_column": "right_glove"},
            {"key": "ble:CC:33:44:55:66:77", "kind": "ble",
             "name": "蓝牙耳机", "slots": []},
        ],
        "device_names": {"cam_a": "桌面摄像头", "cam_b": "侧方摄像头"},
    }
    with open(os.path.join(d, "meta", "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return d


def main():
    app = QApplication(sys.argv)

    print("── 1. 主时钟 fps ──")
    info = {
        "cameras": {"cam_a": {"fps": 30}, "cam_b": {"fps": 25}},
    }
    check(_get_effective_fps(info) == 30.0,
          f"主时钟 = max(30, 25) = 30: {_get_effective_fps(info)}")
    check(_get_effective_fps(
            {"cameras": {"stereo_left": {}, "stereo_right": {}}}) == 30.0,
          "旧数据双目命名回退 settings.STEREO_FPS")
    check(_get_effective_fps({"fps": 24}) == 24.0, "旧数据全局 fps 回退")
    check(_get_effective_fps({}) == 30.0, "空元数据回退 30fps")

    root = tempfile.mkdtemp(prefix="playback_multifps_")
    try:
        session_dir = _make_session(root)

        print("── 2. 加载：主时钟 + 逐路 fps/帧数 ──")
        dlg = PlaybackDialog()
        dlg._load_session(session_dir)
        deadline = time.time() + 30
        while time.time() < deadline and dlg._total_frames <= 0:
            app.processEvents()
            time.sleep(0.02)
        app.processEvents()

        check(dlg._fps == 30.0, f"主时钟 30fps: {dlg._fps}")
        check(dlg._cam_fps == {"cam_a": 30.0, "cam_b": 25.0},
              f"逐路 fps: {dlg._cam_fps}")
        check(dlg._cam_total == {"cam_a": 60, "cam_b": 50},
              f"逐路总帧数: {dlg._cam_total}")
        check(dlg._total_frames == 60,
              f"主时钟总帧数 = max(时长) × 30 = 60: {dlg._total_frames}")
        check(dlg._camera_ids == ["cam_a", "cam_b"],
              f"camera_ids: {dlg._camera_ids}")

        print("── 3. 用户命名叠加 + 统一网格 ──")
        titles = [dlg.grid.camera_widget(sid).name_label.text()
                  for sid in dlg._camera_ids]
        check(any("桌面摄像头 (cam_a)" in t for t in titles),
              f"视频格标题含用户命名: {titles}")
        check(any("侧方摄像头 (cam_b)" in t for t in titles),
              "第二路命名叠加")
        check(dlg.grid.slot_ids() == ["cam_a", "cam_b", "sensor:right_glove"],
              f"网格槽位顺序（摄像机 + 传感器同格）: {dlg.grid.slot_ids()}")
        cell = dlg._sensor_cells[0] if dlg._sensor_cells else None
        check(cell is not None
              and cell.title_label.text() == "右手手套 (right_glove)",
              f"传感器格标题: {cell.title_label.text() if cell else None}")
        check(len(dlg._sensor_widgets) == 1,
              f"传感器面板按 info['sensors'] 动态 1 路: {len(dlg._sensor_widgets)}")
        check(dlg._sensor_modes == ["hand"],
              f"默认模式仿生手掌: {dlg._sensor_modes}")
        check(tr("🎧 其他蓝牙: {} 台", 1) in dlg._info_label.text(),
              f"信息栏蓝牙台数: {dlg._info_label.text()}")

        print("── 4. 逐路独立 seek ──")
        dlg._seek(0)
        check(dlg._last_read_frame == {"cam_a": 0, "cam_b": 0},
              f"seek(0) 逐路首帧: {dlg._last_read_frame}")
        dlg._seek(1)
        check(dlg._last_read_frame == {"cam_a": 1, "cam_b": 1},
              f"seek(1) 顺序读: {dlg._last_read_frame}")
        dlg._seek(2)
        dlg._seek(3)
        check(dlg._last_read_frame == {"cam_a": 3, "cam_b": 3},
              f"seek(3) 25fps 路半帧进位: {dlg._last_read_frame}")
        dlg._seek(4)
        check(dlg._last_read_frame == {"cam_a": 4, "cam_b": 3},
              f"seek(4) 低帧率路重复帧保持: {dlg._last_read_frame}")
        dlg._seek(59)
        check(dlg._last_read_frame == {"cam_a": 59, "cam_b": 49},
              f"seek(59) 末帧按各自总帧数截断: {dlg._last_read_frame}")
        check(dlg._time_label.text() == "00:01 / 00:02",
              f"主时钟时间标签: {dlg._time_label.text()}")
        check(dlg.grid.camera_widget("cam_a").fps_label.text().startswith("#"),
              f"帧号显示在信息条: "
              f"{dlg.grid.camera_widget('cam_a').fps_label.text()}")

        print("── 5. 混合帧率传感器按时间二分 ──")
        tl = dlg._timeline
        calls = []
        orig = tl.nearest_for_column_time
        def spy(col, t):
            calls.append((col, t))
            return orig(col, t)
        tl.nearest_for_column_time = spy
        dlg._seek(30)   # 主时钟 t = 1.0s
        tl.nearest_for_column_time = orig
        check(any(c == "observation.right_glove" and abs(t - 1.0) < 1e-6
                  for c, t in calls),
              f"时间二分查询: {calls}")
        check(dlg._sensor_widgets[0]._has_frame, "传感器画面渲染出帧")
        check(dlg._sensor_ts_labels[0].text() != "",
              f"传感器 TS 标签: {dlg._sensor_ts_labels[0].text()}")
        dlg.close()
        app.processEvents()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("── 6. SensorTimeline.nearest_for_column_time ──")
    fi = np.arange(10, dtype=np.int64)
    ts = np.arange(10, dtype=np.float64) / 10.0
    tl2 = SensorTimeline(fi, ts,
                         {"observation.c": np.random.RandomState(0)
                          .rand(10, 4).astype(np.float32)},
                         np.ones(10, dtype=bool))
    row, dist = tl2.nearest_for_column_time("observation.c", 0.57)
    check(row == 6 and abs(dist - 0.03) < 1e-9,
          f"单调时间戳二分: (row={row}, dist={dist})")
    row, dist = tl2.nearest_for_column_time("observation.c", 5.0)
    check(row == 9 and abs(dist - 4.1) < 1e-9,
          f"超界取最近: (row={row}, dist={dist})")
    row, dist = tl2.nearest_for_column_time("observation.missing", 0.5)
    check(row is None and dist is None, "列不存在返回 (None, None)")

    ts3 = np.array([0.0, 0.1, 0.2, 0.05, 0.15, 0.25])
    tl3 = SensorTimeline(np.arange(6, dtype=np.int64), ts3,
                         {"observation.c": np.random.RandomState(1)
                          .rand(6, 4).astype(np.float32)},
                         np.ones(6, dtype=bool))
    row, dist = tl3.nearest_for_column_time("observation.c", 0.16)
    check(row == 4 and abs(dist - 0.01) < 1e-9,
          f"非单调（暂停回跳）回退 argmin: (row={row}, dist={dist})")

    print("── 7. 幽灵传感器过滤 ──")
    fi7 = np.arange(3, dtype=np.int64)
    ts7 = np.arange(3, dtype=np.float64)
    tl7 = SensorTimeline(
        fi7, ts7,
        {"observation.right_glove": np.random.RandomState(2)
         .rand(3, 256).astype(np.float32),
         "observation.imu": np.random.RandomState(3)
         .rand(3, 6).astype(np.float32)},
        np.ones(3, dtype=bool))
    kept = PlaybackDialog._valid_sensor_names(
        tl7, ["right_glove", "imu", "left_glove"])
    check(kept == ["right_glove"],
          f"只留 16×16 压力矩阵列（过滤 imu/缺失列）: {kept}")
    check(PlaybackDialog._valid_sensor_names(None, ["right_glove"]) == [],
          "timeline 为空返回空列表")

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项未通过")
        return 1
    print("PASS: 回放多帧率测试全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

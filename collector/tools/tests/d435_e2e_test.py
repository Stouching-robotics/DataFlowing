"""D435 端到端录制集成测试（无头）。

模拟主窗口 _add_d435_camera / _on_d435_frames 的完整链路:
  pipeline 注册外部源 + 深度伪相机 → D435Worker 采帧
  → write_external_frame / write_depth → 录制 N 秒 → 校验落盘结构。

用法:
    venv/bin/python tools/tests/d435_e2e_test.py [秒数]
"""
import os
import sys
import json
import glob
import time
import shutil

# 脚本直跑时把项目根加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import pyarrow.parquet as pq
from PyQt5.QtCore import QCoreApplication

from config import settings
from core.pipeline import CameraPipeline
from core.d435_camera import D435Worker, d435_available, list_d400_devices
from core.helpers import (episode_video_files, task_dir_of, pooled_info_path,
                           pooled_data_parquet_path)
from core import depth_reader

OUT_ROOT = "/tmp/d435_e2e_test"
DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0


def main():
    shutil.rmtree(OUT_ROOT, ignore_errors=True)
    os.makedirs(OUT_ROOT, exist_ok=True)

    app = QCoreApplication([])

    if not d435_available():
        print("FAIL: 未检测到 D435 设备")
        return 1

    # 多台 D400 并存时锁定 D435/D435I（baseline ≈0.050 断言以它为准；
    # 只有 D405 时跑 tools/tests/d405_worker_test.py）
    target = next(((n, s) for (n, s) in list_d400_devices() if "435" in n), None)
    if target is None:
        print("SKIP: 未连接 D435/D435I（仅 D405 在位）")
        return 0
    model_name, serial = target
    print(f"目标设备: {model_name} {serial}")

    pipe = CameraPipeline(output_dir=OUT_ROOT)
    pipe.error_occurred.connect(lambda sid, msg: print(f"[pipeline error] {msg}"))

    h, w = settings.D435_RESOLUTION[1], settings.D435_RESOLUTION[0]
    rgb_h, rgb_w = (settings.D435_RGB_RESOLUTION[1],
                    settings.D435_RGB_RESOLUTION[0])
    pipe.register_external_source(settings.D435_SLOT_RGB, (rgb_h, rgb_w),
                                  fps=settings.D435_FPS)
    pipe.set_depth_camera(settings.D435_SLOT_DEPTH, (h, w),
                          fps=settings.D435_FPS,
                          master_slot=settings.D435_SLOT_RGB,
                          heatmap_near_mm=settings.D435_DEPTH_NEAR_MM,
                          heatmap_far_mm=settings.D435_DEPTH_FAR_MM)

    worker = D435Worker(width=w, height=h, fps=settings.D435_FPS,
                        serial=serial, model_name=model_name,
                        ts_log=os.environ.get("D435_TS_LOG"))
    calib_sent = [False]
    stats = {"frames": 0}
    rx_log = open("/tmp/d435_rx_log.txt", "w") if os.environ.get("D435_RX_LOG") else None

    def on_frames(slot_id, frame, hw_ns, imu):
        # 完全复刻 MainWindow._on_d435_frames 的录制链路
        if rx_log:
            rx_log.write(f"{slot_id},{hw_ns}\n")
            rx_log.flush()
        if not calib_sent[0]:
            c = worker.get_calibration()
            if c is not None:
                pipe.set_external_calibration(c)
                calib_sent[0] = True
        if slot_id == settings.D435_SLOT_DEPTH:
            pipe.write_depth(frame)
        else:
            pipe.write_external_frame(slot_id, frame.copy(),
                                      hardware_ns=hw_ns, imu_samples=None)
        stats["frames"] += 1

    worker.frames_ready.connect(on_frames)
    worker.error_occurred.connect(lambda m: print(f"[d435 error] {m}"))
    worker.start()

    # 等首帧 + 标定就绪
    deadline = time.time() + 10
    while not calib_sent[0] and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    if not calib_sent[0]:
        print("FAIL: 标定未就绪")
        worker.stop()
        return 1
    print(f"标定就绪: baseline={worker.get_calibration().baseline:.6f} m, "
          f"depth_scale={worker.get_calibration().depth_scale}")

    pipe.start_recording(settings.D435_SLOT_RGB, task_name="d435_e2e_test")
    t_end = time.time() + DURATION
    while time.time() < t_end:
        app.processEvents()
        time.sleep(0.01)

    worker.stop()
    pipe.finish_recording("")
    # 等待 _finish_async（end_episode + 元数据落盘）
    deadline = time.time() + 15
    while pipe._session_path is not None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)

    ep = task_dir_of(OUT_ROOT, "d435_e2e_test")
    if not os.path.isfile(pooled_info_path(ep)):
        print("FAIL: 未生成任务目录")
        return 1
    print(f"\ntask: {ep}")

    errors = []

    # ── 1. 视频 MP4（仅 RGB；红外不落盘） ─────────────
    files0 = episode_video_files(ep, 1)
    rgb_vid = files0.get(settings.D435_SLOT_RGB)
    if not rgb_vid or not os.path.isfile(rgb_vid):
        errors.append(f"缺少 videos/{settings.D435_SLOT_RGB} MP4")
    else:
        print(f"  videos/{settings.D435_SLOT_RGB}: "
              f"{os.path.basename(rgb_vid)} "
              f"({os.path.getsize(rgb_vid) // 1024} KB)")
    # 红外槽位不应出现在任何落盘结构中
    for slot in ["d435_left", "d435_right"]:
        if os.path.isdir(os.path.join(ep, "videos", "chunk-000", slot)) \
                or slot in files0:
            errors.append(f"红外槽位不应落盘: videos/{slot}")

    # ── 2. 深度通道（12-bit 灰度 MP4：gray12le 对数码；
    #    x265 无能力回落 FFV1 灰 MKV） ────────────────
    dv = files0.get(settings.D435_SLOT_DEPTH)
    pngs = glob.glob(os.path.join(ep, "videos", "chunk-000",
                                  settings.D435_SLOT_DEPTH, "*.png"))
    if not dv or not os.path.isfile(dv):
        errors.append(f"缺少深度视频: {dv}")
    else:
        print(f"  videos/{settings.D435_SLOT_DEPTH}: "
              f"{os.path.basename(dv)} ({os.path.getsize(dv) // 1024} KB)")
    if pngs:
        errors.append(f"不应再有深度 PNG16: {len(pngs)} 个")
    if dv and os.path.isfile(dv):
        frames = list(depth_reader.iter_pooled_depth_frames(
            ep, settings.D435_SLOT_DEPTH, 1))
        if len(frames) < 50:
            errors.append(f"深度帧过少: {len(frames)}")
        elif frames[0].dtype != np.uint16 or frames[0].shape != (h, w):
            errors.append(f"深度帧异常: {frames[0].dtype} {frames[0].shape}")
        if frames:
            valid = frames[-1][frames[-1] > 0]
            print(f"  深度末帧: dtype={frames[-1].dtype} "
                  f"shape={frames[-1].shape} "
                  f"有效值 {len(valid)}px, "
                  f"中位 {np.median(valid):.0f}mm")

    # ── 3. 元数据 ────────────────────────────────────
    with open(pooled_info_path(ep), "r", encoding="utf-8") as f:
        meta = json.load(f)
    cams = meta.get("cameras", {})
    print(f"  info.json cameras: {list(cams.keys())}")
    if settings.D435_SLOT_DEPTH in cams:
        errors.append(f"深度槽不应进 cameras 段: {settings.D435_SLOT_DEPTH}")
    if settings.D435_SLOT_RGB not in cams:
        errors.append("info.json 缺少 RGB 槽位")
    if {"d435_left", "d435_right"} & set(cams.keys()):
        errors.append("info.json 不应含红外槽位")
    vext = (meta.get("video_extensions") or {}).get(settings.D435_SLOT_DEPTH)
    if vext not in ("mp4", "mkv"):
        errors.append(f"video_extensions[{settings.D435_SLOT_DEPTH}] 不符: {vext}")

    # ── 4. 标定 ──────────────────────────────────────
    cal = (meta.get("calibration") or {}).get("head_stereo") or {}
    print(f"  calibration: baseline={cal.get('baseline')}, "
          f"depth_scale={cal.get('depth_scale')}, "
          f"depth_camera={'depth_camera' in cal}")
    if "depth_camera" not in cal:
        errors.append("calibration 缺少 depth_camera 字段")
    if not (0.040 < cal.get("baseline", 0) < 0.060):
        errors.append(f"baseline 异常: {cal.get('baseline')}")

    # ── 5. 时间戳: data parquet 的 hardware_ns 列 ──────
    pq_path = pooled_data_parquet_path(ep, 1)
    if not os.path.isfile(pq_path):
        errors.append("缺少 data/chunk-000/episode-000.parquet")
    else:
        t = pq.read_table(pq_path)
        n_ts = len(t.column("hardware_ns"))
        print(f"  hardware_ns 行数: {n_ts}")

    # ── 6. 无 IMU（D435 不携带 IMU 样本） ────────────
    if os.path.isfile(pq_path):
        t = pq.read_table(pq_path)
        n_imu = sum(len(x.as_py()) for x in t.column("observation.imu"))
        if n_imu:
            errors.append(f"不应存在 IMU 样本: {n_imu} 个")

    print()
    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS: D435 端到端录制结构全部正确")
    return 0


if __name__ == "__main__":
    sys.exit(main())

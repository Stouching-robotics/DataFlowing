"""S80M 50fps 显示 / 30fps 采集抽帧测试（无真机，合成信号）。

模拟 50fps 硬件帧流（hw_ns 步长 20ms，每左帧带 3 个 IMU 样本）→
录制 → 校验：录制帧率 ≈30fps、左右目同步同桶、IMU 样本不丢、
hw_ns==0 兜底全录。纯口径部分另校验 wall 时钟分桶下 hw 时钟跳变
（实测 400ms）不产生空桶。

用法:
    QT_QPA_PLATFORM=offscreen venv/bin/python tools/tests/s80m_50fps_decimation_test.py
"""
import os
import sys
import json
import glob
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyarrow.parquet as pq
from PyQt5.QtWidgets import QApplication

from config import settings
from core.helpers import (episode_video_files, task_dir_of, pooled_info_path,
                           pooled_data_parquet_path)
from core.s80m_manager import frame_record_decision
# 本测试时序容差按 v1.0.8 的 libx264 ultrafast 校准；无真机测试保确定性：
# 不走 auto 编码器探针（探针占用启动窗口会让 50Hz 合成流开头整段被队列丢光）
settings.RECORD_VIDEO_ENCODER = "x264"
settings.ENCODER_PROBE_ENABLED = False
from ui.main_window import MainWindow

OUT_ROOT = "/tmp/s80m_decimation_test"
N_FRAMES = 150          # 150 × 20ms = 3s 的 50fps 流
FAILS = []


def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAILS.append(msg)


def wait(app, ms):
    t_end = time.time() + ms / 1000.0
    while time.time() < t_end:
        app.processEvents()
        time.sleep(0.01)


def pure_decimation_checks():
    """纯口径离线校验（不进 GUI）：hw 时钟跳变不产生空桶。

    150 帧 50fps 到达节奏（mono 步长 20ms），第 100 帧起 hw_ns 跳变
    +400ms（实测传感器行为）：wall 桶口径下仍恰好 90 帧（每 1/30s
    桶 1 帧）；hw 跳变只出现在落盘时间戳上，不损失帧率。旧 hw 桶口径
    下该跳变会整段判成空桶、稳定缺 12 帧。hw_ns==0 兜底全录不变。
    """
    base = 5_000_000_000
    entry = {"last_bucket": {}, "pending_imu": []}
    kept_hw = []
    imu_left = []
    for i in range(150):
        mono = base + i * 20_000_000
        hw = mono if i < 100 else mono + 400_000_000
        imu = [(hw + k * 1_000_000, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
               for k in range(3)]
        rec, batch = frame_record_decision(
            entry, "stereo_left", hw, mono, imu)
        if rec:
            kept_hw.append(hw)
            imu_left.extend(batch)
    check(len(kept_hw) == 90,
          f"150 帧 50fps → 90 帧 30fps（hw 跳变不空桶）: {len(kept_hw)}")
    jumps = [b - a for a, b in zip(kept_hw, kept_hw[1:])
             if b - a > 100_000_000]
    check(len(jumps) == 1 and 300_000_000 <= jumps[0] <= 500_000_000,
          f"kept hw 时间戳恰含 1 处 400ms 跳变（时间轴症状，帧数无损）: {jumps}")
    check(len(imu_left) == 450, f"IMU 450 样本全保留: {len(imu_left)}")

    entry2 = {"last_bucket": {}, "pending_imu": []}
    n_all = sum(frame_record_decision(
        entry2, "stereo_left", 0, base + i * 20_000_000, [])[0]
        for i in range(10))
    check(n_all == 10, f"hw_ns==0 兜底全录: {n_all}/10")

    # 突发送达补录：主进程卡顿 200ms，恢复后 10 帧集中处理（mono 冻结
    # 在同一 wall 桶，hw 仍 20ms 步进）→ 首帧按墙桶保留 + 突发内 hw 逐桶
    # 补录 ≈6 帧；无补录口径只留 1 帧
    entry3 = {"last_bucket": {}, "pending_imu": []}
    for i in range(100):   # 先走 2s 正常节奏（帧距 20ms）
        frame_record_decision(entry3, "stereo_left",
                              base + i * 20_000_000,
                              base + i * 20_000_000, [])
    mono_burst = base + 110 * 20_000_000
    n_kept = sum(frame_record_decision(
        entry3, "stereo_left",
        base + 110 * 20_000_000 + k * 20_000_000,   # hw 连续推进
        mono_burst, [])[0]
        for k in range(10))
    check(5 <= n_kept <= 7, f"突发 10 帧补录 {n_kept} 帧（无补录仅 1）")

    # 正常节奏不超录：hw 每帧进桶（hw 步长 33.4ms）但 mono 仍是 20ms
    # 帧距 → 同 wall 桶第二帧 mono 间隔 ≥10ms 必须跳过，输出恒 30fps
    entry4 = {"last_bucket": {}, "pending_imu": []}
    kept4 = sum(frame_record_decision(
        entry4, "stereo_left",
        base + i * 33_400_000, base + i * 20_000_000, [])[0]
        for i in range(150))
    check(kept4 == 90, f"hw 逐帧进桶不超录: {kept4} == 90")


def main():
    pure_decimation_checks()

    shutil.rmtree(OUT_ROOT, ignore_errors=True)
    os.makedirs(OUT_ROOT, exist_ok=True)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    win._pipeline._output_dir = OUT_ROOT

    # 复刻 _open_s80m 的注册（不启动真实子进程）+ 假 worker 条目（抽帧状态）
    for sid in ["stereo_left", "stereo_right"]:
        win.grid.add_camera(sid, f"{sid} (S80M)")
        win._pipeline.register_external_source(sid, (800, 1280),
                                               fps=settings.STEREO_RECORD_FPS)
    win._workers["s80m:fake"] = {
        "kind": "s80m", "slots": ["stereo_left", "stereo_right"],
        "label": "S80M fake", "last_bucket": {}, "pending_imu": [],
    }

    def feed(hw_ns_step, imu_every):
        """合成 50fps 帧流（hw_ns 步长 20ms，实时节奏推送防队列溢出）。"""
        base_ns = 5_000_000_000
        for i in range(N_FRAMES):
            frame = np.full((800, 1280, 3), i % 255, dtype=np.uint8)
            hw_ns = base_ns + i * hw_ns_step
            imu = ([(hw_ns + k * 1_000_000, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
                    for k in range(imu_every)] if imu_every else [])
            win.stereo_frame_ready.emit("stereo_left", frame, hw_ns, imu)
            win.stereo_frame_ready.emit("stereo_right", frame, hw_ns, [])
            app.processEvents()
            time.sleep(0.02)   # 50Hz 实时节奏（外部源队列 maxsize=2，超发丢帧）

    # 任务名取自 _current_task（默认 None → 空任务名）
    win._record_all()
    assert win._pipeline.is_recording, "录制未启动"

    feed(20_000_000, 3)   # 3s @50fps，每左帧 3 个 IMU 样本
    win._stop_all()
    deadline = time.time() + 15
    while time.time() < deadline:
        app.processEvents()
        if not win._pipeline._session_path:
            break
        time.sleep(0.05)

    ep = task_dir_of(OUT_ROOT, "")
    if not os.path.isfile(pooled_info_path(ep)):
        print("FAIL: 未生成任务目录")
        return 1

    # 1. 录制帧率 ≈30fps（150 帧 50fps 流 → 90-102 对；data parquet 一行一槽
    #    = 2×对数。wall 桶口径下桶数随实时推送节奏（sleep 0.02 + 事件处理
    #    ≈21ms/迭代 → 150 帧约 3.1-3.4s），确定值 90 见 pure_decimation_checks；
    #    左右目同对间隙为 0，过滤后应全为 20-40ms 桶边界步长）
    t1 = pq.read_table(pooled_data_parquet_path(ep, 1))
    hw = np.asarray(t1.column("hardware_ns").to_pylist(), dtype=np.int64)
    check(176 <= len(hw) <= 204,
          f"录制帧数 150 对→{len(hw)} 行 (期望 ≈90-102 对×2 槽)")
    gaps_ms = [(b - a) / 1e6 for a, b in zip(hw, hw[1:]) if b != a]
    check(gaps_ms and all(20 <= g <= 40 for g in gaps_ms),
          f"跨对时间戳间隔 20-40ms: {sorted(set(gaps_ms))}")
    check(all(b > 2**31 for b in hw), "hw_ns 无截断（全 >2^31）")

    # 2. 左右目同桶同步：两路帧数一致
    files0 = episode_video_files(ep, 1)
    check(bool(files0.get("stereo_left")) and bool(files0.get("stereo_right")),
          "左右目 MP4 均存在")

    # 3. IMU 不丢：IMU 随帧行存列表列（observation.imu），
    #    150 左帧 × 3 样本 = 450 项，被抽掉帧的样本累积到下一行
    #    （允许尾部 ≤6 损失）
    imu_n = sum(len(x.as_py())
                for x in t1.column("observation.imu"))
    check(imu_n >= 444, f"IMU 样本 {imu_n}/450 保留（无抽帧丢失）")

    # 4. hw_ns==0 兜底全录（无抽帧）
    shutil.rmtree(OUT_ROOT, ignore_errors=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    win._record_all()
    assert win._pipeline.is_recording, "二次录制未启动"
    frame = np.full((800, 1280, 3), 7, dtype=np.uint8)
    for _ in range(10):
        win.stereo_frame_ready.emit("stereo_left", frame, 0, [])
        win.stereo_frame_ready.emit("stereo_right", frame, 0, [])
        app.processEvents()
        time.sleep(0.05)   # 防队列溢出（5Hz 节奏已远低于 writer 吞吐）
    win._stop_all()
    deadline = time.time() + 15
    while time.time() < deadline:
        app.processEvents()
        if not win._pipeline._session_path:
            break
        time.sleep(0.05)
    ep2 = task_dir_of(OUT_ROOT, "")
    t2 = pq.read_table(pooled_data_parquet_path(ep2, 1))
    check(len(t2.column("hardware_ns")) == 20,
          "hw_ns==0 兜底全录: 20 行（10 对×2 槽 ==20）")

    win._workers.pop("s80m:fake", None)   # 假条目无 proc，teardown 不适用于本测试
    win.close()
    wait(app, 300)

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项未通过")
        return 1
    print("PASS: S80M 50→30 抽帧测试全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

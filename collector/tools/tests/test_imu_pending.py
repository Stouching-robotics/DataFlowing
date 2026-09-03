"""IMU 防丢缓冲测试（v1.0.9）—— 合成帧 + 假 writer，无真机无真 ffmpeg。

覆盖：外部队列满 → 帧丢弃计数 + IMU 批次入防丢缓冲；缓冲超限丢最旧+
每 episode 只告警一次；写盘线程按时间戳挂靠（被丢帧的 IMU 晚于队内
旧帧 → 只挂靠 ts ≤ 本帧 hw_ns 的样本，跨行时序单调）；finish_recording
快照注入 writer 元数据回写。

用法:
    QT_QPA_PLATFORM=offscreen venv/bin/python tools/tests/test_imu_pending.py
"""
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from config import settings
from core.pipeline import CameraPipeline

OUT_ROOT = "/tmp/imu_pending_test"
FAILS = []

# 合成时钟：帧 i 的 hw_ns = BASE + i×33.3ms，IMU 样本窗口结束于帧 hw_ns
BASE = 5_000_000_000
STEP = 33_333_333


def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAILS.append(msg)


def imu_of(i, n, tag=1.0):
    """帧 i 的 n 个 IMU 样本（ts 单调递增且全部 ≤ 帧 hw_ns）。"""
    hw = BASE + i * STEP
    return [(hw - (n - k) * 1_000_000, tag, 0.2, 0.3, 0.4, 0.5, 0.6)
            for k in range(n)]


class FakeWriter:
    """假 writer：记录 write_frame_row 收到的 IMU 与 set_drop_stats。"""

    encoder_label = "TEST"

    def __init__(self):
        self.rows = []
        self.drop_stats = None
        self.ended = False

    def write_video_frame(self, *a, **k):
        pass

    def write_frame_row(self, *a, **k):
        self.rows.append(k.get("imu_samples") or [])

    def set_drop_stats(self, stats):
        self.drop_stats = dict(stats)

    def end_episode(self):
        self.ended = True


def main():
    app = QApplication(sys.argv)
    pip = CameraPipeline(OUT_ROOT)
    pip.register_external_source("stereo_left", (800, 1280), fps=30)
    pip._recording = True   # 绕过 start_recording 的异步启动路径

    logs = []
    pip.recording_log.connect(logs.append)

    frame = np.zeros((800, 1280, 3), dtype=np.uint8)

    # 1. 队列满（maxsize=2）→ 丢帧计数 + IMU 入缓冲
    pip.write_external_frame("stereo_left", frame, BASE + 0 * STEP, imu_of(0, 3))
    pip.write_external_frame("stereo_left", frame, BASE + 1 * STEP, imu_of(1, 3))
    pip.write_external_frame("stereo_left", frame, BASE + 2 * STEP, imu_of(2, 3))  # Full
    check(pip._drop_stats.snapshot() == {"ext:stereo_left": 1},
          "队列满丢帧计数 ext:stereo_left=1")
    check(len(pip._pending_imu.get("stereo_left", [])) == 3,
          "被丢帧的 IMU 批次进入防丢缓冲")

    # 2. 缓冲超限：丢最旧 + 每 episode 只告警一次
    old_max = settings.IMU_PENDING_MAX_SAMPLES
    settings.IMU_PENDING_MAX_SAMPLES = 4
    pip.write_external_frame("stereo_left", frame, BASE + 3 * STEP, imu_of(3, 5))  # Full
    pip.write_external_frame("stereo_left", frame, BASE + 4 * STEP, imu_of(4, 5))  # Full
    settings.IMU_PENDING_MAX_SAMPLES = old_max
    check(len(pip._pending_imu.get("stereo_left", [])) == 4,
          "超限丢最旧：缓冲维持上限 4")
    check(pip._imu_overflow_count == 2, "IMU 溢出计数 = 2")
    check(sum(1 for l in logs if "IMU 防丢缓冲超限" in l) == 1,
          "溢出告警每 episode 只发一次")
    check(pip._drop_stats.snapshot() == {"ext:stereo_left": 3},
          "累计丢帧计数 ext:stereo_left=3")

    # 3. 写盘线程：队内旧帧 A/B 不挂靠晚样本；新帧 F 挂靠全部 pending
    writer = FakeWriter()
    pip._writer = writer
    pip._episode_start_s = time.time()
    pip._per_cam_frame = {"stereo_left": 0}
    t = threading.Thread(target=pip._write_loop, daemon=True)
    t.start()
    time.sleep(0.12)   # A、B 各被一个 tick 消费（pending 晚于两者 → 不挂靠）
    pip.write_external_frame("stereo_left", frame, BASE + 5 * STEP,
                             imu_of(5, 3))   # F：hw 晚于全部 pending
    time.sleep(0.15)
    pip._recording = False
    t.join(timeout=3)
    check(len(writer.rows) == 3, f"消费 3 帧（A/B/F，实际 {len(writer.rows)}）")
    check(len(writer.rows) == 3 and len(writer.rows[0]) == 3,
          "A 行 IMU = 本帧 3（pending 晚于 A，不挂靠）")
    check(len(writer.rows) == 3 and len(writer.rows[1]) == 3,
          "B 行 IMU = 本帧 3（pending 晚于 B，不挂靠）")
    check(len(writer.rows) == 3 and len(writer.rows[2]) == 7,
          "F 行 IMU = 缓冲 4 + 本帧 3 = 7（ts ≤ F hw 全量挂靠）")
    check(not pip._pending_imu, "挂靠后防丢缓冲清空")
    if len(writer.rows) == 3 and len(writer.rows[2]) == 7:
        tss = [s[0] for s in writer.rows[2]]
        check(tss == sorted(tss), f"F 行 IMU 时间戳时序单调: {tss}")

    # 4. finish_recording：快照含 imu_overflow 并注入 writer
    pip._write_thread = None
    pip._recording = True
    pip._session_path = os.path.join(OUT_ROOT, "fake_session")
    finished = []
    pip.recording_finished.connect(lambda sid, p: finished.append((sid, p)))
    pip.finish_recording("stereo_left")
    deadline = time.time() + 5
    while time.time() < deadline and not finished:
        app.processEvents()
        time.sleep(0.02)
    check(bool(finished), "recording_finished 触发（_finish_async 完成）")
    check(writer.ended, "writer.end_episode 被调用")
    check(pip.last_drop_stats == {"ext:stereo_left": 3, "imu_overflow": 2},
          f"last_drop_stats 快照: {pip.last_drop_stats}")
    check(writer.drop_stats == {"ext:stereo_left": 3, "imu_overflow": 2},
          "丢帧统计注入 writer（元数据回写路径）")

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项未通过")
        return 1
    print("PASS: imu_pending 单测全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

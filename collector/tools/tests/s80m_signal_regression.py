"""S80M 回归测试（无设备,合成帧）—— 验证 stereo_frame_ready 信号修复
(qint32 截断)与 S80C 深度链路（depth_ready 信号 + 深度伪相机 + 12-bit
灰度 MP4（gray12le 对数码；x265 无能力回落 FFV1 灰 MKV）+ 补拍时长
对齐），不受 D435 改动影响。v1.1.0 池化布局校验。

模拟真实链路:后台线程 emit stereo_frame_ready（1280×800 BGR +
hw_ns > 2^31 + IMU 样本，30Hz），隔帧 emit _s80m_manager.depth_ready
（uint16 毫米合成帧，~15Hz，低于录制 30fps → 触发管线补拍）。
录制 5s 后校验落盘。

用法:
    QT_QPA_PLATFORM=offscreen venv/bin/python tools/tests/s80m_signal_regression.py
"""
import os
import sys
import json
import glob
import time
import shutil
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import cv2
import pyarrow.parquet as pq
from PyQt5.QtWidgets import QApplication

from config import settings
from ui.main_window import MainWindow
from core import depth_reader
from core.helpers import episode_video_files

OUT_ROOT = "/tmp/s80m_reg_test"
DURATION = 5.0


def _count_mp4_frames(mp4: str):
    """ffprobe 数视频流0帧数；无可用 ffprobe/失败 → None（断言优雅跳过）。

    逐个候选尝试（conda base 的 ffprobe 有 openvino/tbb 符号损坏，
    失败后落到 lerobot env 的已知可用版）。
    """
    import subprocess
    import shutil
    candidates = [shutil.which("ffprobe"),
                  "/home/stouch/miniconda3/envs/lerobot/bin/ffprobe"]
    for ffprobe in candidates:
        if not ffprobe:
            continue
        try:
            out = subprocess.run(
                [ffprobe, "-v", "error", "-count_frames",
                 "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_frames",
                 "-of", "default=nokey=1:noprint_wrappers=1", mp4],
                capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.returncode != 0:
            continue
        try:
            return int(out.stdout.strip())
        except ValueError:
            return None
    return None


def main():
    shutil.rmtree(OUT_ROOT, ignore_errors=True)
    os.makedirs(OUT_ROOT, exist_ok=True)

    app = QApplication(sys.argv)
    # DEPTH_ENABLED 常量保持 False（遗留开关）；S80C 深度走显式注册的
    # 深度伪相机（set_depth_camera），与 settings.S80M_DEPTH_* 口径一致
    assert settings.DEPTH_ENABLED is False
    win = MainWindow()
    win.show()
    win._pipeline._output_dir = OUT_ROOT

    # 复刻 _open_s80m 的注册步骤(不启动真实子进程)：左右目外部源 +
    # 深度伪相机 stereo_depth（D435 同款管线）
    for sid in ["stereo_left", "stereo_right"]:
        win.grid.add_camera(sid, f"{sid} (S80M)")
        win._pipeline.register_external_source(sid, (800, 1280),
                                               fps=settings.STEREO_RECORD_FPS)
    dslot = settings.S80M_DEPTH_SLOT
    win.grid.add_camera(dslot, "Depth (S80M)")
    win._pipeline.set_depth_camera(
        dslot, (800, 1280), fps=settings.STEREO_RECORD_FPS,
        master_slot="stereo_left",
        heatmap_near_mm=settings.S80M_DEPTH_NEAR_MM,
        heatmap_far_mm=settings.S80M_DEPTH_FAR_MM,
        heatmap_smooth_k=settings.S80M_DEPTH_SMOOTH_K)

    stop = threading.Event()

    def feeder():
        """模拟 read_pipe:30Hz 帧 + 大于 2^31 的硬件纳秒 + IMU；
        隔帧 emit depth_ready（~15Hz 深度，低于录制帧率 → 补拍）。"""
        i = 0
        base_ns = 5_000_000_000  # 5s,远超 qint32 上限 2^31(≈2.147s)
        while not stop.is_set():
            frame = np.full((1280, 800, 3), i % 255, dtype=np.uint8)
            hw_ns = base_ns + i * 40_000_000  # 40ms 步长
            imu = [(hw_ns + k * 1000_000, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
                   for k in range(3)]
            win.stereo_frame_ready.emit("stereo_left", frame, hw_ns, imu)
            win.stereo_frame_ready.emit("stereo_right", frame, hw_ns, [])
            if i % 2 == 0:
                # 子进程深度引擎输出的 uint16 毫米合成帧（真实信号链路）
                depth = np.full((800, 1280), 1000 + i % 500,
                                dtype=np.uint16)
                win._s80m_manager.depth_ready.emit(dslot, depth, hw_ns)
            i += 1
            time.sleep(1 / 30.0)

    t = threading.Thread(target=feeder, daemon=True)
    t.start()

    def wait(ms):
        t_end = time.time() + ms / 1000.0
        while time.time() < t_end:
            app.processEvents()
            time.sleep(0.01)

    wait(1000)  # 等首帧进入显示

    # 任务名取自 _current_task（默认 None → 空任务名）
    win._record_all()
    assert win._pipeline.is_recording, "录制未启动"
    wait(DURATION * 1000)
    win._stop_all()

    deadline = time.time() + 15
    while time.time() < deadline:
        app.processEvents()
        if not win._pipeline._session_path:
            break
        time.sleep(0.05)
    stop.set()

    task_dir = os.path.join(OUT_ROOT, "session")
    if not os.path.isfile(os.path.join(task_dir, "meta", "info.json")):
        print("FAIL: 未生成任务目录")
        win.close()
        return 1
    ep = task_dir
    print(f"task: {ep}")

    errors = []

    # 1. 时间戳: 池化 data parquet 的 hardware_ns 应 > 2^31 且无截断
    #    (单调、全正、首 ≈5e9)
    data_pq = glob.glob(os.path.join(ep, "data", "chunk-000", "*.parquet"))
    t = pq.read_table(data_pq[0])
    hw = np.asarray(t.column("hardware_ns").to_pylist(), dtype=np.int64)
    mono = bool(np.all(np.diff(hw) >= 0))
    big = int((hw > 2**31).sum())
    neg = int((hw < 0).sum())
    print(f"hardware_ns: {len(hw)} 行, 首={hw[0]/1e9:.2f}s, 超2^31={big}, "
          f"负值={neg}, 单调={mono}")
    if not mono or neg or big < len(hw) // 2:
        errors.append("hardware_ns 截断或非单调(信号修复失效)")

    # 2. 深度: 伪相机 stereo_depth → 12-bit 灰度 MP4（gray12le 对数码；
    #    x265 无能力回落 FFV1 灰 MKV）。文件在池化 videos/chunk-000/
    depth_vid = episode_video_files(ep, 1).get(dslot)
    print(f"depth 视频: {depth_vid}")
    if not depth_vid or not os.path.isfile(depth_vid):
        errors.append(f"{dslot} 深度视频缺失")
        depth_vid = None
    if glob.glob(os.path.join(ep, "videos", "chunk-000", dslot, "*.png")):
        errors.append(f"{dslot} 不应再有 PNG16")

    # 3. 补拍时长对齐: 深度源 ~15Hz 补拍至录制帧率 → 深度视频帧数须与
    #    主槽位 stereo_left 视频帧数一致（±3，首帧进入前的空窗）；
    #    无 ffprobe 环境优雅跳过
    if depth_vid:
        nframes = _count_mp4_frames(depth_vid)
        left_mp4 = episode_video_files(ep, 1).get("stereo_left")
        lframes = _count_mp4_frames(left_mp4) if left_mp4 else None
        if nframes is not None and lframes is not None:
            print(f"depth 帧数: {nframes} (stereo_left 视频 {lframes} 帧)")
            if abs(nframes - lframes) > 3:
                errors.append(f"深度补拍失效: {nframes} 帧 "
                              f"≠ stereo_left {lframes} 帧")

    # 4. metadata: video_extensions[dslot] 应为 mp4（12-bit 灰）或 mkv
    #    （FFV1 回落）；深度槽不进 cameras 段（深度为独立数据轨）
    meta = json.load(open(os.path.join(ep, "meta", "info.json")))
    vext = (meta.get("video_extensions") or {}).get(dslot)
    print(f"metadata video_extensions[{dslot}]: {vext}")
    if vext not in ("mp4", "mkv"):
        errors.append(f"{dslot} video_extensions 不符: {vext}")

    # 5. 深度内容: 合成帧值 1000+i%500 经 12-bit 量化往返应还原
    #    （±3mm 容差；feeder 隔帧发射 → 去重后按 +2 步进、1498→1000
    #    回绕；深度队列 maxsize=3 允许偶发丢帧 → 偶数大步跳变）
    if depth_vid:
        frames = list(depth_reader.iter_pooled_depth_frames(ep, dslot, 1))
        if not frames:
            errors.append("深度解码为空")
        else:
            arr = frames[0]
            if arr.shape != (800, 1280) or arr.dtype != np.uint16:
                errors.append(f"深度首帧异常: {arr.dtype} {arr.shape}")
            dedup = []
            for f in frames:
                v = int(f[0, 0])
                if not dedup or abs(dedup[-1] - v) > 3:
                    dedup.append(v)
            ok_seq = True
            skips = 0
            for a, b in zip(dedup, dedup[1:]):
                if 1 <= b - a <= 7:
                    continue   # 正常步进 +2（量化噪声/相邻值合并 → +4/+6）
                if a >= 1490 and 1000 <= b <= 1015:
                    continue                  # 500 周期回绕（1498→1000）
                if 2 < b - a <= 40:
                    skips += 1                # 队列合法丢帧
                    continue
                ok_seq = False
                break
            if skips > 5:
                ok_seq = False
            if not ok_seq:
                errors.append(f"深度值序列异常: {dedup[:10]}")
            print(f"深度帧数: {len(frames)}, 去重后首值: {dedup[:4]}, "
                  f"丢帧跳变: {skips}")

    # 6. IMU: data parquet 的 observation.imu 列非空(stereo_left 携带样本)
    n_imu = sum(len(x.as_py()) for x in t.column("observation.imu"))
    print(f"observation.imu 样本: {n_imu} 个")
    if n_imu == 0:
        errors.append("IMU 样本缺失")

    # 7. 视频
    files0 = episode_video_files(ep, 1)
    for sid in ["stereo_left", "stereo_right"]:
        if not files0.get(sid) or not os.path.isfile(files0[sid]):
            errors.append(f"{sid} MP4 缺失")

    # 8. 标定: S80M 无外部标定时 info.json calibration.head_stereo 应为
    #    默认 StereoCalibration（无 depth_camera 字段）
    cal = (meta.get("calibration") or {}).get("head_stereo") or {}
    print(f"calibration.head_stereo: {sorted(cal.keys())[:6]}")
    if "depth_camera" in cal:
        errors.append("S80M 标定不应含 depth_camera 字段")

    win.close()
    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS: S80M 信号回归通过(hw_ns 无损、深度 12-bit 灰视频补拍/IMU 链路正常)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

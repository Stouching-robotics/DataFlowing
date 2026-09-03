"""
S80M 双目设备管理器 —— 子进程生命周期、stdout 帧管道读取、曝光协议与
50→30 抽帧口径（QtCore QObject + 信号，不依赖 QtWidgets）。

自包含双目模块：tools/stereo_s80m/ 内含 read_stereo_rgb.py +
libfays_vikit.so (3.9.0) + fays_vikit.yaml，与外部 SDK 目录完全独立——
外部目录 git 更新、跑官方 demo 均不影响 DAQ 程序。

  - S80MDeviceManager  每台一个条目：Popen 子进程 / stdin 曝光通道 /
                       watchdog / stdout 帧读取线程 / 临时 50fps yaml
  - frame_record_decision  录制 50→30 抽帧纯口径（wall 时钟每 1/30s 桶保留首帧，突发补录）
  - s80m_drop_watch  录制期空桶统计纯口径（wall 时钟，超阈值由 UI 告警）
  - load_default_exposure  「恢复默认」基线（yaml stereo_init_exposure）

帧信号 frame_ready(str, ndarray, object, list) 与深度信号
depth_ready(str, ndarray, object) 的 hw_ns 必须用 object 封装：PyQt5
队列信号把 Python int 按 C++ qint32 封送，超过 2^31 (≈2.1s 纳秒)
即静默截断为负数，录制数据受害。
"""

from __future__ import annotations

import os
import sys
import struct
import shutil
import tempfile
import threading
import subprocess

import numpy as np
import cv2

from PyQt5.QtCore import QObject, pyqtSignal, QTimer

from config import settings
from config.i18n import tr

# ★ 自包含双目模块：tools/stereo_s80m/（路径锚定仓库根，不随 cwd 漂移）
SDK_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tools", "stereo_s80m")
STEREO_DEMO = os.path.join(SDK_ROOT, "read_stereo_rgb.py")
STEREO_AVAILABLE = os.path.isfile(STEREO_DEMO)
STEREO_CFG_25 = os.path.join(SDK_ROOT, "config", "fays_vikit.yaml")
STEREO_CFG_50 = os.path.join(SDK_ROOT, "config", "fays_vikit_50fps.yaml")
# 相机档（settings.STEREO_CAM_FPS）：50=默认，50→30 桶抽帧才有真 30fps
# 录制（回调取帧已根治 SDK 交付撕裂，官方 GUI 同款组合）；25=回退档
#（25fps 相机下 30 桶只填到 25 帧，录制内容等效 25fps）
STEREO_CFG = STEREO_CFG_50 if settings.STEREO_CAM_FPS >= 50 else STEREO_CFG_25

# S80C 深度：SDK 深度引擎自包含包（third_party/ 内 vikit 3.9.0 +
# 深度库 + OpenCV 4.2 + 深度配置/model，与 hand_3d_s80c 共用）
DEPTH_SDK_ROOT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools", "hand_3d_s80c", "third_party")

# 回调取帧桥接（官方 GUI 同款消费方案；shim 在 S80C 自包含包内）。
# 轮询 GetStereoFrames + 深度引擎绑定叠加会在交付帧内留下水平错位带
#（8/31 主程序录制回归：25fps 档降缝率但不根治）——回调在 SDK 装配
# 完成后才触发、帧必完整。settings.STEREO_CB_BRIDGE 关掉即回轮询；
# shim 缺失/注册失败子进程也自动回退（协议不变）。
CB_BRIDGE_LIB = os.path.join(DEPTH_SDK_ROOT, "cb_bridge",
                             "libfays_cb_bridge.so")


def s80m_depth_available() -> bool:
    """S80C 深度可用性：开关开 + 深度包关键文件齐备。

    关闭（settings.S80M_DEPTH_ENABLED=False）→ spawn 不传 --depth-sdk-dir，
    子进程协议无深度块、零深度 CPU 开销（与旧版逐字节一致）。
    """
    if not settings.S80M_DEPTH_ENABLED:
        return False
    return all(os.path.isfile(p) for p in (
        os.path.join(DEPTH_SDK_ROOT, "lib", "libfays_vikit.so"),
        os.path.join(DEPTH_SDK_ROOT, "lib", "libfayssense_aikit_depth.so"),
        os.path.join(DEPTH_SDK_ROOT, "config", "stereo_depth.yaml"),
        os.path.join(DEPTH_SDK_ROOT, "config", "models",
                     "rk3588", "stereo_s_general.rknn"),
    )) and os.path.isdir(os.path.join(DEPTH_SDK_ROOT, "opencv4.2", "lib406"))


def load_default_exposure() -> tuple:
    """「恢复默认」基线：SDK 无曝光读回接口，出厂值即 yaml
    stereo_init_exposure（-1 = 自动曝光，1.0~885.0 = 手动）。"""
    try:
        import yaml as _yaml
        with open(STEREO_CFG, encoding="utf-8") as _f:
            _si = float(_yaml.safe_load(_f).get(
                "stereo_init_exposure", 400.0))
        return (_si < 0, 400.0 if _si < 0 else _si)
    except Exception:
        return (False, 400.0)


def frame_record_decision(entry: dict | None, slot_id: str,
                          hardware_ns: int, mono_ns: int,
                          imu_samples: list) -> tuple:
    """相机帧 / 30fps 采集抽帧：wall 时钟每 1/30s 桶保留首帧，突发补录。

    正常节奏（帧距 20ms）按 wall 桶取每桶首帧：桶号用调用方到达时刻
    mono_ns（time.monotonic_ns，与视频 PTS 同源）而非 hardware_ns——
    传感器 hw 时钟会跳变（实测 400ms），按 hw 桶会把跳变区间整段判成
    空桶 → 30fps 抽帧每次稳定缺帧；wall 口径下 hw 跳变不影响抽帧率。

    突发补录：主进程卡顿（编码器子进程启动/GIL 争抢）期间帧堆积在
    队列里，恢复后集中处理时 mono 间隔 <10ms 且落在同一 wall 桶——
    这些帧的真实时间差在 hw 上仍逐桶推进，按 hw 桶号 ≥1 补录；正常
    20ms 节奏的同桶第二帧 mono 间隔 ≥10ms，不受影响，输出恒 30fps。
    补录帧由写线程 30fps 均匀消费（配合深外部队列），视频时间轴延后
    但不出空洞。

    左右目同 ts 天然同步；hw_ns==0 兜底全录（无时间戳可依）。被抽掉帧
    携带的 IMU 块累积到下一帧（stereo_left 主槽），不丢样本。entry 至少
    带 last_bucket / pending_imu 键（由 S80MDeviceManager.new_entry 初始化，
    离线测试假条目同款形状）。返回 (record, imu_batch)。
    """
    record = True
    imu_batch = imu_samples
    if entry is not None and hardware_ns > 0:
        interval_ns = int(settings.STEREO_RECORD_MIN_INTERVAL_S * 1e9)
        wall_b = mono_ns // interval_ns
        hw_b = hardware_ns // interval_ns
        prev = entry["last_bucket"].get(slot_id)
        if prev is not None:
            if wall_b == prev["wall"]:
                # 同 wall 桶：突发补录（mono 间隔 <10ms = 队列积压集中
                # 处理，帧真实间隔看 hw 桶推进）；正常节奏同桶第二帧
                # （mono 间隔 ≈20ms）跳过
                record = (mono_ns - prev["mono"] < _BURST_MONO_GAP_NS
                          and hw_b >= prev["hw"] + 1)
            else:
                record = True
        if record:
            entry["last_bucket"][slot_id] = {
                "wall": wall_b, "hw": hw_b, "mono": mono_ns}
        if slot_id == "stereo_left":
            if record:
                if entry["pending_imu"]:
                    imu_batch = entry["pending_imu"] + (imu_samples or [])
                    entry["pending_imu"] = []
            else:
                entry["pending_imu"].extend(imu_samples or [])
    return record, imu_batch


# 突发补录判定：同 wall 桶内 mono 间隔低于此值视为队列积压集中处理
# （正常 50fps 节奏帧距 20ms；Qt 队列事件连续处理间隔亚毫秒级）
_BURST_MONO_GAP_NS = 10_000_000

# 空桶告警阈值：录制期空桶率超过此值弹一次告警（8/28 健康录制 ~2%；
# 9/2 三槽齐录实测 ~15%，视频 PTS 与数据时间轴错位 ~1.2s）
STEREO_DROP_ALERT_RATE = 0.10


def s80m_drop_watch(entry: dict | None, mono_ns: int,
                    interval_ns: int) -> tuple:
    """录制期空桶统计（stereo_left 主槽口径，纯函数可离线测）。

    用 wall 时钟（time.monotonic_ns，与 data 帧时间戳同源）而非
    hardware_ns：传感器时钟跳变（实测 400ms）按 hw 桶统计会误计
    为丢桶，wall 口径与"视频 PTS 均匀 30fps、数据时间轴被拉长"
    的错位症状一致。每帧推进：跨过的桶数 elapsed=⌊gap/interval⌋，
    跨 k≥2 桶即计 k-1 个空桶；间隔 20ms 的同桶弃帧自然不误计。
    hw_ns==0 兜底全录时帧距 20ms 跨 0 桶，统计恒 0 不告警。

    entry 需带 drop_watch 键（new_entry 初始化、录制起止由
    device_manager.reset_s80m_record_state 重置）。返回
    (本帧新增空桶, 累计空桶, 累计桶数)；entry 为 None 返回 (0, 0, 0)。
    """
    if entry is None:
        return (0, 0, 0)
    w = entry["drop_watch"]
    last = w["last_mono_ns"]
    new_drops = 0
    if last is not None and mono_ns > last:
        elapsed = (mono_ns - last) // interval_ns
        if elapsed >= 2:
            new_drops = elapsed - 1
            w["dropped"] += new_drops
        w["elapsed"] += elapsed
    w["last_mono_ns"] = mono_ns
    return (new_drops, w["dropped"], w["elapsed"])


class S80MDeviceManager(QObject):
    """S80M 双目子进程生命周期管理（每台一个条目）。

    信号
    ----
    frame_ready(str, np.ndarray, object, list)
        — (slot_id, frame, hw_ns, imu_samples)，reader 线程 → 主线程
    depth_ready(str, np.ndarray, object)
        — (depth_slot, depth_uint16_mm, hw_ns)，子进程 SDK 深度引擎输出
          （--depth-sdk-dir 模式；hw_ns 同帧 SDK 硬件纳秒时钟）
    device_closed(str)
        — dev_key，进程/文件/线程清理完毕；UI 侧注销管线源、拆槽、
          清标定由调用方槽执行（顺序与原 _close_s80m 一致）
    log(str)
        — 跨线程日志

    reader 线程与 watchdog 不触碰 UI；主线程创建、QObject 亲和性在主线程。
    """

    frame_ready = pyqtSignal(str, np.ndarray, object, list)
    depth_ready = pyqtSignal(str, np.ndarray, object)
    device_closed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, pipeline, parent=None):
        super().__init__(parent)
        self._pipeline = pipeline
        self._entries: dict = {}   # dev_key → 注册表条目（与主窗口共享同一 dict）
        self.shutting_down = False

    @staticmethod
    def new_entry(label: str) -> dict:
        """注册表条目骨架（kind/slots/label 由主窗口注册，进程字段由 spawn 填充）。"""
        return {
            "kind": "s80m",
            # 深度可用时设备 meta 多记录一路深度槽（spawn 后真实生效）
            "slots": ["stereo_left", "stereo_right"]
            + (["stereo_depth"] if s80m_depth_available() else []),
            "label": label,
            "proc": None,
            "stdin": None,          # 曝光命令通道（close 时关闭）
            "stderr_file": None,
            "config_file": None,
            "original_exp": None,   # 恢复默认基线（yaml 出厂值）
            "watchdog": None,
            "reader_thread": None,
            "depth_active": False,  # spawn 后填充：子进程协议带深度块
            "last_bucket": {},      # 50→30 抽帧：每槽最近保留帧 {wall,hw,mono}（突发补录口径）
            "pending_imu": [],      # 被抽掉帧的 IMU 块缓冲（随下一帧落盘）
            "drop_watch": {         # 录制期空桶统计（wall 时钟，UI 告警用）
                "last_mono_ns": None,
                "dropped": 0,       # 累计空桶数
                "elapsed": 0,       # 累计跨过的桶数（= 已填桶 + 空桶）
                "alerted": False,   # 本段已弹过一次告警
            },
        }

    def spawn(self, dev_key: str, entry: dict) -> bool:
        """启动子进程 + watchdog + reader 线程，填充 entry 进程字段。

        用户已在 video 组，无需 sudo（sudo 在 no-new-privileges 环境下会
        直接失败）。stdout 管道传帧数据（--pipe -），避免 FIFO 文件权限
        问题（AppArmor）。
        """
        if not STEREO_AVAILABLE:
            return False

        # ★ 相机档 / 30fps 采集：SDK 跑 STEREO_CAM_FPS 档 yaml 副本（tempfile
        #   拷贝，仓库内原文件不动，关闭时清理），录制按 wall 时钟 1/30s 桶抽帧
        cfg_tmp = None
        cfg_args = []
        if os.path.isfile(STEREO_CFG):
            cfg_tmp = tempfile.mktemp(suffix="_s80m_cam.yaml")
            shutil.copyfile(STEREO_CFG, cfg_tmp)
            cfg_args = ["--config", cfg_tmp]
        entry["original_exp"] = load_default_exposure()

        stereo_stderr = tempfile.mktemp(suffix="_s80m_stderr.log")
        # 深度模式：cmd 追加 --depth-sdk-dir + env 注入 lib406
        # （深度库依赖链用裸 SONAME 必须靠动态链接器搜索路径解析；
        # venv cv2 走自身 rpath 绑自家 OpenCV，不受影响）
        depth_args = []
        spawn_env = None
        if s80m_depth_available():
            depth_args = ["--depth-sdk-dir", DEPTH_SDK_ROOT]
            _lib406 = os.path.join(DEPTH_SDK_ROOT, "opencv4.2", "lib406")
            spawn_env = dict(os.environ)
            spawn_env["LD_LIBRARY_PATH"] = _lib406 + ":" + \
                spawn_env.get("LD_LIBRARY_PATH", "")
        entry["depth_active"] = bool(depth_args)
        # 回调取帧（官方 GUI 同款；开关/shim 齐备才传，子进程另有回退）
        cb_args = (["--cb-bridge"]
                   if settings.STEREO_CB_BRIDGE and os.path.isfile(CB_BRIDGE_LIB)
                   else [])
        # stdin 管道 = 曝光控制通道（行协议 "SET_EXPOSURE <float>"，
        # SDK 运行时生效无需重启；子进程句柄创建完成前命令先积在管道缓冲）
        proc = subprocess.Popen(
            [sys.executable, STEREO_DEMO, "--pipe", "-"] + cfg_args
            + depth_args + cb_args,
            cwd=SDK_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open(stereo_stderr, "w"),
            env=spawn_env,
        )
        entry.update(proc=proc, stdin=proc.stdin,
                     stderr_file=stereo_stderr, config_file=cfg_tmp)
        self._entries[dev_key] = entry

        # 监控子进程退出（定时检查）
        watchdog = QTimer(self)
        watchdog.setInterval(1000)
        watchdog.timeout.connect(
            lambda key=dev_key: self.check_alive(key))
        watchdog.start()
        entry["watchdog"] = watchdog

        # 启动帧读取线程
        reader = threading.Thread(
            target=self.read_pipe, args=(dev_key,), daemon=True)
        entry["reader_thread"] = reader
        reader.start()
        return True

    def read_pipe(self, dev_key: str):
        """从子进程 stdout 管道读取双目帧，经 frame_ready 信号发主线程。"""
        entry = self._entries.get(dev_key)
        if not entry or entry["kind"] != "s80m":
            return
        fifo = entry["proc"].stdout  # subprocess.PIPE 的读端
        self.log.emit(tr("双目管道已连接，等待帧数据…"))

        frame_count = 0
        while entry["proc"] and entry["proc"].poll() is None:
            if self.shutting_down:
                break
            try:
                # 帧格式:
                #   [4B left_len][8B left_ts_ns][left_jpg]
                #   [4B right_len][8B right_ts_ns][right_jpg]
                #   [4B imu_count][imu_count × (8B ts_ns + 8B×6 gyro/acc)]
                #   [4B depth_len]                       ← 仅 depth_active
                #     >0: [8B depth_ts][4B w][4B h][w*h*2 字节 uint16 毫米]
                # 时间戳均为 SDK 硬件纳秒时钟（帧/IMU 同源，SLAM 对齐用）
                header = fifo.read(4)
                if len(header) < 4:
                    break
                left_len = struct.unpack(">I", header)[0]
                # 帧头自检：0 或超 8MB 均属管道流错位（如子进程打印进入
                # 管道），防巨额分配/空帧 imdecode 抛异常；正常 1280×800
                # JPEG 远小于 8MB
                if not (0 < left_len <= 8 * 1024 * 1024):
                    self.log.emit(tr(
                        "[双目错误] 帧头异常 left_len={}，管道流错位", left_len))
                    break
                ts_raw = fifo.read(8)
                if len(ts_raw) < 8:
                    break
                left_ts = struct.unpack(">Q", ts_raw)[0]
                left_data = fifo.read(left_len)
                if len(left_data) < left_len:
                    break

                header = fifo.read(4)
                if len(header) < 4:
                    break
                right_len = struct.unpack(">I", header)[0]
                if not (0 < right_len <= 8 * 1024 * 1024):
                    self.log.emit(tr(
                        "[双目错误] 帧头异常 right_len={}，管道流错位", right_len))
                    break
                ts_raw = fifo.read(8)
                if len(ts_raw) < 8:
                    break
                right_ts = struct.unpack(">Q", ts_raw)[0]
                right_data = fifo.read(right_len)
                if len(right_data) < right_len:
                    break

                # IMU 样本块
                hdr = fifo.read(4)
                if len(hdr) < 4:
                    break
                imu_count = struct.unpack(">I", hdr)[0]
                imu_samples = []
                for _ in range(imu_count):
                    rec = fifo.read(56)
                    if len(rec) < 56:
                        break
                    imu_samples.append(struct.unpack(">Q6d", rec))
                if len(imu_samples) < imu_count:
                    break

                # 深度块（子进程 SDK 深度引擎输出；每帧必写保证解析确定性。
                # 尺寸校验防协议错位；校验失败仍已消费 payload 保持同步）
                if entry.get("depth_active"):
                    hdr = fifo.read(4)
                    if len(hdr) < 4:
                        break
                    depth_len = struct.unpack(">I", hdr)[0]
                    if depth_len > 0:
                        if depth_len > 64 * 1024 * 1024:   # 防坏流巨额分配
                            break
                        dh = fifo.read(16)                # >QII = 8+4+4
                        if len(dh) < 16:
                            break
                        # unpack_from（而非 unpack(dh)：本机 Python 严格版
                        # 要求 buffer 恰好 == calcsize，16=16 亦可，但
                        # unpack_from 语义更稳）。注意此处必须 16 字节：
                        # 写端 struct.pack(">QII")=16；多读 4 字节会把
                        # 下一帧帧头吃进本块（实测全流错位→0 帧断连）
                        depth_ts, dw, dh2 = struct.unpack_from(">QII", dh)
                        payload = fifo.read(depth_len)
                        if len(payload) < depth_len:
                            break
                        if depth_len == dw * dh2 * 2 \
                                and 0 < dw <= 4096 and 0 < dh2 <= 4096:
                            depth = np.frombuffer(payload, np.uint16) \
                                .reshape(dh2, dw)
                            self.depth_ready.emit(
                                settings.S80M_DEPTH_SLOT, depth, depth_ts)
                        else:
                            self.log.emit(tr(
                                "[双目] 深度块尺寸不符 w={} h={} len={}，跳过",
                                dw, dh2, depth_len))

                # JPEG 解码（cv2 5.0.0 起空/坏缓冲直接抛 cv2.error 而非
                # 返回 None；异常按流损坏处理，交由下方异常出口收尾）
                try:
                    left = cv2.imdecode(np.frombuffer(left_data, np.uint8),
                                        cv2.IMREAD_COLOR)
                    right = cv2.imdecode(np.frombuffer(right_data, np.uint8),
                                         cv2.IMREAD_COLOR)
                except cv2.error as e:
                    self.log.emit(tr("[双目错误] JPEG 解码失败: {}", e))
                    break
                if left is None or right is None:
                    continue

                # ★ S80M 子进程直接输出左右两个完整单目画面 (1280×800)
                # ★ 镜头方向已由 SDK config (left_cam_rotate_180 /
                #   right_cam_rotate_180 / stereo_swap_lr) 处理，此处不再翻转
                # ★ 显示翻转由主线程 _on_stereo_frame 统一执行（与单目一致）
                self.frame_ready.emit("stereo_left", left, left_ts, imu_samples)
                self.frame_ready.emit("stereo_right", right, right_ts, imu_samples)

                frame_count += 1
                if frame_count == 1:
                    self.log.emit(tr("双目摄像机画面已开始传输"))

            except BrokenPipeError:
                if not self.shutting_down:
                    self.log.emit(tr("[双目] 管道断开"))
                break
            except Exception as e:
                if not self.shutting_down:
                    self.log.emit(tr("[双目错误] 读取帧异常: {}", e))
                break

        if not self.shutting_down:
            # 异常退出（流错位/解码失败/EOF）时确保子进程终止：SIGTERM
            # 走脚本内优雅退出（finally → _fn_destroy 释放 SDK），避免
            # 半死子进程占着相机不死。正常 close() 路径已 terminate，
            # poll() 非 None 不会误杀。
            proc = entry.get("proc")
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            self.log.emit(tr("双目摄像机已断开 (共 {} 帧)", frame_count))

    def check_alive(self, dev_key: str):
        """定时检查双目子进程是否仍在运行（watchdog 槽）。"""
        entry = self._entries.get(dev_key)
        if not entry or entry["kind"] != "s80m":
            return
        ret = entry["proc"].poll()
        if ret is not None:
            entry["watchdog"].stop()
            self.log.emit(tr("[双目] 子进程已退出，退出码: {}", ret))
            # 读取 stderr 末尾（最后 500 字符）
            err_file = entry["stderr_file"]
            if err_file and os.path.isfile(err_file):
                try:
                    with open(err_file, "r") as f:
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        f.seek(max(0, size - 500))
                        tail = f.read()
                    if tail.strip():
                        self.log.emit(tr("[双目 stderr] {}", tail.strip()))
                except Exception:
                    pass

    def close(self, dev_key: str):
        """停止 S80M 子进程并清理进程侧资源，完毕 emit device_closed。

        （主窗口注册表条目由调用方弹出；管线注销/拆槽/清标定在
        device_closed 槽中执行，整体顺序与原 _close_s80m 一致。）
        """
        entry = self._entries.pop(dev_key, None)
        if not entry or entry["kind"] != "s80m":
            return
        # 1. 停止 watchdog 定时器
        wd = entry.get("watchdog")
        if wd:
            wd.stop()
            wd.deleteLater()

        # 2. 先关曝光命令通道（子进程 stdin 线程随 EOF 退出），再终止子进程
        stdin = entry.get("stdin")
        if stdin:
            try:
                stdin.close()
            except Exception:
                pass
        proc = entry["proc"]
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
        # 关闭 stdout 管道 → 唤醒 reader 线程的阻塞 read()
        if proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass

        # 3. 等待 reader 线程退出（管道已关闭，不再阻塞）
        reader = entry.get("reader_thread")
        if reader and reader.is_alive():
            reader.join(timeout=3.0)

        # 4. 清理 stderr 日志文件 + 50fps yaml 临时副本
        err_file = entry.get("stderr_file")
        if err_file and os.path.isfile(err_file):
            try:
                os.unlink(err_file)
            except Exception:
                pass
        cfg_tmp = entry.get("config_file")
        if cfg_tmp and os.path.isfile(cfg_tmp):
            try:
                os.unlink(cfg_tmp)
            except Exception:
                pass

        self.device_closed.emit(dev_key)

    def send_exposure(self, entry: dict, auto: bool, value: float):
        """向 S80M 子进程下发曝光（stdin 行协议；-1.0 = 自动曝光）。

        entry 由调用方按注册表取到（主窗口注册表与离线测试假条目同款
        形状），行协议口径集中于此。
        """
        if not entry:
            return
        stdin = entry.get("stdin")
        if stdin is None:
            return
        v = -1.0 if auto else float(value)
        try:
            stdin.write(f"SET_EXPOSURE {v}\n".encode())
            stdin.flush()
        except (BrokenPipeError, OSError):
            self.log.emit(tr("[双目] 曝光下发失败（子进程已退出）"))

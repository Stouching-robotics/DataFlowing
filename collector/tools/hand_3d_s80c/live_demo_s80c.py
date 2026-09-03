#!/usr/bin/env python3
"""S80C 双目鱼眼实时裸手关键点 2D/3D 渲染 demo。

数据链：
  s80c_depth_worker.py 子进程（SDK ctypes + 预载 OpenCV 4.2 隔离）
    → stdout 管道：矫正左目 BGR（JPEG）+ 深度（float32 毫米，P0 空间）
  S80CSource（本文件）读管道 → _run_3d_chain（复用 hand_3d_d435 全链：
  检测 → 对齐抬升 → 槽位 → 平滑 → 三窗口渲染/导出）。

坐标空间：rgb、2D 关键点、深度、3D 全部在引擎矫正左目 P0 空间
（fx≈457 @1280×800），depth_to_color 恒等（DepthAligner 自测证明
恒等对齐逐点精确）→ 无需跨相机对齐。

性能预期：检测/显示同步（默认，D435 裸手同款口径——关键点与画面
逐帧严格对应、快动手不落后；0.5 缩放下 det 15.7ms/帧，headless
实测 ~45fps、窗口模式 ~35fps）。--det-async 打开时显示全帧直推
~50fps（latest-result 语义，关键点滞后 1-3 帧靠 --extrap-2d 显示
外推补偿）。深度引擎 CPU SGBM（SDK 文档标称 ~0.7s/帧，本机实测
~20fps 深度更新）。深度更新间隔内 worker 重复发送最近一张深度，
tracker 不丢手不硬顶（即便深度慢到 1.4fps，3D 骨架也只在更新点跳变、
帧间平滑）。

用法：
    ./run_live_s80c.sh                 # 实机（相机必须空闲）
按键（与 D435 demo 一致）：
    q/Esc 退出 | d 深度伪彩叠层 | g 黑手套模式 | b 手套姿态后端 |
    s 截图 | r 复位 3D 视角 | 鼠标左键拖拽旋转 3D 视角
"""

import argparse
import json
import os
import queue
import signal
import struct
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import cv2

_WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.dirname(_WORKER_DIR)
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import hand_3d_d435.live_demo as _hd435   # noqa: E402
from hand_3d_d435.live_demo import _run_3d_chain   # noqa: E402

_HEADER = struct.Struct(">BIQIII")
_META_TYPE, _RGB_TYPE, _DEPTH_TYPE, _RIGHT_TYPE = 0, 1, 2, 3
_RAW_RGB_TYPE, _RAW_RIGHT_TYPE = 4, 5    # raw BGR 帧（worker 默认）
_MAX_PAYLOAD = 16 * 1024 * 1024   # raw 1280×800×3B=3MB、深度 4MB，留足余量
_MAX_WH = 8192
_MAX_RESYNC_BYTES = 1_000_000     # 头失步逐字节重同步上限


class S80CSource:
    """拉起 s80c_depth_worker 子进程，后台线程解析管道消息。

    next() → (矫正左目 BGR, 深度毫米 float32)；深度预热期间为
    (bgr, None)（链自动 2D-only）。worker 死亡/管道 EOF → (None, None)
    链正常退出。
    """

    def __init__(self, sdk_dir=None, vikit_config=None, depth_config=None,
                 opencv_dir=None, rect_mode="remap", stereo_view=True,
                 pipe_format=None, raw_dump=None, raw_ring=32,
                 raw_full=False, race_probe=False, double_buffer=False,
                 settle_poll=False, cb_bridge=True):
        worker = os.path.join(_WORKER_DIR, "s80c_depth_worker.py")
        cmd = [sys.executable, worker]
        for flag, val in (("--sdk-dir", sdk_dir), ("--vikit-config", vikit_config),
                          ("--depth-config", depth_config),
                          ("--opencv-dir", opencv_dir),
                          ("--rect-mode", rect_mode)):
            if val:
                cmd += [flag, val]
        if stereo_view:
            cmd += ["--stereo-view"]
        if pipe_format:
            cmd += ["--pipe-format", pipe_format]
        if raw_dump:
            cmd += ["--raw-dump", raw_dump]
        if raw_ring and raw_ring != 32:
            cmd += ["--raw-ring", str(raw_ring)]
        if raw_full:
            cmd += ["--raw-full"]
        if race_probe:
            cmd += ["--race-probe"]
        if double_buffer:
            cmd += ["--double-buffer"]
        if settle_poll:
            cmd += ["--settle-poll"]
        if cb_bridge:
            cmd += ["--cb-bridge"]
        else:
            cmd += ["--no-cb-bridge"]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        # worker 日志走 stderr 直通控制台；stdout 是二进制帧流
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=None, env=env)
        self._q = queue.Queue(maxsize=2)
        self._meta = None
        self._meta_ev = threading.Event()
        self._last_depth = None
        self._last_right = None
        self._cur_right = None
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        # 等握手（P0 内参）：worker 初始化失败会很快退出——轮询 proc
        # 状态提前报错，不干等 15s
        _deadline = time.monotonic() + 15.0
        while not self._meta_ev.is_set():
            if self._proc.poll() is not None:
                self.close()
                raise RuntimeError(
                    f"S80C worker 启动失败（exit={self._proc.returncode}，"
                    "日志见上方 stderr）。\n"
                    "  排查: 相机是否连接且空闲？USB 插拔后端口是否漂移？\n"
                    "  单独跑 worker 看日志: venv/bin/python "
                    "tools/hand_3d_s80c/s80c_depth_worker.py")
            if time.monotonic() > _deadline:
                self.close()
                raise RuntimeError(
                    "S80C worker 初始化超时（15s 未收到握手）。\n"
                    "  排查: 相机是否连接且空闲？USB 插拔后端口是否漂移？\n"
                    "  单独跑 worker 看日志: venv/bin/python "
                    "tools/hand_3d_s80c/s80c_depth_worker.py")
            time.sleep(0.1)
        _sv = " stereo-view=on" if self._meta.get("stereo_view") else ""
        print(f"✓ S80C worker 已就绪: P0 fx={self._meta['fx']:.2f} "
              f"cx={self._meta['cx']:.2f} 基线 "
              f"{self._meta['baseline_mm']:.2f}mm（rect_mode="
              f"{self._meta['rect_mode']}{_sv}）", flush=True)
        self.align_calib = self._build_align_calib(self._meta)
        # 右目手套共享框的视差平移用（链经 getattr 读取；D435 source
        # 无此属性 → 0 → 不平移）
        self.baseline_mm = float(self._meta.get("baseline_mm", 0.0))

    @staticmethod
    def _build_align_calib(meta: dict) -> dict:
        """P0 空间同坐标系：color = depth = P0 内参，depth_to_color 恒等。

        ★ 坑：P0 是 3×4 行主序（[fx,0,cx,tx, 0,fy,cy,ty, 0,0,1,0]），
        fy 在下标 5、cy 在下标 6——曾错取 p0[4]/p0[5]（fy=0）导致对齐
        全零、深度窗全暗、3D 全 NaN。直接用 worker 握手里的显式
        fx/fy/cx/cy 键，不重解析 P0 布局。
        """
        intr = {"fx": float(meta["fx"]), "fy": float(meta["fy"]),
                "cx": float(meta["cx"]), "cy": float(meta["cy"]),
                "width": int(meta["width"]), "height": int(meta["height"])}
        return {
            "color_intrinsics": dict(intr),
            "depth_intrinsics": dict(intr),
            "depth_to_color": {
                "rotation": [[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]],
                "translation": [0.0, 0.0, 0.0]},
        }

    # ── 管道读取线程 ──────────────────────────────────────────

    def _read_exact(self, n: int):
        buf = b""
        try:
            while len(buf) < n:
                chunk = self._proc.stdout.read(n - len(buf))
                if not chunk:
                    return None
                buf += chunk
            return buf
        except (OSError, ValueError):
            return None

    def _reader(self):
        try:
            self._reader_loop()
        except Exception:
            traceback.print_exc(file=sys.stderr)
        finally:
            # 线程无论正常 EOF 还是异常退出都通知链收尾
            self._q.put(None)

    def _reader_loop(self):
        _skipped = 0
        while True:
            hdr = self._read_exact(_HEADER.size)
            if hdr is None:
                break
            typ, seq, ts, w, h, ln = _HEADER.unpack(hdr)
            # 头不合法 = stdout 被污染（如 SDK printf 抢在 worker 重定向
            # 之前进管道）→ 逐字节重同步，保护性兜底（worker 侧已把
            # 重定向提前到任何 SDK 活动之前，正常不会触发）
            if not (typ in (_META_TYPE, _RGB_TYPE, _DEPTH_TYPE, _RIGHT_TYPE,
                            _RAW_RGB_TYPE, _RAW_RIGHT_TYPE)
                    and 0 <= ln <= _MAX_PAYLOAD
                    and 0 <= w <= _MAX_WH and 0 <= h <= _MAX_WH):
                _skipped += 1
                if _skipped > _MAX_RESYNC_BYTES or \
                        self._read_exact(1) is None:
                    print(f"[S80CSource] 管道头持续失步"
                          f"（已跳过 {_skipped}B），放弃读取",
                          file=sys.stderr, flush=True)
                    break
                continue
            payload = self._read_exact(ln)
            if payload is None:
                break
            if typ == _META_TYPE:
                try:
                    self._meta = json.loads(payload.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
                self._meta_ev.set()
            elif typ in (_RGB_TYPE, _RAW_RGB_TYPE):
                img = self._decode(typ, payload, w, h)
                if img is not None:
                    self._push(img, self._last_depth, self._last_right)
            elif typ in (_RIGHT_TYPE, _RAW_RIGHT_TYPE):
                img = self._decode(typ, payload, w, h)
                if img is not None:
                    self._last_right = img
            elif typ == _DEPTH_TYPE:
                try:
                    self._last_depth = np.frombuffer(
                        payload, np.float32).reshape(h, w)
                except ValueError:
                    continue

    @staticmethod
    def _decode(typ, payload, w, h):
        """type=4/5 raw BGR：frombuffer 视图 + copy（保证可写、连续——
        display_frame 拼接与 cv2 原地操作用）；type=1/3 JPEG：imdecode。"""
        if typ in (_RAW_RGB_TYPE, _RAW_RIGHT_TYPE):
            try:
                return np.frombuffer(payload, np.uint8) \
                    .reshape(h, w, 3).copy()
            except ValueError:
                return None
        return cv2.imdecode(np.frombuffer(payload, np.uint8),
                            cv2.IMREAD_COLOR)

    def _push(self, rgb, depth, right):
        """只保留最新帧（队列满丢旧帧，渲染不滞后累积）。"""
        try:
            self._q.put_nowait((rgb, depth, right))
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait((rgb, depth, right))
            except queue.Full:
                pass

    # ── 链契约 ────────────────────────────────────────────────

    def next(self):
        item = self._q.get()      # 阻塞等帧；worker 死亡 → None
        if item is None:
            return None, None
        rgb, depth, right = item
        self._cur_right = right
        return rgb, depth

    def display_frame(self, rgb):
        """win1 显示基底：--stereo-view 时左右并排（检测/深度/3D 仍基于
        rgb 左目帧；右目仅显示 + 右半边独立 2D 关键点渲染）；无右目帧
        时原样返回。"""
        if self._cur_right is not None and \
                self._cur_right.shape[:2] == rgb.shape[:2]:
            return np.concatenate([rgb, self._cur_right], axis=1)
        return rgb

    def right_frame(self):
        """与最近一次 next() 返回的 rgb 同帧配对的右目矫正 BGR；无则 None。"""
        return self._cur_right

    def close(self):
        if self._proc.poll() is None:
            self._proc.terminate()          # SIGTERM → worker 优雅销毁 SDK
            try:
                # 8s 余量：--raw-dump 大环（160 帧半尺寸 ~1.5s）导出 +
                # SDK 销毁都走 worker 的 finally，常规退出 <1s 不受影响
                self._proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2)


# ── 主程序 ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdk-dir", default=None,
                    help="3.9.0 库目录（默认 worker 内 third_party/lib "
                         "自包含副本；勿用 3.9.1——RGB 失败后有段错误史）")
    ap.add_argument("--vikit-config",
                    help="相机配置模板（默认 tools/stereo_s80m/config/"
                         "fays_vikit_50fps.yaml，与主程序同款 50fps；"
                         "端口自动解析）")
    ap.add_argument("--depth-config",
                    help="深度引擎配置（默认 FaysSense_VI_Kit_Release/"
                         "config/perception/stereo_depth/stereo_depth.yaml）")
    ap.add_argument("--opencv-dir",
                    help="SDK 自带 OpenCV 4.2 lib406 目录（默认 "
                         "FaysSense_VI_Kit_Release/thirdparty/opencv-4.2.0-"
                         "linux-x86_64/lib406）")
    ap.add_argument("--rect-mode", choices=("remap", "sdk"), default="remap",
                    help="2D 视图来源：remap=自身鱼眼矫正（默认）；sdk=引擎"
                         "矫正图（验证对照，尺寸可能非 1280×800 致 3D 失效）")
    ap.add_argument("--no-stereo-view", action="store_false",
                    dest="stereo_view", default=True,
                    help="关闭 win1 左右目并排显示（默认开：右目独立 "
                         "MediaPipe 检测并渲染 2D 关键点，右半边仅显示；"
                         "深度/3D 仍只用左目）")
    ap.add_argument("--delegate", default="auto", choices=("cpu", "gpu", "auto"),
                    help="MediaPipe delegate（auto=GPU 冒烟成功则 GPU）")
    ap.add_argument("--fill", type=int, default=0, choices=(0, 1, 2, 3),
                    help="对齐空穴回填轮数（默认 0——恒等对齐 1:1 无上采样"
                         "空穴，纯裁剪最快；需要更密 3D 可 1-3）")
    ap.add_argument("--propagate-max", type=int, default=15,
                    help="槽位丢失帧数硬顶（超限 absent 不幻觉）")
    ap.add_argument("--det-conf", type=float, default=0.4,
                    help="掌心检测置信度阈值（默认 0.4；动作快/丢手可再降到 0.3）")
    ap.add_argument("--track-conf", type=float, default=0.4,
                    help="手部跟踪置信度阈值（默认 0.4；丢手可再降到 0.3）")
    ap.add_argument("--smooth-2d", action="store_true", default=False,
                    help="2D 关键点逐点 OneEuro 平滑（默认关——平滑引入"
                         "~2-3 帧跟随滞后，快动时骨架明显落后于手；raw "
                         "管道 + 50fps 下闪动已基本消除，仍觉抖再开）")
    ap.add_argument("--smooth3d-freq-min", type=float, default=1.5,
                    help="3D 关键点 OneEuro freq_min（逐点 + M1 质心锚定"
                         "共用；默认 1.5）——S80C/S80M 深度图比 D435 噪"
                         "（~20fps 更新、有效率低），3D 抖动更大，默认在"
                         " D435 口径（3.0）上加一倍静止平滑压抖动；3.0="
                         "D435 同款，越大越跟手但越抖，越小越稳但快动滞后"
                         "越大")
    ap.add_argument("--det-scale", type=float, default=0.5,
                    help="裸手检测输入缩放比（默认 0.5 半分辨率——CPU "
                         "XNNPACK 全分辨率 2 手 ~14ms 是窗口模式帧率主瓶颈，"
                         "半分辨率省 ~7ms/帧；丢小/远手可提回 1.0）")
    ap.add_argument("--det-async", action="store_true", dest="det_async",
                    default=False,
                    help="开异步检测（默认关：检测/显示同步——D435 裸手同款"
                         "口径，关键点与画面逐帧严格对应，快动手不落后；"
                         "实测 0.5 缩放下 det 15.7ms/帧，窗口模式 ~35fps。"
                         "打开则检测在后台线程跑 latest-result，显示全帧直推"
                         " ~50fps 但关键点滞后 1-3 帧，靠 --extrap-2d 显示"
                         "外推补偿，快动手仍不如同步跟手）")
    ap.add_argument("--no-extrap-2d", action="store_false", dest="extrap_2d",
                    default=True,
                    help="关闭关键点显示外推（默认开，仅 --det-async 时生效："
                         "异步检测结果来自 1-3 帧前图像，直接画当前帧=快动手"
                         "时骨架落后于手；显示层按框中心速度把关键点平移投影"
                         "到当前时刻，仅显示路径——3D 链/槽位/--export 全用"
                         "原始检测值；同步模式关键点本就与画面逐帧对应，"
                         "外推不介入）")
    ap.add_argument("--tear-probe", action="store_true",
                    dest="tear_probe", default=False,
                    help="开启撕裂自动捕获（默认关——启动不保存帧缓冲、"
                         "退出不导出任何东西；排查画面撕裂时再开：后台保存"
                         "最近 ~2.7s 内部显示缓冲，按 q 退出时自动导出 "
                         "tear_exit_* 到 keypoints_output/，t 键可随时手动"
                         "导出，不持续写盘——看到撕裂直接退出即可，无需掐"
                         "时机；屏幕合成撕裂不会出现在内部帧里，导出帧内有"
                         "水平缝=数据/相机侧，全干净而画面撕裂=显示/远程"
                         "桌面层）")
    ap.add_argument("--no-tear-probe", action="store_false",
                    dest="tear_probe",
                    help="（兼容旧脚本；默认已是关，此 flag 仅显式声明）")
    ap.add_argument("--pipe-format", choices=("raw", "jpeg"),
                    help="透传 worker 管道帧格式（默认 raw；撕裂诊断对照："
                         "raw 有缝而 jpeg 无→指向管道层，两者一致→排除管道层）")
    ap.add_argument("--raw-dump", metavar="DIR",
                    help="透传 worker：退出时把最近若干张 pre-remap 原始帧"
                         "导出 JPEG 到 DIR——与 --tear-probe 同轮捕获，"
                         "对比缝在 raw 还是 remap 后（浏览工具 "
                         "browse_tear_dump.py 可直接开此目录）")
    ap.add_argument("--raw-ring", type=int, default=32, metavar="N",
                    help="透传 worker：raw 环帧数（默认 32≈0.64s@50fps）。"
                         "要覆盖 tear 环全部 96 帧时段用 --raw-ring 160"
                         "（内存 ~6MB×N）")
    ap.add_argument("--raw-full", action="store_true",
                    help="透传 worker：raw 导出全尺寸（默认半尺寸——S80C "
                         "缝比 S80M 细，半尺寸机器检测可能漏）")
    ap.add_argument("--race-probe", action="store_true",
                    help="透传 worker：拷贝窗口竞态探测（实验，默认关——"
                         "每秒日志统计『拷贝窗口内 SDK 改写缓冲』帧数，"
                         "坐实/排除写缓冲与拷贝竞态）")
    ap.add_argument("--double-buffer", action="store_true",
                    help="透传 worker：双缓冲实验（默认关——交替 "
                         "stereo_img.data 指向 A/B 两块缓冲；SDK 黑盒风险，"
                         "仅在 --race-probe 出正信号后试，验证无效即关）")
    ap.add_argument("--settle-poll", action="store_true",
                    help="透传 worker：缓冲稳定轮询（实验，默认关）——取帧"
                         "返回后连续两次快照一致才拷贝使用。若 SDK 返回早于"
                         "其写入线程写完缓冲（拷贝与写入按行交错=水平缝且"
                         "缝行漂移），该等待消除撕裂；稳定时零额外等待。"
                         "2026-08-31 实机已证无效（带率 21.0%%→18.2%%），"
                         "仅留作诊断")
    ap.add_argument("--cb-bridge", action="store_true", default=True,
                    help="透传 worker：回调取帧（默认开=主程序同款撕裂修复）"
                         "——改用 SDK RegisterStereoImageCallback 回调取帧，"
                         "官方 stereo_depth_gui 同款路径（SDK 装配线程写完帧"
                         "才回调，绕过 GetStereoFrames 内部拷贝与装配的竞态，"
                         "该竞态疑为水平缝根因）。经 third_party/cb_bridge/"
                         "桥接 std::function→C ABI")
    ap.add_argument("--no-cb-bridge", action="store_false", dest="cb_bridge",
                    help="回退轮询取帧（诊断对照用）")
    ap.add_argument("--fixed-view", action="store_true",
                    dest="fixed_view3d", default=False,
                    help="开启 3D 固定世界视角（默认关=与 D435 相同：相机"
                         "目标/缩放随手动）；开启后首帧有手时锁定相机目标/"
                         "缩放/网格，手在世界内自由运动（手移出视场就出画），"
                         "按 r 在当前手位置重锁")
    ap.add_argument("--max-bone-len", type=float, default=0.15,
                    help="3D 单个关节（骨）长度上限，米（默认 0.15——正常"
                         "手骨最长腕→MCP ~0.12m，只截深度噪声离群点；0=关闭"
                         "约束）")
    ap.add_argument("--no-raw-2d", action="store_false", dest="raw_2d",
                    default=True,
                    help="关闭原始检测直绘，2D 显示走 3D 槽位链（默认开——"
                         "与 3D 槽位链解耦，槽位 propagated/门控不影响 2D "
                         "骨架显示；裸手/手套模式均生效，手套模式直绘 "
                         "GloveDetector 稳定层输出）")
    ap.add_argument("--glove", action="store_true",
                    help="启动即黑手套模式：glove_package YOLO-World 框 + "
                         "关键点（后端见 --glove-pose-backend；运行中按 g 切换）")
    ap.add_argument("--glove-detector", choices=("world", "det"),
                    default="world",
                    help="黑手套检测器选择：world=yolov8m-worldv2.pt（开放"
                         "词汇，提示词 hand/glove，默认）；det=glove_package"
                         "/runs/hand_det/weights/best.pt（yolo11n 单类 hand "
                         "训练产物，自动走 YOLO 后端）。运行中按 v 键热切换"
                         "（track/滤波状态重置）")
    ap.add_argument("--glove-imgsz", type=int, default=640,
                    help="world 检测输入边长（仅 world 后端；S80C 默认 "
                         "640——远手/小面积手在 320 下等效 4× 缩小掉出模型"
                         "有效尺度（S80C 矫正图 fx≈457，同距离手比 D435 再"
                         "小一半），640 放大 2 倍找回；代价 GPU ~23→31ms/帧，"
                         "近景大召回可能略降（40 张实测 320:40/40 vs 640:"
                         "30/40）；D435 默认 320）")
    ap.add_argument("--glove-weights",
                    help="黑手套检测权重显式路径（优先于 --glove-detector；"
                         "按文件名是否含 world 自动判后端）")
    ap.add_argument("--glove-pose-backend", choices=("rtmpose", "mediapipe"),
                    default="rtmpose",
                    help="黑手套关键点后端：rtmpose=RTMPose hand5（ONNX "
                         "SIMCC 256x256，默认）；mediapipe=MediaPipe "
                         "HandLandmarker（框内裁剪检测，21 点同拓扑）。"
                         "运行中按 b 键热切换")
    ap.add_argument("--glove-det-conf", type=float, default=None,
                    help="黑手套模式检测框阈值（默认 world 0.05 / best.pt 0.3）")
    ap.add_argument("--glove-pose-conf", type=float, default=0.15,
                    help="RTMPose 逐点置信均值门（0=关）：低于门持出上次输出"
                         "并按平滑框位移平移补偿，置信恢复自动续上。S80C 默认"
                         " 0.15（D435 0.3）：鱼眼矫正图逐点置信系统性偏低，"
                         "握拳/抓取等姿势在 0.3 门下一律被 hold 冻结无法捕捉；"
                         "连续低置信满 --glove-hold-max 帧仍会放行本轮骨架")
    ap.add_argument("--glove-hold-max", type=int, default=12,
                    help="黑手套低置信 hold 逃逸帧数（0=立即放行，-1=无限 hold"
                         "旧行为）：连续低置信满该帧数即放行本轮骨架——持续"
                         "低置信=真实新姿势（握拳/抓取），不无限冻结；瞬时低"
                         "置信（运动模糊）仍持旧点防抖")
    ap.add_argument("--glove-nms-iou", type=float, default=0.6,
                    help="world 检测器 NMS IoU（双手重叠框频繁合并→手数闪变"
                         "时降 0.45）")
    ap.add_argument("--glove-lost-timeout", type=int, default=8,
                    help="track 丢框容忍帧数（S80C 默认 8：远手/手背框闪烁"
                         "时 3 帧即死会持续丢手；双手交叉/重叠导致框合并又"
                         "拆开时可继续调高；按会话取舍）")
    ap.add_argument("--glove-new-track-conf", type=float, default=0.1,
                    help="新建 track 的最低框置信度（双阈值，0=关）。S80C "
                         "默认 0.1：world 框置信天然偏低（离线实测 93%% 的框"
                         "<0.25），0.25 门会挡死远手重捕获（track 丢后再也"
                         "建不起来）；flicker 框有碎片框拒绝兜底（≥70%% 面积"
                         "落于已匹配框不新建）")
    ap.add_argument("--glove-box-alpha", type=float, default=0.7,
                    help="track 框 EMA 平滑系数（身份匹配/显示用）")
    ap.add_argument("--glove-pose-box", choices=("raw", "smooth"),
                    default="smooth",
                    help="RTMPose 裁剪框来源：smooth=EMA 平滑框（默认，链稳"
                         "定）；raw=原始检测框（消除稳态滞后但框抖动直通下游）")
    ap.add_argument("--glove-freeze-max", type=int, default=15,
                    help="连续退化冻结输出上限帧数")
    ap.add_argument("--stats", action="store_true",
                    help="统计诊断：退出时打印单手帧/同标签帧/wholesale 等计数")
    ap.add_argument("--depth-overlay", action="store_true",
                    help="启动即开深度伪彩叠层（运行中按 d 切换）")
    ap.add_argument("--no-window", action="store_true",
                    help="不开窗口只跑处理链并打印 fps（验证用）")
    ap.add_argument("--export", help="导出目录：keypoints_2d.parquet（槽位 "
                                    "2D 关键点，像素）、keypoints_3d.parquet"
                                    "（质心锚定 3D，相机系米）、render.mp4"
                                    "（3D 旋转渲染）、rgb_overlay.mp4（原视频"
                                    "叠 2D 关键点）；窗口/无窗口均可")
    args = ap.parse_args()

    source = None
    for attempt in (1, 2):
        try:
            source = S80CSource(args.sdk_dir, args.vikit_config,
                                args.depth_config, args.opencv_dir,
                                args.rect_mode, args.stereo_view,
                                args.pipe_format, args.raw_dump,
                                args.raw_ring, args.raw_full,
                                args.race_probe, args.double_buffer,
                                args.settle_poll, args.cb_bridge)
            break
        except RuntimeError as e:
            print(f"错误: {e}", flush=True)
            if attempt == 2:
                sys.exit(1)
            print("自动重试一次（worker 可能因瞬时设备占用失败）…",
                  flush=True)
            time.sleep(1.0)
    print(f"3D 平滑加强: freq_min={args.smooth3d_freq_min}Hz"
          f"（D435 默认 3.0Hz；S80C/S80M 深度噪声更大，默认 1.5 压抖动）",
          flush=True)
    print(f"3D 视图: "
          f"{'固定世界视角（r 重锁）' if args.fixed_view3d else '相机随手动（D435 同款）'}"
          f"；骨长上限={args.max_bone_len if args.max_bone_len > 0 else '关'}m",
          flush=True)
    if args.det_async:
        print("异步检测: 显示全帧直推 ~50fps，关键点滞后 1-3 帧靠 "
              "--extrap-2d 外推补偿（快动手跟手不如默认同步口径）",
              flush=True)
    else:
        print("同步检测: D435 裸手同款口径——关键点与画面逐帧对应，"
              "快动手不落后（--det-async 可切回 50fps 直推）", flush=True)
    if args.raw_dump:
        # t 键撕裂精确捕获：链的 t 键/退出导出 tear 环时同步发 SIGUSR1
        # 给 worker → 手动导出 raw 环 t_N（与 tear_dump_N 同轮配对，
        # 见撕立即按 t）。worker 仅在 --raw-dump 下装 SIGUSR1 handler
        # （无 handler 时 SIGUSR1 默认动作=杀进程，所以只在 raw-dump
        # 下挂钩子）。
        _orig_wtd = _hd435._write_tear_dump

        def _wtd_hook(ring, tdir):
            _orig_wtd(ring, tdir)
            try:
                if source._proc is not None and source._proc.poll() is None:
                    source._proc.send_signal(signal.SIGUSR1)
                    print("[S80C] 已通知 worker 同步导出 raw 环（t_N）→ "
                          f"{args.raw_dump}", flush=True)
            except Exception:
                pass
        _hd435._write_tear_dump = _wtd_hook

    # win23_every=2：win2 3D / win3 深度每 2 帧更新一次（内容本身以
    # 检测/深度速率变化）——省下的 X11 传输压低显示循环帧预算，
    # 同步检测主循环 ~35fps 不丢帧不跳拍（异步 50fps 直推同理）。
    _run_3d_chain(args, source, source.align_calib, win_title="S80C live",
                  smooth3d_cfg={"freq_min": args.smooth3d_freq_min},
                  fixed_view3d=args.fixed_view3d,
                  max_bone_len=(args.max_bone_len
                                if args.max_bone_len > 0 else None),
                  det_async=args.det_async, win23_every=2,
                  extrap2d=args.extrap_2d, tear_probe=args.tear_probe)


if __name__ == "__main__":
    main()

"""
EgoData 格式录制器 —— 多模态机器人遥操作数据采集。

目录布局（v1.1.0 任务级池化，LeRobot v3 命名，见 docs/data.md）:
  <output_dir>/<task_tag>/
  ├── videos/chunk-{c:03d}/<image_key>/episode-{f:03d}.{ext}
  │                        # 每 episode 每流一文件；RGB=mp4；深度=12-bit
  │                        # 灰度 MP4（hevc Rext gray12le，对数深度码，
  │                        # 见 core/depth_codec.py；x265 无 12-bit 时
  │                        # 回落 FFV1 gray16le 无损 MKV）
  ├── data/chunk-{c:03d}/episode-{f:03d}.parquet
  │                        # 每 episode 一个，zstd，稀疏列（有数据才写）
  └── meta/
      ├── info.json        # 任务级 schema/fps/版本/路径模板/calibration
      ├── stats.json       # 任务级全局统计，每块 {count, mean, std,
      │                    #   min, max}——count 可反推 sum/sum_sq，
      │                    #   文件自身即增量累加器（v1.1.1，无边车）
      ├── tasks.jsonl
      └── episodes/chunk-{c:03d}/episode-{f:03d}.parquet
                           # 每 episode 一个文件（单行），与 data/videos
                           # 同编号；录制结束原子写，重录覆盖
"""

from __future__ import annotations
import os
import json
import time
import subprocess
from typing import Optional, Dict, List

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PyQt5.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker

from config import settings
from core.encoder_probe import (select_encoder, list_working_ffmpegs,
                                find_depth_12bit_ffmpeg)
from core.depth_codec import quantize_depth, depth_video_encoder_args
from core.calibration import StereoCalibration
from core.helpers import (
    task_dir_of, episode_chunk_file, POOLED_CHUNK_SIZE,
    pooled_video_path, pooled_video_dir,
    pooled_data_parquet_path, pooled_episodes_path,
    _legacy_episodes_shard_path,
    pooled_info_path, pooled_stats_path, pooled_tasks_jsonl_path,
    load_stats_acc, merge_stat_block, acc_to_stats_json, recalc_stats,
    list_task_episodes, next_pooled_episode_index,
    mark_recycled_episode, clear_recycled_episode,
)

# 跨进程文件锁（POSIX）；Windows 无 fcntl 时退化为无锁（单机假设，
# 多机共享任务目录须为 POSIX，见 docs/data.md 并发说明）
try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover
    _fcntl = None


# ── 全局 ffmpeg ────────────────────────────────────────

def _get_ffmpeg() -> Optional[str]:
    """解析可用的 ffmpeg 可执行文件（v1.0.9 起委托 core.encoder_probe）。

    优先使用 imageio_ffmpeg 自带的静态二进制（无动态库依赖，最可靠）；
    PATH 中的 ffmpeg（如 conda 版）必须通过 -version 自检才采用 ——
    conda 的 ffmpeg 曾因 libopenvino 符号缺失启动即崩，导致视频静默丢失。
    候选列表进程内缓存（encoder_probe.list_working_ffmpegs）。
    """
    bins = list_working_ffmpegs()
    return bins[0] if bins else None


# ── Parquet 模式 (LeRobot v3 兼容) ──────────────────────

# meta/episodes 分片行：既有 5 列 + v1.1.0 新增（JSON 序列化字符串列）
_EPISODE_COLUMNS = [
    ("episode_index",       pa.int64(),    0),
    ("task_index",          pa.int64(),    0),
    ("start_frame_index",   pa.int64(),    0),
    ("end_frame_index",     pa.int64(),    0),
    ("length",              pa.int64(),    0),
    ("created_at",          pa.float64(),  0.0),
    ("duration_sec",        pa.float64(),  0.0),
    ("drop_stats",          pa.string(),   "{}"),
    ("video_codec",         pa.string(),   "{}"),
    ("calibration",         pa.string(),   "{}"),
]
_EPISODE_SCHEMA = pa.schema([(n, t) for n, t, _d in _EPISODE_COLUMNS])


def _episode_rows_table(rows: List[dict]) -> pa.Table:
    """episode 行列表 → 分片表（缺失键填默认值）。"""
    arrs = {}
    for name, typ, default in _EPISODE_COLUMNS:
        arrs[name] = pa.array([r.get(name, default) for r in rows], type=typ)
    return pa.table(arrs, schema=_EPISODE_SCHEMA)


def _read_episode_rows(path: str) -> List[dict]:
    """读 episodes 文件为行 dict 列表；文件缺失/损坏返回 []。"""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    try:
        return pq.read_table(path).to_pylist()
    except Exception:
        return []


# ═══════════════════════════════════════════════════════
#  EgoData 录制器
# ═══════════════════════════════════════════════════════

class EgoDataWriter(QObject):
    """EgoData 格式录制器 —— 任务级池化布局（v1.1.0）。

    用法:
      w = EgoDataWriter()
      w.start_episode(output_dir, cameras, fps=30,
                      sensors=["right_glove", "left_glove"],
                      task_name="grasp_cup")
      w.write_video_frame("head_left_rgb", bgr_frame)
      w.write_depth_frame(frame_index, depth_uint16)
      w.write_frame_row(frame_index, timestamp_s,
                        sensors={"right_glove": data})
      w.end_episode()
    """

    episode_started = pyqtSignal(str)
    episode_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    log_occurred = pyqtSignal(str)   # 编码器探测等录制期日志（v1.0.9）

    def __init__(self):
        super().__init__()
        self._task_dir: str = ""
        self._episode_index: int = 0
        self._ffmpeg_procs: Dict[str, subprocess.Popen] = {}
        self._camera_dims: Dict[str, tuple] = {}
        self._camera_fps: Dict[str, float] = {}
        self._fps: float = 30.0
        self._frame_count = 0
        self._max_frame_idx = 0          # 单路最大帧号（用于时长计算）
        self._episode_start_s: float = 0.0
        self._episode_created_at: float = 0.0
        self._rows: List[dict] = []
        self._mutex = QMutex()
        self._sensor_names: List[str] = []
        self._sensor_dim: int = settings.SENSOR_DIM
        self._stats: Dict[str, dict] = {}
        self._last_task: str = ""
        self._device_ids: List[str] = []
        self._devices: List[dict] = []
        self._calibrations: Dict[str, object] = {}
        self._calib_dict: dict = {}      # 本 episode 标定（info.json/episodes 行）
        # 稀疏列跟踪：只落盘本 episode 实际有数据的列
        self._present_sensors: set = set()
        self._present_imu: bool = False
        # 深度多槽位（D435 第 n 台 = d435_depth[_n]；S80M 传统路径兜底
        # settings.CAMERA_DEPTH）：槽位键 → depth image_key
        self._depth_slots: Dict[str, str] = {}
        self._depth_dirs: Dict[str, str] = {}        # 每槽深度 image_key 目录（惰性创建）
        self._depth_final_ext: Dict[str, str] = {}   # 每槽最终扩展名（12-bit 灰 mp4；回落 mkv）
        self._depth_12bit: Dict[str, bool] = {}      # 每槽是否 12-bit 灰度编码（否则 FFV1 无损）
        self._depth_proc_started: Dict[str, bool] = {}  # 每槽深度 ffmpeg 是否已启动
        self._depth_enabled: bool = False        # 业务开关：是否录制深度
        self._encoder_choice = None              # 本会话编码器选择（v1.0.9）
        self._drop_stats: Dict[str, int] = {}    # 丢帧统计（pipeline 注入）
        # abort 清理清单：本 episode 落盘的最终文件 + 创建的目录
        self._created_files: List[str] = []
        self._created_dirs: List[str] = []

    # ── 对外属性 ──────────────────────────────────────

    @property
    def task_dir(self) -> str:
        """本 episode 归属的任务目录（池化布局根）。"""
        return self._task_dir

    @property
    def episode_index(self) -> int:
        """本 episode 的全局序号（1 起）。"""
        return self._episode_index

    # ── Episode ────────────────────────────────────────

    def start_episode(self, output_dir: str, cameras: Dict[str, tuple],
                      fps: float = 30,
                      sensors: Optional[List[str]] = None,
                      task_name: str = "",
                      device_ids: Optional[List[str]] = None,
                      calibration: Optional[StereoCalibration] = None,
                      camera_fps: Optional[Dict[str, float]] = None,
                      depth_enabled: Optional[bool] = None,
                      heatmap_near_mm: float = 0.0,
                      heatmap_far_mm: float = 0.0,
                      heatmap_smooth_k: int = 0,
                      heatmap_temporal_alpha: float = 0.0,
                      depth_heatmaps: Optional[Dict[str, dict]] = None,
                      depth_slots: Optional[List[str]] = None,
                      devices: Optional[List[dict]] = None,
                      calibrations: Optional[Dict[str, object]] = None,
                      batch_index: int = 0) -> bool:
        """创建池化任务目录并启动 ffmpeg（直接写最终路径）。

        Args:
            output_dir: 录制根目录
            cameras: {camera_name: (height, width)}，名称如 head_left_rgb
            fps: 默认录制帧率（未在 camera_fps 中指定的摄像机使用）
            sensors: 传感器名称列表
            task_name: 任务标注
            device_ids: 设备 ID 列表（用于 status 列）
            calibration: 双目标定数据（旧单值路径）
            camera_fps: 每路摄像机的独立帧率（用于 ffmpeg -r）
            depth_enabled: 深度录制开关；None 时回退 settings.DEPTH_ENABLED
            heatmap_near_mm/far_mm: 已废弃（v1.1.2 起显示与存储统一为
                core.depth_codec 规范码值 JET，色标固定对数域），仅保留
                签名兼容
            depth_heatmaps: 已废弃（同上），仅保留签名兼容
            depth_slots: 显式深度槽位列表（pipeline 中经 set_depth_camera
                注册的真实深度设备）。空/None → S80M 传统路径兜底单槽
            devices: 录制设备信息（写入 info.json 的 devices 段）:
                [{"key","kind","name","serial","slots"}]
            calibrations: device_key → 标定（多路）；存 info.json 任务级
                最新 + episodes 行，不落盘 calibration/ 目录
            batch_index: 任务进度序号（录制完成次数 + 1）；>0 时按进度
                递增（与文件删除无关），0 时退化为目录扫描取最大 + 1
        """
        ffmpeg = _get_ffmpeg()
        if not ffmpeg:
            self.error_occurred.emit("ffmpeg 未找到"); return False

        self._fps = fps
        self._camera_dims = dict(cameras)
        self._camera_fps = dict(camera_fps) if camera_fps else {}
        self._sensor_names = list(sensors) if sensors else []
        self._sensor_dim = settings.SENSOR_DIM
        self._device_ids = list(device_ids) if device_ids else []
        self._calibrations = dict(calibrations) if calibrations else {}
        self._devices = list(devices) if devices else []
        self._calib_dict = self._plan_calibration_dict(calibration)

        # 任务目录: <output_dir>/<task_tag>/，episode 序号全局递增
        self._task_dir = task_dir_of(output_dir, task_name)
        self._episode_index = next_pooled_episode_index(self._task_dir,
                                                        batch_index)
        cidx, _fidx = episode_chunk_file(self._episode_index)

        # 深度配置：以显式注册的深度设备（pipeline.set_depth_camera →
        # depth_slots）为准，不用槽名是否含 "depth" 猜测——D435/D405 等
        # 槽名可被消歧编号（d435_depth_2），型号与槽名都不该决定是否落盘。
        depth_requested = (depth_enabled if depth_enabled is not None
                           else settings.DEPTH_ENABLED)
        if depth_slots:
            self._depth_slots = {dn: dn for dn in depth_slots if dn}
        else:
            self._depth_slots = {settings.CAMERA_DEPTH: settings.CAMERA_DEPTH}
        # 显式注册的深度设备打开即录（与旧语义一致：D435/D405 不受
        # settings.DEPTH_ENABLED 门控——该开关只作用于 S80M 视差路径）
        self._depth_enabled = depth_requested and bool(depth_slots)
        self._depth_proc_started = {}
        self._depth_dirs = {}
        self._depth_12bit = {}
        # 深度存 12-bit 灰度 MP4（heatmap_near_mm/far_mm/depth_heatmaps
        # 参数自 v1.1.2 起不再作用于落盘：显示与存储统一为
        # core.depth_codec 的规范码值 JET，色标固定对数域）
        self._depth_final_ext = {slot: "mp4" for slot in self._depth_slots}

        # ★ v1.0.9 编码器自动选择：本会话 RGB 流数/最大分辨率驱动探针。
        # 本函数运行在 _start_async 后台线程（写盘线程在其返回后才启动），
        # 探针阻塞不占 UI 节拍；结果进程内缓存，同参数不重探。深度热力图
        # 惰性启动时复用同一选择（同编码器同档位）。
        rgb_cams = {n: d for n, d in cameras.items()
                    if n not in self._depth_slots}
        if rgb_cams:
            max_h, max_w = max(rgb_cams.values(),
                               key=lambda hw: hw[0] * hw[1])
            n_streams = len(rgb_cams)
        else:
            max_h, max_w = (max(cameras.values(),
                                key=lambda hw: hw[0] * hw[1])
                            if cameras else (0, 0))
            n_streams = 1
        if max_w <= 0 or max_h <= 0:
            self.error_occurred.emit("无有效视频流，无法启动录制"); return False
        prefer = settings.RECORD_VIDEO_ENCODER if settings.ENCODER_PROBE_ENABLED \
            else "x264"
        choice = select_encoder(prefer, max_w, max_h, fps, n_streams,
                                log=self.log_occurred.emit)
        if choice is None:
            self.error_occurred.emit("未找到可用视频编码器 (ffmpeg -encoders)")
            return False
        self._encoder_choice = choice

        # 目录：meta/episodes chunk + data chunk + 每路视频 image_key 目录
        # （关键点数据独立存于 keypoints_output/，此处不创建）
        self._created_files = []
        self._created_dirs = []
        dirs = [
            os.path.join("meta", "episodes", f"chunk-{cidx:03d}"),
            os.path.join("data", f"chunk-{cidx:03d}"),
        ]
        for cam_name in cameras:
            if cam_name in self._depth_slots:
                continue
            vdir = pooled_video_dir(self._task_dir, cam_name,
                                    self._episode_index)
            dirs.append(vdir)
            self._created_files.append(
                pooled_video_path(self._task_dir, cam_name,
                                  self._episode_index))
        for sub in dirs:
            d = os.path.join(self._task_dir, sub)
            try:
                os.makedirs(d, exist_ok=True)
            except OSError as e:
                self.error_occurred.emit(str(e)); return False
            self._created_dirs.append(d)

        # 启动 ffmpeg（仅 RGB 相机），每路使用独立帧率；
        # 编码器取本会话选择（v1.0.9 动态化，见上方 select_encoder）
        for cam_name, (h, w) in cameras.items():
            if cam_name in self._depth_slots:
                continue  # 深度不走 MP4（显式注册驱动，不猜槽名）
            cam_r = self._camera_fps.get(cam_name, fps)
            video_path = pooled_video_path(self._task_dir, cam_name,
                                           self._episode_index)
            cmd = [
                choice.ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}", "-r", str(cam_r), "-i", "-",
                "-c:v", choice.encoder, *choice.args,
                "-pix_fmt", "yuv420p", "-f", "mp4", video_path,
            ]
            try:
                self._ffmpeg_procs[cam_name] = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                self.error_occurred.emit(str(e)); return False

        self._episode_start_s = time.time()
        self._episode_created_at = time.time()
        self._frame_count = 0
        self._max_frame_idx = 0
        self._rows.clear()
        self._present_sensors = set()
        self._present_imu = False
        self._reset_stats()
        self._last_task = task_name

        self.episode_started.emit(self._task_dir)
        return True

    # ── 标定规划 ───────────────────────────────────────

    def _plan_calibration_dict(self, legacy_calib) -> dict:
        """规划标定 dict（存 info.json/episodes 行，不落盘 calibration/）。

        顺序由 devices 列表决定（calibrations 按 device_key 取）；旧单值
        calibration 参数写 "head_stereo" 键；全都缺时仍写默认
        head_stereo（保持既有行为）。首台双目型设备同时以 "head_stereo"
        键保存（服务器/回放/stereo_triangulate 兼容路径）。
        """
        ordered = [(d.get("key", ""), d) for d in self._devices]
        if not ordered or not self._calibrations:
            c = legacy_calib if legacy_calib is not None else StereoCalibration()
            return {"head_stereo": c.to_dict()}
        out: Dict[str, dict] = {}
        first_stereo = True
        for dev_key, d in ordered:
            c = self._calibrations.get(dev_key)
            if c is None or not isinstance(c, StereoCalibration):
                continue
            if first_stereo:
                out["head_stereo"] = c.to_dict()
                first_stereo = False
            else:
                out[f"{self._device_slot_prefix(d)}_calibration"] = c.to_dict()
        if not out:
            out["head_stereo"] = StereoCalibration().to_dict()
        return out

    @staticmethod
    def _device_slot_prefix(device: dict) -> str:
        """设备槽前缀: 首槽去掉 "_N" 消歧编号（d435_rgb_2 → d435_rgb）。"""
        slots = device.get("slots") or []
        slot = slots[0] if slots else (device.get("key") or "device")
        import re as _re
        return _re.sub(r"_\d+$", "", str(slot))

    def _video_codec_meta(self) -> dict:
        """本会话视频编码信息（episodes 行 video_codec 段）。"""
        c = self._encoder_choice
        if c is None:
            meta = {"encoder": "libx264", "codec": "H.264",
                    "crf": settings.RECORD_VIDEO_X264_CRF,
                    "selected_by": settings.RECORD_VIDEO_ENCODER}
        else:
            meta = {"encoder": c.encoder, "codec": c.codec, "crf": c.crf,
                    "ffmpeg": c.ffmpeg, "selected_by": c.selected_by,
                    "probe": c.probe}
        # 深度槽编码（12-bit 灰度 HEVC 或 FFV1 回落），按槽记录
        depth_meta = {}
        for slot in self._depth_slots:
            if self._depth_12bit.get(slot):
                depth_meta[slot] = {"codec": "HEVC (Rext) gray12le",
                                    "qp": 6, "format": "12bit log codes"}
            else:
                depth_meta[slot] = {"codec": "FFV1 gray16le",
                                    "format": "uint16 mm"}
        if depth_meta:
            meta["depth"] = depth_meta
        return meta

    # ── 视频帧写入 ─────────────────────────────────────

    def write_video_frame(self, camera_name: str, frame: np.ndarray,
                          flip_vertical: bool = True):
        """写入一帧 BGR 到对应摄像机的 ffmpeg 管道。

        Args:
            camera_name: 摄像机标识
            frame: BGR numpy array
            flip_vertical: 是否上下翻转（默认 True，单目 CameraSlot 路径使用；
                           双目显示帧路径传入 False，因为已在 _on_stereo_frame 完成翻转）
        """
        proc = self._ffmpeg_procs.get(camera_name)
        if proc and proc.stdin and not proc.stdin.closed:
            try:
                if flip_vertical:
                    frame = cv2.flip(frame, 0)
                proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                self.error_occurred.emit(f"ffmpeg pipe broken ({camera_name})")

    # ── 深度帧写入 ─────────────────────────────────────

    def _resolve_depth_slot(self, depth_slot: str) -> str:
        """把调用方深度槽名映射到本 episode 的落盘槽名。

        规则: 槽名精确匹配 → 原样；否则若只有单一深度槽（S80M 传统路径
        的 "stereo_left" 名义槽）→ 回落到该槽；多槽且无匹配 → 原样返回，
        由调用方发现无 ffmpeg proc 时自然跳过。
        """
        if depth_slot in self._depth_slots:
            return depth_slot
        if len(self._depth_slots) == 1:
            return next(iter(self._depth_slots))
        return depth_slot

    def write_depth_frame(self, frame_index: int, depth_frame: np.ndarray,
                          depth_slot: str = ""):
        """写入一帧深度图 —— 12-bit 灰度 MP4（v1.1.2，lerobot v3 同款）。

        单条 ffmpeg 管道：gray12le → libx265 qp=6 近无损（hevc Rext
        单平面，12-bit 对数深度码，见 core/depth_codec.py），直接写
        最终路径 videos/chunk-NNN/<slot>/file-NNN.mp4（单流无需合封）。
        x265 无 12-bit 灰度能力时回落 FFV1 gray16le 无损 MKV（旧格式，
        uint16 毫米原值）。

        帧节拍：调用方（pipeline）按主槽位帧节拍调用本函数，深度源缺帧时
        重复最近帧——深度帧 i 与 RGB 帧 i 严格对齐。

        Args:
            frame_index: 帧序号（与 RGB 严格对应；仅存档签名，未参与写入）
            depth_frame: uint16 numpy array (H, W)，单位毫米
            depth_slot: 深度槽名（多深度相机时区分落盘目录；
                        空/未注册时按单槽回落，多槽无匹配则跳过）
        """
        if not settings.DEPTH_ENABLED and not self._depth_enabled:
            return  # 深度录制已关闭，直接跳过
        slot = self._resolve_depth_slot(depth_slot)
        if slot not in self._depth_slots:
            return  # 多槽下无匹配（如 stereo_left 名义帧）→ 不落盘
        if not self._depth_proc_started.get(slot):
            # 首次调用：创建目录 + 启动单流深度 ffmpeg（12-bit 优先）
            self._depth_dirs[slot] = pooled_video_dir(
                self._task_dir, slot, self._episode_index)
            try:
                os.makedirs(self._depth_dirs[slot], exist_ok=True)
            except OSError:
                pass
            self._created_dirs.append(self._depth_dirs[slot])

            h, w = depth_frame.shape[:2]
            ffmpeg = _get_ffmpeg()
            if ffmpeg:
                depth_r = (self._camera_fps.get(slot)
                           or self._camera_fps.get("stereo_left", self._fps))
                d12 = find_depth_12bit_ffmpeg()
                if d12:
                    # 12-bit 对数深度码 → gray12le HEVC（直接写最终路径）
                    final = pooled_video_path(self._task_dir, slot,
                                              self._episode_index, "mp4")
                    cmd = [
                        d12, "-y", "-f", "rawvideo", "-pix_fmt", "gray12le",
                        "-s", f"{w}x{h}", "-r", str(depth_r), "-i", "-",
                        "-c:v", "libx265", *depth_video_encoder_args(), final,
                    ]
                    self._depth_12bit[slot] = True
                else:
                    # 回落：FFV1 gray16le 无损 uint16 毫米（旧格式 MKV）
                    final = pooled_video_path(self._task_dir, slot,
                                              self._episode_index, "mkv")
                    self._depth_final_ext[slot] = "mkv"
                    cmd = [
                        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "gray16le",
                        "-s", f"{w}x{h}", "-r", str(depth_r), "-i", "-",
                        "-c:v", "ffv1", "-level", "3",
                        "-pix_fmt", "gray16le", final,
                    ]
                    self._depth_12bit[slot] = False
                try:
                    self._ffmpeg_procs[slot] = subprocess.Popen(
                        cmd, stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    self._created_files.append(final)
                except Exception as e:
                    self.error_occurred.emit(f"depth ffmpeg failed: {e}")

            self._depth_proc_started[slot] = True

        proc = self._ffmpeg_procs.get(slot)
        if not proc or not proc.stdin or proc.stdin.closed:
            return
        if self._depth_12bit.get(slot):
            payload = np.ascontiguousarray(
                quantize_depth(depth_frame), dtype="<u2").tobytes()
        else:
            payload = np.ascontiguousarray(depth_frame, dtype="<u2").tobytes()
        try:
            proc.stdin.write(payload)
        except (BrokenPipeError, OSError):
            pass

    # ── 数据行写入 (LeRobot v3 兼容 Parquet) ────────────

    def write_frame_row(self, frame_index: int, timestamp_s: float,
                        sensors: Optional[Dict[str, np.ndarray]] = None,
                        connection_status: Optional[Dict[str, str]] = None,
                        hardware_ns: int = 0,
                        imu_samples: Optional[List] = None):
        """写入一行到 data parquet 缓冲区。

        Args:
            frame_index: 帧序号（与视频帧一致）
            timestamp_s: episode 起始相对秒（f32 墙钟）
            sensors: {传感器名: 16×16 压力数据}，缺失帧填零
            connection_status: {device_id: "connected"/"disconnected"}
            hardware_ns: 帧的 SDK 硬件纳秒时间戳（双目相机，与 IMU 同源时钟；
                         单目/传感器路径为 0）
            imu_samples: 本帧窗口内采集的 IMU 样本列表
                         [(ts_ns, gx, gy, gz, ax, ay, az), ...]
                         仅双目 stereo_left 帧携带（左右目共享同一份样本）
        """
        sensors = sensors or {}
        row = {
            "episode_index": self._episode_index,
            "frame_index":   frame_index,
            "timestamp":     np.float32(timestamp_s),
            "task_index":    0,
            "wall_time":     float(time.time()),   # 绝对 Unix 秒（宿主锚定）
            "hardware_ns":   int(hardware_ns or 0),
            "action":        [0.0],
        }
        for sn in self._sensor_names:
            data = sensors.get(sn)
            if data is None:
                data = np.zeros(self._sensor_dim, dtype=np.float32)
            else:
                data = data.astype(np.float32).ravel()[:self._sensor_dim]
                self._present_sensors.add(sn)   # 稀疏列：有真实数据才落盘
            row[f"observation.{sn}"] = data.tolist()
            self._update_stats(sn, data)

        # IMU: 样本列表 → 嵌套列 (ts 单独一列 int64 保证纳秒精度)；
        # 稀疏：仅本 episode 有 IMU 样本时才写这两列
        imu_samples = imu_samples or []
        if imu_samples:
            self._present_imu = True
            row["imu_ts_ns"] = [int(s[0]) for s in imu_samples]
            row["observation.imu"] = [
                [float(v) for v in s[1:7]] for s in imu_samples
            ]
            self._update_imu_stats(row["observation.imu"])

        row[f"observation.{settings.HAND_POSE_LEFT}"] = \
            [0.0] * settings.HAND_POSE_DIM
        row[f"observation.{settings.HAND_POSE_RIGHT}"] = \
            [0.0] * settings.HAND_POSE_DIM

        cs = connection_status or {}
        for did in self._device_ids:
            row[f"status.{did}"] = cs.get(did, "connected")

        with QMutexLocker(self._mutex):
            self._rows.append(row)
        self._frame_count += 1
        if frame_index + 1 > self._max_frame_idx:
            self._max_frame_idx = frame_index + 1

    # ── 结束 ──────────────────────────────────────────

    def end_episode(self):
        """关闭 ffmpeg，写出全部文件（顺序见 docs/data.md 写入流程）。"""
        self._close_ffmpeg()
        self._write_data_parquet()
        self._append_episode_row()
        self._write_info_json()
        self._merge_stats()
        self._write_tasks_jsonl()
        clear_recycled_episode(self._task_dir)
        self.episode_finished.emit(self._task_dir)

    @property
    def encoder_label(self) -> str:
        """本会话视频编码的 UI 显示名（pipeline 录制状态栏用）。"""
        c = self._encoder_choice
        return c.label if c is not None else "H.264/MP4"

    def set_drop_stats(self, stats: Dict[str, int]) -> None:
        """注入录制丢帧统计（pipeline 在 end_episode 前调用）。"""
        self._drop_stats = dict(stats or {})

    def abort_episode(self):
        """丢弃本 episode 的全部数据（文件级清理）。

        abort 只发生在录制期（end_episode 之前），本 episode 未写 data
        parquet / episodes 行 / stats 合并；只需删除 ffmpeg 已落盘的视频
        终稿与深度临时件，并回滚空目录。防御性回滚 episodes 行（若存在）。

        异常终止不占号：标记本段序号可复用，下次录制优先取回
        （next_pooled_episode_index 读到标记即复用；正常完成时清除）。
        """
        if getattr(self, "_episode_index", 0) > 0 and getattr(
                self, "_task_dir", ""):
            mark_recycled_episode(self._task_dir, self._episode_index)
        self._close_ffmpeg()
        for p in list(self._created_files):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass
        # 防御性：episodes 行回滚（正常流程不可达）
        self._rollback_episode_row()
        # 空目录回滚（深到浅，只删空目录；共享目录自然保留）
        for d in reversed(sorted(self._created_dirs, key=len)):
            try:
                os.rmdir(d)
            except OSError:
                pass
        try:
            os.rmdir(self._task_dir)
        except OSError:
            pass
        self.episode_finished.emit("")

    def _rollback_episode_row(self):
        """移除本 episode 的 episodes 元数据（每段一文件 → os.remove；
        旧分片回退 → 读-改-写删行，保证 episode_row(N) 回退后返回 {}）。
        N=1 时分片与每段文件同路径，先按内容判每段文件才删。"""
        path = pooled_episodes_path(self._task_dir, self._episode_index)
        with self._task_lock():
            removed_path = False
            if os.path.isfile(path):
                rows = _read_episode_rows(path)
                if len(rows) <= 1 and (not rows or rows[0].get(
                        "episode_index") == self._episode_index):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    removed_path = True
            cidx, _fidx = episode_chunk_file(self._episode_index)
            legacy = _legacy_episodes_shard_path(self._task_dir, cidx)
            if os.path.isfile(legacy) and not (removed_path
                                               and legacy == path):
                rows = _read_episode_rows(legacy)
                kept = [r for r in rows
                        if r.get("episode_index") != self._episode_index]
                if len(kept) != len(rows):
                    if kept:
                        _atomic_write_parquet(_episode_rows_table(kept), legacy)
                    else:
                        try:
                            os.remove(legacy)
                        except OSError:
                            pass

    # ── 内部 ──────────────────────────────────────────

    def _close_ffmpeg(self):
        for proc in list(self._ffmpeg_procs.values()):
            try:
                if proc.stdin: proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
        self._ffmpeg_procs.clear()

    def _write_data_parquet(self):
        """写入本 episode 的 data parquet（单文件，稀疏列，zstd）。

        键列恒有（episode_index/frame_index/timestamp/task_index/
        wall_time/hardware_ns/action/手部占位列/status 列）；观测列稀疏：
        只写本 episode 实际有数据的传感器与 IMU 列。
        """
        with QMutexLocker(self._mutex):
            if not self._rows:
                return
            rows = self._rows[:]
            self._rows.clear()

        cols = {
            "episode_index": pa.array([r["episode_index"] for r in rows],
                                      pa.int64()),
            "frame_index":   pa.array([r["frame_index"] for r in rows],
                                      pa.int64()),
            "timestamp":     pa.array([r["timestamp"] for r in rows],
                                      pa.float32()),
            "task_index":    pa.array([r["task_index"] for r in rows],
                                      pa.int64()),
            "wall_time":     pa.array([r["wall_time"] for r in rows],
                                      pa.float64()),
            "hardware_ns":   pa.array([r["hardware_ns"] for r in rows],
                                      pa.int64()),
            "action":        pa.array([r["action"] for r in rows],
                                      pa.list_(pa.float32(), 1)),
        }
        # 稀疏传感器列（有真实数据的才写）
        for sn in sorted(self._present_sensors):
            name = f"observation.{sn}"
            cols[name] = pa.array(
                [r.get(name, [0.0] * self._sensor_dim) for r in rows],
                pa.list_(pa.float32(), self._sensor_dim),
            )
        # 稀疏 IMU 列
        if self._present_imu:
            cols["imu_ts_ns"] = pa.array(
                [r.get("imu_ts_ns", []) for r in rows], pa.list_(pa.int64()))
            cols["observation.imu"] = pa.array(
                [r.get("observation.imu", []) for r in rows],
                pa.list_(pa.list_(pa.float32(), 6)))
        # 手部关键点占位列（后处理回填，恒写）
        for pose_name in [settings.HAND_POSE_LEFT, settings.HAND_POSE_RIGHT]:
            name = f"observation.{pose_name}"
            cols[name] = pa.array(
                [r[name] for r in rows],
                pa.list_(pa.float32(), settings.HAND_POSE_DIM),
            )
        for did in self._device_ids:
            name = f"status.{did}"
            cols[name] = pa.array(
                [r.get(name, "connected") for r in rows], pa.string())

        table = pa.table(cols)
        path = pooled_data_parquet_path(self._task_dir, self._episode_index)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _atomic_write_parquet(table, path)

    def _append_episode_row(self):
        """meta/episodes/chunk-NNN/episode-NNN.parquet 每段一个文件、单行原子写。

        与 data/videos 同编号（episode-000 = episode 1）：每采一条新任务
        就多一个 episode 文件。重录同 episode_index = 覆盖该文件。旧分片
        （每 chunk 一个多行文件）里的同号行一并删除，避免双份。
        跨进程安全：fcntl.flock 锁 meta/episodes/.lock（Windows 无
        fcntl 退化为无锁，多机共享须 POSIX）；写入走临时文件 +
        os.replace 原子替换。
        """
        path = pooled_episodes_path(self._task_dir, self._episode_index)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        row = {
            "episode_index": self._episode_index,
            "task_index": 0,
            "start_frame_index": 0,
            "end_frame_index": max(0, self._max_frame_idx - 1),
            "length": self._max_frame_idx,
            "created_at": float(self._episode_created_at),
            "duration_sec": float(self._max_frame_idx / max(self._fps, 1.0)),
            "drop_stats": json.dumps(self._drop_stats or {},
                                     ensure_ascii=False),
            "video_codec": json.dumps(self._video_codec_meta(),
                                      ensure_ascii=False),
            "calibration": json.dumps(self._calib_dict, ensure_ascii=False),
        }
        with self._task_lock():
            _atomic_write_parquet(_episode_rows_table([row]), path)
            # 旧分片里若有同号行则删掉（避免 episode_row 回退到旧记录）
            cidx, _fidx = episode_chunk_file(self._episode_index)
            legacy = _legacy_episodes_shard_path(self._task_dir, cidx)
            if legacy != path and os.path.isfile(legacy):
                rows = _read_episode_rows(legacy)
                kept = [r for r in rows
                        if r.get("episode_index") != self._episode_index]
                if len(kept) != len(rows):
                    if kept:
                        _atomic_write_parquet(_episode_rows_table(kept),
                                              legacy)
                    else:
                        try:
                            os.remove(legacy)
                        except OSError:
                            pass

    def _task_lock(self):
        """任务级跨进程锁（meta/episodes/.lock）；上下文管理器。"""
        lock_path = os.path.join(self._task_dir, "meta", "episodes", ".lock")
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        fh = open(lock_path, "a+")
        if _fcntl is not None:
            _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
        return _LockedFile(fh)

    def _write_info_json(self):
        """任务级 meta/info.json 读-改-写。

        服务器上传严格依赖既有字段名/类型/结构（features/cameras/
        devices/device_names/sensors/sensor_dim/fps/video/task_name/
        codebase_version/created_at），v1.1.0 新增 format/chunks_size/
        data_path/video_path/episodes_path/video_extensions/calibration/
        total_episodes/app_version。既有键保留合并（其他机器/旧版本写入
        的不丢）；features/cameras/video_extensions 按并集合并（key 一旦
        存在不因后续 episode 缺席而丢失），值以最新 episode 为准。
        """
        # cameras: 必须是 dict 格式 {key: {height, width, fps}}，不能用 list
        # 深度槽过滤以显式注册为准，不猜槽名
        cameras_dict = {}
        for name, (h, w) in self._camera_dims.items():
            if name in self._depth_slots:
                continue
            entry = {"height": h, "width": w}
            cam_fps = self._camera_fps.get(name, self._fps)
            if cam_fps != self._fps:
                entry["fps"] = cam_fps
            cameras_dict[name] = entry

        # features: shape 必须是 2D [16, 16]，不是 1D [256]；
        # 只声明本 episode 实际存在的列（稀疏契约）
        features = {}
        for sn in sorted(self._present_sensors):
            features[f"observation.{sn}"] = {
                "dtype": "float32",
                "shape": [16, 16],
            }
        # 双目 IMU: 每帧窗口内变长样本序列, 每样本 6 轴 (gx,gy,gz,ax,ay,az)
        if self._present_imu:
            features["observation.imu"] = {"dtype": "float32", "shape": [6]}
        features["action"] = {"dtype": "float32", "shape": [1]}

        # devices 紧凑段 + device_names（槽位 → 用户命名）——只加字段，
        # 既有键名/类型一律不动（服务器严格依赖）
        devices_compact = [{
            "key": d.get("key", ""), "kind": d.get("kind", ""),
            "name": d.get("name", ""), "slots": list(d.get("slots") or []),
        } for d in self._devices]
        device_names = {}
        for d in self._devices:
            if d.get("name"):
                for slot in (d.get("slots") or []):
                    device_names[slot] = d["name"]

        # video_extensions: 每 image_key 的文件扩展名（RGB=mp4；深度=mp4
        # 12-bit 灰度，x265 无能力回落 mkv）。并集合并，key 不丢。
        video_extensions = {
            name: "mp4" for name in cameras_dict
        }
        for slot in self._depth_slots:
            video_extensions[slot] = self._depth_final_ext.get(slot, "mp4")

        base = {}
        path = pooled_info_path(self._task_dir)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    base = json.load(f)
            except (OSError, json.JSONDecodeError):
                base = {}
        if not isinstance(base, dict):
            base = {}

        merged_features = dict(base.get("features") or {})
        merged_features.update(features)
        merged_cameras = dict(base.get("cameras") or {})
        merged_cameras.update(cameras_dict)
        merged_exts = dict(base.get("video_extensions") or {})
        merged_exts.update(video_extensions)

        info = dict(base)
        info.update({
            "format": "pooled_episodes_v1",
            "chunks_size": POOLED_CHUNK_SIZE,
            "data_path": "data/chunk-{c:03d}/episode-{f:03d}.parquet",
            "video_path": "videos/chunk-{c:03d}/{image_key}/episode-{f:03d}.{ext}",
            "episodes_path": "meta/episodes/chunk-{c:03d}/episode-{f:03d}.parquet",
            "codebase_version": "v3.0",
            "app_version": settings.APP_VERSION,
            "fps": self._fps,
            "video": len(cameras_dict) > 0,
            "task_name": self._last_task or "",
            "features": merged_features,
            "cameras": merged_cameras,
            "devices": devices_compact,
            "device_names": device_names,
            "sensors": sorted(self._present_sensors),
            "sensor_dim": self._sensor_dim,
            "created_at": time.time(),
            "calibration": self._calib_dict,   # 任务级最新标定
            "total_episodes": len(list_task_episodes(self._task_dir)),
            "video_extensions": merged_exts,
        })
        self._atomic_write_json(path, info)

    def _merge_stats(self):
        """把本 episode 统计合并进任务级 stats.json（v1.1.1 自含累加器）。

        stats.json 每块 {count, mean, std, min, max}——count 可反推
        sum/sum_sq，因此文件自身就是增量累加器，无需 .stats_state 边车。
        旧格式（无 count）或损坏 → recalc_stats 全量重扫重建一次。
        只统计实际存在的列（present 传感器 + present IMU）；
        action 恒为占位（mean 0/std 1，与旧格式一致）。
        """
        acc, need_recalc = load_stats_acc(self._task_dir)
        if need_recalc:
            acc = recalc_stats(self._task_dir)

        for sn in sorted(self._present_sensors):
            merge_stat_block(acc, f"observation.{sn}",
                             self._stats.get(sn), dim=self._sensor_dim)
        if self._present_imu:
            merge_stat_block(acc, "observation.imu",
                             self._stats.get("imu"), dim=6)

        self._atomic_write_json(pooled_stats_path(self._task_dir),
                                acc_to_stats_json(acc))

    def _write_tasks_jsonl(self):
        """任务级 meta/tasks.jsonl（不存在才建，服务器依赖此文件识别任务）。"""
        path = pooled_tasks_jsonl_path(self._task_dir)
        if os.path.isfile(path):
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "task_index": 0,
                "task": self._last_task or "default recording",
            }, f, ensure_ascii=False)
            f.write("\n")

    # ── 原子写入工具 ──────────────────────────────────

    def _atomic_write_json(self, path: str, data: dict):
        """JSON 原子写：临时文件 + os.replace（同盘保证原子）。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = os.path.join(os.path.dirname(path),
                           f".{os.path.basename(path)}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    # ── 统计量 ────────────────────────────────────────

    def _reset_stats(self):
        self._stats.clear()

    def _ensure_stats(self, sensor_name: str):
        if sensor_name not in self._stats:
            self._stats[sensor_name] = {
                "sum":    np.zeros(self._sensor_dim, dtype=np.float64),
                "sum_sq": np.zeros(self._sensor_dim, dtype=np.float64),
                "min":    np.full(self._sensor_dim, np.inf, dtype=np.float64),
                "max":    np.full(self._sensor_dim, -np.inf, dtype=np.float64),
                "count":  0,
            }

    def _update_stats(self, sensor_name: str, arr: np.ndarray):
        self._ensure_stats(sensor_name)
        st = self._stats[sensor_name]
        a = arr.astype(np.float64)
        st["sum"] += a
        st["sum_sq"] += a * a
        st["min"] = np.minimum(st["min"], a)
        st["max"] = np.maximum(st["max"], a)
        st["count"] += 1

    # ── IMU 统计（样本级, 6 轴, 区别于传感器的 256 维帧级统计）──

    def _ensure_imu_stats(self):
        if "imu" not in self._stats:
            self._stats["imu"] = {
                "sum":    np.zeros(6, dtype=np.float64),
                "sum_sq": np.zeros(6, dtype=np.float64),
                "min":    np.full(6, np.inf, dtype=np.float64),
                "max":    np.full(6, -np.inf, dtype=np.float64),
                "count":  0,
            }

    def _update_imu_stats(self, samples: List[List[float]]):
        """按 IMU 样本累积 6 轴统计（count = 样本总数）。

        samples: 本帧窗口的样本列表, 每样本 [gx, gy, gz, ax, ay, az]
        """
        if not samples:
            return
        self._ensure_imu_stats()
        st = self._stats["imu"]
        a = np.asarray(samples, dtype=np.float64).reshape(-1, 6)
        st["sum"] += a.sum(axis=0)
        st["sum_sq"] += np.square(a).sum(axis=0)
        st["min"] = np.minimum(st["min"], a.min(axis=0))
        st["max"] = np.maximum(st["max"], a.max(axis=0))
        st["count"] += len(samples)


# ── 模块级工具 ────────────────────────────────────────

class _LockedFile:
    """flock 锁句柄上下文（释放锁 + 关闭文件）。"""

    def __init__(self, fh):
        self._fh = fh

    def __enter__(self):
        return self._fh

    def __exit__(self, *exc):
        try:
            if _fcntl is not None:
                _fcntl.flock(self._fh.fileno(), _fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        return False


def _atomic_write_parquet(table: pa.Table, path: str):
    """Parquet 原子写：临时文件 + os.replace（同盘保证原子）。"""
    tmp = os.path.join(os.path.dirname(path),
                       f".{os.path.basename(path)}.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)

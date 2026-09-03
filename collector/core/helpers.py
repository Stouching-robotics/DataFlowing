"""
通用工具函数 —— ID 生成、时间戳、时长格式化。
"""

import os
import json
import re
import uuid
import ctypes
from datetime import datetime


def new_id() -> str:
    """生成一个 12 位十六进制短 ID。"""
    return uuid.uuid4().hex[:12]


def utcnow() -> str:
    """返回 ISO-8601 格式的 UTC 时间戳字符串。"""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def send_to_recycle_bin(path: str) -> bool:
    """将文件或文件夹移入回收站。

    Windows: SHFileOperationW (FOF_ALLOWUNDO → 回收站)
    Linux:   gio trash（GNOME/GLib 自带）；无 gio 时手动移入
             ~/.local/share/Trash（GNOME 标准回收站）。

    返回 True 表示成功，False 表示失败。
    """
    if not os.path.exists(path):
        return False

    # ── Linux: gio trash（GNOME 桌面自带，移入桌面回收站）──
    if os.name != "nt":
        import shutil
        import subprocess
        gio = shutil.which("gio")
        if gio:
            try:
                ret = subprocess.run([gio, "trash", path],
                                     capture_output=True, timeout=30)
                if ret.returncode == 0:
                    return True
            except Exception:
                pass
        # fallback: 手动移入 GNOME 回收站目录（记录原始路径以便恢复）
        trash_dir = os.path.expanduser("~/.local/share/Trash")
        try:
            files_dir = os.path.join(trash_dir, "files")
            info_dir = os.path.join(trash_dir, "info")
            os.makedirs(files_dir, exist_ok=True)
            os.makedirs(info_dir, exist_ok=True)
            base = os.path.basename(path.rstrip("/")) or "item"
            dest = os.path.join(files_dir, base)
            if os.path.exists(dest):  # 重名时加后缀
                stem, ext = os.path.splitext(base)
                i = 1
                while os.path.exists(dest):
                    dest = os.path.join(files_dir, f"{stem}_{i}{ext}")
                    i += 1
            os.rename(path, dest)
            import time
            with open(os.path.join(info_dir, os.path.basename(dest) + ".trashinfo"), "w") as f:
                f.write("[Trash Info]\n")
                f.write(f"Path={path}\n")
                f.write(f"DeletionDate={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
            return True
        except Exception:
            return False

    # ── Windows: SHFileOperationW ──
    try:
        # SHFileOperationW 需要双 null 结尾的宽字符串
        from ctypes import wintypes
        SHF = ctypes.windll.shell32.SHFileOperationW

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("wFunc", wintypes.UINT),
                ("pFrom", wintypes.LPCWSTR),
                ("pTo", wintypes.LPCWSTR),
                ("fFlags", wintypes.WORD),
                ("fAnyOperationsAborted", wintypes.BOOL),
                ("hNameMappings", wintypes.LPVOID),
                ("lpszProgressTitle", wintypes.LPCWSTR),
            ]

        FO_DELETE = 3
        FOF_ALLOWUNDO = 0x40
        FOF_NOCONFIRMATION = 0x10
        FOF_SILENT = 0x04

        fileop = SHFILEOPSTRUCTW()
        fileop.wFunc = FO_DELETE
        fileop.pFrom = path + "\0"    # null 结尾（ctypes 自动处理宽字符）
        fileop.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION

        result = SHF(ctypes.byref(fileop))
        return result == 0
    except Exception:
        return False


def format_duration(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS 字符串。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── EgoData 会话摘要 ──────────────────────────────

def session_size_mb(session_dir: str) -> float:
    """遍历会话目录累加所有文件大小，返回 MB。目录不存在/出错返回 0。"""
    if not session_dir or not os.path.isdir(session_dir):
        return 0.0
    total = 0
    try:
        for root, _dirs, files in os.walk(session_dir):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except OSError:
                    pass
    except OSError:
        return 0.0
    return total / (1024.0 * 1024.0)


def read_metadata(session_dir: str) -> dict:
    """读 EgoData metadata.json，任何失败返回 {}。"""
    try:
        p = egodata_metadata_path(session_dir)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _slot_base(name: str, all_names) -> str:
    """流名 → 设备基名（legacy 会话无 devices 段时的启发式）。

    兼容三种命名惯例：
      xxx_rgb    → xxx
      xxx_depth  → xxx（同组存在 xxx_rgb 时）
      xxx_depth  → xxx_depth（同组存在 xxx_depth_rgb，即设备本身被命名为
                   xxx_depth，如 D435_depth）
    同前缀多台设备的 _N 编号后缀（settings 自动追加）保留在基名上：
      d435_rgb_2 / d435_depth_2 → d435_2
    其余原样返回（stereo_left/stereo_right 等）。
    """
    m = re.match(r"^(.*)_(\d+)$", name)
    core, num = (m.group(1), m.group(2)) if m else (name, "")
    if core.endswith("_rgb"):
        base = core[:-4]
    elif core.endswith("_depth"):
        if core + "_rgb" in all_names:
            base = core
        elif core[:-6] + "_rgb" in all_names:
            base = core[:-6]
        else:
            base = core
    else:
        base = core
    return base + ("_" + num if num else "")


def _sub_name(slot: str, dev_name: str, slots: list) -> str:
    """槽名 → 子画面名：优先去设备名前缀，其次去槽名公共前缀。

    slot == dev_name 的惯例是深度槽以设备本身命名（如 D435_depth），
    子画面名即 "depth"。
    """
    if slot == dev_name:
        return "depth"
    if dev_name and slot.startswith(dev_name + "_"):
        return slot[len(dev_name) + 1:]
    if len(slots) > 1:
        prefix = os.path.commonprefix(slots)
        if prefix:
            return slot[len(prefix):].lstrip("_")
    return slot


def _device_display_names(meta: dict) -> list:
    """录制历史"摄像机"列文本：按设备聚合，不把子画面平铺当相机。

    D435 一台设备出 rgb+depth 两路流 → "D435_depth (rgb, depth)"；
    双目 S80M → "FaysSense S80M (left, right)"；单流设备只显示设备名。
    有 devices 段按设备条目聚合（跳过 data_ble 手套/无槽设备）；
    legacy 会话无 devices 段时按 _slot_base 对 cameras 键分组。
    """
    devices = meta.get("devices") or []
    groups = []      # [(设备名, [槽名...])]
    if devices:
        for d in devices:
            slots = list(d.get("slots") or [])
            if not slots or d.get("kind") == "data_ble":
                continue
            name = d.get("name") or ""
            if not name:
                if len(slots) > 1:
                    prefix = os.path.commonprefix(slots).rstrip("_")
                    name = prefix or _slot_base(slots[0], set(slots))
                else:
                    name = _slot_base(slots[0], set(slots))
            groups.append((name, slots))
    else:
        cam_names = list((meta.get("cameras") or {}).keys())
        by_base = {}
        for n in cam_names:
            by_base.setdefault(_slot_base(n, set(cam_names)), []).append(n)
        groups = list(by_base.items())
    disp = []
    for name, slots in groups:
        subs = [s for s in (_sub_name(s, name, slots) for s in slots) if s]
        if len(subs) >= 2:
            disp.append(f"{name} ({', '.join(subs)})")
        else:
            disp.append(name)
    return disp


def session_summary(session_dir: str,
                    frames_by_cam: dict,
                    episode_index: int = 0) -> tuple:
    """会话展示摘要。返回 (camera_list, duration_sec, size_mb)。

    episode_index > 0 且目录为池化布局时走任务级口径：
      - camera_list: info.json devices 段按设备聚合
      - duration_sec: episodes 每段文件行 duration_sec
      - size_mb: episode_size_mb(task_dir, N)（只统计本 episode 文件组）
    否则走旧会话目录口径（历史自愈/未迁移残留）：
      - camera_list: 按设备聚合的显示名（D435 一台设备出 rgb+depth 两路
        子画面 → "D435_depth (rgb, depth)"，见 _device_display_names）；
        metadata 缺失时用 frames_by_cam 中有帧的键（sorted）；
        两者皆空返回 ""（调用方兜底）
      - duration_sec: max(每相机 帧数/fps)；每相机 fps 优先
        cameras[cam].fps（egodata_writer 只在异于全局时写入），缺省
        metadata["fps"]，再缺省 settings.RECORDING_FPS；帧数为 0 的相机
        贡献 0；帧数全部缺失（旧记录自愈）时降级 timestamps.json 最后一
        帧相对时间，再降级 total_frames / 最小 fps（近似）
      - size_mb: session_size_mb(session_dir)
    """
    if episode_index and detect_session_format(session_dir) == "pooled":
        info = {}
        try:
            with open(pooled_info_path(session_dir), "r",
                      encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, json.JSONDecodeError):
            info = {}
        row = episode_row(session_dir, episode_index)
        duration = float(row.get("duration_sec") or 0.0)
        display = ", ".join(_device_display_names(info)) if info else ""
        if not display:
            # 无 devices 段兜底：本 episode 实际存在的视频 key
            display = ", ".join(
                sorted(episode_video_files(session_dir, episode_index)))
        return display, duration, episode_size_mb(session_dir, episode_index)

    meta = read_metadata(session_dir)
    cam_names = list((meta.get("cameras") or {}).keys())
    if not cam_names:
        cam_names = sorted(k for k, v in (frames_by_cam or {}).items() if v > 0)
    fps_list = []
    for cam in cam_names:
        info = (meta.get("cameras") or {}).get(cam) or {}
        fps = float(info.get("fps") or meta.get("fps") or _settings.RECORDING_FPS)
        frames = float((frames_by_cam or {}).get(cam, 0))
        if frames > 0:
            fps_list.append(frames / max(fps, 1.0))
    duration = max(fps_list) if fps_list else 0.0
    if duration == 0.0 and cam_names:
        duration = _timestamps_duration(session_dir, meta, cam_names)
    display = ", ".join(_device_display_names(meta)) if meta else ""
    if not display:
        display = ", ".join(cam_names)      # 无 cameras 元数据 → 旧行为
    return display, duration, session_size_mb(session_dir)


def _timestamps_duration(session_dir: str, meta: dict, cam_names: list) -> float:
    """timestamps.json 降级补算时长：最后一条相对时间（精确），
    再降级 total_frames / 最小 fps（近似，同 playback 的 _get_effective_fps 思路）。"""
    try:
        p = egodata_timestamps_path(session_dir)
        if not os.path.isfile(p):
            return 0.0
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts_list = data.get("timestamps") or []
        if ts_list and isinstance(ts_list[-1], dict):
            last_ts = ts_list[-1].get("timestamp")
            if isinstance(last_ts, (int, float)) and last_ts > 0:
                return float(last_ts)
        total = data.get("total_frames")
        if isinstance(total, (int, float)) and total > 0:
            fps_values = []
            for name in cam_names:
                info = (meta.get("cameras") or {}).get(name) or {}
                fps_values.append(float(info.get("fps") or meta.get("fps")
                                        or _settings.RECORDING_FPS))
            fps = min(fps_values) if fps_values else float(_settings.RECORDING_FPS)
            return float(total) / max(fps, 1.0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return 0.0


def format_size_mb(mb: float) -> str:
    """大小列文本: >0 显示 "12.3 MB"，0/缺失显示 "-"。"""
    if not mb or mb <= 0:
        return "-"
    return f"{mb:.1f} MB"


def timestamp_filename(prefix: str = "rec", ext: str = ".avi") -> str:
    """生成带时间戳的文件名，例如 rec_20260804_143052.avi"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}{ext}"


def _sanitize_tag(raw: str) -> str:
    """清洗标签名，保留字母数字、中文和下划线。"""
    tag = raw.strip() if raw else "session"
    return "".join(c if c.isalnum() or c in "_一-鿿" else "_" for c in tag)


def task_tag(task_name: str = "") -> str:
    """生成任务分类标签（用于子文件夹名）。

    Args:
        task_name: 任务标注（如 "grasp_cup"），为空时返回 "session"

    Returns:
        清洗后的标签名，例如 "grasp_cup"、"test00"、"session"
    """
    return _sanitize_tag(task_name)


def session_dirname(task_name: str = "", batch_index: int = 0) -> str:
    """生成会话目录名。

    Args:
        task_name: 任务标注（如 "grasp_cup"），为空时默认用 "session"

    Returns:
        目录名字符串，例如 "grasp_cup_20260805_195046" 或 "session_20260805_195046"
    """
    tag = _sanitize_tag(task_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if batch_index > 0:
        return f"{tag}_{batch_index:04d}_{stamp}"
    return f"{tag}_{stamp}"


# ═══════════════════════════════════════════════════════════
#  LeRobot v3 分块存储路径工具
# ═══════════════════════════════════════════════════════════

def chunk_dir(chunk_index: int = 0) -> str:
    """chunk 目录名: chunk-000"""
    return f"chunk-{chunk_index:03d}"


def chunk_file(chunk_index: int = 0, name: str = "file") -> str:
    """chunk 内文件名: file-000.parquet / file-000.mp4"""
    return f"{name}-{chunk_index:03d}"


def data_parquet_path(session_dir: str, chunk_index: int = 0) -> str:
    """data/chunk-000/file-000.parquet"""
    return os.path.join(session_dir, "data", chunk_dir(chunk_index),
                        chunk_file(chunk_index) + ".parquet")


def episode_parquet_path(session_dir: str, chunk_index: int = 0) -> str:
    """meta/episodes/chunk-000/file-000.parquet"""
    return os.path.join(session_dir, "meta", "episodes", chunk_dir(chunk_index),
                        chunk_file(chunk_index) + ".parquet")


def video_mp4_path(session_dir: str, camera_key: str, chunk_index: int = 0) -> str:
    """videos/<camera_key>/chunk-000.mp4"""
    return os.path.join(session_dir, "videos", camera_key,
                        f"{chunk_dir(chunk_index)}.mp4")


def tasks_jsonl_path(session_dir: str) -> str:
    """meta/tasks.jsonl"""
    return os.path.join(session_dir, "meta", "tasks.jsonl")


# ═══════════════════════════════════════════════════════════
#  池化任务布局路径工具（v1.1.0）
# ═══════════════════════════════════════════════════════════

POOLED_CHUNK_SIZE = 1000


def episode_chunk_file(episode_index: int) -> tuple:
    """全局 episode 序号 N（1 起）→ (chunk_index, file_index)。

    每 chunk 1000 个 episode，episode 文件号三位零起：
      N=1    → (0, 0)    chunk-000/episode-000
      N=1000 → (0, 999)
      N=1001 → (1, 0)    chunk-001/episode-000
    """
    n = max(int(episode_index) - 1, 0)
    return n // POOLED_CHUNK_SIZE, n % POOLED_CHUNK_SIZE


def pooled_file_stem(file_index: int, ext: str = "") -> str:
    """chunk 内文件名: episode-000.parquet / episode-000.mp4（v1.1.2 起）"""
    return f"episode-{file_index:03d}{ext}"


def episode_file_suffix(episode_index: int) -> int:
    """episode N 的 episode 文件号（0 基）：episode-000 = 第 1 段（= 本地文件后缀）。

    上传/回放界面的任务名后缀与上传表单 episode_index 均用此值，
    与本地 episode-NNN 完全对齐；真实全局序号 N 始终在 parquet 行内。
    """
    return episode_chunk_file(episode_index)[1]


def task_dir_of(base_dir: str, task_name: str) -> str:
    """池化任务目录: <base_dir>/<task_tag>（目录名与旧任务目录一致）。"""
    return os.path.join(base_dir, task_tag(task_name))


def pooled_video_path(task_dir: str, image_key: str, episode_index: int,
                      ext: str = "mp4") -> str:
    """videos/chunk-NNN/<image_key>/episode-NNN.<ext>（每 episode 每流一文件）"""
    c, f = episode_chunk_file(episode_index)
    return os.path.join(task_dir, "videos", chunk_dir(c), image_key,
                        pooled_file_stem(f, f".{ext}"))


def pooled_video_dir(task_dir: str, image_key: str, episode_index: int) -> str:
    """videos/chunk-NNN/<image_key>/ 目录路径"""
    c, _ = episode_chunk_file(episode_index)
    return os.path.join(task_dir, "videos", chunk_dir(c), image_key)


def pooled_data_parquet_path(task_dir: str, episode_index: int) -> str:
    """data/chunk-NNN/episode-NNN.parquet（每 episode 一个，稀疏列）"""
    c, f = episode_chunk_file(episode_index)
    return os.path.join(task_dir, "data", chunk_dir(c),
                        pooled_file_stem(f, ".parquet"))


def pooled_episodes_path(task_dir: str, episode_index: int) -> str:
    """meta/episodes/chunk-NNN/episode-NNN.parquet（每 episode 一个文件，
    与 data/videos 同编号：episode-000 = episode 1；单行 10 列）。"""
    c, f = episode_chunk_file(episode_index)
    return os.path.join(task_dir, "meta", "episodes", chunk_dir(c),
                        pooled_file_stem(f, ".parquet"))


def _legacy_episodes_shard_path(task_dir: str, chunk_index: int) -> str:
    """v1.1.0 旧布局的 episodes 分片路径（每 chunk 一个 ≤1000 行文件，
    文件名 = chunk 号，即 file-{chunk:03d}.parquet——v1.1.2 每段文件改
    episode- 前缀后两者不再重名）。仅作旧数据回退读取/删行用。"""
    return os.path.join(task_dir, "meta", "episodes", chunk_dir(chunk_index),
                        chunk_file(chunk_index) + ".parquet")


def pooled_info_path(task_dir: str) -> str:
    """meta/info.json（任务级）"""
    return os.path.join(task_dir, "meta", "info.json")


def pooled_stats_path(task_dir: str) -> str:
    """meta/stats.json（任务级全局统计，块含 count/mean/std/min/max）。

    v1.1.1 起 stats.json 自身即增量累加器（count 可反推 sum/sum_sq），
    不再有 .stats_state.json 边车。
    """
    return os.path.join(task_dir, "meta", "stats.json")


def pooled_tasks_jsonl_path(task_dir: str) -> str:
    """meta/tasks.jsonl（任务级）"""
    return os.path.join(task_dir, "meta", "tasks.jsonl")


def list_task_episodes(task_dir: str) -> list:
    """扫描池化任务目录，返回已存在的全局 episode 序号列表（升序）。

    依据 data/ 与 videos/ 下的 episode-NNN 文件组（episodes 行可能因
    abort 回滚而缺失，数据文件才是权威）。任务目录不存在返回 []。
    """
    if not os.path.isdir(task_dir):
        return []
    ns = set()
    for sub in ("data", "videos"):
        root = os.path.join(task_dir, sub)
        if not os.path.isdir(root):
            continue
        for cname in os.listdir(root):
            m = re.match(r"^chunk-(\d+)$", cname)
            if not m:
                continue
            cdir = os.path.join(root, cname)
            if not os.path.isdir(cdir):
                continue
            chunk = int(m.group(1))
            for entry in os.listdir(cdir):
                p = os.path.join(cdir, entry)
                if os.path.isdir(p):      # videos: image_key 目录
                    for fn in os.listdir(p):
                        fm = re.match(r"^episode-(\d+)\.", fn)
                        if fm:
                            ns.add(chunk * POOLED_CHUNK_SIZE
                                   + int(fm.group(1)) + 1)
                else:                     # data: 直接 episode-NNN.parquet
                    fm = re.match(r"^episode-(\d+)\.parquet$", entry)
                    if fm:
                        ns.add(chunk * POOLED_CHUNK_SIZE
                               + int(fm.group(1)) + 1)
    return sorted(ns)


def recycled_episode_path(task_dir: str) -> str:
    """异常终止回退标记文件路径（meta/recycled_episode.json）。"""
    return os.path.join(task_dir, "meta", "recycled_episode.json")


def read_recycled_episode(task_dir: str) -> int:
    """读取上次异常终止释放的 episode 序号；无标记/损坏返回 0。"""
    try:
        with open(recycled_episode_path(task_dir), "r",
                  encoding="utf-8") as f:
            data = json.load(f)
        n = int(data.get("episode_index", 0) or 0)
        return n if n > 0 else 0
    except (OSError, ValueError, TypeError):
        return 0


def mark_recycled_episode(task_dir: str, episode_index: int) -> None:
    """异常终止后标记该序号可复用（下次录制优先取回——不占号语义）。

    原子写（临时件 + os.replace）；episode_index <= 0 视为清除。
    标记由本段正常完成（writer.end_episode）时清除。
    """
    path = recycled_episode_path(task_dir)
    if episode_index <= 0:
        clear_recycled_episode(task_dir)
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"episode_index": int(episode_index),
                       "freed_at": utcnow()}, f)
        os.replace(tmp, path)
    except OSError:
        pass


def clear_recycled_episode(task_dir: str) -> None:
    """录制正常完成后清除回退标记（该号已被本段重新占住）。"""
    try:
        os.remove(recycled_episode_path(task_dir))
    except OSError:
        pass


def next_pooled_episode_index(task_dir: str, batch_index: int = 0) -> int:
    """返回任务的下一个全局 episode 序号。

    权威 = 任务进度序号 batch_index（录制完成次数 + 1，与本地文件删除
    无关——上传后自动删除不会让序号回退）与目录扫描 max + 1 取大
    （防与现存文件组重名）。batch_index <= 0 退化为纯目录扫描。

    异常终止回退（不占号）：上次中止写下的 recycled 标记若其号未被
    文件组占用，优先复用该号——batch_index 水位跑在前面也
    不跳号；号已被占（跨机共享目录/遗留）时自动放弃标记并走常规取号。
    """
    existing = list_task_episodes(task_dir)
    recycled = read_recycled_episode(task_dir)
    if recycled:
        if recycled not in existing:
            return recycled
        clear_recycled_episode(task_dir)
    scan_next = (existing[-1] + 1) if existing else 1
    if batch_index > 0:
        return max(batch_index, scan_next)
    return scan_next


def episode_video_files(task_dir: str, episode_index: int) -> dict:
    """episode N 的视频文件组 {image_key: 文件路径}。

    key 可跨 episode 改名（同一物理相机不同会话槽名不同），以本 episode
    实际存在的文件为准，不按任务级 video_extensions 推断。
    """
    c, f = episode_chunk_file(episode_index)
    stem = pooled_file_stem(f, "")
    out = {}
    vroot = os.path.join(task_dir, "videos", chunk_dir(c))
    if os.path.isdir(vroot):
        for key in sorted(os.listdir(vroot)):
            kd = os.path.join(vroot, key)
            if not os.path.isdir(kd):
                continue
            for fn in sorted(os.listdir(kd)):
                if fn.startswith(stem + "."):
                    out[key] = os.path.join(kd, fn)
    return out


def episode_size_mb(task_dir: str, episode_index: int) -> float:
    """episode N 的文件组总大小（MB；视频 + data parquet）。"""
    total = 0
    for p in episode_video_files(task_dir, episode_index).values():
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
    dp = pooled_data_parquet_path(task_dir, episode_index)
    if os.path.isfile(dp):
        try:
            total += os.path.getsize(dp)
        except OSError:
            pass
    return total / (1024.0 * 1024.0)


def episode_row(task_dir: str, episode_index: int) -> dict:
    """episode N 的元数据行；不存在返回 {}。

    新布局：每段一个文件（pooled_episodes_path 直读）。
    旧布局回退：v1.1.0 分片（每 chunk 一个 ≤1000 行文件，文件名 =
    chunk 号）里按 episode_index 列过滤。
    """
    import pyarrow.parquet as pq
    p = pooled_episodes_path(task_dir, episode_index)
    if os.path.isfile(p):
        try:
            for r in pq.read_table(p).to_pylist():
                if r.get("episode_index") == episode_index:
                    return r
        except Exception:
            pass
    # 旧分片回退（分片名 = chunk 号 file-{c:03d}，v1.1.2 起与每段文件
    # episode-{f:03d} 不再重名）
    c, _ = episode_chunk_file(episode_index)
    legacy = _legacy_episodes_shard_path(task_dir, c)
    if legacy != p and os.path.isfile(legacy):
        try:
            for r in pq.read_table(legacy).to_pylist():
                if r.get("episode_index") == episode_index:
                    return r
        except Exception:
            pass
    return {}


def episode_refs(base_dir: str) -> list:
    """扫描全部池化任务的 episode 列表（读侧唯一枚举入口）。

    Returns:
        [{"task", "task_dir", "episode_index", "name"}, ...]
        name 后缀 = episode 文件号（0 基，与本地 episode-NNN 对齐）；按名称倒序
        （最新任务/最新序号在前）。旧格式会话目录不再列出
        （v1.1.0 读侧只认池化布局）。
    """
    out = []
    if not os.path.isdir(base_dir):
        return out
    for task in sorted(os.listdir(base_dir)):
        task_dir = os.path.join(base_dir, task)
        if not os.path.isdir(task_dir) or task.startswith((".", "_")):
            continue
        if detect_session_format(task_dir) != "pooled":
            continue
        for n in list_task_episodes(task_dir):
            out.append({"task": task, "task_dir": task_dir,
                        "episode_index": n,
                        "name": f"{task}_ep{episode_file_suffix(n):06d}"})
    return sorted(out, key=lambda s: s["name"], reverse=True)


# ═══════════════════════════════════════════════════════
#  任务级统计（v1.1.1：stats.json 自含累加器，无 .stats_state 边车）
# ═══════════════════════════════════════════════════════

def stat_block_to_sum(blk: dict) -> dict:
    """统计块归一为 sum-form {count, sum, sum_sq, min, max}。

    兼容两种入参：writer/迁移产出的 sum-form；stats.json 的 json-form
    {count, mean, std, min, max}（按 mean/std/count 反推 sum/sum_sq——
    这是 stats.json 能当累加器的基础）。
    """
    import numpy as np
    n = int(blk.get("count", 0) or 0)
    if "sum" in blk:
        s = np.asarray(blk.get("sum", []), dtype=np.float64)
        sq = np.asarray(blk.get("sum_sq", []), dtype=np.float64)
        dim = len(s)
    elif "mean" in blk:
        m = np.asarray(blk.get("mean", []), dtype=np.float64)
        sd = np.asarray(blk.get("std", []), dtype=np.float64)
        dim = len(m)
        s = m * n
        sq = (sd * sd + m * m) * n
    else:
        return {"count": 0, "sum": [], "sum_sq": [], "min": [], "max": []}
    mn = blk.get("min")
    mx = blk.get("max")
    if mn is None or len(mn) != dim:
        mn = [float("inf")] * dim
    if mx is None or len(mx) != dim:
        mx = [float("-inf")] * dim
    return {"count": n, "sum": s.tolist(), "sum_sq": sq.tolist(),
            "min": [float(v) for v in mn], "max": [float(v) for v in mx]}


def merge_stat_block(acc: dict, key: str, blk: dict, dim: int) -> None:
    """把新统计块合并进累加器（sum-form；count<=0 跳过）。

    blk 可为 sum-form（writer 的 np 数组块/迁移块）或 json-form；
    acc 中旧块按 sum-form 读取，无则从零起。
    """
    if not blk or int(blk.get("count", 0) or 0) <= 0:
        return
    import numpy as np
    new = stat_block_to_sum(blk)
    n1 = int(new["count"])
    old = acc.get(key)
    if isinstance(old, dict) and int(old.get("count", 0) or 0) > 0:
        old = stat_block_to_sum(old)
        n0 = int(old["count"])
        s = (np.asarray(old["sum"], dtype=np.float64)
             + np.asarray(new["sum"], dtype=np.float64))
        sq = (np.asarray(old["sum_sq"], dtype=np.float64)
              + np.asarray(new["sum_sq"], dtype=np.float64))
        mn = np.minimum(np.asarray(old["min"], dtype=np.float64),
                        np.asarray(new["min"], dtype=np.float64))
        mx = np.maximum(np.asarray(old["max"], dtype=np.float64),
                        np.asarray(new["max"], dtype=np.float64))
    else:
        n0 = 0
        s = np.asarray(new["sum"], dtype=np.float64)
        sq = np.asarray(new["sum_sq"], dtype=np.float64)
        mn = np.asarray(new["min"], dtype=np.float64)
        mx = np.asarray(new["max"], dtype=np.float64)
    acc[key] = {"count": n0 + n1, "sum": s.tolist(), "sum_sq": sq.tolist(),
                "min": mn.tolist(), "max": mx.tolist()}


def acc_to_stats_json(acc: dict) -> dict:
    """sum-form 累加器 → stats.json 内容：每块 {count, mean, std, min, max}。

    count=0 的块写占位（mean 0/std 1，与旧格式空块语义一致）；
    action 恒为占位块（count 0，对外只表示列形状，不参与统计）。
    """
    import numpy as np
    stats: dict = {}
    for key, blk in acc.items():
        if not isinstance(blk, dict):
            continue
        sblk = stat_block_to_sum(blk)
        n = int(sblk["count"])
        dim = len(sblk["sum"])
        if n <= 0 or dim == 0:
            stats[key] = {"count": 0, "mean": [0.0] * dim, "std": [1.0] * dim,
                          "min": [0.0] * dim, "max": [0.0] * dim}
            continue
        s = np.asarray(sblk["sum"], dtype=np.float64)
        sq = np.asarray(sblk["sum_sq"], dtype=np.float64)
        mean = s / n
        var = np.maximum(sq / n - mean * mean, 0.0)
        stats[key] = {"count": n, "mean": mean.tolist(),
                      "std": np.sqrt(var).tolist(),
                      "min": sblk["min"], "max": sblk["max"]}
    stats["action"] = {"count": 0, "mean": [0.0], "std": [1.0],
                       "min": [0.0], "max": [0.0]}
    return stats


def load_stats_acc(task_dir: str) -> tuple:
    """读任务 stats.json 归一为 sum-form 累加器。

    返回 (acc, need_recalc)：文件缺失/损坏 → ({}, False)；任一块缺 count
    （旧格式，无法增量合并）→ ({}, True)，调用方应 recalc_stats 重建。
    """
    path = pooled_stats_path(task_dir)
    if not os.path.isfile(path):
        return {}, False
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}, False
    if not isinstance(raw, dict):
        return {}, False
    acc = {}
    for key, blk in raw.items():
        if not isinstance(blk, dict):
            continue
        if "count" not in blk:
            return {}, True          # 旧格式（无 count）→ 全量重算一次
        acc[key] = stat_block_to_sum(blk)
    return acc, False


def recalc_stats(task_dir: str) -> dict:
    """全量重扫任务 data parquet，重建 stats.json（自含 count），返回 sum-form acc。

    只统计实际存在的观测列：observation.*（帧级，逐行计数，与写入器同口径）
    与 observation.imu（样本级 6 轴）；observation.*hand_pose 为恒写占位
    零列（后处理回填），跳过不计（与写入器增量口径一致，且避免 std=0
    块毒化归一化）。旧格式 stats.json 迁移、删边车后的重建兜底都用它。
    无数据列时产出仅含 action 占位块。
    """
    import numpy as np
    import pyarrow.parquet as pq
    acc: dict = {}
    for ep in list_task_episodes(task_dir):
        p = pooled_data_parquet_path(task_dir, ep)
        if not os.path.isfile(p):
            continue
        try:
            tbl = pq.read_table(p)
        except Exception:
            continue
        for col in tbl.column_names:
            if not col.startswith("observation."):
                continue
            if col.endswith("hand_pose"):
                continue          # 恒写占位零列，后处理回填，不计统计
            try:
                if col == "observation.imu":
                    samples = [np.asarray(v, dtype=np.float64)
                               for v in tbl.column(col).to_pylist() if v]
                    if not samples:
                        continue
                    arr = np.concatenate([a.reshape(-1, 6) for a in samples])
                else:
                    vals = [np.asarray(v, dtype=np.float64)
                            for v in tbl.column(col).to_pylist()
                            if v is not None]
                    if not vals:
                        continue
                    arr = np.stack(vals)
            except (TypeError, ValueError):
                continue
            merge_stat_block(acc, col, {
                "count": int(arr.shape[0]),
                "sum": arr.sum(axis=0).tolist(),
                "sum_sq": (arr * arr).sum(axis=0).tolist(),
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
            }, arr.shape[1])
    # 原子写（indent=2：stats.json 是给人读的契约文件）
    path = pooled_stats_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(acc_to_stats_json(acc), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return acc


def _rmdir_empty(path: str) -> bool:
    """删除空目录（自底向上连带空父目录）；非空/不存在返回 False。"""
    if not os.path.isdir(path):
        return False
    try:
        os.rmdir(path)
        return True
    except OSError:
        return False


def delete_pooled_episode(task_dir: str, episode_index: int) -> bool:
    """彻底删除 episode N 的文件组（用户裁决：GUI 删除不走回收区）。

    - videos/<key>/episode-{f}.{ext} 各流文件
    - data/chunk-{c}/episode-{f}.parquet
    - episodes 每段文件直接删；旧分片回退=flock 读-改-写删行（与 writer
      同口径）
    统计不回退（stats/total_episodes 保持，与进度语义一致）。

    幂等：文件组已不存在也返回 True（视为已删）。
    """
    files = episode_video_files(task_dir, episode_index)
    for key, src in files.items():
        try:
            os.remove(src)
        except OSError:
            pass
        _rmdir_empty(os.path.dirname(src))
    dp = pooled_data_parquet_path(task_dir, episode_index)
    if os.path.isfile(dp):
        try:
            os.remove(dp)
        except OSError:
            pass
        _rmdir_empty(os.path.dirname(dp))

    # episodes 元数据：新布局=每段一个文件 → 直接删；旧分片回退=删行
    from core.egodata_writer import (_read_episode_rows, _episode_rows_table,
                                     _atomic_write_parquet)
    cidx, fidx = episode_chunk_file(episode_index)
    ep_file = pooled_episodes_path(task_dir, episode_index)
    handled = False
    if os.path.isfile(ep_file):
        try:
            rows = _read_episode_rows(ep_file)
        except Exception:
            rows = []
        if len(rows) == 1 and rows[0].get("episode_index") == episode_index:
            # 单行且正是 N → 直接删（含旧分片只剩该行的退化情形）
            try:
                os.remove(ep_file)
                handled = True
            except OSError:
                pass
            _rmdir_empty(os.path.dirname(ep_file))
    if not handled:
        # 旧分片（每 chunk 一个多行文件）→ flock 读-改-写删行
        legacy = _legacy_episodes_shard_path(task_dir, cidx)
        if os.path.isfile(legacy):
            try:
                import fcntl as _fcntl
            except ImportError:
                _fcntl = None
            lock_dir = os.path.join(task_dir, "meta", "episodes")
            os.makedirs(lock_dir, exist_ok=True)
            fh = open(os.path.join(lock_dir, ".lock"), "a+")
            try:
                if _fcntl is not None:
                    _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
                rows = [r for r in _read_episode_rows(legacy)
                        if r.get("episode_index") != episode_index]
                _atomic_write_parquet(_episode_rows_table(rows), legacy)
            finally:
                if _fcntl is not None:
                    _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)
                fh.close()
    return True


# ═══════════════════════════════════════════════════════════
#  关键点数据路径 —— 独立输出到 keypoints_output/，与 data/ 镜像结构
# ═══════════════════════════════════════════════════════════

from config import settings as _settings


def _keypoints_session_dir(session_dir: str) -> str:
    """将录制 session 路径映射到 keypoints_output 下的镜像目录。

    data/recordings/Test005/Test005_000024
    → keypoints_output/Test005/Test005_000024
    """
    recordings_root = os.path.normpath(_settings.RECORDING_DIR)
    sd = os.path.normpath(session_dir)
    if sd.startswith(recordings_root + os.sep):
        rel = os.path.relpath(sd, recordings_root)
    else:
        parts = sd.replace(os.sep, "/").rstrip("/").split("/")
        rel = os.path.join(parts[-2], parts[-1]) if len(parts) >= 2 else parts[-1]
    return os.path.join(_settings.KEYPOINTS_OUTPUT_DIR, rel)


def episode_keypoints_dir(task_dir: str, episode_index: int) -> str:
    """池化 episode 的 keypoints 镜像目录。

    data/recordings/<task> + N → keypoints_output/<task>/episode_{N:06d}
    """
    task = os.path.basename(os.path.normpath(task_dir))
    return os.path.join(_settings.KEYPOINTS_OUTPUT_DIR, task,
                        f"episode_{episode_index:06d}")


def episode_keypoints_video_dir(task_dir: str, episode_index: int) -> str:
    """keypoints_output/<task>/episode_{N:06d}/videos/"""
    return os.path.join(episode_keypoints_dir(task_dir, episode_index), "videos")


def episode_hand_kpts_parquet_path(task_dir: str, episode_index: int) -> str:
    """keypoints_output/<task>/episode_{N:06d}/hand_pose/chunk-000.parquet"""
    return os.path.join(episode_keypoints_dir(task_dir, episode_index),
                        "hand_pose", f"{chunk_dir(0)}.parquet")


def episode_auto_labels_parquet_path(task_dir: str, episode_index: int) -> str:
    """keypoints_output/<task>/episode_{N:06d}/auto_labels/auto_labels.parquet"""
    return os.path.join(episode_keypoints_dir(task_dir, episode_index),
                        "auto_labels", "auto_labels.parquet")


def episode_hand_3d_parquet_path(task_dir: str, episode_index: int) -> str:
    """keypoints_output/<task>/episode_{N:06d}/hand_pose_3d/chunk-000.parquet"""
    return os.path.join(episode_keypoints_dir(task_dir, episode_index),
                        "hand_pose_3d", f"{chunk_dir(0)}.parquet")


def keypoints_video_dir(session_dir: str) -> str:
    """keypoints_output/<project>/<session>/videos/"""
    return os.path.join(_keypoints_session_dir(session_dir), "videos")


def hand_kpts_parquet_path(session_dir: str) -> str:
    """keypoints_output/<project>/<session>/hand_pose/chunk-000.parquet"""
    return os.path.join(_keypoints_session_dir(session_dir), "hand_pose",
                        f"{chunk_dir(0)}.parquet")


def auto_labels_parquet_path(session_dir: str) -> str:
    """keypoints_output/<project>/<session>/auto_labels/auto_labels.parquet"""
    return os.path.join(_keypoints_session_dir(session_dir), "auto_labels",
                        "auto_labels.parquet")


def hand_3d_parquet_path(session_dir: str) -> str:
    """keypoints_output/<project>/<session>/hand_pose_3d/chunk-000.parquet"""
    return os.path.join(_keypoints_session_dir(session_dir), "hand_pose_3d",
                        f"{chunk_dir(0)}.parquet")


# ── 回退路径 1: session 目录内的 keypoints/ ──────────────

def _session_kpts_hand_kpts_path(session_dir: str) -> str:
    return os.path.join(session_dir, "keypoints", "hand_pose",
                        f"{chunk_dir(0)}.parquet")

def _session_kpts_hand_3d_path(session_dir: str) -> str:
    return os.path.join(session_dir, "keypoints", "hand_pose_3d",
                        f"{chunk_dir(0)}.parquet")

def _session_kpts_auto_labels_path(session_dir: str) -> str:
    return os.path.join(session_dir, "keypoints", "auto_labels",
                        "auto_labels.parquet")

# ── 回退路径 2: session 目录内的 annotations/（最旧版兼容）──

def _legacy_hand_kpts_path(session_dir: str) -> str:
    return os.path.join(session_dir, "annotations", "hand_pose",
                        f"{chunk_dir(0)}.parquet")

def _legacy_hand_3d_path(session_dir: str) -> str:
    return os.path.join(session_dir, "annotations", "hand_pose_3d",
                        f"{chunk_dir(0)}.parquet")

def _legacy_auto_labels_path(session_dir: str) -> str:
    return os.path.join(session_dir, "annotations", "mmpose", "auto_labels.parquet")


def list_all_sessions(base_dir: str) -> list:
    """扫描录制根目录，返回全部池化 episode（v1.1.0 读侧唯一枚举入口）。

    base_dir 是录制根目录（如 RECORDING_DIR）。
    旧格式会话目录不再列出——迁移后只认任务级池化布局。

    返回: [(task_dir, episode_index, display_name), ...] 按名称倒序，
    display_name 后缀 = episode 文件号（0 基，与本地 episode-NNN 对齐）。
    """
    return [(r["task_dir"], r["episode_index"], r["name"])
            for r in episode_refs(base_dir)]


# ═══════════════════════════════════════════════════════════
#  EgoData 格式路径工具
# ═══════════════════════════════════════════════════════════

def episode_dirname(episode_index: int = 1, task_name: str = "",
                    digits: int = 6) -> str:
    """EgoData episode 目录名。

    有 task_name 时: Chew_gum_000001
    无 task_name 时: episode_000001（兼容旧命名）
    """
    from config import settings
    d = digits if digits else settings.EPISODE_DIGITS
    if task_name:
        prefix = _sanitize_tag(task_name)
    else:
        prefix = settings.EPISODE_PREFIX
    return f"{prefix}_{episode_index:0{d}d}"


def next_episode_index(base_dir: str, task_name: str = "",
                       batch_index: int = 0) -> int:
    """返回下一个 episode 索引。

    优先取任务进度序号 batch_index（录制完成次数 + 1，与本地文件是否
    删除无关——上传后自动删除不会让序号回退）；同时至少大于目录扫描出
    的最大索引，防止与现存目录重名（删除失败/人工留存等场景）。
    batch_index <= 0 时退化为纯目录扫描（旧行为）。

    兼容新命名 (<tag>_NNNNNN) 和旧命名 (episode_NNNNNN)。
    """
    import re
    from config import settings

    tag = task_tag(task_name) if task_name else ""

    # 构建匹配模式：新格式 (task_tag_NNN) + 旧格式 (episode_NNN) 兼容
    patterns = []
    if tag:
        patterns.append(re.compile(rf"^{re.escape(tag)}_(\d+)$"))
    patterns.append(re.compile(rf"^{settings.EPISODE_PREFIX}_(\d+)$"))

    max_idx = 0
    search_dirs = []
    if tag:
        tagged = os.path.join(base_dir, tag)
        if os.path.isdir(tagged):
            search_dirs.append(tagged)
    search_dirs.append(base_dir)

    for sd in search_dirs:
        if not os.path.isdir(sd):
            continue
        for entry in os.listdir(sd):
            for pat in patterns:
                m = pat.match(entry)
                if m and os.path.isdir(os.path.join(sd, entry)):
                    idx = int(m.group(1))
                    if idx > max_idx:
                        max_idx = idx
    scan_next = max_idx + 1
    if batch_index > 0:
        return max(batch_index, scan_next)
    return scan_next


# ── EgoData 目录内路径 ──────────────────────────────

def egodata_video_path(episode_dir: str, camera_name: str) -> str:
    """videos/<base_cam>/chunk-0000/<camera_name>.mp4

    _aux 后缀的摄像头归入其主摄像头文件夹:
      stereo_left_aux → videos/stereo_left/chunk-0000/stereo_left_aux.mp4
    """
    base_cam = camera_name[:-4] if camera_name.endswith("_aux") else camera_name
    return os.path.join(episode_dir, "videos", base_cam, "chunk-0000",
                        f"{camera_name}.mp4")


def egodata_video_dir(episode_dir: str, camera_name: str) -> str:
    """videos/<base_cam>/chunk-0000/ 目录路径。"""
    base_cam = camera_name[:-4] if camera_name.endswith("_aux") else camera_name
    return os.path.join(episode_dir, "videos", base_cam, "chunk-0000")


def egodata_depth_path(episode_dir: str, depth_name: str, frame_index: int) -> str:
    """episode_dir/depth/head_depth/000001.png（PNG 16-bit grayscale，uint16 毫米）

    旧版 png16 格式（v1.0.13 及以前），保留给历史会话读取回退。
    """
    return os.path.join(episode_dir, "depth", depth_name,
                        f"{frame_index:06d}.png")


def egodata_depth_video_path(episode_dir: str, depth_name: str) -> str:
    """episode_dir/depth/<slot>/<slot>.mp4（12-bit 灰度 HEVC，v1.1.2 起）

    单视频轨 hevc (Rext) gray12le，12-bit 对数深度码（见
    core/depth_codec.py）。旧格式回退 <slot>.mkv（v1.0.14 双流
    FFV1）；哪个存在返回哪个，都无则按 mp4 返回。
    """
    base = os.path.join(episode_dir, "depth", depth_name, depth_name)
    for ext in (".mp4", ".mkv"):
        path = base + ext
        if os.path.isfile(path):
            return path
    return base + ".mp4"


def egodata_image_path(episode_dir: str, camera_name: str, frame_index: int) -> str:
    """episode_dir/images/head_left_rgb/000001.jpg"""
    return os.path.join(episode_dir, "images", camera_name,
                        f"{frame_index:06d}.jpg")


def egodata_calibration_path(episode_dir: str,
                              calib_name: str = "head_stereo") -> str:
    """episode_dir/calibration/head_stereo.json"""
    return os.path.join(episode_dir, "calibration", f"{calib_name}.json")


def egodata_metadata_path(episode_dir: str) -> str:
    """episode_dir/metadata.json"""
    return os.path.join(episode_dir, "metadata.json")


def egodata_timestamps_path(episode_dir: str) -> str:
    """episode_dir/timestamps.json"""
    return os.path.join(episode_dir, "timestamps.json")


def egodata_sensor_data_dir(episode_dir: str, sensor_name: str) -> str:
    """data/<sensor_name>/chunk-0000/ 目录路径。"""
    return os.path.join(episode_dir, "data", sensor_name, "chunk-0000")


def egodata_sensor_parquet_path(episode_dir: str, sensor_name: str) -> str:
    """data/<sensor_name>/chunk-0000/chunk_000000.parquet"""
    return os.path.join(episode_dir, "data", sensor_name, "chunk-0000",
                        "chunk_000000.parquet")


# ── 格式检测 ────────────────────────────────────────

def detect_session_format(session_dir: str) -> str:
    """检测会话目录格式。

    Returns:
        "pooled"     — v1.1.0 任务池化布局（meta/info.json format 字段）
        "egodata"    — 有 metadata.json
        "lerobot_v3" — 有 meta/info.json（旧 LeRobot v3 会话目录）
        "unknown"    — 无法识别
    """
    info_path = os.path.join(session_dir, "meta", "info.json")
    if os.path.isfile(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            if isinstance(info, dict) and \
                    info.get("format") == "pooled_episodes_v1":
                return "pooled"
        except (OSError, json.JSONDecodeError):
            pass
        return "lerobot_v3"
    if os.path.isfile(os.path.join(session_dir, "metadata.json")):
        return "egodata"
    return "unknown"


# ── 扩展 list_all_sessions 兼容 EgoData ──────────────

def list_all_egodata_sessions(base_dir: str) -> list:
    """扫描全部 EgoData episode 目录。

    目录结构: base_dir/<task_tag>/episode_000001/

    Returns: [(ep_path, ep_name, tag_name), ...] 按名称倒序
    """
    import re
    from config import settings

    sessions = []
    if not os.path.isdir(base_dir):
        return sessions

    d = settings.EPISODE_DIGITS
    # 匹配: 任意前缀_NNNNNN (新任务命名) 或 episode_NNNNNN (旧兼容)
    pattern = re.compile(rf"^(\w+)_(\d{{{d}}})$")

    for entry in sorted(os.listdir(base_dir)):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        # EgoData: 本身是 episode 目录（有 metadata.json）
        if pattern.match(entry) and os.path.isfile(
            os.path.join(entry_path, "metadata.json")):
            sessions.append((entry_path, entry, entry.split("_")[0]))
        else:
            # 任务子文件夹
            for sub in sorted(os.listdir(entry_path)):
                sub_path = os.path.join(entry_path, sub)
                if pattern.match(sub) and os.path.isfile(
                    os.path.join(sub_path, "metadata.json")):
                    sessions.append((sub_path, sub, entry))

    return sorted(sessions, key=lambda s: s[1], reverse=True)

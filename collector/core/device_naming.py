"""
设备命名辅助 —— 型号短名、槽名清洗、曝光基线归一化与槽名分配（零 Qt 依赖）。

供主窗口设备接入链路（D435 槽名分配 / 曝光对话框基线）与离线测试共用同一套口径：
  - realsense_short      型号名 → 短名（"RealSense D405" → "D405"）
  - slot_base            用户命名 → 槽名前缀（文件系统安全 + "depth" 后缀约定不破坏）
  - normalize_original   各种来源的「最一开始」曝光基线 → (auto, value) 二元组
  - allocate_slot_names  RealSense RGB/深度槽名对分配（同前缀多台编号追加）
"""

from __future__ import annotations

import re


def realsense_short(model_name: str) -> str:
    """型号名 → 短名（"RealSense D405" → "D405"；无型号段回落 "RealSense"）。"""
    m = re.search(r"D\d{3}", model_name or "")
    return m.group(0) if m else "RealSense"


def slot_base(name: str, fallback: str) -> str:
    """用户命名 → 槽名前缀（仅保留字母/数字/下划线/中文；空或全非法回落）。

    GUI 命名可含空格、emoji 等，槽名要作为文件夹名与 info.json 键，
    须保证文件系统安全且不破坏 "depth" 后缀约定。
    """
    clean = re.sub(r"[^0-9A-Za-z_一-鿿]+", "_",
                   (name or "").strip()).strip("_")
    return clean or fallback


def normalize_original(orig) -> tuple | None:
    """「最一开始」曝光基线归一化为 (auto, value) 元组；无/坏数据返回 None。

    settings.device_original 返回 dict，worker.original_exposure 返回
    tuple，S80M entry 存 tuple —— 统一在此收口成对话框要的二元组。
    """
    if isinstance(orig, dict) and "auto" in orig and "value" in orig:
        return (bool(orig["auto"]), float(orig["value"]))
    if (isinstance(orig, (tuple, list)) and len(orig) == 2
            and orig[0] is not None and orig[1] is not None):
        return (bool(orig[0]), float(orig[1]))
    return None


def allocate_slot_names(user_name: str, model_name: str,
                        occupied_rgb_slots: list[str]) -> tuple[str, str]:
    """分配 RealSense RGB/深度槽名对，同前缀多台编号追加。

    RGB 槽 = 前缀 + "_rgb"；深度槽 = 前缀本身若已以 "_depth" 结尾
    （用户命名 "D405_depth" → depth 文件夹恰为用户名原样），否则补
    "_depth"（保持回放/录制链路 "depth" 后缀约定）。同前缀多台编号
    追加在槽名之后（d435_depth_2 经典式）。首台未命名 D435 仍得
    d435_rgb/d435_depth 经典名。

    Args:
        user_name:          GUI 用户命名（device_names.json，可空回落型号）
        model_name:         RealSense 显示名（如 "RealSense D405"）
        occupied_rgb_slots: 已占用 RGB 槽名列表（用于同前缀计数编号）
    """
    base = slot_base(user_name, realsense_short(model_name).lower())
    n = sum(1 for s in occupied_rgb_slots if s.startswith(f"{base}_rgb"))
    rgb_slot = f"{base}_rgb" if n == 0 else f"{base}_rgb_{n + 1}"
    depth_slot = base if base.endswith("_depth") else f"{base}_depth"
    if n > 0:
        depth_slot = f"{depth_slot}_{n + 1}"
    return rgb_slot, depth_slot

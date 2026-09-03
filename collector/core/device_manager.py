"""
统一设备 worker 注册表 + 面板开关分派 + 录制元数据口径（纯 Python，
零 Qt 依赖）。

具体开启/关闭动作（建槽、弹窗、widget 接线）留在 UI 侧回调，本模块
只做核心口径：
  - DeviceManager    注册表容器：条目骨架构造 + 查询 + 元数据/抽帧状态
  - dispatch_toggle  面板开关分派（录制锁双保险 + kind 路由）
  - build_device_meta / reset_s80m_record_state / teardown_all
                     注册表级纯口径（离线测试直接注入假条目即可）

注册表条目形状（与旧主窗口 _workers 口径一致）：
  kind ∈ {"uvc", "d435", "s80m", "data_ble", "ble"}
  d435/s80m 条目由各自 manager 的 new_entry 构造
  （core.d435_manager / core.s80m_manager），uvc/ble/glove 骨架由本类构造。
"""

from __future__ import annotations

from config import settings


class DeviceManager:
    """统一设备 worker 注册表（多路并发核心）。

    entries: dev_key → entry dict（主窗口 _workers 直接引用同一 dict，
    离线测试注入假条目沿用同一形状）。UI 侧负责把条目塞进注册表，
    本类只提供骨架构造、查询与录制口径。
    """

    def __init__(self):
        self.entries: dict = {}

    # ── 条目骨架 ──
    @staticmethod
    def uvc_entry(slots, label) -> dict:
        """UVC 条目（无专属字段，槽位即状态）。"""
        return {"kind": "uvc", "slots": list(slots), "label": label}

    @staticmethod
    def ble_entry(slot: str, label: str) -> dict:
        """无数据蓝牙（耳机类）占位条目。"""
        return {"kind": "ble", "slots": [slot], "label": label}

    @staticmethod
    def glove_entry(slot: str, role: str, glove_widget, label: str) -> dict:
        """数据手套条目：传感器列名 + 画面 widget（close 时停 BLE/撤画面）。"""
        return {"kind": "data_ble", "slots": [slot],
                "sensor_column": role, "glove": glove_widget,
                "label": label}

    # ── 查询 ──
    def get(self, dev_key: str) -> dict | None:
        return self.entries.get(dev_key)

    def has_kind(self, kind: str) -> bool:
        return any(e["kind"] == kind for e in self.entries.values())

    def has_serial(self, kind: str, serial: str) -> bool:
        """按 serial 查重（d435 同机重复开关检测）。"""
        return any(e["kind"] == kind and e.get("serial") == serial
                   for e in self.entries.values())

    # ── 录制口径 ──
    def build_device_meta(self) -> list:
        """按注册表构建录制设备信息（写入 meta 的 devices 段）。"""
        return build_device_meta(self.entries)

    def reset_s80m_record_state(self):
        """录制起止时重置 50→30 抽帧状态（桶号 + 待写 IMU 缓冲）。"""
        reset_s80m_record_state(self.entries)

    def teardown_all(self, close_fns: dict) -> None:
        """关闭全部已开启设备（遍历注册表分类清理，未知 kind 兜底弹出）。"""
        teardown_all(self.entries, close_fns)


def build_device_meta(entries: dict) -> list:
    """按注册表构建录制设备信息（写入 meta 的 devices 段）。

    手套：无画面槽位，附 parquet 传感器列名；
    无数据蓝牙（ble）不进 devices 段。
    """
    meta = []
    for key, e in entries.items():
        if e["kind"] == "data_ble":
            meta.append({
                "key": key, "kind": e["kind"],
                "name": settings.device_name(key) or e.get("label", ""),
                "slots": [],
                "sensor_column": e.get("sensor_column", "")})
            continue
        if e["kind"] not in ("uvc", "d435", "s80m"):
            continue
        d = {"key": key, "kind": e["kind"],
             "name": settings.device_name(key) or e.get("label", ""),
             "slots": list(e.get("slots", []))}
        if e.get("serial"):
            d["serial"] = str(e["serial"])
        meta.append(d)
    return meta


def reset_s80m_record_state(entries: dict) -> None:
    """录制起止时重置 50→30 抽帧状态（桶号 + 待写 IMU 缓冲 + 空桶统计）。"""
    for e in entries.values():
        if e["kind"] == "s80m":
            e["last_bucket"] = {}
            e["pending_imu"] = []
            e["drop_watch"] = {"last_mono_ns": None, "dropped": 0,
                               "elapsed": 0, "alerted": False}


def teardown_all(entries: dict, close_fns: dict) -> None:
    """关闭全部已开启设备（遍历注册表分类清理，未知 kind 兜底弹出）。

    s80m/d435 的关闭回调会自行 unregister（pop），故遍历键副本。
    """
    for key in list(entries):
        entry = entries.get(key)
        if not entry:
            continue
        fn = (close_fns or {}).get(entry["kind"])
        if fn:
            fn(key)
        else:
            entries.pop(key, None)
    entries.clear()


def dispatch_toggle(dev, on: bool, is_recording: bool,
                    open_fns: dict, close_fns: dict) -> bool:
    """面板开关分派口径（录制锁双保险 + kind 路由）。

    open_fns / close_fns: kind → 可调用（具体动作在 UI 侧，
    open 接收 dev 并返回是否成功，close 接收 dev_key）。
    返回 opened：打开成功 True；录制中/打开失败/缺回调 False
    （UI 侧按 on 区分「回退勾选」与「关闭路径」）。
    """
    if on:
        if is_recording:
            return False   # 面板已锁死，双保险：UI 侧回退勾选
        fn = (open_fns or {}).get(dev.kind)
        return bool(fn(dev)) if fn else False
    fn = (close_fns or {}).get(dev.kind)
    if fn:
        fn(dev.key)
    return False

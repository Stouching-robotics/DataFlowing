"""
每设备曝光控制 —— ☀ 对话框参数解析与下发/持久化口径（零 Qt 依赖）。

供主窗口曝光对话框入口与离线测试共用同一套口径：
  - exposure_dialog_params  按设备类型解析对话框上下文（量程/自动/当前值/
                            恢复默认基线 + 首见基线落盘锁定）；UVC 返回 None
  - apply_exposure          按设备类型下发并持久化（下次开启自动应用）
"""

from __future__ import annotations

from config import settings
from core.device_naming import normalize_original


def exposure_dialog_params(kind: str, entry: dict,
                           dev_key: str) -> dict | None:
    """☀ 曝光对话框参数解析（对话框只做 0..1000 刻度映射）。

    量程/单位语义：
      D435 → µs，流启动后读真实 rs.option.exposure 量程
      S80M → SDK 曝光 1.0~885.0（与 yaml stereo_init_exposure 同单位）
    UVC 无曝光入口（固定自动曝光，保证正常亮度 + 30fps）→ 返回 None。

    「恢复默认」基线（original）：优先持久化的首见基线
    （settings.device_original，首次开启设备时锁定），无则用
    worker 本次开流读回的硬件状态，并在对话框弹出时落盘锁定
    （之后 worker 读回值不再覆盖）。
    """
    if kind == "uvc":
        return None
    label = entry["label"]
    if kind == "d435":
        rng, auto, value = entry["worker"].exposure_info()
        if rng is None:
            rng = (1.0, 66000.0)   # 流未启动兜底量程（µs）
        if auto is None:
            auto = True
        if value is None:
            value = rng[1] / 2.0
        orig = settings.device_original(dev_key)
        if orig is None:
            orig = entry["worker"].original_exposure()
        decimals = 0
    elif kind == "s80m":
        exp = settings.device_exposure(dev_key)
        auto = True if exp is None else exp["auto"]
        value = 400.0 if exp is None else exp["value"]
        orig = settings.device_original(dev_key)
        if orig is None:
            orig = entry.get("original_exp")
        rng = (1.0, 885.0)
        decimals = 1
    else:
        return None

    orig = normalize_original(orig)
    if orig is not None:
        settings.ensure_device_original(dev_key, orig[0], orig[1])
    return {"label": label, "rng": rng, "value": float(value),
            "auto": bool(auto), "decimals": decimals, "original": orig}


def apply_exposure(dev_key: str, entry: dict, auto: bool, value: float,
                   send_s80m=None) -> None:
    """按设备类型下发曝光并持久化（下次开启自动应用）。

    UVC 无曝光入口（固定自动曝光），不会进入此函数。
    S80M 的 stdin 行协议由 send_s80m(entry, auto, value) 回调提供
    （见 core.s80m_manager.S80MDeviceManager.send_exposure）。
    """
    kind = entry["kind"]
    if kind == "d435":
        entry["worker"].set_exposure(auto, value)
    elif kind == "s80m":
        if send_s80m is not None:
            send_s80m(entry, auto, value)
    else:
        return
    settings.save_device_exposure(dev_key, auto, value)

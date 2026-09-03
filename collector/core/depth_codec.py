"""深度 12-bit 灰度视频编码（v1.1.2：gray12le HEVC MP4，lerobot v3 同款）。

存储格式: 单视频轨 hevc (Rext) gray12le（12-bit 单平面灰度）——
Y 平面 = 对数深度码，量化域 [DEPTH_MIN_MM, DEPTH_MAX_MM] 对数映射到
[0, DEPTH_QMAX]。播放器直接打开显示为灰度视频；读取端 gray12le 解码
后反量化回 uint16 毫米。文件为 .mp4（hvc1 容器标记）。

显示对齐: 所有实时/回放深度画面一律走 depth_to_heatmap_bgr()
（毫米 → 码值 → JET），与存储数据同构——颜色与码值一一对应，
不再有 near/far 线性色标（log 域色标对手部近场场景区分度更好）。

精度账（实测，848×480 真实深度）:
  量化自身: 每 LSB 相对误差 ln(50)/4095 ≈ 0.096%（0.1mm@0.1m，
            1mm@1m），mean≈0.34mm——低于 D435 传感器噪声 ±3-5mm
  编码 qp=6: 码误差 mean≈2.4/4095（近场深度误差 mean≈1.2mm）
  体积:     ~42.6KB/帧（vs FFV1 无损 77KB/帧、PNG16 168KB/帧）
  编码速度: x265 preset=fast 实测 44fps@1280×800 纯噪声下界（真实
            深度更快；medium 仅 35fps，负载下会反压写线程）

编码注意:
  - 必须不带 -profile:v（main12 与 4:0:0 不兼容，去掉后 x265 自动
    落 hevc Rext 单平面）
  - 必须 range=full（缺省有限域换算会把 600→402 之类毁掉码值）
  - CRF 模式禁用（x265 感知优化把深度噪声当纹理抹掉，近场误差
    3-5×）；CQP qp=6 近无损
  - nvenc 不支持 12-bit（p016 族是 10-bit 数据装 16-bit 容器）
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# ── 量化域常量 ─────────────────────────────────────────
DEPTH_MIN_MM = 100.0    # 量化域下限（毫米）；低于此值裁剪为码 0。
                        # D405 最近工作距 ~70mm，0.1m 覆盖手部工作域
DEPTH_MAX_MM = 5000.0   # 量化域上限（毫米）；D435 有效量程 ~4m 留余量
DEPTH_QMAX = 4095       # 12-bit 码值上限
DEPTH_QP = 6            # x265 近无损 CQP 档（CRF 模式禁用，见模块注释）

_LOG_LO = math.log(DEPTH_MIN_MM)
_LOG_SPAN = math.log(DEPTH_MAX_MM) - _LOG_LO
_LOG_STEP = DEPTH_QMAX / _LOG_SPAN   # 每 LSB 的对数步长


_QUANTIZE_LUT: np.ndarray | None = None    # 65536 项 mm→码 LUT（惰性构建）


def _quantize_lut() -> np.ndarray:
    """mm 值 0..65535 → 12-bit 码值查表（与 float64 公式逐位一致）。

    写线程每帧 3.0ms 的 np.log（1280×800 float64）降到 ~0.4ms
    查表——2026-09-03 写线程时间轴漂移残余修复（preset fast 已把
    漂移 1.74s→0.71s，LUT 再省 ~2.6ms/拍）。
    """
    global _QUANTIZE_LUT
    if _QUANTIZE_LUT is None:
        lut = np.zeros(65536, dtype="<u2")
        mm = np.arange(1, 65536, dtype=np.float64)
        lut[1:] = np.clip(
            np.rint((np.log(mm) - _LOG_LO) * _LOG_STEP),
            0, DEPTH_QMAX).astype("<u2")
        _QUANTIZE_LUT = lut
    return _QUANTIZE_LUT


def quantize_depth(depth_mm: np.ndarray) -> np.ndarray:
    """uint16 毫米深度 → 12-bit 对数码（uint16 存储，值域 0..4095）。

    无效值（=0）映射为码 0；低于 DEPTH_MIN_MM 裁剪为码 0，
    高于 DEPTH_MAX_MM 饱和为 DEPTH_QMAX。uint16 输入走查表快路径
    （与 float64 公式逐位一致）；非 uint16 输入回退原公式。
    """
    mm = np.asarray(depth_mm)
    if mm.dtype == np.uint16:
        return np.take(_quantize_lut(), mm)
    mm = mm.astype(np.float64)
    valid = mm > 0
    if not valid.any():
        return np.zeros(mm.shape, dtype="<u2")
    codes = np.empty(mm.shape, dtype="<u2")
    codes[~valid] = 0
    codes[valid] = np.clip(
        np.rint((np.log(mm[valid]) - _LOG_LO) * _LOG_STEP),
        0, DEPTH_QMAX).astype("<u2")
    return codes


def dequantize_depth(codes: np.ndarray) -> np.ndarray:
    """12-bit 对数码 → uint16 毫米深度（反量化，四舍五入）。"""
    c = np.asarray(codes, dtype=np.float64)
    return np.rint(np.exp(_LOG_LO + c / _LOG_STEP)).astype("<u2")


def codes_to_heatmap_bgr(codes: np.ndarray) -> np.ndarray:
    """12-bit 深度码 → BGR 热力图（规范显示口径：码值 → JET）。

    所有实时/回放深度画面的统一颜色映射；码 0（无效/低于量化域下限）
    显示为 JET(0) 深蓝，不置黑（与旧 depth_to_heatmap 口径一致）。
    """
    # 注意：codes 是 uint16，直接 *255 会按 u2 回绕（码 >257 全错），
    # 必须先升 int32 再乘（实测 1799*255 回绕后 c8 从 112 错成 16）。
    c8 = ((np.clip(codes, 0, DEPTH_QMAX).astype(np.int32) * 255)
          // DEPTH_QMAX).astype(np.uint8)
    return cv2.applyColorMap(c8, cv2.COLORMAP_JET)


def depth_to_heatmap_bgr(depth_mm: np.ndarray) -> np.ndarray:
    """uint16 毫米深度 → BGR 热力图（量化 → 规范 JET，与存储同构）。"""
    return codes_to_heatmap_bgr(quantize_depth(depth_mm))


def depth_video_encoder_args() -> list:
    """深度视频编码参数（不含 -c:v）：gray12le → libx265 qp=6 近无损。

    必须不带 -profile:v（main12 与 4:0:0 不兼容，去掉后 x265 自动
    落 hevc Rext 单平面）。range=full 保 12-bit 码值不被有限域换算
    破坏（实测缺省会把 600→402）。preset=fast：实测 1280×800 纯噪声
    下界 44fps（medium 仅 35fps），机器全负载（双彩色流+GUI+USB）时
    编码器不跌破 30fps 反压写线程——2026-09-03 实机写线程时间轴
    漂移 1.7s/20s 的根因修复；精度代价实测 +0.26 码值（≈0.15mm@600mm），
    可忽略（转码对比见 session 验证）。
    """
    return ["-pix_fmt", "gray12le", "-tag:v", "hvc1",
            "-preset", "fast",
            "-x265-params", f"qp={DEPTH_QP}:range=full"]

"""
传感器数据渲染引擎 —— 5 种可视化模式。

所有渲染函数接收 (processed_data, max_signal, config, window_size)
返回 BGR 格式的 numpy 帧数组，由 UI 层转为 QPixmap 显示。
"""

import json
import os
import numpy as np
import cv2

from config import settings

# ── 常量 ──────────────────────────────────────────────
MATRIX_ROWS = 16
MATRIX_COLS = 16
VIRIDIS_LUT: np.ndarray = None

# ── 配置路径 ──────────────────────────────────────────
# 仿生手掌（左/右手）映射配置。原定义在 sensors.sensor_panel，
# 迁到这里：glove_widget / playback_dialog 需要路径但不应引入 bleak。
CONFIG_DIR = os.path.join(settings.BASE_DIR, "config", "sensors")
CONFIG_FILE = os.path.join(CONFIG_DIR, "hand_ble_config.json")
CONFIG_FILE_LEFT = os.path.join(CONFIG_DIR, "hand_ble_config_left.json")


def _get_viridis_lut() -> np.ndarray:
    """延迟初始化 Viridis 颜色查找表。"""
    global VIRIDIS_LUT
    if VIRIDIS_LUT is None:
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            lut[i, 0, 0] = i
        VIRIDIS_LUT = cv2.applyColorMap(lut, cv2.COLORMAP_VIRIDIS)
    return VIRIDIS_LUT


# ── HUD 绘制 ──────────────────────────────────────────

def _draw_hud(frame: np.ndarray, lines: list, color=(255, 255, 255)):
    """在帧左上角绘制半透明 HUD 信息栏。"""
    font, scale, thick, lh, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 22, 10
    max_w = (
        max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines)
        if lines else 0
    )
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (0, 0), (max_w + pad * 2, len(lines) * lh + pad * 2),
        (0, 0, 0), -1,
    )
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (pad, pad + lh - 5 + i * lh),
            font, scale, color, thick, cv2.LINE_AA,
        )


# ── 配置加载 ──────────────────────────────────────────

def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(default, dict):
                    default.update(loaded)
                    return default
                return loaded
        except Exception:
            pass
    return default


# ═══════════════════════════════════════════════════════
#  模式 1: 热力图
# ═══════════════════════════════════════════════════════

def render_heatmap(
    processed: np.ndarray,
    max_signal: float,
    config: dict,
    window_size: tuple,
    current_vmax: float,
    fps: float,
    noise_gate: int,
    dyn_ratio: float,
    spatial_on: bool,
) -> tuple:
    """返回 (frame, new_vmax)。"""
    ww, wh = window_size
    rows = config.get("rows", list(range(16)))
    cols = config.get("cols", list(range(16)))
    order = config.get("axis_order", "row_col")

    # 增强参数（来自 config，向后兼容）
    subpixel = config.get("subpixel", True)
    gamma = config.get("gamma", 1.0)
    blur_ksize = config.get("blur", 5)

    nr, nc = len(rows), len(cols)
    if nr == 0 or nc == 0:
        frame = np.zeros((wh, ww, 3), dtype=np.uint8)
        _draw_hud(frame, ["Press [M] to select Heatmap Rows/Cols!"], (0, 0, 255))
        return frame, current_vmax

    # 提取子矩阵
    sub = processed[np.ix_(rows, cols)]
    if order != "row_col":
        sub = sub.T

    # ── Subpixel 超分 ──────────────────────────────
    if subpixel and sub.shape[0] >= 2 and sub.shape[1] >= 2:
        H0, W0 = sub.shape
        shrink = 1.5
        padded = np.pad(sub, 1, mode='constant')
        top = padded[0:H0, 1:W0 + 1]
        bottom = padded[2:H0 + 2, 1:W0 + 1]
        left = padded[1:H0 + 1, 0:W0]
        right = padded[1:H0 + 1, 2:W0 + 2]
        sum_x = left + right + sub + 1e-6
        sum_y = top + bottom + sub + 1e-6
        dx = np.clip((right - left) / sum_x * shrink, -1, 1)
        dy = np.clip((bottom - top) / sum_y * shrink, -1, 1)
        super_res = np.zeros((H0 * 2, W0 * 2), dtype=np.float32)
        super_res[0::2, 0::2] = sub * np.maximum(0, 1.0 - dx - dy)
        super_res[0::2, 1::2] = sub * np.maximum(0, 1.0 + dx - dy)
        super_res[1::2, 0::2] = sub * np.maximum(0, 1.0 - dx + dy)
        super_res[1::2, 1::2] = sub * np.maximum(0, 1.0 + dx + dy)
        sub = super_res

    H, W = sub.shape
    scale_w = (ww - 200) / W
    scale_h = (wh - 120) / H
    scale = min(scale_w, scale_h)
    target_w = int(W * scale)
    target_h = int(H * scale)

    resized = cv2.resize(sub, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    if blur_ksize > 0:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        resized = cv2.GaussianBlur(resized, (k, k), 0)

    # 平滑 vmax
    vmax_cand = max(5000, np.percentile(resized, 99.7))
    new_vmax = current_vmax * 0.95 + vmax_cand * 0.05 if vmax_cand > current_vmax else vmax_cand

    # ── Gamma 校正 ────────────────────────────────
    norm = np.clip(resized / new_vmax, 0, 1) if new_vmax > 0 else resized
    if gamma != 1.0:
        norm = np.power(norm, gamma)
    img_8u = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(img_8u, cv2.COLORMAP_VIRIDIS)

    frame = np.zeros((wh, ww, 3), dtype=np.uint8)
    sx = (ww - target_w) // 2
    sy = (wh - target_h) // 2 + 20
    frame[sy:sy + target_h, sx:sx + target_w] = color

    _draw_hud(frame, [
        f"Heatmap | FPS:{fps:.1f} | Max:{int(max_signal)}",
        f"Gate:{noise_gate} | Dyn:{dyn_ratio:.2f} | Filter:{'ON' if spatial_on else 'OFF'}",
        f"SP:{'ON' if subpixel else 'OFF'} G:{gamma:.1f} B:{blur_ksize} | [M] Config",
    ])
    return frame, new_vmax


# ═══════════════════════════════════════════════════════
#  模式 2: 轨迹模式
# ═══════════════════════════════════════════════════════

_trace_canvas = None

def render_trace(
    processed: np.ndarray,
    max_signal: float,
    config: dict,
    window_size: tuple,
    current_vmax: float,
    fps: float,
    noise_gate: int,
    dyn_ratio: float,
    spatial_on: bool,
) -> tuple:
    """轨迹模式：叠加历史数据。"""
    global _trace_canvas
    ww, wh = window_size
    rows = config.get("rows", list(range(16)))
    cols = config.get("cols", list(range(16)))
    order = config.get("axis_order", "row_col")

    # 增强参数
    subpixel = config.get("subpixel", True)
    gamma = config.get("gamma", 1.0)
    blur_ksize = config.get("blur", 5)

    nr, nc = len(rows), len(cols)
    if nr == 0 or nc == 0:
        frame = np.zeros((wh, ww, 3), dtype=np.uint8)
        _draw_hud(frame, ["Press hotkey to config"], (0, 0, 255))
        return frame, current_vmax, False

    sub = processed[np.ix_(rows, cols)]
    if order != "row_col":
        sub = sub.T

    # ── Subpixel 超分 ──────────────────────────────
    if subpixel and sub.shape[0] >= 2 and sub.shape[1] >= 2:
        H0, W0 = sub.shape
        shrink = 1.5
        padded = np.pad(sub, 1, mode='constant')
        top = padded[0:H0, 1:W0 + 1]
        bottom = padded[2:H0 + 2, 1:W0 + 1]
        left = padded[1:H0 + 1, 0:W0]
        right = padded[1:H0 + 1, 2:W0 + 2]
        sum_x = left + right + sub + 1e-6
        sum_y = top + bottom + sub + 1e-6
        dx = np.clip((right - left) / sum_x * shrink, -1, 1)
        dy = np.clip((bottom - top) / sum_y * shrink, -1, 1)
        super_res = np.zeros((H0 * 2, W0 * 2), dtype=np.float32)
        super_res[0::2, 0::2] = sub * np.maximum(0, 1.0 - dx - dy)
        super_res[0::2, 1::2] = sub * np.maximum(0, 1.0 + dx - dy)
        super_res[1::2, 0::2] = sub * np.maximum(0, 1.0 - dx + dy)
        super_res[1::2, 1::2] = sub * np.maximum(0, 1.0 + dx + dy)
        sub = super_res

    if _trace_canvas is None or _trace_canvas.shape != sub.shape:
        _trace_canvas = np.zeros_like(sub)
    _trace_canvas = np.maximum(_trace_canvas, sub)
    display = _trace_canvas

    H, W = display.shape
    scale_w = (ww - 200) / W
    scale_h = (wh - 120) / H
    scale = min(scale_w, scale_h)
    tw, th = int(W * scale), int(H * scale)

    resized = cv2.resize(display, (tw, th), interpolation=cv2.INTER_CUBIC)
    if blur_ksize > 0:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        resized = cv2.GaussianBlur(resized, (k, k), 0)

    vmax_cand = max(5000, np.percentile(resized, 99.7))
    new_vmax = current_vmax * 0.95 + vmax_cand * 0.05 if vmax_cand > current_vmax else vmax_cand

    norm = np.clip(resized / new_vmax, 0, 1) if new_vmax > 0 else resized
    if gamma != 1.0:
        norm = np.power(norm, gamma)
    img_8u = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(img_8u, cv2.COLORMAP_VIRIDIS)

    frame = np.zeros((wh, ww, 3), dtype=np.uint8)
    sx, sy = (ww - tw) // 2, (wh - th) // 2 + 20
    frame[sy:sy + th, sx:sx + tw] = color

    _draw_hud(frame, [
        f"Trace | FPS:{fps:.1f} | Max:{int(max_signal)}",
        f"Gate:{noise_gate} | Dyn:{dyn_ratio:.2f} | Filter:{'ON' if spatial_on else 'OFF'}",
        f"SP:{'ON' if subpixel else 'OFF'} G:{gamma:.1f} B:{blur_ksize} | [X] Clear | [M] Config",
    ])
    return frame, new_vmax, True

def clear_trace_canvas():
    global _trace_canvas
    _trace_canvas = None


# ═══════════════════════════════════════════════════════
#  模式 3: 网格数据
# ═══════════════════════════════════════════════════════

def render_grid(
    processed: np.ndarray,
    max_signal: float,
    config: dict,
    window_size: tuple,
    fps: float,
) -> np.ndarray:
    """网格模式：每个单元格显示数值。"""
    ww, wh = window_size
    rows = config.get("rows", list(range(16)))
    cols = config.get("cols", list(range(16)))
    order = config.get("axis_order", "row_col")

    nr, nc = len(rows), len(cols)
    visual_rows = nr if order == "row_col" else nc
    visual_cols = nc if order == "row_col" else nr

    if nr == 0 or nc == 0:
        frame = np.zeros((wh, ww, 3), dtype=np.uint8)
        _draw_hud(frame, ["Press hotkey to config"], (0, 0, 255))
        return frame

    cell_size = min((ww - 100) / max(1, visual_cols), (wh - 120) / max(1, visual_rows))
    cell_size = max(5.0, min(80.0, cell_size))

    grid_w = int(visual_cols * cell_size)
    grid_h = int(visual_rows * cell_size)
    sx = (ww - grid_w) // 2
    sy = (wh - grid_h) // 2 + 30

    frame = np.zeros((wh, ww, 3), dtype=np.uint8)
    grid_vmax = max(5000.0, max_signal)

    for vi in range(visual_rows):
        for vj in range(visual_cols):
            if order == "row_col":
                r_idx, c_idx = rows[vi], cols[vj]
            else:
                r_idx, c_idx = rows[vj], cols[vi]
            if r_idx >= MATRIX_ROWS or c_idx >= MATRIX_COLS:
                continue

            val = int(processed[r_idx, c_idx])
            x1 = int(sx + vj * cell_size)
            y1 = int(sy + vi * cell_size)
            x2 = int(sx + (vj + 1) * cell_size)
            y2 = int(sy + (vi + 1) * cell_size)

            if val > 0:
                blue = min(255, int((val / grid_vmax) * 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (blue, 0, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), 1)

            if val > 0 and cell_size > 15:
                fs = max(0.3, cell_size / 60.0)
                text = str(val)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fs, 1)
                if tw > cell_size * 0.9:
                    fs *= (cell_size * 0.9 / tw)
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fs, 1)
                cv2.putText(
                    frame, text,
                    (x1 + (int(cell_size) - tw) // 2, y1 + (int(cell_size) + th) // 2),
                    cv2.FONT_HERSHEY_PLAIN, fs, (255, 255, 255), 1, cv2.LINE_AA,
                )

    _draw_hud(frame, [
        f"Grid | FPS:{fps:.1f} | Max:{int(max_signal)}",
        "[M] Config Matrix",
    ])
    return frame


# ═══════════════════════════════════════════════════════
#  模式 4: 仿生手掌
# ═══════════════════════════════════════════════════════

# 手部锚点坐标（相对于窗口）
HAND_ANCHORS = {
    "thumb_tip": (350, 280), "thumb_joint": (420, 360), "thumb_base": (480, 420),
    "index_tip": (510, 120), "index_joint": (540, 240), "index_base": (560, 340),
    "middle_tip": (640, 80), "middle_joint": (640, 200), "middle_base": (640, 320),
    "ring_tip": (770, 120), "ring_joint": (740, 240), "ring_base": (720, 340),
    "pinky_tip": (930, 280), "pinky_joint": (860, 360), "pinky_base": (800, 420),
    "palm": (640, 480),
}
WRIST_ANCHOR = (640, 600)

# 默认手部配置（16×16 传感器映射到手指区域）
DEFAULT_HAND = {
    "thumb_joint": {"rows": [0, 1, 2], "cols": [14, 12, 13, 15], "axis_order": "col_row"},
    "index_joint": {"rows": [3, 4, 5], "cols": [14, 12, 13, 15], "axis_order": "col_row"},
    "middle_joint": {"rows": [6, 7, 8], "cols": [14, 12, 13, 15], "axis_order": "col_row"},
    "ring_joint": {"rows": [9, 10, 11], "cols": [14, 12, 13, 15], "axis_order": "col_row"},
    "pinky_joint": {"rows": [12, 13, 14], "cols": [14, 12, 13, 15], "axis_order": "col_row"},
    "palm": {"rows": list(range(15)), "cols": [10, 9, 8, 6, 4], "axis_order": "col_row"},
}
for k in DEFAULT_HAND:
    if k != "palm":
        DEFAULT_HAND[k]["name"] = k


def _scale_point(x: float, y: float, sx: float, sy: float) -> tuple:
    """按缩放因子缩放坐标点。"""
    return (int(x * sx), int(y * sy))


def render_hand(
    processed: np.ndarray,
    max_signal: float,
    config: dict,
    window_size: tuple,
    current_vmax: float,
    fps: float,
    noise_gate: int,
    dyn_ratio: float,
    spatial_on: bool,
    drift: float,
) -> tuple:
    """仿生手掌映射模式。

    硬编码锚点坐标基于 1280×720 画布设计，运行时根据实际 window_size
    等比缩放，适配任意尺寸（如回放面板的 640×400）。
    """
    ww, wh = window_size

    # ── 计算缩放因子（锚点基于 1280×720 设计） ──────────
    NOM_W, NOM_H = 1280.0, 720.0
    sx = ww / NOM_W
    sy = wh / NOM_H

    frame = np.full((wh, ww, 3), 15, dtype=np.uint8)
    lut = _get_viridis_lut()

    # ── 缩放后的锚点 ────────────────────────────────────
    def _sp(x, y):
        return _scale_point(x, y, sx, sy)

    wrist = _sp(*WRIST_ANCHOR)
    scaled_anchors = {k: _sp(*v) for k, v in HAND_ANCHORS.items()}

    # ── 缩放线宽和圆半径 ──────────────────────────────
    lw_scale = min(sx, sy)
    _lw = lambda v: max(1, int(v * lw_scale))
    _cr = lambda v: max(1, int(v * lw_scale))

    # 画骨骼框架
    line_c, joint_c = (60, 60, 60), (80, 80, 80)
    cv2.line(frame, wrist, scaled_anchors["palm"], line_c, _lw(6), cv2.LINE_AA)
    for fg in ["thumb", "index", "middle", "ring", "pinky"]:
        b = scaled_anchors[f"{fg}_base"]
        j = scaled_anchors[f"{fg}_joint"]
        t = scaled_anchors[f"{fg}_tip"]
        cv2.line(frame, scaled_anchors["palm"], b, line_c, _lw(5), cv2.LINE_AA)
        cv2.line(frame, b, j, line_c, _lw(4), cv2.LINE_AA)
        cv2.line(frame, j, t, line_c, _lw(3), cv2.LINE_AA)
        cv2.circle(frame, b, _cr(10), joint_c, -1, cv2.LINE_AA)
        cv2.circle(frame, j, _cr(8), joint_c, -1, cv2.LINE_AA)
        cv2.circle(frame, t, _cr(6), joint_c, -1, cv2.LINE_AA)
    cv2.circle(frame, scaled_anchors["palm"], _cr(16), joint_c, -1, cv2.LINE_AA)

    # 平滑 vmax
    vmax = max(max_signal, 5000)
    new_vmax = current_vmax * 0.95 + vmax * 0.05 if vmax < current_vmax else vmax

    CELL = max(1, int(14 * min(sx, sy)))
    for part_key, cfg in config.items():
        if part_key not in scaled_anchors:
            continue
        ax, ay = scaled_anchors[part_key]
        rows = cfg.get("rows", [])
        cols = cfg.get("cols", [])
        order = cfg.get("axis_order", "row_col")
        if not rows or not cols:
            continue

        nrr, ncc = len(rows), len(cols)
        w = ncc * CELL if order == "row_col" else nrr * CELL
        h = nrr * CELL if order == "row_col" else ncc * CELL
        cx, cy = ax - w // 2, ay - h // 2

        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                if r >= MATRIX_ROWS or c >= MATRIX_COLS:
                    continue
                val = processed[r, c]
                x1 = cx + (j if order == "row_col" else i) * CELL
                y1 = cy + (i if order == "row_col" else j) * CELL
                x2, y2 = x1 + CELL, y1 + CELL

                if val > 0:
                    idx = int(min(255, (val / new_vmax) * 255))
                    b, g, rr = lut[idx, 0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (int(b), int(g), int(rr)), -1)
                    # 单元格内显示数据值
                    if CELL > 14:
                        vtext = str(int(val))
                        fs = 0.4
                        (tw, th), _ = cv2.getTextSize(vtext, cv2.FONT_HERSHEY_PLAIN, fs, 1)
                        tc = (0, 0, 0) if (0.299 * rr + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255)
                        cv2.putText(frame, vtext,
                                    (x1 + (CELL - tw) // 2, y1 + (CELL + th) // 2),
                                    cv2.FONT_HERSHEY_PLAIN, fs, tc, 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

    _draw_hud(frame, [
        f"Bionic Hand | FPS:{fps:.1f} | Max:{int(max_signal)} | Drift:{int(drift)}",
        f"Gate:{noise_gate} | Dyn:{dyn_ratio:.2f} | Filter:{'ON' if spatial_on else 'OFF'}",
        "[M] Config | [C] Calibrate",
    ])
    return frame, new_vmax


# ═══════════════════════════════════════════════════════
#  模式 5: 拓扑形变
# ═══════════════════════════════════════════════════════

class DeformMeshState:
    """模式5 的交互状态。"""

    def __init__(self):
        self.holes = []
        self.flip_x = False
        self.flip_y = False
        self.deform_strength = 1.0
        self.cached_x = None
        self.cached_y = None
        self.cache_valid = False
        # 绘制中
        self.drawing = False
        self.draw_start = (0, 0)
        self.draw_end = (0, 0)


def _update_mesh_cache(state: DeformMeshState, rows, cols, order, ww, wh):
    """更新形变网格缓存。"""
    nr, nc = len(rows), len(cols)
    if nr == 0 or nc == 0:
        state.cached_x = state.cached_y = None
        return

    CELL = 26
    tw = nc * CELL if order == "row_col" else nr * CELL
    th = nr * CELL if order == "row_col" else nc * CELL
    sx = (ww - tw) // 2
    sy = (wh - th) // 2 + 30

    jj, ii = np.meshgrid(np.arange(nc), np.arange(nr))
    if order == "row_col":
        mx = sx + jj * CELL
        my = sy + ii * CELL
    else:
        mx = sx + ii * CELL
        my = sy + jj * CELL

    if state.flip_x:
        mx, my = np.fliplr(mx), np.fliplr(my)
    if state.flip_y:
        mx, my = np.flipud(mx), np.flipud(my)

    # 应用孔洞形变
    holes = list(state.holes)
    if state.drawing:
        cx1, cy1 = state.draw_start
        cx2, cy2 = state.draw_end
        rx, ry = max(1, abs(cx2 - cx1)), max(1, abs(cy2 - cy1))
        if rx > 5 and ry > 5:
            holes.append({"cx": cx1, "cy": cy1, "rx": rx, "ry": ry})

    for h in holes:
        cx, cy, rx, ry = h["cx"], h["cy"], h["rx"], h["ry"]
        vx, vy = mx - cx, my - cy
        d = np.sqrt((vx / rx) ** 2 + (vy / ry) ** 2) + 1e-6
        target_d = d + state.deform_strength * np.exp(-d * 1.5)
        scale = target_d / d
        mx = cx + vx * scale
        my = cy + vy * scale

    state.cached_x, state.cached_y = mx, my
    state.cache_valid = True


def render_deform_mesh(
    processed: np.ndarray,
    max_signal: float,
    config: dict,
    window_size: tuple,
    current_vmax: float,
    fps: float,
    state: DeformMeshState,
) -> tuple:
    """拓扑形变模式。"""
    ww, wh = window_size
    frame = np.full((wh, ww, 3), 15, dtype=np.uint8)
    lut = _get_viridis_lut()

    vmax = max(5000, max_signal)
    new_vmax = current_vmax * 0.95 + vmax * 0.05 if vmax < current_vmax else vmax

    rows = config.get("rows", list(range(16)))
    cols = config.get("cols", list(range(16)))
    order = config.get("axis_order", "row_col")

    if not state.cache_valid:
        _update_mesh_cache(state, rows, cols, order, ww, wh)

    mx, my = state.cached_x, state.cached_y
    if mx is None:
        _draw_hud(frame, ["Press [M] to select Matrix Rows/Cols!"], (0, 0, 255))
        return frame, new_vmax

    nr, nc = len(rows), len(cols)

    # 画网格线
    pts = np.stack([mx, my], axis=-1).astype(np.int32)
    cv2.polylines(frame, pts, False, (60, 60, 60), 1, cv2.LINE_AA)
    pts_t = np.ascontiguousarray(np.transpose(pts, (1, 0, 2)))
    cv2.polylines(frame, pts_t, False, (60, 60, 60), 1, cv2.LINE_AA)

    # 画孔洞
    holes = list(state.holes)
    if state.drawing:
        cx1, cy1 = state.draw_start
        cx2, cy2 = state.draw_end
        rx, ry = max(1, abs(cx2 - cx1)), max(1, abs(cy2 - cy1))
        if rx > 5 and ry > 5:
            holes.append({"cx": cx1, "cy": cy1, "rx": rx, "ry": ry})
    for h in holes:
        cv2.ellipse(frame, (h["cx"], h["cy"]), (h["rx"], h["ry"]),
                     0, 0, 360, (90, 90, 90), 2, cv2.LINE_AA)

    # 画数据点
    for i in range(nr):
        for j in range(nc):
            ri, cj = rows[i], cols[j]
            if ri >= MATRIX_ROWS or cj >= MATRIX_COLS:
                continue
            val = processed[ri, cj]
            px, py = int(mx[i, j]), int(my[i, j])
            if val > 0:
                idx = int(min(255, (val / new_vmax) * 255))
                b, g, rr = lut[idx, 0]
                cv2.circle(frame, (px, py), 10, (int(b), int(g), int(rr)), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), 3, (80, 80, 80), -1, cv2.LINE_AA)

    s = state.deform_strength
    _draw_hud(frame, [
        f"Deform Mesh | FPS:{fps:.1f} | Max:{int(max_signal)} | Strength:{s:.1f}",
        f"Flip X:{state.flip_x} | Flip Y:{state.flip_y}",
        "[Drag] Draw Hole | [R] Clear | [U/I] Flip | [O/P] Strength | [M] Config",
    ])
    return frame, new_vmax

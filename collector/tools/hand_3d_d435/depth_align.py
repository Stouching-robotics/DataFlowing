#!/usr/bin/env python3
"""D435 深度→彩色对齐 + 关键点深度采样。

深度流（848×480，HFOV≈89°）与彩色流（1280×720，HFOV≈70°）视角不同，采集链路
（core/d435_camera.py）无 rs2.align——本模块做离线前向对齐：

    逐深度像素 (u_d, v_d) @ Z_d = png值(mm)
      → P_d = [(u_d−cx_d)/fx_d, (v_d−cy_d)/fy_d, 1] · Z_d   深度相机系
      → P_c = R @ P_d + t                                    彩色相机系
      → (u_c, v_c) = (fx_c·X_c/Z_c+cx_c, fy_c·Y_c/Z_c+cy_c)  彩色投影
      → aligned(ch, cw) float32 mm，z-buffer 保最近（0=无效）

标定来源：彩色内参/外参 = tools/extract_d435_color_calib.py 固化 JSON；
深度侧内参 = 录制期 calibration/head_stereo.json（权威，固化值仅交叉核对）。

独立自测（合成数据）：python depth_align.py
"""

from __future__ import annotations

import json
import os

import numpy as np

_MOD_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CALIB = os.path.join(_MOD_DIR, "calibration", "d435_color_calib.json")

_MAX_DEPTH_MM = 8000.0      # 远背景离群剔除上限（实测有效值 p95≈1.1m）
_FX_REL_TOL = 0.01          # 深度内参交叉核对容差（>1% 告警，可能换机）


def load_session_depth_intr(session_dir: str) -> dict | None:
    """录制期 head_stereo.json → 深度相机内参 dict（fx/fy/cx/cy/width/height）。"""
    path = os.path.join(session_dir, "calibration", "head_stereo.json")
    try:
        with open(path, encoding="utf-8") as f:
            head = json.load(f)
        dc = head["depth_camera"]
        return {
            "fx": float(dc["intrinsic"][0]), "fy": float(dc["intrinsic"][1]),
            "cx": float(dc["intrinsic"][2]), "cy": float(dc["intrinsic"][3]),
            "width": int(dc.get("resolution", [848, 480])[0]),
            "height": int(dc.get("resolution", [848, 480])[1]),
        }
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        return None


def load_session_depth_shape(session_dir: str,
                             depth_slot: str = "d435_depth") -> tuple | None:
    """深度帧尺寸 (height, width)：优先 head_stereo.json 深度内参
    resolution，回落 metadata.json cameras.<slot>.width/height。

    尺寸仅用于 v1.0.11 窗口会话的 raw16 bin 回退读取
    （np.fromfile(dtype=uint16).reshape(h, w)）；png16 不需要。
    """
    sd = load_session_depth_intr(session_dir)
    if sd is not None:
        return int(sd["height"]), int(sd["width"])
    try:
        with open(os.path.join(session_dir, "metadata.json"),
                  encoding="utf-8") as f:
            meta = json.load(f)
        cam = meta["cameras"][depth_slot]
        return int(cam["height"]), int(cam["width"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def load_session_depth_files(depth_dir: str) -> dict:
    """深度帧索引 {1-based 序号: 路径}。

    v1.0.12 起 png16 优先；v1.0.11 窗口会话（raw16 bin）自动回退
    兼容——读取统一走 load_depth_frame（按扩展名分流）。
    """
    import glob as _glob
    files = {}
    for pat in ("*.png", "*.bin"):
        for p in _glob.glob(os.path.join(depth_dir, pat)):
            try:
                files.setdefault(int(os.path.basename(p).split(".")[0]), p)
            except ValueError:
                pass
        if files:
            break
    return files


def load_depth_frame(path: str, shape: tuple | None) -> np.ndarray | None:
    """读一帧深度 → uint16 (H, W)（毫米）。

    png16（cv2.imread）；v1.0.11 窗口会话 raw16 bin 回退
    （np.fromfile + reshape，shape 取 load_session_depth_shape）。
    shape 未知或 bin 元素数与尺寸不匹配 → None（调用方按缺深度处理）。
    """
    try:
        if path.endswith(".bin"):
            data = np.fromfile(path, dtype=np.uint16)
            if shape is None or int(data.size) != int(shape[0]) * int(shape[1]):
                return None
            return data.reshape(shape)
        import cv2
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except (OSError, ValueError):
        return None


def load_calib(calib_path: str = None) -> dict:
    """固化标定 JSON。找不到时抛 FileNotFoundError（提示先跑提取脚本）。"""
    path = calib_path or _DEFAULT_CALIB
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"标定文件不存在: {path}\n"
            f"请先接入 D435 并（设备空闲时）运行:\n"
            f"  venv/bin/python tools/hand_3d_d435/tools/extract_d435_color_calib.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class DepthAligner:
    """深度图 → 彩色视口 aligned 深度图（毫米，0=无效）。"""

    def __init__(self, color_intr: dict, depth_to_color: dict,
                 depth_intr: dict):
        self.fx_c = float(color_intr["fx"])
        self.fy_c = float(color_intr["fy"])
        self.cx_c = float(color_intr["cx"])
        self.cy_c = float(color_intr["cy"])
        self.cw = int(color_intr.get("width", 1280))
        self.ch = int(color_intr.get("height", 720))
        fxd = float(depth_intr["fx"])
        fyd = float(depth_intr["fy"])
        cxd = float(depth_intr["cx"])
        cyd = float(depth_intr["cy"])
        self.fx_d = fxd
        self.fy_d = fyd
        self.cx_d = cxd
        self.cy_d = cyd
        self.dw = int(depth_intr.get("width", 848))
        self.dh = int(depth_intr.get("height", 480))
        uu, vv = np.meshgrid(np.arange(self.dw, dtype=np.float32),
                             np.arange(self.dh, dtype=np.float32))
        # (dh,dw,3) 深度相机系单位射线（标定不变，构造时预计算一次）
        self._ray = np.stack([(uu - cxd) / fxd, (vv - cyd) / fyd,
                              np.ones_like(uu)], axis=-1).astype(np.float32)
        self._R = np.asarray(depth_to_color["rotation"], np.float64)
        self._t_mm = (np.asarray(depth_to_color["translation"], np.float64)
                      * 1000.0)
        # 恒等对齐判定（S80C：深度与彩色同 P0 空间）。R≈I、t≈0、
        # 内外参同、分辨率同 → 对齐即"裁剪+轻量填洞"，跳过逐点射线
        # 反投影（1280×800 全网格 ~29ms → 几 ms）。D435 的跨相机
        # 标定（真旋转+平移+异分辨率）永不触发此路径，行为零变化。
        self._identity = (
            np.allclose(self._R, np.eye(3), atol=1e-6)
            and np.allclose(self._t_mm, 0.0, atol=1e-3)
            and abs(fxd - self.fx_c) <= 1e-6 * max(1.0, abs(fxd))
            and abs(fyd - self.fy_c) <= 1e-6 * max(1.0, abs(fyd))
            and abs(cxd - self.cx_c) <= 1e-3
            and abs(cyd - self.cy_c) <= 1e-3
            and self.dw == self.cw and self.dh == self.ch)

    def align_depth_to_color(self, depth_mm: np.ndarray) -> np.ndarray:
        """(dh,dw) uint16/float 毫米深度 → (ch,cw) float32 aligned 深度（0=无效）。"""
        if self._identity:
            # 同空间同分辨率：有效区原样保留；填洞轮数随 fill_passes
            # （LiveAligner 注入，默认 1；S80C 实时默认 0——1:1 无上采样
            # 空穴，sample_points 自带 3×3 中位窗口，不填也稳）。
            # copy：frombuffer 管道数组只读，原地裁剪会炸
            z = np.array(depth_mm, np.float32, copy=True)
            z[(z <= 0) | (z > _MAX_DEPTH_MM)] = 0.0
            return self._fill_holes(z)
        z = np.asarray(depth_mm, np.float32)
        valid = (z > 0) & (z <= _MAX_DEPTH_MM)
        p_d = self._ray * z[..., None]                    # 深度相机系, mm
        p_c = p_d[valid] @ self._R.T + self._t_mm         # 彩色相机系, mm
        with np.errstate(divide="ignore", invalid="ignore"):
            u_c = self.fx_c * p_c[:, 0] / p_c[:, 2] + self.cx_c
            v_c = self.fy_c * p_c[:, 1] / p_c[:, 2] + self.cy_c
            z_c = p_c[:, 2]
        iu = np.rint(u_c).astype(np.int32)
        iv = np.rint(v_c).astype(np.int32)
        inside = ((iu >= 0) & (iu < self.cw) & (iv >= 0) & (iv < self.ch)
                  & (z_c > 0))
        # z-buffer：初始化为 +inf（0 是无效哨兵，用 0 初始化会让
        # minimum.at 的 min(0, z)=0 永远写不进任何正深度——已实测坑）
        aligned = np.full((self.ch, self.cw), np.inf, np.float32)
        np.minimum.at(aligned, (iv[inside], iu[inside]),
                      z_c[inside].astype(np.float32))     # 最近保留
        aligned[~np.isfinite(aligned)] = 0.0
        return self._fill_holes(aligned)

    def _fill_holes(self, aligned: np.ndarray, passes: int = 3) -> np.ndarray:
        """空穴回填：每轮对 0 像素取 3×3 有效邻域最小值（最近表面语义）。

        前向投影是 848×480 → 1280×720 的 ~2.12× 上采样：z-buffer 后 rint
        跳列/跳行留下 ~50% 空穴（覆盖仅 ~19%，边缘探针全噪声）。缺口 ≤3px，
        3 轮即可补满；值只写空穴，已有值不碰（不腐蚀有效区）。
        """
        if passes <= 0:
            # 0 轮 = 不填洞（S80C 实时默认：1:1 无上采样空穴，sample_points
            # 自带 3×3 中位窗口）。守卫放在导入前——原实现 scipy 懒导入在
            # for 循环外无条件执行，fill=0 也白付一次 ~150ms 导入。
            return aligned
        from scipy.ndimage import minimum_filter
        for _ in range(passes):
            holes = aligned == 0
            if not holes.any():
                break
            work = np.where(holes, np.inf, aligned)
            nb = minimum_filter(work, size=3, mode="constant", cval=np.inf)
            ok = holes & np.isfinite(nb)
            if not ok.any():
                break
            aligned[ok] = nb[ok]
        return aligned

    def sample_points(self, aligned: np.ndarray, uv, band=None) -> np.ndarray:
        """(N,2) 像素坐标（亚像素取最近整像素）→ (N,) 深度 mm，无效 NaN。

        3×3 窗口剔除 0/非有限后取中位，有效数 ≥2 才出数——中位天然抗
        边缘混入背景（手 ~445mm vs 背景 >1000mm，窗口内少数背景点不赢中位）。
        band=(z_lo, z_hi) 时窗口像素先按深度带过滤再取中位（手缘点窗口
        混入背景时背景像素被剔除，中位必落手上；带内有效 <2 → NaN，
        由上层 tracker 预测补全）。band 单位与 aligned 一致（mm）。
        """
        uv = np.asarray(uv, np.float32).reshape(-1, 2)
        u = np.rint(uv[:, 0]).astype(np.int32)
        v = np.rint(uv[:, 1]).astype(np.int32)
        out = np.full(len(uv), np.nan, np.float32)
        for k in range(len(uv)):
            i0, i1 = max(v[k] - 1, 0), min(v[k] + 2, self.ch)
            j0, j1 = max(u[k] - 1, 0), min(u[k] + 2, self.cw)
            w = aligned[i0:i1, j0:j1]
            w = w[(w > 0) & np.isfinite(w)]
            if band is not None:
                w = w[(w >= band[0]) & (w <= band[1])]
            if w.size >= 2:
                out[k] = float(np.median(w))
        return out


# ── 合成自测 ──────────────────────────────────────────────────

def _selftest():
    """R=I,t=0 同分辨率时对齐应精确复现深度图；t_x=25mm 平移验证方向。"""
    intr = {"fx": 429.4733, "fy": 429.4733, "cx": 420.4592, "cy": 231.8084,
            "width": 848, "height": 480}

    # 1) 恒等外参 + 同内参同分辨率：aligned == depth（1:1 映射，逐点精确）
    a0 = DepthAligner(intr, {"rotation": np.eye(3).tolist(),
                             "translation": [0.0, 0.0, 0.0]}, intr)
    rng = np.random.default_rng(7)
    dep = (rng.uniform(300.0, 2000.0, (480, 848))).astype(np.float32)
    dep[rng.random((480, 848)) < 0.15] = 0.0
    al = a0.align_depth_to_color(dep)
    assert al.shape == (480, 848), al.shape
    # 有效像素必须逐点精确复现（原 0 空穴会被 min 邻域回填，属新语义）
    assert np.array_equal(al[dep > 0], dep[dep > 0]), f"恒等对齐不精确: max|diff|=" \
        f"{np.abs(al[dep > 0] - dep[dep > 0]).max()}"
    print("✓ selftest 1: 恒等外参对齐逐点精确复现")

    # 2) t=[25,0,0]mm：深度像 (424,240)@1000mm 的柱应横向平移
    #    Δu = fx·t_x/Z = 429.47×25/1000 ≈ 10.7px → 落在 u≈435
    a25 = DepthAligner(intr, {"rotation": np.eye(3).tolist(),
                              "translation": [0.025, 0.0, 0.0]}, intr)
    pillar = np.zeros((480, 848), np.float32)
    pillar[:, 424] = 1000.0
    al25 = a25.align_depth_to_color(pillar)
    nz = np.nonzero(al25)
    u_hit = int(np.median(nz[1]))
    expect = int(np.rint(424 + 429.4733 * 25.0 / 1000.0))
    assert u_hit == expect, f"t_x=25mm 柱平移 {u_hit}px，期望 {expect}px"
    assert np.allclose(al25[nz][:5], 1000.0), "平移后深度值应保持 1000mm"
    print(f"✓ selftest 2: t_x=25mm 柱平移 {u_hit}px（期望 {expect}px），深度值保持")

    # 3) 3×3 中位采样：窗口内 1 个背景点不赢中位；有效 <2 → NaN
    grid = np.zeros((10, 10), np.float32)
    grid[4:7, 4:7] = 445.0            # 手
    grid[4, 4] = 1200.0               # 单点背景混入
    assert a0.sample_points(grid, [[5.2, 5.1]])[0] == 445.0, "3×3 中位未抗单点离群"
    assert np.isnan(a0.sample_points(grid, [[9.9, 9.9]])[0]), "全无效窗口应 NaN"
    grid2 = np.zeros((10, 10), np.float32)
    grid2[5, 5] = 500.0               # 仅 1 个有效值 → 不足 2 → NaN
    assert np.isnan(a0.sample_points(grid2, [[5.0, 5.0]])[0]), "有效<2 应 NaN"
    print("✓ selftest 3: 3×3 中位抗离群 / 有效数<2 出 NaN")

    # 4) 空穴回填：模拟 2.12× 上采样稀疏网格（隔列隔行有值 + 3px 缺口），
    #    3 轮后应补满；回填值取最近表面（min）；已有值不被腐蚀
    a = np.zeros((12, 12), np.float32)
    a[::2, ::2] = 500.0                # 稀疏网格
    a[6:8, 6] = 0.0                    # 人为 2px 缺口
    filled = a0._fill_holes(a.copy())
    assert (filled > 0).mean() > 0.9, f"回填不充分: {(filled > 0).mean():.2f}"
    assert filled[3, 3] == 500.0, "已有值被腐蚀"
    f2 = a0._fill_holes(filled, passes=1)   # 已满时再填应无变化
    assert np.array_equal(f2, filled), "已满图被多余修改"
    print(f"✓ selftest 4: 稀疏上采样空穴 3 轮回填（覆盖 {(filled > 0).mean() * 100:.0f}%），不腐蚀已有值")

    print("全部自测通过")


if __name__ == "__main__":
    _selftest()

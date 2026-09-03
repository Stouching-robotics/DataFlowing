#!/usr/bin/env python3
"""
Episode SLAM 轨迹对比分析器 (多档压缩 / 多 episode 两两对比)。

用于把多份来自**同一段录制、不同处理参数**(如不同视频 CRF)的 SLAM 轨迹
放在一起对比,量化轨迹间差异。参考 analyze_test.py 的 RPE / ATE / 逐帧
偏差方法,但对比对象从"SLAM vs 机械臂FK"换成"N 个 episode 目录互相对比"。

核心前提(已验证):
  - 每份 episode 的 results/trajectory/trajectory_online_style.txt 采用
    相同的坐标修正(rotation_correction + 原点平移),起点时间与起始四元数
    完全一致 → 无需手眼标定,可直接在同一时间轴上逐帧比较。
  - 时间轴 = 相对时间(第1列, 从 0 起),三档起点相同。

用法:
  # 两个及以上 episode 目录, 两两对比:
  python analyze_episode_crf.py /path/episode_00009.zip.new \
                                 /path/episode_00009_crf30.zip.new \
                                 /path/episode_00009_crf37.zip.new --plot
  # 指定基准(否则两两):
  python analyze_episode_crf.py A B C --ref A --plot
  # 切换轨迹文件 / 报告输出:
  python analyze_episode_crf.py A B C --traj tum --outdir reports --plot

输出:
  每次运行在 --outdir 下新建独立时间戳子目录 run_<时间戳>/, 历史结果互不覆盖:
  - report_<时间戳>.txt + summary.txt: 每档质量表 + 逐对 RPE/ATE/最大偏差 + [总结]
  - --plot 时: traj_3d/traj_xy/traj_xz/traj_yz/pos_vs_time/dev_vs_time/rpe_bar/tracking_ok.png
"""

import sys, os, math, time, argparse
import numpy as np


# ---------------------------------------------------------------------------
# 基础 SE3 / 四元数工具 (与 analyze_test.py 一致)
# ---------------------------------------------------------------------------

def quat_to_rot(q_xyzw):
    """[qx,qy,qz,qw] → 3×3"""
    x, y, z, w = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
    return np.array([
        [1 - 2 * y * y - 2 * z * z,   2 * x * y - 2 * w * z,   2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z,     2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y,   2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def rot_to_axis_angle(R):
    """3×3 → axis*angle 向量, ||v|| = angle(rad)"""
    theta = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if theta < 1e-10:
        return np.zeros(3)
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    s = math.sqrt(rx * rx + ry * ry + rz * rz)
    if s < 1e-10:
        return np.zeros(3)
    axis = np.array([rx, ry, rz]) / s
    return axis * theta


def make_T(xyz, q_xyzw):
    """→ 4×4 SE3, 输入四元数顺序为 [x,y,z,w]。"""
    T = np.eye(4)
    T[0:3, 0:3] = quat_to_rot(q_xyzw)
    T[0:3, 3] = xyz
    return T


def normalize_quat(q_xyzw):
    n = np.linalg.norm(q_xyzw)
    return q_xyzw / n if n > 1e-15 else np.array([0.0, 0.0, 0.0, 1.0])


def quat_rot_error(qa, qb):
    """两姿态间相对旋转角(rad), 输入 [x,y,z,w]"""
    Ra, Rb = quat_to_rot(qa), quat_to_rot(qb)
    E = Ra.T @ Rb
    return np.linalg.norm(rot_to_axis_angle(E))


def umeyama(src_pts, dst_pts):
    """src → dst 的刚体变换估计 (Umeyama, 无缩放)。返回 (R, t)。"""
    c_s = src_pts.mean(axis=0)
    c_d = dst_pts.mean(axis=0)
    sc = src_pts - c_s
    dc = dst_pts - c_d
    H = (sc.T @ dc) / len(sc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_d - R @ c_s
    return R, t


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def resolve_episode_root(path):
    """给定 episode 目录(可能是 zip 名目录或 episode_XXXX 子目录) → episode 根。

    根目录特征: 存在 results/ 子目录。若传入的是父目录, 自动找唯一/第一个
    episode_* 子目录。
    """
    if os.path.isdir(os.path.join(path, "results")):
        return path
    cands = sorted(
        d for d in os.listdir(path)
        if d.startswith("episode") and os.path.isdir(os.path.join(path, d))
    )
    if not cands:
        raise ValueError(f"无法从 {path} 定位 episode 根目录(找不到 results/ 或 episode_* 子目录)")
    return os.path.join(path, cands[0])


def load_trajectory(episode_root, traj_kind="online"):
    """读取 SLAM 轨迹。

    traj_kind:
      online (默认) -> results/trajectory/trajectory_online_style.txt
                       9列: t_rel x y z qx qy qz qw abs_sensor_s
      tum    -> results/trajectory/camera_trajectory_tum.txt
                       (严格 TUM: ts_s x y z qx qy qz qw, ts为秒)
      body   -> results/trajectory/body_trajectory_euroc.txt
                       (EuRoC: ts_ns px py pz vx vy vz qx qy qz qw bg ba)
    """
    traj_dir = os.path.join(episode_root, "results", "trajectory")
    if traj_kind == "online":
        p = os.path.join(traj_dir, "trajectory_online_style.txt")
        arr = np.loadtxt(p)
        if arr.ndim == 1 or arr.shape[1] < 8:
            raise ValueError(f"{p} 列数不足")
        ts = arr[:, 0].copy()
        xyz = arr[:, 1:4].copy()
        q = arr[:, 4:8].copy()
        abs_s = arr[:, 8].copy() if arr.shape[1] >= 9 else np.full(len(ts), np.nan)
    elif traj_kind == "tum":
        p = os.path.join(traj_dir, "camera_trajectory_tum.txt")
        arr = np.loadtxt(p)
        ts = arr[:, 0].copy() - arr[0, 0]          # 相对秒
        xyz = arr[:, 1:4].copy()
        q = arr[:, 4:8].copy()
        abs_s = arr[:, 0].copy()
    elif traj_kind == "body":
        p = os.path.join(traj_dir, "body_trajectory_euroc.txt")
        arr = np.loadtxt(p)
        # EuRoC: 列0为纳秒时间戳; 位姿平移在第1:4列; 四元数第7:11列
        ts = (arr[:, 0] - arr[0, 0]) * 1e-9         # 纳秒→相对秒
        xyz = arr[:, 1:4].copy()
        q = arr[:, 7:11].copy()
        abs_s = arr[:, 0] * 1e-9
    else:
        raise ValueError(f"未知轨迹类型: {traj_kind}")

    # 恢复单位四元数, 避免微小范数误差放大为旋转误差
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norms, 1e-15)

    # 相对时间起点归零, 且严格递增
    ts = ts - ts[0]
    order = np.argsort(ts)
    ts, xyz, q, abs_s = ts[order], xyz[order], q[order], abs_s[order]

    # 四元数符号连续性解卷: q 与 -q 表示同一旋转, ORB 输出符号可能跳变。
    # 若不统一, 跨符号跳变做分量插值会在中点得到近零向量 → 虚假旋转误差。
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]

    # 命名用外层变体目录(如 episode_00009_crf30.zip.new), 而非 episode_0000
    base = os.path.basename(episode_root)
    if base.startswith("episode"):
        name = os.path.basename(os.path.dirname(episode_root))
    else:
        name = base

    T_list = [make_T(xyz[i], q[i]) for i in range(len(ts))]
    return {"name": name, "root": episode_root, "path": p,
            "ts": ts, "xyz": xyz, "q": q, "abs_s": abs_s, "T": T_list}


def load_run_report(episode_root):
    """读取 results/trajectory/run_report.yaml 关键字段。"""
    meta = {}
    p = os.path.join(episode_root, "results", "trajectory", "run_report.yaml")
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if ":" in line and not line.startswith("#"):
                    k, v = line.split(":", 1)
                    k, v = k.strip(), v.strip()
                    try:
                        meta[k] = float(v)
                    except ValueError:
                        meta[k] = v
    return meta


def load_tracking_states(episode_root):
    """读取 results/trajectory/tracking.csv → (aligned_ts_s, tracking_state)。

    tracking_state 约定(与离线工具一致): 2 = 跟踪OK, 其余为未就绪/丢失。
    """
    p = os.path.join(episode_root, "results", "trajectory", "tracking.csv")
    if not os.path.exists(p):
        return None, None
    rows = []
    with open(p) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header is None:
                header = line.split(",")
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                rows.append((float(parts[2]), int(parts[3])))
            except ValueError:
                continue
    if not rows:
        return None, None
    ts = np.array([r[0] for r in rows])
    states = np.array([r[1] for r in rows])
    ts = ts - ts[0]
    return ts, states


# ---------------------------------------------------------------------------
# 公共时间轴插值 (位置逐轴线性, 四元数分量插值后归一化)
# ---------------------------------------------------------------------------

def interpolate_pose(trajectory, grid):
    """把轨迹插值到公共时间轴 grid → (xyz_g, q_g)。grid 需在轨迹时间范围内。"""
    ts, xyz, q = trajectory["ts"], trajectory["xyz"], trajectory["q"]
    xyz_g = np.column_stack([np.interp(grid, ts, xyz[:, k]) for k in range(3)])
    q_g = np.column_stack([np.interp(grid, ts, q[:, k]) for k in range(4)])
    q_g = q_g / np.maximum(np.linalg.norm(q_g, axis=1, keepdims=True), 1e-15)
    return xyz_g, q_g


def valid_mask(ts, grid, gap_thresh=0.3):
    """公共时间轴上"该轨迹有有效位姿"的掩码。

    若 grid 点落在这条轨迹的某个大于 gap_thresh 的样本间隙内部
    (离最近样本超过 gap_thresh/2), 说明该处 SLAM 没有估计值,
    插值得到的姿态无意义 → 标记为无效, 避免跨跟丢间隙的虚假大偏差。
    """
    idx = np.searchsorted(ts, grid)
    left = np.clip(idx - 1, 0, len(ts) - 1)
    right = np.clip(idx, 0, len(ts) - 1)
    d = np.minimum(np.abs(ts[left] - grid), np.abs(ts[right] - grid))
    return (d <= gap_thresh / 2) & (grid >= ts[0]) & (grid <= ts[-1])


# ---------------------------------------------------------------------------
# 对比指标
# ---------------------------------------------------------------------------

def compute_deviation(ta, tb, grid, valid):
    """同一公共时间轴上的逐帧偏差: 位置(米) + 旋转(度)。无效点置 NaN。"""
    xa, qa = interpolate_pose(ta, grid)
    xb, qb = interpolate_pose(tb, grid)
    pos_dev = np.full(len(grid), np.nan)
    rot_dev = np.full(len(grid), np.nan)
    pos_dev[valid] = np.linalg.norm(xa[valid] - xb[valid], axis=1)
    rot_dev[valid] = np.array([
        math.degrees(quat_rot_error(qa[i], qb[i])) for i in np.flatnonzero(valid)
    ])
    return pos_dev, rot_dev


def compute_rpe_time(ta, tb, grid, delta_s, valid):
    """时间增量 RPE: 每隔 delta_s 比较两轨迹的帧间相对运动。

    E = inv(ΔB) @ ΔA, ΔX = inv(T_X(t)) @ T_X(t+delta)
    RPE_trans = |trans(E)|, RPE_rot = |axisangle(rot(E))| (度)。
    只比较运动大小, 不需要坐标系对齐。仅统计两端都有效的点。
    """
    xa, qa = interpolate_pose(ta, grid)
    xb, qb = interpolate_pose(tb, grid)
    trans_errs, rot_errs = [], []
    n = len(grid)
    for i in range(n):
        if not valid[i]:
            continue
        j = np.searchsorted(grid, grid[i] + delta_s)
        if j >= n or not valid[j]:
            continue
        dA = np.linalg.inv(make_T(xa[i], qa[i])) @ make_T(xa[j], qa[j])
        dB = np.linalg.inv(make_T(xb[i], qb[i])) @ make_T(xb[j], qb[j])
        E = np.linalg.inv(dA) @ dB
        trans_errs.append(np.linalg.norm(E[0:3, 3]))
        rot_errs.append(math.degrees(np.linalg.norm(rot_to_axis_angle(E[0:3, 0:3]))))
    return np.array(trans_errs), np.array(rot_errs)


def compute_ate(ta, tb, grid, valid):
    """ATE: Umeyama 对齐 B→A 后的平移残差 (米)。仅统计有效点。"""
    xa, _ = interpolate_pose(ta, grid)
    xb, _ = interpolate_pose(tb, grid)
    R, t = umeyama(xb[valid], xa[valid])
    aligned = (R @ xb.T).T + t
    errs = np.full(len(grid), np.nan)
    errs[valid] = np.linalg.norm(aligned[valid] - xa[valid], axis=1)
    return errs


def trajectory_quality(tr, report, states):
    """每档单条轨迹的质量指标。"""
    xyz = tr["xyz"]
    path_len = np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
    max_disp = np.linalg.norm(xyz - xyz[0], axis=1).max()
    ok = report.get("tracking_ok_frames", None)
    total = report.get("input_frames", None)
    if states is not None:
        ok_cnt = int(np.sum(states == 2))
        if total is None:
            total = len(states)
        ok = ok_cnt
    ok_ratio = (ok / total * 100) if (ok is not None and total) else float("nan")
    return {
        "frames": len(tr["ts"]),
        "duration_s": float(tr["ts"][-1] - tr["ts"][0]),
        "path_len_m": path_len,
        "max_disp_m": max_disp,
        "tracking_ok": ok,
        "input_frames": total,
        "ok_ratio_pct": ok_ratio,
        "max_gap_s": float(np.max(np.diff(tr["ts"]))) if len(tr["ts"]) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# 报告 & 绘图
# ---------------------------------------------------------------------------

def build_report(episodes, trajs, ref_name, out_lines):
    qualities = []
    for tr in trajs:
        name = tr["name"]
        report = load_run_report(tr["root"])
        states_ts, states = load_tracking_states(tr["root"])
        q = trajectory_quality(tr, report, states)
        q["name"] = name
        qualities.append(q)
        out_lines.append(
            f"  {name:<28s} 帧数 {q['frames']:>4d}  时长 {q['duration_s']:>6.2f}s  "
            f"路径 {q['path_len_m']:>5.2f}m  最大位移 {q['max_disp_m']:>5.2f}m  "
            f"跟踪 {q['tracking_ok']}/{q['input_frames']} ({q['ok_ratio_pct']:.0f}%)  "
            f"最大间隙 {q['max_gap_s']:.1f}s"
        )
    out_lines.append("")
    return qualities


def build_summary(qualities, names, metrics_map, ref_name, delta, out_lines):
    """生成 [总结] 摘要段: 跟踪完整度排序 + 与基准差异最小的档位 + 结论。"""
    out_lines.append("")
    out_lines.append("=" * 72)
    out_lines.append("  [总结]")
    out_lines.append("=" * 72)
    if ref_name:
        out_lines.append(f"  对比基准: {ref_name}")
    else:
        out_lines.append("  对比模式: 两两对比 (无预设基准)")
    out_lines.append(f"  对比对象: {len(qualities)} 档 → {', '.join(q['name'] for q in qualities)}")

    # 跟踪完整度排序 (从高到低)
    by_ok = sorted(qualities, key=lambda q: q["ok_ratio_pct"], reverse=True)
    out_lines.append("")
    out_lines.append(f"  跟踪完整度排序 (输入 {by_ok[0]['input_frames']} 帧):")
    for k, q in enumerate(by_ok, 1):
        out_lines.append(
            f"    {k}. {q['name']:<28s} {q['ok_ratio_pct']:>5.1f}%  "
            f"({q['tracking_ok']}/{q['input_frames']})  最大间隙 {q['max_gap_s']:.1f}s")

    if metrics_map:
        # RPE 平移排序 (低 = 差异更小 / 更接近基准)
        rpe_key = f"rpe_t_{delta}s_mean_mm"
        ranked = sorted(metrics_map.items(), key=lambda kv: kv[1][2][rpe_key])
        sort_note = "更接近基准" if ref_name else "差异更小"
        out_lines.append("")
        out_lines.append(f"  轨迹差异 (以 RPE@{delta}s 平移排序, 低 = {sort_note}):")
        for key, (i, j, res) in ranked:
            out_lines.append(
                f"    {key:<52s} RPE {res[rpe_key]:>6.1f}mm / {res[f'rpe_r_{delta}s_mean_deg']:>5.2f}°  "
                f"ATE {res['ate_rmse_m']*1000:>6.1f}mm")
        if ref_name:
            worst = ranked[-1]
            out_lines.append("")
            out_lines.append(
                f"  → 与基准差异最小: {ranked[0][0]}  "
                f"(RPE@{delta}s {ranked[0][1][2][rpe_key]:.1f}mm, ATE {ranked[0][1][2]['ate_rmse_m']*1000:.1f}mm)")
            out_lines.append(
                f"  → 与基准差异最大: {worst[0]}  "
                f"(RPE@{delta}s {worst[1][2][rpe_key]:.1f}mm, ATE {worst[1][2]['ate_rmse_m']*1000:.1f}mm)")

    # 结论建议: 点名完整度最差/最好档位
    ok_min = min(q["ok_ratio_pct"] for q in qualities)
    worst = min(qualities, key=lambda q: q["ok_ratio_pct"])
    best = max(qualities, key=lambda q: q["ok_ratio_pct"])
    out_lines.append("")
    if ok_min < 70:
        out_lines.append(
            f"  结论: {worst['name']} 跟踪完整度仅 {worst['ok_ratio_pct']:.0f}%"
            f"(最大间隙 {worst['max_gap_s']:.1f}s), 压缩过大导致轨迹不可信, 不建议使用。")
        out_lines.append(
            f"  可用档位: {best['name']} 完整度最高 ({best['ok_ratio_pct']:.0f}%), 建议采用。")
    elif ok_min < 90:
        out_lines.append(
            f"  结论: 各档位跟踪完整度基本可用 (最低 {worst['name']} {worst['ok_ratio_pct']:.0f}%), "
            "存在轻微跟踪损失, 建议在可接受误差内选择压缩率更高的一档。")
    else:
        out_lines.append(
            f"  结论: 各档位跟踪完整度良好 (最低 {worst['name']} {worst['ok_ratio_pct']:.0f}%), "
            "压缩对 SLAM 影响有限。")
    out_lines.append("")


def pairwise_metrics(ta, tb, deltas):
    """ta 为基准(参考), tb 为被测。返回汇总 dict。

    只在两侧都有有效位姿的时间点上统计(屏蔽跟踪间隙, 见 valid_mask)。
    """
    end = min(ta["ts"][-1], tb["ts"][-1])
    grid = np.arange(0.0, end, 0.02)
    valid = valid_mask(ta["ts"], grid) & valid_mask(tb["ts"], grid)
    pos_dev, rot_dev = compute_deviation(ta, tb, grid, valid)
    ate = compute_ate(ta, tb, grid, valid)
    res = {
        "pos_dev_mean_mm": float(np.nanmean(pos_dev) * 1000),
        "pos_dev_max_m": float(np.nanmax(pos_dev)),
        "rot_dev_mean_deg": float(np.nanmean(rot_dev)),
        "rot_dev_max_deg": float(np.nanmax(rot_dev)),
        "ate_rmse_m": float(np.sqrt(np.nanmean(ate ** 2))),
        "ate_mean_m": float(np.nanmean(ate)),
        "ate_max_m": float(np.nanmax(ate)),
    }
    for d in deltas:
        t_err, r_err = compute_rpe_time(ta, tb, grid, d, valid)
        res[f"rpe_t_{d}s_mean_mm"] = float(np.mean(t_err) * 1000)
        res[f"rpe_t_{d}s_max_mm"] = float(np.max(t_err) * 1000)
        res[f"rpe_r_{d}s_mean_deg"] = float(np.mean(r_err))
        res[f"rpe_r_{d}s_max_deg"] = float(np.max(r_err))
    res["_grid"] = grid
    res["_valid"] = valid
    res["_pos_dev"] = pos_dev
    res["_rot_dev"] = rot_dev
    res["_ate"] = ate
    return res


def main():
    p = argparse.ArgumentParser(
        description="多档 episode SLAM 轨迹对比 (RPE/ATE/逐帧偏差)")
    p.add_argument("episodes", nargs="+",
                   help="episode 目录(1 个以上); 每目录下需含 results/trajectory/")
    p.add_argument("--ref", default=None,
                   help="基准 episode 目录; 不指定则两两对比")
    p.add_argument("--traj", choices=["online", "tum", "body"], default="online",
                   help="对比用的轨迹文件 (默认 online = trajectory_online_style.txt)")
    p.add_argument("--deltas", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                   help="RPE 时间增量(秒), 默认 0.5 1.0 2.0")
    p.add_argument("--outdir", default="",
                   help="输出根目录(每次运行在其下新建 run_<时间戳>/ 子目录; 默认 crf_report)")
    p.add_argument("--plot", action="store_true", help="生成可视化图")
    p.add_argument("--title", default="Episode SLAM 对比 (视频压缩档)",
                   help="图标题前缀")
    args = p.parse_args()

    # ── 解析 episode 根 ──
    roots = []
    for ep in args.episodes:
        root = resolve_episode_root(os.path.abspath(ep))
        if root not in roots:
            roots.append(root)
    if len(roots) < 1:
        sys.exit("至少需要 1 个 episode 目录")
    if args.ref is not None:
        ref_root = resolve_episode_root(os.path.abspath(args.ref))

    # ── 加载轨迹 ──
    trajs = [load_trajectory(r, args.traj) for r in roots]
    names = [tr["name"] for tr in trajs]

    # ── 确定对比对 ──
    pairs = []
    if args.ref is not None:
        if ref_root not in roots:
            roots = [ref_root] + roots
            trajs = [load_trajectory(ref_root, args.traj)] + trajs
            names = [trajs[0]["name"]] + names
        idx_ref = roots.index(ref_root)
        pairs = [(idx_ref, k) for k in range(len(roots)) if k != idx_ref]
        ref_name = names[idx_ref]
    else:
        pairs = [(i, j) for i in range(len(roots)) for j in range(i + 1, len(roots))]
        ref_name = None

    if not pairs:
        print("仅一个目录, 跳过对比; 输出单档质量。")

    # ── 输出目录: 每次运行独立时间戳子目录, 历史结果不被覆盖 ──
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    base_out = args.outdir or "crf_report"
    outdir = os.path.join(base_out, f"run_{run_ts}")
    k = 1
    while os.path.exists(outdir):          # 同秒内连跑时追加序号
        k += 1
        outdir = os.path.join(base_out, f"run_{run_ts}_{k}")
    os.makedirs(outdir, exist_ok=True)
    report_path = os.path.join(outdir, f"report_{run_ts}.txt")

    lines = []
    lines.append("=" * 72)
    lines.append(f"  {args.title}")
    lines.append(f"  轨迹文件: {args.traj} ({trajs[0]['path']})")
    lines.append(f"  对比模式: {'以 ' + ref_name + ' 为基准' if ref_name else '两两对比'}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("[各档质量]")
    qualities = build_report(episodes=roots, trajs=trajs, ref_name=ref_name, out_lines=lines)
    lines.append("")

    metrics_map = {}
    if pairs:
        lines.append("[轨迹对比] (基准 → 被测; 位移 RPE 单位 mm, 旋转 RPE 单位 °)")
        lines.append(f"  {'基准 → 被测':<42s} {'Δ=0.5s':>10s} {'Δ=1.0s':>10s} {'Δ=2.0s':>10s} {'ATE RMSE':>10s}")
        for i, j in pairs:
            res = pairwise_metrics(trajs[i], trajs[j], args.deltas)
            key = f"{names[i]} → {names[j]}"
            metrics_map[key] = (i, j, res)
            rpe_line = "  ".join(
                f"t {res[f'rpe_t_{d}s_mean_mm']:>4.1f}mm/r {res[f'rpe_r_{d}s_mean_deg']:>4.2f}°"
                for d in args.deltas
            )
            lines.append(f"  {key:<42s} {rpe_line}   {res['ate_rmse_m']*1000:>6.1f}mm")
        lines.append("")
        lines.append("  最大偏差(公共时间轴上, 未对齐):")
        for key, (i, j, res) in metrics_map.items():
            lines.append(
                f"    {key:<42s} 位置 {res['pos_dev_mean_mm']:>6.1f}mm(均) / "
                f"{res['pos_dev_max_m']*1000:>6.0f}mm(峰) | 旋转 "
                f"{res['rot_dev_mean_deg']:>5.2f}°(均) / {res['rot_dev_max_deg']:>5.1f}°(峰)")

    # ── [总结] 摘要段 ──
    summary_delta = args.deltas[0]
    build_summary(qualities, names, metrics_map, ref_name, summary_delta, lines)

    text = "\n".join(lines) + "\n"
    print(text)
    with open(report_path, "w") as f:
        f.write(text)
    print(f"报告: {report_path}")

    # 另存一份固定名 summary.txt, 便于直接打开看结论(位于本次运行的 run_<时间戳>/ 内)
    summary_path = os.path.join(outdir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(text)
    print(f"摘要: {summary_path}")
    print(f"本次运行输出目录: {outdir}")

    # ── 绘图 ──
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.font_manager as _fm
        import matplotlib.pyplot as plt
        # 中文字体: 优先 Noto Sans CJK, 找不到则退回 DejaVu (中文显示为方框)
        _available = {f.name for f in _fm.fontManager.ttflist}
        _cjk = next((n for n in [
            "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei",
            "AR PL UMing CN", "AR PL UKai CN",
        ] if n in _available), None)
        if _cjk:
            plt.rcParams["font.sans-serif"] = [_cjk, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
        else:
            print("[WARN] 未找到 CJK 字体, 图中中文将显示为方框")
        colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
        c = {n: colors[k % len(colors)] for k, n in enumerate(names)}

        grid_common = None
        if pairs:
            i0, j0, res0 = next(iter(metrics_map.values()))
            grid_common = res0["_grid"]

        # 1) 3D 叠加
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for tr in trajs:
            xyz = tr["xyz"]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=1.2,
                    color=c[tr["name"]], label=tr["name"])
        ax.set_xlabel("X(m)"); ax.set_ylabel("Y(m)"); ax.set_zlabel("Z(m)")
        ax.legend(fontsize=8)
        ax.set_title(f"{args.title} — 3D 轨迹叠加")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "traj_3d.png"), dpi=150)

        # 2) 俯视 / 侧视
        for view, (u, v) in {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}.items():
            fig, ax = plt.subplots(figsize=(9, 6))
            for tr in trajs:
                xyz = tr["xyz"]
                ax.plot(xyz[:, u], xyz[:, v], lw=1.2, color=c[tr["name"]],
                        label=tr["name"])
            ax.set_xlabel(f"{'XYZ'[u]}(m)"); ax.set_ylabel(f"{'XYZ'[v]}(m)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.legend(fontsize=8); ax.grid(True)
            ax.set_title(f"{args.title} — {view} 俯视/侧视")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"traj_{view.lower()}.png"), dpi=150)

        # 3) XYZ vs 时间
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for tr in trajs:
            xyz0 = tr["xyz"][0]
            for k, ax in enumerate(axes):
                ax.plot(tr["ts"], tr["xyz"][:, k] - xyz0[k], lw=0.8,
                        color=c[tr["name"]], label=tr["name"])
                ax.set_ylabel(["X(m)", "Y(m)", "Z(m)"][k]); ax.grid(True)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel("t(s)")
        fig.suptitle(f"{args.title} — 位姿分量 vs 时间 (相对各自起点)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "pos_vs_time.png"), dpi=150)

        # 4) 逐帧偏差 vs 时间
        if pairs:
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            for key, (i, j, res) in metrics_map.items():
                g = res["_grid"]
                axes[0].plot(g, res["_pos_dev"] * 1000, lw=0.8, color=c[names[j]],
                             label=key)
                axes[1].plot(g, res["_rot_dev"], lw=0.8, color=c[names[j]],
                             label=key)
            axes[0].set_ylabel("位置偏差(mm)"); axes[0].grid(True); axes[0].legend(fontsize=8)
            axes[1].set_ylabel("旋转偏差(°)"); axes[1].grid(True); axes[1].legend(fontsize=8)
            axes[1].set_xlabel("t(s)")
            fig.suptitle(f"{args.title} — 公共时间轴逐帧偏差")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "dev_vs_time.png"), dpi=150)

            # 5) RPE 柱状图 (取最接近 1.0s 的增量)
            bar_delta = min(args.deltas, key=lambda d: abs(d - 1.0))
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            keys = list(metrics_map.keys())
            t_means = [metrics_map[k][2][f"rpe_t_{bar_delta}s_mean_mm"] for k in keys]
            r_means = [metrics_map[k][2][f"rpe_r_{bar_delta}s_mean_deg"] for k in keys]
            axes[0].bar(range(len(keys)), t_means, color="#1f77b4")
            axes[0].set_xticks(range(len(keys))); axes[0].set_xticklabels(keys, rotation=25, ha="right", fontsize=7)
            axes[0].set_ylabel(f"RPE 平移(mm) @Δ={bar_delta}s"); axes[0].grid(True, axis="y")
            axes[1].bar(range(len(keys)), r_means, color="#d62728")
            axes[1].set_xticks(range(len(keys))); axes[1].set_xticklabels(keys, rotation=25, ha="right", fontsize=7)
            axes[1].set_ylabel(f"RPE 旋转(°) @Δ={bar_delta}s"); axes[1].grid(True, axis="y")
            fig.suptitle(f"{args.title} — 平均 RPE (Δ={bar_delta}s)")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "rpe_bar.png"), dpi=150)

        # 6) 跟踪完整度
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ok_ratios = []
        for tr in trajs:
            report = load_run_report(tr["root"])
            states_ts, states = load_tracking_states(tr["root"])
            q = trajectory_quality(tr, report, states)
            ok_ratios.append(q["ok_ratio_pct"])
        ax.bar(range(len(names)), ok_ratios, color=[c[n] for n in names])
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("跟踪完整度(%)"); ax.set_ylim(0, 105)
        ax.grid(True, axis="y")
        for k, v in enumerate(ok_ratios):
            ax.text(k, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
        ax.set_title(f"{args.title} — 跟踪完整度")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "tracking_ok.png"), dpi=150)

        print(f"图已保存到: {outdir}/ (traj_3d/traj_xy/traj_xz/pos_vs_time/dev_vs_time/rpe_bar/tracking_ok)")


if __name__ == "__main__":
    main()

"""摄像头实时手指/关节识别 Demo — MMPose 版（YOLO-World + RTMPose）

专为深色手套场景设计。MediaPipe 在手套上完全失效（0/40 检出），
MMPose 检出 40/40，关键点贴合张开的手掌。

用法:
    python hand_demo_mmpose.py              # 默认摄像头 + CUDA
    python hand_demo_mmpose.py --camera 1   # 指定摄像头
    python hand_demo_mmpose.py --cpu        # 强制 CPU（慢很多）
    python hand_demo_mmpose.py --hands 2    # 最多识别几只手
    python hand_demo_mmpose.py --realsense  # RealSense D435 颜色流（需 pyrealsense2）

快捷键:
    q / ESC  退出
    a        切换关节角度数值显示
    s        切换骨架/关键点显示
    b        切换检测框显示
    h        切换帮助信息
    c        抓拍当前帧（存原图 + 检测 + 关键点预标注）
    r        连续采集开关
    m        只在漏检时自动采集（最需要的就是这类难样本）
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# ── 复用同目录模块 ──────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

import hand_common as hc
import world_detector as wd

MODEL_PATH = os.path.join(_HERE, wd.DEFAULT_MODEL)


# ── 自定义训练检测器包装 ──────────────────────────
class YOLOWrapper:
    """自己训的 YOLO 手部检测器"""

    def __init__(self, weights, device):
        from ultralytics import YOLO
        self.m = YOLO(weights)
        self.device = device
        self.name = f"YOLO ({os.path.basename(weights)})"

    def __call__(self, frame, conf):
        r = self.m(frame, conf=conf, device=self.device, verbose=False)[0]
        return [list(map(float, b)) + [float(c)]
                for b, c in zip(r.boxes.xyxy.cpu().numpy(),
                                r.boxes.conf.cpu().numpy())]


# ═══════════════════════════════════════════════════════════════════
# 数据采集
# ═══════════════════════════════════════════════════════════════════

class Recorder:
    """把原始帧连同当帧的检测结果一起存下来。

    存的图像是镜像后、未画骨架的干净原图；关键点作为预标注写进 labels.jsonl。

    目录结构:
        captures1/20260803-114530/
            session.json        采集参数
            frames/000001.jpg
            labels.jsonl        每帧一行，含 21 个关键点与 bbox
    """

    def __init__(self, root, meta, fmt="jpg", quality=95):
        self.root, self.meta, self.fmt, self.quality = root, meta, fmt, quality
        self.count = 0
        self._dir = None
        self._fp = None

    def _ensure_dir(self):
        if self._dir:
            return
        self._dir = os.path.join(self.root, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(os.path.join(self._dir, "frames"), exist_ok=True)
        with open(os.path.join(self._dir, "session.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        self._fp = open(os.path.join(self._dir, "labels.jsonl"), "a", encoding="utf-8")
        print(f"采集目录: {self._dir}")

    @staticmethod
    def _hand_record(kpts, scores, bbox):
        """kpts: (21,2) 像素坐标; scores: (21,) 置信度"""
        return {
            "landmarks": [[round(float(k[0]), 2), round(float(k[1]), 2)] for k in kpts],
            "kpt_scores": [round(float(s), 4) for s in scores],
            "bbox": [round(float(v), 1) for v in bbox],
        }

    def save(self, frame, boxes, kpts_list, scores_list, h, w, trigger):
        self._ensure_dir()
        self.count += 1
        rel = f"frames/{self.count:06d}.{self.fmt}"
        params = [cv2.IMWRITE_JPEG_QUALITY, self.quality] if self.fmt == "jpg" else []
        cv2.imwrite(os.path.join(self._dir, rel), frame, params)

        hands = [self._hand_record(kpts_list[i], scores_list[i], boxes[i])
                 for i in range(len(boxes))]
        self._fp.write(json.dumps({
            "file": rel, "width": w, "height": h,
            "trigger": trigger, "hands": hands,
        }, ensure_ascii=False) + "\n")
        self._fp.flush()
        return self.count

    def close(self):
        if self._fp:
            self._fp.close()
            print(f"共保存 {self.count} 帧到 {self._dir}")


# ═══════════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="手指/关节实时识别 — MMPose (RTMPose)")
    ap.add_argument("--camera", type=int, default=0, help="摄像头序号")
    ap.add_argument("--source", default=None,
                    help="视频文件路径（指定后优先于摄像头）")
    ap.add_argument("--hands", type=int, default=2, help="最多识别几只手")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--no-mirror", action="store_true", help="不做左右镜像")
    ap.add_argument("--realsense", action="store_true",
                    help="用 RealSense D435 颜色流（需 pip install pyrealsense2，--camera 失效）")
    ap.add_argument("--rs-serial", default=None,
                    help="RealSense 设备序列号（多台 D435 时指定）")

    # 检测器参数
    ap.add_argument("--detector", default="world",
                    help="'world' 用 YOLO-World 文本提示 / 或给自己训的 .pt 路径")
    wd.add_args(ap, max_boxes_default=2)
    ap.set_defaults(weights=MODEL_PATH)

    ap.add_argument("--det-device", default=None,
                    help="检测器 device（默认自动: CUDA>MPS>CPU）")
    ap.add_argument("--det-conf", type=float, default=0.45,
                    help="训练模型检测置信度阈值。默认 0.45（实测 0.05 在无手套"
                         "场景满天飞框；误检最终由关键点有效性门兜底）")
    ap.add_argument("--pose-device", default="cpu",
                    help="关键点 device（默认 cpu: ONNX GPU 需系统 CUDA Toolkit）")
    ap.add_argument("--kpt-thr", type=float, default=0.3,
                    help="关键点置信度阈值（低于则画空心）")
    ap.add_argument("--cpu", action="store_true",
                    help="强制全部 CPU（等价于 --det-device cpu --pose-device cpu）")

    # 采集参数
    ap.add_argument("--capture-dir", default="captures1", help="采集数据保存目录")
    ap.add_argument("--capture-format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--capture-quality", type=int, default=95)
    ap.add_argument("--capture-interval", type=float, default=0.5,
                    help="连续采集/漏检采集的最小间隔秒数")
    ap.add_argument("--output", default=None,
                    help="输出视频路径（处理视频时可用）")
    ap.add_argument("--data-out", default=None,
                    help="输出检测数据 JSONL 文件（每帧的框+关键点+置信度）")

    args = ap.parse_args()

    if args.cpu:
        det_device = pose_device = "cpu"
    else:
        det_device = args.det_device or hc.auto_device()
        pose_device = args.pose_device

    if not os.path.exists(args.weights):
        sys.exit(f"找不到检测器权重: {args.weights}\n"
                 "首次运行会自动下载，或手动放到 rtmpose/ 目录。")

    # ── 初始化 ──────────────────────────────────────────
    if args.realsense:
        try:
            cap = hc.RSCapture(args.rs_serial, args.width, args.height)
        except ImportError:
            sys.exit("未安装 pyrealsense2：pip install pyrealsense2")
        except Exception as e:
            sys.exit(f"无法打开 RealSense（未连接或被其他程序占用）: {e}")
        live_source = True
    elif args.source:
        cap = cv2.VideoCapture(args.source)
        live_source = False
    else:
        cap = cv2.VideoCapture(args.camera)
        live_source = True
    if not cap.isOpened():
        sys.exit(f"无法打开: {args.source or f'摄像头 {args.camera}'}")
    if live_source:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # 输出视频写入器
    out_writer = None
    if args.output:
        out_fps = cap.get(cv2.CAP_PROP_FPS)
        if out_fps <= 0:
            out_fps = 30
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.output, fourcc, out_fps, (out_w, out_h))
        print(f"输出视频: {args.output}  ({out_w}x{out_h} @{out_fps:.1f}fps)")

    # 检测数据输出
    data_fp = None
    if args.data_out:
        data_fp = open(args.data_out, "w", encoding="utf-8")
        print(f"输出数据: {args.data_out}")

    if args.detector == "world":
        if not os.path.exists(args.weights):
            sys.exit(f"找不到检测器权重: {args.weights}\n"
                     "首次运行会自动下载，或手动放到 rtmpose/ 目录。")
        print("加载 YOLO-World 检测器...")
        det = wd.WorldDetector(args.weights, args.prompt, args.imgsz, det_device)
    else:
        print(f"加载自定义检测器: {args.detector}")
        det = YOLOWrapper(args.detector, det_device)
    print(f"  检测器: {det.name}")

    print("加载 RTMPose 关键点模型...")
    pose = hc.build_pose(pose_device)
    print(f"  关键点: RTMPose-m hand5 @{pose_device}")

    recorder = Recorder(
        args.capture_dir,
        {"camera": args.camera, "width": args.width, "height": args.height,
         "mirrored": not args.no_mirror, "num_hands": args.hands,
         "detector": det.name, "pose": "RTMPose-m hand5",
         "note": "frames 为镜像后、未画骨架的原图；hands 含关键点和检测框预标注"},
        args.capture_format, args.capture_quality)

    print(f"  分辨率: {args.width}x{args.height} | "
          f"检测@{det_device} + 关键点@{pose_device}")
    print("摄像头已开启。q 退出 | a 角度 | s 骨架 | b 检测框 | h 帮助")
    print("采集: c 抓拍 | r 连续采集 | m 只存漏检帧")

    # ── 状态变量 ────────────────────────────────────────
    show_angles = show_skeleton = show_box = show_help = True
    show_ui = False            # UI 浮窗：FPS、手部信息等（i 键切换）
    paused = False             # 视频暂停（空格键切换）
    recording = False
    capture_miss = False
    last_capture = 0.0
    hit_window = deque(maxlen=100)
    frame_idx = 0              # 帧序号（用于数据输出）
    fps, last_t = 0.0, time.perf_counter()

    # 手部身份追踪器（替换原来的裸状态变量，解决 ID 交换导致的关键点粘连）
    from hand_tracker import HandTracker
    tracker = HandTracker(
        max_hands=args.max_boxes,
        iou_match_thr=0.3,
        lost_timeout=3,         # 丢失 3 帧后才删除 track（解决检测闪断）
        movement_thresh=3,      # 5→3 px，更灵敏
        skip_timeout=10,        # 20→10 帧，减少粘滞
        box_smooth_alpha=0.7,   # EMA 平滑
    )
    KPT_FREEZE_THR = 0.2       # 逐点置信度冻结阈值：弯曲手指上低于此值的点冻结
    confirm = {}                # track_id -> 连续通过关键点有效性门的帧数（误检抑制）

    # 关键点 → 手指映射（0=腕, 1-4=拇指, 5-8=食指, 9-12=中指, 13-16=无名指, 17-20=小指）
    _kpt_to_finger = {}
    for _fn, (_ids, _) in hc.FINGERS.items():
        for _idx in _ids:
            _kpt_to_finger[_idx] = _fn
    win = "Hand & Finger Joints - MMPose (RTMPose)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if not live_source:
                        print("视频播放完毕。按 q 退出。")
                        paused = True
                        # 等待用户按键
                        key = cv2.waitKey(0) & 0xFF
                        if key in (ord("q"), 27):
                            break
                        continue
                    else:
                        print("读取摄像头帧失败，退出。")
                        break

            if not ok and live_source:
                break

            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    paused = not paused
                continue

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            # 上下翻转：UVC 相机倒装校正。RealSense 颜色帧原生直立
            # （与主程序 _open_d435 一致：set_frame(flip_vertical=False)），不翻。
            if not args.realsense:
                frame = cv2.flip(frame, 0)
            h_img, w_img = frame.shape[:2]

            # 留一份干净原图给采集用
            clean = frame.copy()
            canvas = frame.copy()

            # ── 检测 ──────────────────────────────────────
            t_det = time.perf_counter()
            if args.detector == "world":
                det_boxes, det_confs = det(clean, args.conf, args.max_boxes,
                                           (w_img, h_img))
                boxes_raw = [list(b) for b in det_boxes]
            else:
                raw = det(clean, args.det_conf)
                bb = np.array([r[:4] for r in raw], np.float32).reshape(-1, 4)
                cc = np.array([r[4] for r in raw], np.float32)
                boxes_raw = [list(b) for b in
                             wd.WorldDetector.postprocess(
                                 bb, cc, w_img, h_img, args.max_boxes)[0]]
            t_det = (time.perf_counter() - t_det) * 1000

            # ── 追踪：IoU 匹配到稳定手部 ID ───────────────
            tracker.update_detections(boxes_raw)

            # ── 关键点推理（只对需要的手跑）───────────────
            boxes_for_pose, pose_indices = tracker.get_boxes_for_pose()
            t_pose = 0.0
            if boxes_for_pose:
                t_pose = time.perf_counter()
                raw_kpts, raw_scores = pose(clean, bboxes=boxes_for_pose)
                tracker.update_pose_results(pose_indices,
                                            np.array(raw_kpts),
                                            np.array(raw_scores))
                t_pose = (time.perf_counter() - t_pose) * 1000

            # ── 获取身份稳定的结果 ────────────────────────
            boxes, kpts, scores, track_ids = tracker.get_results()

            # ── 无手套误检抑制 ────────────────────────────
            # 背景/手套状物体的框，RTMPose 解不出像样的 21 点（kpt 均值 <0.45
            # 或高置信点 <15）→ 判定无手套：不画框、不解算关键点。
            # 连续 2 帧有效才显示，掐掉单帧闪框（见 hand_common.pose_is_glove）。
            if kpts is not None and len(kpts):
                keep = []
                for i, tid in enumerate(track_ids):
                    ok = hc.pose_is_glove(kpts[i], scores[i], boxes[i])
                    confirm[tid] = confirm.get(tid, 0) + 1 if ok else 0
                    if confirm[tid] >= 2:
                        keep.append(i)
                confirm = {t: n for t, n in confirm.items() if t in track_ids}
                if keep:
                    boxes = [boxes[i] for i in keep]
                    kpts = kpts[keep]
                    scores = scores[keep]
                    track_ids = [track_ids[i] for i in keep]
                else:
                    boxes, kpts, scores, track_ids = [], None, None, []

            # ── 输出检测数据 ────────────────────────────
            if data_fp and kpts is not None:
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                hands_data = []
                for i in range(len(kpts)):
                    hands_data.append({
                        "track_id": track_ids[i],
                        "bbox": [round(float(v), 1) for v in boxes[i]],
                        "landmarks": [[round(float(k[0]), 2), round(float(k[1]), 2)]
                                      for k in kpts[i][:21]],
                        "kpt_scores": [round(float(s), 4) for s in scores[i][:21]],
                    })
                data_fp.write(json.dumps({
                    "frame": frame_idx,
                    "timestamp_ms": round(ts_ms, 1),
                    "hands": hands_data,
                }, ensure_ascii=False) + "\n")
                data_fp.flush()

            # ── 遮挡判断 + 逐点置信度门控（按手指粒度）───
            # 1) 遮挡：≥90% 关键点置信度很低 → 整只手冻结
            # 2) 否则按手指粒度：弯曲手指的低置信度点才冻结
            if kpts is not None:
                OCCLUSION_RATIO = 0.9   # 低置信度点占比超过此值判定为遮挡

                for t_idx in range(len(kpts)):
                    # 统计低置信度关键点数量（用 KPT_FREEZE_THR 同一阈值）
                    low_kpt_count = int(np.sum(scores[t_idx][:21] < KPT_FREEZE_THR))
                    occluded = (low_kpt_count / 21 >= OCCLUSION_RATIO)

                    if occluded:
                        # 整只手被遮挡 → 全部冻结
                        for j in range(21):
                            good = tracker.get_last_good(t_idx, j)
                            if good is not None:
                                kpts[t_idx][j] = good.copy()
                    else:
                        ang_check = hc.compute_joint_angles(kpts[t_idx][:21])
                        ext = hc.count_extended_fingers(ang_check)

                        for j in range(21):
                            finger = _kpt_to_finger.get(j)         # None for wrist
                            finger_ok = (finger is None            # wrist → always update
                                         or finger in ext          # extended → update
                                         or scores[t_idx][j] >= KPT_FREEZE_THR)

                            if finger_ok:
                                tracker.update_last_good(t_idx, j, kpts[t_idx][j])
                            else:
                                good = tracker.get_last_good(t_idx, j)
                                if good is not None:
                                    kpts[t_idx][j] = good.copy()
            else:
                tracker.clear()

            hit_window.append(len(boxes))

            # ── 采集 ─────────────────────────────────────
            now_t = time.perf_counter()
            want = recording or (capture_miss and not boxes)
            if want and now_t - last_capture >= args.capture_interval:
                if kpts is not None:
                    recorder.save(clean, boxes, kpts, scores,
                                  h_img, w_img,
                                  "record" if recording else "miss")
                elif boxes:
                    recorder.save(clean, boxes, [], [],
                                  h_img, w_img,
                                  "record" if recording else "miss")
                last_capture = now_t

            # ── 绘制 ─────────────────────────────────────
            if show_box:
                for b in boxes:
                    cv2.rectangle(canvas, (int(b[0]), int(b[1])),
                                  (int(b[2]), int(b[3])), (0, 255, 255), 2)

            info_lines = []
            if kpts is not None:
                for i, p in enumerate(kpts):
                    ang = hc.compute_joint_angles(p[:21])
                    if show_skeleton:
                        hc.draw_hand(canvas, p, ang, show_angles=show_angles,
                                     kpt_scores=scores[i], thr=args.kpt_thr)
                    if show_ui:
                        ext = hc.count_extended_fingers(ang)
                        mean_score = float(scores[i][:21].mean())
                        info_lines.append(
                            (f"[hand {track_ids[i]}] open {len(ext)}/5  kpt {mean_score:.2f}",
                             (255, 255, 255)))
                        for f, (_ids, col) in hc.FINGERS.items():
                            j = "IP" if f == "Thumb" else "PIP"
                            state = "OPEN" if f in ext else "BEND"
                            info_lines.append(
                                (f"  {f:<7}{state}  {ang[(f, j)]:5.0f}deg", col))

            # ── UI 面板 ─────────────────────────────────
            if show_ui:
                now = time.perf_counter()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt

                hit_rate = 100.0 * sum(1 for n in hit_window if n) / len(hit_window)
                avg_hands = sum(hit_window) / len(hit_window) if hit_window else 0

                head = [
                    (f"FPS {fps:4.1f}   hands {len(boxes)}", (0, 255, 180)),
                    (f"det {t_det:5.1f}ms + pose {t_pose:5.1f}ms", (120, 220, 255)),
                    (f"last100f: {hit_rate:5.1f}%  avg {avg_hands:.2f} hands",
                     (0, 255, 0) if hit_rate > 80 else (0, 200, 255) if hit_rate > 40 else (0, 120, 255)),
                ]
                if recording or capture_miss or recorder.count:
                    tag = "REC" if recording else ("MISS-CAP" if capture_miss else "saved")
                    head.append((f"[{tag}] captured {recorder.count}",
                                 (0, 80, 255) if (recording or capture_miss) else (180, 180, 180)))
                if show_help:
                    head.append(("q:quit spc:pause a:ang s:skel b:box i:ui h:help", (180, 180, 180)))
                    head.append(("c:shot r:rec m:miss-only", (180, 180, 180)))
                hc.draw_panel(canvas, 10, 10, head)
                if info_lines:
                    hc.draw_panel(canvas, 10, 10 + 22 * len(head) + 24, info_lines)

            cv2.imshow(win, canvas)

            if out_writer:
                out_writer.write(canvas)

            frame_idx += 1

            # ── 键盘 ─────────────────────────────────────
            delay = 1 if live_source else 1   # 实时 1ms，视频也用正常速度
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("a"):
                show_angles = not show_angles
            elif key == ord("s"):
                show_skeleton = not show_skeleton
            elif key == ord("b"):
                show_box = not show_box
            elif key == ord("i"):
                show_ui = not show_ui
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("c"):
                if kpts is not None:
                    recorder.save(clean, boxes, kpts, scores,
                                  h_img, w_img, "manual")
                elif boxes:
                    recorder.save(clean, boxes, [], [],
                                  h_img, w_img, "manual")
                print(f"已抓拍第 {recorder.count} 帧（检出 {len(boxes)} 只手）")
            elif key == ord("r"):
                recording = not recording
                if recording:
                    capture_miss = False
                print(f"连续采集 {'开' if recording else '关'}")
            elif key == ord("m"):
                capture_miss = not capture_miss
                if capture_miss:
                    recording = False
                print(f"漏检采集 {'开' if capture_miss else '关'}")
    finally:
        cap.release()
        if out_writer:
            out_writer.release()
            print(f"输出视频已保存: {args.output}")
        if data_fp:
            data_fp.close()
            print(f"检测数据已保存: {args.data_out}  ({frame_idx} 帧)")
        cv2.destroyAllWindows()
        recorder.close()


if __name__ == "__main__":
    main()

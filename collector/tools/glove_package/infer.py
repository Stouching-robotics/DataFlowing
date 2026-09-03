"""RTMPose 实时推理：检测器出框 -> RTMPose 出 21 个关键点 -> 画骨架和关节角度。

检测器可以换成自己训的（--detector runs/hand_det/weights/best.pt），
关键点模型一直用 RTMPose 官方的 hand5 权重，不需要自己训。

用法：
    python infer.py                                   # 摄像头 + YOLO-World（免训练，默认）
    python infer.py --detector runs/hand_det/weights/best.pt
    python infer.py --detector runs/hand_det/weights/best.pt --realsense   # RealSense D435
    python infer.py --source ../captures/20260803-144759/frames   # 跑一个图片目录
    python infer.py --source clip.mp4

快捷键：
    q / ESC  退出        a  关节角度      s  骨架
    b        检测框      h  帮助          空格  暂停（图片/视频源）
"""

import argparse
import glob
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

import hand_common as hc
import world_detector as wd

_HERE = os.path.dirname(os.path.abspath(__file__))
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


class RTMDetWrapper:
    """rtmlib 自带的 RTMDet-nano 手部检测器（在深色手套上会失效，仅作基线对照）"""

    name = "RTMDet-nano (官方)"

    def __init__(self, device):
        from rtmlib import RTMDet
        url = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
               "rtmdet_nano_8xb32-300e_hand-267f9c8f.zip")
        self.m = RTMDet(url, model_input_size=(320, 320),
                        backend="onnxruntime", device=device)

    def __call__(self, frame, conf):
        return [list(map(float, b)) + [1.0] for b in self.m(frame)]   # rtmlib 不返回分数


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
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]


def frame_source(source, width, height, realsense=False, rs_serial=None):
    """统一成一个产出 BGR 帧的生成器；返回 (生成器, 是否实时源)"""
    if realsense:
        try:
            cap = hc.RSCapture(rs_serial, width, height)
        except ImportError:
            sys.exit("未安装 pyrealsense2：pip install pyrealsense2")
        except Exception as e:
            sys.exit(f"无法打开 RealSense（未连接或被其他程序占用）: {e}")

        def gen():
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield f
            cap.release()
        return gen(), True

    if os.path.isdir(source):
        files = sorted(f for f in glob.glob(os.path.join(source, "*"))
                       if f.lower().endswith(IMG_EXT))
        if not files:
            sys.exit(f"{source} 下没有图片")

        def gen():
            for f in files:
                im = cv2.imread(f)
                if im is not None:
                    yield im
        return gen(), False

    if os.path.isfile(source):
        cap = cv2.VideoCapture(source)

        def gen():
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield f
            cap.release()
        return gen(), False

    cap = cv2.VideoCapture(int(source))
    if not cap.isOpened():
        sys.exit(f"打不开摄像头 {source}。macOS 需在 系统设置 > 隐私与安全性 > 摄像头 授权终端。")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def gen():
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f
        cap.release()
    return gen(), True


def main():
    ap = argparse.ArgumentParser(description="RTMPose 手部关键点实时推理")
    ap.add_argument("--source", default="0", help="摄像头序号 / 图片目录 / 视频文件")
    ap.add_argument("--realsense", action="store_true",
                    help="用 RealSense D435 颜色流（需 pyrealsense2），忽略 --source")
    ap.add_argument("--rs-serial", default=None, help="D435 序列号（多台时指定）")
    ap.add_argument("--detector", default="world",
                    help="'world' 用 YOLO-World 文本提示(免训练,推荐) / "
                         "'rtmdet' 官方检测器 / 或给一个自己训的 YOLO .pt 路径")
    wd.add_args(ap, max_boxes_default=2)   # 与 auto_label.py 共用同一套默认值
    ap.add_argument("--det-device", default=None,
                    help="检测器 device。默认自动检测（CUDA > CPU）")
    ap.add_argument("--det-conf", type=float, default=0.45,
                    help="训练模型检测置信度阈值。默认 0.45（实测 0.05 在无手套"
                         "场景满天飞框；误检最终由关键点有效性门兜底）")
    ap.add_argument("--pose-device", default="cpu",
                    help="关键点模型 device。默认 cpu（ONNX Runtime GPU 依赖系统 CUDA Toolkit，不现实）")
    ap.add_argument("--kpt-thr", type=float, default=0.3, help="关键点置信度阈值（低于则画空心）")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--no-mirror", action="store_true", help="不做左右镜像")
    args = ap.parse_args()

    # 默认模型路径相对于脚本所在目录
    if args.detector == "world" and args.weights == wd.DEFAULT_MODEL:
        args.weights = os.path.join(_HERE, args.weights)

    det_device = args.det_device or hc.auto_device()
    pose_device = args.pose_device
    if args.detector == "world":
        det = wd.from_args(args, det_device)
    elif args.detector == "rtmdet":
        det = RTMDetWrapper(det_device)
    else:
        det = YOLOWrapper(args.detector, det_device)
    pose = hc.build_pose(pose_device)

    frames, live = frame_source(args.source, args.width, args.height,
                                args.realsense, args.rs_serial)
    show_angles = show_skel = show_box = show_help = True
    paused = False
    hit = deque(maxlen=100)
    fps, last_t = 0.0, time.perf_counter()
    det_skip = max(0, args.det_skip)
    frame_count = 0
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
    KPT_FREEZE_THR = 0.3        # 逐点置信度冻结阈值：弯曲手指上低于此值的点冻结
    confirm = {}                 # track_id -> 连续通过关键点有效性门的帧数（误检抑制）

    # 关键点 → 手指映射（0=腕, 1-4=拇指, 5-8=食指, 9-12=中指, 13-16=无名指, 17-20=小指）
    _kpt_to_finger = {}
    for _fn, (_ids, _) in hc.FINGERS.items():
        for _idx in _ids:
            _kpt_to_finger[_idx] = _fn
    track_ids = []               # 当前帧活跃 track 的稳定 ID
    win = "RTMPose Hand"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"检测器: {det.name} | 关键点: RTMPose-m hand5 @{pose_device}")
    if det_skip:
        print(f"检测跳过: 每 {det_skip + 1} 帧检测一次（CPU 加速）")

    frame = None
    try:
        while True:
            if not paused or frame is None:
                frame = next(frames, None)
                if frame is None:
                    print("源已结束。")
                    break
                if live and not args.no_mirror:
                    frame = cv2.flip(frame, 1)
                # 上下翻转：UVC 相机倒装校正（与 hand_demo_mmpose.py 一致；
                # RealSense 颜色帧原生直立，不翻）
                if live and not args.realsense:
                    frame = cv2.flip(frame, 0)

            canvas = frame.copy()
            H, W = frame.shape[:2]

            # ── 检测 ──────────────────────────────────────
            reuse = det_skip > 0 and frame_count % (det_skip + 1) != 0
            t_det = time.perf_counter()
            if not reuse:
                if args.detector == "world":
                    det_boxes, det_confs = det(frame, args.conf, args.max_boxes,
                                               (W, H))
                    boxes_raw = [list(b) for b in det_boxes]
                else:
                    raw = det(frame, args.det_conf)
                    bb = np.array([r[:4] for r in raw], np.float32).reshape(-1, 4)
                    cc = np.array([r[4] if len(r) > 4 else 1.0 for r in raw],
                                  np.float32)
                    boxes_raw = [list(b) for b in
                                 wd.WorldDetector.postprocess(
                                     bb, cc, W, H, args.max_boxes)[0]]
                t_det = (time.perf_counter() - t_det) * 1000

                # ── 追踪：IoU 匹配到稳定手部 ID ────────────
                tracker.update_detections(boxes_raw)

                # ── 关键点推理（只对需要的手跑）────────────
                boxes_for_pose, pose_indices = tracker.get_boxes_for_pose()
                t_pose = 0.0
                if boxes_for_pose:
                    t_pose = time.perf_counter()
                    raw_kpts, raw_scores = pose(frame, bboxes=boxes_for_pose)
                    tracker.update_pose_results(pose_indices,
                                                np.array(raw_kpts),
                                                np.array(raw_scores))
                    t_pose = (time.perf_counter() - t_pose) * 1000
            else:
                t_det = 0.0   # detection skipped (reusing previous result)
                t_pose = 0.0  # pose skipped (tracker handles gating internally)

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
            hit.append(len(boxes))

            # ── 遮挡判断 + 逐点置信度门控（按手指粒度）───
            # 1) 遮挡：≥90% 关键点置信度很低 → 整只手冻结
            # 2) 否则按手指粒度：弯曲手指的低置信度点才冻结
            if kpts is not None:
                OCCLUSION_RATIO = 0.9   # 低置信度点占比超过此值判定为遮挡

                for t_idx in range(len(kpts)):
                    low_kpt_count = int(np.sum(scores[t_idx][:21] < KPT_FREEZE_THR))
                    occluded = (low_kpt_count / 21 >= OCCLUSION_RATIO)

                    if occluded:
                        for j in range(21):
                            good = tracker.get_last_good(t_idx, j)
                            if good is not None:
                                kpts[t_idx][j] = good.copy()
                    else:
                        ang_check = hc.compute_joint_angles(kpts[t_idx][:21])
                        ext = hc.count_extended_fingers(ang_check)

                        for j in range(21):
                            finger = _kpt_to_finger.get(j)
                            finger_ok = (finger is None
                                         or finger in ext
                                         or scores[t_idx][j] >= KPT_FREEZE_THR)

                            if finger_ok:
                                tracker.update_last_good(t_idx, j, kpts[t_idx][j])
                            else:
                                good = tracker.get_last_good(t_idx, j)
                                if good is not None:
                                    kpts[t_idx][j] = good.copy()
            else:
                tracker.clear()

            info = []
            if show_box:
                for b in boxes:
                    cv2.rectangle(canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                                  (0, 255, 255), 2)
            if kpts is not None:
                for i, p in enumerate(kpts):
                    ang = hc.compute_joint_angles(p[:21])
                    if show_skel:
                        hc.draw_hand(canvas, p, ang, show_angles=show_angles,
                                     kpt_scores=scores[i], thr=args.kpt_thr)
                    ext = hc.count_extended_fingers(ang)
                    info.append((f"[hand {track_ids[i]}] open {len(ext)}/5  kpt {scores[i][:21].mean():.2f}",
                                 (255, 255, 255)))
                    for f, (_ids, col) in hc.FINGERS.items():
                        j = "IP" if f == "Thumb" else "PIP"
                        info.append((f"  {f:<7}{'OPEN' if f in ext else 'BEND'}"
                                     f"  {ang[(f, j)]:5.0f}deg", col))

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt
            rate = 100.0 * sum(1 for n in hit if n) / len(hit)
            head = [
                (f"FPS {fps:4.1f}   hands {len(boxes)}", (0, 255, 180)),
                (f"det {t_det:5.1f}ms + pose {t_pose:5.1f}ms", (120, 220, 255)),
                (f"last100f: {rate:5.1f}% detected",
                 (0, 255, 0) if rate > 80 else (0, 200, 255) if rate > 40 else (0, 120, 255)),
                (det.name[:34], (180, 180, 180)),
            ]
            if show_help:
                head.append(("q:quit a:ang s:skel b:box h:help", (180, 180, 180)))
            hc.draw_panel(canvas, 10, 10, head)
            if info:
                hc.draw_panel(canvas, 10, 10 + 22 * len(head) + 24, info)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(1 if (live and not paused) else 30) & 0xFF
            frame_count += 1
            if key in (ord("q"), 27):
                break
            elif key == ord("a"):
                show_angles = not show_angles
            elif key == ord("s"):
                show_skel = not show_skel
            elif key == ord("b"):
                show_box = not show_box
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord(" "):
                paused = not paused
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

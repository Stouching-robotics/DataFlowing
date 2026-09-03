"""自动标注手部框：零样本开放词汇检测 + 人工只做复核。

两种模式：

1. 零样本（默认）：YOLO-World 用文本提示 "hand"/"glove" 直接检测，一张图都不用先标。
2. 自举：用你已经训好的检测器（--model best.pt）标新数据，越训越准。

在用户实拍的 40 张黑色防滑手套数据上实测（人工标注当 ground truth）：

    yolov8m-worldv2 @320  "hand"+"glove"   命中 40/40   平均 IoU 0.78

**每张图只取置信最高的 N 个框**是关键——原始输出有 2.6 框/图（大量误检），
按置信度取 Top-N 之后误检基本被滤干净。N 用 --max-boxes 指定，默认 1。

检测器的权重/尺寸/阈值统一定义在 world_detector.py，和 infer.py 共用一套，
避免"标注用的框"和"推理用的框"不是同一套。

用法：
    # 先在自己的标注上验一遍准不准
    python auto_label.py --dir to_label --eval

    # 给新数据自动标注（跳过已标的），然后用 annotate.py 复核修正
    python auto_label.py --dir batch2
    python annotate.py --dir batch2

首次运行需要联网下载 YOLO-World 权重和 CLIP 权重（约 400MB）。
"""

import argparse
import glob
import os
import sys
import warnings

import cv2
import numpy as np

from world_detector import WorldDetector, add_args, from_args

warnings.filterwarnings("ignore")

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_PROMPT = ["hand", "glove"]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def read_yolo(path, w, h):
    out = []
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return out
    for line in open(path):
        t = line.split()
        if len(t) >= 5:
            _c, cx, cy, bw, bh = (float(v) for v in t[:5])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def write_yolo(path, boxes, w, h):
    with open(path, "w") as f:
        for x1, y1, x2, y2 in boxes:
            f.write(f"0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                    f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n")


class Trained:
    """自己训好的检测器（自举用）"""

    def __init__(self, weights, device):
        from ultralytics import YOLO
        self.m = YOLO(weights)
        self.device = device
        self.name = f"YOLO {os.path.basename(weights)}"

    def __call__(self, src, conf, max_boxes=1, wh=None):
        r = self.m.predict(src, conf=conf, verbose=False, device=self.device)[0]
        if not len(r.boxes):
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        if wh is None:
            wh = (r.orig_shape[1], r.orig_shape[0])
        return WorldDetector.postprocess(
            r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
            wh[0], wh[1], max_boxes)


def render(img_path, boxes, dst, title, gt_boxes=None):
    """画自动标注的框；给了 gt_boxes 就同时画人工框做对比（绿=人工，橙=自动）。"""
    im = cv2.imread(img_path)
    if im is None:
        return
    for x1, y1, x2, y2 in (gt_boxes or []):
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    color = (0, 140, 255) if gt_boxes else (0, 255, 255)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
    if gt_boxes:
        title += "   green=manual  orange=auto"
    cv2.putText(im, title, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(im, title, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, im, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser(description="手部框自动标注（零样本 / 自举）")
    ap.add_argument("--dir", default="to_label", help="含 images/ 的目录，标注写到同级 labels/")
    ap.add_argument("--model", help="用自己训的检测器代替零样本（自举模式）")
    add_args(ap, max_boxes_default=1)     # 与 infer.py 共用同一套默认值
    ap.add_argument("--device", default="mps")
    ap.add_argument("--eval", action="store_true",
                    help="不写标注，只拿现有 labels/ 当 ground truth 打分")
    ap.add_argument("--overwrite", action="store_true",
                    help="覆盖已存在的标注（默认跳过，避免冲掉人工标好的）")
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()

    img_dir = os.path.join(args.dir, "images")
    lab_dir = os.path.join(args.dir, "labels")
    files = sorted(f for f in glob.glob(os.path.join(img_dir, "*"))
                   if f.lower().endswith(IMG_EXT))
    if not files:
        sys.exit(f"{img_dir} 下没有图片")
    os.makedirs(lab_dir, exist_ok=True)

    det = Trained(args.model, args.device) if args.model else from_args(args, args.device)
    print(f"检测器: {det.name}\n")

    if args.eval:
        # eval 模式只读不写标注，输出对比图到 viz_eval/，人工框和自动框画在一起
        viz_dir = os.path.join(args.dir, "viz_eval")
        tp = ngt = 0
        ious = []
        worst = []
        for f in files:
            im = cv2.imread(f)
            if im is None:
                continue
            h, w = im.shape[:2]
            stem = os.path.splitext(os.path.basename(f))[0]
            gts = read_yolo(os.path.join(lab_dir, stem + ".txt"), w, h)
            if not gts:
                continue
            pred = [list(b) for b in det(f, args.conf, args.max_boxes)[0]]
            ngt += len(gts)
            best_here = 0.0
            for g in gts:
                best = max((iou(g, p) for p in pred), default=0.0)
                best_here = max(best_here, best)
                if best >= 0.5:
                    tp += 1
                    ious.append(best)
            worst.append((best_here, stem))
            if not args.no_viz:
                render(f, pred, os.path.join(viz_dir, f"{best_here:.2f}_{stem}.jpg"),
                       f"{stem}  IoU={best_here:.2f}", gt_boxes=gts)
        print(f"对照 {ngt} 个人工框：IoU>=0.5 命中 {tp}/{ngt} ({100*tp/max(ngt,1):.0f}%)，"
              f"匹配框平均 IoU {np.mean(ious) if ious else 0:.3f}")
        worst.sort()
        print("IoU 最低的几张（优先检查这些）：",
              ", ".join(f"{s}({v:.2f})" for v, s in worst[:5]))
        if not args.no_viz:
            print(f"对比图: {viz_dir}/   文件名以 IoU 开头，排序后最差的排在最前面")
        print("命中率高说明自动标注可用——但仍要人工过一遍，改掉少数错的。")
        return

    n_new = n_skip = n_empty = 0
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        lab = os.path.join(lab_dir, stem + ".txt")
        if os.path.exists(lab) and not args.overwrite:
            n_skip += 1
            continue
        im = cv2.imread(f)
        if im is None:
            continue
        h, w = im.shape[:2]
        boxes = [list(b) for b in det(f, args.conf, args.max_boxes)[0]]
        write_yolo(lab, boxes, w, h)
        if not boxes:
            n_empty += 1
        n_new += 1
        if not args.no_viz:
            render(f, boxes, os.path.join(args.dir, "viz_auto", stem + ".jpg"),
                   f"{stem}  auto boxes={len(boxes)}")

    print(f"自动标注 {n_new} 张（其中 {n_empty} 张没检出），跳过已标注 {n_skip} 张")
    if not args.no_viz:
        print(f"核对图: {os.path.join(args.dir, 'viz_auto')}")
    print(f"\n下一步——务必人工复核，自动标注不是 100% 准：")
    print(f"  python annotate.py --dir {args.dir}      # 逐张翻，错的直接改")


if __name__ == "__main__":
    main()

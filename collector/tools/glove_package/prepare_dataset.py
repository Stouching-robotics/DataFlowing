"""把 hand_demo.py 采集的数据转成 YOLO 检测数据集（单类别 hand）。

思路：MediaPipe 已经检出手的那些帧，bbox 可以从 21 个关键点直接算出来，
属于**白捡的标注**；只有漏检帧（labels.jsonl 里 hands 为空）才需要人工拉框，
而那恰恰是训练检测器最需要的难样本。

用法：
    # 1) 生成数据集 + 导出待人工标注的漏检帧
    python prepare_dataset.py --captures ../captures --out dataset

    # 2) 用 labelImg / CVAT 标 to_label/images，标注写到 to_label/labels
    #    （YOLO 格式，单类别，class id 填 0）

    # 3) 把标好的合并进数据集
    python prepare_dataset.py --merge to_label --out dataset
"""

import argparse
import json
import os
import random
import shutil
import sys

import cv2
from collections import Counter

CLASS_NAME = "hand"


def bbox_to_yolo(b, w, h, pad):
    """labels.jsonl 的 [x1,y1,x2,y2]（像素坐标）-> YOLO 的 cx cy w h（归一化 0-1）。

    注意：labels.jsonl 存的是像素坐标（不是归一化），需要除以图像宽高。
    """
    x1, y1, x2, y2 = b
    # 确保 x1<=x2, y1<=y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    # 归一化
    x1, x2 = x1 / w, x2 / w
    y1, y2 = y1 / h, y2 / h
    # padding
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0.0, x1 - pw), max(0.0, y1 - ph)
    x2, y2 = min(1.0, x2 + pw), min(1.0, y2 + ph)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def iter_records(captures_dir):
    """遍历所有采集 session，产出 (session名, 记录, 图片绝对路径)"""
    if not os.path.isdir(captures_dir):
        sys.exit(f"找不到采集目录: {captures_dir}")
    sessions = sorted(d for d in os.listdir(captures_dir)
                      if os.path.isfile(os.path.join(captures_dir, d, "labels.jsonl")))
    if not sessions:
        sys.exit(f"{captures_dir} 下没有找到任何含 labels.jsonl 的 session")
    for s in sessions:
        with open(os.path.join(captures_dir, s, "labels.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                img = os.path.join(captures_dir, s, rec["file"])
                if os.path.exists(img):
                    yield s, rec, img


def split_indices(items, val_split):
    """每个 session 内部按时间连续切块：末尾 val_split 做验证集。

    绝对不能随机切——相邻帧几乎一模一样，随机切会让验证集泄漏进训练集，
    验证指标虚高。也不要整 session 做验证集：只有两个 session 时会把最新
    一次采集整个变成验证集，新数据一点都进不了训练。
    """
    train, val = [], []
    for s in sorted({s for s, _, _ in items}):
        sess_idx = [i for i, (s2, _, _) in enumerate(items) if s2 == s]
        n_val = max(1, round(len(sess_idx) * val_split))
        cut = len(sess_idx) - n_val
        train += sess_idx[:cut]
        val += sess_idx[cut:]
    how = f"每个 session 按时间切末尾 {val_split:.0%} 做验证集"
    return train, val, how


def render_viz(img_path, yolo_lines, dst, title):
    """把 YOLO 标签画回图上，供人工核对标注对不对。

    直接从写进数据集的 .txt 反解坐标来画——这样画出来的就是模型真正会读到的框，
    而不是另算一遍。转换写错了在这里一眼就能看出来。
    """
    im = cv2.imread(img_path)
    if im is None:
        return
    h, w = im.shape[:2]
    for line in yolo_lines:
        _c, cx, cy, bw, bh = (float(v) for v in line.split())
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(im, "hand", (x1 + 4, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    tag = f"{title}  boxes={len(yolo_lines)}"
    cv2.putText(im, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(im, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2, cv2.LINE_AA)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, im, [cv2.IMWRITE_JPEG_QUALITY, 88])


def write_dataset(out, groups, pad, viz=True):
    for sub in ("images", "labels"):
        for sp in ("train", "val"):
            os.makedirs(os.path.join(out, sub, sp), exist_ok=True)
    n = Counter()
    for split, records in groups.items():
        for idx, (sess, rec, img) in enumerate(records):
            stem = f"{sess}_{os.path.splitext(os.path.basename(rec['file']))[0]}"
            ext = os.path.splitext(img)[1]
            shutil.copy2(img, os.path.join(out, "images", split, stem + ext))
            lines = [f"0 {' '.join(f'{v:.6f}' for v in bbox_to_yolo(h['bbox'], rec['width'], rec['height'], pad))}"
                     for h in rec["hands"]]
            with open(os.path.join(out, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            if viz:
                render_viz(img, lines, os.path.join(out, "viz", split, stem + ".jpg"),
                           f"{split} / {stem}")
            n[split] += 1
            n[split + "_boxes"] += len(lines)
    return n


def write_yaml(out):
    path = os.path.abspath(out)
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {path}\ntrain: images/train\nval: images/val\n\n"
                f"nc: 1\nnames: [{CLASS_NAME}]\n")


def cmd_prepare(args):
    items = list(iter_records(args.captures))
    hit = [x for x in items if x[1]["hands"]]
    miss = [x for x in items if not x[1]["hands"]]
    print(f"扫描到 {len(items)} 帧：{len(hit)} 帧有检测结果（可自动标注），"
          f"{len(miss)} 帧漏检（需人工标）")

    if hit:
        # 清掉上一次导出的残留，防止 split 变化后同一张图在两个 split 各留一份
        # （人工复核结果走 to_label + --merge 原位更新，不会被这次清理误伤）
        for sub in ("images", "labels", "viz"):
            shutil.rmtree(os.path.join(args.out, sub), ignore_errors=True)
        train, val, how = split_indices(hit, args.val_split)
        print(f"  {how}")
        n = write_dataset(args.out, {"train": [hit[i] for i in train],
                                     "val": [hit[i] for i in val]}, args.pad, not args.no_viz)
        write_yaml(args.out)
        print(f"  -> {args.out}: train {n['train']} 图/{n['train_boxes']} 框，"
              f"val {n['val']} 图/{n['val_boxes']} 框")
        if not args.no_viz:
            print(f"  -> {args.out}/viz: 画好框的核对图，逐张看一遍确认标注没问题")
    else:
        # 全程用 m（只存漏检帧）采集时会走到这里：没有白捡的标注，但漏检帧照样要导出去标
        print("  没有可自动标注的帧——如果你全程用的是 m（只存漏检）模式，这是正常的。")
        print("  下面的帧全部需要人工拉框；标完用 --merge 就能直接建出数据集。")

    # 漏检帧单独放，绝不能带着空标签进训练集
    if miss:
        td = args.to_label
        os.makedirs(os.path.join(td, "images"), exist_ok=True)
        os.makedirs(os.path.join(td, "labels"), exist_ok=True)
        for sess, rec, img in miss:
            stem = f"{sess}_{os.path.splitext(os.path.basename(rec['file']))[0]}"
            shutil.copy2(img, os.path.join(td, "images", stem + os.path.splitext(img)[1]))
        with open(os.path.join(td, "README.txt"), "w", encoding="utf-8") as f:
            f.write(
                "这些是模型漏检的帧，需要人工拉框（单类别 hand，class id = 0）。\n"
                "标注工具输出 YOLO 格式，txt 放到同级的 labels/ 下，文件名与图片同名。\n"
                "标完执行：python prepare_dataset.py --merge to_label --out dataset\n\n"
                "注意：没标注的图片不要放进训练集。空标签在 YOLO 里表示"
                "「这张图没有目标」，会把戴手套的手教成背景，越训越差。\n")
        print(f"  -> {td}/images: {len(miss)} 张待人工标注（详见 {td}/README.txt）")


def cmd_merge(args):
    src_img = os.path.join(args.merge, "images")
    src_lab = os.path.join(args.merge, "labels")
    if not os.path.isdir(src_lab):
        sys.exit(f"找不到 {src_lab}")
    if not os.path.isdir(os.path.join(args.out, "images", "train")):
        # 全 miss 采集时数据集还没建起来，这里补上骨架
        for sub in ("images", "labels"):
            for sp in ("train", "val"):
                os.makedirs(os.path.join(args.out, sub, sp), exist_ok=True)
        write_yaml(args.out)
        print(f"{args.out} 还是空的，已新建数据集骨架")

    labeled = []
    for lab in sorted(os.listdir(src_lab)):
        if not lab.endswith(".txt"):
            continue
        p = os.path.join(src_lab, lab)
        if os.path.getsize(p) == 0:      # 空文件 = 还没标，跳过
            continue
        stem = os.path.splitext(lab)[0]
        img = next((os.path.join(src_img, stem + e)
                    for e in (".jpg", ".png") if os.path.exists(os.path.join(src_img, stem + e))),
                   None)
        if img:
            labeled.append((stem, img, p))

    if not labeled:
        sys.exit(f"{src_lab} 里没有非空的标注文件——还没标或者输出目录不对")

    # 已经在数据集里的帧（prepare 从采集记录导出的）→ 原位更新标签，不换 split，
    # 否则同一张图可能同时出现在 train 和 val 两个 split 里（验证集泄漏）
    new_labeled = []
    for stem, img, lab in labeled:
        for split in ("train", "val"):
            dst_lab = os.path.join(args.out, "labels", split, stem + ".txt")
            if os.path.exists(dst_lab):
                shutil.copy2(lab, dst_lab)
                if not args.no_viz:
                    with open(lab) as f:
                        lines = [x.strip() for x in f if x.strip()]
                    render_viz(img, lines, os.path.join(args.out, "viz", split, stem + ".jpg"),
                               f"{split} / {stem} (人工标注)")
                break
        else:
            new_labeled.append((stem, img, lab))
    if len(new_labeled) != len(labeled):
        print(f"原位更新 {len(labeled) - len(new_labeled)} 张已存在帧的标签")
    labeled = new_labeled

    n_val = max(1, round(len(labeled) * args.val_split))
    random.Random(0).shuffle(labeled)    # 人工标的多为独立难样本，随机切可接受
    for i, (stem, img, lab) in enumerate(labeled):
        split = "val" if i < n_val else "train"
        shutil.copy2(img, os.path.join(args.out, "images", split,
                                       stem + os.path.splitext(img)[1]))
        shutil.copy2(lab, os.path.join(args.out, "labels", split, stem + ".txt"))
        if not args.no_viz:
            with open(lab) as f:
                lines = [x.strip() for x in f if x.strip()]
            render_viz(img, lines, os.path.join(args.out, "viz", split, stem + ".jpg"),
                       f"{split} / {stem} (人工标注)")
    if labeled:
        print(f"合并 {len(labeled)} 张人工标注（val {n_val} / train {len(labeled)-n_val}）到 {args.out}")


def main():
    ap = argparse.ArgumentParser(description="采集数据 -> YOLO 手部检测数据集")
    ap.add_argument("--captures", default="../captures", help="hand_demo.py 的采集目录")
    ap.add_argument("--out", default="dataset", help="输出数据集目录")
    ap.add_argument("--to-label", default="to_label", help="漏检帧导出目录（待人工标注）")
    ap.add_argument("--pad", type=float, default=0.12,
                    help="关键点外扩比例，让自动框和人工框风格接近 (默认 0.12)")
    ap.add_argument("--val-split", type=float, default=0.2, help="验证集比例")
    ap.add_argument("--no-viz", action="store_true",
                    help="不生成 viz/ 核对图（默认会生成）")
    ap.add_argument("--merge", metavar="DIR",
                    help="把 DIR 下人工标好的数据合并进 --out 数据集")
    args = ap.parse_args()
    (cmd_merge if args.merge else cmd_prepare)(args)


if __name__ == "__main__":
    main()

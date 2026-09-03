"""拿人工标好的框，检验 RTMPose 关键点模型在你的手套上到底行不行。

**先跑这个再决定要不要大批量标注。**

整套「只标框、不标 21 个点」的方案，前提是：给一个准确的手部框，
RTMPose 的关键点模型就能在戴手套的手上输出正确关键点。
这个前提在合成手套上验证成立（误差 3.9px，与裸手持平），
但在真实数据上我没能确证——因为拿不到准确的框（自动检测器全失败，
肉眼读框会偏，穷举搜索的置信度会被空白背景骗到）。

你只要标 10~20 张，这个脚本就能给出确定答案：
    - 输出的骨架贴合手指  -> 前提成立，放心去标框，21 个点不用标
    - 骨架是乱的          -> 前提不成立，得改方案（标关键点微调，或换模型）

用法：
    python verify_pose.py --dir to_label            # 读 to_label/{images,labels}
    open verify_pose_out/                           # 逐张看
"""

import argparse
import glob
import os
import sys
import warnings

import cv2
import numpy as np

import hand_common as hc

warnings.filterwarnings("ignore")

POSE_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip")


def load_yolo_boxes(path, w, h):
    boxes = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            _c, cx, cy, bw, bh = (float(v) for v in p[:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def main():
    ap = argparse.ArgumentParser(description="用人工框检验 RTMPose 关键点质量")
    ap.add_argument("--dir", default="to_label", help="含 images/ 和 labels/ 的目录")
    ap.add_argument("--out", default="verify_pose_out", help="输出目录")
    ap.add_argument("--device", default=None,
                    help="关键点模型 device（留空自动选 CUDA>MPS>CPU）")
    ap.add_argument("--kpt-thr", type=float, default=0.3)
    args = ap.parse_args()

    if args.device is None:
        from hand_common import auto_device
        args.device = auto_device()

    img_dir = os.path.join(args.dir, "images")
    lab_dir = os.path.join(args.dir, "labels")
    if not os.path.isdir(lab_dir):
        sys.exit(f"找不到 {lab_dir}")

    pairs = []
    for lab in sorted(glob.glob(os.path.join(lab_dir, "*.txt"))):
        if os.path.getsize(lab) == 0:
            continue
        stem = os.path.splitext(os.path.basename(lab))[0]
        img = next((os.path.join(img_dir, stem + e) for e in (".jpg", ".png")
                    if os.path.exists(os.path.join(img_dir, stem + e))), None)
        if img:
            pairs.append((stem, img, lab))
    if not pairs:
        sys.exit(f"{lab_dir} 里没有非空标注。先标几张（单类别 hand，YOLO 格式）再跑。")

    from rtmlib import RTMPose
    pose = RTMPose(POSE_URL, model_input_size=(256, 256),
                   backend="onnxruntime", device=args.device)

    os.makedirs(args.out, exist_ok=True)
    confs = []
    for stem, img_p, lab_p in pairs:
        im = cv2.imread(img_p)
        h, w = im.shape[:2]
        boxes = load_yolo_boxes(lab_p, w, h)
        if not boxes:
            continue
        k, sc = pose(im, bboxes=boxes)
        k, sc = np.array(k), np.array(sc)
        canvas = im.copy()
        for b in boxes:
            cv2.rectangle(canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (0, 255, 255), 3)
        for i in range(len(k)):
            hc.draw_hand(canvas, k[i], hc.compute_joint_angles(k[i][:21]),
                         show_angles=False, kpt_scores=sc[i], thr=args.kpt_thr)
        c = float(sc[:, :21].mean())
        confs.append(c)
        tag = f"{stem}  kpt_conf={c:.2f}"
        cv2.putText(canvas, tag, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(canvas, tag, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 180), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.out, stem + ".jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {stem}: {len(boxes)} 框, 关键点置信 {c:.3f}")

    print(f"\n{len(confs)} 张已输出到 {args.out}/，平均关键点置信 {np.mean(confs):.3f}")
    print("光看置信度不作数——它在空白背景上反而会更高。**必须逐张打开图看骨架是否贴合手指。**")
    print("  贴合  -> 只标框的方案成立，继续标注即可")
    print("  不贴合 -> 关键点模型也扛不住这副手套，需要换方案（见 README）")


if __name__ == "__main__":
    main()

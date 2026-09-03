"""训练手部检测器（单类别 hand），用来替换 RTMDet —— 它在黑手套上会直接输出 0 个框。

只训检测，不训关键点：实测 RTMPose 的关键点模型在黑手套上本来就准
（给定正确的框，误差 3.9px vs 裸手 3.9px），瓶颈只在"找不到手"。

用法：
    python train_detector.py                          # 默认 yolo11n，100 轮
    python train_detector.py --model yolo11s.pt --epochs 200
    python train_detector.py --resume runs/hand_det/weights/last.pt
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="训练单类别手部检测器")
    ap.add_argument("--data", default="dataset/data.yaml", help="数据集 yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="预训练权重（n/s/m 依次更大更准）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="",
                    help="训练设备（留空自动选 CUDA>MPS>CPU；或显式 0 / cpu / mps）")
    ap.add_argument("--project", default="runs",
                    help="输出根目录（相对路径锚定在本包目录下）")
    ap.add_argument("--name", default="hand_det", help="本次运行名")
    ap.add_argument("--resume", metavar="CKPT", help="从 last.pt 续训")
    ap.add_argument("--no-augment-hsv", action="store_true",
                    help="关掉色相/饱和度增强（手套是纯黑的，色彩增强意义不大）")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("需要先装 ultralytics:  pip install -r requirements.txt")

    if not args.resume and not os.path.exists(args.data):
        sys.exit(f"找不到 {args.data}，先跑 prepare_dataset.py 生成数据集")

    # 留空 → 自动选设备（本机 CUDA > macOS MPS > CPU），启动前明确打印
    if not args.device:
        from hand_common import auto_device
        args.device = auto_device()
        label = args.device
        if args.device == "cuda":
            import torch
            label = f"cuda:0 ({torch.cuda.get_device_name(0)})"
        print(f"训练设备: {label}")

    model = YOLO(args.resume or args.model)
    kw = {}
    if args.no_augment_hsv:
        # 手套本身没什么色彩信息，把色彩增强的算力让给几何增强
        kw.update(hsv_h=0.0, hsv_s=0.0, hsv_v=0.4)

    # 相对 project 锚定到本包目录：ultralytics 8.4 会把相对路径解析到
    # 全局 settings 的 runs_dir（这台机器是 /home/REDACTED/collector/runs），
    # 产物会落到包外面去，破坏包的自包含性
    project = args.project
    if not os.path.isabs(project):
        project = os.path.join(os.path.dirname(os.path.abspath(__file__)), project)

    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=project, name=args.name,
        resume=bool(args.resume), exist_ok=True, **kw,
    )

    best = os.path.join(project, args.name, "weights", "best.pt")
    print(f"\n训练完成。最佳权重: {best}")
    print(f"跑起来看看:  python infer.py --detector {best}")


if __name__ == "__main__":
    main()

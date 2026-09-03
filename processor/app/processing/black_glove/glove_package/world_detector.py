"""YOLO-World 开放词汇手部检测器——auto_label.py 和 infer.py 的唯一真值来源。

以前两边各写了一份配置（权重 x vs m、imgsz 640 vs 320、conf 0.01 vs 0.05），
结果是**自动标注用的框和实时推理用的框不是同一套**，标出来的数据训完
换到推理时行为对不上。所以统一放这里，改一处两边都变。

默认值来自 40 张真实手套数据的实测（见 README）：

    模型              imgsz   召回     平均IoU   检测耗时
    yolov8s-worldv2   320     27/40   0.743    18.6ms
    yolov8m-worldv2   320     40/40   0.780    22.6ms   <- 默认
    yolov8m-worldv2   640     30/40   0.766    30.7ms
    yolov8x-worldv2   640     39/40   0.781    85.7ms

两个反直觉的点：imgsz=320 比 640 准（手占比大时才落进模型习惯的目标尺度），
m 比 x 快 3.8 倍而召回还高一点。
"""

import math
import os

import numpy as np

DEFAULT_MODEL = "yolov8m-worldv2.pt"
DEFAULT_PROMPT = ["hand", "glove"]
DEFAULT_IMGSZ = 320
DEFAULT_CONF = 0.05
DEFAULT_NMS_IOU = 0.6



class WorldDetector:
    """文本提示驱动的检测器，不需要训练。

    后端选择：
      - 默认用 PyTorch（.pt 文件）——GPU 上 ~12ms，CPU 上 ~85ms
      - 也可传 .onnx 文件显式指定 ONNX（CPU ~54ms，GPU 不可用因为缺系统 CUDA Toolkit）

    __call__ 接受 BGR numpy 帧，
    统一返回 (boxes_xyxy [N,4], confs [N,])，已做过滤 + 类别无关 NMS + Top-N。
    """

    def __init__(self, model=DEFAULT_MODEL, prompt=None, imgsz=DEFAULT_IMGSZ,
                 device="cpu", nms_iou=DEFAULT_NMS_IOU, use_onnx=False):
        self.prompt = list(prompt or DEFAULT_PROMPT)
        self.imgsz = imgsz
        self.device = device
        self.nms_iou = nms_iou
        self._last_boxes = None   # 帧跳过缓存

        if use_onnx and model.endswith('.onnx'):
            self._init_onnx(model)
        elif model.endswith('.onnx'):
            self._init_onnx(model)
        else:
            self._init_pt(model)

    def _init_onnx(self, path):
        from ultralytics import YOLO
        self.m = YOLO(path, task='detect')
        self.device = 'cpu'  # ONNX GPU EP 需要系统 CUDA Toolkit，不现实
        self.name = f"YOLO-World ONNX {os.path.basename(path)} @{self.imgsz} {self.prompt}"

    def _init_pt(self, path):
        from ultralytics import YOLOWorld
        self.m = YOLOWorld(path)
        self.m.set_classes(self.prompt)
        name_base = os.path.basename(path).replace('.pt', '')
        self.name = f"YOLO-World {name_base} @{self.imgsz} {self.prompt}"

    def __call__(self, src, conf=DEFAULT_CONF, max_boxes=1, wh=None,
                 reuse_boxes=False):
        """检测手部。reuse_boxes=True 时复用上一帧结果，不发推理。"""
        if reuse_boxes and self._last_boxes is not None:
            return self._last_boxes
        r = self.m.predict(src, conf=conf, imgsz=self.imgsz,
                           device=self.device, verbose=False)[0]
        if not len(r.boxes):
            self._last_boxes = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
            return self._last_boxes
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = r.boxes.conf.cpu().numpy().astype(np.float32)
        if wh is None:
            wh = (r.orig_shape[1], r.orig_shape[0])
        result = self.postprocess(boxes, confs, wh[0], wh[1], max_boxes, self.nms_iou)
        self._last_boxes = result
        return result

    @staticmethod
    def postprocess(boxes, confs, w, h, max_boxes, nms_iou=DEFAULT_NMS_IOU):
        """裁到画面内 + 丢掉非法/过小框 + 两级 NMS + 取 Top-N。

        必须按**置信度**排序而不是面积：画面边缘的误检往往面积不小，
        按面积排会把它留下、把真手挤掉。

        NMS 是必需的：提示词给 ["hand","glove"] 时同一只手会被两个词各检出一次，
        不去重的话下游会对同一只手重复跑一遍关键点模型。

        两级 NMS 判定（解决"两只手靠近时误删一只"的问题）：
          - 同一只手重复检出：中心距 < 框宽的 15% 且 IoU > 0.85 → 抑制
          - 两只不同的手靠近：不满足上述条件 → 即使 IoU 较高也不抑制
        """
        cand = []
        for b, c in zip(boxes, confs):
            v = [float(x) for x in b[:4]]
            if not all(map(math.isfinite, v)):
                continue
            x1, y1 = max(0.0, min(v[0], v[2])), max(0.0, min(v[1], v[3]))
            x2, y2 = min(float(w), max(v[0], v[2])), min(float(h), max(v[1], v[3]))
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                cand.append(([x1, y1, x2, y2], float(c)))
        cand.sort(key=lambda t: t[1], reverse=True)

        keep_b, keep_c = [], []
        for box, c in cand:
            dup = False
            for k in keep_b:
                iou_val = iou(box, k)
                if _center_close(box, k, ratio=0.15):
                    # 中心几乎重合 → 很可能是同一只手 → 用标准 NMS 阈值
                    if iou_val >= nms_iou:
                        dup = True
                        break
                else:
                    # 中心有偏移 → 很可能是两只不同的手 → 只用极高阈值
                    # （只在极端重复的情况下才抑制）
                    if iou_val > 0.85:
                        dup = True
                        break
            if not dup:
                keep_b.append(box)
                keep_c.append(c)
            if len(keep_b) >= max_boxes:
                break
        return (np.array(keep_b, np.float32).reshape(-1, 4),
                np.array(keep_c, np.float32))

    @staticmethod
    def postprocess_legacy(boxes, confs, w, h, max_boxes, nms_iou=DEFAULT_NMS_IOU):
        """旧版 NMS（纯 IoU 阈值，不做空间距离检查）。保留给兼容性测试。"""
        cand = []
        for b, c in zip(boxes, confs):
            v = [float(x) for x in b[:4]]
            if not all(map(math.isfinite, v)):
                continue
            x1, y1 = max(0.0, min(v[0], v[2])), max(0.0, min(v[1], v[3]))
            x2, y2 = min(float(w), max(v[0], v[2])), min(float(h), max(v[1], v[3]))
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                cand.append(([x1, y1, x2, y2], float(c)))
        cand.sort(key=lambda t: t[1], reverse=True)

        keep_b, keep_c = [], []
        for box, c in cand:
            if all(iou(box, k) < nms_iou for k in keep_b):
                keep_b.append(box)
                keep_c.append(c)
            if len(keep_b) >= max_boxes:
                break
        return (np.array(keep_b, np.float32).reshape(-1, 4),
                np.array(keep_c, np.float32))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _center_close(a, b, ratio=0.15):
    """两个框的中心距离是否 < 较小框宽度的 ratio 倍。

    用于 NMS 时区分"同一只手重复检出"（中心近）和"两只靠近的手"（中心远）。
    """
    ca_x, ca_y = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    cb_x, cb_y = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    dist = math.hypot(ca_x - cb_x, ca_y - cb_y)
    ref = min(a[2] - a[0], b[2] - b[0]) * ratio
    return dist < ref


def add_args(ap, max_boxes_default=1):
    """给 argparse 挂上这套检测器的公共参数，保证两边命令行也一致。"""
    ap.add_argument("--weights", default=DEFAULT_MODEL, help="YOLO-World 权重 (.pt 或 .onnx)")
    ap.add_argument("--prompt", nargs="+", default=DEFAULT_PROMPT, help="文本提示词")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                    help="输入尺寸（手占比大时 320 比 640 更准也更快）")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF, help="检测置信度阈值")
    ap.add_argument("--max-boxes", type=int, default=max_boxes_default,
                    help="每帧最多保留几个框（按置信度）")
    ap.add_argument("--det-skip", type=int, default=0,
                    help="每 N 帧跑一次检测，中间帧复用上一次的框。0=每帧都检测")
    return ap


def from_args(args, device):
    return WorldDetector(args.weights, args.prompt, args.imgsz, device)

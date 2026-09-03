"""手部检测 + 关键点推理管线 —— 供外部程序调用的统一入口。

用法:
    from hand_pipeline import HandPipeline

    pipe = HandPipeline(detector="runs/detect/runs/hand_det/weights/best.pt",
                        det_device="cuda", pose_device="cpu")

    for frame in video_frames:
        # 返回: (boxes, kpts, scores, track_ids)
        #   boxes:      [(x1,y1,x2,y2), ...]  像素坐标
        #   kpts:       np.ndarray (N,21,2)    关键点像素坐标
        #   scores:     np.ndarray (N,21)      每点置信度
        #   track_ids:  [int, ...]             稳定手部ID
        result = pipe.process(frame)

        # 可选：带冻结和遮挡判断
        result = pipe.process(frame, apply_freeze=True)
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import hand_common as hc
import world_detector as wd
from hand_tracker import HandTracker

MODEL_PATH = os.path.join(_HERE, wd.DEFAULT_MODEL)


class HandPipeline:
    """手部检测+关键点推理管线。

    参数
    ----
    detector : str
        "world" 用 YOLO-World（免训练），或 .pt 路径用自己训练的模型。
    det_device : str
        检测器设备（"cuda" / "cpu"）。
    pose_device : str
        关键点设备（"cuda" / "cpu"，默认 cpu）。
    max_hands : int
        最多检测手数（默认 2）。
    conf : float
        检测置信度阈值（默认 0.05）。
    det_imgsz : int
        YOLO 检测器推理尺寸（默认 1280）。
        best.pt 实测对 640 缩放过敏感：同框 conf 掉到 0.4-0.65、位置漂移幻觉，
        1280 原生推理才能还原训练时尺度下的检测质量（glove_detector.py 实测结论）。
    """

    def __init__(
        self,
        detector: str = "world",
        det_device: str = "cpu",
        pose_device: str = "cpu",
        max_hands: int = 2,
        conf: float = 0.05,
        det_imgsz: int = 1280,
    ):
        self.max_hands = max_hands
        self.max_boxes = max_hands
        self.conf = conf
        self.pose_device = pose_device

        # ── 检测器 ──────────────────────────────────────
        if detector == "world":
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(
                    f"YOLO-World 权重不存在: {MODEL_PATH}\n"
                    "首次运行时会自动下载，或手动放到 rtmpose/ 目录。")
            self._det = wd.WorldDetector(
                MODEL_PATH, wd.DEFAULT_PROMPT, wd.DEFAULT_IMGSZ, det_device)
        else:
            self._det = _YOLOWrapper(detector, det_device, det_imgsz)

        # ── 关键点模型 ──────────────────────────────────
        self._pose = hc.build_pose(pose_device)

        # ── 追踪器 ──────────────────────────────────────
        self._tracker = HandTracker(
            max_hands=max_hands,
            iou_match_thr=0.3,
            lost_timeout=3,
            movement_thresh=3,
            skip_timeout=10,
            box_smooth_alpha=0.7,
        )

        # ── 手指映射 ────────────────────────────────────
        self._kpt_to_finger = {}
        for fn, (ids, _) in hc.FINGERS.items():
            for idx in ids:
                self._kpt_to_finger[idx] = fn

        # 可调参数
        self.kpt_freeze_thr = 0.2    # 逐点冻结阈值
        self.occlusion_ratio = 0.9   # 遮挡判定比例

    # ── 公共 API ──────────────────────────────────────────

    def process(
        self,
        frame: np.ndarray,
        apply_freeze: bool = True,
    ) -> Tuple[
        List[List[float]],
        Optional[np.ndarray],
        Optional[np.ndarray],
        List[int],
    ]:
        """处理一帧图像。

        参数
        ----
        frame : np.ndarray
            BGR 图像 (H, W, 3)。
        apply_freeze : bool
            是否应用逐点置信度冻结 + 遮挡判断（默认 True）。

        返回
        ----
        (boxes, kpts, scores, track_ids)
            boxes:      [[x1,y1,x2,y2], ...]  像素坐标
            kpts:       (N,21,2) 或 None
            scores:     (N,21) 或 None
            track_ids:  [int, ...]  稳定手部ID
        """
        h_img, w_img = frame.shape[:2]

        # ── 检测 ──────────────────────────────────────
        if isinstance(self._det, wd.WorldDetector):
            det_boxes, _ = self._det(frame, self.conf, self.max_boxes, (w_img, h_img))
            boxes_raw = [list(b) for b in det_boxes]
        else:
            raw = self._det(frame, self.conf)
            bb = np.array([r[:4] for r in raw], np.float32).reshape(-1, 4)
            cc = np.array([r[4] for r in raw], np.float32)
            boxes_raw = [list(b) for b in
                         wd.WorldDetector.postprocess(bb, cc, w_img, h_img,
                                                      self.max_boxes)[0]]

        # ── 追踪 ──────────────────────────────────────
        self._tracker.update_detections(boxes_raw)

        # ── 关键点推理 ────────────────────────────────
        boxes_for_pose, pose_indices = self._tracker.get_boxes_for_pose()
        if boxes_for_pose:
            raw_kpts, raw_scores = self._pose(frame, bboxes=boxes_for_pose)
            self._tracker.update_pose_results(
                pose_indices, np.array(raw_kpts), np.array(raw_scores))

        # ── 获取结果 ──────────────────────────────────
        boxes, kpts, scores, track_ids = self._tracker.get_results()

        # ── 无手套误检抑制 ─────────────────────────────
        # 背景/手套状物体的框，RTMPose 解不出像样的 21 点（kpt 均值 <0.45
        # 或高置信点 <15）→ 判定无手套：不画框、不解算（见 hand_common.pose_is_glove）
        if kpts is not None and len(kpts):
            keep = [i for i in range(len(kpts))
                    if hc.pose_is_glove(kpts[i], scores[i], boxes[i])]
            if keep:
                boxes = [boxes[i] for i in keep]
                kpts = kpts[keep]
                scores = scores[keep]
                track_ids = [track_ids[i] for i in keep]
            else:
                boxes, kpts, scores, track_ids = [], None, None, []

        # ── 冻结 + 遮挡 ───────────────────────────────
        if apply_freeze and kpts is not None:
            self._apply_freeze(kpts, scores)

        return boxes, kpts, scores, track_ids

    def reset(self) -> None:
        """重置追踪状态（切换视频源时调用）。"""
        self._tracker.clear()

    # ── 内部 ──────────────────────────────────────────────

    def _apply_freeze(self, kpts: np.ndarray, scores: np.ndarray) -> None:
        """遮挡判断 + 按手指粒度逐点冻结（原地修改 kpts）。"""
        for t_idx in range(len(kpts)):
            low_count = int(np.sum(scores[t_idx][:21] < self.kpt_freeze_thr))
            occluded = (low_count / 21 >= self.occlusion_ratio)

            if occluded:
                # 整只手遮挡 → 全部冻结
                for j in range(21):
                    good = self._tracker.get_last_good(t_idx, j)
                    if good is not None:
                        kpts[t_idx][j] = good.copy()
            else:
                ang_check = hc.compute_joint_angles(kpts[t_idx][:21])
                ext = hc.count_extended_fingers(ang_check)

                for j in range(21):
                    finger = self._kpt_to_finger.get(j)
                    finger_ok = (finger is None
                                 or finger in ext
                                 or scores[t_idx][j] >= self.kpt_freeze_thr)

                    if finger_ok:
                        self._tracker.update_last_good(t_idx, j, kpts[t_idx][j])
                    else:
                        good = self._tracker.get_last_good(t_idx, j)
                        if good is not None:
                            kpts[t_idx][j] = good.copy()

    @property
    def detector_name(self) -> str:
        return self._det.name


# ── 内部：自定义 YOLO 包装 ─────────────────────────────
class _YOLOWrapper:
    """best.pt 需 imgsz=1280 原生推理；640 下 conf 掉 + 位置幻觉（实测）。"""

    def __init__(self, weights, device, imgsz=1280):
        from ultralytics import YOLO
        self.m = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.name = f"YOLO ({os.path.basename(weights)})"

    def __call__(self, frame, conf):
        r = self.m(frame, conf=conf, device=self.device,
                   imgsz=self.imgsz, verbose=False)[0]
        return [list(map(float, b)) + [float(c)]
                for b, c in zip(r.boxes.xyxy.cpu().numpy(),
                                r.boxes.conf.cpu().numpy())]

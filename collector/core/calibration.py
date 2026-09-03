"""
EgoData 标定数据模型 —— 双目相机内外参 + 深度缩放。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import json
import numpy as np


@dataclass
class CameraIntrinsics:
    """单目相机内参。"""
    intrinsic: List[float]   # [fx, fy, cx, cy]
    distortion: List[float] = field(default_factory=list)  # k1,k2,p1,p2[,k3]

    @classmethod
    def from_matrix(cls, K: np.ndarray, dist: np.ndarray = None) -> "CameraIntrinsics":
        """从 OpenCV 标定矩阵构建。"""
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        d = dist.flatten().tolist() if dist is not None and dist.size > 0 else []
        return cls(intrinsic=[fx, fy, cx, cy], distortion=d)

    def to_matrix(self) -> np.ndarray:
        """还原为 3×3 相机矩阵。"""
        fx, fy, cx, cy = self.intrinsic
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def to_dist_array(self) -> np.ndarray:
        """还原为畸变系数数组。"""
        return np.array(self.distortion, dtype=np.float64) if self.distortion else np.zeros(4)


@dataclass
class StereoCalibration:
    """双目 + 深度相机标定（EgoData 标准）。

    JSON 格式:
    {
      "type": "stereo_rgbd_camera",
      "name": "head_stereo",
      "resolution": [1280, 800],
      "fps": 25,
      "baseline": 0.095,
      "left_camera":  {"intrinsic": [fx, fy, cx, cy], "distortion": []},
      "right_camera": {"intrinsic": [fx, fy, cx, cy], "distortion": []},
      "depth_scale": 0.001,
      "cam_imu_timeshift": -0.0019
    }
    """
    type: str = "stereo_rgbd_camera"
    name: str = "head_stereo"
    resolution: List[int] = field(default_factory=lambda: [1280, 800])
    fps: float = 25.0
    baseline: float = 0.095               # 基线 (米)
    left_camera: CameraIntrinsics = field(
        default_factory=lambda: CameraIntrinsics(intrinsic=[0, 0, 0, 0]))
    right_camera: CameraIntrinsics = field(
        default_factory=lambda: CameraIntrinsics(intrinsic=[0, 0, 0, 0]))
    depth_scale: float = 0.001             # 像素值 × depth_scale = 米
    cam_imu_timeshift: float = -0.0019    # 相机-IMU 时间偏移 (秒), IMU 比相机早
                                          # 来源: SDK calib.yaml timeshift_cam_imu
                                          # (cam0=-0.001947, cam1=-0.001851)
    depth_camera: Optional[CameraIntrinsics] = None   # 深度相机内参(D435 独有;S80M 无此字段)
                                                      # 注: rs2 畸变系数为 Inverse Brown Conrady
                                                      # 模型,原样存储,消费方需按 rs2 模型处理

    # ── 序列化 ──────────────────────────────────────

    def to_dict(self) -> dict:
        d = {
            "type": self.type,
            "name": self.name,
            "resolution": list(self.resolution),
            "fps": self.fps,
            "baseline": self.baseline,
            "left_camera": {
                "intrinsic": list(self.left_camera.intrinsic),
                "distortion": list(self.left_camera.distortion),
            },
            "right_camera": {
                "intrinsic": list(self.right_camera.intrinsic),
                "distortion": list(self.right_camera.distortion),
            },
            "depth_scale": self.depth_scale,
            "cam_imu_timeshift": self.cam_imu_timeshift,
        }
        if self.depth_camera is not None:
            d["depth_camera"] = {
                "intrinsic": list(self.depth_camera.intrinsic),
                "distortion": list(self.depth_camera.distortion),
            }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StereoCalibration":
        def _intrinsics(key: str) -> CameraIntrinsics:
            cam = d.get(key, {})
            return CameraIntrinsics(
                intrinsic=list(cam.get("intrinsic", [0, 0, 0, 0])),
                distortion=list(cam.get("distortion", [])),
            )
        return cls(
            type=d.get("type", "stereo_rgbd_camera"),
            name=d.get("name", "head_stereo"),
            resolution=list(d.get("resolution", [1280, 800])),
            fps=float(d.get("fps", 25.0)),
            baseline=float(d.get("baseline", 0.095)),
            left_camera=_intrinsics("left_camera"),
            right_camera=_intrinsics("right_camera"),
            depth_scale=float(d.get("depth_scale", 0.001)),
            cam_imu_timeshift=float(d.get("cam_imu_timeshift", -0.0019)),
            depth_camera=_intrinsics("depth_camera") if "depth_camera" in d else None,
        )

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "StereoCalibration":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ── 工厂方法 ────────────────────────────────────

    @classmethod
    def from_opencv(cls,
                    K_left: np.ndarray, dist_left: np.ndarray,
                    K_right: np.ndarray, dist_right: np.ndarray,
                    baseline: float = 0.095,
                    resolution: tuple = (1280, 800),
                    fps: float = 25.0,
                    depth_scale: float = 0.001) -> "StereoCalibration":
        """从 OpenCV stereoCalibrate 结果构建。"""
        return cls(
            type="stereo_rgbd_camera",
            name="head_stereo",
            resolution=list(resolution),
            fps=fps,
            baseline=baseline,
            left_camera=CameraIntrinsics.from_matrix(K_left, dist_left),
            right_camera=CameraIntrinsics.from_matrix(K_right, dist_right),
            depth_scale=depth_scale,
        )

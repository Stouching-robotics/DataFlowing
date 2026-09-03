#!/usr/bin/env python3
"""
stereo_s80m.hand_3d —— 双目鱼眼 3D 手部关键点检测 + 3D 渲染独立模块。

两阶段管线（Hur et al. 2025）：每目 MediaPipe 2D → 双目三角化粗 3D →
透视裁剪图精修 2D → 二次三角化 → 3D 域 One-Euro 平滑 → 3D 旋转视角渲染。

只读复用 stereo_s80m / hand_detection，不修改任何主程序文件；
全部输出落在 keypoints_output/<tag>/<session>/ 下。

用法::

    python stereo_s80m/hand_3d/run_pipeline.py <session_dir> [选项]
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stereo_s80m.hand_3d.run_pipeline import main  # noqa: E402

__all__ = ["main"]

"""MediaPipe Hand — 手部骨骼识别 + 手势识别 + 骨架视频渲染模块。

模块元数据 + run(ctx)(worker 分发入口);run_local() 是真正的逐帧推理实现,
保留为独立函数便于测试。无视频输入时 skip(不 fallback 乱找、不失败)。

手势识别(阶段 1):关键点由 mp.solutions.hands 输出,手势类别由 Task API
GestureRecognizer 输出(官方 7 类:Closed_Fist/Open_Palm/Pointing_Up/
Thumbs_Down/Thumbs_Up/Victory/ILoveYou)。扩展点:config.gesture_model 指向
自定义手势模型即可替换;模型文件缺失 → 手势列保持 -1,不失败。
"""

import re
from pathlib import Path
from typing import Callable

from app.processing import ProcessingModule, JobContext, ArtifactRef, field
from app.processing.hand_render import draw_demo_style
from app.processing.registry import register
from app.processing.theme import HAND3D_COLOR

# 默认手势识别模型(官方 float16,~8MB);可被 config.gesture_model 覆盖
GESTURE_MODEL_DEFAULT = str(Path(__file__).parents[3] / "models" / "gesture_recognizer.task")


@register
class MediaPipeHandModule(ProcessingModule):
    slug = "mediapipe_hand"
    version = "1.0"
    category = "process"
    label = "MediaPipe Hand"
    icon = "ant-design:aim-outlined"
    color = HAND3D_COLOR
    inputs = ({"key": "video", "label": "RGB Video"},)
    outputs = ({"key": "hand_keypoints", "label": "Hand 2D"},)
    default_config = {"max_hands": 2, "min_detection_conf": 0.5, "min_tracking_conf": 0.5,
                      "device": "auto"}
    config_schema = (
        field("max_hands", "number", "Max hands", 2, min=1, max=2),
        field("min_detection_conf", "number", "Detection confidence", 0.5, min=0, max=1, step=0.05),
        field("min_tracking_conf", "number", "Tracking confidence", 0.5, min=0, max=1, step=0.05),
        field("device", "select", "Device", "auto", options=["auto", "cpu", "cuda:0"]),
    )
    execution_target = "worker"
    capabilities = ("hand_keypoints", "video_overlay", "mediapipe")

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        # 收集全部上游 video 类 artifact:双目左右目两条边 → 两路都识别;
        # 没有 → 跳过,绝不 fallback 到批次目录乱找
        video_refs = [r for r in ctx.incoming.values() if r.kind == "video"]
        if not video_refs:
            ctx.skip("No video input — skipped")
        cfg = dict(ctx.config)
        # 原图上的 SVG 覆盖层是唯一前端显示链路。历史工作流里可能还
        # 保存了 generate_skeleton_video=true,但从现在起不再生成独立视频。
        cfg["generate_skeleton_video"] = False
        out: dict[str, ArtifactRef] = {}
        for i, vref in enumerate(video_refs):
            video_path = ctx.resolve(vref)
            if not video_path or not video_path.exists():
                ctx.skip(f"Video artifact missing: {vref.path}")
            # 产物按相机分流写入 worker 临时目录；完成阶段会把关键点列
            # 合并进 canonical episode parquet。
            legacy = not vref.source_key and len(video_refs) == 1
            key = re.sub(r"[^A-Za-z0-9_.-]", "_", str(vref.source_key or video_path.stem))
            result = run_local(
                video_path,
                ctx.output_dir if legacy else ctx.output_root / "hand_keypoints",
                cfg, ctx.progress,
                kp_name="hand_keypoints.parquet" if legacy else f"{key}.parquet",
                skel_dir=None,
                output_name=None,
            )
            suffix = "" if i == 0 else f"#{i + 1}"
            kp_path = (ctx.output_dir if legacy else ctx.output_root / "hand_keypoints") / result["keypoints_parquet"]
            out[f"hand_keypoints{suffix}"] = ctx.ref(
                "hand_keypoints", kp_path, source_key=vref.source_key, metadata=result)
            if result.get("skeleton_video"):
                skel_path = (ctx.output_dir if legacy else ctx.output_root / "skeleton") / result["skeleton_video"]
                out[f"skeleton_video{suffix}"] = ctx.ref(
                    "video", skel_path, source_key=vref.source_key, metadata=result)
        return out


def run_local(
    video_path: Path,
    output_dir: Path,
    config: dict | None = None,
    progress_callback: Callable[[float], None] | None = None,
    *,
    kp_name: str = "hand_keypoints.parquet",
    skel_dir: Path | None = None,
    output_name: str | None = None,
    with_gesture: bool = True,
) -> dict:
    """Run MediaPipe Hands locally and write a worker artifact set.

    ``output_dir`` 是关键点 parquet 的输出目录;``skel_dir`` 指定骨架视频
    目录(多路时按相机分目录);``kp_name`` / ``output_name`` 用于多路
    场景按相机命名,避免互相覆盖。默认参数保持旧的单路行为。

    The function intentionally uses the stable MediaPipe 21-landmark output
    contract. ``device`` is recorded in metadata; the Python MediaPipe Hands
    solution is CPU-backed in this first worker implementation.
    """
    import json
    import os
    from app.config import settings

    # Keep native-library caches inside the project workspace as well. Some
    # MediaPipe dependencies initialize Matplotlib on Windows.
    matplotlib_cache = settings.temp_root / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    # On Windows MediaPipe and OpenCV ship native DLLs with overlapping
    # dependencies. Loading MediaPipe first avoids an intermittent DLL
    # initialization failure when pandas/OpenCV has already loaded NumPy.
    import mediapipe as mp
    import cv2
    import pandas as pd

    cfg = config or {}
    output_dir.mkdir(parents=True, exist_ok=True)
    if skel_dir is not None:
        skel_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / kp_name
    base_name = output_name or f"{video_path.stem}_skeleton"
    video_out = (skel_dir or output_dir) / f"{base_name}{video_path.suffix}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    max_hands = max(1, min(2, int(cfg.get("max_hands", 2))))
    min_detection = float(cfg.get("min_detection_conf", 0.5))
    min_tracking = float(cfg.get("min_tracking_conf", 0.5))
    generate_video = bool(cfg.get("generate_skeleton_video", False))
    requested_device = str(cfg.get("device", "auto"))

    rows: list[dict] = []
    writer = None
    if generate_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            writer = None

    # ── 手势识别(MediaPipe Gesture Recognizer,官方 7 类)─────────
    # 阶段 1:关键点由 mp.solutions.hands 输出,手势由 Task API 输出;
    # 模型文件缺失 → 手势列保持 -1,不影响骨骼识别主流程。
    gesture_recognizer = None
    gesture_ts_ms = 0
    try:
        # 注意:mediapipe.tasks.python 不能用 import mediapipe.tasks.python 链式
        # 导入(包结构限制),必须用 from 形式
        # with_gesture=False(双目 3D 链路)跳过手势推理:每帧省一次模型
        # 前向(≈关键点识别同等耗时),手部 3D 数据不用手势列。
        if not with_gesture:
            raise RuntimeError("gesture disabled by with_gesture=False")
        from mediapipe.tasks.python import BaseOptions as _BaseOptions
        from mediapipe.tasks.python import vision as mp_vision
        model_path = str(cfg.get("gesture_model") or GESTURE_MODEL_DEFAULT)
        if Path(model_path).exists():
            gesture_recognizer = mp_vision.GestureRecognizer.create_from_options(
                mp_vision.GestureRecognizerOptions(
                    base_options=_BaseOptions(model_asset_path=model_path),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=max_hands,
                ))
    except Exception as gesture_err:
        print(f"[MediaPipe] Gesture recognizer disabled: {gesture_err}")

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        model_complexity=int(cfg.get("model_complexity", 1)),
        min_detection_confidence=min_detection,
        min_tracking_confidence=min_tracking,
    ) as hands:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            keypoints: list[list[list[float]] | None] = [None, None]
            handedness: list[str | None] = [None, None]
            confidence: list[float | None] = [None, None]

            detected = result.multi_hand_landmarks or []
            handed = result.multi_handedness or []
            for hand_index, landmarks in enumerate(detected[:2]):
                keypoints[hand_index] = [
                    [float(point.x), float(point.y), float(point.z)]
                    for point in landmarks.landmark[:21]
                ]
                if hand_index < len(handed):
                    classification = handed[hand_index].classification[0]
                    handedness[hand_index] = str(classification.label)
                    confidence[hand_index] = float(classification.score)

            # 手势分类(每只手 top-1:类别索引 0-6 + 分数;未检测或不属于
            # 官方 7 类(模型输出 index=-1/None)→ -1,分数清零)
            gesture: list[int] = [-1, -1]
            gesture_score: list[float] = [0.0, 0.0]
            if gesture_recognizer is not None:
                try:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    gres = gesture_recognizer.recognize_for_video(mp_image, gesture_ts_ms)
                    for gi, glist in enumerate(gres.gestures[:2]):
                        if glist and glist[0].index >= 0:
                            gesture[gi] = int(glist[0].index)
                            gesture_score[gi] = float(glist[0].score)
                except Exception:
                    pass
                gesture_ts_ms += int(1000.0 / max(1.0, fps))

            row = {
                "frame_index": frame_index,
                "hand_0_keypoints": keypoints[0],
                "hand_1_keypoints": keypoints[1],
                "hand_0_handedness": handedness[0],
                "hand_1_handedness": handedness[1],
                "hand_0_confidence": confidence[0],
                "hand_1_confidence": confidence[1],
                "hand_0_gesture": gesture[0],
                "hand_1_gesture": gesture[1],
                "hand_0_gesture_score": gesture_score[0],
                "hand_1_gesture_score": gesture_score[1],
                "gesture_model": "mediapipe_gesture_recognizer" if gesture_recognizer else None,
                "model": "mediapipe_hands",
                "model_version": getattr(mp, "__version__", "unknown"),
                "device": "cpu",
                "device_requested": requested_device,
            }
            rows.append(row)

            if writer is not None:
                _draw_local_skeleton(frame, keypoints)
                writer.write(frame)

            frame_index += 1
            if progress_callback and total:
                progress_callback(min(0.99, frame_index / total))

    cap.release()
    if writer is not None:
        writer.release()

    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    manifest = {
        "keypoints_parquet": str(parquet_path.name),
        "frames": len(rows),
        "fps": fps,
        "width": width,
        "height": height,
        "model": "mediapipe_hands",
        "model_version": getattr(mp, "__version__", "unknown"),
        "device": "cpu",
        "device_requested": requested_device,
        "gesture_model": "mediapipe_gesture_recognizer" if gesture_recognizer else None,
        "gesture_classes": 7 if gesture_recognizer else 0,
    }
    if writer is not None:
        # OpenCV writes MPEG-4 Part 2 (mp4v), which browsers cannot play.
        # Transcode to H.264 with ffmpeg when available so the skeleton
        # video streams directly in the Review page.
        h264_out = video_out.with_name(video_out.stem + "_h264.mp4")
        if _transcode_to_h264(video_out, h264_out):
            try:
                video_out.unlink(missing_ok=True)
            except OSError:
                pass
            video_out = h264_out
        manifest["skeleton_video"] = str(video_out.name)
    # 多路场景 manifest 按关键点名命名,避免互相覆盖;单路名与旧行为一致
    manifest_name = f"{Path(kp_name).stem}.manifest.json"
    (output_dir / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if progress_callback:
        progress_callback(1.0)
    return {"output_dir": str(output_dir), **manifest}


def _transcode_to_h264(src: Path, dst: Path) -> bool:
    """Transcode ``src`` to a browser-playable H.264 MP4 via ffmpeg.

    Returns False when ffmpeg is unavailable or the transcode fails — the
    original (mp4v) file is kept in that case.
    """
    import shutil
    import subprocess

    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    try:
        subprocess.run(
            [exe, "-y", "-loglevel", "error", "-i", str(src),
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-an", str(dst)],
            check=True, capture_output=True, timeout=600,
        )
        return dst.exists() and dst.stat().st_size > 0
    except Exception:
        return False


def _draw_local_skeleton(frame, keypoints: list):
    """Draw the same canonical style used by the hand 3D module."""
    height, width = frame.shape[:2]
    for points in keypoints:
        if not points or len(points) < 21:
            continue
        px = [(int(max(0, min(width - 1, p[0] * width))),
               int(max(0, min(height - 1, p[1] * height)))) for p in points]
        draw_demo_style(frame, px)

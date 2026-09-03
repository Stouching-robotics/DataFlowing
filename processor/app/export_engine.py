"""LeRobot v3.0 dataset export engine.

Reads episodes + frames from the database, writes the standard LeRobot
directory layout:

exports/<dataset_name>/
├── meta/
│   ├── info.json            # schema, fps, features, path templates
│   ├── stats.json           # global normalization statistics
│   ├── tasks.jsonl          # task descriptions
│   └── episodes/            # chunked Parquet episode index
├── data/                    # chunked Parquet frames
└── videos/                  # per-camera MP4 shards
"""

import json
import os
from pathlib import Path
from typing import Callable, Sequence
from uuid import UUID

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from app.config import settings
from app.database import async_session
from app.models import Episode, Frame, AnnotationSegment
from sqlalchemy import select
from sqlalchemy.orm import selectinload


class DatasetExporter:
    """Export collected episodes to LeRobot v3.0 format."""

    def __init__(
        self,
        job_id: UUID,
        dataset_name: str,
        episode_ids: Sequence[UUID],
        split_ratio: float = 0.9,
    ):
        self.job_id = job_id
        self.dataset_name = dataset_name
        self.episode_ids = episode_ids
        self.split_ratio = split_ratio
        self.output_dir = str(settings.storage_root / "exports" / dataset_name)

    async def run(self, progress_callback: Callable[[float], None] | None = None):
        """Execute the full export pipeline."""
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Load all frames for selected episodes
        frames_df, episode_meta = await self._load_data()

        if frames_df.empty:
            raise ValueError("No frames found for selected episodes")

        total_eps = len(episode_meta)
        self._report(progress_callback, 5)

        # 2. Assign tasks
        task_map = await self._build_task_map(episode_meta)
        self._report(progress_callback, 10)

        # 3. Train/val split at episode level
        frames_df, episode_meta = self._assign_split(frames_df, episode_meta)
        self._report(progress_callback, 15)

        # 4. Write meta/
        camera_names = self._detect_cameras(frames_df)
        features = self._build_features(frames_df, camera_names)
        fps = episode_meta[0].get("fps", 30) if episode_meta else 30

        self._write_info_json(features, fps, total_eps, len(frames_df))
        self._write_tasks_jsonl(task_map, episode_meta)
        self._report(progress_callback, 25)

        # 5. Write data/ Parquet shards
        await self._write_data_parquet(frames_df, progress_callback)
        self._report(progress_callback, 70)

        # 6. Write videos/ MP4 shards
        await self._write_videos(episode_meta, frames_df, camera_names, fps, progress_callback)
        self._report(progress_callback, 85)

        # 7. meta/episodes/ index — 由 lerobot_export / hdf5_export 流程
        # 统一写为每个 episode 一个 episode_XXXXXX.parquet 文件，这里不再写
        # (旧引擎曾写 chunk-000.parquet 顶层文件,与标准命名冲突)。
        self._report(progress_callback, 90)

        # 8. Compute and write stats.json
        self._write_stats_json(frames_df, features)
        from app.storage import sync_tree_to_remote_async
        await sync_tree_to_remote_async(
            Path(self.output_dir),
            Path("exports") / self.dataset_name,
        )
        self._report(progress_callback, 100)

    # ── Data loading ──────────────────────────────────

    async def _load_data(self) -> tuple[pd.DataFrame, list[dict]]:
        """Load frames, episode metadata, and annotations.

        Frames are read from the database first; if empty, fall back to
        reading the original LeRobot parquet files from the session directory.
        """
        async with async_session() as session:
            frames_list = []
            episode_meta = []

            for ep_id in self.episode_ids:
                ep = await session.get(Episode, ep_id)
                if not ep:
                    continue

                # ── Load annotations for this episode ──
                anno_q = (
                    select(AnnotationSegment)
                    .options(selectinload(AnnotationSegment.keyframes))
                    .where(AnnotationSegment.episode_id == ep_id)
                    .order_by(AnnotationSegment.sort_order, AnnotationSegment.start_frame_index)
                )
                anno_segs = (await session.execute(anno_q)).scalars().all()

                # Build frame → (label, index) map
                frame_anno: dict[int, tuple[str | None, int]] = {}
                anno_defs = []
                for idx, seg in enumerate(anno_segs):
                    for fi in range(seg.start_frame_index, seg.end_frame_index + 1):
                        if fi not in frame_anno:
                            frame_anno[fi] = (seg.label, idx)
                    kf_frames = []
                    for kf in seg.keyframes:
                        kf_frames.append({
                            "frame_index": kf.frame_index,
                            "event": kf.event or "",
                        })
                    anno_defs.append({
                        "idx": idx,
                        "label": seg.label,
                        "start_frame": seg.start_frame_index,
                        "end_frame": seg.end_frame_index,
                        "frames": seg.end_frame_index - seg.start_frame_index + 1,
                        "color": seg.color,
                        "keyframes": kf_frames,
                    })

                episode_meta.append({
                    "episode_id": str(ep.id),
                    "task_description": ep.task_description or "",
                    "fps": ep.fps,
                    "frame_count": ep.frame_count,
                    "camera_names": ep.camera_names or [],
                    "annotations": anno_defs,
                })

                # ── Load frames: DB first, then parquet fallback ──
                q = (
                    select(Frame)
                    .where(Frame.episode_id == ep_id)
                    .order_by(Frame.frame_index)
                )
                db_frames = (await session.execute(q)).scalars().all()

                if db_frames:
                    # Path 1: frames stored in database
                    for f in db_frames:
                        row = self._build_frame_row(f, frame_anno)
                        frames_list.append(row)
                else:
                    # Path 2: frames in session parquet files — read directly
                    parquet_frames = await self._load_frames_from_parquet(
                        ep, frame_anno
                    )
                    frames_list.extend(parquet_frames)

            df = pd.DataFrame(frames_list)
            return df, episode_meta

    def _build_frame_row(self, f: Frame, frame_anno: dict) -> dict:
        """Build a single frame row from a DB Frame record with annotations."""
        anno_label, anno_idx = frame_anno.get(f.frame_index, (None, -1))
        row = {
            "episode_id": str(f.episode_id),
            "frame_index": f.frame_index,
            "timestamp": f.timestamp,
            "reward": f.reward,
            "is_terminal": f.is_terminal,
            "is_truncated": f.is_truncated,
            "annotation": anno_label,
            "annotation_index": anno_idx,
            "annotation.language_persistent": anno_label,  # 避开官方保留列名(结构化消息行)
        }
        obs = f.observation or {}
        if "state" in obs:
            row["observation.state"] = obs["state"]
        for k, v in obs.items():
            if k != "state":
                row[f"observation.{k}"] = v
        act = f.action or {}
        if act:
            if "joint_positions" in act:
                row["action"] = act["joint_positions"]
            elif isinstance(act, list):
                row["action"] = act
            else:
                row["action"] = act
        if f.image_paths:
            for cam, path in f.image_paths.items():
                row[f"image.{cam}"] = path
        return row

    async def _load_frames_from_parquet(
        self, ep: Episode, frame_anno: dict
    ) -> list[dict]:
        """Read frames from the original LeRobot parquet files in the session directory."""
        from app.storage import find_session_dir_async

        frames_list = []
        fps = ep.fps or 30
        ep_id_str = str(ep.id)

        # Find the correct session directory for this episode
        session_dir = None
        if ep.session_id:
            async with async_session() as s:
                session_dir = await find_session_dir_async(str(ep.session_id), s)

        if session_dir is None:
            return frames_list

        # Scan for parquet files
        data_dir = None
        for d in session_dir.rglob("data"):
            if d.is_dir():
                data_dir = d
                break
        if data_dir is None:
            for d in session_dir.rglob("*.parquet"):
                data_dir = d.parent
                break
        if data_dir is None:
            return frames_list

        # Read and annotate parquet files
        for parq_file in sorted(data_dir.rglob("*.parquet")):
            try:
                df = pd.read_parquet(parq_file)
            except Exception:
                continue

            # Skip meta parquets
            if not any(c for c in df.columns if "frame_index" in str(c).lower()):
                continue

            for _, row in df.iterrows():
                fi = int(row.get("frame_index", len(frames_list)))
                anno_label, anno_idx = frame_anno.get(fi, (None, -1))

                frame_row = {
                    "episode_id": ep_id_str,
                    "frame_index": fi,
                    "timestamp": float(row.get("timestamp", fi / fps)),
                    "reward": float(row.get("reward", 0.0)) if "reward" in df.columns else None,
                    "is_terminal": bool(row.get("is_terminal", False)) if "is_terminal" in df.columns else False,
                    "is_truncated": bool(row.get("is_truncated", False)) if "is_truncated" in df.columns else False,
                    "annotation": anno_label,
                    "annotation_index": anno_idx,
                    "annotation.language_persistent": anno_label,  # 避开官方保留列名(结构化消息行)
                }

                # Copy observation/action columns
                for col in df.columns:
                    if col.startswith("observation."):
                        val = row[col]
                        if isinstance(val, (list, pd.Series)):
                            val = list(val)
                        frame_row[col] = val
                    elif col == "action" or col.startswith("action"):
                        val = row[col]
                        if isinstance(val, (list, pd.Series)):
                            val = list(val)
                        elif hasattr(val, 'item'):
                            val = val.item() if hasattr(val, 'size') and val.size == 1 else list(val)
                        frame_row["action"] = val
                    elif col.startswith("episode_index"):
                        pass  # handled above
                    elif col.startswith("task_index"):
                        pass  # handled by task assignment later

                # Copy image paths
                for col in df.columns:
                    if col.startswith("image.") or col.startswith("video."):
                        frame_row[col] = str(row[col]) if not pd.isna(row[col]) else None

                frames_list.append(frame_row)

        return frames_list

    # ── Tasks ─────────────────────────────────────────

    async def _build_task_map(self, episode_meta: list[dict]) -> dict[int, str]:
        """Build a map of task_id → description."""
        tasks: dict[str, int] = {}
        task_map: dict[int, str] = {}
        idx = 0

        for ep in episode_meta:
            desc = ep.get("task_description", "").strip()
            if desc and desc not in tasks:
                tasks[desc] = idx
                task_map[idx] = desc
                idx += 1

        # Assign task_id to each episode
        for ep in episode_meta:
            desc = ep.get("task_description", "").strip()
            ep["task_id"] = tasks.get(desc, -1) if desc else -1

        return task_map

    # ── Split ─────────────────────────────────────────

    def _assign_split(self, df: pd.DataFrame, episode_meta: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
        """Assign train/val split at episode level."""
        rng = np.random.default_rng(42)
        ep_list = list(set(df["episode_id"]))
        rng.shuffle(ep_list)
        n_train = max(1, int(len(ep_list) * self.split_ratio))
        train_eps = set(ep_list[:n_train])

        df["split"] = df["episode_id"].apply(lambda e: "train" if e in train_eps else "val")

        for ep in episode_meta:
            ep["split"] = "train" if ep["episode_id"] in train_eps else "val"

        return df, episode_meta

    # ── Features detection ────────────────────────────

    def _detect_cameras(self, df: pd.DataFrame) -> list[str]:
        """Detect camera names from the dataframe columns."""
        cameras = set()
        for col in df.columns:
            if col.startswith("image."):
                cameras.add(col.replace("image.", ""))
        return sorted(cameras)

    def _build_features(self, df: pd.DataFrame, camera_names: list[str]) -> dict:
        """Build the features dict for info.json."""
        features = {}
        for col in df.columns:
            if col.startswith("observation.") and col != "observation.state":
                val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if val is not None:
                    if isinstance(val, (list, np.ndarray)):
                        features[col] = {"dtype": "float32", "shape": [len(val)]}
                    else:
                        features[col] = {"dtype": "float32", "shape": [1]}

            elif col == "observation.state":
                val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if val is not None and isinstance(val, (list, np.ndarray)):
                    features[col] = {"dtype": "float32", "shape": [len(val)]}

            elif col == "action":
                val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if val is not None and isinstance(val, (list, np.ndarray)):
                    features[col] = {"dtype": "float32", "shape": [len(val)]}

        for cam in camera_names:
            features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": [480, 640, 3],
                "info": {"camera": cam},
            }

        # Annotation features
        if "annotation" in df.columns:
            features["annotation"] = {"dtype": "string", "shape": [1]}
        if "annotation_index" in df.columns:
            features["annotation_index"] = {"dtype": "int64", "shape": [1]}
        if "annotation.language_persistent" in df.columns:
            features["annotation.language_persistent"] = {"dtype": "string", "shape": [1]}

        return features

    # ── meta/ writers ─────────────────────────────────

    def _write_info_json(self, features: dict, fps: int, total_episodes: int, total_frames: int):
        """Write meta/info.json."""
        info = {
            "codebase_version": "v3.0",
            "robot_type": "unknown",
            "fps": fps,
            "features": features,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "splits": {
                "train": "data/train-*.parquet",
                "val": "data/val-*.parquet",
            },
            "data_path": "data/{split}-{shard_idx:03d}.parquet",
            "video_path": "videos/chunk-{chunk_idx:03d}/{video_key}/file-{file_idx:03d}.mp4",
            "video_keys": [
                f"observation.images.{cam}"
                for cam in self._detect_cameras(pd.DataFrame())
            ] or [],
        }

        # Add detected camera keys
        if features:
            cam_keys = [k for k in features if k.startswith("observation.images.")]
            if cam_keys:
                info["video_keys"] = cam_keys

        path = Path(self.output_dir) / "meta"
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    def _write_tasks_jsonl(self, task_map: dict[int, str], episode_meta: list[dict] | None = None):
        """Write meta/tasks.jsonl with optional annotation definitions."""
        path = Path(self.output_dir) / "meta"
        path.mkdir(parents=True, exist_ok=True)

        # Collect annotation definitions per task
        task_annotations: dict[int, list[dict]] = {}
        if episode_meta:
            for ep in episode_meta:
                tid = ep.get("task_id", -1)
                annos = ep.get("annotations", [])
                if annos and tid >= 0 and tid not in task_annotations:
                    task_annotations[tid] = annos

        with open(path / "tasks.jsonl", "w") as f:
            for tid, desc in sorted(task_map.items()):
                entry: dict = {"task_id": tid, "description": desc}
                if tid in task_annotations:
                    entry["annotations"] = task_annotations[tid]
                    entry["annotation_schema"] = {
                        "column": "annotation",
                        "dtype": "string",
                        "description": "Per-frame action label",
                    }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── data/ Parquet ─────────────────────────────────

    async def _write_data_parquet(self, df: pd.DataFrame, progress_callback=None):
        """Write data/ as chunked Parquet files, split by train/val."""
        data_path = Path(self.output_dir) / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        for split_name in ["train", "val"]:
            split_df = df[df["split"] == split_name]
            if split_df.empty:
                continue

            shard_idx = 0
            for i in range(0, len(split_df), settings.EXPORT_SHARD_SIZE):
                chunk = split_df.iloc[i : i + settings.EXPORT_SHARD_SIZE]
                t = pa.Table.from_pandas(chunk)
                pq.write_table(t, str(data_path / f"{split_name}-{shard_idx:03d}.parquet"))
                shard_idx += 1

                if progress_callback:
                    progress_callback(25 + int(45 * (i / len(split_df)) * 0.5))

    # ── videos/ MP4 ───────────────────────────────────

    async def _write_videos(
        self,
        episode_meta: list[dict],
        df: pd.DataFrame,
        camera_names: list[str],
        fps: int,
        progress_callback=None,
    ):
        """Write videos/ as per-camera MP4 shards from saved frame images.

        Uses OpenCV to encode JPEG frames into MP4.
        """
        if not camera_names:
            return

        try:
            import cv2
        except ImportError:
            print("[Export] opencv not available, skipping video encoding")
            return

        video_path = Path(self.output_dir) / "videos"
        video_path.mkdir(parents=True, exist_ok=True)

        for cam in camera_names:
            cam_dir = video_path / "chunk-000" / f"observation.images.{cam}"
            cam_dir.mkdir(parents=True, exist_ok=True)

            image_col = f"image.{cam}"
            if image_col not in df.columns:
                continue

            # Group frames by episode and encode
            for ep in episode_meta:
                ep_df = df[df["episode_id"] == ep["episode_id"]].sort_values("frame_index")
                images = ep_df[image_col].dropna()

                if images.empty:
                    continue

                # Determine frame size from first image
                first_img_path = Path(settings.STORAGE_DIR) / images.iloc[0]
                if not first_img_path.exists():
                    continue

                sample = cv2.imread(str(first_img_path))
                if sample is None:
                    continue

                h, w = sample.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_file = str(cam_dir / f"episode_{ep['episode_id'][:8]}.mp4")
                writer = cv2.VideoWriter(out_file, fourcc, fps, (w, h))

                for _, row in ep_df.iterrows():
                    img_rel = row.get(image_col)
                    if not img_rel or pd.isna(img_rel):
                        # Write duplicate frame to maintain timing
                        if writer.isOpened():
                            writer.write(sample)
                        continue

                    img_abs = Path(settings.STORAGE_DIR) / img_rel
                    if img_abs.exists():
                        frame = cv2.imread(str(img_abs))
                        if frame is not None:
                            # Resize if needed
                            if frame.shape[:2] != (h, w):
                                frame = cv2.resize(frame, (w, h))
                            writer.write(frame)
                            sample = frame  # fallback for missing frames

                writer.release()

    # ── stats.json ────────────────────────────────────

    def _write_stats_json(self, df: pd.DataFrame, features: dict):
        """Compute and write meta/stats.json with global normalization statistics."""
        stats = {}
        for feat_name, feat_info in features.items():
            if feat_name.startswith("observation.images."):
                continue  # skip video features

            if feat_name not in df.columns:
                continue

            series = df[feat_name].dropna()
            if series.empty:
                continue

            # Flatten array values
            values = []
            for v in series:
                if isinstance(v, (list, np.ndarray)):
                    values.append(np.array(v, dtype=np.float32))
                elif isinstance(v, (int, float)):
                    values.append(np.array([v], dtype=np.float32))

            if not values:
                continue

            arr = np.stack(values)
            stats[feat_name] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
                "count": len(values),
            }

        path = Path(self.output_dir) / "meta"
        with open(path / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    def _report(self, callback, pct):
        if callback:
            callback(pct)

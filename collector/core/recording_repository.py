"""
数据仓库层 —— 录制记录的增删改查操作。
"""

from typing import List, Optional
from core.database import db
from core.recording_record import RecordingRecord


class RecordingRepo:
    """录制记录的 CRUD 仓库。"""

    @staticmethod
    def save(r: RecordingRecord):
        """保存（插入或替换）一条录制记录。"""
        db.conn.execute(
            """INSERT OR REPLACE INTO recording
               (id, camera_index, camera_name, file_path, episode_index,
                file_size_mb, duration_sec, resolution_w, resolution_h,
                status, started_at, finished_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (r.id, r.camera_index, r.camera_name, r.file_path,
             r.episode_index, r.file_size_mb, r.duration_sec,
             r.resolution_w, r.resolution_h, r.status,
             r.started_at, r.finished_at),
        )
        db.conn.commit()

    @staticmethod
    def list_all(limit: int = 100) -> List[RecordingRecord]:
        """按时间倒序返回最近 N 条录制记录。"""
        rows = db.conn.execute(
            "SELECT * FROM recording ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [RecordingRecord.from_row(r) for r in rows]

    @staticmethod
    def list_by_camera(camera_index: int, limit: int = 50) -> List[RecordingRecord]:
        """按摄像机索引筛选录制记录。"""
        rows = db.conn.execute(
            "SELECT * FROM recording WHERE camera_index=? ORDER BY started_at DESC LIMIT ?",
            (camera_index, limit),
        ).fetchall()
        return [RecordingRecord.from_row(r) for r in rows]

    @staticmethod
    def delete(record_id: str):
        """删除指定记录。"""
        db.conn.execute("DELETE FROM recording WHERE id=?", (record_id,))
        db.conn.commit()

    @staticmethod
    def _where(file_path: str, episode_index):
        """匹配条件：v1.1.0 起按 (file_path, episode_index) 定位一个 episode；
        episode_index 为 None 时退回旧路径匹配（历史行/旧调用方兼容）。"""
        if episode_index is None:
            return "file_path=?", (file_path,)
        return "file_path=? AND episode_index=?", (file_path, episode_index)

    @staticmethod
    def mark_uploaded(file_path: str, episode_index=None) -> int:
        """把指定 episode 的录制行状态改为 uploaded（上传成功、本地保留）。

        行保留（历史可查"已上传"），返回受影响行数。
        """
        cond, args = RecordingRepo._where(file_path, episode_index)
        cur = db.conn.execute(
            f"UPDATE recording SET status='uploaded' WHERE {cond}", args)
        db.conn.commit()
        return cur.rowcount

    @staticmethod
    def mark_uploaded_deleted(file_path: str, episode_index=None) -> int:
        """把指定 episode 的录制行状态改为 uploaded_deleted（上传成功后本地已删）。

        行保留（用户要求历史可查"已上传，本地已删"），返回受影响行数。
        """
        cond, args = RecordingRepo._where(file_path, episode_index)
        cur = db.conn.execute(
            f"UPDATE recording SET status='uploaded_deleted' WHERE {cond}", args)
        db.conn.commit()
        return cur.rowcount

    @staticmethod
    def mark_deleted(file_path: str, episode_index=None) -> int:
        """把指定 episode 的录制行状态改为 deleted（本地已删、未上传）。

        行保留（历史可查"已删除，未上传"），返回受影响行数。
        """
        cond, args = RecordingRepo._where(file_path, episode_index)
        cur = db.conn.execute(
            f"UPDATE recording SET status='deleted' WHERE {cond}", args)
        db.conn.commit()
        return cur.rowcount

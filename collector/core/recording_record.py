"""
数据模型 —— 录制记录的数据类定义。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from core.helpers import new_id, utcnow


@dataclass
class RecordingRecord:
    """录制历史表中的一条记录。"""

    id: str = field(default_factory=new_id)
    camera_index: int = 0
    camera_name: str = ""
    file_path: str = ""                # v1.1.0 池化：任务目录（一行 = 一个 episode）
    episode_index: int = 0             # v1.1.0 池化：全局 episode 序号（1 起）
    file_size_mb: float = 0.0          # 文件大小（MB）
    duration_sec: float = 0.0          # 录制时长（秒）
    resolution_w: int = 0
    resolution_h: int = 0
    status: str = "completed"          # "completed"（已完成）| "aborted"（已丢弃）| "uploaded_deleted"（已上传，本地已删）| "deleted"（已删除，未上传）
    started_at: str = field(default_factory=utcnow)
    finished_at: str = field(default_factory=utcnow)

    @classmethod
    def from_row(cls, row) -> RecordingRecord:
        """从 sqlite3.Row 构造实例。"""
        return cls(**dict(row))

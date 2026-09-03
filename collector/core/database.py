"""
SQLite 数据库管理器 —— 线程安全的单例模式，管理录制历史记录表。
"""

import sqlite3
import os
import threading
from config.settings import DB_PATH, DATA_DIR


class Database:
    """简单的 SQLite 封装，含建表逻辑和线程本地连接。"""

    _local = threading.local()

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path

    @property
    def conn(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接（自动创建）。"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            c = sqlite3.connect(self._db_path)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            self._local.conn = c
        return self._local.conn

    def close(self):
        """关闭当前线程的数据库连接。"""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ── 建表 SQL ──────────────────────────────────────

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS recording (
        id              TEXT PRIMARY KEY,
        camera_index    INTEGER NOT NULL,
        camera_name     TEXT NOT NULL,
        file_path       TEXT,
        episode_index   INTEGER DEFAULT 0,   -- v1.1.0 池化：任务目录内的全局 episode 序号
        file_size_mb    REAL DEFAULT 0,
        duration_sec    REAL DEFAULT 0,
        resolution_w    INTEGER DEFAULT 0,
        resolution_h    INTEGER DEFAULT 0,
        status          TEXT NOT NULL DEFAULT 'completed',   -- completed | uploaded | aborted | deleted | uploaded_deleted
        started_at      TEXT NOT NULL,
        finished_at     TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_recording_camera ON recording(camera_index);
    CREATE INDEX IF NOT EXISTS idx_recording_date ON recording(started_at);

    CREATE TABLE IF NOT EXISTS upload_task (
        id              TEXT PRIMARY KEY,
        session_path    TEXT NOT NULL,
        session_name    TEXT NOT NULL,
        episode_index   INTEGER DEFAULT 0,   -- v1.1.0 池化：全局 episode 序号
        status          TEXT NOT NULL DEFAULT 'pending',   -- pending | uploading | completed | failed | skipped
        progress        REAL DEFAULT 0.0,
        retry_count     INTEGER DEFAULT 0,
        server_url      TEXT NOT NULL,
        server_session_id TEXT DEFAULT '',
        error_message   TEXT DEFAULT '',
        created_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_upload_status ON upload_task(status);
    CREATE INDEX IF NOT EXISTS idx_upload_session ON upload_task(session_path);
    """

    def init_schema(self):
        """执行建表语句，并做既有库的列迁移（向后兼容）。"""
        self.conn.executescript(self.SCHEMA_SQL)
        # CREATE TABLE IF NOT EXISTS 不给既有表加列，需 ALTER 迁移
        self._ensure_column("recording", "episode_index", "INTEGER DEFAULT 0")
        self._ensure_column("upload_task", "episode_index", "INTEGER DEFAULT 0")
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, decl: str):
        """表缺列时 ALTER TABLE ADD COLUMN（幂等）。"""
        cols = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            self.conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


# 模块级单例
db = Database()

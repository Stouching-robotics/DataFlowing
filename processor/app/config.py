"""Application configuration via environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database. In local mode this normally points to the server through an
    # SSH tunnel (for example 127.0.0.1:15432).
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/data_acq"
    DB_CONNECT_TIMEOUT: float = 5.0
    DB_COMMAND_TIMEOUT: float = 10.0
    DB_POOL_TIMEOUT: float = 5.0

    # Local cache/staging directory. In SFTP mode the authoritative copy is
    # SFTP_ROOT on the storage server.
    STORAGE_DIR: str = "data"
    STORAGE_BACKEND: str = "local"  # local | sftp
    SFTP_HOST: str = "localhost"
    SFTP_PORT: int = 22
    SFTP_USERNAME: str = ""
    SFTP_PASSWORD: str = ""
    SFTP_KEY_FILE: str = ""
    SFTP_ROOT: str = ""
    SFTP_KNOWN_HOSTS: str = ""
    SFTP_STRICT_HOST_KEY: bool = False
    SFTP_CONNECT_TIMEOUT: int = 15

    # Auth
    API_KEY: str = "change-me"
    WORKER_API_KEY: str = ""
    DEFAULT_WORKFLOW_ID: str = ""
    # A cold NAS input-package build can take longer than the normal polling
    # request. Active jobs still send heartbeats during processing.
    WORKER_LEASE_SECONDS: int = 300

    # Upload limits
    MAX_UPLOAD_SIZE: int = 2 * 1024 * 1024 * 1024  # 2GB
    # Uploads are first spooled on the server's local disk.  The authoritative
    # dataset may live on the SSHFS/NAS mount, but receiving a large archive
    # must not make the API wait on remote filesystem writes.
    UPLOAD_STAGING_DIR: str = ""
    ALLOWED_VIDEO_EXTENSIONS: set[str] = {"mp4", "mkv", "mov", "avi", "m4v"}
    ALLOWED_IMAGE_EXTENSIONS: set[str] = {"jpg", "jpeg", "png", "bmp"}

    # Local server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    PUBLIC_BASE_URL: str = "http://127.0.0.1:8000"
    LOG_LEVEL: str = "info"

    # Cleaning
    CLEANING_ENABLED: bool = False

    # Export
    EXPORT_SHARD_SIZE: int = 100000
    VIDEO_SHARD_FRAMES: int = 5000

    model_config = {
        "env_prefix": "",
        "case_sensitive": False,
        "env_file": ".env",
        "extra": "ignore",
    }

    @property
    def storage_root(self) -> Path:
        """Absolute path to storage directory."""
        p = Path(self.STORAGE_DIR)
        if not p.is_absolute():
            p = Path(__file__).parent.parent / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def temp_root(self) -> Path:
        """Project-local temporary workspace; never falls back to the OS temp dir."""
        p = self.storage_root / "tmp"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def upload_staging_root(self) -> Path:
        """Local durable spool for incoming archives.

        ``storage_root`` can be an SSHFS mount.  Keep the receive spool on the
        local machine so the upload response is independent of NAS latency.
        The queue persists its small job records in the authoritative state
        directory and resumes jobs whose spool file still exists.
        """
        raw = str(self.UPLOAD_STAGING_DIR or "").strip()
        p = Path(raw).expanduser() if raw else Path.home() / ".cache" / "egodata-ingest"
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()

# Init subdirectories
# exports/cache 已弃用(导出改到 state/exports,缓存为内存级),不再自动创建
for sub in ["videos", "sessions", "tmp"]:
    (settings.storage_root / sub).mkdir(parents=True, exist_ok=True)

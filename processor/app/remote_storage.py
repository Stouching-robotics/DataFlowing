"""SFTP-backed storage for the local-first deployment.

The application keeps a working copy under ``STORAGE_DIR`` so existing
video/parquet code can continue using pathlib.  SFTP is the authoritative
storage when ``STORAGE_BACKEND=sftp``.  Connections are deliberately opened
per operation because uploads and downloads are long-running and the API is
served by multiple async workers.
"""

from __future__ import annotations

import posixpath
import stat
from pathlib import Path, PurePosixPath

from app.config import settings


def _relative_path(value: str | Path) -> str:
    """Validate a repository-relative path before using it on SFTP."""
    raw = str(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or not raw.strip("/"):
        raise ValueError(f"Unsafe storage path: {value}")
    return path.as_posix()


class SFTPStorage:
    def __init__(self):
        self.root = settings.SFTP_ROOT.rstrip("/") or "/"

    def _connect(self):
        try:
            import paramiko
        except ImportError as exc:  # pragma: no cover - exercised by setup
            raise RuntimeError("SFTP storage requires paramiko; install requirements.txt") from exc

        ssh = paramiko.SSHClient()
        if settings.SFTP_STRICT_HOST_KEY:
            if settings.SFTP_KNOWN_HOSTS:
                ssh.load_host_keys(settings.SFTP_KNOWN_HOSTS)
            else:
                ssh.load_system_host_keys()
            ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
        else:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_args = {
            "hostname": settings.SFTP_HOST,
            "port": settings.SFTP_PORT,
            "username": settings.SFTP_USERNAME,
            "timeout": settings.SFTP_CONNECT_TIMEOUT,
            "banner_timeout": settings.SFTP_CONNECT_TIMEOUT,
            "auth_timeout": settings.SFTP_CONNECT_TIMEOUT,
        }
        if settings.SFTP_KEY_FILE:
            connect_args["key_filename"] = settings.SFTP_KEY_FILE
        else:
            connect_args["password"] = settings.SFTP_PASSWORD
        ssh.connect(**connect_args)
        return ssh, ssh.open_sftp()

    def _remote_path(self, relative: str | Path) -> str:
        return posixpath.join(self.root, _relative_path(relative))

    @staticmethod
    def _mkdir_p(sftp, path: str) -> None:
        current = "/" if path.startswith("/") else ""
        for part in path.split("/"):
            if not part:
                continue
            current = posixpath.join(current, part)
            try:
                sftp.stat(current)
            except OSError:
                sftp.mkdir(current)

    @staticmethod
    def _remove_if_exists(sftp, path: str) -> None:
        try:
            sftp.remove(path)
        except OSError:
            pass

    def exists(self, relative: str | Path) -> bool:
        ssh, sftp = self._connect()
        try:
            sftp.stat(self._remote_path(relative))
            return True
        except OSError:
            return False
        finally:
            sftp.close()
            ssh.close()

    def check(self) -> bool:
        ssh, sftp = self._connect()
        try:
            sftp.stat(self.root)
            return True
        finally:
            sftp.close()
            ssh.close()

    def upload_file(self, local_path: Path, relative: str | Path) -> None:
        remote_path = self._remote_path(relative)
        remote_dir = posixpath.dirname(remote_path)
        temporary = f"{remote_path}.part"
        ssh, sftp = self._connect()
        try:
            self._mkdir_p(sftp, remote_dir)
            self._remove_if_exists(sftp, temporary)
            sftp.put(str(local_path), temporary, confirm=True)
            self._remove_if_exists(sftp, remote_path)
            sftp.rename(temporary, remote_path)
        finally:
            sftp.close()
            ssh.close()

    def upload_tree(self, local_root: Path, remote_relative: str | Path) -> None:
        remote_root = self._remote_path(remote_relative)
        ssh, sftp = self._connect()
        try:
            self._mkdir_p(sftp, remote_root)
            for local_path in local_root.rglob("*"):
                if not local_path.is_file():
                    continue
                relative = local_path.relative_to(local_root).as_posix()
                remote_path = posixpath.join(remote_root, _relative_path(relative))
                self._mkdir_p(sftp, posixpath.dirname(remote_path))
                temporary = f"{remote_path}.part"
                self._remove_if_exists(sftp, temporary)
                sftp.put(str(local_path), temporary, confirm=True)
                self._remove_if_exists(sftp, remote_path)
                sftp.rename(temporary, remote_path)
        finally:
            sftp.close()
            ssh.close()

    def _walk(self, sftp, root: str):
        try:
            entries = sftp.listdir_attr(root)
        except OSError:
            return
        for entry in entries:
            child = posixpath.join(root, entry.filename)
            if stat.S_ISDIR(entry.st_mode):
                yield from self._walk(sftp, child)
            elif stat.S_ISREG(entry.st_mode):
                yield child

    def download_file(self, relative: str | Path, local_path: Path) -> None:
        remote_path = self._remote_path(relative)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = local_path.with_name(f".{local_path.name}.part")
        ssh, sftp = self._connect()
        try:
            sftp.get(remote_path, str(temporary))
            temporary.replace(local_path)
        finally:
            temporary.unlink(missing_ok=True)
            sftp.close()
            ssh.close()

    def download_tree(self, remote_relative: str | Path, local_root: Path) -> None:
        remote_root = self._remote_path(remote_relative)
        local_root.mkdir(parents=True, exist_ok=True)
        ssh, sftp = self._connect()
        try:
            for remote_path in self._walk(sftp, remote_root):
                relative = posixpath.relpath(remote_path, remote_root)
                local_path = local_root / Path(relative)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = local_path.with_name(f".{local_path.name}.part")
                sftp.get(remote_path, str(temporary))
                temporary.replace(local_path)
                temporary.unlink(missing_ok=True)
        finally:
            sftp.close()
            ssh.close()

    def remove_file(self, relative: str | Path) -> None:
        ssh, sftp = self._connect()
        try:
            self._remove_if_exists(sftp, self._remote_path(relative))
        finally:
            sftp.close()
            ssh.close()

    def remove_tree(self, relative: str | Path) -> None:
        root = self._remote_path(relative)
        ssh, sftp = self._connect()
        try:
            files: list[str] = []
            directories: list[str] = []

            def collect(path: str) -> None:
                try:
                    entries = sftp.listdir_attr(path)
                except OSError:
                    return
                for entry in entries:
                    child = posixpath.join(path, entry.filename)
                    if stat.S_ISDIR(entry.st_mode):
                        collect(child)
                        directories.append(child)
                    elif stat.S_ISREG(entry.st_mode):
                        files.append(child)

            collect(root)
            for path in files:
                self._remove_if_exists(sftp, path)
            for path in sorted(directories, key=len, reverse=True):
                try:
                    sftp.rmdir(path)
                except OSError:
                    pass
            try:
                sftp.rmdir(root)
            except OSError:
                pass
        finally:
            sftp.close()
            ssh.close()


def remote_storage() -> SFTPStorage | None:
    if settings.STORAGE_BACKEND.lower() != "sftp":
        return None
    return SFTPStorage()

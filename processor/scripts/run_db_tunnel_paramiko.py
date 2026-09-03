"""Keep a local PostgreSQL SSH tunnel open without executing remote commands."""

from __future__ import annotations

import os
import select
import socketserver
import threading

import paramiko


SSH_HOST = os.getenv("EGODATA_SSH_HOST", "")
SSH_PORT = int(os.getenv("EGODATA_SSH_PORT", "22"))
SSH_USER = os.getenv("EGODATA_SSH_USER", "")
SSH_PASSWORD = os.environ.get("EGODATA_SSH_PASSWORD", "")
LOCAL_HOST = os.getenv("EGODATA_DB_LOCAL_HOST", "127.0.0.1")
LOCAL_PORT = int(os.getenv("EGODATA_DB_LOCAL_PORT", "15432"))
REMOTE_HOST = os.getenv("EGODATA_DB_REMOTE_HOST", "")
REMOTE_PORT = int(os.getenv("EGODATA_DB_REMOTE_PORT", "5432"))


class ForwardServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler_class, transport):
        super().__init__(server_address, handler_class)
        self.transport = transport


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        peer = self.request.getpeername()
        channel = self.server.transport.open_channel(
            "direct-tcpip", (REMOTE_HOST, REMOTE_PORT), peer
        )
        if channel is None:
            raise OSError("SSH server refused the PostgreSQL forwarding channel")
        try:
            sockets = [self.request, channel]
            while sockets:
                readable, _, _ = select.select(sockets, [], [], 1.0)
                for source in readable:
                    payload = source.recv(64 * 1024)
                    if not payload:
                        sockets.clear()
                        break
                    target = channel if source is self.request else self.request
                    target.sendall(payload)
        finally:
            channel.close()


def main() -> None:
    if not SSH_PASSWORD:
        raise SystemExit("EGODATA_SSH_PASSWORD is required")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        SSH_HOST,
        port=SSH_PORT,
        username=SSH_USER,
        password=SSH_PASSWORD,
        look_for_keys=False,
        allow_agent=False,
        timeout=15,
        banner_timeout=15,
        auth_timeout=15,
    )
    transport = client.get_transport()
    if transport is None or not transport.is_active():
        client.close()
        raise SystemExit("SSH transport is not active")

    server = ForwardServer((LOCAL_HOST, LOCAL_PORT), Handler, transport)
    print(
        f"PostgreSQL tunnel: {LOCAL_HOST}:{LOCAL_PORT} -> "
        f"{REMOTE_HOST}:{REMOTE_PORT} via {SSH_USER}@{SSH_HOST}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        client.close()


if __name__ == "__main__":
    main()

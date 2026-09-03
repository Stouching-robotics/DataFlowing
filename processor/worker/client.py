from __future__ import annotations

import json
from pathlib import Path

import httpx


class WorkerClient:
    def __init__(self, server_url: str, api_key: str, timeout: float = 60.0):
        # trust_env=False: the worker talks to the local backend directly.
        # On Windows httpx would otherwise honor the system proxy
        # (e.g. Clash on 127.0.0.1:7897), which cannot reach localhost and
        # turns every claim into a 502 Bad Gateway.
        self.client = httpx.Client(
            base_url=server_url.rstrip("/"),
            headers={"X-API-Key": api_key},
            timeout=httpx.Timeout(timeout, connect=10.0),
            trust_env=False,
        )

    def close(self):
        self.client.close()

    def claim(self, worker_id: str, capabilities: list[str], device: str) -> dict | None:
        response = self.client.post(
            "/api/v1/worker/jobs/claim",
            json={"worker_id": worker_id, "capabilities": capabilities, "device": device},
        )
        if response.status_code == 204:
            return None
        response.raise_for_status()
        return response.json()

    def download_input(self, run_id: str, destination: Path) -> None:
        # A cold input package may need to scan the NAS/SSHFS mount.  The
        # normal 60-second polling timeout must not abort a valid upload.
        download_timeout = httpx.Timeout(None, connect=10.0)
        with self.client.stream(
            "GET", f"/api/v1/worker/jobs/{run_id}/input",
            timeout=download_timeout,
        ) as response:
            response.raise_for_status()
            with destination.open("wb") as output:
                for chunk in response.iter_bytes(1024 * 1024):
                    output.write(chunk)

    def heartbeat(self, run_id: str, worker_id: str, lease_token: str,
                  progress: float, node_states: dict):
        response = self.client.post(
            f"/api/v1/worker/jobs/{run_id}/heartbeat",
            json={"worker_id": worker_id, "lease_token": lease_token,
                  "progress": progress, "node_states": node_states},
        )
        response.raise_for_status()

    def complete(self, run_id: str, worker_id: str, lease_token: str,
                 result_zip: Path, node_states: dict, outputs: dict):
        with result_zip.open("rb") as source:
            response = self.client.post(
                f"/api/v1/worker/jobs/{run_id}/complete",
                data={
                    "worker_id": worker_id,
                    "lease_token": lease_token,
                    "node_states": json.dumps(node_states),
                    "outputs": json.dumps(outputs),
                },
                files={"result_zip": (result_zip.name, source, "application/zip")},
                timeout=None,
            )
        response.raise_for_status()
        return response.json()

    def fail(self, run_id: str, worker_id: str, lease_token: str,
             error: str, retry: bool = True):
        response = self.client.post(
            f"/api/v1/worker/jobs/{run_id}/fail",
            json={"worker_id": worker_id, "lease_token": lease_token,
                  "error": error, "retry": retry},
        )
        response.raise_for_status()

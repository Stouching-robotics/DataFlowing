from __future__ import annotations

import logging
import os
import socket
from pathlib import Path

from app.config import settings
from worker.runner import run_forever


def main():
    logging.basicConfig(level=os.getenv("EGODATA_LOG_LEVEL", "INFO"))
    run_forever(
        server_url=os.getenv("EGODATA_SERVER_URL", "http://127.0.0.1:8000"),
        api_key=os.getenv("EGODATA_WORKER_API_KEY", os.getenv("API_KEY", "change-me")),
        worker_id=os.getenv("EGODATA_WORKER_ID", socket.gethostname()),
        device=os.getenv("EGODATA_DEVICE", "auto"),
        poll_seconds=float(os.getenv("EGODATA_POLL_SECONDS", "2")),
        work_dir=Path(os.getenv("EGODATA_WORK_DIR", str(settings.temp_root / "worker"))),
    )


if __name__ == "__main__":
    main()

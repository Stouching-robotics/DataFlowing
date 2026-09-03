"""Run a standalone local MediaPipe Hand test without using the OS temp dir."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MediaPipe Hand on one local video")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory; defaults to data/tmp/manual-tests/<video-stem>-mediapipe",
    )
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-detection-conf", type=float, default=0.5)
    parser.add_argument("--min-tracking-conf", type=float, default=0.5)
    parser.add_argument("--skeleton-video", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda:0"])
    args = parser.parse_args()

    video = args.video if args.video.is_absolute() else PROJECT_ROOT / args.video
    video = video.resolve()
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")

    output = args.output
    if output is None:
        output = PROJECT_ROOT / "data" / "tmp" / "manual-tests" / f"{video.stem}-mediapipe"
    elif not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()

    # Import after argument parsing so a missing/invalid path fails before
    # loading native MediaPipe/OpenCV libraries.
    from app.processing.modules.mediapipe_hand import run_local

    result = run_local(
        video,
        output,
        {
            "max_hands": args.max_hands,
            "min_detection_conf": args.min_detection_conf,
            "min_tracking_conf": args.min_tracking_conf,
            "generate_skeleton_video": args.skeleton_video,
            "device": args.device,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

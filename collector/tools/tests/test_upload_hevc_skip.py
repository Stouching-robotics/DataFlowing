"""上传 HEVC 跳过单测（v1.0.9）—— 罐头 stderr + 假子进程，不跑真实 ffmpeg。

覆盖：_parse_video_codec 罐头解析（hevc/h265/h264/无流/空）、_is_hevc 三态
（True/False/None）、_precompress_videos 跳过分支（HEVC → 不调编码）与
照旧压缩分支（未知/非 HEVC → 编码 + mapping）。

用法:
    QT_QPA_PLATFORM=offscreen venv/bin/python tools/tests/test_upload_hevc_skip.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

import core.uploader as up
from core.uploader import UploadManager

FAILS = []


def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAILS.append(msg)


class FakeResult:
    def __init__(self, stderr=b"", returncode=0):
        self.stderr = stderr
        self.returncode = returncode


def _simulate_encode(cmd, **k):   # 必须是 **k：调用点传 capture_output/timeout 关键字
    """假编码：把目标文件建出来（供 mapping 判定），返回成功。"""
    try:
        open(cmd[-1], "w").close()
    except OSError:
        pass
    return FakeResult(returncode=0)


def main():
    app = QApplication(sys.argv)
    m = UploadManager()   # __init__ 无副作用（不建线程/不联网）
    statuses = []
    m.task_status.connect(lambda tid, msg: statuses.append(msg))

    # 1. _parse_video_codec 罐头解析
    parse = UploadManager._parse_video_codec
    check(parse("  Stream #0:0[0x1]: Video: hevc (Main) (hvc1 / 0x31637668), "
                "yuv420p, 1280x960") == "hevc", "hevc 流解析")
    check(parse("  Stream #0:0: Video: h264 (High) (avc1 / 0x31637661), "
                "yuv420p, 1280x960") == "h264", "h264 流解析")
    check(parse("  Stream #0:0[0x1]: Video: h265 (Main) (hev1 / 0x31657668), "
                "yuv420p") == "h265", "h265 别名解析")
    check(parse("  Stream #0:0: Audio: aac (LC), 44100 Hz") is None,
          "无 Video 流 → None")
    check(parse("") is None, "空输出 → None")

    # 2. _is_hevc 三态
    orig_run = up.subprocess.run
    try:
        up.subprocess.run = lambda cmd, **k: FakeResult(
            b"Stream #0:0: Video: hevc (Main) (hvc1 / 0x31637668)")
        check(m._is_hevc("ffmpeg", "/x.mp4") is True, "_is_hevc → True (hevc)")
        up.subprocess.run = lambda cmd, **k: FakeResult(
            b"Stream #0:0: Video: h264 (High)")
        check(m._is_hevc("ffmpeg", "/x.mp4") is False, "_is_hevc → False (h264)")
        up.subprocess.run = lambda cmd, **k: (_ for _ in ()).throw(OSError("boom"))
        check(m._is_hevc("ffmpeg", "/x.mp4") is None, "_is_hevc 异常 → None")
        up.subprocess.run = lambda cmd, **k: FakeResult(b"no video stream here")
        check(m._is_hevc("ffmpeg", "/x.mp4") is None,
              "_is_hevc 无流信息 → None")
    finally:
        up.subprocess.run = orig_run

    # 3. 跳过分支：HEVC 源 → mapping 空、不调 ffmpeg 编码、状态播报
    with tempfile.TemporaryDirectory() as td:
        vdir = os.path.join(td, "session", "videos")
        os.makedirs(vdir)
        for i in range(2):
            open(os.path.join(vdir, f"cam{i}.mp4"), "w").close()
        m._find_working_ffmpeg = lambda: "ffmpeg"
        m._is_hevc = lambda ffmpeg, path: True
        encode_called = {"n": 0}
        up.subprocess.run = lambda cmd, **k: encode_called.__setitem__(
            "n", encode_called["n"] + 1) or FakeResult()
        mapping = m._precompress_videos(os.path.join(td, "session"), "t1")
        up.subprocess.run = orig_run
        check(mapping == {} and encode_called["n"] == 0,
              "HEVC 源跳过预压（无 ffmpeg 编码调用）")
        check(any("跳过" in s for s in statuses), "状态播报「跳过预压缩」")

    # 4. 照旧压缩分支：未知/非 HEVC 源 → 编码调用 + mapping 非空
    with tempfile.TemporaryDirectory() as td:
        vdir = os.path.join(td, "session", "videos")
        os.makedirs(vdir)
        src = os.path.join(vdir, "cam0.mp4")
        open(src, "w").close()
        m._find_working_ffmpeg = lambda: "ffmpeg"
        m._is_hevc = lambda ffmpeg, path: None   # 未知 → 安全方向照旧压缩
        up.subprocess.run = _simulate_encode
        mapping = m._precompress_videos(os.path.join(td, "session"), "t2")
        up.subprocess.run = orig_run
        check(len(mapping) == 1 and mapping.get(src),
              "非 HEVC/未知源照旧压缩（mapping 含原文件）")

    # 5. episodes 元数据进包：每段文件直传（新 arcname）/ 旧分片切片回退
    import pyarrow as pa
    import pyarrow.parquet as pq
    import zipfile
    cols = {
        "episode_index": pa.array([1], pa.int64()),
        "task_index": pa.array([0], pa.int64()),
        "start_frame_index": pa.array([0], pa.int64()),
        "end_frame_index": pa.array([9], pa.int64()),
        "length": pa.array([10], pa.int64()),
        "created_at": pa.array([1.7e9], pa.float64()),
        "duration_sec": pa.array([1.0], pa.float64()),
        "drop_stats": pa.array(["{}"], pa.string()),
        "video_codec": pa.array(["{}"], pa.string()),
        "calibration": pa.array(["{}"], pa.string()),
    }
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "meta", "episodes", "chunk-000"))
        # info.json / stats.json 最小快照（_zip_episode 需要存在才打包）
        open(os.path.join(td, "meta", "info.json"), "w").close()
        open(os.path.join(td, "meta", "stats.json"), "w").close()
        open(os.path.join(td, "meta", "tasks.jsonl"), "w").close()
        ep_file = os.path.join(td, "meta", "episodes", "chunk-000",
                               "episode-000.parquet")
        pq.write_table(pa.table(cols), ep_file)
        zip_path = m._zip_episode(td, "T", 1, "t5")
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            check("meta/episodes/chunk-000/episode-000.parquet" in names,
                  "每段文件直传：新 arcname 进包")
            check(not any("_episodes_" in n for n in names),
                  "直传路径无临时切片 arcname")
            rows = pq.read_table(
                zf.open("meta/episodes/chunk-000/episode-000.parquet")).to_pylist()
            check(len(rows) == 1 and rows[0]["episode_index"] == 1,
                  "包内 parquet 单行 episode 1")
        os.remove(zip_path)
        # 旧分片回退：file-000.parquet（分片名 = chunk 号）多行（N=1,2），
        # 上传 N=2 → 单行切片，arcname 为新命名 episode-001.parquet
        os.remove(ep_file)
        shard = os.path.join(td, "meta", "episodes", "chunk-000",
                             "file-000.parquet")
        cols2 = dict(cols)
        for k in cols2:
            if k == "episode_index":
                cols2[k] = pa.array([1, 2], pa.int64())
            else:
                cols2[k] = pa.array([cols2[k][0].as_py(), cols2[k][0].as_py()],
                                    cols2[k].type)
        pq.write_table(pa.table(cols2), shard)
        zip_path = m._zip_episode(td, "T", 2, "t5")
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            check("meta/episodes/chunk-000/episode-001.parquet" in names,
                  "旧分片回退：切片 arcname 为新命名 episode-001")
            rows = pq.read_table(
                zf.open("meta/episodes/chunk-000/episode-001.parquet")).to_pylist()
            check(len(rows) == 1 and rows[0]["episode_index"] == 2,
                  "旧分片回退：包内单行 episode 2")
        os.remove(zip_path)

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项未通过")
        return 1
    print("PASS: upload_hevc_skip 单测全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""池化写入器回归测试（无真机）—— v1.1.0 任务级布局：
编号映射/chunk 边界、两段短录 episode-000/episode-001 文件组、episodes 每段
一文件（与 data/videos 同编号）、任务级 info/stats 累加、abort 文件级
清理、batch_index 权威。

用法:
    QT_QPA_PLATFORM=offscreen venv/bin/python tools/tests/test_pooled_writer.py
"""
import os
import sys
import json
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from config import settings
from core.egodata_writer import EgoDataWriter
from core.helpers import (episode_chunk_file, POOLED_CHUNK_SIZE,
                          pooled_info_path, pooled_stats_path,
                          pooled_data_parquet_path,
                          pooled_episodes_path,
                          episode_row, delete_pooled_episode,
                          list_task_episodes, next_pooled_episode_index,
                          read_recycled_episode, mark_recycled_episode)

FAILS = []


def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAILS.append(msg)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("── 1. 编号映射 / chunk 边界 ──")
    check(episode_chunk_file(1) == (0, 0), "N=1 → chunk-000/episode-000")
    check(episode_chunk_file(1000) == (0, 999),
          "N=1000 → chunk-000/episode-999")
    check(episode_chunk_file(1001) == (1, 0), "N=1001 → chunk-001/episode-000")
    check(episode_chunk_file(2001) == (2, 0), "N=2001 → chunk-002/episode-000")

    out = tempfile.mkdtemp(prefix="pooled_writer_")
    try:
        print("── 2. 两段短录: episode-000 / episode-001 ──")
        w1 = EgoDataWriter()
        ok = w1.start_episode(out, {"head_left_rgb": (480, 640)}, 30.0,
                              sensors=["right_glove"], task_name="PoolTest")
        check(ok and w1.episode_index == 1, "episode 1 启动")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(2):
            w1.write_video_frame("head_left_rgb", frame)
            w1.write_frame_row(i, i / 30.0,
                               sensors={"right_glove": np.ones(256, np.float32)},
                               hardware_ns=1_000_000_000 + i * 33_333_333)
        w1.end_episode()

        w2 = EgoDataWriter()
        ok = w2.start_episode(out, {"head_left_rgb": (480, 640)}, 30.0,
                              sensors=["right_glove"], task_name="PoolTest")
        check(ok and w2.episode_index == 2, "episode 2 序号递增（目录扫描）")
        for i in range(2):
            w2.write_video_frame("head_left_rgb", frame)
            w2.write_frame_row(i, i / 30.0,
                               sensors={"right_glove": np.ones(256, np.float32)})
        w2.end_episode()
        task_dir = w2.task_dir
        check(list_task_episodes(task_dir) == [1, 2],
              f"list_task_episodes: {list_task_episodes(task_dir)}")
        check(os.path.isfile(os.path.join(task_dir, "videos", "chunk-000",
                                          "head_left_rgb", "episode-000.mp4"))
              and os.path.isfile(os.path.join(task_dir, "videos", "chunk-000",
                                              "head_left_rgb", "episode-001.mp4")),
              "视频 episode-000/episode-001 文件组")
        check(os.path.isfile(pooled_data_parquet_path(task_dir, 1))
              and os.path.isfile(pooled_data_parquet_path(task_dir, 2)),
              "data episode-000/episode-001.parquet 文件组")

        # episodes 每段一个文件（与 data/videos 同编号），单行
        rows1 = pq.read_table(pooled_episodes_path(task_dir, 1)).to_pylist()
        rows2 = pq.read_table(pooled_episodes_path(task_dir, 2)).to_pylist()
        check([r["episode_index"] for r in rows1] == [1]
              and [r["episode_index"] for r in rows2] == [2],
              f"episodes 每段单行: {[r['episode_index'] for r in rows1]}, "
              f"{[r['episode_index'] for r in rows2]}")
        check(rows2[0]["duration_sec"] > 0 and rows2[0]["length"] == 2,
              "episode 2 行 length/duration_sec")

        # 任务级 info/stats 累加
        info = read_json(pooled_info_path(task_dir))
        check(info.get("total_episodes") == 2
              and info.get("format") == "pooled_episodes_v1",
              f"info total_episodes=2: {info.get('total_episodes')}")
        stats = read_json(pooled_stats_path(task_dir))
        check(stats["observation.right_glove"]["mean"][0] == 1.0,
              "stats.json 全局 mean（两段合并）")
        check(stats["observation.right_glove"]["count"] == 4,
              f"stats.json 自含累加器 count=4: "
              f"{stats['observation.right_glove']['count']}")

        print("── 3. batch_index 权威（进度不回退） ──")
        n = next_pooled_episode_index(task_dir, batch_index=50)
        check(n == 50, f"batch_index=50 覆盖扫描: {n}")
        n = next_pooled_episode_index(task_dir, batch_index=0)
        check(n == 3, f"batch_index=0 退化扫描: {n}")

        print("── 4. abort 文件级清理 ──")
        w3 = EgoDataWriter()
        ok = w3.start_episode(out, {"head_left_rgb": (480, 640)}, 30.0,
                              sensors=["right_glove"], task_name="PoolTest",
                              batch_index=3)
        check(ok and w3.episode_index == 3, "episode 3 启动")
        w3.write_video_frame("head_left_rgb", frame)
        w3.write_frame_row(0, 0.0,
                           sensors={"right_glove": np.ones(256, np.float32)})
        w3.abort_episode()
        check(not os.path.exists(os.path.join(
                  task_dir, "videos", "chunk-000", "head_left_rgb",
                  "episode-002.mp4")),
              "abort 后无 episode-002 视频")
        check(not os.path.exists(pooled_data_parquet_path(task_dir, 3)),
              "abort 后无 episode-002 parquet")
        check([r["episode_index"] for r in pq.read_table(
                   pooled_episodes_path(task_dir, 1)).to_pylist()] == [1]
              and [r["episode_index"] for r in pq.read_table(
                   pooled_episodes_path(task_dir, 2)).to_pylist()] == [2]
              and not os.path.isfile(pooled_episodes_path(task_dir, 3)),
              "abort 后 episodes 每段文件仍 2 个（无 3）")
        stats = read_json(pooled_stats_path(task_dir))
        check(stats["observation.right_glove"]["mean"][0] == 1.0,
              "abort 不回滚 stats（无合并发生）")

        # 下一段仍取 3（abort 不占号；回退标记 3 优先复用）
        w4 = EgoDataWriter()
        ok = w4.start_episode(out, {"head_left_rgb": (480, 640)}, 30.0,
                              sensors=["right_glove"], task_name="PoolTest")
        check(ok and w4.episode_index == 3, "abort 后序号不跳（仍为 3）")
        w4.write_video_frame("head_left_rgb", frame)
        w4.write_frame_row(0, 0.0,
                           sensors={"right_glove": np.ones(256, np.float32)})
        w4.end_episode()
        check(os.path.isfile(os.path.join(task_dir, "videos", "chunk-000",
                                          "head_left_rgb", "episode-002.mp4")),
              "episode 3 复用 episode-002")

        print("── 5. 异常终止回退标记（batch 水位漂移也不跳号） ──")
        # 中止释放的号在 batch_index 跑前时仍被复用（不占号语义）；
        # 正常完成后标记清除；号已被占时自动放弃回退
        w5 = EgoDataWriter()
        ok = w5.start_episode(out, {"head_left_rgb": (480, 640)}, 30.0,
                              sensors=["right_glove"], task_name="PoolTest",
                              batch_index=4)
        check(ok and w5.episode_index == 4, "episode 4 启动（batch=4）")
        w5.write_video_frame("head_left_rgb", frame)
        w5.abort_episode()
        check(read_recycled_episode(task_dir) == 4,
              "abort 后 recycled 标记=4")
        n = next_pooled_episode_index(task_dir, batch_index=50)
        check(n == 4, f"batch=50 漂移不跳号（回退复用 4）: {n}")
        w6 = EgoDataWriter()
        ok = w6.start_episode(out, {"head_left_rgb": (480, 640)}, 30.0,
                              sensors=["right_glove"], task_name="PoolTest",
                              batch_index=50)
        check(ok and w6.episode_index == 4, "回退号 4 被复用")
        w6.write_video_frame("head_left_rgb", frame)
        w6.write_frame_row(0, 0.0,
                           sensors={"right_glove": np.ones(256, np.float32)})
        w6.end_episode()
        check(read_recycled_episode(task_dir) == 0,
              "正常完成后 recycled 标记清除")
        # 标记的号已被占（跨机共享目录）→ 自动放弃，走 batch 下限
        mark_recycled_episode(task_dir, 2)
        n = next_pooled_episode_index(task_dir, batch_index=50)
        check(n == 50, f"占用号自动放弃（batch=50 生效）: {n}")
        check(read_recycled_episode(task_dir) == 0,
              "占用后标记自动清除")

        print("── 6. episode_row 直读 + delete 每段文件彻底删除 ──")
        check(episode_row(task_dir, 1).get("episode_index") == 1
              and episode_row(task_dir, 2).get("episode_index") == 2,
              "episode_row 直读每段文件")
        check(episode_row(task_dir, 99) == {},
              "episode_row 缺失返回 {}")
        check(delete_pooled_episode(task_dir, 1),
              "delete episode 1 返回 True")
        check(not os.path.isfile(pooled_episodes_path(task_dir, 1))
              and not os.path.isdir(os.path.join(task_dir, "_trash")),
              "episodes 每段文件彻底删除（无 _trash）")
        check(episode_row(task_dir, 1) == {},
              "delete 后 episode_row(1) 返回 {}")
    finally:
        shutil.rmtree(out, ignore_errors=True)

    print("── 7. 旧分片回退（多行分片） ──")
    out2 = tempfile.mkdtemp(prefix="pooled_legacy_")
    try:
        os.makedirs(os.path.join(out2, "meta", "episodes", "chunk-000"),
                    exist_ok=True)
        shard = os.path.join(out2, "meta", "episodes", "chunk-000",
                             "file-000.parquet")
        pq.write_table(pa.table({
            "episode_index": pa.array([4, 5], pa.int64()),
            "task_index": pa.array([0, 0], pa.int64()),
            "start_frame_index": pa.array([0, 0], pa.int64()),
            "end_frame_index": pa.array([9, 9], pa.int64()),
            "length": pa.array([10, 10], pa.int64()),
            "created_at": pa.array([1.7e9, 1.7e9], pa.float64()),
            "duration_sec": pa.array([1.0, 1.0], pa.float64()),
            "drop_stats": pa.array(["{}", "{}"], pa.string()),
            "video_codec": pa.array(["{}", "{}"], pa.string()),
            "calibration": pa.array(["{}", "{}"], pa.string()),
        }), shard)
        check(episode_row(out2, 5).get("episode_index") == 5,
              "旧分片回退：episode_row(5) 命中分片行")
        check(episode_row(out2, 1) == {},
              "旧分片回退：episode_row(1) 缺失返回 {}")
        check(delete_pooled_episode(out2, 5),
              "旧分片 delete episode 5 返回 True")
        check(episode_row(out2, 5) == {}
              and episode_row(out2, 4).get("episode_index") == 4,
              "旧分片 delete 只删第 5 行，第 4 行保留")
    finally:
        shutil.rmtree(out2, ignore_errors=True)

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项未通过")
        return 1
    print("PASS: 池化写入器测试全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

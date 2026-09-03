"""DropStats 计数器单测（v1.0.9）。

覆盖：inc 累计、snapshot 副本语义、clear、多线程自增一致性。

用法:
    venv/bin/python tools/tests/test_drop_stats.py
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.pipeline import DropStats

FAILS = []


def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAILS.append(msg)


def main():
    ds = DropStats()
    check(ds.snapshot() == {}, "初始为空")

    ds.inc("a")
    ds.inc("a", 2)
    ds.inc("b", 5)
    check(ds.snapshot() == {"a": 3, "b": 5}, "inc 累计")

    snap = ds.snapshot()
    snap["a"] = 999
    check(ds.snapshot()["a"] == 3, "snapshot 返回副本（改副本不影响计数）")

    ds.clear()
    check(ds.snapshot() == {}, "clear 清空")

    # 多线程自增：8 线程 × 5000 次，两个键各半
    N_THREADS, N_INC = 8, 5000
    def worker(key):
        for _ in range(N_INC):
            ds.inc(key)
    ts = [threading.Thread(target=worker, args=(f"k{i % 2}",))
          for i in range(N_THREADS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    snap = ds.snapshot()
    check(snap == {"k0": 20000, "k1": 20000}, f"多线程自增无丢失: {snap}")

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项未通过")
        return 1
    print("PASS: drop_stats 单测全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

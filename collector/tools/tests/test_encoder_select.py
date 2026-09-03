"""编码器选择单测（v1.0.9）—— 假探针，不跑真实 ffmpeg。

覆盖：auto 三档顺序（nvenc → x265 达标 → x264）、x265 门槛判定、
显式指定（跳门槛/失败回退）、None 兜底、进程内缓存、参数映射。

用法:
    venv/bin/python tools/tests/test_encoder_select.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import core.encoder_probe as ep
from config import settings

FAILS = []


def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAILS.append(msg)


PASS_PROBE = lambda *a, **k: {"per_stream_fps": [55.0, 55.0],
                              "min_fps": 55.0, "passed": True,
                              "threshold_fps": 45.0, "streams": 2,
                              "frames": 45, "elapsed_s": 1.0}
LOW_PROBE = lambda *a, **k: {"per_stream_fps": [20.0, 20.0],
                             "min_fps": 20.0, "passed": False,
                             "threshold_fps": 45.0, "streams": 2,
                             "frames": 45, "elapsed_s": 2.3}


def main():
    # 环境：单个假 ffmpeg 二进制，支持全部编码器
    ep._FFMPEG_CACHE = ["/fake/ffmpeg"]
    ep._ENCODER_CACHE[("/fake/ffmpeg", "libx264")] = True
    ep._ENCODER_CACHE[("/fake/ffmpeg", "libx265")] = True
    ep._ENCODER_CACHE[("/fake/ffmpeg", "hevc_nvenc")] = True

    # 1. auto：nvenc 探针可用 → 首选 nvenc
    ep._SELECT_CACHE.clear()
    ep.probe_nvenc = lambda *a, **k: True
    c = ep.select_encoder("auto", 1280, 960, 30, 2)
    check(c is not None and c.encoder == "hevc_nvenc",
          f"auto 链首选 hevc_nvenc（得到 {c and c.encoder}）")
    check(c is not None and c.codec == "HEVC" and c.args[-1] == "hvc1",
          "nvenc 选择带 HEVC/hvc1")
    check(c is not None and c.args == ep.encoder_args("hevc_nvenc"),
          "nvenc 参数映射一致")

    # 2. nvenc 失败 → x265 达标 → libx265
    ep._SELECT_CACHE.clear()
    ep.probe_nvenc = lambda *a, **k: False
    ep.probe_x265 = PASS_PROBE
    c = ep.select_encoder("auto", 1280, 960, 30, 2)
    check(c is not None and c.encoder == "libx265"
          and c.probe["min_fps"] == 55.0,
          "nvenc 失败且 x265 达标 → libx265")
    check(c is not None and c.crf == settings.RECORD_VIDEO_CRF,
          "x265 crf=RECORD_VIDEO_CRF")

    # 3. x265 不达标 → x264 兜底
    ep._SELECT_CACHE.clear()
    ep.probe_x265 = LOW_PROBE
    c = ep.select_encoder("auto", 1280, 960, 30, 2)
    check(c is not None and c.encoder == "libx264"
          and c.crf == settings.RECORD_VIDEO_X264_CRF,
          "x265 不达标 → x264 兜底（CRF 回退档）")

    # 4. 显式 x265：跳过速度探针（探针被调用即失败）
    ep._SELECT_CACHE.clear()
    def boom(*a, **k):
        raise AssertionError("probe should not be called")
    ep.probe_x265 = boom
    ep.probe_nvenc = boom
    try:
        c = ep.select_encoder("x265", 1280, 960, 30, 2)
        ok = c is not None and c.encoder == "libx265"
    except AssertionError:
        ok = False
    check(ok, "显式 x265 跳过速度门槛（不调探针）")

    # 5. 显式 nvenc 失败 → auto 链回退（此处 x265 达标）
    ep._SELECT_CACHE.clear()
    ep.probe_nvenc = lambda *a, **k: False
    ep.probe_x265 = PASS_PROBE
    c = ep.select_encoder("nvenc", 1280, 960, 30, 2)
    check(c is not None and c.encoder == "libx265",
          "显式 nvenc 失败 → auto 链回退")

    # 6. 显式 x264：不调任何探针
    ep._SELECT_CACHE.clear()
    calls = {"n": 0}
    ep.probe_nvenc = lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or True
    ep.probe_x265 = lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or PASS_PROBE()
    c = ep.select_encoder("x264", 1280, 960, 30, 2)
    check(c is not None and c.encoder == "libx264" and calls["n"] == 0,
          "显式 x264 跳过探针直接采用")

    # 7. 支持表缺 libx264 → None（全链失败）
    ep._SELECT_CACHE.clear()
    ep._ENCODER_CACHE[("/fake/ffmpeg", "libx264")] = False
    ep.probe_nvenc = lambda *a, **k: False
    ep.probe_x265 = LOW_PROBE
    c = ep.select_encoder("auto", 1280, 960, 30, 2)
    check(c is None, "无任何可用编码器 → None")
    ep._ENCODER_CACHE[("/fake/ffmpeg", "libx264")] = True

    # 8. 显式 x265 但二进制不支持 → auto 链回退 + 日志播报
    ep._SELECT_CACHE.clear()
    ep._ENCODER_CACHE[("/fake/ffmpeg", "libx265")] = False
    ep.probe_nvenc = lambda *a, **k: False
    ep.probe_x265 = PASS_PROBE
    logs = []
    c = ep.select_encoder("x265", 1280, 960, 30, 2, log=logs.append)
    check(c is not None and c.encoder == "libx264"
          and any("回退" in l for l in logs),
          "显式 x265 不支持 → auto 回退且播报原因")
    ep._ENCODER_CACHE[("/fake/ffmpeg", "libx265")] = True

    # 9. 进程内缓存：同参第二次不重探（返回同一对象）
    ep._SELECT_CACHE.clear()
    calls = {"n": 0}
    ep.probe_nvenc = lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or True
    c1 = ep.select_encoder("auto", 640, 480, 30, 1)
    n1 = calls["n"]
    c2 = ep.select_encoder("auto", 640, 480, 30, 1)
    check(c1 is c2 and calls["n"] == n1, "进程内缓存：同参不重探")

    # 10. 无 ffmpeg 二进制 → None
    ep._FFMPEG_CACHE = []
    ep._SELECT_CACHE.clear()
    check(ep.select_encoder("auto", 640, 480, 30, 1) is None,
          "无 ffmpeg → None")

    # 11. 参数映射表（x264 用回退档、x265/nvenc 用 HEVC 档）
    check(ep.encoder_args("libx265")[-1] == "hvc1", "libx265 带 hvc1")
    check(ep.encoder_args("libx265", 28)[1] == "28", "libx265 crf 透传")
    check("-cq" in ep.encoder_args("hevc_nvenc"), "nvenc 用 -cq")
    check(ep.encoder_args("libx264")[-2:] ==
          ["-crf", str(settings.RECORD_VIDEO_X264_CRF)],
          "libx264 用回退 CRF")

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项未通过")
        return 1
    print("PASS: encoder_select 单测全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

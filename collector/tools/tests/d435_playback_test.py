"""D435 回放冒烟测试（offscreen）—— 验证 RGB + 12-bit 灰深度 MP4 进入回放。"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from ui.playback_dialog import PlaybackDialog


def main():
    app = QApplication(sys.argv)
    dlg = PlaybackDialog()

    # 用 GUI 冒烟测试录制的任务目录（池化布局，任务名留空 → session）
    ep = "/tmp/d435_gui_test/session"
    if not os.path.isfile(os.path.join(ep, "meta", "info.json")):
        print("FAIL: 找不到 GUI 测试录制的任务目录")
        return 1
    print(f"加载: {ep}")

    dlg._load_session(ep)

    # 等后台加载完成(_on_session_loaded 打开视频/深度访问器)
    deadline = time.time() + 20
    while time.time() < deadline and not dlg._caps \
            and not dlg._depth_videos:
        app.processEvents()
        time.sleep(0.05)

    ids = dlg._camera_ids
    print(f"_camera_ids: {ids}")
    print(f"_caps: {sorted(dlg._caps.keys())}")
    print(f"_depth_videos: {sorted(dlg._depth_videos.keys())}")

    # 期望槽集 = video_extensions 键（RGB mp4 + 深度 12-bit 灰 mp4/mkv；
    # 槽名随 GUI 用户命名，按任务信息推导不硬编码）
    with open(os.path.join(ep, "meta", "info.json"), encoding="utf-8") as f:
        info = json.load(f)
    expect = set(info["video_extensions"].keys())
    have = set(dlg._caps) | set(dlg._depth_videos)
    if set(ids) != expect or have != expect:
        print(f"FAIL: 回放集不完整 (期望 {sorted(expect)})")
        dlg.close()
        return 1

    # 逐帧 seek 到中间帧,读取各槽画面
    mid = dlg._total_frames // 2
    dlg._seek(mid)
    app.processEvents()
    ok = True
    for sid in ids:
        dv = dlg._depth_videos.get(sid)
        if dv is not None:
            frame = dv.read(0)   # 深度访问器：12-bit 解码 → JET BGR
            shape = None if frame is None else frame.shape
            print(f"  {sid}: 深度可读 shape={shape}")
            if shape is None:
                ok = False
            continue
        cap = dlg._caps.get(sid)
        if cap is None:
            ok = False
            continue
        cap.set(0, 0)  # 直接读 cap 验证流本身
        got, frame = cap.read()
        shape = None if frame is None else frame.shape
        print(f"  {sid}: 可读={got} shape={shape}")
        if not got or shape is None:
            ok = False
    if not ok:
        print("FAIL: 视频流读取失败")
        dlg.close()
        return 1

    # 播放几帧(走 _tick 路径)
    dlg._toggle_play()
    for _ in range(30):
        app.processEvents()
        time.sleep(0.03)
    dlg._stop()
    print(f"播放推进: {dlg._play_idx} / {dlg._total_frames} 帧")

    dlg.close()
    print("PASS: 回放冒烟测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

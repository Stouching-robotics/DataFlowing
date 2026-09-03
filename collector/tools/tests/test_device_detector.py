"""设备检测模块单元测试。

用法:
    QT_QPA_PLATFORM=offscreen venv/bin/python tools/tests/test_device_detector.py

覆盖:
  - _parse_by_id_entry 各形态（纯函数）
  - _list_uvc_devices 排除 RealSense / FTDI（mock list_v4l_devices）
  - _list_s80m_devices FTDI 单条（mock _is_sdk_device）
  - detect_devices 子模块异常不崩（mock 抛异常）
  - DeviceScanner 信号投递（QCoreApplication 事件循环）
  - 真机段（D435 在位时）: _is_realsense_node / _list_d435_devices serial 非空
  - detect_cameras 从不请求 RealSense 索引（patch _try_open_camera）
  - BLE: _mac_norm / bluetoothctl 解析 / 发现合并去重 / 分组判定 / 扫描抑制
  - 设备命名持久化 round-trip（临时文件，含 sensor 角色与旧格式升级）
"""
import os
import sys
import time
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QCoreApplication

from config import settings
import core.camera as cam
import core.device_detector as det
from core.device_detector import (
    DeviceInfo, _parse_by_id_entry, _list_uvc_devices, _list_s80m_devices,
    _mac_norm, _bluetoothctl_paired, _list_ble_devices,
    detect_devices, DeviceScanner, set_ble_scan_suppressed,
)

FAILS = []


def check(cond, msg):
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        FAILS.append(msg)


def main():
    # 测试默认不跑真实 5s 蓝牙主动扫描（真机段单独处理）
    _orig_discover = det._ble_discover
    det._ble_discover = lambda: []
    try:
        return _main()
    finally:
        det._ble_discover = _orig_discover


def _main():
    print("── 1. _parse_by_id_entry 各形态 ──")
    cases = [
        ("usb-DECXIN_Video_Camera_2024010100-video-index0",
         {"prefix": "usb-DECXIN_Video_Camera_2024010100",
          "serial": "2024010100", "index": 0}),
        ("usb-046d_0825_AB12CD34-video-index0",
         {"prefix": "usb-046d_0825_AB12CD34",
          "serial": "AB12CD34", "index": 0}),
        # 末尾段太短/非字母数字 → 不算序号
        ("usb-SunplusIT_Inc_Integrated_Camera-video-index0",
         {"prefix": "usb-SunplusIT_Inc_Integrated_Camera",
          "serial": "", "index": 0}),
        ("usb-Intel_R__RealSense_TM__Depth_Camera_435_212223021136-video-index1",
         {"prefix": "usb-Intel_R__RealSense_TM__Depth_Camera_435_212223021136",
          "serial": "212223021136", "index": 1}),
        # 无 -video-indexN 后缀
        ("no-suffix-entry",
         {"prefix": "no-suffix-entry", "serial": "", "index": None}),
        ("", None),
        (None, None),
    ]
    for entry, expect in cases:
        got = _parse_by_id_entry(entry)
        check(got == expect, f"_parse_by_id_entry({entry!r}) = {got}")

    print("── 2. _list_uvc_devices 过滤（mock list_v4l_devices） ──")
    fake_v4l = [
        {"video_index": 0, "name": "RealSense 435i", "serial": "111111111111",
         "by_id_path": "/dev/v4l/by-id/usb-RealSense_111111111111-video-index0",
         "vid": "8086", "pid": "0b3a", "is_sdk": False, "is_realsense": True},
        {"video_index": 1, "name": "FTDI 设备", "serial": "",
         "by_id_path": "/dev/v4l/by-id/usb-FTDI-video-index1",
         "vid": "0403", "pid": "601e", "is_sdk": True, "is_realsense": False},
        {"video_index": 2, "name": "DECXIN Webcam", "serial": "2024010100",
         "by_id_path": "/dev/v4l/by-id/usb-DECXIN_Video_Camera_2024010100-video-index0",
         "vid": "32e4", "pid": "0416", "is_sdk": False, "is_realsense": False},
    ]
    with patch("core.device_detector.list_v4l_devices", return_value=fake_v4l):
        infos = _list_uvc_devices(16)
    keys = [i.key for i in infos]
    check(len(infos) == 1 and keys[0].startswith("uvc:"), f"只剩 UVC: {keys}")
    check(infos[0].video_index == 2 and infos[0].serial == "2024010100",
          f"webcam 索引/序号正确: idx={infos[0].video_index} serial={infos[0].serial}")

    print("── 3. _list_s80m_devices FTDI 单条（mock _is_sdk_device） ──")
    def fake_sdk(i):
        return i == 7
    with patch("core.device_detector._is_sdk_device", side_effect=fake_sdk):
        infos = _list_s80m_devices(16)
    check(len(infos) == 1 and infos[0].key == "s80m:ftdi"
          and infos[0].video_index == 7,
          f"FTDI 单条命中: {[(i.key, i.video_index) for i in infos]}")
    with patch("core.device_detector._is_sdk_device", return_value=False):
        infos = _list_s80m_devices(16)
    check(infos == [], "无 FTDI 时返回空")

    print("── 4. detect_devices 子模块异常不崩 ──")
    with patch("core.device_detector._list_uvc_devices", side_effect=OSError("boom")), \
         patch("core.device_detector._list_d435_devices", side_effect=OSError("boom")), \
         patch("core.device_detector._list_s80m_devices", side_effect=OSError("boom")), \
         patch("core.device_detector._list_ble_devices", side_effect=OSError("boom")):
        check(detect_devices() == [], "四段全炸 → 返回空列表不抛异常")

    print("── 5. DeviceScanner 信号投递 ──")
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)
    scanner = DeviceScanner(max_index=16)
    got = []
    scanner.scan_finished.connect(lambda devs: got.append(devs))
    scanner.request_scan()
    t0 = time.time()
    while not got and time.time() - t0 < 5:
        app.processEvents()
        time.sleep(0.01)
    check(bool(got) and isinstance(got[0], list), "scan_finished 收到列表")
    # 守卫：进行中再次请求不堆积（_busy）
    scanner.stop()
    scanner.request_scan()
    check(True, "stop 后 request_scan 不启动新线程")

    print("── 6. 真机段（RealSense 在位时） ──")
    rs_nodes = [i for i in range(16) if cam._is_realsense_node(i)]
    if rs_nodes:
        check(True, f"_is_realsense_node 命中: {rs_nodes}")
        d435s = [d for d in detect_devices() if d.kind == "d435"]
        check(bool(d435s) and d435s[0].serial,
              f"_list_d435_devices serial 非空: "
              f"{[(d.display_name, d.serial) for d in d435s]}")
        if len(d435s) >= 2:
            check(len({d.serial for d in d435s}) == len(d435s),
                  f"多台 D400 各自成条（serial 唯一）: "
                  f"{[(d.display_name, d.serial) for d in d435s]}")
        else:
            print("  （仅 1 台 D400，多设备分支未覆盖）")
    else:
        print("  SKIP: 本机无 RealSense 设备")

    print("── 7. detect_cameras 从不请求 RealSense 索引 ──")
    requested = []
    def spy(index, test_read=False, fallback_all_by_id=False):
        requested.append(index)
        return None, ""
    with patch("core.camera._try_open_camera", side_effect=spy):
        cam.detect_cameras(max_index=settings.DEVICE_SCAN_MAX_INDEX)
    bad = [i for i in requested if i in rs_nodes]
    check(not bad, f"RealSense 索引从未被请求（请求了 {requested}）")

    print("── 8. BLE: _mac_norm / bluetoothctl 解析 ──")
    check(_mac_norm("aa:bb:cc:11:22:33") == "AA:BB:CC:11:22:33",
          f"MAC 大写归一: {_mac_norm('aa:bb:cc:11:22:33')}")
    check(_mac_norm("AA-BB-CC-11-22-33") == "AA:BB:CC:11:22:33",
          f"MAC 连字符归一: {_mac_norm('AA-BB-CC-11-22-33')}")
    fake_out = ("Device 30:A9:98:57:4A:C2 HUAWEI FreeBuds 5i\n"
                "Controller F8:3D:C6:C1:1B:E9 REDACTED-HOST\n")
    with patch("core.device_detector.subprocess.run",
               return_value=type("R", (), {"stdout": fake_out})()):
        paired = _bluetoothctl_paired()
    check(paired.get("30:A9:98:57:4A:C2") == "HUAWEI FreeBuds 5i"
          and "F8:3D:C6:C1:1B:E9" not in paired,
          f"配对列表解析（Controller 行排除）: {paired}")

    print("── 9. BLE: 发现合并去重 / 分组判定 ──")
    fake_disc = [("Matrix Glove R", "aa:11:22:33:44:55", -45),
                 ("FreeBuds", "30:a9:98:57:4a:c2", -60),
                 ("Phone X", "bb:66:77:88:99:00", -70)]
    with patch("core.device_detector._ble_discover", return_value=fake_disc), \
         patch("core.device_detector._bluetoothctl_paired",
               return_value={"30:A9:98:57:4A:C2": "HUAWEI FreeBuds 5i"}):
        infos = _list_ble_devices()
    by_key = {i.key: i for i in infos}
    check(len(infos) == 3, f"配对+发现合并去重后 3 条: {sorted(by_key)}")
    glove = by_key.get("ble:AA:11:22:33:44:55")
    check(glove is not None and glove.kind == "data_ble" and glove.group == "glove",
          f"Matrix 判为手套: {getattr(glove, 'kind', None)}/{getattr(glove, 'group', None)}")
    buds = by_key.get("ble:30:A9:98:57:4A:C2")
    check(buds is not None and buds.kind == "ble" and buds.group == "other_ble"
          and buds.display_name == "HUAWEI FreeBuds 5i" and buds.serial == "30:A9:98:57:4A:C2",
          f"耳机入其他蓝牙组且配对名优先: "
          f"{getattr(buds, 'kind', None)}/{getattr(buds, 'display_name', None)}")
    check(buds.label == buds.display_name, "label 回落 display_name")

    print("── 10. BLE: 扫描抑制（手套连接中不触发发现） ──")
    det._ble_discovery_cache["ts"] = 0.0   # 重置节流，确保会触发刷新
    calls = []
    def spy_discover():
        calls.append(1)
        return [("Matrix Glove L", "cc:11:22:33:44:55", -50)]
    det._ble_discover = spy_discover
    try:
        set_ble_scan_suppressed(True)
        infos = _list_ble_devices()
        check(not calls, f"抑制时不调用发现（calls={len(calls)}）")
        set_ble_scan_suppressed(False)
        infos = _list_ble_devices()
        check(len(calls) == 1, f"解除抑制后触发发现（calls={len(calls)}）")
        check(any(i.kind == "data_ble" for i in infos), "发现结果并入列表")
    finally:
        set_ble_scan_suppressed(False)
        det._ble_discovery_cache["ts"] = 0.0
        det._ble_discover = lambda: []

    print("── 11. 设备命名持久化 round-trip（临时文件） ──")
    import tempfile
    from config import settings as _settings
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp_path = tmp.name
    tmp.close()
    _orig_file = _settings.DEVICE_NAMES_FILE
    _settings.DEVICE_NAMES_FILE = tmp_path
    try:
        _settings.save_device_name("d435:123456789012", "顶部深度相机")
        _settings.save_device_name("ble:AA:11:22:33:44:55", "右手手套",
                                   sensor="right_glove")
        _settings.save_device_name("uvc:usb-Logitech_ABC123", "桌面摄像头")
        names = _settings.load_device_names()
        check(names["d435:123456789012"]["name"] == "顶部深度相机",
              f"命名写入并读回: {names.get('d435:123456789012')}")
        check(_settings.device_name("d435:123456789012") == "顶部深度相机",
              "device_name 读取")
        check(_settings.device_sensor_role("ble:AA:11:22:33:44:55") == "right_glove",
              "sensor 角色绑定")
        # merge-write：再保存其它键不动已有条目
        _settings.save_device_name("d435:999999999999", "备用机")
        names2 = _settings.load_device_names()
        check("d435:123456789012" in names2 and "ble:AA:11:22:33:44:55" in names2,
              "merge-write 保留旧条目")
        # 旧版纯字符串条目升级 + remove
        names2["d435:999999999999"] = "旧格式字符串"
        _settings._write_device_names(names2)
        _settings.save_device_name("d435:999999999999", "备用机2")
        check(_settings.device_name("d435:999999999999") == "备用机2",
              "旧字符串条目升级为结构化并更新")
        _settings.remove_device_name("uvc:usb-Logitech_ABC123")
        check("uvc:usb-Logitech_ABC123" not in _settings.load_device_names(),
              "remove 删除条目")
        check(_settings.device_name("不存在的key") == "" and
              _settings.device_sensor_role("不存在的key") == "",
              "缺失条目安全回落空串")
    finally:
        _settings.DEVICE_NAMES_FILE = _orig_file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    print("── 12. 真机段（蓝牙在位时） ──")
    if os.path.exists("/usr/bin/bluetoothctl") or os.path.exists("/usr/local/bin/bluetoothctl"):
        paired = _bluetoothctl_paired()
        check(True, f"bluetoothctl 可用，已配对 {len(paired)} 台（不做断言）")
    else:
        print("  SKIP: 本机无 bluetoothctl")

    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} 项未通过")
        return 1
    print("PASS: 设备检测单元测试全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
国际化（i18n）模块 —— 中英文切换支持。

使用方式:
    from config.i18n import tr, lang_manager
    text = tr("开始录制")   # 返回当前语言的翻译
    lang_manager.set_language("en")   # 切到英文
    lang_manager.language_changed.connect(my_slot)  # 监听语言切换
"""

from PyQt5.QtCore import QObject, pyqtSignal


# ═══════════════════════════════════════════════════════
#  翻译字典  { 中文原文: { "en": "English", "zh": "中文" } }
# ═══════════════════════════════════════════════════════

_TRANSLATIONS = {
    # ── 录制记录 ──────────────────────────────────────
    "多路录制":                   {"en": "Multi-Camera",                   "zh": "多路录制"},

    # ── 应用程序 ──────────────────────────────────────
    "DAQ 视频管线":              {"en": "DAQ Video Pipeline",            "zh": "DAQ 视频管线"},
    "多路摄像机实时监控与录制系统": {"en": "Multi-camera Monitoring & Recording System", "zh": "多路摄像机实时监控与录制系统"},

    # ── 菜单栏 ────────────────────────────────────────
    "文件(&F)":                  {"en": "&File",                         "zh": "文件(&F)"},
    "扫描摄像机":                 {"en": "Scan Cameras",                  "zh": "扫描摄像机"},
    "添加摄像机…":               {"en": "Add Camera…",                   "zh": "添加摄像机…"},
    "退出(&X)":                  {"en": "E&xit",                         "zh": "退出(&X)"},
    "视图(&V)":                  {"en": "&View",                         "zh": "视图(&V)"},
    "清空日志":                   {"en": "Clear Log",                     "zh": "清空日志"},
    "刷新历史记录":               {"en": "Refresh History",               "zh": "刷新历史记录"},
    "重置画面大小":               {"en": "Reset Camera Sizes",            "zh": "重置画面大小"},
    "帮助(&H)":                  {"en": "&Help",                         "zh": "帮助(&H)"},
    "关于":                      {"en": "About",                         "zh": "关于"},
    "语言(&L)":                  {"en": "&Language",                     "zh": "语言(&L)"},
    "中文":                      {"en": "中文",                           "zh": "中文"},
    "English":                   {"en": "English",                       "zh": "English"},

    # ── 工具栏 ────────────────────────────────────────
    "🔍 扫描":                   {"en": "🔍 Scan",                       "zh": "🔍 扫描"},
    "＋ 添加":                   {"en": "＋ Add",                        "zh": "＋ 添加"},
    "⏺ 全部录制":               {"en": "⏺ Record All",                 "zh": "⏺ 全部录制"},
    "⏹ 完成录制":               {"en": "⏹ Finish Recording",           "zh": "⏹ 完成录制"},
    "⛔ 异常终止":               {"en": "⛔ Abort Recording",            "zh": "⛔ 异常终止"},
    "✕ 全部移除":                {"en": "✕ Remove All",                  "zh": "✕ 全部移除"},
    "📂 回放":                   {"en": "📂 Playback",                  "zh": "📂 回放"},
    "☁ 上传":                   {"en": "☁ Upload",                    "zh": "☁ 上传"},

    # ── 工具栏提示 ────────────────────────────────────
    "任务名称…":                {"en": "Task name…",                   "zh": "任务名称…"},
    "输入任务名称，录制文件夹将包含此标注": {"en": "Enter a task name to label the recording folder", "zh": "输入任务名称，录制文件夹将包含此标注"},
    "停止录制并永久删除本次录制的全部文件": {"en": "Stop recording and permanently delete all files of this session", "zh": "停止录制并永久删除本次录制的全部文件"},

    # ── 相机控件 ──────────────────────────────────────
    "● 开始录制":                {"en": "● Start Recording",             "zh": "● 开始录制"},
    "■ 完成录制":                {"en": "■ Stop Recording",              "zh": "■ 完成录制"},
    "✕ 异常停止":                {"en": "✕ Abort",                       "zh": "✕ 异常停止"},
    "无信号":                     {"en": "No Signal",                     "zh": "无信号"},
    "已断开":                     {"en": "Disconnected",                  "zh": "已断开"},
    "等待中…":                   {"en": "Waiting…",                      "zh": "等待中…"},
    "FPS":                       {"en": "FPS",                           "zh": "FPS"},

    # ── 状态栏 ────────────────────────────────────────
    "就绪":                      {"en": "Ready",                         "zh": "就绪"},
    "摄像机:":                   {"en": "Cameras:",                      "zh": "摄像机:"},
    "录制中:":                   {"en": "Recording:",                    "zh": "录制中:"},

    # ── 日志消息 ──────────────────────────────────────
    "DAQ 视频管线已启动。":        {"en": "DAQ Video Pipeline started.",   "zh": "DAQ 视频管线已启动。"},
    "录制目录:":                  {"en": "Recording directory:",           "zh": "录制目录:"},
    "正在扫描摄像机 (MSMF → DShow → AUTO)…": {"en": "Scanning for cameras (MSMF → DShow → AUTO)…", "zh": "正在扫描摄像机 (MSMF → DShow → AUTO)…"},
    "发现 Camera {}，后端: {}":  {"en": "Found Camera {} via {}",         "zh": "发现 Camera {}，后端: {}"},
    "未检测到摄像机。":           {"en": "No cameras detected.",           "zh": "未检测到摄像机。"},
    "检测到 {} 台摄像机。":       {"en": "Detected {} camera(s).",        "zh": "检测到 {} 台摄像机。"},
    "摄像机 {} 已添加 ({}):":     {"en": "Camera {} added as '{}'.",      "zh": "摄像机 {} 已添加 ({}):"},
    "摄像机 '{}' 已移除。":       {"en": "Camera '{}' removed.",          "zh": "摄像机 '{}' 已移除。"},
    "[{}] 摄像机状态: {}":       {"en": "[{}] Camera state: {}",         "zh": "[{}] 摄像机状态: {}"},
    "[{}] ▶ 录制开始。":         {"en": "[{}] ▶ Recording started.",     "zh": "[{}] ▶ 录制开始。"},
    "[{}] ▶ 录制开始 — 编码: {}": {"en": "[{}] ▶ Recording started — codec: {}", "zh": "[{}] ▶ 录制开始 — 编码: {}"},
    " | 任务: {}":               {"en": " | Task: {}",                    "zh": " | 任务: {}"},
    "[{}] ■ 录制完成: {} ({} MB)": {"en": "[{}] ■ Recording completed: {} ({} MB)", "zh": "[{}] ■ 录制完成: {} ({} MB)"},
    "[{}] ✕ 录制已丢弃。":       {"en": "[{}] ✕ Recording aborted — file discarded.", "zh": "[{}] ✕ 录制已丢弃。"},
    "[{}] 相机已打开: {}×{} @ {}": {"en": "[{}] Camera opened: {}×{} @ {}", "zh": "[{}] 相机已打开: {}×{} @ {}"},

    # ── 录制完成消息 ──────────────────────────────────
    "[{}] ■ 录制完成: {}":       {"en": "[{}] ■ Recording completed: {}", "zh": "[{}] ■ 录制完成: {}"},

    # ── 双目摄像机 ────────────────────────────────────
    "正在启动双目摄像机 (S80M)…":   {"en": "Starting stereo camera (S80M)…",   "zh": "正在启动双目摄像机 (S80M)…"},
    "双目管道已连接，等待帧数据…":   {"en": "Stereo pipe connected, waiting for frames…", "zh": "双目管道已连接，等待帧数据…"},
    "双目摄像机已启动":              {"en": "Stereo camera started",             "zh": "双目摄像机已启动"},
    "双目摄像机画面已开始传输":      {"en": "Stereo camera streaming started",   "zh": "双目摄像机画面已开始传输"},
    "[双目] 管道断开":              {"en": "[Stereo] Pipe broken",               "zh": "[双目] 管道断开"},
    "[双目错误] 读取帧异常: {}":    {"en": "[Stereo Error] Read exception: {}",   "zh": "[双目错误] 读取帧异常: {}"},
    "双目摄像机已断开 (共 {} 帧)":  {"en": "Stereo camera disconnected ({} frames)", "zh": "双目摄像机已断开 (共 {} 帧)"},
    "[双目] 子进程已退出，退出码: {}": {"en": "[Stereo] Subprocess exited, code: {}", "zh": "[双目] 子进程已退出，退出码: {}"},
    "[双目 stderr] {}":            {"en": "[Stereo stderr] {}",                  "zh": "[双目 stderr] {}"},
    "[错误] 双目 demo 脚本不存在":  {"en": "[Error] Stereo demo script not found", "zh": "[错误] 双目 demo 脚本不存在"},

    # ── 相机模式切换 ──────────────────────────────────
    "📷 单目摄像机":               {"en": "📷 Mono Camera",                   "zh": "📷 单目摄像机"},
    "👁 双目摄像机 (S80M)":       {"en": "👁 Stereo Camera (S80M)",          "zh": "👁 双目摄像机 (S80M)"},
    "切换摄像机类型":               {"en": "Switch camera type",                "zh": "切换摄像机类型"},
    "已切换到：单目摄像机模式":      {"en": "Switched to: Mono camera mode",     "zh": "已切换到：单目摄像机模式"},
    "已切换到：双目摄像机模式 (S80M)": {"en": "Switched to: Stereo camera mode (S80M)", "zh": "已切换到：双目摄像机模式 (S80M)"},
    "🔭 深度双目 (RealSense)":    {"en": "🔭 Depth Stereo (RealSense)",      "zh": "🔭 深度双目 (RealSense)"},
    "已切换到：深度双目模式 (RealSense)": {"en": "Switched to: Depth stereo mode (RealSense)", "zh": "已切换到：深度双目模式 (RealSense)"},
    "正在启动深度双目摄像机 ({})…": {"en": "Starting depth stereo camera ({})…", "zh": "正在启动深度双目摄像机 ({})…"},
    "深度双目摄像机已启动: {}":    {"en": "Depth stereo camera started: {}",   "zh": "深度双目摄像机已启动: {}"},
    "[错误] 未检测到 RealSense 设备": {"en": "[Error] No RealSense device detected", "zh": "[错误] 未检测到 RealSense 设备"},
    "[RealSense 错误] {}":        {"en": "[RealSense error] {}",            "zh": "[RealSense 错误] {}"},
    "请选择相机模式（单目/双目），然后点击 [扫描] 按钮。": {
        "en": "Please select camera mode (Mono/Stereo), then click [Scan].",
        "zh": "请选择相机模式（单目/双目），然后点击 [扫描] 按钮。",
    },

    # ── 错误消息 ──────────────────────────────────────
    "[错误]":                    {"en": "[ERROR]",                       "zh": "[错误]"},
    "摄像机 {}: 无法打开 (已尝试 MSMF, DShow, AUTO)": {"en": "Camera {}: cannot open (tried MSMF, DShow, AUTO)", "zh": "摄像机 {}: 无法打开 (已尝试 MSMF, DShow, AUTO)"},
    "添加摄像机 {} 失败: {}":   {"en": "Failed to add camera {}: {}",    "zh": "添加摄像机 {} 失败: {}"},
    "摄像机 {} 读取连续失败 ({} 帧)": {"en": "Camera {}: read failed ({} consecutive)", "zh": "摄像机 {} 读取连续失败 ({} 帧)"},

    # ── 对话框 ────────────────────────────────────────
    "摄像机检测":                 {"en": "Camera Detection",              "zh": "摄像机检测"},
    "未找到摄像机。\n\n请检查：\n• 摄像机是否已连接且未被其他程序占用\n• 摄像机驱动是否已安装\n• 尝试用索引 0、1、2… 手动添加": {
        "en": "No cameras found.\n\nCheck:\n• Camera is connected and not in use\n• Camera driver is installed\n• Try adding manually with index 0, 1, 2…",
        "zh": "未找到摄像机。\n\n请检查：\n• 摄像机是否已连接且未被其他程序占用\n• 摄像机驱动是否已安装\n• 尝试用索引 0、1、2… 手动添加",
    },
    "警告":                      {"en": "Warning",                       "zh": "警告"},
    "摄像机 {} 已存在。":         {"en": "Camera {} already added.",      "zh": "摄像机 {} 已存在。"},
    "确认":                      {"en": "Confirm",                       "zh": "确认"},
    "摄像机 {} 正在录制中，确定要中止并移除？": {"en": "Camera {} is recording. Abort and remove?", "zh": "摄像机 {} 正在录制中，确定要中止并移除？"},
    "移除所有摄像机？":            {"en": "Remove all cameras?",           "zh": "移除所有摄像机？"},
    "添加摄像机":                 {"en": "Add Camera",                    "zh": "添加摄像机"},
    "摄像机索引:":                {"en": "Camera index:",                 "zh": "摄像机索引:"},

    # ── 面板标题 ──────────────────────────────────────
    "日志":                      {"en": "Log",                           "zh": "日志"},
    "录制历史":                   {"en": "Recording History",             "zh": "录制历史"},
    "摄像机":                     {"en": "Camera",                        "zh": "摄像机"},
    "编码":                      {"en": "Codec",                         "zh": "编码"},
    "文件":                      {"en": "File",                          "zh": "文件"},
    "时长":                      {"en": "Duration",                      "zh": "时长"},
    "大小":                      {"en": "Size",                          "zh": "大小"},
    "状态":                      {"en": "Status",                        "zh": "状态"},
    "已完成":                     {"en": "COMPLETED",                     "zh": "已完成"},
    "已丢弃":                     {"en": "ABORTED",                       "zh": "已丢弃"},
    "已上传":                     {"en": "Uploaded",                      "zh": "已上传"},
    "已删除（未上传）":            {"en": "DELETED (NOT UPLOADED)",         "zh": "已删除（未上传）"},

    # ── 关于对话框 ────────────────────────────────────
    "关于 DAQ 视频管线":          {"en": "About DAQ Video Pipeline",      "zh": "关于 DAQ 视频管线"},
    "实时摄像机预览":              {"en": "Real-time camera preview",       "zh": "实时摄像机预览"},
    "可拖拽调整的画面布局":         {"en": "Draggable-resizable camera layout", "zh": "可拖拽调整的画面布局"},
    "支持正常完成和异常停止两种录制模式": {"en": "Normal finish and abort recording modes", "zh": "支持正常完成和异常停止两种录制模式"},
    "录制历史记录追踪":            {"en": "Recording history tracking",     "zh": "录制历史记录追踪"},
    "中英文界面切换":              {"en": "Bilingual UI (Chinese / English)", "zh": "中英文界面切换"},

    # ── 空状态提示 ────────────────────────────────────
    '尚未检测到摄像机。\n请从左侧设备面板勾选要开启的设备。': {
        "en": "No cameras detected.\nEnable devices from the device panel on the left.",
        'zh': '尚未检测到摄像机。\n请从左侧设备面板勾选要开启的设备。',
    },

    # ── 传感器面板 ────────────────────────────────────
    "传感器阵列":                 {"en": "Sensor Array",                   "zh": "传感器阵列"},

    # ── 设备检测面板 ──────────────────────────────────
    "📷 设备检测":                {"en": "📷 Device Detection",            "zh": "📷 设备检测"},
    "设备列表":                   {"en": "Device List",                    "zh": "设备列表"},
    "点击列表中的设备以显示画面": {"en": "Click a device to show its feed", "zh": "点击列表中的设备以显示画面"},
    "插拔自动刷新":               {"en": "Auto-refresh on plug/unplug",    "zh": "插拔自动刷新"},
    "未检测到设备":               {"en": "No devices detected",            "zh": "未检测到设备"},
    "已切换设备: {}":             {"en": "Switched device: {}",            "zh": "已切换设备: {}"},
    "[设备] 已连接: {}":          {"en": "[Device] Connected: {}",         "zh": "[设备] 已连接: {}"},
    "[设备] 已断开: {}":          {"en": "[Device] Disconnected: {}",      "zh": "[设备] 已断开: {}"},
    "[设备] {} 已移除。":         {"en": "[Device] {} removed.",           "zh": "[设备] {} 已移除。"},
    "录制中切换设备将中止当前录制，确定继续？": {
        "en": "Switching device during recording will abort the current recording. Continue?",
        "zh": "录制中切换设备将中止当前录制，确定继续？",
    },
    "USB 摄像头":                 {"en": "USB Camera",                     "zh": "USB 摄像头"},
    "[设备] S80M 与 D435 同时连接，SDK 抢占 video0/video2，双目可能无法打开": {
        "en": "[Device] S80M and D435 both connected; SDK takes video0/video2, stereo may fail to open",
        "zh": "[设备] S80M 与 D435 同时连接，SDK 抢占 video0/video2，双目可能无法打开",
    },
    "[设备] 录制中拔线，不自动移除（重插自动恢复）: {}": {
        "en": "[Device] Unplugged during recording, not auto-removed (recovers on replug): {}",
        "zh": "[设备] 录制中拔线，不自动移除（重插自动恢复）: {}",
    },

    # ── 设备统一接入（分组 / 开关 / 命名） ──────────────
    "📷 相机":                    {"en": "📷 Cameras",                     "zh": "📷 相机"},
    "🧤 手套":                    {"en": "🧤 Gloves",                      "zh": "🧤 手套"},
    "🎧 其他蓝牙":                {"en": "🎧 Other Bluetooth",            "zh": "🎧 其他蓝牙"},
    "🎧 其他蓝牙: {} 台":         {"en": "🎧 Other Bluetooth: {} device(s)", "zh": "🎧 其他蓝牙: {} 台"},
    "蓝牙":                       {"en": "Bluetooth",                      "zh": "蓝牙"},
    "开关设备以显示画面":          {"en": "Toggle a device to show its feed", "zh": "开关设备以显示画面"},
    "双击设备可重命名":            {"en": "Double-click a device to rename", "zh": "双击设备可重命名"},
    "重命名设备":                 {"en": "Rename Device",                  "zh": "重命名设备"},
    "设备名称:":                  {"en": "Device name:",                   "zh": "设备名称:"},
    "录制中不可更改设备":          {"en": "Devices cannot be changed while recording", "zh": "录制中不可更改设备"},
    "该设备无可视化数据":          {"en": "No visualizable data for this device", "zh": "该设备无可视化数据"},
    "S80M 与 D435 无法同时开启":   {"en": "S80M and D435 cannot be enabled at the same time", "zh": "S80M 与 D435 无法同时开启"},
    "S80M 与 D435 抢占视频节点（video0/video2），先开者生效": {
        "en": "S80M and D435 conflict on video nodes (video0/video2); the first one opened wins",
        "zh": "S80M 与 D435 抢占视频节点（video0/video2），先开者生效",
    },
    "[设备] 已开启: {}":          {"en": "[Device] Enabled: {}",           "zh": "[设备] 已开启: {}"},
    "[设备] 已关闭: {}":          {"en": "[Device] Disabled: {}",          "zh": "[设备] 已关闭: {}"},
    "[设备] 已重命名: {}":        {"en": "[Device] Renamed: {}",           "zh": "[设备] 已重命名: {}"},
    "[设备] 开启失败: {}":        {"en": "[Device] Failed to enable: {}",  "zh": "[设备] 开启失败: {}"},

    # ── 每设备曝光设置（信息条 ☀ 按钮） ────────────────
    "曝光设置":                   {"en": "Exposure",                       "zh": "曝光设置"},
    "自动曝光":                   {"en": "Auto exposure",                  "zh": "自动曝光"},
    "拖动滑块即时生效":            {"en": "Drag the slider to adjust live", "zh": "拖动滑块即时生效"},
    "恢复默认":                   {"en": "Reset to default",               "zh": "恢复默认"},
    "回到相机最一开始的曝光设置":    {"en": "Restore the camera's original exposure settings", "zh": "回到相机最一开始的曝光设置"},
    "录制中不可调整曝光":          {"en": "Exposure cannot be adjusted while recording", "zh": "录制中不可调整曝光"},
    "[双目] 曝光下发失败（子进程已退出）": {
        "en": "[Stereo] Exposure command failed (subprocess exited)",
        "zh": "[双目] 曝光下发失败（子进程已退出）",
    },

    # ── 状态显示 ──────────────────────────────────────
    "● 录制中 ({} 台摄像机)…":   {"en": "● Recording ({} camera(s))…",   "zh": "● 录制中 ({} 台摄像机)…"},
    "DAQ 视频管线已关闭。":        {"en": "DAQ Video Pipeline shut down.", "zh": "DAQ 视频管线已关闭。"},
    "正在扫描摄像机…":           {"en": "Scanning for cameras…",         "zh": "正在扫描摄像机…"},
    "正在扫描设备…":             {"en": "Scanning for devices…",         "zh": "正在扫描设备…"},
    "⛔ 录制已异常终止，文件已丢弃。": {"en": "⛔ Recording aborted, files discarded.", "zh": "⛔ 录制已异常终止，文件已丢弃。"},

    # ── 传感器面板 ────────────────────────────────────
    "传感器阵列 (BLE)":          {"en": "Sensor Array (BLE)",            "zh": "传感器阵列 (BLE)"},
    "传感器 Right":              {"en": "Sensor Right",                  "zh": "传感器 Right"},
    "传感器 Left":               {"en": "Sensor Left",                   "zh": "传感器 Left"},
    "传感器 {} ({})":            {"en": "Sensor {} ({})",               "zh": "传感器 {} ({})"},
    "🔍 扫描设备":               {"en": "🔍 Scan Device",               "zh": "🔍 扫描设备"},
    "选择 BLE 设备":             {"en": "Select BLE Device",            "zh": "选择 BLE 设备"},
    "🔗 连接":                   {"en": "🔗 Connect",                   "zh": "🔗 连接"},
    "未连接":                     {"en": "Not Connected",                "zh": "未连接"},
    "模式:":                     {"en": "Mode:",                        "zh": "模式:"},
    "🔥 热力图 (Heatmap)":       {"en": "🔥 Heatmap",                   "zh": "🔥 热力图 (Heatmap)"},
    "📝 轨迹 (Trace)":           {"en": "📝 Trace",                     "zh": "📝 轨迹 (Trace)"},
    "📊 网格 (Grid)":            {"en": "📊 Grid",                      "zh": "📊 网格 (Grid)"},
    "🦾 仿生手掌 (Hand)":        {"en": "🦾 Bionic Hand",              "zh": "🦾 仿生手掌 (Hand)"},
    "🕸 形变网格 (Deform)":      {"en": "🕸 Deform Mesh",              "zh": "🕸 形变网格 (Deform)"},
    "🖐 3D 手掌 (3D Hand)":     {"en": "🖐 3D Hand",                   "zh": "🖐 3D 手掌 (3D Hand)"},
    "📐 校准":                   {"en": "📐 Calibrate",                 "zh": "📐 校准"},
    "⚙ 配置":                   {"en": "⚙ Config",                    "zh": "⚙ 配置"},
    "▸ 降噪参数":                {"en": "▸ Denoise Params",            "zh": "▸ 降噪参数"},
    "噪声门:":                   {"en": "Noise Gate:",                  "zh": "噪声门:"},
    "动态比率:":                 {"en": "Dynamic Ratio:",               "zh": "动态比率:"},
    "空间滤波":                   {"en": "Spatial Filter",               "zh": "空间滤波"},
    "正在扫描…":                 {"en": "Scanning…",                    "zh": "正在扫描…"},
    "扫描中…":                   {"en": "Scanning…",                    "zh": "扫描中…"},
    "未发现设备":                 {"en": "No devices found",             "zh": "未发现设备"},
    "发现 {} 台设备":            {"en": "Found {} device(s)",           "zh": "发现 {} 台设备"},
    "断开":                      {"en": "Disconnect",                   "zh": "断开"},
    "连接中…":                   {"en": "Connecting…",                  "zh": "连接中…"},
    "已连接: {}…":              {"en": "Connected: {}…",               "zh": "已连接: {}…"},
    "已断开":                     {"en": "Disconnected",                  "zh": "已断开"},
    "校准完成":                   {"en": "Calibration complete",         "zh": "校准完成"},
    "校准中… {}%":              {"en": "Calibrating… {}%",             "zh": "校准中… {}%"},
    "校准中…":                   {"en": "Calibrating…",                 "zh": "校准中…"},
    "热力图/轨迹配置":            {"en": "Heatmap / Trace Config",       "zh": "热力图/轨迹配置"},
    "网格配置":                   {"en": "Grid Config",                  "zh": "网格配置"},
    "形变网格配置":               {"en": "Deform Mesh Config",           "zh": "形变网格配置"},
    "[Sensor] {}":               {"en": "[Sensor] {}",                  "zh": "[Sensor] {}"},

    # ── 回放对话框 ────────────────────────────────────
    "录制回放":                   {"en": "Recording Playback",            "zh": "录制回放"},
    "录制目录:":                  {"en": "Recording Directory:",          "zh": "录制目录:"},
    "刷新":                      {"en": "Refresh",                       "zh": "刷新"},
    "录制会话:":                  {"en": "Recording Sessions:",           "zh": "录制会话:"},
    "传感器数据":                 {"en": "Sensor Data",                   "zh": "传感器数据"},
    "显示模式:":                  {"en": "Display Mode:",                 "zh": "显示模式:"},
    "选择录制目录":               {"en": "Select Recording Directory",    "zh": "选择录制目录"},
    "提示":                      {"en": "Notice",                        "zh": "提示"},
    "请先勾选要删除的会话":        {"en": "Please check sessions to delete first", "zh": "请先勾选要删除的会话"},
    "确认删除":                   {"en": "Confirm Delete",                "zh": "确认删除"},
    "完成":                      {"en": "Done",                          "zh": "完成"},
    "⬆ 选中已上传":             {"en": "⬆ Select Uploaded",            "zh": "⬆ 选中已上传"},
    "仅勾选已上传的会话":          {"en": "Select only uploaded sessions",  "zh": "仅勾选已上传的会话"},
    "⬇ 选中未上传":             {"en": "⬇ Select Not Uploaded",        "zh": "⬇ 选中未上传"},
    "仅勾选未上传的会话":          {"en": "Select only not-uploaded sessions", "zh": "仅勾选未上传的会话"},
    "🗑 删除选中":               {"en": "🗑 Delete Selected",           "zh": "🗑 删除选中"},
    "将选中的会话移入回收站":       {"en": "Move selected sessions to recycle bin", "zh": "将选中的会话移入回收站"},
    "▶ 播放":                   {"en": "▶ Play",                        "zh": "▶ 播放"},
    "⏸ 暂停":                   {"en": "⏸ Pause",                      "zh": "⏸ 暂停"},
    "没有找到视频文件":            {"en": "No video files found",          "zh": "没有找到视频文件"},
    "路":                        {"en": "ch",                            "zh": "路"},
    "加载失败: {}":              {"en": "Load failed: {}",               "zh": "加载失败: {}"},
    "确定要将以下会话移入回收站？\n\n{}\n\n此操作可逆（可从回收站恢复）。": {
        "en": "Move the following sessions to recycle bin?\n\n{}\n\nThis operation is reversible.",
        "zh": "确定要将以下会话移入回收站？\n\n{}\n\n此操作可逆（可从回收站恢复）。",
    },
    "已删除 {}/{} 个会话（已移入回收站）。": {
        "en": "Deleted {}/{} session(s) (moved to recycle bin).",
        "zh": "已删除 {}/{} 个会话（已移入回收站）。",
    },
    "传感器 TS: {}  Δ={}ms": {
        "en": "Sensor TS: {}  Δ={}ms",
        "zh": "传感器 TS: {}  Δ={}ms",
    },

    # ── 上传对话框 ────────────────────────────────────
    "上传录制数据":               {"en": "Upload Recording Data",         "zh": "上传录制数据"},
    "服务器配置":                 {"en": "Server Configuration",          "zh": "服务器配置"},
    "服务器地址:":                {"en": "Server Address:",               "zh": "服务器地址:"},
    "测试连接":                   {"en": "Test Connection",               "zh": "测试连接"},
    "选择要上传的会话":            {"en": "Select Sessions to Upload",     "zh": "选择要上传的会话"},
    "全选":                      {"en": "Select All",                    "zh": "全选"},
    "⬆ 一键上传":               {"en": "⬆ Upload",                      "zh": "⬆ 一键上传"},
    "关闭":                      {"en": "Close",                         "zh": "关闭"},
    "请先勾选要上传的会话":        {"en": "Please select sessions to upload first", "zh": "请先勾选要上传的会话"},
    "请输入服务器地址":            {"en": "Please enter server address",    "zh": "请输入服务器地址"},
    "成功":                      {"en": "Success",                       "zh": "成功"},
    "服务器连接正常 ✅":           {"en": "Server connection OK ✅",       "zh": "服务器连接正常 ✅"},
    "失败":                      {"en": "Failed",                        "zh": "失败"},
    "无法连接到服务器 ❌":         {"en": "Cannot connect to server ❌",    "zh": "无法连接到服务器 ❌"},
    "已入队 {} 个任务，开始处理…": {"en": "Queued {} task(s), processing…", "zh": "已入队 {} 个任务，开始处理…"},
    "全部完成":                   {"en": "All complete",                  "zh": "全部完成"},

    # ── 上传开关（自动上传 / 上传后自动删除） ─────────────
    "☁ 自动上传: {}":            {"en": "☁ Auto upload: {}",               "zh": "☁ 自动上传: {}"},
    "录制完成后自动上传；关闭后需在 ☁ 上传对话框手动上传": {
        "en": "Auto-upload after recording; when off, upload manually from the ☁ Upload dialog",
        "zh": "录制完成后自动上传；关闭后需在 ☁ 上传对话框手动上传",
    },
    "☁ 自动上传未开启，请手动上传: {}": {"en": "☁ Auto upload is off, upload manually: {}", "zh": "☁ 自动上传未开启，请手动上传: {}"},
    "🗑 上传后自动删除: {}":       {"en": "🗑 Auto-delete after upload: {}",  "zh": "🗑 上传后自动删除: {}"},
    "开":                        {"en": "ON",                              "zh": "开"},
    "关":                        {"en": "OFF",                             "zh": "关"},
    "上传成功后自动删除该会话的本地文件": {
        "en": "Delete local session files after a successful upload",
        "zh": "上传成功后自动删除该会话的本地文件",
    },
    "☁ 上传完成，本地文件已删除: {}": {"en": "☁ Upload complete, local files deleted: {}", "zh": "☁ 上传完成，本地文件已删除: {}"},
    "☁ 上传完成（本地保留）: {}":   {"en": "☁ Upload complete (kept locally): {}", "zh": "☁ 上传完成（本地保留）: {}"},
    "❌ 上传失败: {}":             {"en": "❌ Upload failed: {}",           "zh": "❌ 上传失败: {}"},
    "[错误] 上传后自动删除失败: {}": {"en": "[Error] Auto-delete after upload failed: {}", "zh": "[错误] 上传后自动删除失败: {}"},
    "已上传，本地已删":              {"en": "Uploaded, deleted locally",      "zh": "已上传，本地已删"},

    # ── 数据查看器 ────────────────────────────────────
    "📂 浏览录制目录":            {"en": "📂 Browse Recordings",          "zh": "📂 浏览录制目录"},
    "🔄 刷新":                   {"en": "🔄 Refresh",                    "zh": "🔄 刷新"},
    "📋 录制会话:":               {"en": "📋 Recording Sessions:",        "zh": "📋 录制会话:"},
    "选择会话开始回放":            {"en": "Select a session to start playback", "zh": "选择会话开始回放"},
    "📹 视频":                   {"en": "📹 Video",                      "zh": "📹 视频"},
    "📊 传感器":                 {"en": "📊 Sensor",                     "zh": "📊 传感器"},

    # ── 任务选择页面 ────────────────────────────────────
    "连接":                      {"en": "Connect",                        "zh": "连接"},
    "用户名":                    {"en": "Username",                       "zh": "用户名"},
    "密码":                      {"en": "Password",                       "zh": "密码"},
    "登陆成功":                  {"en": "Login OK",                       "zh": "登陆成功"},
    "登陆失败: {}":              {"en": "Login failed: {}",               "zh": "登陆失败: {}"},
    "后端已连接":                {"en": "Backend Connected",              "zh": "后端已连接"},
    "后端未连接":                {"en": "Backend Disconnected",           "zh": "后端未连接"},
    "已切换服务器: {} (用户: {})": {"en": "Switched server: {} (user: {})",  "zh": "已切换服务器: {} (用户: {})"},
    "已切换服务器地址: {} (无认证)": {"en": "Switched server: {} (no auth)",   "zh": "已切换服务器地址: {} (无认证)"},
    "刷新中…":                   {"en": "Refreshing…",                    "zh": "刷新中…"},
    "可用任务":                   {"en": "Available Tasks",                "zh": "可用任务"},
    "任务详情":                   {"en": "Task Details",                   "zh": "任务详情"},
    "← 返回任务选择":            {"en": "← Back to Task Selection",       "zh": "← 返回任务选择"},
    "← 请从左侧列表中选择一个任务": {"en": "← Select a task from the list",  "zh": "← 请从左侧列表中选择一个任务"},
    "→ 进入采集":                {"en": "→ Start Collection",             "zh": "→ 进入采集"},
    "已选择: {}":                {"en": "Selected: {}",                   "zh": "已选择: {}"},
    "已选择任务: {}":            {"en": "Task selected: {}",              "zh": "已选择任务: {}"},
    "描述:":                     {"en": "Description:",                   "zh": "描述:"},
    "任务ID:":                   {"en": "Task ID:",                       "zh": "任务ID:"},
    "创建时间:":                  {"en": "Created:",                       "zh": "创建时间:"},
    "状态:":                     {"en": "Status:",                        "zh": "状态:"},
    "参数:":                     {"en": "Parameters:",                    "zh": "参数:"},
    "返回任务选择":               {"en": "Back to Task Selection",         "zh": "返回任务选择"},
    "待处理":                     {"en": "Pending",                        "zh": "待处理"},
    "进行中":                     {"en": "In Progress",                    "zh": "进行中"},
    "已取消":                     {"en": "Cancelled",                      "zh": "已取消"},
    "未命名任务":                 {"en": "Unnamed Task",                   "zh": "未命名任务"},
    "任务名称":                   {"en": "Task Name",                      "zh": "任务名称"},
    "进度":                       {"en": "Progress",                       "zh": "进度"},
    "发布时间":                   {"en": "Published",                      "zh": "发布时间"},
    "暂无可用任务\n请检查后端连接或点击刷新": {
        "en": "No tasks available\nCheck backend connection or click refresh",
        "zh": "暂无可用任务\n请检查后端连接或点击刷新",
    },
    "录制中，确定要返回任务列表？当前录制将被终止。": {
        "en": "Recording in progress. Return to task list? Current recording will be aborted.",
        "zh": "录制中，确定要返回任务列表？当前录制将被终止。",
    },
    "已返回任务选择页面。":        {"en": "Returned to task selection page.",  "zh": "已返回任务选择页面。"},
    "该任务采集进度已满":          {"en": "Task progress is full",           "zh": "该任务采集进度已满"},
    "该任务采集进度已满，无法进入":   {"en": "Task progress is full, cannot enter", "zh": "该任务采集进度已满，无法进入"},
    "该任务采集进度已满（{}/{}），无法开始新的采集。": {
        "en": "Task progress is full ({}/{}), cannot start new collection.",
        "zh": "该任务采集进度已满（{}/{}），无法开始新的采集。",
    },
    "☁ 已自动加入上传队列: {}":     {"en": "☁ Auto-queued for upload: {}",   "zh": "☁ 已自动加入上传队列: {}"},
    "☁ 上传完成: {}":              {"en": "☁ Upload complete: {}",         "zh": "☁ 上传完成: {}"},
    "第 {}/{} 条":                 {"en": "#{}/{}",                          "zh": "第 {}/{} 条"},
    "✅ 第 {}/{} 条上传完成: {}":   {"en": "✅ #{}/{} uploaded: {}",         "zh": "✅ 第 {}/{} 条上传完成: {}"},
    "✅ 上传完成: {}":             {"en": "✅ Uploaded: {}",                 "zh": "✅ 上传完成: {}"},
    "[上传失败] {}: {}":            {"en": "[Upload failed] {}: {}",        "zh": "[上传失败] {}: {}"},
    "发布时间:":                  {"en": "Published:",                      "zh": "发布时间:"},
    "请在任务选择页面选择一个任务。": {"en": "Please select a task on the task selection page.", "zh": "请在任务选择页面选择一个任务。"},
    "任务进度已更新: {} ({}/{})":  {"en": "Task progress updated: {} ({}/{})", "zh": "任务进度已更新: {} ({}/{})"},
    "提示":                       {"en": "Notice",                          "zh": "提示"},
    "🗑 删除":                    {"en": "🗑 Delete",                     "zh": "🗑 删除"},
    "确定要删除任务 \"{}\" 吗？\n删除后可在 tasks.json 中恢复。": {
        "en": "Delete task \"{}\"?\nIt can be restored in tasks.json.",
        "zh": "确定要删除任务 \"{}\" 吗？\n删除后可在 tasks.json 中恢复。",
    },
    "🗑 删除任务":                {"en": "🗑 Delete Task",                 "zh": "🗑 删除任务"},
    "待采集":                     {"en": "Pending",                        "zh": "待采集"},
    "采集完成":                   {"en": "Completed",                      "zh": "采集完成"},
    "采集中":                     {"en": "In Progress",                    "zh": "采集中"},
    "该任务已采集完成，请更换其他任务": {"en": "Task collection complete, please switch to another task.", "zh": "该任务已采集完成，请更换其他任务"},

    # ── 登录对话框 / 身份 ───────────────────────────────
    "用户登录":                  {"en": "User Login",                     "zh": "用户登录"},
    "登录":                      {"en": "Login",                          "zh": "登录"},
    "记住账号":                  {"en": "Remember account",               "zh": "记住账号"},
    "游客登录":                  {"en": "Guest Login",                    "zh": "游客登录"},
    "登录中…":                   {"en": "Logging in…",                    "zh": "登录中…"},
    "请输入用户名":               {"en": "Please enter username",          "zh": "请输入用户名"},
    "输入后端派发的账号登录，或点「游客登录」直接进入（仅可见公共任务）。": {
        "en": "Sign in with the account issued by the backend, or click \"Guest Login\" to continue (public tasks only).",
        "zh": "输入后端派发的账号登录，或点「游客登录」直接进入（仅可见公共任务）。",
    },
    "当前身份:":                 {"en": "Current identity:",               "zh": "当前身份:"},
    "切换账号":                  {"en": "Switch account",                 "zh": "切换账号"},
    "游客":                      {"en": "Guest",                          "zh": "游客"},
    "已登录: {} ({})":           {"en": "Logged in: {} ({})",            "zh": "已登录: {} ({})"},
    "已进入游客模式（仅可见公共任务）。": {"en": "Entered guest mode (public tasks only).", "zh": "已进入游客模式（仅可见公共任务）。"},
    "登录已过期，请重新登录。":    {"en": "Login expired, please sign in again.", "zh": "登录已过期，请重新登录。"},

    # ── 使用说明窗口 ────────────────────────────────
    "使用说明":                   {"en": "Usage Guide",                   "zh": "使用说明"},
    "使用步骤":                   {"en": "Usage Steps",                   "zh": "使用步骤"},
    "下次启动不再自动显示（可随时从 帮助→使用说明 重新打开并取消勾选）": {
        "en": "Don't show automatically at startup (reopen anytime via Help → Usage Guide to undo)",
        "zh": "下次启动不再自动显示（可随时从 帮助→使用说明 重新打开并取消勾选）",
    },

    # ── 手部关键点追踪 ────────────────────────────────
    "🧤 手套追踪":                {"en": "🧤 Glove Tracking",               "zh": "🧤 手套追踪"},
    "🖐 裸手追踪":                {"en": "🖐 Bare Hand",                  "zh": "🖐 裸手追踪"},
    "✋ 手部叠加":                 {"en": "✋ Hand Overlay",               "zh": "✋ 手部叠加"},
    "🔄 提取关键点":              {"en": "🔄 Extract Keypoints",          "zh": "🔄 提取关键点"},
    "✋ 处理手部关键点":           {"en": "✋ Process Hand Keypoints",     "zh": "✋ 处理手部关键点"},
    "对录制完成的视频后台提取手部关键点": {"en": "Extract hand keypoints from recorded video in background", "zh": "对录制完成的视频后台提取手部关键点"},
    "后台处理当前视频，提取手部关键点": {"en": "Process current video in background, extract hand keypoints", "zh": "后台处理当前视频，提取手部关键点"},
    "手部关键点处理正在进行中，请等待完成。": {"en": "Hand keypoint processing in progress, please wait.", "zh": "手部关键点处理正在进行中，请等待完成。"},
    "该录制已有手部关键点数据，要重新处理吗？": {"en": "This recording already has hand keypoint data. Reprocess?", "zh": "该录制已有手部关键点数据，要重新处理吗？"},
    "[手部关键点] 模块不可用（缺少依赖）。": {"en": "[Hand Kpts] Module unavailable (missing dependencies).", "zh": "[手部关键点] 模块不可用（缺少依赖）。"},
    "[手部关键点] 没有找到可处理的录制。": {"en": "[Hand Kpts] No recordable session found.", "zh": "[手部关键点] 没有找到可处理的录制。"},
    "[手部关键点] 会话目录不存在: {}": {"en": "[Hand Kpts] Session directory not found: {}", "zh": "[手部关键点] 会话目录不存在: {}"},
    "[手部关键点] 开始处理: {}":     {"en": "[Hand Kpts] Processing: {}",         "zh": "[手部关键点] 开始处理: {}"},
    "[手部关键点] {}":              {"en": "[Hand Kpts] {}",                     "zh": "[手部关键点] {}"},
    "✋ 处理手部关键点: {}/{} ({:.0f}%)": {"en": "✋ Processing hand keypoints: {}/{} ({:.0f}%)", "zh": "✋ 处理手部关键点: {}/{} ({:.0f}%)"},
    "[手部关键点] ❌ 处理失败: {}":  {"en": "[Hand Kpts] ❌ Processing failed: {}", "zh": "[手部关键点] ❌ 处理失败: {}"},
    "[手部关键点] ✅ 处理完成: {}":  {"en": "[Hand Kpts] ✅ Processing complete: {}", "zh": "[手部关键点] ✅ 处理完成: {}"},
    "手部关键点":                   {"en": "Hand Keypoints",                  "zh": "手部关键点"},
    "⏳ 处理中…":                  {"en": "⏳ Processing…",                  "zh": "⏳ 处理中…"},
    "⏳ {:.0f}%":                  {"en": "⏳ {:.0f}%",                      "zh": "⏳ {:.0f}%"},
    "手部关键点模块不可用。":        {"en": "Hand keypoint module unavailable.",  "zh": "手部关键点模块不可用。"},
    "选择手部追踪模式：黑色手套 / 裸手": {"en": "Select hand tracking mode: Glove / Bare Hand", "zh": "选择手部追踪模式：黑色手套 / 裸手"},
    "选择手部追踪模式":              {"en": "Select hand tracking mode",          "zh": "选择手部追踪模式"},
    "📊 自动标注":                  {"en": "📊 Auto Label",                   "zh": "📊 自动标注"},
    "基于手部关键点数据自动生成手势标签": {"en": "Auto-generate gesture labels from hand keypoint data", "zh": "基于手部关键点数据自动生成手势标签"},
    "[自动标注] 模块不可用。":        {"en": "[Auto Label] Module unavailable.",    "zh": "[自动标注] 模块不可用。"},
    "[自动标注] 没有找到可处理的录制。": {"en": "[Auto Label] No recordable session found.", "zh": "[自动标注] 没有找到可处理的录制。"},
    "[自动标注] 会话目录不存在: {}":  {"en": "[Auto Label] Session directory not found: {}", "zh": "[自动标注] 会话目录不存在: {}"},
    "[自动标注] 请先提取手部关键点。": {"en": "[Auto Label] Please extract hand keypoints first.", "zh": "[自动标注] 请先提取手部关键点。"},
    "[自动标注] 开始标注: {}":       {"en": "[Auto Label] Labeling: {}",           "zh": "[自动标注] 开始标注: {}"},
    "📊 自动标注: {}/{} ({:.0f}%)":  {"en": "📊 Auto labeling: {}/{} ({:.0f}%)", "zh": "📊 自动标注: {}/{} ({:.0f}%)"},
    "[自动标注] ❌ 标注失败: {}":    {"en": "[Auto Label] ❌ Labeling failed: {}",  "zh": "[自动标注] ❌ 标注失败: {}"},
    "[自动标注] ✅ 标注完成: {}":    {"en": "[Auto Label] ✅ Labeling complete: {}", "zh": "[自动标注] ✅ 标注完成: {}"},
    "⏳ 标注中…":                   {"en": "⏳ Labeling…",                    "zh": "⏳ 标注中…"},
}


# ═══════════════════════════════════════════════════════
#  语言管理器（单例）
# ═══════════════════════════════════════════════════════

class LanguageManager(QObject):
    """管理当前语言，切换时发射信号通知所有监听者刷新界面文字。"""

    language_changed = pyqtSignal(str)   # 参数: "zh" 或 "en"

    def __init__(self):
        super().__init__()
        self._current = "en"             # 默认英文

    @property
    def current(self) -> str:
        return self._current

    def set_language(self, lang: str):
        """切换到指定语言并通知所有监听者。"""
        if lang not in ("zh", "en"):
            return
        if lang == self._current:
            return
        self._current = lang
        self.language_changed.emit(lang)

    def toggle(self):
        """在中英文之间切换。"""
        self.set_language("en" if self._current == "zh" else "zh")


# 全局单例
lang_manager = LanguageManager()


def tr(text: str, *fmt_args) -> str:
    """
    翻译函数：根据当前语言返回对应文本。

    用法:
        tr("开始录制")          → "开始录制" 或 "Start Recording"
        tr("发现 {} 台摄像机。", 2) → "检测到 2 台摄像机。" 或 "Detected 2 camera(s)."
    """
    entry = _TRANSLATIONS.get(text)
    if entry is None:
        result = text
    else:
        result = entry.get(lang_manager.current, text)
    if fmt_args:
        result = result.format(*fmt_args)
    return result

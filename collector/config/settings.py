"""
应用程序全局配置 —— 路径、颜色、相机参数等。
"""

import os
import sys

from config import __version__ as APP_VERSION   # 版本号唯一定义在 config/__init__.py

# ── 路径 ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
DB_PATH = os.path.join(DATA_DIR, "pipeline.db")
KEYPOINTS_OUTPUT_DIR = os.path.join(BASE_DIR, "keypoints_output")

# ── 应用程序 ──────────────────────────────────────────
APP_NAME = "EGO数据管线"
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900

# ── 相机默认参数 ──────────────────────────────────────
DEFAULT_FPS = 30
if sys.platform == "win32":
    # Windows：DECXIN 驱动对 1280×960「直连」只暴露 5fps 档（UVC 帧率表
    # 怪癖，设 MJPG 也一样）。请求 1280×720 高会命中 30fps 档，实测驱动
    # 协商回 1280×960@30（读回与真实帧尺寸仍为 960 高），画面分辨率不变。
    DEFAULT_RESOLUTION = (1280, 720)
else:
    DEFAULT_RESOLUTION = (1280, 960)   # Linux：UVC 满分辨率（DECXIN）；须配合 UVC_FOURCC=MJPG 才能 30fps
UVC_FOURCC = "MJPG"                 # UVC 采集像素格式：YUYV 在 1280×960 硬限 5fps
                                    # 且驱动默认格式就是 YUYV，不显式设 MJPG 会静默掉帧率
CAMERA_RECONNECT_INTERVAL_MS = 2000   # 断线后重试间隔（毫秒）
MAX_CAMERAS = 8

# ── 录制参数 ──────────────────────────────────────────
RECORDING_FPS = 30                    # 录制帧率
RECORDING_DIR = RECORDINGS_DIR        # 录制文件输出目录
DEPTH_ENABLED = False                 # 遗留开关：旧 S80M 视差录制路径（已被 S80M_DEPTH_ENABLED
                                      # 取代；仅剩 egodata_writer 守卫/测试/文档引用，勿再启用）

# ── 录制编码（v1.0.9：录制直出 HEVC，上传免预压）─────
RECORD_VIDEO_ENCODER = "auto"      # "auto" | "nvenc" | "x265" | "x264"
                                   # auto: hevc_nvenc(可用) → libx265(速度达标) → libx264
                                   # 显式指定: 跳过速度门槛强制用该编码器；所有候选
                                   # 二进制都不支持时按 auto 链逐级回退（警告日志）
RECORD_VIDEO_CRF = 30              # HEVC 目标档（nvenc 用 -cq，x265 用 -crf）
RECORD_VIDEO_X264_CRF = 23         # x264 回退档（与 v1.0.8 现状一致）
ENCODER_PROBE_ENABLED = True       # 编码器探针开关（离线测试/极低端机可关，关=直接 x264）
ENCODER_PROBE_FRAME_COUNT = 45     # x265 速度探针每流合成帧数
ENCODER_PROBE_MAX_STREAMS = 4      # x265 探针并行流数封顶
ENCODER_PROBE_TIMEOUT_S = 15       # 单个探针子进程超时（秒）
ENCODER_X265_MIN_FPS_RATIO = 1.5   # x265 采用门槛：实测 fps ≥ 录制帧率 × 该比值
IMU_PENDING_MAX_SAMPLES = 18000    # 双目 IMU 防丢缓冲上限（~1 分钟 @300Hz）
DROP_WARN_RATIO = 0.01             # 丢帧提示阈值：丢帧 > 总帧数×该比例 或
DROP_WARN_MIN_COUNT = 30           #   丢帧 > 该帧数 → 提示换编码器/降分辨率

# ── 设备命名规范 (EgoData 标准) ───────────────────────
# 命名约定: <位置>_<模态>
#   头戴左目: head_left_rgb
#   头戴右目: head_right_rgb
#   头戴深度: head_depth
#   触觉传感器: left_glove / right_glove
#   手部关键点: left_hand_pose / right_hand_pose
#
CAMERA_LEFT  = "head_left_rgb"
CAMERA_RIGHT = "head_right_rgb"
CAMERA_DEPTH = "head_depth"

def _camera_slot_name(index: int) -> str:
    """OpenCV 设备索引 → EgoData 槽位名。

    index 0 → head_left_rgb（主相机 / 单目左目）
    index 1 → head_right_rgb（双目右目）
    index 2+ → head_right_rgb_2, head_right_rgb_3, …
    """
    if index == 0:
        return CAMERA_LEFT
    if index == 1:
        return CAMERA_RIGHT
    return f"{CAMERA_RIGHT}_{index}"

# 兼容旧命名（供读取旧 LeRobot v3 会话使用）
CAMERA_PRIMARY_LEGACY = "head_rgb"
def _camera_slot_name_legacy(index: int) -> str:
    if index == 0:
        return CAMERA_PRIMARY_LEGACY
    return f"{CAMERA_PRIMARY_LEGACY}_{index}"

# ── EgoData 输出格式 ─────────────────────────────────
EPISODE_PREFIX = "episode"
EPISODE_DIGITS = 6                 # episode_000001, episode_000002, …
DEPTH_FORMAT = "depth_mkv"          # 双流 MKV：流0=热力图 h264（默认播放画面），
                                   # 流1=FFV1 gray16le 无损 uint16 毫米深度。
                                   # 旧版 png16（PNG 序列）与 raw16 bin 均已弃用——
                                   # 后者实测体积 6× 且上传被拒收（v1.0.11 回退），
                                   # 前者 2026-09 被双流 MKV 取代（体积减半+彩色画面）
DEPTH_SCALE = 0.001                # 深度单位: 像素值 × DEPTH_SCALE = 米

# ── 双目标定默认值 ───────────────────────────────────
STEREO_BASELINE = 0.095            # 基线 (米)
STEREO_RESOLUTION = (1280, 800)    # 双目默认分辨率
STEREO_FPS = 30                   # 双目默认帧率
STEREO_CAM_FPS = 50               # S80M/S80C 相机档（50=默认，50→30 桶抽帧才录到 30fps；8/31 撕裂根因已由回调取帧修复——官方 GUI 同款回调+深度组合实测无撕裂。25 档保留为回退：25fps 相机下 30 桶只填到 25 帧）
STEREO_RECORD_FPS = 30             # S80M 录制帧率（相机帧按 wall 时钟 1/30s 桶抽帧，hw 时钟跳变不空桶）
STEREO_RECORD_MIN_INTERVAL_S = 1.0 / STEREO_RECORD_FPS  # 抽帧桶长（秒）
STEREO_CB_BRIDGE = True           # S80M/S80C 取帧用官方 GUI 同款回调（装配完成才交付；shim 缺失/注册失败自动回退轮询）

# ── S80M/S80C 深度（SDK 深度引擎，子进程内计算） ─────────────
S80M_DEPTH_ENABLED = True      # S80C 深度开关（关闭→子进程不加 --depth-sdk-dir，协议字节不变）
S80M_DEPTH_SLOT    = "stereo_depth"   # 深度槽位（depth/stereo_depth/…，双流 MKV）
S80M_DEPTH_NEAR_MM = 300       # 热力图固定色标下限 mm（引擎量程 300-10000，中位 ~1.7m）
S80M_DEPTH_FAR_MM  = 3000      # 热力图固定色标上限 mm
S80M_DEPTH_SMOOTH_K = 3        # 热力图中值滤波核

# ── D435 深度双目配置 ───────────────────────────────────
D435_SLOT_RGB    = "d435_rgb"     # RGB 彩色槽位（录制为 videos/d435_rgb/…）
D435_SLOT_DEPTH  = "d435_depth"   # 深度槽位（depth/d435_depth/…，双流 MKV）
D435_RESOLUTION  = (848, 480)     # (width, height)，Depth Z16 @30；左右红外仅内部启用
D435_RGB_RESOLUTION = (1280, 720) # (width, height)，RGB 彩色流（D435 标准 720p 全视场）
D435_FPS         = 30
D435_DEPTH_NEAR_MM = 300           # 实时显示深度范围过滤（毫米，参考 S80M 深度 demo 0.3m）
D435_DEPTH_FAR_MM  = 4000          # 实时显示深度范围过滤（毫米，4.0m）
D435_PNG_COMPRESSION = 1           # 原始深度 PNG 压缩级（1 最快）
D435_STALL_TIMEOUT_S = 5           # 流停滞阈值（秒）：连续无帧超过该值触发重连
                                   # （双 RealSense 第二台开流偶发静默停滞，无此
                                   # 看门狗会无限空转且无任何报错）
D435_LOW_FPS_WINDOW_S = 10         # 帧率看门狗窗口（秒）：窗口内帧数低于期望
                                   # fps 的 D435_LOW_FPS_FRACTION → 触发重连
                                   # （半死状态——wait_for_frames 零星出帧不报错
                                   # 但画面极卡，静默看门狗测不到）
D435_LOW_FPS_FRACTION = 0.5        # 帧率下限比例（0.5 = 期望帧率的一半）

# ── RealSense D400 系列多型号支持 ──────────────────────
# 同一采集模式（面板按序列号区分具体设备），但不同型号的深度原生
# 分辨率 / 适用距离不同，按型号给采集配置（resolution 为 (width, height)）。
REALSENSE_PROFILES = {
    "D405": {
        "depth_resolution": (1280, 720),   # D405 原生深度分辨率（短距近景相机）
        "rgb_resolution": (1280, 720),     # D405 RGB 最高 720p
        "fps": 30,
        "depth_near_mm": 100,              # 显示/热力图固定色标（D405 理想工作距 0.07-0.5m）
        "depth_far_mm": 1000,
        "heatmap_smooth_k": 3,             # 热力图 3×3 中值（单像素椒盐噪点）
        "heatmap_temporal_alpha": 0.5,     # 热力图时域 EMA 权重（帧间抖动；仅可视化）
    },
}

def realsense_profile(model_name: str) -> dict:
    """按设备型号名返回采集配置；未收录型号回落 D435 默认。"""
    for key, prof in REALSENSE_PROFILES.items():
        if key in (model_name or ""):
            return prof
    return {
        "depth_resolution": D435_RESOLUTION,
        "rgb_resolution": D435_RGB_RESOLUTION,
        "fps": D435_FPS,
        "depth_near_mm": D435_DEPTH_NEAR_MM,
        "depth_far_mm": D435_DEPTH_FAR_MM,
        "heatmap_smooth_k": 0,
        "heatmap_temporal_alpha": 0.0,
    }

# ── 设备检测面板 ─────────────────────────────────────
DEVICE_SCAN_MAX_INDEX = 16         # V4L2 sysfs 枚举上限（D435 占 0-5 时 webcam 常落在 6+）
DEVICE_POLL_INTERVAL_MS = 2000     # 设备列表轮询间隔（毫秒，sysfs 只读扫描）
REALSENSE_VID = "8086"             # Intel vendor ID（排除 RealSense UVC 节点用）
REALSENSE_PID = "0b07"             # D435 的 product ID；其它型号（如 D435i=0b3a）靠驱动 name 含 "RealSense" 兜底

# 传感器命名（写入 parquet observation.<name> 列）
SENSOR_NAMES = [
    "right_glove",     # 右手手套传感器（16×16 压力矩阵）
    "left_glove",      # 左手手套传感器
]

# 手部关键点命名（每列 21 关节 × 3 坐标 xyz）
HAND_POSE_LEFT  = "left_hand_pose"    # observation.left_hand_pose
HAND_POSE_RIGHT = "right_hand_pose"   # observation.right_hand_pose
HAND_POSE_DIM   = 21 * 3              # 63 维 (x,y,z 关节坐标)

# 传感器数据维度（16×16 触觉矩阵展平）
SENSOR_DIM = 256

# ── 显示参数 ──────────────────────────────────────────
CAMERA_MIRROR_HORIZONTAL = True       # 单目相机左右镜像翻转（显示 + 录制）
DISPLAY_FPS_LIMIT = 30                # GUI 显示帧率上限
FEED_MIN_WIDTH = 320
FEED_MIN_HEIGHT = 240

# ── 暗色主题颜色（Qt-Material dark_teal） ─────────────
COLOR_RECORDING = "#EF5350"           # 红色 — 录制中
COLOR_STOPPED = "#66BB6A"             # 绿色 — 空闲
COLOR_ABNORMAL = "#FFA726"            # 橙色 — 异常 / 警告

# 主背景
COLOR_BG_MAIN = "#1E1E1E"             # 窗口主背景
COLOR_BG_PANEL = "#252525"            # 面板背景
COLOR_BG_WIDGET = "#2D2D2D"           # 控件 / 容器背景
COLOR_BG_CANVAS = "#1A1A1A"           # 画布 / 网格背景

# 文字
COLOR_TEXT_PRIMARY = "#E0E0E0"        # 主文字
COLOR_TEXT_SECONDARY = "#9E9E9E"      # 次要文字
COLOR_TEXT_HINT = "#616161"           # 提示文字

# 边框与分割线
COLOR_BORDER = "#424242"
COLOR_BORDER_STRONG = "#616161"

# 按钮颜色
COLOR_BTN_START = "#43A047"           # 开始录制按钮（绿）
COLOR_BTN_STOP = "#E53935"            # 停止录制按钮（红）
COLOR_BTN_ABORT = "#FF8F00"           # 异常停止按钮（橙）
COLOR_BTN_DEFAULT_BG = "#424242"      # 普通按钮背景
COLOR_BTN_HOVER = "#555555"           # 按钮悬停
COLOR_BTN_DISABLED_BG = "#333333"     # 禁用按钮背景
COLOR_BTN_DISABLED_TEXT = "#616161"   # 禁用按钮文字

# ── 服务器上传配置 ────────────────────────────────────
# 出厂默认占位地址；实际地址由用户在页面中填写并持久化到 server_config.json。
SERVER_URL = "http://127.0.0.1:8000"
SERVER_CONFIG_FILE = os.path.join(DATA_DIR, "server_config.json")

def _load_server_config() -> dict:
    """读取 server_config.json，返回完整字典。"""
    import json
    try:
        if os.path.isfile(SERVER_CONFIG_FILE):
            with open(SERVER_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return {}

def _save_server_config(data: dict):
    """持久化 server_config.json（合并写入，保留未传入的旧字段）。"""
    import json
    current = _load_server_config()
    current.update(data)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SERVER_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)

def load_server_url() -> str:
    """读取用户保存的服务器地址，若无则返回出厂默认值。"""
    data = _load_server_config()
    url = data.get("server_url", "").strip()
    return url if url else SERVER_URL

def save_server_url(url: str):
    """持久化用户输入的服务器地址。"""
    _save_server_config({"server_url": url.strip()})

def load_credentials() -> tuple[str, str]:
    """读取保存的登陆凭据，返回 (username, password)。"""
    data = _load_server_config()
    return data.get("username", ""), data.get("password", "")

def save_credentials(username: str, password: str):
    """持久化登陆凭据。"""
    _save_server_config({"username": username.strip(), "password": password})

def load_remembered_username() -> str:
    """读取登录对话框「记住账号」保存的用户名（server_config.json）。"""
    return str(_load_server_config().get("remembered_username", "") or "").strip()

def save_remembered_username(username: str):
    """持久化「记住账号」的用户名（空串 = 清除）。"""
    _save_server_config({"remembered_username": (username or "").strip()})

def load_upload_auto_sync() -> bool:
    """读取"录制完成后自动上传"开关（server_config.json，默认开=沿用历史行为）。"""
    return bool(_load_server_config().get("upload_auto_sync", True))

def save_upload_auto_sync(on: bool):
    """持久化"录制完成后自动上传"开关。"""
    _save_server_config({"upload_auto_sync": bool(on)})

def load_upload_delete_after() -> bool:
    """读取"上传成功后自动删除本地文件"开关（server_config.json，默认关）。"""
    return bool(_load_server_config().get("upload_delete_after", False))

def save_upload_delete_after(on: bool):
    """持久化"上传成功后自动删除本地文件"开关。"""
    _save_server_config({"upload_delete_after": bool(on)})

def load_upload_project_id() -> str:
    """读取上传目标项目 ID（server_config.json）。

    空字符串 = 不指定，由服务器按会话名自动匹配项目
    （服务器上存在多个同名/近似项目时会返回 409 歧义错误）。
    """
    return str(_load_server_config().get("upload_project_id", "") or "").strip()

def save_upload_project_id(project_id: str):
    """持久化上传目标项目 ID（空 = 恢复服务器自动匹配）。"""
    _save_server_config({"upload_project_id": (project_id or "").strip()})
# ── 设备命名持久化 ────────────────────────────────────
# key = DeviceInfo.stable_key（"uvc:{by-id前缀}" / "d435:{serial}" / "ble:{MAC}" 等）
# value = 用户命名字符串；data_ble 设备另带 "sensor" 字段绑定 parquet 列名
# （"right_glove"/"left_glove"，按 MAC 持久化，左右手配置各不同）
DEVICE_NAMES_FILE = os.path.join(DATA_DIR, "device_names.json")

def load_device_names() -> dict:
    """读取设备命名表，异常/不存在返回 {}。"""
    try:
        if os.path.isfile(DEVICE_NAMES_FILE):
            import json
            with open(DEVICE_NAMES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        pass
    return {}

def _write_device_names(data: dict):
    import json
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DEVICE_NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_device_name(key: str, name: str, sensor: str = ""):
    """持久化设备命名（merge-write，保留未传入条目）。

    sensor: 仅 data_ble 用，绑定 parquet 列名（"" 表示不动原值）。
    """
    import json
    current = load_device_names()
    # 条目统一规范为 {"name": str, "sensor"?: str}；旧版纯字符串自动升级
    entry = current.get(key, "")
    if isinstance(entry, str):
        entry = {"name": entry}
    entry = dict(entry) if entry else {}
    if name:
        entry["name"] = name
    if sensor:
        entry["sensor"] = sensor
    if not entry.get("name") and not entry.get("sensor"):
        return
    current[key] = entry
    _write_device_names(current)

def device_name(key: str) -> str:
    """读单个设备命名（兼容 {"name":…} 结构或纯字符串）。"""
    entry = load_device_names().get(key, "")
    return entry["name"] if isinstance(entry, dict) else (entry or "")

def device_sensor_role(key: str) -> str:
    """读 data_ble 设备绑定的 parquet 列名（无则空串）。"""
    entry = load_device_names().get(key, "")
    return entry.get("sensor", "") if isinstance(entry, dict) else ""

def assign_glove_sensor_role(key: str, prefer: str = "") -> str:
    """为首次连接的未知手套分配下一个空闲 parquet 列名并持久化。

    按 MAC 绑定（重连保持）；prefer 为期望列名（广播名 'L' → left_glove、
    'R' → right_glove），空闲则优先占用，否则按 SENSOR_NAMES 顺序取
    下一个空余名。无空余名时兜底取最后一个。
    """
    role = device_sensor_role(key)
    if role:
        return role
    used = {device_sensor_role(k) for k in load_device_names()}
    candidates = [n for n in ([prefer] + SENSOR_NAMES) if n]
    for name in dict.fromkeys(candidates):   # 去重保持顺序
        if name not in used:
            save_device_name(key, device_name(key), sensor=name)
            return name
    return SENSOR_NAMES[-1] if SENSOR_NAMES else ""

def remove_device_name(key: str):
    """删除设备命名条目。"""
    current = load_device_names()
    if key in current:
        del current[key]
        _write_device_names(current)

# ── 每设备相机参数持久化（曝光） ──────────────────────
# key 与 device_names.json 同为 DeviceInfo.stable_key；
# value = {"exposure": {"auto": bool, "value": float}}（auto=True 时 value 忽略）
DEVICE_PARAMS_FILE = os.path.join(DATA_DIR, "device_params.json")

def load_device_params() -> dict:
    """读取每设备相机参数表，异常/不存在返回 {}。"""
    try:
        if os.path.isfile(DEVICE_PARAMS_FILE):
            import json
            with open(DEVICE_PARAMS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        pass
    return {}

def _write_device_params(data: dict):
    import json
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DEVICE_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_device_exposure(key: str, auto: bool, value: float):
    """持久化单设备曝光（merge-write，保留其他条目/参数）。"""
    current = load_device_params()
    entry = dict(current.get(key) or {})
    entry["exposure"] = {"auto": bool(auto), "value": float(value)}
    current[key] = entry
    _write_device_params(current)

def device_exposure(key: str) -> dict:
    """读单设备曝光设置 {"auto": bool, "value": float}，无则 None。"""
    entry = load_device_params().get(key) or {}
    exp = entry.get("exposure")
    if isinstance(exp, dict) and "auto" in exp and "value" in exp:
        return {"auto": bool(exp["auto"]), "value": float(exp["value"])}
    return None


def ensure_device_original(key: str, auto: bool, value: float) -> bool:
    """记录设备「最一开始」的曝光基线（首次看到才写，之后永不覆盖）。

    恢复默认按钮的基线：无论之后怎么调整，都能回到首次开启设备时
    读回的原厂曝光状态。返回 True 表示本次为新写入。
    """
    current = load_device_params()
    entry = dict(current.get(key) or {})
    if isinstance(entry.get("original"), dict):
        return False
    entry["original"] = {"auto": bool(auto), "value": float(value)}
    current[key] = entry
    _write_device_params(current)
    return True


def device_original(key: str) -> dict:
    """读设备原始曝光基线 {"auto": bool, "value": float}，无则 None。"""
    entry = load_device_params().get(key) or {}
    orig = entry.get("original")
    if isinstance(orig, dict) and "auto" in orig and "value" in orig:
        return {"auto": bool(orig["auto"]), "value": float(orig["value"])}
    return None

UPLOAD_ENABLED = True                          # 是否启用自动上传
UPLOAD_MAX_CONCURRENT = 1                      # 最大并发上传数：串行=1（并发时预压缩临时文件会互相覆盖致视频损坏，且进度无法区分当前是哪条数据）
UPLOAD_RETRY_MAX = 3                           # 最大重试次数
UPLOAD_AUTO_SYNC = load_upload_auto_sync()     # 录制完成后是否自动上传（持久化，启动时从 server_config.json 读取）
UPLOAD_DELETE_AFTER = load_upload_delete_after()  # 上传成功后是否删除本地文件（持久化，启动时从 server_config.json 读取）
UPLOAD_PRECOMPRESS_VIDEO = True                # 上传前是否把 videos 重编码到低码率。开=HEVC(libx265) CRF 档再压（tools/hevc_test 实测 CRF30 体积约原录制件的 7%、SSIM 0.967；大会话必备）。关=原样上传 CRF23 录制品（无二代有损；60min 双目 ~7.6GB，上传带宽/后端超时需能承受）
UPLOAD_VIDEO_CRF = 30                          # HEVC 目标码率档（x265 CRF 越大越小；30 实测体积约 7%；仅 UPLOAD_PRECOMPRESS_VIDEO=True 时生效）

# ── 任务服务配置 ──────────────────────────────────────
TASK_POLL_INTERVAL_MS = 30000                  # 任务轮询间隔（毫秒）
TASK_API_URL = SERVER_URL                      # 任务 API 地址（复用服务器地址）
DEVICE_NAME = "EGO_001"                        # 设备认领名（后端任务分配依据）

# ── 手部关键点追踪配置 ──────────────────────────────────
HAND_TRACK_ENABLED = False                       # 是否启用录制后手部关键点处理（需 ultralytics）
HAND_DETECTION_DIR = os.path.join(BASE_DIR, "tools", "hand_detection")
HAND_DET_MODEL = os.path.join(HAND_DETECTION_DIR, "best.pt")           # YOLO 手套检测模型
HAND_MEDIAPIPE_MODEL = os.path.join(HAND_DETECTION_DIR, "hand_landmarker.task")  # MediaPipe 裸手模型
HAND_TRACK_MODE = "glove"                        # 追踪模式: "glove"(黑色手套) / "bare"(裸手)
HAND_DET_DEVICE = "cuda"                         # 检测器设备 ("cuda" / "cpu")
HAND_POSE_DEVICE = "cuda"                        # 关键点设备 ("cuda" / "cpu")
HAND_TRACK_MAX_HANDS = 2                         # 最多追踪手数
HAND_DATA_DIM = 21 * 2 * 2 + 4 * 2 + 1           # 展平后数据维度

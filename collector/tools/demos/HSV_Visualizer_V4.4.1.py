import sys
import platform
import cv2
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QInputDialog, QMessageBox,
                             QDialog, QSlider, QFormLayout, QGroupBox, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QCheckBox, QSpinBox, QDoubleSpinBox

# HSV_Visualizer_V4.4.1 //复刻 sightac 热力图完整链路 + Hue曲线 + 传感器ROI裁切

# ================= 平台检测 =================
# Windows=DirectShow, Linux=V4L2, 其他(macOS等)=自动选择
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'
if IS_WINDOWS:
    CAP_BACKEND = cv2.CAP_DSHOW
elif IS_LINUX:
    CAP_BACKEND = cv2.CAP_V4L2
else:
    CAP_BACKEND = cv2.CAP_ANY  # macOS 等使用默认后端


def open_camera(port):
    """打开摄像头；Linux 下固定使用 V4L2 + MJPG 640x480@30fps。"""
    backend = cv2.CAP_V4L2 if IS_LINUX else CAP_BACKEND
    cap = cv2.VideoCapture(port, backend)

    if IS_LINUX and cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    return cap
# ================= 相机参数定义 =================
# OpenCV CAP_PROP 映射 + 滑块范围
CAM_PARAMS = [
    ("亮度 (Brightness)", cv2.CAP_PROP_BRIGHTNESS,   0, 255, 10),
    ("对比度 (Contrast)",  cv2.CAP_PROP_CONTRAST,    0, 255, 10),
    ("色调 (Hue)",         cv2.CAP_PROP_HUE,         0, 180, 0),
    ("饱和度 (Saturation)", cv2.CAP_PROP_SATURATION,  0, 255, 64),
    ("锐度 (Sharpness)",   cv2.CAP_PROP_SHARPNESS,   0, 255, 0),
    ("伽马 (Gamma)",       cv2.CAP_PROP_GAMMA,       1, 500, 100),
    ("增益 (Gain)",        cv2.CAP_PROP_GAIN,        16, 248, 16),
    ("曝光 (Exposure)",    cv2.CAP_PROP_EXPOSURE,    -13, -1, -5),
    ("白平衡 (WB Temp)",    cv2.CAP_PROP_WB_TEMPERATURE, 2800, 6500, 4600),
]

# 自动重连参数
AUTO_RECONNECT_THRESHOLD = 30   # 连续失败30帧(~1秒)触发自动重连
RETRY_INTERVAL_FRAMES = 67      # 重连失败后每~2秒重试一次(67帧×30ms)

# ================= 独立的 ROI 设置弹窗 =================
class ROIDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROI 详细设置")
        self.setFixedSize(350, 480) 
        self.parent = parent 
        
        # 保存一份默认参数，用于重置功能
        self.default_params = {
            'r1_x': 275, 'r1_y': 320, 'r1_w': 50, 'r1_h': 55,
            'r2_x': 275, 'r2_y': 120, 'r2_w': 50, 'r2_h': 55
        }
        
        layout = QVBoxLayout()

        # --- 分组 1: 绿色 ROI 1 ---
        group1 = QGroupBox("ROI 1 (绿色区域)")
        form1 = QFormLayout()
        self.r1_x, self.lbl_1x, lay1_x = self.create_slider_with_label(0, 640, parent.roi_params['r1_x'])
        self.r1_y, self.lbl_1y, lay1_y = self.create_slider_with_label(0, 480, parent.roi_params['r1_y'])
        self.r1_w, self.lbl_1w, lay1_w = self.create_slider_with_label(1, 320, parent.roi_params['r1_w'])
        self.r1_h, self.lbl_1h, lay1_h = self.create_slider_with_label(1, 240, parent.roi_params['r1_h'])
        form1.addRow("X 坐标:", lay1_x)
        form1.addRow("Y 坐标:", lay1_y)
        form1.addRow("宽度:", lay1_w)
        form1.addRow("高度:", lay1_h)
        group1.setLayout(form1)

        # --- 分组 2: 红色 ROI 2 ---
        group2 = QGroupBox("ROI 2 (红色区域)")
        form2 = QFormLayout()
        self.r2_x, self.lbl_2x, lay2_x = self.create_slider_with_label(0, 640, parent.roi_params['r2_x'])
        self.r2_y, self.lbl_2y, lay2_y = self.create_slider_with_label(0, 480, parent.roi_params['r2_y'])
        self.r2_w, self.lbl_2w, lay2_w = self.create_slider_with_label(1, 640, parent.roi_params['r2_w'])
        self.r2_h, self.lbl_2h, lay2_h = self.create_slider_with_label(1, 480, parent.roi_params['r2_h'])
        form2.addRow("X 坐标:", lay2_x)
        form2.addRow("Y 坐标:", lay2_y)
        form2.addRow("宽度:", lay2_w)
        form2.addRow("高度:", lay2_h)
        group2.setLayout(form2)

        layout.addWidget(group1)
        layout.addWidget(group2)

        # --- 重置按钮 ---
        self.btn_reset = QPushButton("重置为默认参数")
        self.btn_reset.clicked.connect(self.reset_to_defaults)
        layout.addWidget(self.btn_reset)

        self.setLayout(layout)

    def create_slider_with_label(self, min_val, max_val, default_val):
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0) 
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        
        label = QLabel(str(default_val))
        label.setFixedWidth(35) 
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        h_layout.addWidget(slider)
        h_layout.addWidget(label)
        
        slider.valueChanged.connect(lambda val, l=label: self.on_slider_changed(val, l))
        return slider, label, container

    def on_slider_changed(self, val, label):
        label.setText(str(val)) 
        self.update_params()    

    def update_params(self):
        self.parent.roi_params = {
            'r1_x': self.r1_x.value(), 'r1_y': self.r1_y.value(),
            'r1_w': self.r1_w.value(), 'r1_h': self.r1_h.value(),
            'r2_x': self.r2_x.value(), 'r2_y': self.r2_y.value(),
            'r2_w': self.r2_w.value(), 'r2_h': self.r2_h.value()
        }

    def reset_to_defaults(self):
        self.r1_x.setValue(self.default_params['r1_x'])
        self.r1_y.setValue(self.default_params['r1_y'])
        self.r1_w.setValue(self.default_params['r1_w'])
        self.r1_h.setValue(self.default_params['r1_h'])
        
        self.r2_x.setValue(self.default_params['r2_x'])
        self.r2_y.setValue(self.default_params['r2_y'])
        self.r2_w.setValue(self.default_params['r2_w'])
        self.r2_h.setValue(self.default_params['r2_h'])


# ================= 相机参数设置弹窗 =================
class CameraParamsDialog(QDialog):
    """弹窗式相机参数调节，滑块拖动实时写入 cv2.VideoCapture"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("相机参数设置")
        self.setFixedSize(550, 450)
        self.parent = parent

        layout = QVBoxLayout()

        grp = QGroupBox("实时参数调节（拖动滑块即时生效）")
        form = QFormLayout()

        self._sliders = {}
        self._spinboxes = {}
        self._updating = False  # 防循环

        for label, prop_id, lo, hi, default in CAM_PARAMS:
            # 读取相机当前值（Linux V4L2 和 Windows DirectShow 范围不同）
            try:
                cur_val = int(parent.cap.get(prop_id)) if parent.cap.isOpened() else default
            except Exception:
                cur_val = default

            if IS_LINUX:
                if prop_id == cv2.CAP_PROP_EXPOSURE and cur_val > 0:
                    # V4L2 100µs → log2 秒（与 Windows 显示一致）
                    cur_val = int(round(np.log2(cur_val * 0.0001)))
                    cur_val = max(lo, min(hi, cur_val))
                    actual_lo, actual_hi = lo, hi
                else:
                    # 其他参数：动态扩展范围以适应 V4L2 实际值
                    actual_lo = min(lo, cur_val)
                    actual_hi = max(hi, cur_val)
            else:
                # Windows: -1 表示不支持，fallback 到默认值
                if cur_val == -1:
                    cur_val = default
                cur_val = max(lo, min(hi, cur_val))
                actual_lo, actual_hi = lo, hi

            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(actual_lo, actual_hi)
            slider.setValue(cur_val)
            slider.setToolTip(f"{actual_lo} ~ {actual_hi}")

            spin = QSpinBox()
            spin.setRange(actual_lo, actual_hi)
            spin.setValue(cur_val)
            spin.setFixedWidth(65)

            slider.valueChanged.connect(lambda v, p=prop_id, s=spin: self._on_slider(p, v, s))
            spin.valueChanged.connect(lambda v, p=prop_id, sl=slider: self._on_spin(p, v, sl))

            h.addWidget(slider, stretch=1)
            h.addWidget(spin)

            self._sliders[prop_id] = slider
            self._spinboxes[prop_id] = spin
            form.addRow(label, row)

        grp.setLayout(form)
        layout.addWidget(grp)

        # --- 自动曝光 / 自动白平衡 复选框 ---
        grp_auto = QGroupBox("自动控制")
        auto_layout = QHBoxLayout()

        self._chk_auto_exp = QCheckBox("自动曝光")
        self._chk_auto_wb = QCheckBox("自动白平衡")

        # Linux V4L2: AUTO_EXPOSURE 0=自动 1=手动（与Windows相反）
        # Windows DShow: AUTO_EXPOSURE 0=手动 1=自动
        # 读取相机当前状态来设置复选框初始值（不写入）
        self._chk_auto_exp.setChecked(False)
        self._chk_auto_wb.setChecked(False)
        if parent.cap.isOpened():
            try:
                auto_exp_val = parent.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                if IS_LINUX:
                    # V4L2: 0=自动模式 → 勾选；1=手动模式 → 不勾选
                    self._chk_auto_exp.setChecked(auto_exp_val == 0)
                else:
                    # Windows: >0 表示自动模式开启
                    self._chk_auto_exp.setChecked(auto_exp_val > 0)
            except Exception:
                pass
            try:
                auto_wb_val = parent.cap.get(cv2.CAP_PROP_AUTO_WB)
                self._chk_auto_wb.setChecked(auto_wb_val > 0)
            except Exception:
                pass

        self._chk_auto_exp.toggled.connect(self._on_auto_exp_toggled)
        self._chk_auto_wb.toggled.connect(self._on_auto_wb_toggled)

        auto_layout.addWidget(self._chk_auto_exp)
        auto_layout.addWidget(self._chk_auto_wb)
        grp_auto.setLayout(auto_layout)
        layout.addWidget(grp_auto)

        # --- 重置按钮 ---
        btn_reset = QPushButton("重置为默认值")
        btn_reset.clicked.connect(self._reset_defaults)
        layout.addWidget(btn_reset)

        self.setLayout(layout)

    def _on_slider(self, prop_id, value, spin):
        if self._updating:
            return
        self._updating = True
        spin.setValue(value)
        self._updating = False
        self._write_camera(prop_id, value)

    def _on_spin(self, prop_id, value, slider):
        if self._updating:
            return
        self._updating = True
        slider.setValue(value)
        self._updating = False
        self._write_camera(prop_id, value)

    def _write_camera(self, prop_id, value):
        """实时写入相机（Linux V4L2 曝光值需从 log2 转回 100µs 单位）"""
        if self.parent.cap.isOpened():
            try:
                if IS_LINUX and prop_id == cv2.CAP_PROP_EXPOSURE:
                    # log2 秒 → V4L2 100µs 单位
                    value = max(1, int(round((2 ** value) * 10000)))
                self.parent.cap.set(prop_id, value)
            except Exception:
                pass  # V4L2 驱动可能不支持某些属性

    def _on_auto_exp_toggled(self, checked):
        if IS_LINUX:
            # V4L2: 0=自动 1=手动（与Windows相反）
            self._write_camera(cv2.CAP_PROP_AUTO_EXPOSURE, 0 if checked else 1)
        else:
            self._write_camera(cv2.CAP_PROP_AUTO_EXPOSURE, 1 if checked else 0)

    def _on_auto_wb_toggled(self, checked):
        self._write_camera(cv2.CAP_PROP_AUTO_WB, 1 if checked else 0)

    def _reset_defaults(self):
        # 先关闭自动曝光/白平衡，再写入参数值，否则自动算法会覆盖写入
        self._chk_auto_exp.setChecked(False)
        self._chk_auto_wb.setChecked(False)
        for label, prop_id, lo, hi, default in CAM_PARAMS:
            self._updating = True
            self._sliders[prop_id].setValue(default)
            self._spinboxes[prop_id].setValue(default)
            self._updating = False
            self._write_camera(prop_id, default)


# ================= 🆕 V4.4.1: 热力图参数设置弹窗 =================
class HeatmapSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("热力图参数设置")
        self.setFixedSize(380, 310)
        self.parent = parent

        self.default_params = {'scale': 10.0, 'mask_b': 50, 'mask_r': 130,
                               'hue_threshold': 5.0, 'fz_p1': 1.0}

        layout = QVBoxLayout()

        grp = QGroupBox("热力图链路参数")
        form = QFormLayout()

        # --- 热力图缩放 (1.0 ~ 500.0，滑块+输入框双向同步) ---
        scale_row = QWidget()
        scale_h = QHBoxLayout(scale_row)
        scale_h.setContentsMargins(0, 0, 0, 0)
        self._scale_sync = False  # 防循环标志
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(10, 5000)
        self.scale_slider.setValue(int(parent.heatmap_scale * 10))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(1.0, 500.0)
        self.scale_spin.setDecimals(1)
        self.scale_spin.setValue(parent.heatmap_scale)
        self.scale_spin.setFixedWidth(75)
        self.scale_slider.valueChanged.connect(self._on_scale_slider)
        self.scale_spin.valueChanged.connect(self._on_scale_spin)
        scale_h.addWidget(self.scale_slider)
        scale_h.addWidget(self.scale_spin)
        form.addRow("热力图缩放:", scale_row)

        # --- Mask B阈值 ---
        self.mask_b_spin = QSpinBox()
        self.mask_b_spin.setRange(0, 255)
        self.mask_b_spin.setValue(parent.mask_b_threshold)
        form.addRow("Mask B阈值:", self.mask_b_spin)

        # --- Mask R阈值 ---
        self.mask_r_spin = QSpinBox()
        self.mask_r_spin.setRange(0, 255)
        self.mask_r_spin.setValue(parent.mask_r_threshold)
        form.addRow("Mask R阈值:", self.mask_r_spin)

        # --- Hue 阈值 ---
        self.hue_thresh_spin = QDoubleSpinBox()
        self.hue_thresh_spin.setRange(0.0, 180.0)
        self.hue_thresh_spin.setDecimals(1)
        self.hue_thresh_spin.setValue(parent.hue_threshold)
        form.addRow("Hue 阈值:", self.hue_thresh_spin)

        # --- 标定系数 ---
        self.fz_p1_spin = QDoubleSpinBox()
        self.fz_p1_spin.setRange(0.01, 100.0)
        self.fz_p1_spin.setDecimals(2)
        self.fz_p1_spin.setValue(parent.fz_p1)
        form.addRow("标定系数 fz_p1:", self.fz_p1_spin)

        grp.setLayout(form)
        layout.addWidget(grp)

        # --- 按钮行: 重置 | 确认 ---
        btn_row = QWidget()
        btn_h = QHBoxLayout(btn_row)
        btn_h.setContentsMargins(0, 0, 0, 0)
        btn_reset = QPushButton("重置为默认值")
        btn_reset.setAutoDefault(False)
        btn_reset.clicked.connect(self._reset_defaults)
        btn_confirm = QPushButton("确认修改")
        btn_confirm.setDefault(True)
        btn_confirm.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        btn_confirm.clicked.connect(self._confirm)
        btn_h.addWidget(btn_reset)
        btn_h.addWidget(btn_confirm)
        layout.addWidget(btn_row)

        self.setLayout(layout)

    def _on_scale_slider(self, v):
        if self._scale_sync: return
        self._scale_sync = True
        self.scale_spin.setValue(v / 10.0)
        self._scale_sync = False

    def _on_scale_spin(self, v):
        if self._scale_sync: return
        self._scale_sync = True
        self.scale_slider.setValue(int(v * 10))
        self._scale_sync = False

    def _reset_defaults(self):
        self.scale_slider.setValue(int(self.default_params['scale'] * 10))
        self.mask_b_spin.setValue(self.default_params['mask_b'])
        self.mask_r_spin.setValue(self.default_params['mask_r'])
        self.hue_thresh_spin.setValue(self.default_params['hue_threshold'])
        self.fz_p1_spin.setValue(self.default_params['fz_p1'])

    def _confirm(self):
        p = self.parent
        p.heatmap_scale = self.scale_spin.value()
        p.mask_b_threshold = self.mask_b_spin.value()
        p.mask_r_threshold = self.mask_r_spin.value()
        p.hue_threshold = self.hue_thresh_spin.value()
        p.fz_p1 = self.fz_p1_spin.value()
        self.accept()


# ================= 主界面窗口 =================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSV_Visualizer_V4.4.1")
        
        # 初始窗口大小匹配 16:9 宽高比（2560×1440 → 1280×720 图区 + 按钮栏 ~50px）
        self.resize(1280, 770)
        
        # --- 初始化变量 ---
        self.cam_port = 0
        # 默认使用 DirectShow 后端
        self.cap = open_camera(self.cam_port)
        self._setup_camera_defaults()

        # 🆕 V4.4: 传感器 ROI 裁切（复刻 touch_sensor.py:702，不翻转）
        self.sensor_roi = (0, 640, 0, 480)  # (x1, x2, y1, y2) 默认全帧

        # 🆕 V4.4: 热力图链路参数（复刻 touch_sensor.py 对应逻辑）
        self.fz_p1 = 1.0               # 力标定系数（应填入传感器出厂值）
        self.hue_threshold = 5.0       # Hue差分阈值（平面款默认5.0）
        self.hue_baseline = None       # H通道基准
        self._fz_matrix = None         # 热力图数据源 (float32)
        self.mask_b_threshold = 50     # Mask B阈值
        self.mask_r_threshold = 130    # Mask R阈值

        # 相机打开后自动校准基线
        if self.cap.isOpened():
            self.calibrate_baseline()

        self.roi_params = {
            'r1_x': 150, 'r1_y': 200, 'r1_w': 50, 'r1_h': 55,
            'r2_x': 275, 'r2_y': 120, 'r2_w': 200, 'r2_h': 200
        }
        
        self.roi_dialog = None
        self.cam_params_dialog = None

        # 用于控制统计值刷新频率（每秒3次）
        self.frame_count = 0
        self.stats_h = [0, 0]  # [mean, var] 红色ROI
        self.stats_s = [0, 0]
        self.stats_v = [0, 0]
        self.stats_r = [0, 0]
        self.stats_g = [0, 0]
        self.stats_b = [0, 0]
        self.stats_gray = [0, 0]
        self.stats_green = [0, 0, 0]  # [avg_h1, avg_s1, avg_v1] 绿色ROI

        # 🆕 V4.2: 红色ROI的R/G/B均值历史（最近500帧）
        self.rgb_history = {
            'r': deque(maxlen=500),
            'g': deque(maxlen=500),
            'b': deque(maxlen=500)
        }

        # 🆕 V4.4: H/S/V均值历史
        self.hsv_history = {
            'h': deque(maxlen=500),
            's': deque(maxlen=500),
            'v': deque(maxlen=500)
        }

        # 🆕 V4.4: 平面热力图（照搬 sightac_sdk 热力图逻辑）
        self.heatmap_scale = 10.0

        # 鼠标拖拽绘制ROI相关
        self.dragging = False
        self.drag_start = None
        self.drag_current = None

        # --- 🆕 V4: 自动重连状态 ---
        self._fail_count = 0           # 连续读取失败帧数
        self._reconnecting = False     # 是否正在重连中
        self._retry_countdown = 0      # 重试间隔倒计时(帧)
        self._camera_connected = self.cap.isOpened()  # 当前连接状态

        # --- 构建 UI 布局 ---
        self.image_label = QLabel(self)
        self.image_label.setMinimumSize(400, 300)
        # 允许标签在水平和垂直方向上自由伸展
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)
        # 🆕 V4: 让 image_label 能接收键盘事件
        self.image_label.setFocusPolicy(Qt.StrongFocus)

        self.btn_port = QPushButton("设置相机端口")
        self.btn_roi = QPushButton("设置 ROI 参数")
        self.btn_cam = QPushButton("相机参数修改")
        # 🆕 V4: 重新连接按钮
        self.btn_reconnect = QPushButton("重新连接")
        self.btn_reconnect.setToolTip("重新连接摄像头（快捷键: R）")
        self._update_reconnect_button_style()

        self.btn_port.clicked.connect(self.change_port)
        self.btn_roi.clicked.connect(self.open_roi_dialog)
        self.btn_cam.clicked.connect(self.open_cam_params_dialog)
        self.btn_reconnect.clicked.connect(self.reconnect_camera)

        # 🆕 V4.4: 校准基线 + 热力图设置按钮
        self.btn_calibrate = QPushButton("校准基线")
        self.btn_calibrate.setToolTip("重新校准H通道基线（请确保传感器无触摸）")
        self.btn_calibrate.clicked.connect(lambda: self.calibrate_baseline())

        self.btn_heatmap_settings = QPushButton("热力图设置")
        self.btn_heatmap_settings.setToolTip("设置热力图缩放/Mask/Hue阈值/标定系数")
        self.btn_heatmap_settings.clicked.connect(self.open_heatmap_settings)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_port)
        btn_layout.addWidget(self.btn_roi)
        btn_layout.addWidget(self.btn_cam)
        btn_layout.addWidget(self.btn_reconnect)
        btn_layout.addSpacing(15)
        btn_layout.addWidget(self.btn_calibrate)
        btn_layout.addWidget(self.btn_heatmap_settings)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

        # --- 设置定时器刷新画面 ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ========== 🆕 V4: 摄像头重连 ==========

    def _setup_camera_defaults(self):
        """摄像头打开后默认关闭自动曝光和自动白平衡，保持手动控制"""
        if self.cap is not None and self.cap.isOpened():
            try:
                if IS_LINUX:
                    # V4L2: AUTO_EXPOSURE 0=自动 1=手动（与Windows相反）
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                else:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
                # 关闭 AWB 之前先保存当前色温值，避免被相机固件复位覆盖
                saved_wb = self.cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                # 部分摄像头关闭 AWB 后需锁定色温才生效，用之前保存的值恢复
                if saved_wb > 0:
                    self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, saved_wb)
            except Exception:
                pass  # 部分驱动不支持这些属性，忽略即可

    def reconnect_camera(self):
        """释放当前摄像头并重新连接，可手动/自动/快捷键触发"""
        # 避免重复重连
        self._reconnecting = True
        self._fail_count = 0
        self._retry_countdown = 0

        print(f"[V4] 正在重连摄像头端口 {self.cam_port} ...")

        # 释放旧的
        if self.cap is not None:
            self.cap.release()

        # 尝试重新打开
        self.cap = open_camera(self.cam_port)
        if self.cap.isOpened():
            self._camera_connected = True
            self._reconnecting = False
            self._setup_camera_defaults()
            print(f"[V4] ✅ 重连成功！摄像头端口 {self.cam_port} 已恢复。")
        else:
            self._camera_connected = False
            self._reconnecting = True
            self._retry_countdown = RETRY_INTERVAL_FRAMES
            print(f"[V4] ❌ 重连失败，将在 ~2 秒后自动重试...")

        self._update_reconnect_button_style()

    def _update_reconnect_button_style(self):
        """根据连接状态更新按钮样式和文字"""
        if self._reconnecting:
            self.btn_reconnect.setText("重连中...")
            self.btn_reconnect.setStyleSheet(
                "background-color: #ff9800; color: white; font-weight: bold;"
            )
            self.btn_reconnect.setEnabled(False)
        elif self._camera_connected:
            self.btn_reconnect.setText("重新连接")
            self.btn_reconnect.setStyleSheet("")
            self.btn_reconnect.setEnabled(True)
        else:
            self.btn_reconnect.setText("重新连接 ⚠")
            self.btn_reconnect.setStyleSheet(
                "background-color: #f44336; color: white; font-weight: bold;"
            )
            self.btn_reconnect.setEnabled(True)

    def keyPressEvent(self, event):
        """🆕 V4: 按 R 键触发重连"""
        if event.key() == Qt.Key_R:
            # 不区分大小写
            self.reconnect_camera()
        super().keyPressEvent(event)

    # ========== 🆕 V4.4.1: 复刻 sightac 热力图链路 ==========

    def calibrate_baseline(self, num_frames=5):
        """复刻 touch_sensor.py:516 — 连拍 num_frames 帧取H通道均值作基线"""
        if not self.cap.isOpened():
            print("[V4.4.1] 相机未打开，无法校准基线")
            return
        print(f"[V4.4.1] 正在校准H通道基线（连拍{num_frames}帧，请确保传感器无触摸）...")
        base_sum = None
        count = 0
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
            rx1, rx2, ry1, ry2 = self.sensor_roi
            frame = frame[ry1:ry2, rx1:rx2]
            frame = cv2.resize(frame, (640, 480))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h_channel = hsv[:, :, 0].astype(np.float32)
            if base_sum is None:
                base_sum = np.zeros_like(h_channel)
            base_sum += h_channel
            count += 1
        if count > 0:
            self.hue_baseline = base_sum / count
            print(f"[V4.4.1] ✅ 基线校准完成 ({count}帧)，尺寸={self.hue_baseline.shape}")
        else:
            print("[V4.4.1] ❌ 基线校准失败：无有效帧")

    def compute_fz_matrix(self, frame):
        """复刻 touch_sensor.py 完整链路:
        H提取 → 差分 → mask → 阈值过滤 → 力标定 → 返回 fz_matrix"""
        if self.hue_baseline is None:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0].astype(np.float32)
        if h_channel.shape != self.hue_baseline.shape:
            h_channel = cv2.resize(h_channel,
                                   (self.hue_baseline.shape[1], self.hue_baseline.shape[0]))
        hue_matrix = h_channel - self.hue_baseline
        mask2 = (frame[:, :, 0] > self.mask_b_threshold) & (frame[:, :, 2] > self.mask_r_threshold)
        if mask2.shape != hue_matrix.shape:
            mask2 = cv2.resize(mask2.astype(np.uint8),
                               (hue_matrix.shape[1], hue_matrix.shape[0])).astype(bool)
        hue_matrix[mask2] = 0.0
        filtered = hue_matrix.astype(np.float32, copy=True)
        filtered[np.abs(filtered) < self.hue_threshold] = 0.0
        return filtered * self.fz_p1

    # ========== 原有功能 ==========

    def change_port(self):
        port, ok = QInputDialog.getInt(self, "设置相机", f"当前端口为 {self.cam_port}\n请输入新的相机端口号:", self.cam_port, 0, 10, 1)
        if ok:
            print(f"[*] 尝试连接新端口: {port} ...")
            temp_cap = open_camera(port)
            if temp_cap.isOpened():
                self.cap.release()
                self.cap = temp_cap
                self.cam_port = port
                self._camera_connected = True
                self._setup_camera_defaults()
                self._fail_count = 0
                self._reconnecting = False
                self._update_reconnect_button_style()
                QMessageBox.information(self, "成功", f"成功切换到相机端口 {port}！")
            else:
                temp_cap.release()
                QMessageBox.critical(self, "错误", f"无法打开端口 {port}！\n原相机画面继续保留。")

    def open_roi_dialog(self):
        if self.roi_dialog is None:
            self.roi_dialog = ROIDialog(self)
        self.roi_dialog.show() 

    def open_cam_params_dialog(self):
        if self.cam_params_dialog is None:
            self.cam_params_dialog = CameraParamsDialog(self)
        self.cam_params_dialog.show()

    def open_heatmap_settings(self):
        """打开热力图参数设置弹窗"""
        dialog = HeatmapSettingsDialog(self)
        dialog.exec_()

    def update_frame(self):
        # --- 🆕 V4: 自动重连逻辑 ---
        if self._reconnecting:
            # 倒计时中，等待下一次重试
            if self._retry_countdown > 0:
                self._retry_countdown -= 1
            else:
                # 倒计时到，尝试重连
                self._do_retry_reconnect()

        # 🆕 V4.4: 动态 cell 尺寸 — 计算格子像素
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        if label_w > 0 and label_h > 0:
            cell_w, cell_h = label_w // 4, label_h // 3
        else:
            cell_w, cell_h = 640, 480  # fallback

        # 读取帧
        ret = False
        frame = None
        if not self._reconnecting and self.cap.isOpened():
            ret, frame = self.cap.read()

        if not ret:
            # 🆕 V4: 累计失败帧数，达到阈值触发自动重连
            self._fail_count += 1
            if self._fail_count >= AUTO_RECONNECT_THRESHOLD and not self._reconnecting:
                print(f"[V4] 连续 {self._fail_count} 帧读取失败，触发自动重连...")
                self.reconnect_camera()
                # reconnect_camera 可能立即成功或进入重试模式

            # 显示状态提示
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            if self._reconnecting:
                msg = "Reconnecting..."
                color = (0, 200, 255)
            elif self._camera_connected:
                msg = "Loading..."
                color = (0, 165, 255)
            else:
                msg = "Camera Disconnected\nPress R or click button to reconnect"
                color = (0, 0, 255)
            # 多行文字分行绘制
            lines = msg.split('\n')
            y_start = 200
            for i, line in enumerate(lines):
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                x_pos = (640 - text_size[0]) // 2
                cv2.putText(frame, line, (x_pos, y_start + i * 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            # 读取成功，重置失败计数
            self._fail_count = 0
            # 🆕 V4.4: 裁切传感器ROI（复刻 touch_sensor.py:702，不翻转）
            rx1, rx2, ry1, ry2 = self.sensor_roi
            frame = frame[ry1:ry2, rx1:rx2]
            frame = cv2.resize(frame, (640, 480))
            # 🆕 V4.4: 热力图数据源 — 复刻 sightac 完整链路
            self._fz_matrix = self.compute_fz_matrix(frame)

        p = self.roi_params
        x1 = max(0, min(p['r1_x'], 639))
        y1 = max(0, min(p['r1_y'], 479))
        x2 = max(x1 + 1, min(x1 + p['r1_w'], 640))
        y2 = max(y1 + 1, min(y1 + p['r1_h'], 480))

        x1_2 = max(0, min(p['r2_x'], 639))
        y1_2 = max(0, min(p['r2_y'], 479))
        x2_2 = max(x1_2 + 1, min(x1_2 + p['r2_w'], 640))
        y2_2 = max(y1_2 + 1, min(y1_2 + p['r2_h'], 480))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        roi_h1, roi_s1, roi_v1 = h[y1:y2, x1:x2], s[y1:y2, x1:x2], v[y1:y2, x1:x2]
        avg_h1 = np.mean(roi_h1) if roi_h1.size > 0 else 0
        avg_s1 = np.mean(roi_s1) if roi_s1.size > 0 else 0
        avg_v1 = np.mean(roi_v1) if roi_v1.size > 0 else 0

        roi_h2, roi_s2, roi_v2 = h[y1_2:y2_2, x1_2:x2_2], s[y1_2:y2_2, x1_2:x2_2], v[y1_2:y2_2, x1_2:x2_2]
        avg_h2 = np.mean(roi_h2) if roi_h2.size > 0 else 0
        avg_s2 = np.mean(roi_s2) if roi_s2.size > 0 else 0
        avg_v2 = np.mean(roi_v2) if roi_v2.size > 0 else 0

        h_show = cv2.cvtColor(h, cv2.COLOR_GRAY2BGR)
        s_show = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
        v_show = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

        # --- 分离 BGR 通道 ---
        b_ch, g_ch, r_ch = cv2.split(frame)
        r_show = cv2.cvtColor(r_ch, cv2.COLOR_GRAY2BGR)
        g_show_c = cv2.cvtColor(g_ch, cv2.COLOR_GRAY2BGR)
        b_show = cv2.cvtColor(b_ch, cv2.COLOR_GRAY2BGR)

        # 红色ROI区域 — R/G/B 通道
        roi_r2 = r_ch[y1_2:y2_2, x1_2:x2_2]
        roi_g2 = g_ch[y1_2:y2_2, x1_2:x2_2]
        roi_b2 = b_ch[y1_2:y2_2, x1_2:x2_2]
        roi_bgr2 = frame[y1_2:y2_2, x1_2:x2_2]

        # 🆕 V4.2: 记录红色ROI的R/G/B均值到历史缓冲区（最近500帧）
        avg_r2 = np.mean(roi_r2) if roi_r2.size > 0 else 0
        avg_g2 = np.mean(roi_g2) if roi_g2.size > 0 else 0
        avg_b2 = np.mean(roi_b2) if roi_b2.size > 0 else 0
        self.rgb_history['r'].append(avg_r2)
        self.rgb_history['g'].append(avg_g2)
        self.rgb_history['b'].append(avg_b2)

        # 🆕 V4.4: 记录H/S/V均值到历史缓冲区
        self.hsv_history['h'].append(avg_h2)
        self.hsv_history['s'].append(avg_s2)
        self.hsv_history['v'].append(avg_v2)

        # 每约333ms（10帧）更新一次统计值（每秒3次）
        self.frame_count += 1
        if self.frame_count >= 10:
            self.frame_count = 0
            self.stats_h = [np.mean(roi_h2) if roi_h2.size > 0 else 0, np.var(roi_h2) if roi_h2.size > 0 else 0]
            self.stats_s = [np.mean(roi_s2) if roi_s2.size > 0 else 0, np.var(roi_s2) if roi_s2.size > 0 else 0]
            self.stats_v = [np.mean(roi_v2) if roi_v2.size > 0 else 0, np.var(roi_v2) if roi_v2.size > 0 else 0]
            self.stats_green = [np.mean(roi_h1) if roi_h1.size > 0 else 0,
                                np.mean(roi_s1) if roi_s1.size > 0 else 0,
                                np.mean(roi_v1) if roi_v1.size > 0 else 0]
            self.stats_r = [np.mean(roi_r2) if roi_r2.size > 0 else 0, np.var(roi_r2) if roi_r2.size > 0 else 0]
            self.stats_g = [np.mean(roi_g2) if roi_g2.size > 0 else 0, np.var(roi_g2) if roi_g2.size > 0 else 0]
            self.stats_b = [np.mean(roi_b2) if roi_b2.size > 0 else 0, np.var(roi_b2) if roi_b2.size > 0 else 0]

        # 🆕 V4.4: 缩放7个子图到 cell 尺寸，再画 overlay（字/框直接渲染不缩放）
        S = min(cell_w / 640, cell_h / 480)  # 缩放因子
        SZ = (cell_w, cell_h)
        images_cell = [
            cv2.resize(frame,     SZ, interpolation=cv2.INTER_LANCZOS4),
            cv2.resize(h_show,    SZ, interpolation=cv2.INTER_LANCZOS4),
            cv2.resize(s_show,    SZ, interpolation=cv2.INTER_LANCZOS4),
            cv2.resize(v_show,    SZ, interpolation=cv2.INTER_LANCZOS4),
            cv2.resize(r_show,    SZ, interpolation=cv2.INTER_LANCZOS4),
            cv2.resize(g_show_c,  SZ, interpolation=cv2.INTER_LANCZOS4),
            cv2.resize(b_show,    SZ, interpolation=cv2.INTER_LANCZOS4),
        ]
        # 名字对应: 0=原图, 1=H, 2=S, 3=V, 4=R, 5=G, 6=B
        titles = ["Original", "Hue (H)", "Saturation (S)", "Value (V)",
                  "Red (R)", "Green (G)", "Blue (B)"]
        roi2_label_indices = [1, 2, 3]  # H/S/V 显示 ROI1 标签

        # 比例化 ROI 坐标
        def _sx(v): return int(v * cell_w / 640)
        def _sy(v): return int(v * cell_h / 480)
        cx1, cy1, cx2, cy2 = _sx(x1), _sy(y1), _sx(x2), _sy(y2)
        cx1_2, cy1_2, cx2_2, cy2_2 = _sx(x1_2), _sy(y1_2), _sx(x2_2), _sy(y2_2)

        # 比例化字体
        font_title   = max(0.5, 1.2 * S)
        font_info    = max(0.4, 0.7 * S)
        font_small   = max(0.35, 0.6 * S)
        thick        = max(1, int(1.5 * S))
        thick_roi    = max(1, int(2 * S))

        text_color = (0, 255, 255)
        red_color  = (0, 0, 255)
        c_roi1, c_roi2 = (0, 255, 0), (0, 0, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX

        for i, img in enumerate(images_cell):
            # ROI 框
            cv2.rectangle(img, (cx1, cy1), (cx2, cy2), c_roi1, thick_roi)
            cv2.rectangle(img, (cx1_2, cy1_2), (cx2_2, cy2_2), c_roi2, thick_roi)
            # 标题
            cv2.putText(img, titles[i], (_sx(20), _sy(40)), font, font_title,
                        text_color, thick, cv2.LINE_AA)
            # ROI1 标签 (仅 H/S/V)
            if i in roi2_label_indices:
                gi = i - 1  # green stats index
                cv2.putText(img, f"Avg {titles[i][0]}: {self.stats_green[gi]:.1f}",
                            (cx1, max(_sy(15), cy1 - _sy(10))),
                            font, font_small, c_roi1, thick, cv2.LINE_AA)
            # ROI2 均值+方差 (仅 H/S/V/R/G/B, i>=1)
            if i >= 1:
                si = i - 1  # stats index
                st = [self.stats_h, self.stats_s, self.stats_v,
                      self.stats_r, self.stats_g, self.stats_b][si]
                cv2.putText(img, f"Mean: {st[0]:.1f}", (_sx(20), _sy(75)),
                            font, font_info, red_color, thick, cv2.LINE_AA)
                cv2.putText(img, f"Var:  {st[1]:.1f}", (_sx(20), _sy(100)),
                            font, font_info, red_color, thick, cv2.LINE_AA)

        # 原图(索引0)额外显示 ROI1 数值
        cv2.putText(images_cell[0],
                    f"R1: {self.stats_green[0]:.0f},{self.stats_green[1]:.0f},{self.stats_green[2]:.0f}",
                    (cx1, max(_sy(15), cy1 - _sy(10))),
                    font, font_small, c_roi1, thick, cv2.LINE_AA)

        # 右键拖拽预览矩形（cell 空间）
        if self.dragging and self.drag_start and self.drag_current:
            dx1, dy1, idx = self.drag_start
            dx2, dy2, _ = self.drag_current
            rx1, ry1 = _sx(min(dx1, dx2)), _sy(min(dy1, dy2))
            rx2, ry2 = _sx(max(dx1, dx2)), _sy(max(dy1, dy2))
            if 0 <= idx <= 6 and rx2 > rx1 and ry2 > ry1:
                cv2.rectangle(images_cell[idx], (rx1, ry1), (rx2, ry2),
                              (0, 255, 255), thick_roi)

        # 组装
        frame_d, h_show_d, s_show_d, v_show_d, r_show_d, g_show_d, b_show_d = images_cell
        heatmap_cell = self.draw_heatmap(self._fz_matrix, cell_w, cell_h)
        # 🆕 V4.4: 热力图ROI框用动态反色 — 每框独立采样背景，确保JET色图全区域可见
        hm_roi1_color = self._get_contrast_color(heatmap_cell, cx1, cy1, cx2, cy2)
        hm_roi2_color = self._get_contrast_color(heatmap_cell, cx1_2, cy1_2, cx2_2, cy2_2)
        cv2.rectangle(heatmap_cell, (cx1, cy1), (cx2, cy2), hm_roi1_color, thick_roi)
        cv2.rectangle(heatmap_cell, (cx1_2, cy1_2), (cx2_2, cy2_2), hm_roi2_color, thick_roi)
        top_row    = np.hstack((frame_d, h_show_d, s_show_d, v_show_d))
        bottom_row = np.hstack((r_show_d, g_show_d, b_show_d, heatmap_cell))
        rgb_curve  = self.draw_rgb_curve(cell_w, cell_h)

        # 曲线文字 overlay
        curve_th = max(1, int(1.5 * S))
        cv2.putText(rgb_curve, "RGB + Hue Time Curve",
                    (_sx(20), _sy(40)), font, font_title,
                    text_color, curve_th, cv2.LINE_AA)
        cv2.putText(rgb_curve,
                    f"R:{self.stats_r[0]:.1f} G:{self.stats_g[0]:.1f} B:{self.stats_b[0]:.1f} H:{self.stats_h[0]:.1f}",
                    (_sx(20), _sy(75)), font, font_info,
                    red_color, curve_th, cv2.LINE_AA)

        combined_img = np.vstack((top_row, bottom_row, rgb_curve))
        rgb_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        q_img = QImage(rgb_img.data, w, h, w * ch, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def _do_retry_reconnect(self):
        """🆕 V4: 重连重试（由 update_frame 中的倒计时触发）"""
        print(f"[V4] 重试连接摄像头端口 {self.cam_port} ...")
        if self.cap is not None:
            self.cap.release()

        self.cap = open_camera(self.cam_port)
        if self.cap.isOpened():
            self._camera_connected = True
            self._reconnecting = False
            self._setup_camera_defaults()
            self._fail_count = 0
            self._retry_countdown = 0
            print(f"[V4] ✅ 重连成功！摄像头端口 {self.cam_port} 已恢复。")
        else:
            # 失败，设定下一次重试倒计时
            self._retry_countdown = RETRY_INTERVAL_FRAMES
            print(f"[V4] ❌ 重连失败，将在 ~2 秒后再次重试...")

        self._update_reconnect_button_style()

    def draw_rgb_curve(self, cell_w, cell_h):
        """🆕 V4.4: 动态 cell 尺寸 — 比例化坐标"""
        CW = cell_w * 4  # 通栏宽
        CH = cell_h
        S = min(cell_w / 640, cell_h / 480)  # 缩放因子

        canvas = np.zeros((CH, CW, 3), dtype=np.uint8)
        axis_color = (200, 200, 200)

        # 比例坐标
        margin_l = int(55 * S)
        margin_r = int(40 * S)
        margin_t = int(20 * S)
        margin_b = int(40 * S)
        legend_x = CW - margin_r
        plt_x0, plt_y0 = margin_l, CH - margin_b
        plt_w = CW - margin_l - int(60 * S)
        plt_h = CH - margin_t - margin_b

        # Y轴
        cv2.line(canvas, (plt_x0, plt_y0), (plt_x0, margin_t), axis_color, max(1, int(2 * S)))
        cv2.line(canvas, (plt_x0, plt_y0), (CW, plt_y0), axis_color, max(1, int(2 * S)))

        for val_255, y_abs in [(0, plt_y0), (128, plt_y0 - int(128 * plt_h / 255)), (255, margin_t)]:
            fs = max(0.4, 0.8 * S)
            cv2.putText(canvas, str(val_255), (int(3 * S), y_abs + int(8 * S)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, axis_color, max(1, int(2 * S)), cv2.LINE_AA)
            tick_x = plt_x0 - int(5 * S)
            cv2.line(canvas, (tick_x, y_abs), (plt_x0, y_abs), axis_color, max(1, int(2 * S)))

        n = len(self.rgb_history['r'])
        text_th = max(1, int(1.5 * S))  # 文字厚度随 cell 缩放
        if n < 2:
            cv2.putText(canvas, "Collecting...", (int(20 * S), int(40 * S)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0 * S, (200, 200, 200), text_th)
            return canvas

        # B/G/R 曲线
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for key, color in zip(['b', 'g', 'r'], colors):
            mean_vals = self.rgb_history[key]
            pts = []
            for i, val in enumerate(mean_vals):
                x = plt_x0 + int(i * plt_w / max(n - 1, 1))
                y = plt_y0 - int(val * plt_h / 255)
                pts.append([x, y])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], False, color, max(1, int(3 * S)))

        # H 曲线
        mean_vals_h = self.hsv_history['h']
        pts_h = []
        for i, val in enumerate(mean_vals_h):
            x = plt_x0 + int(i * plt_w / max(n - 1, 1))
            y = plt_y0 - int(val * plt_h / 255)
            pts_h.append([x, y])
        pts_h = np.array(pts_h, np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts_h], False, (200, 200, 200), max(1, int(3 * S)))

        # 图例
        fs_l = max(0.4, 0.9 * S)
        for i, (label, color) in enumerate([("B", (255, 0, 0)), ("G", (0, 255, 0)),
                                              ("R", (0, 0, 255)), ("H", (200, 200, 200))]):
            cv2.putText(canvas, label, (legend_x, int((40 + 30 * i) * S)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_l, color, text_th)

        return canvas

    def _get_contrast_color(self, img, x1, y1, x2, y2):
        """采样ROI边框路径上的背景像素均值，返回高对比度反色。
        性能：每框采样 ~2*(w+h) 像素做 np.mean，微秒级，不影响30fps。"""
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return (0, 255, 0)  # fallback
        # 采样四条边（取2px宽度的边框区域，容错性好）
        top    = img[y1:y1+2, x1:x2]
        bottom = img[y2-2:y2, x1:x2]
        left   = img[y1:y2, x1:x1+2]
        right  = img[y1:y2, x2-2:x2]
        border = np.concatenate([top.reshape(-1, 3), bottom.reshape(-1, 3),
                                 left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)
        if border.size == 0:
            return (0, 255, 0)
        mean_bgr = np.mean(border, axis=0)
        # 反色 = 高对比度
        return (255 - int(mean_bgr[0]), 255 - int(mean_bgr[1]), 255 - int(mean_bgr[2]))

    def draw_heatmap(self, fz_matrix, cell_w, cell_h):
        """🆕 V4.4.1: 平面热力图 — sightac 核心逻辑 + cell 尺寸"""
        if fz_matrix is None or fz_matrix.size == 0:
            return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

        preview_gray = np.clip(fz_matrix * self.heatmap_scale, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(preview_gray, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (cell_w, cell_h), interpolation=cv2.INTER_LANCZOS4)

        S = min(cell_w / 640, cell_h / 480)
        fs = max(0.4, 0.7 * S)
        cv2.putText(heatmap, "Planar Heatmap (JET)", (int(15 * S), int(30 * S)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, int(2 * S)), cv2.LINE_AA)
        cv2.putText(heatmap, f"Scale: {self.heatmap_scale:.1f}", (int(15 * S), int(60 * S)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (200, 200, 200), max(1, int(1 * S)), cv2.LINE_AA)

        return heatmap

    def get_sub_image_coords(self, pos, label_rect):
        """将鼠标位置映射到对应子图的坐标和子图索引（适配动态 cell 尺寸）
        返回坐标归一化到 640×480 空间，与 ROI 参数一致。"""
        lw, lh = label_rect.width(), label_rect.height()
        cell_w, cell_h = lw // 4, lh // 3

        # 鼠标在合并图像中的像素坐标
        img_x = int(pos.x() * lw / max(lw, 1))
        img_y = int(pos.y() * lh / max(lh, 1))

        col = min(img_x // max(cell_w, 1), 3)
        row = min(img_y // max(cell_h, 1), 2)

        sub_x_in_cell = img_x - col * cell_w
        sub_y_in_cell = img_y - row * cell_h

        # 转回 640×480 空间（ROI 参数使用的坐标系）
        sub_x = int(sub_x_in_cell * 640 / max(cell_w, 1))
        sub_y = int(sub_y_in_cell * 480 / max(cell_h, 1))

        idx_map = [[0, 1, 2, 3], [4, 5, 6, 8], [7, 7, 7, 7]]
        return sub_x, sub_y, idx_map[row][col]

    def mousePressEvent(self, event):
        # 只处理右键点击
        if event.button() == Qt.RightButton:
            # 获取点击位置相对于 image_label 的坐标
            pos = self.image_label.mapFrom(self, event.pos())
            label_rect = self.image_label.rect()

            # 检查是否在 image_label 内
            if not label_rect.contains(pos):
                return

            # 获取子图坐标和索引
            sub_x, sub_y, sub_idx = self.get_sub_image_coords(pos, label_rect)

            # 空白占位格(-1)和通栏曲线(7)不是图像空间，禁止拖拽画 ROI
            if sub_idx < 0 or sub_idx == 7:
                return

            self.dragging = True
            self.drag_start = (sub_x, sub_y, sub_idx)
            self.drag_current = (sub_x, sub_y, sub_idx)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging:
            pos = self.image_label.mapFrom(self, event.pos())
            label_rect = self.image_label.rect()

            # 获取子图坐标和索引
            sub_x, sub_y, sub_idx = self.get_sub_image_coords(pos, label_rect)

            # 确保在同一个子图内拖拽
            if sub_idx == self.drag_start[2]:
                # 限制在子图范围内 (640x480)
                sub_x = max(0, min(sub_x, 639))
                sub_y = max(0, min(sub_y, 479))
                self.drag_current = (sub_x, sub_y, sub_idx)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self.dragging:
            self.dragging = False

            if self.drag_start and self.drag_current:
                x1, y1, idx1 = self.drag_start
                x2, y2, idx2 = self.drag_current

                # 确保在同一个子图内
                if idx1 == idx2:
                    # 计算矩形（确保 x1 < x2, y1 < y2）
                    roi_x = min(x1, x2)
                    roi_y = min(y1, y2)
                    roi_w = abs(x2 - x1)
                    roi_h = abs(y2 - y1)

                    # 矩形需大于5x5像素
                    if roi_w > 5 and roi_h > 5:
                        # 限制在640x480范围内
                        roi_x = max(0, min(roi_x, 639))
                        roi_y = max(0, min(roi_y, 479))
                        max_x = min(roi_x + roi_w, 640)
                        max_y = min(roi_y + roi_h, 480)
                        roi_w = max_x - roi_x
                        roi_h = max_y - roi_y

                        # 更新红色ROI参数
                        self.roi_params['r2_x'] = roi_x
                        self.roi_params['r2_y'] = roi_y
                        self.roi_params['r2_w'] = roi_w
                        self.roi_params['r2_h'] = roi_h

                        # 同步更新ROI对话框中的滑块
                        if self.roi_dialog is not None:
                            self.roi_dialog.r2_x.setValue(roi_x)
                            self.roi_dialog.r2_y.setValue(roi_y)
                            self.roi_dialog.r2_w.setValue(roi_w)
                            self.roi_dialog.r2_h.setValue(roi_h)

            self.drag_start = None
            self.drag_current = None

        super().mouseReleaseEvent(event)

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

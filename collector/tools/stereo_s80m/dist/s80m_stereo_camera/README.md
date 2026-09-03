# S80M 双目 RGB 相机采集模块

基于 Fays VI Kit SDK 的双目 RGB + IMU 帧流采集模块，**自包含、可直接发送到其他机器运行**。
左右目各 1280×800@25fps JPEG 帧流 + IMU 六轴数据，通过标准输入输出管道（`--pipe`）对外传输，
适合作为采集后端被主程序/训练管线调用。

## 目录结构

```
s80m_stereo_camera/
├── run.sh                        # 一键启动（自带运行时库环境）
├── read_stereo_rgb.py            # 主程序（SDK 采集 + --pipe 帧流协议）
├── export_calibration.py         # 标定导出（设备 ROM → 平台 JSON/YAML）
├── pipe_consumer.py              # 后端消费示例（解析 --pipe 帧流协议）
├── requirements.txt              # Python 依赖
├── lib/fays_atrak/x86_64/Release/
│   └── libfays_vikit.so          # 相机 SDK（3.9.0）
├── config/
│   └── fays_vikit.yaml           # 设备节点配置（见下）
├── runtime/
│   ├── lib/                      # libft602.so（FTDI 桥接库）
│   ├── opencv4.2/                # OpenCV 4.2 运行库 + tbb/webp ABI shims
│   └── udev/
│       └── 51-ftd3xx.rules       # USB 设备权限规则（需 root 安装）
```

## 环境要求

- **Linux x86_64**（Ubuntu 20.04/22.04/24.04），Python ≥ 3.8
- 相机通过 USB3 连接（FT602 芯片）

依赖分三层：pip 包（numpy + opencv-python）、包内运行时（SDK / OpenCV 4.2 /
tbb / webp / tiff 已自带，见 runtime/）、以及下面一行 **apt 系统库**
（OpenCV 4.2 还依赖 libGL/Qt5/libjpeg/libpng，pip 不会装这些）：

```bash
sudo apt install -y libgl1 libjpeg-turbo8 libpng16-16 qtbase5 libqt5opengl5
```

## 安装

```bash
# 1) 系统库 + Python 依赖
sudo apt install -y libgl1 libjpeg-turbo8 libpng16-16 qtbase5 libqt5opengl5
pip install -r requirements.txt

# 2) 安装 USB 设备权限规则（一次性，需 sudo）
sudo cp runtime/udev/51-ftd3xx.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

# 3) 无需手工确认设备节点：启动时自动按设备名（FTDI Superspeed
#    Video Bridge）与 USB 接口号解析端口（1.0=双目 / 1.2=IMU）并
#    重写临时 yaml。仅当自动解析失败（打印 [Ports] 解析失败）时才
#    需要手工改 config/fays_vikit.yaml 的 stereo_dev_port/imu_dev_port。
```

## 用法

```bash
# 显示模式（需图形环境，Q/Esc 退出，S 截图）
./run.sh

# 帧流模式 → 实时管道（打印信息走 stderr，二进制帧走 stdout）
./run.sh --pipe - | python3 pipe_consumer.py -

# 帧流模式 → 写入文件
./run.sh --pipe out.bin
python3 pipe_consumer.py out.bin --stats

# 指定 IMU 采样率（默认 200 Hz，0 = 关闭采集）
IMU_HZ=100 ./run.sh --pipe -

# JPEG 质量（默认 85，调低可提帧率；左右目编码已并行化，多核上零副作用）
JPEG_Q=75 ./run.sh --pipe -
```

> **必须通过 SIGTERM 正常停止**（Ctrl+C 即可），SDK 需要走销毁流程释放 FT602；
> 严禁 SIGKILL，否则 USB 芯片可能进入需断电才能恢复的挂死状态。

## 标定导出（采集端职责，新增设备必须执行一次）

出厂标定烧在设备 ROM 里，通过 `FAYS_VIK_GetCalibrationParam` 读取
（SDK 示例只取帧没取标定，所以平台会话的 head_stereo.json 内参全是 0）：

```bash
# 导出到平台会话目录 → <session>/calibration/head_stereo.json（覆盖占位文件）
#   + <session>/calibration/s80m_dump_calib.yaml（SDK DumpCalib 原始备份）
./export_calibration.py --session data/recordings/222/222_000002

# 只导出设备级默认标定（供三角化回退链）→ config/s80m_stereo_calibration.json
./export_calibration.py

# 自定义输出路径
./export_calibration.py --output /tmp/calib.json
```

导出内容（平台 head_stereo.json 格式 + 三角化扩展字段）：
内参 ×2 缩放到录制分辨率 1280×800、畸变系数 + 模型（equidistant/radtan）、
`T_cn_cnm1` 旋转/平移（基线）、`timeshift_cam_imu`、IMU 噪声参数。
末尾会打印 `三角化可用: ✓` 确认平台 worker 可直接消费。

> 注意：本脚本同样**绝不调用 GetStereoFrames**（3.9.1 段错误风险），
> 只调 Create / GetCalibrationParam / DumpCalib / Destroy。

## 帧流协议（--pipe）

全部字段**大端**，单帧格式：

```
[4B  left_jpg_len] [8B left_ts_ns] [left_jpg_len 字节 JPEG]
[4B  right_jpg_len] [8B right_ts_ns] [right_jpg_len 字节 JPEG]
[4B  imu_count] 然后 imu_count 个样本，每个样本:
    [8B ts_ns] [8B gx] [8B gy] [8B gz] [8B ax] [8B ay] [8B az]   # 双精度
```

- 左右目 JPEG：BGR 编码（与 OpenCV 默认一致），1280×800
- 时间戳单位：纳秒（单调时钟）
- IMU 单位：gyro rad/s，acc m/s²（六轴 + 时间戳 = 56B/样本）
- Python `struct` 格式：`>IQ`（左右目头）、`>I`（IMU 计数）、`>Q6d`（IMU 样本）
- 帧流消费示例见 `pipe_consumer.py`

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `Failed to open device` | 设备节点不对，改 `config/fays_vikit.yaml` |
| `Permission denied` | 未安装 udev 规则，或用户不在 dialout 组 |
| 帧率低于 25fps | USB3 线/口问题；确认 `--pipe` 下游及时消费不阻塞 |
| 停止后相机无响应 | 曾用 SIGKILL 强杀，断电（拔 USB）恢复 |

## 版本说明

- **SDK 必须保持 3.9.0**。3.9.1 已知缺陷：RGB 帧获取失败后调用
  `GetStereoFrames` 会段错误崩溃，本模块不会升级。
- 本包内置的 OpenCV 为 SDK 配套的 4.2 版本（`.so.406`），与主程序环境隔离，互不影响。

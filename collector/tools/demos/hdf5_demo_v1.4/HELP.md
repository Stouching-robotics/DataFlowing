# HDF5 Data Viewer — 使用说明

单文件 hdf5 回放查看器。一个窗口同时显示：

```
┌─────────────────────┬─────────────────────┐
│  左上               │  右上               │
│  相机视频           │  渲染好的视频       │
│  (左 | 右)          │  (h5 内已渲染)      │
├─────────────────────┼─────────────────────┤
│  左下               │  右下               │
│  仿生手掌           │  触觉热力图         │
│  (触觉矩阵映射)     │  (16x16 矩阵)       │
├─────────────────────┴─────────────────────┤
│  底部: IMU 波形 (加速度 | 角速度)         │
└───────────────────────────────────────────┘
```

**传感器自适应布局**：触觉面板（仿生手掌 / 热力图）和 IMU 面板**只有当文件里
确实含有该传感器数据时才显示**（见下方"传感器检测"）。没有数据的面板会被
隐藏，其余面板自动扩大填满窗口——中间不会空出一块。

---

## 1. 打开 hdf5 文件

### 方式 A. 命令行（推荐）

```bash
python hdf5_demo.py /your/path/data_001.h5
```

### 方式 B. Open h5 按钮

不带参数启动后，点击左下角的 **Open h5** 按钮，在对话框中选择文件。

### 方式 C. 播放控制

使用底部控制栏：

- **Play / Pause** 播放 / 暂停
- **|< / >|** 单帧步进（上一帧 / 下一帧）
- **滑块** 拖动跳转到任意帧

---

## 2. 数据格式

数据从 `episode_*` 组中读取（`*` 可以是任意名称）：

```
<episode>/observation/images/stereo_left   (N, 800, 1280, 3) uint8 RGB  [必需]
<episode>/observation/images/stereo_right  (N, 800, 1280, 3) uint8 RGB  [必需]
<episode>/observation/tactile/left         (N, 16, 16) float            [可选]
<episode>/observation/tactile/right        (N, 16, 16) float            [可选]
<episode>/observation/imu                  (M, 6) float                 [可选]
                                           列: [acc_x acc_y acc_z gyr_x gyr_y gyr_z]
<episode>/observation/imu_frame_index      (M,) int32  每条 imu 数据对应的帧号 [可选]
```

- **双目视频为必需**——缺失时打开会报错
- 帧数 N 可以为任意值；查看器以视频帧为准对齐其他数据
- `videos/hand_skeleton` 数据集（MP4 字节）用于右上角渲染视频面板——见第 4 节

## 3. 传感器检测

查看器根据 h5 文件里实际包含的数据决定显示什么：

| 文件中的数据 | 仿生手掌 + 热力图 | IMU 面板 |
|---|---|---|
| tactile 存在且有非零样本 | 显示 | - |
| tactile 缺失，或全零（传感器从未被接触） | 隐藏 | - |
| imu + imu_frame_index 存在 | - | 显示 |
| imu 缺失 | - | 隐藏 |

触觉数据全零（例如录制过程中没有任何东西触碰传感器）会被当作"无数据"：
面板隐藏而不是显示空白。某面板被隐藏后，其余面板会扩大使用空出的空间。

## 4. 渲染视频（右上角）

右上角面板播放 **h5 文件里已经渲染好的视频**——查看器本身不做渲染。
探测顺序：

1. **优先**：`<episode>/videos/hand_skeleton` —— 以原始 MP4 字节存储的
   手部骨架渲染视频（attr `format=mp4`），自动解码播放
2. **兜底**：`images` 组内名称包含以下任一关键字的帧数组数据集：
   ```
   preview  keypoint  annot  render  overlay  visual  vis  kp  skeleton  slam
   ```

- 找到 → 视频在右上角面板播放
- 找不到 → 显示 "No rendered video found"，其余功能不受影响
- 如果你的渲染视频用了其他名字，把名字加到 `hdf5_demo.py` 顶部的
  `RENDERED_KEYWORDS` 元组里，重新运行即可

## 5. IMU 波形

底部面板绘制 IMU 六个通道（每帧一个点，取该帧的第一个 imu 样本）：

- **上半部分**：加速度 acc_x / acc_y / acc_z
- **下半部分**：角速度 gyr_x / gyr_y / gyr_z
- **白色竖线**：当前视频帧的位置
- **右侧数值**：当前帧号、该帧内的 imu 样本数（imu 采样率高于视频，
  约 19 个样本/帧），以及各通道的实时数值

IMU 以平铺样本数组存储，通过 `imu_frame_index` 把每个样本对应到视频帧；
查看器会自动重新聚合。

## 6. 常见问题

| 现象 | 处理方式 |
|---|---|
| 打开报错: "no stereo_left/stereo_right" | 文件没有双目视频数据集，检查数据格式 |
| 打开报错: "no episode_* group" | 文件结构不匹配，见第 2 节 |
| 右上角显示 "No rendered video found" | 正常——该 h5 里没有渲染视频，可忽略或见第 4 节 |
| 仿生手掌 / 热力图 / IMU 面板缺失 | 正常——文件里没有（非零的）对应传感器数据 |
| 播放卡顿 | 降低播放速率：把 `hdf5_demo.py` 里的 `timer.setInterval(40)` 改大（例如 80） |

## 7. 运行环境

- Python 3.8 或更高版本
- 安装依赖：`pip install -r requirements.txt`
- Linux：需要桌面环境（X11/Wayland）；Windows：直接运行

# stereo_2d_demo —— 双目/单目视频 2D 手部关键点检测 + 渲染 Demo

**自包含**：整个 demo 只有一个 Python 文件，不依赖仓库里任何其他模块或目录。
拿到压缩包解压后，装好依赖即可运行。

## 目录内容

```
stereo_2d_demo/
├── stereo_2d_demo.py     # 主程序（单文件：检测 + 平滑 + 绘制 + 视频输出）
├── hand_landmarker.task  # MediaPipe 手部关键点模型（约 7.5MB）
├── requirements.txt      # Python 依赖
├── run.sh                # 一键运行脚本
└── README.md             # 本说明
```

## 环境要求

- Python 3.9+（3.10/3.11 均可）
- 建议新建虚拟环境安装依赖：

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> 模型 `hand_landmarker.task` 必须和 `stereo_2d_demo.py` 放在**同一目录**。
> 若机器无法联网，用别人装好的环境直接拷贝 venv 目录也可用。

## 使用方法

### 一键脚本（自动检测 venv）

```bash
# 双目（左右目并排输出）
./run.sh left.mp4 right.mp4

# 单目
./run.sh demo.mp4

# 指定输出文件
./run.sh left.mp4 right.mp4 -o out.mp4
```

### 直接运行

```bash
# 双目（左右目并排输出，默认输出 stereo_2d_output.mp4）
python stereo_2d_demo.py left.mp4 right.mp4

# 单目
python stereo_2d_demo.py demo.mp4

# 指定输出 + 跟手调优
python stereo_2d_demo.py left.mp4 right.mp4 -o out.mp4 --freq-min 15 --beta 0.6

# 完全关闭平滑（最跟手，但抖动全部保留）
python stereo_2d_demo.py left.mp4 right.mp4 --no-smooth
```

## 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `videos` | — | 1 个视频=单目；2 个视频=左右目并排输出 |
| `-o, --output` | `stereo_2d_output.mp4` | 输出视频路径 |
| `--model` | 本目录 `hand_landmarker.task` | 模型路径 |
| `--freq-min` | `15.0` | One-Euro 滤波截止频率 Hz。**越大越跟手**（同时越抖），越小越平滑（越滞后） |
| `--beta` | `0.6` | 速度自适应系数。快速运动时滞后就调大 |
| `--no-smooth` | 关 | 关闭 One-Euro 平滑，关键点完全原始（最跟手，抖动全保留） |

**跟手性调参建议**：
- 感觉关键点"粘在上一帧、跟不上手" → 调大 `--freq-min`（如 15→30），或加 `--no-smooth`
- 感觉关键点抖动厉害 → 调小 `--freq-min`（如 5）
- 快速挥动手臂时明显滞后 → 调大 `--beta`

## 输出效果

- 每只手：**青绿色包围框** + 底衬标签 `Hand #0 / #1`
- **五指分色骨架**：拇指橙、食指绿、中指黄、无名指紫、小指蓝；掌心灰色连接线、腕部白色大圆
- 框下方**手势文本**：`open: 食指,中指`（伸开手指名）或 `fist`（握拳）
- 左上角显示摄像头名与帧进度，底部图例说明

## 常见问题

**Q1: 报错 `无法打开视频`**
视频路径不存在或编码不支持（如 HEVC）。先用 ffplay/VLC 确认能打开，或转成 H.264 再跑。

**Q2: 输出的是 mp4v（不压缩）或没有声音**
输出视频只有画面没有声音（demo 不做音频）。若机器没有 ffmpeg，会自动退回 mp4v 编码；
装了 ffmpeg 则会自动转成 H.264 压缩，体积小很多。装 ffmpeg：

```bash
# Ubuntu/Debian
sudo apt install ffmpeg
# macOS
brew install ffmpeg
```

**Q3: 检测不到手 / 双手只显示一只**
- 手离镜头太远或太小 → 拉近拍摄
- 调整置信度：编辑 `stereo_2d_demo.py` 中 `min_hand_detection_confidence=0.5` 等处
  （数字越小越容易检出，误检越多）

**Q4: 运行时很慢 / 卡顿**
检测是逐帧 CPU 推理，720p 双目约 40-60 帧/秒（含渲染）。慢机器可先缩小视频
（如 `ffmpeg -i in.mp4 -vf scale=640:400 out.mp4`）再跑。

**Q5: 关掉后重跑为什么还是老输出？**
同名输出文件会被覆盖；如需保留请用 `-o` 指定新名字。

## 免责说明

手部关键点检测结果受光照、遮挡、手部大小影响，仅供参考，请结合实际场景评估效果。

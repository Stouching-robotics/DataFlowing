# scripts/

## 定位

`scripts/` 提供两类离线 CLI：对已录制会话的离线批处理（手部关键点提取与手势标注、可视化渲染导出），以及 Windows 离线部署包打包。`process_hands.py` 在开头把仓库根目录插入 `sys.path`，从任意位置以 `python scripts/xxx.py ...` 运行即可（`pack_wheels.py` 只依赖标准库，无此 shim）。

主程序不调用这些脚本（全库检索 `scripts/` 除 `scripts/` 自身外无任何引用）。`process_hands.py` 反向依赖 `core/`（`core.hand_tracking` 与 `core.helpers`）；`pack_wheels.py` 零 core 依赖。

## 文件清单

| 文件 | 一句话作用 | 运行方式 |
|---|---|---|
| `scripts/process_hands.py` | 手部关键点提取（2D+可选 3D）、手势自动标注、可视化渲染，子命令 `kpts/label/all/show/render` | `python scripts/process_hands.py kpts <session> --mode bare` |
| `scripts/pack_wheels.py` | 生成 Windows 3.12 离线部署包（`wheels/`：依赖轮子 + Python 安装包），供客户无外网一键安装 | `python scripts/pack_wheels.py [--extras|--torch|--out]` |

## 各文件详解

### `scripts/process_hands.py`

**作用**：手部关键点提取 CLI，对已录制会话离线处理。`kpts` 子命令调 `core/hand_tracking.process_session()` 提取关键点（`glove`/`bare` 双模式）；`label` 子命令做手势自动标注；`render` 子命令把已有 2D 关键点叠加回视频导出可视化；`show` 查看已有数据摘要。它是整条离线后处理链的入口。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `_progress_bar` | `(cur, total, width=40)` | 写 stderr 的进度条 | — |
| `cmd_kpts` | `(args)` | 调 `process_session()` 提取关键点 | 成功打印帧数/耗时/fps，失败 `sys.exit(1)` |
| `cmd_label` | `(args)` | 调 `label_session()` 手势自动标注 | 同上 |
| `cmd_all` | `(args)` | 先 `cmd_kpts` 再 `cmd_label` | — |
| `cmd_show` | `(args)` | 调 `load_hand_kpts()` 打印数据摘要（帧数/维度/最多手数） | — |
| `_find_video` | `(session_path)` | 在 session 目录找第一个非 depth 的 RGB 视频（egodata 格式先查 metadata 相机列表，再扫 `videos/` 与 `chunk-0000/`） | 返回视频路径或 `""` |
| `cmd_render` | `(args)` | 关键点叠加到视频导出：ffmpeg 子进程（`libx264 -crf 23`）优先，`avc1`/`mp4v` 的 `cv2.VideoWriter` 回退 | 写 `<cam>_hand_kpts.mp4` |
| `main` | 子命令解析 | — | — |

**关键数据**：
- 子命令与参数：
  - `kpts <session> [--mode glove|bare] [--device cuda]`（默认 `glove`、`cuda`）
  - `label <session>`
  - `all <session> [--mode] [--device]`
  - `show <session>`
  - `render <session>`（无附加参数）
- 输入：会话目录（如 `data/recordings/Test005/episode_000001`）；egodata 格式视频路径经 `core.helpers.egodata_video_path` 解析。
- 输出：关键点 parquet 落盘由 `core/hand_tracking` 负责（路径口径 `core.helpers.hand_kpts_parquet_path`）；`render` 输出到 `keypoints_video_dir(session)/<相机名>_hand_kpts.mp4`。

**调用关系**：导入 `core.hand_tracking`（`process_session`、`label_session`、`load_hand_kpts`、`draw_kpts_overlay`）；`cmd_render` 内部使用 `core.helpers`（`detect_session_format`、`egodata_video_path`、`egodata_metadata_path`、`keypoints_video_dir`）。不被主程序调用。

### `scripts/pack_wheels.py`

**作用**：生成 Windows 离线部署包（`wheels/` 目录），供 `start.bat`/`start.sh` 在无外网环境一键安装。在任意有网机器（Linux 开发机即可，跨平台拉取 Windows 轮子）上运行，把产出的 `wheels/` 整个目录拷到客户项目根目录后，启动脚本检测到 `wheels/*.whl` 自动改走纯离线安装。

**类/函数**：

| 名称 | 签名要点 | 作用 | 返回/副作用 |
|---|---|---|---|
| `parse_requirements` | `(path)` | `requirements.txt` → 包规格列表（去注释/空行） | list[str] |
| `download_closure` | `(specs, out, ...)` | 拉取依赖闭包：优先 `uv pip compile --python-platform windows` 跨平台解析（正确剔除 Linux-only 依赖如 bleak 的 dbus-fast），无 uv 回退 `pip download` | 写 `<out>/*.whl` |
| `download_python_installer` | `(out)` | 下载 `python-3.12.10-amd64.exe`（阿里云/npmmirror/华为云镜像回退） | 写 `<out>/python-3.12.10-amd64.exe` |
| `main` | 参数解析 | — | — |

**关键数据**：
- 参数：`--extras`（+ mediapipe / pyrealsense2）、`--torch`（+ CPU 版 torch，走阿里云/官方 CPU 索引）、`--out <dir>`（默认 `wheels/`）、`--no-python-installer`
- 镜像链：阿里云 → 清华 → 官方源（与 start.bat 安装顺序一致）
- 产物：`<out>/*.whl`（Windows 3.12 依赖闭包）+ `<out>/python-3.12.10-amd64.exe`

**调用关系**：不 import 仓库任何模块（纯标准库）；产物被 `start.bat`/`start.sh` 的离线安装逻辑消费。

# Data Acquisition

[中文](README.zh-CN.md) | [English](README.md)

> 文档版本：2026-09-02  ·  后端版本：`1.6.8`  ·  WEB 前端版本：`1.5.6`

Data Acquisition 是 EgoData 的数据采集、处理、审核和导出服务，面向具身智能与机器人
操作数据。系统由 FastAPI 后端、异步 Worker、Web 审核页面和 Workflow Studio 组成。

```text
采集端上传 → 项目匹配 → 工作流处理 → AI/人工审核 → LeRobot/HDF5 导出
```

## 文档导航

- [系统组成](#1-系统组成)
- [安装与启动](#7-安装与启动)
- [代码目录](#2-代码目录)
- [项目数据结构](#3-项目数据结构)
- [视频数据规范](#4-视频数据规范)
- [工作流模块](#5-工作流模块)
- [Web Workflow Studio 前端模块说明](#web-workflow-studio-前端模块说明)
- [处理结果保存](#6-处理结果保存)
- [配置](#8-配置)
- [API](#9-api)
- [测试与排障](#10-测试与排障)

## 1. 系统组成

```text
采集端
  │
  ▼
FastAPI 后端
  ├── 项目、Episode、工作流和审核 API
  ├── 上传接收、解压、规范化和数据校验
  └── 调度 Worker 处理任务
  │
  ▼
Worker
  ├── MediaPipe 手部关键点
  ├── RGB-D metric 3D 手部关键点
  ├── 通过 API 进行 AI 标注
  └── 视频与标注质量审核
  │
  ▼
Web 审核页面
  ├── 原始 RGB 视频
  ├── 浏览器 SVG/Canvas 关键点叠加
  └── 深度视频伪彩色预览
  │
  ▼
LeRobot v2.1 / v3.0 或 HDF5
```

## 2. 代码目录

<details>
<summary>点击展开：代码目录</summary>

```text
Data Acquisition/
├── app/                    # FastAPI、存储、工作流、处理和导出逻辑
├── worker/                 # 异步处理 Worker
├── scripts/                # 部署、启动、检查和数据维护脚本
├── web/
│   ├── templates/          # Web 页面模板
│   ├── static/             # CSS、JavaScript 和前端构建产物
│   └── workflow-studio/    # React + Vite 工作流编辑器源码
├── tests/                  # 自动化测试
├── requirements*.txt       # Python 依赖清单
├── deploy.py               # 跨平台部署入口
├── deploy.sh               # Linux 部署入口
├── deploy.bat              # Windows 部署入口
└── .env.example            # 配置模板
```

运行数据默认存储在 `STORAGE_DIR`：

```text
data/
├── sessions/               # 项目数据集
├── state/                  # 系统状态、队列、运行记录和导出任务
└── tmp/                    # 上传和 Worker 临时文件
```

运行数据、模型权重、压缩包、凭据和虚拟环境不应提交到 Git。`requirements*.txt` 和
`package.json` 等依赖清单应保留在仓库中。

</details>

## 3. 项目数据结构

<details>
<summary>点击展开：项目数据结构、Episode、Chunk 和 Meta</summary>

一个项目就是一个数据集根目录。规范的项目根目录固定只有 `data/`、`meta/`、`videos/`
三个目录，不在项目下创建额外的批次级数据目录：

```text
data/sessions/<project>/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── episode_000001.parquet
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.json
│   └── episodes/
│       └── chunk-000/
│           ├── episode_000000.parquet
│           └── episode_000001.parquet
└── videos/
    └── chunk-000/
        ├── observation.images.head_rgb/
        │   └── episode_000000.mp4
        └── observation.images.head_depth/
            └── episode_000000.mp4
```

### Episode 和 Chunk

- Episode 编号从 `000000` 开始；
- `episode_000000` 至 `episode_000999` 存储在 `chunk-000`；
- `episode_001000` 开始使用 `chunk-001`；
- 每个 chunk 最多包含 1000 个 Episode；
- `data`、`meta/episodes` 和 `videos` 使用相同的 Episode/Chunk 对应关系；
- 采集批次名称用于业务追踪，不替代数据集的 `episode_index`。

### Meta 目录

规范的 `meta/` 目录包含：

- `info.json`：数据集版本、FPS、帧数、feature 和路径模板；
- `stats.json`：数据字段统计信息；
- `tasks.json`：任务 ID 和任务描述；
- `episodes/chunk-XXX/episode_XXXXXX.parquet`：单个 Episode 的时间戳、设备信息、视频
  信息、采集元数据、校准信息和处理状态。

时间戳、设备信息和校准信息写入对应的 Episode。审核页和标注页只使用当前 Episode 的
真实视频、深度流和元数据。

</details>

## 4. 视频数据规范

<details>
<summary>点击展开：视频源、深度存储和前端伪彩色预览</summary>

### 视频源

| 类型 | 源名称示例 | 说明 |
| --- | --- | --- |
| 单目 RGB | `head_rgb` | 普通 RGB 视频 |
| Metric 深度 | `head_depth` | 独立的真实深度视频 |
| 双目 RGB | `stereo_left_rgb`、`stereo_right_rgb` | 左、右 RGB 视频 |
| 双目深度 | `stereo_depth` | 独立深度视频 |

RGB 和 Depth 始终作为独立数据流保存。纯 Depth 流不会被当作 RGB，也不会与 RGB 流重复
保存。

### 深度存储

新的真实深度视频统一使用：

```text
HEVC (H.265) in MP4
```

深度视频保存的是 12-bit 对数深度码，不是伪彩色图像：

```text
100–5000 mm 对数深度 → 0..4095 gray12le 码值
```

`meta/info.json` 中对应的 feature 使用 `video.is_depth_map=true` 标记，并记录深度编码
参数。读取端将码值反量化为毫米。原始深度视频在处理和导出时保持不变。

### 前端深度预览

深度伪彩色只用于浏览器预览，不写回数据集：

```text
HEVC gray12le
      │
      └── 浏览器 Canvas：深度码 → JET 伪彩色
```

预览颜色规范：

- 无效深度：黑色；
- 近距离：蓝紫色；
- 中距离：绿色、黄色；
- 远距离：橙色、红色。

打开 Episode 时会预加载深度预览数据。RGB 视频、双目 RGB 视频和导出数据不经过伪彩色
转换，也不会生成额外的伪彩色视频。

</details>

## 5. 工作流模块

<details>
<summary>点击展开：输入、处理、审核、导出模块和节点状态</summary>

工作流由 Input、Process、Review 和 Export 四类模块组成。端口具有明确的数据类型，
每个分支只处理连接到对应端口的数据。没有输入连接的分支标记为
`Skipped · Missing Input`，不会阻塞其他已连接分支。

### 输入模块

- `RGB Camera`：单路 RGB 视频；
- `RGB-D Camera`：RGB 视频和 Depth；
- `Stereo RGB Camera`：Left RGB Video 和 Right RGB Video；
- `Stereo RGB-D Camera`：左右 RGB 视频和 Depth；
- `Glove Sensor`：手套压力、关节或传感器数据。

### 处理模块

- `MediaPipe Hand`：从 RGB 视频提取 2D 手部关键点；
- `Human Annotation`：人工标注；
- `AI Annotation`：通过配置的 API 进行视频/片段标注；
- `RGB_TO_2D_BareHand`：裸手 2D 手部关键点；
- `RGB_TO_2D_BlackGlove`：黑手套 2D 手部关键点；
- `RGB-D_3D_BareHand`：使用 RGB 视频和 Depth 计算真实 metric 3D 裸手关键点；
- `RGB-D_3D_BlackGlove`：使用 RGB 视频和 Depth 计算真实 metric 3D 黑手套关键点。

`RGB_TO_2D_*` 的 `Spatial` 按钮只控制浏览器中的空间预览，不生成虚假 3D 数据，也不会
向导出数据写入 3D 值。`RGB-D_3D_*` 的 `RGB Video` 和 `Depth` 是独立输入端，应分别连接
对应的采集端输出。

### Web Workflow Studio 前端模块说明

以下模块是当前前端调色板和画布实际注册的模块。模块名称、端口名称和连接类型由前端
注册表统一维护，不会被陈旧工作流或废弃后端目录覆盖。旧类型
`rgb_hand_3d`、`black_hand_rgb_3d`、`stereo_triangulate` 和 `black_glove_hand` 仅在加载
旧工作流时做映射，新工作流调色板不显示这些类型。

#### 采集端模块（Input）

| 前端模块 | 输入端 | 输出端 | 用途与规则 |
| --- | --- | --- | --- |
| `RGB Camera` | 无 | `RGB Video` | 单目 RGB 视频。设备从当前项目数据中选择；留空时自动匹配。 |
| `RGB-D Camera` | 无 | `RGB Video`、`Depth` | 输出同一设备的 RGB 和真实深度流；Depth 只能连接到深度输入端。 |
| `Stereo RGB Camera` | 无 | `Left RGB Video`、`Right RGB Video` | 双目 RGB 输入；连接一侧时建立对应的左右关系。 |
| `Stereo RGB-D Camera` | 无 | `Left RGB Video`、`Right RGB Video`、`Depth` | 双目 RGB 加真实深度；Depth 用于 RGB-D 3D 处理。 |
| `Glove Sensor` | 无 | `Glove Sensor Data` | 压力、关节或手套传感器数据，不输出视频；可直接连接质量审核。 |

#### 视频和关键点处理模块（Process）

| 前端模块 | 输入端 | 输出端 | 用途与规则 |
| --- | --- | --- | --- |
| `Human Annotation` | `RGB Video` | `Annotation` | 在 RGB 视频上进行帧级人工标注。 |
| `AI Annotation` | `RGB Video` | `Annotation` | 通过本地或 API VLM 进行视频/片段标注；部署时推荐使用 API。 |
| `RGB_TO_2D_BareHand` | `RGB Video` | `Hand 2D` | 裸手 2D 关键点；`Spatial` 仅用于显示，不导出 metric 3D。 |
| `RGB_TO_2D_BlackGlove` | `RGB Video` | `Hand 2D` | 黑手套 2D 关键点；没有 Depth 时不计算 metric 3D。 |
| `RGB-D_3D_BareHand` | `RGB Video`、`Depth` | `Hand 3D` | 使用 RGB 和真实 Depth 计算 metric 3D 裸手关键点；缺少分支时跳过。 |
| `RGB-D_3D_BlackGlove` | `RGB Video`、`Depth` | `Hand 3D` | 使用 RGB 和真实 Depth 计算 metric 3D 黑手套关键点；不会从磁盘寻找或回退到 RGB 估计。 |

`MediaPipe Hand` 保留用于历史工作流和后端兼容，但新工作流调色板不显示。所有节点都
支持鼠标悬停说明，提示模块用途和允许连接的数据类型。

#### 审核模块（Review）

| 前端模块 | 输入端 | 输出端 | 用途与规则 |
| --- | --- | --- | --- |
| `Human Review` | `Review Target` | `Reviewed Data` | 人工检查视频、关键点或标注结果，再进入后续流程。 |
| `AI Quality Review` | `Quality Review Target` | `Reviewed Data` | 检查视频解码、帧连续性、黑屏、冻结、标注覆盖率和传感器数据。 |

#### 导出模块（Export）

| 前端模块 | 输入端 | 输出端 | 用途与规则 |
| --- | --- | --- | --- |
| `LeRobot Export` | `Exportable Data` | `Dataset` | 导出 LeRobot v2.1 或 v3.0；版本在节点上选择。 |
| `HDF5 Export` | `Exportable Data` | `Dataset` | 使用配置的压缩参数导出 HDF5 数据集。 |

#### Workflow Studio 界面功能

<details>
<summary>点击展开：前端界面组件说明</summary>

| 界面组件 | 功能 |
| --- | --- |
| `NodePalette` | 按 Input、Process、Review、Export 分类显示节点；支持搜索、分类折叠和悬停说明。 |
| `WorkflowCanvas` | 支持拖拽创建节点、端口连线、移动、选择、缩放、网格对齐和小地图导航。 |
| `WorkflowNode` | 渲染卡片标题、输入输出端口、项目设备选择/输入框、`Spatial` 和 API 设置按钮。 |
| `WorkflowDrawer` | 查看工作流列表、创建和加载工作流，并刷新当前项目的输入源。 |
| `PipelineToolbar` | 提供新建、保存、另存为、导出工作流 JSON 和运行操作；黄色圆点表示有未保存修改。 |
| `NodeSettingsModal` | 配置 AI Annotation 的 API 厂商、模型、地址和密钥字段。 |
| `DeletableEdge` | 显示数据连接线；选中后可以删除。 |

设备选择始终绑定当前项目。新的空工作流不会显示其他项目的设备；数据上传后，每个卡片
的选择框只显示与该卡片 RGB、RGB-D、双目或手套类型匹配的真实数据源。

</details>

### 审核与执行规则

输入端和输出端必须按照端口名称连接。没有输入连接的处理分支直接跳过，不影响其他分支。
只有在 RGB Video 和 Depth 都可用时，RGB-D 3D 才会产生真实 metric 3D 数据。

节点状态限制为：

```text
Connected
Processing
Completed
Skipped · Missing Input
Completed with Warning
```

</details>

## 6. 处理结果保存

<details>
<summary>点击展开：处理结果、上传解压和清理规则</summary>

Worker 完成处理后，结果合并到对应 Episode：

```text
data/chunk-000/episode_000000.parquet
  └── 2D 关键点、metric 3D、传感器和标注字段

meta/episodes/chunk-000/episode_000000.parquet
  └── processing_*、run_id、节点状态和输出索引
```

原始 RGB/Depth 视频不会被覆盖。手部关键点预览在浏览器中叠加到原始 RGB 视频上，不保存
第二份骨骼视频。系统级运行记录位于 `data/state/runs/`，导出临时产品位于
`data/state/exports/`；两者都不属于项目数据集目录。

上传压缩包的流程为：接收 → 解压 → 规范化 → 写入项目 → 校验 → 更新状态。只有完整解压、
原子提交和校验全部成功后，上传临时文件和解压临时目录才会清理；失败时保留暂存内容供
重试和诊断。

</details>

## 7. 安装与启动

### Linux

```bash
cd "Data Acquisition"
cp .env.example .env
# 编辑 .env，设置 API_KEY、WORKER_API_KEY、JWT_SECRET 和存储配置。
chmod +x deploy.sh
./deploy.sh
```

常用参数：

```bash
python deploy.py --check-only       # 只检查，不安装、不修改配置、不启动服务。
python deploy.py --skip-vllm        # 不启动本地 VLM，AI 标注使用 API。
python deploy.py --no-services      # 前台运行，不安装自启动服务。
```

当前 AI 标注部署使用 API。使用 `--skip-vllm`，然后在 Workflow Studio 的
`AI Annotation` 设置中配置 API 厂商、模型和地址。

### Windows

```bat
cd /d "Data Acquisition"
copy .env.example .env && deploy.bat
```

### 手动启动后端

```bash
cd "Data Acquisition"
source .venv-linux/bin/activate
export PYTHONPATH="$PWD"
./scripts/run_backend_linux.sh
```

也可以直接运行：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 启动 Worker

```bash
cd "Data Acquisition"
export EGODATA_SERVER_URL=http://127.0.0.1:8000
export EGODATA_WORKER_API_KEY='与服务端一致的 Worker API Key'
./scripts/run_worker_linux.sh
```

Windows 使用 `scripts/run_worker_windows.ps1`。Linux 和 Windows 的虚拟环境相互独立，
不能跨系统复制。

### Workflow Studio

```bash
cd "Data Acquisition/web/workflow-studio"
npm ci
npm run dev
```

生产构建：

```bash
npm run build
```

构建产物写入 `web/static/workflow-studio/`，由 FastAPI 页面服务加载。

## 8. 配置

<details>
<summary>点击展开：环境变量和安全配置</summary>

配置写入 `Data Acquisition/.env` 或环境变量，不提交到仓库：

| 配置 | 作用 |
| --- | --- |
| `STORAGE_DIR` | 数据和系统状态根目录 |
| `STORAGE_BACKEND` | `local` 或 `sftp` |
| `SFTP_*` | 远端数据目录和 SSH 连接配置 |
| `API_KEY` | 采集端和受保护接口的 API Key |
| `WORKER_API_KEY` | Worker 领取和回传任务使用的 API Key |
| `JWT_SECRET` | Web 登录态签名密钥 |
| `UPLOAD_STAGING_DIR` | 上传压缩包的本地暂存目录 |
| `HOST` / `PORT` | Web/API 监听地址和端口 |
| `PUBLIC_BASE_URL` | 对外访问地址 |

完整模板见 [`.env.example`](.env.example)。生产环境必须使用随机密钥并限制 `.env` 权限。
不要在日志、工作流 JSON、前端代码或文档中写入密码、Token、API Key 和数据库凭据。

</details>

## 9. API

<details>
<summary>点击展开：主要 API 和 OpenAPI 文档</summary>

服务启动后可访问：

```text
GET /health
GET /docs
GET /openapi.json
```

主要 API 分组：

```text
/api/v1/auth/*          登录和认证
/api/v1/projects/*      项目和 Episode
/api/v1/workflows/*     工作流定义
/api/v1/session/*       采集端上传
/api/v1/video/*         视频、深度和关键点预览
/api/v1/annotations/*   标注和审核
/api/v1/export/*        数据导出
/api/v1/worker/*        Worker 任务领取与回传
```

完整接口契约以运行中的 FastAPI OpenAPI 文档为准。

</details>

## 10. 测试与排障

<details>
<summary>点击展开：测试命令和常见排障步骤</summary>

后端和 Worker 基础检查：

```bash
cd "Data Acquisition"
python -m compileall app worker
python deploy.py --check-only
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_depth_codec.py
```

前端检查：

```bash
cd "Data Acquisition/web/workflow-studio"
npm run build
```

排障顺序：

1. 访问 `/health`，确认服务和存储可用；
2. 检查 `STORAGE_DIR` 是否指向实际数据目录；
3. 检查上传状态和当前 Episode 的实际文件；
4. 检查 Worker 是否在线、能领取任务并发送心跳；
5. 检查 `data`、`meta/episodes`、`videos` 的 Episode 编号是否一致；
6. RGB-D 3D 节点检查 `RGB Video` 和 `Depth` 是否分别接线；
7. 深度预览异常时检查视频是否为 `gray12le`，不要把伪彩色预览当作原始深度数据。

</details>

## 11. 版本约定

<details>
<summary>点击展开：版本维护规则</summary>

- 后端版本在 `app/version.py` 中维护；
- Web 前端版本在 `web/workflow-studio/package.json` 中维护；
- 修复和小调整增加修订号；新增功能增加次版本号；重大架构变更增加主版本号；
- 发布前应同步记录软件版本、数据格式和 API 变化。

当前版本：后端 `1.6.8`，WEB 前端 `1.5.6`。

</details>

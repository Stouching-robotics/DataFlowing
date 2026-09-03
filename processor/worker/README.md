# EgoData Worker

Worker 是 Data Acquisition 的异步处理进程。它从 FastAPI 服务领取工作流任务，读取当前
episode 的 canonical 数据，执行手部关键点、深度 3D、AI 标注和质量审核模块，再把结果
合并回对应项目的 `data/` 和 `meta/episodes/`。

Worker 不提供前端页面，也不直接修改原始 RGB/Depth 视频。手部预览由浏览器叠加关键点完成，
不会生成或保存第二份骨骼视频。

## 运行前提

- 后端服务已经启动并可访问；
- Worker 使用的 `EGODATA_WORKER_API_KEY` 与服务端 `WORKER_API_KEY` 一致；
- Linux 和 Windows 使用各自独立的虚拟环境；
- Worker 的临时目录有足够空间，处理完成后临时结果由服务端清理。

## Linux

```bash
cd "/path/to/Data Acquisition"
bash scripts/setup_linux.sh

export EGODATA_SERVER_URL=http://127.0.0.1:8000
export EGODATA_WORKER_API_KEY='服务器配置的 Worker API Key'
export EGODATA_WORKER_ID="linux-$(hostname)"
bash scripts/run_worker_linux.sh
```

默认值：

```text
EGODATA_SERVER_URL=http://127.0.0.1:8000
EGODATA_DEVICE=auto
EGODATA_WORK_DIR=<项目>/data/tmp/worker
EGODATA_POLL_SECONDS=2
```

## Windows

```powershell
cd "E:\Company-File\Date-V\Data Acquisition"
.\scripts\setup_windows.ps1
$env:EGODATA_SERVER_URL = 'http://127.0.0.1:8000'
$env:EGODATA_WORKER_API_KEY = '服务器配置的 Worker API Key'
.\scripts\run_worker_windows.ps1
```

Windows 环境位置为 `.venv-windows\Scripts\python.exe`，Linux 环境位置为
`.venv-linux/bin/python`。不能跨系统复制虚拟环境。

## 处理结果规则

Worker 返回的处理结果不会写入项目级 `processed/`、`meta/processing/` 或
`processing_manifest.json`。服务端完成任务时执行以下合并：

```text
项目/data/chunk-XXX/episode_XXXXXX.parquet
  └── 2D 关键点、metric 3D、传感器和标注字段

项目/meta/episodes/chunk-XXX/episode_XXXXXX.parquet
  └── processing_* 状态、run_id、节点状态和输出索引
```

`data/state/runs/` 是系统级任务队列、租约和审计记录，位于项目目录外，不属于数据集。
缺少某个端口输入的节点会标记为 `Skipped · Missing Input`，不会阻止其他已连接分支执行。

## 日志与排障

优先检查：

1. `GET ${EGODATA_SERVER_URL}/health` 是否返回正常；
2. Worker API Key 是否和后端配置一致；
3. Worker 日志中是否持续出现领取任务、心跳和完成/跳过信息；
4. 项目的 `data/`、`meta/episodes/`、`videos/` 是否使用同一个 episode 编号；
5. RGB-D 3D 节点是否同时接入了 `RGB Video` 和 `Depth`，以及深度视频是否为
   canonical HEVC `gray12le`；
6. 浏览器预览问题是否只是解码器兼容问题，不要把临时预览缓存当成原始数据。

查看后端 OpenAPI：

```text
http://127.0.0.1:8000/docs
```

不要在日志、环境变量示例、工作流 JSON 或 Issue 中粘贴密码、Token、API Key 或数据库
连接串。

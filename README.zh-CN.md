# DataFlowing

数据采集、数据处理、AI 标注、审核和 LeRobot 导出服务。

[English](README.md)

## 仓库结构

```text
DataFlowing/
├── collector/                 # 采集端和设备数据上传
├── processor/                 # 服务器端处理服务
│   ├── app/                   # API、存储、工作流和导出
│   ├── worker/                # 异步处理 Worker
│   ├── web/                   # 数据审核界面和 Workflow Studio
│   ├── scripts/               # 部署、迁移和检查脚本
│   ├── tests/                 # processor 测试
│   ├── examples/              # 短小脱敏示例数据
│   ├── requirements*.txt      # Python 依赖清单
│   └── README.md              # processor 详细文档
├── README.md
└── README.zh-CN.md
```

采集端运行在连接相机和传感器的电脑上；processor 运行在服务器上，负责上传校验、工作流执行、AI 标注、后处理、人工审核和数据导出。

典型链路：

```text
采集 → 上传 → 校验 → 工作流处理 → 审核 → LeRobot/HDF5 导出
```

## 数据和隐私

生产会话、密码、设备标识、模型权重和备份文件均属于运行时数据，不提交到本仓库。示例仅包含短时长的脱敏视频和示例元数据。

深度视频保留原始 12-bit `gray12le` 深度码；伪彩色只在浏览器预览时生成，不会写入保存数据。

## LeRobot 导出

processor 支持按照工作流选择 LeRobot v2.1 或 v3.0，支持单个 Episode 和批量导出，并将后处理后的数值数据写入导出的 Parquet 文件。

详细的安装、配置、工作流和验证说明请查看 [`processor/README.md`](processor/README.md)。

## 快速启动

```bash
cd processor
cp .env.example .env
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

数据库、存储、鉴权、SFTP 和可选 AI 服务通过 `.env` 配置。不要提交 `.env`、生产数据或模型文件。

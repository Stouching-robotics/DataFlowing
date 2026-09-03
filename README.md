# DataFlowing

Data acquisition, processing, AI annotation, review and LeRobot export services.

[中文说明](README.zh-CN.md)

## Repository layout

```text
DataFlowing/
├── collector/                 # acquisition client and device upload
├── processor/                 # server-side processing service
│   ├── app/                   # API, storage, workflows and exporters
│   ├── worker/                # asynchronous processing worker
│   ├── web/                   # review UI and Workflow Studio
│   ├── scripts/               # deployment, migration and inspection tools
│   ├── tests/                 # processor tests
│   ├── examples/              # short sanitized example sessions
│   ├── requirements*.txt      # Python dependency manifests
│   └── README.md              # processor documentation
├── README.md
└── README.zh-CN.md
```

The collector runs near the recording devices. The processor runs on the server and handles upload validation, workflow execution, AI annotation, post-processing, review and export.

Typical pipeline:

```text
Capture → Upload → Validate → Workflow processing → Review → LeRobot/HDF5 export
```

## Data and privacy policy

Production sessions, credentials, device identifiers, model weights and backups are runtime assets and are not part of this repository. The examples contain only short, sanitized clips and representative metadata. Depth examples retain the original 12-bit `gray12le` depth stream; pseudo-color is generated only for browser preview and is never stored in the dataset.

## LeRobot export

The processor supports the workflow-selected LeRobot format (v2.1 or v3.0), single-episode and batch export, and carries post-processed numeric data into the exported Parquet files. See [`processor/README.md`](processor/README.md) for setup, configuration and validation commands.

## Quick start

```bash
cd processor
cp .env.example .env
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

Configure database, storage, authentication and optional SFTP/AI services through `.env`. Never commit `.env`, production sessions or model assets.

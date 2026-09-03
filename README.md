# DataFlowing

Data acquisition, processing, AI annotation and LeRobot export services.

## Repository layout

```text
DataFlowing/
├── collector/                 # acquisition client and device upload
├── processor/                 # server-side processing service
│   ├── app/                   # API, storage, workflows and exports
│   ├── worker/                # asynchronous processing worker
│   ├── web/                   # review UI and workflow studio
│   ├── scripts/               # deployment, migration and inspection tools
│   ├── tests/                 # processor tests
│   ├── examples/              # small sanitized example datasets
│   ├── requirements*.txt
│   └── README.md
└── README.md
```

The collector and processor are intentionally separated: the collector runs near the recording devices, while the processor runs on the server and handles validation, post-processing, review and export.

## Data and privacy policy

Production sessions, credentials, device identifiers, model weights and backups are runtime assets and are not part of this repository. The examples contain only short, sanitized clips and representative metadata. Depth examples retain the original 12-bit `gray12le` depth stream; no pseudo-color frames are stored.

## LeRobot export

The processor supports the project workflow's selected LeRobot format (v2.1 or v3.0), single-episode and batch export, and carries post-processed numeric data into the exported Parquet files. See [`processor/README.md`](processor/README.md) for setup and validation commands.

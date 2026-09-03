# Data Acquisition

[English](README.md) | [中文](README.zh-CN.md)

> Documentation: 2026-09-03  ·  Backend: `1.6.8`  ·  Web frontend: `1.5.6`

Data Acquisition is EgoData's service for data capture, processing, review, and export. It is
designed for embodied AI and robot manipulation data. The system consists of a FastAPI backend,
an asynchronous Worker, Web review pages, and Workflow Studio.

```text
Capture upload → Project matching → Workflow processing → AI/Human review → LeRobot/HDF5 export
```

## Documentation navigation

- [System overview](#1-system-overview)
- [Installation and startup](#7-installation-and-startup)
- [Code layout and repository tree](#2-code-layout-and-repository-tree)
- [Project data structure](#3-project-data-structure)
- [Video data specification](#4-video-data-specification)
- [Workflow modules](#5-workflow-modules)
- [Web Workflow Studio modules](#web-workflow-studio-frontend-module-reference)
- [Processing result storage](#6-processing-result-storage)
- [Configuration](#8-configuration)
- [API](#9-api)
- [Testing and troubleshooting](#10-testing-and-troubleshooting)

## 1. System overview

```text
Capture client
  │
  ▼
FastAPI backend
  ├── Project, Episode, workflow, and review APIs
  ├── Upload reception, extraction, normalization, and validation
  └── Worker task scheduling
  │
  ▼
Worker
  ├── MediaPipe hand keypoints
  ├── RGB-D metric 3D hand keypoints
  ├── AI annotation through an API
  └── Video and annotation quality review
  │
  ▼
Web review pages
  ├── Original RGB video
  ├── Browser-side SVG/Canvas keypoint overlays
  └── Pseudo-color depth preview
  │
  ▼
LeRobot v2.1 / v3.0 or HDF5
```

## 2. Code layout and repository tree

<details>
<summary>Click to expand: source tree and runtime directories</summary>

The current maintained source tree is shown below. Runtime data, caches, virtual environments,
local backups, test exports, and downloaded model weights are intentionally omitted.

```text
Data Acquisition/
├── app/
│   ├── api/                          # Project, workflow, user, and Worker APIs
│   ├── processing/
│   │   ├── black_glove/              # Black-glove detection, tracking, and pose backend
│   │   └── modules/                  # RGB, RGB-D, stereo, hand, glove, review, and export nodes
│   ├── prompts/                      # AI annotation prompts and vocabularies
│   ├── routes/                       # Page, session, video, annotation, and export routes
│   ├── ai_annotation.py              # Local/API VLM annotation pipeline
│   ├── browser_preview.py             # Browser preview metadata and media preparation
│   ├── export_engine.py               # Export job orchestration
│   ├── hdf5_export.py                 # HDF5 dataset writer
│   ├── lerobot_export.py              # LeRobot v3.0 exporter and metadata writer
│   ├── lerobot_v21.py                 # LeRobot v2.1 normalization support
│   ├── project_dataset.py              # Canonical project and Episode operations
│   ├── storage.py / remote_storage.py # Local and SFTP-backed storage
│   ├── workflow_*.py                  # Workflow schema, binding, and dispatch
│   ├── models.py / database.py         # Runtime state models and persistence
│   └── main.py                         # FastAPI application entry point
├── worker/
│   ├── runner.py                      # Job claim, heartbeat, execution, and callback loop
│   ├── client.py                      # Backend Worker API client
│   └── README.md                      # Worker-specific notes
├── web/
│   ├── templates/                     # Server-rendered application pages
│   ├── static/
│   │   ├── js/                        # Player, depth preview, overlays, review, and UI JS
│   │   └── workflow-studio/            # Built Workflow Studio assets served by FastAPI
│   └── workflow-studio/
│       └── src/
│           ├── api/                   # Frontend API clients
│           ├── components/             # Canvas, nodes, palette, drawer, and settings UI
│           ├── store/                  # Workflow and UI state stores
│           └── App.tsx                 # Workflow Studio application root
├── scripts/                           # Setup, startup, migration, and maintenance scripts
│   ├── run_backend_linux.sh           # Linux FastAPI startup
│   ├── run_worker_linux.sh            # Linux Worker startup
│   ├── hot_reload_backend.py          # Development backend reload supervisor
│   ├── hot_reload_worker.py           # Development Worker reload supervisor
│   ├── migrate_*.py / repack_*.py      # Dataset migration and repair tools
│   └── systemd/                       # Linux service templates
├── tests/
│   └── test_depth_codec.py             # 12-bit depth codec tests
├── requirements*.txt                  # Python dependency manifests
├── deploy.py                          # Cross-platform deployment entry point
├── deploy.sh / deploy.bat             # Linux and Windows deployment entry points
├── Dockerfile / .dockerignore         # Container build files
├── .env.example                       # Safe configuration template
├── .gitignore                         # Runtime data, secrets, models, and cache exclusions
├── README.md                          # English documentation
└── README.zh-CN.md                    # Chinese documentation
```

Runtime data is stored under `STORAGE_DIR` by default:

```text
data/
├── sessions/               # Project datasets
├── state/                  # System state, queues, run records, and export jobs
├── logs/                   # Runtime and archive-sync logs
├── .meta-history/          # Metadata migration rollback snapshots
└── tmp/                    # Upload, AI, and Worker temporary files
```

Runtime data, model weights, archives, credentials, and virtual environments must not be
committed to Git. Dependency manifests such as `requirements*.txt` and `package.json` should be
kept in the repository.

</details>

## 3. Project data structure

<details>
<summary>Click to expand: project data structure, Episodes, Chunks, and metadata</summary>

A project is the root directory of one dataset. Its canonical root contains exactly `data/`,
`meta/`, and `videos/`; no additional batch-level data directory is created under the project:

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

### Episodes and Chunks

- Episode numbering starts at `000000`;
- `episode_000000` through `episode_000999` are stored in `chunk-000`;
- `episode_001000` starts `chunk-001`;
- each chunk contains at most 1,000 Episodes;
- `data`, `meta/episodes`, and `videos` use the same Episode/Chunk mapping;
- the capture batch name is used for business tracking and does not replace the dataset's
  `episode_index`.

### Meta directory

The canonical `meta/` directory contains:

- `info.json`: dataset version, FPS, frame counts, features, and path templates;
- `stats.json`: statistics for dataset fields;
- `tasks.json`: task IDs and task descriptions;
- `episodes/chunk-XXX/episode_XXXXXX.parquet`: per-Episode timestamps, device information, video
  information, capture metadata, calibration information, and processing state.

Timestamps, device information, and calibration information are written to the corresponding
Episode. Review and annotation pages use only the actual videos, depth streams, and metadata of
the current Episode.

</details>

## 4. Video data specification

<details>
<summary>Click to expand: video sources, depth storage, and browser pseudo-color preview</summary>

### Video sources

| Type | Example source name | Description |
| --- | --- | --- |
| Mono RGB | `head_rgb` | Regular RGB video |
| Metric depth | `head_depth` | Independent metric-depth video |
| Stereo RGB | `stereo_left_rgb`, `stereo_right_rgb` | Left and right RGB videos |
| Stereo depth | `stereo_depth` | Independent depth video |

RGB and Depth are always stored as independent streams. A depth-only stream is never treated as
RGB and is never duplicated alongside the RGB stream.

### Depth storage

New metric-depth videos use:

```text
HEVC (H.265) in MP4
```

The depth video stores 12-bit logarithmic depth codes, not pseudo-color images:

```text
100–5000 mm logarithmic depth → 0..4095 gray12le codes
```

The corresponding feature in `meta/info.json` is marked with `video.is_depth_map=true` and the
depth encoding parameters. Readers dequantize the codes back to millimeters. The original depth
video is preserved during processing and export.

Depth is never exported as a JET/RGB pseudo-color video. Exported depth remains a single-channel
`gray12le` video containing the original 12-bit logarithmic codes. The frontend must use the
`codes_to_heatmap_bgr()` rule for decoded depth codes; it must not quantize decoded codes a second
time as if they were millimeter values.

### Frontend depth preview

Pseudo-color depth is used only for browser preview and is never written back to the dataset:

```text
HEVC gray12le
      │
      └── Browser Canvas: depth codes → JET pseudo-color
```

Preview color convention:

- invalid depth: black;
- near range: blue-violet;
- middle range: green and yellow;
- far range: orange and red.

Depth preview data is prefetched when an Episode is opened. RGB video, stereo RGB video, and
export data are not pseudo-colorized, and no additional pseudo-color video is generated.

</details>

## 5. Workflow modules

<details>
<summary>Click to expand: input, processing, review, export modules, and node states</summary>

Workflows consist of Input, Process, Review, and Export modules. Ports have explicit data types;
each connected branch processes only the stream connected to that port. A branch without an input
connection is marked `Skipped · Missing Input` and does not block other connected branches.

### Input modules

- `RGB Camera`: single-view RGB video;
- `RGB-D Camera`: RGB video and Depth;
- `Stereo RGB Camera`: Left RGB Video and Right RGB Video;
- `Stereo RGB-D Camera`: left/right RGB video and Depth;
- `Glove Sensor`: glove pressure, joint, or sensor data.

### Process modules

- `MediaPipe Hand`: extracts 2D hand keypoints from RGB video;
- `Human Annotation`: manual annotation;
- `AI Annotation`: video/segment annotation through the configured API;
- `RGB_TO_2D_BareHand`: 2D bare-hand keypoints;
- `RGB_TO_2D_BlackGlove`: 2D black-glove keypoints;
- `RGB-D_3D_BareHand`: true metric 3D bare-hand keypoints from RGB video + Depth;
- `RGB-D_3D_BlackGlove`: true metric 3D black-glove keypoints from RGB video + Depth.

The `Spatial` button on `RGB_TO_2D_*` only controls a browser visualization. It does not create
fake 3D data or add 3D values to exports. The `RGB Video` and `Depth` ports on `RGB-D_3D_*` are
independent inputs and should be connected to the corresponding capture outputs.

### Web Workflow Studio frontend module reference

The following modules are the ones currently registered by the frontend palette and canvas. Node
names, port names, and connection types are maintained by the frontend registry and are not
overwritten by a stale workflow or deprecated backend catalog. The legacy types
`rgb_hand_3d`, `black_hand_rgb_3d`, `stereo_triangulate`, and `black_glove_hand` are mapped only
when old workflows are loaded; they are not shown in the new-workflow palette.

#### Capture modules (Input)

| Frontend module | Inputs | Outputs | Purpose and rules |
| --- | --- | --- | --- |
| `RGB Camera` | None | `RGB Video` | Mono RGB video. The device is selected from the current project's data; blank means auto-match. |
| `RGB-D Camera` | None | `RGB Video`, `Depth` | Emits RGB and real depth streams from the same device; Depth can connect only to a depth input. |
| `Stereo RGB Camera` | None | `Left RGB Video`, `Right RGB Video` | Stereo RGB input. Connecting one side creates the corresponding left/right relationship. |
| `Stereo RGB-D Camera` | None | `Left RGB Video`, `Right RGB Video`, `Depth` | Stereo RGB plus real depth; Depth is used by RGB-D 3D processing. |
| `Glove Sensor` | None | `Glove Sensor Data` | Pressure, joint, or glove sensor data; no video output. It can connect directly to quality review. |

#### Video and keypoint processing modules (Process)

| Frontend module | Inputs | Outputs | Purpose and rules |
| --- | --- | --- | --- |
| `Human Annotation` | `RGB Video` | `Annotation` | Frame-level manual annotation on RGB video. |
| `AI Annotation` | `RGB Video` | `Annotation` | Video/segment annotation through a local or API VLM; API is recommended for deployment. |
| `RGB_TO_2D_BareHand` | `RGB Video` | `Hand 2D` | Bare-hand 2D keypoints. `Spatial` is display-only and exports no metric 3D. |
| `RGB_TO_2D_BlackGlove` | `RGB Video` | `Hand 2D` | Black-glove 2D keypoints. Without Depth, it does not calculate metric 3D. |
| `RGB-D_3D_BareHand` | `RGB Video`, `Depth` | `Hand 3D` | Metric 3D bare-hand keypoints from RGB and real Depth; a missing branch is skipped. |
| `RGB-D_3D_BlackGlove` | `RGB Video`, `Depth` | `Hand 3D` | Metric 3D black-glove keypoints from RGB and real Depth; it does not search the disk or fall back to RGB estimation. |

`MediaPipe Hand` remains available for historical workflows and backend compatibility but is hidden
from the new-workflow palette. All nodes provide hover text describing their purpose and accepted
connection types.

#### Review modules (Review)

| Frontend module | Inputs | Outputs | Purpose and rules |
| --- | --- | --- | --- |
| `Human Review` | `Review Target` | `Reviewed Data` | Human inspection of video, keypoints, or annotation results before downstream processing. |
| `AI Quality Review` | `Quality Review Target` | `Reviewed Data` | Checks video decoding, frame continuity, black screens, freezes, annotation coverage, and sensor data. |

#### Export modules (Export)

| Frontend module | Inputs | Outputs | Purpose and rules |
| --- | --- | --- | --- |
| `LeRobot Export` | `Exportable Data` | `Dataset` | Exports LeRobot v2.1 or v3.0; the version is selected on the node. |
| `HDF5 Export` | `Exportable Data` | `Dataset` | Exports HDF5 with the configured compression parameters. |

#### Workflow Studio UI components

<details>
<summary>Click to expand: frontend component reference</summary>

| UI component | Function |
| --- | --- |
| `NodePalette` | Displays Input, Process, Review, and Export categories; supports search, category collapse, and hover descriptions. |
| `WorkflowCanvas` | Supports drag-and-drop node creation, typed-port connections, node movement, selection, zoom, grid snapping, and minimap navigation. |
| `WorkflowNode` | Renders the card title, ports, project device selector/input, `Spatial` button, and API settings button. |
| `WorkflowDrawer` | Lists workflows, creates new workflows, loads workflows, and refreshes project-specific input sources. |
| `PipelineToolbar` | Provides New, Save, Save As, workflow JSON Export, and Run actions; the yellow dot indicates unsaved changes. |
| `NodeSettingsModal` | Configures AI Annotation API vendor, model, endpoint, and key fields. |
| `DeletableEdge` | Renders data connections; a selected edge can be deleted. |

Device selection is always scoped to the current project. A new empty workflow does not display a
device from another project. After data is uploaded, each card's selector shows only real sources
matching that card's RGB, RGB-D, stereo, or glove category.

</details>

### Review and execution rules

Connect inputs and outputs by their port names. An unconnected processing branch is skipped and
does not affect other branches. RGB-D 3D produces true metric 3D only when both RGB Video and
Depth are available.

Node states are limited to:

```text
Connected
Processing
Completed
Skipped · Missing Input
Completed with Warning
```

</details>

## 6. Processing result storage

<details>
<summary>Click to expand: processing results, upload extraction, and cleanup rules</summary>

When a Worker finishes processing, its results are merged into the corresponding Episode:

```text
data/chunk-000/episode_000000.parquet
  └── 2D keypoints, metric 3D, sensor, and annotation fields

meta/episodes/chunk-000/episode_000000.parquet
  └── processing_*, run_id, node states, and output indexes
```

Original RGB/Depth videos are not overwritten. Hand-keypoint previews are overlaid on the original
RGB video in the browser; a second skeleton video is not saved. System-level run records are kept
under `data/state/runs/`, and temporary export products under `data/state/exports/`; neither is
part of the project dataset directory.

Glove tactile data is kept as fixed-size numeric arrays in the Parquet data, for example
`observation.tactile.left` and `observation.tactile.right` with shape `[256]`. The current
LeRobot training export does not generate or retain tactile MP4 files. Tactile visualization is a
frontend preview concern; it is not required as a video feature by the LeRobot export.

The export module supports LeRobot v2.1 and v3.0. A v3.0 export is directly readable by the
current official LeRobot reader. A v2.1 export follows the official v2.1 layout and must be
converted with the official `convert_dataset_v21_to_v30` tool before using a v3.0 reader.
Batch export creates one dataset directory per selected Episode or project selection; each output
contains only the canonical `data/`, `meta/`, and `videos/` directories.

The upload archive lifecycle is: receive → extract → normalize → write to project → validate →
update status. Temporary upload files and extraction directories are cleaned only after extraction,
atomic commit, and validation all succeed. On failure, staging content is retained for retry and
diagnosis.

</details>

## 7. Installation and startup

### Linux

```bash
cd "Data Acquisition"
cp .env.example .env
# Edit .env and set API_KEY, WORKER_API_KEY, JWT_SECRET, and storage settings.
chmod +x deploy.sh
./deploy.sh
```

Common options:

```bash
python deploy.py --check-only       # Check only; do not install, change config, or start services.
python deploy.py --skip-vllm        # Do not start local VLM; use an API for AI annotation.
python deploy.py --no-services      # Run in the foreground without installing startup services.
```

The current AI annotation deployment uses an API. Use `--skip-vllm`, then configure the API vendor,
model, and endpoint in the `AI Annotation` settings in Workflow Studio.

### Windows

```bat
cd /d "Data Acquisition"
copy .env.example .env && deploy.bat
```

### Start the backend manually

```bash
cd "Data Acquisition"
source .venv-linux/bin/activate
export PYTHONPATH="$PWD"
./scripts/run_backend_linux.sh
```

Alternatively:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Start the Worker

```bash
cd "Data Acquisition"
export EGODATA_SERVER_URL=http://127.0.0.1:8000
export EGODATA_WORKER_API_KEY='same Worker API key as the server'
./scripts/run_worker_linux.sh
```

On Windows, use `scripts/run_worker_windows.ps1`. Linux and Windows virtual environments are
independent and must not be copied between operating systems.

### Workflow Studio

```bash
cd "Data Acquisition/web/workflow-studio"
npm ci
npm run dev
```

Production build:

```bash
npm run build
```

The build output is written to `web/static/workflow-studio/` and served by FastAPI.

## 8. Configuration

<details>
<summary>Click to expand: environment variables and security configuration</summary>

Configuration is stored in `Data Acquisition/.env` or environment variables and must not be
committed:

| Setting | Purpose |
| --- | --- |
| `STORAGE_DIR` | Root directory for data and system state |
| `STORAGE_BACKEND` | `local` or `sftp` |
| `SFTP_*` | Remote data directory and SSH connection settings |
| `API_KEY` | API key for the capture client and protected endpoints |
| `WORKER_API_KEY` | API key used by the Worker to claim and return jobs |
| `JWT_SECRET` | Secret used to sign Web login sessions |
| `UPLOAD_STAGING_DIR` | Local staging directory for upload archives |
| `HOST` / `PORT` | Web/API listen address and port |
| `PUBLIC_BASE_URL` | Public service URL |

See [`.env.example`](.env.example) for the complete template. Production environments must use
random secrets and restrict `.env` permissions. Never put passwords, tokens, API keys, or database
credentials in logs, workflow JSON, frontend code, or documentation.

</details>

### GitHub publication checklist

Before publishing this directory:

```bash
git status --short --ignored
git check-ignore -v .env data .backups .venv-linux models
git ls-files | rg -i '(^|/)(\.env|.*secret.*|.*token.*|.*credential.*|.*password.*|.*\.pem|.*\.key)$'
```

The repository excludes runtime datasets, temporary files, local state, backups, virtual
environments, archives, and secrets through `.gitignore`. Keep `.env` local and commit only
`.env.example`. Do not use `git add -f` for `data/`, `.backups/`, model weights, or generated
exports. If a real credential was ever placed in a tracked file, rotate it before publishing;
removing it from the working tree alone does not remove it from Git history.

## 9. API

<details>
<summary>Click to expand: primary APIs and OpenAPI documentation</summary>

After startup, these endpoints are available:

```text
GET /health
GET /docs
GET /openapi.json
```

Primary API groups:

```text
/api/v1/auth/*          Login and authentication
/api/v1/projects/*      Projects and Episodes
/api/v1/workflows/*     Workflow definitions
/api/v1/session/*       Capture-client uploads
/api/v1/video/*         Video, depth, and keypoint previews
/api/v1/annotations/*   Annotation and review
/api/v1/export/*        Dataset export
/api/v1/worker/*        Worker job claiming and callbacks
```

The running FastAPI OpenAPI document is the authoritative API contract.

</details>

## 10. Testing and troubleshooting

<details>
<summary>Click to expand: test commands and common troubleshooting steps</summary>

Backend and Worker checks:

```bash
cd "Data Acquisition"
python -m compileall app worker
python deploy.py --check-only
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_depth_codec.py
```

Frontend check:

```bash
cd "Data Acquisition/web/workflow-studio"
npm run build
```

Troubleshooting order:

1. Open `/health` and confirm that the service and storage are available;
2. check that `STORAGE_DIR` points to the actual data directory;
3. inspect the upload status and the actual files for the current Episode;
4. check that the Worker is online, can claim jobs, and sends heartbeats;
5. verify that Episode numbering matches across `data`, `meta/episodes`, and `videos`;
6. for RGB-D 3D nodes, verify that `RGB Video` and `Depth` are connected separately;
7. if depth preview is incorrect, verify that the video uses `gray12le` and is not a pseudo-color
   preview saved as source depth data.

</details>

## 11. Versioning

<details>
<summary>Click to expand: version maintenance rules</summary>

- The backend version is maintained in `app/version.py`;
- the Web frontend version is maintained in `web/workflow-studio/package.json`;
- increment the patch version for fixes and small adjustments, the minor version for new features,
  and the major version for architectural changes;
- before release, record the software version, data format, and API changes together.

Current versions: backend `1.6.8`, Web frontend `1.5.6`.

</details>

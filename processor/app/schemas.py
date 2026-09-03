"""Pydantic v2 schemas."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID
from pydantic import BaseModel, Field, Field


class EpisodeStart(BaseModel):
    task_description: Optional[str] = None
    name: Optional[str] = None
    fps: int = 30
    meta: dict[str, Any] = Field(default_factory=dict)


class EpisodeStartResponse(BaseModel):
    episode_id: UUID
    status: str = "collecting"


class FrameUpload(BaseModel):
    frame_index: int
    timestamp: Optional[float] = None
    observation: dict[str, Any] = Field(default_factory=dict)
    action: dict[str, Any] = Field(default_factory=dict)
    reward: Optional[float] = None
    is_terminal: bool = False
    is_truncated: bool = False


class FrameUploadResponse(BaseModel):
    frame_id: int
    image_paths: dict[str, str] = Field(default_factory=dict)


class EpisodeEndResponse(BaseModel):
    episode_id: UUID
    frame_count: int
    status: str = "completed"


class EpisodeOut(BaseModel):
    id: UUID
    session_id: Optional[UUID] = None
    name: Optional[str] = None
    task_description: Optional[str] = None
    task_id: Optional[int] = None
    status: str
    frame_count: int
    fps: int
    camera_names: Optional[list[str]] = None
    meta: Optional[dict[str, Any]] = None
    cleaning_report: Optional[dict[str, Any]] = None
    received_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None


class EpisodeListOut(BaseModel):
    episodes: list[EpisodeOut]
    total: int
    limit: int
    offset: int


class ExportStart(BaseModel):
    dataset_name: str
    episode_ids: Optional[list[UUID]] = None
    split_ratio: float = 0.9
    export_format: str = "lerobot_v3"


class BatchDownloadRequest(BaseModel):
    episode_ids: list[UUID] = Field(..., min_length=1)


class ExportJobOut(BaseModel):
    id: UUID
    dataset_name: str
    status: str
    episode_ids: Optional[list[UUID]] = None
    split_ratio: float
    progress: float
    output_dir: Optional[str] = None
    export_format: str
    error: Optional[str] = None
    created_at: datetime
    finished_at: Optional[datetime] = None


class HealthOut(BaseModel):
    status: str = "ok"
    db: str = "ok"
    storage: str = "ok"
    version: str = "0.1.0"


class MessageOut(BaseModel):
    message: str
    detail: Optional[str] = None


# ── Annotation Schemas ────────────────────────────────────────

class AnnotationKeyframeCreate(BaseModel):
    frame_index: int = Field(..., ge=0)
    event: Optional[str] = None


class AnnotationKeyframeOut(BaseModel):
    id: UUID
    segment_id: Optional[UUID] = None
    episode_id: UUID
    frame_index: int
    event: Optional[str] = None
    created_at: datetime


class AnnotationSegmentCreate(BaseModel):
    label: str = Field(..., min_length=1, max_length=256)
    start_frame_index: int = Field(..., ge=0)
    end_frame_index: int = Field(..., ge=0)
    color: Optional[str] = "#3B82F6"
    sort_order: int = 0
    notes: Optional[str] = None
    keyframes: list[AnnotationKeyframeCreate] = Field(default_factory=list)


class AnnotationSegmentUpdate(BaseModel):
    label: Optional[str] = None
    start_frame_index: Optional[int] = None
    end_frame_index: Optional[int] = None
    color: Optional[str] = None
    sort_order: Optional[int] = None
    notes: Optional[str] = None


class AnnotationSegmentOut(BaseModel):
    id: UUID
    episode_id: UUID
    label: str
    start_frame_index: int
    end_frame_index: int
    color: Optional[str] = None
    sort_order: int
    notes: Optional[str] = None
    keyframes: list[AnnotationKeyframeOut] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class AnnotationListOut(BaseModel):
    annotations: list[AnnotationSegmentOut]
    total: int
    episode_id: UUID


class AnnotationPerFrameOut(BaseModel):
    episode_id: UUID
    frame_count: int
    fps: int
    per_frame: list[dict[str, Any]]  # [{frame_index, annotation, annotation_index}, ...]


# ── Task Definitions ────────────────────────────────────────

class TaskDefinitionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: Optional[str] = None
    claimer: Optional[str] = Field(None, max_length=128)
    target_episodes: int = Field(0, ge=0)
    params: Optional[dict[str, Any]] = None
    status: str = Field("active")


class TaskDefinitionUpdate(BaseModel):
    description: Optional[str] = None
    claimer: Optional[str] = Field(None, max_length=128)
    target_episodes: Optional[int] = Field(None, ge=0)
    params: Optional[dict[str, Any]] = None
    status: Optional[str] = None


class TaskDefinitionOut(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    claimer: Optional[str] = None
    target_episodes: int
    params: Optional[dict[str, Any]] = None
    status: str
    current_episodes: int = 0
    episode_progress_pct: float = 0.0
    is_complete: bool = False
    last_upload_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class TaskDefinitionListOut(BaseModel):
    definitions: list[TaskDefinitionOut]
    total: int


# ── Devices ─────────────────────────────────────────────────

class DeviceHeartbeat(BaseModel):
    device_name: str = Field(..., min_length=1, max_length=128)
    meta: Optional[dict[str, Any]] = None


class DeviceOut(BaseModel):
    id: UUID
    name: str
    status: str
    meta: Optional[dict[str, Any]] = None
    first_seen_at: datetime
    last_seen_at: Optional[datetime] = None


class DeviceListOut(BaseModel):
    devices: list[DeviceOut]


class DeviceTaskOut(BaseModel):
    id: str  # task definition id as string
    name: str
    description: Optional[str] = None
    status: str
    total_required: int
    current_count: int
    assigned_at: Optional[datetime] = None
    params: Optional[dict[str, Any]] = None


# ── Dashboard Schemas ────────────────────────────────────

class StatCard(BaseModel):
    total: int
    today: int = 0
    label: str = ""


class DashboardOverview(BaseModel):
    total: StatCard
    reviewing: StatCard
    approved: StatCard
    failed: StatCard
    active_tasks: StatCard = StatCard(total=0, today=0, label="Active Tasks")
    datasets: StatCard = StatCard(total=0, today=0, label="Datasets")
    updated_at: str


class TaskProgressOut(BaseModel):
    task_name: str
    current_episodes: int
    target_episodes: int
    progress_pct: float
    status: str


class RecentEpisodeOut(BaseModel):
    id: UUID
    task_name: str
    status: str
    received_at: Optional[str] = None
    duration_sec: float = 0.0
    frame_count: int = 0
    cleaning_passed: Optional[bool] = None


class PipelineStage(BaseModel):
    name: str
    count: int


class PipelineStatus(BaseModel):
    stages: list[PipelineStage]


# ── Hand Keypoints ────────────────────────────────────

class HandKeypointData(BaseModel):
    """Per-hand keypoint + gesture data for a single frame."""
    keypoints: Optional[list[list[float]]] = None   # 21 × 2 pixel coords
    gesture: Optional[str] = None
    extended: Optional[list[str]] = None
    extended_count: Optional[int] = None
    fist: Optional[bool] = None
    pinch: Optional[bool] = None
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    motion: Optional[float] = None


class HandKeypointsOut(BaseModel):
    """Per-frame hand keypoint response for skeleton overlay."""
    frame_index: int
    source: str = "none"                            # detected | interpolated | none
    hand_0: Optional[HandKeypointData] = None
    hand_1: Optional[HandKeypointData] = None
    two_hand_distance: Optional[float] = None
    contact: Optional[str] = None
    interpolated_from: Optional[int] = None          # source=interpolated
    distance_frames: Optional[int] = None


# ── Workflow Studio ─────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = None
    workflow_id: Optional[UUID] = None
    device_type: Optional[str] = None
    params: dict = Field(default_factory=dict)


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=256)
    description: Optional[str] = None
    workflow_id: Optional[UUID] = None
    device_type: Optional[str] = None
    params: Optional[dict] = None
    status: Optional[str] = None


class ProjectOut(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    workflow_id: Optional[UUID] = None
    workflow_name: Optional[str] = None
    device_type: Optional[str] = None
    params: dict = Field(default_factory=dict)
    status: str
    task_count: int = 0
    episode_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProjectListOut(BaseModel):
    projects: list[ProjectOut]
    total: int


class WorkflowCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = None
    graph: dict = Field(default_factory=dict, description="React Flow JSON: {nodes, edges}")
    node_configs: dict = Field(default_factory=dict)


class WorkflowUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=256)
    description: Optional[str] = None
    graph: Optional[dict] = None
    node_configs: Optional[dict] = None
    status: Optional[str] = None


class WorkflowOut(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    graph: dict
    node_configs: dict
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class WorkflowListItem(BaseModel):
    id: UUID
    name: str
    status: str
    updated_at: datetime

    model_config = {"from_attributes": True}


class WorkflowListOut(BaseModel):
    workflows: list[WorkflowListItem]
    total: int
    limit: int
    offset: int


class WorkflowRunOut(BaseModel):
    id: UUID
    workflow_id: UUID
    episode_id: Optional[UUID] = None
    status: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    node_states: dict
    error_log: Optional[str] = None
    worker_id: Optional[str] = None
    heartbeat_at: Optional[datetime] = None
    attempt: int = 0
    progress: float = 0.0
    outputs: dict = Field(default_factory=dict)
    created_at: datetime

    model_config = {"from_attributes": True}


class WorkflowModuleOut(BaseModel):
    type: str
    slug: str
    version: str
    category: str
    label: str
    icon: str
    color: str
    inputs: list[dict] = Field(default_factory=list)
    outputs: list[dict] = Field(default_factory=list)
    defaultConfig: dict = Field(default_factory=dict)
    configSchema: list[dict] = Field(default_factory=list)
    executionTarget: str = "worker"
    capabilities: list[str] = Field(default_factory=list)


class WorkerClaimRequest(BaseModel):
    worker_id: str = Field(..., min_length=1, max_length=128)
    capabilities: list[str] = Field(default_factory=list)
    device: str = "auto"


class WorkerHeartbeatRequest(BaseModel):
    worker_id: str
    progress: float = Field(0.0, ge=0.0, le=1.0)
    node_states: dict = Field(default_factory=dict)


class WorkerFailureRequest(BaseModel):
    worker_id: str
    error: str
    retry: bool = True


class WorkerJobOut(BaseModel):
    run_id: UUID
    workflow_id: UUID
    episode_id: UUID
    workflow_name: str
    graph: dict
    node_configs: dict = Field(default_factory=dict)
    attempt: int = 0
    video_path: Optional[str] = None
    video_paths: dict[str, str] = Field(default_factory=dict)
    cameras: list[str] = Field(default_factory=list)
    camera: Optional[str] = None
    fps: int = 30
    input_url: str


class WorkerCompletionOut(BaseModel):
    run_id: UUID
    status: str
    outputs: dict = Field(default_factory=dict)

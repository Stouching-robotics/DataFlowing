"""SQLAlchemy ORM models — PostgreSQL."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, Text, Boolean, DateTime,
    ForeignKey, JSON, BigInteger, ARRAY
)
from sqlalchemy.dialects.postgresql import UUID as PUUID
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(PUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(256), nullable=True)
    project_id = Column(PUUID(as_uuid=True), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True)
    codebase_version = Column(String(16), nullable=True)
    info = Column(JSONB, nullable=True)
    episode_count = Column(Integer, default=0)
    original_archive = Column(String(512), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    episodes = relationship("Episode", back_populates="session", cascade="all, delete-orphan")


class Episode(Base):
    __tablename__ = "episodes"

    id = Column(PUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(PUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    name = Column(String(256), nullable=True)
    task_description = Column(Text, nullable=True)
    task_id = Column(Integer, nullable=True)
    # Status: collecting | completed | reviewed (legacy)
    #          received | processing | to_review | approved | rejected | failed (new pipeline)
    status = Column(String(16), default="collecting", index=True)
    frame_count = Column(Integer, default=0)
    fps = Column(Integer, default=30)
    camera_names = Column(JSONB, nullable=True)  # ["cam_front", "cam_wrist"]
    meta = Column(JSONB, nullable=True)
    # Pipeline timestamps
    received_at = Column(DateTime(timezone=True), nullable=True, index=True)      # 上传完成时间
    processing_started_at = Column(DateTime(timezone=True), nullable=True)  # 开始处理
    review_ready_at = Column(DateTime(timezone=True), nullable=True)  # 处理完成，待审核
    approved_at = Column(DateTime(timezone=True), nullable=True)      # 审核通过
    rejected_at = Column(DateTime(timezone=True), nullable=True)      # 审核拒绝
    # Cleaning / validation report
    cleaning_report = Column(JSONB, nullable=True)  # {passed, checks: [{name, status, detail}], summary}
    created_at = Column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    deleted_at = Column(DateTime(timezone=True), nullable=True)  # 软删除时间戳

    session = relationship("Session", back_populates="episodes")
    frames = relationship("Frame", back_populates="episode", cascade="all, delete-orphan")
    annotations = relationship("AnnotationSegment", back_populates="episode", cascade="all, delete-orphan")


class Frame(Base):
    __tablename__ = "frames"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    episode_id = Column(PUUID(as_uuid=True), ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False, index=True)
    frame_index = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=True)
    observation = Column(JSONB, nullable=True)
    action = Column(JSONB, nullable=True)
    reward = Column(Float, nullable=True)
    is_terminal = Column(Boolean, default=False)
    is_truncated = Column(Boolean, default=False)
    image_paths = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    episode = relationship("Episode", back_populates="frames")


class ExportJob(Base):
    __tablename__ = "export_jobs"

    id = Column(PUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_name = Column(String(128), nullable=False)
    status = Column(String(16), default="pending")  # pending | running | completed | failed
    episode_ids = Column(ARRAY(PUUID(as_uuid=True)), nullable=True)
    split_ratio = Column(Float, default=0.9)
    progress = Column(Float, default=0.0)
    output_dir = Column(String(512), nullable=True)
    export_format = Column(String(16), default="lerobot_v3")
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    finished_at = Column(DateTime(timezone=True), nullable=True)


class AnnotationSegment(Base):
    """Frame-precise annotation segment — a labeled range of frames within an episode."""

    __tablename__ = "annotation_segments"

    id = Column(PUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    episode_id = Column(PUUID(as_uuid=True), ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False, index=True)
    label = Column(String(256), nullable=False)
    start_frame_index = Column(Integer, nullable=False)
    end_frame_index = Column(Integer, nullable=False)
    color = Column(String(7), nullable=True, default="#3B82F6")  # hex color
    sort_order = Column(Integer, default=0)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    episode = relationship("Episode", back_populates="annotations")
    keyframes = relationship("AnnotationKeyframe", back_populates="segment", cascade="all, delete-orphan")


class AnnotationKeyframe(Base):
    """A single keyframe marker within an annotation segment — marks a critical moment."""

    __tablename__ = "annotation_keyframes"

    id = Column(PUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    segment_id = Column(PUUID(as_uuid=True), ForeignKey("annotation_segments.id", ondelete="CASCADE"), nullable=True, index=True)
    episode_id = Column(PUUID(as_uuid=True), ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False, index=True)
    frame_index = Column(Integer, nullable=False)
    event = Column(String(256), nullable=True)  # e.g. "手指接触杯身"
    created_at = Column(DateTime(timezone=True), default=utcnow)

    segment = relationship("AnnotationSegment", back_populates="keyframes")


class TaskDefinition(Base):
    """Pre-defined collection task with targets. Matches uploaded episodes by task_description."""

    __tablename__ = "task_definitions"

    id = Column(PUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(256), nullable=False, unique=True, index=True)  # matches episode.task_description
    description = Column(Text, nullable=True)
    claimer = Column(String(128), nullable=True, index=True)  # who claims this task (e.g. HandTest_001)
    target_episodes = Column(Integer, default=0)       # target number of batches (1 zip upload = 1 batch)
    target_frames = Column(Integer, default=0)         # (deprecated, kept for compat)
    target_duration_sec = Column(Integer, default=0)   # (deprecated, kept for compat)
    fps = Column(Integer, default=30)
    params = Column(JSONB, nullable=True)               # extra params, e.g. {"object":"cylinder","hand":"right"}
    status = Column(String(16), default="active")       # active | paused | completed
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Device(Base):
    """Registered data-collection device (robot / test rig / DAQ client)."""

    __tablename__ = "devices"

    id = Column(PUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(128), nullable=False, unique=True, index=True)
    meta = Column(JSONB, nullable=True)
    last_seen_at = Column(DateTime(timezone=True), default=utcnow)
    first_seen_at = Column(DateTime(timezone=True), default=utcnow)
    status = Column(String(16), default="online")       # online | offline
    created_at = Column(DateTime(timezone=True), default=utcnow)


class User(Base):
    """Platform user accounts."""

    __tablename__ = "users"

    id = Column(PUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    email = Column(String(128), nullable=True)
    role = Column(String(16), default="engineer")  # admin | reviewer | engineer
    status = Column(String(16), default="active")  # active | disabled
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


# ── Workflow Studio ─────────────────────────────────────

class Project(Base):
    """A collection project — binds one workflow to a device/task flow.

    Example: "鱼眼相机采集项目" binds the fisheye pipeline (fisheye_camera
    → mediapipe_hand → export) so uploads under this project automatically
    run its workflow. Calibration params follow the LeRobot dataset layout
    (calibration/<camera>.json), so the worker reads them without extra input.
    """

    __tablename__ = "projects"

    id = Column(PUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(256), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    workflow_id = Column(PUUID(as_uuid=True), ForeignKey("workflows.id", ondelete="SET NULL"), nullable=True, index=True)
    device_type = Column(String(64), nullable=True)   # e.g. "fisheye" | "stereo" | "glove"
    params = Column(JSONB, nullable=True)             # project-level defaults, e.g. {"position": "head"}
    status = Column(String(16), default="active", index=True)  # active | paused | archived
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    workflow = relationship("Workflow")


class Workflow(Base):
    """A saved workflow pipeline — stores the React Flow graph as JSON."""

    __tablename__ = "workflows"

    id = Column(PUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    graph = Column(JSONB, nullable=False, default=dict)        # {nodes: [...], edges: [...]}
    node_configs = Column(JSONB, nullable=False, default=dict)  # per-node config overrides
    status = Column(String(16), default="draft", index=True)    # draft | active | archived
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    runs = relationship("WorkflowRun", back_populates="workflow", cascade="all, delete-orphan")


class WorkflowRun(Base):
    """A single execution run of a workflow."""

    __tablename__ = "workflow_runs"

    id = Column(PUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id = Column(PUUID(as_uuid=True), ForeignKey("workflows.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    episode_id = Column(PUUID(as_uuid=True), ForeignKey("episodes.id", ondelete="SET NULL"),
                        nullable=True, index=True)
    status = Column(String(16), default="queued", index=True)  # queued | running | completed | failed
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    node_states = Column(JSONB, nullable=False, default=dict)   # {node_id: {status, progress}}
    error_log = Column(Text, nullable=True)
    worker_id = Column(String(128), nullable=True, index=True)
    lease_until = Column(DateTime(timezone=True), nullable=True, index=True)
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    attempt = Column(Integer, nullable=False, default=0)
    progress = Column(Float, nullable=False, default=0.0)
    outputs = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    workflow = relationship("Workflow", back_populates="runs")

"""Async database engine — PostgreSQL via asyncpg."""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from app.config import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    connect_args={
        "timeout": settings.DB_CONNECT_TIMEOUT,
        "command_timeout": settings.DB_COMMAND_TIMEOUT,
    },
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def create_database_if_not_exists():
    """Create data_acq database if missing."""
    import asyncpg
    from urllib.parse import urlparse

    u = urlparse(settings.DATABASE_URL.replace("+asyncpg", ""))
    admin_url = f"postgresql://{u.username}:{u.password}@{u.hostname}:{u.port or 5432}/postgres"

    try:
        conn = await asyncpg.connect(admin_url)
    except Exception as exc:
        # Some deployments expose only the application database through the
        # SSH tunnel and reject connections to the postgres maintenance DB.
        # The application database is already provisioned in that setup, so
        # let init_db() continue and verify it through the main engine.
        print(f"[DB] Skipping maintenance-database check: {exc}")
        return
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname=$1", u.path.lstrip("/"))
        if not exists:
            await conn.execute(f'CREATE DATABASE "{u.path.lstrip("/")}"')
            print(f"[DB] Created database: {u.path.lstrip('/')}")
    finally:
        await conn.close()


async def init_db():
    # Import model classes before create_all(); otherwise SQLAlchemy's metadata
    # is empty and a fresh EGOData database receives no tables.
    from app import models  # noqa: F401

    await create_database_if_not_exists()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Ensure new columns exist on existing tables (create_all won't add columns)
    await _migrate_episode_columns()
    await _migrate_task_definition_columns()
    await _migrate_device_columns()
    await _migrate_workflow_run_columns()
    await _migrate_session_columns()
    await _migrate_user_columns()
    await _seed_default_users()
    print("[DB] Tables created/verified")


async def _migrate_episode_columns():
    """Add new pipeline columns to episodes table if they don't exist yet."""
    new_columns = [
        ("received_at", "TIMESTAMP WITH TIME ZONE"),
        ("processing_started_at", "TIMESTAMP WITH TIME ZONE"),
        ("review_ready_at", "TIMESTAMP WITH TIME ZONE"),
        ("approved_at", "TIMESTAMP WITH TIME ZONE"),
        ("rejected_at", "TIMESTAMP WITH TIME ZONE"),
        ("cleaning_report", "JSONB"),
    ]
    async with engine.begin() as conn:
        for col_name, col_type in new_columns:
            try:
                await conn.execute(
                    text(f"ALTER TABLE episodes ADD COLUMN IF NOT EXISTS {col_name} {col_type}")
                )
            except Exception:
                pass  # column already exists or DB doesn't support IF NOT EXISTS


async def _migrate_task_definition_columns():
    """Add new columns to task_definitions table."""
    new_columns = [
        ("claimer", "VARCHAR(128)"),
        ("params", "JSONB"),
    ]
    async with engine.begin() as conn:
        for col_name, col_type in new_columns:
            try:
                await conn.execute(
                    text(f"ALTER TABLE task_definitions ADD COLUMN IF NOT EXISTS {col_name} {col_type}")
                )
            except Exception:
                pass


async def _migrate_device_columns():
    """Add columns to devices table if they don't exist yet."""
    new_columns = [
        ("meta", "JSONB"),
        ("first_seen_at", "TIMESTAMP WITH TIME ZONE"),
    ]
    async with engine.begin() as conn:
        for col_name, col_type in new_columns:
            try:
                await conn.execute(
                    text(f"ALTER TABLE devices ADD COLUMN IF NOT EXISTS {col_name} {col_type}")
                )
            except Exception:
                pass


async def _migrate_workflow_run_columns():
    """Add worker queue columns to existing workflow_runs tables."""
    new_columns = [
        ("episode_id", "UUID"),
        ("worker_id", "VARCHAR(128)"),
        ("lease_until", "TIMESTAMP WITH TIME ZONE"),
        ("heartbeat_at", "TIMESTAMP WITH TIME ZONE"),
        ("attempt", "INTEGER NOT NULL DEFAULT 0"),
        ("progress", "DOUBLE PRECISION NOT NULL DEFAULT 0"),
        ("outputs", "JSONB NOT NULL DEFAULT '{}'::jsonb"),
    ]
    async with engine.begin() as conn:
        for col_name, col_type in new_columns:
            try:
                await conn.execute(text(
                    f"ALTER TABLE workflow_runs ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
                ))
            except Exception:
                pass


async def _migrate_session_columns():
    """Add project binding column to sessions table."""
    async with engine.begin() as conn:
        try:
            await conn.execute(text(
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS project_id UUID"
            ))
        except Exception:
            pass


async def _migrate_user_columns():
    """Add account lifecycle fields to existing EGOData installations."""
    new_columns = [
        ("expires_at", "TIMESTAMP WITH TIME ZONE"),
        ("last_login_at", "TIMESTAMP WITH TIME ZONE"),
    ]
    async with engine.begin() as conn:
        for col_name, col_type in new_columns:
            try:
                await conn.execute(text(
                    f"ALTER TABLE users ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
                ))
            except Exception:
                pass


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session


async def _seed_default_users():
    """Insert the configurable bootstrap administrator if the table is empty.

    The public template uses a configurable demo account for a first local demo only. Existing
    databases are never modified because seeding stops when any user exists.
    Production deployments should set both bootstrap environment variables to
    private values before the first startup.
    """
    import os
    from app.models import User
    from app.auth import hash_password
    from sqlalchemy import select, func

    async with async_session() as db:
        cnt = (await db.execute(select(func.count(User.id)))).scalar() or 0
        if cnt > 0:
            return  # Already seeded

        username = (os.environ.get("EGODATA_BOOTSTRAP_USERNAME", "demo-admin").strip()
                    or "demo-admin")
        password = os.environ.get("EGODATA_BOOTSTRAP_PASSWORD", "change-me")
        email = (os.environ.get("EGODATA_BOOTSTRAP_EMAIL", "").strip()
                 or f"{username.lower()}@egodata.local")
        defaults = [User(
            username=username,
            password_hash=hash_password(password),
            email=email,
            role="admin",
            status="active",
        )]
        db.add_all(defaults)
        await db.commit()
        print(f"[DB] Seeded {len(defaults)} default users")

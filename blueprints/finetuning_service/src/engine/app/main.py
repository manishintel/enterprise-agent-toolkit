# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Fine-tuning engine — HTTP surface.

This process is CPU-only. It owns the job records and the Kubernetes Jobs that
do the training; it never loads a model itself. See app/services/k8s_runtime.py
for the dispatch side and trainer/ for what runs on the GPU.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from sqlalchemy import text
import asyncio
import logging
import time
from datetime import datetime, timezone

from app.routers import jobs_router
from app.database import db_engine, Base
from app.config import settings, GPU_INFO
from app.limiter import limiter

# Configure logging
logger = logging.getLogger("uvicorn")

_PROGRESS_FLUSH_INTERVAL = 5.0
_progress_flusher: "asyncio.Task | None" = None


# ---------------------------------------------------------------------------
# Security Headers Middleware
# ---------------------------------------------------------------------------

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add OWASP-recommended security headers to every HTTP response."""

    # Paths that serve Swagger/ReDoc UI — need relaxed CSP to load JS/CSS.
    _DOCS_PATHS = {"/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )
        if request.url.path in self._DOCS_PATHS:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data: https://fastapi.tiangolo.com; "
                "frame-ancestors 'none'"
            )
        else:
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; frame-ancestors 'none'"
            )
        return response


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

async def _probe_gpu_cluster() -> None:
    """Fill GPU_INFO from the GPU cluster, in place.

    Nothing here is fatal. The service has to come up even when the GPU cluster
    is unreachable, because /availability reporting "no GPU" with a reason is far
    more useful to a caller than a pod stuck in CrashLoopBackOff.
    """
    from app.config import get_gpu_info

    try:
        await asyncio.to_thread(get_gpu_info)
    except Exception as e:  # noqa: BLE001 - startup must not depend on the cluster
        logger.warning(f"GPU cluster probe failed: {e}")
        return

    if GPU_INFO.get("available"):
        logger.info(
            f"GPU cluster: {GPU_INFO.get('count')} concurrent slot(s) of "
            f"{GPU_INFO.get('name')} ({GPU_INFO.get('total_memory_gb') or 0:.0f} GB each), "
            f"namespace {GPU_INFO.get('namespace')} on {GPU_INFO.get('nodes')}"
        )
    else:
        logger.warning(
            f"No GPU capacity available: {GPU_INFO.get('reason')}. The service will "
            "accept no jobs until this clears."
        )


async def _publish_worker_bundle() -> None:
    """Publish the current trainer sources and sweep superseded ones.

    Done at startup so a broken bundle - a source file renamed without updating
    worker_bundle._LAYOUT, say - surfaces in the deployment log rather than on
    the first job an hour later.
    """
    from app.services import k8s_runtime, worker_bundle

    try:
        name = await asyncio.to_thread(worker_bundle.configmap_name)
    except FileNotFoundError as e:
        # The bundle is this service's other half; without it no job can run.
        logger.error(f"Trainer bundle is incomplete: {e}")
        raise

    logger.info(f"Trainer bundle: {name} ({len(worker_bundle.files())} file(s))")

    if not k8s_runtime.cluster_is_available():
        return
    try:
        pruned = await asyncio.to_thread(k8s_runtime.prune_worker_bundles, name)
        if pruned:
            logger.info(f"Pruned {pruned} superseded trainer bundle(s)")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Trainer bundle prune failed: {e}")


async def _reap_orphaned_training_jobs() -> None:
    """Delete training Jobs left behind by a previous instance of this process.

    In-flight jobs are driven from memory, so a restart loses the ability to
    follow them: their pods would keep a GPU slot the restarted service believes
    is free, and nothing would ever collect their results.
    """
    from app.services import k8s_runtime

    try:
        removed = await asyncio.to_thread(k8s_runtime.cleanup_orphaned_jobs)
        if removed:
            logger.info(f"Deleted {removed} orphaned training Job(s)")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Orphaned training Job cleanup failed: {e}")


async def _init_database() -> None:
    """Create tables, then apply additive DDL."""
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized successfully")

    # create_all only creates missing *tables*, so a column added to a model
    # after first deployment never appears on an existing database. There is no
    # migration tool here, so additive DDL goes in this list: keep every entry
    # idempotent and non-destructive, and never widen or drop an existing column
    # this way.
    try:
        async with db_engine.begin() as conn:
            for statement in (
                "ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS current_phase VARCHAR",
                "ALTER TABLE training_jobs ADD COLUMN IF NOT EXISTS num_train_epochs DOUBLE PRECISION",
            ):
                await conn.execute(text(statement))
    except Exception as e:
        # A missing progress column costs progress reporting, not training.
        logger.warning(f"Additive schema check failed: {e}")


async def _fail_orphaned_job_rows() -> None:
    """Close out job rows whose driving task died with the previous process."""
    from app.database import AsyncSessionLocal
    from app.models import TrainingJob
    from sqlalchemy.future import select

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(TrainingJob).where(TrainingJob.status.in_(["RUNNING", "PENDING"]))
            )
            orphaned_jobs = result.scalars().all()

            for job in orphaned_jobs:
                logger.warning(f"Cleaning up orphaned job {job.id} (status: {job.status})")
                job.status = "FAILED"
                job.error_log = "Job terminated due to service restart"
                job.completed_at = datetime.now(timezone.utc)
                if job.started_at:
                    job.elapsed_seconds = int((job.completed_at - job.started_at).total_seconds())

            if orphaned_jobs:
                await db.commit()
                logger.info(f"Cleaned up {len(orphaned_jobs)} orphaned jobs")
    except Exception as e:
        logger.warning(f"Orphaned job cleanup failed: {e}")


async def _flush_progress_periodically() -> None:
    """Write buffered per-step progress onto the job rows it belongs to.

    The trainer logs every step, so writing each record as it arrives would mean
    one UPDATE per step on the loop that is also relaying the worker's output.
    Coalescing into one pass every few seconds keeps that off the hot path while
    still bounding how stale a direct database read can be.

    Terminal rows are skipped: ``background_training_task`` owns those, and a
    step record can outlive the commit that finished the job.
    """
    from app.database import AsyncSessionLocal
    from app.models import TrainingJob
    from app.services import progress

    while True:
        await asyncio.sleep(_PROGRESS_FLUSH_INTERVAL)
        try:
            pending = progress.take_dirty()
            if not pending:
                continue

            async with AsyncSessionLocal() as db:
                written = 0
                for job_id, values in pending.items():
                    job = await db.get(TrainingJob, job_id)
                    if job is None or job.status in progress.TERMINAL_STATUSES:
                        continue
                    for field in progress.TRACKED_FIELDS:
                        value = values.get(field)
                        if value is not None:
                            setattr(job, field, value)
                    written += 1
                if written:
                    await db.commit()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Losing a flush costs freshness in the stored row and nothing else;
            # the next pass rewrites from the live values.
            logger.warning(f"Progress flush failed: {e}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Startup and shutdown, in the order they have to happen."""
    global _progress_flusher

    logger.info(f"Starting {settings.APP_NAME} v2.0.0")
    logger.info(f"Environment: {settings.ENV}")

    # The database first: without it nothing below has anywhere to record what it
    # did, and it is the one dependency worth refusing to start without.
    try:
        await _init_database()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

    await _probe_gpu_cluster()
    await _publish_worker_bundle()
    # Reap the cluster before the rows, so a Job deleted here is already
    # accounted for by the sweep that follows.
    await _reap_orphaned_training_jobs()
    await _fail_orphaned_job_rows()

    # Persist live training progress in the background. Read paths overlay the
    # in-memory values already, so this exists purely so the stored row is also
    # true for anything reading Postgres directly.
    _progress_flusher = asyncio.create_task(
        _flush_progress_periodically(), name="progress_flusher"
    )

    logger.info("Application startup complete")

    yield

    logger.info("Shutting down application...")

    if _progress_flusher is not None:
        _progress_flusher.cancel()
        try:
            await _progress_flusher
        except asyncio.CancelledError:
            pass

    # Release the GPUs. A training Job outlives this pod, and nothing would be
    # left to follow it: the restarted service would see slots as free while the
    # devices were still held.
    try:
        from app.services import k8s_runtime

        terminated = await asyncio.to_thread(k8s_runtime.terminate_all)
        if terminated:
            logger.info(f"Deleted {terminated} in-flight training Job(s)")
        logger.info("GPU resources released")
    except Exception as e:
        logger.warning(f"GPU cleanup failed: {e}")

    logger.info("Application shutdown complete")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version="2.0.0",
    description="Production-grade LLM fine-tuning service using Unsloth on NVIDIA GPUs",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Attach rate-limiter state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security headers on every response
app.add_middleware(SecurityHeadersMiddleware)

# CORS – origins are driven by the ALLOWED_ORIGINS environment variable so
# that production deployments are never accidentally left with wildcard access.
_allowed_origins = [
    origin.strip()
    for origin in settings.ALLOWED_ORIGINS.split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(jobs_router)

# Middleware for request logging and timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    try:
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"Status: {response.status_code} Time: {process_time:.3f}s"
        )

        return response
    except Exception as e:
        logger.error(f"Request failed: {request.method} {request.url.path} Error: {e}")
        raise

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if settings.ENV == "dev" else "An unexpected error occurred"
        }
    )


@app.get("/health")
async def health():
    """Liveness probe.

    Deliberately dependency-free. A restart cannot fix an unreachable database or
    GPU cluster, and it would abandon every job this process is driving, so
    neither belongs in the check that decides whether to kill the pod.
    """
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness probe: can this instance serve requests?

    The database is required - every endpoint touches it. The GPU cluster is not:
    a caller asking /availability why no jobs are being accepted needs a reply,
    and taking the pod out of service would give them a connection error instead.
    """
    try:
        async with db_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.warning(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": "database unreachable"},
        )

    return {
        "status": "ready",
        "gpu_available": GPU_INFO.get("available", False),
        "gpu_detail": GPU_INFO.get("reason"),
    }


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": settings.APP_NAME,
        "version": "2.0.0",
        "status": "operational",
        "gpu_available": GPU_INFO.get("available", False),
        "gpu_name": GPU_INFO.get("name"),
        "documentation": "/docs"
    }

@app.get("/info")
async def service_info():
    """Detailed service information"""
    return {
        "service": settings.APP_NAME,
        "version": "2.0.0",
        "environment": settings.ENV,
        "gpu_info": GPU_INFO,
        "config": {
            "max_concurrent_jobs": settings.MAX_CONCURRENT_JOBS,
            "gpu_memory_threshold_gb": settings.GPU_MEMORY_THRESHOLD_GB,
            "default_max_seq_length": settings.DEFAULT_MAX_SEQ_LENGTH,
            "train_namespace": settings.TRAIN_NAMESPACE,
            "train_image": settings.TRAIN_IMAGE,
        }
    }

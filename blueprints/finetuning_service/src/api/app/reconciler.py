"""
Background job status reconciler.

The job *detail* endpoint refreshes a job from the training engine on every read,
but the *list* endpoint answers straight from Postgres. Without this loop a row
therefore keeps whatever status it was given at submit time until somebody opens
that job's page, so the jobs list shows a stale status and a progress bar that
never moves — polling the list from the UI cannot fix that on its own, because it
only re-reads a row nobody has refreshed.

This runs one reconcile pass per interval for jobs that are still active, reusing
the adapter's own ``get_job_status`` so there is exactly one implementation of
"read the engine and write the row". Terminal jobs are never polled again.
"""

import asyncio
from typing import Any, Dict, List, Optional

from .adapter_config import build_adapter_config
from .adapters.base import ResourceAdapterFactory
from .config import get_settings
from .database import db_manager
from .observability import get_logger
from .schemas import JobStatusRequest, ResourceType

logger = get_logger(__name__)
settings = get_settings()

# Mirrors the statuses the UI treats as active.
ACTIVE_STATUSES = ("validating_files", "queued", "running")


async def _fetch_active_jobs() -> List[Dict[str, Any]]:
    """Jobs that are still moving and that the engine can be asked about."""
    return await db_manager.fetch_all(
        """
        SELECT id, resource_type, resource_job_id, user_id
        FROM fine_tuning_jobs
        WHERE status = ANY($1::VARCHAR[])
          AND resource_job_id IS NOT NULL
        ORDER BY created_at DESC
        """,
        list(ACTIVE_STATUSES),
        timeout=30,
    )


async def _reconcile_job(row: Dict[str, Any]) -> None:
    """Refresh one job. get_job_status writes the row as a side effect."""
    job_id = row["id"]
    try:
        resource_type = ResourceType(row["resource_type"])
    except ValueError:
        logger.warning(
            "Skipping reconcile for job with unknown resource type",
            extra={"job_id": job_id, "resource_type": row["resource_type"]},
        )
        return

    # The engine authenticates with the service's own client credentials, so the
    # owner is passed only to keep the adapter's user fields consistent with how
    # the job was submitted. "username" is deliberately absent rather than None:
    # build_adapter_config uses dict.get with a default, which a present-but-None
    # key would defeat.
    adapter_config = build_adapter_config(resource_type, {"user_id": row["user_id"]})
    adapter_config["api_timeout"] = 8.0

    adapter = ResourceAdapterFactory.create_adapter(resource_type, config=adapter_config)
    await adapter.get_job_status(
        JobStatusRequest(job_id=job_id, resource_job_id=row["resource_job_id"])
    )


async def reconcile_once() -> int:
    """Run a single pass. Returns how many jobs were refreshed."""
    rows = await _fetch_active_jobs()
    if not rows:
        return 0

    # Sequential on purpose: a handful of active jobs at most (the engine caps
    # concurrency), and this keeps a slow engine from opening one connection per
    # job on every tick.
    for row in rows:
        try:
            await _reconcile_job(row)
        except Exception as exc:  # one bad job must not stop the others
            logger.warning(
                "Job reconcile failed",
                extra={"job_id": row["id"], "error": f"{type(exc).__name__}: {exc}"},
            )
    return len(rows)


async def _reconcile_loop(interval: int) -> None:
    logger.info(f"📈 Job status reconciler started (every {interval}s)")
    while True:
        try:
            await asyncio.sleep(interval)
            refreshed = await reconcile_once()
            if refreshed:
                logger.debug(f"Reconciled {refreshed} active job(s)")
        except asyncio.CancelledError:
            logger.info("Job status reconciler stopped")
            raise
        except Exception as exc:
            # Never let the loop die: without it the jobs list silently goes stale.
            logger.error(
                f"Job reconcile pass failed: {type(exc).__name__}: {exc}", exc_info=True
            )


def start_reconciler() -> Optional[asyncio.Task]:
    """Start the loop unless it is disabled. Returns the task, if started."""
    if not settings.job_reconcile_enabled:
        logger.info("Job status reconciler disabled (job_reconcile_enabled=false)")
        return None
    return asyncio.create_task(
        _reconcile_loop(settings.job_reconcile_interval_seconds),
        name="job-status-reconciler",
    )


async def stop_reconciler(task: Optional[asyncio.Task]) -> None:
    """Cancel the loop and wait for it to unwind."""
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

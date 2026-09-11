# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Fine-tuning job endpoints.

The division of labour changed with the move to Kubernetes and is worth stating
plainly, because it is what most of this module's shape follows from: this
process owns the *job record* and nothing else. Dataset download, validation,
training, merging, watermarking and upload all happen inside the training pod
(see trainer/train_worker.py) on the GPU cluster. This process creates the Job,
relays its progress into the database, and reports the outcome.

That is not tidiness for its own sake. The API and the GPU are in different
clusters, so there is no shared filesystem across which to hand a 16GB merged
model, and routing the bytes through here would send them over the wire twice.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func
from datetime import datetime, timezone
import logging
import asyncio

from app.database import get_db, AsyncSessionLocal
from app.models import TrainingJob
from app.schemas import (
    TrainingRequest, JobStatusResponse, AvailabilityResponse,
    GPUStatusResponse, JobCancelResponse, JobLogsResponse
)
from app.auth import verify_access_token, get_files_api_token
from app.config import settings, GPU_INFO
from app.services import job_logs, k8s_runtime, progress
from app.validators.training_data_validator import (
    validate_model_allowlist,
    TrainingDataValidationError,
)
from app.middleware.model_extraction_detector import extraction_detector
from app.limiter import limiter

router = APIRouter(prefix="/finetune", tags=["Finetuning"])
logger = logging.getLogger("uvicorn")

# Store for job cancellation flags
job_cancellation_flags: dict[int, bool] = {}

# Store for active asyncio tasks so they can be properly cancelled on delete
active_training_tasks: dict[int, asyncio.Task] = {}


def _now() -> datetime:
    """Timezone-aware "now".

    The timestamp columns are DateTime(timezone=True); writing a naive value into
    one leaves the database to guess an offset, and the difference showed up as
    hours-long errors in the durations the UI renders.
    """
    return datetime.now(timezone.utc)


def update_job_progress_sync(job_id: int, progress_data: dict):
    """
    Receive one progress record from the trainer.

    Called from the event loop for every logged step (logging_steps=1), so it
    must not touch the database: it hands the record to the in-memory store,
    which read paths overlay and a periodic task persists. Before this existed
    the record was logged and dropped, which is why a finished job still read
    progress_percent=0 / current_step=0 / total_steps=NULL.
    """
    progress.record(job_id, progress_data)
    logger.info(f"Job {job_id} - Step {progress_data.get('current_step')}/{progress_data.get('max_steps')} - Loss: {progress_data.get('loss', 'N/A')}")


def _persist_last_progress(job: TrainingJob) -> None:
    """Freeze the live numbers onto a job that is about to become terminal.

    Once the status is terminal nothing overlays the row any more, so a job that
    failed or was cancelled mid-run would otherwise report the values it had at
    submission. Keeping the phase is the useful part: "failed during merging" is
    a different problem from "failed during downloading_data".
    """
    latest = progress.snapshot(job.id)
    if not latest:
        return
    for field in progress.TRACKED_FIELDS:
        value = latest.get(field)
        if value is not None:
            setattr(job, field, value)


def _job_response(job: TrainingJob) -> JobStatusResponse:
    """A job row with the in-flight numbers layered on top.

    The row is only written at the start and end of a job, so on its own it
    reports nothing useful mid-run. Overlaying at read time means a poller sees
    current values without waiting for the flush interval, and a terminal job is
    returned untouched.
    """
    data = JobStatusResponse.model_validate(job).model_dump()
    data.update(progress.overlay(job.id, job.status))
    return JobStatusResponse(**data)


def _mark_cancelled(job: TrainingJob, reason: str = "Job cancelled by user") -> None:
    """Close a job record out as cancelled."""
    job.status = "CANCELLED"
    job.error_log = reason
    job.completed_at = _now()
    if job.started_at:
        job.elapsed_seconds = int((job.completed_at - job.started_at).total_seconds())
    _persist_last_progress(job)
    job.current_phase = None


async def background_training_task(job_id: int, request: TrainingRequest, bearer_token: str):
    """Drive one job through the GPU cluster and record what happened."""
    async with AsyncSessionLocal() as db:
        job = await db.get(TrainingJob, job_id)

        try:
            # Check for cancellation
            if job_cancellation_flags.get(job_id, False):
                _mark_cancelled(job)
                await db.commit()
                return

            job.status = "RUNNING"
            job.started_at = _now()
            # Everything up to the trainer's first step is one phase as far as a
            # progress bar is concerned: scheduling the pod, pulling the image,
            # downloading the dataset, loading the model. The trainer takes over
            # the phase from its first [PROGRESS] record onwards.
            progress.set_phase(job_id, progress.PHASE_PREPARING)
            job.current_phase = progress.PHASE_PREPARING
            await db.commit()

            # --- AI/ML Security: Base model allowlist check ---
            # Cheap, and done here so an unapproved model is refused before a GPU
            # slot is taken. The trainer checks it again as the backstop.
            try:
                validate_model_allowlist(request.model_name)
            except TrainingDataValidationError as e:
                # Re-raise as a plain exception so it's caught by the outer try/except
                raise ValueError(str(e))

            def check_cancellation(jid: int) -> bool:
                return job_cancellation_flags.get(jid, False)

            # Hand the whole pipeline to a single-GPU Kubernetes Job. The dataset
            # id and the caller's Files API key go with it: the pod does its own
            # download and upload, so those bytes never come through here.
            logger.info(f"Job {job_id}: dispatching to the GPU cluster...")
            results = await k8s_runtime.execute_finetuning(
                model_name=request.model_name,
                input_file_id=request.input_filenames[0],
                username=request.username,
                params=request.hyperparameters,
                job_id=job_id,
                bearer_token=bearer_token,
                progress_callback=update_job_progress_sync,
                cancellation_check=check_cancellation,
            )

            # A cancelled run reports itself as a successful, deliberately
            # incomplete one: the trainer exits cleanly so Kubernetes records the
            # Job as Complete, and says so in the result.
            if results.get("cancelled") or job_cancellation_flags.get(job_id, False):
                _mark_cancelled(job)
                await db.commit()
                logger.info(f"Job {job_id}: cancelled")
                return

            output_file_id = results.get("output_file_id")
            if not output_file_id:
                # Trained but never published. Reporting COMPLETED here would
                # hand the user a job with no model to fetch.
                raise RuntimeError(
                    "Training finished but the trainer reported no uploaded model"
                )

            job.status = "COMPLETED"
            job.output_file_id = output_file_id
            job.output_path = results.get("model_path")
            job.completed_at = _now()
            job.elapsed_seconds = results.get("elapsed_seconds")
            job.training_loss = results.get("training_loss")
            job.dataset_size = results.get("dataset_size")

            # Final progress, from the trainer's own totals. The row is the only
            # source once the job is terminal, so it has to be complete here -
            # `overlay` deliberately stops contributing at this point.
            job.current_phase = None
            job.progress_percent = 100.0
            job.total_steps = results.get("total_steps") or job.total_steps
            job.num_train_epochs = results.get("num_train_epochs")
            if job.total_steps:
                job.current_step = job.total_steps

            if results.get("final_memory_gb"):
                job.gpu_memory_used_gb = results["final_memory_gb"].get("allocated_gb")
                job.gpu_utilization_percent = results["final_memory_gb"].get("utilization_percent")

            await db.commit()
            logger.info(f"Job {job_id}: Completed successfully, model {output_file_id}")

        except asyncio.CancelledError:
            logger.info(f"Job {job_id}: asyncio task was cancelled")
            _mark_cancelled(job)
            try:
                await db.commit()
            except Exception:
                pass
            raise  # Re-raise so asyncio marks the task as cancelled
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            job.status = "FAILED"
            job.error_log = str(e)
            job.completed_at = _now()
            if job.started_at:
                job.elapsed_seconds = int((job.completed_at - job.started_at).total_seconds())
            _persist_last_progress(job)
            await db.commit()
        finally:
            # Clean up all three stores regardless of outcome. Dropping the live
            # progress entry last means the terminal row above is already the
            # only answer by the time anything can read it.
            progress.discard(job_id)
            job_cancellation_flags.pop(job_id, None)
            active_training_tasks.pop(job_id, None)


@router.get("/gpu-status", response_model=GPUStatusResponse, dependencies=[Depends(verify_access_token)])
async def get_gpu_status():
    """Get current GPU status and memory usage"""
    # This process has no GPU. The figures come from the GPU cluster: the device
    # inventory from the node labels the NVIDIA operator publishes, and live
    # memory from nvidia-smi inside the running trainer pods (cached briefly).
    memory_info = await k8s_runtime.gpu_memory_snapshot()

    return GPUStatusResponse(
        gpu_available=GPU_INFO.get("available", False),
        gpu_name=GPU_INFO.get("name"),
        gpu_count=GPU_INFO.get("count", 0),
        total_memory_gb=memory_info.get("total_gb", 0.0),
        allocated_memory_gb=memory_info.get("allocated_gb", 0.0),
        free_memory_gb=memory_info.get("free_gb", 0.0),
        utilization_percent=memory_info.get("utilization_percent", 0.0),
        cuda_version=GPU_INFO.get("cuda_version")
    )


async def _get_service_availability(db: AsyncSession) -> AvailabilityResponse:
    """Return current service availability status without raising exceptions.

    Centralises the availability logic so that both the /availability endpoint
    and the /start endpoint share a single source of truth.
    """
    # PENDING jobs are already holding a slot (they have a task scheduled), so
    # they must count towards the limit — otherwise a burst of requests is all
    # admitted while every job is still in PENDING.
    result = await db.execute(
        select(func.count(TrainingJob.id)).where(
            TrainingJob.status.in_(["RUNNING", "PENDING"])
        )
    )
    running_jobs = result.scalar()

    # Effective capacity is whichever is smaller: the configured job limit or
    # what the GPU cluster will actually run for this namespace. The second
    # figure is a quota, not a device count — the node has eight GPUs but the
    # namespace may only be allowed to request one at a time.
    max_concurrent = min(settings.MAX_CONCURRENT_JOBS, k8s_runtime.gpu_pool.size) \
        if k8s_runtime.gpu_pool.size else settings.MAX_CONCURRENT_JOBS

    if running_jobs >= max_concurrent:
        return AvailabilityResponse(
            available=False,
            message=f"Service busy: {running_jobs}/{max_concurrent} jobs running",
            running_jobs=running_jobs,
            max_concurrent_jobs=max_concurrent,
        )

    # Live view of the cluster: also catches the GPU cluster becoming
    # unreachable, which is otherwise indistinguishable from an idle one.
    memory_info = await k8s_runtime.gpu_memory_snapshot()

    if not GPU_INFO.get("available") or not memory_info.get("available"):
        return AvailabilityResponse(
            available=False,
            message=GPU_INFO.get("reason") or "No GPU available",
            running_jobs=running_jobs,
            max_concurrent_jobs=max_concurrent,
        )

    # A job gets one whole device, so the meaningful figure is the headroom on
    # the emptiest card — not the sum across all of them.
    free_gb = memory_info.get("max_free_single_gpu_gb", memory_info.get("free_gb", 0))
    if free_gb < settings.GPU_MEMORY_THRESHOLD_GB:
        return AvailabilityResponse(
            available=False,
            message=f"Insufficient GPU memory: {free_gb:.2f}GB available",
            running_jobs=running_jobs,
            max_concurrent_jobs=max_concurrent,
        )

    return AvailabilityResponse(
        available=True,
        message="Service ready",
        running_jobs=running_jobs,
        max_concurrent_jobs=max_concurrent,
    )


@router.get("/availability", response_model=AvailabilityResponse, dependencies=[Depends(verify_access_token)])
async def check_availability(db: AsyncSession = Depends(get_db)):
    """Check if the service is available to accept new jobs"""
    return await _get_service_availability(db)


@router.post("/start", status_code=202, dependencies=[Depends(verify_access_token)])
@limiter.limit("5/minute")
async def start_job(
    request: Request,
    body: TrainingRequest,
    db: AsyncSession = Depends(get_db),
    bearer_token: str = Depends(get_files_api_token),
):
    """
    Start a new fine-tuning job.

    Requires:
    - Authorization: Bearer <keycloak-token>  (Keycloak JWT for service authentication)
    - ft-api-key: <files-api-token>           (Bearer token for FILES API authentication)

    Rate limited: 5 requests per minute per IP.
    """
    # Reuse centralised availability check
    availability = await _get_service_availability(db)
    if not availability.available:
        status_code = 429 if "busy" in availability.message.lower() else 503
        raise HTTPException(status_code=status_code, detail=availability.message)

    # Create new job with optional custom job_id
    if body.job_id is not None:
        existing_job = await db.get(TrainingJob, body.job_id)
        if existing_job:
            raise HTTPException(
                status_code=400,
                detail=f"Job ID {body.job_id} already exists",
            )
        new_job = TrainingJob(
            id=body.job_id,
            username=body.username,
            model_name=body.model_name,
            input_filename=body.input_filenames[0],
            hyperparameters=body.hyperparameters,
            total_steps=body.hyperparameters.get("max_steps"),
            status="PENDING",
        )
    else:
        # Auto-generate job_id
        new_job = TrainingJob(
            username=body.username,
            model_name=body.model_name,
            input_filename=body.input_filenames[0],
            hyperparameters=body.hyperparameters,
            total_steps=body.hyperparameters.get("max_steps"),
            status="PENDING",
        )
    db.add(new_job)
    await db.commit()
    await db.refresh(new_job)

    # Log bearer token receipt (redacted for security)
    logger.info(f"Received FILES API bearer token for job (length: {len(bearer_token)} chars)")

    # Schedule training as a proper asyncio Task so it can be cancelled later
    task = asyncio.create_task(
        background_training_task(new_job.id, body, bearer_token),
        name=f"training_job_{new_job.id}",
    )
    active_training_tasks[new_job.id] = task

    logger.info(f"Job {new_job.id} queued for user {body.username}")
    return {"job_id": new_job.id, "status": "accepted", "message": "Job queued successfully"}


@router.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    token_payload: dict = Depends(verify_access_token),
):
    """Get status of a specific job"""
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # --- AI/ML Security: Extraction detection ---
    # Track when authenticated users retrieve completed model results.
    if job.status == "COMPLETED":
        caller = (
            token_payload.get("preferred_username")
            or token_payload.get("sub")
            or "unknown"
        )
        extraction_detector.record_access(username=caller, job_id=job_id)

    return _job_response(job)


@router.get(
    "/job/{job_id}/logs",
    response_model=JobLogsResponse,
    dependencies=[Depends(verify_access_token)],
)
async def get_job_logs(
    job_id: int,
    tail: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """
    The trainer's output for one job, from this cluster's own copy.

    Not read from the GPU cluster. The training pod's log is deleted along with
    the pod by ``ttlSecondsAfterFinished``, so asking Kubernetes for it works for
    an hour after a job finishes and never again; the engine writes the stream to
    its own volume while it relays progress, and that is what this serves. It
    therefore also answers for a job that is still running.

    ``tail`` returns only the last N lines, for a caller that wants a summary
    rather than a six-hour run's full output.
    """
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    lines, found = await asyncio.to_thread(
        job_logs.read, job_id, tail if tail > 0 else None
    )
    return JobLogsResponse(
        job_id=job_id,
        lines=lines,
        found=found,
        live=job.status in ("PENDING", "RUNNING"),
        truncated=bool(tail > 0 and len(lines) >= tail),
    )


@router.delete("/job/{job_id}", response_model=JobCancelResponse, dependencies=[Depends(verify_access_token)])
async def delete_job(
    job_id: int,
    purge: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """
    Cancel a job, and with ``?purge=true`` forget it as well.

    This is the endpoint the platform calls to cancel, so by default the record
    survives: it is set to CANCELLED and kept. Deleting the row unconditionally —
    which is what this used to do — made the next status poll return 404, and the
    caller reads a 404 as "job not found on backend" and shows the job as
    *failed*. A user who cancelled a job would be told it had crashed.
    """
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    cancelled = False
    if job.status in ("RUNNING", "PENDING"):
        # 1. Raise the cooperative flag. The runtime's watcher picks it up and
        #    writes the CANCEL file into the training pod, which stops at the next
        #    step and uploads nothing.
        job_cancellation_flags[job_id] = True

        # 2. Cancel the asyncio Task so it stops at the next await point. The
        #    runtime deletes the Kubernetes Job on its way out, so the GPU is
        #    released even if the pod ignored the CANCEL file.
        task = active_training_tasks.pop(job_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=30.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.warning(
                    f"Job {job_id}: background task did not finish cleanly within timeout"
                )

        # 3. Backstop: if this process restarted since the job was submitted
        #    there is no task to cancel, and the Job outlives us. Delete it by
        #    name so an orphan cannot hold a GPU slot indefinitely.
        await asyncio.to_thread(k8s_runtime.terminate_job, job_id, True)

        cancelled = True
        await db.refresh(job)
        if job.status in ("RUNNING", "PENDING"):
            # The task did not get far enough to close the record out.
            _mark_cancelled(job)
        logger.info(f"Job {job_id}: cancelled")

    if purge:
        await db.delete(job)
        await db.commit()
        job_cancellation_flags.pop(job_id, None)
        logger.info(f"Job {job_id} deleted from database")
        return JobCancelResponse(
            job_id=job_id,
            status="DELETED",
            message="Job cancelled and deleted successfully",
        )

    await db.commit()
    job_cancellation_flags.pop(job_id, None)
    return JobCancelResponse(
        job_id=job_id,
        status=job.status,
        message="Job cancelled successfully" if cancelled
                else f"Job already {job.status.lower()}; nothing to cancel",
    )


@router.get("/history/{username}", response_model=list[JobStatusResponse], dependencies=[Depends(verify_access_token)])
async def get_history(username: str, limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Get training job history for a user"""
    result = await db.execute(
        select(TrainingJob)
        .where(TrainingJob.username == username)
        .order_by(TrainingJob.created_at.desc())
        .limit(limit)
    )
    return [_job_response(job) for job in result.scalars().all()]

@router.get("/jobs", response_model=list[JobStatusResponse], dependencies=[Depends(verify_access_token)])
async def list_all_jobs(
    username: str = None,
    status: str = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List all jobs with optional username and status filters"""
    query = select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(limit)

    if username:
        query = query.where(TrainingJob.username == username)

    if status:
        query = query.where(TrainingJob.status == status.upper())

    result = await db.execute(query)
    # Overlaid here too, not just on the detail endpoint: a client that polls the
    # list to drive a progress bar is the whole point, and this endpoint returns
    # every job in one call.
    return [_job_response(job) for job in result.scalars().all()]

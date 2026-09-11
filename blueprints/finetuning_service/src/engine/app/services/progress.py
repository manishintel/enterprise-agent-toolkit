# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Live progress for in-flight jobs.

The trainer already reports on every step, but the job row is only written when
training starts and when it finishes - so anything polling this service sees 0%
for the whole run and then a jump to 100%. This module is the buffer in between:

    ProgressCallback (GPU node)  --"[PROGRESS] {...}" on stdout-->
    slurm_runtime._relay_output  --> jobs.update_job_progress_sync -->
    record() here                --> overlaid onto every job read,
                                     and flushed to the row every few seconds

``record`` and ``set_phase`` are called from the event loop while a job streams
output, so they do no I/O - they replace a dict entry and nothing more. Getting
the values into Postgres is the flusher's job (see app/main.py), which matters
only for a reader that queries the database directly; the HTTP responses are
overlaid at read time and never wait for it.

Nothing here is authoritative. ``background_training_task`` writes the final row
itself and calls ``discard``, so once a job reaches a terminal status the stored
row wins and a late step record can no longer move it.

No locking: every entry point runs on the API process' event loop thread. Phases
around a blocking call are set in the coroutine, not inside the worker thread,
to keep that true.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Set

# Phase names are part of the API contract: a client maps them to a coarse
# percentage for the stages that have no step count of their own. Check with
# consumers before renaming one.
PHASE_DOWNLOADING = "downloading_data"
PHASE_PREPARING = "preparing_environment"
PHASE_TRAINING = "training"
PHASE_MERGING = "merging"
PHASE_UPLOADING = "uploading_model"

TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})

# The only fields this module will overlay onto a job row or response.
TRACKED_FIELDS = (
    "current_step",
    "total_steps",
    "progress_percent",
    "training_loss",
    "num_train_epochs",
    "current_phase",
)

# job_id -> latest values. Entries live only for the duration of a job.
_live: Dict[int, Dict] = {}
_dirty: Set[int] = set()

# The worker speaks the trainer's vocabulary; the job row uses its own names.
_FIELD_ALIASES = (
    ("current_step", "current_step"),
    ("max_steps", "total_steps"),      # TrainerState.max_steps is the total
    ("total_steps", "total_steps"),
    ("progress_percent", "progress_percent"),
    ("loss", "training_loss"),
    ("epoch", "num_train_epochs"),
)


def set_phase(job_id: int, phase: str) -> None:
    """Record which stage of the pipeline a job has reached."""
    entry = _live.setdefault(job_id, {})
    if entry.get("current_phase") != phase:
        entry["current_phase"] = phase
        entry["updated_at"] = time.time()
        _dirty.add(job_id)


def record(job_id: int, payload: Dict) -> None:
    """Absorb one progress record from the trainer.

    A record carrying a step count also proves the job has left
    ``preparing_environment``, so the phase is advanced from the payload itself
    rather than guessed by the caller. ``is not None`` throughout: step 0 and
    0.0% are real readings, and a payload only ever carries the keys it knows.
    """
    entry = _live.setdefault(job_id, {})

    for source, field in _FIELD_ALIASES:
        value = payload.get(source)
        if value is not None:
            entry[field] = value

    phase = payload.get("phase")
    if phase:
        entry["current_phase"] = phase
    elif payload.get("current_step") is not None:
        entry["current_phase"] = PHASE_TRAINING

    entry["updated_at"] = time.time()
    _dirty.add(job_id)


def snapshot(job_id: int) -> Optional[Dict]:
    """A copy of everything known about a job, or None if it is not in flight."""
    entry = _live.get(job_id)
    return dict(entry) if entry is not None else None


def overlay(job_id: int, status: Optional[str]) -> Dict:
    """Live values that should be preferred over the stored row.

    Empty for a terminal job: its row is already final, and for a job that was
    never in flight in this process there is nothing to add. Only non-None
    values are returned, so an overlay never blanks a column that has a value.
    """
    if status in TERMINAL_STATUSES:
        return {}
    entry = _live.get(job_id)
    if not entry:
        return {}
    return {
        field: entry[field]
        for field in TRACKED_FIELDS
        if entry.get(field) is not None
    }


def discard(job_id: int) -> None:
    """Forget a job - call once its terminal row has been committed."""
    _live.pop(job_id, None)
    _dirty.discard(job_id)


def take_dirty() -> Dict[int, Dict]:
    """Values changed since the last call, and clear the change set.

    Claiming the ids up front means a record arriving mid-flush is simply picked
    up by the next one instead of being dropped.
    """
    if not _dirty:
        return {}
    claimed = list(_dirty)
    _dirty.clear()
    return {job_id: dict(_live[job_id]) for job_id in claimed if job_id in _live}

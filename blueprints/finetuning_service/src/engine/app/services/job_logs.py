# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Durable storage for a training job's output, on *this* cluster.

The trainer runs in a pod on the GPU cluster, so its log starts life as kubelet
log files on a node we do not own. Those files go away with the pod, which
``ttlSecondsAfterFinished`` deletes an hour after the Job finishes - and before
this module existed that was the only copy. A successful job's output was
therefore unrecoverable, and a failed one left only the 40-line tail that
``_Run.tail()`` keeps in memory for the error message.

The GPU cluster is treated as non-persistent: it runs the training and keeps the
HuggingFace cache warm, and nothing else on it is expected to survive. So the
engine, which is already streaming the pod log line by line in order to relay
progress, writes that same stream to a volume here as it arrives.

Two copies result, on purpose:

* **A plain file on the engine's own PVC**, written during the run. It is what
  makes the log readable *while* the job is training and what survives a restart
  of this pod. Retention is bounded by ``prune()``.
* **A ``.tar.gz`` in the Files API**, uploaded once the job is terminal, so the
  log ends up in the same object store as the model it produced and is subject
  to the same lifecycle as every other user-visible artefact.

The upload goes to the Files API's *in-cluster* Service, not through the public
gateway. The gateway route is OIDC-gated and this pod has no browser session;
the IP exemption in front of that route names the GPU cluster's egress address
(the training pod does its own transfers), not this pod's. In-cluster the
identity is simply ``X-Forwarded-User``, which is also why no user credential is
needed here.
"""
from __future__ import annotations

import io
import logging
import os
import re
import tarfile
import time
from typing import List, Optional, Tuple

import httpx

from app.config import settings

logger = logging.getLogger("uvicorn")

# One line of trainer output can be enormous: tqdm redraws a progress bar in
# place with carriage returns and only terminates it with a newline at the end,
# so the "line" the relay hands over may be an entire bar animation. Truncate
# rather than drop - the tail of such a line is the informative part, but keeping
# all of it would let one job's cosmetics dominate the file.
_MAX_LINE_CHARS = 8192

# Lines buffered before touching the disk. The relay calls write() once per line
# on the event loop, and at logging_steps=1 that is once per training step, so
# a syscall per line would put avoidable I/O on the same task that is parsing
# progress records.
_FLUSH_EVERY_LINES = 200
_FLUSH_EVERY_SECONDS = 10.0

# Job id + run token, as _Run names them. Used to find a job's log again when the
# caller knows only the job id.
_NAME_RE = re.compile(r"^(?P<job>\d+)-(?P<token>[0-9a-f]+)\.log$")

# No surrounding newlines: the buffer is joined with "\n", so they would show up
# as blank lines in the stored log and in every tail of it.
_TRUNCATION_NOTICE = (
    "[engine] log truncated: this job exceeded LOG_STORE_MAX_BYTES "
    "({limit} bytes). Training was not affected."
)


def enabled() -> bool:
    return bool(settings.LOG_STORE_ENABLED and settings.LOG_STORE_DIR)


def _store_dir() -> str:
    os.makedirs(settings.LOG_STORE_DIR, exist_ok=True)
    return settings.LOG_STORE_DIR


def path_for(job_id: int, token: str) -> str:
    return os.path.join(settings.LOG_STORE_DIR, f"{job_id}-{token}.log")


def find(job_id: int) -> Optional[str]:
    """
    The newest stored log for *job_id*, or None.

    A job id can be reused - a restored database, or a caller-supplied id - and
    each run gets its own token, so more than one file can match. The newest is
    the one a caller asking about "job 7" means.
    """
    if not enabled():
        return None
    try:
        candidates = [
            os.path.join(settings.LOG_STORE_DIR, name)
            for name in os.listdir(settings.LOG_STORE_DIR)
            if (m := _NAME_RE.match(name)) and int(m.group("job")) == job_id
        ]
    except OSError as exc:
        logger.warning(f"Could not list the log store: {exc}")
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getmtime(p))


class LogWriter:
    """
    Append-only sink for one run's output. Not thread-safe by design.

    Every method swallows its own I/O errors. A job must not fail because its
    log could not be written: the log is a diagnostic, and losing it is strictly
    less bad than losing the training run that produced it.
    """

    def __init__(self, job_id: int, token: str) -> None:
        self.job_id = job_id
        self.token = token
        self.path = path_for(job_id, token) if enabled() else ""
        self._fh: Optional[io.TextIOBase] = None
        self._buffer: List[str] = []
        self._written = 0
        self._last_flush = time.monotonic()
        self._truncated = False
        self._broken = False

    # -- writing ----------------------------------------------------------

    def write(self, line: str) -> None:
        if self._broken or not self.path:
            return
        if self._truncated:
            return

        if len(line) > _MAX_LINE_CHARS:
            line = line[:_MAX_LINE_CHARS] + " ...[truncated]"

        self._buffer.append(line)
        self._written += len(line) + 1

        if self._written >= settings.LOG_STORE_MAX_BYTES:
            self._truncated = True
            self._buffer.append(
                _TRUNCATION_NOTICE.format(limit=settings.LOG_STORE_MAX_BYTES)
            )
            self.flush()
            logger.warning(
                f"Job {self.job_id} - log exceeded LOG_STORE_MAX_BYTES; "
                f"stopped recording at {self._written} bytes"
            )
            return

        if (
            len(self._buffer) >= _FLUSH_EVERY_LINES
            or time.monotonic() - self._last_flush >= _FLUSH_EVERY_SECONDS
        ):
            self.flush()

    def flush(self) -> None:
        if self._broken or not self._buffer:
            return
        try:
            if self._fh is None:
                _store_dir()
                # Line buffering is deliberately *not* used: the point of
                # buffering in memory is to avoid per-line writes.
                self._fh = open(self.path, "a", encoding="utf-8")
            self._fh.write("\n".join(self._buffer) + "\n")
            self._fh.flush()
        except OSError as exc:
            # Stop trying. A full or unwritable volume would otherwise log a
            # warning per line for the rest of the run.
            self._broken = True
            logger.warning(
                f"Job {self.job_id} - log store write failed, giving up on it: {exc}"
            )
        finally:
            self._buffer.clear()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        self.flush()
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read(job_id: int, tail: Optional[int] = None) -> Tuple[List[str], bool]:
    """
    (lines, found) for *job_id*, optionally only the last *tail* lines.

    Reads whatever is on disk right now, so a running job returns the log so far
    - the writer flushes at most every few seconds behind the live stream.
    """
    path = find(job_id)
    if not path:
        return [], False
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        logger.warning(f"Job {job_id} - could not read stored log: {exc}")
        return [], False
    if tail is not None and tail > 0:
        lines = lines[-tail:]
    return lines, True


# ---------------------------------------------------------------------------
# Archiving to the Files API
# ---------------------------------------------------------------------------

def _archive_bytes(path: str, member_name: str) -> bytes:
    """The log as a gzipped tar, in memory.

    A tar rather than a bare ``.log.gz`` because the Files API validates the
    extension against a fixed list (``.txt``, ``.json``, ``.jsonl``, ``.tar.gz``,
    ...) and rejects anything else with a 400. ``.tar.gz`` also matches how the
    trained model itself is stored, so both artefacts of a job look alike.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(path, arcname=member_name)
    return buffer.getvalue()


def archive(job_id: int, token: str, username: str) -> Optional[str]:
    """
    Upload the finished log to the Files API. Blocking; returns the file id.

    Best-effort: a job that trained and published a model is a success even if
    its log could not be filed, and the copy on the engine's volume is still
    there either way.
    """
    if not settings.LOG_ARCHIVE_URL:
        return None
    path = path_for(job_id, token)
    if not os.path.isfile(path):
        return None

    member = f"job-{job_id}-{token}.log"
    # The '#' prefix is the convention the Files API list uses to group an
    # artefact by kind, the same way merged models are stored as
    # "finetuned_models#...".
    filename = f"finetuning_logs#job-{job_id}-{token}.tar.gz"

    try:
        payload = _archive_bytes(path, member)
    except OSError as exc:
        logger.warning(f"Job {job_id} - could not package log for upload: {exc}")
        return None

    url = f"{settings.LOG_ARCHIVE_URL.rstrip('/')}/v1/files"
    try:
        response = httpx.post(
            url,
            # In-cluster call: the identity is the header, there is no OIDC gate
            # on the Service and no user credential to forward.
            headers={"X-Forwarded-User": username or "default"},
            files={"file": (filename, payload, "application/gzip")},
            data={"purpose": settings.LOG_ARCHIVE_PURPOSE},
            timeout=float(settings.LOG_ARCHIVE_TIMEOUT),
            # Never a proxy for a cluster-local Service.
            trust_env=False,
        )
        response.raise_for_status()
        file_id = response.json().get("id")
    except Exception as exc:  # noqa: BLE001 - diagnostics must not fail a job
        logger.warning(f"Job {job_id} - log archive upload failed: {exc}")
        return None

    logger.info(
        f"Job {job_id} - archived log as {file_id} "
        f"({len(payload) / 1024:.0f} KiB compressed)"
    )
    return file_id


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def prune(keep: Optional[set] = None) -> int:
    """
    Delete stored logs past the retention window. Blocking; returns the count.

    *keep* names files that must survive regardless of age - the runs currently
    in flight, whose writers still hold them open.
    """
    if not enabled() or settings.LOG_STORE_RETENTION_DAYS <= 0:
        return 0

    cutoff = time.time() - settings.LOG_STORE_RETENTION_DAYS * 86400
    protected = keep or set()
    removed = 0
    try:
        names = os.listdir(settings.LOG_STORE_DIR)
    except OSError as exc:
        logger.warning(f"Could not list the log store for pruning: {exc}")
        return 0

    for name in names:
        if not _NAME_RE.match(name) or name in protected:
            continue
        path = os.path.join(settings.LOG_STORE_DIR, name)
        try:
            if os.path.getmtime(path) >= cutoff:
                continue
            os.remove(path)
            removed += 1
        except OSError as exc:
            logger.warning(f"Could not prune {path}: {exc}")
    if removed:
        logger.info(
            f"Pruned {removed} stored training log(s) older than "
            f"{settings.LOG_STORE_RETENTION_DAYS} day(s)"
        )
    return removed

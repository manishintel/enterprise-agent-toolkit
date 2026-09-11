# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Kubernetes execution runtime.

This service exposes its HTTP API from a pod that has no GPU. Training therefore
cannot run in-process. Each job becomes one ``batch/v1`` Job in a namespace on
the GPU cluster:

    Job -> pod(nvidia.com/gpu: 1) -> bash entrypoint.sh -> python -m app.train_worker

Four facts about the target environment drive the design.

1. **The GPU cluster is usually a different cluster.** So the client is built
   from an explicit kubeconfig mounted from a Secret, not from in-cluster
   credentials, and every capability below is restricted to what *namespaced*
   RBAC grants. In particular ``pods/portforward`` is not assumed - only
   ``pods/log`` and ``pods/exec``.

2. **We cannot push an image to a registry the GPU nodes trust.** So the trainer
   runs the *stock, digest-pinned* Unsloth image and our worker code is injected
   as a ConfigMap that the entrypoint assembles into an importable package. The
   ConfigMap is named by a hash of its own content, so a code change produces a
   new object and running jobs keep the tree they started with.

3. **Progress has to travel over the Kubernetes API.** The trainer prints
   ``[PROGRESS]``/``[RESULT]`` marker lines; this module streams the pod log and
   feeds them to the same callback the in-process trainer used, so progress
   reporting is unchanged. The log stream is *resumed* if it breaks, which can
   duplicate a few lines - harmless, because every record is a full snapshot of
   the fields it carries rather than an increment.

4. **A pod's own filesystem is not visible from here.** The trainer's result is
   therefore taken from the log; ``result.json`` on the shared PVC is only a
   fallback, read back with ``pods/exec``. Cancellation travels the same way: we
   create the CANCEL file the trainer polls, and only delete the Job outright if
   it ignores it for the grace period.

The public HTTP contract is unaffected. This module replaces ``slurm_runtime``
and keeps the same names for everything the routers call.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

from kubernetes import client as k8s, config as k8s_config
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream as k8s_stream

from app.config import GPU_INFO, settings
from app.services import job_logs, worker_bundle

logger = logging.getLogger("uvicorn")

# Live description of the GPU cluster. This *is* app.config.GPU_INFO (same
# object), mutated in place rather than rebound, so every reader - the status
# endpoints, the admission gate, /info - always sees current state.
_CLUSTER: Dict = GPU_INFO

# Cached GPU inventory: {"ts": float, "gpus": [...]}
_INVENTORY: Dict = {"ts": 0.0, "gpus": []}
_INVENTORY_LOCK = asyncio.Lock()

# job_id -> live run record, so cancellation and shutdown can reach it.
_RUNS: Dict[int, "_Run"] = {}

# Label keys. `managed-by` is what makes orphan cleanup safe: this module only
# ever deletes objects it can prove it created.
LABEL_MANAGED_BY = "app.kubernetes.io/managed-by"
LABEL_NAME = "app.kubernetes.io/name"
LABEL_COMPONENT = "app.kubernetes.io/component"
LABEL_JOB_ID = "finetuning.intel.com/job-id"
MANAGED_BY = "finetuning-engine"

_MANAGED_SELECTOR = f"{LABEL_MANAGED_BY}={MANAGED_BY}"

_PROGRESS_MARKER = "[PROGRESS]"
_RESULT_MARKER = "[RESULT]"

# Read timeout on a *following* log stream. The trainer logs every step, but it
# goes quiet for minutes while the base model downloads during the merge, so
# this has to be generous; a timeout is treated as "reconnect", not "failed".
_LOG_READ_TIMEOUT = 900

# How long to keep the last lines of output *in memory*, for error reporting. The
# full log is written to the engine's own volume as it streams; see
# app/services/job_logs.py.
_LOG_TAIL_LINES = 40

# Prefix for the volume-sweep Jobs, kept distinct from `ft-<job id>-<token>` so
# the orphan sweep and the cancellation paths never mistake one for a trainer.
_JANITOR_PREFIX = "ft-janitor"


class K8sRuntimeError(RuntimeError):
    """Raised when the GPU cluster is unreachable or a Job cannot be created."""


# ---------------------------------------------------------------------------
# Client
#
# Built lazily and cached: the API must still start (and report *why* it is
# unavailable) when the GPU cluster is down or the kubeconfig is missing.
# ---------------------------------------------------------------------------

_CLIENTS: Optional[Tuple[k8s.CoreV1Api, k8s.BatchV1Api]] = None
_CLIENT_LOCK = threading.Lock()


def _build_clients() -> Tuple[k8s.CoreV1Api, k8s.BatchV1Api]:
    """Create the API clients from TRAIN_KUBECONFIG, or in-cluster credentials."""
    configuration = k8s.Configuration()
    kubeconfig = (settings.TRAIN_KUBECONFIG or "").strip()

    if kubeconfig and os.path.exists(kubeconfig):
        k8s_config.load_kube_config(
            config_file=kubeconfig,
            context=(settings.TRAIN_CONTEXT or None),
            client_configuration=configuration,
        )
        logger.info(f"GPU cluster client from kubeconfig {kubeconfig}")
    else:
        if kubeconfig:
            logger.warning(
                f"TRAIN_KUBECONFIG={kubeconfig} does not exist; "
                "falling back to in-cluster credentials"
            )
        k8s_config.load_incluster_config(client_configuration=configuration)
        logger.info("GPU cluster client from in-cluster service account")

    api = k8s.ApiClient(configuration=configuration)
    return k8s.CoreV1Api(api), k8s.BatchV1Api(api)


def _clients() -> Tuple[k8s.CoreV1Api, k8s.BatchV1Api]:
    global _CLIENTS
    with _CLIENT_LOCK:
        if _CLIENTS is None:
            _CLIENTS = _build_clients()
        return _CLIENTS


def _core() -> k8s.CoreV1Api:
    return _clients()[0]


def _batch() -> k8s.BatchV1Api:
    return _clients()[1]


# ---------------------------------------------------------------------------
# Cluster introspection
#
# Everything the application used to read from torch.cuda is available from the
# node labels the NVIDIA GPU Operator publishes, which needs no GPU and no
# privileged access:
#   nvidia.com/gpu.product          NVIDIA-B300-SXM6-AC
#   nvidia.com/gpu.memory           275040          (MiB, per device)
#   nvidia.com/gpu.count            8
#   nvidia.com/gpu.compute.major    10
#   nvidia.com/cuda.driver-version.full / cuda.runtime-version.full
# ---------------------------------------------------------------------------

_NODE_PRODUCT = "nvidia.com/gpu.product"
_NODE_MEMORY = "nvidia.com/gpu.memory"
_NODE_COMPUTE_MAJOR = "nvidia.com/gpu.compute.major"
_NODE_COMPUTE_MINOR = "nvidia.com/gpu.compute.minor"
_NODE_CUDA_RUNTIME = "nvidia.com/cuda.runtime-version.full"
_NODE_DRIVER = "nvidia.com/cuda.driver-version.full"


def _set_cluster(**fields) -> Dict:
    """Replace the cluster description in place and return it."""
    _CLUSTER.clear()
    _CLUSTER.update(fields)
    return _CLUSTER


def cluster_info() -> Dict:
    """The live cluster description (same object app.config.GPU_INFO holds)."""
    return _CLUSTER


def _node_selector() -> Dict[str, str]:
    """TRAIN_NODE_SELECTOR ('k=v,k2=v2') as a dict."""
    selector: Dict[str, str] = {}
    for entry in (settings.TRAIN_NODE_SELECTOR or "").split(","):
        key, _, value = entry.strip().partition("=")
        if key.strip():
            selector[key.strip()] = value.strip()
    return selector


def _gpu_nodes() -> List[k8s.V1Node]:
    """Schedulable nodes that match TRAIN_NODE_SELECTOR and expose GPUs."""
    selector = ",".join(f"{k}={v}" for k, v in _node_selector().items())
    nodes = _core().list_node(label_selector=selector or None).items

    usable = []
    for node in nodes:
        if (node.spec and node.spec.unschedulable) or not node.status:
            continue
        allocatable = node.status.allocatable or {}
        if int(allocatable.get("nvidia.com/gpu", 0) or 0) > 0:
            usable.append(node)
    return usable


def _namespace_gpu_quota() -> Optional[int]:
    """
    GPU ceiling this namespace may run concurrently, or None if unlimited.

    A ResourceQuota is the honest source for a shared cluster: the node may
    expose eight devices while our tenant is allowed exactly one, and admitting
    on the node figure would leave every second job Pending forever.
    """
    try:
        quotas = _core().list_namespaced_resource_quota(settings.TRAIN_NAMESPACE).items
    except ApiException as exc:
        if exc.status in (403, 404):
            return None  # not permitted to read them, or none exist
        raise

    limits = []
    for quota in quotas:
        hard = (quota.status.hard if quota.status else None) or \
               (quota.spec.hard if quota.spec else None) or {}
        for key in ("requests.nvidia.com/gpu", "nvidia.com/gpu", "limits.nvidia.com/gpu"):
            if key in hard:
                try:
                    limits.append(int(hard[key]))
                except (TypeError, ValueError):
                    pass
    return min(limits) if limits else None


def refresh_cluster_info() -> Dict:
    """
    Re-probe the GPU cluster and update the live description.

    Called at startup and whenever a read path finds the cached inventory stale,
    so the service recovers on its own when the cluster comes back rather than
    needing a restart. Never raises: an unreachable cluster is reported as
    ``available=False`` with the reason, which is what the admission gate and
    /availability surface to the caller.
    """
    try:
        nodes = _gpu_nodes()
    except Exception as exc:
        logger.warning(f"GPU cluster probe failed: {exc}")
        return _set_cluster(available=False, reason=f"GPU cluster unreachable: {exc}")

    if not nodes:
        return _set_cluster(
            available=False,
            reason=(
                f"no schedulable node in the GPU cluster matches "
                f"'{settings.TRAIN_NODE_SELECTOR}' with an allocatable nvidia.com/gpu"
            ),
        )

    labels = nodes[0].metadata.labels or {}
    device_count = sum(
        int((node.status.allocatable or {}).get("nvidia.com/gpu", 0) or 0)
        for node in nodes
    )

    # gpu.memory is per device, in MiB.
    try:
        per_device_gb = round(float(labels.get(_NODE_MEMORY, 0)) / 1024, 2)
    except (TypeError, ValueError):
        per_device_gb = 0.0

    try:
        compute = float(f"{labels.get(_NODE_COMPUTE_MAJOR, 0)}."
                        f"{labels.get(_NODE_COMPUTE_MINOR, 0)}")
    except ValueError:
        compute = 0.0

    try:
        quota = _namespace_gpu_quota()
    except Exception as exc:
        logger.warning(f"Could not read the namespace GPU quota: {exc}")
        quota = None

    # What this tenant can actually run at once.
    usable = min(device_count, quota) if quota is not None else device_count
    if usable <= 0:
        return _set_cluster(
            available=False,
            reason=(
                f"namespace '{settings.TRAIN_NAMESPACE}' has a GPU quota of "
                f"{quota}; no device can be requested"
            ),
        )

    return _set_cluster(
        available=True,
        name=labels.get(_NODE_PRODUCT, "unknown").replace("-", " "),
        # `count` is deliberately the tenant's concurrency ceiling, not the
        # node's device count: it is what sizes the slot pool and what
        # /gpu-status means by "how many GPUs do I have".
        count=usable,
        total_memory_gb=per_device_gb,
        cuda_version=labels.get(_NODE_CUDA_RUNTIME) or labels.get(_NODE_DRIVER),
        bf16_supported=compute >= 8.0,
        compute_capability=compute,
        driver_version=labels.get(_NODE_DRIVER),
        nodes=",".join(node.metadata.name for node in nodes),
        namespace=settings.TRAIN_NAMESPACE,
        device_count=device_count,
        gpu_quota=quota,
    )


def log_cluster_info() -> None:
    """Log what the service found, at the detail level operators expect."""
    info = _CLUSTER
    if not info.get("available"):
        logger.warning(
            f"No usable GPU capacity ({info.get('reason')}). Check "
            f"TRAIN_KUBECONFIG, TRAIN_NAMESPACE={settings.TRAIN_NAMESPACE} and "
            f"TRAIN_NODE_SELECTOR={settings.TRAIN_NODE_SELECTOR}. "
            "Training is unavailable."
        )
        return

    logger.info(
        f"GPU cluster: namespace={info.get('namespace')} node(s)={info.get('nodes')} "
        f"devices={info.get('device_count')} quota={info.get('gpu_quota')}"
    )
    logger.info(f"GPU Detection: {info.get('count')} GPU(s) usable by this service")
    logger.info(f"GPU Name: {info.get('name')}")
    logger.info(f"Total GPU Memory (per device): {info.get('total_memory_gb'):.2f} GB")
    logger.info(f"CUDA Version: {info.get('cuda_version')} "
                f"(driver {info.get('driver_version')})")
    logger.info(f"Compute Capability: {info.get('compute_capability')}")
    logger.info(f"BF16 Supported: {info.get('bf16_supported')}")


def cluster_is_available() -> bool:
    """True when the GPU cluster can currently take a job."""
    return bool(_CLUSTER.get("available"))


# Kept for source compatibility with the callers of the previous runtime.
refresh_allocation = refresh_cluster_info
log_allocation = log_cluster_info


# ---------------------------------------------------------------------------
# Inventory and memory
#
# A GPU handed to a pod is exclusively that pod's, so "free memory" is a
# scheduling question, not a sampling question: an unassigned device has all of
# its memory free, and there is no way for another tenant to be sharing it. That
# is what is modelled here. Live utilisation of a *running* job's device is read
# with nvidia-smi through pods/exec, because that is the one place the number is
# both meaningful and obtainable without a GPU of our own.
# ---------------------------------------------------------------------------

def _busy_slot_count() -> int:
    """GPUs this service currently holds."""
    return gpu_pool.busy_slots()


def _sample_running_devices() -> Dict[int, Dict]:
    """
    nvidia-smi output from each running trainer pod, keyed by job id.

    Best effort: a pod that has just been created, is being torn down, or whose
    exec is refused simply contributes nothing.
    """
    samples: Dict[int, Dict] = {}
    for job_id, run in list(_RUNS.items()):
        if not run.pod_name:
            continue
        try:
            out = _exec_in_pod(
                run.pod_name,
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                timeout=30,
            )
        except Exception as exc:
            logger.debug(f"Job {job_id} - nvidia-smi sample failed: {exc}")
            continue

        line = next((l for l in out.splitlines() if l.strip()), "")
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            total, used, util = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        samples[job_id] = {
            "total_memory_gb": round(total / 1024, 2),
            "used_memory_gb": round(used / 1024, 2),
            "free_memory_gb": round((total - used) / 1024, 2),
            "utilization_percent": util,
        }
    return samples


def _build_inventory() -> List[Dict]:
    """One record per GPU slot this service can use. Blocking - call in a thread."""
    if not _CLUSTER.get("available"):
        return []

    slots = int(_CLUSTER.get("count") or 0)
    per_device = float(_CLUSTER.get("total_memory_gb") or 0.0)
    name = _CLUSTER.get("name", "unknown")
    samples = _sample_running_devices()
    busy_jobs = sorted(samples)

    inventory: List[Dict] = []
    for index in range(slots):
        # Slots are logical: Kubernetes chooses the physical device and hides it
        # behind CUDA_VISIBLE_DEVICES, so index here is our own slot number.
        record = {
            "index": index,
            "name": name,
            "total_memory_gb": per_device,
            "used_memory_gb": 0.0,
            "free_memory_gb": per_device,
            "utilization_percent": 0.0,
            "compute_capability": _CLUSTER.get("compute_capability", 0.0),
            "job_id": None,
        }
        if index < len(busy_jobs):
            job_id = busy_jobs[index]
            record.update(samples[job_id])
            record["job_id"] = job_id
        elif index < _busy_slot_count():
            # Slot is held but the pod is not sampleable yet (still pulling the
            # image, or exec was refused). Report it as taken, not as free, or
            # the admission gate would hand the same device out twice.
            record.update({
                "used_memory_gb": per_device,
                "free_memory_gb": 0.0,
                "utilization_percent": 0.0,
            })
        inventory.append(record)
    return inventory


async def get_gpu_inventory(force: bool = False) -> List[Dict]:
    """
    Per-slot GPU stats, cached for GPU_INVENTORY_TTL_SECONDS so status endpoints
    cannot spam the Kubernetes API.

    An empty result is cached too: if the cluster has gone away the service must
    report that honestly rather than serve stale capacity, but it also must not
    re-probe once per request while it is down.
    """
    async with _INVENTORY_LOCK:
        age = time.time() - _INVENTORY["ts"]
        if not (force or age > settings.GPU_INVENTORY_TTL_SECONDS):
            return list(_INVENTORY["gpus"])

        if not _CLUSTER.get("available"):
            await asyncio.to_thread(refresh_cluster_info)

        gpus = await asyncio.to_thread(_build_inventory)
        _INVENTORY.update({"ts": time.time(), "gpus": gpus})
        return list(gpus)


async def gpu_memory_snapshot() -> Dict:
    """
    Aggregate GPU memory across the slots this service can use.

    Mirrors the shape of the old ``GPUMonitor.get_gpu_memory_info()`` so the
    ``/finetune/gpu-status`` response schema is unchanged.
    """
    gpus = await get_gpu_inventory()
    if not gpus:
        return {"available": False}

    total = sum(g["total_memory_gb"] for g in gpus)
    used = sum(g["used_memory_gb"] for g in gpus)
    free = sum(g["free_memory_gb"] for g in gpus)
    util = sum(g["utilization_percent"] for g in gpus) / len(gpus)

    return {
        "available": True,
        "allocated_gb": round(used, 2),
        "reserved_gb": round(used, 2),
        "free_gb": round(free, 2),
        "total_gb": round(total, 2),
        "utilization_percent": round(util, 2),
        # Largest single-device headroom: what a new job can actually claim.
        "max_free_single_gpu_gb": round(max(g["free_memory_gb"] for g in gpus), 2),
        "per_gpu": gpus,
    }


# ---------------------------------------------------------------------------
# GPU slot pool
#
# Kubernetes will happily accept more Jobs than there are GPUs and leave the
# surplus Pending against the namespace quota. That is not wrong, but it is
# invisible to the caller, so concurrency is arbitrated here instead: a job
# holds a slot for its lifetime and the rest queue in this process, where the
# API can report them as PENDING.
# ---------------------------------------------------------------------------

class GpuSlotPool:
    """
    Hands out exclusive GPU slots to concurrent training jobs.

    Waiters are plain futures rather than an asyncio.Condition on purpose:
    release() must be callable from a cancellation cleanup path, where any
    ``await`` would immediately re-raise CancelledError and leak the slot.
    """

    def __init__(self) -> None:
        self._free: Optional[List[int]] = None
        self._in_use: Dict[int, int] = {}  # slot -> job_id
        self._waiters: List[asyncio.Future] = []

    def _ensure_initialised(self) -> None:
        """
        (Re)size the pool from the current cluster description.

        Resizing only happens while the pool is idle, so a running job can never
        have its slot pulled out from under it. This also means the service picks
        up a quota change or a new GPU node without a restart.
        """
        gpu_count = int(_CLUSTER.get("count") or 0)
        desired = min(gpu_count, settings.MAX_CONCURRENT_JOBS) if gpu_count else 0

        if self._free is None or (not self._in_use and len(self._free) != desired):
            previous = None if self._free is None else len(self._free)
            self._free = list(range(desired))
            if previous != desired:
                logger.info(
                    f"GPU slot pool sized to {desired} slot(s) "
                    f"(cluster offers {gpu_count}, "
                    f"MAX_CONCURRENT_JOBS={settings.MAX_CONCURRENT_JOBS})"
                )

    @property
    def size(self) -> int:
        self._ensure_initialised()
        return len(self._free) + len(self._in_use)

    def free_slots(self) -> int:
        self._ensure_initialised()
        return len(self._free)

    def busy_slots(self) -> int:
        self._ensure_initialised()
        return len(self._in_use)

    async def acquire(self, job_id: int) -> int:
        """Block until a slot is free, then reserve it for *job_id*."""
        self._ensure_initialised()
        if self.size == 0:
            raise K8sRuntimeError(
                _CLUSTER.get("reason") or "No GPU capacity available on the GPU cluster"
            )

        while not self._free:
            waiter = asyncio.get_running_loop().create_future()
            self._waiters.append(waiter)
            logger.info(f"Job {job_id} waiting for a free GPU slot")
            try:
                await waiter
            finally:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)

        # No await between the check above and this claim, so the event loop
        # cannot interleave another acquirer here.
        slot = self._free.pop(0)
        self._in_use[slot] = job_id
        logger.info(f"Job {job_id} acquired GPU slot {slot}")
        return slot

    def release(self, slot: int, job_id: int) -> None:
        """Return a slot to the pool. Synchronous by design (see class doc)."""
        self._ensure_initialised()
        if self._in_use.pop(slot, None) is not None:
            self._free.append(slot)
            self._free.sort()
            logger.info(f"Job {job_id} released GPU slot {slot}")
        self._wake_next_waiter()

    def _wake_next_waiter(self) -> None:
        while self._waiters:
            waiter = self._waiters.pop(0)
            if not waiter.done():
                waiter.set_result(None)
                return


gpu_pool = GpuSlotPool()


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

class _Run:
    """Names and state for a single training Job."""

    def __init__(self, job_id: int, slot: int) -> None:
        self.job_id = job_id
        self.slot = slot
        # A run token keeps every object and every path unique even if a job id
        # is reused (a restored database, a caller-supplied id), so a stale
        # CANCEL file or result.json can never be picked up by the next run.
        self.token = secrets.token_hex(4)
        self.name = f"ft-{job_id}-{self.token}"
        self.work_dir = f"{settings.TRAIN_WORK_DIR}/jobs/{job_id}-{self.token}"
        self.pod_name: Optional[str] = None
        self.log_tail: List[str] = []
        # Durable copy of the trainer's output, on this cluster. Opened lazily on
        # the first line, so a run that never produces any leaves no file behind.
        self.log_writer = job_logs.LogWriter(job_id, self.token)

    @property
    def spec_path(self) -> str:
        return "/etc/ft/job/spec.json"

    @property
    def result_path(self) -> str:
        return f"{self.work_dir}/result.json"

    @property
    def cancel_path(self) -> str:
        return f"{self.work_dir}/CANCEL"

    def labels(self) -> Dict[str, str]:
        return {
            LABEL_NAME: "finetuning-engine",
            LABEL_COMPONENT: "trainer",
            LABEL_MANAGED_BY: MANAGED_BY,
            LABEL_JOB_ID: str(self.job_id),
        }

    def remember(self, line: str) -> None:
        """Record one line of trainer output: durably, and in the error tail.

        The durable write comes first because it is the copy that matters - the
        pod's own log is deleted with the pod - and it cannot raise: LogWriter
        absorbs its own I/O errors rather than let a diagnostic take down the run
        it is describing.
        """
        self.log_writer.write(line)

        self.log_tail.append(line)
        if len(self.log_tail) > _LOG_TAIL_LINES:
            del self.log_tail[0]

    def tail(self) -> str:
        return "\n".join(self.log_tail) or "(no log output)"


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

def _worker_env(run: _Run) -> List[Dict]:
    """
    Environment for the trainer container.

    The trainer talks to the Files API itself, so the whole FILES_API_* block is
    forwarded: it downloads the dataset and uploads the merged model directly
    from the GPU cluster, which keeps multi-gigabyte transfers off the API pod.
    The consequence to remember is that the Files API sees *the GPU cluster's*
    egress address, not this pod's - whatever IP allowlist guards it must list
    that address, or a job trains to completion and then fails on the upload.

    Values proven for Unsloth on this hardware:
      * UNSLOTH_VLLM_STANDBY=0 - the default keeps ~30 GB reserved and defeats
        expandable_segments.
      * Triton's JIT cache and Unsloth's generated trainer sources live on the
        shared PVC, so they stay warm across jobs instead of every run paying
        full compile cost. Unsloth takes a FileLock around the latter and its
        10 s default is tight, hence the raised timeout.
    """
    env = {
        "PYTHONPATH": "/opt/ft",
        "PYTHONUNBUFFERED": "1",
        "FT_JOB_ID": str(run.job_id),
        "FT_SPEC": run.spec_path,
        "FT_API_KEY_FILE": "/etc/ft/secret/ft-api-key",

        "FILES_API_URL": settings.FILES_API_URL,
        "FILES_API_TIMEOUT": str(settings.FILES_API_TIMEOUT),
        "FILES_API_UPLOAD_TIMEOUT": str(settings.FILES_API_UPLOAD_TIMEOUT),
        "FILES_API_CONNECT_TIMEOUT": str(settings.FILES_API_CONNECT_TIMEOUT),
        "FILES_API_METADATA_TIMEOUT": str(settings.FILES_API_METADATA_TIMEOUT),
        "FILES_API_TLS_VERIFY": settings.FILES_API_TLS_VERIFY,
        "FILES_API_FORWARD_USER": str(settings.FILES_API_FORWARD_USER).lower(),
        "FILES_API_RESOLVE": settings.FILES_API_RESOLVE,

        # No LOG_DIR: the trainer logs to stdout only, and the engine writes that
        # stream to its own volume as it relays it (app/services/job_logs.py). A
        # log directory on the training volume was created by every job and never
        # written to, and a copy there would be deleted with everything else the
        # sweep reclaims.
        "TEMP_DATA_DIR": f"{run.work_dir}/data",
        "MODEL_OUTPUT_DIR": f"{run.work_dir}/model",
        "ALLOWED_BASE_MODELS": settings.ALLOWED_BASE_MODELS,

        "HF_HOME": settings.HF_HOME,
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "UNSLOTH_VLLM_STANDBY": "0",
        "TRITON_CACHE_DIR": f"{settings.TRAIN_WORK_DIR}/cache/triton",
        "UNSLOTH_COMPILE_LOCATION": f"{settings.TRAIN_WORK_DIR}/cache/unsloth_compiled",
        "UNSLOTH_LOCK_TIMEOUT": "120",
    }

    if settings.HF_OFFLINE:
        # Only when the GPU nodes genuinely cannot reach the Hub: any base model
        # not already in HF_HOME will then fail to load - and, because merging to
        # 16-bit re-downloads the original weights, fail at the very last step.
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    else:
        # Pod egress on the GPU cluster is not direct, so without a proxy the
        # base-model download simply hangs. Both cases are set because some
        # libraries read only one of them.
        for name, value in (
            ("HTTPS_PROXY", settings.TRAIN_HTTPS_PROXY),
            ("HTTP_PROXY", settings.TRAIN_HTTP_PROXY),
            ("NO_PROXY", settings.TRAIN_NO_PROXY),
        ):
            if value:
                env[name] = value
                env[name.lower()] = value

    return [{"name": key, "value": value} for key, value in sorted(env.items())]


def _job_manifest(run: _Run, bundle_name: str, spec_cm: str, secret_name: str) -> Dict:
    """The batch/v1 Job that runs one fine-tuning job."""
    gpu = str(settings.TRAIN_GPU_COUNT)
    manifest: Dict = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": run.name,
            "namespace": settings.TRAIN_NAMESPACE,
            "labels": run.labels(),
        },
        "spec": {
            # A training failure is not something a retry fixes, and a silent
            # retry would restart from step 0 while the caller watched the
            # progress bar go backwards.
            "backoffLimit": 0,
            "completions": 1,
            "parallelism": 1,
            "activeDeadlineSeconds": settings.TRAIN_JOB_TIMEOUT_SECONDS,
            # Keep the finished pod around briefly so its log is still
            # retrievable for diagnosis after the job row is written.
            "ttlSecondsAfterFinished": settings.TRAIN_JOB_TTL_SECONDS,
            "template": {
                "metadata": {"labels": run.labels()},
                "spec": {
                    "restartPolicy": "Never",
                    # The trainer needs no Kubernetes access of its own; not
                    # mounting a token means a compromised training job (an
                    # untrusted dataset is an input here) cannot reach the API.
                    "automountServiceAccountToken": False,
                    "nodeSelector": _node_selector(),
                    "volumes": [
                        {"name": "work",
                         "persistentVolumeClaim": {"claimName": settings.TRAIN_PVC}},
                        {"name": "worker-src",
                         "configMap": {"name": bundle_name, "defaultMode": 0o444}},
                        {"name": "job-spec",
                         "configMap": {"name": spec_cm, "defaultMode": 0o444}},
                        {"name": "job-secret",
                         "secret": {"secretName": secret_name, "defaultMode": 0o400}},
                        # The dataloader's worker processes talk over /dev/shm and
                        # the image's 64 MiB default is far too small for that.
                        {"name": "dshm",
                         "emptyDir": {"medium": "Memory",
                                      "sizeLimit": settings.TRAIN_SHM_SIZE}},
                        # Assembled worker package: the ConfigMap mount is
                        # read-only, and Python needs a writable tree to put
                        # __pycache__ in.
                        {"name": "worker", "emptyDir": {}},
                    ],
                    "containers": [{
                        "name": "trainer",
                        "image": settings.TRAIN_IMAGE,
                        "imagePullPolicy": settings.TRAIN_IMAGE_PULL_POLICY,
                        "command": ["/bin/bash", "/etc/ft/worker-src/entrypoint.sh"],
                        "env": _worker_env(run),
                        # Explicit on purpose. The namespace LimitRange supplies
                        # defaults (4 CPU / 32Gi here) to any container that
                        # omits these, which is enough to OOM-kill a real job
                        # without ever saying why.
                        "resources": {
                            "requests": {
                                "cpu": settings.TRAIN_CPU_REQUEST,
                                "memory": settings.TRAIN_MEMORY_REQUEST,
                                "nvidia.com/gpu": gpu,
                            },
                            "limits": {
                                "cpu": settings.TRAIN_CPU_LIMIT,
                                "memory": settings.TRAIN_MEMORY_LIMIT,
                                "nvidia.com/gpu": gpu,
                            },
                        },
                        "volumeMounts": [
                            {"name": "work", "mountPath": settings.TRAIN_WORK_DIR},
                            {"name": "worker-src", "mountPath": "/etc/ft/worker-src",
                             "readOnly": True},
                            {"name": "job-spec", "mountPath": "/etc/ft/job",
                             "readOnly": True},
                            {"name": "job-secret", "mountPath": "/etc/ft/secret",
                             "readOnly": True},
                            {"name": "dshm", "mountPath": "/dev/shm"},
                            {"name": "worker", "mountPath": "/opt/ft"},
                        ],
                    }],
                },
            },
        },
    }

    pod_spec = manifest["spec"]["template"]["spec"]
    if settings.TRAIN_TOLERATE_GPU_TAINT:
        # A no-op on an untainted node, and required on a tainted one, so it
        # costs nothing to keep and saves a confusing Pending on clusters that
        # reserve their GPU nodes.
        pod_spec["tolerations"] = [{
            "key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule",
        }]
    if settings.TRAIN_IMAGE_PULL_SECRET:
        pod_spec["imagePullSecrets"] = [{"name": settings.TRAIN_IMAGE_PULL_SECRET}]

    return manifest


def _spec_payload(
    run: _Run,
    model_name: str,
    input_file_id: str,
    username: str,
    params: Dict,
) -> Dict:
    """The job description the trainer reads from its mounted ConfigMap."""
    return {
        "job_id": run.job_id,
        "model_name": model_name,
        "input_file_id": input_file_id,
        "username": username,
        "params": params or {},
        "work_dir": run.work_dir,
        "data_dir": f"{run.work_dir}/data",
        "output_dir": f"{run.work_dir}/model",
        "result_path": run.result_path,
        "cancel_path": run.cancel_path,
        "defaults": {
            "max_seq_length": settings.DEFAULT_MAX_SEQ_LENGTH,
            "batch_size": settings.DEFAULT_BATCH_SIZE,
            "gradient_accumulation_steps": settings.DEFAULT_GRADIENT_ACCUMULATION,
            "learning_rate": settings.DEFAULT_LEARNING_RATE,
            "num_train_epochs": settings.DEFAULT_NUM_EPOCHS,
        },
    }


# ---------------------------------------------------------------------------
# Object lifecycle
# ---------------------------------------------------------------------------

def _ensure_worker_bundle() -> str:
    """
    Make sure the ConfigMap holding the worker source exists, and return its name.

    The name embeds a hash of the content, so deploying new worker code creates a
    new ConfigMap rather than mutating the one running jobs are reading. (A
    mutated ConfigMap is propagated into live pods, which would swap the source
    tree underneath a job that has already imported half of it.) Old bundles are
    swept by ``prune_worker_bundles``.
    """
    core = _core()
    name = worker_bundle.configmap_name()
    namespace = settings.TRAIN_NAMESPACE

    try:
        core.read_namespaced_config_map(name, namespace)
        return name
    except ApiException as exc:
        if exc.status != 404:
            raise

    body = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                LABEL_NAME: "finetuning-engine",
                LABEL_COMPONENT: "worker-bundle",
                LABEL_MANAGED_BY: MANAGED_BY,
            },
        },
        "data": worker_bundle.files(),
    }
    try:
        core.create_namespaced_config_map(namespace, body)
        logger.info(f"Created worker bundle ConfigMap {namespace}/{name}")
    except ApiException as exc:
        if exc.status != 409:  # created concurrently by another replica
            raise
    return name


def prune_worker_bundles(keep: str) -> int:
    """Delete worker bundles other than *keep* that no Job still references."""
    core, batch = _clients()
    namespace = settings.TRAIN_NAMESPACE

    in_use = {keep}
    try:
        for job in batch.list_namespaced_job(
            namespace, label_selector=_MANAGED_SELECTOR
        ).items:
            for volume in (job.spec.template.spec.volumes or []):
                if volume.config_map and volume.name == "worker-src":
                    in_use.add(volume.config_map.name)
    except ApiException as exc:
        logger.warning(f"Could not enumerate Jobs while pruning bundles: {exc}")
        return 0

    removed = 0
    selector = f"{_MANAGED_SELECTOR},{LABEL_COMPONENT}=worker-bundle"
    for cm in core.list_namespaced_config_map(namespace, label_selector=selector).items:
        if cm.metadata.name in in_use:
            continue
        try:
            core.delete_namespaced_config_map(cm.metadata.name, namespace)
            removed += 1
            logger.info(f"Pruned stale worker bundle {cm.metadata.name}")
        except ApiException as exc:
            if exc.status != 404:
                logger.warning(f"Could not delete {cm.metadata.name}: {exc}")
    return removed


def _create_job_objects(run: _Run, spec: Dict, bearer_token: str) -> Tuple[str, str]:
    """Create the per-job spec ConfigMap and credential Secret."""
    core = _core()
    namespace = settings.TRAIN_NAMESPACE
    labels = run.labels()

    core.create_namespaced_config_map(namespace, {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": run.name, "namespace": namespace, "labels": labels},
        "data": {"spec.json": json.dumps(spec, indent=2)},
    })

    # The caller's Files API key. It is the trainer's only credential and it is
    # scoped to one job: a Secret rather than an env var on the Job spec (which
    # `kubectl get job -o yaml` would print), and torn down with the run.
    core.create_namespaced_secret(namespace, {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": run.name, "namespace": namespace, "labels": labels},
        "type": "Opaque",
        "data": {
            "ft-api-key": base64.b64encode(bearer_token.encode()).decode(),
        },
    })
    return run.name, run.name


def _delete_job_objects(run: _Run) -> None:
    """Remove the per-job Secret and ConfigMap. Never raises."""
    core = _core()
    namespace = settings.TRAIN_NAMESPACE
    for delete, kind in (
        (core.delete_namespaced_secret, "Secret"),
        (core.delete_namespaced_config_map, "ConfigMap"),
    ):
        try:
            delete(run.name, namespace)
        except ApiException as exc:
            if exc.status != 404:
                logger.warning(
                    f"Job {run.job_id} - could not delete {kind} {run.name}: {exc}"
                )
        except Exception as exc:  # a dead client must not mask the job's outcome
            logger.warning(f"Job {run.job_id} - could not delete {kind}: {exc}")


def _delete_k8s_job(name: str, quiet: bool = False) -> bool:
    """Delete a Job and its pods. Returns True if the delete was accepted."""
    try:
        _batch().delete_namespaced_job(
            name,
            settings.TRAIN_NAMESPACE,
            body=k8s.V1DeleteOptions(propagation_policy="Background"),
        )
        return True
    except ApiException as exc:
        if exc.status != 404 and not quiet:
            logger.warning(f"Could not delete Job {name}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Pod discovery, logs and exec
# ---------------------------------------------------------------------------

def _job_pod(job_name: str) -> Optional[k8s.V1Pod]:
    """The most recent pod belonging to *job_name*, or None."""
    pods = _core().list_namespaced_pod(
        settings.TRAIN_NAMESPACE, label_selector=f"job-name={job_name}"
    ).items
    if not pods:
        return None
    return sorted(pods, key=lambda p: p.metadata.creation_timestamp or 0)[-1]


def _pod_blocking_reason(pod: k8s.V1Pod) -> Optional[str]:
    """
    A permanent reason this pod will never run, or None.

    Distinguishes "queued behind the GPU quota", which is normal and must be
    waited out, from "will never start", which has to fail the job with
    something better than a timeout hours later.
    """
    status = pod.status
    if not status:
        return None

    for cs in (status.container_statuses or []) + (status.init_container_statuses or []):
        waiting = cs.state.waiting if cs.state else None
        if waiting and waiting.reason in (
            "ErrImagePull", "ImagePullBackOff", "InvalidImageName",
            "CreateContainerConfigError", "CreateContainerError", "RunContainerError",
        ):
            return f"{waiting.reason}: {waiting.message or 'no detail'}"

    for condition in (status.conditions or []):
        if (condition.type == "PodScheduled" and condition.status == "False"
                and condition.reason == "Unschedulable"):
            message = condition.message or ""
            # Insufficient GPU is the quota queue, not a misconfiguration.
            if "Insufficient nvidia.com/gpu" in message:
                return None
            return f"Unschedulable: {message}"
    return None


def _pod_events(pod_name: str, limit: int = 5) -> List[str]:
    """Recent event messages for a pod, for the 'why is this still Pending' case."""
    try:
        events = _core().list_namespaced_event(
            settings.TRAIN_NAMESPACE,
            field_selector=f"involvedObject.name={pod_name}",
        ).items
    except Exception:
        return []
    events.sort(key=lambda e: e.last_timestamp or e.event_time or 0)
    return [f"{e.reason}: {e.message}" for e in events[-limit:] if e.reason]


def _exec_in_pod(pod_name: str, command: List[str], timeout: int = 60) -> str:
    """
    Run a command in the trainer container and return its combined output.

    ``pods/exec`` is the only channel to a running pod's filesystem that
    namespaced RBAC reliably grants (``pods/portforward`` frequently is not), so
    it carries both the cancellation signal and the result-file fallback.
    Blocking - call in a thread.
    """
    client = k8s_stream(
        _core().connect_get_namespaced_pod_exec,
        pod_name,
        settings.TRAIN_NAMESPACE,
        container="trainer",
        command=command,
        stderr=True,
        stdin=False,
        stdout=True,
        tty=False,
        _preload_content=False,
    )
    try:
        client.run_forever(timeout=timeout)
        out = client.read_stdout() or ""
        err = client.read_stderr() or ""
        status = client.read_channel(3) or "{}"
    finally:
        client.close()

    try:
        payload = json.loads(status)
    except json.JSONDecodeError:
        payload = {}
    if payload.get("status") == "Failure":
        raise K8sRuntimeError(
            f"exec {command!r} in {pod_name} failed: "
            f"{payload.get('message') or err.strip() or 'unknown error'}"
        )
    return out + err


async def _wait_for_pod(run: _Run, progress_callback=None) -> k8s.V1Pod:
    """
    Wait until the Job's pod is running (or has already finished).

    While it is Pending the reason is reported through the normal progress
    channel, so a caller watching the UI sees "waiting for a GPU" or "pulling
    image" instead of a blank bar.
    """
    deadline = (
        time.monotonic() + settings.TRAIN_PENDING_TIMEOUT_SECONDS
        if settings.TRAIN_PENDING_TIMEOUT_SECONDS > 0 else None
    )
    reported = ""

    while True:
        pod = await asyncio.to_thread(_job_pod, run.name)
        if pod is not None:
            run.pod_name = pod.metadata.name
            phase = (pod.status.phase if pod.status else None) or "Pending"
            if phase in ("Running", "Succeeded", "Failed"):
                return pod

            blocking = _pod_blocking_reason(pod)
            if blocking:
                raise K8sRuntimeError(
                    f"training pod {pod.metadata.name} cannot start - {blocking}"
                )

            detail = "; ".join(await asyncio.to_thread(_pod_events, pod.metadata.name, 2))
            if detail and detail != reported and progress_callback is not None:
                reported = detail
                progress_callback(run.job_id, {
                    "phase": "preparing_environment", "detail": detail,
                })
                logger.info(f"Job {run.job_id} - pod pending: {detail}")

        if deadline is not None and time.monotonic() > deadline:
            raise K8sRuntimeError(
                f"training pod did not start within "
                f"{settings.TRAIN_PENDING_TIMEOUT_SECONDS}s"
            )
        await asyncio.sleep(3)


def _read_log_lines(pod_name: str, since_seconds: Optional[int], sink) -> None:
    """
    Stream one pod log connection, calling *sink* per line. Blocking.

    Reads raw chunks and splits them, rather than trusting the client to hand
    back whole lines: the trainer's output contains very long progress-bar lines
    and a line-oriented iterator over this stream truncates them unpredictably.
    """
    response = _core().read_namespaced_pod_log(
        name=pod_name,
        namespace=settings.TRAIN_NAMESPACE,
        container="trainer",
        follow=True,
        timestamps=False,
        since_seconds=since_seconds,
        _preload_content=False,
        _request_timeout=(30, _LOG_READ_TIMEOUT),
    )
    buffer = b""
    try:
        for chunk in response.stream(amt=65536, decode_content=False):
            buffer += chunk
            while b"\n" in buffer:
                raw, _, buffer = buffer.partition(b"\n")
                sink(raw.decode("utf-8", errors="replace").rstrip("\r"))
        if buffer:
            sink(buffer.decode("utf-8", errors="replace").rstrip("\r"))
    finally:
        response.release_conn()


async def _relay_output(
    run: _Run,
    progress_callback=None,
    result_sink: Optional[Dict] = None,
) -> None:
    """
    Persist the trainer's output and forward its progress records.

    The trainer runs in another pod on another cluster, so its per-step progress
    arrives as marker lines on stdout; feeding them to the same callback the
    in-process trainer used keeps progress reporting unchanged.

    The final result arrives the same way. It is also written to result.json on
    the shared PVC, but that file is only reachable from here through
    ``pods/exec`` on a pod that may already be gone - so the log is the primary
    channel and the file a fallback (see ``_read_result``).

    A broken stream is reconnected rather than treated as the end of the job: an
    apiserver restart or an idle-connection reaper must not silently truncate a
    six-hour run's progress. Reconnecting can replay a few seconds of output,
    which is harmless because every record is a full snapshot of the fields it
    carries, never an increment.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=2048)
    _DONE = object()

    def pump() -> None:
        since: Optional[int] = None
        while True:
            try:
                _read_log_lines(
                    run.pod_name, since,
                    lambda line: asyncio.run_coroutine_threadsafe(
                        queue.put(line), loop
                    ).result(),
                )
                break  # stream closed cleanly: the container has exited
            except Exception as exc:
                if not _RUNS.get(run.job_id):
                    break  # the run is being torn down
                logger.info(
                    f"Job {run.job_id} - log stream interrupted ({exc}); reconnecting"
                )
                since = 5
                time.sleep(2)
        asyncio.run_coroutine_threadsafe(queue.put(_DONE), loop).result()

    pump_task = asyncio.create_task(asyncio.to_thread(pump))

    try:
        while True:
            line = await queue.get()
            if line is _DONE:
                break

            run.remember(line)
            if line.startswith(_PROGRESS_MARKER):
                payload = line[len(_PROGRESS_MARKER):].strip()
                if progress_callback is None:
                    continue
                try:
                    progress_callback(run.job_id, json.loads(payload))
                except json.JSONDecodeError:
                    logger.info(f"Job {run.job_id} - {payload}")
                except Exception as exc:
                    logger.warning(
                        f"Job {run.job_id} - progress callback failed: {exc}"
                    )
            elif line.startswith(_RESULT_MARKER) and result_sink is not None:
                payload = line[len(_RESULT_MARKER):].strip()
                try:
                    result_sink["result"] = json.loads(payload)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        f"Job {run.job_id} - unparseable result record: {exc}"
                    )
            elif "Traceback (most recent call last)" in line or line.startswith("ERROR"):
                logger.error(f"Job {run.job_id} - {line}")
    finally:
        pump_task.cancel()


def _read_result(run: _Run) -> Optional[Dict]:
    """
    Read result.json out of the pod. Fallback only - blocking.

    Used when the result line never made it through the log (a truncated stream,
    a reconnect that skipped it). The pod has to still exist for this to work,
    which is why ttlSecondsAfterFinished keeps it around for a while.
    """
    if not run.pod_name:
        return None
    try:
        raw = _exec_in_pod(run.pod_name, ["cat", run.result_path], timeout=60)
    except Exception as exc:
        logger.info(f"Job {run.job_id} - result file not readable: {exc}")
        return None

    start = raw.find("{")
    if start < 0:
        return None
    try:
        return json.loads(raw[start:])
    except json.JSONDecodeError as exc:
        logger.warning(f"Job {run.job_id} - unreadable result file: {exc}")
        return None


# ---------------------------------------------------------------------------
# Completion and cancellation
# ---------------------------------------------------------------------------

def _job_conclusion(name: str) -> Optional[Tuple[bool, str]]:
    """
    (succeeded, reason) once the Job is terminal, else None. Blocking.
    """
    try:
        job = _batch().read_namespaced_job(name, settings.TRAIN_NAMESPACE)
    except ApiException as exc:
        if exc.status == 404:
            # Deleted underneath us - by an operator, or by our own cancel path.
            return False, "the training Job was deleted"
        raise

    status = job.status
    if not status:
        return None
    if status.succeeded:
        return True, "completed"
    for condition in (status.conditions or []):
        if condition.type == "Failed" and condition.status == "True":
            return False, (
                f"{condition.reason or 'Failed'}: "
                f"{condition.message or 'no detail'}"
            )
        if condition.type == "Complete" and condition.status == "True":
            return True, "completed"
    return None


async def _await_job(run: _Run) -> Tuple[bool, str]:
    """Poll until the Job reaches a terminal state.

    Polling rather than a watch: a watch on a six-hour Job has to be re-established
    every time its resourceVersion expires, and the only thing we need from it is
    a terminal state that the log stream has usually already told us about.
    """
    while True:
        conclusion = await asyncio.to_thread(_job_conclusion, run.name)
        if conclusion is not None:
            return conclusion
        await asyncio.sleep(5)


async def _watch_cancellation(run: _Run, cancellation_check) -> None:
    """
    Bridge in-process cancellation state to the remote trainer.

    Creating the CANCEL file lets the trainer stop between steps and still record
    what it has (the behaviour callers already rely on). If it ignores the file
    for the grace period, the Job is deleted.
    """
    while not cancellation_check(run.job_id):
        await asyncio.sleep(5)

    signalled = False
    if run.pod_name:
        try:
            await asyncio.to_thread(
                _exec_in_pod,
                run.pod_name,
                ["/bin/sh", "-c",
                 f"mkdir -p {run.work_dir} && date +%s > {run.cancel_path}"],
                30,
            )
            signalled = True
            logger.warning(f"Job {run.job_id} - cancellation signalled to trainer")
        except Exception as exc:
            logger.error(f"Job {run.job_id} - could not write cancel file: {exc}")

    if not signalled:
        # No pod to talk to (still Pending, or already gone): nothing can stop
        # cooperatively, so tear the Job down now rather than after the grace
        # period.
        _delete_k8s_job(run.name)
        return

    await asyncio.sleep(settings.TRAIN_CANCEL_GRACE_SECONDS)
    if await asyncio.to_thread(_job_conclusion, run.name) is None:
        logger.warning(
            f"Job {run.job_id} - trainer did not stop within "
            f"{settings.TRAIN_CANCEL_GRACE_SECONDS}s, deleting the training Job"
        )
        _delete_k8s_job(run.name)


def terminate_job(job_id: int, force: bool = False) -> bool:
    """
    Tear down the training Job for *job_id*.

    Returns True if a delete was issued. ``force`` is accepted for signature
    compatibility with the previous runtime; a Job delete is already immediate.
    """
    run = _RUNS.get(job_id)
    if run is None:
        return False
    return _delete_k8s_job(run.name)


def terminate_all() -> int:
    """Tear down every training Job this process started (used at shutdown)."""
    terminated = 0
    for job_id in list(_RUNS):
        # Flush first. At shutdown the driving tasks may never be scheduled again,
        # so the ``finally`` in execute_finetuning that normally closes the writer
        # cannot be relied on - and the buffered tail is the part that says why
        # the job stopped.
        run = _RUNS.get(job_id)
        if run is not None:
            run.log_writer.close()
        if terminate_job(job_id):
            terminated += 1
    return terminated


def cleanup_orphaned_jobs() -> int:
    """
    Delete training Jobs left behind by a previous run of the API.

    A Job survives an API restart on purpose - a brief hiccup should not kill a
    six-hour run - but nothing is streaming its progress any more and it is
    holding the GPU that the slot pool now believes is free. The database-side
    counterpart of this is the orphaned-job sweep in app.main, which marks the
    same jobs FAILED; the two must agree, so both run at startup.

    Only Jobs carrying our managed-by label are touched, and only ones still
    active: a finished Job is left for its TTL so its log stays readable.
    """
    cancelled = 0
    try:
        jobs = _batch().list_namespaced_job(
            settings.TRAIN_NAMESPACE, label_selector=_MANAGED_SELECTOR
        ).items
    except Exception as exc:
        logger.warning(f"Could not list training Jobs for orphan cleanup: {exc}")
        return 0

    for job in jobs:
        status = job.status
        if not status or not status.active:
            continue
        name = job.metadata.name
        if name.startswith(_JANITOR_PREFIX):
            # A volume sweep carries the same managed-by label but drives no job
            # and holds no GPU. Deleting one mid-pass is harmless, but calling it
            # an orphaned training Job in the log is not.
            continue
        logger.warning(f"Deleting orphaned training Job {name} from a previous run")
        if _delete_k8s_job(name):
            cancelled += 1
        # The per-job spec ConfigMap and credential Secret are named after the
        # Job, so the same sweep reclaims them - and, importantly, does not leave
        # a caller's Files API key sitting in the namespace.
        for delete in (_core().delete_namespaced_secret,
                       _core().delete_namespaced_config_map):
            try:
                delete(name, settings.TRAIN_NAMESPACE)
            except ApiException as exc:
                if exc.status != 404:
                    logger.warning(f"Could not delete {name}: {exc}")
    return cancelled


# Kept for source compatibility with the callers of the previous runtime.
cleanup_orphaned_steps = cleanup_orphaned_jobs


# ---------------------------------------------------------------------------
# Training volume sweep
# ---------------------------------------------------------------------------

_RUN_DIR_RE = re.compile(r"^\d+-[0-9a-f]+$")

_SWEEP_SCRIPT = r"""
set -eu
BASE="$WORK_DIR/jobs"
if [ ! -d "$BASE" ]; then
    echo "no $BASE; nothing to sweep"
    exit 0
fi
cd "$BASE"
removed=0
kept=0
for entry in $(ls -1A); do
    [ -d "$entry" ] || continue
    case " $KEEP " in
        *" $entry "*)
            echo "keep    $entry (in flight)"
            kept=$((kept + 1))
            continue
            ;;
    esac
    if [ -z "$(find "$entry" -maxdepth 0 -mmin +"$RETENTION_MINUTES")" ]; then
        echo "keep    $entry (within retention)"
        kept=$((kept + 1))
        continue
    fi
    size=$(du -sh "$entry" 2>/dev/null | cut -f1 || echo '?')
    rm -rf -- "$entry"
    echo "removed $entry ($size)"
    removed=$((removed + 1))
done
echo "sweep complete: removed=$removed kept=$kept"
"""


def _janitor_manifest(name: str, keep: List[str]) -> Dict:
    """A Job that prunes stale run directories from the training volume.

    Runs the *training* image rather than a small utility one for two reasons
    that both matter more than its size: it is already present on the GPU nodes
    (``IfNotPresent`` on a digest that every trainer pull has already cached, so
    the sweep starts immediately and pulls nothing), and it runs as the same user
    the trainer wrote those files as, so the removals are permitted.

    Only ``$WORK_DIR/jobs`` is ever touched. The HuggingFace cache and the Triton
    and Unsloth compile caches live beside it and are deliberately left alone -
    they are what makes the GPU cluster worth keeping state on at all.
    """
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": settings.TRAIN_NAMESPACE,
            "labels": {
                LABEL_NAME: "finetuning-engine",
                LABEL_COMPONENT: "janitor",
                LABEL_MANAGED_BY: MANAGED_BY,
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 600,
            "ttlSecondsAfterFinished": 300,
            "template": {
                "metadata": {
                    "labels": {
                        LABEL_NAME: "finetuning-engine",
                        LABEL_COMPONENT: "janitor",
                        LABEL_MANAGED_BY: MANAGED_BY,
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "volumes": [
                        {"name": "work",
                         "persistentVolumeClaim": {"claimName": settings.TRAIN_PVC}},
                    ],
                    "containers": [{
                        "name": "janitor",
                        "image": settings.TRAIN_IMAGE,
                        "imagePullPolicy": settings.TRAIN_IMAGE_PULL_POLICY,
                        "command": ["/bin/sh", "-c", _SWEEP_SCRIPT],
                        "env": [
                            {"name": "WORK_DIR", "value": settings.TRAIN_WORK_DIR},
                            {"name": "KEEP", "value": " ".join(keep)},
                            {"name": "RETENTION_MINUTES",
                             "value": str(settings.TRAIN_SWEEP_RETENTION_HOURS * 60)},
                        ],
                        # No GPU, and explicit so the namespace LimitRange does
                        # not hand a directory sweep 4 CPU and 32Gi.
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "128Mi"},
                            "limits": {"cpu": "500m", "memory": "512Mi"},
                        },
                        "volumeMounts": [
                            {"name": "work", "mountPath": settings.TRAIN_WORK_DIR},
                        ],
                    }],
                },
            },
        },
    }


def active_log_names() -> Set[str]:
    """Log filenames belonging to runs this process is still driving.

    Their writers hold the files open and keep appending, so retention must not
    consider them however old the job is.
    """
    return {f"{run.job_id}-{run.token}.log" for run in list(_RUNS.values())}


def sweep_train_volume() -> Optional[str]:
    """
    Reclaim stale run directories on the GPU cluster's volume. Blocking.

    Returns the sweep's own summary line, or None if it could not run.

    This is the only cleanup that does not depend on the trainer getting a chance
    to tidy up after itself. The trainer deletes its dataset and merged model as
    soon as they are uploaded, but a pod that is OOM-killed, evicted or loses its
    node runs no handler at all - and those are precisely the runs holding the
    largest files. Without a sweep the volume fills, and the failure lands on
    whoever submits next.

    The volume is on another cluster with no shared filesystem, so the sweep has
    to happen *there*: it is one short-lived Job that mounts the same claim.
    """
    keep = [
        f"{run.job_id}-{run.token}"
        for run in list(_RUNS.values())
        if _RUN_DIR_RE.match(f"{run.job_id}-{run.token}")
    ]
    name = f"{_JANITOR_PREFIX}-{secrets.token_hex(4)}"

    try:
        _batch().create_namespaced_job(
            settings.TRAIN_NAMESPACE, _janitor_manifest(name, keep)
        )
    except ApiException as exc:
        logger.warning(f"Could not start the training-volume sweep: {exc}")
        return None

    summary: Optional[str] = None
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            conclusion = _job_conclusion(name)
            if conclusion is not None:
                succeeded, reason = conclusion
                if not succeeded:
                    logger.warning(f"Training-volume sweep {name} failed: {reason}")
                break
            time.sleep(5)
        else:
            logger.warning(f"Training-volume sweep {name} did not finish in time")

        pod = _job_pod(name)
        if pod is not None:
            output = _core().read_namespaced_pod_log(
                name=pod.metadata.name,
                namespace=settings.TRAIN_NAMESPACE,
                container="janitor",
            )
            for line in output.splitlines():
                if line.startswith("removed") or line.startswith("sweep complete"):
                    logger.info(f"Volume sweep: {line}")
                    summary = line
    except Exception as exc:  # noqa: BLE001 - housekeeping must not raise
        logger.warning(f"Could not read the sweep result for {name}: {exc}")
    finally:
        # Delete rather than wait out ttlSecondsAfterFinished: the sweep runs on a
        # timer, and leaving each pass behind would put a slow drip of completed
        # Jobs in a namespace somebody else also uses.
        _delete_k8s_job(name, quiet=True)

    return summary

    return summary


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

async def execute_finetuning(
    model_name: str,
    input_file_id: str,
    username: str,
    params: dict,
    job_id: int,
    bearer_token: str,
    progress_callback=None,
    cancellation_check=None,
) -> Dict:
    """
    Run one fine-tuning job as a Kubernetes Job on the GPU cluster.

    Unlike the in-process trainer this replaced, the whole data path lives in the
    pod: it downloads the dataset from the Files API, validates it, trains,
    merges, watermarks and uploads the result, then reports the new file id back
    in its ``[RESULT]`` record. The API pod never touches the bytes.

    Raises:
        RuntimeError: if the job fails, matching the previous error contract.
    """
    if not cluster_is_available():
        await asyncio.to_thread(refresh_cluster_info)
        if not cluster_is_available():
            raise RuntimeError(
                f"Training failed: {_CLUSTER.get('reason', 'no GPU capacity')}"
            )

    slot = await gpu_pool.acquire(job_id)
    run = _Run(job_id, slot)
    _RUNS[job_id] = run

    watcher: Optional[asyncio.Task] = None
    relay: Optional[asyncio.Task] = None
    created_objects = False
    start = time.time()

    try:
        bundle = await asyncio.to_thread(_ensure_worker_bundle)
        spec = _spec_payload(run, model_name, input_file_id, username, params)

        await asyncio.to_thread(_create_job_objects, run, spec, bearer_token)
        created_objects = True

        manifest = _job_manifest(run, bundle, run.name, run.name)
        await asyncio.to_thread(
            _batch().create_namespaced_job, settings.TRAIN_NAMESPACE, manifest
        )
        logger.info(
            f"Job {job_id} - created Job {settings.TRAIN_NAMESPACE}/{run.name} "
            f"on slot {slot} (image {settings.TRAIN_IMAGE})"
        )

        # Cancellation is watched from the moment the Job exists, so a delete
        # arriving while the pod is still Pending is honoured instead of waiting
        # for a pod that may never come.
        if cancellation_check is not None:
            watcher = asyncio.create_task(_watch_cancellation(run, cancellation_check))

        await _wait_for_pod(run, progress_callback)
        logger.info(f"Job {job_id} - trainer pod {run.pod_name} started")

        result_sink: Dict = {}
        relay = asyncio.create_task(_relay_output(run, progress_callback, result_sink))

        try:
            succeeded, reason = await _await_job(run)
        except asyncio.CancelledError:
            # The router cancels this task when a job is deleted.
            _delete_k8s_job(run.name, quiet=True)
            raise

        # Let the relay drain what is left of the stream before reading results.
        try:
            await asyncio.wait_for(relay, timeout=60)
        except asyncio.TimeoutError:
            logger.warning(f"Job {job_id} - timed out flushing trainer output")

        results = result_sink.get("result")
        if results is None:
            results = await asyncio.to_thread(_read_result, run)

        if results is None:
            raise RuntimeError(
                f"Training failed: the trainer produced no result ({reason}). "
                f"Last output:\n{run.tail()}"
            )
        if not results.get("success"):
            # The trainer raises its own "Training failed: ..." and that text is
            # what lands in `error`, so prefixing it again reached the UI as
            # "Training failed: Training failed: ...". Add the prefix only when the
            # message does not already carry it.
            detail = str(results.get("error") or "unknown error")
            raise RuntimeError(
                detail if detail.startswith("Training failed")
                else f"Training failed: {detail}"
            )
        if not succeeded:
            # A result that claims success from a Job Kubernetes failed is not
            # trustworthy - the container was killed after writing it (OOM,
            # deadline, node eviction).
            raise RuntimeError(
                f"Training failed: the training Job did not complete ({reason}). "
                f"Last output:\n{run.tail()}"
            )

        results["gpu_slot"] = slot
        results["k8s_job"] = run.name
        results["k8s_namespace"] = settings.TRAIN_NAMESPACE
        logger.info(
            f"Job {job_id} - finished on slot {slot} in "
            f"{(time.time() - start) / 60:.2f} minutes"
        )
        return results

    finally:
        for task in (watcher, relay):
            if task is not None and not task.done():
                task.cancel()
        _RUNS.pop(job_id, None)

        # Settle the log before anything else: the relay has stopped, so this
        # writes out whatever it had buffered, and the file is then the complete
        # record of a pod that is about to be deleted along with its own copy.
        run.log_writer.close()

        if created_objects:
            # The Secret holds the caller's Files API key; it must not outlive
            # the run, whatever the outcome.
            await asyncio.to_thread(_delete_job_objects, run)
        gpu_pool.release(slot, job_id)

        # File the log alongside the model. Done for failed and cancelled runs
        # too - a failure's log is the more useful of the two - and never allowed
        # to raise, because by this point the outcome of the job is already
        # decided and an archiving problem must not change it.
        try:
            await asyncio.to_thread(job_logs.archive, job_id, run.token, username)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Job {job_id} - could not archive the training log: {exc}")

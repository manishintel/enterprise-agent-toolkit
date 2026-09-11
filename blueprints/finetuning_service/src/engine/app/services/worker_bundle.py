# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
The worker source bundle.

The trainer runs the stock, digest-pinned Unsloth image. That image is the only
one the GPU cluster's nodes can be relied on to pull - we cannot push a derived
image to a registry they trust - so our worker code has to travel *with the job*
rather than baked into a layer. It goes as a ConfigMap that the pod's entrypoint
assembles back into an importable package.

Two properties matter:

* **Content-addressed.** The ConfigMap is named after a hash of its own content,
  so changing the worker code produces a *new* object. Mutating one in place
  would be propagated into every running pod, swapping the source tree
  underneath a job that has already imported half of it.

* **Single source of truth.** The files below are the same files this repository
  holds; nothing is duplicated for the pod's benefit. The one exception is
  ``app/config.py``, where the pod deliberately gets a different module: the API
  service's settings require a database URL and a Keycloak issuer, and the
  trainer needs neither. See ``trainer/config.py``.

ConfigMap keys cannot contain '/', so paths are encoded with '__' and the
entrypoint decodes them.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger("uvicorn")

# .../src/engine — app/services/worker_bundle.py -> app/services -> app -> engine
_ROOT = Path(__file__).resolve().parents[2]

# repository path -> path inside the assembled package.
_LAYOUT: Dict[str, str] = {
    "trainer/entrypoint.sh": "entrypoint.sh",
    # The trainer's own settings module: env-driven, no pydantic-settings, no
    # database. Mounted as app/config.py so the shared modules below can keep
    # importing `from app.config import settings` unchanged.
    "trainer/config.py": "app/config.py",
    "trainer/train_worker.py": "app/train_worker.py",
    "trainer/gpu_engine.py": "app/services/gpu_engine.py",
    "trainer/file_client.py": "app/services/file_client.py",
    "trainer/model_watermarking.py": "app/services/model_watermarking.py",
    # Shared with the API, which validates the model allowlist at submit time
    # while the pod validates the dataset it has just downloaded.
    "app/validators/training_data_validator.py":
        "app/validators/training_data_validator.py",
}

_CACHE: Dict[str, Dict[str, str]] = {}


def _encode(pod_path: str) -> str:
    """Pod-relative path as a valid ConfigMap key."""
    return pod_path.replace("/", "__")


def files() -> Dict[str, str]:
    """
    The bundle as {configmap_key: file content}.

    Read once and cached: the files are baked into the image and cannot change
    under a running process, and re-reading them per job would put filesystem I/O
    on the submit path.
    """
    if _CACHE:
        return dict(_CACHE["files"])

    payload: Dict[str, str] = {}
    for source, pod_path in sorted(_LAYOUT.items()):
        path = _ROOT / source
        if not path.is_file():
            # Fail loudly at the first job rather than shipping a bundle that
            # cannot import: a missing file here means the image was built from
            # an incomplete tree.
            raise FileNotFoundError(
                f"worker bundle source missing: {path} (expected for {pod_path})"
            )
        payload[_encode(pod_path)] = path.read_text(encoding="utf-8")

    _CACHE["files"] = payload
    logger.info(
        f"Worker bundle: {len(payload)} file(s), "
        f"{sum(len(v) for v in payload.values())} bytes, "
        f"digest {_digest(payload)}"
    )
    return dict(payload)


def _digest(payload: Dict[str, str]) -> str:
    """Stable short digest of a bundle payload."""
    digest = hashlib.sha256()
    for key, content in sorted(payload.items()):
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(content.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def content_hash() -> str:
    """Stable short digest of the bundle's content."""
    return _digest(files())


def configmap_name() -> str:
    """Name of the ConfigMap carrying this exact bundle."""
    return f"ft-worker-{content_hash()}"

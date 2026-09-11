# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Trainer settings — mounted into the training pod as ``app/config.py``.

The modules shared with the API service (``file_client``,
``training_data_validator``) do ``from app.config import settings``. Inside the
pod that resolves here rather than to the API's own configuration module, for
two reasons:

* The API's ``Settings`` requires a database URL, a Keycloak issuer and an
  audience. The trainer has, and should have, none of those: it talks to the
  Files API and to nothing else.
* It is built on pydantic-settings, which is not installed in the stock Unsloth
  image. This module deliberately depends on the standard library only, so the
  worker bundle can never be broken by a dependency the image does not ship.

Every value arrives as an environment variable set on the Job's container by
``app/services/k8s_runtime.py::_worker_env``. Defaults here are for the case
where a variable is genuinely optional; anything the trainer cannot invent is
required and fails loudly at import rather than half-way through a job.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("uvicorn")


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set on the training container. The Job manifest is "
            "built by app/services/k8s_runtime.py::_worker_env; a missing value "
            "there means the engine and the worker bundle are out of step."
        )
    return value


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        logger.warning(f"{name} is not an integer; using {default}")
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        logger.warning(f"{name} is not a number; using {default}")
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


class _Settings:
    """Read-only view of the trainer's environment.

    Attribute names match the API service's ``Settings`` exactly, so the modules
    shared between the two need no conditional code.
    """

    def __init__(self) -> None:
        self.APP_NAME = "GPU Accelerated Fine-tuning Trainer"
        self.LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

        # Files API. The trainer both downloads the dataset and uploads the
        # merged model, so this is its entire outside world.
        self.FILES_API_URL = _require("FILES_API_URL").rstrip("/")
        self.FILES_API_TIMEOUT = _int("FILES_API_TIMEOUT", 14400)
        self.FILES_API_UPLOAD_TIMEOUT = _int("FILES_API_UPLOAD_TIMEOUT", 14400)
        self.FILES_API_CONNECT_TIMEOUT = _int("FILES_API_CONNECT_TIMEOUT", 30)
        self.FILES_API_METADATA_TIMEOUT = _int("FILES_API_METADATA_TIMEOUT", 60)
        self.FILES_API_TLS_VERIFY = os.environ.get("FILES_API_TLS_VERIFY", "true")
        self.FILES_API_FORWARD_USER = _bool("FILES_API_FORWARD_USER", True)
        # "hostname:ip[,hostname:ip]". Needed far more often here than in the
        # API pod: the GPU cluster's DNS has no reason to know the toolkit's
        # ingress hostname, and substituting the IP into the URL would miss the
        # host-based route and break certificate verification.
        self.FILES_API_RESOLVE = os.environ.get("FILES_API_RESOLVE", "")

        # Paths. All three are per-job directories on the shared PVC, so a job
        # cannot see, overwrite or leak into another's working set.
        self.TEMP_DATA_DIR = os.environ.get("TEMP_DATA_DIR", "/work/tmp/data")
        self.MODEL_OUTPUT_DIR = os.environ.get("MODEL_OUTPUT_DIR", "/work/tmp/model")
        self.LOG_DIR = os.environ.get("LOG_DIR", "/work/tmp/logs")

        # Training defaults. The spec file normally carries these (the API is the
        # authority on them); these are the fallback for a spec written by an
        # older engine.
        self.DEFAULT_MAX_SEQ_LENGTH = _int("DEFAULT_MAX_SEQ_LENGTH", 2048)
        self.DEFAULT_BATCH_SIZE = _int("DEFAULT_BATCH_SIZE", 2)
        self.DEFAULT_GRADIENT_ACCUMULATION = _int("DEFAULT_GRADIENT_ACCUMULATION", 4)
        self.DEFAULT_LEARNING_RATE = _float("DEFAULT_LEARNING_RATE", 2e-4)
        self.DEFAULT_NUM_EPOCHS = _int("DEFAULT_NUM_EPOCHS", 3)

        self.HF_HOME = os.environ.get("HF_HOME", "/work/hf")
        self.HF_OFFLINE = _bool("HF_HUB_OFFLINE", False)

        # Consumed by app.validators.training_data_validator. The API checks the
        # allowlist at submit time too; keeping it here as well means a bundle
        # that somehow outlives a policy change still refuses the same models.
        self.ALLOWED_BASE_MODELS = os.environ.get("ALLOWED_BASE_MODELS", "")


settings = _Settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

for _directory in (settings.TEMP_DATA_DIR, settings.MODEL_OUTPUT_DIR, settings.LOG_DIR):
    os.makedirs(_directory, exist_ok=True)

# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from pydantic_settings import BaseSettings

logger = logging.getLogger("uvicorn")


class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "GPU Accelerated Fine-tuning Engine"
    ENV: str  # Required: 'development' or 'production'
    LOG_LEVEL: str = "INFO"

    # Keycloak Settings
    KEYCLOAK_ISSUER: str  # Required: e.g. https://keycloak/realms/myrealm
    KEYCLOAK_AUDIENCE: str  # Required: client-id used as audience (azp)
    # Internal URL used for the JWKS fetch when the issuer hostname is not
    # resolvable from inside the cluster. Empty => use KEYCLOAK_ISSUER.
    KEYCLOAK_INTERNAL_URL: str = ""
    # Set to False only in dev/test environments with self-signed certs.
    KEYCLOAK_TLS_VERIFY: bool = True

    # CORS Settings
    ALLOWED_ORIGINS: str = "http://localhost:3000"  # Comma-separated

    # Database Settings
    DATABASE_URL: str  # Required

    # ------------------------------------------------------------------
    # Files API
    #
    # Under Kubernetes the *trainer pod* performs both the dataset download and
    # the model upload, so every value in this block is forwarded into the
    # training Job's environment as well as being read here. See
    # app/services/k8s_runtime.py::_worker_env.
    # ------------------------------------------------------------------
    FILES_API_URL: str  # Required
    FILES_API_TIMEOUT: int = 14400  # 4 hours for large files
    FILES_API_UPLOAD_TIMEOUT: int = 14400
    # FILES_API_TIMEOUT is the *read* budget for a multi-GB transfer; applying it
    # to the TCP connect or to a metadata lookup means an unreachable endpoint
    # hangs a job for four hours.
    FILES_API_CONNECT_TIMEOUT: int = 30
    FILES_API_METADATA_TIMEOUT: int = 60
    # "true", "false", or a path to a CA bundle.
    FILES_API_TLS_VERIFY: str = "true"
    FILES_API_FORWARD_USER: bool = True
    # Comma-separated "hostname:ip" pairs resolved locally, the way an
    # /etc/hosts entry would be, for a Files API published under an ingress
    # hostname with no DNS record. Keeps the hostname in FILES_API_URL so
    # host-based routing and certificate verification both still work.
    #
    # This matters more here than it did under Slurm: the trainer pod runs on a
    # different cluster with its own DNS, which certainly cannot resolve the
    # toolkit's ingress hostname. Example: "api.example.com:10.165.117.174"
    FILES_API_RESOLVE: str = ""

    # Storage Paths (API-pod local; the trainer has its own, on the PVC)
    TEMP_DATA_DIR: str = "/tmp/finetune_data"
    MODEL_OUTPUT_DIR: str = "/tmp/finetune_models"
    LOG_DIR: str = "/tmp/finetune_logs"

    # GPU & Training Settings
    MAX_CONCURRENT_JOBS: int  # Required
    GPU_MEMORY_THRESHOLD_GB: float  # Required
    DEFAULT_MAX_SEQ_LENGTH: int  # Required
    DEFAULT_BATCH_SIZE: int  # Required
    DEFAULT_GRADIENT_ACCUMULATION: int  # Required
    DEFAULT_LEARNING_RATE: float = 2e-4
    DEFAULT_NUM_EPOCHS: int = 3

    # ------------------------------------------------------------------
    # Kubernetes execution settings
    #
    # The API pod has no GPU. Each job becomes one batch/v1 Job in a namespace
    # on the GPU cluster, which is normally a *different* cluster from the one
    # this API runs in — hence an explicit kubeconfig rather than in-cluster
    # credentials. See app/services/k8s_runtime.py.
    # ------------------------------------------------------------------
    # Path to the kubeconfig for the GPU cluster, mounted from a Secret. Empty
    # => fall back to in-cluster credentials (same-cluster deployments).
    TRAIN_KUBECONFIG: str = "/etc/finetune/kubeconfig/config"
    TRAIN_CONTEXT: str = ""  # empty => the kubeconfig's current-context
    TRAIN_NAMESPACE: str = "ftaas"

    # Training image. Pin by digest: the tag `unsloth/unsloth:latest` moves, and
    # a library bump inside it is an API break for the trainer (trl 0.24 dropped
    # the SFTTrainer kwargs this engine used to pass). The digest below is the
    # build validated end-to-end on the B300.
    TRAIN_IMAGE: str = (
        "docker.io/unsloth/unsloth@sha256:"
        "970b6d2229ff6e22f85e26a50a269f66f8f3cb6929ea48b1cf1c251129e42412"
    )
    TRAIN_IMAGE_PULL_POLICY: str = "IfNotPresent"
    TRAIN_IMAGE_PULL_SECRET: str = ""  # only needed for a private registry

    # RWX claim shared by every training Job in the namespace: scratch for the
    # dataset, the merged model, and — importantly — the HuggingFace cache.
    # Merging to 16-bit re-downloads the original fp16 base weights (Unsloth
    # loads a pre-quantised 4-bit repo), so a cold cache adds that download to
    # the end of every job, after training has already succeeded.
    TRAIN_PVC: str = "finetune-engine-work"
    TRAIN_WORK_DIR: str = "/work"

    # Scheduling. The node label is what the GPU Operator sets; the toleration
    # is a no-op on an untainted node and required on a tainted one.
    TRAIN_GPU_COUNT: int = 1
    TRAIN_NODE_SELECTOR: str = "nvidia.com/gpu.present=true"
    TRAIN_TOLERATE_GPU_TAINT: bool = True

    # Resources. A container that omits these inherits the namespace
    # LimitRange default, which on the target cluster is 4 CPU / 32Gi — enough
    # to OOM on a real dataset without ever saying so.
    TRAIN_CPU_REQUEST: str = "8"
    TRAIN_CPU_LIMIT: str = "16"
    TRAIN_MEMORY_REQUEST: str = "48Gi"
    TRAIN_MEMORY_LIMIT: str = "96Gi"
    # /dev/shm for the dataloader. The Docker default of 64Mi is far too small
    # once dataset_num_proc forks workers.
    TRAIN_SHM_SIZE: str = "16Gi"

    # Hard ceiling on one Job, and how long a pod may stay unschedulable before
    # the job is failed rather than left pending forever (the GPU quota on the
    # target namespace is 1, so a second job legitimately waits here).
    TRAIN_JOB_TIMEOUT_SECONDS: int = 604800  # 7 days
    TRAIN_PENDING_TIMEOUT_SECONDS: int = 0  # 0 => wait indefinitely (queue)
    # Kept after completion so `kubectl logs` still works for a short while.
    TRAIN_JOB_TTL_SECONDS: int = 3600
    # Grace period between creating the CANCEL file and deleting the Job.
    TRAIN_CANCEL_GRACE_SECONDS: int = 180
    # How long GPU inventory read off the GPU cluster is cached, in seconds.
    GPU_INVENTORY_TTL_SECONDS: int = 15

    # HuggingFace cache *inside the trainer pod* (on the PVC, hence shared and
    # warm across jobs).
    HF_HOME: str = "/work/hf"
    # Set True only when the GPU nodes have no outbound network: it forces HF to
    # resolve models from HF_HOME alone, and merging will fail for any base
    # model not already cached.
    HF_OFFLINE: bool = False

    # Proxy the trainer pod uses to reach the HuggingFace Hub. Pod egress on the
    # GPU cluster is not direct, so without this the base-model download hangs.
    TRAIN_HTTPS_PROXY: str = ""
    TRAIN_HTTP_PROXY: str = ""
    # Must include the Files API address, or the model upload is handed to the
    # proxy, which answers 403 — training succeeds and only the hand-back fails.
    TRAIN_NO_PROXY: str = "localhost,127.0.0.1,.svc,.cluster.local"

    # Comma-separated base-model allowlist. Empty => the built-in default in
    # app.validators.training_data_validator.
    ALLOWED_BASE_MODELS: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

for directory in [settings.TEMP_DATA_DIR, settings.MODEL_OUTPUT_DIR, settings.LOG_DIR]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Ensured directory exists: {directory}")


# ---------------------------------------------------------------------------
# GPU Detection
#
# The API pod has no GPU, so capability is read from the GPU cluster's node
# objects rather than from torch.cuda.
#
# GPU_INFO is a *live* dict: app.services.k8s_runtime holds a reference to this
# same object and fills it in when it probes the cluster (at startup and on
# every refresh), so readers never see a stale snapshot. It keeps exactly the
# keys the rest of the application already consumes: available, name, count,
# total_memory_gb, cuda_version.
# ---------------------------------------------------------------------------

GPU_INFO: dict = {"available": False, "reason": "GPU cluster not probed yet"}


def get_gpu_info() -> dict:
    """
    Probe the GPU cluster and return the live GPU_INFO dict.

    Kept as a function so callers (app.main's startup hook) can force a
    re-detect; the dict identity never changes.
    """
    from app.services import k8s_runtime  # local import: avoids import cycle

    k8s_runtime.refresh_cluster_info()
    k8s_runtime.log_cluster_info()
    return GPU_INFO

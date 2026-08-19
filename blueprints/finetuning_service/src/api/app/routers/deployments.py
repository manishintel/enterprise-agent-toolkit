"""
Serving a fine-tuned model.

The job detail page prints a ``helm install`` command for bringing a fine-tuned
model up. These endpoints run that same install inside the cluster so it can be
triggered from the UI, and report the progress of it, so the printed command
stays a documented alternative rather than the only way.

The install runs as a Kubernetes Job in the inference namespace rather than in
this process: Helm is a binary, the work outlives a request, and a Job gives the
whole thing an inspectable identity (its own logs, its own RBAC identity, and a
name that makes concurrent attempts collide instead of racing).
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Request

from ..auth import get_current_user
from ..config import get_settings
from ..database import db_manager
from ..errors import (
    ConflictError, InvalidRequestError, ResourceNotFoundError, ServerError,
    ServiceUnavailableError,
    PermissionError as ForbiddenError,
)
from ..k8s import KubeApiError, kube_client
from ..middleware import limiter
from ..model_naming import release_name_for_job, resolve_served_model_name
from ..observability import get_logger
from ..schemas import (
    DeploymentPhase, DeploymentStep, DeploymentStepStatus, ModelDeploymentStatus,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/fine_tuning", tags=["Fine-tuning"])
settings = get_settings()

# Steps a deployment goes through, in order, with the share of the overall
# progress bar each one is worth. Downloading the archive and loading the
# weights are the long poles; the Helm install itself takes seconds.
STEP_INSTALL = "install"
STEP_DOWNLOAD = "download"
STEP_EXTRACT = "extract"
STEP_SERVE = "serve"
STEP_REGISTER = "register"

STEPS: List[Tuple[str, str, int]] = [
    (STEP_INSTALL, "Install Helm release", 10),
    (STEP_DOWNLOAD, "Download model from object storage", 40),
    (STEP_EXTRACT, "Unpack model", 15),
    (STEP_SERVE, "Start vLLM and load model", 30),
    (STEP_REGISTER, "Register with GenAI Gateway", 5),
]

# Init containers of the vllm chart that do the fetch and unpack, and the name
# of the container serving the model. Keep in step with
# core/helm-charts/vllm/templates/deployment.yaml.
FETCH_CONTAINER = "fetch-finetuned-model"
EXTRACT_CONTAINER = "extract-finetuned-model"
SERVE_CONTAINER = "vllm"

PHASE_FOR_STEP = {
    STEP_INSTALL: DeploymentPhase.INSTALLING,
    STEP_DOWNLOAD: DeploymentPhase.DOWNLOADING,
    STEP_EXTRACT: DeploymentPhase.EXTRACTING,
    STEP_SERVE: DeploymentPhase.LOADING,
    STEP_REGISTER: DeploymentPhase.INSTALLING,
}

MESSAGE_FOR_PHASE = {
    DeploymentPhase.NOT_DEPLOYED: "Not deployed",
    DeploymentPhase.INSTALLING: "Creating the model deployment",
    DeploymentPhase.DOWNLOADING: "Downloading the fine-tuned model from object storage",
    DeploymentPhase.EXTRACTING: "Unpacking the fine-tuned model",
    DeploymentPhase.LOADING: "Starting vLLM and loading the model weights",
    DeploymentPhase.READY: "Model is serving and registered with the GenAI Gateway",
    DeploymentPhase.FAILED: "Deployment failed",
    DeploymentPhase.UNINSTALLING: "Removing the model deployment",
    DeploymentPhase.UNAVAILABLE: "Deploying from the UI is not available on this installation",
}


def _deploy_job_name(job_id: str) -> str:
    """Deterministic name, so two clicks collide instead of deploying twice."""
    return f"{release_name_for_job(job_id)}-deploy"[:63]


def _undeploy_job_name(job_id: str) -> str:
    return f"{release_name_for_job(job_id)}-undeploy"[:63]


def _vllm_deployment_name(release_name: str) -> str:
    """Matches the vllm chart's fullname template for a release."""
    return f"{release_name}-vllm"


def _service_url(release_name: str, namespace: str) -> str:
    return f"http://{_vllm_deployment_name(release_name)}-service.{namespace}/v1"


async def _load_owned_job(job_id: str, user_id: str) -> Dict[str, Any]:
    """Fetch a job from the local cache, enforcing ownership."""
    row = await db_manager.fetch_one("""
        SELECT id, model, status, created_at, fine_tuned_model, suffix, user_id
        FROM fine_tuning_jobs
        WHERE id = $1
    """, job_id, timeout=30)

    if not row:
        raise ResourceNotFoundError("fine-tuning job", job_id)

    if str(row.get("user_id", "")) != str(user_id):
        logger.warning(
            "Unauthorized model deployment access attempt — returning 403",
            extra={"job_id": job_id, "requesting_user_id": user_id}
        )
        raise ForbiddenError("You do not have permission to access this fine-tuning job")

    return dict(row)


def _job_failed(job: Dict[str, Any]) -> Optional[str]:
    """Return the failure reason of a Kubernetes Job, or None if not failed."""
    for condition in (job.get("status") or {}).get("conditions") or []:
        if condition.get("type") == "Failed" and condition.get("status") == "True":
            return condition.get("message") or condition.get("reason") or "Job failed"
    return None


def _job_succeeded(job: Dict[str, Any]) -> bool:
    return bool((job.get("status") or {}).get("succeeded"))


def _job_active(job: Dict[str, Any]) -> bool:
    return not _job_succeeded(job) and _job_failed(job) is None


def _container_state(statuses: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for status in statuses or []:
        if status.get("name") == name:
            return status
    return None


def _step_from_container(
    container: Optional[Dict[str, Any]]
) -> Tuple[DeploymentStepStatus, Optional[str]]:
    """Map a container status onto a step status."""
    if not container:
        return DeploymentStepStatus.PENDING, None

    state = container.get("state") or {}

    terminated = state.get("terminated")
    if terminated:
        if terminated.get("exitCode") == 0:
            return DeploymentStepStatus.DONE, None
        return (
            DeploymentStepStatus.ERROR,
            terminated.get("message") or terminated.get("reason") or "Container failed",
        )

    if state.get("running"):
        return DeploymentStepStatus.ACTIVE, None

    waiting = state.get("waiting") or {}
    reason = waiting.get("reason")
    # A container that keeps crashing is not going to become ready on its own.
    if reason in ("CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "CreateContainerConfigError"):
        return DeploymentStepStatus.ERROR, waiting.get("message") or reason
    return DeploymentStepStatus.PENDING, reason


def _pod_is_ready(pod: Dict[str, Any]) -> bool:
    for condition in (pod.get("status") or {}).get("conditions") or []:
        if condition.get("type") == "Ready":
            return condition.get("status") == "True"
    return False


def _newest_serving_pod(pods: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    The most recent pod of the model Deployment.

    The chart's LiteLLM register/deregister hooks carry the same release labels,
    so pods owned by a Job are filtered out; only the Deployment's pods have no
    ``job-name`` label.
    """
    candidates = [
        pod for pod in pods
        if "job-name" not in ((pod.get("metadata") or {}).get("labels") or {})
        and not (pod.get("metadata") or {}).get("deletionTimestamp")
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda pod: (pod.get("metadata") or {}).get("creationTimestamp") or ""
    )


async def _pod_of_job(namespace: str, job_name: str) -> Optional[Dict[str, Any]]:
    pods = await kube_client.list_pods(namespace, label_selector=f"job-name={job_name}")
    if not pods:
        return None
    return max(
        pods,
        key=lambda pod: (pod.get("metadata") or {}).get("creationTimestamp") or ""
    )


def _build_steps(
    deploy_job: Optional[Dict[str, Any]],
    serving_pod: Optional[Dict[str, Any]],
    deployment_exists: bool,
) -> Dict[str, Tuple[DeploymentStepStatus, Optional[str]]]:
    """Derive each step's state from cluster state."""
    states: Dict[str, Tuple[DeploymentStepStatus, Optional[str]]] = {
        key: (DeploymentStepStatus.PENDING, None) for key, _, _ in STEPS
    }

    # The Helm install, and with it the gateway registration: Helm waits for its
    # post-install hook, so a succeeded install Job means the model has been
    # registered with the gateway.
    if deploy_job:
        failure = _job_failed(deploy_job)
        if failure:
            states[STEP_INSTALL] = (DeploymentStepStatus.ERROR, failure)
        elif _job_succeeded(deploy_job):
            states[STEP_INSTALL] = (DeploymentStepStatus.DONE, None)
            states[STEP_REGISTER] = (DeploymentStepStatus.DONE, None)
        else:
            states[STEP_INSTALL] = (DeploymentStepStatus.ACTIVE, "Running helm upgrade --install")
    elif deployment_exists:
        # Deployed earlier, and the Job has since been cleaned up by its TTL.
        states[STEP_INSTALL] = (DeploymentStepStatus.DONE, None)
        states[STEP_REGISTER] = (DeploymentStepStatus.DONE, None)

    if not serving_pod:
        if deployment_exists:
            states[STEP_DOWNLOAD] = (DeploymentStepStatus.PENDING, "Waiting for a pod to be scheduled")
        return states

    pod_status = serving_pod.get("status") or {}
    init_statuses = pod_status.get("initContainerStatuses") or []
    container_statuses = pod_status.get("containerStatuses") or []

    states[STEP_DOWNLOAD] = _step_from_container(_container_state(init_statuses, FETCH_CONTAINER))
    states[STEP_EXTRACT] = _step_from_container(_container_state(init_statuses, EXTRACT_CONTAINER))

    serve_state, serve_detail = _step_from_container(
        _container_state(container_statuses, SERVE_CONTAINER)
    )
    if serve_state == DeploymentStepStatus.ACTIVE and _pod_is_ready(serving_pod):
        serve_state, serve_detail = DeploymentStepStatus.DONE, None
    elif serve_state == DeploymentStepStatus.ACTIVE:
        serve_detail = serve_detail or "Loading model weights"
    states[STEP_SERVE] = (serve_state, serve_detail)

    return states


def _progress(states: Dict[str, Tuple[DeploymentStepStatus, Optional[str]]]) -> int:
    total = 0
    for key, _, weight in STEPS:
        state = states[key][0]
        if state == DeploymentStepStatus.DONE:
            total += weight
        elif state == DeploymentStepStatus.ACTIVE:
            total += weight // 2
    return min(total, 100)


def _phase(
    states: Dict[str, Tuple[DeploymentStepStatus, Optional[str]]]
) -> Tuple[DeploymentPhase, Optional[str]]:
    for key, _, _ in STEPS:
        state, detail = states[key]
        if state == DeploymentStepStatus.ERROR:
            return DeploymentPhase.FAILED, detail
    for key, _, _ in STEPS:
        if states[key][0] != DeploymentStepStatus.DONE:
            return PHASE_FOR_STEP[key], None
    return DeploymentPhase.READY, None


async def _collect_logs(
    namespace: str,
    phase: DeploymentPhase,
    deploy_job_name: str,
    serving_pod: Optional[Dict[str, Any]],
) -> Tuple[List[str], Optional[str]]:
    """Tail the log of whatever is currently doing the work."""
    tail = settings.deployment.log_tail_lines
    container_for_phase = {
        DeploymentPhase.DOWNLOADING: FETCH_CONTAINER,
        DeploymentPhase.EXTRACTING: EXTRACT_CONTAINER,
        DeploymentPhase.LOADING: SERVE_CONTAINER,
        DeploymentPhase.READY: SERVE_CONTAINER,
    }

    if phase in (DeploymentPhase.INSTALLING, DeploymentPhase.UNINSTALLING):
        pod = await _pod_of_job(namespace, deploy_job_name)
        if not pod:
            return [], None
        pod_name = (pod.get("metadata") or {}).get("name")
        log = await kube_client.read_pod_log(namespace, pod_name, "helm", tail)
        return [line for line in log.splitlines() if line.strip()], f"{pod_name}/helm"

    container = container_for_phase.get(phase)
    if phase == DeploymentPhase.FAILED and serving_pod:
        # Show whichever container broke.
        container = SERVE_CONTAINER
        pod_status = serving_pod.get("status") or {}
        for init_status in pod_status.get("initContainerStatuses") or []:
            terminated = (init_status.get("state") or {}).get("terminated") or {}
            if terminated and terminated.get("exitCode", 0) != 0:
                container = init_status.get("name")
                break

    if not container or not serving_pod:
        return [], None

    pod_name = (serving_pod.get("metadata") or {}).get("name")
    log = await kube_client.read_pod_log(namespace, pod_name, container, tail)
    return [line for line in log.splitlines() if line.strip()], f"{pod_name}/{container}"


async def _deployment_status(job_row: Dict[str, Any]) -> ModelDeploymentStatus:
    """Assemble the deployment status of one job's model from cluster state."""
    job_id = job_row["id"]
    namespace = settings.deployment.namespace
    release_name = release_name_for_job(job_id)
    served_model_name = resolve_served_model_name(
        job_row["model"], job_row["created_at"], job_row.get("suffix")
    )
    deployable = job_row["status"] == "succeeded" and bool(job_row.get("fine_tuned_model"))

    base = {
        "job_id": job_id,
        "release_name": release_name,
        "served_model_name": served_model_name,
        "namespace": namespace,
    }

    if not settings.deployment.enabled or not kube_client.available:
        return ModelDeploymentStatus(
            **base,
            phase=DeploymentPhase.UNAVAILABLE,
            message=MESSAGE_FOR_PHASE[DeploymentPhase.UNAVAILABLE],
        )

    deploy_job_name = _deploy_job_name(job_id)
    undeploy_job_name = _undeploy_job_name(job_id)

    deploy_job, undeploy_job, deployment = await asyncio.gather(
        kube_client.get_job(namespace, deploy_job_name),
        kube_client.get_job(namespace, undeploy_job_name),
        kube_client.get_deployment(namespace, _vllm_deployment_name(release_name)),
    )

    # An uninstall in flight overrides everything else: the release is on its way
    # out, so reporting on its pods would be misleading.
    if undeploy_job and _job_active(undeploy_job):
        logs, log_source = await _collect_logs(
            namespace, DeploymentPhase.UNINSTALLING, undeploy_job_name, None
        )
        return ModelDeploymentStatus(
            **base,
            phase=DeploymentPhase.UNINSTALLING,
            message=MESSAGE_FOR_PHASE[DeploymentPhase.UNINSTALLING],
            logs=logs,
            log_source=log_source,
        )

    if not deploy_job and not deployment:
        return ModelDeploymentStatus(
            **base,
            phase=DeploymentPhase.NOT_DEPLOYED,
            message=MESSAGE_FOR_PHASE[DeploymentPhase.NOT_DEPLOYED],
            can_deploy=deployable,
        )

    pods = await kube_client.list_pods(
        namespace, label_selector=f"app.kubernetes.io/instance={release_name}"
    )
    serving_pod = _newest_serving_pod(pods)

    states = _build_steps(deploy_job, serving_pod, bool(deployment))
    phase, failure = _phase(states)
    logs, log_source = await _collect_logs(namespace, phase, deploy_job_name, serving_pod)

    steps = [
        DeploymentStep(key=key, title=title, status=states[key][0], detail=states[key][1])
        for key, title, _ in STEPS
    ]

    return ModelDeploymentStatus(
        **base,
        phase=phase,
        message=MESSAGE_FOR_PHASE[phase],
        progress=_progress(states),
        steps=steps,
        logs=logs,
        log_source=log_source,
        gateway_registered=states[STEP_REGISTER][0] == DeploymentStepStatus.DONE,
        service_url=_service_url(release_name, namespace) if deployment else None,
        # Retrying a failed deployment is the normal way out of it.
        can_deploy=deployable and phase == DeploymentPhase.FAILED,
        can_undeploy=bool(deployment) or phase == DeploymentPhase.FAILED,
        error=failure,
    )


def _helm_job_manifest(
    *,
    name: str,
    job_id: str,
    release_name: str,
    args: List[str],
    action: str,
) -> Dict[str, Any]:
    """A Job that runs one helm command against the inference namespace."""
    config = settings.deployment
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": config.namespace,
            "labels": {
                "app.kubernetes.io/name": "ft-model-deployer",
                "app.kubernetes.io/managed-by": "finetuning-service",
                "finetuning.intel.com/job-id": job_id,
                "finetuning.intel.com/release": release_name,
                "finetuning.intel.com/action": action,
            },
        },
        "spec": {
            # Helm is not safely retried in parallel with itself and a failure
            # here needs looking at, not repeating.
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": config.job_ttl_seconds,
            "activeDeadlineSeconds": config.job_deadline_seconds,
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/name": "ft-model-deployer",
                        "finetuning.intel.com/job-id": job_id,
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "serviceAccountName": config.service_account,
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "runAsGroup": 1000,
                    },
                    "containers": [{
                        "name": "helm",
                        "image": config.helm_image,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["helm"],
                        "args": args,
                        "securityContext": {
                            "allowPrivilegeEscalation": False,
                            "readOnlyRootFilesystem": True,
                            "capabilities": {"drop": ["ALL"]},
                        },
                        # Helm insists on a writable home for its cache and
                        # config even when it only talks to the apiserver.
                        "env": [
                            {"name": "HOME", "value": "/tmp"},
                            {"name": "HELM_CACHE_HOME", "value": "/tmp/helm/cache"},
                            {"name": "HELM_CONFIG_HOME", "value": "/tmp/helm/config"},
                            {"name": "HELM_DATA_HOME", "value": "/tmp/helm/data"},
                        ],
                        "volumeMounts": [
                            {"name": "chart", "mountPath": "/chart", "readOnly": True},
                            {"name": "tmp", "mountPath": "/tmp"},
                        ],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "128Mi"},
                            "limits": {"cpu": "500m", "memory": "512Mi"},
                        },
                    }],
                    "volumes": [
                        {"name": "chart", "configMap": {"name": config.chart_config_map}},
                        {"name": "tmp", "emptyDir": {}},
                    ],
                    "tolerations": [
                        {
                            "key": "node-role.kubernetes.io/control-plane",
                            "operator": "Exists",
                            "effect": "NoSchedule",
                        },
                        {
                            "key": "node-role.kubernetes.io/master",
                            "operator": "Exists",
                            "effect": "NoSchedule",
                        },
                    ],
                },
            },
        },
    }


def _install_args(release_name: str, served_model_name: str, result_file_id: str) -> List[str]:
    """
    The install the job detail page prints, as argv.

    Keep in step with the command rendered by
    ``src/ui/app/finetuning/[id]/page.tsx`` — the button and the copyable
    command are meant to do the same thing.
    """
    config = settings.deployment
    return [
        "upgrade", "--install", release_name, f"/chart/{config.chart_archive_key}",
        "--namespace", config.namespace,
        "-f", f"/chart/{config.values_key}",
        "--set", "finetune.enabled=true",
        "--set", f"finetune.fileId={result_file_id}",
        "--set", f"SERVED_MODEL_NAME={served_model_name}",
        "--set", "litellmRegister.enabled=true",
        "--set", "pvc.enabled=true",
        "--set", f"tensor_parallel_size={config.tensor_parallel_size}",
        "--set", f"pipeline_parallel_size={config.pipeline_parallel_size}",
        "--timeout", config.helm_timeout,
    ]


def _uninstall_args(release_name: str) -> List[str]:
    config = settings.deployment
    return [
        "uninstall", release_name,
        "--namespace", config.namespace,
        # The chart's pre-delete hook deregisters the model from the gateway;
        # waiting keeps the reported phase honest until that has happened.
        "--wait",
        "--timeout", config.helm_timeout,
    ]


async def _replace_finished_job(namespace: str, name: str) -> None:
    """
    Clear a previous attempt out of the way.

    Job names are deterministic so that concurrent attempts collide, which means
    a retry has to delete the finished Job first. Deletion is asynchronous, so
    wait for the name to actually free up rather than racing the apiserver.
    """
    await kube_client.delete_job(namespace, name)
    for _ in range(20):
        await asyncio.sleep(0.5)
        if await kube_client.get_job(namespace, name) is None:
            return
    raise ServiceUnavailableError(
        "The previous deployment attempt is still being cleaned up. Please try again in a moment."
    )


def _require_deployment_support() -> None:
    if not settings.deployment.enabled:
        raise ServiceUnavailableError(
            "Deploying models from the UI is disabled on this installation. "
            "Use the Helm command shown on this page instead."
        )
    if not kube_client.available:
        raise ServiceUnavailableError(
            "This service is not running inside a Kubernetes cluster, so it cannot "
            "deploy models. Use the Helm command shown on this page instead."
        )


@router.post("/jobs/{job_id}/deploy", response_model=ModelDeploymentStatus)
@limiter.limit(f"{settings.rate_limit.job_create}/minute")
async def deploy_fine_tuned_model(
    request: Request,
    job_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Deploy a fine-tuned model and register it with the GenAI Gateway.

    Runs the same Helm install shown on the job detail page as a Job in the
    inference namespace. Poll `GET /v1/fine_tuning/jobs/{job_id}/deployment` for
    progress.
    """
    user_id = current_user["user_id"]
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    try:
        _require_deployment_support()
        job_row = await _load_owned_job(job_id, user_id)

        if job_row["status"] != "succeeded":
            raise InvalidRequestError(
                f"Only a succeeded fine-tuning job can be deployed (this job is {job_row['status']})",
                param="job_id"
            )

        result_file_id = job_row.get("fine_tuned_model")
        if not result_file_id:
            raise InvalidRequestError(
                "This job has no result file to deploy yet", param="job_id"
            )

        config = settings.deployment
        namespace = config.namespace
        release_name = release_name_for_job(job_id)
        served_model_name = resolve_served_model_name(
            job_row["model"], job_row["created_at"], job_row.get("suffix")
        )
        deploy_job_name = _deploy_job_name(job_id)

        # The chart is supplied by a ConfigMap that the plugin playbook keeps in
        # sync with git. Without it the Job would fail with a confusing Helm
        # error, so say what is actually missing.
        chart = await kube_client.get_config_map(namespace, config.chart_config_map)
        if not chart:
            raise ServiceUnavailableError(
                f"The model chart is not published to this cluster (ConfigMap "
                f"'{config.chart_config_map}' in namespace '{namespace}' is missing). "
                f"Re-run the fine-tuning plugin playbook, or use the Helm command shown on this page."
            )

        existing = await kube_client.get_job(namespace, deploy_job_name)
        if existing and _job_active(existing):
            raise ConflictError(
                "This model is already being deployed", code="deployment_in_progress"
            )

        # One model per release, so a running deployment is only redeployed on
        # purpose — upgrading in place would restart a model that is serving.
        deployment = await kube_client.get_deployment(
            namespace, _vllm_deployment_name(release_name)
        )
        if deployment and existing and _job_succeeded(existing):
            raise ConflictError(
                "This model is already deployed. Remove it first to deploy it again.",
                code="already_deployed"
            )

        # Every deployment holds a model volume and a vLLM instance, so the
        # number of them running at once is capped.
        if not deployment:
            live = await kube_client.list_deployments(
                namespace, label_selector="app.kubernetes.io/name=vllm"
            )
            if len(live) >= config.max_deployments:
                raise ConflictError(
                    f"The maximum number of deployed models ({config.max_deployments}) has been "
                    f"reached. Remove a deployed model before deploying another one.",
                    code="deployment_limit_reached"
                )

        if existing:
            await _replace_finished_job(namespace, deploy_job_name)

        manifest = _helm_job_manifest(
            name=deploy_job_name,
            job_id=job_id,
            release_name=release_name,
            args=_install_args(release_name, served_model_name, result_file_id),
            action="install",
        )

        try:
            await kube_client.create_job(namespace, manifest)
        except KubeApiError as exc:
            if exc.status == 409:
                raise ConflictError(
                    "This model is already being deployed", code="deployment_in_progress"
                ) from exc
            raise

        logger.info(
            "Model deployment started",
            extra={
                "job_id": job_id,
                "user_id": user_id,
                "release_name": release_name,
                "served_model_name": served_model_name,
                "result_file_id": result_file_id,
                "correlation_id": correlation_id,
            }
        )

        return await _deployment_status(job_row)

    except (InvalidRequestError, ConflictError, ResourceNotFoundError,
            ForbiddenError, ServiceUnavailableError):
        raise
    except KubeApiError as exc:
        logger.error(
            f"Model deployment failed for job {job_id}: {exc.message}",
            extra={"job_id": job_id, "status": exc.status}
        )
        raise ServiceUnavailableError(
            "Unable to start the deployment. Please try again, or use the Helm command shown on this page."
        ) from exc
    except Exception as exc:
        logger.error(f"Model deployment failed for job {job_id}: {exc}", exc_info=True)
        raise ServerError("Unable to start the model deployment. Please try again later.")


@router.get("/jobs/{job_id}/deployment", response_model=ModelDeploymentStatus)
@limiter.limit(f"{settings.rate_limit.job_read}/minute")
async def get_fine_tuned_model_deployment(
    request: Request,
    job_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Progress of serving a fine-tuned model.

    Derived from cluster state on every call — the Helm Job, the model pod's init
    containers and the pod's readiness — with a tail of the log of whatever is
    doing the work, so a slow or stuck deployment can be diagnosed from the UI.
    """
    try:
        job_row = await _load_owned_job(job_id, current_user["user_id"])
        return await _deployment_status(job_row)
    except (ResourceNotFoundError, ForbiddenError):
        raise
    except KubeApiError as exc:
        logger.warning(
            f"Deployment status unavailable for job {job_id}: {exc.message}",
            extra={"job_id": job_id, "status": exc.status}
        )
        raise ServiceUnavailableError(
            "Unable to read the deployment status from the cluster. Please try again."
        ) from exc
    except Exception as exc:
        logger.error(f"Get deployment status failed for job {job_id}: {exc}", exc_info=True)
        raise ServerError("Unable to retrieve the deployment status. Please try again later.")


@router.delete("/jobs/{job_id}/deployment", response_model=ModelDeploymentStatus)
@limiter.limit(f"{settings.rate_limit.job_cancel}/minute")
async def undeploy_fine_tuned_model(
    request: Request,
    job_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Stop serving a fine-tuned model.

    Uninstalls the Helm release, which deregisters the model from the GenAI
    Gateway through the chart's pre-delete hook and releases the model volume.
    """
    user_id = current_user["user_id"]

    try:
        _require_deployment_support()
        job_row = await _load_owned_job(job_id, user_id)

        namespace = settings.deployment.namespace
        release_name = release_name_for_job(job_id)
        undeploy_job_name = _undeploy_job_name(job_id)

        existing = await kube_client.get_job(namespace, undeploy_job_name)
        if existing and _job_active(existing):
            raise ConflictError(
                "This model is already being removed", code="undeployment_in_progress"
            )

        deploy_job = await kube_client.get_job(namespace, _deploy_job_name(job_id))
        if deploy_job and _job_active(deploy_job):
            raise ConflictError(
                "This model is still being deployed. Wait for that to finish first.",
                code="deployment_in_progress"
            )

        if existing:
            await _replace_finished_job(namespace, undeploy_job_name)

        manifest = _helm_job_manifest(
            name=undeploy_job_name,
            job_id=job_id,
            release_name=release_name,
            args=_uninstall_args(release_name),
            action="uninstall",
        )

        try:
            await kube_client.create_job(namespace, manifest)
        except KubeApiError as exc:
            if exc.status == 409:
                raise ConflictError(
                    "This model is already being removed", code="undeployment_in_progress"
                ) from exc
            raise

        # The install Job is what the status is derived from, so it goes with the
        # release; leaving it behind would report a removed model as installed.
        if deploy_job:
            await kube_client.delete_job(namespace, _deploy_job_name(job_id))

        logger.info(
            "Model deployment removal started",
            extra={"job_id": job_id, "user_id": user_id, "release_name": release_name}
        )

        return await _deployment_status(job_row)

    except (ConflictError, ResourceNotFoundError, ForbiddenError, ServiceUnavailableError):
        raise
    except KubeApiError as exc:
        logger.error(
            f"Model removal failed for job {job_id}: {exc.message}",
            extra={"job_id": job_id, "status": exc.status}
        )
        raise ServiceUnavailableError(
            "Unable to remove the deployment. Please try again."
        ) from exc
    except Exception as exc:
        logger.error(f"Model removal failed for job {job_id}: {exc}", exc_info=True)
        raise ServerError("Unable to remove the model deployment. Please try again later.")

"""Pydantic schemas for OpenAI-compatible fine-tuning service"""

import json
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


def _to_unix_timestamp(value: Any) -> Optional[int]:
    """Convert a datetime object or integer to a Unix timestamp integer.

    DB columns may return either a ``datetime`` instance or a plain ``int``
    (epoch seconds) depending on the driver / mapping layer.  This helper
    handles both cases safely.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(value.timestamp())
    # Already an int/float epoch value
    return int(value)

class JobStatus(str, Enum):
    """Fine-tuning job status"""
    VALIDATING_FILES = "validating_files"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ResourceType(str, Enum):
    """Resource types for fine-tuning"""
    NVIDIA = "nvidia"

class EventLevel(str, Enum):
    """Event log levels"""
    INFO = "info"
    WARN = "warn"
    ERROR = "error"

# Model schemas
class ModelObject(BaseModel):
    """Model object response"""
    id: str
    object: str = "model"
    created: int
    owned_by: str

class ModelListResponse(BaseModel):
    """Model list response"""
    object: str = "list"
    data: List[ModelObject]

# Hyperparameter schemas
class Hyperparameters(BaseModel):
    """Fine-tuning hyperparameters (OpenAI compatible)"""
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    batch_size: Optional[int] = Field(None, ge=1, le=256)
    learning_rate_multiplier: Optional[float] = Field(None, gt=0, le=10)
    n_epochs: Optional[int] = Field(None, ge=1, le=50)
    prompt_loss_weight: Optional[float] = Field(None, ge=0, le=1)
    compute_classification_metrics: Optional[bool] = None
    classification_n_classes: Optional[int] = Field(None, ge=2)
    classification_positive_class: Optional[str] = None
    classification_betas: Optional[List[float]] = None

# Fine-tuning job schemas
class CreateFineTuningJobRequest(BaseModel):
    """Create fine-tuning job request (OpenAI compatible)"""
    model: str = Field(..., min_length=1)
    training_file: str = Field(..., pattern=r"^file-[a-zA-Z0-9_-]+$")
    hyperparameters: Optional[Hyperparameters] = None
    # Name the fine-tuned model is served and registered under. Dots are allowed
    # because base model names contain them (Llama-3.2-3B-Instruct); commas,
    # equals signs and slashes stay out so the name is safe to pass as a Helm
    # value and to use as a vLLM --served-model-name.
    suffix: Optional[str] = Field(None, max_length=64, pattern=r"^[a-zA-Z0-9._-]+$")
    validation_file: Optional[str] = Field(None, pattern=r"^file-[a-zA-Z0-9_-]+$")
    seed: Optional[int] = Field(None, ge=0, le=2147483647)
    resource_type: Optional[str] = None

class FineTuningJobError(BaseModel):
    """Fine-tuning job error details"""
    # Defaulted so a backend failure without an explicit code still serialises;
    # a missing code used to fail response validation and mask the real error.
    code: str = "backend_error"
    message: str
    param: Optional[str] = None

class FineTuningJob(BaseModel):
    """Fine-tuning job response (OpenAI compatible)"""
    model_config = ConfigDict(use_enum_values=True)

    id: str
    object: str = "fine_tuning.job"
    created_at: int
    # When the training engine actually picked the job up. Everything between
    # created_at and this is queue wait, which is usually the bulk of the
    # wall-clock time and would otherwise be reported as training time.
    started_at: Optional[int] = None
    finished_at: Optional[int] = None
    fine_tuned_model: Optional[str] = None
    hyperparameters: Dict[str, Any] = {}
    model: str
    organization_id: Optional[str] = None
    result_files: List[str] = []
    seed: Optional[int] = None
    status: str
    trained_tokens: Optional[int] = None
    training_file: str
    validation_file: Optional[str] = None
    estimated_finish: Optional[int] = None
    error: Optional[FineTuningJobError] = None
    # Model name chosen at submission time; the UI serves and registers the
    # fine-tuned model under it.
    suffix: Optional[str] = None
    # Training telemetry reported by the engine. Not part of the OpenAI schema,
    # but it is what the job page can actually show: the engine never reports
    # trained_tokens, while these are refreshed on every status poll.
    progress_percent: Optional[float] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    current_phase: Optional[str] = None
    # Fractional epoch reached, e.g. 0.35.
    num_train_epochs: Optional[float] = None
    training_loss: Optional[float] = None
    elapsed_seconds: Optional[int] = None

    @classmethod
    def from_db_row(
        cls,
        row: Dict[str, Any],
        requesting_user_id: str,
        include_sensitive: bool = True
    ) -> "FineTuningJob":
        """
        Create FineTuningJob from database row with field-level authorization

        Args:
            row: Database row as dict
            requesting_user_id: User ID making the request
            include_sensitive: Whether to include sensitive fields (for owner only)
        """
        job_owner_id = str(row.get("user_id", ""))
        is_owner = (job_owner_id == requesting_user_id)

        # Parse hyperparameters
        hyperparams = row.get("hyperparameters", {})
        if isinstance(hyperparams, str):
            try:
                hyperparams = json.loads(hyperparams)
            except:
                hyperparams = {}

        # Base fields (always visible)
        job_data = {
            "id": row["id"],
            "object": "fine_tuning.job",
            "created_at": _to_unix_timestamp(row.get("created_at")) or 0,
            "status": row["status"],
            "model": row["model"],
        }

        # Owner-visible or explicitly included sensitive fields
        if is_owner or include_sensitive:
            job_data.update({
                "hyperparameters": hyperparams,
                "training_file": row.get("training_file", ""),
                "validation_file": row.get("validation_file"),
                "fine_tuned_model": row.get("fine_tuned_model"),
                "started_at": _to_unix_timestamp(row.get("started_at")),
                "finished_at": _to_unix_timestamp(row.get("finished_at")),
                "trained_tokens": row.get("trained_tokens"),
                "estimated_finish": row.get("estimated_finish"),
                "progress_percent": row.get("progress_percent"),
                "current_step": row.get("current_step"),
                "total_steps": row.get("total_steps"),
                "current_phase": row.get("current_phase"),
                "num_train_epochs": row.get("num_train_epochs"),
                "training_loss": row.get("training_loss"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "result_files": (json.loads(row["result_files"]) if isinstance(row.get("result_files"), str) else row.get("result_files")) or [],
                "seed": row.get("seed"),
                "suffix": row.get("suffix"),
            })

            # Add error if present
            if row.get("error_code"):
                job_data["error"] = {
                    "code": row["error_code"],
                    "message": row.get("error_message", ""),
                    "param": row.get("error_param")
                }

        return cls(**job_data)

class FineTuningJobListResponse(BaseModel):
    """Fine-tuning job list response"""
    object: str = "list"
    data: List[FineTuningJob]
    has_more: bool = False

# Model deployment schemas
class DeploymentStepStatus(str, Enum):
    """State of one step of bringing a fine-tuned model up"""
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    ERROR = "error"


class DeploymentPhase(str, Enum):
    """Where a fine-tuned model deployment currently is"""
    NOT_DEPLOYED = "not_deployed"
    INSTALLING = "installing"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    LOADING = "loading"
    # The model is serving; all that is left is telling the gateway about it.
    REGISTERING = "registering"
    READY = "ready"
    FAILED = "failed"
    UNINSTALLING = "uninstalling"
    UNAVAILABLE = "unavailable"


class DeploymentStep(BaseModel):
    """One traceable step of a model deployment"""
    key: str
    title: str
    status: DeploymentStepStatus
    detail: Optional[str] = None


class DeployModelRequest(BaseModel):
    """
    Overrides for serving a fine-tuned model.

    Every field is optional: an empty body deploys with the sizing derived from
    the base model, which is what the button did before any of this was
    configurable. ``cpu`` and ``memory`` are Kubernetes quantities ("16",
    "16000m", "32Gi") because they are passed to Helm verbatim.
    """
    cpu: Optional[str] = Field(default=None, description='CPU request, e.g. "16" or "16000m"')
    memory: Optional[str] = Field(default=None, description='Memory request, e.g. "32Gi"')
    tensor_parallel_size: Optional[int] = Field(default=None, ge=1, le=16)
    pipeline_parallel_size: Optional[int] = Field(default=None, ge=1, le=16)
    # vLLM serving flags, appended to the chart's own extraCmdArgs so they
    # override it (argparse keeps the last occurrence of a flag).
    max_model_len: Optional[int] = Field(
        default=None, ge=256, le=1_048_576,
        description="Context window. Unset in the chart, so vLLM uses the model's own maximum",
    )
    max_num_seqs: Optional[int] = Field(default=None, ge=1, le=4096, description="Concurrent sequences")
    max_num_batched_tokens: Optional[int] = Field(default=None, ge=256, le=1_048_576)
    dtype: Optional[Literal["auto", "bfloat16", "float16", "float32"]] = Field(default=None)
    # VLLM_CPU_KVCACHE_SPACE, in GiB. A fixed reservation whatever the model's
    # size, and usually the largest single term in the pod's memory footprint,
    # so changing it should be accompanied by changing `memory`.
    kv_cache_space_gib: Optional[int] = Field(default=None, ge=1, le=512)
    # Sampling *defaults*, applied via --override-generation-config. vLLM has no
    # server-side temperature: a request that sends its own wins, so this cannot
    # enforce anything. Enforcement belongs at the gateway.
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    force: bool = Field(
        default=False,
        description="Deploy even though the request does not fit any node",
    )


class ResourceAmount(BaseModel):
    """A CPU/memory pair, in machine units plus a form fit to print."""
    cpu_millis: int = 0
    memory_bytes: int = 0
    cpu: Optional[str] = None
    memory: Optional[str] = None
    pods: Optional[int] = None


class NodeCapacity(BaseModel):
    """One node's room for another model."""
    name: str
    schedulable: bool
    unschedulable_reason: Optional[str] = None
    allocatable: ResourceAmount
    committed: ResourceAmount
    free: ResourceAmount


class DeploymentCapacity(BaseModel):
    """
    Whether another model will fit, and what to ask for.

    ``basis`` is always "requests": Kubernetes admits a pod by comparing its
    requests against a node's allocatable, and this cluster has no
    metrics-server, so live utilisation is neither used nor available
    (``live_usage_available`` is false). ``available`` is false when the service
    cannot read cluster state at all, in which case the numbers are absent and
    ``message`` explains why rather than showing a misleading zero.
    """
    available: bool
    message: Optional[str] = None
    basis: str = "requests"
    live_usage_available: bool = False
    nodes: List[NodeCapacity] = []
    totals: Optional[Dict[str, ResourceAmount]] = None
    largest_free: Optional[ResourceAmount] = None
    # Sizing suggestion for the model this was requested for.
    recommended: Optional[Dict[str, Any]] = None
    # Result of testing `recommended` against the nodes.
    fits: Optional[bool] = None
    shortfall: Optional[str] = None
    # What the packaged chart will use when a field is left alone, read from the
    # chart itself so the dialog cannot drift from it, plus the ranges the API
    # enforces so the UI does not keep a second copy of them.
    serving_defaults: Optional[Dict[str, Any]] = None
    serving_limits: Optional[Dict[str, Any]] = None
    dtype_choices: List[str] = []
    # The range a CPU and memory request may take: floor from the model's own
    # requirements, ceiling from what one node actually has. Carries the formula
    # behind the floor so the form can show why, and so this service and the form
    # enforce one number rather than two that drift.
    request_limits: Optional[Dict[str, Any]] = None
    # Optional hard cap on how many models may be served at once, independent of
    # size. `deployments_max` is 0 when no cap is configured, which is the default:
    # the CPU and memory check is what decides admission.
    deployments_used: int = 0
    deployments_max: int = 0
    override_allowed: bool = False


class ExtractUtterancesRequest(BaseModel):
    """Knobs for mining utterances out of a training dataset."""
    limit: int = Field(default=30, ge=1, le=200, description="How many utterances to select")
    min_words: int = Field(default=3, ge=1, le=50)
    max_words: int = Field(default=40, ge=2, le=200)
    first_turn_only: bool = Field(
        default=True,
        description="Use only the opening user turn of each conversation; later turns are follow-ups",
    )
    redact_pii: bool = Field(
        default=True,
        description="Replace identifiers and amounts with placeholders. Turning this off puts raw "
                    "trace content into gateway configuration",
    )


class ExtractedUtterance(BaseModel):
    text: str
    # How many near-duplicate phrasings in the dataset this one stands for.
    represents: int = 1


class ExtractUtterancesResponse(BaseModel):
    utterances: List[ExtractedUtterance] = []
    # Counts at each stage of the funnel, so the result is checkable rather than
    # taken on trust: rows -> user turns -> filtered -> unique -> selected.
    report: Dict[str, Any] = {}
    training_file: Optional[str] = None


class SemanticRouteRequest(BaseModel):
    """Utterances and threshold to route to this job's model."""
    utterances: List[str] = Field(..., min_length=1, max_length=500)
    # The router scores a route by the *mean* of its nearest 5 utterances, not by
    # the best match, and a sentence encoder's similarity floor for unrelated text
    # is well above zero. Typically in-domain queries land around 0.55-0.65 and
    # unrelated ones around 0.40-0.45, so 0.5 separates them; the exact figures
    # depend on the encoder and the dataset, which is what /test is for.
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    description: Optional[str] = Field(default=None, max_length=200)
    # Where a query that matches nothing goes. Defaults to the gateway's other
    # chat model when unset.
    default_model: Optional[str] = None
    # Which router to add this route to. Unset means the installation's default
    # (``gateway.router_name``). A name that is not yet registered creates a new
    # router, so a caller can keep unrelated sets of routes apart.
    router_name: Optional[str] = Field(default=None, max_length=120)


class SemanticRouteEntry(BaseModel):
    """One route in the shared router."""
    model: str
    utterances: List[str] = []
    score_threshold: Optional[float] = None
    description: Optional[str] = None
    # True for the route belonging to the job being viewed.
    is_this_job: bool = False


class RouterSummary(BaseModel):
    """One semantic router registered with the gateway."""
    name: str
    routes: int = 0
    # The installation default, which is what a caller gets by not choosing.
    is_default: bool = False
    # True when this job's model already has a route in this router, so the UI can
    # show where a model is routed from without opening each router in turn.
    has_this_model: bool = False


class RoutedModel(BaseModel):
    """
    A model that has a route somewhere, and which router it is in.

    Reported alongside one router's own state so a list of models can say what is
    routed without reading every router in turn -- and, now that more than one can
    exist, say *where* rather than a bare "routed".
    """
    model: str
    router: str
    utterances: int = 0


class SemanticRouteStatus(BaseModel):
    """
    State of the semantic router being viewed, as read back from the gateway.

    The gateway is the source of truth: it stores the router config and returns it
    in clear, so there is no second copy here to drift.
    """
    available: bool
    message: Optional[str] = None
    router_name: str
    # The model clients must call to be routed. Routing is opt-in by model name --
    # it does not intercept traffic addressed to other models.
    configured: bool = False
    this_model: Optional[str] = None
    this_route: Optional[SemanticRouteEntry] = None
    routes: List[SemanticRouteEntry] = []
    default_model: Optional[str] = None
    embedding_model: Optional[str] = None
    available_embedding_models: List[str] = []
    available_chat_models: List[str] = []
    # Every router the gateway knows about, so a caller can add a route to an
    # existing one instead of only ever the default.
    available_routers: List[RouterSummary] = []
    # Every routed model across all routers, so a list view does not have to read
    # each router to say which models are routed and where.
    routed_models: List[RoutedModel] = []
    # Applying a change restarts the gateway, because an auto-router is cached in
    # its process and re-registering the same name does not refresh it.
    restart_required_on_apply: bool = True
    # Set by apply/remove when a restart was triggered, so the caller knows to
    # follow readiness rather than assuming the change is already live.
    gateway_restarting: bool = False


class GatewayReadiness(BaseModel):
    """
    How far along a router change is.

    Applying writes to the gateway and then restarts it, and the change is only
    live once the new process has loaded the router. Two independent signals are
    needed to say that: the Deployment has finished rolling (Kubernetes), and the
    router config is readable with the expected route in it (the gateway itself).
    A caller polls this instead of guessing from a timer.
    """
    ready: bool = False
    restarting: bool = False
    # Rollout progress. None when the deployment cannot be read from here.
    replicas_ready: Optional[int] = None
    replicas_desired: Optional[int] = None
    # The gateway answered and the router carries the expected route.
    gateway_responding: bool = False
    router_present: bool = False
    route_present: bool = False
    router_name: Optional[str] = None
    message: str = ""


class SemanticRouteTestRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    utterances: Optional[List[str]] = Field(
        default=None,
        description="Score against these instead of what is currently applied, to try a set before applying",
    )
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # Which router's other routes to score against, so a test reflects the
    # competition the query will actually face. Unset means the default router.
    router_name: Optional[str] = Field(default=None, max_length=120)


class SemanticRouteTestResponse(BaseModel):
    """
    Which route a query would take, and how close it was.

    ``score`` is the router's own measure: the mean similarity over the route's
    nearest 5 utterances. It reads lower than the closest single match -- 0.60
    where the best utterance scores 0.86 -- and it is the number the threshold is
    compared against.
    """
    query: str
    matched: bool
    matched_model: Optional[str] = None
    score: float = 0.0
    threshold: float = 0.0
    # The utterance it scored highest against, so a surprising result is explainable.
    closest_utterance: Optional[str] = None
    # Every route's best score, for calibrating against the alternatives.
    scores: List[Dict[str, Any]] = []


class ModelDeploymentStatus(BaseModel):
    """Progress of serving a fine-tuned model, derived from cluster state"""
    model_config = ConfigDict(use_enum_values=True)

    job_id: str
    release_name: str
    served_model_name: str
    namespace: str
    phase: str
    message: str
    # 0-100, weighted across the steps below. The long poles are downloading the
    # archive from object storage and vLLM loading the weights.
    progress: int = 0
    steps: List[DeploymentStep] = []
    # Empty unless the caller asked for logs. Status is polled every few seconds
    # while a deployment comes up, and tailing a container on each poll spends a
    # kubelet round-trip on output nobody is looking at; the Logs view fetches
    # them from ``/deployment/logs`` instead.
    logs: List[str] = []
    log_source: Optional[str] = None
    gateway_registered: bool = False
    service_url: Optional[str] = None
    can_deploy: bool = False
    can_undeploy: bool = False
    error: Optional[str] = None


class DeploymentLogs(BaseModel):
    """
    A tail of one deployment's container output, fetched when asked for.

    ``hidden_lines`` is reported rather than silently dropped: a serving model's
    output is mostly liveness probes, so hiding them is the useful default, but a
    reader has to be able to tell the difference between "quiet" and "filtered".
    """
    job_id: str
    logs: List[str] = []
    log_source: Optional[str] = None
    # Which phase the deployment was in, since that decides the container picked.
    phase: str
    tail: int = 0
    hidden_lines: int = 0
    message: Optional[str] = None


# Job event schemas
class FineTuningJobEvent(BaseModel):
    """Fine-tuning job event"""
    id: str
    object: str = "fine_tuning.job.event"
    created_at: int
    level: EventLevel
    message: str
    data: Dict[str, Any] = {}

class FineTuningJobEventListResponse(BaseModel):
    """Fine-tuning job event list response"""
    object: str = "list"
    data: List[FineTuningJobEvent]

# Checkpoint schemas
class CheckpointMetrics(BaseModel):
    """Training checkpoint metrics"""
    step: int
    train_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    valid_loss: Optional[float] = None
    valid_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[float] = None

class JobCheckpoint(BaseModel):
    """Job checkpoint information"""
    id: str
    job_id: str
    step_number: int
    metrics: CheckpointMetrics
    checkpoint_path: Optional[str] = None
    created_at: datetime

# Resource usage schemas
class ResourceUsage(BaseModel):
    """Resource usage tracking"""
    id: str
    job_id: str
    resource_type: ResourceType
    resource_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    cpu_hours: Optional[float] = None
    memory_gb_hours: Optional[float] = None
    gpu_hours: Optional[float] = None
    cost_usd: Optional[float] = None

# API response schemas
class DeleteResponse(BaseModel):
    """Delete operation response"""
    deleted: bool

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: Optional[str] = None
    database: Optional[str] = None

class ErrorResponse(BaseModel):
    """Error response"""
    error: Dict[str, Any]

    @classmethod
    def from_exception(cls, error_type: str, message: str, code: Optional[str] = None, param: Optional[str] = None):
        """Create error response from exception"""
        error_data = {
            "type": error_type,
            "message": message
        }
        if code:
            error_data["code"] = code
        if param:
            error_data["param"] = param

        return cls(error=error_data)

# Configuration schemas
class DatabaseConfig(BaseModel):
    """Database configuration"""
    url: str
    pool_size: int = 20
    max_overflow: int = 0
    pool_timeout: int = 30

class ServiceConfig(BaseModel):
    """Service configuration"""
    database: DatabaseConfig
    log_level: str = "INFO"
    debug: bool = False

# Resource adapter schemas
class ResourceAdapterConfig(BaseModel):
    """Resource adapter configuration"""
    type: ResourceType
    config: Dict[str, Any] = {}

class JobSubmissionRequest(BaseModel):
    """Job submission request to resource adapter"""
    job_id: str
    user_id: str
    model: str
    training_file: str  # OpenAI standard: file ID
    validation_file: Optional[str] = None
    hyperparameters: Dict[str, Any] = {}
    resource_config: Dict[str, Any] = {}  # Backend-specific config (user_uuid, username, etc.)
    user_token: Optional[str] = None  # Keycloak token for backend API authentication

class JobSubmissionResponse(BaseModel):
    """Job submission response from resource adapter"""
    success: bool
    resource_job_id: Optional[str] = None
    error_message: Optional[str] = None
    estimated_duration: Optional[int] = None  # seconds
    backend_status_code: Optional[int] = None  # HTTP status returned by the training engine
    backend_error_detail: Optional[str] = None  # Error text reported by the training engine

class JobStatusRequest(BaseModel):
    """Job status request to resource adapter"""
    job_id: str
    resource_job_id: str

class JobStatusResponse(BaseModel):
    """Job status response from resource adapter"""
    status: JobStatus
    progress: Optional[float] = None  # 0.0 to 1.0
    trained_tokens: Optional[int] = None
    progress_percent: Optional[float] = None  # 0.0 to 100.0 (raw engine value)
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    current_phase: Optional[str] = None   # raw engine phase token, e.g. "merging"
    num_train_epochs: Optional[float] = None  # fractional epoch reached
    training_loss: Optional[float] = None  # latest training loss from engine
    elapsed_seconds: Optional[int] = None  # wall-clock training time
    error_message: Optional[str] = None
    fine_tuned_model: Optional[str] = None
    result_files: Optional[List[str]] = []
    started_at: Optional[int] = None  # Unix timestamp the engine began training
    finished_at: Optional[int] = None  # Unix timestamp
    estimated_finish: Optional[int] = None  # Unix timestamp
    checkpoints: List[CheckpointMetrics] = []
    logs: List[str] = []
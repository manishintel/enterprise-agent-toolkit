"""
Application Configuration with validation and type safety
Production-grade configuration management using Pydantic Settings
"""

from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum


class Environment(str, Enum):
    """Application environment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseSettings(BaseSettings):
    """Database configuration with validation"""
    model_config = SettingsConfigDict(env_prefix='DATABASE_', extra='ignore')

    url: str = Field(..., description="PostgreSQL connection URL")
    pool_min_size: int = Field(default=5, ge=1, le=50, description="Minimum pool size")
    pool_max_size: int = Field(default=20, ge=1, le=100, description="Maximum pool size")
    pool_timeout: int = Field(default=30, ge=5, le=300, description="Pool timeout in seconds")
    command_timeout: int = Field(default=60, ge=10, le=600, description="Command timeout in seconds")

    @field_validator('url')
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format"""
        if not v:
            raise ValueError("DATABASE_URL is required")
        if not v.startswith(('postgresql://', 'postgres://')):
            raise ValueError("DATABASE_URL must start with postgresql:// or postgres://")
        return v


class NvidiaBackendSettings(BaseSettings):
    """Nvidia/Unsloth backend configuration"""
    model_config = SettingsConfigDict(env_prefix='NVIDIA_', extra='ignore')

    api_url: str = Field(..., description="Nvidia backend API URL")
    api_timeout: int = Field(default=120, ge=10, le=600, description="API timeout in seconds")
    max_jobs: int = Field(default=1, ge=1, le=10, description="Max concurrent jobs")

    # Keycloak OAuth2 Client Credentials for backend authentication
    keycloak_token_url: str = Field(..., description="Keycloak token endpoint URL")
    keycloak_client_id: str = Field(..., description="OAuth2 client ID")
    keycloak_client_secret: str = Field(..., description="OAuth2 client secret")
    keycloak_verify_ssl: bool = Field(default=True, description="Verify SSL certificates")

    @field_validator('api_url')
    @classmethod
    def validate_api_url(cls, v: str) -> str:
        """Validate API URL format"""
        if not v:
            raise ValueError("NVIDIA_API_URL is required")
        if not v.startswith(('http://', 'https://')):
            raise ValueError("NVIDIA_API_URL must be a valid HTTP(S) URL")
        return v.rstrip('/')

    @field_validator('keycloak_token_url')
    @classmethod
    def validate_token_url(cls, v: str) -> str:
        """Validate Keycloak token URL"""
        if not v:
            raise ValueError("NVIDIA_KEYCLOAK_TOKEN_URL is required")
        if not v.startswith(('http://', 'https://')):
            raise ValueError("NVIDIA_KEYCLOAK_TOKEN_URL must be a valid HTTP(S) URL")
        return v.rstrip('/')


class DataPrepSettings(BaseSettings):
    """Data preparation service configuration"""
    model_config = SettingsConfigDict(env_prefix='DATAPREP_', extra='ignore')

    api_url: Optional[str] = Field(default=None, description="Data prep API URL")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    timeout: int = Field(default=10, ge=5, le=60, description="Request timeout in seconds")

    @field_validator('api_url')
    @classmethod
    def validate_api_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate API URL if provided"""
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("DATAPREP_API_URL must be a valid HTTP(S) URL")
        return v.rstrip('/') if v else None


class APISettings(BaseSettings):
    """API-specific settings"""
    model_config = SettingsConfigDict(env_prefix='API_', extra='ignore')

    base_path: str = Field(default="/enterprise-ai", description="API base path for sub-path deployment")
    title: str = Field(default="Production Fine-Tuning Service", description="API title")
    version: str = Field(default="1.0.0", description="API version")
    description: str = Field(
        default="Enterprise-grade API for fine-tuning large language models",
        description="API description"
    )

    # CORS settings - Industry best practices for production security
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(default=True, description="Allow credentials in CORS")
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        description="Allowed HTTP methods"
    )
    cors_allow_headers: List[str] = Field(
        default=[
            "Authorization",      # JWT Bearer tokens (Keycloak OIDC)
            "Content-Type",       # Request body type (application/json)
            "Accept",             # Response type negotiation
            "ft-api-key",         # Custom header for nvidia backend
            "X-Requested-With",   # Standard header for AJAX requests
            "Cache-Control",      # Cache directives
            "Pragma",             # HTTP/1.0 cache control
            "Origin",             # CORS origin header
            "User-Agent",         # Client identification
            "Accept-Language",    # Language preferences
            "Accept-Encoding",    # Compression preferences
        ],
        description="Explicitly allowed CORS headers (no wildcards for security)"
    )
    cors_expose_headers: List[str] = Field(
        default=[
            "X-Process-Time",     # Request processing duration
            "X-RateLimit-Limit",  # Rate limit maximum
            "X-RateLimit-Remaining",  # Rate limit remaining
            "X-RateLimit-Reset",  # Rate limit reset time
        ],
        description="Headers exposed to client JavaScript"
    )


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration"""
    model_config = SettingsConfigDict(env_prefix='RATE_LIMIT_', extra='ignore')

    # Global rate limiting toggle
    enabled: bool = Field(default=True, description="Enable rate limiting globally")

    # Default rate limit for all endpoints
    default: int = Field(default=200, ge=1, le=10000, description="Default requests per minute")

    # Health check endpoints (lightweight operations)
    health: int = Field(default=100, ge=1, le=10000, description="Health check requests per minute")

    # Model endpoints (read operations)
    models: int = Field(default=100, ge=1, le=10000, description="Model list/retrieve requests per minute")

    # Job creation (resource-intensive)
    job_create: int = Field(default=10, ge=1, le=1000, description="Job creation requests per minute")

    # Job read operations (listing, retrieve, events)
    job_read: int = Field(default=60, ge=1, le=10000, description="Job read requests per minute")

    # Job cancellation
    job_cancel: int = Field(default=20, ge=1, le=1000, description="Job cancel requests per minute")

    # Job events streaming
    job_events: int = Field(default=60, ge=1, le=10000, description="Job events requests per minute")


class GatewaySettings(BaseSettings):
    """
    The GenAI Gateway (LiteLLM).

    Needed to read which models are registered and to manage the semantic
    auto-router. Egress to the gateway's port has to be open in the API's
    NetworkPolicy -- it is not in the general HTTP allowlist, so a missing rule
    shows up as a timeout rather than a refusal.
    """
    model_config = SettingsConfigDict(env_prefix='GATEWAY_', extra='ignore')

    url: Optional[str] = Field(default=None, description="Gateway base URL")
    master_key: Optional[str] = Field(default=None, description="Gateway admin key")
    timeout: float = Field(default=30.0, ge=5, le=300, description="Request timeout in seconds")

    # One router shared by every fine-tuned model, so callers point at a single
    # name and routes accumulate as models are added. The highest-scoring route
    # wins, which is semantic-router's own behaviour.
    router_name: str = Field(default="smart", description="Model name clients call to be routed")
    # An auto-router is cached in the gateway process by model name and is not
    # refreshed by re-registering it, and this build has no /config/reload -- so
    # applying a change requires restarting the gateway deployment.
    restart_on_apply: bool = Field(
        default=True, description="Restart the gateway after changing the router so the change takes effect"
    )
    namespace: str = Field(default="genai-gateway", description="Namespace the gateway runs in")
    deployment: str = Field(
        default="genai-gateway-deployment", description="Gateway Deployment to restart on apply"
    )


class ModelDeploymentSettings(BaseSettings):
    """
    Serving a fine-tuned model from the UI.

    The "Deploy Model" button runs the same Helm install the job detail page
    prints, as a Job in the inference namespace. The chart is supplied by a
    ConfigMap so this service does not have to carry a copy of it, and git stays
    the single source of the chart (the plugin playbook refreshes the ConfigMap).
    """
    model_config = SettingsConfigDict(env_prefix='MODEL_DEPLOY_', extra='ignore')

    enabled: bool = Field(default=True, description="Allow deploying models from the UI")
    namespace: str = Field(default="default", description="Namespace the vllm chart deploys into")
    chart_config_map: str = Field(default="vllm-chart", description="ConfigMap holding the packaged vllm chart")
    chart_archive_key: str = Field(default="vllm-chart.tgz", description="ConfigMap key of the chart archive")
    values_key: str = Field(default="values.yaml", description="ConfigMap key of the chart values file")
    helm_image: str = Field(default="alpine/helm:3.16.4", description="Image providing the helm binary")
    service_account: str = Field(
        default="ft-model-deployer",
        description="ServiceAccount the Helm Job runs as (needs write access in the inference namespace)"
    )
    # Off by default: a count says nothing about what a deployment costs. Ten 1B
    # models are cheaper than one 70B, so admission is decided by the per-request
    # CPU and memory check against real node capacity, which knows the difference.
    # A count is still available for installations that want a hard ceiling on how
    # many vLLM instances exist regardless of size.
    max_deployments: int = Field(
        default=0, ge=0, le=50,
        description="Maximum concurrent model deployments; 0 means no cap and capacity decides"
    )
    tensor_parallel_size: int = Field(default=1, ge=1, le=16, description="vLLM tensor parallel size")
    pipeline_parallel_size: int = Field(default=1, ge=1, le=16, description="vLLM pipeline parallel size")

    # Resource request for a served model. The vllm chart emits requests and
    # limits only when cpu/memory are set, so leaving these unset deploys a
    # BestEffort pod: the scheduler will place it on a node with nothing left and
    # it will be OOM-killed or starve the model already serving. The derived
    # figures are a starting point the user can change before deploying.
    cpu_cores_per_billion_params: float = Field(
        default=4.0, gt=0, le=64, description="Cores per billion parameters when sizing a deployment"
    )
    memory_gib_per_billion_params: float = Field(
        default=3.0, gt=0, le=128,
        description="GiB of weights and working memory per billion parameters (excludes the KV cache)"
    )
    # vLLM reserves this much for the KV cache whatever the model's size, so it is
    # a separate term in the estimate rather than folded into the per-parameter
    # figure. Only used when the chart's own value cannot be read.
    default_kv_cache_space_gib: int = Field(
        default=40, ge=1, le=512,
        description="Assumed VLLM_CPU_KVCACHE_SPACE when the packaged chart cannot be read"
    )
    memory_overhead_gib: int = Field(
        default=8, ge=0, le=256, description="Fixed GiB added on top of the per-parameter estimate"
    )
    default_cpu_cores: int = Field(
        default=16, ge=1, description="Cores requested when the parameter count cannot be read"
    )
    default_memory_gib: int = Field(
        default=16, ge=1,
        description="GiB of weights and working memory when the parameter count cannot be read"
    )
    # Absolute floors, used when the parameter count cannot be read from the model
    # id. When it can be read, the floor is derived from the model instead (see
    # the two figures below): a flat 1-2 cores and 8Gi is meaningless as a minimum
    # for a 14B model and needlessly high for a 0.5B one.
    min_cpu_cores: int = Field(default=2, ge=1, description="Floor on CPU when the model size is unknown")
    min_memory_gib: int = Field(default=8, ge=1, description="Floor on memory when the model size is unknown")
    # Weights at the serving dtype: bf16/fp16 is 2 bytes per parameter, so 2GiB per
    # billion. This is a hard requirement, not a recommendation -- below it the
    # weights do not fit in the container and vLLM is killed during load. The
    # recommendation uses memory_gib_per_billion_params (higher) to leave working
    # room on top.
    min_memory_gib_per_billion_params: float = Field(
        default=2.0, gt=0, le=128,
        description="GiB of weights per billion parameters at the serving dtype; the hard memory floor"
    )
    # Unlike memory, too few cores does not fail -- it is just slow, so this is a
    # usability floor rather than a physical one. One core per billion parameters
    # keeps a deployment from being configured into uselessness.
    min_cpu_cores_per_billion_params: float = Field(
        default=1.0, gt=0, le=64,
        description="Cores per billion parameters below which serving is impractically slow"
    )
    min_memory_overhead_gib: int = Field(
        default=2, ge=0, le=64,
        description="GiB for the runtime itself, added to the memory floor"
    )
    # Hard ceilings, only a backstop against a typo. The real ceiling is what the
    # roomiest node actually has, which is computed per request.
    max_cpu_cores: int = Field(default=1024, ge=1, description="Absolute ceiling on a CPU request")
    max_memory_gib: int = Field(default=8192, ge=1, description="Absolute ceiling on a memory request")
    # Applied to the roomiest node's allocatable, so a single model cannot take
    # the whole machine even when it is idle.
    max_node_fraction: float = Field(
        default=0.5, gt=0, le=1.0, description="Largest share of one node a single deployment may request"
    )
    # A capacity check is a snapshot and vLLM's real appetite is not exactly its
    # request, so the rejection has to be overridable — with a record of it.
    allow_capacity_override: bool = Field(
        default=True, description="Allow deploying with force=true when the request does not fit"
    )
    helm_timeout: str = Field(default="15m", description="Helm --timeout for install and uninstall")
    job_ttl_seconds: int = Field(default=3600, ge=60, description="How long finished Helm Jobs are kept")
    job_deadline_seconds: int = Field(default=1800, ge=120, description="Hard limit on a Helm Job's runtime")
    log_tail_lines: int = Field(default=40, ge=5, le=500, description="Log lines returned with deployment status")


class ObservabilitySettings(BaseSettings):
    """Observability and monitoring configuration"""
    model_config = SettingsConfigDict(env_prefix='OBSERVABILITY_', extra='ignore')

    # Global observability toggle
    enabled: bool = Field(default=True, description="Enable observability features (metrics, structured logging)")

    # Structured logging
    json_logs: bool = Field(default=True, description="Use JSON structured logging (recommended for production)")
    log_user_actions: bool = Field(default=True, description="Log user actions in metrics")

    # Prometheus metrics
    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics endpoint")
    track_user_metrics: bool = Field(default=True, description="Track per-user request metrics")

    # System metrics
    collect_system_metrics: bool = Field(default=True, description="Collect CPU/memory metrics")


class Settings(BaseSettings):
    """Main application settings"""
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False
    )

    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")

    # Security settings
    max_concurrent_jobs_per_user: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent active jobs per user (Resource consumption control)"
    )

    # Background reconcile of active jobs against the training engine. Without it
    # the jobs list only changes when someone opens a job's detail page, which is
    # the one place that refreshes the row.
    job_reconcile_enabled: bool = Field(
        default=True,
        description="Poll the training engine for active jobs in the background"
    )
    job_reconcile_interval_seconds: int = Field(
        default=15,
        ge=5,
        le=300,
        description="Seconds between background reconcile passes"
    )

    # Sub-configurations
    database: DatabaseSettings
    nvidia: NvidiaBackendSettings
    dataprep: DataPrepSettings
    api: APISettings
    rate_limit: RateLimitSettings
    observability: ObservabilitySettings
    deployment: ModelDeploymentSettings
    gateway: GatewaySettings

    def __init__(self, **kwargs):
        # Initialize sub-configurations
        if 'database' not in kwargs:
            kwargs['database'] = DatabaseSettings()
        if 'nvidia' not in kwargs:
            kwargs['nvidia'] = NvidiaBackendSettings()
        if 'dataprep' not in kwargs:
            kwargs['dataprep'] = DataPrepSettings()
        if 'api' not in kwargs:
            kwargs['api'] = APISettings()
        if 'rate_limit' not in kwargs:
            kwargs['rate_limit'] = RateLimitSettings()
        if 'observability' not in kwargs:
            kwargs['observability'] = ObservabilitySettings()
        if 'deployment' not in kwargs:
            kwargs['deployment'] = ModelDeploymentSettings()
        if 'gateway' not in kwargs:
            kwargs['gateway'] = GatewaySettings()

        super().__init__(**kwargs)

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get application settings singleton

    This function ensures settings are loaded only once and reused across the application.
    """
    global _settings
    if _settings is None:
        try:
            _settings = Settings()
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}") from e
    return _settings


def reload_settings() -> Settings:
    """Force reload settings (useful for testing)"""
    global _settings
    _settings = None
    return get_settings()

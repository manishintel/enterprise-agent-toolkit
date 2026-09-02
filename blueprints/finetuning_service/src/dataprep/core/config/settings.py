import os
from pathlib import Path
from typing import List


class Settings:
    """Application settings and configuration"""
    
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    METADATA_FILE: str = "files_metadata.json"
    
    # API settings
    API_TITLE: str = "Data Preparation Backend for Finetuning"
    API_VERSION: str = "1.0.0"
    HOST: str = os.getenv("HOST")
    PORT: int = int(os.getenv("PORT"))
    # Reverse-proxy prefix stripped by APISIX before reaching this service.
    # Set to /enterprise-ai/dataprep in production so FastAPI passes it as
    # root_path to Starlette — Swagger UI then constructs schema/docs URLs
    # as https://<host>/enterprise-ai/dataprep/openapi.json (which APISIX routes).
    API_BASE_PATH: str = os.getenv("API_BASE_PATH", "")

    # CORS settings - comma-separated list of allowed origins
    # Example: "http://localhost:3000,https://app.example.com"
    # Do NOT use "*" with allow_credentials=True (browsers block it)
    # NOTE: property reads os.environ at call-time so it picks up any runtime
    #       overrides without requiring a process restart in tests.
    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        """Return list of allowed CORS origins from the ALLOWED_ORIGINS env var."""
        raw = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    
    # PostgreSQL settings
    DB_HOST: str = os.getenv("DB_HOST")
    DB_PORT: int = int(os.getenv("DB_PORT"))
    DB_NAME: str = os.getenv("DB_NAME")
    DB_USER: str = os.getenv("DB_USER")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW"))
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE"))
    
    # MinIO/S3 settings
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY")
    MINIO_BUCKET_NAME: str = os.getenv("MINIO_BUCKET_NAME")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() in ["true", "1", "yes"]
    MINIO_REGION: str = os.getenv("MINIO_REGION")
    MINIO_CERT_VERIFY: bool = os.getenv("MINIO_CERT_VERIFY", "false").lower() in ["true", "1", "yes"]

    # Langfuse (optional): used by the /v1/langfuse import endpoints.
    LANGFUSE_URL: str = os.getenv(
        "LANGFUSE_URL",
        "http://genai-gateway-trace-web.genai-gateway.svc.cluster.local:3000",
    )
    # The default project: a Langfuse public API key is scoped to one project,
    # so this pair decides which project's traces the import page shows unless
    # another is selected.
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    # Additional projects to offer in the project dropdown, as
    # "publicKey:secretKey" pairs separated by commas/whitespace. The project id
    # and name are read back from Langfuse, so only the keys go here.
    LANGFUSE_PROJECT_KEYS: str = os.getenv("LANGFUSE_PROJECT_KEYS", "")
    # How long a resolved project list is reused. It only changes when the
    # configured keys change, so this is a courtesy to Langfuse, not a
    # consistency risk.
    LANGFUSE_PROJECT_CACHE_SECONDS: int = int(
        os.getenv("LANGFUSE_PROJECT_CACHE_SECONDS", "300")
    )
    LANGFUSE_TIMEOUT_SECONDS: int = int(os.getenv("LANGFUSE_TIMEOUT_SECONDS", "60"))
    LANGFUSE_MAX_TRACES_PER_IMPORT: int = int(os.getenv("LANGFUSE_MAX_TRACES_PER_IMPORT", "10000"))
    # Cap on GENERATION observations scanned when resolving which models have
    # traces. Langfuse cannot filter traces by model, so the model filter and
    # the model dropdown both walk observations; this bounds that walk.
    LANGFUSE_MAX_OBSERVATIONS_PER_SCAN: int = int(
        os.getenv("LANGFUSE_MAX_OBSERVATIONS_PER_SCAN", "5000")
    )
    # Cap on scores scanned when resolving which traces carry a given annotation
    # (human score). Same reason as the observation cap: traces cannot be
    # filtered by score, so the annotation filter walks scores instead.
    LANGFUSE_MAX_SCORES_PER_SCAN: int = int(
        os.getenv("LANGFUSE_MAX_SCORES_PER_SCAN", "10000")
    )
    # Cap on annotation-queue items scanned when filtering by queue.
    LANGFUSE_MAX_QUEUE_ITEMS_PER_SCAN: int = int(
        os.getenv("LANGFUSE_MAX_QUEUE_ITEMS_PER_SCAN", "10000")
    )

    # GenAI Gateway (LiteLLM): used to restrict the Langfuse model filter to
    # models that are actually deployed. Optional — if unreachable, the model
    # dropdown falls back to every model seen in traces.
    GENAI_GATEWAY_URL: str = os.getenv(
        "GENAI_GATEWAY_URL",
        "http://genai-gateway-service.genai-gateway.svc.cluster.local:4000",
    )
    LITELLM_MASTER_KEY: str = os.getenv("LITELLM_MASTER_KEY", "")
    GENAI_GATEWAY_TIMEOUT_SECONDS: int = int(os.getenv("GENAI_GATEWAY_TIMEOUT_SECONDS", "15"))

    @property
    def metadata_path(self) -> Path:
        """Get full path to metadata file"""
        return self.BASE_DIR / self.METADATA_FILE


settings = Settings()

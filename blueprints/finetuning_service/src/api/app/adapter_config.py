"""
Backend adapter configuration.

Shared by the job routers and the background status reconciler so both talk to
the training engine with identical settings and credentials.
"""

from typing import Any, Dict

from .config import get_settings
from .schemas import ResourceType

settings = get_settings()


def build_adapter_config(resource_type: ResourceType, current_user: Dict[str, Any]) -> Dict[str, Any]:
    """Build backend adapter configuration (backend auth is separate from user auth)."""
    user_id = current_user["user_id"]
    adapter_config: Dict[str, Any] = {
        "user_id": str(user_id),
        "user_uuid": str(user_id),
        "username": current_user.get("username", str(user_id)[:8])
    }

    if resource_type == ResourceType.NVIDIA:
        adapter_config.update({
            "nvidia_api_url": settings.nvidia.api_url,
            "api_timeout": settings.nvidia.api_timeout,
            "max_concurrent_jobs": settings.nvidia.max_jobs,
            "backend_auth_config": {
                "type": "oauth2_client_credentials",
                "token_url": settings.nvidia.keycloak_token_url,
                "client_id": settings.nvidia.keycloak_client_id,
                "client_secret": settings.nvidia.keycloak_client_secret,
                "verify_ssl": settings.nvidia.keycloak_verify_ssl,
                "refresh_buffer_seconds": 300,
                "timeout": 30.0
            }
        })

    return adapter_config

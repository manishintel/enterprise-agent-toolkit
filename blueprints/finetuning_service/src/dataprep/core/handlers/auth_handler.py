"""
Local user identity extraction (no Keycloak).

The UI proxies the resolved username in the `X-Forwarded-User` header
(fallback: the base64-encoded legacy bearer scheme is still accepted).
"""

import base64
import logging
from typing import Optional

from fastapi import Header, HTTPException
from fastapi.openapi.utils import get_openapi

from core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_USER = "default"


def _decode_base64_user(token: str) -> Optional[str]:
    try:
        decoded = base64.b64decode(token, validate=True).decode("utf-8").strip()
        return decoded or None
    except Exception:
        return None


async def get_current_user_id(
    x_forwarded_user: Optional[str] = Header(default=None, alias="X-Forwarded-User"),
    x_user: Optional[str] = Header(default=None, alias="X-User"),
    authorization: Optional[str] = Header(default=None),
) -> str:
    """
    Resolve user identity from (in priority order):
      1. `X-Forwarded-User` header (set by the UI proxy)
      2. `X-User` header
      3. `Authorization: Bearer <base64-username>` (legacy compatibility)
      4. `DEFAULT_USER` fallback
    """
    if x_forwarded_user and x_forwarded_user.strip():
        return x_forwarded_user.strip()
    if x_user and x_user.strip():
        return x_user.strip()

    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            decoded = _decode_base64_user(parts[1])
            if decoded:
                return decoded

    return DEFAULT_USER


def configure_openapi_auth(app):
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        description="File and Data Preparation Backend API's",
        routes=app.routes,
    )

    openapi_schema.setdefault("components", {})["securitySchemes"] = {
        "ForwardedUser": {
            "type": "apiKey",
            "in": "header",
            "name": "X-Forwarded-User",
            "description": "Username set by the UI proxy (fine-tuning-ui)",
        }
    }
    openapi_schema["security"] = [{"ForwardedUser": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Preserved for callers that still reference these names (no-op / legacy path)
async def validate_keycloak_token(_token: str) -> str:
    return DEFAULT_USER


def validate_base64_token(token: str) -> str:
    decoded = _decode_base64_user(token)
    if not decoded:
        raise HTTPException(status_code=401, detail="Invalid token")
    return decoded


__all__ = [
    "get_current_user_id",
    "configure_openapi_auth",
    "validate_keycloak_token",
    "validate_base64_token",
    "DEFAULT_USER",
]

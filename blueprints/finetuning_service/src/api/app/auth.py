"""
Local user identity extraction (no Keycloak, no JWT validation).

Auth is enforced at the UI layer via NextAuth. The UI proxies API requests
with the resolved username in the `X-Forwarded-User` header. This module
turns that header into the same `current_user` dict shape the rest of the
codebase expects, so `Depends(get_current_user)` keeps working unchanged.
"""

import logging
from typing import Dict, Any, Optional

from fastapi import Depends, HTTPException, Request, status

logger = logging.getLogger(__name__)

DEFAULT_USER = "default"


class AuthManager:
    """Kept for backward compatibility with main.py imports. No-op."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def extract_user_from_token(self, _token: str) -> Dict[str, Any]:
        return _anonymous_identity()

    @staticmethod
    def has_role(_user: Dict[str, Any], _required_role: str) -> bool:
        return True

    @staticmethod
    def has_any_role(_user: Dict[str, Any], _required_roles: list) -> bool:
        return True


auth_manager: Optional[AuthManager] = None


def _anonymous_identity(username: str = DEFAULT_USER) -> Dict[str, Any]:
    return {
        "user_id": username,
        "email": f"{username}@local",
        "username": username,
        "token": "",
        "roles": [],
        "client_roles": {},
    }


async def get_current_user(request: Request) -> Dict[str, Any]:
    """
    Resolve the current user from the `X-Forwarded-User` header set by the UI.
    Falls back to a shared default identity when the header is missing.
    """
    username = (
        request.headers.get("x-forwarded-user")
        or request.headers.get("x-user")
        or DEFAULT_USER
    ).strip()
    if not username:
        username = DEFAULT_USER
    identity = _anonymous_identity(username)
    request.state.user = identity
    return identity


async def require_role(_required_role: str):
    """Preserved for API compatibility; roles are not enforced without Keycloak."""

    async def _checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        _ = current_user  # ensure dependency chain still runs
        return None

    return _checker


# Explicitly export symbols the rest of the app imports.
__all__ = ["AuthManager", "auth_manager", "get_current_user", "require_role"]

# Silence "unused import" warnings while keeping HTTPException/status importable
# for downstream code that historically imported them from this module.
_ = HTTPException
_ = status

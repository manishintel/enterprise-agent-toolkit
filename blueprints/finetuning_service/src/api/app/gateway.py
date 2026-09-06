"""
The GenAI Gateway (LiteLLM) admin surface this service needs.

Only three things: read which models are registered, create or delete a model
entry, and embed text with the registered embedding model.

Two properties of the gateway shape everything here.

**Models are runtime state, not configuration.** The deployed gateway has no
``model_list`` in its ConfigMap: every model is created through ``/model/new``
and persisted in the gateway's own Postgres. So a semantic router is created the
same way any model is, and ``/model/info`` reads it back in clear -- including
``auto_router_config``. That makes the gateway the single source of truth for
router state, with nothing to keep in step on this side.

**An auto-router is cached in the gateway process, keyed by model name.** Verified
on 2026-09-03: re-registering the same name with a changed threshold kept routing
on the old config minutes later, while a fresh name honoured it immediately. There
is no ``/config/reload`` route in this build, so applying a change means
restarting the gateway (see ``restart_gateway``) -- deleting and recreating the
entry is not enough.
"""

import json
from typing import Any, Dict, List, Optional

import httpx

from .config import get_settings
from .observability import get_logger

logger = get_logger(__name__)
settings = get_settings()


class GatewayError(Exception):
    """A gateway call failed."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status = status


class GatewayUnavailable(GatewayError):
    """The gateway is not configured or cannot be reached."""


class GatewayClient:
    """Async client for the handful of gateway calls this service makes."""

    def __init__(self) -> None:
        config = settings.gateway
        self._url = (config.url or "").rstrip("/")
        self._key = config.master_key or ""
        self._timeout = config.timeout

    @property
    def configured(self) -> bool:
        """True when both the URL and the master key are present."""
        return bool(self._url and self._key)

    async def _request(
        self, method: str, path: str, body: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None
    ) -> Any:
        if not self.configured:
            raise GatewayUnavailable(
                "The GenAI Gateway is not configured for this service (GATEWAY_URL and "
                "GATEWAY_MASTER_KEY). Semantic routing needs both."
            )
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=timeout or self._timeout) as client:
                response = await client.request(method, f"{self._url}{path}", json=body, headers=headers)
        except httpx.HTTPError as exc:
            # Egress to the gateway's port has to be open in the API's
            # NetworkPolicy; without it this is a timeout rather than a refusal.
            raise GatewayUnavailable(f"GenAI Gateway unreachable: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text
            try:
                payload = response.json()
                detail = (payload.get("error") or {}).get("message") or payload.get("detail") or detail
            except ValueError:
                pass
            raise GatewayError(str(detail)[:600], status=response.status_code)

        try:
            return response.json()
        except ValueError:
            return None

    async def list_models(self) -> List[Dict[str, Any]]:
        result = await self._request("GET", "/model/info")
        return (result or {}).get("data", [])

    async def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        for model in await self.list_models():
            if model.get("model_name") == model_name:
                return model
        return None

    async def chat_models(self) -> List[str]:
        """Names of models that can answer a chat request, for the fallback list."""
        return [
            m["model_name"]
            for m in await self.list_models()
            if (m.get("model_info") or {}).get("mode") == "chat"
        ]

    async def embedding_models(self) -> List[str]:
        return [
            m["model_name"]
            for m in await self.list_models()
            if (m.get("model_info") or {}).get("mode") == "embedding"
        ]

    async def delete_model(self, model_id: str) -> None:
        try:
            await self._request("POST", "/model/delete", {"id": model_id})
        except GatewayError as exc:
            # Already absent is the desired state, not a failure.
            if exc.status in (400, 404):
                logger.info(f"Gateway model {model_id} was already absent")
                return
            raise

    async def create_auto_router(
        self,
        router_name: str,
        routes: List[Dict[str, Any]],
        default_model: str,
        embedding_model: str,
    ) -> Dict[str, Any]:
        """
        Register a semantic auto-router.

        ``routes`` entries are ``{name, utterances, score_threshold, description}``
        where **name is the model to route to** -- LiteLLM resolves the chosen
        route's name as the target model, so it has to match a registered model.
        """
        payload = {
            "model_name": router_name,
            "litellm_params": {
                "model": f"auto_router/{router_name}",
                # Inline JSON rather than a config path: a path would need a file
                # mounted into the gateway pod, and the gateway is not ours to mount into.
                "auto_router_config": json.dumps({"routes": routes}),
                "auto_router_default_model": default_model,
                "auto_router_embedding_model": embedding_model,
            },
            "model_info": {"id": router_name},
        }
        return await self._request("POST", "/model/new", payload)

    async def embed(self, texts: List[str], model: str) -> List[List[float]]:
        """
        Embed texts with a registered embedding model.

        Sends ``encoding_format`` explicitly: LiteLLM forwards the field as null
        when a caller omits it, and vLLM rejects null with a validation error.
        """
        result = await self._request(
            "POST",
            "/v1/embeddings",
            {"model": model, "input": texts, "encoding_format": "float"},
            timeout=max(self._timeout, 120.0),
        )
        rows = sorted((result or {}).get("data", []), key=lambda r: r.get("index", 0))
        return [row["embedding"] for row in rows]


gateway_client = GatewayClient()

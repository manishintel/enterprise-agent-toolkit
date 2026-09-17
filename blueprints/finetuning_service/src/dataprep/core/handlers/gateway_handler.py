"""
GenAI Gateway (LiteLLM) model registry client.

Only used to answer "which models are actually deployed, and which of them can
hold a conversation?" so the Langfuse import UI can offer a model filter limited
to live chat models. Nothing here talks to Langfuse, MinIO or Postgres.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Set

import requests

from core.config import settings

logger = logging.getLogger(__name__)

# LiteLLM's `model_info.mode`. Only models that answer a chat/completion request
# can contribute a user/assistant exchange to a fine-tuning dataset; an
# embedding deployment produces vectors and never an assistant turn.
_CONVERSATIONAL_MODES = {"chat", "completion", "responses"}


class GatewayConfigError(RuntimeError):
    """Raised when the gateway URL is missing."""


class GatewayFetchError(RuntimeError):
    """Raised when the gateway is unreachable or returns a non-2xx response."""


class GatewayClient:
    """Thin sync client over the LiteLLM proxy's OpenAI-compatible model list."""

    def __init__(
        self,
        url: Optional[str] = None,
        master_key: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        self.url = (url or settings.GENAI_GATEWAY_URL).rstrip("/")
        self.master_key = master_key or settings.LITELLM_MASTER_KEY
        self.timeout = timeout_seconds or settings.GENAI_GATEWAY_TIMEOUT_SECONDS

        if not self.url:
            raise GatewayConfigError(
                "GenAI gateway is not configured. Set GENAI_GATEWAY_URL."
            )

    def list_deployed_models(self) -> Set[str]:
        """Return the set of model names currently registered on the gateway."""
        headers = {"Accept": "application/json"}
        if self.master_key:
            headers["Authorization"] = f"Bearer {self.master_key}"

        try:
            resp = requests.get(
                f"{self.url}/v1/models",
                params={"return_wildcard_routes": "false"},
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise GatewayFetchError(f"GenAI gateway unreachable: {e}") from e

        if resp.status_code >= 400:
            raise GatewayFetchError(
                f"GenAI gateway {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json().get("data", []) or []
        except ValueError as e:
            raise GatewayFetchError(f"GenAI gateway returned invalid JSON: {e}") from e

        models = {
            str(m["id"]).strip()
            for m in data
            if isinstance(m, dict) and m.get("id")
        }
        # LiteLLM emits a literal "*" pseudo-model when wildcard routing is on.
        return {m for m in models if m and m != "*"}

    def _model_modes(self) -> Dict[str, str]:
        """
        Map deployed model name -> LiteLLM ``model_info.mode``, or ``{}``.

        ``/v1/models`` is the OpenAI-compatible listing and carries no mode, so
        the kind of a deployment can only come from LiteLLM's own
        ``/model/info``. That route is admin-only, so it is treated as optional:
        every failure returns an empty map and the caller falls back to the
        unfiltered list rather than to an empty one.
        """
        headers = {"Accept": "application/json"}
        if self.master_key:
            headers["Authorization"] = f"Bearer {self.master_key}"

        try:
            resp = requests.get(
                f"{self.url}/model/info", headers=headers, timeout=self.timeout
            )
            if resp.status_code >= 400:
                logger.warning(
                    "Gateway /model/info returned %s; cannot tell embedding "
                    "deployments from chat ones",
                    resp.status_code,
                )
                return {}
            data = resp.json().get("data", []) or []
        except (requests.RequestException, ValueError) as e:
            logger.warning("Gateway /model/info unavailable: %s", e)
            return {}

        modes: Dict[str, str] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("model_name") or "").strip()
            mode = ((entry.get("model_info") or {}) if isinstance(
                entry.get("model_info"), dict
            ) else {}).get("mode")
            if name and isinstance(mode, str) and mode.strip():
                modes[name] = mode.strip().lower()
        return modes

    def list_conversational_models(self) -> Set[str]:
        """
        Deployed models that can produce an assistant turn.

        Used for the Langfuse import model filter: offering an embedding
        deployment there invites an import that can only ever come back empty,
        because an embedding call has no assistant response to learn from.

        Models whose mode the gateway does not report are kept. An unknown mode
        is far more likely to be a chat model registered without metadata than an
        embedding one, and dropping it would hide a legitimate option.
        """
        deployed = self.list_deployed_models()
        modes = self._model_modes()
        if not modes:
            return deployed

        kept = set()
        for model in deployed:
            mode = modes.get(model)
            if mode is None:
                mode = next(
                    (m for name, m in modes.items() if models_match(model, name)),
                    None,
                )
            if mode is None or mode in _CONVERSATIONAL_MODES:
                kept.add(model)
        return kept


def normalize_model_name(name: str) -> str:
    """Lowercase + trim, so gateway and Langfuse spellings compare cleanly."""
    return (name or "").strip().lower()


def models_match(langfuse_name: str, gateway_name: str) -> bool:
    """
    True if a model seen in Langfuse is the same model the gateway serves.

    LiteLLM registers a public ``model_name`` (e.g. ``Qwen/Qwen2.5-Coder-14B-Instruct``)
    but traces may record the provider-prefixed target instead
    (``openai/Qwen/Qwen2.5-Coder-14B-Instruct``), so a trailing-segment match is
    accepted in either direction.
    """
    a = normalize_model_name(langfuse_name)
    b = normalize_model_name(gateway_name)
    if not a or not b:
        return False
    return a == b or a.endswith(f"/{b}") or b.endswith(f"/{a}")

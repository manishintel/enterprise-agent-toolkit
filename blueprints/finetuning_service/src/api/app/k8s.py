"""
Minimal in-cluster Kubernetes client.

Only what deploying a fine-tuned model needs: create, read and delete a Job,
read pods and their logs, and read a Deployment or ConfigMap. It talks to the
apiserver over httpx using the pod's own projected ServiceAccount token, so the
image does not have to carry the full kubernetes client (and its transitive
dependencies) for six API calls.

The token is read on every request because projected tokens are rotated by the
kubelet, so caching it would eventually 401.
"""

import os
from typing import Any, Dict, List, Optional

import httpx

from .observability import get_logger

logger = get_logger(__name__)

SERVICE_ACCOUNT_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
TOKEN_PATH = f"{SERVICE_ACCOUNT_DIR}/token"
CA_PATH = f"{SERVICE_ACCOUNT_DIR}/ca.crt"
NAMESPACE_PATH = f"{SERVICE_ACCOUNT_DIR}/namespace"


class KubeApiError(Exception):
    """An apiserver request failed."""

    def __init__(self, status: int, message: str, reason: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.reason = reason


class KubernetesClient:
    """Async client for the handful of apiserver calls this service makes."""

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout
        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        self._base_url = f"https://{host}:{port}" if host else None

    @property
    def available(self) -> bool:
        """True when running in a pod with a mounted ServiceAccount token."""
        return bool(self._base_url) and os.path.exists(TOKEN_PATH)

    @property
    def pod_namespace(self) -> Optional[str]:
        try:
            with open(NAMESPACE_PATH, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        except OSError:
            return None

    def _token(self) -> str:
        with open(TOKEN_PATH, "r", encoding="utf-8") as handle:
            return handle.read().strip()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        text_response: bool = False,
        missing_ok: bool = False,
    ) -> Any:
        """
        Perform one apiserver call.

        Returns ``None`` for a 404 when ``missing_ok`` is set: "the Job does not
        exist" is an expected answer for every status lookup, not an error.
        """
        if not self.available:
            raise KubeApiError(503, "Not running inside a Kubernetes cluster")

        headers = {
            "Authorization": f"Bearer {self._token()}",
            # The log endpoint streams plain text but only advertises the
            # structured media types, so asking for text/plain gets a 406.
            "Accept": "*/*" if text_response else "application/json",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"

        verify = CA_PATH if os.path.exists(CA_PATH) else True

        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, verify=verify, timeout=self._timeout
            ) as client:
                response = await client.request(
                    method, path, json=body, params=params, headers=headers
                )
        except httpx.HTTPError as exc:
            raise KubeApiError(503, f"Kubernetes API unreachable: {exc}") from exc

        if response.status_code == 404 and missing_ok:
            return None

        if response.status_code >= 400:
            reason, message = None, response.text
            try:
                payload = response.json()
                reason = payload.get("reason")
                message = payload.get("message", message)
            except ValueError:
                pass
            logger.warning(
                "Kubernetes API call failed",
                extra={
                    "method": method,
                    "path": path,
                    "status": response.status_code,
                    "reason": reason,
                },
            )
            raise KubeApiError(response.status_code, message, reason)

        if text_response:
            return response.text
        return response.json()

    # --- ConfigMaps ------------------------------------------------------

    async def get_config_map(self, namespace: str, name: str) -> Optional[Dict[str, Any]]:
        return await self._request(
            "GET",
            f"/api/v1/namespaces/{namespace}/configmaps/{name}",
            missing_ok=True,
        )

    # --- Jobs ------------------------------------------------------------

    async def get_job(self, namespace: str, name: str) -> Optional[Dict[str, Any]]:
        return await self._request(
            "GET",
            f"/apis/batch/v1/namespaces/{namespace}/jobs/{name}",
            missing_ok=True,
        )

    async def create_job(self, namespace: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/apis/batch/v1/namespaces/{namespace}/jobs",
            body=manifest,
        )

    async def delete_job(self, namespace: str, name: str) -> Optional[Dict[str, Any]]:
        # Background propagation so the Job's pods go with it.
        return await self._request(
            "DELETE",
            f"/apis/batch/v1/namespaces/{namespace}/jobs/{name}",
            params={"propagationPolicy": "Background"},
            missing_ok=True,
        )

    # --- Pods ------------------------------------------------------------

    async def list_pods(
        self, namespace: str, label_selector: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        params = {"labelSelector": label_selector} if label_selector else None
        result = await self._request(
            "GET", f"/api/v1/namespaces/{namespace}/pods", params=params
        )
        return (result or {}).get("items", [])

    async def read_pod_log(
        self,
        namespace: str,
        name: str,
        container: Optional[str] = None,
        tail_lines: int = 40,
    ) -> str:
        params: Dict[str, Any] = {"tailLines": tail_lines}
        if container:
            params["container"] = container
        try:
            log = await self._request(
                "GET",
                f"/api/v1/namespaces/{namespace}/pods/{name}/log",
                params=params,
                text_response=True,
                missing_ok=True,
            )
        except KubeApiError as exc:
            # A container that has not started yet cannot be read; that is a
            # normal state during a deployment, not a failure to report.
            logger.debug(f"Pod log unavailable for {name}: {exc.message}")
            return ""
        return log or ""

    # --- Deployments -----------------------------------------------------

    async def get_deployment(self, namespace: str, name: str) -> Optional[Dict[str, Any]]:
        return await self._request(
            "GET",
            f"/apis/apps/v1/namespaces/{namespace}/deployments/{name}",
            missing_ok=True,
        )

    async def list_deployments(
        self, namespace: str, label_selector: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        params = {"labelSelector": label_selector} if label_selector else None
        result = await self._request(
            "GET", f"/apis/apps/v1/namespaces/{namespace}/deployments", params=params
        )
        return (result or {}).get("items", [])


kube_client = KubernetesClient()

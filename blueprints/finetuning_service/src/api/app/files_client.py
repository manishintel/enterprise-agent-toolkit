"""
Reading a training file back out of the Data Prep files API.

Files are owner-scoped there, and the identity travels in the headers rather than
in a token this service holds: ``X-Forwarded-User``, plus the legacy
``Authorization: Bearer <base64-username>`` that the same API also accepts. A
mismatched identity gets **404, not 403**, so "file not found" here usually means
"asked as the wrong user".
"""

import base64
import json
from typing import Any, Dict, List

import httpx

from .config import get_settings
from .errors import InvalidRequestError, ServiceUnavailableError
from .observability import get_logger

logger = get_logger(__name__)
settings = get_settings()

# A dataset large enough to exceed this is not something to hold in memory to
# mine a few dozen utterances from.
MAX_BYTES = 64 * 1024 * 1024


def _identity_headers(user_id: str) -> Dict[str, str]:
    token = base64.b64encode(user_id.encode()).decode()
    return {
        "X-Forwarded-User": user_id,
        "X-User": user_id,
        "Authorization": f"Bearer {token}",
    }


async def read_training_file(file_id: str, user_id: str) -> List[Dict[str, Any]]:
    """
    Fetch a JSONL training file and parse it into rows.

    Malformed lines are skipped rather than failing the request: a single bad line
    should not stop utterances being mined from the thousands of good ones, and the
    count of skipped lines is logged.
    """
    base = (settings.dataprep.api_url or "").rstrip("/")
    if not base:
        raise ServiceUnavailableError(
            "The Data Prep service is not configured for this API (DATAPREP_API_URL), so the "
            "training dataset cannot be read."
        )

    url = f"{base}/v1/files/{file_id}/content"
    try:
        async with httpx.AsyncClient(
            timeout=max(settings.dataprep.timeout, 60), verify=settings.dataprep.verify_ssl
        ) as client:
            response = await client.get(url, headers=_identity_headers(user_id))
    except httpx.HTTPError as exc:
        raise ServiceUnavailableError(f"Could not reach the Data Prep service: {exc}") from exc

    if response.status_code == 404:
        raise InvalidRequestError(
            f"Training file {file_id} was not found for this user. The files API scopes by owner "
            f"and answers 404 rather than 403 when the identity does not match.",
            param="training_file", code="training_file_not_found",
        )
    if response.status_code >= 400:
        raise ServiceUnavailableError(
            f"Data Prep returned {response.status_code} for {file_id}: {response.text[:200]}"
        )

    raw = response.content
    if len(raw) > MAX_BYTES:
        raise InvalidRequestError(
            f"Training file {file_id} is {len(raw) // (1024 * 1024)}MB, over the "
            f"{MAX_BYTES // (1024 * 1024)}MB limit for utterance extraction.",
            param="training_file", code="training_file_too_large",
        )

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
        else:
            skipped += 1

    if skipped:
        logger.info(f"Skipped {skipped} unparseable line(s) in {file_id}")
    return rows

"""NVIDIA/Unsloth Resource Adapter

Handles fine-tuning jobs on NVIDIA GPUs using Unsloth via REST API.
Converts between OpenAI format and Unsloth API format.

Authentication: Uses Keycloak OAuth2 client credentials flow.
"""

import httpx
import json
import logging
from typing import Dict, Any, List, Optional, Final
from datetime import datetime, timezone

from ..base import ResourceAdapter
from ...schemas import (
    JobSubmissionRequest,
    JobSubmissionResponse,
    JobStatusRequest,
    JobStatusResponse,
    JobStatus
)
from ...utils.backend_auth import BackendAuthProvider, create_backend_auth

logger = logging.getLogger(__name__)

# Constants
DEFAULT_BATCH_SIZE: Final[int] = 2
DEFAULT_LEARNING_RATE: Final[float] = 0.0002
DEFAULT_LORA_R: Final[int] = 16
DEFAULT_MAX_SEQ_LENGTH: Final[int] = 2048
DEFAULT_NUM_EPOCHS: Final[int] = 3
DEFAULT_TIMEOUT: Final[float] = 120.0
STATUS_CHECK_TIMEOUT: Final[float] = 30.0
TOKEN_REFRESH_BUFFER: Final[int] = 300
ESTIMATED_JOB_DURATION: Final[int] = 7200
BACKEND_ERROR_DETAIL_MAX_LEN: Final[int] = 500
# How far ahead of our clock an engine timestamp may be before it is treated as a
# clock problem rather than a fact. Ordinary NTP drift is seconds, not minutes.
ENGINE_CLOCK_SKEW_TOLERANCE: Final[int] = 300
# Timezone offsets are whole multiples of 15 minutes, and a genuine queue wait
# lands within a couple of seconds of one only by coincidence.
QUARTER_HOUR: Final[int] = 900
ENGINE_CLOCK_SKEW_EPSILON: Final[float] = 2.0


def _now_ts() -> int:
    """Current time as a Unix timestamp, independent of the container timezone."""
    return int(datetime.now(timezone.utc).timestamp())


def _format_seconds(seconds: int) -> str:
    """Render a duration the way the event messages read best: '7h 11m', '2m 54s'."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _parse_backend_timestamp(value: Any) -> Optional[float]:
    """
    Convert a timestamp reported by the training engine to a Unix timestamp.

    The engine sends ISO-8601 strings (sometimes with a trailing ``Z``, sometimes
    already offset-aware) and older payloads carry raw epoch numbers. Returns
    ``None`` when the value is missing or unparseable so callers keep whatever is
    already on record instead of stamping the current time. The fraction is kept:
    it is what tells a timezone offset apart from a real wait in
    :func:`_engine_clock_offset`.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError) as exc:
        logger.warning(f"Could not parse backend timestamp '{value}': {exc}")
        return None
    if parsed.tzinfo is None:
        # Naive values from the engine are UTC; assuming otherwise would shift
        # the whole timeline by the container's offset.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _engine_clock_offset(job_data: Dict[str, Any], job_id: str) -> float:
    """
    Measure how far the engine's start/finish clock runs ahead of its own submit clock.

    The training service stamps ``created_at`` from a UTC clock but ``started_at``
    and ``completed_at`` from the training host's local clock, labelling all three
    ``Z``. The gap between submit and start is then that host's UTC offset rather
    than a queue wait, which is how a two-minute fine-tune ends up reporting seven
    hours of waiting: observed jobs sat "queued" for exactly 25200s, to within half
    a second, while their result files landed minutes after submission.

    Timezone offsets are whole multiples of 15 minutes and a genuine wait lands
    within a couple of seconds of one only by coincidence, so a gap that does is
    treated as the offset and subtracted. Anything else is left alone and reported
    as the wait it is.
    """
    created = _parse_backend_timestamp(job_data.get('created_at'))
    started = _parse_backend_timestamp(job_data.get('started_at'))
    if created is None or started is None:
        return 0.0
    gap = started - created
    if gap < QUARTER_HOUR / 2:
        return 0.0
    nearest = round(gap / QUARTER_HOUR) * QUARTER_HOUR
    if nearest < QUARTER_HOUR or abs(gap - nearest) > ENGINE_CLOCK_SKEW_EPSILON:
        return 0.0
    logger.warning(
        f"Engine start/finish times for job {job_id} run {nearest}s ahead of its own "
        f"submit time (residual {gap - nearest:+.3f}s); treating that as the training "
        f"host's UTC offset, not queue wait"
    )
    return float(nearest)


def _engine_timestamp(
    value: Any, field: str, job_id: str, offset: float = 0.0
) -> Optional[int]:
    """
    Parse an engine timestamp, undo any measured clock offset, and drop it if it
    still cannot be true.

    Training cannot have started or finished in the future, so a timestamp ahead
    of our own clock means the engine's clock (or its timezone handling) is off in
    a way ``offset`` did not account for. Recording it anyway is what produces
    impossible job pages — hours of "queue wait" that never happened, or a job
    that finishes before it is submitted. The caller falls back to what it can
    observe itself instead.
    """
    parsed = _parse_backend_timestamp(value)
    if parsed is None:
        return None
    corrected = int(parsed - offset)
    limit = _now_ts() + ENGINE_CLOCK_SKEW_TOLERANCE
    if corrected > limit:
        logger.warning(
            f"Ignoring engine {field} '{value}' for job {job_id}: "
            f"{corrected - limit}s in the future, the engine clock looks wrong"
        )
        return None
    return corrected


class NvidiaAdapter(ResourceAdapter):
    """Resource adapter for NVIDIA GPUs using Unsloth (non-OpenAI compatible REST API)"""

    def __init__(self, db_pool, config: Dict[str, Any] = None):
        super().__init__(db_pool, config)

        if not config:
            raise ValueError("Configuration required for NvidiaAdapter")

        # API configuration
        self.nvidia_api_url = config.get('nvidia_api_url', '')
        if not self.nvidia_api_url:
            raise ValueError("'nvidia_api_url' is required in config")

        self.api_timeout = config.get('api_timeout', DEFAULT_TIMEOUT)
        self.verify_ssl = config.get('verify_ssl', True)
        self.max_concurrent_jobs = config.get('max_concurrent_jobs', 1)

        # Initialize backend authentication (separate from user Keycloak auth)
        self.backend_auth: Optional[BackendAuthProvider] = config.get("backend_auth")
        if not self.backend_auth:
            auth_config = config.get("backend_auth_config")

            if not auth_config:
                # Backward compatibility for legacy Nvidia Keycloak config keys
                token_url = config.get('nvidia_keycloak_token_url')
                client_id = config.get('nvidia_keycloak_client_id')
                client_secret = config.get('nvidia_keycloak_client_secret')
                keycloak_verify_ssl = config.get('nvidia_keycloak_verify_ssl', True)

                if not all([token_url, client_id, client_secret]):
                    raise ValueError(
                        "Nvidia backend requires backend_auth_config or legacy Keycloak configuration: "
                        "nvidia_keycloak_token_url, nvidia_keycloak_client_id, nvidia_keycloak_client_secret"
                    )

                auth_config = {
                    "type": "oauth2_client_credentials",
                    "token_url": token_url,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "verify_ssl": keycloak_verify_ssl,
                    "refresh_buffer_seconds": TOKEN_REFRESH_BUFFER,
                    "timeout": 30.0
                }

            self.backend_auth = create_backend_auth(auth_config)

        logger.info("Initialized NvidiaAdapter with backend authentication")

    def _convert_openai_to_unsloth_request(self, request: JobSubmissionRequest) -> Dict[str, Any]:
        """
        Convert OpenAI format to Unsloth/Nvidia API format

        OpenAI format:
        - training_file: file ID (string)
        - hyperparameters: dict with n_epochs, batch_size, learning_rate, etc.

        Nvidia/Unsloth format:
        - username: string (required)
        - user_uuid: string (required for FILES API auth)
        - model_name: string (HuggingFace model name)
        - input_filenames: array of file IDs
        - hyperparameters: dict with num_train_epochs/max_steps, batch_size, learning_rate, etc.
        """
        hyperparams = request.hyperparameters or {}

        # Map OpenAI hyperparameters to Unsloth format
        unsloth_hyperparams = {
            "batch_size": hyperparams.get('batch_size', DEFAULT_BATCH_SIZE),
            "learning_rate": (
                hyperparams.get('learning_rate') or
                hyperparams.get('learning_rate_multiplier') or
                DEFAULT_LEARNING_RATE
            ),
            "lora_r": hyperparams.get('lora_r', DEFAULT_LORA_R),
            "max_seq_length": hyperparams.get('max_seq_length', DEFAULT_MAX_SEQ_LENGTH),
        }

        # Handle epochs vs steps (prefer max_steps if provided)
        max_steps = hyperparams.get('max_steps')
        if max_steps and max_steps > 0:
            unsloth_hyperparams['max_steps'] = max_steps
        else:
            unsloth_hyperparams['num_train_epochs'] = hyperparams.get('n_epochs', DEFAULT_NUM_EPOCHS)

        # Get user info from resource_config
        resource_config = request.resource_config or {}
        user_uuid = resource_config.get('user_uuid', request.user_id)
        username = resource_config.get('username', str(user_uuid)[:8])

        # Build Nvidia API request
        return {
            "username": username,
            "user_uuid": str(user_uuid),
            "model_name": request.model,
            "input_filenames": [request.training_file],  # OpenAI uses single file ID
            "hyperparameters": unsloth_hyperparams
        }

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> Optional[str]:
        """
        Pull the training engine's own error text out of a failed response.

        Unsloth/FastAPI errors look like ``{"detail": "No GPU available"}``, but
        fall back to ``message``/``error`` keys and finally the raw body so no
        engine-side cause is ever silently dropped.
        """
        try:
            payload = response.json()
        except ValueError:
            payload = None

        detail: Any = None
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("message") or payload.get("error")
        elif isinstance(payload, list):
            detail = payload

        if detail is not None and not isinstance(detail, str):
            detail = json.dumps(detail)
        if not detail:
            detail = (response.text or "").strip() or None

        if detail and len(detail) > BACKEND_ERROR_DETAIL_MAX_LEN:
            detail = detail[:BACKEND_ERROR_DETAIL_MAX_LEN] + "..."
        return detail

    def _map_unsloth_status_to_openai(self, unsloth_status: str) -> JobStatus:
        """Map Unsloth status to OpenAI JobStatus"""
        status_mapping = {
            "pending": JobStatus.QUEUED,
            "queued": JobStatus.QUEUED,
            "initializing": JobStatus.VALIDATING_FILES,
            "downloading_data": JobStatus.RUNNING,
            "preparing_environment": JobStatus.RUNNING,
            "running": JobStatus.RUNNING,  # Nvidia returns RUNNING
            "training": JobStatus.RUNNING,
            "merging": JobStatus.RUNNING,
            "uploading_model": JobStatus.RUNNING,
            "succeeded": JobStatus.SUCCEEDED,
            "completed": JobStatus.SUCCEEDED,  # Nvidia returns COMPLETED
            "failed": JobStatus.FAILED,  # Nvidia returns FAILED
            "cancelled": JobStatus.CANCELLED,
            "canceled": JobStatus.CANCELLED,
        }
        return status_mapping.get(unsloth_status.lower(), JobStatus.RUNNING)

    async def submit_job(self, request: JobSubmissionRequest) -> JobSubmissionResponse:
        """Submit fine-tuning job to Nvidia/Unsloth API"""

        try:
            # Convert OpenAI format to Unsloth format
            unsloth_request = self._convert_openai_to_unsloth_request(request)

            # Get OAuth2 token from Keycloak
            auth_header = await self.backend_auth.get_auth_header()

            # Submit to Unsloth API with Keycloak token
            headers = {
                **auth_header,  # Add Bearer token
                'ft-api-key': request.user_token,  # User's token for dataprep access
                'Content-Type': 'application/json'
            }

            logger.info(f"Submitting job {request.job_id} to Nvidia backend")

            async with httpx.AsyncClient(timeout=self.api_timeout, verify=self.verify_ssl) as client:
                response = await client.post(
                    f"{self.nvidia_api_url}/finetune/start",
                    json=unsloth_request,
                    headers=headers
                )

                if response.status_code in [200, 202]:
                    result = response.json()

                    # Unsloth API returns: {"message": "...", "job_id": 123, "status": "pending"}
                    resource_job_id = str(result.get('job_id', result.get('id')))

                    if not resource_job_id or resource_job_id == 'None':
                        return JobSubmissionResponse(
                            success=False,
                            error_message="Nvidia/Unsloth API did not return valid job ID"
                        )

                    logger.info(f"Successfully submitted job {request.job_id} to Nvidia/Unsloth API, resource_job_id: {resource_job_id}")

                    return JobSubmissionResponse(
                        success=True,
                        resource_job_id=resource_job_id,
                        estimated_duration=ESTIMATED_JOB_DURATION
                    )
                elif response.status_code == 401:
                    # Token might be invalid, try to refresh
                    logger.warning("Received 401 from Nvidia API, attempting token refresh")
                    self.backend_auth.invalidate()
                    error_msg = f"Authentication failed with status code {response.status_code}"
                    error_detail = self._extract_error_detail(response)
                    logger.error(f"Authentication error: {response.status_code} - {response.text}")
                    return JobSubmissionResponse(
                        success=False,
                        error_message=error_msg,
                        backend_status_code=response.status_code,
                        backend_error_detail=error_detail
                    )
                else:
                    error_msg = f"Backend API error: status code {response.status_code}"
                    error_detail = self._extract_error_detail(response)
                    logger.error(f"Nvidia API error: {response.status_code} - {response.text}")
                    return JobSubmissionResponse(
                        success=False,
                        error_message=error_msg,
                        backend_status_code=response.status_code,
                        backend_error_detail=error_detail
                    )

        except httpx.TimeoutException:
            logger.error(f"Timeout submitting job {request.job_id} to Nvidia/Unsloth API")
            return JobSubmissionResponse(
                success=False,
                error_message="Backend API request timeout",
                backend_error_detail=(
                    f"No response from the training engine at {self.nvidia_api_url} "
                    f"within {self.api_timeout}s"
                )
            )
        except Exception as e:
            logger.error(f"Error submitting Nvidia job {request.job_id}: {e}", exc_info=True)
            return JobSubmissionResponse(
                success=False,
                error_message="Failed to submit job to backend",
                backend_error_detail=f"{type(e).__name__}: {e}"
            )

    async def get_job_status(self, request: JobStatusRequest) -> JobStatusResponse:
        """Get job status from Nvidia/Unsloth API"""

        try:
            # Get OAuth2 token from Keycloak
            auth_header = await self.backend_auth.get_auth_header()

            async with httpx.AsyncClient(timeout=STATUS_CHECK_TIMEOUT, verify=self.verify_ssl) as client:
                response = await client.get(
                    f"{self.nvidia_api_url}/finetune/job/{request.resource_job_id}",
                    headers=auth_header
                )

                if response.status_code == 200:
                    job_data = response.json()
                    logger.debug(f"Backend response for job {request.job_id}: status={job_data.get('status')}")

                    unsloth_status = job_data.get("status", "failed")
                    job_status = self._map_unsloth_status_to_openai(unsloth_status)

                    async with self.db_pool.acquire() as conn:
                        error_json = None
                        if job_status == JobStatus.FAILED and job_data.get('error_log'):
                            error_json = json.dumps({"message": job_data.get('error_log')})

                        # Ensure output_file_id is a string or None, not a timestamp
                        output_file_id = job_data.get('output_file_id')
                        if output_file_id is not None and not isinstance(output_file_id, str):
                            output_file_id = str(output_file_id)

                        # Build result_files array
                        result_files_json = json.dumps([output_file_id]) if output_file_id else None
                        current_timestamp = _now_ts()

                        # A job starts and finishes once. Writing "now" into
                        # finished_at on every poll used to make a finished job's
                        # duration grow for as long as anyone kept the page open,
                        # so both are recorded once and never moved afterwards.
                        engine_offset = _engine_clock_offset(job_data, request.job_id)
                        started_at = _engine_timestamp(
                            job_data.get('started_at'), 'started_at', request.job_id, engine_offset
                        )
                        finished_at = _engine_timestamp(
                            job_data.get('completed_at'), 'completed_at', request.job_id, engine_offset
                        )
                        if started_at is None and job_status == JobStatus.RUNNING:
                            # The engine's start time is unusable but the job is
                            # running right now, so first sight of it running is
                            # the closest honest answer. COALESCE below keeps the
                            # first such observation.
                            started_at = current_timestamp

                        await conn.execute("""
                            UPDATE fine_tuning_jobs
                            SET status = $1::VARCHAR,
                                updated_at = $6::INTEGER,
                                started_at = COALESCE(started_at, $13::INTEGER),
                                finished_at = CASE
                                    WHEN $1::VARCHAR IN ('succeeded', 'failed', 'cancelled')
                                        THEN COALESCE(finished_at, $14::INTEGER, $6::INTEGER)
                                    ELSE finished_at
                                END,
                                fine_tuned_model = COALESCE($2::VARCHAR, fine_tuned_model),
                                error_message = COALESCE($3::TEXT, error_message),
                                result_files = COALESCE($5::TEXT::JSONB, result_files),
                                progress_percent = $7::REAL,
                                current_step = COALESCE($8::INTEGER, current_step),
                                total_steps = COALESCE($9::INTEGER, total_steps),
                                current_phase = COALESCE($10::VARCHAR, current_phase),
                                training_loss = COALESCE($12::REAL, training_loss),
                                elapsed_seconds = COALESCE($11::INTEGER, elapsed_seconds)
                            WHERE id = $4::VARCHAR
                        """, job_status.value, output_file_id, error_json, request.job_id, result_files_json, current_timestamp,
                            job_data.get("progress_percent", 0.0),
                            job_data.get("current_step"),
                            job_data.get("total_steps"),
                            job_data.get("current_phase"),
                            job_data.get("elapsed_seconds"),
                            job_data.get("training_loss"),
                            started_at,
                            finished_at)

                        # Report what is on record rather than this poll's reading:
                        # the first reading is the one that stuck, and it also
                        # covers a discarded engine timestamp.
                        recorded = await conn.fetchrow(
                            "SELECT started_at, finished_at FROM fine_tuning_jobs WHERE id = $1::VARCHAR",
                            request.job_id,
                        )

                    if recorded:
                        started_at = recorded["started_at"] or started_at
                        finished_at = recorded["finished_at"] or finished_at

                    logger.info(f"Updated job {request.job_id} status to {job_status.value} from Nvidia status: {unsloth_status}")

                    # Calculate progress
                    progress = job_data.get("progress_percent", 0.0) / 100.0
                    if job_status == JobStatus.SUCCEEDED:
                        progress = 1.0

                    # Map Unsloth response to OpenAI format
                    return JobStatusResponse(
                        status=job_status,
                        progress=progress,
                        trained_tokens=None,  # Unsloth doesn't provide this
                        progress_percent=job_data.get("progress_percent", 0.0) if job_status != JobStatus.SUCCEEDED else 100.0,
                        current_step=job_data.get("current_step"),
                        total_steps=job_data.get("total_steps"),
                        current_phase=job_data.get("current_phase"),
                        training_loss=job_data.get("training_loss"),
                        elapsed_seconds=job_data.get("elapsed_seconds"),
                        error_message=job_data.get('error_log') if job_status == JobStatus.FAILED else None,
                        fine_tuned_model=job_data.get('output_file_id'),  # File ID of the trained model
                        result_files=[job_data.get('output_file_id')] if job_data.get('output_file_id') else [],
                        started_at=started_at,
                        finished_at=finished_at,
                        estimated_finish=None,
                        checkpoints=[],
                        logs=[]
                    )
                elif response.status_code == 404:
                    logger.warning(f"Job {request.job_id} not found on backend (404)")
                    return JobStatusResponse(
                        status=JobStatus.FAILED,
                        error_message="Job not found on backend server",
                        progress=0.0
                    )
                else:
                    logger.error(f"Backend API error for job status: {response.status_code} - {response.text}")
                    return JobStatusResponse(
                        status=JobStatus.FAILED,
                        error_message=f"Backend API error: status code {response.status_code}",
                        progress=0.0
                    )

        except Exception as e:
            logger.error(f"Error getting Nvidia job status {request.job_id}: {e}", exc_info=True)
            return JobStatusResponse(
                status=JobStatus.FAILED,
                error_message="Failed to retrieve job status from backend",
                progress=0.0
            )

    async def cancel_job(self, job_id: str, resource_job_id: str, auth_token: str = None) -> bool:
        """Cancel job on Nvidia/Unsloth API"""
        try:
            auth_header = await self.backend_auth.get_auth_header()

            async with httpx.AsyncClient(timeout=STATUS_CHECK_TIMEOUT, verify=self.verify_ssl) as client:
                response = await client.delete(
                    f"{self.nvidia_api_url}/finetune/job/{resource_job_id}",
                    headers=auth_header
                )

                if response.status_code == 200:
                    logger.info(f"Successfully cancelled Nvidia job {job_id}")
                    return True
                else:
                    logger.error(f"Failed to cancel Nvidia job {job_id}: {response.status_code} - {response.text}")
                    return False
        except Exception as e:
            logger.error(f"Error cancelling Nvidia job {job_id}: {e}")
            return False

    async def get_job_logs(self, job_id: str, resource_job_id: str, auth_token: str = None) -> List[str]:
        """Get job logs from Nvidia/Unsloth API (synthesized from status endpoint)"""
        try:
            auth_header = await self.backend_auth.get_auth_header()

            async with httpx.AsyncClient(timeout=STATUS_CHECK_TIMEOUT, verify=self.verify_ssl) as client:
                response = await client.get(
                    f"{self.nvidia_api_url}/finetune/job/{resource_job_id}",
                    headers=auth_header
                )

                if response.status_code == 200:
                    job_data = response.json()
                    logs = []

                    if job_data.get('error_log'):
                        logs.append(f"[ERROR] {job_data['error_log']}")

                    status = job_data.get('status', 'unknown')
                    progress = job_data.get('progress_percent', 0)
                    logs.append(f"[INFO] Status: {status}, Progress: {progress}%")

                    if 'current_step' in job_data:
                        logs.append(f"[INFO] Step {job_data['current_step']}/{job_data.get('total_steps', '?')}")

                    if 'training_loss' in job_data:
                        logs.append(f"[INFO] Training Loss: {job_data['training_loss']}")

                    return logs
                else:
                    logger.error(f"Failed to get Nvidia job info: {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Error getting Nvidia job logs {job_id}: {e}")
            return []

    async def cleanup_job(self, job_id: str, resource_job_id: str) -> bool:
        """Clean up job resources (handled by Unsloth API)"""
        # No explicit cleanup needed - Unsloth API manages its own resources
        return True

    async def list_jobs(self, username: str = None, status: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List all jobs from Nvidia/Unsloth API"""
        try:
            auth_header = await self.backend_auth.get_auth_header()

            params = {"limit": limit}
            if username:
                params["username"] = username
            if status:
                params["status"] = status

            async with httpx.AsyncClient(timeout=STATUS_CHECK_TIMEOUT, verify=self.verify_ssl) as client:
                response = await client.get(
                    f"{self.nvidia_api_url}/finetune/jobs",
                    headers=auth_header,
                    params=params
                )

                if response.status_code == 200:
                    jobs = response.json()
                    # Transform each job to OpenAI format
                    return [self._transform_job_to_openai(job) for job in jobs]
                else:
                    logger.error(f"Failed to list Nvidia jobs: {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Error listing Nvidia jobs: {e}")
            return []

    async def get_job_events(self, job_id: str, resource_job_id: str, limit: int = 20) -> Dict[str, Any]:
        """Get job events from Nvidia/Unsloth API (synthesized from status)"""
        try:
            auth_header = await self.backend_auth.get_auth_header()

            async with httpx.AsyncClient(timeout=STATUS_CHECK_TIMEOUT, verify=self.verify_ssl) as client:
                response = await client.get(
                    f"{self.nvidia_api_url}/finetune/job/{resource_job_id}",
                    headers=auth_header
                )

                if response.status_code == 200:
                    job_data = response.json()

                    current_status = job_data.get('status', 'unknown')
                    mapped_status = self._map_unsloth_status_to_openai(current_status)

                    async with self.db_pool.acquire() as conn:
                        error_json = None
                        if mapped_status == JobStatus.FAILED and job_data.get('error_log'):
                            import json
                            error_json = json.dumps({"message": job_data.get('error_log')})

                        # Ensure output_file_id is a string or None, not a timestamp
                        output_file_id = job_data.get('output_file_id')
                        if output_file_id is not None and not isinstance(output_file_id, str):
                            output_file_id = str(output_file_id)

                        current_timestamp = _now_ts()
                        # See get_job_status: a start or an end is recorded once
                        # and never moved by a later poll.
                        engine_offset = _engine_clock_offset(job_data, job_id)
                        created_ts = _engine_timestamp(job_data.get('created_at'), 'created_at', job_id)
                        started_ts = _engine_timestamp(
                            job_data.get('started_at'), 'started_at', job_id, engine_offset
                        )
                        completed_ts = _engine_timestamp(
                            job_data.get('completed_at'), 'completed_at', job_id, engine_offset
                        )

                        await conn.execute("""
                            UPDATE fine_tuning_jobs
                            SET status = $1::VARCHAR,
                                updated_at = $5::INTEGER,
                                started_at = COALESCE(started_at, $12::INTEGER),
                                finished_at = CASE
                                    WHEN $1::VARCHAR IN ('succeeded', 'failed', 'cancelled')
                                        THEN COALESCE(finished_at, $13::INTEGER, $5::INTEGER)
                                    ELSE finished_at
                                END,
                                fine_tuned_model = COALESCE($2::VARCHAR, fine_tuned_model),
                                error_message = COALESCE($3::TEXT, error_message),
                                progress_percent = COALESCE($6::REAL, progress_percent),
                                current_step = COALESCE($7::INTEGER, current_step),
                                total_steps = COALESCE($8::INTEGER, total_steps),
                                current_phase = COALESCE($9::VARCHAR, current_phase),
                                training_loss = COALESCE($11::REAL, training_loss),
                                elapsed_seconds = COALESCE($10::INTEGER, elapsed_seconds)
                            WHERE id = $4::VARCHAR
                        """, mapped_status.value, output_file_id, error_json, job_id, current_timestamp,
                            job_data.get("progress_percent"),
                            job_data.get("current_step"),
                            job_data.get("total_steps"),
                            job_data.get("current_phase"),
                            job_data.get("elapsed_seconds"),
                            job_data.get("training_loss"),
                            started_ts,
                            completed_ts)

                        # Whatever ended up on record wins: it is the first (and
                        # therefore stable) reading, and it covers the case where
                        # the engine's own timestamp had to be discarded.
                        recorded = await conn.fetchrow(
                            "SELECT started_at, finished_at FROM fine_tuning_jobs WHERE id = $1::VARCHAR",
                            job_id,
                        )

                    if recorded:
                        started_ts = recorded["started_at"] or started_ts
                        completed_ts = recorded["finished_at"] or completed_ts

                    logger.info(f"Updated job {job_id} status to {mapped_status.value}")

                    # The engine has no event log of its own, so the lifecycle is
                    # reconstructed from the timestamps it does report. Every
                    # event that describes a past moment is stamped with that
                    # moment (not "now"), so the list stays stable across polls.
                    status = current_status.lower()
                    progress_percent = job_data.get('progress_percent')
                    current_step = job_data.get('current_step')
                    total_steps = job_data.get('total_steps')
                    training_loss = job_data.get('training_loss')
                    elapsed_seconds = job_data.get('elapsed_seconds')
                    events = []

                    def add_event(suffix: str, created_at: Optional[int], level: str,
                                  message: str, event_type: str = "message",
                                  data: Optional[Dict[str, Any]] = None) -> None:
                        events.append({
                            "object": "fine_tuning.job.event",
                            "id": f"ftevent-{job_id}-{suffix}",
                            "created_at": created_at if created_at else current_timestamp,
                            "level": level,
                            "message": message,
                            "data": {k: v for k, v in (data or {}).items() if v is not None},
                            "type": event_type,
                        })

                    if created_ts:
                        add_event(
                            "created", created_ts, "info",
                            "Fine-tuning job created and submitted to the training engine",
                            data={"model": job_data.get("model_name")},
                        )

                    if status == 'queued':
                        add_event(
                            "queued", created_ts, "info",
                            "Queued — waiting for a training worker to pick the job up",
                        )
                    elif started_ts:
                        # Only meaningful once the job left the queue: how long it
                        # waited is otherwise invisible in the timeline.
                        wait = started_ts - created_ts if created_ts else None
                        message = "Training started"
                        if wait and wait > 0:
                            message = f"Training started after waiting {_format_seconds(wait)} in the queue"
                        add_event("started", started_ts, "info", message,
                                  data={"queued_seconds": wait if wait and wait > 0 else None})

                    if status in ('running', 'training'):
                        if total_steps and current_step:
                            message = f"Training in progress — step {current_step} of {total_steps}"
                        elif progress_percent:
                            message = f"Training in progress — {round(progress_percent)}% complete"
                        else:
                            message = "Training in progress"
                        add_event(
                            "progress", current_timestamp, "info", message, event_type="metrics",
                            data={
                                "progress_percent": progress_percent,
                                "current_step": current_step,
                                "total_steps": total_steps,
                                "training_loss": training_loss,
                                "current_phase": job_data.get("current_phase"),
                                "elapsed_seconds": elapsed_seconds,
                            },
                        )

                    if status == 'completed':
                        message = "Training completed successfully"
                        if elapsed_seconds:
                            # The engine's own figure covers training only, not the
                            # model load and upload around it, so say whose clock
                            # this is rather than contradicting the timeline.
                            message = (
                                "Training completed successfully "
                                f"({_format_seconds(elapsed_seconds)} of engine time)"
                            )
                        add_event(
                            "completed", completed_ts, "info", message, event_type="metrics",
                            data={
                                "output_file_id": output_file_id,
                                "total_steps": total_steps,
                                "training_loss": training_loss,
                                "elapsed_seconds": elapsed_seconds,
                            },
                        )
                    elif status == 'failed':
                        add_event(
                            "error", completed_ts, "error",
                            job_data.get('error_log') or "Training failed",
                            data={
                                "current_step": current_step,
                                "total_steps": total_steps,
                                "elapsed_seconds": elapsed_seconds,
                            },
                        )
                    elif status == 'cancelled':
                        add_event(
                            "cancelled", completed_ts, "warning", "Job was cancelled",
                            data={"current_step": current_step, "total_steps": total_steps},
                        )

                    events.sort(key=lambda event: event["created_at"])
                    return {
                        "object": "list",
                        "data": events[:limit],  # Limit number of events
                        "has_more": len(events) > limit
                    }
                else:
                    logger.error(f"Failed to get Nvidia job for events: {response.status_code}")
                    return {"object": "list", "data": [], "has_more": False}
        except Exception as e:
            logger.error(f"Error getting Nvidia job events: {e}")
            return {"object": "list", "data": [], "has_more": False}

    def _transform_job_to_openai(self, nvidia_job: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Nvidia job format to OpenAI format"""
        # Map status to OpenAI JobStatus enum and convert to string value
        mapped_status = self._map_unsloth_status_to_openai(nvidia_job.get('status', 'unknown'))
        job_id = str(nvidia_job.get('id', ''))
        # Same start/finish clock offset as the status path, corrected the same way.
        engine_offset = _engine_clock_offset(nvidia_job, job_id)
        created_at = _engine_timestamp(nvidia_job.get('created_at'), 'created_at', job_id)
        started_at = _engine_timestamp(
            nvidia_job.get('started_at'), 'started_at', job_id, engine_offset
        )
        finished_at = _engine_timestamp(
            nvidia_job.get('completed_at'), 'completed_at', job_id, engine_offset
        )

        return {
            "id": job_id,
            "object": "fine_tuning.job",
            "created_at": created_at or 0,
            "started_at": started_at,
            "finished_at": finished_at,
            "model": nvidia_job.get('model_name', 'unknown'),
            "fine_tuned_model": nvidia_job.get('output_file_id', None),
            "organization_id": None,
            "result_files": [nvidia_job.get('output_file_id')] if nvidia_job.get('output_file_id') else [],
            "status": mapped_status.value,  # Convert JobStatus enum to string value
            "validation_file": None,
            "training_file": None,  # Not provided in Nvidia response
            "hyperparameters": {},
            "trained_tokens": None,
            "error": {"message": nvidia_job.get('error_log', '')} if nvidia_job.get('error_log') else None,
            "seed": None,
            "estimated_finish": None
        }

    async def get_job_history(self, username: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get training job history for a user"""
        try:
            auth_header = await self.backend_auth.get_auth_header()

            async with httpx.AsyncClient(timeout=STATUS_CHECK_TIMEOUT, verify=self.verify_ssl) as client:
                response = await client.get(
                    f"{self.nvidia_api_url}/finetune/history/{username}",
                    headers=auth_header,
                    params={"limit": limit}
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get history: {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []

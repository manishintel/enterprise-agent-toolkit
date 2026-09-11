# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import base64
import requests
import httpx
import os
import hashlib
import logging
import re
import socket
import time
import gzip
import shutil
import tarfile
from typing import Dict, Optional, Tuple, Union

try:
    # Multi-threaded gzip: roughly Nx faster than the stdlib on a merged model,
    # which is the difference between a two-minute and a ten-minute archive step.
    import pgzip
except ImportError:  # not in the pinned training image
    pgzip = None

from urllib.parse import urlparse
from app.config import settings

logger = logging.getLogger("uvicorn")

class FileClientError(Exception):
    """Custom exception for file client errors"""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        # Carried so the retry decorator can tell a transient failure from a
        # verdict that will be identical on the next attempt (404, 413, ...).
        self.status_code = status_code


def _status_of(exc: BaseException) -> Optional[int]:
    """HTTP status behind an exception, if it carries one."""
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


# 408 and 429 are the client-error codes that *are* worth another attempt.
_RETRYABLE_CLIENT_ERRORS = {408, 429}


def retry_on_failure(max_retries: int = 3, delay: int = 2):
    """
    Decorator for retrying failed operations.

    Client errors other than 408/429 are re-raised immediately: a 404 from the
    Files API is an identity or file-ID verdict, and a 413 is a size limit -
    repeating the request only delays the job's failure and, for an upload,
    re-sends gigabytes to be rejected again.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    status = _status_of(e)
                    if (status is not None and 400 <= status < 500
                            and status not in _RETRYABLE_CLIENT_ERRORS):
                        logger.error(
                            f"{func.__name__} failed with HTTP {status} - not retrying"
                        )
                        raise
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")
            raise last_exception
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Identity and transport
#
# The engine never sees the dataset itself at submit time - only a file ID and
# the caller's ft-api-key. Everything below is about turning that pair into a
# request the Files API accepts as "this user, asking for their own file".
# ---------------------------------------------------------------------------

# A username, not a secret: conservative enough that a random opaque API key
# will not be mistaken for one.
_USERNAME_RE = re.compile(r"^[A-Za-z0-9._@+-]{1,128}$")

_TLS_WARNED = False


def _decode_username(bearer_token: str) -> str:
    """
    Recover the username from the ft-api-key, or '' if it is not one.

    The documented format is base64(username). Anything that is not valid
    base64 of a plain username - a JWT, an opaque key - yields '' and the
    Authorization header is then the only identity we send.
    """
    token = (bearer_token or "").strip()
    if not token or token.count(".") >= 2:  # JWT-shaped: leave it alone
        return ""
    try:
        padded = token + "=" * (-len(token) % 4)
        decoded = base64.b64decode(padded, validate=True).decode("utf-8").strip()
    except Exception:
        return ""
    return decoded if _USERNAME_RE.match(decoded) else ""


def _auth_headers(bearer_token: str) -> Dict[str, str]:
    """Identity headers for every outbound Files API call."""
    headers = {"Authorization": f"Bearer {bearer_token}"}

    if settings.FILES_API_FORWARD_USER:
        username = _decode_username(bearer_token)
        if username:
            headers["X-Forwarded-User"] = username
            headers["X-User"] = username
    return headers


def _identity_hint(bearer_token: str) -> str:
    """Human-readable description of the identity we presented."""
    username = _decode_username(bearer_token)
    return f"user '{username}'" if username else "the supplied ft-api-key"


def _tls_verify() -> Union[bool, str]:
    """requests/httpx 'verify' value from FILES_API_TLS_VERIFY."""
    global _TLS_WARNED

    raw = (settings.FILES_API_TLS_VERIFY or "").strip()
    if not raw:
        return True
    lowered = raw.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        if not _TLS_WARNED:
            logger.warning(
                "FILES_API_TLS_VERIFY is disabled - Files API traffic is not "
                "certificate-verified. Do not use this in production."
            )
            _TLS_WARNED = True
        return False
    return raw  # path to a CA bundle


def _timeout() -> Tuple[int, int]:
    """(connect, read) for a bulk transfer."""
    return (settings.FILES_API_CONNECT_TIMEOUT, settings.FILES_API_TIMEOUT)


def _short_timeout() -> Tuple[int, int]:
    """(connect, read) for a metadata call."""
    return (settings.FILES_API_CONNECT_TIMEOUT, settings.FILES_API_METADATA_TIMEOUT)


_RESOLVE_INSTALLED = False


def _resolve_overrides() -> Dict[str, str]:
    """FILES_API_RESOLVE parsed into {hostname: ip}."""
    mapping: Dict[str, str] = {}
    for entry in (settings.FILES_API_RESOLVE or "").split(","):
        host, _, ip = entry.strip().partition(":")
        host, ip = host.strip().lower(), ip.strip()
        if host and ip:
            mapping[host] = ip
    return mapping


def install_resolve_overrides() -> Dict[str, str]:
    """Resolve the hostnames in FILES_API_RESOLVE locally, like /etc/hosts.

    A Files API published through an ingress is reachable only under the
    hostname that ingress routes on, and that hostname may have no DNS record -
    while editing /etc/hosts needs root this service does not have. Substituting
    the IP into FILES_API_URL is not equivalent: the request would then miss the
    ingress' host-based route, and TLS would be checked against an address the
    certificate does not cover. Redirecting only the lookup keeps the hostname
    in the URL, so host routing, SNI and certificate verification all still work.

    Only exact hostname matches are redirected; every other lookup in the
    process is untouched. Idempotent, and a no-op when the setting is empty.
    """
    global _RESOLVE_INSTALLED

    mapping = _resolve_overrides()
    if not mapping or _RESOLVE_INSTALLED:
        return mapping

    real_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(host, port, *args, **kwargs):
        target = mapping.get(str(host).lower(), host)
        return real_getaddrinfo(target, port, *args, **kwargs)

    socket.getaddrinfo = getaddrinfo
    _RESOLVE_INSTALLED = True
    logger.info(
        "Files API name overrides active: "
        + ", ".join(f"{host} -> {ip}" for host, ip in sorted(mapping.items()))
    )
    return mapping


# At import, so every entry point gets it: the API process, the e2e scripts and
# `python -m` one-offs all reach the Files API through this module.
install_resolve_overrides()


def _trust_env_for(url: str) -> bool:
    """Whether the environment's proxy settings may be applied to this URL.

    Two ways a site proxy breaks Files API traffic that must stay on the
    internal network:

    * requests and httpx disagree about no_proxy - requests resolves CIDR
      entries like '127.0.0.0/8', httpx matches host suffixes only. With a
      proxy exported, that asymmetry let a dataset download go direct while the
      model upload was handed to the proxy, which answers 403: training
      succeeded and only the hand-back failed.
    * a name pinned through FILES_API_RESOLVE has no DNS record, so a proxy
      cannot resolve it, and an intercepting proxy presents its own certificate
      instead of the ingress' - verification then fails against the configured
      CA. Pinning an address means we intend to connect to it directly.

    One verdict, used by both clients, so both directions route identically.
    """
    host = (urlparse(url).hostname or "").lower()
    if host in _resolve_overrides():
        return False
    try:
        if requests.utils.should_bypass_proxies(url, no_proxy=None):
            return False
    except Exception:  # unparseable URL: leave the client's own handling in place
        pass
    return True


def _session(url: str) -> requests.Session:
    """requests Session carrying the same proxy verdict as the upload client.

    trust_env also gates env CA bundles and netrc, which is fine here: 'verify'
    is always passed explicitly from FILES_API_TLS_VERIFY.
    """
    session = requests.Session()
    session.trust_env = _trust_env_for(url)
    return session

def get_file_id_by_filename(filename: str, bearer_token: str) -> str:
    """
    Get file ID by searching for filename in FILES API

    Args:
        filename: The filename to search for (e.g., 'dataset.jsonl')
        bearer_token: Bearer token for FILES API authentication

    Returns:
        File ID if found

    Raises:
        FileClientError: If file not found or API error
    """
    try:
        headers = _auth_headers(bearer_token)

        # List all files
        url = f"{settings.FILES_API_URL}/v1/files"
        logger.info(f"Searching for file: {filename}")

        with _session(url) as session:
            response = session.get(url, headers=headers, timeout=_short_timeout(),
                                   verify=_tls_verify())
        response.raise_for_status()

        result = response.json()
        files = result.get("data", [])

        # Search for matching filename
        for file_info in files:
            if file_info.get("filename") == filename:
                file_id = file_info.get("id")
                logger.info(f"Found file: {filename} -> {file_id}")
                return file_id

        # File not found
        raise FileClientError(
            f"File '{filename}' not found in FILES API. "
            f"Available files: {[f.get('filename') for f in files[:10]]}"
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to search for file {filename}: {e}")
        raise FileClientError(f"Failed to search for file: {str(e)}",
                              status_code=_status_of(e)) from e

def get_file_metadata(file_id: str, bearer_token: str) -> dict:
    """
    Fetch a file's metadata record from the FILES API.

    Returns:
        The metadata dict (contains 'filename', 'bytes', 'purpose', ...).

    Raises:
        FileClientError: If the lookup fails.
    """
    headers = _auth_headers(bearer_token)
    url = f"{settings.FILES_API_URL}/v1/files/{file_id}"
    try:
        with _session(url) as session:
            response = session.get(url, headers=headers, timeout=_short_timeout(),
                                   verify=_tls_verify())
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        raise FileClientError(f"Failed to fetch metadata for {file_id}: {str(e)}",
                              status_code=_status_of(e)) from e


def _local_dataset_path(file_id: str, bearer_token: str) -> str:
    """
    Build the on-disk path for a downloaded dataset.

    The FILES API addresses content by opaque ID ('file-abc123'), which carries
    no extension. The ML01 training-data validator only accepts '.jsonl', so the
    real filename is resolved from the file's metadata and its suffix preserved
    locally — that keeps the extension check meaningful (a '.csv' upload is
    still rejected) instead of it failing on every single job.

    The identity is part of the filename. File IDs are only unique *within* a
    user in the Files API (objects are stored at '{user_id}/{file_id}'), so a
    cache keyed on the ID alone could serve one tenant's dataset to another, or
    keep serving a stale copy after the owner replaced the file.
    """
    suffix = ""
    try:
        metadata = get_file_metadata(file_id, bearer_token)
        remote_name = os.path.basename(str(metadata.get("filename") or ""))
        suffix = os.path.splitext(remote_name)[1]
        if suffix:
            logger.info(f"Resolved remote filename for {file_id}: '{remote_name}'")
    except FileClientError as e:
        logger.warning(f"Could not resolve remote filename for {file_id}: {e}")

    if not suffix:
        suffix = ".jsonl"
        logger.info(f"No extension available for {file_id}; assuming '{suffix}'")

    # Short digest, not the token: this path ends up in logs and job records.
    owner = hashlib.sha256((bearer_token or "").encode()).hexdigest()[:12]
    return os.path.join(settings.TEMP_DATA_DIR, f"{file_id}.{owner}{suffix}")


@retry_on_failure(max_retries=3, delay=2)
def download_dataset(filename: str, bearer_token: str, force_download: bool = False) -> str:
    """
    Download dataset file from FILES API using file_id

    Args:
        filename: File ID to download (e.g., 'file-abc123' from FILES API)
        bearer_token: Bearer token for FILES API authentication
        force_download: If True, re-download even if file exists locally

    Returns:
        Local path to downloaded file

    Raises:
        FileClientError: If download fails
    """
    # Validate that filename looks like a file ID or might be a legacy filename
    if not filename.startswith('file-') and '.' in filename:
        logger.warning(
            f"⚠️  '{filename}' looks like a filename, not a file ID. "
            f"Attempting to resolve to file ID..."
        )
        try:
            filename = get_file_id_by_filename(filename, bearer_token)
            logger.info(f"✓ Resolved to file ID: {filename}")
        except FileClientError as e:
            logger.error(f"Failed to resolve filename: {e}")
            raise FileClientError(
                f"'{filename}' is not a valid file ID and could not be resolved. "
                f"Please use file IDs from FILES API (e.g., 'file-abc123xyz'). "
                f"Upload your file first or list files to get the correct ID."
            ) from e

    local_path = _local_dataset_path(filename, bearer_token)

    # Check if file already exists and is valid
    if os.path.exists(local_path) and not force_download:
        file_size = os.path.getsize(local_path)
        if file_size > 0:
            logger.info(f"Using cached dataset: {filename} ({file_size} bytes)")
            return local_path
        else:
            logger.warning(f"Cached file {filename} is empty, re-downloading...")
            os.remove(local_path)

    headers = _auth_headers(bearer_token)

    try:
        # FILES API v1: Download file content using /content endpoint
        file_id = filename  # The input is the file_id
        url = f"{settings.FILES_API_URL}/v1/files/{file_id}/content"

        logger.info(f"Downloading file content from: {url} as {_identity_hint(bearer_token)}")
        start_time = time.time()

        with _session(url) as session, \
                session.get(url, headers=headers, stream=True, timeout=_timeout(),
                            verify=_tls_verify()) as r:
            r.raise_for_status()

            # Get file size if available
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0

            # Create temp file first
            temp_path = local_path + ".tmp"

            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Log progress every 10MB
                        if downloaded % (10 * 1024 * 1024) == 0:
                            if total_size:
                                progress = (downloaded / total_size) * 100
                                logger.info(f"Download progress: {progress:.1f}% ({downloaded}/{total_size} bytes)")
                            else:
                                logger.info(f"Downloaded: {downloaded} bytes")

            # Verify file was downloaded
            if downloaded == 0:
                raise FileClientError(f"Downloaded file is empty: {filename}")

            # A stream cut short by the gateway would otherwise be trained on:
            # the tail of a truncated JSONL is a partial line the validator may
            # well accept. Content-Length is served by the Files API, so hold
            # the transfer to it. No status code => this attempt is retried.
            if total_size and downloaded != total_size:
                raise FileClientError(
                    f"Truncated download for {filename}: got {downloaded} of "
                    f"{total_size} bytes"
                )

            # Rename temp file to final name
            os.rename(temp_path, local_path)

        elapsed_time = time.time() - start_time
        logger.info(
            f"Download completed: {filename} ({downloaded} bytes) in {elapsed_time:.2f}s "
            f"({downloaded / elapsed_time / 1024 / 1024:.2f} MB/s)"
        )

        return local_path

    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed for {filename}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            if status_code == 404:
                logger.error(
                    f"❌ File not found: '{filename}' for {_identity_hint(bearer_token)}. "
                    f"The Files API scopes files to their owner and answers 404 - not 401 - "
                    f"when the identity it resolves does not own the file, so this is either "
                    f"a wrong file ID or an identity mismatch. "
                    f"List available files: GET /v1/files"
                )
            elif status_code == 401:
                logger.error(f"❌ Authentication failed. Check if the bearer token is valid.")
            try:
                error_detail = e.response.json()
                logger.error(f"API Error Response: {error_detail}")
            except:
                logger.error(f"API Response Text: {e.response.text}")
        # Clean up partial download
        for path in [local_path, local_path + ".tmp"]:
            if os.path.exists(path):
                os.remove(path)

        error_msg = f"Failed to download {filename}: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 404:
                error_msg += (
                    f"\n\nℹ️  Tip: '{filename}' was not found for {_identity_hint(bearer_token)}. "
                    f"Either the file ID is wrong, or the Files API resolved a different user "
                    f"than the owner (it returns 404, never 401, in that case) - check that "
                    f"ft-api-key carries base64(username) for the file's owner."
                )
            elif e.response.status_code == 401:
                error_msg += f"\n\nℹ️  Tip: Check if the bearer token is valid and not expired."
        raise FileClientError(error_msg, status_code=_status_of(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        for path in [local_path + ".tmp"]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        raise FileClientError(f"Download error: {str(e)}",
                              status_code=_status_of(e)) from e

def _open_gzip(output_path: str):
    """
    Open *output_path* as a gzip stream, in parallel when pgzip is installed.

    The stock training image does not ship pgzip, and adding it would mean either
    a derived image the GPU nodes cannot pull or a pip install at job start that
    lands in the wrong virtualenv. So this degrades instead: same archive format,
    single-threaded, slower on a large model. compresslevel=1 either way - the
    model is mostly incompressible float data, so a higher level costs minutes
    and saves very little.
    """
    if pgzip is not None:
        # thread=None uses every available core.
        return pgzip.open(output_path, "wb", thread=None, compresslevel=1)
    logger.info("pgzip is not installed; falling back to single-threaded gzip")
    return gzip.open(output_path, "wb", compresslevel=1)


def create_model_archive(folder_path: str, output_path: Optional[str] = None) -> str:
    """
    Create a tar.gz archive of the model folder.

    Args:
        folder_path: Path to model folder
        output_path: Optional custom output path for tar.gz file

    Returns:
        Path to created tar.gz file
    """
    if not os.path.exists(folder_path):
        raise FileClientError(f"Model folder not found: {folder_path}")

    if output_path is None:
        output_path = f"{folder_path}.tar.gz"

    try:
        logger.info(f"Creating model archive: {output_path}")
        start_time = time.time()

        with _open_gzip(output_path) as f_out:
            # Pipe tarfile into the gzip stream rather than tarring to disk
            # first: the model is already the largest thing on the volume.
            with tarfile.open(mode="w", fileobj=f_out) as tar:
                # arcname prevents storing full absolute path
                tar.add(folder_path, arcname=os.path.basename(folder_path))

        archive_size = os.path.getsize(output_path)
        elapsed_time = time.time() - start_time

        logger.info(
            f"Archive created: {output_path} ({archive_size / 1024 / 1024:.2f} MB) "
            f"in {elapsed_time:.2f}s ({archive_size / elapsed_time / 1024 / 1024:.2f} MB/s)"
        )

        return output_path

    except Exception as e:
        logger.error(f"Failed to create archive: {e}")
        raise FileClientError(f"Archive creation failed: {str(e)}") from e

@retry_on_failure(max_retries=2, delay=10)
def upload_model(folder_path: str, model_name: str, bearer_token: str) -> str:
    """
    Upload fine-tuned model to file service using FILES API v1 with httpx

    Args:
        folder_path: Path to model folder
        model_name: Name of the model for metadata
        bearer_token: Bearer token for FILES API authentication

    Returns:
        File ID from file service

    Raises:
        FileClientError: If upload fails
    """
    # Create tar.gz archive
    zip_path = create_model_archive(folder_path)

    headers = _auth_headers(bearer_token)

    # FILES API v1 endpoint
    url = f"{settings.FILES_API_URL}/v1/files"

    try:
        file_size = os.path.getsize(zip_path)
        logger.info(f"Uploading model: {model_name} from {zip_path}")
        logger.info(f"Upload size: {file_size / 1024 / 1024:.2f} MB")

        # Check if file is very large
        if file_size > 100 * 1024 * 1024:  # > 100MB
            logger.warning(f"Large file upload ({file_size / 1024 / 1024:.2f} MB) - this may take several minutes")

        start_time = time.time()

        # Read and upload file using httpx with streaming
        with open(zip_path, 'rb') as f:
            files = {
                'file': (f"finetuned_models#{os.path.basename(zip_path)}", f, 'application/gzip')
            }
            data = {
                'purpose': 'fine-tune-results'
            }

            # Use httpx client with extended timeout for large files
            # connect: 60s, read: 4 hours (14400s), write: 4 hours, pool: 60s
            # Increased timeout for very large files
            with httpx.Client(
                timeout=httpx.Timeout(
                    connect=float(settings.FILES_API_CONNECT_TIMEOUT),
                    read=float(settings.FILES_API_UPLOAD_TIMEOUT),
                    write=float(settings.FILES_API_UPLOAD_TIMEOUT),
                    pool=60.0,
                ),
                verify=_tls_verify(),
                trust_env=_trust_env_for(url),
            ) as client:
                logger.info(f"Uploading finetuned_models#{os.path.basename(zip_path)} to {url} "
                            f"as {_identity_hint(bearer_token)}...")
                logger.info(f"File size: {file_size / 1024 / 1024:.2f} MB - estimated time: {file_size / (1024 * 1024):.0f} seconds at 1 MB/s")

                response = client.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data
                )

                logger.info(f"Upload response status: {response.status_code}")
                logger.info(f"Upload response body: {response.text[:500]}")  # First 500 chars

                # Check for gateway errors (502, 503, 504) - these indicate server-side issues
                if response.status_code in [502, 503, 504]:
                    error_msg = f"Gateway error: HTTP {response.status_code}"
                    logger.error(f"{error_msg} - Server timeout or unavailable. Response: {response.text}")
                    raise FileClientError(
                        f"{error_msg} - The FILES_API server timed out processing the upload. "
                        f"This usually means the nginx gateway timeout is too short for files of this size ({file_size / 1024 / 1024:.2f} MB). "
                        f"Please increase the nginx gateway timeout on the FILES_API server or use chunked uploads."
                    )

                # Body too large: the ingress or the API rejected the archive
                # outright. Retrying re-sends every byte for the same verdict.
                if response.status_code == 413:
                    error_msg = (
                        f"Upload rejected: HTTP 413 - the archive "
                        f"({file_size / 1024 / 1024:.2f} MB) exceeds the upload size "
                        f"limit of the FILES API or the ingress in front of it "
                        f"(nginx client_max_body_size). Raise that limit for "
                        f"/v1/files, or the merged model cannot be returned."
                    )
                    logger.error(f"{error_msg} Response: {response.text[:300]}")
                    raise FileClientError(error_msg, status_code=413)

                # Check for auth errors
                if response.status_code == 401:
                    logger.error(f"Authentication failed - bearer token may be invalid or expired")
                    raise FileClientError(f"Authentication failed: Invalid or expired bearer token",
                                          status_code=401)

                # Check for other error status codes
                if response.status_code not in [200, 201]:
                    error_msg = f"Upload failed: HTTP {response.status_code}"
                    logger.error(f"{error_msg} - {response.text}")
                    raise FileClientError(f"{error_msg} - {response.text}",
                                          status_code=response.status_code)

                result = response.json()
                logger.info(f"Parsed response: {result}")

        elapsed_time = time.time() - start_time

        # Extract file ID from FILES API response
        file_id = result.get("id")

        if not file_id:
            logger.error(f"No file ID in response! Full response: {result}")
            raise FileClientError("No file ID returned from FILES API")

        filename = result.get("filename", "unknown")
        file_status = result.get("status", "unknown")

        logger.info(
            f"✅ Upload completed: {model_name} (ID: {file_id}, Status: {file_status}) "
            f"in {elapsed_time:.2f}s ({file_size / elapsed_time / 1024 / 1024:.2f} MB/s)"
        )

        # Clean up tar.gz file after successful upload
        try:
            os.remove(zip_path)
            logger.info(f"Cleaned up archive: {zip_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up archive {zip_path}: {e}")

        return file_id

    except httpx.TimeoutException as e:
        logger.error(f"Upload timeout for {model_name} after {time.time() - start_time:.0f}s: {e}")
        logger.error(f"File size was: {file_size / 1024 / 1024:.2f} MB")
        raise FileClientError(
            f"Upload timed out for {model_name}. File size: {file_size / 1024 / 1024:.2f} MB. "
            f"Elapsed time: {time.time() - start_time:.0f}s. "
            f"The client timeout is 4 hours, but the server may have a shorter timeout. "
            f"Consider splitting large models or increasing server-side timeouts."
        ) from e
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during upload of {model_name}: {e}")
        logger.error(f"Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
        raise FileClientError(f"HTTP error during upload: {str(e)}") from e
    except FileClientError:
        # Re-raise FileClientError as-is (already formatted)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        raise FileClientError(f"Upload error: {str(e)}") from e
    finally:
        # Clean up zip file on error
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
                logger.info(f"Cleaned up archive after upload: {zip_path}")
            except Exception:
                pass

def cleanup_old_files(directory: str, max_age_hours: int = 24, protected: Optional[set] = None):
    """
    Clean up old files in a directory

    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files to keep in hours
        protected: Entry names that must never be removed regardless of age.
            With concurrent jobs, one job finishing must not delete the working
            directories of jobs that are still training, so callers pass the
            names still in flight.
    """
    if not os.path.exists(directory):
        return

    protected = protected or set()

    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        cleaned_size = 0

        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)

            if item in protected:
                logger.debug(f"Skipping in-use item during cleanup: {item}")
                continue

            # Get file age
            file_age = current_time - os.path.getmtime(item_path)

            if file_age > max_age_seconds:
                item_size = os.path.getsize(item_path) if os.path.isfile(item_path) else 0

                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

                cleaned_count += 1
                cleaned_size += item_size
                logger.info(f"Cleaned up old file/folder: {item} (age: {file_age / 3600:.1f}h)")

        if cleaned_count > 0:
            logger.info(
                f"Cleanup completed: {cleaned_count} items removed, "
                f"{cleaned_size / 1024 / 1024:.2f} MB freed"
            )

    except Exception as e:
        logger.error(f"Cleanup failed for {directory}: {e}")

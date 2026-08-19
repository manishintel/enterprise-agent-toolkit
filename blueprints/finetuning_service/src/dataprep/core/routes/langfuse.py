"""
Langfuse import routes.

  * GET  /v1/langfuse/projects    — selectable projects (pick one before filtering)
  * GET  /v1/langfuse/annotations — selectable score names and annotation queues
  * POST /v1/langfuse/preview     — first ``limit`` converted records (no upload)
  * POST /v1/langfuse/import      — full import: streams to MinIO, registers a File

Every read is scoped to one Langfuse project. ``project_id`` selects it; leaving
it out falls back to the default project, which is what a single-project
deployment (and any client written before project selection existed) gets.

All endpoints reuse ``get_current_user_id`` so imported files inherit the
requesting user's identity (same behaviour as regular uploads).
"""

import io
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.orm import Session

from core.config import settings
from core.config.database import get_db
from core.handlers import (
    FileHandler,
    GatewayClient,
    GatewayConfigError,
    GatewayFetchError,
    MetadataHandler,
)
from core.handlers.auth_handler import get_current_user_id
from core.handlers.gateway_handler import models_match
from core.handlers.langfuse_handler import (
    FORMAT_CUSTOM,
    FORMAT_OPENAI_CHAT,
    FORMAT_RAW,
    QUEUE_STATUSES,
    SCORE_OPERATORS,
    SCORE_SOURCES,
    LangfuseClient,
    LangfuseConfigError,
    LangfuseFetchError,
    LangfuseProjectError,
    convert_trace,
    project_registry,
    stream_jsonl,
)
from core.middleware import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/langfuse", tags=["langfuse"])

MAX_PREVIEW = 20
_ALLOWED_FMTS = {FORMAT_OPENAI_CHAT, FORMAT_RAW, FORMAT_CUSTOM}


def _one_of(value: Optional[str], allowed: tuple, field: str) -> Optional[str]:
    """Normalise an optional enum-ish string field, or reject it."""
    if value is None or not value.strip():
        return None
    upper = value.strip().upper()
    if upper not in allowed:
        raise ValueError(f"{field} must be one of {', '.join(allowed)}")
    return upper


class LangfuseImportRequest(BaseModel):
    project_id: Optional[str] = Field(
        None,
        description=(
            "Langfuse project to read from. Use GET /v1/langfuse/projects for the "
            "selectable values; omit to use the default project."
        ),
        max_length=255,
    )
    from_timestamp: Optional[str] = Field(None, description="ISO 8601 lower bound")
    to_timestamp: Optional[str] = Field(None, description="ISO 8601 upper bound")
    name: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    environment: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    order_by: Optional[str] = Field(None, examples=["timestamp.desc"])
    model: Optional[str] = Field(
        None,
        description=(
            "Keep only traces whose generations ran against this model. "
            "Use GET /v1/langfuse/models for the selectable values."
        ),
        max_length=255,
    )

    # --- annotations (human review) ------------------------------------------
    score_name: Optional[str] = Field(
        None,
        description=(
            "Keep only traces carrying a score with this name. Use "
            "GET /v1/langfuse/annotations for the selectable values."
        ),
        max_length=255,
    )
    score_source: Optional[str] = Field(
        None,
        description=(
            "Restrict to scores of this origin: ANNOTATION (entered by a human), "
            "API or EVAL. Omit to accept any."
        ),
    )
    score_value: Optional[float] = Field(
        None, description="Numeric score threshold; requires score_operator."
    )
    score_operator: Optional[str] = Field(
        None, description="One of = != > >= < <=; requires score_value."
    )
    score_string_value: Optional[str] = Field(
        None,
        description="Category/label to match for categorical or boolean scores.",
        max_length=500,
    )
    annotation_queue_id: Optional[str] = Field(
        None,
        description="Keep only traces sitting in this Langfuse annotation queue.",
        max_length=255,
    )
    annotation_queue_status: Optional[str] = Field(
        None,
        description="Restrict queue membership to PENDING or COMPLETED items.",
    )

    format: str = Field(FORMAT_OPENAI_CHAT, description="openai_chat | raw | custom")
    fields: List[str] = Field(default_factory=list, description="Only for format=custom")

    filename: Optional[str] = Field(
        None,
        description="Optional filename to store in MinIO. Auto-generated if omitted.",
        max_length=255,
    )
    max_traces: Optional[int] = Field(
        None,
        ge=1,
        le=100_000,
        description="Hard cap on records fetched. Defaults to server setting.",
    )
    preview_limit: int = Field(5, ge=1, le=MAX_PREVIEW, description="Records to return from /preview")

    @field_validator("score_source")
    @classmethod
    def _valid_score_source(cls, value: Optional[str]) -> Optional[str]:
        return _one_of(value, SCORE_SOURCES, "score_source")

    @field_validator("annotation_queue_status")
    @classmethod
    def _valid_queue_status(cls, value: Optional[str]) -> Optional[str]:
        return _one_of(value, QUEUE_STATUSES, "annotation_queue_status")

    @field_validator("score_operator")
    @classmethod
    def _valid_score_operator(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not value.strip():
            return None
        operator = value.strip()
        if operator not in SCORE_OPERATORS:
            raise ValueError(
                f"score_operator must be one of {' '.join(SCORE_OPERATORS)}"
            )
        return operator

    @model_validator(mode="after")
    def _default_score_operator(self) -> "LangfuseImportRequest":
        # A threshold without a comparison is the common slip ("quality 4" ->
        # "quality of at least 4"), so read it the way it was meant.
        if self.score_value is not None and not self.score_operator:
            self.score_operator = ">="
        return self

    def has_annotation_filter(self) -> bool:
        return bool(
            self.score_name
            or self.score_source
            or self.score_string_value
            or self.score_value is not None
        )


def _client(project_id: Optional[str] = None) -> LangfuseClient:
    """A client scoped to the requested project, or the default project."""
    try:
        return project_registry.client(project_id)
    except LangfuseConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except LangfuseProjectError as e:
        # The caller named a project we hold no credentials for — their input,
        # not a server fault.
        raise HTTPException(status_code=400, detail=str(e))
    except LangfuseFetchError as e:
        raise HTTPException(status_code=502, detail=str(e))


def _filename_safe(value: str, max_length: int = 40) -> str:
    """Reduce a project id to something safe to embed in an object name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return cleaned[:max_length] or "project"


def _validated_format(fmt: str, fields: List[str]) -> None:
    if fmt not in _ALLOWED_FMTS:
        raise HTTPException(status_code=400, detail=f"Invalid format: {fmt}")
    if fmt == FORMAT_CUSTOM and not fields:
        raise HTTPException(
            status_code=400,
            detail="format=custom requires at least one entry in `fields`.",
        )


def _iter_converted(client: LangfuseClient, body: LangfuseImportRequest):
    """
    Yield ``(record, was_skipped)`` tuples.

    For ``openai_chat`` we transparently re-fetch traces with observations
    when the list endpoint's trace.output is empty, so nested GENERATION
    outputs (e.g. LiteLLM) still convert successfully.

    Filters Langfuse can't express on ``/traces`` — model, annotation score,
    annotation queue — take a different route entirely; see
    ``_iter_converted_by_ids``.
    """
    restricted = _restricted_trace_ids(client, body)
    if restricted is not None:
        yield from _iter_converted_by_ids(client, body, restricted)
        return

    for trace in client.iter_traces(
        from_timestamp=body.from_timestamp,
        to_timestamp=body.to_timestamp,
        name=body.name,
        user_id=body.user_id,
        session_id=body.session_id,
        environment=body.environment,
        tags=body.tags or None,
        order_by=body.order_by,
        max_traces=body.max_traces,
    ):
        rec = convert_trace(trace, fmt=body.format, fields=body.fields)
        if rec is None and body.format == FORMAT_OPENAI_CHAT and trace.get("id"):
            try:
                enriched = client.fetch_trace_with_observations(trace["id"])
                rec = convert_trace(enriched, fmt=body.format, fields=body.fields)
            except LangfuseFetchError as e:
                logger.warning("Could not fetch observations for %s: %s", trace["id"], e)
        if rec is None:
            yield None, True
        else:
            yield rec, False


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 timestamp, tolerating the trailing ``Z`` Langfuse uses."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _within_range(trace: Dict[str, Any], body: LangfuseImportRequest) -> bool:
    """
    Check a fetched trace against the requested window.

    On the id-based path the window can't be pushed down to Langfuse (an
    annotation is written whenever a reviewer gets to it, long after the trace),
    so the trace's own timestamp is what has to be compared here.
    """
    stamp = _parse_iso(trace.get("timestamp"))
    if stamp is None:
        return True
    lower = _parse_iso(body.from_timestamp)
    upper = _parse_iso(body.to_timestamp)
    if lower and stamp < lower:
        return False
    if upper and stamp > upper:
        return False
    return True


def _trace_matches_filters(trace: Dict[str, Any], body: LangfuseImportRequest) -> bool:
    """
    Apply the row filters Langfuse can't help with on the id-based path.

    ``environment`` is pushed down to the observations/scores queries; the rest
    have to be checked against the fetched trace.
    """
    if not _within_range(trace, body):
        return False
    if body.name and trace.get("name") != body.name:
        return False
    if body.user_id and trace.get("userId") != body.user_id:
        return False
    if body.session_id and trace.get("sessionId") != body.session_id:
        return False
    if body.tags:
        trace_tags = trace.get("tags")
        if not isinstance(trace_tags, list) or not set(body.tags).issubset(trace_tags):
            return False
    return True


def _restricted_trace_ids(
    client: LangfuseClient, body: LangfuseImportRequest
) -> Optional[List[str]]:
    """
    Resolve the filters ``/api/public/traces`` can't express into trace ids.

    Model, annotation score and annotation queue each live on a different
    Langfuse resource, so each is resolved separately and the results are
    intersected — asking for "traces of model X that a reviewer scored 5" means
    both, not either. Returns ``None`` when none of these filters is set, which
    is the caller's signal to walk ``/traces`` directly.

    Ordering comes from the first filter that was applied (model, then score,
    then queue) so ``order_by=timestamp.{desc,asc}`` still means something.
    """
    ordered_maps: List[Dict[str, str]] = []

    if body.model:
        ordered_maps.append(
            client.model_trace_times(
                body.model,
                from_timestamp=body.from_timestamp,
                to_timestamp=body.to_timestamp,
                environment=body.environment,
            )
        )

    if body.has_annotation_filter():
        ordered_maps.append(
            client.score_trace_times(
                name=body.score_name,
                source=body.score_source,
                value=body.score_value,
                operator=body.score_operator,
                string_value=body.score_string_value,
                environment=body.environment,
            )
        )

    if body.annotation_queue_id:
        ordered_maps.append(
            client.queue_trace_times(
                body.annotation_queue_id,
                status=body.annotation_queue_status,
            )
        )

    if not ordered_maps:
        return None

    trace_ids = set(ordered_maps[0])
    for extra in ordered_maps[1:]:
        trace_ids &= set(extra)

    primary = ordered_maps[0]
    newest_first = not (body.order_by or "").endswith(".asc")
    return sorted(
        trace_ids, key=lambda t: primary.get(t, ""), reverse=newest_first
    )


def _iter_converted_by_ids(
    client: LangfuseClient, body: LangfuseImportRequest, trace_ids: List[str]
):
    """
    Same contract as ``_iter_converted``, for an already-resolved id list.

    Each trace is fetched individually, which also brings its nested
    observations, so ``openai_chat`` needs no re-fetch on this path.
    """
    cap = body.max_traces or settings.LANGFUSE_MAX_TRACES_PER_IMPORT
    emitted = 0
    for trace_id in trace_ids:
        if emitted >= cap:
            return
        try:
            trace = client.fetch_trace_with_observations(trace_id)
        except LangfuseFetchError as e:
            logger.warning("Could not fetch trace %s: %s", trace_id, e)
            yield None, True
            continue
        if not _trace_matches_filters(trace, body):
            continue
        emitted += 1
        rec = convert_trace(trace, fmt=body.format, fields=body.fields)
        if rec is None:
            yield None, True
        else:
            yield rec, False


# Top-level fields that Langfuse always includes, used as a fallback if we
# can't fetch a sample (empty project).
_FALLBACK_FIELDS = [
    "id", "projectId", "name", "timestamp", "environment", "tags",
    "bookmarked", "release", "version", "userId", "sessionId", "public",
    "input", "output", "metadata", "createdAt", "updatedAt",
    "observations", "scores", "totalCost", "latency", "htmlPath", "externalId",
]


@router.get("/projects")
@limiter.limit("30/minute")
async def list_langfuse_projects(
    request: Request,
    refresh: bool = Query(False, description="Bypass the resolution cache"),
    _user_id: str = Depends(get_current_user_id),
):
    """
    Projects available to import from — the first choice on the import page.

    One entry per configured key pair (a Langfuse API key is project-scoped), so
    a single-project deployment returns exactly one and multi-project ones return
    the projects whose keys this service holds.
    """
    try:
        projects = project_registry.projects(force_refresh=refresh)
    except LangfuseConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except LangfuseFetchError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "projects": [
            {
                "id": p.id,
                "name": p.name,
                "organization": p.organization or None,
                "is_default": p.is_default,
            }
            for p in projects
        ],
        "default_project_id": next((p.id for p in projects if p.is_default), None),
    }


def _merge_score_options(
    configs: List[Dict[str, Any]],
    observed: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build the annotation dropdown from what's defined and what's been used.

    Score *configs* are what the Langfuse UI offers an annotator (so they carry
    the value range and category labels), while the scores actually written tell
    us how many traces a filter would keep — including ad-hoc names that were
    never configured. Both matter, so both are listed.
    """
    options: Dict[str, Dict[str, Any]] = {}
    for config in configs:
        name = str(config.get("name") or "").strip()
        if not name:
            continue
        categories = config.get("categories")
        options[name] = {
            "name": name,
            "data_type": config.get("dataType"),
            "min_value": config.get("minValue"),
            "max_value": config.get("maxValue"),
            "categories": [
                {"value": c.get("value"), "label": c.get("label")}
                for c in categories
                if isinstance(c, dict)
            ]
            if isinstance(categories, list)
            else [],
            "description": config.get("description"),
            "configured": True,
            "sources": [],
            "trace_count": 0,
        }

    for name, seen in observed.items():
        option = options.setdefault(
            name,
            {
                "name": name,
                "data_type": seen.get("data_type"),
                "min_value": None,
                "max_value": None,
                "categories": [],
                "description": None,
                "configured": False,
            },
        )
        option["trace_count"] = len(seen.get("trace_ids") or ())
        option["sources"] = sorted(seen.get("sources") or ())
        option.setdefault("data_type", seen.get("data_type"))
        if not option.get("categories") and seen.get("labels"):
            # Unconfigured categorical score: the labels people actually used are
            # the only category list there is.
            option["categories"] = [
                {"value": None, "label": label} for label in sorted(seen["labels"])
            ]

    # Most-annotated first: that's the filter most likely to yield a dataset.
    return sorted(
        options.values(), key=lambda o: (-int(o.get("trace_count") or 0), o["name"])
    )


@router.get("/annotations")
@limiter.limit("30/minute")
async def list_langfuse_annotations(
    request: Request,
    project_id: Optional[str] = Query(None, description="Langfuse project to read from"),
    source: Optional[str] = Query(
        None, description=f"Restrict counts to one score source: {', '.join(SCORE_SOURCES)}"
    ),
    environment: Optional[str] = Query(None),
    _user_id: str = Depends(get_current_user_id),
):
    """
    Selectable values for the annotation filter: score names (with their value
    range or categories) and annotation queues, within the selected project.

    Counts are distinct *traces* carrying that score, so an option showing 0 means
    filtering on it would produce an empty dataset.
    """
    client = _client(project_id)
    try:
        normalised_source = _one_of(source, SCORE_SOURCES, "source")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        configs = client.fetch_score_configs()
        observed = client.score_names(
            source=normalised_source, environment=environment
        )
        queues = client.fetch_annotation_queues()
    except LangfuseFetchError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "project_id": client.project_id,
        "scores": _merge_score_options(configs, observed),
        "queues": [
            {
                "id": q.get("id"),
                "name": q.get("name") or q.get("id"),
                "description": q.get("description"),
            }
            for q in queues
            if q.get("id")
        ],
        "sources": list(SCORE_SOURCES),
        "operators": list(SCORE_OPERATORS),
        "queue_statuses": list(QUEUE_STATUSES),
    }


@router.get("/fields")
@limiter.limit("30/minute")
async def list_langfuse_fields(
    request: Request,
    project_id: Optional[str] = Query(None, description="Langfuse project to read from"),
    _user_id: str = Depends(get_current_user_id),
):
    """Return the set of top-level trace fields available to project on."""
    client = _client(project_id)
    try:
        sample = client._fetch_page(1, {"limit": 1})  # noqa: SLF001
    except LangfuseFetchError as e:
        raise HTTPException(status_code=502, detail=str(e))
    fields = sorted({k for t in sample for k in (t or {}).keys()}) or _FALLBACK_FIELDS
    return {"fields": fields}


def _group_by_deployed_name(
    trace_ids_by_model: Dict[str, set],
    deployed: set,
) -> Dict[str, set]:
    """
    Collapse traced model spellings onto the gateway's name for that model.

    Traces may record the same deployed model under different spellings
    (``Qwen/...`` from the SDK, ``openai/Qwen/...`` via the proxy). Keying the
    dropdown on the gateway name means one entry per deployed model, with the
    trace counts added up instead of split across near-duplicate options.
    Traced models that aren't deployed drop out here.
    """
    grouped: Dict[str, set] = {}
    for model, trace_ids in trace_ids_by_model.items():
        if not trace_ids:
            continue
        # An exact hit wins over a suffix hit, so near-identical deployed names
        # can't steal each other's traces.
        canonical = next(
            (d for d in deployed if d.strip().lower() == model.strip().lower()),
            None,
        ) or next((d for d in sorted(deployed) if models_match(model, d)), None)
        if canonical is None:
            continue
        grouped.setdefault(canonical, set()).update(trace_ids)
    return grouped


@router.get("/models")
@limiter.limit("30/minute")
async def list_langfuse_models(
    request: Request,
    project_id: Optional[str] = Query(None, description="Langfuse project to read from"),
    from_timestamp: Optional[str] = Query(None, description="ISO 8601 lower bound"),
    to_timestamp: Optional[str] = Query(None, description="ISO 8601 upper bound"),
    environment: Optional[str] = Query(None),
    _user_id: str = Depends(get_current_user_id),
):
    """
    Selectable values for the model filter: models deployed on the GenAI
    gateway that also have at least one trace in the given window, within the
    selected project.

    If the gateway can't be reached we still return the models seen in traces
    and flag ``deployed_filter_applied=false``, so the import page degrades to
    "traced models" rather than an empty dropdown.
    """
    client = _client(project_id)
    try:
        trace_ids_by_model = client.model_trace_ids(
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            environment=environment,
        )
    except LangfuseFetchError as e:
        raise HTTPException(status_code=502, detail=str(e))

    deployed: Optional[set] = None
    warning: Optional[str] = None
    try:
        deployed = GatewayClient().list_deployed_models()
    except (GatewayConfigError, GatewayFetchError) as e:
        warning = f"Could not verify which models are deployed: {e}"
        logger.warning("Deployed-model filter unavailable: %s", e)

    if deployed is None:
        # No canonical names to key off, so offer the spellings traces used.
        grouped = {m: ids for m, ids in trace_ids_by_model.items() if ids}
    else:
        grouped = _group_by_deployed_name(trace_ids_by_model, deployed)

    models = [{"id": model, "trace_count": len(ids)} for model, ids in grouped.items()]
    # Busiest first — that's the most likely training set.
    models.sort(key=lambda m: (-m["trace_count"], m["id"]))

    return {
        "models": models,
        "project_id": client.project_id,
        "deployed_filter_applied": deployed is not None,
        "warning": warning,
    }


@router.post("/preview")
@limiter.limit("30/minute")
async def preview_langfuse(
    request: Request,
    body: LangfuseImportRequest,
    _user_id: str = Depends(get_current_user_id),
):
    """Return the first ``preview_limit`` converted records without persisting anything."""
    _validated_format(body.format, body.fields)
    client = _client(body.project_id)

    records: list = []
    scanned = 0
    skipped = 0
    try:
        for rec, was_skipped in _iter_converted(client, body):
            scanned += 1
            if was_skipped:
                skipped += 1
                continue
            records.append(rec)
            if len(records) >= body.preview_limit:
                break
    except LangfuseFetchError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "records": records,
        "returned": len(records),
        "scanned": scanned,
        "skipped": skipped,
        "project_id": client.project_id,
    }


@router.post("/import")
@limiter.limit("5/minute")
async def import_langfuse(
    request: Request,
    body: LangfuseImportRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Fetch, convert, upload to MinIO, register as a File. Returns metadata."""
    _validated_format(body.format, body.fields)
    client = _client(body.project_id)

    # Fetch + serialise into memory. For very large exports we would stream to
    # MinIO in chunks, but the current file_handler API takes bytes; the
    # `max_traces` cap keeps memory bounded.
    scanned = 0
    skipped = 0
    try:
        records: list = []
        for rec, was_skipped in _iter_converted(client, body):
            scanned += 1
            if was_skipped:
                skipped += 1
                continue
            records.append(rec)
        payload = b"".join(stream_jsonl(records))
    except LangfuseFetchError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if not payload:
        where = f" in project '{client.project_id}'" if client.project_id else ""
        # An annotation filter is the usual reason for "scanned nothing at all":
        # the traces exist, but nobody has reviewed them yet.
        why = (
            " No trace matched the annotation filter — check that the score has "
            "been given to traces in this range."
            if scanned == 0 and (body.has_annotation_filter() or body.annotation_queue_id)
            else ""
        )
        raise HTTPException(
            status_code=404,
            detail=(
                f"No convertible records{where}. Scanned {scanned} trace(s); "
                f"skipped {skipped} because they had no assistant output. "
                f"Try format='raw' to keep everything as-is.{why}"
            ),
        )

    n_records = payload.count(b"\n")
    filename = (body.filename or "").strip()
    if not filename:
        # Name the project in the file: with several projects to import from,
        # "which traces is this?" is otherwise unanswerable from the file list.
        project_part = f"{_filename_safe(client.project_id)}-" if client.project_id else ""
        filename = (
            f"langfuse-{project_part}"
            f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.jsonl"
        )
    if not filename.endswith(".jsonl"):
        filename += ".jsonl"

    # Reuse StorageHandler + MetadataHandler directly so we can pass raw bytes
    # without needing to fake an UploadFile.
    file_handler = FileHandler()
    file_id = file_handler.generate_file_id()
    object_name = f"{user_id}/{file_id}"
    stream = io.BytesIO(payload)
    if not file_handler.storage.upload_file(
        file_id=object_name,
        file_data=stream,
        file_size=len(payload),
        content_type="application/x-ndjson",
    ):
        raise HTTPException(status_code=500, detail="MinIO upload failed.")

    metadata = {
        "id": file_id,
        "object": "file",
        "bytes": len(payload),
        "created_at": int(datetime.utcnow().timestamp()),
        "filename": filename,
        "purpose": "finetuning",
        "status": "processed",
        "status_details": None,
        "user_id": user_id,
        "source": "langfuse",
        "langfuse_project": client.project_id,
    }
    # Provenance for a curated dataset: months later, "which annotation was this
    # filtered on?" is not recoverable from the file contents.
    annotation_provenance = {
        key: value
        for key, value in (
            ("score_name", body.score_name),
            ("score_source", body.score_source),
            ("score_operator", body.score_operator),
            ("score_value", body.score_value),
            ("score_string_value", body.score_string_value),
            ("annotation_queue_id", body.annotation_queue_id),
            ("annotation_queue_status", body.annotation_queue_status),
        )
        if value is not None
    }
    if annotation_provenance:
        metadata["langfuse_annotation_filter"] = annotation_provenance
    MetadataHandler(db).add(file_id, metadata, user_id)

    return {
        "file_id": file_id,
        "filename": filename,
        "bytes": len(payload),
        "n_records": n_records,
        "scanned": scanned,
        "skipped": skipped,
        "project_id": client.project_id,
    }

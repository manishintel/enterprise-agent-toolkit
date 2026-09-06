"""
Semantic routing for a fine-tuned model.

A model fine-tuned on one domain's traces is better at that domain than the
general model, but nothing sends it that traffic: the gateway routes by model
name, so a caller has to know to ask for it. These endpoints close that gap by
mining example queries out of the dataset the model was trained on and using them
as a semantic route, so a matching query goes to the fine-tuned model and
everything else falls back.

Three properties of the setup drive the shape of this module.

**One router is shared by every fine-tuned model.** Callers point at a single
name (``gateway.router_name``)e i and routes accumulate as models are fine-tuned;
the highest-scoring route wins, which is semantic-router's own behaviour. So an
update has to mergnto the existing config rather than replace it, or applying
one model's route would silently delete another's.

**The gateway is the source of truth.** It returns ``auto_router_config`` in
clear from ``/model/info``, so the applied utterances are read back from there
rather than kept in a second copy here.

**Applying a change restarts the gateway.** An auto-router is cached in the
gateway process by model name and re-registering it does not refresh it (verified;
this build has no ``/config/reload``). The restart is a rollout of one Deployment
and is stated plainly to the caller rather than hidden.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Request

from ..auth import get_current_user
from ..config import get_settings
from ..database import db_manager
from ..errors import (
    InvalidRequestError, ResourceNotFoundError, ServiceUnavailableError,
    PermissionError as ForbiddenError,
)
from ..gateway import GatewayError, GatewayUnavailable, gateway_client
from ..k8s import KubeApiError, kube_client
from ..middleware import limiter
from ..model_naming import resolve_served_model_name
from ..observability import get_logger
from ..schemas import (
    ExtractUtterancesRequest, ExtractUtterancesResponse, SemanticRouteRequest,
    SemanticRouteStatus, SemanticRouteTestRequest, SemanticRouteTestResponse,
)
from ..utterances import ROUTE_TOP_K, extract as extract_utterances
from ..utterances import _cosine, route_score
from ..files_client import read_training_file

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/fine_tuning", tags=["Fine-tuning"])

# Effectively "no limit", for the pass that collects every surviving candidate
# before selection narrows it.
_ALL_CANDIDATES = 100_000
settings = get_settings()


async def _load_owned_job(job_id: str, user_id: str) -> Dict[str, Any]:
    row = await db_manager.fetch_one("""
        SELECT id, model, status, created_at, fine_tuned_model, suffix, user_id, training_file
        FROM fine_tuning_jobs
        WHERE id = $1
    """, job_id, timeout=30)
    if not row:
        raise ResourceNotFoundError("fine-tuning job", job_id)
    if str(row.get("user_id", "")) != str(user_id):
        logger.warning(
            "Unauthorized semantic route access attempt — returning 403",
            extra={"job_id": job_id, "requesting_user_id": user_id},
        )
        raise ForbiddenError("You do not have permission to access this fine-tuning job")
    return dict(row)


def _served_model_name(job_row: Dict[str, Any]) -> str:
    return resolve_served_model_name(job_row["model"], job_row["created_at"], job_row.get("suffix"))


def _parse_router(model_entry: Optional[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    """Existing routes, default model and embedding model from a router entry."""
    if not model_entry:
        return [], None, None
    params = model_entry.get("litellm_params") or {}
    raw = params.get("auto_router_config")
    routes: List[Dict[str, Any]] = []
    if raw:
        try:
            routes = (json.loads(raw) if isinstance(raw, str) else raw).get("routes", []) or []
        except (ValueError, AttributeError):
            logger.warning("Router config on the gateway is not readable JSON; treating it as empty")
    return routes, params.get("auto_router_default_model"), params.get("auto_router_embedding_model")


def _entry(route: Dict[str, Any], this_model: Optional[str]) -> Dict[str, Any]:
    return {
        # A route's name *is* the model it routes to.
        "model": route.get("name") or "",
        "utterances": route.get("utterances") or [],
        "score_threshold": route.get("score_threshold"),
        "description": route.get("description"),
        "is_this_job": bool(this_model) and route.get("name") == this_model,
    }


async def _status(job_row: Dict[str, Any]) -> SemanticRouteStatus:
    config = settings.gateway
    this_model = _served_model_name(job_row)
    base = {
        "router_name": config.router_name,
        "this_model": this_model,
        "restart_required_on_apply": config.restart_on_apply,
    }

    if not gateway_client.configured:
        return SemanticRouteStatus(
            available=False,
            message="The GenAI Gateway is not configured for this service, so semantic routing "
                    "cannot be set up from here.",
            **base,
        )

    try:
        models = await gateway_client.list_models()
    except GatewayUnavailable as exc:
        return SemanticRouteStatus(available=False, message=str(exc), **base)
    except GatewayError as exc:
        return SemanticRouteStatus(available=False, message=f"Gateway rejected the request: {exc}", **base)

    entry = next((m for m in models if m.get("model_name") == config.router_name), None)
    routes, default_model, embedding_model = _parse_router(entry)
    chat_models = [
        m["model_name"] for m in models if (m.get("model_info") or {}).get("mode") == "chat"
    ]
    embedding_models = [
        m["model_name"] for m in models if (m.get("model_info") or {}).get("mode") == "embedding"
    ]

    entries = [_entry(r, this_model) for r in routes]
    mine = next((e for e in entries if e["is_this_job"]), None)

    message = None
    if not embedding_models:
        message = (
            "No embedding model is registered with the gateway. A semantic router needs one to "
            "encode utterances and queries, so routing cannot be set up until one is deployed."
        )
    elif this_model not in chat_models:
        message = (
            f"'{this_model}' is not registered with the gateway. Deploy the model first — a route "
            f"names the model it routes to, so the target has to exist."
        )

    return SemanticRouteStatus(
        available=True,
        message=message,
        configured=entry is not None,
        this_route=mine,
        routes=entries,
        default_model=default_model,
        embedding_model=embedding_model or (embedding_models[0] if embedding_models else None),
        available_embedding_models=embedding_models,
        available_chat_models=chat_models,
        **base,
    )


@router.post("/jobs/{job_id}/utterances", response_model=ExtractUtterancesResponse)
@limiter.limit(f"{settings.rate_limit.job_read}/minute")
async def extract_job_utterances(
    request: Request,
    job_id: str,
    options: Optional[ExtractUtterancesRequest] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Mine candidate route utterances from this job's training dataset.

    Computes and returns; nothing is persisted and the router is untouched, so
    this is safe to run repeatedly while tuning the knobs. The response carries a
    per-stage report of what was dropped and why.
    """
    user_id = current_user["user_id"]
    job_row = await _load_owned_job(job_id, user_id)
    options = options or ExtractUtterancesRequest()

    if options.min_words >= options.max_words:
        raise InvalidRequestError(
            "min_words must be smaller than max_words", param="min_words", code="invalid_range"
        )

    training_file = job_row.get("training_file")
    if not training_file:
        raise InvalidRequestError("This job has no training file recorded", param="job_id")

    rows = await read_training_file(training_file, str(user_id))

    # Selection is more useful when "diverse" means diverse in meaning rather
    # than in wording, which needs embeddings. Optional: extract() falls back to
    # word overlap and says so in the report.
    embedding_model = None
    if gateway_client.configured:
        try:
            available = await gateway_client.embedding_models()
            embedding_model = available[0] if available else None
        except (GatewayError, GatewayUnavailable) as exc:
            logger.info(f"Embedding model lookup failed: {exc}")

    # Two passes: one to find every candidate that survives filtering, so the
    # vectors cover the whole set rather than an already-narrowed slice, then the
    # real pass which selects from among them.
    lookup = None
    if embedding_model:
        candidates = [
            u["text"]
            for u in extract_utterances(
                rows,
                limit=_ALL_CANDIDATES,
                min_words=options.min_words,
                max_words=options.max_words,
                first_turn_only=options.first_turn_only,
                redact_pii=options.redact_pii,
            )["utterances"]
        ]
        if len(candidates) > options.limit:
            try:
                vectors = await gateway_client.embed(candidates, embedding_model)
                by_text = dict(zip(candidates, vectors))
                lookup = lambda texts: [by_text[text] for text in texts]  # noqa: E731
            except (GatewayError, GatewayUnavailable) as exc:
                logger.warning(f"Could not embed candidates, selecting lexically instead: {exc}")

    result = extract_utterances(
        rows,
        limit=options.limit,
        min_words=options.min_words,
        max_words=options.max_words,
        first_turn_only=options.first_turn_only,
        redact_pii=options.redact_pii,
        vectors_for=lookup,
    )
    result["training_file"] = training_file
    return result


@router.get("/jobs/{job_id}/semantic-route", response_model=SemanticRouteStatus)
@limiter.limit(f"{settings.rate_limit.job_read}/minute")
async def get_semantic_route(
    request: Request,
    job_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """The shared router's state, and this job's route within it."""
    job_row = await _load_owned_job(job_id, current_user["user_id"])
    return await _status(job_row)


@router.put("/jobs/{job_id}/semantic-route", response_model=SemanticRouteStatus)
@limiter.limit(f"{settings.rate_limit.job_create}/minute")
async def put_semantic_route(
    request: Request,
    job_id: str,
    body: SemanticRouteRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Point matching queries at this job's model.

    Merges this model's route into the shared router, leaving other models' routes
    alone, then restarts the gateway so the change takes effect.
    """
    job_row = await _load_owned_job(job_id, current_user["user_id"])
    status = await _status(job_row)
    if not status.available:
        raise ServiceUnavailableError(status.message or "Semantic routing is unavailable")
    if status.message:
        # A blocking condition: no encoder, or the target model is not registered.
        raise InvalidRequestError(status.message, code="semantic_route_unavailable")

    config = settings.gateway
    this_model = status.this_model
    utterances = [u.strip() for u in body.utterances if u and u.strip()]
    if not utterances:
        raise InvalidRequestError("At least one utterance is required", param="utterances")

    default_model = body.default_model or status.default_model
    if not default_model:
        # Anything but this model: the fallback exists for queries this model is
        # not the right answer to.
        default_model = next((m for m in status.available_chat_models if m != this_model), None)
    if not default_model:
        raise InvalidRequestError(
            "No fallback model is available. A router needs somewhere to send queries that match "
            "nothing, and the only registered chat model is this one.",
            code="no_default_model",
        )
    if default_model == this_model:
        raise InvalidRequestError(
            "The fallback model cannot be this model — everything would route here.",
            param="default_model", code="invalid_default_model",
        )

    embedding_model = status.embedding_model
    routes = [
        {
            "name": r.model,
            "utterances": r.utterances,
            "score_threshold": r.score_threshold,
            "description": r.description,
        }
        for r in status.routes
        if r.model != this_model
    ]
    routes.append({
        "name": this_model,
        "utterances": utterances,
        "score_threshold": body.score_threshold,
        "description": body.description or f"Fine-tuned on the dataset of job {job_id}",
    })

    try:
        if status.configured:
            await gateway_client.delete_model(config.router_name)
        await gateway_client.create_auto_router(
            router_name=config.router_name,
            routes=routes,
            default_model=default_model,
            embedding_model=embedding_model,
        )
    except (GatewayError, GatewayUnavailable) as exc:
        raise ServiceUnavailableError(f"Could not update the router on the gateway: {exc}") from exc

    restarted = await _restart_gateway_if_needed()

    logger.info(
        "Semantic route applied",
        extra={
            "job_id": job_id,
            "router": config.router_name,
            "model": this_model,
            "utterances": len(utterances),
            "score_threshold": body.score_threshold,
            "gateway_restarted": restarted,
        },
    )

    # Reported from what was just written rather than re-read from the gateway:
    # the restart means a read here races a pod that has not loaded its models
    # yet, which comes back looking like the apply silently failed.
    if restarted:
        message = (
            "Applied. The gateway is restarting so the change takes effect — routing may be "
            "briefly interrupted, and the new route is live once it is back, usually within a minute."
        )
    elif config.restart_on_apply:
        message = (
            "The route was saved but the gateway could not be restarted from here, so it is still "
            "serving the previous router. Restart the gateway deployment for this to take effect."
        )
    else:
        message = (
            "The route was saved. Restarting the gateway is disabled, so it takes effect the next "
            "time the gateway restarts."
        )

    return _applied_status(status, routes, default_model, embedding_model, message)


def _applied_status(
    previous: SemanticRouteStatus,
    routes: List[Dict[str, Any]],
    default_model: Optional[str],
    embedding_model: Optional[str],
    message: Optional[str],
) -> SemanticRouteStatus:
    """The state just written, without reading it back through a restarting gateway."""
    entries = [_entry(r, previous.this_model) for r in routes]
    return SemanticRouteStatus(
        available=True,
        message=message,
        router_name=previous.router_name,
        configured=bool(routes),
        this_model=previous.this_model,
        this_route=next((e for e in entries if e["is_this_job"]), None),
        routes=entries,
        default_model=default_model,
        embedding_model=embedding_model,
        available_embedding_models=previous.available_embedding_models,
        available_chat_models=previous.available_chat_models,
        restart_required_on_apply=previous.restart_required_on_apply,
    )


@router.delete("/jobs/{job_id}/semantic-route", response_model=SemanticRouteStatus)
@limiter.limit(f"{settings.rate_limit.job_cancel}/minute")
async def delete_semantic_route(
    request: Request,
    job_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Stop routing to this job's model, leaving other models' routes in place."""
    job_row = await _load_owned_job(job_id, current_user["user_id"])
    before = await _status(job_row)
    removed = await remove_route_for_model(_served_model_name(job_row))
    if not removed:
        return before
    # As with apply: report what was written, not a read that races the restart.
    remaining = [
        {
            "name": r.model,
            "utterances": r.utterances,
            "score_threshold": r.score_threshold,
            "description": r.description,
        }
        for r in before.routes
        if not r.is_this_job
    ]
    return _applied_status(
        before,
        remaining,
        before.default_model,
        before.embedding_model,
        "Removed. The gateway is restarting so the change takes effect; queries that used to route "
        "here go to the fallback model once it is back.",
    )


async def remove_route_for_model(model_name: str) -> bool:
    """
    Drop a model's route from the shared router.

    Also called when a model is undeployed: a route naming a model that no longer
    exists would send matching queries into a 404.
    """
    config = settings.gateway
    if not gateway_client.configured:
        return False
    try:
        entry = await gateway_client.get_model(config.router_name)
    except (GatewayError, GatewayUnavailable) as exc:
        logger.info(f"Could not read the router while removing {model_name}: {exc}")
        return False
    if not entry:
        return False

    routes, default_model, embedding_model = _parse_router(entry)
    remaining = [r for r in routes if r.get("name") != model_name]
    if len(remaining) == len(routes):
        return False

    try:
        await gateway_client.delete_model(config.router_name)
        if remaining:
            await gateway_client.create_auto_router(
                router_name=config.router_name,
                routes=remaining,
                default_model=default_model,
                embedding_model=embedding_model,
            )
        # With no routes left the router is removed entirely rather than left as
        # an entry that forwards everything to the fallback.
    except (GatewayError, GatewayUnavailable) as exc:
        logger.warning(f"Could not update the router while removing {model_name}: {exc}")
        return False

    await _restart_gateway_if_needed()
    logger.info("Semantic route removed", extra={"model": model_name, "routes_left": len(remaining)})
    return True


async def _restart_gateway_if_needed() -> bool:
    """
    Restart the gateway so a router change takes effect.

    Necessary because the auto-router is cached in the gateway process by model
    name: re-registering the entry does not rebuild it, and this build has no
    config-reload endpoint. Returns False when it could not be done, so the caller
    can say so instead of implying the change is live.
    """
    config = settings.gateway
    if not config.restart_on_apply:
        return False
    if not kube_client.available:
        return False
    try:
        await kube_client.restart_deployment(config.namespace, config.deployment)
        return True
    except KubeApiError as exc:
        logger.warning(
            "Could not restart the gateway after a router change",
            extra={"status": exc.status, "error": exc.message},
        )
        return False


@router.post("/jobs/{job_id}/semantic-route/test", response_model=SemanticRouteTestResponse)
@limiter.limit(f"{settings.rate_limit.job_read}/minute")
async def test_semantic_route(
    request: Request,
    job_id: str,
    body: SemanticRouteTestRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Score a query against the routes, without sending it anywhere.

    Calibration needs numbers: an encoder's similarity floor for *unrelated* text
    is well above zero (~0.46 for bge-base), so a threshold that looks strict can
    still match everything. This reports the score and the utterance responsible,
    so a threshold can be chosen from evidence.
    """
    job_row = await _load_owned_job(job_id, current_user["user_id"])
    status = await _status(job_row)
    if not status.available or not status.embedding_model:
        raise ServiceUnavailableError(
            status.message or "No embedding model is available to score against."
        )

    this_model = status.this_model
    # Either the candidate set being reviewed, or what is actually applied.
    if body.utterances:
        groups: List[Tuple[str, List[str], Optional[float]]] = [
            (this_model, [u for u in body.utterances if u.strip()], body.score_threshold)
        ]
    else:
        groups = [(r.model, r.utterances, r.score_threshold) for r in status.routes if r.utterances]
    if not groups or not any(g[1] for g in groups):
        raise InvalidRequestError(
            "There are no utterances to score against. Extract some first, or pass them in.",
            param="utterances",
        )

    flat = [u for _, utts, _ in groups for u in utts]
    try:
        vectors = await gateway_client.embed([body.query] + flat, status.embedding_model)
    except (GatewayError, GatewayUnavailable) as exc:
        raise ServiceUnavailableError(f"Could not embed the query: {exc}") from exc

    query_vec, rest = vectors[0], vectors[1:]
    offset = 0
    scores: List[Dict[str, Any]] = []
    for model_name, utts, threshold in groups:
        vectors = rest[offset:offset + len(utts)]
        offset += len(utts)
        # Scored the way the gateway's router scores it -- the mean of the nearest
        # ROUTE_TOP_K utterances. Using the best match instead would read far
        # higher and promise matches the router then refuses.
        score = route_score(query_vec, vectors)
        best_utterance, best_single = None, 0.0
        for utterance, vector in zip(utts, vectors):
            single = _cosine(query_vec, vector)
            if single > best_single:
                best_single, best_utterance = single, utterance
        effective = body.score_threshold if body.score_threshold is not None else (threshold or 0.0)
        scores.append({
            "model": model_name,
            "score": round(score, 4),
            "threshold": effective,
            "closest_utterance": best_utterance,
            # The single best match, for explaining *why* -- it is not what the
            # decision is made on.
            "closest_score": round(best_single, 4),
            "utterances_scored": min(len(utts), ROUTE_TOP_K),
            "would_match": score >= effective,
        })

    scores.sort(key=lambda s: s["score"], reverse=True)
    winner = next((s for s in scores if s["would_match"]), None)
    top = scores[0]
    return SemanticRouteTestResponse(
        query=body.query,
        matched=winner is not None,
        matched_model=winner["model"] if winner else status.default_model,
        score=top["score"],
        threshold=top["threshold"],
        closest_utterance=top["closest_utterance"],
        scores=scores,
    )

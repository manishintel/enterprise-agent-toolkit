"""
Semantic routing for a fine-tuned model.

A model fine-tuned on one domain's traces is better at that domain than the
general model, but nothing sends it that traffic: the gateway routes by model
name, so a caller has to know to ask for it. These endpoints close that gap by
mining example queries out of the dataset the model was trained on and using them
as a semantic route, so a matching query goes to the fine-tuned model and
everything else falls back.

Three properties of the setup drive the shape of this module.

**A router is shared by every model routed through it.** Callers point at one
name and routes accumulate as models are fine-tuned; the highest-scoring route
wins, which is semantic-router's own behaviour. So an update has to merge into the
existing config rather than replace it, or applying one model's route would
silently delete another's. Several routers can coexist -- ``gateway.router_name``
is only the default, and a caller may name another to keep unrelated sets of
routes apart -- so anything that removes a model's route has to look in all of
them, not just the default.

**The gateway is the source of truth.** It returns ``auto_router_config`` in
clear from ``/model/info``, so the applied utterances are read back from there
rather than kept in a second copy here.

**Applying a change restarts the gateway.** An auto-router is cached in the
gateway process by model name and re-registering it does not refresh it (verified;
this build has no ``/config/reload``). The restart is a rollout of one Deployment,
which takes long enough that a caller cannot be left guessing: ``/readiness``
reports the rollout and re-reads the router, so "is it live yet" is answered from
the cluster and the gateway rather than from a timer.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Query, Request

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
    ExtractUtterancesRequest, ExtractUtterancesResponse, GatewayReadiness,
    RoutedModel, RouterSummary, SemanticRouteRequest, SemanticRouteStatus,
    SemanticRouteTestRequest, SemanticRouteTestResponse,
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

# A router name is a model name to every client that calls it, and it ends up in
# URLs and config, so keep it to something unambiguous rather than accepting any
# string the UI happens to send.
_ROUTER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,119}$")


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


def _is_router(model_entry: Dict[str, Any]) -> bool:
    """Whether a gateway model entry is an auto-router rather than a served model."""
    return bool((model_entry.get("litellm_params") or {}).get("auto_router_config"))


def _auto_routers(models: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Every auto-router the gateway has registered, by name."""
    return {
        m["model_name"]: m
        for m in models
        if m.get("model_name") and _is_router(m)
    }


def _validate_router_name(name: str, status: SemanticRouteStatus) -> None:
    """
    Check a caller-chosen router name before it becomes a model name.

    Two failures are worth catching here rather than letting the gateway return a
    confusing error: a name that is not usable as a model name, and one already
    taken by a *served* model. The latter would make the router shadow the model it
    collides with, so requests to that name would stop reaching it.
    """
    if not _ROUTER_NAME_RE.match(name):
        raise InvalidRequestError(
            "A router name must be 2-120 characters of letters, digits, dot, dash or underscore, "
            "starting with a letter or digit.",
            param="router_name", code="invalid_router_name",
        )
    existing_routers = {r.name for r in status.available_routers}
    served = set(status.available_chat_models) | set(status.available_embedding_models)
    if name in served - existing_routers:
        raise InvalidRequestError(
            f"'{name}' is already registered as a model on the gateway. Pick a different router "
            f"name — a router with that name would shadow the model.",
            param="router_name", code="router_name_taken",
        )


def _entry(route: Dict[str, Any], this_model: Optional[str]) -> Dict[str, Any]:
    return {
        # A route's name *is* the model it routes to.
        "model": route.get("name") or "",
        "utterances": route.get("utterances") or [],
        "score_threshold": route.get("score_threshold"),
        "description": route.get("description"),
        "is_this_job": bool(this_model) and route.get("name") == this_model,
    }


async def _status(
    job_row: Dict[str, Any], router_name: Optional[str] = None
) -> SemanticRouteStatus:
    """
    The state of one router, and this job's route within it.

    ``router_name`` selects which router is being looked at; unset means the
    installation default. A name with no router behind it is not an error -- it is
    the "create a new one" case, and comes back with ``configured`` false and an
    empty route list, which is exactly what a blank form needs.
    """
    config = settings.gateway
    this_model = _served_model_name(job_row)
    requested = (router_name or config.router_name).strip() or config.router_name
    base = {
        "router_name": requested,
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

    routers = _auto_routers(models)
    entry = routers.get(requested)
    routes, default_model, embedding_model = _parse_router(entry)
    chat_models = [
        m["model_name"] for m in models if (m.get("model_info") or {}).get("mode") == "chat"
    ]
    embedding_models = [
        m["model_name"] for m in models if (m.get("model_info") or {}).get("mode") == "embedding"
    ]

    # Every router, so the caller can offer a choice instead of only ever writing
    # to the default. Sorted with the default first: it is what an unset request
    # gets, so it belongs at the top of a picker.
    summaries: List[RouterSummary] = []
    routed: List[RoutedModel] = []
    for name, router_entry in routers.items():
        other_routes, _, _ = _parse_router(router_entry)
        summaries.append(RouterSummary(
            name=name,
            routes=len(other_routes),
            is_default=name == config.router_name,
            has_this_model=any(r.get("name") == this_model for r in other_routes),
        ))
        routed.extend(
            RoutedModel(
                model=r.get("name") or "",
                router=name,
                utterances=len(r.get("utterances") or []),
            )
            for r in other_routes
            if r.get("name")
        )
    summaries.sort(key=lambda s: (not s.is_default, s.name))

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
        available_routers=summaries,
        routed_models=routed,
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
    router_name: Optional[str] = Query(
        default=None,
        max_length=120,
        description="Which router to read. Unset means the installation default.",
    ),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    A router's state, and this job's route within it.

    Every registered router comes back in ``available_routers`` regardless of which
    one was asked for, so a caller can populate a picker from a single call.
    """
    job_row = await _load_owned_job(job_id, current_user["user_id"])
    return await _status(job_row, router_name)


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

    Merges this model's route into the chosen router (``body.router_name``, or the
    installation default), leaving other models' routes in it alone, then restarts
    the gateway so the change takes effect. A name with no router behind it creates
    one, so a caller can keep unrelated sets of routes apart.
    """
    job_row = await _load_owned_job(job_id, current_user["user_id"])
    requested_name = (body.router_name or "").strip() or None
    status = await _status(job_row, requested_name)
    if not status.available:
        raise ServiceUnavailableError(status.message or "Semantic routing is unavailable")
    if status.message:
        # A blocking condition: no encoder, or the target model is not registered.
        raise InvalidRequestError(status.message, code="semantic_route_unavailable")

    config = settings.gateway
    # status.router_name is the requested name already defaulted, so validate that
    # rather than the raw body: a caller-chosen name and the default both end up
    # registered as a model name and both have to be usable as one.
    router_name = status.router_name
    _validate_router_name(router_name, status)
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
            await gateway_client.delete_model(router_name)
        await gateway_client.create_auto_router(
            router_name=router_name,
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
            "router": router_name,
            "router_created": not status.configured,
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
            "briefly interrupted. Follow the readiness endpoint to see when the route is live."
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

    return _applied_status(
        status, routes, default_model, embedding_model, message, restarting=restarted
    )


def _applied_status(
    previous: SemanticRouteStatus,
    routes: List[Dict[str, Any]],
    default_model: Optional[str],
    embedding_model: Optional[str],
    message: Optional[str],
    restarting: bool = False,
) -> SemanticRouteStatus:
    """The state just written, without reading it back through a restarting gateway."""
    entries = [_entry(r, previous.this_model) for r in routes]
    # The router list is carried over with this router's own count corrected, so a
    # picker does not lose its options or show a stale count after an apply.
    summaries = [r for r in previous.available_routers if r.name != previous.router_name]
    if routes:
        summaries.append(RouterSummary(
            name=previous.router_name,
            routes=len(routes),
            is_default=previous.router_name == settings.gateway.router_name,
            has_this_model=any(e["is_this_job"] for e in entries),
        ))
    summaries.sort(key=lambda s: (not s.is_default, s.name))
    routed = [r for r in previous.routed_models if r.router != previous.router_name]
    routed.extend(
        RoutedModel(
            model=e["model"],
            router=previous.router_name,
            utterances=len(e["utterances"]),
        )
        for e in entries
        if e["model"]
    )
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
        available_routers=summaries,
        routed_models=routed,
        restart_required_on_apply=previous.restart_required_on_apply,
        gateway_restarting=restarting,
    )


@router.delete("/jobs/{job_id}/semantic-route", response_model=SemanticRouteStatus)
@limiter.limit(f"{settings.rate_limit.job_cancel}/minute")
async def delete_semantic_route(
    request: Request,
    job_id: str,
    router_name: Optional[str] = Query(
        default=None,
        max_length=120,
        description="Remove the route from this router only. Unset removes it from every router.",
    ),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Stop routing to this job's model, leaving other models' routes in place."""
    job_row = await _load_owned_job(job_id, current_user["user_id"])
    before = await _status(job_row, router_name)
    removed = await remove_route_for_model(
        _served_model_name(job_row), router_name=before.router_name if router_name else None
    )
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
        restarting=settings.gateway.restart_on_apply and kube_client.available,
    )


async def remove_route_for_model(model_name: str, router_name: Optional[str] = None) -> bool:
    """
    Drop a model's route from a router, or from every router.

    Also called when a model is undeployed, with no router named: a route pointing
    at a model that no longer exists would send matching queries into a 404, and
    since routes can live in any router that cleanup has to sweep all of them
    rather than only the installation default.

    The gateway is restarted once at the end rather than per router, so removing a
    model that appears in three routers is still one rollout.
    """
    if not gateway_client.configured:
        return False

    try:
        if router_name:
            entry = await gateway_client.get_model(router_name)
            targets = {router_name: entry} if entry else {}
        else:
            targets = _auto_routers(await gateway_client.list_models())
    except (GatewayError, GatewayUnavailable) as exc:
        logger.info(f"Could not read the gateway while removing {model_name}: {exc}")
        return False
    if not targets:
        return False

    changed = False
    for name, entry in targets.items():
        routes, default_model, embedding_model = _parse_router(entry)
        remaining = [r for r in routes if r.get("name") != model_name]
        if len(remaining) == len(routes):
            continue
        try:
            await gateway_client.delete_model(name)
            if remaining:
                await gateway_client.create_auto_router(
                    router_name=name,
                    routes=remaining,
                    default_model=default_model,
                    embedding_model=embedding_model,
                )
            # With no routes left the router is removed entirely rather than left as
            # an entry that forwards everything to the fallback.
        except (GatewayError, GatewayUnavailable) as exc:
            logger.warning(f"Could not update router '{name}' while removing {model_name}: {exc}")
            continue
        changed = True
        logger.info(
            "Semantic route removed",
            extra={"model": model_name, "router": name, "routes_left": len(remaining)},
        )

    if not changed:
        return False
    await _restart_gateway_if_needed()
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


# Polled every few seconds while the gateway rolls, so it is held to the default
# allowance rather than the job-read one -- at 60/minute a two-second poll from a
# single open tab would exhaust it and start failing the very check it is watching.
@router.get("/jobs/{job_id}/semantic-route/readiness", response_model=GatewayReadiness)
@limiter.limit(f"{settings.rate_limit.default}/minute")
async def get_route_readiness(
    request: Request,
    job_id: str,
    router_name: Optional[str] = Query(default=None, max_length=120),
    expect_route: bool = Query(
        default=True,
        description="False after a removal, where the change is live once the route is *gone*.",
    ),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Whether a router change is live yet.

    Two signals, because either alone lies. The Deployment can report ready while
    the new process is still loading its model list, and the gateway can answer on
    an old pod that still has the previous router cached. So this waits for the
    rollout to settle *and* for the route to be readable back out of the gateway.

    ``expect_route`` inverts the second signal for a removal: there the change is
    live once the route no longer comes back, and waiting for it to appear would
    never finish.
    """
    job_row = await _load_owned_job(job_id, current_user["user_id"])
    config = settings.gateway
    this_model = _served_model_name(job_row)
    name = (router_name or "").strip() or config.router_name

    result = GatewayReadiness(router_name=name)

    # Rollout progress. Unavailable is not a failure: the restart may have been
    # done by hand, or this service may not have cluster access, in which case the
    # gateway read below is the only signal and is enough on its own.
    rollout_settled = True
    if kube_client.available:
        try:
            deployment = await kube_client.get_deployment(config.namespace, config.deployment)
        except KubeApiError as exc:
            logger.info(f"Could not read the gateway deployment: {exc.message}")
            deployment = None
        if deployment:
            spec = deployment.get("spec") or {}
            state = deployment.get("status") or {}
            desired = spec.get("replicas", 1)
            result.replicas_desired = desired
            result.replicas_ready = state.get("readyReplicas") or 0
            # updatedReplicas counts pods on the *new* template, so a rollout that
            # has not started yet is caught rather than read as already finished.
            updated = state.get("updatedReplicas") or 0
            generation_seen = state.get("observedGeneration", 0) >= (deployment.get("metadata") or {}).get("generation", 0)
            rollout_settled = bool(
                generation_seen
                and updated >= desired
                and (result.replicas_ready or 0) >= desired
                and not (state.get("unavailableReplicas") or 0)
            )

    try:
        entry = await gateway_client.get_model(name)
        result.gateway_responding = True
        routes, _, _ = _parse_router(entry)
        result.router_present = entry is not None and bool(routes)
        result.route_present = any(r.get("name") == this_model for r in routes)
    except (GatewayError, GatewayUnavailable) as exc:
        # Expected while the pod is coming back up; it is progress, not an error.
        logger.debug(f"Gateway not answering during readiness check: {exc}")

    route_as_expected = result.route_present is expect_route
    result.ready = rollout_settled and result.gateway_responding and route_as_expected
    result.restarting = not result.ready

    if result.ready:
        result.message = (
            f"Route is live. Call '{name}' to be routed."
            if expect_route
            else "The route is gone; matching queries now go to the fallback model."
        )
    elif not rollout_settled:
        result.message = "The gateway is restarting so the new router is loaded."
    elif not result.gateway_responding:
        result.message = "Waiting for the gateway to answer again."
    elif not expect_route:
        result.message = "The gateway is back; waiting for the old route to clear."
    elif not result.router_present:
        result.message = "The gateway is back; waiting for it to load the router."
    else:
        result.message = "The gateway is back; waiting for this model's route to appear."
    return result


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
    # Scored against the chosen router's other routes, since those are the routes a
    # query would actually be competing with once this one is applied there.
    status = await _status(job_row, body.router_name)
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

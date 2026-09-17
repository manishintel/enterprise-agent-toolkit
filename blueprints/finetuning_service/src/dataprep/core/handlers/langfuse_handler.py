"""
Langfuse trace fetcher + format converter.

Talks to a Langfuse instance over its public REST API using HTTP Basic auth
(public_key:secret_key). Supports row-level filters and three output formats:

  * ``openai_chat`` — {"messages": [{"role", "content"}, ...]} (fine-tuning ready)
  * ``raw``        — the full trace object as returned by Langfuse
  * ``custom``     — pick specific top-level fields

For ``openai_chat`` the system turn can be kept, dropped or replaced on the way
out — see ``SYSTEM_KEEP``/``SYSTEM_DROP``/``SYSTEM_REPLACE``.

Traces can also be narrowed to those a human judged: Langfuse keeps annotations
as *scores* (source ``ANNOTATION``) and groups review work into *annotation
queues*, and neither can be expressed as a trace filter — so both are resolved to
trace ids first, exactly like the model filter.

A Langfuse public API key belongs to exactly one project, so *which* project we
read traces from is decided by *which* key pair we authenticate with — that is
also what keeps projects apart, since one project's keys cannot see another's
traces. ``LangfuseProjectRegistry`` turns the configured key pairs into a
selectable list of projects by asking each pair which project it belongs to.

Nothing here talks to MinIO/Postgres; the caller wires it up.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import requests

from core.config import settings
from core.handlers.gateway_handler import models_match

logger = logging.getLogger(__name__)

# Supported output formats
FORMAT_OPENAI_CHAT = "openai_chat"
FORMAT_RAW = "raw"
FORMAT_CUSTOM = "custom"
_ALLOWED_FORMATS = {FORMAT_OPENAI_CHAT, FORMAT_RAW, FORMAT_CUSTOM}

# What to do with the system message of an imported exchange. Langfuse records
# the prompt exactly as it was sent, system turn included, so whatever the
# caller used at request time becomes part of the training input.
#
# That is wrong whenever the system turn carried something the model will not
# have at inference. The case this exists for is distillation from a grounded
# teacher: a reference document is put in the system turn so the expensive model
# answers from fact rather than memory. Kept, it teaches the student to copy an
# answer out of its context, which is a skill it cannot use once the context is
# gone - training loss looks excellent and the served model invents figures.
#
# REPLACE is the useful one: drop the per-request grounding and substitute the
# fixed instruction the model really will be served with, so the training
# examples match inference.
SYSTEM_KEEP = "keep"
SYSTEM_DROP = "drop"
SYSTEM_REPLACE = "replace"
_ALLOWED_SYSTEM_MODES = {SYSTEM_KEEP, SYSTEM_DROP, SYSTEM_REPLACE}

_PAGE_SIZE = 100  # Langfuse hard-caps at 100

# Langfuse types every observation. GENERATION is an LLM call that produced an
# assistant turn; EMBEDDING is a vector call, SPAN/CHAIN are plumbing. Only
# GENERATION can yield a training example, and a trace carrying none of them
# holds nothing to import - LiteLLM records its own health checks and every
# embedding request as a trace named exactly like a real chat request.
_GENERATION_TYPE = "GENERATION"

# Score sources Langfuse records. ANNOTATION is a human verdict entered in the
# Langfuse UI (or an annotation queue), API a programmatic score, EVAL an
# evaluator's output — only the first is "annotated by a person".
SCORE_SOURCE_ANNOTATION = "ANNOTATION"
SCORE_SOURCES = (SCORE_SOURCE_ANNOTATION, "API", "EVAL")

# Comparison operators Langfuse accepts alongside a numeric score value.
SCORE_OPERATORS = ("=", "!=", ">", ">=", "<", "<=")

# Annotation queues hold trace, observation and session items; only the trace
# ones can seed a trace import.
_QUEUE_TRACE_OBJECT = "TRACE"
QUEUE_STATUSES = ("PENDING", "COMPLETED")

# Keys a GENERATION observation may carry the model name under, in priority
# order. Which one is populated depends on the ingestion path (native SDK vs
# the OTEL callback LiteLLM uses).
_MODEL_KEYS = ("model", "providedModelName")


class LangfuseConfigError(RuntimeError):
    """Raised when Langfuse credentials/URL are missing."""


class LangfuseFetchError(RuntimeError):
    """Raised when Langfuse returns a non-2xx response."""


class LangfuseProjectError(RuntimeError):
    """Raised when a project is requested that we hold no credentials for."""


class LangfuseClient:
    """Thin sync client over Langfuse ``/api/public/traces``."""

    def __init__(
        self,
        url: Optional[str] = None,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        project_id: Optional[str] = None,
    ) -> None:
        self.url = (url or settings.LANGFUSE_URL).rstrip("/")
        self.public_key = public_key or settings.LANGFUSE_PUBLIC_KEY
        self.secret_key = secret_key or settings.LANGFUSE_SECRET_KEY
        self.timeout = timeout_seconds or settings.LANGFUSE_TIMEOUT_SECONDS
        # The project these credentials belong to, when the caller already knows
        # it. Purely informational — the keys are what scope the reads.
        self.project_id = project_id

        if not self.url or not self.public_key or not self.secret_key:
            raise LangfuseConfigError(
                "Langfuse is not configured. Set LANGFUSE_URL, "
                "LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY."
            )

    def fetch_projects(self) -> List[Dict[str, Any]]:
        """
        Projects these credentials can read: ``GET /api/public/projects``.

        A project-scoped key returns exactly one entry, which is how we learn a
        project's id and display name without asking anyone to configure them.
        """
        resp = requests.get(
            f"{self.url}/api/public/projects",
            auth=(self.public_key, self.secret_key),
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise LangfuseFetchError(
                f"Langfuse {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json().get("data", []) or []

    def fetch_organization_projects(self) -> List[Dict[str, Any]]:
        """
        Every project in this key's organization: ``GET /api/public/organizations/projects``.

        Requires an organization-scoped key and is the only way to see a project
        no one configured credentials for. The entries carry ``id``, ``name`` and
        timestamps but not the organization itself — the key implies it, and
        Langfuse offers no endpoint that names it — so the caller identifies the
        organization from a project it already knows.
        """
        resp = requests.get(
            f"{self.url}/api/public/organizations/projects",
            auth=(self.public_key, self.secret_key),
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise LangfuseFetchError(
                f"Langfuse {resp.status_code}: {resp.text[:200]}"
            )
        # Envelope key differs from /api/public/projects, which uses "data".
        # Both are accepted so a Langfuse upgrade that settles on one name does
        # not silently empty the project list.
        body = resp.json()
        return body.get("projects") or body.get("data") or []

    def fetch_project_api_keys(self, project_id: str) -> List[Dict[str, Any]]:
        """
        A project's API keys (org-scoped key required), secrets masked.

        Only useful for finding keys to *replace*: Langfuse returns
        ``displaySecretKey``, never the secret, so an existing key cannot be
        adopted — see ``provision_project_key``.
        """
        resp = requests.get(
            f"{self.url}/api/public/projects/{project_id}/apiKeys",
            auth=(self.public_key, self.secret_key),
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise LangfuseFetchError(
                f"Langfuse {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json().get("apiKeys", []) or []

    def create_project_api_key(self, project_id: str, note: str) -> Dict[str, Any]:
        """Mint a project-scoped key pair (org-scoped key required)."""
        resp = requests.post(
            f"{self.url}/api/public/projects/{project_id}/apiKeys",
            json={"note": note},
            auth=(self.public_key, self.secret_key),
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise LangfuseFetchError(
                f"Langfuse {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json() or {}

    def delete_project_api_key(self, project_id: str, api_key_id: str) -> None:
        """Delete a project API key (org-scoped key required)."""
        resp = requests.delete(
            f"{self.url}/api/public/projects/{project_id}/apiKeys/{api_key_id}",
            auth=(self.public_key, self.secret_key),
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise LangfuseFetchError(
                f"Langfuse {resp.status_code}: {resp.text[:200]}"
            )

    def _fetch_page(self, page: int, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        merged = {**params, "page": page, "limit": _PAGE_SIZE}
        # Repeatable tag filter is passed via list of values.
        resp = requests.get(
            f"{self.url}/api/public/traces",
            params=merged,
            auth=(self.public_key, self.secret_key),
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise LangfuseFetchError(
                f"Langfuse {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json().get("data", []) or []

    def _fetch_observation_page(
        self, page: int, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        merged = {**params, "page": page, "limit": _PAGE_SIZE}
        resp = requests.get(
            f"{self.url}/api/public/observations",
            params=merged,
            auth=(self.public_key, self.secret_key),
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise LangfuseFetchError(
                f"Langfuse {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json().get("data", []) or []

    def _fetch_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        resp = requests.get(
            f"{self.url}{path}",
            params=params or {},
            auth=(self.public_key, self.secret_key),
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise LangfuseFetchError(
                f"Langfuse {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json() or {}

    def _iter_paginated(
        self, path: str, params: Optional[Dict[str, Any]] = None, cap: Optional[int] = None
    ) -> Iterable[Dict[str, Any]]:
        """Walk a ``{"data": [...], "meta": {...}}`` endpoint page by page."""
        emitted = 0
        page = 1
        while cap is None or emitted < cap:
            batch = self._fetch_json(
                path, {**(params or {}), "page": page, "limit": _PAGE_SIZE}
            ).get("data") or []
            if not batch:
                return
            for item in batch:
                yield item
                emitted += 1
                if cap is not None and emitted >= cap:
                    return
            page += 1

    def fetch_score_configs(self) -> List[Dict[str, Any]]:
        """
        The project's score definitions: ``GET /api/public/score-configs``.

        These are what the Langfuse UI offers an annotator, so they are the
        authoritative list of annotation names, value ranges and categories.
        Archived configs are dropped — they can't receive new annotations.
        """
        return [
            config
            for config in self._iter_paginated("/api/public/score-configs")
            if not config.get("isArchived")
        ]

    def fetch_annotation_queues(self) -> List[Dict[str, Any]]:
        """The project's annotation queues: ``GET /api/public/annotation-queues``."""
        return list(self._iter_paginated("/api/public/annotation-queues"))

    def fetch_trace_with_observations(self, trace_id: str) -> Dict[str, Any]:
        """Fetch a single trace including nested observations."""
        resp = requests.get(
            f"{self.url}/api/public/traces/{trace_id}",
            auth=(self.public_key, self.secret_key),
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise LangfuseFetchError(
                f"Langfuse {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()

    def iter_traces(
        self,
        *,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        max_traces: Optional[int] = None,
    ) -> Iterable[Dict[str, Any]]:
        """Yield trace dicts across all pages, stopping at ``max_traces``."""
        params: Dict[str, Any] = {}
        if from_timestamp:
            params["fromTimestamp"] = from_timestamp
        if to_timestamp:
            params["toTimestamp"] = to_timestamp
        if name:
            params["name"] = name
        if user_id:
            params["userId"] = user_id
        if session_id:
            params["sessionId"] = session_id
        if environment:
            params["environment"] = environment
        if order_by:
            params["orderBy"] = order_by
        if tags:
            params["tags"] = tags  # requests serialises list as repeated param

        cap = max_traces or settings.LANGFUSE_MAX_TRACES_PER_IMPORT
        emitted = 0
        page = 1
        while emitted < cap:
            batch = self._fetch_page(page, params)
            if not batch:
                return
            for t in batch:
                yield t
                emitted += 1
                if emitted >= cap:
                    return
            page += 1

    def iter_generations(
        self,
        *,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        environment: Optional[str] = None,
        max_observations: Optional[int] = None,
    ) -> Iterable[Dict[str, Any]]:
        """
        Yield GENERATION observations across pages, stopping at ``max_observations``.

        The model a trace ran against lives on its generations, not on the trace
        itself, and ``/api/public/traces`` has no model filter — so anything
        model-aware has to come through here.

        The type is filtered server-side *and* re-checked locally. A Langfuse
        build that ignored an unrecognised query parameter would otherwise hand
        back EMBEDDING observations too, and those carry a model name in the same
        field — which would put the embedding deployment in the import page's
        model dropdown and offer traces that can never convert.
        """
        params: Dict[str, Any] = {"type": _GENERATION_TYPE}
        if from_timestamp:
            params["fromStartTime"] = from_timestamp
        if to_timestamp:
            params["toStartTime"] = to_timestamp
        if environment:
            params["environment"] = environment

        cap = max_observations or settings.LANGFUSE_MAX_OBSERVATIONS_PER_SCAN
        emitted = 0
        page = 1
        while emitted < cap:
            batch = self._fetch_observation_page(page, params)
            if not batch:
                return
            for o in batch:
                if o.get("type") != _GENERATION_TYPE:
                    continue
                yield o
                emitted += 1
                if emitted >= cap:
                    return
            page += 1

    def model_trace_ids(
        self,
        *,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        environment: Optional[str] = None,
        max_observations: Optional[int] = None,
    ) -> Dict[str, Set[str]]:
        """
        Map model name (exactly as traces spell it) -> distinct trace ids.

        The caller gets ids rather than a count because the same deployed model
        can be recorded under several spellings (``Qwen/...`` and
        ``openai/Qwen/...``); merging those needs the ids to avoid
        double-counting a trace that used more than one spelling.
        """
        trace_ids_by_model: Dict[str, Set[str]] = {}
        for obs in self.iter_generations(
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            environment=environment,
            max_observations=max_observations,
        ):
            model = observation_model(obs)
            trace_id = obs.get("traceId")
            if not model or not trace_id:
                continue
            trace_ids_by_model.setdefault(model, set()).add(trace_id)
        return trace_ids_by_model

    def model_trace_counts(
        self,
        *,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        environment: Optional[str] = None,
        max_observations: Optional[int] = None,
    ) -> Dict[str, int]:
        """Map model name -> number of distinct traces that used it."""
        return {
            model: len(ids)
            for model, ids in self.model_trace_ids(
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
                environment=environment,
                max_observations=max_observations,
            ).items()
        }

    def trace_ids_for_model(
        self,
        model: str,
        *,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        environment: Optional[str] = None,
        max_observations: Optional[int] = None,
        newest_first: bool = True,
    ) -> List[str]:
        """
        Distinct trace ids whose generations ran against ``model``, newest first.

        Ordered by the trace's earliest matching generation, so the caller's
        ``order_by=timestamp.{desc,asc}`` still means what it says.
        """
        times = self.model_trace_times(
            model,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            environment=environment,
            max_observations=max_observations,
        )
        return sorted(times, key=lambda t: times[t], reverse=newest_first)

    def generation_trace_times(
        self,
        *,
        model: Optional[str] = None,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        environment: Optional[str] = None,
        max_observations: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Map trace id -> earliest generation start time, for traces that made an
        LLM call — optionally only those that called ``model``.

        With no ``model`` this is "every trace holding a real request to an LLM",
        which is what an ``openai_chat`` import wants to walk: the traces that
        carry only an embedding call or only a health-check span have no assistant
        turn in them at all, and enumerating them here rather than discarding them
        later is the difference between scanning what can convert and scanning
        everything the gateway ever logged.

        Model matching is provider-prefix tolerant, so the canonical gateway name
        the dropdown offers also picks up traces recorded under a prefixed alias
        (``openai/Qwen/...``). Without this, selecting a model would silently
        import only the subset of traces that happened to use one spelling.
        """
        earliest: Dict[str, str] = {}
        for obs in self.iter_generations(
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            environment=environment,
            max_observations=max_observations,
        ):
            if model is not None:
                found = observation_model(obs)
                if not found or not models_match(found, model):
                    continue
            trace_id = obs.get("traceId")
            if not trace_id:
                continue
            start = str(obs.get("startTime") or "")
            if trace_id not in earliest or start < earliest[trace_id]:
                earliest[trace_id] = start

        return earliest

    def model_trace_times(
        self,
        model: str,
        *,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        environment: Optional[str] = None,
        max_observations: Optional[int] = None,
    ) -> Dict[str, str]:
        """Map trace id -> earliest generation start time, for traces using ``model``."""
        return self.generation_trace_times(
            model=model,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            environment=environment,
            max_observations=max_observations,
        )

    def iter_scores(
        self,
        *,
        name: Optional[str] = None,
        source: Optional[str] = None,
        value: Optional[float] = None,
        operator: Optional[str] = None,
        data_type: Optional[str] = None,
        config_id: Optional[str] = None,
        queue_id: Optional[str] = None,
        environment: Optional[str] = None,
        user_id: Optional[str] = None,
        max_scores: Optional[int] = None,
    ) -> Iterable[Dict[str, Any]]:
        """
        Yield scores from ``GET /api/public/v2/scores``.

        Deliberately *not* bounded by the import's timestamp range: an annotation
        is written whenever a human gets round to it, which is normally later than
        the trace it judges. The range is applied to the traces themselves
        instead, so "traces from last week that someone scored today" works.
        """
        params: Dict[str, Any] = {}
        if name:
            params["name"] = name
        if source:
            params["source"] = source
        if data_type:
            params["dataType"] = data_type
        if config_id:
            params["configId"] = config_id
        if queue_id:
            params["queueId"] = queue_id
        if environment:
            params["environment"] = environment
        if user_id:
            params["userId"] = user_id
        # Langfuse only compares numerically, and only when both are given.
        if value is not None and operator:
            params["value"] = value
            params["operator"] = operator

        yield from self._iter_paginated(
            "/api/public/v2/scores",
            params,
            cap=max_scores or settings.LANGFUSE_MAX_SCORES_PER_SCAN,
        )

    def score_trace_times(
        self,
        *,
        name: Optional[str] = None,
        source: Optional[str] = None,
        value: Optional[float] = None,
        operator: Optional[str] = None,
        string_value: Optional[str] = None,
        config_id: Optional[str] = None,
        queue_id: Optional[str] = None,
        environment: Optional[str] = None,
        max_scores: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Map trace id -> newest matching score timestamp, for scored traces.

        ``string_value`` is matched here rather than pushed down because the
        scores API compares numerically only; categorical and boolean scores
        carry their label in ``stringValue``.
        """
        wanted_label = (string_value or "").strip().lower()
        times: Dict[str, str] = {}
        for score in self.iter_scores(
            name=name,
            source=source,
            value=value,
            operator=operator,
            config_id=config_id,
            queue_id=queue_id,
            environment=environment,
            max_scores=max_scores,
        ):
            trace_id = score.get("traceId")
            if not trace_id:
                # Session- and dataset-run-level scores have no trace to import.
                continue
            if wanted_label:
                label = str(score.get("stringValue") or "").strip().lower()
                if label != wanted_label:
                    continue
            stamp = str(score.get("timestamp") or score.get("createdAt") or "")
            if stamp > times.get(trace_id, ""):
                times[trace_id] = stamp
        return times

    def score_names(
        self,
        *,
        source: Optional[str] = None,
        environment: Optional[str] = None,
        max_scores: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Map score name -> what the dropdown needs about it.

        Counts distinct scored traces (not scores), because that is the number of
        training records the filter would keep.
        """
        seen: Dict[str, Dict[str, Any]] = {}
        for score in self.iter_scores(
            source=source, environment=environment, max_scores=max_scores
        ):
            name = str(score.get("name") or "").strip()
            trace_id = score.get("traceId")
            if not name or not trace_id:
                continue
            entry = seen.setdefault(
                name,
                {
                    "data_type": score.get("dataType"),
                    "sources": set(),
                    "labels": set(),
                    "trace_ids": set(),
                },
            )
            entry["trace_ids"].add(trace_id)
            if score.get("source"):
                entry["sources"].add(str(score["source"]))
            label = str(score.get("stringValue") or "").strip()
            if label:
                entry["labels"].add(label)
        return seen

    def queue_trace_times(
        self,
        queue_id: str,
        *,
        status: Optional[str] = None,
        max_items: Optional[int] = None,
    ) -> Dict[str, str]:
        """Map trace id -> queue item creation time, for one annotation queue."""
        times: Dict[str, str] = {}
        params = {"status": status} if status else {}
        for item in self._iter_paginated(
            f"/api/public/annotation-queues/{queue_id}/items",
            params,
            cap=max_items or settings.LANGFUSE_MAX_QUEUE_ITEMS_PER_SCAN,
        ):
            if item.get("objectType") != _QUEUE_TRACE_OBJECT:
                continue
            trace_id = item.get("objectId")
            if not trace_id:
                continue
            stamp = str(item.get("createdAt") or "")
            if stamp > times.get(trace_id, ""):
                times[trace_id] = stamp
        return times


class LangfuseProject(NamedTuple):
    """
    A project the service can offer, plus the credentials that reach it.

    ``public_key``/``secret_key`` are empty for a project discovered through an
    organization key but held by no project key: it can be listed and named, and
    its traces can only be read once a key exists for it. ``readable`` says which
    of the two it is, so the UI can show the difference instead of offering a
    choice that fails on click.
    """

    id: str
    name: str
    organization: str
    organization_id: str
    public_key: str
    secret_key: str
    is_default: bool

    @property
    def readable(self) -> bool:
        return bool(self.public_key and self.secret_key)


# Pairs are written ``publicKey:secretKey`` and separated by commas, semicolons
# or newlines, so the value stays readable in a Helm value or a shell export.
_CREDENTIAL_SEPARATORS = re.compile(r"[,;\s]+")

# Written on keys this service creates for itself, so an operator looking at a
# project's keys in Langfuse can tell where it came from and that deleting it
# only costs a round trip.
_AUTO_KEY_NOTE = "enterprise-ai fine-tuning trace import (auto-managed)"


def parse_key_pairs(raw: str, source: str) -> List[Tuple[str, str]]:
    """
    Parse ``publicKey:secretKey`` pairs out of a configured value.

    Only the keys are configured — what each pair reaches (a project, or an
    organization's projects) is read back from Langfuse, so a typo can't silently
    label one project's traces with another project's name. Malformed entries are
    logged and skipped rather than breaking every project.
    """
    pairs: List[Tuple[str, str]] = []
    for entry in _CREDENTIAL_SEPARATORS.split(raw or ""):
        if not entry:
            continue
        public_key, _, secret_key = entry.partition(":")
        if not public_key or not secret_key:
            logger.warning(
                "Ignoring malformed %s entry: expected publicKey:secretKey, got %r",
                source,
                entry[:12] + "…",
            )
            continue
        pairs.append((public_key, secret_key))
    return pairs


class LangfuseProjectRegistry:
    """
    The projects offered to the caller, resolved from configured credentials.

    Two kinds of credential, because Langfuse's API keys are scoped:

      * project keys — ``LANGFUSE_PUBLIC_KEY``/``LANGFUSE_SECRET_KEY`` for the
        default project, plus every pair in ``LANGFUSE_PROJECT_KEYS``. These can
        read traces, and each one sees exactly one project.
      * organization keys — every pair in ``LANGFUSE_ORG_KEYS``. These cannot
        read a single trace, but they list *all* the projects in an organization,
        which is the only way a project created after deployment shows up here at
        all.

    So the list is the union: projects reached by a key, plus projects merely
    discovered. A discovered project becomes readable when a key is minted for
    it, which ``client()`` does on first use (see ``_provision``).

    Resolution costs a request per configured pair, so it is cached; minted keys
    are held for the process lifetime alongside the cached list.
    """

    def __init__(self, cache_seconds: Optional[int] = None) -> None:
        self._cache_seconds = (
            cache_seconds
            if cache_seconds is not None
            else settings.LANGFUSE_PROJECT_CACHE_SECONDS
        )
        self._lock = threading.Lock()
        # Held across the mint, which is a delete followed by a create: two
        # concurrent first imports of the same project would otherwise have the
        # second one revoke the key the first is about to use.
        self._provision_lock = threading.Lock()
        self._cached: List[LangfuseProject] = []
        self._cached_at: float = 0.0
        # project id -> (public_key, secret_key) minted by us this process. Kept
        # so a refresh of the list does not mint a second key for a project, and
        # so the keys survive the 5-minute cache expiry.
        self._provisioned: Dict[str, Tuple[str, str]] = {}

    def _credentials(self) -> List[Tuple[str, str, bool]]:
        """(public_key, secret_key, is_default), default first."""
        creds: List[Tuple[str, str, bool]] = []
        if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
            creds.append(
                (settings.LANGFUSE_PUBLIC_KEY, settings.LANGFUSE_SECRET_KEY, True)
            )
        seen = {c[0] for c in creds}
        for public_key, secret_key in parse_key_pairs(
            settings.LANGFUSE_PROJECT_KEYS, "LANGFUSE_PROJECT_KEYS"
        ):
            if public_key in seen:
                continue
            seen.add(public_key)
            creds.append((public_key, secret_key, False))
        return creds

    def _org_credentials(self) -> List[Tuple[str, str]]:
        return parse_key_pairs(settings.LANGFUSE_ORG_KEYS, "LANGFUSE_ORG_KEYS")

    def _org_client_for_project(self, project_id: str) -> Optional[LangfuseClient]:
        """
        The organization key that can administer ``project_id``, if one is configured.

        An organization key names no organization of its own, so the question is
        asked the only way Langfuse can answer it: which key lists this project.
        That is also exactly the condition for the key being allowed to create a
        key in it, so there is nothing left to infer.
        """
        for public_key, secret_key in self._org_credentials():
            client = LangfuseClient(public_key=public_key, secret_key=secret_key)
            try:
                entries = client.fetch_organization_projects()
            except LangfuseFetchError as e:
                logger.warning("Organization key unusable: %s", e)
                continue
            if any(str(entry.get("id") or "").strip() == project_id for entry in entries):
                return client
        return None

    def can_provision(self) -> bool:
        """Whether a discovered project can be made readable without an operator."""
        return bool(
            settings.LANGFUSE_AUTO_PROVISION_PROJECT_KEYS and self._org_credentials()
        )

    def _resolve(self) -> List[LangfuseProject]:
        credentials = self._credentials()
        org_credentials = self._org_credentials()
        if not credentials and not org_credentials:
            raise LangfuseConfigError(
                "Langfuse is not configured. Set LANGFUSE_URL, "
                "LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY."
            )

        projects: List[LangfuseProject] = []
        seen_ids: Set[str] = set()
        errors: List[str] = []

        # Project keys first: their /projects response is the only one that names
        # the organization, which then identifies what an organization key sees.
        for public_key, secret_key, is_default in credentials:
            client = LangfuseClient(public_key=public_key, secret_key=secret_key)
            try:
                entries = client.fetch_projects()
            except LangfuseFetchError as e:
                # One rejected key pair must not hide the projects that do work.
                logger.warning("Could not resolve Langfuse project for a key pair: %s", e)
                errors.append(str(e))
                continue
            for entry in entries:
                project_id = str(entry.get("id") or "").strip()
                if not project_id or project_id in seen_ids:
                    continue
                seen_ids.add(project_id)
                organization = entry.get("organization") or {}
                if not isinstance(organization, dict):
                    organization = {}
                projects.append(
                    LangfuseProject(
                        id=project_id,
                        name=str(entry.get("name") or project_id),
                        organization=str(organization.get("name") or ""),
                        organization_id=str(organization.get("id") or ""),
                        public_key=public_key,
                        secret_key=secret_key,
                        is_default=is_default,
                    )
                )

        for public_key, secret_key in org_credentials:
            client = LangfuseClient(public_key=public_key, secret_key=secret_key)
            try:
                entries = client.fetch_organization_projects()
            except LangfuseFetchError as e:
                logger.warning("Could not list projects for an organization key: %s", e)
                errors.append(str(e))
                continue

            # Which organization this key belongs to is not in the response, so
            # take it from a project we already resolved with a project key.
            known = {p.id: p for p in projects}
            owner = next(
                (
                    known[str(entry.get("id") or "").strip()]
                    for entry in entries
                    if str(entry.get("id") or "").strip() in known
                ),
                None,
            )
            if owner is not None:
                organization = owner.organization
                organization_id = owner.organization_id
            else:
                # Nothing to name it after, so it is identified by the key that
                # found it rather than left blank: two unnamed organizations must
                # not merge into one entry in the page's filter, which would mix
                # projects that belong to different organizations. Public keys are
                # not secret, and the tail is enough to tell two apart.
                organization_id = f"key:{public_key}"
                organization = f"Organization of key …{public_key[-6:]}"

            for entry in entries:
                project_id = str(entry.get("id") or "").strip()
                if not project_id or project_id in seen_ids:
                    continue
                seen_ids.add(project_id)
                minted = self._provisioned.get(project_id, ("", ""))
                projects.append(
                    LangfuseProject(
                        id=project_id,
                        name=str(entry.get("name") or project_id),
                        organization=organization,
                        organization_id=organization_id,
                        public_key=minted[0],
                        secret_key=minted[1],
                        is_default=False,
                    )
                )

        if not projects:
            raise LangfuseFetchError(
                errors[0] if errors else "Langfuse returned no projects for the configured keys."
            )
        return projects

    def projects(self, force_refresh: bool = False) -> List[LangfuseProject]:
        with self._lock:
            fresh = time.monotonic() - self._cached_at < self._cache_seconds
            if self._cached and fresh and not force_refresh:
                return list(self._cached)
            projects = self._resolve()
            self._cached = projects
            self._cached_at = time.monotonic()
            return list(projects)

    def organizations(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        The organizations behind the project list, for the import page's filter.

        Derived from the projects rather than fetched: Langfuse exposes no
        "list my organizations" endpoint to a key of either scope, and a project
        already carries the organization it belongs to.
        """
        grouped: Dict[Tuple[str, str], int] = {}
        for project in self.projects(force_refresh=force_refresh):
            key = (project.organization_id, project.organization)
            grouped[key] = grouped.get(key, 0) + 1
        return [
            {"id": org_id, "name": name, "project_count": count}
            for (org_id, name), count in grouped.items()
        ]

    def default_project_id(self) -> Optional[str]:
        for project in self.projects():
            if project.is_default:
                return project.id
        return None

    def _provision(self, project: LangfuseProject) -> LangfuseProject:
        """
        Mint a project key for a discovered project, using its organization key.

        Langfuse never returns a secret key twice, so an existing key cannot be
        adopted — ours is replaced rather than added to, which keeps one
        auto-managed key per project however many times this runs.
        """
        with self._provision_lock:
            return self._provision_locked(project)

    def _provision_locked(self, project: LangfuseProject) -> LangfuseProject:
        minted = self._provisioned.get(project.id)
        if minted:
            # Another request minted it while this one waited.
            return project._replace(public_key=minted[0], secret_key=minted[1])

        client = self._org_client_for_project(project.id)
        if client is None:
            raise LangfuseProjectError(
                f"No credentials for Langfuse project '{project.id}' and no "
                f"organization key that can create them. Add the project's key "
                f"pair to LANGFUSE_PROJECT_KEYS, or an organization key to "
                f"LANGFUSE_ORG_KEYS."
            )

        try:
            for existing in client.fetch_project_api_keys(project.id):
                if (existing.get("note") or "") == _AUTO_KEY_NOTE and existing.get("id"):
                    client.delete_project_api_key(project.id, str(existing["id"]))
            created = client.create_project_api_key(project.id, _AUTO_KEY_NOTE)
        except LangfuseFetchError as e:
            raise LangfuseProjectError(
                f"Could not create a Langfuse API key for project "
                f"'{project.id}': {e}"
            ) from e

        public_key = str(created.get("publicKey") or "")
        secret_key = str(created.get("secretKey") or "")
        if not public_key or not secret_key:
            raise LangfuseProjectError(
                f"Langfuse returned no usable key pair for project '{project.id}'."
            )

        logger.info(
            "Created an auto-managed Langfuse API key for project %s", project.id
        )
        provisioned = project._replace(public_key=public_key, secret_key=secret_key)
        with self._lock:
            self._provisioned[project.id] = (public_key, secret_key)
            self._cached = [
                provisioned if p.id == project.id else p for p in self._cached
            ]
        return provisioned

    def client(self, project_id: Optional[str] = None) -> LangfuseClient:
        """
        A client scoped to ``project_id``, or the default project when omitted.

        Omitting the project keeps older callers (and API clients written before
        project selection existed) working against the default credentials
        without a lookup.
        """
        if not project_id:
            return LangfuseClient()

        project = self._find(project_id)
        if project is None:
            # A key pair may have been added, or a project created, since the
            # last resolution.
            project = self._find(project_id, force_refresh=True)
        if project is None:
            available = ", ".join(p.id for p in self.projects()) or "none"
            raise LangfuseProjectError(
                f"No Langfuse credentials for project '{project_id}'. "
                f"Available: {available}."
            )

        if not project.readable:
            if not settings.LANGFUSE_AUTO_PROVISION_PROJECT_KEYS:
                raise LangfuseProjectError(
                    f"No credentials for Langfuse project '{project.id}'. Add its "
                    f"key pair to LANGFUSE_PROJECT_KEYS, or enable "
                    f"LANGFUSE_AUTO_PROVISION_PROJECT_KEYS to create one from the "
                    f"organization key."
                )
            project = self._provision(project)

        return LangfuseClient(
            public_key=project.public_key,
            secret_key=project.secret_key,
            project_id=project.id,
        )

    def _find(self, project_id: str, force_refresh: bool = False) -> Optional[LangfuseProject]:
        return next(
            (p for p in self.projects(force_refresh=force_refresh) if p.id == project_id),
            None,
        )


# Module-level so the resolution cache is shared across requests.
project_registry = LangfuseProjectRegistry()


def observation_model(observation: Dict[str, Any]) -> Optional[str]:
    """Pull the model name off a GENERATION observation, whichever key holds it."""
    for key in _MODEL_KEYS:
        value = observation.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def apply_system_mode(
    msgs: List[Dict[str, Any]], mode: str, text: str = ""
) -> List[Dict[str, Any]]:
    """Rewrite the system turns of one exchange according to ``mode``.

    Every system turn goes, not just the first: a request can carry several, and
    leaving one behind would keep exactly the grounding the caller asked to lose.
    """
    if mode == SYSTEM_KEEP:
        return msgs
    body = [m for m in msgs if m.get("role") != "system"]
    if mode == SYSTEM_REPLACE and text.strip():
        return [{"role": "system", "content": text.strip()}] + body
    return body


def convert_trace(
    trace: Dict[str, Any],
    *,
    fmt: str,
    fields: Optional[List[str]] = None,
    system_mode: str = SYSTEM_KEEP,
    system_text: str = "",
) -> Optional[Dict[str, Any]]:
    """Return a converted record or ``None`` if the trace should be skipped.

    ``system_mode``/``system_text`` only apply to ``openai_chat``; the other two
    formats are defined as passing Langfuse's own shape through untouched.
    """
    if fmt not in _ALLOWED_FORMATS:
        raise ValueError(f"Unsupported format: {fmt}")
    if system_mode not in _ALLOWED_SYSTEM_MODES:
        raise ValueError(f"Unsupported system_mode: {system_mode}")

    if fmt == FORMAT_RAW:
        return trace

    if fmt == FORMAT_CUSTOM:
        picked = {k: trace.get(k) for k in (fields or [])}
        return picked or None

    # OpenAI chat: skip anything without a usable assistant answer.
    #
    # The prompt and the answer are taken from the same GENERATION observation
    # whenever there is one, rather than from the trace envelope. A trace is a
    # whole gateway request and can contain more than one kind of call - a RAG
    # turn embeds the query and *then* asks the model - in which case
    # ``trace.input`` is whatever the first call received, i.e. the batch of texts
    # handed to the embedding model. Pairing that with the LLM's reply would
    # fabricate an exchange that never happened. Reading both halves off the
    # generation keeps the record to what was actually sent to the LLM.
    #
    # ``trace.input``/``trace.output`` remain the fallback: a trace fetched from
    # the list endpoint carries no observations, and integrations that write the
    # exchange onto the trace itself have nothing else to offer.
    generation = _latest_generation(trace)
    if generation is not None:
        input_msgs = generation.get("input")
        output = generation.get("output")
        if not output or input_msgs in (None, "", [], {}):
            # A generation missing one half of the exchange - a streamed reply
            # Langfuse never saw completed, say. Fall back rather than drop it.
            input_msgs = input_msgs or trace.get("input")
            output = output or trace.get("output")
    else:
        input_msgs = trace.get("input")
        output = trace.get("output")

    if not output:
        return None

    if isinstance(input_msgs, list):
        msgs = [m for m in input_msgs if isinstance(m, dict) and m.get("content")]
    elif isinstance(input_msgs, dict) and input_msgs.get("content"):
        msgs = [{"role": input_msgs.get("role", "user"), "content": input_msgs["content"]}]
    elif isinstance(input_msgs, str) and input_msgs.strip():
        msgs = [{"role": "user", "content": input_msgs}]
    else:
        return None

    msgs = apply_system_mode(msgs, system_mode, system_text)
    # No question, nothing to learn: an exchange whose prompt is system turns
    # only teaches a fixed reply to a fixed instruction. Dropping the system turn
    # is one way to arrive here, a request that sent nothing else is the other.
    if not any(m.get("role") != "system" for m in msgs):
        return None

    if isinstance(output, dict) and output.get("content"):
        assistant = {"role": output.get("role", "assistant"), "content": output["content"]}
    elif isinstance(output, str) and output.strip():
        assistant = {"role": "assistant", "content": output}
    else:
        return None

    msgs.append(assistant)
    return {"messages": msgs}


def _latest_generation(trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    The last nested GENERATION observation that produced an output, or None.

    "Last" because a trace with several generations - a retry, or an agent loop -
    ends on the answer the caller actually received. Observations of any other
    type are ignored outright: an EMBEDDING carries a model name and an ``input``
    shaped exactly like a chat history, so anything that matched loosely here
    would happily turn a batch of embedded documents into a training example.
    """
    obs = trace.get("observations")
    if not isinstance(obs, list):
        return None
    latest = None
    for o in obs:
        if isinstance(o, dict) and o.get("type") == _GENERATION_TYPE and o.get("output"):
            latest = o
    return latest


def stream_jsonl(records: Iterable[Dict[str, Any]]) -> Iterable[bytes]:
    """Yield newline-terminated JSON bytes for each record."""
    for r in records:
        if r is None:
            continue
        yield (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8")

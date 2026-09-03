"""
Node capacity, so the UI can say whether another model will actually fit.

Kubernetes admits a pod by comparing its **requests** against a node's
**allocatable**, not against live utilisation, so that is what this measures:
allocatable minus the requests already reserved on each node. It is also the
only thing measurable here — the cluster has no metrics-server, so actual CPU
and memory in use are simply not available (``live_usage_available`` says so in
the response rather than leaving the caller to guess).

A model whose requests do not fit anywhere is rejected before the Helm Job is
created. Note that a pod with *no* requests always "fits" as far as the
scheduler is concerned and will be placed on a full node, which is why the
deployment path sets cpu/memory at all — see ``routers/deployments.py``.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .k8s import KubeApiError, kube_client
from .observability import get_logger

logger = get_logger(__name__)

# Binary and decimal suffixes a Kubernetes quantity may carry.
_BINARY_SUFFIXES = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4, "Pi": 1024**5, "Ei": 1024**6}
_DECIMAL_SUFFIXES = {"n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15, "E": 1e18}

_QUANTITY = re.compile(r"^\s*([+-]?[0-9.]+(?:[eE][+-]?[0-9]+)?)\s*([A-Za-z]*)\s*$")

GIB = 1024**3


class QuantityError(ValueError):
    """A resource quantity could not be understood."""


def parse_quantity(value: Any) -> float:
    """
    Parse a Kubernetes quantity into a plain number of base units.

    ``"64"`` → 64, ``"500m"`` → 0.5, ``"128Gi"`` → 137438953472. Note that ``m``
    means milli and ``M`` means mega: the case matters and mixing them up is a
    factor of a billion, so both are handled explicitly rather than by
    lower-casing.
    """
    if value is None:
        raise QuantityError("missing quantity")
    if isinstance(value, (int, float)):
        return float(value)

    match = _QUANTITY.match(str(value))
    if not match:
        raise QuantityError(f"not a valid quantity: {value!r}")
    number, suffix = match.group(1), match.group(2)

    try:
        amount = float(number)
    except ValueError as exc:
        raise QuantityError(f"not a valid quantity: {value!r}") from exc

    if not suffix:
        return amount
    if suffix in _BINARY_SUFFIXES:
        return amount * _BINARY_SUFFIXES[suffix]
    if suffix in _DECIMAL_SUFFIXES:
        return amount * _DECIMAL_SUFFIXES[suffix]
    raise QuantityError(f"unknown unit {suffix!r} in {value!r}")


def parse_cpu_millis(value: Any) -> int:
    """CPU quantity as whole millicores."""
    return int(round(parse_quantity(value) * 1000))


def parse_memory_bytes(value: Any) -> int:
    """Memory quantity as whole bytes."""
    return int(parse_quantity(value))


def format_cpu(millis: int) -> str:
    """Cores, for a message a person reads. 87300 -> '87.3'."""
    return f"{millis / 1000:.1f}".rstrip("0").rstrip(".")


def format_memory(num_bytes: int) -> str:
    """GiB, for a message a person reads."""
    return f"{num_bytes / GIB:.1f}".rstrip("0").rstrip(".") + "Gi"


def _container_requests(pod: Dict[str, Any]) -> Tuple[int, int]:
    """
    CPU millicores and memory bytes a pod reserves on its node.

    Init containers run one at a time and before the app containers, so a pod's
    reservation is the greater of "all app containers" and "the largest init
    container" — the same rule the scheduler uses. Sidecar init containers
    (restartPolicy: Always) run alongside the app containers, so they add.
    """
    spec = pod.get("spec") or {}
    cpu = mem = 0
    for container in spec.get("containers") or []:
        requests = ((container.get("resources") or {}).get("requests")) or {}
        cpu += _safe_cpu(requests.get("cpu"))
        mem += _safe_mem(requests.get("memory"))

    init_cpu = init_mem = 0
    for container in spec.get("initContainers") or []:
        requests = ((container.get("resources") or {}).get("requests")) or {}
        c, m = _safe_cpu(requests.get("cpu")), _safe_mem(requests.get("memory"))
        if container.get("restartPolicy") == "Always":
            cpu += c
            mem += m
        else:
            init_cpu = max(init_cpu, c)
            init_mem = max(init_mem, m)

    return max(cpu, init_cpu), max(mem, init_mem)


def _safe_cpu(value: Any) -> int:
    try:
        return parse_cpu_millis(value) if value is not None else 0
    except QuantityError:
        return 0


def _safe_mem(value: Any) -> int:
    try:
        return parse_memory_bytes(value) if value is not None else 0
    except QuantityError:
        return 0


def _node_is_schedulable(node: Dict[str, Any]) -> Tuple[bool, str]:
    """Whether a new pod could land here, and why not when it could not."""
    if (node.get("spec") or {}).get("unschedulable"):
        return False, "cordoned"
    conditions = ((node.get("status") or {}).get("conditions")) or []
    ready = next((c for c in conditions if c.get("type") == "Ready"), None)
    if not ready or ready.get("status") != "True":
        return False, "not ready"
    for condition in conditions:
        # Pressure conditions do not stop scheduling outright, but a node under
        # memory or disk pressure is not somewhere to put a model.
        if condition.get("type") in ("MemoryPressure", "DiskPressure") and condition.get("status") == "True":
            return False, condition["type"]
    return True, ""


class CapacityUnavailable(Exception):
    """Capacity could not be read — usually the ClusterRole is not applied."""


async def snapshot() -> Dict[str, Any]:
    """
    Per-node allocatable, committed and free resources.

    Raises ``CapacityUnavailable`` rather than returning zeros when the
    apiserver cannot be read, so a caller never mistakes "we do not know" for
    "there is no room".
    """
    if not kube_client.available:
        raise CapacityUnavailable("not running inside a Kubernetes cluster")

    try:
        nodes = await kube_client.list_nodes()
        pods = await kube_client.list_pods_all_namespaces()
    except KubeApiError as exc:
        if exc.status == 403:
            raise CapacityUnavailable(
                "this service is not allowed to read nodes or pods across "
                "namespaces (apply the ClusterRole from the API chart)"
            ) from exc
        raise CapacityUnavailable(exc.message) from exc

    committed: Dict[str, Tuple[int, int, int]] = {}
    for pod in pods:
        node_name = (pod.get("spec") or {}).get("nodeName")
        if not node_name:
            continue  # still pending: not holding anything yet
        cpu, mem = _container_requests(pod)
        prev = committed.get(node_name, (0, 0, 0))
        committed[node_name] = (prev[0] + cpu, prev[1] + mem, prev[2] + 1)

    node_rows: List[Dict[str, Any]] = []
    for node in nodes:
        name = (node.get("metadata") or {}).get("name", "")
        allocatable = ((node.get("status") or {}).get("allocatable")) or {}
        alloc_cpu = _safe_cpu(allocatable.get("cpu"))
        alloc_mem = _safe_mem(allocatable.get("memory"))
        alloc_pods = int(parse_quantity(allocatable.get("pods") or 0))
        used_cpu, used_mem, used_pods = committed.get(name, (0, 0, 0))
        schedulable, reason = _node_is_schedulable(node)

        node_rows.append({
            "name": name,
            "schedulable": schedulable,
            "unschedulable_reason": reason or None,
            "allocatable": {"cpu_millis": alloc_cpu, "memory_bytes": alloc_mem, "pods": alloc_pods},
            "committed": {"cpu_millis": used_cpu, "memory_bytes": used_mem, "pods": used_pods},
            "free": {
                "cpu_millis": max(0, alloc_cpu - used_cpu),
                "memory_bytes": max(0, alloc_mem - used_mem),
                "pods": max(0, alloc_pods - used_pods),
            },
        })

    node_rows.sort(key=lambda row: row["free"]["cpu_millis"], reverse=True)
    usable = [row for row in node_rows if row["schedulable"]]

    return {
        "nodes": node_rows,
        "totals": {
            "allocatable": {
                "cpu_millis": sum(r["allocatable"]["cpu_millis"] for r in usable),
                "memory_bytes": sum(r["allocatable"]["memory_bytes"] for r in usable),
            },
            "committed": {
                "cpu_millis": sum(r["committed"]["cpu_millis"] for r in usable),
                "memory_bytes": sum(r["committed"]["memory_bytes"] for r in usable),
            },
            "free": {
                "cpu_millis": sum(r["free"]["cpu_millis"] for r in usable),
                "memory_bytes": sum(r["free"]["memory_bytes"] for r in usable),
            },
        },
        # No metrics-server in this cluster, so these numbers are reservations
        # rather than measurements. Say so instead of implying otherwise.
        "live_usage_available": False,
        "basis": "requests",
    }


def largest_free(snap: Dict[str, Any]) -> Dict[str, int]:
    """Free resources on the roomiest schedulable node."""
    usable = [row for row in snap["nodes"] if row["schedulable"]]
    if not usable:
        return {"cpu_millis": 0, "memory_bytes": 0, "pods": 0}
    return max(usable, key=lambda row: (row["free"]["cpu_millis"], row["free"]["memory_bytes"]))["free"]


def check_fit(snap: Dict[str, Any], cpu_millis: int, memory_bytes: int) -> Dict[str, Any]:
    """
    Whether one pod asking for this much could be placed.

    A pod runs on a single node, so the test is against the roomiest node rather
    than the cluster total: 200 free cores spread over two nodes will not host a
    model that wants 150 on one.
    """
    fits_on: List[str] = []
    for row in snap["nodes"]:
        if not row["schedulable"]:
            continue
        free = row["free"]
        if free["cpu_millis"] >= cpu_millis and free["memory_bytes"] >= memory_bytes and free["pods"] >= 1:
            fits_on.append(row["name"])

    headroom = largest_free(snap)
    shortfall = {
        "cpu_millis": max(0, cpu_millis - headroom["cpu_millis"]),
        "memory_bytes": max(0, memory_bytes - headroom["memory_bytes"]),
    }
    return {
        "fits": bool(fits_on),
        "fits_on": fits_on,
        "requested": {"cpu_millis": cpu_millis, "memory_bytes": memory_bytes},
        "largest_free": headroom,
        "shortfall": shortfall,
    }


def describe_shortfall(fit: Dict[str, Any]) -> str:
    """One sentence naming what is missing, for an error a person will read."""
    parts = []
    if fit["shortfall"]["cpu_millis"]:
        parts.append(
            f"{format_cpu(fit['shortfall']['cpu_millis'])} more cores "
            f"({format_cpu(fit['requested']['cpu_millis'])} requested, "
            f"{format_cpu(fit['largest_free']['cpu_millis'])} free)"
        )
    if fit["shortfall"]["memory_bytes"]:
        parts.append(
            f"{format_memory(fit['shortfall']['memory_bytes'])} more memory "
            f"({format_memory(fit['requested']['memory_bytes'])} requested, "
            f"{format_memory(fit['largest_free']['memory_bytes'])} free)"
        )
    if not parts:
        return "no node has a free pod slot"
    return " and ".join(parts)

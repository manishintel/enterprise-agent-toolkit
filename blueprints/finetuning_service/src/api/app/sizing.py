"""
A starting CPU and memory request for a model about to be served.

There is no way to know what a model needs without running it, so this is an
opening bid the user can change in the deploy dialog, not a promise. It is
derived from the parameter count in the model id, which is the only sizing
information available at this point.

Two things make it a bid rather than an answer:

* the vllm chart sets requests and limits to the *same* value, so the number is
  both the guarantee and the ceiling — too low is an OOM kill, too high will not
  schedule;
* a large part of the footprint is the KV cache, and on CPU that is a *fixed
  reservation* (``VLLM_CPU_KVCACHE_SPACE``, 40GiB in the packaged chart) rather
  than something derived from the model — so it dwarfs the weights of a small
  model and has to be added on top of them, not assumed to scale with them.
"""

import math
import re
from typing import Any, Dict, List, Optional

from .capacity import GIB, format_cpu, format_memory
from .config import get_settings
from .observability import get_logger

logger = get_logger(__name__)
settings = get_settings()

# "Llama-3.2-3B-Instruct" -> 3, "Mixtral-8x7B" -> 56, "Qwen2.5-Coder-14B" -> 14.
_PARAM_PATTERN = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*[bB](?![a-z0-9])")
_MOE_PATTERN = re.compile(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*[bB]", re.IGNORECASE)


def parameters_billions(model_id: str) -> Optional[float]:
    """
    Parameter count in billions, read off the model id, or None if it says nothing.

    Deliberately conservative: a version number like ``Llama-3.2`` must not be
    read as 3.2B, so a digit run only counts when it is followed by ``b`` and is
    not part of a longer word.
    """
    if not model_id:
        return None

    name = model_id.split("/")[-1]

    moe = _MOE_PATTERN.search(name)
    if moe:
        # 8x7B is 8 experts of 7B: all of it has to be resident to serve.
        return float(moe.group(1)) * float(moe.group(2))

    matches = _PARAM_PATTERN.findall(name)
    if not matches:
        return None
    # Largest wins: "Llama-3-8B" has one candidate, but a name carrying both a
    # version and a size should size on the size.
    return max(float(m) for m in matches)


def recommend(
    model_id: str,
    *,
    max_cpu_millis: Optional[int] = None,
    max_memory_bytes: Optional[int] = None,
    kv_cache_space_gib: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Suggest a request for this model, capped so one deploy cannot take a whole node.

    ``kv_cache_space_gib`` is ``VLLM_CPU_KVCACHE_SPACE`` from the chart, and it
    has to be added on rather than assumed away: vLLM reserves that much for the
    KV cache regardless of how small the model is, so a 3B model under a 40GiB
    cache setting needs upwards of 50GiB and a request sized only from the
    parameter count is an OOM kill during load.

    ``max_cpu_millis``/``max_memory_bytes`` are the ceiling to clamp to — pass the
    node-fraction budget. When the derived size is clamped, ``notes`` says so:
    the number is then small enough to schedule but possibly too small to run,
    and the user needs to know which of the two problems they have.
    """
    config = settings.deployment
    notes: List[str] = []
    kv_gib = config.default_kv_cache_space_gib if kv_cache_space_gib is None else kv_cache_space_gib

    params = parameters_billions(model_id)
    if params is None:
        cpu_millis = config.default_cpu_cores * 1000
        memory_bytes = (config.default_memory_gib + kv_gib) * GIB
        notes.append(
            f"Could not read a parameter count from '{model_id}', so this is the "
            f"installation default of {config.default_memory_gib}Gi plus {kv_gib}Gi of KV "
            f"cache. Check it against the model's own requirements."
        )
    else:
        cpu_millis = int(round(params * config.cpu_cores_per_billion_params)) * 1000
        weights_gib = round(params * config.memory_gib_per_billion_params)
        memory_bytes = int(weights_gib + kv_gib + config.memory_overhead_gib) * GIB
        notes.append(
            f"~{params:g}B parameters: {weights_gib}Gi of weights and working memory "
            f"({config.memory_gib_per_billion_params}Gi per billion), {kv_gib}Gi of KV cache "
            f"(VLLM_CPU_KVCACHE_SPACE) and {config.memory_overhead_gib}Gi overhead. "
            f"Lowering the KV cache lowers this."
        )

    # Floor from the same function the form and the request validator use, so the
    # opening value can never be below the minimum the user is then held to.
    bounds = request_bounds(model_id, kv_cache_space_gib=kv_gib)
    floor_cpu, floor_mem = bounds["cpu_min_millis"], bounds["memory_min_bytes"]
    if cpu_millis < floor_cpu or memory_bytes < floor_mem:
        cpu_millis = max(cpu_millis, floor_cpu)
        memory_bytes = max(memory_bytes, floor_mem)
        notes.append(
            f"Raised to this model's minimum of {bounds['cpu_min']} cores / {bounds['memory_min']}."
        )

    ceiling_cpu, ceiling_mem = config.max_cpu_cores * 1000, config.max_memory_gib * GIB
    if max_cpu_millis is not None:
        ceiling_cpu = min(ceiling_cpu, max_cpu_millis)
    if max_memory_bytes is not None:
        ceiling_mem = min(ceiling_mem, max_memory_bytes)

    if cpu_millis > ceiling_cpu or memory_bytes > ceiling_mem:
        notes.append(
            f"Capped at {format_cpu(ceiling_cpu)} cores / {format_memory(ceiling_mem)} so one "
            f"deployment cannot claim a whole node — this may be less than the model needs."
        )
        cpu_millis = min(cpu_millis, ceiling_cpu)
        memory_bytes = min(memory_bytes, ceiling_mem)

    # The node-fraction ceiling can land under the model's own floor. That is not
    # a suggestion the user can act on by editing the form, so name it as the
    # hardware problem it is rather than leaving a number that cannot work.
    if cpu_millis < floor_cpu or memory_bytes < floor_mem:
        notes.append(
            f"This is below the {bounds['cpu_min']}-core / {bounds['memory_min']} minimum this model "
            f"needs, because that minimum is more than one deployment's share of the roomiest node. "
            f"Lower the KV cache to reduce the minimum, or serve this model on a larger node."
        )

    return {
        "cpu": format_cpu(cpu_millis),
        "memory": format_memory(memory_bytes),
        "cpu_millis": cpu_millis,
        "memory_bytes": memory_bytes,
        "parameters_billions": params,
        "notes": notes,
    }


def request_bounds(
    model_id: str,
    *,
    kv_cache_space_gib: Optional[int] = None,
    node_allocatable: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    The range a CPU and memory request for this model may legally take.

    The floor comes from the *model*, the ceiling from the *hardware*. Both were
    previously fixed numbers, which is what let a 3B model be configured with 1
    core and 1GiB -- values the form accepted and the pod then died on.

    Memory floor (a hard requirement)::

        weights + KV cache + runtime
        = params_B * min_memory_gib_per_billion_params   (2GiB per billion at bf16)
        + VLLM_CPU_KVCACHE_SPACE                          (reserved up front, whatever the model size)
        + min_memory_overhead_gib

    The KV cache term is why the floor moves when the caller changes that field:
    vLLM reserves it before loading anything, so it is part of the minimum rather
    than headroom. Below this total the container cannot hold the weights and the
    pod is OOM-killed during load.

    CPU floor (a usability limit, not a physical one)::

        params_B * min_cpu_cores_per_billion_params       (1 core per billion)

    Too few cores does not fail, it just serves unusably slowly, so this stops a
    deployment being configured into uselessness rather than preventing a crash.

    When the parameter count cannot be read from the model id, both fall back to
    the installation's flat minimums and ``basis`` says so.

    The ceiling is the roomiest schedulable node's allocatable, because a pod runs
    on one node and cannot straddle two. ``max_node_fraction`` deliberately does
    *not* apply here -- it shapes the recommendation, but forbidding a deliberate
    request for most of an idle machine only invites the force flag.
    """
    config = settings.deployment
    kv_gib = config.default_kv_cache_space_gib if kv_cache_space_gib is None else kv_cache_space_gib
    params = parameters_billions(model_id)

    if params is None:
        # The KV cache is reserved whether or not the model size is known, so it
        # belongs in the floor even here: a flat 8Gi minimum under a 40Gi cache
        # setting is a number no pod can start on.
        min_cores = config.min_cpu_cores
        min_gib = max(config.min_memory_gib, kv_gib + config.min_memory_overhead_gib)
        basis = "installation-default"
        cpu_formula = f"{min_cores} cores (model size unknown)"
        memory_formula = (
            f"{kv_gib}Gi KV cache + {config.min_memory_overhead_gib}Gi runtime = {min_gib}Gi "
            f"(model size unknown, so the weights are not counted — check the model's own "
            f"requirements)"
        )
    else:
        weights_gib = params * config.min_memory_gib_per_billion_params
        derived_cores = params * config.min_cpu_cores_per_billion_params
        derived_gib = weights_gib + kv_gib + config.min_memory_overhead_gib
        basis = "model-derived"

        # Rounded up, not to nearest: a sub-1B model deriving "0 cores" is not a
        # minimum, and half a core is not a request the chart can express.
        derived_cores_int = max(1, math.ceil(derived_cores))
        derived_gib_int = math.ceil(derived_gib)
        min_cores = max(derived_cores_int, config.min_cpu_cores)
        min_gib = max(derived_gib_int, config.min_memory_gib)

        per_billion = config.min_cpu_cores_per_billion_params
        cpu_formula = (
            f"{params:g}B x {per_billion:g} {'core' if per_billion == 1 else 'cores'} per billion "
            f"= {derived_cores_int} {'core' if derived_cores_int == 1 else 'cores'}"
        )
        memory_formula = (
            f"{params:g}B x {config.min_memory_gib_per_billion_params:g}Gi weights "
            f"({weights_gib:.0f}Gi) + {kv_gib}Gi KV cache + "
            f"{config.min_memory_overhead_gib}Gi runtime = {derived_gib_int}Gi"
        )
        # A very small model can derive less than the installation's own floor. Say
        # that in the formula rather than printing arithmetic that does not add up
        # to the number the field is actually held to.
        if min_cores > derived_cores_int:
            cpu_formula += f", raised to the {config.min_cpu_cores}-core installation minimum"
        if min_gib > derived_gib_int:
            memory_formula += f", raised to the {config.min_memory_gib}Gi installation minimum"

    min_cpu_millis = min_cores * 1000
    min_memory_bytes = min_gib * GIB

    max_cpu_millis = config.max_cpu_cores * 1000
    max_memory_bytes = config.max_memory_gib * GIB
    node_name = None
    if node_allocatable:
        node_name = node_allocatable.get("name")
        if node_allocatable.get("cpu_millis"):
            max_cpu_millis = min(max_cpu_millis, int(node_allocatable["cpu_millis"]))
        if node_allocatable.get("memory_bytes"):
            max_memory_bytes = min(max_memory_bytes, int(node_allocatable["memory_bytes"]))

    # A model too big for the hardware would otherwise produce min > max, which is
    # an unfillable form. Report the collision instead of silently inverting.
    exceeds_hardware = min_cpu_millis > max_cpu_millis or min_memory_bytes > max_memory_bytes

    return {
        "cpu_min_millis": min_cpu_millis,
        "cpu_max_millis": max(max_cpu_millis, min_cpu_millis),
        "memory_min_bytes": min_memory_bytes,
        "memory_max_bytes": max(max_memory_bytes, min_memory_bytes),
        "cpu_min": format_cpu(min_cpu_millis),
        "memory_min": format_memory(min_memory_bytes),
        "parameters_billions": params,
        "basis": basis,
        "cpu_formula": cpu_formula,
        "memory_formula": memory_formula,
        "ceiling_node": node_name,
        "exceeds_hardware": exceeds_hardware,
    }


def node_budget(free_or_allocatable: Dict[str, int], fraction: float) -> Dict[str, int]:
    """The share of a node a single deployment may claim."""
    return {
        "cpu_millis": int(free_or_allocatable.get("cpu_millis", 0) * fraction),
        "memory_bytes": int(free_or_allocatable.get("memory_bytes", 0) * fraction),
    }

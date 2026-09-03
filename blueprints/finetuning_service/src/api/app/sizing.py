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
* most of a served model's memory is the KV cache, which scales with
  ``max_model_len × max_num_seqs`` and not with the parameter count, so a long
  context can outweigh the weights.
"""

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
) -> Dict[str, Any]:
    """
    Suggest a request for this model, capped so one deploy cannot take a whole node.

    ``max_cpu_millis``/``max_memory_bytes`` are the ceiling to clamp to — pass the
    node-fraction budget. When the derived size is clamped, ``notes`` says so:
    the number is then small enough to schedule but possibly too small to run,
    and the user needs to know which of the two problems they have.
    """
    config = settings.deployment
    notes: List[str] = []

    params = parameters_billions(model_id)
    if params is None:
        cpu_millis = config.default_cpu_cores * 1000
        memory_bytes = config.default_memory_gib * GIB
        notes.append(
            f"Could not read a parameter count from '{model_id}', so this is the "
            f"installation default. Check it against the model's own requirements."
        )
    else:
        cpu_millis = int(round(params * config.cpu_cores_per_billion_params)) * 1000
        memory_bytes = int(
            round(params * config.memory_gib_per_billion_params + config.memory_overhead_gib)
        ) * GIB
        notes.append(
            f"Derived from ~{params:g}B parameters at "
            f"{config.cpu_cores_per_billion_params} cores and "
            f"{config.memory_gib_per_billion_params}Gi per billion, plus "
            f"{config.memory_overhead_gib}Gi overhead."
        )

    floor_cpu, floor_mem = config.min_cpu_cores * 1000, config.min_memory_gib * GIB
    if cpu_millis < floor_cpu or memory_bytes < floor_mem:
        cpu_millis = max(cpu_millis, floor_cpu)
        memory_bytes = max(memory_bytes, floor_mem)
        notes.append(f"Raised to the {config.min_cpu_cores}-core / {config.min_memory_gib}Gi minimum.")

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

    return {
        "cpu": format_cpu(cpu_millis),
        "memory": format_memory(memory_bytes),
        "cpu_millis": cpu_millis,
        "memory_bytes": memory_bytes,
        "parameters_billions": params,
        "notes": notes,
    }


def node_budget(free_or_allocatable: Dict[str, int], fraction: float) -> Dict[str, int]:
    """The share of a node a single deployment may claim."""
    return {
        "cpu_millis": int(free_or_allocatable.get("cpu_millis", 0) * fraction),
        "memory_bytes": int(free_or_allocatable.get("memory_bytes", 0) * fraction),
    }

"""
The serving settings a fine-tuned deployment will actually use.

Read out of the packaged chart rather than restated here, because two of them
were already stale-by-construction once: ``max_model_len`` and ``resources`` sit
at the top level of ``xeon-values.yaml`` where **nothing reads them** — values
files are not Helm-templated and the deployment template takes its vLLM flags
from ``extraCmdArgs``. Anything hard-coded in this service or in the UI would
drift the same way, so the defaults shown to a user come from the chart the
install will run with.

A fine-tuned model is not listed in ``modelConfigs``, so it falls through to
``defaultModelConfigs`` — that is the block to read.

Two different mechanisms carry these settings, which is why overriding them is
not uniform:

* vLLM CLI flags, from ``extraCmdArgs``. Overridden by appending to
  ``finetune.extraCmdArgs``; argparse keeps the last occurrence of a flag.
* environment, from ``configMapValues``. ``VLLM_CPU_KVCACHE_SPACE`` lives here
  and is the KV cache size in GiB — on CPU it is the single largest term in the
  pod's memory footprint, and it is set with ``--set`` on that map key.
"""

from typing import Any, Dict, List, Optional, Union

import yaml

from .observability import get_logger

logger = get_logger(__name__)

# What a deployment falls back to when the chart cannot be read. These mirror
# core/helm-charts/vllm/xeon-values.yaml at the time of writing and exist only so
# the dialog has something to show; `source` says which is in force.
FALLBACK_DEFAULTS: Dict[str, Any] = {
    "max_model_len": None,
    "max_num_seqs": 256,
    "max_num_batched_tokens": 2048,
    "dtype": "bfloat16",
    "block_size": 128,
    "kv_cache_space_gib": 40,
}

# Ranges the API enforces, published so the UI does not carry its own copy.
LIMITS: Dict[str, Dict[str, Any]] = {
    "max_model_len": {"min": 256, "max": 1048576, "unit": "tokens"},
    "max_num_seqs": {"min": 1, "max": 4096, "unit": "sequences"},
    "max_num_batched_tokens": {"min": 256, "max": 1048576, "unit": "tokens"},
    "kv_cache_space_gib": {"min": 1, "max": 512, "unit": "GiB"},
    "temperature": {"min": 0.0, "max": 2.0},
    "top_p": {"min": 0.0, "max": 1.0},
}

DTYPE_CHOICES = ["auto", "bfloat16", "float16", "float32"]

# extraCmdArgs flag -> the name used in the API and UI.
_FLAG_TO_FIELD = {
    "--max-model-len": "max_model_len",
    "--max_model_len": "max_model_len",
    "--max-num-seqs": "max_num_seqs",
    "--max_num_seqs": "max_num_seqs",
    "--max-num-batched-tokens": "max_num_batched_tokens",
    "--max_num_batched_tokens": "max_num_batched_tokens",
    "--dtype": "dtype",
    "--block-size": "block_size",
    "--block_size": "block_size",
}

_INT_FIELDS = {"max_model_len", "max_num_seqs", "max_num_batched_tokens", "block_size", "kv_cache_space_gib"}


def parse_extra_cmd_args(args: List[Any]) -> Dict[str, Union[str, bool]]:
    """
    A flat vLLM argv list as a mapping.

    ``["--dtype", "bfloat16", "--enforce-eager"]`` becomes
    ``{"--dtype": "bfloat16", "--enforce-eager": True}``. Later occurrences win,
    matching argparse, so the mapping agrees with what vLLM would end up using.
    """
    out: Dict[str, Union[str, bool]] = {}
    tokens = [str(a) for a in args or []]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("-"):
            index += 1
            continue
        if "=" in token:
            flag, _, value = token.partition("=")
            out[flag] = value
            index += 1
            continue
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if following is not None and not following.startswith("-"):
            out[token] = following
            index += 2
        else:
            out[token] = True
            index += 1
    return out


def from_values_yaml(values_yaml: Optional[str]) -> Dict[str, Any]:
    """
    Serving defaults for a fine-tuned deployment, read from the chart's values.

    Returns the fallbacks with ``source: "fallback"`` if the values cannot be
    parsed: a dialog showing approximately-right numbers is better than one
    showing none, and the caller can say which it is looking at.
    """
    if not values_yaml:
        return {**FALLBACK_DEFAULTS, "source": "fallback"}

    try:
        parsed = yaml.safe_load(values_yaml) or {}
    except yaml.YAMLError as exc:
        logger.warning(f"Could not parse the chart values file: {exc}")
        return {**FALLBACK_DEFAULTS, "source": "fallback"}

    if not isinstance(parsed, dict):
        return {**FALLBACK_DEFAULTS, "source": "fallback"}

    block = parsed.get("defaultModelConfigs") or {}
    flags = parse_extra_cmd_args(block.get("extraCmdArgs") or [])
    env = block.get("configMapValues") or {}

    defaults: Dict[str, Any] = dict(FALLBACK_DEFAULTS)
    for flag, field in _FLAG_TO_FIELD.items():
        if flag in flags and flags[flag] is not True:
            defaults[field] = flags[flag]

    if "VLLM_CPU_KVCACHE_SPACE" in env:
        defaults["kv_cache_space_gib"] = env["VLLM_CPU_KVCACHE_SPACE"]

    for field in _INT_FIELDS:
        value = defaults.get(field)
        if value is None:
            continue
        try:
            defaults[field] = int(str(value).strip())
        except (TypeError, ValueError):
            logger.warning(f"Chart default for {field} is not a number: {value!r}")
            defaults[field] = FALLBACK_DEFAULTS.get(field)

    defaults["source"] = "chart"
    return defaults


def to_cli_args(overrides: Dict[str, Any]) -> List[str]:
    """
    Turn requested settings into vLLM flags to append to ``finetune.extraCmdArgs``.

    ``temperature`` and ``top_p`` are **not** server settings in vLLM — they are
    per-request sampling parameters. The closest thing is
    ``--override-generation-config``, which changes the *default* the server
    applies when a request omits them; any client can still send its own. Use the
    gateway if a value has to be enforced.
    """
    args: List[str] = []
    if overrides.get("max_model_len") is not None:
        args += ["--max-model-len", str(overrides["max_model_len"])]
    if overrides.get("max_num_seqs") is not None:
        args += ["--max-num-seqs", str(overrides["max_num_seqs"])]
    if overrides.get("max_num_batched_tokens") is not None:
        args += ["--max-num-batched-tokens", str(overrides["max_num_batched_tokens"])]
    if overrides.get("dtype"):
        args += ["--dtype", str(overrides["dtype"])]

    generation: Dict[str, Any] = {}
    if overrides.get("temperature") is not None:
        generation["temperature"] = overrides["temperature"]
    if overrides.get("top_p") is not None:
        generation["top_p"] = overrides["top_p"]
    if generation:
        # Passed to helm and then to the container as a single argv entry; there
        # is no shell in between (the Job runs `helm` directly), so the JSON
        # needs no quoting beyond being one string.
        import json

        args += ["--override-generation-config", json.dumps(generation, separators=(",", ":"))]
    return args

#!/usr/bin/env bash
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Entrypoint for the training pod.
#
# The pod runs the stock, digest-pinned Unsloth image; our worker code arrives as
# a read-only ConfigMap mount. This script turns that mount back into an
# importable package and hands over to the trainer.
#
# Two things it must not do:
#   * install anything. The image is pinned because its library versions are
#     validated, and `pip` on PATH here belongs to a *different* virtualenv
#     (/opt/unsloth-nb) than `python` (/opt/unsloth-venv), so a casual install
#     lands somewhere the trainer will never import from.
#   * print a line starting with [PROGRESS] or [RESULT]. Those are the engine's
#     control channel on stdout; see app/services/k8s_runtime.py.
set -euo pipefail

SRC=/etc/ft/worker-src
DST=/opt/ft
SPEC=${FT_SPEC:-/etc/ft/job/spec.json}

echo "=== finetuning trainer starting ==="
echo "job:        ${FT_JOB_ID:-unknown}"
echo "spec:       ${SPEC}"
echo "python:     $(command -v python) ($(python -c 'import sys;print(sys.version.split()[0])'))"
echo "hostname:   $(hostname)"

# --- assemble the worker package -------------------------------------------
# ConfigMap keys cannot contain '/', so paths were encoded with '__' by
# app/services/worker_bundle.py. Decode them back into a tree.
mkdir -p "$DST"
for path in "$SRC"/*; do
    key=$(basename "$path")
    [ "$key" = "entrypoint.sh" ] && continue
    rel=${key//__/\/}
    mkdir -p "$DST/$(dirname "$rel")"
    # cp, not ln: the mount is read-only and Python wants to write __pycache__
    # next to the sources.
    cp "$path" "$DST/$rel"
done

# Package markers. Shipping three empty ConfigMap keys purely to create these
# would be noise, and without them `import app.services.gpu_engine` fails.
while IFS= read -r dir; do
    [ "$dir" = "$DST" ] && continue
    [ -f "$dir/__init__.py" ] || : > "$dir/__init__.py"
done < <(find "$DST" -type d)

echo "worker:     $(find "$DST" -name '*.py' | wc -l) module(s) assembled in $DST"

# --- working directories ----------------------------------------------------
# On the shared PVC, so the HuggingFace and compile caches stay warm between
# jobs. That matters more than it looks: merging to 16-bit re-downloads the
# original base weights, and on a cold cache that download happens *after*
# training has already succeeded.
mkdir -p \
    "${TEMP_DATA_DIR:?}" \
    "${MODEL_OUTPUT_DIR:?}" \
    "${HF_HOME:?}" \
    "${TRITON_CACHE_DIR:?}" \
    "${UNSLOTH_COMPILE_LOCATION:?}"

if [ ! -r "${FT_API_KEY_FILE:-/etc/ft/secret/ft-api-key}" ]; then
    echo "ERROR: Files API key not mounted at ${FT_API_KEY_FILE:-/etc/ft/secret/ft-api-key}" >&2
    exit 78  # EX_CONFIG
fi

if [ ! -r "$SPEC" ]; then
    echo "ERROR: job spec not mounted at $SPEC" >&2
    exit 78
fi

nvidia-smi --query-gpu=index,name,memory.total,compute_cap \
           --format=csv,noheader || echo "WARNING: nvidia-smi unavailable"

# --- run --------------------------------------------------------------------
cd "$DST"
# exec so the trainer is PID 1's only child and receives SIGTERM directly when
# the Job is deleted.
exec python -m app.train_worker "$SPEC"

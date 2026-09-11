# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Training worker - the GPU-side half of the service.

One process per fine-tuning job, launched by ``trainer/entrypoint.sh`` inside a
Kubernetes Job on a GPU node:

    python -m app.train_worker /etc/ft/job/spec.json

Unlike the version this replaced, the worker owns the *whole* data path: it
downloads the dataset from the Files API, validates it, trains, merges,
watermarks, uploads the result and reports the new file id back. The API pod
never touches the bytes. That is not a stylistic choice - the API runs in a
different cluster from the GPU, so there is no shared filesystem to hand a
multi-gigabyte model across, and pushing it through the API pod would mean
sending it over the wire twice.

The consequence worth remembering: the Files API sees the *GPU cluster's* egress
address and the caller's ``ft-api-key``, not the API pod's address. An IP
allowlist in front of the Files API has to list the GPU cluster, or a job trains
for hours and then fails on the upload.

Contract with the API process (see app/services/k8s_runtime.py):
  * input    - spec.json: job parameters and control-file paths; the caller's
               Files API key from a separate mounted Secret
  * progress - lines on stdout prefixed with "[PROGRESS]"
  * output   - a "[RESULT]" line on stdout, mirrored to result.json:
               {"success": true, "output_file_id": "file-...", ...metrics} or
               {"success": false, "error": "...", "traceback": "..."}
  * cancel   - the API creates the CANCEL file; training stops at the next step
               and nothing is uploaded
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import traceback
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("uvicorn")  # same logger name as the API side

# Must match app.services.k8s_runtime._PROGRESS_MARKER / _RESULT_MARKER.
PROGRESS_MARKER = "[PROGRESS]"
RESULT_MARKER = "[RESULT]"

# Must match app.services.progress. Duplicated rather than imported: that module
# is the API's in-memory store and has no business running in the pod, while
# these five strings are part of the wire contract and change together with it.
PHASE_DOWNLOADING = "downloading_data"
PHASE_PREPARING = "preparing_environment"
PHASE_UPLOADING = "uploading_model"


def _load_spec(path: str) -> Dict:
    with open(path) as fh:
        return json.load(fh)


def _read_api_key() -> str:
    """
    The caller's Files API key, from the per-job Secret mounted by the engine.

    A file rather than an environment variable: the key belongs to the user who
    submitted the job, and an env var is visible to anything that can read
    /proc/<pid>/environ and tends to end up in crash dumps and library debug
    output.
    """
    path = os.environ.get("FT_API_KEY_FILE", "/etc/ft/secret/ft-api-key")
    with open(path) as fh:
        token = fh.read().strip()
    if not token:
        raise RuntimeError(f"Files API key at {path} is empty")
    return token


def _emit(progress: Dict) -> None:
    """Emit one progress record for the engine to relay."""
    print(f"{PROGRESS_MARKER} {json.dumps(progress, default=str)}", flush=True)


def _emit_result(path: str, payload: Dict) -> None:
    """
    Hand the result back to the API process, twice.

    The stdout marker is the primary channel: it arrives on the pod log, which
    the engine is already streaming. result.json is written atomically as a
    durable record and as the fallback the engine reads back with ``pods/exec``
    when a reconnecting log stream skipped the marker line.
    """
    print(f"{RESULT_MARKER} {json.dumps(payload, default=str)}", flush=True)

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        # Not fatal: the marker line above has already been printed, and it is
        # the channel the engine prefers.
        logger.warning(f"Could not write {path}: {exc}")


def _log_gpu_binding() -> None:
    """
    Confirm this process really owns exactly one GPU.

    Isolation comes from the NVIDIA device plugin, which sets
    CUDA_VISIBLE_DEVICES from the ``nvidia.com/gpu`` request. If it did not take
    effect the pod would see every device on the node - which on a shared GPU
    node means training next to another tenant's work. Worth failing loudly.
    """
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    logger.info(f"CUDA_VISIBLE_DEVICES={visible}")

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU visible inside the training pod")

    count = torch.cuda.device_count()
    if count != 1:
        raise RuntimeError(
            f"Expected exactly 1 visible GPU, found {count}. The device plugin "
            "did not scope this pod to its request - refusing to share a device."
        )

    props = torch.cuda.get_device_properties(0)
    logger.info(
        f"GPU: {props.name} | {props.total_memory / 1024**3:.1f} GB | "
        f"sm_{props.major}{props.minor} | bf16={torch.cuda.is_bf16_supported()}"
    )


def _reclaim(spec: Dict) -> None:
    """
    Delete the dataset and the merged model once they are safely uploaded.

    The PVC is shared by every job in the namespace and a merged 8B model is
    ~16 GB before the archive is even built, so a run that left its output behind
    would fill the volume within a handful of jobs and the failure would land on
    somebody else's job. The run directory itself stays: it holds result.json,
    which the engine may still read back.
    """
    for key in ("output_dir", "data_dir"):
        path = spec.get(key)
        if not path or not os.path.isdir(path):
            continue
        try:
            shutil.rmtree(path)
            logger.info(f"Reclaimed {path}")
        except OSError as exc:
            logger.warning(f"Could not reclaim {path}: {exc}")


class _Cancelled(Exception):
    """Raised internally when the CANCEL file appears between stages."""


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m app.train_worker <spec.json>", file=sys.stderr)
        return 2

    spec = _load_spec(argv[1])
    job_id = spec["job_id"]
    result_path = spec["result_path"]
    cancel_path = spec.get("cancel_path", "")

    # Defaults resolved by the API process from its settings, so gpu_engine never
    # has to reach for configuration of its own.
    defaults = spec.get("defaults", {})

    def progress_callback(jid: int, progress: Dict) -> None:
        _emit(progress)

    def cancellation_check(jid: int = job_id) -> bool:
        return bool(cancel_path) and os.path.exists(cancel_path)

    output_file_id: Optional[str] = None

    try:
        bearer_token = _read_api_key()
        _log_gpu_binding()

        # Unsloth must be imported before transformers/peft/trl or it cannot
        # install its patches ("Unsloth should be imported before ..."), which
        # costs speed and memory. Every module below imports transformers at
        # module level, so unsloth goes first.
        import unsloth  # noqa: F401,PLC0415 - import for side effects, needs CUDA

        from app.services import file_client  # noqa: PLC0415
        from app.services import gpu_engine  # noqa: PLC0415 - needs CUDA
        from app.services.model_watermarking import watermark_model  # noqa: PLC0415
        from app.validators.training_data_validator import (  # noqa: PLC0415
            validate_model_allowlist,
            validate_training_file,
        )

        model_name = spec["model_name"]

        # Cheapest check first: refuse an unapproved base model before spending
        # a dataset download on it. The API checks this at submit time too; here
        # it is the backstop for a policy that changed in between.
        validate_model_allowlist(model_name)

        _emit({"phase": PHASE_DOWNLOADING})
        logger.info(f"Job {job_id} - downloading dataset {spec['input_file_id']}")
        data_path = file_client.download_dataset(
            filename=spec["input_file_id"], bearer_token=bearer_token
        )

        if cancellation_check():
            raise _Cancelled()

        # Validation and model loading both happen before the trainer's first
        # step and neither has a step count, so they are one "preparing" phase as
        # far as a progress bar is concerned. The first record carrying a step
        # moves the phase on to `training` by itself.
        _emit({"phase": PHASE_PREPARING})
        logger.info(f"Job {job_id} - scanning training data for malicious content")
        validate_training_file(data_path)
        logger.info(f"Job {job_id} - training data validation passed")

        output_dir = spec["output_dir"]
        results = gpu_engine.execute_finetuning(
            model_name=model_name,
            data_path=data_path,
            output_dir=output_dir,
            params=spec.get("params") or {},
            job_id=job_id,
            progress_callback=progress_callback,
            cancellation_check=cancellation_check,
            defaults=defaults,
        )

        if cancellation_check():
            # Stopped between steps. The partial model is deliberately not
            # uploaded: the caller asked for the job to stop, and publishing a
            # half-trained model as its result would be worse than nothing.
            raise _Cancelled()

        # Watermark before the archive is built, not after the upload. The
        # previous implementation did this the other way round, so the artefact
        # the user received carried no watermark at all.
        try:
            watermark_model(
                output_dir=output_dir,
                job_id=job_id,
                username=spec.get("username", "unknown"),
                model_name=model_name,
            )
        except Exception as exc:  # provenance metadata, not the deliverable
            logger.warning(f"Job {job_id} - watermarking failed (non-fatal): {exc}")

        _emit({"phase": PHASE_UPLOADING})
        logger.info(f"Job {job_id} - uploading merged model from {output_dir}")
        output_file_id = file_client.upload_model(
            folder_path=output_dir,
            model_name=model_name,
            bearer_token=bearer_token,
        )
        logger.info(f"Job {job_id} - uploaded as {output_file_id}")

        results["output_file_id"] = output_file_id
        results["cancelled"] = False
        _reclaim(spec)
        _emit_result(result_path, results)
        logger.info(f"Job {job_id} - worker finished successfully")
        return 0

    except _Cancelled:
        logger.warning(f"Job {job_id} - cancelled; nothing was uploaded")
        _reclaim(spec)
        _emit_result(result_path, {
            "success": True,
            "cancelled": True,
            "error": "Job cancelled by user",
        })
        # Exit 0 so Kubernetes records the Job as Complete: a cancellation is an
        # expected outcome, and a Failed Job here would make the engine report
        # the run as an error instead of as cancelled.
        return 0

    except Exception as exc:
        logger.error(f"Job {job_id} - worker failed: {exc}", exc_info=True)
        _emit_result(result_path, {
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "cancelled": cancellation_check(),
            "output_file_id": output_file_id,
        })
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

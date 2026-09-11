# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import inspect
import torch
import logging
import os
import gc
import time
from typing import Dict, Optional, Callable
# unsloth and trl are imported lazily inside execute_finetuning() — they require
# CUDA and should not be loaded when the API server starts on a CPU-only host.
from transformers import TrainerCallback
from datasets import load_dataset

# NOTE: this module runs inside the training container (see app/train_worker.py),
# which has no pydantic-settings / web stack. It must therefore stay free of any
# app.config import — every tunable arrives through `params` / `defaults`.

logger = logging.getLogger("uvicorn")

# Below this much device memory, load the base weights 4-bit quantised by
# default. 4-bit trades accuracy for a ~4x smaller resident model; it is the
# right trade on a 16-24GB card and the wrong one on a 275GB one. Only a default:
# a request can set load_in_4bit either way.
_QUANTISE_BELOW_GB = 48


def _trl_version() -> tuple:
    """(major, minor) of the installed trl, or () if it cannot be determined."""
    import trl  # noqa: PLC0415 - lazy: pulls in transformers

    raw = getattr(trl, "__version__", "")
    try:
        return tuple(int(part) for part in raw.split(".")[:2])
    except ValueError:
        logger.warning(f"Could not parse trl version {raw!r}; assuming current API")
        return ()


def _adapt_formatter(formatter):
    """Present a per-example formatter under whichever convention trl expects.

    trl below 0.20 called ``formatting_func`` once per *batch* and expected a
    list of strings back; 0.20 and later call it once per example and require a
    single string. The mismatch fails in neither direction cleanly - a list
    handed to the newer trl dies inside its EOS handling with "'list' object has
    no attribute 'endswith'" once the dataset is already mapped. So the
    formatters above are written per-example, and batched back up here if the
    installed trl turns out to be an old one.
    """
    version = _trl_version()
    if not version or version >= (0, 20):
        return formatter

    def batched(examples):
        keys = list(examples.keys())
        if not keys:
            return []
        size = len(examples[keys[0]])
        return [formatter({key: examples[key][i] for key in keys}) for i in range(size)]

    logger.info(f"trl {'.'.join(map(str, version))} formats per batch; adapting formatter")
    return batched


def _build_sft_config(config_cls, wanted: Dict, sequence_length: int):
    """Instantiate trl's SFTConfig from `wanted`, tolerating API drift.

    Several settings moved between SFTTrainer's signature and SFTConfig, and
    ``max_seq_length`` was renamed ``max_length`` along the way. Passing an
    argument the installed version does not know raises; quietly dropping one it
    does know would train at the wrong sequence length or re-enable packing. So
    the wanted values are matched against the dataclass' real fields, the
    sequence-length key is resolved by name, and anything dropped is logged.
    """
    accepted = {field.name for field in dataclasses.fields(config_cls)}

    for name in ("max_length", "max_seq_length"):
        if name in accepted:
            wanted[name] = sequence_length
            break
    else:
        raise RuntimeError(
            f"{config_cls.__name__} accepts neither max_length nor max_seq_length; "
            "refusing to train at an unknown sequence length. The trl version in "
            "the training image has changed incompatibly."
        )

    dropped = sorted(set(wanted) - accepted)
    if dropped:
        logger.warning(
            f"{config_cls.__name__} does not accept {dropped}; these settings are "
            "not being applied"
        )

    return config_cls(**{key: value for key, value in wanted.items() if key in accepted})

class GPUMonitor:
    """Monitor GPU usage and memory for the device this process is bound to.

    The process is pinned to a single GPU via CUDA_VISIBLE_DEVICES by the
    dispatcher, so device 0 here is always *this job's* card. Free memory is
    read from the driver rather than derived from torch's own bookkeeping, so
    the numbers stay honest when other processes share the node.
    """

    @staticmethod
    def get_gpu_memory_info() -> Dict:
        """Get current GPU memory usage including peak stats"""
        if not torch.cuda.is_available():
            return {"available": False}

        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
        # Driver-level view: accounts for every process on the device.
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free = free_bytes / (1024**3)
        total = total_bytes / (1024**3)

        return {
            "available": True,
            "device_index": device,
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "max_allocated_gb": round(max_allocated, 2),
            "free_gb": round(free, 2),
            "total_gb": round(total, 2),
            "utilization_percent": round(((total - free) / total) * 100, 2) if total else 0.0
        }

    @staticmethod
    def clear_gpu_memory():
        """Clear GPU cache, reset peak stats, and run garbage collection"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            # Scoped to this device: sibling jobs on other cards are untouched.
            torch.cuda.reset_peak_memory_stats(device)
        gc.collect()
        logger.info("GPU memory cleared and peak stats reset")

class ProgressCallback(TrainerCallback):
    """Custom callback to track training progress and handle cancellation"""

    def __init__(self, job_id: int, update_callback: Optional[Callable] = None, cancellation_check: Optional[Callable] = None):
        self.job_id = job_id
        self.update_callback = update_callback
        self.cancellation_check = cancellation_check
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        """Check for cancellation after each training step"""
        if self.cancellation_check and self.cancellation_check(self.job_id):
            logger.warning(f"Job {self.job_id} - Cancellation requested, stopping training...")
            control.should_training_stop = True
            return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging"""
        # Check for cancellation
        if self.cancellation_check and self.cancellation_check(self.job_id):
            logger.warning(f"Job {self.job_id} - Cancellation requested during logging")
            control.should_training_stop = True
            return control

        if logs and self.update_callback:
            elapsed_time = time.time() - self.start_time
            progress = {
                "phase": "training",
                "current_step": state.global_step,
                "max_steps": state.max_steps,
                "epoch": round(state.epoch, 4) if state.epoch is not None else None,
                "loss": logs.get("loss", None),
                "learning_rate": logs.get("learning_rate", None),
                "elapsed_seconds": int(elapsed_time),
                "progress_percent": round((state.global_step / state.max_steps) * 100, 2) if state.max_steps else 0
            }
            self.update_callback(self.job_id, progress)
            logger.info(f"Job {self.job_id} - Step {state.global_step}/{state.max_steps} - Loss: {logs.get('loss', 'N/A')}")

def validate_gpu_availability() -> tuple[bool, str]:
    """Check if GPU is available and has sufficient memory"""
    if not torch.cuda.is_available():
        return False, "No GPU available"

    memory_info = GPUMonitor.get_gpu_memory_info()

    if not memory_info.get("available", False):
        return False, "GPU not available"

    # Format memory info as string
    memory_str = f"(Allocated: {memory_info['allocated_gb']}GB, Free: {memory_info['free_gb']}GB, Total: {memory_info['total_gb']}GB)"
    return True, f"GPU ready {memory_str}"

def execute_finetuning(
    model_name: str,
    data_path: str,
    output_dir: str,
    params: dict,
    job_id: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    cancellation_check: Optional[Callable] = None,
    defaults: Optional[dict] = None
) -> Dict:
    """
    Execute fine-tuning with comprehensive error handling and monitoring

    Args:
        model_name: HuggingFace model name or path
        data_path: Path to training data (JSON/JSONL)
        output_dir: Directory to save the fine-tuned model
        params: Hyperparameters dict
        job_id: Job ID for tracking
        progress_callback: Callback function for progress updates
        cancellation_check: Function to check if job should be cancelled
        defaults: Service-level defaults for any hyperparameter the request
            omitted (supplied by the API process, which owns the config)

    Returns:
        Dict with training results and metrics
    """
    start_time = time.time()
    defaults = defaults or {}

    def _param(name: str, fallback):
        """params → service defaults → hard-coded fallback."""
        value = params.get(name, None)
        if value is None:
            value = defaults.get(name, None)
        return fallback if value is None else value

    def _emit_phase(phase: str) -> None:
        """Report a stage that has no step count of its own.

        Weight merging runs for as long as a short training run and logs nothing
        the trainer callback would pick up, so without this a client sees the
        last training step until the job completes. Never fatal: losing a
        progress record must not lose a trained model.
        """
        if not (progress_callback and job_id):
            return
        try:
            progress_callback(job_id, {"phase": phase})
        except Exception as exc:  # noqa: BLE001 - cosmetic channel only
            logger.warning(f"Job {job_id} - could not report phase '{phase}': {exc}")

    try:
        # Validate GPU
        gpu_available, gpu_message = validate_gpu_availability()
        if not gpu_available:
            raise RuntimeError(gpu_message)

        logger.info(f"Starting Training Job {job_id}: {model_name}")
        logger.info(f"Data path: {data_path}")
        logger.info(f"Output dir: {output_dir}")
        logger.info(f"Parameters: {params}")

        # Clear GPU memory before starting
        GPUMonitor.clear_gpu_memory()
        initial_memory = GPUMonitor.get_gpu_memory_info()
        logger.info(f"Initial GPU memory: {initial_memory}")

        # Extract hyperparameters with defaults
        max_seq_length = _param("max_seq_length", 2048)
        batch_size = _param("batch_size", 2)
        gradient_accumulation = _param("gradient_accumulation_steps", 8)

        # Memory headroom warnings, sized against the card we actually got
        # rather than a fixed assumption about the hardware.
        total_gb = initial_memory.get("total_gb", 0)
        if total_gb and total_gb < 24 and max_seq_length > 1024:
            logger.warning(f"max_seq_length={max_seq_length} may cause OOM on a {total_gb:.0f}GB GPU. Consider reducing to 1024 or less.")
        if total_gb and total_gb < 24 and batch_size > 1:
            logger.warning(f"batch_size={batch_size} may cause OOM on a {total_gb:.0f}GB GPU. Consider using batch_size=1 with gradient_accumulation={gradient_accumulation * batch_size}.")

        # Training duration: either num_train_epochs or max_steps.
        # max_steps takes precedence when the request supplies it; otherwise it
        # stays -1 so the full epoch count is trained.
        num_train_epochs = _param("num_train_epochs", 3)
        max_steps = params.get("max_steps", None)
        if max_steps is None or int(max_steps) <= 0:
            max_steps = -1
        else:
            max_steps = int(max_steps)
            logger.info(f"max_steps={max_steps} requested; it overrides num_train_epochs={num_train_epochs}")

        learning_rate = _param("learning_rate", 1e-5)
        warmup_steps = _param("warmup_steps", 5)
        lora_r = _param("lora_r", 16)
        lora_alpha = _param("lora_alpha", 16)
        lora_dropout = _param("lora_dropout", 0.05)
        target_modules = _param("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

        # Quantisation. This used to be hard-coded to 4-bit "because of T4 GPU";
        # on a card with hundreds of gigabytes it costs accuracy for memory that
        # is not scarce, so it is now a decision. The default follows the card:
        # quantise only when the weights would not otherwise fit comfortably.
        load_in_4bit = _param("load_in_4bit", None)
        if load_in_4bit is None:
            load_in_4bit = bool(total_gb) and total_gb < _QUANTISE_BELOW_GB
            logger.info(
                f"load_in_4bit={load_in_4bit} chosen automatically for a "
                f"{total_gb:.0f}GB GPU (threshold {_QUANTISE_BELOW_GB}GB); pass "
                "load_in_4bit explicitly to override"
            )
        else:
            load_in_4bit = bool(load_in_4bit)
            logger.info(f"load_in_4bit={load_in_4bit} requested")

        # Load model with Unsloth
        from unsloth import FastLanguageModel  # noqa: PLC0415 — lazy: needs CUDA
        from trl import SFTConfig, SFTTrainer  # noqa: PLC0415 — lazy: needs CUDA
        logger.info(f"Loading {model_name} (4-bit={load_in_4bit}, max_seq_length={max_seq_length})...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )

        # Log memory usage after model load
        mem_after_load = GPUMonitor.get_gpu_memory_info()
        logger.info(f"GPU memory after model load: {mem_after_load['allocated_gb']:.2f}GB allocated, {mem_after_load['free_gb']:.2f}GB free")

        logger.info("Applying LoRA adapters with gradient checkpointing...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Critical for memory efficiency
            random_state=3407,
        )

        # Log memory usage after LoRA adapters
        mem_after_lora = GPUMonitor.get_gpu_memory_info()
        logger.info(f"GPU memory after LoRA: {mem_after_lora['allocated_gb']:.2f}GB allocated, {mem_after_lora['free_gb']:.2f}GB free")

        # Load and validate dataset
        logger.info(f"Loading dataset from {data_path}...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        dataset = load_dataset("json", data_files=data_path, split="train")  # nosec B615 - loading local file, not HF Hub dataset; revision pinning not applicable
        logger.info(f"Dataset loaded: {len(dataset)} examples")

        EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN


        # Formatting functions take *one* example and return *one* string; see
        # _adapt_formatter for why, and for what happens on an older trl.
        def formatting_prompts_func(example):
            """Render one instruction/input/output record as a single string."""
            instruction = example.get("instruction") or ""
            input_text = example.get("input") or ""
            output = example.get("output") or ""

            text = f"### Instruction:\n{instruction}\n\n"
            if input_text:
                text += f"### Input:\n{input_text}\n\n"
            text += f"### Response:\n{output}{EOS_TOKEN}"
            return text

        # Determine if we need a formatting function
        dataset_text_field = params.get("dataset_text_field", "text")
        formatting_func = None

        # Conversation datasets: 'messages' is what the dataprep service exports
        # (OpenAI style), 'conversations' is the ShareGPT spelling.
        chat_field = next((f for f in ("messages", "conversations")
                           if f in dataset.column_names), None)

        _ROLE_ALIASES = {"human": "user", "gpt": "assistant", "chatgpt": "assistant",
                         "bot": "assistant", "assistant": "assistant",
                         "user": "user", "system": "system", "tool": "tool"}

        def normalize_turns(turns):
            """One conversation as [{'role': ..., 'content': ...}, ...].

            ShareGPT names the keys from/value instead of role/content, and
            exported traces sometimes carry empty or malformed turns.
            """
            normalized = []
            for turn in turns or []:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("role") or turn.get("from") or "user").lower()
                content = turn.get("content")
                if content is None:
                    content = turn.get("value", "")
                if content:
                    normalized.append({"role": _ROLE_ALIASES.get(role, role),
                                       "content": str(content)})
            return normalized

        def formatting_chat_func(example):
            """Render one conversation with the model's own chat template.

            The template has to come from the tokenizer: its role markers and
            special tokens are what the base model was instruction-tuned on, so
            hand-rolling the format would train against the wrong boundaries.
            """
            conversation = normalize_turns(example.get(chat_field))
            if not conversation:
                return ""
            if getattr(tokenizer, "chat_template", None):
                return tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False)
            # A base model with no template: keep the roles explicit rather than
            # silently concatenating the turns.
            body = "\n".join(f"{t['role']}: {t['content']}" for t in conversation)
            return f"{body}{EOS_TOKEN}"

        def formatting_completion_func(example):
            """Render one prompt/completion record as a single string."""
            prompt = str(example.get("prompt") or "")
            completion = str(example.get("completion") or "")
            return f"{prompt}{completion}{EOS_TOKEN}"

        # Check if dataset has Alpaca format (instruction/input/output)
        if "instruction" in dataset.column_names and "output" in dataset.column_names:
            logger.info("Detected Alpaca-style dataset format (instruction/input/output)")
            formatting_func = formatting_prompts_func
            dataset_text_field = None  # Don't use text field when using formatting_func
        # prompt/completion, the layout the Fine-Tuning API's own validator
        # requires. It has to be matched BEFORE the single-text-field fallback
        # below: 'prompt' is in that fallback's candidate list, so without this
        # branch such a dataset trains on the prompts alone and every completion is
        # discarded. The job still succeeds, and the model it produces has been
        # taught to continue questions rather than answer them.
        elif ("prompt" in dataset.column_names
              and "completion" in dataset.column_names):
            # No formatting_func, deliberately. trl consumes these two columns
            # natively and masks the prompt out of the loss, so the model is graded
            # on the answer rather than on reciting the question — better than
            # anything a formatter here could do. Supplying one is also a hard
            # error: trl turns on `completion_only_loss` for this layout and
            # rejects the combination at trainer construction ("A formatting
            # function was provided while `completion_only_loss=True`"), failing
            # the job after the base model has already been downloaded.
            #
            # Older trl has no such support, and there the columns have to be
            # concatenated by hand, so pick by capability rather than assuming the
            # pinned image. Asked of SFTConfig for the same reason the
            # `processing_class` rename is: a version string does not tell us
            # which spelling this install actually has.
            native_prompt_completion = "completion_only_loss" in getattr(
                SFTConfig, "__dataclass_fields__", {})
            logger.info(
                "Detected prompt/completion dataset format"
                + ("; training on the completion only" if native_prompt_completion
                   else " (concatenated: this trl has no completion-only loss)")
            )
            formatting_func = (None if native_prompt_completion
                               else formatting_completion_func)
            dataset_text_field = None  # The columns are consumed as they are.
        elif chat_field and dataset_text_field not in dataset.column_names:
            has_template = bool(getattr(tokenizer, "chat_template", None))
            logger.info(f"Detected chat-style dataset format ('{chat_field}'); rendering with "
                        f"{'the model chat template' if has_template else 'a plain role prefix (model has no chat template)'}")
            formatting_func = formatting_chat_func
            dataset_text_field = None  # Don't use text field when using formatting_func
        elif dataset_text_field not in dataset.column_names:
            # Try to find a suitable field
            possible_fields = ["text", "prompt", "input", "content"]
            found_field = None
            for field in possible_fields:
                if field in dataset.column_names:
                    found_field = field
                    break
            if found_field:
                dataset_text_field = found_field
                logger.warning(f"Using '{dataset_text_field}' as text field")
            else:
                raise ValueError(
                    f"Could not find text field in dataset. Available fields: {dataset.column_names}. "
                    f"Supported layouts: 'messages'/'conversations' (chat), "
                    f"'instruction'+'output' (Alpaca), or a single text column "
                    f"(text/prompt/input/content)."
                )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        training_output_dir = os.path.join(output_dir, "checkpoints")

        # Setup trainer with aggressive memory optimizations
        effective_batch_size = batch_size * gradient_accumulation
        logger.info(f"Setting up trainer with effective batch size: {effective_batch_size} (batch_size={batch_size} × gradient_accumulation={gradient_accumulation})")

        # SFTConfig is a TrainingArguments subclass, and is where trl now expects
        # the sequence length, packing and dataset settings that used to be
        # SFTTrainer arguments.
        config_kwargs = {
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation,
            "warmup_steps": warmup_steps,
            "num_train_epochs": num_train_epochs if num_train_epochs is not None else 3,
            "max_steps": max_steps if max_steps is not None else -1,
            "learning_rate": learning_rate,
            "fp16": not torch.cuda.is_bf16_supported(),
            "bf16": torch.cuda.is_bf16_supported(),
            "logging_steps": 1,
            "optim": "adamw_8bit",  # 8-bit optimizer saves memory
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "seed": 3407,
            "output_dir": training_output_dir,
            "save_strategy": "epoch" if max_steps < 0 else ("steps" if max_steps > 100 else "no"),
            "save_steps": max(max_steps // 4, 1) if max_steps > 100 else 999999,
            "save_total_limit": 2,
            # Memory optimization flags
            "gradient_checkpointing": True,  # Critical for reducing memory
            "max_grad_norm": 0.3,  # Gradient clipping
            "dataloader_pin_memory": False,  # Reduce pinned memory usage
            # Do not auto-enable third-party experiment trackers. Leaving this
            # unset makes transformers instantiate every integration it can
            # import (wandb/trackio/...), which both reaches out to the network
            # and breaks on version skew inside the training image.
            "report_to": "none",
            # Moved here from SFTTrainer's signature in trl 0.20.
            "packing": False,  # Packing can increase memory usage
            "dataset_num_proc": 4,  # Use multiple processes for data preparation
        }
        # Left unset for a prompt/completion dataset, which has neither a formatter
        # nor a single text column: naming one there would point trl at a column
        # that is only half of each record.
        if not formatting_func and dataset_text_field:
            config_kwargs["dataset_text_field"] = dataset_text_field

        trainer_kwargs = {
            "model": model,
            "train_dataset": dataset,
            "args": _build_sft_config(SFTConfig, config_kwargs, max_seq_length),
            "callbacks": [ProgressCallback(job_id, progress_callback, cancellation_check)] if job_id else [],
        }

        # trl renamed `tokenizer` to `processing_class` when it generalised the
        # trainer to multimodal processors. Ask the signature rather than the
        # version: passing the wrong one is a hard TypeError at construction.
        accepts = inspect.signature(SFTTrainer.__init__).parameters
        if "processing_class" in accepts:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer

        if formatting_func:
            trainer_kwargs["formatting_func"] = _adapt_formatter(formatting_func)

        trainer = SFTTrainer(**trainer_kwargs)

        # Clear memory cache before training
        GPUMonitor.clear_gpu_memory()
        mem_before_train = GPUMonitor.get_gpu_memory_info()
        logger.info(f"GPU memory before training: {mem_before_train['allocated_gb']:.2f}GB allocated, {mem_before_train['free_gb']:.2f}GB free")

        # Start training
        logger.info("Starting training...")
        logger.info(f"Effective batch size: {batch_size * gradient_accumulation}, Max seq length: {max_seq_length}")
        train_result = trainer.train()

        # Save model
        _emit_phase("merging")
        logger.info(f"Saving model to {output_dir}...")
        # This is to save lora adapter
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # This is to save model for vLLM ready - vivek

        model.save_pretrained_merged(output_dir, tokenizer, save_method = "merged_16bit")

        # Save training arguments for reproducibility
        with open(os.path.join(output_dir, "training_args.json"), "w") as f:
            import json
            json.dump(params, f, indent=2)

        # Get final metrics including peak memory usage
        final_memory = GPUMonitor.get_gpu_memory_info()
        elapsed_time = time.time() - start_time

        logger.info(f"Peak GPU memory during training: {final_memory.get('max_allocated_gb', 'N/A')}GB")

        results = {
            "success": True,
            "model_path": output_dir,
            "training_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
            "total_steps": train_result.global_step if hasattr(train_result, 'global_step') else (max_steps if max_steps > 0 else None),
            "num_train_epochs": num_train_epochs,
            "max_steps_requested": max_steps if max_steps > 0 else None,
            "elapsed_seconds": int(elapsed_time),
            "elapsed_hours": round(elapsed_time / 3600, 2),
            "initial_memory_gb": initial_memory,
            "final_memory_gb": final_memory,
            "peak_memory_gb": final_memory.get('max_allocated_gb'),
            "dataset_size": len(dataset)
        }

        logger.info(f"Training completed successfully in {elapsed_time/60:.2f} minutes")
        logger.info(f"Results: {results}")

        # Clean up
        GPUMonitor.clear_gpu_memory()

        return results

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        GPUMonitor.clear_gpu_memory()
        raise RuntimeError(f"Training failed: {str(e)}") from e

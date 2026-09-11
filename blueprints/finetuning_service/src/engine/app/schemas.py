# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class TrainingRequest(BaseModel):
    job_id: Optional[int] = Field(None, description="Optional custom job ID. If not provided, will be auto-generated")
    username: str = Field(..., description="Username of the user submitting the job")
    model_name: str = Field(..., description="HuggingFace model name (e.g., 'unsloth/llama-3-8b-bnb-4bit')")
    input_filenames: List[str] = Field(..., description="List of file IDs from FILES API (e.g., ['file-abc123xyz'])")
    hyperparameters: Optional[Dict] = Field(
        default_factory=dict,
        description="Training hyperparameters. Use either 'num_train_epochs' (epoch-based) or 'max_steps' (step-based). If both provided, max_steps takes precedence.",
        examples=[
            {
                "num_train_epochs": 3,
                "learning_rate": 2e-4,
                "batch_size": 2,
                "lora_r": 16,
                "max_seq_length": 2048
            },
            {
                "max_steps": 100,
                "learning_rate": 2e-4,
                "batch_size": 2,
                "gradient_accumulation_steps": 4
            }
        ]
    )

class JobStatusResponse(BaseModel):
    id: int
    username: str
    status: str
    model_name: str
    progress_percent: Optional[float] = 0.0
    current_step: Optional[int] = 0
    total_steps: Optional[int] = None
    num_train_epochs: Optional[float] = None
    # Stage within a RUNNING job: downloading_data, preparing_environment,
    # training, merging, uploading_model (app/services/progress.py). Only
    # `training` has a step count behind it, so a client showing a bar should
    # derive a coarse value from this phase while the rest are current.
    current_phase: Optional[str] = None
    training_loss: Optional[float] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    elapsed_seconds: Optional[int] = None
    output_file_id: Optional[str] = None
    error_log: Optional[str] = None
    gpu_memory_used_gb: Optional[float] = None

    class Config:
        from_attributes = True

class AvailabilityResponse(BaseModel):
    available: bool
    message: str
    running_jobs: int = 0
    max_concurrent_jobs: int = 1

class GPUStatusResponse(BaseModel):
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_count: Optional[int] = 0
    total_memory_gb: Optional[float] = 0.0
    allocated_memory_gb: Optional[float] = 0.0
    free_memory_gb: Optional[float] = 0.0
    utilization_percent: Optional[float] = 0.0
    cuda_version: Optional[str] = None

class JobCancelResponse(BaseModel):
    job_id: int
    status: str
    message: str

class JobLogsResponse(BaseModel):
    job_id: int
    # The trainer's own output, as the engine recorded it while relaying the pod
    # log. Empty with found=False for a job that predates the log store or whose
    # log has aged out of it: the GPU cluster keeps no second copy to fall back on
    # once the training pod's TTL has expired.
    lines: List[str] = Field(default_factory=list)
    found: bool = True
    # True while the job is still running, so a client knows to poll again.
    live: bool = False
    truncated: bool = Field(
        False, description="True when only the tail was requested, not the whole log"
    )

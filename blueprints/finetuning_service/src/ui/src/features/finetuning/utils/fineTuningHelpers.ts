import {
  FineTuningJob,
  FineTuningJobStatus,
  FineTuningJobDisplay,
  FineTuningHyperparameters,
} from '../types';

export const transformFineTuningJobForDisplay = (job: FineTuningJob): FineTuningJobDisplay => {
  const displayName = `Fine-tune ${job.model} - ${job.id.substring(0, 8)}`;
  const displayStatus = getFineTuningStatusText(job.status);
  const displayProgress = getFineTuningProgress(job.status);
  const displayModel = job.model.split('/').pop() || job.model;
  const displayDataset = job.training_file || 'Unknown Dataset';

  return {
    ...job,
    key: job.id,
    displayName,
    displayStatus,
    displayProgress,
    displayModel,
    displayDataset,
  };
};

export const getFineTuningStatusColor = (status: FineTuningJobStatus): string => {
  switch (status) {
    case 'validating_files':
      return 'blue';
    case 'queued':
      return 'orange';
    case 'running':
      return 'processing';
    case 'succeeded':
      return 'success';
    case 'failed':
      return 'error';
    case 'cancelled':
      return 'default';
    default:
      return 'default';
  }
};

export const getFineTuningStatusText = (status: FineTuningJobStatus): string => {
  switch (status) {
    case 'validating_files':
      return 'Validating Files';
    case 'queued':
      return 'Queued';
    case 'running':
      return 'Running';
    case 'succeeded':
      return 'Succeeded';
    case 'failed':
      return 'Failed';
    case 'cancelled':
      return 'Cancelled';
    default:
      return `${status}`.charAt(0).toUpperCase() + `${status}`.slice(1);
  }
};

export const getFineTuningProgress = (status: FineTuningJobStatus): number => {
  switch (status) {
    case 'validating_files':
      return 10;
    case 'queued':
      return 20;
    case 'running':
      return 70;
    case 'succeeded':
      return 100;
    case 'failed':
    case 'cancelled':
      return 0;
    default:
      return 0;
  }
};

export const canCancelFineTuningJob = (job: FineTuningJob): boolean => {
  return job.status === 'validating_files' || job.status === 'queued' || job.status === 'running';
};

export const isFineTuningJobCompleted = (job: FineTuningJob): boolean => {
  return job.status === 'succeeded' || job.status === 'failed' || job.status === 'cancelled';
};

export const isFineTuningJobRunning = (job: FineTuningJob): boolean => {
  return job.status === 'running';
};

export const formatHyperparameters = (
  hyperparameters?: FineTuningHyperparameters | null
): string => {
  if (!hyperparameters) return 'Default';

  const params = [] as string[];
  if (hyperparameters.n_epochs) params.push(`Epochs: ${hyperparameters.n_epochs}`);
  if (hyperparameters.batch_size) params.push(`Batch: ${hyperparameters.batch_size}`);
  if (hyperparameters.learning_rate_multiplier) params.push(`LR: ${hyperparameters.learning_rate_multiplier}`);

  return params.length > 0 ? params.join(', ') : 'Default';
};

export const formatCreatedAt = (timestamp: number): string => {
  return new Date(timestamp * 1000).toLocaleString();
};

// A fine-tuned model is served under this name and shows up under it in the
// GenAI Gateway model list, so it has to say what the model actually is. Job
// ids and result file ids do not. The name is the base model plus a UTC
// timestamp, which keeps repeated fine-tunes of the same base model apart.
export const MODEL_NAME_PATTERN = /^[a-zA-Z0-9._-]+$/;
export const MAX_MODEL_NAME_LENGTH = 64;

const FINETUNED_MARKER = '-finetuned-';

const pad = (value: number, width: number): string => String(value).padStart(width, '0');

// Digits only — no separators, so the name stays safe as a Helm value, a vLLM
// --served-model-name and a LiteLLM model name.
export const formatModelNameTimestamp = (date: Date): string =>
  `${pad(date.getUTCFullYear(), 4)}${pad(date.getUTCMonth() + 1, 2)}${pad(date.getUTCDate(), 2)}` +
  `${pad(date.getUTCHours(), 2)}${pad(date.getUTCMinutes(), 2)}${pad(date.getUTCSeconds(), 2)}`;

export const sanitizeModelNamePart = (value: string): string =>
  (value.split('/').pop() || '')
    .replace(/[^a-zA-Z0-9._-]+/g, '-')
    .replace(/^[-._]+/, '')
    .replace(/[-._]+$/, '');

export const buildFineTunedModelName = (
  baseModel: string,
  timestamp: Date | number
): string => {
  // Job timestamps come off the API as unix seconds; the create form passes a Date.
  const date = typeof timestamp === 'number' ? new Date(timestamp * 1000) : timestamp;
  const stamp = formatModelNameTimestamp(date);
  const budget = MAX_MODEL_NAME_LENGTH - FINETUNED_MARKER.length - stamp.length;
  const base = (sanitizeModelNamePart(baseModel) || 'model')
    .slice(0, budget)
    .replace(/[-._]+$/, '');

  return `${base}${FINETUNED_MARKER}${stamp}`;
};

// The name the model is deployed and registered under: whatever was chosen when
// the job was submitted, else the same name derived from the job itself so jobs
// created before the name field existed still get a readable one.
export const getFineTunedModelName = (job: FineTuningJob): string => {
  const chosen = job.suffix?.trim();
  if (chosen && MODEL_NAME_PATTERN.test(chosen) && chosen.length <= MAX_MODEL_NAME_LENGTH) {
    return chosen;
  }
  return buildFineTunedModelName(job.model, job.created_at);
};

// Seconds the job spent waiting for a training worker: submission until the
// engine picked it up. On a busy cluster this is most of the wall-clock time, so
// folding it into "duration" makes a two-minute fine-tune look like a seven-hour
// one.
export const getJobQueueSeconds = (job: FineTuningJob): number | null => {
  if (!job.created_at) return null;
  const start = job.started_at;
  if (start) return Math.max(0, start - job.created_at);
  // Still waiting — the queue is as long as the job is old.
  if (job.status === 'queued' || job.status === 'validating_files') {
    return Math.max(0, Math.floor(Date.now() / 1000) - job.created_at);
  }
  return null;
};

// Seconds the job occupied a worker: picked up until finished. Wall clock rather
// than the engine's own elapsed_seconds, so queue wait plus this always adds up
// to the total below; elapsed_seconds (training only, no model load or upload) is
// the fallback for jobs recorded before started_at existed.
export const getJobTrainingSeconds = (job: FineTuningJob): number | null => {
  if (job.started_at) {
    const end = job.finished_at ?? (job.status === 'running' ? Math.floor(Date.now() / 1000) : null);
    if (end) return Math.max(0, end - job.started_at);
  }
  return job.elapsed_seconds ?? null;
};

// Submission to completion, including the queue wait.
export const getJobTotalSeconds = (job: FineTuningJob): number | null => {
  if (!job.created_at) return null;
  const end =
    job.finished_at ??
    (job.status === 'running' || job.status === 'queued' || job.status === 'validating_files'
      ? Math.floor(Date.now() / 1000)
      : null);
  if (!end) return null;
  return Math.max(0, end - job.created_at);
};

export const formatJobDuration = (job: FineTuningJob): string => {
  const total = getJobTotalSeconds(job);
  return total === null ? '-' : formatDuration(total);
};

export const formatDuration = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
};

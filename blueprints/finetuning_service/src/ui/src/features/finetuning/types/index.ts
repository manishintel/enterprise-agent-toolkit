import type {
  FineTuningJob,
  FineTuningJobStatus,
  FineTuningHyperparameters,
} from '@core/types';

export type {
  FineTuningJob,
  FineTuningJobStatus,
  FineTuningHyperparameters,
};

export type Hyperparameters = FineTuningHyperparameters;

export interface CreateFineTuningJobRequest {
  model: string;
  training_file: string;
  validation_file?: string | null;
  hyperparameters?: Hyperparameters | null;
  suffix?: string | null;
  resource_type?: string | null;
}

// Metrics the API attaches to an event. Anything not listed still comes through,
// so an engine that starts reporting more does not need a UI change to show it.
export interface FineTuningJobEventData {
  progress_percent?: number;
  current_step?: number;
  total_steps?: number;
  training_loss?: number;
  current_phase?: string;
  elapsed_seconds?: number;
  queued_seconds?: number;
  output_file_id?: string;
  model?: string;
  [key: string]: unknown;
}

export interface FineTuningJobEvent {
  id: string;
  object: string;
  created_at: number;
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
  data?: FineTuningJobEventData | null;
  type?: string;
}

export interface ListFineTuningJobsResponse {
  object: string;
  data: FineTuningJob[];
  has_more: boolean;
}

export interface ListJobEventsResponse {
  object: string;
  data: FineTuningJobEvent[];
}

export interface FineTuningApiResponse<T = unknown> {
  data: T;
  message?: string;
  success: boolean;
}

export interface FineTuningJobDisplay extends FineTuningJob {
  key?: string;
  displayName: string;
  displayStatus: string;
  displayProgress: number;
  displayModel: string;
  displayDataset: string;
}

// Serving a fine-tuned model. Mirrors ModelDeploymentStatus in the API's
// app/schemas.py.
export type DeploymentStepStatus = 'pending' | 'active' | 'done' | 'error';

export type DeploymentPhase =
  | 'not_deployed'
  | 'installing'
  | 'downloading'
  | 'extracting'
  | 'loading'
  | 'ready'
  | 'failed'
  | 'uninstalling'
  | 'unavailable';

export interface DeploymentStep {
  key: string;
  title: string;
  status: DeploymentStepStatus;
  detail?: string | null;
}

export interface ModelDeploymentStatus {
  job_id: string;
  release_name: string;
  served_model_name: string;
  namespace: string;
  phase: DeploymentPhase;
  message: string;
  progress: number;
  steps: DeploymentStep[];
  logs: string[];
  log_source?: string | null;
  gateway_registered: boolean;
  service_url?: string | null;
  can_deploy: boolean;
  can_undeploy: boolean;
  error?: string | null;
}

export interface Model {
  id: string;
  object: string;
  created: number;
  owned_by: string;
}

export interface ListModelsResponse {
  object: string;
  data: Model[];
}

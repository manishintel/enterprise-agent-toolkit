export interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'user' | 'viewer';
  avatar?: string;
}

export type ThemeMode = 'light' | 'dark';

export interface ThemeState {
  mode: ThemeMode;
}

export interface ApiResponse<T> {
  data: T;
  message?: string;
  success: boolean;
}

export interface ApiError {
  message: string;
  code: string;
  details?: unknown;
}

export type FineTuningJobStatus =
  | 'validating_files'
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled';

export interface FineTuningHyperparameters {
  n_epochs?: number | null;
  batch_size?: number | null;
  learning_rate_multiplier?: number | null;
}

export interface FineTuningJob {
  id: string;
  object: string;
  created_at: number;
  /** When the training engine picked the job up; time before this is queue wait. */
  started_at?: number | null;
  finished_at?: number | null;
  fine_tuned_model?: string | null;
  hyperparameters: FineTuningHyperparameters;
  model: string;
  organization_id: string;
  result_files: string[];
  status: FineTuningJobStatus;
  trained_tokens?: number | null;
  training_file: string;
  validation_file?: string | null;
  user_id?: string;
  resource_type?: string;
  suffix?: string | null;
  /** Training telemetry from the engine, refreshed on every status poll. */
  progress_percent?: number | null;
  current_step?: number | null;
  total_steps?: number | null;
  current_phase?: string | null;
  training_loss?: number | null;
  /** Wall-clock seconds the engine spent training, excluding queue wait. */
  elapsed_seconds?: number | null;
  error?: {
    code: string;
    message: string;
    param?: string | null;
  } | null;
}

export interface GlobalState {
  theme: ThemeState;
}

export type GlobalAction =
  | { type: 'SET_THEME'; payload: ThemeMode }
  | { type: 'TOGGLE_THEME' };

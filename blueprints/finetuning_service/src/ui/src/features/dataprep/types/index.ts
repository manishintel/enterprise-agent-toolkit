// Data Preparation Types based on OpenAPI specification

// Data Preparation Status Enum
export enum DataPrepStatus {
  PROCESSING = 'PROCESSING',
  SUCCESS = 'SUCCESS',
  FAILURE = 'FAILURE'
}

export interface PrepareDataRequest {
  file_ids: string[];
}

export interface PrepareDataResponse {
  submitted_job_ids: string[];
}

export interface DataPrepResult {
  aggregated_file_id: string;
  total_qa_pairs: number;
  successful_files: number;
  failed_files: number;
  status: string;
  message: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: string;
  result?: DataPrepResult;
  error?: string | null;
}

export interface JobWithStatus {
  job_id: string;
  user_id: string;
  file_id: string;
  submitted_at: string;
  status: string;
  result?: DataPrepResult;
  error?: string | null;
  metadata?: Record<string, unknown>;
}

export interface JobListResponse {
  user_id: string;
  total_jobs: number;
  jobs: JobWithStatus[];
}

export interface DataPrepApiError {
  message: string;
  type: string;
  param?: string | null;
  code?: string | null;
}

export interface DataPrepApiResponse<T> {
  data?: T;
  error?: DataPrepApiError;
}

// Common job statuses
export type JobStatus =
  | DataPrepStatus.PROCESSING
  | DataPrepStatus.SUCCESS
  | DataPrepStatus.FAILURE;

// UI specific types
export interface DataPrepJob extends JobWithStatus {
  progress?: number;
  duration?: number;
  created_at?: string;
  updated_at?: string;
}

export interface DataPrepFormData {
  selectedFileIds: string[];
  description?: string;
}

// ---------------------------------------------------------------------------
// Langfuse import
// ---------------------------------------------------------------------------

export type LangfuseImportFormat = 'openai_chat' | 'raw' | 'custom';

/**
 * A Langfuse project the service holds credentials for. Trace reads are
 * project-scoped, so this is the first choice on the import page — every other
 * filter applies within the project selected here.
 */
export interface LangfuseProject {
  id: string;
  name: string;
  organization?: string | null;
  is_default: boolean;
}

export interface LangfuseProjectsResponse {
  projects: LangfuseProject[];
  default_project_id?: string | null;
}

/** Where a Langfuse score came from. ANNOTATION is a human verdict. */
export type LangfuseScoreSource = 'ANNOTATION' | 'API' | 'EVAL';

export type LangfuseScoreDataType =
  | 'NUMERIC'
  | 'BOOLEAN'
  | 'CATEGORICAL'
  | 'CORRECTION'
  | 'TEXT';

export interface LangfuseScoreCategory {
  /** null for labels seen on scores that have no score config behind them. */
  value?: number | null;
  label: string;
}

/**
 * A score (annotation) that can be filtered on. `trace_count` is distinct traces
 * carrying it, so 0 means filtering on it would yield an empty dataset.
 */
export interface LangfuseScoreOption {
  name: string;
  data_type?: LangfuseScoreDataType | null;
  min_value?: number | null;
  max_value?: number | null;
  categories: LangfuseScoreCategory[];
  description?: string | null;
  /** true when a score config defines it, false for ad-hoc score names. */
  configured: boolean;
  sources: LangfuseScoreSource[];
  trace_count: number;
}

export interface LangfuseAnnotationQueue {
  id: string;
  name: string;
  description?: string | null;
}

export interface LangfuseAnnotationsResponse {
  project_id?: string | null;
  scores: LangfuseScoreOption[];
  queues: LangfuseAnnotationQueue[];
  sources: LangfuseScoreSource[];
  operators: string[];
  queue_statuses: string[];
}

export interface LangfuseAnnotationsQuery {
  project_id?: string;
  source?: LangfuseScoreSource;
  environment?: string;
}

export interface LangfuseImportRequest {
  project_id?: string;
  from_timestamp?: string;
  to_timestamp?: string;
  name?: string;
  user_id?: string;
  session_id?: string;
  environment?: string;
  tags?: string[];
  order_by?: string;
  model?: string;
  score_name?: string;
  score_source?: LangfuseScoreSource;
  score_value?: number;
  score_operator?: string;
  score_string_value?: string;
  annotation_queue_id?: string;
  annotation_queue_status?: string;
  format: LangfuseImportFormat;
  fields?: string[];
  filename?: string;
  max_traces?: number;
  preview_limit?: number;
}

export interface LangfusePreviewResponse {
  records: Record<string, unknown>[];
  returned: number;
  scanned?: number;
  skipped?: number;
  /** Project the records came from — null when the server used its default. */
  project_id?: string | null;
}

export interface LangfuseImportResponse {
  file_id: string;
  filename: string;
  bytes: number;
  n_records: number;
  scanned?: number;
  skipped?: number;
  project_id?: string | null;
}

/** A model that is deployed on the gateway and has traces in the chosen window. */
export interface LangfuseModelOption {
  id: string;
  trace_count: number;
}

export interface LangfuseModelsResponse {
  models: LangfuseModelOption[];
  project_id?: string | null;
  /** false when the gateway was unreachable, so the list is traced models only. */
  deployed_filter_applied: boolean;
  warning?: string | null;
}

export interface LangfuseModelsQuery {
  project_id?: string;
  from_timestamp?: string;
  to_timestamp?: string;
  environment?: string;
}
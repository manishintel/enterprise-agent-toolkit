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
 * What becomes of the system turn Langfuse recorded, for format='openai_chat'.
 * 'replace' swaps per-request context for the prompt used at inference.
 */
export type LangfuseSystemMode = 'keep' | 'drop' | 'replace';

/**
 * A Langfuse project available to import from. Trace reads are project-scoped,
 * so this is the first choice on the import page — every other filter applies
 * within the project selected here.
 *
 * ``has_credentials`` is false for a project the service can see but not yet
 * read: Langfuse lists an organization's projects to an organization key, while
 * reading traces needs a project key. Such a project is still selectable when
 * the response's ``can_provision`` is set, because the server mints itself a key
 * on first use.
 */
export interface LangfuseProject {
  id: string;
  name: string;
  organization?: string | null;
  organization_id?: string | null;
  has_credentials?: boolean;
  is_default: boolean;
}

/** An organization the projects above belong to, for the import page's filter. */
export interface LangfuseOrganization {
  id?: string | null;
  name?: string | null;
  project_count: number;
}

export interface LangfuseProjectsResponse {
  projects: LangfuseProject[];
  organizations?: LangfuseOrganization[];
  /** Whether a project with no key can be read anyway (server mints one). */
  can_provision?: boolean;
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
  system_mode?: LangfuseSystemMode;
  /** Required by the server when system_mode is 'replace'. */
  system_text?: string;
  filename?: string;
  /** Bounds preview and import alike; there is no separate preview cap. */
  max_traces?: number;
}

export interface LangfusePreviewResponse {
  records: Record<string, unknown>[];
  returned: number;
  scanned?: number;
  skipped?: number;
  /** The max_traces in force, whether the request set it or the server did. */
  cap?: number;
  /** The scan stopped on the cap, so there are probably more traces to import. */
  capped?: boolean;
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
  cap?: number;
  /** The scan stopped on max_traces, so the dataset written is a truncation. */
  capped?: boolean;
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
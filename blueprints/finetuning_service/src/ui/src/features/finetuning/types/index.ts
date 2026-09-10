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

/** A CPU/memory pair, in machine units plus a form ready to print. */
export interface ResourceAmount {
  cpu_millis: number;
  memory_bytes: number;
  cpu?: string | null;
  memory?: string | null;
  pods?: number | null;
}

export interface NodeCapacity {
  name: string;
  schedulable: boolean;
  unschedulable_reason?: string | null;
  allocatable: ResourceAmount;
  committed: ResourceAmount;
  free: ResourceAmount;
}

export interface SizingRecommendation {
  cpu: string;
  memory: string;
  cpu_millis: number;
  memory_bytes: number;
  parameters_billions?: number | null;
  notes: string[];
}

/**
 * Room to serve another model.
 *
 * `basis` is always "requests": Kubernetes admits a pod by comparing its
 * requests against a node's allocatable, and this cluster has no metrics-server,
 * so `live_usage_available` is false and these are reservations rather than
 * measurements. When `available` is false the service could not read cluster
 * state at all and `message` says why -- that is not the same as "no room", and
 * the dialog must not present it as such.
 */
export interface DeploymentCapacity {
  available: boolean;
  message?: string | null;
  basis: string;
  live_usage_available: boolean;
  nodes: NodeCapacity[];
  totals?: Record<string, ResourceAmount> | null;
  largest_free?: ResourceAmount | null;
  recommended?: SizingRecommendation | null;
  fits?: boolean | null;
  shortfall?: string | null;
  serving_defaults?: ServingDefaults | null;
  serving_limits?: Record<string, ServingLimit> | null;
  dtype_choices: string[];
  request_limits?: RequestLimits | null;
  deployments_used: number;
  /** 0 when no count cap is configured, which is the default. */
  deployments_max: number;
  override_allowed: boolean;
}

/**
 * The range a CPU and memory request may take for one model.
 *
 * The floor is what the model needs, the ceiling is what one node has. Both are
 * computed by the API and enforced there too, so the form does not keep a second
 * copy of the rules -- a mismatch between the two is a deployment that the form
 * accepts and the API then rejects.
 */
export interface RequestLimits {
  cpu_min_millis: number;
  cpu_max_millis: number;
  memory_min_bytes: number;
  memory_max_bytes: number;
  /** Formatted for display, e.g. "3" and "48Gi". */
  cpu_min: string;
  memory_min: string;
  parameters_billions?: number | null;
  /** "model-derived", or "installation-default" when the size could not be read. */
  basis: string;
  /** Human-readable arithmetic behind each floor, shown next to the field. */
  cpu_formula: string;
  memory_formula: string;
  ceiling_node?: string | null;
  /** True when the model's minimum exceeds what any single node has. */
  exceeds_hardware: boolean;
}

/**
 * What the packaged chart serves with when a field is left alone, read from the
 * chart itself so this cannot drift from it. `max_model_len` is null in the
 * chart, which means vLLM uses the model's own maximum.
 */
export interface ServingDefaults {
  max_model_len: number | null;
  max_num_seqs: number | null;
  max_num_batched_tokens: number | null;
  dtype: string | null;
  block_size: number | null;
  kv_cache_space_gib: number | null;
  source: 'chart' | 'fallback';
}

export interface ServingLimit {
  min: number;
  max: number;
  unit?: string;
}

export interface ExtractUtterancesRequest {
  limit?: number;
  min_words?: number;
  max_words?: number;
  first_turn_only?: boolean;
  redact_pii?: boolean;
}

export interface ExtractedUtterance {
  text: string;
  /** How many near-duplicate phrasings in the dataset this one stands for. */
  represents: number;
}

/** Per-stage counts, so the result is checkable rather than taken on trust. */
export interface UtteranceReport {
  rows: number;
  user_turns: number;
  dropped: Record<string, number>;
  redacted: Record<string, number>;
  unique: number;
  selected: number;
  selection_basis: 'embeddings' | 'lexical';
  warnings: string[];
}

export interface ExtractUtterancesResponse {
  utterances: ExtractedUtterance[];
  report: UtteranceReport;
  training_file?: string | null;
}

export interface SemanticRouteEntry {
  model: string;
  utterances: string[];
  score_threshold?: number | null;
  description?: string | null;
  is_this_job: boolean;
}

export interface SemanticRouteTestRequest {
  query: string;
  utterances?: string[];
  score_threshold?: number;
  router_name?: string;
}

/** One router registered with the gateway, for choosing where a route goes. */
export interface RouterSummary {
  name: string;
  routes: number;
  /** What a request that names no router gets. */
  is_default: boolean;
  /** This job's model already has a route in this one. */
  has_this_model: boolean;
}

/** A model that has a route somewhere, and which router it is in. */
export interface RoutedModel {
  model: string;
  router: string;
  utterances: number;
}

/**
 * State of one router. A router serves every model routed through it, and routing
 * is opt-in by model name: callers have to address `router_name` to be routed at
 * all. Several routers can coexist, so `available_routers` is what a picker shows
 * and `router_name` is the one this status describes.
 */
export interface SemanticRouteStatus {
  available: boolean;
  message?: string | null;
  router_name: string;
  configured: boolean;
  this_model?: string | null;
  this_route?: SemanticRouteEntry | null;
  routes: SemanticRouteEntry[];
  default_model?: string | null;
  embedding_model?: string | null;
  available_embedding_models: string[];
  available_chat_models: string[];
  available_routers: RouterSummary[];
  /** Every routed model across all routers, so a list view needs one call. */
  routed_models: RoutedModel[];
  restart_required_on_apply: boolean;
  /** Set by an apply or remove that triggered a gateway restart. */
  gateway_restarting?: boolean;
}

export interface SemanticRouteRequest {
  utterances: string[];
  score_threshold?: number;
  description?: string;
  default_model?: string;
  /** Unset means the installation default; an unknown name creates a router. */
  router_name?: string;
}

/**
 * How far along a router change is. Polled after an apply: the gateway caches a
 * router in memory, so the change is only live once it has restarted *and* the
 * route reads back out of it.
 */
export interface GatewayReadiness {
  ready: boolean;
  restarting: boolean;
  replicas_ready?: number | null;
  replicas_desired?: number | null;
  gateway_responding: boolean;
  router_present: boolean;
  route_present: boolean;
  router_name?: string | null;
  message: string;
}

export interface SemanticRouteScore {
  model: string;
  score: number;
  threshold: number;
  closest_utterance?: string | null;
  closest_score?: number;
  utterances_scored?: number;
  would_match: boolean;
}

/**
 * `score` is the router's own measure: the mean similarity over the route's
 * nearest 5 utterances, which reads lower than the closest single match.
 */
export interface SemanticRouteTestResponse {
  query: string;
  matched: boolean;
  matched_model?: string | null;
  score: number;
  threshold: number;
  closest_utterance?: string | null;
  scores: SemanticRouteScore[];
}

/** Overrides for a deploy. Every field is optional. */
export interface DeployModelRequest {
  cpu?: string;
  memory?: string;
  tensor_parallel_size?: number;
  pipeline_parallel_size?: number;
  max_model_len?: number;
  max_num_seqs?: number;
  max_num_batched_tokens?: number;
  dtype?: string;
  kv_cache_space_gib?: number;
  // Sampling defaults only: vLLM has no server-side temperature, so a request
  // that sends its own wins. Enforcement belongs at the gateway.
  temperature?: number;
  top_p?: number;
  force?: boolean;
}

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
  | 'registering'
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
  /** Empty unless logs were asked for; the Logs view fetches them separately. */
  logs: string[];
  log_source?: string | null;
  gateway_registered: boolean;
  service_url?: string | null;
  can_deploy: boolean;
  can_undeploy: boolean;
  error?: string | null;
}

/** A tail of one deployment's container output, fetched when asked for. */
export interface DeploymentLogs {
  job_id: string;
  logs: string[];
  log_source?: string | null;
  phase: DeploymentPhase;
  tail: number;
  /** Health-check lines dropped, so "quiet" and "filtered" can be told apart. */
  hidden_lines: number;
  message?: string | null;
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

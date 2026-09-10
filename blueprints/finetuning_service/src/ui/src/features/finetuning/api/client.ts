import { config } from '@core/config/appConfig';
import { nextAuthTokenStorage, getNextAuthUsername } from '../../auth/api/client';
import type {
  FineTuningJob,
  ListFineTuningJobsResponse,
  ListJobEventsResponse,
  CreateFineTuningJobRequest,
  ListModelsResponse,
  ModelDeploymentStatus,
  DeploymentCapacity,
  DeployModelRequest,
  DeploymentLogs,
  ExtractUtterancesRequest,
  ExtractUtterancesResponse,
  GatewayReadiness,
  SemanticRouteRequest,
  SemanticRouteStatus,
  SemanticRouteTestRequest,
  SemanticRouteTestResponse,
} from '../types';

const API_BASE_URL = config.endpoints.fineTuning;
const API_TIMEOUT = config.endpoints.timeout;

const DEFAULT_HEADERS = {
  Accept: 'application/json',
  'Content-Type': 'application/json',
  'Cache-Control': 'no-cache',
};

/**
 * `?a=1&b=2` for the parameters that have a value, or `''` for none.
 *
 * Undefined is dropped rather than sent, so "not specified" reaches the API as an
 * absent parameter and picks up its documented default instead of arriving as the
 * string "undefined".
 */
function queryString(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== '') search.set(key, String(value));
  });
  const query = search.toString();
  return query ? `?${query}` : '';
}

export class FineTuningApiError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: unknown
  ) {
    super(message);
    this.name = 'FineTuningApiError';
  }
}

async function apiRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort(new Error(`Request timeout after ${API_TIMEOUT}ms`));
  }, API_TIMEOUT);

  const token = await nextAuthTokenStorage.get();
  const username = await getNextAuthUsername();
  const headers: HeadersInit = { ...DEFAULT_HEADERS };

  // Merge with any existing headers
  if (options.headers) {
    Object.assign(headers, options.headers);
  }

  // Add Authorization header if token exists
  if (token) {
    (headers as Record<string, string>).Authorization = `Bearer ${token}`;
  }
  if (username) {
    (headers as Record<string, string>)['X-Forwarded-User'] = username;
  }

  const fullUrl = `${API_BASE_URL}${endpoint}`;

  try {
    const response = await fetch(fullUrl, {
      ...options,
      signal: controller.signal,
      headers,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));

      // Handle nested error format: { error: { message, code, type } }
      const errorObj = (errorData as { error?: { message?: string; code?: string; type?: string } }).error;

      const errorMessage =
        errorObj?.message ||
        (errorData as { detail?: string; message?: string }).detail ||
        (errorData as { message?: string }).message ||
        `HTTP ${response.status}: ${response.statusText}`;

      const errorCode =
        errorObj?.code ||
        (errorData as { code?: string }).code ||
        'HTTP_ERROR';

      throw new FineTuningApiError(
        errorMessage,
        errorCode,
        errorData
      );
    }

    const data = await response.json();

    return data as T;
  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof FineTuningApiError) {
      throw error;
    }

    if (error instanceof Error && error.name === 'AbortError') {
      const timeoutMessage = error.cause instanceof Error
        ? error.cause.message
        : `Request timeout after ${API_TIMEOUT}ms`;
      throw new FineTuningApiError(timeoutMessage, 'TIMEOUT', error);
    }

    throw new FineTuningApiError(
      error instanceof Error ? error.message : 'Unknown error occurred',
      'NETWORK_ERROR',
      error
    );
  }
}

export const fineTuningApi = {
  async createFineTuningJob(request: CreateFineTuningJobRequest): Promise<FineTuningJob> {
    return apiRequest<FineTuningJob>('/v1/fine_tuning/jobs', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  async listFineTuningJobs(params?: { limit?: number; after?: string }): Promise<ListFineTuningJobsResponse> {
    const queryParams = new URLSearchParams();

    if (params?.limit) {
      queryParams.append('limit', params.limit.toString());
    }

    if (params?.after) {
      queryParams.append('after', params.after);
    }

    const endpoint = queryParams.toString()
      ? `/v1/fine_tuning/jobs?${queryParams.toString()}`
      : '/v1/fine_tuning/jobs';

    return apiRequest<ListFineTuningJobsResponse>(endpoint);
  },

  async getFineTuningJob(jobId: string): Promise<FineTuningJob> {
    return apiRequest<FineTuningJob>(`/v1/fine_tuning/jobs/${jobId}`);
  },

  async cancelFineTuningJob(jobId: string): Promise<FineTuningJob> {
    return apiRequest<FineTuningJob>(`/v1/fine_tuning/jobs/${jobId}/cancel`, {
      method: 'POST',
    });
  },

  async listJobEvents(jobId: string, params?: { limit?: number }): Promise<ListJobEventsResponse> {
    const queryParams = new URLSearchParams();

    if (params?.limit) {
      queryParams.append('limit', params.limit.toString());
    }

    const endpoint = queryParams.toString()
      ? `/v1/fine_tuning/jobs/${jobId}/events?${queryParams.toString()}`
      : `/v1/fine_tuning/jobs/${jobId}/events`;

    return apiRequest<ListJobEventsResponse>(endpoint);
  },

  async listModels(): Promise<ListModelsResponse> {
    return apiRequest<ListModelsResponse>('/v1/models');
  },

  // Serving a fine-tuned model: the API runs the same Helm install the job
  // detail page prints, and reports its progress from cluster state.
  async deployModel(jobId: string, overrides?: DeployModelRequest): Promise<ModelDeploymentStatus> {
    return apiRequest<ModelDeploymentStatus>(`/v1/fine_tuning/jobs/${jobId}/deploy`, {
      method: 'POST',
      // An absent body deploys with the sizing derived from the base model,
      // which is what this button did before any of it was configurable.
      body: overrides ? JSON.stringify(overrides) : undefined,
    });
  },

  async getDeploymentCapacity(jobId: string): Promise<DeploymentCapacity> {
    return apiRequest<DeploymentCapacity>(`/v1/fine_tuning/jobs/${jobId}/deployment-capacity`);
  },

  async extractUtterances(
    jobId: string,
    options?: ExtractUtterancesRequest
  ): Promise<ExtractUtterancesResponse> {
    return apiRequest<ExtractUtterancesResponse>(`/v1/fine_tuning/jobs/${jobId}/utterances`, {
      method: 'POST',
      body: JSON.stringify(options ?? {}),
    });
  },

  /** One router's state. Unnamed reads the installation default. */
  async getSemanticRoute(jobId: string, routerName?: string): Promise<SemanticRouteStatus> {
    return apiRequest<SemanticRouteStatus>(
      `/v1/fine_tuning/jobs/${jobId}/semantic-route${queryString({ router_name: routerName })}`
    );
  },

  async applySemanticRoute(
    jobId: string,
    body: SemanticRouteRequest
  ): Promise<SemanticRouteStatus> {
    return apiRequest<SemanticRouteStatus>(`/v1/fine_tuning/jobs/${jobId}/semantic-route`, {
      method: 'PUT',
      body: JSON.stringify(body),
    });
  },

  /** Named, removes the route from that router only; unnamed, from every router. */
  async removeSemanticRoute(jobId: string, routerName?: string): Promise<SemanticRouteStatus> {
    return apiRequest<SemanticRouteStatus>(
      `/v1/fine_tuning/jobs/${jobId}/semantic-route${queryString({ router_name: routerName })}`,
      { method: 'DELETE' }
    );
  },

  async testSemanticRoute(
    jobId: string,
    body: SemanticRouteTestRequest
  ): Promise<SemanticRouteTestResponse> {
    return apiRequest<SemanticRouteTestResponse>(
      `/v1/fine_tuning/jobs/${jobId}/semantic-route/test`,
      { method: 'POST', body: JSON.stringify(body) }
    );
  },

  /**
   * Whether a router change is live yet. `expectRoute` is false after a removal,
   * where the change has landed once the route is *gone*.
   */
  async getRouteReadiness(
    jobId: string,
    routerName?: string,
    expectRoute = true
  ): Promise<GatewayReadiness> {
    return apiRequest<GatewayReadiness>(
      `/v1/fine_tuning/jobs/${jobId}/semantic-route/readiness${queryString({
        router_name: routerName,
        expect_route: expectRoute ? undefined : 'false',
      })}`
    );
  },

  async getModelDeployment(jobId: string): Promise<ModelDeploymentStatus> {
    return apiRequest<ModelDeploymentStatus>(`/v1/fine_tuning/jobs/${jobId}/deployment`);
  },

  /**
   * A tail of the deployment's container output. Separate from the status call so
   * logs are read when someone is looking at them, not on every status poll.
   */
  async getDeploymentLogs(
    jobId: string,
    options?: { tail?: number; hideProbes?: boolean }
  ): Promise<DeploymentLogs> {
    return apiRequest<DeploymentLogs>(
      `/v1/fine_tuning/jobs/${jobId}/deployment/logs${queryString({
        tail: options?.tail,
        hide_probes: options?.hideProbes === false ? 'false' : undefined,
      })}`
    );
  },

  async undeployModel(jobId: string): Promise<ModelDeploymentStatus> {
    return apiRequest<ModelDeploymentStatus>(`/v1/fine_tuning/jobs/${jobId}/deployment`, {
      method: 'DELETE',
    });
  },
};

export default fineTuningApi;

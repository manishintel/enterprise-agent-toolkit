import {
  useMutation,
  useQuery,
  useQueryClient,
  UseMutationOptions,
  UseQueryOptions,
} from '@tanstack/react-query';
import { fineTuningApi, FineTuningApiError } from '../api/client';
import type {
  ExtractUtterancesRequest,
  ExtractUtterancesResponse,
  GatewayReadiness,
  SemanticRouteRequest,
  SemanticRouteStatus,
  SemanticRouteTestResponse,
} from '../types';
import { queryKeys, handleQueryError } from '@core/query/queryClient';

/**
 * State of one router, and this job's route within it.
 *
 * Only fetched when something on screen needs it: it reads the gateway's model
 * registry, which is not worth doing for every visitor to a job page. Every
 * registered router comes back in `available_routers` whichever one is asked for,
 * so a picker can be populated from this alone.
 */
export function useSemanticRoute(
  jobId: string,
  enabled: boolean,
  routerName?: string,
  options?: Omit<UseQueryOptions<SemanticRouteStatus, FineTuningApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.fineTuning.jobs.semanticRoute(jobId, routerName),
    queryFn: () => fineTuningApi.getSemanticRoute(jobId, routerName),
    enabled: !!jobId && enabled,
    staleTime: 10 * 1000,
    ...options,
  });
}

/**
 * Whether a router change has actually landed.
 *
 * Applying restarts the gateway, which takes tens of seconds, and the old
 * behaviour was to say nothing for the whole of it. This polls until the rollout
 * has finished *and* the route reads back out of the gateway, so the UI can show
 * real progress and then confirm the route is live.
 *
 * Polling stops on its own once ready: there is nothing further to learn, and a
 * tab left open should not keep asking.
 */
export function useRouteReadiness(
  jobId: string,
  enabled: boolean,
  routerName?: string,
  expectRoute = true,
  options?: Omit<UseQueryOptions<GatewayReadiness, FineTuningApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.fineTuning.jobs.routeReadiness(jobId, routerName, expectRoute),
    queryFn: () => fineTuningApi.getRouteReadiness(jobId, routerName, expectRoute),
    enabled: !!jobId && enabled,
    // A restart is tens of seconds, so this is frequent enough to feel live and
    // slow enough to stay well inside the endpoint's allowance.
    refetchInterval: (query) => (query.state.data?.ready ? false : 2500),
    // The gateway refusing connections mid-restart is expected, not a failure to
    // report: keep polling instead of surfacing an error and giving up.
    retry: false,
    gcTime: 0,
    ...options,
  });
}

/**
 * Mine utterances from the training dataset.
 *
 * A mutation rather than a query because it is an action the user takes with
 * knobs, and it is deliberately not cached: re-running with different settings is
 * the point. It persists nothing.
 */
export function useExtractUtterances(
  options?: UseMutationOptions<
    ExtractUtterancesResponse,
    FineTuningApiError,
    { jobId: string; options?: ExtractUtterancesRequest }
  >
) {
  return useMutation({
    mutationFn: ({ jobId, options: extractOptions }) =>
      fineTuningApi.extractUtterances(jobId, extractOptions),
    onError: (error) => {
      handleQueryError(error);
    },
    ...options,
  });
}

export function useApplySemanticRoute(
  options?: UseMutationOptions<
    SemanticRouteStatus,
    FineTuningApiError,
    { jobId: string; body: SemanticRouteRequest }
  >
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ jobId, body }) => fineTuningApi.applySemanticRoute(jobId, body),
    onSuccess: (data, variables) => {
      // The response describes what was just written; the gateway is restarting,
      // so a refetch now would race a pod that has not loaded its models yet.
      // Seeded under both the requested key and the applied router's own key: a
      // route applied to a newly named router has no cache entry yet, and the
      // panel switches to that name as soon as the apply returns.
      queryClient.setQueryData(
        queryKeys.fineTuning.jobs.semanticRoute(variables.jobId, variables.body.router_name),
        data
      );
      queryClient.setQueryData(
        queryKeys.fineTuning.jobs.semanticRoute(variables.jobId, data.router_name),
        data
      );
    },
    onError: (error) => {
      handleQueryError(error);
    },
    ...options,
  });
}

export function useRemoveSemanticRoute(
  options?: UseMutationOptions<
    SemanticRouteStatus,
    FineTuningApiError,
    { jobId: string; routerName?: string }
  >
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ jobId, routerName }) => fineTuningApi.removeSemanticRoute(jobId, routerName),
    onSuccess: (data, { jobId, routerName }) => {
      queryClient.setQueryData(queryKeys.fineTuning.jobs.semanticRoute(jobId, routerName), data);
    },
    onError: (error) => {
      handleQueryError(error);
    },
    ...options,
  });
}

/**
 * Score a query without sending it anywhere.
 *
 * The score returned is the router's own measure — the mean over the route's
 * nearest few utterances — so what this reports and what the router decides
 * agree. Scoring on the single best match instead reads much higher and would
 * promise matches the router refuses.
 */
export function useTestSemanticRoute(
  options?: UseMutationOptions<
    SemanticRouteTestResponse,
    FineTuningApiError,
    {
      jobId: string;
      query: string;
      utterances?: string[];
      score_threshold?: number;
      router_name?: string;
    }
  >
) {
  return useMutation({
    mutationFn: ({ jobId, ...body }) => fineTuningApi.testSemanticRoute(jobId, body),
    onError: (error) => {
      handleQueryError(error);
    },
    ...options,
  });
}

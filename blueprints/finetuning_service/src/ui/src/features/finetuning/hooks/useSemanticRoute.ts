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
  SemanticRouteRequest,
  SemanticRouteStatus,
  SemanticRouteTestResponse,
} from '../types';
import { queryKeys, handleQueryError } from '@core/query/queryClient';

/**
 * State of the shared router, and this job's route within it.
 *
 * Only fetched when something on screen needs it: it reads the gateway's model
 * registry, which is not worth doing for every visitor to a job page.
 */
export function useSemanticRoute(
  jobId: string,
  enabled: boolean,
  options?: Omit<UseQueryOptions<SemanticRouteStatus, FineTuningApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.fineTuning.jobs.semanticRoute(jobId),
    queryFn: () => fineTuningApi.getSemanticRoute(jobId),
    enabled: !!jobId && enabled,
    staleTime: 10 * 1000,
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
      queryClient.setQueryData(queryKeys.fineTuning.jobs.semanticRoute(variables.jobId), data);
    },
    onError: (error) => {
      handleQueryError(error);
    },
    ...options,
  });
}

export function useRemoveSemanticRoute(
  options?: UseMutationOptions<SemanticRouteStatus, FineTuningApiError, string>
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => fineTuningApi.removeSemanticRoute(jobId),
    onSuccess: (data, jobId) => {
      queryClient.setQueryData(queryKeys.fineTuning.jobs.semanticRoute(jobId), data);
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
    { jobId: string; query: string; utterances?: string[]; score_threshold?: number }
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

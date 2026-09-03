import {
  useQuery,
  useInfiniteQuery,
  UseQueryOptions,
  UseInfiniteQueryOptions,
} from '@tanstack/react-query';
import { fineTuningApi, FineTuningApiError } from '../api/client';
import type {
  FineTuningJob,
  ListFineTuningJobsResponse,
  ListJobEventsResponse,
  FineTuningJobStatus,
  ListModelsResponse,
} from '../types';
import { queryKeys } from '@core/query/queryClient';

export function useFineTuningJobsList(
  params?: { limit?: number; after?: string },
  options?: Omit<UseQueryOptions<ListFineTuningJobsResponse, FineTuningApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.fineTuning.jobs.list(params),
    queryFn: () => fineTuningApi.listFineTuningJobs(params),
    // Short, because the API reconciles active jobs against the engine in the
    // background: the row this reads is refreshed whether or not anyone has the
    // job's own page open.
    staleTime: 5 * 1000,
    refetchInterval: (query) => {
      const hasActiveJobs = query.state.data?.data?.some((job: FineTuningJob) =>
        ['validating_files', 'queued', 'running'].includes(job.status)
      );
      return hasActiveJobs ? 10 * 1000 : false;
    },
    ...options,
  });
}

export function useFineTuningJobsInfinite(
  params?: { limit?: number },
  options?: Omit<
    UseInfiniteQueryOptions<
      ListFineTuningJobsResponse,
      FineTuningApiError
    >,
    'queryKey' | 'queryFn' | 'getNextPageParam' | 'initialPageParam'
  >
) {
  return useInfiniteQuery({
    queryKey: [...queryKeys.fineTuning.jobs.lists(), 'infinite', params],
    queryFn: ({ pageParam }) =>
      fineTuningApi.listFineTuningJobs({
        ...params,
        after: pageParam as string | undefined,
      }),
    initialPageParam: undefined,
    getNextPageParam: (lastPage) => {
      return lastPage.has_more && lastPage.data.length > 0
        ? lastPage.data[lastPage.data.length - 1].id
        : undefined;
    },
    staleTime: 30 * 1000,
    ...options,
  });
}

export function useFineTuningJob(
  jobId: string,
  options?: Omit<UseQueryOptions<FineTuningJob, FineTuningApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.fineTuning.jobs.detail(jobId),
    queryFn: () => fineTuningApi.getFineTuningJob(jobId),
    enabled: !!jobId,
    staleTime: 5 * 1000,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return false;
      const activeStatuses: FineTuningJobStatus[] = ['validating_files', 'queued', 'running'];
      // 8s: the engine reports per-step progress from an in-memory store, so a
      // poll is cheap and a short job would otherwise finish in a few samples.
      return activeStatuses.includes(data.status) ? 8 * 1000 : false;
    },
    ...options,
  });
}

export function useJobEvents(
  jobId: string,
  params?: { limit?: number },
  options?: Omit<UseQueryOptions<ListJobEventsResponse, FineTuningApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.fineTuning.jobs.events(jobId, params),
    queryFn: () => fineTuningApi.listJobEvents(jobId, params),
    enabled: !!jobId,
    staleTime: 5 * 1000,
    ...options,
  });
}

export function useModels(
  options?: Omit<UseQueryOptions<ListModelsResponse, FineTuningApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.fineTuning.models(),
    queryFn: () => fineTuningApi.listModels(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
}

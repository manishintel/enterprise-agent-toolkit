import {
  useMutation,
  useQuery,
  useQueryClient,
  UseMutationOptions,
  UseQueryOptions,
} from '@tanstack/react-query';
import { fineTuningApi, FineTuningApiError } from '../api/client';
import type { DeploymentPhase, ModelDeploymentStatus } from '../types';
import { queryKeys, handleQueryError } from '@core/query/queryClient';

// Phases where something is still happening in the cluster, so the status is
// worth polling. Everything else is a resting state.
const ACTIVE_PHASES: DeploymentPhase[] = [
  'installing',
  'downloading',
  'extracting',
  'loading',
  'uninstalling',
];

export const isDeploymentInProgress = (phase?: DeploymentPhase): boolean =>
  !!phase && ACTIVE_PHASES.includes(phase);

export function useModelDeployment(
  jobId: string,
  options?: Omit<UseQueryOptions<ModelDeploymentStatus, FineTuningApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.fineTuning.jobs.deployment(jobId),
    queryFn: () => fineTuningApi.getModelDeployment(jobId),
    enabled: !!jobId,
    staleTime: 2 * 1000,
    // Bringing a model up takes minutes (download, unpack, load weights), so
    // follow it while it is in flight and stop once it settles.
    refetchInterval: (query) => (isDeploymentInProgress(query.state.data?.phase) ? 5 * 1000 : false),
    ...options,
  });
}

export function useDeployModel(
  options?: UseMutationOptions<ModelDeploymentStatus, FineTuningApiError, string>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) => fineTuningApi.deployModel(jobId),
    onSuccess: (data) => {
      // The response is the status right after the deployment was started, so
      // seeding it starts the polling above without waiting for a refetch.
      queryClient.setQueryData(queryKeys.fineTuning.jobs.deployment(data.job_id), data);
    },
    onError: (error) => {
      handleQueryError(error);
    },
    ...options,
  });
}

export function useUndeployModel(
  options?: UseMutationOptions<ModelDeploymentStatus, FineTuningApiError, string>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) => fineTuningApi.undeployModel(jobId),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.fineTuning.jobs.deployment(data.job_id), data);
    },
    onError: (error) => {
      handleQueryError(error);
    },
    ...options,
  });
}
